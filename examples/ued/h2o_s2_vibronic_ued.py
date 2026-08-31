#!/usr/bin/env python3
"""Ab initio H2O LDR dynamics and aligned UED.

The workflow is intentionally direct:

    Molecule -> RHF/CASCI scanner -> Triatomic LDR -> UED

Electronic structure data stay with the Triatomic object after ``scan_pes``.
The UED calculation receives only the LDR result and the requested lab-frame
``s`` grid.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed import au2angstrom
from pyqed.namd.triatomic import Triatomic
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf import CASCI
from pyqed.qchem.mol import Molecule
from pyqed.ued.ued import UED
from pyqed.units import au2fs


DEFAULTS = {
    "basis": "sto-3g",
    "ncas": 4,
    "nelecas": 4,
    "nstates": 3,
    "initial_state": 2,
    "fc_surface": 0,
    "coordinates": "jacobi",
    "n_r": 11,
    "n_R": 11,
    "n_theta": 15,
    "r_min": 1.25,
    "r_max": 2.85,
    "R_min": 1.20,
    "R_max": 5.00,
    "theta_min_deg": 65.0,
    "theta_max_deg": 145.0,
    "gamma_min_deg": 0.0,
    "gamma_max_deg": 180.0,
    "n_s": 31,
    "s_max": 8.0,
    "dt_fs": 0.1,
    "tmax_fs": 10.0,
    "ued_dt_fs": 0.5,
    "overlap_method": "link-only",
    "kinetic_propagator": "dense",
    "outdir": Path("/private/tmp/h2o_s2_vibronic_ued"),
    "scf_tol": 1e-9,
    "conv_tol_dm": 1e-6,
    "max_cycle": 120,
}


def h2o_reference_geometry():
    r_oh = 1.81
    theta = np.deg2rad(104.5)
    return [
        ["H", (r_oh, 0.0, 0.0)],
        ["O", (0.0, 0.0, 0.0)],
        ["H", (r_oh * np.cos(theta), r_oh * np.sin(theta), 0.0)],
    ]


def make_driver(cfg):
    mol = Molecule(
        atom=h2o_reference_geometry(),
        basis=cfg.basis,
        charge=0,
        spin=0,
        unit="bohr",
    )
    mol.build()
    mf = RHF(mol).run(
        tol=cfg.scf_tol,
        conv_tol_dm=cfg.conv_tol_dm,
        max_cycle=cfg.max_cycle,
    )
    return CASCI(mf, ncas=cfg.ncas, nelecas=cfg.nelecas).as_scanner(
        nstates=cfg.nstates
    )


def make_ldr(cfg, driver) -> Triatomic:
    ldr = Triatomic(
        h2o_reference_geometry(),
        nstates=cfg.nstates,
        charge=0,
        spin=0,
        unit="bohr",
        driver=driver,
        coordinates=cfg.coordinates,
    )
    if ldr.coordinates == "jacobi":
        domains = [
            [cfg.r_min, cfg.r_max],
            [cfg.R_min, cfg.R_max],
            [np.deg2rad(cfg.gamma_min_deg), np.deg2rad(cfg.gamma_max_deg)],
        ]
        npts = [cfg.n_r, cfg.n_R, cfg.n_theta]
    else:
        domains = [
            [cfg.r_min, cfg.r_max],
            [cfg.r_min, cfg.r_max],
            [np.deg2rad(cfg.theta_min_deg), np.deg2rad(cfg.theta_max_deg)],
        ]
        npts = [cfg.n_r, cfg.n_r, cfg.n_theta]
    ldr.set_dvr(domains=domains, npts=npts, dvr_type=["sine", "sine", "legendre"])
    return ldr


def momentum_grid(s_max: float, n_s: int):
    axis = np.linspace(-s_max, s_max, n_s)
    sx, sy = np.meshgrid(axis, axis, indexing="ij")
    s = np.column_stack(
        [
            sx.ravel() * au2angstrom,
            sy.ravel() * au2angstrom,
            np.zeros(n_s * n_s),
        ]
    )
    return axis, axis, s


def linked_overlap_to_reference(ldr: Triatomic, idx, ref_idx):
    links = getattr(ldr, "overlap_links", None)
    if links is not None:
        return ldr._linked_overlap_between(idx, ref_idx, links, ldr.nstates)

    indices = ldr._grid_indices()
    flat = {grid_idx: i for i, grid_idx in enumerate(indices)}
    overlap = ldr.overlap_matrix.reshape(
        len(indices),
        ldr.nstates,
        len(indices),
        ldr.nstates,
    )
    return overlap[flat[idx], :, flat[ref_idx], :]


def fc_packet_on_state(ldr: Triatomic, pes: np.ndarray, cfg):
    kinetic = ldr.buildK(sparse=False)
    surface = np.asarray(pes[..., cfg.fc_surface], dtype=float)
    hmat = kinetic + np.diag(surface.ravel())
    hmat = 0.5 * (hmat + hmat.conj().T)
    energies, vectors = np.linalg.eigh(hmat)

    chi = vectors[:, 0].reshape(*ldr.nx).astype(complex)
    chi /= np.sqrt(np.sum(np.abs(chi) ** 2))
    ref_idx = np.unravel_index(int(np.argmax(np.abs(chi) ** 2)), tuple(ldr.nx))

    psi = np.zeros((*ldr.nx, ldr.nstates), dtype=complex)
    for idx in ldr._grid_indices():
        overlap = linked_overlap_to_reference(ldr, idx, ref_idx)
        phase = overlap[cfg.initial_state, cfg.initial_state]
        psi[idx + (cfg.initial_state,)] = (
            chi[idx] * phase / abs(phase) if abs(phase) > 1.0e-14 else chi[idx]
        )

    psi /= ldr.norm(psi)
    q = [ldr.x[axis][ref_idx[axis]] for axis in range(ldr.ndim)]
    labels = ldr.coordinate_labels
    print(
        "[initial packet] "
        f"S{cfg.fc_surface} ground packet projected to S{cfg.initial_state}; "
        f"E={energies[0].real:.10f} Eh; "
        f"peak {labels[0]}={q[0]:.6f}, {labels[1]}={q[1]:.6f}, "
        f"{labels[2]}={np.rad2deg(q[2]):.3f} deg"
    )
    return psi, float(energies[0].real), ref_idx


def populations(result):
    psi = np.asarray(result["psilist"])
    return np.sum(np.abs(psi) ** 2, axis=tuple(range(1, psi.ndim - 1))).real


def marginals(ldr: Triatomic, result):
    psi = np.asarray(result["psilist"])
    psi_values = psi / np.sqrt(np.asarray(ldr.grid_weights))[None, ..., None]
    theta_w = np.asarray(ldr.w[2], dtype=float)
    weights = np.asarray(ldr.grid_weights, dtype=float)

    r12_state = np.sum(
        np.abs(psi_values) ** 2 * theta_w[None, None, None, :, None],
        axis=3,
    ).real
    r12_total = np.sum(r12_state, axis=-1)
    theta = np.sum(
        np.abs(psi_values) ** 2 * weights[None, ..., None],
        axis=(1, 2, 4),
    ).real
    theta_sum = theta.sum(axis=1)
    theta_sum[theta_sum == 0.0] = 1.0
    return r12_total, r12_state, theta / theta_sum[:, None]


def frame_indices(times_fs, count=3):
    if len(times_fs) <= count:
        return np.arange(len(times_fs), dtype=int)
    return np.unique(np.linspace(0, len(times_fs) - 1, count).round().astype(int))


def normalize_panels(stack):
    stack = np.asarray(stack, dtype=float)
    scale = np.max(stack.reshape(stack.shape[0], -1), axis=1)
    scale[scale == 0.0] = 1.0
    return stack / scale[:, None, None]


def plot_overview(path, ldr, times_fs, pops, r12, sx, sy, intensity):
    frames = frame_indices(times_fs)
    r12 = normalize_panels(r12[frames])
    labels = ldr.coordinate_labels
    fig = plt.figure(figsize=(4.0 * (len(frames) + 1), 7.0), dpi=200)
    gs = fig.add_gridspec(2, len(frames) + 1)

    ax = fig.add_subplot(gs[:, 0])
    for state in range(pops.shape[1]):
        ax.plot(times_fs, pops[:, state], lw=2, label=f"S{state}")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("population")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)

    for col, frame in enumerate(frames, start=1):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            r12[col - 1].T,
            origin="lower",
            extent=[ldr.x[0][0], ldr.x[0][-1], ldr.x[1][0], ldr.x[1][-1]],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_xlabel(f"{labels[0]} (bohr)")
        ax.set_ylabel(f"{labels[1]} (bohr)")
        ax.set_title(rf"$|\chi|^2$ {times_fs[frame]:.1f} fs")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(
            intensity[frame].T,
            origin="lower",
            extent=[sx[0], sx[-1], sy[0], sy[-1]],
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        ax.set_xlabel(r"$s_x$ ($\AA^{-1}$)")
        ax.set_ylabel(r"$s_y$ ($\AA^{-1}$)")
        ax.set_title(f"UED {times_fs[frame]:.1f} fs")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_difference(path, times_fs, sx, sy, intensity):
    frames = frame_indices(times_fs)
    delta = intensity - intensity[0][None, :, :]
    vmax = float(np.max(np.abs(delta[frames]))) or 1.0
    fig, axes = plt.subplots(
        1,
        len(frames),
        figsize=(3.4 * len(frames), 3.1),
        dpi=200,
        squeeze=False,
    )
    for ax, frame in zip(axes[0], frames):
        im = ax.imshow(
            delta[frame].T,
            origin="lower",
            extent=[sx[0], sx[-1], sy[0], sy[-1]],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_xlabel(r"$s_x$ ($\AA^{-1}$)")
        ax.set_ylabel(r"$s_y$ ($\AA^{-1}$)")
        ax.set_title(rf"$\Delta I$ {times_fs[frame]:.1f} fs")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_wavepacket(path, ldr, times_fs, r12_state, pops):
    frames = frame_indices(times_fs)
    rows = [state for state in (1, 2) if state < r12_state.shape[-1]]
    labels = ldr.coordinate_labels
    fig, axes = plt.subplots(
        len(rows),
        len(frames),
        figsize=(3.0 * len(frames), 2.6 * len(rows)),
        dpi=200,
        squeeze=False,
    )
    for row, state in enumerate(rows):
        panels = normalize_panels(r12_state[frames, :, :, state])
        for col, frame in enumerate(frames):
            ax = axes[row, col]
            im = ax.imshow(
                panels[col].T,
                origin="lower",
                extent=[ldr.x[0][0], ldr.x[0][-1], ldr.x[1][0], ldr.x[1][-1]],
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                interpolation="bicubic",
                aspect="auto",
            )
            ax.set_xlabel(f"{labels[0]} (bohr)")
            ax.set_ylabel(f"{labels[1]} (bohr)")
            ax.set_title(f"S{state} {times_fs[frame]:.1f} fs, P={pops[frame, state]:.2e}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_theta(path, ldr, times_fs, pops, theta_density):
    angle_deg = np.rad2deg(np.asarray(ldr.x[2], dtype=float))
    angle_label = ldr.coordinate_labels[2]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), dpi=200)
    for state in range(pops.shape[1]):
        axes[0].plot(times_fs, pops[:, state], lw=2, label=f"S{state}")
    axes[0].set_xlabel("time (fs)")
    axes[0].set_ylabel("population")
    axes[0].legend(frameon=False)

    im = axes[1].imshow(
        theta_density.T,
        origin="lower",
        extent=[times_fs[0], times_fs[-1], angle_deg[0], angle_deg[-1]],
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_xlabel("time (fs)")
    axes[1].set_ylabel(f"{angle_label} (deg)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        kwargs = {"default": default}
        if name == "overlap_method":
            kwargs["choices"] = ("link-only", "linked", "full", "none")
        elif name == "kinetic_propagator":
            kwargs["choices"] = ("dense", "expm_multiply", "chebyshev")
        elif name == "coordinates":
            kwargs["choices"] = ("valence", "jacobi")
        elif isinstance(default, Path):
            kwargs["type"] = Path
        else:
            kwargs["type"] = type(default)
        parser.add_argument(flag, **kwargs)
    return parser.parse_args()


def run(cfg):
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    driver = make_driver(cfg)
    ldr = make_ldr(cfg, driver)
    sx, sy, s = momentum_grid(cfg.s_max, cfg.n_s)

    cwd = Path.cwd()
    try:
        os.chdir(cfg.outdir)
        pes, overlap_data, _ = ldr.scan_pes(
            nstates=cfg.nstates,
            overlap_method=cfg.overlap_method,
            driver=driver,
        )
    finally:
        os.chdir(cwd)

    psi0, fc_energy, ref_idx = fc_packet_on_state(ldr, pes, cfg)
    result = ldr.run(
        psi0,
        dt=cfg.dt_fs / au2fs,
        nt=int(round(cfg.tmax_fs / cfg.dt_fs)),
        nout=max(1, int(round(cfg.ued_dt_fs / cfg.dt_fs))),
        kinetic_propagator=cfg.kinetic_propagator,
    )

    ued = UED(ldr, aligned=True)
    signal = ued.run(s=s, verbose=True)
    times_fs = np.asarray(signal["times"], dtype=float) * au2fs
    pops = populations(result)
    r12, r12_state, theta_density = marginals(ldr, result)
    intensity = signal["I_signal"].reshape(len(times_fs), cfg.n_s, cfg.n_s)
    intensity /= float(np.max(intensity)) or 1.0

    npz_path = cfg.outdir / "h2o_s2_vibronic_ued.npz"
    np.savez_compressed(
        npz_path,
        times_fs=times_fs,
        populations=pops,
        coordinate_labels=np.array(ldr.coordinate_labels),
        coordinates=np.array(ldr.coordinates),
        q0_bohr=ldr.x[0],
        q1_bohr=ldr.x[1],
        q2_rad=ldr.x[2],
        sx_angstrom_inv=sx,
        sy_angstrom_inv=sy,
        s_bohr_inv=s,
        pes=pes,
        coords_bohr=signal["coords"],
        rho_el_FT_ii=ued.electronic_ft_ii,
        rho_el_FT_ij=ued.electronic_fts,
        nuclear_r1r2_marginal=r12,
        nuclear_r1r2_state_marginal=r12_state,
        nuclear_theta_marginal=theta_density,
        I_signal=intensity,
        I_total=signal["I_total"].reshape(len(times_fs), cfg.n_s, cfg.n_s),
        I_nuc=signal["I_nuc"].reshape(len(times_fs), cfg.n_s, cfg.n_s),
        I_el=signal["I_el"].reshape(len(times_fs), cfg.n_s, cfg.n_s),
        I_cross=signal["I_cross"].reshape(len(times_fs), cfg.n_s, cfg.n_s),
        norms=signal["norms"],
        fc_energy=np.array(fc_energy),
        initial_reference_index=np.array(ref_idx, dtype=int),
        overlap_kind=np.array(cfg.overlap_method),
        overlap_size=np.array(
            0
            if overlap_data is None
            else len(overlap_data)
            if isinstance(overlap_data, dict)
            else overlap_data.size
        ),
    )

    overview = cfg.outdir / "h2o_s2_vibronic_ued.png"
    difference = cfg.outdir / "h2o_s2_difference_ued.png"
    wavepacket = cfg.outdir / "h2o_s2_state_resolved_wavepackets.png"
    theta_png = cfg.outdir / "h2o_s2_population_theta.png"
    plot_overview(overview, ldr, times_fs, pops, r12, sx, sy, intensity)
    plot_difference(difference, times_fs, sx, sy, intensity)
    plot_wavepacket(wavepacket, ldr, times_fs, r12_state, pops)
    plot_theta(theta_png, ldr, times_fs, pops, theta_density)

    print(f"npz={npz_path}")
    print(f"png={overview}")
    print(f"diff_png={difference}")
    print(f"state_png={wavepacket}")
    print(f"theta_png={theta_png}")
    print(f"times_fs={times_fs[0]:.3f}..{times_fs[-1]:.3f} n={len(times_fs)}")
    print(f"population final={np.array2string(pops[-1], precision=8)}")
    print(f"norm range={signal['norms'].min():.10f} {signal['norms'].max():.10f}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
