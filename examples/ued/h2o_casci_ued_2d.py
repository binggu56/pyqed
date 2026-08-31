#!/usr/bin/env python3
"""Minimal ab initio H2O CASCI -> LDR -> aligned 2D UED example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.ldr.curvilinear_2d import LDR2_Curvilinear
from pyqed.qchem import CASCI, Molecule
from pyqed.ued.ued import UED
from pyqed.units import atomic_mass, au2fs


H2O_THETA_DEG = 104.52
H2O_OH_BOHR = 1.80965
H2O_HOH_BOHR = (
    "H 1.809650000000 0.000000000000 0.000000000000; "
    "O 0.000000000000 0.000000000000 0.000000000000; "
    "H -0.452035934426 1.752274839706 0.000000000000"
)


def symbols_from_atom(atom):
    rows = atom if isinstance(atom, (list, tuple)) else atom.replace(";", "\n").splitlines()
    return tuple(row[0] if isinstance(row, (list, tuple)) else row.split()[0] for row in rows if row)


def triatomic_xyz(r1, r2, theta):
    return np.array(
        [[r1, 0.0, 0.0], [0.0, 0.0, 0.0], [r2 * np.cos(theta), r2 * np.sin(theta), 0.0]],
        dtype=float,
    )


def atom_string(symbols, coords):
    return "; ".join(
        f"{sym} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"
        for sym, xyz in zip(symbols, coords, strict=True)
    )


def casci_point(mol0, symbols, coords, ncas, nelecas, nstates=1, method="direct_ci"):
    mol = Molecule(
        atom=atom_string(symbols, coords),
        basis=mol0.basis,
        charge=mol0.charge,
        spin=mol0.spin,
        unit="bohr",
    )
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=100)
    mc = CASCI(mf, ncas, nelecas, verbose=0).run(nstates=nstates, method=method)
    return mc, ao_density_matrices(mf, mc, nstates)


def ao_density_matrices(mf, mc, nstates):
    nao = mc.make_rdm1(0, with_core=True, representation="ao").shape[0]
    dm1 = np.empty((nstates, nao, nao), dtype=complex)
    tdm1 = np.empty((nstates, nstates, nao, nao), dtype=complex)

    for i in range(nstates):
        dm1[i] = mc.make_rdm1(
            i,
            with_core=True,
            representation="ao",
        )
    for i in range(nstates):
        for j in range(nstates):
            tdm1[i, j] = mc.make_tdm1(
                i,
                ket_id=j,
                with_core=True,
                representation="ao",
            )
    return dm1, tdm1


def casci_overlap(cas_grid):
    shape = cas_grid.shape
    flat = cas_grid.ravel()
    ngrid, nstates = len(flat), np.asarray(flat[0].e_tot).size
    overlap = np.empty((ngrid, nstates, ngrid, nstates), dtype=float)
    for i in range(ngrid):
        for j in range(ngrid):
            overlap[i, :, j, :] = np.real_if_close(flat[i].overlap(flat[j]))
    return overlap.reshape(*shape, nstates, *shape, nstates)


def nuclear_ground_packet(dvr, pes, overlap):
    kinetic = dvr.buildK()
    h = kinetic * overlap.reshape(kinetic.shape) + np.diag(pes.ravel())
    h = 0.5 * (h + h.conj().T)
    evals, evecs = np.linalg.eigh(h)
    psi = evecs[:, 0].reshape(len(dvr.x[0]), len(dvr.x[1])).astype(complex)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dvr.dv)
    return psi, float(evals[0])


def vibronic_hamiltonian(dvr, pes, overlap_matrix):
    ngrid = int(np.prod(dvr.nx))
    nstates = int(dvr.nstates)
    h = np.zeros((ngrid, nstates, ngrid, nstates), dtype=complex)
    idx = np.arange(ngrid)

    for state in range(nstates):
        h[idx, state, idx, state] = pes.reshape(ngrid, nstates)[:, state]
    h += np.einsum(
        "mn,manb->manb",
        dvr.buildK(),
        overlap_matrix.reshape(ngrid, nstates, ngrid, nstates),
    )
    h = h.reshape(ngrid * nstates, ngrid * nstates)
    return 0.5 * (h + h.conj().T)


def propagate_exact(dvr, pes, overlap_matrix, psi0, dt, nt, nout):
    times = float(dt) * int(nout) * np.arange(int(nt) // int(nout) + 1)
    evals, evecs = np.linalg.eigh(vibronic_hamiltonian(dvr, pes, overlap_matrix))
    coeff0 = evecs.conj().T @ psi0.ravel()
    psilist = [
        (evecs @ (np.exp(-1j * evals * t) * coeff0)).reshape(*dvr.nx, dvr.nstates)
        for t in times
    ]
    return {"times": times, "psilist": psilist}


def run_ldr(
    mol,
    ranges,
    npts,
    theta,
    ncas,
    nelecas,
    nt=None,
    dt=None,
    nout=1,
    nstates=1,
    state=0,
    packet_state=None,
):
    if not 0 <= state < nstates:
        raise ValueError(f"state={state} needs 0 <= state < nstates={nstates}.")
    packet_state = state if packet_state is None else int(packet_state)
    if not 0 <= packet_state < nstates:
        raise ValueError(
            f"packet_state={packet_state} needs 0 <= packet_state < nstates={nstates}."
        )

    symbols = symbols_from_atom(mol.atom)
    masses = [atomic_mass[s.upper()] for s in symbols]
    dvr = LDR2_Curvilinear(masses, theta=theta, nstates=nstates)
    dvr.set_dvr(ranges, [npts, npts], dvr_type="sine")

    shape = tuple(map(len, dvr.x))
    pes = np.empty((*shape, nstates))
    coords = np.empty((*shape, len(symbols), 3))
    dm1 = np.empty(shape, dtype=object)
    tdm1 = np.empty(shape, dtype=object)
    cas = np.empty(shape, dtype=object)

    for count, idx in enumerate(np.ndindex(*shape), start=1):
        xyz = triatomic_xyz(dvr.x[0][idx[0]], dvr.x[1][idx[1]], theta)
        cas[idx], (dm1[idx], tdm1[idx]) = casci_point(mol, symbols, xyz, ncas, nelecas, nstates)
        pes[idx] = np.asarray(cas[idx].e_tot).ravel()[:nstates]
        coords[idx] = xyz
        print(f"[CASCI] {count:3d}/{np.prod(shape)} E(S{state})={pes[idx][state]:.8f}")

    overlap_matrix = casci_overlap(cas)
    psi_nuc, vib_energy = nuclear_ground_packet(
        dvr,
        pes[..., packet_state],
        overlap_matrix[:, :, packet_state, :, :, packet_state],
    )
    psi = np.zeros((*shape, nstates), dtype=complex)
    psi[..., state] = psi_nuc
    if nt is None or int(nt) <= 0:
        result = {"times": np.array([0.0]), "psilist": [psi.copy()]}
    else:
        result = propagate_exact(
            dvr,
            pes,
            overlap_matrix,
            psi,
            dt=1.0 if dt is None else float(dt),
            nt=int(nt),
            nout=int(nout),
        )

    return SimpleNamespace(
        x=dvr.x,
        dv=dvr.dv,
        theta=theta,
        symbols=symbols,
        nstates=nstates,
        state=state,
        packet_state=packet_state,
        apes=pes,
        overlap_matrix=overlap_matrix,
        overlap=overlap_matrix[:, :, state, :, :, state],
        psi=psi,
        vib_energy=vib_energy,
        ued_result=result,
        ed={
            "coords": coords,
            "symbols": symbols,
            "basis": mol.basis,
            "charge": mol.charge,
            "spin": mol.spin,
            "unit": "bohr",
            "dm1_ao": dm1,
            "tdm1_ao": tdm1,
            "backend": "pyscf",
        },
    )


def plot_map(path, x, y, z, title, xlabel, ylabel, label, cmap="magma"):
    fig, ax = plt.subplots(figsize=(5.2, 4.7), dpi=200)
    im = ax.imshow(z.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap, aspect="equal")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_population(path, ldr):
    psis = np.asarray(ldr.ued_result["psilist"])
    times_fs = np.asarray(ldr.ued_result["times"], dtype=float) * au2fs
    populations = np.sum(np.abs(psis) ** 2, axis=(1, 2)) * float(ldr.dv)

    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=200)
    for state in range(populations.shape[1]):
        ax.plot(times_fs, populations[:, state], lw=2, label=f"S{state}")
    ax.set(xlabel="time (fs)", ylabel="population", ylim=(-0.02, 1.02))
    ax.legend(frameon=False, ncol=min(3, populations.shape[1]))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return populations


def plot_wavepacket_snapshots(path, ldr, nsnap=6):
    psis = np.asarray(ldr.ued_result["psilist"])
    times_fs = np.asarray(ldr.ued_result["times"], dtype=float) * au2fs
    r1, r2 = map(np.asarray, ldr.x)
    indices = np.unique(np.linspace(0, len(times_fs) - 1, min(nsnap, len(times_fs)), dtype=int))
    densities = np.sum(np.abs(psis[indices]) ** 2, axis=-1)
    vmax = max(float(densities.max()), 1e-30)

    ncols = 3 if len(indices) > 2 else len(indices)
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax, idx, rho in zip(axes, indices, densities, strict=False):
        im = ax.imshow(
            (rho / vmax).T,
            origin="lower",
            extent=[r1[0], r1[-1], r2[0], r2[-1]],
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        ax.set(title=f"t = {times_fs[idx]:.1f} fs", xlabel=r"$r_1$ (bohr)", ylabel=r"$r_2$ (bohr)")
    for ax in axes[len(indices):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[: len(indices)], fraction=0.035, pad=0.03).set_label(r"$|\psi|^2$ / global max")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return indices


def save_outputs(outdir, ldr, ued, signal):
    outdir.mkdir(parents=True, exist_ok=True)
    r1, r2 = map(np.asarray, ldr.x)
    sx, sy = signal["s_axes"]
    s_shape = signal["s_shape"]
    intensity = signal["I_signal"][0].reshape(s_shape)
    intensity /= intensity.max()
    populations = plot_population(outdir / "h2o_casci_populations.png", ldr)
    snapshot_indices = plot_wavepacket_snapshots(outdir / "h2o_casci_wavepacket_times.png", ldr)

    with h5py.File(outdir / "h2o_casci_ued_fts.h5", "w") as hf:
        hf["r1_grid"] = r1
        hf["r2_grid"] = r2
        hf["theta"] = ldr.theta
        hf["s"] = signal["s"]
        hf["rho_el_FT_ii"] = ued.electronic_ft_ii
        hf["rho_el_FT_ij"] = ued.electronic_fts

    np.savez_compressed(
        outdir / "h2o_casci_ued_2d.npz",
        sx=sx,
        sy=sy,
        intensity=intensity,
        times=np.asarray(ldr.ued_result["times"]),
        times_fs=np.asarray(ldr.ued_result["times"]) * au2fs,
        psilist=np.asarray(ldr.ued_result["psilist"]),
        populations=populations,
        wavepacket_snapshot_indices=snapshot_indices,
        psi=ldr.psi,
        pes=ldr.apes,
        overlap=ldr.overlap,
        vib_energy=ldr.vib_energy,
    )
    pes = ldr.apes[:, :, ldr.state]
    plot_map(outdir / "h2o_casci_pes.png", r1, r2, (pes - pes.min()) * 1000.0,
             f"H2O CASCI S{ldr.state} PES", r"$r_1$ (bohr)", r"$r_2$ (bohr)", "relative energy (mEh)", "viridis")
    rho = np.sum(np.abs(ldr.psi) ** 2, axis=-1)
    plot_map(outdir / "h2o_casci_wavepacket.png", r1, r2, rho / rho.max(),
             f"H2O S{ldr.state} wavepacket from S{ldr.packet_state}", r"$r_1$ (bohr)", r"$r_2$ (bohr)", r"$|\psi|^2$ / max", "cividis")
    plot_map(outdir / "h2o_casci_ued_2d.png", sx, sy, intensity,
             "H2O CASCI UED", r"$s_x$ ($\AA^{-1}$)", r"$s_y$ ($\AA^{-1}$)", "relative intensity")
    return intensity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--nelecas", type=int, default=4)
    parser.add_argument("--nstates", type=int, default=1)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--packet-state", type=int)
    parser.add_argument("--n-r", type=int, default=15)
    parser.add_argument("--r-min", type=float, default=1.4)
    parser.add_argument("--r-max", type=float, default=2.4)
    parser.add_argument("--theta-deg", type=float, default=H2O_THETA_DEG)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--dt-fs", type=float)
    parser.add_argument("--nout", type=int, default=1)
    parser.add_argument("--n-s", type=int, default=81)
    parser.add_argument("--s-max", type=float, default=8.0)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/h2o_casci_ued"))
    args = parser.parse_args()
    if args.dt_fs is not None:
        args.dt = args.dt_fs / au2fs

    mol = Molecule(atom=H2O_HOH_BOHR, basis=args.basis, charge=0, spin=0, unit="bohr")
    ldr = run_ldr(
        mol,
        ranges=[[args.r_min, args.r_max], [args.r_min, args.r_max]],
        npts=args.n_r,
        theta=np.deg2rad(args.theta_deg),
        ncas=args.ncas,
        nelecas=args.nelecas,
        nt=args.nt,
        dt=args.dt,
        nout=args.nout,
        nstates=args.nstates,
        state=args.state,
        packet_state=args.packet_state,
    )
    ued = UED(ldr, n_s=args.n_s, s_max=args.s_max)
    signal = ued.run(verbose=True)
    intensity = save_outputs(args.outdir, ldr, ued, signal)

    print(f"outdir={args.outdir}")
    print(f"vib_energy={ldr.vib_energy:.10f}")
    print(f"norm={signal['norms'][0]:.10f}")
    print(f"I range={intensity.min():.8e} {intensity.max():.8e}")


if __name__ == "__main__":
    main()
