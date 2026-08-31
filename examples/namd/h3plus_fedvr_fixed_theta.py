#!/usr/bin/env python3
"""Fixed-angle H3+ benchmark on FE-DVR and sine-DVR (r1, r2) grids.

The script scans a small CASCI ground-state APES for body-fixed H3+ at fixed
bond angle, builds two-dimensional stretch Hamiltonians, and prints the lowest
vibrational energies.  It is intended as a compact benchmark for using sparse
FE-DVR kinetic energy with future link-local LDR propagation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import FEDVR, SineDVR
from pyqed.qchem import CASCI, Molecule
from pyqed.units import amu2au, au2ev, au2fs

HARTREE_TO_EV = au2ev


def h3plus_body_frame(r1, r2, theta):
    return [
        ["H", (float(r1), 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (float(r2) * np.cos(theta), float(r2) * np.sin(theta), 0.0)],
    ]


def set_thread_limits(nthreads):
    if nthreads is None:
        return
    value = str(int(nthreads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def casci_energy(r1, r2, theta, basis, ncas, nelecas, nstates, verbose=0):
    mol = Molecule(
        atom=h3plus_body_frame(r1, r2, theta),
        basis=basis,
        charge=1,
        spin=0,
        unit="bohr",
    )
    mol.build()
    mf = mol.RHF(verbose=verbose).run(max_cycle=80)
    mc = CASCI(mf, ncas=ncas, nelecas=nelecas, verbose=verbose).run(nstates=nstates)
    return np.asarray(mc.e_tot[:nstates], dtype=float)


def scan_apes(r1_dvr, r2_dvr, theta, basis, ncas, nelecas, nstates, worker_threads):
    set_thread_limits(worker_threads)
    apes = np.zeros((r1_dvr.npts, r2_dvr.npts, nstates), dtype=float)
    total = r1_dvr.npts * r2_dvr.npts
    count = 0
    t0 = time.perf_counter()
    for i, r1 in enumerate(r1_dvr.x):
        for j, r2 in enumerate(r2_dvr.x):
            count += 1
            apes[i, j] = casci_energy(r1, r2, theta, basis, ncas, nelecas, nstates)
            print(
                f"[scan] {count:3d}/{total}: r1={r1:.6f} r2={r2:.6f} "
                f"E0={apes[i, j, 0]:.10f}"
            )
    print(f"[scan] completed in {time.perf_counter() - t0:.2f} s")
    return apes


def _kinetic_matrix(dvr):
    if hasattr(dvr, "kinetic_sparse"):
        return dvr.kinetic_sparse()
    return sp.csr_matrix(dvr.t())


def _momentum_matrix(dvr):
    try:
        return dvr.momentum(sparse=True)
    except TypeError:
        return sp.csr_matrix(dvr.momentum())


def _node_grid_lines(dvr):
    return getattr(dvr, "full_x", dvr.x)


def fixed_theta_stretch_kinetic(r1_dvr, r2_dvr, theta, masses_au):
    """Sparse/dense fixed-angle stretch kinetic operator for H-A-H coordinates."""
    m_end1, m_center, m_end2 = masses_au
    g11 = 1.0 / m_end1 + 1.0 / m_center
    g22 = 1.0 / m_end2 + 1.0 / m_center
    g12 = np.cos(theta) / m_center

    T1 = _kinetic_matrix(r1_dvr)
    T2 = _kinetic_matrix(r2_dvr)
    P1 = _momentum_matrix(r1_dvr)
    P2 = _momentum_matrix(r2_dvr)
    I1 = sp.eye(r1_dvr.npts, format="csr", dtype=complex)
    I2 = sp.eye(r2_dvr.npts, format="csr", dtype=complex)

    kinetic = (
        g11 * sp.kron(T1, I2, format="csr")
        + g22 * sp.kron(I1, T2, format="csr")
        + g12 * sp.kron(P1, P2, format="csr")
    )
    return 0.5 * (kinetic + kinetic.conj().T)


def solve_ground_surface(r1_dvr, r2_dvr, theta, apes, nlevels):
    proton_mass = 1.00782503223 * amu2au
    masses_au = np.array([proton_mass, proton_mass, proton_mass])
    kinetic = fixed_theta_stretch_kinetic(r1_dvr, r2_dvr, theta, masses_au)
    potential = sp.diags(apes[:, :, 0].reshape(-1), format="csr")
    hamiltonian = kinetic + potential

    if nlevels >= hamiltonian.shape[0]:
        evals = np.linalg.eigvalsh(hamiltonian.toarray())
        evals = evals[:nlevels]
    else:
        evals = sla.eigsh(hamiltonian, k=nlevels, which="SA", return_eigenvectors=False)
        evals = np.sort(evals)
    return evals, kinetic, hamiltonian


def plot_apes(r1_dvr, r2_dvr, theta, apes, outpath, title_prefix):
    values = (apes[:, :, 0] - apes[:, :, 0].min()) * HARTREE_TO_EV
    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    sc = ax.scatter(
        np.repeat(r2_dvr.x[None, :], r1_dvr.npts, axis=0).reshape(-1),
        np.repeat(r1_dvr.x[:, None], r2_dvr.npts, axis=1).reshape(-1),
        c=values.reshape(-1),
        s=95,
        cmap="magma",
        edgecolors="k",
        linewidths=0.35,
    )
    for x in _node_grid_lines(r1_dvr):
        ax.axhline(x, color="0.86", lw=0.5, zorder=0)
    for x in _node_grid_lines(r2_dvr):
        ax.axvline(x, color="0.86", lw=0.5, zorder=0)
    ax.set_title(f"H3+ {title_prefix} APES, theta={np.rad2deg(theta):.1f} deg")
    ax.set_xlabel("r2 / bohr")
    ax.set_ylabel("r1 / bohr")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("E0 - min(E0) / eV")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _grid_weights(dvr):
    if hasattr(dvr, "w"):
        return np.asarray(dvr.w, dtype=float)
    return np.ones(dvr.npts, dtype=float) * float(dvr.dx)


def initial_packet(r1_dvr, r2_dvr, center, width):
    r1 = np.asarray(r1_dvr.x, dtype=float)[:, None]
    r2 = np.asarray(r2_dvr.x, dtype=float)[None, :]
    w1 = _grid_weights(r1_dvr)[:, None]
    w2 = _grid_weights(r2_dvr)[None, :]
    r10, r20 = center
    values = np.exp(-width * ((r1 - r10) ** 2 + (r2 - r20) ** 2))
    psi = (values * np.sqrt(w1 * w2)).reshape(-1).astype(complex)
    norm = np.linalg.norm(psi)
    if norm == 0.0:
        raise ValueError("Initial packet has zero norm on this DVR grid.")
    return psi / norm


def region_masks(r1_dvr, r2_dvr):
    r1 = np.asarray(r1_dvr.x, dtype=float)[:, None]
    r2 = np.asarray(r2_dvr.x, dtype=float)[None, :]
    lower = np.broadcast_to(r1 < r2, (r1_dvr.npts, r2_dvr.npts)).reshape(-1)
    upper = np.broadcast_to(r1 > r2, (r1_dvr.npts, r2_dvr.npts)).reshape(-1)
    bridge = ~(lower | upper)
    return lower, upper, bridge


def region_population(psi, lower, upper, bridge):
    density = np.abs(psi) ** 2
    return np.array(
        [
            density[lower].sum(),
            density[upper].sum(),
            density[bridge].sum(),
            density.sum(),
        ],
        dtype=float,
    )


def project_packet(hamiltonian, psi, nlevels):
    if nlevels is None:
        return psi
    nlevels = int(nlevels)
    if nlevels <= 0:
        return psi

    dim = hamiltonian.shape[0]
    nlevels = min(nlevels, dim)
    if nlevels >= dim:
        return psi

    if dim <= 512 or nlevels > dim // 3:
        _, vecs = np.linalg.eigh(hamiltonian.toarray())
        basis = vecs[:, :nlevels]
    else:
        _, basis = sla.eigsh(hamiltonian, k=nlevels, which="SA")
    projected = basis @ (basis.conj().T @ psi)
    norm = np.linalg.norm(projected)
    if norm == 0.0:
        raise ValueError("Projected packet has zero norm.")
    return projected / norm


def spectral_weights(hamiltonian, psi, counts=(4, 8, 16)):
    dim = hamiltonian.shape[0]
    max_count = min(max(counts), dim)
    if dim <= 512 or max_count > dim // 3:
        evals, vecs = np.linalg.eigh(hamiltonian.toarray())
        evals = evals[:max_count]
        vecs = vecs[:, :max_count]
    else:
        evals, vecs = sla.eigsh(hamiltonian, k=max_count, which="SA")
        order = np.argsort(evals)
        evals = evals[order]
        vecs = vecs[:, order]
    coeff = vecs.conj().T @ psi
    probs = np.abs(coeff) ** 2
    weights = {n: float(probs[: min(n, max_count)].sum()) for n in counts}
    hpsi = hamiltonian @ psi
    eavg = float(np.vdot(psi, hpsi).real)
    return eavg, weights


def propagate_region_populations(
    hamiltonian,
    r1_dvr,
    r2_dvr,
    dt_fs,
    nt,
    nout,
    center,
    width,
    project_levels=None,
):
    psi = initial_packet(r1_dvr, r2_dvr, center, width)
    psi = project_packet(hamiltonian, psi, project_levels)
    lower, upper, bridge = region_masks(r1_dvr, r2_dvr)
    nrecords = nt // nout + 1
    times = np.zeros(nrecords, dtype=float)
    pops = np.zeros((nrecords, 4), dtype=float)
    pops[0] = region_population(psi, lower, upper, bridge)

    shift = float(np.real(hamiltonian.diagonal()).min())
    h_shifted = hamiltonian - shift * sp.eye(
        hamiltonian.shape[0],
        format="csr",
        dtype=hamiltonian.dtype,
    )
    step_op = (-1j * dt_fs / au2fs) * h_shifted

    irec = 1
    for step in range(1, nt + 1):
        psi = sla.expm_multiply(step_op, psi)
        if step % nout == 0:
            times[irec] = step * dt_fs
            pops[irec] = region_population(psi, lower, upper, bridge)
            irec += 1
    return times, pops


def plot_populations(dynamics, outpath):
    fig, (ax_pop, ax_norm) = plt.subplots(
        2,
        1,
        figsize=(6.2, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    colors = {"FE-DVR": "tab:blue", "sine DVR": "tab:orange"}
    for label, (times, pops) in dynamics.items():
        color = colors.get(label)
        ax_pop.plot(times, pops[:, 0], color=color, lw=2.0, label=f"{label}: r1 < r2")
        ax_pop.plot(
            times,
            pops[:, 1],
            color=color,
            lw=1.7,
            ls="--",
            label=f"{label}: r1 > r2",
        )
        ax_norm.plot(times, pops[:, 3], color=color, lw=1.8, label=label)

    ax_pop.set_ylabel("population")
    ax_pop.set_ylim(-0.03, 1.03)
    ax_pop.legend(frameon=False, ncol=2, fontsize=8)
    ax_norm.set_xlabel("time / fs")
    ax_norm.set_ylabel("norm")
    ax_norm.set_ylim(0.995, 1.005)
    ax_norm.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def packet_region_populations(r1_dvr, r2_dvr, center, width, hamiltonian=None, project_levels=None):
    psi = initial_packet(r1_dvr, r2_dvr, center, width)
    if hamiltonian is not None:
        psi = project_packet(hamiltonian, psi, project_levels)
    lower, upper, bridge = region_masks(r1_dvr, r2_dvr)
    return region_population(psi, lower, upper, bridge)


def run_basis(label, r1_dvr, r2_dvr, theta, args, cache, plot_path):
    if cache.exists() and not args.force:
        data = np.load(cache)
        apes = data["apes"]
        print(f"[cache] loaded {cache}")
    else:
        print(f"[scan] starting {label} grid")
        apes = scan_apes(
            r1_dvr,
            r2_dvr,
            theta,
            args.basis,
            args.ncas,
            args.nelecas,
            args.nstates,
            args.worker_threads,
        )
        np.savez(
            cache,
            apes=apes,
            r1=r1_dvr.x,
            r2=r2_dvr.x,
            theta=theta,
            basis=np.asarray(args.basis),
            ncas=args.ncas,
            nelecas=args.nelecas,
            grid_label=np.asarray(label),
        )
        print(f"[cache] saved {cache}")

    levels, kinetic, hamiltonian = solve_ground_surface(
        r1_dvr,
        r2_dvr,
        theta,
        apes,
        min(args.nlevels, r1_dvr.npts * r2_dvr.npts - 1),
    )
    plot_apes(r1_dvr, r2_dvr, theta, apes, plot_path, label)
    return apes, levels, kinetic, hamiltonian


def print_result(label, dvr, apes, levels, kinetic, hamiltonian, plot_path):
    rel = (levels - levels[0]) * HARTREE_TO_EV
    print(f"\n[{label}]")
    print("[grid] active nodes =", np.array2string(dvr.x, precision=8))
    print(f"[matrix] kinetic shape={kinetic.shape}, nnz={kinetic.nnz}")
    print(f"[matrix] hamiltonian nnz={hamiltonian.nnz}")
    print("[apes] min/max E0 Eh =", float(apes[:, :, 0].min()), float(apes[:, :, 0].max()))
    print("[levels] absolute Eh =", np.array2string(levels, precision=10))
    print("[levels] relative eV =", np.array2string(rel, precision=8))
    print(f"[plot] {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--nelecas", type=int, default=2)
    parser.add_argument("--nstates", type=int, default=1)
    parser.add_argument("--r-min", type=float, default=0.90)
    parser.add_argument("--r-max", type=float, default=3.20)
    parser.add_argument("--theta-deg", type=float, default=60.0)
    parser.add_argument("--n-elements", type=int, default=2)
    parser.add_argument("--n-lobatto", type=int, default=3)
    parser.add_argument(
        "--sine-npts",
        type=int,
        default=None,
        help="Sine-DVR points per coordinate. Defaults to the FE-DVR active node count.",
    )
    parser.add_argument("--skip-sine", action="store_true")
    parser.add_argument("--skip-dynamics", action="store_true")
    parser.add_argument("--nlevels", type=int, default=5)
    parser.add_argument("--nt", type=int, default=200)
    parser.add_argument("--dt-fs", type=float, default=0.05)
    parser.add_argument("--nout", type=int, default=2)
    parser.add_argument("--packet-r1", type=float, default=1.45)
    parser.add_argument("--packet-r2", type=float, default=2.45)
    parser.add_argument("--packet-width", type=float, default=20.0)
    parser.add_argument(
        "--packet-project-levels",
        type=int,
        default=None,
        help="Project the initial packet into the lowest N eigenstates before propagation.",
    )
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_fedvr_fixed_theta"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    fedvr_cache = args.outdir / (
        f"h3plus_fedvr_theta{args.theta_deg:.1f}_"
        f"r{args.r_min:.2f}_{args.r_max:.2f}_"
        f"e{args.n_elements}_p{args.n_lobatto}.npz"
    )
    fedvr_plot = fedvr_cache.with_suffix(".png")

    theta = np.deg2rad(args.theta_deg)
    r1_dvr = FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto)
    r2_dvr = FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto)

    apes, levels, kinetic, hamiltonian = run_basis(
        "FE-DVR",
        r1_dvr,
        r2_dvr,
        theta,
        args,
        fedvr_cache,
        fedvr_plot,
    )
    print_result("FE-DVR", r1_dvr, apes, levels, kinetic, hamiltonian, fedvr_plot)

    if not args.skip_sine:
        sine_npts = args.sine_npts or r1_dvr.npts
        sine_r1 = SineDVR(args.r_min, args.r_max, sine_npts)
        sine_r2 = SineDVR(args.r_min, args.r_max, sine_npts)
        sine_cache = args.outdir / (
            f"h3plus_sine_theta{args.theta_deg:.1f}_"
            f"r{args.r_min:.2f}_{args.r_max:.2f}_n{sine_npts}.npz"
        )
        sine_plot = sine_cache.with_suffix(".png")
        sine_apes, sine_levels, sine_kinetic, sine_hamiltonian = run_basis(
            "sine DVR",
            sine_r1,
            sine_r2,
            theta,
            args,
            sine_cache,
            sine_plot,
        )
        print_result(
            "sine DVR",
            sine_r1,
            sine_apes,
            sine_levels,
            sine_kinetic,
            sine_hamiltonian,
            sine_plot,
        )

        ncompare = min(len(levels), len(sine_levels))
        print("\n[compare] FE-DVR minus sine DVR levels")
        print(
            "[compare] absolute Eh =",
            np.array2string(levels[:ncompare] - sine_levels[:ncompare], precision=10),
        )
        print(
            "[compare] relative eV =",
            np.array2string(
                (levels[:ncompare] - sine_levels[:ncompare]) * HARTREE_TO_EV,
                precision=8,
            ),
        )

        if not args.skip_dynamics:
            center = (args.packet_r1, args.packet_r2)
            dynamics = {}
            print(
                "\n[dynamics] propagating Gaussian packet centered at "
                f"r1={center[0]:.3f}, r2={center[1]:.3f} bohr"
            )
            print(
                f"[dynamics] nt={args.nt}, dt={args.dt_fs:.4f} fs, "
                f"output every {args.nout} steps"
            )
            if args.packet_project_levels is not None:
                print(
                    "[dynamics] projecting packet onto lowest "
                    f"{args.packet_project_levels} eigenstates"
                )

            for label, dvr_a, dvr_b, hmat in (
                ("FE-DVR", r1_dvr, r2_dvr, hamiltonian),
                ("sine DVR", sine_r1, sine_r2, sine_hamiltonian),
            ):
                psi0 = initial_packet(dvr_a, dvr_b, center, args.packet_width)
                eavg, weights = spectral_weights(hmat, psi0)
                initial_pops = packet_region_populations(
                    dvr_a,
                    dvr_b,
                    center,
                    args.packet_width,
                    hmat,
                    args.packet_project_levels,
                )
                rel_e = (eavg - levels[0]) * HARTREE_TO_EV
                if label == "sine DVR":
                    rel_e = (eavg - sine_levels[0]) * HARTREE_TO_EV
                weights_text = ", ".join(
                    f"first {n}: {w:.3f}" for n, w in weights.items()
                )
                print(
                    f"[dynamics:{label}] unprojected packet <E>-E0={rel_e:.3f} eV; "
                    f"weights {weights_text}"
                )
                print(
                    f"[dynamics:{label}] initial r1<r2={initial_pops[0]:.6f}, "
                    f"r1>r2={initial_pops[1]:.6f}, "
                    f"diagonal={initial_pops[2]:.6f}, norm={initial_pops[3]:.8f}"
                )

            dynamics["FE-DVR"] = propagate_region_populations(
                hamiltonian,
                r1_dvr,
                r2_dvr,
                args.dt_fs,
                args.nt,
                args.nout,
                center,
                args.packet_width,
                args.packet_project_levels,
            )
            dynamics["sine DVR"] = propagate_region_populations(
                sine_hamiltonian,
                sine_r1,
                sine_r2,
                args.dt_fs,
                args.nt,
                args.nout,
                center,
                args.packet_width,
                args.packet_project_levels,
            )
            project_suffix = ""
            if args.packet_project_levels is not None:
                project_suffix = f"_proj{args.packet_project_levels}"
            pop_plot = args.outdir / (
                f"h3plus_fixed_theta_populations_"
                f"r{args.r_min:.2f}_{args.r_max:.2f}_"
                f"fe{args.n_elements}p{args.n_lobatto}_"
                f"sine{sine_npts}{project_suffix}.png"
            )
            plot_populations(dynamics, pop_plot)
            for label, (_, pops) in dynamics.items():
                print(
                    f"[dynamics:{label}] final r1<r2={pops[-1, 0]:.6f}, "
                    f"r1>r2={pops[-1, 1]:.6f}, diagonal={pops[-1, 2]:.6f}, "
                    f"norm={pops[-1, 3]:.8f}"
                )
            print(f"[plot] {pop_plot}")


if __name__ == "__main__":
    main()
