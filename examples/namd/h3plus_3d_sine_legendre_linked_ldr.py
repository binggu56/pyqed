#!/usr/bin/env python3
"""3D H3+ sine/Legendre DVR-LDR dynamics with linked electronic overlaps.

Coordinates are (r1, r2, theta).  The default grid uses sine DVRs for the two
bond stretches and a Legendre DVR for the bend angle.  Electronic structure is
native AM1/MECI, using the same calculations for APES values and nearest-
neighbor electronic overlap links.  Propagation uses ``expm_multiply`` with a
matrix-free linked-overlap kinetic action.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom
from pyqed.units import au2fs


@contextmanager
def working_directory(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def h3plus_body_frame(r: float = 1.65, theta: float = np.pi / 3.0):
    return [
        ["H", (float(r), 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


def nearest_index_per_axis(solver: Triatom, center):
    return tuple(
        int(np.argmin(np.abs(np.asarray(axis_grid) - center[axis])))
        for axis, axis_grid in enumerate(solver.x)
    )


def phase_projected_packet(
    solver: Triatom,
    state: int,
    center,
    sigma_r: float,
    sigma_theta: float,
    mode: str = "adiabatic",
):
    """Gaussian nuclear packet with linked-overlap phase control.

    ``mode='adiabatic'`` initializes the local adiabatic ``state`` at every
    node, with the diagonal linked product used only as a smooth phase.  This
    gives a pure local-state population at t=0.  ``mode='reference'`` projects
    the electronic state at the reference geometry onto each local adiabatic
    basis, which is useful for diabatic-like launches but can start with mixed
    local adiabatic populations.
    """
    if state < 0 or state >= solver.nstates:
        raise ValueError(f"initial state {state} is outside 0..{solver.nstates - 1}")
    if mode not in ("adiabatic", "reference"):
        raise ValueError("mode must be 'adiabatic' or 'reference'")

    center = np.asarray(center, dtype=float)
    sigmas = np.asarray([sigma_r, sigma_r, sigma_theta], dtype=float)
    ref_idx = nearest_index_per_axis(solver, center)
    links = getattr(solver, "overlap_links", None)

    psi_values = np.zeros((*solver.nx, solver.nstates), dtype=complex)
    for idx in solver._grid_indices():
        q = np.asarray([solver.x[axis][idx[axis]] for axis in range(solver.ndim)])
        scalar = np.exp(-0.5 * np.sum(((q - center) / sigmas) ** 2))
        if links is None:
            coeff = np.zeros(solver.nstates, dtype=complex)
            coeff[state] = 1.0
        else:
            link_to_ref = solver._linked_overlap_between(idx, ref_idx, links, solver.nstates)
            if mode == "reference":
                coeff = link_to_ref[:, state]
            else:
                coeff = np.zeros(solver.nstates, dtype=complex)
                phase = link_to_ref[state, state]
                coeff[state] = phase / abs(phase) if abs(phase) > 1.0e-14 else 1.0
        psi_values[idx] = scalar * coeff

    psi = solver.to_quadrature_normalized(psi_values)
    norm = solver.norm(psi)
    if norm == 0.0:
        raise RuntimeError("Initial wavepacket has zero norm.")
    return psi / norm, ref_idx


def probability_density(psi):
    return np.sum(np.abs(psi) ** 2, axis=-1)


def electronic_populations(psilist):
    return np.asarray([np.sum(np.abs(psi) ** 2, axis=(0, 1, 2)) for psi in psilist])


def arrangement_populations(solver: Triatom, psilist, delta: float):
    r1, r2, _ = np.meshgrid(*solver.x, indexing="ij")
    masks = [
        r1 < r2 - delta,
        r2 < r1 - delta,
        np.abs(r1 - r2) <= delta,
    ]
    labels = [f"r1 < r2 - {delta:g}", f"r2 < r1 - {delta:g}", f"|r1-r2| <= {delta:g}"]
    pops = []
    for psi in psilist:
        rho = probability_density(psi)
        pops.append([float(np.sum(rho[mask])) for mask in masks])
    return labels, np.asarray(pops)


def theta_density(solver: Triatom, psilist):
    weights = np.asarray(solver.w[2], dtype=float)
    densities = []
    for psi in psilist:
        marginal = np.sum(probability_density(psi), axis=(0, 1))
        densities.append(marginal / weights)
    return np.asarray(densities)


def grid_edges(nodes, lo, hi):
    nodes = np.asarray(nodes, dtype=float)
    edges = np.empty(nodes.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
    edges[0] = lo
    edges[-1] = hi
    return edges


def save_observables_npz(
    path,
    solver,
    result,
    times_fs,
    electronic,
    arrangement,
    theta_rho,
    arrangement_labels,
):
    np.savez(
        path,
        times_fs=times_fs,
        psi_t=np.asarray(result["psilist"]),
        r1=solver.x[0],
        r2=solver.x[1],
        theta=solver.x[2],
        theta_deg=np.rad2deg(solver.x[2]),
        electronic_populations=electronic,
        arrangement_populations=arrangement,
        arrangement_labels=np.asarray(arrangement_labels),
        theta_density=theta_rho,
    )


def plot_electronic(times_fs, pops, outpath: Path):
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    for state in range(pops.shape[1]):
        ax.plot(times_fs, pops[:, state], marker="o", markersize=3, label=f"S{state}")
    ax.set_xlabel("time / fs")
    ax.set_ylabel("population")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Electronic population")
    ax.legend()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_arrangement(times_fs, labels, pops, outpath: Path):
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    for label, values in zip(labels, pops.T):
        ax.plot(times_fs, values, marker="o", markersize=3, label=label)
    ax.set_xlabel("time / fs")
    ax.set_ylabel("population")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Permutation/arrangement population")
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_theta_density(solver, times_fs, theta_rho, theta_min, theta_max, outpath: Path):
    theta_edges = np.rad2deg(grid_edges(solver.x[2], theta_min, theta_max))
    time_edges = grid_edges(times_fs, times_fs[0], times_fs[-1])
    if len(times_fs) == 1:
        time_edges = np.asarray([times_fs[0] - 0.5, times_fs[0] + 0.5])

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    mesh = ax.pcolormesh(theta_edges, time_edges, theta_rho, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=r"$P(\theta)$")
    ax.set_xlabel(r"$\theta$ / degree")
    ax.set_ylabel("time / fs")
    ax.set_title(r"Bending density")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_r1r2_snapshots(solver, result, times_fs, requested_times, outpath: Path):
    if not requested_times:
        return
    chosen = []
    for target in requested_times:
        idx = int(np.argmin(np.abs(times_fs - target)))
        if idx not in chosen:
            chosen.append(idx)

    ncols = len(chosen)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    r1, r2 = np.meshgrid(solver.x[0], solver.x[1], indexing="ij")
    densities = []
    for idx in chosen:
        rho = np.sum(probability_density(result["psilist"][idx]), axis=2)
        peak = float(np.max(rho))
        if peak > 0.0:
            rho = rho / peak
        densities.append(rho)

    for ax, idx, rho in zip(axes, chosen, densities):
        mesh = ax.pcolormesh(r1, r2, rho, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_xlabel("r1 / bohr")
        ax.set_ylabel("r2 / bohr")
        ax.set_title(f"{times_fs[idx]:.2f} fs")
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="relative r1/r2 density")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_state_resolved_r1r2_wavepackets(solver, result, times_fs, requested_times, outpath: Path):
    if not requested_times:
        return
    chosen = []
    for target in requested_times:
        idx = int(np.argmin(np.abs(times_fs - target)))
        if idx not in chosen:
            chosen.append(idx)

    nrows = solver.nstates
    ncols = len(chosen)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 2.55 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    r1, r2 = np.meshgrid(solver.x[0], solver.x[1], indexing="ij")
    global_max = 0.0
    densities = {}
    for state in range(solver.nstates):
        for idx in chosen:
            rho = np.sum(np.abs(result["psilist"][idx][..., state]) ** 2, axis=2)
            densities[(state, idx)] = rho
            global_max = max(global_max, float(np.max(rho)))

    mesh = None
    for row, state in enumerate(range(solver.nstates)):
        vmax = global_max if global_max > 0.0 else 1.0
        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                r1,
                r2,
                densities[(state, idx)],
                shading="auto",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            if row == 0:
                ax.set_title(f"{times_fs[idx]:.2f} fs")
            if col == 0:
                ax.set_ylabel(f"S{state}\nr2 / bohr")
            else:
                ax.set_yticklabels([])
            if row == nrows - 1:
                ax.set_xlabel("r1 / bohr")
            else:
                ax.set_xticklabels([])
            ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="probability integrated over theta")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def parse_float_list(text: str):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def load_cached_scan(solver: Triatom, outdir: Path):
    apes_path = outdir / "apes.npz"
    links_path = outdir / "overlap_links.npz"
    if not apes_path.exists() or not links_path.exists():
        return False
    solver.apes = np.load(apes_path, allow_pickle=True)["data"]
    if solver.apes.shape != (*solver.nx, solver.nstates):
        solver.apes = None
        return False
    packed = np.load(links_path, allow_pickle=True)
    solver.overlap_links = solver._unpack_overlap_links(
        packed["axes"],
        packed["indices"],
        packed["data"],
    )
    solver.overlap_matrix = None
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r", type=int, default=11, help="Use the same point count for r1 and r2.")
    parser.add_argument("--n-r1", type=int, default=None)
    parser.add_argument("--n-r2", type=int, default=None)
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument("--r-min", type=float, default=1.25)
    parser.add_argument("--r-max", type=float, default=2.25)
    parser.add_argument("--theta-min-deg", type=float, default=35.0)
    parser.add_argument("--theta-max-deg", type=float, default=115.0)
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--initial-state", type=int, default=1)
    parser.add_argument(
        "--initial-mode",
        choices=("adiabatic", "reference"),
        default="adiabatic",
        help="Use phase-aligned local adiabatic state or full projection of the reference electronic state.",
    )
    parser.add_argument("--center-r1", type=float, default=1.65)
    parser.add_argument("--center-r2", type=float, default=1.65)
    parser.add_argument("--center-theta-deg", type=float, default=60.0)
    parser.add_argument("--sigma-r", type=float, default=0.12)
    parser.add_argument("--sigma-theta-deg", type=float, default=7.5)
    parser.add_argument("--arrangement-delta", type=float, default=0.05)
    parser.add_argument("--dt-fs", type=float, default=0.05)
    parser.add_argument("--nt", type=int, default=80)
    parser.add_argument("--nout", type=int, default=2)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--unitarize-overlap-links", action="store_true")
    parser.add_argument("--reuse-scan", action="store_true")
    parser.add_argument("--snapshots-fs", default="0,1,2,4")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_3d_sine_legendre_linked_ldr"),
    )
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    n_r1 = args.n_r if args.n_r1 is None else args.n_r1
    n_r2 = args.n_r if args.n_r2 is None else args.n_r2
    theta_min = np.deg2rad(args.theta_min_deg)
    theta_max = np.deg2rad(args.theta_max_deg)
    center = np.asarray(
        [args.center_r1, args.center_r2, np.deg2rad(args.center_theta_deg)],
        dtype=float,
    )

    solver = Triatom(
        h3plus_body_frame(r=args.center_r1, theta=center[2]),
        nstates=args.nstates,
        charge=1,
        spin=0,
        unit="bohr",
        dvr_type=["sine", "sine", "legendre"],
    )
    solver.set_dvr(
        domains=[[args.r_min, args.r_max], [args.r_min, args.r_max], [theta_min, theta_max]],
        npts=[n_r1, n_r2, args.n_theta],
        dvr_type=["sine", "sine", "legendre"],
    )

    print(
        "[grid] sine(r1) x sine(r2) x legendre(theta) = "
        f"{n_r1} x {n_r2} x {args.n_theta} ({np.prod(solver.nx)} nuclear points)"
    )
    print(
        "[theta] Legendre nodes/deg =",
        np.array2string(np.rad2deg(solver.x[2]), precision=4),
    )

    scan_start = time.perf_counter()
    with working_directory(outdir):
        reused = args.reuse_scan and load_cached_scan(solver, outdir)
        if reused:
            print(f"[scan] Reused APES and overlap links from {outdir}")
        else:
            solver.scan_pes(
                electronic_method="am1/meci",
                nstates=args.nstates,
                ncas=args.ncas,
                nelecas=2,
                overlap_method="link-only",
                unitarize_overlap_links=args.unitarize_overlap_links,
                n_workers=args.n_workers,
                worker_threads=1,
                scf_tol=args.scf_tol,
                max_cycle=args.max_cycle,
                damping=args.damping,
            )
    scan_time = time.perf_counter() - scan_start
    print(f"[timing] APES + nearest-neighbor overlaps: {scan_time:.2f} s")
    print("[apes] min energies/Eh =", np.array2string(np.nanmin(solver.apes, axis=(0, 1, 2)), precision=10))
    print("[overlap] nearest-neighbor links =", len(solver.overlap_links))

    psi0, ref_idx = phase_projected_packet(
        solver,
        state=args.initial_state,
        center=center,
        sigma_r=args.sigma_r,
        sigma_theta=np.deg2rad(args.sigma_theta_deg),
        mode=args.initial_mode,
    )
    print(f"[initial] reference grid index = {ref_idx}, norm = {solver.norm(psi0):.12f}")

    prop_start = time.perf_counter()
    result = solver.run(
        psi0,
        dt=args.dt_fs / au2fs,
        nt=args.nt,
        nout=args.nout,
        kinetic_propagator="expm_multiply",
        kinetic_action="matrix-free",
    )
    prop_time = time.perf_counter() - prop_start
    times_fs = np.asarray(result["times"]) * au2fs
    print(f"[timing] matrix-free Krylov propagation: {prop_time:.2f} s")

    electronic = electronic_populations(result["psilist"])
    arrangement_labels, arrangement = arrangement_populations(
        solver,
        result["psilist"],
        delta=args.arrangement_delta,
    )
    theta_rho = theta_density(solver, result["psilist"])

    data_path = outdir / "h3plus_3d_sine_legendre_linked_ldr_observables.npz"
    save_observables_npz(
        data_path,
        solver,
        result,
        times_fs,
        electronic,
        arrangement,
        theta_rho,
        arrangement_labels,
    )

    pop_png = outdir / "h3plus_3d_electronic_population.png"
    arr_png = outdir / "h3plus_3d_arrangement_population.png"
    theta_png = outdir / "h3plus_3d_theta_density.png"
    snap_png = outdir / "h3plus_3d_r1r2_density_snapshots.png"
    state_wavepacket_png = outdir / "h3plus_3d_state_resolved_r1r2_wavepackets.png"
    plot_electronic(times_fs, electronic, pop_png)
    plot_arrangement(times_fs, arrangement_labels, arrangement, arr_png)
    plot_theta_density(solver, times_fs, theta_rho, theta_min, theta_max, theta_png)
    plot_r1r2_snapshots(
        solver,
        result,
        times_fs,
        parse_float_list(args.snapshots_fs),
        snap_png,
    )
    plot_state_resolved_r1r2_wavepackets(
        solver,
        result,
        times_fs,
        parse_float_list(args.snapshots_fs),
        state_wavepacket_png,
    )

    print("[final electronic]", np.array2string(electronic[-1], precision=8))
    print("[final arrangement]", np.array2string(arrangement[-1], precision=8))
    print(f"[data] {data_path}")
    print(f"[plot] {pop_png}")
    print(f"[plot] {arr_png}")
    print(f"[plot] {theta_png}")
    if parse_float_list(args.snapshots_fs):
        print(f"[plot] {snap_png}")
        print(f"[plot] {state_wavepacket_png}")


if __name__ == "__main__":
    main()
