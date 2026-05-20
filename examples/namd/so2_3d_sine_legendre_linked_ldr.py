#!/usr/bin/env python3
"""SO2 3D sine/Legendre DVR-LDR test with linked AM1/MECI overlaps."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom
from pyqed.units import au2fs

from h3plus_3d_sine_legendre_linked_ldr import (
    arrangement_populations,
    electronic_populations,
    load_cached_scan,
    parse_float_list,
    phase_projected_packet,
    plot_arrangement,
    plot_electronic,
    plot_theta_density,
    save_observables_npz,
    theta_density,
    working_directory,
)

HARTREE_TO_EV = 27.211386245988


def so2_body_frame(r: float = 2.70, theta: float = np.deg2rad(119.5)):
    return [
        ["O", (float(r), 0.0, 0.0)],
        ["S", (0.0, 0.0, 0.0)],
        ["O", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


def nearest_axis_index(axis, value):
    return int(np.argmin(np.abs(np.asarray(axis) - value)))


def relative_apes_ev(apes):
    return (apes - np.nanmin(apes[..., 0])) * HARTREE_TO_EV


def plot_symmetric_stretch_surfaces(solver: Triatom, center, outpath: Path):
    theta_idx = nearest_axis_index(solver.x[2], center[2])
    ndiag = min(solver.nx[0], solver.nx[1])
    r = np.asarray([0.5 * (solver.x[0][i] + solver.x[1][i]) for i in range(ndiag)])
    e = relative_apes_ev(solver.apes)

    fig, ax = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    for state in range(solver.nstates):
        ax.plot(r, [e[i, i, theta_idx, state] for i in range(ndiag)], marker="o", label=f"S{state}")
    ax.set_xlabel("symmetric S-O stretch / bohr")
    ax.set_ylabel("energy / eV")
    ax.set_title(rf"SO2 APES, $\theta \approx {np.rad2deg(solver.x[2][theta_idx]):.1f}^\circ$")
    ax.legend()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_bending_surfaces(solver: Triatom, center, outpath: Path):
    r1_idx = nearest_axis_index(solver.x[0], center[0])
    r2_idx = nearest_axis_index(solver.x[1], center[1])
    theta_deg = np.rad2deg(solver.x[2])
    e = relative_apes_ev(solver.apes)

    fig, ax = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    for state in range(solver.nstates):
        ax.plot(theta_deg, e[r1_idx, r2_idx, :, state], marker="o", label=f"S{state}")
    ax.set_xlabel(r"$\theta$ / degree")
    ax.set_ylabel("energy / eV")
    ax.set_title(rf"SO2 bending APES, r $\approx {solver.x[0][r1_idx]:.2f}$ bohr")
    ax.legend()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_r1r2_surfaces(solver: Triatom, center, outpath: Path):
    theta_idx = nearest_axis_index(solver.x[2], center[2])
    r1, r2 = np.meshgrid(solver.x[0], solver.x[1], indexing="ij")
    e = relative_apes_ev(solver.apes)
    fig, axes = plt.subplots(
        1,
        solver.nstates,
        figsize=(3.4 * solver.nstates, 3.25),
        squeeze=False,
        constrained_layout=True,
    )
    for state, ax in enumerate(axes[0]):
        z = e[:, :, theta_idx, state]
        mesh = ax.contourf(r1, r2, z, levels=14, cmap="viridis")
        ax.plot(center[0], center[1], "wo", markersize=4, markeredgecolor="black")
        ax.set_title(f"S{state}")
        ax.set_xlabel("r1 / bohr")
        if state == 0:
            ax.set_ylabel("r2 / bohr")
        else:
            ax.set_yticklabels([])
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, label="eV")
    fig.suptitle(rf"SO2 r1/r2 APES, $\theta \approx {np.rad2deg(solver.x[2][theta_idx]):.1f}^\circ$")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_surfaces(solver: Triatom, center, outdir: Path):
    paths = {
        "stretch": outdir / "so2_apes_symmetric_stretch.png",
        "bend": outdir / "so2_apes_bend.png",
        "r1r2": outdir / "so2_apes_r1r2_contours.png",
    }
    plot_symmetric_stretch_surfaces(solver, center, paths["stretch"])
    plot_bending_surfaces(solver, center, paths["bend"])
    plot_r1r2_surfaces(solver, center, paths["r1r2"])
    return paths


def smooth_r1r2_density(density: np.ndarray, sigma: float = 0.75) -> np.ndarray:
    if sigma <= 0.0:
        return density
    return gaussian_filter(density, sigma=sigma, mode="nearest")


def plot_r1r2_snapshots_smoothed(
    solver: Triatom,
    result: dict,
    times_fs: np.ndarray,
    requested_times: list[float],
    outpath: Path,
    *,
    sigma: float = 0.75,
):
    if not requested_times:
        return
    chosen = []
    for target in requested_times:
        idx = int(np.argmin(np.abs(times_fs - target)))
        if idx not in chosen:
            chosen.append(idx)

    ncols = len(chosen)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(3.1 * ncols, 3.0),
        squeeze=False,
        constrained_layout=True,
    )
    extent = [solver.x[0][0], solver.x[0][-1], solver.x[1][0], solver.x[1][-1]]
    densities = []
    for idx in chosen:
        rho = np.sum(np.abs(result["psilist"][idx]) ** 2, axis=(2, 3))
        rho = smooth_r1r2_density(rho, sigma=sigma)
        peak = float(np.max(rho))
        if peak > 0.0:
            rho = rho / peak
        densities.append(rho)

    image = None
    for ax, idx, rho in zip(axes[0], chosen, densities):
        image = ax.imshow(
            rho.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="magma",
            interpolation="bicubic",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"{times_fs[idx]:.0f} fs")
        ax.set_xlabel("r1 / bohr")
        ax.set_ylabel("r2 / bohr")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="relative r1/r2 density")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_state_resolved_r1r2_wavepackets_smoothed(
    solver: Triatom,
    result: dict,
    times_fs: np.ndarray,
    requested_times: list[float],
    outpath: Path,
    *,
    sigma: float = 0.75,
    omit_states: tuple[int, ...] = (),
    normalize_each_panel: bool = True,
):
    if not requested_times:
        return
    chosen = []
    for target in requested_times:
        idx = int(np.argmin(np.abs(times_fs - target)))
        if idx not in chosen:
            chosen.append(idx)

    states = [state for state in range(solver.nstates) if state not in omit_states]
    nrows = len(states)
    ncols = len(chosen)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.55 * ncols, 2.25 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    extent = [solver.x[0][0], solver.x[0][-1], solver.x[1][0], solver.x[1][-1]]
    densities = {}
    for state in states:
        for idx in chosen:
            rho = np.sum(np.abs(result["psilist"][idx][..., state]) ** 2, axis=2)
            rho = smooth_r1r2_density(rho, sigma=sigma)
            densities[(state, idx)] = rho

    image = None
    for row, state in enumerate(states):
        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            vmax = float(np.max(densities[(state, idx)])) if normalize_each_panel else 0.0
            if vmax <= 0.0:
                vmax = max(
                    float(np.max(densities[(state, other_idx)])) for other_idx in chosen
                )
            if vmax <= 0.0:
                vmax = 1.0
            image = ax.imshow(
                densities[(state, idx)].T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="magma",
                interpolation="bicubic",
                vmin=0.0,
                vmax=vmax,
            )
            state_pop = float(np.sum(np.abs(result["psilist"][idx][..., state]) ** 2))
            ax.text(
                0.03,
                0.93,
                f"P={state_pop:.3f}",
                transform=ax.transAxes,
                color="white",
                fontsize=8,
                ha="left",
                va="top",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
            )
            if row == 0:
                ax.set_title(f"{times_fs[idx]:.0f} fs")
            if col == 0:
                ax.set_ylabel(f"S{state}\nr2 / bohr")
            else:
                ax.set_yticklabels([])
            if row == nrows - 1:
                ax.set_xlabel("r1 / bohr")
            else:
                ax.set_xticklabels([])
    if normalize_each_panel:
        fig.suptitle("Each panel is independently z-scaled")
    else:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="probability integrated over theta")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r", type=int, default=7)
    parser.add_argument("--n-r1", type=int, default=None)
    parser.add_argument("--n-r2", type=int, default=None)
    parser.add_argument("--n-theta", type=int, default=7)
    parser.add_argument("--r-min", type=float, default=2.20)
    parser.add_argument("--r-max", type=float, default=3.40)
    parser.add_argument("--theta-min-deg", type=float, default=90.0)
    parser.add_argument("--theta-max-deg", type=float, default=150.0)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--initial-mode", choices=("adiabatic", "reference"), default="adiabatic")
    parser.add_argument("--center-r1", type=float, default=2.70)
    parser.add_argument("--center-r2", type=float, default=2.70)
    parser.add_argument("--center-theta-deg", type=float, default=119.5)
    parser.add_argument("--sigma-r", type=float, default=0.16)
    parser.add_argument("--sigma-theta-deg", type=float, default=7.5)
    parser.add_argument("--arrangement-delta", type=float, default=0.05)
    parser.add_argument("--dt-fs", type=float, default=0.10)
    parser.add_argument("--nt", type=int, default=80)
    parser.add_argument("--nout", type=int, default=2)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--scf-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--unitarize-overlap-links", action="store_true")
    parser.add_argument("--reuse-scan", action="store_true")
    parser.add_argument("--snapshots-fs", default="0,2,5,8")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("so2_3d_sine_legendre_linked_ldr"),
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
        so2_body_frame(args.center_r1, center[2]),
        nstates=args.nstates,
        charge=0,
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
        "[grid] SO2 sine(r1) x sine(r2) x legendre(theta) = "
        f"{n_r1} x {n_r2} x {args.n_theta} ({np.prod(solver.nx)} nuclear points)"
    )
    print("[theta] Legendre nodes/deg =", np.array2string(np.rad2deg(solver.x[2]), precision=4))

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
    print(f"[timing] APES + nearest-neighbor overlaps: {time.perf_counter() - scan_start:.2f} s")
    print("[apes] min energies/Eh =", np.array2string(np.nanmin(solver.apes, axis=(0, 1, 2)), precision=10))
    print("[overlap] nearest-neighbor links =", len(solver.overlap_links))

    surface_paths = plot_surfaces(solver, center, outdir)
    for path in surface_paths.values():
        print(f"[plot] {path}")

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
        matrix_free_kinetic=True,
    )
    print(f"[timing] matrix-free Krylov propagation: {time.perf_counter() - prop_start:.2f} s")

    times_fs = np.asarray(result["times"]) * au2fs
    electronic = electronic_populations(result["psilist"])
    arrangement_labels, arrangement = arrangement_populations(
        solver,
        result["psilist"],
        delta=args.arrangement_delta,
    )
    theta_rho = theta_density(solver, result["psilist"])
    data_path = outdir / "so2_3d_sine_legendre_linked_ldr_observables.npz"
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

    pop_png = outdir / "so2_3d_electronic_population.png"
    arr_png = outdir / "so2_3d_arrangement_population.png"
    theta_png = outdir / "so2_3d_theta_density.png"
    snap_png = outdir / "so2_3d_r1r2_density_snapshots.png"
    state_png = outdir / "so2_3d_state_resolved_r1r2_wavepackets.png"
    snapshots = parse_float_list(args.snapshots_fs)
    plot_electronic(times_fs, electronic, pop_png)
    plot_arrangement(times_fs, arrangement_labels, arrangement, arr_png)
    plot_theta_density(solver, times_fs, theta_rho, theta_min, theta_max, theta_png)
    plot_r1r2_snapshots_smoothed(solver, result, times_fs, snapshots, snap_png)
    plot_state_resolved_r1r2_wavepackets_smoothed(
        solver,
        result,
        times_fs,
        snapshots,
        state_png,
        omit_states=(0,) if solver.nstates > 3 else (),
    )

    print("[final electronic]", np.array2string(electronic[-1], precision=8))
    print("[final arrangement]", np.array2string(arrangement[-1], precision=8))
    print(f"[data] {data_path}")
    for path in (pop_png, arr_png, theta_png, snap_png, state_png):
        print(f"[plot] {path}")


if __name__ == "__main__":
    main()
