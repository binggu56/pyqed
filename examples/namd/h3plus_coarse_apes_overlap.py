#!/usr/bin/env python3
"""Coarse H3+ CASCI/LDR scan with APES and overlap diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom

HARTREE_TO_EV = 27.211386245988


def h3plus_body_frame(r1: float, r2: float, theta: float):
    return [
        ["H", (float(r1), 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (float(r2) * np.cos(theta), float(r2) * np.sin(theta), 0.0)],
    ]


def overlap_trace_matrix(overlap_matrix, nstates):
    shape = overlap_matrix.shape
    ngrid = int(np.prod(shape[:3]))
    A = overlap_matrix.reshape(ngrid, nstates, ngrid, nstates)
    return np.real(np.trace(A, axis1=1, axis2=3)) / nstates


def grid_point_table(solver):
    points = []
    for idx in np.ndindex(*solver.nx):
        points.append((
            idx,
            solver.x[0][idx[0]],
            solver.x[1][idx[1]],
            np.rad2deg(solver.x[2][idx[2]]),
        ))
    return points


def plot_apes(solver, outpath):
    apes_ev = (solver.apes - solver.apes[..., [0]].min()) * HARTREE_TO_EV
    mid_theta = len(solver.x[2]) // 2
    theta_deg = np.rad2deg(solver.x[2][mid_theta])
    extent = [solver.x[1][0], solver.x[1][-1], solver.x[0][0], solver.x[0][-1]]

    fig, axes = plt.subplots(1, solver.nstates, figsize=(4.2 * solver.nstates, 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for state, ax in enumerate(axes):
        im = ax.imshow(
            apes_ev[:, :, mid_theta, state],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="magma",
        )
        ax.set_title(f"S{state}, theta={theta_deg:.1f} deg")
        ax.set_xlabel("r2 / bohr")
        ax.set_ylabel("r1 / bohr")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("E - min(S0) / eV")

    fig.suptitle("H3+ coarse CASCI APES slice")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_overlap(solver, outpath):
    O = overlap_trace_matrix(solver.overlap_matrix, solver.nstates)

    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    im = ax.imshow(O, origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Linked-product overlap diagnostic")
    ax.set_xlabel("grid point")
    ax.set_ylabel("grid point")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Tr(A_nm) / nstates")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_apes_3d(solver, outpath):
    apes_ev = (solver.apes - solver.apes[..., [0]].min()) * HARTREE_TO_EV
    points = grid_point_table(solver)
    coords = np.asarray([[r1, r2, theta] for _, r1, r2, theta in points])

    fig = plt.figure(figsize=(4.6 * solver.nstates, 4.2), constrained_layout=True)
    for state in range(solver.nstates):
        ax = fig.add_subplot(1, solver.nstates, state + 1, projection="3d")
        values = np.asarray([apes_ev[idx + (state,)] for idx, *_ in points])
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=values,
            s=95,
            cmap="viridis",
            edgecolors="k",
            linewidths=0.35,
        )
        ax.set_title(f"S{state}")
        ax.set_xlabel("r1 / bohr")
        ax.set_ylabel("r2 / bohr")
        ax.set_zlabel("theta / deg")
        ax.view_init(elev=24, azim=-55)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.08)
        cbar.set_label("E - min(S0) / eV")

    fig.suptitle("H3+ coarse 3D APES grid")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_apes_fixed_theta_surface(solver, outpath, theta_index=None):
    if theta_index is None:
        theta_index = len(solver.x[2]) // 2

    apes_ev = (solver.apes - solver.apes[..., [0]].min()) * HARTREE_TO_EV
    r1, r2 = np.meshgrid(solver.x[0], solver.x[1], indexing="ij")
    theta_deg = np.rad2deg(solver.x[2][theta_index])

    fig = plt.figure(figsize=(4.6 * solver.nstates, 4.0), constrained_layout=True)
    for state in range(solver.nstates):
        ax = fig.add_subplot(1, solver.nstates, state + 1, projection="3d")
        z = apes_ev[:, :, theta_index, state]
        surf = ax.plot_surface(
            r1,
            r2,
            z,
            cmap="magma",
            edgecolor="k",
            linewidth=0.35,
            antialiased=True,
            alpha=0.94,
        )
        ax.scatter(r1, r2, z, c="k", s=18, depthshade=False)
        ax.set_title(f"S{state}")
        ax.set_xlabel("r1 / bohr")
        ax.set_ylabel("r2 / bohr")
        ax.set_zlabel("E - min(S0) / eV")
        ax.view_init(elev=28, azim=-130)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.68, pad=0.08)
        cbar.set_label("eV")

    fig.suptitle(f"H3+ APES at fixed theta={theta_deg:.1f} deg")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def nearest_neighbor_link_matrix(solver):
    O = overlap_trace_matrix(solver.overlap_matrix, solver.nstates)
    ngrid = int(np.prod(solver.nx))
    links = np.full((ngrid, ngrid), np.nan)

    for idx in np.ndindex(*solver.nx):
        i = np.ravel_multi_index(idx, solver.nx)
        links[i, i] = 1.0
        for axis in range(solver.ndim):
            nxt = list(idx)
            nxt[axis] += 1
            if nxt[axis] >= solver.nx[axis]:
                continue
            nxt = tuple(nxt)
            j = np.ravel_multi_index(nxt, solver.nx)
            links[i, j] = O[i, j]
            links[j, i] = O[j, i]

    return links


def link_value_for_grid_points(solver, idx_a, idx_b):
    flat_a = np.ravel_multi_index(idx_a, solver.nx)
    flat_b = np.ravel_multi_index(idx_b, solver.nx)
    O = overlap_trace_matrix(solver.overlap_matrix, solver.nstates)
    return float(O[flat_a, flat_b])


def state_link_value_for_grid_points(solver, idx_a, idx_b, bra_state, ket_state):
    flat_a = np.ravel_multi_index(idx_a, solver.nx)
    flat_b = np.ravel_multi_index(idx_b, solver.nx)
    A = solver.overlap_matrix.reshape(
        int(np.prod(solver.nx)),
        solver.nstates,
        int(np.prod(solver.nx)),
        solver.nstates,
    )
    return float(np.real(A[flat_a, bra_state, flat_b, ket_state]))


def plot_overlap_links(solver, outpath):
    links = nearest_neighbor_link_matrix(solver)
    masked = np.ma.masked_invalid(links)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    im = ax.imshow(masked, origin="lower", cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_title("Nearest-neighbor overlap links only")
    ax.set_xlabel("grid point")
    ax.set_ylabel("grid point")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Tr(L_nm) / nstates")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_overlap_links_fixed_theta(solver, outpath, theta_index=None):
    if theta_index is None:
        theta_index = len(solver.x[2]) // 2

    theta_deg = np.rad2deg(solver.x[2][theta_index])
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    for i, r1 in enumerate(solver.x[0]):
        for j, r2 in enumerate(solver.x[1]):
            ax.scatter(r2, r1, s=90, color="black", zorder=3)
            ax.text(r2, r1, f"{i},{j}", color="white", ha="center", va="center", fontsize=7, zorder=4)

            idx = (i, j, theta_index)
            if i + 1 < solver.nx[0]:
                nxt = (i + 1, j, theta_index)
                val = link_value_for_grid_points(solver, idx, nxt)
                ax.plot(
                    [r2, r2],
                    [r1, solver.x[0][i + 1]],
                    color=cmap(norm(val)),
                    linewidth=0.6 + 5.0 * abs(val),
                    alpha=0.9,
                    solid_capstyle="round",
                    zorder=2,
                )
            if j + 1 < solver.nx[1]:
                nxt = (i, j + 1, theta_index)
                val = link_value_for_grid_points(solver, idx, nxt)
                ax.plot(
                    [r2, solver.x[1][j + 1]],
                    [r1, r1],
                    color=cmap(norm(val)),
                    linewidth=0.6 + 5.0 * abs(val),
                    alpha=0.9,
                    solid_capstyle="round",
                    zorder=2,
                )

    ax.set_title(f"Nearest-neighbor links at theta={theta_deg:.1f} deg")
    ax.set_xlabel("r2 / bohr")
    ax.set_ylabel("r1 / bohr")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.9", linewidth=0.8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Tr(L_nm) / nstates; linewidth = |value|")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_state_overlap_links_fixed_theta(solver, outpath, bra_state=1, ket_state=2, theta_index=None):
    if theta_index is None:
        theta_index = len(solver.x[2]) // 2
    if bra_state >= solver.nstates or ket_state >= solver.nstates:
        return

    theta_deg = np.rad2deg(solver.x[2][theta_index])
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    for i, r1 in enumerate(solver.x[0]):
        for j, r2 in enumerate(solver.x[1]):
            ax.scatter(r2, r1, s=90, color="black", zorder=3)
            ax.text(r2, r1, f"{i},{j}", color="white", ha="center", va="center", fontsize=7, zorder=4)

            idx = (i, j, theta_index)
            if i + 1 < solver.nx[0]:
                nxt = (i + 1, j, theta_index)
                val = state_link_value_for_grid_points(solver, idx, nxt, bra_state, ket_state)
                ax.plot(
                    [r2, r2],
                    [r1, solver.x[0][i + 1]],
                    color=cmap(norm(val)),
                    linewidth=0.6 + 7.0 * abs(val),
                    alpha=0.92,
                    solid_capstyle="round",
                    zorder=2,
                )
            if j + 1 < solver.nx[1]:
                nxt = (i, j + 1, theta_index)
                val = state_link_value_for_grid_points(solver, idx, nxt, bra_state, ket_state)
                ax.plot(
                    [r2, solver.x[1][j + 1]],
                    [r1, r1],
                    color=cmap(norm(val)),
                    linewidth=0.6 + 7.0 * abs(val),
                    alpha=0.92,
                    solid_capstyle="round",
                    zorder=2,
                )

    ax.set_title(f"S{bra_state}/S{ket_state} overlap links at theta={theta_deg:.1f} deg")
    ax.set_xlabel("r2 / bohr")
    ax.set_ylabel("r1 / bohr")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.9", linewidth=0.8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"Re L_nm[S{bra_state}, S{ket_state}]; linewidth = |value|")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--nelecas", type=int, default=2)
    parser.add_argument("--npts", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--r-min", type=float, default=1.35)
    parser.add_argument("--r-max", type=float, default=1.85)
    parser.add_argument("--theta-min-deg", type=float, default=50.0)
    parser.add_argument("--theta-max-deg", type=float, default=80.0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_coarse_scan"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache = args.outdir / "h3plus_coarse_casci3_linked.npz"
    apes_png = args.outdir / "h3plus_coarse_apes.png"
    apes_3d_png = args.outdir / "h3plus_coarse_apes_3d.png"
    apes_theta_surface_png = args.outdir / "h3plus_coarse_apes_fixed_theta_surface.png"
    overlap_png = args.outdir / "h3plus_coarse_overlap.png"
    links_png = args.outdir / "h3plus_coarse_overlap_links.png"
    links_theta_png = args.outdir / "h3plus_coarse_overlap_links_fixed_theta.png"
    s1s2_links_theta_png = args.outdir / "h3plus_coarse_s1_s2_overlap_links_fixed_theta.png"

    theta_min = np.deg2rad(args.theta_min_deg)
    theta_max = np.deg2rad(args.theta_max_deg)
    solver = Triatom(
        h3plus_body_frame(1.6, 1.6, np.pi / 3.0),
        basis=args.basis,
        nstates=args.nstates,
        charge=1,
        spin=0,
        unit="bohr",
    )
    solver.set_dvr(
        domains=[[args.r_min, args.r_max], [args.r_min, args.r_max], [theta_min, theta_max]],
        npts=args.npts,
        dvr_params=[
            {"De": 0.2, "a": 1.0, "re": 1.6},
            {"De": 0.2, "a": 1.0, "re": 1.6},
            {},
        ],
    )

    if cache.exists() and not args.force:
        data = np.load(cache)
        solver.apes = data["apes"]
        solver.overlap_matrix = data["overlap_matrix"]
        print(f"[cache] loaded {cache}")
    else:
        solver.scan_pes(
            basis=args.basis,
            nstates=args.nstates,
            ncas=args.ncas,
            nelecas=args.nelecas,
            overlap_method="linked",
            n_workers=args.n_workers,
            worker_threads=args.worker_threads,
        )
        np.savez(
            cache,
            apes=solver.apes,
            overlap_matrix=solver.overlap_matrix,
            r1=solver.x[0],
            r2=solver.x[1],
            theta=solver.x[2],
            npts=np.asarray(args.npts),
        )
        print(f"[cache] saved {cache}")

    plot_apes(solver, apes_png)
    plot_apes_3d(solver, apes_3d_png)
    plot_apes_fixed_theta_surface(solver, apes_theta_surface_png)
    plot_overlap(solver, overlap_png)
    plot_overlap_links(solver, links_png)
    plot_overlap_links_fixed_theta(solver, links_theta_png)
    plot_state_overlap_links_fixed_theta(solver, s1s2_links_theta_png, bra_state=1, ket_state=2)

    rel_ev = (solver.apes - solver.apes[..., [0]].min()) * HARTREE_TO_EV
    O = overlap_trace_matrix(solver.overlap_matrix, solver.nstates)
    print("[grid] r1 =", np.array2string(solver.x[0], precision=6))
    print("[grid] r2 =", np.array2string(solver.x[1], precision=6))
    print("[grid] theta/deg =", np.array2string(np.rad2deg(solver.x[2]), precision=6))
    print("[apes] min energies/Eh by state =", np.array2string(solver.apes.reshape(-1, args.nstates).min(axis=0), precision=10))
    print("[apes] max relative energies/eV by state =", np.array2string(rel_ev.reshape(-1, args.nstates).max(axis=0), precision=6))
    print("[overlap] trace diagnostic min/max =", float(O.min()), float(O.max()))
    print(f"[plot] APES: {apes_png}")
    print(f"[plot] APES 3D: {apes_3d_png}")
    print(f"[plot] APES fixed-theta surface: {apes_theta_surface_png}")
    print(f"[plot] overlap: {overlap_png}")
    print(f"[plot] overlap links: {links_png}")
    print(f"[plot] overlap fixed-theta links: {links_theta_png}")
    print(f"[plot] S1/S2 fixed-theta links: {s1s2_links_theta_png}")


if __name__ == "__main__":
    main()
