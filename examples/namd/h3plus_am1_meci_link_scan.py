#!/usr/bin/env python3
"""Coarse H3+ AM1/MECI scan and nearest-neighbor LDR-link plots."""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.units import au2ev
from pyqed.namd.triatomic import Triatom

HARTREE_TO_EV = au2ev


def h3plus_body_frame(r=1.65, theta=np.pi / 3.0):
    return [
        ["H", (float(r), 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


@contextlib.contextmanager
def pushd(path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def build_solver(args):
    if args.theta_deg is None:
        theta_min_deg = args.theta_min_deg
        theta_max_deg = args.theta_max_deg
        n_theta = args.n_theta
    else:
        theta_min_deg = args.theta_deg - 0.5 * args.theta_width_deg
        theta_max_deg = args.theta_deg + 0.5 * args.theta_width_deg
        n_theta = 1

    theta0 = np.deg2rad(0.5 * (theta_min_deg + theta_max_deg))
    solver = Triatom(
        h3plus_body_frame(theta=theta0),
        nstates=args.nstates,
        charge=1,
        spin=0,
        unit="bohr",
        dvr_type=["sine", "sine", "legendre"],
    )
    solver.set_dvr(
        domains=[
            [args.r_min, args.r_max],
            [args.r_min, args.r_max],
            [np.deg2rad(theta_min_deg), np.deg2rad(theta_max_deg)],
        ],
        npts=[args.n_r, args.n_r, n_theta],
        dvr_type=["sine", "sine", "legendre"],
    )
    return solver


def load_links(path):
    data = np.load(path, allow_pickle=True)
    return {
        (int(axis), tuple(int(i) for i in idx)): np.asarray(mat, dtype=complex)
        for axis, idx, mat in zip(data["axes"], data["indices"], data["data"])
    }


def scan_or_load(solver, args):
    apes_path = args.outdir / "apes.npz"
    links_path = args.outdir / "overlap_links.npz"
    if args.reuse_cache and apes_path.exists() and links_path.exists():
        solver.apes = np.load(apes_path, allow_pickle=True)["data"]
        solver.overlap_links = load_links(links_path)
        print(f"[cache] loaded {apes_path}")
        print(f"[cache] loaded {links_path}")
        return

    with pushd(args.outdir):
        solver.scan_pes(
            electronic_method="am1/meci",
            nstates=args.nstates,
            ncas=args.ncas,
            nelecas=args.nelecas,
            overlap_method="link-only",
            unitarize_overlap_links=args.unitarize,
            n_workers=args.n_workers,
            worker_threads=args.worker_threads,
            scf_tol=args.scf_tol,
            max_cycle=args.max_cycle,
            damping=args.damping,
        )


def edge_coordinates(solver, axis, idx):
    start = np.array([solver.x[k][idx[k]] for k in range(solver.ndim)], dtype=float)
    nxt = list(idx)
    nxt[axis] += 1
    nxt = tuple(nxt)
    end = np.array([solver.x[k][nxt[k]] for k in range(solver.ndim)], dtype=float)
    return start, end


def write_link_table(solver, outpath):
    with outpath.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "axis",
            "i_r1",
            "i_r2",
            "i_theta",
            "r1",
            "r2",
            "theta_deg",
            "r1_next",
            "r2_next",
            "theta_next_deg",
        ]
        for a in range(solver.nstates):
            for b in range(solver.nstates):
                header += [f"re_S{a}_S{b}", f"abs_S{a}_S{b}"]
        writer.writerow(header)

        for (axis, idx), mat in sorted(solver.overlap_links.items()):
            start, end = edge_coordinates(solver, axis, idx)
            row = [
                axis,
                *idx,
                start[0],
                start[1],
                np.rad2deg(start[2]),
                end[0],
                end[1],
                np.rad2deg(end[2]),
            ]
            for a in range(solver.nstates):
                for b in range(solver.nstates):
                    row += [float(np.real(mat[a, b])), float(abs(mat[a, b]))]
            writer.writerow(row)


def fixed_theta_sign_gauge(solver, theta_index=None):
    """Return state-wise +/- signs that align intrastate links on a 2D slice."""
    if theta_index is None:
        theta_index = len(solver.x[2]) // 2

    signs = np.zeros((solver.nx[0], solver.nx[1], solver.nstates), dtype=int)
    signs[0, 0, :] = 1

    def edge_sign(idx, nxt, state):
        axis = 0 if nxt[0] != idx[0] else 1
        src = idx if nxt[axis] > idx[axis] else nxt
        mat = solver.overlap_links[(axis, src)]
        val = float(np.real(mat[state, state]))
        return 1 if val >= 0.0 else -1

    for state in range(solver.nstates):
        queue = [(0, 0)]
        while queue:
            i, j = queue.pop(0)
            idx = (i, j, theta_index)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if ni < 0 or nj < 0 or ni >= solver.nx[0] or nj >= solver.nx[1]:
                    continue
                if signs[ni, nj, state] != 0:
                    continue
                nxt = (ni, nj, theta_index)
                signs[ni, nj, state] = signs[i, j, state] * edge_sign(idx, nxt, state)
                queue.append((ni, nj))

    return signs


def write_gauge_signs(solver, signs, outpath, theta_index=None):
    if theta_index is None:
        theta_index = len(solver.x[2]) // 2
    with outpath.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["i_r1", "i_r2", "theta_deg", *[f"sigma_S{s}" for s in range(solver.nstates)]])
        for i, _ in enumerate(solver.x[0]):
            for j, _ in enumerate(solver.x[1]):
                writer.writerow([
                    i,
                    j,
                    float(np.rad2deg(solver.x[2][theta_index])),
                    *[int(signs[i, j, s]) for s in range(solver.nstates)],
                ])


def aligned_link_value(solver, mat, idx, nxt, a, b, signs=None):
    val = float(np.real(mat[a, b]))
    if signs is None:
        return val
    return val * int(signs[idx[0], idx[1], a]) * int(signs[nxt[0], nxt[1], b])


def plot_apes_fixed_theta(solver, outpath, theta_index=None):
    if theta_index is None:
        theta_index = len(solver.x[2]) // 2
    theta_deg = np.rad2deg(solver.x[2][theta_index])
    apes_ev = (solver.apes - np.min(solver.apes[..., 0])) * HARTREE_TO_EV
    r2_extent = [solver.x[1][0], solver.x[1][-1]]
    r1_extent = [solver.x[0][0], solver.x[0][-1]]

    fig, axes = plt.subplots(
        1,
        solver.nstates,
        figsize=(4.0 * solver.nstates, 3.4),
        constrained_layout=True,
    )
    for state, ax in enumerate(np.atleast_1d(axes)):
        im = ax.imshow(
            apes_ev[:, :, theta_index, state],
            origin="lower",
            extent=[*r2_extent, *r1_extent],
            aspect="auto",
            cmap="magma",
        )
        ax.set_title(f"S{state}")
        ax.set_xlabel("r2 / bohr")
        ax.set_ylabel("r1 / bohr")
        fig.colorbar(im, ax=ax, label="E - min(S0) / eV")
    fig.suptitle(f"H3+ AM1/MECI APES, theta={theta_deg:.2f} deg")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_state_pair_links_fixed_theta(
    solver,
    outpath,
    theta_index=None,
    signs=None,
    title_suffix="",
):
    if theta_index is None:
        theta_index = len(solver.x[2]) // 2
    theta_deg = np.rad2deg(solver.x[2][theta_index])
    pairs = [(a, b) for a in range(solver.nstates) for b in range(a, solver.nstates)]
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.7 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (a, b) in zip(axes.ravel(), pairs):
        for i, r1 in enumerate(solver.x[0]):
            for j, r2 in enumerate(solver.x[1]):
                ax.scatter(r2, r1, s=42, color="0.12", zorder=3)
                idx = (i, j, theta_index)
                for axis in (0, 1):
                    if idx[axis] + 1 >= solver.nx[axis]:
                        continue
                    mat = solver.overlap_links[(axis, idx)]
                    nxt = list(idx)
                    nxt[axis] += 1
                    nxt = tuple(nxt)
                    val = aligned_link_value(solver, mat, idx, nxt, a, b, signs=signs)
                    x0, y0 = solver.x[1][idx[1]], solver.x[0][idx[0]]
                    x1, y1 = solver.x[1][nxt[1]], solver.x[0][nxt[0]]
                    ax.plot(
                        [x0, x1],
                        [y0, y1],
                        color=cmap(norm(val)),
                        linewidth=0.5 + 6.0 * abs(val),
                        alpha=0.92,
                        solid_capstyle="round",
                    )
        kind = "intrastate" if a == b else "interstate"
        ax.set_title(f"S{a}/S{b} {kind}")
        ax.set_xlabel("r2 / bohr")
        ax.set_ylabel("r1 / bohr")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.9", linewidth=0.8)

    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Re nearest-neighbor link; linewidth = |link|")
    fig.suptitle(f"H3+ AM1/MECI links at theta={theta_deg:.2f} deg{title_suffix}")
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def summarize_aligned_links(solver, signs, theta_index=None):
    if theta_index is None:
        theta_index = len(solver.x[2]) // 2
    for state in range(solver.nstates):
        vals = []
        negatives = 0
        for (axis, idx), mat in solver.overlap_links.items():
            if idx[2] != theta_index or axis not in (0, 1):
                continue
            nxt = list(idx)
            nxt[axis] += 1
            nxt = tuple(nxt)
            val = aligned_link_value(solver, mat, idx, nxt, state, state, signs=signs)
            vals.append(val)
            negatives += int(val < 0.0)
        vals = np.asarray(vals)
        print(
            f"[gauge] aligned S{state}/S{state} negative links = "
            f"{negatives}/{len(vals)}, min/max = {vals.min(): .6f} {vals.max(): .6f}"
        )


def plot_link_histograms(solver, outpath):
    pairs = [(a, b) for a in range(solver.nstates) for b in range(a, solver.nstates)]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.0), constrained_layout=True)
    for a, b in pairs:
        vals = np.array([np.real(mat[a, b]) for mat in solver.overlap_links.values()], dtype=float)
        label = f"S{a}/S{b}"
        axes[0].hist(vals, bins=18, histtype="step", linewidth=1.7, label=label)
        axes[1].hist(np.abs(vals), bins=18, histtype="step", linewidth=1.7, label=label)
    axes[0].set_xlabel("Re link")
    axes[0].set_ylabel("count")
    axes[1].set_xlabel("|Re link|")
    axes[1].set_ylabel("count")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title("Signed nearest-neighbor links")
    axes[1].set_title("Absolute nearest-neighbor links")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def summarize(solver):
    print("[grid] nx =", solver.nx)
    print("[grid] r1 =", np.array2string(solver.x[0], precision=6))
    print("[grid] r2 =", np.array2string(solver.x[1], precision=6))
    print("[grid] theta/deg =", np.array2string(np.rad2deg(solver.x[2]), precision=6))
    print("[apes] min energies/Eh by state =", np.array2string(solver.apes.reshape(-1, solver.nstates).min(axis=0), precision=10))
    print("[links] number of nearest-neighbor links =", len(solver.overlap_links))
    for a in range(solver.nstates):
        vals = np.array([np.real(mat[a, a]) for mat in solver.overlap_links.values()])
        print(f"[links] S{a}/S{a} Re min/max/mean = {vals.min(): .6f} {vals.max(): .6f} {vals.mean(): .6f}")
    for a in range(solver.nstates):
        for b in range(a + 1, solver.nstates):
            vals = np.array([np.real(mat[a, b]) for mat in solver.overlap_links.values()])
            print(f"[links] S{a}/S{b} Re min/max/mean = {vals.min(): .6f} {vals.max(): .6f} {vals.mean(): .6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r", type=int, default=3)
    parser.add_argument("--n-theta", type=int, default=3)
    parser.add_argument("--r-min", type=float, default=1.35)
    parser.add_argument("--r-max", type=float, default=1.85)
    parser.add_argument("--theta-min-deg", type=float, default=50.0)
    parser.add_argument("--theta-max-deg", type=float, default=80.0)
    parser.add_argument("--theta-deg", type=float, default=None)
    parser.add_argument("--theta-width-deg", type=float, default=1.0e-6)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--nelecas", type=int, default=2)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--unitarize", action="store_true")
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_am1_meci_link_scan"),
    )
    args = parser.parse_args()

    solver = build_solver(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    scan_or_load(solver, args)

    table = args.outdir / "h3plus_am1_meci_links.csv"
    signs_csv = args.outdir / "h3plus_am1_meci_gauge_signs.csv"
    apes_png = args.outdir / "h3plus_am1_meci_apes_fixed_theta.png"
    links_png = args.outdir / "h3plus_am1_meci_links_fixed_theta.png"
    aligned_links_png = args.outdir / "h3plus_am1_meci_links_gauge_aligned_fixed_theta.png"
    hist_png = args.outdir / "h3plus_am1_meci_link_histograms.png"
    theta_index = len(solver.x[2]) // 2
    signs = fixed_theta_sign_gauge(solver, theta_index=theta_index)

    write_link_table(solver, table)
    write_gauge_signs(solver, signs, signs_csv, theta_index=theta_index)
    plot_apes_fixed_theta(solver, apes_png, theta_index=theta_index)
    plot_state_pair_links_fixed_theta(solver, links_png, theta_index=theta_index)
    plot_state_pair_links_fixed_theta(
        solver,
        aligned_links_png,
        theta_index=theta_index,
        signs=signs,
        title_suffix=" (state-wise sign gauge)",
    )
    plot_link_histograms(solver, hist_png)
    summarize(solver)
    summarize_aligned_links(solver, signs, theta_index=theta_index)
    print(f"[data] link table: {table}")
    print(f"[data] gauge signs: {signs_csv}")
    print(f"[plot] APES fixed theta: {apes_png}")
    print(f"[plot] intrastate/interstate links: {links_png}")
    print(f"[plot] gauge-aligned links: {aligned_links_png}")
    print(f"[plot] link histograms: {hist_png}")


if __name__ == "__main__":
    main()
