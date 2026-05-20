#!/usr/bin/env python3
"""H3+ AM1/MECI nearest-neighbor links on a fixed-r2 r1-theta slice."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from h3plus_am1_meci_link_scan import (  # noqa: E402
    HARTREE_TO_EV,
    h3plus_body_frame,
    load_links,
    plot_link_histograms,
    scan_or_load,
    summarize,
    write_link_table,
)
from pyqed.namd.triatomic import Triatom  # noqa: E402


def build_solver(args):
    r2_min = args.r2 - 0.5 * args.r2_width
    r2_max = args.r2 + 0.5 * args.r2_width
    theta0 = np.deg2rad(0.5 * (args.theta_min_deg + args.theta_max_deg))
    solver = Triatom(
        h3plus_body_frame(r=args.r2, theta=theta0),
        nstates=args.nstates,
        charge=1,
        spin=0,
        unit="bohr",
        dvr_type=["sine", "sine", "legendre"],
    )
    solver.set_dvr(
        domains=[
            [args.r1_min, args.r1_max],
            [r2_min, r2_max],
            [np.deg2rad(args.theta_min_deg), np.deg2rad(args.theta_max_deg)],
        ],
        npts=[args.n_r1, 1, args.n_theta],
        dvr_type=["sine", "sine", "legendre"],
    )
    return solver


def fixed_r2_sign_gauge(solver, r2_index=0):
    """State-wise +/- signs aligned on the fixed-r2 r1-theta graph."""
    signs = np.zeros((solver.nx[0], solver.nx[2], solver.nstates), dtype=int)
    signs[0, 0, :] = 1

    def edge_sign(idx, nxt, state):
        axis = 0 if nxt[0] != idx[0] else 2
        src = idx if nxt[axis] > idx[axis] else nxt
        mat = solver.overlap_links[(axis, src)]
        val = float(np.real(mat[state, state]))
        return 1 if val >= 0.0 else -1

    for state in range(solver.nstates):
        queue = [(0, 0)]
        while queue:
            i, k = queue.pop(0)
            idx = (i, r2_index, k)
            for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nk = i + di, k + dk
                if ni < 0 or nk < 0 or ni >= solver.nx[0] or nk >= solver.nx[2]:
                    continue
                if signs[ni, nk, state] != 0:
                    continue
                nxt = (ni, r2_index, nk)
                signs[ni, nk, state] = signs[i, k, state] * edge_sign(idx, nxt, state)
                queue.append((ni, nk))

    return signs


def aligned_link_value(mat, idx, nxt, a, b, signs=None):
    val = float(np.real(mat[a, b]))
    if signs is None:
        return val
    return val * int(signs[idx[0], idx[2], a]) * int(signs[nxt[0], nxt[2], b])


def write_gauge_signs(solver, signs, outpath, r2_index=0):
    with outpath.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["i_r1", "i_theta", "r2", "theta_deg", *[f"sigma_S{s}" for s in range(solver.nstates)]])
        for i, _ in enumerate(solver.x[0]):
            for k, theta in enumerate(solver.x[2]):
                writer.writerow([
                    i,
                    k,
                    float(solver.x[1][r2_index]),
                    float(np.rad2deg(theta)),
                    *[int(signs[i, k, s]) for s in range(solver.nstates)],
                ])


def plot_apes_r1theta(solver, outpath, r2_index=0):
    apes_ev = (solver.apes - np.min(solver.apes[..., 0])) * HARTREE_TO_EV
    theta_extent = [np.rad2deg(solver.x[2][0]), np.rad2deg(solver.x[2][-1])]
    r1_extent = [solver.x[0][0], solver.x[0][-1]]
    r2 = solver.x[1][r2_index]

    fig, axes = plt.subplots(
        1,
        solver.nstates,
        figsize=(4.0 * solver.nstates, 3.4),
        constrained_layout=True,
    )
    for state, ax in enumerate(np.atleast_1d(axes)):
        im = ax.imshow(
            apes_ev[:, r2_index, :, state],
            origin="lower",
            extent=[*theta_extent, *r1_extent],
            aspect="auto",
            cmap="magma",
        )
        ax.set_title(f"S{state}")
        ax.set_xlabel("theta / deg")
        ax.set_ylabel("r1 / bohr")
        fig.colorbar(im, ax=ax, label="E - min(S0) / eV")
    fig.suptitle(f"H3+ AM1/MECI APES, r2={r2:.3f} bohr")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_state_pair_links_r1theta(
    solver,
    outpath,
    r2_index=0,
    signs=None,
    title_suffix="",
):
    pairs = [(a, b) for a in range(solver.nstates) for b in range(a, solver.nstates)]
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    r2 = solver.x[1][r2_index]

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.7 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (a, b) in zip(axes.ravel(), pairs):
        for i, r1 in enumerate(solver.x[0]):
            for k, theta in enumerate(np.rad2deg(solver.x[2])):
                ax.scatter(theta, r1, s=42, color="0.12", zorder=3)
                idx = (i, r2_index, k)
                for axis in (0, 2):
                    if idx[axis] + 1 >= solver.nx[axis]:
                        continue
                    mat = solver.overlap_links[(axis, idx)]
                    nxt = list(idx)
                    nxt[axis] += 1
                    nxt = tuple(nxt)
                    val = aligned_link_value(mat, idx, nxt, a, b, signs=signs)
                    x0, y0 = np.rad2deg(solver.x[2][idx[2]]), solver.x[0][idx[0]]
                    x1, y1 = np.rad2deg(solver.x[2][nxt[2]]), solver.x[0][nxt[0]]
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
        ax.set_xlabel("theta / deg")
        ax.set_ylabel("r1 / bohr")
        ax.grid(True, color="0.9", linewidth=0.8)

    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Re nearest-neighbor link; linewidth = |link|")
    fig.suptitle(f"H3+ AM1/MECI r1-theta links, r2={r2:.3f} bohr{title_suffix}")
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def summarize_aligned_links(solver, signs, r2_index=0):
    for state in range(solver.nstates):
        vals = []
        negatives = 0
        for (axis, idx), mat in solver.overlap_links.items():
            if idx[1] != r2_index or axis not in (0, 2):
                continue
            nxt = list(idx)
            nxt[axis] += 1
            nxt = tuple(nxt)
            val = aligned_link_value(mat, idx, nxt, state, state, signs=signs)
            vals.append(val)
            negatives += int(val < 0.0)
        vals = np.asarray(vals)
        print(
            f"[gauge] aligned S{state}/S{state} negative links = "
            f"{negatives}/{len(vals)}, min/max = {vals.min(): .6f} {vals.max(): .6f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r1", type=int, default=9)
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument("--r1-min", type=float, default=1.25)
    parser.add_argument("--r1-max", type=float, default=2.25)
    parser.add_argument("--r2", type=float, default=1.60)
    parser.add_argument("--r2-width", type=float, default=1.0e-6)
    parser.add_argument("--theta-min-deg", type=float, default=45.0)
    parser.add_argument("--theta-max-deg", type=float, default=75.0)
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
        default=Path(__file__).with_name("h3plus_am1_meci_r1theta_link_scan_r2_1p60"),
    )
    args = parser.parse_args()

    solver = build_solver(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    scan_or_load(solver, args)

    table = args.outdir / "h3plus_am1_meci_r1theta_links.csv"
    signs_csv = args.outdir / "h3plus_am1_meci_r1theta_gauge_signs.csv"
    apes_png = args.outdir / "h3plus_am1_meci_r1theta_apes.png"
    links_png = args.outdir / "h3plus_am1_meci_r1theta_links.png"
    aligned_png = args.outdir / "h3plus_am1_meci_r1theta_links_gauge_aligned.png"
    hist_png = args.outdir / "h3plus_am1_meci_r1theta_link_histograms.png"

    signs = fixed_r2_sign_gauge(solver, r2_index=0)
    write_link_table(solver, table)
    write_gauge_signs(solver, signs, signs_csv, r2_index=0)
    plot_apes_r1theta(solver, apes_png, r2_index=0)
    plot_state_pair_links_r1theta(solver, links_png, r2_index=0)
    plot_state_pair_links_r1theta(
        solver,
        aligned_png,
        r2_index=0,
        signs=signs,
        title_suffix=" (state-wise sign gauge)",
    )
    plot_link_histograms(solver, hist_png)
    summarize(solver)
    summarize_aligned_links(solver, signs, r2_index=0)
    print(f"[data] link table: {table}")
    print(f"[data] gauge signs: {signs_csv}")
    print(f"[plot] APES r1-theta: {apes_png}")
    print(f"[plot] raw r1-theta links: {links_png}")
    print(f"[plot] gauge-aligned r1-theta links: {aligned_png}")
    print(f"[plot] link histograms: {hist_png}")


if __name__ == "__main__":
    main()
