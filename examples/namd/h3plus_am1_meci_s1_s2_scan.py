#!/usr/bin/env python3
"""Coarse fixed-angle H3+ AM1/MECI S1/S2 scan.

The geometry is parameterized in Jacobi-like internal coordinates:

    H2 = (0, 0, 0)
    H1 = (r1, 0, 0)
    H3 = (r2 cos(theta), r2 sin(theta), 0)

Distances are in bohr.  The script stores the raw energies and writes a
3D surface plot for S1/S2 plus a contour plot for the S2-S1 gap.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message="AM1 model is under testing")

from pyqed.qchem import Molecule
from pyqed.qchem.semiempirical.am1 import HARTREE2EV, RAM1


def h3plus_atom_string(r1: float, r2: float, theta: float) -> str:
    """Return H3+ body-frame geometry for the fixed-angle scan."""
    return (
        f"H {r1:.12f} 0.0 0.0; "
        "H 0.0 0.0 0.0; "
        f"H {r2 * np.cos(theta):.12f} {r2 * np.sin(theta):.12f} 0.0"
    )


def run_meci_point(r1: float, r2: float, theta: float, args) -> tuple[np.ndarray, float, int]:
    mol = Molecule(
        atom=h3plus_atom_string(r1, r2, theta),
        charge=1,
        spin=0,
        unit="bohr",
    )
    mf = RAM1(mol).run(
        conv_tol=args.scf_tol,
        max_cycle=args.max_cycle,
        verbose=0,
        damping=args.damping,
    )
    meci = mf.MECI(nstates=args.nstates, ncas=args.ncas).run()
    if len(meci.e) < args.nstates:
        raise RuntimeError(
            f"MECI returned {len(meci.e)} states, fewer than requested {args.nstates}."
        )
    return np.asarray(meci.e[: args.nstates], dtype=float), float(mf.e_tot), int(mf.cycles)


def scan_grid(args):
    theta = np.deg2rad(args.theta_deg)
    r1_grid = np.linspace(args.r_min, args.r_max, args.npts)
    r2_grid = np.linspace(args.r_min, args.r_max, args.npts)
    energies = np.full((args.npts, args.npts, args.nstates), np.nan)
    scf_energies = np.full((args.npts, args.npts), np.nan)
    scf_cycles = np.zeros((args.npts, args.npts), dtype=int)

    total = args.npts * args.npts
    count = 0
    for i, r1 in enumerate(r1_grid):
        for j, r2 in enumerate(r2_grid):
            count += 1
            try:
                energies[i, j], scf_energies[i, j], scf_cycles[i, j] = run_meci_point(
                    float(r1),
                    float(r2),
                    theta,
                    args,
                )
                print(
                    f"[{count:3d}/{total}] r1={r1:.4f} r2={r2:.4f} "
                    f"S1={energies[i, j, 1]: .10f} Eh "
                    f"S2={energies[i, j, 2]: .10f} Eh "
                    f"gap={(energies[i, j, 2] - energies[i, j, 1]) * HARTREE2EV: .5f} eV"
                )
            except Exception as exc:
                print(f"[{count:3d}/{total}] r1={r1:.4f} r2={r2:.4f} failed: {exc}")

    return r1_grid, r2_grid, theta, energies, scf_energies, scf_cycles


def plot_surfaces(r1_grid, r2_grid, theta, energies, outpath):
    r1, r2 = np.meshgrid(r1_grid, r2_grid, indexing="ij")
    rel_ev = (energies - np.nanmin(energies[:, :, 0])) * HARTREE2EV
    theta_deg = np.rad2deg(theta)

    fig = plt.figure(figsize=(10.5, 4.7), constrained_layout=True)
    for panel, state in enumerate((1, 2), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        z = rel_ev[:, :, state]
        surf = ax.plot_surface(
            r1,
            r2,
            z,
            cmap="viridis" if state == 1 else "magma",
            edgecolor="k",
            linewidth=0.25,
            antialiased=True,
            alpha=0.96,
        )
        ax.scatter(r1, r2, z, color="black", s=13, depthshade=False)
        ax.set_title(f"S{state} at theta={theta_deg:.1f} deg")
        ax.set_xlabel("r1 / bohr")
        ax.set_ylabel("r2 / bohr")
        ax.set_zlabel("E - min(S0) / eV")
        ax.view_init(elev=28, azim=-132)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.66, pad=0.08)
        cbar.set_label("eV")

    fig.suptitle("H3+ AM1/MECI fixed-angle excited-state surfaces")
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def plot_gap(r1_grid, r2_grid, theta, energies, outpath):
    gap_ev = (energies[:, :, 2] - energies[:, :, 1]) * HARTREE2EV
    theta_deg = np.rad2deg(theta)

    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    levels = 16
    im = ax.contourf(r2_grid, r1_grid, gap_ev, levels=levels, cmap="cividis")
    cs = ax.contour(r2_grid, r1_grid, gap_ev, levels=levels, colors="black", linewidths=0.35)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    ax.plot(r2_grid, r2_grid, color="white", linestyle="--", linewidth=1.4, label="r1 = r2")
    ax.set_title(f"H3+ S2-S1 gap at theta={theta_deg:.1f} deg")
    ax.set_xlabel("r2 / bohr")
    ax.set_ylabel("r1 / bohr")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("S2 - S1 / eV")
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def plot_combined_surfaces(r1_grid, r2_grid, theta, energies, outpath):
    r1, r2 = np.meshgrid(r1_grid, r2_grid, indexing="ij")
    rel_ev = (energies - np.nanmin(energies[:, :, 0])) * HARTREE2EV
    theta_deg = np.rad2deg(theta)

    fig = plt.figure(figsize=(7.4, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    s1 = rel_ev[:, :, 1]
    s2 = rel_ev[:, :, 2]
    surf1 = ax.plot_surface(
        r1,
        r2,
        s1,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.22,
        antialiased=True,
        alpha=0.76,
    )
    surf2 = ax.plot_surface(
        r1,
        r2,
        s2,
        cmap="magma",
        edgecolor="k",
        linewidth=0.22,
        antialiased=True,
        alpha=0.56,
    )
    ax.scatter(r1, r2, s1, color="black", s=11, depthshade=False)
    ax.scatter(r1, r2, s2, color="white", edgecolor="black", linewidth=0.35, s=13, depthshade=False)
    ax.set_title(f"H3+ S1/S2 together at theta={theta_deg:.1f} deg")
    ax.set_xlabel("r1 / bohr")
    ax.set_ylabel("r2 / bohr")
    ax.set_zlabel("E - min(S0) / eV")
    ax.view_init(elev=27, azim=-133)

    cbar1 = fig.colorbar(surf1, ax=ax, shrink=0.55, pad=0.02, location="left")
    cbar1.set_label("S1 / eV")
    cbar2 = fig.colorbar(surf2, ax=ax, shrink=0.55, pad=0.08, location="right")
    cbar2.set_label("S2 / eV")
    fig.savefig(outpath, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=9)
    parser.add_argument("--r-min", type=float, default=1.20)
    parser.add_argument("--r-max", type=float, default=2.40)
    parser.add_argument("--theta-deg", type=float, default=60.0)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_am1_meci_s1_s2_scan"),
    )
    args = parser.parse_args()
    if args.nstates < 3:
        raise ValueError("Need at least three states to plot S1 and S2.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    data_path = args.outdir / (
        f"h3plus_am1_meci_s1_s2_theta{args.theta_deg:g}_"
        f"r{args.r_min:g}_{args.r_max:g}_n{args.npts}.npz"
    )
    surface_path = args.outdir / "h3plus_am1_meci_s1_s2_surfaces.png"
    combined_path = args.outdir / "h3plus_am1_meci_s1_s2_together.png"
    gap_path = args.outdir / "h3plus_am1_meci_s2_s1_gap.png"

    if data_path.exists() and not args.force:
        data = np.load(data_path)
        r1_grid = data["r1"]
        r2_grid = data["r2"]
        theta = float(data["theta"])
        energies = data["energies"]
        scf_energies = data["scf_energies"]
        scf_cycles = data["scf_cycles"]
        print(f"[cache] loaded {data_path}")
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="AM1 model is under testing")
            r1_grid, r2_grid, theta, energies, scf_energies, scf_cycles = scan_grid(args)
        np.savez(
            data_path,
            r1=r1_grid,
            r2=r2_grid,
            theta=theta,
            energies=energies,
            scf_energies=scf_energies,
            scf_cycles=scf_cycles,
        )
        print(f"[cache] saved {data_path}")

    plot_surfaces(r1_grid, r2_grid, theta, energies, surface_path)
    plot_combined_surfaces(r1_grid, r2_grid, theta, energies, combined_path)
    plot_gap(r1_grid, r2_grid, theta, energies, gap_path)

    gap_ev = (energies[:, :, 2] - energies[:, :, 1]) * HARTREE2EV
    min_gap_idx = np.unravel_index(np.nanargmin(gap_ev), gap_ev.shape)
    print("[grid] r1 =", np.array2string(r1_grid, precision=6))
    print("[grid] r2 =", np.array2string(r2_grid, precision=6))
    print(f"[grid] theta = {np.rad2deg(theta):.6f} deg")
    print("[energy] min S0/S1/S2 Eh =", np.array2string(np.nanmin(energies, axis=(0, 1)), precision=10))
    print("[energy] max S0/S1/S2 Eh =", np.array2string(np.nanmax(energies, axis=(0, 1)), precision=10))
    print(
        "[gap] min S2-S1 = "
        f"{gap_ev[min_gap_idx]:.8f} eV at "
        f"r1={r1_grid[min_gap_idx[0]]:.6f}, r2={r2_grid[min_gap_idx[1]]:.6f}"
    )
    print(f"[plot] surfaces: {surface_path}")
    print(f"[plot] S1/S2 together: {combined_path}")
    print(f"[plot] gap: {gap_path}")
    print(f"[data] {data_path}")


if __name__ == "__main__":
    main()
