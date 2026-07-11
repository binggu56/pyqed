#!/usr/bin/env python3
"""H3+ LDR setup using native AM1/MECI electronic states.

This example scans a small body-fixed H3+ DVR grid with
``Triatom.scan_pes(electronic_method="am1/meci")`` and optionally propagates a
Gaussian wavepacket initialized on S2.
"""

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
from pyqed.units import au2fs


def h3plus_body_frame(r: float = 1.65, theta: float = np.pi / 3.0):
    return [
        ["H", (float(r), 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


def initial_packet(solver: Triatom, state: int = 2, width: float = 80.0):
    psi_values = np.zeros((*solver.nx, solver.nstates), dtype=complex)
    center = np.array([axis[len(axis) // 2] for axis in solver.x])
    for idx in np.ndindex(*solver.nx):
        q = np.array([solver.x[axis][idx[axis]] for axis in range(solver.ndim)])
        psi_values[idx + (state,)] = np.exp(-width * np.sum((q - center) ** 2))

    psi = solver.to_quadrature_normalized(psi_values)
    norm = solver.norm(psi)
    if norm == 0:
        raise RuntimeError("Initial wavepacket norm is zero.")
    return psi / norm


def plot_populations(solver: Triatom, result, outpath: Path):
    pops = solver.get_population(result, plot=False)
    times_fs = np.asarray(result["times"]) * au2fs

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    for state in range(solver.nstates):
        ax.plot(times_fs, pops[:, state], marker="o", label=f"S{state}")
    ax.set_xlabel("time / fs")
    ax.set_ylabel("population")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("H3+ AM1/MECI-LDR populations")
    ax.legend()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r", type=int, default=3)
    parser.add_argument("--n-theta", type=int, default=3)
    parser.add_argument("--r-min", type=float, default=1.45)
    parser.add_argument("--r-max", type=float, default=1.85)
    parser.add_argument("--theta-min-deg", type=float, default=55.0)
    parser.add_argument("--theta-max-deg", type=float, default=65.0)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--overlap-method", choices=("linked", "link-only", "full"), default="link-only")
    parser.add_argument("--propagate", action="store_true")
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--nt", type=int, default=10)
    parser.add_argument("--nout", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).with_name("h3plus_am1_meci_ldr"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    theta0 = np.deg2rad(0.5 * (args.theta_min_deg + args.theta_max_deg))
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
            [np.deg2rad(args.theta_min_deg), np.deg2rad(args.theta_max_deg)],
        ],
        npts=[args.n_r, args.n_r, args.n_theta],
        dvr_type=["sine", "sine", "legendre"],
    )

    apes, overlap_data, _ = solver.scan_pes(
        electronic_method="am1/meci",
        nstates=args.nstates,
        ncas=3,
        nelecas=2,
        overlap_method=args.overlap_method,
        n_workers=args.n_workers,
        worker_threads=1,
    )
    print("[apes] shape =", apes.shape)
    print("[apes] min energies/Eh =", np.array2string(np.nanmin(apes, axis=(0, 1, 2)), precision=10))
    if isinstance(overlap_data, dict):
        print("[overlap] nearest-neighbor links =", len(overlap_data))
    else:
        print("[overlap] matrix shape =", overlap_data.shape)

    if not args.propagate:
        print("[done] Add --propagate to run a short LDR wavepacket propagation.")
        return

    psi0 = initial_packet(solver, state=min(2, args.nstates - 1))
    result = solver.run(
        psi0,
        dt=args.dt_fs / au2fs,
        nt=args.nt,
        nout=args.nout,
        kinetic_propagator="expm_multiply",
        kinetic_action="matrix-free",
    )
    pop_png = args.outdir / "h3plus_am1_meci_ldr_populations.png"
    plot_populations(solver, result, pop_png)
    print(f"[plot] populations: {pop_png}")


if __name__ == "__main__":
    main()
