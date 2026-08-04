#!/usr/bin/env python3
"""Wavepacket dynamics for the Hahn--Stock retinal vibronic-coupling model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.retinal_dvr import RetinalDVRDynamics


def plot_trajectory(dynamics: RetinalDVRDynamics, path: Path) -> None:
    time = dynamics.times_fs
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(time, dynamics.diabatic_populations[:, 0], label="state 0")
    axes[0].plot(time, dynamics.diabatic_populations[:, 1], label="state 1")
    axes[0].set_ylabel("diabatic population")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False)

    axes[1].plot(time, dynamics.cis_populations.sum(axis=1), label="cis")
    axes[1].plot(time, dynamics.trans_populations.sum(axis=1), label="trans")
    axes[1].plot(
        time,
        dynamics.trans_populations[:, 1],
        "--",
        label=r"product $P_{\mathrm{trans}}^{(1)}$",
    )
    axes[1].set_ylabel("configuration population")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False)

    axes[2].plot(time, dynamics.cos_phi, label=r"$\langle\cos\phi\rangle$")
    axes[2].plot(time, dynamics.q_mean, label=r"$\langle q\rangle$")
    axes[2].set_xlabel("time / fs")
    axes[2].set_ylabel("coordinate moment")
    axes[2].legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nphi", type=int, default=301)
    parser.add_argument("--nq", type=int, default=32)
    parser.add_argument("--tmax-fs", type=float, default=300.0)
    parser.add_argument("--dt-fs", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/private/tmp/retinal_hahn_stock_dvr.npz"),
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    dynamics = RetinalDVRDynamics(nphi=args.nphi, nq=args.nq)
    ci_phi, ci_q = dynamics.model.conical_intersection()
    print(
        f"grid={args.nphi}x{args.nq}, CI=(phi={ci_phi:.6f}, q={ci_q:.1f}), "
        f"steps={round(args.tmax_fs / args.dt_fs)}"
    )
    dynamics.run(
        tmax_fs=args.tmax_fs,
        dt_fs=args.dt_fs,
        save_every=args.save_every,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **dynamics.as_dict())
    figure_path = args.out.with_suffix(".png")
    if not args.no_plot:
        plot_trajectory(dynamics, figure_path)

    total_trans = dynamics.trans_populations[-1].sum()
    product = dynamics.trans_populations[-1, 1]
    max_norm_error = np.max(np.abs(dynamics.norm - 1.0))
    print(f"final diabatic populations: {dynamics.diabatic_populations[-1]}")
    print(f"final total trans population: {total_trans:.8f}")
    print(f"final trans-state-1 product population: {product:.8f}")
    print(f"maximum norm error: {max_norm_error:.3e}")
    print(f"saved trajectory: {args.out}")
    if not args.no_plot:
        print(f"saved figure: {figure_path}")


if __name__ == "__main__":
    main()
