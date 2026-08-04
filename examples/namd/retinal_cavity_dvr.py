#!/usr/bin/env python3
"""Retinal photoisomerization in a quantized, optionally lossy cavity."""

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

from pyqed.models.retinal import RetinalHahnStock
from pyqed.namd.retinal_cavity_dvr import RetinalCavityDVRDynamics


def plot_result(dynamics: RetinalCavityDVRDynamics, path: Path) -> None:
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

    axes[1].plot(time, dynamics.trans_adiabatic[:, 0], label="trans lower")
    axes[1].plot(time, dynamics.trans_adiabatic[:, 1], label="trans upper")
    axes[1].plot(time, dynamics.trans_population, "--", label="total trans")
    axes[1].set_ylabel("product population")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False)

    axes[2].plot(time, dynamics.photon_number, label=r"$\langle n_c\rangle$")
    axes[2].plot(time, dynamics.mean_jump_count, label="leaked photons")
    axes[2].set_xlabel("time / fs")
    axes[2].set_ylabel("photons per trajectory")
    axes[2].legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cavity-ev", type=float, default=2.24)
    parser.add_argument("--g-ev", type=float, default=0.05)
    parser.add_argument("--inverse-inertia-ev", type=float, default=4.84e-4)
    parser.add_argument("--e1-ev", type=float, default=2.48)
    parser.add_argument("--w0-ev", type=float, default=3.6)
    parser.add_argument("--w1-ev", type=float, default=1.09)
    parser.add_argument("--omega-ev", type=float, default=0.19)
    parser.add_argument("--kappa-ev", type=float, default=0.10)
    parser.add_argument("--lambda-ev", type=float, default=0.19)
    parser.add_argument("--nphotons", type=int, default=3)
    parser.add_argument("--ntheta", type=int, default=201)
    parser.add_argument("--nq", type=int, default=20)
    parser.add_argument("--lifetime-fs", type=float)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tmax-fs", type=float, default=300.0)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/private/tmp/retinal_cavity_dvr.npz"),
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    model = RetinalHahnStock(
        inverse_inertia_ev=args.inverse_inertia_ev,
        e1_ev=args.e1_ev,
        w0_ev=args.w0_ev,
        w1_ev=args.w1_ev,
        omega_ev=args.omega_ev,
        kappa_ev=args.kappa_ev,
        lambda_ev=args.lambda_ev,
    )
    dynamics = RetinalCavityDVRDynamics(
        model,
        cavity_energy_ev=args.cavity_ev,
        coupling_ev=args.g_ev,
        nphotons=args.nphotons,
        ntheta=args.ntheta,
        nq=args.nq,
        cavity_lifetime_fs=args.lifetime_fs,
    )
    dynamics.run(
        tmax_fs=args.tmax_fs,
        dt_fs=args.dt_fs,
        save_every=args.save_every,
        trajectories=args.trajectories,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **dynamics.as_dict())
    figure_path = args.out.with_suffix(".png")
    if not args.no_plot:
        plot_result(dynamics, figure_path)

    peak = int(np.argmax(dynamics.trans_adiabatic[:, 0]))
    print(
        f"cavity={args.cavity_ev:.4f} eV, g={args.g_ev:.4f} eV, "
        f"lifetime={args.lifetime_fs}, trajectories={dynamics.trajectories}"
    )
    print(
        f"model: E1={args.e1_ev:.4f} eV, W0={args.w0_ev:.4f} eV, "
        f"W1={args.w1_ev:.4f} eV, lambda={args.lambda_ev:.4f} eV"
    )
    print(
        "peak lower-trans: "
        f"{dynamics.trans_adiabatic[peak, 0]:.8f} "
        f"at {dynamics.times_fs[peak]:.2f} fs"
    )
    print(f"final total trans: {dynamics.trans_population[-1]:.8f}")
    print(f"final lower-trans: {dynamics.trans_adiabatic[-1, 0]:.8f}")
    print(f"final upper-trans: {dynamics.trans_adiabatic[-1, 1]:.8f}")
    print(f"peak cavity photons: {dynamics.photon_number.max():.8f}")
    print(f"mean leaked photons: {dynamics.mean_jump_count[-1]:.8f}")
    print(f"saved trajectory: {args.out}")
    if not args.no_plot:
        print(f"saved figure: {figure_path}")


if __name__ == "__main__":
    main()
