"""Thermal two-molecule test of the HS cavity relay mechanism."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.namd.retinal_hs_cavity import (
    RetinalHSTwoMoleculeCavityDynamics,
)


def run(args, coupling_ev: float):
    return RetinalHSTwoMoleculeCavityDynamics(
        nmolecules=2,
        nphi=args.nphi,
        nphotons=args.nphotons,
        cavity_energy_ev=args.cavity_energy,
        coupling_ev=coupling_ev,
    ).run_thermal_ensemble(
        temperature_k=args.temperature,
        samples=args.samples,
        seed=args.seed,
        tmax_fs=args.tmax,
        dt_fs=args.dt,
        save_every=args.save_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cavity-energy", type=float, default=0.173)
    parser.add_argument("--coupling", type=float, default=0.02)
    parser.add_argument("--nphotons", type=int, default=4)
    parser.add_argument("--nphi", type=int, default=51)
    parser.add_argument("--tmax", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hs-thermal-two-molecule-cavity"),
    )
    args = parser.parse_args()

    bare = run(args, 0.0)
    cavity = run(args, args.coupling)
    data: dict[str, np.ndarray] = {}
    for label, dynamics in (("bare", bare), ("cavity", cavity)):
        for name, values in dynamics.as_dict().items():
            if name != "state":
                data[f"{label}__{name}"] = values
    np.savez(args.output.with_suffix(".npz"), **data)

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), sharex=True)
    for label, dynamics, color in (
        ("bare", bare, "0.35"),
        ("shared cavity", cavity, "#D65F5F"),
    ):
        time = dynamics.times_fs
        receiver_a = dynamics.electronic_populations[:, :, 0].mean(axis=1)
        receiver_a_error = np.sqrt(
            np.sum(dynamics.electronic_populations_stderr[:, :, 0] ** 2, axis=1)
        ) / 2.0
        axes[0, 0].plot(time, receiver_a, color=color, label=label)
        axes[0, 0].fill_between(
            time,
            receiver_a - receiver_a_error,
            receiver_a + receiver_a_error,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        axes[0, 1].plot(time, dynamics.photon_number, color=color)
        axes[1, 0].plot(
            time, dynamics.product_region.mean(axis=1), color=color
        )
        axes[1, 1].plot(time, dynamics.connected_a, color=color)
    axes[0, 0].set_ylabel(r"mean molecular $P_a$")
    axes[0, 1].set_ylabel(r"$\langle n_\mathrm{ph}\rangle$")
    axes[1, 0].set_ylabel("mean cis-region population")
    axes[1, 1].set_ylabel(r"$P_{aa}-P_{a1}P_{a2}$")
    for axis in axes[1]:
        axis.set_xlabel("time (fs)")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output.with_suffix(".png"), dpi=180)

    for label, dynamics in (("bare", bare), ("shared cavity", cavity)):
        print(
            f"{label:14s}"
            f" mean_Pa={dynamics.electronic_populations[-1, :, 0].mean():.6f}"
            f" photons={dynamics.photon_number[-1]:.6f}"
            f" mean_cis={dynamics.product_region[-1].mean():.6e}"
            f" connected_Paa={dynamics.connected_a[-1]:+.6e}"
            f" connected_reacted={dynamics.connected_reacted[-1]:+.6e}"
        )
    print(args.output.with_suffix(".npz"))
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
