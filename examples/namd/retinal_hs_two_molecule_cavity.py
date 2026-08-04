"""Test a photon-relay step with two HS retinal molecules in one cavity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.namd.retinal_hs_cavity import (
    RetinalHSTwoMoleculeCavityDynamics,
)


def run_case(args, *, coupling_ev: float, seeded: bool):
    dynamics = RetinalHSTwoMoleculeCavityDynamics(
        nmolecules=2,
        nphi=args.nphi,
        nphotons=args.nphotons,
        cavity_energy_ev=args.cavity_energy,
        coupling_ev=coupling_ev,
    )
    state = None
    if seeded:
        emitter = dynamics.molecular_wavepacket(
            center_rad=np.deg2rad(args.emitter_angle),
            width_rad=np.deg2rad(args.emitter_width),
            electronic_state=2,
        )
        state = dynamics.factorized_state(
            (emitter, dynamics.initial_molecular_state)
        )
    return dynamics.run(
        tmax_fs=args.tmax,
        dt_fs=args.dt,
        save_every=args.save_every,
        state=state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cavity-energy", type=float, default=0.173)
    parser.add_argument("--coupling", type=float, default=0.02)
    parser.add_argument("--nphotons", type=int, default=4)
    parser.add_argument("--nphi", type=int, default=81)
    parser.add_argument("--tmax", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--emitter-angle", type=float, default=40.0)
    parser.add_argument("--emitter-width", type=float, default=10.3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hs-two-molecule-cavity"),
    )
    args = parser.parse_args()

    cases = {
        "bare relay seed": run_case(args, coupling_ev=0.0, seeded=True),
        "shared cavity, both trans": run_case(
            args, coupling_ev=args.coupling, seeded=False
        ),
        "shared cavity, relay seed": run_case(
            args, coupling_ev=args.coupling, seeded=True
        ),
    }

    archive: dict[str, np.ndarray] = {}
    for label, dynamics in cases.items():
        key = label.replace(" ", "_").replace(",", "")
        for name, values in dynamics.as_dict().items():
            if name != "state":
                archive[f"{key}__{name}"] = values
    np.savez(args.output.with_suffix(".npz"), **archive)

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), sharex=True)
    colors = ("0.35", "#4878CF", "#D65F5F")
    for (label, dynamics), color in zip(cases.items(), colors):
        time = dynamics.times_fs
        axes[0, 0].plot(
            time, dynamics.electronic_populations[:, 1, 0],
            color=color, label=label,
        )
        axes[0, 1].plot(time, dynamics.photon_number, color=color)
        axes[1, 0].plot(time, dynamics.connected_a, color=color)
        axes[1, 1].plot(
            time, dynamics.product_region[:, 1], color=color
        )
    axes[0, 0].set_ylabel(r"receiver $P_a$")
    axes[0, 1].set_ylabel(r"$\langle n_\mathrm{ph}\rangle$")
    axes[1, 0].set_ylabel(r"$P_{aa}-P_{a1}P_{a2}$")
    axes[1, 1].set_ylabel("receiver cis-region population")
    for axis in axes[1]:
        axis.set_xlabel("time (fs)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output.with_suffix(".png"), dpi=180)

    for label, dynamics in cases.items():
        print(
            f"{label:27s}"
            f" receiver_Pa={dynamics.electronic_populations[-1, 1, 0]:.6f}"
            f" photons={dynamics.photon_number[-1]:.6f}"
            f" connected_Paa={dynamics.connected_a[-1]:+.6f}"
            f" receiver_cis={dynamics.product_region[-1, 1]:.6e}"
        )
    print(args.output.with_suffix(".npz"))
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
