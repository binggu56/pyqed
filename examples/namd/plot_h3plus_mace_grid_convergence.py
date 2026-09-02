#!/usr/bin/env python3
"""Compare H3+ MACE/FTT/TNLDR dynamics on two product grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[3]
    runs = root / "data" / "h3plus_fci_augccpvdz" / "runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, default=runs / "mace_tnldr_13")
    parser.add_argument("--fine", type=Path, default=runs / "mace_tnldr_17")
    parser.add_argument(
        "--output", type=Path,
        default=runs / "mace_tnldr_17" / "h3plus_mace_grid_convergence.png",
    )
    args = parser.parse_args()
    coarse = np.load(args.coarse / "h3plus_mace_tnldr_dynamics.npz")
    fine = np.load(args.fine / "h3plus_mace_tnldr_dynamics.npz")
    if not np.allclose(coarse["time_fs"], fine["time_fs"]):
        raise ValueError("the dynamics time grids differ")

    time = fine["time_fs"]
    difference = np.abs(coarse["populations"] - fine["populations"])
    figure, panels = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    colors = ("#0072B2", "#D55E00")
    for state, color in enumerate(colors):
        panels[0].plot(
            time, coarse["populations"][:, state], color=color,
            label=fr"$13^3$, $S_{state + 1}$",
        )
        panels[0].plot(
            time, fine["populations"][:, state], "--", color=color,
            label=fr"$17^3$, $S_{state + 1}$",
        )
    panels[0].set(
        xlabel="time / fs", ylabel="adiabatic population",
        title="Grid comparison", ylim=(-0.02, 1.02),
    )
    panels[0].legend(frameon=False, ncol=2)
    panels[1].semilogy(
        time, np.maximum(np.max(difference, axis=1), 1.0e-15), color="#7A3E9D"
    )
    panels[1].set(
        xlabel="time / fs", ylabel="maximum population difference",
        title=r"$13^3$ versus $17^3$",
    )
    for panel in panels:
        panel.grid(alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=320)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)

    report = {
        "maximum_population_difference": float(np.max(difference)),
        "final_population_difference": difference[-1].tolist(),
        "maximum_norm_difference": float(
            np.max(np.abs(coarse["norms"] - fine["norms"]))
        ),
        "final_outer_layer_probability_13": float(
            coarse["snapshot_edge_probability"][-1]
        ),
        "final_outer_layer_probability_17": float(
            fine["snapshot_edge_probability"][-1]
        ),
        "figure": str(args.output),
    }
    args.output.with_name("grid_convergence.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
