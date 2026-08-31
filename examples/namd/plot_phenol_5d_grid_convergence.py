#!/usr/bin/env python3
"""Compare phenol 5D TTLDR dynamics across three nuclear DVR grids."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = ("#0072B2", "#D55E00", "#009E73")
STYLES = (":", "--", "-")


def load_run(path):
    path = Path(path)
    if path.is_dir():
        path = path / "phenol_sa_casscf_5d_ftt_ttldr.npz"
    with np.load(path, allow_pickle=True) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    result["radial_axis"] = np.asarray(result["axes"][0], dtype=float)
    return result


def radial_density(run, common_axis):
    axis = run["radial_axis"]
    probability = np.asarray(run["final_radial"], dtype=float)
    density = probability / np.gradient(axis)
    density = np.interp(common_axis, axis, density)
    density /= np.trapezoid(density, common_axis)
    return density


def pair_metrics(reference, candidate):
    if not np.allclose(candidate["times_fs"], reference["times_fs"]):
        raise ValueError("all runs must use the same observation times")
    common_axis = np.linspace(
        max(candidate["radial_axis"][0], reference["radial_axis"][0]),
        min(candidate["radial_axis"][-1], reference["radial_axis"][-1]),
        2001,
    )
    reference_density = radial_density(reference, common_axis)
    candidate_density = radial_density(candidate, common_axis)
    population_error = np.abs(candidate["populations"] - reference["populations"])
    absorbed_error = np.abs(
        candidate["absorbed_probabilities"]
        - reference["absorbed_probabilities"]
    )
    yield_error = np.abs(candidate["cap_yields"] - reference["cap_yields"])
    return {
        "maximum_population_error": float(np.max(population_error)),
        "final_maximum_population_error": float(np.max(population_error[-1])),
        "maximum_absorbed_probability_error": float(np.max(absorbed_error)),
        "final_absorbed_probability_error": float(absorbed_error[-1]),
        "maximum_channel_yield_error": float(np.max(yield_error)),
        "final_maximum_channel_yield_error": float(np.max(yield_error[-1])),
        "final_radial_density_l1_error": float(
            np.trapezoid(np.abs(candidate_density - reference_density), common_axis)
        ),
        "final_mean_radius_error_angstrom": float(
            abs(
                np.sum(candidate["radial_axis"] * candidate["final_radial"])
                - np.sum(reference["radial_axis"] * reference["final_radial"])
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("intermediate", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [load_run(path) for path in (args.coarse, args.intermediate, args.target)]
    labels = ["49x5x3x5x5", "49x7x5x9x7", "65x9x7x11x9"]
    times = runs[-1]["times_fs"]
    for run in runs[:-1]:
        if not np.allclose(run["times_fs"], times):
            raise ValueError("all runs must use the same observation times")

    figure, panels = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    for grid, (run, label, style) in enumerate(zip(runs, labels, STYLES)):
        for state, color in enumerate(COLORS):
            panels[0, 0].plot(
                times,
                run["populations"][:, state],
                color=color,
                ls=style,
                lw=1.25 + 0.25 * grid,
            )
        panels[0, 1].plot(
            times,
            run["absorbed_probabilities"],
            color="black",
            ls=style,
            lw=1.25 + 0.25 * grid,
            label=label,
        )
        axis = run["radial_axis"]
        density = run["final_radial"] / np.gradient(axis)
        density /= np.trapezoid(density, axis)
        panels[1, 0].plot(
            axis, density, ls=style, lw=1.25 + 0.25 * grid, label=label
        )

    population_handles = [
        Line2D([], [], color=color, lw=1.7, label=f"P{state}")
        for state, color in enumerate(COLORS)
    ]
    grid_handles = [
        Line2D([], [], color="0.35", ls=style, lw=1.7, label=label)
        for label, style in zip(labels, STYLES)
    ]

    reference = runs[-1]
    for run, label, color in zip(runs[:-1], labels[:-1], COLORS[:2]):
        population_error = np.max(
            np.abs(run["populations"] - reference["populations"]), axis=1
        )
        absorbed_error = np.abs(
            run["absorbed_probabilities"]
            - reference["absorbed_probabilities"]
        )
        panels[1, 1].semilogy(
            times, np.maximum(population_error, 1.0e-12),
            color=color, lw=1.6, label=fr"max $|\Delta P_s|$, {label}",
        )
        panels[1, 1].semilogy(
            times, np.maximum(absorbed_error, 1.0e-12),
            color=color, ls="--", lw=1.3, label=fr"$|\Delta A|$, {label}",
        )

    panels[0, 0].set(
        xlabel="time (fs)", ylabel="absolute P-gauge population",
        title="Electronic populations",
    )
    panels[0, 1].set(
        xlabel="time (fs)", ylabel="absorbed probability",
        title="CAP dissociation probability",
    )
    panels[1, 0].set(
        xlabel=r"$R_{OH}$ (angstrom)", ylabel="normalized probability density",
        title="Final radial marginal",
    )
    panels[1, 1].set(
        xlabel="time (fs)", ylabel="absolute difference",
        title="Difference from target grid",
    )
    panels[0, 0].legend(
        handles=population_handles + grid_handles,
        frameon=False,
        fontsize=7,
        ncol=2,
    )
    panels[0, 1].legend(frameon=False, fontsize=7)
    panels[1, 0].legend(frameon=False, fontsize=7)
    panels[1, 1].legend(frameon=False, fontsize=6.8)
    for panel in panels.flat:
        panel.spines[["top", "right"]].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=280)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)

    metrics = {
        f"{labels[0]}_vs_{labels[1]}": pair_metrics(runs[1], runs[0]),
        f"{labels[1]}_vs_{labels[2]}": pair_metrics(runs[2], runs[1]),
        f"{labels[0]}_vs_{labels[2]}": pair_metrics(runs[2], runs[0]),
        "final_values": {
            label: {
                "populations": run["populations"][-1].tolist(),
                "absorbed_probability": float(run["absorbed_probabilities"][-1]),
                "channel_yields": run["cap_yields"][-1].tolist(),
            }
            for label, run in zip(labels, runs)
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"figure: {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
