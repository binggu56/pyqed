#!/usr/bin/env python3
"""Quantify timestep, tensor-rank, and grid convergence for D3+ dynamics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def load(path):
    return {name: np.asarray(value) for name, value in np.load(path).items()}


def population_error(left, right):
    times = right["time_fs"]
    interpolated = np.column_stack([
        np.interp(times, left["time_fs"], left["populations"][:, state])
        for state in range(2)
    ])
    error = np.abs(interpolated - right["populations"])
    return float(np.max(error)), float(error[-1, 0])


def population_change_error(left, right):
    times = right["time_fs"]
    interpolated = np.interp(
        times, left["time_fs"], left["populations"][:, 0]
    )
    reference = right["populations"][:, 0]
    error = (interpolated - interpolated[0]) - (reference - reference[0])
    return float(np.max(np.abs(error))), float(abs(error[-1]))


def density_l1(coarse, fine):
    coarse_axes = tuple(coarse["axes"])
    fine_axes = tuple(fine["axes"])
    fine_mesh = np.meshgrid(*fine_axes, indexing="ij")
    points = np.stack(fine_mesh, axis=-1)
    errors = []
    for time, fine_density in zip(
        fine["snapshot_times_fs"], fine["snapshot_densities"]
    ):
        index = int(np.argmin(np.abs(coarse["snapshot_times_fs"] - time)))
        source = coarse["snapshot_densities"][index]
        source /= np.sum(source)
        interpolated = RegularGridInterpolator(
            coarse_axes, source, bounds_error=False, fill_value=0.0
        )(points)
        interpolated /= np.sum(interpolated)
        target = fine_density / np.sum(fine_density)
        errors.append(float(np.sum(np.abs(interpolated - target))))
    return np.asarray(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--timestep", type=Path, required=True)
    parser.add_argument("--rank", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        r"$17^3$, 0.02 fs, r32": load(args.baseline),
        r"$17^3$, 0.01 fs, r32": load(args.timestep),
        r"$17^3$, 0.01 fs, r48": load(args.rank),
        r"$21^3$, 0.01 fs, r48": load(args.grid),
    }
    values = list(runs.values())
    dt_max, dt_final = population_error(values[0], values[1])
    rank_max, rank_final = population_error(values[1], values[2])
    grid_max, grid_final = population_error(values[2], values[3])
    dt_change_max, dt_change_final = population_change_error(values[0], values[1])
    rank_change_max, rank_change_final = population_change_error(values[1], values[2])
    grid_change_max, grid_change_final = population_change_error(values[2], values[3])
    density_error = density_l1(values[2], values[3])
    metrics = {
        "maximum_population_error": {
            "dt_0.02_vs_0.01": dt_max,
            "standard_vs_high_rank": rank_max,
            "grid_17_vs_21": grid_max,
        },
        "final_s1_population_error": {
            "dt_0.02_vs_0.01": dt_final,
            "standard_vs_high_rank": rank_final,
            "grid_17_vs_21": grid_final,
        },
        "maximum_population_change_error": {
            "dt_0.02_vs_0.01": dt_change_max,
            "standard_vs_high_rank": rank_change_max,
            "grid_17_vs_21": grid_change_max,
        },
        "final_population_change_error": {
            "dt_0.02_vs_0.01": dt_change_final,
            "standard_vs_high_rank": rank_change_final,
            "grid_17_vs_21": grid_change_final,
        },
        "grid_density_l1_by_snapshot": density_error.tolist(),
        "maximum_grid_density_l1": float(np.max(density_error)),
        "maximum_boundary_probability": {
            label: float(np.max(run["snapshot_edge_probability"]))
            for label, run in runs.items()
        },
        "maximum_norm_error": {
            label: float(np.max(np.abs(run["norms"] - run["norms"][0])))
            for label, run in runs.items()
        },
        "paper_gates": {
            "maximum_timestep_population_error": 1.0e-3,
            "maximum_rank_population_error": 1.0e-2,
            "maximum_grid_population_error": 2.0e-2,
            "maximum_boundary_probability": 5.0e-2,
        },
    }
    metrics["passes_numerical_gates"] = bool(
        dt_max <= 1.0e-3
        and rank_max <= 1.0e-2
        and grid_max <= 2.0e-2
        and max(metrics["maximum_boundary_probability"].values()) <= 5.0e-2
    )
    (args.output_dir / "d3plus_convergence_report.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    figure, panels = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    reference = values[-1]
    for label, run in runs.items():
        panels[0, 0].plot(
            run["time_fs"],
            run["populations"][:, 0] - run["populations"][0, 0],
            label=label,
        )
        reference_population = np.interp(
            run["time_fs"], reference["time_fs"], reference["populations"][:, 0]
        )
        panels[0, 1].plot(
            run["time_fs"],
            (run["populations"][:, 0] - run["populations"][0, 0])
            - (reference_population - reference["populations"][0, 0]),
            label=label,
        )
        panels[1, 0].plot(
            run["snapshot_times_fs"], run["snapshot_edge_probability"],
            "o-", label=label,
        )
    panels[0, 0].set(
        xlabel="time / fs", ylabel=r"$P_{S_1}(t)-P_{S_1}(0)$",
        title="D$_3^+$ convergence",
    )
    panels[0, 1].axhline(0.0, color="0.3", linewidth=0.8)
    panels[0, 1].set(
        xlabel="time / fs", ylabel=r"$\Delta P_{S_1}-\Delta P_{S_1}^{21^3}$",
        title="Deviation from high-grid result",
    )
    panels[1, 0].axhline(0.05, color="0.3", linestyle="--")
    panels[1, 0].set(
        xlabel="time / fs", ylabel="outer-layer probability",
        title="Boundary diagnostic",
    )
    panels[1, 1].plot(
        values[3]["snapshot_times_fs"], density_error, "o-", color="tab:purple"
    )
    panels[1, 1].set(
        xlabel="time / fs", ylabel=r"$L_1$ density difference",
        title=r"$17^3$ vs $21^3$ total density",
    )
    for panel in panels.flat:
        panel.grid(alpha=0.2)
    panels[0, 0].legend(frameon=False, fontsize=7)
    output = args.output_dir / "d3plus_convergence.png"
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
