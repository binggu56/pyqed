#!/usr/bin/env python3
"""Plot the parameter-matched 4x4 frontier LETTA versus MPS benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "results" / "frontier_letta_vs_mps_4x4.json"
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_vs_mps_4x4"

METHODS = {
    "mps_d8": {
        "label": r"MPS, $D=8$",
        "short_label": "MPS",
        "color": "#4c78a8",
        "linestyle": "-",
        "marker": "o",
    },
    "tied_letta_d4": {
        "label": r"LETTA, $D=4$ + 24 ties",
        "short_label": "LETTA",
        "color": "#e45756",
        "linestyle": "--",
        "marker": "D",
    },
}


def _load_data(path):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    required = set(METHODS)
    if set(data["runs"]) != required:
        raise ValueError("result file must contain tied_letta_d4 and mps_d8 runs.")
    seeds = {
        key: tuple(sorted(int(run["seed"]) for run in data["runs"][key]))
        for key in required
    }
    if len(set(seeds.values())) != 1:
        raise ValueError("LETTA and MPS runs do not use matched seeds.")
    parameter_counts = {
        int(run["parameters"]) for key in required for run in data["runs"][key]
    }
    if len(parameter_counts) != 1:
        raise ValueError("LETTA and MPS runs are not parameter matched.")
    if any(not run["sweep_energies"] for key in required for run in data["runs"][key]):
        raise ValueError("every run must contain an optimization trajectory.")
    return data, seeds["mps_d8"], parameter_counts.pop()


def _trajectory_matrix(runs, exact_energy, nsites):
    minimum_length = min(len(run["sweep_energies"]) for run in runs)
    trajectories = np.asarray(
        [run["sweep_energies"][:minimum_length] for run in runs],
        dtype=float,
    )
    return (trajectories - exact_energy) / nsites


def _metric_values(runs, metric):
    if metric == "infidelity":
        return np.asarray([1.0 - run["ground_state_fidelity"] for run in runs])
    return np.asarray([run[metric] for run in runs], dtype=float)


def plot_comparison(data_path, output_stem):
    """Render convergence, final accuracy, and optimization-cost panels."""
    data, seeds, parameters = _load_data(data_path)
    model = data["model"]
    exact_energy = float(data["exact_reference"]["energy"])
    nsites = int(model["nrows"]) * int(model["ncols"])

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(7.35, 5.25))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.96,
        bottom=0.165,
        top=0.87,
        hspace=0.52,
        wspace=0.42,
        height_ratios=(1.2, 1.0),
    )
    convergence_ax = fig.add_subplot(grid[0, :])
    accuracy_ax = fig.add_subplot(grid[1, 0])
    cost_ax = fig.add_subplot(grid[1, 1])

    for key in ("mps_d8", "tied_letta_d4"):
        runs = sorted(data["runs"][key], key=lambda run: int(run["seed"]))
        style = METHODS[key]
        for run in runs:
            errors = (np.asarray(run["sweep_energies"]) - exact_energy) / nsites
            convergence_ax.plot(
                np.arange(1, len(errors) + 1),
                errors,
                color=style["color"],
                linewidth=0.7,
                alpha=0.20,
                zorder=1,
            )
        common_errors = _trajectory_matrix(runs, exact_energy, nsites)
        passes = np.arange(1, common_errors.shape[1] + 1)
        convergence_ax.plot(
            passes,
            np.median(common_errors, axis=0),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=max(len(passes) // 5, 1),
            markersize=3.8,
            linewidth=1.8,
            zorder=3,
        )

    convergence_ax.set_xscale("log")
    convergence_ax.set_yscale("log")
    convergence_ax.set_xlim(1, 220)
    convergence_ax.set_xticks((1, 2, 5, 10, 20, 50, 100, 200))
    convergence_ax.set_xticklabels(("1", "2", "5", "10", "20", "50", "100", "200"))
    convergence_ax.set_xlabel("directional pass")
    convergence_ax.set_ylabel(r"$\Delta e=(E-E_0)/N$")
    convergence_ax.set_title(
        rf"Equal-parameter optimization trajectories: "
        rf"${model['nrows']}\times{model['ncols']}$ $J_1$–$J_2$ model"
    )
    convergence_ax.grid(color="#dedede", linewidth=0.55, which="major")
    convergence_ax.text(
        0.99,
        0.94,
        rf"five matched seeds; $E_0={exact_energy:.6f}$",
        transform=convergence_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#4c4c4c",
    )
    mps_endpoint_error = float(
        np.median([run["energy_error_per_site"] for run in data["runs"]["mps_d8"]])
    )
    convergence_ax.annotate(
        "LETTA passes the final MPS energy\nafter 3–5 directional passes",
        xy=(3.4, mps_endpoint_error),
        xytext=(7.0, 3.2e-2),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#4c4c4c",
        arrowprops={
            "arrowstyle": "->",
            "color": "#777777",
            "linewidth": 0.7,
            "shrinkA": 2.0,
            "shrinkB": 2.0,
        },
    )
    convergence_ax.text(
        -0.08,
        1.04,
        "a",
        transform=convergence_ax.transAxes,
        fontweight="bold",
        fontsize=10.0,
    )

    metrics = (
        ("energy_error_per_site", r"$(E-E_0)/N$"),
        ("variance", "variance"),
        ("infidelity", r"$1-\mathcal{F}$"),
    )
    metric_positions = np.arange(len(metrics))[::-1]
    offsets = {"mps_d8": -0.12, "tied_letta_d4": 0.12}
    seed_jitter = np.linspace(-0.045, 0.045, len(seeds))
    for metric_position, (metric, _label) in zip(metric_positions, metrics):
        for key in ("mps_d8", "tied_letta_d4"):
            style = METHODS[key]
            values = _metric_values(data["runs"][key], metric)
            y_values = metric_position + offsets[key] + seed_jitter
            accuracy_ax.scatter(
                values,
                y_values,
                s=21,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.78,
                zorder=3,
            )
            median = float(np.median(values))
            accuracy_ax.scatter(
                median,
                metric_position + offsets[key],
                s=51,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="#222222",
                linewidth=0.7,
                zorder=5,
            )
            accuracy_ax.annotate(
                f"{median:.3g}",
                (median, metric_position + offsets[key]),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.0,
                color=style["color"],
            )

    accuracy_ax.set_xscale("log")
    accuracy_ax.set_xlim(3.0e-3, 8.0e-1)
    accuracy_ax.set_yticks(metric_positions, [label for _metric, label in metrics])
    accuracy_ax.set_xlabel("final error metric (log scale; lower is better)")
    accuracy_ax.set_title("Final-state accuracy")
    accuracy_ax.grid(axis="x", color="#e0e0e0", linewidth=0.55, which="major")
    accuracy_ax.text(
        -0.18,
        1.05,
        "b",
        transform=accuracy_ax.transAxes,
        fontweight="bold",
        fontsize=10.0,
    )

    method_keys = ("mps_d8", "tied_letta_d4")
    median_times = np.asarray(
        [
            np.median([run["optimization_seconds"] for run in data["runs"][key]])
            for key in method_keys
        ]
    )
    median_passes = np.asarray(
        [
            int(np.median([run["sweeps_completed"] for run in data["runs"][key]]))
            for key in method_keys
        ]
    )
    converged_runs = np.asarray(
        [
            sum(bool(run["converged"]) for run in data["runs"][key])
            for key in method_keys
        ]
    )
    bars = cost_ax.bar(
        np.arange(2),
        median_times,
        width=0.58,
        color=[METHODS[key]["color"] for key in method_keys],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    for index, (bar, seconds, passes_completed) in enumerate(
        zip(bars, median_times, median_passes)
    ):
        cost_ax.annotate(
            f"{seconds:.2f} s\n{passes_completed} passes",
            (bar.get_x() + bar.get_width() / 2.0, seconds),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        cost_ax.text(
            index,
            1.25,
            f"{converged_runs[index]}/5 converged",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#3f3f3f",
        )
    cost_ax.set_yscale("log")
    cost_ax.set_ylim(1.0, 160.0)
    cost_ax.set_xticks(
        np.arange(2),
        (r"MPS, $D=8$", r"LETTA, $D=4$ + ties"),
    )
    cost_ax.set_ylabel("recorded optimization time (s)")
    cost_ax.set_title(f"Recorded endpoint cost\n(same {parameters:,} raw parameters)")
    cost_ax.grid(axis="y", color="#e0e0e0", linewidth=0.55, which="major")
    cost_ax.text(
        -0.06,
        0.97,
        "c",
        transform=cost_ax.transAxes,
        fontweight="bold",
        fontsize=10.0,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHODS[key]["color"],
            linestyle=METHODS[key]["linestyle"],
            marker=METHODS[key]["marker"],
            markersize=4.2,
            linewidth=1.7,
            label=METHODS[key]["label"],
        )
        for key in method_keys
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncols=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.5,
    )
    fig.text(
        0.5,
        0.035,
        "Equal raw-parameter comparison: MPS uses $D=8$; LETTA uses $D=4$ "
        "and 24 nearest-neighbor physical ties. LETTA did not converge within "
        "its 200-pass budget.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#4c4c4c",
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pdf_path, png_path = plot_comparison(args.data, args.output)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
