#!/usr/bin/env python3
"""Plot the 4x4 frontier-gauge LETTA benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_UNIFORM_DATA = HERE / "results" / "frontier_letta_gauge20_4x4.json"
DEFAULT_PROBABILITY_DATA = (
    HERE / "results" / "frontier_letta_gauge_probability20_4x4.json"
)
DEFAULT_BLOCK_DATA = HERE / "results" / "frontier_letta_optimization_4x4.json"
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_gauge_results_4x4"

STYLES = {
    "plain": {
        "label": "MPS warm + sweeps",
        "short_label": "plain sweeps",
        "color": "#4c78a8",
        "linestyle": "-",
        "marker": "o",
    },
    "uniform": {
        "label": "uniform frontier gauge",
        "short_label": "uniform gauge",
        "color": "#f58518",
        "linestyle": "--",
        "marker": "s",
    },
    "probability": {
        "label": "probability-weighted gauge",
        "short_label": "probability gauge",
        "color": "#009e73",
        "linestyle": "-.",
        "marker": "D",
    },
    "block_metric": {
        "label": "block-metric relaxation",
        "short_label": "block metric",
        "color": "#8f63b8",
        "linestyle": ":",
        "marker": "^",
    },
}


def _load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _runs_by_seed(runs):
    return {int(run["seed"]): run for run in runs}


def _same_model(left, right):
    keys = ("nrows", "ncols", "j1", "j2", "boundary", "site_order")
    return all(left.get(key) == right.get(key) for key in keys)


def _paired_summary(document, key):
    summary = document["summary"]["paired"][key]
    deltas = {
        int(seed): float(delta)
        for seed, delta in summary["energy_delta_by_seed"].items()
    }
    return {
        "deltas": deltas,
        "median": float(summary["median_energy_delta"]),
        "wins": int(summary["candidate_wins"]),
        "extra_seconds": float(summary["median_standalone_seconds_delta"]),
    }


def _validated_data(uniform_path, probability_path, block_path):
    uniform = _load_json(uniform_path)
    probability = _load_json(probability_path)
    block = _load_json(block_path)
    documents = (uniform, probability, block)
    model = probability["model"]
    if not all(_same_model(model, document["model"]) for document in documents):
        raise ValueError("benchmark files use different models.")

    exact_energy = float(probability["exact_reference"]["energy"])
    if not all(
        np.isclose(
            exact_energy,
            float(document["exact_reference"]["energy"]),
            atol=1.0e-12,
            rtol=0.0,
        )
        for document in documents
    ):
        raise ValueError("benchmark files use different exact references.")

    trajectories = {
        "plain": probability["runs"]["mps_warm_sweep"],
        "probability": probability["runs"]["mps_warm_frontier_gauge"],
    }
    seed_sets = {
        key: tuple(sorted(_runs_by_seed(runs))) for key, runs in trajectories.items()
    }
    if len(set(seed_sets.values())) != 1:
        raise ValueError("probability-gauge trajectories do not use matched seeds.")
    lengths = {
        len(run["directional_pass_energies"])
        for runs in trajectories.values()
        for run in runs
    }
    if len(lengths) != 1:
        raise ValueError("benchmark trajectories have inconsistent lengths.")

    paired = {
        "uniform": _paired_summary(
            uniform,
            "mps_warm_frontier_gauge_minus_mps_warm_sweep",
        ),
        "probability": _paired_summary(
            probability,
            "mps_warm_frontier_gauge_minus_mps_warm_sweep",
        ),
        "block_metric": _paired_summary(
            block,
            "mps_warm_block_metric_minus_mps_warm_sweep",
        ),
    }
    paired_seed_sets = {tuple(sorted(result["deltas"])) for result in paired.values()}
    if len(paired_seed_sets) != 1:
        raise ValueError("paired benchmarks do not use the same seeds.")
    return model, exact_energy, trajectories, paired


def _trajectory_matrix(runs, exact_energy, nsites):
    energies = np.asarray(
        [
            [float(run["initial_energy"]), *run["directional_pass_energies"]]
            for run in sorted(runs, key=lambda item: int(item["seed"]))
        ]
    )
    return (energies - exact_energy) / nsites


def _format_signed(value):
    return rf"${value:+.1e}$"


def plot_results(uniform_data, probability_data, block_data, output_stem):
    """Render convergence, paired energies, and conditioning diagnostics."""
    model, exact_energy, trajectories, paired = _validated_data(
        uniform_data,
        probability_data,
        block_data,
    )
    nsites = int(model["nrows"]) * int(model["ncols"])
    n_passes = len(trajectories["plain"][0]["directional_pass_energies"])
    passes = np.arange(n_passes + 1)

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(7.35, 5.35))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.91,
        bottom=0.155,
        top=0.87,
        hspace=0.54,
        wspace=0.46,
        height_ratios=(1.22, 1.0),
    )
    convergence_ax = fig.add_subplot(grid[0, :])
    paired_ax = fig.add_subplot(grid[1, 0])
    conditioning_ax = fig.add_subplot(grid[1, 1])

    for key in ("plain", "probability"):
        style = STYLES[key]
        errors = _trajectory_matrix(trajectories[key], exact_energy, nsites)
        for trajectory in errors:
            convergence_ax.plot(
                passes,
                trajectory,
                color=style["color"],
                linewidth=0.7,
                alpha=0.22,
                zorder=1,
            )
        convergence_ax.plot(
            passes,
            np.median(errors, axis=0),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=[0, 5, 10, 15, 20],
            markersize=3.8,
            linewidth=1.8,
            zorder=3,
        )

    convergence_ax.set_yscale("log")
    convergence_ax.set_xlim(0, n_passes)
    convergence_ax.set_xticks((0, 5, 10, 15, 20))
    convergence_ax.set_xlabel("directional pass")
    convergence_ax.set_ylabel(r"energy error per site $(E-E_0)/N$")
    convergence_ax.set_title(
        rf"Paired fixed-budget trajectories: "
        rf"${model['nrows']}\times{model['ncols']}$ $J_1$–$J_2$, $D=4$"
    )
    convergence_ax.grid(axis="y", which="major", color="#d9d9d9", linewidth=0.6)
    convergence_ax.text(
        0.99,
        0.94,
        rf"five seeds; $E_0={exact_energy:.6f}$",
        transform=convergence_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#4c4c4c",
    )
    convergence_ax.text(
        -0.075,
        1.04,
        "a",
        transform=convergence_ax.transAxes,
        fontweight="bold",
        fontsize=10.0,
    )

    paired_keys = ("uniform", "probability", "block_metric")
    seed_jitter = np.linspace(-0.16, 0.16, 5)
    for index, key in enumerate(paired_keys):
        result = paired[key]
        style = STYLES[key]
        deltas = np.asarray(
            [result["deltas"][seed] for seed in sorted(result["deltas"])],
            dtype=float,
        )
        paired_ax.scatter(
            index + seed_jitter,
            deltas,
            s=24,
            marker="o",
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.82,
            zorder=3,
        )
        paired_ax.scatter(
            index,
            result["median"],
            s=49,
            marker="D",
            facecolor=style["color"],
            edgecolor="#222222",
            linewidth=0.65,
            zorder=5,
        )
        paired_ax.annotate(
            _format_signed(result["median"]),
            (index, result["median"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    paired_ax.axhline(0.0, color="#555555", linewidth=0.8, zorder=1)
    paired_ax.set_yscale("symlog", linthresh=1.0e-3, linscale=1.0)
    paired_ax.set_ylim(-3.6e-2, 6.0e-3)
    paired_ax.set_yticks((-3.0e-2, -1.0e-2, -3.0e-3, -1.0e-3, 0.0, 1.0e-3, 3.0e-3))
    paired_ax.set_yticklabels(
        (
            r"$-0.03$",
            r"$-0.01$",
            r"$-0.003$",
            r"$-0.001$",
            "$0$",
            "$0.001$",
            "$0.003$",
        )
    )
    paired_ax.set_xticks(
        np.arange(len(paired_keys)),
        [
            (
                f"{STYLES[key]['short_label']}\n"
                f"{paired[key]['wins']}/5 wins; "
                f"+{paired[key]['extra_seconds']:.1f} s"
            )
            for key in paired_keys
        ],
    )
    paired_ax.set_ylabel(r"$E_{\rm candidate}-E_{\rm plain}$ (symlog)")
    paired_ax.set_title("Paired 20-pass endpoints\n(negative is better)")
    paired_ax.grid(axis="y", which="major", color="#e0e0e0", linewidth=0.55)
    paired_ax.text(
        -0.2,
        1.05,
        "b",
        transform=paired_ax.transAxes,
        fontweight="bold",
        fontsize=10.0,
    )

    plain_by_seed = _runs_by_seed(trajectories["plain"])
    gauge_by_seed = _runs_by_seed(trajectories["probability"])
    seeds = sorted(plain_by_seed)
    for seed in seeds:
        plain = plain_by_seed[seed]
        gauge = gauge_by_seed[seed]
        conditioning_ax.plot(
            (
                plain["lower_quartile_local_metric_rank_fraction"],
                gauge["lower_quartile_local_metric_rank_fraction"],
            ),
            (
                plain["maximum_local_residual_norm"],
                gauge["maximum_local_residual_norm"],
            ),
            color="#aaaaaa",
            linewidth=0.75,
            alpha=0.65,
            zorder=1,
        )

    for key, by_seed in (("plain", plain_by_seed), ("probability", gauge_by_seed)):
        style = STYLES[key]
        ranks = np.asarray(
            [
                by_seed[seed]["lower_quartile_local_metric_rank_fraction"]
                for seed in seeds
            ]
        )
        residuals = np.asarray(
            [by_seed[seed]["maximum_local_residual_norm"] for seed in seeds]
        )
        conditioning_ax.scatter(
            ranks,
            residuals,
            s=31,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        conditioning_ax.scatter(
            np.median(ranks),
            np.median(residuals),
            s=72,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="#222222",
            linewidth=0.8,
            zorder=5,
        )

    conditioning_ax.set_yscale("log")
    conditioning_ax.set_xlim(0.79, 0.965)
    conditioning_ax.set_ylim(5.0e-6, 1.5e-3)
    conditioning_ax.set_xlabel("lower-quartile metric-rank fraction")
    conditioning_ax.set_ylabel("maximum local residual norm")
    conditioning_ax.set_title(
        "Within-run local conditioning\n(right and down are better)"
    )
    conditioning_ax.grid(color="#e0e0e0", linewidth=0.55)
    conditioning_ax.text(
        -0.06,
        0.97,
        "c",
        transform=conditioning_ax.transAxes,
        fontweight="bold",
        fontsize=10.0,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[key]["color"],
            linestyle=STYLES[key]["linestyle"],
            marker=STYLES[key]["marker"],
            markersize=4.0,
            linewidth=1.6,
            label=STYLES[key]["label"],
        )
        for key in ("plain", "uniform", "probability", "block_metric")
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncols=4,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.25,
    )
    fig.text(
        0.5,
        0.025,
        "Uniform, probability-weighted, and block-metric endpoints come from "
        "separate matched-seed benchmark runs; none converged within 20 passes.",
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
    parser.add_argument("--uniform-data", type=Path, default=DEFAULT_UNIFORM_DATA)
    parser.add_argument(
        "--probability-data",
        type=Path,
        default=DEFAULT_PROBABILITY_DATA,
    )
    parser.add_argument("--block-data", type=Path, default=DEFAULT_BLOCK_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pdf_path, png_path = plot_results(
        args.uniform_data,
        args.probability_data,
        args.block_data,
        args.output,
    )
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
