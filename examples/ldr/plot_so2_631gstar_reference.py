#!/usr/bin/env python3
"""Plot SO2 CASCI/6-31G* CGLDR dynamics against full LDR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt


SOURCES = {
    "Full LDR": Path(
        "/private/tmp/so2_cas6e6o_631gstar_full_ldr_9x9x9_20fs/"
        "full_ldr_dynamics.npz"
    ),
    "2P+1S (5A)": Path(
        "/private/tmp/so2_cas6e6o_631gstar_2p1s_inner_five_anchor_9x9x9_20fs/"
        "so2_cgldr_dynamics_dense.npz"
    ),
    "2P+1S (1A F/G)": Path(
        "/private/tmp/so2_cas6e6o_631gstar_2p1s_single_anchor_9x9x9_20fs/"
        "so2_cgldr_dynamics_dense.npz"
    ),
    "2P+1S (3A)": Path(
        "/private/tmp/so2_cas6e6o_631gstar_2p1s_inner_three_anchor_9x9x9_20fs/"
        "so2_cgldr_dynamics_dense.npz"
    ),
    "1P+2S (3A)": Path(
        "/private/tmp/so2_cas6e6o_631gstar_1p2s_axial_9x9x9_20fs/"
        "so2_cgldr_dynamics_dense.npz"
    ),
}
PLOT_LABELS = (
    "Full LDR",
    "2P+1S (1A F/G)",
    "2P+1S (3A)",
    "1P+2S (3A)",
)


def load(path):
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def summary(results):
    reference = results["Full LDR"]
    output = {
        "method": "RHF/CASCI(6e,6o)/6-31G*",
        "reference": "full 9x9x9 LDR",
        "comparisons": {},
    }
    for label in (
        "2P+1S (5A)",
        "2P+1S (1A F/G)",
        "2P+1S (3A)",
        "1P+2S (3A)",
    ):
        values = results[label]
        pop_delta = values["reference_populations"] - reference["reference_populations"]
        active_pop_delta = pop_delta[:, 1:]
        mean_delta = values["means"] - reference["means"]
        std_delta = np.sqrt(values["variances"]) - np.sqrt(reference["variances"])
        output["comparisons"][label] = {
            "final_population_abs_error": np.abs(pop_delta[-1]).tolist(),
            "max_population_abs_error": np.max(np.abs(pop_delta), axis=0).tolist(),
            "active_population_rmse": float(
                np.sqrt(np.mean(active_pop_delta**2))
            ),
            "active_population_mae": float(np.mean(np.abs(active_pop_delta))),
            "final_mean_abs_error_qs_theta_qa": np.abs(mean_delta[-1]).tolist(),
            "max_mean_abs_error_qs_theta_qa": np.max(
                np.abs(mean_delta), axis=0
            ).tolist(),
            "final_std_abs_error_qs_theta_qa": np.abs(std_delta[-1]).tolist(),
            "max_std_abs_error_qs_theta_qa": np.max(
                np.abs(std_delta), axis=0
            ).tolist(),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_cas6e6o_631gstar_reference_comparison"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {label: load(path) for label, path in SOURCES.items()}
    times = results["Full LDR"]["times_fs"]
    for values in results.values():
        np.testing.assert_allclose(values["times_fs"], times)

    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.45,
            "savefig.transparent": False,
        }
    )
    figure, axes = uplt.subplots(
        array=[
            [1, 1, 1, 3, 3],
            [1, 1, 1, 3, 3],
            [1, 1, 1, 4, 4],
            [2, 2, 2, 4, 4],
            [2, 2, 2, 5, 5],
            [2, 2, 2, 5, 5],
        ],
        width=7.2,
        height=5.55,
        share=False,
        hspace=2.3,
        wspace=6.0,
    )
    styles = {
        "Full LDR": dict(color="black", linestyle="-", linewidth=1.7),
        "2P+1S (1A F/G)": dict(
            color="#009E73", linestyle=":", linewidth=1.55
        ),
        "2P+1S (3A)": dict(
            color="#0072B2", linestyle="--", linewidth=1.5
        ),
        "1P+2S (3A)": dict(
            color="#D55E00", linestyle="-.", linewidth=1.45
        ),
    }
    panels = (
        ("reference_populations", 1, lambda value: value),
        ("reference_populations", 2, lambda value: value),
        ("means", 0, lambda value: value),
        ("means", 1, np.rad2deg),
        ("means", 2, lambda value: value),
    )
    for axis, (field, index, transform) in zip(axes, panels):
        for label in PLOT_LABELS:
            values = results[label]
            axis.plot(
                times,
                transform(values[field][:, index]),
                label=label,
                **styles[label],
            )
        axis.format(
            xlim=(0.0, 20.0),
            xticks=np.arange(0.0, 20.1, 5.0),
            grid=False,
            tickdir="in",
            tickminor=True,
        )
        axis.grid(axis="y", color="0.88", linewidth=0.45, alpha=0.75)

    titles = (
        r"$\mathbf{(a)}$",
        r"$\mathbf{(b)}$",
        r"$\mathbf{(c)}$  Symmetric stretch",
        r"$\mathbf{(d)}$  Bend angle",
        r"$\mathbf{(e)}$  Antisymmetric stretch",
    )
    ylabels = (
        r"$P_1$",
        r"$P_2$",
        r"$\langle q_s\rangle$ (bohr)",
        r"$\langle\theta\rangle$ (deg)",
        r"$\langle q_a\rangle$ (bohr)",
    )
    ylimits = (
        (-0.01, 0.50),
        (0.50, 1.01),
        (3.815, 3.882),
        (111.5, 120.1),
        (-0.0205, 0.0050),
    )
    for axis, title, ylabel, ylim in zip(axes, titles, ylabels, ylimits):
        axis.format(title=title, titleloc="left", ylabel=ylabel, ylim=ylim)
    axes[0].tick_params(labelbottom=False)
    axes[1].format(xlabel="Time (fs)")
    axes[2].tick_params(labelbottom=False)
    axes[3].tick_params(labelbottom=False)
    axes[4].format(xlabel="Time (fs)")
    axes[4].axhline(0.0, color="0.55", linewidth=0.65, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="top", ncols=4, frame=False, order="F")

    png = args.output_dir / "so2_631gstar_cgldr_vs_full_ldr.png"
    pdf = args.output_dir / "so2_631gstar_cgldr_vs_full_ldr.pdf"
    figure.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary(results), indent=2) + "\n"
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
