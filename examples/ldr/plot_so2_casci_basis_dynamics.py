#!/usr/bin/env python3
"""Plot SO2 CGLDR basis and primary/secondary partition convergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt


DEFAULT_SOURCES = {
    "STO-3G, 2P+1S": Path(
        "/private/tmp/so2_spin_pure_cas6e6o_inner_three_anchor_9x9x9_20fs/"
        "inner_three_anchor_relaxed_pt_cgldr_observables.npz"
    ),
    "6-31G*, 2P+1S": Path(
        "/private/tmp/so2_cas6e6o_631gstar_2p1s_inner_three_anchor_9x9x9_20fs/"
        "so2_cgldr_dynamics_dense.npz"
    ),
    "STO-3G, 1P+2S": Path(
        "/private/tmp/so2_spin_pure_cas6e6o_1p2s_axial_9x9x9_20fs/"
        "one_primary_two_secondary_observables.npz"
    ),
    "6-31G*, 1P+2S": Path(
        "/private/tmp/so2_cas6e6o_631gstar_1p2s_axial_9x9x9_20fs/"
        "so2_cgldr_dynamics_dense.npz"
    ),
}


def load(path):
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def comparison_summary(results):
    summary = {"runs": {}}
    for label, values in results.items():
        summary["runs"][label] = {
            "final_reference_populations": values["reference_populations"][-1].tolist(),
            "final_means_qs_theta_qa": values["means"][-1].tolist(),
            "final_stds_qs_theta_qa": np.sqrt(values["variances"][-1]).tolist(),
            "max_norm_error": float(np.max(np.abs(values["norms"] - 1.0))),
        }
    pairs = {
        "basis_effect_2p1s": ("STO-3G, 2P+1S", "6-31G*, 2P+1S"),
        "basis_effect_1p2s": ("STO-3G, 1P+2S", "6-31G*, 1P+2S"),
        "partition_effect_sto3g": ("STO-3G, 2P+1S", "STO-3G, 1P+2S"),
        "partition_effect_631gstar": ("6-31G*, 2P+1S", "6-31G*, 1P+2S"),
    }
    for name, (left, right) in pairs.items():
        pop_delta = (
            results[right]["reference_populations"]
            - results[left]["reference_populations"]
        )
        mean_delta = results[right]["means"] - results[left]["means"]
        summary[name] = {
            "final_population_delta": pop_delta[-1].tolist(),
            "max_population_abs_delta": np.max(np.abs(pop_delta), axis=0).tolist(),
            "final_mean_delta_qs_theta_qa": mean_delta[-1].tolist(),
            "max_mean_abs_delta_qs_theta_qa": np.max(
                np.abs(mean_delta), axis=0
            ).tolist(),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_cas6e6o_basis_comparison"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {label: load(path) for label, path in DEFAULT_SOURCES.items()}
    times = results["STO-3G, 2P+1S"]["times_fs"]
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
        height=5.65,
        share=False,
        hspace=2.3,
        wspace=6.0,
    )
    styles = {
        "STO-3G, 2P+1S": dict(color="#0072B2", linestyle="-", linewidth=1.55),
        "6-31G*, 2P+1S": dict(color="#D55E00", linestyle="-", linewidth=1.55),
        "STO-3G, 1P+2S": dict(color="#0072B2", linestyle="--", linewidth=1.45),
        "6-31G*, 1P+2S": dict(color="#D55E00", linestyle="--", linewidth=1.45),
    }
    panels = (
        ("reference_populations", 1, lambda value: value),
        ("reference_populations", 2, lambda value: value),
        ("means", 0, lambda value: value),
        ("means", 1, np.rad2deg),
        ("means", 2, lambda value: value),
    )
    for axis, (field, index, transform) in zip(axes, panels):
        for label, values in results.items():
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
        (-0.01, 0.46),
        (0.54, 1.01),
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
    figure.legend(handles, labels, loc="top", ncols=2, frame=False, order="F")
    png = args.output_dir / "so2_casci_basis_partition_comparison.png"
    pdf = args.output_dir / "so2_casci_basis_partition_comparison.pdf"
    figure.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary(results), indent=2) + "\n"
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
