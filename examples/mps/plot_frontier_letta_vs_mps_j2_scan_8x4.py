#!/usr/bin/env python3
"""Plot the selected 8x4 MPS and graph-LETTA J1-J2 benchmark.

The main figure keeps only the variational energies and their distance to the
finite-D reference.  The derived LETTA-advantage panel is intentionally omitted
because it repeats the energy comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.lines import Line2D
import numpy as np
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "results" / "frontier_letta_vs_mps_j2_scan_8x4_best.json"
DEFAULT_OUTPUT = HERE / "results" / "frontier_letta_vs_mps_j2_scan_8x4_best"

METHODS = {
    "mps_d4": {
        "label": r"MPS $D=4$ (936)",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "o",
        "open_marker": True,
    },
    "letta_d2": {
        "label": r"LETTA $D=2$ (820)",
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "s",
        "open_marker": True,
    },
    "mps_d8": {
        "label": r"MPS $D=8$ (3,496)",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "open_marker": False,
    },
    "letta_d4": {
        "label": r"LETTA $D=4$ (3,240)",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "s",
        "open_marker": False,
    },
}

REFERENCE_STYLE = {
    "label": r"MPS $D=32$ reference (finite $D$)",
    "color": "#202020",
    "linestyle": ":",
    "marker": "x",
}


def _load_data(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("records", ())
    present = {row["method"] for row in records}
    missing = set(METHODS) - present
    if missing:
        raise ValueError(f"scan is missing methods: {sorted(missing)}")
    if not data.get("reference_runs"):
        raise ValueError("scan is missing MPS reference runs.")
    return data


def _aggregate(records, method, field):
    rows = [row for row in records if row["method"] == method]
    ratios = np.asarray(sorted({float(row["j2_ratio"]) for row in rows}))
    center = []
    low = []
    high = []
    for ratio in ratios:
        values = np.asarray(
            [
                float(row[field])
                for row in rows
                if np.isclose(row["j2_ratio"], ratio, atol=1.0e-14, rtol=0.0)
            ]
        )
        center.append(np.median(values))
        low.append(np.quantile(values, 0.25))
        high.append(np.quantile(values, 0.75))
    return ratios, np.asarray(center), np.asarray(low), np.asarray(high)


def _reference_curve(reference_runs):
    rows = [row for row in reference_runs if int(row["bond_dim"]) == 32]
    ratios = np.asarray(sorted({float(row["j2_ratio"]) for row in rows}))
    energies = np.asarray(
        [
            np.median(
                [
                    float(row["energy_per_site"])
                    for row in rows
                    if np.isclose(row["j2_ratio"], ratio, atol=1.0e-14, rtol=0.0)
                ]
            )
            for ratio in ratios
        ]
    )
    return ratios, energies


def _plot_method(axis, records, method, field):
    style = METHODS[method]
    ratios, center, low, high = _aggregate(records, method, field)
    if field == "energy_above_reference_per_site":
        tiny = np.finfo(float).tiny
        center = np.maximum(center, tiny)
        low = np.maximum(low, tiny)
        high = np.maximum(high, tiny)
    axis.fill_between(
        ratios,
        low,
        high,
        color=style["color"],
        alpha=0.13,
        linewidth=0.0,
        zorder=1,
    )
    axis.plot(
        ratios,
        center,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=4.1,
        markerfacecolor="white" if style["open_marker"] else style["color"],
        markeredgewidth=0.85,
        linewidth=1.45,
        zorder=3,
    )


def _panel_label(axis, label):
    axis.text(
        -0.10,
        1.055,
        label,
        transform=axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=8,
    )


def plot_scan(data_path, output_stem):
    data = _load_data(data_path)
    records = data["records"]
    references = data["reference_runs"]

    uplt.rc.update(
        {
            "font.size": 8.8,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.4,
            "legend.fontsize": 7.5,
            "tick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = uplt.subplots(
        ncols=2,
        refwidth=2.48,
        refheight=2.18,
        share=False,
        wspace=4.45,
    )
    energy_axis, error_axis = list(axes)

    for method in METHODS:
        _plot_method(energy_axis, records, method, "energy_per_site")
        _plot_method(
            error_axis,
            records,
            method,
            "energy_above_reference_per_site",
        )

    ref_ratios, ref_energies = _reference_curve(references)
    energy_axis.plot(
        ref_ratios,
        ref_energies,
        color=REFERENCE_STYLE["color"],
        linestyle=REFERENCE_STYLE["linestyle"],
        marker=REFERENCE_STYLE["marker"],
        markersize=4.0,
        markeredgewidth=0.9,
        linewidth=1.25,
        zorder=4,
    )

    common_format = {
        "xlabel": r"$J_2/J_1$",
        "xlim": (-0.035, 1.035),
        "xticks": np.linspace(0.0, 1.0, 6),
        "grid": False,
    }
    energy_axis.format(
        ylabel=r"$E/N$",
        **common_format,
    )
    error_axis.format(
        ylabel=r"$[E-E_{D=32}]/N$",
        yscale="log",
        ylim=(3.0e-3, 9.0e-2),
        **common_format,
    )
    error_axis.set_yticks((5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2))
    error_axis.set_yticklabels(("0.005", "0.01", "0.02", "0.05"))
    for axis in (energy_axis, error_axis):
        axis.yaxis.set_label_position("left")
        axis.yaxis.tick_left()
        axis.tick_params(axis="y", labelleft=True, labelright=False)
        axis.grid(color="#dddddd", linewidth=0.48, which="major")
        axis.tick_params(which="both", direction="out")

    for label, axis in zip("ab", (energy_axis, error_axis)):
        _panel_label(axis, label)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.2,
            markerfacecolor="white" if style["open_marker"] else style["color"],
            markeredgewidth=0.85,
            linewidth=1.45,
            label=style["label"],
        )
        for style in METHODS.values()
    ]
    method_handles.append(
        Line2D(
            [0],
            [0],
            color=REFERENCE_STYLE["color"],
            linestyle=REFERENCE_STYLE["linestyle"],
            marker=REFERENCE_STYLE["marker"],
            markersize=4.0,
            linewidth=1.25,
            label=REFERENCE_STYLE["label"],
        )
    )
    figure.legend(
        method_handles,
        [handle.get_label() for handle in method_handles],
        loc="bottom",
        ncols=5,
        frame=False,
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    uplt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_scan(args.data, args.output)


if __name__ == "__main__":
    main()
