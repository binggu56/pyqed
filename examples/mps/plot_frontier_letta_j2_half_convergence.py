#!/usr/bin/env python3
"""Plot the extended J2/J1=0.5 graph-LETTA convergence checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.ticker import LogFormatterMathtext
import numpy as np
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_4X4 = RESULTS / "frontier_letta_j2_half_convergence_4x4_seed7.json"
DEFAULT_8X4 = RESULTS / "frontier_letta_j2_half_convergence_8x4_seed7.json"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_j2_half_convergence"

STYLES = {
    "4x4": {"label": r"$4\times4$", "color": "#0072B2", "marker": "o"},
    "8x4": {"label": r"$8\times4$", "color": "#D55E00", "marker": "s"},
}


def _series(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    geometry = data["model"]["geometry"]
    nsites = int(np.prod([int(value) for value in geometry.split("x")]))
    continuation = data["continuation"]
    tolerance = float(data["settings"]["target_tolerance"])
    baseline = int(continuation["baseline_passes"])
    source_energy = float(data["source_record"]["energy"])
    source_delta = float(data["source_record"]["final_delta_energy"])
    rows = [row for row in continuation["trace"] if row["phase"] == "continuation"]
    passes = np.asarray([0] + [int(row["sweep"]) - baseline for row in rows])
    energies = np.asarray([source_energy] + [float(row["energy"]) for row in rows])
    deltas = np.asarray([source_delta] + [float(row["delta_energy"]) for row in rows])
    return {
        "geometry": geometry,
        "passes": passes,
        "lowering_per_site": (source_energy - energies) / nsites,
        "normalized_delta": deltas / tolerance,
    }


def _panel_label(axis, label):
    axis.text(
        -0.13,
        1.035,
        label,
        transform=axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_convergence(path_4x4, path_8x4, output_stem):
    series = (_series(path_4x4), _series(path_8x4))
    uplt.rc.update(
        {
            "font.size": 8.8,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "tick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = uplt.subplots(
        ncols=2,
        refwidth=2.45,
        refheight=2.10,
        share=False,
        wspace=4.6,
    )
    lowering_axis, delta_axis = list(axes)
    for values in series:
        style = STYLES[values["geometry"]]
        marker_positions = np.linspace(
            0,
            len(values["passes"]) - 1,
            min(8, len(values["passes"])),
            dtype=int,
        )
        for axis, field in (
            (lowering_axis, "lowering_per_site"),
            (delta_axis, "normalized_delta"),
        ):
            axis.plot(
                values["passes"],
                values[field],
                color=style["color"],
                linewidth=1.45,
                marker=style["marker"],
                markevery=marker_positions,
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=style["label"],
            )

    lowering_axis.format(
        xlabel=r"additional directional passes, $q$",
        ylabel=r"energy lowering per site, $[E(0)-E(q)]/N$",
        grid=False,
    )
    delta_axis.format(
        xlabel=r"additional directional passes, $q$",
        ylabel=r"normalized pass change, $|\Delta E_q|/\tau_E$",
        yscale="log",
        grid=False,
    )
    delta_axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    delta_axis.axhline(1.0, color="#555555", linestyle=":", linewidth=0.9)
    for label, axis in zip("ab", (lowering_axis, delta_axis)):
        axis.grid(color="#dddddd", linewidth=0.48, which="major")
        axis.tick_params(which="both", direction="out")
        _panel_label(axis, label)
    lowering_axis.legend(loc="lower right", ncols=1, frame=False)

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    uplt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-by-four", type=Path, default=DEFAULT_4X4)
    parser.add_argument("--eight-by-four", type=Path, default=DEFAULT_8X4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_convergence(args.four_by_four, args.eight_by_four, args.output)


if __name__ == "__main__":
    main()
