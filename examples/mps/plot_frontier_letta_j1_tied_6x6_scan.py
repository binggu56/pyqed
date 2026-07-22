#!/usr/bin/env python3
"""Plot the 6x6 J1-tied LETTA scan for the frustrated Heisenberg model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_j1_tied_6x6_scan"
MANUSCRIPT_FIGURES = (
    HERE.parents[1] / "docs" / "variational_letta_latex" / "figures"
)

INPUTS = {
    0.0: RESULTS / "frontier_letta_j1_tied_6x6_j2_0p0.json",
    0.25: RESULTS / "frontier_letta_j1_tied_6x6_j2_0p25.json",
    0.5: RESULTS / "frontier_letta_block_sparse_6x6.json",
    0.7: RESULTS / "frontier_letta_j1_tied_6x6_j2_0p7.json",
    0.75: RESULTS / "frontier_letta_j1_tied_6x6_j2_0p75.json",
    0.8: RESULTS / "frontier_letta_j1_tied_6x6_j2_0p8.json",
    1.0: RESULTS / "frontier_letta_j1_tied_6x6_j2_1p0.json",
}
J2_HALF_REFERENCES = RESULTS / "frontier_letta_block_sparse_6x6_mps_references.json"

METHODS = {
    "mps_d8": {
        "label": r"MPS $D=8$ (4,008)",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "o",
        "open_marker": True,
    },
    "mps_d16": {
        "label": r"MPS $D=16$ (15,016)",
        "color": "#56B4E9",
        "linestyle": "-.",
        "marker": "^",
        "open_marker": True,
    },
    "mps_d32": {
        "label": r"MPS $D=32$ (55,976)",
        "color": "#202020",
        "linestyle": ":",
        "marker": "x",
        "open_marker": False,
    },
    "letta_d4": {
        "label": r"$J_1$-LETTA $D=4$ (3,752)",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "s",
        "open_marker": False,
    },
}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result_rows():
    rows = []
    for ratio, path in INPUTS.items():
        data = _load(path)
        results = dict(data["results"])
        if np.isclose(ratio, 0.5):
            reference_data = _load(J2_HALF_REFERENCES)
            for key in ("mps_d16", "mps_d32"):
                results[key] = reference_data["results"][key]
        convergence = data.get("convergence", {})
        for method in METHODS:
            result = results[method]
            energy = float(result["energy"])
            rows.append(
                {
                    "j2_ratio": float(ratio),
                    "method": method,
                    "energy": energy,
                    "energy_per_site": float(result.get("energy_per_site", energy / 36)),
                    "parameters": int(
                        result.get(
                            "parameters",
                            result.get(
                                "stored_parameters",
                                result.get("parameter_capacity"),
                            ),
                        )
                    ),
                    "converged": bool(result.get("converged", False)),
                    "directional_passes_completed": int(
                        result.get("directional_passes_completed", 0)
                    ),
                    "final_delta_energy": result.get("final_delta_energy"),
                    "cycle_gain_per_site": (
                        convergence.get("cycle_records", [{}])[-1].get("gain_per_site")
                        if method == "letta_d4"
                        else None
                    ),
                }
            )
    d32_by_ratio = {
        row["j2_ratio"]: row["energy_per_site"]
        for row in rows
        if row["method"] == "mps_d32"
    }
    for row in rows:
        row["energy_above_mps_d32_per_site"] = (
            row["energy_per_site"] - d32_by_ratio[row["j2_ratio"]]
        )
    return rows


def write_data(path, rows):
    payload = {
        "model": {
            "nrows": 6,
            "ncols": 6,
            "j1": 1.0,
            "j2_ratios": sorted(INPUTS),
            "boundary": "open",
            "site_order": "row-wise snake",
            "letta_tie_graph": "all nearest-neighbor J1 bonds",
        },
        "reference_note": (
            "MPS D=32 is a finite-D reference, not an exact ground-state energy."
        ),
        "records": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _series(rows, method, field):
    selected = sorted(
        [row for row in rows if row["method"] == method],
        key=lambda row: row["j2_ratio"],
    )
    return (
        np.asarray([row["j2_ratio"] for row in selected], dtype=float),
        np.asarray([row[field] for row in selected], dtype=float),
        selected,
    )


def _plot_method(axis, rows, method, field):
    style = METHODS[method]
    x, y, selected = _series(rows, method, field)
    axis.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=4.2,
        markerfacecolor="white" if style["open_marker"] else style["color"],
        markeredgewidth=0.9,
        linewidth=1.45,
        zorder=3,
    )
    if method == "letta_d4":
        unconverged = [
            (row["j2_ratio"], row[field])
            for row in selected
            if not row["converged"]
        ]
        if unconverged:
            ux, uy = np.asarray(unconverged).T
            axis.scatter(
                ux,
                uy,
                marker="x",
                s=34,
                linewidths=1.2,
                color="#8c2d04",
                zorder=5,
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


def plot_scan(rows, output_stem):
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.8,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.4,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 500,
            "savefig.facecolor": "white",
        }
    )
    figure, (energy_axis, offset_axis) = plt.subplots(
        1,
        2,
        figsize=(5.9, 2.55),
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.90,
        bottom=0.33,
        top=0.88,
        wspace=0.34,
    )

    for method in METHODS:
        _plot_method(energy_axis, rows, method, "energy_per_site")
        if method != "mps_d32":
            _plot_method(offset_axis, rows, method, "energy_above_mps_d32_per_site")

    common = {
        "xlabel": r"$J_2/J_1$",
        "xlim": (-0.04, 1.04),
        "xticks": (0.0, 0.25, 0.5, 0.75, 1.0),
        "grid": False,
    }
    for axis in (energy_axis, offset_axis):
        axis.set_xlabel(common["xlabel"])
        axis.set_xlim(*common["xlim"])
        axis.set_xticks(common["xticks"])
    energy_axis.set_ylabel(r"$E/N$")
    offset_axis.set_ylabel(r"$[E-E_{\mathrm{MPS}\,D=32}]/N$")
    offset_axis.axhline(0.0, color="#202020", linewidth=0.8, linestyle=":")
    offset_axis.set_ylim(-0.0045, 0.043)
    offset_axis.set_yticks((-0.004, 0.0, 0.01, 0.02, 0.03, 0.04))
    for axis in (energy_axis, offset_axis):
        axis.yaxis.set_label_position("left")
        axis.yaxis.tick_left()
        axis.tick_params(axis="y", labelleft=True, labelright=False)
        axis.grid(color="#dddddd", linewidth=0.48, which="major")
        axis.tick_params(which="both", direction="out")
    offset_axis.yaxis.set_label_position("right")
    for label, axis in zip("ab", (energy_axis, offset_axis)):
        _panel_label(axis, label)

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.2,
            markerfacecolor="white" if style["open_marker"] else style["color"],
            markeredgewidth=0.9,
            linewidth=1.45,
            label=style["label"],
        )
        for style in METHODS.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#8c2d04",
            linestyle="None",
            marker="x",
            markersize=4.8,
            markeredgewidth=1.2,
            label=r"LETTA not at $10^{-6}$/site criterion",
        )
    )
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.4,
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manuscript-output",
        type=Path,
        default=MANUSCRIPT_FIGURES / "heisenberg_6x6_j1_letta_scan",
    )
    args = parser.parse_args()
    rows = _result_rows()
    write_data(args.output.with_suffix(".json"), rows)
    plot_scan(rows, args.output)
    if args.manuscript_output:
        write_data(args.manuscript_output.with_suffix(".json"), rows)
        plot_scan(rows, args.manuscript_output)


if __name__ == "__main__":
    main()
