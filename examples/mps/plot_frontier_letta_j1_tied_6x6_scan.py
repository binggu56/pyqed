#!/usr/bin/env python3
"""Plot the converged 6x6 MPS and projected-U(1) J1-tied LETTA scan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_OUTPUT = RESULTS / "frontier_letta_j1_tied_6x6_scan"
DEFAULT_DATA = DEFAULT_OUTPUT.with_suffix(".csv")
MANUSCRIPT_FIGURES = (
    HERE.parents[1] / "docs" / "variational_letta_latex" / "figures"
)

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
        "label": r"LETTA $D=4$ (4,008)",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "s",
        "open_marker": False,
    },
}


def _display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(HERE.parents[1]))
    except ValueError:
        return str(path)


def _as_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def _result_rows(path=DEFAULT_DATA):
    path = Path(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for result in csv.DictReader(stream):
            method = result["method"]
            if method not in METHODS:
                raise ValueError(f"unknown method in {path}: {method}")
            energy = float(result["energy"])
            gain_per_site = float(result["final_gain_per_site"])
            rows.append(
                {
                    "j2_ratio": float(result["j2_ratio"]),
                    "method": method,
                    "energy": energy,
                    "energy_per_site": float(result["energy_per_site"]),
                    "parameters": int(result["parameters"]),
                    "converged": _as_bool(result["converged"]),
                    "directional_passes_completed": int(
                        result["directional_passes_completed"]
                    ),
                    "final_delta_energy": 36.0 * gain_per_site,
                    "final_delta_energy_per_site": gain_per_site,
                    "cycle_gain_per_site": gain_per_site,
                    "source_run": result["source_run"],
                    "source_file": _display_path(path),
                }
            )

    grids = {
        method: {
            row["j2_ratio"] for row in rows if row["method"] == method
        }
        for method in METHODS
    }
    if any(not grid for grid in grids.values()):
        raise ValueError(f"scan is missing a plotted method: {grids}")
    reference_grid = grids["mps_d32"]
    if any(grid != reference_grid for grid in grids.values()):
        raise ValueError(f"methods do not share a common J2 grid: {grids}")

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


def write_data(path, rows, source_data=DEFAULT_DATA):
    ratios = sorted({row["j2_ratio"] for row in rows})
    payload = {
        "model": {
            "nrows": 6,
            "ncols": 6,
            "j1": 1.0,
            "j2_ratios": ratios,
            "boundary": "open",
            "site_order": "row-wise snake",
            "letta_tie_graph": "all nearest-neighbor J1 bonds",
        },
        "letta_protocol_note": (
            "LETTA rows use exact variation-after-projection U(1), retain all "
            "unrestricted A-tensor coordinates, and optimize with directional "
            "one-site sweeps and exact identity-block frontier contractions."
        ),
        "mps_protocol_note": (
            "MPS rows use unrestricted progressive-D two-site DMRG. Every "
            "D=4,8,16,32 stage reached its requested actual bond dimension."
        ),
        "convergence_note": (
            "Converged points require two directional-cycle gains per site "
            "below 1e-6. LETTA points at J2/J1=0.8,0.9,1.0 reached the "
            "200-pass cap and are marked separately."
        ),
        "reference_note": (
            "MPS D=32 is a finite-D reference, not an exact ground-state energy."
        ),
        "source_data": _display_path(source_data),
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
    figure, energy_axis = plt.subplots(
        1, 1, figsize=(3.55, 2.65), constrained_layout=False
    )
    figure.subplots_adjust(
        left=0.17,
        right=0.98,
        bottom=0.34,
        top=0.97,
    )

    for method in METHODS:
        _plot_method(energy_axis, rows, method, "energy_per_site")

    energy_axis.set_xlabel(r"$J_2/J_1$")
    energy_axis.set_xlim(-0.04, 1.04)
    energy_axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    energy_axis.set_ylabel(r"$E/N$")
    energy_axis.yaxis.set_label_position("left")
    energy_axis.yaxis.tick_left()
    energy_axis.tick_params(axis="y", labelleft=True, labelright=False)
    energy_axis.grid(color="#dddddd", linewidth=0.48, which="major")
    energy_axis.tick_params(which="both", direction="out")

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
    # Matplotlib fills multi-row legends column-first. This ordering keeps
    # increasing MPS dimensions on the first row and MPS/LETTA on the second.
    handles = [handles[0], handles[2], handles[1], handles[3]]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.4,
    )

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manuscript-output",
        type=Path,
        default=MANUSCRIPT_FIGURES / "heisenberg_6x6_j1_letta_scan",
    )
    args = parser.parse_args()
    rows = _result_rows(args.data)
    write_data(args.output.with_suffix(".json"), rows, args.data)
    plot_scan(rows, args.output)
    if args.manuscript_output:
        write_data(
            args.manuscript_output.with_suffix(".json"),
            rows,
            args.data,
        )
        plot_scan(rows, args.manuscript_output)


if __name__ == "__main__":
    main()
