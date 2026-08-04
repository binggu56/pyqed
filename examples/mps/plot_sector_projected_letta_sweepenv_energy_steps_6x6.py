#!/usr/bin/env python3
"""Plot energy versus directional sweep step for projected 6x6 LETTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_INPUT = (
    RESULTS
    / "frontier_letta_sector_projected_u1_two_site_speed_sweepenv_coldonly_6x6_j2_0p5.json"
)
DEFAULT_OUTPUT = (
    RESULTS
    / "frontier_letta_sector_projected_u1_two_site_sweepenv_energy_steps_6x6_j2_0p5"
)


def _energy_trace(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["directional_passes"]
    if not rows:
        raise ValueError("the result contains no directional passes.")
    energies = np.asarray(
        [float(rows[0]["energy_before"])]
        + [float(row["energy"]) for row in rows]
    )
    completed_before = int(payload["protocol"]["starting_directional_sweep"])
    directions = ["checkpoint"] + [
        r"$L\!\to\!R$"
        if row["direction"] == "left_to_right"
        else r"$R\!\to\!L$"
        for row in rows
    ]
    completed_passes = completed_before + np.arange(energies.size)
    return completed_passes, energies, directions


def plot_energy_steps(input_path, output_stem):
    steps, energies, directions = _energy_trace(input_path)
    uplt.rc.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "tick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = uplt.subplots(refwidth=3.35, refheight=2.35)
    axis.plot(
        steps,
        energies,
        color="#0072B2",
        linewidth=1.5,
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.1,
    )
    axis.format(
        xlabel=r"directional passes completed, $q$",
        ylabel=r"total energy, $E$",
        xlim=(steps[0] - 0.18, steps[-1] + 0.18),
        grid=False,
    )
    axis.set_xticks(steps)
    axis.set_xticklabels(
        [f"{step}\n{direction}" for step, direction in zip(steps, directions)]
    )
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.tick_params(which="both", direction="out")

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    uplt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_energy_steps(args.input, args.output)


if __name__ == "__main__":
    main()
