#!/usr/bin/env python3
"""Compare the saved N=5, 7, and 9 dynamical Wilson-DVR ED runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_INPUTS = (
    RESULTS / "dynamical_schwinger_dvr_n5" / "dynamical_schwinger_data.json",
    RESULTS / "dynamical_schwinger_dvr" / "dynamical_schwinger_data.json",
    RESULTS / "dynamical_schwinger_dvr_n9" / "dynamical_schwinger_data.json",
)
DEFAULT_OUTPUT = RESULTS / "dynamical_schwinger_dvr_n9" / (
    "09_dynamical_spatial_convergence.png"
)


def load_points(paths):
    points = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        record = max(
            payload["flux_convergence"],
            key=lambda item: item["flux_cutoff"],
        )
        points.append(
            {
                "npts": payload["parameters"]["npts"],
                **record,
            }
        )
    return sorted(points, key=lambda item: item["npts"])


def plot(points, output):
    npts = np.asarray([point["npts"] for point in points])
    vector = np.asarray([point["vector_gap"] for point in points])
    scalar = np.asarray([point["scalar_gap"] for point in points])
    dimensions = np.asarray([point["dimension"] for point in points])
    seconds = np.asarray([point["seconds"] for point in points])
    exact_vector = 1.0 / np.sqrt(np.pi)
    exact_scalar = 2.0 / np.sqrt(np.pi)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)
    axes[0].plot(npts, vector, "o-", label=r"$M_V/g$")
    axes[0].plot(npts, scalar, "s-", label=r"$M_S/g$")
    axes[0].axhline(exact_vector, color="C0", linestyle="--", alpha=0.65)
    axes[0].axhline(exact_scalar, color="C1", linestyle="--", alpha=0.65)
    axes[0].set_ylabel("dimensionless mass gap")
    axes[0].set_title("Spatial-cutoff dependence")
    axes[0].legend(frameon=False)

    axes[1].semilogy(npts, np.abs(vector - exact_vector), "o-", label=r"$M_V$")
    axes[1].semilogy(npts, np.abs(scalar - exact_scalar), "s-", label=r"$M_S$")
    axes[1].set_ylabel("absolute error from continuum")
    axes[1].set_title("No smooth extrapolation yet")
    axes[1].legend(frameon=False)

    axes[2].semilogy(npts, dimensions, "o-", color="C2", label="basis dimension")
    time_axis = axes[2].twinx()
    time_axis.semilogy(npts, seconds, "s--", color="C3", label="wall time")
    axes[2].set_ylabel("physical-basis dimension", color="C2")
    time_axis.set_ylabel("wall time (s)", color="C3")
    axes[2].set_title("Exact-diagonalization cost")
    lines = axes[2].lines + time_axis.lines
    axes[2].legend(lines, [line.get_label() for line in lines], frameon=False)

    for axis in axes:
        axis.set_xlabel("DVR points $N$")
        axis.set_xticks(npts)
        axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    time_axis.spines["top"].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    points = load_points(args.inputs)
    plot(points, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
