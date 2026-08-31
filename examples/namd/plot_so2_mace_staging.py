"""Compare gauge-covariant SO2 MACE training strategies from JSON metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [path.stem for path in args.metrics]
    if len(labels) != len(args.metrics):
        raise ValueError("--labels must match the number of metric files")
    records = [json.loads(path.read_text()) for path in args.metrics]
    values = np.asarray(
        [
            [
                record["energy_relative_error"]["held_out"],
                *(record["link_relative_error"][str(axis)]["held_out"] for axis in range(3)),
            ]
            for record in records
        ]
    )
    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    positions = np.arange(4)
    width = 0.8 / len(records)
    for index, (label, row) in enumerate(zip(labels, values)):
        axis.bar(
            positions + (index - 0.5 * (len(records) - 1)) * width,
            row,
            width=width,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xticks(positions, (r"$H$", r"$L_{r_1}$", r"$L_{r_2}$", r"$L_\theta$"))
    axis.set_ylabel("Held-out relative Frobenius error")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
