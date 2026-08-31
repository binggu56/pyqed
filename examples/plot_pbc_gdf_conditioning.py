"""Plot periodic-GDF metric conditioning and PySCF factor discrepancies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#5B5B5B"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_metric_conditioning.png"),
    )
    args = parser.parse_args()

    payload = json.loads(args.input.expanduser().read_text(encoding="utf-8"))
    row = payload["studies"][0]
    qrows = row["q_blocks"]
    worst = max(qrows, key=lambda item: item["metric_condition_number"])
    eigenvalues = np.asarray(worst["metric_eigenvalues"], dtype=float)
    threshold = float(worst["metric_eigenvalue_threshold"])
    conditioning = np.asarray(
        [item["metric_whitening_condition_number"] for item in qrows]
    )
    errors = np.asarray(
        [item["pair_metric_relative_error"] for item in qrows]
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 360,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    indices = np.arange(1, len(eigenvalues) + 1)
    axes[0].semilogy(indices, eigenvalues, "o", ms=3.2, color=BLUE)
    axes[0].axhline(
        threshold,
        color=ORANGE,
        ls="--",
        label=r"retention threshold",
    )
    axes[0].set(
        xlabel="Metric eigenvalue index",
        ylabel=r"$\lambda_a$ (a.u.)",
        title="a  Worst-q auxiliary metric",
    )
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].loglog(conditioning, errors, "o", ms=4.5, color=ORANGE)
    axes[1].set(
        xlabel=r"$\kappa(J_{\mathrm{ret}}^{-1/2})$",
        ylabel="Pair-metric relative difference",
        title="b  Whitening sensitivity",
    )
    axes[1].set_xticks(
        [2.0e6, 5.0e6, 1.0e7],
        [r"$2\times10^6$", r"$5\times10^6$", r"$10^7$"],
    )
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    for axis in axes:
        axis.grid(alpha=0.2, lw=0.6, which="both")
        axis.spines[["top", "right"]].set_visible(False)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"figure: {output}")
    print(f"pdf: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
