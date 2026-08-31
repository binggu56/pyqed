#!/usr/bin/env python3
"""Plot low-rank MPS errors from a spin-boson parameter scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_row(path):
    metadata = json.loads(
        (path / "physical_metadata.json").read_text(encoding="utf-8")
    )
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    row = next(item for item in summary["cases"] if item["case"] == "mps_d4")
    return {
        "path": str(path),
        "s": float(metadata["s"]),
        "alpha": float(metadata["alpha"]),
        "delta": float(metadata["Delta"]),
        "population_error": float(row["max_sigma_z_error"]),
        "coherence_error": float(row["max_rho01_error"]),
    }


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = sorted(
        (load_row(path) for path in args.inputs),
        key=lambda row: (row["s"], row["alpha"], row["delta"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sbm_mps_difficulty_scan.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    colors = {0.3: "tab:blue", 0.5: "tab:orange", 0.8: "tab:green"}
    markers = ["o", "s", "^", "D"]
    groups = sorted({(row["s"], row["alpha"]) for row in rows})
    for group_index, (exponent, coupling) in enumerate(groups):
        selected = [
            row
            for row in rows
            if row["s"] == exponent and row["alpha"] == coupling
        ]
        if not selected:
            continue
        label = rf"$s={exponent:g}$, $\alpha={coupling:g}$"
        for axis, field in zip(
            axes, ("population_error", "coherence_error")
        ):
            axis.semilogy(
                [row["delta"] for row in selected],
                [row[field] for row in selected],
                marker=markers[group_index % len(markers)],
                color=colors[exponent],
                linewidth=1.8,
                markersize=7,
                label=label,
            )

    winner = max(rows, key=lambda row: row["population_error"])
    for axis, field in zip(axes, ("population_error", "coherence_error")):
        axis.scatter(
            [winner["delta"]],
            [winner[field]],
            s=180,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            zorder=5,
        )
        axis.annotate(
            "validated candidate",
            (winner["delta"], winner[field]),
            xytext=(-8, 12),
            textcoords="offset points",
            ha="right",
            color="red",
        )
        axis.set(xlabel=r"tunneling $\Delta$", ylabel="maximum error")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_title(r"MPS $D=4$: population error")
    axes[1].set_title(r"MPS $D=4$: coherence error")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(r"Spin–boson difficulty scan ($t_{\max}=15$, reference MPS $D=12$)")
    fig.savefig(args.output / "sbm_mps_difficulty_scan.png", dpi=180)
    fig.savefig(args.output / "sbm_mps_difficulty_scan.pdf")
    plt.close(fig)

    print(json.dumps(winner, indent=2))
    print(args.output)


if __name__ == "__main__":
    cli()
