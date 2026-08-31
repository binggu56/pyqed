#!/usr/bin/env python3
"""Compare two-arm and interleaved thermofield orderings for LETTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(directory):
    directory = Path(directory)
    with (directory / "summary.json").open() as handle:
        summary = json.load(handle)
    data = np.load(directory / "trajectories.npz")
    runs = {}
    for index, record in enumerate(summary["runs"]):
        rank = int(record["rank"])
        runs[rank] = {
            **record,
            "sigma_z": np.asarray(data[f"case{index}_sigma_z"]),
        }
    return {
        "time": np.asarray(data["time"]),
        "reference": np.asarray(data["reference_sigma_z"]),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--interleaved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = _load(args.arms)
    interleaved = _load(args.interleaved)
    np.testing.assert_allclose(arms["time"], interleaved["time"])
    np.testing.assert_allclose(arms["reference"], interleaved["reference"])
    time = arms["time"]
    reference = arms["reference"]
    ranks = sorted(set(arms["runs"]) & set(interleaved["runs"]))
    colors = dict(zip(ranks, plt.cm.plasma(np.linspace(0.12, 0.78, len(ranks)))))

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.7))
    axes[0].plot(time, reference, color="black", lw=2.3, label="MPS $D=32$")
    for rank in ranks:
        color = colors[rank]
        axes[0].plot(
            time, arms["runs"][rank]["sigma_z"], color=color, lw=1.4,
            ls="--", label=rf"arms $D={rank}$",
        )
        axes[0].plot(
            time, interleaved["runs"][rank]["sigma_z"], color=color, lw=1.8,
            label=rf"interleaved $D={rank}$",
        )
        axes[1].semilogy(
            time,
            np.maximum(np.abs(arms["runs"][rank]["sigma_z"] - reference), 1.0e-15),
            color=color, lw=1.3, ls="--",
        )
        axes[1].semilogy(
            time,
            np.maximum(
                np.abs(interleaved["runs"][rank]["sigma_z"] - reference), 1.0e-15
            ),
            color=color, lw=1.8,
        )

    for label, result, style, marker in (
        ("arms", arms, "--", "o"),
        ("interleaved", interleaved, "-", "s"),
    ):
        parameters = [result["runs"][rank]["parameters"] for rank in ranks]
        errors = [
            result["runs"][rank]["max_sigma_z_error_vs_mps"] for rank in ranks
        ]
        axes[2].loglog(
            parameters, errors, color="0.25", ls=style, marker=marker,
            ms=6, label=label,
        )
        for rank, x, y in zip(ranks, parameters, errors):
            axes[2].annotate(
                rf"$D={rank}$", (x, y), xytext=(5, 4),
                textcoords="offset points", fontsize=8,
            )

    axes[0].set_title(
        r"High-$T$ thermofield SBM: LETTA ordering comparison, $N=14$, $d=24$"
    )
    axes[0].set_ylabel(r"$\langle\sigma_z\rangle$")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    axes[1].set_ylabel("absolute error vs MPS")
    axes[1].set_xlabel(r"time $t\,\omega_c$")
    axes[2].set_xlabel("stored complex tensor entries")
    axes[2].set_ylabel(r"max $|\Delta\langle\sigma_z\rangle|$")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
