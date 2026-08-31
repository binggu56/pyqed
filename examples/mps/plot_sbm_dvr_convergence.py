#!/usr/bin/env python3
"""Plot local-DVR convergence of spin-boson observables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_case(path):
    table = np.genfromtxt(
        path / "mps_d16" / "TDVP_observables.csv", delimiter=",", names=True
    )
    metadata = json.loads(
        (path / "mps_d16" / "TDVP_metadata.json").read_text(encoding="utf-8")
    )
    return table, int(metadata["d"])


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cases = sorted((load_case(path) for path in args.results), key=lambda item: item[1])
    if any(not np.array_equal(cases[0][0]["time"], case[0]["time"]) for case in cases[1:]):
        raise ValueError("DVR convergence cases use different time grids.")
    comparisons = []
    for (left, d_left), (right, d_right) in zip(cases[:-1], cases[1:]):
        left_rho = left["rho01_real"] + 1.0j * left["rho01_imag"]
        right_rho = right["rho01_real"] + 1.0j * right["rho01_imag"]
        comparisons.append(
            {
                "label": f"$d={d_left}\\to{d_right}$",
                "d_left": d_left,
                "d_right": d_right,
                "sigma": np.abs(left["sigma_z"] - right["sigma_z"]),
                "rho": np.abs(left_rho - right_rho),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "d_left": item["d_left"],
            "d_right": item["d_right"],
            "max_sigma_z_difference": float(np.max(item["sigma"])),
            "max_rho01_difference": float(np.max(item["rho"])),
        }
        for item in comparisons
    ]
    (args.output / "sbm_dvr_convergence.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    time = cases[0][0]["time"]
    for item in comparisons:
        axes[0].semilogy(time, np.maximum(item["sigma"], 1.0e-16), label=item["label"])
        axes[1].semilogy(time, np.maximum(item["rho"], 1.0e-16), label=item["label"])
    axes[0].set(
        title=r"Population convergence",
        xlabel="time",
        ylabel=r"$|\Delta\langle\sigma_z\rangle|$",
    )
    axes[1].set(
        title=r"Coherence convergence",
        xlabel="time",
        ylabel=r"$|\Delta\rho_{01}|$",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(frameon=False)
    fig.suptitle(r"Local-DVR convergence at MPS $D=16$")
    fig.savefig(args.output / "sbm_dvr_convergence.png", dpi=180)
    fig.savefig(args.output / "sbm_dvr_convergence.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    cli()
