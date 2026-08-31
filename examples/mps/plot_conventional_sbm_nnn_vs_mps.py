#!/usr/bin/env python3
"""Combine conventional-SBM NNN-LETTA trajectories with an MPS rank scan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("time", "sigma_z", "rho01_abs")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnn-dir", type=Path, required=True)
    parser.add_argument("--mps-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with (args.nnn_dir / "summary.json").open() as handle:
        nnn_summary = json.load(handle)
    with (args.mps_dir / "summary.json").open() as handle:
        mps_summary = json.load(handle)
    nnn = np.load(args.nnn_dir / "trajectories.npz")
    reference = read_csv(args.mps_dir / "mps_d24" / "TDVP_observables.csv")
    times = nnn["time"]
    if not np.allclose(times, reference["time"]):
        raise SystemExit("NNN and MPS trajectories use different time grids")

    nnn_rows = []
    for row in nnn_summary["runs"]:
        rank = int(row["rank"])
        sigma = nnn[f"nnn_d{rank}_sigma_z"]
        coherence = nnn[f"nnn_d{rank}_rho01_abs"]
        nnn_rows.append(
            {
                **row,
                "case": f"nnn_d{rank}",
                "backend": "nnn",
                "max_sigma_z_error": float(
                    np.max(np.abs(sigma - reference["sigma_z"]))
                ),
                "max_rho01_error": float(
                    np.max(np.abs(coherence - reference["rho01_abs"]))
                ),
            }
        )
    mps_rows = [
        {**row, "parameters": row["peak_parameters"]}
        for row in mps_summary["cases"]
    ]
    combined = {
        "model": nnn_summary["model"],
        "reference": "MPS D=24",
        "nnn": nnn_rows,
        "mps": mps_rows,
    }
    with (args.output / "summary.json").open("w") as handle:
        json.dump(combined, handle, indent=2)
    for row in nnn_rows:
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "case", "parameters", "wall_seconds",
                        "max_sigma_z_error", "max_rho01_error",
                    )
                }
            ),
            flush=True,
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    selected_mps = (4, 12, 24)
    mps_colors = {4: "#56B4E9", 12: "#0072B2", 24: "black"}
    for rank in selected_mps:
        trajectory = read_csv(
            args.mps_dir / f"mps_d{rank}" / "TDVP_observables.csv"
        )
        axes[0, 0].plot(
            times,
            trajectory["sigma_z"],
            lw=2.0 if rank == 24 else 1.2,
            color=mps_colors[rank],
            label=rf"MPS $D={rank}$",
        )
    nnn_colors = {1: "#D55E00", 2: "#009E73"}
    for row in nnn_rows:
        rank = int(row["rank"])
        sigma = nnn[f"nnn_d{rank}_sigma_z"]
        axes[0, 0].plot(
            times, sigma, lw=1.5, color=nnn_colors[rank],
            label=rf"NNN-LETTA $D={rank}$",
        )
        axes[1, 0].semilogy(
            times,
            np.maximum(np.abs(sigma - reference["sigma_z"]), 1.0e-15),
            color=nnn_colors[rank],
            lw=1.5,
            label=rf"NNN $D={rank}$",
        )
    for rank in (4, 8, 12, 16, 20):
        trajectory = read_csv(
            args.mps_dir / f"mps_d{rank}" / "TDVP_observables.csv"
        )
        axes[1, 0].semilogy(
            times,
            np.maximum(
                np.abs(trajectory["sigma_z"] - reference["sigma_z"]), 1.0e-15
            ),
            lw=1.0,
            label=rf"MPS $D={rank}$",
        )

    for row in mps_rows[:-1]:
        axes[0, 1].loglog(
            row["parameters"], row["max_sigma_z_error"], "o", color="#0072B2"
        )
        axes[1, 1].loglog(
            row["wall_seconds"], row["max_sigma_z_error"], "o", color="#0072B2"
        )
        axes[0, 1].annotate(
            f"MPS {row['rank_cap']}",
            (row["parameters"], row["max_sigma_z_error"]),
            xytext=(4, 3), textcoords="offset points", fontsize=7,
        )
    for row in nnn_rows:
        rank = int(row["rank"])
        axes[0, 1].loglog(
            row["parameters"], row["max_sigma_z_error"], "s",
            color=nnn_colors[rank], ms=7,
        )
        axes[1, 1].loglog(
            row["wall_seconds"], row["max_sigma_z_error"], "s",
            color=nnn_colors[rank], ms=7,
        )
        axes[0, 1].annotate(
            f"NNN {rank}",
            (row["parameters"], row["max_sigma_z_error"]),
            xytext=(4, 3), textcoords="offset points", fontsize=8,
        )
        axes[1, 1].annotate(
            f"NNN {rank}",
            (row["wall_seconds"], row["max_sigma_z_error"]),
            xytext=(4, 3), textcoords="offset points", fontsize=8,
        )
    axes[0, 0].set(
        title="Population dynamics", ylabel=r"$\langle\sigma_z\rangle$"
    )
    axes[1, 0].set(
        title="Error against MPS $D=24$", xlabel="time",
        ylabel=r"$|\Delta\langle\sigma_z\rangle|$",
    )
    axes[0, 1].set(
        title="Parameter efficiency", xlabel="peak complex parameters",
        ylabel="maximum population error",
    )
    axes[1, 1].set(
        title="Runtime efficiency", xlabel="wall time (s)",
        ylabel="maximum population error",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=0.2, which="both")
    fig.suptitle(
        r"Conventional SBM: $s=0.8,\ \alpha=0.5,\ \Delta=0.8$, "
        r"$N=8,\ d=12$",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(args.output / "conventional_sbm_nnn_vs_mps.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
