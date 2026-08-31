#!/usr/bin/env python3
"""Plot NumPy/PyTorch LETTA trajectories and Torch thread scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_result(path):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    row = next(item for item in summary["cases"] if item["backend"] == "letta")
    table = np.genfromtxt(
        path / row["case"] / "TDVP_observables.csv", delimiter=",", names=True
    )
    metadata = json.loads(
        (path / row["case"] / "TDVP_metadata.json").read_text(encoding="utf-8")
    )
    return row, table, metadata


def _load_thread_run(path):
    metadata = json.loads(
        (path / "letta_d2" / "TDVP_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "threads": int(metadata["tensor_threads"]),
        "wall_seconds": float(metadata["wall_seconds"]),
        "mean_seconds_per_step": float(metadata["mean_seconds_per_step"]),
    }


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy-result", type=Path, required=True)
    parser.add_argument("--baseline-numpy-result", type=Path)
    parser.add_argument("--torch-result", type=Path, required=True)
    parser.add_argument("--baseline-torch-result", type=Path)
    parser.add_argument("--thread-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    numpy_row, numpy_table, _ = _load_result(args.numpy_result)
    torch_row, torch_table, torch_metadata = _load_result(args.torch_result)
    baseline_row = None
    if args.baseline_torch_result is not None:
        baseline_row, _, _ = _load_result(args.baseline_torch_result)
    baseline_numpy_row = None
    if args.baseline_numpy_result is not None:
        baseline_numpy_row, _, _ = _load_result(args.baseline_numpy_result)
    if not np.array_equal(numpy_table["time"], torch_table["time"]):
        raise ValueError("NumPy and Torch trajectories use different time grids.")
    thread_rows = sorted((_load_thread_run(path) for path in args.thread_runs), key=lambda x: x["threads"])

    numpy_rho = numpy_table["rho01_real"] + 1.0j * numpy_table["rho01_imag"]
    torch_rho = torch_table["rho01_real"] + 1.0j * torch_table["rho01_imag"]
    sigma_difference = np.abs(numpy_table["sigma_z"] - torch_table["sigma_z"])
    rho_difference = np.abs(numpy_rho - torch_rho)
    numpy_wall = float(numpy_row["wall_seconds"])
    torch_wall = float(torch_row["wall_seconds"])

    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "numpy_wall_seconds": numpy_wall,
        "torch_wall_seconds": torch_wall,
        "torch_speedup": numpy_wall / torch_wall,
        "max_sigma_z_backend_difference": float(np.max(sigma_difference)),
        "max_rho01_backend_difference": float(np.max(rho_difference)),
        "thread_scaling": thread_rows,
    }
    if baseline_row is not None:
        result["baseline_torch_wall_seconds"] = float(
            baseline_row["wall_seconds"]
        )
        result["kernel_optimization_speedup"] = (
            float(baseline_row["wall_seconds"]) / torch_wall
        )
    if baseline_numpy_row is not None:
        result["baseline_numpy_wall_seconds"] = float(
            baseline_numpy_row["wall_seconds"]
        )
        result["numpy_kernel_optimization_speedup"] = (
            float(baseline_numpy_row["wall_seconds"]) / numpy_wall
        )
    (args.output / "letta_backend_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
    time = numpy_table["time"]
    axes[0, 0].plot(time, numpy_table["sigma_z"], label="NumPy", linewidth=2.2)
    thread_count = int(torch_metadata.get("tensor_threads") or 1)
    thread_label = "thread" if thread_count == 1 else "threads"
    axes[0, 0].plot(
        time,
        torch_table["sigma_z"],
        "--",
        label=f"PyTorch ({thread_count} {thread_label})",
        linewidth=1.8,
    )
    axes[0, 0].set(
        title=r"SBM dynamics ($N=20$, $d=12$, $D=2$)",
        xlabel="time",
        ylabel=r"$\langle\sigma_z\rangle$",
    )
    axes[0, 0].legend(frameon=False)

    axes[0, 1].semilogy(time, np.maximum(sigma_difference, 1.0e-17), label=r"$\sigma_z$")
    axes[0, 1].semilogy(time, np.maximum(rho_difference, 1.0e-17), label=r"$\rho_{01}$")
    axes[0, 1].set(
        title="Cross-backend numerical difference",
        xlabel="time",
        ylabel="absolute difference",
    )
    axes[0, 1].legend(frameon=False)

    labels = ["Optimized\nNumPy", "Optimized\nPyTorch"]
    walls = [numpy_wall, torch_wall]
    colors = ["tab:blue", "tab:orange"]
    if baseline_row is not None:
        labels.insert(1, "Original\nPyTorch")
        walls.insert(1, float(baseline_row["wall_seconds"]))
        colors.insert(1, "tab:gray")
    if baseline_numpy_row is not None:
        labels.insert(0, "Original\nNumPy")
        walls.insert(0, float(baseline_numpy_row["wall_seconds"]))
        colors.insert(0, "tab:cyan")
    bars = axes[1, 0].bar(labels, walls, color=colors)
    axes[1, 0].bar_label(bars, fmt="%.1f s", padding=3)
    if torch_wall < numpy_wall:
        runtime_title = f"Full trajectory: Torch {numpy_wall / torch_wall:.2f}x faster"
    else:
        runtime_title = f"Full trajectory: NumPy {torch_wall / numpy_wall:.2f}x faster"
    axes[1, 0].set(title=runtime_title, ylabel="wall time (s)")

    threads = [row["threads"] for row in thread_rows]
    walls = [row["wall_seconds"] for row in thread_rows]
    axes[1, 1].plot(threads, walls, "o-", color="tab:orange")
    axes[1, 1].set(
        title="Two-step TDVP2 thread scaling",
        xlabel="Torch CPU threads",
        ylabel="wall time (s)",
        xticks=threads,
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25, which="both")
    fig.savefig(args.output / "letta_backend_benchmark.png", dpi=180)
    fig.savefig(args.output / "letta_backend_benchmark.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    cli()
