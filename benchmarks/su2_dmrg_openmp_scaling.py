"""Benchmark dependency-wave OpenMP in reduced-space SU(2) contractions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment


def _packed_pool(arrays):
    arrays = tuple(np.ascontiguousarray(value, dtype=float) for value in arrays)
    return (
        np.concatenate([value.reshape(-1) for value in arrays]),
        np.cumsum([0, *(value.size for value in arrays)], dtype=np.int64),
        np.cumsum([0, *(value.ndim for value in arrays)], dtype=np.int64),
        np.asarray(
            [dimension for value in arrays for dimension in value.shape],
            dtype=np.int64,
        ),
    )


def _raw_source(boundary, local_operator):
    boundary_pool = _packed_pool((boundary,))
    operator_pool = _packed_pool((local_operator,))
    return (
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        boundary_pool[1], boundary_pool[2], boundary_pool[3], boundary_pool[0],
        operator_pool[1], operator_pool[2], operator_pool[3], operator_pool[0],
    )


def _wave_environment(sector_dim, sectors, rng):
    block_dimension = sector_dim * sector_dim
    environment = SU2MovingEnvironment(
        np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), 2
    )
    left_source = _raw_source(
        rng.normal(size=(2, sector_dim, sector_dim)),
        rng.normal(size=(2, 1, 1, 1)),
    )
    right_source = _raw_source(
        rng.normal(size=(2, sector_dim, sector_dim)),
        rng.normal(size=(1, 2, 1, 1)),
    )
    indices = np.arange(sectors, dtype=np.int32)
    environment.install_raw_factor_routes(
        "scaling",
        indices,
        indices,
        np.zeros(sectors, dtype=np.int64),
        np.zeros(sectors, dtype=np.int64),
        block_dimension * np.arange(sectors, dtype=np.int64),
        np.tile([sector_dim, 1, 1, sector_dim], (sectors, 1)),
        np.zeros(1, dtype=np.int64),
        left_source,
        np.zeros(1, dtype=np.int64),
        right_source,
        block_dimension * sectors,
        701,
        702,
    )
    return environment, block_dimension * sectors


def benchmark(
    sector_dim=20,
    sectors=24,
    repeats=15,
    calls_per_sample=20,
    thread_counts=(1, 2, 4, 8),
):
    rng = np.random.default_rng(23)
    environment, dimension = _wave_environment(sector_dim, sectors, rng)
    vector = rng.normal(size=dimension)
    environment.set_num_threads(1)
    reference = environment.factor_route_real_matvec("scaling", vector)

    rows = []
    for requested in thread_counts:
        environment.set_num_threads(requested)
        environment.factor_route_real_matvec("scaling", vector)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            for _ in range(calls_per_sample):
                result = environment.factor_route_real_matvec("scaling", vector)
            samples.append(
                (time.perf_counter() - started) / calls_per_sample
            )
        np.testing.assert_allclose(result, reference, rtol=2.0e-14, atol=2.0e-14)
        info = environment.threading_info
        rows.append(
            {
                "requested_threads": requested,
                "actual_threads": info["n_threads"],
                "median_seconds": float(np.median(samples)),
                "min_seconds": float(np.min(samples)),
            }
        )
    baseline = rows[0]["median_seconds"]
    for row in rows:
        row["speedup"] = baseline / row["median_seconds"]
    max_threads = max(row["actual_threads"] for row in rows)
    return {
        "sector_dim": sector_dim,
        "sectors": sectors,
        "reduced_dimension": dimension,
        "repeats": repeats,
        "calls_per_sample": calls_per_sample,
        "scheduler": {
            key: environment.stats[key]
            for key in (
                "dense_pair_scheduler",
                "dense_pair_execution_count",
                "dense_pair_wave_count",
                "dense_pair_max_wave_width",
                "dense_pair_thread_workspace_bytes",
            )
        },
        "full_output_scratch": {
            "dependency_waves_bytes": 0,
            "previous_thread_reduction_bytes": max_threads * dimension * 8,
        },
        "threading": info,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector-dim", type=int, default=20)
    parser.add_argument("--sectors", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--calls-per-sample", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/pyqed_su2_openmp_scaling.json"))
    parser.add_argument("--figure", type=Path, default=Path("/private/tmp/pyqed_su2_openmp_scaling.png"))
    args = parser.parse_args()
    data = benchmark(
        args.sector_dim,
        args.sectors,
        args.repeats,
        args.calls_per_sample,
    )
    args.output.write_text(json.dumps(data, indent=2) + "\n")

    threads = [row["actual_threads"] for row in data["rows"]]
    speedups = [row["speedup"] for row in data["rows"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(threads, speedups, "o-", linewidth=2, label="measured")
    ax.plot(threads, threads, "--", color="0.55", label="ideal")
    ax.set(
        xlabel="SU(2) OpenMP threads",
        ylabel="speedup",
        title="SU(2) dependency-wave contraction scaling",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
