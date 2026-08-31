#!/usr/bin/env python3
"""Benchmark native dense-DMRG OpenMP matvec scaling and plot the result."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps import cpp_davidson


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def benchmark(*, bond_dim, physical_dim, mpo_dim, threads, repeats, seed):
    if not cpp_davidson.CPP_DAVIDSON_AVAILABLE:
        raise RuntimeError(cpp_davidson.CPP_DAVIDSON_BUILD_ERROR)
    if not callable(cpp_davidson.openmp_available) or not cpp_davidson.openmp_available():
        raise RuntimeError("the native DMRG backend was built without OpenMP")

    rng = np.random.default_rng(seed)
    e = random_complex(rng, (mpo_dim, bond_dim, bond_dim))
    w = random_complex(rng, (mpo_dim, mpo_dim, physical_dim, physical_dim))
    f = random_complex(rng, (mpo_dim, bond_dim, bond_dim))
    v = random_complex(rng, bond_dim * physical_dim * bond_dim)
    workspace = cpp_davidson.DenseDavidsonWorkspace()
    workspace.bind(e, w, f)

    rows = []
    reference = None
    for count in threads:
        cpp_davidson.set_num_threads(count)
        workspace.matvec(v, "openmp")
        samples = []
        result = None
        for _ in range(repeats):
            started = time.perf_counter()
            result = np.asarray(workspace.matvec(v, "openmp"))
            samples.append(time.perf_counter() - started)
        if reference is None:
            reference = result
        else:
            np.testing.assert_allclose(result, reference, rtol=1.0e-12, atol=1.0e-12)
        rows.append(
            {
                "threads": count,
                "seconds": float(np.median(samples)),
                "samples": [float(value) for value in samples],
            }
        )
    baseline = rows[0]["seconds"]
    for row in rows:
        row["speedup"] = baseline / row["seconds"]
        row["efficiency"] = row["speedup"] / row["threads"]
    return rows


def plot(rows, output):
    counts = [row["threads"] for row in rows]
    seconds = [row["seconds"] for row in rows]
    speedups = [row["speedup"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(counts, seconds, "o-", color="#2b6cb0")
    axes[0].set(xlabel="OpenMP threads", ylabel="Median matvec time (s)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(counts, counts, "--", color="0.65", label="ideal")
    axes[1].plot(counts, speedups, "o-", color="#c05621", label="measured")
    axes[1].set(xlabel="OpenMP threads", ylabel="Speedup")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("PyQED dense-DMRG effective-Hamiltonian matvec")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond-dim", type=int, default=12)
    parser.add_argument("--physical-dim", type=int, default=4)
    parser.add_argument("--mpo-dim", type=int, default=3)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path, default=Path("dmrg_openmp_scaling.json"))
    parser.add_argument("--figure", type=Path, default=Path("dmrg_openmp_scaling.png"))
    args = parser.parse_args()
    thread_counts = [int(value) for value in args.threads.split(",")]
    rows = benchmark(
        bond_dim=args.bond_dim,
        physical_dim=args.physical_dim,
        mpo_dim=args.mpo_dim,
        threads=thread_counts,
        repeats=args.repeats,
        seed=args.seed,
    )
    payload = {
        "problem": {
            "bond_dim": args.bond_dim,
            "physical_dim": args.physical_dim,
            "mpo_dim": args.mpo_dim,
            "repeats": args.repeats,
        },
        "openmp": dict(cpp_davidson.openmp_info()),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    plot(rows, args.figure)
    print(json.dumps(payload, indent=2))
    print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
