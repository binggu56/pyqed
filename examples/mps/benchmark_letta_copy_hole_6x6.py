#!/usr/bin/env python3
"""Benchmark native copy-aware pair-hole contraction against opt_einsum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.mps.converge_sector_projected_letta_two_site_6x6 import (
    _projected_state,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    HERE
    / "results"
    / "frontier_letta_sector_projected_u1_two_site_speed_sweepenv_coldonly_6x6_j2_0p5.json"
)


def benchmark(source, *, site=15, workers=4, repeats=3):
    source = Path(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    state = _projected_state(payload["model"], source.with_suffix(".npz"))
    plan = state._pair_plan(site)
    environment = state.pair_environment(site)
    engine = plan.hamiltonian_engine
    arguments = (
        site,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
    )
    options = {
        "max_workers": int(workers),
        "parallel_min_size": 1,
    }
    matrices = {}
    timings = {}
    for backend in ("python", "native"):
        engine.hole_matrix(*arguments, copy_backend=backend, **options)
        samples = []
        for _ in range(int(repeats)):
            start = perf_counter()
            matrices[backend] = engine.hole_matrix(
                *arguments,
                copy_backend=backend,
                **options,
            )
            samples.append(perf_counter() - start)
        timings[backend] = samples
    difference = matrices["native"] - matrices["python"]
    reference_norm = max(
        float(np.linalg.norm(matrices["python"])),
        np.finfo(float).tiny,
    )
    return {
        "site": int(site),
        "dimension": int(np.prod(plan.merged_shape)),
        "workers": int(workers),
        "repeats": int(repeats),
        "python_seconds": timings["python"],
        "native_seconds": timings["native"],
        "best_speedup": float(min(timings["python"]) / min(timings["native"])),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "relative_frobenius_difference": float(
            np.linalg.norm(difference) / reference_norm
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--site", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                options.source,
                site=options.site,
                workers=options.workers,
                repeats=options.repeats,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
