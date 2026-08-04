#!/usr/bin/env python3
"""Benchmark native copy-aware 6x6 frontier-message advancement."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
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


def _time(function, *, repeats, warmups):
    for _ in range(int(warmups)):
        function()
    values = []
    result = None
    for _ in range(int(repeats)):
        start = perf_counter()
        result = function()
        values.append(perf_counter() - start)
    return result, values


def _message_difference(actual, expected):
    difference_squared = 0.0
    reference_squared = 0.0
    maximum = 0.0
    for actual_message, expected_message in zip(actual, expected):
        for actual_block, expected_block in zip(
            actual_message.blocks,
            expected_message.blocks,
        ):
            if not np.size(expected_block):
                continue
            difference = np.asarray(actual_block) - np.asarray(expected_block)
            maximum = max(maximum, float(np.max(np.abs(difference))))
            difference_squared += float(np.vdot(difference, difference).real)
            reference_squared += float(
                np.vdot(expected_block, expected_block).real
            )
    return {
        "maximum_absolute_difference": maximum,
        "relative_l2_difference": float(
            np.sqrt(difference_squared)
            / max(np.sqrt(reference_squared), np.finfo(float).tiny)
        ),
    }


def benchmark(source, *, workers=4, repeats=1, warmups=1):
    source = Path(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    state = _projected_state(payload["model"], source.with_suffix(".npz"))
    engine = state._hamiltonian_frontier
    reference_right = engine.build_right(
        state.tensors,
        copy_backend="python",
    )

    def identity_messages(backend, executor):
        messages = []
        for site in range(len(state.dims) - 1):
            following = site + 1
            pair_engine = state._pair_plan(site).hamiltonian_engine
            messages.append(
                pair_engine.advance_right_identity(
                    reference_right[following + 1],
                    following,
                    max_workers=workers,
                    executor=executor,
                    copy_backend=backend,
                )
            )
        return messages

    outputs = {}
    timings = {}
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        for backend in ("python", "auto", "native"):
            outputs[(backend, "left")], left = _time(
                lambda backend=backend: engine.build_left(
                    state.tensors,
                    max_workers=workers,
                    executor=executor,
                    copy_backend=backend,
                ),
                repeats=repeats,
                warmups=warmups,
            )
            outputs[(backend, "right")], right = _time(
                lambda backend=backend: engine.build_right(
                    state.tensors,
                    max_workers=workers,
                    executor=executor,
                    copy_backend=backend,
                ),
                repeats=repeats,
                warmups=warmups,
            )
            outputs[(backend, "identity")], identity = _time(
                lambda backend=backend: identity_messages(backend, executor),
                repeats=repeats,
                warmups=warmups,
            )
            timings[backend] = {
                "left_seconds": left,
                "right_seconds": right,
                "all_pair_identity_seconds": identity,
            }

    comparisons = {}
    for backend in ("auto", "native"):
        comparisons[backend] = {
            operation: _message_difference(
                outputs[(backend, operation)],
                outputs[("python", operation)],
            )
            for operation in ("left", "right", "identity")
        }
    return {
        "source": str(source),
        "workers": int(workers),
        "repeats": int(repeats),
        "warmups": int(warmups),
        "timings": timings,
        "auto_best_speedup": {
            operation: float(
                min(timings["python"][f"{operation}_seconds"])
                / min(timings["auto"][f"{operation}_seconds"])
            )
            for operation in ("left", "right", "all_pair_identity")
        },
        "comparisons_to_python": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    options = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                options.source,
                workers=options.workers,
                repeats=options.repeats,
                warmups=options.warmups,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
