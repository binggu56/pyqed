"""Benchmark batched NumPy and C++ native-LETTA conditional gauges."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from pyqed.letta.conditional_gauge import (
    CONDITIONAL_GAUGE_CPP_AVAILABLE,
    apply_conditional_gauges,
)


def _case(bond_dim, *, group_size, dtype, seed):
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(bond_dim, 2, 2, bond_dim)).astype(dtype)
    right = rng.normal(size=(bond_dim, 2, 2, bond_dim)).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        left += 1j * rng.normal(size=left.shape)
        right += 1j * rng.normal(size=right.shape)

    transforms = []
    for shared_state in range(2):
        for start in range(0, bond_dim, group_size):
            stop = min(start + group_size, bond_dim)
            group = np.arange(start, stop, dtype=np.intp)
            matrix = rng.normal(size=(group.size, group.size))
            if np.issubdtype(np.dtype(dtype), np.complexfloating):
                matrix = matrix + 1j * rng.normal(size=matrix.shape)
            q, _ = np.linalg.qr(matrix)
            transforms.append((shared_state, group, q, q.T.conj()))
    return left, right, tuple(transforms)


def _timed(left, right, transforms, *, backend, repeats):
    work_left = left.copy()
    work_right = right.copy()
    work_left, work_right = apply_conditional_gauges(
        work_left,
        work_right,
        transforms,
        backend=backend,
    )
    start = perf_counter()
    for _ in range(repeats):
        work_left, work_right = apply_conditional_gauges(
            work_left,
            work_right,
            transforms,
            backend=backend,
        )
    elapsed = perf_counter() - start
    return elapsed / repeats


def benchmark(bond_dims, *, group_size, repeats, dtype):
    label = np.dtype(dtype).name
    print(f"dtype={label} group_size={group_size} repeats={repeats}")
    print("D,numpy_us,cpp_us,speedup")
    for bond_dim in bond_dims:
        left, right, transforms = _case(
            bond_dim,
            group_size=group_size,
            dtype=dtype,
            seed=100 + bond_dim,
        )
        numpy_time = _timed(
            left,
            right,
            transforms,
            backend="numpy",
            repeats=repeats,
        )
        if CONDITIONAL_GAUGE_CPP_AVAILABLE:
            cpp_left, cpp_right = apply_conditional_gauges(
                left.copy(),
                right.copy(),
                transforms,
                backend="cpp",
            )
            numpy_left, numpy_right = apply_conditional_gauges(
                left.copy(),
                right.copy(),
                transforms,
                backend="numpy",
            )
            np.testing.assert_allclose(cpp_left, numpy_left, rtol=2.0e-13, atol=2.0e-13)
            np.testing.assert_allclose(cpp_right, numpy_right, rtol=2.0e-13, atol=2.0e-13)
            cpp_time = _timed(
                left,
                right,
                transforms,
                backend="cpp",
                repeats=repeats,
            )
            print(
                f"{bond_dim},{1.0e6 * numpy_time:.3f},"
                f"{1.0e6 * cpp_time:.3f},{numpy_time / cpp_time:.3f}"
            )
        else:
            print(f"{bond_dim},{1.0e6 * numpy_time:.3f},unavailable,unavailable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond-dims", nargs="+", type=int, default=(4, 8, 16, 32))
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument(
        "--dtype",
        choices=("float64", "complex128"),
        default="float64",
    )
    args = parser.parse_args()
    benchmark(
        args.bond_dims,
        group_size=args.group_size,
        repeats=args.repeats,
        dtype=np.dtype(args.dtype),
    )


if __name__ == "__main__":
    main()
