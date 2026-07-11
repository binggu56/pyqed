#!/usr/bin/env python3
"""Benchmark dense-A and link-only LDR kinetic assembly.

This benchmark uses synthetic nearest-neighbor electronic overlap links, so it
measures the LDR matrix-building path without running electronic structure.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy.linalg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom


def make_solver(npts, nstates):
    solver = object.__new__(Triatom)
    solver.ndim = len(npts)
    solver.nx = list(npts)
    solver.nstates = int(nstates)
    solver.J = 0
    solver.nrot = 1
    solver.overlap_matrix = None
    solver.overlap_links = None
    return solver


def random_unitary_link(nstates, rng, strength):
    z = rng.normal(size=(nstates, nstates)) + 1j * rng.normal(size=(nstates, nstates))
    antiherm = z - z.conj().T
    return scipy.linalg.expm(strength * antiherm)


def make_links(solver, rng, strength):
    links = {}
    for idx in np.ndindex(*solver.nx):
        for axis in range(solver.ndim):
            if idx[axis] + 1 >= solver.nx[axis]:
                continue
            links[(axis, idx)] = random_unitary_link(solver.nstates, rng, strength)
    return links


def make_kinetic(ngrid, rng):
    z = rng.normal(size=(ngrid, ngrid)) + 1j * rng.normal(size=(ngrid, ngrid))
    kinetic = 0.5 * (z + z.conj().T)
    kinetic += ngrid * np.eye(ngrid)
    return np.ascontiguousarray(kinetic)


def timed(label, fn):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return label, value, elapsed, peak


def fmt_bytes(nbytes):
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(nbytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def run_case(npts, nstates, seed, strength, check):
    rng = np.random.default_rng(seed)
    solver = make_solver(npts, nstates)
    ngrid = int(np.prod(solver.nx))

    links = make_links(solver, rng, strength)
    kinetic = make_kinetic(ngrid, rng)
    link_bytes = sum(mat.nbytes for mat in links.values())

    _, dense_A, dense_A_time, dense_A_peak = timed(
        "dense_A",
        lambda: solver._build_linked_overlap_from_links(links, nstates),
    )

    solver.overlap_matrix = dense_A
    solver.overlap_links = None
    _, dense_K, dense_K_time, dense_K_peak = timed(
        "dense_K",
        lambda: solver._build_flat_kinetic_matrix(kinetic),
    )

    solver.overlap_matrix = None
    solver.overlap_links = links
    _, link_K, link_K_time, link_K_peak = timed(
        "link_only_K",
        lambda: solver._build_flat_kinetic_matrix(kinetic),
    )
    vec = rng.normal(size=ngrid * nstates) + 1j * rng.normal(size=ngrid * nstates)
    _, link_op, link_op_time, link_op_peak = timed(
        "link_only_LinearOperator",
        lambda: solver._build_kinetic_linear_operator(kinetic),
    )
    _, link_op_vec, link_op_matvec_time, link_op_matvec_peak = timed(
        "link_only_LinearOperator_matvec",
        lambda: link_op @ vec,
    )
    _, dense_vec, dense_matvec_time, dense_matvec_peak = timed(
        "dense_K_matvec",
        lambda: dense_K @ vec,
    )

    max_abs_diff = np.nan
    max_matvec_diff = np.nan
    if check:
        max_abs_diff = float(np.max(np.abs(dense_K - link_K)))
        max_matvec_diff = float(np.max(np.abs(dense_vec - link_op_vec)))

    dense_A_bytes = dense_A.nbytes
    dense_K_bytes = dense_K.nbytes
    del dense_A, dense_K, link_K, link_op_vec, dense_vec
    gc.collect()

    return {
        "npts": tuple(npts),
        "ngrid": ngrid,
        "nstates": nstates,
        "nlinks": len(links),
        "link_bytes": link_bytes,
        "dense_A_bytes": dense_A_bytes,
        "dense_K_bytes": dense_K_bytes,
        "dense_A_time": dense_A_time,
        "dense_A_peak": dense_A_peak,
        "dense_K_time": dense_K_time,
        "dense_K_peak": dense_K_peak,
        "link_K_time": link_K_time,
        "link_K_peak": link_K_peak,
        "link_op_time": link_op_time,
        "link_op_peak": link_op_peak,
        "link_op_matvec_time": link_op_matvec_time,
        "link_op_matvec_peak": link_op_matvec_peak,
        "dense_matvec_time": dense_matvec_time,
        "dense_matvec_peak": dense_matvec_peak,
        "max_abs_diff": max_abs_diff,
        "max_matvec_diff": max_matvec_diff,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7],
        help="Cubic grid sizes to benchmark, e.g. 4 5 6 means 4^3, 5^3, 6^3.",
    )
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strength", type=float, default=0.02)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    print(
        "size, ngrid, nlinks, links, dense_A, dense_K, "
        "build_A_s, dense_K_s, link_only_K_s, link_op_s, "
        "dense_matvec_s, link_op_matvec_s, total_dense_s, speedup_build, "
        "peak_A, peak_dense_K, peak_link_K, peak_link_op, "
        "max_abs_diff, max_matvec_diff"
    )
    for size in args.sizes:
        result = run_case(
            (size, size, size),
            args.nstates,
            args.seed + size,
            args.strength,
            check=not args.no_check,
        )
        total_dense = result["dense_A_time"] + result["dense_K_time"]
        speedup_build = total_dense / result["link_op_time"] if result["link_op_time"] else np.inf
        print(
            f"{size}^3, {result['ngrid']}, {result['nlinks']}, "
            f"{fmt_bytes(result['link_bytes'])}, "
            f"{fmt_bytes(result['dense_A_bytes'])}, "
            f"{fmt_bytes(result['dense_K_bytes'])}, "
            f"{result['dense_A_time']:.6f}, "
            f"{result['dense_K_time']:.6f}, "
            f"{result['link_K_time']:.6f}, "
            f"{result['link_op_time']:.6f}, "
            f"{result['dense_matvec_time']:.6f}, "
            f"{result['link_op_matvec_time']:.6f}, "
            f"{total_dense:.6f}, "
            f"{speedup_build:.3f}, "
            f"{fmt_bytes(result['dense_A_peak'])}, "
            f"{fmt_bytes(result['dense_K_peak'])}, "
            f"{fmt_bytes(result['link_K_peak'])}, "
            f"{fmt_bytes(result['link_op_peak'])}, "
            f"{result['max_abs_diff']:.3e}, "
            f"{result['max_matvec_diff']:.3e}"
        )


if __name__ == "__main__":
    main()
