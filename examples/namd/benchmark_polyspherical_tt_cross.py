"""Benchmark block TT-cross sampling and Hermitian MPO compression."""

import argparse
import time

import numpy as np

from pyqed.dvr.dvr_1d import SineDVR
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.namd.polyspherical import (
    PolysphericalTree,
    build_analytic_keo_mpo,
    build_keo_mpo_cross,
)


def dvrs(npts):
    return [
        SineDVR(2.0, 3.0, npts),
        SineDVR(1.4, 2.2, npts),
        SineDVR(0.7, 1.5, npts),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("native", "tntorch"), default="native"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    tree = PolysphericalTree(((1, 2), 0), np.array([1.0, 16.0, 1.0]))

    small_dvrs = dvrs(4)
    reference = _mpo_to_dense_operator(
        build_analytic_keo_mpo(tree, small_dvrs)
    )
    sampled, small_info = build_keo_mpo_cross(
        tree,
        small_dvrs,
        cross_max_rank=8,
        cross_rtol=1.0e-11,
        cross_validation=64,
        backend=args.backend,
        verbose=args.verbose,
        return_info=True,
    )
    sampled_dense = _mpo_to_dense_operator(sampled)
    compressed = sampled.compress_hermitian(12)
    compressed_dense = _mpo_to_dense_operator(compressed)
    print("4^3 validation")
    print("  sampled/full relative error:", np.linalg.norm(sampled_dense - reference) / np.linalg.norm(reference))
    print("  compressed relative error:", np.linalg.norm(compressed_dense - reference) / np.linalg.norm(reference))
    print("  compressed Hermiticity:", np.max(np.abs(compressed_dense - compressed_dense.conj().T)))
    print("  point samples:", small_info["point_samples"], "/", small_info["grid_size"])

    large_dvrs = dvrs(17)
    start = time.perf_counter()
    large, large_info = build_keo_mpo_cross(
        tree,
        large_dvrs,
        cross_max_rank=8,
        cross_rtol=1.0e-10,
        cross_validation=128,
        mpo_max_rank=20,
        backend=args.backend,
        verbose=args.verbose,
        return_info=True,
    )
    print("17^3 sampled build")
    print("  point samples:", large_info["point_samples"], "/", large_info["grid_size"])
    print("  cross validation error:", large_info["cross"]["validation_error"])
    print("  rank history:", large_info["cross"].get("rank_history"))
    print("  evaluator batches:", large_info["cross"].get("batch_calls"))
    print("  MPO bonds:", large.bond_orders())
    print("  elapsed seconds:", time.perf_counter() - start)


if __name__ == "__main__":
    main()
