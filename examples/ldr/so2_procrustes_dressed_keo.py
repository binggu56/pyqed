#!/usr/bin/env python3
"""Directly TT-cross the cached SO2 overlap-dressed kinetic operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

from examples.ldr.so2_casci_cgldr import DEFAULT_SCAN_DIR, load_so2_linked_scan
from examples.ldr.so2_casci_cgldr_dense import dense_kinetic
from examples.ldr.so2_procrustes_overlap_mpo import (
    DEFAULT_LINK_DIR,
    DEFAULT_REFERENCE,
    LABELS,
    fitted_links,
    mpo_block,
)
from pyqed.ldr.ttfit import LinkPath, fit_kinetic
from pyqed.mps.functional import FunctionalTT


def relative_error(predicted, exact):
    scale = max(float(np.linalg.norm(exact)), np.finfo(float).tiny)
    return float(np.linalg.norm(predicted - exact) / scale)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--link-dir", type=Path, default=DEFAULT_LINK_DIR)
    parser.add_argument("--link-rank", type=int, default=32)
    parser.add_argument("--patch", choices=("single", "two"), default="two")
    parser.add_argument("--order", type=int, nargs=3, default=(0, 1, 2))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--operator-rank", type=int)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--validation", type=int, default=256)
    parser.add_argument("--validation-pairs", type=int, default=512)
    parser.add_argument("--start-rank", type=int, default=4)
    parser.add_argument("--kick-rank", type=int, default=2)
    parser.add_argument("--zero-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_procrustes_dressed_keo"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.reference) as archive:
        grids = tuple(
            np.asarray(archive[name], dtype=float)
            for name in ("qs", "theta", "qa")
        )
        nstates = int(archive["energies"].shape[-1])
    shape = tuple(len(grid) for grid in grids)
    models = tuple(
        FunctionalTT.load(
            args.link_dir
            / f"link_{label}_{args.patch}_rank{args.link_rank}.npz"
        )
        for label in LABELS
    )
    oracle = LinkPath(
        shape,
        nstates,
        fitted_links(models, grids),
        order=args.order,
    )
    scan = load_so2_linked_scan(args.scan_dir)
    dense, axes = dense_kinetic(scan, *grids)
    terms = scan.solver.buildK_qsqa_terms(
        axes,
        symmetrize=True,
        svd_tol=0.0,
    )

    started = time.perf_counter()
    components, info = fit_kinetic(
        oracle,
        terms,
        shape,
        nstates,
        max_rank=args.rank,
        operator_rank=args.operator_rank,
        sweeps=args.sweeps,
        rtol=args.rtol,
        validation=args.validation,
        seed=args.seed,
        start_rank=args.start_rank,
        kick_rank=args.kick_rank,
        zero_tolerance=args.zero_tolerance,
        split=True,
    )
    elapsed = time.perf_counter() - started

    rng = np.random.default_rng(args.seed + 1)
    nonzero = np.argwhere(np.abs(dense) > args.zero_tolerance)
    zero = np.argwhere(np.abs(dense) <= args.zero_tolerance)

    def select(candidates):
        count = min(args.validation_pairs, len(candidates))
        return candidates[rng.choice(len(candidates), size=count, replace=False)]

    def compare(candidates):
        exact = []
        predicted = []
        for left_flat, right_flat in select(candidates):
            left = tuple(np.unravel_index(int(left_flat), shape))
            right = tuple(np.unravel_index(int(right_flat), shape))
            exact.append(dense[left_flat, right_flat] * oracle.between(left, right))
            predicted.append(
                sum(
                    mpo_block(component, left, right, nstates)
                    for component in components
                )
            )
        return np.asarray(predicted), np.asarray(exact)

    predicted, exact = compare(nonzero)
    predicted_zero, exact_zero = compare(zero)
    output_path = args.output_dir / "dressed_keo.pkl"
    with output_path.open("wb") as stream:
        pickle.dump(components, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "method": "direct SOP-group TT-cross of T(I,J) Sbar(I,J)",
        "grid": list(shape),
        "states": nstates,
        "sop_terms": len(terms),
        "dense_kinetic_nonzeros": int(len(nonzero)),
        "dense_kinetic_fraction": float(len(nonzero) / dense.size),
        "path_order": [LABELS[axis] for axis in args.order],
        "fit": info,
        "seconds": elapsed,
        "nonzero_relative_frobenius_error": relative_error(predicted, exact),
        "nonzero_max_abs_error": float(np.max(np.abs(predicted - exact))),
        "zero_max_abs_error": float(np.max(np.abs(predicted_zero - exact_zero))),
        "output": str(output_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(
        f"rank={args.rank}, component ranks={info['operator_ranks']}, "
        f"samples={info['scalar_samples']}, "
        f"transport pairs={info['unique_transport_pairs']}, "
        f"relF={summary['nonzero_relative_frobenius_error']:.3e}, "
        f"zero={summary['zero_max_abs_error']:.3e}, time={elapsed:.3f} s",
        flush=True,
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
