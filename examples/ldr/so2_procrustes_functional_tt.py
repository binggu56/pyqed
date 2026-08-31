#!/usr/bin/env python3
"""Fit SO2 Procrustes-gauge electronic energies with FunctionalTT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyqed.mps import FunctionalTT
from pyqed.mps.decompose import tt_to_tensor


DEFAULT_REFERENCE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_full_ldr_9x9x9_20fs/"
    "electronic_reference.npz"
)
DEFAULT_SINGLE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_procrustes_gauge_9x9x9/"
    "procrustes_gauge.npz"
)
DEFAULT_TWO = Path(
    "/private/tmp/so2_cas6e6o_631gstar_procrustes_two_patch_9x9x9/"
    "procrustes_gauge.npz"
)


def relative_error(predicted, exact):
    scale = max(float(np.linalg.norm(exact)), np.finfo(float).tiny)
    return float(np.linalg.norm(predicted - exact) / scale)


def hermitian_error(values):
    return relative_error(values.conj().swapaxes(-1, -2), values)


def fit_field(name, path, grids, coordinates, args):
    with np.load(path) as archive:
        exact = np.asarray(archive["aligned_local_hamiltonian"], dtype=complex)
    grid_shape = tuple(len(grid) for grid in grids)
    if exact.shape[:3] != grid_shape or exact.shape[-2] != exact.shape[-1]:
        raise ValueError(f"{name} aligned energy has incompatible shape {exact.shape}")
    bounds = tuple((float(grid[0]), float(grid[-1])) for grid in grids)
    nstates = exact.shape[-1]
    output = {
        "source": str(path),
        "shape": list(exact.shape),
        "input_hermitian_error": hermitian_error(exact),
        "fits": {},
    }
    flat_exact = exact.reshape(-1, nstates, nstates)

    for middle_rank in args.ranks:
        model = FunctionalTT(
            degrees=args.degree,
            rank=(args.edge_rank, middle_rank, args.edge_rank),
            bounds=bounds,
            normalization="frobenius",
            hermitian=True,
            regularization=args.regularization,
            random_state=args.seed,
        )
        started = time.perf_counter()
        model.fit_grid(grids, exact)
        elapsed = time.perf_counter() - started
        predicted = model.predict(coordinates).reshape(exact.shape)
        grid_cores = model.tensor_cores(grids)
        reconstructed = np.asarray(tt_to_tensor(grid_cores)).reshape(exact.shape)
        filename = args.output_dir / f"ebar_{name}_rank{middle_rank}.npz"
        model.save(filename)
        loaded = FunctionalTT.load(filename)
        loaded_values = loaded.predict(coordinates).reshape(exact.shape)
        metrics = {
            "ranks": list(model.ranks_),
            "seconds": elapsed,
            "relative_frobenius_error": relative_error(predicted, exact),
            "max_abs_error": float(np.max(np.abs(predicted - exact))),
            "hermitian_error": hermitian_error(predicted),
            "grid_tt_max_abs_error": float(np.max(np.abs(reconstructed - predicted))),
            "reload_max_abs_error": float(np.max(np.abs(loaded_values - predicted))),
            "real_coordinate_cores": all(np.isrealobj(core) for core in model.cores),
            "real_output_core": bool(np.isrealobj(model.output_core)),
            "model": str(filename),
        }
        output["fits"][str(middle_rank)] = metrics
        print(
            f"[{name}] central rank {middle_rank:2d}: "
            f"relF={metrics['relative_frobenius_error']:.6e}, "
            f"max={metrics['max_abs_error']:.6e}, "
            f"time={elapsed:.3f} s",
            flush=True,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--single-gauge", type=Path, default=DEFAULT_SINGLE)
    parser.add_argument("--two-gauge", type=Path, default=DEFAULT_TWO)
    parser.add_argument(
        "--patches",
        nargs="+",
        choices=("single", "two"),
        default=("single", "two"),
    )
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--edge-rank", type=int, default=9)
    parser.add_argument("--ranks", type=int, nargs="+", default=(8, 12, 16, 24, 32, 48))
    parser.add_argument("--regularization", type=float, default=1.0e-12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_procrustes_functional_tt"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ranks = tuple(sorted(set(int(rank) for rank in args.ranks)))

    with np.load(args.reference) as archive:
        grids = tuple(
            np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa")
        )
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    paths = {"single": args.single_gauge, "two": args.two_gauge}
    summary = {
        "reference": str(args.reference),
        "grid": [len(grid) for grid in grids],
        "degree": args.degree,
        "normalization": "frobenius",
        "hermitian_representation": "real orthonormal packed basis",
        "patches": {},
    }
    for name in args.patches:
        summary["patches"][name] = fit_field(
            name,
            paths[name],
            grids,
            coordinates,
            args,
        )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
