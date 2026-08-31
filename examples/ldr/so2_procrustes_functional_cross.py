#!/usr/bin/env python3
"""Benchmark FunctionalTT cross fits of SO2 aligned energies and features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from examples.ldr.so2_casci_full_ldr import path_overlap
from examples.ldr.so2_procrustes_gauge import rotate_kernel
from pyqed.ldr.oracle import FeatureOracle
from pyqed.ldr.overlap import unpack
from pyqed.ldr.ttfit import fit_features, fit_hamiltonian


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


class ArrayHamiltonianOracle:
    """Stand in for on-demand chemistry while counting requested geometries."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.batches = 0
        self.requested = 0

    def hamiltonian_many(self, indices):
        self.batches += 1
        self.requested += len(indices)
        return np.asarray([self.values[tuple(index)] for index in indices])


class ArrayGaugeOracle(ArrayHamiltonianOracle):
    """Cached aligned matrix fields standing in for the chemistry oracle."""

    def __init__(self, local, overlap):
        super().__init__(local)
        self.overlap = np.asarray(overlap)
        self.shape = self.values.shape[:-2]

    def overlap_many(self, pairs):
        blocks = []
        for left, right in pairs:
            first = np.ravel_multi_index(left, self.shape)
            second = np.ravel_multi_index(right, self.shape)
            blocks.append(self.overlap[first, :, second, :])
        return np.asarray(blocks)


def relative_error(predicted, exact):
    scale = max(float(np.linalg.norm(exact)), np.finfo(float).tiny)
    return float(np.linalg.norm(predicted - exact) / scale)


def neighbor_pairs(shape):
    pairs = []
    for left in np.ndindex(shape):
        for axis, size in enumerate(shape):
            if left[axis] + 1 >= size:
                continue
            right = list(left)
            right[axis] += 1
            pairs.append((left, tuple(right)))
    return pairs


def link_errors(feature, overlap, shape):
    flat = np.asarray(feature).reshape(np.prod(shape), *feature.shape[-2:])
    fitted = []
    exact = []
    for left, right in neighbor_pairs(shape):
        first = np.ravel_multi_index(left, shape)
        second = np.ravel_multi_index(right, shape)
        fitted.append(flat[first].conj().T @ flat[second])
        exact.append(overlap[first, :, second, :])
    fitted = np.asarray(fitted)
    exact = np.asarray(exact)
    return {
        "relative_frobenius_error": relative_error(fitted, exact),
        "max_abs_error": float(np.max(np.abs(fitted - exact))),
    }, fitted


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


def patch_anchors(name, path, center):
    anchors = [tuple(map(int, center))]
    if name != "two":
        return anchors
    caches = sorted(path.parent.glob("direct_reference_overlaps_theta*.npz"))
    if not caches:
        raise FileNotFoundError(
            f"No secondary-reference overlap cache beside {path}"
        )
    with np.load(caches[0]) as archive:
        anchors.append(tuple(map(int, archive["center"])))
    return anchors


def fit_patch(name, path, grids, coordinates, raw_overlap, args):
    with np.load(path) as archive:
        exact = np.asarray(archive["aligned_local_hamiltonian"], dtype=complex)
        gauge = np.asarray(archive["gauge"], dtype=complex)
        center = tuple(map(int, archive["center"]))
    nstates = exact.shape[-1]
    shape = exact.shape[:-2]
    aligned_overlap = rotate_kernel(
        raw_overlap,
        gauge.reshape(-1, nstates, nstates),
    )
    anchors = patch_anchors(name, path, center)
    output = {
        "source": str(path),
        "anchors": [list(anchor) for anchor in anchors],
        "energy_fits": {},
        "feature_fits": {},
    }
    for rank in args.ranks:
        oracle = ArrayHamiltonianOracle(exact)
        started = time.perf_counter()
        model, info = fit_hamiltonian(
            oracle,
            grids,
            exact.shape[-1],
            max_rank=rank,
            degrees=args.degree,
            sweeps=args.sweeps,
            rtol=args.rtol,
            validation=args.validation,
            seed=args.seed,
            start_rank=rank if args.start_rank is None else args.start_rank,
            kick_rank=args.kick_rank,
        )
        elapsed = time.perf_counter() - started
        predicted = model.predict(coordinates).reshape(exact.shape)
        filename = args.output_dir / f"ebar_{name}_cross_rank{rank}.npz"
        model.save(filename)
        metrics = {
            **info,
            "seconds": elapsed,
            "relative_frobenius_error": relative_error(predicted, exact),
            "max_abs_error": float(np.max(np.abs(predicted - exact))),
            "hermitian_error": relative_error(
                predicted.conj().swapaxes(-1, -2), predicted
            ),
            "oracle_batches": oracle.batches,
            "oracle_requested_geometries": oracle.requested,
            "model": str(filename),
        }
        output["energy_fits"][str(rank)] = jsonable(metrics)
        print(
            f"[{name} E] rank {rank:2d}: "
            f"geometries={info['unique_geometries']:3d}/729, "
            f"relF={metrics['relative_frobenius_error']:.6e}, "
            f"time={elapsed:.3f} s",
            flush=True,
        )

        feature = FeatureOracle(
            ArrayGaugeOracle(exact, aligned_overlap),
            anchors,
            tolerance=args.feature_tolerance,
        )
        started = time.perf_counter()
        model, info = fit_features(
            feature,
            grids,
            max_rank=rank,
            degrees=args.degree,
            sweeps=args.sweeps,
            rtol=args.rtol,
            validation=args.validation,
            seed=args.seed + 1,
            start_rank=rank if args.start_rank is None else args.start_rank,
            kick_rank=args.kick_rank,
        )
        elapsed = time.perf_counter() - started
        requested_before_validation = len(feature.points)
        exact_feature = feature.feature_many(np.ndindex(shape)).reshape(
            *shape, feature.rank, nstates
        )
        predicted = model.predict(coordinates).reshape(exact_feature.shape)
        nystrom_links, exact_feature_links = link_errors(
            exact_feature, aligned_overlap, shape
        )
        fitted_links, predicted_feature_links = link_errors(
            predicted, aligned_overlap, shape
        )
        filename = args.output_dir / f"ybar_{name}_cross_rank{rank}.npz"
        model.save(filename)
        metrics = {
            **info,
            "seconds": elapsed,
            "relative_frobenius_error": relative_error(predicted, exact_feature),
            "max_abs_error": float(np.max(np.abs(predicted - exact_feature))),
            "nystrom_neighbor_links": nystrom_links,
            "fitted_neighbor_links": fitted_links,
            "fit_induced_neighbor_link_relative_error": relative_error(
                predicted_feature_links, exact_feature_links
            ),
            "feature_rank": feature.rank,
            "oracle_geometries_before_full_validation": requested_before_validation,
            "model": str(filename),
        }
        output["feature_fits"][str(rank)] = jsonable(metrics)
        print(
            f"[{name} Y] rank {rank:2d}: "
            f"feature={feature.rank}x{nstates}, "
            f"geometries={info['unique_geometries']:3d}/729, "
            f"relF={metrics['relative_frobenius_error']:.6e}, "
            f"link={fitted_links['relative_frobenius_error']:.6e}, "
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
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=(4, 8, 12, 16, 24, 32),
    )
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--feature-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--start-rank", type=int)
    parser.add_argument("--kick-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_procrustes_functional_cross"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ranks = tuple(sorted(set(int(rank) for rank in args.ranks)))

    with np.load(args.reference) as archive:
        grids = tuple(
            np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa")
        )
        shape = tuple(len(grid) for grid in grids)
        links = unpack(
            archive["link_axes"], archive["link_indices"], archive["link_data"]
        )
        nstates = int(archive["energies"].shape[-1])
    raw_overlap = path_overlap(shape, links).reshape(
        np.prod(shape), nstates, np.prod(shape), nstates
    )
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    paths = {"single": args.single_gauge, "two": args.two_gauge}
    summary = {
        "reference": str(args.reference),
        "grid": [len(grid) for grid in grids],
        "method": "matrix-cached FunctionalTT cross of Ebar and Ybar",
        "patches": {},
    }
    for patch in args.patches:
        summary["patches"][patch] = fit_patch(
            patch, paths[patch], grids, coordinates, raw_overlap, args
        )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
