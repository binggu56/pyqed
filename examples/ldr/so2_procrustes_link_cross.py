#!/usr/bin/env python3
"""Fit the three aligned SO2 nearest-neighbor link fields by FunctionalTT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyqed.ldr.overlap import unpack
from pyqed.ldr.ttfit import fit_links


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


class LinkOracle:
    """Serve directly aligned cached nearest-neighbor overlap blocks."""

    def __init__(self, shape, links, gauge):
        self.shape = tuple(int(size) for size in shape)
        self.links = links
        self.gauge = np.asarray(gauge, dtype=complex).reshape(
            *self.shape, gauge.shape[-2], gauge.shape[-1]
        )
        self.batches = 0
        self.requested_pairs = 0

    def overlap_many(self, pairs):
        self.batches += 1
        self.requested_pairs += len(pairs)
        output = []
        for left, right in pairs:
            left = tuple(map(int, left))
            right = tuple(map(int, right))
            delta = np.asarray(right) - left
            active = np.flatnonzero(delta)
            if len(active) != 1 or abs(delta[active[0]]) != 1:
                raise ValueError("link oracle only accepts nearest neighbors")
            axis = int(active[0])
            if delta[axis] < 0:
                forward = self.overlap_many(((right, left),))[0]
                output.append(forward.conj().T)
                continue
            raw = np.asarray(self.links[(axis, left)], dtype=complex)
            output.append(
                self.gauge[left].conj().T @ raw @ self.gauge[right]
            )
        return np.asarray(output)


def edge_grids(grids, axis):
    output = list(grids)
    output[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
    return tuple(output)


def exact_links(oracle, axis):
    shape = list(oracle.shape)
    shape[axis] -= 1
    indices = list(np.ndindex(tuple(shape)))
    pairs = []
    for left in indices:
        right = list(left)
        right[axis] += 1
        pairs.append((left, tuple(right)))
    return oracle.overlap_many(pairs).reshape(
        *shape, oracle.gauge.shape[-1], oracle.gauge.shape[-1]
    )


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


def fit_patch(name, gauge_path, grids, links, nstates, args):
    with np.load(gauge_path) as archive:
        gauge = np.asarray(archive["gauge"], dtype=complex)
    shape = tuple(len(grid) for grid in grids)
    exact_oracle = LinkOracle(shape, links, gauge)
    exact = tuple(exact_links(exact_oracle, axis) for axis in range(len(shape)))
    output = {"source": str(gauge_path), "fits": {}}
    labels = ("qs", "theta", "qa")
    for rank in args.ranks:
        oracle = LinkOracle(shape, links, gauge)
        started = time.perf_counter()
        models, info = fit_links(
            oracle,
            grids,
            nstates,
            max_rank=rank,
            degrees=args.degree,
            sweeps=args.sweeps,
            rtol=args.rtol,
            validation=args.validation,
            seed=args.seed,
            start_rank=args.start_rank,
            kick_rank=args.kick_rank,
        )
        elapsed = time.perf_counter() - started
        requested_before_validation = oracle.requested_pairs
        direction_metrics = []
        squared_error = 0.0
        squared_scale = 0.0
        for axis, (label, model, target) in enumerate(zip(labels, models, exact)):
            axes = edge_grids(grids, axis)
            mesh = np.meshgrid(*axes, indexing="ij")
            coordinates = np.stack(
                [coordinate.reshape(-1) for coordinate in mesh], axis=1
            )
            predicted = model.predict(coordinates).reshape(target.shape)
            difference = predicted - target
            squared_error += float(np.linalg.norm(difference) ** 2)
            squared_scale += float(np.linalg.norm(target) ** 2)
            filename = args.output_dir / f"link_{label}_{name}_rank{rank}.npz"
            model.save(filename)
            direction_metrics.append(
                {
                    **info["directions"][axis],
                    "label": label,
                    "relative_frobenius_error": relative_error(predicted, target),
                    "max_abs_error": float(np.max(np.abs(difference))),
                    "model": str(filename),
                }
            )
        metrics = {
            **info,
            "directions": direction_metrics,
            "seconds": elapsed,
            "relative_frobenius_error": float(
                np.sqrt(squared_error / max(squared_scale, np.finfo(float).tiny))
            ),
            "oracle_batches": oracle.batches,
            "oracle_link_requests_before_validation": requested_before_validation,
        }
        output["fits"][str(rank)] = jsonable(metrics)
        errors = ", ".join(
            f"{item['label']}={item['relative_frobenius_error']:.2e}"
            for item in direction_metrics
        )
        print(
            f"[{name}] rank {rank:2d}: "
            f"points={info['unique_geometries']:3d}/{np.prod(shape)}, "
            f"links={info['unique_links']:4d}/{len(links)}, "
            f"relF={metrics['relative_frobenius_error']:.3e} "
            f"({errors}), time={elapsed:.3f} s",
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
    parser.add_argument("--ranks", type=int, nargs="+", default=(2, 4, 8, 12, 16))
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--kick-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_procrustes_link_cross"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ranks = tuple(sorted(set(map(int, args.ranks))))

    with np.load(args.reference) as archive:
        grids = tuple(
            np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa")
        )
        nstates = int(archive["energies"].shape[-1])
        links = unpack(
            archive["link_axes"], archive["link_indices"], archive["link_data"]
        )
    paths = {"single": args.single_gauge, "two": args.two_gauge}
    summary = {
        "reference": str(args.reference),
        "grid": [len(grid) for grid in grids],
        "method": "separate matrix-cached FunctionalTT crosses of aligned links",
        "patches": {},
    }
    for patch in args.patches:
        summary["patches"][patch] = fit_patch(
            patch, paths[patch], grids, links, nstates, args
        )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
