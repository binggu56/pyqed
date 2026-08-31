#!/usr/bin/env python3
"""Build a fixed-path SO2 overlap MPO from fitted directional links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

from pyqed.ldr.ttfit import LinkPath, coupled_mpo, fit_overlap
from pyqed.mps.decompose import decompose
from pyqed.mps.functional import FunctionalTT


DEFAULT_REFERENCE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_full_ldr_9x9x9_20fs/"
    "electronic_reference.npz"
)
DEFAULT_LINK_DIR = Path("/private/tmp/so2_procrustes_link_cross_r32")
LABELS = ("qs", "theta", "qa")


def edge_grids(grids, axis):
    output = list(grids)
    output[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
    return tuple(output)


def fitted_links(models, grids):
    links = {}
    for axis, model in enumerate(models):
        axes = edge_grids(grids, axis)
        mesh = np.meshgrid(*axes, indexing="ij")
        coordinates = np.stack(
            [coordinate.reshape(-1) for coordinate in mesh], axis=1
        )
        shape = tuple(len(grid) for grid in axes)
        values = model.predict(coordinates).reshape(
            *shape, model.output_shape_[0], model.output_shape_[1]
        )
        for index in np.ndindex(shape):
            links[(axis, index)] = values[index]
    return links


def mpo_block(mpo, bra, ket, nstates):
    blocks = []
    for alpha in range(nstates):
        row = []
        for beta in range(nstates):
            value = np.ones((1,), dtype=complex)
            indices = (*zip(bra, ket), (alpha, beta))
            for factor, (left, right) in zip(mpo.factors, indices):
                value = value @ factor[:, :, left, right]
            row.append(value.item())
        blocks.append(row)
    return np.asarray(blocks)


def relative_error(predicted, exact):
    scale = max(float(np.linalg.norm(exact)), np.finfo(float).tiny)
    return float(np.linalg.norm(predicted - exact) / scale)


def full_tensor(oracle):
    shape = oracle.shape
    nstates = oracle.nstates
    values = np.empty(tuple(size**2 for size in shape) + (nstates**2,), complex)
    indices = list(np.ndindex(shape))
    for left_flat, left in enumerate(indices):
        for right_flat in range(left_flat, len(indices)):
            right = indices[right_flat]
            block = oracle.between(left, right)
            forward = tuple(
                first * size + second
                for first, second, size in zip(left, right, shape)
            )
            values[forward] = block.reshape(-1)
            if left != right:
                reverse = tuple(
                    second * size + first
                    for first, second, size in zip(left, right, shape)
                )
                values[reverse] = block.conj().T.reshape(-1)
    return values


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
    parser.add_argument("--link-dir", type=Path, default=DEFAULT_LINK_DIR)
    parser.add_argument("--link-rank", type=int, default=32)
    parser.add_argument("--patch", choices=("single", "two"), default="two")
    parser.add_argument("--order", type=int, nargs=3, default=(0, 1, 2))
    parser.add_argument("--backend", choices=("cross", "svd"), default="cross")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--operator-rank", type=int)
    parser.add_argument(
        "--electronic-mode",
        choices=("coupled", "blockwise"),
        default="blockwise",
    )
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--validation", type=int, default=256)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--kick-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_procrustes_overlap_mpo"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.reference) as archive:
        grids = tuple(
            np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa")
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
    links = fitted_links(models, grids)
    oracle = LinkPath(shape, nstates, links, order=args.order)

    started = time.perf_counter()
    if args.backend == "cross":
        overlap, info = fit_overlap(
            oracle,
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
            hermitize=True,
            diagonal_exact=True,
            electronic_mode=args.electronic_mode,
        )
    else:
        tensor_started = time.perf_counter()
        values = full_tensor(oracle)
        tensor_seconds = time.perf_counter() - tensor_started
        cores = decompose(values, rank=args.rank)
        overlap = coupled_mpo(
            cores,
            shape,
            nstates,
            active=tuple(range(len(shape))),
        )
        info = {
            "backend": "path-overlap-grid-tt-svd",
            "full_tensor_shape": values.shape,
            "full_tensor_entries": int(values.size),
            "tensor_seconds": tensor_seconds,
            "operator_ranks": tuple(overlap.bond_orders()),
        }
    elapsed = time.perf_counter() - started

    rng = np.random.default_rng(args.seed + 1)
    flat_pairs = rng.integers(
        0, np.prod(shape), size=(args.validation_pairs, 2)
    )
    pairs = [
        (
            tuple(np.unravel_index(int(left), shape)),
            tuple(np.unravel_index(int(right), shape)),
        )
        for left, right in flat_pairs
    ]
    exact = oracle.overlap_many(pairs)
    predicted = np.asarray([
        mpo_block(overlap, left, right, nstates) for left, right in pairs
    ])
    diagonal = np.asarray([
        mpo_block(overlap, index, index, nstates) for index in np.ndindex(shape)
    ])
    identity = np.eye(nstates, dtype=complex)
    output_path = args.output_dir / "overlap_mpo.pkl"
    with output_path.open("wb") as stream:
        pickle.dump(overlap, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "method": "fixed-coordinate-path overlap MPO from fitted links",
        "backend": args.backend,
        "grid": list(shape),
        "path_order": [LABELS[axis] for axis in args.order],
        "link_models": [str(args.link_dir / f"link_{label}_{args.patch}_rank{args.link_rank}.npz") for label in LABELS],
        "fit": info,
        "seconds": elapsed,
        "held_out_pairs": len(pairs),
        "held_out_relative_frobenius_error": relative_error(predicted, exact),
        "held_out_max_abs_error": float(np.max(np.abs(predicted - exact))),
        "diagonal_identity_max_abs_error": float(
            np.max(np.abs(diagonal - identity))
        ),
        "path_blocks_evaluated": len(oracle.blocks) // 2,
        "directional_links_used": len(oracle.used_links),
        "output": str(output_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    sample_count = info.get("scalar_samples", info.get("full_tensor_entries", 0))
    block_count = info.get("unique_overlap_blocks", np.prod(shape) ** 2)
    print(
        f"rank={args.rank}, MPO ranks={tuple(overlap.bond_orders())}, "
        f"samples={sample_count}, "
        f"blocks={block_count}, "
        f"held-out relF={summary['held_out_relative_frobenius_error']:.3e}, "
        f"diag={summary['diagonal_identity_max_abs_error']:.3e}, "
        f"time={elapsed:.3f} s",
        flush=True,
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
