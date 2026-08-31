"""Discrete tensor-train cross interpolation from point evaluations."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def tt_value(cores, index):
    """Evaluate TT cores at one integer multi-index."""
    value = np.ones((1,), dtype=np.result_type(*cores))
    for core, position in zip(cores, index):
        value = value @ core[:, int(position), :]
    return value.item()


def _ranks(shape, max_rank):
    ranks = [1]
    for split in range(1, len(shape)):
        left = int(np.prod(shape[:split]))
        right = int(np.prod(shape[split:]))
        ranks.append(min(int(max_rank), left, right))
    return ranks + [1]


def _grow_ranks(shape, ranks, max_rank, kick_rank):
    limits = _ranks(shape, max_rank)
    return [
        1,
        *[
            min(limits[split], ranks[split] + int(kick_rank))
            for split in range(1, len(shape))
        ],
        1,
    ]


def _pivot_rows(matrix, count):
    _q, _r, pivots = qr(matrix.T, pivoting=True, mode="economic")
    selected = np.asarray(pivots[:count], dtype=int)
    for _ in range(20):
        cross = matrix[selected, :]
        coefficients = matrix @ np.linalg.pinv(cross)
        row, column = np.unravel_index(
            np.argmax(np.abs(coefficients)), coefficients.shape
        )
        if abs(coefficients[row, column]) <= 1.05:
            break
        selected[column] = row
    return selected


def _pivot_columns(matrix, count):
    return _pivot_rows(matrix.T, count)


def _resize_suffixes(existing, shape, count, rng):
    existing = [] if existing is None else list(existing)
    unique = list(dict.fromkeys(tuple(item) for item in existing))
    if len(unique) >= count:
        return unique[:count]
    size = int(np.prod(shape))
    used = {int(np.ravel_multi_index(item, shape)) for item in unique}
    available = np.asarray([item for item in range(size) if item not in used])
    choices = rng.choice(available, size=count - len(unique), replace=False)
    unique.extend(tuple(np.unravel_index(int(item), shape)) for item in choices)
    return unique


def tt_cross(
    shape,
    evaluator,
    *,
    max_rank=8,
    sweeps=4,
    rtol=1.0e-8,
    validation=64,
    seed=0,
    start_rank=1,
    kick_rank=2,
    batch_evaluator=None,
    initial=None,
    return_state=False,
):
    """Fit a tensor train using cached adaptive cross interpolation.

    ``evaluator(index)`` is called only for selected integer multi-indices.
    The returned information dictionary reports unique sample counts and the
    maximum relative error on a fixed random validation set.
    """
    shape = tuple(int(size) for size in shape)
    if not shape or any(size < 1 for size in shape):
        raise ValueError("shape must contain positive dimensions")
    if int(max_rank) < 1 or int(sweeps) < 1 or int(start_rank) < 1:
        raise ValueError("max_rank and sweeps must be positive")
    if int(kick_rank) < 1:
        raise ValueError("kick_rank must be positive")
    if rtol < 0.0 or int(validation) < 0:
        raise ValueError("rtol and validation must be nonnegative")

    rng = np.random.default_rng(seed)
    initial_ranks = None if initial is None else initial.get("ranks")
    if initial_ranks is None:
        rank_cap = max_rank if int(validation) == 0 else start_rank
        ranks = _ranks(shape, min(int(rank_cap), int(max_rank)))
    else:
        if len(initial_ranks) != len(shape) + 1:
            raise ValueError("initial TT-cross ranks have an incompatible shape")
        limits = _ranks(shape, max_rank)
        ranks = [
            min(int(rank), limit)
            for rank, limit in zip(initial_ranks, limits)
        ]
    cache = {} if initial is None else dict(initial.get("cache", {}))
    initial_samples = len(cache)
    batch_calls = 0

    def evaluate_many(indices):
        nonlocal batch_calls
        indices = [tuple(int(item) for item in index) for index in indices]
        missing = list(dict.fromkeys(index for index in indices if index not in cache))
        if missing:
            if batch_evaluator is None:
                values = [evaluator(index) for index in missing]
            else:
                values = np.asarray(batch_evaluator(np.asarray(missing, dtype=int)))
                if values.shape != (len(missing),):
                    raise ValueError(
                        "batch_evaluator must return one scalar per index"
                    )
                batch_calls += 1
            for index, value in zip(missing, values):
                value = np.asarray(value)
                if value.ndim != 0:
                    raise ValueError("TT-cross evaluator must return a scalar")
                cache[index] = value.item()
        return np.asarray([cache[index] for index in indices])

    if len(shape) == 1:
        values = evaluate_many([(i,) for i in range(shape[0])])
        return [values.reshape(1, shape[0], 1)], {
            "backend": "native",
            "samples": len(cache),
            "new_samples": len(cache) - initial_samples,
            "batch_calls": batch_calls,
            "sweeps": 0,
            "validation_error": 0.0,
            "validation_rms_error": 0.0,
            "ranks": (1, 1),
            "rank_history": (),
        }

    left_sets = [[()]] + [None] * (len(shape) - 1)
    right_sets = [None] * len(shape) + [[()]]
    initial_right = None if initial is None else initial.get("right_sets")
    if initial is not None and tuple(initial.get("shape", ())) != shape:
        raise ValueError("initial TT-cross state has an incompatible shape")
    for split in range(1, len(shape)):
        existing = None if initial_right is None else initial_right[split]
        right_sets[split] = _resize_suffixes(
            existing, shape[split:], ranks[split], rng
        )

    total = int(np.prod(shape))
    nvalidation = min(int(validation), total)
    validation_indices = [
        tuple(np.unravel_index(int(item), shape))
        for item in rng.choice(total, size=nvalidation, replace=False)
    ]
    completed = 0
    error = np.inf
    rms_error = np.inf
    rank_history = []

    for sweep in range(int(sweeps)):
        for site in range(len(shape) - 1):
            prefixes = left_sets[site]
            suffixes = right_sets[site + 1]
            rows = [
                prefix + (coordinate,)
                for prefix in prefixes
                for coordinate in range(shape[site])
            ]
            queries = [prefix + suffix for prefix in rows for suffix in suffixes]
            matrix = evaluate_many(queries).reshape(len(rows), len(suffixes))
            selected = _pivot_rows(matrix, ranks[site + 1])
            left_sets[site + 1] = [rows[item] for item in selected]

        for site in range(len(shape) - 1, 0, -1):
            prefixes = left_sets[site]
            suffixes = right_sets[site + 1]
            columns = [
                (coordinate,) + suffix
                for coordinate in range(shape[site])
                for suffix in suffixes
            ]
            queries = [prefix + suffix for prefix in prefixes for suffix in columns]
            matrix = evaluate_many(queries).reshape(len(prefixes), len(columns))
            selected = _pivot_columns(matrix, ranks[site])
            right_sets[site] = [columns[item] for item in selected]

        cores = _cross_cores(shape, evaluate_many, left_sets, right_sets, ranks)
        completed = sweep + 1
        rank_history.append(tuple(ranks))
        if validation_indices:
            exact = evaluate_many(validation_indices)
            fitted = np.asarray([tt_value(cores, index) for index in validation_indices])
            scale = max(float(np.max(np.abs(exact))), 1.0)
            error = float(np.max(np.abs(fitted - exact)) / scale)
            norm = max(float(np.linalg.norm(exact)), 1.0)
            rms_error = float(np.linalg.norm(fitted - exact) / norm)
            if error <= rtol:
                break
        else:
            error = 0.0
            rms_error = 0.0
            break

        if sweep == int(sweeps) - 1:
            continue
        new_ranks = _grow_ranks(shape, ranks, max_rank, kick_rank)
        if new_ranks == ranks:
            continue
        ranks = new_ranks
        for split in range(1, len(shape)):
            right_sets[split] = _resize_suffixes(
                right_sets[split], shape[split:], ranks[split], rng
            )

    info = {
        "backend": "native",
        "samples": len(cache),
        "new_samples": len(cache) - initial_samples,
        "batch_calls": batch_calls,
        "sweeps": completed,
        "validation_error": error,
        "validation_rms_error": rms_error,
        "ranks": tuple(ranks),
        "rank_history": tuple(rank_history),
    }
    if return_state:
        info["state"] = {
            "shape": shape,
            "ranks": tuple(ranks),
            "cache": dict(cache),
            "right_sets": tuple(
                None if item is None else tuple(item) for item in right_sets
            ),
        }
    return cores, info


def tt_cross_tntorch(
    shape,
    evaluator,
    *,
    max_rank=100,
    sweeps=25,
    rtol=1.0e-8,
    validation=1000,
    seed=0,
    device=None,
    verbose=False,
    start_rank=1,
    kick_rank=2,
    batch_evaluator=None,
    initial=None,
    return_state=False,
):
    """Fit a TT with the optional adaptive :mod:`tntorch` backend."""
    try:
        import torch
        import tntorch
    except ImportError as error:
        raise ImportError(
            "backend='tntorch' requires the optional torch and tntorch packages"
        ) from error

    shape = tuple(int(size) for size in shape)
    if not shape or any(size < 1 for size in shape):
        raise ValueError("shape must contain positive dimensions")
    device = torch.device("cpu" if device is None else device)
    if device.type != "cpu":
        raise ValueError("tntorch 1.1.2 cross interpolation is CPU-only")
    if initial is not None:
        raise ValueError("tntorch does not support native TT-cross warm starts")
    if int(start_rank) != 1:
        raise ValueError("tntorch adaptive cross requires start_rank=1")
    cache = {}

    def evaluate(index):
        index = tuple(int(item) for item in index)
        if index not in cache:
            value = np.asarray(evaluator(index))
            if value.ndim != 0:
                raise ValueError("TT-cross evaluator must return a scalar")
            if np.iscomplexobj(value):
                raise ValueError("the tntorch cross backend currently requires real fields")
            cache[index] = float(value)
        return cache[index]

    def batched(points):
        indices = np.rint(points.detach().cpu().numpy()).astype(int)
        missing = list(
            dict.fromkeys(
                tuple(int(item) for item in row)
                for row in indices
                if tuple(int(item) for item in row) not in cache
            )
        )
        if missing and batch_evaluator is not None:
            values = np.asarray(
                batch_evaluator(np.asarray(missing, dtype=int)), dtype=float
            )
            if values.shape != (len(missing),):
                raise ValueError(
                    "batch_evaluator must return one scalar per index"
                )
            cache.update(zip(missing, values.tolist()))
        values = np.asarray([evaluate(row) for row in indices], dtype=float)
        return torch.as_tensor(
            values,
            dtype=torch.float64,
            device=points.device,
        )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    domain = [
        torch.arange(size, dtype=torch.float64, device=device)
        for size in shape
    ]
    tensor, raw_info = tntorch.cross(
        function=batched,
        domain=domain,
        function_arg="matrix",
        kickrank=int(kick_rank),
        rmax=int(max_rank),
        eps=float(rtol),
        max_iter=int(sweeps),
        val_size=max(1, int(validation)),
        verbose=bool(verbose),
        return_info=True,
    )
    cores = [core.detach().cpu().numpy() for core in tensor.cores]
    validation_error = raw_info.get("val_eps", np.nan)
    if hasattr(validation_error, "detach"):
        validation_error = validation_error.detach().cpu().item()
    info = {
        "backend": "tntorch",
        "samples": len(cache),
        "function_evaluations": int(raw_info.get("nsamples", len(cache))),
        "sweeps": len(raw_info.get("val_epss", ())),
        "validation_error": float(validation_error),
        "ranks": tuple([cores[0].shape[0]] + [core.shape[2] for core in cores]),
    }
    if return_state:
        info["state"] = None
    return cores, info


def _cross_cores(shape, evaluate_many, left_sets, right_sets, ranks):
    cores = []
    for site, size in enumerate(shape):
        prefixes = left_sets[site]
        suffixes = right_sets[site + 1]
        queries = [
            prefix + (coordinate,) + suffix
            for prefix in prefixes
            for coordinate in range(size)
            for suffix in suffixes
        ]
        block = evaluate_many(queries).reshape(
            ranks[site], size, ranks[site + 1]
        )
        if site == len(shape) - 1:
            cores.append(block.reshape(ranks[site], size, 1))
            continue
        queries = [
            prefix + suffix
            for prefix in left_sets[site + 1]
            for suffix in suffixes
        ]
        cross = evaluate_many(queries).reshape(
            ranks[site + 1], ranks[site + 1]
        )
        unfolding = block.reshape(ranks[site] * size, ranks[site + 1])
        core = unfolding @ np.linalg.pinv(cross)
        cores.append(core.reshape(ranks[site], size, ranks[site + 1]))
    return cores


__all__ = ["tt_cross", "tt_cross_tntorch", "tt_value"]
