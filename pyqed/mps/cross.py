"""Discrete tensor-train cross interpolation from point evaluations."""

from __future__ import annotations

from math import prod

import numpy as np
from scipy.linalg import qr, svd


def tt_value(cores, index):
    """Evaluate TT cores at one integer multi-index."""
    value = np.ones((1,), dtype=np.result_type(*cores))
    for core, position in zip(cores, index):
        value = value @ core[:, int(position), :]
    return value.item()


def _ranks(shape, max_rank):
    ranks = [1]
    for split in range(1, len(shape)):
        left = prod(shape[:split])
        right = prod(shape[split:])
        ranks.append(min(int(max_rank), left, right))
    return ranks + [1]


def _grow_ranks(shape, ranks, max_rank, kick_rank, active=None):
    limits = _ranks(shape, max_rank)
    grown = [
        1,
        *[
            min(limits[split], ranks[split] + int(kick_rank))
            if active is None or active[split] else ranks[split]
            for split in range(1, len(shape))
        ],
        1,
    ]
    for split in range(1,len(shape)):
        grown[split] = min(grown[split],grown[split-1]*shape[split-1])
    for split in range(len(shape)-1,0,-1):
        grown[split] = min(grown[split],grown[split+1]*shape[split])
    return grown


def _pinv(matrix):
    """Retry a failed divide-and-conquer SVD with the QR-based LAPACK SVD."""
    try:
        return np.linalg.pinv(matrix)
    except np.linalg.LinAlgError:
        u,s,vh=svd(matrix,full_matrices=False,lapack_driver='gesvd')
        inverse=np.zeros_like(s)
        mask=s>1.e-15*np.max(s,initial=0.)
        inverse[mask]=1/s[mask]
        return (vh.conj().T*inverse)@u.conj().T


def _pivot_rows(matrix, count):
    matrix, _r = qr(matrix, mode="economic")
    _q, _r, pivots = qr(matrix.T, pivoting=True, mode="economic")
    selected = np.asarray(pivots[:count], dtype=int)
    coefficients = matrix @ _pinv(matrix[selected, :])
    for _ in range(20):
        row, column = np.unravel_index(
            np.argmax(np.abs(coefficients)), coefficients.shape
        )
        if abs(coefficients[row, column]) <= 1.05:
            break
        # Exact max-volume row replacement; avoid an SVD at every exchange.
        delta=coefficients[row].copy();delta[column]-=1
        coefficients-=np.outer(coefficients[:,column].copy()/coefficients[row,column],delta)
        selected[column] = row
    return selected


def _pivot_columns(matrix, count):
    return _pivot_rows(matrix.T, count)


def _resize_suffixes(existing, shape, count, rng):
    existing = [] if existing is None else list(existing)
    unique = list(dict.fromkeys(tuple(item) for item in existing))
    if len(unique) >= count:
        return unique[:count]
    size = prod(shape)
    used = set(unique)
    if size <= 2 * count:
        available = [item for item in np.ndindex(*shape) if item not in used]
        choices = rng.choice(len(available), size=count - len(unique), replace=False)
        unique.extend(available[item] for item in choices)
    else:
        while len(unique) < count:
            item = tuple(int(rng.integers(dim)) for dim in shape)
            if item not in used:
                used.add(item)
                unique.append(item)
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
    stagnation_sweeps=2,
    max_restarts=1,
    rank_growth="uniform",
    probes=None,
    cache_size=None,
    batch_size=4096,
):
    """Fit a tensor train using cached adaptive cross interpolation.

    ``cache_size`` optionally bounds retained scalar samples (FIFO eviction).
    Evicted values are recomputed without changing interpolation or gates.
    ``batch_size`` bounds evaluator batches; ``samples`` counts retained cache
    entries and ``evaluations`` counts actual evaluations in this invocation.

    ``evaluator(index)`` is called only for selected integer multi-indices.
    The returned information dictionary reports unique sample counts and the
    maximum scaled error on accumulated random validation probes.
    A fresh probe set must also pass before convergence is reported.
    ``stagnation_sweeps`` bounds non-improving sweeps at the rank cap;
    ``max_restarts`` limits random suffix-pivot restarts. ``sweeps`` bounds
    all work, including restarts. ``info["success"]`` is false when checks
    are disabled or the fit fails; inspect ``message`` and ``history``.

    Adaptation of I. Oseledets and E. Tyrtyshnikov, Linear Algebra Appl.
    432, 70–88 (2010), https://doi.org/10.1016/j.laa.2009.07.024.
    QR-orthonormalized interpolation, pivoted-QR/max-volume row selection,
    and heuristic rank growth are used. Max-volume row exchanges use exact
    rank-one coefficient updates after the initial pseudoinverse.
    Sampled stopping is not a global
    accuracy certificate; setting validation=0 disables error checking.
    Restarts and validation refresh are heuristic safeguards, not AMEn
    residual enrichment or a reproduction of its convergence properties.

    ``rank_growth="bond"`` is a heuristic adaptation: numerical ranks of
    the sampled forward/backward matrices identify saturated bonds. Only
    those bonds grow; if none can grow while global validation fails, all
    eligible bonds are explored. This reuses sampled matrices, adds no probe
    evaluations, and is not a certified rank estimator or a two-site method.

    ``probes`` supplies additional deterministic integer multi-indices for
    validation (e.g. symmetry-fixed coordinate fibers). They supplement,
    rather than replace, accumulated and fresh random checks.
    """
    shape = tuple(int(size) for size in shape)
    if rank_growth not in ("uniform", "bond"):
        raise ValueError("rank_growth must be 'uniform' or 'bond'")
    if not shape or any(size < 1 for size in shape):
        raise ValueError("shape must contain positive dimensions")
    if int(max_rank) < 1 or int(sweeps) < 1 or int(start_rank) < 1:
        raise ValueError("max_rank and sweeps must be positive")
    if int(kick_rank) < 1:
        raise ValueError("kick_rank must be positive")
    if int(stagnation_sweeps) < 1 or int(max_restarts) < 0:
        raise ValueError("stagnation_sweeps must be positive and max_restarts nonnegative")
    if not np.isfinite(rtol) or rtol < 0.0 or int(validation) < 0:
        raise ValueError("rtol and validation must be nonnegative")

    if initial is not None and tuple(initial.get("shape", ())) != shape:
        raise ValueError("initial TT-cross state has an incompatible shape")
    rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    initial_ranks = None if initial is None else initial.get("ranks")
    if initial_ranks is None:
        rank_cap = max_rank if int(validation) == 0 else start_rank
        ranks = _ranks(shape, min(int(rank_cap), int(max_rank)))
    else:
        if len(initial_ranks) != len(shape) + 1:
            raise ValueError("initial TT-cross ranks have an incompatible shape")
        if (initial_ranks[0] != 1 or initial_ranks[-1] != 1
                or any(int(rank) != rank or rank < 1 for rank in initial_ranks)):
            raise ValueError("initial TT-cross ranks must be positive with unit boundaries")
        limits = _ranks(shape, max_rank)
        ranks = [
            min(int(rank), limit)
            for rank, limit in zip(initial_ranks, limits)
        ]
    if any(ranks[i + 1] > ranks[i] * shape[i]
           or ranks[i] > shape[i] * ranks[i + 1] for i in range(len(shape))):
        raise ValueError("initial TT-cross ranks violate adjacent dimension bounds")
    if cache_size is not None and (int(cache_size) != cache_size or cache_size < 1):
        raise ValueError("cache_size must be a positive integer or None")
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    cache = {} if initial is None else dict(initial.get("cache", {}))
    if cache_size is not None:
        for key in list(cache)[:-int(cache_size)]:
            del cache[key]
    for index, value in cache.items():
        if (len(index) != len(shape) or any(
                int(i) != i or not 0 <= i < dim for i, dim in zip(index, shape))
                or np.ndim(value) != 0 or not np.isfinite(value)):
            raise ValueError("initial cache must contain valid indices and finite scalars")
    initial_samples = len(cache)
    batch_calls = 0
    evaluations = 0

    def evaluate_many(indices):
        chunks = []
        for offset in range(0, len(indices), int(batch_size)):
            chunks.append(evaluate_chunk(indices[offset:offset+int(batch_size)]))
        return np.concatenate(chunks) if chunks else np.empty(0)

    def evaluate_chunk(indices):
        nonlocal batch_calls, evaluations
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
                if not np.isfinite(value):
                    raise ValueError("TT-cross evaluator must return a finite scalar")
                cache[index] = value.item()
            evaluations += len(missing)
        result = np.asarray([cache[index] for index in indices])
        if cache_size is not None:
            for key in list(cache)[:max(0, len(cache)-int(cache_size))]:
                del cache[key]
        return result

    if len(shape) == 1:
        values = evaluate_many([(i,) for i in range(shape[0])])
        info = {
            "backend": "native",
            "samples": len(cache),
            "new_samples": len(cache) - initial_samples,
            "batch_calls": batch_calls,
            "evaluations": evaluations,
            "sweeps": 0,
            "validation_error": 0.0,
            "validation_rms_error": 0.0,
            "ranks": (1, 1),
            "rank_history": (),
            "success": True,
            "message": "Exact one-dimensional evaluation.",
            "history": (),
            "restarts": 0,
            "validation_points": shape[0],
        }
        if return_state:
            info["state"] = {"shape": shape, "ranks": (1, 1),
                             "cache": dict(cache), "right_sets": (None, ((),))}
        return [values.reshape(1, shape[0], 1)], info

    left_sets = [[()]] + [None] * (len(shape) - 1)
    right_sets = [None] * len(shape) + [[()]]
    initial_right = None if initial is None else initial.get("right_sets")
    if initial_right is not None:
        if len(initial_right) != len(shape) + 1:
            raise ValueError("initial TT-cross suffix sets have an incompatible shape")
        for split in range(1, len(shape)):
            for index in initial_right[split] or ():
                if len(index) != len(shape) - split or any(
                        int(i) != i or not 0 <= i < dim
                        for i, dim in zip(index, shape[split:])):
                    raise ValueError("initial TT-cross suffix sets contain invalid indices")
    for split in range(1, len(shape)):
        existing = None if initial_right is None else initial_right[split]
        right_sets[split] = _resize_suffixes(
            existing, shape[split:], ranks[split], rng
        )

    total = prod(shape)
    nvalidation = min(int(validation), total)
    validation_indices = _resize_suffixes(None, shape, nvalidation, validation_rng)
    if probes is not None:
        probes = [tuple(int(i) for i in row) for row in probes]
        if any(len(row) != len(shape) or any(i < 0 or i >= n for i, n in zip(row, shape)) for row in probes):
            raise ValueError('probes must be valid integer multi-indices')
        validation_indices = list(dict.fromkeys(validation_indices+probes))
    completed = 0
    error = np.inf
    rms_error = np.inf
    rank_history = []
    history = []
    restarts = stale = 0
    previous_error = np.inf
    success = False
    message = "Sweep limit reached before sampled convergence."

    def errors(cores):
        exact = evaluate_many(validation_indices)
        fitted = np.asarray([tt_value(cores, index) for index in validation_indices])
        residual = np.abs(fitted - exact)
        if not np.all(np.isfinite(residual)):
            raise ValueError("TT-cross produced non-finite validation residuals")
        return (float(np.max(residual) / max(float(np.max(np.abs(exact))), 1.0)),
                float(np.linalg.norm(residual) / max(float(np.linalg.norm(exact)), 1.0)))

    for sweep in range(int(sweeps)):
        sampled_ranks = [0] * (len(shape)+1)
        def record_rank(matrix, split):
            if rank_growth == "bond":
                singular = np.linalg.svd(matrix,compute_uv=False)
                threshold = max(rtol/(10*len(shape)), np.finfo(float).eps*max(matrix.shape)) * max(1.,float(np.max(abs(matrix))))
                sampled_ranks[split] = max(sampled_ranks[split],int(np.count_nonzero(singular>threshold)))
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
            record_rank(matrix,site+1)
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
            record_rank(matrix,site)
            selected = _pivot_columns(matrix, ranks[site])
            right_sets[site] = [columns[item] for item in selected]

        cores = _cross_cores(shape, evaluate_many, left_sets, right_sets, ranks)
        completed = sweep + 1
        rank_history.append(tuple(ranks))
        if validation_indices:
            error, rms_error = errors(cores)
            if error <= rtol:
                fresh = _resize_suffixes(None, shape, nvalidation, validation_rng)
                validation_indices = list(dict.fromkeys(validation_indices + fresh))
                error, rms_error = errors(cores)
                success = error <= rtol
            history.append({"sweep": completed, "ranks": tuple(ranks),
                            "validation_error": error, "validation_rms_error": rms_error,
                            "samples": len(cache), "restarts": restarts,
                            "validation_points": len(validation_indices)})
            if rank_growth == "bond":
                history[-1]["sampled_bond_ranks"] = tuple(sampled_ranks)
            if success:
                message = "Accumulated and fresh sampled validation passed."
                break
        else:
            message = "Validation disabled; approximation accuracy is unchecked."
            break

        stale = stale + 1 if error >= 0.99 * previous_error else 0
        previous_error = error
        if sweep == int(sweeps) - 1:
            continue
        active = None if rank_growth == "uniform" else [s>=r for s,r in zip(sampled_ranks,ranks)]
        new_ranks = _grow_ranks(shape, ranks, max_rank, kick_rank, active)
        if rank_growth == "bond" and new_ranks == ranks:
            new_ranks = _grow_ranks(shape, ranks, max_rank, kick_rank)
            if history:
                history[-1]["exploratory_growth"] = new_ranks != ranks
        if new_ranks == ranks:
            if stale >= int(stagnation_sweeps):
                if restarts >= int(max_restarts):
                    message = "Stagnated at the rank cap; pivot restart budget exhausted."
                    break
                restarts += 1
                stale = 0
                previous_error = np.inf
                for split in range(1, len(shape)):
                    right_sets[split] = _resize_suffixes(
                        None, shape[split:], ranks[split], rng)
                fresh = _resize_suffixes(None, shape, nvalidation, validation_rng)
                validation_indices = list(dict.fromkeys(validation_indices + fresh))
            continue
        ranks = new_ranks
        stale = 0
        previous_error = np.inf
        for split in range(1, len(shape)):
            right_sets[split] = _resize_suffixes(
                right_sets[split], shape[split:], ranks[split], rng
            )

    info = {
        "backend": "native",
        "rank_growth": rank_growth,
        "samples": len(cache),
        "new_samples": len(cache) - initial_samples,
        "batch_calls": batch_calls,
        "evaluations": evaluations,
        "sweeps": completed,
        "validation_error": error,
        "validation_rms_error": rms_error,
        "ranks": tuple(ranks),
        "rank_history": tuple(rank_history),
        "success": success,
        "message": message,
        "history": tuple(history),
        "restarts": restarts,
        "validation_points": len(validation_indices),
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
        unfolding = block.reshape(ranks[site] * size, ranks[site + 1])
        # Interpolate an orthonormal basis: inverting sampled physical values
        # directly amplifies roundoff when a requested rank is redundant.
        basis, _r = qr(unfolding, mode="economic")
        rows = {prefix + (coordinate,): row * size + coordinate
                for row, prefix in enumerate(prefixes) for coordinate in range(size)}
        selected = [rows[index] for index in left_sets[site + 1]]
        core = basis @ _pinv(basis[selected])
        cores.append(core.reshape(ranks[site], size, ranks[site + 1]))
    return cores


__all__ = ["tt_cross", "tt_cross_tntorch", "tt_value"]
