"""Strictly one-site Abelian DMRG with AMEn/3S subspace expansion."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from pyqed.mps.abelian_direct import (
    AbelianLocalVectorLayout,
    AbelianSiteTensorData,
    abelian_tensor_data_tensordot,
    abelian_transpose_tensor_data,
)
from pyqed.mps.abelian_storage import abelian_environment_scalar
from pyqed.mps.mps import contract_from_left, initial_E, initial_F
from pyqed.mps.tdvp import (
    _advance_block_environment,
    _build_block_right_envs,
    _cached_block_heff_plan,
    _cpp_table_kernel,
    _make_planned_block_site_heff,
    _new_cpp_moving_environment,
)


def _sector_sort_key(value):
    labels = tuple(getattr(value, "labels", ()))
    components = tuple(getattr(value, "components", value if isinstance(value, tuple) else (value,)))
    return type(value).__name__, labels, tuple(repr(item) for item in components)


def _axis_dims(tensor, axis):
    dims = {}
    for key, block in tensor.data.items():
        qn = key[axis]
        dim = int(np.asarray(block).shape[axis])
        previous = dims.setdefault(qn, dim)
        if previous != dim:
            raise ValueError(f"inconsistent degeneracy for sector {qn!r} on axis {axis}")
    return dims


def _site_block_allowed(q_left, q_right, q_phys):
    try:
        return q_left + q_phys == q_right
    except TypeError:
        return False


def _fused_sector(old_sector, mpo_sector):
    try:
        return old_sector - mpo_sector
    except TypeError as exc:
        raise TypeError("DMRG3S currently requires Abelian additive sectors.") from exc


def _partial_left_expansion_python(site, mpo_site, left):
    """Return ``L W A`` with the outgoing MPO and MPS legs fused."""

    partial = abelian_tensor_data_tensordot(left, site, ([2], [0]))
    partial = abelian_tensor_data_tensordot(partial, mpo_site, ([0, 3], [0, 3]))

    pairs = {}
    for key, block in partial.data.items():
        q_old, q_mpo = key[1], key[2]
        q_new = _fused_sector(q_old, q_mpo)
        pair = (q_old, q_mpo)
        dim = int(block.shape[1]) * int(block.shape[2])
        previous = pairs.setdefault(q_new, {}).setdefault(pair, dim)
        if previous != dim:
            raise ValueError("inconsistent fused left-expansion degeneracy")

    offsets = {}
    totals = {}
    for q_new, sector_pairs in pairs.items():
        offset = 0
        offsets[q_new] = {}
        for pair in sorted(sector_pairs, key=lambda item: (_sector_sort_key(item[0]), _sector_sort_key(item[1]))):
            offsets[q_new][pair] = offset
            offset += sector_pairs[pair]
        totals[q_new] = offset

    data = OrderedDict()
    for key, block in partial.data.items():
        q_left, q_old, q_mpo, q_phys = key
        q_new = _fused_sector(q_old, q_mpo)
        out_key = (q_left, q_new, q_phys)
        shape = (int(block.shape[0]), totals[q_new], int(block.shape[3]))
        out = data.setdefault(out_key, np.zeros(shape, dtype=np.asarray(block).dtype))
        start = offsets[q_new][(q_old, q_mpo)]
        width = int(block.shape[1]) * int(block.shape[2])
        out[:, start : start + width, :] += np.asarray(block).reshape(
            int(block.shape[0]), width, int(block.shape[3])
        )

    return AbelianSiteTensorData(
        data,
        [
            sorted({key[0] for key in data}, key=_sector_sort_key),
            sorted(totals, key=_sector_sort_key),
            sorted({key[2] for key in data}, key=_sector_sort_key),
        ],
        site.dirs,
        copy=False,
    )


def _partial_right_expansion_python(site, mpo_site, right):
    """Return ``A W F`` with the incoming MPS and MPO legs fused."""

    partial = abelian_tensor_data_tensordot(site, right, ([1], [2]))
    partial = abelian_tensor_data_tensordot(partial, mpo_site, ([1, 2], [3, 1]))

    pairs = {}
    for key, block in partial.data.items():
        q_old, q_mpo = key[0], key[2]
        q_new = _fused_sector(q_old, q_mpo)
        pair = (q_old, q_mpo)
        dim = int(block.shape[0]) * int(block.shape[2])
        previous = pairs.setdefault(q_new, {}).setdefault(pair, dim)
        if previous != dim:
            raise ValueError("inconsistent fused right-expansion degeneracy")

    offsets = {}
    totals = {}
    for q_new, sector_pairs in pairs.items():
        offset = 0
        offsets[q_new] = {}
        for pair in sorted(sector_pairs, key=lambda item: (_sector_sort_key(item[0]), _sector_sort_key(item[1]))):
            offsets[q_new][pair] = offset
            offset += sector_pairs[pair]
        totals[q_new] = offset

    data = OrderedDict()
    for key, block in partial.data.items():
        q_old, q_right, q_mpo, q_phys = key
        q_new = _fused_sector(q_old, q_mpo)
        out_key = (q_new, q_right, q_phys)
        shape = (totals[q_new], int(block.shape[1]), int(block.shape[3]))
        out = data.setdefault(out_key, np.zeros(shape, dtype=np.asarray(block).dtype))
        start = offsets[q_new][(q_old, q_mpo)]
        width = int(block.shape[0]) * int(block.shape[2])
        fused = np.asarray(block).transpose(0, 2, 1, 3).reshape(
            width, int(block.shape[1]), int(block.shape[3])
        )
        out[start : start + width, :, :] += fused

    return AbelianSiteTensorData(
        data,
        [
            sorted(totals, key=_sector_sort_key),
            sorted({key[1] for key in data}, key=_sector_sort_key),
            sorted({key[2] for key in data}, key=_sector_sort_key),
        ],
        site.dirs,
        copy=False,
    )


def _native_expansion(payload, site):
    data = OrderedDict(
        (tuple(key), np.asarray(block))
        for key, block in dict(payload).items()
    )
    return AbelianSiteTensorData(
        data,
        [
            sorted({key[0] for key in data}, key=_sector_sort_key),
            sorted({key[1] for key in data}, key=_sector_sort_key),
            sorted({key[2] for key in data}, key=_sector_sort_key),
        ],
        site.dirs,
        copy=False,
    )


def _partial_left_expansion(site, mpo_site, left, *, sketch_rank=None, seed=0):
    kernel = _cpp_table_kernel("abelian_dmrg3s_left_expansion_data")
    if kernel is not None:
        try:
            rank = 0 if sketch_rank is None else max(1, int(sketch_rank))
            return _native_expansion(
                kernel(site, mpo_site, left, rank, int(seed)),
                site,
            )
        except Exception:
            pass
    return _partial_left_expansion_python(site, mpo_site, left)


def _partial_right_expansion(site, mpo_site, right, *, sketch_rank=None, seed=0):
    kernel = _cpp_table_kernel("abelian_dmrg3s_right_expansion_data")
    if kernel is not None:
        try:
            rank = 0 if sketch_rank is None else max(1, int(sketch_rank))
            return _native_expansion(
                kernel(site, mpo_site, right, rank, int(seed)),
                site,
            )
        except Exception:
            pass
    return _partial_right_expansion_python(site, mpo_site, right)


def _empty_expansion_like(site):
    return AbelianSiteTensorData(
        {},
        [list(site.qns[0]), list(site.qns[1]), list(site.qns[2])],
        site.dirs,
        copy=False,
    )


def _global_kept_counts(svds, max_bond_dim, cutoff):
    entries = []
    for sector, (_u, singular, _vh, _rows, _old_dim, _expanded_dim) in svds.items():
        for index, value in enumerate(singular):
            if float(value) > float(cutoff):
                entries.append((float(value), sector, int(index)))
    entries.sort(key=lambda item: (-item[0], _sector_sort_key(item[1]), item[2]))
    if max_bond_dim is not None:
        entries = entries[: max(1, int(max_bond_dim))]
    kept = {}
    for _value, sector, index in entries:
        kept[sector] = max(kept.get(sector, 0), index + 1)
    if not kept:
        sector = max(svds, key=lambda qn: float(svds[qn][1][0]))
        kept[sector] = 1
    return kept


def _factorization_qr(matrix):
    kernel = _cpp_table_kernel("lapack_qr")
    if kernel is not None:
        try:
            q, r = kernel(np.ascontiguousarray(matrix, dtype=np.complex128))
            return np.asarray(q), np.asarray(r)
        except Exception:
            pass
    return np.linalg.qr(matrix, mode="reduced")


def _factorization_svd(matrix):
    kernel = _cpp_table_kernel("lapack_svd")
    if kernel is not None:
        try:
            u, singular, vh = kernel(
                np.ascontiguousarray(matrix, dtype=np.complex128)
            )
            return np.asarray(u), np.asarray(singular), np.asarray(vh)
        except Exception:
            pass
    return np.linalg.svd(matrix, full_matrices=False)


def _residual_kept_counts(factorized, max_rank, rtol):
    entries = []
    largest = max(
        (
            float(singular[0])
            for _sector, singular, _payload in factorized
            if singular.size
        ),
        default=0.0,
    )
    threshold = max(0.0, float(rtol)) * largest
    for sector, singular, _payload in factorized:
        for index, value in enumerate(singular):
            if float(value) > threshold and float(value) > 0.0:
                entries.append((float(value), sector, int(index)))
    entries.sort(key=lambda item: (-item[0], _sector_sort_key(item[1]), item[2]))
    entries = entries[: max(0, int(max_rank))]
    kept = {}
    for _value, sector, index in entries:
        kept[sector] = max(kept.get(sector, 0), index + 1)
    return kept


def _compress_left_expansion(site, expansion, max_rank, rtol):
    old_dims = _axis_dims(site, 1)
    expansion_dims = _axis_dims(expansion, 1)
    sectors = sorted(expansion_dims, key=_sector_sort_key)
    factorized = []
    for sector in sectors:
        keys = sorted(
            {key for key in site.data if key[1] == sector}
            | {key for key in expansion.data if key[1] == sector},
            key=lambda key: tuple(_sector_sort_key(item) for item in key),
        )
        rows = []
        nrows = 0
        for key in keys:
            block = site.data.get(key, expansion.data.get(key))
            row_dim = int(block.shape[0]) * int(block.shape[2])
            rows.append(
                (key, nrows, row_dim, int(block.shape[0]), int(block.shape[2]))
            )
            nrows += row_dim
        old_dim = int(old_dims.get(sector, 0))
        sketch_dim = int(expansion_dims[sector])
        current = np.zeros((nrows, old_dim), dtype=complex)
        sketch = np.zeros((nrows, sketch_dim), dtype=complex)
        for key, start, row_dim, left_dim, phys_dim in rows:
            if key in site.data:
                current[start : start + row_dim] = np.asarray(
                    site.data[key]
                ).transpose(0, 2, 1).reshape(row_dim, old_dim)
            if key in expansion.data:
                sketch[start : start + row_dim] = np.asarray(
                    expansion.data[key]
                ).transpose(0, 2, 1).reshape(row_dim, sketch_dim)
        if old_dim:
            q_current, _ = _factorization_qr(current)
            sketch -= q_current @ (q_current.conj().T @ sketch)
        u, singular, _vh = _factorization_svd(sketch)
        factorized.append((sector, singular, (u, rows)))

    kept = _residual_kept_counts(factorized, max_rank, rtol)
    data = OrderedDict()
    total_weight = 0.0
    kept_weight = 0.0
    for sector, singular, (u, rows) in factorized:
        count = int(kept.get(sector, 0))
        total_weight += float(np.vdot(singular, singular).real)
        kept_weight += float(np.vdot(singular[:count], singular[:count]).real)
        if count == 0:
            continue
        weighted = u[:, :count] * singular[:count]
        for key, start, row_dim, left_dim, phys_dim in rows:
            data[key] = weighted[start : start + row_dim].reshape(
                left_dim, phys_dim, count
            ).transpose(0, 2, 1)
    compressed = AbelianSiteTensorData(
        data,
        [
            sorted({key[0] for key in data}, key=_sector_sort_key),
            sorted(kept, key=_sector_sort_key),
            sorted({key[2] for key in data}, key=_sector_sort_key),
        ],
        site.dirs,
        copy=False,
    )
    return compressed, {
        "enrichment_mode": "streamed_low_rank",
        "enrichment_sketch_states": int(sum(expansion_dims.values())),
        "enrichment_states": int(sum(kept.values())),
        "enrichment_discarded": (
            0.0 if total_weight == 0.0 else max(0.0, 1.0 - kept_weight / total_weight)
        ),
    }


def _compress_right_expansion(site, expansion, max_rank, rtol):
    old_dims = _axis_dims(site, 0)
    expansion_dims = _axis_dims(expansion, 0)
    sectors = sorted(expansion_dims, key=_sector_sort_key)
    factorized = []
    for sector in sectors:
        keys = sorted(
            {key for key in site.data if key[0] == sector}
            | {key for key in expansion.data if key[0] == sector},
            key=lambda key: tuple(_sector_sort_key(item) for item in key),
        )
        columns = []
        ncols = 0
        for key in keys:
            block = site.data.get(key, expansion.data.get(key))
            col_dim = int(block.shape[1]) * int(block.shape[2])
            columns.append(
                (key, ncols, col_dim, int(block.shape[1]), int(block.shape[2]))
            )
            ncols += col_dim
        old_dim = int(old_dims.get(sector, 0))
        sketch_dim = int(expansion_dims[sector])
        current = np.zeros((old_dim, ncols), dtype=complex)
        sketch = np.zeros((sketch_dim, ncols), dtype=complex)
        for key, start, col_dim, right_dim, phys_dim in columns:
            if key in site.data:
                current[:, start : start + col_dim] = np.asarray(
                    site.data[key]
                ).transpose(0, 2, 1).reshape(old_dim, col_dim)
            if key in expansion.data:
                sketch[:, start : start + col_dim] = np.asarray(
                    expansion.data[key]
                ).transpose(0, 2, 1).reshape(sketch_dim, col_dim)
        if old_dim:
            q_current, _ = _factorization_qr(current.T)
            sketch -= (sketch @ q_current) @ q_current.conj().T
        _u, singular, vh = _factorization_svd(sketch)
        factorized.append((sector, singular, (vh, columns)))

    kept = _residual_kept_counts(factorized, max_rank, rtol)
    data = OrderedDict()
    total_weight = 0.0
    kept_weight = 0.0
    for sector, singular, (vh, columns) in factorized:
        count = int(kept.get(sector, 0))
        total_weight += float(np.vdot(singular, singular).real)
        kept_weight += float(np.vdot(singular[:count], singular[:count]).real)
        if count == 0:
            continue
        weighted = singular[:count, None] * vh[:count]
        for key, start, col_dim, right_dim, phys_dim in columns:
            data[key] = weighted[:, start : start + col_dim].reshape(
                count, phys_dim, right_dim
            ).transpose(0, 2, 1)
    compressed = AbelianSiteTensorData(
        data,
        [
            sorted(kept, key=_sector_sort_key),
            sorted({key[1] for key in data}, key=_sector_sort_key),
            sorted({key[2] for key in data}, key=_sector_sort_key),
        ],
        site.dirs,
        copy=False,
    )
    return compressed, {
        "enrichment_mode": "streamed_low_rank",
        "enrichment_sketch_states": int(sum(expansion_dims.values())),
        "enrichment_states": int(sum(kept.values())),
        "enrichment_discarded": (
            0.0 if total_weight == 0.0 else max(0.0, 1.0 - kept_weight / total_weight)
        ),
    }


def _low_rank_left_expansion(
    site,
    mpo_site,
    left,
    *,
    rank,
    rtol,
    oversample,
    seed,
):
    sketch = _partial_left_expansion(
        site,
        mpo_site,
        left,
        sketch_rank=int(rank) + max(0, int(oversample)),
        seed=seed,
    )
    return _compress_left_expansion(site, sketch, rank, rtol)


def _low_rank_right_expansion(
    site,
    mpo_site,
    right,
    *,
    rank,
    rtol,
    oversample,
    seed,
):
    sketch = _partial_right_expansion(
        site,
        mpo_site,
        right,
        sketch_rank=int(rank) + max(0, int(oversample)),
        seed=seed,
    )
    return _compress_right_expansion(site, sketch, rank, rtol)


def _map_sector_factorizations(sectors, factor, executor):
    if executor is None or len(sectors) < 2:
        return [factor(sector) for sector in sectors]
    return list(executor.map(factor, sectors))


def _left_enriched_factorization(
    site,
    expansion,
    max_bond_dim,
    strength,
    cutoff,
    *,
    executor=None,
):
    old_dims = _axis_dims(site, 1)
    expansion_dims = _axis_dims(expansion, 1)
    sectors = sorted(set(old_dims) | set(expansion_dims), key=_sector_sort_key)

    def factor_sector(sector):
        keys = sorted(
            {key for key in site.data if key[1] == sector}
            | {key for key in expansion.data if key[1] == sector},
            key=lambda key: tuple(_sector_sort_key(item) for item in key),
        )
        rows = []
        nrows = 0
        for key in keys:
            block = site.data.get(key, expansion.data.get(key))
            row_dim = int(block.shape[0]) * int(block.shape[2])
            rows.append((key, nrows, row_dim, int(block.shape[0]), int(block.shape[2])))
            nrows += row_dim
        old_dim = int(old_dims.get(sector, 0))
        expanded_dim = int(expansion_dims.get(sector, 0))
        matrix = np.zeros((nrows, old_dim + expanded_dim), dtype=complex)
        for key, start, row_dim, left_dim, phys_dim in rows:
            if key in site.data:
                matrix[start : start + row_dim, :old_dim] = np.asarray(
                    site.data[key]
                ).transpose(0, 2, 1).reshape(row_dim, old_dim)
            if key in expansion.data:
                matrix[start : start + row_dim, old_dim:] = np.asarray(
                    expansion.data[key]
                ).transpose(0, 2, 1).reshape(row_dim, expanded_dim)

        if old_dim and expanded_dim:
            q_current, _ = _factorization_qr(matrix[:, :old_dim])
            candidate = matrix[:, old_dim:]
            matrix[:, old_dim:] = candidate - q_current @ (q_current.conj().T @ candidate)
        matrix[:, old_dim:] *= float(strength)
        u, singular, vh = _factorization_svd(matrix)
        return (
            sector,
            (u, singular, vh, rows, old_dim, expanded_dim),
            float(np.vdot(singular, singular).real),
        )

    factorized = _map_sector_factorizations(sectors, factor_sector, executor)
    svds = OrderedDict((sector, result) for sector, result, _weight in factorized)
    total_weight = sum(weight for _sector, _result, weight in factorized)

    kept = _global_kept_counts(svds, max_bond_dim, cutoff)
    site_data = OrderedDict()
    center_data = OrderedDict()
    combined_dims = {}
    discarded = 0.0
    kept_states = 0
    for sector, (u, singular, vh, rows, old_dim, expanded_dim) in svds.items():
        count = int(kept.get(sector, 0))
        combined_dims[sector] = old_dim + expanded_dim
        discarded += float(np.vdot(singular[count:], singular[count:]).real)
        if count == 0:
            continue
        kept_states += count
        for key, start, row_dim, left_dim, phys_dim in rows:
            block = u[start : start + row_dim, :count].reshape(
                left_dim, phys_dim, count
            ).transpose(0, 2, 1)
            site_data[key] = block
        center_data[(sector, sector)] = singular[:count, None] * vh[:count, :]

    qns = [
        sorted({key[0] for key in site_data}, key=_sector_sort_key),
        sorted(kept, key=_sector_sort_key),
        sorted({key[2] for key in site_data}, key=_sector_sort_key),
    ]
    left_site = AbelianSiteTensorData(site_data, qns, site.dirs, copy=False)
    center = AbelianSiteTensorData(
        center_data,
        [sorted(kept, key=_sector_sort_key), sorted(combined_dims, key=_sector_sort_key)],
        [-1, 1],
        copy=False,
    )
    return left_site, center, combined_dims, {
        "states_kept": kept_states,
        "truncation": 0.0 if total_weight == 0.0 else discarded / total_weight,
        "expanded_states": int(sum(expansion_dims.values())),
    }


def _right_enriched_factorization(
    site,
    expansion,
    max_bond_dim,
    strength,
    cutoff,
    *,
    executor=None,
):
    old_dims = _axis_dims(site, 0)
    expansion_dims = _axis_dims(expansion, 0)
    sectors = sorted(set(old_dims) | set(expansion_dims), key=_sector_sort_key)

    def factor_sector(sector):
        keys = sorted(
            {key for key in site.data if key[0] == sector}
            | {key for key in expansion.data if key[0] == sector},
            key=lambda key: tuple(_sector_sort_key(item) for item in key),
        )
        columns = []
        ncols = 0
        for key in keys:
            block = site.data.get(key, expansion.data.get(key))
            col_dim = int(block.shape[1]) * int(block.shape[2])
            columns.append((key, ncols, col_dim, int(block.shape[1]), int(block.shape[2])))
            ncols += col_dim
        old_dim = int(old_dims.get(sector, 0))
        expanded_dim = int(expansion_dims.get(sector, 0))
        matrix = np.zeros((old_dim + expanded_dim, ncols), dtype=complex)
        for key, start, col_dim, right_dim, phys_dim in columns:
            if key in site.data:
                matrix[:old_dim, start : start + col_dim] = np.asarray(
                    site.data[key]
                ).transpose(0, 2, 1).reshape(old_dim, col_dim)
            if key in expansion.data:
                matrix[old_dim:, start : start + col_dim] = np.asarray(
                    expansion.data[key]
                ).transpose(0, 2, 1).reshape(expanded_dim, col_dim)

        if old_dim and expanded_dim:
            q_current, _ = _factorization_qr(matrix[:old_dim, :].T)
            candidate = matrix[old_dim:, :]
            matrix[old_dim:, :] = candidate - (candidate @ q_current) @ q_current.conj().T
        matrix[old_dim:, :] *= float(strength)
        u, singular, vh = _factorization_svd(matrix)
        return (
            sector,
            (u, singular, vh, columns, old_dim, expanded_dim),
            float(np.vdot(singular, singular).real),
        )

    factorized = _map_sector_factorizations(sectors, factor_sector, executor)
    svds = OrderedDict((sector, result) for sector, result, _weight in factorized)
    total_weight = sum(weight for _sector, _result, weight in factorized)

    kept = _global_kept_counts(svds, max_bond_dim, cutoff)
    site_data = OrderedDict()
    center_data = OrderedDict()
    combined_dims = {}
    discarded = 0.0
    kept_states = 0
    for sector, (u, singular, vh, columns, old_dim, expanded_dim) in svds.items():
        count = int(kept.get(sector, 0))
        combined_dims[sector] = old_dim + expanded_dim
        discarded += float(np.vdot(singular[count:], singular[count:]).real)
        if count == 0:
            continue
        kept_states += count
        for key, start, col_dim, right_dim, phys_dim in columns:
            block = vh[:count, start : start + col_dim].reshape(
                count, phys_dim, right_dim
            ).transpose(0, 2, 1)
            site_data[key] = block
        center_data[(sector, sector)] = u[:, :count] * singular[:count]

    qns = [
        sorted(kept, key=_sector_sort_key),
        sorted({key[1] for key in site_data}, key=_sector_sort_key),
        sorted({key[2] for key in site_data}, key=_sector_sort_key),
    ]
    right_site = AbelianSiteTensorData(site_data, qns, site.dirs, copy=False)
    center = AbelianSiteTensorData(
        center_data,
        [sorted(combined_dims, key=_sector_sort_key), sorted(kept, key=_sector_sort_key)],
        [-1, 1],
        copy=False,
    )
    return center, right_site, combined_dims, {
        "states_kept": kept_states,
        "truncation": 0.0 if total_weight == 0.0 else discarded / total_weight,
        "expanded_states": int(sum(expansion_dims.values())),
    }


def _pad_site_left(site, combined_dims):
    old_dims = _axis_dims(site, 0)
    right_dims = _axis_dims(site, 1)
    phys_dims = _axis_dims(site, 2)
    dtype = np.result_type(*(np.asarray(block).dtype for block in site.data.values()), complex)
    data = OrderedDict()
    for q_left, total in combined_dims.items():
        old = int(old_dims.get(q_left, 0))
        for q_right, right_dim in right_dims.items():
            for q_phys, phys_dim in phys_dims.items():
                key = (q_left, q_right, q_phys)
                if key not in site.data and not _site_block_allowed(q_left, q_right, q_phys):
                    continue
                block = np.zeros((int(total), right_dim, phys_dim), dtype=dtype)
                if key in site.data:
                    block[:old] = site.data[key]
                data[key] = block
    return AbelianSiteTensorData(
        data,
        [sorted(combined_dims, key=_sector_sort_key), list(site.qns[1]), list(site.qns[2])],
        site.dirs,
        copy=False,
    )


def _pad_site_right(site, combined_dims):
    old_dims = _axis_dims(site, 1)
    left_dims = _axis_dims(site, 0)
    phys_dims = _axis_dims(site, 2)
    dtype = np.result_type(*(np.asarray(block).dtype for block in site.data.values()), complex)
    data = OrderedDict()
    for q_left, left_dim in left_dims.items():
        for q_right, total in combined_dims.items():
            old = int(old_dims.get(q_right, 0))
            for q_phys, phys_dim in phys_dims.items():
                key = (q_left, q_right, q_phys)
                if key not in site.data and not _site_block_allowed(q_left, q_right, q_phys):
                    continue
                block = np.zeros((left_dim, int(total), phys_dim), dtype=dtype)
                if key in site.data:
                    block[:, :old, :] = site.data[key]
                data[key] = block
    return AbelianSiteTensorData(
        data,
        [list(site.qns[0]), sorted(combined_dims, key=_sector_sort_key), list(site.qns[2])],
        site.dirs,
        copy=False,
    )


def _absorb_center_left(center, site):
    return abelian_tensor_data_tensordot(center, site, ([1], [0]))


def _absorb_center_right(site, center):
    merged = abelian_tensor_data_tensordot(site, center, ([1], [0]))
    return abelian_transpose_tensor_data(
        merged,
        (0, 2, 1),
        carrier=AbelianSiteTensorData,
    )


def _absorb_enriched_center_left(center, site, combined_dims):
    """Apply ``center @ pad(site)`` without materializing padded zero blocks."""

    old_dims = _axis_dims(site, 0)
    right_dims = _axis_dims(site, 1)
    phys_dims = _axis_dims(site, 2)
    center_by_sector = {key[1]: np.asarray(block) for key, block in center.data.items()}
    dtype = np.result_type(
        *(np.asarray(block).dtype for block in (*center.data.values(), *site.data.values())),
        complex,
    )
    data = OrderedDict()
    for q_left in sorted(combined_dims, key=_sector_sort_key):
        center_block = center_by_sector.get(q_left)
        if center_block is None:
            continue
        old = int(old_dims.get(q_left, 0))
        kept = int(center_block.shape[0])
        for q_right, right_dim in right_dims.items():
            for q_phys, phys_dim in phys_dims.items():
                key = (q_left, q_right, q_phys)
                if key not in site.data and not _site_block_allowed(
                    q_left, q_right, q_phys
                ):
                    continue
                if key in site.data and old:
                    block = np.tensordot(
                        center_block[:, :old],
                        np.asarray(site.data[key]),
                        axes=([1], [0]),
                    )
                else:
                    block = np.zeros((kept, right_dim, phys_dim), dtype=dtype)
                data[key] = block
    return AbelianSiteTensorData(
        data,
        [list(center.qns[0]), list(site.qns[1]), list(site.qns[2])],
        site.dirs,
        copy=False,
    )


def _absorb_enriched_center_right(site, center, combined_dims):
    """Apply ``pad(site) @ center`` without materializing padded zero blocks."""

    old_dims = _axis_dims(site, 1)
    left_dims = _axis_dims(site, 0)
    phys_dims = _axis_dims(site, 2)
    center_by_sector = {key[0]: np.asarray(block) for key, block in center.data.items()}
    dtype = np.result_type(
        *(np.asarray(block).dtype for block in (*center.data.values(), *site.data.values())),
        complex,
    )
    data = OrderedDict()
    for q_left, left_dim in left_dims.items():
        for q_right in sorted(combined_dims, key=_sector_sort_key):
            center_block = center_by_sector.get(q_right)
            if center_block is None:
                continue
            old = int(old_dims.get(q_right, 0))
            kept = int(center_block.shape[1])
            for q_phys, phys_dim in phys_dims.items():
                key = (q_left, q_right, q_phys)
                if key not in site.data and not _site_block_allowed(
                    q_left, q_right, q_phys
                ):
                    continue
                if key in site.data and old:
                    merged = np.tensordot(
                        np.asarray(site.data[key]),
                        center_block[:old, :],
                        axes=([1], [0]),
                    )
                    block = merged.transpose(0, 2, 1)
                else:
                    block = np.zeros((left_dim, kept, phys_dim), dtype=dtype)
                data[key] = block
    return AbelianSiteTensorData(
        data,
        [list(site.qns[0]), list(center.qns[1]), list(site.qns[2])],
        site.dirs,
        copy=False,
    )


def _one_site_davidson(
    site,
    left,
    mpo_site,
    right,
    *,
    tol,
    max_iter,
    restart_dim,
    backend="auto",
):
    layout = AbelianLocalVectorLayout.from_tensor(site)
    if layout.size == 0:
        raise ValueError("cannot optimize an empty one-site tensor")

    backend = str(backend).strip().lower().replace("_", "-")
    if backend not in {"auto", "cpp", "python"}:
        raise ValueError("one-site Davidson backend must be 'auto', 'cpp', or 'python'")

    def unpack(vector):
        return AbelianSiteTensorData(
            layout.unflatten_data(vector),
            layout.qns,
            layout.dirs,
            copy=False,
        )

    cpp_error = None
    if backend != "python":
        plan = _cached_block_heff_plan("site", site, left, mpo_site, right)
        if plan is not None and hasattr(plan, "davidson"):
            try:
                result = dict(
                    plan.davidson(
                        site,
                        left,
                        mpo_site,
                        right,
                        layout.layout,
                        tol=float(tol),
                        max_iter=max(1, int(max_iter)),
                        restart_dim=max(2, int(restart_dim)),
                        accept_unconverged=True,
                    )
                )
                if bool(result.get("accepted", False)):
                    vector = np.asarray(result.pop("vector"), dtype=np.complex128)
                    norm = float(np.linalg.norm(vector))
                    if vector.size != layout.size or norm <= 1.0e-14:
                        raise ValueError("packed C++ Davidson returned an invalid vector")
                    vector = vector / norm
                    info = {
                        "dimension": int(result.get("dimension", layout.size)),
                        "iterations": int(result.get("iterations", 0)),
                        "residual_norm": float(result.get("residual_norm", np.inf)),
                        "converged": bool(result.get("converged", False)),
                        "backend": str(result.get("backend", "cpp-u1-site-davidson")),
                        "basis_size": int(result.get("basis_size", 0)),
                        "restarts": int(result.get("restarts", 0)),
                        "routes": int(result.get("routes", 0)),
                        "workspace_reused": bool(result.get("workspace_reused", False)),
                    }
                    return float(np.real(result["energy"])), unpack(vector), info
                cpp_error = "packed C++ Davidson did not return an accepted vector"
            except Exception as exc:
                cpp_error = repr(exc)
        else:
            cpp_error = "packed C++ one-site Davidson is unavailable"
        if backend == "cpp":
            raise RuntimeError(cpp_error)

    apply_heff = _make_planned_block_site_heff(site, left, mpo_site, right)

    def matvec(vector):
        return layout.flatten_tensor(apply_heff(unpack(vector)), dtype=np.complex128)

    vector = layout.flatten_tensor(site, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError("one-site Davidson received a zero initial tensor")
    basis = [vector / norm]
    images = []
    max_subspace = max(2, int(restart_dim))
    best = None
    matvec_calls = 0
    converged = False

    for _iteration in range(max(1, int(max_iter))):
        while len(images) < len(basis):
            images.append(matvec(basis[len(images)]))
            matvec_calls += 1
        projected = np.array(
            [[np.vdot(v_i, h_j) for h_j in images] for v_i in basis],
            dtype=complex,
        )
        projected = 0.5 * (projected + projected.conj().T)
        values, vectors = np.linalg.eigh(projected)
        coeff = vectors[:, int(np.argmin(np.real(values)))]
        energy = values[int(np.argmin(np.real(values)))]
        ritz = sum((coeff[i] * basis[i] for i in range(len(basis))), np.zeros_like(basis[0]))
        hritz = sum((coeff[i] * images[i] for i in range(len(images))), np.zeros_like(images[0]))
        residual = hritz - energy * ritz
        residual_norm = float(np.linalg.norm(residual))
        best = (energy, ritz, residual_norm)
        if residual_norm <= float(tol):
            converged = True
            break

        correction = -residual
        if len(basis) >= max_subspace:
            ritz_norm = float(np.linalg.norm(ritz))
            hritz = hritz / ritz_norm
            ritz = ritz / ritz_norm
            basis = [ritz]
            images = [hritz]
        for _ in range(2):
            for trial in basis:
                correction -= trial * np.vdot(trial, correction)
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm <= 1.0e-13:
            break
        basis.append(correction / correction_norm)

    energy, vector, residual_norm = best
    vector = vector / np.linalg.norm(vector)
    info = {
        "dimension": int(layout.size),
        "iterations": int(matvec_calls),
        "residual_norm": float(residual_norm),
        "converged": bool(converged),
        "backend": "python-u1-site-davidson",
    }
    if cpp_error is not None:
        info["cpp_fallback"] = cpp_error
    return float(np.real(energy)), unpack(vector), info


def _normalized_energy(factors, mpo):
    norm_env = initial_E(mpo[0])
    energy_env = initial_E(mpo[0])
    for site, mpo_site in zip(factors, mpo):
        energy_env = contract_from_left(mpo_site, site, energy_env, site)
        identity_data = OrderedDict()
        zero = mpo_site.qns[0][0] - mpo_site.qns[0][0]
        for q_phys, dim in _axis_dims(site, 2).items():
            identity_data[(zero, zero, q_phys, q_phys)] = np.eye(dim).reshape(1, 1, dim, dim)
        identity = AbelianSiteTensorData(
            identity_data,
            [[zero], [zero], list(_axis_dims(site, 2)), list(_axis_dims(site, 2))],
            [-1, 1, 1, -1],
            copy=False,
        )
        norm_env = contract_from_left(identity, site, norm_env, site)
    norm = abelian_environment_scalar(norm_env)
    return float(np.real(abelian_environment_scalar(energy_env) / norm))


def one_site_dmrg3s(
    factors,
    mpo,
    max_bond_dim,
    nsweeps,
    *,
    target_qn,
    conv=1.0e-6,
    davidson_tol=1.0e-5,
    adaptive_solver=False,
    davidson_tol_initial=1.0e-3,
    davidson_max_iter=30,
    davidson_restart_dim=12,
    enrichment=1.0e-4,
    enrichment_decay=0.1,
    enrichment_cutoff=1.0e-9,
    enrich_trigger=None,
    enrich_rank=32,
    enrich_rtol=1.0e-7,
    enrich_oversample=8,
    enrich_seed=0,
    svd_cutoff=1.0e-14,
    workers=1,
    not_conv_err=True,
    verbose=0,
    sweep_callback=None,
):
    """Optimize a native Abelian MPS with strictly one-site DMRG3S.

    Local eigensolves use rank-3 U(1) block tensors and prefer the packed C++
    Davidson plan, with the Python solver as a safe fallback. Missing bond
    directions are supplied by the AMEn/3S partial residuals ``L W A`` and
    ``A W F``; no two-site effective Hamiltonian or dense sector projector is
    formed. By default, the fused residual leg is formed as a streamed
    CountSketch and compressed to at most ``enrich_rank`` states globally
    across all symmetry sectors. Set ``enrich_rank=None`` to recover the exact
    full-rank expansion.
    """

    factors = [site.copy() for site in factors]
    mpo = list(mpo)
    if not factors or len(factors) != len(mpo):
        raise ValueError("MPS and MPO must be non-empty and have equal lengths.")
    if target_qn is None:
        raise ValueError("DMRG3S requires an Abelian target_qn.")
    if not all(isinstance(site, AbelianSiteTensorData) for site in factors + mpo):
        raise TypeError("DMRG3S requires native AbelianSiteTensorData storage.")

    moving_environment = _new_cpp_moving_environment()
    workers = max(1, int(workers))
    adaptive_solver = bool(adaptive_solver)
    davidson_tol = float(davidson_tol)
    davidson_tol_initial = float(davidson_tol_initial)
    if not np.isfinite(davidson_tol) or davidson_tol < 0.0:
        raise ValueError("davidson_tol must be finite and nonnegative.")
    if (
        not np.isfinite(davidson_tol_initial)
        or davidson_tol_initial < davidson_tol
    ):
        raise ValueError(
            "davidson_tol_initial must be finite and at least davidson_tol."
        )
    if enrich_trigger is not None:
        enrich_trigger = float(enrich_trigger)
        if not np.isfinite(enrich_trigger) or enrich_trigger < 0.0:
            raise ValueError("enrich_trigger must be finite and nonnegative or None.")
    if enrich_rank is not None and int(enrich_rank) < 1:
        raise ValueError("enrich_rank must be positive or None for the exact expansion.")
    if float(enrich_rtol) < 0.0:
        raise ValueError("enrich_rtol must be nonnegative.")
    if int(enrich_oversample) < 0:
        raise ValueError("enrich_oversample must be nonnegative.")
    factorization_executor = (
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dmrg3s-sector")
        if workers > 1
        else None
    )
    previous_energy = None
    converged = False
    last_energy = None
    active_enrichment = float(enrichment)
    start_total = time.perf_counter()
    right_envs = None
    right_environment_builds = 0
    right_environment_reuses = 0
    last_relative_gain = float("inf")

    for sweep in range(max(0, int(nsweeps))):
        sweep_start = time.perf_counter()
        active_davidson_tol = davidson_tol
        if adaptive_solver:
            gain_tolerance = (
                davidson_tol_initial
                if not np.isfinite(last_relative_gain)
                else 0.1 * last_relative_gain
            )
            active_davidson_tol = max(
                davidson_tol,
                min(davidson_tol_initial, gain_tolerance),
            )
        enrichment_due = bool(
            enrich_trigger is None
            or sweep == 0
            or last_relative_gain <= enrich_trigger
        )
        local_enrichment = active_enrichment if enrichment_due else 0.0
        updates = []
        if right_envs is None:
            right_envs = _build_block_right_envs(
                factors,
                mpo,
                target_qn,
                moving_environment=moving_environment,
                env_plan_prefix=f"dmrg3s:{sweep}",
            )
            right_environment_source = "built"
            right_environment_builds += 1
        else:
            right_environment_source = "reused_from_reverse_sweep"
            right_environment_reuses += 1
        left_envs = [None] * len(factors)
        left = initial_E(mpo[0])
        left_envs[0] = left

        for site_index in range(len(factors) - 1):
            local_start = time.perf_counter()
            local_energy, factors[site_index], solver_info = _one_site_davidson(
                factors[site_index],
                left,
                mpo[site_index],
                right_envs[site_index + 1],
                tol=active_davidson_tol,
                max_iter=davidson_max_iter,
                restart_dim=davidson_restart_dim,
            )
            expansion_info = {}
            if local_enrichment == 0.0:
                expansion = _empty_expansion_like(factors[site_index])
            elif enrich_rank is None:
                expansion = _partial_left_expansion(
                    factors[site_index], mpo[site_index], left
                )
                expansion_info = {"enrichment_mode": "exact"}
            else:
                expansion, expansion_info = _low_rank_left_expansion(
                    factors[site_index],
                    mpo[site_index],
                    left,
                    rank=int(enrich_rank),
                    rtol=float(enrich_rtol),
                    oversample=int(enrich_oversample),
                    seed=int(enrich_seed) + 2 * (sweep * len(factors) + site_index),
                )
            left_site, center, combined_dims, split_info = _left_enriched_factorization(
                factors[site_index],
                expansion,
                max_bond_dim,
                local_enrichment,
                svd_cutoff,
                executor=factorization_executor,
            )
            split_info.update(expansion_info)
            factors[site_index] = left_site
            factors[site_index + 1] = _absorb_enriched_center_left(
                center,
                factors[site_index + 1],
                combined_dims,
            )
            left = _advance_block_environment(
                "left",
                mpo[site_index],
                left_site,
                left,
                left_site,
                moving_environment=moving_environment,
                plan_key=f"dmrg3s:{sweep}:left:{site_index}",
            )
            left_envs[site_index + 1] = left
            updates.append(
                {
                    "site": site_index,
                    "direction": "right",
                    "local_energy": local_energy,
                    "seconds": float(time.perf_counter() - local_start),
                    **solver_info,
                    **split_info,
                }
            )

        local_energy, factors[-1], solver_info = _one_site_davidson(
            factors[-1],
            left_envs[-1],
            mpo[-1],
            initial_F(mpo[-1], target_qn=target_qn),
            tol=active_davidson_tol,
            max_iter=davidson_max_iter,
            restart_dim=davidson_restart_dim,
        )
        updates.append(
            {
                "site": len(factors) - 1,
                "direction": "right",
                "local_energy": local_energy,
                **solver_info,
            }
        )

        right = initial_F(mpo[-1], target_qn=target_qn)
        next_right_envs = [None] * (len(factors) + 1)
        next_right_envs[len(factors)] = right
        for site_index in range(len(factors) - 1, 0, -1):
            local_start = time.perf_counter()
            local_energy, factors[site_index], solver_info = _one_site_davidson(
                factors[site_index],
                left_envs[site_index],
                mpo[site_index],
                right,
                tol=active_davidson_tol,
                max_iter=davidson_max_iter,
                restart_dim=davidson_restart_dim,
            )
            expansion_info = {}
            if local_enrichment == 0.0:
                expansion = _empty_expansion_like(factors[site_index])
            elif enrich_rank is None:
                expansion = _partial_right_expansion(
                    factors[site_index], mpo[site_index], right
                )
                expansion_info = {"enrichment_mode": "exact"}
            else:
                expansion, expansion_info = _low_rank_right_expansion(
                    factors[site_index],
                    mpo[site_index],
                    right,
                    rank=int(enrich_rank),
                    rtol=float(enrich_rtol),
                    oversample=int(enrich_oversample),
                    seed=int(enrich_seed) + 2 * (sweep * len(factors) + site_index) + 1,
                )
            center, right_site, combined_dims, split_info = _right_enriched_factorization(
                factors[site_index],
                expansion,
                max_bond_dim,
                local_enrichment,
                svd_cutoff,
                executor=factorization_executor,
            )
            split_info.update(expansion_info)
            factors[site_index] = right_site
            factors[site_index - 1] = _absorb_enriched_center_right(
                factors[site_index - 1],
                center,
                combined_dims,
            )
            right = _advance_block_environment(
                "right",
                mpo[site_index],
                right_site,
                right,
                right_site,
                moving_environment=moving_environment,
                plan_key=f"dmrg3s:{sweep}:right:{site_index}",
            )
            next_right_envs[site_index] = right
            updates.append(
                {
                    "site": site_index,
                    "direction": "left",
                    "local_energy": local_energy,
                    "seconds": float(time.perf_counter() - local_start),
                    **solver_info,
                    **split_info,
                }
            )

        local_energy, factors[0], solver_info = _one_site_davidson(
            factors[0],
            initial_E(mpo[0]),
            mpo[0],
            right,
            tol=active_davidson_tol,
            max_iter=davidson_max_iter,
            restart_dim=davidson_restart_dim,
        )
        updates.append(
            {
                "site": 0,
                "direction": "left",
                "local_energy": local_energy,
                **solver_info,
            }
        )
        right_envs = next_right_envs

        last_energy = float(local_energy)
        energy_change = None if previous_energy is None else abs(last_energy - previous_energy)
        last_relative_gain = (
            float("inf")
            if previous_energy is None
            else energy_change / max(1.0, abs(previous_energy))
        )
        converged = energy_change is not None and energy_change < float(conv)
        row = {
            "sweep": sweep,
            "direction": "both",
            "energy": last_energy,
            "energy_change": energy_change,
            "enrichment": active_enrichment,
            "enrichment_applied": local_enrichment,
            "enrichment_due": enrichment_due,
            "enrich_trigger": enrich_trigger,
            "adaptive_solver": adaptive_solver,
            "davidson_tol": active_davidson_tol,
            "truncation": max((float(item.get("truncation", 0.0)) for item in updates), default=0.0),
            "states_kept": max((int(item.get("states_kept", 0)) for item in updates), default=0),
            "gauge": "right",
            "updates": updates,
            "seconds": float(time.perf_counter() - sweep_start),
            "algorithm": "dmrg3s",
            "local_tensor_rank": 3,
            "energy_source": "canonical_boundary_local_problem",
            "right_environment_source": right_environment_source,
        }
        if sweep_callback is not None:
            sweep_callback(mps=factors, **row)
        if verbose:
            print(
                f"DMRG3S sweep {sweep + 1}: E={last_energy:.12f} "
                f"dE={energy_change if energy_change is not None else float('nan'):.3e} "
                f"alpha={active_enrichment:.2e}"
            )
        if converged:
            break
        previous_energy = last_energy
        active_enrichment *= float(enrichment_decay)
        if abs(active_enrichment) < float(enrichment_cutoff):
            active_enrichment = 0.0

    if last_energy is None:
        last_energy = _normalized_energy(factors, mpo)
    if factorization_executor is not None:
        factorization_executor.shutdown(wait=True)
    if not converged and not_conv_err:
        raise ValueError(
            "DMRG3S did not converge within the requested sweeps; increase nsweeps "
            "or set not_conv_err=False."
        )
    one_site_dmrg3s.last_profile = {
        "algorithm": "dmrg3s",
        "local_tensor_rank": 3,
        "elapsed_seconds": float(time.perf_counter() - start_total),
        "cpp_environment": moving_environment is not None,
        "factorization_workers": workers,
        "enrich_rank": None if enrich_rank is None else int(enrich_rank),
        "enrich_rtol": float(enrich_rtol),
        "enrich_oversample": int(enrich_oversample),
        "enrich_seed": int(enrich_seed),
        "adaptive_solver": adaptive_solver,
        "davidson_tol": davidson_tol,
        "davidson_tol_initial": davidson_tol_initial,
        "enrich_trigger": enrich_trigger,
        "energy_source": (
            "canonical_boundary_local_problem" if int(nsweeps) > 0 else "full_contraction"
        ),
        "right_environment_builds": right_environment_builds,
        "right_environment_reuses": right_environment_reuses,
    }
    return last_energy, factors, "right", converged


one_site_dmrg3s.last_profile = None
