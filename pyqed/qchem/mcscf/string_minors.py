"""Exact spin-string minors with shared cofactor intermediates."""

from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np


@lru_cache(maxsize=8)
def _cofactor_plan(nbra, nket, electrons):
    """Cache only orbital index maps, never overlaps or CI coefficients."""
    rows = np.array(list(combinations(range(nbra), electrons)), dtype=np.intp)
    cols = np.array(list(combinations(range(nket), electrons)), dtype=np.intp)
    row_order = {tuple(row): i for i, row in enumerate(rows)}
    col_order = {tuple(row): i for i, row in enumerate(cols)}
    levels = []
    for size in range(electrons, 1, -1):
        previous_rows, row_parent = np.unique(rows[:, :-1], axis=0, return_inverse=True)
        previous_cols, col_parent = np.unique(
            np.concatenate([np.delete(cols, p, axis=1) for p in range(size)]),
            axis=0, return_inverse=True)
        levels.append((rows[:, -1], cols, row_parent, col_parent.reshape(size, len(cols))))
        rows, cols = previous_rows, previous_cols
    return rows[:, 0], cols[:, 0], tuple(reversed(levels)), row_order, col_order


def _cofactor_minors(matrix, plan, dtype):
    first_rows, first_cols, levels, _, _ = plan
    previous = np.asarray(matrix[np.ix_(first_rows, first_cols)], dtype=dtype)
    for last_rows, cols, row_parent, col_parent in levels:
        size = cols.shape[1]
        out = np.zeros((len(last_rows), len(cols)), dtype=dtype)
        # Bound each gathered temporary to 1 MiB; retain only adjacent levels.
        batch = max(1, (1024**2)//max(1, len(cols)*out.dtype.itemsize))
        for start in range(0, len(last_rows), batch):
            stop = start+batch
            parent = row_parent[start:stop, None]
            last = last_rows[start:stop, None]
            for p in range(size):
                term = (matrix[last, cols[None, :, p]]
                        * previous[parent, col_parent[None, p, :]])
                if (size-1+p) % 2:
                    out[start:stop] -= term
                else:
                    out[start:stop] += term
        previous = out
    return previous


def string_minor_matrix(matrix, bra_occ, ket_occ, dtype):
    """Return exact determinant minors in the supplied spin-string order.

    For complete fixed-electron string spaces up to half filling, shared
    Laplace expansions build minors from lower-order minors. Every cofactor
    term is retained, with no inverse, screening, small-step assumption, or
    singular-value truncation. Only adjacent numeric levels are retained;
    index plans are bounded-cache reusable across unrelated geometries.
    Selected, higher-occupancy and small spaces use bounded batched LU
    determinants. Both paths support complex and singular orbital matrices.

    This is an exact algebraic implementation of the Slater determinant
    metric in P.-O. Lowdin, Phys. Rev. 97, 1474-1489 (1955),
    doi:10.1103/PhysRev.97.1474, using standard cofactor expansion. It does
    not implement the truncated Leibniz formula of Lee et al., JCTC 15,
    882-891 (2019), doi:10.1021/acs.jctc.8b01049. Exactness is algebraic;
    floating-point accumulation still incurs roundoff.
    """
    matrix = np.asarray(matrix)
    if not len(bra_occ) or not len(ket_occ):
        return np.empty((len(bra_occ), len(ket_occ)), dtype=dtype)
    bra, ket = np.asarray(bra_occ, dtype=np.intp), np.asarray(ket_occ, dtype=np.intp)
    electrons = bra.shape[1]
    if (electrons >= 2 and ket.shape[1] == electrons
            and 2*electrons <= min(matrix.shape)
            and len(bra)*len(ket) >= 4096
            and len(bra) == comb(matrix.shape[0], electrons)
            and len(ket) == comb(matrix.shape[1], electrons)):
        plan = _cofactor_plan(*matrix.shape, electrons)
        try:
            row_index = np.array([plan[3][tuple(row)] for row in bra], dtype=np.intp)
            col_index = np.array([plan[4][tuple(row)] for row in ket], dtype=np.intp)
        except KeyError:
            pass
        else:
            minors = _cofactor_minors(matrix, plan, dtype)
            if (np.array_equal(row_index, np.arange(len(bra)))
                    and np.array_equal(col_index, np.arange(len(ket)))):
                return minors
            return minors[np.ix_(row_index, col_index)]
    return _batched_minors(matrix, bra, ket, dtype)


def _batched_minors(matrix, bra, ket, dtype):
    """Direct determinant reference and fallback with bounded gathering."""
    out = np.empty((len(bra), len(ket)), dtype=dtype)
    row_bytes = max(1, len(ket)*bra.shape[1]*ket.shape[1]*matrix.dtype.itemsize)
    batch = max(1, (32*1024**2)//row_bytes)
    for start in range(0, len(bra), batch):
        minors = matrix[bra[start:start+batch, None, :, None], ket[None, :, None, :]]
        out[start:start+batch] = np.linalg.det(minors)
    return out


def plasser_minor_matrix(matrix, bra_occ, ket_occ, dtype):
    """Exact LU/cofactor spin-string overlaps, adapted from Plasser et al.

    F. Plasser et al., J. Chem. Theory Comput. 12, 1207–1219 (2016),
    doi:10.1021/acs.jctc.5b01148, section 2.2. Ket strings sharing all but
    their final orbital reuse a single last-column Laplace expansion. Each
    cofactor is evaluated by pivoted LU (NumPy determinant), without further
    recursion or matrix inverses. A cubic operation estimate selects shared
    cofactors or direct LU for each group; this batching and cost rule are
    adaptations, not a reproduction of the authors' code. No CI truncation,
    core omission, screening or overlap orthogonalization is applied.
    Rectangular, selected, reordered and singular metrics are supported.
    Exactness is algebraic, subject to floating-point roundoff.
    """
    matrix = np.asarray(matrix)
    bra = np.asarray(bra_occ, dtype=np.intp)
    ket = np.asarray(ket_occ, dtype=np.intp)
    if not len(bra) or not len(ket):
        return np.empty((len(bra), len(ket)), dtype=dtype)
    n = bra.shape[1]
    if ket.shape[1] != n:
        raise ValueError('Bra and ket strings must have equal electron counts.')
    if n < 2:
        return _batched_minors(matrix, bra, ket, dtype)
    prefixes, inverse = np.unique(ket[:, :-1], axis=0, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(prefixes))]
    shared = [i for i, g in enumerate(groups)
              if n*(n-1)**3+len(g)*n < len(g)*n**3]
    shared_set = set(shared)
    direct = np.array([j for i, g in enumerate(groups) if i not in shared_set for j in g], dtype=np.intp)
    out = np.empty((len(bra), len(ket)), dtype=dtype)
    if len(direct):
        out[:, direct] = _batched_minors(matrix, bra, ket[direct], dtype)
    if not shared:
        return out
    # Batch distinct prefix groups together, bounding LU input to about 16 MiB.
    deleted_rows = np.array([np.delete(np.arange(n), i) for i in range(n)])
    signs = (-1.)**(np.arange(n)+n-1)
    row_batch = max(1, (16*1024**2)//max(1, len(shared)*n*(n-1)**2*matrix.dtype.itemsize))
    prefix = prefixes[shared]
    for start in range(0, len(bra), row_batch):
        rows = bra[start:start+row_batch]
        blocks = matrix[rows[:, None, :, None], prefix[None, :, None, :]]
        cofactors = np.linalg.det(blocks[:, :, deleted_rows, :])*signs
        for index, group in enumerate(shared):
            columns = groups[group]
            last = matrix[rows[:, :, None], ket[None, columns, -1]]
            out[start:start+len(rows), columns] = np.einsum('ri,rij->rj', cofactors[:, index], last)
    return out
