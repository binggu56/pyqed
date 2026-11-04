"""Factorized one-particle transition densities on unrestricted intermediates.

Evaluate <L|a_p^dagger a_q|R> as (a_p L)^T(a_q R). This is the exact
second-quantized identity, including intermediate determinants outside a RAS
reference. The supplied columns can be internally contracted excitation
vectors. No density cumulant or intermediate RAS projection is introduced.
"""

import numpy as np


def annihilation_factor(rows, columns, orbital, phases=None):
    """Return sorted N-1 bitstrings and their sparse amplitude columns."""
    from .casci import _annihilate_bit

    rows = np.asarray(rows)
    if rows.dtype != object:
        rows = rows.astype(np.uint64, copy=False)
        selected = np.flatnonzero(rows & np.uint64(1 << orbital))
        bits = rows[selected] ^ np.uint64(1 << orbital)
        signs = 1 - 2 * (
            np.bitwise_count(rows[selected] & np.uint64((1 << orbital) - 1)) & 1
        ).astype(np.int8)
    else:
        entries = [
            (i, *_annihilate_bit(int(row), orbital)) for i, row in enumerate(rows)
        ]
        entries = [(i, bit, sign) for i, bit, sign in entries if sign]
        selected = np.array([e[0] for e in entries], dtype=int)
        bits = np.array([e[1] for e in entries], dtype=object)
        signs = np.array([e[2] for e in entries])
    if phases is not None:
        signs = signs * phases[selected]
    order = np.argsort(bits)
    amplitudes = columns[selected[order]].multiply(signs[order, None]).tocsr()
    return bits[order], amplitudes, amplitudes.T.tocsr()


def contract_factors(left, right):
    """Contract factors on their common open intermediate determinant space."""
    lb, lc, lt = left
    rb, rc, _ = right
    if np.array_equal(lb, rb):
        return (lt @ rc).toarray()
    positions = np.searchsorted(lb, rb)
    ri = np.flatnonzero(positions < len(lb))
    ri = ri[lb[positions[ri]] == rb[ri]]
    li = positions[ri]
    l = lt if len(li) == len(lb) else lc[li].T.tocsr()
    r = rc if len(ri) == len(rb) else rc[ri]
    return (l @ r).toarray()


def one_body_moment(rows, columns, fock):
    """Exact contracted Fock moment; diagonal operators use weighted Grams."""
    n = len(fock)
    if np.array_equal(fock, np.diag(np.diag(fock))):
        energies = np.array(
            [
                sum(
                    fock[p, p] * (((int(b) >> p) & 1) + ((int(b) >> (p + n)) & 1))
                    for p in range(n)
                )
                for b in rows
            ]
        )
        return (
            columns.T.tocsr() @ columns.multiply(energies[:, None]).tocsr()
        ).toarray()
    from functools import lru_cache

    @lru_cache(maxsize=4)
    def factor(orbital):
        return annihilation_factor(rows, columns, orbital)

    result = np.zeros((columns.shape[1], columns.shape[1]))
    for p, q in zip(*np.nonzero(fock)):
        for spin in (0, 1):
            result += fock[p, q] * contract_factors(
                factor(p + spin * n), factor(q + spin * n)
            )
    return result
