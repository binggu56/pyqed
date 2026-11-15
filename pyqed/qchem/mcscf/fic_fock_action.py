"""Shared-index Fock action for internally contracted perturbation theory."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _accumulate_fock_edges(
    rows,
    columns,
    labels,
    weights,
    source,
    target,
    forward,
    reverse,
    forward_sign,
    reverse_sign,
):
    cross = labels.shape[0] == 2
    for k in range(len(rows)):
        row, column = rows[k], columns[k]
        value = weights[labels[0, k], labels[1, k] if cross else 0]
        if abs(value) >= 1e-14:
            fval, rval = forward_sign * value, reverse_sign * value
            for channel in range(source.shape[1]):
                forward[row, channel] += fval * source[column, channel]
                reverse[column, channel] += rval * target[row, channel]


@njit(cache=True, nogil=True)
def _subtract_fock_edges(indices, labels, weights, products, target):
    cross = labels.shape[0] == 2
    for k in range(len(indices)):
        value = weights[labels[0, k], labels[1, k] if cross else 0]
        if abs(value) >= 1e-14:
            for channel in range(products.shape[1]):
                target[indices[k], channel] -= value * products[k, channel]


def apply_compact_fock_graph(x, out, terms, gathered, projected):
    """Contract edge batches in both directions using bounded scratch arrays.

    Signed weights and all couplings are retained. Whole-class projections
    are shared when they fit the panels. Otherwise edge batches may project
    repeated input rows more than once to keep scratch storage bounded.
    """
    for s, t, rows, columns, labels, weights, matrix in terms:
        xs, xt = x[s], x[t]
        rs, rt = xs.shape[1], xt.shape[1]
        ns, nt = len(xs), len(xt)
        size = ns * rt if rt <= rs else nt * rs
        if size <= min(len(gathered), len(projected)):
            if rt <= rs:
                w = gathered[:size].reshape(ns, rt)
                z = projected[:size].reshape(ns, rt)
                np.matmul(xs, matrix.T, out=w)
                z.fill(0.0)
                _accumulate_fock_edges(
                    rows, columns, labels, weights, w, xt, out[t], z, -1.0, 1.0
                )
                target, lhs, rhs = out[s], z, matrix
            else:
                w = gathered[:size].reshape(nt, rs)
                z = projected[:size].reshape(nt, rs)
                w.fill(0.0)
                np.matmul(xt, matrix, out=z)
                _accumulate_fock_edges(
                    rows, columns, labels, weights, xs, z, w, out[s], 1.0, -1.0
                )
                target, lhs, rhs = out[t], w, matrix.T
            count = max(1, len(gathered) // target.shape[1])
            for first in range(0, len(target), count):
                target[first : first + count] -= lhs[first : first + count] @ rhs
            continue
        block = max(1, min(len(gathered), len(projected)) // max(rs, rt))
        for first in range(0, len(rows), block):
            last = min(first + block, len(rows))
            count = last - first
            source = gathered[: count * rs].reshape(count, rs)
            product = projected[: count * rt].reshape(count, rt)
            np.take(xs, columns[first:last], axis=0, out=source)
            np.matmul(source, matrix.T, out=product)
            _subtract_fock_edges(
                rows[first:last], labels[:, first:last], weights, product, out[t]
            )
            source = gathered[: count * rt].reshape(count, rt)
            product = projected[: count * rs].reshape(count, rs)
            np.take(xt, rows[first:last], axis=0, out=source)
            np.matmul(source, matrix, out=product)
            _subtract_fock_edges(
                columns[first:last], labels[:, first:last], weights, product, out[s]
            )
