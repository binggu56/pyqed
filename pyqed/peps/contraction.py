"""Exact and boundary-MPS contractions for finite PEPS."""

from __future__ import annotations

import numpy as np

from pyqed.mps.mps import MPS, MPO

from .state import PEPS


def _double_layer(bra, ket, operator=None):
    bra = np.asarray(bra)
    ket = np.asarray(ket)
    if operator is None:
        operator = np.eye(bra.shape[4], ket.shape[4], dtype=np.result_type(bra, ket))
    operator = np.asarray(operator)
    if operator.shape != (bra.shape[4], ket.shape[4]):
        raise ValueError("local operator shape does not match the physical dimensions.")
    tensor = np.einsum(
        "lrudp,LRUDq,pq->lLrRuUdD",
        bra.conj(),
        ket,
        operator,
        optimize=True,
    )
    return tensor.reshape(
        bra.shape[0] * ket.shape[0],
        bra.shape[1] * ket.shape[1],
        bra.shape[2] * ket.shape[2],
        bra.shape[3] * ket.shape[3],
    )


def _layers(bra, ket, insertions):
    if bra.shape != ket.shape:
        raise ValueError("PEPS rectangles must match.")
    return [
        [
            _double_layer(
                bra.network_tensor((row, col)),
                ket.network_tensor((row, col)),
                insertions.get((row, col)),
            )
            for col in range(bra.ncols)
        ]
        for row in range(bra.nrows)
    ]


def _exact_contract(layers):
    operands = []
    next_label = 0
    horizontal = {}
    vertical = {}
    boundary_vectors = []
    nrows, ncols = len(layers), len(layers[0])
    for row in range(nrows):
        for col in range(ncols):
            labels = []
            dimensions = layers[row][col].shape
            for axis in range(4):
                if axis == 0 and col > 0:
                    label = horizontal[(row, col - 1)]
                elif axis == 2 and row > 0:
                    label = vertical[(row - 1, col)]
                else:
                    label = next_label
                    next_label += 1
                    if axis == 1 and col + 1 < ncols:
                        horizontal[(row, col)] = label
                    elif axis == 3 and row + 1 < nrows:
                        vertical[(row, col)] = label
                    else:
                        boundary_vectors.append((label, dimensions[axis]))
                labels.append(label)
            operands.extend((layers[row][col], labels))
    for label, dimension in boundary_vectors:
        operands.extend((np.ones(dimension), [label]))
    operands.append([])
    return np.einsum(*operands, optimize=True)


def _boundary_contract(layers, max_bond):
    ncols = len(layers[0])
    boundary = MPS([np.ones((1, 1, 1)) for _ in range(ncols)])
    for row in layers:
        transfer = MPO([tensor.transpose(0, 1, 3, 2) for tensor in row])
        boundary = transfer.apply(boundary, max_bond=max_bond)
    value = np.ones(1, dtype=np.result_type(*[factor.dtype for factor in boundary.factors]))
    for site in range(boundary.L):
        standard = boundary._get_std_B(site)
        if standard.shape[1] != 1:
            raise RuntimeError("final PEPS boundary did not close to scalar physical legs.")
        value = value @ standard[:, 0, :]
    return value[0]


def contract(bra, ket=None, *, insertions=None, method="boundary", max_bond=None):
    """Contract ``<bra| operators |ket>``.

    ``insertions`` maps ``(row, col)`` to local bra-by-ket operator matrices.
    """
    if not isinstance(bra, PEPS):
        raise TypeError("bra must be a PEPS.")
    ket = bra if ket is None else ket
    if not isinstance(ket, PEPS):
        raise TypeError("ket must be a PEPS.")
    layers = _layers(bra, ket, {} if insertions is None else dict(insertions))
    if method == "exact":
        return _exact_contract(layers)
    if method == "boundary":
        return _boundary_contract(layers, max_bond)
    raise ValueError("method must be 'exact' or 'boundary'.")


def overlap(bra, ket, *, method="boundary", max_bond=None):
    return contract(bra, ket, method=method, max_bond=max_bond)


def local_expectation(state, coordinate, operator, *, method="boundary", max_bond=None):
    numerator = contract(
        state,
        insertions={tuple(coordinate): np.asarray(operator)},
        method=method,
        max_bond=max_bond,
    )
    denominator = contract(state, method=method, max_bond=max_bond)
    if abs(denominator) < 1.0e-30:
        raise ValueError("cannot normalize an expectation value for a zero PEPS.")
    return numerator / denominator
