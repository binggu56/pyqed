"""Nearest-neighbor gates and a finite-PEPS simple update."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .state import AbelianPEPSTensor, PEPS


def two_site_gate(hamiltonian, step, *, imaginary=True):
    """Return ``exp(-step H)`` or ``exp(-i step H)`` as a rank-four gate."""
    hamiltonian = np.asarray(hamiltonian)
    if hamiltonian.ndim == 4:
        d1, d2, e1, e2 = hamiltonian.shape
        if (d1, d2) != (e1, e2):
            raise ValueError("rank-four Hamiltonian must have (d1,d2,d1,d2) shape.")
        matrix = hamiltonian.reshape(d1 * d2, d1 * d2)
    elif hamiltonian.ndim == 2 and hamiltonian.shape[0] == hamiltonian.shape[1]:
        root = int(round(np.sqrt(hamiltonian.shape[0])))
        if root * root != hamiltonian.shape[0]:
            raise ValueError("matrix Hamiltonians currently require equal local dimensions.")
        d1 = d2 = root
        matrix = hamiltonian
    else:
        raise ValueError("Hamiltonian must be square rank two or paired rank four.")
    coefficient = -float(step) if imaginary else -1j * float(step)
    return expm(coefficient * matrix).reshape(d1, d2, d1, d2)


def _neighbor_axes(first, second):
    dr, dc = second[0] - first[0], second[1] - first[1]
    try:
        return {(0, 1): (1, 0), (0, -1): (0, 1), (1, 0): (3, 2), (-1, 0): (2, 3)}[(dr, dc)]
    except KeyError as error:
        raise ValueError("simple update requires nearest-neighbor coordinates.") from error


def _multiply_axis(tensor, values, axis):
    shape = [1] * tensor.ndim
    shape[axis] = len(values)
    return tensor * np.asarray(values).reshape(shape)


def _external_weighted(state, coordinate, excluded_axis):
    tensor = state.dense_tensor(coordinate).copy()
    row, col = coordinate
    neighbors = ((row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col))
    for axis, neighbor in enumerate(neighbors):
        if axis == excluded_axis or not (0 <= neighbor[0] < state.nrows and 0 <= neighbor[1] < state.ncols):
            continue
        values = state.bond_singular_values[state.bond_key(coordinate, neighbor)]
        tensor = _multiply_axis(tensor, values, axis)
    return tensor


def _remove_external_weights(state, tensor, coordinate, excluded_axis):
    row, col = coordinate
    neighbors = ((row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col))
    for axis, neighbor in enumerate(neighbors):
        if axis == excluded_axis or not (0 <= neighbor[0] < state.nrows and 0 <= neighbor[1] < state.ncols):
            continue
        values = state.bond_singular_values[state.bond_key(coordinate, neighbor)]
        inverse = np.zeros_like(values)
        mask = np.abs(values) > 1.0e-14
        inverse[mask] = 1.0 / values[mask]
        tensor = _multiply_axis(tensor, inverse, axis)
    return tensor


def _restore_tensor(core, connection_axis, free_axes):
    # core ordering is free virtual legs, physical, new connection.
    physical_axis = len(free_axes)
    connection_source = physical_axis + 1
    sources = {axis: index for index, axis in enumerate(free_axes)}
    sources[connection_axis] = connection_source
    permutation = [sources[axis] for axis in range(4)] + [physical_axis]
    return core.transpose(permutation)


def _restore_right_tensor(core, connection_axis, free_axes):
    # core ordering is new connection, free virtual legs, physical.
    sources = {axis: index + 1 for index, axis in enumerate(free_axes)}
    sources[connection_axis] = 0
    physical_axis = len(free_axes) + 1
    permutation = [sources[axis] for axis in range(4)] + [physical_axis]
    return core.transpose(permutation)


def simple_update_bond(
    state,
    first,
    second,
    gate,
    *,
    max_bond,
    cutoff=0.0,
    normalize=True,
):
    """Apply one two-site gate and truncate the connecting PEPS bond."""
    if not isinstance(state, PEPS):
        raise TypeError("state must be a PEPS.")
    if any(isinstance(state.tensors[r][c], AbelianPEPSTensor) for r, c in (first, second)):
        raise NotImplementedError("charge-preserving block SVD is not yet implemented for AbelianPEPS.")
    first, second = state._coordinate(first), state._coordinate(second)
    axis_a, axis_b = _neighbor_axes(first, second)
    a = _external_weighted(state, first, axis_a)
    b = _external_weighted(state, second, axis_b)
    bond = state.bond_singular_values[state.bond_key(first, second)]
    a = _multiply_axis(a, bond, axis_a)
    theta = np.tensordot(a, b, axes=([axis_a], [axis_b]))
    free_a = [axis for axis in range(4) if axis != axis_a]
    free_b = [axis for axis in range(4) if axis != axis_b]
    gate = np.asarray(gate)
    if gate.shape[2:] != (a.shape[4], b.shape[4]):
        raise ValueError("gate input dimensions do not match the two physical legs.")
    theta = np.tensordot(gate, theta, axes=([2, 3], [3, 7]))
    theta = theta.transpose(2, 3, 4, 0, 5, 6, 7, 1)
    left_shape = tuple(a.shape[axis] for axis in free_a) + (gate.shape[0],)
    right_shape = tuple(b.shape[axis] for axis in free_b) + (gate.shape[1],)
    matrix = theta.reshape(int(np.prod(left_shape)), int(np.prod(right_shape)))
    u, all_singular, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = min(int(max_bond), all_singular.size)
    if cutoff:
        keep = min(keep, max(1, int(np.count_nonzero(all_singular > cutoff))))
    u, singular, vh = u[:, :keep], all_singular[:keep], vh[:keep]
    if normalize and np.linalg.norm(singular):
        singular = singular / np.linalg.norm(singular)
    a_new = _restore_tensor(u.reshape(*left_shape, keep), axis_a, free_a)
    b_new = _restore_right_tensor(vh.reshape(keep, *right_shape), axis_b, free_b)
    state.tensors[first[0]][first[1]] = _remove_external_weights(state, a_new, first, axis_a)
    state.tensors[second[0]][second[1]] = _remove_external_weights(state, b_new, second, axis_b)
    state.bond_singular_values[state.bond_key(first, second)] = singular
    state._check_bonds()
    discarded = float(np.sum(np.square(np.abs(all_singular[keep:]))))
    return discarded


def simple_update_sweep(
    state,
    gates,
    *,
    max_bond,
    cutoff=0.0,
    normalize=True,
):
    """Apply ``(first, second, gate)`` entries sequentially in place."""
    discarded = 0.0
    for first, second, gate in gates:
        discarded += simple_update_bond(
            state,
            first,
            second,
            gate,
            max_bond=max_bond,
            cutoff=cutoff,
            normalize=normalize,
        )
    return discarded
