#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:19:15 2024

@author: bingg
"""
import numpy as np
from scipy.special import entr

def reduce_dm(density_matrix, indices, check_state=False, c_dtype="complex128"):
    """Compute the density matrix from a state represented with a density matrix.

    Args:
        density_matrix (tensor_like): 2D or 3D density matrix tensor. This tensor should be of size ``(2**N, 2**N)`` or
            ``(batch_dim, 2**N, 2**N)``, for some integer number of wires``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))`` or ``(batch_dim, 2**len(indices), 2**len(indices))``

    .. seealso:: :func:`pennylane.math.reduce_statevector`, and :func:`pennylane.density_matrix`

    **Example**

    >>> x = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> reduce_dm(x, indices=[0])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> y = [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
    >>> reduce_dm(y, indices=[0])
    [[0.5+0.j 0.5+0.j]
     [0.5+0.j 0.5+0.j]]

    >>> reduce_dm(y, indices=[1])
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]]

    >>> z = tf.Variable([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=tf.complex128)
    >>> reduce_dm(z, indices=[1])
    tf.Tensor(
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)

    >>> x = np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ...               [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    >>> reduce_dm(x, indices=[1])
    array([[[1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j]],
           [[0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]])
    """
    density_matrix = cast(density_matrix, dtype=c_dtype)

    if check_state:
        _check_density_matrix(density_matrix)

    if len(np.shape(density_matrix)) == 2:
        batch_dim, dim = None, density_matrix.shape[0]
    else:
        batch_dim, dim = density_matrix.shape[:2]

    num_indices = int(np.log2(dim))
    consecutive_indices = list(range(num_indices))

    # Return the full density matrix if all the wires are given, potentially permuted
    if len(indices) == num_indices:
        return _permute_dense_matrix(density_matrix, consecutive_indices, indices, batch_dim)

    if batch_dim is None:
        density_matrix = qml.math.stack([density_matrix])

    # Compute the partial trace
    traced_wires = [x for x in consecutive_indices if x not in indices]
    density_matrix = partial_trace(density_matrix, traced_wires, c_dtype=c_dtype)

    if batch_dim is None:
        density_matrix = density_matrix[0]

    # Permute the remaining indices of the density matrix
    return _permute_dense_matrix(density_matrix, sorted(indices), indices, batch_dim)


def mutual_info(
    state,
    indices0,
    indices1,
    base=None,
    check_state=False,
    c_dtype="complex128",
):
    r"""Compute the mutual information between two subsystems given a state:

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system. It supports all interfaces
    (NumPy, Autograd, Torch, TensorFlow and Jax).

    Each state must be given as a density matrix. To find the mutual information given
    a pure state, call :func:`~.math.dm_from_state_vector` first.

    Args:
        state (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        indices0 (list[int]): List of indices in the first subsystem.
        indices1 (list[int]): List of indices in the second subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Mutual information between the subsystems

    **Examples**

    The mutual information between subsystems for a state vector can be returned as follows:

    >>> x = np.array([1, 0, 0, 1]) / np.sqrt(2)
    >>> x = qml.math.dm_from_state_vector(x)
    >>> qml.math.mutual_info(x, indices0=[0], indices1=[1])
    1.3862943611198906

    It is also possible to change the log basis.

    >>> qml.math.mutual_info(x, indices0=[0], indices1=[1], base=2)
    2.0

    Similarly the quantum state can be provided as a density matrix:

    >>> y = np.array([[1/2, 1/2, 0, 1/2], [1/2, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]])
    >>> qml.math.mutual_info(y, indices0=[0], indices1=[1])
    0.4682351577408206

    .. seealso:: :func:`~.math.vn_entropy`, :func:`pennylane.qinfo.transforms.mutual_info` and :func:`pennylane.mutual_info`

    Refs

    https://docs.pennylane.ai/en/stable/_modules/pennylane/math/quantum.html#mutual_info

    """

    # the subsystems cannot overlap
    if len([index for index in indices0 if index in indices1]) > 0:
        raise ValueError("Subsystems for computing mutual information must not overlap.")

    return _compute_mutual_info(
        state,
        indices0,
        indices1,
        base=base,
        check_state=check_state,
        c_dtype=c_dtype,
    )

# pylint: disable=too-many-arguments
def _compute_mutual_info(
    state,
    indices0,
    indices1,
    base=None,
    check_state=False,
    c_dtype="complex128",
):
    """Compute the mutual information between the subsystems."""
    all_indices = sorted([*indices0, *indices1])
    vn_entropy_1 = vn_entropy(
        state,
        indices=indices0,
        base=base,
        check_state=check_state,
        c_dtype=c_dtype,
    )
    vn_entropy_2 = vn_entropy(
        state,
        indices=indices1,
        base=base,
        check_state=check_state,
        c_dtype=c_dtype,
    )
    vn_entropy_12 = vn_entropy(
        state,
        indices=all_indices,
        base=base,
        check_state=check_state,
        c_dtype=c_dtype,
    )

    return vn_entropy_1 + vn_entropy_2 - vn_entropy_12

def vn_entropy(state, indices, base=None, check_state=False, c_dtype="complex128"):
    r"""Compute the Von Neumann entropy from a density matrix on a given subsystem. It supports all
    interfaces (NumPy, Autograd, Torch, TensorFlow and Jax).

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        state (tensor_like): Density matrix of shape ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)``.
        indices (list(int)): List of indices in the considered subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Von Neumann entropy of the considered subsystem.

    **Example**

    The entropy of a subsystem for any state vectors can be obtained. Here is an example for the
    maximally entangled state, where the subsystem entropy is maximal (default base for log is exponential).

    >>> x = [1, 0, 0, 1] / np.sqrt(2)
    >>> x = dm_from_state_vector(x)
    >>> vn_entropy(x, indices=[0])
    0.6931472

    The logarithm base can be switched to 2 for example.

    >>> vn_entropy(x, indices=[0], base=2)
    1.0

    .. seealso:: :func:`pennylane.qinfo.transforms.vn_entropy` and :func:`pennylane.vn_entropy`
    """
    density_matrix = reduce_dm(state, indices, check_state, c_dtype)
    entropy = _compute_vn_entropy(density_matrix, base)
    return entropy

def _compute_vn_entropy(density_matrix, base=None):
    """Compute the Von Neumann entropy from a density matrix

    Args:
        density_matrix (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` tensor for an integer `N`.
        base (float, int): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        float: Von Neumann entropy of the density matrix.

    **Example**

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_vn_entropy(x)
    0.6931472

    >>> x = [[1/2, 0], [0, 1/2]]
    >>> _compute_vn_entropy(x, base=2)
    1.0

    """
    # Change basis if necessary
    if base:
        div_base = np.log(base)
    else:
        div_base = 1

    evs = np.linalg.eigvalsh(density_matrix)
    evs = np.where(evs > 0, evs, 1.0)
    entropy = entr(evs) / div_base

    return entropy

if __name__ == '__main__':
    pass