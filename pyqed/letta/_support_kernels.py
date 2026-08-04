"""Support-space Hamiltonian kernels for native pair LETTA tensors.

The optional C++ extension implements the same API as the NumPy reference
below.  Keeping the reference path here makes the native extension optional
and gives the low-level kernels a small, independently testable interface.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability depends on optional build artifacts.
    from . import _support_kernels_cpp
except Exception:  # pragma: no cover
    _support_kernels_cpp = None


def native_available() -> bool:
    """Return whether the optional C++ support kernels are importable."""
    return _support_kernels_cpp is not None


def _validate_backend(backend: str) -> str:
    backend = str(backend).strip().lower()
    if backend not in {"auto", "native", "python"}:
        raise ValueError("backend must be 'auto', 'native', or 'python'.")
    if backend == "native" and _support_kernels_cpp is None:
        raise RuntimeError("the optional LETTA support-kernel extension is unavailable.")
    return backend


def _validated_inputs(
    coords,
    left,
    right,
    bra_i,
    ket_i,
    bra_j,
    ket_j,
    entry_starts,
    entry_m,
    entry_n,
    entry_values,
):
    coords = np.ascontiguousarray(coords, dtype=np.int64)
    left = np.ascontiguousarray(left, dtype=np.complex128)
    right = np.ascontiguousarray(right, dtype=np.complex128)
    bra_i = np.ascontiguousarray(bra_i, dtype=np.int64)
    ket_i = np.ascontiguousarray(ket_i, dtype=np.int64)
    bra_j = np.ascontiguousarray(bra_j, dtype=np.int64)
    ket_j = np.ascontiguousarray(ket_j, dtype=np.int64)
    entry_starts = np.ascontiguousarray(entry_starts, dtype=np.int64)
    entry_m = np.ascontiguousarray(entry_m, dtype=np.int64)
    entry_n = np.ascontiguousarray(entry_n, dtype=np.int64)
    entry_values = np.ascontiguousarray(entry_values, dtype=np.complex128)

    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape (support_size, 4).")
    if left.ndim != 5 or right.ndim != 5:
        raise ValueError("left and right environments must be five-dimensional.")
    transition_arrays = (bra_i, ket_i, bra_j, ket_j)
    if any(array.ndim != 1 for array in transition_arrays):
        raise ValueError("physical transition arrays must be one-dimensional.")
    ntransitions = bra_i.size
    if any(array.size != ntransitions for array in transition_arrays[1:]):
        raise ValueError("physical transition arrays must have equal lengths.")
    if entry_starts.shape != (ntransitions + 1,):
        raise ValueError("entry_starts must have one more item than the transitions.")
    if entry_m.ndim != 1 or entry_n.ndim != 1 or entry_values.ndim != 1:
        raise ValueError("compact MPO entry arrays must be one-dimensional.")
    if entry_m.size != entry_n.size or entry_m.size != entry_values.size:
        raise ValueError("compact MPO entry arrays must have equal lengths.")
    if entry_starts.size and (
        entry_starts[0] != 0
        or entry_starts[-1] != entry_m.size
        or np.any(entry_starts[1:] < entry_starts[:-1])
    ):
        raise ValueError("entry_starts must delimit the compact MPO entries.")

    if left.shape[3] != left.shape[4] or right.shape[3] != right.shape[4]:
        raise ValueError("environment bra and ket physical dimensions must agree.")
    if coords.size:
        lower_ok = np.all(coords >= 0)
        upper = np.asarray(
            [
                min(left.shape[0], left.shape[1]),
                left.shape[3],
                right.shape[3],
                min(right.shape[0], right.shape[1]),
            ],
            dtype=np.int64,
        )
        if not lower_ok or np.any(coords >= upper):
            raise ValueError("coords contains an out-of-range support coordinate.")
    if ntransitions:
        if (
            np.any(bra_i < 0)
            or np.any(bra_i >= left.shape[3])
            or np.any(ket_i < 0)
            or np.any(ket_i >= left.shape[4])
            or np.any(bra_j < 0)
            or np.any(bra_j >= right.shape[3])
            or np.any(ket_j < 0)
            or np.any(ket_j >= right.shape[4])
        ):
            raise ValueError("a physical transition index is out of range.")
    if entry_m.size and (
        np.any(entry_m < 0)
        or np.any(entry_m >= left.shape[2])
        or np.any(entry_n < 0)
        or np.any(entry_n >= right.shape[2])
    ):
        raise ValueError("an MPO environment index is out of range.")

    return (
        coords,
        left,
        right,
        bra_i,
        ket_i,
        bra_j,
        ket_j,
        entry_starts,
        entry_m,
        entry_n,
        entry_values,
    )


def _physical_groups(coords):
    groups = {}
    for position, (_left, si, sj, _right) in enumerate(coords):
        groups.setdefault((int(si), int(sj)), []).append(position)
    return {
        key: np.asarray(positions, dtype=np.int64)
        for key, positions in groups.items()
    }


def _assemble_python(inputs):
    (
        coords,
        left,
        right,
        bra_i,
        ket_i,
        bra_j,
        ket_j,
        entry_starts,
        entry_m,
        entry_n,
        entry_values,
    ) = inputs
    result = np.zeros((coords.shape[0], coords.shape[0]), dtype=np.complex128)
    groups = _physical_groups(coords)
    for transition, physical in enumerate(zip(bra_i, ket_i, bra_j, ket_j)):
        pbra_i, pket_i, pbra_j, pket_j = map(int, physical)
        rows = groups.get((pbra_i, pbra_j))
        columns = groups.get((pket_i, pket_j))
        if rows is None or columns is None:
            continue
        row_coords = coords[rows]
        column_coords = coords[columns]
        block = np.zeros((rows.size, columns.size), dtype=np.complex128)
        for entry in range(
            int(entry_starts[transition]),
            int(entry_starts[transition + 1]),
        ):
            m = int(entry_m[entry])
            n = int(entry_n[entry])
            block += (
                entry_values[entry]
                * left[
                    row_coords[:, 0, None],
                    column_coords[None, :, 0],
                    m,
                    pbra_i,
                    pket_i,
                ]
                * right[
                    row_coords[:, 3, None],
                    column_coords[None, :, 3],
                    n,
                    pbra_j,
                    pket_j,
                ]
            )
        result[np.ix_(rows, columns)] += block
    return result


def _apply_python(inputs, vectors):
    (
        coords,
        left,
        right,
        bra_i,
        ket_i,
        bra_j,
        ket_j,
        entry_starts,
        entry_m,
        entry_n,
        entry_values,
    ) = inputs
    result = np.zeros_like(vectors, dtype=np.complex128)
    groups = _physical_groups(coords)
    for transition, physical in enumerate(zip(bra_i, ket_i, bra_j, ket_j)):
        pbra_i, pket_i, pbra_j, pket_j = map(int, physical)
        rows = groups.get((pbra_i, pbra_j))
        columns = groups.get((pket_i, pket_j))
        if rows is None or columns is None:
            continue
        row_coords = coords[rows]
        column_coords = coords[columns]
        block = np.zeros((rows.size, columns.size), dtype=np.complex128)
        for entry in range(
            int(entry_starts[transition]),
            int(entry_starts[transition + 1]),
        ):
            m = int(entry_m[entry])
            n = int(entry_n[entry])
            block += (
                entry_values[entry]
                * left[
                    row_coords[:, 0, None],
                    column_coords[None, :, 0],
                    m,
                    pbra_i,
                    pket_i,
                ]
                * right[
                    row_coords[:, 3, None],
                    column_coords[None, :, 3],
                    n,
                    pbra_j,
                    pket_j,
                ]
            )
        result[rows] += block @ vectors[columns]
    return result


def assemble_support_hamiltonian(
    coords,
    left,
    right,
    bra_i,
    ket_i,
    bra_j,
    ket_j,
    entry_starts,
    entry_m,
    entry_n,
    entry_values,
    *,
    backend="auto",
):
    r"""Build the exact dense support Hamiltonian.

    The returned matrix is

    .. math:: H_{\mathcal A}=P_{\mathcal A}^{\dagger}H_{\rm eff}P_{\mathcal A}.

    ``coords`` uses native pair-LETTA order
    ``(left_virtual, s_i, s_{i+1}, right_virtual)``.  The compact transition
    arrays describe each nonzero two-site MPO physical transition and the
    associated ``(left_mpo, right_mpo, value)`` entries.
    """
    backend = _validate_backend(backend)
    inputs = _validated_inputs(
        coords,
        left,
        right,
        bra_i,
        ket_i,
        bra_j,
        ket_j,
        entry_starts,
        entry_m,
        entry_n,
        entry_values,
    )
    if backend != "python" and _support_kernels_cpp is not None:
        return _support_kernels_cpp.assemble_dense(*inputs)
    return _assemble_python(inputs)


def apply_support_hamiltonian(
    coords,
    left,
    right,
    bra_i,
    ket_i,
    bra_j,
    ket_j,
    entry_starts,
    entry_m,
    entry_n,
    entry_values,
    vectors,
    *,
    backend="auto",
):
    r"""Apply ``P_A^H H_eff P_A`` to one vector or a batch of column vectors."""
    backend = _validate_backend(backend)
    inputs = _validated_inputs(
        coords,
        left,
        right,
        bra_i,
        ket_i,
        bra_j,
        ket_j,
        entry_starts,
        entry_m,
        entry_n,
        entry_values,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.complex128)
    if vectors.ndim not in {1, 2}:
        raise ValueError("vectors must have shape (support_size,) or (support_size, nvec).")
    if vectors.shape[0] != inputs[0].shape[0]:
        raise ValueError("vectors has the wrong support dimension.")
    if backend != "python" and _support_kernels_cpp is not None:
        return _support_kernels_cpp.apply_batched(*inputs, vectors)
    return _apply_python(inputs, vectors)


__all__ = [
    "apply_support_hamiltonian",
    "assemble_support_hamiltonian",
    "native_available",
]
