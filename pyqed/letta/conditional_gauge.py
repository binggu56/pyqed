"""Batched application of physical-conditioned LETTA bond gauges.

The metric square root remains in NumPy/SciPy.  This module accelerates the
many small, disjoint tensor transformations that follow it while retaining a
NumPy implementation as the exact reference path.
"""

from __future__ import annotations

from collections.abc import Iterable
import os

import numpy as np

try:  # Optional extension; the NumPy implementation below is always valid.
    from . import _conditional_gauge_cpp
except Exception:  # pragma: no cover - depends on optional build artifacts.
    _conditional_gauge_cpp = None


_NATIVE_DTYPES = {np.dtype(np.float64), np.dtype(np.complex128)}
_NATIVE_MAX_GROUP_SIZE = 8


def _disabled(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


CONDITIONAL_GAUGE_CPP_AVAILABLE = bool(
    _conditional_gauge_cpp is not None
    and not _disabled(os.environ.get("PYQED_LETTA_DISABLE_CPP_CONDITIONAL_GAUGE", "0"))
)


def _writable_contiguous(array, dtype) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    if not result.flags.c_contiguous or not result.flags.writeable:
        result = np.array(result, dtype=dtype, order="C", copy=True)
    return result


def _validate_tensor_pair(left, right) -> None:
    if left.ndim != 4:
        raise ValueError("left must be a rank-4 native LETTA pair tensor.")
    if right.ndim == 4:
        if right.shape[:2] != (left.shape[3], left.shape[2]):
            raise ValueError("neighboring pair tensors have incompatible bond dimensions.")
    elif right.ndim == 2:
        if right.shape != (left.shape[2], left.shape[3]):
            raise ValueError("terminal tensor dimensions are incompatible with left.")
    else:
        raise ValueError("right must be a rank-4 pair tensor or rank-2 terminal tensor.")


def _pack_transforms(transforms, *, shared_dim, bond_dim, dtype):
    records = tuple(transforms)
    states = np.empty(len(records), dtype=np.intp)
    offsets = np.empty(len(records) + 1, dtype=np.intp)
    offsets[0] = 0
    groups = []
    gauges = []
    inverses = []
    for record, transform in enumerate(records):
        try:
            shared_state, group, gauge, gauge_inverse = transform
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each transform must be (shared_state, group, gauge, gauge_inverse)."
            ) from exc
        shared_state = int(shared_state)
        if shared_state < 0 or shared_state >= shared_dim:
            raise ValueError("conditional gauge shared state is out of range.")
        group = np.asarray(group, dtype=np.intp)
        if group.ndim != 1 or group.size == 0:
            raise ValueError("each conditional gauge group must be a nonempty vector.")
        if np.any(group < 0) or np.any(group >= bond_dim):
            raise ValueError("conditional gauge group index is out of range.")
        if np.unique(group).size != group.size:
            raise ValueError("conditional gauge groups cannot contain duplicate indices.")
        expected = (group.size, group.size)
        gauge = np.asarray(gauge, dtype=dtype)
        gauge_inverse = np.asarray(gauge_inverse, dtype=dtype)
        if gauge.shape != expected or gauge_inverse.shape != expected:
            raise ValueError(f"conditional gauge matrices must have shape {expected}.")

        states[record] = shared_state
        groups.append(np.ascontiguousarray(group))
        gauges.append(np.ascontiguousarray(gauge).reshape(-1))
        inverses.append(np.ascontiguousarray(gauge_inverse).reshape(-1))
        offsets[record + 1] = offsets[record] + group.size

    if not records:
        return (
            states,
            offsets,
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=dtype),
            np.empty(0, dtype=dtype),
        )
    return (
        states,
        offsets,
        np.concatenate(groups),
        np.concatenate(gauges),
        np.concatenate(inverses),
    )


def _numpy_apply_packed(
    left,
    right,
    states,
    offsets,
    indices,
    gauges,
    inverses,
) -> None:
    matrix_offset = 0
    for record, shared_state in enumerate(states):
        start = int(offsets[record])
        stop = int(offsets[record + 1])
        group = indices[start:stop]
        size = stop - start
        matrix_stop = matrix_offset + size * size
        gauge = gauges[matrix_offset:matrix_stop].reshape(size, size)
        gauge_inverse = inverses[matrix_offset:matrix_stop].reshape(size, size)

        left_slice = left[:, :, int(shared_state), :]
        left_block = left_slice[:, :, group]
        left_slice[:, :, group] = np.tensordot(
            left_block,
            gauge,
            axes=([2], [0]),
        )
        if right.ndim == 4:
            right_slice = right[:, int(shared_state), :, :]
            right_block = right_slice[group, :, :]
            right_slice[group, :, :] = np.tensordot(
                gauge_inverse,
                right_block,
                axes=([1], [0]),
            )
        else:
            right_block = right[int(shared_state), group]
            right[int(shared_state), group] = gauge_inverse @ right_block
        matrix_offset = matrix_stop


def apply_conditional_gauges(
    left,
    right,
    transforms: Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    *,
    backend="auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a batch of state-preserving conditional bond gauges.

    Each transform is ``(shared_state, group, gauge, gauge_inverse)`` and acts
    as

    ``left[..., shared_state, group] @ gauge`` and
    ``gauge_inverse @ right[group, shared_state, ...]``.

    The returned tensors may be the input arrays themselves when their dtype,
    contiguity, and writeability already permit in-place application.
    """
    backend = str(backend).lower()
    if backend not in {"auto", "numpy", "cpp"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'cpp'.")

    left_input = np.asarray(left)
    right_input = np.asarray(right)
    _validate_tensor_pair(left_input, right_input)
    transforms = tuple(transforms)
    if not transforms:
        return left_input, right_input
    matrix_dtypes = [
        np.asarray(matrix).dtype
        for _state, _group, gauge, inverse in transforms
        for matrix in (gauge, inverse)
    ]
    dtype = np.result_type(left_input.dtype, right_input.dtype, *matrix_dtypes)
    left_output = _writable_contiguous(left_input, dtype)
    right_output = _writable_contiguous(right_input, dtype)
    states, offsets, indices, gauges, inverses = _pack_transforms(
        transforms,
        shared_dim=left_output.shape[2],
        bond_dim=left_output.shape[3],
        dtype=dtype,
    )

    use_cpp = bool(
        CONDITIONAL_GAUGE_CPP_AVAILABLE
        and dtype in _NATIVE_DTYPES
        and (
            backend == "cpp"
            or (
                backend == "auto"
                and int(np.max(np.diff(offsets), initial=0))
                <= _NATIVE_MAX_GROUP_SIZE
            )
        )
    )
    if backend == "cpp" and not use_cpp:
        if not CONDITIONAL_GAUGE_CPP_AVAILABLE:
            raise RuntimeError("the optional conditional-gauge C++ extension is unavailable.")
        raise TypeError("the C++ conditional-gauge backend requires float64 or complex128.")

    if use_cpp:
        _conditional_gauge_cpp.apply_conditional_gauges_inplace(
            left_output,
            right_output,
            states,
            offsets,
            indices,
            gauges,
            inverses,
        )
    else:
        _numpy_apply_packed(
            left_output,
            right_output,
            states,
            offsets,
            indices,
            gauges,
            inverses,
        )
    return left_output, right_output


__all__ = [
    "CONDITIONAL_GAUGE_CPP_AVAILABLE",
    "apply_conditional_gauges",
]
