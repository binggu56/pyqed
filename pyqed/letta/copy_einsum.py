"""Optional native equality-constrained three-operand contractions."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability depends on optional build artifacts.
    from . import _copy_einsum_cpp
except Exception:  # pragma: no cover
    _copy_einsum_cpp = None


def native_available() -> bool:
    """Return whether the optional copy-aware contraction kernel is importable."""
    return _copy_einsum_cpp is not None


def contract_copy_einsum(
    left,
    right,
    operator,
    left_labels,
    right_labels,
    operator_labels,
    output_labels,
    copy_labels,
    copy_dimensions,
):
    """Contract three operands with exact three-way equality constraints."""
    if _copy_einsum_cpp is None:
        raise RuntimeError("the optional LETTA copy-einsum extension is unavailable.")
    dtype = np.dtype(np.result_type(left, right, operator, np.float64))
    if dtype not in {np.dtype(np.float64), np.dtype(np.complex128)}:
        raise TypeError("native copy-einsum supports only float64 and complex128.")
    output_labels = tuple(int(label) for label in output_labels)
    if len(set(output_labels)) != len(output_labels):
        raise ValueError("raw copy-einsum output labels must be unique.")
    arrays = (
        np.ascontiguousarray(left, dtype=dtype),
        np.ascontiguousarray(right, dtype=dtype),
        np.ascontiguousarray(operator, dtype=dtype),
        np.ascontiguousarray(left_labels, dtype=np.intp),
        np.ascontiguousarray(right_labels, dtype=np.intp),
        np.ascontiguousarray(operator_labels, dtype=np.intp),
        np.ascontiguousarray(output_labels, dtype=np.intp),
        np.ascontiguousarray(copy_labels, dtype=np.intp).reshape(-1, 3),
        np.ascontiguousarray(copy_dimensions, dtype=np.intp),
    )
    return _copy_einsum_cpp.contract_copy_einsum(*arrays)


def contract_class_einsum(
    operands,
    operand_classes,
    output_classes,
    class_dimensions,
):
    """Contract operands whose axes reference precompiled equality classes."""
    if _copy_einsum_cpp is None:
        raise RuntimeError("the optional LETTA copy-einsum extension is unavailable.")
    operands = tuple(operands)
    operand_classes = tuple(operand_classes)
    if not operands or len(operand_classes) != len(operands):
        raise ValueError(
            "operands and operand_classes must have equal nonzero lengths."
        )
    dtype = np.dtype(np.result_type(*operands))
    if dtype not in {np.dtype(np.float64), np.dtype(np.complex128)}:
        raise TypeError(
            "native class-einsum supports only float64 and complex128."
        )
    arrays = tuple(
        np.asarray(value, dtype=dtype, order="C") for value in operands
    )
    classes = tuple(
        np.ascontiguousarray(value, dtype=np.intp) for value in operand_classes
    )
    return _copy_einsum_cpp.contract_class_einsum(
        arrays,
        classes,
        np.ascontiguousarray(output_classes, dtype=np.intp),
        np.ascontiguousarray(class_dimensions, dtype=np.intp),
    )


__all__ = [
    "contract_class_einsum",
    "contract_copy_einsum",
    "native_available",
]
