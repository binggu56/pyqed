"""Packed block actions shared by tensor-network local eigensolvers."""

from __future__ import annotations

from collections import defaultdict
import importlib
import os

import numpy as np


def resolve_workers(workers, *, maximum=4):
    """Choose one parallel layer without oversubscribing threaded BLAS."""
    if isinstance(workers, str):
        if workers.strip().lower() != "auto":
            raise ValueError("workers must be a positive integer or 'auto'.")
        blas_threads = max(
            (
                int(os.environ.get(name, "1") or 1)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            ),
            default=1,
        )
        return 1 if blas_threads > 1 else max(
            1, min(int(maximum), int(os.cpu_count() or 1))
        )
    if isinstance(workers, (bool, np.bool_)):
        raise TypeError("workers must be a positive integer or 'auto'.")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be a positive integer or 'auto'.")
    return workers


def _array_module(device):
    device = str(device).strip().lower().replace("_", "-")
    if device in {"cpu", "numpy"}:
        return np, "cpu"
    if device in {"cuda", "gpu", "cupy"}:
        try:
            import cupy
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "device='cuda' requires CuPy; install a CuPy build matching CUDA."
            ) from error
        return cupy, "cuda"
    if device == "auto":
        try:  # pragma: no cover - depends on optional accelerator
            import cupy

            if cupy.cuda.runtime.getDeviceCount():
                return cupy, "cuda"
        except Exception:
            pass
        return np, "cpu"
    raise ValueError("device must be 'cpu', 'cuda', or 'auto'.")


class PackedBlockEffectiveOperator:
    """Execute a block-sparse matrix through grouped dense contractions.

    Blocks sharing an output row are stacked once.  Scalar and multi-vector
    actions then consume contiguous route tables rather than dispatching one
    Python operation per matrix block. Fixed blocks remain resident on the
    selected array backend for the lifetime of the operator.
    """

    def __init__(
        self,
        block_indices,
        blocks,
        *,
        dtype=None,
        compute_dtype=None,
        device="cpu",
    ):
        indices = tuple(np.asarray(item, dtype=np.intp) for item in block_indices)
        if not indices:
            raise ValueError("block_indices must not be empty.")
        width = int(indices[0].size)
        if width < 1 or any(item.shape != (width,) for item in indices):
            raise ValueError("all block index arrays must have equal positive size.")
        size = int(sum(item.size for item in indices))
        if sorted(np.concatenate(indices).tolist()) != list(range(size)):
            raise ValueError("block_indices must partition the flattened vector.")

        inferred = [np.asarray(value).dtype for value in blocks.values()]
        storage_dtype = np.dtype(
            np.result_type(*(inferred or [np.float64])) if dtype is None else dtype
        )
        if compute_dtype is None or str(compute_dtype).lower() in {"same", "native"}:
            compute_dtype = storage_dtype
        compute_dtype = np.dtype(compute_dtype)
        if storage_dtype.kind == "c" and compute_dtype.kind != "c":
            compute_dtype = np.dtype(
                np.complex64
                if compute_dtype.itemsize <= 4
                else np.complex128
            )
        if compute_dtype.kind not in "fc":
            raise TypeError("compute_dtype must be a real or complex floating dtype.")
        xp, resolved_device = _array_module(device)

        grouped = defaultdict(list)
        for pair, block in blocks.items():
            row, column = map(int, pair)
            if not (0 <= row < len(indices) and 0 <= column < len(indices)):
                raise ValueError("block route index is out of range.")
            value = np.asarray(block)
            if value.shape != (width, width):
                raise ValueError(f"blocks must have shape {(width, width)}.")
            grouped[row].append((column, value))

        routes = []
        compiled_columns = []
        compiled_matrices = []
        for row in sorted(grouped):
            entries = sorted(grouped[row], key=lambda item: item[0])
            columns = np.asarray([column for column, _value in entries], dtype=np.intp)
            matrices = np.stack([value for _column, value in entries]).astype(
                compute_dtype, copy=False
            )
            routes.append((row, columns, xp.asarray(columns), xp.asarray(matrices)))

        for row in range(len(indices)):
            entries = sorted(grouped.get(row, ()), key=lambda item: item[0])
            compiled_columns.append(
                np.asarray([column for column, _value in entries], dtype=np.int64)
            )
            compiled_matrices.append(
                np.concatenate(
                    [np.asarray(value, dtype=compute_dtype) for _column, value in entries],
                    axis=1,
                )
                if entries
                else np.empty((width, 0), dtype=compute_dtype)
            )

        self.block_indices = indices
        self._index_table = np.stack(indices)
        self._device_index_table = xp.asarray(self._index_table)
        self._routes = tuple(routes)
        self.block_width = width
        self.nblocks = len(indices)
        self.size = size
        self.shape = (size, size)
        self.dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.device = resolved_device
        self.xp = xp
        self._compiled_action = None
        self.backend = "cupy-einsum" if resolved_device == "cuda" else "numpy-einsum"
        if resolved_device == "cpu" and compute_dtype in {
            np.dtype(np.float64),
            np.dtype(np.complex128),
        }:
            try:
                cpp = importlib.import_module("pyqed.mps.cpp_davidson")
                action_type = getattr(cpp, "PackedGroupedAction", None)
                if action_type is not None:
                    self._compiled_action = action_type(
                        np.asarray(self._index_table, dtype=np.int64),
                        compiled_columns,
                        compiled_matrices,
                        compute_dtype.kind == "c",
                    )
                    self.backend = str(self._compiled_action.backend)
            except Exception:
                self._compiled_action = None

    def _host(self, value):
        if self.device == "cuda":  # pragma: no cover - optional dependency
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def matvecs(self, vectors):
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.size:
            raise ValueError(f"vectors must have shape (batch, {self.size}).")
        if self._compiled_action is not None and not (
            vectors.dtype.kind == "c" and self.compute_dtype.kind != "c"
        ):
            return np.asarray(self._compiled_action.matvecs(vectors))
        xp = self.xp
        action_dtype = self.compute_dtype
        if vectors.dtype.kind == "c" and action_dtype.kind != "c":
            action_dtype = np.dtype(
                np.complex64
                if action_dtype.itemsize <= 4
                else np.complex128
            )
        device_vectors = xp.asarray(vectors, dtype=action_dtype)
        inputs = device_vectors[:, self._device_index_table]
        outputs = xp.zeros(
            (vectors.shape[0], self.nblocks, self.block_width),
            dtype=action_dtype,
        )
        for row, _columns, device_columns, matrices in self._routes:
            selected = inputs[:, device_columns]
            outputs[:, row] = xp.einsum(
                "rij,brj->bi", matrices, selected, optimize=True
            )
        flat = xp.empty((vectors.shape[0], self.size), dtype=action_dtype)
        flat[:, self._device_index_table] = outputs
        return self._host(flat)

    def matvec(self, vector):
        vector = np.asarray(vector)
        if vector.size != self.size:
            raise ValueError(f"vector must contain {self.size} entries.")
        return self.matvecs(vector.reshape(1, -1))[0]

    def diagonal(self):
        if self._compiled_action is not None:
            return np.asarray(self._compiled_action.diagonal())
        diagonal = np.zeros(self.size, dtype=self.compute_dtype)
        for row, columns, _device_columns, matrices in self._routes:
            matches = np.nonzero(columns == row)[0]
            if matches.size:
                matrix = self._host(matrices[int(matches[0])])
                diagonal[self.block_indices[row]] = np.diag(matrix)
        return diagonal


__all__ = ["PackedBlockEffectiveOperator", "resolve_workers"]
