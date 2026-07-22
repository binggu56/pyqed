"""Backend boundary for fast SU(2)-NARG numeric kernels.

The SU(2)-NARG physics code should stay in Python: it owns the growth order,
spin/charge sectors, reduced tensor metadata, and truncation decisions.  This
module is the narrow numeric boundary where sector matvecs, Davidson solves,
and reduced-operator projections can be replaced by compiled kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import numpy as np
from scipy.linalg import eigh


ROTATION_BATCH_MIN_BLOCKS = int(os.environ.get("SU2_NARG_ROTATION_BATCH_MIN_BLOCKS", "8"))


def _as_hermitian(block) -> np.ndarray:
    array = np.asarray(block, dtype=np.complex128)
    return 0.5 * (array + array.T.conj())


@dataclass(frozen=True)
class SU2NARGCapabilities:
    """Feature flags exposed by an SU(2)-NARG backend."""

    sector_matvec: bool = False
    davidson: bool = False
    operator_projection: str = "batched-blas"


@dataclass(frozen=True)
class SectorDiagonalization:
    """Eigenpairs returned by a sector solver."""

    values: np.ndarray
    vectors: np.ndarray
    method: str
    info: dict[str, Any] = field(default_factory=dict)


class SU2NARGBackend:
    """Pure-Python/BLAS backend used as the reference implementation."""

    name = "python"
    capabilities = SU2NARGCapabilities()

    def sector_matvec(self, block, vector) -> np.ndarray:
        return np.asarray(block) @ np.asarray(vector)

    def diagonalize_sector(self, block, nroots: int | None = None) -> SectorDiagonalization:
        hermitian = _as_hermitian(block)
        values, vectors = eigh(hermitian, check_finite=False)
        if nroots is not None:
            nroots = max(0, min(int(nroots), values.size))
            values = values[:nroots]
            vectors = vectors[:, :nroots]
        return SectorDiagonalization(values=values, vectors=vectors, method="dense-eigh")

    def rotate_operator_block(self, u_bra, old_block, u_ket) -> np.ndarray:
        return np.asarray(u_bra).conj().T @ np.asarray(old_block) @ np.asarray(u_ket)

    def rotate_operator_blocks(self, block_specs) -> list[tuple[Any, np.ndarray]]:
        """Rotate a batch of ``(key, U_bra, O, U_ket)`` block specs."""
        block_specs = list(block_specs)
        if not block_specs:
            return []

        grouped: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], list] = {}
        for index, (key, u_bra, old_block, u_ket) in enumerate(block_specs):
            u_bra_array = np.asarray(u_bra)
            old_array = np.asarray(old_block)
            u_ket_array = np.asarray(u_ket)
            shape_key = (u_bra_array.shape, old_array.shape, u_ket_array.shape)
            grouped.setdefault(shape_key, []).append(
                (index, key, u_bra_array, old_array, u_ket_array)
            )

        rotated: list[tuple[Any, np.ndarray] | None] = [None] * len(block_specs)
        for group in grouped.values():
            if len(group) < ROTATION_BATCH_MIN_BLOCKS:
                for index, key, u_bra, old_block, u_ket in group:
                    rotated[index] = (key, self.rotate_operator_block(u_bra, old_block, u_ket))
                continue

            u_bras = np.stack([item[2] for item in group], axis=0)
            old_blocks = np.stack([item[3] for item in group], axis=0)
            u_kets = np.stack([item[4] for item in group], axis=0)
            batch = np.matmul(np.matmul(u_bras.conj().transpose(0, 2, 1), old_blocks), u_kets)
            for offset, (index, key, _, _, _) in enumerate(group):
                rotated[index] = (key, batch[offset])

        return [item for item in rotated if item is not None]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sector_matvec": self.capabilities.sector_matvec,
            "davidson": self.capabilities.davidson,
            "operator_projection": self.capabilities.operator_projection,
        }


class CompiledSU2NARGBackend(SU2NARGBackend):
    """Backend using the existing C++ block-table Davidson kernels when possible."""

    name = "compiled"

    def __init__(
        self,
        *,
        require: bool = False,
        davidson_min_dim: int = 96,
        davidson_tol: float = 1.0e-12,
        davidson_max_iter: int = 80,
        davidson_restart_dim: int = 24,
        accept_unconverged: bool = False,
    ):
        self.davidson_min_dim = int(davidson_min_dim)
        self.davidson_tol = float(davidson_tol)
        self.davidson_max_iter = int(davidson_max_iter)
        self.davidson_restart_dim = int(davidson_restart_dim)
        self.accept_unconverged = bool(accept_unconverged)
        self._cpp = self._load_cpp(require=require)
        self.capabilities = SU2NARGCapabilities(
            sector_matvec=self._cpp is not None,
            davidson=self._cpp is not None,
            operator_projection="batched-blas",
        )

    @staticmethod
    def _load_cpp(*, require: bool):
        try:
            from pyqed.mps import cpp_davidson
        except Exception as exc:
            if require:
                raise ImportError("compiled SU2-NARG backend requested, but C++ kernels failed to import") from exc
            return None
        if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
            if require:
                detail = getattr(cpp_davidson, "CPP_DAVIDSON_BUILD_ERROR", None)
                raise ImportError(f"compiled SU2-NARG backend unavailable: {detail}")
            return None
        return cpp_davidson

    def _block_table(self, block):
        if self._cpp is None:
            return None
        hermitian = np.ascontiguousarray(_as_hermitian(block), dtype=np.complex128)
        starts = np.asarray([0], dtype=np.int64)
        return self._cpp.BlockTable([hermitian], starts, starts, hermitian.shape[0])

    def sector_matvec(self, block, vector) -> np.ndarray:
        table = self._block_table(block)
        if table is None:
            return super().sector_matvec(block, vector)
        return np.asarray(table.matvec(np.asarray(vector, dtype=np.complex128)))

    def _davidson_lowest(self, block) -> SectorDiagonalization | None:
        table = self._block_table(block)
        if table is None:
            return None
        dim = int(np.asarray(block).shape[0])
        if dim <= 1:
            return None
        diag = np.asarray(table.diagonal(), dtype=np.complex128)
        v0 = np.zeros(dim, dtype=np.complex128)
        v0[int(np.argmin(np.real(diag)))] = 1.0
        result = table.davidson(
            diag,
            v0,
            self.davidson_tol,
            self.davidson_max_iter,
            self.davidson_restart_dim,
            self.accept_unconverged,
        )
        if not bool(result.get("accepted", False)):
            return None
        value = np.asarray([float(result["energy"])], dtype=float)
        vector = np.asarray(result["vector"], dtype=np.complex128).reshape(dim, 1)
        info = {
            "residual_norm": float(result.get("residual_norm", np.nan)),
            "iterations": int(result.get("iterations", 0)),
            "converged": bool(result.get("converged", False)),
        }
        return SectorDiagonalization(value, vector, method="cpp-davidson", info=info)

    def diagonalize_sector(self, block, nroots: int | None = None) -> SectorDiagonalization:
        dim = int(np.asarray(block).shape[0])
        requested = None if nroots is None else int(nroots)
        if requested == 1 and dim >= self.davidson_min_dim:
            result = self._davidson_lowest(block)
            if result is not None:
                return result
        return super().diagonalize_sector(block, nroots=nroots)


_PYTHON_BACKEND = SU2NARGBackend()
_COMPILED_BACKEND: CompiledSU2NARGBackend | None = None


def resolve_su2_narg_backend(backend=None) -> SU2NARGBackend:
    """Resolve a backend spec into an object with SU2-NARG numeric hooks."""
    global _COMPILED_BACKEND

    if isinstance(backend, SU2NARGBackend):
        return backend
    if backend is None:
        backend = os.environ.get("SU2_NARG_BACKEND", "auto")
    key = str(backend).strip().lower().replace("-", "_")
    if key in {"python", "dense", "numpy", "blas"}:
        return _PYTHON_BACKEND
    if key in {"compiled", "cpp", "native"}:
        if _COMPILED_BACKEND is None or not _COMPILED_BACKEND.capabilities.davidson:
            _COMPILED_BACKEND = CompiledSU2NARGBackend(require=True)
        return _COMPILED_BACKEND
    if key == "auto":
        if _COMPILED_BACKEND is None:
            _COMPILED_BACKEND = CompiledSU2NARGBackend(require=False)
        return _COMPILED_BACKEND if _COMPILED_BACKEND.capabilities.davidson else _PYTHON_BACKEND
    raise ValueError("SU2-NARG backend must be 'auto', 'python', or 'compiled'.")


__all__ = [
    "CompiledSU2NARGBackend",
    "SU2NARGBackend",
    "SU2NARGCapabilities",
    "SectorDiagonalization",
    "resolve_su2_narg_backend",
]
