"""Retained-state RTLDR for one nuclear coordinate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import expm


ArrayLike = np.ndarray
ElectronicHamiltonian = ArrayLike | Callable[..., ArrayLike]


def _as_hermitian_matrix(matrix, *, name):
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError(f"{name} must be Hermitian.")
    return matrix


def _call_electronic_hamiltonian(source, index, coordinate, time):
    if callable(source):
        errors = []
        for args in ((coordinate, time), (index, coordinate, time), (time,)):
            try:
                value = source(*args)
                break
            except TypeError as exc:
                errors.append(exc)
        else:
            raise TypeError(
                "electronic_hamiltonian callable must accept (R, t), "
                "(index, R, t), or (t)."
            ) from errors[-1]
    else:
        value = source

    value = np.asarray(value, dtype=complex)
    if value.ndim == 3:
        value = value[index]
    return _as_hermitian_matrix(value, name=f"electronic_hamiltonian[{index}]")


def frames_from_overlap(overlap: ArrayLike, *, atol: float = 1.0e-12) -> np.ndarray:
    """Embed local states from ``O_ia,jb = <phi_a(R_i)|phi_b(R_j)>``."""

    overlap = np.asarray(overlap, dtype=complex)
    if overlap.ndim != 4 or overlap.shape[0] != overlap.shape[2] or overlap.shape[1] != overlap.shape[3]:
        raise ValueError(
            "overlap must have shape (ngrid, nstates, ngrid, nstates); "
            f"got {overlap.shape}."
        )
    ngrid, nstates = overlap.shape[:2]
    gram = overlap.reshape(ngrid * nstates, ngrid * nstates)
    if not np.allclose(gram, gram.conj().T, atol=10.0 * atol):
        raise ValueError("overlap tensor must define a Hermitian Gram matrix.")

    evals, evecs = np.linalg.eigh(0.5 * (gram + gram.conj().T))
    if np.min(evals) < -atol:
        raise ValueError("overlap tensor is not positive semidefinite.")
    keep = evals > atol
    if not np.any(keep):
        raise ValueError("overlap tensor has zero numerical rank.")
    factors = np.conj(evecs[:, keep] * np.sqrt(evals[keep])[None, :])
    return factors.reshape(ngrid, nstates, -1).transpose(0, 2, 1)


@dataclass(frozen=True)
class RetainedStateTrajectory:
    """Stored retained-state RTLDR trajectory."""

    times: np.ndarray
    coefficients: np.ndarray
    frames: np.ndarray
    overlaps: np.ndarray
    kinetic_hamiltonians: np.ndarray | None = None

    @property
    def norm(self):
        return np.sum(np.abs(self.coefficients) ** 2, axis=tuple(range(1, self.coefficients.ndim)))

    @property
    def coordinate_density(self):
        return np.sum(np.abs(self.coefficients) ** 2, axis=2)

    @property
    def state_populations(self):
        return np.sum(np.abs(self.coefficients) ** 2, axis=1)


class RetainedStateRTLDR:
    """One-coordinate time-dependent local diabatization representation."""

    def __init__(
        self,
        grid: ArrayLike,
        kinetic: ArrayLike,
        electronic_hamiltonian: ElectronicHamiltonian,
        initial_frames: ArrayLike | None = None,
        *,
        hbar: float = 1.0,
    ) -> None:
        self.grid = np.asarray(grid, dtype=float).reshape(-1)
        if self.grid.size < 1:
            raise ValueError("grid must contain at least one point.")
        self.ngrid = int(self.grid.size)

        self.kinetic = _as_hermitian_matrix(kinetic, name="kinetic")
        if self.kinetic.shape != (self.ngrid, self.ngrid):
            raise ValueError(f"kinetic shape {self.kinetic.shape} != {(self.ngrid, self.ngrid)}.")
        self.electronic_hamiltonian_source = electronic_hamiltonian
        self.hbar = float(hbar)
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")

        if initial_frames is None:
            h0 = self.local_hamiltonian(0, 0.0)
            for index in range(1, self.ngrid):
                h = self.local_hamiltonian(index, 0.0)
                if h.shape != h0.shape:
                    raise ValueError("All local electronic Hamiltonians must have the same shape.")
            eye = np.eye(h0.shape[0], dtype=complex)
            initial_frames = np.broadcast_to(eye, (self.ngrid, *eye.shape)).copy()

        self.initial_frames = self._validate_frames(initial_frames)
        _, self.nelec, self.nstates = self.initial_frames.shape
        for index in range(self.ngrid):
            h = self.local_hamiltonian(index, 0.0)
            if h.shape not in ((self.nelec, self.nelec), (self.nstates, self.nstates)):
                raise ValueError(
                    f"electronic_hamiltonian[{index}] shape {h.shape} "
                    f"must be {(self.nelec, self.nelec)} or {(self.nstates, self.nstates)}."
                )

    def _validate_frames(self, frames):
        frames = np.asarray(frames, dtype=complex)
        if frames.ndim != 3:
            raise ValueError("frames must have shape (ngrid, nelec, nstates).")
        if frames.shape[0] != self.ngrid:
            raise ValueError(f"frames first dimension {frames.shape[0]} != ngrid {self.ngrid}.")
        if frames.shape[2] > frames.shape[1]:
            raise ValueError("nstates cannot exceed nelec.")
        for index in range(self.ngrid):
            metric = frames[index].conj().T @ frames[index]
            if not np.allclose(metric, np.eye(frames.shape[2], dtype=complex), atol=1.0e-12):
                raise ValueError(f"frames[{index}] columns must be orthonormal.")
        return frames

    @property
    def coefficient_shape(self):
        return (self.ngrid, self.nstates)

    @property
    def dimension(self):
        return self.ngrid * self.nstates

    def metric(self):
        return np.eye(self.dimension, dtype=complex)

    def local_hamiltonian(self, index, time):
        index = int(index)
        if index < 0 or index >= self.ngrid:
            raise ValueError(f"index must be in [0, {self.ngrid}).")
        return _call_electronic_hamiltonian(
            self.electronic_hamiltonian_source,
            index,
            float(self.grid[index]),
            float(time),
        )

    def propagate_frames(self, frames, time, dt):
        """Propagate all local electronic frames by midpoint electronic TDSE."""

        frames = self._validate_frames(frames)
        out = np.empty_like(frames)
        midpoint = float(time) + 0.5 * float(dt)
        for index in range(self.ngrid):
            h = self.local_hamiltonian(index, midpoint)
            if h.shape == (frames.shape[1], frames.shape[1]):
                u = expm(-1j * h * float(dt) / self.hbar)
                out[index] = u @ frames[index]
            elif h.shape == (frames.shape[2], frames.shape[2]):
                u = expm(-1j * h * float(dt) / self.hbar)
                out[index] = frames[index] @ u
            else:
                raise ValueError(
                    f"electronic_hamiltonian[{index}] shape {h.shape} "
                    f"must be {(frames.shape[1], frames.shape[1])} or {(frames.shape[2], frames.shape[2])}."
                )
        return out

    def overlap_tensor(self, frames=None):
        if frames is None:
            frames = self.initial_frames
        frames = self._validate_frames(frames)
        return np.einsum("ima,jmb->iajb", frames.conj(), frames, optimize=True)

    def kinetic_hamiltonian(self, frames=None):
        overlap = self.overlap_tensor(frames)
        return self.kinetic[:, None, :, None] * overlap

    def kinetic_matrix(self, frames=None):
        return self.kinetic_hamiltonian(frames).reshape(self.dimension, self.dimension)

    def rhs(self, coefficients, frames=None):
        coefficients = np.asarray(coefficients, dtype=complex)
        if coefficients.shape != self.coefficient_shape:
            raise ValueError(f"coefficients shape {coefficients.shape} != {self.coefficient_shape}.")
        cdot = -1j * (self.kinetic_matrix(frames) @ coefficients.reshape(-1)) / self.hbar
        return cdot.reshape(self.coefficient_shape)

    def step(self, coefficients, frames, time, dt):
        coefficients = np.asarray(coefficients, dtype=complex)
        if coefficients.shape != self.coefficient_shape:
            raise ValueError(f"coefficients shape {coefficients.shape} != {self.coefficient_shape}.")
        frames = self._validate_frames(frames)
        frames_mid = self.propagate_frames(frames, time, 0.5 * dt)
        u_coeff = expm(-1j * self.kinetic_matrix(frames_mid) * float(dt) / self.hbar)
        next_coefficients = (u_coeff @ coefficients.reshape(-1)).reshape(self.coefficient_shape)
        next_frames = self.propagate_frames(frames, time, dt)
        return next_coefficients, next_frames

    def run(self, coefficients0, *, dt, nsteps, t0=0.0, store_hamiltonians=False):
        coefficients = np.asarray(coefficients0, dtype=complex)
        if coefficients.shape != self.coefficient_shape:
            raise ValueError(f"coefficients0 shape {coefficients.shape} != {self.coefficient_shape}.")
        nsteps = int(nsteps)
        if nsteps < 0:
            raise ValueError("nsteps must be non-negative.")
        dt = float(dt)

        times = float(t0) + dt * np.arange(nsteps + 1, dtype=float)
        coeffs = np.empty((nsteps + 1, *self.coefficient_shape), dtype=complex)
        frames_hist = np.empty((nsteps + 1, *self.initial_frames.shape), dtype=complex)
        overlaps = np.empty(
            (nsteps + 1, self.ngrid, self.nstates, self.ngrid, self.nstates),
            dtype=complex,
        )
        kinetic_hist = (
            np.empty((nsteps + 1, self.dimension, self.dimension), dtype=complex)
            if store_hamiltonians
            else None
        )

        frames = self.initial_frames.copy()
        for step in range(nsteps + 1):
            coeffs[step] = coefficients
            frames_hist[step] = frames
            overlaps[step] = self.overlap_tensor(frames)
            if kinetic_hist is not None:
                kinetic_hist[step] = self.kinetic_matrix(frames)
            if step < nsteps:
                coefficients, frames = self.step(coefficients, frames, times[step], dt)

        return RetainedStateTrajectory(
            times=times,
            coefficients=coeffs,
            frames=frames_hist,
            overlaps=overlaps,
            kinetic_hamiltonians=kinetic_hist,
        )
