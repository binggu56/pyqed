"""Hybrid LDR--frozen Gaussian equations of motion.

The LDRFG ansatz treats selected nuclear coordinates with an LDR/DVR basis and
the remaining coordinates with one moving frozen Gaussian.  This module
implements the dense TDVP equations

    i hbar Cdot = (H - i hbar qdot . d) C
    qdot = M^{-1} p
    pdot = - <C|dH/dq|C>

for the coefficient vector ``C`` and Gaussian phase-space variables ``q, p``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayLike = np.ndarray
QDependent = ArrayLike | Callable[[np.ndarray], ArrayLike]


def _asarray_or_call(value: QDependent, q: np.ndarray, *, dtype=None) -> np.ndarray:
    if callable(value):
        value = value(q)
    return np.asarray(value, dtype=dtype)


def _zeros_like_q_shape(q: np.ndarray, shape: tuple[int, ...], dtype=complex) -> np.ndarray:
    return np.zeros((q.size, *shape), dtype=dtype)


@dataclass
class LDRFGRHS:
    """Right-hand side of the LDRFG TDVP equations."""

    c_dot: np.ndarray
    q_dot: np.ndarray
    p_dot: np.ndarray


class LDRFG:
    r"""Dense TDVP engine for the hybrid LDR--frozen Gaussian ansatz.

    Parameters
    ----------
    kinetic_x
        DVR kinetic matrix \(T^x_{mn}\) for the LDR coordinates, shape
        ``(ngrid, ngrid)``.
    masses_y
        Masses for the frozen-Gaussian coordinates.
    energies
        Adiabatic energies \(E_\alpha(x_n,q)\), shape ``(ngrid, nstates)``,
        or a callable ``energies(q)`` returning that shape.
    overlap
        LDR electronic overlap
        \(A_{m\beta,n\alpha}(q)=\langle\phi_\beta(x_m,q)|\phi_\alpha(x_n,q)\rangle\),
        shape ``(ngrid, nstates, ngrid, nstates)``, or a callable.
    grad_energies
        Gradient of ``energies`` with respect to ``q``, shape
        ``(ny, ngrid, nstates)``.  If omitted, the energy gradient is zero.
    grad_overlap
        Gradient of ``overlap`` with respect to ``q``, shape
        ``(ny, ngrid, nstates, ngrid, nstates)``.  If omitted, the overlap
        gradient is zero.
    berry
        Optional Berry-connection matrices \(d^{q_j}_{m\beta,n\alpha}\), shape
        ``(ny, ngrid, nstates, ngrid, nstates)``.  If omitted, the
        parallel-transport/zero-connection gauge is used.
    gamma
        Fixed Gaussian width matrix.  If provided, the constant frozen-width
        kinetic contribution is ``hbar**2 / 4 * trace(inv(M) @ gamma)``.
    hbar
        Planck constant in the units used by the model.  Defaults to atomic
        units, ``hbar=1``.
    """

    def __init__(
        self,
        kinetic_x: ArrayLike,
        masses_y: ArrayLike,
        energies: QDependent,
        overlap: QDependent,
        *,
        grad_energies: QDependent | None = None,
        grad_overlap: QDependent | None = None,
        berry: QDependent | None = None,
        gamma: ArrayLike | None = None,
        hbar: float = 1.0,
    ) -> None:
        self.kinetic_x = np.asarray(kinetic_x, dtype=complex)
        if self.kinetic_x.ndim != 2 or self.kinetic_x.shape[0] != self.kinetic_x.shape[1]:
            raise ValueError("kinetic_x must be a square matrix.")

        self.masses_y = np.asarray(masses_y, dtype=float)
        if self.masses_y.ndim != 1:
            raise ValueError("masses_y must be a one-dimensional array.")
        if np.any(self.masses_y <= 0.0):
            raise ValueError("masses_y must contain positive masses.")

        self.inv_masses_y = 1.0 / self.masses_y
        self.energies = energies
        self.overlap = overlap
        self.grad_energies = grad_energies
        self.grad_overlap = grad_overlap
        self.berry = berry
        self.hbar = float(hbar)

        if gamma is None:
            self.gamma = None
            self.width_energy = 0.0
        else:
            self.gamma = np.asarray(gamma, dtype=float)
            if self.gamma.shape != (self.ny, self.ny):
                raise ValueError(f"gamma shape {self.gamma.shape} != {(self.ny, self.ny)}.")
            self.width_energy = 0.25 * self.hbar**2 * np.sum(
                self.inv_masses_y[:, None] * self.gamma * np.eye(self.ny)
            )

    @property
    def ngrid(self) -> int:
        return self.kinetic_x.shape[0]

    @property
    def ny(self) -> int:
        return self.masses_y.size

    def electronic_shape(self, q: ArrayLike) -> tuple[int, int]:
        energies = self.energies_at(q)
        return energies.shape

    def energies_at(self, q: ArrayLike) -> np.ndarray:
        q = self._validate_q(q)
        energies = _asarray_or_call(self.energies, q, dtype=float)
        expected0 = self.ngrid
        if energies.ndim != 2 or energies.shape[0] != expected0:
            raise ValueError(
                f"energies(q) must have shape (ngrid, nstates); got {energies.shape}."
            )
        return energies

    def overlap_at(self, q: ArrayLike) -> np.ndarray:
        q = self._validate_q(q)
        overlap = _asarray_or_call(self.overlap, q, dtype=complex)
        energies = self.energies_at(q)
        expected = (self.ngrid, energies.shape[1], self.ngrid, energies.shape[1])
        if overlap.shape != expected:
            raise ValueError(f"overlap(q) shape {overlap.shape} != expected {expected}.")
        return overlap

    def grad_energies_at(self, q: ArrayLike) -> np.ndarray:
        q = self._validate_q(q)
        energies = self.energies_at(q)
        if self.grad_energies is None:
            return np.zeros((self.ny, *energies.shape), dtype=float)
        grad = _asarray_or_call(self.grad_energies, q, dtype=float)
        expected = (self.ny, *energies.shape)
        if grad.shape != expected:
            raise ValueError(f"grad_energies(q) shape {grad.shape} != expected {expected}.")
        return grad

    def grad_overlap_at(self, q: ArrayLike) -> np.ndarray:
        q = self._validate_q(q)
        overlap = self.overlap_at(q)
        if self.grad_overlap is None:
            return _zeros_like_q_shape(q, overlap.shape, dtype=complex)
        grad = _asarray_or_call(self.grad_overlap, q, dtype=complex)
        expected = (self.ny, *overlap.shape)
        if grad.shape != expected:
            raise ValueError(f"grad_overlap(q) shape {grad.shape} != expected {expected}.")
        return grad

    def berry_at(self, q: ArrayLike) -> np.ndarray:
        q = self._validate_q(q)
        overlap = self.overlap_at(q)
        if self.berry is None:
            return _zeros_like_q_shape(q, overlap.shape, dtype=complex)
        berry = _asarray_or_call(self.berry, q, dtype=complex)
        expected = (self.ny, *overlap.shape)
        if berry.shape != expected:
            raise ValueError(f"berry(q) shape {berry.shape} != expected {expected}.")
        return berry

    def hamiltonian_tensor(self, q: ArrayLike, p: ArrayLike) -> np.ndarray:
        r"""Return \(H_{m\beta,n\alpha}(q,p)\)."""
        q = self._validate_q(q)
        p = self._validate_p(p)
        energies = self.energies_at(q)
        overlap = self.overlap_at(q)
        nstates = energies.shape[1]

        h_tensor = np.einsum("mn,mbna->mbna", self.kinetic_x, overlap)
        kinetic_y = 0.5 * np.sum(self.inv_masses_y * p**2)
        diagonal = energies + kinetic_y + self.width_energy
        for n in range(self.ngrid):
            for alpha in range(nstates):
                h_tensor[n, alpha, n, alpha] += diagonal[n, alpha]
        return h_tensor

    def hamiltonian(self, q: ArrayLike, p: ArrayLike) -> np.ndarray:
        """Return the flattened Hamiltonian matrix acting on ``C.ravel()``."""
        h_tensor = self.hamiltonian_tensor(q, p)
        dim = h_tensor.shape[0] * h_tensor.shape[1]
        return h_tensor.reshape(dim, dim)

    def grad_hamiltonian_tensor(self, q: ArrayLike, p: ArrayLike | None = None) -> np.ndarray:
        r"""Return \(\partial H_{m\beta,n\alpha}/\partial q_j\)."""
        q = self._validate_q(q)
        energies = self.energies_at(q)
        grad_e = self.grad_energies_at(q)
        grad_a = self.grad_overlap_at(q)
        nstates = energies.shape[1]

        grad_h = np.einsum("mn,jmbna->jmbna", self.kinetic_x, grad_a)
        for j in range(self.ny):
            for n in range(self.ngrid):
                for alpha in range(nstates):
                    grad_h[j, n, alpha, n, alpha] += grad_e[j, n, alpha]
        return grad_h

    def grad_hamiltonian(self, q: ArrayLike, p: ArrayLike | None = None) -> np.ndarray:
        """Return flattened Hamiltonian gradients, shape ``(ny, dim, dim)``."""
        grad_h = self.grad_hamiltonian_tensor(q, p)
        dim = grad_h.shape[1] * grad_h.shape[2]
        return grad_h.reshape(self.ny, dim, dim)

    def expectation(self, operator: np.ndarray, c: ArrayLike, *, normalize: bool = True) -> complex:
        c_flat = self._validate_c(c)
        denom = np.vdot(c_flat, c_flat) if normalize else 1.0
        if normalize and np.isclose(denom, 0.0):
            raise ValueError("Cannot normalize an expectation value with zero-norm C.")
        return np.vdot(c_flat, operator @ c_flat) / denom

    def energy(self, c: ArrayLike, q: ArrayLike, p: ArrayLike, *, normalize: bool = True) -> complex:
        """Return ``<C|H(q,p)|C>``."""
        return self.expectation(self.hamiltonian(q, p), c, normalize=normalize)

    def rhs(self, c: ArrayLike, q: ArrayLike, p: ArrayLike, *, normalize_force: bool = True) -> LDRFGRHS:
        """Evaluate the TDVP equations of motion."""
        q = self._validate_q(q)
        p = self._validate_p(p)
        c_flat = self._validate_c(c)

        h = self.hamiltonian(q, p)
        q_dot = self.inv_masses_y * p

        berry = self.berry_at(q).reshape(self.ny, c_flat.size, c_flat.size)
        c_dot = -1j / self.hbar * (h @ c_flat)
        c_dot -= np.einsum("j,jab,b->a", q_dot, berry, c_flat)

        grad_h = self.grad_hamiltonian(q, p)
        norm = np.vdot(c_flat, c_flat) if normalize_force else 1.0
        if normalize_force and np.isclose(norm, 0.0):
            raise ValueError("Cannot evaluate LDRFG force with zero-norm C.")
        p_dot = np.empty(self.ny, dtype=float)
        for j in range(self.ny):
            force = -np.vdot(c_flat, grad_h[j] @ c_flat) / norm
            if self.berry is not None:
                d_j = berry[j]
                commutator = h @ d_j - d_j @ h
                force += np.vdot(c_flat, commutator @ c_flat) / norm
            p_dot[j] = float(np.real_if_close(force))

        return LDRFGRHS(
            c_dot=c_dot.reshape(np.asarray(c).shape),
            q_dot=q_dot,
            p_dot=p_dot,
        )

    def step_rk4(self, c: ArrayLike, q: ArrayLike, p: ArrayLike, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance ``(C, q, p)`` by one fourth-order Runge--Kutta step."""
        c0 = np.asarray(c, dtype=complex)
        q0 = self._validate_q(q)
        p0 = self._validate_p(p)

        def add(state, scale, deriv):
            c_s, q_s, p_s = state
            return (
                c_s + scale * deriv.c_dot,
                q_s + scale * deriv.q_dot,
                p_s + scale * deriv.p_dot,
            )

        y0 = (c0, q0, p0)
        k1 = self.rhs(*y0)
        k2 = self.rhs(*add(y0, 0.5 * dt, k1))
        k3 = self.rhs(*add(y0, 0.5 * dt, k2))
        k4 = self.rhs(*add(y0, dt, k3))

        c_new = c0 + dt / 6.0 * (k1.c_dot + 2.0 * k2.c_dot + 2.0 * k3.c_dot + k4.c_dot)
        q_new = q0 + dt / 6.0 * (k1.q_dot + 2.0 * k2.q_dot + 2.0 * k3.q_dot + k4.q_dot)
        p_new = p0 + dt / 6.0 * (k1.p_dot + 2.0 * k2.p_dot + 2.0 * k3.p_dot + k4.p_dot)
        return c_new, q_new, p_new

    def _validate_q(self, q: ArrayLike) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.ny,):
            raise ValueError(f"q shape {q.shape} != {(self.ny,)}.")
        return q

    def _validate_p(self, p: ArrayLike) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if p.shape != (self.ny,):
            raise ValueError(f"p shape {p.shape} != {(self.ny,)}.")
        return p

    def _validate_c(self, c: ArrayLike) -> np.ndarray:
        c = np.asarray(c, dtype=complex)
        if c.size == 0:
            raise ValueError("C must not be empty.")
        return c.reshape(-1)
