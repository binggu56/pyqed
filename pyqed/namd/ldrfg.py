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
HamiltonianAction = Callable[[np.ndarray, np.ndarray, np.ndarray], ArrayLike]
HamiltonianTrace = Callable[[np.ndarray, np.ndarray], complex]


def _asarray_or_call(value: QDependent, q: np.ndarray, *, dtype=None) -> np.ndarray:
    if callable(value):
        value = value(q)
    return np.asarray(value, dtype=dtype)


def _zeros_like_q_shape(q: np.ndarray, shape: tuple[int, ...], dtype=complex) -> np.ndarray:
    return np.zeros((q.size, *shape), dtype=dtype)


def grad_overlap_from_derivative_couplings(overlap: ArrayLike, derivative_couplings: ArrayLike) -> np.ndarray:
    r"""Return LDR overlap gradients from local derivative couplings.

    The convention is

    ``derivative_couplings[j, n, beta, alpha] =
    <phi_beta(R_n)|d phi_alpha(R_n)/dq_j>``.

    For each LDR block \(A_{mn}\),

    \[
        \partial_j A_{mn} = -D_j(R_m) A_{mn} + A_{mn} D_j(R_n).
    \]
    """

    overlap = np.asarray(overlap, dtype=complex)
    derivative_couplings = np.asarray(derivative_couplings, dtype=complex)
    if overlap.ndim != 4 or overlap.shape[0] != overlap.shape[2] or overlap.shape[1] != overlap.shape[3]:
        raise ValueError(
            "overlap must have shape (ngrid, nstates, ngrid, nstates); "
            f"got {overlap.shape}."
        )
    expected = (derivative_couplings.shape[0], overlap.shape[0], overlap.shape[1], overlap.shape[1])
    if derivative_couplings.shape != expected:
        raise ValueError(
            "derivative_couplings must have shape (ncoord, ngrid, nstates, nstates); "
            f"got {derivative_couplings.shape}, expected {expected}."
        )

    ncoord, ngrid, _, _ = derivative_couplings.shape
    grad = np.empty((ncoord, *overlap.shape), dtype=complex)
    for j in range(ncoord):
        for m in range(ngrid):
            left = derivative_couplings[j, m]
            for n in range(ngrid):
                right = derivative_couplings[j, n]
                block = overlap[m, :, n, :]
                grad[j, m, :, n, :] = -left @ block + block @ right
    return grad


@dataclass
class LDRFGRHS:
    """Right-hand side of the LDRFG TDVP equations."""

    c_dot: np.ndarray
    q_dot: np.ndarray
    p_dot: np.ndarray


class AbInitioLDRFGAdapter:
    """Adapter from ab initio NAC scanners to :class:`LDRFG` callables.

    ``scanner(coords)`` must return ``(energies, gradients, nac)`` with
    ``gradients[state, cart]`` and ``nac[bra, ket, cart]``.  ``geometry`` maps
    one LDR grid point and the current frozen-Gaussian coordinates to the
    Cartesian coordinates consumed by the scanner.  ``fg_vectors`` projects
    Cartesian gradients and NACs onto the frozen-Gaussian coordinates.
    """

    def __init__(
        self,
        ldr_grid: ArrayLike,
        scanner,
        geometry: Callable[[np.ndarray, np.ndarray], ArrayLike],
        fg_vectors: ArrayLike | None,
        overlap: QDependent | None = None,
        *,
        masses_y: ArrayLike | None = None,
        kinetic_x: ArrayLike | None = None,
        gamma: ArrayLike | None = None,
        hbar: float = 1.0,
        cache: bool = True,
    ) -> None:
        self.ldr_grid = np.asarray(ldr_grid, dtype=float)
        if self.ldr_grid.ndim == 1:
            self.ldr_grid = self.ldr_grid[:, None]
        if self.ldr_grid.ndim != 2:
            raise ValueError("ldr_grid must have shape (ngrid, nldr).")

        if hasattr(scanner, "as_scanner"):
            scanner = scanner.as_scanner()
        if not callable(scanner):
            raise TypeError("scanner must be callable or provide as_scanner().")
        self.scanner = scanner
        self.geometry = geometry
        self.overlap = overlap
        self.kinetic_x = None if kinetic_x is None else np.asarray(kinetic_x, dtype=complex)
        self.masses_y = None if masses_y is None else np.asarray(masses_y, dtype=float)
        self.gamma = gamma
        self.hbar = float(hbar)
        self.cache = bool(cache)
        self._cache_key = None
        self._cache_data = None

        if fg_vectors is None:
            self.fg_vectors = None
            if self.masses_y is None:
                raise ValueError("masses_y is required when fg_vectors is omitted.")
            self.ny = int(self.masses_y.size)
        else:
            self.fg_vectors = np.asarray(fg_vectors, dtype=float)
            if self.fg_vectors.ndim > 2:
                self.fg_vectors = self.fg_vectors.reshape(self.fg_vectors.shape[0], -1)
            if self.fg_vectors.ndim != 2:
                raise ValueError("fg_vectors must have shape (ny, ncart) or (ny, natom, 3).")
            self.ny = int(self.fg_vectors.shape[0])
            if self.masses_y is not None and self.masses_y.shape != (self.ny,):
                raise ValueError(f"masses_y shape {self.masses_y.shape} != {(self.ny,)}.")

    @property
    def ngrid(self) -> int:
        return int(self.ldr_grid.shape[0])

    def _key(self, q: np.ndarray) -> tuple[float, ...]:
        return tuple(np.asarray(q, dtype=float).reshape(-1))

    def _project_gradients(self, gradients: np.ndarray) -> np.ndarray:
        gradients = np.asarray(gradients, dtype=float)
        if self.fg_vectors is None:
            if gradients.shape[-1] != self.ny:
                raise ValueError(
                    "scanner gradients must already be projected onto q when fg_vectors is omitted."
                )
            return gradients
        flat = gradients.reshape(gradients.shape[0], -1)
        if flat.shape[1] != self.fg_vectors.shape[1]:
            raise ValueError(
                f"scanner gradient coordinate size {flat.shape[1]} != fg vector size {self.fg_vectors.shape[1]}."
            )
        return np.einsum("ac,jc->ja", flat, self.fg_vectors, optimize=True)

    def _project_nac(self, nac: np.ndarray) -> np.ndarray:
        nac = np.asarray(nac, dtype=float)
        if self.fg_vectors is None:
            if nac.shape[-1] != self.ny:
                raise ValueError("scanner NACs must already be projected onto q when fg_vectors is omitted.")
            return np.moveaxis(nac, -1, 0)
        flat = nac.reshape(nac.shape[0], nac.shape[1], -1)
        if flat.shape[2] != self.fg_vectors.shape[1]:
            raise ValueError(
                f"scanner NAC coordinate size {flat.shape[2]} != fg vector size {self.fg_vectors.shape[1]}."
            )
        return np.einsum("bac,jc->jba", flat, self.fg_vectors, optimize=True)

    def local_data(self, q: ArrayLike) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.ny,):
            raise ValueError(f"q shape {q.shape} != {(self.ny,)}.")
        key = self._key(q)
        if self.cache and self._cache_key == key and self._cache_data is not None:
            return self._cache_data

        energies = []
        grad_energies = []
        derivative_couplings = []
        for grid_point in self.ldr_grid:
            coords = np.asarray(self.geometry(grid_point, q), dtype=float)
            e, grad, nac = self.scanner(coords)
            energies.append(np.asarray(e, dtype=float))
            grad_energies.append(self._project_gradients(np.asarray(grad, dtype=float)))
            derivative_couplings.append(self._project_nac(np.asarray(nac, dtype=float)))

        data = {
            "energies": np.asarray(energies, dtype=float),
            "grad_energies": np.moveaxis(np.asarray(grad_energies, dtype=float), 0, 1),
            "derivative_couplings": np.moveaxis(np.asarray(derivative_couplings, dtype=float), 0, 1),
        }
        if self.cache:
            self._cache_key = key
            self._cache_data = data
        return data

    def energies(self, q: ArrayLike) -> np.ndarray:
        return self.local_data(q)["energies"]

    def grad_energies(self, q: ArrayLike) -> np.ndarray:
        return self.local_data(q)["grad_energies"]

    def derivative_couplings(self, q: ArrayLike) -> np.ndarray:
        return self.local_data(q)["derivative_couplings"]

    def overlap_at(self, q: ArrayLike) -> np.ndarray:
        if self.overlap is not None:
            return _asarray_or_call(self.overlap, np.asarray(q, dtype=float), dtype=complex)
        nstates = self.energies(q).shape[1]
        overlap = np.zeros((self.ngrid, nstates, self.ngrid, nstates), dtype=complex)
        eye = np.eye(nstates, dtype=complex)
        for m in range(self.ngrid):
            for n in range(self.ngrid):
                overlap[m, :, n, :] = eye
        return overlap

    def grad_overlap(self, q: ArrayLike) -> np.ndarray:
        return grad_overlap_from_derivative_couplings(
            self.overlap_at(q),
            self.derivative_couplings(q),
        )

    def solver(self, *, kinetic_x: ArrayLike | None = None, masses_y: ArrayLike | None = None) -> "LDRFG":
        kinetic = self.kinetic_x if kinetic_x is None else np.asarray(kinetic_x, dtype=complex)
        masses = self.masses_y if masses_y is None else np.asarray(masses_y, dtype=float)
        if kinetic is None:
            raise ValueError("kinetic_x must be supplied to build an LDRFG solver.")
        if masses is None:
            raise ValueError("masses_y must be supplied to build an LDRFG solver.")
        return LDRFG(
            kinetic,
            masses,
            energies=self.energies,
            overlap=self.overlap_at,
            grad_energies=self.grad_energies,
            grad_overlap=self.grad_overlap,
            gamma=self.gamma,
            hbar=self.hbar,
        )


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
    electronic_hamiltonian
        Optional full local electronic Hamiltonian
        \(V_{\beta\alpha}(x_n,q)\), shape ``(ngrid, nstates, nstates)``.
        If omitted, ``energies`` are used as diagonal local Hamiltonians.
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
        electronic_hamiltonian: QDependent | None = None,
        grad_electronic_hamiltonian: QDependent | None = None,
        grad_overlap: QDependent | None = None,
        berry: QDependent | None = None,
        hamiltonian_action: HamiltonianAction | None = None,
        hamiltonian_trace: HamiltonianTrace | None = None,
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
        self.electronic_hamiltonian = electronic_hamiltonian
        self.grad_electronic_hamiltonian = grad_electronic_hamiltonian
        self.overlap = overlap
        self.grad_energies = grad_energies
        self.grad_overlap = grad_overlap
        self.berry = berry
        self.hamiltonian_action = hamiltonian_action
        self.hamiltonian_trace = hamiltonian_trace
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
        electronic_h = self.electronic_hamiltonian_at(q)
        return electronic_h.shape[:2]

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
        _, nstates = self.electronic_shape(q)
        expected = (self.ngrid, nstates, self.ngrid, nstates)
        if overlap.shape != expected:
            raise ValueError(f"overlap(q) shape {overlap.shape} != expected {expected}.")
        return overlap

    def electronic_hamiltonian_at(self, q: ArrayLike) -> np.ndarray:
        q = self._validate_q(q)
        if self.electronic_hamiltonian is None:
            energies = self.energies_at(q)
            local = np.zeros((*energies.shape, energies.shape[1]), dtype=complex)
            for n in range(self.ngrid):
                local[n] = np.diag(energies[n])
            return local

        local = _asarray_or_call(self.electronic_hamiltonian, q, dtype=complex)
        if local.ndim != 3 or local.shape[0] != self.ngrid or local.shape[1] != local.shape[2]:
            raise ValueError(
                "electronic_hamiltonian(q) must have shape "
                f"(ngrid, nstates, nstates); got {local.shape}."
            )
        return local

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

    def grad_electronic_hamiltonian_at(self, q: ArrayLike) -> np.ndarray:
        q = self._validate_q(q)
        local = self.electronic_hamiltonian_at(q)
        expected = (self.ny, *local.shape)
        if self.grad_electronic_hamiltonian is None:
            grad_e = self.grad_energies_at(q)
            grad_local = np.zeros(expected, dtype=complex)
            for j in range(self.ny):
                for n in range(self.ngrid):
                    grad_local[j, n] = np.diag(grad_e[j, n])
            return grad_local

        grad = _asarray_or_call(self.grad_electronic_hamiltonian, q, dtype=complex)
        if grad.shape != expected:
            raise ValueError(
                f"grad_electronic_hamiltonian(q) shape {grad.shape} != expected {expected}."
            )
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
        electronic_h = self.electronic_hamiltonian_at(q)
        overlap = self.overlap_at(q)
        nstates = electronic_h.shape[1]

        h_tensor = np.einsum("mn,mbna->mbna", self.kinetic_x, overlap)
        kinetic_y = 0.5 * np.sum(self.inv_masses_y * p**2)
        for n in range(self.ngrid):
            h_tensor[n, :, n, :] += electronic_h[n]
            for alpha in range(nstates):
                h_tensor[n, alpha, n, alpha] += kinetic_y + self.width_energy
        return h_tensor

    def hamiltonian(self, q: ArrayLike, p: ArrayLike) -> np.ndarray:
        """Return the flattened Hamiltonian matrix acting on ``C.ravel()``."""
        h_tensor = self.hamiltonian_tensor(q, p)
        dim = h_tensor.shape[0] * h_tensor.shape[1]
        return h_tensor.reshape(dim, dim)

    def grad_hamiltonian_tensor(self, q: ArrayLike, p: ArrayLike | None = None) -> np.ndarray:
        r"""Return \(\partial H_{m\beta,n\alpha}/\partial q_j\)."""
        q = self._validate_q(q)
        electronic_h = self.electronic_hamiltonian_at(q)
        grad_electronic_h = self.grad_electronic_hamiltonian_at(q)
        nstates = electronic_h.shape[1]

        if self.grad_overlap is None:
            grad_h = np.zeros((self.ny, self.ngrid, nstates, self.ngrid, nstates), dtype=complex)
        else:
            grad_a = self.grad_overlap_at(q)
            grad_h = np.einsum("mn,jmbna->jmbna", self.kinetic_x, grad_a)
        for j in range(self.ny):
            for n in range(self.ngrid):
                grad_h[j, n, :, n, :] += grad_electronic_h[j, n]
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

    def apply_hamiltonian(self, c: ArrayLike, q: ArrayLike, p: ArrayLike) -> np.ndarray:
        """Return ``H(q,p) @ C`` as a flattened vector."""
        c0 = np.asarray(c, dtype=complex)
        c_flat = self._validate_c(c0)
        q = self._validate_q(q)
        p = self._validate_p(p)
        if self.hamiltonian_action is None:
            return self.hamiltonian(q, p) @ c_flat

        value = self.hamiltonian_action(q, p, c0)
        return np.asarray(value, dtype=complex).reshape(-1)

    def trace_hamiltonian(self, q: ArrayLike, p: ArrayLike) -> complex | None:
        """Return ``trace(H(q,p))`` when a cheap trace callback is available."""
        q = self._validate_q(q)
        p = self._validate_p(p)
        if self.hamiltonian_trace is None:
            return None
        return complex(self.hamiltonian_trace(q, p))

    def energy(self, c: ArrayLike, q: ArrayLike, p: ArrayLike, *, normalize: bool = True) -> complex:
        """Return ``<C|H(q,p)|C>``."""
        c_flat = self._validate_c(c)
        denom = np.vdot(c_flat, c_flat) if normalize else 1.0
        if normalize and np.isclose(denom, 0.0):
            raise ValueError("Cannot normalize an energy expectation value with zero-norm C.")
        return np.vdot(c_flat, self.apply_hamiltonian(c, q, p)) / denom

    def force(
        self,
        c: ArrayLike,
        q: ArrayLike,
        p: ArrayLike | None = None,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        """Return the TDVP force ``-<C|dH/dq|C>`` on the FG coordinates."""
        q = self._validate_q(q)
        if p is None:
            p = np.zeros(self.ny, dtype=float)
        else:
            p = self._validate_p(p)
        c_flat = self._validate_c(c)
        grad_h = self.grad_hamiltonian(q, p)
        norm = np.vdot(c_flat, c_flat) if normalize else 1.0
        if normalize and np.isclose(norm, 0.0):
            raise ValueError("Cannot evaluate LDRFG force with zero-norm C.")

        force = np.empty(self.ny, dtype=float)
        if self.berry is not None:
            h = self.hamiltonian(q, p)
            berry = self.berry_at(q).reshape(self.ny, c_flat.size, c_flat.size)
        else:
            h = None
            berry = None

        for j in range(self.ny):
            value = -np.vdot(c_flat, grad_h[j] @ c_flat) / norm
            if berry is not None and h is not None:
                d_j = berry[j]
                commutator = h @ d_j - d_j @ h
                value += np.vdot(c_flat, commutator @ c_flat) / norm
            force[j] = float(np.real_if_close(value))
        return force

    def rhs(self, c: ArrayLike, q: ArrayLike, p: ArrayLike, *, normalize_force: bool = True) -> LDRFGRHS:
        """Evaluate the TDVP equations of motion."""
        q = self._validate_q(q)
        p = self._validate_p(p)
        c_flat = self._validate_c(c)

        h_c = self.apply_hamiltonian(c_flat, q, p)
        q_dot = self.inv_masses_y * p

        berry = self.berry_at(q).reshape(self.ny, c_flat.size, c_flat.size)
        c_dot = -1j / self.hbar * h_c
        c_dot -= np.einsum("j,jab,b->a", q_dot, berry, c_flat)

        return LDRFGRHS(
            c_dot=c_dot.reshape(np.asarray(c).shape),
            q_dot=q_dot,
            p_dot=self.force(c_flat, q, p, normalize=normalize_force),
        )

    def propagate_coefficients(self, c: ArrayLike, q: ArrayLike, p: ArrayLike, dt: float) -> np.ndarray:
        """Propagate ``C`` at fixed ``q, p`` with a matrix exponential."""
        from scipy.sparse.linalg import expm_multiply
        from scipy.sparse.linalg import LinearOperator

        c0 = np.asarray(c, dtype=complex)
        c_flat = self._validate_c(c0)
        q = self._validate_q(q)
        p = self._validate_p(p)

        if self.berry is not None:
            q_dot = self.inv_masses_y * p
            berry = self.berry_at(q).reshape(self.ny, c_flat.size, c_flat.size)
        else:
            q_dot = None
            berry = None

        if self.hamiltonian_action is None:
            h = self.hamiltonian(q, p)
            generator = (-1j / self.hbar) * h
            if berry is not None and q_dot is not None:
                generator -= np.einsum("j,jab->ab", q_dot, berry, optimize=True)
            c_new = expm_multiply(dt * generator, c_flat)
            return np.asarray(c_new, dtype=complex).reshape(c0.shape)

        dim = c_flat.size

        def matvec(v):
            value = (-1j / self.hbar) * self.apply_hamiltonian(v.reshape(c0.shape), q, p)
            if berry is not None and q_dot is not None:
                value -= np.einsum("j,jab,b->a", q_dot, berry, v, optimize=True)
            return value

        def rmatvec(v):
            value = (1j / self.hbar) * self.apply_hamiltonian(v.reshape(c0.shape), q, p)
            if berry is not None and q_dot is not None:
                value += np.einsum("j,jab,b->a", q_dot, berry, v, optimize=True)
            return value

        def matmat(vectors):
            return np.column_stack([matvec(vectors[:, col]) for col in range(vectors.shape[1])])

        generator = LinearOperator((dim, dim), matvec=matvec, rmatvec=rmatvec, matmat=matmat, dtype=complex)
        trace_h = self.trace_hamiltonian(q, p)
        trace_generator = None if trace_h is None else (-1j / self.hbar) * trace_h

        c_new = expm_multiply(
            dt * generator,
            c_flat,
            traceA=None if trace_generator is None else dt * trace_generator,
        )
        return np.asarray(c_new, dtype=complex).reshape(c0.shape)

    def step_split(self, c: ArrayLike, q: ArrayLike, p: ArrayLike, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance one Strang split unitary-electronic/Verlet-FG step."""
        c0 = np.asarray(c, dtype=complex)
        q0 = self._validate_q(q)
        p0 = self._validate_p(p)

        c_half = self.propagate_coefficients(c0, q0, p0, 0.5 * dt)
        p_half = p0 + 0.5 * dt * self.force(c_half, q0, p0)
        q_new = q0 + dt * self.inv_masses_y * p_half
        p_new = p_half + 0.5 * dt * self.force(c_half, q_new, p_half)
        c_new = self.propagate_coefficients(c_half, q_new, p_new, 0.5 * dt)
        return c_new, q_new, p_new

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
