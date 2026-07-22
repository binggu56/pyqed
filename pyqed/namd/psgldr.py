"""Y-only pseudospectral Gaussian local diabatic representation.

This module implements a compact prototype for the PSG-LDR equations.  The
ansatz uses moving frozen Gaussians for all nuclear coordinates and a local
electronic basis attached to each Gaussian center.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayLike = np.ndarray
CentersDependent = ArrayLike | Callable[[np.ndarray], ArrayLike]
ForceCallback = Callable[[np.ndarray, np.ndarray, np.ndarray, "PSGLDR"], ArrayLike]
PhaseCallback = Callable[[np.ndarray, np.ndarray, np.ndarray, "PSGLDR"], ArrayLike]


def _asarray_or_call(value: CentersDependent, centers: np.ndarray, *, dtype=None) -> np.ndarray:
    if callable(value):
        value = value(centers)
    return np.asarray(value, dtype=dtype)


@dataclass
class PSGLDRRHS:
    """Right-hand side of the y-only PSG-LDR equations."""

    c_dot: np.ndarray
    q_dot: np.ndarray
    p_dot: np.ndarray
    gamma_dot: np.ndarray
    singular_values: np.ndarray


class PSGLDR:
    r"""Y-only pseudospectral Gaussian LDR solver.

    The wavefunction is expanded as

    \[
        |\Psi\rangle = \sum_{a\alpha} C_{a\alpha}
        g_a(y; q_a, p_a) |\phi_\alpha(q_a)\rangle .
    \]

    The TDSE is tested with

    \[
        \langle \chi_{a\beta}| =
        \delta(y-q_a)\langle \phi_\beta(q_a)| .
    \]

    This gives

    \[
        \Phi \dot C = -\frac{i}{\hbar} H C - D C,
    \]

    where \(\Phi\), \(H\), and \(D\) are assembled by evaluating frozen
    Gaussian basis functions and local electronic links at the Gaussian
    centers.
    """

    def __init__(
        self,
        masses: ArrayLike,
        widths: ArrayLike,
        electronic_hamiltonian: CentersDependent,
        *,
        overlap: CentersDependent | None = None,
        ket_derivative_couplings: CentersDependent | None = None,
        force: ForceCallback | None = None,
        phase: PhaseCallback | None = None,
        hbar: float = 1.0,
        svd_rcond: float = 1.0e-9,
    ) -> None:
        self.masses = np.asarray(masses, dtype=float)
        if self.masses.ndim != 1:
            raise ValueError("masses must be a one-dimensional array.")
        if np.any(self.masses <= 0.0):
            raise ValueError("masses must contain positive values.")

        self.widths = np.asarray(widths, dtype=float)
        if self.widths.ndim == 0:
            self.widths = np.full((1, self.ncoord), float(self.widths))
        elif self.widths.ndim == 1:
            if self.widths.shape != (self.ncoord,):
                raise ValueError(f"widths shape {self.widths.shape} != {(self.ncoord,)}.")
            self.widths = self.widths[None, :]
        elif self.widths.ndim != 2 or self.widths.shape[1] != self.ncoord:
            raise ValueError("widths must be scalar, (ncoord,), or (ngauss, ncoord).")
        if np.any(self.widths <= 0.0):
            raise ValueError("widths must contain positive values.")

        self.electronic_hamiltonian = electronic_hamiltonian
        self.overlap = overlap
        self.ket_derivative_couplings = ket_derivative_couplings
        self.force_callback = force
        self.phase_callback = phase
        self.hbar = float(hbar)
        self.svd_rcond = float(svd_rcond)

    @property
    def ncoord(self) -> int:
        return self.masses.size

    def widths_for(self, centers: ArrayLike) -> np.ndarray:
        centers = self._validate_centers(centers)
        if self.widths.shape[0] == 1:
            return np.repeat(self.widths, centers.shape[0], axis=0)
        if self.widths.shape != centers.shape:
            raise ValueError(f"widths shape {self.widths.shape} != centers shape {centers.shape}.")
        return self.widths

    def electronic_hamiltonian_at(self, centers: ArrayLike) -> np.ndarray:
        centers = self._validate_centers(centers)
        h_el = _asarray_or_call(self.electronic_hamiltonian, centers, dtype=complex)
        if h_el.ndim != 3 or h_el.shape[0] != centers.shape[0] or h_el.shape[1] != h_el.shape[2]:
            raise ValueError(
                "electronic_hamiltonian(q) must have shape "
                f"(ngauss, nstates, nstates); got {h_el.shape}."
            )
        return h_el

    def overlap_at(self, centers: ArrayLike) -> np.ndarray:
        centers = self._validate_centers(centers)
        h_el = self.electronic_hamiltonian_at(centers)
        ngauss, nstates = h_el.shape[:2]
        if self.overlap is None:
            overlap = np.empty((ngauss, nstates, ngauss, nstates), dtype=complex)
            eye = np.eye(nstates, dtype=complex)
            for a in range(ngauss):
                for b in range(ngauss):
                    overlap[a, :, b, :] = eye
            return overlap

        overlap = _asarray_or_call(self.overlap, centers, dtype=complex)
        expected = (ngauss, nstates, ngauss, nstates)
        if overlap.shape != expected:
            raise ValueError(f"overlap(q) shape {overlap.shape} != expected {expected}.")
        return overlap

    def ket_derivative_couplings_at(self, centers: ArrayLike) -> np.ndarray:
        centers = self._validate_centers(centers)
        overlap = self.overlap_at(centers)
        expected = (self.ncoord, *overlap.shape)
        if self.ket_derivative_couplings is None:
            return np.zeros(expected, dtype=complex)
        value = _asarray_or_call(self.ket_derivative_couplings, centers, dtype=complex)
        if value.shape != expected:
            raise ValueError(f"ket_derivative_couplings(q) shape {value.shape} != expected {expected}.")
        return value

    def gaussian_values(
        self,
        centers: ArrayLike,
        momenta: ArrayLike,
        gamma: ArrayLike | None = None,
    ) -> np.ndarray:
        centers = self._validate_centers(centers)
        momenta = self._validate_momenta(momenta, centers.shape[0])
        widths = self.widths_for(centers)
        gamma = self._validate_gamma(gamma, centers.shape[0])

        delta = centers[:, None, :] - centers[None, :, :]
        exponent = -np.einsum("bk,abk->ab", widths, delta * delta, optimize=True)
        exponent = exponent + 1j * np.einsum("bk,abk->ab", momenta, delta, optimize=True)
        return np.exp(gamma[None, :] + exponent)

    def gaussian_laplacian_matrix(
        self,
        centers: ArrayLike,
        momenta: ArrayLike,
        gamma: ArrayLike | None = None,
    ) -> np.ndarray:
        centers = self._validate_centers(centers)
        momenta = self._validate_momenta(momenta, centers.shape[0])
        widths = self.widths_for(centers)
        g = self.gaussian_values(centers, momenta, gamma)
        delta = centers[:, None, :] - centers[None, :, :]
        first = -2.0 * widths[None, :, :] * delta + 1j * momenta[None, :, :]
        lap_factor = -2.0 * widths[None, :, :] + first * first
        return np.einsum("k,abk,ab->ab", -0.5 * self.hbar**2 / self.masses, lap_factor, g, optimize=True)

    def collocation_overlap(
        self,
        centers: ArrayLike,
        momenta: ArrayLike,
        gamma: ArrayLike | None = None,
    ) -> np.ndarray:
        centers = self._validate_centers(centers)
        g = self.gaussian_values(centers, momenta, gamma)
        links = self.overlap_at(centers)
        return np.einsum("ab,aBbc->aBbc", g, links, optimize=True).reshape(g.shape[0] * links.shape[1], -1)

    def hamiltonian(
        self,
        centers: ArrayLike,
        momenta: ArrayLike,
        gamma: ArrayLike | None = None,
    ) -> np.ndarray:
        centers = self._validate_centers(centers)
        g = self.gaussian_values(centers, momenta, gamma)
        kinetic = self.gaussian_laplacian_matrix(centers, momenta, gamma)
        links = self.overlap_at(centers)
        h_el = self.electronic_hamiltonian_at(centers)

        h_tensor = np.einsum("ab,aBbc->aBbc", kinetic, links, optimize=True)
        local = np.einsum("aBD,aDbc->aBbc", h_el, links, optimize=True)
        h_tensor += np.einsum("ab,aBbc->aBbc", g, local, optimize=True)
        return h_tensor.reshape(g.shape[0] * h_el.shape[1], -1)

    def moving_basis_matrix(
        self,
        centers: ArrayLike,
        momenta: ArrayLike,
        q_dot: ArrayLike,
        p_dot: ArrayLike,
        gamma_dot: ArrayLike | None = None,
        gamma: ArrayLike | None = None,
    ) -> np.ndarray:
        centers = self._validate_centers(centers)
        momenta = self._validate_momenta(momenta, centers.shape[0])
        q_dot = self._validate_momenta(q_dot, centers.shape[0])
        p_dot = self._validate_momenta(p_dot, centers.shape[0])
        widths = self.widths_for(centers)
        gamma_dot = self._validate_gamma(gamma_dot, centers.shape[0])

        g = self.gaussian_values(centers, momenta, gamma)
        links = self.overlap_at(centers)
        ket_links_dot = self.ket_derivative_couplings_at(centers)
        delta = centers[:, None, :] - centers[None, :, :]

        log_g_dot = gamma_dot[None, :]
        log_g_dot = log_g_dot + np.einsum(
            "bk,bk,abk->ab",
            2.0 * widths,
            q_dot,
            delta,
            optimize=True,
        )
        log_g_dot = log_g_dot + 1j * np.einsum("bk,abk->ab", p_dot, delta, optimize=True)
        log_g_dot = log_g_dot - 1j * np.einsum("bk,bk->b", momenta, q_dot, optimize=True)[None, :]
        g_dot = log_g_dot * g

        d_tensor = np.einsum("ab,aBbc->aBbc", g_dot, links, optimize=True)
        electronic_dot = np.einsum("bk,kaBbc->aBbc", q_dot, ket_links_dot, optimize=True)
        d_tensor += np.einsum("ab,aBbc->aBbc", g, electronic_dot, optimize=True)
        return d_tensor.reshape(g.shape[0] * links.shape[1], -1)

    def coefficient_rhs(
        self,
        c: ArrayLike,
        centers: ArrayLike,
        momenta: ArrayLike,
        *,
        q_dot: ArrayLike | None = None,
        p_dot: ArrayLike | None = None,
        gamma: ArrayLike | None = None,
        gamma_dot: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        centers = self._validate_centers(centers)
        momenta = self._validate_momenta(momenta, centers.shape[0])
        h_el = self.electronic_hamiltonian_at(centers)
        c0 = np.asarray(c, dtype=complex)
        if c0.shape != h_el.shape[:2]:
            raise ValueError(f"c shape {c0.shape} != {(h_el.shape[0], h_el.shape[1])}.")

        if q_dot is None:
            q_dot = momenta / self.masses[None, :]
        if p_dot is None:
            p_dot = np.zeros_like(momenta)

        phi = self.collocation_overlap(centers, momenta, gamma)
        h = self.hamiltonian(centers, momenta, gamma)
        d = self.moving_basis_matrix(centers, momenta, q_dot, p_dot, gamma_dot, gamma)
        rhs = (-1j / self.hbar) * (h @ c0.reshape(-1)) - d @ c0.reshape(-1)
        c_dot, singular_values = self._svd_solve(phi, rhs)
        return c_dot.reshape(c0.shape), singular_values

    def rhs(
        self,
        c: ArrayLike,
        centers: ArrayLike,
        momenta: ArrayLike,
        *,
        gamma: ArrayLike | None = None,
        gamma_dot: ArrayLike | None = None,
    ) -> PSGLDRRHS:
        centers = self._validate_centers(centers)
        momenta = self._validate_momenta(momenta, centers.shape[0])
        q_dot = momenta / self.masses[None, :]
        if self.force_callback is None:
            p_dot = np.zeros_like(momenta)
        else:
            p_dot = np.asarray(self.force_callback(np.asarray(c, dtype=complex), centers, momenta, self), dtype=float)
            if p_dot.shape != centers.shape:
                raise ValueError(f"force callback returned shape {p_dot.shape} != {centers.shape}.")
        if gamma_dot is None:
            if self.phase_callback is None:
                gamma_dot = np.zeros(centers.shape[0], dtype=complex)
            else:
                gamma_dot = np.asarray(
                    self.phase_callback(np.asarray(c, dtype=complex), centers, momenta, self),
                    dtype=complex,
                )
                if gamma_dot.shape != (centers.shape[0],):
                    raise ValueError(f"phase callback returned shape {gamma_dot.shape} != {(centers.shape[0],)}.")
        c_dot, singular_values = self.coefficient_rhs(
            c,
            centers,
            momenta,
            q_dot=q_dot,
            p_dot=p_dot,
            gamma=gamma,
            gamma_dot=gamma_dot,
        )
        return PSGLDRRHS(
            c_dot=c_dot,
            q_dot=q_dot,
            p_dot=p_dot,
            gamma_dot=gamma_dot,
            singular_values=singular_values,
        )

    def _svd_solve(self, matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        if s.size == 0:
            raise ValueError("Cannot solve an empty PSG-LDR system.")
        cutoff = self.svd_rcond * s[0]
        keep = s > cutoff
        if not np.any(keep):
            raise np.linalg.LinAlgError("All PSG-LDR singular values were truncated.")
        solution = vh[keep].conj().T @ ((u[:, keep].conj().T @ rhs) / s[keep])
        return solution, s

    def _validate_centers(self, centers: ArrayLike) -> np.ndarray:
        centers = np.asarray(centers, dtype=float)
        if centers.ndim == 1:
            centers = centers[:, None]
        if centers.ndim != 2 or centers.shape[1] != self.ncoord:
            raise ValueError(f"centers must have shape (ngauss, {self.ncoord}); got {centers.shape}.")
        return centers

    def _validate_momenta(self, momenta: ArrayLike, ngauss: int) -> np.ndarray:
        momenta = np.asarray(momenta, dtype=float)
        if momenta.ndim == 1 and self.ncoord == 1:
            momenta = momenta[:, None]
        if momenta.shape != (ngauss, self.ncoord):
            raise ValueError(f"momenta/velocity array shape {momenta.shape} != {(ngauss, self.ncoord)}.")
        return momenta

    def _validate_gamma(self, gamma: ArrayLike | None, ngauss: int) -> np.ndarray:
        if gamma is None:
            return np.zeros(ngauss, dtype=complex)
        gamma = np.asarray(gamma, dtype=complex)
        if gamma.shape != (ngauss,):
            raise ValueError(f"gamma shape {gamma.shape} != {(ngauss,)}.")
        return gamma
