#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian-wavepacket finite-basis representations for SD-DVR.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.special import ndtri
from scipy.stats import qmc

from .sddvr import SDDVR


def _as_centers(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2:
        raise ValueError("centers must have shape (nbasis, ndim) or (nbasis,).")
    return centers


def _as_widths(widths, nbasis, ndim):
    widths = np.asarray(widths, dtype=float)
    if widths.ndim == 0:
        widths = np.full((nbasis, ndim), float(widths))
    elif widths.ndim == 1:
        if widths.shape[0] == ndim:
            widths = np.tile(widths[None, :], (nbasis, 1))
        elif widths.shape[0] == nbasis and ndim == 1:
            widths = widths[:, None]
        else:
            raise ValueError(
                "1D widths must have length ndim, or length nbasis in 1D."
            )
    elif widths.shape != (nbasis, ndim):
        raise ValueError("widths must have shape (nbasis, ndim).")

    if np.any(widths <= 0.0):
        raise ValueError("Gaussian widths/exponents must be positive.")
    return widths


def _pairwise_overlap_block(centers_a, widths_a, centers_b, widths_b):
    centers_a = np.asarray(centers_a, dtype=float)
    centers_b = np.asarray(centers_b, dtype=float)
    widths_a = np.asarray(widths_a, dtype=float)
    widths_b = np.asarray(widths_b, dtype=float)

    s = np.ones((centers_a.shape[0], centers_b.shape[0]))
    ndim = centers_a.shape[1]
    for dim in range(ndim):
        a = widths_a[:, dim][:, None]
        b = widths_b[:, dim][None, :]
        ca = centers_a[:, dim][:, None]
        cb = centers_b[:, dim][None, :]
        pref = np.sqrt(2.0 * np.sqrt(a * b) / (a + b))
        expo = np.exp(-(a * b) * (ca - cb) ** 2 / (2.0 * (a + b)))
        s *= pref * expo
    return s


def _as_width_matrices(width_mats, nbasis, ndim):
    mats = np.asarray(width_mats, dtype=float)
    if mats.ndim == 0:
        mats = np.tile(np.eye(ndim)[None, :, :] * float(mats), (nbasis, 1, 1))
    elif mats.ndim == 1:
        if mats.shape[0] != ndim:
            raise ValueError("1D width_mats must have length ndim.")
        mats = np.tile(np.diag(mats)[None, :, :], (nbasis, 1, 1))
    elif mats.ndim == 2:
        if mats.shape != (ndim, ndim):
            raise ValueError("2D width_mats must have shape (ndim, ndim).")
        mats = np.tile(mats[None, :, :], (nbasis, 1, 1))
    elif mats.shape != (nbasis, ndim, ndim):
        raise ValueError("width_mats must have shape (nbasis, ndim, ndim).")

    out = np.array(mats, dtype=float, copy=True)
    for i in range(nbasis):
        out[i] = 0.5 * (out[i] + out[i].T)
        evals = np.linalg.eigvalsh(out[i])
        if np.min(evals) <= 0.0:
            raise ValueError("Each Gaussian width matrix must be symmetric positive definite.")
    return out


def _as_momenta(momenta, nbasis, ndim):
    if momenta is None:
        return np.zeros((nbasis, ndim), dtype=float)

    momenta = np.asarray(momenta, dtype=float)
    if momenta.ndim == 1:
        if momenta.shape[0] != ndim:
            raise ValueError("1D momenta must have length ndim.")
        momenta = np.tile(momenta[None, :], (nbasis, 1))
    elif momenta.shape != (nbasis, ndim):
        raise ValueError("momenta must have shape (nbasis, ndim).")
    return np.array(momenta, dtype=float, copy=True)


def _parse_ho_scales(omega, mass):
    omega = np.asarray(omega, dtype=float)
    if omega.ndim == 0:
        omega = omega[None]
    if np.any(omega <= 0.0):
        raise ValueError("omega must be positive.")

    ndim = omega.size
    if np.isscalar(mass):
        mass = np.full(ndim, float(mass))
    else:
        mass = np.asarray(mass, dtype=float)
        if mass.shape != (ndim,):
            raise ValueError("mass must be a scalar or have shape (ndim,).")
    if np.any(mass <= 0.0):
        raise ValueError("mass must be positive.")

    return omega, mass


def _as_omega(omega, ndim):
    omega = np.asarray(omega, dtype=float)
    if omega.ndim == 0:
        omega = np.full(ndim, float(omega))
    if omega.shape != (ndim,):
        raise ValueError("omega must be a scalar or have shape (ndim,).")
    if np.any(omega <= 0.0):
        raise ValueError("omega must be positive.")
    return omega


def _normal_quasi_sample(engine, ndim):
    sample = np.asarray(engine.random(1)[0], dtype=float)
    sample = np.clip(sample, 1e-12, 1.0 - 1e-12)
    if sample.shape != (ndim,):
        raise ValueError("Low-discrepancy engine returned an unexpected sample shape.")
    return ndtri(sample)


def _centered_fourier_indices(n):
    if n % 2 == 0:
        return np.arange(-n // 2 + 1, n // 2 + 1)
    return np.arange(-(n // 2), n // 2 + 1)


class PeriodicVonNeumannBasis:
    """
    One-dimensional projected periodic von Neumann basis on a Fourier grid.

    The PvN functions are Gaussian wavepackets projected into the finite
    Fourier/sinc grid before evaluation.  The biorthogonal partner ``B`` is
    built from the PvN overlap ``S`` so that ``dx * B.conj().T @ G = I``.
    """

    def __init__(
        self,
        n_position,
        n_momentum,
        length,
        x_min=0.0,
        hbar=1.0,
        sigma_x=None,
        s_thresh=1e-12,
    ):
        self.n_position = int(n_position)
        self.n_momentum = int(n_momentum)
        if self.n_position <= 0 or self.n_momentum <= 0:
            raise ValueError("n_position and n_momentum must be positive.")

        self.nbasis = self.n_position * self.n_momentum
        self.length = float(length)
        self.x_min = float(x_min)
        self.hbar = float(hbar)
        self.s_thresh = float(s_thresh)
        if self.length <= 0.0:
            raise ValueError("length must be positive.")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")

        self.dx = self.length / self.nbasis
        self.position_spacing = self.length / self.n_position
        self.momentum_spacing = 2.0 * np.pi * self.hbar / self.position_spacing
        if sigma_x is None:
            sigma_x = np.sqrt(
                self.hbar * self.position_spacing / self.momentum_spacing
            )
        self.sigma_x = float(sigma_x)
        if self.sigma_x <= 0.0:
            raise ValueError("sigma_x must be positive.")

        y_grid = np.arange(self.nbasis, dtype=float) * self.dx
        self.grid = self.x_min + y_grid
        self.positions = self.x_min + np.arange(self.n_position) * self.position_spacing
        bandwidth = np.pi * self.hbar * self.nbasis / self.length
        self.momenta = (
            -bandwidth
            + (np.arange(self.n_momentum, dtype=float) + 0.5)
            * self.momentum_spacing
        )

        self.fourier_indices = _centered_fourier_indices(self.nbasis)
        self.wave_numbers = 2.0 * np.pi * self.fourier_indices / self.length
        self.values = self._build_projected_values(y_grid)
        self.overlap = self.dx * (self.values.conj().T @ self.values)
        self.biorthogonal_values = self._build_biorthogonal_values()

    def _build_projected_values(self, y_grid):
        eikx = np.exp(1j * np.outer(y_grid, self.wave_numbers)) / np.sqrt(self.length)
        norm = (1.0 / (2.0 * np.pi * self.sigma_x**2)) ** 0.25
        gaussian_ft = norm * np.sqrt(4.0 * np.pi * self.sigma_x**2)

        columns = []
        for q_abs in self.positions:
            q = q_abs - self.x_min
            phase = np.exp(-1j * self.wave_numbers * q)
            for p in self.momenta:
                coeff = gaussian_ft * np.exp(
                    -self.sigma_x**2 * (self.wave_numbers - p / self.hbar) ** 2
                ) * phase
                values = eikx @ coeff
                col_norm = np.sqrt(self.dx * np.vdot(values, values).real)
                if col_norm <= 0.0:
                    raise ValueError("Projected PvN function has zero grid norm.")
                columns.append(values / col_norm)
        return np.column_stack(columns)

    def _build_biorthogonal_values(self):
        evals = np.linalg.eigvalsh(0.5 * (self.overlap + self.overlap.conj().T))
        if np.min(evals) <= self.s_thresh:
            raise ValueError(
                "PvN overlap is linearly dependent or ill-conditioned; "
                "adjust the grid or lower s_thresh."
            )
        return np.linalg.solve(self.overlap.T, self.values.T).T

    def _fourier_grid_operator(self, diagonal):
        diagonal = np.asarray(diagonal)
        if diagonal.shape != (self.nbasis,):
            raise ValueError("diagonal must have shape (nbasis,).")
        y_grid = self.grid - self.x_min
        fourier = np.exp(1j * np.outer(y_grid, self.wave_numbers)) / np.sqrt(self.length)
        return fourier @ np.diag(diagonal) @ (self.dx * fourier.conj().T)

    def momentum_grid_operator(self):
        """Momentum operator acting on grid-value vectors."""
        return self._fourier_grid_operator(self.hbar * self.wave_numbers)

    def kinetic_grid_operator(self, mass=1.0):
        """Kinetic-energy operator acting on grid-value vectors."""
        mass = float(mass)
        if mass <= 0.0:
            raise ValueError("mass must be positive.")
        return self._fourier_grid_operator(
            (self.hbar * self.wave_numbers) ** 2 / (2.0 * mass)
        )

    def pvb_operator(self, grid_operator):
        """
        Return mixed PvB matrix elements ``<g_i|O|b_j>``.
        """
        op = np.asarray(grid_operator)
        if op.shape != (self.nbasis, self.nbasis):
            raise ValueError("grid_operator must have shape (nbasis, nbasis).")
        return self.dx * (self.values.conj().T @ op @ self.biorthogonal_values)

    def local_operator(self, values):
        """
        Return ``<g_i|f(x)|b_j>`` from values on the coordinate grid.
        """
        values = np.asarray(values)
        if values.shape != (self.nbasis,):
            raise ValueError("values must have shape (nbasis,).")
        weighted_b = values[:, None] * self.biorthogonal_values
        return self.dx * (self.values.conj().T @ weighted_b)

    def local_matrix_operator(self, values):
        """
        Return ``<g_i|V_ab(x)|b_j>`` for grid-local matrix-valued operators.
        """
        values = np.asarray(values)
        if values.ndim != 3 or values.shape[0] != self.nbasis:
            raise ValueError("values must have shape (nbasis, nrow, ncol).")
        return self.dx * np.einsum(
            "xi,xab,xj->iajb",
            self.values.conj(),
            values,
            self.biorthogonal_values,
            optimize=True,
        )

    def kinetic_operator(self, mass=1.0):
        """Return mixed PvB matrix elements of ``p^2 / (2 mass)``."""
        return self.pvb_operator(self.kinetic_grid_operator(mass=mass))

    def pvn_coefficients(self, wavefunction):
        """
        Return ``<g_j|psi>`` coefficients for expansion in the biorthogonal basis.
        """
        psi = np.asarray(wavefunction, dtype=complex)
        if psi.shape != (self.nbasis,):
            raise ValueError("wavefunction must have shape (nbasis,).")
        return self.dx * (self.values.conj().T @ psi)

    def biorthogonal_coefficients(self, wavefunction):
        """
        Return ``<b_j|psi>`` coefficients for expansion in the PvN basis.
        """
        psi = np.asarray(wavefunction, dtype=complex)
        if psi.shape != (self.nbasis,):
            raise ValueError("wavefunction must have shape (nbasis,).")
        return self.dx * (self.biorthogonal_values.conj().T @ psi)

    def reconstruct_from_pvn_coefficients(self, coefficients):
        """
        Reconstruct from localized PvN overlaps using the biorthogonal basis.
        """
        coeff = np.asarray(coefficients, dtype=complex)
        if coeff.shape != (self.nbasis,):
            raise ValueError("coefficients must have shape (nbasis,).")
        return self.biorthogonal_values @ coeff

    def reconstruct_from_biorthogonal_coefficients(self, coefficients):
        """
        Reconstruct from biorthogonal overlaps using the PvN basis.
        """
        coeff = np.asarray(coefficients, dtype=complex)
        if coeff.shape != (self.nbasis,):
            raise ValueError("coefficients must have shape (nbasis,).")
        return self.values @ coeff

    def biorthogonal_exchange(self, wavefunction):
        """
        PvB exchange: project with PvN functions and reconstruct with BvN functions.
        """
        return self.reconstruct_from_pvn_coefficients(
            self.pvn_coefficients(wavefunction)
        )


class GaussianWavepacketFBR:
    """
    Product Gaussian-wavepacket finite basis in arbitrary dimension.

    Notes
    -----
    The basis functions are normalized real Gaussians

        phi_i(q) = prod_k (a_ik / pi)^(1/4) exp[-a_ik (q_k - c_ik)^2 / 2]

    with centers ``c_ik`` and positive exponents ``a_ik``.
    """

    def __init__(self, centers, widths=1.0, labels=None, s_thresh=1e-12):
        self.centers = _as_centers(centers)
        self.nbasis, self.ndim = self.centers.shape
        self.widths = _as_widths(widths, self.nbasis, self.ndim)
        self.labels = list(labels) if labels is not None else [f"q{i}" for i in range(self.ndim)]
        self.s_thresh = s_thresh

        if len(self.labels) != self.ndim:
            raise ValueError("labels must have length ndim.")

        self.overlap = self._build_overlap()
        self.coordinate_ops = self._build_coordinate_ops()
        self.orthogonalizer = self._build_orthogonalizer()

    @classmethod
    def random_ho(
        cls,
        nbasis,
        omega,
        mass=1.0,
        widths=None,
        center_scale=1.0,
        overlap_cutoff=0.9,
        max_draws=None,
        seed=None,
        labels=None,
        s_thresh=1e-12,
    ):
        """
        Build a Gaussian basis by random sampling from oscillator-scaled widths.

        Parameters
        ----------
        nbasis : int
            Number of Gaussian functions to retain.
        omega : float or array-like
            Harmonic frequencies that set the natural coordinate scales.
        mass : float or array-like, optional
            Per-dimension masses. A scalar applies to every coordinate.
        widths : None, float, or array-like, optional
            Gaussian exponents. By default ``widths = mass * omega``, i.e. the
            ground-state harmonic-oscillator width in atomic units.
        center_scale : float, optional
            Multiplier for the sampling standard deviation of the centers.
        overlap_cutoff : float, optional
            Reject newly sampled Gaussians when their overlap with an accepted
            basis function exceeds this value.
        max_draws : int, optional
            Maximum number of random proposals. The default scales with
            ``nbasis``.
        seed : int, optional
            Seed for reproducible sampling.
        labels : list[str], optional
            Coordinate labels.
        s_thresh : float, optional
            Linear-dependence threshold passed to :class:`GaussianWavepacketFBR`.
        """
        nbasis = int(nbasis)
        if nbasis <= 0:
            raise ValueError("nbasis must be positive.")

        omega, mass = _parse_ho_scales(omega, mass)
        ndim = omega.size

        if widths is None:
            width_template = mass * omega
        else:
            width_template = _as_widths(widths, 1, ndim)[0]

        osc_length = 1.0 / np.sqrt(mass * omega)
        sigma = center_scale * osc_length
        rng = np.random.default_rng(seed)

        if max_draws is None:
            max_draws = max(50 * nbasis, 200)

        accepted = []
        width_rows = []
        draws = 0
        while len(accepted) < nbasis and draws < max_draws:
            draws += 1
            cand = rng.normal(scale=sigma, size=ndim)

            if accepted:
                accepted_arr = np.asarray(accepted, dtype=float)
                width_arr = np.asarray(width_rows, dtype=float)
                s_cand = _pairwise_overlap_block(
                    accepted_arr,
                    width_arr,
                    cand[None, :],
                    width_template[None, :],
                )[:, 0]
                if np.max(np.abs(s_cand)) >= overlap_cutoff:
                    continue

                test_centers = np.vstack((accepted_arr, cand[None, :]))
                test_widths = np.vstack((width_arr, width_template[None, :]))
                s_test = _pairwise_overlap_block(
                    test_centers,
                    test_widths,
                    test_centers,
                    test_widths,
                )
                if np.min(np.linalg.eigvalsh(0.5 * (s_test + s_test.T))) <= s_thresh:
                    continue

            accepted.append(cand)
            width_rows.append(width_template.copy())

        if len(accepted) < nbasis:
            raise ValueError(
                "Failed to generate a well-conditioned random HO Gaussian basis. "
                "Try increasing max_draws, lowering overlap_cutoff, or reducing nbasis."
            )

        return cls(
            centers=np.asarray(accepted, dtype=float),
            widths=np.asarray(width_rows, dtype=float),
            labels=labels,
            s_thresh=s_thresh,
        )

    def _pairwise_overlap(self, dim):
        a = self.widths[:, dim][:, None]
        b = self.widths[:, dim][None, :]
        ca = self.centers[:, dim][:, None]
        cb = self.centers[:, dim][None, :]

        pref = np.sqrt(2.0 * np.sqrt(a * b) / (a + b))
        expo = np.exp(-(a * b) * (ca - cb) ** 2 / (2.0 * (a + b)))
        return pref * expo

    def _build_overlap(self):
        s = np.ones((self.nbasis, self.nbasis))
        for dim in range(self.ndim):
            s *= self._pairwise_overlap(dim)
        return 0.5 * (s + s.T)

    def _build_coordinate_ops(self):
        s = self.overlap
        ops = []
        for dim in range(self.ndim):
            a = self.widths[:, dim][:, None]
            b = self.widths[:, dim][None, :]
            ca = self.centers[:, dim][:, None]
            cb = self.centers[:, dim][None, :]
            mu = (a * ca + b * cb) / (a + b)
            op = mu * s
            ops.append(0.5 * (op + op.T))
        return np.stack(ops, axis=0)

    def _build_orthogonalizer(self):
        evals, evecs = eigh(self.overlap)
        if np.min(evals) <= self.s_thresh:
            raise ValueError(
                "Gaussian basis is linearly dependent or ill-conditioned; "
                "adjust centers/widths or lower s_thresh."
            )
        return evecs @ np.diag(evals ** -0.5) @ evecs.T

    def orthonormalize(self, operator):
        op = np.asarray(operator, dtype=float)
        if op.shape != (self.nbasis, self.nbasis):
            raise ValueError("operator shape does not match the Gaussian basis size.")
        x = self.orthogonalizer
        return 0.5 * (x.T @ op @ x + (x.T @ op @ x).T)

    def orthonormal_coordinate_ops(self):
        return np.stack([self.orthonormalize(op) for op in self.coordinate_ops], axis=0)

    def diagonal_grid(self):
        """
        Diagonal coordinate values in the orthonormalized basis.
        """
        grid = np.stack(
            [np.diag(op).copy() for op in self.orthonormal_coordinate_ops()],
            axis=1,
        )
        if np.max(np.abs(np.imag(grid))) <= 1e-12:
            return grid.real
        return grid

    def diagonal_local_operator(self, func):
        """
        Build a diagonal local-operator approximation in the orthonormal basis.
        """
        grid = self.diagonal_grid()
        values = np.asarray([func(*point) for point in grid])
        op = np.diag(values)
        return 0.5 * (op + op.conj().T)

    def kinetic(self, mass=1.0):
        """
        Total kinetic-energy matrix in the non-orthogonal Gaussian basis.

        Parameters
        ----------
        mass : float or array-like, optional
            Per-dimension masses. A scalar applies to every coordinate.
        """
        if np.isscalar(mass):
            mass = np.full(self.ndim, float(mass))
        else:
            mass = np.asarray(mass, dtype=float)
            if mass.shape != (self.ndim,):
                raise ValueError("mass must be a scalar or have shape (ndim,).")
        if np.any(mass <= 0.0):
            raise ValueError("All masses must be positive.")

        t = np.zeros_like(self.overlap)
        for dim in range(self.ndim):
            a = self.widths[:, dim][:, None]
            b = self.widths[:, dim][None, :]
            delta = self.centers[:, dim][:, None] - self.centers[:, dim][None, :]
            factor = 0.5 / mass[dim] * (a * b / (a + b)) * (
                1.0 - (a * b / (a + b)) * delta ** 2
            )
            t += self.overlap * factor
        return 0.5 * (t + t.T)

    def orthonormal_kinetic(self, mass=1.0):
        """
        Total kinetic-energy matrix in the Löwdin-orthonormalized basis.
        """
        return self.orthonormalize(self.kinetic(mass=mass))

    def harmonic_hamiltonian(self, omega, mass=1.0, approximation="diagonal"):
        """
        Build an independent-HO Hamiltonian in the orthonormal Gaussian basis.

        Parameters
        ----------
        omega : float or array-like
            Per-dimension oscillator frequencies.
        mass : float or array-like, optional
            Per-dimension masses. A scalar applies to every coordinate.
        approximation : {'diagonal', 'projected'}, optional
            Potential representation. ``diagonal`` uses the diagonal local
            operator approximation (default). ``projected`` uses the projected
            quadratic operator ``0.5 * sum_i omega_i^2 Q_i^2``.
        """
        omega = _as_omega(omega, self.ndim)
        t = self.orthonormal_kinetic(mass=mass)
        mode = str(approximation).lower()
        if mode in ("diagonal", "diag", "local"):
            v = self.diagonal_local_operator(
                lambda *q: 0.5 * sum((omega[i] ** 2) * (q[i] ** 2) for i in range(self.ndim))
            )
        elif mode in ("projected", "exact", "quadratic"):
            q_ops = self.orthonormal_coordinate_ops()
            v = np.zeros((self.nbasis, self.nbasis), dtype=q_ops.dtype)
            for i in range(self.ndim):
                v += 0.5 * omega[i] ** 2 * (q_ops[i] @ q_ops[i])
            v = 0.5 * (v + v.conj().T)
        else:
            raise ValueError(
                "approximation must be 'diagonal' or 'projected'."
            )
        return t + v

    def harmonic_hamiltonian_sddvr(
        self,
        omega,
        mass=1.0,
        approximation="diagonal",
        sddvr=None,
        tol=1e-10,
        max_iter=1000,
        verbose=False,
    ):
        """
        Build an independent-HO Hamiltonian in the SD-DVR basis.

        The default potential mode is the diagonal local approximation.
        """
        omega = _as_omega(omega, self.ndim)
        sd = self.to_sddvr(tol=tol, max_iter=max_iter, verbose=verbose) if sddvr is None else sddvr
        t_sd = sd.fbr2dvr(self.orthonormal_kinetic(mass=mass))
        mode = str(approximation).lower()
        if mode in ("diagonal", "diag", "local"):
            v_sd = sd.local_operator(
                lambda *q: 0.5 * sum((omega[i] ** 2) * (q[i] ** 2) for i in range(self.ndim))
            )
        elif mode in ("projected", "exact", "quadratic"):
            q_ops = self.orthonormal_coordinate_ops()
            v_fbr = np.zeros((self.nbasis, self.nbasis), dtype=q_ops.dtype)
            for i in range(self.ndim):
                v_fbr += 0.5 * omega[i] ** 2 * (q_ops[i] @ q_ops[i])
            v_fbr = 0.5 * (v_fbr + v_fbr.conj().T)
            v_sd = sd.fbr2dvr(v_fbr)
        else:
            raise ValueError(
                "approximation must be 'diagonal' or 'projected'."
            )
        return t_sd + v_sd, sd

    def to_sddvr(self, tol=1e-10, max_iter=1000, verbose=False):
        """
        Build an SD-DVR from the orthonormalized Gaussian coordinate operators.
        """
        return SDDVR(
            self.orthonormal_coordinate_ops(),
            labels=self.labels,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )


class MatrixGaussianWavepacketFBR:
    """
    Gaussian-wavepacket FBR with full real anisotropic width matrices.
    """

    def __init__(self, centers, width_mats, labels=None, s_thresh=1e-12, momenta=None):
        self.centers = _as_centers(centers)
        self.nbasis, self.ndim = self.centers.shape
        self.width_mats = _as_width_matrices(width_mats, self.nbasis, self.ndim)
        self.momenta = _as_momenta(momenta, self.nbasis, self.ndim)
        self.labels = list(labels) if labels is not None else [f"q{i}" for i in range(self.ndim)]
        self.s_thresh = s_thresh

        if len(self.labels) != self.ndim:
            raise ValueError("labels must have length ndim.")

        self.overlap = self._build_overlap()
        self.coordinate_ops = self._build_coordinate_ops()
        self.orthogonalizer = self._build_orthogonalizer()

    def _pair_data(self, i, j):
        ai = self.width_mats[i]
        aj = self.width_mats[j]
        ci = self.centers[i]
        cj = self.centers[j]
        pi = self.momenta[i]
        pj = self.momenta[j]

        mat = ai + aj
        rhs = ai @ ci + aj @ cj + 1j * (pi - pj)
        mu = np.linalg.solve(mat, rhs)

        sign_i, logdet_i = np.linalg.slogdet(ai)
        sign_j, logdet_j = np.linalg.slogdet(aj)
        sign_m, logdet_m = np.linalg.slogdet(mat)
        if sign_i <= 0 or sign_j <= 0 or sign_m <= 0:
            raise ValueError("Width matrices must be positive definite.")

        exponent = 0.5 * rhs.T @ mu
        exponent -= 0.5 * ci.T @ ai @ ci
        exponent -= 0.5 * cj.T @ aj @ cj
        log_pref = 0.5 * self.ndim * np.log(2.0) + 0.25 * (logdet_i + logdet_j) - 0.5 * logdet_m
        sij = np.exp(log_pref + exponent)
        sigma = np.linalg.inv(mat)
        return sij, mu, sigma, aj, cj, pj

    def _build_overlap(self):
        s = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        for i in range(self.nbasis):
            for j in range(i, self.nbasis):
                sij, _, _, _, _, _ = self._pair_data(i, j)
                s[i, j] = sij
                s[j, i] = np.conjugate(sij)
        return 0.5 * (s + s.conj().T)

    def _build_coordinate_ops(self):
        ops = np.zeros((self.ndim, self.nbasis, self.nbasis), dtype=complex)
        for i in range(self.nbasis):
            for j in range(i, self.nbasis):
                sij, mu, _, _, _, _ = self._pair_data(i, j)
                for dim in range(self.ndim):
                    val = sij * mu[dim]
                    ops[dim, i, j] = val
                    ops[dim, j, i] = np.conjugate(val)
        return 0.5 * (ops + np.swapaxes(ops.conj(), -1, -2))

    def _build_orthogonalizer(self):
        evals, evecs = eigh(self.overlap)
        if np.min(evals) <= self.s_thresh:
            raise ValueError(
                "Gaussian basis is linearly dependent or ill-conditioned; "
                "adjust centers/width matrices or lower s_thresh."
            )
        return evecs @ np.diag(evals ** -0.5) @ evecs.conj().T

    def orthonormalize(self, operator):
        op = np.asarray(operator)
        if op.shape != (self.nbasis, self.nbasis):
            raise ValueError("operator shape does not match the Gaussian basis size.")
        x = self.orthogonalizer
        out = x.conj().T @ op @ x
        return 0.5 * (out + out.conj().T)

    def orthonormal_coordinate_ops(self):
        return np.stack([self.orthonormalize(op) for op in self.coordinate_ops], axis=0)

    def diagonal_grid(self):
        """
        Diagonal coordinate values in the orthonormalized basis.
        """
        grid = np.stack(
            [np.diag(op).copy() for op in self.orthonormal_coordinate_ops()],
            axis=1,
        )
        if np.max(np.abs(np.imag(grid))) <= 1e-12:
            return grid.real
        return grid

    def diagonal_local_operator(self, func):
        """
        Build a diagonal local-operator approximation in the orthonormal basis.
        """
        grid = self.diagonal_grid()
        values = np.asarray([func(*point) for point in grid])
        op = np.diag(values)
        return 0.5 * (op + op.conj().T)

    def kinetic(self, mass=1.0):
        if np.isscalar(mass):
            mass = np.full(self.ndim, float(mass))
        else:
            mass = np.asarray(mass, dtype=float)
            if mass.shape != (self.ndim,):
                raise ValueError("mass must be a scalar or have shape (ndim,).")
        if np.any(mass <= 0.0):
            raise ValueError("All masses must be positive.")

        weight = np.diag(1.0 / mass)
        t = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        for i in range(self.nbasis):
            for j in range(i, self.nbasis):
                sij, mu, sigma, aj, cj, pj = self._pair_data(i, j)
                eta = -aj @ mu + aj @ cj + 1j * pj
                term = eta.T @ weight @ eta + np.trace(weight @ aj @ sigma @ aj) - np.trace(weight @ aj)
                val = -0.5 * sij * term
                t[i, j] = val
                t[j, i] = np.conjugate(val)
        return 0.5 * (t + t.conj().T)

    def orthonormal_kinetic(self, mass=1.0):
        return self.orthonormalize(self.kinetic(mass=mass))

    def harmonic_hamiltonian(self, omega, mass=1.0, approximation="diagonal"):
        """
        Build an independent-HO Hamiltonian in the orthonormal Gaussian basis.

        Parameters
        ----------
        omega : float or array-like
            Per-dimension oscillator frequencies.
        mass : float or array-like, optional
            Per-dimension masses. A scalar applies to every coordinate.
        approximation : {'diagonal', 'projected'}, optional
            Potential representation. ``diagonal`` uses the diagonal local
            operator approximation (default). ``projected`` uses the projected
            quadratic operator ``0.5 * sum_i omega_i^2 Q_i^2``.
        """
        omega = _as_omega(omega, self.ndim)
        t = self.orthonormal_kinetic(mass=mass)
        mode = str(approximation).lower()
        if mode in ("diagonal", "diag", "local"):
            v = self.diagonal_local_operator(
                lambda *q: 0.5 * sum((omega[i] ** 2) * (q[i] ** 2) for i in range(self.ndim))
            )
        elif mode in ("projected", "exact", "quadratic"):
            q_ops = self.orthonormal_coordinate_ops()
            v = np.zeros((self.nbasis, self.nbasis), dtype=q_ops.dtype)
            for i in range(self.ndim):
                v += 0.5 * omega[i] ** 2 * (q_ops[i] @ q_ops[i])
            v = 0.5 * (v + v.conj().T)
        else:
            raise ValueError(
                "approximation must be 'diagonal' or 'projected'."
            )
        return t + v

    def harmonic_hamiltonian_sddvr(
        self,
        omega,
        mass=1.0,
        approximation="diagonal",
        sddvr=None,
        tol=1e-10,
        max_iter=1000,
        verbose=False,
    ):
        """
        Build an independent-HO Hamiltonian in the SD-DVR basis.

        The default potential mode is the diagonal local approximation.
        """
        omega = _as_omega(omega, self.ndim)
        sd = self.to_sddvr(tol=tol, max_iter=max_iter, verbose=verbose) if sddvr is None else sddvr
        t_sd = sd.fbr2dvr(self.orthonormal_kinetic(mass=mass))
        mode = str(approximation).lower()
        if mode in ("diagonal", "diag", "local"):
            v_sd = sd.local_operator(
                lambda *q: 0.5 * sum((omega[i] ** 2) * (q[i] ** 2) for i in range(self.ndim))
            )
        elif mode in ("projected", "exact", "quadratic"):
            q_ops = self.orthonormal_coordinate_ops()
            v_fbr = np.zeros((self.nbasis, self.nbasis), dtype=q_ops.dtype)
            for i in range(self.ndim):
                v_fbr += 0.5 * omega[i] ** 2 * (q_ops[i] @ q_ops[i])
            v_fbr = 0.5 * (v_fbr + v_fbr.conj().T)
            v_sd = sd.fbr2dvr(v_fbr)
        else:
            raise ValueError(
                "approximation must be 'diagonal' or 'projected'."
            )
        return t_sd + v_sd, sd

    def to_sddvr(self, tol=1e-10, max_iter=1000, verbose=False):
        ops = self.orthonormal_coordinate_ops()
        if np.max(np.abs(ops.imag)) > 1e-10:
            raise NotImplementedError(
                "SD-DVR currently supports only real-symmetric coordinate operators. "
                "Use zero momenta or a complex joint diagonalizer."
            )
        return SDDVR(
            ops.real,
            labels=self.labels,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )


class AnisotropicGaussianWavepacketFBR(MatrixGaussianWavepacketFBR):
    """
    Real anisotropic Gaussian FBR with full width matrices and zero momenta.
    """

    def __init__(self, centers, width_mats, labels=None, s_thresh=1e-12):
        super().__init__(
            centers=centers,
            width_mats=width_mats,
            labels=labels,
            s_thresh=s_thresh,
            momenta=None,
        )


class ComplexGaussianWavepacketFBR(MatrixGaussianWavepacketFBR):
    """
    Complex phase-bearing Gaussian FBR with anisotropic width matrices.
    """

    @classmethod
    def importance_sampled_ho(
        cls,
        nbasis,
        omega,
        mass=1.0,
        width_mats=None,
        center_scale=1.0,
        momentum_scale=1.0,
        overlap_cutoff=0.9,
        max_draws=None,
        seed=None,
        sampling="random",
        labels=None,
        s_thresh=1e-12,
    ):
        """
        Build a complex Gaussian basis by HO importance sampling in phase space.

        Centers are sampled from oscillator-scaled normal distributions with
        standard deviation ``center_scale / sqrt(mass * omega)`` and momenta
        from ``momentum_scale * sqrt(mass * omega)``. Candidates are rejected
        when they overlap too strongly with the accepted basis or make the
        overlap matrix ill-conditioned.

        Parameters
        ----------
        sampling : {'random', 'sobol', 'halton'}, optional
            Phase-space sampling strategy. ``sobol`` and ``halton`` use
            low-discrepancy quasi-random sequences before transforming the
            samples to normal variates.
        """
        nbasis = int(nbasis)
        if nbasis <= 0:
            raise ValueError("nbasis must be positive.")

        omega, mass = _parse_ho_scales(omega, mass)
        ndim = omega.size
        osc_length = 1.0 / np.sqrt(mass * omega)
        osc_momentum = np.sqrt(mass * omega)

        if width_mats is None:
            width_template = np.diag(mass * omega)
        else:
            width_template = _as_width_matrices(width_mats, 1, ndim)[0]

        rng = np.random.default_rng(seed)
        sampling = str(sampling).lower()
        if sampling == "random":
            phase_space_sample = lambda: rng.normal(size=2 * ndim)
        elif sampling == "sobol":
            engine = qmc.Sobol(d=2 * ndim, scramble=True, seed=seed)
            phase_space_sample = lambda: _normal_quasi_sample(engine, 2 * ndim)
        elif sampling == "halton":
            engine = qmc.Halton(d=2 * ndim, scramble=True, seed=seed)
            phase_space_sample = lambda: _normal_quasi_sample(engine, 2 * ndim)
        else:
            raise ValueError("sampling must be 'random', 'sobol', or 'halton'.")

        if max_draws is None:
            max_draws = max(60 * nbasis, 250)

        accepted_centers = []
        accepted_momenta = []
        draws = 0
        while len(accepted_centers) < nbasis and draws < max_draws:
            draws += 1
            phase_point = phase_space_sample()
            center = phase_point[:ndim] * (center_scale * osc_length)
            momentum = phase_point[ndim:] * (momentum_scale * osc_momentum)

            if accepted_centers:
                trial_centers = np.vstack(
                    (np.asarray(accepted_centers, dtype=float), center[None, :])
                )
                trial_momenta = np.vstack(
                    (np.asarray(accepted_momenta, dtype=float), momentum[None, :])
                )
                trial_widths = np.tile(width_template[None, :, :], (len(trial_centers), 1, 1))
                try:
                    trial = cls(
                        centers=trial_centers,
                        width_mats=trial_widths,
                        momenta=trial_momenta,
                        labels=labels,
                        s_thresh=s_thresh,
                    )
                except ValueError:
                    continue
                last_overlap = np.abs(trial.overlap[:-1, -1])
                if np.max(last_overlap) >= overlap_cutoff:
                    continue
            accepted_centers.append(center)
            accepted_momenta.append(momentum)

        if len(accepted_centers) < nbasis:
            raise ValueError(
                "Failed to generate a well-conditioned importance-sampled complex HO "
                "Gaussian basis. Try increasing max_draws, lowering overlap_cutoff, "
                "or reducing nbasis."
            )

        return cls(
            centers=np.asarray(accepted_centers, dtype=float),
            width_mats=np.tile(width_template[None, :, :], (nbasis, 1, 1)),
            momenta=np.asarray(accepted_momenta, dtype=float),
            labels=labels,
            s_thresh=s_thresh,
        )

    def __init__(self, centers, width_mats, momenta, labels=None, s_thresh=1e-12):
        super().__init__(
            centers=centers,
            width_mats=width_mats,
            labels=labels,
            s_thresh=s_thresh,
            momenta=momenta,
        )
