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


def _normal_quasi_sample(engine, ndim):
    sample = np.asarray(engine.random(1)[0], dtype=float)
    sample = np.clip(sample, 1e-12, 1.0 - 1e-12)
    if sample.shape != (ndim,):
        raise ValueError("Low-discrepancy engine returned an unexpected sample shape.")
    return ndtri(sample)


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
