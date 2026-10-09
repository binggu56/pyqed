"""Auxiliary-basis contour-deformation molecular G0W0."""

from functools import lru_cache

import numpy as np
import scipy.linalg as la

from .screening import StaticScreening


class ContourDeformation:
    """Restricted direct-RPA contour integration with real CD/RI factors.

    Molecular adaptation of Zhu and Chan, JCTC 17, 727–741 (2021),
    doi:10.1021/acs.jctc.0c00704: imaginary-axis quadrature plus residues,
    without analytic continuation or screening-pole truncation. A static
    subtraction integrates the orbital-pole discontinuity analytically.
    Only diagonal G0W0 is supported, with a gapped, real restricted reference.
    Finite quadrature and the real-frequency eta regulator must be converged;
    agreement with the spectral solver is in the eta -> 0 limit. This is not
    a reproduction of the periodic, unrestricted, or self-consistent methods.
    Dielectric matrices are built one frequency at a time in auxiliary space.
    The zero-frequency Cholesky factor is retained for exact reuse by static
    full BSE and TDA; no screening eigenvectors are required for those kernels.
    """

    def __init__(self, gw, energy, orbs, nw=64, quadrature_scale=0.5):
        from pyqed.gw.gw import _screening_storage

        if gw.screening != "TDH" or gw._pair_factors is None:
            raise NotImplementedError(
                "Contour GW requires TDH screening and CD/RI factors"
            )
        if isinstance(nw, bool) or int(nw) != nw or nw < 8:
            raise ValueError("nw must be an integer of at least 8")
        if not np.isfinite(quadrature_scale) or quadrature_scale <= 0:
            raise ValueError("quadrature_scale must be positive and finite")
        if not np.isfinite(gw.eta) or gw.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        self.energy = np.asarray(energy, dtype=float).copy()
        self.mo_coeff = gw.mo_coeff.copy()
        self.nocc = gw.nocc // 2
        self.factors = gw._pair_factors
        if np.iscomplexobj(self.factors):
            raise NotImplementedError("Contour GW requires real orbital factors")
        self.gaps = (self.energy[self.nocc :] - self.energy[: self.nocc, None]).ravel()
        if (
            not len(self.gaps)
            or np.any(self.gaps <= 0)
            or not np.all(np.isfinite(self.energy))
        ):
            raise ValueError(
                "Contour GW requires finite energies and positive occupied-virtual gaps"
            )
        self.pairs = self.factors[:, : self.nocc, self.nocc :].reshape(
            len(self.factors), -1
        )
        self.orbs = np.asarray(orbs)
        if (
            self.orbs.ndim != 1
            or not len(self.orbs)
            or self.orbs.dtype.kind not in "iu"
            or len(np.unique(self.orbs)) != len(self.orbs)
            or np.any(self.orbs < 0)
            or np.any(self.orbs >= len(self.energy))
        ):
            raise ValueError("orbs must contain distinct spatial orbital indices")
        self.columns = {int(p): j for j, p in enumerate(self.orbs)}
        self.eta = gw.eta
        self.block = max(
            1, min(256, int(max(1e-6, gw.max_memory) * 1e6 / (64 * len(self.factors))))
        )
        nodes, weights = np.polynomial.legendre.leggauss(int(nw))
        self.freqs = quadrature_scale * (1 + nodes) / (1 - nodes)
        self.weights = weights * 2 * quadrature_scale / (1 - nodes) ** 2
        self.w0 = np.empty((len(self.orbs), len(self.energy)))
        self.w_imag = _screening_storage(gw, (*self.w0.shape, int(nw)), float)
        for k, frequency in enumerate(np.r_[0.0, self.freqs]):
            chol = la.cho_factor(
                self.dielectric(frequency, imaginary=True),
                lower=True,
                overwrite_a=True,
                check_finite=False,
            )
            if k == 0:

                self.static_screening = StaticScreening(
                    self.factors, chol, gw.max_memory
                )
            for j, p in enumerate(self.orbs):
                rhs = self.factors[:, :, p]
                screened = la.cho_solve(chol, rhs, check_finite=False) - rhs
                values = np.sum(rhs * screened, axis=0)
                if k == 0:
                    self.w0[j] = values
                else:
                    self.w_imag[j, :, k - 1] = values

    def dielectric(self, frequency, *, imaginary=False):
        """Return 1-Pi with transition-blocked direct-RPA polarization."""
        z = frequency + 1j * self.eta
        weights = (
            -4 * self.gaps / (frequency**2 + self.gaps**2)
            if imaginary
            else 4 * self.gaps / (z * z - self.gaps**2)
        )
        matrix = np.eye(len(self.factors), dtype=weights.dtype, order="F")
        for start in range(0, len(self.gaps), self.block):
            s = slice(start, start + self.block)
            pairs = self.pairs[:, s]
            matrix -= (pairs * weights[s]) @ pairs.T
        return matrix

    def residue(self, frequency, orbital, intermediate):
        """Screened matrix element and its derivative at a positive real frequency."""
        rhs = self.factors[:, intermediate, orbital]
        solution = la.solve(
            self.dielectric(frequency),
            rhs,
            assume_a="gen",
            overwrite_a=True,
            check_finite=False,
        )
        value = rhs @ (solution - rhs)
        z = frequency + 1j * self.eta
        weights = -8 * z * self.gaps / (z * z - self.gaps**2) ** 2
        derivative = 0j
        for start in range(0, len(self.gaps), self.block):
            s = slice(start, start + self.block)
            projected = self.pairs[:, s].T @ solution
            derivative += np.sum(weights[s] * projected**2)
        return value, derivative

    def evaluate(self, orbital, omega):
        """Return diagonal correlation self-energy and its frequency derivative."""
        j = self.columns[orbital]
        delta = float(omega) - self.energy
        difference = self.w_imag[j] - self.w0[j, :, None]
        denominator = delta[:, None] ** 2 + self.freqs**2
        occupation = np.arange(len(delta)) < self.nocc
        value = np.sum((0.5 - occupation) * self.w0[j])
        value -= (
            np.sum(difference * self.weights * delta[:, None] / denominator) / np.pi
        )
        derivative = (
            -np.sum(
                difference
                * self.weights
                * (self.freqs**2 - delta[:, None] ** 2)
                / denominator**2
            )
            / np.pi
        )
        active = np.where(np.where(occupation, delta < 0, delta > 0))[0]
        for m in active:
            screened, slope = self.residue(abs(delta[m]), orbital, m)
            value += (-1 if occupation[m] else 1) * (screened - self.w0[j, m])
            derivative += slope
        return value, derivative


def kernel(
    gw, mo_energy, mo_coeff, *, orbs=None, nw=64, quadrature_scale=0.5, tol=1e-9
):
    """Solve diagonal restricted G0W0 by :class:`ContourDeformation`.

    Unrequested spatial orbitals have NaN energies, weights and residuals.
    All intermediate occupied/virtual states remain in the screening and
    self-energy. QP roots use the spectral driver's continuation prescription.
    """
    from pyqed.gw.gw import _continue_qp_root, _set_rhf_orbitals, _sigma_x_matrix

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if not np.array_equal(mo_coeff, gw.mo_coeff):
        _set_rhf_orbitals(gw, mo_energy, mo_coeff)
    gw._M = gw._charge_screening = None
    gw._qp_energy_so = np.repeat(mo_energy, 2)
    gw._contour = None
    orbs = list(range(len(mo_energy))) if orbs is None else list(orbs)
    contour = ContourDeformation(gw, mo_energy, orbs, nw, quadrature_scale)
    gw._contour = contour
    exchange = _sigma_x_matrix(gw)
    energies = np.full(len(mo_energy), np.nan)
    gw.qp_weights = np.full_like(energies, np.nan)
    gw.qp_residuals = np.full_like(energies, np.nan)
    gw.orbital_indices = contour.orbs.copy()
    for p in contour.orbs:
        static = exchange[2 * p, 2 * p] - gw.v_mf[2 * p, 2 * p]
        evaluate = lru_cache(maxsize=4)(lambda w: contour.evaluate(int(p), w))
        correction = lambda w: evaluate(float(w))[0] + static
        derivative = lambda w: evaluate(float(w))[1]
        energies[p] = _continue_qp_root(
            correction, derivative, float(mo_energy[p]), tol=tol
        )
        gw.qp_weights[p] = 1 / (1 - derivative(energies[p]).real)
        gw.qp_residuals[p] = abs(
            energies[p] - mo_energy[p] - correction(energies[p]).real
        )
    gw.contour_info = dict(
        nw=int(nw),
        quadrature_scale=quadrature_scale,
        orbs=contour.orbs.tolist(),
        eta=gw.eta,
        screening="auxiliary direct RPA",
        static_subtraction=True,
    )
    return energies
