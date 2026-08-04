"""Two-state, two-mode vibronic-coupling model of retinal photoisomerization."""

from __future__ import annotations

import numpy as np

from pyqed.units import au2ev


class RetinalHahnStock:
    r"""Hahn--Stock retinal model in the diabatic electronic representation.

    The dimensionless nuclear coordinates are the periodic torsion ``phi`` and
    the harmonic coupling mode ``q``.  In atomic units,

    .. math::

        T &= -\frac{I^{-1}}{2}\partial_\phi^2
             -\frac{\omega}{2}\partial_q^2,\\
        V_{00} &= \frac{W_0}{2}(1-\cos\phi)+\frac{\omega}{2}q^2,\\
        V_{11} &= E_1-\frac{W_1}{2}(1-\cos\phi)
                  +\frac{\omega}{2}q^2+\kappa q,\\
        V_{01} &= V_{10}=\lambda q.

    The defaults are the standard parameters of Hahn and Stock,
    J. Phys. Chem. B 104, 1146 (2000), DOI: 10.1021/jp992939g.
    Input parameters are specified in eV and stored internally in Hartree.
    """

    nstates = 2
    nmodes = 2

    def __init__(
        self,
        *,
        inverse_inertia_ev: float = 4.84e-4,
        e1_ev: float = 2.48,
        w0_ev: float = 3.6,
        w1_ev: float = 1.09,
        omega_ev: float = 0.19,
        kappa_ev: float = 0.10,
        lambda_ev: float = 0.19,
    ):
        self.inverse_inertia = float(inverse_inertia_ev) / au2ev
        self.e1 = float(e1_ev) / au2ev
        self.w0 = float(w0_ev) / au2ev
        self.w1 = float(w1_ev) / au2ev
        self.omega = float(omega_ev) / au2ev
        self.kappa = float(kappa_ev) / au2ev
        self.lam = float(lambda_ev) / au2ev
        self.edip = np.array([[0.0, 1.0], [1.0, 0.0]])

    @property
    def parameters_ev(self) -> dict[str, float]:
        """Return the Hamiltonian parameters in eV."""

        return {
            "inverse_inertia": self.inverse_inertia * au2ev,
            "e1": self.e1 * au2ev,
            "w0": self.w0 * au2ev,
            "w1": self.w1 * au2ev,
            "omega": self.omega * au2ev,
            "kappa": self.kappa * au2ev,
            "lambda": self.lam * au2ev,
        }

    def diabatic_potential(self, phi, q) -> np.ndarray:
        """Return ``V(phi, q)`` with shape ``broadcast(phi, q) + (2, 2)``."""

        phi, q = np.broadcast_arrays(
            np.asarray(phi, dtype=float),
            np.asarray(q, dtype=float),
        )
        harmonic = 0.5 * self.omega * q**2
        torsion = 1.0 - np.cos(phi)
        potential = np.zeros(phi.shape + (2, 2), dtype=float)
        potential[..., 0, 0] = 0.5 * self.w0 * torsion + harmonic
        potential[..., 1, 1] = (
            self.e1
            - 0.5 * self.w1 * torsion
            + harmonic
            + self.kappa * q
        )
        potential[..., 0, 1] = self.lam * q
        potential[..., 1, 0] = self.lam * q
        return potential

    dpes = diabatic_potential

    def adiabatic_potential(self, phi, q) -> np.ndarray:
        """Return the lower and upper adiabatic potential energies."""

        return np.linalg.eigvalsh(self.diabatic_potential(phi, q))

    apes = adiabatic_potential

    def conical_intersection(self) -> tuple[float, float]:
        """Return the conical-intersection position ``(phi, q)``."""

        cosine = 1.0 - 2.0 * self.e1 / (self.w0 + self.w1)
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("these parameters do not place a crossing at q=0")
        return float(np.arccos(cosine)), 0.0

    @staticmethod
    def cis_mask(phi) -> np.ndarray:
        """Identify the cis region, ``-pi/2 <= phi < pi/2`` modulo ``2*pi``."""

        wrapped = (np.asarray(phi) + np.pi) % (2.0 * np.pi) - np.pi
        return (wrapped >= -0.5 * np.pi) & (wrapped < 0.5 * np.pi)

    @classmethod
    def trans_mask(cls, phi) -> np.ndarray:
        """Identify the complementary trans region."""

        return ~cls.cis_mask(phi)
