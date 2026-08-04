"""Three-state torsional model of retinal photoisomerization."""

from __future__ import annotations

import numpy as np

from pyqed.units import au2ev, kcalmol2au


class RetinalHumphreySchulten:
    r"""One-coordinate, three-state Humphrey--Schulten retinal model.

    The electronic basis is ordered as ``(a, b, c)``.  In the reaction-
    coordinate convention of Fig. 3 of Humphrey *et al.*, Biophys. J. 75,
    1689 (1998), the diabatic curves are

    .. math::

        E_a &= 54\cos^2(\phi/2),\\
        E_b &= 54\sin^2(\phi/2),\\
        E_c &= 50 + 4\sin^2\phi

    in kcal/mol.  Thus ``b`` is the all-trans ground state, ``c`` is the
    optically prepared state, and ``a`` becomes the 13-cis product state.
    Constant diabatic couplings are 0.5, 1.0, and 1.0 kcal/mol for ``ab``,
    ``ac``, and ``bc``, respectively.

    The original model used a full classical chromophore/protein force field.
    ``inverse_inertia_ev`` therefore supplies an effective one-coordinate
    quantum kinetic parameter; its default is the standard retinal torsional
    value used by Hahn and Stock.
    """

    nstates = 3
    nmodes = 1
    state_labels = ("a", "b", "c")

    def __init__(
        self,
        *,
        inverse_inertia_ev: float = 4.84e-4,
        ka_kcalmol: float = 54.0,
        kb_kcalmol: float = 54.0,
        c_offset_kcalmol: float = 50.0,
        kc_kcalmol: float = 4.0,
        vab_kcalmol: float = 0.5,
        vac_kcalmol: float = 1.0,
        vbc_kcalmol: float = 1.0,
    ):
        if inverse_inertia_ev <= 0.0:
            raise ValueError("inverse_inertia_ev must be positive")
        self.inverse_inertia = float(inverse_inertia_ev) / au2ev
        self.ka = float(ka_kcalmol) * kcalmol2au
        self.kb = float(kb_kcalmol) * kcalmol2au
        self.c_offset = float(c_offset_kcalmol) * kcalmol2au
        self.kc = float(kc_kcalmol) * kcalmol2au
        self.vab = float(vab_kcalmol) * kcalmol2au
        self.vac = float(vac_kcalmol) * kcalmol2au
        self.vbc = float(vbc_kcalmol) * kcalmol2au

        self.ac_transition = np.zeros((3, 3))
        self.ac_transition[0, 2] = self.ac_transition[2, 0] = 1.0

    @property
    def parameters_kcalmol(self) -> dict[str, float]:
        """Return the published potential and coupling parameters."""

        return {
            "ka": self.ka / kcalmol2au,
            "kb": self.kb / kcalmol2au,
            "c_offset": self.c_offset / kcalmol2au,
            "kc": self.kc / kcalmol2au,
            "vab": self.vab / kcalmol2au,
            "vac": self.vac / kcalmol2au,
            "vbc": self.vbc / kcalmol2au,
        }

    def diabatic_potential(self, phi) -> np.ndarray:
        """Return the ``(a,b,c)`` diabatic Hamiltonian at ``phi``."""

        phi = np.asarray(phi, dtype=float)
        potential = np.zeros(phi.shape + (3, 3), dtype=float)
        potential[..., 0, 0] = self.ka * np.cos(0.5 * phi) ** 2
        potential[..., 1, 1] = self.kb * np.sin(0.5 * phi) ** 2
        potential[..., 2, 2] = self.c_offset + self.kc * np.sin(phi) ** 2
        potential[..., 0, 1] = potential[..., 1, 0] = self.vab
        potential[..., 0, 2] = potential[..., 2, 0] = self.vac
        potential[..., 1, 2] = potential[..., 2, 1] = self.vbc
        return potential

    dpes = diabatic_potential

    def adiabatic_potential(self, phi) -> np.ndarray:
        """Return the three adiabatic energies at ``phi``."""

        return np.linalg.eigvalsh(self.diabatic_potential(phi))

    apes = adiabatic_potential

    @staticmethod
    def wrapped_angle(phi) -> np.ndarray:
        """Wrap angles to ``[-pi, pi)``."""

        return (np.asarray(phi) + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def trans_mask(cls, phi) -> np.ndarray:
        """Select the all-trans basin, ``|phi| < pi/2``."""

        return np.abs(cls.wrapped_angle(phi)) < 0.5 * np.pi

    @classmethod
    def product_mask(cls, phi) -> np.ndarray:
        """Select the complementary 13-cis product basin."""

        return ~cls.trans_mask(phi)
