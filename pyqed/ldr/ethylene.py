"""Electronic-data helpers for a local ethylene conical-intersection chart.

The chart is an adaptation of the twisted--pyramidalized ethylene MECI used by
Westermayr and co-workers, J. Phys. Chem. Lett. 2023,
https://doi.org/10.1021/acs.jpclett.3c01649.  It is not an optimized branching
plane: the two coordinates are finite internal deformations centered on the
published MRCI geometry.  Electronic data are generated with either
SA(2)-CASSCF(2,2) or fixed-orbital CASCI(2,2), so neither option reproduces the
published MRCI energies or its dynamic-correlation accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jax import config as jax_config
from jax import numpy as jnp
import numpy as np

from pyqed.units import au2angstrom


jax_config.update("jax_enable_x64", True)


ETHYLENE_SPECIES = ("C", "H", "H", "C", "H", "H")
ETHYLENE_MECI_ANGSTROM = np.asarray(
    (
        (0.00000000, 0.00000000, 0.66796400),
        (0.92288300, 0.00000000, 1.24294900),
        (-0.92288300, 0.00000000, 1.24294900),
        (0.00000000, 0.00000000, -0.66796400),
        (0.54030916, 0.92288300, -0.86462045),
        (0.54030916, -0.92288300, -0.86462045),
    ),
    dtype=float,
)
ETHYLENE_MECI_BOHR = ETHYLENE_MECI_ANGSTROM / au2angstrom
ETHYLENE_CI_PYRAMID_SHIFT = -0.72488
ETHYLENE_CI_BOUNDS = ((-np.pi, np.pi), (-1.6, 1.6))


def _rotation_z(angle):
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    return jnp.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def _rotation_y(angle):
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    return jnp.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
    )


def ethylene_ci_geometry(coordinates):
    """Return the local ethylene MECI chart in Bohr.

    ``coordinates = (torsion, pyramid)`` are angular displacements in radians.
    ``torsion`` rotates the lower CH2 group about the C--C axis in a body-fixed
    frame and is exactly 2π periodic. ``pyramid`` changes the lower-carbon
    pyramidalization. The origin is the
    restricted SA(2)-CASSCF(2,2)/6-31G* crossing obtained by applying an
    additional pyramidalization to the published MRCI template.  These
    coordinates are chemically transparent finite deformations, not
    mass-weighted normal modes and not the exact gradient-difference/
    derivative-coupling branching vectors.
    """

    torsion, pyramid = coordinates
    pyramid = pyramid + ETHYLENE_CI_PYRAMID_SHIFT
    reference = jnp.asarray(ETHYLENE_MECI_BOHR)
    geometry = reference
    upper_carbon, lower_carbon = reference[0], reference[3]
    upper = reference[1:3] - upper_carbon
    lower = (reference[4:6] - lower_carbon) @ _rotation_y(pyramid).T
    lower = lower @ _rotation_z(-torsion).T
    geometry = geometry.at[1:3].set(upper + upper_carbon)
    geometry = geometry.at[4:6].set(lower + lower_carbon)
    return geometry


def ethylene_ci_protocol(*, basis="6-31g*", method="sa-casscf", nroots=2):
    """Return the complete electronic protocol stored in database keys."""

    method = str(method).lower()
    if method not in {"sa-casscf", "casci"}:
        raise ValueError("method must be 'sa-casscf' or 'casci'")
    nroots = int(nroots)
    if nroots != 2:
        raise ValueError("the ethylene CI benchmark currently requires two roots")
    return {
        "schema": "pyqed-ethylene-ci-periodic-2d-v6",
        "system": "C2H4",
        "geometry_unit": "bohr",
        "reference_geometry_source_unit": "angstrom",
        "reference_geometry": "twisted-pyramidalized MRCI MECI source template",
        "reference_doi": "10.1021/acs.jpclett.3c01649",
        "chart": {
            "coordinates": ["torsion", "pyramidalization"],
            "units": ["radian", "radian"],
            "boundary_conditions": ["periodic", "dirichlet-domain-converged"],
            "torsion_period_radian": float(2.0 * np.pi),
            "origin_pyramidalization_shift_radian": ETHYLENE_CI_PYRAMID_SHIFT,
            "origin_calibration": "restricted SA(2)-CASSCF(2,2)/6-31G* crossing",
            "fidelity": "local finite-deformation adaptation; not exact branching vectors",
        },
        "charge": 0,
        "spin": 0,
        "basis": str(basis),
        "method": method,
        "nroots": nroots,
        "active_space": {"electrons": 2, "orbitals": 2},
        "state_average": (
            {"roots": nroots, "weights": [1.0 / nroots] * nroots}
            if method == "sa-casscf"
            else None
        ),
        "ci_solver": "direct_spin0_symm",
        "limitations": [
            "no MRCI dynamic correlation",
            "no MECI reoptimization",
            "two-coordinate frozen-geometry model",
            "crossing-center calibration is specific to SA-CASSCF/6-31G*",
        ],
    }


def one_drive_data_root():
    """Return the requested OneDrive ``data`` directory on macOS."""

    return (
        Path.home()
        / "Library"
        / "CloudStorage"
        / "OneDrive-西湖大学"
        / "data"
    )


def default_ethylene_database_path():
    """Return the external persistent database path for this benchmark."""

    return (
        one_drive_data_root()
        / "pyqed"
        / "ethylene_ci_periodic_2d"
        / "electronic.sqlite"
    )


@dataclass(frozen=True)
class _EthyleneElectronicResult:
    e_tot: np.ndarray
    electronic_frame: object

    def frame(self):
        return self.electronic_frame


class _EthyleneElectronicScanner:
    def __init__(self, driver, nstates):
        self.driver = driver
        self.nstates = int(nstates)

    def __call__(self, molecule):
        return self.driver._solve(molecule, self.nstates)


class EthyleneCIElectronicDriver:
    """Two-root ab initio driver for the local ethylene CI benchmark.

    The default is equally weighted SA(2)-CASSCF(2,2) with a singlet-adapted
    CI solver.  ``method='casci'`` is provided only for inexpensive workflow
    smoke tests.  The model is an adaptation of the MRCI benchmark geometry in
    https://doi.org/10.1021/acs.jpclett.3c01649 and does not reproduce MRCI.
    """

    def __init__(
        self,
        *,
        basis="6-31g*",
        method="sa-casscf",
        nroots=2,
        max_cycle=40,
        verbose=0,
    ):
        from pyqed.qchem import Molecule

        self.basis = str(basis)
        self.method = str(method).lower()
        self.nstates = int(nroots)
        self.max_cycle = int(max_cycle)
        self.verbose = int(verbose)
        self.protocol = ethylene_ci_protocol(
            basis=self.basis, method=self.method, nroots=self.nstates
        )
        origin = np.asarray(ethylene_ci_geometry((0.0, 0.0)), dtype=float)
        self.mol = Molecule(
            atom=list(zip(ETHYLENE_SPECIES, origin)),
            charge=0,
            spin=0,
            unit="bohr",
            basis=self.basis,
        ).build(eri="dense")
        self.e_tot = None
        self._result = None

    def _solve(self, molecule, nstates):
        from pyqed.qchem import CASSCF
        from pyqed.qchem.mcscf.casci import CASCIFrame

        if int(nstates) != self.nstates:
            raise ValueError("requested root count does not match the ethylene protocol")
        mean_field = molecule.RHF(verbose=max(0, self.verbose - 1))
        mean_field.conv_tol = 1.0e-10
        mean_field.run()
        if self.method == "casci":
            electronic = molecule.casci(
                2,
                2,
                nstates=self.nstates,
                method="direct_spin0_symm",
                mf=mean_field,
                ms2=0,
                multiplicity=1,
                verbose=max(0, self.verbose - 1),
            ).run(nstates=self.nstates, method="direct_spin0_symm")
            frame = electronic.frame()
            energies = np.asarray(electronic.e_tot, dtype=float)
        else:
            electronic = CASSCF(
                mean_field,
                ncas=2,
                nelecas=2,
                ms2=0,
                multiplicity=1,
                max_cycle=self.max_cycle,
                max_micro_cycle=6,
                conv_tol=2.0e-7,
                conv_tol_grad=2.0e-5,
                conv_tol_step=1.0e-3,
                max_step=0.05,
                ci_method="direct_spin0_symm",
                coupling="qn",
                verbose=self.verbose,
            )
            electronic.state_average(np.full(self.nstates, 1.0 / self.nstates))
            electronic.run(nstates=self.nstates)
            frame = CASCIFrame.from_casci(electronic.casci)
            energies = np.asarray(electronic.e_tot, dtype=float)
        return _EthyleneElectronicResult(energies, frame)

    def run(self, nstates=None):
        nstates = self.nstates if nstates is None else int(nstates)
        self._result = self._solve(self.mol, nstates)
        self.e_tot = np.array(self._result.e_tot, copy=True)
        return self

    def frame(self):
        if self._result is None:
            raise ValueError("run the ethylene electronic driver first")
        return self._result.frame()

    def as_scanner(self, nstates=None, **_kwargs):
        return _EthyleneElectronicScanner(
            self, self.nstates if nstates is None else int(nstates)
        )


__all__ = [
    "ETHYLENE_CI_BOUNDS",
    "ETHYLENE_MECI_ANGSTROM",
    "ETHYLENE_MECI_BOHR",
    "ETHYLENE_CI_PYRAMID_SHIFT",
    "ETHYLENE_SPECIES",
    "EthyleneCIElectronicDriver",
    "default_ethylene_database_path",
    "ethylene_ci_geometry",
    "ethylene_ci_protocol",
    "one_drive_data_root",
]
