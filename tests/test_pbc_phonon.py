import numpy as np
import pytest

from pyqed.pbc import (
    FiniteDisplacementPhonon,
    KRHFForceCalculator,
    PeriodicPhononMode,
    interpolate_q_path,
)
from pyqed.qchem.pbc import Cell
from pyqed.units import amu_to_au


class _PeriodicChainForces:
    def __init__(self, lattice_constant, ncell, spring_constant):
        self.equilibrium = np.zeros((ncell, 3), dtype=float)
        self.equilibrium[:, 0] = np.arange(ncell) * lattice_constant
        self.spring_constant = float(spring_constant)
        self.calls = 0

    def forces(self, symbols, positions, lattice):
        del symbols, lattice
        self.calls += 1
        displacement = np.asarray(positions) - self.equilibrium
        return -self.spring_constant * (
            2.0 * displacement
            - np.roll(displacement, 1, axis=0)
            - np.roll(displacement, -1, axis=0)
        )


def _helium_chain_cell():
    return Cell(
        atom="He 0 0 0",
        a=np.diag([2.0, 7.0, 7.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()


def test_finite_displacement_phonon_recovers_periodic_chain_dispersion():
    spring_constant = 0.3
    calculator = _PeriodicChainForces(2.0, 3, spring_constant)
    phonon = FiniteDisplacementPhonon(
        _helium_chain_cell(),
        calculator,
        supercell=(3, 1, 1),
        displacement=1.0e-3,
        masses=[1.0],
    ).run()

    assert phonon.success
    assert calculator.calls == 6
    assert phonon.force_constants.shape == (1, 3, 3, 3)
    assert phonon.acoustic_sum_rule_residual < 1.0e-14

    gamma = phonon.frequencies([0.0, 0.0, 0.0], units="au")
    np.testing.assert_allclose(gamma, 0.0, atol=1.0e-10)

    quarter = phonon.frequencies([0.25, 0.0, 0.0], units="au")
    expected = np.sqrt(2.0 * spring_constant / amu_to_au)
    np.testing.assert_allclose(quarter, expected, atol=1.0e-12)


def test_phonon_band_structure_interpolates_fractional_q_path():
    calculator = _PeriodicChainForces(2.0, 3, 0.2)
    phonon = FiniteDisplacementPhonon(
        _helium_chain_cell(),
        calculator,
        supercell=(3, 1, 1),
        displacement=1.0e-3,
    ).run()
    vertices = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])

    bands = phonon.band_structure(
        vertices,
        labels=("Gamma", "X", "Gamma"),
        points_per_segment=5,
    )

    assert bands["qpoints"].shape == (9, 3)
    assert bands["frequencies"].shape == (9, 3)
    np.testing.assert_array_equal(bands["ticks"], [0, 4, 8])
    assert np.all(np.diff(bands["distances"]) > 0.0)
    np.testing.assert_allclose(bands["frequencies"][0], 0.0, atol=1.0e-5)
    np.testing.assert_allclose(bands["frequencies"][-1], 0.0, atol=1.0e-5)


def test_finite_displacement_phonon_returns_mass_weighted_mode():
    spring_constant = 0.3
    phonon = FiniteDisplacementPhonon(
        _helium_chain_cell(),
        _PeriodicChainForces(2.0, 3, spring_constant),
        supercell=(3, 1, 1),
        displacement=1.0e-3,
        masses=[1.0],
    ).run()

    mode = phonon.mode([0.25, 0.0, 0.0], branch=1)
    expected_frequency = np.sqrt(2.0 * spring_constant / amu_to_au)

    assert isinstance(mode, PeriodicPhononMode)
    assert mode.branch == 1
    assert mode.stable
    assert mode.source == "FiniteDisplacementPhonon"
    np.testing.assert_allclose(mode.qpoint, [0.25, 0.0, 0.0])
    np.testing.assert_allclose(mode.frequency, expected_frequency, atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.norm(mode.eigenvector), 1.0)
    np.testing.assert_allclose(
        mode.cartesian_displacement,
        mode.eigenvector / np.sqrt(amu_to_au),
    )


def test_interpolate_q_path_validates_shape_and_resolution():
    lattice = np.eye(3)
    with pytest.raises(ValueError, match="vertices"):
        interpolate_q_path([[0.0, 0.0, 0.0]], lattice)
    with pytest.raises(ValueError, match="points_per_segment"):
        interpolate_q_path([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], lattice, 1)


def test_native_krhf_force_calculator_runs_without_external_backend():
    calculator = KRHFForceCalculator(
        "sto-3g",
        scf_options={
            "eta": 0.7,
            "real_cut": 0,
            "pair_cut": 0,
            "recip_cut": 2,
            "one_body_nuclear_cut": 1,
            "eri_screen_tol": 0.0,
            "pair_ft_screen_tol": 0.0,
            "one_body_screen_tol": 0.0,
        },
        run_options={
            "max_cycle": 60,
            "conv_tol": 1.0e-11,
            "conv_tol_dm": 1.0e-9,
        },
    )
    forces = calculator.forces(
        ("H", "H"),
        np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]]),
        np.eye(3) * 6.0,
    )

    assert calculator.converged
    assert calculator.mean_field.jk_builder == "reciprocal"
    assert calculator.history[-1]["seconds"] > 0.0
    np.testing.assert_allclose(np.sum(forces, axis=0), 0.0, atol=2.0e-8)
    assert abs(forces[0, 0]) > 1.0e-3
    np.testing.assert_allclose(forces[:, 1:], 0.0, atol=2.0e-8)


def test_native_krhf_force_calculator_accepts_explicit_gth_pseudo():
    pseudo = {
        "H": [[1], 0.35, 2, [-3.2, 0.45], 1, [0.28, 1, [[1.7]]]],
    }
    calculator = KRHFForceCalculator(
        "sto-3g",
        pseudo=pseudo,
        scf_options={
            "eta": 0.7,
            "real_cut": 0,
            "pair_cut": 2,
            "recip_cut": 2,
            "pseudo_cut": 0,
            "one_body_nuclear_cut": 1,
            "eri_screen_tol": 0.0,
            "pair_ft_screen_tol": 0.0,
            "pseudo_local_screen_tol": 0.0,
            "one_body_screen_tol": 0.0,
        },
        run_options={
            "max_cycle": 80,
            "conv_tol": 1.0e-11,
            "conv_tol_dm": 1.0e-9,
        },
    )
    forces = calculator.forces(
        ("H", "H"),
        np.asarray([[2.1, 3.0, 3.1], [3.65, 3.25, 3.3]]),
        np.diag([6.0, 6.4, 6.8]),
    )

    assert calculator.mean_field.cell.has_pseudo
    np.testing.assert_allclose(np.sum(forces, axis=0), 0.0, atol=2.0e-12)
    assert np.max(np.abs(forces)) > 1.0e-3
