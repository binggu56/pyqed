import numpy as np
import pytest

from pyqed.mps import (
    ContinuousMPS,
    ExponentialLuttingerModel,
    GaussianLuttingerCLETTA,
    cmps_luttinger_energy_shift_density,
    cmps_luttinger_parameter,
    cmps_luttinger_spectra,
    optimize_luttinger_cletta,
    pack_canonical_parameters,
)


def test_exponential_luttinger_interaction_fourier_transform():
    model = ExponentialLuttingerModel(
        decay_rates=[0.5, 2.0],
        strengths=[1.2, -0.3],
        fermi_velocity=1.4,
    )

    expected_zero = 2.0 * (1.2 / 0.5 - 0.3 / 2.0)
    np.testing.assert_allclose(model.interaction_momentum(0.0), expected_zero)
    np.testing.assert_allclose(
        model.interaction_real_space([0.0, 1.0]),
        [0.9, 1.2 * np.exp(-0.5) - 0.3 * np.exp(-2.0)],
    )


def test_exponential_luttinger_free_limit():
    velocity = 1.7
    momentum = np.array([-2.0, 0.0, 0.4, 3.0])
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[0.0],
        fermi_velocity=velocity,
    )

    np.testing.assert_allclose(model.luttinger_parameter(momentum), 1.0)
    np.testing.assert_allclose(model.mode_velocity(momentum), velocity)
    np.testing.assert_allclose(model.dispersion(momentum), velocity * np.abs(momentum))
    np.testing.assert_allclose(
        model.static_structure_factor(momentum),
        np.abs(momentum) / (2.0 * np.pi),
    )
    energy, error = model.ground_state_energy_shift_density()
    np.testing.assert_allclose(energy, 0.0, atol=1.0e-14)
    assert error <= 1.0e-14


def test_exponential_luttinger_bogoliubov_identities():
    model = ExponentialLuttingerModel(
        decay_rates=[0.7, 2.3],
        strengths=[0.8, 0.25],
        fermi_velocity=1.2,
    )
    momentum = np.linspace(0.0, 6.0, 31)
    interaction = model.interaction_momentum(momentum)
    a_value = model.fermi_velocity + interaction / (2.0 * np.pi)
    b_value = interaction / (2.0 * np.pi)
    velocity = model.mode_velocity(momentum)
    parameter = model.luttinger_parameter(momentum)

    np.testing.assert_allclose(velocity * parameter, model.fermi_velocity)
    np.testing.assert_allclose(velocity**2, a_value**2 - b_value**2)
    assert np.all(parameter < 1.0)
    energy, error = model.ground_state_energy_shift_density()
    assert energy < 0.0
    assert error < 1.0e-8


def test_exponential_luttinger_rejects_unstable_attraction():
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[-2.0],
        fermi_velocity=0.1,
    )

    with pytest.raises(ValueError, match="unstable"):
        model.mode_velocity(0.0)


def test_gaussian_cletta_zero_modes_is_free_vacuum():
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[2.0],
        fermi_velocity=1.0,
    )
    state = GaussianLuttingerCLETTA.optimize(model, num_modes=0)
    momentum = np.array([0.0, 0.5, 2.0])

    np.testing.assert_allclose(state.squeezing(momentum), 0.0)
    np.testing.assert_allclose(state.luttinger_parameter(momentum), 1.0)
    np.testing.assert_allclose(state.energy_shift_density, 0.0)
    assert state.energy_shift_density > state.exact_energy_shift_density


def test_gaussian_cletta_is_variational_and_improves_with_memory():
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[2.0],
        fermi_velocity=1.0,
    )
    one_mode = GaussianLuttingerCLETTA.optimize(
        model,
        num_modes=1,
        restarts=2,
        seed=4,
        maxiter=250,
        quadrature_points=350,
    )
    two_mode = GaussianLuttingerCLETTA.optimize(
        model,
        num_modes=2,
        seed_states=[one_mode],
        restarts=2,
        seed=5,
        maxiter=250,
        quadrature_points=350,
    )
    exact = model.ground_state_energy_shift_density()[0]

    assert one_mode.energy_shift_density >= exact - 1.0e-10
    assert two_mode.energy_shift_density >= exact - 1.0e-10
    assert two_mode.energy_shift_density <= one_mode.energy_shift_density + 1.0e-10
    assert abs(one_mode.energy_shift_density - exact) < 2.0e-5
    assert abs(two_mode.energy_shift_density - exact) < 2.0e-7


def test_matrix_cmps_coherent_state_has_only_zero_momentum_weight():
    theta = pack_canonical_parameters([], np.array([[0.7]]))
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)
    momentum = np.array([0.0, 0.2, 1.0, 4.0])
    normal, anomalous = cmps_luttinger_spectra(state, momentum)

    np.testing.assert_allclose(normal, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(anomalous, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        cmps_luttinger_parameter(state, momentum),
        1.0,
        atol=1.0e-14,
    )


def test_matrix_cletta_optimizer_carries_core_bond_dimension():
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[2.0],
        fermi_velocity=1.0,
    )
    state = optimize_luttinger_cletta(
        model,
        bond_dim=2,
        num_modes=0,
        restarts=2,
        seed=17,
        maxiter=120,
        quadrature_points=80,
    )
    exact = model.ground_state_energy_shift_density()[0]
    validated = cmps_luttinger_energy_shift_density(
        model,
        state,
        quadrature_points=160,
    )

    assert state.cletta_base.bond_dim == 2
    assert state.bond_dim == 2
    assert state.luttinger_bond_dim == 2
    assert state.luttinger_num_modes == 0
    assert validated >= exact - 1.0e-9
    assert validated < -1.0e-3
