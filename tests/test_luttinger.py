import numpy as np
import pytest
from scipy.special import k0

from pyqed.mps import (
    ContinuousMPS,
    CoulombLuttingerModel,
    ExponentialLuttingerModel,
    GaussianLuttingerCLETTA,
    cletta_luttinger_spectra_hierarchy,
    cmps_luttinger_density_correlation,
    cmps_luttinger_energy_shift_density,
    cmps_luttinger_parameter,
    cmps_luttinger_spectra,
    optimize_luttinger_cletta,
    pack_canonical_parameters,
)
from pyqed.mps.luttinger import (
    _luttinger_cletta_sparse_implicit_value_gradient,
)


def test_coulomb_luttinger_model_preserves_the_infrared_singularity():
    model = CoulombLuttingerModel(
        coupling=2.0,
        softening=0.5,
        fermi_velocity=1.3,
    )
    momentum = np.array([1.0e-8, 1.0e-4, 1.0])

    np.testing.assert_allclose(model.interaction_real_space(0.0), 4.0)
    np.testing.assert_allclose(
        model.interaction_momentum(momentum),
        4.0 * k0(0.5 * momentum),
    )
    assert np.all(np.diff(model.luttinger_parameter(momentum)) > 0.0)
    assert model.luttinger_parameter(0.0) == 0.0
    energy, error = model.ground_state_energy_shift_density()
    assert energy < 0.0
    assert error < 1.0e-8


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


def test_matrix_free_cletta_luttinger_spectra_match_explicit_transfer():
    base = ContinuousMPS.from_canonical_parameters(
        pack_canonical_parameters(
            [0.12],
            np.array([[0.35, -0.08], [0.11, 0.27]]),
        ),
        bond_dim=2,
    )
    state = base.cletta_memory_state(
        np.array([[[0.05, 0.02], [-0.03, 0.04]]]),
        [0.8],
        depth=1,
    )
    momentum = np.array([0.0, 0.2, 1.3])

    explicit = cmps_luttinger_spectra(state, momentum)
    hierarchy = cletta_luttinger_spectra_hierarchy(
        state,
        momentum,
        tolerance=1.0e-11,
    )

    np.testing.assert_allclose(hierarchy, explicit, rtol=2.0e-9, atol=1.0e-10)


def test_multimode_cletta_luttinger_hierarchy_matches_explicit_transfer():
    base = ContinuousMPS.from_canonical_parameters(
        pack_canonical_parameters(
            [0.08],
            np.array([[0.31, -0.05], [0.09, 0.24]]),
        ),
        bond_dim=2,
    )
    state = base.cletta_memory_state(
        np.array(
            [
                [[0.04, 0.01], [-0.02, 0.03]],
                [[-0.02, 0.03], [0.01, 0.02]],
            ]
        ),
        [0.7, 1.6],
        depth=2,
    )
    momentum = np.array([0.0, 0.15, 0.9])

    explicit = cmps_luttinger_spectra(state, momentum)
    hierarchy = cletta_luttinger_spectra_hierarchy(
        state,
        momentum,
        tolerance=1.0e-11,
    )

    np.testing.assert_allclose(hierarchy, explicit, rtol=2.0e-9, atol=1.0e-10)


def test_luttinger_cletta_implicit_gradient_matches_finite_difference():
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[1.4],
        fermi_velocity=1.0,
    )
    theta = pack_canonical_parameters(
        [0.11],
        np.array([[0.32, -0.06], [0.09, 0.25]]),
    )
    tie = np.array([[[0.04, 0.015], [-0.02, 0.03]]])
    parameters = np.concatenate([theta, tie.reshape(-1), np.log([0.8])])
    regularization = 1.0e-9
    value_gradient = _luttinger_cletta_sparse_implicit_value_gradient(
        model,
        bond_dim=2,
        num_modes=1,
        depth=1,
        reference_rates=[0.8],
        optimize_memory_rates=True,
        lower_rate=1.0e-3,
        upper_rate=1.0e3,
        quadrature_points=16,
        regularization=regularization,
        tolerance=1.0e-11,
        maxiter=500,
    )
    value, gradient = value_gradient(parameters)

    def objective(candidate):
        base = ContinuousMPS.from_canonical_parameters(candidate[:5], 2)
        state = base.cletta_memory_state(
            candidate[5:9].reshape(1, 2, 2),
            np.exp(candidate[9:10]),
            depth=1,
        )
        return cmps_luttinger_energy_shift_density(
            model,
            state,
            quadrature_points=16,
        ) + regularization * np.dot(candidate, candidate)

    step = 2.0e-6
    finite_difference = np.empty_like(parameters)
    for index in range(parameters.size):
        displacement = np.zeros_like(parameters)
        displacement[index] = step
        finite_difference[index] = (
            objective(parameters + displacement)
            - objective(parameters - displacement)
        ) / (2.0 * step)

    np.testing.assert_allclose(value, objective(parameters), atol=2.0e-12)
    np.testing.assert_allclose(
        gradient,
        finite_difference,
        rtol=2.0e-6,
        atol=2.0e-8,
    )


def test_luttinger_cletta_iterative_implicit_directional_derivative():
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[1.4],
        fermi_velocity=1.0,
    )
    theta = pack_canonical_parameters(
        [0.08],
        np.array([[0.31, -0.05], [0.09, 0.24]]),
    )
    ties = np.array(
        [
            [[0.04, 0.01], [-0.02, 0.03]],
            [[-0.02, 0.03], [0.01, 0.02]],
        ]
    )
    parameters = np.concatenate(
        [theta, ties.reshape(-1), np.log([0.7, 1.6])]
    )
    regularization = 1.0e-9
    value_gradient = _luttinger_cletta_sparse_implicit_value_gradient(
        model,
        bond_dim=2,
        num_modes=2,
        depth=2,
        reference_rates=[0.7, 1.6],
        optimize_memory_rates=True,
        lower_rate=1.0e-3,
        upper_rate=1.0e3,
        quadrature_points=16,
        regularization=regularization,
        tolerance=1.0e-10,
        maxiter=1200,
    )
    _value, gradient = value_gradient(parameters)
    direction = np.random.default_rng(5).normal(size=parameters.size)
    direction /= np.linalg.norm(direction)

    def objective(candidate):
        base = ContinuousMPS.from_canonical_parameters(candidate[:5], 2)
        state = base.cletta_memory_state(
            candidate[5:13].reshape(2, 2, 2),
            np.exp(candidate[13:]),
            depth=2,
        )
        return cmps_luttinger_energy_shift_density(
            model,
            state,
            quadrature_points=16,
            contraction_backend="hierarchy_iterative",
            iterative_tolerance=1.0e-10,
            iterative_maxiter=1200,
        ) + regularization * np.dot(candidate, candidate)

    step = 2.0e-6
    finite_difference = (
        objective(parameters + step * direction)
        - objective(parameters - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        np.dot(gradient, direction),
        finite_difference,
        rtol=3.0e-6,
        atol=3.0e-9,
    )


def test_matrix_cmps_free_density_correlation_matches_regulated_integral():
    theta = pack_canonical_parameters([], np.array([[0.7]]))
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)
    cutoff = 3.0
    distances = np.array([0.0, 0.2, 0.8])
    inverse_cutoff = 1.0 / cutoff
    expected = (
        inverse_cutoff**2 - distances**2
    ) / (
        2.0
        * np.pi**2
        * (inverse_cutoff**2 + distances**2) ** 2
    )

    actual = cmps_luttinger_density_correlation(
        state,
        distances,
        uv_cutoff=cutoff,
        points=30000,
        integration_max=24.0 * cutoff,
    )

    np.testing.assert_allclose(actual, expected, atol=2.0e-7)


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
    assert "implicit" in state.algorithm
    assert validated >= exact - 1.0e-9
    assert validated < -1.0e-3
