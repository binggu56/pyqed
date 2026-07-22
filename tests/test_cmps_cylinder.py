import numpy as np
import pytest

from pyqed.mps import (
    ContinuousMPS,
    cletta_multifield_memory_matrices,
    commuting_cylinder_parameter_size,
    cylinder_density_mode_correlation,
    cylinder_fixed_density_observables,
    cylinder_static_structure_factor,
    pack_commuting_cylinder_parameters,
    unpack_commuting_cylinder_parameters,
)
from pyqed.mps.cylinder import _cylinder_cletta_jax_value_gradient


def test_commuting_cylinder_chart_is_canonical_and_regular():
    reference = np.array(
        [
            [0.2, 0.4, 0.0],
            [-0.1, 0.3, 0.5],
            [0.0, -0.2, 0.1],
        ]
    )
    coefficients = np.array([[0.1, 0.3, -0.2], [-0.4, 0.2, 0.1]])
    theta = pack_commuting_cylinder_parameters(
        [0.1, -0.2, 0.3], reference, coefficients
    )
    q, r_ops, _a = unpack_commuting_cylinder_parameters(
        theta, 3, 3, reference_field=1
    )
    state = ContinuousMPS(q, r_ops)

    assert theta.size == commuting_cylinder_parameter_size(3, 3)
    np.testing.assert_allclose(state.left_canonical_error(), 0.0, atol=1.0e-14)
    for left in r_ops:
        for right in r_ops:
            np.testing.assert_allclose(left @ right, right @ left, atol=1.0e-14)


def test_cylinder_single_field_reduces_to_exponential_bose_gas():
    state = ContinuousMPS.random_left_canonical(2, seed=4, scale=0.15)
    rates = np.array([0.4, 1.7])
    strengths = np.array([0.8, -0.1])

    reference = state.exponential_bose_gas_fixed_density_observables(
        decay_rates=rates,
        strengths=strengths,
        density=0.7,
        connected=True,
    )
    cylinder = cylinder_fixed_density_observables(
        state,
        mode_numbers=[0],
        transverse_momenta=[0.0],
        interaction_kernels={0: (rates, strengths)},
        circumference=1.0,
        density=0.7,
        connected=True,
    )

    np.testing.assert_allclose(cylinder["energy_density"], reference["energy_density"])
    np.testing.assert_allclose(cylinder["kinetic"], reference["kinetic"])
    np.testing.assert_allclose(cylinder["interaction"], reference["interaction"])


def test_cylinder_product_state_has_zero_connected_energy():
    theta = pack_commuting_cylinder_parameters([], [[1.0]], [[0.0], [0.0]])
    q, r_ops, _a = unpack_commuting_cylinder_parameters(
        theta, 1, 3, reference_field=1
    )
    state = ContinuousMPS(q, r_ops)
    values = cylinder_fixed_density_observables(
        state,
        mode_numbers=[-1, 0, 1],
        transverse_momenta=[-0.8, 0.0, 0.8],
        interaction_kernels={
            0: ([0.3], [1.0]),
            1: ([0.7], [0.4]),
            2: ([1.1], [0.2]),
        },
        circumference=8.0,
        density=1.0,
        connected=True,
    )

    np.testing.assert_allclose(values["energy_density"], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(values["field_densities"], [0.0, 1.0, 0.0])


def test_cylinder_fourier_channels_include_orientations_and_circumference():
    amplitudes = np.array([0.5, 1.0, -0.25])
    theta = pack_commuting_cylinder_parameters(
        [], [[amplitudes[1]]], [[amplitudes[0]], [amplitudes[2]]]
    )
    q, r_ops, _a = unpack_commuting_cylinder_parameters(
        theta, 1, 3, reference_field=1
    )
    state = ContinuousMPS(q, r_ops)
    linear_density = float(np.dot(amplitudes, amplitudes))
    circumference = 4.0
    rate = 2.0
    strength = 3.0
    values = cylinder_fixed_density_observables(
        state,
        mode_numbers=[-1, 0, 1],
        transverse_momenta=[-0.8, 0.0, 0.8],
        interaction_kernels={
            0: ([rate], [strength]),
            1: ([rate], [strength]),
            2: ([rate], [strength]),
        },
        circumference=circumference,
        density=linear_density,
        connected=False,
    )

    rho_0 = float(np.dot(amplitudes, amplitudes))
    rho_1 = float(amplitudes[0] * amplitudes[1] + amplitudes[1] * amplitudes[2])
    rho_2 = float(amplitudes[0] * amplitudes[2])
    prefactor = strength / (rate * circumference)
    expected_channels = {
        0: prefactor * rho_0**2,
        1: 2.0 * prefactor * rho_1**2,
        2: 2.0 * prefactor * rho_2**2,
    }
    for transfer, expected in expected_channels.items():
        np.testing.assert_allclose(values["channel_interactions"][transfer], expected)
    np.testing.assert_allclose(values["interaction"], sum(expected_channels.values()))
    np.testing.assert_allclose(values["linear_density"], linear_density)
    np.testing.assert_allclose(values["areal_density"], linear_density / circumference)


def test_cylinder_product_state_has_no_connected_structure():
    amplitudes = np.array([0.5, 1.0, -0.25])
    theta = pack_commuting_cylinder_parameters(
        [], [[amplitudes[1]]], [[amplitudes[0]], [amplitudes[2]]]
    )
    q, r_ops, _a = unpack_commuting_cylinder_parameters(
        theta, 1, 3, reference_field=1
    )
    state = ContinuousMPS(q, r_ops)
    density = float(np.dot(amplitudes, amplitudes))
    distances = np.array([0.0, 0.4, 1.7])
    for transfer in (0, 1, 2):
        correlation = cylinder_density_mode_correlation(
            state,
            distances,
            mode_numbers=[-1, 0, 1],
            transfer=transfer,
            density=density,
            connected=True,
        )
        structure = cylinder_static_structure_factor(
            state,
            [0.0, 0.7],
            mode_numbers=[-1, 0, 1],
            transfer=transfer,
            density=density,
        )
        np.testing.assert_allclose(correlation, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(structure, 1.0, atol=1.0e-14)


def test_multifield_cletta_memory_preserves_commuting_fields():
    theta = pack_commuting_cylinder_parameters(
        [0.2],
        [[0.1, 0.3], [-0.2, 0.4]],
        [[0.2, -0.1], [-0.3, 0.15]],
    )
    q, r_ops, _a = unpack_commuting_cylinder_parameters(
        theta, 2, 3, reference_field=1
    )
    reference = r_ops[1]
    ties = np.array(
        [
            0.05 * np.eye(2) - 0.03 * reference,
            -0.02 * np.eye(2) + 0.04 * reference,
        ]
    )
    field_couplings = np.array(
        [[0.0, 1.0, 0.0], [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)]]
    )
    q_memory, r_memory = cletta_multifield_memory_matrices(
        q,
        r_ops,
        ties,
        [0.8, 1.3],
        field=1,
        field_couplings=field_couplings,
        depth=1,
    )

    state = ContinuousMPS(q_memory, r_memory)
    assert state.num_fields == 3
    assert state.bond_dim == 8
    for left in r_memory:
        for right in r_memory:
            np.testing.assert_allclose(left @ right, right @ left, atol=1.0e-14)


def test_cylinder_cletta_jax_value_and_gradient_match_numpy_and_finite_difference():
    pytest.importorskip("jax")
    modes = np.array([-1, 0, 1])
    momenta = 2.0 * np.pi * modes / 8.0
    kernels = {
        0: (np.array([0.4, 1.2]), np.array([0.7, -0.08])),
        1: (np.array([0.8]), np.array([0.2])),
        2: (np.array([1.3]), np.array([0.05])),
    }
    density = 0.8
    regularization = 1.0e-7
    gauge_penalty = 1.0e-3
    parameters = np.array([1.0, 0.2, -0.1, 0.05, np.log(0.7)])
    field_couplings = np.array([[0.0, 1.0, 0.0]])
    objective_options = dict(
        bond_dim=1,
        mode_numbers=modes,
        transverse_momenta=momenta,
        interaction_kernels=kernels,
        circumference=8.0,
        density=density,
        num_memory_modes=1,
        depth=1,
        coupled_field=1,
        field_couplings=field_couplings,
        base_size=3,
        tie_size=1,
        lower_rate=0.05,
        upper_rate=5.0,
        connected=True,
        regularization=regularization,
        density_gauge_penalty=gauge_penalty,
    )
    value_gradient = _cylinder_cletta_jax_value_gradient(
        **objective_options,
        eigensolver="dense",
        eigen_iterations=128,
        linear_solver="dense",
        linear_tolerance=1.0e-11,
        linear_maxiter=100,
    )
    value, gradient = value_gradient(parameters)

    q, r_ops, _a = unpack_commuting_cylinder_parameters(
        parameters[:3], 1, 3, reference_field=1
    )
    q_memory, r_memory = cletta_multifield_memory_matrices(
        q,
        r_ops,
        np.array([[[parameters[3]]]]),
        [np.exp(parameters[4])],
        field=1,
        field_couplings=field_couplings,
        depth=1,
    )
    state = ContinuousMPS(q_memory, r_memory)
    observables = cylinder_fixed_density_observables(
        state,
        mode_numbers=modes,
        transverse_momenta=momenta,
        interaction_kernels=kernels,
        circumference=8.0,
        density=density,
        connected=True,
        canonical=False,
    )
    expected = (
        observables["energy_density"]
        + gauge_penalty * np.log(observables["raw_density"] / density) ** 2
        + regularization * np.dot(parameters, parameters)
    )
    np.testing.assert_allclose(value, expected, rtol=2.0e-8, atol=2.0e-9)

    step = 2.0e-5
    finite_difference = np.zeros_like(parameters)
    for index in range(parameters.size):
        direction = np.zeros_like(parameters)
        direction[index] = step
        plus = value_gradient(parameters + direction)[0]
        minus = value_gradient(parameters - direction)[0]
        finite_difference[index] = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=3.0e-4, atol=2.0e-6)

    iterative_value_gradient = _cylinder_cletta_jax_value_gradient(
        **objective_options,
        eigensolver="iterative",
        eigen_iterations=512,
        linear_solver="iterative",
        linear_tolerance=1.0e-11,
        linear_maxiter=100,
    )
    iterative_value, iterative_gradient = iterative_value_gradient(parameters)
    np.testing.assert_allclose(iterative_value, value, rtol=2.0e-7, atol=2.0e-8)
    np.testing.assert_allclose(
        iterative_gradient, gradient, rtol=2.0e-4, atol=2.0e-6
    )
