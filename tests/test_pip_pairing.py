import numpy as np
from scipy.linalg import expm

from pyqed.mps.pip_pairing import (
    ContinuumPipPairingModel,
    ThermodynamicPipBCS,
    ThermodynamicPipCLETTA,
    ExactOnePairPipState,
    OneScalePipCLETTA,
    TwoPairPipCLETTA,
    TwoPairPipD3CLETTA,
)


def test_fixed_density_thermodynamic_state_solves_direct_continuum_equations():
    model = ContinuumPipPairingModel(coupling=3.0)
    state = ThermodynamicPipBCS.solve(model, fermion_density=0.5)

    assert state.bond_dim == 1
    assert state.num_memory_modes == 0
    assert state.pair_filling == 0.25
    assert state.phase == "weak pairing"
    np.testing.assert_allclose(state.chemical_potential, 0.07920853, atol=2.0e-8)
    np.testing.assert_allclose(state.gap, 0.76726715, atol=2.0e-8)
    np.testing.assert_allclose(state.energy_density, -0.0209676841, atol=2.0e-10)
    np.testing.assert_allclose(state.integrated_fermion_density(), 0.5, atol=2.0e-10)
    np.testing.assert_allclose(state.gap_equation_residual(), 0.0, atol=2.0e-10)


def test_fixed_density_state_uses_hard_core_local_pair_space():
    state = ThermodynamicPipBCS.solve(
        ContinuumPipPairingModel(coupling=3.0),
        fermion_density=0.5,
    )
    energies = np.linspace(0.0, state.model.energy_cutoff, 17)
    amplitudes = state.local_hard_core_amplitudes(energies, theta=0.37)

    assert amplitudes.shape == (17, 2)
    np.testing.assert_allclose(
        np.sum(abs(amplitudes) ** 2, axis=-1),
        1.0,
        atol=2.0e-14,
    )
    assert np.all(state.pair_occupation(energies) >= 0.0)
    assert np.all(state.pair_occupation(energies) <= 1.0)


def test_fixed_density_dilute_limit_matches_one_pair_binding_scale():
    model = ContinuumPipPairingModel(coupling=3.0)
    state = ThermodynamicPipBCS.solve(model, fermion_density=1.0e-5)

    np.testing.assert_allclose(
        state.chemical_potential,
        -0.5 * model.binding_energy(),
        atol=2.0e-5,
    )


def test_real_cletta_has_hard_core_physical_leg_and_explicit_memory():
    reference = ThermodynamicPipBCS.solve(
        ContinuumPipPairingModel(coupling=3.0),
        fermion_density=0.5,
    )
    state = ThermodynamicPipCLETTA.evaluate(
        reference,
        radial_decay=2.0,
        tie_strength=0.4,
        memory_decay=1.3,
        quadrature_points=64,
    )
    q_matrix, r_matrix = state.combined_matrices()
    tensors = state.hard_core_tensors(0.37, 0.01)
    rotated = state.hard_core_tensors(0.37, 0.01, theta=0.41)

    assert state.bond_dim == 2
    assert state.num_memory_modes == 1
    assert state.memory_depth == 1
    assert state.effective_bond_dim == 4
    assert q_matrix.shape == (4, 4)
    assert r_matrix.shape == (4, 4)
    assert tensors.shape == (2, 4, 4)
    assert np.linalg.norm(r_matrix) > 0.0
    np.testing.assert_allclose(rotated[0], tensors[0], atol=0.0)
    np.testing.assert_allclose(
        rotated[1],
        np.exp(0.41j) * tensors[1],
        atol=2.0e-15,
    )
    np.testing.assert_allclose(state.fermion_density, 0.5, atol=3.0e-10)


def test_real_cletta_nonzero_tie_correction_vanishes_in_continuum():
    reference = ThermodynamicPipBCS.solve(
        ContinuumPipPairingModel(coupling=3.0),
        fermion_density=0.5,
    )
    coarse = ThermodynamicPipCLETTA.evaluate(
        reference,
        radial_decay=2.0,
        tie_strength=0.4,
        memory_decay=1.3,
        quadrature_points=64,
    )
    fine = ThermodynamicPipCLETTA.evaluate(
        reference,
        radial_decay=2.0,
        tie_strength=0.4,
        memory_decay=1.3,
        quadrature_points=128,
    )

    coarse_error = coarse.energy_density - reference.energy_density
    fine_error = fine.energy_density - reference.energy_density
    assert coarse_error > 0.0
    assert fine_error > 0.0
    assert fine_error < 0.51 * coarse_error
    assert abs(fine.pairing_amplitude_density) < (
        reference.gap / reference.model.coupling
    )


def test_real_cletta_optimization_collapses_to_exact_bcs_thermodynamic_energy():
    reference = ThermodynamicPipBCS.solve(
        ContinuumPipPairingModel(coupling=3.0),
        fermion_density=0.5,
    )
    state = ThermodynamicPipCLETTA.optimize(
        reference,
        quadrature_points=32,
        validation_points=96,
    )

    np.testing.assert_allclose(state.tie_strength, 0.0, atol=0.0)
    np.testing.assert_allclose(
        state.energy_density,
        reference.energy_density,
        atol=3.0e-10,
    )


def test_continuum_pip_bound_state_satisfies_analytic_equation():
    model = ContinuumPipPairingModel(coupling=3.0)
    binding = model.binding_energy()

    assert binding > 0.0
    np.testing.assert_allclose(
        model.pair_susceptibility(binding),
        1.0 / model.coupling,
        atol=1.0e-13,
    )


def test_exact_pair_reference_is_normalized_and_has_exact_energy():
    state = ExactOnePairPipState.from_model(
        ContinuumPipPairingModel(coupling=3.0)
    )

    assert state.bond_dim == 2
    np.testing.assert_allclose(state.norm(), 1.0, atol=2.0e-12)
    np.testing.assert_allclose(
        state.energy_expectation(),
        state.energy,
        atol=2.0e-11,
    )


def test_d2_m1_variational_state_is_restricted_and_variational():
    model = ContinuumPipPairingModel(coupling=3.0)
    reference = ExactOnePairPipState.from_model(model)
    state = OneScalePipCLETTA.optimize(model)

    assert state.bond_dim == 2
    assert state.num_tie_channels == 1
    assert state.num_memory_scales == 1
    np.testing.assert_allclose(state.norm(), 1.0, atol=2.0e-12)
    assert state.energy > reference.energy
    assert state.energy - reference.energy > 1.0e-4
    np.testing.assert_allclose(state.energy, -0.341753466664, atol=2.0e-11)


def test_d2_outer_insertion_selects_exactly_one_pair():
    state = OneScalePipCLETTA.optimize(
        ContinuumPipPairingModel(coupling=3.0)
    )
    _, insertion = state.outer_matrices()
    boundary_right = np.array([1.0, 0.0])
    boundary_left = np.array([0.0, 1.0])

    assert boundary_left @ boundary_right == 0.0
    assert boundary_left @ insertion @ boundary_right != 0.0
    np.testing.assert_allclose(insertion @ insertion, 0.0, atol=0.0)


def test_d2_transfer_contraction_generates_restricted_radial_amplitude():
    state = OneScalePipCLETTA.optimize(
        ContinuumPipPairingModel(coupling=3.0)
    )
    energy = 0.37
    q_matrix, insertion = state.outer_matrices()
    boundary_right = np.array([1.0, 0.0])
    boundary_left = np.array([0.0, 1.0])
    transfer_amplitude = (
        state.model.form_factor(energy)
        * boundary_left
        @ expm(q_matrix * (state.model.energy_cutoff - energy))
        @ insertion
        @ expm(q_matrix * energy)
        @ boundary_right
    )

    np.testing.assert_allclose(
        transfer_amplitude,
        state.radial_amplitude(energy),
        atol=1.0e-13,
    )


def test_pip_angular_tie_has_unit_winding():
    state = OneScalePipCLETTA.optimize(
        ContinuumPipPairingModel(coupling=3.0)
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 17)
    amplitude = state.angular_amplitude(angles)

    np.testing.assert_allclose(np.abs(amplitude), 1.0, atol=1.0e-14)
    unwrapped = np.unwrap(np.angle(amplitude))
    np.testing.assert_allclose(unwrapped[-1] - unwrapped[0], 2.0 * np.pi)


def test_cutoff_p_wave_pair_has_binding_threshold():
    model = ContinuumPipPairingModel(coupling=1.9)

    with np.testing.assert_raises_regex(ValueError, "binds only"):
        model.binding_energy()


def test_genuine_d2_m1_cletta_contracts_open_propagate_close_amplitude():
    state = TwoPairPipCLETTA.optimize(
        ContinuumPipPairingModel(coupling=3.0),
        quadrature_points=48,
        validation_points=96,
    )
    q_matrix, r_matrix = state.combined_matrices()

    assert state.bond_dim == 2
    assert state.num_memory_modes == 1
    assert state.memory_depth == 1
    assert q_matrix.shape == (4, 4)
    assert r_matrix.shape == (4, 4)
    np.testing.assert_allclose(
        state.contracted_ordered_amplitude(0.2, 0.7),
        state.ordered_amplitude(0.2, 0.7),
        atol=2.0e-13,
    )


def test_genuine_d2_m1_boundaries_select_exactly_two_pair_insertions():
    state = TwoPairPipCLETTA.optimize(
        ContinuumPipPairingModel(coupling=3.0),
        quadrature_points=32,
        validation_points=48,
    )
    _, insertion = state.combined_matrices()
    left, right = state.boundary_vectors()

    np.testing.assert_allclose(left @ right, 0.0, atol=0.0)
    np.testing.assert_allclose(left @ insertion @ right, 0.0, atol=0.0)
    assert abs(left @ np.linalg.matrix_power(insertion, 2) @ right) > 0.0
    np.testing.assert_allclose(
        left @ np.linalg.matrix_power(insertion, 3) @ right,
        0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        left @ np.linalg.matrix_power(insertion, 4) @ right,
        0.0,
        atol=0.0,
    )


def test_genuine_d2_m1_two_pair_energy_is_variational():
    state = TwoPairPipCLETTA.optimize(
        ContinuumPipPairingModel(coupling=3.0),
        quadrature_points=64,
        validation_points=128,
    )

    assert state.energy > state.exact_dilute_pair_energy
    assert state.energy - state.exact_dilute_pair_energy > 1.0e-3
    np.testing.assert_allclose(state.radial_decay, 3.09807, atol=2.0e-4)
    np.testing.assert_allclose(state.memory_decay, 1.38747, atol=2.0e-4)
    np.testing.assert_allclose(state.energy, -0.68577307, atol=3.0e-7)


def test_d3_m1_enlarges_virtual_space_at_fixed_two_pair_number():
    state = TwoPairPipD3CLETTA.optimize(
        ContinuumPipPairingModel(coupling=3.0),
        quadrature_points=48,
        validation_points=96,
    )
    q_matrix, r_matrix = state.combined_matrices()
    left, right = state.boundary_vectors()

    assert state.bond_dim == 3
    assert state.num_memory_modes == 1
    assert q_matrix.shape == (6, 6)
    assert r_matrix.shape == (6, 6)
    np.testing.assert_allclose(left @ right, 0.0, atol=0.0)
    np.testing.assert_allclose(left @ r_matrix @ right, 0.0, atol=0.0)
    assert abs(left @ np.linalg.matrix_power(r_matrix, 2) @ right) > 0.0
    np.testing.assert_allclose(
        left @ np.linalg.matrix_power(r_matrix, 3) @ right,
        0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        state.contracted_ordered_amplitude(0.2, 0.7),
        state.ordered_amplitude(0.2, 0.7),
        atol=3.0e-13,
    )


def test_d3_m1_improves_d2_at_fixed_two_pair_number():
    model = ContinuumPipPairingModel(coupling=3.0)
    d2 = TwoPairPipCLETTA.optimize(
        model,
        quadrature_points=64,
        validation_points=128,
    )
    d3 = TwoPairPipD3CLETTA.optimize(
        model,
        quadrature_points=64,
        validation_points=128,
    )

    assert d3.energy < d2.energy
    assert d3.energy > d3.exact_dilute_pair_energy
    assert d2.energy - d3.energy > 5.0e-3
