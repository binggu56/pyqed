import numpy as np
import pytest

from pyqed.mps import (
    CMPS,
    ContinuousMPS,
    canonical_parameter_size,
    fit_exponential_kernel_nonlinear,
    fit_exponential_kernel_prony,
    pack_canonical_parameters,
    softened_yukawa_kernel,
)


def test_continuous_mps_exports_alias_and_parameter_size():
    assert CMPS is ContinuousMPS
    assert canonical_parameter_size(3) == 3 + 9
    assert canonical_parameter_size(3, num_fields=2) == 3 + 18


def test_continuous_mps_lieb_liniger_product_observables():
    density = 0.7
    coupling = 1.3
    mu = 0.4
    theta = pack_canonical_parameters([], np.array([[np.sqrt(density)]]))
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)

    values = state.lieb_liniger_observables(coupling=coupling, mu=mu)

    np.testing.assert_allclose(state.left_canonical_error(), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(values["density"], density, atol=1.0e-14)
    np.testing.assert_allclose(values["kinetic"], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(values["contact"], density * density, atol=1.0e-14)
    np.testing.assert_allclose(values["energy_density"], coupling * density * density - mu * density)

    fixed = state.lieb_liniger_fixed_density_observables(coupling=coupling, density=1.0)
    np.testing.assert_allclose(fixed["density"], 1.0)
    np.testing.assert_allclose(fixed["raw_density"], density, atol=1.0e-14)
    np.testing.assert_allclose(fixed["kinetic"], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(fixed["contact"], 1.0, atol=1.0e-14)
    np.testing.assert_allclose(fixed["energy_density"], coupling, atol=1.0e-14)


def test_continuum_prony_exponential_kernel_fit_recovers_synthetic_kernel():
    distances = 0.2 * np.arange(1, 30, dtype=float)
    rates = np.array([0.4, 1.7, 5.0])
    strengths = np.array([0.8, -0.2, 0.05])
    values = np.exp(-distances[:, None] * rates[None, :]) @ strengths

    fit = fit_exponential_kernel_prony(distances, values, rank=3)

    np.testing.assert_allclose(fit["fitted"], values, atol=1.0e-11)
    np.testing.assert_allclose(fit["decay_rates"], rates, atol=1.0e-10)
    np.testing.assert_allclose(fit["strengths"], strengths, atol=1.0e-10)
    assert fit["rel_error"] < 1.0e-11


def test_continuum_prony_softened_yukawa_kernel_fit_is_accurate():
    distances = 0.1 * np.arange(1, 121, dtype=float)
    values = softened_yukawa_kernel(distances, strength=1.0, screening=0.3, softening=0.5)

    fit = fit_exponential_kernel_prony(distances, values, rank=6)

    assert fit["decay_rates"].shape == (6,)
    assert fit["rel_error"] < 2.0e-3


def test_continuum_nonlinear_softened_coulomb_fit_includes_origin():
    distances = np.unique(
        np.concatenate(
            [np.linspace(0.0, 2.0, 101), np.geomspace(0.02, 64.0, 240)]
        )
    )
    values = softened_yukawa_kernel(
        distances,
        strength=1.0,
        screening=0.0,
        softening=0.5,
    )

    fit = fit_exponential_kernel_nonlinear(
        distances,
        values,
        rank=12,
        starts=1,
        max_nfev=1000,
    )

    assert fit["decay_rates"].shape == (12,)
    assert fit["rel_error"] < 2.0e-5
    assert fit["max_rel_error"] < 1.0e-4
    assert fit["relative_rms_error"] < 3.0e-5
    np.testing.assert_allclose(fit["fitted"][0], values[0], atol=1.0e-4)

    scaled_fit = fit_exponential_kernel_nonlinear(
        distances,
        10.0 * values,
        rank=12,
        starts=1,
        max_nfev=1000,
    )
    np.testing.assert_allclose(scaled_fit["decay_rates"], fit["decay_rates"], rtol=1.0e-7)
    np.testing.assert_allclose(scaled_fit["strengths"], 10.0 * fit["strengths"], rtol=1.0e-7)

    screening = 0.2
    screened_fit = fit_exponential_kernel_nonlinear(
        distances,
        np.exp(-screening * distances) * values,
        rank=12,
        starts=1,
        max_nfev=1000,
        rate_offset=screening,
    )
    np.testing.assert_allclose(
        screened_fit["decay_rates"],
        fit["decay_rates"] + screening,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(screened_fit["strengths"], fit["strengths"], rtol=2.0e-6)


def test_continuous_mps_exponential_bose_gas_product_observables():
    density = 0.7
    target = 0.9
    theta = pack_canonical_parameters([], np.array([[np.sqrt(density)]]))
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)
    rates = np.array([0.5, 2.0])
    strengths = np.array([1.2, 0.3])

    values = state.exponential_bose_gas_fixed_density_observables(
        decay_rates=rates,
        strengths=strengths,
        density=target,
    )
    expected_interaction = target * target * np.sum(strengths / rates)

    np.testing.assert_allclose(values["density"], target)
    np.testing.assert_allclose(values["kinetic"], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(values["interaction"], expected_interaction, atol=1.0e-13)
    np.testing.assert_allclose(values["energy_density"], expected_interaction, atol=1.0e-13)

    with_contact = state.exponential_bose_gas_fixed_density_observables(
        decay_rates=rates,
        strengths=strengths,
        density=target,
        contact_coupling=0.8,
    )
    expected_contact = target * target
    np.testing.assert_allclose(with_contact["contact"], expected_contact, atol=1.0e-13)
    np.testing.assert_allclose(
        with_contact["energy_density"],
        expected_interaction + 0.8 * expected_contact,
        atol=1.0e-13,
    )

    connected = state.exponential_bose_gas_fixed_density_observables(
        decay_rates=rates,
        strengths=strengths,
        density=target,
        connected=True,
    )
    np.testing.assert_allclose(connected["interaction"], 0.0, atol=1.0e-13)
    np.testing.assert_allclose(connected["energy_density"], 0.0, atol=1.0e-13)


def test_continuous_mps_lindblad_liouvillian_matches_transfer_matrix():
    theta = ContinuousMPS.random_canonical_parameters(3, seed=12, scale=0.2)
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=3)

    np.testing.assert_allclose(state.left_canonical_error(), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        state.lindblad_liouvillian(dense=True),
        state.transfer_matrix(),
        atol=1.0e-12,
    )

    solver = state.to_lindblad_solver()
    np.testing.assert_allclose(
        solver.liouvillian().toarray(),
        state.transfer_matrix(),
        atol=1.0e-12,
    )


def test_continuous_mps_exponential_heom_zero_tier_matches_aux_lindblad():
    theta = ContinuousMPS.random_canonical_parameters(2, seed=18, scale=0.2)
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=2)
    coupling = np.array([[0.2, 0.5], [0.5, -0.1]])

    kernel = state.heom_kernel(
        coupling=coupling,
        gamma=[1.4],
        eta=[0.3],
        depth=1,
    )

    assert kernel["n_ado"] == 2
    np.testing.assert_array_equal(kernel["keys"], np.array([[0], [1]]))
    np.testing.assert_allclose(
        kernel["zero"],
        kernel["aux"],
        atol=1.0e-12,
    )
    assert kernel["error"] < 1.0e-12
    assert kernel["L"].shape == (8, 8)
    assert np.linalg.norm(kernel["L"][:4, 4:]) > 0.0


def test_continuous_mps_heom_contract_depth_zero_matches_transfer():
    theta = ContinuousMPS.random_canonical_parameters(2, seed=20, scale=0.2)
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=2)
    coupling = np.array([[0.1, -0.3], [-0.3, 0.2]])
    r = state.r
    distances = np.array([0.0, 0.2, 0.6])

    heom = state.heom_contract(
        distances,
        coupling=coupling,
        gamma=[1.4],
        eta=[0.3],
        final_operator=r.conj().T,
        initial_operator=r,
        depth=0,
    )
    transfer = state.two_point_correlation(
        distances,
        final_operator=r.conj().T,
        initial_operator=r,
    )

    np.testing.assert_allclose(heom, transfer, atol=1.0e-10)


def test_continuous_mps_product_field_and_density_correlations():
    density = 0.7
    theta = pack_canonical_parameters([], np.array([[np.sqrt(density)]]))
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)
    distances = np.array([0.0, 0.4, 1.3])

    np.testing.assert_allclose(state.field_correlation(distances), density, atol=1.0e-14)
    np.testing.assert_allclose(state.field_correlation(0.4), density, atol=1.0e-14)
    np.testing.assert_allclose(state.field_correlation(distances, backend="lindblad"), density, atol=1.0e-14)

    np.testing.assert_allclose(state.density_correlation(distances), density * density, atol=1.0e-14)
    np.testing.assert_allclose(state.density_correlation(0.4), density * density, atol=1.0e-14)
    np.testing.assert_allclose(
        state.density_correlation(distances, connected=True),
        np.zeros_like(distances),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        state.density_correlation(distances, backend="lindblad"),
        density * density,
        atol=1.0e-14,
    )


def test_continuous_mps_anyonic_string_reduces_to_field_correlation():
    state = ContinuousMPS.from_canonical_parameters(
        ContinuousMPS.random_canonical_parameters(2, rng=np.random.default_rng(91)),
        bond_dim=2,
    )
    distances = np.array([0.0, 0.2, 0.7])

    np.testing.assert_allclose(
        state.anyonic_field_correlation(distances, statistical_angle=0.0),
        state.field_correlation(distances),
        atol=1.0e-10,
    )


def test_continuous_mps_anyonic_string_fixed_density_normalization():
    density = 0.8
    theta = pack_canonical_parameters([], np.array([[np.sqrt(0.3)]]))
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)

    values = state.anyonic_field_correlation(
        [0.0, 0.4],
        statistical_angle=0.6,
        density=density,
        normalized=True,
    )

    np.testing.assert_allclose(values[0], 1.0, atol=1.0e-13)
    np.testing.assert_allclose(values[1], np.exp(density * 0.4 * (np.exp(0.6j) - 1.0)))


def test_continuous_mps_correlation_backends_match():
    theta = ContinuousMPS.random_canonical_parameters(2, seed=40, scale=0.3)
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=2)
    distances = np.array([0.0, 0.2, 0.7])

    np.testing.assert_allclose(
        state.field_correlation(distances, backend="transfer"),
        state.field_correlation(distances, backend="lindblad"),
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        state.density_correlation(distances, backend="transfer"),
        state.density_correlation(distances, backend="lindblad"),
        atol=1.0e-7,
    )
    with pytest.raises(ValueError, match="transfer.*lindblad"):
        state.field_correlation(distances, backend="heom")


def test_continuous_mps_cletta_zero_ties_recovers_fixed_density_observables():
    state = ContinuousMPS.random_left_canonical(2, seed=41, scale=0.2)
    ties = np.zeros((2, state.bond_dim, state.bond_dim))
    rates = np.array([0.5, 2.0])

    cletta = state.cletta_lieb_liniger_fixed_density_observables(
        ties,
        rates,
        coupling=1.3,
        density=0.8,
        depth=2,
    )
    reference = state.lieb_liniger_fixed_density_observables(
        coupling=1.3,
        density=0.8,
    )

    for key in ("energy_density", "density", "kinetic", "contact"):
        np.testing.assert_allclose(cletta[key], reference[key], atol=2.0e-10)


def test_continuous_mps_cletta_aux_lindblad_state_matches_memory_state():
    state = ContinuousMPS.random_left_canonical(2, seed=45, scale=0.2)
    ties = np.array([
        [[0.02, -0.01], [0.03, 0.04]],
        [[-0.01, 0.05], [0.02, -0.03]],
    ])
    rates = np.array([0.5, 2.0])
    frequencies = np.array([0.3, -0.7])

    memory = state.cletta_memory_state(ties, rates, depth=1, frequencies=frequencies)
    aux = state.cletta_aux_lindblad_state(ties, rates, cutoff=1, frequencies=frequencies)

    np.testing.assert_allclose(aux.q, memory.q, atol=1.0e-14)
    np.testing.assert_allclose(aux.r, memory.r, atol=1.0e-14)
    np.testing.assert_allclose(aux.cletta_decay_rates, rates, atol=1.0e-14)
    np.testing.assert_allclose(aux.cletta_frequencies, frequencies, atol=1.0e-14)
    assert aux.cletta_depth == 1


def test_continuous_mps_cletta_complex_memory_connected_exponential_observable():
    state = ContinuousMPS.random_left_canonical(2, seed=46, scale=0.2)
    ties = np.array([
        [[0.02, -0.01], [0.03, 0.04]],
        [[-0.01, 0.05], [0.02, -0.03]],
        [[0.04, 0.01], [-0.02, 0.03]],
    ])
    rates = np.array([0.3, 1.1, 2.4])
    frequencies = np.array([0.2, -0.5, 0.9])
    state = state.cletta_aux_lindblad_state(ties, rates, cutoff=1, frequencies=frequencies)

    values = state.exponential_bose_gas_fixed_density_observables(
        decay_rates=[0.2, 0.8, 1.9],
        strengths=[0.4, 0.3, 0.2],
        density=1.0,
        canonical=False,
        connected=True,
    )

    assert np.isfinite(values["energy_density"])
    assert np.isfinite(values["interaction"])
    assert isinstance(values["interaction"], float)


def test_continuous_mps_cletta_zero_ties_recovers_exponential_observables():
    state = ContinuousMPS.random_left_canonical(2, seed=42, scale=0.2)
    ties = np.zeros((2, state.bond_dim, state.bond_dim))
    memory_rates = np.array([0.5, 2.0])
    interaction_rates = np.array([0.8, 2.5])
    strengths = np.array([1.1, 0.4])

    cletta = state.cletta_exponential_bose_gas_fixed_density_observables(
        ties,
        memory_rates,
        interaction_decay_rates=interaction_rates,
        strengths=strengths,
        density=0.8,
        depth=2,
    )
    reference = state.exponential_bose_gas_fixed_density_observables(
        decay_rates=interaction_rates,
        strengths=strengths,
        density=0.8,
    )

    for key in ("energy_density", "density", "kinetic", "interaction"):
        np.testing.assert_allclose(cletta[key], reference[key], atol=2.0e-10)


def test_continuous_mps_cletta_hierarchy_observables_match_explicit_pseudomode():
    state = ContinuousMPS.random_left_canonical(2, seed=47, scale=0.2)
    ties = np.array([
        [[0.02, -0.01], [0.03, 0.04]],
        [[-0.01, 0.05], [0.02, -0.03]],
    ])
    memory_rates = np.array([0.6, 1.4])
    frequencies = np.array([0.2, -0.35])
    interaction_rates = np.array([0.4, 1.3, 2.7])
    strengths = np.array([0.7, 0.2, -0.05])

    explicit = state.cletta_exponential_bose_gas_fixed_density_observables(
        ties,
        memory_rates,
        interaction_decay_rates=interaction_rates,
        strengths=strengths,
        density=0.9,
        depth=2,
        frequencies=frequencies,
        connected=True,
        contact_coupling=0.6,
        contraction_backend="explicit",
    )
    hierarchy = state.cletta_exponential_bose_gas_fixed_density_observables(
        ties,
        memory_rates,
        interaction_decay_rates=interaction_rates,
        strengths=strengths,
        density=0.9,
        depth=2,
        frequencies=frequencies,
        connected=True,
        contact_coupling=0.6,
        contraction_backend="hierarchy",
    )
    iterative = state.cletta_exponential_bose_gas_fixed_density_observables(
        ties,
        memory_rates,
        interaction_decay_rates=interaction_rates,
        strengths=strengths,
        density=0.9,
        depth=2,
        frequencies=frequencies,
        connected=True,
        contact_coupling=0.6,
        contraction_backend="hierarchy_iterative",
        iterative_tolerance=1.0e-10,
    )

    for key in (
        "energy_density",
        "density",
        "kinetic",
        "contact",
        "interaction",
        "raw_density",
        "scale",
    ):
        np.testing.assert_allclose(hierarchy[key], explicit[key], atol=2.0e-9)
        np.testing.assert_allclose(iterative[key], explicit[key], atol=2.0e-8)
    assert iterative["hierarchy_size"] == 144
    assert iterative["gmres_iterations"] > 0


def test_cletta_iterative_hierarchy_accepts_zero_tie_multimode_seed():
    state = ContinuousMPS.from_canonical_parameters(np.array([1.0]), bond_dim=1)
    ties = np.zeros((4, 1, 1))
    memory_rates = np.geomspace(0.3, 3.0, 4)
    keywords = dict(
        interaction_decay_rates=[0.4, 1.5],
        strengths=[0.7, -0.1],
        density=1.0,
        depth=1,
        connected=True,
    )

    explicit = state.cletta_exponential_bose_gas_fixed_density_observables(
        ties,
        memory_rates,
        contraction_backend="explicit",
        **keywords,
    )
    iterative = state.cletta_exponential_bose_gas_fixed_density_observables(
        ties,
        memory_rates,
        contraction_backend="hierarchy_iterative",
        iterative_tolerance=1.0e-10,
        **keywords,
    )

    for key in ("energy_density", "kinetic", "interaction", "raw_density", "scale"):
        np.testing.assert_allclose(iterative[key], explicit[key], atol=2.0e-9)
    assert iterative["hierarchy_size"] == 25


def test_cletta_iterative_hierarchy_retries_mismatched_complex_fixed_points():
    theta = np.array(
        [-1.37879245, 0.925656026, 1.27421573, -0.662880437, -0.752149559]
    )
    state = ContinuousMPS.from_canonical_parameters(theta, bond_dim=2)
    ties = 1.0e-10 * np.ones((2, 2, 2))
    keywords = dict(
        interaction_decay_rates=[0.4, 1.5],
        strengths=[0.7, 0.2],
        density=1.0,
        depth=1,
        frequencies=[3.0 * np.pi, -3.0 * np.pi],
    )

    explicit = state.cletta_exponential_bose_gas_fixed_density_observables(
        ties,
        [0.3, 0.3],
        contraction_backend="explicit",
        **keywords,
    )
    iterative = state.cletta_exponential_bose_gas_fixed_density_observables(
        ties,
        [0.3, 0.3],
        contraction_backend="hierarchy_iterative",
        iterative_tolerance=1.0e-9,
        iterative_maxiter=700,
        **keywords,
    )

    for key in ("energy_density", "kinetic", "interaction", "raw_density", "scale"):
        np.testing.assert_allclose(iterative[key], explicit[key], atol=2.0e-8)


def test_cletta_complex_pair_implicit_value_retries_invalid_fixed_point():
    theta = np.array(
        [-1.37879245, 0.925656026, 1.27421573, -0.662880437, -0.752149559]
    )
    tie = np.full((2, 2), 1.0e-10)
    parameters = np.concatenate([theta, tie.reshape(-1)])
    value_gradient = (
        ContinuousMPS._exponential_bose_gas_cletta_fixed_density_sparse_implicit_value_gradient(
            bond_dim=2,
            depth=1,
            interaction_rates=[0.4, 1.5],
            weights=[0.7, 0.2],
            target_density=1.0,
            memory_rates=[0.3, 0.3],
            memory_frequencies=[3.0 * np.pi, -3.0 * np.pi],
            contact_coupling=0.0,
            regularization=0.0,
            density_gauge_penalty=0.0,
            tolerance=1.0e-9,
            maxiter=700,
            conjugate_pair=True,
        )
    )

    value, gradient = value_gradient(parameters)
    base = ContinuousMPS.from_canonical_parameters(theta, bond_dim=2)
    expected = base.cletta_exponential_bose_gas_fixed_density_observables(
        np.stack([tie, tie]),
        [0.3, 0.3],
        interaction_decay_rates=[0.4, 1.5],
        strengths=[0.7, 0.2],
        density=1.0,
        depth=1,
        frequencies=[3.0 * np.pi, -3.0 * np.pi],
        contraction_backend="explicit",
    )

    np.testing.assert_allclose(value, expected["energy_density"], atol=2.0e-8)
    assert np.all(np.isfinite(gradient))


def test_cletta_sparse_implicit_gradient_matches_finite_difference():
    bond_dim = 2
    depth = 2
    base = ContinuousMPS.random_left_canonical(bond_dim, seed=123, scale=0.2)
    memory_rates = np.array([0.26169006338941725] * 2)
    frequencies = np.array([0.747685895398335, -0.747685895398335])
    tie = 0.02 * np.random.default_rng(5).normal(size=(bond_dim, bond_dim))
    parameters = np.concatenate([base.theta, tie.reshape(-1)])
    value_gradient = (
        ContinuousMPS._exponential_bose_gas_cletta_fixed_density_sparse_implicit_value_gradient(
            bond_dim=bond_dim,
            depth=depth,
            interaction_rates=[0.45, 2.2],
            weights=[0.8, -1.6],
            target_density=1.0,
            memory_rates=memory_rates,
            memory_frequencies=frequencies,
            contact_coupling=0.5,
            regularization=1.0e-10,
            density_gauge_penalty=1.0e-4,
            tolerance=1.0e-10,
            maxiter=500,
            conjugate_pair=True,
        )
    )
    _value, gradient = value_gradient(parameters)

    def objective(candidate):
        candidate_base = ContinuousMPS.from_canonical_parameters(
            candidate[: canonical_parameter_size(bond_dim)],
            bond_dim,
        )
        candidate_tie = candidate[canonical_parameter_size(bond_dim) :].reshape(
            bond_dim, bond_dim
        )
        values = candidate_base.cletta_exponential_bose_gas_fixed_density_observables(
            np.stack([candidate_tie, candidate_tie]),
            memory_rates,
            interaction_decay_rates=[0.45, 2.2],
            strengths=[0.8, -1.6],
            density=1.0,
            depth=depth,
            frequencies=frequencies,
            contact_coupling=0.5,
            contraction_backend="hierarchy_iterative",
            iterative_tolerance=1.0e-10,
        )
        gauge = 1.0e-4 * np.log(values["raw_density"]) ** 2
        return values["energy_density"] + gauge + 1.0e-10 * np.dot(candidate, candidate)

    step = 1.0e-5
    finite_difference = np.empty_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-8, atol=5.0e-6)


def test_cletta_multimode_connected_implicit_gradient_matches_finite_difference():
    bond_dim = 2
    depth = 1
    target_density = 0.9
    base = ContinuousMPS.random_left_canonical(bond_dim, seed=331, scale=0.15)
    memory_rates = np.array([0.35, 0.9, 1.8])
    frequencies = np.array([0.2, -0.15, 0.4])
    ties = 0.015 * np.random.default_rng(337).normal(
        size=(3, bond_dim, bond_dim)
    )
    parameters = np.concatenate([base.theta, ties.reshape(-1)])
    base_size = canonical_parameter_size(bond_dim)
    value_gradient = (
        ContinuousMPS._exponential_bose_gas_cletta_fixed_density_sparse_implicit_value_gradient(
            bond_dim=bond_dim,
            depth=depth,
            interaction_rates=[0.45, 1.4],
            weights=[0.8, -0.25],
            target_density=target_density,
            memory_rates=memory_rates,
            memory_frequencies=frequencies,
            contact_coupling=0.3,
            regularization=1.0e-10,
            density_gauge_penalty=1.0e-4,
            tolerance=1.0e-10,
            maxiter=500,
            connected=True,
        )
    )
    _value, gradient = value_gradient(parameters)

    def objective(candidate):
        candidate_base = ContinuousMPS.from_canonical_parameters(
            candidate[:base_size],
            bond_dim,
        )
        candidate_ties = candidate[base_size:].reshape(3, bond_dim, bond_dim)
        values = candidate_base.cletta_exponential_bose_gas_fixed_density_observables(
            candidate_ties,
            memory_rates,
            interaction_decay_rates=[0.45, 1.4],
            strengths=[0.8, -0.25],
            density=target_density,
            depth=depth,
            frequencies=frequencies,
            connected=True,
            contact_coupling=0.3,
            contraction_backend="hierarchy_iterative",
            iterative_tolerance=1.0e-10,
        )
        gauge = 1.0e-4 * np.log(values["raw_density"] / target_density) ** 2
        return values["energy_density"] + gauge + 1.0e-10 * np.dot(candidate, candidate)

    step = 1.0e-5
    finite_difference = np.empty_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-7, atol=8.0e-6)


def test_cletta_multimode_fixed_poles_select_implicit_optimizer():
    state = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
        bond_dim=1,
        interaction_decay_rates=[0.4, 1.3],
        strengths=[0.7, -0.1],
        density=1.0,
        num_modes=3,
        depth=1,
        memory_decay_rates=[0.3, 0.8, 2.0],
        memory_frequencies=[0.0, 0.2, -0.1],
        optimize_memory_rates=False,
        optimize_memory_frequencies=False,
        connected=True,
        contraction_backend="hierarchy_iterative",
        restarts=1,
        seed=347,
        maxiter=2,
        use_jax=False,
    )

    assert np.isfinite(state.energy)
    assert "implicit-heom" in state.algorithm


def test_cletta_large_hierarchy_adjoint_directional_derivative():
    bond_dim = 1
    num_modes = 8
    depth = 2
    target_density = 0.9
    base_size = canonical_parameter_size(bond_dim)
    memory_rates = np.geomspace(0.3, 2.5, num_modes)
    frequencies = np.zeros(num_modes)
    rng = np.random.default_rng(353)
    parameters = np.concatenate(
        [np.array([0.85]), 0.01 * rng.normal(size=num_modes)]
    )
    direction = rng.normal(size=parameters.size)
    direction /= np.linalg.norm(direction)
    value_gradient = (
        ContinuousMPS._exponential_bose_gas_cletta_fixed_density_sparse_implicit_value_gradient(
            bond_dim=bond_dim,
            depth=depth,
            interaction_rates=[0.7],
            weights=[0.5],
            target_density=target_density,
            memory_rates=memory_rates,
            memory_frequencies=frequencies,
            contact_coupling=0.2,
            regularization=1.0e-10,
            density_gauge_penalty=1.0e-4,
            tolerance=1.0e-9,
            maxiter=400,
            connected=True,
        )
    )
    _value, gradient = value_gradient(parameters)

    def objective(candidate):
        candidate_base = ContinuousMPS.from_canonical_parameters(
            candidate[:base_size],
            bond_dim,
        )
        candidate_ties = candidate[base_size:].reshape(
            num_modes,
            bond_dim,
            bond_dim,
        )
        values = candidate_base.cletta_exponential_bose_gas_fixed_density_observables(
            candidate_ties,
            memory_rates,
            interaction_decay_rates=[0.7],
            strengths=[0.5],
            density=target_density,
            depth=depth,
            frequencies=frequencies,
            connected=True,
            contact_coupling=0.2,
            contraction_backend="hierarchy_iterative",
            iterative_tolerance=1.0e-9,
            iterative_maxiter=400,
        )
        gauge = 1.0e-4 * np.log(values["raw_density"] / target_density) ** 2
        return values["energy_density"] + gauge + 1.0e-10 * np.dot(candidate, candidate)

    step = 2.0e-5
    finite_difference = (
        objective(parameters + step * direction)
        - objective(parameters - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        np.dot(gradient, direction),
        finite_difference,
        rtol=2.0e-4,
        atol=2.0e-5,
    )


def test_continuous_mps_fixed_density_optimizer_smoke():
    state = ContinuousMPS.optimize_lieb_liniger_fixed_density(
        bond_dim=1,
        coupling=2.0,
        density=1.0,
        restarts=1,
        seed=19,
        maxiter=40,
        use_jax=False,
    )

    np.testing.assert_allclose(state.energy, 2.0, atol=1.0e-10)
    np.testing.assert_allclose(state.density, 1.0)
    assert state.raw_density > 0.0


def test_continuous_mps_cletta_fixed_density_optimizer_depth_zero_smoke():
    state = ContinuousMPS.optimize_lieb_liniger_cletta_fixed_density(
        bond_dim=1,
        coupling=2.0,
        density=1.0,
        num_modes=1,
        depth=0,
        restarts=1,
        seed=43,
        maxiter=5,
        optimize_rates=False,
    )

    np.testing.assert_allclose(state.energy, 2.0, atol=1.0e-10)
    np.testing.assert_allclose(state.density, 1.0)
    assert state.raw_density > 0.0
    assert state.cletta_base is not None
    assert state.cletta_depth == 0


def test_continuous_mps_exponential_cletta_optimizer_depth_zero_smoke():
    state = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
        bond_dim=1,
        interaction_decay_rates=[0.5, 2.0],
        strengths=[1.2, 0.3],
        density=1.0,
        num_modes=1,
        depth=0,
        restarts=1,
        seed=44,
        maxiter=5,
        optimize_memory_rates=False,
        optimize_memory_frequencies=False,
        contact_coupling=0.7,
    )

    expected = 1.2 / 0.5 + 0.3 / 2.0 + 0.7
    np.testing.assert_allclose(state.energy, expected, atol=1.0e-10)
    np.testing.assert_allclose(state.density, 1.0)
    np.testing.assert_allclose(state.contact, 1.0, atol=1.0e-10)
    np.testing.assert_allclose(state.interaction, expected - 0.7, atol=1.0e-10)
    assert state.raw_density > 0.0


def test_continuous_mps_exponential_cletta_iterative_optimizer_depth_zero_smoke():
    state = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
        bond_dim=1,
        interaction_decay_rates=[0.5],
        strengths=[1.0],
        density=1.0,
        num_modes=2,
        depth=0,
        memory_decay_rates=[1.0, 1.0],
        memory_frequencies=[0.2, -0.2],
        optimize_memory_rates=False,
        optimize_memory_frequencies=False,
        conjugate_pair=True,
        contraction_backend="hierarchy_iterative",
        restarts=1,
        maxiter=2,
        use_jax=False,
    )

    np.testing.assert_allclose(state.energy, 2.0, atol=1.0e-10)
    np.testing.assert_allclose(state.density, 1.0)
    assert state.raw_density > 0.0
    assert "finite-diff" in state.algorithm


def test_continuous_mps_exponential_cletta_conjugate_pair_constraint():
    state = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
        bond_dim=1,
        interaction_decay_rates=[0.5, 2.0],
        strengths=[1.2, 0.3],
        density=1.0,
        num_modes=2,
        depth=1,
        memory_decay_rates=[0.7, 1.3],
        memory_frequencies=[0.4, -0.2],
        conjugate_pair=True,
        restarts=1,
        seed=45,
        maxiter=2,
    )

    np.testing.assert_allclose(
        state.cletta_decay_rates[0],
        state.cletta_decay_rates[1],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        state.cletta_frequencies[0],
        -state.cletta_frequencies[1],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        state.cletta_tie_matrices[0],
        state.cletta_tie_matrices[1],
        atol=1.0e-14,
    )
    assert state.cletta_parameters.size == 4
    assert "conjugate-pair" in state.algorithm


def test_continuous_mps_exponential_cletta_conjugate_pair_requires_two_modes():
    with pytest.raises(ValueError, match="num_modes=2"):
        ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
            bond_dim=1,
            interaction_decay_rates=[1.0],
            num_modes=1,
            conjugate_pair=True,
            restarts=1,
            maxiter=1,
        )
