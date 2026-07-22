import numpy as np

from pyqed.letta import FrontierTiedLETTA, LocalHamiltonian, LocalTerm
from pyqed.letta.vmc import (
    LETTAVMC,
    LETTAWavefunction,
    MetropolisDiagnostics,
    SRProposal,
    VMCSamples,
)


def _complex_frontier_state(seed=4):
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
    exchange = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    hamiltonian = LocalHamiltonian(
        (2, 2, 2),
        (
            LocalTerm((0,), 0.17 * sy),
            LocalTerm((0, 1), exchange),
            LocalTerm((1, 2), 0.7 * exchange),
            LocalTerm((0, 2), -0.11 * np.kron(sz, sz)),
        ),
        constant=0.03,
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((2,), (2,), ()),
        bond_dim=2,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 91)
    state.tensors = [
        tensor + 0.08j * rng.normal(size=tensor.shape) for tensor in state.tensors
    ]
    return state, hamiltonian


def test_vmc_amplitudes_and_local_energies_match_exact_state():
    state, hamiltonian = _complex_frontier_state()
    vmc = LETTAVMC(state, hamiltonian, seed=12)
    configurations = np.asarray(list(np.ndindex(*hamiltonian.dims)), dtype=np.intp)
    vector = state.state_vector()

    np.testing.assert_allclose(vmc.amplitudes(configurations), vector, atol=2.0e-14)
    local_energies = np.asarray(
        [vmc.local_energy(configuration) for configuration in configurations]
    )
    probabilities = np.abs(vector) ** 2 / np.vdot(vector, vector).real
    exact_energy = np.vdot(vector, hamiltonian @ vector) / np.vdot(vector, vector)
    sampled_energy = probabilities @ local_energies
    np.testing.assert_allclose(sampled_energy, exact_energy, atol=3.0e-14)
    np.testing.assert_allclose(sampled_energy, state.expectation(), atol=3.0e-14)

    exact_variance = (
        np.vdot(hamiltonian @ vector, hamiltonian @ vector) / np.vdot(vector, vector)
        - abs(exact_energy) ** 2
    ).real
    local_variance = probabilities @ np.abs(local_energies - exact_energy) ** 2
    np.testing.assert_allclose(local_variance, exact_variance, atol=4.0e-14)


def test_product_cache_updates_every_tensor_affected_by_a_tied_spin():
    state, _hamiltonian = _complex_frontier_state(seed=8)
    wavefunction = LETTAWavefunction.from_state(state)
    assert wavefunction.dependent_tensors[2] == (0, 1, 2)
    configuration = np.array([1, 0, 0])
    cache = wavefunction.product_cache(configuration)
    proposed = configuration.copy()
    proposed[2] = 1

    np.testing.assert_allclose(
        cache.amplitude_after_local_update(2, 1),
        wavefunction.amplitude(proposed),
        atol=2.0e-14,
    )
    accepted = cache.accept_local_update(2, 1)
    np.testing.assert_array_equal(cache.configuration, proposed)
    np.testing.assert_allclose(accepted, wavefunction.amplitude(proposed), atol=2.0e-14)

    # A two-spin Hamiltonian action can change several physical variables at once.
    target = np.array([0, 1, 0])
    np.testing.assert_allclose(
        cache.amplitude_for(target), wavefunction.amplitude(target), atol=2.0e-14
    )
    accepted = cache.accept_configuration(target)
    np.testing.assert_array_equal(cache.configuration, target)
    np.testing.assert_allclose(accepted, wavefunction.amplitude(target), atol=2.0e-14)


def test_complex_log_derivative_matches_finite_difference():
    state, hamiltonian = _complex_frontier_state(seed=3)
    vmc = LETTAVMC(state, hamiltonian, seed=7)
    configuration = np.array([1, 0, 1])
    amplitude = vmc.amplitude(configuration)
    derivative = vmc.log_derivative(configuration)
    active = np.flatnonzero(np.abs(derivative) > 0.0)
    parameters = vmc.wavefunction.parameter_vector()
    epsilon = 2.0e-7

    for index in active[:: max(1, len(active) // 4)]:
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        vmc.wavefunction.set_parameter_vector(plus)
        amplitude_plus = vmc.amplitude(configuration)
        vmc.wavefunction.set_parameter_vector(minus)
        amplitude_minus = vmc.amplitude(configuration)
        numerical = (amplitude_plus - amplitude_minus) / (2.0 * epsilon * amplitude)
        np.testing.assert_allclose(
            numerical, derivative[index], rtol=2.0e-7, atol=2.0e-8
        )
    vmc.wavefunction.set_parameter_vector(parameters)
    assert np.issubdtype(vmc.wavefunction.dtype, np.complexfloating)


def test_integer_input_tensors_are_promoted_for_log_derivatives():
    tensor = np.array([[[2, 1]]], dtype=np.int64)
    hamiltonian = LocalHamiltonian((2,), ())
    vmc = LETTAVMC(
        (tensor,),
        hamiltonian,
        physical_sites=((0,),),
        initial_configuration=np.array([0]),
    )

    assert np.issubdtype(vmc.wavefunction.dtype, np.floating)
    np.testing.assert_allclose(vmc.log_derivative(np.array([0])), [0.5, 0.0])


def test_metropolis_sampling_is_reproducible_and_reports_diagnostics():
    state, hamiltonian = _complex_frontier_state(seed=9)
    kwargs = dict(
        seed=31,
        initial_configuration=np.array([0, 0, 0]),
    )
    first = LETTAVMC(state, hamiltonian, **kwargs)
    second = LETTAVMC(state, hamiltonian, **kwargs)
    samples_a = first.sample(
        40, burn_in=7, sweeps_between=2, include_log_derivatives=True
    )
    samples_b = second.sample(
        40, burn_in=7, sweeps_between=2, include_log_derivatives=True
    )

    np.testing.assert_array_equal(samples_a.configurations, samples_b.configurations)
    np.testing.assert_allclose(samples_a.local_energies, samples_b.local_energies)
    np.testing.assert_allclose(samples_a.log_derivatives, samples_b.log_derivatives)
    assert samples_a.diagnostics == samples_b.diagnostics
    assert samples_a.diagnostics.attempts == (7 + 2 * 40) * 3
    assert 0.0 <= samples_a.diagnostics.acceptance_rate <= 1.0
    assert sum(samples_a.diagnostics.site_attempts) == samples_a.diagnostics.attempts
    assert samples_a.diagnostics.single_site_attempts == samples_a.diagnostics.attempts
    assert samples_a.diagnostics.exchange_attempts == 0


def test_exchange_proposals_sample_a_fixed_magnetization_sector():
    # This two-site MPS has support only on |01> and |10>.  Single flips leave
    # the sector and are rejected, whereas an exchange connects its two states.
    tensors = (
        np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        np.array([[[0.0, 1.0]], [[1.0, 0.0]]]),
    )
    hamiltonian = LocalHamiltonian((2, 2), ())
    initial = np.array([0, 1])

    single = LETTAVMC(
        tensors,
        hamiltonian,
        physical_sites=((0,), (1,)),
        seed=5,
        initial_configuration=initial,
    )
    single_samples = single.sample(4, burn_in=1, sweeps_between=1)
    assert single_samples.diagnostics.accepted == 0
    assert single_samples.diagnostics.zero_amplitude_rejections == 10
    np.testing.assert_array_equal(single_samples.configurations, np.tile(initial, (4, 1)))

    exchange = LETTAVMC(
        tensors,
        hamiltonian,
        physical_sites=((0,), (1,)),
        seed=5,
        initial_configuration=initial,
        proposal="exchange",
    )
    assert exchange.sampler.step(pair=(0, 1))
    np.testing.assert_array_equal(exchange.sampler.cache.configuration, [1, 0])
    exchange_samples = exchange.sample(4, burn_in=1, sweeps_between=1)
    diagnostics = exchange_samples.diagnostics
    assert diagnostics.exchange_attempts == diagnostics.attempts == 10
    assert diagnostics.exchange_accepts == diagnostics.accepted == 10
    assert diagnostics.exchange_acceptance_rate == 1.0
    assert diagnostics.single_site_attempts == 0
    assert np.all(np.sum(exchange_samples.configurations, axis=1) == 1)


def test_mixed_proposal_reports_each_move_type_and_validates_controls():
    tensors = (
        np.ones((1, 1, 2)),
        np.ones((1, 1, 2)),
    )
    hamiltonian = LocalHamiltonian((2, 2), ())
    vmc = LETTAVMC(
        tensors,
        hamiltonian,
        physical_sites=((0,), (1,)),
        seed=13,
        proposal="mixed",
        exchange_probability=0.4,
    )
    samples = vmc.sample(20, burn_in=2, sweeps_between=1)
    diagnostics = samples.diagnostics
    assert (
        diagnostics.single_site_attempts + diagnostics.exchange_attempts
        == diagnostics.attempts
    )
    assert diagnostics.single_site_attempts > 0
    assert diagnostics.exchange_attempts > 0

    with np.testing.assert_raises_regex(ValueError, "proposal"):
        LETTAVMC(
            tensors,
            hamiltonian,
            physical_sites=((0,), (1,)),
            proposal="bad",
        )
    with np.testing.assert_raises_regex(ValueError, "exchange_probability"):
        LETTAVMC(
            tensors,
            hamiltonian,
            physical_sites=((0,), (1,)),
            proposal="mixed",
            exchange_probability=1.1,
        )


def test_energy_estimate_reports_autocorrelation_effective_sample_size():
    rng = np.random.default_rng(115)
    values = np.empty(800)
    values[0] = rng.normal()
    for index in range(1, len(values)):
        values[index] = 0.92 * values[index - 1] + rng.normal(scale=0.3)
    diagnostics = MetropolisDiagnostics(
        attempts=0,
        accepted=0,
        acceptance_rate=0.0,
        zero_amplitude_rejections=0,
        initialization_attempts=1,
        site_attempts=(0,),
        site_accepts=(0,),
    )
    samples = VMCSamples(
        configurations=np.zeros((len(values), 1), dtype=np.intp),
        amplitudes=np.ones(len(values)),
        local_energies=values,
        log_derivatives=None,
        diagnostics=diagnostics,
    )
    estimate = LETTAVMC.estimate_from_samples(samples)

    np.testing.assert_allclose(estimate.real_variance, np.var(values))
    np.testing.assert_allclose(
        estimate.standard_error,
        np.sqrt(estimate.real_variance / len(values)),
    )
    assert estimate.integrated_autocorrelation_time > 1.0
    assert estimate.effective_sample_size < 0.5 * len(values)
    assert estimate.autocorrelation_standard_error > estimate.standard_error


def test_regularized_matrix_free_sr_proposal_lowers_one_spin_energy():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = LocalHamiltonian((2,), (LocalTerm((0,), -sx),))
    tensor = np.array([[[0.82 + 0.03j, 0.57 - 0.02j]]])
    vmc = LETTAVMC(
        (tensor,),
        hamiltonian,
        physical_sites=((0,),),
        seed=17,
        initial_configuration=np.array([0]),
    )
    before_vector = vmc.amplitudes(np.array([[0], [1]]))
    before = hamiltonian.expectation(before_vector)
    samples = vmc.sample(
        5000,
        burn_in=100,
        sweeps_between=1,
        include_log_derivatives=True,
    )
    proposal = vmc.propose_sr(
        samples,
        step_size=0.04,
        diagonal_shift=1.0e-2,
        diagonal_floor=1.0e-9,
        tolerance=1.0e-10,
    )

    assert isinstance(proposal, SRProposal)
    assert proposal.direction.converged
    assert proposal.direction.iterations > 0
    assert np.all(np.isfinite(proposal.delta))
    assert np.issubdtype(proposal.tensors[0].dtype, np.complexfloating)
    vmc.apply_sr(proposal)
    after_vector = vmc.amplitudes(np.array([[0], [1]]))
    after = hamiltonian.expectation(after_vector)
    assert after < before - 1.0e-4


def test_vmc_parameters_sync_explicitly_to_source_state():
    state, hamiltonian = _complex_frontier_state(seed=14)
    original = [tensor.copy() for tensor in state.tensors]
    vmc = LETTAVMC(state, hamiltonian, seed=9)
    parameters = vmc.wavefunction.parameter_vector()
    parameters[0] += 0.125 - 0.03j
    vmc.wavefunction.set_parameter_vector(parameters)
    state.history = [{"stale": True}]

    np.testing.assert_allclose(state.tensors[0], original[0])
    returned = vmc.sync_to_state()
    assert returned is state
    for actual, expected in zip(state.tensors, vmc.tensors):
        np.testing.assert_allclose(actual, expected)
        assert actual is not expected
    assert state.energy is None
    assert not state.converged
    assert state.history == []

    detached = LETTAVMC(
        tuple(original),
        hamiltonian,
        physical_sites=state.physical_sites,
        seed=9,
    )
    with np.testing.assert_raises_regex(ValueError, "no source state"):
        detached.sync_to_state()
