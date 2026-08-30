import numpy as np

from pyqed.lattice import Site, SpinHalfSite
from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    ConditionalLETTAWavefunction,
    ConditionalFrontierLETTA,
    FrontierLETTA,
    FrontierTiedLETTA,
    FutureLETTA,
    GraphLETTA,
    LETTAWavefunction,
    U1ConditionalFrontierLETTA,
    VMC,
)
from pyqed.tn import Hamiltonian


def _heisenberg(sites, edges):
    hamiltonian = Hamiltonian(sites)
    for left, right in edges:
        for name in ("Sx", "Sy", "Sz"):
            hamiltonian.add_product(1.0, (left, name), (right, name))
    return hamiltonian


def test_conditional_factors_match_materialized_exact_frontier():
    sites = (Site(2),) * 4
    hamiltonian = Hamiltonian(sites)
    parents = ((1, 2, 3), (2, 3), (3,), ())
    state = ConditionalFrontierLETTA(
        hamiltonian,
        parents,
        bond_dim=2,
        chi=2,
        seed=4,
    )
    reference = FrontierTiedLETTA(
        hamiltonian,
        parents,
        bond_dims=state.bond_dims,
        tensors=state.materialize_tensors(),
    )

    np.testing.assert_allclose(state.norm(), reference.norm(), atol=2.0e-13)
    np.testing.assert_allclose(state.expectation(), reference.expectation(), atol=2.0e-13)
    np.testing.assert_allclose(state.state_vector(), reference.state_vector(), atol=2.0e-13)
    assert state.stored_tensor_elements == state.nparameters
    assert not isinstance(state.tensors, list)


def test_pair_grouped_conditional_factor_matches_exact_frontier_and_sweeps():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3), (0, 3)))
    parents = ((1, 2), (2, 3), (3,), ())
    state = ConditionalFrontierLETTA(
        hamiltonian,
        parents,
        bond_dim=2,
        chi=2,
        parent_group_size=2,
        seed=14,
    )
    reference = FrontierTiedLETTA(
        hamiltonian,
        parents,
        bond_dims=state.bond_dims,
        tensors=state.materialize_tensors(),
    )

    assert len(state.factors[0]) == 2
    assert state.factors[0][1].shape == (2, 2, 2, 2, 2)
    np.testing.assert_allclose(state.state_vector(), reference.state_vector(), atol=3e-13)
    initial = state.expectation()
    state.run(nsweeps=1, tol=0.0)
    assert state.energy <= initial + 3.0e-12

    public = FrontierLETTA(
        hamiltonian,
        graph=((0, 1), (0, 2), (1, 3)),
        D=2,
        chi=2,
        tie_group=2,
        seed=14,
    )
    assert public.tie_group == 2
    assert public.factors[0][1].ndim == 5


def test_conditional_frontier_factor_sweep_is_variational():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        D=2,
        chi=2,
        seed=3,
    )
    initial = state.energy

    state.run(nsweeps=2, tol=0.0, environment_cache="checkpointed")

    assert isinstance(state, ConditionalFrontierLETTA)
    energies = [initial] + [record["energy"] for record in state.history]
    assert np.all(np.diff(energies) <= 2.0e-11)
    assert all(record["environment_cache"] == "checkpointed" for record in state.history)
    assert any(record["accepted_factors"] for record in state.history)


def test_u1_conditional_factors_are_neutral_controls_and_match_dense_blocks():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge={"Sz": 0},
        D=2,
        chi=2,
        frontier_backend="identity_block",
        seed=5,
    )
    reference = AbelianFrontierTiedLETTA(
        hamiltonian,
        state.parent_sets,
        abelian_layout=state.abelian_layout,
        bond_dims=state.bond_dims,
        tensors=state.materialize_tensors(),
        frontier_backend="identity_block",
    )

    assert isinstance(state, U1ConditionalFrontierLETTA)
    np.testing.assert_allclose(state.expectation(), reference.expectation(), atol=3.0e-13)
    assert state.unfactorized_nparameters == sum(
        np.count_nonzero(mask) for mask in state.local_masks
    )
    for tensor, mask in zip(state.tensors, state.local_masks):
        assert np.count_nonzero(tensor[~mask]) == 0

    environment = state.site_environment(0)
    rng = np.random.default_rng(13)
    vectors = rng.normal(size=(3, state.tensors[0].size))
    for frontier, left, right in (
        (
            state._norm_frontier,
            environment.norm_left,
            environment.norm_right,
        ),
        (
            state._hamiltonian_frontier,
            environment.hamiltonian_left,
            environment.hamiltonian_right,
        ),
    ):
        batched = frontier.hole_actions(0, left, right, vectors)
        separate = np.stack(
            [frontier.hole_action(0, left, right, vector) for vector in vectors]
        )
        np.testing.assert_allclose(batched, separate, atol=3.0e-13)

    clone = state.copy()
    np.testing.assert_allclose(clone.expectation(), state.expectation(), atol=3.0e-13)
    np.testing.assert_allclose(clone.state_vector(), state.state_vector(), atol=3.0e-13)
    initial = state.energy

    state.run(nsweeps=1, tol=0.0)

    assert state.energy <= initial + 2.0e-12
    for tensor, mask in zip(state.tensors, state.local_masks):
        assert np.count_nonzero(tensor[~mask]) == 0


def test_conditional_frontier_rejects_adaptive_backbone():
    sites = (SpinHalfSite(),) * 2
    hamiltonian = _heisenberg(sites, ((0, 1),))

    with np.testing.assert_raises_regex(ValueError, "require fixed D"):
        FrontierLETTA(
            hamiltonian,
            D=2,
            chi=2,
            adaptive_bond=True,
        )


def test_complete_graph_shorthand_ties_each_site_to_every_future_site():
    sites = (Site(2),) * 4
    hamiltonian = Hamiltonian(sites)

    state = FrontierLETTA(
        hamiltonian,
        graph="complete",
        D=1,
        chi=1,
        seed=2,
    )

    assert state.parent_sets == ((1, 2, 3), (2, 3), (3,), ())
    assert state.graph == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )


def test_future_enumeration_matches_exact_complete_frontier_contraction():
    sites = (SpinHalfSite(),) * 5
    hamiltonian = _heisenberg(
        sites,
        ((0, 1), (1, 2), (2, 3), (3, 4), (0, 3), (1, 4)),
    )
    state = FutureLETTA(
        hamiltonian,
        D=2,
        chi=2,
        init="random",
        seed=17,
    )
    frontier = ConditionalFrontierLETTA(
        hamiltonian,
        state.parent_sets,
        bond_dim=2,
        chi=2,
        factors=state.factors,
        init="random",
    )

    np.testing.assert_allclose(
        state.expectation(),
        frontier.expectation(),
        atol=3.0e-13,
    )


def test_future_letta_is_factor_native_and_matches_small_dense_reference():
    sites = (Site(2),) * 5
    hamiltonian = Hamiltonian(sites)
    state = FutureLETTA(
        hamiltonian,
        D=2,
        chi=2,
        init="random",
        seed=7,
    )
    wavefunction = ConditionalLETTAWavefunction.from_state(state)
    reference = LETTAWavefunction(
        state.materialize_tensors(),
        state.physical_groups,
        state.dims,
    )
    configurations = np.asarray(list(np.ndindex(*state.dims)), dtype=np.intp)

    assert state.parent_sets == (
        (1, 2, 3, 4),
        (2, 3, 4),
        (3, 4),
        (4,),
        (),
    )
    assert not hasattr(state, "tensors")
    np.testing.assert_allclose(
        wavefunction.amplitudes(configurations),
        reference.amplitudes(configurations),
        atol=2.0e-13,
    )
    configuration = configurations[11]
    parameters = wavefunction.parameter_vector()
    derivative = wavefunction.log_derivative(configuration)
    active = np.flatnonzero(np.abs(derivative) > 0)
    index = int(active[len(active) // 2])
    epsilon = 2.0e-7
    plus = parameters.copy()
    minus = parameters.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    amplitude = wavefunction.amplitude(configuration)
    wavefunction.set_parameter_vector(plus)
    amplitude_plus = wavefunction.amplitude(configuration)
    wavefunction.set_parameter_vector(minus)
    amplitude_minus = wavefunction.amplitude(configuration)
    wavefunction.set_parameter_vector(parameters)
    np.testing.assert_allclose(
        (amplitude_plus - amplitude_minus) / (2 * epsilon * amplitude),
        derivative[index],
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        wavefunction.sparse_log_derivatives(configurations[:3]).toarray(),
        wavefunction.log_derivatives(configurations[:3]),
    )
    reference_vector = reference.amplitudes(configurations)
    np.testing.assert_allclose(
        state.state_vector(),
        reference_vector,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.norm(),
        np.vdot(reference_vector, reference_vector).real,
        atol=2.0e-13,
    )


def test_future_letta_constructs_large_complete_dependence_without_dense_tensors():
    nsites = 16
    hamiltonian = Hamiltonian((Site(2),) * nsites)
    state = FutureLETTA(hamiltonian, D=2, chi=2, seed=11)
    configuration = np.arange(nsites, dtype=np.intp) % 2

    assert len(state.graph) == nsites * (nsites - 1) // 2
    assert len(state.factors[0]) == nsites
    assert state.stored_tensor_elements < 10_000
    assert np.isfinite(state.amplitude(configuration))
    with np.testing.assert_raises_regex(ValueError, "use VMC"):
        state.state_vector(max_states=1000)


def test_future_u1_state_uses_factor_native_vmc():
    sites = (SpinHalfSite(),) * 6
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
    state = FutureLETTA(
        hamiltonian,
        target_charge={"Sz": 0},
        D=4,
        chi=4,
        seed=8,
    )
    configuration = np.array([0, 1, 0, 1, 0, 1])
    vmc = VMC(
        state,
        seed=9,
        initial_configuration=configuration,
        proposal="exchange",
    )

    assert isinstance(vmc.wavefunction, ConditionalLETTAWavefunction)
    assert state.target_charge == {"2sz": 0}
    assert np.isfinite(vmc.amplitude(configuration))
    assert vmc.sample(3, burn_in=1).configurations.shape == (3, 6)


def test_autoregressive_future_letta_is_exactly_normalized_and_samples_directly():
    sites = (Site(2),) * 4
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(0.7, (0, "I"), (3, "I"))
    state = FutureLETTA(
        hamiltonian,
        D=2,
        chi=2,
        init="random",
        autoregressive=True,
        seed=23,
    )
    vector = state.state_vector()
    probabilities = np.abs(vector) ** 2

    np.testing.assert_allclose(np.sum(probabilities), 1.0, atol=3.0e-13)
    np.testing.assert_allclose(state.norm(), 1.0, atol=3.0e-13)
    samples, sampled_amplitudes = state.sample(
        12_000,
        seed=5,
        return_amplitudes=True,
    )
    sampled_indices = np.ravel_multi_index(samples.T, state.dims)
    empirical = np.bincount(sampled_indices, minlength=len(vector)) / len(samples)
    np.testing.assert_allclose(empirical, probabilities, atol=0.018)
    np.testing.assert_allclose(
        sampled_amplitudes,
        state.amplitudes(samples),
        atol=3.0e-14,
    )
    with np.testing.assert_raises_regex(TypeError, "SR derivative"):
        VMC(state)


def test_autoregressive_future_u1_samples_remain_in_target_sector():
    sites = (SpinHalfSite(),) * 6
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
    state = FutureLETTA(
        hamiltonian,
        target_charge={"Sz": 0},
        D=4,
        chi=4,
        init="random",
        autoregressive=True,
        seed=29,
    )

    samples = state.sample(2000, seed=7)

    assert np.all(np.sum(samples, axis=1) == 3)
    np.testing.assert_allclose(state.norm(), 1.0, atol=5.0e-13)


def test_autoregressive_graph_letta_adds_neutral_ties_without_changing_state():
    sites = (Site(2),) * 5
    hamiltonian = Hamiltonian(sites)
    graph = ((0, 1), (1, 2), (2, 3), (3, 4))
    state = GraphLETTA(
        hamiltonian,
        graph=graph,
        D=2,
        chi=2,
        init="random",
        seed=31,
    )
    configurations = np.asarray(list(np.ndindex(*state.dims)), dtype=np.intp)
    before = state.amplitudes(configurations)

    expanded = state.with_added_ties(((0, 3), (1, 4)))

    assert set(expanded.graph) == set(graph) | {(0, 3), (1, 4)}
    assert expanded.nparameters > state.nparameters
    np.testing.assert_allclose(
        expanded.amplitudes(configurations),
        before,
        atol=4.0e-13,
    )
    np.testing.assert_allclose(expanded.norm(), 1.0, atol=4.0e-13)


def test_autoregressive_graph_letta_adapts_selected_long_range_ties():
    sites = (Site(2),) * 5
    hamiltonian = Hamiltonian(sites)
    state = GraphLETTA(
        hamiltonian,
        graph=((0, 1), (1, 2), (2, 3), (3, 4)),
        D=2,
        chi=2,
        init="random",
        seed=37,
    )
    before = state.state_vector()

    adapted = state.adapt_ties(
        n_ties=2,
        nsamples=3000,
        candidate_edges=((0, 2), (0, 4), (1, 3), (2, 4)),
        seed=41,
    )

    record = adapted.adaptation_history[-1]
    assert len(record["added_ties"]) == 2
    assert len(adapted.graph) == len(state.graph) + 2
    np.testing.assert_allclose(adapted.state_vector(), before, atol=5.0e-13)


def test_autoregressive_u1_graph_tie_migration_preserves_charge_and_amplitudes():
    sites = (SpinHalfSite(),) * 6
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
    state = GraphLETTA(
        hamiltonian,
        graph=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
        target_charge={"Sz": 0},
        D=4,
        chi=4,
        init="random",
        seed=43,
    )
    before = state.state_vector()

    expanded = state.with_added_ties(((0, 4), (1, 5)))
    samples = expanded.sample(1000, seed=47)

    np.testing.assert_allclose(expanded.state_vector(), before, atol=8.0e-13)
    assert np.all(np.sum(samples, axis=1) == 3)
