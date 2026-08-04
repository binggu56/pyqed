import numpy as np
import pytest

from pyqed.letta import (
    FrontierTiedLETTA,
    LocalHamiltonian,
    LocalTerm,
    TieSignal,
    TieSignalBatch,
    adaptive_tie_graph_step,
    evaluate_tie_graph_proposal,
    graph_signals_from_samples,
    rank_tie_graph_proposals,
    state_with_tie_graph_proposal,
    tie_frontier_cost,
)


def _two_site_state(seed=3):
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = LocalHamiltonian(
        (2, 2),
        (LocalTerm((0, 1), -np.kron(sx, sx)),),
    )
    return FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((), ()),
        bond_dim=1,
        seed=seed,
        frontier_backend="identity_block",
    )


def _signal_batch(edge=(0, 1), score=1.0):
    signal = TieSignal(
        edge=edge,
        connected_correlation=0.25,
        residual_coupling=score - 0.25,
        score=score,
    )
    return TieSignalBatch(
        signals=(signal,),
        nsamples=16,
        mean_local_energy=0.0j,
        local_energy_variance=1.0,
        acceptance_rate=1.0,
    )


def test_tie_frontier_cost_tracks_distinct_crossing_variables():
    cost = tie_frontier_cost(
        (2, 2, 2, 2),
        ((2, 3), (3,), (), ()),
        bond_dim=3,
    )
    assert cost.peak_width == 2
    assert cost.peak_physical_states == 4
    assert cost.peak_norm_elements == 36
    assert cost.cuts[0].frontier_sites == (2, 3)
    assert cost.cuts[1].frontier_sites == (2, 3)
    assert cost.cuts[2].frontier_sites == (3,)


def test_tie_frontier_cost_and_migration_preserve_variable_bonds():
    cost = tie_frontier_cost(
        (2, 2, 2),
        ((2,), (), ()),
        bond_dims=(1, 2, 5, 1),
    )
    assert tuple(item.norm_elements for item in cost.cuts) == (8, 50)

    hamiltonian = LocalHamiltonian(
        (2, 2, 2),
        (LocalTerm((0,), -np.diag([1.0, -1.0])),),
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((), (), ()),
        bond_dims=(1, 2, 3, 1),
        seed=5,
        frontier_backend="identity_block",
    )
    proposal = rank_tie_graph_proposals(
        state,
        _signal_batch(edge=(0, 2)),
        cost_weight=0.0,
    )[0]
    candidate = state_with_tie_graph_proposal(state, proposal)
    assert candidate.bond_dims == (1, 2, 3, 1)
    np.testing.assert_allclose(candidate.state_vector(), state.state_vector())


def test_sample_signals_separate_correlation_and_pair_energy_residual():
    independent = np.asarray(list(np.ndindex(2, 2)), dtype=np.intp)
    xor_energy = np.asarray([1.0, -1.0, -1.0, 1.0])
    residual = graph_signals_from_samples(
        independent,
        xor_energy,
        (2, 2),
        ((0, 1),),
    ).signals[0]
    assert residual.connected_correlation == 0.0
    np.testing.assert_allclose(residual.residual_coupling, 1.0)

    correlated = np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.intp)
    constant_energy = np.zeros(4)
    correlation = graph_signals_from_samples(
        correlated,
        constant_energy,
        (2, 2),
        ((0, 1),),
    ).signals[0]
    np.testing.assert_allclose(correlation.connected_correlation, 0.5)
    assert correlation.residual_coupling == 0.0


def test_add_tie_embedding_preserves_the_current_wavefunction_exactly():
    state = _two_site_state()
    proposal = rank_tie_graph_proposals(
        state,
        _signal_batch(),
        cost_weight=0.0,
    )[0]
    candidate = state_with_tie_graph_proposal(state, proposal)

    np.testing.assert_allclose(candidate.state_vector(), state.state_vector())
    np.testing.assert_allclose(candidate.expectation(), state.expectation())
    assert candidate.parent_sets == ((1,), ())
    assert state.parent_sets == ((), ())


def test_short_relaxation_is_monotone_and_does_not_mutate_incumbent():
    state = _two_site_state(seed=7)
    vector_before = state.state_vector().copy()
    energy_before = state.expectation()
    proposal = rank_tie_graph_proposals(
        state,
        _signal_batch(),
        cost_weight=0.0,
    )[0]
    evaluation = evaluate_tie_graph_proposal(
        state,
        proposal,
        relaxation_sweeps=1,
        run_options={"solver": "whitened", "tol": 0.0},
    )

    assert evaluation.accepted
    assert evaluation.fresh_energy_check_passed
    assert evaluation.energy_after <= energy_before + 1.0e-12
    np.testing.assert_allclose(evaluation.migrated_energy, energy_before)
    np.testing.assert_allclose(state.state_vector(), vector_before)
    assert state.parent_sets == ((), ())


def test_adaptive_step_obeys_frontier_cap_and_selects_relaxed_copy():
    state = _two_site_state(seed=11)
    blocked = adaptive_tie_graph_step(
        state,
        signal_batch=_signal_batch(),
        max_frontier_width=0,
        shortlist=1,
    )
    assert blocked.proposals == ()
    assert blocked.selected is None
    assert blocked.state is state

    selected = adaptive_tie_graph_step(
        state,
        signal_batch=_signal_batch(),
        max_frontier_width=1,
        shortlist=1,
        cost_weight=0.0,
        relaxation_sweeps=1,
        run_options={"solver": "whitened", "tol": 0.0},
    )
    assert selected.selected is not None
    assert selected.state is selected.selected.candidate_state
    assert selected.state.parent_sets == ((1,), ())
    assert selected.state.expectation() <= state.expectation() + 1.0e-12


def test_graph_migration_rejects_subclasses_instead_of_dropping_structure():
    class SymmetryAwareState(FrontierTiedLETTA):
        pass

    reference = _two_site_state()
    state = SymmetryAwareState(
        reference.hamiltonian,
        reference.dims,
        reference.parent_sets,
        bond_dim=reference.bond_dim,
        tensors=reference.tensors,
        frontier_backend="identity_block",
    )
    proposal = rank_tie_graph_proposals(
        state,
        _signal_batch(),
        cost_weight=0.0,
    )[0]
    with pytest.raises(TypeError, match="projected subclass"):
        state_with_tie_graph_proposal(state, proposal)
