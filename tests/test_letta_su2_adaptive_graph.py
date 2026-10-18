"""Reduced adaptive ties: exact migration, budgets and variational guards."""
import numpy as np
import pytest

from pyqed.letta import SU2LETTA
from pyqed.letta.su2_adaptive_graph import _sector_information


@pytest.fixture
def state():
    h = np.diag([-1., -.5, .25, .75])
    h[0, 2] = h[2, 0] = -.2
    s = SU2LETTA.from_integrals(h, nelec=4, graph=[(0, 1)], D=1, seed=4,
                               hamiltonian_representation='mpo')
    yield s
    s.close()


@pytest.mark.parametrize('representation', ['mpo', 'complementary'])
def test_added_nonadjacent_ties_preserve_reduced_state(representation):
    h = np.diag([-1., -.5, .25, .75])
    h[0, 2] = h[2, 0] = -.2
    h[1, 3] = h[3, 1] = -.15
    s = SU2LETTA.from_integrals(h, nelec=4,
                               graph=[(0, 2)], D=1, seed=4,
                               hamiltonian_representation=representation)
    rng = np.random.default_rng(7)
    for i in range(s.nsites):
        x = s._pack_site(i)
        factor = 1 + .2 * rng.normal(size=x.size)
        if representation == 'mpo':
            factor = factor + .2j * rng.normal(size=x.size)
        s._set_site_vector(i, x * factor)
    t = s.with_tie_graph([(0, 1), (0, 2), (1, 3)])
    try:
        np.testing.assert_allclose(t.expectation(), s.expectation(), atol=1e-11)
        np.testing.assert_allclose(t.norm(), s.norm(), atol=1e-11)
        np.testing.assert_allclose(s.overlap(t), s.overlap(s), atol=1e-10)
        assert t.target_sector == s.target_sector
        assert t.tie == s.tie == 'physical'
        assert t.nparameters > s.nparameters
        for old, new in zip(s._base_sites, t._base_sites):
            assert old.qns == new.qns
        t._set_site_vector(0, t._pack_site(0) * 2)
        np.testing.assert_allclose(t.norm(), 4 * s.norm(), atol=1e-10)
    finally:
        s.close()
        t.close()


def test_deferred_migration_avoids_frontier_contractions(state, monkeypatch):
    expected = state.expectation()
    def forbidden(*args, **kwargs):
        raise AssertionError('deferred migration must not contract')
    with monkeypatch.context() as patch:
        patch.setattr(SU2LETTA, 'expectation', forbidden)
        migrated = state.with_tie_graph([(0, 1), (0, 2), (1, 3)],
                                       evaluate_energy=False)
    try:
        assert migrated.energy is None
        assert not migrated._materialized_site_cache
        np.testing.assert_allclose(migrated.expectation(), expected, atol=1e-11)
    finally:
        migrated.close()


def test_migration_rejects_removal_and_frontier_overflow(state):
    with pytest.raises(ValueError, match='additions only'):
        state.with_tie_graph([])
    with pytest.raises(MemoryError, match='frontier'):
        state.with_tie_graph([(0, 1), (0, 2)], max_frontier_states=3)


def test_sector_information_agrees_with_dense_reference(state):
    # Dense determinant expansion is confined to this four-orbital reference.
    from test_letta_su2_qchem import _dense_vector_from_reduced_spatial_mps
    vector = _dense_vector_from_reduced_spatial_mps(state.materialize()).reshape((4,) * 4)
    probability = abs(vector) ** 2
    probability /= probability.sum()
    # Empty, alpha, beta, double -> invariant empty, single, double labels.
    joint = np.zeros((3, 3))
    labels = (0, 1, 1, 2)
    for configuration in np.ndindex(probability.shape):
        joint[labels[configuration[0]], labels[configuration[2]]] += probability[configuration]
    independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    mask = joint > 0
    expected = np.sum(joint[mask] * np.log(joint[mask] / independent[mask]))
    np.testing.assert_allclose(_sector_information(state, [(0, 2)])[0, 2], expected, atol=1e-12)


def test_cost_caps_prune_before_ranking(state, monkeypatch):
    from pyqed.letta import su2_adaptive_graph as module
    def no_candidates(s, edges):
        assert not edges
        return {}
    monkeypatch.setattr(module, '_sector_information', no_candidates)
    old = [state._pack_site(i).copy() for i in range(state.nsites)]
    state.adapt_ties(candidate_edges=[(0, 2)], max_parents=1)
    assert state.tie_history[-1]['proposals'] == []
    for i, x in enumerate(old):
        np.testing.assert_array_equal(state._pack_site(i), x)


def test_rejected_trials_preserve_graph_tensors_history(state):
    old = [state._pack_site(i).copy() for i in range(state.nsites)]
    state.adapt_ties(candidate_edges=[(0, 2)], minimum_energy_gain=1e6,
                     run_options={'solver': 'wigner_eckart'})
    assert state.graph == ((0, 1),)
    assert state.history == []
    assert state.tie_history[-1]['accepted_edge'] is None
    assert len(state.tie_history[-1]['proposals']) == 1
    for i, x in enumerate(old):
        np.testing.assert_array_equal(state._pack_site(i), x)


def test_acceptance_and_checkpoint_keep_reduced_graph(state, monkeypatch, tmp_path):
    # Control has no improvement; a controlled valid candidate is optimized.
    original_run = SU2LETTA.run
    def trial(s, **kwargs):
        if (0, 2) in s.graph:
            return original_run(s, nsweeps=1, solver='wigner_eckart')
        return s
    monkeypatch.setattr(SU2LETTA, 'run', trial)
    before = state.expectation()
    state.adapt_ties(candidate_edges=[(0, 2)], minimum_energy_gain=0)
    assert state.graph == ((0, 1), (0, 2))
    assert state.energy < before
    assert not state.converged
    assert state.tie_history[-1]['accepted_edge'] == (0, 2)
    assert state.tie_history[-1]['proposals'][0]['tie_energy_gain'] > 0
    path = state.save_checkpoint(tmp_path / 'adaptive.pkl')
    restored = SU2LETTA.load_checkpoint(path)
    try:
        assert restored.graph == state.graph
        assert restored.tie_history == state.tie_history
        np.testing.assert_allclose(restored.expectation(), state.energy, atol=1e-11)
    finally:
        restored.close()


def test_run_adaptive_dispatch_and_interval(state, monkeypatch):
    calls = []
    def adapt(s, **kwargs):
        calls.append((len(s.history), kwargs))
        return s
    monkeypatch.setattr(SU2LETTA, 'adapt_ties', adapt)
    state.run(nsweeps=3, adaptive_ties=True, tie_options={'every': 2, 'shortlist': 1},
              tol=0., solver='wigner_eckart')
    assert [count for count, _ in calls] == [2, 3]
    assert all(options['shortlist'] == 1 for _, options in calls)


@pytest.mark.parametrize('options', [{'shortlist': 0}, {'max_parents': 0},
                                     {'cost_weight': -1}, {'minimum_energy_gain': np.nan}])
def test_adaptive_options_validate_before_mutation(state, options):
    with pytest.raises(ValueError):
        state.adapt_ties(**options)
    assert state.tie_history == []


def test_fusion_ties_preserve_invariant_state():
    s = SU2LETTA.from_integrals(np.diag([-1., -.5, .25, .75]), nelec=4,
                               graph=[], tie='fusion', D=1, seed=4,
                               hamiltonian_representation='mpo')
    t = s.with_tie_graph([(0, 2), (1, 3)])
    try:
        assert t.tie == 'fusion'
        np.testing.assert_allclose(s.overlap(t), s.overlap(s), atol=1e-10)
        np.testing.assert_allclose(t.expectation(), s.expectation(), atol=1e-11)
        assert np.isfinite(_sector_information(t, [(0, 2)])[0, 2])
    finally:
        s.close()
        t.close()


def test_adaptive_run_with_actual_trials_and_checkpoint(state, tmp_path):
    path = tmp_path / 'run.pkl'
    state.run(nsweeps=2, adaptive_ties=True,
              tie_options={'every': 1, 'candidate_edges': [(0, 2)],
                           'shortlist': 1, 'minimum_energy_gain': 1e6},
              tol=0., solver='wigner_eckart', checkpoint=path)
    assert len(state.history) == 2
    assert len(state.tie_history) == 2
    assert state.graph == ((0, 1),)
    restored = SU2LETTA.load_checkpoint(path)
    try:
        assert restored.tie_history == state.tie_history
        np.testing.assert_allclose(restored.expectation(), state.energy, atol=1e-11)
    finally:
        restored.close()


def test_control_rejects_improvement_available_without_new_tie():
    s = SU2LETTA.from_integrals(np.diag([-1., -.5, .25, .75]), nelec=4,
                               graph=[], D=1, seed=4, hamiltonian_representation='mpo')
    old = [s._pack_site(i).copy() for i in range(s.nsites)]
    try:
        s.adapt_ties(candidate_edges=[(0, 2)], minimum_energy_gain=0,
                     run_options={'solver': 'wigner_eckart'})
        record = s.tie_history[-1]
        assert record['control_energy'] < record['energy_before']
        assert record['proposals'][0]['energy_gain'] > 0
        assert record['accepted_edge'] is None
        assert s.graph == ()
        for i, x in enumerate(old):
            np.testing.assert_array_equal(s._pack_site(i), x)
    finally:
        s.close()


def test_direct_complementary_adaptive_trials():
    h = np.diag([-1., -.5, .25, .75])
    h[0, 2] = h[2, 0] = -.2
    s = SU2LETTA.from_integrals(h, nelec=4, graph=[(0, 1)], D=1, seed=4)
    try:
        s.run(nsweeps=1, adaptive_ties=True,
              tie_options={'every': 1, 'candidate_edges': [(0, 2)],
                           'shortlist': 1, 'minimum_energy_gain': 1e6},
              solver='wigner_eckart')
        assert s.hamiltonian_representation == 'complementary'
        assert len(s.tie_history[-1]['proposals']) == 1
        assert s.tie_history[-1]['accepted_edge'] is None
        assert np.isfinite(s.energy)
    finally:
        s.close()


def test_runtime_memory_limit_rejects_trial_without_losing_state(state, monkeypatch):
    original = SU2LETTA.run
    def limited(s, **options):
        if (0, 2) in s.graph:
            raise MemoryError('bounded route workspace')
        return original(s, **options)
    monkeypatch.setattr(SU2LETTA, 'run', limited)
    vectors = [state._pack_site(i).copy() for i in range(state.nsites)]
    state.adapt_ties(candidate_edges=[(0, 2)], run_options={'solver': 'wigner_eckart'})
    record = state.tie_history[-1]
    assert record['accepted_edge'] is None
    assert record['proposals'][0]['status'] == 'memory_limit'
    assert record['proposals'][0]['error'] == 'bounded route workspace'
    assert state.graph == ((0, 1),)
    for i, vector in enumerate(vectors):
        np.testing.assert_array_equal(state._pack_site(i), vector)
