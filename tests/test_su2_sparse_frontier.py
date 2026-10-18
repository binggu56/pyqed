import numpy as np
import pytest

from pyqed.letta import SU2LETTA


@pytest.fixture(params=[False, True])
def state(request):
    h = np.zeros((4, 4))
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        h[a, b] = h[b, a] = -1
    eri = np.zeros((4,) * 4)
    i = np.arange(4)
    eri[i, i, i, i] = 4
    state = SU2LETTA.from_integrals(h, eri, nelec=4, D=1, seed=9,
        graph=[(0, 1), (1, 2), (2, 3), (0, 3)],
        hamiltonian_representation='mpo', n_threads=1)
    rng = np.random.default_rng(3)
    for i in range(4):
        vector = state._pack_site(i)
        vector = vector * (1 + .1 * rng.normal(size=vector.size))
        if request.param:
            vector = vector + .1j * rng.normal(size=vector.size)
        state._set_site_vector(i, vector)
    yield state
    state.close()


def test_sparse_energy_norm_without_dense_frontier(state, monkeypatch):
    energy, norm = state.expectation(), state.norm()
    state._materialized_site_cache.clear()
    backend = state.frontier_contractions(chunk_bytes=256)
    def forbidden(*args, **kwargs):
        raise AssertionError('Dense frontier materialization')
    monkeypatch.setattr(state, 'materialize', forbidden)
    monkeypatch.setattr(state, 'materialize_site', forbidden)
    np.testing.assert_allclose(backend.norm(), norm, rtol=1e-11)
    np.testing.assert_allclose(backend.expectation(), energy, atol=1e-10)
    assert not state._materialized_site_cache


def test_parameter_actions_match_dense_reference(state, tmp_path):
    backend = state.frontier_contractions(chunk_bytes=4096)
    reference = state.reduced_parameter_actions()
    for identity, expected in zip((False, True), reference):
        actual = list(backend.parameter_actions(identity=identity, directory=tmp_path))
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
        assert not list(tmp_path.iterdir())


def test_local_actions_are_hermitian_and_match_rayleigh(state):
    backend = state.frontier_contractions(chunk_bytes=4096)
    i = 1
    h, n = backend.local_operator(i), backend.local_operator(i, identity=True)
    v = state._pack_site(i)
    rng = np.random.default_rng(11)
    x = rng.normal(size=v.size) + 1j * rng.normal(size=v.size)
    y = rng.normal(size=v.size) + 1j * rng.normal(size=v.size)
    for operator in (h, n):
        np.testing.assert_allclose(np.vdot(x, operator @ y), np.vdot(operator @ x, y), atol=1e-10)
    np.testing.assert_allclose(np.vdot(v, h @ v) / np.vdot(v, n @ v), state.expectation(), atol=1e-10)
    # Rebuild neither environments nor routes when only the active site changes.
    state._set_site_vector(i, x)
    np.testing.assert_allclose(np.vdot(x, h @ x) / np.vdot(x, n @ x), state.expectation(), atol=1e-10)


def test_budget_failure_does_not_modify_state(state):
    original = [state._pack_site(i).copy() for i in range(4)]
    backend = state.frontier_contractions(memory_bytes=32, chunk_bytes=32)
    with pytest.raises(MemoryError):
        backend.norm()
    for i, vector in enumerate(original):
        np.testing.assert_array_equal(state._pack_site(i), vector)


def test_sparse_sweep_preserves_layout_and_lowers_energy(state, tmp_path, monkeypatch):
    from pyqed.letta.su2_qchem import _FrontierTieRoutePlan
    def forbidden(*args, **kwargs):
        raise AssertionError('Dense pair space or parameter basis')
    monkeypatch.setattr(state, '_pair_coordinate_space', forbidden)
    monkeypatch.setattr(_FrontierTieRoutePlan, 'active_block_basis', forbidden)
    before = state.expectation()
    layout = [s.qns for s in state._base_sites]
    backend = state.frontier_contractions(chunk_bytes=4096)
    for direction in ('lr', 'rl'):
        updates = backend.sweep(direction=direction, directory=tmp_path,
                                tolerance=1e-7, maxiter=200, max_space=64)
        assert len(updates) == state.nsites
        assert max(r['residual'] for r in updates) <= 1e-7
        fresh = state.expectation()
        np.testing.assert_allclose(state.energy, fresh, atol=1e-10)
        assert fresh <= before + 1e-10
        before = fresh
    assert [s.qns for s in state._base_sites] == layout
    assert not list(tmp_path.iterdir())


def test_unevaluated_checkpoint_can_load_without_dense_contraction(state, tmp_path, monkeypatch):
    state.energy = None
    file = state.save_checkpoint(tmp_path/'state.chk')
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected dense contraction')
    monkeypatch.setattr(SU2LETTA, 'expectation', forbidden)
    restored = SU2LETTA.load_checkpoint(file, n_threads=1)
    try:
        assert restored.energy is None
        np.testing.assert_allclose(restored.frontier_contractions().norm(),
                                   state.frontier_contractions().norm(), atol=1e-11)
    finally:
        restored.close()


def test_full_rectangle_support_masks_and_nonzero_core_shift():
    # Snake ordering for a 3x2 rectangle; ties include every lattice edge.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]
    h = np.diag(np.linspace(-.2, .3, 6))
    for a, b in edges:
        h[a, b] = h[b, a] = -.7
    eri = np.zeros((6,) * 4)
    i = np.arange(6)
    eri[i, i, i, i] = 2
    state = SU2LETTA.from_integrals(h, eri, nelec=6, graph=edges, D=1,
                                   seed=4, ecore=1.3, n_threads=1,
                                   hamiltonian_representation='mpo')
    try:
        rng = np.random.default_rng(42)
        for i in range(6):
            v = state._pack_site(i)
            v *= 1 + .2 * rng.normal(size=v.size)
            state._set_site_vector(i, v)
        backend = state.frontier_contractions(chunk_bytes=4096)
        np.testing.assert_allclose(backend.expectation(), state.expectation(), atol=1e-10)
        # Compare pruning to the same sparse recursion without pruning.
        factors = backend.hamiltonian
        masked = backend.contract()
        backend._supports[id(factors)] = None
        np.testing.assert_allclose(backend.contract(), masked, atol=1e-10)
    finally:
        state.close()


def test_complementary_frontier_matches_explicit_mpo_actions(state, tmp_path):
    h = np.zeros((4, 4))
    graph = [(0, 1), (1, 2), (2, 3), (0, 3)]
    for a, b in graph:
        h[a, b] = h[b, a] = -1
    eri = np.zeros((4,) * 4)
    i = np.arange(4)
    eri[i, i, i, i] = 4
    complementary = SU2LETTA.from_integrals(
        h, eri, nelec=4, D=1, seed=9, graph=graph,
        hamiltonian_representation='complementary', n_threads=1)
    try:
        for i in range(4):
            complementary._set_site_vector(i, state._pack_site(i))
        backend = complementary.frontier_contractions(chunk_bytes=4096)
        np.testing.assert_allclose(backend.expectation(), state.expectation(), atol=1e-10)
        for identity, reference in zip((False, True), state.reduced_parameter_actions()):
            actual = list(backend.parameter_actions(identity=identity, directory=tmp_path))
            for value, expected in zip(actual, reference):
                np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-10)
        # Local operators must recompute shared products for each trial vector.
        operator = backend.local_operator(1)
        reference = state.frontier_contractions().local_operator(1)
        rng = np.random.default_rng(219)
        for _ in range(3):
            vector = rng.normal(size=operator.shape[0]) + 1j*rng.normal(size=operator.shape[0])
            np.testing.assert_allclose(operator @ vector, reference @ vector,
                                       atol=1e-10, rtol=1e-10)
        assert backend.stats['dense_site_allocations'] == 0
    finally:
        complementary.close()
