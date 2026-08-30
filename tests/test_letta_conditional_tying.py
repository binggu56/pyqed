import numpy as np

from pyqed.letta import ConditionalTTLETTA


def _projector_hamiltonian(state):
    state = np.asarray(state, dtype=complex).reshape(-1)
    state /= np.linalg.norm(state)
    return -np.outer(state, state.conj())


def test_chi_one_is_matrix_valued_pair_product():
    dims = (2, 2, 2)
    parents = ((1, 2), (), ())
    B = np.array([[[1.0], [2.0]]])
    C01 = np.array(
        [[[[1.0], [3.0]], [[2.0], [5.0]]]]
    )
    C02 = np.array(
        [[[[7.0], [11.0]], [[13.0], [17.0]]]]
    )
    terminal = np.ones((1, 2, 1))
    state = ConditionalTTLETTA(
        -np.eye(8),
        dims,
        parents,
        D=1,
        chi=1,
        factors=((B, C01, C02), (terminal,), (terminal,)),
    )

    expected = np.empty(dims)
    for s0, s1, s2 in np.ndindex(*dims):
        expected[s0, s1, s2] = (
            B[0, s0, 0]
            * C01[0, s0, s1, 0]
            * C02[0, s0, s2, 0]
        )
    expected = expected.reshape(-1)
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(state.state_vector(normalize=True), expected)
    np.testing.assert_allclose(
        state.materialize_tensor(0)[0, 0],
        expected.reshape(dims),
    )


def test_chi_two_resolves_parent_correlation_missed_by_chi_one():
    dims = (2, 2, 2)
    parents = ((1, 2), (), ())
    target = np.zeros(dims)
    for s0, s1 in np.ndindex(2, 2):
        target[s0, s1, s1] = 0.5
    tensors = (
        target[None, None, ...],
        np.ones((1, 1, 2)),
        np.ones((1, 1, 2)),
    )
    hamiltonian = _projector_hamiltonian(target)

    rank_one = ConditionalTTLETTA.from_dense(
        hamiltonian,
        dims,
        parents,
        tensors,
        chi=1,
    )
    rank_two = ConditionalTTLETTA.from_dense(
        hamiltonian,
        dims,
        parents,
        tensors,
        chi=2,
    )

    assert rank_one.fidelity(target) < 0.75
    np.testing.assert_allclose(rank_two.fidelity(target), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(rank_two.factorization_errors, 0.0, atol=1.0e-12)
    assert rank_two.local_ranks[0] == (2, 2)


def test_conditional_tt_factor_sweep_is_variational():
    rng = np.random.default_rng(7)
    target = rng.normal(size=8) + 1.0j * rng.normal(size=8)
    state = ConditionalTTLETTA(
        _projector_hamiltonian(target),
        (2, 2, 2),
        ((1, 2), (), ()),
        D=1,
        chi=2,
        seed=5,
    )
    initial = state.energy

    state.run(nsweeps=2, tol=0.0)

    energies = [initial] + [record["energy"] for record in state.history]
    assert np.all(np.diff(energies) <= 1.0e-11)
    assert state.energy <= initial + 1.0e-12
    assert any(
        update.accepted
        for record in state.history
        for update in record["updates"]
    )


def test_conditional_tt_reduces_high_degree_local_storage():
    dims = (2,) * 7
    parents = ((1, 2, 3, 4, 5, 6), (), (), (), (), (), ())
    state = ConditionalTTLETTA(
        -np.eye(2**7),
        dims,
        parents,
        D=2,
        chi=2,
        seed=3,
    )

    dense_local = 1 * 2 * 2**7
    factored_local = sum(factor.size for factor in state.factors[0])
    assert factored_local < dense_local
    assert state.compression_ratio < 1.0
