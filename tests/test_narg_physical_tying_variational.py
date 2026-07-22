import numpy as np
import pytest

from pyqed.letta.physical_tying import (
    VariationalPhysicalTie,
    compress_physical_ties,
    fixed_range_parent_sets,
)


_IDENTITY = np.eye(2, dtype=complex)
_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_PAULI_Z = np.diag([1.0, -1.0]).astype(complex)


def _kron(*operators):
    product = operators[0]
    for operator in operators[1:]:
        product = np.kron(product, operator)
    return product


def _crossed_bell_state():
    state = np.zeros((2, 2, 2, 2), dtype=complex)
    for x0 in range(2):
        for x1 in range(2):
            state[x0, x1, x0, x1] = 0.5
    return state


def _crossed_bell_hamiltonian():
    return -(
        _kron(_PAULI_X, _IDENTITY, _PAULI_X, _IDENTITY)
        + _kron(_PAULI_Z, _IDENTITY, _PAULI_Z, _IDENTITY)
        + _kron(_IDENTITY, _PAULI_X, _IDENTITY, _PAULI_X)
        + _kron(_IDENTITY, _PAULI_Z, _IDENTITY, _PAULI_Z)
    )


def _fidelity(left, right):
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(abs(np.vdot(left, right)) ** 2)


def test_exact_crossed_bell_physical_ties_are_a_variational_fixed_point():
    target = _crossed_bell_state()
    hamiltonian = _crossed_bell_hamiltonian()
    compressed = compress_physical_ties(
        target,
        target.shape,
        max_parents=1,
        relative_tolerance=1.0e-14,
    )

    ansatz = compressed.variational_ansatz(hamiltonian)
    ansatz.run(nsweeps=2, tol=0.0)

    assert ansatz.parent_sets == ((2,), (3,), ())
    np.testing.assert_allclose(ansatz.energy, -4.0, atol=1.0e-13)
    np.testing.assert_allclose(ansatz.norm(), 1.0, atol=1.0e-13)
    np.testing.assert_allclose(
        _fidelity(ansatz.state_vector(), target),
        1.0,
        atol=1.0e-13,
    )


def test_crossed_bell_fixed_graph_sweep_recovers_ground_state_monotonically():
    target = _crossed_bell_state()
    hamiltonian = _crossed_bell_hamiltonian()
    parent_sets = ((2,), (3,), ())
    product_seed = np.ones(target.shape, dtype=complex)
    compressed = compress_physical_ties(
        product_seed,
        target.shape,
        parent_sets=parent_sets,
    )
    ansatz = compressed.variational_ansatz(hamiltonian)
    initial_energy = ansatz.energy

    ansatz.run(nsweeps=1, tol=0.0)

    updates = ansatz.history[0]["updates"]
    energies = [initial_energy] + [update["energy"] for update in updates]
    assert np.all(np.diff(energies) <= 1.0e-12)
    assert any(update["accepted"] for update in updates)
    assert ansatz.parent_sets == parent_sets
    np.testing.assert_allclose(initial_energy, -2.0, atol=1.0e-13)
    np.testing.assert_allclose(ansatz.energy, -4.0, atol=1.0e-13)
    np.testing.assert_allclose(
        _fidelity(ansatz.state_vector(), target),
        1.0,
        atol=1.0e-13,
    )


def test_parent_rr_search_finds_crossed_bell_ties_from_nearest_graph():
    target = _crossed_bell_state()
    hamiltonian = _crossed_bell_hamiltonian()
    compressed = compress_physical_ties(
        np.ones(target.shape),
        target.shape,
        parent_sets=fixed_range_parent_sets(4, tie_range=1),
    )
    ansatz = compressed.variational_ansatz(hamiltonian)
    initial_energy = ansatz.energy

    ansatz.run_parent_search(max_parents=1, nsweeps=1, tensor_sweeps=0)

    updates = ansatz.graph_history[0]["graph_updates"]
    assert ansatz.parent_sets == ((2,), (3,), (3,))
    assert [(update["accepted"], update["graph_changed"]) for update in updates] == [
        (True, True),
        (True, True),
        (False, False),
    ]
    assert {trial["parents"] for trial in updates[0]["trials"]} == {
        (),
        (1,),
        (2,),
        (3,),
    }
    assert all(update["energy"] <= update["energy_before"] for update in updates)
    np.testing.assert_allclose(initial_energy, -2.0, atol=1.0e-13)
    np.testing.assert_allclose(ansatz.energy, -4.0, atol=1.0e-13)
    np.testing.assert_allclose(
        _fidelity(ansatz.state_vector(), target),
        1.0,
        atol=1.0e-13,
    )


def test_parent_rr_two_parent_search_captures_synergistic_tie():
    hamiltonian = -(
        _kron(_PAULI_Z, _PAULI_Z, _PAULI_Z)
        + _kron(_PAULI_X, _PAULI_X, _IDENTITY)
        + _kron(_PAULI_X, _IDENTITY, _PAULI_X)
    )
    seed = np.ones((2, 2, 2))
    compressed = compress_physical_ties(
        seed,
        seed.shape,
        parent_sets=fixed_range_parent_sets(3, tie_range=0),
    )
    one_parent = compressed.variational_ansatz(hamiltonian)
    two_parent = compressed.variational_ansatz(hamiltonian)

    one_parent.run_parent_search(max_parents=1, nsweeps=1, tensor_sweeps=0)
    two_parent.run_parent_search(max_parents=2, nsweeps=1, tensor_sweeps=0)

    assert all(len(parents) <= 1 for parents in one_parent.parent_sets)
    np.testing.assert_allclose(one_parent.energy, -2.0, atol=1.0e-13)
    site_zero_trials = two_parent.graph_history[0]["graph_updates"][0]["trials"]
    assert {trial["parents"] for trial in site_zero_trials} == {
        (),
        (1,),
        (2,),
        (1, 2),
    }
    assert two_parent.parent_sets[0] == (1, 2)
    np.testing.assert_allclose(two_parent.energy, -3.0, atol=1.0e-13)


def test_parent_rr_three_parent_search_captures_four_spin_parity():
    hamiltonian = -(
        _kron(_PAULI_Z, _PAULI_Z, _PAULI_Z, _PAULI_Z)
        + _kron(_PAULI_X, _PAULI_X, _IDENTITY, _IDENTITY)
        + _kron(_PAULI_X, _IDENTITY, _PAULI_X, _IDENTITY)
        + _kron(_PAULI_X, _IDENTITY, _IDENTITY, _PAULI_X)
    )
    seed = np.ones((2, 2, 2, 2))
    compressed = compress_physical_ties(
        seed,
        seed.shape,
        parent_sets=fixed_range_parent_sets(4, tie_range=0),
    )
    two_parent = compressed.variational_ansatz(hamiltonian)
    three_parent = compressed.variational_ansatz(hamiltonian)

    two_parent.run_parent_search(max_parents=2, nsweeps=1, tensor_sweeps=0)
    three_parent.run_parent_search(max_parents=3, nsweeps=1, tensor_sweeps=0)

    assert two_parent.energy > -4.0 + 1.0e-8
    assert three_parent.parent_sets[0] == (1, 2, 3)
    assert three_parent.factors[0].shape == (2, 2, 2, 2)
    np.testing.assert_allclose(three_parent.energy, -4.0, atol=1.0e-13)


def test_parent_rr_four_parent_search_captures_five_spin_parity():
    hamiltonian = -(
        _kron(_PAULI_Z, _PAULI_Z, _PAULI_Z, _PAULI_Z, _PAULI_Z)
        + _kron(_PAULI_X, _PAULI_X, _IDENTITY, _IDENTITY, _IDENTITY)
        + _kron(_PAULI_X, _IDENTITY, _PAULI_X, _IDENTITY, _IDENTITY)
        + _kron(_PAULI_X, _IDENTITY, _IDENTITY, _PAULI_X, _IDENTITY)
        + _kron(_PAULI_X, _IDENTITY, _IDENTITY, _IDENTITY, _PAULI_X)
    )
    seed = np.ones((2, 2, 2, 2, 2))
    compressed = compress_physical_ties(
        seed,
        seed.shape,
        parent_sets=fixed_range_parent_sets(5, tie_range=0),
    )
    three_parent = compressed.variational_ansatz(hamiltonian)
    four_parent = compressed.variational_ansatz(hamiltonian)

    three_parent.run_parent_search(max_parents=3, nsweeps=1, tensor_sweeps=0)
    four_parent.run_parent_search(max_parents=4, nsweeps=1, tensor_sweeps=0)

    assert three_parent.energy > -5.0 + 1.0e-8
    assert four_parent.parent_sets[0] == (1, 2, 3, 4)
    assert four_parent.factors[0].shape == (2, 2, 2, 2, 2)
    assert len(four_parent.graph_history[0]["graph_updates"][0]["trials"]) == 16
    np.testing.assert_allclose(four_parent.energy, -5.0, atol=1.0e-13)


def test_parent_rr_search_preserves_complex_tie_phase():
    hamiltonian = -(
        _kron(_PAULI_Z, _IDENTITY, _PAULI_Z)
        + _kron(_PAULI_X, _IDENTITY, _PAULI_Y)
        + _kron(_IDENTITY, _PAULI_X, _IDENTITY)
    )
    target = np.zeros((2, 2, 2), dtype=complex)
    target[0, :, 0] = 0.5
    target[1, :, 1] = 0.5j
    seed = np.ones((2, 2, 2))
    compressed = compress_physical_ties(
        seed,
        seed.shape,
        parent_sets=fixed_range_parent_sets(3, tie_range=1),
    )
    ansatz = compressed.variational_ansatz(hamiltonian)

    ansatz.run_parent_search(max_parents=1, nsweeps=1, tensor_sweeps=0)

    assert ansatz.parent_sets[0] == (2,)
    np.testing.assert_allclose(ansatz.energy, -3.0, atol=1.0e-13)
    np.testing.assert_allclose(
        _fidelity(ansatz.state_vector(), target),
        1.0,
        atol=1.0e-13,
    )


def test_parent_rr_no_improvement_leaves_state_unchanged():
    rng = np.random.default_rng(713)
    seed = rng.normal(size=(2, 2, 2)) + 1.0j * rng.normal(size=(2, 2, 2))
    compressed = compress_physical_ties(
        seed,
        seed.shape,
        parent_sets=fixed_range_parent_sets(3, tie_range=1),
    )
    ansatz = compressed.variational_ansatz(np.eye(8))
    factors_before = [factor.copy() for factor in ansatz.factors]
    terminal_before = ansatz.terminal.copy()
    parents_before = ansatz.parent_sets
    state_before = ansatz.state_vector().copy()

    update = ansatz.optimize_parent_set(0, [(), (1,), (2,)])

    assert not update["accepted"]
    assert not update["graph_changed"]
    assert ansatz.parent_sets == parents_before
    for factor, before in zip(ansatz.factors, factors_before):
        np.testing.assert_array_equal(factor, before)
    np.testing.assert_array_equal(ansatz.terminal, terminal_before)
    np.testing.assert_array_equal(ansatz.state_vector(), state_before)
    np.testing.assert_allclose(ansatz.energy, 1.0, atol=1.0e-13)
    with pytest.raises(ValueError, match="exceeds max_parents"):
        ansatz.run_parent_search(max_parents=0)


def test_parent_rr_metric_filtering_current_graph_is_safe_noop():
    compressed = compress_physical_ties(
        np.ones((2, 2, 2)),
        (2, 2, 2),
        parent_sets=((1, 2), ()),
    )
    ansatz = compressed.variational_ansatz(np.eye(8))
    state_before = ansatz.state_vector().copy()
    parents_before = ansatz.parent_sets

    update = ansatz.optimize_parent_set(0, [()], metric_tol=0.2)

    assert not update["accepted"]
    assert ansatz.parent_sets == parents_before
    np.testing.assert_array_equal(ansatz.state_vector(), state_before)


def test_parent_rr_run_rolls_back_if_a_later_site_raises(monkeypatch):
    target = _crossed_bell_state()
    compressed = compress_physical_ties(
        np.ones(target.shape),
        target.shape,
        parent_sets=fixed_range_parent_sets(4, tie_range=1),
    )
    ansatz = compressed.variational_ansatz(_crossed_bell_hamiltonian())
    state_before = ansatz.state_vector().copy()
    parents_before = ansatz.parent_sets
    energy_before = ansatz.energy
    original_update = ansatz.optimize_parent_set

    def fail_after_first_move(site, *args, **kwargs):
        if site == 1:
            raise RuntimeError("injected parent-search failure")
        return original_update(site, *args, **kwargs)

    monkeypatch.setattr(ansatz, "optimize_parent_set", fail_after_first_move)
    with pytest.raises(RuntimeError, match="injected"):
        ansatz.run_parent_search(max_parents=1, nsweeps=1, tensor_sweeps=0)

    assert ansatz.parent_sets == parents_before
    np.testing.assert_array_equal(ansatz.state_vector(), state_before)
    np.testing.assert_allclose(ansatz.energy, energy_before, atol=1.0e-13)


def test_relaxed_parent_search_escapes_uphill_local_rr_barrier():
    raw = np.array(
        [1, -1, -1, -3, -3, -3, -2, -3, -1, -2, -2, 2, -2, 3, -3, 2],
        dtype=float,
    )
    target = raw / np.linalg.norm(raw)
    hamiltonian = -np.outer(target, target)
    dims = (2, 2, 2, 2)
    compressed = compress_physical_ties(
        target,
        dims,
        parent_sets=fixed_range_parent_sets(4, tie_range=1),
    )
    ansatz = compressed.variational_ansatz(hamiltonian)
    ansatz.run(nsweeps=30, tol=0.0)
    energy_before = ansatz.energy
    greedy = ansatz.copy()
    narrow = ansatz.copy()
    for source, child in zip(ansatz.factors, narrow.factors):
        assert not np.shares_memory(source, child)
    assert not np.shares_memory(ansatz.terminal, narrow.terminal)

    greedy.run_parent_search(max_parents=1, nsweeps=1, tensor_sweeps=0)
    narrow_update = narrow.optimize_parent_graph_relaxed(
        max_parents=1,
        candidate_budget=4,
        per_site_candidates=2,
        tensor_sweeps=1,
    )
    ansatz.run_relaxed_parent_search(
        max_parents=1,
        nsweeps=1,
        candidate_budget=6,
        per_site_candidates=2,
        tensor_sweeps=1,
    )

    assert sum(sweep["graph_changes"] for sweep in greedy.graph_history) == 0
    assert not narrow_update["accepted"]
    np.testing.assert_allclose(narrow.energy, energy_before, atol=1.0e-13)
    update = ansatz.relaxed_graph_history[0]
    chosen = next(
        branch
        for branch in update["branches"]
        if branch["site"] == update["site"]
        and branch["parents"] == update["parents"]
    )
    assert update["accepted"]
    assert not ansatz.converged
    assert update["trial_candidates"] <= 6
    assert update["site"] == 0
    assert update["parents"] == (3,)
    assert chosen["forced_energy"] > energy_before
    assert chosen["relaxed_energy"] < energy_before
    assert ansatz.parent_sets == ((3,), (2,), (3,))
    assert ansatz.energy < -0.88


def test_relaxed_parent_search_no_improvement_preserves_original():
    rng = np.random.default_rng(991)
    seed = rng.normal(size=(2, 2, 2)) + 1.0j * rng.normal(size=(2, 2, 2))
    compressed = compress_physical_ties(
        seed,
        seed.shape,
        parent_sets=fixed_range_parent_sets(3, tie_range=1),
    )
    ansatz = compressed.variational_ansatz(np.eye(8))
    state_before = ansatz.state_vector().copy()
    factors_before = [factor.copy() for factor in ansatz.factors]
    terminal_before = ansatz.terminal.copy()
    parents_before = ansatz.parent_sets

    update = ansatz.optimize_parent_graph_relaxed(
        max_parents=1,
        candidate_budget=3,
        tensor_sweeps=1,
    )

    assert not update["accepted"]
    assert ansatz.parent_sets == parents_before
    for factor, before in zip(ansatz.factors, factors_before):
        np.testing.assert_array_equal(factor, before)
    np.testing.assert_array_equal(ansatz.terminal, terminal_before)
    np.testing.assert_array_equal(ansatz.state_vector(), state_before)


def test_relaxed_parent_search_rolls_back_after_late_failure(monkeypatch):
    raw = np.array(
        [1, -1, -1, -3, -3, -3, -2, -3, -1, -2, -2, 2, -2, 3, -3, 2],
        dtype=float,
    )
    target = raw / np.linalg.norm(raw)
    hamiltonian = -np.outer(target, target)
    compressed = compress_physical_ties(
        target,
        (2, 2, 2, 2),
        parent_sets=fixed_range_parent_sets(4, tie_range=1),
    )
    ansatz = compressed.variational_ansatz(hamiltonian)
    ansatz.run(nsweeps=30, tol=0.0)
    factors_before = [factor.copy() for factor in ansatz.factors]
    terminal_before = ansatz.terminal.copy()
    parents_before = ansatz.parent_sets
    energy_before = ansatz.energy
    history_length = len(ansatz.history)
    ansatz.relaxed_graph_history = [{"sentinel": True}]
    ansatz.converged = True
    real_update = ansatz.optimize_parent_graph_relaxed
    calls = 0

    def fail_second_round(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected relaxed-search failure")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(ansatz, "optimize_parent_graph_relaxed", fail_second_round)
    with pytest.raises(RuntimeError, match="injected"):
        ansatz.run_relaxed_parent_search(
            max_parents=1,
            nsweeps=2,
            candidate_budget=6,
            per_site_candidates=2,
            tensor_sweeps=1,
        )

    assert ansatz.parent_sets == parents_before
    assert len(ansatz.history) == history_length
    assert ansatz.relaxed_graph_history == [{"sentinel": True}]
    assert ansatz.converged
    for factor, before in zip(ansatz.factors, factors_before):
        np.testing.assert_array_equal(factor, before)
    np.testing.assert_array_equal(ansatz.terminal, terminal_before)
    np.testing.assert_allclose(ansatz.energy, energy_before, atol=1.0e-13)


def test_relaxed_parent_update_validates_branch_before_atomic_commit(monkeypatch):
    raw = np.array(
        [1, -1, -1, -3, -3, -3, -2, -3, -1, -2, -2, 2, -2, 3, -3, 2],
        dtype=float,
    )
    target = raw / np.linalg.norm(raw)
    hamiltonian = -np.outer(target, target)
    compressed = compress_physical_ties(
        target,
        (2, 2, 2, 2),
        parent_sets=fixed_range_parent_sets(4, tie_range=1),
    )
    ansatz = compressed.variational_ansatz(hamiltonian)
    ansatz.run(nsweeps=30, tol=0.0)
    state_before = ansatz.state_vector().copy()
    factors_before = [factor.copy() for factor in ansatz.factors]
    terminal_before = ansatz.terminal.copy()
    parents_before = ansatz.parent_sets
    energy_before = ansatz.energy
    real_branch = ansatz._relaxed_parent_branch

    def poison_branch(*args, **kwargs):
        branch, forced_energy = real_branch(*args, **kwargs)

        def fail_validation():
            raise RuntimeError("injected branch-validation failure")

        branch.expectation = fail_validation
        return branch, forced_energy

    monkeypatch.setattr(ansatz, "_relaxed_parent_branch", poison_branch)
    with pytest.raises(RuntimeError, match="validation"):
        ansatz.optimize_parent_graph_relaxed(
            max_parents=1,
            candidate_budget=6,
            per_site_candidates=2,
            tensor_sweeps=1,
        )

    assert ansatz.parent_sets == parents_before
    for factor, before in zip(ansatz.factors, factors_before):
        np.testing.assert_array_equal(factor, before)
    np.testing.assert_array_equal(ansatz.terminal, terminal_before)
    np.testing.assert_array_equal(ansatz.state_vector(), state_before)
    np.testing.assert_allclose(ansatz.energy, energy_before, atol=1.0e-13)


def test_every_dense_variational_update_is_energy_nonincreasing():
    rng = np.random.default_rng(93)
    dims = (2, 2, 2)
    seed = rng.normal(size=dims) + 1.0j * rng.normal(size=dims)
    raw_hamiltonian = rng.normal(size=(8, 8)) + 1.0j * rng.normal(size=(8, 8))
    hamiltonian = raw_hamiltonian + raw_hamiltonian.T.conj()
    compressed = compress_physical_ties(
        seed,
        dims,
        parent_sets=fixed_range_parent_sets(len(dims), tie_range=1),
    )
    ansatz = compressed.variational_ansatz(hamiltonian)
    parent_sets = ansatz.parent_sets

    for variable in range(ansatz.nvariables):
        energy_before = ansatz.energy
        update = ansatz.optimize_variable(variable)
        np.testing.assert_allclose(update["energy_before"], energy_before, atol=1.0e-13)
        np.testing.assert_allclose(update["energy"], ansatz.energy, atol=1.0e-13)
        assert update["energy"] <= energy_before + 1.0e-11
        assert update["metric_rank"] <= update["raw_dim"]

    assert ansatz.parent_sets == parent_sets


def test_complex_hamiltonian_optimizes_physical_tie_phase():
    hamiltonian = -(
        _kron(_PAULI_Z, _PAULI_Z)
        + _kron(_PAULI_X, _PAULI_Y)
    )
    target = np.array([1.0, 0.0, 0.0, 1.0j], dtype=complex) / np.sqrt(2.0)
    compressed = compress_physical_ties(
        np.ones((2, 2), dtype=complex),
        (2, 2),
        parent_sets=((1,),),
    )
    ansatz = compressed.variational_ansatz(hamiltonian)

    update = ansatz.optimize_variable(0)

    assert update["accepted"]
    assert update["energy"] < update["energy_before"]
    np.testing.assert_allclose(ansatz.energy, -2.0, atol=1.0e-13)
    np.testing.assert_allclose(
        _fidelity(ansatz.state_vector(), target),
        1.0,
        atol=1.0e-13,
    )


def test_variational_constructor_balances_extreme_scalar_gauge():
    factor = 1.0e200 * np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    terminal = 1.0e-200 * np.array([2.0, 1.0], dtype=complex)
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0])

    ansatz = VariationalPhysicalTie(
        hamiltonian,
        (2, 2),
        [factor],
        ((1,),),
        terminal,
    )
    update = ansatz.optimize_variable(0)

    np.testing.assert_allclose(ansatz.norm(), 1.0, atol=1.0e-13)
    assert update["accepted"]
    np.testing.assert_allclose(ansatz.energy, 0.0, atol=1.0e-13)

    cancelling = VariationalPhysicalTie(
        np.zeros((1, 1)),
        (1, 1, 1, 1, 1),
        [np.array([scale]) for scale in (1.0e300, 1.0e300, 1.0e-300, 1.0e-300)],
        ((), (), (), ()),
        np.ones(1),
    )
    np.testing.assert_allclose(cancelling.norm(), 1.0, atol=1.0e-14)

    for scale in (1.0e100, 1.0e-100):
        uniform = VariationalPhysicalTie(
            np.zeros((1, 1)),
            (1, 1, 1, 1, 1),
            [np.array([scale]) for _ in range(4)],
            ((), (), (), ()),
            np.array([scale]),
        )
        np.testing.assert_allclose(uniform.norm(), 1.0, atol=1.0e-14)


def test_variational_constructor_normalizes_tiny_nonzero_state():
    epsilon = 1.0e-20
    factor = np.array([[1.0, epsilon], [0.0, 0.0]])
    terminal = np.array([epsilon, 1.0])

    ansatz = VariationalPhysicalTie(
        np.zeros((4, 4)),
        (2, 2),
        [factor],
        ((1,),),
        terminal,
    )

    np.testing.assert_allclose(ansatz.norm(), 1.0, atol=1.0e-14)


@pytest.mark.parametrize(
    ("factors", "terminal"),
    [
        ([np.array([[np.inf]])], np.ones(1)),
        ([np.ones((1, 1))], np.array([np.nan])),
    ],
)
def test_variational_constructor_rejects_nonfinite_tensors(factors, terminal):
    with pytest.raises(ValueError, match="finite"):
        VariationalPhysicalTie(
            np.zeros((1, 1)),
            (1, 1),
            factors,
            ((1,),),
            terminal,
        )


@pytest.mark.parametrize("scale", [np.nan, np.inf])
def test_variational_perturbation_rejects_nonfinite_scale(scale):
    ansatz = VariationalPhysicalTie(
        np.zeros((1, 1)),
        (1,),
        [],
        (),
        np.ones(1),
    )
    with pytest.raises(ValueError, match="finite"):
        ansatz.perturb(scale)


def test_variational_constructor_rejects_invalid_parent_graph():
    with pytest.raises(ValueError, match="future site"):
        VariationalPhysicalTie(
            np.eye(4),
            (2, 2),
            [np.ones((2, 2))],
            ((-1,),),
            np.ones(2),
        )

    valid = VariationalPhysicalTie(
        np.eye(4),
        (2, 2),
        [np.ones((2, 2))],
        ((1,),),
        np.ones(2),
    )
    with pytest.raises(ValueError, match="metric_tol"):
        valid.optimize_variable(0, metric_tol=-1.0)
    with pytest.raises(ValueError, match="candidate_budget"):
        valid.run_relaxed_parent_search(
            max_parents=1,
            nsweeps=0,
            candidate_budget=0,
        )
