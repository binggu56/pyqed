import numpy as np
import pytest
import examples.mps.adaptive_cp_letta_annni as cp_benchmark

from examples.mps.adaptive_cp_letta_annni import (
    benchmark_cp_rank_sweep,
    quantum_mutual_information_parent_sets,
    select_adaptive_tie_ranks,
)
from pyqed.letta import CPTiedLETTA
from pyqed.letta.physical_tying import compress_physical_ties, fixed_range_parent_sets


_IDENTITY = np.eye(2)
_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_PAULI_Z = np.diag([1.0, -1.0])


def _projector_hamiltonian(state):
    state = np.asarray(state).reshape(-1)
    state = state / np.linalg.norm(state)
    return -np.outer(state, state.conj())


def test_cp_tied_letta_rank_two_reconstructs_ghz_physical_tie():
    target = np.zeros(8, dtype=complex)
    target[0] = 1.0 / np.sqrt(2.0)
    target[-1] = 1.0j / np.sqrt(2.0)
    compressed = compress_physical_ties(
        target,
        (2, 2, 2),
        parent_sets=((1, 2), (2,)),
    )

    ansatz = CPTiedLETTA.from_physical_tie_state(
        compressed,
        _projector_hamiltonian(target),
        tie_ranks=2,
        bond_dim=1,
        seed=4,
    )

    np.testing.assert_allclose(ansatz.fidelity(target), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(ansatz.energy, -1.0, atol=1.0e-12)


def test_cp_tied_letta_joint_cp_and_virtual_sweep_is_monotone():
    rng = np.random.default_rng(12)
    target = rng.normal(size=16)
    target /= np.linalg.norm(target)
    dims = (2, 2, 2, 2)
    compressed = compress_physical_ties(
        target,
        dims,
        parent_sets=fixed_range_parent_sets(4, tie_range=1),
    )
    ansatz = CPTiedLETTA.from_physical_tie_state(
        compressed,
        _projector_hamiltonian(target),
        tie_ranks=2,
        bond_dim=2,
        seed=7,
    )
    initial = ansatz.energy

    assert any(np.any(np.abs(core[1:]) > 0.0) for core in ansatz.cores[1:-1])

    ansatz.run(nsweeps=2, tol=0.0)

    energies = [initial] + [record["energy"] for record in ansatz.history]
    assert np.all(np.diff(energies) <= 1.0e-11)
    assert any(
        update.kind == "core" and update.accepted
        for record in ansatz.history
        for update in record["updates"]
    )
    assert ansatz.energy <= initial + 1.0e-12


def test_cp_tied_letta_rank_growth_preserves_state_before_relaxation():
    rng = np.random.default_rng(21)
    target = rng.normal(size=16)
    dims = (2, 2, 2, 2)
    compressed = compress_physical_ties(
        target,
        dims,
        parent_sets=fixed_range_parent_sets(4, tie_range=2),
    )
    ansatz = CPTiedLETTA.from_physical_tie_state(
        compressed,
        _projector_hamiltonian(target),
        tie_ranks=1,
        bond_dim=2,
        virtual_noise=1.0e-3,
        seed=9,
    )
    before = ansatz.state_vector(normalize=True)
    energy_before = ansatz.energy

    ansatz.expand_tie_ranks(3, seed=10)

    np.testing.assert_allclose(
        abs(np.vdot(before, ansatz.state_vector(normalize=True))) ** 2,
        1.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(ansatz.energy, energy_before, atol=1.0e-12)
    assert ansatz.tie_ranks == (3, 3, 3, 3)


def test_direct_cp_tied_letta_can_optimize_complex_state():
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    ansatz = CPTiedLETTA(-pauli_y, (2,), (), tie_ranks=2, seed=3)

    ansatz.run(nsweeps=2, tol=0.0)

    assert np.iscomplexobj(ansatz.cores[0])
    np.testing.assert_allclose(ansatz.energy, -1.0, atol=1.0e-12)


def test_cp_tied_letta_copy_preserves_rng_state():
    ansatz = CPTiedLETTA(-_PAULI_X, (2,), (), tie_ranks=1, seed=17)

    clone = ansatz.copy()

    np.testing.assert_allclose(ansatz.rng.normal(size=8), clone.rng.normal(size=8))


@pytest.mark.parametrize("name", ("residual_tol", "energy_tol", "metric_tol"))
def test_residual_adaptive_rejects_negative_tolerances(name):
    ansatz = CPTiedLETTA(-_PAULI_X, (2,), (), tie_ranks=1, seed=17)

    with pytest.raises(ValueError, match=f"{name} must be finite and nonnegative"):
        ansatz.run_residual_adaptive(
            max_parents=0,
            max_tie_rank=1,
            ncycles=0,
            **{name: -1.0},
        )


@pytest.mark.parametrize("proposal_kinds", ((), ("unknown",)))
def test_residual_adaptive_rejects_invalid_proposal_kinds(proposal_kinds):
    ansatz = CPTiedLETTA(-_PAULI_X, (2,), (), tie_ranks=1, seed=17)

    with pytest.raises(ValueError, match="proposal_kinds must be a nonempty subset"):
        ansatz.run_residual_adaptive(
            max_parents=0,
            max_tie_rank=1,
            ncycles=0,
            proposal_kinds=proposal_kinds,
        )


def test_qmi_parent_selector_finds_crossed_bell_partners():
    target = np.zeros((2, 2, 2, 2), dtype=complex)
    for x0 in range(2):
        for x1 in range(2):
            target[x0, x1, x0, x1] = 0.5

    parents = quantum_mutual_information_parent_sets(target, target.shape, 1)

    assert parents == ((2,), (3,), (3,))


def test_cp_letta_annni_rank_sweep_smoke():
    rows = benchmark_cp_rank_sweep(
        nsites=4,
        nearest=1.0,
        next_nearest=0.8,
        field=0.8,
        max_parents=2,
        bond_dim=2,
        ranks=(1, 2),
        selectors=("gram",),
        variational_sweeps=1,
        seed=5,
    )

    assert [row["rank"] for row in rows] == [1, 2]
    assert all(np.isfinite(row["energy_error"]) for row in rows)
    assert all(row["doublet_fidelity"] >= row["fidelity"] for row in rows)


def test_adaptive_tie_rank_selection_stops_after_useful_growth():
    target = np.zeros(8)
    target[0] = target[-1] = 1.0 / np.sqrt(2.0)
    compressed = compress_physical_ties(
        target,
        (2, 2, 2),
        parent_sets=((1, 2), (2,)),
    )

    ranks, history = select_adaptive_tie_ranks(
        compressed,
        _projector_hamiltonian(target),
        target,
        candidate_ranks=(1, 2, 4),
        objective="fidelity",
        gain_tolerance=1.0e-12,
        seed=2,
    )

    assert ranks[0] == 2
    assert max(ranks) == 2
    assert ranks[-1] == 1
    np.testing.assert_allclose(history[-1]["score"], 1.0, atol=1.0e-12)


def test_residual_adaptive_rank_growth_finds_ghz_without_target_seed():
    target = np.zeros(8, dtype=complex)
    target[0] = 1.0 / np.sqrt(2.0)
    target[-1] = 1.0j / np.sqrt(2.0)
    ansatz = CPTiedLETTA(
        _projector_hamiltonian(target),
        (2, 2, 2),
        ((1, 2), (2,), ()),
        tie_ranks=1,
        bond_dim=1,
        seed=0,
    )

    ansatz.run_residual_adaptive(
        max_parents=2,
        max_tie_rank=2,
        ncycles=3,
        initial_sweeps=2,
        branch_sweeps=5,
        candidate_budget=3,
        proposal_kinds=("rank",),
        seed=0,
    )

    np.testing.assert_allclose(ansatz.energy, -1.0, atol=1.0e-12)
    np.testing.assert_allclose(ansatz.fidelity(target), 1.0, atol=1.0e-12)
    assert ansatz.tie_ranks == (2, 1, 1)
    assert ansatz.parent_sets == ((1, 2), (2,), ())
    assert ansatz.adaptive_history[0]["chosen"]["kind"] == "rank"
    assert ansatz.energy_residual()[2] < 1.0e-10


def test_residual_adaptive_graph_finds_crossed_bell_without_target_seed():
    hamiltonian = -(
        np.kron(np.kron(np.kron(_PAULI_X, _IDENTITY), _PAULI_X), _IDENTITY)
        + np.kron(np.kron(np.kron(_PAULI_Z, _IDENTITY), _PAULI_Z), _IDENTITY)
        + np.kron(np.kron(np.kron(_IDENTITY, _PAULI_X), _IDENTITY), _PAULI_X)
        + np.kron(np.kron(np.kron(_IDENTITY, _PAULI_Z), _IDENTITY), _PAULI_Z)
    )
    ansatz = CPTiedLETTA(
        hamiltonian,
        (2, 2, 2, 2),
        ((1,), (2,), (3,), ()),
        tie_ranks=1,
        bond_dim=1,
        seed=0,
    )

    ansatz.run_residual_adaptive(
        max_parents=1,
        max_tie_rank=2,
        ncycles=5,
        initial_sweeps=3,
        branch_sweeps=5,
        candidate_budget=5,
        per_site_graph_candidates=3,
        seed=0,
    )

    np.testing.assert_allclose(ansatz.energy, -4.0, atol=1.0e-12)
    assert ansatz.parent_sets[:2] == ((2,), (3,))
    assert ansatz.tie_ranks[:2] == (2, 2)
    assert all(len(step["trials"]) <= 5 for step in ansatz.adaptive_history)
    assert ansatz.adaptive_history[0]["scheduled_kinds"][:2] == (
        "graph",
        "rank",
    )
    accepted_energies = [
        step["energy"] for step in ansatz.adaptive_history if step["accepted"]
    ]
    assert np.all(np.diff(accepted_energies) < 0.0)
    assert all(
        step["energy"] < step["energy_before"]
        for step in ansatz.adaptive_history
        if step["accepted"]
    )
    assert ansatz.energy_residual()[2] < 1.0e-10


def test_residual_annni_benchmark_diagonalizes_only_after_solve(monkeypatch):
    solved = {"done": False}
    original_run = CPTiedLETTA.run_residual_adaptive
    original_eigh = cp_benchmark.eigh

    def tracked_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        solved["done"] = True
        return result

    def guarded_eigh(*args, **kwargs):
        assert solved["done"]
        return original_eigh(*args, **kwargs)

    monkeypatch.setattr(CPTiedLETTA, "run_residual_adaptive", tracked_run)
    monkeypatch.setattr(cp_benchmark, "eigh", guarded_eigh)

    row = cp_benchmark.benchmark_residual_adaptive(
        nsites=3,
        nearest=1.0,
        next_nearest=0.8,
        field=0.8,
        initial_tie_range=1,
        initial_tie_rank=1,
        bond_dim=1,
        max_parents=1,
        max_tie_rank=2,
        ncycles=1,
        initial_sweeps=1,
        branch_sweeps=1,
        candidate_budget=1,
        per_site_graph_candidates=1,
        seed=4,
    )

    assert solved["done"]
    assert np.isfinite(row["energy_error"])
