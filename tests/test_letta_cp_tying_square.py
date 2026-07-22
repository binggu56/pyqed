import numpy as np
import pytest

import examples.mps.adaptive_cp_letta_j1j2_square as square_benchmark
from pyqed.letta import CPTiedLETTA


def test_square_benchmark_requires_two_spins():
    with pytest.raises(ValueError, match="requires at least two spins"):
        square_benchmark.benchmark_square_j1_j2(
            nrows=1,
            ncols=1,
            modes=("fixed-chain-r2",),
        )


def test_square_j1_j2_snake_graph_has_expected_bonds_and_parent_degree():
    nearest, diagonals = square_benchmark.square_j1_j2_bonds(3, 4)
    parents = square_benchmark.parent_sets_from_edges(
        12,
        set(nearest) | set(diagonals),
    )

    assert len(nearest) == 17
    assert len(diagonals) == 12
    assert max(map(len, parents)) == 4
    assert parents == (
        (1, 6, 7),
        (2, 5, 6, 7),
        (3, 4, 5, 6),
        (4, 5),
        (5, 10, 11),
        (6, 9, 10, 11),
        (7, 8, 9, 10),
        (8, 9),
        (9,),
        (10,),
        (11,),
        (),
    )


def test_sparse_heisenberg_two_spin_spectrum_and_basis_order():
    hamiltonian = square_benchmark.sparse_heisenberg_hamiltonian(
        2,
        ((0, 1, 1.0),),
    ).toarray()

    np.testing.assert_allclose(
        np.linalg.eigvalsh(hamiltonian),
        (-0.75, 0.25, 0.25, 0.25),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(hamiltonian[1, 2], 0.5, atol=1.0e-14)
    three_site = square_benchmark.sparse_heisenberg_hamiltonian(
        3,
        ((0, 1, 1.0),),
    ).toarray()
    np.testing.assert_allclose(three_site[4, 2], 0.5, atol=1.0e-14)
    np.testing.assert_allclose(three_site[1, 2], 0.0, atol=1.0e-14)


def test_graph_diagnostics_separate_inherited_and_discovered_edges():
    diagnostics = square_benchmark._graph_diagnostics(
        ((1, 2), (), ()),
        {(0, 1), (0, 2), (1, 2)},
        baseline_edges={(0, 1)},
    )

    assert diagnostics["physical_overlap"] == 2
    assert diagnostics["added_edges"] == 1
    assert diagnostics["added_physical_overlap"] == 1
    assert diagnostics["added_graph_precision"] == 1.0
    assert diagnostics["added_graph_recall"] == 0.5


def test_square_benchmark_diagonalizes_only_after_adaptive_solve(monkeypatch):
    solved = {"done": False}
    original_run = CPTiedLETTA.run_residual_adaptive
    original_eigsh = square_benchmark.eigsh

    def tracked_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        solved["done"] = True
        return result

    def guarded_eigsh(*args, **kwargs):
        assert solved["done"]
        return original_eigsh(*args, **kwargs)

    monkeypatch.setattr(CPTiedLETTA, "run_residual_adaptive", tracked_run)
    monkeypatch.setattr(square_benchmark, "eigsh", guarded_eigsh)

    result = square_benchmark.benchmark_square_j1_j2(
        nrows=2,
        ncols=2,
        fixed_sweeps=1,
        adaptive_cycles=1,
        initial_sweeps=1,
        branch_sweeps=1,
        candidate_budget=1,
        max_parents=2,
        max_tie_rank=2,
        modes=("adaptive-graph",),
        seed=3,
    )

    assert solved["done"]
    assert len(result["rows"]) == 1
    assert result["rows"][0]["tie_ranks"] == (2, 2, 2, 2)
    assert np.isfinite(result["rows"][0]["energy_error"])


def test_zero_j2_excludes_diagonals_from_physical_graph():
    result = square_benchmark.benchmark_square_j1_j2(
        nrows=2,
        ncols=2,
        j2=0.0,
        fixed_sweeps=0,
        modes=("fixed-chain-r2",),
        seed=3,
    )

    assert result["rows"][0]["physical_edges"] == 4


def test_staged_history_has_global_cycle_indices():
    result = square_benchmark.benchmark_square_j1_j2(
        nrows=2,
        ncols=2,
        fixed_sweeps=0,
        adaptive_cycles=2,
        initial_sweeps=0,
        branch_sweeps=0,
        candidate_budget=1,
        max_parents=2,
        max_tie_rank=3,
        probe_noise=0.0,
        modes=("adaptive-staged",),
        seed=3,
    )

    history = result["rows"][0]["adaptive_history"]
    assert [record["global_cycle"] for record in history] == list(
        range(len(history))
    )
    assert tuple(record["stage"] for record in history) == ("graph", "rank")
