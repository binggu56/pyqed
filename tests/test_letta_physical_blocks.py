import numpy as np
import pytest

from pyqed.letta.core import _lowest_generalized_eigenpair
from pyqed.letta.frontier_tying import FrontierTiedLETTA
from pyqed.letta.local_terms import LocalHamiltonian, LocalTerm
from pyqed.letta.physical_blocks import (
    PhysicalBlockGeneralizedProblem,
    PhysicalBlockLayout,
    PhysicalBlockLinearOperator,
    hamiltonian_physical_connectivity,
)


def _exchange():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def _frustrated_graph_state(seed=17):
    # Row-major 2 x 3 square with both diagonals in every plaquette.
    nearest = (
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (0, 3),
        (1, 4),
        (2, 5),
    )
    diagonals = ((0, 4), (1, 3), (1, 5), (2, 4))
    exchange = _exchange()
    terms = tuple(LocalTerm(edge, exchange) for edge in nearest) + tuple(
        LocalTerm(edge, 0.5 * exchange) for edge in diagonals
    )
    hamiltonian = LocalHamiltonian((2,) * 6, terms)
    parents = ((1, 3), (2, 4), (5,), (4,), (5,), ())
    return FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        seed=seed,
    )


def test_physical_block_layout_preserves_native_tensor_flattening():
    layout = PhysicalBlockLayout((2, 3, 2, 4))
    vector = np.arange(layout.size)
    blocks = layout.as_blocks(vector)

    assert blocks.shape == (8, 6)
    assert layout.configurations[5] == (1, 1)
    np.testing.assert_array_equal(layout.from_blocks(blocks), vector)
    np.testing.assert_array_equal(
        blocks[5], vector.reshape(6, 8)[:, 5]
    )


def test_block_linear_operator_applies_only_supplied_connections():
    layout = PhysicalBlockLayout((1, 2, 2, 2))
    calls = []
    pairs = ((0, 0), (1, 2), (2, 1), (3, 3))

    def factory(row, column):
        calls.append((row, column))
        return (1 + row + 2 * column) * np.eye(layout.virtual_size)

    operator = PhysicalBlockLinearOperator.from_block_factory(
        layout, pairs, factory
    )
    vector = np.linspace(-0.4, 0.9, layout.size)

    assert tuple(calls) == pairs
    assert operator.stored_elements == len(pairs) * layout.virtual_size**2
    np.testing.assert_allclose(operator.matvec(vector), operator.to_dense() @ vector)
    np.testing.assert_allclose(
        operator.rmatvec(vector), operator.to_dense().T.conj() @ vector
    )


def test_connectivity_is_generated_from_local_transitions():
    nsites = 9
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = LocalHamiltonian(
        (2,) * nsites,
        tuple(LocalTerm((site,), flip) for site in range(nsites)),
    )

    pairs = hamiltonian_physical_connectivity(
        hamiltonian,
        tuple(range(nsites)),
    )

    assert len(pairs) == nsites * 2**nsites
    assert all(row != column for row, column in pairs)


def test_frustrated_graph_physical_blocks_match_dense_local_problem():
    state = _frustrated_graph_state()
    site = 0
    metric, hamiltonian = state.local_operators(site)
    pairs = hamiltonian_physical_connectivity(
        state.hamiltonian, state.physical_sites[site]
    )
    problem = PhysicalBlockGeneralizedProblem.from_dense(
        metric,
        hamiltonian,
        state.tensors[site].shape,
        hamiltonian_pairs=pairs,
        omitted_atol=2.0e-12,
    )

    assert problem.layout.nblocks == 8
    assert len(problem.metric.connected_pairs) == 8
    assert len(problem.hamiltonian.connected_pairs) < 8**2
    np.testing.assert_allclose(problem.metric.to_dense(), metric, atol=2.0e-12)
    np.testing.assert_allclose(
        problem.hamiltonian.to_dense(), hamiltonian, atol=2.0e-12
    )

    rng = np.random.default_rng(23)
    vector = rng.normal(size=metric.shape[0]) + 1.0j * rng.normal(
        size=metric.shape[0]
    )
    np.testing.assert_allclose(problem.metric @ vector, metric @ vector, atol=2.0e-12)
    np.testing.assert_allclose(
        problem.hamiltonian @ vector, hamiltonian @ vector, atol=2.0e-12
    )


def test_block_generalized_davidson_matches_dense_local_eigenpair():
    state = _frustrated_graph_state(seed=31)
    site = 0
    metric, hamiltonian = state.local_operators(site)
    problem = PhysicalBlockGeneralizedProblem.from_frontier_state(
        state, site, omitted_atol=2.0e-12
    )
    reference_energy, _reference_vector = _lowest_generalized_eigenpair(
        hamiltonian, metric, metric_tol=1.0e-12
    )
    energy, vector, diagnostics = problem.solve(
        state.tensors[site],
        tol=1.0e-11,
        metric_tol=1.0e-12,
        maxiter=300,
        max_subspace=16,
        random_seed=5,
    )

    assert diagnostics.converged, diagnostics.message
    np.testing.assert_allclose(energy, reference_energy, atol=2.0e-9)
    residual = problem.hamiltonian @ vector - energy * (problem.metric @ vector)
    assert np.linalg.norm(residual) < 2.0e-9


def test_component_solver_finds_an_unoccupied_lower_component():
    metric = np.eye(2)
    hamiltonian = np.diag([1.0, -1.0])
    problem = PhysicalBlockGeneralizedProblem.from_dense(
        metric,
        hamiltonian,
        (1, 1, 2),
        hamiltonian_pairs=((0, 0), (1, 1)),
        omitted_atol=0.0,
    )

    energy, vector, diagnostics = problem.solve(
        np.array([1.0, 0.0]),
        tol=1.0e-13,
        metric_tol=1.0e-13,
    )

    np.testing.assert_allclose(energy, -1.0, atol=1.0e-14)
    np.testing.assert_allclose(np.abs(vector), [0.0, 1.0], atol=1.0e-14)
    assert diagnostics.component_sizes == (1, 1)
    assert diagnostics.selected_component == 1
    assert diagnostics.positive_metric_components == 2


def test_component_solver_verifies_only_the_retained_metric_range():
    metric = np.diag([1.0, 1.0e-14])
    hamiltonian = np.array([[2.0, 3.0e-6], [3.0e-6, 0.0]])
    problem = PhysicalBlockGeneralizedProblem.from_dense(
        metric,
        hamiltonian,
        (1, 2, 1),
        hamiltonian_pairs=((0, 0),),
        omitted_atol=0.0,
    )

    energy, vector, diagnostics = problem.solve(
        np.array([1.0, 0.0]),
        tol=1.0e-10,
        metric_tol=1.0e-12,
    )

    np.testing.assert_allclose(energy, 2.0, atol=1.0e-14)
    np.testing.assert_allclose(vector, [1.0, 0.0], atol=1.0e-14)
    assert diagnostics.converged, diagnostics.message
    assert diagnostics.residual_norm < 1.0e-14
    assert diagnostics.reconstructed_residual_norm < 1.0e-14
    np.testing.assert_allclose(diagnostics.full_residual_norm, 3.0e-6)


@pytest.mark.parametrize("frontier_backend", ["compressed", "identity_block"])
def test_frontier_block_problem_matches_dense_without_calling_hole_matrix(
    monkeypatch,
    frontier_backend,
):
    initial = _frustrated_graph_state(seed=43)
    state = FrontierTiedLETTA(
        initial.hamiltonian,
        initial.dims,
        initial.parent_sets,
        bond_dim=initial.bond_dim,
        tensors=initial.tensors,
        frontier_backend=frontier_backend,
    )
    site = 0
    environment = state.site_environment(site)
    metric, hamiltonian = state.local_operators(site, environment=environment)
    layout = PhysicalBlockLayout(state.tensors[site].shape)
    for row, column in ((0, 1), (1, 0), (2, 7)):
        block = state._norm_frontier.hole_block(
            site,
            environment.norm_left,
            environment.norm_right,
            layout.configurations[row],
            layout.configurations[column],
        )
        reference = metric[
            np.ix_(layout.block_indices[row], layout.block_indices[column])
        ]
        np.testing.assert_allclose(block, reference, atol=2.0e-12)

    def forbidden_dense_hole(*_args, **_kwargs):
        raise AssertionError("the block path must not form a dense hole matrix")

    monkeypatch.setattr(state._norm_frontier, "hole_matrix", forbidden_dense_hole)
    monkeypatch.setattr(
        state._hamiltonian_frontier, "hole_matrix", forbidden_dense_hole
    )
    problem = state.local_block_problem(site, environment=environment)

    np.testing.assert_allclose(problem.metric.to_dense(), metric, atol=2.0e-12)
    np.testing.assert_allclose(
        problem.hamiltonian.to_dense(), hamiltonian, atol=2.0e-12
    )
    assert len(problem.metric.blocks) == problem.layout.nblocks
    assert len(problem.hamiltonian.blocks) < problem.layout.nblocks**2
    assert problem.stored_elements < problem.dense_elements


@pytest.mark.parametrize("frontier_backend", ["compressed", "identity_block"])
def test_frontier_block_sparse_update_matches_direct(frontier_backend):
    initial = _frustrated_graph_state(seed=47)
    direct = FrontierTiedLETTA(
        initial.hamiltonian,
        initial.dims,
        initial.parent_sets,
        bond_dim=initial.bond_dim,
        tensors=initial.tensors,
        frontier_backend=frontier_backend,
    )
    blocked = direct.copy()

    direct_update = direct.optimize_site(0, solver="direct")
    block_update = blocked.optimize_site(
        0,
        solver="block_sparse",
        eig_tol=1.0e-11,
        maxiter=400,
    )

    assert direct_update.accepted
    assert block_update.accepted
    assert block_update.solver == "block_sparse"
    assert block_update.physical_blocks == 8
    assert block_update.hamiltonian_blocks < 64
    assert block_update.stored_operator_elements < 2 * blocked.tensors[0].size**2
    np.testing.assert_allclose(block_update.energy, direct_update.energy, atol=3.0e-9)


def test_auto_selects_block_sparse_solver_for_a_large_sparse_local_problem():
    state = _frustrated_graph_state(seed=53)
    update = state.optimize_site(
        0,
        solver="auto",
        matrix_free_threshold=1,
        eig_tol=1.0e-11,
        maxiter=400,
    )

    assert update.accepted
    assert update.solver == "block_sparse"


def test_auto_respects_the_block_storage_cap():
    state = _frustrated_graph_state(seed=54)
    update = state.optimize_site(
        0,
        solver="auto",
        matrix_free_threshold=1,
        block_sparse_max_elements=1,
        eig_tol=1.0e-11,
        maxiter=400,
    )

    assert update.accepted
    assert update.solver == "matrix_free"


def test_block_sparse_directional_sweep_is_variational():
    state = _frustrated_graph_state(seed=59)
    initial_energy = state.energy

    state.run(nsweeps=1, tol=0.0, solver="block_sparse", eig_tol=1.0e-10)

    assert state.energy <= initial_energy + 2.0e-11
    assert state.history[0]["solver_failures"] == 0
    assert all(
        update.solver == "block_sparse" for update in state.history[0]["updates"]
    )
