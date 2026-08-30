import numpy as np
import pytest

from pyqed.letta import (
    DenseTiedLETTA,
    FrontierBondExpansion,
    FrontierBondReduction,
    FrontierGaugeUpdate,
    FrontierNaturalGradientUpdate,
    FrontierSiteUpdate,
    FrontierTiedLETTA,
    FrontierTwoSiteUpdate,
    LocalHamiltonian,
    LocalTerm,
    PhysicalBlockLinearOperator,
)
from pyqed.letta.physical_blocks import PhysicalBlockLayout
import pyqed.letta.physical_blocks as physical_blocks_module


def _local_hamiltonian():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
    exchange = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    return LocalHamiltonian(
        (2, 2, 2, 2),
        (
            LocalTerm((0,), 0.13 * sz),
            LocalTerm((0, 3), 0.7 * exchange),
            LocalTerm((1, 2), -0.2 * exchange),
            LocalTerm((0, 3), 0.05 * np.kron(sz, sz)),
        ),
        constant=0.17,
    )


def _states(seed=5):
    local_hamiltonian = _local_hamiltonian()
    parents = ((1, 3), (2,), (3,), ())
    frontier = FrontierTiedLETTA(
        local_hamiltonian,
        local_hamiltonian.dims,
        parents,
        bond_dim=2,
        seed=seed,
    )
    dense_hamiltonian = local_hamiltonian.to_dense()
    dense = DenseTiedLETTA(
        dense_hamiltonian,
        local_hamiltonian.dims,
        parents,
        bond_dim=2,
        tensors=frontier.tensors,
    )
    dense.tensors = [tensor.copy() for tensor in frontier.tensors]
    dense.energy = dense.expectation()
    return frontier, dense


def _temporary_pair_local_operators(state, site):
    """Reference the historical merged-pair temporary-state contraction."""
    following = site + 1
    merged, union_sites = state._merged_pair_tensor(site)
    right_dimension = state.bond_dims[following + 1]
    right_sites = state.physical_sites[following]
    identity = np.eye(right_dimension, dtype=merged.dtype)
    identity_tensor = np.broadcast_to(
        identity.reshape(
            right_dimension,
            right_dimension,
            *((1,) * len(right_sites)),
        ),
        (
            right_dimension,
            right_dimension,
            *(state.dims[index] for index in right_sites),
        ),
    ).copy()
    temporary_tensors = [tensor.copy() for tensor in state.tensors]
    temporary_tensors[site] = merged
    temporary_tensors[following] = identity_tensor
    temporary_parents = list(state.parent_sets)
    temporary_parents[site] = tuple(index for index in union_sites if index != site)
    temporary_bonds = list(state.bond_dims)
    temporary_bonds[following] = right_dimension
    temporary = FrontierTiedLETTA(
        state.hamiltonian,
        state.dims,
        tuple(temporary_parents),
        bond_dims=tuple(temporary_bonds),
        tensors=temporary_tensors,
        frontier_backend=state.frontier_backend,
        path_optimizer=state.path_optimizer,
    )
    # Construction balances gauges.  Restore the exact reference tensors used
    # by the former two-site implementation before forming its local pencil.
    temporary.tensors = temporary_tensors
    return temporary.local_operators(site)


def test_physical_block_native_matvec_matches_python_reference():
    if physical_blocks_module._physical_blocks_cpp is None:
        pytest.skip("optional physical-block C++ extension is not built")
    rng = np.random.default_rng(123)
    layout = PhysicalBlockLayout((3, 4, 2, 2))
    blocks = {
        (0, 0): rng.normal(size=(12, 12)),
        (3, 1): rng.normal(size=(12, 12)),
        (2, 3): rng.normal(size=(12, 12)),
    }
    for key in blocks:
        blocks[key] = blocks[key] + 1j * rng.normal(size=(12, 12))
    operator = PhysicalBlockLinearOperator(layout, blocks)
    vector = rng.normal(size=layout.size) + 1j * rng.normal(size=layout.size)
    native = physical_blocks_module._physical_blocks_cpp
    native_matvec = operator.matvec(vector)
    native_rmatvec = operator.rmatvec(vector)
    physical_blocks_module._physical_blocks_cpp = None
    try:
        reference_matvec = operator.matvec(vector)
        reference_rmatvec = operator.rmatvec(vector)
    finally:
        physical_blocks_module._physical_blocks_cpp = native
    np.testing.assert_allclose(native_matvec, reference_matvec, atol=1.0e-12)
    np.testing.assert_allclose(native_rmatvec, reference_rmatvec, atol=1.0e-12)


def test_local_hamiltonian_combines_supports_and_matches_dense_matvec():
    hamiltonian = _local_hamiltonian()
    assert hamiltonian.nterms == 3
    assert hamiltonian.supports == ((0,), (0, 3), (1, 2))

    rng = np.random.default_rng(4)
    vector = rng.normal(size=16) + 1.0j * rng.normal(size=16)
    np.testing.assert_allclose(
        hamiltonian @ vector,
        hamiltonian.to_dense() @ vector,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        hamiltonian.to_dense(),
        hamiltonian.to_dense().T.conj(),
        atol=2.0e-14,
    )


def test_frontier_norm_energy_and_local_operators_match_explicit_projectors():
    frontier, dense = _states()
    assert not hasattr(frontier, "_configs")
    np.testing.assert_allclose(
        frontier.state_vector(), dense.state_vector(), atol=2.0e-14
    )
    np.testing.assert_allclose(frontier.norm(), dense.norm(), atol=2.0e-13)
    np.testing.assert_allclose(
        frontier.expectation(), dense.expectation(), atol=2.0e-13
    )

    for site, tensor in enumerate(frontier.tensors):
        projector = dense.local_projector(site)
        reference_metric = projector.T.conj() @ projector
        reference_effective = projector.T.conj() @ dense.hamiltonian @ projector
        metric, effective = frontier.local_operators(site)
        np.testing.assert_allclose(metric, reference_metric, atol=5.0e-13)
        np.testing.assert_allclose(effective, reference_effective, atol=5.0e-13)

        vector = np.linspace(-0.7, 0.9, tensor.size).astype(complex)
        np.testing.assert_allclose(
            frontier.metric_action(site, vector),
            reference_metric @ vector,
            atol=5.0e-13,
        )
        np.testing.assert_allclose(
            frontier.hamiltonian_action(site, vector),
            reference_effective @ vector,
            atol=5.0e-13,
        )
    assert frontier.contraction_plans > 0


def test_frontier_direct_and_matrix_free_updates_are_variational_and_agree():
    initial, _dense = _states(seed=9)
    direct = initial.copy()
    iterative = initial.copy()

    direct_update = direct.optimize_site(1, solver="direct")
    iterative_update = iterative.optimize_site(1, solver="matrix_free", eig_tol=1.0e-12)

    assert direct_update.accepted
    assert iterative_update.accepted
    assert iterative_update.hamiltonian_matvecs > 0
    assert direct_update.energy <= direct_update.energy_before + 1.0e-12
    assert iterative_update.energy <= iterative_update.energy_before + 1.0e-12
    np.testing.assert_allclose(
        iterative_update.energy,
        direct_update.energy,
        atol=2.0e-10,
    )


def test_frontier_whitened_solver_uses_an_exact_local_identity_metric():
    initial, _dense = _states(seed=10)
    site = 1
    metric = initial.local_metric(site)
    basis, whitened_hamiltonian, frame = initial.local_whitened_problem(site)

    identity_metric = basis.T.conj() @ metric @ basis
    np.testing.assert_allclose(
        identity_metric,
        np.eye(frame["metric_rank"]),
        atol=2.0e-11,
    )
    assert whitened_hamiltonian.shape == (
        frame["metric_rank"],
        frame["metric_rank"],
    )
    assert frame["identity_metric_error"] < 2.0e-11

    direct = initial.copy()
    whitened = initial.copy()
    orthonormal = initial.copy()
    direct_update = direct.optimize_site(site, solver="direct")
    whitened_update = whitened.optimize_site(site, solver="whitened")
    orthonormal_update = orthonormal.optimize_site(
        site,
        solver="metric_orthonormal",
    )

    assert direct_update.accepted
    assert whitened_update.accepted
    assert whitened_update.solver == "whitened"
    assert whitened_update.solver_metric_is_identity
    assert whitened_update.solver_metric_identity_error < 2.0e-11
    assert whitened_update.solver_coordinate_residual_norm < 2.0e-10
    assert orthonormal_update.solver == "metric_orthonormal"
    assert orthonormal_update.solver_metric_is_identity
    np.testing.assert_allclose(
        orthonormal_update.energy,
        whitened_update.energy,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        whitened_update.energy,
        direct_update.energy,
        atol=2.0e-10,
    )


def test_frontier_sweep_decreases_energy_without_materializing_basis():
    state, _dense = _states(seed=12)
    initial = state.energy
    state.run(nsweeps=1, tol=0.0, solver="direct")

    assert not hasattr(state, "_configs")
    assert state.energy <= initial + 2.0e-11
    assert state.history[0]["accepted_sites"] > 0


def test_frontier_sweep_offset_preserves_direction_across_resumed_runs():
    state, _dense = _states(seed=12)

    state.run(nsweeps=1, sweep_offset=1, tol=0.0, solver="direct")

    assert state.history[0]["sweep"] == 1
    assert [update.site for update in state.history[0]["updates"]] == [3, 2, 1, 0]


@pytest.mark.parametrize("frontier_backend", ["compressed", "identity_block"])
def test_frontier_sweep_reuses_completed_messages_for_energy(
    monkeypatch,
    frontier_backend,
):
    initial, _dense = _states(seed=13)
    state = FrontierTiedLETTA(
        initial.hamiltonian,
        initial.dims,
        initial.parent_sets,
        bond_dim=initial.bond_dim,
        tensors=initial.tensors,
        frontier_backend=frontier_backend,
    )
    calls = {"norm": 0, "hamiltonian": 0}
    norm_scalar = state._norm_frontier.scalar
    hamiltonian_scalar = state._hamiltonian_frontier.scalar

    def counted_norm(tensors):
        calls["norm"] += 1
        return norm_scalar(tensors)

    def counted_hamiltonian(tensors):
        calls["hamiltonian"] += 1
        return hamiltonian_scalar(tensors)

    monkeypatch.setattr(state._norm_frontier, "scalar", counted_norm)
    monkeypatch.setattr(state._hamiltonian_frontier, "scalar", counted_hamiltonian)
    state.run(nsweeps=2, tol=0.0, solver="direct")

    # The only full contractions are the initial expectation-value check.
    assert calls == {"norm": 1, "hamiltonian": 1}
    np.testing.assert_allclose(state.energy, state.expectation(), atol=5.0e-13)


def test_frontier_natural_gradient_step_is_variational_and_projector_free():
    state, _dense = _states(seed=4)
    initial = state.energy

    update = state.natural_gradient_step(trust_radius=0.1)

    assert isinstance(update, FrontierNaturalGradientUpdate)
    assert update.accepted
    assert update.energy < initial
    assert state.energy == update.energy
    assert update.step_size > 0.0
    assert update.step_size * update.metric_direction_norm <= 0.1 + 1.0e-14
    assert update.directional_derivative < 0.0
    assert len(update.metric_ranks) == len(state.dims)
    assert not hasattr(state, "_configs")


def test_frontier_natural_gradient_armijo_failure_restores_tensors(monkeypatch):
    state, _dense = _states(seed=4)
    initial_energy = state.expectation()
    initial_tensors = [tensor.copy() for tensor in state.tensors]
    original_scalar = state._hamiltonian_frontier.scalar
    calls = 0

    def rejected_scalar(tensors):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_scalar(tensors)
        candidate_norm = state._norm_frontier.scalar(tensors)
        return (initial_energy + 1.0) * candidate_norm

    monkeypatch.setattr(state._hamiltonian_frontier, "scalar", rejected_scalar)
    update = state.natural_gradient_step(
        trust_radius=0.1,
        max_backtracks=2,
    )

    assert not update.accepted
    assert update.message == "backtracking found no energy decrease"
    assert update.step_size == 0.0
    assert update.backtracks == 2
    assert state.energy == initial_energy
    assert calls == 4
    for tensor, reference in zip(state.tensors, initial_tensors):
        np.testing.assert_array_equal(tensor, reference)


def test_virtual_canonicalization_preserves_tied_state_and_tensor_shapes():
    initial, _dense = _states(seed=6)
    vector = initial.state_vector()
    energy = initial.energy
    shapes = [tensor.shape for tensor in initial.tensors]

    for direction in ("left", "right"):
        state = initial.copy()
        state.canonicalize_virtual(direction)
        np.testing.assert_allclose(state.state_vector(), vector, atol=2.0e-14)
        np.testing.assert_allclose(state.expectation(), energy, atol=2.0e-14)
        assert [tensor.shape for tensor in state.tensors] == shapes


def test_frontier_accepts_variable_internal_bond_dimensions_and_copies_them():
    hamiltonian = _local_hamiltonian()
    parents = ((1, 3), (2,), (3,), ())
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dims=(2, 3, 2),
        seed=8,
    )

    assert state.bond_dims == (1, 2, 3, 2, 1)
    assert state.bond_dim == 3
    assert [tensor.shape[:2] for tensor in state.tensors] == [
        (1, 2),
        (2, 3),
        (3, 2),
        (2, 1),
    ]
    copied = state.copy()
    assert copied.bond_dims == state.bond_dims
    np.testing.assert_allclose(copied.state_vector(), state.state_vector(), atol=0.0)
    np.testing.assert_allclose(copied.energy, state.energy, atol=3.0e-14)


@pytest.mark.parametrize("direction,relaxed_site", [("right", 2), ("left", 1)])
def test_residual_bond_expansion_preserves_state_and_opens_local_subspace(
    direction,
    relaxed_site,
):
    state, _dense = _states(seed=5)
    vector = state.state_vector()
    energy = state.energy

    update = state.expand_bond(
        2,
        4,
        direction=direction,
        strategy="residual",
        seed=3,
    )

    assert isinstance(update, FrontierBondExpansion)
    assert update.seeded_directions == 2
    assert update.source_norm > 0.0
    assert update.norm_error < 2.0e-14
    assert state.bond_dims == (1, 2, 4, 2, 1)
    assert state.tensors[1].shape[1] == 4
    assert state.tensors[2].shape[0] == 4
    np.testing.assert_array_equal(state.state_vector(), vector)
    np.testing.assert_allclose(state.energy, energy, atol=2.0e-14)

    local_update = state.optimize_site(relaxed_site, solver="direct")
    assert local_update.accepted
    assert local_update.energy < energy


def test_multi_bond_zero_expansion_preserves_state_with_nonuniform_dimensions():
    state, _dense = _states(seed=17)
    vector = state.state_vector()
    energy = state.energy

    updates = state.expand_bond_dims(
        (3, 4, 2),
        strategy="zero",
        direction="right",
    )

    assert len(updates) == 2
    assert all(update.seeded_directions == 0 for update in updates)
    assert state.bond_dims == (1, 3, 4, 2, 1)
    np.testing.assert_array_equal(state.state_vector(), vector)
    np.testing.assert_allclose(state.energy, energy, atol=3.0e-14)

    reductions = state.reduce_null_bonds()

    assert [update.cut for update in reductions] == [1, 2]
    assert state.bond_dims == (1, 2, 2, 2, 1)
    np.testing.assert_allclose(state.state_vector(), vector, atol=3.0e-14)
    np.testing.assert_allclose(state.energy, energy, atol=3.0e-14)


def test_null_bond_reduction_removes_zero_expansion_without_changing_state():
    state, _dense = _states(seed=23)
    vector = state.state_vector()
    energy = state.energy
    state.expand_bond(2, 4, strategy="zero")

    reductions = state.reduce_null_bonds()

    assert len(reductions) == 1
    assert isinstance(reductions[0], FrontierBondReduction)
    assert reductions[0].cut == 2
    assert reductions[0].old_dimension == 4
    assert reductions[0].new_dimension == 2
    assert reductions[0].relative_discarded_weight < 1.0e-14
    assert reductions[0].norm_error < 2.0e-14
    assert state.bond_dims == (1, 2, 2, 2, 1)
    np.testing.assert_allclose(state.state_vector(), vector, atol=2.0e-14)
    np.testing.assert_allclose(state.energy, energy, atol=3.0e-14)


def test_two_site_update_with_tied_overlap_reaches_exact_two_spin_energy():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = LocalHamiltonian(
        (2, 2),
        (LocalTerm((0, 1), -np.kron(sx, sx)),),
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((1,), ()),
        bond_dim=1,
        seed=41,
        frontier_backend="identity_block",
    )

    update = state.optimize_two_sites(0, solver="whitened")

    assert isinstance(update, FrontierTwoSiteUpdate)
    assert update.accepted
    assert update.local_update.solver_converged
    assert update.overlap_sites == (1,)
    assert update.conditional_ranks == (1, 1)
    np.testing.assert_allclose(update.energy, -1.0, atol=2.0e-12)
    np.testing.assert_allclose(state.expectation(), -1.0, atol=2.0e-12)


def test_two_site_merge_split_preserves_overlapping_graph_pair_and_variable_bonds():
    state, _dense = _states(seed=43)
    state.expand_bond(2, 3, strategy="zero")
    merged, union_sites = state._merged_pair_tensor(1)
    left, right, overlap, _ranks, error = state._split_merged_pair_tensor(
        1,
        merged,
        union_sites,
    )
    original_left = state.tensors[1]
    original_right = state.tensors[2]
    state.tensors[1] = left
    state.tensors[2] = right
    reconstructed, reconstructed_sites = state._merged_pair_tensor(1)
    state.tensors[1] = original_left
    state.tensors[2] = original_right

    assert reconstructed_sites == union_sites
    assert overlap == (2,)
    np.testing.assert_allclose(reconstructed, merged, atol=2.0e-12)
    np.testing.assert_allclose(error, 0.0, atol=2.0e-14)


@pytest.mark.parametrize("frontier_backend", ["compressed", "identity_block"])
def test_cached_pair_local_operators_match_temporary_state_reference(
    frontier_backend,
):
    hamiltonian = _local_hamiltonian()
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((1, 3), (2,), (3,), ()),
        bond_dim=2,
        seed=45,
        frontier_backend=frontier_backend,
    )

    for site in range(len(state.dims) - 1):
        metric, effective = state.pair_local_operators(site)
        reference_metric, reference_effective = _temporary_pair_local_operators(
            state,
            site,
        )
        merged, union_sites = state._merged_pair_tensor(site)

        assert state.pair_environment(site).union_sites == union_sites
        assert metric.shape == (merged.size, merged.size)
        assert effective.shape == (merged.size, merged.size)
        np.testing.assert_allclose(metric, reference_metric, atol=3.0e-13)
        np.testing.assert_allclose(effective, reference_effective, atol=8.0e-13)


@pytest.mark.parametrize(
    "sweep_offset,sites",
    [(0, range(3)), (1, range(2, -1, -1))],
)
def test_cached_two_site_directional_sweep_matches_rebuilt_environments(
    sweep_offset,
    sites,
):
    cached, _dense = _states(seed=47)
    initial_energy = cached.expectation()
    rebuilt = cached.copy()
    options = {
        "solver": "whitened",
        "outer_cycles": 1,
        "factor_solver": "dense",
        "split_random_starts": 0,
    }

    rebuilt_updates = [rebuilt.optimize_two_sites(site, **options) for site in sites]
    cached.run_two_site(
        nsweeps=1,
        sweep_offset=sweep_offset,
        verify_pair_energies=False,
        **options,
    )
    cached_updates = cached.history[0]["updates"]

    assert cached.history[0]["accepted"]
    assert cached.history[0]["energy"] <= initial_energy + 3.0e-13
    np.testing.assert_allclose(
        [update.energy for update in cached_updates],
        [update.energy for update in rebuilt_updates],
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        cached.expectation(),
        rebuilt.expectation(),
        atol=2.0e-11,
    )


def test_cached_two_site_sweep_rolls_back_an_increased_endpoint(monkeypatch):
    state, _dense = _states(seed=48)
    tensors = [tensor.copy() for tensor in state.tensors]
    energy = state.expectation()
    completed = state._completed_frontier_scalar

    def inflated_endpoint(frontier, message, cut):
        if frontier is state._hamiltonian_frontier:
            return 1.0e6
        return completed(frontier, message, cut)

    monkeypatch.setattr(state, "_completed_frontier_scalar", inflated_endpoint)
    state.run_two_site(
        nsweeps=1,
        outer_cycles=1,
        factor_solver="dense",
    )

    assert not state.history[0]["accepted"]
    np.testing.assert_allclose(state.energy, energy, atol=0.0)
    for tensor, reference in zip(state.tensors, tensors):
        np.testing.assert_array_equal(tensor, reference)


def test_cached_two_site_sweep_retains_both_directional_history_rows():
    state, _dense = _states(seed=50)
    reference = state.copy()
    options = {
        "solver": "whitened",
        "outer_cycles": 1,
        "factor_solver": "dense",
        "split_random_starts": 0,
        "tol": 0.0,
    }

    state.run_two_site(nsweeps=2, **options)
    reference_rows = []
    for sweep_offset in range(2):
        reference.run_two_site(
            nsweeps=1,
            sweep_offset=sweep_offset,
            **options,
        )
        reference_rows.append(reference.history[0])

    assert [row["sweep"] for row in state.history] == [0, 1]
    assert [row["direction"] for row in state.history] == [
        "left_to_right",
        "right_to_left",
    ]
    np.testing.assert_allclose(
        [row["energy"] for row in state.history],
        [row["energy"] for row in reference_rows],
        atol=2.0e-11,
    )
    np.testing.assert_allclose(state.energy, reference.energy, atol=2.0e-11)
    for tensor, expected in zip(state.tensors, reference.tensors):
        np.testing.assert_allclose(tensor, expected, atol=2.0e-11)


def test_pair_conditional_blocks_reconstruct_dense_operators():
    state, _dense = _states(seed=49)
    metric, effective = state.pair_local_operators(1)
    problem = state.pair_local_block_problem(1)

    np.testing.assert_allclose(problem.metric.to_dense(), metric, atol=3.0e-13)
    np.testing.assert_allclose(
        problem.hamiltonian.to_dense(),
        effective,
        atol=3.0e-13,
    )
    assert problem.stored_elements < metric.size + effective.size


def test_pair_conditional_blocks_contract_one_hermitian_orientation(monkeypatch):
    state, _dense = _states(seed=49)
    plan = state._pair_plan(1)
    original = plan.hamiltonian_engine.hole_blocks
    requested = []

    def record_requests(site, left, right, configuration_pairs):
        requested.extend(configuration_pairs)
        return original(site, left, right, configuration_pairs)

    monkeypatch.setattr(
        plan.hamiltonian_engine,
        "hole_blocks",
        record_requests,
    )
    problem = state.pair_local_block_problem(1)

    assert requested
    assert all(row <= column for row, column, _bra, _ket in requested)
    assert len(requested) < len(problem.hamiltonian.blocks)
    np.testing.assert_allclose(
        problem.hamiltonian.to_dense(),
        state.pair_local_operators(1)[1],
        atol=3.0e-13,
    )


def test_auto_pair_backend_reuses_dense_pencil_for_factor_solve(monkeypatch):
    state, _dense = _states(seed=51)
    original = state._variational_split_merged_pair
    observed = {}

    def record_factor_solver(*args, **kwargs):
        observed["factor_solver"] = kwargs["factor_solver"]
        observed["has_metric"] = args[3] is not None
        observed["has_effective"] = args[4] is not None
        return original(*args, **kwargs)

    monkeypatch.setattr(
        state,
        "_variational_split_merged_pair",
        record_factor_solver,
    )
    update = state.optimize_two_sites(
        1,
        pair_operator_backend="auto",
        factor_solver="auto",
        pair_dense_max_elements=4_000_000,
        pair_operator_workers=2,
        outer_cycles=1,
    )

    itemsize = np.dtype(
        np.result_type(state.hamiltonian.dtype, *state.tensors)
    ).itemsize
    assert update.accepted
    assert update.pair_operator_requested_backend == "auto"
    assert update.pair_operator_backend == "dense"
    assert update.factor_solver == "dense"
    assert update.pair_operator_workers == 2
    assert observed == {
        "factor_solver": "dense",
        "has_metric": True,
        "has_effective": True,
    }
    assert update.pair_operator_stored_elements == 2 * update.raw_merged_dim**2
    assert (
        update.pair_operator_stored_bytes
        == update.pair_operator_stored_elements * itemsize
    )
    assert "fit within" in update.pair_operator_selection_reason


def test_auto_pair_backend_uses_matrix_free_factors_above_dense_budget(
    monkeypatch,
):
    state, _dense = _states(seed=52)
    original = state._variational_split_merged_pair
    observed = {}

    def record_factor_solver(*args, **kwargs):
        observed["factor_solver"] = kwargs["factor_solver"]
        observed["has_metric"] = args[3] is not None
        observed["has_effective"] = args[4] is not None
        return original(*args, **kwargs)

    monkeypatch.setattr(
        state,
        "_variational_split_merged_pair",
        record_factor_solver,
    )
    update = state.optimize_two_sites(
        1,
        pair_operator_backend="auto",
        factor_solver="auto",
        pair_dense_max_elements=1,
        outer_cycles=1,
    )

    assert update.accepted
    assert update.pair_operator_backend == "block"
    assert update.factor_solver == "matrix_free"
    assert observed == {
        "factor_solver": "matrix_free",
        "has_metric": False,
        "has_effective": False,
    }
    assert "exceed" in update.pair_operator_selection_reason


def test_block_pair_update_uses_no_dense_pair_operators_or_design(
    monkeypatch,
):
    state, _dense = _states(seed=51)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense pair storage is forbidden in block mode")

    monkeypatch.setattr(state, "pair_local_operators", forbidden)
    monkeypatch.setattr(state, "_pair_factor_design", forbidden)
    monkeypatch.setattr(PhysicalBlockLinearOperator, "to_dense", forbidden)

    update = state.optimize_two_sites(
        1,
        pair_operator_backend="block",
        factor_solver="matrix_free",
        outer_cycles=2,
    )

    assert update.accepted
    assert update.merged_solve.verified
    assert update.merged_solve.lowest_root_certified
    assert not update.merged_solve.dense_fallback
    assert update.pair_operator_backend == "block"
    assert np.isfinite(update.metric_projection_error)
    assert update.pair_operator_stored_elements < 2 * update.raw_merged_dim**2


def test_cached_pair_plan_is_reused_and_invalidated_after_bond_expansion():
    state, _dense = _states(seed=46)
    site = 1
    first = state._pair_plan(site)
    metric_before, _effective_before = state.pair_local_operators(site)

    assert state._pair_plan(site) is first
    assert state._pair_plan_cache[site] is first

    state.expand_bond(3, 3, strategy="zero")

    assert state._pair_plan_cache == {}
    second = state._pair_plan(site)
    metric_after, _effective_after = state.pair_local_operators(site)
    merged_after, _union_sites = state._merged_pair_tensor(site)
    assert second is not first
    assert second.fingerprint != first.fingerprint
    assert metric_after.shape == (merged_after.size, merged_after.size)
    assert metric_after.shape[0] > metric_before.shape[0]


def test_two_site_design_maps_merge_complex_tied_overlap_and_nonuniform_bonds():
    hamiltonian = _local_hamiltonian()
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((1, 3), (2,), (3,), ()),
        bond_dims=(1, 2, 3, 2, 1),
        seed=47,
    )
    rng = np.random.default_rng(48)
    state.tensors = [
        tensor.astype(complex) + 0.2j * rng.normal(size=tensor.shape)
        for tensor in state.tensors
    ]
    site = 1
    merged, union_sites = state._merged_pair_tensor(site)
    left_tensor = state.tensors[site]
    right_tensor = state.tensors[site + 1]

    for variable, factor in (
        ("left", left_tensor),
        ("right", right_tensor),
    ):
        design = state._pair_factor_design(
            site,
            union_sites,
            left_tensor,
            right_tensor,
            variable=variable,
        )
        reconstructed = (design @ factor.reshape(-1)).reshape(merged.shape)
        np.testing.assert_allclose(reconstructed, merged, atol=3.0e-14)


def test_two_site_direct_factor_actions_match_design_and_are_adjoint():
    hamiltonian = _local_hamiltonian()
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((1, 3), (2,), (3,), ()),
        bond_dims=(1, 2, 3, 2, 1),
        seed=49,
    )
    rng = np.random.default_rng(50)
    state.tensors = [
        tensor.astype(complex) + 0.2j * rng.normal(size=tensor.shape)
        for tensor in state.tensors
    ]
    site = 1
    merged, union_sites = state._merged_pair_tensor(site)
    left_tensor = state.tensors[site]
    right_tensor = state.tensors[site + 1]
    cotangent = rng.normal(size=merged.size) + 1j * rng.normal(size=merged.size)

    for variable, factor in (
        ("left", left_tensor),
        ("right", right_tensor),
    ):
        variation = rng.normal(size=factor.size) + 1j * rng.normal(size=factor.size)
        design = state._pair_factor_design(
            site,
            union_sites,
            left_tensor,
            right_tensor,
            variable=variable,
        )
        action = state._pair_factor_action(
            site,
            union_sites,
            left_tensor,
            right_tensor,
            variation,
            variable=variable,
        )
        adjoint = state._pair_factor_adjoint(
            site,
            union_sites,
            left_tensor,
            right_tensor,
            cotangent,
            variable=variable,
        )

        np.testing.assert_allclose(action, design @ variation, atol=3.0e-14)
        np.testing.assert_allclose(
            adjoint,
            design.T.conj() @ cotangent,
            atol=3.0e-14,
        )
        np.testing.assert_allclose(
            np.vdot(action, cotangent),
            np.vdot(variation, adjoint),
            atol=3.0e-13,
        )


def test_two_site_variational_split_is_covariant_to_environment_scale():
    reference, _dense = _states(seed=43)
    scaled = reference.copy()
    scaled.tensors[0] *= 1.0e-20

    updates = [
        state.optimize_two_sites(
            1,
            solver="whitened",
            split_random_starts=0,
        )
        for state in (reference, scaled)
    ]

    assert all(update.accepted for update in updates)
    assert all(np.isfinite(update.energy) for update in updates)
    np.testing.assert_allclose(
        updates[1].energy,
        updates[0].energy,
        atol=3.0e-11,
    )


def test_two_site_update_rejects_harmful_truncation_and_restores_tensors():
    excited = np.array([-0.6, 0.0, 0.0, 0.8], dtype=complex)
    ket_01 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    ket_10 = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    operator = (
        10.0 * np.outer(excited, excited.conj())
        + np.outer(ket_01, ket_01.conj())
        + 2.0 * np.outer(ket_10, ket_10.conj())
    )
    hamiltonian = LocalHamiltonian(
        (2, 2),
        (LocalTerm((0, 1), operator),),
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((), ()),
        bond_dim=1,
        tensors=(
            np.array([[[1.0, 0.0]]], dtype=complex),
            np.array([[[0.0, 1.0]]], dtype=complex),
        ),
    )
    tensors = [tensor.copy() for tensor in state.tensors]

    update = state.optimize_two_sites(
        0,
        solver="whitened",
        split_strategy="svd",
    )

    assert update.local_update.accepted
    assert not update.accepted
    assert update.merged_solve.verified
    assert update.merged_solve.dense_fallback
    assert update.merged_solve.method == "dense_certified"
    assert update.merged_solve.attempts == (
        "warm_davidson",
        "dense_certification",
    )
    assert update.merged_solve.metric_dual_relative_residual < 1.0e-12
    np.testing.assert_allclose(update.energy_before, 1.0, atol=2.0e-12)
    np.testing.assert_allclose(update.merged_energy, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(update.attempted_energy, 3.6, atol=2.0e-12)
    np.testing.assert_allclose(
        update.relative_truncation_error,
        0.6,
        atol=2.0e-12,
    )
    for tensor, reference in zip(state.tensors, tensors):
        np.testing.assert_array_equal(tensor, reference)


def test_verified_pair_solver_certifies_beyond_warm_invariant_subspace():
    state, _dense = _states(seed=53)
    metric = np.eye(4)
    effective = np.diag([0.0, 1.0, 2.0, 3.0])
    warm = np.array([0.0, 0.0, 1.0, 1.0])

    energy, _vector, update, diagnostics = state._solve_verified_pair_pencil(
        0,
        metric,
        effective,
        warm,
        metric_tol=1.0e-12,
        eig_tol=1.0e-12,
        maxiter=50,
        max_subspace=4,
        dense_fallback_dim=4,
    )

    np.testing.assert_allclose(energy, 0.0, atol=1.0e-14)
    assert update.solver_converged
    assert diagnostics.verified
    assert diagnostics.lowest_root_certified
    assert diagnostics.method == "dense_certified"
    assert diagnostics.dense_fallback


def test_verified_pair_solver_skips_redundant_full_rank_davidson():
    state, _dense = _states(seed=53)
    metric = np.eye(4)
    effective = np.diag([0.0, 1.0, 2.0, 3.0])
    warm = np.array([0.0, 0.0, 1.0, 1.0])

    reference = state._solve_verified_pair_pencil(
        0,
        metric,
        effective,
        warm,
        metric_tol=1.0e-12,
        eig_tol=1.0e-12,
        maxiter=50,
        max_subspace=4,
        dense_fallback_dim=4,
    )
    skipped = state._solve_verified_pair_pencil(
        0,
        metric,
        effective,
        warm,
        metric_tol=1.0e-12,
        eig_tol=1.0e-12,
        maxiter=50,
        max_subspace=4,
        dense_fallback_dim=4,
        skip_redundant_full_rank_davidson=True,
    )

    np.testing.assert_allclose(skipped[0], reference[0], atol=1.0e-14)
    np.testing.assert_allclose(
        abs(np.vdot(skipped[1], reference[1])),
        1.0,
        atol=1.0e-14,
    )
    assert skipped[2].solver_converged
    assert skipped[3].lowest_root_certified
    assert skipped[3].attempts == (
        "warm_davidson_skipped_full_rank",
        "dense_certification",
    )

    below_threshold = state._solve_verified_pair_pencil(
        0,
        metric,
        effective,
        warm,
        metric_tol=1.0e-12,
        eig_tol=1.0e-12,
        maxiter=50,
        max_subspace=4,
        dense_fallback_dim=4,
        skip_redundant_full_rank_davidson=True,
        redundant_full_rank_davidson_min_dimension=5,
    )
    assert below_threshold[3].attempts == (
        "warm_davidson",
        "dense_certification",
    )

    rank_deficient = state._solve_verified_pair_pencil(
        0,
        np.diag([1.0, 0.0]),
        np.diag([0.0, 1.0]),
        np.array([1.0, 0.0]),
        metric_tol=1.0e-12,
        eig_tol=1.0e-12,
        maxiter=50,
        max_subspace=4,
        dense_fallback_dim=4,
        skip_redundant_full_rank_davidson=True,
    )
    assert rank_deficient[3].attempts == (
        "warm_davidson",
        "dense_certification",
    )


def test_verified_pair_solver_distinguishes_regularized_and_numerical_support():
    state, _dense = _states(seed=54)
    metric = np.diag([1.0, 5.0e-14])
    effective = np.diag([0.0, -5.0e-13])
    warm = np.array([1.0, 0.0])
    rows = {}

    for support in ("regularized", "numerical"):
        rows[support] = state._solve_verified_pair_pencil(
            0,
            metric,
            effective,
            warm,
            metric_tol=1.0e-12,
            eig_tol=1.0e-12,
            maxiter=20,
            max_subspace=2,
            dense_fallback_dim=2,
            metric_support=support,
        )

    regularized_energy, _vector, regularized_update, regularized = rows["regularized"]
    numerical_energy, _vector, numerical_update, numerical = rows["numerical"]
    np.testing.assert_allclose(regularized_energy, 0.0, atol=1.0e-14)
    assert regularized_update.metric_rank == 1
    assert regularized_update.metric_rank_is_projected
    assert regularized.metric_support == "regularized"
    np.testing.assert_allclose(numerical_energy, -10.0, atol=1.0e-12)
    assert numerical_update.metric_rank == 2
    assert not numerical_update.metric_rank_is_projected
    assert numerical.metric_support == "numerical"


def test_verified_pair_solver_retains_verified_warm_below_support_root(
    monkeypatch,
):
    state, _dense = _states(seed=55)
    metric = np.diag([1.0, 5.0e-14])
    effective = np.diag([0.0, -5.0e-14])
    warm = np.array([0.0, 1.0])

    def failed_davidson(*_args, **_kwargs):
        raise ValueError("forced Davidson failure")

    monkeypatch.setattr(
        "pyqed.letta.frontier_tying.lowest_generalized_davidson",
        failed_davidson,
    )
    energy, vector, update, diagnostics = state._solve_verified_pair_pencil(
        0,
        metric,
        effective,
        warm,
        metric_tol=1.0e-12,
        eig_tol=1.0e-12,
        maxiter=20,
        max_subspace=2,
        dense_fallback_dim=2,
        metric_support="regularized",
    )

    np.testing.assert_allclose(energy, -1.0, atol=1.0e-14)
    np.testing.assert_allclose(
        state._pair_rayleigh(vector, metric, effective),
        -1.0,
        atol=1.0e-14,
    )
    assert diagnostics.method == "warm"
    assert diagnostics.verified
    assert diagnostics.lowest_root_certified
    assert diagnostics.metric_requested_rank == 1
    assert diagnostics.metric_numerical_rank == 2
    assert "above the retained variational state" in diagnostics.fallback_reason
    assert update.solver_converged
    assert "retained residual-verified warm pair" in update.message


def test_two_site_variational_tangent_escapes_coordinate_saddle():
    excited = np.array([-0.6, 0.0, 0.0, 0.8], dtype=complex)
    ket_01 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    ket_10 = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    operator = (
        10.0 * np.outer(excited, excited.conj())
        + np.outer(ket_01, ket_01.conj())
        + 2.0 * np.outer(ket_10, ket_10.conj())
    )
    hamiltonian = LocalHamiltonian(
        (2, 2),
        (LocalTerm((0, 1), operator),),
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((), ()),
        bond_dim=1,
        tensors=(
            np.array([[[1.0, 0.0]]], dtype=complex),
            np.array([[[0.0, 1.0]]], dtype=complex),
        ),
    )
    single_cycle = state.copy()

    single_update = single_cycle.optimize_two_sites(
        0,
        solver="whitened",
        split_random_starts=0,
        outer_cycles=1,
    )

    update = state.optimize_two_sites(
        0,
        solver="whitened",
        split_random_starts=0,
    )

    assert update.accepted
    assert update.split_strategy == "variational"
    assert update.selected_start.startswith("tangent(")
    assert update.factor_random_starts == 0
    assert update.outer_cycles > 1
    assert update.energy <= single_update.energy
    assert np.all(np.diff(update.factor_energy_history) <= 2.0e-12)
    assert update.energy < 0.8
    np.testing.assert_allclose(update.energy, 0.65259042733, atol=1.0e-5)
    np.testing.assert_allclose(state.expectation(), update.energy, atol=2.0e-12)


def test_frontier_bond_grams_transform_covariantly():
    state, _dense = _states(seed=5)
    cut = 2
    gauge = np.array(
        [[1.2 + 0.1j, -0.3j], [0.2 - 0.1j, 0.8 + 0.2j]],
        dtype=complex,
    )
    inverse = np.linalg.inv(gauge)

    transformed = state.copy()
    left_tensor = np.tensordot(
        transformed.tensors[cut - 1],
        gauge,
        axes=(1, 0),
    )
    transformed.tensors[cut - 1] = np.moveaxis(left_tensor, -1, 1)
    transformed.tensors[cut] = np.tensordot(
        inverse,
        transformed.tensors[cut],
        axes=(1, 0),
    )
    for weighting in ("uniform", "probability"):
        left, right = state.frontier_bond_grams(cut, weighting=weighting)
        transformed_left, transformed_right = transformed.frontier_bond_grams(
            cut,
            weighting=weighting,
        )
        np.testing.assert_allclose(
            transformed_left,
            gauge.conj().T @ left @ gauge,
            atol=3.0e-13,
        )
        np.testing.assert_allclose(
            transformed_right,
            inverse @ right @ inverse.conj().T,
            atol=3.0e-13,
        )
    np.testing.assert_allclose(
        transformed.state_vector(),
        state.state_vector(),
        atol=3.0e-13,
    )


def test_frontier_gauge_preserves_state_and_balances_full_rank_marginals():
    state, _dense = _states(seed=5)
    vector = state.state_vector()
    energy = state.expectation()
    shapes = [tensor.shape for tensor in state.tensors]

    updates = state.canonicalize_frontier_gauge()

    assert len(updates) == len(state.dims) - 1
    assert all(isinstance(update, FrontierGaugeUpdate) for update in updates)
    assert all(update.applied for update in updates)
    assert max(update.imbalance_after for update in updates) < 2.0e-12
    np.testing.assert_allclose(state.state_vector(), vector, atol=3.0e-13)
    np.testing.assert_allclose(state.expectation(), energy, atol=3.0e-13)
    assert [tensor.shape for tensor in state.tensors] == shapes

    left_messages = state._norm_frontier.build_left(state.tensors)
    right_messages = state._norm_frontier.build_right(state.tensors)
    for cut in range(1, len(state.dims)):
        left, right = state.frontier_bond_grams(
            cut,
            left_messages=left_messages,
            right_messages=right_messages,
        )
        np.testing.assert_allclose(left, right, atol=8.0e-13)


def test_frontier_gauge_skips_rank_deficient_uniform_endpoint():
    hamiltonian = _local_hamiltonian()
    parents = ((1, 3), (2,), (3,), ())
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=4,
        seed=6,
    )
    vector = state.state_vector()

    updates = state.canonicalize_frontier_gauge()

    assert updates[-1].left_rank == 4
    assert updates[-1].right_rank == 2
    assert not updates[-1].applied
    assert updates[-1].message == "rank-deficient frontier marginal"
    assert any(update.applied for update in updates[:-1])
    np.testing.assert_allclose(state.state_vector(), vector, atol=4.0e-13)


def test_virtual_canonicalization_preserves_endpoint_capped_d4_state():
    hamiltonian = _local_hamiltonian()
    parents = ((1, 3), (2,), (3,), ())
    initial = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=4,
        seed=6,
    )
    vector = initial.state_vector()
    energy = initial.energy
    shapes = [tensor.shape for tensor in initial.tensors]

    assert initial.tensors[-1].shape[0] == 4
    assert np.prod(initial.tensors[-1].shape[1:]) == 2
    for direction in ("left", "right"):
        state = initial.copy()
        state.canonicalize_virtual(direction)
        np.testing.assert_allclose(state.state_vector(), vector, atol=2.0e-14)
        np.testing.assert_allclose(state.expectation(), energy, atol=2.0e-14)
        assert [tensor.shape for tensor in state.tensors] == shapes


def test_frontier_sweep_can_interleave_natural_gradient_relaxation():
    state, _dense = _states(seed=3)
    state.run(
        nsweeps=2,
        tol=0.0,
        solver="direct",
        natural_gradient_every=2,
    )

    assert state.history[0]["natural_gradient"] is None
    update = state.history[1]["natural_gradient"]
    assert isinstance(update, FrontierNaturalGradientUpdate)
    assert update.energy <= update.energy_before
    assert state.history[1]["energy"] == update.energy


def test_adaptive_natural_gradient_backs_off_rejected_steps(monkeypatch):
    state, _dense = _states(seed=3)
    calls = []

    def rejected_step(**options):
        calls.append(
            (
                len(state.history) + 1,
                float(options["trust_radius"]),
            )
        )
        energy = float(state.energy)
        return FrontierNaturalGradientUpdate(
            energy_before=energy,
            energy=energy,
            accepted=False,
            message="rejected for controller test",
            step_size=0.0,
            backtracks=2,
            gradient_norm=1.0,
            preconditioned_norm=1.0,
            metric_direction_norm=1.0,
            directional_derivative=-1.0,
            max_relative_direction=0.0,
            metric_ranks=(1,) * len(state.dims),
        )

    monkeypatch.setattr(state, "natural_gradient_step", rejected_step)
    state.run(
        nsweeps=7,
        tol=-1.0,
        solver="direct",
        natural_gradient_every=2,
        natural_gradient_trust_radius=0.1,
        natural_gradient_adaptive=True,
        natural_gradient_max_interval=8,
    )

    assert calls == [(2, 0.1), (6, 0.05)]
    assert state.history[1]["natural_gradient_interval"] == 4
    assert state.history[1]["natural_gradient_next_sweep"] == 6
    assert state.history[1]["natural_gradient_next_trust_radius"] == 0.05
    assert state.history[5]["natural_gradient_interval"] == 8
    assert state.history[5]["natural_gradient_next_sweep"] == 14
    assert state.history[5]["natural_gradient_next_trust_radius"] == 0.025


def test_fixed_natural_gradient_schedule_remains_available(monkeypatch):
    state, _dense = _states(seed=3)
    calls = []

    def rejected_step(**options):
        calls.append(
            (
                len(state.history) + 1,
                float(options["trust_radius"]),
            )
        )
        energy = float(state.energy)
        return FrontierNaturalGradientUpdate(
            energy_before=energy,
            energy=energy,
            accepted=False,
            message="rejected for controller test",
            step_size=0.0,
            backtracks=2,
            gradient_norm=1.0,
            preconditioned_norm=1.0,
            metric_direction_norm=1.0,
            directional_derivative=-1.0,
            max_relative_direction=0.0,
            metric_ranks=(1,) * len(state.dims),
        )

    monkeypatch.setattr(state, "natural_gradient_step", rejected_step)
    state.run(
        nsweeps=6,
        tol=-1.0,
        solver="direct",
        natural_gradient_every=2,
        natural_gradient_trust_radius=0.1,
        natural_gradient_adaptive=False,
    )

    assert calls == [(2, 0.1), (4, 0.1), (6, 0.1)]
    assert state.history[1]["natural_gradient_interval"] == 2
    assert state.history[3]["natural_gradient_interval"] == 2
    assert state.history[5]["natural_gradient_interval"] == 2


def test_adaptive_natural_gradient_grows_trust_for_accurate_step(monkeypatch):
    state, _dense = _states(seed=3)

    def accurate_step(**_options):
        energy_before = float(state.energy)
        energy = energy_before - 0.02
        state.energy = energy
        return FrontierNaturalGradientUpdate(
            energy_before=energy_before,
            energy=energy,
            accepted=True,
            message="accepted for controller test",
            step_size=0.1,
            backtracks=0,
            gradient_norm=1.0,
            preconditioned_norm=1.0,
            metric_direction_norm=1.0,
            directional_derivative=-0.2,
            max_relative_direction=0.1,
            metric_ranks=(1,) * len(state.dims),
        )

    monkeypatch.setattr(state, "natural_gradient_step", accurate_step)
    state.run(
        nsweeps=2,
        tol=0.0,
        solver="direct",
        natural_gradient_every=2,
        natural_gradient_trust_radius=0.1,
        natural_gradient_adaptive=True,
    )

    record = state.history[1]
    np.testing.assert_allclose(record["natural_gradient_quality_ratio"], 1.0)
    np.testing.assert_allclose(record["natural_gradient_relative_gain"], 0.02)
    np.testing.assert_allclose(
        record["natural_gradient_next_trust_radius"],
        0.125,
    )
    assert record["natural_gradient_interval"] == 2
    assert record["natural_gradient_next_sweep"] == 4


def test_frontier_sweep_can_use_frontier_canonical_gauge():
    state, _dense = _states(seed=3)
    state.run(
        nsweeps=1,
        tol=0.0,
        solver="direct",
        gauge="frontier",
    )

    gauge_updates = state.history[0]["gauge_update"]
    assert len(gauge_updates) == len(state.dims) - 1
    assert all(update.applied for update in gauge_updates)


def test_frontier_sweep_defaults_to_metric_orthonormal_conditional_gauge():
    state, _dense = _states(seed=3)
    state.run(nsweeps=1, tol=0.0)

    record = state.history[0]
    assert all(
        update.solver == "metric_orthonormal"
        for update in record["updates"]
    )
    assert len(record["gauge_update"]) == len(state.dims) - 1
    assert all(update.applied for update in record["gauge_update"])


def test_cached_bidirectional_frontier_sweep_energies_match_explicit_updates():
    frontier, dense = _states(seed=23)
    frontier.run(nsweeps=2, tol=0.0, solver="direct")
    dense.run(nsweeps=2, tol=0.0)

    np.testing.assert_allclose(frontier.energy, dense.energy, atol=3.0e-11)
    np.testing.assert_allclose(
        [record["energy"] for record in frontier.history],
        [record["energy"] for record in dense.history],
        atol=3.0e-11,
    )


def test_checkpointed_frontier_sweeps_match_full_cache_in_both_directions():
    reference, _dense = _states(seed=41)
    full = reference.copy()
    checkpointed = reference.copy()

    full.run(
        nsweeps=2,
        tol=0.0,
        solver="direct",
        environment_cache="full",
    )
    checkpointed.run(
        nsweeps=2,
        tol=0.0,
        solver="direct",
        environment_cache="checkpointed",
        environment_checkpoint_interval=2,
    )

    np.testing.assert_allclose(
        [record["energy"] for record in checkpointed.history],
        [record["energy"] for record in full.history],
        atol=4.0e-11,
    )
    assert all(
        record["environment_cache"] == "checkpointed" for record in checkpointed.history
    )
    assert checkpointed.fixed_environment_cache_elements(
        interval=2
    ) <= checkpointed.fixed_environment_cache_elements(mode="full")
    assert (
        len(
            checkpointed._build_environment_checkpoints(
                checkpointed._norm_frontier,
                direction="right",
                interval=2,
            )
        )
        == 3
    )


def test_frontier_sweep_does_not_claim_convergence_after_solver_failures(
    monkeypatch,
):
    state, _dense = _states(seed=29)

    def failed_update(site, **kwargs):
        energy = float(kwargs["energy_before"])
        return FrontierSiteUpdate(
            site=site,
            raw_dim=state.tensors[site].size,
            metric_rank=0,
            metric_rank_is_projected=True,
            solver="matrix_free",
            solver_converged=False,
            message="forced failure",
            hamiltonian_matvecs=1,
            metric_matvecs=1,
            iterations=1,
            residual_norm=1.0,
            energy_before=energy,
            energy=energy,
            accepted=False,
        )

    monkeypatch.setattr(state, "optimize_site", failed_update)
    state.run(nsweeps=1, tol=np.inf, solver="matrix_free")

    assert not state.converged
    assert state.history[0]["solver_failures"] == len(state.dims)


def test_one_site_convergence_requires_complete_directional_cycle(monkeypatch):
    one_direction, _dense = _states(seed=32)
    complete_cycle = one_direction.copy()
    reverse_then_forward = one_direction.copy()

    def stationary_update(state):
        def update(site, **kwargs):
            energy = float(kwargs["energy_before"])
            return FrontierSiteUpdate(
                site=site,
                raw_dim=state.tensors[site].size,
                metric_rank=state.tensors[site].size,
                metric_rank_is_projected=False,
                solver="direct",
                solver_converged=True,
                message="stationary test update",
                energy_before=energy,
                energy=energy,
                accepted=True,
                residual_norm=0.0,
                hamiltonian_matvecs=0,
                metric_matvecs=0,
                iterations=0,
            )

        return update

    monkeypatch.setattr(one_direction, "optimize_site", stationary_update(one_direction))
    monkeypatch.setattr(complete_cycle, "optimize_site", stationary_update(complete_cycle))
    monkeypatch.setattr(
        reverse_then_forward,
        "optimize_site",
        stationary_update(reverse_then_forward),
    )
    one_direction.run(nsweeps=1, tol=1.0e-12, gauge=None)
    complete_cycle.run(nsweeps=2, tol=1.0e-12, gauge=None)
    reverse_then_forward.run(
        nsweeps=2,
        sweep_offset=1,
        tol=1.0e-12,
        gauge=None,
    )

    assert not one_direction.converged
    assert complete_cycle.converged
    assert not reverse_then_forward.converged
    assert not any(row["cycle_complete"] for row in reverse_then_forward.history)
    assert [row["cycle_complete"] for row in complete_cycle.history] == [
        False,
        True,
    ]
    assert complete_cycle.history[-1]["cycle_stationary"]


def test_identity_block_backend_matches_compressed_backend_and_uses_less_memory():
    compressed, _dense = _states(seed=33)
    block = FrontierTiedLETTA(
        compressed.hamiltonian,
        compressed.dims,
        compressed.parent_sets,
        bond_dim=compressed.bond_dim,
        tensors=compressed.tensors,
        frontier_backend="identity_block",
    )
    block.tensors = [tensor.copy() for tensor in compressed.tensors]

    np.testing.assert_allclose(
        block.expectation(), compressed.expectation(), atol=3e-13
    )
    for site in range(len(block.dims)):
        block_metric, block_hamiltonian = block.local_operators(site)
        dense_metric, dense_hamiltonian = compressed.local_operators(site)
        np.testing.assert_allclose(block_metric, dense_metric, atol=6e-13)
        np.testing.assert_allclose(block_hamiltonian, dense_hamiltonian, atol=6e-13)

    assert (
        block.hamiltonian_peak_frontier_elements
        < compressed.hamiltonian_peak_frontier_elements
    )


def test_identity_block_exact_local_tt_matches_dense_local_absorption():
    dense_local, _dense = _states(seed=35)
    dense_local = FrontierTiedLETTA(
        dense_local.hamiltonian,
        dense_local.dims,
        dense_local.parent_sets,
        bond_dim=dense_local.bond_dim,
        tensors=dense_local.tensors,
        frontier_backend="identity_block",
        local_backend="dense",
    )
    sequential = FrontierTiedLETTA(
        dense_local.hamiltonian,
        dense_local.dims,
        dense_local.parent_sets,
        bond_dim=dense_local.bond_dim,
        tensors=dense_local.tensors,
        frontier_backend="identity_block",
        local_backend="tensor_train",
    )
    dense_local.tensors = [tensor.copy() for tensor in sequential.tensors]
    dense_local.energy = dense_local.expectation()

    np.testing.assert_allclose(
        sequential.expectation(), dense_local.expectation(), atol=2.0e-12
    )
    sequential.run(nsweeps=1, tol=0.0, solver="direct")
    dense_local.run(nsweeps=1, tol=0.0, solver="direct")
    np.testing.assert_allclose(sequential.energy, dense_local.energy, atol=4.0e-11)


def test_identity_block_truncated_local_tt_uses_exact_energy_gate():
    reference, _dense = _states(seed=37)
    exact = FrontierTiedLETTA(
        reference.hamiltonian,
        reference.dims,
        reference.parent_sets,
        bond_dim=reference.bond_dim,
        tensors=reference.tensors,
        frontier_backend="identity_block",
    )
    approximate = FrontierTiedLETTA(
        reference.hamiltonian,
        reference.dims,
        reference.parent_sets,
        bond_dim=reference.bond_dim,
        tensors=reference.tensors,
        frontier_backend="identity_block",
        local_backend="tensor_train",
        local_rank=1,
    )

    assert not approximate.hamiltonian_contraction_is_exact
    np.testing.assert_allclose(
        approximate.expectation(), exact.expectation(), atol=2e-13
    )
    energy_before = approximate.energy
    approximate.run(nsweeps=1, tol=0.0, solver="direct")
    assert approximate.energy <= energy_before + 2.0e-12
    assert (
        approximate.history[0]["tt_diagnostics"]["hamiltonian"][
            "max_relative_discarded_norm"
        ]
        > 0.0
    )
