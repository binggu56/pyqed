from collections import Counter

import numpy as np
import pytest

from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    FrontierAbelianLayout,
    LETTA,
    LocalHamiltonian,
    LocalTerm,
    SymmetricLETTA,
    SymmetryLayout,
    abelian_frontier_tied_letta_from_mps,
    conditional_frontier_letta_from_mps,
    exact_block_factor_layout,
)
from pyqed.letta.physical_blocks import (
    PhysicalBlockLinearOperator,
    hamiltonian_physical_connectivity,
)
from pyqed.letta.tt_frontier import TermwiseTTMPOFrontier


def _heisenberg_chain(nsites=4):
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    exchange = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    return LocalHamiltonian(
        (2,) * nsites,
        tuple(LocalTerm((site, site + 1), exchange) for site in range(nsites - 1)),
    )


def _state(seed=5):
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2,), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    return AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        seed=seed,
    )


def _compressed_state(seed=5):
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2,), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    return AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        frontier_backend="compressed",
        seed=seed,
    )


def _identity_block_state(seed=5):
    base = _state(seed=seed)
    return AbelianFrontierTiedLETTA(
        base.hamiltonian,
        base.dims,
        base.parent_sets,
        abelian_layout=base.abelian_layout,
        tensors=base.tensors,
        frontier_backend="identity_block",
    )


def test_letta_u1_constructor_selects_symmetric_identity_frontier():
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2,), (3,), ())
    state = LETTA(
        hamiltonian,
        parents=parents,
        symmetry="u1",
        charges=((1, -1),) * 4,
        target=0,
        bond_dim=2,
        seed=19,
    )

    assert type(state) is SymmetricLETTA
    assert state.symmetry == "u1"
    assert isinstance(state.layout, SymmetryLayout)
    assert state.layout is state.abelian_layout
    assert state.layout.target == (0,)
    assert state.charge_assignment == "physical"
    assert state.charges == (((1,), (-1,)),) * 4
    assert state.frontier_backend == "identity_block"
    assert AbelianFrontierTiedLETTA is SymmetricLETTA
    assert FrontierAbelianLayout is SymmetryLayout


def test_letta_u1_constructor_charges_physical_copies_are_neutral():
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2,), (3,), ())
    state = LETTA(
        hamiltonian,
        parents=parents,
        symmetry="u1",
        charges=((1, -1),) * 4,
        target=0,
        bond_dim=2,
        seed=19,
    )

    assert state.charge_rules == (((0, 1),), ((1, 1),), ((2, 1),), ((3, 1),))
    assert state.local_support_sizes()[0][0] < state.local_support_sizes()[0][1]
    assert np.array_equal(state.local_masks[0][:, :, :, 0, :], state.local_masks[0][:, :, :, 1, :])

    vector = state.state_vector()
    for configuration, amplitude in zip(np.ndindex(*state.dims), vector):
        two_sz = sum(1 if local == 0 else -1 for local in configuration)
        if two_sz != 0:
            np.testing.assert_allclose(amplitude, 0.0, atol=1.0e-14)


def test_letta_u1_constructor_occurrence_mode_is_diagnostic():
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2,), (3,), ())
    state = LETTA(
        hamiltonian,
        parents=parents,
        symmetry="u1",
        charges=((1, -1),) * 4,
        target=0,
        bond_dim=2,
        charge_assignment="occurrence",
        seed=19,
    )

    assert state.charge_assignment == "occurrence"
    assert state.charge_rules[0] == ((0, 1), (1, 1), (2, 1))
    assert state.charge_rules[2] == ((2, -1), (3, 1))
    assert state.charge_rules[3] == ()
    assert not np.array_equal(
        state.local_masks[0][:, :, :, 0, :],
        state.local_masks[0][:, :, :, 1, :],
    )


def test_letta_u1_constructor_requires_charge_representation():
    hamiltonian = _heisenberg_chain()
    with pytest.raises(TypeError, match="requires charges or layout"):
        LETTA(
            hamiltonian,
            parents=((1,), (2,), (3,), ()),
            symmetry="u1",
            target=0,
            bond_dim=2,
        )


def _charge_resolved_mps(seed=23):
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=(1, 2, 3, 2, 1),
    )
    physical_sites = tuple((site,) for site in range(4))
    masks = layout.local_masks(physical_sites)
    rng = np.random.default_rng(seed)
    cores = []
    for mask in masks:
        native = np.where(mask, rng.normal(size=mask.shape), 0.0)
        cores.append(native.transpose(0, 2, 1))
    return layout, tuple(cores)


def _mps_vector(cores):
    dims = tuple(core.shape[1] for core in cores)
    values = []
    for configuration in np.ndindex(*dims):
        environment = np.ones(1)
        for core, physical in zip(cores, configuration):
            environment = environment @ core[:, physical, :]
        values.append(environment[0])
    vector = np.asarray(values)
    return vector / np.linalg.norm(vector)


def test_frontier_abelian_parent_legs_are_neutral_spectators():
    state = _state()
    mask = state.local_masks[0]

    # Site zero owns only its leading physical leg.  Changing either tied
    # parent therefore cannot alter charge compatibility.
    for left in range(mask.shape[0]):
        for right in range(mask.shape[1]):
            for physical in range(mask.shape[2]):
                assert np.all(
                    mask[left, right, physical]
                    == mask[left, right, physical, 0, 0]
                )

    assert state.nparameters == sum(
        allowed for allowed, _total in state.local_support_sizes()
    )
    assert state.nparameters < state.dense_nparameters


def test_frontier_abelian_state_has_only_target_charge_amplitudes():
    state = _state()
    vector = state.state_vector()
    for configuration, amplitude in zip(np.ndindex(*state.dims), vector):
        two_sz = sum(1 if local == 0 else -1 for local in configuration)
        if two_sz != 0:
            np.testing.assert_allclose(amplitude, 0.0, atol=1.0e-14)


def test_frontier_abelian_two_site_update_preserves_support():
    state = _compressed_state(seed=41)
    energy_before = state.expectation()

    update = state.optimize_two_sites(
        0,
        solver="verified",
        pair_operator_backend="dense",
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
    )

    assert update.accepted
    assert update.energy <= energy_before + 1.0e-10
    assert update.merged_solve is not None
    assert update.merged_solve.verified
    assert update.split_strategy == "sector_variational"
    assert update.pair_operator_backend == "dense_sector"
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)
    for configuration, amplitude in zip(np.ndindex(*state.dims), state.state_vector()):
        two_sz = sum(1 if local == 0 else -1 for local in configuration)
        if two_sz != 0:
            np.testing.assert_allclose(amplitude, 0.0, atol=1.0e-14)


def test_frontier_abelian_two_site_variational_split_is_sector_guarded():
    state = _compressed_state(seed=41)
    energy_before = state.expectation()

    update = state.optimize_two_sites(
        0,
        solver="verified",
        pair_operator_backend="dense",
        split_strategy="variational",
        split_variational_sweeps=2,
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
    )

    assert update.accepted
    assert update.energy <= energy_before + 1.0e-10
    assert update.split_strategy == "sector_variational"
    assert update.factor_sweeps > 0
    assert update.pair_operator_backend == "dense_sector"
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_two_site_update_can_expand_middle_bond():
    state = _compressed_state(seed=41)
    energy_before = state.expectation()

    update = state.optimize_two_sites(
        0,
        solver="verified",
        pair_operator_backend="dense",
        temporary_bond_dimension=4,
        split_variational_sweeps=1,
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
    )

    assert update.accepted
    assert update.old_bond_dimension == 2
    assert update.temporary_bond_dimension == 4
    assert state.bond_dims[1] == 4
    assert state.abelian_layout.bond_dims == state.bond_dims
    assert state.energy <= energy_before + 1.0e-10
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_two_site_sweep_supports_shape_changing_updates():
    state = _compressed_state(seed=42)
    energy_before = state.expectation()

    state.run_two_site(
        nsweeps=1,
        solver="verified",
        pair_operator_backend="dense",
        temporary_bond_dimension=4,
        split_variational_sweeps=1,
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
        verify_pair_energies=False,
    )

    assert state.energy <= energy_before + 1.0e-10
    assert state.history
    assert max(state.bond_dims) == 4
    assert any(
        update.temporary_bond_dimension > update.old_bond_dimension
        for update in state.history[0]["updates"]
    )
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_two_site_sweep_runs_with_compressed_frontier():
    state = _compressed_state(seed=42)
    energy_before = state.expectation()

    state.run_two_site(
        nsweeps=1,
        solver="verified",
        pair_operator_backend="dense",
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
        verify_pair_energies=False,
    )

    assert state.energy <= energy_before + 1.0e-10
    assert state.history
    assert any(update.accepted for update in state.history[0]["updates"])
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_two_site_identity_block_pair_environments():
    state = _identity_block_state(seed=43)

    for site in range(len(state.dims) - 1):
        environment = state.pair_environment(site)
        assert environment.sites == (site, site + 1)
        assert environment.hamiltonian_right.cut == site + 1


def test_frontier_abelian_identity_block_pair_operator_matches_compressed():
    hamiltonian = _heisenberg_chain()
    layout, cores = _charge_resolved_mps(seed=43)
    parents = ((),) * len(cores)
    compressed = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        cores,
        abelian_layout=layout,
        frontier_backend="compressed",
    )
    block = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        cores,
        abelian_layout=layout,
        frontier_backend="identity_block",
    )

    for site in range(len(cores) - 1):
        reference_metric, reference_hamiltonian = compressed.pair_local_operators(site)
        metric, effective = block.pair_local_operators(site)
        np.testing.assert_allclose(metric, reference_metric, atol=8e-13)
        np.testing.assert_allclose(effective, reference_hamiltonian, atol=8e-13)


@pytest.mark.parametrize("site", range(3))
def test_frontier_abelian_dense_support_slice_matches_restricted_blocks(site):
    state = _identity_block_state(seed=43)
    plan = state._pair_plan(site)
    environment = state.pair_environment(site)
    metric, hamiltonian = state.pair_local_operators(
        site,
        environment=environment,
    )
    support = state._pair_support_indices(
        site,
        plan.union_sites,
        plan.merged_shape,
    )
    block_problem = state._sector_restrict_pair_block_problem(
        site,
        plan.union_sites,
        state.pair_local_block_problem(site, environment=environment),
    )

    expected_metric = metric[np.ix_(support, support)]
    expected_hamiltonian = hamiltonian[np.ix_(support, support)]
    actual_metric = block_problem.metric.to_dense()[np.ix_(support, support)]
    actual_hamiltonian = block_problem.hamiltonian.to_dense()[
        np.ix_(support, support)
    ]
    np.testing.assert_allclose(actual_metric, expected_metric, atol=8e-13)
    np.testing.assert_allclose(actual_hamiltonian, expected_hamiltonian, atol=8e-13)


@pytest.mark.parametrize("site", range(3))
def test_frontier_abelian_support_metric_avoids_full_pair_matrix(site):
    state = _identity_block_state(seed=43)
    plan = state._pair_plan(site)
    environment = state.pair_environment(site)
    support = state._pair_support_indices(
        site,
        plan.union_sites,
        plan.merged_shape,
    )
    expected = plan.norm_engine.hole_matrix(
        site,
        environment.norm_left,
        environment.norm_right,
    )[np.ix_(support, support)]

    actual, raw_elements = state._pair_support_metric(
        site,
        plan,
        environment,
        support,
    )

    np.testing.assert_allclose(actual, expected, atol=8e-13)
    assert raw_elements < int(np.prod(plan.merged_shape)) ** 2


@pytest.mark.parametrize(
    ("budget_offset", "expected_backend"),
    ((-1, "matrix_free_sector"), (0, "dense_sector"), (1, "dense_sector")),
)
def test_frontier_abelian_auto_pair_backend_uses_adaptive_byte_threshold(
    budget_offset,
    expected_backend,
):
    state = _identity_block_state(seed=44)
    site = 0
    plan = state._pair_plan(site)
    raw_dim = int(np.prod(plan.merged_shape))
    support_size = state._pair_support_indices(
        site,
        plan.union_sites,
        plan.merged_shape,
    ).size
    operator_itemsize = np.dtype(
        np.result_type(
            state.hamiltonian.dtype,
            state.tensors[site].dtype,
            state.tensors[site + 1].dtype,
        )
    ).itemsize
    expected_stored = 2 * support_size**2
    expected_peak = 2 * raw_dim**2 + expected_stored
    dense_peak_bytes = expected_peak * operator_itemsize

    update = state.optimize_two_sites(
        site,
        solver="block_sparse",
        pair_operator_backend="auto",
        pair_dense_max_bytes=dense_peak_bytes + budget_offset,
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
    )

    assert update.pair_operator_backend == expected_backend
    if expected_backend != "dense_sector":
        return
    assert update.pair_operator_stored_elements == expected_stored
    assert update.pair_operator_peak_elements == expected_peak
    assert update.pair_operator_stored_bytes == expected_stored * operator_itemsize
    assert update.pair_operator_peak_bytes == expected_peak * operator_itemsize


def test_frontier_abelian_auto_pair_backend_respects_matrix_free_budget_and_profile():
    state = _identity_block_state(seed=45)
    site = 0
    plan = state._pair_plan(site)
    support_size = state._pair_support_indices(
        site,
        plan.union_sites,
        plan.merged_shape,
    ).size
    virtual_size = plan.merged_shape[0] * plan.merged_shape[1]
    physical_blocks = int(np.prod(plan.merged_shape[2:]))
    connected_blocks = len(
        hamiltonian_physical_connectivity(
            state.hamiltonian,
            plan.union_sites,
        )
    )
    itemsize = np.dtype(
        np.result_type(state.hamiltonian.dtype, *state.tensors)
    ).itemsize
    matrix_free_peak = (
        3 * support_size**2
        + max(physical_blocks, connected_blocks) * virtual_size**2
        + physical_blocks * virtual_size**2
    )

    blocked = state.copy().optimize_two_sites(
        site,
        pair_operator_backend="auto",
        pair_dense_max_bytes=matrix_free_peak * itemsize - 1,
        outer_cycles=1,
    )
    assert blocked.pair_operator_backend == "block_sector"
    assert (
        blocked.pair_operator_selection_reason
        == "matrix_free_metric_workspace_exceeds_budget"
    )

    budget = matrix_free_peak * itemsize
    probed = state.optimize_two_sites(
        site,
        pair_operator_backend="auto",
        pair_dense_max_bytes=budget,
        outer_cycles=1,
        matrix_free_max_action_vectors=10_000,
    )
    assert probed.pair_operator_backend == "matrix_free_sector"
    assert probed.merged_solve.hamiltonian_vector_products > 1

    profiled = state.optimize_two_sites(
        site,
        pair_operator_backend="auto",
        pair_dense_max_bytes=budget,
        outer_cycles=1,
        matrix_free_max_action_vectors=1,
    )
    assert profiled.pair_operator_backend == "block_sector"
    assert (
        profiled.pair_operator_selection_reason
        == "matrix_free_action_profile_exceeds_limit"
    )


def test_frontier_abelian_two_site_update_runs_with_identity_block_frontier():
    state = _identity_block_state(seed=44)
    energy_before = state.expectation()

    update = state.optimize_two_sites(
        0,
        solver="verified",
        pair_operator_backend="dense",
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
    )

    assert update.accepted
    assert update.energy <= energy_before + 1.0e-10
    assert update.merged_solve is not None
    assert update.merged_solve.verified
    assert update.pair_operator_backend == "dense_sector"
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_block_pair_update_uses_no_dense_pair_operator(
    monkeypatch,
):
    state = _identity_block_state(seed=46)
    energy_before = state.expectation()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense pair storage is forbidden in block-sector mode")

    monkeypatch.setattr(state, "pair_local_operators", forbidden)
    monkeypatch.setattr(PhysicalBlockLinearOperator, "to_dense", forbidden)
    update = state.optimize_two_sites(
        0,
        solver="verified",
        pair_operator_backend="block",
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
        merged_dense_fallback_dim=0,
    )

    assert update.accepted
    assert update.energy <= energy_before + 1.0e-10
    assert update.merged_solve is not None
    assert update.merged_solve.verified
    assert not update.merged_solve.dense_fallback
    assert update.pair_operator_backend == "block_sector"
    assert update.pair_operator_stored_elements < 2 * update.raw_merged_dim**2
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_two_site_sweep_runs_with_identity_block_frontier():
    state = _identity_block_state(seed=45)
    energy_before = state.expectation()

    state.run_two_site(
        nsweeps=1,
        solver="verified",
        pair_operator_backend="dense",
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
        verify_pair_energies=False,
    )

    assert state.energy <= energy_before + 1.0e-10
    assert state.history
    assert any(update.accepted for update in state.history[0]["updates"])
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_block_pair_sweep_uses_no_dense_pair_operator(
    monkeypatch,
):
    state = _identity_block_state(seed=47)
    energy_before = state.expectation()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense pair storage is forbidden in block-sector mode")

    monkeypatch.setattr(state, "pair_local_operators", forbidden)
    monkeypatch.setattr(PhysicalBlockLinearOperator, "to_dense", forbidden)
    state.run_two_site(
        nsweeps=1,
        solver="verified",
        pair_operator_backend="block",
        outer_cycles=1,
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=32,
        merged_dense_fallback_dim=0,
        verify_pair_energies=False,
    )

    assert state.energy <= energy_before + 1.0e-10
    assert state.history
    assert any(update.accepted for update in state.history[0]["updates"])
    assert all(
        update.pair_operator_backend == "block_sector"
        for update in state.history[0]["updates"]
    )
    assert all(
        not update.merged_solve.dense_fallback
        for update in state.history[0]["updates"]
        if update.merged_solve is not None
    )
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_identity_blocks_store_only_charge_compatible_pairs():
    reference = _state(seed=35)
    state = AbelianFrontierTiedLETTA(
        reference.hamiltonian,
        reference.dims,
        reference.parent_sets,
        abelian_layout=reference.abelian_layout,
        tensors=reference.tensors,
        frontier_backend="identity_block",
    )
    dense_blocks = type(state._hamiltonian_frontier)(
        state.dims,
        state.physical_sites,
        [tensor.shape for tensor in state.tensors],
        state.hamiltonian_mpo.tensors,
    )

    np.testing.assert_allclose(state.expectation(), reference.expectation(), atol=8e-13)
    assert state._hamiltonian_frontier.charge_resolved
    assert (
        state._hamiltonian_frontier.total_message_elements
        < dense_blocks.total_message_elements
    )


def test_frontier_abelian_identity_blocks_batch_exact_charge_pair_maps():
    hamiltonian = _heisenberg_chain()
    layout, cores = _charge_resolved_mps(seed=41)
    parents = ((),) * len(cores)
    compressed = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        cores,
        abelian_layout=layout,
        frontier_backend="compressed",
    )
    block = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        cores,
        abelian_layout=layout,
        frontier_backend="identity_block",
    )

    np.testing.assert_allclose(block.expectation(), compressed.expectation(), atol=8e-13)


def test_frontier_abelian_local_tt_truncation_preserves_charge_blocks():
    base = _state(seed=36)
    state = AbelianFrontierTiedLETTA(
        base.hamiltonian,
        base.dims,
        base.parent_sets,
        abelian_layout=base.abelian_layout,
        tensors=base.tensors,
        frontier_backend="identity_block",
        local_backend="tensor_train",
        local_rank=1,
    )

    state._hamiltonian_frontier.scalar(state.tensors)
    diagnostics = state._hamiltonian_frontier.diagnostics
    assert diagnostics["last"]["charge_resolved"]
    assert diagnostics["peak_rank"] == 1
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_termwise_tt_checkpointed_sweep_preserves_blocks():
    base = _state(seed=37)
    state = AbelianFrontierTiedLETTA(
        base.hamiltonian,
        base.dims,
        base.parent_sets,
        abelian_layout=base.abelian_layout,
        tensors=base.tensors,
        frontier_backend="tensor_train",
    )

    state.run(
        nsweeps=1,
        tol=0.0,
        solver="matrix_free",
        eig_tol=1.0e-11,
        maxiter=300,
        environment_cache="checkpointed",
        environment_checkpoint_interval=2,
    )

    assert state.history[0]["environment_cache"] == "checkpointed"
    assert state.hamiltonian_contraction_is_exact
    assert state._hamiltonian_frontier.local_qns == state.abelian_layout.local_qns
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_termwise_tt_rejects_transfer_truncation():
    base = _state(seed=39)
    with pytest.raises(ValueError, match="requires exact local transfers"):
        AbelianFrontierTiedLETTA(
            base.hamiltonian,
            base.dims,
            base.parent_sets,
            abelian_layout=base.abelian_layout,
            tensors=base.tensors,
            frontier_backend="tensor_train",
            max_rank=8,
            transfer_max_rank=8,
        )


def test_frontier_abelian_projects_support_before_first_tt_contraction(
    monkeypatch,
):
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    original = TermwiseTTMPOFrontier.scalar
    calls = []

    def checked_scalar(frontier, tensors):
        masks = layout.local_masks(frontier.physical_sites)
        for tensor, mask in zip(tensors, masks):
            np.testing.assert_array_equal(tensor[~mask], 0.0)
        calls.append(True)
        return original(frontier, tensors)

    monkeypatch.setattr(TermwiseTTMPOFrontier, "scalar", checked_scalar)
    AbelianFrontierTiedLETTA(
        _heisenberg_chain(),
        (2,) * 4,
        ((1, 2), (2,), (3,), ()),
        abelian_layout=layout,
        frontier_backend="tensor_train",
        seed=43,
    )

    assert calls


@pytest.mark.parametrize(
    "solver",
    ["direct", "whitened", "matrix_free", "block_sparse"],
)
def test_frontier_abelian_local_solvers_preserve_support_and_agree(solver):
    reference = _state(seed=7)
    direct = reference.copy()
    expected = direct.optimize_site(1, solver="direct")
    state = reference.copy()
    update = state.optimize_site(
        1,
        solver=solver,
        eig_tol=1.0e-11,
        maxiter=300,
    )

    assert update.accepted
    assert update.solver_converged
    np.testing.assert_allclose(update.energy, expected.energy, atol=2.0e-9)
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_two_site_split_preserves_charge_and_lowers_energy():
    hamiltonian = _heisenberg_chain()
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((),) * 4,
        abelian_layout=layout,
        frontier_backend="identity_block",
        seed=29,
    )
    energy_before = state.expectation()

    update = state.optimize_two_sites(
        1,
        solver="direct",
        pair_operator_backend="dense",
        split_strategy="svd",
        eig_tol=1.0e-11,
    )

    assert update.accepted
    assert update.energy <= energy_before
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)
    for configuration, amplitude in zip(
        np.ndindex(*state.dims),
        state.state_vector(),
    ):
        two_sz = sum(1 if local == 0 else -1 for local in configuration)
        if two_sz != 0:
            np.testing.assert_allclose(amplitude, 0.0, atol=1.0e-14)


def test_frontier_abelian_tied_two_site_supports_action_blocks():
    state = _state(seed=30)
    energy_before = state.expectation()

    update = state.optimize_two_sites(
        1,
        solver="matrix_free",
        pair_operator_backend="action",
        split_strategy="svd",
        eig_tol=1.0e-11,
        maxiter=300,
        max_subspace=24,
    )

    assert update.accepted
    assert update.merged_solve.verified
    assert update.pair_operator_backend == "action"
    assert update.energy <= energy_before
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_variational_pair_split_stays_in_charge_support():
    state = _state(seed=30)
    energy_before = state.expectation()

    update = state.optimize_two_sites(
        1,
        solver="matrix_free",
        pair_operator_backend="action",
        split_strategy="variational",
        split_metric_sweeps=1,
        split_variational_sweeps=2,
        outer_cycles=1,
        metric_tol=1.0e-10,
        eig_tol=1.0e-9,
        maxiter=300,
        max_subspace=24,
    )

    assert update.accepted
    assert update.energy <= energy_before + 1.0e-11
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_optimized_schedule_records_certified_phases(monkeypatch):
    state = _state(seed=32)
    certificates = state.pair_residual_certificates
    certificate_calls = 0

    def counted_certificates(**kwargs):
        nonlocal certificate_calls
        certificate_calls += 1
        return certificates(**kwargs)

    monkeypatch.setattr(state, "pair_residual_certificates", counted_certificates)
    state.run_optimized(
        warmup_sweeps=1,
        two_site_cycles=1,
        polish_sweeps=1,
        tol=1.0e-7,
        two_site_options={
            "max_pairs": 1,
            "split_metric_sweeps": 1,
            "split_variational_sweeps": 1,
            "maxiter": 100,
            "max_subspace": 24,
        },
    )

    assert tuple(name for name, _history in state.optimization_history) == (
        "warmup",
        "two_site",
        "polish",
    )
    assert np.isfinite(state.optimization_summary["maximum_pair_residual"])
    assert certificate_calls == 1
    pair_history = dict(state.optimization_history)["two_site"]
    assert not any(row["residual_certification_due"] for row in pair_history)
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_pair_problem_removes_forbidden_charge_coordinates():
    state = _state(seed=30)
    site = 1
    plan = state._pair_plan(site)
    merged, _union = state._merged_pair_tensor(site)
    mask = state._pair_action_mask(site, plan).reshape(-1)
    problem = state.pair_local_action_block_problem(site)
    rng = np.random.default_rng(91)
    vector = rng.normal(size=merged.size)

    assert np.count_nonzero(mask) < mask.size
    np.testing.assert_array_equal(merged.reshape(-1)[~mask], 0.0)
    metric = problem.metric.to_dense()
    np.testing.assert_array_equal(metric[~mask], 0.0)
    np.testing.assert_array_equal(metric[:, ~mask], 0.0)
    applied = problem.hamiltonian.matvec(vector)
    np.testing.assert_array_equal(applied[~mask], 0.0)
    np.testing.assert_allclose(
        problem.hamiltonian.matvec(np.where(mask, 0.0, vector)),
        0.0,
        atol=0.0,
    )
    assert problem.metric_rank() <= np.count_nonzero(mask)


def test_frontier_abelian_packed_pair_action_matches_generic_action():
    state = _state(seed=33)
    site = 1
    generic = state.pair_local_action_block_problem(site)
    packed = state.pair_local_packed_action_block_problem(site)
    vector = np.random.default_rng(17).normal(size=generic.layout.size)

    np.testing.assert_allclose(
        packed.metric.to_dense(),
        generic.metric.to_dense(),
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        packed.hamiltonian.matvec(vector),
        generic.hamiltonian.matvec(vector),
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        packed.hamiltonian.matvecs(np.stack((vector, -0.3 * vector))),
        np.stack(
            (
                generic.hamiltonian.matvec(vector),
                generic.hamiltonian.matvec(-0.3 * vector),
            )
        ),
        atol=3.0e-13,
    )
    assert packed.hamiltonian.backend == "packed-u1-pair-blocks-cpu"
    assert packed.hamiltonian.stored_elements > 0
    assert packed.hamiltonian.stored_elements < generic.layout.size**2


def test_frontier_abelian_packed_pair_update_is_variational():
    state = _state(seed=35)
    energy = state.expectation()
    update = state.optimize_two_sites(
        1,
        solver="matrix_free",
        pair_operator_backend="packed",
        split_strategy="svd",
        eig_tol=1.0e-10,
        maxiter=300,
        max_subspace=24,
        block_size=2,
        recycle=True,
        recycle_min_size=1,
        preconditioner="auto",
    )

    assert update.accepted
    assert update.energy <= energy
    assert update.pair_operator_backend == "packed"
    assert state._davidson_recycle


def test_frontier_abelian_packed_pair_mixed_precision_has_exact_verifier():
    base = _state(seed=36)
    state = AbelianFrontierTiedLETTA(
        base.hamiltonian,
        base.parent_sets,
        abelian_layout=base.abelian_layout,
        tensors=base.tensors,
        frontier_backend="identity_block",
        compute_dtype=np.float32,
    )
    problem = state.pair_local_packed_action_block_problem(1)
    vector = state._merged_pair_tensor(1)[0].reshape(-1)

    assert problem.hamiltonian.has_verification_action
    np.testing.assert_allclose(
        problem.hamiltonian.matvec(vector),
        problem.hamiltonian.verification_matvec(vector),
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_frontier_abelian_pair_residual_scores_are_finite():
    state = _state(seed=34)
    scores = state.pair_residual_scores()

    assert len(scores) == len(state.dims) - 1
    assert np.all(np.isfinite(scores))
    assert np.all(np.asarray(scores) >= 0.0)


def test_frontier_abelian_two_site_sweep_reports_complete_cycle():
    state = _state(seed=32)
    energy_before = state.expectation()

    state.run_two_site(
        nsweeps=2,
        tol=0.0,
        solver="matrix_free",
        pair_operator_backend="action",
        factor_solver="matrix_free",
        split_strategy="svd",
        outer_cycles=1,
        eig_tol=1.0e-10,
        maxiter=300,
        max_subspace=24,
    )

    assert [row["cycle_complete"] for row in state.history] == [False, True]
    assert state.history[-1]["cycle_endpoints_accepted"]
    assert state.history[-1]["cycle_delta"] == pytest.approx(
        abs(state.energy - energy_before),
        abs=2.0e-11,
    )
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_two_site_rejects_unknown_split_strategy():
    state = _state(seed=31)

    with pytest.raises(ValueError, match="must be 'svd', 'variational', or 'hybrid'"):
        state.optimize_two_sites(1, split_strategy="unsupported")


def test_frontier_abelian_natural_gradient_preserves_support_and_lowers_energy():
    state = _state(seed=7)
    energy = state.expectation()

    update = state.natural_gradient_step(trust_radius=0.1)

    assert update.accepted
    assert update.energy < energy
    assert state.energy == update.energy
    assert update.metric_ranks == (5, 5, 3, 2)
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)
    for configuration, amplitude in zip(np.ndindex(*state.dims), state.state_vector()):
        two_sz = sum(1 if local == 0 else -1 for local in configuration)
        if two_sz != 0:
            np.testing.assert_allclose(amplitude, 0.0, atol=1.0e-14)


def test_frontier_abelian_natural_gradient_data_match_dense_support():
    state = _state(seed=7)
    energy = state.expectation()

    for site in range(len(state.dims)):
        environment = state.site_environment(site)
        metric, effective = state.local_operators(site, environment=environment)
        support = state._support_indices(site)
        reduced_metric, vector, residual, returned_support = (
            state._natural_gradient_local_data(site, environment, energy)
        )

        np.testing.assert_array_equal(returned_support, support)
        np.testing.assert_array_equal(
            vector,
            state.tensors[site].reshape(-1)[support],
        )
        np.testing.assert_allclose(
            reduced_metric,
            metric[np.ix_(support, support)],
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            residual,
            (
                effective @ state.tensors[site].reshape(-1)
                - energy * (metric @ state.tensors[site].reshape(-1))
            )[support],
            atol=2.0e-14,
        )


def test_frontier_abelian_natural_gradient_accepts_cached_energy_and_norm(
    monkeypatch,
):
    state = _state(seed=7)
    state.normalize()
    energy = state.expectation()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cached scalar data must be reused")

    monkeypatch.setattr(state, "expectation", forbidden)
    monkeypatch.setattr(state, "norm", forbidden)
    update = state.natural_gradient_step(
        trust_radius=0.1,
        energy_before=energy,
        state_norm=1.0,
    )

    assert update.accepted
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_sweep_can_interleave_natural_gradient():
    state = _state(seed=13)

    state.run(
        nsweeps=1,
        tol=0.0,
        solver="direct",
        natural_gradient_every=1,
        natural_gradient_trust_radius=0.1,
    )

    update = state.history[0]["natural_gradient"]
    assert update is not None
    assert update.energy <= update.energy_before
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_sector_gauge_preserves_state_and_support():
    state = _state(seed=11)
    vector = state.state_vector()
    energy = state.expectation()

    updates = state.canonicalize_frontier_gauge(weighting="uniform")

    assert updates
    assert all(update.applied for update in updates)
    assert all(update.message == "sector-balanced" for update in updates)
    np.testing.assert_allclose(state.state_vector(), vector, atol=3.0e-13)
    np.testing.assert_allclose(state.expectation(), energy, atol=3.0e-13)
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_rejects_sector_mixing_virtual_qr():
    state = _state()
    with pytest.raises(NotImplementedError, match="mix Abelian sectors"):
        state.canonicalize_virtual("left")


@pytest.mark.parametrize("direction", ["left", "right"])
def test_frontier_abelian_bond_expansion_updates_masks_without_changing_state(
    direction,
):
    state = _state(seed=19)
    vector = state.state_vector()
    energy = state.expectation()
    parameters = state.nparameters

    record = state.expand_bond(
        2,
        3,
        direction=direction,
        strategy="random",
        scale=1.0e-3,
        seed=3,
    )

    assert state.bond_dims == (1, 2, 3, 2, 1)
    assert state.abelian_layout.bond_dims == state.bond_dims
    assert record.old_dimension == 2
    assert record.new_dimension == 3
    assert state.nparameters > parameters
    np.testing.assert_allclose(state.state_vector(), vector, atol=3.0e-13)
    np.testing.assert_allclose(state.expectation(), energy, atol=3.0e-13)
    for tensor, mask in zip(state.tensors, state.local_masks):
        assert tensor.shape == mask.shape
        np.testing.assert_array_equal(tensor[~mask], 0.0)

    copied = state.copy()
    assert copied.bond_dims == state.bond_dims
    assert copied.abelian_layout == state.abelian_layout
    np.testing.assert_allclose(copied.state_vector(), vector, atol=3.0e-13)


def test_frontier_abelian_layout_rejects_disconnected_explicit_sectors():
    local_qns = (((1,), (-1,)),) * 2
    with pytest.raises(ValueError, match="removes every entry"):
        layout = FrontierAbelianLayout(
            local_qns=local_qns,
            bond_qns=(((0,),), ((4,),), ((0,),)),
            target=(0,),
        )
        layout.local_masks(((0,), (1,)))


def test_charge_resolved_mps_lift_preserves_nonuniform_bonds_and_state():
    layout, cores = _charge_resolved_mps()
    hamiltonian = _heisenberg_chain()
    state = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        ((1, 2), (2,), (3,), ()),
        cores,
        local_qns=layout.local_qns,
        bond_qns=layout.bond_qns,
        target=layout.target,
        tie_noise=0.0,
    )

    assert state.bond_dims == layout.bond_dims
    np.testing.assert_allclose(
        state.state_vector(),
        _mps_vector(cores),
        atol=3.0e-13,
    )
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_charge_resolved_mps_lift_into_conditional_factors_is_exact():
    layout, cores = _charge_resolved_mps()
    state = conditional_frontier_letta_from_mps(
        _heisenberg_chain(),
        ((1, 2), (2, 3), (3,), ()),
        cores,
        abelian_layout=layout,
        frontier_backend="identity_block",
    )

    expected = _mps_vector(cores)
    expected /= np.linalg.norm(expected)
    assert isinstance(state, U1ConditionalFrontierLETTA)
    np.testing.assert_allclose(state.state_vector(), expected, atol=3.0e-13)
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)

    paired = conditional_frontier_letta_from_mps(
        _heisenberg_chain(),
        ((1, 2), (2, 3), (3,), ()),
        cores,
        abelian_layout=layout,
        parent_group_size=2,
        frontier_backend="identity_block",
    )
    np.testing.assert_allclose(paired.state_vector(), expected, atol=3.0e-13)
    assert paired.factors[0][1].ndim == 5
    for tensor, mask in zip(paired.tensors, paired.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_charge_resolved_mps_lift_rejects_forbidden_core_entry():
    layout, cores = _charge_resolved_mps()
    cores = list(cores)
    native_mask = layout.local_masks(tuple((site,) for site in range(4)))[1]
    forbidden = np.argwhere(~native_mask)[0]
    core_coord = (forbidden[0], forbidden[2], forbidden[1])
    cores[1] = cores[1].copy()
    cores[1][core_coord] = 0.2

    with pytest.raises(ValueError, match="outside its Abelian charge support"):
        abelian_frontier_tied_letta_from_mps(
            _heisenberg_chain(),
            ((1,), (2,), (3,), ()),
            cores,
            abelian_layout=layout,
        )


def test_frontier_abelian_constructor_defaults_bonds_from_layout():
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=(1, 2, 3, 2, 1),
    )
    state = AbelianFrontierTiedLETTA(
        _heisenberg_chain(),
        (2,) * 4,
        ((1,), (2,), (3,), ()),
        abelian_layout=layout,
        seed=31,
    )
    assert state.bond_dims == layout.bond_dims

    with pytest.raises(ValueError, match="inconsistent with layout"):
        AbelianFrontierTiedLETTA(
            _heisenberg_chain(),
            (2,) * 4,
            ((1,), (2,), (3,), ()),
            abelian_layout=layout,
            bond_dim=4,
            seed=31,
        )
