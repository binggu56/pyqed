from collections import Counter

import numpy as np
import pytest

from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    FrontierAbelianLayout,
    U1ConditionalFrontierLETTA,
    abelian_frontier_tied_letta_from_mps,
    conditional_frontier_letta_from_mps,
    exact_block_factor_layout,
)
from pyqed.tn import LocalHamiltonian, LocalTerm
from pyqed.letta.tt_frontier import TermwiseTTMPOFrontier


def test_exact_block_factor_layout_has_full_fixed_charge_schmidt_capacity():
    base = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=1,
    )
    layout = exact_block_factor_layout(
        base,
        ((0, 1, 2, 3),),
        ((0,), (1,), (2,), (3,)),
    )

    assert layout.bond_dims == (1, 2, 4, 2, 1)
    assert tuple(Counter(labels) for labels in layout.bond_qns[1:4]) == (
        Counter({(-1,): 1, (1,): 1}),
        Counter({(-2,): 1, (0,): 2, (2,): 1}),
        Counter({(-1,): 1, (1,): 1}),
    )


def test_direct_four_site_action_matches_dense_and_reaches_exact_energy():
    hamiltonian = _heisenberg_chain(4)
    base = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=1,
    )
    layout = exact_block_factor_layout(
        base,
        ((0, 1, 2, 3),),
        ((0,), (1,), (2,), (3,)),
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((),) * 4,
        abelian_layout=layout,
        seed=19,
        frontier_backend="identity_block",
    )
    merged, _union = state._merged_block_tensor(0, 4)
    metric, effective = state.block_local_operators(0, 4)
    np.testing.assert_allclose(
        state.block_metric_action(0, 4, merged),
        metric @ merged.reshape(-1),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        state.block_hamiltonian_action(0, 4, merged),
        effective @ merged.reshape(-1),
        atol=2.0e-12,
    )

    energy_before = state.energy
    update = state.optimize_block(
        0,
        4,
        operator_backend="action",
        eig_tol=1.0e-12,
        maxiter=200,
        max_subspace=32,
        merged_dense_fallback_dim=64,
    )
    exact = float(np.linalg.eigvalsh(hamiltonian.to_dense())[0])

    assert update.accepted
    assert update.operator_backend == "action"
    assert update.relative_truncation_error < 1.0e-12
    assert state.energy < energy_before
    assert state.energy == pytest.approx(exact, abs=2.0e-10)
    for tensor, mask in zip(state.tensors, state.local_masks):
        assert np.count_nonzero(tensor[~mask]) == 0

    state.run_blocks(
        ((0, 1, 2, 3),),
        nsweeps=1,
        operator_backend="action",
        eig_tol=1.0e-12,
        maxiter=200,
        max_subspace=32,
        merged_dense_fallback_dim=64,
    )
    assert len(state.history) == 1
    assert state.history[0]["energy"] == pytest.approx(exact, abs=2.0e-10)


def test_direct_block_checkpointed_sweeps_match_full_cache_both_directions():
    hamiltonian = _heisenberg_chain(4)
    blocks = ((0, 1), (2, 3))
    base = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    layout = exact_block_factor_layout(
        base,
        blocks,
        ((0,), (1,), (2,), (3,)),
    )
    initial = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((),) * 4,
        abelian_layout=layout,
        seed=23,
        frontier_backend="identity_block",
    )
    full = initial.copy()
    checkpointed = initial.copy()
    options = {
        "nsweeps": 2,
        "operator_backend": "action",
        "eig_tol": 1.0e-12,
        "maxiter": 200,
        "max_subspace": 32,
        "merged_dense_fallback_dim": 64,
    }
    full.run_blocks(blocks, environment_cache="full", **options)
    checkpointed.run_blocks(
        blocks,
        environment_cache="checkpointed",
        environment_checkpoint_interval=1,
        **options,
    )

    np.testing.assert_allclose(
        [record["energy"] for record in checkpointed.history],
        [record["energy"] for record in full.history],
        atol=2.0e-11,
    )
    assert all(
        record["environment_cache"] == "checkpointed"
        for record in checkpointed.history
    )
    assert checkpointed.fixed_block_environment_cache_elements(
        blocks,
        interval=1,
    ) < checkpointed.fixed_block_environment_cache_elements(
        blocks,
        mode="full",
    )


def test_termwise_four_site_action_matches_supersite_action_with_ties():
    hamiltonian = _heisenberg_chain(4)
    parents = ((2,), (3,), (), ())
    physical_sites = tuple(
        (site,) + parent_sites
        for site, parent_sites in enumerate(parents)
    )
    base = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=1,
    )
    layout = exact_block_factor_layout(
        base,
        ((0, 1, 2, 3),),
        physical_sites,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        abelian_layout=layout,
        seed=29,
        frontier_backend="identity_block",
    )
    merged, _union = state._merged_block_tensor(0, 4)
    _metric, effective = state.block_local_operators(0, 4)

    np.testing.assert_allclose(
        state.block_hamiltonian_action(0, 4, merged),
        effective @ merged.reshape(-1),
        atol=3.0e-12,
    )


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


def test_u1_identity_frontier_packs_physical_sector_support_exactly():
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2, 3), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        seed=31,
        frontier_backend="identity_block",
    )
    engine = state._hamiltonian_frontier

    assert engine.charge_resolved
    assert engine.physical_packing_ratio < 0.5
    assert any(
        engine.storage_shape(cut, channel) != engine.block_shape(cut, channel)
        for cut in range(engine.nsites + 1)
        for channel in engine._active_channels[cut]
    )
    left = engine.build_left(state.tensors)
    for message in left:
        engine._validated_message(message, message.cut)
    numerator = engine.boundary_scalar(left[-1], engine.nsites)
    norm = state._norm_frontier.scalar(state.tensors)
    assert numerator / norm == pytest.approx(
        state.expectation(),
        abs=2.0e-12,
    )


def test_u1_packed_coordinates_allow_empty_charge_channels(monkeypatch):
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2,), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        seed=5,
        frontier_backend="identity_block",
    )
    engine = state._hamiltonian_frontier
    engine._storage_coordinate_cache.clear()
    monkeypatch.setattr(engine, "block_shape", lambda *_args: (0, 2, 3))
    monkeypatch.setattr(
        engine,
        "_physical_support_indices",
        lambda *_args: None,
    )

    coordinates = engine._storage_coordinates(0, 0)

    assert coordinates.dtype == np.int32
    assert coordinates.shape == (0, 3)


def test_u1_termwise_frontier_preserves_physical_packing_and_energy(monkeypatch):
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2, 3), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    identity = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        seed=41,
        frontier_backend="identity_block",
    )
    termwise = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        tensors=identity.tensors,
        frontier_backend="termwise",
    )
    engine = termwise._hamiltonian_frontier

    assert all(component.charge_resolved for component in engine._engines)
    assert any(component.physical_packing_ratio < 1.0 for component in engine._engines)
    assert termwise.expectation() == pytest.approx(identity.expectation(), abs=2.0e-12)
    reference_numerator = engine.scalar(termwise.tensors)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("logical frontier expansion is forbidden")

    monkeypatch.setattr(engine._identity_engine, "_logical_block", forbidden)
    for component in engine._engines:
        monkeypatch.setattr(component, "_logical_block", forbidden)
    assert engine.scalar(termwise.tensors) == pytest.approx(
        reference_numerator,
        abs=2.0e-12,
    )


def test_u1_core_contractions_never_expand_packed_frontier_blocks(monkeypatch):
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2, 3), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        seed=47,
        frontier_backend="identity_block",
    )
    engine = state._hamiltonian_frontier
    reference_left = engine.build_left(state.tensors)
    reference_right = engine.build_right(state.tensors)
    site = 1
    vector = np.asarray(state.tensors[site]).reshape(-1)
    reference_scalar = engine.boundary_scalar(reference_left[-1], engine.nsites)
    reference_matrix = engine.hole_matrix(
        site, reference_left[site], reference_right[site + 1]
    )
    reference_left_enrichment = tuple(
        engine.left_enrichment_components(
            site, reference_left[site], vector
        )
    )
    reference_right_enrichment = tuple(
        engine.right_enrichment_components(
            site, reference_right[site + 1], vector
        )
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("logical frontier expansion is forbidden")

    monkeypatch.setattr(engine, "_logical_block", forbidden)
    left = engine.build_left(state.tensors)
    right = engine.build_right(state.tensors)
    assert engine.boundary_scalar(left[-1], engine.nsites) == pytest.approx(
        reference_scalar,
        abs=2.0e-12,
    )
    np.testing.assert_allclose(
        engine.hole_matrix(site, left[site], right[site + 1]),
        reference_matrix,
        atol=2.0e-12,
    )
    action = engine.prepare_hole_action(site, left[site], right[site + 1])
    assert action.backend in {"packed-u1-csr", "packed-u1-dense-fused"}
    np.testing.assert_allclose(action(vector), reference_matrix @ vector, atol=2.0e-12)
    np.testing.assert_allclose(
        action.many(np.stack((vector, 0.5 * vector))),
        np.stack((reference_matrix @ vector, 0.5 * reference_matrix @ vector)),
        atol=2.0e-12,
    )
    for actual, reference in zip(
        engine.left_enrichment_components(site, left[site], vector),
        reference_left_enrichment,
    ):
        np.testing.assert_allclose(actual, reference, atol=2.0e-12)
    for actual, reference in zip(
        engine.right_enrichment_components(site, right[site + 1], vector),
        reference_right_enrichment,
    ):
        np.testing.assert_allclose(actual, reference, atol=2.0e-12)


def test_compiled_u1_route_builders_match_python_fallback():
    from pyqed.mps import cpp_davidson

    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.build_packed_advance_routes is None
        or cpp_davidson.build_packed_hole_routes is None
    ):
        pytest.skip("compiled packed U(1) route builders are unavailable")
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2, 3), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    compiled = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        seed=53,
        frontier_backend="identity_block",
    )
    fallback = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        tensors=compiled.tensors,
        frontier_backend="identity_block",
    )
    fallback._hamiltonian_frontier._compiled_route_builders_cache = (None, None)
    compiled_engine = compiled._hamiltonian_frontier
    fallback_engine = fallback._hamiltonian_frontier
    compiled_left = compiled_engine.build_left(compiled.tensors)
    compiled_right = compiled_engine.build_right(compiled.tensors)
    fallback_left = fallback_engine.build_left(fallback.tensors)
    fallback_right = fallback_engine.build_right(fallback.tensors)
    site = 1
    vector = np.asarray(compiled.tensors[site]).reshape(-1)

    assert compiled_engine.packed_route_backend in {"cpp", "cpp-fused"}
    np.testing.assert_allclose(
        compiled_engine.hole_action(
            site, compiled_left[site], compiled_right[site + 1], vector
        ),
        fallback_engine.hole_action(
            site, fallback_left[site], fallback_right[site + 1], vector
        ),
        atol=2.0e-12,
    )
    assert compiled_engine.boundary_scalar(
        compiled_left[-1], compiled_engine.nsites
    ) == pytest.approx(
        fallback_engine.boundary_scalar(
            fallback_left[-1], fallback_engine.nsites
        ),
        abs=2.0e-12,
    )


def test_packed_u1_routes_reuse_compact_topology_within_memory_budget():
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2, 3), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        parents,
        abelian_layout=layout,
        seed=59,
        frontier_backend="identity_block",
        route_memory=1,
    )
    engine = state._hamiltonian_frontier
    engine.clear_contraction_plans()
    before = engine.packed_route_cache_stats.copy()

    first = engine.build_left(state.tensors)
    after_first = engine.packed_route_cache_stats.copy()
    second = engine.build_left(state.tensors)
    after_second = engine.packed_route_cache_stats.copy()

    assert after_first["misses"] > before["misses"]
    assert after_second["misses"] == after_first["misses"]
    assert after_second["hits"] > after_first["hits"]
    assert after_second["bytes"] <= after_second["limit_bytes"]
    assert after_second["plans"] > 0
    for left, right in zip(first, second):
        for left_block, right_block in zip(left.blocks, right.blocks):
            np.testing.assert_allclose(left_block, right_block, atol=2.0e-13)
    for coordinates in engine._storage_coordinate_cache.values():
        assert coordinates.dtype == np.int32
    for inverse in engine._storage_inverse_cache.values():
        assert inverse.dtype == np.int32
    for plan in engine._packed_advance_plan_cache.values():
        assert all(values.dtype == np.int32 for values in plan[:4])


def test_packed_u1_auto_environment_reuses_full_fixed_side_when_bounded():
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2, 3), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        parents,
        abelian_layout=layout,
        seed=61,
        frontier_backend="identity_block",
    )

    state.run(
        nsweeps=1,
        solver="direct",
        gauge=None,
        environment_cache="auto",
        environment_memory=1,
    )

    assert state.history[-1]["environment_cache_requested"] == "auto"
    assert state.history[-1]["environment_cache"] == "full"


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
        masks = layout.local_masks(frontier.physical_groups)
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

    with pytest.raises(ValueError, match="inconsistent with abelian_layout"):
        AbelianFrontierTiedLETTA(
            _heisenberg_chain(),
            (2,) * 4,
            ((1,), (2,), (3,), ()),
            abelian_layout=layout,
            bond_dim=4,
            seed=31,
        )
