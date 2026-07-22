import numpy as np
import pytest

from pyqed.letta import (
    DenseTiedLETTA,
    FrontierBondExpansion,
    FrontierGaugeUpdate,
    FrontierNaturalGradientUpdate,
    FrontierSiteUpdate,
    FrontierTiedLETTA,
    FrontierTwoSiteUpdate,
    LocalHamiltonian,
    LocalTerm,
)


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
    direct_update = direct.optimize_site(site, solver="direct")
    whitened_update = whitened.optimize_site(site, solver="whitened")

    assert direct_update.accepted
    assert whitened_update.accepted
    assert whitened_update.solver == "whitened"
    assert whitened_update.solver_metric_is_identity
    assert whitened_update.solver_metric_identity_error < 2.0e-11
    assert whitened_update.solver_coordinate_residual_norm < 2.0e-10
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

    update = state.optimize_two_sites(
        0,
        solver="whitened",
        split_random_starts=0,
    )

    assert update.accepted
    assert update.split_strategy == "variational"
    assert update.selected_start.startswith("tangent(")
    assert update.factor_random_starts == 0
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


def test_frontier_sweep_can_use_frontier_canonical_gauge():
    state, _dense = _states(seed=3)
    state.run(
        nsweeps=1,
        tol=0.0,
        solver="direct",
        frontier_canonicalization=True,
    )

    gauge_updates = state.history[0]["frontier_gauge"]
    assert len(gauge_updates) == len(state.dims) - 1
    assert all(update.applied for update in gauge_updates)


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
