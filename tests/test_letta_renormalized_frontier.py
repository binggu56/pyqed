import numpy as np
import pytest

from examples.mps.converge_frontier_letta_u1_two_site_6x6 import (
    _certify_pair_stationarity,
    _model,
    _save_snapshot,
    _state_from_snapshot,
)
from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    FrontierTiedLETTA,
    LocalHamiltonian,
    LocalMPO,
    LocalTerm,
    SymmetryLayout,
    TermRenormalizedFrontier,
)
from pyqed.letta.renormalized_frontier import _renormalized_operator_tensors


def _exchange():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def _hamiltonian():
    return LocalHamiltonian(
        (2,) * 4,
        (
            LocalTerm((0,), 0.13 * np.diag([1.0, -1.0])),
            LocalTerm((0, 1), 0.8 * _exchange()),
            LocalTerm((0, 3), 0.3 * _exchange()),
            LocalTerm((1, 2), -0.2 * _exchange()),
            LocalTerm((2, 3), 0.7 * _exchange()),
        ),
        constant=0.17,
    )


def test_renormalized_operator_network_matches_dense_local_hamiltonian():
    hamiltonian = _hamiltonian()
    tensors, diagnostics = _renormalized_operator_tensors(hamiltonian)
    represented = LocalMPO(hamiltonian.dims, tensors).to_dense()

    np.testing.assert_allclose(
        represented,
        hamiltonian.to_dense(),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert diagnostics["max_bond_dim"] <= max(hamiltonian.to_mpo().bond_dims)
    assert diagnostics["representation"] == "renormalized_complementary_operators"

    tiny = LocalHamiltonian(
        (2, 2),
        (LocalTerm((0, 1), 1.0e-30 * _exchange()),),
    )
    tensors, _diagnostics = _renormalized_operator_tensors(tiny)
    np.testing.assert_allclose(
        LocalMPO(tiny.dims, tensors).to_dense(),
        tiny.to_dense(),
        rtol=2.0e-14,
        atol=1.0e-44,
    )


def test_renormalized_frontier_matches_identity_block_graph_contractions():
    hamiltonian = _hamiltonian()
    parents = ((1, 3), (2,), (3,), ())
    reference = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        frontier_backend="identity_block",
        seed=23,
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        tensors=reference.tensors,
        frontier_backend="renormalized",
    )
    state.tensors = [tensor.copy() for tensor in reference.tensors]

    assert isinstance(state._hamiltonian_frontier, TermRenormalizedFrontier)
    assert state.hamiltonian_mpo is None
    np.testing.assert_allclose(
        state.expectation(),
        reference.expectation(),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    for site in range(len(state.dims)):
        metric, effective = state.local_operators(site)
        reference_metric, reference_effective = reference.local_operators(site)
        np.testing.assert_allclose(metric, reference_metric, atol=5.0e-13)
        np.testing.assert_allclose(effective, reference_effective, atol=5.0e-13)
    for site in range(len(state.dims) - 1):
        metric, effective = state.pair_local_operators(site)
        reference_metric, reference_effective = reference.pair_local_operators(site)
        np.testing.assert_allclose(metric, reference_metric, atol=2.0e-12)
        np.testing.assert_allclose(effective, reference_effective, atol=2.0e-12)


def test_renormalized_u1_pair_actions_match_identity_block():
    hamiltonian = _hamiltonian()
    parents = ((1, 3), (2,), (3,), ())
    layout = SymmetryLayout.spin_half(4, target_two_sz=0, bond_dims=2)
    reference = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        abelian_layout=layout,
        frontier_backend="identity_block",
        seed=29,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        abelian_layout=layout,
        tensors=reference.tensors,
        frontier_backend="renormalized",
    )
    state.tensors = [tensor.copy() for tensor in reference.tensors]

    plan = state._pair_plan(1)
    assert plan.hamiltonian_engine.uses_outer_messages
    metric, effective = state.pair_local_operators(1)
    reference_metric, reference_effective = reference.pair_local_operators(1)
    np.testing.assert_allclose(metric, reference_metric, atol=2.0e-12)
    np.testing.assert_allclose(effective, reference_effective, atol=2.0e-12)

    merged, _union_sites = state._merged_pair_tensor(1)
    rng = np.random.default_rng(31)
    vector = rng.normal(size=merged.size) + 1.0j * rng.normal(size=merged.size)
    np.testing.assert_allclose(
        state.pair_metric_action(1, vector),
        reference.pair_metric_action(1, vector),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        state.pair_hamiltonian_action(1, vector),
        reference.pair_hamiltonian_action(1, vector),
        atol=2.0e-12,
    )
    problem = state.pair_local_block_problem(1)
    reference_problem = reference.pair_local_block_problem(1)
    np.testing.assert_allclose(
        problem.metric.to_dense(),
        reference_problem.metric.to_dense(),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        problem.hamiltonian.to_dense(),
        reference_problem.hamiltonian.to_dense(),
        atol=2.0e-12,
    )


def test_matrix_free_u1_pair_update_uses_actions_without_lowest_certificate(
    monkeypatch,
):
    hamiltonian = _hamiltonian()
    parents = ((1, 3), (2,), (3,), ())
    layout = SymmetryLayout.spin_half(4, target_two_sz=0, bond_dims=2)
    dense = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        abelian_layout=layout,
        frontier_backend="renormalized",
        seed=33,
    )
    state = dense.copy()
    plan = state._pair_plan(1)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("matrix-free backend assembled a Hamiltonian matrix")

    monkeypatch.setattr(plan.hamiltonian_engine, "hole_matrix", forbidden)
    monkeypatch.setattr(state, "pair_local_block_problem", forbidden)
    update = state.optimize_two_sites(
        1,
        pair_operator_backend="matrix_free",
        merged_dense_fallback_dim=0,
        outer_cycles=1,
        eig_tol=1.0e-9,
        maxiter=128,
        max_subspace=64,
    )
    dense_update = dense.optimize_two_sites(
        1,
        pair_operator_backend="dense",
        outer_cycles=1,
        eig_tol=1.0e-9,
        maxiter=128,
        max_subspace=64,
    )

    assert update.accepted
    assert update.pair_operator_backend == "matrix_free_sector"
    assert update.merged_solve.verified
    assert not update.merged_solve.lowest_root_certified
    assert update.merged_solve.verification_kind == "action_residual"
    assert update.merged_solve.method == "recycled_block_action_davidson"
    assert update.merged_solve.hamiltonian_batch_calls > 0
    assert update.merged_solve.recycled_vectors > 0
    assert update.merged_solve.preconditioner_blocks > 0
    assert state._pair_matrix_free_recycle_cache[1].ndim == 2
    assert np.isnan(update.merged_solve.metric_dual_relative_residual)
    recycled_update = state.optimize_two_sites(
        1,
        pair_operator_backend="matrix_free",
        merged_dense_fallback_dim=0,
        outer_cycles=1,
        eig_tol=1.0e-9,
        maxiter=128,
        max_subspace=64,
    )
    assert recycled_update.merged_solve.recycled_vectors > 0
    assert recycled_update.merged_solve.preconditioner_blocks == 0
    np.testing.assert_allclose(update.energy, dense_update.energy, atol=2.0e-11)


def test_u1_pair_support_action_policy_uses_profiled_preparation():
    hamiltonian = _hamiltonian()
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((1, 3), (2,), (3,), ()),
        abelian_layout=SymmetryLayout.spin_half(
            4,
            target_two_sz=0,
            bond_dims=2,
        ),
        frontier_backend="renormalized",
        seed=34,
    )
    site = 1
    plan = state._pair_plan(site)
    environment = state.pair_environment(site)
    support = state._pair_support_indices(
        site,
        plan.union_sites,
        plan.merged_shape,
    )

    easy = state.pair_hamiltonian_support_operator(
        site,
        support,
        environment=environment,
        action_backend="auto",
        expected_action_calls=4,
        prepared_min_action_calls=24,
    )
    hard = state.pair_hamiltonian_support_operator(
        site,
        support,
        environment=environment,
        action_backend="auto",
        expected_action_calls=24,
        prepared_min_action_calls=24,
    )
    fused = state.pair_hamiltonian_support_operator(
        site,
        support,
        environment=environment,
        action_backend="fused",
    )

    assert easy.backend == "full_scatter"
    assert hard.backend == "prepared_support_csr"
    assert hard.stored_elements > 0
    assert fused.backend == "fused_support"
    assert fused.peak_elements >= fused.stored_elements > 0


def test_matrix_free_action_root_is_not_mislabeled_lowest():
    state = FrontierTiedLETTA(
        _hamiltonian(),
        (2,) * 4,
        ((1, 3), (2,), (3,), ()),
        bond_dim=1,
        frontier_backend="renormalized",
        seed=35,
    )
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0])
    warm = np.array([0.0, 0.0, 1.0, 0.0])
    energy, _vector, _local, diagnostics = state._solve_verified_pair_actions(
        0,
        np.eye(4),
        lambda vector: hamiltonian @ vector,
        warm,
        metric_tol=1.0e-12,
        eig_tol=1.0e-10,
        maxiter=32,
        max_subspace=4,
    )

    np.testing.assert_allclose(energy, 2.0, atol=2.0e-14)
    assert diagnostics.verified
    assert not diagnostics.lowest_root_certified
    assert diagnostics.verification_kind == "action_residual"


def test_final_dense_pair_certificate_uses_full_numerical_metric_support():
    hamiltonian = LocalHamiltonian(
        (2, 2),
        (LocalTerm((0, 1), _exchange()),),
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((), ()),
        abelian_layout=SymmetryLayout.spin_half(
            2,
            target_two_sz=0,
            bond_dims=2,
        ),
        frontier_backend="renormalized",
        seed=36,
    )
    record = _certify_pair_stationarity(
        state,
        sweep_offset=0,
        stopping_gain_per_site=1.0,
        pair_dense_max_bytes=64 * 1024**2,
        dense_estimated_peak_bytes=1,
        backend="auto",
        maxiter=128,
        max_subspace=32,
    )

    assert record["attempted"]
    assert record["passed"]
    assert record["selected_backend"] == "dense"
    assert record["roots_certified"] == record["roots_total"] == 2
    assert record["full_metric_rank_roots"] == record["roots_total"]


def test_matrix_free_recycle_and_profile_caches_survive_checkpoint(tmp_path):
    hamiltonian, parents, _nearest, _ties = _model(1, 2, 0.5)
    layout = SymmetryLayout.spin_half(
        2,
        target_two_sz=0,
        bond_dims=2,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        abelian_layout=layout,
        frontier_backend="renormalized",
        seed=38,
    )
    state.optimize_two_sites(
        0,
        pair_operator_backend="matrix_free",
        merged_dense_fallback_dim=0,
        outer_cycles=1,
    )
    path = tmp_path / "checkpoint.npz"
    _save_snapshot(
        path,
        state,
        cycle=1,
        low_gain_streak=0,
        protocol_fingerprint="test",
    )
    restored = _state_from_snapshot(
        path,
        model={
            "nrows": 1,
            "ncols": 2,
            "j2": 0.5,
            "tie_graph_mode": "all-j1",
        },
        layout=layout,
        frontier_backend="renormalized",
    )

    np.testing.assert_allclose(
        restored._pair_matrix_free_recycle_cache[0],
        state._pair_matrix_free_recycle_cache[0],
    )
    assert (
        restored._pair_backend_profile_cache[0]
        == state._pair_backend_profile_cache[0]
    )


def test_renormalized_frontier_does_not_build_a_hamiltonian_mpo(monkeypatch):
    hamiltonian = _hamiltonian()

    def fail_to_mpo(_self):
        raise AssertionError("the renormalized backend must not call to_mpo")

    monkeypatch.setattr(LocalHamiltonian, "to_mpo", fail_to_mpo)
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((1, 3), (2,), (3,), ()),
        bond_dim=2,
        frontier_backend="term-recursive",
        seed=37,
    )

    assert np.isfinite(state.expectation())
    assert state.copy().frontier_backend == "renormalized"


def test_renormalized_frontier_rejects_terms_beyond_two_sites():
    hamiltonian = LocalHamiltonian(
        (2, 2, 2),
        (LocalTerm((0, 1, 2), np.eye(8)),),
    )
    with pytest.raises(NotImplementedError, match="one- and two-site"):
        _renormalized_operator_tensors(hamiltonian)


@pytest.mark.parametrize("nsites", [1, 3])
def test_renormalized_frontier_represents_a_zero_hamiltonian(nsites):
    hamiltonian = LocalHamiltonian((2,) * nsites)
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((),) * nsites,
        frontier_backend="renormalized",
        seed=41,
    )

    assert state.expectation() == 0.0
    for site in range(nsites):
        _metric, effective = state.local_operators(site)
        assert not np.any(effective)
    if nsites > 1:
        _pair_metric, pair_effective = state.pair_local_operators(0)
        assert not np.any(pair_effective)
