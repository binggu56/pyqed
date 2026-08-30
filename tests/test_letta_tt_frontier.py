import numpy as np
import pytest

from pyqed.letta import FrontierTiedLETTA
from pyqed.letta.mpo_frontier import MPOFrontier
from pyqed.letta.tt_frontier import (
    TermwiseBlockMPOFrontier,
    TermwiseTTMPOFrontier,
    TTFrontier,
    TTMPOFrontier,
    _term_product_mpos,
)
from pyqed.letta import LocalTerm
from tests.test_letta_frontier_tying import _states
from tests.test_letta_mpo_frontier import _identity_mpo


def _engines(state, mpo, *, paired_sites=None, **tt_options):
    arguments = (
        state.dims,
        state.physical_groups,
        [tensor.shape for tensor in state.tensors],
        mpo.tensors,
    )
    return (
        MPOFrontier(*arguments, paired_sites=paired_sites),
        TTMPOFrontier(*arguments, paired_sites=paired_sites, **tt_options),
    )


def test_dense_tt_svd_round_trip_and_rounding_diagnostics():
    rng = np.random.default_rng(72)
    array = rng.normal(size=(3, 4, 2, 5)) + 0.3j * rng.normal(size=(3, 4, 2, 5))
    labels = ("virtual", "bra-4", "ket-4", "operator")

    exact = TTFrontier.from_dense(array, labels)
    np.testing.assert_allclose(exact.to_dense(), array, atol=3.0e-14)
    assert exact.labels == labels
    assert exact.shape == array.shape
    assert exact.last_round.algorithm == "dense_tt_svd"
    assert exact.last_round.densified_input
    assert exact.last_round.discarded_weight == 0.0

    rounded = exact.round(max_rank=2)
    error_sq = float(np.linalg.norm(rounded.to_dense() - array) ** 2)
    assert max(rounded.ranks) <= 2
    assert rounded.last_round.algorithm == "tt_round"
    assert not rounded.last_round.densified_input
    np.testing.assert_allclose(
        rounded.last_round.discarded_weight,
        error_sq,
        rtol=3.0e-13,
        atol=3.0e-13,
    )

    tolerance = 0.25 * np.linalg.norm(array)
    tolerance_truncated = TTFrontier.from_dense(array, labels, rtol=0.25)
    assert np.linalg.norm(tolerance_truncated.to_dense() - array) <= tolerance
    assert tolerance_truncated.last_round.discarded_weight > 0.0


def test_charge_resolved_operator_schmidt_keeps_u1_transfers_homogeneous():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    exchange = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    qns = (((1,), (-1,)),) * 2
    components = _term_product_mpos(
        (2, 2),
        LocalTerm((0, 1), exchange),
        local_qns=qns,
    )

    assert len(components) == 3
    np.testing.assert_allclose(
        sum(component.to_dense() for component in components),
        exchange,
        atol=3.0e-15,
    )
    for component in components:
        transfers = []
        for site in range(2):
            operator = component.tensors[site][0, 0]
            local = {
                tuple(qns[site][bra][axis] - qns[site][ket][axis]
                      for axis in range(len(qns[site][bra])))
                for bra, ket in zip(*np.nonzero(np.abs(operator) > 1.0e-14))
            }
            assert len(local) == 1
            transfers.append(next(iter(local)))
        assert tuple(a + b for a, b in zip(*transfers)) == (0,)


def test_labelled_tt_factor_operations_preserve_values_and_order():
    left_array = np.arange(12.0).reshape(2, 3, 2) + 1.0
    right_array = np.linspace(0.2, 1.3, 12).reshape(2, 3, 2)
    left = TTFrontier.from_dense(left_array, ("a", "b", "c"))
    right = TTFrontier.from_dense(right_array, ("a", "b", "c"))

    permuted = left.transpose_labels(("c", "a", "b"))
    np.testing.assert_allclose(
        permuted.to_dense(), left_array.transpose(2, 0, 1), atol=2.0e-13
    )
    restored = permuted.transpose_labels(left.labels)
    np.testing.assert_allclose(restored.to_dense(), left_array, atol=2.0e-13)

    product = left.hadamard(right)
    np.testing.assert_allclose(
        product.to_dense(), left_array * right_array, atol=2.0e-13
    )
    reduced = product.sum_over(("a", "c"))
    assert reduced.labels == ("b",)
    np.testing.assert_allclose(
        reduced.to_dense(), np.sum(left_array * right_array, axis=(0, 2)), atol=2.0e-13
    )

    embedded = left.embed(("new", "a", "b", "c"), (4, 2, 3, 2))
    expected = np.broadcast_to(left_array, (4,) + left_array.shape)
    np.testing.assert_allclose(embedded.to_dense(), expected, atol=2.0e-13)

    conjugated = left.conjugate_relabel(
        {"a": "c", "c": "a"}, labels=("a", "b", "c")
    )
    np.testing.assert_allclose(
        conjugated.to_dense(), left_array.conj().transpose(2, 1, 0), atol=2.0e-13
    )


def test_structured_hole_adjoint_is_exact_for_truncated_complex_messages():
    state, _dense = _states(seed=17)
    mpo_tensors = [np.eye(dim, dtype=complex)[None, None] for dim in state.dims]
    mpo_tensors[1] = np.array(
        [[1.0 + 0.2j, 0.7 - 0.4j], [-0.3 + 0.6j, 0.2 - 0.1j]]
    )[None, None]
    mpo = MPO(mpo_tensors, sites=state.sites)
    exact, tt = _engines(state, mpo, max_rank=2)
    site = 1
    rng = np.random.default_rng(923)
    raw_left = rng.normal(size=tt.message_shape(site)) + 1.0j * rng.normal(
        size=tt.message_shape(site)
    )
    raw_right = rng.normal(size=tt.message_shape(site + 1)) + 1.0j * rng.normal(
        size=tt.message_shape(site + 1)
    )
    left = TTFrontier.from_dense(
        raw_left, tt.message_labels(site), max_rank=2
    )
    right = TTFrontier.from_dense(
        raw_right, tt.message_labels(site + 1), max_rank=2
    )
    assert left.last_round.discarded_weight > 0.0
    assert right.last_round.discarded_weight > 0.0

    dense_left = left.to_dense()
    dense_right = right.to_dense()
    matrix = exact.hole_matrix(site, dense_left, dense_right)
    x = rng.normal(size=matrix.shape[0]) + 1.0j * rng.normal(size=matrix.shape[0])
    y = rng.normal(size=matrix.shape[1]) + 1.0j * rng.normal(size=matrix.shape[1])

    # The structured paths must not call either input frontier's dense method.
    def dense_frontier_forbidden():
        raise AssertionError("a dense input frontier was materialized")

    left.to_dense = dense_frontier_forbidden
    right.to_dense = dense_frontier_forbidden
    action_y = tt.hole_action(site, left, right, y)
    adjoint_x = tt.hole_adjoint_action(site, left, right, x)

    np.testing.assert_allclose(action_y, matrix @ y, atol=8.0e-13)
    np.testing.assert_allclose(adjoint_x, matrix.T.conj() @ x, atol=8.0e-13)
    np.testing.assert_allclose(
        np.vdot(x, action_y), np.vdot(adjoint_x, y), atol=1.5e-12
    )
    assert tt.last_hole_diagnostics.method == "structured_adjoint"
    assert not tt.last_hole_diagnostics.used_dense_frontier


def test_structured_identity_and_generic_mpo_messages_match_dense_frontiers():
    state, _dense = _states(seed=17)
    cases = (
        (_identity_mpo(state.dims), ()),
        (state.hamiltonian.to_mpo(), None),
    )
    for mpo, paired_sites in cases:
        exact, tt = _engines(state, mpo, paired_sites=paired_sites)
        exact_left = exact.build_left(state.tensors)
        tt_left = tt.build_left(state.tensors)
        for cut, (actual, reference) in enumerate(zip(tt_left, exact_left)):
            assert actual.labels == tt.message_labels(cut)
            np.testing.assert_allclose(actual.to_dense(), reference, atol=7.0e-13)

        exact_right = exact.build_right(state.tensors)
        tt_right = tt.build_right(state.tensors)
        for cut, (actual, reference) in enumerate(zip(tt_right, exact_right)):
            assert actual.labels == tt.message_labels(cut)
            np.testing.assert_allclose(actual.to_dense(), reference, atol=7.0e-13)

        probe = np.linspace(-0.35, 0.65, state.tensors[1].size).astype(complex)
        np.testing.assert_allclose(
            tt.hole_action(1, tt_left[1], tt_right[2], probe),
            exact.hole_action(1, exact_left[1], exact_right[2], probe),
            atol=9.0e-13,
        )
        assert not tt.last_hole_diagnostics.used_dense_frontier

        np.testing.assert_allclose(
            tt.scalar(state.tensors), exact.scalar(state.tensors), atol=7.0e-13
        )
        diagnostics = tt.diagnostics
        assert len(diagnostics.advances) == len(state.dims)
        assert diagnostics.dense_frontier_absorptions == 0
        assert diagnostics.peak_local_factor_elements > 0
        assert diagnostics.peak_product_storage_elements > 0
        assert all(not item.used_dense_frontier for item in diagnostics.advances)


def test_identity_block_hamiltonian_can_use_an_exact_boundary_tt_norm():
    reference, _dense = _states(seed=61)
    hybrid = FrontierTiedLETTA(
        reference.hamiltonian,
        reference.dims,
        reference.parent_sets,
        bond_dim=reference.bond_dim,
        tensors=reference.tensors,
        frontier_backend="identity_block",
        tt_norm_backend="tensor_train",
    )
    exact = FrontierTiedLETTA(
        reference.hamiltonian,
        reference.dims,
        reference.parent_sets,
        bond_dim=reference.bond_dim,
        tensors=hybrid.tensors,
        frontier_backend="identity_block",
    )
    exact.tensors = [tensor.copy() for tensor in hybrid.tensors]

    assert hybrid.uses_tensor_train_frontier
    assert hybrid.norm_contraction_is_exact
    assert hybrid.hamiltonian_contraction_is_exact
    np.testing.assert_allclose(hybrid.norm(), exact.norm(), atol=8.0e-13)
    np.testing.assert_allclose(
        hybrid.expectation(), exact.expectation(), atol=8.0e-13
    )
    site = 1
    probe = np.linspace(-0.4, 0.7, hybrid.tensors[site].size)
    np.testing.assert_allclose(
        hybrid.metric_action(site, probe),
        exact.metric_action(site, probe),
        atol=1.0e-12,
    )


def test_rank_limited_scalar_converges_to_exact_norm_and_energy():
    state, _dense = _states(seed=17)
    cases = (
        (_identity_mpo(state.dims), (), (1, 2, 4)),
        (state.hamiltonian.to_mpo(), None, (1, 4, 16)),
    )
    for mpo, paired_sites, ranks in cases:
        exact, _tt = _engines(state, mpo, paired_sites=paired_sites)
        reference = exact.scalar(state.tensors)
        errors = []
        for rank in ranks:
            _exact, tt = _engines(state, mpo, paired_sites=paired_sites, max_rank=rank)
            errors.append(abs(tt.scalar(state.tensors) - reference))
            assert all(
                max(message.target_ranks, default=1) <= rank
                for message in tt.diagnostics.advances
            )
        assert errors[-1] < 8.0e-13
        assert errors[-1] < errors[0]

    mpo = state.hamiltonian.to_mpo()
    _exact, compressed_transfer = _engines(
        state,
        mpo,
        max_rank=4,
        transfer_max_rank=3,
    )
    compressed_transfer.scalar(state.tensors)
    assert any(
        item.local_factor_discarded_weight > 0.0
        for item in compressed_transfer.diagnostics.advances
    )
    assert all(
        max(item.local_factor_ranks, default=1) <= 3
        for item in compressed_transfer.diagnostics.advances
    )


def test_structured_hole_action_and_dense_validation_fallback_match_exact():
    state, _dense = _states(seed=23)
    exact, tt = _engines(
        state,
        state.hamiltonian.to_mpo(),
        absorption="dense",
        max_rank=None,
    )
    np.testing.assert_allclose(
        tt.scalar(state.tensors), exact.scalar(state.tensors), atol=5.0e-13
    )
    assert tt.diagnostics.dense_frontier_absorptions == len(state.dims)

    left = tt.build_left(state.tensors)
    right = tt.build_right(state.tensors)
    vector = np.linspace(-0.4, 0.7, state.tensors[1].size)
    reference = exact.hole_action(
        1,
        left[1].to_dense(),
        right[2].to_dense(),
        vector,
    )
    np.testing.assert_allclose(
        tt.hole_action(1, left[1], right[2], vector),
        reference,
        atol=8.0e-13,
    )
    assert not tt.last_hole_diagnostics.used_dense_frontier
    assert tt.last_hole_diagnostics.local_factor_elements > 0
    np.testing.assert_allclose(
        tt.hole_action(1, left[1], right[2], vector, allow_dense=True),
        reference,
        atol=5.0e-13,
    )
    assert tt.last_hole_diagnostics.used_dense_frontier


def test_frontier_tied_letta_tensor_train_backend_matches_exact_actions():
    exact, _dense = _states(seed=29)
    tt = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tt",
    )

    assert tt.frontier_backend == "tensor_train"
    assert tt.contraction_is_exact
    np.testing.assert_allclose(tt.norm(), exact.norm(), atol=8.0e-13)
    np.testing.assert_allclose(tt.expectation(), exact.expectation(), atol=8.0e-13)
    probe = np.linspace(-0.4, 0.7, tt.tensors[1].size)
    np.testing.assert_allclose(
        tt.metric_action(1, probe), exact.metric_action(1, probe), atol=1.0e-12
    )
    np.testing.assert_allclose(
        tt.hamiltonian_action(1, probe),
        exact.hamiltonian_action(1, probe),
        atol=1.0e-12,
    )
    assert isinstance(tt._hamiltonian_frontier, TermwiseTTMPOFrontier)
    assert tt.peak_frontier_elements <= exact.peak_frontier_elements
    assert tt.peak_compressed_frontier_elements > 0
    assert tt.tt_diagnostics is not None


def test_term_grouped_tt_channels_are_exact_and_reduce_engine_count():
    exact, _dense = _states(seed=229)
    component = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tensor_train",
        tt_channels="component",
    )
    grouped = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tensor_train",
        tt_channels="term",
    )

    assert grouped.tt_channels == "term"
    assert len(grouped._hamiltonian_frontier._engines) < len(
        component._hamiltonian_frontier._engines
    )
    np.testing.assert_allclose(grouped.expectation(), exact.expectation(), atol=1e-12)
    probe = np.linspace(-0.4, 0.7, grouped.tensors[1].size)
    np.testing.assert_allclose(
        grouped.hamiltonian_action(1, probe),
        exact.hamiltonian_action(1, probe),
        atol=2e-12,
    )


def test_tt_frontier_direct_sum_addition_is_exact():
    rng = np.random.default_rng(230)
    left_dense = rng.normal(size=(2, 3, 2))
    right_dense = rng.normal(size=(2, 3, 2))
    left = TTFrontier.from_dense(left_dense, labels=("a", "b", "c"))
    right = TTFrontier.from_dense(right_dense, labels=("a", "b", "c"))

    combined = left.add(right)

    np.testing.assert_allclose(combined.to_dense(), left_dense + right_dense, atol=1e-13)


def test_tensor_train_backend_tracks_approximation_and_matrix_free_guards():
    exact, _dense = _states(seed=31)
    tt = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tensor_train",
        max_rank=3,
        transfer_max_rank=4,
    )
    copied = tt.copy()

    assert not tt.contraction_is_exact
    assert tt.norm_contraction_is_exact
    assert not tt.hamiltonian_contraction_is_exact
    assert copied.tt_options == tt.tt_options
    assert copied.tt_norm_backend == "exact"
    assert copied.tt_hermitize
    for actual, reference in zip(copied.tensors, tt.tensors):
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_allclose(copied.energy, copied.expectation(), atol=1.0e-13)
    assert tt.hamiltonian_action_is_hermitian
    np.testing.assert_allclose(tt.local_metric(0), exact.local_metric(0), atol=1.0e-12)
    with pytest.raises(ValueError, match="matrix_free"):
        tt.optimize_site(0, solver="direct")
    with pytest.raises(NotImplementedError, match="dense local"):
        tt.local_operators(0)
    with pytest.raises(ValueError, match="stochastic reconfiguration"):
        tt.run(nsweeps=0, natural_gradient_every=1)

    all_tt = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tensor_train",
        tt_norm_backend="tensor_train",
        max_rank=3,
        transfer_max_rank=4,
    )
    assert not all_tt.norm_contraction_is_exact
    with pytest.raises(NotImplementedError, match="frontier canonicalization"):
        all_tt.canonicalize_frontier_gauge()
    with pytest.raises(ValueError, match="exact norm contraction"):
        all_tt.run(nsweeps=1)

    nonhermitian = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tensor_train",
        max_rank=2,
        transfer_max_rank=3,
        tt_hermitize=False,
    )
    assert not nonhermitian.hamiltonian_action_is_hermitian
    with pytest.raises(ValueError, match="Hermitized"):
        nonhermitian.run(nsweeps=1)


def test_integrated_truncated_tt_hamiltonian_action_is_hermitized():
    exact, _dense = _states(seed=43)
    tt = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tensor_train",
        max_rank=2,
        transfer_max_rank=3,
        tt_hermitize=True,
    )
    site = 1
    environment = tt.site_environment(site)
    rng = np.random.default_rng(991)
    left = rng.normal(size=tt.tensors[site].size) + 0.3j * rng.normal(
        size=tt.tensors[site].size
    )
    right = rng.normal(size=tt.tensors[site].size) + 0.2j * rng.normal(
        size=tt.tensors[site].size
    )
    action_right = tt.hamiltonian_action(site, right, environment=environment)
    action_left = tt.hamiltonian_action(site, left, environment=environment)

    np.testing.assert_allclose(
        np.vdot(left, action_right),
        np.vdot(action_left, right),
        atol=2.0e-11,
    )


def test_exact_tensor_train_backend_completes_matrix_free_sweep():
    exact, _dense = _states(seed=37)
    tt = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tensor_train",
    )
    initial = tt.energy
    tt.run(nsweeps=1, solver="auto", eig_tol=1.0e-8, maxiter=50)

    assert len(tt.history) == 1
    assert tt.history[0]["contraction_is_exact"]
    assert all(update.solver == "matrix_free" for update in tt.history[0]["updates"])
    assert tt.energy <= initial + 1.0e-11


@pytest.mark.parametrize("seed", [1, 3])
def test_truncated_tt_sweep_globally_checks_proposals_and_reports_fresh_energy(seed):
    exact, _dense = _states(seed=seed)
    tt = FrontierTiedLETTA(
        exact.hamiltonian,
        exact.dims,
        exact.parent_sets,
        bond_dim=exact.bond_dim,
        tensors=[tensor.copy() for tensor in exact.tensors],
        frontier_backend="tensor_train",
        max_rank=1,
        transfer_max_rank=2,
        tt_hermitize=True,
    )
    initial = tt.energy
    tt.run(nsweeps=2, solver="matrix_free", eig_tol=1.0e-8, maxiter=50)

    np.testing.assert_allclose(tt.energy, tt.expectation(), atol=1.0e-11)
    assert tt.energy <= initial + 1.0e-10
    for sweep in tt.history:
        for update in sweep["updates"]:
            assert update.energy <= update.energy_before + 1.0e-10
            if update.accepted:
                assert "exact global energy check" in update.message
