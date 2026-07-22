import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta.block_mpo_frontier import BlockMPOFrontier
from pyqed.letta.mpo_frontier import MPOFrontier
from tests.test_letta_frontier_tying import _states
from tests.test_letta_mpo_frontier import _identity_mpo


def _engines(state, mpo):
    arguments = (
        state.dims,
        state.physical_sites,
        [tensor.shape for tensor in state.tensors],
        mpo.tensors,
    )
    return MPOFrontier(*arguments), BlockMPOFrontier(*arguments)


def _assert_messages_close(left, right, *, atol=2.0e-13):
    assert left.cut == right.cut
    assert len(left.blocks) == len(right.blocks)
    for left_block, right_block in zip(left.blocks, right.blocks):
        np.testing.assert_allclose(left_block, right_block, atol=atol)


def test_identity_aware_blocks_match_dense_channel_frontiers():
    state, _dense = _states(seed=41)
    rng = np.random.default_rng(12)
    for mpo in (_identity_mpo(state.dims), state.hamiltonian.to_mpo()):
        dense_engine, block_engine = _engines(state, mpo)
        np.testing.assert_allclose(
            block_engine.scalar(state.tensors),
            dense_engine.scalar(state.tensors),
            atol=3.0e-13,
        )
        dense_left = dense_engine.build_left(state.tensors)
        dense_right = dense_engine.build_right(state.tensors)
        block_left = block_engine.build_left(state.tensors)
        block_right = block_engine.build_right(state.tensors)
        for site, tensor in enumerate(state.tensors):
            dense_matrix = dense_engine.hole_matrix(
                site, dense_left[site], dense_right[site + 1]
            )
            block_matrix = block_engine.hole_matrix(
                site, block_left[site], block_right[site + 1]
            )
            np.testing.assert_allclose(block_matrix, dense_matrix, atol=5.0e-13)

            probe = rng.normal(size=tensor.size) + 1.0j * rng.normal(size=tensor.size)
            dense_action = dense_engine.hole_action(
                site, dense_left[site], dense_right[site + 1], probe
            )
            block_action = block_engine.hole_action(
                site, block_left[site], block_right[site + 1], probe
            )
            np.testing.assert_allclose(block_action, dense_action, atol=5.0e-13)

        assert block_engine.peak_message_elements <= dense_engine.peak_message_elements
        assert (
            block_engine.dense_peak_message_elements
            == dense_engine.peak_message_elements
        )


def test_block_messages_remain_valid_during_directional_updates():
    state, _dense = _states(seed=43)
    _dense_engine, engine = _engines(state, state.hamiltonian.to_mpo())
    fixed_right = engine.build_right(state.tensors)
    moving_left = engine.left_boundary()
    rng = np.random.default_rng(19)

    for site in range(len(state.dims)):
        state.tensors[site] += 1.0e-3 * rng.normal(size=state.tensors[site].shape)
        moving_left = engine.advance_left(moving_left, state.tensors, site)
        fresh_left = engine.build_left(state.tensors)[site + 1]
        _assert_messages_close(moving_left, fresh_left)
        if site + 1 < len(state.dims):
            fresh_right = engine.build_right(state.tensors)[site + 1]
            _assert_messages_close(fixed_right[site + 1], fresh_right)

    fixed_left = engine.build_left(state.tensors)
    moving_right = engine.right_boundary()
    for site in range(len(state.dims) - 1, -1, -1):
        state.tensors[site] += 1.0e-3 * rng.normal(size=state.tensors[site].shape)
        moving_right = engine.advance_right(moving_right, state.tensors, site)
        fresh_right = engine.build_right(state.tensors)[site]
        _assert_messages_close(moving_right, fresh_right)
        if site > 0:
            fresh_left = engine.build_left(state.tensors)[site]
            _assert_messages_close(fixed_left[site], fresh_left)

    np.testing.assert_allclose(
        engine.boundary_scalar(moving_right, 0),
        engine.scalar(state.tensors),
        atol=2.0e-13,
    )


def test_transition_batching_reduces_exact_contraction_plan_count():
    state, _dense = _states(seed=47)
    _dense_engine, engine = _engines(state, state.hamiltonian.to_mpo())
    left = engine.build_left(state.tensors)
    right = engine.build_right(state.tensors)
    for site, tensor in enumerate(state.tensors):
        engine.hole_matrix(site, left[site], right[site + 1])
        engine.hole_action(
            site,
            left[site],
            right[site + 1],
            np.ones(tensor.size),
        )

    grouped_plans = sum(
        len(groups)
        for family in (
            engine._left_transition_groups,
            engine._right_transition_groups,
            engine._hole_transition_groups,
            engine._hole_transition_groups,
        )
        for groups in family
    )
    ungrouped_plans = 4 * sum(len(values) for values in engine._transitions)
    assert engine.plan_count == grouped_plans
    assert grouped_plans < ungrouped_plans


def test_four_by_four_identity_channel_message_reduction():
    nsites = 16
    nearest, diagonals = square_j1_j2_bonds(4, 4)
    weighted_bonds = tuple((left, right, 1.0) for left, right in nearest)
    weighted_bonds += tuple((left, right, 0.5) for left, right in diagonals)
    parent_sets = parent_sets_from_edges(nsites, nearest)
    physical_sites = tuple((site,) + tuple(parent_sets[site]) for site in range(nsites))
    bonds = (1,) + (4,) * (nsites - 1) + (1,)
    tensor_shapes = tuple(
        (bonds[site], bonds[site + 1])
        + tuple(2 for _physical_site in physical_sites[site])
        for site in range(nsites)
    )
    mpo = heisenberg_local_hamiltonian(nsites, weighted_bonds).to_mpo()
    engine = BlockMPOFrontier(
        (2,) * nsites,
        physical_sites,
        tensor_shapes,
        mpo.tensors,
    )

    assert engine.dense_peak_message_elements == 143_360
    assert engine.peak_message_elements == 17_920
    assert engine.dense_total_message_elements == 1_430_978
    assert engine.total_message_elements == 189_922
