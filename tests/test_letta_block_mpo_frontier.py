import numpy as np
import pytest

import pyqed.letta.block_mpo_frontier as block_mpo_module
from examples.mps.adaptive_cp_letta_j1j2_square import (
    parent_sets_from_edges,
    square_j1_j2_bonds,
)
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from pyqed.letta.block_mpo_frontier import (
    BlockMPOFrontier,
    _exact_local_tt,
    _truncated_local_tt,
)
from pyqed.letta.copy_einsum import native_available as copy_einsum_available
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


def _assert_messages_equal(left, right):
    assert left.cut == right.cut
    assert len(left.blocks) == len(right.blocks)
    for left_block, right_block in zip(left.blocks, right.blocks):
        np.testing.assert_array_equal(left_block, right_block)


def _local_tt_tensor(cores, nphysical):
    reconstructed = cores[0][0]
    for core in cores[1:]:
        reconstructed = np.tensordot(reconstructed, core, axes=(-1, 0))
    reconstructed = reconstructed[..., 0]
    inverse = (0, nphysical + 1, *range(1, nphysical + 1))
    return reconstructed.transpose(inverse)


def _expression_hole_matrix(engine, site, left, right):
    size = int(np.prod(engine.tensor_shapes[site]))
    dtype = np.result_type(
        engine.dtype,
        *[block.dtype for block in left.blocks if block is not None],
        *[block.dtype for block in right.blocks if block is not None],
    )
    result = np.zeros((size, size), dtype=dtype)
    for transitions in engine._hole_transition_groups[site]:
        left_channel, right_channel = transitions[0]
        expression, copy_factors = engine._hole_expression(
            "matrix",
            site,
            left_channel,
            right_channel,
            batch_size=len(transitions),
        )
        left_blocks = left.blocks[left_channel]
        right_blocks = right.blocks[right_channel]
        operators = engine._operators_for_transitions(site, transitions)
        if len(transitions) > 1:
            left_blocks = np.stack(
                [left.blocks[channel] for channel, _ in transitions]
            )
            right_blocks = np.stack(
                [right.blocks[channel] for _, channel in transitions]
            )
        result += np.asarray(
            expression(
                left_blocks,
                right_blocks,
                operators,
                *copy_factors,
            )
        ).reshape(size, size)
    return result


def test_virtual_identity_advance_matches_explicit_tensor():
    initial, _dense = _states(seed=97)
    state = type(initial)(
        initial.hamiltonian,
        initial.dims,
        initial.parent_sets,
        bond_dim=initial.bond_dim,
        tensors=initial.tensors,
        frontier_backend="identity_block",
    )
    right_messages = state._hamiltonian_frontier.build_right(state.tensors)

    for site in range(len(state.dims) - 1):
        following = site + 1
        plan = state._pair_plan(site)
        pair_tensors = list(state.tensors)
        pair_tensors[following] = plan.identity_tensor
        expected = plan.hamiltonian_engine.advance_right(
            right_messages[following + 1],
            pair_tensors,
            following,
            max_workers=2,
        )
        actual = plan.hamiltonian_engine.advance_right_identity(
            right_messages[following + 1],
            following,
            max_workers=2,
        )
        _assert_messages_close(actual, expected, atol=0.0)


def test_no_copy_hole_gemm_matches_complex_expression_and_output_order():
    rng = np.random.default_rng(91)
    dims = (2,) * 5
    physical_sites = (
        (0, 1, 2, 3, 4),
        (1, 2, 4),
        (2,),
        (3,),
        (4,),
    )
    virtual_bonds = (1, 2, 3, 2, 2, 1)
    tensor_shapes = tuple(
        (
            virtual_bonds[site],
            virtual_bonds[site + 1],
            *(2 for _physical_site in sites),
        )
        for site, sites in enumerate(physical_sites)
    )
    mpo_bonds = (1, 2, 2, 2, 2, 1)
    mpo_tensors = tuple(
        rng.normal(size=(left, right, 2, 2))
        + 1.0j * rng.normal(size=(left, right, 2, 2))
        for left, right in zip(mpo_bonds, mpo_bonds[1:])
    )
    tensors = [
        rng.normal(size=shape) + 1.0j * rng.normal(size=shape)
        for shape in tensor_shapes
    ]
    engine = BlockMPOFrontier(
        dims,
        physical_sites,
        tensor_shapes,
        mpo_tensors,
    )
    site = 1
    left = engine.build_left(tensors)[site]
    right = engine.build_right(tensors)[site + 1]
    plans = [
        engine._hole_matrix_gemm_plan(
            site,
            *transitions[0],
            batch_size=len(transitions),
        )
        for transitions in engine._hole_transition_groups[site]
    ]

    assert all(plan is not None for plan in plans)
    assert any(
        plan["output_axes"] != tuple(range(len(plan["output_axes"])))
        for plan in plans
    )
    expected = _expression_hole_matrix(engine, site, left, right)
    actual = engine.hole_matrix(site, left, right)
    np.testing.assert_allclose(actual, expected, rtol=5.0e-15, atol=1.0e-11)


def test_hole_gemm_falls_back_when_copy_selectors_are_required(monkeypatch):
    state, _dense = _states(seed=93)
    dense, block = _engines(state, _identity_mpo(state.dims))
    left = block.build_left(state.tensors)
    right = block.build_right(state.tensors)
    site = next(
        site
        for site in range(len(state.dims))
        if block.frontier_sites[site]
        and block._hole_transition_groups[site]
    )
    assert all(
        block._hole_matrix_gemm_plan(
            site,
            *transitions[0],
            batch_size=len(transitions),
        )
        is None
        for transitions in block._hole_transition_groups[site]
    )

    def forbidden_gemm(*_args, **_kwargs):
        raise AssertionError("the no-copy GEMM path must not run")

    monkeypatch.setattr(block, "_evaluate_hole_matrix_gemm", forbidden_gemm)
    dense_left = dense.build_left(state.tensors)
    dense_right = dense.build_right(state.tensors)
    np.testing.assert_allclose(
        block.hole_matrix(site, left[site], right[site + 1]),
        dense.hole_matrix(site, dense_left[site], dense_right[site + 1]),
        atol=5.0e-13,
    )


@pytest.mark.skipif(
    not copy_einsum_available(),
    reason="optional LETTA copy-einsum extension is unavailable",
)
def test_native_copy_hole_matches_python_for_complex_messages(monkeypatch):
    state, _dense = _states(seed=95)
    _dense_engine, engine = _engines(state, _identity_mpo(state.dims))
    rng = np.random.default_rng(96)
    tensors = [
        tensor + 1.0j * 1.0e-2 * rng.normal(size=tensor.shape)
        for tensor in state.tensors
    ]
    left = engine.build_left(tensors)
    right = engine.build_right(tensors)
    site = next(
        site
        for site in range(len(state.dims))
        if engine.frontier_sites[site]
        and engine._hole_transition_groups[site]
    )
    calls = 0
    original = engine._evaluate_hole_matrix_copy

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(engine, "_evaluate_hole_matrix_copy", counted)
    expected = engine.hole_matrix(
        site,
        left[site],
        right[site + 1],
        copy_backend="python",
    )
    actual = engine.hole_matrix(
        site,
        left[site],
        right[site + 1],
        copy_backend="native",
    )
    parallel = engine.hole_matrix(
        site,
        left[site],
        right[site + 1],
        max_workers=2,
        parallel_min_size=1,
        copy_backend="native",
    )

    assert calls > 0
    np.testing.assert_allclose(actual, expected, rtol=3.0e-15, atol=5.0e-13)
    np.testing.assert_array_equal(parallel, actual)


@pytest.mark.skipif(
    not copy_einsum_available(),
    reason="optional LETTA copy-einsum extension is unavailable",
)
def test_copy_hole_auto_gate_uses_exact_python_fallback(monkeypatch):
    state, _dense = _states(seed=98)
    _dense_engine, engine = _engines(state, _identity_mpo(state.dims))
    left = engine.build_left(state.tensors)
    right = engine.build_right(state.tensors)
    site = next(
        site
        for site in range(len(state.dims))
        if engine.frontier_sites[site]
        and engine._hole_transition_groups[site]
    )
    expected = engine.hole_matrix(
        site,
        left[site],
        right[site + 1],
        copy_backend="python",
    )

    monkeypatch.setattr(
        block_mpo_module,
        "_COPY_HOLE_AUTO_MAX_OPERATIONS",
        0,
    )

    def forbidden(*_args):
        raise AssertionError("the auto operation gate must use Python")

    monkeypatch.setattr(engine, "_evaluate_hole_matrix_copy", forbidden)
    actual = engine.hole_matrix(
        site,
        left[site],
        right[site + 1],
        copy_backend="auto",
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not copy_einsum_available(),
    reason="optional LETTA copy-einsum extension is unavailable",
)
def test_native_copy_advances_match_python_and_are_parallel_deterministic():
    state, _dense = _states(seed=99)
    _dense_engine, engine = _engines(state, _identity_mpo(state.dims))
    rng = np.random.default_rng(100)
    tensors = [
        tensor + 1.0j * 1.0e-2 * rng.normal(size=tensor.shape)
        for tensor in state.tensors
    ]
    left_messages = engine.build_left(tensors, copy_backend="python")
    right_messages = engine.build_right(tensors, copy_backend="python")

    for site in range(len(state.dims)):
        expected_left = engine.advance_left(
            left_messages[site],
            tensors,
            site,
            copy_backend="python",
        )
        native_left = engine.advance_left(
            left_messages[site],
            tensors,
            site,
            copy_backend="native",
        )
        parallel_left = engine.advance_left(
            left_messages[site],
            tensors,
            site,
            max_workers=2,
            copy_backend="native",
        )
        _assert_messages_close(native_left, expected_left, atol=8.0e-13)
        _assert_messages_equal(parallel_left, native_left)

        expected_right = engine.advance_right(
            right_messages[site + 1],
            tensors,
            site,
            copy_backend="python",
        )
        native_right = engine.advance_right(
            right_messages[site + 1],
            tensors,
            site,
            copy_backend="native",
        )
        parallel_right = engine.advance_right(
            right_messages[site + 1],
            tensors,
            site,
            max_workers=2,
            copy_backend="native",
        )
        _assert_messages_close(native_right, expected_right, atol=8.0e-13)
        _assert_messages_equal(parallel_right, native_right)


@pytest.mark.skipif(
    not copy_einsum_available(),
    reason="optional LETTA copy-einsum extension is unavailable",
)
def test_native_identity_advance_matches_python_and_is_deterministic():
    initial, _dense = _states(seed=101)
    state = type(initial)(
        initial.hamiltonian,
        initial.dims,
        initial.parent_sets,
        bond_dim=initial.bond_dim,
        tensors=initial.tensors,
        frontier_backend="identity_block",
    )
    right_messages = state._hamiltonian_frontier.build_right(
        state.tensors,
        copy_backend="python",
    )

    for site in range(len(state.dims) - 1):
        following = site + 1
        engine = state._pair_plan(site).hamiltonian_engine
        expected = engine.advance_right_identity(
            right_messages[following + 1],
            following,
            copy_backend="python",
        )
        native = engine.advance_right_identity(
            right_messages[following + 1],
            following,
            copy_backend="native",
        )
        parallel = engine.advance_right_identity(
            right_messages[following + 1],
            following,
            max_workers=2,
            copy_backend="native",
        )
        _assert_messages_close(native, expected, atol=8.0e-13)
        _assert_messages_equal(parallel, native)


@pytest.mark.skipif(
    not copy_einsum_available(),
    reason="optional LETTA copy-einsum extension is unavailable",
)
def test_copy_advance_auto_gate_uses_python_fallback(monkeypatch):
    state, _dense = _states(seed=102)
    _dense_engine, engine = _engines(state, _identity_mpo(state.dims))
    left = engine.build_left(state.tensors, copy_backend="python")
    site = 1
    expected = engine.advance_left(
        left[site],
        state.tensors,
        site,
        copy_backend="python",
    )
    monkeypatch.setattr(
        block_mpo_module,
        "_COPY_ADVANCE_AUTO_MAX_OPERATIONS",
        0,
    )

    def forbidden(*_args):
        raise AssertionError("the auto operation gate must use Python")

    monkeypatch.setattr(engine, "_evaluate_advance_copy", forbidden)
    actual = engine.advance_left(
        left[site],
        state.tensors,
        site,
        copy_backend="auto",
    )
    _assert_messages_equal(actual, expected)


@pytest.mark.skipif(
    not copy_einsum_available(),
    reason="optional LETTA copy-einsum extension is unavailable",
)
def test_copy_advance_auto_keeps_no_copy_groups_on_python(monkeypatch):
    rng = np.random.default_rng(103)
    dims = (2, 2, 2)
    physical_sites = ((0,), (1,), (2,))
    tensor_shapes = ((1, 2, 2), (2, 2, 2), (2, 1, 2))
    tensors = [rng.normal(size=shape) for shape in tensor_shapes]
    engine = BlockMPOFrontier(
        dims,
        physical_sites,
        tensor_shapes,
        _identity_mpo(dims).tensors,
    )
    message = engine.left_boundary()
    assert all(
        engine._advance_copy_plan(
            "left",
            0,
            *transitions[0],
            batch_size=len(transitions),
        )["copy_count"]
        == 0
        for transitions in engine._left_transition_groups[0]
    )
    expected = engine.advance_left(
        message,
        tensors,
        0,
        copy_backend="python",
    )

    def forbidden(*_args):
        raise AssertionError("auto must keep no-copy groups on opt_einsum")

    monkeypatch.setattr(engine, "_evaluate_advance_copy", forbidden)
    actual = engine.advance_left(
        message,
        tensors,
        0,
        copy_backend="auto",
    )
    _assert_messages_equal(actual, expected)


def test_exact_local_tt_round_trip_and_block_messages():
    state, _dense = _states(seed=39)
    tensor = state.tensors[0]
    cores = _exact_local_tt(tensor)
    reconstructed = cores[0][0]
    for core in cores[1:]:
        reconstructed = np.tensordot(reconstructed, core, axes=(-1, 0))
    reconstructed = reconstructed[..., 0]
    inverse = (0, len(state.physical_sites[0]) + 1, *range(1, len(state.physical_sites[0]) + 1))
    np.testing.assert_allclose(reconstructed.transpose(inverse), tensor, atol=3.0e-14)

    mpo = state.hamiltonian.to_mpo()
    arguments = (
        state.dims,
        state.physical_sites,
        [value.shape for value in state.tensors],
        mpo.tensors,
    )
    dense = BlockMPOFrontier(*arguments, local_backend="dense")
    sequential = BlockMPOFrontier(*arguments, local_backend="tensor_train")
    dense_left = dense.build_left(state.tensors)
    sequential_left = sequential.build_left(state.tensors)
    dense_right = dense.build_right(state.tensors)
    sequential_right = sequential.build_right(state.tensors)
    for reference, actual in zip(dense_left, sequential_left):
        _assert_messages_close(actual, reference, atol=8.0e-13)
    for reference, actual in zip(dense_right, sequential_right):
        _assert_messages_close(actual, reference, atol=8.0e-13)


def test_truncated_local_tt_caps_ranks_and_reports_error():
    rng = np.random.default_rng(71)
    tensor = rng.normal(size=(3, 4, 2, 2, 2))
    cores, diagnostics = _truncated_local_tt(tensor, max_rank=2)
    reconstructed = _local_tt_tensor(cores, 3)

    assert max(diagnostics["ranks"]) <= 2
    np.testing.assert_allclose(
        np.linalg.norm(reconstructed - tensor),
        diagnostics["discarded_norm"],
        atol=2.0e-14,
    )
    assert diagnostics["relative_discarded_norm"] > 0.0


def test_truncated_local_tt_preserves_charge_support():
    rng = np.random.default_rng(73)
    left_qns = ((0,), (1,))
    local_qns = ((1,), (-1,))
    right_qns = ((-1,), (0,), (1,), (2,))
    tensor = np.zeros((2, 4, 2, 2))
    for left, q_left in enumerate(left_qns):
        for right, q_right in enumerate(right_qns):
            for physical, q_physical in enumerate(local_qns):
                if q_left[0] + q_physical[0] == q_right[0]:
                    tensor[left, right, physical] = rng.normal(size=2)
    cores, diagnostics = _truncated_local_tt(
        tensor,
        max_rank=1,
        axis_qns=(
            left_qns,
            local_qns,
            ((0,), (0,)),
            tuple((-charge[0],) for charge in right_qns),
        ),
    )
    reconstructed = _local_tt_tensor(cores, 2)

    assert diagnostics["charge_resolved"]
    for left, q_left in enumerate(left_qns):
        for right, q_right in enumerate(right_qns):
            for physical, q_physical in enumerate(local_qns):
                if q_left[0] + q_physical[0] != q_right[0]:
                    np.testing.assert_array_equal(
                        reconstructed[left, right, physical],
                        0.0,
                    )


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
        parallel_left = block_engine.build_left(
            state.tensors,
            max_workers=2,
        )
        parallel_right = block_engine.build_right(
            state.tensors,
            max_workers=2,
        )
        for actual, expected in zip(parallel_left, block_left):
            _assert_messages_close(actual, expected, atol=0.0)
        for actual, expected in zip(parallel_right, block_right):
            _assert_messages_close(actual, expected, atol=0.0)
        for site, tensor in enumerate(state.tensors):
            dense_matrix = dense_engine.hole_matrix(
                site, dense_left[site], dense_right[site + 1]
            )
            block_matrix = block_engine.hole_matrix(
                site, block_left[site], block_right[site + 1]
            )
            parallel_matrix = block_engine.hole_matrix(
                site,
                block_left[site],
                block_right[site + 1],
                max_workers=2,
                parallel_min_size=1,
            )
            np.testing.assert_allclose(block_matrix, dense_matrix, atol=5.0e-13)
            np.testing.assert_array_equal(parallel_matrix, block_matrix)

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
    assert engine.plan_count <= grouped_plans
    assert grouped_plans < ungrouped_plans


def test_hole_blocks_batch_physical_requests(monkeypatch):
    state, _dense = _states(seed=49)
    rng = np.random.default_rng(50)
    mpo_bonds = (1, 4, 4, 4, 1)
    identity = np.eye(2)
    mpo_tensors = tuple(
        rng.uniform(0.2, 1.0, size=(left, right))[:, :, None, None]
        * identity
        for left, right in zip(mpo_bonds, mpo_bonds[1:])
    )
    engine = BlockMPOFrontier(
        state.dims,
        state.physical_sites,
        [tensor.shape for tensor in state.tensors],
        mpo_tensors,
    )
    site = 1
    left = engine.build_left(state.tensors)[site]
    right = engine.build_right(state.tensors)[site + 1]
    configurations = tuple(
        np.ndindex(
            *(state.dims[index] for index in state.physical_sites[site])
        )
    )
    requests = tuple(
        (index, index + 1, bra, ket)
        for index, (bra, ket) in enumerate(
            zip(configurations, reversed(configurations))
        )
    )
    maximum_transition_batch = max(
        len(group) for group in engine._hole_transition_groups[site]
    )
    assert maximum_transition_batch > 2
    uncapped = engine.hole_blocks(
        site,
        left,
        right,
        requests,
        request_batch_size=3,
        transition_batch_size=maximum_transition_batch,
    )

    expression_calls = 0
    original = engine._hole_expression

    def counted_expression(mode, *args, **kwargs):
        nonlocal expression_calls
        if mode == "physical_blocks":
            expression_calls += 1
        return original(mode, *args, **kwargs)

    monkeypatch.setattr(engine, "_hole_expression", counted_expression)
    batch_size = 3
    actual = engine.hole_blocks(
        site,
        left,
        right,
        requests,
        request_batch_size=batch_size,
    )
    transition_chunks = sum(
        (len(group) + 1) // 2
        for group in engine._hole_transition_groups[site]
    )
    expected_calls = transition_chunks * (
        (len(requests) + batch_size - 1) // batch_size
    )

    assert expression_calls == expected_calls
    for row, column, bra, ket in requests:
        expected = engine.hole_block(site, left, right, bra, ket)
        np.testing.assert_allclose(actual[(row, column)], expected, atol=5.0e-13)
        np.testing.assert_allclose(
            actual[(row, column)],
            uncapped[(row, column)],
            atol=5.0e-13,
        )


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
    assert engine.total_message_elements == 189_826
    assert any(
        len(engine._active_channels[cut]) < engine.mpo_bonds[cut]
        for cut in range(nsites + 1)
    )
