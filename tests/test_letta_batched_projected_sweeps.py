import numpy as np
import pytest
from types import SimpleNamespace

from examples.mps.converge_sector_projected_letta_two_site_batched_6x6 import (
    _advance_directional_messages,
    _complete_directional_messages,
    _result_record,
    _start_directional_messages,
    _target_model,
)
from pyqed.letta import (
    FrontierTiedLETTA,
    LocalHamiltonian,
    LocalTerm,
    SectorProjectedLETTA,
)


def _projected_chain():
    dims = (2,) * 4
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    exchange = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    hamiltonian = LocalHamiltonian(
        dims,
        tuple(LocalTerm((site, site + 1), exchange) for site in range(len(dims) - 1)),
    )
    unrestricted = FrontierTiedLETTA(
        hamiltonian,
        dims,
        ((1,), (2,), (3,), ()),
        bond_dim=2,
        seed=19,
        frontier_backend="identity_block",
    )
    return SectorProjectedLETTA.from_unrestricted(
        unrestricted,
        local_charges=((1, -1),) * len(dims),
        target=0,
        frontier_backend="identity_block",
    )


def _assert_messages_close(actual, expected):
    for left, right in zip(actual["norm"], expected[0]):
        np.testing.assert_allclose(left, right, atol=2.0e-12)
    for left, right in zip(actual["hamiltonian"], expected[1]):
        for left_block, right_block in zip(left.blocks, right.blocks):
            np.testing.assert_allclose(left_block, right_block, atol=2.0e-12)


def _modify_pair(state, site, step):
    state.tensors[site] = state.tensors[site] + step
    state.tensors[site + 1] = state.tensors[site + 1] - 0.3 * step


def test_target_model_rebuilds_j2_without_mutating_source():
    source = {"j1": 1.0, "j2": 0.5, "nrows": 6, "ncols": 6}
    model, source_j2, target_j2 = _target_model(source, 0.65)

    assert source["j2"] == 0.5
    assert model["j2"] == 0.65
    assert source_j2 == 0.5
    assert target_j2 == 0.65

    inherited, source_j2, target_j2 = _target_model(source, None)
    assert inherited == source
    assert source_j2 == target_j2 == 0.5

    for invalid in (-0.1, np.inf, np.nan):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            _target_model(source, invalid)


def test_moving_messages_become_exact_reverse_fixed_cache():
    state = _projected_chain()
    messages = _start_directional_messages(
        state,
        0,
        None,
        frontier_workers=1,
        frontier_executor=None,
    )
    for site in range(len(state.dims) - 1):
        _modify_pair(state, site, 1.0e-3 * (site + 1))
        _advance_directional_messages(
            state,
            0,
            site,
            messages,
            frontier_workers=1,
            frontier_executor=None,
        )
    _energy, _norm, left_cache = _complete_directional_messages(
        state,
        0,
        messages,
        frontier_workers=1,
        frontier_executor=None,
    )
    _assert_messages_close(
        left_cache,
        (
            state._norm_frontier.build_left(state.tensors),
            state._hamiltonian_frontier.build_left(state.tensors),
        ),
    )

    reverse = _start_directional_messages(
        state,
        1,
        left_cache,
        frontier_workers=1,
        frontier_executor=None,
    )
    assert reverse["fixed_reused"]
    for site in range(len(state.dims) - 2, -1, -1):
        _modify_pair(state, site, -7.0e-4 * (site + 1))
        _advance_directional_messages(
            state,
            1,
            site,
            reverse,
            frontier_workers=1,
            frontier_executor=None,
        )
    _energy, _norm, right_cache = _complete_directional_messages(
        state,
        1,
        reverse,
        frontier_workers=1,
        frontier_executor=None,
    )
    _assert_messages_close(
        right_cache,
        (
            state._norm_frontier.build_right(state.tensors),
            state._hamiltonian_frontier.build_right(state.tensors),
        ),
    )

    state.tensors[0] = 1.01 * state.tensors[0]
    with pytest.raises(ValueError, match="stale"):
        _start_directional_messages(
            state,
            0,
            right_cache,
            frontier_workers=1,
            frontier_executor=None,
        )


def test_bound_pair_right_environment_matches_direct_binding():
    state = _projected_chain()
    site = 1
    norm_left = state._norm_frontier.build_left(state.tensors)
    norm_right = state._norm_frontier.build_right(state.tensors)
    hamiltonian_left = state._hamiltonian_frontier.build_left(
        state.tensors
    )
    hamiltonian_right = state._hamiltonian_frontier.build_right(
        state.tensors
    )
    direct = state._pair_environment_from_outer_messages(
        site,
        norm_left[site],
        norm_right[site + 2],
        hamiltonian_left[site],
        hamiltonian_right[site + 2],
    )
    bound_right = state._bind_pair_right_environment(
        site,
        norm_right[site + 2],
        hamiltonian_right[site + 2],
    )
    rebound = state._pair_environment_from_bound_right(
        site,
        norm_left[site],
        hamiltonian_left[site],
        bound_right,
    )

    np.testing.assert_array_equal(rebound.norm_left, direct.norm_left)
    np.testing.assert_array_equal(rebound.norm_right, direct.norm_right)
    for actual, expected in (
        (rebound.hamiltonian_left, direct.hamiltonian_left),
        (rebound.hamiltonian_right, direct.hamiltonian_right),
    ):
        assert actual.cut == expected.cut
        for actual_block, expected_block in zip(
            actual.blocks,
            expected.blocks,
        ):
            np.testing.assert_array_equal(actual_block, expected_block)


def test_convergence_uses_both_directional_gains_per_site(tmp_path):
    state = SimpleNamespace(
        energy=-1.0,
        nparameters=4,
        dense_nparameters=4,
        bond_dims=(1, 1),
    )
    rows = [
        {
            "accepted": True,
            "energy_gain": 2.0e-5,
            "fixed_messages_reused": False,
        },
        {
            "accepted": True,
            "energy_gain": 4.0e-5,
            "fixed_messages_reused": True,
        },
    ]
    result = _result_record(
        state,
        source_energy=-0.9,
        directional_passes=rows,
        next_directional_sweep=2,
        gain_tolerance=1.0e-6,
        maximum_directional_passes=2,
        snapshot=tmp_path / "state.npz",
    )
    assert not result["converged"]
    assert result["last_cycle_maximum_gain"] == pytest.approx(4.0e-5)
    assert result["last_cycle_maximum_gain_per_site"] == pytest.approx(
        4.0e-5 / 36
    )
    assert result["last_directional_gain_per_site"] == pytest.approx(
        4.0e-5 / 36
    )
    assert result["energy_gain_from_source_per_site"] == pytest.approx(
        0.1 / 36
    )

    rows[1]["energy_gain"] = 3.0e-5
    result = _result_record(
        state,
        source_energy=-0.9,
        directional_passes=rows,
        next_directional_sweep=2,
        gain_tolerance=1.0e-6,
        maximum_directional_passes=2,
        snapshot=tmp_path / "state.npz",
    )
    assert result["converged"]

    rows[1]["energy_gain"] = -4.0e-5
    result = _result_record(
        state,
        source_energy=-0.9,
        directional_passes=rows,
        next_directional_sweep=2,
        gain_tolerance=1.0e-6,
        maximum_directional_passes=2,
        snapshot=tmp_path / "state.npz",
    )
    assert not result["converged"]
