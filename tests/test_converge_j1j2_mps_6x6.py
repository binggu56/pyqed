import json

import numpy as np
import pytest

from examples.mps.converge_j1j2_mps_6x6 import (
    _below_gain_tolerance,
    _monotonic_direction_record,
    converge,
)


def test_monotonic_direction_record_rejects_an_energy_increase():
    record = _monotonic_direction_record(
        -1.0,
        (("lr", -1.5), ("rl", -0.75)),
        nsites=4,
    )
    assert record["best_energy"] == pytest.approx(-1.5)
    assert record["directional_energy_gains"] == pytest.approx([0.5, 0.0])
    assert record["maximum_accepted_directional_gain_per_site"] == pytest.approx(
        0.125
    )
    assert record["rejected_energy_increases"] == 1
    assert record["maximum_absolute_directional_change_per_site"] == pytest.approx(
        0.1875
    )
    assert not _below_gain_tolerance(
        record,
        {"reported_solver_failures": 0},
        gain_tolerance=0.1,
    )


def test_rejected_endpoint_increase_does_not_block_variational_convergence():
    record = _monotonic_direction_record(
        -1.0,
        (("lr", -1.000001), ("rl", -0.5)),
        nsites=4,
    )
    assert record["maximum_absolute_directional_change_per_site"] > 0.1
    assert _below_gain_tolerance(
        record,
        {"reported_solver_failures": 0},
        gain_tolerance=1.0e-6,
    )


def test_saturated_2x2_mps_keeps_the_exact_ground_state_and_resumes(tmp_path):
    output = tmp_path / "result.json"
    snapshot = tmp_path / "state.npz"
    first = converge(
        nrows=2,
        ncols=2,
        j2=0.5,
        bond_dims=(4,),
        maximum_directional_passes=2,
        gain_tolerance=1.0e-10,
        required_consecutive_cycles=2,
        performance="auto",
        output=output,
        snapshot=snapshot,
        resume=False,
        verbose=False,
    )
    assert not first["result"]["converged"]
    assert first["status"] == "maximum_passes"
    assert first["result"]["energy"] == pytest.approx(-1.75, abs=1.0e-10)
    assert [
        cycle["energy"] for cycle in first["stages"][0]["cycles"]
    ] == sorted(
        (cycle["energy"] for cycle in first["stages"][0]["cycles"]),
        reverse=True,
    )
    checkpoint_id = first["result"]["checkpoint_id"]
    with np.load(snapshot, allow_pickle=False) as archive:
        assert str(archive["checkpoint_id"].item()) == checkpoint_id
        assert float(archive["recorded_energy"].item()) == pytest.approx(
            -1.75,
            abs=5.0e-10,
        )

    resumed = converge(
        nrows=2,
        ncols=2,
        j2=0.5,
        bond_dims=(4,),
        maximum_directional_passes=6,
        gain_tolerance=1.0e-10,
        required_consecutive_cycles=2,
        performance="auto",
        output=output,
        snapshot=snapshot,
        resume=True,
        verbose=False,
    )
    assert resumed["result"]["converged"]
    assert resumed["result"]["checkpoint_id"] != checkpoint_id
    assert json.loads(output.read_text())["result"]["energy"] == pytest.approx(
        -1.75
    )
