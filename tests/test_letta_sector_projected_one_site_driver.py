from types import SimpleNamespace

import numpy as np
import pytest

from examples.mps.converge_sector_projected_letta_6x6 import (
    NSITES,
    _fingerprint,
    _result_record,
    _source_projection_metadata,
    _validate_resume,
)


def _state():
    return SimpleNamespace(
        nparameters=4008,
        dense_nparameters=4008,
        bond_dims=(1, 4, 1),
        tensors=(object(),) * NSITES,
    )


def _pass(gain, *, failures=0):
    return {
        "energy_gain": float(gain),
        "solver_failures": int(failures),
    }


def test_one_site_fingerprint_allows_only_a_larger_pass_cap():
    protocol = {
        "maximum_directional_passes": 20,
        "gain_tolerance": 1.0e-6,
        "sweep_offset": 40,
    }
    raised_cap = {**protocol, "maximum_directional_passes": 80}
    tighter = {**protocol, "gain_tolerance": 1.0e-7}
    shifted = {**protocol, "sweep_offset": 41}

    assert _fingerprint(protocol) == _fingerprint(raised_cap)
    assert _fingerprint(protocol) != _fingerprint(tighter)
    assert _fingerprint(protocol) != _fingerprint(shifted)


def test_one_site_convergence_requires_two_clean_directional_passes(tmp_path):
    options = {
        "state": _state(),
        "energy": -17.0,
        "initial_projected_energy": -16.9,
        "raw_source_energy": -16.8,
        "mps_d32_energy": None,
        "snapshot": tmp_path / "state.npz",
        "checkpoint_id": "checkpoint",
        "source_is_projected": True,
        "gain_tolerance": 1.0e-6,
        "maximum_passes": 20,
        "sweep_offset": 40,
    }

    one_pass = _result_record(passes=[_pass(2.0e-5)], **options)
    assert not one_pass["converged"]

    converged = _result_record(
        passes=[_pass(2.0e-5), _pass(3.0e-5)],
        **options,
    )
    assert converged["converged"]
    assert converged["next_directional_sweep"] == 42
    assert converged["last_cycle_maximum_gain"] == pytest.approx(3.0e-5)
    assert converged["last_cycle_maximum_gain_per_site"] == pytest.approx(
        3.0e-5 / NSITES
    )
    assert converged["last_directional_gain_per_site"] == pytest.approx(
        3.0e-5 / NSITES
    )

    failed = _result_record(
        passes=[_pass(2.0e-5), _pass(3.0e-5, failures=1)],
        **options,
    )
    assert not failed["converged"]

    energy_increase = _result_record(
        passes=[_pass(2.0e-5), _pass(-4.0e-5)],
        **options,
    )
    assert not energy_increase["converged"]

    above_per_site_threshold = _result_record(
        passes=[_pass(2.0e-5), _pass(4.0e-5)],
        **options,
    )
    assert not above_per_site_threshold["converged"]


def test_unrestricted_source_with_recorded_energy_is_not_projected(tmp_path):
    unrestricted = tmp_path / "unrestricted.npz"
    np.savez(unrestricted, recorded_energy=np.asarray(-1.25))

    assert _source_projection_metadata(unrestricted) is None

    incomplete = tmp_path / "incomplete_projected.npz"
    np.savez(
        incomplete,
        recorded_energy=np.asarray(-1.25),
        target_two_sz=np.asarray(0),
    )
    with pytest.raises(RuntimeError, match="missing metadata"):
        _source_projection_metadata(incomplete)

    projected = tmp_path / "projected.npz"
    np.savez(
        projected,
        recorded_energy=np.asarray(-1.25),
        target_two_sz=np.asarray(0),
        local_two_sz=np.asarray(((1, -1),) * NSITES),
    )
    assert _source_projection_metadata(projected) == {"recorded_energy": -1.25}


def test_one_site_resume_accepts_only_a_non_decreasing_cap():
    fingerprint = "protocol"
    payload = {
        "protocol_fingerprint": fingerprint,
        "protocol": {"checkpoint_pair_energy_tolerance": 1.0e-8},
        "directional_passes": [{}, {}],
        "result": {
            "checkpoint_id": "checkpoint",
            "energy": -2.0,
        },
    }
    metadata = {
        "protocol_fingerprint": fingerprint,
        "checkpoint_id": "checkpoint",
        "completed_passes": 2,
        "tensor_count": NSITES,
        "target_two_sz": 0,
        "recorded_energy": -2.0,
        "local_two_sz": np.asarray(((1, -1),) * NSITES),
    }

    _validate_resume(
        payload,
        metadata,
        protocol_fingerprint=fingerprint,
        maximum_passes=4,
    )
    with pytest.raises(ValueError, match="below the 2 already completed"):
        _validate_resume(
            payload,
            metadata,
            protocol_fingerprint=fingerprint,
            maximum_passes=1,
        )
