import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "letta_hpc.py"
SPEC = importlib.util.spec_from_file_location("pyqed_test_letta_hpc", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
letta_hpc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(letta_hpc)


def _raw_config():
    return {
        "cluster": {
            "host": "cluster.example",
            "port": 10022,
            "remote_source_root": "/cluster/sources/current",
            "remote_sources_root": "/cluster/sources",
            "remote_runs_root": "/cluster/runs",
            "python": "/cluster/env/bin/python",
            "partition": "compute",
            "qos": "normal",
            "cpus_per_task": 16,
            "memory": "16G",
            "time_limit": "04:00:00",
            "concurrency": 2,
        },
        "scan": {
            "name": "j1j2-test",
            "values": [0.0, 0.5, 1.0],
            "gain_tolerance": 1.0e-6,
            "gain_tolerance_units": "energy_per_site",
            "maximum_directional_passes": 200,
            "pair_workers": 8,
            "frontier_workers": 4,
            "source_run_root": "/cluster/parent",
            "worker": "examples/mps/cluster/run_task.sh",
        },
        "local": {
            "source_root": ".",
            "runs_root": ".letta_hpc/tests",
            "source_paths": ["pyqed", "setup.py"],
        },
    }


def _normalized(raw=None):
    return letta_hpc.normalize_config(
        _raw_config() if raw is None else raw,
        config_path=ROOT / "letta-hpc-test.json",
    )


@pytest.mark.parametrize(
    "values",
    (
        [0.1, "0.10"],
        [0.0, float("nan")],
        [0.0, float("inf")],
        [0.0, -0.1],
    ),
)
def test_config_rejects_duplicate_nonfinite_and_negative_scan_values(values):
    raw = _raw_config()
    raw["scan"]["values"] = values
    with pytest.raises(letta_hpc.WorkflowError):
        _normalized(raw)


@pytest.mark.parametrize("source_path", ("../secret", "/tmp/secret"))
def test_config_rejects_source_paths_outside_source_root(source_path):
    raw = _raw_config()
    raw["local"]["source_paths"] = ["pyqed", source_path]
    with pytest.raises(
        letta_hpc.WorkflowError,
        match="stay inside source_root",
    ):
        _normalized(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("remote_runs_root", "/cluster/runs;touch_bad"),
        ("remote_source_root", "/cluster/$HOME/source"),
        ("partition", "compute\n/bin/false"),
        ("qos", "normal;bad"),
        ("memory", "16G\n/bin/false"),
        ("time_limit", "04:00:00\n/bin/false"),
    ),
)
def test_config_rejects_shell_interpretable_remote_and_sbatch_values(key, value):
    raw = _raw_config()
    raw["cluster"][key] = value
    with pytest.raises(letta_hpc.WorkflowError):
        _normalized(raw)


@pytest.mark.parametrize(
    "run_id",
    (
        "",
        "../escape",
        "/absolute",
        "has space",
        "shell;command",
        "x" * 129,
    ),
)
def test_run_id_rejects_unsafe_names(run_id):
    with pytest.raises(letta_hpc.WorkflowError):
        letta_hpc._run_id(run_id)


@pytest.mark.parametrize(
    ("indices", "expected"),
    (
        ([0], "0"),
        ([0, 1, 2, 3], "0-3"),
        ([0, 2, 3, 7, 9, 10, 11, 12], "0,2-3,7,9-12"),
        ([3, 2, 2, 1], "1-3"),
    ),
)
def test_array_index_compression(indices, expected):
    assert letta_hpc._compress_indices(indices) == expected


@pytest.mark.parametrize("indices", ([], [-1], [-1, 0]))
def test_array_index_compression_rejects_empty_or_negative(indices):
    with pytest.raises(letta_hpc.WorkflowError):
        letta_hpc._compress_indices(indices)


def test_sbatch_rendering_contains_array_source_and_numerical_settings():
    text = letta_hpc.render_sbatch(
        _normalized(),
        remote_run_dir="/cluster/runs/run-001",
        source_run_root="/cluster/runs/parent/frozen",
        indices=[0, 2],
        maximum_directional_passes=400,
    )

    assert "#SBATCH --array=0,2%2" in text
    assert "export PYQED_ROOT=/cluster/sources/current" in text
    assert "export LETTA_SOURCE_RUN_ROOT=/cluster/runs/parent/frozen" in text
    tolerance_line = next(
        line for line in text.splitlines() if "LETTA_GAIN_TOLERANCE=" in line
    )
    assert float(tolerance_line.split("=", 1)[1]) == pytest.approx(1.0e-6)
    assert "export LETTA_GAIN_TOLERANCE_UNITS=energy_per_site" in text
    assert "export LETTA_MAXIMUM_DIRECTIONAL_PASSES=400" in text
    assert "export LETTA_PAIR_WORKERS=8" in text
    assert "export LETTA_FRONTIER_WORKERS=4" in text
    assert "LETTA_SEED_RESULT" not in text
    assert "LETTA_SEED_SNAPSHOT" not in text
    assert '\"$SLURM_ARRAY_TASK_ID\"' in text


def test_source_free_scan_omits_checkpoint_environment():
    raw = _raw_config()
    raw["scan"].pop("source_run_root")
    raw["scan"]["source_free"] = True
    spec = _normalized(raw)

    text = letta_hpc.render_sbatch(
        spec,
        remote_run_dir="/cluster/runs/mps-001",
        source_run_root=None,
    )

    assert spec["scan"]["source_free"] is True
    assert "LETTA_SOURCE_RUN_ROOT" not in text
    assert "LETTA_SEED_RESULT" not in text
    assert "LETTA_SEED_SNAPSHOT" not in text
    assert "examples/mps/cluster/run_task.sh" in text


def test_source_free_scan_rejects_checkpoint_sources():
    raw = _raw_config()
    raw["scan"]["source_free"] = True
    with pytest.raises(letta_hpc.WorkflowError, match="cannot be combined"):
        _normalized(raw)


def test_mps_6x6_cluster_config_is_a_source_free_13_point_scan():
    spec, _ = letta_hpc.load_config(
        ROOT / "examples/mps/cluster/letta_hpc_mps_6x6.toml"
    )
    text = letta_hpc.render_sbatch(
        spec,
        remote_run_dir="/cluster/runs/mps-6x6",
        source_run_root=None,
    )

    assert spec["scan"]["source_free"] is True
    assert spec["scan"]["values"] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    assert "#SBATCH --array=0-12%13" in text
    assert "#SBATCH --account=gubing" in text
    assert "#SBATCH --nodelist=xnode[102-103]" in text
    assert "run_mps_j1j2_task_6x6.sh" in text


@pytest.mark.parametrize(
    ("output", "expected"),
    (
        ("12345\n", "12345"),
        ("12345;cluster-a\n", "12345"),
        ("diagnostic\n12345;cluster-a\n", "12345"),
    ),
)
def test_sbatch_job_id_parsing(output, expected):
    assert letta_hpc._parse_job_id(output) == expected


@pytest.mark.parametrize("output", ("", "Submitted batch job 12345", "12x"))
def test_sbatch_job_id_parsing_rejects_nonparsable_output(output):
    with pytest.raises(letta_hpc.WorkflowError):
        letta_hpc._parse_job_id(output)


def test_remote_submission_journal_is_idempotent(tmp_path):
    sbatch = tmp_path / "sbatch"
    count = tmp_path / "invocations.txt"
    sbatch.write_text(
        "#!/bin/sh\n"
        f"printf 'called\\n' >> {count}\n"
        "printf '12345\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    (tmp_path / "submit.sbatch").write_text("#!/bin/sh\n", encoding="utf-8")
    request = "a" * 64
    record = json.dumps(
        {
            "kind": "resume",
            "submitted_at": "2026-01-01T00:00:00+00:00",
            "indices": [0, 2],
            "maximum_directional_passes": 400,
        }
    )
    command = [
        sys.executable,
        "-c",
        letta_hpc._REMOTE_SUBMIT_CODE,
        str(tmp_path),
        str(sbatch),
        "submit.sbatch",
        request,
        record,
    ]

    first = subprocess.run(
        command, check=True, text=True, capture_output=True
    )
    second = subprocess.run(
        command, check=True, text=True, capture_output=True
    )

    assert json.loads(first.stdout)["job_id"] == "12345"
    assert json.loads(second.stdout)["job_id"] == "12345"
    assert count.read_text(encoding="utf-8").splitlines() == ["called"]


def test_remote_submission_journal_never_retries_ambiguous_success(tmp_path):
    sbatch = tmp_path / "sbatch"
    count = tmp_path / "invocations.txt"
    sbatch.write_text(
        "#!/bin/sh\n"
        f"printf 'called\\n' >> {count}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    (tmp_path / "submit.sbatch").write_text("#!/bin/sh\n", encoding="utf-8")
    request = "b" * 64
    command = [
        sys.executable,
        "-c",
        letta_hpc._REMOTE_SUBMIT_CODE,
        str(tmp_path),
        str(sbatch),
        "submit.sbatch",
        request,
        json.dumps({"kind": "resume"}),
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode != 0
    assert second.returncode != 0
    assert "refusing a possibly duplicate" in second.stderr
    assert count.read_text(encoding="utf-8").splitlines() == ["called"]
    journal = json.loads(
        (
            tmp_path / ".submissions" / f"{request}.json"
        ).read_text(encoding="utf-8")
    )
    assert journal["state"] == "ambiguous"


def test_sacct_parses_array_rows_and_normalizes_cancelled_reason():
    output = "\n".join(
        (
            "991_0|COMPLETED|00:10:00|0:0|2800M",
            "991_1|CANCELLED by 12345|00:01:02|0:15|40M",
            "991_2|OUT_OF_MEMORY+|00:03:00|0:9|16000M",
            "991_2.batch|FAILED|00:03:00|1:0|16000M",
            "unrelated_0|RUNNING|00:00:03|0:0|1M",
        )
    )

    rows = letta_hpc.parse_sacct(output, job_id="991")

    assert rows[0]["state"] == "COMPLETED"
    assert rows[1]["state"] == "CANCELLED"
    assert rows[1]["exit_code"] == "0:15"
    assert rows[2]["state"] == "OUT_OF_MEMORY"
    assert set(rows) == {0, 1, 2}


def test_squeue_parses_expanded_live_array_rows():
    output = "\n".join(
        (
            "991_1|RUNNING|00:01:02|node17",
            "991_2|PENDING|00:00:00|Resources",
            "other_0|RUNNING|00:00:01|node18",
        )
    )

    rows = letta_hpc.parse_squeue(output, job_id="991")

    assert rows[1]["state"] == "RUNNING"
    assert rows[1]["reason"] == "node17"
    assert rows[2]["state"] == "PENDING"
    assert set(rows) == {1, 2}


def _result(index, *, status, converged=False, valid=True):
    return {
        "task_index": index,
        "has_result": True,
        "has_snapshot": True,
        "checkpoint_pair_valid": valid,
        "status": status,
        "converged": converged,
        "energy": -1.0 - index,
        "passes": 10,
    }


def test_status_aggregation_distinguishes_scheduler_and_solver_states():
    metadata = {"config": {"scan": {"values": [0.1 * i for i in range(9)]}}}
    results = [
        _result(0, status="converged", converged=True),
        _result(2, status="maximum_passes"),
        _result(3, status="endpoint_rejected"),
        {
            "task_index": 4,
            "has_result": True,
            "has_snapshot": False,
            "status": "running",
        },
        _result(8, status="running"),
    ]
    slurm = {
        0: {"state": "COMPLETED"},
        1: {"state": "RUNNING"},
        2: {"state": "COMPLETED"},
        3: {"state": "COMPLETED"},
        4: {"state": "COMPLETED"},
        5: {"state": "TIMEOUT"},
        6: {"state": "COMPLETED"},
        7: {"state": "PENDING"},
        8: {"state": "CANCELLED"},
    }

    rows = letta_hpc.aggregate_status(
        metadata,
        results=results,
        slurm=slurm,
    )

    assert [row["workflow_state"] for row in rows] == [
        "converged",
        "running",
        "max-passes",
        "endpoint-rejected",
        "unsafe-checkpoint",
        "timeout",
        "incomplete",
        "pending",
        "cancelled",
    ]


def test_resume_selection_skips_active_converged_capped_and_rejected_tasks():
    rows = [
        {"task_index": 0, "slurm_state": "COMPLETED", "passes": 12},
        {"task_index": 1, "slurm_state": "RUNNING", "passes": 8},
        {
            "task_index": 2,
            "slurm_state": "COMPLETED",
            "passes": 6,
            "converged": True,
        },
        {"task_index": 3, "slurm_state": "COMPLETED", "passes": 40},
        {
            "task_index": 4,
            "slurm_state": "COMPLETED",
            "status": "endpoint_rejected",
            "passes": 9,
        },
    ]

    selected, capped, rejected = letta_hpc.select_resume_indices(
        rows,
        maximum_directional_passes=40,
    )
    assert selected == [0]
    assert capped == [3]
    assert rejected == [4]

    selected, capped, rejected = letta_hpc.select_resume_indices(
        rows,
        maximum_directional_passes=41,
        requested=[3, 4],
        allow_endpoint_rejected=True,
    )
    assert selected == [3, 4]
    assert capped == []
    assert rejected == []

    with pytest.raises(letta_hpc.WorkflowError, match="unknown task indices"):
        letta_hpc.select_resume_indices(
            rows,
            maximum_directional_passes=41,
            requested=[99],
        )


def test_unsafe_checkpoint_pair_detection():
    rows = [
        {
            "task_index": 0,
            "has_result": False,
            "has_snapshot": False,
        },
        {
            "task_index": 1,
            "has_result": True,
            "has_snapshot": False,
        },
        {
            "task_index": 2,
            "has_result": False,
            "has_snapshot": True,
        },
        {
            "task_index": 3,
            "has_result": True,
            "has_snapshot": True,
            "checkpoint_pair_valid": False,
        },
        {
            "task_index": 4,
            "has_result": True,
            "has_snapshot": True,
            "checkpoint_pair_valid": True,
        },
    ]
    assert letta_hpc.unsafe_checkpoint_indices(rows) == [1, 2, 3]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("scan", "gain_tolerance", 1.0e-7),
        ("scan", "pair_workers", 7),
        ("scan", "frontier_workers", 3),
        ("scan", "source_run_root", "/cluster/other-parent"),
        ("cluster", "remote_source_root", "/cluster/sources/other"),
        ("cluster", "source_id", "sha256:different"),
    ),
)
def test_numerical_fingerprint_changes_with_protocol_or_source(
    section,
    key,
    value,
):
    base = _normalized()
    changed = copy.deepcopy(base)
    changed[section][key] = value
    assert (
        letta_hpc.numerical_protocol_fingerprint(changed)
        != letta_hpc.numerical_protocol_fingerprint(base)
    )


def test_source_free_default_does_not_change_legacy_fingerprint():
    base = _normalized()
    legacy = copy.deepcopy(base)
    legacy["scan"].pop("source_free")
    assert (
        letta_hpc.numerical_protocol_fingerprint(base)
        == letta_hpc.numerical_protocol_fingerprint(legacy)
    )


def test_config_rejects_non_per_site_gain_tolerance_units():
    raw = _raw_config()
    raw["scan"]["gain_tolerance_units"] = "total_energy"
    with pytest.raises(letta_hpc.WorkflowError, match="energy_per_site"):
        _normalized(raw)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("scan", "maximum_directional_passes", 500),
        ("cluster", "partition", "other"),
        ("cluster", "qos", "debug"),
        ("cluster", "cpus_per_task", 32),
        ("cluster", "memory", "64G"),
        ("cluster", "time_limit", "12:00:00"),
        ("cluster", "concurrency", 7),
        ("cluster", "slurm_directives", ["--constraint=fast"]),
    ),
)
def test_numerical_fingerprint_ignores_pass_cap_and_slurm_resources(
    section,
    key,
    value,
):
    base = _normalized()
    changed = copy.deepcopy(base)
    changed[section][key] = value
    assert (
        letta_hpc.numerical_protocol_fingerprint(changed)
        == letta_hpc.numerical_protocol_fingerprint(base)
    )


def test_source_content_id_is_order_and_mtime_independent(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.py").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "nested" / "b.py").write_text("beta\n", encoding="utf-8")
    files = [Path("a.py"), Path("nested/b.py")]

    expected = letta_hpc.source_content_id(tmp_path, files)
    os.utime(tmp_path / "a.py", ns=(1_000_000_000, 1_000_000_000))
    os.utime(
        tmp_path / "nested" / "b.py",
        ns=(2_000_000_000, 2_000_000_000),
    )

    assert letta_hpc.source_content_id(tmp_path, list(reversed(files))) == expected
    (tmp_path / "a.py").write_text("changed\n", encoding="utf-8")
    assert letta_hpc.source_content_id(tmp_path, files) != expected


def test_remote_source_verifier_rejects_changed_upload(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    records = [
        {
            "path": "module.py",
            "sha256": letta_hpc._sha256_file(tmp_path / "module.py"),
        }
    ]
    marker = tmp_path / ".letta_source_manifest.json"
    marker.write_text(
        json.dumps(
            {
                "source_id": "sha256:"
                + letta_hpc._source_records_id(records),
                "files": records,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            letta_hpc._VERIFY_SOURCE_CODE,
            str(tmp_path),
            str(marker),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "module.py").write_text("value = 2\n", encoding="utf-8")
    failed = subprocess.run(
        [
            sys.executable,
            "-c",
            letta_hpc._VERIFY_SOURCE_CODE,
            str(tmp_path),
            str(marker),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "source hash mismatch" in failed.stderr


def test_freeze_source_code_copies_and_verifies_real_checkpoint_schema(tmp_path):
    source_root = tmp_path / "parent"
    source_task = source_root / "task_000_j2_0p5"
    source_task.mkdir(parents=True)
    result = {
        "model": {"j2": 0.5},
        "result": {"checkpoint_id": "checkpoint-1", "energy": -17.25},
    }
    (source_task / "result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    np.savez_compressed(
        source_task / "state.npz",
        checkpoint_id=np.asarray("checkpoint-1"),
        recorded_energy=np.asarray(-17.25),
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("0.5\n", encoding="utf-8")
    frozen = tmp_path / "frozen"

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            letta_hpc._FREEZE_SOURCE_CODE,
            str(source_root),
            str(frozen),
            str(manifest),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(process.stdout)
    assert summary["copied"] == 1
    frozen_task = frozen / "task_000_j2_0p5"
    assert json.loads((frozen_task / "result.json").read_text()) == result
    frozen_manifest = json.loads(
        (frozen / "source_manifest.json").read_text()
    )
    assert frozen_manifest[0]["checkpoint_id"] == "checkpoint-1"
    assert frozen_manifest[0]["energy"] == pytest.approx(-17.25)


def _write_local_result(path, *, j2, energy, converged):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": {"j2": j2, "nsites": 36},
                "status": "converged" if converged else "maximum_passes",
                "result": {
                    "converged": converged,
                    "energy": energy,
                    "energy_per_site": energy / 36.0,
                    "directional_passes_completed": 20,
                    "last_directional_gain": 2.0e-7,
                    "last_directional_gain_per_site": 2.0e-7 / 36.0,
                    "last_cycle_maximum_gain": 3.0e-7,
                    "last_cycle_maximum_gain_per_site": 3.0e-7 / 36.0,
                },
                "timing_seconds": {"total": 42.0},
            }
        ),
        encoding="utf-8",
    )


def test_local_result_aggregation_skips_partial_and_corrupt_results(tmp_path):
    artifacts = tmp_path / "artifacts"
    _write_local_result(
        artifacts / "task_000_j2_0p0" / "result.json",
        j2=0.0,
        energy=-21.0,
        converged=True,
    )
    corrupt = artifacts / "task_001_j2_0p5" / "result.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{not-json", encoding="utf-8")
    partial = artifacts / "task_002_j2_1p0" / "result.json"
    partial.parent.mkdir(parents=True)
    partial.write_text(json.dumps({"status": "running"}), encoding="utf-8")

    rows = letta_hpc._local_result_rows(tmp_path, metadata={})

    assert len(rows) == 1
    assert rows[0]["task_index"] == 0
    assert rows[0]["j2"] == pytest.approx(0.0)
    assert rows[0]["energy"] == pytest.approx(-21.0)
