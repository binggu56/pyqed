#!/usr/bin/env python3
"""Reproducible SSH/Slurm workflow control for LETTA parameter scans.

The controller deliberately keeps authentication outside its metadata.  Start
one OpenSSH control connection with ``connect`` (including any required OTP);
the remaining commands reuse that connection for SSH and rsync operations.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fnmatch
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
CONTROLLER_VERSION = "0.1.0"
ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}
FAILED_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "LAUNCH_FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
DEFAULT_RSYNC_EXCLUDES = (
    ".git/",
    ".letta_hpc/",
    ".venv/",
    ".pytest_cache/",
    ".ruff_cache/",
    "__pycache__/",
    "build/",
    "dist/",
    "hpc_runs/",
    "examples/mps/results/",
    "*.pyc",
    "*.so",
    "*.dylib",
    "*.npz",
    "*.npy",
    "*.log",
    "*.out",
    "*.pdf",
    "*.png",
    "*.svg",
)
DEFAULT_SOURCE_PATHS = (
    "pyqed",
    "examples/mps",
    "pyproject.toml",
    "setup.py",
    "MANIFEST.in",
)


class WorkflowError(RuntimeError):
    """User-facing workflow configuration or execution error."""


class CommandRunner:
    """Small injectable subprocess boundary used by the CLI and tests."""

    def __init__(self, *, dry_run: bool = False, verbose: bool = False):
        self.dry_run = bool(dry_run)
        self.verbose = bool(verbose or dry_run)

    def run(
        self,
        argv: Sequence[str | os.PathLike[str]],
        *,
        capture: bool = False,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [os.fspath(item) for item in argv]
        if self.verbose:
            print("+", shlex.join(command), file=sys.stderr)
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.run(
            command,
            cwd=None if cwd is None else os.fspath(cwd),
            check=check,
            text=True,
            capture_output=capture,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not slug:
        raise WorkflowError(f"cannot form a run name from {value!r}")
    return slug


def _run_id(value: str) -> str:
    cleaned = _slug(value)
    if value != cleaned or len(value) > 128:
        raise WorkflowError(
            "run IDs may contain only letters, digits, '.', '_', and '-'"
        )
    return value


def _default_run_id(name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{_slug(name)}"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "pyqed").is_dir():
            return candidate
    raise WorkflowError(f"cannot find the pyqed repository above {start}")


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".json":
        payload = json.loads(data)
    elif suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError as error:  # pragma: no cover - Python < 3.11
            raise WorkflowError(
                "TOML configuration requires Python 3.11+; use JSON otherwise."
            ) from error
        payload = tomllib.loads(data.decode("utf-8"))
    else:
        raise WorkflowError("configuration must be JSON or TOML")
    if not isinstance(payload, dict):
        raise WorkflowError("configuration root must be a mapping")
    return payload


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise WorkflowError(f"missing [{section}] {key}")
    return value


def _remote_path(value: Any, label: str) -> str:
    path = str(value)
    if not path.startswith("/"):
        raise WorkflowError(f"{label} must be an absolute remote path: {path!r}")
    if not re.fullmatch(r"/[A-Za-z0-9._/-]*", path):
        raise WorkflowError(
            f"{label} contains unsafe remote-shell characters: {path!r}"
        )
    return path.rstrip("/") or "/"


def _safe_token(value: Any, label: str, pattern: str) -> str:
    token = str(value)
    if not re.fullmatch(pattern, token):
        raise WorkflowError(f"{label} has an invalid value: {token!r}")
    return token


def _positive_int(value: Any, label: str) -> int:
    result = int(value)
    if result < 1:
        raise WorkflowError(f"{label} must be positive")
    return result


def normalize_config(raw: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    """Validate and normalize a user configuration into JSON-compatible data."""

    cluster_raw = dict(raw.get("cluster", {}))
    scan_raw = dict(raw.get("scan", {}))
    local_raw = dict(raw.get("local", {}))
    repository = _repo_root(config_path.parent)

    host = str(_required(cluster_raw, "host", "cluster"))
    if not re.fullmatch(
        r"(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+", host
    ):
        raise WorkflowError("[cluster] host has unsafe characters")
    port = _positive_int(cluster_raw.get("port", 22), "cluster.port")
    if port > 65535:
        raise WorkflowError("cluster.port must be at most 65535")
    concurrency = _positive_int(
        cluster_raw.get("concurrency", 1), "cluster.concurrency"
    )
    cpus = _positive_int(
        cluster_raw.get("cpus_per_task", 1), "cluster.cpus_per_task"
    )

    values_raw = _required(scan_raw, "values", "scan")
    if not isinstance(values_raw, list) or not values_raw:
        raise WorkflowError("[scan] values must be a nonempty array")
    values = [float(value) for value in values_raw]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise WorkflowError("[scan] values must be finite and nonnegative")
    if len(set(values)) != len(values):
        raise WorkflowError("[scan] values must not contain duplicates")

    tolerance = float(scan_raw.get("gain_tolerance", 1.0e-4))
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise WorkflowError("[scan] gain_tolerance must be finite and nonnegative")
    tolerance_units = str(
        scan_raw.get("gain_tolerance_units", "energy_per_site")
    )
    if tolerance_units != "energy_per_site":
        raise WorkflowError(
            "[scan] gain_tolerance_units must be 'energy_per_site'"
        )
    maximum_passes = _positive_int(
        scan_raw.get("maximum_directional_passes", 40),
        "scan.maximum_directional_passes",
    )

    source_free = bool(scan_raw.get("source_free", False))
    source_run_root = scan_raw.get("source_run_root")
    seed_result = scan_raw.get("seed_result")
    seed_snapshot = scan_raw.get("seed_snapshot")
    if source_free:
        if source_run_root or seed_result or seed_snapshot:
            raise WorkflowError(
                "[scan] source_free cannot be combined with source_run_root "
                "or seed checkpoints"
            )
    elif source_run_root:
        source_run_root = _remote_path(source_run_root, "scan.source_run_root")
    elif not seed_result or not seed_snapshot:
        raise WorkflowError(
            "[scan] requires source_run_root, both seed_result and "
            "seed_snapshot, or source_free = true"
        )

    local_source = Path(str(local_raw.get("source_root", ".")))
    if not local_source.is_absolute():
        local_source = repository / local_source
    local_runs = Path(str(local_raw.get("runs_root", ".letta_hpc/runs")))
    if not local_runs.is_absolute():
        local_runs = repository / local_runs

    extra_ssh = cluster_raw.get("ssh_options", [])
    extra_slurm = cluster_raw.get("slurm_directives", [])
    excludes = local_raw.get("rsync_excludes", list(DEFAULT_RSYNC_EXCLUDES))
    source_paths = local_raw.get("source_paths", list(DEFAULT_SOURCE_PATHS))
    for collection, label in (
        (extra_ssh, "cluster.ssh_options"),
        (extra_slurm, "cluster.slurm_directives"),
        (excludes, "local.rsync_excludes"),
        (source_paths, "local.source_paths"),
    ):
        if not isinstance(collection, list) or any(
            not isinstance(item, str) or "\n" in item for item in collection
        ):
            raise WorkflowError(f"{label} must be an array of one-line strings")

    worker = str(
        scan_raw.get(
            "worker",
            "examples/mps/cluster/run_sector_projected_letta_j2_task_6x6.sh",
        )
    )
    if "\n" in worker or not worker:
        raise WorkflowError("[scan] worker must be a nonempty one-line path")
    for source_path in source_paths:
        candidate = Path(source_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise WorkflowError(
                "local.source_paths entries must stay inside source_root"
            )

    normalized = {
        "cluster": {
            "host": host,
            "port": port,
            "remote_source_root": _remote_path(
                _required(cluster_raw, "remote_source_root", "cluster"),
                "cluster.remote_source_root",
            ),
            "remote_sources_root": _remote_path(
                cluster_raw.get(
                    "remote_sources_root",
                    str(
                        Path(
                            _remote_path(
                                _required(
                                    cluster_raw,
                                    "remote_source_root",
                                    "cluster",
                                ),
                                "cluster.remote_source_root",
                            )
                        ).parent
                        / "sources"
                    ),
                ),
                "cluster.remote_sources_root",
            ),
            "source_id": str(
                cluster_raw.get(
                    "source_id",
                    "external:"
                    + _remote_path(
                        _required(cluster_raw, "remote_source_root", "cluster"),
                        "cluster.remote_source_root",
                    ),
                )
            ),
            "remote_runs_root": _remote_path(
                _required(cluster_raw, "remote_runs_root", "cluster"),
                "cluster.remote_runs_root",
            ),
            "python": _remote_path(
                _required(cluster_raw, "python", "cluster"), "cluster.python"
            ),
            "partition": _safe_token(
                _required(cluster_raw, "partition", "cluster"),
                "cluster.partition",
                r"[A-Za-z0-9._-]+",
            ),
            "slurm_bin": (
                ""
                if not cluster_raw.get("slurm_bin")
                else _remote_path(cluster_raw["slurm_bin"], "cluster.slurm_bin")
            ),
            "qos": (
                ""
                if not cluster_raw.get("qos")
                else _safe_token(
                    cluster_raw["qos"],
                    "cluster.qos",
                    r"[A-Za-z0-9._-]+",
                )
            ),
            "cpus_per_task": cpus,
            "memory": _safe_token(
                cluster_raw.get("memory", "16G"),
                "cluster.memory",
                r"[0-9]+(?:[KMGTP]i?B?|[KMGTP])?",
            ),
            "time_limit": _safe_token(
                cluster_raw.get("time_limit", "04:00:00"),
                "cluster.time_limit",
                r"(?:[0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}",
            ),
            "concurrency": concurrency,
            "control_persist": str(cluster_raw.get("control_persist", "8h")),
            "server_alive_interval": _positive_int(
                cluster_raw.get("server_alive_interval", 30),
                "cluster.server_alive_interval",
            ),
            "server_alive_count_max": _positive_int(
                cluster_raw.get("server_alive_count_max", 20),
                "cluster.server_alive_count_max",
            ),
            "ssh_options": list(extra_ssh),
            "slurm_directives": list(extra_slurm),
            "requeue": bool(cluster_raw.get("requeue", True)),
        },
        "scan": {
            "name": _slug(str(scan_raw.get("name", "letta-scan"))),
            "values": values,
            "gain_tolerance": tolerance,
            "gain_tolerance_units": tolerance_units,
            "maximum_directional_passes": maximum_passes,
            "pair_workers": _positive_int(
                scan_raw.get("pair_workers", 4), "scan.pair_workers"
            ),
            "frontier_workers": _positive_int(
                scan_raw.get("frontier_workers", 2), "scan.frontier_workers"
            ),
            "source_free": source_free,
            "source_run_root": source_run_root,
            "seed_result": (
                None
                if seed_result is None
                else _remote_path(seed_result, "scan.seed_result")
            ),
            "seed_snapshot": (
                None
                if seed_snapshot is None
                else _remote_path(seed_snapshot, "scan.seed_snapshot")
            ),
            "freeze_source": bool(scan_raw.get("freeze_source", bool(source_run_root))),
            "worker": worker,
        },
        "local": {
            "repository": os.fspath(repository),
            "runs_root": os.fspath(local_runs.resolve()),
            "source_root": os.fspath(local_source.resolve()),
            "sync_source": bool(local_raw.get("sync_source", False)),
            "rsync_excludes": list(excludes),
            "source_paths": list(source_paths),
            "build_command": str(local_raw.get("build_command", "")),
        },
    }
    return normalized


def load_config(path: str | os.PathLike[str]) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise WorkflowError(f"configuration does not exist: {config_path}")
    return normalize_config(
        _load_raw_config(config_path), config_path=config_path
    ), config_path


def _control_path() -> str:
    override = os.environ.get("LETTA_HPC_CONTROL_PATH")
    if override:
        return os.fspath(Path(override).expanduser())
    return os.fspath(Path.home() / ".ssh" / "pyqed-hpc-%C")


def ssh_options(
    cluster: dict[str, Any], *, batch_mode: bool = False
) -> list[str]:
    options = [
        "-p",
        str(cluster["port"]),
        "-o",
        f"ServerAliveInterval={cluster['server_alive_interval']}",
        "-o",
        f"ServerAliveCountMax={cluster['server_alive_count_max']}",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPersist={cluster['control_persist']}",
        "-o",
        f"ControlPath={_control_path()}",
    ]
    if batch_mode:
        options.extend(["-o", "BatchMode=yes"])
    options.extend(cluster.get("ssh_options", []))
    return options


def ssh_argv(
    cluster: dict[str, Any],
    remote_command: str | None = None,
    *,
    batch_mode: bool = True,
) -> list[str]:
    argv = [
        "ssh",
        *ssh_options(cluster, batch_mode=batch_mode),
        cluster["host"],
    ]
    if remote_command is not None:
        argv.append(remote_command)
    return argv


def rsync_shell(cluster: dict[str, Any]) -> str:
    return shlex.join(["ssh", *ssh_options(cluster, batch_mode=True)])


def remote_run(
    runner: CommandRunner,
    cluster: dict[str, Any],
    argv: Sequence[str],
    *,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = shlex.join([str(item) for item in argv])
    return runner.run(
        ssh_argv(cluster, command), capture=capture, check=check
    )


def _git_provenance(repository: Path) -> dict[str, Any]:
    def capture(*argv: str) -> str | None:
        try:
            return subprocess.run(
                argv,
                cwd=repository,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = capture("git", "status", "--porcelain")
    return {
        "repository": os.fspath(repository),
        "commit": capture("git", "rev-parse", "HEAD"),
        "branch": capture("git", "branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "python": sys.version,
        "platform": sys.platform,
    }


def _excluded_source_path(relative: Path, patterns: Sequence[str]) -> bool:
    posix = relative.as_posix()
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if normalized.endswith("/"):
            prefix = normalized.rstrip("/")
            if posix == prefix or posix.startswith(prefix + "/"):
                return True
            if "/" not in prefix and prefix in relative.parts:
                return True
        elif fnmatch.fnmatch(relative.name, normalized) or fnmatch.fnmatch(
            posix, normalized
        ):
            return True
    return False


def source_files(spec: dict[str, Any]) -> list[Path]:
    """Return the deterministic file set used for an immutable source upload."""

    local = spec["local"]
    root = Path(local["source_root"])
    if not root.is_dir():
        raise WorkflowError(f"local source root does not exist: {root}")
    files: set[Path] = set()
    for entry in local["source_paths"]:
        path = root / entry
        if not path.exists():
            raise WorkflowError(f"configured source path does not exist: {path}")
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(root)
            if not _excluded_source_path(
                relative, local["rsync_excludes"]
            ):
                files.add(relative)
    if not files:
        raise WorkflowError("source selection is empty")
    return sorted(files, key=lambda path: path.as_posix())


def source_content_id(root: Path, files: Sequence[Path]) -> str:
    """Hash paths and bytes, independent of mtimes and enumeration order."""

    records = [
        {
            "path": relative.as_posix(),
            "sha256": _sha256_file(root / relative),
        }
        for relative in sorted(files, key=lambda path: path.as_posix())
    ]
    return _source_records_id(records)


def _source_records_id(records: Sequence[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        encoded = record["path"].encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(record["sha256"]))
    return digest.hexdigest()


def numerical_protocol_fingerprint(spec: dict[str, Any]) -> str:
    """Fingerprint settings the current driver requires to remain unchanged."""

    scan = spec["scan"]
    cluster = spec["cluster"]
    stable = {
        "values": scan["values"],
        "gain_tolerance": scan["gain_tolerance"],
        "pair_workers": scan["pair_workers"],
        "frontier_workers": scan["frontier_workers"],
        "source_run_root": scan.get("source_run_root"),
        "seed_result": scan.get("seed_result"),
        "seed_snapshot": scan.get("seed_snapshot"),
        "freeze_source": scan.get("freeze_source"),
        "worker": scan["worker"],
        "starting_directional_sweep": "auto",
        "remote_source_root": cluster["remote_source_root"],
        "source_id": cluster["source_id"],
        "python": cluster["python"],
    }
    if scan.get("source_free", False):
        stable["source_free"] = True
    if "gain_tolerance_units" in scan:
        stable["gain_tolerance_units"] = scan["gain_tolerance_units"]
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _compress_indices(indices: Iterable[int]) -> str:
    unique = sorted(set(int(index) for index in indices))
    if not unique or unique[0] < 0:
        raise WorkflowError("Slurm array indices must be nonempty and nonnegative")
    ranges: list[str] = []
    start = stop = unique[0]
    for index in unique[1:]:
        if index == stop + 1:
            stop = index
            continue
        ranges.append(str(start) if start == stop else f"{start}-{stop}")
        start = stop = index
    ranges.append(str(start) if start == stop else f"{start}-{stop}")
    return ",".join(ranges)


def _array_spec(indices: Iterable[int], concurrency: int) -> str:
    return f"{_compress_indices(indices)}%{int(concurrency)}"


def _worker_path(cluster: dict[str, Any], scan: dict[str, Any]) -> str:
    worker = scan["worker"]
    if worker.startswith("/"):
        return worker
    return f"{cluster['remote_source_root']}/{worker}"


def _slurm_command(cluster: dict[str, Any], name: str) -> str:
    directory = cluster.get("slurm_bin", "")
    return f"{directory}/{name}" if directory else name


def render_sbatch(
    spec: dict[str, Any],
    *,
    remote_run_dir: str,
    source_run_root: str | None,
    indices: Iterable[int] | None = None,
    maximum_directional_passes: int | None = None,
) -> str:
    """Render the run-specific Slurm script."""

    cluster = spec["cluster"]
    scan = spec["scan"]
    task_indices = (
        list(range(len(scan["values"]))) if indices is None else list(indices)
    )
    maximum = (
        scan["maximum_directional_passes"]
        if maximum_directional_passes is None
        else _positive_int(
            maximum_directional_passes, "maximum_directional_passes"
        )
    )
    lines = [
        "#!/bin/bash",
        f"#SBATCH -p {cluster['partition']}",
    ]
    if cluster["qos"]:
        lines.append(f"#SBATCH -q {cluster['qos']}")
    lines.extend(
        [
            f"#SBATCH -J {_slug(scan['name'])[:64]}",
            f"#SBATCH -c {cluster['cpus_per_task']}",
            f"#SBATCH --mem={cluster['memory']}",
            f"#SBATCH --time={cluster['time_limit']}",
            f"#SBATCH --array={_array_spec(task_indices, cluster['concurrency'])}",
            f"#SBATCH -o {remote_run_dir}/%A_%a.sbatch.out",
            f"#SBATCH -e {remote_run_dir}/%A_%a.sbatch.err",
        ]
    )
    if cluster.get("requeue", True):
        lines.append("#SBATCH --requeue")
    for directive in cluster.get("slurm_directives", []):
        lines.append(
            directive if directive.startswith("#SBATCH") else f"#SBATCH {directive}"
        )
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            f"export PYQED_ROOT={shlex.quote(cluster['remote_source_root'])}",
            f"export PYTHON={shlex.quote(cluster['python'])}",
            f"export LETTA_RUN_ROOT={shlex.quote(remote_run_dir)}",
            "export LETTA_STARTING_DIRECTIONAL_SWEEP=auto",
            f"export LETTA_MAXIMUM_DIRECTIONAL_PASSES={maximum}",
            f"export LETTA_GAIN_TOLERANCE={float(scan['gain_tolerance'])!r}",
            "export LETTA_GAIN_TOLERANCE_UNITS=energy_per_site",
            "export LETTA_COLOCATE_LOGS=1",
            f"export LETTA_PAIR_WORKERS={scan['pair_workers']}",
            f"export LETTA_FRONTIER_WORKERS={scan['frontier_workers']}",
        ]
    )
    if source_run_root:
        lines.append(
            f"export LETTA_SOURCE_RUN_ROOT={shlex.quote(source_run_root)}"
        )
    elif not scan.get("source_free", False):
        lines.extend(
            [
                f"export LETTA_SEED_RESULT={shlex.quote(scan['seed_result'])}",
                f"export LETTA_SEED_SNAPSHOT={shlex.quote(scan['seed_snapshot'])}",
            ]
        )
    lines.extend(
        [
            "",
            "exec bash \\",
            f"    {shlex.quote(_worker_path(cluster, scan))} \\",
            '    "$SLURM_ARRAY_TASK_ID" \\',
            f"    {shlex.quote(remote_run_dir + '/manifest.txt')}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest(path: Path, values: Sequence[float]) -> None:
    path.write_text(
        "".join(f"{float(value)!r}\n" for value in values),
        encoding="utf-8",
    )


def _execution_source_root(
    spec: dict[str, Any], remote_run_dir: str
) -> tuple[str | None, str | None]:
    original = spec["scan"].get("source_run_root")
    if original and spec["scan"].get("freeze_source", True):
        return f"{remote_run_dir}/source_checkpoints", original
    return original, None


def _prepare_run(
    spec: dict[str, Any],
    *,
    config_path: Path | None,
    run_id: str,
    parent_run_id: str | None = None,
    remote_run_dir: str | None = None,
    existing_job_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    run_id = _run_id(run_id)
    runs_root = Path(spec["local"]["runs_root"])
    local_run_dir = runs_root / run_id
    if local_run_dir.exists():
        raise WorkflowError(f"local run already exists: {local_run_dir}")
    local_run_dir.mkdir(parents=True)
    remote_dir = (
        _remote_path(remote_run_dir, "remote run directory")
        if remote_run_dir
        else f"{spec['cluster']['remote_runs_root']}/{run_id}"
    )
    execution_source, frozen_from = _execution_source_root(spec, remote_dir)

    _write_manifest(local_run_dir / "manifest.txt", spec["scan"]["values"])
    (local_run_dir / "submit.sbatch").write_text(
        render_sbatch(
            spec,
            remote_run_dir=remote_dir,
            source_run_root=execution_source,
        ),
        encoding="utf-8",
    )
    _atomic_json(local_run_dir / "config.normalized.json", spec)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "controller_version": CONTROLLER_VERSION,
        "run_id": run_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "state": "adopted" if existing_job_id else "prepared",
        "config_path": None if config_path is None else os.fspath(config_path),
        "config_sha256": (
            None if config_path is None else _sha256_file(config_path)
        ),
        "controller_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol_fingerprint": numerical_protocol_fingerprint(spec),
        "source_id": spec["cluster"]["source_id"],
        "source_content_addressed": spec["cluster"]["source_id"].startswith(
            "sha256:"
        ),
        "config": spec,
        "local_run_dir": os.fspath(local_run_dir.resolve()),
        "remote_run_dir": remote_dir,
        "execution": {
            "source_run_root": execution_source,
            "frozen_from": frozen_from,
            "manifest": f"{remote_dir}/manifest.txt",
            "sbatch": f"{remote_dir}/submit.sbatch",
        },
        "parent_run_id": parent_run_id,
        "provenance": _git_provenance(Path(spec["local"]["repository"])),
        "submissions": [],
    }
    if existing_job_id:
        metadata["submissions"].append(
            {
                "job_id": str(existing_job_id),
                "kind": "adopted",
                "submitted_at": _utc_now(),
                "indices": list(range(len(spec["scan"]["values"]))),
                "maximum_directional_passes": spec["scan"][
                    "maximum_directional_passes"
                ],
                "protocol_fingerprint": metadata["protocol_fingerprint"],
                "source_id": metadata["source_id"],
            }
        )
    _atomic_json(local_run_dir / "run.json", metadata)
    return metadata, local_run_dir


def _load_run(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        raise WorkflowError(f"run metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise WorkflowError(f"unsupported run metadata schema in {path}")
    return payload


def _save_run(run_dir: Path, metadata: dict[str, Any]) -> None:
    metadata["updated_at"] = _utc_now()
    _atomic_json(run_dir / "run.json", metadata)


@contextmanager
def _run_mutation_lock(run_dir: Path):
    path = run_dir / ".controller.lock"
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _resolve_runs_root(
    config: dict[str, Any] | None, explicit: str | None
) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if config:
        return Path(config["local"]["runs_root"])
    try:
        repository = _repo_root(Path.cwd())
    except WorkflowError:
        repository = Path.cwd()
    return repository / ".letta_hpc" / "runs"


def resolve_run_dir(
    reference: str | None,
    *,
    runs_root: Path,
) -> Path:
    if reference:
        path = Path(reference).expanduser()
        if path.is_dir():
            return path.resolve()
        candidate = runs_root / reference
        if candidate.is_dir():
            return candidate.resolve()
        raise WorkflowError(f"unknown run: {reference}")
    candidates = sorted(
        (path.parent for path in runs_root.glob("*/run.json")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise WorkflowError(f"no runs found under {runs_root}")
    return candidates[-1].resolve()


def _rsync_files(
    runner: CommandRunner,
    cluster: dict[str, Any],
    sources: Sequence[Path],
    remote_directory: str,
) -> None:
    argv = [
        "rsync",
        "-az",
        "--partial",
        "-e",
        rsync_shell(cluster),
        *(os.fspath(path) for path in sources),
        f"{cluster['host']}:{remote_directory}/",
    ]
    runner.run(argv)


_FREEZE_SOURCE_CODE = r"""
import hashlib
import json
import os
import shutil
import sys

source_root, target_root, manifest_path = sys.argv[1:]
entries = []
for line in open(manifest_path, encoding="utf-8"):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    token = line.split()[0]
    entries.append((token, float(token)))
records = []
for index, (token, value) in enumerate(entries):
    tag = token.replace("-", "m").replace(".", "p")
    candidates = [
        os.path.join(source_root, f"task_{index:03d}_j2_{tag}"),
        *sorted(
            path
            for path in __import__("glob").glob(
                os.path.join(source_root, f"task_{index:03d}_j2_*")
            )
        ),
    ]
    valid = []
    for path in dict.fromkeys(candidates):
        result_path = os.path.join(path, "result.json")
        snapshot_path = os.path.join(path, "state.npz")
        if not os.path.isfile(result_path) or not os.path.isfile(snapshot_path):
            continue
        try:
            payload = json.load(open(result_path, encoding="utf-8"))
            actual = float(payload["model"]["j2"])
        except Exception:
            continue
        if abs(actual - value) <= 1e-12:
            valid.append(path)
    if len(valid) != 1:
        raise SystemExit(f"missing source checkpoint for task {index}")
    source_dir = valid[0]
    target_dir = os.path.join(target_root, f"task_{index:03d}_j2_{tag}")
    os.makedirs(target_dir, exist_ok=False)
    copied = {}
    for name in ("result.json", "state.npz"):
        source = os.path.join(source_dir, name)
        target = os.path.join(target_dir, name)
        shutil.copy2(source, target)
        digest = hashlib.sha256()
        with open(target, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        copied[name] = digest.hexdigest()
    result = json.load(
        open(os.path.join(target_dir, "result.json"), encoding="utf-8")
    )
    checkpoint_id = str(result["result"].get("checkpoint_id") or "")
    energy = float(result["result"]["energy"])
    if not checkpoint_id:
        raise SystemExit(f"source task {index} has no checkpoint ID")
    import numpy as np

    with np.load(
        os.path.join(target_dir, "state.npz"), allow_pickle=False
    ) as snapshot:
        snapshot_id = str(np.asarray(snapshot["checkpoint_id"]).item())
        snapshot_energy = float(
            np.asarray(snapshot["recorded_energy"]).item()
        )
    if snapshot_id != checkpoint_id:
        raise SystemExit(f"source task {index} checkpoint IDs disagree")
    if abs(snapshot_energy - energy) > 5e-10:
        raise SystemExit(f"source task {index} checkpoint energies disagree")
    records.append(
        {
            "task_index": index,
            "j2": value,
            "source": source_dir,
            "target": target_dir,
            "checkpoint_id": checkpoint_id,
            "energy": energy,
            "sha256": copied,
        }
    )
with open(
    os.path.join(target_root, "source_manifest.json"),
    "w",
    encoding="utf-8",
) as stream:
    json.dump(records, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({"copied": len(records), "target": target_root}))
""".strip()


def _freeze_sources(
    runner: CommandRunner,
    metadata: dict[str, Any],
) -> None:
    frozen_from = metadata["execution"].get("frozen_from")
    target = metadata["execution"].get("source_run_root")
    if not frozen_from or not target:
        return
    cluster = metadata["config"]["cluster"]
    remote_run(
        runner,
        cluster,
        [
            cluster["python"],
            "-c",
            _FREEZE_SOURCE_CODE,
            frozen_from,
            target,
            metadata["execution"]["manifest"],
        ],
    )


def _parse_job_id(output: str) -> str:
    lines = output.strip().splitlines()
    if not lines:
        raise WorkflowError(f"cannot parse sbatch job ID from {output!r}")
    token = lines[-1].split(";", 1)[0].strip()
    if not re.fullmatch(r"\d+", token):
        raise WorkflowError(f"cannot parse sbatch job ID from {output!r}")
    return token


def _submission_request_id(
    metadata: dict[str, Any],
    *,
    kind: str,
    indices: Sequence[int],
    maximum_directional_passes: int,
) -> str:
    payload = {
        "run_id": metadata["run_id"],
        "kind": kind,
        "indices": list(map(int, indices)),
        "maximum_directional_passes": int(maximum_directional_passes),
        "protocol_fingerprint": metadata["protocol_fingerprint"],
        "source_id": metadata["source_id"],
        "predecessors": [
            {
                "request_id": item.get("request_id"),
                "job_id": item["job_id"],
            }
            for item in metadata.get("submissions", [])
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


_REMOTE_SUBMIT_CODE = r"""
import fcntl
import json
import os
import re
import subprocess
import sys

root, sbatch, script, request_id, encoded = sys.argv[1:]
directory = os.path.join(root, ".submissions")
os.makedirs(directory, exist_ok=True)
record_path = os.path.join(directory, request_id + ".json")
lock_path = os.path.join(directory, ".lock")

def write_atomic(payload):
    temporary = record_path + f".tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, record_path)

with open(lock_path, "a+", encoding="utf-8") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if os.path.isfile(record_path):
        prior = json.load(open(record_path, encoding="utf-8"))
        if prior.get("state") == "submitted":
            print(json.dumps(prior, separators=(",", ":")))
            raise SystemExit(0)
        if prior.get("state") in {"submitting", "ambiguous"}:
            raise SystemExit(
                "an interrupted submission journal exists; refusing a "
                "possibly duplicate sbatch"
            )
    record = json.loads(encoded)
    record.update(
        {
            "request_id": request_id,
            "state": "submitting",
            "job_name": "letta-" + request_id[:16],
        }
    )
    write_atomic(record)
    try:
        process = subprocess.run(
            [
                sbatch,
                "--parsable",
                "--job-name=" + record["job_name"],
                script,
            ],
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception as error:
        record["state"] = "failed"
        record["error"] = repr(error)
        write_atomic(record)
        raise
    if process.returncode != 0:
        record["state"] = "failed"
        record["error"] = (
            f"sbatch exited {process.returncode}: "
            f"{process.stderr.strip()}"
        )
        write_atomic(record)
        raise SystemExit(record["error"])
    record["state"] = "ambiguous"
    record["sbatch_stdout"] = process.stdout
    record["sbatch_stderr"] = process.stderr
    write_atomic(record)
    try:
        token = process.stdout.strip().splitlines()[-1].split(";", 1)[0]
        if not re.fullmatch(r"\d+", token):
            raise RuntimeError(
                f"cannot parse sbatch job ID from {process.stdout!r}"
            )
    except Exception:
        raise
    record["state"] = "submitted"
    record["job_id"] = token
    write_atomic(record)
    print(json.dumps(record, separators=(",", ":")))
""".strip()


def _record_submission(
    metadata: dict[str, Any],
    *,
    job_id: str,
    kind: str,
    indices: Sequence[int],
    maximum_directional_passes: int,
    request_id: str,
    submitted_at: str,
) -> tuple[dict[str, Any], bool]:
    for existing in metadata["submissions"]:
        if existing.get("request_id") != request_id:
            continue
        if str(existing["job_id"]) != str(job_id):
            raise WorkflowError(
                f"submission request {request_id} maps to conflicting jobs"
            )
        return existing, False
    record = {
        "job_id": str(job_id),
        "kind": kind,
        "submitted_at": submitted_at,
        "indices": list(map(int, indices)),
        "maximum_directional_passes": int(maximum_directional_passes),
        "protocol_fingerprint": metadata["protocol_fingerprint"],
        "source_id": metadata["source_id"],
        "request_id": request_id,
    }
    metadata["submissions"].append(record)
    metadata["state"] = "submitted"
    return record, True


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )


def _submit_prepared(
    runner: CommandRunner,
    metadata: dict[str, Any],
    run_dir: Path,
    *,
    kind: str = "submit",
    script_name: str = "submit.sbatch",
    indices: Sequence[int] | None = None,
    maximum_directional_passes: int | None = None,
    create_remote: bool = True,
) -> str:
    spec = metadata["config"]
    cluster = spec["cluster"]
    remote_dir = metadata["remote_run_dir"]
    task_indices = (
        list(range(len(spec["scan"]["values"])))
        if indices is None
        else list(map(int, indices))
    )
    maximum = (
        spec["scan"]["maximum_directional_passes"]
        if maximum_directional_passes is None
        else int(maximum_directional_passes)
    )

    if create_remote:
        remote_parent = str(Path(remote_dir).parent)
        remote_run(
            runner,
            cluster,
            [
                "sh",
                "-c",
                (
                    f"mkdir -p {shlex.quote(remote_parent)} && "
                    f"mkdir {shlex.quote(remote_dir)}"
                ),
            ],
        )
        _rsync_files(
            runner,
            cluster,
            [
                run_dir / "manifest.txt",
                run_dir / "config.normalized.json",
                run_dir / script_name,
                run_dir / "run.json",
            ],
            remote_dir,
        )
        _freeze_sources(runner, metadata)
    else:
        _rsync_files(
            runner,
            cluster,
            [run_dir / script_name],
            remote_dir,
        )

    request_id = _submission_request_id(
        metadata,
        kind=kind,
        indices=task_indices,
        maximum_directional_passes=maximum,
    )
    submitted_at = _utc_now()
    remote_record = {
        "kind": kind,
        "submitted_at": submitted_at,
        "indices": task_indices,
        "maximum_directional_passes": maximum,
        "protocol_fingerprint": metadata["protocol_fingerprint"],
        "source_id": metadata["source_id"],
    }
    submission = remote_run(
        runner,
        cluster,
        [
            cluster["python"],
            "-c",
            _REMOTE_SUBMIT_CODE,
            remote_dir,
            _slurm_command(cluster, "sbatch"),
            script_name,
            request_id,
            json.dumps(
                remote_record, sort_keys=True, separators=(",", ":")
            ),
        ],
    )
    if runner.dry_run:
        job_id = "DRY-RUN"
    else:
        try:
            remote_record = json.loads(submission.stdout.strip())
            job_id = _parse_job_id(str(remote_record["job_id"]))
            submitted_at = str(remote_record["submitted_at"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise WorkflowError(
                f"cannot parse remote submission journal: {submission.stdout!r}"
            ) from error
    record, is_new = _record_submission(
        metadata,
        job_id=job_id,
        kind=kind,
        indices=task_indices,
        maximum_directional_passes=maximum,
        request_id=request_id,
        submitted_at=submitted_at,
    )
    _save_run(run_dir, metadata)
    if is_new:
        _append_json_line(run_dir / "jobs.jsonl", record)
    if not runner.dry_run:
        _rsync_files(
            runner,
            cluster,
            [run_dir / "run.json", run_dir / "jobs.jsonl"],
            remote_dir,
        )
    return job_id


_RESULT_QUERY_CODE = r"""
import glob
import json
import os
import re
import sys

root = sys.argv[1]
rows = []
for directory in sorted(glob.glob(os.path.join(root, "task_*"))):
    if not os.path.isdir(directory):
        continue
    match = re.match(r"task_(\d+)_", os.path.basename(directory))
    if not match:
        continue
    path = os.path.join(directory, "result.json")
    snapshot = os.path.join(directory, "state.npz")
    row = {
        "task_index": int(match.group(1)),
        "task_dir": os.path.basename(directory),
        "has_result": os.path.isfile(path),
        "has_snapshot": os.path.isfile(snapshot),
    }
    try:
        if not row["has_result"]:
            rows.append(row)
            continue
        payload = json.load(open(path, encoding="utf-8"))
        result = payload.get("result", {})
        model = payload.get("model", {})
        nsites = float(model.get("nsites") or 1)
        timing = payload.get("timing_seconds", {})
        stderr = os.path.join(directory, "stderr.log")
        last_gain = result.get("last_directional_gain")
        last_cycle_gain = result.get("last_cycle_maximum_gain")
        last_gain_per_site = result.get("last_directional_gain_per_site")
        if last_gain_per_site is None and last_gain is not None:
            last_gain_per_site = float(last_gain) / nsites
        last_cycle_gain_per_site = result.get(
            "last_cycle_maximum_gain_per_site"
        )
        if last_cycle_gain_per_site is None and last_cycle_gain is not None:
            last_cycle_gain_per_site = float(last_cycle_gain) / nsites
        row.update(
            {
                "j2": model.get("j2"),
                "status": payload.get("status"),
                "converged": result.get("converged"),
                "stop_reason": result.get("stop_reason"),
                "energy": result.get("energy"),
                "energy_per_site": result.get("energy_per_site"),
                "passes": result.get("directional_passes_completed"),
                "next_sweep": result.get("next_directional_sweep"),
                "last_gain": last_gain,
                "last_gain_per_site": last_gain_per_site,
                "last_cycle_max_gain": last_cycle_gain,
                "last_cycle_max_gain_per_site": last_cycle_gain_per_site,
                "elapsed_seconds": timing.get("total"),
                "checkpoint_id": result.get("checkpoint_id"),
                "stderr_bytes": (
                    os.path.getsize(stderr) if os.path.isfile(stderr) else 0
                ),
            }
        )
        if row["has_snapshot"]:
            try:
                import numpy as np

                with np.load(snapshot, allow_pickle=False) as archive:
                    snapshot_id = str(
                        np.asarray(archive["checkpoint_id"]).item()
                    )
                    snapshot_energy = float(
                        np.asarray(archive["recorded_energy"]).item()
                    )
                row["snapshot_checkpoint_id"] = snapshot_id
                row["snapshot_energy"] = snapshot_energy
                row["checkpoint_pair_valid"] = bool(
                    snapshot_id == str(result.get("checkpoint_id"))
                    and abs(snapshot_energy - float(result["energy"])) <= 5e-10
                )
            except Exception as error:
                row["checkpoint_pair_valid"] = False
                row["snapshot_error"] = repr(error)
        else:
            row["checkpoint_pair_valid"] = False
        rows.append(row)
    except Exception as error:
        row["read_error"] = repr(error)
        row["checkpoint_pair_valid"] = False
        rows.append(row)
print(json.dumps(rows, separators=(",", ":")))
""".strip()


def _query_results(
    runner: CommandRunner, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    cluster = metadata["config"]["cluster"]
    process = remote_run(
        runner,
        cluster,
        [
            cluster["python"],
            "-c",
            _RESULT_QUERY_CODE,
            metadata["remote_run_dir"],
        ],
    )
    if runner.dry_run:
        return []
    try:
        payload = json.loads(process.stdout.strip() or "[]")
    except json.JSONDecodeError as error:
        raise WorkflowError(
            f"cannot parse remote result summary: {process.stdout!r}"
        ) from error
    if not isinstance(payload, list):
        raise WorkflowError("remote result summary is not an array")
    return payload


def parse_sacct(output: str, *, job_id: str) -> dict[int, dict[str, Any]]:
    """Parse allocation rows from ``sacct --parsable2`` output."""

    rows: dict[int, dict[str, Any]] = {}
    pattern = re.compile(rf"^{re.escape(str(job_id))}_(\d+)$")
    for line in output.splitlines():
        fields = line.rstrip().split("|")
        if len(fields) < 5:
            continue
        match = pattern.match(fields[0])
        if not match:
            continue
        state = fields[1].strip().upper().split()[0].split("+", 1)[0]
        rows[int(match.group(1))] = {
            "job_id": fields[0],
            "state": state,
            "elapsed": fields[2],
            "exit_code": fields[3],
            "max_rss": fields[4],
        }
    return rows


def parse_squeue(output: str, *, job_id: str) -> dict[int, dict[str, Any]]:
    """Parse expanded job-array rows from ``squeue --array``."""

    rows: dict[int, dict[str, Any]] = {}
    pattern = re.compile(rf"^{re.escape(str(job_id))}_(\d+)$")
    for line in output.splitlines():
        fields = line.rstrip().split("|", 3)
        if len(fields) < 4:
            continue
        match = pattern.match(fields[0])
        if not match:
            continue
        rows[int(match.group(1))] = {
            "job_id": fields[0],
            "state": fields[1].strip().upper().replace(" ", "_"),
            "elapsed": fields[2],
            "reason": fields[3],
            "exit_code": "",
            "max_rss": "",
        }
    return rows


def _query_slurm(
    runner: CommandRunner, metadata: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    cluster = metadata["config"]["cluster"]
    merged: dict[int, dict[str, Any]] = {}
    for submission in metadata.get("submissions", []):
        job_id = str(submission.get("job_id", ""))
        if not job_id.isdigit():
            continue
        process = remote_run(
            runner,
            cluster,
            [
                _slurm_command(cluster, "sacct"),
                "-X",
                "--array",
                "--noheader",
                "--parsable2",
                "-j",
                job_id,
                "--format=JobID,State,Elapsed,ExitCode,MaxRSS",
            ],
        )
        if not runner.dry_run:
            attempt_rows = parse_sacct(process.stdout, job_id=job_id)
            current = remote_run(
                runner,
                cluster,
                [
                    _slurm_command(cluster, "squeue"),
                    "--noheader",
                    "--array",
                    "-j",
                    job_id,
                    "-o",
                    "%i|%T|%M|%R",
                ],
                check=False,
            )
            attempt_rows.update(parse_squeue(current.stdout, job_id=job_id))
            for index, allocation in attempt_rows.items():
                allocation["attempt_job_id"] = job_id
                previous = merged.get(index)
                if (
                    previous
                    and previous.get("state") in ACTIVE_SLURM_STATES
                    and allocation.get("state") not in ACTIVE_SLURM_STATES
                ):
                    continue
                merged[index] = allocation
    return merged


def aggregate_status(
    metadata: dict[str, Any],
    *,
    results: Sequence[dict[str, Any]],
    slurm: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_index = {
        int(row["task_index"]): dict(row)
        for row in results
        if "task_index" in row
    }
    rows = []
    for index, j2 in enumerate(metadata["config"]["scan"]["values"]):
        result = by_index.get(index, {})
        allocation = slurm.get(index, {})
        slurm_state = allocation.get("state", "UNKNOWN")
        if result.get("converged"):
            workflow_state = "converged"
        elif slurm_state in ACTIVE_SLURM_STATES:
            workflow_state = slurm_state.lower()
        elif result.get("has_result") != result.get("has_snapshot"):
            workflow_state = "unsafe-checkpoint"
        elif result.get("checkpoint_pair_valid") is False:
            workflow_state = "unsafe-checkpoint"
        elif result.get("status") == "endpoint_rejected":
            workflow_state = "endpoint-rejected"
        elif result.get("status") == "maximum_passes":
            workflow_state = "max-passes"
        elif slurm_state in FAILED_SLURM_STATES:
            workflow_state = slurm_state.lower().replace("_", "-")
        elif slurm_state == "COMPLETED" and not result.get("has_result"):
            workflow_state = "incomplete"
        elif result:
            workflow_state = str(result.get("status") or slurm_state).lower()
        else:
            workflow_state = slurm_state.lower()
        rows.append(
            {
                "task_index": index,
                "j2": float(j2),
                "workflow_state": workflow_state,
                "slurm_state": slurm_state,
                **allocation,
                **result,
            }
        )
    return rows


def collect_status(
    runner: CommandRunner,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    return aggregate_status(
        metadata,
        results=_query_results(runner, metadata),
        slurm=_query_slurm(runner, metadata),
    )


def _format_number(value: Any, precision: int = 10) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def print_status(metadata: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    print(
        f"run={metadata['run_id']} remote={metadata['remote_run_dir']} "
        f"jobs={','.join(item['job_id'] for item in metadata['submissions'])}"
    )
    header = (
        "idx",
        "J2",
        "workflow",
        "Slurm",
        "passes",
        "E/site",
        "gain/N",
        "err",
    )
    print(
        f"{header[0]:>3} {header[1]:>5} {header[2]:>11} {header[3]:>11} "
        f"{header[4]:>6} {header[5]:>14} {header[6]:>11} {header[7]:>6}"
    )
    for row in rows:
        print(
            f"{row['task_index']:3d} {row['j2']:5.2f} "
            f"{row['workflow_state']:>11} {row['slurm_state']:>11} "
            f"{str(row.get('passes', '-')):>6} "
            f"{_format_number(row.get('energy_per_site'), 11):>14} "
            f"{_format_number(row.get('last_cycle_max_gain_per_site') or row.get('last_gain_per_site'), 4):>11} "
            f"{str(row.get('stderr_bytes', 0)):>6}"
        )
    counts: dict[str, int] = {}
    for row in rows:
        state = row["workflow_state"]
        counts[state] = counts.get(state, 0) + 1
    print("summary:", " ".join(f"{key}={counts[key]}" for key in sorted(counts)))


def _status_payload(
    metadata: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "run_id": metadata["run_id"],
        "queried_at": _utc_now(),
        "remote_run_dir": metadata["remote_run_dir"],
        "submissions": metadata["submissions"],
        "tasks": list(rows),
    }


def _write_status_artifacts(
    run_dir: Path, metadata: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> None:
    _atomic_json(run_dir / "status.json", _status_payload(metadata, rows))
    _write_summary_csv(run_dir / "summary.csv", rows)


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "task_index",
        "j2",
        "workflow_state",
        "slurm_state",
        "converged",
        "energy",
        "energy_per_site",
        "passes",
        "last_gain",
        "last_gain_per_site",
        "last_cycle_max_gain",
        "last_cycle_max_gain_per_site",
        "elapsed_seconds",
        "stderr_bytes",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_resume_indices(
    rows: Sequence[dict[str, Any]],
    *,
    maximum_directional_passes: int,
    requested: Sequence[int] | None = None,
    allow_endpoint_rejected: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    requested_set = None if requested is None else set(map(int, requested))
    selected = []
    capped = []
    endpoint_rejected = []
    for row in rows:
        index = int(row["task_index"])
        if requested_set is not None and index not in requested_set:
            continue
        if row.get("converged"):
            continue
        if str(row.get("slurm_state", "")).upper() in ACTIVE_SLURM_STATES:
            continue
        if (
            row.get("status") == "endpoint_rejected"
            and not allow_endpoint_rejected
        ):
            endpoint_rejected.append(index)
            continue
        passes = int(row.get("passes") or 0)
        if passes >= int(maximum_directional_passes):
            capped.append(index)
            continue
        selected.append(index)
    if requested_set is not None:
        known = {int(row["task_index"]) for row in rows}
        unknown = requested_set - known
        if unknown:
            raise WorkflowError(f"unknown task indices: {sorted(unknown)}")
    return selected, capped, endpoint_rejected


def unsafe_checkpoint_indices(rows: Sequence[dict[str, Any]]) -> list[int]:
    unsafe = []
    for row in rows:
        has_result = bool(row.get("has_result"))
        has_snapshot = bool(row.get("has_snapshot"))
        if has_result != has_snapshot:
            unsafe.append(int(row["task_index"]))
        elif has_result and row.get("checkpoint_pair_valid") is not True:
            unsafe.append(int(row["task_index"]))
    return unsafe


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    indices: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise WorkflowError(f"invalid index range: {token}")
            indices.update(range(start, stop + 1))
        else:
            indices.add(int(token))
    if not indices or min(indices) < 0:
        raise WorkflowError("indices must be nonnegative")
    return sorted(indices)


def _fetch_run(
    runner: CommandRunner,
    metadata: dict[str, Any],
    run_dir: Path,
    *,
    include_checkpoints: bool = False,
) -> Path:
    cluster = metadata["config"]["cluster"]
    destination = run_dir / "artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    argv = [
        "rsync",
        "-az",
        "--partial",
        "--exclude=source_checkpoints/",
        "--exclude=.task.lock",
    ]
    if not include_checkpoints:
        argv.append("--exclude=*.npz")
    argv.extend(
        [
            "-e",
            rsync_shell(cluster),
            f"{cluster['host']}:{metadata['remote_run_dir']}/",
            os.fspath(destination) + "/",
        ]
    )
    runner.run(argv)
    return destination


def _local_result_rows(run_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "artifacts").glob("task_*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = payload.get("result", {})
            timing = payload.get("timing_seconds", {})
            match = re.match(r"task_(\d+)_", path.parent.name)
            if not match or not isinstance(result, dict) or not isinstance(
                timing, dict
            ):
                continue
            model = payload["model"]
            j2 = float(model["j2"])
            nsites = float(model.get("nsites") or 1)
            last_gain = result.get("last_directional_gain")
            last_cycle_gain = result.get("last_cycle_maximum_gain")
            last_gain_per_site = result.get("last_directional_gain_per_site")
            if last_gain_per_site is None and last_gain is not None:
                last_gain_per_site = float(last_gain) / nsites
            last_cycle_gain_per_site = result.get(
                "last_cycle_maximum_gain_per_site"
            )
            if last_cycle_gain_per_site is None and last_cycle_gain is not None:
                last_cycle_gain_per_site = float(last_cycle_gain) / nsites
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            continue
        rows.append(
            {
                "task_index": int(match.group(1)),
                "j2": j2,
                "workflow_state": (
                    "converged"
                    if result.get("converged")
                    else str(payload.get("status", "unknown"))
                ),
                "slurm_state": "FETCHED",
                "converged": bool(result.get("converged")),
                "energy": result.get("energy"),
                "energy_per_site": result.get("energy_per_site"),
                "passes": result.get("directional_passes_completed"),
                "last_gain": last_gain,
                "last_gain_per_site": last_gain_per_site,
                "last_cycle_max_gain": last_cycle_gain,
                "last_cycle_max_gain_per_site": last_cycle_gain_per_site,
                "elapsed_seconds": timing.get("total"),
                "stderr_bytes": (
                    (path.parent / "stderr.log").stat().st_size
                    if (path.parent / "stderr.log").is_file()
                    else 0
                ),
            }
        )
    if not rows:
        raise WorkflowError("no valid fetched results found; run fetch first")
    return rows


def _plot_rows(
    run_dir: Path,
    rows: Sequence[dict[str, Any]],
    *,
    gain_tolerance: float,
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise WorkflowError(
            "plot requires matplotlib; install the pyqed plot extra"
        ) from error

    ordered = sorted(
        (row for row in rows if row.get("energy_per_site") is not None),
        key=lambda row: row["j2"],
    )
    if not ordered:
        raise WorkflowError("no finite energies are available to plot")
    x = [row["j2"] for row in ordered]
    energy = [row["energy_per_site"] for row in ordered]
    gains = [
        row.get("last_cycle_max_gain_per_site") or row.get("last_gain_per_site")
        for row in ordered
    ]
    converged = [bool(row.get("converged")) for row in ordered]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(5.2, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    axes[0].plot(x, energy, color="#315A9A", linewidth=1.4, zorder=1)
    for state in (True, False):
        selected = [index for index, flag in enumerate(converged) if flag is state]
        axes[0].scatter(
            [x[index] for index in selected],
            [energy[index] for index in selected],
            s=34,
            facecolors="#315A9A" if state else "white",
            edgecolors="#315A9A",
            linewidths=1.1,
            label="converged" if state else "not converged",
            zorder=2,
        )
    axes[0].set_ylabel(r"$E/N$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2, linewidth=0.6)

    gain_points = [
        (position, float(gain))
        for position, gain in zip(x, gains)
        if gain is not None and float(gain) > 0.0
    ]
    if gain_points:
        axes[1].semilogy(
            [item[0] for item in gain_points],
            [item[1] for item in gain_points],
            "o-",
            color="#A64B35",
            markersize=4,
        )
    if gain_tolerance > 0.0:
        axes[1].axhline(
            gain_tolerance, color="0.35", linestyle="--", linewidth=0.9
        )
    axes[1].set_xlabel(r"$J_2/J_1$")
    axes[1].set_ylabel(r"last gain$/N$")
    axes[1].grid(alpha=0.2, linewidth=0.6)
    figure.tight_layout()
    png = run_dir / "energy_scan.png"
    pdf = run_dir / "energy_scan.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _require_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if not args.config:
        raise WorkflowError("--config is required for this command")
    spec, path = load_config(args.config)
    if args.runs_root:
        spec = copy.deepcopy(spec)
        spec["local"]["runs_root"] = os.fspath(
            Path(args.runs_root).expanduser().resolve()
        )
    return spec, path


def command_connect(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec, _ = _require_config(args)
    Path(_control_path()).parent.mkdir(parents=True, exist_ok=True)
    cluster = spec["cluster"]
    runner.run(
        [
            "ssh",
            *ssh_options(cluster),
            "-MNf",
            cluster["host"],
        ]
    )
    print(f"persistent SSH connection ready for {cluster['host']}")
    return 0


def command_check(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec, _ = _require_config(args)
    cluster = spec["cluster"]
    if cluster.get("slurm_bin"):
        slurm_check = " && ".join(
            f"test -x {shlex.quote(_slurm_command(cluster, name))}"
            for name in ("sbatch", "squeue", "sacct")
        )
    else:
        slurm_check = "command -v sbatch squeue sacct >/dev/null"
    runner.run(
        [
            "ssh",
            *ssh_options(cluster),
            "-O",
            "check",
            cluster["host"],
        ]
    )
    process = remote_run(
        runner,
        cluster,
        [
            "sh",
            "-c",
            (
                "command -v rsync flock >/dev/null && "
                f"{slurm_check} && "
                f"test -x {shlex.quote(cluster['python'])} && "
                f"test -d {shlex.quote(cluster['remote_source_root'])} && "
                f"{shlex.quote(cluster['python'])} --version && "
                f"{shlex.quote(_slurm_command(cluster, 'sbatch'))} --version"
            ),
        ],
    )
    if process.stdout:
        print(process.stdout.strip())
    return 0


def command_disconnect(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec, _ = _require_config(args)
    cluster = spec["cluster"]
    runner.run(
        [
            "ssh",
            *ssh_options(cluster),
            "-O",
            "exit",
            cluster["host"],
        ],
        check=False,
    )
    print(f"closed persistent SSH connection for {cluster['host']}")
    return 0


_SOURCE_STATUS_CODE = r"""
import json
import os
import sys

target, expected = sys.argv[1:]
marker = os.path.join(target, ".letta_source_manifest.json")
if not os.path.exists(target):
    print(json.dumps({"state": "missing"}))
elif not os.path.isfile(marker):
    print(json.dumps({"state": "unverified"}))
else:
    payload = json.load(open(marker, encoding="utf-8"))
    print(
        json.dumps(
            {
                "state": (
                    "ready"
                    if payload.get("source_id") == expected
                    else "mismatch"
                ),
                "actual": payload.get("source_id"),
            }
        )
    )
""".strip()

_VERIFY_SOURCE_CODE = r"""
import hashlib
import json
import os
import sys

root, marker_path = sys.argv[1:]
payload = json.load(open(marker_path, encoding="utf-8"))
records = sorted(payload["files"], key=lambda item: item["path"])
combined = hashlib.sha256()
for record in records:
    relative = record["path"]
    path = os.path.join(root, *relative.split("/"))
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != record["sha256"]:
        raise SystemExit(f"source hash mismatch: {relative}")
    encoded = relative.encode("utf-8")
    combined.update(len(encoded).to_bytes(8, "big"))
    combined.update(encoded)
    combined.update(bytes.fromhex(actual))
expected = "sha256:" + combined.hexdigest()
if payload["source_id"] != expected:
    raise SystemExit(
        f"source ID mismatch: {payload['source_id']} != {expected}"
    )
print(json.dumps({"verified": len(records), "source_id": expected}))
""".strip()


def _sync_source_bundle(
    runner: CommandRunner, spec: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Upload an immutable, content-addressed source tree and return its spec."""

    cluster = spec["cluster"]
    local = spec["local"]
    source = Path(local["source_root"])
    files = source_files(spec)
    records = [
        {
            "path": relative.as_posix(),
            "sha256": _sha256_file(source / relative),
        }
        for relative in files
    ]
    digest = _source_records_id(records)
    source_id = f"sha256:{digest}"
    target = f"{cluster['remote_sources_root']}/{digest}"
    cache = Path(local["runs_root"]) / ".sources"
    cache.mkdir(parents=True, exist_ok=True)
    files_path = cache / f"{digest}.files"
    files_path.write_text(
        "".join(path.as_posix() + "\n" for path in files), encoding="utf-8"
    )
    marker = {
        "schema_version": 1,
        "source_id": source_id,
        "created_at": _utc_now(),
        "file_count": len(files),
        "controller_version": CONTROLLER_VERSION,
        "git": _git_provenance(Path(local["repository"])),
        "files": records,
    }
    marker_dir = cache / digest
    marker_dir.mkdir(exist_ok=True)
    marker_path = marker_dir / ".letta_source_manifest.json"
    _atomic_json(marker_path, marker)

    status_process = remote_run(
        runner,
        cluster,
        [cluster["python"], "-c", _SOURCE_STATUS_CODE, target, source_id],
    )
    status = (
        {"state": "missing"}
        if runner.dry_run
        else json.loads(status_process.stdout.strip())
    )
    if status.get("state") == "ready":
        remote_run(
            runner,
            cluster,
            [
                cluster["python"],
                "-c",
                _VERIFY_SOURCE_CODE,
                target,
                f"{target}/.letta_source_manifest.json",
            ],
        )
    elif status.get("state") != "missing":
        raise WorkflowError(
            f"refusing unverified source directory {target}: {status}"
        )
    else:
        staging = (
            f"{cluster['remote_sources_root']}/"
            f".upload-{digest[:16]}-{os.getpid()}-{int(time.time())}"
        )
        remote_run(
            runner,
            cluster,
            [
                "sh",
                "-c",
                (
                    f"mkdir -p {shlex.quote(cluster['remote_sources_root'])} && "
                    f"mkdir {shlex.quote(staging)}"
                ),
            ],
        )
        runner.run(
            [
                "rsync",
                "-az",
                "--partial",
                "--relative",
                f"--files-from={files_path}",
                "-e",
                rsync_shell(cluster),
                os.fspath(source) + "/",
                f"{cluster['host']}:{staging}/",
            ]
        )
        _rsync_files(runner, cluster, [marker_path], staging)
        remote_run(
            runner,
            cluster,
            [
                cluster["python"],
                "-c",
                _VERIFY_SOURCE_CODE,
                staging,
                f"{staging}/.letta_source_manifest.json",
            ],
        )
        if local["build_command"]:
            remote_run(
                runner,
                cluster,
                [
                    "env",
                    f"PYQED_ROOT={staging}",
                    f"PYTHON={cluster['python']}",
                    "sh",
                    "-lc",
                    local["build_command"],
                ],
                capture=False,
            )
        remote_run(
            runner,
            cluster,
            [
                cluster["python"],
                "-c",
                _VERIFY_SOURCE_CODE,
                staging,
                f"{staging}/.letta_source_manifest.json",
            ],
        )
        remote_run(
            runner,
            cluster,
            [
                "sh",
                "-c",
                (
                    f"test ! -e {shlex.quote(target)} && "
                    f"mv {shlex.quote(staging)} {shlex.quote(target)}"
                ),
            ],
        )

    resolved = copy.deepcopy(spec)
    resolved["cluster"]["remote_source_root"] = target
    resolved["cluster"]["source_id"] = source_id
    verification = remote_run(
        runner,
        cluster,
        [
            cluster["python"],
            "-c",
            _SOURCE_STATUS_CODE,
            target,
            source_id,
        ],
    )
    if not runner.dry_run:
        verified = json.loads(verification.stdout.strip())
        if verified.get("state") != "ready":
            raise WorkflowError(
                f"source upload did not verify at {target}: {verified}"
            )
        remote_run(
            runner,
            cluster,
            [
                cluster["python"],
                "-c",
                _VERIFY_SOURCE_CODE,
                target,
                f"{target}/.letta_source_manifest.json",
            ],
        )
    return resolved, source_id


def command_sync(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec, _ = _require_config(args)
    resolved, source_id = _sync_source_bundle(runner, spec)
    print(
        f"source {source_id} ready at "
        f"{resolved['cluster']['host']}:"
        f"{resolved['cluster']['remote_source_root']}"
    )
    return 0


def command_submit(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec, config_path = _require_config(args)
    if args.sync or spec["local"]["sync_source"]:
        spec, source_id = _sync_source_bundle(runner, spec)
        print(
            f"using immutable source {source_id} at "
            f"{spec['cluster']['remote_source_root']}"
        )
    run_id = args.run_id or _default_run_id(spec["scan"]["name"])
    metadata, run_dir = _prepare_run(
        spec,
        config_path=config_path,
        run_id=run_id,
    )
    job_id = _submit_prepared(runner, metadata, run_dir)
    print(f"submitted run {run_id}: job {job_id}")
    print(f"local:  {run_dir}")
    print(f"remote: {metadata['remote_run_dir']}")
    return 0


def command_adopt(args: argparse.Namespace, runner: CommandRunner) -> int:
    del runner
    spec, config_path = _require_config(args)
    run_id = args.run_id or _slug(Path(args.remote_run_dir).name)
    metadata, run_dir = _prepare_run(
        spec,
        config_path=config_path,
        run_id=run_id,
        remote_run_dir=args.remote_run_dir,
        existing_job_id=args.job_id,
    )
    metadata["execution"]["source_run_root"] = spec["scan"].get("source_run_root")
    metadata["execution"]["frozen_from"] = None
    (run_dir / "submit.sbatch").write_text(
        render_sbatch(
            spec,
            remote_run_dir=metadata["remote_run_dir"],
            source_run_root=metadata["execution"]["source_run_root"],
        ),
        encoding="utf-8",
    )
    _save_run(run_dir, metadata)
    _append_json_line(run_dir / "jobs.jsonl", metadata["submissions"][0])
    print(f"adopted remote run {metadata['remote_run_dir']} as {run_id}")
    return 0


def command_status(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec = None
    if args.config:
        spec, _ = load_config(args.config)
    runs_root = _resolve_runs_root(spec, args.runs_root)
    run_dir = resolve_run_dir(args.run_id, runs_root=runs_root)
    metadata = _load_run(run_dir)
    interval = max(0.0, float(args.watch))
    while True:
        rows = collect_status(runner, metadata)
        _write_status_artifacts(run_dir, metadata, rows)
        if args.json:
            print(json.dumps(_status_payload(metadata, rows), indent=2))
        else:
            print_status(metadata, rows)
        if interval <= 0.0 or not any(
            row["slurm_state"] in ACTIVE_SLURM_STATES for row in rows
        ):
            break
        time.sleep(interval)
        if not args.json:
            print()
    return 0


def command_resume(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec_for_root = None
    if args.config:
        spec_for_root, _ = load_config(args.config)
    runs_root = _resolve_runs_root(spec_for_root, args.runs_root)
    run_dir = resolve_run_dir(args.run_id, runs_root=runs_root)
    with _run_mutation_lock(run_dir):
        return _resume_locked(args, runner, run_dir)


def _resume_locked(
    args: argparse.Namespace,
    runner: CommandRunner,
    run_dir: Path,
) -> int:
    metadata = _load_run(run_dir)
    expected_fingerprint = numerical_protocol_fingerprint(metadata["config"])
    if metadata.get("protocol_fingerprint") != expected_fingerprint:
        raise WorkflowError("run metadata numerical protocol was modified")
    if any(
        attempt.get("protocol_fingerprint") != expected_fingerprint
        or attempt.get("source_id") != metadata.get("source_id")
        for attempt in metadata.get("submissions", [])
    ):
        raise WorkflowError("prior attempts do not match the immutable run spec")
    rows = collect_status(runner, metadata)
    unsafe = unsafe_checkpoint_indices(rows)
    if unsafe:
        raise WorkflowError(
            f"refusing resume with incomplete or inconsistent checkpoints: {unsafe}"
        )
    current_max = max(
        [
            metadata["config"]["scan"]["maximum_directional_passes"],
            *[
                int(item["maximum_directional_passes"])
                for item in metadata.get("submissions", [])
            ],
        ]
    )
    maximum = current_max if args.maximum_directional_passes is None else int(
        args.maximum_directional_passes
    )
    if maximum < current_max:
        raise WorkflowError(
            f"resume cannot lower the pass cap ({maximum} < {current_max})"
        )
    requested = _parse_indices(args.indices)
    indices, capped, endpoint_rejected = select_resume_indices(
        rows,
        maximum_directional_passes=maximum,
        requested=requested,
        allow_endpoint_rejected=args.allow_endpoint_rejected,
    )
    if not indices:
        message = "no inactive unconverged tasks are eligible for resume"
        if capped:
            message += (
                f"; capped tasks {capped} require --maximum-directional-passes "
                f"greater than {maximum}"
            )
        if endpoint_rejected:
            message += (
                f"; endpoint-rejected tasks {endpoint_rejected} require "
                "--allow-endpoint-rejected"
            )
        raise WorkflowError(message)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    script_name = f"resume-{stamp}.sbatch"
    (run_dir / script_name).write_text(
        render_sbatch(
            metadata["config"],
            remote_run_dir=metadata["remote_run_dir"],
            source_run_root=metadata["execution"].get("source_run_root"),
            indices=indices,
            maximum_directional_passes=maximum,
        ),
        encoding="utf-8",
    )
    job_id = _submit_prepared(
        runner,
        metadata,
        run_dir,
        kind="resume",
        script_name=script_name,
        indices=indices,
        maximum_directional_passes=maximum,
        create_remote=False,
    )
    print(f"resumed tasks {indices}: job {job_id}")
    if capped:
        print(f"not submitted at cap: {capped}")
    if endpoint_rejected:
        print(f"not submitted after rejected endpoint: {endpoint_rejected}")
    return 0


def command_refine(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec_for_root = None
    if args.config:
        spec_for_root, _ = load_config(args.config)
    runs_root = _resolve_runs_root(spec_for_root, args.runs_root)
    parent_dir = resolve_run_dir(args.run_id, runs_root=runs_root)
    parent = _load_run(parent_dir)
    if parent.get("protocol_fingerprint") != numerical_protocol_fingerprint(
        parent["config"]
    ):
        raise WorkflowError("parent run metadata numerical protocol was modified")
    rows = collect_status(runner, parent)
    active = [
        row["task_index"]
        for row in rows
        if row["slurm_state"] in ACTIVE_SLURM_STATES
    ]
    missing = [
        row["task_index"]
        for row in rows
        if row.get("checkpoint_id") is None
        or row.get("checkpoint_pair_valid") is not True
    ]
    if active:
        raise WorkflowError(f"parent tasks are still active: {active}")
    if missing:
        raise WorkflowError(f"parent tasks lack verified checkpoints: {missing}")
    rejected = [
        row["task_index"]
        for row in rows
        if row.get("status") == "endpoint_rejected"
    ]
    if rejected and not args.allow_endpoint_rejected:
        raise WorkflowError(
            f"parent tasks stopped after rejected endpoints: {rejected}; "
            "use --allow-endpoint-rejected to continue from their rolled-back "
            "checkpoints"
        )

    tolerance = float(args.gain_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise WorkflowError("gain tolerance must be finite and nonnegative")
    parent_tolerance = float(parent["config"]["scan"]["gain_tolerance"])
    if tolerance >= parent_tolerance:
        raise WorkflowError(
            "refine requires a stricter gain tolerance; use resume to raise "
            "only the pass cap"
        )
    maximum = _positive_int(
        args.maximum_directional_passes, "maximum_directional_passes"
    )
    child_spec = copy.deepcopy(parent["config"])
    child_spec["scan"]["gain_tolerance"] = tolerance
    child_spec["scan"]["maximum_directional_passes"] = maximum
    child_spec["scan"]["source_run_root"] = parent["remote_run_dir"]
    child_spec["scan"]["seed_result"] = None
    child_spec["scan"]["seed_snapshot"] = None
    child_spec["scan"]["freeze_source"] = True
    child_spec["scan"]["name"] = _slug(
        args.name
        or f"{parent['config']['scan']['name']}-tol{tolerance:.0e}"
    )
    child_spec["local"]["runs_root"] = os.fspath(runs_root)
    run_id = args.new_run_id or _default_run_id(child_spec["scan"]["name"])
    child, child_dir = _prepare_run(
        child_spec,
        config_path=None,
        run_id=run_id,
        parent_run_id=parent["run_id"],
    )
    job_id = _submit_prepared(runner, child, child_dir, kind="refine")
    print(f"submitted refinement {run_id}: job {job_id}")
    print(f"frozen parent checkpoints: {child['execution']['source_run_root']}")
    return 0


def command_fetch(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec = None
    if args.config:
        spec, _ = load_config(args.config)
    runs_root = _resolve_runs_root(spec, args.runs_root)
    run_dir = resolve_run_dir(args.run_id, runs_root=runs_root)
    metadata = _load_run(run_dir)
    destination = _fetch_run(
        runner,
        metadata,
        run_dir,
        include_checkpoints=args.checkpoints,
    )
    print(f"fetched {metadata['remote_run_dir']} -> {destination}")
    return 0


def command_plot(args: argparse.Namespace, runner: CommandRunner) -> int:
    spec = None
    if args.config:
        spec, _ = load_config(args.config)
    runs_root = _resolve_runs_root(spec, args.runs_root)
    run_dir = resolve_run_dir(args.run_id, runs_root=runs_root)
    metadata = _load_run(run_dir)
    if args.fetch:
        _fetch_run(runner, metadata, run_dir)
    rows = _local_result_rows(run_dir, metadata)
    _write_summary_csv(run_dir / "summary.csv", rows)
    png, pdf = _plot_rows(
        run_dir,
        rows,
        gain_tolerance=float(metadata["config"]["scan"]["gain_tolerance"]),
    )
    print(png)
    print(pdf)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON or TOML cluster/scan configuration")
    parser.add_argument(
        "--runs-root",
        help="override the local run registry (default from config or .letta_hpc/runs)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, function in (
        ("connect", command_connect),
        ("check", command_check),
        ("disconnect", command_disconnect),
        ("sync", command_sync),
    ):
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(function=function)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--run-id")
    submit.add_argument("--sync", action="store_true")
    submit.set_defaults(function=command_submit)

    adopt = subparsers.add_parser("adopt")
    adopt.add_argument("job_id")
    adopt.add_argument("remote_run_dir")
    adopt.add_argument("--run-id")
    adopt.set_defaults(function=command_adopt)

    status = subparsers.add_parser("status")
    status.add_argument("run_id", nargs="?")
    status.add_argument("--watch", type=float, default=0.0, metavar="SECONDS")
    status.add_argument("--json", action="store_true")
    status.set_defaults(function=command_status)

    resume = subparsers.add_parser("resume")
    resume.add_argument("run_id", nargs="?")
    resume.add_argument("--indices", help="comma-separated indices/ranges")
    resume.add_argument("--maximum-directional-passes", type=int)
    resume.add_argument("--allow-endpoint-rejected", action="store_true")
    resume.set_defaults(function=command_resume)

    refine = subparsers.add_parser("refine")
    refine.add_argument("run_id", nargs="?")
    refine.add_argument("--gain-tolerance", type=float, required=True)
    refine.add_argument("--maximum-directional-passes", type=int, required=True)
    refine.add_argument("--new-run-id")
    refine.add_argument("--name")
    refine.add_argument("--allow-endpoint-rejected", action="store_true")
    refine.set_defaults(function=command_refine)

    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("run_id", nargs="?")
    fetch.add_argument("--checkpoints", action="store_true")
    fetch.set_defaults(function=command_fetch)

    plot = subparsers.add_parser("plot")
    plot.add_argument("run_id", nargs="?")
    plot.add_argument("--fetch", action="store_true")
    plot.set_defaults(function=command_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runner = CommandRunner(dry_run=args.dry_run, verbose=args.verbose)
    try:
        return int(args.function(args, runner))
    except WorkflowError as error:
        parser.error(str(error))
    except subprocess.CalledProcessError as error:
        command = shlex.join(map(str, error.cmd))
        detail = (error.stderr or error.stdout or "").strip()
        parser.error(
            f"command failed ({error.returncode}): {command}"
            + (f"\n{detail}" if detail else "")
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
