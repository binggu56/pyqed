#!/usr/bin/env python3
"""Stage and submit the split 50 fs phenol GP/NGP Slurm calculation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess

from tools.letta_hpc import load_config, rsync_shell, ssh_argv


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/phenol_5d_production"
SOURCE_PATHS = (
    "pyqed",
    "examples/namd/phenol_gp_cluster_portabilize.py",
    "examples/namd/phenol_gp_ngp_merge.py",
    "examples/namd/phenol_sa_casscf_3d_ftt_ttldr.py",
    "examples/namd/phenol_sa_casscf_3d_gp_control.py",
    "examples/namd/phenol_sa_casscf_5d_ftt_ttldr.py",
    "examples/namd/phenol_sa_casscf_5d_gp_control.py",
    "examples/namd/phenol_sa_casscf_5d_quasibound.py",
    "examples/namd/phenol_sa_casscf_paths.py",
    "cluster",
    "pyproject.toml",
    "setup.py",
)
DATA_PATHS = (
    "model/mace_y_probability_expanded_final_polished/phenol_sa6_5d_mace_y.pt",
    "model/radial_correction_probability_expanded/phenol_sa6_5d_radial_delta.npz",
    "states/s1_origin_5d_quasibound_localwell_h3_corrected/phenol_sa_casscf_5d_s1_quasibound.npz",
    "cache/quasibound_scalar_keo_65x21x23x21x17",
    "cache/gp_ngp_s1_full_overlap_fields_65x21x23x21x17",
    "cache/gp_ngp_s1_full_overlap_base_qualified_65x21x23x21x17",
    "cache/gp_s1_full_overlap_operator_r16_65x21x23x21x17",
    "cache/ngp_s1_full_overlap_operator_r16_65x21x23x21x17",
)
LOCAL_GP_CHECKPOINT = (
    DATASET
    / "dynamics/gp_ngp_full_overlap_r16_hybrid_state32_50fs/gp_checkpoint.npz"
)


def _run(command, *, capture=False, dry_run=False):
    print("+", shlex.join([str(item) for item in command]), flush=True)
    if dry_run:
        return ""
    completed = subprocess.run(
        [str(item) for item in command],
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def _remote(cluster, command, *, capture=False, dry_run=False):
    return _run(
        ssh_argv(cluster, shlex.join([str(item) for item in command])),
        capture=capture,
        dry_run=dry_run,
    )


def _rsync(cluster, source, destination, *, source_code=False, dry_run=False):
    excludes = [
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=*.so",
        "--exclude=.DS_Store",
    ]
    if source_code:
        excludes.extend(
            [
                "--exclude=*.c",
                "--exclude=*.npy",
                "--exclude=*.npz",
                "--exclude=*.sqlite",
                "--exclude=*.log",
                "--exclude=*.out",
                "--exclude=*.pdf",
                "--exclude=*.png",
            ]
        )
    return _run(
        [
            "rsync",
            "-a",
            "--partial",
            *excludes,
            "-e",
            rsync_shell(cluster),
            str(source),
            f"{cluster['host']}:{destination}",
        ],
        dry_run=dry_run,
    )


def submit(args):
    spec, _config = load_config(args.config)
    cluster = spec["cluster"]
    run_id = args.run_id or datetime.now().strftime("phenol-gp-ngp-%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id may contain only letters, digits, '-' and '_'")
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", args.remote_root):
        raise ValueError("remote-root must be a safe absolute path")
    run_root = f"{args.remote_root.rstrip('/')}/{run_id}"
    source_root = f"{run_root}/source"
    dataset_root = f"{run_root}/dataset/phenol_5d_production"
    output_root = f"{run_root}/output"
    directories = [
        source_root,
        f"{source_root}/examples/namd",
        dataset_root,
        f"{output_root}/gp",
        f"{output_root}/ngp",
        f"{run_root}/logs",
        f"{run_root}/tmp",
        f"{run_root}/build/mps-cpp",
        f"{run_root}/deps",
    ]
    for relative in DATA_PATHS:
        directories.append(str(Path(dataset_root) / Path(relative).parent))
    _remote(cluster, ["mkdir", "-p", *directories], dry_run=args.dry_run)

    for relative in SOURCE_PATHS:
        parent = Path(relative).parent
        destination = source_root if parent == Path(".") else f"{source_root}/{parent}"
        _rsync(
            cluster,
            ROOT / relative,
            f"{destination}/",
            source_code=True,
            dry_run=args.dry_run,
        )
    for relative in DATA_PATHS:
        source = DATASET / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = str(Path(dataset_root) / Path(relative).parent) + "/"
        _rsync(cluster, source, destination, dry_run=args.dry_run)
    if not LOCAL_GP_CHECKPOINT.is_file():
        raise FileNotFoundError(LOCAL_GP_CHECKPOINT)
    _rsync(
        cluster,
        LOCAL_GP_CHECKPOINT,
        f"{output_root}/gp/",
        dry_run=args.dry_run,
    )

    python = cluster["python"]
    common = {
        "checkpoint": f"{dataset_root}/{DATA_PATHS[0]}",
        "radial-correction": f"{dataset_root}/{DATA_PATHS[1]}",
        "initial-state": f"{dataset_root}/{DATA_PATHS[2]}",
        "keo-cache": f"{dataset_root}/{DATA_PATHS[3]}",
        "field-cache": f"{dataset_root}/{DATA_PATHS[4]}",
        "residual-cache": f"{dataset_root}/{DATA_PATHS[5]}",
        "gp-operator-cache": f"{dataset_root}/{DATA_PATHS[6]}",
        "ngp-operator-cache": f"{dataset_root}/{DATA_PATHS[7]}",
    }
    portabilize = [
        "env",
        f"PYTHONPATH={source_root}",
        python,
        f"{source_root}/examples/namd/phenol_gp_cluster_portabilize.py",
    ]
    for key, value in common.items():
        portabilize.extend([f"--{key}", value])
    portabilize.extend(
        [
            "--branch-checkpoint",
            f"{output_root}/gp/gp_checkpoint.npz",
            "--manifest",
            f"{run_root}/cache_manifest.json",
        ]
    )
    _remote(cluster, portabilize, dry_run=args.dry_run)

    _remote(
        cluster,
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            "--target",
            f"{run_root}/deps",
            "pybind11",
        ],
        dry_run=args.dry_run,
    )

    preflight = (
        "from pyqed.mps import tdvp_cpp as m; "
        "print('native TDVP:', m.CPP_TDVP_AVAILABLE, 'BLAS:', m.CPP_TDVP_HAS_BLAS); "
        "assert m.CPP_TDVP_AVAILABLE and m.CPP_TDVP_HAS_BLAS, m.CPP_TDVP_BUILD_ERROR"
    )
    openblas_root = "/soft/mathlib/OpenBLAS/OpenBLAS-0.3.30_install"
    _remote(
        cluster,
        [
            "env",
            f"PYTHONPATH={run_root}/deps:{source_root}",
            f"PYQED_MPS_CPP_BUILD={run_root}/build/mps-cpp",
            "CXX=/usr/bin/g++",
            (
                "PYQED_TDVP_BLAS_FLAGS="
                f"-I{openblas_root}/include -L{openblas_root}/lib "
                f"-Wl,-rpath,{openblas_root}/lib -lopenblas"
            ),
            f"LD_LIBRARY_PATH={openblas_root}/lib",
            python,
            "-c",
            preflight,
        ],
        dry_run=args.dry_run,
    )

    slurm_bin = cluster.get("slurm_bin", "")
    sbatch = f"{slurm_bin}/sbatch" if slurm_bin else "sbatch"
    export = ",".join(
        (
            "ALL",
            f"PHENOL_RUN_ROOT={run_root}",
            f"PYQED_SOURCE_ROOT={source_root}",
            f"PYQED_PYTHON={python}",
        )
    )
    job_id = _remote(
        cluster,
        [
            sbatch,
            "--parsable",
            f"--output={run_root}/logs/%x-%A_%a.out",
            f"--error={run_root}/logs/%x-%A_%a.err",
            f"--export={export}",
            f"{source_root}/cluster/slurm/phenol_gp_ngp_5d_50fs.sbatch",
        ],
        capture=True,
        dry_run=args.dry_run,
    )
    record = {
        "run_id": run_id,
        "remote_run_root": run_root,
        "job_id": job_id or None,
        "host": cluster["host"],
        "local_fallback_checkpoint": str(LOCAL_GP_CHECKPOINT),
    }
    if not args.dry_run:
        destination = DATASET / "dynamics" / "gp_ngp_full_overlap_r16_hybrid_state32_50fs" / "cluster_submission.json"
        destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="examples/mps/cluster/letta_hpc_mps_6x6.toml",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--remote-root",
        default="/share/home/gubingLab/gubing/phenol_dynamics/runs",
    )
    parser.add_argument("--dry-run", action="store_true")
    submit(parser.parse_args())


if __name__ == "__main__":
    main()
