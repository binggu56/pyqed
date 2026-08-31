#!/usr/bin/env python3
"""Run the production phenol 5D trajectory in recoverable cumulative segments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from examples.namd.phenol_sa_casscf_paths import (
    DEFAULT_PHENOL_5D_DISTILLATION_CACHE,
    DEFAULT_PHENOL_5D_OPERATOR_CACHE,
    PHENOL_5D_PRODUCTION,
    PROJECT_ROOT,
)


RESULT_NAME = "phenol_sa_casscf_5d_ftt_ttldr.npz"


def _complete(directory, expected_time):
    summary_path = directory / "summary.json"
    result_path = directory / RESULT_NAME
    if not summary_path.is_file() or not result_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    actual = float(summary["dynamics"]["time_fs"])
    return abs(actual - expected_time) <= 1.0e-9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-fs", type=float, default=200.0)
    parser.add_argument("--segment-fs", type=float, default=25.0)
    parser.add_argument("--dt-fs", type=float, default=0.05)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PHENOL_5D_PRODUCTION / "dynamics" / "200fs",
    )
    parser.add_argument(
        "--distilled-cache",
        type=Path,
        default=DEFAULT_PHENOL_5D_DISTILLATION_CACHE,
    )
    parser.add_argument(
        "--operator-cache",
        type=Path,
        default=DEFAULT_PHENOL_5D_OPERATOR_CACHE,
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.target_fs <= 0.0 or args.segment_fs <= 0.0 or args.dt_fs <= 0.0:
        raise ValueError("target, segment, and timestep must be positive")
    if args.interval < 1:
        raise ValueError("interval must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "target_fs": args.target_fs,
        "segment_fs": args.segment_fs,
        "dt_fs": args.dt_fs,
        "interval": args.interval,
        "grid_shape": [65, 9, 7, 11, 9],
        "state_rank": 24,
        "operator_rank": 64,
        "krylov_dim": 8,
        "krylov_tolerance": 1.0e-10,
        "workers": 8,
        "integrator": "tdvp",
        "cap": {"start_angstrom": 2.45, "strength_hartree": 0.02, "order": 4},
        "distilled_cache": str(args.distilled_cache.resolve()),
        "operator_cache": str(args.operator_cache.resolve()),
    }
    config_path = args.output_root / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(f"existing run configuration differs: {config_path}")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    driver_script = PROJECT_ROOT / "examples" / "namd" / "phenol_sa_casscf_5d_ftt_ttldr.py"
    previous = None
    current_time = 0.0
    while current_time < args.target_fs - 1.0e-12:
        duration = min(args.segment_fs, args.target_fs - current_time)
        steps = int(round(duration / args.dt_fs))
        if abs(steps * args.dt_fs - duration) > 1.0e-12:
            raise ValueError("each segment duration must be an integer number of timesteps")
        end_time = current_time + duration
        directory = args.output_root / f"{int(round(end_time)):03d}fs"
        result = directory / RESULT_NAME
        if _complete(directory, end_time):
            print(f"[segment] reuse completed {end_time:.2f} fs: {directory}", flush=True)
            previous = result
            current_time = end_time
            continue

        command = [
            sys.executable,
            "-u",
            str(driver_script),
            "--grid-shape", "65,9,7,11,9",
            "--distill-grid-shape", "49,7,5,9,7",
            "--distilled-cache", str(args.distilled_cache),
            "--operator-cache", str(args.operator_cache),
            "--keo-cross-rank", "8",
            "--keo-cross-sweeps", "5",
            "--operator-rank", "64",
            "--state-rank", "24",
            "--tmax-fs", str(duration),
            "--steps", str(steps),
            "--interval", str(args.interval),
            "--krylov-dim", "8",
            "--krylov-tol", "1e-10",
            "--workers", "8",
            "--integrator", "tdvp",
            "--output", str(directory),
        ]
        if previous is not None:
            command.extend(("--resume-from", str(previous)))
        if not args.quiet:
            command.append("--progress")
        print(
            f"[segment] propagate {current_time:.2f} -> {end_time:.2f} fs "
            f"({steps} steps)",
            flush=True,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(PROJECT_ROOT),
                "MPLCONFIGDIR": "/private/tmp/matplotlib-codex",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        if not _complete(directory, end_time):
            raise RuntimeError(f"segment did not produce a complete checkpoint: {directory}")
        previous = result
        current_time = end_time

    print(f"completed cumulative 5D dynamics: {previous}", flush=True)


if __name__ == "__main__":
    main()
