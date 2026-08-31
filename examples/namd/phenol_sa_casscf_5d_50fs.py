#!/usr/bin/env python3
"""Restartable, durable phenol 5D SA-CASSCF 50 fs photodynamics workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = (
    PROJECT_ROOT / "dataset" / "phenol_sa6_casscf_production" / "electronic.sqlite"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "dataset" / "phenol_sa6_casscf_production"
    / "dynamics" / "5d_50fs"
)


def _qualified(summary, key):
    summary = Path(summary)
    if not summary.is_file():
        return False
    return bool(json.loads(summary.read_text()).get(key, False))


def _complete(paths):
    return all(Path(path).is_file() for path in paths)


def _planar_inward_ready(directory):
    directory = Path(directory)
    summary_path = directory / "summary.json"
    data_path = directory / "phenol_sa6_tracked3_p_gauge.npz"
    if not _complete((summary_path, data_path)):
        return False
    summary = json.loads(summary_path.read_text())
    lower, upper = summary["continuous_component_angstrom"]
    gates = summary["gates"]
    return bool(
        lower <= 0.75 + 1.0e-8
        and upper >= 1.15 - 1.0e-8
        and gates["P_links_positive_on_continuous_component"]
        and gates["reflection_characters_pure"]
    )


def _production_trajectory_ready(directory, *, cap):
    directory = Path(directory)
    summary_path = directory / "summary.json"
    arrays = directory / "phenol_sa_casscf_5d_ftt_ttldr.npz"
    figure = directory / "phenol_sa_casscf_5d_ftt_ttldr.png"
    if not _complete((summary_path, arrays, figure)):
        return False
    summary = json.loads(summary_path.read_text())
    cap_summary = summary["cap"]
    return bool(
        summary["grid_shape"] == [49, 5, 3, 5, 5]
        and summary["dynamics"]["time_fs"] == 50.0
        and summary["validation_spectral_rms_mev"] <= 10.0
        and summary["validation_spectral_max_mev"] <= 50.0
        and summary["maximum_population_sum_defect"] <= 1.0e-8
        and cap_summary["enabled"] is bool(cap)
        and cap_summary["maximum_absorption_closure_defect"] <= 1.0e-8
        and (cap or abs(cap_summary["final_norm"] - 1.0) <= 1.0e-6)
    )


def _run_stage(name, command, output, *, complete, dry_run, force):
    output = Path(output)
    command = [str(item) for item in command]
    if complete() and not force:
        print(f"[{name}] complete; reusing {output}", flush=True)
        return {"name": name, "status": "reused", "command": command}
    print(f"[{name}] {'would run' if dry_run else 'running'}", flush=True)
    print("  " + " ".join(command), flush=True)
    if dry_run:
        return {"name": name, "status": "planned", "command": command}
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
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
    log = output / "workflow.log"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{name}] {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    seconds = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log}")
    if not complete():
        raise RuntimeError(f"{name} finished but did not pass its completion gate; see {log}")
    print(f"[{name}] complete in {seconds:.1f} s", flush=True)
    return {
        "name": name,
        "status": "completed",
        "seconds": seconds,
        "command": command,
        "log": str(log),
    }


def workflow(args):
    root = args.output.resolve()
    inputs = root / "inputs"
    field = inputs / "p_gauge_5d"
    planar = inputs / "p_gauge_planar"
    inward = inputs / "p_gauge_5d_inward"
    validation = root / "model" / "validation"
    refinement = root / "model" / "refinement"
    production = root / "model" / "production"
    final_model = root / "model" / "final"
    correction = root / "model" / "radial_correction"
    smoke = root / "smoke"
    cap = root / "cap"
    reference = root / "no_cap_reference"
    cache = root / "cache" / "ftt_r49"
    python = Path(sys.executable)
    scripts = PROJECT_ROOT / "examples" / "namd"
    forced = set(args.force_stage)
    records = []

    stages = [
        (
            "electronic_field",
            [
                python, scripts / "phenol_sa_casscf_5d_pilot.py",
                "--database", args.database,
                "--source-run-id", args.source_run_id,
                "--output", field,
                "--samples", "128",
                "--workers", str(args.workers),
                "--diagnostic-workers", str(args.workers),
                "--overlap-workers", str(args.workers),
                "--quiet",
            ],
            field,
            lambda: _qualified(field / "summary.json", "passed"),
        ),
        (
            "planar_p_gauge",
            [
                python, scripts / "phenol_sa_casscf_p_gauge.py",
                "--database", args.database,
                "--output", planar,
            ],
            planar,
            lambda: _planar_inward_ready(planar),
        ),
        (
            "inward_backbone",
            [
                python, scripts / "phenol_sa_casscf_5d_inward_backbone.py",
                "--base", field / "phenol_sa6_5d_p_gauge.npz",
                "--inward", planar / "phenol_sa6_tracked3_p_gauge.npz",
                "--output", inward,
            ],
            inward,
            lambda: _qualified(inward / "summary.json", "passed"),
        ),
    ]
    for name, command, output, complete in stages:
        records.append(
            _run_stage(
                name, command, output, complete=complete,
                dry_run=args.dry_run, force=name in forced,
            )
        )
    if args.through == "inputs":
        return _write_manifest(args, root, records)

    model_stages = [
        (
            "model_validation",
            [
                python, scripts / "phenol_sa_casscf_5d_mace_y.py",
                "--data", field / "phenol_sa6_5d_p_gauge.npz",
                "--output", validation,
                "--epochs", str(args.validation_epochs),
                "--sync-steps", str(args.sync_steps),
                "--seed", "61",
            ],
            validation,
            lambda: _complete(
                (
                    validation / "summary.json",
                    validation / "phenol_sa6_5d_mace_y.pt",
                )
            ),
        ),
        (
            "model_refinement",
            [
                python, scripts / "phenol_sa_casscf_5d_mace_y.py",
                "--data", field / "phenol_sa6_5d_p_gauge.npz",
                "--output", refinement,
                "--initial", validation / "phenol_sa6_5d_mace_y.pt",
                "--reuse-initial-feature-field",
                "--focus-energy-above-ev", "0.05",
                "--focus-repeats", "6",
                "--energy-weight", "200",
                "--learning-rate", "0.001",
                "--epochs", str(args.refinement_epochs),
                "--sync-steps", str(args.sync_steps),
                "--seed", "61",
            ],
            refinement,
            lambda: _qualified(refinement / "summary.json", "passed"),
        ),
        (
            "model_production",
            [
                python, scripts / "phenol_sa_casscf_5d_mace_y.py",
                "--data", field / "phenol_sa6_5d_p_gauge.npz",
                "--output", production,
                "--initial", refinement / "phenol_sa6_5d_mace_y.pt",
                "--reuse-initial-feature-field",
                "--fit-all",
                "--epochs", str(args.production_epochs),
                "--sync-steps", str(args.sync_steps),
                "--seed", "61",
            ],
            production,
            lambda: _complete(
                (
                    production / "summary.json",
                    production / "phenol_sa6_5d_mace_y.pt",
                )
            ),
        ),
        (
            "model_final",
            [
                python, scripts / "phenol_sa_casscf_5d_mace_y.py",
                "--data", field / "phenol_sa6_5d_p_gauge.npz",
                "--output", final_model,
                "--initial", production / "phenol_sa6_5d_mace_y.pt",
                "--reuse-initial-feature-field",
                "--fit-all",
                "--focus-energy-above-ev", "0.10",
                "--focus-repeats", "10",
                "--energy-weight", "300",
                "--learning-rate", "0.0005",
                "--epochs", str(args.final_epochs),
                "--sync-steps", str(args.sync_steps),
                "--seed", "61",
            ],
            final_model,
            lambda: _qualified(final_model / "summary.json", "passed"),
        ),
        (
            "radial_correction",
            [
                python, scripts / "phenol_sa_casscf_5d_radial_delta.py",
                "--data", inward / "phenol_sa6_5d_p_gauge_inward.npz",
                "--checkpoint", final_model / "phenol_sa6_5d_mace_y.pt",
                "--output", correction,
            ],
            correction,
            lambda: _qualified(correction / "summary.json", "passed"),
        ),
    ]
    for name, command, output, complete in model_stages:
        records.append(
            _run_stage(
                name, command, output, complete=complete,
                dry_run=args.dry_run, force=name in forced,
            )
        )
    if args.through == "model":
        return _write_manifest(args, root, records)

    common = [
        python, scripts / "phenol_sa_casscf_5d_ftt_ttldr.py",
        "--data", inward / "phenol_sa6_5d_p_gauge_inward.npz",
        "--checkpoint", final_model / "phenol_sa6_5d_mace_y.pt",
        "--radial-correction", correction / "phenol_sa6_5d_radial_delta.npz",
        "--bright-state", str(args.bright_state),
        "--integrator", "tdvp",
        "--workers", str(args.workers),
    ]
    smoke_command = common + [
        "--grid-shape", "9,3,3,3,3",
        "--tmax-fs", "0.5",
        "--steps", "1",
        "--interval", "1",
        "--state-rank", "16",
        "--output", smoke,
    ]
    records.append(
        _run_stage(
            "smoke", smoke_command, smoke,
            complete=lambda: _complete(
                (
                    smoke / "summary.json",
                    smoke / "phenol_sa_casscf_5d_ftt_ttldr.npz",
                    smoke / "phenol_sa_casscf_5d_ftt_ttldr.png",
                )
            ),
            dry_run=args.dry_run, force="smoke" in forced,
        )
    )
    if args.through == "smoke":
        return _write_manifest(args, root, records)

    full_common = common + [
        "--grid-shape", "49,5,3,5,5",
        "--distilled-cache", cache,
        "--tmax-fs", "50",
        "--steps", str(args.steps),
        "--interval", str(max(1, args.steps // 100)),
        "--progress",
    ]
    records.append(
        _run_stage(
            "cap_50fs", full_common + ["--output", cap], cap,
            complete=lambda: _production_trajectory_ready(cap, cap=True),
            dry_run=args.dry_run, force="cap_50fs" in forced,
        )
    )
    records.append(
        _run_stage(
            "reference_50fs",
            full_common + ["--cap-strength", "0", "--output", reference],
            reference,
            complete=lambda: _production_trajectory_ready(reference, cap=False),
            dry_run=args.dry_run, force="reference_50fs" in forced,
        )
    )
    comparison = root / "phenol_5d_cap_vs_no_cap.png"
    records.append(
        _run_stage(
            "comparison_plot",
            [
                python, scripts / "plot_phenol_5d_cap_comparison.py",
                "--cap", cap / "phenol_sa_casscf_5d_ftt_ttldr.npz",
                "--reference", reference / "phenol_sa_casscf_5d_ftt_ttldr.npz",
                "--output", comparison,
            ],
            root,
            complete=lambda: _complete((comparison, comparison.with_suffix(".pdf"))),
            dry_run=args.dry_run, force="comparison_plot" in forced,
        )
    )
    return _write_manifest(args, root, records)


def _write_manifest(args, root, records):
    manifest = {
        "workflow": "phenol 5D SA-CASSCF MACE-Y + FTT + TTLDR",
        "time_fs": 50.0,
        "grid_shape": [49, 5, 3, 5, 5],
        "steps": int(args.steps),
        "bright_state": int(args.bright_state),
        "database": str(args.database.resolve()),
        "output": str(root),
        "through": args.through,
        "dry_run": bool(args.dry_run),
        "stages": records,
    }
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        (root / "workflow_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-run-id", default="phenol-sa6-5d-pilot-v1-s61-n128"
    )
    parser.add_argument(
        "--through", choices=("inputs", "model", "smoke", "production"),
        default="production",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validation-epochs", type=int, default=1000)
    parser.add_argument("--refinement-epochs", type=int, default=600)
    parser.add_argument("--production-epochs", type=int, default=600)
    parser.add_argument("--final-epochs", type=int, default=200)
    parser.add_argument("--sync-steps", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--bright-state", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-stage", action="append", default=[],
        choices=(
            "electronic_field", "planar_p_gauge", "inward_backbone",
            "model_validation", "model_refinement", "model_production",
            "model_final",
            "radial_correction",
            "smoke", "cap_50fs", "reference_50fs", "comparison_plot",
        ),
    )
    workflow(parser.parse_args())


if __name__ == "__main__":
    main()
