#!/usr/bin/env python3
"""Converged Heisenberg-chain comparison of MPS and nearest-neighbor LETTA.

Exact diagonalization in the smallest-|Sz| sector supplies references through
``L=20``.  Larger chains use an independent, high-bond-dimension TeNPy MPS
reference, checked at two bond dimensions.  Low-D LETTA calculations use
multiple random starts and retain the lowest converged energy.

The output is checkpointed after every calculation and can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_heisenberg_1d_letta_vs_mps import (  # noqa: E402
    comma_separated_ints,
    exact_ground_state,
    git_revision,
    letta_run,
)
from pyqed.models.heisenberg import Heisenberg  # noqa: E402


DEFAULT_OUTPUT = HERE / "results" / "heisenberg_1d_converged.json"


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def float_list(values) -> list[float]:
    return [float(np.real(value)) for value in values]


def tenpy_mps_run(
    length: int,
    chi_max: int,
    *,
    max_sweeps: int,
    energy_tolerance: float,
    entropy_tolerance: float,
    svd_min: float,
) -> dict:
    """Return an independent high-D MPS reference from TeNPy."""
    import tenpy
    from tenpy.algorithms import dmrg
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS

    model = SpinChain(
        {
            "L": int(length),
            "S": 0.5,
            "Jx": 1.0,
            "Jy": 1.0,
            "Jz": 1.0,
            "hz": 0.0,
            "bc_MPS": "finite",
            "conserve": "Sz",
        }
    )
    state = MPS.from_product_state(
        model.lat.mps_sites(),
        ["up" if site % 2 == 0 else "down" for site in range(length)],
        bc=model.lat.bc_MPS,
        unit_cell_width=length,
    )
    options = {
        "active_sites": 2,
        "mixer": True,
        "max_sweeps": int(max_sweeps),
        "min_sweeps": 4,
        "max_E_err": float(energy_tolerance),
        "max_S_err": float(entropy_tolerance),
        # Low-chi benchmark runs intentionally retain a large discarded
        # weight; convergence is judged from the variational energy instead.
        "max_trunc_err": 1.0,
        "trunc_params": {
            "chi_max": int(chi_max),
            "svd_min": float(svd_min),
        },
    }
    # TeNPy wraps nested option dictionaries in Config objects in-place.
    # Preserve a JSON-serializable copy before calling the engine.
    settings = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in options.items()
    }
    started = perf_counter()
    info = dmrg.run(state, model, options)
    seconds = perf_counter() - started

    energy = float(model.H_MPO.expectation_value(state))
    variance = float(model.H_MPO.variance(state))
    statistics = info["sweep_statistics"]
    energies = float_list(statistics["E"])
    final_delta = None if len(energies) < 2 else abs(energies[-1] - energies[-2])
    result = {
        "method": "TeNPy two-site DMRG",
        "tenpy_version": tenpy.__version__,
        "energy": energy,
        "variance": variance,
        "variance_abs": abs(variance),
        "chi_max_requested": int(chi_max),
        "chi_max_reached": int(max(state.chi)),
        "sweeps_completed": len(energies),
        "final_delta_energy": final_delta,
        "converged_posthoc": bool(
            final_delta is not None and final_delta <= 10.0 * float(energy_tolerance)
        ),
        "seconds": seconds,
        "energy_history": energies,
        "settings": settings,
    }
    for source, target in (
        ("max_trunc_err", "max_truncation_error_history"),
        ("S", "entropy_history"),
    ):
        values = statistics.get(source)
        if values is not None:
            result[target] = float_list(values)
    return result


def protocol_from_args(args: argparse.Namespace) -> dict:
    return {
        "lengths": list(args.lengths),
        "bond_dimensions": list(args.bond_dims),
        "letta_seeds": list(args.seeds),
        "maximum_passes": int(args.max_sweeps),
        "energy_tolerance": float(args.tol),
        "letta_gauge": args.letta_gauge,
        "mps_initial_state": "Neel product state",
        "mps_backend": "TeNPy two-site DMRG",
        "mps_entropy_tolerance": float(args.mps_entropy_tol),
        "mps_reported_energy": "normalized final-state expectation value",
        "exact_diagonalization_max_length": int(args.exact_max_length),
        "exact_tolerance": float(args.ed_tol),
        "reference_backend": "TeNPy two-site DMRG",
        "reference_chi_checks": list(args.reference_chis),
        "reference_max_sweeps": int(args.reference_max_sweeps),
        "reference_energy_tolerance": float(args.reference_tol),
        "reference_entropy_tolerance": float(args.reference_entropy_tol),
        "reference_svd_min": float(args.reference_svd_min),
    }


def fresh_payload(protocol: dict) -> dict:
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "open spin-1/2 antiferromagnetic Heisenberg chain, J=1",
        "protocol": protocol,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "git_revision": git_revision(),
        },
        "references": {},
        "runs": {},
        "summary": [],
    }


def load_or_initialize(args: argparse.Namespace, protocol: dict) -> dict:
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text())
        if payload.get("protocol") != protocol:
            raise ValueError(
                "existing output uses a different protocol; choose a new --output "
                "or pass --restart"
            )
        return payload
    return fresh_payload(protocol)


def reference_for_length(
    payload: dict,
    args: argparse.Namespace,
    length: int,
) -> dict:
    length_key = str(length)
    existing = payload["references"].get(length_key)
    if existing is not None and existing.get("complete"):
        return existing

    if length <= args.exact_max_length:
        exact = exact_ground_state(length, args.ed_tol)
        record = {
            "complete": True,
            "method": "fixed-Sz sparse exact diagonalization",
            "energy": float(exact["energy"]),
            "uncertainty_energy": float(exact["residual_norm"]),
            "details": exact,
        }
        payload["references"][length_key] = record
        atomic_json(payload, args.output)
        print(
            f"L={length:2d} exact reference E={record['energy']:.15f} "
            f"residual={exact['residual_norm']:.2e}",
            flush=True,
        )
        return record

    record = existing or {
        "complete": False,
        "method": "TeNPy high-D MPS",
        "checks": {},
    }
    payload["references"][length_key] = record
    for chi_max in args.reference_chis:
        chi_key = str(chi_max)
        if chi_key in record["checks"]:
            continue
        print(f"L={length:2d} reference chi={chi_max} ...", flush=True)
        record["checks"][chi_key] = tenpy_mps_run(
            length,
            chi_max,
            max_sweeps=args.reference_max_sweeps,
            energy_tolerance=args.reference_tol,
            entropy_tolerance=args.reference_entropy_tol,
            svd_min=args.reference_svd_min,
        )
        atomic_json(payload, args.output)
        check = record["checks"][chi_key]
        print(
            f"  E={check['energy']:.15f} variance={check['variance_abs']:.2e} "
            f"sweeps={check['sweeps_completed']} t={check['seconds']:.2f}s",
            flush=True,
        )

    selected_chi = max(args.reference_chis)
    selected = record["checks"][str(selected_chi)]
    sorted_checks = sorted(record["checks"].values(), key=lambda item: item["chi_max_requested"])
    chi_gap = None
    if len(sorted_checks) >= 2:
        chi_gap = abs(sorted_checks[-1]["energy"] - sorted_checks[-2]["energy"])
    record.update(
        {
            "complete": True,
            "energy": float(selected["energy"]),
            "selected_chi": int(selected_chi),
            "chi_check_gap": chi_gap,
            "uncertainty_energy": max(
                float(selected["variance_abs"]) ** 0.5,
                0.0 if chi_gap is None else float(chi_gap),
            ),
        }
    )
    atomic_json(payload, args.output)
    return record


def run_key(method: str, length: int, bond_dim: int, seed: int | None = None) -> str:
    suffix = "" if seed is None else f":seed={seed}"
    return f"{method}:L={length}:D={bond_dim}{suffix}"


def run_variational_calculations(
    payload: dict,
    args: argparse.Namespace,
    length: int,
    reference_energy: float,
) -> None:
    model = Heisenberg(L=length)
    mpo = model.build_H_mpo().factors
    for bond_dim in args.bond_dims:
        key = run_key("mps", length, bond_dim)
        if key not in payload["runs"]:
            print(f"L={length:2d} D={bond_dim} MPS ...", flush=True)
            raw = tenpy_mps_run(
                length,
                bond_dim,
                max_sweeps=args.max_sweeps,
                energy_tolerance=args.tol,
                entropy_tolerance=args.mps_entropy_tol,
                svd_min=args.reference_svd_min,
            )
            result = {
                "energy": float(raw["energy"]),
                "seconds": float(raw["seconds"]),
                "converged": bool(raw["converged_posthoc"]),
                "passes_completed": int(raw["sweeps_completed"]),
                "final_delta_energy": raw["final_delta_energy"],
                "variance": float(raw["variance"]),
                "variance_abs": float(raw["variance_abs"]),
                "history": [
                    {"sweep": sweep, "energy": energy}
                    for sweep, energy in enumerate(raw["energy_history"])
                ],
                "backend_details": raw,
            }
            result.update(
                {
                    "method": "MPS/DMRG (TeNPy)",
                    "length": int(length),
                    "bond_dim": int(bond_dim),
                    "error": float(result["energy"] - reference_energy),
                    "error_per_site": float((result["energy"] - reference_energy) / length),
                }
            )
            payload["runs"][key] = result
            atomic_json(payload, args.output)
            print(
                f"  E={result['energy']:.15f} de/L={result['error_per_site']:.3e} "
                f"conv={result['converged']} passes={result['passes_completed']} "
                f"t={result['seconds']:.2f}s",
                flush=True,
            )

        for seed in args.seeds:
            key = run_key("letta", length, bond_dim, seed)
            if key in payload["runs"]:
                continue
            print(f"L={length:2d} D={bond_dim} LETTA seed={seed} ...", flush=True)
            result = letta_run(
                mpo,
                length,
                bond_dim,
                seed,
                args.max_sweeps,
                args.tol,
                args.letta_gauge,
            )
            result.update(
                {
                    "method": "LETTA",
                    "length": int(length),
                    "bond_dim": int(bond_dim),
                    "error": float(result["energy"] - reference_energy),
                    "error_per_site": float((result["energy"] - reference_energy) / length),
                }
            )
            payload["runs"][key] = result
            atomic_json(payload, args.output)
            print(
                f"  E={result['energy']:.15f} de/L={result['error_per_site']:.3e} "
                f"conv={result['converged']} passes={result['passes_completed']} "
                f"t={result['seconds']:.2f}s",
                flush=True,
            )


def build_summary(payload: dict) -> list[dict]:
    protocol = payload["protocol"]
    summary = []
    for length in protocol["lengths"]:
        reference = payload["references"][str(length)]
        reference_energy = float(reference["energy"])
        for bond_dim in protocol["bond_dimensions"]:
            mps = payload["runs"][run_key("mps", length, bond_dim)]
            starts = [
                payload["runs"][run_key("letta", length, bond_dim, seed)]
                for seed in protocol["letta_seeds"]
            ]
            converged_starts = [record for record in starts if record["converged"]]
            candidate_starts = converged_starts or starts
            selected = min(candidate_starts, key=lambda record: record["energy"])
            errors = np.asarray([record["error_per_site"] for record in starts], dtype=float)
            summary.append(
                {
                    "length": int(length),
                    "bond_dim": int(bond_dim),
                    "reference_method": reference["method"],
                    "reference_energy": reference_energy,
                    "reference_uncertainty_energy": reference.get("uncertainty_energy"),
                    "mps_energy": float(mps["energy"]),
                    "mps_error": float(mps["error"]),
                    "mps_error_per_site": float(mps["error_per_site"]),
                    "mps_converged": bool(mps["converged"]),
                    "mps_passes": int(mps["passes_completed"]),
                    "mps_seconds": float(mps["seconds"]),
                    "letta_energy": float(selected["energy"]),
                    "letta_error": float(selected["error"]),
                    "letta_error_per_site": float(selected["error_per_site"]),
                    "letta_selected_seed": int(selected["seed"]),
                    "letta_selected_converged": bool(selected["converged"]),
                    "letta_converged_starts": len(converged_starts),
                    "letta_total_starts": len(starts),
                    "letta_passes": int(selected["passes_completed"]),
                    "letta_seconds": float(selected["seconds"]),
                    "letta_error_per_site_min": float(np.min(errors)),
                    "letta_error_per_site_max": float(np.max(errors)),
                    "letta_error_per_site_std": float(np.std(errors)),
                    "gain_per_site": float(
                        mps["error_per_site"] - selected["error_per_site"]
                    ),
                }
            )
    return summary


def write_csv_files(payload: dict, output: Path) -> tuple[Path, Path, Path]:
    summary_path = output.with_name(output.stem + "_summary.csv")
    starts_path = output.with_name(output.stem + "_starts.csv")
    reference_path = output.with_name(output.stem + "_references.csv")

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["summary"][0]))
        writer.writeheader()
        writer.writerows(payload["summary"])

    run_rows = []
    for key, record in payload["runs"].items():
        run_rows.append(
            {
                "key": key,
                "method": record["method"],
                "length": record["length"],
                "bond_dim": record["bond_dim"],
                "seed": record.get("seed"),
                "energy": record["energy"],
                "error": record["error"],
                "error_per_site": record["error_per_site"],
                "converged": record["converged"],
                "passes_completed": record["passes_completed"],
                "seconds": record["seconds"],
            }
        )
    with starts_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    reference_rows = []
    for length, record in payload["references"].items():
        base = {
            "length": int(length),
            "method": record["method"],
            "selected": True,
            "chi": record.get("selected_chi"),
            "energy": record["energy"],
            "uncertainty_energy": record.get("uncertainty_energy"),
            "variance_abs": None,
            "chi_check_gap": record.get("chi_check_gap"),
        }
        checks = record.get("checks", {})
        if not checks:
            reference_rows.append(base)
        for check in checks.values():
            row = dict(base)
            row.update(
                {
                    "selected": check["chi_max_requested"] == record.get("selected_chi"),
                    "chi": check["chi_max_requested"],
                    "energy": check["energy"],
                    "variance_abs": check["variance_abs"],
                }
            )
            reference_rows.append(row)
    with reference_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)
    return summary_path, starts_path, reference_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=comma_separated_ints,
        default=(6, 8, 10, 12, 14, 16, 18, 20, 24, 32, 48),
    )
    parser.add_argument("--bond-dims", type=comma_separated_ints, default=(1, 2, 4))
    parser.add_argument("--seeds", type=comma_separated_ints, default=(1, 2, 3, 4, 5))
    parser.add_argument("--max-sweeps", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--mps-entropy-tol", type=float, default=1.0e-9)
    parser.add_argument("--letta-gauge", choices=("virtual", "conditional"), default="conditional")
    parser.add_argument("--exact-max-length", type=int, default=20)
    parser.add_argument("--ed-tol", type=float, default=1.0e-12)
    parser.add_argument("--reference-chis", type=comma_separated_ints, default=(128, 256))
    parser.add_argument("--reference-max-sweeps", type=int, default=100)
    parser.add_argument("--reference-tol", type=float, default=1.0e-13)
    parser.add_argument("--reference-entropy-tol", type=float, default=1.0e-11)
    parser.add_argument("--reference-svd-min", type=float, default=1.0e-15)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", default=True)
    mode.add_argument("--restart", action="store_false", dest="resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = protocol_from_args(args)
    payload = load_or_initialize(args, protocol)

    for length in args.lengths:
        reference = reference_for_length(payload, args, length)
        run_variational_calculations(payload, args, length, float(reference["energy"]))

    payload["summary"] = build_summary(payload)
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(payload, args.output)
    csv_paths = write_csv_files(payload, args.output)
    print(f"wrote {args.output}", flush=True)
    for path in csv_paths:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
