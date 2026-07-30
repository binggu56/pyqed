#!/usr/bin/env python3
"""GDVR-DMRG ground-state PES for a two-mode linear H4 grid."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pyqed.qchem.gdvr import AtomicChain


def h4_mode_geometry(q1, q2):
    """Return linear H4 coordinates for the two active middle-atom modes."""
    fixed_positions = (-3.6, 3.6)
    r0_2 = -1.2
    r0_3 = 1.2
    sqrt2 = np.sqrt(2.0)

    r2 = r0_2 + (float(q1) + float(q2)) / sqrt2
    r3 = r0_3 + (float(q1) - float(q2)) / sqrt2
    return [
        (0.0, 0.0, fixed_positions[0]),
        (0.0, 0.0, float(r2)),
        (0.0, 0.0, float(r3)),
        (0.0, 0.0, fixed_positions[1]),
    ]


def linspace_from_args(start, stop, num):
    if int(num) <= 1:
        return np.array([0.5 * (float(start) + float(stop))], dtype=float)
    return np.linspace(float(start), float(stop), int(num))


def run_point(task):
    idx, iq1, iq2, q1, q2, args_dict = task
    started = time.perf_counter()
    coords = h4_mode_geometry(q1, q2)
    z = np.array([c[2] for c in coords], dtype=float)

    record = {
        "index": int(idx),
        "iq1": int(iq1),
        "iq2": int(iq2),
        "q1": float(q1),
        "q2": float(q2),
        "z1": float(z[0]),
        "z2": float(z[1]),
        "z3": float(z[2]),
        "z4": float(z[3]),
        "r12": float(z[1] - z[0]),
        "r23": float(z[2] - z[1]),
        "r34": float(z[3] - z[2]),
        "rhf_energy": np.nan,
        "too_energy": np.nan,
        "too_cycles": 0,
        "too_converged": False,
        "dmrg_energy": np.nan,
        "mode": args_dict["mode"],
        "ncas": int(args_dict["ncas"]),
        "elapsed_s": np.nan,
        "success": False,
        "message": "",
    }

    if bool(args_dict["reject_crossings"]) and not (z[0] < z[1] < z[2] < z[3]):
        record["message"] = "skipped: atom order crossing"
        record["elapsed_s"] = float(time.perf_counter() - started)
        return record

    try:
        mol = AtomicChain(["H", "H", "H", "H"], coords=coords, spin=0)
        mol.build(
            Lz=float(args_dict["lz"]),
            Nz=int(args_dict["nz"]),
            M=int(args_dict["m"]),
            transverse_basis=args_dict["transverse_basis"],
            dvr_method=args_dict["dvr_method"],
            verbose=bool(args_dict["verbose"]),
        )
        mf = mol.RHF().run(
            newton=False,
            conv=float(args_dict["rhf_conv"]),
            verbose=bool(args_dict["verbose"]),
        )
        record["rhf_energy"] = float(mf.e_tot)

        mf.newton(
            max_cycles=int(args_dict["too_cycles"]),
            sweeps=int(args_dict["too_sweep_iterations"]),
            tol=float(args_dict["too_tol"]),
            ridge=float(args_dict["too_ridge"]),
            trust_step=float(args_dict["too_trust_step"]),
            trust_radius=float(args_dict["too_trust_radius"]),
            scf_conv=float(args_dict["rhf_conv"]),
            scf_max_iter=int(args_dict["rhf_max_iter"]),
            verbose=bool(args_dict["verbose"]),
        )
        record["too_energy"] = float(mf.e_tot)
        record["too_cycles"] = int(mf.info.get("newton_cycles", 0))
        record["too_converged"] = bool(mf.info.get("newton_converged", False))
        if not record["too_converged"] and not bool(args_dict["allow_unconverged_too"]):
            raise RuntimeError(
                "TOO/Newton did not converge "
                f"after {record['too_cycles']} cycles at q1={q1}, q2={q2}; "
                "increase --too-cycles, loosen --too-tol, or pass --allow-unconverged-too."
            )

        if bool(args_dict["skip_dmrg"]):
            record["dmrg_energy"] = float(record["too_energy"])
            record["success"] = True
            record["message"] = "TOO only"
            return record

        if args_dict["mode"] == "direct":
            td = mf.TDDMRG(symbolic_algo=args_dict["symbolic_algo"]).build()
        else:
            td = mf.TDDMRG(
                ncas=int(args_dict["ncas"]),
                nelecas=4,
                spin=0,
            ).build()
        td.optimize_ground_state(
            D=int(args_dict["bond"]),
            nstates=1,
            nsweeps=int(args_dict["sweeps"]),
            symmetry_list=None if bool(args_dict["no_symmetry"]) else ["charge", "sz"],
            compute_s2=False,
            davidson_tol=float(args_dict["dmrg_tol"]),
            not_conv_err=bool(args_dict["not_conv_err"]),
        )
        record["dmrg_energy"] = float(td.e_tot)
        record["success"] = True
        record["message"] = "ok"
    except Exception as exc:
        record["message"] = f"{type(exc).__name__}: {exc}"
        if bool(args_dict["strict"]):
            raise
    finally:
        record["elapsed_s"] = float(time.perf_counter() - started)
    return record


def write_csv(path, rows):
    fieldnames = [
        "index",
        "iq1",
        "iq2",
        "q1",
        "q2",
        "z1",
        "z2",
        "z3",
        "z4",
        "r12",
        "r23",
        "r34",
        "rhf_energy",
        "too_energy",
        "too_cycles",
        "too_converged",
        "dmrg_energy",
        "mode",
        "ncas",
        "elapsed_s",
        "success",
        "message",
    ]
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_outputs(outdir, q1_values, q2_values, rows, args):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["index"]))

    csv_path = outdir / "h4_gdvr_dmrg_ground_state.csv"
    write_csv(csv_path, rows)

    shape = (len(q1_values), len(q2_values))
    rhf = np.full(shape, np.nan, dtype=float)
    too = np.full(shape, np.nan, dtype=float)
    dmrg = np.full(shape, np.nan, dtype=float)
    success = np.zeros(shape, dtype=bool)
    r23 = np.full(shape, np.nan, dtype=float)
    for row in rows:
        i = int(row["iq1"])
        j = int(row["iq2"])
        rhf[i, j] = float(row["rhf_energy"])
        too[i, j] = float(row["too_energy"])
        dmrg[i, j] = float(row["dmrg_energy"])
        success[i, j] = bool(row["success"])
        r23[i, j] = float(row["r23"])

    np.savez(
        outdir / "h4_gdvr_dmrg_ground_state.npz",
        q1=np.asarray(q1_values, dtype=float),
        q2=np.asarray(q2_values, dtype=float),
        rhf_energy=rhf,
        too_energy=too,
        dmrg_energy=dmrg,
        success=success,
        r23=r23,
    )
    metadata = {
        "script": str(Path(__file__).resolve()),
        "arguments": {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
        "n_points": int(len(rows)),
        "n_success": int(sum(bool(row["success"]) for row in rows)),
        "csv": str(csv_path),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-min", type=float, default=-0.4)
    parser.add_argument("--q1-max", type=float, default=0.4)
    parser.add_argument("--n-q1", type=int, default=5)
    parser.add_argument("--q2-min", type=float, default=-0.4)
    parser.add_argument("--q2-max", type=float, default=0.4)
    parser.add_argument("--n-q2", type=int, default=5)
    parser.add_argument("--lz", type=float, default=6.0)
    parser.add_argument("--nz", type=int, default=31)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--transverse-basis", default=None)
    parser.add_argument("--dvr-method", default="sine")
    parser.add_argument("--rhf-conv", type=float, default=1.0e-8)
    parser.add_argument("--rhf-max-iter", type=int, default=100)
    parser.add_argument("--too-cycles", type=int, default=50)
    parser.add_argument("--too-sweep-iterations", type=int, default=1)
    parser.add_argument("--too-tol", type=float, default=1.0e-6)
    parser.add_argument("--too-ridge", type=float, default=0.5)
    parser.add_argument("--too-trust-step", type=float, default=0.5)
    parser.add_argument("--too-trust-radius", type=float, default=1.0)
    parser.add_argument("--allow-unconverged-too", action="store_true")
    parser.add_argument("--bond", type=int, default=20)
    parser.add_argument("--mode", choices=("active", "direct"), default="active")
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=50)
    parser.add_argument("--dmrg-tol", type=float, default=1.0e-6)
    parser.add_argument("--symbolic-algo", default="qr")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/gdvr_h4_modes_dmrg"))
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--not-conv-err", action="store_true")
    parser.add_argument("--reject-crossings", action="store_true", default=True)
    parser.add_argument("--allow-crossings", action="store_false", dest="reject_crossings")
    parser.add_argument("--skip-dmrg", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    q1_values = linspace_from_args(args.q1_min, args.q1_max, args.n_q1)
    q2_values = linspace_from_args(args.q2_min, args.q2_max, args.n_q2)
    tasks = []
    for iq1, q1 in enumerate(q1_values):
        for iq2, q2 in enumerate(q2_values):
            tasks.append((len(tasks), iq1, iq2, float(q1), float(q2), vars(args)))

    print(
        f"H4 GDVR-DMRG ground-state scan: {len(tasks)} geometries, "
        f"Nz={args.nz}, D={args.bond}, sweeps={args.sweeps}, workers={args.workers}"
    )
    print(f"output: {args.outdir}")
    if args.dry_run:
        for task in tasks[: min(8, len(tasks))]:
            _idx, _iq1, _iq2, q1, q2, _args_dict = task
            z = [c[2] for c in h4_mode_geometry(q1, q2)]
            print(f"q=({q1:+.6f}, {q2:+.6f}) z={z}")
        return 0

    rows = []
    if int(args.workers) <= 1:
        for task in tasks:
            row = run_point(task)
            rows.append(row)
            print(
                f"[{row['index'] + 1:4d}/{len(tasks)}] q=({row['q1']:+.4f},{row['q2']:+.4f}) "
                f"E_DMRG={row['dmrg_energy']:.12f} success={row['success']} {row['message']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [pool.submit(run_point, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"[{len(rows):4d}/{len(tasks)}] q=({row['q1']:+.4f},{row['q2']:+.4f}) "
                    f"E_DMRG={row['dmrg_energy']:.12f} success={row['success']} {row['message']}"
                )

    csv_path = write_outputs(args.outdir, q1_values, q2_values, rows, args)
    print(f"wrote {csv_path}")
    failures = [row for row in rows if not bool(row["success"])]
    if failures:
        print(f"failed points: {len(failures)}")
        for row in failures:
            print(f"  index={row['index']} q=({row['q1']:+.6f},{row['q2']:+.6f}) {row['message']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
