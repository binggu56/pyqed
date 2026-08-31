"""PyQED native RHF test for the repaired [Fe(bpy)3]2+ scaffold.

This is a scale/robustness check for PyQED's own RHF path.  The default
geometry comes from ``fe_bpy3_pyscf_ri_feasibility.py`` but the calculation
itself uses PyQED's Molecule/RHF implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem import Molecule, RHF
from pyqed.units import au2ev


def _load_fe_helpers():
    helper = Path(__file__).with_name("fe_bpy3_pyscf_ri_feasibility.py")
    spec = importlib.util.spec_from_file_location("fe_bpy3_pyscf_ri_feasibility", helper)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atom_string(atoms):
    return "; ".join(f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, x, y, z in atoms)


def timed(label, func):
    print(f"{label} ...", flush=True)
    t0 = time.perf_counter()
    out = func()
    seconds = time.perf_counter() - t0
    print(f"{label} finished in {seconds:.2f} s", flush=True)
    return out, seconds


def frontier_summary(mf, n=8):
    mo_energy = np.asarray(mf.mo_energy, dtype=float)
    mo_occ = np.asarray(mf.mo_occ, dtype=float)
    occ_idx = np.where(mo_occ > 0.0)[0]
    homo = int(occ_idx[-1]) if occ_idx.size else -1
    lo = max(0, homo - n)
    hi = min(mo_energy.size, homo + n + 2)
    rows = []
    for idx in range(lo, hi):
        rows.append(
            {
                "index": int(idx),
                "occ": float(mo_occ[idx]),
                "energy_au": float(mo_energy[idx]),
                "energy_ev": float(mo_energy[idx] * au2ev),
            }
        )
    return homo, rows


def print_frontier(rows, homo):
    print("\nFrontier orbital energies")
    print("idx      occ       eps/eV")
    for row in rows:
        idx = row["index"]
        marker = "H" if idx == homo else ("L" if idx == homo + 1 else " ")
        print(f"{idx:4d}{marker} {row['occ']:8.3f} {row['energy_ev']:12.5f}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, help="Optional XYZ geometry; defaults to repaired scaffold.")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--charge", type=int, default=2)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument(
        "--eri",
        choices=("auto", "dense", "s4", "s8", "direct", "factors", "ri"),
        default="ri",
    )
    parser.add_argument("--auxbasis", default=None)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eri-workers", type=int, default=None)
    parser.add_argument("--ri-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ri-cache-dir", type=Path, default=None)
    parser.add_argument("--one-electron-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ri-storage", choices=("auto", "packed", "full"), default="auto")
    parser.add_argument(
        "--ri-tensor-backend",
        choices=("auto", "cython", "python", "native"),
        default="auto",
        help="Backend for native RI three-center tensors.",
    )
    parser.add_argument("--ri-screen-tol", type=float, default=None)
    parser.add_argument("--low-rank-tol", type=float, default=1e-8)
    parser.add_argument("--low-rank-max-rank", type=int, default=None)
    parser.add_argument(
        "--rhf-density-fit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run PyQED RHF through its PySCF-backed density-fitting branch.",
    )
    parser.add_argument("--build-only", action="store_true", help="Stop after the integral/metadata build.")
    parser.add_argument("--max-cycle", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--conv-tol-dm", type=float, default=1e-6)
    parser.add_argument("--conv-tol-grad", type=float, default=None)
    parser.add_argument("--damping", type=float, default=0.2)
    parser.add_argument("--damping-mode", choices=("density", "fock"), default="density")
    parser.add_argument(
        "--damping-decay",
        type=float,
        default=1.0,
        help="Multiply RHF density damping by this factor each cycle after --damping-decay-start.",
    )
    parser.add_argument("--damping-decay-start", type=int, default=0)
    parser.add_argument("--damping-min", type=float, default=0.0)
    parser.add_argument("--level-shift", type=float, default=0.5)
    parser.add_argument(
        "--level-shift-decay",
        type=float,
        default=1.0,
        help="Multiply the RHF level shift by this factor each cycle after --level-shift-decay-start.",
    )
    parser.add_argument("--level-shift-decay-start", type=int, default=0)
    parser.add_argument("--level-shift-min", type=float, default=0.0)
    parser.add_argument("--scf-diis", choices=("cdiis", "ediis", "adiis", "hybrid"), default="cdiis")
    parser.add_argument("--diis-switch-tol", type=float, default=1e-3)
    parser.add_argument("--diis-start-cycle", type=int, default=2)
    parser.add_argument("--diis-space", type=int, default=12)
    parser.add_argument("--init-guess", default="minao")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    helper = _load_fe_helpers()
    atoms = helper.load_xyz(args.xyz) if args.xyz else helper.generated_fe_bpy3()

    print("Model:", "XYZ input" if args.xyz else "repaired generated [Fe(bpy)3]2+")
    print(f"Atoms={len(atoms)}, charge={args.charge}, spin={args.spin}, basis={args.basis}")
    print(f"PyQED native build: eri={args.eri}, auxbasis={args.auxbasis}")

    mol = Molecule(
        atom=atom_string(atoms),
        unit="angstrom",
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
    )
    build_options = {
        "parallel": args.parallel,
        "eri_workers": args.eri_workers,
        "ri_cache": args.ri_cache,
        "ri_cache_dir": None if args.ri_cache_dir is None else str(args.ri_cache_dir),
        "one_electron_cache": args.one_electron_cache,
        "ri_storage": args.ri_storage,
        "ri_tensor_backend": args.ri_tensor_backend,
        "ri_screen_tol": args.ri_screen_tol,
        "low_rank_tol": args.low_rank_tol,
        "low_rank_max_rank": args.low_rank_max_rank,
    }
    build_options = {key: value for key, value in build_options.items() if value is not None}

    _, build_seconds = timed(
        "PyQED integral build",
        lambda: mol.build(
            eri=args.eri,
            auxbasis=args.auxbasis,
            options=build_options,
        ),
    )
    print(f"Electrons={mol.nelec}, AOs={mol.nao}")
    eri_factors = getattr(mol, "eri_factors", None)
    build_info = getattr(mol, "_builtin_build_info", {}) or {}
    ri_info = dict(build_info.get("ri", {}) or {})
    build_timings = dict(build_info.get("timings", {}) or {})
    if eri_factors is not None:
        print(f"RI/low-rank factors: rank={eri_factors.shape[0]}")
    if ri_info:
        print(
            "RI diagnostics: "
            f"builder={ri_info.get('tensor_builder')}, "
            f"workers={ri_info.get('workers')}, "
            f"cache_hit={ri_info.get('cache_hit')}"
        )

    if args.build_only:
        summary = {
            "model": "xyz" if args.xyz else "generated_fe_bpy3_repaired",
            "natom": len(atoms),
            "charge": args.charge,
            "spin": args.spin,
            "basis": args.basis,
            "integral_engine": "native",
            "eri": args.eri,
            "auxbasis": args.auxbasis,
            "rhf_density_fit": bool(args.rhf_density_fit),
            "build_only": True,
            "build_options": build_options,
            "nelectron": int(mol.nelec),
            "nao": int(mol.nao),
            "build_seconds": float(build_seconds),
            "ri_rank": None if eri_factors is None else int(eri_factors.shape[0]),
            "ri_info": ri_info,
            "build_timings": build_timings,
        }
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"\nWrote {args.out}")
        return 0

    dm0 = None
    init_guess = args.init_guess
    init_guess_info = {"source": "pyqed", "key": args.init_guess}

    mf, rhf_seconds = timed(
        "PyQED RHF",
        lambda: RHF(mol).run(
            dm0=dm0,
            density_fit=args.rhf_density_fit,
            auxbasis=args.auxbasis,
            max_cycle=args.max_cycle,
            tol=args.tol,
            conv_tol_dm=args.conv_tol_dm,
            conv_tol_grad=args.conv_tol_grad,
            damping=args.damping,
            damping_mode=args.damping_mode,
            damping_decay=args.damping_decay,
            damping_decay_start=args.damping_decay_start,
            damping_min=args.damping_min,
            level_shift=args.level_shift,
            level_shift_decay=args.level_shift_decay,
            level_shift_decay_start=args.level_shift_decay_start,
            level_shift_min=args.level_shift_min,
            scf_diis=args.scf_diis,
            diis_switch_tol=args.diis_switch_tol,
            diis_start_cycle=args.diis_start_cycle,
            diis_space=args.diis_space,
            init_guess=init_guess,
            verbose=args.verbose,
        ),
    )
    print(f"RHF converged={bool(getattr(mf, 'converged', False))}")
    print(f"E = {mf.e_tot:.12f} Ha")
    scf_info = dict(getattr(mf, "scf_info", {}) or {})
    if scf_info:
        print(
            "SCF diagnostics: "
            f"iterations={scf_info.get('iterations')}, "
            f"dE={scf_info.get('last_energy_change'):.3e}, "
            f"dD={scf_info.get('last_density_change'):.3e}, "
            f"DIIS={scf_info.get('last_diis_error'):.3e}, "
            f"|g|={scf_info.get('final_orbital_gradient_rms'):.3e}"
        )

    homo, frontier = frontier_summary(mf)
    print_frontier(frontier, homo)

    summary = {
        "model": "xyz" if args.xyz else "generated_fe_bpy3_repaired",
        "natom": len(atoms),
        "charge": args.charge,
        "spin": args.spin,
        "basis": args.basis,
        "integral_engine": "native",
        "eri": args.eri,
        "auxbasis": args.auxbasis,
        "rhf_density_fit": bool(args.rhf_density_fit),
        "build_options": build_options,
        "nelectron": int(mol.nelec),
        "nao": int(mol.nao),
        "build_seconds": float(build_seconds),
        "rhf_seconds": float(rhf_seconds),
        "init_guess": init_guess_info,
        "converged": bool(getattr(mf, "converged", False)),
        "energy_ha": float(mf.e_tot),
        "scf_info": scf_info,
        "homo": int(homo),
        "frontier": frontier,
        "ri_rank": None if eri_factors is None else int(eri_factors.shape[0]),
        "ri_info": ri_info,
        "build_timings": build_timings,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
