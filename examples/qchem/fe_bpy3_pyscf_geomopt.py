"""Standalone PySCF RI geometry optimization for [Fe(bpy)3]2+.

This optional driver is intentionally outside pyqed's package code. It provides
a lightweight Cartesian L-BFGS-B optimizer when the usual PySCF geometry
optimizers (geomeTRIC/pyberny) are unavailable.

The generated geometry is only an initial scaffold.  Prefer ``--xyz`` with a
reasonable experimental or pre-optimized structure for production work.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from pyqed.units import angstrom2au

BOHR_PER_ANG = angstrom2au
ANG_PER_BOHR = 1.0 / BOHR_PER_ANG


def _load_fe_helpers():
    helper = Path(__file__).with_name("fe_bpy3_pyscf_ri_feasibility.py")
    spec = importlib.util.spec_from_file_location("fe_bpy3_pyscf_ri_feasibility", helper)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_xyz(path, symbols, coords_ang, comment):
    path = Path(path)
    with path.open("w") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for sym, (x, y, z) in zip(symbols, coords_ang):
            handle.write(f"{sym:2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")


def make_mol(symbols, coords_bohr, args):
    from pyscf import gto

    atom = [(sym, tuple(coord)) for sym, coord in zip(symbols, coords_bohr)]
    return gto.M(
        atom=atom,
        unit="Bohr",
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        verbose=0,
    )


def make_scf(mol, args):
    from pyscf import dft, scf

    reference = args.reference
    if reference == "auto":
        if args.method == "hf":
            reference = "rhf" if mol.spin == 0 else "rohf"
        else:
            reference = "rks" if mol.spin == 0 else "uks"

    if reference == "rhf":
        mf = scf.RHF(mol)
    elif reference == "rohf":
        mf = scf.ROHF(mol)
    elif reference == "uhf":
        mf = scf.UHF(mol)
    elif reference == "rks":
        mf = dft.RKS(mol)
        mf.xc = args.xc
        mf.grids.level = args.grid_level
    elif reference == "uks":
        mf = dft.UKS(mol)
        mf.xc = args.xc
        mf.grids.level = args.grid_level
    else:
        raise ValueError(f"Unsupported reference {reference!r}")

    if args.density_fit:
        mf = mf.density_fit(auxbasis=args.auxbasis)
    mf.max_cycle = args.scf_max_cycle
    mf.conv_tol = args.scf_conv_tol
    mf.level_shift = args.level_shift
    mf.damp = args.damping
    mf.diis_space = args.diis_space
    mf.init_guess = args.init_guess
    mf.verbose = args.verbose
    return mf


def optimize(symbols, coords_ang, args):
    coords0 = np.asarray(coords_ang, dtype=float) * BOHR_PER_ANG
    x0 = coords0.reshape(-1)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    write_xyz(outdir / "initial.xyz", symbols, np.asarray(coords_ang, dtype=float), "initial geometry")

    state = {
        "eval": 0,
        "best_energy": None,
        "best_x": x0.copy(),
        "last_dm": None,
        "history": [],
    }

    def objective(x):
        coords_bohr = np.asarray(x, dtype=float).reshape((-1, 3))
        mol = make_mol(symbols, coords_bohr, args)
        mf = make_scf(mol, args)
        t0 = time.perf_counter()
        energy = mf.kernel(dm0=state["last_dm"])
        seconds = time.perf_counter() - t0
        converged = bool(getattr(mf, "converged", False))
        if converged:
            state["last_dm"] = mf.make_rdm1()
        grad = mf.nuc_grad_method().kernel().reshape(-1)
        grad_norm = float(np.linalg.norm(grad))
        max_grad = float(np.max(np.abs(grad)))

        state["eval"] += 1
        idx = state["eval"]
        coords_out = coords_bohr * ANG_PER_BOHR
        write_xyz(
            outdir / f"step_{idx:03d}.xyz",
            symbols,
            coords_out,
            f"E={energy:.12f} Ha converged={converged} |g|={grad_norm:.6e}",
        )

        if state["best_energy"] is None or energy < state["best_energy"]:
            state["best_energy"] = float(energy)
            state["best_x"] = np.asarray(x, dtype=float).copy()
            write_xyz(
                outdir / "best.xyz",
                symbols,
                coords_out,
                f"E={energy:.12f} Ha converged={converged} |g|={grad_norm:.6e}",
            )

        state["history"].append(
            {
                "eval": idx,
                "energy_ha": float(energy),
                "converged": converged,
                "grad_norm": grad_norm,
                "max_grad": max_grad,
                "seconds": float(seconds),
            }
        )
        print(
            f"eval {idx:03d}: E={energy:.12f} Ha "
            f"conv={converged} |g|={grad_norm:.4e} maxg={max_grad:.4e} "
            f"scf={seconds:.1f}s",
            flush=True,
        )
        return float(energy), np.asarray(grad, dtype=float)

    result = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": args.maxiter,
            "maxfun": args.maxfun,
            "gtol": args.gtol,
            "ftol": args.ftol,
            "maxls": args.maxls,
            "disp": False,
        },
    )

    write_xyz(
        outdir / "final.xyz",
        symbols,
        np.asarray(result.x).reshape((-1, 3)) * ANG_PER_BOHR,
        f"E={float(result.fun):.12f} Ha success={bool(result.success)}",
    )
    summary = {
        "success": bool(result.success),
        "message": str(result.message),
        "energy_ha": float(result.fun),
        "best_energy_ha": state["best_energy"],
        "n_eval": int(state["eval"]),
        "basis": args.basis,
        "auxbasis": args.auxbasis,
        "density_fit": bool(args.density_fit),
        "method": args.method,
        "xc": args.xc if args.method == "dft" else None,
        "charge": args.charge,
        "spin": args.spin,
        "history": state["history"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return result, summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/fe_bpy3_geomopt"))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--auxbasis", default=None)
    parser.add_argument(
        "--density-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use density fitting for SCF; for small bases, --no-density-fit can make gradients faster.",
    )
    parser.add_argument("--charge", type=int, default=2)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--method", choices=("hf", "dft"), default="hf")
    parser.add_argument(
        "--reference",
        choices=("auto", "rhf", "rohf", "uhf", "rks", "uks"),
        default="auto",
    )
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--maxfun", type=int, default=80)
    parser.add_argument("--gtol", type=float, default=3e-4)
    parser.add_argument("--ftol", type=float, default=1e-7)
    parser.add_argument("--maxls", type=int, default=8)
    parser.add_argument("--write-initial-only", action="store_true")
    parser.add_argument("--scf-max-cycle", type=int, default=80)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-7)
    parser.add_argument("--level-shift", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=0.2)
    parser.add_argument("--diis-space", type=int, default=12)
    parser.add_argument("--init-guess", default="minao")
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    helper = _load_fe_helpers()
    if args.xyz:
        atoms = helper.load_xyz(args.xyz)
    else:
        atoms = helper.generated_fe_bpy3()
    symbols = [atom[0] for atom in atoms]
    coords_ang = np.array([atom[1:] for atom in atoms], dtype=float)
    print(
        f"Optimizing {len(symbols)} atoms, charge={args.charge}, spin={args.spin}, "
        f"method={args.method}, basis={args.basis}",
        flush=True,
    )
    print(f"Output directory: {args.outdir}", flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_xyz(args.outdir / "initial.xyz", symbols, coords_ang, "initial geometry")
    if args.write_initial_only:
        print(f"Wrote initial geometry: {args.outdir / 'initial.xyz'}")
        return 0
    result, summary = optimize(symbols, coords_ang, args)
    print(f"Done: success={result.success}, E={result.fun:.12f} Ha")
    print(f"Best geometry: {args.outdir / 'best.xyz'}")
    print(f"Final geometry: {args.outdir / 'final.xyz'}")
    print(f"Summary: {args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
