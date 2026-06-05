"""Benchmark native PyQED PCM-RHF against PySCF.

The default H2O/STO-3G case is intentionally tiny but strict: PyQED uses the
same PySCF AO ordering for the molecular integrals, then evaluates the PCM
reaction field with either the native or PySCF-backed surface integral path.
This separates SCF wiring regressions from basis-ordering noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem import Molecule, RHF
from pyqed.qchem.solvent import PCM as attach_pcm
from pyqed.qchem.solvent.pcm import PCM


MOLECULES = {
    "h2o": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    "lih": "Li 0 0 0; H 0 0 1.6",
    "h2o2": (
        "O 0.000000 0.000000 0.000000; "
        "O 1.450000 0.000000 0.000000; "
        "H -0.450000 0.760000 0.000000; "
        "H 1.900000 0.760000 0.600000"
    ),
}


def timed(label, func):
    t0 = time.perf_counter()
    out = func()
    seconds = time.perf_counter() - t0
    print(f"{label} finished in {seconds:.3f} s", flush=True)
    return out, seconds


def configured_pyqed_pcm(mol, args, backend):
    pcm = PCM(mol)
    pcm.eps = args.eps
    pcm.method = args.method
    pcm.lebedev_order = args.lebedev_order
    pcm.integral_backend = backend
    pcm.max_memory = args.max_memory
    pcm.verbose = args.verbose
    return pcm


def run_pyscf(atom, args):
    from pyscf import gto, scf
    from pyscf.solvent import pcm as pyscf_pcm

    pmol = gto.M(
        atom=atom,
        unit=args.unit,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        verbose=0,
    )
    gas = scf.RHF(pmol)
    gas.conv_tol = args.tol
    gas.max_cycle = args.max_cycle
    gas.verbose = args.verbose
    e_gas = gas.kernel()
    dm_gas = gas.make_rdm1()

    frozen_solvent = pyscf_pcm.PCM(pmol)
    frozen_solvent.eps = args.eps
    frozen_solvent.method = args.method
    frozen_solvent.lebedev_order = args.lebedev_order
    frozen_solvent.verbose = args.verbose
    e_frozen, v_frozen = frozen_solvent.kernel(dm_gas)

    pcm_mf = scf.RHF(pmol).PCM()
    pcm_mf.conv_tol = args.tol
    pcm_mf.max_cycle = args.max_cycle
    pcm_mf.verbose = args.verbose
    pcm_mf.with_solvent.eps = args.eps
    pcm_mf.with_solvent.method = args.method
    pcm_mf.with_solvent.lebedev_order = args.lebedev_order
    pcm_mf.with_solvent.verbose = args.verbose
    e_pcm = pcm_mf.kernel(dm0=dm_gas)

    return {
        "gas_energy_ha": float(e_gas),
        "gas_converged": bool(gas.converged),
        "frozen_pcm_energy_ha": float(e_frozen),
        "frozen_total_energy_ha": float(e_gas + e_frozen),
        "frozen_v_norm": float(np.linalg.norm(v_frozen)),
        "pcm_energy_ha": float(e_pcm),
        "pcm_converged": bool(pcm_mf.converged),
        "pcm_solvent_energy_ha": float(pcm_mf.with_solvent.e),
        "pcm_v_norm": float(np.linalg.norm(pcm_mf.with_solvent.v)),
        "pcm_ngrids": int(len(pcm_mf.with_solvent.surface["grid_coords"])),
        "pcm_iterations": int(getattr(pcm_mf, "cycles", -1)),
    }


def run_pyqed(atom, args, backend):
    mol = Molecule(
        atom=atom,
        unit=args.unit.lower(),
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
    )
    build_kwargs = {"driver": args.pyqed_driver}
    if str(args.pyqed_driver).lower() in {"builtin", "native"}:
        build_kwargs["eri"] = args.eri
    mol.build(**build_kwargs)

    gas = RHF(mol).run(
        max_cycle=args.max_cycle,
        tol=args.tol,
        conv_tol_grad=args.conv_tol_grad,
        init_guess=args.init_guess,
        verbose=args.verbose,
    )

    frozen_pcm = configured_pyqed_pcm(mol, args, backend)
    e_frozen, v_frozen = frozen_pcm.kernel(gas.dm)

    pcm = configured_pyqed_pcm(mol, args, backend)
    pcm_mf = attach_pcm(RHF(mol), pcm).run(
        dm0=gas.dm,
        init_guess="dm",
        max_cycle=args.max_cycle,
        tol=args.tol,
        conv_tol_grad=args.conv_tol_grad,
        verbose=args.verbose,
    )

    return {
        "backend": backend,
        "gas_energy_ha": float(gas.e_tot),
        "gas_converged": bool(gas.converged),
        "frozen_pcm_energy_ha": float(e_frozen),
        "frozen_total_energy_ha": float(gas.e_tot + e_frozen),
        "frozen_v_norm": float(np.linalg.norm(v_frozen)),
        "pcm_energy_ha": float(pcm_mf.e_tot),
        "pcm_converged": bool(pcm_mf.converged),
        "pcm_solvent_energy_ha": float(pcm_mf.scf_info["solvent_energy"]),
        "pcm_v_norm": float(pcm_mf.scf_info["solvent_potential_norm"]),
        "pcm_ngrids": int(pcm_mf.scf_info["solvent_ngrids"]),
        "pcm_iterations": int(pcm_mf.scf_info["iterations"]),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=sorted(MOLECULES), default="h2o")
    parser.add_argument("--atom", help="Override molecule with a semicolon-separated atom string.")
    parser.add_argument("--unit", default="Angstrom")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--pyqed-driver", default="pyscf")
    parser.add_argument("--eri", default="s8")
    parser.add_argument("--pcm-backends", default="native,pyscf")
    parser.add_argument("--eps", type=float, default=35.688)
    parser.add_argument("--method", default="C-PCM")
    parser.add_argument("--lebedev-order", type=int, default=3)
    parser.add_argument("--max-memory", type=float, default=1000.0)
    parser.add_argument("--max-cycle", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--conv-tol-grad", type=float, default=1e-7)
    parser.add_argument("--init-guess", default="hcore")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    atom = args.atom if args.atom else MOLECULES[args.molecule]
    backends = [x.strip().lower() for x in args.pcm_backends.split(",") if x.strip()]

    print(
        f"Benchmark: molecule={args.molecule if args.atom is None else 'custom'}, "
        f"basis={args.basis}, method={args.method}, eps={args.eps}, "
        f"lebedev_order={args.lebedev_order}"
    )

    pyscf_result, pyscf_seconds = timed("PySCF RHF/PCM", lambda: run_pyscf(atom, args))
    pyqed_results = {}
    pyqed_seconds = {}
    for backend in backends:
        result, seconds = timed(
            f"PyQED RHF/PCM backend={backend}",
            lambda backend=backend: run_pyqed(atom, args, backend),
        )
        pyqed_results[backend] = result
        pyqed_seconds[backend] = seconds

    print("\nquantity                    PySCF", end="")
    for backend in backends:
        print(f"        PyQED/{backend:>6}      delta", end="")
    print()
    for key in (
        "gas_energy_ha",
        "frozen_pcm_energy_ha",
        "frozen_total_energy_ha",
        "pcm_energy_ha",
        "pcm_solvent_energy_ha",
    ):
        print(f"{key:26s} {pyscf_result[key]:16.10f}", end="")
        for backend in backends:
            value = pyqed_results[backend][key]
            print(f" {value:16.10f} {value - pyscf_result[key]:10.3e}", end="")
        print()

    print("\nstatus")
    print(
        f"PySCF: gas_converged={pyscf_result['gas_converged']}, "
        f"pcm_converged={pyscf_result['pcm_converged']}, "
        f"ngrids={pyscf_result['pcm_ngrids']}, seconds={pyscf_seconds:.3f}"
    )
    for backend in backends:
        result = pyqed_results[backend]
        print(
            f"PyQED/{backend}: gas_converged={result['gas_converged']}, "
            f"pcm_converged={result['pcm_converged']}, "
            f"ngrids={result['pcm_ngrids']}, seconds={pyqed_seconds[backend]:.3f}"
        )

    summary = {
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "pyscf": pyscf_result,
        "pyqed": pyqed_results,
        "seconds": {"pyscf": pyscf_seconds, "pyqed": pyqed_seconds},
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
