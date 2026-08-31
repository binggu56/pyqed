#!/usr/bin/env python3
"""Lowest 10 exact CASCI states for an H8 chain in STO-6G.

This example treats the full STO-6G valence space of neutral H8 as CAS(8e,8o),
which is exact within that basis.  The default geometry is an equally spaced
linear chain along z.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem import CASCI, Molecule


def build_h8_chain(distance_bohr: float) -> str:
    return "\n".join(f"H 0 0 {i * distance_bohr:.10f}" for i in range(8))


def run_pyscf_reference(atom: str, nstates: int):
    from pyscf import gto, mcscf, scf

    mol = gto.M(atom=atom, unit="Bohr", basis="sto-6g", spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    mc = mcscf.CASCI(mf, 8, 8)
    mc.fcisolver.nroots = nstates
    kernel_out = mc.kernel()

    e_states = getattr(mc, "e_states", None)
    if e_states is None:
        if isinstance(kernel_out, tuple) and len(kernel_out) >= 1:
            e_states = kernel_out[0]
        else:
            e_states = mc.e_tot

    e_states = list(map(float, (e_states if np.ndim(e_states) > 0 else [e_states])))
    return float(np.asarray(mf.e_tot).reshape(-1)[0]), e_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distance",
        type=float,
        default=1.8,
        help="nearest-neighbor H-H distance in bohr (default: 1.8)",
    )
    parser.add_argument(
        "--nstates",
        type=int,
        default=10,
        help="number of lowest singlet-sector states to compute (default: 10)",
    )
    parser.add_argument(
        "--use-cholesky",
        action="store_true",
        help="enable the factorized CASCI backend when available",
    )
    parser.add_argument(
        "--compare-pyscf",
        action="store_true",
        help="also compute the same CASCI roots with PySCF for comparison",
    )
    args = parser.parse_args()

    atom = build_h8_chain(args.distance)
    mol = Molecule(atom=atom, unit="bohr", basis="sto-6g", spin=0)

    build_options = None
    if args.use_cholesky:
        build_options = {"eri_representation": "factors"}

    if build_options is None:
        mol.build()
    else:
        mol.build(options=build_options)

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        mf = mol.RHF().run()
        mc = CASCI(mf, ncas=8, nelecas=8).run(
            nstates=args.nstates,
            use_cholesky=args.use_cholesky,
        )

    pyscf_hf = None
    pyscf_states = None
    pyscf_error = None
    if args.compare_pyscf:
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                pyscf_hf, pyscf_states = run_pyscf_reference(atom, args.nstates)
        except Exception as exc:  # pragma: no cover - example fallback
            pyscf_error = str(exc)

    print("H8 / STO-6G exact CASCI(8e,8o)")
    print("integrals : native")
    print(f"distance  : {args.distance:.6f} bohr")
    print(f"backend   : {getattr(mc, 'solver_backend', 'unknown')}")
    print(f"E(HF)     : {float(mf.e_tot):.12f} Ha")
    print("Lowest states:")
    for i, e in enumerate(mc.e_tot):
        print(f"  root {i:2d}  {float(e): .12f} Ha")

    if args.compare_pyscf:
        print()
        print("PySCF comparison:")
        if pyscf_error is not None:
            print(f"  failed : {pyscf_error}")
        else:
            print(f"  E(HF)  : {pyscf_hf:.12f} Ha")
            print("  Lowest states:")
            for i, e in enumerate(pyscf_states):
                delta = float(mc.e_tot[i]) - float(e)
                print(f"    root {i:2d}  {float(e): .12f} Ha   dE(pyqed-pyscf) = {delta:+.3e} Ha")


if __name__ == "__main__":
    main()
