#!/usr/bin/env python3
"""Single-geometry GDVR RHF calculation for linear Li_n chains."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyqed.qchem.gdvr import Molecule, local_ecp_terms_from_pyscf


def _chain_coords(n_atoms, spacing_bohr):
    z = (np.arange(int(n_atoms), dtype=float) - 0.5 * (int(n_atoms) - 1)) * float(spacing_bohr)
    return [(0.0, 0.0, float(zi)) for zi in z]


def _build_model(args):
    n_atoms = int(args.n_atoms)
    coords = _chain_coords(n_atoms, args.spacing)
    if args.core_model == "valence-charge":
        return Molecule([1.0] * n_atoms, coords=coords, nelec=n_atoms, spin=0)

    if args.core_model in ("real-ecp-scalar", "real-ecp-semilocal"):
        scalarize = args.core_model == "real-ecp-scalar"
        ecp = local_ecp_terms_from_pyscf("Li", args.ecp_name, scalarize_nonlocal=scalarize)
        slice_ecp = (
            local_ecp_terms_from_pyscf("Li", args.ecp_name, scalarize_nonlocal=True)
            if args.core_model == "real-ecp-semilocal"
            else ecp
        )
        valence_charge = float(3 - int(ecp["core_electrons"]))
        return Molecule(
            [valence_charge] * n_atoms,
            coords=coords,
            nelec=n_atoms,
            spin=0,
            basis_charges=[3.0] * n_atoms,
            local_ecp_terms=[ecp["local_terms"]] * n_atoms,
            semilocal_ecp_terms=(
                [ecp["semilocal_terms"]] * n_atoms
                if args.core_model == "real-ecp-semilocal"
                else None
            ),
            slice_local_ecp_terms=(
                [slice_ecp["local_terms"]] * n_atoms
                if args.core_model == "real-ecp-semilocal"
                else None
            ),
            ecp_metadata={
                "ecp_name": str(args.ecp_name),
                "source": "pyscf.gto.basis.load_ecp",
                "approximation": (
                    "semilocal projectors retained as nonlocal GDVR blocks"
                    if args.core_model == "real-ecp-semilocal"
                    else "semilocal radial terms folded into the scalar GDVR potential"
                ),
            },
        )

    raise ValueError(f"unknown core model {args.core_model!r}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-atoms", type=int, default=10)
    parser.add_argument("--spacing", type=float, default=4.0, help="Nearest-neighbor Li-Li spacing in bohr.")
    parser.add_argument("--lz", type=float, default=26.0)
    parser.add_argument("--nz", type=int, default=52)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--transverse-basis", default="sto3g")
    parser.add_argument("--dvr-method", choices=("sine", "exp", "sinc"), default="exp")
    parser.add_argument("--core-model", choices=("valence-charge", "real-ecp-scalar", "real-ecp-semilocal"), default="real-ecp-semilocal")
    parser.add_argument("--ecp-name", default="bfd")
    parser.add_argument("--scf-conv", type=float, default=1e-7)
    parser.add_argument("--scf-max-iter", type=int, default=100)
    parser.add_argument("--transverse-opt", action="store_true")
    parser.add_argument("--transverse-opt-cycles", type=int, default=10)
    parser.add_argument("--transverse-opt-sweeps", type=int, default=1)
    parser.add_argument("--transverse-opt-tol", type=float, default=1e-7)
    parser.add_argument("--transverse-opt-ridge", type=float, default=0.5)
    parser.add_argument("--transverse-opt-step", type=float, default=0.5)
    parser.add_argument("--transverse-opt-radius", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("/private/tmp/gdvr_li_chain.json"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if int(args.n_atoms) <= 0:
        raise ValueError("--n-atoms must be positive.")
    if int(args.n_atoms) % 2:
        raise ValueError("This RHF chain driver currently expects an even number of atoms/electrons.")
    if args.transverse_opt and int(args.m) != 1:
        raise ValueError("--transverse-opt currently requires --m 1.")

    mol = _build_model(args)
    total_start = time.perf_counter()
    build_start = time.perf_counter()
    mol.build(
        Lz=float(args.lz),
        Nz=int(args.nz),
        M=int(args.m),
        transverse_basis=args.transverse_basis,
        dvr_method=args.dvr_method,
        verbose=bool(args.verbose),
    )
    build_seconds = time.perf_counter() - build_start

    scf_start = time.perf_counter()
    mf = mol.RHF().run(
        newton=False,
        conv=float(args.scf_conv),
        max_iter=int(args.scf_max_iter),
        verbose=bool(args.verbose),
    )
    scf_seconds = time.perf_counter() - scf_start
    e_before_transverse_opt = None
    transverse_opt_seconds = 0.0
    if args.transverse_opt:
        e_before_transverse_opt = float(mf.e_tot)
        opt_start = time.perf_counter()
        mf.newton(
            tol=float(args.transverse_opt_tol),
            max_cycles=int(args.transverse_opt_cycles),
            sweeps=int(args.transverse_opt_sweeps),
            ridge=float(args.transverse_opt_ridge),
            trust_step=float(args.transverse_opt_step),
            trust_radius=float(args.transverse_opt_radius),
            scf_conv=float(args.scf_conv),
            scf_max_iter=int(args.scf_max_iter),
            verbose=bool(args.verbose),
        )
        transverse_opt_seconds = time.perf_counter() - opt_start

    nocc = int(mol.nelec // 2)
    result = {
        "n_atoms": int(args.n_atoms),
        "spacing_bohr": float(args.spacing),
        "z_range_bohr": [float(mol.coords[:, 2].min()), float(mol.coords[:, 2].max())],
        "grid": {
            "Lz_bohr": float(args.lz),
            "Nz": int(args.nz),
            "M": int(args.m),
            "dvr_method": str(args.dvr_method),
            "transverse_basis": str(args.transverse_basis),
            "eri_offsets": "full",
        },
        "core_model": str(args.core_model),
        "ecp_name": str(args.ecp_name) if args.core_model.startswith("real-ecp") else "",
        "nelec": int(mol.nelec),
        "E_RHF_Ha": float(mf.e_tot),
        "E_before_transverse_opt_Ha": e_before_transverse_opt,
        "E_per_atom_Ha": float(mf.e_tot / int(args.n_atoms)),
        "E_nuc_Ha": float(mol.nuclear_repulsion_energy()),
        "HOMO_Ha": float(mf.mo_energy[nocc - 1]),
        "LUMO_Ha": float(mf.mo_energy[nocc]),
        "gap_Ha": float(mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]),
        "scf_info": dict(mf.info),
        "transverse_opt": {
            "enabled": bool(args.transverse_opt),
            "seconds": float(transverse_opt_seconds),
            "cycles": int(mf.info.get("newton_cycles", 0)) if args.transverse_opt else 0,
            "converged": bool(mf.info.get("newton_converged", False)) if args.transverse_opt else False,
            "history_Ha": [float(x) for x in mf.info.get("newton_energy_history", [])] if args.transverse_opt else [],
        },
        "timing_seconds": {
            "build": float(build_seconds),
            "scf": float(scf_seconds),
            "total": float(time.perf_counter() - total_start),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"E = {mf.e_tot:.12f} Ha")
    print(f"E/atom = {mf.e_tot / int(args.n_atoms):.12f} Ha")
    print(f"gap = {result['gap_Ha']:.8f} Ha")
    print(f"build = {build_seconds:.2f}s  scf = {scf_seconds:.2f}s  opt = {transverse_opt_seconds:.2f}s")
    print(f"Saved summary: {args.out}")


if __name__ == "__main__":
    main()
