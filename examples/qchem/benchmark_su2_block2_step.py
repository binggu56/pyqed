#!/usr/bin/env python3
"""Step-by-step SU(2) DMRG comparison between PyQED and block2."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from pyblock2._pyscf.ao2mo import integrals as block2_integrals
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import gto, mcscf, scf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.hf import RHF


PRESETS = {
    "h4": {
        "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        "ncas": 4,
        "nelecas": 4,
    },
    "h6": {
        "atom": "H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8; H 0 0 6.4; H 0 0 8.0",
        "ncas": 6,
        "nelecas": 6,
    },
}


def scalar_energy(value):
    """Return the first scalar energy from a possible root array."""

    return float(np.asarray(value).reshape(-1)[0])


def run_pyscf_casci(case, basis):
    """Build the PySCF reference molecule, RHF, and exact CASCI."""

    mol = gto.M(
        atom=case["atom"],
        basis=basis,
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.kernel()
    mc = mcscf.CASCI(mf, case["ncas"], case["nelecas"])
    e_casci = float(mc.kernel()[0])
    return mol, mf, mc, e_casci


def run_pyqed(case, basis, *, symmetry, bond_dim, nsweeps, max_bond_mode):
    """Run a PyQED DMRG calculation and return diagnostics."""

    mol = Molecule(atom=case["atom"], basis=basis, unit="bohr")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    dmrg = DMRG(
        mf,
        ncas=case["ncas"],
        nelecas=case["nelecas"],
        D=bond_dim,
        init_guess="cid",
        verbose=0,
        symmetry=symmetry,
    )
    t0 = time.perf_counter()
    dmrg.run(nsweeps=nsweeps, max_bond_mode=max_bond_mode)
    elapsed = time.perf_counter() - t0
    history = getattr(getattr(dmrg, "dmrg", None), "history", []) or []
    objectives = [
        objective
        for entry in history
        for objective in entry.get("bond_objectives", [])
    ]
    return {
        "energy": scalar_energy(dmrg.e_tot),
        "time": elapsed,
        "sweep_energies": [
            scalar_energy(entry["energy"])
            for entry in history
            if entry.get("energy") is not None
        ],
        "local_problems": Counter(
            objective.get("effective_local_problem")
            for objective in objectives
        ),
        "preconditioners": Counter(
            objective.get("preconditioner_mode")
            for objective in objectives
            if "preconditioner_mode" in objective
        ),
        "iterations": [
            objective.get("davidson_iterations")
            for objective in objectives
            if "davidson_iterations" in objective
        ],
    }


def run_pyqed_su2_strict(case, basis, *, bond_dim, nsweeps):
    """Run PyQED SU(2) with stricter local solves and no mixer noise."""

    mol = Molecule(atom=case["atom"], basis=basis, unit="bohr")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    dmrg = DMRG(
        mf,
        ncas=case["ncas"],
        nelecas=case["nelecas"],
        D=bond_dim,
        init_guess="cid",
        verbose=0,
        symmetry="su2",
    )
    t0 = time.perf_counter()
    dmrg.run(
        nsweeps=nsweeps,
        max_bond_mode="reduced",
        warm_start_bonds=True,
        mixer_zero_block_noise_scale=0.0,
        mixer_nsweeps=0,
        local_solver_kwargs={
            "tol": 1.0e-10,
            "tol_residual": 1.0e-8,
            "itermax": 160,
            "max_space": 96,
            "orthonormalized_dense_dim": None,
            "orthonormalize_generalized_dim": None,
            "use_block_preconditioner": False,
        },
    )
    elapsed = time.perf_counter() - t0
    history = getattr(getattr(dmrg, "dmrg", None), "history", []) or []
    objectives = [
        objective
        for entry in history
        for objective in entry.get("bond_objectives", [])
    ]
    return {
        "energy": scalar_energy(dmrg.e_tot),
        "time": elapsed,
        "sweep_energies": [
            scalar_energy(entry["energy"])
            for entry in history
            if entry.get("energy") is not None
        ],
        "local_problems": Counter(
            objective.get("effective_local_problem")
            for objective in objectives
        ),
        "preconditioners": Counter(
            objective.get("preconditioner_mode")
            for objective in objectives
            if "preconditioner_mode" in objective
        ),
        "iterations": [
            objective.get("davidson_iterations")
            for objective in objectives
            if "davidson_iterations" in objective
        ],
        "residuals": [
            objective.get("residual")
            for objective in objectives
            if "residual" in objective
        ],
    }


def run_block2_su2(pyscf_mf, pyscf_mc, *, bond_dim, nsweeps):
    """Run native pyblock2 SU(2) DMRG from the same PySCF active integrals."""

    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = block2_integrals.get_rhf_integrals(
        pyscf_mf,
        pyscf_mc.ncore,
        pyscf_mc.ncas,
        g2e_symm=1,
    )
    with tempfile.TemporaryDirectory(prefix="block2_su2_step_") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=1,
        )
        driver.initialize_system(
            n_sites=ncas,
            n_elec=n_elec,
            spin=spin,
            orb_sym=orb_sym,
        )
        mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore, iprint=0)
        ket = driver.get_random_mps(
            tag="KET",
            bond_dim=bond_dim,
            nroots=1,
        )
        t0 = time.perf_counter()
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=nsweeps,
            bond_dims=[bond_dim] * nsweeps,
            noises=[1.0e-6, 1.0e-7] + [0.0] * max(0, nsweeps - 2),
            tol=1.0e-9,
            iprint=0,
            dav_max_iter=100,
            dav_def_max_size=50,
        )
        elapsed = time.perf_counter() - t0
    return {"energy": float(energy), "time": elapsed}


def run_block2_su2_stepwise(pyscf_mf, pyscf_mc, *, bond_dim, nsweeps):
    """Run block2 one sweep at a time and report sweep-resolved energies."""

    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = block2_integrals.get_rhf_integrals(
        pyscf_mf,
        pyscf_mc.ncore,
        pyscf_mc.ncas,
        g2e_symm=1,
    )
    with tempfile.TemporaryDirectory(prefix="block2_su2_step_") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=1,
        )
        driver.initialize_system(
            n_sites=ncas,
            n_elec=n_elec,
            spin=spin,
            orb_sym=orb_sym,
        )
        mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore, iprint=0)
        ket = driver.get_random_mps(
            tag="KET",
            bond_dim=bond_dim,
            nroots=1,
        )
        energies = []
        t0 = time.perf_counter()
        for sweep in range(nsweeps):
            _ = sweep
            energy = driver.dmrg(
                mpo,
                ket,
                n_sweeps=1,
                bond_dims=[bond_dim],
                noises=[0.0],
                tol=1.0e-9,
                iprint=0,
                dav_max_iter=100,
                dav_def_max_size=50,
            )
            energies.append(float(energy))
        elapsed = time.perf_counter() - t0
    return {
        "energy": energies[-1],
        "time": elapsed,
        "sweep_energies": energies,
    }


def print_result(label, result, reference):
    """Print one aligned benchmark result."""

    print(
        f"{label:<22} {result['energy']: .12f} "
        f"dE={result['energy'] - reference:+.3e} "
        f"time={result['time']:.3f}s"
    )
    if result.get("sweep_energies"):
        print(f"  sweeps       {result['sweep_energies']}")
    if result.get("local_problems"):
        print(f"  local        {dict(result['local_problems'])}")
    if result.get("preconditioners"):
        print(f"  precond      {dict(result['preconditioners'])}")
    if result.get("iterations"):
        print(f"  iterations   {result['iterations']}")
    if result.get("residuals"):
        print(f"  residuals    {result['residuals']}")


def main():
    """Run the step-by-step SU(2) comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(PRESETS), default="h4")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--nsweeps", type=int, default=4)
    args = parser.parse_args()

    case = PRESETS[args.system]
    print(
        f"{args.system.upper()} {args.basis} "
        f"CAS({case['nelecas']},{case['ncas']}), D={args.D}, "
        f"nsweeps={args.nsweeps}"
    )
    pyscf_mol, pyscf_mf, pyscf_mc, e_casci = run_pyscf_casci(case, args.basis)
    _ = pyscf_mol
    print(f"PySCF RHF              {float(pyscf_mf.e_tot): .12f}")
    print(f"PySCF CASCI            {e_casci: .12f}")

    print_result(
        "PyQED SZ",
        run_pyqed(
            case,
            args.basis,
            symmetry="sz",
            bond_dim=args.D,
            nsweeps=args.nsweeps,
            max_bond_mode="states",
        ),
        e_casci,
    )
    print_result(
        "PyQED SU2 states-D",
        run_pyqed(
            case,
            args.basis,
            symmetry="su2",
            bond_dim=args.D,
            nsweeps=args.nsweeps,
            max_bond_mode="states",
        ),
        e_casci,
    )
    print_result(
        "PyQED SU2 reduced-D",
        run_pyqed(
            case,
            args.basis,
            symmetry="su2",
            bond_dim=args.D,
            nsweeps=args.nsweeps,
            max_bond_mode="reduced",
        ),
        e_casci,
    )
    print_result(
        "PyQED SU2 strict",
        run_pyqed_su2_strict(
            case,
            args.basis,
            bond_dim=args.D,
            nsweeps=args.nsweeps,
        ),
        e_casci,
    )
    print_result(
        "PyQED SU2 block2-D",
        run_pyqed(
            case,
            args.basis,
            symmetry="su2",
            bond_dim=args.D,
            nsweeps=args.nsweeps,
            max_bond_mode="per_sector",
        ),
        e_casci,
    )
    print_result(
        "block2 SU2",
        run_block2_su2(pyscf_mf, pyscf_mc, bond_dim=args.D, nsweeps=args.nsweeps),
        e_casci,
    )
    print_result(
        "block2 SU2 stepwise",
        run_block2_su2_stepwise(
            pyscf_mf,
            pyscf_mc,
            bond_dim=args.D,
            nsweeps=args.nsweeps,
        ),
        e_casci,
    )


if __name__ == "__main__":
    main()
