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
    "h8": {
        "atom": "; ".join(f"H 0 0 {1.6 * i}" for i in range(8)),
        "ncas": 8,
        "nelecas": 8,
    },
    "h10": {
        "atom": "; ".join(f"H 0 0 {1.6 * i}" for i in range(10)),
        "ncas": 10,
        "nelecas": 10,
    },
    "h12": {
        "atom": "; ".join(f"H 0 0 {1.6 * i}" for i in range(12)),
        "ncas": 12,
        "nelecas": 12,
    },
    "h14": {
        "atom": "; ".join(f"H 0 0 {1.6 * i}" for i in range(14)),
        "ncas": 14,
        "nelecas": 14,
    },
}


def build_pyqed_cpp_molecule(case, basis):
    """Build PyQED AO integrals with the compiled C++ backend."""

    mol = Molecule(atom=case["atom"], basis=basis, unit="bohr")
    mol.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    build_info = mol._builtin_build_info
    if (
        build_info.get("eri_backend") != "cpp"
        or not str(build_info.get("dense_builder", "")).startswith("cpp-")
    ):
        raise RuntimeError("The SU(2)/block2 benchmark requires the compiled C++ ERI backend.")
    return mol


def scalar_energy(value):
    """Return the first scalar energy from a possible root array."""

    return float(np.asarray(value).reshape(-1)[0])


def run_pyscf_casci(case, basis):
    """Build an independent PySCF reference molecule, RHF, and exact CASCI."""

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


def build_pyqed_cpp_active_hamiltonian(case, basis):
    """Return block2-ready active tensors derived from PyQED's C++ AO ERIs."""

    mol = build_pyqed_cpp_molecule(case, basis)
    mf = RHF(mol).run()
    dmrg = DMRG(
        mf,
        ncas=case["ncas"],
        nelecas=case["nelecas"],
        D=1,
        init_guess="cid",
        verbose=0,
        symmetry="su2",
    )
    dmrg.build()

    h1e_spin = np.asarray(dmrg.h1e)
    g2e_spin = np.asarray(dmrg.h2e)
    if h1e_spin.shape != (2, dmrg.ncas, dmrg.ncas):
        raise RuntimeError(f"Unexpected PyQED one-body tensor shape {h1e_spin.shape}.")
    if g2e_spin.shape != (2, 2, dmrg.ncas, dmrg.ncas, dmrg.ncas, dmrg.ncas):
        raise RuntimeError(f"Unexpected PyQED two-body tensor shape {g2e_spin.shape}.")
    if not np.allclose(h1e_spin[0], h1e_spin[1]):
        raise RuntimeError("The SU(2) benchmark requires spin-independent one-body integrals.")
    if not all(
        np.allclose(g2e_spin[a, b], g2e_spin[0, 0])
        for a in range(2)
        for b in range(2)
    ):
        raise RuntimeError("The SU(2) benchmark requires spin-independent two-body integrals.")

    return {
        "ncas": int(dmrg.ncas),
        "n_elec": int(dmrg.nelecas),
        "spin": int(dmrg.spin),
        "ecore": float(dmrg.e_core),
        "h1e": np.ascontiguousarray(h1e_spin[0]),
        "g2e": np.ascontiguousarray(g2e_spin[0, 0]),
        "orb_sym": [0] * int(dmrg.ncas),
        "rhf_energy": scalar_energy(mf.e_tot),
    }


def run_pyqed(
    case,
    basis,
    *,
    symmetry,
    bond_dim,
    nsweeps,
    max_bond_mode=None,
):
    """Run a PyQED DMRG calculation and return diagnostics."""

    mol = build_pyqed_cpp_molecule(case, basis)
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
    # Match block2's timer, which starts after its Hamiltonian MPO is built.
    dmrg.build()
    t0 = time.perf_counter()
    run_kwargs = {
        "nsweeps": nsweeps,
        "conv_tol": -1.0,
        "require_convergence": False,
    }
    if max_bond_mode is not None:
        run_kwargs["max_bond_mode"] = max_bond_mode
    dmrg.run(**run_kwargs)
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
            if entry.get("energy") is not None and entry.get("sweep_complete")
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



def run_block2_su2(active, *, bond_dim, nsweeps, seed=1234):
    """Run block2 SU(2) DMRG on PyQED's C++-integral Hamiltonian."""

    half_sweeps = 2 * int(nsweeps)
    with tempfile.TemporaryDirectory(prefix="block2_su2_step_") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=1,
        )
        driver.bw.b.Random.rand_seed(int(seed))
        driver.initialize_system(
            n_sites=active["ncas"],
            n_elec=active["n_elec"],
            spin=active["spin"],
            orb_sym=active["orb_sym"],
        )
        mpo = driver.get_qc_mpo(
            active["h1e"],
            active["g2e"],
            ecore=active["ecore"],
            iprint=0,
        )
        ket = driver.get_random_mps(
            tag="KET",
            bond_dim=bond_dim,
            nroots=1,
        )
        t0 = time.perf_counter()
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=half_sweeps,
            bond_dims=[bond_dim] * half_sweeps,
            noises=[1.0e-6, 1.0e-7] + [0.0] * max(0, half_sweeps - 2),
            tol=1.0e-9,
            iprint=0,
            dav_max_iter=100,
            dav_def_max_size=50,
        )
        elapsed = time.perf_counter() - t0
    return {"energy": float(energy), "time": elapsed}


def run_block2_su2_stepwise(active, *, bond_dim, nsweeps, seed=1234):
    """Run block2 one sweep at a time on PyQED's C++-integral Hamiltonian."""

    with tempfile.TemporaryDirectory(prefix="block2_su2_step_") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=1,
        )
        driver.bw.b.Random.rand_seed(int(seed))
        driver.initialize_system(
            n_sites=active["ncas"],
            n_elec=active["n_elec"],
            spin=active["spin"],
            orb_sym=active["orb_sym"],
        )
        mpo = driver.get_qc_mpo(
            active["h1e"],
            active["g2e"],
            ecore=active["ecore"],
            iprint=0,
        )
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
                n_sweeps=2,
                bond_dims=[bond_dim, bond_dim],
                noises=[0.0, 0.0],
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
    active = build_pyqed_cpp_active_hamiltonian(case, args.basis)
    print(f"PyQED C++ RHF          {active['rhf_energy']: .12f}")
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
        "PyQED SU2 native C++",
        run_pyqed(
            case,
            args.basis,
            symmetry="su2",
            bond_dim=args.D,
            nsweeps=args.nsweeps,
        ),
        e_casci,
    )
    print_result(
        "block2 SU2",
        run_block2_su2(active, bond_dim=args.D, nsweeps=args.nsweeps),
        e_casci,
    )
    print_result(
        "block2 SU2 stepwise",
        run_block2_su2_stepwise(
            active,
            bond_dim=args.D,
            nsweeps=args.nsweeps,
        ),
        e_casci,
    )


if __name__ == "__main__":
    main()
