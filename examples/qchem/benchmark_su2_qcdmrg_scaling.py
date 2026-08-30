#!/usr/bin/env python3
"""Benchmark fully reduced PyQED SU(2)-QCDMRG against block2.

Run one backend per process so ``peak_rss_mib`` is attributable to that
backend.  The hydrogen-chain presets use the complete STO-3G orbital space,
giving CAS(n,n) without an orbital-selection ambiguity.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def hydrogen_chain(n_sites: int, spacing: float) -> str:
    return "; ".join(
        f"H 0 0 {site * spacing:.12g}" for site in range(int(n_sites))
    )


def peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def run_pyqed(args):
    from pyqed.qchem import Molecule
    from pyqed.qchem.dmrg.dmrg import DMRG
    from pyqed.qchem.hf import RHF

    atom = hydrogen_chain(args.ncas, args.spacing)
    started = time.perf_counter()
    mol = Molecule(atom=atom, unit="bohr", basis=args.basis)
    mol.build(
        eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    mf = RHF(mol).run()
    setup_seconds = time.perf_counter() - started

    solver = DMRG(
        mf,
        ncas=args.ncas,
        nelecas=args.ncas,
        D=args.D,
        init_guess="cid",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    solve_started = time.perf_counter()
    solver.run(
        nsweeps=args.nsweeps,
        mixer_zero_block_noise_scale=0.0,
        conv_tol=-1.0,
        profile=True,
    )
    solve_seconds = time.perf_counter() - solve_started

    history = list(solver.dmrg.history)
    objectives = [
        objective
        for sweep in history
        for objective in sweep.get("bond_objectives", ())
    ]
    metadata = [
        objective.get("renormalized_operator_metadata") or {}
        for objective in objectives
    ]
    schedule_stats = [
        item.get("channel_resolved_su2_schedule")
        for item in metadata
        if item.get("channel_resolved_su2_schedule") is not None
    ]
    table_stats = [
        objective.get("renormalized_operator_table_stats") or {}
        for objective in objectives
    ]
    ecore = float(solver.e_core)
    return {
        "backend": "pyqed-su2",
        "ncas": int(args.ncas),
        "nelecas": int(args.ncas),
        "D": int(args.D),
        "max_bond_mode": "reduced",
        "site_basis": "fully_reduced",
        "nsweeps": int(args.nsweeps),
        "energy": float(np.asarray(solver.e_tot).reshape(-1)[0]),
        "rhf_energy": float(mf.e_tot),
        "sweep_energies": [
            float(np.asarray(sweep["energy"]).reshape(-1)[0]) + ecore
            for sweep in history
            if sweep.get("energy") is not None
        ],
        "sweep_seconds": [
            float((sweep.get("timing") or {}).get("total", 0.0))
            for sweep in history
        ],
        "setup_seconds": float(setup_seconds),
        "solve_seconds": float(solve_seconds),
        "peak_rss_mib": float(peak_rss_mib()),
        "converged": bool(solver.dmrg.converged),
        "local_problems": int(len(objectives)),
        "max_residual": float(
            max(
                (
                    float(objective.get("residual", 0.0))
                    for objective in objectives
                    if objective.get("residual") is not None
                ),
                default=0.0,
            )
        ),
        "schedule_cache_hits": int(
            sum(bool(stats.get("cache_hit")) for stats in schedule_stats)
        ),
        "schedule_cache_misses": int(
            sum(not bool(stats.get("cache_hit")) for stats in schedule_stats)
        ),
        "max_factorized_terms": int(
            max((stats.get("factorized_terms", 0) for stats in schedule_stats), default=0)
        ),
        "max_local_dim": int(
            max((stats.get("local_dim", 0) for stats in schedule_stats), default=0)
        ),
        "max_compiled_sector_pairs": int(
            max((stats.get("compiled_sector_pairs", 0) for stats in schedule_stats), default=0)
        ),
        "max_compiled_stored_kernel_elements": int(
            max(
                (stats.get("compiled_stored_kernel_elements", 0) for stats in schedule_stats),
                default=0,
            )
        ),
        "max_orthonormal_stored_kernel_elements": int(
            max((stats.get("stored_kernel_elements", 0) for stats in table_stats), default=0)
        ),
        "slow_fallback_used": bool(
            any(item.get("slow_fallback_used", False) for item in metadata)
        ),
    }


def pyscf_reference(args):
    from pyscf import gto, mcscf, scf

    atom = hydrogen_chain(args.ncas, args.spacing)
    mol = gto.M(
        atom=atom,
        basis=args.basis,
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1.0e-10)
    mc = mcscf.CASCI(mf, args.ncas, args.ncas)
    return mol, mf, mc


def run_block2(args):
    from pyblock2._pyscf.ao2mo import integrals as block2_integrals
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    started = time.perf_counter()
    _mol, mf, mc = pyscf_reference(args)
    ncas, nelec, spin, ecore, h1e, g2e, orb_sym = (
        block2_integrals.get_rhf_integrals(
            mf,
            mc.ncore,
            mc.ncas,
            g2e_symm=1,
        )
    )
    setup_seconds = time.perf_counter() - started
    energies = []
    solve_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pyqed_block2_scaling_") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=1,
        )
        driver.initialize_system(
            n_sites=ncas,
            n_elec=nelec,
            spin=spin,
            orb_sym=orb_sym,
        )
        mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore, iprint=0)
        ket = driver.get_random_mps(tag="KET", bond_dim=args.D, nroots=1)
        for sweep in range(args.nsweeps):
            noise = 1.0e-6 if sweep == 0 else (1.0e-7 if sweep == 1 else 0.0)
            energy = driver.dmrg(
                mpo,
                ket,
                n_sweeps=1,
                bond_dims=[args.D],
                noises=[noise],
                tol=1.0e-9,
                iprint=0,
                dav_max_iter=100,
                dav_def_max_size=50,
            )
            energies.append(float(energy))
    solve_seconds = time.perf_counter() - solve_started
    return {
        "backend": "block2-su2",
        "ncas": int(args.ncas),
        "nelecas": int(args.ncas),
        "D": int(args.D),
        "nsweeps": int(args.nsweeps),
        "energy": float(energies[-1]),
        "rhf_energy": float(mf.e_tot),
        "sweep_energies": energies,
        "setup_seconds": float(setup_seconds),
        "solve_seconds": float(solve_seconds),
        "peak_rss_mib": float(peak_rss_mib()),
    }


def run_casci(args):
    started = time.perf_counter()
    _mol, mf, mc = pyscf_reference(args)
    setup_seconds = time.perf_counter() - started
    solve_started = time.perf_counter()
    energy = float(mc.kernel()[0])
    solve_seconds = time.perf_counter() - solve_started
    return {
        "backend": "pyscf-casci",
        "ncas": int(args.ncas),
        "nelecas": int(args.ncas),
        "energy": energy,
        "rhf_energy": float(mf.e_tot),
        "setup_seconds": float(setup_seconds),
        "solve_seconds": float(solve_seconds),
        "peak_rss_mib": float(peak_rss_mib()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pyqed", "block2", "casci"), required=True)
    parser.add_argument("--ncas", type=int, choices=(8, 12), required=True)
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--nsweeps", type=int, default=4)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--spacing", type=float, default=1.6)
    args = parser.parse_args()

    runner = {
        "pyqed": run_pyqed,
        "block2": run_block2,
        "casci": run_casci,
    }[args.backend]
    result = runner(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
