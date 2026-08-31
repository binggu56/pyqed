#!/usr/bin/env python3
"""Compare PyQED and Block2 Abelian DMRG for pyrazine CAS(10,10)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import fci

from examples.qchem.pyrazine_dmrgscf import PYRAZINE_GEOMETRY_BOHR
from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrg import QCDMRG


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-dimensions", type=int, nargs="+", default=(16, 32, 64, 128))
    parser.add_argument("--half-sweeps", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--site", choices=("spin_orbital", "spatial"), default="spatial"
    )
    parser.add_argument("--spatial-backend", default="none")
    parser.add_argument(
        "--dmrgscf-data",
        type=Path,
        default=Path("/private/tmp/pyrazine_cas1010_dmrgscf/pyrazine_cas1010_dmrgscf.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/pyrazine_cas1010_u1_vs_block2"),
    )
    return parser.parse_args()


def active_hamiltonian(mo_coeff):
    mol = Molecule(atom=PYRAZINE_GEOMETRY_BOHR, unit="bohr", basis="sto-3g")
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mf = mol.RHF().run(tol=1.0e-10, max_cycle=100)
    solver = QCDMRG(
        mf,
        ncas=10,
        nelecas=10,
        D=1,
        site="spin_orbital",
        symmetry="sz",
        verbose=0,
    )
    solver.build(mo_coeff=mo_coeff)
    return mf, np.asarray(solver.h1e), np.asarray(solver.h2e), float(solver.e_core)


def run_pyqed(mf, mo_coeff, bond_dim, half_sweeps, tol, site, spatial_backend):
    solver = QCDMRG(
        mf,
        ncas=10,
        nelecas=10,
        D=bond_dim,
        site=site,
        spatial_family_environment_backend=spatial_backend,
        symmetry="sz",
        init_guess="cid",
        dmrg_performance="symmetric",
        verbose=0,
    )
    t0 = time.perf_counter()
    solver.build(mo_coeff=mo_coeff)
    build_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    solver.run(
        nsweeps=max(1, half_sweeps // 2),
        sweep_tol=tol,
        noise=1.0e-4,
        noise_decay=0.1,
        noise_cutoff=1.0e-9,
        require_convergence=False,
    )
    solve_seconds = time.perf_counter() - t0
    moving = solver.dmrg.environment_profile.get("moving_environment", {})
    profile_keys = (
        "owner_half_sweep_backend_actual",
        "owner_local_optimize_solve_actual",
        "owner_local_optimize_commit_actual",
        "owner_local_grouped_solve_update_backend_actual",
        "cpp_moving_environment_owner_local_grouped_solve_update_calls",
        "cpp_moving_environment_owner_local_grouped_solve_update_accepted",
        "cpp_moving_environment_owner_local_grouped_solve_update_rejections",
        "cpp_moving_environment_owner_local_grouped_solve_update_last_reason",
        "cpp_moving_environment_owner_local_grouped_solve_update_last_error",
        "owner_local_grouped_solve_update_rejected_reason",
        "owner_local_grouped_direct_prepare_last_error",
        "owner_local_grouped_direct_solve_update_last_error",
        "cpp_moving_environment_owner_local_grouped_solve_update_seconds",
        "cpp_moving_environment_owner_local_optimize_runner_seconds",
        "cpp_moving_environment_owner_sweep_schedule_plan_seconds",
        "compact_plan_build_seconds",
        "compact_plan_refresh_seconds",
        "compact_plan_matvec_seconds",
        "cpp_davidson_seconds",
    )
    return {
        "energy": float(solver.e_tot),
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "converged": bool(solver.dmrg.converged),
        "completed_half_sweeps": int(solver.dmrg.ncompleted_half_sweeps),
        "profile": {key: moving.get(key) for key in profile_keys},
        "direct_family_profile": (
            (
                solver.dmrg.sweep_history[-1].get("complementary_split_stats")
                or {}
            ).get("direct_family_table_builder", {})
            if solver.dmrg.sweep_history
            else {}
        ),
    }


def run_block2(h1e, h2e, ecore, bond_dim, half_sweeps, tol, scratch):
    driver = DMRGDriver(
        scratch=str(scratch),
        clean_scratch=True,
        stack_mem=2 << 30,
        n_threads=1,
        n_mkl_threads=1,
        symm_type=SymmetryTypes.SZ,
    )
    driver.initialize_system(n_sites=10, n_elec=10, spin=0)
    t0 = time.perf_counter()
    mpo = driver.get_qc_mpo(
        [h1e[0], h1e[1]],
        [h2e[0, 0], h2e[0, 1], h2e[1, 1]],
        ecore=ecore,
        iprint=0,
    )
    build_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    ket = driver.get_random_mps(
        tag="KET",
        bond_dim=bond_dim,
        dot=2,
        occs=[2.0] * 5 + [0.0] * 5,
    )
    noises = ([1.0e-4] * 2 + [1.0e-5] * 2 + [0.0] * half_sweeps)[:half_sweeps]
    energy = driver.dmrg(
        mpo,
        ket,
        n_sweeps=half_sweeps,
        tol=tol,
        bond_dims=[bond_dim] * half_sweeps,
        noises=noises,
        thrds=[tol] * half_sweeps,
        iprint=0,
    )
    solve_seconds = time.perf_counter() - t0
    return {
        "energy": float(energy),
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "requested_half_sweeps": half_sweeps,
    }


def plot_results(results, output):
    dims = np.asarray([row["bond_dimension"] for row in results])
    exact = float(results[0]["fci_energy"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for backend, marker in (("pyqed", "o"), ("block2", "s")):
        energies = np.asarray([row[backend]["energy"] for row in results])
        seconds = np.asarray([row[backend]["solve_seconds"] for row in results])
        axes[0].loglog(dims, 1000.0 * np.maximum(energies - exact, 1.0e-12), marker + "-", label=backend)
        axes[1].loglog(dims, seconds, marker + "-", label=backend)
    axes[0].set(xlabel="Bond dimension", ylabel="Variational error (m$E_h$)", title="Energy accuracy")
    axes[1].set(xlabel="Bond dimension", ylabel="Solver time (s)", title="Matched sweep budget")
    for axis in axes:
        axis.grid(which="both", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mo_coeff = np.load(args.dmrgscf_data)["mo_coeff"]
    mf, h1e, h2e, ecore = active_hamiltonian(mo_coeff)
    t0 = time.perf_counter()
    fci_energy, _ = fci.direct_spin1.kernel(
        h1e[0],
        h2e[0, 0],
        10,
        (5, 5),
        ecore=ecore,
        tol=1.0e-10,
        max_cycle=100,
        verbose=0,
    )
    fci_seconds = time.perf_counter() - t0

    results = []
    for bond_dim in args.bond_dimensions:
        row = {
            "bond_dimension": bond_dim,
            "fci_energy": float(fci_energy),
            "pyqed": run_pyqed(
                mf,
                mo_coeff,
                bond_dim,
                args.half_sweeps,
                args.tol,
                args.site,
                args.spatial_backend,
            ),
            "block2": run_block2(
                h1e,
                h2e,
                ecore,
                bond_dim,
                args.half_sweeps,
                args.tol,
                args.output_dir / f"block2_scratch_D{bond_dim}",
            ),
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "system": "pyrazine STO-3G CAS(10,10), optimized PyQED DMRG-SCF orbitals",
        "symmetry": "N and Sz",
        "threads": 1,
        "requested_half_sweeps": args.half_sweeps,
        "tolerance_hartree": args.tol,
        "fci_energy_hartree": float(fci_energy),
        "fci_seconds": fci_seconds,
        "results": results,
    }
    (args.output_dir / "pyrazine_u1_vs_block2.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_results(results, args.output_dir / "pyrazine_u1_vs_block2.png")


if __name__ == "__main__":
    main()
