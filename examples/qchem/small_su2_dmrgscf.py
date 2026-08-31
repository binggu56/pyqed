#!/usr/bin/env python3
"""Run a small, nontrivial SU(2) DMRG-SCF calculation on H2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrgscf import DMRGSCF
from pyqed.qchem.hf import RHF


ATOM = "H 0 0 0; H 0 0 1.40"
BASIS = "6-31g"
UNIT = "bohr"


def _scalar(value):
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _pyscf_reference():
    from pyscf import gto, mcscf, scf

    mol = gto.M(atom=ATOM, unit=UNIT, basis=BASIS, spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.conv_tol = 1.0e-10
    mc.max_cycle_macro = 50
    energy = mc.kernel()[0]
    if not mc.converged:
        raise RuntimeError("PySCF CAS(4,4) reference did not converge.")
    return float(energy)


def _plot(result, output):
    import matplotlib.pyplot as plt

    energy = np.asarray(result["macro_energy_hartree"], dtype=float)
    reference = float(result["pyscf_casscf_energy_hartree"])
    timing = result["dmrgscf_timing_seconds"]
    names = ["RDM", "gradient", "orbital opt"]
    values = [
        timing.get("rdm_seconds", 0.0),
        timing.get("orbital_gradient_seconds", 0.0),
        timing.get("orbital_opt_seconds", 0.0),
    ]
    residual = max(float(result["dmrgscf_wall_seconds"]) - sum(values), 0.0)
    names.append("DMRG + other")
    values.append(residual)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    cycles = np.arange(len(energy))
    axes[0].plot(cycles, 1.0e3 * (energy - reference), "o-", lw=1.8)
    axes[0].axhline(0.0, color="black", lw=0.9, ls="--")
    axes[0].set_xlabel("Accepted macro step")
    axes[0].set_ylabel(r"$E-E_{\mathrm{PySCF}}$ (m$E_h$)")
    axes[0].set_title("SU(2) DMRG-SCF convergence")
    axes[0].grid(alpha=0.25)

    axes[1].bar(names, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    axes[1].set_ylabel("Wall time (s)")
    axes[1].set_title("DMRG-SCF timing")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("H2/6-31G CAS(2,2), fully reduced SU(2)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mol = Molecule(atom=ATOM, unit=UNIT, basis=BASIS, spin=0)
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mf = RHF(mol).run()
    hf_wall = time.perf_counter() - t0

    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=args.bond_dim,
        max_cycles=args.max_macro,
        macro_tol=args.macro_tol,
        dmrg_conv_tol=args.sweep_tol,
        optimizer="LBFGS",
        optimizer_tol=args.orbital_tol,
        optimizer_max_steps=20,
        symmetry="su2",
        init_guess="cid",
        verbose=args.verbose,
    )
    t0 = time.perf_counter()
    mc.run(
        nstates=1,
        nsweeps=args.sweeps,
        local_basis_policy="block2_like",
        orthonormalized_operator_dim=512,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
        su2_kernel_backend="cpp",
        profile=True,
        orb_grad_tol=args.orbital_tol,
        require_conv=not args.allow_unconverged,
    )
    dmrgscf_wall = time.perf_counter() - t0

    dm1, dm2 = mc.casci.make_rdm12(0, spatial=True, with_core=False)
    dm1 = np.asarray(dm1)
    dm2 = np.asarray(dm2)
    dm1_trace = float(np.trace(dm1).real)
    dm1_hermiticity = float(np.linalg.norm(dm1 - dm1.conj().T))
    if not np.isclose(dm1_trace, 2.0, atol=1.0e-8):
        raise RuntimeError(f"Active-space 1-RDM trace is {dm1_trace}, expected 2.")

    t0 = time.perf_counter()
    reference = _pyscf_reference()
    reference_wall = time.perf_counter() - t0

    history = list(getattr(mc.dmrg, "history", []) or [])
    last_history = history[-1] if history else {}
    diagnostics = dict(getattr(mc.dmrg, "diagnostics", {}) or {})
    result = {
        "system": "H2/6-31G",
        "active_space": "CAS(2,2)",
        "bond_dimension": int(args.bond_dim),
        "requested_sweeps": int(args.sweeps),
        "energy_hartree": _scalar(mc.e_tot),
        "pyscf_casscf_energy_hartree": reference,
        "energy_error_hartree": abs(_scalar(mc.e_tot) - reference),
        "hf_wall_seconds": hf_wall,
        "dmrgscf_wall_seconds": dmrgscf_wall,
        "pyscf_reference_wall_seconds": reference_wall,
        "macro_energy_hartree": [_scalar(value) for value in mc.e_history],
        "macro_diagnostics": mc.macro_diagnostics,
        "dmrgscf_timing_seconds": mc.dmrgscf_timing,
        "converged": bool(mc.converged),
        "macro_converged": bool(mc.macro_converged),
        "solver_converged": bool(mc.solver_converged),
        "macro_iterations": int(mc.macro_iterations),
        "dmrg_backend": getattr(mc.dmrg, "backend", None),
        "kernel_backend": diagnostics.get("kernel_backend"),
        "kernel_backend_actual": last_history.get("su2_kernel_backend_actual"),
        "site_basis": getattr(mc.casci, "spatial_site_basis", None),
        "target_charge": int(mc.dmrg.target_sector.charge),
        "target_two_s": int(mc.dmrg.target_sector.irrep.two_j),
        "completed_sweeps": int(getattr(mc.dmrg, "ncompleted", 0)),
        "rdm1_trace": dm1_trace,
        "rdm1_hermiticity_error": dm1_hermiticity,
        "rdm2_frobenius_norm": float(np.linalg.norm(dm2)),
    }
    json_path = output_dir / "small_su2_dmrgscf.json"
    figure_path = output_dir / "small_su2_dmrgscf.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    _plot(result, figure_path)
    print(json.dumps(result, indent=2))
    print(f"results: {json_path}")
    print(f"figure:  {figure_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-dim", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--max-macro", type=int, default=8)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-7)
    parser.add_argument("--macro-tol", type=float, default=1.0e-7)
    parser.add_argument("--orbital-tol", type=float, default=1.0e-4)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="/private/tmp/pyqed_small_su2_dmrgscf",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
