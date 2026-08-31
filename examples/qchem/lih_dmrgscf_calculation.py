#!/usr/bin/env python3
"""Run and plot a native SU(2) DMRG-SCF calculation for LiH."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pyqed-lif-dmrgscf-mpl")

import matplotlib.pyplot as plt

from pyqed.qchem import Molecule, RHF
from pyqed.qchem.dmrg import DMRGSCF


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _plot_convergence(mc, path):
    rows = list(getattr(mc, "macro_diagnostics", []) or [])
    if rows:
        gradient_cycles = np.asarray([row["macro"] for row in rows], dtype=int)
        gradients = np.asarray([row.get("gn", np.nan) for row in rows], dtype=float)
        energies = np.asarray(mc.e_history, dtype=float).reshape(-1)
        energy_cycles = np.arange(energies.size)
    else:
        energies = np.asarray(mc.e_history, dtype=float).reshape(-1)
        energy_cycles = np.arange(energies.size)
        gradient_cycles = energy_cycles
        gradients = np.full(energies.shape, np.nan)

    fig, ax_energy = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    energy_error = (energies - energies[-1]) * 1000.0
    ax_energy.plot(energy_cycles, energy_error, "o-", color="#2369bd", label="Energy")
    ax_energy.axhline(0.0, color="0.65", lw=0.8)
    ax_energy.set_xlabel("DMRG-SCF macro iteration")
    ax_energy.set_ylabel(r"$E-E_{\mathrm{final}}$ (m$E_h$)", color="#2369bd")
    ax_energy.tick_params(axis="y", labelcolor="#2369bd")

    finite = np.isfinite(gradients) & (gradients > 0.0)
    if np.any(finite):
        ax_gradient = ax_energy.twinx()
        ax_gradient.semilogy(
            gradient_cycles[finite], gradients[finite], "s--", color="#c44e52", label="Orbital gradient"
        )
        ax_gradient.set_ylabel(r"Orbital-gradient norm", color="#c44e52")
        ax_gradient.tick_params(axis="y", labelcolor="#c44e52")

    ax_energy.set_title("LiH/STO-3G SU(2) DMRG-SCF convergence")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/private/tmp/pyqed-lih-dmrgscf")
    parser.add_argument("--bond", type=float, default=3.015, help="Li-H distance in bohr")
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--macro-cycles", type=int, default=8)
    parser.add_argument("--allow-unconverged", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    total_start = time.perf_counter()
    mol = Molecule(
        atom=f"Li 0 0 0; H 0 0 {args.bond}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})

    rhf_start = time.perf_counter()
    mf = RHF(mol, verbose=0).run()
    rhf_seconds = time.perf_counter() - rhf_start

    mc = DMRGSCF(
        mf,
        ncas=3,
        nelecas=2,
        D=args.D,
        max_cycles=args.macro_cycles,
        macro_tol=1.0e-7,
        dmrg_conv_tol=1.0e-8,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )
    dmrgscf_start = time.perf_counter()
    mc.run(
        nstates=1,
        nsweeps=args.sweeps,
        symmetry="su2",
        compute_s2=True,
        sweep_tol=1.0e-8,
        orb_grad_tol=1.0e-5,
        optimizer="RCG",
        optimizer_tol=1.0e-6,
        optimizer_max_steps=200,
        macro_trust_radius=0.25,
        mixer_zero_block_noise_scale=0.0,
        require_conv=not args.allow_unconverged,
    )
    dmrgscf_seconds = time.perf_counter() - dmrgscf_start

    figure_path = output_dir / "lih_sto3g_su2_dmrgscf_convergence.png"
    results_path = output_dir / "lih_sto3g_su2_dmrgscf_results.json"
    _plot_convergence(mc, figure_path)

    result = {
        "system": "LiH",
        "bond_bohr": args.bond,
        "basis": "STO-3G",
        "active_space": {"electrons": 2, "orbitals": 3},
        "symmetry": "SU(2)",
        "site_basis": "fully reduced spatial orbital",
        "bond_dimension": args.D,
        "maximum_sweeps": args.sweeps,
        "maximum_macro_cycles": args.macro_cycles,
        "rhf_energy_hartree": mf.e_tot,
        "dmrgscf_energy_hartree": mc.e_tot,
        "correlation_energy_hartree": np.asarray(mc.e_tot) - mf.e_tot,
        "converged": mc.converged,
        "macro_converged": mc.macro_converged,
        "solver_converged": mc.solver_converged,
        "macro_iterations": mc.macro_iterations,
        "spin_square": getattr(mc.casci, "s2", None),
        "energy_history_hartree": mc.e_history,
        "macro_diagnostics": mc.macro_diagnostics,
        "timing_seconds": {
            "rhf": rhf_seconds,
            "dmrgscf": dmrgscf_seconds,
            "total": time.perf_counter() - total_start,
        },
    }
    results_path.write_text(json.dumps(_plain(result), indent=2) + "\n")

    print("\nDMRG-SCF calculation complete")
    print(f"E(RHF)      = {mf.e_tot:.12f} Ha")
    print(f"E(DMRG-SCF) = {float(np.asarray(mc.e_tot)):.12f} Ha")
    print(f"Converged   = {mc.converged}")
    print(f"Macro cycles= {mc.macro_iterations}")
    print(f"Results     = {results_path}")
    print(f"Figure      = {figure_path}")


if __name__ == "__main__":
    main()
