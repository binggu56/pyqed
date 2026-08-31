#!/usr/bin/env python3
"""DMRG-SCF calculation for planar pyrazine CAS(10,10)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrgscf import DMRGSCF


PYRAZINE_GEOMETRY_BOHR = [
    ["N", 0.0000000000, 0.0000046126, 2.9751681209],
    ["C", 0.0000000000, 2.0213606485, 1.3447521663],
    ["C", 0.0000000000, 2.0213594563, -1.3447637764],
    ["N", 0.0000000000, -0.0000049244, -2.9751696399],
    ["C", 0.0000000000, -2.0213693403, -1.3447570196],
    ["C", 0.0000000000, -2.0213627060, 1.3447652675],
    ["H", 0.0000000000, 3.8979353927, 2.1970440670],
    ["H", 0.0000000000, 3.8979280273, -2.1970658170],
    ["H", 0.0000000000, -3.8979425319, -2.1970514056],
    ["H", 0.0000000000, -3.8979294535, 2.1970704549],
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--bond-dimension", type=int, default=64)
    parser.add_argument("--nsweeps", type=int, default=12)
    parser.add_argument("--max-cycles", type=int, default=12)
    parser.add_argument("--macro-tol", type=float, default=1.0e-6)
    parser.add_argument("--orb-grad-tol", type=float, default=1.0e-4)
    parser.add_argument("--dmrg-tol", type=float, default=1.0e-7)
    parser.add_argument(
        "--initial-data",
        type=Path,
        help="Continue from mo_coeff stored in a previous pyrazine DMRG-SCF NPZ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/pyrazine_cas1010_dmrgscf"),
    )
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def plot_convergence(energy_history, output, macro_tol, orb_grad_tol, diagnostics=()):
    energies = np.asarray(energy_history, dtype=float).reshape(-1)
    steps = np.arange(len(energies))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    relative_millihartree = 1000.0 * (energies - energies[-1])
    axes[0].plot(steps, relative_millihartree, "o-", lw=1.5)
    axes[0].set(
        xlabel="Macro iteration",
        ylabel=r"$E-E_{\rm final}$ (m$E_h$)",
        title=f"Final energy = {energies[-1]:.12f} Ha",
    )
    axes[0].grid(alpha=0.3)
    if len(energies) > 1:
        axes[1].semilogy(
            steps[1:],
            np.maximum(np.abs(np.diff(energies)), 1.0e-16),
            "o-",
            label=r"$|\Delta E|$",
        )
    if diagnostics:
        gradients = np.asarray([row["gn"] for row in diagnostics], dtype=float)
        axes[1].semilogy(
            np.arange(1, len(gradients) + 1), gradients, "s-", label="orbital gradient"
        )
    axes[1].axhline(macro_tol, color="k", ls="--", lw=1, label="energy tolerance")
    axes[1].axhline(
        orb_grad_tol, color="0.4", ls=":", lw=1, label="gradient tolerance"
    )
    axes[1].set(xlabel="Macro iteration", ylabel="Convergence measure", title="Convergence checks")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    mol = Molecule(atom=PYRAZINE_GEOMETRY_BOHR, unit="bohr", basis=args.basis)
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mf = mol.RHF().run(tol=1.0e-10, max_cycle=100)
    initial_mo_coeff = None
    if args.initial_data is not None:
        with np.load(args.initial_data) as previous:
            initial_mo_coeff = np.asarray(previous["mo_coeff"])

    mc = DMRGSCF(
        mf,
        ncas=10,
        nelecas=10,
        D=args.bond_dimension,
        max_cycles=args.max_cycles,
        macro_tol=args.macro_tol,
        dmrg_conv_tol=args.dmrg_tol,
        site="spatial",
        symmetry="sz",
        init_guess="cid",
        verbose=args.verbose,
    )
    mc.run(
        nstates=1,
        mo_coeff=initial_mo_coeff,
        orb_grad_tol=args.orb_grad_tol,
        nsweeps=args.nsweeps,
        require_conv=False,
        mixer_zero_block_noise_scale=0.0,
        mixer_nsweeps=0,
    )

    energy_history = np.asarray(mc.e_history, dtype=float).reshape(-1)
    np.savez(
        args.output_dir / "pyrazine_cas1010_dmrgscf.npz",
        energy=np.asarray(mc.e_tot),
        rhf_energy=float(mf.e_tot),
        energy_history=energy_history,
        mo_coeff=np.asarray(mc.mo_coeff),
    )
    plot_convergence(
        energy_history,
        args.output_dir / "pyrazine_cas1010_dmrgscf_convergence.png",
        args.macro_tol,
        args.orb_grad_tol,
        mc.macro_diagnostics,
    )

    summary = {
        "molecule": "pyrazine",
        "geometry_unit": "bohr",
        "basis": args.basis,
        "active_space": {"electrons": 10, "orbitals": 10, "ncore": int(mc.ncore)},
        "symmetry": "charge + Sz",
        "bond_dimension": args.bond_dimension,
        "requested_sweeps": args.nsweeps,
        "requested_macro_cycles": args.max_cycles,
        "macro_tolerance_hartree": args.macro_tol,
        "orbital_gradient_tolerance": args.orb_grad_tol,
        "initial_data": None if args.initial_data is None else str(args.initial_data),
        "rhf_energy_hartree": float(mf.e_tot),
        "dmrgscf_energy_hartree": float(np.asarray(mc.e_tot)),
        "correlation_below_rhf_hartree": float(np.asarray(mc.e_tot) - mf.e_tot),
        "converged": bool(mc.converged),
        "macro_converged": bool(mc.macro_converged),
        "solver_converged": bool(mc.solver_converged),
        "macro_iterations": int(mc.macro_iterations),
        "macro_diagnostics": mc.macro_diagnostics,
        "dmrgscf_timing": mc.dmrgscf_timing,
        "orbital_integral_backend": mc.orbital_integral_backend_actual,
        "wall_time_seconds": time.perf_counter() - t0,
    }
    (args.output_dir / "pyrazine_cas1010_dmrgscf_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n"
    )
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
