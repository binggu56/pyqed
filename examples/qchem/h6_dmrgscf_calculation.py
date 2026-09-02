#!/usr/bin/env python3
"""Run and plot a small fully reduced SU(2) DMRG-SCF calculation for H6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRGSCF
from pyqed.qchem.hf import RHF


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


def _plot(result, output):
    energies = np.asarray(result["energy_history_hartree"], dtype=float).reshape(-1)
    cycles = np.arange(energies.size)
    rows = result["macro_diagnostics"]
    gradient_cycles = np.asarray(
        [row.get("macro", index + 1) for index, row in enumerate(rows)],
        dtype=int,
    )
    gradients = np.asarray([row.get("gn", np.nan) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    errors = (energies - energies[-1]) * 1000.0
    axes[0].plot(cycles, errors, "o-", color="#3266a8")
    axes[0].axhline(0.0, color="0.55", lw=0.8)
    axes[0].set(
        xlabel="DMRG-SCF macro iteration",
        ylabel=r"$E-E_{\mathrm{final}}$ (m$E_h$)",
        title="Energy convergence",
    )
    finite = np.isfinite(gradients) & (gradients > 0.0)
    if np.any(finite):
        gradient_axis = axes[0].twinx()
        gradient_axis.semilogy(
            gradient_cycles[finite],
            gradients[finite],
            "s--",
            color="#d06b32",
        )
        gradient_axis.set_ylabel("Orbital-gradient norm", color="#d06b32")
        gradient_axis.tick_params(axis="y", labelcolor="#d06b32")

    timings = result["timing_seconds"]
    labels = ["build", "RHF", "DMRG-SCF", "final RDM"]
    values = [
        timings["build"],
        timings["rhf"],
        timings["dmrgscf"],
        timings["final_rdm"],
    ]
    axes[1].bar(labels, values, color=["#56a3d9", "#4b9b67", "#bb6ba5", "#e19545"])
    axes[1].set(ylabel="wall time / s", title="Calculation timing")
    axes[1].tick_params(axis="x", rotation=15)
    for index, value in enumerate(values):
        axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom")

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "H6/6-31G fully reduced SU(2) DMRG-SCF "
        f"CAS(6,6), D={result['bond_dimension']}"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/private/tmp/pyqed-h6-dmrgscf")
    parser.add_argument("--spacing", type=float, default=1.8)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--macro-cycles", type=int, default=8)
    parser.add_argument("--cold-macro-start", action="store_true")
    parser.add_argument("--allow-unconverged", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    timings = {}
    total_started = time.perf_counter()

    atom = "; ".join(f"H 0 0 {args.spacing * site}" for site in range(6))
    started = time.perf_counter()
    molecule = Molecule(atom=atom, unit="bohr", basis="6-31g")
    molecule.build(
        eri="factors",
        options={"eri_backend": "cpp", "low_rank_tol": 1.0e-12},
    )
    timings["build"] = time.perf_counter() - started

    started = time.perf_counter()
    mean_field = RHF(molecule, verbose=0).run(tol=1.0e-11)
    timings["rhf"] = time.perf_counter() - started

    calculation = DMRGSCF(
        mean_field,
        ncas=6,
        nelecas=6,
        D=args.D,
        max_cycles=args.macro_cycles,
        macro_tol=1.0e-6,
        dmrg_conv_tol=1.0e-8,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )
    started = time.perf_counter()
    calculation.run(
        nstates=1,
        nsweeps=args.sweeps,
        sweep_tol=1.0e-8,
        orb_grad_tol=1.0e-4,
        optimizer="RCG",
        optimizer_tol=1.0e-5,
        optimizer_max_steps=50,
        optimizer_max_step_norm=0.20,
        macro_trust_radius=0.20,
        warm_start_dmrg=not args.cold_macro_start,
        warm_start_bonds=not args.cold_macro_start,
        mixer_zero_block_noise_scale=0.0,
        require_conv=not args.allow_unconverged,
    )
    timings["dmrgscf"] = time.perf_counter() - started

    started = time.perf_counter()
    rdm1, rdm2 = calculation.casci.make_rdm12(spatial=True)
    timings["final_rdm"] = time.perf_counter() - started
    timings["total"] = time.perf_counter() - total_started

    active_info = dict(calculation.casci.build_info or {})
    result = {
        "system": "linear H6",
        "spacing_bohr": args.spacing,
        "basis": "6-31G",
        "active_space": {"electrons": 6, "orbitals": 6},
        "symmetry": list(calculation.casci.symmetry),
        "site_basis": calculation.casci.spatial_site_basis,
        "integrals": "pivoted Cholesky factors",
        "bond_dimension": args.D,
        "maximum_sweeps": args.sweeps,
        "maximum_macro_cycles": args.macro_cycles,
        "rhf_energy_hartree": mean_field.e_tot,
        "dmrgscf_energy_hartree": calculation.e_tot,
        "correlation_energy_hartree": np.asarray(calculation.e_tot) - mean_field.e_tot,
        "converged": calculation.converged,
        "macro_converged": calculation.macro_converged,
        "solver_converged": calculation.solver_converged,
        "macro_iterations": calculation.macro_iterations,
        "energy_history_hartree": calculation.e_history,
        "macro_diagnostics": calculation.macro_diagnostics,
        "rdm1_trace": np.trace(rdm1),
        "rdm2_trace": np.einsum("pprr->", rdm2),
        "spatial_rdm_diagnostics": calculation.casci.spatial_rdm_diagnostics,
        "factorized_orbital_integrals": calculation.use_cholesky,
        "su2_runtime_reused": active_info.get("su2_runtime_reused"),
        "final_su2_runtime_rebuilt": active_info.get(
            "final_su2_runtime_rebuilt"
        ),
        "timing_seconds": timings,
    }

    stem = output_dir / "h6_631g_su2_dmrgscf"
    results_path = stem.with_suffix(".json")
    figure_path = stem.with_suffix(".png")
    results_path.write_text(json.dumps(_plain(result), indent=2) + "\n", encoding="utf-8")
    _plot(_plain(result), figure_path)

    print(json.dumps(_plain(result), indent=2))
    print(f"Results: {results_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
