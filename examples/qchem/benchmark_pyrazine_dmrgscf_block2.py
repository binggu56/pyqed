#!/usr/bin/env python3
"""Compare a pyrazine SU(2) DMRG-SCF pilot with PySCF/block2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRGSCF

ATOM = """
N   0.000000   1.340000   0.000000
C   1.160000   0.670000   0.000000
C   1.160000  -0.670000   0.000000
N   0.000000  -1.340000   0.000000
C  -1.160000  -0.670000   0.000000
C  -1.160000   0.670000   0.000000
H   2.095000   1.210000   0.000000
H   2.095000  -1.210000   0.000000
H  -2.095000  -1.210000   0.000000
H  -2.095000   1.210000   0.000000
"""


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


def run_pyqed(args):
    timings = {}
    started = time.perf_counter()
    molecule = Molecule(atom=ATOM, basis="aug-cc-pvdz", unit="angstrom")
    molecule.build(
        eri="cd",
        options={
            "coord_type": "spherical",
            "low_rank_tol": args.cd_tol,
            "eri_screen_tol": 0.0,
            "parallel": False,
            "rys_cache_mib": 256,
        },
    )
    timings["integrals"] = time.perf_counter() - started

    started = time.perf_counter()
    mean_field = molecule.RHF().run(
        tol=1.0e-10,
        conv_tol_grad=1.0e-5,
        verbose=0,
    )
    timings["rhf"] = time.perf_counter() - started

    calculation = DMRGSCF(
        mean_field,
        ncas=args.ncas,
        nelecas=args.nelecas,
        D=args.D,
        max_cycles=args.macro_cycles,
        macro_tol=args.macro_tol,
        dmrg_conv_tol=args.sweep_tol,
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        init_guess="hf",
        verbose=0,
    )
    started = time.perf_counter()
    restart_mo = None
    if args.restart_mo is not None:
        restart_mo = np.load(args.restart_mo)
    calculation.run(
        nstates=1,
        nsweeps=args.sweeps,
        sweep_tol=args.sweep_tol,
        orb_grad_tol=args.orb_grad_tol,
        orb_grad_tol_relaxed=args.orb_grad_tol_relaxed,
        orbital_driver=args.orbital_driver,
        orbital_micro_cycles=args.orbital_micro_cycles,
        optimizer=args.optimizer,
        optimizer_tol=1.0e-5,
        optimizer_max_steps=args.optimizer_steps,
        optimizer_max_step_norm=args.optimizer_max_step_norm,
        macro_trust_radius=args.optimizer_max_step_norm,
        diis=args.orbital_diis,
        warm_start_bonds=True,
        mixer_zero_block_noise_scale=0.0,
        require_conv=args.require_convergence,
        mo_coeff=restart_mo,
    )
    timings["dmrgscf"] = time.perf_counter() - started

    started = time.perf_counter()
    rdm1, rdm2 = calculation.casci.make_rdm12(spatial=True)
    timings["final_rdm"] = time.perf_counter() - started
    active = dict(calculation.casci.build_info or {})
    checkpoint = (
        Path(args.output_dir) / "pyrazine_aug_cc_pvdz_pyqed_mo_coeff.npy"
    )
    np.save(checkpoint, np.asarray(calculation.mo_coeff))
    return {
        "backend": "PyQED fully reduced SU(2)",
        "rhf_energy_hartree": float(mean_field.e_tot),
        "dmrgscf_energy_hartree": float(np.asarray(calculation.e_tot)),
        "correlation_energy_hartree": float(
            np.asarray(calculation.e_tot) - mean_field.e_tot
        ),
        "converged": bool(calculation.converged),
        "macro_converged": bool(calculation.macro_converged),
        "solver_converged": bool(calculation.solver_converged),
        "macro_iterations": int(calculation.macro_iterations),
        "dmrg_solves": int(calculation.dmrg_solve_count),
        "rdm_builds": int(calculation.rdm_build_count),
        "fixed_mps_trials": int(calculation.fixed_mps_trial_count),
        "energy_history_hartree": calculation.e_history,
        "macro_diagnostics": calculation.macro_diagnostics,
        "rdm1_trace": float(np.trace(rdm1)),
        "rdm2_trace": float(np.einsum("pprr->", rdm2)),
        "rdm_diagnostics": calculation.casci.spatial_rdm_diagnostics,
        "factorized_orbital_integrals": bool(
            calculation.use_cholesky
        ),
        "mo_checkpoint": str(checkpoint),
        "final_su2_runtime_rebuilt": active.get(
            "final_su2_runtime_rebuilt"
        ),
        "timing_seconds": timings,
    }


def run_block2(args):
    from pyscf import dmrgscf, gto, mcscf, scf

    timings = {}
    started = time.perf_counter()
    molecule = gto.M(
        atom=ATOM,
        basis="aug-cc-pvdz",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )
    timings["integrals"] = time.perf_counter() - started

    started = time.perf_counter()
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-10
    mean_field.conv_tol_grad = 1.0e-5
    mean_field.kernel()
    timings["rhf"] = time.perf_counter() - started

    with tempfile.TemporaryDirectory(prefix="pyrazine-block2-") as scratch:
        macro_history = []

        def callback(envs):
            macro = int(envs.get("imacro", len(macro_history)))
            row = {
                "macro": macro,
                "energy": float(envs["e_tot"]),
                "gradient_norm": float(envs.get("norm_gorb", np.nan)),
            }
            if macro_history and macro_history[-1]["macro"] == macro:
                macro_history[-1] = row
            else:
                macro_history.append(row)

        calculation = mcscf.CASSCF(
            mean_field,
            args.ncas,
            args.nelecas,
        )
        calculation.max_cycle_macro = args.macro_cycles
        calculation.max_cycle_micro = args.optimizer_steps
        calculation.conv_tol = args.macro_tol
        calculation.conv_tol_grad = args.orb_grad_tol

        solver = dmrgscf.DMRGCI(
            molecule,
            maxM=args.D,
            tol=args.sweep_tol,
            num_thrds=1,
        )
        solver.runtimeDir = scratch
        solver.scratchDirectory = scratch
        solver.threads = 1
        solver.scheduleSweeps = [0]
        solver.scheduleMaxMs = [args.D]
        solver.scheduleTols = [args.sweep_tol]
        solver.scheduleNoises = [0.0]
        solver.maxIter = args.sweeps
        solver.twodot_to_onedot = max(0, args.sweeps - 1)
        solver.outputlevel = 0
        solver.block_extra_keyword = [
            "full_fci_space",
            "cutoff 0",
            "fp_cps_cutoff 0",
        ]
        calculation.fcisolver = solver

        started = time.perf_counter()
        restart_mo = None
        if args.restart_block2_mo is not None:
            restart_mo = np.load(args.restart_block2_mo)
        calculation.kernel(mo_coeff=restart_mo, callback=callback)
        timings["dmrgscf"] = time.perf_counter() - started

        checkpoint = (
            Path(args.output_dir) / "pyrazine_aug_cc_pvdz_block2_mo_coeff.npy"
        )
        np.save(checkpoint, np.asarray(calculation.mo_coeff))

        started = time.perf_counter()
        rdm1, rdm2 = solver.make_rdm12(
            calculation.ci,
            args.ncas,
            args.nelecas,
        )
        timings["final_rdm"] = time.perf_counter() - started

    return {
        "backend": "PySCF CASSCF + block2 SU(2)",
        "block2_executable": str(dmrgscf.settings.BLOCKEXE),
        "rhf_energy_hartree": float(mean_field.e_tot),
        "dmrgscf_energy_hartree": float(calculation.e_tot),
        "correlation_energy_hartree": float(
            calculation.e_tot - mean_field.e_tot
        ),
        "converged": bool(calculation.converged),
        "macro_history": macro_history,
        "mo_checkpoint": str(checkpoint),
        "rdm1_trace": float(np.trace(rdm1)),
        "rdm2_trace": float(np.einsum("pprr->", rdm2)),
        "timing_seconds": timings,
    }


def _plot(result, output):
    labels = ["PyQED SU(2)", "block2 SU(2)"]
    records = [result["pyqed"], result["block2"]]
    colors = ["#3266a8", "#d06b32"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)

    correlations = [1000.0 * row["correlation_energy_hartree"] for row in records]
    axes[0].bar(labels, correlations, color=colors)
    axes[0].set(ylabel=r"$E-E_{\mathrm{RHF}}$ (m$E_h$)", title="Pilot energy lowering")
    axes[0].tick_params(axis="x", rotation=12)

    times = [row["timing_seconds"]["dmrgscf"] for row in records]
    axes[1].bar(labels, times, color=colors)
    axes[1].set(ylabel="wall time / s", title="DMRG-SCF wall time")
    axes[1].tick_params(axis="x", rotation=12)
    for index, value in enumerate(times):
        axes[1].text(index, value, f"{value:.1f}", ha="center", va="bottom")

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Pyrazine/aug-cc-pVDZ CAS"
        f"({result['settings']['nelecas']},{result['settings']['ncas']}), "
        f"D={result['settings']['bond_dimension']}, "
        f"{result['settings']['macro_cycles']} macro cycle"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_pyqed_timing(result, output):
    record = result["pyqed"]
    timing = record["timing_seconds"]
    labels = ["Cholesky\nintegrals", "RHF", "DMRG-SCF", "Final\nRDM"]
    values = [
        timing["integrals"],
        timing["rhf"],
        timing["dmrgscf"],
        timing["final_rdm"],
    ]
    colors = ["#56B4E9", "#009E73", "#CC79A7", "#E69F00"]
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors)
    axis.bar_label(bars, fmt="%.2f s", padding=3)
    axis.set(ylabel="Wall time (s)", title="Native SU(2) calculation timing")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    settings = result["settings"]
    fig.suptitle(
        "Pyrazine/aug-cc-pVDZ CAS"
        f"({settings['nelecas']},{settings['ncas']}), "
        f"$D={settings['bond_dimension']}$, second-order/AH orbital optimization"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/private/tmp/pyqed-pyrazine-block2")
    parser.add_argument("--backend", choices=("both", "pyqed", "block2"), default="both")
    parser.add_argument("--ncas", type=int, default=10)
    parser.add_argument("--nelecas", type=int, default=10)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--macro-cycles", type=int, default=1)
    parser.add_argument("--optimizer-steps", type=int, default=4)
    parser.add_argument("--optimizer-max-step-norm", type=float, default=0.20)
    parser.add_argument("--orbital-micro-cycles", type=int, default=4)
    parser.add_argument(
        "--optimizer",
        choices=("RCG", "LBFGS", "NEWTON", "AH", "SD"),
        default="RCG",
    )
    parser.add_argument(
        "--orbital-driver",
        choices=("constrained", "nonredundant", "second_order"),
        default="constrained",
    )
    parser.add_argument("--orbital-diis", action="store_true")
    parser.add_argument("--require-convergence", action="store_true")
    parser.add_argument(
        "--restart-mo",
        help="continue PyQED orbital optimization from a saved MO checkpoint",
    )
    parser.add_argument(
        "--restart-block2-mo",
        help="continue block2/PySCF orbital optimization from its MO checkpoint",
    )
    parser.add_argument("--cd-tol", type=float, default=1.0e-8)
    parser.add_argument("--macro-tol", type=float, default=1.0e-6)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-7)
    parser.add_argument("--orb-grad-tol", type=float, default=1.0e-4)
    parser.add_argument("--orb-grad-tol-relaxed", type=float)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="retain a completed backend from the existing output JSON",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "pyrazine_aug_cc_pvdz_su2_dmrgscf_block2"
    json_path = stem.with_suffix(".json")
    figure_path = stem.with_suffix(".png")

    result = {
        "system": "pyrazine/aug-cc-pVDZ",
        "geometry": "idealized planar D2h",
        "comparison_scope": "matched DMRG-SCF convergence benchmark",
        "settings": {
            "ncas": args.ncas,
            "nelecas": args.nelecas,
            "bond_dimension": args.D,
            "sweeps": args.sweeps,
            "macro_cycles": args.macro_cycles,
            "optimizer_steps": args.optimizer_steps,
            "pyqed_optimizer": (
                "AH" if args.orbital_driver == "second_order" else args.optimizer
            ),
            "pyqed_orbital_driver": args.orbital_driver,
            "pyqed_orbital_diis": args.orbital_diis,
            "pyqed_optimizer_max_step_norm": args.optimizer_max_step_norm,
            "pyqed_orbital_micro_cycles": args.orbital_micro_cycles,
            "threads": 1,
            "orbital_gradient_tolerance": args.orb_grad_tol,
            "relaxed_zero_step_gradient_tolerance": args.orb_grad_tol_relaxed,
        },
    }
    if args.resume and json_path.exists():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        if previous.get("settings") != result["settings"]:
            raise ValueError("Existing benchmark settings do not match this run.")
        for backend in ("pyqed", "block2"):
            if backend in previous:
                result[backend] = previous[backend]
    if args.backend in {"both", "pyqed"}:
        print("Running PyQED fully reduced SU(2) DMRG-SCF", flush=True)
        result["pyqed"] = run_pyqed(args)
    if args.backend in {"both", "block2"}:
        print("Running PySCF/block2 SU(2) DMRG-SCF", flush=True)
        result["block2"] = run_block2(args)
    if "pyqed" in result and "block2" in result:
        result["comparison"] = {
            "energy_difference_hartree": (
                result["pyqed"]["dmrgscf_energy_hartree"]
                - result["block2"]["dmrgscf_energy_hartree"]
            ),
            "dmrgscf_time_ratio_pyqed_over_block2": (
                result["pyqed"]["timing_seconds"]["dmrgscf"]
                / result["block2"]["timing_seconds"]["dmrgscf"]
            ),
        }

    json_path.write_text(json.dumps(_plain(result), indent=2) + "\n", encoding="utf-8")
    if "pyqed" in result and "block2" in result:
        _plot(result, figure_path)
    elif "pyqed" in result:
        _plot_pyqed_timing(result, figure_path)
    print(json.dumps(_plain(result), indent=2))
    print(f"Results: {json_path}")
    if figure_path.exists():
        print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
