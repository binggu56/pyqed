#!/usr/bin/env python3
"""Compare converged SA(10)-DMRG-SCF and SA(10)-CASSCF for pyrazine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import CASSCF, Molecule
from pyqed.qchem.dmrg import DMRGSCF
from pyqed.units import au2ev


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


def _rss_mib():
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0**2 if sys.platform == "darwin" else 1024.0)


def _root_summary(energies):
    energies = np.asarray(energies, dtype=float).reshape(-1)
    return {
        "energies_hartree": energies,
        "state_average_energy_hartree": float(np.mean(energies)),
        "excitation_energies_ev": (energies - energies[0]) * au2ev,
    }


def run_dmrgscf(mf, args, weights, mo_coeff=None):
    calculation = DMRGSCF(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        D=args.bond_dimension,
        max_cycles=args.macro_cycles,
        macro_tol=args.macro_tol,
        dmrg_conv_tol=args.sweep_tol,
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        init_guess="hf",
        verbose=args.verbose,
    )
    started = time.perf_counter()
    calculation.run(
        nstates=args.nstates,
        weights=weights,
        mo_coeff=mo_coeff,
        nsweeps=args.sweeps,
        sweep_tol=args.sweep_tol,
        orb_grad_tol=args.orb_grad_tol,
        orbital_driver="second_order",
        orbital_micro_cycles=args.orbital_micro_cycles,
        micro_ci_mode=args.micro_ci_mode,
        optimizer_max_step_norm=args.max_step,
        macro_trust_radius=args.max_step,
        warm_start_bonds=False,
        mixer_zero_block_noise_scale=0.0,
        require_conv=False,
    )
    elapsed = time.perf_counter() - started
    root_traces = []
    validation_started = time.perf_counter()
    for root in range(args.nstates):
        root_traces.append(
            float(np.trace(calculation.casci.make_rdm1(root, spatial=True)))
        )
    summary = _root_summary(calculation.e_tot)
    summary.update(
        {
            "wall_time_seconds": elapsed,
            "validation_rdm1_seconds": time.perf_counter() - validation_started,
            "rdm1_traces": root_traces,
            "converged": bool(calculation.converged),
            "macro_converged": bool(calculation.macro_converged),
            "solver_converged": bool(calculation.solver_converged),
            "macro_iterations": int(calculation.macro_iterations),
            "dmrg_solves": int(calculation.dmrg_solve_count),
            "rdm_builds": int(calculation.rdm_build_count),
            "fixed_mps_trials": int(calculation.fixed_mps_trial_count),
            "macro_diagnostics": calculation.macro_diagnostics,
            "component_timing_seconds": dict(
                getattr(calculation.casci, "dmrgscf_timing", {}) or {}
            ),
            "maximum_resident_memory_mib": _rss_mib(),
            "reduced_root_bond_dimensions": [
                max([1] + [len(site.qns[2]) for site in state.sites[:-1]])
                for state in calculation.dmrg.states
            ],
        }
    )
    return calculation, summary


def run_casscf(mf, args, weights, mo_coeff=None, ci0=None):
    calculation = CASSCF(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        multiplicity=1,
        max_cycle=args.macro_cycles,
        max_micro_cycle=args.orbital_micro_cycles,
        conv_tol=args.macro_tol,
        conv_tol_grad=args.orb_grad_tol,
        conv_tol_grad_relaxed=args.orb_grad_tol,
        max_step=args.max_step,
        optimizer="AH",
        coupling="qn",
        micro_ci_mode=args.micro_ci_mode,
        use_cholesky=True,
        auto_active_restarts=False,
        verbose=args.verbose,
    ).state_average(weights)
    started = time.perf_counter()
    message = None
    try:
        calculation.run(
            nstates=args.nstates,
            mo_coeff=mo_coeff,
            ci0=ci0,
            use_cholesky=True,
        )
    except RuntimeError as exc:
        message = str(exc)
    elapsed = time.perf_counter() - started
    validation_started = time.perf_counter()
    root_traces = [
        float(np.trace(calculation.casci.make_rdm1(root)))
        for root in range(args.nstates)
    ]
    summary = _root_summary(calculation.e_tot)
    direct_diagnostics = dict(calculation.casci.direct_ci_diagnostics or {})
    spin0_pairs = calculation.casci.spin0_pair_indices
    summary.update(
        {
            "backend": calculation.casci.solver_backend,
            "wall_time_seconds": elapsed,
            "validation_rdm1_seconds": time.perf_counter() - validation_started,
            "rdm1_traces": root_traces,
            "converged": bool(calculation.converged),
            "macro_iterations": len(calculation.history),
            "macro_history": calculation.history,
            "termination_message": message,
            "determinant_dimension": int(np.asarray(calculation.ci[0]).size),
            "spin0_configuration_dimension": int(
                0 if spin0_pairs is None else len(spin0_pairs)
            ),
            "direct_ci_diagnostics": direct_diagnostics,
            "maximum_resident_memory_mib": _rss_mib(),
        }
    )
    return calculation, summary


def plot(result, output):
    dmrg = result["dmrgscf"]
    casscf = result["casscf"]
    roots = np.arange(result["settings"]["nstates"])
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)

    axes[0].plot(roots, dmrg["excitation_energies_ev"], "o-", label="DMRG-SCF")
    axes[0].plot(roots, casscf["excitation_energies_ev"], "s--", label="CASSCF")
    axes[0].set(
        xlabel="Singlet root",
        ylabel="Excitation energy (eV)",
        title="SA(10) root spectrum",
        xticks=roots,
    )
    axes[0].legend()

    difference = 1000.0 * (
        np.asarray(dmrg["energies_hartree"])
        - np.asarray(casscf["energies_hartree"])
    )
    axes[1].bar(roots, difference, color="#6A5ACD")
    axes[1].set(
        xlabel="Singlet root",
        ylabel=r"$E_{\rm DMRG}-E_{\rm CAS}$ (m$E_h$)",
        title="Finite-$D$ energy error",
        xticks=roots,
    )

    labels = ["SA(10)-\nDMRG-SCF", "SA(10)-\nCASSCF"]
    values = [dmrg["wall_time_seconds"], casscf["wall_time_seconds"]]
    bars = axes[2].bar(labels, values, color=["#3266A8", "#D06B32"])
    axes[2].bar_label(bars, fmt="%.1f s", padding=3)
    timing_title = "Converged orbital optimization"
    if dmrg.get("restart_input") and casscf.get("restart_input"):
        timing_title = "Final segments from shared CAS orbitals"
    axes[2].set(ylabel="Wall time (s)", title=timing_title)

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    settings = result["settings"]
    fig.suptitle(
        "Pyrazine/aug-cc-pVDZ SA(10) CAS"
        f"({settings['nelecas']},{settings['ncas']}), "
        f"DMRG $D={settings['bond_dimension']}$"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="/private/tmp/pyrazine_sa10_dmrgscf_casscf"
    )
    parser.add_argument("--ncas", type=int, default=12)
    parser.add_argument("--nelecas", type=int, default=12)
    parser.add_argument("--nstates", type=int, default=10)
    parser.add_argument("--bond-dimension", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--macro-cycles", type=int, default=16)
    parser.add_argument("--orbital-micro-cycles", type=int, default=4)
    parser.add_argument(
        "--micro-ci-mode",
        choices=("full", "keyframe"),
        default="keyframe",
    )
    parser.add_argument("--max-step", type=float, default=0.20)
    parser.add_argument("--cd-tol", type=float, default=1.0e-8)
    parser.add_argument("--macro-tol", type=float, default=1.0e-6)
    parser.add_argument("--orb-grad-tol", type=float, default=1.0e-4)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-7)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--restart-dmrg-mo")
    parser.add_argument("--restart-casscf-mo")
    parser.add_argument("--restart-casscf-ci")
    parser.add_argument(
        "--backend",
        choices=("both", "dmrgscf", "casscf"),
        default="both",
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = np.full(args.nstates, 1.0 / args.nstates)
    result = {
        "system": "planar pyrazine/aug-cc-pVDZ",
        "comparison_scope": (
            "matched converged second-order orbital optimization; "
            "native PyQED DMRG-SCF versus native spin-zero direct-CI CASSCF"
        ),
        "settings": {
            "ncas": args.ncas,
            "nelecas": args.nelecas,
            "nstates": args.nstates,
            "weights": weights,
            "bond_dimension": args.bond_dimension,
            "sweeps": args.sweeps,
            "macro_cycles": args.macro_cycles,
            "orbital_micro_cycles": args.orbital_micro_cycles,
            "micro_ci_mode": args.micro_ci_mode,
            "maximum_orbital_step": args.max_step,
            "macro_energy_tolerance_hartree": args.macro_tol,
            "orbital_gradient_tolerance": args.orb_grad_tol,
            "dmrg_sweep_tolerance_hartree": args.sweep_tol,
            "threads": 1,
        },
    }

    stem = output_dir / "pyrazine_sa10_cas1212_dmrgscf_vs_casscf"
    json_path = stem.with_suffix(".json")
    figure_path = stem.with_suffix(".png")
    dmrg_mo_path = stem.with_name(stem.name + "_dmrg_mo.npy")
    casscf_mo_path = stem.with_name(stem.name + "_casscf_mo.npy")
    casscf_ci_path = stem.with_name(stem.name + "_casscf_ci.npy")

    if args.plot_only:
        saved = json.loads(json_path.read_text())
        plot(saved, figure_path)
        print(f"Figure: {figure_path}", flush=True)
        return

    if args.backend == "casscf" and json_path.exists():
        previous = json.loads(json_path.read_text())
        if "dmrgscf" in previous:
            result["dmrgscf"] = previous["dmrgscf"]
        for key in ("integral_build_seconds", "rhf_seconds", "rhf_energy_hartree"):
            if key in previous:
                result[key] = previous[key]

    if args.backend in {"both", "dmrgscf"}:
        print("Building PyQED Cholesky integrals...", flush=True)
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
        result["integral_build_seconds"] = time.perf_counter() - started
        print("Running PyQED RHF...", flush=True)
        started = time.perf_counter()
        mean_field = molecule.RHF().run(
            tol=1.0e-10,
            conv_tol_grad=1.0e-5,
            verbose=0,
        )
        result["rhf_seconds"] = time.perf_counter() - started
        result["rhf_energy_hartree"] = float(mean_field.e_tot)

        print("Running SA(10)-DMRG-SCF...", flush=True)
        dmrg_mo0 = (
            None
            if args.restart_dmrg_mo is None
            else np.load(args.restart_dmrg_mo)
        )
        _dmrg, result["dmrgscf"] = run_dmrgscf(
            mean_field, args, weights, mo_coeff=dmrg_mo0
        )
        np.save(dmrg_mo_path, _dmrg.mo_coeff)
        result["dmrgscf"]["restart_input"] = args.restart_dmrg_mo
        result["dmrgscf"]["orbital_checkpoint"] = str(dmrg_mo_path)
        json_path.write_text(json.dumps(_plain(result), indent=2) + "\n")
        print(
            f"SA(10)-DMRG-SCF finished in {result['dmrgscf']['wall_time_seconds']:.3f} s",
            flush=True,
        )

    if args.backend in {"both", "casscf"}:
        if args.backend == "casscf":
            print("Building PyQED Cholesky integrals...", flush=True)
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
            mean_field = molecule.RHF().run(
                tol=1.0e-10,
                conv_tol_grad=1.0e-5,
                verbose=0,
            )
        print("Running native conventional SA(10)-CASSCF...", flush=True)
        casscf_mo0 = (
            None
            if args.restart_casscf_mo is None
            else np.load(args.restart_casscf_mo)
        )
        casscf_ci0 = (
            None
            if args.restart_casscf_ci is None
            else list(np.load(args.restart_casscf_ci))
        )
        _casscf, result["casscf"] = run_casscf(
            mean_field,
            args,
            weights,
            mo_coeff=casscf_mo0,
            ci0=casscf_ci0,
        )
        np.save(casscf_mo_path, _casscf.mo_coeff)
        np.save(casscf_ci_path, np.stack(_casscf.ci))
        result["casscf"]["restart_input"] = args.restart_casscf_mo
        result["casscf"]["ci_restart_input"] = args.restart_casscf_ci
        result["casscf"]["orbital_checkpoint"] = str(casscf_mo_path)
        result["casscf"]["ci_checkpoint"] = str(casscf_ci_path)
        print(
            f"SA(10)-CASSCF finished in {result['casscf']['wall_time_seconds']:.3f} s",
            flush=True,
        )

    if "dmrgscf" in result and "casscf" in result:
        result["comparison"] = {
            "root_energy_difference_hartree": (
                np.asarray(result["dmrgscf"]["energies_hartree"])
                - np.asarray(result["casscf"]["energies_hartree"])
            ),
            "state_average_energy_difference_hartree": (
                result["dmrgscf"]["state_average_energy_hartree"]
                - result["casscf"]["state_average_energy_hartree"]
            ),
            "wall_time_ratio_dmrgscf_over_casscf": (
                result["dmrgscf"]["wall_time_seconds"]
                / result["casscf"]["wall_time_seconds"]
            ),
        }
    json_path.write_text(json.dumps(_plain(result), indent=2) + "\n")
    if "dmrgscf" in result and "casscf" in result:
        plot(result, figure_path)
    print(json.dumps(_plain(result), indent=2), flush=True)
    print(f"Results: {json_path}", flush=True)
    print(f"Figure: {figure_path}", flush=True)


if __name__ == "__main__":
    main()
