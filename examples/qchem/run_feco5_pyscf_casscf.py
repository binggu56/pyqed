#!/usr/bin/env python3
"""PySCF Fe(CO)5 CASSCF(8,8)/def2-SVP from a fresh RHF frontier guess."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/feco5-matplotlib")

import numpy as np
import pyscf
from pyscf import gto, lib, mcscf, scf
from examples.qchem.fe_co5_su2_narg import ATOM


def plot_result(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = result["history"]
    cycles = [0] + [row["macro"] for row in rows]
    energies = np.array([result["initial_direct_casci_energy_hartree"]] + [row["energy_hartree"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    scale, unit = (1e3, "mHartree") if np.ptp(energies) > 1e-3 else (1e6, "µHartree")
    axes[0].plot(cycles, (energies-energies[0])*scale, "o-")
    axes[0].set(xlabel="CASSCF macroiteration", ylabel=f"Energy relative to start ({unit})")
    axes[1].semilogy(cycles[1:], np.maximum(abs(np.diff(energies)), 1e-16), "o-")
    axes[1].axhline(result["energy_tolerance"], color="gray", linestyle="--")
    axes[1].set(xlabel="CASSCF macroiteration", ylabel="Absolute energy change (Hartree)")
    axes[2].semilogy(cycles[1:], [row["initial_orbital_gradient"] for row in rows], "o-", label="Macro initial gradient")
    axes[2].axhline(result["gradient_tolerance"], color="gray", linestyle="--")
    if "final_orbital_gradient" in result:
        axes[2].scatter(cycles[-1], result["final_orbital_gradient"], marker="*", s=100, label="Returned orbitals", zorder=3)
        axes[2].legend()
    axes[2].set(xlabel="CASSCF macroiteration", ylabel="Orbital gradient norm (Hartree)")
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Fe(CO)₅ · PySCF CASSCF(8,8) · def2-SVP · RHF frontier start\n"
                 f"Converged: {result.get('converged', False)} · E = {energies[-1]:.12f} Hartree")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"feco5_def2svp_pyscf_casscf88_convergence.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/feco5_def2svp_pyscf_casscf88_rhf_start"))
    parser.add_argument("--max-macro", type=int, default=50)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Use a new output directory to preserve the completed run")
    lib.param.TMPDIR = str(output)
    lib.num_threads(1)
    start = time.perf_counter()
    config = dict(method="PySCF one-step CASSCF", pyscf_version=pyscf.__version__,
                  basis="def2-svp", atom_angstrom=ATOM,
                  ncas=8, nelecas=8, ncore=44, charge=0, spin=0,
                  integral_representation="direct AO JK and out-of-core MO transformation",
                  rhf_initial_guess="minao", initial_orbitals="Fresh canonical RHF frontier orbitals",
                  initial_active_orbitals_one_based=list(range(45, 53)),
                  energy_tolerance=1e-9, gradient_tolerance=1e-5,
                  ci_energy_tolerance=1e-12, max_macro=args.max_macro, threads=1)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy2(__file__, output / "calculation_script.py")
    mol = gto.M(atom=config["atom_angstrom"], basis=config["basis"], unit="Angstrom",
                charge=0, spin=0, verbose=4, max_memory=512)
    mf = scf.RHF(mol)
    mf.chkfile = str(output / "pyscf_rhf.chk")
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    mf.init_guess = "minao"
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF failed to converge")
    assert mf._eri is None
    coeff = mf.mo_coeff.copy()
    assert np.max(abs(coeff.T @ mf.get_ovlp() @ coeff-np.eye(coeff.shape[1]))) < 1e-10
    np.savez(output / "rhf_orbitals.npz", mo_coeff=coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ)
    np.save(output / "orbitals_initial.npy", coeff)
    mc = mcscf.CASSCF(mf, 8, 8, ncore=44)
    mc.chkfile = str(output / "pyscf_casscf.chk")
    mc.conv_tol = config["energy_tolerance"]
    mc.conv_tol_grad = config["gradient_tolerance"]
    mc.fcisolver.conv_tol = config["ci_energy_tolerance"]
    mc.max_cycle_macro = config["max_macro"]
    initial_energy, _, initial_ci = mc.casci(coeff)
    config["initial_direct_casci_energy_hartree"] = float(initial_energy)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    rows = []

    def checkpoint(env):
        # Full macro callbacks share the freshly solved density as casdm1_last.
        if env["casdm1"] is not env["casdm1_last"]:
            return
        row = dict(macro=int(env["imacro"]), energy_hartree=float(env["e_tot"]),
                   energy_change_hartree=float(env["de"]),
                   initial_orbital_gradient=float(env["norm_gorb0"]),
                   final_micro_gradient=float(env["norm_gorb"]),
                   density_change_norm=float(env["norm_ddm"]))
        rows.append(row)
        np.save(output / "orbitals_latest.npy", env["mo"])
        np.save(output / "ci_latest.npy", env["fcivec"])
        progress = dict(config, history=rows, energy_hartree=row["energy_hartree"])
        (output / "progress.json").write_text(json.dumps(progress, indent=2))
        print(f"PySCF CASSCF macro {row['macro']}: E={row['energy_hartree']:.12f}; "
              f"initial gradient={row['initial_orbital_gradient']:.3e}", flush=True)

    mc.kernel(coeff, ci0=initial_ci, callback=checkpoint)
    dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, 8, (4, 4))
    eris = mc.ao2mo(mc.mo_coeff)
    gradient = mc.gen_g_hop(mc.mo_coeff, np.eye(mc.mo_coeff.shape[1]), dm1, dm2, eris)[0]
    spin_square = float(mc.fcisolver.spin_square(mc.ci, 8, (4, 4))[0])
    result = dict(config, rhf_energy_hartree=float(mf.e_tot), energy_hartree=float(mc.e_tot),
                  history=rows, macro_iterations=rows[-1]["macro"], converged=bool(mc.converged),
                  ci_converged=bool(mc.fcisolver.converged), spin_square=spin_square,
                  final_orbital_gradient=float(np.linalg.norm(gradient)),
                  ci_norm=float(np.vdot(mc.ci, mc.ci).real),
                  orbital_overlap_error=float(np.max(abs(mc.mo_coeff.T @ mf.get_ovlp() @ mc.mo_coeff-np.eye(mc.mo_coeff.shape[1])))),
                  active_rdm_trace=float(np.trace(dm1)),
                  active_natural_occupations=np.linalg.eigvalsh(dm1)[::-1].tolist(),
                  energy_change_from_initial_direct_casci_hartree=float(mc.e_tot)-float(initial_energy),
                  wall_time_s=time.perf_counter()-start)
    np.savez(output / "state_final.npz", mo_coeff=mc.mo_coeff, ci=mc.ci,
             mo_energy=mc.mo_energy, active_rdm1=dm1)
    (output / "result.json").write_text(json.dumps(result, indent=2))
    plot_result(output, result)
    assert abs(result["ci_norm"]-1) < 1e-9
    assert abs(result["active_rdm_trace"]-8) < 1e-8
    assert result["orbital_overlap_error"] < 1e-7 and spin_square < 1e-6
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
