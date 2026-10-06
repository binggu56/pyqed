#!/usr/bin/env python3
"""Fresh Fe(CO)5 frontier-CAS CO-DMRG run using the current reduced solver."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import traceback

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/feco5-matplotlib")
# The fused executor fails the bond-4 Hermiticity audit for this CAS.
# Use the validated direct reduced contextual contractions.
os.environ.setdefault("PYQED_SU2_DISABLE_OUTPUT_FUSION", "1")

import numpy as np

from examples.qchem.fe_co5_su2_narg import ATOM
from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrgscf import DMRGSCF


def save_pickle(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def plot_result(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energy = np.asarray(result["energy_history_hartree"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    axes[0].plot(np.arange(len(energy)), (energy-energy[0])*1000, "o-", label="Accepted energies")
    rejected = [row for row in result.get("macro_diagnostics", []) if row.get("accepted") is False]
    if rejected:
        axes[0].scatter([row["macro"] for row in rejected], [(row["energy"]-energy[0])*1000 for row in rejected],
                        marker="x", color="tab:red", s=65, label="Rejected trials")
    axes[0].legend(fontsize=8)
    axes[0].set(xlabel="CO macroiteration", ylabel="Energy change from restart (mHartree)")
    axes[0].set_title(f"Last accepted E = {energy[-1]:.10f} Hartree", fontsize=9)
    if len(energy) > 1:
        axes[1].semilogy(np.arange(1, len(energy)), np.maximum(abs(np.diff(energy)), 1e-16), "o-")
        axes[1].axhline(1e-6, color="gray", linestyle="--")
    else:
        axes[1].text(.5, .5, "No accepted CO step", ha="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="CO macroiteration", ylabel="Absolute energy change (Hartree)")
    gradients = [row for row in result.get("macro_diagnostics", []) if "gn" in row]
    if gradients:
        steps = [row["macro"] for row in gradients]
        norms = [max(row["gn"], 1e-16) for row in gradients]
        if "gn_start" in gradients[0]:
            steps.insert(0, gradients[0]["macro"] - 1)
            norms.insert(0, max(gradients[0]["gn_start"], 1e-16))
        axes[2].semilogy(steps, norms, "o-")
        axes[2].axhline(result.get("orbital_gradient_tol", 1e-4), color="gray", linestyle="--")
    else:
        axes[2].text(.5, .5, "No gradient diagnostics", ha="center", transform=axes[2].transAxes)
    axes[2].set(xlabel="CO macroiteration", ylabel="Orbital gradient norm (Hartree)")
    fig.suptitle(f"Fe(CO)₅ · def2-SVP · CAS(12,12) · D={result['D']}/sector · cutoff={result['schmidt_cutoff']:g}\n"
                 f"Macro converged: {result['macro_converged']} · Returned-state DMRG converged: {result['dmrg_converged']}")
    fig.savefig(output / "feco5_def2svp_cas1212_codmrg_convergence.png", dpi=180)
    fig.savefig(output / "feco5_def2svp_cas1212_codmrg_convergence.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=int, default=100)
    parser.add_argument("--max-macro", type=int, default=150)
    parser.add_argument("--dmrg-sweeps", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=1e-5)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--davidson-iterations", type=int, default=150)
    parser.add_argument("--davidson-space", type=int, default=96)
    parser.add_argument("--davidson-tol", type=float, default=1e-10)
    parser.add_argument("--optimizer", default="ISD")
    parser.add_argument("--diis", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=100)
    parser.add_argument("--optimizer-history", type=int, default=7)
    parser.add_argument("--optimizer-tol", type=float, default=1e-4)
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--trust-max", type=float, default=0.1)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--orbitals", type=Path, help="Starting full MO coefficients (.npy).")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"/private/tmp/feco5_def2svp_cas1212_codmrg_D{args.D}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = dict(basis="def2-svp", ncas=12, nelecas=12, D=args.D,
                  max_bond_mode="per_sector", atom_angstrom=ATOM,
                  active_selection="RHF frontier", orbital_driver="constrained",
                  max_macro=args.max_macro, dmrg_sweeps=args.dmrg_sweeps,
                  davidson_tol=args.davidson_tol, davidson_max_iter=args.davidson_iterations,
                  davidson_max_space=args.davidson_space,
                  davidson_workspace_mib=args.workspace_mib,
                  davidson_environment={key: os.environ.get(key) for key in
                                        ("PYQED_SU2_FORCE_COMPLEX_DAVIDSON", "PYQED_SU2_USE_REAL_DAVIDSON",
                                         "PYQED_SU2_DISABLE_OUTPUT_FUSION")},
                  schmidt_cutoff=args.cutoff, orbital_gradient_tol=1e-4,
                  optimizer=args.optimizer, optimizer_max_steps=args.optimizer_steps, optimizer_history=args.optimizer_history, optimizer_tol=args.optimizer_tol, diis=args.diis,
                  macro_trust_radius=args.trust_radius, macro_trust_max=args.trust_max)
    if args.initial_state:
        config["initial_state_sha256"] = hashlib.sha256(args.initial_state.read_bytes()).hexdigest()
    if args.orbitals:
        config["initial_orbitals_sha256"] = hashlib.sha256(args.orbitals.read_bytes()).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2))
    import pyqed.qchem.dmrg.dmrgscf as driver
    import pyqed.qchem.mcscf.cocas as co
    import pyqed.mps.nonabelian.sweep as sweep
    import pyqed.qchem.dmrg.dmrg as dmrg_module
    import pyqed.qchem.dmrg.backends.nonabelian as adapter
    import pyqed.optimize as optimizer_module
    orbital_minimize = co.minimize

    def logged_minimize(function, initial, *positional, **options):
        gradient = options["gradient_fn"]
        evaluations = 0
        last = None

        def counted_gradient(*values):
            nonlocal evaluations, last
            value = gradient(*values)
            evaluations += 1
            last = (values[0], value)
            if evaluations > 1 and (evaluations - 1) % 200 == 0:
                residual = optimizer_module.norm(optimizer_module.grad(*last))
                print(f"CO inner iteration {evaluations-1}: gradient={residual:.6e}", flush=True)
            return value

        options["gradient_fn"] = counted_gradient
        answer = orbital_minimize(function, initial, *positional, **options)
        norm = optimizer_module.norm(optimizer_module.grad(*last))
        print(f"CO orbital subproblem: iterations={evaluations-1}; "
              f"gradient={norm:.6e}; objective={answer[1]:.12f}", flush=True)
        return answer

    co.minimize = logged_minimize
    from pyqed.mps.nonabelian import _su2_kernel
    provenance = {str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                  for module in (driver, co, sweep, dmrg_module, adapter, optimizer_module, _su2_kernel)}
    provenance[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(provenance, indent=2))
    try:
        reference = output / "rhf.pkl"
        if reference.exists():
            with reference.open("rb") as handle:
                mf = pickle.load(handle)
            print("Loaded saved RHF reference", flush=True)
        else:
            print("Building factorized RI integrals", flush=True)
            mol = Molecule(atom=ATOM, unit="angstrom", basis="def2-svp")
            mol.build(eri="ri", options={"eri_backend": "cpp", "ri_tensor_backend": "cpp",
                                        "ri_storage": "packed", "ri_cache_dir": str(output / "ri_cache")})
            print(f"Built molecule: nao={mol.nao}, electrons={mol.nelec}", flush=True)
            mf = mol.RHF().run(tol=1e-10, max_cycle=150)
            if not mf.converged:
                raise RuntimeError("RHF did not converge")
            save_pickle(reference, mf)
        np.savez(output / "rhf_orbitals.npz", mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy)
        print(f"RHF energy: {float(mf.e_tot):.12f} Hartree", flush=True)
        initial_state = "hf"
        if args.initial_state:
            with args.initial_state.open("rb") as handle:
                initial_state = pickle.load(handle)
            saved_input = output / "mps_state_initial.pkl"
            if args.initial_state.resolve() != saved_input.resolve():
                shutil.copyfile(args.initial_state, saved_input)
        solver = DMRGSCF(mf, ncas=12, nelecas=12, D=args.D, max_cycles=args.max_macro,
                         macro_tol=1e-6, dmrg_conv_tol=1e-7, symmetry="su2",
                         integral_backend="ri", init_guess=initial_state, verbose=2)
        solver.spatial_rdm2_algorithm = "npdm"
        print(f"Starting CO-DMRG; core={solver.ncore}; active orbital indices (zero based): "
              f"{list(range(solver.ncore, solver.ncore + 12))}", flush=True)
        starting_orbitals = np.load(args.orbitals, allow_pickle=False) if args.orbitals else mf.mo_coeff
        np.save(output / "orbitals_initial.npy", starting_orbitals)
        macro_rows = []

        def checkpoint(event):
            row = event["diagnostics"]
            macro_rows.append(row)
            label = "accepted" if row["accepted"] else "rejected"
            stem = f"macro_{event['macro']:04d}_{label}"
            save_pickle(output / f"{stem}_state.pkl", event["casci"].export_ground_state())
            np.save(output / f"{stem}_orbitals.npy", event["mo_coeff"])
            save_pickle(output / f"{stem}_sweeps.pkl", event["casci"].dmrg.history)
            (output / "macro_progress.json").write_text(json.dumps(macro_rows, indent=2))
            print(f"CO macro {event['macro']} {label}: E={event['energy']:.12f}; "
                  f"gradient={row.get('gn')}; checkpoint={stem}", flush=True)

        solver.run(nstates=1, mo_coeff=starting_orbitals, nsweeps=args.dmrg_sweeps, conv_tol=1e-7,
                   su2_kernel_backend="cpp", max_bond_mode="per_sector",
                   orb_grad_tol=1e-4, macro_callback=checkpoint,
                   optimizer=args.optimizer, optimizer_max_steps=args.optimizer_steps, optimizer_history=args.optimizer_history, optimizer_tol=args.optimizer_tol, diis=args.diis,
                   macro_trust_radius=args.trust_radius, macro_trust_max=args.trust_max,
                   warm_start_dmrg=True, require_conv=False, macro_energy_rise_tol=1e-5,
                   davidson_tol=args.davidson_tol, davidson_max_iter=args.davidson_iterations,
                   local_solver_kwargs={"max_space": args.davidson_space, "tol_residual": 1e-9,
                                        "workspace_budget_bytes": args.workspace_mib * 1024**2},
                   cutoff=args.cutoff)
        state = solver.export_ground_state(state=0)
        save_pickle(output / "mps_state_final.pkl", state)
        np.save(output / "orbitals_final.npy", solver.mo_coeff)
        dimensions = [{repr(s): int(basis.dims[s]) for s in basis.sectors}
                      for basis in (state.bond_basis(b) for b in range(len(state)-1))]
        if any(d > args.D for bond in dimensions for d in bond.values()):
            raise RuntimeError("Final state exceeds the per-sector bond budget")
        result = dict(config, energy_hartree=float(np.asarray(solver.e_tot).reshape(-1)[0]),
                      rhf_energy_hartree=float(mf.e_tot),
                      energy_history_hartree=np.asarray(solver.e_history).reshape(-1).tolist(),
                      macro_iterations=int(solver.macro_iterations),
                      macro_diagnostics=solver.macro_diagnostics,
                      macro_converged=bool(solver.macro_converged),
                      dmrg_converged=bool(solver.solver_converged), converged=bool(solver.converged),
                      bond_sector_dimensions=dimensions, wall_time_s=time.perf_counter()-started)
        (output / "result.json").write_text(json.dumps(result, indent=2))
        save_pickle(output / "diagnostics.pkl", dict(macros=solver.macro_diagnostics, sweeps=solver.dmrg.history))
        plot_result(output, result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        (output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
