#!/usr/bin/env python3
"""Fe(CO)5 frontier COCAS with Cholesky integrals and exact CASCI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import traceback

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/feco5-matplotlib")

import numpy as np
from examples.qchem.fe_co5_su2_narg import ATOM
from pyqed.qchem import COCAS, Molecule


def save_pickle(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def plot_result(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    energies = np.asarray(result["energy_history_hartree"])
    rows = result.get("macro_diagnostics", [])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    axes[0].plot(np.arange(len(energies)), energies, "o-")
    axes[0].set(xlabel="CO macroiteration", ylabel="Energy (Hartree)")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    if len(energies) > 1:
        axes[1].semilogy(np.arange(1, len(energies)), np.maximum(abs(np.diff(energies)), 1e-16), "o-")
    axes[1].axhline(result["macro_tol"], color="gray", linestyle="--")
    axes[1].set(xlabel="CO macroiteration", ylabel="Absolute energy change (Hartree)")
    gradients = [row for row in rows if "gn" in row]
    if gradients:
        axes[2].semilogy([row["macro"] for row in gradients], [max(row["gn"], 1e-16) for row in gradients], "o-")
    axes[2].axhline(result["orbital_gradient_tol"], color="gray", linestyle="--")
    axes[2].set(xlabel="CO macroiteration", ylabel="Orbital gradient norm (Hartree)")
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    cas = f"{result['nelecas']},{result['ncas']}"
    name = f"feco5_{result['basis'].replace('-', '')}_cocas{result['nelecas']}{result['ncas']}_cd_convergence"
    fig.suptitle(f"Fe(CO)₅ · {result['basis']} · COCAS({cas}) · {result['optimizer']} · CD threshold {result['cd_tol']:g}\n"
                 f"CO converged: {result.get('macro_converged', False)}")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cas", nargs=2, type=int, metavar=("ELECTRONS", "ORBITALS"), default=(8, 8))
    parser.add_argument("--max-macro", type=int, default=150)
    parser.add_argument("--cd-tol", type=float, default=1e-8)
    parser.add_argument("--optimizer-max-steps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-diis", action="store_true")
    parser.add_argument("--diis-residual", choices=("step", "transported_step", "gradient"), default="step")
    parser.add_argument("--optimizer", choices=("ISD", "RCG", "LBFGS"), default="RCG")
    parser.add_argument("--isd-gap-floor", type=float)
    parser.add_argument("--physical-inner", action="store_true")
    parser.add_argument("--orbital-update", choices=("fixed_rdm", "relaxed_lbfgs"), default="fixed_rdm")
    args = parser.parse_args()
    nelecas, ncas = args.cas
    if nelecas <= 0 or nelecas % 2 or ncas <= 0 or nelecas > 2*ncas:
        parser.error("Use a positive even active electron count and sufficient active orbitals")
    output = args.output or Path(f"/private/tmp/feco5_{args.basis.replace('-', '')}_cocas{nelecas}{ncas}_cd")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config = dict(basis=args.basis, atom_angstrom=ATOM, ncas=ncas, nelecas=nelecas,
                  active_selection="RHF frontier", integral_representation="CD",
                  cd_tol=args.cd_tol, max_macro=args.max_macro, macro_tol=1e-6,
                  orbital_gradient_tol=1e-4, ci_tol=1e-10, optimizer=args.optimizer,
                  optimizer_tol=1e-4, optimizer_max_steps=args.optimizer_max_steps,
                  diis=not args.no_diis and args.orbital_update == "fixed_rdm", diis_residual=args.diis_residual,
                  orbital_update=args.orbital_update,
                  isd_gap_floor=args.isd_gap_floor,
                  physical_inner=args.physical_inner, threads=1)
    config.update(macro_reject_max=20 if args.orbital_update == "relaxed_lbfgs" else 8,
                  macro_trust_min=1e-8 if args.orbital_update == "relaxed_lbfgs" else 1e-4)
    if (output / "rhf.pkl").exists():
        stored = json.loads((output / "config.json").read_text())
        for key in ("basis", "atom_angstrom", "cd_tol"):
            if stored[key] != config[key]:
                raise ValueError(f"Cached RHF changes {key}; use a new output directory")
    previous = json.loads((output / "progress.json").read_text()) if args.resume else None
    prefix = previous["energy_history_hartree"] if previous else []
    offset = max(row["macro"] for row in previous["macro_diagnostics"]) if previous else 0
    if previous:
        for key in ("basis", "atom_angstrom", "ncas", "nelecas", "cd_tol"):
            if previous[key] != config[key]:
                raise ValueError(f"Restart changes {key}")
        config["continued_after_macro"] = offset
    (output / "config.json").write_text(json.dumps(config, indent=2))
    import pyqed.qchem.mcscf.cocas as co
    import pyqed.qchem.mcscf.direct_ci as ci
    import pyqed.optimize as optimizer
    import pyqed.qchem.mol as molecular
    hashes = {str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
              for module in (co, ci, optimizer, molecular)}
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
    rows = list(previous["macro_diagnostics"]) if previous else []

    def combined_history(values):
        current = [float(np.asarray(e).reshape(-1)[0]) for e in values]
        if prefix:
            if abs(prefix[-1] - current[0]) > 1e-7:
                raise RuntimeError("Restart orbitals do not reproduce the boundary energy")
            return prefix + current[1:]
        return current

    def checkpoint(event):
        row = dict(event["diagnostics"], macro=event["macro"]+offset)
        rows.append(row)
        mc = event["casci"]
        if row.get("accepted", True):
            save_pickle(output / "ci_latest.pkl", dict(ci=mc.ci, binary=mc.binary,
                        ncas=ncas, nelecas=nelecas, solver_backend=mc.solver_backend))
            np.save(output / "orbitals_latest.npy", event["mo_coeff"])
        progress = dict(config, energy_history_hartree=combined_history(event["energy_history"]),
                        macro_diagnostics=rows, energy_hartree=float(event["energy"]))
        (output / "progress.json").write_text(json.dumps(progress, indent=2))
        plot_result(output, progress)
        print(f"CO macro {row['macro']}: E={float(event['energy']):.12f}; "
              f"gradient={row.get('gn')}; accepted={row.get('accepted', True)}; checkpoint saved", flush=True)

    try:
        reference = output / "rhf.pkl"
        if reference.exists():
            with reference.open("rb") as handle:
                mf = pickle.load(handle)
        else:
            print("Building matrix-free CD integrals", flush=True)
            mol = Molecule(atom=ATOM, unit="angstrom", basis=args.basis)
            mol.build(eri="cd", options={"low_rank_tol": args.cd_tol, "eri_screen_tol": 0.0,
                                        "workers": 1, "eri_backend": "cpp"})
            print(f"Built CD molecule: nao={mol.nao}, electrons={mol.nelec}", flush=True)
            mf = mol.RHF().run(tol=1e-10, max_cycle=150)
            if not mf.converged:
                raise RuntimeError("RHF did not converge")
            save_pickle(reference, mf)
        np.savez(output / "rhf_orbitals.npz", mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy)
        print(f"CD RHF energy: {mf.e_tot:.12f} Hartree", flush=True)
        if previous:
            from pyqed.qchem.dmrg.dmrgscf import _complete_mo_basis
            mf.mo_coeff = _complete_mo_basis(mf, np.load(output / "orbitals_latest.npy", allow_pickle=False))
        solver = COCAS(mf, ncas=ncas, nelecas=nelecas, max_cycles=max(1, args.max_macro-offset),
                       ci_tol=1e-10, macro_tol=1e-6, orb_grad_tol=1e-4,
                       optimizer=args.optimizer,
                       optimizer_max_steps=args.optimizer_max_steps,
                       physical_inner=args.physical_inner,
                       orbital_update=args.orbital_update,
                       isd_gap_floor=args.isd_gap_floor,
                       diis=config["diis"], diis_residual=args.diis_residual,
                       use_cholesky=True, verbose=1)
        print(f"Initial active orbitals (one based): {list(range(solver.ncore+1, solver.ncore+ncas+1))}", flush=True)
        solver.run(nstates=1, use_cholesky=True, macro_callback=checkpoint,
                   raise_on_nonconvergence=False)
        save_pickle(output / "ci_final.pkl", dict(ci=solver.ci, binary=solver.binary,
                    ncas=ncas, nelecas=nelecas, solver_backend=solver.solver_backend))
        np.save(output / "orbitals_final.npy", solver.mo_coeff)
        result = dict(config, energy_hartree=float(np.asarray(solver.e_tot).reshape(-1)[0]),
                      rhf_energy_hartree=float(mf.e_tot), use_cholesky=bool(solver.use_cholesky),
                      energy_history_hartree=combined_history(solver.e_history),
                      macro_diagnostics=rows, macro_iterations=solver.macro_iterations+offset,
                      macro_converged=solver.macro_converged, solver_converged=solver.solver_converged,
                      converged=solver.converged, wall_time_s=time.perf_counter()-start)
        (output / "result.json").write_text(json.dumps(result, indent=2))
        plot_result(output, result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        (output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
