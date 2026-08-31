"""Compare reduced SU(2)-DMRG and SU(2)-LETTA on pyrazine CAS(4,4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from pyqed.letta import SU2LETTA
from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRG
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf import CASCI


def pyrazine_geometry():
    """Return the idealized planar regular-ring geometry in Angstrom."""
    ring_radius = 1.39
    hydrogen_radius = 2.47
    atoms = []
    for index, element in enumerate(("N", "C", "C", "N", "C", "C")):
        angle = index * np.pi / 3.0
        atoms.append(
            (element, (ring_radius * np.cos(angle), ring_radius * np.sin(angle), 0.0))
        )
        if element == "C":
            atoms.append(
                (
                    "H",
                    (
                        hydrogen_radius * np.cos(angle),
                        hydrogen_radius * np.sin(angle),
                        0.0,
                    ),
                )
            )
    return atoms


def select_pi_active_space(mol, mf, ncas=4):
    """Select balanced occupied/virtual frontier orbitals from the six pi MOs."""
    overlap = mf.get_ovlp()
    values, vectors = np.linalg.eigh(overlap)
    orthogonal_coefficients = (
        (vectors * np.sqrt(values)[None, :]) @ vectors.T @ mf.mo_coeff
    )
    pz_rows = np.array(
        [index for index, label in enumerate(mol.ao_labels()) if "2pz" in label],
        dtype=int,
    )
    if pz_rows.size != 6:
        raise RuntimeError(f"expected six C/N 2pz AOs, found {pz_rows.size}.")
    pi_weights = np.sum(np.abs(orthogonal_coefficients[pz_rows]) ** 2, axis=0)
    occupied = int(mol.nelec) // 2
    pi_space = tuple(int(index) for index in np.argsort(pi_weights)[-6:])
    occupied_pi = sorted(
        (index for index in pi_space if index < occupied),
        key=lambda index: mf.mo_energy[index],
        reverse=True,
    )
    virtual_pi = sorted(
        (index for index in pi_space if index >= occupied),
        key=lambda index: mf.mo_energy[index],
    )
    half = int(ncas) // 2
    active = tuple(
        sorted(
            occupied_pi[:half] + virtual_pi[:half],
            key=lambda index: mf.mo_energy[index],
        )
    )
    active_set = set(active)
    core = tuple(index for index in range(occupied) if index not in active_set)
    external = tuple(
        index
        for index in range(mf.mo_coeff.shape[1])
        if index not in active_set and index not in core
    )
    return active, pi_weights, mf.mo_coeff[:, core + active + external]


def _state_size(sites):
    return {
        "nparameters": int(
            sum(np.asarray(block).size for site in sites for block in site.data.values())
        ),
        "state_nbytes": int(
            sum(np.asarray(block).nbytes for site in sites for block in site.data.values())
        ),
    }


def run_dmrg(mf, mo_coeff, exact_energy, *, D, nsweeps):
    solver = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=D,
        init_guess="hf",
        symmetry="su2",
        spin=0,
        tol=1.0e-8,
        verbose=0,
    )
    started = perf_counter()
    solver.build(mo_coeff=mo_coeff)
    setup_s = perf_counter() - started
    started = perf_counter()
    solver.run(
        nsweeps=nsweeps,
        su2_kernel_backend="cpp",
        n_threads=1,
        require_convergence=False,
        sweep_tol=1.0e-8,
        noise=0.0,
    )
    optimization_s = perf_counter() - started
    cycle_history = [
        {
            "cycle": int(row["sweep"]),
            "energy": float(row["energy"]),
            "error": float(row["energy"] - exact_energy),
        }
        for row in solver.history
        if row.get("sweep_complete", False)
    ]
    result = {
        "method": "SU2-DMRG",
        "D": int(D),
        "energy": float(solver.energy),
        "error": float(solver.energy - exact_energy),
        "converged": bool(solver.converged),
        "cycles": int(solver.ncompleted),
        "setup_s": float(setup_s),
        "optimization_s": float(optimization_s),
        "runtime_memory_bytes": int(solver.dmrg.diagnostics.get("memory_bytes", 0)),
        "history": cycle_history,
    }
    result.update(_state_size(solver.ground_state.sites))
    return result, np.asarray(solver.h1e[0]), np.asarray(solver.h2e[0, 0]), float(solver.e_core)


def run_letta(
    h1e,
    eri,
    ecore,
    exact_energy,
    *,
    D,
    cycles,
    graph,
    gauge="conditional",
    reuse_environments=True,
    dense_dim=0,
):
    started = perf_counter()
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=4,
        spin=0,
        graph=graph,
        D=D,
        ecore=ecore,
        seed=7,
        workers=1,
    )
    setup_s = perf_counter() - started
    initial_energy = float(state.energy)
    started = perf_counter()
    try:
        state.run(
            nsweeps=cycles,
            algorithm="two_site",
            tol=1.0e-9,
            residual_tol=1.0e-7,
            truncation_tol=1.0e-6,
            consecutive_cycles=2,
            max_local_parameters=4096,
            gauge=None if gauge == "none" else gauge,
            reuse_environments=reuse_environments,
            dense_dim=dense_dim,
        )
        optimization_s = perf_counter() - started
        history = [
            {
                "cycle": 0,
                "energy": initial_energy,
                "error": float(initial_energy - exact_energy),
            }
        ] + [
            {
                "cycle": int(row["sweep"]),
                "energy": float(row["energy"]),
                "error": float(row["energy"] - exact_energy),
            }
            for row in state.history
        ]
        return {
            "method": "SU2-LETTA",
            "D": int(D),
            "energy": float(state.energy),
            "error": float(state.energy - exact_energy),
            "converged": bool(state.converged),
            "cycles": int(len(state.history)),
            "setup_s": float(setup_s),
            "optimization_s": float(optimization_s),
            "nparameters": int(state.nparameters),
            "state_nbytes": int(
                sum(
                    np.asarray(block).nbytes
                    for tensor in state.tensors
                    for block in tensor.values()
                )
            ),
            "runtime_memory_bytes": int(state.storage_nbytes),
            "frontier_states": list(state.frontier_states),
            "gauge": state.history[-1].get("gauge") if state.history else None,
            "environment_reuse": bool(
                state.history[-1].get("environment_reuse", False)
                if state.history
                else False
            ),
            "moving_environment_backend": (
                state.history[-1].get("moving_environment_backend")
                if state.history
                else None
            ),
            "cpp_local_operator_stats": (
                state.history[-1].get("cpp_local_operator_stats")
                if state.history
                else None
            ),
            "cycle_diagnostics": [
                {
                    "cycle": int(row["sweep"]),
                    "elapsed_s": float(row.get("elapsed_s", 0.0)),
                    "environment_build_s": float(
                        row.get("environment_build_s", 0.0)
                    ),
                    "max_local_residual": float(
                        row.get("max_local_residual", 0.0)
                    ),
                    "max_truncation_error": float(
                        row.get("max_truncation_error", 0.0)
                    ),
                    "update_s": float(
                        sum(update.get("elapsed", 0.0) for update in row["updates"])
                    ),
                }
                for row in state.history
            ],
            "history": history,
        }
    finally:
        state.close()


def plot_results(results, exact_energy, output):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    colors = {"SU2-DMRG": "#277da1", "SU2-LETTA": "#f3722c"}
    markers = {"SU2-DMRG": "o", "SU2-LETTA": "s"}
    for method in colors:
        rows = [row for row in results if row["method"] == method]
        axes[0].semilogy(
            [row["D"] for row in rows],
            [max(abs(row["error"]), 1.0e-15) for row in rows],
            marker=markers[method],
            color=colors[method],
            label=method,
        )
    axes[0].set(xlabel="reduced D", ylabel="absolute CASCI error (Ha)")
    axes[0].legend(frameon=False)

    labels = [f"{row['method'].replace('SU2-', '')}\nD={row['D']}" for row in results]
    times = [row["setup_s"] + row["optimization_s"] for row in results]
    axes[1].bar(
        np.arange(len(results)),
        times,
        color=[colors[row["method"]] for row in results],
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(results)), labels, fontsize=8)
    axes[1].set_ylabel("setup + optimization (s)")

    for row in results:
        history = row["history"]
        if not history:
            continue
        axes[2].semilogy(
            [item["cycle"] for item in history],
            [max(abs(item["energy"] - exact_energy), 1.0e-15) for item in history],
            marker=markers[row["method"]],
            color=colors[row["method"]],
            alpha=min(0.95, 0.45 + 0.07 * row["D"]),
            label=f"{row['method']} D={row['D']}",
        )
    axes[2].set(xlabel="complete cycle", ylabel="absolute CASCI error (Ha)")
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("Pyrazine CAS(4,4)/STO-3G, singlet; identical reduced Hamiltonian")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_dimensions(value):
    return tuple(int(item) for item in value.split(",") if item.strip())


def run(args):
    mol = Molecule(
        atom=pyrazine_geometry(),
        basis=args.basis,
        unit="angstrom",
        charge=0,
        spin=0,
    )
    started = perf_counter()
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mf = RHF(mol).run(tol=1.0e-11, max_cycle=100, verbose=0)
    reference_s = perf_counter() - started
    if not mf.converged:
        raise RuntimeError("pyrazine RHF did not converge.")
    active, pi_weights, mo_coeff = select_pi_active_space(mol, mf)
    started = perf_counter()
    exact = CASCI(mf, ncas=4, nelecas=4, multiplicity=1).run(
        nstates=1,
        mo_coeff=mo_coeff,
        method="direct_spin0_symm",
    )
    exact_s = perf_counter() - started
    exact_energy = float(np.asarray(exact.e_tot).reshape(-1)[0])

    results = []
    active_integrals = None
    for D in args.dmrg_D:
        row, h1e, eri, ecore = run_dmrg(
            mf,
            mo_coeff,
            exact_energy,
            D=D,
            nsweeps=args.dmrg_sweeps,
        )
        results.append(row)
        if active_integrals is None:
            active_integrals = (h1e, eri, ecore)
    h1e, eri, ecore = active_integrals
    graph = tuple((site, site + 1) for site in range(3))
    for D in args.letta_D:
        results.append(
            run_letta(
                h1e,
                eri,
                ecore,
                exact_energy,
                D=D,
                cycles=args.letta_cycles,
                graph=graph,
                gauge=args.letta_gauge,
                reuse_environments=args.letta_reuse_environments,
                dense_dim=args.letta_dense_dim,
            )
        )

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "molecule": "pyrazine",
        "geometry": "idealized planar regular ring",
        "basis": args.basis,
        "active_space": [4, 4],
        "active_mos": list(active),
        "active_pz_weights": [float(pi_weights[index]) for index in active],
        "target": {"nelec": 4, "two_s": 0},
        "hf_energy": float(mf.e_tot),
        "exact_casci_energy": exact_energy,
        "core_energy": ecore,
        "reference_build_s": float(reference_s),
        "exact_casci_s": float(exact_s),
        "letta_tie_graph": [list(edge) for edge in graph],
        "hamiltonian_is_nearest_neighbor": False,
        "results": results,
    }
    json_path = prefix.with_suffix(".json")
    figure_path = prefix.with_suffix(".png")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    plot_results(results, exact_energy, figure_path)
    print(json.dumps(payload, indent=2))
    print(f"figure: {figure_path}")
    print(f"data: {json_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--dmrg-D", type=_parse_dimensions, default=(1, 2, 4, 8))
    parser.add_argument("--letta-D", type=_parse_dimensions, default=(1, 2))
    parser.add_argument("--dmrg-sweeps", type=int, default=6)
    parser.add_argument("--letta-cycles", type=int, default=6)
    parser.add_argument("--letta-dense-dim", type=int, default=0)
    parser.add_argument(
        "--letta-gauge", choices=("conditional", "none"), default="conditional"
    )
    parser.add_argument(
        "--letta-reuse-environments",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/private/tmp/pyrazine_su2_dmrg_vs_letta"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
