"""Compare U(1) and SU(2) FrontierLETTA on a pyrazine active space."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf import gto, scf

from pyqed.lattice import SpinHalfFermionSite
from pyqed.letta import FrontierLETTA
from pyqed.mps.nonabelian.models import (
    _canonical_spatial_jw_site_operators,
    spatial_annihilate_down,
    spatial_annihilate_up,
    spatial_create_down,
    spatial_create_up,
)
from pyqed.mps.nonabelian.operators import physical_leg_from_spatial_orbital
from pyqed.qchem.dmrg.backends.reduced import build_spatial_reduced_hamiltonian_mpo
from pyqed.tn import Hamiltonian, LocalTerm

from pyrazine_su2_letta import (
    active_space_integrals,
    pyrazine_geometry,
    select_pi_active_orbitals,
)


def _local_term(site_operators, coefficient):
    sites = tuple(site for site, _operator in site_operators)
    matrix = np.ones((1, 1), dtype=complex)
    for _site, operator in site_operators:
        matrix = np.kron(matrix, operator.as_dense())
    return LocalTerm(sites, matrix, coefficient=coefficient)


def build_spatial_u1_hamiltonian(h1e, eri, *, ecore=0.0, cutoff=1.0e-10):
    r"""Build the spatial Hamiltonian with conserved particle number and $2S_z$.

    This small-active-space reference uses exact local Jordan--Wigner kernels.
    It intentionally groups equal supports as :class:`LocalTerm` objects; a
    scalable molecular U(1) builder should retain the operator strings instead.
    """
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError("h1e must be square.")
    nsites = h1e.shape[0]
    if eri.shape != (nsites, nsites, nsites, nsites):
        raise ValueError("eri must have shape (ncas, ncas, ncas, ncas).")

    physical_leg = physical_leg_from_spatial_orbital()
    create = (spatial_create_up(), spatial_create_down())
    annihilate = (spatial_annihilate_up(), spatial_annihilate_down())
    terms = []

    def add(coefficient, operators, positions):
        if abs(coefficient) <= cutoff:
            return
        site_operators = _canonical_spatial_jw_site_operators(
            operators,
            positions,
            phys_leg=physical_leg,
            dtype=complex,
            cutoff=cutoff,
        )
        if site_operators:
            terms.append(_local_term(site_operators, coefficient))

    for p, q in np.argwhere(np.abs(h1e) > cutoff):
        for sigma in (0, 1):
            add(h1e[p, q], (create[sigma], annihilate[sigma]), (p, q))

    # H2 = 1/2 sum_(pqrs) (pq|rs) [E_pq E_rs - delta_qr E_ps].
    for p, q, r, s in np.argwhere(np.abs(eri) > cutoff):
        coefficient = 0.5 * eri[p, q, r, s]
        for sigma in (0, 1):
            for tau in (0, 1):
                add(
                    coefficient,
                    (
                        create[sigma],
                        annihilate[sigma],
                        create[tau],
                        annihilate[tau],
                    ),
                    (p, q, r, s),
                )
        if q == r:
            for sigma in (0, 1):
                add(
                    -coefficient,
                    (create[sigma], annihilate[sigma]),
                    (p, s),
                )

    return Hamiltonian(
        (SpinHalfFermionSite(),) * nsites,
        terms=terms,
        constant=ecore,
    )


def _target_sector_energy(hamiltonian, *, nelec, two_sz=0):
    local_charges = hamiltonian.sites[0].charges
    configurations = []
    for labels in product(range(4), repeat=len(hamiltonian.sites)):
        charge = tuple(
            sum(local_charges[label][component] for label in labels)
            for component in range(2)
        )
        if charge == (int(nelec), int(two_sz)):
            configurations.append(
                np.ravel_multi_index(labels, hamiltonian.dims)
            )
    dense = hamiltonian.to_dense()
    sector = dense[np.ix_(configurations, configurations)]
    return float(np.linalg.eigvalsh(sector)[0])


def _run_su2(hamiltonian, graph, args):
    started = perf_counter()
    state = FrontierLETTA(
        hamiltonian,
        graph=graph,
        D=args.su2_D,
        seed=args.seed,
        workers=1,
        we_route_memory=args.route_memory,
    )
    setup_seconds = perf_counter() - started
    initial_energy = float(state.energy)
    started = perf_counter()
    try:
        state.run(
            nsweeps=args.su2_cycles,
            algorithm="two_site",
            tol=args.tol,
            residual_tol=args.residual_tol,
            truncation_tol=args.su2_truncation_tol,
            consecutive_cycles=2,
            max_local_parameters=args.max_local_parameters,
            verbose=args.verbose,
        )
        optimization_seconds = perf_counter() - started
        summary = state.convergence_summary
        return {
            "symmetry": "SU(2)",
            "D": int(args.su2_D),
            "initial_energy": initial_energy,
            "energy": float(state.energy),
            "setup_s": float(setup_seconds),
            "optimization_s": float(optimization_seconds),
            "converged": bool(summary["converged"]),
            "cycles": int(summary["cycles"]),
            "storage_nbytes": int(summary["storage_nbytes"]),
            "max_local_residual": float(summary["max_local_residual"]),
            "max_truncation_error": float(summary["max_truncation_error"]),
        }
    finally:
        state.close()


def _run_u1(hamiltonian, graph, args, *, D=None):
    D = args.u1_D if D is None else int(D)
    started = perf_counter()
    state = FrontierLETTA(
        hamiltonian,
        graph=graph,
        target_charge={"n": args.ncas, "2sz": 0},
        D=D,
        seed=args.seed,
        workers=1,
        frontier_backend="identity_block",
    )
    setup_seconds = perf_counter() - started
    initial_energy = float(state.energy)
    started = perf_counter()
    try:
        state.run_two_site(
            nsweeps=args.u1_sweeps,
            tol=args.tol,
            solver="verified",
            eig_tol=args.eig_tol,
            residual_tol=args.residual_tol,
            split_strategy="svd",
            verbose=args.verbose,
        )
        optimization_seconds = perf_counter() - started
        maximum_residual = max(
            (
                update.local_update.residual_norm
                for row in state.history
                for update in row.get("updates", ())
            ),
            default=float("nan"),
        )
        return {
            "symmetry": "U(1) x U(1)",
            "D": int(D),
            "initial_energy": initial_energy,
            "energy": float(state.energy),
            "setup_s": float(setup_seconds),
            "optimization_s": float(optimization_seconds),
            "converged": bool(state.converged),
            "directional_sweeps": int(len(state.history)),
            "storage_nbytes": int(sum(tensor.nbytes for tensor in state.tensors)),
            "max_local_residual": float(maximum_residual),
        }
    finally:
        state.close()


def run(args):
    mol = gto.M(
        atom=pyrazine_geometry(),
        basis=args.basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-11
    mf.max_cycle = 100
    mf.verbose = 0
    hf_energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("pyrazine RHF did not converge.")

    active, _weights = select_pi_active_orbitals(mol, mf, ncas=args.ncas)
    h1e, eri, ecore, exact_energy, nelec = active_space_integrals(mol, mf, active)
    graph = tuple((site, site + 1) for site in range(args.ncas - 1))

    started = perf_counter()
    su2_hamiltonian = build_spatial_reduced_hamiltonian_mpo(
        h1e,
        eri=eri[None, None, ...],
        fully_reduced=True,
        nelec=nelec,
        spin=0,
        ecore=ecore,
        cutoff=args.integral_cutoff,
    )
    su2_build_seconds = perf_counter() - started
    started = perf_counter()
    u1_hamiltonian = build_spatial_u1_hamiltonian(
        h1e,
        eri,
        ecore=ecore,
        cutoff=args.integral_cutoff,
    )
    u1_build_seconds = perf_counter() - started
    u1_sector_energy = _target_sector_energy(
        u1_hamiltonian,
        nelec=nelec,
        two_sz=0,
    )
    if not np.isclose(u1_sector_energy, exact_energy, rtol=0.0, atol=2.0e-10):
        raise RuntimeError(
            "U(1) Hamiltonian validation failed: "
            f"{u1_sector_energy:.14f} != {exact_energy:.14f}."
        )

    su2 = _run_su2(su2_hamiltonian, graph, args)
    u1_matched_D = _run_u1(u1_hamiltonian, graph, args, D=args.su2_D)
    u1 = (
        u1_matched_D
        if args.u1_D == args.su2_D
        else _run_u1(u1_hamiltonian, graph, args)
    )
    for record in (su2, u1_matched_D, u1):
        record["energy_error"] = float(record["energy"] - exact_energy)
    result = {
        "molecule": "pyrazine",
        "basis": args.basis,
        "active_space": [int(nelec), int(args.ncas)],
        "active_mos": [int(index) for index in active],
        "hf_energy": hf_energy,
        "exact_casci_energy": exact_energy,
        "u1_hamiltonian_sector_energy": u1_sector_energy,
        "hamiltonian_build_s": {
            "su2": float(su2_build_seconds),
            "u1": float(u1_build_seconds),
        },
        "su2": su2,
        "u1_matched_D": u1_matched_D,
        "u1": u1,
        "matched_accuracy_speedup_su2_over_u1": float(
            u1["optimization_s"] / su2["optimization_s"]
        ),
        "matched_accuracy_storage_ratio_u1_over_su2": float(
            u1["storage_nbytes"] / su2["storage_nbytes"]
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--ncas", type=int, choices=(4, 6), default=4)
    parser.add_argument("--su2-D", type=int, default=2)
    parser.add_argument("--u1-D", type=int, default=12)
    parser.add_argument("--su2-cycles", type=int, default=8)
    parser.add_argument("--u1-sweeps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--integral-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--eig-tol", type=float, default=1.0e-10)
    parser.add_argument("--residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--su2-truncation-tol", type=float, default=5.0e-2)
    parser.add_argument("--max-local-parameters", type=int, default=4096)
    parser.add_argument("--route-memory", type=float, default=256.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
