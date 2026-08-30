"""Pyrazine pi-active-space calculation with the reduced SU(2) FrontierLETTA backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf

from pyqed.letta import FrontierLETTA
from pyqed.qchem.dmrg.backends.reduced import build_spatial_reduced_hamiltonian_mpo


def pyrazine_geometry():
    """Return a planar regular-ring pyrazine geometry in Angstrom."""
    ring_radius = 1.39
    hydrogen_radius = 2.47
    atoms = []
    elements = ("N", "C", "C", "N", "C", "C")
    for index, element in enumerate(elements):
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


def select_pi_active_orbitals(mol, mf, *, ncas=6):
    """Select equal occupied/virtual frontier MOs from the six-orbital pi space."""
    overlap = mf.get_ovlp()
    values, vectors = np.linalg.eigh(overlap)
    overlap_sqrt = (vectors * np.sqrt(values)[None, :]) @ vectors.T
    orthogonal_coefficients = overlap_sqrt @ mf.mo_coeff
    pz_rows = np.array(
        [index for index, label in enumerate(mol.ao_labels()) if "2pz" in label],
        dtype=int,
    )
    if pz_rows.size != 6:
        raise RuntimeError(f"expected six C/N 2pz AOs, found {pz_rows.size}.")
    weights = np.sum(np.abs(orthogonal_coefficients[pz_rows]) ** 2, axis=0)
    ncas = int(ncas)
    if ncas < 2 or ncas > 6 or ncas % 2:
        raise ValueError("ncas must be an even integer from 2 through 6.")
    pi_space = tuple(int(index) for index in np.argsort(weights)[-6:])
    occupied_boundary = mol.nelectron // 2
    occupied_pi = sorted(
        (index for index in pi_space if index < occupied_boundary),
        key=lambda index: mf.mo_energy[index],
        reverse=True,
    )
    virtual_pi = sorted(
        (index for index in pi_space if index >= occupied_boundary),
        key=lambda index: mf.mo_energy[index],
    )
    half = ncas // 2
    if len(occupied_pi) < half or len(virtual_pi) < half:
        raise RuntimeError("could not identify balanced occupied/virtual pyrazine pi orbitals.")
    selected = np.array(
        sorted(occupied_pi[:half] + virtual_pi[:half], key=lambda index: mf.mo_energy[index]),
        dtype=int,
    )
    return tuple(int(index) for index in selected), weights


def active_space_integrals(mol, mf, active):
    """Return CAS Hamiltonian integrals, core energy, and an exact CASCI reference."""
    active = tuple(active)
    ncas = len(active)
    occupied = mol.nelectron // 2
    active_occupied = sum(index < occupied for index in active)
    nelecas = 2 * active_occupied
    if nelecas != ncas:
        raise RuntimeError(
            f"selected active space is CAS({nelecas},{ncas}), expected half-filled CAS({ncas},{ncas})."
        )
    active_set = set(active)
    core = tuple(index for index in range(occupied) if index not in active_set)
    external = tuple(
        index
        for index in range(mf.mo_coeff.shape[1])
        if index not in active_set and index not in core
    )
    order = core + active + external
    mo = mf.mo_coeff[:, order]
    cas = mcscf.CASCI(mf, ncas, nelecas)
    cas.fcisolver.spin = 0
    cas.verbose = 0
    h1e, ecore = cas.get_h1eff(mo)
    eri = ao2mo.restore(1, cas.get_h2eff(mo), ncas)
    cas_result = cas.kernel(mo)
    exact_energy = cas_result[0] if isinstance(cas_result, tuple) else cas_result
    return np.asarray(h1e), np.asarray(eri), float(ecore), float(exact_energy), nelecas


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

    active, pi_weights = select_pi_active_orbitals(mol, mf, ncas=args.ncas)
    h1e, eri, ecore, exact_energy, nelecas = active_space_integrals(mol, mf, active)
    graph = tuple((site, site + 1) for site in range(args.ncas - 1))
    hamiltonian = build_spatial_reduced_hamiltonian_mpo(
        h1e,
        eri=eri[None, None, ...],
        fully_reduced=True,
        nelec=nelecas,
        spin=0,
        ecore=ecore,
        cutoff=args.integral_cutoff,
    )

    started = time.perf_counter()
    state = FrontierLETTA(
        hamiltonian,
        graph=graph,
        D=args.D,
        adaptive_bond=args.adaptive_bond,
        seed=args.seed,
        workers=args.workers,
        we_route_memory=args.route_memory,
    )
    try:
        initial_energy = float(state.energy)
        state.run(
            nsweeps=args.cycles,
            algorithm=args.algorithm,
            tol=args.tol,
            residual_tol=args.residual_tol,
            truncation_tol=args.truncation_tol,
            consecutive_cycles=args.consecutive_cycles,
            max_local_parameters=args.max_local_parameters,
            growth_truncation_tol=args.growth_truncation_tol,
            bond_growth=args.bond_growth,
            verbose=args.verbose,
        )
        elapsed = time.perf_counter() - started
        result = {
            "molecule": "pyrazine",
            "backend": type(state).__name__,
            "geometry": "planar regular ring",
            "basis": args.basis,
            "active_space": [int(nelecas), int(args.ncas)],
            "active_mos": list(active),
            "active_pz_weights": [float(pi_weights[index]) for index in active],
            "hf_energy": hf_energy,
            "active_core_energy": ecore,
            "exact_casci_energy": exact_energy,
            "initial_su2_letta_energy": initial_energy,
            "su2_letta_energy": float(state.energy),
            "su2_letta_error": float(state.energy - exact_energy),
            "D": int(args.D),
            "adaptive_bond": bool(args.adaptive_bond),
            "bond_multiplicities": [
                {
                    f"N={sector.charge},2S={sector.irrep.two_j}": int(value)
                    for sector, value in state.reduced_bond_multiplicities(bond).items()
                }
                for bond in range(state.nsites - 1)
            ],
            "tie_graph": [list(edge) for edge in graph],
            "frontier_states": list(state.frontier_states),
            "elapsed_s": float(elapsed),
            "convergence": state.convergence_summary,
        }
    finally:
        state.close()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--ncas",
        type=int,
        choices=(2, 4, 6),
        default=2,
        help="Validated half-filled pi active-space size.",
    )
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--adaptive-bond", action="store_true")
    parser.add_argument("--growth-truncation-tol", type=float, default=5.0e-2)
    parser.add_argument("--bond-growth", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--algorithm", choices=("one_site", "two_site"), default="two_site")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--integral-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--residual-tol", type=float, default=1.0e-7)
    parser.add_argument("--truncation-tol", type=float, default=1.0e-6)
    parser.add_argument("--consecutive-cycles", type=int, default=2)
    parser.add_argument("--max-local-parameters", type=int, default=4096)
    parser.add_argument("--route-memory", type=float, default=256.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
