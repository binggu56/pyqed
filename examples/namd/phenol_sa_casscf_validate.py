#!/usr/bin/env python3
"""Validate the phenol O--H photodissociation cut before bulk sampling.

The calculation uses six spin-pure singlet roots in a CAS(10e,10o) space.
An optimized SA-CASSCF orbital frame is projected along the cut and a cheap
CASCI screening scan locates the two crossing regions.  Candidate crossings
should subsequently be rerun with ``--macro-cycles`` greater than zero.
Every point is cached independently so interrupted scans are restartable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci, gto, mcscf, scf
from scipy.optimize import linear_sum_assignment

from pyqed.units import au2ev
from pyqed.models.phenol_coordinates import (
    PHENOL_SPECIES,
    PhenolReactiveChart,
)

HARTREE_TO_EV = au2ev


def molecule(geometry, basis):
    return gto.M(
        atom=list(zip(PHENOL_SPECIES, np.asarray(geometry))),
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
        max_memory=12000,
    )


def lowdin_coefficients(mol, coefficients):
    overlap = mol.intor_symmetric("int1e_ovlp")
    values, vectors = np.linalg.eigh(overlap)
    overlap_half = (vectors * np.sqrt(values)) @ vectors.T
    return overlap_half @ np.asarray(coefficients)


def ao_subspaces(mol):
    labels = mol.ao_labels()
    pi = np.asarray(
        [
            index
            for index, label in enumerate(labels)
            if (" C " in label or " O " in label) and "2pz" in label
        ],
        dtype=int,
    )
    oh = np.asarray(
        [
            index
            for index, label in enumerate(labels)
            if (
                label.startswith("6 O")
                and ("2px" in label or "2py" in label)
            )
            or label.startswith("7 H")
        ],
        dtype=int,
    )
    return pi, oh


def initial_active_space(mf):
    """Select 4 occupied pi + 1 occupied O--H and 3 pi* + 2 diffuse orbitals."""

    nocc = mf.mol.nelectron // 2
    transformed = lowdin_coefficients(mf.mol, mf.mo_coeff)
    pi, oh = ao_subspaces(mf.mol)
    pi_weight = np.sum(np.abs(transformed[pi]) ** 2, axis=0)
    oh_weight = np.sum(np.abs(transformed[oh]) ** 2, axis=0)
    occupied_pi = list(np.argsort(pi_weight[:nocc])[::-1][:4])
    occupied_oh = [
        int(index)
        for index in np.argsort(oh_weight[:nocc])[::-1]
        if int(index) not in occupied_pi
    ][:1]
    virtual_pi = [
        int(index)
        for index in nocc + np.argsort(pi_weight[nocc:])[::-1]
        if mf.mo_energy[index] < 0.5
    ][:3]
    virtual_sigma = sorted(
        [index for index in range(nocc, mf.mol.nao_nr()) if index not in virtual_pi],
        key=lambda index: mf.mo_energy[index],
    )[:2]
    active = [int(index) for index in occupied_pi + occupied_oh + virtual_pi + virtual_sigma]
    core = [index for index in range(nocc) if index not in active]
    external = [
        index for index in range(nocc, mf.mol.nao_nr()) if index not in active
    ]
    return mf.mo_coeff[:, core + active + external], np.asarray(active, dtype=int)


def singlet_solver(mol, nstates):
    solver = fci.addons.fix_spin_(
        fci.direct_spin1.FCI(mol), shift=1.0, ss=0
    )
    solver.nroots = int(nstates)
    solver.conv_tol = 1.0e-8
    solver.max_cycle = 100
    return solver


def orthonormalize(mol, coefficients):
    overlap = mol.intor_symmetric("int1e_ovlp")
    metric = np.asarray(coefficients).T @ overlap @ np.asarray(coefficients)
    values, vectors = np.linalg.eigh(metric)
    return np.asarray(coefficients) @ (
        (vectors * values ** -0.5) @ vectors.T
    )


def optimize_anchor(chart, basis, nstates, macro_cycles):
    geometry = chart.geometry(chart.equilibrium)
    mol = molecule(geometry, basis)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-9)
    mo0, active = initial_active_space(mf)
    mc = mcscf.DFCASSCF(mf, 10, 10)
    mc.fcisolver = singlet_solver(mol, nstates)
    mc = mc.state_average_([1.0 / nstates] * nstates)
    mc.max_cycle_macro = int(macro_cycles)
    mc.max_cycle_micro = 3
    mc.conv_tol = 2.0e-7
    mc.conv_tol_grad = 1.0e-3
    mc.kernel(mo0)
    return {
        "geometry": geometry,
        "mo_coeff": np.asarray(mc.mo_coeff),
        "ci": np.asarray(mc.ci),
        "energies": np.asarray(mc.e_states),
        "spins": np.asarray(
            [fci.spin_op.spin_square0(state, 10, (5, 5))[0] for state in mc.ci]
        ),
        "converged": np.asarray(bool(mc.converged)),
        "active_initial": active,
    }


def projected_orbitals(old_mol, old_mo, new_mol):
    projected = scf.addons.project_mo_nr2nr(old_mol, old_mo, new_mol)
    return orthonormalize(new_mol, projected)


def tracked_rhf_orbitals(previous_mol, previous_active, mf):
    """Track only the active subspace while retaining relaxed RHF spectators."""

    cross = gto.intor_cross("int1e_ovlp", previous_mol, mf.mol)
    overlap = np.asarray(previous_active).T @ cross @ mf.mo_coeff
    scores = np.sum(np.abs(overlap) ** 2, axis=0)
    nocc = mf.mol.nelectron // 2
    active_occupied = np.argsort(scores[:nocc])[::-1][:5]
    active_virtual = nocc + np.argsort(scores[nocc:])[::-1][:5]
    active = [int(index) for index in (*active_occupied, *active_virtual)]
    core = [index for index in range(nocc) if index not in active]
    external = [
        index for index in range(nocc, mf.mol.nao_nr()) if index not in active
    ]
    mo0 = mf.mo_coeff[:, core + active + external]
    selected_overlap = overlap[:, active]
    return mo0, np.asarray(active, dtype=int), np.linalg.svd(
        selected_overlap, compute_uv=False
    )


def _orthogonal_component(coefficients, basis, overlap):
    coefficients = np.asarray(coefficients)
    if basis.size:
        coefficients = coefficients - basis @ (
            basis.conj().T @ overlap @ coefficients
        )
    return coefficients


def _metric_subspace(coefficients, overlap, size):
    metric = coefficients.conj().T @ overlap @ coefficients
    values, vectors = np.linalg.eigh(metric)
    selected = np.argsort(values)[::-1][: int(size)]
    if np.min(values[selected]) < 1.0e-9:
        raise RuntimeError("orbital complement lost numerical rank")
    return coefficients @ (
        vectors[:, selected] / np.sqrt(values[selected])[None, :]
    )


def transported_active_orbitals(previous_mol, previous_active, mf):
    """Transport the CASSCF active space and relax its orthogonal spectators."""

    overlap = mf.get_ovlp()
    active = scf.addons.project_mo_nr2nr(
        previous_mol, previous_active, mf.mol
    )
    active = orthonormalize(mf.mol, active)
    core_candidates = _orthogonal_component(
        mf.mo_coeff[:, : mf.mol.nelectron // 2], active, overlap
    )
    core = _metric_subspace(core_candidates, overlap, 20)
    occupied_active = np.column_stack((core, active))
    external_candidates = _orthogonal_component(
        mf.mo_coeff, occupied_active, overlap
    )
    external = _metric_subspace(
        external_candidates, overlap, mf.mol.nao_nr() - 30
    )
    mo0 = np.column_stack((core, active, external))
    cross = gto.intor_cross("int1e_ovlp", previous_mol, mf.mol)
    continuity = np.linalg.svd(
        previous_active.T @ cross @ active, compute_uv=False
    )
    return mo0, np.arange(20, 30), continuity


def run_point(geometry, basis, nstates, mo0, macro_cycles, *, mf=None):
    mol = molecule(geometry, basis) if mf is None else mf.mol
    if mf is None:
        mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-9)
    if macro_cycles:
        mc = mcscf.DFCASSCF(mf, 10, 10)
        mc.fcisolver = singlet_solver(mol, nstates)
        mc = mc.state_average_([1.0 / nstates] * nstates)
        mc.max_cycle_macro = int(macro_cycles)
        mc.max_cycle_micro = 3
        mc.conv_tol = 2.0e-7
        mc.conv_tol_grad = 1.0e-3
    else:
        mc = mcscf.DFCASCI(mf, 10, 10)
        mc.fcisolver = singlet_solver(mol, nstates)
    mc.kernel(mo0)
    energies = np.asarray(getattr(mc, "e_states", mc.e_tot))
    states = np.asarray(mc.ci)
    if states.ndim != 3 or states.shape[0] != int(nstates):
        raise RuntimeError(
            "the multiroot CASSCF solve did not return all requested CI roots; "
            f"received CI shape {states.shape} for nstates={nstates}"
        )
    return {
        "geometry": np.asarray(geometry),
        "mo_coeff": np.asarray(mc.mo_coeff),
        "ci": states,
        "energies": energies,
        "spins": np.asarray(
            [fci.spin_op.spin_square0(state, 10, (5, 5))[0] for state in states]
        ),
        "converged": np.asarray(bool(getattr(mc, "converged", True))),
    }


def nto_characters(mol, mo_coeff, states):
    active = np.asarray(mo_coeff)[:, 20:30]
    transformed = lowdin_coefficients(mol, active)
    pi, oh = ao_subspaces(mol)
    p_pi = transformed[pi].conj().T @ transformed[pi]
    p_oh = transformed[oh].conj().T @ transformed[oh]
    result = np.zeros((len(states), 5))
    for state in range(1, len(states)):
        # This ordering gives the ground -> excited transition convention.
        tdm = fci.direct_spin1.trans_rdm1(states[0], states[state], 10, (5, 5))
        particle, singular, hole_h = np.linalg.svd(tdm)
        hole = hole_h.conj().T[:, 0]
        particle = particle[:, 0]
        result[state] = (
            singular[0],
            np.real(hole.conj() @ p_pi @ hole),
            np.real(particle.conj() @ p_pi @ particle),
            np.real(hole.conj() @ p_oh @ hole),
            np.real(particle.conj() @ p_oh @ particle),
        )
    return result


def state_overlap(left, right, basis):
    left_mol = molecule(left["geometry"], basis)
    right_mol = molecule(right["geometry"], basis)
    cross = gto.intor_cross("int1e_ovlp", left_mol, right_mol)
    active_overlap = (
        left["mo_coeff"][:, 20:30].T
        @ cross
        @ right["mo_coeff"][:, 20:30]
    )
    matrix = np.empty((len(left["ci"]), len(right["ci"])))
    for bra in range(matrix.shape[0]):
        for ket in range(matrix.shape[1]):
            matrix[bra, ket] = abs(
                fci.addons.overlap(
                    left["ci"][bra], right["ci"][ket], 10, (5, 5), active_overlap
                )
            )
    return matrix


def plot_results(
    output,
    distances,
    energies,
    characters,
    overlaps,
    crossing_indices,
    energy_zero,
    screening,
):
    # A PES requires one geometry-independent zero.  Subtracting E0(R) at
    # every point would instead produce vertical excitation gaps.
    relative = (energies - float(energy_zero)) * HARTREE_TO_EV
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, energies.shape[1]))
    fig, panels = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)
    for state, color in enumerate(colors):
        panels[0, 0].plot(distances, relative[:, state], "o-", ms=3.2, color=color, label=f"S{state}")
    overview_title = (
        "CASCI screening energies (not relaxed PESs)"
        if screening
        else "Six adiabatic PESs"
    )
    low_title = (
        "Lowest three screening energies"
        if screening
        else "Lowest three adiabatic PESs"
    )
    panels[0, 0].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel=r"$E_i(R)-E_{S_0}(R_{\rm eq})$ (eV)",
        title=overview_title,
    )
    panels[0, 0].legend(ncol=2, fontsize=8)

    for state, color in enumerate(colors[:3]):
        panels[0, 1].plot(
            distances,
            relative[:, state],
            "o-",
            ms=4.0,
            color=color,
            label=f"S{state}",
        )
    panels[0, 1].axvline(
        distances[crossing_indices[0]],
        color="0.45",
        lw=0.9,
        ls="--",
        label=r"$S_1/S_2$ candidate",
    )
    panels[0, 1].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel=r"$E_i(R)-E_{S_0}(R_{\rm eq})$ (eV)",
        title=low_title,
    )
    panels[0, 1].legend(fontsize=7.5)

    for state in range(1, min(4, energies.shape[1])):
        panels[1, 0].plot(distances, characters[:, state, 2], "o-", ms=3, label=fr"S{state} particle $\pi$")
    panels[1, 0].set(xlabel=r"$R_{OH}$ ($\AA$)", ylabel="leading-NTO projection", ylim=(-0.03, 1.03), title="Electronic character")
    panels[1, 0].legend(fontsize=8)

    if len(overlaps):
        panels[1, 1].plot(distances[1:], np.min(np.linalg.svd(overlaps, compute_uv=False), axis=1), "o-", label="minimum overlap singular value")
        panels[1, 1].plot(distances[1:], np.max(overlaps, axis=(1, 2)), "s-", label="largest state overlap")
    panels[1, 1].set(xlabel=r"$R_{OH}$ ($\AA$)", ylabel="overlap diagnostic", ylim=(-0.03, 1.03), title="State continuity")
    panels[1, 1].legend(fontsize=8)
    png = output / "phenol_sa6_cas1010_oh_validation.png"
    pdf = output / "phenol_sa6_cas1010_oh_validation.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31+g*")
    parser.add_argument("--nstates", type=int, default=6)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--anchor-macro-cycles", type=int, default=20)
    parser.add_argument("--macro-cycles", type=int, default=0, help="0 gives the cheap projected-orbital CASCI screen")
    parser.add_argument(
        "--screen-orbitals",
        choices=("transported-active", "tracked-rhf", "projected"),
        default="transported-active",
        help="Transport the full active space and relax its spectators (default); "
        "'tracked-rhf' selects canonical RHF orbitals; "
        "'projected' freezes all anchor orbitals and is diagnostic only.",
    )
    parser.add_argument("--distances", type=float, nargs="*", default=[0.90, 1.00, 1.10, 1.20, 1.40, 1.65, 1.85, 1.95, 2.05, 2.20, 2.50, 3.00])
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/phenol_sa_casscf_validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    chart = PhenolReactiveChart()
    anchor_path = args.anchor or args.output / "phenol_sa6_anchor.npz"
    if anchor_path.exists():
        with np.load(anchor_path, allow_pickle=False) as archive:
            anchor = {key: np.asarray(archive[key]) for key in archive.files}
        if "energies" not in anchor and "e_states" in anchor:
            anchor["energies"] = anchor["e_states"]
        if "spins" not in anchor:
            anchor["spins"] = np.asarray(
                [
                    fci.spin_op.spin_square0(state, 10, (5, 5))[0]
                    for state in anchor["ci"]
                ]
            )
        anchor.setdefault("converged", np.asarray(False))
    else:
        anchor = optimize_anchor(chart, args.basis, args.nstates, args.anchor_macro_cycles)
        np.savez(anchor_path, **anchor)
    if float(np.max(anchor["spins"])) > 1.0e-5:
        raise RuntimeError("anchor contains spin-contaminated roots")

    anchor_mol = molecule(anchor["geometry"], args.basis)
    distances = np.asarray(args.distances, dtype=float)
    records = [None] * len(distances)
    anchor_distance = float(chart.equilibrium[0])
    upper = sorted(
        [index for index, value in enumerate(distances) if value >= anchor_distance],
        key=lambda index: distances[index],
    )
    lower = sorted(
        [index for index, value in enumerate(distances) if value < anchor_distance],
        key=lambda index: distances[index],
        reverse=True,
    )
    for branch in (upper, lower):
        previous_mol = anchor_mol
        previous_active = np.asarray(anchor["mo_coeff"])[:, 20:30]
        for index in branch:
            distance = float(distances[index])
            prefix = args.screen_orbitals.replace("-", "_")
            path = args.output / f"{prefix}_point_{index:03d}_r{distance:.4f}.npz"
            coordinate = chart.equilibrium.copy()
            coordinate[0] = distance
            geometry = chart.geometry(coordinate)
            mol = molecule(geometry, args.basis)
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    record = {key: np.asarray(archive[key]) for key in archive.files}
            else:
                mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-9)
                if args.screen_orbitals == "transported-active":
                    mo0, active_indices, active_singular_values = transported_active_orbitals(
                        previous_mol, previous_active, mf
                    )
                elif args.screen_orbitals == "tracked-rhf":
                    mo0, active_indices, active_singular_values = tracked_rhf_orbitals(
                        previous_mol, previous_active, mf
                    )
                else:
                    mo0 = projected_orbitals(anchor_mol, anchor["mo_coeff"], mol)
                    active_indices = np.arange(20, 30)
                    active_singular_values = np.ones(10)
                record = run_point(
                    geometry,
                    args.basis,
                    args.nstates,
                    mo0,
                    args.macro_cycles,
                    mf=mf,
                )
                record["active_indices"] = np.asarray(active_indices)
                record["active_singular_values"] = np.asarray(active_singular_values)
                np.savez(path, **record)
            if float(np.max(record["spins"])) > 1.0e-5:
                raise RuntimeError(f"spin contamination at R={distance:.4f} Angstrom")
            records[index] = record
            previous_mol = mol
            previous_active = np.asarray(record["mo_coeff"])[:, 20:30]
            print(f"R={distance:6.3f}  gaps/eV={(record['energies'][1:3] - record['energies'][:2]) * HARTREE_TO_EV}", flush=True)

    energies = np.asarray([record["energies"] for record in records])
    characters = np.asarray([
        nto_characters(molecule(record["geometry"], args.basis), record["mo_coeff"], record["ci"])
        for record in records
    ])
    overlaps = np.asarray([
        state_overlap(records[index - 1], records[index], args.basis)
        for index in range(1, len(records))
    ])
    gap12 = energies[:, 2] - energies[:, 1]
    gap01 = energies[:, 1] - energies[:, 0]
    crossing_indices = (int(np.argmin(gap12)), int(np.argmin(gap01)))
    energy_zero = float(anchor["energies"][0])
    png, pdf = plot_results(
        args.output,
        np.asarray(args.distances),
        energies,
        characters,
        overlaps,
        crossing_indices,
        energy_zero,
        not bool(args.macro_cycles),
    )
    data = args.output / "phenol_sa6_cas1010_oh_validation.npz"
    np.savez(data, distances=args.distances, energies=energies, characters=characters, overlaps=overlaps, spins=np.asarray([record["spins"] for record in records]))
    summary = {
        "method": f"SA({args.nstates})-CASSCF(10e,10o)/{args.basis}" if args.macro_cycles else f"{args.screen_orbitals} SA({args.nstates})-CASSCF active frame + CASCI(10e,10o)/{args.basis}",
        "screen_orbitals": args.screen_orbitals,
        "screening_only": not bool(args.macro_cycles),
        "candidate_pi_pi_pi_sigma_angstrom": float(args.distances[crossing_indices[0]]),
        "candidate_pi_sigma_s0_angstrom": float(args.distances[crossing_indices[1]]),
        "candidate_gaps_ev": [float(gap12[crossing_indices[0]] * HARTREE_TO_EV), float(gap01[crossing_indices[1]] * HARTREE_TO_EV)],
        "energy_zero_hartree": energy_zero,
        "maximum_spin_square": float(max(np.max(record["spins"]) for record in records)),
        "all_converged": bool(all(record["converged"] for record in records)),
        "anchor": str(anchor_path),
        "data": str(data),
        "figure": str(png),
    }
    summary_path = args.output / "phenol_sa6_cas1010_oh_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"pdf={pdf}")


if __name__ == "__main__":
    main()
