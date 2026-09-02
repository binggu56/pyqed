#!/usr/bin/env python3
"""Build a resumable SO2 CAS(8,8)/6-31G*/SOMF electronic database.

The calculation uses equally weighted SA-CASSCF orbitals for the three lowest
singlets.  A larger pool of singlet and triplet CASCI roots is then solved in
that common active-orbital frame.  The lowest roots spanning one plane-even and
two plane-odd sectors are retained, preventing the selected state window from
changing character across geometry.  This is a pilot model: dynamic
correlation and two-center terms in the one-electron Breit--Pauli operator are
omitted.

The state-interaction construction follows the Breit--Pauli/SOMF formulation
of Heß et al., Chem. Phys. Lett. 251, 365 (1996),
https://doi.org/10.1016/0009-2614(96)00119-4.  It is an adaptation using the
native PyQED CASCI transition densities and a one-center one-electron SOC term.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci as pyscf_fci
from pyscf import mcscf as pyscf_mcscf
from pyscf import scf as pyscf_scf
from scipy.stats import qmc

from pyqed.ldr import ElectronicDatabase
from pyqed.qchem import CASSCF, Molecule
from pyqed.qchem.hf.rhf import RHF, _cross_ao_overlap_matrix
from pyqed.qchem.mcscf.casci import CASCI, CASCIFrame, make_tdm1_spin_orbital
from pyqed.qchem.mcscf.soc_si import (
    align_triplet_multiplet_phases,
    soc_state_interaction,
)
from pyqed.qchem.soc import get_soc_somf_spin_orbital
from pyqed.ldr.so2 import frame_parities, select_root_sectors
from pyqed.units import au2ev, au2wavenumber


SPECIES = ("O", "S", "O")
MS_VALUES = (-1, 0, 1)
TARGET_PLANE_PARITIES = (1, -1, -1)


def default_output():
    return (
        Path.home()
        / "Library"
        / "CloudStorage"
        / "OneDrive-西湖大学"
        / "data"
        / "pyqed"
        / "so2_cas88_somf"
    )


def geometry(coordinate):
    """Return O-S-O Cartesian coordinates in Bohr."""

    r1, r2, theta = np.asarray(coordinate, dtype=float)
    half = 0.5 * theta
    return np.asarray(
        (
            (r1 * np.cos(half), r1 * np.sin(half), 0.0),
            (0.0, 0.0, 0.0),
            (r2 * np.cos(half), -r2 * np.sin(half), 0.0),
        )
    )


def protocol(args):
    value = {
        "schema": "pyqed-so2-cas88-somf-v6",
        "system": "SO2",
        "geometry_unit": "bohr",
        "coordinates": ["r1", "r2", "theta"],
        "coordinate_units": ["bohr", "bohr", "radian"],
        "basis": str(args.basis),
        "active_space": {"electrons": args.nelecas, "orbitals": args.ncas},
        "orbitals": {
            "method": "equal-weight singlet SA-CASSCF",
            "roots": args.singlet_roots,
            "optimizer": args.orbital_backend,
            "point_group_constrained": bool(args.symmetry_adapted),
        },
        "state_interaction": {
            "singlet_roots": args.singlet_roots,
            "triplet_roots": args.triplet_roots,
            "singlet_candidate_roots": args.singlet_candidates,
            "triplet_candidate_roots": args.triplet_candidates,
            "root_selection": {
                "symmetry": "molecular-plane reflection",
                "target_parities": list(TARGET_PLANE_PARITIES),
                "ordering": "lowest energy within each requested sector",
            },
            "triplet_ms": list(MS_VALUES),
            "soc": "Breit-Pauli SOMF",
            "somf_density": "equal average over spatial singlet and Ms=0 triplet roots",
            "one_electron_soc": "one-center",
            "spin_orbital_order": "grouped",
            "spin_orbital_layout": "grouped spin-first Kronecker product",
            "exact_spin_selection": "validate zero S=0 to S=0 SOC block",
            "triplet_phase": "spin-ladder canonical",
        },
        "limitations": [
            "no dynamic correlation",
            "singlet-only orbital optimization",
            "one-center one-electron SOC approximation",
        ],
    }
    if getattr(args, "reuse_saved_orbitals", False):
        value["orbitals"].update(
            {
                "source_schema": "pyqed-so2-cas88-somf-v5",
                "optimization_repeated": False,
            }
        )
    return value


def specification(coordinate, electronic_protocol):
    return {
        "geometry": {
            "species": list(SPECIES),
            "coordinates_bohr": np.asarray(geometry(coordinate)).round(14).tolist(),
        },
        "protocol": electronic_protocol,
    }


def molecule_at(coordinate, basis):
    mol = Molecule(
        atom=list(zip(SPECIES, geometry(coordinate))),
        charge=0,
        spin=0,
        unit="bohr",
        basis=basis,
    )
    return mol.build(eri="dense")


def _metric_orthonormalize(coefficients, overlap):
    metric = coefficients.conj().T @ overlap @ coefficients
    values, vectors = np.linalg.eigh(0.5 * (metric + metric.conj().T))
    if np.min(values) < 1.0e-10:
        raise RuntimeError("transported molecular orbitals lost numerical rank")
    return coefficients @ ((vectors * values**-0.5) @ vectors.conj().T)


def transport_orbitals(old_mol, old_mo, new_mol, ncore, ncas):
    """Project the core/active/external blocks without mixing them."""

    overlap = np.asarray(new_mol.overlap)
    cross = _cross_ao_overlap_matrix(new_mol, old_mol)
    projected = np.linalg.solve(overlap, cross @ np.asarray(old_mo))
    transported = np.empty_like(projected)
    completed = np.empty((len(overlap), 0), dtype=projected.dtype)
    blocks = (
        slice(0, ncore),
        slice(ncore, ncore + ncas),
        slice(ncore + ncas, projected.shape[1]),
    )
    for block in blocks:
        candidate = np.array(projected[:, block], copy=True)
        if completed.shape[1]:
            candidate -= completed @ (completed.conj().T @ overlap @ candidate)
        candidate = _metric_orthonormalize(candidate, overlap)
        transported[:, block] = candidate
        completed = np.column_stack((completed, candidate))
    defect = np.linalg.norm(
        transported.conj().T @ overlap @ transported
        - np.eye(transported.shape[1])
    )
    if defect > 1.0e-7:
        raise RuntimeError(f"orbital transport orthogonality defect {defect:.3e}")
    return transported


def _sector(mean_field, mo_coeff, args, *, ms2, multiplicity, nroots):
    solver = CASCI(
        mean_field,
        ncas=args.ncas,
        nelecas=args.nelecas,
        ms2=ms2,
        multiplicity=multiplicity,
        verbose=max(0, args.verbose - 1),
    )
    solver.spin_root_cushion = args.spin_root_cushion
    solver.run(
        nstates=nroots,
        mo_coeff=mo_coeff,
        method="direct_ci",
        spin_root_cushion=args.spin_root_cushion,
        spin_selection_tol=args.spin_tol,
    )
    return solver


def _selected_frame(solver, roots):
    frame = solver.frame()
    return CASCIFrame(
        mol=frame.mol,
        mo_coeff=frame.mo_coeff,
        ci=tuple(frame.ci[int(root)] for root in roots),
        binary=frame.binary,
        ncore=frame.ncore,
        ncas=frame.ncas,
    )


def _select_symmetry_sectors(mol, singlet, triplets):
    singlet_parities, singlet_info = frame_parities(singlet.frame(), mol)
    triplet_parities = {}
    triplet_info = {}
    for ms, solver in triplets.items():
        values, info = frame_parities(solver.frame(), mol)
        triplet_parities[ms] = values
        triplet_info[ms] = info
    reference = triplet_parities[0]
    for ms in MS_VALUES:
        if not np.array_equal(triplet_parities[ms], reference):
            raise RuntimeError(
                f"triplet candidate symmetry order differs between Ms=0 and Ms={ms:+d}"
            )
    singlet_roots = select_root_sectors(
        singlet_parities, TARGET_PLANE_PARITIES
    )
    triplet_roots = select_root_sectors(reference, TARGET_PLANE_PARITIES)
    return singlet_roots, triplet_roots, {
        "target_plane_parities": np.asarray(TARGET_PLANE_PARITIES, dtype=int),
        "singlet_candidate_parities": singlet_parities,
        "triplet_candidate_parities": reference,
        "singlet_selected_indices": singlet_roots,
        "triplet_selected_indices": triplet_roots,
        "singlet": singlet_info,
        "triplet": triplet_info,
    }


def build_state_interaction_record(
    coordinate,
    args,
    *,
    mol,
    mean_field,
    common_mo,
    orbital_history=(),
    rhf_energy=None,
    orbital_source="optimized",
    started=None,
):
    """Build the fixed-sector v6 state interaction in supplied CASSCF orbitals."""

    if args.singlet_roots != len(TARGET_PLANE_PARITIES):
        raise ValueError(
            f"v6 requires {len(TARGET_PLANE_PARITIES)} selected singlet roots"
        )
    if args.triplet_roots != len(TARGET_PLANE_PARITIES):
        raise ValueError(
            f"v6 requires {len(TARGET_PLANE_PARITIES)} selected triplet roots"
        )
    if args.singlet_candidates < args.singlet_roots:
        raise ValueError("singlet candidate pool is smaller than the selected window")
    if args.triplet_candidates < args.triplet_roots:
        raise ValueError("triplet candidate pool is smaller than the selected window")
    if started is None:
        started = time.perf_counter()

    singlet = _sector(
        mean_field,
        common_mo,
        args,
        ms2=0,
        multiplicity=1,
        nroots=args.singlet_candidates,
    )
    triplets = {
        ms: _sector(
            mean_field,
            common_mo,
            args,
            ms2=2 * ms,
            multiplicity=3,
            nroots=args.triplet_candidates,
        )
        for ms in MS_VALUES
    }
    phase_diagnostics = align_triplet_multiplet_phases(triplets)
    singlet_roots, triplet_roots, sector_diagnostics = (
        _select_symmetry_sectors(mol, singlet, triplets)
    )

    states = [(singlet, int(root)) for root in singlet_roots]
    labels = [f"S{root}" for root in range(args.singlet_roots)]
    for selected_root, candidate_root in enumerate(triplet_roots):
        for ms in MS_VALUES:
            states.append((triplets[ms], int(candidate_root)))
            labels.append(f"T{selected_root}(Ms={ms:+d})")

    density_states = states[: args.singlet_roots] + [
        (triplets[0], int(root)) for root in triplet_roots
    ]
    hso = get_soc_somf_spin_orbital(
        mean_field,
        representation="mo",
        mo_coeff=singlet.mo_cas,
        states=density_states,
        one_center=True,
        order="grouped",
    )
    interaction = soc_state_interaction(states, hso=hso, order="grouped")

    spin_square = np.asarray(
        np.real_if_close(
            [solver.spin_square(root) for solver, root in states], tol=1000
        ),
        dtype=float,
    )
    expected = np.asarray(
        [0.0] * args.singlet_roots
        + [2.0] * (3 * args.triplet_roots)
    )
    spin_error = float(np.max(np.abs(spin_square - expected)))
    hermiticity = float(
        np.linalg.norm(interaction.h_total - interaction.h_total.conj().T)
    )
    triplet_components = np.asarray(
        [
            [triplets[ms].e_tot[int(root)] for ms in MS_VALUES]
            for root in triplet_roots
        ]
    )
    triplet_degeneracy = float(np.max(np.ptp(triplet_components, axis=1)))
    if spin_error > args.spin_tol:
        raise RuntimeError(f"spin-purity error {spin_error:.3e}")
    if hermiticity > 1.0e-10:
        raise RuntimeError(f"SOC Hamiltonian Hermiticity defect {hermiticity:.3e}")
    if triplet_degeneracy > 1.0e-8:
        raise RuntimeError(
            f"triplet Ms energy splitting before SOC {triplet_degeneracy:.3e} Eh"
        )

    record = {
        "coordinate": np.asarray(coordinate, dtype=float),
        "geometry": geometry(coordinate),
        "labels": labels,
        "scalar_energies": interaction.energies,
        "spin_square": spin_square,
        "h_scalar": interaction.h_scalar,
        "h_soc": interaction.h_soc,
        "h_total": interaction.h_total,
        "soc_eigenvalues": interaction.eigenvalues,
        "soc_eigenvectors": interaction.eigenvectors,
        "hso_active_spin_orbital": hso,
        "mo_coeff": np.asarray(common_mo),
        "active_orbitals": np.asarray(singlet.mo_cas),
        "singlet_frame": _selected_frame(singlet, singlet_roots),
        "triplet_frames": {
            ms: _selected_frame(triplets[ms], triplet_roots) for ms in MS_VALUES
        },
        "candidate_roots": {
            **sector_diagnostics,
            "singlet_energies": np.asarray(singlet.e_tot),
            "triplet_energies": np.asarray(triplets[0].e_tot),
        },
        "rhf_energy": np.asarray(np.nan if rhf_energy is None else rhf_energy),
        "orbital_history": list(orbital_history),
        "diagnostics": {
            "spin_error": spin_error,
            "hermiticity_defect": hermiticity,
            "triplet_ms_degeneracy_eh": triplet_degeneracy,
            "maximum_soc_cm-1": float(
                np.max(np.abs(interaction.h_soc)) * au2wavenumber
            ),
            "seconds": time.perf_counter() - started,
            "orbital_source": str(orbital_source),
            "triplet_phase_off_diagonal": phase_diagnostics["off_diagonal"],
            "triplet_phase_amplitude_error": phase_diagnostics["amplitude_error"],
        },
    }
    return validate_spin_selection(record, args.singlet_roots)


def _pyscf_sa_casscf_orbitals(mol, mo0, args):
    """Optimize common singlet orbitals with density-fitted PySCF CASSCF."""

    pmol = mol.topyscf()
    pmol.symmetry = bool(args.symmetry_adapted)
    pmol.build()
    mean_field = pyscf_scf.RHF(pmol).density_fit()
    mean_field.conv_tol = args.scf_tol
    mean_field.max_cycle = args.scf_cycles
    mean_field.verbose = max(0, args.verbose - 1)
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton().run(
            conv_tol=args.scf_tol,
            max_cycle=args.scf_cycles,
            verbose=max(0, args.verbose - 1),
        )
    if not mean_field.converged:
        raise RuntimeError("density-fitted PySCF RHF did not converge")

    ci_solver = pyscf_fci.addons.fix_spin_(
        pyscf_fci.direct_spin1.FCI(pmol), shift=1.0, ss=0.0
    )
    ci_solver.nroots = args.singlet_roots
    ci_solver.conv_tol = min(args.casscf_tol * 0.1, 1.0e-9)
    driver = pyscf_mcscf.DFCASSCF(mean_field, args.ncas, args.nelecas)
    driver.fcisolver = ci_solver
    if args.singlet_roots > 1:
        driver = driver.state_average_(
            np.full(args.singlet_roots, 1.0 / args.singlet_roots)
        )
    driver.max_cycle_macro = args.casscf_cycles
    driver.max_cycle_micro = args.micro_cycles
    driver.conv_tol = args.casscf_tol
    driver.conv_tol_grad = args.casscf_grad_tol
    driver.max_memory = args.max_memory
    driver.verbose = args.verbose
    history = []

    def callback(environment):
        item = {
            "macro": int(environment.get("imacro", len(history))),
            "energy": float(environment.get("e_tot", np.nan)),
            "gradient_norm": float(environment.get("norm_gorb", np.nan)),
        }
        if history and history[-1]["macro"] == item["macro"]:
            history[-1] = item
        else:
            if history and args.verbose:
                previous = history[-1]
                print(
                    f"PySCF CASSCF cycle {previous['macro']:3d}  "
                    f"E = {previous['energy']:.10f}  "
                    f"|g| = {previous['gradient_norm']:.3e}",
                    flush=True,
                )
            history.append(item)

    driver.callback = callback
    permutation = np.asarray(mol.pyscf_ao_permutation(pmol), dtype=int)
    initial = None if mo0 is None else np.asarray(mo0)[np.argsort(permutation)]
    driver.kernel(initial)
    if history and args.verbose:
        item = history[-1]
        print(
            f"PySCF CASSCF cycle {item['macro']:3d}  E = {item['energy']:.10f}  "
            f"|g| = {item['gradient_norm']:.3e}",
            flush=True,
        )
    final_gradient = np.inf if not history else history[-1]["gradient_norm"]
    if not driver.converged and final_gradient > args.casscf_grad_tol:
        raise RuntimeError(
            "density-fitted PySCF SA-CASSCF did not converge: "
            f"|g|={final_gradient:.3e}"
        )
    return np.asarray(driver.mo_coeff)[permutation], history


def validate_spin_selection(record, n_singlets, *, tolerance_cm=1.0e-7):
    """Require the exact rank-one singlet--singlet SOC selection rule."""

    h_soc = np.asarray(record["h_soc"], dtype=complex)
    forbidden_cm = float(
        np.max(np.abs(h_soc[:n_singlets, :n_singlets])) * au2wavenumber
    )
    diagnostics = dict(record["diagnostics"])
    diagnostics["singlet_selection_rule_cm-1"] = forbidden_cm
    record["diagnostics"] = diagnostics
    if forbidden_cm > float(tolerance_cm):
        raise RuntimeError(
            "spin-orbit matrix violates the S=0 to S=0 selection rule: "
            f"{forbidden_cm:.3e} cm^-1"
        )
    return record


def repair_v2_grouped_soc_operator(matrix):
    """Undo the v2 grouped/interleaved permutation without new integrals."""

    matrix = np.asarray(matrix, dtype=complex)
    n = matrix.shape[0] // 2
    interleaved_from_grouped = np.empty(2 * n, dtype=int)
    interleaved_from_grouped[0::2] = np.arange(n)
    interleaved_from_grouped[1::2] = n + np.arange(n)
    return matrix[np.ix_(interleaved_from_grouped, interleaved_from_grouped)]


def migrate_v2_record(record, n_singlets, n_triplets):
    """Rebuild a v3 SOC state interaction from stored v2 CI frames."""

    started = time.perf_counter()
    record = copy.deepcopy(record)
    hso = repair_v2_grouped_soc_operator(record["hso_active_spin_orbital"])
    states = [
        (record["singlet_frame"], root) for root in range(n_singlets)
    ] + [
        (record["triplet_frames"][ms], root)
        for root in range(n_triplets)
        for ms in MS_VALUES
    ]
    h_soc = np.zeros((len(states), len(states)), dtype=complex)
    for left, (left_frame, left_root) in enumerate(states):
        for right in range(left, len(states)):
            if left < n_singlets and right < n_singlets:
                value = 0.0j
            else:
                right_frame, right_root = states[right]
                density = make_tdm1_spin_orbital(
                    left_frame.ci[left_root],
                    right_frame.ci[right_root],
                    left_frame.binary,
                    right_frame.binary,
                    order="grouped",
                )
                value = np.einsum("uv,uv->", hso, density, optimize=True)
            if left == right:
                h_soc[left, left] = 0.5 * (value + value.conjugate())
            else:
                h_soc[left, right] = value
                h_soc[right, left] = value.conjugate()
    h_total = np.asarray(record["h_scalar"], dtype=complex) + h_soc
    eigenvalues, eigenvectors = np.linalg.eigh(h_total)
    record.update(
        {
            "hso_active_spin_orbital": hso,
            "h_soc": h_soc,
            "h_total": h_total,
            "soc_eigenvalues": eigenvalues,
            "soc_eigenvectors": eigenvectors,
        }
    )
    diagnostics = dict(record["diagnostics"])
    diagnostics.pop("raw_forbidden_singlet_soc_cm-1", None)
    diagnostics.pop("corrected_singlet_soc_cm-1", None)
    diagnostics.update(
        {
            "hermiticity_defect": float(
                np.linalg.norm(h_total - h_total.conj().T)
            ),
            "maximum_soc_cm-1": float(
                np.max(np.abs(h_soc)) * au2wavenumber
            ),
            "soc_ordering_migration_seconds": time.perf_counter() - started,
            "soc_ordering_migrated_from": "v2 interleaved/grouped permutation",
        }
    )
    record["diagnostics"] = diagnostics
    return validate_spin_selection(record, n_singlets)


def augment_from_saved_orbitals(source_record, args):
    """Build a v6 record without repeating SCF or CASSCF optimization."""

    started = time.perf_counter()
    coordinate = np.asarray(source_record["coordinate"], dtype=float)
    mol = molecule_at(coordinate, args.basis)
    common_mo = np.asarray(source_record["mo_coeff"])
    overlap_defect = float(
        np.linalg.norm(
            common_mo.conj().T @ np.asarray(mol.overlap) @ common_mo
            - np.eye(common_mo.shape[1])
        )
    )
    if overlap_defect > 1.0e-7:
        raise RuntimeError(
            "saved CASSCF orbitals are incompatible with the rebuilt AO basis: "
            f"orthonormality defect={overlap_defect:.3e}"
        )
    mean_field = RHF(mol)
    mean_field.mo_coeff = common_mo
    mean_field.e_nuc = float(mol.energy_nuc())
    mean_field.e_tot = float(np.asarray(source_record["rhf_energy"]))
    record = build_state_interaction_record(
        coordinate,
        args,
        mol=mol,
        mean_field=mean_field,
        common_mo=common_mo,
        orbital_history=source_record.get("orbital_history", ()),
        rhf_energy=source_record["rhf_energy"],
        orbital_source="reused-v5-casscf",
        started=started,
    )
    record["diagnostics"]["saved_orbital_orthonormality_defect"] = overlap_defect
    return record


def optimize_orbitals(coordinate, args, *, anchor_record=None):
    """Optimize the common SA-CASSCF orbitals at one geometry."""

    started = time.perf_counter()
    mol = molecule_at(coordinate, args.basis)
    mean_field = RHF(mol).run(
        tol=args.scf_tol,
        max_cycle=args.scf_cycles,
        verbose=max(0, args.verbose - 1),
    )
    if not mean_field.converged:
        raise RuntimeError("RHF did not converge")

    ncore = (int(mol.nelec) - args.nelecas) // 2
    mo0 = None
    if anchor_record is not None:
        old_mol = molecule_at(anchor_record["coordinate"], args.basis)
        mo0 = transport_orbitals(
            old_mol, anchor_record["mo_coeff"], mol, ncore, args.ncas
        )

    if args.casscf_cycles and args.orbital_backend == "pyscf":
        common_mo, orbital_history = _pyscf_sa_casscf_orbitals(mol, mo0, args)
    elif args.casscf_cycles:
        orbital_model = CASSCF(
            mean_field,
            ncas=args.ncas,
            nelecas=args.nelecas,
            ms2=0,
            multiplicity=1,
            max_cycle=args.casscf_cycles,
            max_micro_cycle=args.micro_cycles,
            conv_tol=args.casscf_tol,
            conv_tol_grad=args.casscf_grad_tol,
            conv_tol_step=args.casscf_step_tol,
            max_step=args.max_step,
            ci_method="direct_ci",
            coupling="qn",
            verbose=args.verbose,
        )
        orbital_model.state_average(
            np.full(args.singlet_roots, 1.0 / args.singlet_roots)
        )
        orbital_model.run(nstates=args.singlet_roots, mo_coeff=mo0)
        if not orbital_model.converged:
            raise RuntimeError("SA-CASSCF orbital optimization did not converge")
        common_mo = np.asarray(orbital_model.mo_coeff)
        orbital_history = orbital_model.history
    else:
        common_mo = np.asarray(mean_field.mo_coeff if mo0 is None else mo0)
        orbital_history = []

    return mol, mean_field, common_mo, orbital_history, started


def calculate(coordinate, args, *, anchor_record=None):
    mol, mean_field, common_mo, orbital_history, started = optimize_orbitals(
        coordinate, args, anchor_record=anchor_record
    )

    return build_state_interaction_record(
        coordinate,
        args,
        mol=mol,
        mean_field=mean_field,
        common_mo=common_mo,
        orbital_history=orbital_history,
        rhf_energy=mean_field.e_tot,
        orbital_source="optimized",
        started=started,
    )


def design(args):
    if args.points_file is not None:
        payload = json.loads(Path(args.points_file).read_text())
        values = payload.get("points", payload)
        points = [
            (str(item["name"]), np.asarray(item["coordinate"], dtype=float))
            for item in values
        ]
        if not points:
            raise ValueError("--points-file contains no geometries")
        return points
    center = np.asarray((args.r0, args.r0, np.deg2rad(args.theta0_deg)))
    points = [("center", center)]
    if args.design in {"star", "sobol"}:
        dr = args.dr
        dt = np.deg2rad(args.dtheta_deg)
        points.extend(
            (
                ("symmetric-minus", center + (-dr, -dr, 0.0)),
                ("symmetric-plus", center + (dr, dr, 0.0)),
                ("asymmetric-canonical", center + (dr, -dr, 0.0)),
                ("bend-minus", center + (0.0, 0.0, -dt)),
                ("bend-plus", center + (0.0, 0.0, dt)),
            )
        )
    if args.design == "sobol":
        if args.samples < len(points):
            raise ValueError(
                f"--samples must be at least {len(points)} for the nested star"
            )
        sampler = qmc.Sobol(d=3, scramble=True, seed=args.seed)
        count = max(1, int(np.ceil(np.log2(2 * args.samples))))
        candidates = sampler.random_base2(count)
        existing = {
            tuple(np.round(coordinate, 12)) for _name, coordinate in points
        }
        for candidate in candidates:
            symmetric = args.rs_min + candidate[0] * (args.rs_max - args.rs_min)
            asymmetric = candidate[1] * args.ra_max
            angle = np.deg2rad(
                args.theta_min_deg
                + candidate[2] * (args.theta_max_deg - args.theta_min_deg)
            )
            coordinate = np.asarray(
                (symmetric + asymmetric, symmetric - asymmetric, angle)
            )
            key = tuple(np.round(coordinate, 12))
            if key in existing:
                continue
            existing.add(key)
            points.append((f"sobol-{len(points) - 6:03d}", coordinate))
            if len(points) == args.samples:
                break
        if len(points) != args.samples:
            raise RuntimeError("Sobol candidate pool did not fill the requested design")
    return points


def _calculate_worker(name, coordinate, args, anchor_record):
    return name, calculate(coordinate, args, anchor_record=anchor_record)


def _sampling_coordinate(coordinate):
    r1, r2, theta = np.asarray(coordinate, dtype=float)
    return np.asarray((0.5 * (r1 + r2), 0.5 * abs(r1 - r2), theta))


def _nearest_anchor(coordinate, records, args):
    scale = np.asarray(
        (
            args.rs_max - args.rs_min,
            args.ra_max,
            np.deg2rad(args.theta_max_deg - args.theta_min_deg),
        )
    )
    target = _sampling_coordinate(coordinate)
    distances = {
        name: float(
            np.linalg.norm(
                (target - _sampling_coordinate(record["coordinate"])) / scale
            )
        )
        for name, record in records.items()
    }
    name = min(distances, key=distances.get)
    return distances[name], name, records[name]


def plot(records, output):
    center = records.get("center", next(iter(records.values())))
    zero = float(np.min(center["scalar_energies"]))
    figure, axes = plt.subplots(
        1, 3,
        figsize=(11.0, 3.35),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.5, 1.0, 1.0)},
    )
    for index, (name, record) in enumerate(records.items()):
        scalar = (record["scalar_energies"] - zero) * au2ev
        mixed = (record["soc_eigenvalues"] - zero) * au2ev
        axes[0].plot(
            np.full(len(scalar), index) - 0.08, scalar, "o", ms=3.2,
            color="#009E73",
        )
        axes[0].plot(
            np.full(len(mixed), index) + 0.08, mixed, "x", ms=3.5,
            color="#D55E00",
        )
    names = list(records)
    nsinglet = sum(label.startswith("S") for label in center["labels"])
    ntriplet = (len(center["labels"]) - nsinglet) // 3
    coupling = np.empty((nsinglet * ntriplet, len(records)))
    coupling_labels = []
    pairs = [
        (singlet, triplet)
        for singlet in range(nsinglet)
        for triplet in range(ntriplet)
    ]
    for row, (singlet, triplet) in enumerate(pairs):
        coupling_labels.append(rf"S{singlet}$\leftrightarrow$T{triplet}")
        triplet_slice = slice(nsinglet + 3 * triplet, nsinglet + 3 * triplet + 3)
        for column, record in enumerate(records.values()):
            coupling[row, column] = (
                np.linalg.norm(record["h_soc"][singlet, triplet_slice])
                * au2wavenumber
            )
    coupling_image = axes[1].imshow(
        coupling, origin="upper", cmap="viridis", aspect="auto"
    )
    figure.colorbar(
        coupling_image,
        ax=axes[1],
        pad=0.02,
        label=r"$(\sum_{M_S}|H_{ST}|^2)^{1/2}$ (cm$^{-1}$)",
    )
    tick_step = max(1, int(np.ceil(len(names) / 6)))
    ticks = np.unique(np.r_[np.arange(0, len(names), tick_step), len(names) - 1])
    axes[1].set(
        xticks=ticks,
        xticklabels=ticks,
        xlabel="sample index",
        yticks=np.arange(len(coupling_labels)),
        yticklabels=coupling_labels,
    )
    axes[1].tick_params(axis="x", labelsize=8)

    matrix = np.abs(center["h_soc"]) * au2wavenumber
    image = axes[2].imshow(matrix, origin="upper", cmap="magma")
    figure.colorbar(image, ax=axes[2], pad=0.02, label=r"$|H_{\rm SO}|$ (cm$^{-1}$)")
    axes[0].set(
        xticks=ticks,
        xticklabels=ticks,
        xlabel="sample index",
        ylabel="Energy relative to pilot minimum (eV)",
    )
    axes[2].set(xlabel="state", ylabel="state")
    axes[0].plot([], [], "o", ms=4, color="#009E73", label="spin-free")
    axes[0].plot([], [], "x", ms=4, color="#D55E00", label="SOC-mixed")
    axes[0].legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
    )
    for label, axis in zip("abc", axes):
        axis.text(
            -0.10,
            1.02,
            label,
            transform=axis.transAxes,
            va="bottom",
            fontweight="bold",
        )
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument(
        "--design", choices=("anchor", "star", "sobol"), default="star"
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--points-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--ncas", type=int, default=8)
    parser.add_argument("--nelecas", type=int, default=8)
    parser.add_argument("--singlet-roots", type=int, default=3)
    parser.add_argument("--triplet-roots", type=int, default=3)
    parser.add_argument("--singlet-candidates", type=int, default=6)
    parser.add_argument("--triplet-candidates", type=int, default=6)
    parser.add_argument("--r0", type=float, default=2.70)
    parser.add_argument("--theta0-deg", type=float, default=119.5)
    parser.add_argument("--dr", type=float, default=0.06)
    parser.add_argument("--dtheta-deg", type=float, default=4.0)
    parser.add_argument("--rs-min", type=float, default=2.55)
    parser.add_argument("--rs-max", type=float, default=3.05)
    parser.add_argument("--ra-max", type=float, default=0.25)
    parser.add_argument("--theta-min-deg", type=float, default=100.0)
    parser.add_argument("--theta-max-deg", type=float, default=140.0)
    parser.add_argument("--scf-tol", type=float, default=1.0e-10)
    parser.add_argument("--scf-cycles", type=int, default=120)
    parser.add_argument("--casscf-cycles", type=int, default=40)
    parser.add_argument(
        "--orbital-backend", choices=("pyscf", "pyqed"), default="pyscf"
    )
    parser.add_argument(
        "--symmetry-adapted",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--micro-cycles", type=int, default=6)
    parser.add_argument("--casscf-tol", type=float, default=2.0e-7)
    parser.add_argument("--casscf-grad-tol", type=float, default=2.0e-5)
    parser.add_argument("--casscf-step-tol", type=float, default=1.0e-3)
    parser.add_argument("--max-step", type=float, default=0.04)
    parser.add_argument("--max-memory", type=int, default=8000)
    parser.add_argument("--spin-root-cushion", type=int, default=10)
    parser.add_argument("--spin-tol", type=float, default=1.0e-6)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    database_path = args.output / "electronic.sqlite"
    electronic_protocol = protocol(args)
    points = design(args)
    run_id = "so2-cas88-somf-" + time.strftime("%Y%m%dT%H%M%S")
    database = ElectronicDatabase(database_path)
    database.start_run(
        run_id,
        metadata={"protocol": electronic_protocol, "design": args.design},
        status="sampling",
    )
    records = {}
    try:
        point_index = {name: index for index, (name, _coordinate) in enumerate(points)}

        def cached_record(name, coordinate):
            key = specification(coordinate, electronic_protocol)
            record = database.get(key)
            source = "database"
            if record is None:
                legacy_protocol = json.loads(json.dumps(electronic_protocol))
                legacy_protocol["schema"] = "pyqed-so2-cas88-somf-v2"
                legacy_protocol["state_interaction"].pop("spin_orbital_layout")
                legacy_protocol["state_interaction"]["exact_spin_selection"] = (
                    "S=0 to S=0 SOC block is zero"
                )
                legacy_key = specification(coordinate, legacy_protocol)
                legacy = database.get(legacy_key)
                if legacy is not None:
                    print(f"[SO2] migrating SOC ordering for {name}", flush=True)
                    record = migrate_v2_record(
                        legacy, args.singlet_roots, args.triplet_roots
                    )
                    _record_id, inserted = database.put(
                        key,
                        record,
                        metadata={
                            "name": name,
                            "diagnostics": record["diagnostics"],
                            "migrated_from": database.identifier(legacy_key),
                        },
                    )
                    source = "soc-ordering-migration" if inserted else "database-race"
            return key, record, source

        def store_record(name, coordinate, key, record, source):
            records[name] = record
            record_id = database.identifier(key)
            database.note_run_record(
                run_id,
                record_id,
                {
                    "index": [point_index[name]],
                    "name": name,
                    "coordinate": np.asarray(coordinate).tolist(),
                },
                source,
            )
            print(
                f"[SO2] {name}: max SOC={record['diagnostics']['maximum_soc_cm-1']:.2f} "
                f"cm^-1, time={record['diagnostics']['seconds']:.1f} s",
                flush=True,
            )

        anchor_name, anchor_coordinate = points[0]
        anchor_key, anchor, anchor_source = cached_record(
            anchor_name, anchor_coordinate
        )
        if anchor is None:
            print(f"[SO2] calculating {anchor_name}: {anchor_coordinate}", flush=True)
            anchor = calculate(anchor_coordinate, args)
            _record_id, inserted = database.put(
                anchor_key,
                anchor,
                metadata={
                    "name": anchor_name,
                    "diagnostics": anchor["diagnostics"],
                },
            )
            anchor_source = "calculated" if inserted else "database-race"
        else:
            print(f"[SO2] reusing {anchor_name}", flush=True)
        store_record(
            anchor_name, anchor_coordinate, anchor_key, anchor, anchor_source
        )

        pending = []
        for name, coordinate in points[1:]:
            key, record, source = cached_record(name, coordinate)
            if record is None:
                pending.append((name, coordinate, key))
            else:
                print(f"[SO2] reusing {name}", flush=True)
                store_record(name, coordinate, key, record, source)

        missing = []
        if args.cache_only:
            missing = [name for name, _coordinate, _key in pending]
            pending = []

        if args.workers == 1:
            for name, coordinate, key in pending:
                print(f"[SO2] calculating {name}: {coordinate}", flush=True)
                record = calculate(coordinate, args, anchor_record=anchor)
                _record_id, inserted = database.put(
                    key,
                    record,
                    metadata={"name": name, "diagnostics": record["diagnostics"]},
                )
                source = "calculated" if inserted else "database-race"
                store_record(name, coordinate, key, record, source)
        elif pending:
            context = mp.get_context("spawn")
            attempts = {name: 0 for name, _coordinate, _key in pending}
            errors = {}
            exhausted_errors = {}
            while pending:
                ranked = []
                for item in pending:
                    name, coordinate, _key = item
                    distance, anchor_name, local_anchor = _nearest_anchor(
                        coordinate, records, args
                    )
                    ranked.append(
                        (attempts[name], distance, item, anchor_name, local_anchor)
                    )
                ranked.sort(key=lambda entry: (entry[0], entry[1]))
                batch = ranked[: min(args.workers, len(ranked))]
                print(
                    "[SO2] local transport batch: "
                    + ", ".join(
                        f"{item[0]}<-{anchor_name} (d={distance:.3f})"
                        for _attempt, distance, item, anchor_name, _anchor in batch
                    ),
                    flush=True,
                )
                with ProcessPoolExecutor(
                    max_workers=len(batch), mp_context=context
                ) as executor:
                    futures = {}
                    for attempt, _distance, item, _anchor_name, local_anchor in batch:
                        name, coordinate, key = item
                        worker_args = copy.copy(args)
                        worker_args.casscf_cycles = args.casscf_cycles * (attempt + 1)
                        future = executor.submit(
                            _calculate_worker,
                            name,
                            coordinate,
                            worker_args,
                            local_anchor,
                        )
                        futures[future] = item
                    for future in as_completed(futures):
                        name, coordinate, key = futures[future]
                        attempts[name] += 1
                        try:
                            _returned_name, record = future.result()
                        except Exception as error:
                            errors[name] = str(error)
                            print(
                                f"[SO2] {name} attempt {attempts[name]} failed: "
                                f"{error}",
                                flush=True,
                            )
                            continue
                        _record_id, inserted = database.put(
                            key,
                            record,
                            metadata={
                                "name": name,
                                "diagnostics": record["diagnostics"],
                            },
                        )
                        source = "calculated" if inserted else "database-race"
                        store_record(name, coordinate, key, record, source)
                        pending = [item for item in pending if item[0] != name]
                        errors.pop(name, None)
                exhausted = [
                    item
                    for item in pending
                    if attempts[item[0]] >= args.max_attempts
                ]
                if exhausted:
                    exhausted_names = {item[0] for item in exhausted}
                    exhausted_errors.update(
                        {name: errors[name] for name in exhausted_names}
                    )
                    pending = [
                        item for item in pending if item[0] not in exhausted_names
                    ]
            if exhausted_errors:
                missing.extend(sorted(exhausted_errors))
                print(
                    "[SO2] unconverged points excluded: "
                    + "; ".join(
                        f"{name}: {error}"
                        for name, error in sorted(exhausted_errors.items())
                    ),
                    flush=True,
                )

        records = {
            name: records[name]
            for name, _coordinate in points
            if name in records
        }
        database.update_run(run_id, "sampled-partial" if missing else "sampled")
        figure = args.output / f"{run_id}.png"
        plot(records, figure)
        summary = {
            "run_id": run_id,
            "database": str(database_path),
            "records": len(records),
            "requested_records": len(points),
            "missing": missing,
            "new_records": database.writes,
            "database_hits": database.hits,
            "protocol": electronic_protocol,
            "points": {
                name: {
                    "coordinate": np.asarray(record["coordinate"]).tolist(),
                    "diagnostics": record["diagnostics"],
                }
                for name, record in records.items()
            },
            "figure": str(figure),
        }
        summary_path = args.output / f"{run_id}.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        database.update_run(run_id, "failed")
        raise
    finally:
        database.close()


if __name__ == "__main__":
    main()
