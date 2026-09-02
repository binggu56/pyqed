#!/usr/bin/env python3
"""Build graph-homed SO2 CAS(8,8)/SOMF records from saved v5 orbitals.

This two-layer migration stores the complete six-root CASCI candidate manifold
before choosing the three-state window.  A maximum-overlap spanning tree then
selects one plane-even and two plane-odd roots globally.  The raw overlaps are
never unitarized.  SCF and CASSCF orbital optimization are never repeated.

The SOMF construction follows Heß et al., Chem. Phys. Lett. 251, 365 (1996),
https://doi.org/10.1016/0009-2614(96)00119-4.  This remains an adaptation using
native PyQED CASCI transition densities, a one-center one-electron SOC term,
and a density averaged over the complete stored candidate manifold.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from examples.namd.augment_so2_cas88_somf_v6 import _source_entries, _settings
from examples.namd.generate_so2_cas88_somf import (
    MS_VALUES,
    TARGET_PLANE_PARITIES,
    _sector,
    default_output,
    geometry,
    molecule_at,
    plot,
    protocol,
    specification,
    validate_spin_selection,
)
from pyqed.ldr import ElectronicDatabase
from pyqed.ldr.so2 import (
    frame_parities,
    full_spin_overlap,
    select_root_sectors,
    sparse_overlap_graph,
)
from pyqed.qchem.hf.rhf import RHF
from pyqed.qchem.mcscf.casci import (
    CASCIFrame,
    make_tdm1_spin_orbital,
    overlap as casci_overlap,
)
from pyqed.qchem.mcscf.soc_si import (
    align_triplet_multiplet_phases,
)
from pyqed.qchem.soc import get_soc_somf_spin_orbital
from pyqed.units import au2wavenumber


CANDIDATE_SCHEMA = "pyqed-so2-cas88-somf-candidates-v1"
HOMED_SCHEMA = "pyqed-so2-cas88-somf-v7"
LINK_GATE = 0.1
BOUNDS = (2.55, 3.05, 0.25, np.deg2rad(100.0), np.deg2rad(140.0))


def candidate_protocol(settings):
    value = protocol(settings)
    value["schema"] = CANDIDATE_SCHEMA
    interaction = value["state_interaction"]
    interaction["root_selection"] = "deferred to graph root homing"
    interaction["somf_density"] = (
        "equal average over every stored singlet and Ms=0 triplet candidate root"
    )
    interaction["candidate_storage"] = (
        "complete CASCI frames and candidate-averaged SOMF operator"
    )
    return value


def homed_protocol(settings, *, neighbors, graph_id):
    value = candidate_protocol(settings)
    value["schema"] = HOMED_SCHEMA
    interaction = value["state_interaction"]
    interaction["root_selection"] = {
        "symmetry": "molecular-plane reflection",
        "target_parities": list(TARGET_PLANE_PARITIES),
        "method": "maximum-overlap tree dynamic programming",
        "raw_links_unitarized": False,
        "candidate_graph_neighbors": int(neighbors),
        "coordinate_bounds": list(map(float, BOUNDS)),
        "graph_id": str(graph_id),
    }
    interaction["candidate_source_schema"] = CANDIDATE_SCHEMA
    value["orbitals"]["source_schema"] = "stored SA-CASSCF orbitals"
    return value


def subset_frame(frame, roots):
    return CASCIFrame(
        mol=frame.mol,
        mo_coeff=frame.mo_coeff,
        ci=tuple(frame.ci[int(root)] for root in roots),
        binary=frame.binary,
        ncore=frame.ncore,
        ncas=frame.ncas,
    )


def build_candidate_record(source, settings):
    """Rebuild only CASCI/SOMF in saved CASSCF orbitals."""

    started = time.perf_counter()
    coordinate = np.asarray(source["coordinate"], dtype=float)
    mol = molecule_at(coordinate, settings.basis)
    orbitals = np.asarray(source["mo_coeff"])
    orbital_defect = float(
        np.linalg.norm(
            orbitals.conj().T @ np.asarray(mol.overlap) @ orbitals
            - np.eye(orbitals.shape[1])
        )
    )
    if orbital_defect > 1.0e-7:
        raise RuntimeError(f"saved-orbital defect {orbital_defect:.3e}")
    mean_field = RHF(mol)
    mean_field.mo_coeff = orbitals
    mean_field.e_nuc = float(mol.energy_nuc())
    mean_field.e_tot = float(np.asarray(source["rhf_energy"]))

    singlet = _sector(
        mean_field,
        orbitals,
        settings,
        ms2=0,
        multiplicity=1,
        nroots=settings.singlet_candidates,
    )
    triplets = {
        ms: _sector(
            mean_field,
            orbitals,
            settings,
            ms2=2 * ms,
            multiplicity=3,
            nroots=settings.triplet_candidates,
        )
        for ms in MS_VALUES
    }
    phase = align_triplet_multiplet_phases(triplets)
    singlet_parities, singlet_symmetry = frame_parities(singlet.frame(), mol)
    triplet_parities = {}
    triplet_symmetry = {}
    for ms, solver in triplets.items():
        triplet_parities[ms], triplet_symmetry[ms] = frame_parities(
            solver.frame(), mol
        )
    if any(
        not np.array_equal(triplet_parities[ms], triplet_parities[0])
        for ms in MS_VALUES
    ):
        raise RuntimeError("triplet symmetry order differs between Ms sectors")

    states = [
        (singlet, root) for root in range(settings.singlet_candidates)
    ] + [
        (triplets[ms], root)
        for root in range(settings.triplet_candidates)
        for ms in MS_VALUES
    ]
    density_states = [
        (singlet, root) for root in range(settings.singlet_candidates)
    ] + [
        (triplets[0], root) for root in range(settings.triplet_candidates)
    ]
    hso = get_soc_somf_spin_orbital(
        mean_field,
        representation="mo",
        mo_coeff=singlet.mo_cas,
        states=density_states,
        one_center=True,
        order="grouped",
    )
    spin_square = np.asarray(
        np.real_if_close(
            [solver.spin_square(root) for solver, root in states], tol=1000
        ),
        dtype=float,
    )
    expected = np.r_[
        np.zeros(settings.singlet_candidates),
        np.full(3 * settings.triplet_candidates, 2.0),
    ]
    spin_error = float(np.max(np.abs(spin_square - expected)))
    triplet_components = np.asarray(
        [
            [triplets[ms].e_tot[root] for ms in MS_VALUES]
            for root in range(settings.triplet_candidates)
        ]
    )
    triplet_degeneracy = float(np.max(np.ptp(triplet_components, axis=1)))
    if spin_error > settings.spin_tol:
        raise RuntimeError(f"spin-purity error {spin_error:.3e}")
    if triplet_degeneracy > 1.0e-8:
        raise RuntimeError(f"triplet Ms splitting {triplet_degeneracy:.3e} Eh")
    return {
        "coordinate": coordinate,
        "geometry": geometry(coordinate),
        "mo_coeff": orbitals,
        "active_orbitals": np.asarray(singlet.mo_cas),
        "singlet_frame": singlet.frame(),
        "triplet_frames": {ms: solver.frame() for ms, solver in triplets.items()},
        "singlet_energies": np.asarray(singlet.e_tot),
        "triplet_energies": np.asarray(triplets[0].e_tot),
        "singlet_parities": singlet_parities,
        "triplet_parities": triplet_parities[0],
        "spin_square": spin_square,
        "hso_active_spin_orbital": hso,
        "rhf_energy": np.asarray(source["rhf_energy"]),
        "orbital_history": source.get("orbital_history", []),
        "diagnostics": {
            "spin_error": spin_error,
            "triplet_ms_degeneracy_eh": triplet_degeneracy,
            "seconds": time.perf_counter() - started,
            "orbital_source": "reused-v5-casscf",
            "saved_orbital_orthonormality_defect": orbital_defect,
            "triplet_phase_off_diagonal": phase["off_diagonal"],
            "triplet_phase_amplitude_error": phase["amplitude_error"],
            "singlet_symmetry": singlet_symmetry,
            "triplet_symmetry": triplet_symmetry,
        },
    }


def candidate_choices(parities, target=TARGET_PLANE_PARITIES):
    """Enumerate energy-ordered root subsets matching the target sectors."""

    parities = np.asarray(parities, dtype=int)
    target = np.asarray(target, dtype=int)
    counts = {sign: int(np.count_nonzero(target == sign)) for sign in (-1, 1)}
    pools = {sign: np.flatnonzero(parities == sign).tolist() for sign in (-1, 1)}
    if any(len(pools[sign]) < counts[sign] for sign in (-1, 1)):
        raise ValueError("candidate roots do not span the target symmetry sectors")
    grouped = [
        list(combinations(pools[sign], counts[sign]))
        if counts[sign]
        else [()]
        for sign in (-1, 1)
    ]
    choices = []
    for negative, positive in product(*grouped):
        available = {-1: iter(negative), 1: iter(positive)}
        choices.append(tuple(next(available[int(sign)]) for sign in target))
    return tuple(sorted(choices))


def choice_scores(overlap, left_choices, right_choices):
    """Return raw minimum singular values between candidate subspaces."""

    overlap = np.asarray(overlap)
    scores = np.empty((len(left_choices), len(right_choices)))
    for left, left_roots in enumerate(left_choices):
        for right, right_roots in enumerate(right_choices):
            block = overlap[np.ix_(left_roots, right_roots)]
            scores[left, right] = np.min(np.linalg.svd(block, compute_uv=False))
    return scores


def maximum_overlap_tree(pairs, distances, singlet_scores, triplet_scores, npoints):
    """Choose a connected local tree with the best attainable bottlenecks."""

    costs = np.zeros((npoints, npoints))
    for edge, ((left, right), distance) in enumerate(zip(pairs, distances)):
        attainable = min(
            float(np.max(singlet_scores[edge])),
            float(np.max(triplet_scores[edge])),
        )
        cost = -np.log(max(attainable, 1.0e-14)) + 1.0e-8 * float(distance)
        costs[left, right] = costs[right, left] = max(cost, 1.0e-14)
    tree = minimum_spanning_tree(csr_matrix(costs)).tocoo()
    edge_lookup = {tuple(pair): index for index, pair in enumerate(map(tuple, pairs))}
    selected = []
    for left, right in zip(tree.row, tree.col):
        selected.append(edge_lookup[tuple(sorted((int(left), int(right))))])
    if len(selected) != npoints - 1:
        raise RuntimeError("candidate overlap graph is disconnected")
    return np.asarray(selected, dtype=int)


def home_choices_on_tree(choices, pairs, score_matrices, tree_edges, anchor, anchor_choice):
    """Globally maximize summed log-overlap on a fixed tree by dynamic programming."""

    npoints = len(choices)
    adjacency = [[] for _ in range(npoints)]
    for edge in tree_edges:
        left, right = map(int, pairs[edge])
        adjacency[left].append((right, int(edge)))
        adjacency[right].append((left, int(edge)))
    parent = np.full(npoints, -1, dtype=int)
    parent_edge = np.full(npoints, -1, dtype=int)
    order = [int(anchor)]
    for node in order:
        for neighbor, edge in adjacency[node]:
            if neighbor == parent[node]:
                continue
            if neighbor == anchor or parent[neighbor] >= 0:
                continue
            parent[neighbor] = node
            parent_edge[neighbor] = edge
            order.append(neighbor)
    if len(order) != npoints:
        raise RuntimeError("root-homing tree traversal did not reach every point")

    values = [np.zeros(len(item)) for item in choices]
    back = {}
    for node in reversed(order):
        for child, edge in adjacency[node]:
            if parent[child] != node:
                continue
            left, right = map(int, pairs[edge])
            matrix = score_matrices[edge]
            if node == right:
                matrix = matrix.T
            objective = np.log(np.maximum(matrix, 1.0e-14)) + values[child][None, :]
            best = np.argmax(objective, axis=1)
            values[node] += objective[np.arange(len(choices[node])), best]
            back[child] = best

    root_index = choices[anchor].index(tuple(anchor_choice))
    selected = np.full(npoints, -1, dtype=int)
    selected[anchor] = root_index
    for node in order:
        for child, _edge in adjacency[node]:
            if parent[child] == node:
                selected[child] = back[child][selected[node]]
    return tuple(choices[index][choice] for index, choice in enumerate(selected))


def materialize(candidate, singlet_roots, triplet_roots, homing):
    ns = len(candidate["singlet_energies"])
    permutation = list(map(int, singlet_roots))
    for root in triplet_roots:
        permutation.extend(ns + 3 * int(root) + offset for offset in range(3))
    permutation = np.asarray(permutation, dtype=int)
    states = [
        (candidate["singlet_frame"], int(root)) for root in singlet_roots
    ] + [
        (candidate["triplet_frames"][ms], int(root))
        for root in triplet_roots
        for ms in MS_VALUES
    ]
    energies = np.r_[
        np.asarray(candidate["singlet_energies"])[list(singlet_roots)],
        np.repeat(
            np.asarray(candidate["triplet_energies"])[list(triplet_roots)], 3
        ),
    ]
    h_scalar = np.diag(energies).astype(complex)
    h_soc = np.zeros_like(h_scalar)
    hso = np.asarray(candidate["hso_active_spin_orbital"])
    n_singlets = len(singlet_roots)
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
    h_total = h_scalar + h_soc
    eigenvalues, eigenvectors = np.linalg.eigh(h_total)
    labels = [f"S{root}" for root in range(len(singlet_roots))]
    labels.extend(
        f"T{root}(Ms={ms:+d})"
        for root in range(len(triplet_roots))
        for ms in MS_VALUES
    )
    diagnostics = copy.deepcopy(candidate["diagnostics"])
    diagnostics.update(
        {
            "hermiticity_defect": float(np.linalg.norm(h_total - h_total.conj().T)),
            "maximum_soc_cm-1": float(np.max(np.abs(h_soc)) * au2wavenumber),
            "root_selection": "maximum-overlap tree dynamic programming",
        }
    )
    record = {
        "coordinate": np.asarray(candidate["coordinate"]),
        "geometry": np.asarray(candidate["geometry"]),
        "labels": labels,
        "scalar_energies": np.real(np.diag(h_scalar)),
        "spin_square": np.asarray(candidate["spin_square"])[permutation],
        "h_scalar": h_scalar,
        "h_soc": h_soc,
        "h_total": h_total,
        "soc_eigenvalues": eigenvalues,
        "soc_eigenvectors": eigenvectors,
        "hso_active_spin_orbital": hso,
        "mo_coeff": np.asarray(candidate["mo_coeff"]),
        "active_orbitals": np.asarray(candidate["active_orbitals"]),
        "singlet_frame": subset_frame(candidate["singlet_frame"], singlet_roots),
        "triplet_frames": {
            ms: subset_frame(candidate["triplet_frames"][ms], triplet_roots)
            for ms in MS_VALUES
        },
        "candidate_roots": {
            "target_plane_parities": np.asarray(TARGET_PLANE_PARITIES),
            "singlet_candidate_parities": np.asarray(candidate["singlet_parities"]),
            "triplet_candidate_parities": np.asarray(candidate["triplet_parities"]),
            "singlet_selected_indices": np.asarray(singlet_roots),
            "triplet_selected_indices": np.asarray(triplet_roots),
            "singlet_energies": np.asarray(candidate["singlet_energies"]),
            "triplet_energies": np.asarray(candidate["triplet_energies"]),
            "homing": homing,
        },
        "rhf_energy": np.asarray(candidate["rhf_energy"]),
        "orbital_history": candidate["orbital_history"],
        "diagnostics": diagnostics,
    }
    return validate_spin_selection(record, len(singlet_roots))


def link_minima(records, pairs):
    return np.asarray(
        [
            np.min(
                np.linalg.svd(
                    full_spin_overlap(records[int(left)], records[int(right)]),
                    compute_uv=False,
                )
            )
            for left, right in pairs
        ]
    )


def overlap_record(candidate, singlet_roots, triplet_roots):
    """Return only the selected frames required by raw-link diagnostics."""

    return {
        "singlet_frame": subset_frame(candidate["singlet_frame"], singlet_roots),
        "triplet_frames": {
            ms: subset_frame(candidate["triplet_frames"][ms], triplet_roots)
            for ms in MS_VALUES
        },
    }


def plot_homing(names, baseline, homed, tree_mask, output):
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=True)
    x = np.arange(len(names))
    axes[0].plot(x, baseline["singlet"], "o--", color="#999999", label="S lowest-sector")
    axes[0].plot(x, homed["singlet"], "o-", color="#0072B2", label="S homed")
    axes[0].plot(x, baseline["triplet"], "s--", color="#CC79A7", label="T lowest-sector")
    axes[0].plot(x, homed["triplet"], "s-", color="#D55E00", label="T homed")
    axes[0].set(xlabel="geometry index", ylabel="selected candidate-root sum")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    before = np.sort(baseline["links"])
    after = np.sort(homed["links"])
    axes[1].plot(before, "o--", ms=3.5, color="#999999", label="before homing")
    axes[1].plot(after, "o-", ms=3.5, color="#0072B2", label="after homing")
    axes[1].axhline(LINK_GATE, color="0.4", ls=":")
    axes[1].set(xlabel="local raw link (sorted)", ylabel=r"$\sigma_{\min}$")
    axes[1].legend(frameon=False, fontsize=8)

    tree_values = np.sort(homed["links"][tree_mask])
    passed = tree_values >= LINK_GATE
    axes[2].plot(tree_values, color="0.5", lw=1.0)
    axes[2].scatter(np.flatnonzero(~passed), tree_values[~passed], color="#D55E00", label="below gate")
    axes[2].scatter(np.flatnonzero(passed), tree_values[passed], color="#0072B2", label="passed")
    axes[2].axhline(LINK_GATE, color="0.4", ls=":")
    axes[2].set(xlabel="homing-tree link (sorted)", ylabel=r"$\sigma_{\min}$")
    axes[2].legend(frameon=False, fontsize=8)
    for label, axis in zip("abc", axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.12, 1.03, label, transform=axis.transAxes, fontweight="bold")
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=default_output() / "electronic.sqlite")
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--extra-summary", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--singlet-candidates", type=int, default=6)
    parser.add_argument("--triplet-candidates", type=int, default=6)
    parser.add_argument("--spin-root-cushion", type=int, default=10)
    parser.add_argument("--spin-tol", type=float, default=1.0e-6)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    database = ElectronicDatabase(args.database)
    entries = _source_entries(database, source_summary=args.source_summary)
    known_names = {entry["metadata"]["name"] for entry in entries}
    catalog = {entry["id"]: entry for entry in database.entries()}
    for path in args.extra_summary:
        payload = json.loads(path.read_text())
        for name, point in payload["points"].items():
            if name in known_names:
                continue
            key = specification(point["coordinate"], payload["protocol"])
            identifier = database.identifier(key)
            if identifier not in catalog:
                raise KeyError(f"extra source record {name!r} is absent")
            entry = copy.deepcopy(catalog[identifier])
            entry["metadata"]["name"] = name
            entries.append(entry)
            known_names.add(name)
    settings = _settings(entries[0]["specification"]["protocol"], args)
    candidate_key_protocol = candidate_protocol(settings)
    jobs = []
    for index, entry in enumerate(entries):
        source = database.get(entry["specification"])
        name = entry["metadata"]["name"]
        key = specification(source["coordinate"], candidate_key_protocol)
        jobs.append(
            {
                "index": index,
                "name": name,
                "source_id": entry["id"],
                "source": source,
                "candidate_key": key,
                "candidate": database.get(key),
            }
        )

    missing = [job for job in jobs if job["candidate"] is None]
    if missing:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(missing)), mp_context=context
        ) as executor:
            futures = {
                executor.submit(build_candidate_record, job["source"], settings): job
                for job in missing
            }
            for future in as_completed(futures):
                job = futures[future]
                job["candidate"] = future.result()
                print(
                    f"[SO2 v7] candidates {job['name']} "
                    f"{job['candidate']['diagnostics']['seconds']:.1f} s",
                    flush=True,
                )
    for job in jobs:
        identifier, _inserted = database.put(
            job["candidate_key"],
            job["candidate"],
            metadata={
                "name": job["name"],
                "source_record_id": job["source_id"],
                "repeated_scf": False,
                "repeated_casscf": False,
            },
        )
        job["candidate_id"] = identifier

    graph_payload = [
        {
            "name": job["name"],
            "candidate_record_id": job["candidate_id"],
            "coordinate": np.asarray(job["candidate"]["coordinate"]).round(14).tolist(),
        }
        for job in jobs
    ]
    graph_id = hashlib.sha256(
        json.dumps(graph_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    final_protocol = homed_protocol(
        settings, neighbors=args.neighbors, graph_id=graph_id
    )

    candidates = [job["candidate"] for job in jobs]
    names = [job["name"] for job in jobs]
    coordinates = np.asarray([record["coordinate"] for record in candidates])
    pairs, distances = sparse_overlap_graph(coordinates, BOUNDS, neighbors=args.neighbors)
    singlet_choices = [candidate_choices(record["singlet_parities"]) for record in candidates]
    triplet_choices = [candidate_choices(record["triplet_parities"]) for record in candidates]
    singlet_overlaps = []
    triplet_overlaps = []
    singlet_scores = []
    triplet_scores = []
    for left, right in pairs:
        s_overlap = casci_overlap(candidates[left]["singlet_frame"], candidates[right]["singlet_frame"])
        t_overlap = casci_overlap(candidates[left]["triplet_frames"][0], candidates[right]["triplet_frames"][0])
        singlet_overlaps.append(s_overlap)
        triplet_overlaps.append(t_overlap)
        singlet_scores.append(choice_scores(s_overlap, singlet_choices[left], singlet_choices[right]))
        triplet_scores.append(choice_scores(t_overlap, triplet_choices[left], triplet_choices[right]))
    tree_edges = maximum_overlap_tree(
        pairs, distances, singlet_scores, triplet_scores, len(candidates)
    )
    anchor = names.index("center") if "center" in names else 0
    anchor_s = tuple(select_root_sectors(candidates[anchor]["singlet_parities"], TARGET_PLANE_PARITIES))
    anchor_t = tuple(select_root_sectors(candidates[anchor]["triplet_parities"], TARGET_PLANE_PARITIES))
    selected_s = home_choices_on_tree(
        singlet_choices, pairs, singlet_scores, tree_edges, anchor, anchor_s
    )
    selected_t = home_choices_on_tree(
        triplet_choices, pairs, triplet_scores, tree_edges, anchor, anchor_t
    )
    baseline_s = tuple(
        tuple(select_root_sectors(record["singlet_parities"], TARGET_PLANE_PARITIES))
        for record in candidates
    )
    baseline_t = tuple(
        tuple(select_root_sectors(record["triplet_parities"], TARGET_PLANE_PARITIES))
        for record in candidates
    )
    homing_info = {
        "method": "maximum-overlap tree dynamic programming",
        "neighbors": int(args.neighbors),
        "anchor": names[anchor],
        "raw_links_unitarized": False,
    }
    baseline_records = [
        overlap_record(record, sroots, troots)
        for record, sroots, troots in zip(candidates, baseline_s, baseline_t)
    ]
    homed_overlap_records = [
        overlap_record(record, sroots, troots)
        for record, sroots, troots in zip(candidates, selected_s, selected_t)
    ]
    baseline_links = link_minima(baseline_records, pairs)
    homed_links = link_minima(homed_overlap_records, pairs)
    tree_mask = np.zeros(len(pairs), dtype=bool)
    tree_mask[tree_edges] = True

    records = [None] * len(candidates)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(candidates)), mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                materialize, candidate, sroots, troots, homing_info
            ): index
            for index, (candidate, sroots, troots) in enumerate(
                zip(candidates, selected_s, selected_t)
            )
        }
        for future in as_completed(futures):
            index = futures[future]
            records[index] = future.result()
            print(f"[SO2 v7] selected interaction {names[index]}", flush=True)

    run_id = "so2-cas88-somf-v7-home-" + time.strftime("%Y%m%dT%H%M%S")
    database.start_run(
        run_id,
        metadata={
            "protocol": final_protocol,
            "candidate_schema": CANDIDATE_SCHEMA,
            "repeated_scf": False,
            "repeated_casscf": False,
        },
        status="homing",
    )
    identifiers = []
    for job, record in zip(jobs, records):
        key = specification(record["coordinate"], final_protocol)
        identifier, inserted = database.put(
            key,
            record,
            metadata={
                "name": job["name"],
                "candidate_record_id": job["candidate_id"],
                "diagnostics": record["diagnostics"],
            },
        )
        identifiers.append(identifier)
        database.note_run_record(
            run_id,
            identifier,
            {
                "index": [job["index"]],
                "name": job["name"],
                "coordinate": np.asarray(record["coordinate"]).tolist(),
                "candidate_record_id": job["candidate_id"],
            },
            "graph-root-homing" if inserted else "database",
        )
    overlap_protocol = {
        "method": "raw CASCI wavefunction overlap",
        "state_window": HOMED_SCHEMA,
        "unitarized": False,
    }
    for (left, right), value in zip(pairs, homed_links):
        database.put_overlap(
            identifiers[left],
            identifiers[right],
            overlap_protocol,
            full_spin_overlap(records[left], records[right]),
            metadata={"minimum_singular_value": float(value), "unitarized": False},
        )
    database.update_run(run_id, "complete")

    result_figure = args.output / f"{run_id}.png"
    plot(dict(zip(names, records)), result_figure)
    diagnostic_figure = args.output / f"{run_id}-homing.png"
    plot_homing(
        names,
        {
            "singlet": np.asarray([sum(item) for item in baseline_s]),
            "triplet": np.asarray([sum(item) for item in baseline_t]),
            "links": baseline_links,
        },
        {
            "singlet": np.asarray([sum(item) for item in selected_s]),
            "triplet": np.asarray([sum(item) for item in selected_t]),
            "links": homed_links,
        },
        tree_mask,
        diagnostic_figure,
    )
    tree_minimum = float(np.min(homed_links[tree_mask]))
    summary = {
        "run_id": run_id,
        "database": str(args.database),
        "protocol": final_protocol,
        "candidate_schema": CANDIDATE_SCHEMA,
        "records": len(records),
        "repeated_scf": False,
        "repeated_casscf": False,
        "raw_links_unitarized": False,
        "baseline_minimum_raw_link": float(np.min(baseline_links)),
        "homed_minimum_raw_link": float(np.min(homed_links)),
        "homed_tree_minimum_raw_link": tree_minimum,
        "raw_link_gate": LINK_GATE,
        "dynamics_ready": bool(tree_minimum >= LINK_GATE),
        "tree_edges": [
            {
                "left": names[int(pairs[edge, 0])],
                "right": names[int(pairs[edge, 1])],
                "minimum_singular_value": float(homed_links[edge]),
            }
            for edge in tree_edges
        ],
        "points": {
            name: {
                "coordinate": np.asarray(record["coordinate"]).tolist(),
                "candidate_record_id": job["candidate_id"],
                "singlet_roots": list(map(int, sroots)),
                "triplet_roots": list(map(int, troots)),
                "diagnostics": record["diagnostics"],
            }
            for name, record, job, sroots, troots in zip(
                names, records, jobs, selected_s, selected_t
            )
        },
        "result_figure": str(result_figure),
        "diagnostic_figure": str(diagnostic_figure),
    }
    summary_path = args.output / f"{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    database.close()


if __name__ == "__main__":
    main()
