#!/usr/bin/env python3
"""Build and qualify a symmetry-reduced 5D phenol SA(6)-CASSCF pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import uuid

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.phenol_sa_casscf_paths import DEFAULT_PHENOL_SA6_DATABASE
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from pyqed.units import au2ev
from pyqed.dvr import SineDVR
from pyqed.ldr import (
    AbInitioFit,
    ElectronicDatabase,
    PhenolCASSCFOverlap,
    PhenolReflectionSymmetry,
    PhenolSACASSCFProvider,
    SamplingSymmetryImage,
    phenol_sa6_protocol,
)
from pyqed.ldr.overlap import procrustes, synchronize_link_gauge, track_states_graph
from pyqed.ldr.database import canonical_json
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.namd.phenol import build_phenol_5d_keo_mpo, phenol_metric_evaluators


HARTREE_TO_EV = au2ev
PARITIES = np.asarray((1.0, -1.0, 1.0, -1.0, 1.0))
R_GRID = np.asarray((0.95, 1.15, 1.30, 1.55, 1.85, 2.10, 2.40, 2.70, 3.00))
PHI_GRID = np.asarray((-0.40, -0.20, 0.0, 0.20, 0.40))
THETA_GRID = np.deg2rad(np.asarray((104.0, 108.8, 114.0)))
Q16A_GRID = np.asarray((-0.50, -0.25, 0.0, 0.25, 0.50))
Q8A_GRID = np.asarray((-0.20, -0.10, 0.0, 0.10, 0.20))
GRIDS = (R_GRID, PHI_GRID, THETA_GRID, Q16A_GRID, Q8A_GRID)
ORIGINAL_GRIDS = tuple(np.array(grid, copy=True) for grid in GRIDS)
SCALES = np.asarray((0.25, 0.20, np.deg2rad(5.0), 0.25, 0.10))
ANCHOR = (0, 2, 1, 2, 2)


def configure_probability_expanded_grid():
    """Use the box required by the converged S1 quasibound-state marginals."""

    global R_GRID, PHI_GRID, THETA_GRID, Q16A_GRID, Q8A_GRID, GRIDS, ANCHOR
    R_GRID = np.asarray((0.95, 1.15, 1.30, 1.55, 1.85, 2.10, 2.40, 2.70, 3.00))
    PHI_GRID = np.linspace(-1.0, 1.0, 11)
    THETA_GRID = np.deg2rad(
        np.asarray((90.0, 94.0, 99.0, 104.0, 108.8, 114.0, 119.0, 124.0, 129.0, 134.0))
    )
    Q16A_GRID = np.linspace(-1.0, 1.0, 9)
    Q8A_GRID = np.linspace(-0.4, 0.4, 9)
    GRIDS = (R_GRID, PHI_GRID, THETA_GRID, Q16A_GRID, Q8A_GRID)
    ANCHOR = tuple(
        int(np.argmin(np.abs(grid - value)))
        for grid, value in zip(
            GRIDS, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
        )
    )


def configure_quasibound_expanded_grid():
    """Extend the two broad odd coordinates beyond the first range probe."""

    global R_GRID, PHI_GRID, THETA_GRID, Q16A_GRID, Q8A_GRID, GRIDS, ANCHOR
    R_GRID = np.asarray((0.95, 1.15, 1.30, 1.55, 1.85, 2.10, 2.40, 2.70, 3.00))
    old_phi = np.linspace(-1.0, 1.0, 11)
    PHI_GRID = np.concatenate(
        ((-1.8, -1.6, -1.4, -1.2), old_phi, (1.2, 1.4, 1.6, 1.8))
    )
    THETA_GRID = np.deg2rad(
        np.asarray((90.0, 94.0, 99.0, 104.0, 108.8, 114.0, 119.0, 124.0, 129.0, 134.0))
    )
    Q16A_GRID = np.asarray(
        (-1.8, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
         0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.8)
    )
    Q8A_GRID = np.linspace(-0.4, 0.4, 9)
    GRIDS = (R_GRID, PHI_GRID, THETA_GRID, Q16A_GRID, Q8A_GRID)
    ANCHOR = tuple(
        int(np.argmin(np.abs(grid - value)))
        for grid, value in zip(
            GRIDS, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
        )
    )


def configure_periodic_torsion_grid():
    """Cover the complete CCOH torsional period while retaining old nodes."""

    global R_GRID, PHI_GRID, THETA_GRID, Q16A_GRID, Q8A_GRID, GRIDS, ANCHOR
    configure_quasibound_expanded_grid()
    inner = PHI_GRID.copy()
    outer = np.asarray((2.0, 2.3, 2.6, 2.9, np.pi))
    PHI_GRID = np.concatenate((-outer[::-1], inner, outer))
    GRIDS = (R_GRID, PHI_GRID, THETA_GRID, Q16A_GRID, Q8A_GRID)
    ANCHOR = tuple(
        int(np.argmin(np.abs(grid - value)))
        for grid, value in zip(
            GRIDS, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
        )
    )


class PhenolGeometry:
    """Pickle-safe 5D chart callback for process-pool workers."""

    def __init__(self, modes, exact_geometries=None):
        self.modes = np.asarray(modes, dtype=float)
        self.exact_geometries = {
            tuple(map(float, coordinate)): np.asarray(geometry, dtype=float)
            for coordinate, geometry in (exact_geometries or {}).items()
        }

    def __call__(self, coordinate):
        coordinate = np.asarray(coordinate, dtype=float)
        exact = self.exact_geometries.get(tuple(map(float, coordinate)))
        if exact is None:
            closest = None
            closest_distance = np.inf
            for stored_coordinate, stored_geometry in self.exact_geometries.items():
                distance = float(
                    np.max(np.abs(np.asarray(stored_coordinate) - coordinate))
                )
                if distance < closest_distance:
                    closest = stored_geometry
                    closest_distance = distance
            if closest_distance <= 1.0e-12:
                exact = closest
        if exact is not None:
            return np.array(exact, copy=True)
        return PhenolReactiveChart(modes=self.modes).geometry(coordinate)


def diagnostic_protocol(sa_protocol, parent_record_id, nroots):
    """Describe a CASCI root window derived from one immutable SA(6) record."""

    return {
        "system": "phenol",
        "geometry_unit": "angstrom",
        "basis": sa_protocol["basis"],
        "method": "diagnostic-CASCI",
        "backend": "pyqed",
        "parent_record_id": str(parent_record_id),
        "parent_method": "SA(6)-CASSCF",
        "active_space": sa_protocol["active_space"],
        "roots": int(nroots),
        "spin_constraint": sa_protocol["spin_constraint"],
        "orbitals": "unchanged converged parent SA(6)-CASSCF orbitals",
        "energy_ordered": True,
    }


def _diagnostic_task(database_path, object_dir, protocol, record, nroots):
    database = ElectronicDatabase(database_path, object_dir=object_dir)
    provider = PhenolSACASSCFProvider(database, protocol, verbose=0)
    try:
        return provider.diagnostic_casci(record, nroots=nroots, workers=1)
    finally:
        provider.close()
        database.close()


def match_parent_sa_roots(parent, diagnostic):
    """Match parent SA roots inside a larger diagnostic window by CI overlap."""

    from scipy.optimize import linear_sum_assignment

    parent_ci = np.asarray(parent["ci"])
    diagnostic_ci = np.asarray(diagnostic["ci"])
    overlap = np.abs(
        parent_ci.reshape(len(parent_ci), -1).conj()
        @ diagnostic_ci.reshape(len(diagnostic_ci), -1).T
    )
    parent_ids, diagnostic_ids = linear_sum_assignment(-overlap)
    roots = np.empty(len(parent_ci), dtype=int)
    roots[parent_ids] = diagnostic_ids
    value = dict(diagnostic)
    value["sa_root_indices"] = roots
    value["sa_energy_agreement"] = np.asarray(
        np.max(
            np.abs(
                np.asarray(diagnostic["energies"])[roots]
                - np.asarray(parent["energies"])
            )
        )
    )
    return value


def diagnostic_records(
    fit,
    points,
    canonical,
    production_records,
    database,
    protocol,
    symmetry,
    *,
    nroots,
    workers,
    run_id,
):
    """Load or calculate restartable diagnostic CASCI records."""

    point_ids = {point: index for index, point in enumerate(points)}
    production = dict(zip(points, production_records))
    diagnostic_run = f"{run_id}-diagnostic-{int(nroots)}"
    database.start_run(
        diagnostic_run,
        status="running",
        metadata={
            "purpose": "root-window transport on fixed SA(6)-CASSCF orbitals",
            "roots": int(nroots),
            "workers": int(workers),
        },
    )
    owner = f"{diagnostic_run}:{uuid.uuid4().hex}"
    by_representative = {}
    ids = {}
    protocols = {}
    sources = {}
    pending = {}
    claimed = []
    for point in canonical:
        parent = production[point]
        parent_id = fit.frames.record_id(point)
        derived_protocol = diagnostic_protocol(protocol, parent_id, nroots)
        specification = {
            "geometry": np.asarray(parent["geometry"]),
            "protocol": derived_protocol,
        }
        value = database.get(specification)
        if value is not None:
            value = match_parent_sa_roots(parent, value)
            by_representative[point] = value
            ids[point] = database.identifier(specification)
            protocols[point] = derived_protocol
            sources[point] = "database"
            continue
        status = database.claim(specification, owner)
        if status == "complete":
            by_representative[point] = match_parent_sa_roots(
                parent, database.get(specification)
            )
            ids[point] = database.identifier(specification)
            protocols[point] = derived_protocol
            sources[point] = "database"
        elif status == "acquired":
            pending[point] = (parent, parent_id, specification, derived_protocol)
            claimed.append(specification)
        else:
            raise RuntimeError(f"diagnostic CASCI record for {point} is claimed elsewhere")

    built = 0
    try:
        if pending:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = {
                    executor.submit(
                        _diagnostic_task,
                        str(database.path),
                        str(database.object_dir),
                        protocol,
                        parent,
                        int(nroots),
                    ): point
                    for point, (parent, _parent_id, _spec, _derived) in pending.items()
                }
                for future in as_completed(futures):
                    point = futures[future]
                    parent, parent_id, specification, derived_protocol = pending[point]
                    diagnostic = match_parent_sa_roots(parent, future.result())
                    value = {
                        "geometry": np.asarray(parent["geometry"]),
                        "mo_coeff": np.asarray(parent["mo_coeff"]),
                        "ci": np.asarray(diagnostic["ci"]),
                        "energies": np.asarray(diagnostic["energies"]),
                        "spins": np.asarray(diagnostic["spins"]),
                        "sa_energy_agreement": np.asarray(
                            diagnostic["sa_energy_agreement"]
                        ),
                        "sa_root_indices": np.asarray(
                            diagnostic["sa_root_indices"]
                        ),
                        "parent_record_id": np.asarray(str(parent_id)),
                        "wall_seconds": np.asarray(diagnostic["wall_seconds"]),
                        "solver_backend": np.asarray(diagnostic["solver_backend"]),
                        "iterations": np.asarray(diagnostic["iterations"]),
                    }
                    record_id, _inserted = database.put(
                        specification,
                        value,
                        metadata={
                            "parent_record_id": parent_id,
                            "coordinates": fit.coordinates(point),
                            "run_id": diagnostic_run,
                        },
                    )
                    database.release_claim(specification, owner)
                    by_representative[point] = value
                    ids[point] = record_id
                    protocols[point] = derived_protocol
                    sources[point] = "calculated"
                    built += 1
                    print(
                        f"[5D diagnostic] {built}/{len(pending)}: {point}; "
                        f"{float(value['wall_seconds']):.2f} s",
                        flush=True,
                    )
    finally:
        for specification in claimed:
            database.release_claim(specification, owner)

    records = []
    record_ids = []
    for point in points:
        representative = fit.frames.representative(point)
        value = by_representative[representative]
        if point != representative:
            image = symmetry.resolve(fit.coordinates(point))
            value = symmetry.transform_record(
                value,
                image,
                representative_geometry=fit.representative_geometry(point),
                requested_geometry=fit.sample_geometry(point),
                protocol=protocols[representative],
            )
        records.append(value)
        record_ids.append(ids[representative])
        database.note_run_record(
            diagnostic_run,
            ids[representative],
            fit.sample(point),
            sources[representative] if point == representative else "sampling-symmetry",
        )
    database.update_run(diagnostic_run, "complete")
    return records, np.asarray(record_ids), {
        "built": built,
        "database_hits": len(canonical) - built,
        "run_id": diagnostic_run,
    }


def diagnostic_overlap_many(
    database,
    points,
    pairs,
    records,
    record_ids,
    symmetry,
    overlap,
    *,
    workers,
    coordinate_of=None,
):
    """Return persistent symmetry-view-aware overlaps for diagnostic records."""

    point_ids = {point: index for index, point in enumerate(points)}
    records = dict(zip(points, records))
    record_ids = dict(zip(points, record_ids))
    if coordinate_of is None:
        coordinate_of = lambda point: tuple(
            GRIDS[axis][value] for axis, value in enumerate(point)
        )
    requests = []
    groups = {}
    for number, (left, right) in enumerate(pairs):
        left_view = symmetry.view_key(symmetry.resolve(coordinate_of(left)))
        right_view = symmetry.view_key(symmetry.resolve(coordinate_of(right)))
        left_id, right_id = str(record_ids[left]), str(record_ids[right])
        left_token = canonical_json({"record": left_id, "view": left_view})
        right_token = canonical_json({"record": right_id, "view": right_view})
        reverse = right_token < left_token
        if reverse:
            stored_left, stored_right = right_id, left_id
            stored_views = [right_view, left_view]
        else:
            stored_left, stored_right = left_id, right_id
            stored_views = [left_view, right_view]
        overlap_protocol = {
            "base": overlap.protocol,
            "electronic_manifold": "diagnostic-CASCI",
            "sampling_symmetry_views": stored_views,
            "version": 1,
        }
        key = (stored_left, stored_right, canonical_json(overlap_protocol))
        request = {
            "number": number,
            "left": left,
            "right": right,
            "stored_left": stored_left,
            "stored_right": stored_right,
            "protocol": overlap_protocol,
            "reverse": reverse,
        }
        requests.append(request)
        groups.setdefault(key, []).append(request)

    nroots = len(np.asarray(next(iter(records.values()))["energies"]))
    result = np.empty((len(pairs), nroots, nroots), dtype=complex)
    missing = {}
    hits = 0
    for key, group in groups.items():
        request = group[0]
        stored = database.get_overlap(
            request["stored_left"], request["stored_right"], request["protocol"]
        )
        if stored is None:
            missing[key] = request
            continue
        hits += 1
        for item in group:
            result[item["number"]] = stored.conj().T if item["reverse"] else stored

    completed = 0
    if missing:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(
                    overlap, records[request["left"]], records[request["right"]]
                ): (key, request)
                for key, request in missing.items()
            }
            for future in as_completed(futures):
                key, request = futures[future]
                raw = np.asarray(future.result())
                stored = raw.conj().T if request["reverse"] else raw
                database.put_overlap(
                    request["stored_left"],
                    request["stored_right"],
                    request["protocol"],
                    stored,
                    metadata={"purpose": "5D diagnostic-root state transport"},
                )
                for item in groups[key]:
                    result[item["number"]] = (
                        stored.conj().T if item["reverse"] else stored
                    )
                completed += 1
                if completed % 25 == 0 or completed == len(missing):
                    print(
                        f"[5D diagnostic overlap] {completed}/{len(missing)}",
                        flush=True,
                    )
    return result, {"database_hits": hits, "built": len(missing)}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def load_chart(path):
    with np.load(path, allow_pickle=False) as archive:
        modes = np.asarray(archive["modes"], dtype=float)
        frequencies = np.asarray(archive["selected_frequencies_cm1"], dtype=float)
        labels = tuple(map(str, archive["labels"]))
    return PhenolReactiveChart(modes=modes), frequencies, labels


def recover_chart(database, run_id):
    """Recover the production normal modes and exact geometry map from a run."""

    run = database.run(run_id)
    exact = {}
    representatives = {}
    for item in run["records"]:
        sample = item["sample"]
        coordinate = tuple(map(float, sample["coordinates"]))
        exact[coordinate] = np.asarray(sample["geometry"], dtype=float)
        representative_coordinate = tuple(
            map(float, sample.get("representative_coordinates", coordinate))
        )
        representative_geometry = np.asarray(
            sample.get("representative_geometry", sample["geometry"]), dtype=float
        )
        exact[representative_coordinate] = representative_geometry
        representatives[representative_coordinate] = representative_geometry
    if not representatives:
        raise RuntimeError(f"electronic run {run_id!r} contains no geometries")

    coordinates = np.asarray(list(representatives), dtype=float)
    geometries = np.asarray(list(representatives.values()), dtype=float)
    base = np.asarray(
        [PhenolReactiveChart().geometry((*coordinate[:3], 0.0, 0.0))
         for coordinate in coordinates]
    )
    amplitudes = coordinates[:, 3:5]
    if np.linalg.matrix_rank(amplitudes) != 2:
        raise RuntimeError(f"electronic run {run_id!r} does not span both normal modes")
    modes = np.linalg.lstsq(
        amplitudes,
        (geometries - base).reshape(len(coordinates), -1),
        rcond=None,
    )[0].reshape(2, *geometries.shape[1:])
    chart = PhenolReactiveChart(modes=modes)
    reconstructed = base + np.einsum(
        "nk,kia->nia", amplitudes, chart.modes, optimize=True
    )
    maximum_defect = float(np.max(np.abs(reconstructed - geometries)))
    if maximum_defect > 1.0e-12:
        raise RuntimeError(
            f"recovered normal modes miss stored geometries by {maximum_defect:.3e} A"
        )
    metadata = run["metadata"]
    frequencies = np.asarray(metadata["mode_frequencies_cm1"], dtype=float)
    labels = tuple(map(str, metadata["mode_labels"]))
    return chart, frequencies, labels, exact, {
        "source_run_id": str(run_id),
        "stored_geometries": len(exact),
        "canonical_geometries": len(representatives),
        "maximum_reconstruction_defect_angstrom": maximum_defect,
    }


def chart_reflection_defect(chart, indices):
    reflection = np.diag((1.0, 1.0, -1.0))
    defect = 0.0
    for index in indices:
        coordinate = np.asarray(
            [grid[value] for grid, value in zip(GRIDS, index)]
        )
        reflected = coordinate * PARITIES
        defect = max(
            defect,
            float(
                np.max(
                    np.abs(
                        chart.geometry(coordinate) @ reflection.T
                        - chart.geometry(reflected)
                    )
                )
            ),
        )
    return defect


def _canonical(index):
    index = tuple(map(int, index))
    coordinates = np.asarray([grid[value] for grid, value in zip(GRIDS, index)])
    for axis in (1, 3):
        if abs(coordinates[axis]) > 1.0e-12:
            if coordinates[axis] < 0.0:
                reflected = list(index)
                reflected[1] = len(PHI_GRID) - 1 - reflected[1]
                reflected[3] = len(Q16A_GRID) - 1 - reflected[3]
                return tuple(reflected)
            break
    return index


def design_points(count, seed):
    """Return a deterministic 5D design containing all chemically useful crosses."""

    count = int(count)
    if count < len(R_GRID) * 11:
        raise ValueError(f"at least {len(R_GRID) * 11} representatives are required")
    points = []
    center = (2, 1, 2, 2)
    for radial in range(len(R_GRID)):
        # Existing three-dimensional radial/torsion/bend cross.
        for torsion in (2, 3, 4):
            points.append((radial, torsion, 1, 2, 2))
        for bend in (0, 1, 2):
            points.append((radial, 2, bend, 2, 2))
        # One-dimensional Q16a and Q8a cuts through every dissociation radius.
        for q16a in (3, 4):
            points.append((radial, 2, 1, q16a, 2))
        for q8a in (0, 1, 3, 4):
            points.append((radial, 2, 1, 2, q8a))
    points = list(dict.fromkeys(_canonical(point) for point in points))
    engine = qmc.Sobol(5, scramble=True, seed=int(seed))
    power = int(np.ceil(np.log2(max(4 * count, 2))))
    for unit in engine.random_base2(power):
        index = tuple(
            min(int(value * size), size - 1)
            for value, size in zip(unit, map(len, GRIDS))
        )
        index = _canonical(index)
        if index not in points:
            points.append(index)
        if len(points) >= count:
            break
    if len(points) < count:
        for index in np.ndindex(*(len(grid) for grid in GRIDS)):
            index = _canonical(index)
            if index not in points:
                points.append(index)
            if len(points) >= count:
                break
    if ANCHOR in points:
        points.remove(ANCHOR)
    return tuple((ANCHOR, *points))[:count]


def _grid_index(coordinate):
    index = tuple(
        int(np.argmin(np.abs(grid - value)))
        for grid, value in zip(GRIDS, coordinate)
    )
    reconstructed = np.asarray([grid[item] for grid, item in zip(GRIDS, index)])
    if not np.allclose(reconstructed, coordinate, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"coordinate {tuple(coordinate)!r} is absent from the active grid")
    return index


def probability_expanded_design(count, seed, *, base_count=128):
    """Retain the qualified design and add wavefunction-weighted outer-q points."""

    count = int(count)
    base_count = int(base_count)
    if base_count != 128:
        raise ValueError("the qualified phenol base design contains 128 representatives")
    if count < base_count + 48:
        raise ValueError("the probability-expanded design requires at least 176 representatives")

    active_grids = GRIDS
    try:
        globals()["R_GRID"], globals()["PHI_GRID"], globals()["THETA_GRID"], \
            globals()["Q16A_GRID"], globals()["Q8A_GRID"] = ORIGINAL_GRIDS
        globals()["GRIDS"] = ORIGINAL_GRIDS
        globals()["ANCHOR"] = (0, 2, 1, 2, 2)
        original = design_points(base_count, 61)
        original_coordinates = tuple(
            tuple(grid[item] for grid, item in zip(ORIGINAL_GRIDS, point))
            for point in original
        )
    finally:
        globals()["R_GRID"], globals()["PHI_GRID"], globals()["THETA_GRID"], \
            globals()["Q16A_GRID"], globals()["Q8A_GRID"] = active_grids
        globals()["GRIDS"] = active_grids
        globals()["ANCHOR"] = tuple(
            int(np.argmin(np.abs(grid - value)))
            for grid, value in zip(
                active_grids, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
            )
        )

    points = [_grid_index(coordinate) for coordinate in original_coordinates]
    old_bounds = np.asarray([(grid.min(), grid.max()) for grid in ORIGINAL_GRIDS])

    radial_indices = tuple(
        int(np.argmin(np.abs(R_GRID - value))) for value in (0.95, 1.15, 1.30)
    )
    centers = tuple(int(np.argmin(np.abs(grid - value))) for grid, value in zip(
        GRIDS, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
    ))
    new_axis_values = (
        (),
        (0.6, 0.8, 1.0),
        tuple(np.deg2rad((90.0, 94.0, 99.0, 119.0, 124.0, 129.0, 134.0))),
        (0.75, 1.0),
        (-0.4, -0.3, 0.3, 0.4),
    )
    for radial in radial_indices:
        for axis in range(1, 5):
            for value in new_axis_values[axis]:
                point = list(centers)
                point[0] = radial
                point[axis] = int(np.argmin(np.abs(GRIDS[axis] - value)))
                point = _canonical(point)
                if point not in points:
                    points.append(point)

    weights = []
    centers_values = (0.984, 0.0, np.deg2rad(110.72), 0.0, -0.012)
    widths = (0.15, 0.32, np.deg2rad(10.0), 0.36, 0.13)
    for grid, center, width in zip(GRIDS, centers_values, widths):
        weight = np.exp(-0.5 * ((grid - center) / width) ** 2)
        weights.append(np.cumsum(weight / np.sum(weight)))
    engine = qmc.Sobol(5, scramble=True, seed=int(seed))
    power = int(np.ceil(np.log2(max(16 * count, 2))))
    for unit in engine.random_base2(power):
        point = tuple(
            min(int(np.searchsorted(cumulative, value, side="right")), len(cumulative) - 1)
            for cumulative, value in zip(weights, unit)
        )
        point = _canonical(point)
        coordinate = np.asarray([grid[item] for grid, item in zip(GRIDS, point)])
        outside_old_box = np.any(
            (coordinate[1:] < old_bounds[1:, 0] - 1.0e-12)
            | (coordinate[1:] > old_bounds[1:, 1] + 1.0e-12)
        )
        if outside_old_box and point not in points:
            points.append(point)
        if len(points) >= count:
            break
    if len(points) < count:
        raise RuntimeError(
            f"only {len(points)} distinct probability-expanded representatives were generated"
        )
    if ANCHOR in points:
        points.remove(ANCHOR)
    return tuple((ANCHOR, *points))[:count]


def quasibound_expanded_design(count, seed, *, base_count=224):
    """Retain the first expanded design and add outer odd-coordinate support."""

    count = int(count)
    base_count = int(base_count)
    if base_count != 224:
        raise ValueError("the first probability-expanded design contains 224 representatives")
    if count < base_count + 48:
        raise ValueError("the quasibound-expanded design requires at least 272 representatives")

    active_grids = GRIDS
    try:
        configure_probability_expanded_grid()
        base = probability_expanded_design(base_count, 61)
        base_coordinates = tuple(
            tuple(grid[item] for grid, item in zip(GRIDS, point))
            for point in base
        )
    finally:
        globals()["R_GRID"], globals()["PHI_GRID"], globals()["THETA_GRID"], \
            globals()["Q16A_GRID"], globals()["Q8A_GRID"] = active_grids
        globals()["GRIDS"] = active_grids
        globals()["ANCHOR"] = tuple(
            int(np.argmin(np.abs(grid - value)))
            for grid, value in zip(
                active_grids, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
            )
        )

    points = [_grid_index(coordinate) for coordinate in base_coordinates]
    previous_bounds = np.asarray(
        ((0.95, 3.0), (-1.0, 1.0), (np.deg2rad(90.0), np.deg2rad(134.0)),
         (-1.0, 1.0), (-0.4, 0.4))
    )
    radial_indices = tuple(
        int(np.argmin(np.abs(R_GRID - value))) for value in (0.95, 1.15, 1.30)
    )
    centers = tuple(
        int(np.argmin(np.abs(grid - value)))
        for grid, value in zip(
            GRIDS, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
        )
    )
    for radial in radial_indices:
        for axis, values in ((1, (1.2, 1.4, 1.6, 1.8)),
                             (3, (1.25, 1.5, 1.8))):
            for value in values:
                point = list(centers)
                point[0] = radial
                point[axis] = int(np.argmin(np.abs(GRIDS[axis] - value)))
                point = _canonical(point)
                if point not in points:
                    points.append(point)

    weights = []
    center_values = (0.984, -0.10, np.deg2rad(109.1), 0.10, -0.012)
    widths = (0.15, 0.68, np.deg2rad(10.0), 0.63, 0.13)
    for grid, center, width in zip(GRIDS, center_values, widths):
        weight = np.exp(-0.5 * ((grid - center) / width) ** 2)
        weights.append(np.cumsum(weight / np.sum(weight)))
    engine = qmc.Sobol(5, scramble=True, seed=int(seed))
    power = int(np.ceil(np.log2(max(16 * count, 2))))
    for unit in engine.random_base2(power):
        point = tuple(
            min(int(np.searchsorted(cumulative, value, side="right")), len(cumulative) - 1)
            for cumulative, value in zip(weights, unit)
        )
        point = _canonical(point)
        coordinate = np.asarray([grid[item] for grid, item in zip(GRIDS, point)])
        outside_previous_box = np.any(
            (coordinate[1:] < previous_bounds[1:, 0] - 1.0e-12)
            | (coordinate[1:] > previous_bounds[1:, 1] + 1.0e-12)
        )
        if outside_previous_box and point not in points:
            points.append(point)
        if len(points) >= count:
            break
    if len(points) < count:
        raise RuntimeError(
            f"only {len(points)} distinct quasibound-expanded representatives were generated"
        )
    if ANCHOR in points:
        points.remove(ANCHOR)
    return tuple((ANCHOR, *points))[:count]


def periodic_torsion_design(count, seed, *, base_count=320):
    """Retain the 320-point design and add support from 1.8 radians to pi."""

    count = int(count)
    if int(base_count) != 320:
        raise ValueError("the quasibound-expanded design contains 320 representatives")
    if count < int(base_count) + 64:
        raise ValueError("the periodic-torsion design requires at least 384 representatives")

    active_grids = GRIDS
    try:
        configure_quasibound_expanded_grid()
        base = quasibound_expanded_design(base_count, 61)
        base_coordinates = tuple(
            tuple(grid[item] for grid, item in zip(GRIDS, point))
            for point in base
        )
    finally:
        globals()["R_GRID"], globals()["PHI_GRID"], globals()["THETA_GRID"], \
            globals()["Q16A_GRID"], globals()["Q8A_GRID"] = active_grids
        globals()["GRIDS"] = active_grids
        globals()["ANCHOR"] = tuple(
            int(np.argmin(np.abs(grid - value)))
            for grid, value in zip(
                active_grids, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
            )
        )

    points = [_grid_index(coordinate) for coordinate in base_coordinates]
    centers = tuple(
        int(np.argmin(np.abs(grid - value)))
        for grid, value in zip(
            GRIDS, (0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0)
        )
    )
    radial_indices = tuple(
        int(np.argmin(np.abs(R_GRID - value))) for value in (0.95, 1.15, 1.30)
    )
    outer_phi = (2.0, 2.3, 2.6, 2.9, np.pi)
    for radial in radial_indices:
        for phi in outer_phi:
            for q16a in (-1.0, 0.0, 1.0):
                point = list(centers)
                point[0] = radial
                point[1] = int(np.argmin(np.abs(PHI_GRID - phi)))
                point[3] = int(np.argmin(np.abs(Q16A_GRID - q16a)))
                point = _canonical(point)
                if point not in points:
                    points.append(point)

    weights = []
    centers_values = (0.984, 0.0, np.deg2rad(109.1), 0.0, -0.012)
    widths = (0.17, 1.0, np.deg2rad(11.0), 0.72, 0.14)
    for grid, center, width in zip(GRIDS, centers_values, widths):
        weight = np.exp(-0.5 * ((grid - center) / width) ** 2)
        if grid is PHI_GRID:
            weight = np.where(np.abs(grid) > 1.8 + 1.0e-12, 1.0, 0.0)
        weights.append(np.cumsum(weight / np.sum(weight)))
    engine = qmc.Sobol(5, scramble=True, seed=int(seed))
    power = int(np.ceil(np.log2(max(24 * count, 2))))
    for unit in engine.random_base2(power):
        point = tuple(
            min(int(np.searchsorted(cumulative, value, side="right")), len(cumulative) - 1)
            for cumulative, value in zip(weights, unit)
        )
        point = _canonical(point)
        coordinate = np.asarray([grid[item] for grid, item in zip(GRIDS, point)])
        if abs(coordinate[1]) > 1.8 + 1.0e-12 and point not in points:
            points.append(point)
        if len(points) >= count:
            break
    if len(points) < count:
        raise RuntimeError(
            f"only {len(points)} distinct periodic-torsion representatives were generated"
        )
    if ANCHOR in points:
        points.remove(ANCHOR)
    return tuple((ANCHOR, *points))[:count]


def overlap_graph(points, coordinates, neighbors, *, periods=None):
    coordinates = np.asarray(coordinates, dtype=float)
    delta = np.abs(coordinates[:, None, :] - coordinates[None, :, :])
    for axis, period in ({} if periods is None else periods).items():
        axis = int(axis)
        period = float(period)
        delta[..., axis] = np.minimum(
            delta[..., axis], period - delta[..., axis]
        )
    distance = np.linalg.norm(delta / SCALES, axis=-1)
    tree = minimum_spanning_tree(distance).tocoo()
    edges = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(tree.row, tree.col)
    }
    tree_edges = set(edges)
    for left in range(len(points)):
        order = [
            int(right)
            for right in np.argsort(distance[left])
            if int(right) != left
        ]
        for right in order[: int(neighbors)]:
            edges.add(tuple(sorted((left, int(right)))))
    edges = tuple(sorted(edges))
    pairs = tuple((points[left], points[right]) for left, right in edges)
    tree_mask = np.asarray([edge in tree_edges for edge in edges], dtype=bool)
    lengths = np.asarray([distance[left, right] for left, right in edges])
    return pairs, np.asarray(edges, dtype=int), tree_mask, lengths


def spanning_tree_mask(size, edges, order=None):
    parent = np.arange(int(size))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    mask = np.zeros(len(edges), dtype=bool)
    order = np.arange(len(edges)) if order is None else np.asarray(order, dtype=int)
    for edge in order:
        left, right = map(int, edges[edge])
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        mask[edge] = True
    if len({find(index) for index in range(int(size))}) != 1:
        raise RuntimeError("qualified electronic-overlap graph is disconnected")
    return mask


def connected_link_mask(size, edges, scores, threshold):
    """Keep all qualified links plus the strongest spanning-tree bridges."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(edges),):
        raise ValueError("link scores and edges have incompatible shapes")
    threshold_mask = scores >= float(threshold)
    tree_mask = spanning_tree_mask(size, edges, np.argsort(-scores))
    return threshold_mask | tree_mask, threshold_mask, tree_mask


def symmetry_adapt_gauges(points, roots, gauges, reflection):
    """Impose one exact reflection representation on a synchronized gauge."""

    points = tuple(map(tuple, points))
    point_ids = {point: index for index, point in enumerate(points)}
    roots = np.array(roots, copy=True)
    gauges = np.array(gauges, copy=True)
    reflection = np.asarray(reflection, dtype=complex)
    signs = np.sign(np.real(np.diag(reflection)))
    if np.any(signs == 0.0):
        raise RuntimeError("electronic reflection has an unresolved parity")

    def reflected(point):
        image = list(point)
        image[1] = len(PHI_GRID) - 1 - image[1]
        image[3] = len(Q16A_GRID) - 1 - image[3]
        return tuple(image)

    representatives = {}
    for point in points:
        representative = _canonical(point)
        if representative not in point_ids:
            raise RuntimeError("symmetry representative is missing from the sampled graph")
        representatives[point] = representative

    for point in points:
        if point != representatives[point] or reflected(point) != point:
            continue
        source = gauges[point_ids[point]]
        projected = np.zeros_like(source)
        for parity in (-1.0, 1.0):
            block = np.flatnonzero(signs == parity)
            if len(block):
                projected[np.ix_(block, block)] = procrustes(
                    source[np.ix_(block, block)]
                )[0]
        gauges[point_ids[point]] = projected

    for point in points:
        representative = representatives[point]
        if point == representative:
            continue
        left = point_ids[representative]
        right = point_ids[point]
        roots[right] = roots[left]
        gauges[right] = gauges[left] @ reflection
    return roots, gauges


def holdout_masks(size, edges, tree_mask, seed):
    rng = np.random.default_rng(int(seed) + 917)
    energy = np.zeros(int(size), dtype=bool)
    candidates = np.arange(1, int(size))
    energy[rng.choice(candidates, size=max(1, round(0.15 * len(candidates))), replace=False)] = True
    link = np.zeros(len(edges), dtype=bool)
    candidates = np.flatnonzero(~tree_mask)
    if len(candidates):
        link[rng.choice(candidates, size=max(1, round(0.15 * len(candidates))), replace=False)] = True
    return energy, link


def bridge_root_constraints(points):
    """Pin the two sides of the resolved 1.80--2.10 Angstrom root bridge."""

    fixed = {}
    center = ANCHOR[1:]
    for point in points:
        radial, torsion, bend, q16a, q8a = point
        if (torsion, bend, q16a, q8a) != center:
            continue
        radius = R_GRID[radial]
        if np.isclose(radius, 1.85):
            fixed[point] = (0, 5, 1)
        elif radius >= 2.10 - 1.0e-12:
            fixed[point] = (1, 6, 0)
    return fixed


def small_keo(chart, output, seed):
    bounds = np.asarray(
        (
            (0.90, 3.10),
            (-0.45, 0.45),
            (np.deg2rad(102.0), np.deg2rad(116.0)),
            (-0.55, 0.55),
            (-0.22, 0.22),
        )
    )
    atomic = np.asarray([chart.coordinate_to_atomic(bound) for bound in bounds.T]).T
    dvrs = tuple(SineDVR(lower, upper, 3) for lower, upper in atomic)
    operator, info = build_phenol_5d_keo_mpo(
        dvrs,
        chart,
        cross_max_rank=8,
        cross_sweeps=5,
        cross_rtol=2.0e-7,
        cross_validation=64,
        mpo_max_rank=96,
        seed=seed,
        return_info=True,
    )
    from pyqed.mps.mps import _mpo_to_dense_operator

    dense = _mpo_to_dense_operator(operator)
    hermiticity = float(np.linalg.norm(dense - dense.conj().T))
    core_path = output / "phenol_5d_podolsky_keo_mpo.npz"
    np.savez_compressed(
        core_path,
        **{f"core_{site}": np.asarray(core) for site, core in enumerate(operator.factors)},
        coordinate_bounds_atomic=atomic,
        dimensions=np.asarray([dvr.npts for dvr in dvrs]),
    )
    return {
        "artifact": str(core_path),
        "grid_shape": [dvr.npts for dvr in dvrs],
        "mpo_bond_dimensions": operator.bond_orders(),
        "hermiticity_defect": hermiticity,
        "minimum_eigenvalue": float(np.linalg.eigvalsh(dense)[0]),
        "cross": info,
    }


def plot_results(output, coordinates, energies, p_hamiltonian, records, singular, metric_min):
    reference = float(np.min(np.linalg.eigvalsh(p_hamiltonian[0])))
    figure, panels = plt.subplots(2, 2, figsize=(10.4, 7.5), constrained_layout=True)
    planar = (
        np.isclose(coordinates[:, 1], 0.0)
        & np.isclose(coordinates[:, 2], THETA_GRID[1])
        & np.isclose(coordinates[:, 3], 0.0)
        & np.isclose(coordinates[:, 4], 0.0)
    )
    order = np.argsort(coordinates[planar, 0])
    planar_r = coordinates[planar, 0][order]
    planar_e = np.linalg.eigvalsh(p_hamiltonian[planar])[order]
    for state, color in enumerate(("#0072B2", "#D55E00", "#009E73")):
        panels[0, 0].plot(
            planar_r,
            (planar_e[:, state] - reference) * HARTREE_TO_EV,
            "o-",
            color=color,
            label=f"P{state}",
        )
    panels[0, 0].set(
        xlabel=r"$R_{OH}$ ($\AA$)", ylabel="relative energy (eV)",
        title=r"5D data: planar $Q_{16a}=Q_{8a}=0$ cut",
    )
    panels[0, 0].legend(frameon=False)

    scatter = panels[0, 1].scatter(
        coordinates[:, 0], coordinates[:, 3], c=coordinates[:, 4],
        s=19, cmap="coolwarm", alpha=0.75,
    )
    panels[0, 1].set(
        xlabel=r"$R_{OH}$ ($\AA$)", ylabel=r"$Q_{16a}$ ($\AA\sqrt{amu}$)",
        title="Effective 5D sample projection",
    )
    figure.colorbar(scatter, ax=panels[0, 1], label=r"$Q_{8a}$ ($\AA\sqrt{amu}$)")

    macro = np.asarray([len(np.asarray(record.get("macro_history", ()))) for record in records])
    panels[1, 0].hist(macro, bins=np.arange(macro.min(), macro.max() + 2) - 0.5, color="#6A3D9A")
    panels[1, 0].set(xlabel="macroiterations", ylabel="geometries", title="Orbital convergence cost")
    panels[1, 1].semilogy(np.sort(singular[:, -1]), color="#E66101", label="tracked 3-state link")
    panels[1, 1].semilogy(np.sort(metric_min), color="#5E3C99", label=r"minimum eigenvalue of $G$")
    panels[1, 1].set(xlabel="sorted sample/edge", ylabel="value", title="Overlap and metric conditioning")
    panels[1, 1].legend(frameon=False, fontsize=8)
    for label, panel in zip("abcd", panels.flat):
        panel.text(0.02, 0.97, label, transform=panel.transAxes, va="top", fontweight="bold")
        panel.grid(alpha=0.18)
    png = output / "phenol_sa6_5d_pilot_diagnostics.png"
    pdf = output / "phenol_sa6_5d_pilot_diagnostics.pdf"
    figure.savefig(png, dpi=320)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path,
        default=DEFAULT_PHENOL_SA6_DATABASE,
    )
    parser.add_argument(
        "--modes", type=Path,
        help="normal-mode archive; recover it from --source-run-id when omitted",
    )
    parser.add_argument(
        "--source-run-id",
        default="phenol-sa6-5d-pilot-v1-s61-n128",
        help="completed electronic run used to recover exact stored geometries",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa6_5d_pilot_20260822"),
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--sampling-profile",
        choices=(
            "qualified",
            "probability-expanded",
            "quasibound-expanded",
            "periodic-torsion",
        ),
        default="qualified",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--minimum-link", type=float, default=0.10)
    parser.add_argument("--diagnostic-roots", type=int, default=10)
    parser.add_argument("--diagnostic-workers", type=int, default=6)
    parser.add_argument("--overlap-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.sampling_profile == "probability-expanded":
        configure_probability_expanded_grid()
    elif args.sampling_profile == "quasibound-expanded":
        configure_quasibound_expanded_grid()
    elif args.sampling_profile == "periodic-torsion":
        configure_periodic_torsion_grid()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    database = ElectronicDatabase(args.database)
    if args.modes is None:
        chart, frequencies, labels, exact_geometries, chart_recovery = recover_chart(
            database, args.source_run_id
        )
        recovered_modes = args.output / "phenol_hessian_modes_recovered.npz"
        np.savez_compressed(
            recovered_modes,
            modes=chart.modes,
            selected_frequencies_cm1=frequencies,
            labels=np.asarray(labels),
            source_run_id=np.asarray(args.source_run_id),
            maximum_reconstruction_defect_angstrom=np.asarray(
                chart_recovery["maximum_reconstruction_defect_angstrom"]
            ),
        )
        chart_recovery["artifact"] = str(recovered_modes)
    else:
        chart, frequencies, labels = load_chart(args.modes)
        exact_geometries = None
        chart_recovery = None
    keo = small_keo(chart, args.output, args.seed)
    protocol = phenol_sa6_protocol()
    overlap = PhenolCASSCFOverlap()
    provider = PhenolSACASSCFProvider(
        database,
        protocol,
        coordinate_scale=SCALES,
        verbose=0 if args.quiet else 1,
    )
    if args.sampling_profile == "probability-expanded":
        canonical = probability_expanded_design(args.samples, args.seed)
    elif args.sampling_profile == "quasibound-expanded":
        canonical = quasibound_expanded_design(args.samples, args.seed)
    elif args.sampling_profile == "periodic-torsion":
        canonical = periodic_torsion_design(args.samples, args.seed)
    else:
        canonical = design_points(args.samples, args.seed)
    input_mode_reflection_defect = chart_reflection_defect(chart, canonical)
    symmetry = PhenolReflectionSymmetry(
        torsion_axis=1,
        odd_axes=(1, 3),
        tolerance=max(1.0e-10, 1.05 * input_mode_reflection_defect),
    )

    def progress(index, stats):
        coordinate = tuple(GRIDS[axis][value] for axis, value in enumerate(index))
        print(
            f"[5D] built {stats['built']}/{len(canonical)} canonical: "
            f"R={coordinate[0]:.2f}, phi={coordinate[1]:+.2f}, "
            f"theta={np.rad2deg(coordinate[2]):.1f}, "
            f"Q16a={coordinate[3]:+.2f}, Q8a={coordinate[4]:+.2f}",
            flush=True,
        )

    if args.sampling_profile == "qualified":
        run_id = f"phenol-sa6-5d-pilot-v1-s{args.seed}-n{args.samples}"
    elif args.sampling_profile == "probability-expanded":
        run_id = f"phenol-sa6-5d-probability-expanded-v1-s{args.seed}-n{args.samples}"
    elif args.sampling_profile == "quasibound-expanded":
        run_id = f"phenol-sa6-5d-quasibound-expanded-v1-s{args.seed}-n{args.samples}"
    else:
        run_id = f"phenol-sa6-5d-periodic-torsion-v1-s{args.seed}-n{args.samples}"
    with AbInitioFit(
        GRIDS,
        6,
        electronic=provider,
        geometry=PhenolGeometry(chart.modes, exact_geometries),
        symmetry=symmetry,
        database=database,
        protocol=protocol,
        run_id=run_id,
        run_metadata={
            "purpose": "symmetry-reduced five-dimensional pilot",
            "coordinates": chart.names,
            "mode_labels": labels,
            "mode_frequencies_cm1": frequencies,
            "chart_recovery": chart_recovery,
            "input_mode_maximum_cartesian_reflection_defect_angstrom": input_mode_reflection_defect,
            "canonical_samples": args.samples,
            "workers": args.workers,
            "sampling_profile": args.sampling_profile,
        },
        frame=lambda record: record,
        energies=lambda record: record["energies"],
        overlap=overlap,
        overlap_protocol=overlap.protocol,
        anchor=ANCHOR,
        workers=args.workers,
        progress=progress,
        energy_shift=None,
    ) as fit:
        points = fit.expand_points(canonical)
        production_records = fit.frames.get_many(points)
        records, diagnostic_record_ids, diagnostic_stats = diagnostic_records(
            fit,
            points,
            canonical,
            production_records,
            database,
            protocol,
            symmetry,
            nroots=args.diagnostic_roots,
            workers=args.diagnostic_workers,
            run_id=run_id,
        )
        coordinates = np.asarray([fit.coordinates(point) for point in points])
        candidate_pairs, candidate_edge_ids, _candidate_tree, candidate_lengths = overlap_graph(
            points,
            coordinates,
            args.neighbors,
            periods=({1: 2.0 * np.pi} if args.sampling_profile == "periodic-torsion" else None),
        )
        candidate_raw, diagnostic_overlap_stats = diagnostic_overlap_many(
            database,
            points,
            candidate_pairs,
            records,
            diagnostic_record_ids,
            symmetry,
            overlap,
            workers=args.overlap_workers,
        )
        fixed_roots = bridge_root_constraints(points)
        _candidate_roots, candidate_selected = track_states_graph(
            points,
            candidate_pairs,
            candidate_raw,
            anchor=ANCHOR,
            states=(0, 1, 2),
            fixed=fixed_roots,
        )
        candidate_singular = np.linalg.svd(candidate_selected, compute_uv=False)
        retained, threshold_qualified, transport_tree = connected_link_mask(
            len(points),
            candidate_edge_ids,
            candidate_singular[:, -1],
            args.minimum_link,
        )
        qualified = retained
        pairs = tuple(pair for pair, keep in zip(candidate_pairs, qualified) if keep)
        edge_ids = candidate_edge_ids[qualified]
        raw_overlaps = candidate_raw[qualified]
        link_lengths = candidate_lengths[qualified]
        roots, selected = track_states_graph(
            points,
            pairs,
            raw_overlaps,
            anchor=ANCHOR,
            states=(0, 1, 2),
            fixed=fixed_roots,
        )
        singular = np.linalg.svd(selected, compute_uv=False)
        tree_mask = spanning_tree_mask(
            len(points), edge_ids, np.argsort(-singular[:, -1])
        )
        gauges, _unconstrained_links = synchronize_link_gauge(
            points,
            pairs,
            selected,
            anchor=ANCHOR,
            weights=np.maximum(singular[:, -1], 1.0e-4),
        )
        energies = np.asarray([record["energies"] for record in records])
        anchor_id = points.index(ANCHOR)
        anchor_record = records[anchor_id]
        reflected_anchor = symmetry.transform_record(
            anchor_record,
            SamplingSymmetryImage(fit.coordinates(ANCHOR), symmetry.operation),
            representative_geometry=anchor_record["geometry"],
            requested_geometry=anchor_record["geometry"],
            protocol=protocol,
        )
        parity_full = overlap(anchor_record, reflected_anchor)
        raw_reflection = procrustes(
            parity_full[np.ix_((0, 1, 2), (0, 1, 2))]
        )[0]
        reflection = np.diag(
            np.sign(np.real(np.diag(raw_reflection)))
        ).astype(complex)
        roots, gauges = symmetry_adapt_gauges(
            points, roots, gauges, reflection
        )
        point_ids = {point: index for index, point in enumerate(points)}
        selected = np.asarray(
            [
                block[
                    np.ix_(roots[point_ids[left]], roots[point_ids[right]])
                ]
                for (left, right), block in zip(pairs, raw_overlaps)
            ]
        )
        singular = np.linalg.svd(selected, compute_uv=False)
        p_links = np.asarray(
            [
                gauges[point_ids[left]].conj().T
                @ block
                @ gauges[point_ids[right]]
                for (left, right), block in zip(pairs, selected)
            ]
        )
        p_hamiltonian = np.asarray(
            [
                gauge.conj().T @ np.diag(energy[root]) @ gauge
                for gauge, energy, root in zip(gauges, energies, roots)
            ]
        )
        p_hamiltonian = 0.5 * (
            p_hamiltonian + p_hamiltonian.conj().swapaxes(-1, -2)
        )
        reflected_ids = np.asarray(
            [
                point_ids[
                    (
                        point[0],
                        len(PHI_GRID) - 1 - point[1],
                        point[2],
                        len(Q16A_GRID) - 1 - point[3],
                        point[4],
                    )
                ]
                for point in points
            ],
            dtype=int,
        )
        reflected_hamiltonian = np.einsum(
            "ab,nbc,cd->nad",
            reflection.conj().T,
            p_hamiltonian,
            reflection,
            optimize=True,
        )
        reflection_covariance_defect = float(
            np.max(
                np.linalg.norm(
                    p_hamiltonian[reflected_ids] - reflected_hamiltonian,
                    axis=(1, 2),
                )
            )
        )
        reflection_spectral_defect = float(
            np.max(
                np.abs(
                    np.linalg.eigvalsh(p_hamiltonian[reflected_ids])
                    - np.linalg.eigvalsh(p_hamiltonian)
                )
            )
        )
        metric_evaluate, metric_batch = phenol_metric_evaluators(chart)
        del metric_evaluate
        metrics, pseudopotential = metric_batch(
            np.asarray([chart.coordinate_to_atomic(point) for point in coordinates])
        )
        metric_eigenvalues = np.linalg.eigvalsh(metrics)
        energy_holdout, link_holdout = holdout_masks(
            len(points), edge_ids, tree_mask, args.seed
        )
        pair_axes = np.argmax(
            np.abs((coordinates[edge_ids[:, 1]] - coordinates[edge_ids[:, 0]]) / SCALES),
            axis=1,
        )
        data_path = args.output / "phenol_sa6_5d_p_gauge.npz"
        np.savez_compressed(
            data_path,
            coordinates=coordinates,
            point_indices=np.asarray(points),
            canonical_indices=np.asarray(canonical),
            energies=energies,
            root_indices=roots,
            gauges=gauges,
            p_hamiltonian=p_hamiltonian,
            pairs=edge_ids,
            index_pairs=np.asarray(pairs),
            raw_overlaps=raw_overlaps,
            selected_overlaps=selected,
            p_links=p_links,
            selected_singular_values=singular,
            link_scaled_lengths=link_lengths,
            pair_axes=pair_axes,
            spanning_tree=tree_mask,
            energy_holdout=energy_holdout,
            link_holdout=link_holdout,
            reflection=reflection,
            metric=metrics,
            metric_eigenvalues=metric_eigenvalues,
            pseudopotential=pseudopotential,
            mode_frequencies_cm1=frequencies,
            mode_labels=np.asarray(labels),
            modes=chart.modes,
            coordinate_parities=PARITIES,
            coordinate_scales=SCALES,
            record_ids=diagnostic_record_ids,
            parent_record_ids=np.asarray(
                [fit.frames.record_id(point) for point in points]
            ),
            diagnostic_record_ids=diagnostic_record_ids,
            sa_energies=np.asarray(
                [record["energies"] for record in production_records]
            ),
            diagnostic_sa_energy_agreement=np.asarray(
                [record["sa_energy_agreement"] for record in records]
            ),
            candidate_pairs=candidate_edge_ids,
            candidate_index_pairs=np.asarray(candidate_pairs),
            candidate_raw_overlaps=candidate_raw,
            candidate_selected_singular_values=candidate_singular,
            candidate_link_scaled_lengths=candidate_lengths,
            qualified_candidate_links=qualified,
            threshold_qualified_candidate_links=threshold_qualified,
            transport_spanning_tree=transport_tree,
        )
        png, pdf = plot_results(
            args.output,
            coordinates,
            energies,
            p_hamiltonian,
            production_records,
            singular,
            metric_eigenvalues[:, 0],
        )
        macro = np.asarray([
            len(np.asarray(record.get("macro_history", ())))
            for record in production_records
        ])
        wall = np.asarray([
            float(record.get("wall_seconds", np.nan))
            for record in production_records
        ])
        full_singular = np.linalg.svd(raw_overlaps, compute_uv=False)
        gates = {
            "all_records_orbitally_relaxed": all(
                bool(record["orbital_relaxed"])
                for record in production_records
            ),
            "all_diagnostic_roots_singlets": bool(
                max(np.max(np.abs(record["spins"])) for record in records)
                <= 1.0e-5
            ),
            "diagnostic_first6_reproduce_sa6": bool(
                max(float(record["sa_energy_agreement"]) for record in records)
                <= 1.0e-6
            ),
            "outer_P1_uses_diagnostic_root6": all(
                int(roots[points.index(point), 1]) == 6
                for point in fixed_roots
                if R_GRID[point[0]] >= 2.10 - 1.0e-12
            ),
            "all_links_finite": bool(np.all(np.isfinite(raw_overlaps))),
            "tracked_link_minimum_above_threshold": bool(
                np.min(singular[:, -1]) >= float(args.minimum_link)
            ),
            "p_hamiltonian_reflection_covariant": bool(
                reflection_covariance_defect <= 1.0e-10
            ),
            "reflection_preserves_tracked_spectra": bool(
                reflection_spectral_defect <= 1.0e-10
            ),
            "metric_positive_definite": bool(np.min(metric_eigenvalues) > 0.0),
            "metric_and_pseudopotential_finite": bool(np.all(np.isfinite(metrics)) and np.all(np.isfinite(pseudopotential))),
            "keo_hermitian": bool(keo["hermiticity_defect"] <= 1.0e-9),
            "no_active_claims": database.stats["claims"] == 0,
        }
        passed = bool(all(gates.values()))
        database.update_run(run_id, "sampled" if passed else "failed")
        run = database.run(run_id)
        created = {entry["id"]: entry["created_at"] for entry in database.entries()}
        run_record_ids = {item["record_id"] for item in run["records"]}
        pilot_new_records = sum(
            created[record_id] >= run["created_at"] for record_id in run_record_ids
        )
        summary = {
            "passed": passed,
            "gates": gates,
            "dimensions": 5,
            "coordinates": chart.names,
            "mode_labels": labels,
            "mode_frequencies_cm1": frequencies,
            "input_mode_maximum_cartesian_reflection_defect_angstrom": input_mode_reflection_defect,
            "canonical_geometries": len(canonical),
            "effective_geometries": len(points),
            "candidate_overlap_edges": len(candidate_pairs),
            "overlap_edges": len(pairs),
            "pruned_low_quality_edges": int(np.count_nonzero(~qualified)),
            "retained_below_threshold_tree_edges": int(
                np.count_nonzero(qualified & ~threshold_qualified)
            ),
            "minimum_link_threshold": args.minimum_link,
            "workers": args.workers,
            "diagnostic_roots": args.diagnostic_roots,
            "diagnostic_records": diagnostic_stats,
            "diagnostic_overlaps": diagnostic_overlap_stats,
            "new_records_this_invocation": fit.frames.stats["built"],
            "reused_records_this_invocation": fit.frames.stats["database_hits"],
            "new_canonical_records_created_by_pilot": pilot_new_records,
            "preexisting_canonical_records_reused_by_pilot": len(canonical) - pilot_new_records,
            "sampling_reduction_fraction": 1.0 - len(canonical) / len(points),
            "median_macroiterations": float(np.median(macro)),
            "maximum_macroiterations": int(np.max(macro)),
            "median_record_wall_seconds": float(np.nanmedian(wall)),
            "minimum_full_diagnostic_window_overlap_singular_value": float(
                np.min(full_singular[:, -1])
            ),
            "minimum_tracked3_overlap_singular_value": float(np.min(singular[:, -1])),
            "median_tracked3_overlap_singular_value": float(np.median(singular[:, -1])),
            "maximum_p_hamiltonian_reflection_covariance_defect": reflection_covariance_defect,
            "maximum_reflection_spectral_defect_hartree": reflection_spectral_defect,
            "anchor_raw_reflection_offdiagonal_defect": float(
                np.linalg.norm(
                    raw_reflection - np.diag(np.diag(raw_reflection))
                )
            ),
            "maximum_scaled_link_length": float(np.max(link_lengths)),
            "minimum_metric_eigenvalue": float(np.min(metric_eigenvalues)),
            "maximum_metric_condition_number": float(np.max(metric_eigenvalues[:, -1] / metric_eigenvalues[:, 0])),
            "keo": keo,
            "database": str(args.database),
            "database_stats": database.stats,
            "frame_stats": fit.frames.stats,
            "wall_seconds": time.perf_counter() - started,
            "artifacts": {
                "data": str(data_path),
                "figure": str(png),
                "figure_pdf": str(pdf),
                "keo_mpo": keo["artifact"],
            },
        }
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
        print(json.dumps(_jsonable(summary), indent=2), flush=True)
    database.close()


if __name__ == "__main__":
    main()
