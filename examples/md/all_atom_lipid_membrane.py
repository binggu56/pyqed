#!/usr/bin/env python3
"""Tiny all-atom lipid-like bilayer MD smoke test.

This example is intentionally modest.  It uses explicit atom sites and
hydrogens, but the parameters are hand-built for a stability/topology test,
not a validated CHARMM/AMBER/OPLS lipid force field.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.md import (
    AU_PRESSURE_TO_BAR,
    Atoms,
    EnergyLogger,
    FixBondLengths,
    MCBarostatLogger,
    MDEngine,
    MonteCarloSemiIsotropicBarostat,
    MolecularMechanics,
    SemiIsotropicPressureController,
    XYZTrajectoryWriter,
    scale_molecule_centers,
    set_maxwell_boltzmann_velocities,
    semi_isotropic_pressure,
    soft_relaxation,
    write_minimization_log,
)
from pyqed.units import au2angstrom, au2fs, fs, kcalmol2au


def angstrom(value):
    return float(value) / au2angstrom


def kcal_per_mol(value):
    return float(value) * kcalmol2au


def kcal_per_mol_angstrom2(value):
    return float(value) * kcalmol2au * au2angstrom**2


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def safe_periodic_cutoff(box, requested_angstrom=10.0, fraction=0.45):
    return min(angstrom(requested_angstrom), fraction * float(np.min(box)))


TYPE_PARAMS = {
    "N_HEAD": {"epsilon": 0.12, "sigma": 3.25, "charge": 0.30},
    "P_HEAD": {"epsilon": 0.16, "sigma": 3.70, "charge": 0.50},
    "O_HEAD": {"epsilon": 0.12, "sigma": 3.00, "charge": -0.40},
    "C_HEAD": {"epsilon": 0.06, "sigma": 3.40, "charge": 0.00},
    "C_TAIL": {"epsilon": 0.09, "sigma": 3.55, "charge": 0.00},
    "H_TAIL": {"epsilon": 0.015, "sigma": 2.50, "charge": 0.00},
    "O_WATER": {"epsilon": 0.1521, "sigma": 3.1507, "charge": -0.834},
    "H_WATER": {"epsilon": 0.0, "sigma": 0.0, "charge": 0.417},
    "NA_ION": {"epsilon": 0.10, "sigma": 2.60, "charge": 1.0},
    "CL_ION": {"epsilon": 0.10, "sigma": 4.40, "charge": -1.0},
}


def _unit_vector(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return np.array([1.0, 0.0, 0.0])
    return np.asarray(vector, dtype=float) / norm


def _orthogonal_basis(axis):
    axis = _unit_vector(axis)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = _unit_vector(np.cross(axis, reference))
    v = _unit_vector(np.cross(axis, u))
    return u, v


STABILITY_PRESETS = {
    "stability-probe": {
        "nx": 1,
        "ny": 1,
        "steps": 4,
        "timestep_fs": 0.005,
        "temperature": 20.0,
        "friction": 0.2,
        "equilibration_steps": 2,
        "waters_per_lipid": 1,
        "salt_pairs": 0,
        "pme_mesh": 8,
        "trajectory_interval": 1,
        "energy_interval": 1,
        "pressure_control": True,
        "pressure_interval": 1,
        "pressure_coupling": 0.001,
        "pressure_max_scale": 0.001,
        "skip_relax": True,
    },
    "stability-0.01ps": {
        "nx": 1,
        "ny": 1,
        "steps": 200,
        "timestep_fs": 0.05,
        "temperature": 30.0,
        "friction": 0.2,
        "equilibration_steps": 20,
        "waters_per_lipid": 1,
        "salt_pairs": 0,
        "pme_mesh": 8,
        "trajectory_interval": 20,
        "energy_interval": 5,
        "pressure_control": True,
        "pressure_interval": 5,
        "pressure_coupling": 0.0008,
        "pressure_max_scale": 0.0008,
        "box_scale_lateral": 1.2,
        "box_scale_normal": 1.2,
        "skip_relax": False,
    },
    "stability-10ps": {
        "nx": 2,
        "ny": 2,
        "steps": 100000,
        "timestep_fs": 0.1,
        "temperature": 50.0,
        "friction": 0.2,
        "equilibration_steps": 1000,
        "waters_per_lipid": 2,
        "salt_pairs": 1,
        "pme_mesh": 16,
        "trajectory_interval": 1000,
        "energy_interval": 100,
        "pressure_control": True,
        "pressure_interval": 10,
        "pressure_coupling": 0.0005,
        "pressure_max_scale": 0.0005,
        "skip_relax": False,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(STABILITY_PRESETS), default=None)
    parser.add_argument("--nx", type=int, default=4, help="Lipids along x per leaflet.")
    parser.add_argument("--ny", type=int, default=4, help="Lipids along y per leaflet.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timestep-fs", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--friction", type=float, default=0.15)
    parser.add_argument("--equilibration-steps", type=int, default=5)
    parser.add_argument("--waters-per-lipid", type=int, default=3)
    parser.add_argument("--salt-pairs", type=int, default=2)
    parser.add_argument("--lipid-spacing", type=float, default=8.5, help="Initial lateral spacing between lipids in Angstrom.")
    parser.add_argument("--packing-seed", type=int, default=17, help="Seed for deterministic solvent/ion placement.")
    parser.add_argument("--coulomb-method", choices=("cutoff", "pme"), default="pme")
    parser.add_argument("--pme-mesh", type=int, default=24)
    parser.add_argument("--ewald-alpha", type=float, default=0.10, help="Ewald/PME alpha in bohr^-1.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--trajectory-interval", type=int, default=10)
    parser.add_argument("--energy-interval", type=int, default=5)
    parser.add_argument("--pressure-control", action="store_true", help="Attach the native semi-isotropic pressure controller.")
    parser.add_argument("--pressure-interval", type=int, default=5)
    parser.add_argument("--target-pressure-bar", type=float, default=1.0)
    parser.add_argument("--target-normal-pressure-bar", type=float, default=None)
    parser.add_argument("--compressibility-bar", type=float, default=4.5e-5)
    parser.add_argument("--pressure-coupling", type=float, default=0.01)
    parser.add_argument("--pressure-max-scale", type=float, default=0.005)
    parser.add_argument("--mc-barostat", action="store_true", help="Attach the native Metropolis semi-isotropic MC barostat.")
    parser.add_argument("--mc-interval", type=int, default=None, help="MC barostat attempt interval; defaults to --pressure-interval.")
    parser.add_argument("--mc-max-area-change", type=float, default=0.01, help="Maximum log area change per MC area move.")
    parser.add_argument("--mc-max-z-change", type=float, default=0.01, help="Maximum log z-length change per MC normal move.")
    parser.add_argument("--mc-stretch-atoms", action="store_true", help="Scale raw coordinates instead of molecule centers for MC moves.")
    parser.add_argument("--mc-log", default=None, help="MC barostat attempt log path; defaults inside output-dir.")
    parser.add_argument("--box-scale-lateral", type=float, default=1.0)
    parser.add_argument("--box-scale-normal", type=float, default=1.0)
    parser.add_argument("--relax-max-step-angstrom", type=float, default=0.02)
    parser.add_argument("--relax-fmax", type=float, default=5e-4)
    parser.add_argument("--output-dir", default="/private/tmp/pyqed_all_atom_lipid_membrane")
    parser.add_argument("--skip-relax", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="Skip PNG/density-plot generation for fast CI smoke runs.")
    args = parser.parse_args(argv)
    return apply_preset(args, sys.argv[1:] if argv is None else argv)


def apply_preset(args, argv):
    if args.preset is None:
        return args
    explicit = set()
    for token in argv:
        if token.startswith("--"):
            explicit.add(token[2:].split("=", 1)[0].replace("-", "_"))
    for name, value in STABILITY_PRESETS[args.preset].items():
        if name not in explicit:
            setattr(args, name, value)
    return args


def build_lipid(origin, leaflet_sign, lipid_id):
    x, y, _ = origin
    s = float(leaflet_sign)

    heavy = [
        ("N", "N_HEAD", np.array([x, y, s * angstrom(13.0)])),
        ("O", "O_HEAD", np.array([x - angstrom(0.8), y, s * angstrom(11.6)])),
        ("P", "P_HEAD", np.array([x, y, s * angstrom(10.4)])),
        ("O", "O_HEAD", np.array([x + angstrom(0.8), y, s * angstrom(9.2)])),
        ("C", "C_HEAD", np.array([x, y, s * angstrom(7.8)])),
        ("C", "C_TAIL", np.array([x - angstrom(1.7), y + angstrom(0.4), s * angstrom(6.3)])),
        ("C", "C_TAIL", np.array([x - angstrom(2.1), y - angstrom(0.5), s * angstrom(4.8)])),
        ("C", "C_TAIL", np.array([x - angstrom(1.8), y + angstrom(0.4), s * angstrom(3.3)])),
        ("C", "C_TAIL", np.array([x - angstrom(2.2), y - angstrom(0.4), s * angstrom(1.8)])),
        ("C", "C_TAIL", np.array([x + angstrom(1.7), y - angstrom(0.4), s * angstrom(6.3)])),
        ("C", "C_TAIL", np.array([x + angstrom(2.1), y + angstrom(0.5), s * angstrom(4.8)])),
        ("C", "C_TAIL", np.array([x + angstrom(1.8), y - angstrom(0.4), s * angstrom(3.3)])),
        ("C", "C_TAIL", np.array([x + angstrom(2.2), y + angstrom(0.4), s * angstrom(1.8)])),
    ]
    heavy_bonds = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (4, 9),
        (9, 10),
        (10, 11),
        (11, 12),
    ]

    records = []
    bonds = []
    constraints = []
    constraint_distances = []
    charges = []
    epsilon = []
    sigma = []
    atom_types = []
    lipid_ids = []
    regions = []

    def add_atom(symbol, atom_type, position, region):
        params = TYPE_PARAMS[atom_type]
        records.append([symbol, tuple(position)])
        charges.append(params["charge"])
        epsilon.append(kcal_per_mol(params["epsilon"]))
        sigma.append(angstrom(params["sigma"]))
        atom_types.append(atom_type)
        lipid_ids.append(lipid_id)
        regions.append(region)
        return len(records) - 1

    heavy_indices = []
    for symbol, atom_type, position in heavy:
        region = 0 if "HEAD" in atom_type else 1
        heavy_indices.append(add_atom(symbol, atom_type, position, region))

    for i, j in heavy_bonds:
        rij = heavy[i][2] - heavy[j][2]
        constraints.append((heavy_indices[i], heavy_indices[j]))
        constraint_distances.append(float(np.linalg.norm(rij)))

    # Explicit tail hydrogens.  C-H bonds are constrained to avoid the tiny
    # timestep that flexible all-atom hydrogen stretches would require.
    heavy_neighbor_graph = {i: set() for i in range(len(heavy))}
    for i, j in heavy_bonds:
        heavy_neighbor_graph[i].add(j)
        heavy_neighbor_graph[j].add(i)
    ch = angstrom(1.09)
    terminal_tail_atoms = {8, 12}
    for heavy_local_index in range(5, 13):
        parent = heavy_indices[heavy_local_index]
        parent_position = heavy[heavy_local_index][2]
        nh = 3 if heavy_local_index in terminal_tail_atoms else 2
        neighbors = sorted(heavy_neighbor_graph[heavy_local_index])
        tail_neighbors = [idx for idx in neighbors if idx >= 5]
        if len(tail_neighbors) == 2:
            axis = heavy[tail_neighbors[1]][2] - heavy[tail_neighbors[0]][2]
        elif tail_neighbors:
            axis = parent_position - heavy[tail_neighbors[0]][2]
        else:
            axis = heavy[neighbors[-1]][2] - heavy[neighbors[0]][2]
        u, v = _orthogonal_basis(axis)
        phase = (0.5 * np.pi * (heavy_local_index % 2)) + (0.25 * np.pi if lipid_id % 2 else 0.0)
        if nh == 2:
            angles = (phase, phase + np.pi)
        else:
            angles = (phase, phase + 2.0 * np.pi / 3.0, phase + 4.0 * np.pi / 3.0)
        for angle in angles:
            offset = ch * (np.cos(angle) * u + np.sin(angle) * v)
            h_index = add_atom("H", "H_TAIL", parent_position + offset, 1)
            constraints.append((parent, h_index))
            constraint_distances.append(ch)

    return (
        records,
        bonds,
        constraints,
        constraint_distances,
        charges,
        epsilon,
        sigma,
        atom_types,
        lipid_ids,
        regions,
    )


def build_membrane(
    nx=4,
    ny=4,
    waters_per_lipid=3,
    salt_pairs=2,
    coulomb_method="pme",
    pme_mesh=24,
    ewald_alpha=0.10,
    lipid_spacing_angstrom=8.5,
    packing_seed=17,
):
    spacing = angstrom(lipid_spacing_angstrom)
    box = np.array([nx * spacing, ny * spacing, angstrom(58.0)])
    atom_records = []
    bonds = []
    constraints = []
    constraint_distances = []
    charges = []
    epsilon = []
    sigma = []
    atom_types = []
    lipid_ids = []
    regions = []

    lipid_id = 0
    for leaflet_sign in (1.0, -1.0):
        for ix in range(nx):
            for iy in range(ny):
                offset = len(atom_records)
                leaflet_offset = 0.0 if leaflet_sign > 0.0 else 0.5
                lipid = build_lipid(
                    (
                        np.mod((ix + 0.5 + leaflet_offset) * spacing, box[0]),
                        np.mod((iy + 0.5 + leaflet_offset) * spacing, box[1]),
                        0.0,
                    ),
                    leaflet_sign,
                    lipid_id,
                )
                records, lipid_bonds, lipid_constraints, lipid_constraint_distances, q, eps, sig, types, ids, region = lipid
                atom_records.extend(records)
                bonds.extend(
                    (i + offset, j + offset, k, r0)
                    for i, j, k, r0 in lipid_bonds
                )
                constraints.extend((i + offset, j + offset) for i, j in lipid_constraints)
                constraint_distances.extend(lipid_constraint_distances)
                charges.extend(q)
                epsilon.extend(eps)
                sigma.extend(sig)
                atom_types.extend(types)
                lipid_ids.extend(ids)
                regions.extend(region)
                lipid_id += 1

    _add_water_and_ions(
        atom_records,
        constraints,
        constraint_distances,
        charges,
        epsilon,
        sigma,
        atom_types,
        lipid_ids,
        regions,
        box,
        nlipids=lipid_id,
        waters_per_lipid=waters_per_lipid,
        salt_pairs=salt_pairs,
        seed=packing_seed,
    )
    excluded_pairs = _topological_pairs_by_separation(constraints, max_separation=2)
    one_four_pairs = _topological_pairs_by_separation(
        constraints,
        max_separation=3,
    ) - excluded_pairs
    one_four_scales = {pair: 0.5 for pair in one_four_pairs}
    nonbonded_cutoff = safe_periodic_cutoff(box)
    calc = MolecularMechanics(
        bonds=bonds,
        charges=charges,
        coulomb_constant=1.0,
        coulomb_method=coulomb_method,
        coulomb_cutoff=nonbonded_cutoff,
        pme_mesh=(pme_mesh, pme_mesh, pme_mesh),
        ewald_alpha=ewald_alpha,
        lj_epsilon=epsilon,
        lj_sigma=sigma,
        lj_cutoff=nonbonded_cutoff,
        lj_energy_shift=True,
        exclude_bonded=True,
        exclude_angles=True,
        nonbonded_exclusions=excluded_pairs,
        lj_pair_scales=one_four_scales,
        coulomb_pair_scales=one_four_scales,
        nonbonded_skin=angstrom(1.0),
    )
    constraint = FixBondLengths(constraints, distances=constraint_distances)
    atoms = Atoms(atom_records, cell=box, pbc=True, calculator=calc, constraint=constraint)
    atoms.set_array("charges", charges, float, ())
    atoms.set_array("lipid_ids", lipid_ids, int, ())
    atoms.set_array("regions", regions, int, ())
    return atoms, atom_types


def _append_particle(
    records,
    charges,
    epsilon,
    sigma,
    atom_types,
    lipid_ids,
    regions,
    symbol,
    atom_type,
    position,
    lipid_id,
    region,
):
    params = TYPE_PARAMS[atom_type]
    records.append([symbol, tuple(position)])
    charges.append(params["charge"])
    epsilon.append(kcal_per_mol(params["epsilon"]))
    sigma.append(angstrom(params["sigma"]))
    atom_types.append(atom_type)
    lipid_ids.append(lipid_id)
    regions.append(region)
    return len(records) - 1


def _minimum_image_displacement(delta, box):
    delta = np.asarray(delta, dtype=float).copy()
    lengths = np.asarray(box, dtype=float)
    for axis in range(3):
        if lengths[axis] > 0.0:
            delta[axis] -= lengths[axis] * np.rint(delta[axis] / lengths[axis])
    return delta


def _placement_clear(candidate_positions, candidate_types, records, atom_types, box, scale=0.72):
    if not records:
        return True
    existing_positions = np.asarray([position for _symbol, position in records], dtype=float)
    for position, atom_type in zip(candidate_positions, candidate_types):
        candidate_params = TYPE_PARAMS[atom_type]
        candidate_sigma = angstrom(candidate_params["sigma"])
        candidate_epsilon = candidate_params["epsilon"]
        deltas = position - existing_positions
        deltas = np.array([_minimum_image_displacement(delta, box) for delta in deltas])
        distances = np.linalg.norm(deltas, axis=1)
        for distance, other_type in zip(distances, atom_types):
            other_params = TYPE_PARAMS[other_type]
            other_sigma = angstrom(other_params["sigma"])
            other_epsilon = other_params["epsilon"]
            if candidate_epsilon > 0.0 and other_epsilon > 0.0:
                contact = scale * 0.5 * (candidate_sigma + other_sigma)
            elif "H" in atom_type or "H" in other_type:
                contact = angstrom(1.15)
            else:
                contact = angstrom(1.8)
            if "ION" in atom_type or "ION" in other_type:
                contact = max(contact, angstrom(2.6))
            if distance < contact:
                return False
    return True


def _jittered_xy(base_xy, attempt, box):
    jitter = np.array(
        [
            0.37 * np.sin(1.618 * (attempt + 1.0)),
            0.31 * np.cos(2.414 * (attempt + 1.0)),
        ]
    )
    return np.mod(base_xy + jitter * angstrom(1.0), box[:2])


def _add_water_and_ions(
    records,
    constraints,
    constraint_distances,
    charges,
    epsilon,
    sigma,
    atom_types,
    lipid_ids,
    regions,
    box,
    nlipids,
    waters_per_lipid,
    salt_pairs,
    seed=17,
):
    nwaters = max(int(nlipids * waters_per_lipid), 0)
    water_rows = int(np.ceil(np.sqrt(max(nwaters // 2, 1))))
    water_spacing_x = box[0] / water_rows
    water_spacing_y = box[1] / water_rows
    oh = angstrom(0.9572)
    hh = angstrom(1.5139)
    water_id = nlipids
    rng = np.random.default_rng(int(seed))

    placed = 0
    max_attempts = max(200, nwaters * 120)
    for attempt in range(max_attempts):
        if placed >= nwaters:
            break
        sign = 1.0 if placed % 2 == 0 else -1.0
        xy = rng.random(2) * box[:2]
        z_abs = angstrom(float(rng.uniform(15.8, 25.2)))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        oxygen = np.array(
            [
                xy[0],
                xy[1],
                sign * z_abs,
            ]
        )
        h1 = oxygen + oh * np.array([np.cos(angle), np.sin(angle), 0.0])
        h2_angle = angle + np.deg2rad(104.52)
        h2 = oxygen + oh * np.array([np.cos(h2_angle), np.sin(h2_angle), 0.0])
        candidate_positions = (oxygen, h1, h2)
        candidate_types = ("O_WATER", "H_WATER", "H_WATER")
        if not _placement_clear(candidate_positions, candidate_types, records, atom_types, box):
            continue
        o_index = _append_particle(
            records, charges, epsilon, sigma, atom_types, lipid_ids, regions,
            "O", "O_WATER", oxygen, water_id, 2
        )
        h1_index = _append_particle(
            records, charges, epsilon, sigma, atom_types, lipid_ids, regions,
            "H", "H_WATER", h1, water_id, 2
        )
        h2_index = _append_particle(
            records, charges, epsilon, sigma, atom_types, lipid_ids, regions,
            "H", "H_WATER", h2, water_id, 2
        )
        constraints.extend([(o_index, h1_index), (o_index, h2_index), (h1_index, h2_index)])
        constraint_distances.extend([oh, oh, hh])
        water_id += 1
        placed += 1

    if placed < nwaters:
        for sign in (1.0, -1.0):
            for ix in range(water_rows):
                for iy in range(water_rows):
                    if placed >= nwaters:
                        break
                    layer = placed // max(2 * water_rows * water_rows, 1)
                    oxygen = np.array(
                        [
                            (ix + 0.45) * water_spacing_x,
                            (iy + 0.55) * water_spacing_y,
                            sign * angstrom(22.0 + 2.8 * layer),
                        ]
                    )
                    oxygen[:2] = np.mod(oxygen[:2], box[:2])
                    h1 = oxygen + np.array([oh, 0.0, 0.0])
                    h2 = oxygen + np.array([-0.2399872 / au2angstrom, 0.9266272 / au2angstrom, 0.0])
                    o_index = _append_particle(
                        records, charges, epsilon, sigma, atom_types, lipid_ids, regions,
                        "O", "O_WATER", oxygen, water_id, 2
                    )
                    h1_index = _append_particle(
                        records, charges, epsilon, sigma, atom_types, lipid_ids, regions,
                        "H", "H_WATER", h1, water_id, 2
                    )
                    h2_index = _append_particle(
                        records, charges, epsilon, sigma, atom_types, lipid_ids, regions,
                        "H", "H_WATER", h2, water_id, 2
                    )
                    constraints.extend([(o_index, h1_index), (o_index, h2_index), (h1_index, h2_index)])
                    constraint_distances.extend([oh, oh, hh])
                    water_id += 1
                    placed += 1
                if placed >= nwaters:
                    break

    ion_id = water_id
    for pair in range(max(int(salt_pairs), 0)):
        z = angstrom(float(rng.uniform(18.0, 25.0)))
        positions = [
            ("Na", "NA_ION", np.array([box[0] * rng.random(), box[1] * rng.random(), z])),
            ("Cl", "CL_ION", np.array([box[0] * rng.random(), box[1] * rng.random(), -z])),
        ]
        for symbol, atom_type, position in positions:
            for attempt in range(20):
                candidate = position.copy()
                candidate[:2] = _jittered_xy(candidate[:2], attempt + 11 * pair, box)
                if _placement_clear((candidate,), (atom_type,), records, atom_types, box, scale=0.85):
                    position = candidate
                    break
            position[:2] = np.mod(position[:2], box[:2])
            _append_particle(
                records, charges, epsilon, sigma, atom_types, lipid_ids, regions,
                symbol, atom_type, position, ion_id, 3
            )
            ion_id += 1


def _topological_pairs_by_separation(bonds, max_separation=2):
    """Return pairs separated by at most ``max_separation`` covalent links."""
    graph = {}
    for i, j in bonds:
        graph.setdefault(int(i), set()).add(int(j))
        graph.setdefault(int(j), set()).add(int(i))

    exclusions = set()
    for root in graph:
        visited = {root}
        frontier = {root}
        for _depth in range(max_separation):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(graph.get(node, ()))
            next_frontier -= visited
            for node in next_frontier:
                exclusions.add(tuple(sorted((root, node))))
            visited.update(next_frontier)
            frontier = next_frontier
    return exclusions


def _topological_exclusions(bonds, separation=2):
    return _topological_pairs_by_separation(bonds, max_separation=separation)


def refresh_safe_nonbonded_cutoff(atoms, requested_angstrom=10.0):
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return None
    cutoff = safe_periodic_cutoff(atoms.get_cell().lengths(), requested_angstrom=requested_angstrom)
    if hasattr(calc, "lj_cutoff") and calc.lj_cutoff is not None:
        calc.lj_cutoff = cutoff
    if hasattr(calc, "coulomb_cutoff") and calc.coulomb_cutoff is not None:
        calc.coulomb_cutoff = cutoff
    return cutoff


def analyze_membrane(atoms):
    positions = atoms.get_positions()
    forces = atoms.get_forces()
    lipid_ids = atoms.get_array("lipid_ids")
    regions = atoms.get_array("regions")
    lipid_mask = regions <= 1
    head_positions = positions[regions == 0]
    tail_positions = positions[regions == 1]
    water_positions = positions[regions == 2]
    ion_positions = positions[regions == 3]
    head_z = head_positions[:, 2]
    upper_heads = head_z[head_z > 0.0]
    lower_heads = head_z[head_z < 0.0]
    box_lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    nlipids = len(np.unique(lipid_ids[lipid_mask]))
    area_per_lipid = box_lengths[0] * box_lengths[1] / (0.5 * nlipids)
    return {
        "lipids": int(nlipids),
        "atoms": len(atoms),
        "net_charge": float(np.sum(atoms.get_array("charges"))),
        "area_per_lipid_angstrom2": area_per_lipid * au2angstrom**2,
        "upper_head_z_angstrom": float(np.mean(upper_heads) * au2angstrom),
        "lower_head_z_angstrom": float(np.mean(lower_heads) * au2angstrom),
        "head_head_thickness_angstrom": float((np.mean(upper_heads) - np.mean(lower_heads)) * au2angstrom),
        "tail_core_half_width_angstrom": float(np.mean(np.abs(tail_positions[:, 2])) * au2angstrom),
        "waters": sum(
            1
            for symbol, region in zip(atoms.atom_symbols(), regions)
            if region == 2 and symbol == "O"
        ),
        "ions": int(len(ion_positions)),
        "water_abs_z_mean_angstrom": float(np.mean(np.abs(water_positions[:, 2])) * au2angstrom) if len(water_positions) else 0.0,
        "temperature_K": float(atoms.get_temperature(remove_center_of_mass=True)),
        "potential_hartree": float(atoms.get_potential_energy()),
        "kinetic_hartree": float(atoms.get_kinetic_energy()),
        "finite_positions": bool(np.all(np.isfinite(positions))),
        "finite_forces": bool(np.all(np.isfinite(forces))),
        "max_force_hartree_per_bohr": float(np.max(np.linalg.norm(forces, axis=1))),
        "nonbonded_cutoff_angstrom": float(atoms.calc.lj_cutoff * au2angstrom),
        **constraint_metrics(atoms),
        **pressure_metrics(atoms),
    }


def constraint_metrics(atoms):
    errors = [
        constraint.max_error(atoms)
        for constraint in atoms.constraints
        if hasattr(constraint, "max_error")
    ]
    max_error = max(errors) if errors else 0.0
    return {
        "constraints": int(sum(
            len(getattr(constraint, "pairs", ()))
            for constraint in atoms.constraints
        )),
        "max_constraint_error_bohr": float(max_error),
        "max_constraint_error_angstrom": float(max_error * au2angstrom),
    }


def pressure_metrics(atoms):
    lateral, normal, tensor = semi_isotropic_pressure(atoms)
    diagonal = np.diag(tensor) * AU_PRESSURE_TO_BAR
    return {
        "pressure_lateral_bar": float(lateral * AU_PRESSURE_TO_BAR),
        "pressure_normal_bar": float(normal * AU_PRESSURE_TO_BAR),
        "pressure_xx_bar": float(diagonal[0]),
        "pressure_yy_bar": float(diagonal[1]),
        "pressure_zz_bar": float(diagonal[2]),
    }


def energy_component_metrics(atoms):
    calc = getattr(atoms, "calc", None)
    if calc is None or not hasattr(calc, "energy_components"):
        return {}
    components = calc.energy_components(atoms)
    return {
        f"energy_component_{name}_hartree": float(value)
        for name, value in components.items()
    }


def relaxation_metrics(relaxation):
    if not relaxation:
        return {
            "relaxation_enabled": False,
            "relaxation_total_steps": 0,
        }
    final = relaxation[-1]
    total_steps = sum(int(stage.get("steps", 0)) for stage in relaxation)
    return {
        "relaxation_enabled": True,
        "relaxation_total_steps": int(total_steps),
        "relaxation_final_fmax_hartree_per_bohr": float(final.get("fmax", np.nan)),
        "relaxation_final_energy_hartree": float(final.get("energy", np.nan)),
        "relaxation_converged": bool(final.get("converged", False)),
    }


def energy_log_metrics(path):
    path = Path(path)
    if not path.exists():
        return {"energy_log_rows": 0, "finite_energy_log": False}
    lines = path.read_text().splitlines()
    if not lines:
        return {"energy_log_rows": 0, "finite_energy_log": False}
    header = lines[0].split()
    data_lines = [line for line in lines[1:] if line.strip()]
    if not data_lines:
        return {"energy_log_rows": 0, "finite_energy_log": True}
    data = np.loadtxt(data_lines, ndmin=2)
    columns = {name: data[:, index] for index, name in enumerate(header)}
    finite = bool(np.all(np.isfinite(data)))
    metrics = {
        "energy_log_rows": int(data.shape[0]),
        "finite_energy_log": finite,
    }

    def add_range(name, key=None):
        if name not in columns:
            return
        metric_key = key or name
        values = columns[name]
        metrics[f"{metric_key}_min"] = float(np.min(values))
        metrics[f"{metric_key}_max"] = float(np.max(values))

    add_range("temperature_K")
    add_range("pressure_lateral_bar")
    add_range("pressure_normal_bar")
    add_range("pressure_xx_bar")
    add_range("pressure_yy_bar")
    add_range("pressure_zz_bar")
    if "total" in columns:
        total = columns["total"]
        drift = float(total[-1] - total[0])
        metrics["total_energy_drift_hartree"] = drift
        if "time" in columns:
            time_fs = columns["time"] * au2fs
            span_ps = float((time_fs[-1] - time_fs[0]) / 1000.0)
            metrics["energy_log_time_span_ps"] = span_ps
            if span_ps > 0.0:
                metrics["total_energy_drift_per_ps_hartree"] = float(drift / span_ps)
    return metrics


def mc_barostat_log_metrics(path):
    path = Path(path)
    if not path.exists():
        return {"mc_log_rows": 0, "finite_mc_log": False}
    lines = path.read_text().splitlines()
    if not lines:
        return {"mc_log_rows": 0, "finite_mc_log": False}
    header = lines[0].split()
    data_lines = [line for line in lines[1:] if line.strip()]
    if not data_lines:
        return {"mc_log_rows": 0, "finite_mc_log": True}
    data = np.loadtxt(data_lines, ndmin=2, usecols=tuple(index for index, name in enumerate(header) if name != "move"))
    numeric_names = [name for name in header if name != "move"]
    columns = {name: data[:, index] for index, name in enumerate(numeric_names)}
    finite = bool(np.all(np.isfinite(data)))
    metrics = {
        "mc_log_rows": int(data.shape[0]),
        "finite_mc_log": finite,
    }
    if "accepted" in columns:
        accepted = columns["accepted"]
        attempts = len(accepted)
        metrics["mc_log_accepted"] = int(np.count_nonzero(accepted > 0))
        metrics["mc_log_acceptance_rate"] = float(metrics["mc_log_accepted"] / attempts) if attempts else 0.0

    def add_range(name, key=None):
        if name not in columns:
            return
        metric_key = key or name
        values = columns[name]
        metrics[f"{metric_key}_min"] = float(np.min(values))
        metrics[f"{metric_key}_max"] = float(np.max(values))

    for name in (
        "area_per_lipid_angstrom2",
        "lz",
        "pressure_lateral_bar",
        "pressure_normal_bar",
        "log_acceptance",
        "work",
        "delta_energy",
    ):
        add_range(name, f"mc_{name}")
    return metrics


def density_profiles(atoms, bins=80):
    positions = atoms.get_positions() * au2angstrom
    regions = atoms.get_array("regions")
    z = positions[:, 2]
    zmin, zmax = -0.5 * atoms.get_cell().lengths()[2] * au2angstrom, 0.5 * atoms.get_cell().lengths()[2] * au2angstrom
    edges = np.linspace(zmin, zmax, int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profiles = {"head": regions == 0, "tail": regions == 1, "water": regions == 2, "ion": regions == 3}
    counts = {name: np.histogram(z[mask], bins=edges)[0] for name, mask in profiles.items()}
    return centers, counts


def write_density_profiles(atoms, data_path, plot_path):
    plt = _pyplot()
    centers, counts = density_profiles(atoms)
    with open(data_path, "w") as handle:
        handle.write("z_angstrom head tail water ion\n")
        for index, z_value in enumerate(centers):
            handle.write(
                f"{z_value:.6f} {counts['head'][index]} {counts['tail'][index]} "
                f"{counts['water'][index]} {counts['ion'][index]}\n"
            )

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax.plot(centers, counts["head"], color="#d84a3a", label="head")
    ax.plot(centers, counts["tail"], color="#2f684e", label="tail")
    ax.plot(centers, counts["water"], color="#4b8fd8", label="water")
    ax.plot(centers, counts["ion"], color="#7d4ab8", label="ions")
    ax.set_xlabel("z / A")
    ax.set_ylabel("atom count per bin")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def membrane_display_positions(atoms, molecule_array="lipid_ids"):
    """Return molecule-unwrapped, bilayer-centered positions for diagnostics."""
    positions = np.asarray(atoms.get_positions(), dtype=float)
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    display = positions.copy()
    molecule_ids = atoms.get_array(molecule_array) if atoms.has(molecule_array) else np.arange(len(atoms))

    for molecule_id in np.unique(molecule_ids):
        indices = np.flatnonzero(molecule_ids == molecule_id)
        if len(indices) <= 1:
            continue
        reference = positions[indices[0]]
        deltas = positions[indices] - reference
        for axis in range(3):
            if pbc[axis] and lengths[axis] > 0.0:
                deltas[:, axis] -= np.round(deltas[:, axis] / lengths[axis]) * lengths[axis]
        display[indices] = reference + deltas

    regions = atoms.get_array("regions") if atoms.has("regions") else np.zeros(len(atoms), dtype=int)
    lipid_mask = regions <= 1
    center = np.zeros(3)
    center[:2] = 0.5 * lengths[:2]
    if np.any(regions == 0):
        center[2] = float(np.mean(display[regions == 0, 2]))
    elif np.any(lipid_mask):
        center[2] = float(np.mean(display[lipid_mask, 2]))

    display -= center
    for molecule_id in np.unique(molecule_ids):
        indices = np.flatnonzero(molecule_ids == molecule_id)
        molecule_center = np.mean(display[indices], axis=0)
        shift = np.zeros(3)
        for axis in (0, 1):
            if pbc[axis] and lengths[axis] > 0.0:
                shift[axis] = -np.round(molecule_center[axis] / lengths[axis]) * lengths[axis]
        display[indices] += shift
    return display * au2angstrom


def membrane_render_bonds(atoms):
    pairs = []
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        pairs.extend((int(i), int(j)) for i, j, *_ in getattr(calc, "bonds", ()))
    for constraint in getattr(atoms, "constraints", ()):
        pairs.extend(
            (int(i), int(j))
            for i, j in getattr(constraint, "pairs", ())
        )
    return sorted({tuple(sorted(pair)) for pair in pairs})


def _add_bond_lines_3d(ax, positions, pairs, mask=None, color="#9b9b9b", linewidth=0.45, alpha=0.48):
    if not pairs:
        return
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segments = []
    for i, j in pairs:
        if mask is not None and not (mask[i] and mask[j]):
            continue
        segments.append([positions[i], positions[j]])
    if segments:
        ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=linewidth, alpha=alpha))


def _add_bond_lines_2d(ax, positions, pairs, mask=None, color="#9b9b9b", linewidth=0.45, alpha=0.40):
    for i, j in pairs:
        if mask is not None and not (mask[i] and mask[j]):
            continue
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 2], positions[j, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )


def render_membrane(atoms, image_path):
    plt = _pyplot()
    positions = membrane_display_positions(atoms)
    symbols = np.array(atoms.atom_symbols())
    regions = atoms.get_array("regions")
    bonds = membrane_render_bonds(atoms)
    colors = {
        "head": "#d84a3a",
        "carbon": "#2f684e",
        "hydrogen": "#d9d9d9",
    }

    fig = plt.figure(figsize=(8, 6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    masks = [
        ("head atoms", regions == 0, colors["head"], 30),
        ("tail carbons", (regions == 1) & (symbols == "C"), colors["carbon"], 16),
        ("tail hydrogens", (regions == 1) & (symbols == "H"), colors["hydrogen"], 5),
        ("water oxygens", (regions == 2) & (symbols == "O"), "#4b8fd8", 12),
        ("ions", regions == 3, "#7d4ab8", 34),
    ]
    _add_bond_lines_3d(ax, positions, bonds, mask=regions <= 2)
    for label, mask, color, size in masks:
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            positions[mask, 2],
            s=size,
            c=color,
            edgecolors="#222222" if size > 10 else "none",
            linewidths=0.25,
            alpha=0.92,
            depthshade=True,
            label=label,
        )

    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    ax.set_xlim(-0.5 * lengths[0], 0.5 * lengths[0])
    ax.set_ylim(-0.5 * lengths[1], 0.5 * lengths[1])
    ax.set_zlim(-0.5 * lengths[2], 0.5 * lengths[2])
    ax.set_xlabel("x / A")
    ax.set_ylabel("y / A")
    ax.set_zlabel("z / A")
    ax.view_init(elev=22, azim=-52)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(image_path)
    plt.close(fig)


def render_cross_section(atoms, image_path):
    plt = _pyplot()
    positions = membrane_display_positions(atoms)
    symbols = np.array(atoms.atom_symbols())
    regions = atoms.get_array("regions")
    bonds = membrane_render_bonds(atoms)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    layers = [
        ("tail H", (regions == 1) & (symbols == "H"), "#d0d0d0", 4, 0.35),
        ("tail C", (regions == 1) & (symbols == "C"), "#2f684e", 12, 0.8),
        ("head", regions == 0, "#d84a3a", 20, 0.9),
        ("water O", (regions == 2) & (symbols == "O"), "#4b8fd8", 12, 0.75),
        ("ions", regions == 3, "#7d4ab8", 32, 0.95),
    ]
    _add_bond_lines_2d(ax, positions, bonds, mask=regions <= 2)
    for label, mask, color, size, alpha in layers:
        ax.scatter(positions[mask, 0], positions[mask, 2], s=size, c=color, alpha=alpha, label=label, zorder=2)
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.35)
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    ax.set_xlim(-0.5 * lengths[0], 0.5 * lengths[0])
    ax.set_ylim(-0.5 * lengths[2], 0.5 * lengths[2])
    ax.set_xlabel("x / A")
    ax.set_ylabel("z / A")
    ax.legend(frameon=False, ncol=5, loc="upper center")
    ax.set_title("PBC-unwrapped hydrated bilayer cross-section")
    fig.tight_layout()
    fig.savefig(image_path)
    plt.close(fig)


def write_summary(
    path,
    metrics,
    dynamics,
    trajectory_path,
    energy_path,
    image_path,
    cross_section_path,
    density_data_path,
    density_plot_path,
):
    with open(path, "w") as handle:
        handle.write(f"steps: {dynamics.get_number_of_steps()}\n")
        handle.write(f"time_fs: {dynamics.get_time() * au2fs:.6f}\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                handle.write(f"{key}: {value:.8f}\n")
            else:
                handle.write(f"{key}: {value}\n")
        handle.write(f"trajectory: {trajectory_path}\n")
        handle.write(f"energy_log: {energy_path}\n")
        handle.write(f"image: {image_path}\n")
        handle.write(f"cross_section: {cross_section_path}\n")
        handle.write(f"density_profile: {density_data_path}\n")
        handle.write(f"density_plot: {density_plot_path}\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atoms, _atom_types = build_membrane(
        args.nx,
        args.ny,
        waters_per_lipid=args.waters_per_lipid,
        salt_pairs=args.salt_pairs,
        coulomb_method=args.coulomb_method,
        pme_mesh=args.pme_mesh,
        ewald_alpha=args.ewald_alpha,
        lipid_spacing_angstrom=args.lipid_spacing,
        packing_seed=args.packing_seed,
    )
    applied_box_scale = scale_molecule_centers(
        atoms,
        lateral_scale=args.box_scale_lateral,
        normal_scale=args.box_scale_normal,
        array_name="lipid_ids",
    )
    refresh_safe_nonbonded_cutoff(atoms)
    relaxation = []
    if not args.skip_relax:
        relaxation = soft_relaxation(
            atoms,
            stages=((0.1, 0.05, 25), (0.3, 0.2, 25), (0.6, 0.5, 25), (1.0, 1.0, 25)),
            max_step=angstrom(args.relax_max_step_angstrom),
            fmax=args.relax_fmax,
        )
        write_minimization_log(output_dir / "all_atom_lipid_membrane_relax.dat", relaxation)

    set_maxwell_boltzmann_velocities(atoms, args.temperature, seed=args.seed)
    engine = MDEngine(
        atoms,
        timestep=args.timestep_fs * fs,
        ensemble="langevin",
        temperature_K=args.temperature,
        friction=args.friction,
    )
    dynamics = engine.dynamics
    pressure_controller = None
    mc_logger = None
    mc_log_path = None
    if args.mc_barostat:
        pressure_controller = MonteCarloSemiIsotropicBarostat.from_bar(
            atoms,
            temperature_K=args.temperature,
            target_lateral_pressure_bar=args.target_pressure_bar,
            target_normal_pressure_bar=args.target_normal_pressure_bar,
            max_area_change=args.mc_max_area_change,
            max_z_change=args.mc_max_z_change,
            scale_molecule_centers=not args.mc_stretch_atoms,
            molecule_array="lipid_ids",
            seed=args.seed + 1009,
        )
        dynamics.attach(pressure_controller, interval=args.mc_interval or args.pressure_interval)
        mc_log_path = Path(args.mc_log) if args.mc_log is not None else output_dir / "mc_barostat.dat"
        mc_logger = MCBarostatLogger(
            pressure_controller,
            mc_log_path,
            dynamics=dynamics,
            lipids_per_leaflet=args.nx * args.ny,
        )
        dynamics.attach(mc_logger, interval=args.mc_interval or args.pressure_interval)
    elif args.pressure_control:
        pressure_controller = SemiIsotropicPressureController.from_bar(
            atoms,
            target_lateral_pressure_bar=args.target_pressure_bar,
            target_normal_pressure_bar=args.target_normal_pressure_bar,
            compressibility_bar=args.compressibility_bar,
            coupling=args.pressure_coupling,
            max_scale=args.pressure_max_scale,
        )
        dynamics.attach(pressure_controller, interval=args.pressure_interval)
    trajectory_path = output_dir / "all_atom_lipid_membrane.xyz"
    energy_path = output_dir / "all_atom_lipid_membrane_energy.dat"
    image_path = output_dir / "all_atom_lipid_membrane_final.png"
    cross_section_path = output_dir / "all_atom_lipid_membrane_cross_section.png"
    density_data_path = output_dir / "density_profile_z.dat"
    density_plot_path = output_dir / "density_profile_z.png"
    summary_path = output_dir / "summary.txt"
    if args.equilibration_steps > 0:
        engine.run(args.equilibration_steps)
    writer = XYZTrajectoryWriter(atoms, trajectory_path, dynamics=dynamics)
    logger = EnergyLogger(atoms, energy_path, dynamics=dynamics)
    dynamics.attach(writer, interval=args.trajectory_interval)
    dynamics.attach(logger, interval=args.energy_interval)
    try:
        engine.run(args.steps)
    finally:
        writer.close()
        logger.close()
        if mc_logger is not None:
            mc_logger.close()

    metrics = analyze_membrane(atoms)
    metrics.update(energy_component_metrics(atoms))
    metrics.update(relaxation_metrics(relaxation))
    metrics.update(energy_log_metrics(energy_path))
    if mc_log_path is not None:
        metrics.update(mc_barostat_log_metrics(mc_log_path))
    metrics["preset"] = args.preset or "custom"
    metrics["equilibration_steps"] = int(max(args.equilibration_steps, 0))
    metrics["production_steps"] = int(args.steps)
    metrics["production_time_ps"] = float(args.steps * args.timestep_fs / 1000.0)
    metrics["pressure_control"] = bool(args.pressure_control or args.mc_barostat)
    metrics["pressure_control_mode"] = "mc" if args.mc_barostat else ("weak" if args.pressure_control else "none")
    metrics["render_outputs"] = not bool(args.no_render)
    metrics["packing_seed"] = int(args.packing_seed)
    metrics["box_scale_lateral"] = float(applied_box_scale[0])
    metrics["box_scale_normal"] = float(applied_box_scale[2])
    if pressure_controller is not None:
        if isinstance(pressure_controller, MonteCarloSemiIsotropicBarostat):
            metrics["mc_barostat_attempts"] = int(pressure_controller.attempts)
            metrics["mc_barostat_accepted"] = int(pressure_controller.accepted)
            metrics["mc_barostat_acceptance_rate"] = float(pressure_controller.acceptance_rate)
            metrics["mc_barostat_last_accepted"] = bool(pressure_controller.last_accepted)
            metrics["mc_barostat_last_move"] = str(pressure_controller.last_move)
            metrics["mc_barostat_last_scale_x"] = float(pressure_controller.last_scale[0])
            metrics["mc_barostat_last_scale_z"] = float(pressure_controller.last_scale[2])
            metrics["mc_barostat_last_work_hartree"] = float(pressure_controller.last_work)
            metrics["mc_barostat_last_delta_energy_hartree"] = float(pressure_controller.last_delta_energy)
            metrics["mc_barostat_last_log_acceptance"] = float(pressure_controller.last_log_acceptance)
            metrics["mc_barostat_log"] = str(mc_log_path)
        else:
            metrics["pressure_controller_last_scale_x"] = float(pressure_controller.last_scale[0])
            metrics["pressure_controller_last_scale_z"] = float(pressure_controller.last_scale[2])
    if not args.no_render:
        render_membrane(atoms, image_path)
        render_cross_section(atoms, cross_section_path)
        write_density_profiles(atoms, density_data_path, density_plot_path)
    write_summary(
        summary_path,
        metrics,
        dynamics,
        trajectory_path,
        energy_path,
        image_path,
        cross_section_path,
        density_data_path,
        density_plot_path,
    )
    print(summary_path)
    print(summary_path.read_text(), end="")


if __name__ == "__main__":
    main()
