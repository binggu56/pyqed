"""Small solvent-box builders for :mod:`pyqed.md`."""

import numpy as np

from pyqed.units import au2angstrom

from .atoms import Atoms
from .calculators import MM
from .constraints import FixBondLengths
from .topology import Topology, combine_topologies
from .water import TIP3P_COULOMB_CONSTANT, TIP3P_HOH_ANGLE, TIP3P_OH_DISTANCE, tip3p_waters

WATER_MOLAR_MASS = 18.01528
AVOGADRO = 6.02214076e23
ANGSTROM3_PER_CM3 = 1.0e24


def combine_systems(
    systems,
    cell=None,
    pbc=False,
    calculator=True,
    coulomb_constant=TIP3P_COULOMB_CONSTANT,
    coulomb_method="cutoff",
    lj_cutoff=None,
    coulomb_cutoff=None,
    ewald_alpha=0.35,
    ewald_kmax=5,
    pme_mesh=(16, 16, 16),
):
    """Combine multiple :class:`Atoms` objects and their topology metadata."""
    atom_records = []
    topologies = []
    constraint_pairs = []
    constraint_distances = []
    atom_offset = 0
    molecule_offset = 0
    for system in systems:
        positions = system.get_positions()
        symbols = system.atom_symbols()
        atom_records.extend([[symbol, tuple(xyz)] for symbol, xyz in zip(symbols, positions)])
        topology = _topology_with_defaults(system).shifted(atom_offset, molecule_offset)
        topologies.append(topology)
        for constraint in system.constraints:
            if isinstance(constraint, FixBondLengths):
                targets = constraint._targets(system)
                constraint_pairs.extend(
                    [(i + atom_offset, j + atom_offset) for i, j in constraint.pairs]
                )
                constraint_distances.extend(targets)
        atom_offset += len(system)
        if topology.molecule_ids is not None and len(topology.molecule_ids):
            molecule_offset = int(np.max(topology.molecule_ids)) + 1
        else:
            molecule_offset += 1

    topology = combine_topologies(topologies)
    mm = None
    if calculator:
        mm = MM(
            bonds=topology.bonds,
            angles=topology.angles,
            torsions=topology.torsions,
            angle_unit="degree",
            torsion_unit="degree",
            charges=topology.charges,
            coulomb_constant=coulomb_constant,
            coulomb_method=coulomb_method,
            coulomb_cutoff=coulomb_cutoff,
            ewald_alpha=ewald_alpha,
            ewald_kmax=ewald_kmax,
            pme_mesh=pme_mesh,
            lj_epsilon=topology.lj_epsilon,
            lj_sigma=topology.lj_sigma,
            lj_cutoff=lj_cutoff,
            exclude_bonded=True,
            exclude_angles=True,
        )

    constraint = None
    if constraint_pairs:
        constraint = FixBondLengths(constraint_pairs, distances=constraint_distances)

    combined = Atoms(atom_records, cell=cell, pbc=pbc, calculator=mm, constraint=constraint)
    combined.topology = topology
    combined.set_array("charges", topology.charges, float, ())
    combined.set_array("lj_epsilon", topology.lj_epsilon, float, ())
    combined.set_array("lj_sigma", topology.lj_sigma, float, ())
    combined.set_array("molecule_ids", topology.molecule_ids, int, ())
    return combined


def solvate_box(
    solute=None,
    box_size=(18.0, 18.0, 18.0),
    spacing=3.1,
    min_distance=2.2,
    max_waters=None,
    pbc=True,
    lj_cutoff=9.0,
    coulomb_cutoff=9.0,
    coulomb_method="cutoff",
    ewald_alpha=0.35,
    ewald_kmax=5,
    pme_mesh=(16, 16, 16),
    rigid=False,
    placement="grid",
    seed=None,
    max_attempts=10000,
    density=None,
    water_oxygen_min_distance=None,
    placement_relaxation=1.0,
):
    """Build a simple TIP3P solvent box, optionally around a solute.

    This is useful for local development and short smoke simulations, not
    production water thermodynamics.
    """
    box_size = np.asarray(box_size, dtype=float)
    if box_size.shape != (3,):
        raise ValueError("box_size must have shape (3,).")
    if density is not None and max_waters is None:
        max_waters = water_count_for_density(box_size, density=density)

    placement = placement.lower()
    if placement == "grid":
        origins = _water_origins(box_size, spacing)
        rotations = None
        origins = _reject_overlaps(origins, solute, min_distance)
        if max_waters is not None:
            origins = origins[: int(max_waters)]
    elif placement == "random":
        if max_waters is None:
            raise ValueError("random placement requires max_waters.")
        origins, rotations = _random_water_origins(
            box_size,
            int(max_waters),
            min_distance,
            solute=solute,
            seed=seed,
            max_attempts=max_attempts,
            water_oxygen_min_distance=water_oxygen_min_distance,
            relaxation=placement_relaxation,
        )
    else:
        raise ValueError("placement must be 'grid' or 'random'.")

    waters = tip3p_waters(
        origins,
        cell=box_size,
        pbc=pbc,
        calculator=False,
        rigid=rigid,
        rotations=rotations,
    )
    if solute is None:
        system = combine_systems(
            [waters],
            cell=box_size,
            pbc=pbc,
            lj_cutoff=lj_cutoff,
            coulomb_cutoff=coulomb_cutoff,
            coulomb_method=coulomb_method,
            ewald_alpha=ewald_alpha,
            ewald_kmax=ewald_kmax,
            pme_mesh=pme_mesh,
        )
    else:
        system = combine_systems(
            [solute, waters],
            cell=box_size,
            pbc=pbc,
            lj_cutoff=lj_cutoff,
            coulomb_cutoff=coulomb_cutoff,
            coulomb_method=coulomb_method,
            ewald_alpha=ewald_alpha,
            ewald_kmax=ewald_kmax,
            pme_mesh=pme_mesh,
        )
    system.solvation = {
        "placement": placement,
        "requested_waters": None if max_waters is None else int(max_waters),
        "placed_waters": int(len(origins)),
        "density_g_cm3": water_density(system, solute_atoms=0 if solute is None else len(solute)),
        "water_oxygen_min_distance": water_oxygen_min_distance,
    }
    return system


def water_count_for_density(box_size, density=1.0, molar_mass=WATER_MOLAR_MASS):
    """Return the nearest water count for a box and target density.

    ``box_size`` is in Bohr, and density is in g/cm^3.
    """
    lengths_angstrom = np.asarray(box_size, dtype=float) * au2angstrom
    volume_angstrom3 = float(np.prod(lengths_angstrom))
    number_density = float(density) * AVOGADRO / molar_mass / ANGSTROM3_PER_CM3
    return max(int(round(number_density * volume_angstrom3)), 0)


def water_density(atoms, solute_atoms=0, molar_mass=WATER_MOLAR_MASS):
    """Estimate water density in g/cm^3 from water molecules after solute atoms."""
    nwater_atoms = max(len(atoms) - int(solute_atoms), 0)
    nwaters = nwater_atoms // 3
    lengths_angstrom = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    volume_angstrom3 = float(np.prod(lengths_angstrom))
    if volume_angstrom3 == 0.0:
        return 0.0
    mass_g = nwaters * molar_mass / AVOGADRO
    volume_cm3 = volume_angstrom3 / ANGSTROM3_PER_CM3
    return mass_g / volume_cm3


def _topology_with_defaults(atoms):
    topology = Topology.from_atoms(atoms)
    natoms = len(atoms)
    molecule_ids = topology.molecule_ids
    if molecule_ids is None:
        molecule_ids = np.zeros(natoms, dtype=int)
    return Topology(
        bonds=topology.bonds,
        angles=topology.angles,
        torsions=getattr(topology, "torsions", []),
        charges=_array_or_zeros(topology.charges, natoms),
        lj_epsilon=_array_or_zeros(topology.lj_epsilon, natoms),
        lj_sigma=_array_or_zeros(topology.lj_sigma, natoms),
        molecule_ids=molecule_ids,
    )


def _array_or_zeros(array, natoms):
    if array is None:
        return np.zeros(natoms)
    array = np.asarray(array, dtype=float)
    if array.shape != (natoms,):
        raise ValueError(f"topology array has shape {array.shape}, expected ({natoms},).")
    return array


def _water_origins(box_size, spacing):
    axes = [np.arange(0.5 * spacing, length - 0.5 * spacing + 1e-12, spacing) for length in box_size]
    if any(len(axis) == 0 for axis in axes):
        return np.empty((0, 3))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([axis.ravel() for axis in mesh])


def _reject_overlaps(origins, solute, min_distance):
    if solute is None or len(origins) == 0:
        return origins
    solute_positions = solute.get_positions()
    local = _tip3p_local_positions()
    accepted = []
    min_distance2 = min_distance * min_distance
    for origin in origins:
        water_positions = origin + local
        deltas = water_positions[:, None, :] - solute_positions[None, :, :]
        if np.all(np.sum(deltas * deltas, axis=-1) >= min_distance2):
            accepted.append(origin)
    return np.asarray(accepted, dtype=float)


def _tip3p_local_positions():
    theta = np.deg2rad(TIP3P_HOH_ANGLE)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [TIP3P_OH_DISTANCE, 0.0, 0.0],
            [
                TIP3P_OH_DISTANCE * np.cos(theta),
                TIP3P_OH_DISTANCE * np.sin(theta),
                0.0,
            ],
        ]
    )


def _random_water_origins(
    box_size,
    nwaters,
    min_distance,
    solute=None,
    seed=None,
    max_attempts=10000,
    water_oxygen_min_distance=None,
    relaxation=1.0,
):
    rng = np.random.default_rng(seed)
    local = _tip3p_local_positions()
    margin = TIP3P_OH_DISTANCE
    low = np.full(3, margin)
    high = box_size - margin
    if np.any(high <= low):
        raise ValueError("box is too small for random water placement.")

    water_oxygen_min_distance = (
        min_distance if water_oxygen_min_distance is None else float(water_oxygen_min_distance)
    )
    relaxation = float(relaxation)
    if relaxation <= 0.0 or relaxation > 1.0:
        raise ValueError("placement_relaxation must be in (0, 1].")

    accepted_origins = []
    accepted_rotations = []
    solute_positions = []
    if solute is not None:
        solute_positions = solute.get_positions()
    solute_positions = np.asarray(solute_positions, dtype=float).reshape(-1, 3)

    attempts = 0
    solute_min = float(min_distance)
    water_oxygen_min = float(water_oxygen_min_distance)
    while len(accepted_origins) < nwaters and attempts < max_attempts:
        block_attempts = max(max_attempts // 5, 1)
        block_end = min(attempts + block_attempts, max_attempts)
        solute_min2 = solute_min * solute_min
        oo_min2 = water_oxygen_min * water_oxygen_min
        while len(accepted_origins) < nwaters and attempts < block_end:
            attempts += 1
            rotation = _random_rotation_matrix(rng)
            origin = rng.uniform(low, high)
            water_positions = origin + local @ rotation.T
            if solute_positions.size:
                deltas = water_positions[:, None, :] - solute_positions[None, :, :]
                if np.any(np.sum(deltas * deltas, axis=-1) < solute_min2):
                    continue
            if accepted_origins:
                oo_deltas = origin - np.asarray(accepted_origins)
                if np.any(np.sum(oo_deltas * oo_deltas, axis=1) < oo_min2):
                    continue
            accepted_origins.append(origin)
            accepted_rotations.append(rotation)
        solute_min *= relaxation
        water_oxygen_min *= relaxation

    if len(accepted_origins) < nwaters:
        raise RuntimeError(
            f"placed {len(accepted_origins)} waters after {max_attempts} attempts; "
            "try fewer waters, a larger box, or a smaller min_distance."
        )

    return np.asarray(accepted_origins), np.asarray(accepted_rotations)


def _random_rotation_matrix(rng):
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ]
    )
