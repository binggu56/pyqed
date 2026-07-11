"""Water model builders for :mod:`pyqed.md`."""

import numpy as np

from pyqed.units import au2angstrom, kcalmol2au

from .atoms import Atoms
from .calculators import MM
from .constraints import FixBondLengths
from .topology import Topology


TIP3P_OH_DISTANCE_ANGSTROM = 0.9572
TIP3P_LJ_SIGMA_ANGSTROM = 3.1507
TIP3P_LJ_EPSILON_KCAL_MOL = 0.1521
TIP3P_BOND_FORCE_CONSTANT_KCAL_MOL_ANGSTROM2 = 450.0
TIP3P_ANGLE_FORCE_CONSTANT_KCAL_MOL_RAD2 = 55.0

TIP3P_OH_DISTANCE = TIP3P_OH_DISTANCE_ANGSTROM / au2angstrom
TIP3P_HOH_ANGLE = 104.52
TIP3P_CHARGES = np.array([-0.834, 0.417, 0.417])
TIP3P_LJ_EPSILON = np.array([TIP3P_LJ_EPSILON_KCAL_MOL * kcalmol2au, 0.0, 0.0])
TIP3P_LJ_SIGMA = np.array([TIP3P_LJ_SIGMA_ANGSTROM / au2angstrom, 0.0, 0.0])
TIP3P_BOND_FORCE_CONSTANT = (
    TIP3P_BOND_FORCE_CONSTANT_KCAL_MOL_ANGSTROM2 * kcalmol2au * au2angstrom**2
)
TIP3P_ANGLE_FORCE_CONSTANT = TIP3P_ANGLE_FORCE_CONSTANT_KCAL_MOL_RAD2 * kcalmol2au
TIP3P_COULOMB_CONSTANT = 1.0


def tip3p_parameters():
    """Return flexible TIP3P-style parameters in atomic units."""
    hh_distance = 2.0 * TIP3P_OH_DISTANCE * np.sin(0.5 * np.deg2rad(TIP3P_HOH_ANGLE))
    return {
        "oh_distance": TIP3P_OH_DISTANCE,
        "hh_distance": hh_distance,
        "hoh_angle": TIP3P_HOH_ANGLE,
        "charges": TIP3P_CHARGES.copy(),
        "lj_epsilon": TIP3P_LJ_EPSILON.copy(),
        "lj_sigma": TIP3P_LJ_SIGMA.copy(),
        "bond_force_constant": TIP3P_BOND_FORCE_CONSTANT,
        "angle_force_constant": TIP3P_ANGLE_FORCE_CONSTANT,
        "coulomb_constant": TIP3P_COULOMB_CONSTANT,
    }


def tip3p_water(origin=(0.0, 0.0, 0.0), **kwargs):
    """Build one flexible TIP3P-style water molecule."""
    return tip3p_waters([origin], **kwargs)


def tip3p_waters(
    origins,
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
    nonbonded_skin=1.0,
    rigid=False,
    rotations=None,
):
    """Build flexible TIP3P-style water molecules.

    Coordinates and force-field parameters are in atomic units: Bohr for
    positions, Hartree for energies, and electron charge for charges.
    """
    origins = np.asarray(origins, dtype=float)
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins must have shape (nwaters, 3).")
    if rotations is None:
        rotations = np.broadcast_to(np.eye(3), (len(origins), 3, 3))
    else:
        rotations = np.asarray(rotations, dtype=float)
        if rotations.shape != (len(origins), 3, 3):
            raise ValueError("rotations must have shape (nwaters, 3, 3).")

    theta = np.deg2rad(TIP3P_HOH_ANGLE)
    local = np.array(
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

    atoms = []
    bonds = []
    angles = []
    constraints = []
    constraint_distances = []
    charges = []
    lj_epsilon = []
    lj_sigma = []
    molecule_ids = []
    for water_index, (origin, rotation) in enumerate(zip(origins, rotations)):
        offset = 3 * water_index
        rotated = local @ rotation.T
        atoms.extend(
            [
                ["O", tuple(origin + rotated[0])],
                ["H", tuple(origin + rotated[1])],
                ["H", tuple(origin + rotated[2])],
            ]
        )
        if rigid:
            hh_distance = 2.0 * TIP3P_OH_DISTANCE * np.sin(0.5 * theta)
            constraints.extend(
                [
                    (offset, offset + 1),
                    (offset, offset + 2),
                    (offset + 1, offset + 2),
                ]
            )
            constraint_distances.extend(
                [TIP3P_OH_DISTANCE, TIP3P_OH_DISTANCE, hh_distance]
            )
        else:
            bonds.extend(
                [
                    (offset, offset + 1, TIP3P_BOND_FORCE_CONSTANT, TIP3P_OH_DISTANCE),
                    (offset, offset + 2, TIP3P_BOND_FORCE_CONSTANT, TIP3P_OH_DISTANCE),
                ]
            )
            angles.append(
                (
                    offset + 1,
                    offset,
                    offset + 2,
                    TIP3P_ANGLE_FORCE_CONSTANT,
                    TIP3P_HOH_ANGLE,
                )
            )
        charges.extend(TIP3P_CHARGES)
        lj_epsilon.extend(TIP3P_LJ_EPSILON)
        lj_sigma.extend(TIP3P_LJ_SIGMA)
        molecule_ids.extend([water_index] * 3)

    topology = Topology(
        bonds=bonds,
        angles=angles,
        charges=charges,
        lj_epsilon=lj_epsilon,
        lj_sigma=lj_sigma,
        molecule_ids=molecule_ids,
    )

    mm = None
    if calculator:
        mm = MM(
            bonds=topology.bonds,
            angles=topology.angles,
            angle_unit="degree",
            charges=topology.charges,
            coulomb_constant=coulomb_constant,
            coulomb_method=coulomb_method,
            coulomb_cutoff=coulomb_cutoff,
            ewald_alpha=ewald_alpha,
            ewald_kmax=ewald_kmax,
            pme_mesh=pme_mesh,
            nonbonded_skin=nonbonded_skin,
            lj_epsilon=topology.lj_epsilon,
            lj_sigma=topology.lj_sigma,
            lj_cutoff=lj_cutoff,
            exclude_bonded=True,
            exclude_angles=True,
        )

    constraint = None
    if rigid:
        constraint = FixBondLengths(constraints, distances=constraint_distances)

    water = Atoms(atoms, cell=cell, pbc=pbc, calculator=mm, constraint=constraint)
    water.topology = topology
    water.set_array("charges", topology.charges, float, ())
    water.set_array("lj_epsilon", topology.lj_epsilon, float, ())
    water.set_array("lj_sigma", topology.lj_sigma, float, ())
    water.set_array("molecule_ids", topology.molecule_ids, int, ())
    water.set_array("atom_names", np.tile(["O", "H1", "H2"], len(origins)), str, ())
    water.set_array("residue_names", np.full(len(water), "HOH"), str, ())
    water.set_array("residue_ids", np.repeat(np.arange(1, len(origins) + 1), 3), int, ())
    return water
