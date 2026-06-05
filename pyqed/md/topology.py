"""Topology metadata for simple classical MD systems."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Topology:
    """Minimal topology for :class:`pyqed.md.MolecularMechanics`."""

    bonds: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    torsions: list = field(default_factory=list)
    impropers: list = field(default_factory=list)
    cmaps: list = field(default_factory=list)
    cmap_grids: list = field(default_factory=list)
    charges: object = None
    lj_epsilon: object = None
    lj_sigma: object = None
    molecule_ids: object = None
    masses_amu: object = None
    atom_types: object = None
    atom_names: object = None
    lj_pair_overrides: object = None
    lj_pair_parameters: object = None
    coulomb_pair_parameters: object = None
    nonbonded_exclusions: object = None
    lj_exclusions: object = None
    coulomb_exclusions: object = None
    lj_pair_scales: object = None
    coulomb_pair_scales: object = None

    def __post_init__(self):
        self.bonds = [
            (int(i), int(j), float(k), float(r0))
            for i, j, k, r0 in self.bonds
        ]
        self.angles = [
            (int(i), int(j), int(k), float(ktheta), float(theta0))
            for i, j, k, ktheta, theta0 in self.angles
        ]
        self.torsions = [
            (
                int(i),
                int(j),
                int(k),
                int(l),
                float(barrier),
                int(periodicity),
                float(phase),
            )
            for i, j, k, l, barrier, periodicity, phase in self.torsions
        ]
        self.impropers = [
            (int(i), int(j), int(k), int(l), float(force_constant), float(phase))
            for i, j, k, l, force_constant, phase in self.impropers
        ]
        self.cmaps = [
            (int(map_index), tuple(int(atom) for atom in atoms))
            for map_index, atoms in self.cmaps
        ]
        self.cmap_grids = [
            (int(size), np.asarray(values, dtype=float).reshape(int(size), int(size)))
            for size, values in self.cmap_grids
        ]
        self.charges = None if self.charges is None else np.asarray(self.charges, dtype=float)
        self.lj_epsilon = None if self.lj_epsilon is None else np.asarray(self.lj_epsilon, dtype=float)
        self.lj_sigma = None if self.lj_sigma is None else np.asarray(self.lj_sigma, dtype=float)
        self.molecule_ids = (
            None if self.molecule_ids is None else np.asarray(self.molecule_ids, dtype=int)
        )
        self.masses_amu = None if self.masses_amu is None else np.asarray(self.masses_amu, dtype=float)
        self.atom_types = None if self.atom_types is None else np.asarray(self.atom_types, dtype=str)
        self.atom_names = None if self.atom_names is None else np.asarray(self.atom_names, dtype=str)
        self.lj_pair_overrides = _pair_override_dict(self.lj_pair_overrides)
        self.lj_pair_parameters = _pair_parameter_dict(self.lj_pair_parameters)
        self.coulomb_pair_parameters = _pair_float_dict(self.coulomb_pair_parameters)
        self.nonbonded_exclusions = _pair_set(self.nonbonded_exclusions)
        self.lj_exclusions = _pair_set(self.lj_exclusions)
        self.coulomb_exclusions = _pair_set(self.coulomb_exclusions)
        self.lj_pair_scales = _pair_scale_dict(self.lj_pair_scales)
        self.coulomb_pair_scales = _pair_scale_dict(self.coulomb_pair_scales)

    def shifted(self, atom_offset=0, molecule_offset=0):
        molecule_ids = None
        if self.molecule_ids is not None:
            molecule_ids = self.molecule_ids + molecule_offset
        return Topology(
            bonds=[
                (i + atom_offset, j + atom_offset, k, r0)
                for i, j, k, r0 in self.bonds
            ],
            angles=[
                (i + atom_offset, j + atom_offset, k + atom_offset, ktheta, theta0)
                for i, j, k, ktheta, theta0 in self.angles
            ],
            torsions=[
                (
                    i + atom_offset,
                    j + atom_offset,
                    k + atom_offset,
                    l + atom_offset,
                    barrier,
                    periodicity,
                    phase,
                )
                for i, j, k, l, barrier, periodicity, phase in self.torsions
            ],
            impropers=[
                (i + atom_offset, j + atom_offset, k + atom_offset, l + atom_offset, force_constant, phase)
                for i, j, k, l, force_constant, phase in self.impropers
            ],
            cmaps=[
                (map_index, tuple(atom + atom_offset for atom in atoms))
                for map_index, atoms in self.cmaps
            ],
            cmap_grids=[
                (size, values.copy())
                for size, values in self.cmap_grids
            ],
            charges=None if self.charges is None else self.charges.copy(),
            lj_epsilon=None if self.lj_epsilon is None else self.lj_epsilon.copy(),
            lj_sigma=None if self.lj_sigma is None else self.lj_sigma.copy(),
            molecule_ids=molecule_ids,
            masses_amu=None if self.masses_amu is None else self.masses_amu.copy(),
            atom_types=None if self.atom_types is None else self.atom_types.copy(),
            atom_names=None if self.atom_names is None else self.atom_names.copy(),
            lj_pair_overrides=self.lj_pair_overrides.copy(),
            lj_pair_parameters=_shift_pair_parameters(self.lj_pair_parameters, atom_offset),
            coulomb_pair_parameters=_shift_pair_floats(self.coulomb_pair_parameters, atom_offset),
            nonbonded_exclusions=_shift_pairs(self.nonbonded_exclusions, atom_offset),
            lj_exclusions=_shift_pairs(self.lj_exclusions, atom_offset),
            coulomb_exclusions=_shift_pairs(self.coulomb_exclusions, atom_offset),
            lj_pair_scales=_shift_pair_scales(self.lj_pair_scales, atom_offset),
            coulomb_pair_scales=_shift_pair_scales(self.coulomb_pair_scales, atom_offset),
        )

    @classmethod
    def from_atoms(cls, atoms):
        topology = getattr(atoms, "topology", None)
        if topology is not None:
            return topology
        arrays = getattr(atoms, "arrays", {})
        return cls(
            bonds=getattr(topology, "bonds", []),
            angles=getattr(topology, "angles", []),
            torsions=getattr(topology, "torsions", []),
            impropers=getattr(topology, "impropers", []),
            cmaps=getattr(topology, "cmaps", []),
            cmap_grids=getattr(topology, "cmap_grids", []),
            charges=arrays.get("charges"),
            lj_epsilon=arrays.get("lj_epsilon"),
            lj_sigma=arrays.get("lj_sigma"),
            molecule_ids=arrays.get("molecule_ids"),
            masses_amu=arrays.get("masses_amu"),
            atom_types=arrays.get("atom_types"),
            atom_names=arrays.get("atom_names"),
            lj_pair_overrides=getattr(topology, "lj_pair_overrides", None),
            lj_pair_parameters=getattr(topology, "lj_pair_parameters", None),
            coulomb_pair_parameters=getattr(topology, "coulomb_pair_parameters", None),
            nonbonded_exclusions=getattr(topology, "nonbonded_exclusions", None),
            lj_exclusions=getattr(topology, "lj_exclusions", None),
            coulomb_exclusions=getattr(topology, "coulomb_exclusions", None),
            lj_pair_scales=getattr(topology, "lj_pair_scales", None),
            coulomb_pair_scales=getattr(topology, "coulomb_pair_scales", None),
        )


def combine_topologies(topologies):
    """Combine already shifted topologies."""
    bonds = []
    angles = []
    torsions = []
    impropers = []
    cmaps = []
    cmap_grids = []
    charges = []
    lj_epsilon = []
    lj_sigma = []
    molecule_ids = []
    masses_amu = []
    atom_types = []
    atom_names = []
    lj_pair_overrides = {}
    lj_pair_parameters = {}
    coulomb_pair_parameters = {}
    nonbonded_exclusions = set()
    lj_exclusions = set()
    coulomb_exclusions = set()
    lj_pair_scales = {}
    coulomb_pair_scales = {}
    have_charges = have_lj_epsilon = have_lj_sigma = have_molecule_ids = False
    have_masses = have_atom_types = have_atom_names = False

    for topology in topologies:
        natoms = _topology_size(topology)
        bonds.extend(topology.bonds)
        angles.extend(topology.angles)
        torsions.extend(getattr(topology, "torsions", []))
        impropers.extend(getattr(topology, "impropers", []))
        cmap_offset = len(cmap_grids)
        cmap_grids.extend(getattr(topology, "cmap_grids", []))
        cmaps.extend(
            (map_index + cmap_offset, atoms)
            for map_index, atoms in getattr(topology, "cmaps", [])
        )
        if topology.charges is not None:
            have_charges = True
            charges.extend(topology.charges)
        if topology.lj_epsilon is not None:
            have_lj_epsilon = True
            lj_epsilon.extend(topology.lj_epsilon)
        if topology.lj_sigma is not None:
            have_lj_sigma = True
            lj_sigma.extend(topology.lj_sigma)
        if topology.molecule_ids is not None:
            have_molecule_ids = True
            molecule_ids.extend(topology.molecule_ids)
        if getattr(topology, "masses_amu", None) is not None:
            have_masses = True
            masses_amu.extend(topology.masses_amu)
        else:
            masses_amu.extend([0.0] * natoms)
        if getattr(topology, "atom_types", None) is not None:
            have_atom_types = True
            atom_types.extend(topology.atom_types)
        else:
            atom_types.extend([""] * natoms)
        if getattr(topology, "atom_names", None) is not None:
            have_atom_names = True
            atom_names.extend(topology.atom_names)
        else:
            atom_names.extend([""] * natoms)
        lj_pair_overrides.update(getattr(topology, "lj_pair_overrides", {}))
        lj_pair_parameters.update(getattr(topology, "lj_pair_parameters", {}))
        coulomb_pair_parameters.update(getattr(topology, "coulomb_pair_parameters", {}))
        nonbonded_exclusions.update(getattr(topology, "nonbonded_exclusions", set()))
        lj_exclusions.update(getattr(topology, "lj_exclusions", set()))
        coulomb_exclusions.update(getattr(topology, "coulomb_exclusions", set()))
        lj_pair_scales.update(getattr(topology, "lj_pair_scales", {}))
        coulomb_pair_scales.update(getattr(topology, "coulomb_pair_scales", {}))

    return Topology(
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        impropers=impropers,
        cmaps=cmaps,
        cmap_grids=cmap_grids,
        charges=charges if have_charges else None,
        lj_epsilon=lj_epsilon if have_lj_epsilon else None,
        lj_sigma=lj_sigma if have_lj_sigma else None,
        molecule_ids=molecule_ids if have_molecule_ids else None,
        masses_amu=masses_amu if have_masses else None,
        atom_types=atom_types if have_atom_types else None,
        atom_names=atom_names if have_atom_names else None,
        lj_pair_overrides=lj_pair_overrides,
        lj_pair_parameters=lj_pair_parameters,
        coulomb_pair_parameters=coulomb_pair_parameters,
        nonbonded_exclusions=nonbonded_exclusions,
        lj_exclusions=lj_exclusions,
        coulomb_exclusions=coulomb_exclusions,
        lj_pair_scales=lj_pair_scales,
        coulomb_pair_scales=coulomb_pair_scales,
    )


def _pair_override_dict(overrides):
    if overrides is None:
        return {}
    if hasattr(overrides, "items"):
        items = overrides.items()
    else:
        items = overrides
    return {
        tuple(sorted((str(pair[0]), str(pair[1])))): (float(values[0]), float(values[1]))
        for pair, values in items
    }


def _pair_parameter_dict(parameters):
    if parameters is None:
        return {}
    if hasattr(parameters, "items"):
        items = parameters.items()
    else:
        items = parameters
    return {
        tuple(sorted((int(pair[0]), int(pair[1])))): (float(values[0]), float(values[1]))
        for pair, values in items
    }


def _pair_set(pairs):
    if pairs is None:
        return set()
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def _pair_scale_dict(pair_scales):
    if pair_scales is None:
        return {}
    if hasattr(pair_scales, "items"):
        items = pair_scales.items()
    else:
        items = pair_scales
    return {
        tuple(sorted((int(pair[0]), int(pair[1])))): float(scale)
        for pair, scale in items
    }


def _pair_float_dict(pair_values):
    if pair_values is None:
        return {}
    if hasattr(pair_values, "items"):
        items = pair_values.items()
    else:
        items = pair_values
    return {
        tuple(sorted((int(pair[0]), int(pair[1])))): float(value)
        for pair, value in items
    }


def _shift_pairs(pairs, atom_offset):
    return {
        tuple(sorted((int(i) + atom_offset, int(j) + atom_offset)))
        for i, j in pairs
    }


def _shift_pair_scales(pair_scales, atom_offset):
    return {
        tuple(sorted((int(i) + atom_offset, int(j) + atom_offset))): scale
        for (i, j), scale in pair_scales.items()
    }


def _shift_pair_floats(pair_values, atom_offset):
    return {
        tuple(sorted((int(i) + atom_offset, int(j) + atom_offset))): value
        for (i, j), value in pair_values.items()
    }


def _shift_pair_parameters(pair_parameters, atom_offset):
    return {
        tuple(sorted((int(i) + atom_offset, int(j) + atom_offset))): values
        for (i, j), values in pair_parameters.items()
    }


def _topology_size(topology):
    for name in ("charges", "lj_epsilon", "lj_sigma", "molecule_ids", "masses_amu", "atom_types", "atom_names"):
        values = getattr(topology, name, None)
        if values is not None:
            return len(values)
    max_index = -1
    for terms in (topology.bonds, topology.angles, getattr(topology, "torsions", []), getattr(topology, "impropers", [])):
        for term in terms:
            max_index = max(max_index, *(int(index) for index in term[:4] if isinstance(index, (int, np.integer))))
    for _map_index, atoms in getattr(topology, "cmaps", []):
        max_index = max(max_index, *(int(index) for index in atoms))
    return max_index + 1
