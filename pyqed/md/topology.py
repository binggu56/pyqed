"""Topology metadata for simple classical MD systems."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Topology:
    """Minimal topology for :class:`pyqed.md.MolecularMechanics`."""

    bonds: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    torsions: list = field(default_factory=list)
    charges: object = None
    lj_epsilon: object = None
    lj_sigma: object = None
    molecule_ids: object = None

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
        self.charges = None if self.charges is None else np.asarray(self.charges, dtype=float)
        self.lj_epsilon = None if self.lj_epsilon is None else np.asarray(self.lj_epsilon, dtype=float)
        self.lj_sigma = None if self.lj_sigma is None else np.asarray(self.lj_sigma, dtype=float)
        self.molecule_ids = (
            None if self.molecule_ids is None else np.asarray(self.molecule_ids, dtype=int)
        )

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
            charges=None if self.charges is None else self.charges.copy(),
            lj_epsilon=None if self.lj_epsilon is None else self.lj_epsilon.copy(),
            lj_sigma=None if self.lj_sigma is None else self.lj_sigma.copy(),
            molecule_ids=molecule_ids,
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
            charges=arrays.get("charges"),
            lj_epsilon=arrays.get("lj_epsilon"),
            lj_sigma=arrays.get("lj_sigma"),
            molecule_ids=arrays.get("molecule_ids"),
        )


def combine_topologies(topologies):
    """Combine already shifted topologies."""
    bonds = []
    angles = []
    torsions = []
    charges = []
    lj_epsilon = []
    lj_sigma = []
    molecule_ids = []
    have_charges = have_lj_epsilon = have_lj_sigma = have_molecule_ids = False

    for topology in topologies:
        bonds.extend(topology.bonds)
        angles.extend(topology.angles)
        torsions.extend(getattr(topology, "torsions", []))
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

    return Topology(
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        charges=charges if have_charges else None,
        lj_epsilon=lj_epsilon if have_lj_epsilon else None,
        lj_sigma=lj_sigma if have_lj_sigma else None,
        molecule_ids=molecule_ids if have_molecule_ids else None,
    )
