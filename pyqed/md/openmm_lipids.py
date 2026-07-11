"""OpenMM lipid force-field template extraction.

The helpers in this module read OpenMM force-field XML files directly.  They
do not import OpenMM, so the parser remains usable in lightweight PyQED
installations whenever the XML files are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from pyqed.units import au2angstrom, kcalmol2au

from .topology import Topology


KJMOL_TO_AU = kcalmol2au / 4.184


@dataclass(frozen=True)
class OpenMMLipidTemplate:
    """A single residue template extracted from an OpenMM lipid XML file."""

    residue_name: str
    source: str
    atom_names: tuple[str, ...]
    atom_types: tuple[str, ...]
    elements: tuple[str, ...]
    masses_amu: tuple[float, ...]
    charges: tuple[float, ...]
    lj_epsilon: tuple[float, ...]
    lj_sigma: tuple[float, ...]
    bonds: tuple[tuple[int, int, float, float], ...]
    angles: tuple[tuple[int, int, int, float, float], ...]
    torsions: tuple[tuple[int, int, int, int, float, int, float], ...]
    coulomb14scale: float = 1.0
    lj14scale: float = 1.0
    validated: bool = True

    @property
    def name(self):
        return self.residue_name

    @property
    def natoms(self):
        return len(self.atom_names)

    @property
    def net_charge(self):
        return float(np.sum(self.charges))

    def topology(self, molecule_id=0):
        """Return a PyQED :class:`Topology` for one molecule of this residue."""
        one_four_pairs = _one_four_pairs_from_torsions(self.torsions)
        return Topology(
            bonds=self.bonds,
            angles=self.angles,
            torsions=self.torsions,
            charges=self.charges,
            lj_epsilon=self.lj_epsilon,
            lj_sigma=self.lj_sigma,
            molecule_ids=np.full(self.natoms, int(molecule_id), dtype=int),
            masses_amu=self.masses_amu,
            atom_types=self.atom_types,
            atom_names=self.atom_names,
            lj_pair_scales={pair: self.lj14scale for pair in one_four_pairs},
            coulomb_pair_scales={pair: self.coulomb14scale for pair in one_four_pairs},
        )


def openmm_lipid_template(residue_name="DPPC", source=None):
    """Extract one lipid residue template from an OpenMM force-field XML file.

    Parameters
    ----------
    residue_name
        Residue name such as ``"DPPC"`` or ``"POPC"``.
    source
        Optional XML path.  When omitted, the installed OpenMM
        ``amber14/lipid17.xml`` file is used if it can be found.
    """
    source_path = _resolve_source(source)
    root = ET.parse(source_path).getroot()
    residue_name = str(residue_name).upper()
    residue = root.find(f"./Residues/Residue[@name='{residue_name}']")
    if residue is None:
        available = ", ".join(available_openmm_lipid_templates(source_path)[:8])
        raise ValueError(f"OpenMM lipid residue {residue_name!r} not found. Available examples: {available}")

    type_records = _atom_type_records(root)
    nonbonded = _nonbonded_records(root)
    atoms = list(residue.findall("Atom"))
    atom_names = tuple(atom.attrib["name"] for atom in atoms)
    atom_types = tuple(atom.attrib["type"] for atom in atoms)
    elements = tuple(type_records[atom_type]["element"] for atom_type in atom_types)
    masses = tuple(float(type_records[atom_type]["mass"]) for atom_type in atom_types)
    charges = tuple(float(atom.attrib.get("charge", 0.0)) for atom in atoms)
    lj_epsilon = tuple(_energy_au(nonbonded[atom_type]["epsilon"]) for atom_type in atom_types)
    lj_sigma = tuple(_length_bohr(nonbonded[atom_type]["sigma"]) for atom_type in atom_types)

    name_to_index = {name: index for index, name in enumerate(atom_names)}
    bond_edges = _residue_bonds(residue, name_to_index)
    bond_params = _bond_parameters(root)
    angle_params = _angle_parameters(root)
    torsion_params = _torsion_parameters(root)

    bonds = tuple(
        (
            i,
            j,
            _bond_force_au(bond_params[_lookup_key((atom_types[i], atom_types[j]), bond_params)]["k"]),
            _length_bohr(bond_params[_lookup_key((atom_types[i], atom_types[j]), bond_params)]["length"]),
        )
        for i, j in bond_edges
    )

    angles = tuple(
        (
            i,
            j,
            k,
            _angle_force_au(angle_params[_lookup_key((atom_types[i], atom_types[j], atom_types[k]), angle_params)]["k"]),
            _angle_degree(angle_params[_lookup_key((atom_types[i], atom_types[j], atom_types[k]), angle_params)]["angle"]),
        )
        for i, j, k in _angles_from_edges(bond_edges)
    )

    torsions = []
    for i, j, k, l in _torsions_from_edges(bond_edges):
        record = torsion_params[_lookup_key((atom_types[i], atom_types[j], atom_types[k], atom_types[l]), torsion_params)]
        torsions.extend(_periodic_terms(record, i, j, k, l))

    nonbonded_force = root.find("NonbondedForce")
    return OpenMMLipidTemplate(
        residue_name=residue_name,
        source=str(source_path),
        atom_names=atom_names,
        atom_types=atom_types,
        elements=elements,
        masses_amu=masses,
        charges=charges,
        lj_epsilon=lj_epsilon,
        lj_sigma=lj_sigma,
        bonds=bonds,
        angles=angles,
        torsions=tuple(torsions),
        coulomb14scale=float(nonbonded_force.attrib.get("coulomb14scale", 1.0)),
        lj14scale=float(nonbonded_force.attrib.get("lj14scale", 1.0)),
    )


def available_openmm_lipid_templates(source=None):
    """Return residue names available in an OpenMM lipid XML file."""
    source_path = _resolve_source(source)
    root = ET.parse(source_path).getroot()
    residues = root.find("Residues")
    if residues is None:
        return []
    return [residue.attrib["name"] for residue in residues.findall("Residue")]


def find_openmm_lipid_xml():
    """Return the installed OpenMM Amber lipid XML path.

    Raises
    ------
    FileNotFoundError
        If the OpenMM package or its Amber lipid XML file is unavailable.
    """
    spec = find_spec("openmm.app")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("OpenMM is not installed; amber14/lipid17.xml is unavailable.")
    candidate = Path(spec.origin).resolve().parent / "data" / "amber14" / "lipid17.xml"
    if not candidate.exists():
        raise FileNotFoundError(f"OpenMM lipid XML not found at {candidate}.")
    return candidate


def _resolve_source(source):
    if source is None:
        return find_openmm_lipid_xml()
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"OpenMM lipid XML not found: {path}")
    return path


def _atom_type_records(root):
    records = {}
    for record in root.find("AtomTypes").findall("Type"):
        records[record.attrib["name"]] = record.attrib
    return records


def _nonbonded_records(root):
    records = {}
    for record in root.find("NonbondedForce").findall("Atom"):
        records[record.attrib["type"]] = record.attrib
    return records


def _bond_parameters(root):
    records = {}
    for record in root.find("HarmonicBondForce").findall("Bond"):
        key = (record.attrib["type1"], record.attrib["type2"])
        records[key] = record.attrib
        records[tuple(reversed(key))] = record.attrib
    return records


def _angle_parameters(root):
    records = {}
    for record in root.find("HarmonicAngleForce").findall("Angle"):
        key = (record.attrib["type1"], record.attrib["type2"], record.attrib["type3"])
        records[key] = record.attrib
        records[tuple(reversed(key))] = record.attrib
    return records


def _torsion_parameters(root):
    records = {}
    force = root.find("PeriodicTorsionForce")
    if force is None:
        return records
    for record in force.findall("Proper"):
        key = tuple(record.attrib[f"type{index}"] for index in range(1, 5))
        records[key] = record.attrib
        records[tuple(reversed(key))] = record.attrib
    return records


def _residue_bonds(residue, name_to_index):
    bonds = []
    for record in residue.findall("Bond"):
        if "atomName1" in record.attrib:
            i = name_to_index[record.attrib["atomName1"]]
            j = name_to_index[record.attrib["atomName2"]]
        else:
            i = int(record.attrib["from"])
            j = int(record.attrib["to"])
        bonds.append(tuple(sorted((i, j))))
    return tuple(sorted(set(bonds)))


def _angles_from_edges(edges):
    neighbors = _neighbors(edges)
    angles = []
    for center, bonded in neighbors.items():
        bonded = sorted(bonded)
        for left_index, left in enumerate(bonded):
            for right in bonded[left_index + 1 :]:
                angles.append((left, center, right))
    return tuple(sorted(angles))


def _torsions_from_edges(edges):
    neighbors = _neighbors(edges)
    torsions = set()
    for j, k in edges:
        for i in neighbors[j] - {k}:
            for l in neighbors[k] - {j}:
                if i == l:
                    continue
                torsion = (i, j, k, l)
                torsions.add(min(torsion, tuple(reversed(torsion))))
    return tuple(sorted(torsions))


def _neighbors(edges):
    neighbors = {}
    for i, j in edges:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)
    return neighbors


def _lookup_key(key, records):
    if key in records:
        return key
    reverse = tuple(reversed(key))
    if reverse in records:
        return reverse
    raise ValueError(f"No OpenMM lipid parameter found for atom-type tuple {key!r}.")


def _periodic_terms(record, i, j, k, l):
    terms = []
    term_index = 1
    while f"k{term_index}" in record:
        terms.append(
            (
                i,
                j,
                k,
                l,
                _energy_au(record[f"k{term_index}"]),
                int(record[f"periodicity{term_index}"]),
                _angle_degree(record[f"phase{term_index}"]),
            )
        )
        term_index += 1
    return terms


def _one_four_pairs_from_torsions(torsions):
    return {tuple(sorted((int(i), int(l)))) for i, _j, _k, l, *_rest in torsions}


def _energy_au(value):
    return float(value) * KJMOL_TO_AU


def _length_bohr(value):
    return float(value) * 10.0 / au2angstrom


def _bond_force_au(value):
    return float(value) * KJMOL_TO_AU * (au2angstrom / 10.0) ** 2


def _angle_force_au(value):
    return _energy_au(value)


def _angle_degree(value):
    return float(np.rad2deg(float(value)))
