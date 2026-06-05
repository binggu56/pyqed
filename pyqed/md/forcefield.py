"""Small force-field helpers for molecule-in-solvent MD examples."""

import json
from pathlib import Path

import numpy as np

from pyqed.units import au2angstrom, kcalmol2au

from .atoms import Atoms
from .calculators import MM
from .topology import Topology


def load_forcefield(filename):
    """Load a JSON force-field/solute description."""
    with open(filename) as handle:
        return json.load(handle)


def solute_from_parameters(parameters, calculator=True, **calculator_kwargs):
    """Build a parametrized solute from a JSON-like dictionary.

    The compact schema is intentionally explicit: each atom gives a symbol,
    coordinate, charge, and/or type.  Units default to atomic units, but
    ``{"units": {"length": "angstrom", "energy": "kcal/mol", "angle":
    "degree"}}`` is supported for human-readable example files.
    """
    if isinstance(parameters, (str, Path)):
        parameters = load_forcefield(parameters)

    units = parameters.get("units", {})
    length_unit = units.get("length", parameters.get("unit", "bohr"))
    energy_unit = units.get("energy", "hartree")
    angle_unit = units.get("angle", "degree")
    atom_types = parameters.get("atom_types", {})

    symbols = []
    positions = []
    names = []
    types = []
    charges = []
    lj_epsilon = []
    lj_sigma = []
    for index, atom in enumerate(parameters["atoms"]):
        atom_type = atom.get("type")
        type_parameters = atom_types.get(atom_type, {})
        symbols.append(atom["symbol"])
        positions.append(_length_array(atom["position"], length_unit))
        names.append(atom.get("name", f"{atom['symbol']}{index + 1}"))
        types.append("" if atom_type is None else atom_type)
        charges.append(float(atom.get("charge", type_parameters.get("charge", 0.0))))
        lj_epsilon.append(_energy(atom.get("lj_epsilon", type_parameters.get("lj_epsilon", 0.0)), energy_unit))
        lj_sigma.append(_length(atom.get("lj_sigma", type_parameters.get("lj_sigma", 0.0)), length_unit))

    topology = Topology(
        bonds=[
            (int(i), int(j), _bond_force(k, length_unit, energy_unit), _length(r0, length_unit))
            for i, j, k, r0 in parameters.get("bonds", [])
        ],
        angles=[
            (int(i), int(j), int(k), _energy(ktheta, energy_unit), _angle(theta0, angle_unit))
            for i, j, k, ktheta, theta0 in parameters.get("angles", [])
        ],
        torsions=[
            (
                int(i),
                int(j),
                int(k),
                int(l),
                _energy(barrier, energy_unit),
                int(periodicity),
                _angle(phase, angle_unit),
            )
            for i, j, k, l, barrier, periodicity, phase in parameters.get("torsions", [])
        ],
        charges=charges,
        lj_epsilon=lj_epsilon,
        lj_sigma=lj_sigma,
        molecule_ids=np.zeros(len(symbols), dtype=int),
    )

    atoms = Atoms([[symbol, tuple(xyz)] for symbol, xyz in zip(symbols, positions)])
    atoms.topology = topology
    atoms.set_array("charges", topology.charges, float, ())
    atoms.set_array("lj_epsilon", topology.lj_epsilon, float, ())
    atoms.set_array("lj_sigma", topology.lj_sigma, float, ())
    atoms.set_array("molecule_ids", topology.molecule_ids, int, ())
    atoms.set_array("atom_names", np.asarray(names), str, ())
    atoms.set_array("atom_types", np.asarray(types), str, ())
    if calculator:
        atoms.calc = MM(
            bonds=topology.bonds,
            angles=topology.angles,
            torsions=topology.torsions,
            angle_unit="degree",
            torsion_unit="degree",
            charges=topology.charges,
            lj_epsilon=topology.lj_epsilon,
            lj_sigma=topology.lj_sigma,
            **calculator_kwargs,
        )
    return atoms


def mm_from_topology(topology, **kwargs):
    """Build an ``MM`` calculator from topology arrays."""
    return MM(
        bonds=topology.bonds,
        angles=topology.angles,
        torsions=getattr(topology, "torsions", []),
        angle_unit="degree",
        torsion_unit="degree",
        charges=topology.charges,
        lj_epsilon=topology.lj_epsilon,
        lj_sigma=topology.lj_sigma,
        exclude_bonded=True,
        exclude_angles=True,
        nonbonded_exclusions=getattr(topology, "nonbonded_exclusions", None),
        lj_exclusions=getattr(topology, "lj_exclusions", None),
        coulomb_exclusions=getattr(topology, "coulomb_exclusions", None),
        lj_pair_parameters=getattr(topology, "lj_pair_parameters", None),
        lj_pair_scales=getattr(topology, "lj_pair_scales", None),
        coulomb_pair_scales=getattr(topology, "coulomb_pair_scales", None),
        **kwargs,
    )


def _length(value, unit):
    value = float(value)
    if unit.lower() in {"bohr", "b", "au", "atomic"}:
        return value
    if unit.lower() in {"angstrom", "ang", "a"}:
        return value / au2angstrom
    raise ValueError("length unit must be 'bohr' or 'angstrom'.")


def _length_array(values, unit):
    return np.asarray([_length(value, unit) for value in values], dtype=float)


def _energy(value, unit):
    value = float(value)
    if unit.lower() in {"hartree", "ha", "au", "atomic"}:
        return value
    if unit.lower() in {"kcal/mol", "kcalmol", "kcal_mol"}:
        return value * kcalmol2au
    raise ValueError("energy unit must be 'hartree' or 'kcal/mol'.")


def _bond_force(value, length_unit, energy_unit):
    value = _energy(value, energy_unit)
    if length_unit.lower() in {"bohr", "b", "au", "atomic"}:
        return value
    if length_unit.lower() in {"angstrom", "ang", "a"}:
        return value * au2angstrom**2
    raise ValueError("length unit must be 'bohr' or 'angstrom'.")


def _angle(value, unit):
    value = float(value)
    if unit.lower() in {"degree", "degrees", "deg"}:
        return value
    if unit.lower() in {"radian", "radians", "rad"}:
        return float(np.rad2deg(value))
    raise ValueError("angle unit must be 'degree' or 'radian'.")
