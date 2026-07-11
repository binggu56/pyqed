"""Restart/checkpoint I/O for :mod:`pyqed.md`."""

import json

import numpy as np

from .atoms import Atoms
from .constraints import FixBondLengths
from .topology import Topology


def write_restart(atoms, filename, step=0, time=0.0, metadata=None):
    """Write an ``Atoms`` checkpoint as a portable ``.npz`` file."""
    topology = Topology.from_atoms(atoms)
    data = {
        "symbols": np.asarray(atoms.atom_symbols(), dtype=str),
        "positions": atoms.get_positions(),
        "momenta": atoms.get_momenta(),
        "cell": np.asarray(atoms.get_cell(), dtype=float),
        "pbc": atoms.get_pbc(),
        "bonds": _array_or_empty(topology.bonds, (0, 4), float),
        "angles": _array_or_empty(topology.angles, (0, 5), float),
        "torsions": _array_or_empty(topology.torsions, (0, 7), float),
        "charges": _optional_array(topology.charges),
        "lj_epsilon": _optional_array(topology.lj_epsilon),
        "lj_sigma": _optional_array(topology.lj_sigma),
        "molecule_ids": _optional_array(topology.molecule_ids, dtype=int),
        "constraints": _constraint_payload(atoms),
        "metadata": json.dumps({"step": int(step), "time": float(time), **(metadata or {})}),
    }
    np.savez(filename, **data)


def read_restart(filename, calculator=None):
    """Read a restart file and return ``(atoms, metadata)``."""
    with np.load(filename, allow_pickle=False) as data:
        symbols = data["symbols"].astype(str).tolist()
        positions = data["positions"]
        atoms = Atoms(
            [[symbol, tuple(xyz)] for symbol, xyz in zip(symbols, positions)],
            cell=data["cell"],
            pbc=data["pbc"],
            calculator=calculator,
        )
        atoms.set_momenta(data["momenta"], apply_constraint=False)

        topology = Topology(
            bonds=_records(data["bonds"], int_fields=(0, 1)),
            angles=_records(data["angles"], int_fields=(0, 1, 2)),
            torsions=_records(data["torsions"], int_fields=(0, 1, 2, 3, 5)),
            charges=_none_if_empty(data["charges"]),
            lj_epsilon=_none_if_empty(data["lj_epsilon"]),
            lj_sigma=_none_if_empty(data["lj_sigma"]),
            molecule_ids=_none_if_empty(data["molecule_ids"], dtype=int),
        )
        atoms.topology = topology
        _set_optional_array(atoms, "charges", topology.charges, float)
        _set_optional_array(atoms, "lj_epsilon", topology.lj_epsilon, float)
        _set_optional_array(atoms, "lj_sigma", topology.lj_sigma, float)
        _set_optional_array(atoms, "molecule_ids", topology.molecule_ids, int)

        constraints = data["constraints"]
        if constraints.size:
            pairs = constraints[:, :2].astype(int)
            distances = constraints[:, 2].astype(float)
            atoms.constraints = [FixBondLengths(pairs, distances=distances)]

        metadata = json.loads(str(data["metadata"]))
    return atoms, metadata


def _array_or_empty(records, shape, dtype):
    if not records:
        return np.empty(shape, dtype=dtype)
    return np.asarray(records, dtype=dtype)


def _optional_array(array, dtype=float):
    if array is None:
        return np.empty((0,), dtype=dtype)
    return np.asarray(array, dtype=dtype)


def _none_if_empty(array, dtype=float):
    if array.size == 0:
        return None
    return np.asarray(array, dtype=dtype)


def _records(array, int_fields=()):
    records = []
    for row in np.asarray(array):
        item = []
        for index, value in enumerate(row):
            item.append(int(value) if index in int_fields else float(value))
        records.append(tuple(item))
    return records


def _set_optional_array(atoms, name, array, dtype):
    if array is not None:
        atoms.set_array(name, array, dtype, ())


def _constraint_payload(atoms):
    rows = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixBondLengths):
            distances = constraint._targets(atoms)
            for (i, j), distance in zip(constraint.pairs, distances):
                rows.append((i, j, distance))
    if not rows:
        return np.empty((0, 3), dtype=float)
    return np.asarray(rows, dtype=float)
