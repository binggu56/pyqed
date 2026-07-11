"""Ion builders and placement helpers for MD boxes."""

import numpy as np

from pyqed.units import au2angstrom, kcalmol2au

from .atoms import Atoms
from .solvation import combine_systems
from .topology import Topology


ION_PARAMETERS = {
    "Na": {"charge": 1.0, "epsilon": 0.0469 * kcalmol2au, "sigma": 2.43 / au2angstrom},
    "Cl": {"charge": -1.0, "epsilon": 0.1500 * kcalmol2au, "sigma": 4.40 / au2angstrom},
    "K": {"charge": 1.0, "epsilon": 0.0870 * kcalmol2au, "sigma": 3.33 / au2angstrom},
}


def monatomic_ions(symbols, positions, start_molecule_id=0):
    """Build monatomic ions with compact CHARMM-like parameters."""
    positions = np.asarray(positions, dtype=float)
    if len(symbols) != len(positions):
        raise ValueError("symbols and positions must have the same length.")
    charges = []
    epsilon = []
    sigma = []
    atom_types = []
    for symbol in symbols:
        if symbol not in ION_PARAMETERS:
            raise ValueError(f"unsupported ion {symbol!r}.")
        params = ION_PARAMETERS[symbol]
        charges.append(params["charge"])
        epsilon.append(params["epsilon"])
        sigma.append(params["sigma"])
        atom_types.append(symbol.upper())
    topology = Topology(
        charges=charges,
        lj_epsilon=epsilon,
        lj_sigma=sigma,
        molecule_ids=np.arange(start_molecule_id, start_molecule_id + len(symbols)),
        atom_types=atom_types,
        atom_names=symbols,
    )
    ions = Atoms([[symbol, tuple(xyz)] for symbol, xyz in zip(symbols, positions)])
    ions.topology = topology
    ions.set_array("charges", topology.charges, float, ())
    ions.set_array("lj_epsilon", topology.lj_epsilon, float, ())
    ions.set_array("lj_sigma", topology.lj_sigma, float, ())
    ions.set_array("molecule_ids", topology.molecule_ids, int, ())
    ions.set_array("atom_types", topology.atom_types, str, ())
    ions.set_array("atom_names", topology.atom_names, str, ())
    return ions


def add_ions_random(
    atoms,
    ions=("Na", "Cl"),
    min_distance=2.5 / au2angstrom,
    seed=None,
    max_attempts=10000,
    calculator=True,
    **calculator_kwargs,
):
    """Place monatomic ions randomly into an existing periodic box."""
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    if np.any(lengths <= 0.0):
        raise ValueError("ion placement requires a finite orthorhombic box.")
    rng = np.random.default_rng(seed)
    existing = atoms.get_positions()
    positions = []
    min_distance2 = float(min_distance) ** 2
    attempts = 0
    while len(positions) < len(ions) and attempts < max_attempts:
        attempts += 1
        trial = rng.uniform(0.0, lengths)
        if existing.size:
            deltas = trial - existing
            if np.any(np.sum(deltas * deltas, axis=1) < min_distance2):
                continue
        if positions:
            deltas = trial - np.asarray(positions)
            if np.any(np.sum(deltas * deltas, axis=1) < min_distance2):
                continue
        positions.append(trial)
    if len(positions) < len(ions):
        raise RuntimeError("could not place all ions; try fewer ions or a smaller min_distance.")
    start_molecule_id = 0
    if atoms.has("molecule_ids"):
        start_molecule_id = int(np.max(atoms.get_array("molecule_ids"))) + 1
    ion_atoms = monatomic_ions(list(ions), positions, start_molecule_id=start_molecule_id)
    combined = combine_systems(
        [atoms, ion_atoms],
        cell=lengths,
        pbc=atoms.get_pbc(),
        calculator=calculator,
        **calculator_kwargs,
    )
    for name in ("solvation", "membrane"):
        if hasattr(atoms, name):
            setattr(combined, name, dict(getattr(atoms, name)))
    combined.ions = {"placed_ions": list(ions)}
    return combined
