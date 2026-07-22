"""Small trajectory and energy-log helpers for :mod:`pyqed.md`."""

from os import PathLike

import numpy as np

from pyqed.units import au2angstrom

from .barostat import AU_PRESSURE_TO_BAR, semi_isotropic_pressure


def write_xyz(atoms, fileobj, comment=""):
    """Write one XYZ frame to a file-like object."""
    positions = atoms.get_positions()
    fileobj.write(f"{len(atoms)}\n")
    fileobj.write(f"{comment}\n")
    for symbol, xyz in zip(atoms.atom_symbols(), positions):
        fileobj.write(f"{symbol} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def write_pdb(atoms, fileobj, positions=None):
    """Write a residue-aware PDB snapshot.

    Coordinates are written in Angstrom.  When present, ``atom_names``,
    ``residue_names`` and ``residue_ids`` arrays are used for PDB metadata.
    Otherwise conservative per-atom fallbacks are generated.
    """
    close = False
    if isinstance(fileobj, (str, bytes, PathLike)):
        fileobj = open(fileobj, "w")
        close = True
    try:
        positions = atoms.get_positions() if positions is None else np.asarray(positions, dtype=float)
        positions_angstrom = positions * au2angstrom
        lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
        if lengths.shape == (3,) and np.all(lengths > 0.0):
            lengths_angstrom = lengths * au2angstrom
            fileobj.write(
                f"CRYST1{lengths_angstrom[0]:9.3f}{lengths_angstrom[1]:9.3f}{lengths_angstrom[2]:9.3f}"
                "  90.00  90.00  90.00 P 1           1\n"
            )

        symbols = np.asarray(atoms.atom_symbols(), dtype=str)
        natoms = len(atoms)
        atom_names = _string_array(atoms, "atom_names", symbols)
        residue_names = _string_array(atoms, "residue_names", np.full(natoms, "MOL"))
        residue_ids = _array_or_default(atoms, "residue_ids", np.arange(1, natoms + 1))
        chain_ids = _chain_ids(atoms, natoms)
        for index, (symbol, name, resname, resid, chain, xyz) in enumerate(
            zip(symbols, atom_names, residue_names, residue_ids, chain_ids, positions_angstrom),
            start=1,
        ):
            record = "ATOM  " if str(resname).upper() in {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"} else "HETATM"
            fileobj.write(
                f"{record}{index % 100000:5d} {_pdb_atom_name(name, symbol)} "
                f"{_pdb_resname(resname):>3s} {str(chain)[:1]:1s}{_pdb_resid(resid):4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {str(symbol).upper()[:2]:>2s}\n"
            )
        _write_conect_records(fileobj, atoms, natoms)
        fileobj.write("END\n")
    finally:
        if close:
            fileobj.close()


class XYZTrajectoryWriter:
    """Observer-compatible XYZ trajectory writer."""

    def __init__(self, atoms, filename, dynamics=None):
        self.atoms = atoms
        self.dynamics = dynamics
        self.fileobj = open(filename, "w")

    def __call__(self):
        time = "" if self.dynamics is None else f"time={self.dynamics.get_time():.8f}"
        write_xyz(self.atoms, self.fileobj, comment=time)
        self.fileobj.flush()

    def close(self):
        self.fileobj.close()


class PDBSnapshotWriter:
    """Observer-compatible writer for the current PDB snapshot."""

    def __init__(self, atoms, filename):
        self.atoms = atoms
        self.filename = filename

    def __call__(self):
        write_pdb(self.atoms, self.filename)


class EnergyLogger:
    """Observer-compatible energy logger."""

    def __init__(self, atoms, filename, dynamics=None, include_pressure="auto"):
        self.atoms = atoms
        self.dynamics = dynamics
        self.include_pressure = self._resolve_pressure_logging(include_pressure)
        self.fileobj = open(filename, "w")
        header = "step time potential kinetic total temperature_K"
        if self.include_pressure:
            header += " pressure_lateral_bar pressure_normal_bar pressure_xx_bar pressure_yy_bar pressure_zz_bar"
        self.fileobj.write(header + "\n")

    def __call__(self):
        step = 0 if self.dynamics is None else self.dynamics.get_number_of_steps()
        time = 0.0 if self.dynamics is None else self.dynamics.get_time()
        potential = self.atoms.get_potential_energy()
        kinetic = self.atoms.get_kinetic_energy()
        line = (
            f"{step:d} {time:.10f} {potential:.12e} {kinetic:.12e} "
            f"{potential + kinetic:.12e} {self.atoms.get_temperature():.8f}"
        )
        if self.include_pressure:
            lateral, normal, tensor = semi_isotropic_pressure(self.atoms)
            diagonal = np.diag(tensor) * AU_PRESSURE_TO_BAR
            line += (
                f" {lateral * AU_PRESSURE_TO_BAR:.12e}"
                f" {normal * AU_PRESSURE_TO_BAR:.12e}"
                f" {diagonal[0]:.12e} {diagonal[1]:.12e} {diagonal[2]:.12e}"
            )
        self.fileobj.write(line + "\n")
        self.fileobj.flush()

    def close(self):
        self.fileobj.close()

    def _resolve_pressure_logging(self, include_pressure):
        if include_pressure not in {True, False, "auto"}:
            raise ValueError("include_pressure must be True, False, or 'auto'.")
        if include_pressure is False:
            return False
        enabled = self._has_positive_cell()
        if include_pressure is True and not enabled:
            raise ValueError("pressure logging requires a positive 3D cell.")
        return enabled

    def _has_positive_cell(self):
        try:
            lengths = np.asarray(self.atoms.get_cell().lengths(), dtype=float)
        except Exception:
            return False
        return bool(lengths.shape == (3,) and np.all(lengths > 0.0))


def _array_or_default(atoms, name, default):
    if hasattr(atoms, "has") and atoms.has(name):
        return atoms.get_array(name)
    return np.asarray(default)


def _string_array(atoms, name, default):
    return np.asarray(_array_or_default(atoms, name, default), dtype=str)


def _chain_ids(atoms, natoms):
    if hasattr(atoms, "has") and atoms.has("chain_ids"):
        chain_ids = np.asarray(atoms.get_array("chain_ids"), dtype=str)
        return np.asarray([(str(chain).strip() or "A")[:1] for chain in chain_ids], dtype=str)
    if hasattr(atoms, "has") and atoms.has("leaflets"):
        leaflets = atoms.get_array("leaflets")
        result = []
        for leaflet in leaflets:
            result.append("U" if int(leaflet) > 0 else "L" if int(leaflet) < 0 else "A")
        return np.asarray(result, dtype=str)
    return np.full(natoms, "A", dtype=str)


def _pdb_atom_name(name, symbol):
    name = str(name).strip() or str(symbol).strip()
    name = name[:4]
    if len(name) < 4 and len(str(symbol).strip()) == 1:
        return f" {name:<3s}"
    return f"{name:<4s}"


def _pdb_resid(value):
    try:
        return int(value) % 10000
    except (TypeError, ValueError):
        digits = "".join(character for character in str(value) if character.isdigit())
        return int(digits or 1) % 10000


def _pdb_resname(value):
    return (str(value).strip().upper() or "MOL")[:3]


def _write_conect_records(fileobj, atoms, natoms):
    adjacency = [set() for _ in range(natoms)]
    for i, j in _pdb_conect_pairs(atoms):
        if 0 <= i < natoms and 0 <= j < natoms and i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)
    for index, neighbors in enumerate(adjacency, start=1):
        sorted_neighbors = sorted(neighbor + 1 for neighbor in neighbors)
        for start in range(0, len(sorted_neighbors), 4):
            chunk = sorted_neighbors[start : start + 4]
            fileobj.write(f"CONECT{index:5d}" + "".join(f"{neighbor:5d}" for neighbor in chunk) + "\n")


def _pdb_conect_pairs(atoms):
    if hasattr(atoms, "pdb_bonds"):
        return [(int(i), int(j)) for i, j in getattr(atoms, "pdb_bonds")]
    pairs = []
    topology = getattr(atoms, "topology", None)
    if topology is not None:
        for bond in getattr(topology, "bonds", ()) or ():
            if len(bond) >= 2:
                pairs.append((int(bond[0]), int(bond[1])))
    return pairs


class MCBarostatLogger:
    """Observer-compatible logger for Metropolis barostat attempts."""

    def __init__(
        self,
        barostat,
        filename,
        dynamics=None,
        lipids_per_leaflet=None,
        include_pressure="auto",
    ):
        self.barostat = barostat
        self.atoms = barostat.atoms
        self.dynamics = dynamics
        self.lipids_per_leaflet = None if lipids_per_leaflet is None else int(lipids_per_leaflet)
        self.include_pressure = self._resolve_pressure_logging(include_pressure)
        self._last_attempt = 0
        self.fileobj = open(filename, "w")
        header = (
            "step time attempt move accepted acceptance_rate "
            "scale_x scale_y scale_z old_energy new_energy delta_energy work "
            "log_jacobian log_acceptance lx ly lz area_per_lipid_angstrom2"
        )
        if self.include_pressure:
            header += " pressure_lateral_bar pressure_normal_bar"
        self.fileobj.write(header + "\n")

    def __call__(self):
        if self.barostat.attempts <= self._last_attempt:
            return
        self._last_attempt = self.barostat.attempts
        step = 0 if self.dynamics is None else self.dynamics.get_number_of_steps()
        time = 0.0 if self.dynamics is None else self.dynamics.get_time()
        lengths = np.asarray(self.atoms.get_cell().lengths(), dtype=float)
        area = lengths[0] * lengths[1]
        area_per_lipid = np.nan
        if self.lipids_per_leaflet:
            area_per_lipid = area * au2angstrom**2 / self.lipids_per_leaflet
        accepted = -1 if self.barostat.last_accepted is None else int(bool(self.barostat.last_accepted))
        scale = np.asarray(self.barostat.last_scale, dtype=float)
        line = (
            f"{step:d} {time:.10f} {self.barostat.attempts:d} {self.barostat.last_move} "
            f"{accepted:d} {self.barostat.acceptance_rate:.8f} "
            f"{scale[0]:.12e} {scale[1]:.12e} {scale[2]:.12e} "
            f"{self.barostat.last_old_energy:.12e} {self.barostat.last_new_energy:.12e} "
            f"{self.barostat.last_delta_energy:.12e} {self.barostat.last_work:.12e} "
            f"{self.barostat.last_log_jacobian:.12e} {self.barostat.last_log_acceptance:.12e} "
            f"{lengths[0] * au2angstrom:.8f} {lengths[1] * au2angstrom:.8f} "
            f"{lengths[2] * au2angstrom:.8f} {area_per_lipid:.8f}"
        )
        if self.include_pressure:
            lateral, normal, _tensor = semi_isotropic_pressure(self.atoms)
            line += (
                f" {lateral * AU_PRESSURE_TO_BAR:.12e}"
                f" {normal * AU_PRESSURE_TO_BAR:.12e}"
            )
        self.fileobj.write(line + "\n")
        self.fileobj.flush()

    def close(self):
        self.fileobj.close()

    def _resolve_pressure_logging(self, include_pressure):
        if include_pressure not in {True, False, "auto"}:
            raise ValueError("include_pressure must be True, False, or 'auto'.")
        if include_pressure is False:
            return False
        try:
            lengths = np.asarray(self.atoms.get_cell().lengths(), dtype=float)
        except Exception:
            lengths = np.zeros(3)
        enabled = bool(lengths.shape == (3,) and np.all(lengths > 0.0))
        if include_pressure is True and not enabled:
            raise ValueError("pressure logging requires a positive 3D cell.")
        return enabled
