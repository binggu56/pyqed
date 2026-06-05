"""Small CHARMM-style parameter helpers for :mod:`pyqed.md`.

This is a focused compatibility layer for common lipid-force-field terms.  It
parses a practical subset of CHARMM parameter files: bonds, angles, dihedrals,
impropers, nonbonded LJ parameters, and NBFIX pair overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyqed.units import au2angstrom, kcalmol2au

from .atoms import Atoms
from .calculators import MM
from .topology import Topology


@dataclass
class CharmmParameters:
    masses: dict = field(default_factory=dict)
    bonds: dict = field(default_factory=dict)
    angles: dict = field(default_factory=dict)
    dihedrals: dict = field(default_factory=dict)
    impropers: dict = field(default_factory=dict)
    cmaps: dict = field(default_factory=dict)
    nonbonded: dict = field(default_factory=dict)
    nbfix: dict = field(default_factory=dict)
    unsupported_sections: list = field(default_factory=list)

    def merge(self, other):
        """Merge another parameter block into this one, with later terms winning."""
        self.masses.update(other.masses)
        self.bonds.update(other.bonds)
        self.angles.update(other.angles)
        for key, values in other.dihedrals.items():
            self.dihedrals.setdefault(key, []).extend(values)
        self.impropers.update(other.impropers)
        self.cmaps.update(other.cmaps)
        self.nonbonded.update(other.nonbonded)
        self.nbfix.update(other.nbfix)
        self.unsupported_sections.extend(other.unsupported_sections)
        return self


@dataclass
class CharmmPsf:
    atom_names: list
    atom_types: list
    charges: list
    masses: list
    molecule_ids: list
    residue_ids: list
    residue_names: list
    segment_ids: list
    bonds: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    torsions: list = field(default_factory=list)
    impropers: list = field(default_factory=list)
    cmaps: list = field(default_factory=list)


def read_charmm_parameters(filename):
    """Read one or more CHARMM ``.prm``/``.str`` parameter files."""
    if _is_path_sequence(filename):
        params = CharmmParameters()
        for item in filename:
            params.merge(read_charmm_parameters(item))
        return params
    params = CharmmParameters()
    section = None
    with open(filename) as handle:
        for raw_line in handle:
            line = _strip_comment(raw_line)
            if not line:
                continue
            head = line.split()[0].upper()
            if head in {
                "MASS",
                "BOND",
                "BONDS",
                "ANGLE",
                "ANGLES",
                "DIHEDRAL",
                "DIHEDRALS",
                "IMPROPER",
                "IMPROPERS",
                "NONBONDED",
                "NBFIX",
                "CMAP",
                "HBOND",
                "NBTHOLE",
                "END",
            }:
                if head == "END":
                    break
                if head == "MASS":
                    _parse_mass(line, params)
                    section = "MASS"
                    continue
                if head == "CMAP":
                    section = "CMAP"
                    continue
                if head in {"HBOND", "NBTHOLE"}:
                    if head not in params.unsupported_sections:
                        params.unsupported_sections.append(head)
                    section = f"UNSUPPORTED:{head}"
                    continue
                section = head.rstrip("S")
                continue
            fields = line.split()
            if section == "MASS":
                _parse_mass(line, params)
            elif section == "BOND":
                if len(fields) < 4 or not _all_float(fields[2:4]):
                    continue
                key = tuple(fields[:2])
                params.bonds[_sorted_key(key)] = (
                    _bond_force(fields[2]),
                    _length(fields[3]),
                )
            elif section == "ANGLE":
                if len(fields) < 5 or not _all_float(fields[3:5]):
                    continue
                key = tuple(fields[:3])
                params.angles[key] = (_angle_force(fields[3]), float(fields[4]))
            elif section == "DIHEDRAL":
                if len(fields) < 7 or not _all_float([fields[4], fields[5], fields[6]]):
                    continue
                key = tuple(fields[:4])
                params.dihedrals.setdefault(key, []).append(
                    (_energy(fields[4]), int(fields[5]), float(fields[6]))
                )
            elif section == "IMPROPER":
                if len(fields) < 6 or not _all_float(fields[4:6]):
                    continue
                key = tuple(fields[:4])
                params.impropers[key] = (_angle_force(fields[4]), float(fields[5]))
            elif section == "CMAP":
                if len(fields) < 9 or not _is_int(fields[8]):
                    continue
                key = tuple(fields[:8])
                size = int(fields[8])
                values = [float(value) for value in fields[9:] if _is_float(value)]
                while len(values) < size * size:
                    try:
                        continuation = next(handle)
                    except StopIteration:
                        raise ValueError(f"incomplete CHARMM CMAP grid for {'-'.join(key)}")
                    values.extend(
                        float(value)
                        for value in _strip_comment(continuation).split()
                        if _is_float(value)
                    )
                params.cmaps[key] = (size, np.asarray(values[: size * size], dtype=float) * kcalmol2au)
            elif section == "NONBONDED":
                if len(fields) >= 4 and _all_float(fields[1:4]):
                    epsilon = abs(float(fields[2])) * kcalmol2au
                    sigma = _sigma_from_charmm_rmin2(fields[3])
                    params.nonbonded[fields[0]] = (epsilon, sigma)
            elif section == "NBFIX":
                if len(fields) < 4 or not _all_float(fields[2:4]):
                    continue
                key = _sorted_key(fields[:2])
                epsilon = abs(float(fields[2])) * kcalmol2au
                sigma = _sigma_from_charmm_rmin(fields[3])
                params.nbfix[key] = (epsilon, sigma)
    return params


def read_charmm_psf(filename):
    """Read a practical subset of a CHARMM/XPLOR PSF file."""
    atom_names = []
    atom_types = []
    charges = []
    masses = []
    molecule_ids = []
    residue_ids = []
    residue_names = []
    segment_ids = []
    bonds = []
    angles = []
    torsions = []
    impropers = []
    cmaps = []
    with open(filename) as handle:
        lines = list(handle)

    index = 0
    residue_to_molecule = {}
    while index < len(lines):
        line = lines[index]
        fields = line.split()
        index += 1
        if len(fields) < 2 or not fields[1].startswith("!"):
            continue
        count = int(fields[0])
        tag = fields[1].upper()
        if tag.startswith("!NATOM"):
            for _ in range(count):
                atom_fields = lines[index].split()
                index += 1
                if len(atom_fields) < 8:
                    raise ValueError("PSF atom line must contain at least 8 fields.")
                segment, resid, resname, atom_name, atom_type = atom_fields[1:6]
                key = (segment, resid)
                if key not in residue_to_molecule:
                    residue_to_molecule[key] = len(residue_to_molecule)
                segment_ids.append(segment)
                residue_ids.append(resid)
                residue_names.append(resname)
                atom_names.append(atom_name)
                atom_types.append(atom_type)
                charges.append(float(atom_fields[6]))
                masses.append(float(atom_fields[7]))
                molecule_ids.append(residue_to_molecule[key])
        elif tag.startswith("!NBOND"):
            pairs, index = _read_psf_int_records(lines, index, count * 2, 2)
            bonds.extend(pairs)
        elif tag.startswith("!NTHETA"):
            records, index = _read_psf_int_records(lines, index, count * 3, 3)
            angles.extend(records)
        elif tag.startswith("!NPHI"):
            records, index = _read_psf_int_records(lines, index, count * 4, 4)
            torsions.extend(records)
        elif tag.startswith("!NIMPHI"):
            records, index = _read_psf_int_records(lines, index, count * 4, 4)
            impropers.extend(records)
        elif tag.startswith("!NCRTERM"):
            records, index = _read_psf_int_records(lines, index, count * 8, 8)
            cmaps.extend(records)
        elif tag.startswith("!NDON") or tag.startswith("!NACC"):
            _records, index = _read_psf_int_records(lines, index, count * 2, 2)
        elif tag.startswith("!NNB"):
            index = _skip_to_next_psf_header(lines, index)
        else:
            index = _skip_to_next_psf_header(lines, index)
    return CharmmPsf(
        atom_names=atom_names,
        atom_types=atom_types,
        charges=charges,
        masses=masses,
        molecule_ids=molecule_ids,
        residue_ids=residue_ids,
        residue_names=residue_names,
        segment_ids=segment_ids,
        bonds=[tuple(i - 1 for i in pair) for pair in bonds],
        angles=[tuple(i - 1 for i in angle) for angle in angles],
        torsions=[tuple(i - 1 for i in torsion) for torsion in torsions],
        impropers=[tuple(i - 1 for i in improper) for improper in impropers],
        cmaps=[tuple(i - 1 for i in cmap) for cmap in cmaps],
    )


def read_pdb_coordinates(filename):
    """Read atom symbols, coordinates, and optional orthorhombic cell from PDB."""
    symbols = []
    positions = []
    cell = None
    with open(filename) as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record == "CRYST1":
                lengths = [float(line[6:15]), float(line[15:24]), float(line[24:33])]
                angles = [float(line[33:40]), float(line[40:47]), float(line[47:54])]
                if all(abs(angle - 90.0) < 1e-8 for angle in angles):
                    cell = np.asarray(lengths, dtype=float) / au2angstrom
            elif record in {"ATOM", "HETATM"}:
                name = line[12:16].strip()
                element = line[76:78].strip() or _symbol_from_name(name)
                xyz = [
                    float(line[30:38]) / au2angstrom,
                    float(line[38:46]) / au2angstrom,
                    float(line[46:54]) / au2angstrom,
                ]
                symbols.append(_clean_symbol(element))
                positions.append(xyz)
    return symbols, np.asarray(positions, dtype=float), cell


def atoms_from_charmm(
    psf_file,
    parameter_file,
    pdb_file=None,
    coordinates=None,
    symbols=None,
    cell=None,
    pbc=True,
    calculator=True,
    **calculator_kwargs,
):
    """Build typed :class:`Atoms` from CHARMM PSF plus parameters and coordinates."""
    psf = read_charmm_psf(psf_file)
    parameters = (
        read_charmm_parameters(parameter_file)
        if isinstance(parameter_file, (str, bytes, Path)) or _is_path_sequence(parameter_file)
        else parameter_file
    )
    pdb_cell = None
    if pdb_file is not None:
        pdb_symbols, pdb_positions, pdb_cell = read_pdb_coordinates(pdb_file)
        if coordinates is None:
            coordinates = pdb_positions
        if symbols is None:
            symbols = pdb_symbols
    if coordinates is None:
        coordinates = np.zeros((len(psf.atom_types), 3), dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(psf.atom_types), 3):
        raise ValueError("coordinates must have shape (natoms, 3).")
    if symbols is None:
        symbols = [_symbol_from_name(name) for name in psf.atom_names]
    if len(symbols) != len(psf.atom_types):
        raise ValueError("symbols must match the PSF atom count.")
    topology = charmm_topology_from_types(
        psf.atom_types,
        psf.charges,
        bonds=psf.bonds,
        angles=psf.angles,
        torsions=psf.torsions,
        impropers=psf.impropers,
        cmaps=psf.cmaps,
        masses_amu=psf.masses,
        molecule_ids=psf.molecule_ids,
        atom_names=psf.atom_names,
        parameters=parameters,
    )
    atoms = Atoms(
        [[symbol, tuple(xyz)] for symbol, xyz in zip(symbols, coordinates)],
        cell=cell if cell is not None else pdb_cell,
        pbc=pbc,
    )
    atoms.topology = topology
    atoms.set_array("charges", topology.charges, float, ())
    atoms.set_array("lj_epsilon", topology.lj_epsilon, float, ())
    atoms.set_array("lj_sigma", topology.lj_sigma, float, ())
    atoms.set_array("molecule_ids", topology.molecule_ids, int, ())
    atoms.set_array("masses_amu", topology.masses_amu, float, ())
    atoms.set_array("atom_types", topology.atom_types, str, ())
    atoms.set_array("atom_names", topology.atom_names, str, ())
    atoms.set_array("residue_ids", np.asarray(psf.residue_ids), str, ())
    atoms.set_array("residue_names", np.asarray(psf.residue_names), str, ())
    atoms.set_array("segment_ids", np.asarray(psf.segment_ids), str, ())
    if calculator:
        atoms.calc = MM(
            bonds=topology.bonds,
            angles=topology.angles,
            torsions=topology.torsions,
            impropers=topology.impropers,
            cmaps=topology.cmaps,
            cmap_grids=topology.cmap_grids,
            angle_unit="degree",
            torsion_unit="degree",
            improper_unit="degree",
            charges=topology.charges,
            lj_epsilon=topology.lj_epsilon,
            lj_sigma=topology.lj_sigma,
            atom_types=topology.atom_types,
            lj_pair_overrides=topology.lj_pair_overrides,
            exclude_bonded=True,
            exclude_angles=True,
            **calculator_kwargs,
        )
    return atoms


def charmm_topology_from_types(
    atom_types,
    charges,
    bonds=(),
    angles=(),
    torsions=(),
    impropers=(),
    cmaps=(),
    masses_amu=None,
    molecule_ids=None,
    atom_names=None,
    parameters=None,
):
    """Build a :class:`Topology` by assigning CHARMM parameters to typed atoms."""
    if parameters is None:
        raise ValueError("parameters is required.")
    atom_types = np.asarray(atom_types, dtype=str)
    charges = np.asarray(charges, dtype=float)
    lj_epsilon = []
    lj_sigma = []
    for atom_type in atom_types:
        try:
            epsilon, sigma = parameters.nonbonded[str(atom_type)]
        except KeyError as exc:
            raise KeyError(f"missing NONBONDED parameter for atom type {atom_type!r}") from exc
        lj_epsilon.append(epsilon)
        lj_sigma.append(sigma)

    bond_terms = []
    for i, j in bonds:
        k, r0 = _lookup_one(parameters.bonds, (atom_types[i], atom_types[j]), "bond")
        bond_terms.append((int(i), int(j), k, r0))

    angle_terms = []
    for i, j, k in angles:
        ktheta, theta0 = _lookup_one(
            parameters.angles,
            (atom_types[i], atom_types[j], atom_types[k]),
            "angle",
        )
        angle_terms.append((int(i), int(j), int(k), ktheta, theta0))

    torsion_terms = []
    for i, j, k, l in torsions:
        matches = _lookup_many(
            parameters.dihedrals,
            (atom_types[i], atom_types[j], atom_types[k], atom_types[l]),
            "dihedral",
        )
        for barrier, periodicity, phase in matches:
            torsion_terms.append((int(i), int(j), int(k), int(l), barrier, periodicity, phase))

    improper_terms = []
    for i, j, k, l in impropers:
        force_constant, phase = _lookup_one(
            parameters.impropers,
            (atom_types[i], atom_types[j], atom_types[k], atom_types[l]),
            "improper",
        )
        improper_terms.append((int(i), int(j), int(k), int(l), force_constant, phase))

    cmap_grids = []
    cmap_map_indices = {}
    cmap_terms = []
    for cmap in cmaps:
        key = tuple(atom_types[int(index)] for index in cmap)
        reverse_key = tuple(reversed(key))
        try:
            matched_key, grid = _lookup_cmap(parameters.cmaps, key, reverse_key)
        except KeyError as exc:
            raise KeyError(f"missing CHARMM CMAP parameter for {'-'.join(key)}") from exc
        if matched_key not in cmap_map_indices:
            cmap_map_indices[matched_key] = len(cmap_grids)
            cmap_grids.append(grid)
        cmap_terms.append((cmap_map_indices[matched_key], tuple(int(index) for index in cmap)))

    return Topology(
        bonds=bond_terms,
        angles=angle_terms,
        torsions=torsion_terms,
        impropers=improper_terms,
        cmaps=cmap_terms,
        cmap_grids=cmap_grids,
        charges=charges,
        lj_epsilon=lj_epsilon,
        lj_sigma=lj_sigma,
        molecule_ids=np.zeros(len(atom_types), dtype=int) if molecule_ids is None else molecule_ids,
        masses_amu=masses_amu,
        atom_types=atom_types,
        atom_names=atom_names,
        lj_pair_overrides=parameters.nbfix,
    )


def _strip_comment(line):
    return line.split("!", 1)[0].strip()


def _is_path_sequence(value):
    return not isinstance(value, (str, bytes, Path, CharmmParameters)) and hasattr(value, "__iter__")


def _all_float(values):
    try:
        for value in values:
            float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_float(value):
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _read_psf_int_records(lines, index, nvalues, width):
    values = []
    while len(values) < nvalues:
        values.extend(int(field) for field in lines[index].split())
        index += 1
    records = [tuple(values[i:i + width]) for i in range(0, nvalues, width)]
    return records, index


def _skip_to_next_psf_header(lines, index):
    while index < len(lines):
        fields = lines[index].split()
        if len(fields) >= 2 and fields[1].startswith("!") and _is_int(fields[0]):
            break
        index += 1
    return index


def _is_int(value):
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _parse_mass(line, params):
    fields = line.split()
    if len(fields) >= 4:
        params.masses[fields[2]] = float(fields[3])


def _lookup_one(table, key, label):
    matches = _lookup_many(table, key, label)
    return matches[0]


def _lookup_many(table, key, label):
    key = tuple(str(part) for part in key)
    candidates = [
        key,
        tuple(reversed(key)),
        ("X",) + key[1:-1] + ("X",),
        ("X",) + tuple(reversed(key))[1:-1] + ("X",),
    ]
    for candidate in candidates:
        if len(candidate) == 2:
            candidate = _sorted_key(candidate)
        if candidate in table:
            value = table[candidate]
            return value if isinstance(value, list) else [value]
    raise KeyError(f"missing CHARMM {label} parameter for {'-'.join(key)}")


def _lookup_cmap(table, *keys):
    for key in keys:
        key = tuple(str(part) for part in key)
        if key in table:
            return key, table[key]
    wildcard_keys = []
    for key in keys:
        key = tuple(str(part) for part in key)
        wildcard_keys.append(("X",) + key[1:3] + ("X", "X") + key[5:7] + ("X",))
    for key in wildcard_keys:
        if key in table:
            return key, table[key]
    raise KeyError("missing CHARMM CMAP parameter")


def _sorted_key(key):
    return tuple(sorted(str(part) for part in key))


def _energy(value):
    return float(value) * kcalmol2au


def _length(value):
    return float(value) / au2angstrom


def _bond_force(value):
    return float(value) * kcalmol2au * au2angstrom**2


def _angle_force(value):
    return float(value) * kcalmol2au


def _sigma_from_charmm_rmin2(value):
    return 2.0 * float(value) / (2.0 ** (1.0 / 6.0)) / au2angstrom


def _sigma_from_charmm_rmin(value):
    return float(value) / (2.0 ** (1.0 / 6.0)) / au2angstrom


def _symbol_from_name(name):
    letters = "".join(char for char in str(name).strip() if char.isalpha())
    if not letters:
        return "X"
    if len(letters) >= 2 and letters[:2].upper() in {"CL", "BR", "NA", "MG", "ZN", "CA"}:
        return _clean_symbol(letters[:2])
    return _clean_symbol(letters[0])


def _clean_symbol(symbol):
    symbol = str(symbol).strip()
    if not symbol:
        return "X"
    return symbol[0].upper() + symbol[1:].lower()
