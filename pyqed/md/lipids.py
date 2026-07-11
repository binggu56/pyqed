"""Native lipid-template membrane builders for :mod:`pyqed.md`.

The first template in this module is a compact DPPC-like development template.
It is meant to exercise native membrane topology, packing, and MD workflows.
It is not a validated CHARMM36/AMBER lipid force field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.units import au2angstrom, kcalmol2au

from .atoms import Atoms
from .membrane import solvate_membrane
from .solvation import combine_systems
from .topology import Topology


@dataclass(frozen=True)
class LipidTemplate:
    """One native lipid residue template.

    Coordinates are a reference single-lipid conformation in Bohr.  Force-field
    parameters are in atomic units and are intentionally stored directly on the
    template so a bilayer can be built without external files.
    """

    name: str
    residue_name: str
    description: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    atom_types: tuple[str, ...]
    masses_amu: tuple[float, ...]
    charges: tuple[float, ...]
    lj_epsilon: tuple[float, ...]
    lj_sigma: tuple[float, ...]
    positions: np.ndarray
    bonds: tuple[tuple[int, int, float, float], ...]
    angles: tuple[tuple[int, int, int, float, float], ...]
    torsions: tuple[tuple[int, int, int, int, float, int, float], ...]
    head_atom_names: tuple[str, ...]
    tail_pairs: tuple[tuple[int, int], ...]
    validated: bool = False
    forcefield: str = "pyqed-dev"
    lj_pair_scales: object = None
    coulomb_pair_scales: object = None

    def __post_init__(self):
        natoms = len(self.atom_names)
        if len(self.elements) != natoms or len(self.atom_types) != natoms:
            raise ValueError("template atom metadata arrays must have matching length.")
        if len(self.charges) != natoms or len(self.lj_epsilon) != natoms or len(self.lj_sigma) != natoms:
            raise ValueError("template parameter arrays must have matching length.")
        positions = np.asarray(self.positions, dtype=float)
        if positions.shape != (natoms, 3):
            raise ValueError(f"template positions must have shape ({natoms}, 3).")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "lj_pair_scales", _pair_scale_dict(self.lj_pair_scales))
        object.__setattr__(self, "coulomb_pair_scales", _pair_scale_dict(self.coulomb_pair_scales))

    @property
    def natoms(self):
        return len(self.atom_names)

    @property
    def net_charge(self):
        return float(np.sum(self.charges))

    def head_indices(self):
        names = set(self.head_atom_names)
        return np.asarray(
            [index for index, name in enumerate(self.atom_names) if name in names],
            dtype=int,
        )


def lipid_template(name="DPPC", openmm_source=None):
    """Return a native lipid template by name.

    ``DPPC`` currently resolves to the compact development template
    ``DPPC-DEV``.  The alias gives callers a stable API while the parameters are
    upgraded toward a validated source.  ``DPPC-OPENMM`` returns a full
    all-atom template with OpenMM Amber lipid17 parameters and PyQED-generated
    reference coordinates.
    """
    key = str(name).upper().replace("_", "-")
    if key in {"DPPC", "DPPC-DEV", "DPPC_DEV"}:
        return _dppc_dev_template()
    if key in {"DPPC-OPENMM", "DPPC-AMBER", "DPPC-LIPID17", "DPPC-OPENMM-LIPID17"}:
        return _dppc_openmm_template(source=openmm_source)
    raise ValueError(f"unsupported lipid template {name!r}; available templates: DPPC, DPPC-OPENMM")


def available_lipid_templates():
    """Return native lipid-template names understood by the builder."""
    return ("DPPC", "DPPC-DEV", "DPPC-OPENMM", "DPPC-OPENMM-LIPID17")


def lipid_from_template(
    template="DPPC",
    origin=(0.0, 0.0, 0.0),
    leaflet=1,
    molecule_id=0,
    residue_id=1,
    rotation=0.0,
    openmm_source=None,
):
    """Build one lipid molecule from a native template."""
    template = lipid_template(template, openmm_source=openmm_source) if isinstance(template, str) else template
    origin = np.asarray(origin, dtype=float)
    sign = 1.0 if leaflet >= 0 else -1.0
    rot = _rotation_z(float(rotation))
    local = template.positions.copy()
    xy = local[:, :2] @ rot.T
    positions = np.column_stack([xy, sign * local[:, 2]]) + origin

    topology = Topology(
        bonds=template.bonds,
        angles=template.angles,
        torsions=template.torsions,
        charges=template.charges,
        lj_epsilon=template.lj_epsilon,
        lj_sigma=template.lj_sigma,
        molecule_ids=np.full(template.natoms, int(molecule_id)),
        masses_amu=template.masses_amu,
        atom_types=template.atom_types,
        atom_names=template.atom_names,
        lj_pair_scales=template.lj_pair_scales,
        coulomb_pair_scales=template.coulomb_pair_scales,
    )
    atoms = Atoms([[element, tuple(position)] for element, position in zip(template.elements, positions)])
    atoms.topology = topology
    atoms.set_array("charges", topology.charges, float, ())
    atoms.set_array("lj_epsilon", topology.lj_epsilon, float, ())
    atoms.set_array("lj_sigma", topology.lj_sigma, float, ())
    atoms.set_array("molecule_ids", topology.molecule_ids, int, ())
    atoms.set_array("masses_amu", topology.masses_amu, float, ())
    atoms.set_array("atom_types", topology.atom_types, str, ())
    atoms.set_array("atom_names", topology.atom_names, str, ())
    atoms.set_array("residue_ids", np.full(template.natoms, int(residue_id)), int, ())
    atoms.set_array("residue_names", np.full(template.natoms, template.residue_name), str, ())
    atoms.set_array("leaflets", np.full(template.natoms, 1 if leaflet >= 0 else -1), int, ())
    return atoms


def lipid_bilayer_from_template(
    lipid="DPPC",
    nx=2,
    ny=2,
    area_per_lipid=64.0,
    thickness=38.0,
    water_padding=18.0,
    pbc=True,
    calculator=True,
    coulomb_method="pme",
    coulomb_cutoff=10.0,
    lj_cutoff=10.0,
    pme_mesh=(16, 16, 24),
    seed=None,
    openmm_source=None,
):
    """Build a two-leaflet bilayer from a native lipid template.

    Public geometry arguments and cutoffs are in Angstrom.
    """
    template = lipid_template(lipid, openmm_source=openmm_source) if isinstance(lipid, str) else lipid
    nx = int(nx)
    ny = int(ny)
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")

    spacing = np.sqrt(float(area_per_lipid)) / au2angstrom
    lx = nx * spacing
    ly = ny * spacing
    thickness_bohr = float(thickness) / au2angstrom
    lz = (float(thickness) + 2.0 * float(water_padding)) / au2angstrom
    center_z = 0.5 * lz
    head_offset = 0.5 * thickness_bohr
    rng = np.random.default_rng(seed)

    lipids = []
    leaflets = []
    residue_names = []
    residue_ids = []
    molecule_id = 0
    for leaflet in (1, -1):
        head_z = center_z + leaflet * head_offset
        leaflet_shift = 0.0 if leaflet > 0 else 0.5
        for ix in range(nx):
            for iy in range(ny):
                jitter = rng.uniform(-0.05, 0.05, size=2) * spacing if seed is not None else 0.0
                origin = np.array(
                    [
                        ((ix + 0.5 + leaflet_shift) % nx) * spacing + (jitter[0] if seed is not None else 0.0),
                        ((iy + 0.5 + leaflet_shift) % ny) * spacing + (jitter[1] if seed is not None else 0.0),
                        head_z,
                    ]
                )
                origin[:2] = np.mod(origin[:2], [lx, ly])
                rotation = 0.5 * np.pi * ((ix + iy + (0 if leaflet > 0 else 1)) % 4)
                lipid_atoms = lipid_from_template(
                    template,
                    origin=origin,
                    leaflet=leaflet,
                    molecule_id=molecule_id,
                    residue_id=molecule_id + 1,
                    rotation=rotation,
                )
                lipids.append(lipid_atoms)
                leaflets.extend([leaflet] * template.natoms)
                residue_names.extend([template.residue_name] * template.natoms)
                residue_ids.extend([molecule_id + 1] * template.natoms)
                molecule_id += 1

    cell = np.array([lx, ly, lz], dtype=float)
    cutoff = _safe_cutoff(cell, coulomb_cutoff)
    lj_cutoff_bohr = _safe_cutoff(cell, lj_cutoff)
    system = combine_systems(
        lipids,
        cell=cell,
        pbc=pbc,
        calculator=calculator,
        coulomb_method=coulomb_method,
        coulomb_cutoff=cutoff,
        lj_cutoff=lj_cutoff_bohr,
        pme_mesh=pme_mesh,
    )
    system.set_array("leaflets", np.asarray(leaflets, dtype=int), int, ())
    system.set_array("residue_names", np.asarray(residue_names), str, ())
    system.set_array("residue_ids", np.asarray(residue_ids, dtype=int), int, ())
    system.membrane = {
        "kind": "template_bilayer",
        "template": template.name,
        "residue_name": template.residue_name,
        "forcefield": template.forcefield,
        "validated": template.validated,
        "lipids_per_leaflet": nx * ny,
        "total_lipids": 2 * nx * ny,
        "atoms_per_lipid": template.natoms,
        "area_per_lipid_angstrom2": float(area_per_lipid),
        "thickness_angstrom": float(thickness),
        "water_padding_angstrom": float(water_padding),
        "center_z": center_z,
        "head_z": (center_z + head_offset, center_z - head_offset),
        "head_atom_names": template.head_atom_names,
        "tail_pairs": template.tail_pairs,
        "nonbonded_cutoff_angstrom": float(cutoff * au2angstrom),
    }
    return system


def hydrated_lipid_bilayer_from_template(
    lipid="DPPC",
    nx=2,
    ny=2,
    waters_per_side=4,
    water_spacing=3.2,
    seed=None,
    **kwargs,
):
    """Build a template bilayer with TIP3P water slabs."""
    builder_kwargs = dict(kwargs)
    builder_kwargs.pop("calculator", None)
    membrane = lipid_bilayer_from_template(lipid=lipid, nx=nx, ny=ny, seed=seed, calculator=False, **builder_kwargs)
    hydrated = solvate_membrane(
        membrane,
        spacing=water_spacing,
        max_waters_per_side=waters_per_side,
        rigid=True,
        seed=seed,
        calculator=True,
        coulomb_method=kwargs.get("coulomb_method", "pme"),
        coulomb_cutoff=kwargs.get("coulomb_cutoff", 10.0),
        lj_cutoff=kwargs.get("lj_cutoff", 10.0),
        pme_mesh=kwargs.get("pme_mesh", (16, 16, 24)),
    )
    hydrated.membrane.update(getattr(membrane, "membrane", {}))
    return hydrated


def _dppc_dev_template():
    names = (
        "N", "C11", "C12", "C13", "C14", "C15", "O11", "P", "O12", "O13", "O14",
        "GL1", "GL2", "GL3", "O21", "C21", "C22", "C23", "C24", "O31", "C31", "C32", "C33", "C34",
    )
    elements = (
        "N", "C", "C", "C", "C", "C", "O", "P", "O", "O", "O",
        "C", "C", "C", "O", "C", "C", "C", "C", "O", "C", "C", "C", "C",
    )
    atom_types = (
        "N_HEAD", "C_HEAD", "C_HEAD", "C_HEAD", "C_HEAD", "C_HEAD", "O_HEAD", "P_HEAD", "O_HEAD", "O_HEAD", "O_HEAD",
        "C_HEAD", "C_HEAD", "C_HEAD", "O_HEAD", "C_TAIL", "C_TAIL", "C_TAIL", "C_TAIL", "O_HEAD", "C_TAIL", "C_TAIL", "C_TAIL", "C_TAIL",
    )
    positions_angstrom = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.15, 0.0, 0.35],
            [-0.55, 1.00, 0.35],
            [-0.55, -1.00, 0.35],
            [0.0, 0.0, -1.45],
            [0.0, 0.0, -2.90],
            [0.0, 0.0, -4.15],
            [0.0, 0.0, -5.55],
            [1.10, 0.0, -5.70],
            [-1.10, 0.0, -5.70],
            [0.0, 0.0, -6.95],
            [0.0, 0.0, -8.30],
            [-0.85, 0.0, -9.45],
            [0.85, 0.0, -9.45],
            [-1.40, 0.0, -10.50],
            [-2.05, 0.30, -11.75],
            [-2.25, -0.25, -13.20],
            [-2.05, 0.25, -14.65],
            [-2.30, -0.20, -16.10],
            [1.40, 0.0, -10.50],
            [2.05, -0.30, -11.75],
            [2.25, 0.25, -13.20],
            [2.05, -0.25, -14.65],
            [2.30, 0.20, -16.10],
        ],
        dtype=float,
    )
    charges = np.array(
        [
            0.70, 0.05, 0.05, 0.05, 0.20, 0.20, -0.40, 1.10, -0.55, -0.55, -0.55,
            0.10, 0.10, 0.10, -0.30, 0.0, 0.0, 0.0, 0.0, -0.30, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=float,
    )
    charges[-1] -= np.sum(charges)
    epsilon_by_type = {
        "N_HEAD": 0.12,
        "P_HEAD": 0.16,
        "O_HEAD": 0.12,
        "C_HEAD": 0.06,
        "C_TAIL": 0.09,
    }
    sigma_by_type = {
        "N_HEAD": 3.25,
        "P_HEAD": 3.70,
        "O_HEAD": 3.00,
        "C_HEAD": 3.40,
        "C_TAIL": 3.55,
    }
    mass_by_element = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974}
    masses = tuple(mass_by_element[element] for element in elements)
    epsilon = tuple(epsilon_by_type[atom_type] * kcalmol2au for atom_type in atom_types)
    sigma = tuple(sigma_by_type[atom_type] / au2angstrom for atom_type in atom_types)

    adjacency = {
        0: (1, 2, 3, 4),
        4: (5,),
        5: (6,),
        6: (7,),
        7: (8, 9, 10),
        10: (11,),
        11: (12, 13),
        12: (14,),
        14: (15,),
        15: (16,),
        16: (17,),
        17: (18,),
        13: (19,),
        19: (20,),
        20: (21,),
        21: (22,),
        22: (23,),
    }
    undirected = _undirected_edges(adjacency)
    bonds = tuple(
        (i, j, _bond_k(atom_types[i], atom_types[j]), _distance(positions_angstrom[i], positions_angstrom[j]) / au2angstrom)
        for i, j in undirected
    )
    angles = tuple(
        (i, j, k, 30.0 * kcalmol2au, _angle(positions_angstrom[i], positions_angstrom[j], positions_angstrom[k]))
        for i, j, k in _angles_from_edges(undirected)
    )
    torsions = tuple(
        (i, j, k, l, 0.05 * kcalmol2au, 3, 0.0)
        for i, j, k, l in _torsions_from_edges(undirected)
        if _valid_torsion(
            positions_angstrom[i],
            positions_angstrom[j],
            positions_angstrom[k],
            positions_angstrom[l],
        )
    )
    tail_pairs = ((15, 16), (16, 17), (17, 18), (20, 21), (21, 22), (22, 23))
    return LipidTemplate(
        name="DPPC-DEV",
        residue_name="DPPC",
        description="Compact united-atom DPPC-like development template; not a validated lipid force field.",
        atom_names=names,
        elements=elements,
        atom_types=atom_types,
        masses_amu=masses,
        charges=tuple(float(value) for value in charges),
        lj_epsilon=epsilon,
        lj_sigma=sigma,
        positions=positions_angstrom / au2angstrom,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        head_atom_names=("N", "P"),
        tail_pairs=tail_pairs,
        validated=False,
    )


def _dppc_openmm_template(source=None):
    from .openmm_lipids import openmm_lipid_template

    template = openmm_lipid_template("DPPC", source=source)
    positions = _openmm_dppc_reference_positions(template)
    tail_pairs = _named_tail_pairs(
        template.atom_names,
        (
            ("C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C210", "C211", "C212", "C213", "C214", "C215", "C216"),
            ("C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C310", "C311", "C312", "C313", "C314", "C315", "C316"),
        ),
    )
    one_four_pairs = _one_four_pairs_from_torsions(template.torsions)
    return LipidTemplate(
        name="DPPC-OPENMM-LIPID17",
        residue_name=template.residue_name,
        description="Full all-atom DPPC template using OpenMM Amber lipid17 parameters and generated PyQED reference coordinates.",
        atom_names=template.atom_names,
        elements=template.elements,
        atom_types=template.atom_types,
        masses_amu=template.masses_amu,
        charges=template.charges,
        lj_epsilon=template.lj_epsilon,
        lj_sigma=template.lj_sigma,
        positions=positions,
        bonds=template.bonds,
        angles=template.angles,
        torsions=template.torsions,
        head_atom_names=("N", "P"),
        tail_pairs=tail_pairs,
        validated=True,
        forcefield=f"OpenMM Amber lipid17: {template.source}",
        lj_pair_scales={pair: template.lj14scale for pair in one_four_pairs},
        coulomb_pair_scales={pair: template.coulomb14scale for pair in one_four_pairs},
    )


def _openmm_dppc_reference_positions(template):
    names = template.atom_names
    name_to_index = {name: index for index, name in enumerate(names)}
    positions = np.full((template.natoms, 3), np.nan, dtype=float)

    def set_atom(name, xyz):
        if name in name_to_index:
            positions[name_to_index[name]] = np.asarray(xyz, dtype=float)

    head_positions = {
        "N": (0.0, 0.0, 0.0),
        "C13": (1.35, 0.00, 0.35),
        "C14": (-0.68, 1.17, 0.35),
        "C15": (-0.68, -1.17, 0.35),
        "C12": (0.24, -0.10, -1.45),
        "C11": (-0.18, 0.18, -2.90),
        "P": (0.10, -0.12, -5.55),
        "O13": (1.35, -0.06, -5.55),
        "O14": (-1.15, 0.10, -5.55),
        "O11": (-0.06, 1.18, -5.40),
        "O12": (0.08, -0.04, -6.85),
        "C1": (-0.08, 0.08, -8.20),
        "C2": (-0.85, 0.0, -9.40),
        "C3": (0.85, 0.0, -9.40),
        "O21": (-1.35, 0.0, -10.55),
        "C21": (-1.90, 0.25, -11.70),
        "O22": (-2.85, 0.65, -11.70),
        "O31": (1.35, 0.0, -10.55),
        "C31": (1.90, -0.25, -11.70),
        "O32": (2.85, -0.65, -11.70),
    }
    for name, xyz in head_positions.items():
        set_atom(name, xyz)

    _set_openmm_tail_positions(
        positions,
        name_to_index,
        ("C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C210", "C211", "C212", "C213", "C214", "C215", "C216"),
        base=(-2.05, -0.45, -12.95),
        lateral_sign=-1.0,
    )
    _set_openmm_tail_positions(
        positions,
        name_to_index,
        ("C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C310", "C311", "C312", "C313", "C314", "C315", "C316"),
        base=(2.05, 0.45, -12.95),
        lateral_sign=1.0,
    )

    _place_bonded_hydrogens(template, positions)
    missing = [name for name, position in zip(names, positions) if not np.all(np.isfinite(position))]
    if missing:
        raise ValueError(f"could not generate DPPC-OPENMM reference positions for atoms: {missing}")
    return positions / au2angstrom


def _set_openmm_tail_positions(positions, name_to_index, names, base, lateral_sign):
    base = np.asarray(base, dtype=float)
    for index, name in enumerate(names):
        if name not in name_to_index:
            continue
        zig = -1.0 if index % 2 == 0 else 1.0
        positions[name_to_index[name]] = base + np.array(
            [
                lateral_sign * 0.20 * zig,
                0.42 * zig,
                -1.27 * index,
            ],
            dtype=float,
        )


def _place_bonded_hydrogens(template, positions):
    graph = {}
    bond_lengths = {}
    for i, j, _k, r0 in template.bonds:
        graph.setdefault(i, set()).add(j)
        graph.setdefault(j, set()).add(i)
        bond_lengths[tuple(sorted((i, j)))] = float(r0) * au2angstrom

    hydrogens_by_parent = {}
    for index, element in enumerate(template.elements):
        if element != "H" or np.all(np.isfinite(positions[index])):
            continue
        heavy_neighbors = [neighbor for neighbor in graph.get(index, ()) if template.elements[neighbor] != "H"]
        if not heavy_neighbors:
            continue
        hydrogens_by_parent.setdefault(heavy_neighbors[0], []).append(index)

    for parent, hydrogen_indices in hydrogens_by_parent.items():
        bonded_heavy = [
            neighbor
            for neighbor in graph.get(parent, ())
            if template.elements[neighbor] != "H" and np.all(np.isfinite(positions[neighbor]))
        ]
        if bonded_heavy:
            neighbor_center = np.mean(positions[bonded_heavy], axis=0)
            away = _unit_vector(positions[parent] - neighbor_center)
        else:
            away = np.array([0.0, 0.0, 1.0], dtype=float)
        axis_a, axis_b = _orthogonal_axes(away)
        count = len(hydrogen_indices)
        phase = 0.37 * (parent % 7)
        for local_index, hydrogen in enumerate(sorted(hydrogen_indices)):
            angle = 2.0 * np.pi * local_index / max(count, 1) + phase
            direction = _unit_vector(
                1.20 * away
                + 0.45 * np.cos(angle) * axis_a
                + 0.45 * np.sin(angle) * axis_b
            )
            length = bond_lengths.get(tuple(sorted((parent, hydrogen))), 1.09)
            positions[hydrogen] = positions[parent] + length * direction


def _named_tail_pairs(atom_names, chains):
    name_to_index = {name: index for index, name in enumerate(atom_names)}
    pairs = []
    for chain in chains:
        for left, right in zip(chain[:-1], chain[1:]):
            if left in name_to_index and right in name_to_index:
                pairs.append((name_to_index[left], name_to_index[right]))
    return tuple(pairs)


def _one_four_pairs_from_torsions(torsions):
    return {tuple(sorted((int(i), int(l)))) for i, _j, _k, l, *_rest in torsions}


def _pair_scale_dict(pairs):
    if not pairs:
        return {}
    return {tuple(sorted((int(i), int(j)))): float(scale) for (i, j), scale in dict(pairs).items()}


def _unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vector / norm


def _orthogonal_axes(axis):
    axis = _unit_vector(axis)
    trial = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, trial))) > 0.9:
        trial = np.array([1.0, 0.0, 0.0], dtype=float)
    axis_a = _unit_vector(np.cross(axis, trial))
    axis_b = _unit_vector(np.cross(axis, axis_a))
    return axis_a, axis_b


def _safe_cutoff(cell, requested_angstrom, fraction=0.45):
    return min(float(requested_angstrom) / au2angstrom, fraction * float(np.min(cell)))


def _rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def _undirected_edges(adjacency):
    edges = set()
    for i, neighbors in adjacency.items():
        for j in neighbors:
            edges.add(tuple(sorted((int(i), int(j)))))
    return sorted(edges)


def _angles_from_edges(edges):
    graph = {}
    for i, j in edges:
        graph.setdefault(i, set()).add(j)
        graph.setdefault(j, set()).add(i)
    angles = set()
    for center, neighbors in graph.items():
        neighbors = sorted(neighbors)
        for a, i in enumerate(neighbors):
            for k in neighbors[a + 1:]:
                angles.add((i, center, k))
    return sorted(angles)


def _torsions_from_edges(edges):
    graph = {}
    for i, j in edges:
        graph.setdefault(i, set()).add(j)
        graph.setdefault(j, set()).add(i)
    torsions = set()
    for j, k in edges:
        for i in graph.get(j, ()):
            if i == k:
                continue
            for l in graph.get(k, ()):
                if l == j or l == i:
                    continue
                torsion = (i, j, k, l)
                reverse = tuple(reversed(torsion))
                torsions.add(min(torsion, reverse))
    return sorted(torsions)


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _angle(a, b, c):
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    cosine = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _bond_k(type_i, type_j):
    if "P_HEAD" in {type_i, type_j}:
        return 140.0 * kcalmol2au * au2angstrom**2
    if "C_TAIL" in {type_i, type_j}:
        return 80.0 * kcalmol2au * au2angstrom**2
    return 100.0 * kcalmol2au * au2angstrom**2


def _valid_torsion(a, b, c, d, tolerance=1.0e-10):
    b0 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    b1 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    b2 = np.asarray(d, dtype=float) - np.asarray(c, dtype=float)
    return bool(
        np.linalg.norm(np.cross(b0, b1)) > tolerance
        and np.linalg.norm(np.cross(b1, b2)) > tolerance
    )
