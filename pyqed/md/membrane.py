"""Toy membrane builders for :mod:`pyqed.md`.

The builders in this module are meant for code-path development and smoke
tests.  They are not a substitute for a production lipid force field or a
CHARMM-GUI/GROMACS membrane preparation workflow.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyqed.units import au2angstrom, kcalmol2au

from .atoms import Atoms
from .calculators import MM
from .neighborlist import minimum_image
from .solvation import combine_systems
from .topology import Topology
from .water import TIP3P_OH_DISTANCE, TIP3P_HOH_ANGLE, tip3p_waters


TOY_LIPID_SYMBOLS = ("P", "O", "C", "C", "C")
TOY_LIPID_CHARGES = np.array([0.6, -0.6, 0.0, 0.0, 0.0])
TOY_LIPID_LJ_EPSILON = np.array([0.10, 0.10, 0.12, 0.12, 0.12]) * kcalmol2au
TOY_LIPID_LJ_SIGMA = np.array([3.6, 3.3, 4.0, 4.0, 4.0]) / au2angstrom
TOY_LIPID_BOND_K = 60.0 * kcalmol2au * au2angstrom**2
TOY_LIPID_ANGLE_K = 8.0 * kcalmol2au
TOY_LIPID_TORSION_BARRIER = 0.15 * kcalmol2au


@dataclass
class MembraneEmbeddingSnapshot:
    """Point-charge environment extracted from one membrane snapshot.

    Coordinates are in Bohr and charges are in electron-charge units, matching
    :func:`pyqed.qchem.embed_point_charges`.
    """

    qm_indices: np.ndarray
    mm_indices: np.ndarray
    qm_coords: np.ndarray
    charge_coords: np.ndarray
    charges: np.ndarray
    owners: np.ndarray
    shifts: np.ndarray
    center: np.ndarray
    membrane_normal: np.ndarray
    depth: float
    cutoff: Optional[float] = None
    charge_array: str = "charges"


def toy_lipid(
    head_position=(0.0, 0.0, 0.0),
    leaflet=1,
    bead_spacing=4.0 / au2angstrom,
    molecule_id=0,
    calculator=False,
    **calculator_kwargs,
):
    """Build one five-bead amphiphile in atomic units.

    ``leaflet=1`` points the tail toward lower ``z``; ``leaflet=-1`` points it
    toward higher ``z``.  The molecule is neutral and intentionally compact.
    """
    sign = 1 if leaflet >= 0 else -1
    head_position = np.asarray(head_position, dtype=float)
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -sign * bead_spacing],
            [0.25 * bead_spacing, 0.0, -sign * 2.0 * bead_spacing],
            [0.25 * bead_spacing, 0.20 * bead_spacing, -sign * 3.0 * bead_spacing],
            [0.05 * bead_spacing, 0.35 * bead_spacing, -sign * 4.0 * bead_spacing],
        ]
    )
    positions = [head_position + xyz for xyz in local]
    bond_lengths = [
        _distance(positions[0], positions[1]),
        _distance(positions[1], positions[2]),
        _distance(positions[2], positions[3]),
        _distance(positions[3], positions[4]),
    ]
    angles = [
        _angle_degrees(positions[0], positions[1], positions[2]),
        _angle_degrees(positions[1], positions[2], positions[3]),
        _angle_degrees(positions[2], positions[3], positions[4]),
    ]

    topology = Topology(
        bonds=[
            (0, 1, TOY_LIPID_BOND_K, bond_lengths[0]),
            (1, 2, TOY_LIPID_BOND_K, bond_lengths[1]),
            (2, 3, TOY_LIPID_BOND_K, bond_lengths[2]),
            (3, 4, TOY_LIPID_BOND_K, bond_lengths[3]),
        ],
        angles=[
            (0, 1, 2, TOY_LIPID_ANGLE_K, angles[0]),
            (1, 2, 3, TOY_LIPID_ANGLE_K, angles[1]),
            (2, 3, 4, TOY_LIPID_ANGLE_K, angles[2]),
        ],
        torsions=[
            (0, 1, 2, 3, TOY_LIPID_TORSION_BARRIER, 1, 180.0),
            (1, 2, 3, 4, TOY_LIPID_TORSION_BARRIER, 1, 180.0),
        ],
        charges=TOY_LIPID_CHARGES,
        lj_epsilon=TOY_LIPID_LJ_EPSILON,
        lj_sigma=TOY_LIPID_LJ_SIGMA,
        molecule_ids=np.full(len(TOY_LIPID_SYMBOLS), int(molecule_id)),
    )
    atoms = Atoms([[symbol, tuple(xyz)] for symbol, xyz in zip(TOY_LIPID_SYMBOLS, positions)])
    atoms.topology = topology
    atoms.set_array("charges", topology.charges, float, ())
    atoms.set_array("lj_epsilon", topology.lj_epsilon, float, ())
    atoms.set_array("lj_sigma", topology.lj_sigma, float, ())
    atoms.set_array("molecule_ids", topology.molecule_ids, int, ())
    atoms.set_array("leaflets", np.full(len(atoms), sign), int, ())
    atoms.set_array("bead_names", np.asarray(["HEAD", "GLY", "T1", "T2", "T3"]), str, ())
    if calculator:
        atoms.calc = _mm_from_topology(topology, **calculator_kwargs)
    return atoms


def lipid_bilayer(
    nx=2,
    ny=2,
    area_per_lipid=60.0,
    thickness=36.0,
    water_padding=18.0,
    bead_spacing=4.0,
    pbc=True,
    calculator=True,
    coulomb_method="pme",
    coulomb_cutoff=10.0,
    lj_cutoff=10.0,
    pme_mesh=(24, 24, 32),
    seed=None,
):
    """Build a two-leaflet toy lipid bilayer.

    Public geometric arguments are in Angstrom and converted to atomic units.
    """
    nx = int(nx)
    ny = int(ny)
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")
    spacing_xy = np.sqrt(float(area_per_lipid)) / au2angstrom
    lx = nx * spacing_xy
    ly = ny * spacing_xy
    thickness_bohr = float(thickness) / au2angstrom
    lz = (float(thickness) + 2.0 * float(water_padding)) / au2angstrom
    center_z = 0.5 * lz
    head_offset = 0.5 * thickness_bohr
    bead_spacing_bohr = float(bead_spacing) / au2angstrom
    rng = np.random.default_rng(seed)

    lipids = []
    leaflets = []
    molecule_id = 0
    for leaflet in (1, -1):
        head_z = center_z + leaflet * head_offset
        for ix in range(nx):
            for iy in range(ny):
                jitter = rng.uniform(-0.08, 0.08, size=2) * spacing_xy if seed is not None else 0.0
                x = (ix + 0.5) * spacing_xy + (jitter[0] if seed is not None else 0.0)
                y = (iy + 0.5) * spacing_xy + (jitter[1] if seed is not None else 0.0)
                lipid = toy_lipid(
                    (x, y, head_z),
                    leaflet=leaflet,
                    bead_spacing=bead_spacing_bohr,
                    molecule_id=molecule_id,
                    calculator=False,
                )
                lipids.append(lipid)
                leaflets.extend([leaflet] * len(lipid))
                molecule_id += 1

    cutoff = float(coulomb_cutoff) / au2angstrom
    lj_cutoff_bohr = float(lj_cutoff) / au2angstrom
    system = combine_systems(
        lipids,
        cell=(lx, ly, lz),
        pbc=pbc,
        calculator=calculator,
        coulomb_method=coulomb_method,
        coulomb_cutoff=cutoff,
        lj_cutoff=lj_cutoff_bohr,
        pme_mesh=pme_mesh,
    )
    system.set_array("leaflets", np.asarray(leaflets, dtype=int), int, ())
    system.membrane = {
        "kind": "toy_bilayer",
        "lipids_per_leaflet": nx * ny,
        "total_lipids": 2 * nx * ny,
        "atoms_per_lipid": len(TOY_LIPID_SYMBOLS),
        "area_per_lipid_angstrom2": area_per_lipid,
        "thickness_angstrom": thickness,
        "water_padding_angstrom": water_padding,
        "center_z": center_z,
        "head_z": (center_z + head_offset, center_z - head_offset),
    }
    return system


def solvate_membrane(
    membrane,
    spacing=3.2,
    min_distance=2.4,
    max_waters_per_side=None,
    rigid=True,
    seed=None,
    calculator=True,
    coulomb_method="pme",
    coulomb_cutoff=10.0,
    lj_cutoff=10.0,
    pme_mesh=(24, 24, 32),
):
    """Add TIP3P water slabs above and below a membrane."""
    lengths = np.asarray(membrane.get_cell().lengths(), dtype=float)
    if np.any(lengths <= 0.0):
        raise ValueError("membrane must have a finite orthorhombic cell.")
    metadata = getattr(membrane, "membrane", {})
    if "head_z" not in metadata:
        raise ValueError("membrane must come from lipid_bilayer or provide membrane metadata.")

    spacing_bohr = float(spacing) / au2angstrom
    min_distance_bohr = float(min_distance) / au2angstrom
    rng = np.random.default_rng(seed)
    upper_head_z, lower_head_z = metadata["head_z"]
    origins, rotations, side_labels = _water_slab_origins(
        lengths,
        lower_max=lower_head_z - min_distance_bohr,
        upper_min=upper_head_z + min_distance_bohr,
        spacing=spacing_bohr,
        max_waters_per_side=max_waters_per_side,
        rng=rng,
    )
    origins, rotations, side_labels = _reject_membrane_overlaps(
        origins,
        rotations,
        side_labels,
        membrane.get_positions(),
        min_distance_bohr,
    )
    waters = tip3p_waters(
        origins,
        cell=lengths,
        pbc=membrane.get_pbc(),
        calculator=False,
        rigid=rigid,
        rotations=rotations,
    )

    cutoff = float(coulomb_cutoff) / au2angstrom
    lj_cutoff_bohr = float(lj_cutoff) / au2angstrom
    system = combine_systems(
        [membrane, waters],
        cell=lengths,
        pbc=membrane.get_pbc(),
        calculator=calculator,
        coulomb_method=coulomb_method,
        coulomb_cutoff=cutoff,
        lj_cutoff=lj_cutoff_bohr,
        pme_mesh=pme_mesh,
    )
    lipid_leaflets = membrane.get_array("leaflets") if membrane.has("leaflets") else np.zeros(len(membrane), dtype=int)
    system.set_array("leaflets", np.concatenate([lipid_leaflets, np.zeros(len(waters), dtype=int)]), int, ())
    system.membrane = dict(metadata)
    system.solvation = {
        "kind": "membrane_water_slabs",
        "placed_waters": len(origins),
        "lower_waters": int(np.count_nonzero(side_labels < 0)),
        "upper_waters": int(np.count_nonzero(side_labels > 0)),
        "spacing_angstrom": spacing,
        "min_distance_angstrom": min_distance,
    }
    return system


def scale_molecule_centers(atoms, lateral_scale=1.0, normal_scale=1.0, array_name="molecule_ids"):
    """Scale molecule centers with the box while preserving internal geometry.

    This preconditions membrane boxes without stretching constrained bonds
    inside lipids, waters, or ions.  It is a deterministic setup helper, not a
    barostat.
    """
    scale = np.array([float(lateral_scale), float(lateral_scale), float(normal_scale)])
    if np.any(scale <= 0.0):
        raise ValueError("box scale factors must be positive.")
    if np.allclose(scale, 1.0):
        return np.ones(3)
    if not atoms.has(array_name):
        raise ValueError(f"atoms must define a '{array_name}' array for molecule-center scaling.")

    positions = atoms.get_positions()
    molecule_ids = atoms.get_array(array_name)
    scaled_positions = positions.copy()
    for molecule_id in np.unique(molecule_ids):
        mask = molecule_ids == molecule_id
        center = np.mean(positions[mask], axis=0)
        scaled_positions[mask] = center * scale + (positions[mask] - center)

    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        raise ValueError("molecule-center scaling requires positive orthorhombic cell lengths.")
    atoms.set_cell(np.diag(lengths * scale), scale_atoms=False)
    atoms.set_positions(scaled_positions)
    return scale


def detect_leaflets(atoms, head_indices=None, axis=2, assign=False):
    """Classify atoms into upper/lower membrane leaflets.

    If ``head_indices`` is provided, leaflet assignment is based on those
    reference atoms per molecule. Otherwise molecule centers are used when
    ``molecule_ids`` exist; as a fallback, atoms are split by their coordinate
    along ``axis``.
    """
    if atoms.has("leaflets"):
        labels = atoms.get_array("leaflets")
        if np.any(labels):
            return labels
    positions = atoms.get_positions()
    molecule_ids = atoms.get_array("molecule_ids") if atoms.has("molecule_ids") else None
    axis = int(axis)
    if molecule_ids is None:
        coordinates = positions[:, axis]
        midpoint = float(np.median(coordinates))
        labels = np.where(coordinates >= midpoint, 1, -1)
        if assign:
            atoms.set_array("leaflets", labels, int, ())
        return labels

    molecule_ids = np.asarray(molecule_ids, dtype=int)
    reference = np.arange(len(atoms)) if head_indices is None else np.asarray(head_indices, dtype=int)
    molecule_z = {}
    for molecule_id in np.unique(molecule_ids):
        indices = reference[molecule_ids[reference] == molecule_id]
        if len(indices) == 0:
            indices = np.nonzero(molecule_ids == molecule_id)[0]
        molecule_z[int(molecule_id)] = float(np.mean(positions[indices, axis]))
    midpoint = float(np.median(list(molecule_z.values())))
    labels = np.zeros(len(atoms), dtype=int)
    for molecule_id, z_value in molecule_z.items():
        labels[molecule_ids == molecule_id] = 1 if z_value >= midpoint else -1
    if assign:
        atoms.set_array("leaflets", labels, int, ())
    return labels


def area_per_lipid(atoms, unit="angstrom", lipids_per_leaflet=None, leaflet_labels=None):
    """Return membrane area per lipid for one leaflet."""
    metadata = getattr(atoms, "membrane", {})
    lipids_per_leaflet = metadata.get("lipids_per_leaflet") if lipids_per_leaflet is None else lipids_per_leaflet
    if lipids_per_leaflet is None and atoms.has("molecule_ids"):
        labels = detect_leaflets(atoms) if leaflet_labels is None else np.asarray(leaflet_labels, dtype=int)
        molecule_ids = atoms.get_array("molecule_ids")
        counts = []
        for sign in (-1, 1):
            leaflet_molecules = np.unique(molecule_ids[labels == sign])
            leaflet_molecules = leaflet_molecules[leaflet_molecules >= 0]
            if len(leaflet_molecules):
                counts.append(len(leaflet_molecules))
        if counts:
            lipids_per_leaflet = int(round(float(np.mean(counts))))
    if not lipids_per_leaflet:
        raise ValueError("atoms object does not carry membrane lipids_per_leaflet metadata.")
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    area = lengths[0] * lengths[1] / int(lipids_per_leaflet)
    if unit.lower() in {"angstrom", "angstrom2", "a2"}:
        return area * au2angstrom**2
    if unit.lower() in {"bohr", "bohr2", "au", "atomic"}:
        return area
    raise ValueError("unit must be 'angstrom' or 'bohr'.")


def leaflet_indices(atoms, leaflet):
    """Return atom indices assigned to a membrane leaflet."""
    leaflets = detect_leaflets(atoms)
    sign = 1 if leaflet >= 0 else -1
    return np.nonzero(leaflets == sign)[0]


def bilayer_thickness(atoms, head_indices=None, axis=2, unit="angstrom"):
    """Return the distance between upper/lower leaflet reference planes."""
    labels = detect_leaflets(atoms, head_indices=head_indices, axis=axis)
    positions = atoms.get_positions()
    reference = np.arange(len(atoms)) if head_indices is None else np.asarray(head_indices, dtype=int)
    upper = reference[labels[reference] > 0]
    lower = reference[labels[reference] < 0]
    if len(upper) == 0 or len(lower) == 0:
        raise ValueError("both leaflets must contain reference atoms.")
    thickness = abs(float(np.mean(positions[upper, axis]) - np.mean(positions[lower, axis])))
    if unit.lower() in {"angstrom", "a"}:
        return thickness * au2angstrom
    if unit.lower() in {"bohr", "au", "atomic"}:
        return thickness
    raise ValueError("unit must be 'angstrom' or 'bohr'.")


def tail_order_parameters(atoms, tail_pairs, axis=2):
    """Return deuterium-order-like orientation values for tail bond vectors.

    The returned value for each pair is ``0.5 * (3 cos(theta)^2 - 1)`` relative
    to the membrane normal along ``axis``.
    """
    positions = atoms.get_positions()
    axis = int(axis)
    normal = np.zeros(3)
    normal[axis] = 1.0
    values = []
    for i, j in tail_pairs:
        vector = positions[int(j)] - positions[int(i)]
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            values.append(np.nan)
            continue
        cosine = abs(float(np.dot(vector / norm, normal)))
        values.append(0.5 * (3.0 * cosine * cosine - 1.0))
    return np.asarray(values, dtype=float)


def membrane_summary(atoms, head_indices=None, tail_pairs=None, axis=2):
    """Return a compact dictionary of common membrane observables."""
    labels = detect_leaflets(atoms, head_indices=head_indices, axis=axis)
    summary = {
        "area_per_lipid_angstrom2": area_per_lipid(atoms, leaflet_labels=labels),
        "bilayer_thickness_angstrom": bilayer_thickness(
            atoms,
            head_indices=head_indices,
            axis=axis,
        ),
        "upper_atoms": int(np.count_nonzero(labels > 0)),
        "lower_atoms": int(np.count_nonzero(labels < 0)),
    }
    if tail_pairs is not None:
        order = tail_order_parameters(atoms, tail_pairs, axis=axis)
        summary["tail_order_mean"] = float(np.nanmean(order))
        summary["tail_order_values"] = order.tolist()
    return summary


def membrane_diagnostics(atoms, head_indices=None, tail_pairs=None, axis=2):
    """Return membrane observables plus basic simulation sanity checks."""
    positions = atoms.get_positions()
    summary = {
        "atoms": len(atoms),
        "finite_positions": bool(np.all(np.isfinite(positions))),
        "total_charge": float(np.sum(atoms.get_array("charges"))) if atoms.has("charges") else 0.0,
    }
    if atoms.has("molecule_ids"):
        summary["molecules"] = int(len(np.unique(atoms.get_array("molecule_ids"))))
    if atoms.calc is not None:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        summary.update(
            {
                "potential_hartree": float(energy),
                "finite_forces": bool(np.all(np.isfinite(forces))),
                "max_force_hartree_per_bohr": float(np.max(np.linalg.norm(forces, axis=1))),
            }
        )
    try:
        summary.update(membrane_summary(atoms, head_indices=head_indices, tail_pairs=tail_pairs, axis=axis))
    except ValueError as exc:
        summary["membrane_summary_error"] = str(exc)
    metadata = getattr(atoms, "solvation", None)
    if metadata:
        summary.update(
            {
                "placed_waters": int(metadata.get("placed_waters", 0)),
                "upper_waters": int(metadata.get("upper_waters", 0)),
                "lower_waters": int(metadata.get("lower_waters", 0)),
            }
        )
    ions = getattr(atoms, "ions", None)
    if ions:
        summary["placed_ions"] = len(ions.get("placed_ions", []))
    return summary


def membrane_analysis(atoms, head_indices=None, tail_pairs=None, axis=2):
    """Return a JSON-friendly membrane analysis report.

    This is intended for smoke tests and workflow artifacts: it combines the
    common membrane diagnostics with compact topology, leaflet, tail-order, and
    box summaries.
    """
    report = dict(membrane_diagnostics(atoms, head_indices=head_indices, tail_pairs=tail_pairs, axis=axis))
    labels = detect_leaflets(atoms, head_indices=head_indices, axis=axis)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    molecule_ids = atoms.get_array("molecule_ids") if atoms.has("molecule_ids") else np.arange(len(atoms))
    head_indices = np.asarray(head_indices, dtype=int) if head_indices is not None else np.arange(len(atoms))
    topology = getattr(atoms, "topology", None)
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)

    report["cell_lengths_angstrom"] = (lengths * au2angstrom).tolist()
    report["cell_volume_angstrom3"] = float(np.prod(lengths * au2angstrom)) if lengths.shape == (3,) else 0.0
    if topology is not None:
        report["topology"] = {
            "bonds": int(len(getattr(topology, "bonds", []))),
            "angles": int(len(getattr(topology, "angles", []))),
            "torsions": int(len(getattr(topology, "torsions", []))),
            "impropers": int(len(getattr(topology, "impropers", []))),
            "cmaps": int(len(getattr(topology, "cmaps", []))),
            "lj_pair_scales": int(len(getattr(topology, "lj_pair_scales", {}))),
            "coulomb_pair_scales": int(len(getattr(topology, "coulomb_pair_scales", {}))),
        }

    leaflet_report = {}
    for sign, name in ((1, "upper"), (-1, "lower")):
        mask = labels == sign
        molecules = np.unique(molecule_ids[mask]) if np.any(mask) else np.asarray([], dtype=int)
        heads = head_indices[labels[head_indices] == sign] if len(head_indices) else np.asarray([], dtype=int)
        leaflet_report[name] = {
            "atoms": int(np.count_nonzero(mask)),
            "molecules": int(len(molecules)),
            "z_mean_angstrom": float(np.mean(positions[mask, axis]) * au2angstrom) if np.any(mask) else None,
            "head_z_mean_angstrom": float(np.mean(positions[heads, axis]) * au2angstrom) if len(heads) else None,
        }
    report["leaflets"] = leaflet_report

    if tail_pairs is not None:
        order = tail_order_parameters(atoms, tail_pairs, axis=axis)
        finite = order[np.isfinite(order)]
        report.pop("tail_order_mean", None)
        report.pop("tail_order_values", None)
        report["tail_order"] = {
            "count": int(len(order)),
            "finite_count": int(len(finite)),
            "mean": float(np.mean(finite)) if len(finite) else None,
            "std": float(np.std(finite)) if len(finite) else None,
            "min": float(np.min(finite)) if len(finite) else None,
            "max": float(np.max(finite)) if len(finite) else None,
        }

    if atoms.has("residue_names"):
        names = np.char.upper(np.asarray(atoms.get_array("residue_names"), dtype=str))
        water = np.isin(names, ["HOH", "TIP3", "TP3", "WAT", "SOL", "H2O"])
        report["water_atoms"] = int(np.count_nonzero(water))
        if atoms.has("molecule_ids") and np.any(water):
            report["water_molecules"] = int(len(np.unique(molecule_ids[water])))
    return report


def membrane_embedding_snapshot(
    atoms,
    qm_indices,
    mm_indices=None,
    charge_array="charges",
    cutoff=None,
    embedding_pbc="nearest",
    min_qm_distance=None,
    cap_charge_distance=None,
    axis=2,
):
    """Extract MM point charges around a QM molecule in a membrane snapshot.

    This is the bridge we need for molecular CD in membranes: the chromophore
    stays quantum mechanical, while lipids, water, and ions become an explicit
    electrostatic environment.

    Parameters
    ----------
    atoms
        Full membrane system with positions in Bohr and a charge array.
    qm_indices
        Atom indices belonging to the QM chromophore.
    mm_indices
        Optional environment indices.  Defaults to all non-QM atoms.
    charge_array
        Name of the atom array containing MM point charges.
    cutoff
        Optional maximum distance from any QM atom to retain an MM charge.
    embedding_pbc
        ``'none'`` keeps raw coordinates, ``'nearest'`` maps each MM charge to
        the nearest periodic image relative to the QM center, and ``'images'``
        expands periodic images within ``cutoff``.
    min_qm_distance
        Optional exclusion distance; charges closer than this to any QM atom are
        dropped.
    cap_charge_distance
        Optional lower bound on charge-QM distance; charges inside this distance
        are moved radially away from the nearest QM atom instead of discarded.
    axis
        Membrane-normal axis used for the signed depth metadata.
    """

    if not atoms.has(charge_array):
        raise ValueError(f"atoms object does not have charge array {charge_array!r}.")

    qm_indices = np.asarray(qm_indices, dtype=int).reshape(-1)
    if qm_indices.size == 0:
        raise ValueError("qm_indices must contain at least one atom.")

    natoms = len(atoms)
    if np.any(qm_indices < 0) or np.any(qm_indices >= natoms):
        raise IndexError("qm_indices contains an atom index outside the system.")

    if mm_indices is None:
        qm_mask = np.zeros(natoms, dtype=bool)
        qm_mask[qm_indices] = True
        mm_indices = np.nonzero(~qm_mask)[0]
    else:
        mm_indices = np.asarray(mm_indices, dtype=int).reshape(-1)
        if np.any(mm_indices < 0) or np.any(mm_indices >= natoms):
            raise IndexError("mm_indices contains an atom index outside the system.")

    positions = np.asarray(atoms.get_positions(), dtype=float)
    charges_all = np.asarray(atoms.get_array(charge_array), dtype=float)
    qm_coords = positions[qm_indices]
    mm_coords = positions[mm_indices]
    charges = charges_all[mm_indices]
    center = np.mean(qm_coords, axis=0)

    coords, owner_local, shifts = _embedding_charge_images(
        atoms,
        qm_coords,
        mm_coords,
        cutoff=cutoff,
        embedding_pbc=embedding_pbc,
        center=center,
    )
    charges = charges[owner_local]
    owners = mm_indices[owner_local]

    if coords.size:
        distances = _min_distances_to_qm(coords, qm_coords)
    else:
        distances = np.empty(0, dtype=float)

    if cutoff is not None:
        keep = distances <= float(cutoff)
        coords, charges, owners, shifts, distances = _select_charges(
            keep,
            coords,
            charges,
            owners,
            shifts,
            distances,
        )

    if min_qm_distance is not None:
        keep = distances >= float(min_qm_distance)
        coords, charges, owners, shifts, distances = _select_charges(
            keep,
            coords,
            charges,
            owners,
            shifts,
            distances,
        )

    if cap_charge_distance is not None and coords.size:
        coords = _cap_close_charge_distances(coords, qm_coords, float(cap_charge_distance))
        distances = _min_distances_to_qm(coords, qm_coords)

    normal = np.zeros(3, dtype=float)
    normal[int(axis)] = 1.0
    midplane = _membrane_midplane(atoms, axis=axis)
    depth = float(center[int(axis)] - midplane)

    return MembraneEmbeddingSnapshot(
        qm_indices=qm_indices.copy(),
        mm_indices=np.asarray(owners, dtype=int).copy(),
        qm_coords=qm_coords.copy(),
        charge_coords=np.asarray(coords, dtype=float).reshape(-1, 3),
        charges=np.asarray(charges, dtype=float).reshape(-1),
        owners=np.asarray(owners, dtype=int).reshape(-1),
        shifts=np.asarray(shifts, dtype=float).reshape(-1, 3),
        center=center.copy(),
        membrane_normal=normal,
        depth=depth,
        cutoff=None if cutoff is None else float(cutoff),
        charge_array=str(charge_array),
    )


def _mm_from_topology(topology, **kwargs):
    return MM(
        bonds=topology.bonds,
        angles=topology.angles,
        torsions=topology.torsions,
        impropers=topology.impropers,
        cmaps=getattr(topology, "cmaps", []),
        cmap_grids=getattr(topology, "cmap_grids", []),
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
        **kwargs,
    )


def _embedding_charge_images(atoms, qm_coords, mm_coords, cutoff, embedding_pbc, center):
    key = "nearest" if embedding_pbc is None else str(embedding_pbc).lower()
    owners = np.arange(len(mm_coords), dtype=int)
    shifts = np.zeros_like(mm_coords, dtype=float)
    if key in {"none", "false", "off"} or len(mm_coords) == 0:
        return mm_coords.copy(), owners, shifts

    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if not np.any(pbc):
        return mm_coords.copy(), owners, shifts
    _validate_orthorhombic_cell(cell, pbc)

    if key == "nearest":
        coords = np.asarray(
            [center + minimum_image(coord - center, cell, pbc) for coord in mm_coords],
            dtype=float,
        )
        return coords, owners, coords - mm_coords

    if key != "images":
        raise ValueError("embedding_pbc must be 'none', 'nearest', or 'images'.")
    if cutoff is None or float(cutoff) <= 0.0:
        raise ValueError("cutoff must be positive when embedding_pbc='images'.")

    lengths = np.diag(cell)
    ranges = [
        range(-int(np.ceil(float(cutoff) / lengths[axis])) - 1,
              int(np.ceil(float(cutoff) / lengths[axis])) + 2)
        if pbc[axis]
        else range(0, 1)
        for axis in range(3)
    ]
    coords = []
    image_owners = []
    image_shifts = []
    cutoff_value = float(cutoff)
    for local_owner, coord in enumerate(mm_coords):
        for ix in ranges[0]:
            for iy in ranges[1]:
                for iz in ranges[2]:
                    shift = np.array([ix, iy, iz], dtype=float) * lengths
                    image = coord + shift
                    if np.min(np.linalg.norm(image[None, :] - qm_coords, axis=1)) <= cutoff_value:
                        coords.append(image)
                        image_owners.append(local_owner)
                        image_shifts.append(shift)
    return (
        np.asarray(coords, dtype=float).reshape(-1, 3),
        np.asarray(image_owners, dtype=int),
        np.asarray(image_shifts, dtype=float).reshape(-1, 3),
    )


def _validate_orthorhombic_cell(cell, pbc):
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("Periodic membrane embedding requires a 3x3 cell.")
    offdiag = cell - np.diag(np.diag(cell))
    if np.any(np.abs(offdiag) > 1.0e-12):
        raise ValueError("Periodic membrane embedding currently requires an orthorhombic cell.")
    lengths = np.diag(cell)
    if np.any(lengths[pbc] <= 0.0):
        raise ValueError("Periodic membrane embedding requires nonzero periodic cell lengths.")


def _min_distances_to_qm(coords, qm_coords):
    if len(coords) == 0:
        return np.empty(0, dtype=float)
    deltas = coords[:, None, :] - qm_coords[None, :, :]
    return np.min(np.linalg.norm(deltas, axis=-1), axis=1)


def _select_charges(keep, coords, charges, owners, shifts, distances):
    keep = np.asarray(keep, dtype=bool)
    return coords[keep], charges[keep], owners[keep], shifts[keep], distances[keep]


def _cap_close_charge_distances(coords, qm_coords, cap_distance):
    if cap_distance <= 0.0:
        raise ValueError("cap_charge_distance must be positive.")
    adjusted = np.asarray(coords, dtype=float).copy()
    deltas = adjusted[:, None, :] - qm_coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    nearest = np.argmin(distances, axis=1)
    for charge_index, atom_index in enumerate(nearest):
        distance = float(distances[charge_index, atom_index])
        if distance >= cap_distance:
            continue
        direction = adjusted[charge_index] - qm_coords[atom_index]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm
        adjusted[charge_index] = qm_coords[atom_index] + cap_distance * direction
    return adjusted


def _membrane_midplane(atoms, axis=2):
    axis = int(axis)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    try:
        labels = detect_leaflets(atoms, axis=axis)
        upper = positions[labels > 0, axis]
        lower = positions[labels < 0, axis]
        if len(upper) and len(lower):
            return 0.5 * (float(np.mean(upper)) + float(np.mean(lower)))
    except ValueError:
        pass
    return float(np.median(positions[:, axis]))


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _angle_degrees(a, b, c):
    ba = np.asarray(a) - np.asarray(b)
    bc = np.asarray(c) - np.asarray(b)
    cosine = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _water_slab_origins(lengths, lower_max, upper_min, spacing, max_waters_per_side, rng):
    axes = [
        np.arange(0.5 * spacing, lengths[0], spacing),
        np.arange(0.5 * spacing, lengths[1], spacing),
    ]
    lower_z = np.arange(0.5 * spacing, max(lower_max, 0.0), spacing)
    upper_z = np.arange(max(upper_min, 0.0), lengths[2] - 0.5 * spacing, spacing)
    origins = []
    labels = []
    for z_values, label in ((lower_z, -1), (upper_z, 1)):
        side = []
        for x in axes[0]:
            for y in axes[1]:
                for z in z_values:
                    side.append((x, y, z))
        if side:
            rng.shuffle(side)
        if max_waters_per_side is not None:
            side = side[: int(max_waters_per_side)]
        origins.extend(side)
        labels.extend([label] * len(side))
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    rotations = np.asarray([_random_rotation_matrix(rng) for _ in origins], dtype=float).reshape(-1, 3, 3)
    labels = np.asarray(labels, dtype=int)
    return origins, rotations, labels


def _reject_membrane_overlaps(origins, rotations, labels, membrane_positions, min_distance):
    local = _tip3p_local_positions()
    accepted_origins = []
    accepted_rotations = []
    accepted_labels = []
    min_distance2 = min_distance * min_distance
    for origin, rotation, label in zip(origins, rotations, labels):
        water_positions = origin + local @ rotation.T
        deltas = water_positions[:, None, :] - membrane_positions[None, :, :]
        if np.any(np.sum(deltas * deltas, axis=-1) < min_distance2):
            continue
        accepted_origins.append(origin)
        accepted_rotations.append(rotation)
        accepted_labels.append(label)
    return (
        np.asarray(accepted_origins, dtype=float).reshape(-1, 3),
        np.asarray(accepted_rotations, dtype=float).reshape(-1, 3, 3),
        np.asarray(accepted_labels, dtype=int),
    )


def _tip3p_local_positions():
    theta = np.deg2rad(TIP3P_HOH_ANGLE)
    return np.array(
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


def _random_rotation_matrix(rng):
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ]
    )
