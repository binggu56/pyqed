"""Native protein-membrane seed construction helpers."""

from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

import numpy as np

from pyqed.protein import parse_pdb_atoms
from pyqed.units import au2angstrom

from .atoms import Atoms
from .composition import residue_composition
from .ions import monatomic_ions
from .io import write_pdb
from .lipids import lipid_from_template, lipid_template
from .membrane import solvate_membrane
from .solvation import AVOGADRO, combine_systems
from .topology import Topology


def read_protein_pdb(
    pdb_path,
    *,
    model=1,
    include_hetero=False,
    protein_net_charge=0.0,
):
    """Read PDB ``ATOM`` records into an MD ``Atoms`` object.

    Coordinates are converted from Angstrom to Bohr.  The returned object is a
    geometry seed: protein bonded and nonbonded parameters are placeholders
    until the seed is passed through a real force-field importer.
    """
    records = parse_pdb_atoms(pdb_path, include_hetero=include_hetero, model=model)
    if not records:
        raise ValueError("protein PDB contains no readable ATOM records.")

    symbols = [record.element for record in records]
    positions = np.asarray([record.coord_angstrom for record in records], dtype=float) / au2angstrom
    charges = _protein_placeholder_charges(len(records), protein_net_charge)
    protein = Atoms([[symbol, tuple(position)] for symbol, position in zip(symbols, positions)])
    protein.topology = Topology(
        charges=charges,
        lj_epsilon=np.zeros(len(protein)),
        lj_sigma=np.zeros(len(protein)),
        molecule_ids=np.zeros(len(protein), dtype=int),
        atom_types=np.asarray([symbol.upper() for symbol in symbols], dtype=str),
        atom_names=np.asarray([record.name for record in records], dtype=str),
    )
    protein.set_array("charges", protein.topology.charges, float, ())
    protein.set_array("lj_epsilon", protein.topology.lj_epsilon, float, ())
    protein.set_array("lj_sigma", protein.topology.lj_sigma, float, ())
    protein.set_array("molecule_ids", protein.topology.molecule_ids, int, ())
    protein.set_array("atom_types", protein.topology.atom_types, str, ())
    protein.set_array("atom_names", [record.name for record in records], str, ())
    protein.set_array("residue_names", [record.residue_name for record in records], str, ())
    protein.set_array("residue_ids", [record.residue_id for record in records], int, ())
    protein.set_array("chain_ids", [record.chain_id or "A" for record in records], str, ())
    protein.protein = {
        "source": str(pdb_path),
        "model": int(model),
        "include_hetero": bool(include_hetero),
        "placeholder_net_charge": float(protein_net_charge),
    }
    return protein


def protein_membrane_seed(
    protein_pdb,
    *,
    lipid="DPPC-OPENMM",
    nx=None,
    ny=None,
    lipids_per_leaflet=16,
    area_per_lipid=64.0,
    thickness=38.0,
    water_padding=20.0,
    waters_per_side=None,
    water_spacing=3.2,
    salt_molar=0.0,
    protein_net_charge=0.0,
    neutralize=True,
    seed=None,
    protein_xy_padding=12.0,
    lipid_protein_min_distance=2.2,
    water_min_distance=2.4,
    ion_min_distance=2.5,
    ion_region="water",
    flip_z=False,
    output_pdb=None,
    calculator=False,
    coulomb_method="pme",
    coulomb_cutoff=10.0,
    lj_cutoff=10.0,
    pme_mesh=(32, 32, 48),
    openmm_lipid_source=None,
):
    """Create a native geometry seed for a protein embedded in a lipid membrane.

    Public geometry arguments are in Angstrom.  The input protein is assumed to
    already use the membrane normal as ``z``; v1 only centers and optionally
    flips the protein.
    """
    rng = np.random.default_rng(seed)
    template = lipid_template(lipid, openmm_source=openmm_lipid_source)
    protein = read_protein_pdb(protein_pdb, protein_net_charge=protein_net_charge)

    nx, ny = _resolve_leaflet_grid(
        protein,
        nx=nx,
        ny=ny,
        lipids_per_leaflet=lipids_per_leaflet,
        area_per_lipid=area_per_lipid,
        protein_xy_padding=protein_xy_padding,
    )
    spacing = sqrt(float(area_per_lipid)) / au2angstrom
    lx = nx * spacing
    ly = ny * spacing
    protein_extent_z = float(np.ptp(protein.get_positions()[:, 2]) * au2angstrom)
    lz_angstrom = max(float(thickness) + 2.0 * float(water_padding), protein_extent_z + 2.0 * float(water_padding))
    lz = lz_angstrom / au2angstrom
    cell = np.asarray([lx, ly, lz], dtype=float)
    center_z = 0.5 * lz
    head_offset = 0.5 * float(thickness) / au2angstrom

    _center_protein(protein, cell, flip_z=flip_z)
    lipids, removed_lipids = _protein_excluded_lipids(
        protein,
        template,
        nx=nx,
        ny=ny,
        spacing=spacing,
        lx=lx,
        ly=ly,
        center_z=center_z,
        head_offset=head_offset,
        min_distance=lipid_protein_min_distance / au2angstrom,
        rng=rng,
    )
    dry_seed = _combine_seed_components(
        protein,
        lipids,
        cell,
        calculator=False,
        coulomb_method=coulomb_method,
        coulomb_cutoff=coulomb_cutoff,
        lj_cutoff=lj_cutoff,
        pme_mesh=pme_mesh,
        membrane_metadata={
            "kind": "protein_membrane_seed",
            "template": template.name,
            "residue_name": template.residue_name,
            "forcefield": template.forcefield,
            "validated": template.validated,
            "requested_grid": [int(nx), int(ny)],
            "requested_lipid_sites_per_leaflet": int(nx * ny),
            "placed_lipids": int(len(lipids)),
            "removed_lipids": int(removed_lipids),
            "area_per_lipid_angstrom2": float(area_per_lipid),
            "thickness_angstrom": float(thickness),
            "water_padding_angstrom": float(water_padding),
            "center_z": center_z,
            "head_z": (center_z + head_offset, center_z - head_offset),
            "head_atom_names": template.head_atom_names,
            "tail_pairs": template.tail_pairs,
        },
    )
    hydrated = solvate_membrane(
        dry_seed,
        spacing=water_spacing,
        min_distance=water_min_distance,
        max_waters_per_side=waters_per_side,
        rigid=True,
        seed=seed,
        calculator=False,
        coulomb_method=coulomb_method,
        coulomb_cutoff=coulomb_cutoff,
        lj_cutoff=lj_cutoff,
        pme_mesh=pme_mesh,
    )
    _restore_component_arrays_after_hydration(dry_seed, hydrated)

    ion_symbols = _ion_symbols_for_seed(
        hydrated,
        salt_molar=salt_molar,
        neutralize=neutralize,
    )
    final = hydrated
    if ion_symbols:
        before = len(final)
        old_chain_ids = final.get_array("chain_ids") if final.has("chain_ids") else np.full(before, "A")
        old_leaflets = final.get_array("leaflets") if final.has("leaflets") else np.zeros(before, dtype=int)
        final = add_ions_to_seed(
            final,
            ions=ion_symbols,
            region=ion_region,
            min_distance=ion_min_distance / au2angstrom,
            seed=seed,
            calculator=calculator,
            coulomb_method=coulomb_method,
            coulomb_cutoff=coulomb_cutoff / au2angstrom,
            lj_cutoff=lj_cutoff / au2angstrom,
            pme_mesh=pme_mesh,
        )
        final.set_array("chain_ids", np.concatenate([old_chain_ids, np.full(len(final) - before, "I")]), str, ())
        final.set_array("leaflets", np.concatenate([old_leaflets, np.zeros(len(final) - before, dtype=int)]), int, ())
    elif calculator:
        old_chain_ids = final.get_array("chain_ids") if final.has("chain_ids") else None
        old_leaflets = final.get_array("leaflets") if final.has("leaflets") else None
        old_membrane = dict(getattr(final, "membrane", {}))
        old_solvation = dict(getattr(final, "solvation", {}))
        final = combine_systems(
            [final],
            cell=cell,
            pbc=True,
            calculator=True,
            coulomb_method=coulomb_method,
            coulomb_cutoff=coulomb_cutoff / au2angstrom,
            lj_cutoff=lj_cutoff / au2angstrom,
            pme_mesh=pme_mesh,
        )
        if old_chain_ids is not None:
            final.set_array("chain_ids", old_chain_ids, str, ())
        if old_leaflets is not None:
            final.set_array("leaflets", old_leaflets, int, ())
        final.membrane = old_membrane
        final.solvation = old_solvation

    final.seed_builder = _seed_summary(
        final,
        protein,
        nx=nx,
        ny=ny,
        lipid=template,
        removed_lipids=removed_lipids,
        salt_molar=salt_molar,
        ion_symbols=ion_symbols,
        flip_z=flip_z,
    )
    final.membrane = dict(getattr(final, "membrane", {}))
    final.membrane.update(final.seed_builder["membrane"])
    if output_pdb is not None:
        write_pdb(final, output_pdb)
    return final


def _protein_placeholder_charges(natoms, net_charge):
    charges = np.zeros(int(natoms), dtype=float)
    if natoms and float(net_charge) != 0.0:
        charges[:] = float(net_charge) / int(natoms)
    return charges


def _resolve_leaflet_grid(protein, *, nx, ny, lipids_per_leaflet, area_per_lipid, protein_xy_padding):
    spacing_angstrom = sqrt(float(area_per_lipid))
    positions_angstrom = protein.get_positions() * au2angstrom
    extent = np.ptp(positions_angstrom[:, :2], axis=0)
    min_nx = max(1, int(ceil((extent[0] + 2.0 * float(protein_xy_padding)) / spacing_angstrom)))
    min_ny = max(1, int(ceil((extent[1] + 2.0 * float(protein_xy_padding)) / spacing_angstrom)))
    if nx is None and ny is None:
        sites = max(int(lipids_per_leaflet or 1), 1)
        nx = int(ceil(sqrt(sites)))
        ny = int(ceil(sites / nx))
    elif nx is None:
        ny = int(ny)
        nx = int(ceil(max(int(lipids_per_leaflet or ny), 1) / ny))
    elif ny is None:
        nx = int(nx)
        ny = int(ceil(max(int(lipids_per_leaflet or nx), 1) / nx))
    nx = max(int(nx), min_nx)
    ny = max(int(ny), min_ny)
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")
    return nx, ny


def _center_protein(protein, cell, *, flip_z=False):
    positions = protein.get_positions()
    bbox_center = 0.5 * (np.min(positions, axis=0) + np.max(positions, axis=0))
    centered = positions - bbox_center
    if flip_z:
        centered[:, 2] *= -1.0
    protein.set_positions(centered + 0.5 * np.asarray(cell, dtype=float))
    protein.set_cell(cell)
    protein.set_pbc(True)


def _protein_excluded_lipids(
    protein,
    template,
    *,
    nx,
    ny,
    spacing,
    lx,
    ly,
    center_z,
    head_offset,
    min_distance,
    rng,
):
    protein_positions = protein.get_positions()
    lipids = []
    removed = 0
    molecule_id = 1
    for leaflet in (1, -1):
        leaflet_shift = 0.0 if leaflet > 0 else 0.5
        for ix in range(nx):
            for iy in range(ny):
                jitter = rng.uniform(-0.05, 0.05, size=2) * spacing
                origin = np.array(
                    [
                        ((ix + 0.5 + leaflet_shift) % nx) * spacing + jitter[0],
                        ((iy + 0.5 + leaflet_shift) % ny) * spacing + jitter[1],
                        center_z + leaflet * head_offset,
                    ],
                    dtype=float,
                )
                origin[:2] = np.mod(origin[:2], [lx, ly])
                rotation = rng.uniform(0.0, 2.0 * np.pi)
                lipid_atoms = lipid_from_template(
                    template,
                    origin=origin,
                    leaflet=leaflet,
                    molecule_id=molecule_id,
                    residue_id=molecule_id,
                    rotation=rotation,
                )
                if _has_overlap(lipid_atoms.get_positions(), protein_positions, min_distance):
                    removed += 1
                    continue
                lipids.append(lipid_atoms)
                molecule_id += 1
    return lipids, removed


def _combine_seed_components(
    protein,
    lipids,
    cell,
    *,
    calculator,
    coulomb_method,
    coulomb_cutoff,
    lj_cutoff,
    pme_mesh,
    membrane_metadata,
):
    systems = [protein, *lipids]
    combined = combine_systems(
        systems,
        cell=cell,
        pbc=True,
        calculator=calculator,
        coulomb_method=coulomb_method,
        coulomb_cutoff=coulomb_cutoff / au2angstrom,
        lj_cutoff=lj_cutoff / au2angstrom,
        pme_mesh=pme_mesh,
    )
    chain_ids = [protein.get_array("chain_ids")]
    leaflets = [np.zeros(len(protein), dtype=int)]
    for lipid_atoms in lipids:
        chain_ids.append(np.full(len(lipid_atoms), "U" if lipid_atoms.get_array("leaflets")[0] > 0 else "L"))
        leaflets.append(lipid_atoms.get_array("leaflets"))
    combined.set_array("chain_ids", np.concatenate(chain_ids), str, ())
    combined.set_array("leaflets", np.concatenate(leaflets), int, ())
    combined.membrane = dict(membrane_metadata)
    return combined


def _restore_component_arrays_after_hydration(dry_seed, hydrated):
    n_dry = len(dry_seed)
    water_atoms = len(hydrated) - n_dry
    chain_ids = np.concatenate(
        [
            dry_seed.get_array("chain_ids"),
            np.full(water_atoms, "W"),
        ]
    )
    hydrated.set_array("chain_ids", chain_ids, str, ())


def _ion_symbols_for_seed(atoms, *, salt_molar, neutralize):
    symbols = []
    charge = float(np.sum(atoms.get_array("charges"))) if atoms.has("charges") else 0.0
    if neutralize:
        nearest = int(round(charge))
        if nearest > 0:
            symbols.extend(["Cl"] * nearest)
        elif nearest < 0:
            symbols.extend(["Na"] * abs(nearest))
    salt_pairs = _salt_pairs_for_box(atoms, salt_molar)
    symbols.extend(["Na", "Cl"] * salt_pairs)
    return tuple(symbols)


def _salt_pairs_for_box(atoms, salt_molar):
    salt_molar = float(salt_molar)
    if salt_molar <= 0.0:
        return 0
    lengths_angstrom = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    volume_liter = float(np.prod(lengths_angstrom)) * 1.0e-27
    return max(int(round(salt_molar * volume_liter * AVOGADRO)), 0)


def _seed_summary(final, protein, *, nx, ny, lipid, removed_lipids, salt_molar, ion_symbols, flip_z):
    lengths = np.asarray(final.get_cell().lengths(), dtype=float) * au2angstrom
    composition = residue_composition(final)
    placed_lipids = int(composition.get("lipid_residues", 0))
    return {
        "workflow": "protein_membrane_seed",
        "atoms": int(len(final)),
        "box_lengths_angstrom": [float(value) for value in lengths],
        "composition": composition,
        "protein": {
            "atoms": int(len(protein)),
            "residues": int(residue_composition(protein).get("protein_residues", 0)),
            "chains": int(residue_composition(protein).get("protein_chains", 0)),
            "flip_z": bool(flip_z),
        },
        "membrane": {
            "lipid": lipid.name,
            "grid": [int(nx), int(ny)],
            "requested_lipid_sites_per_leaflet": int(nx * ny),
            "placed_lipids": placed_lipids,
            "removed_lipids": int(removed_lipids),
        },
        "solvation": dict(getattr(final, "solvation", {})),
        "ions": {
            "salt_molar": float(salt_molar),
            "placed_ions": list(ion_symbols),
            "ion_count": int(len(ion_symbols)),
            "region": getattr(final, "ions", {}).get("region"),
        },
    }


def _has_overlap(candidate_positions, reference_positions, min_distance):
    if len(candidate_positions) == 0 or len(reference_positions) == 0:
        return False
    deltas = np.asarray(candidate_positions)[:, None, :] - np.asarray(reference_positions)[None, :, :]
    return bool(np.any(np.sum(deltas * deltas, axis=-1) < float(min_distance) ** 2))


def write_protein_membrane_seed(atoms, output_pdb):
    """Write a protein-membrane seed PDB and return the absolute path."""
    path = Path(output_pdb)
    write_pdb(atoms, path)
    return path.resolve()


def add_ions_to_seed(
    atoms,
    ions,
    *,
    region="water",
    min_distance=2.5 / au2angstrom,
    seed=None,
    max_attempts=10000,
    calculator=True,
    **calculator_kwargs,
):
    """Place ions in a protein-membrane seed, optionally restricted to water slabs."""
    ions = tuple(ions)
    if not ions:
        return atoms
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    if np.any(lengths <= 0.0):
        raise ValueError("ion placement requires a finite orthorhombic box.")
    positions = _ion_positions(
        atoms,
        len(ions),
        region=region,
        min_distance=min_distance,
        seed=seed,
        max_attempts=max_attempts,
    )
    start_molecule_id = int(np.max(atoms.get_array("molecule_ids"))) + 1 if atoms.has("molecule_ids") else 0
    ion_atoms = monatomic_ions(list(ions), positions, start_molecule_id=start_molecule_id)
    combined = combine_systems(
        [atoms, ion_atoms],
        cell=lengths,
        pbc=atoms.get_pbc(),
        calculator=calculator,
        **calculator_kwargs,
    )
    for name in ("solvation", "membrane", "protein"):
        if hasattr(atoms, name):
            setattr(combined, name, dict(getattr(atoms, name)))
    combined.ions = {
        "placed_ions": list(ions),
        "region": str(region),
        "min_distance_angstrom": float(min_distance * au2angstrom),
    }
    return combined


def _ion_positions(atoms, nions, *, region, min_distance, seed, max_attempts):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    existing = atoms.get_positions()
    min_distance2 = float(min_distance) ** 2
    positions = []
    attempts = 0
    while len(positions) < int(nions) and attempts < int(max_attempts):
        attempts += 1
        trial = _trial_ion_position(atoms, lengths, region, rng)
        if existing.size:
            deltas = trial - existing
            if np.any(np.sum(deltas * deltas, axis=1) < min_distance2):
                continue
        if positions:
            deltas = trial - np.asarray(positions)
            if np.any(np.sum(deltas * deltas, axis=1) < min_distance2):
                continue
        positions.append(trial)
    if len(positions) < int(nions):
        raise RuntimeError("could not place all ions; try fewer ions or a smaller min_distance.")
    return np.asarray(positions, dtype=float)


def _trial_ion_position(atoms, lengths, region, rng):
    region = str(region).lower()
    if region in {"box", "any", "all"}:
        return rng.uniform(0.0, lengths)
    if region != "water":
        raise ValueError("ion region must be 'water' or 'box'.")
    water_positions = _water_positions(atoms)
    if len(water_positions) == 0:
        return rng.uniform(0.0, lengths)
    center_z = float(getattr(atoms, "membrane", {}).get("center_z", 0.5 * lengths[2]))
    lower = water_positions[water_positions[:, 2] < center_z]
    upper = water_positions[water_positions[:, 2] >= center_z]
    slab = upper if (len(lower) == 0 or rng.random() < 0.5) and len(upper) else lower
    zmin = float(np.min(slab[:, 2]))
    zmax = float(np.max(slab[:, 2]))
    return np.array(
        [
            rng.uniform(0.0, lengths[0]),
            rng.uniform(0.0, lengths[1]),
            rng.uniform(zmin, zmax),
        ],
        dtype=float,
    )


def _water_positions(atoms):
    if not atoms.has("residue_names"):
        return np.empty((0, 3), dtype=float)
    residue_names = np.char.upper(np.asarray(atoms.get_array("residue_names"), dtype=str))
    water_mask = np.isin(residue_names, ["HOH", "WAT", "TIP3", "SOL"])
    return atoms.get_positions()[water_mask]
