"""Import equilibrated OpenMM structures for PyQED QM/MM workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyqed.units import au2nm

from .atoms import Atoms
from .calculators import MM
from .constraints import FixBondLengths
from .topology import Topology


HARTREE_TO_KJMOL = 2625.4996394799


@dataclass(frozen=True)
class OpenMMAtomRecord:
    """Metadata for one atom in an OpenMM topology."""

    index: int
    name: str
    element: str
    residue_name: str
    residue_id: str
    chain_id: str


@dataclass
class OpenMMImportedFrame:
    """One frame imported from an OpenMM topology and coordinate file."""

    atoms: Atoms
    atom_records: tuple[OpenMMAtomRecord, ...]
    qm_indices: np.ndarray | None = None
    frame: int = 0


def atoms_from_openmm_pdb(
    pdb_path,
    forcefield_files=(),
    frame=0,
    qm_indices=None,
    qm_resname=None,
    qm_resid=None,
    qm_chain=None,
    qm_atom_names=None,
    ignore_external_bonds=True,
):
    """Read an OpenMM PDB and return a PyQED ``Atoms`` frame with MM charges.

    Parameters
    ----------
    pdb_path
        Coordinate/topology PDB file readable by ``openmm.app.PDBFile``.
    forcefield_files
        OpenMM force-field XML files used to create the ``System``.  Charges
        are extracted from the resulting ``NonbondedForce``.
    frame
        PDB model/frame index, zero based.
    qm_indices
        Optional explicit zero-based atom indices for the QM chromophore.
    qm_resname, qm_resid, qm_chain, qm_atom_names
        Optional metadata filters used to select the QM region.
    ignore_external_bonds
        Passed to ``ForceField.createSystem``.  This is useful for PDBs where
        not every chain is chemically complete.
    """

    openmm, app, unit = _openmm_modules()
    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    positions_nm = _positions_nm(pdb, frame, unit)
    records = _atom_records(topology)
    symbols = [record.element for record in records]
    charges = _charges_from_forcefield(
        topology,
        forcefield_files=forcefield_files,
        openmm=openmm,
        app=app,
        ignore_external_bonds=ignore_external_bonds,
    )

    cell = _orthorhombic_cell_bohr(topology, unit)
    pbc = cell is not None
    atoms = Atoms(
        [[symbol, tuple(coord_nm / au2nm)] for symbol, coord_nm in zip(symbols, positions_nm)],
        cell=cell if cell is not None else None,
        pbc=pbc,
    )
    atoms.set_array("charges", charges, float, ())
    atoms.set_array("atom_names", [record.name for record in records], str, ())
    atoms.set_array("residue_names", [record.residue_name for record in records], str, ())
    atoms.set_array("residue_ids", [record.residue_id for record in records], str, ())
    atoms.set_array("chain_ids", [record.chain_id for record in records], str, ())
    atoms.set_array("openmm_indices", [record.index for record in records], int, ())

    selected = select_openmm_atoms(
        records,
        indices=qm_indices,
        resname=qm_resname,
        resid=qm_resid,
        chain=qm_chain,
        atom_names=qm_atom_names,
    )
    return OpenMMImportedFrame(
        atoms=atoms,
        atom_records=tuple(records),
        qm_indices=selected,
        frame=int(frame),
    )


def atoms_from_openmm_pdb_system(
    pdb_path,
    forcefield_files,
    frame=0,
    qm_indices=None,
    qm_resname=None,
    qm_resid=None,
    qm_chain=None,
    qm_atom_names=None,
    ignore_external_bonds=True,
    constraints="HBonds",
    rigid_water=True,
    nonbonded_method="pme",
    nonbonded_cutoff_nm=1.0,
    ewald_alpha_per_nm=0.0,
    pme_mesh=None,
    reaction_field_dielectric=78.3,
    attach_calculator=True,
    nonbonded_skin=1.0,
):
    """Import an OpenMM PDB+ForceField as a native PyQED MM system.

    Unlike :func:`atoms_from_openmm_pdb`, this extracts the full classical
    model represented by OpenMM's ``System``: harmonic bonds/angles, periodic
    torsions, nonbonded particle parameters, exception scales, constraints,
    masses, coordinates, and residue metadata.  It is intended as the bridge
    for moving equilibrated OpenMM membrane seeds toward native PyQED MD.
    """

    openmm, app, unit = _openmm_modules()
    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    positions_nm = _positions_nm(pdb, frame, unit)
    records = _atom_records(topology)
    symbols = [record.element for record in records]

    forcefield = app.ForceField(*(str(path) for path in forcefield_files))
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=_openmm_nonbonded_method(app, nonbonded_method),
        nonbondedCutoff=float(nonbonded_cutoff_nm) * unit.nanometer,
        constraints=_openmm_constraints(app, constraints),
        rigidWater=bool(rigid_water),
        ignoreExternalBonds=bool(ignore_external_bonds),
        ewaldErrorTolerance=5.0e-4,
    )

    topology_data, constraint_pairs, constraint_distances = _topology_from_openmm_system(
        system,
        openmm,
        unit,
    )
    cell = _orthorhombic_cell_bohr(topology, unit)
    pbc = cell is not None
    atoms = Atoms(
        [[symbol, tuple(coord_nm / au2nm)] for symbol, coord_nm in zip(symbols, positions_nm)],
        cell=cell if cell is not None else None,
        pbc=pbc,
    )
    atoms.topology = topology_data
    atoms.set_array("charges", topology_data.charges, float, ())
    atoms.set_array("lj_epsilon", topology_data.lj_epsilon, float, ())
    atoms.set_array("lj_sigma", topology_data.lj_sigma, float, ())
    atoms.set_array("masses_amu", topology_data.masses_amu, float, ())
    atoms.set_array("atom_names", [record.name for record in records], str, ())
    atoms.set_array("atom_types", topology_data.atom_types, str, ())
    atoms.set_array("residue_names", [record.residue_name for record in records], str, ())
    atoms.set_array("residue_ids", [record.residue_id for record in records], str, ())
    atoms.set_array("chain_ids", [record.chain_id for record in records], str, ())
    atoms.set_array("openmm_indices", [record.index for record in records], int, ())
    atoms.set_array("molecule_ids", _molecule_ids_from_residues(topology), int, ())
    if constraint_pairs:
        atoms.constraints = [FixBondLengths(constraint_pairs, distances=constraint_distances)]

    method = str(nonbonded_method).lower()
    if attach_calculator:
        cutoff_bohr = None if method in {"none", "nocutoff", "no-cutoff"} else float(nonbonded_cutoff_nm) / au2nm
        calculator_method = "pme" if method == "pme" else "cutoff"
        calc_kwargs = {}
        if calculator_method == "pme":
            calc_kwargs["ewald_alpha"] = (
                0.35 if float(ewald_alpha_per_nm) == 0.0 else float(ewald_alpha_per_nm) * au2nm
            )
            if pme_mesh is not None:
                calc_kwargs["pme_mesh"] = tuple(int(value) for value in pme_mesh)
        else:
            calc_kwargs["coulomb_reaction_field_dielectric"] = float(reaction_field_dielectric)
        atoms.calc = MM(
            bonds=topology_data.bonds,
            angles=topology_data.angles,
            torsions=topology_data.torsions,
            impropers=topology_data.impropers,
            angle_unit="radian",
            torsion_unit="radian",
            improper_unit="radian",
            charges=topology_data.charges,
            lj_epsilon=topology_data.lj_epsilon,
            lj_sigma=topology_data.lj_sigma,
            atom_types=topology_data.atom_types,
            lj_pair_overrides=topology_data.lj_pair_overrides,
            lj_pair_parameters=topology_data.lj_pair_parameters,
            coulomb_pair_parameters=topology_data.coulomb_pair_parameters,
            coulomb_method=calculator_method,
            coulomb_cutoff=cutoff_bohr,
            lj_cutoff=cutoff_bohr,
            lj_energy_shift=False,
            coulomb_energy_shift=False,
            exclude_bonded=False,
            exclude_angles=False,
            nonbonded_exclusions=topology_data.nonbonded_exclusions,
            lj_exclusions=topology_data.lj_exclusions,
            coulomb_exclusions=topology_data.coulomb_exclusions,
            lj_pair_scales=topology_data.lj_pair_scales,
            coulomb_pair_scales={},
            nonbonded_skin=nonbonded_skin,
            **calc_kwargs,
        )

    selected = select_openmm_atoms(
        records,
        indices=qm_indices,
        resname=qm_resname,
        resid=qm_resid,
        chain=qm_chain,
        atom_names=qm_atom_names,
    )
    return OpenMMImportedFrame(
        atoms=atoms,
        atom_records=tuple(records),
        qm_indices=selected,
        frame=int(frame),
    )


def select_openmm_atoms(
    records,
    indices=None,
    resname=None,
    resid=None,
    chain=None,
    atom_names=None,
):
    """Select OpenMM atoms by explicit indices and/or topology metadata."""

    records = tuple(records)
    selected = np.ones(len(records), dtype=bool)
    any_filter = False

    if indices is not None:
        explicit = np.asarray(indices, dtype=int).reshape(-1)
        if explicit.size == 0:
            raise ValueError("indices is empty.")
        if explicit.min() < 0 or explicit.max() >= len(records):
            raise ValueError("indices contains an atom index outside the topology.")
        mask = np.zeros(len(records), dtype=bool)
        mask[explicit] = True
        selected &= mask
        any_filter = True

    if resname is not None:
        names = _as_upper_set(resname)
        selected &= np.asarray([record.residue_name.upper() in names for record in records])
        any_filter = True

    if resid is not None:
        ids = {str(value) for value in _as_list(resid)}
        selected &= np.asarray([record.residue_id in ids for record in records])
        any_filter = True

    if chain is not None:
        chains = {str(value) for value in _as_list(chain)}
        selected &= np.asarray([record.chain_id in chains for record in records])
        any_filter = True

    if atom_names is not None:
        names = _as_upper_set(atom_names)
        selected &= np.asarray([record.name.upper() in names for record in records])
        any_filter = True

    if not any_filter:
        return None
    result = np.flatnonzero(selected)
    if result.size == 0:
        raise ValueError("QM atom selection matched no atoms.")
    return result.astype(int)


def _openmm_modules():
    try:
        import openmm
        from openmm import app, unit
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenMM is required to import OpenMM systems.") from exc
    return openmm, app, unit


def _positions_nm(pdb, frame, unit):
    frame = int(frame)
    if frame < 0 or frame >= pdb.getNumFrames():
        raise ValueError(f"frame {frame} is outside the PDB frame range 0..{pdb.getNumFrames() - 1}.")
    positions = pdb.getPositions(frame=frame)
    return np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)


def _atom_records(topology):
    records = []
    for atom in topology.atoms():
        residue = atom.residue
        chain = residue.chain
        element = atom.element.symbol if atom.element is not None else atom.name[:1]
        records.append(
            OpenMMAtomRecord(
                index=int(atom.index),
                name=str(atom.name),
                element=str(element),
                residue_name=str(residue.name),
                residue_id=str(residue.id),
                chain_id=str(chain.id),
            )
        )
    return records


def _charges_from_forcefield(
    topology,
    forcefield_files,
    openmm,
    app,
    ignore_external_bonds=True,
):
    files = tuple(str(path) for path in forcefield_files)
    if not files:
        raise ValueError("forcefield_files is required so MM point charges can be extracted.")
    forcefield = app.ForceField(*files)
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        ignoreExternalBonds=bool(ignore_external_bonds),
    )
    nonbonded = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise ValueError("OpenMM System has no NonbondedForce; cannot extract MM charges.")
    charges = []
    for index in range(nonbonded.getNumParticles()):
        charge, _sigma, _epsilon = nonbonded.getParticleParameters(index)
        charges.append(float(charge.value_in_unit(openmm.unit.elementary_charge)))
    return np.asarray(charges, dtype=float)


def _topology_from_openmm_system(system, openmm, unit):
    bonds = []
    angles = []
    torsions = []
    impropers = []
    charges = np.zeros(system.getNumParticles(), dtype=float)
    lj_sigma = np.zeros(system.getNumParticles(), dtype=float)
    lj_epsilon = np.zeros(system.getNumParticles(), dtype=float)
    masses = np.array(
        [
            float(system.getParticleMass(index).value_in_unit(unit.dalton))
            for index in range(system.getNumParticles())
        ],
        dtype=float,
    )
    nonbonded_exclusions = set()
    lj_exclusions = set()
    lj_pair_overrides = {}
    lj_pair_scales = {}
    lj_pair_parameters = {}
    coulomb_exclusions = set()
    coulomb_pair_parameters = {}
    coulomb_pair_scales = {}
    atom_types = np.asarray([f"openmm_{index}" for index in range(system.getNumParticles())], dtype=str)
    custom_lj_present = False

    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for index in range(force.getNumBonds()):
                i, j, length, k = force.getBondParameters(index)
                bonds.append(
                    (
                        int(i),
                        int(j),
                        _bond_k_au(k, unit),
                        _length_bohr(length, unit),
                    )
                )
        elif isinstance(force, openmm.HarmonicAngleForce):
            for index in range(force.getNumAngles()):
                i, j, k, theta0, ktheta = force.getAngleParameters(index)
                angles.append(
                    (
                        int(i),
                        int(j),
                        int(k),
                        _angle_k_au(ktheta, unit),
                        _angle_rad(theta0, unit),
                    )
                )
        elif isinstance(force, openmm.PeriodicTorsionForce):
            for index in range(force.getNumTorsions()):
                i, j, k, l, periodicity, phase, barrier = force.getTorsionParameters(index)
                torsions.append(
                    (
                        int(i),
                        int(j),
                        int(k),
                        int(l),
                        _energy_au(barrier, unit),
                        int(periodicity),
                        _angle_rad(phase, unit),
                    )
                )
        elif isinstance(force, openmm.CustomTorsionForce):
            if _is_harmonic_improper_custom_torsion(force):
                for index in range(force.getNumTorsions()):
                    i, j, k, l, parameters = force.getTorsionParameters(index)
                    force_constant, phase = _custom_torsion_parameters(force, parameters, unit)
                    impropers.append((int(i), int(j), int(k), int(l), force_constant, phase))
        elif isinstance(force, openmm.NonbondedForce):
            for index in range(force.getNumParticles()):
                charge, sigma, epsilon = force.getParticleParameters(index)
                charges[index] = float(charge.value_in_unit(unit.elementary_charge))
                lj_sigma[index] = _length_bohr(sigma, unit)
                lj_epsilon[index] = _energy_au(epsilon, unit)
            for index in range(force.getNumExceptions()):
                i, j, charge_product, sigma, epsilon = force.getExceptionParameters(index)
                i = int(i)
                j = int(j)
                pair = tuple(sorted((i, j)))
                charge_product = float(charge_product.value_in_unit(unit.elementary_charge**2))
                sigma = _length_bohr(sigma, unit)
                epsilon = _energy_au(epsilon, unit)
                if abs(charge_product) < 1.0e-14 and abs(epsilon) < 1.0e-14:
                    nonbonded_exclusions.add(pair)
                    continue
                coulomb_exclusions.add(pair)
                coulomb_pair_parameters[pair] = charge_product
                coulomb_pair_scales[pair] = _safe_scale(charge_product, charges[i] * charges[j])
                epsilon_default = float(np.sqrt(max(lj_epsilon[i] * lj_epsilon[j], 0.0)))
                sigma_default = float(0.5 * (lj_sigma[i] + lj_sigma[j]))
                if abs(epsilon) < 1.0e-14:
                    lj_pair_scales[pair] = 0.0
                elif np.isclose(sigma, sigma_default, rtol=1.0e-8, atol=1.0e-10):
                    lj_pair_scales[pair] = _safe_scale(epsilon, epsilon_default)
                else:
                    # PyQED's native exception model currently supports
                    # scale factors, not pair-specific 1-4 sigma overrides.
                    lj_pair_scales[pair] = _safe_scale(epsilon, epsilon_default)
        elif isinstance(force, openmm.CustomNonbondedForce):
            custom_atom_types, pair_overrides, diagonal = _custom_lj_overrides(force, unit)
            if custom_atom_types is not None:
                custom_lj_present = True
                atom_types = custom_atom_types
            if pair_overrides:
                lj_pair_overrides = pair_overrides
                for index, atom_type in enumerate(atom_types):
                    if atom_type in diagonal:
                        lj_epsilon[index], lj_sigma[index] = diagonal[atom_type]
            for index in range(force.getNumExclusions()):
                i, j = force.getExclusionParticles(index)
                lj_exclusions.add(tuple(sorted((int(i), int(j)))))
        elif isinstance(force, openmm.CustomBondForce):
            if _is_lj_custom_bond(force):
                for index in range(force.getNumBonds()):
                    i, j, parameters = force.getBondParameters(index)
                    sigma, epsilon = parameters
                    lj_pair_parameters[tuple(sorted((int(i), int(j))))] = (
                        _energy_au(epsilon, unit),
                        _length_bohr(sigma, unit),
                    )

    if custom_lj_present:
        lj_pair_scales = {}

    constraint_pairs = []
    constraint_distances = []
    for index in range(system.getNumConstraints()):
        i, j, distance = system.getConstraintParameters(index)
        constraint_pairs.append((int(i), int(j)))
        constraint_distances.append(_length_bohr(distance, unit))

    topology = Topology(
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        impropers=impropers,
        charges=charges,
        lj_epsilon=lj_epsilon,
        lj_sigma=lj_sigma,
        masses_amu=masses,
        atom_types=atom_types,
        lj_pair_overrides=lj_pair_overrides,
        lj_pair_parameters=lj_pair_parameters,
        coulomb_pair_parameters=coulomb_pair_parameters,
        nonbonded_exclusions=nonbonded_exclusions,
        lj_exclusions=lj_exclusions,
        coulomb_exclusions=coulomb_exclusions,
        lj_pair_scales=lj_pair_scales,
        coulomb_pair_scales=coulomb_pair_scales,
    )
    return topology, constraint_pairs, np.asarray(constraint_distances, dtype=float)


def _openmm_nonbonded_method(app, method):
    method = str(method).lower()
    if method == "pme":
        return app.PME
    if method in {"cutoff", "cutoffperiodic", "cutoff-periodic"}:
        return app.CutoffPeriodic
    if method in {"none", "nocutoff", "no-cutoff"}:
        return app.NoCutoff
    raise ValueError("nonbonded_method must be 'pme', 'cutoff', or 'nocutoff'.")


def _custom_lj_overrides(force, unit):
    expression = force.getEnergyFunction()
    if "acoef" not in expression or "bcoef" not in expression:
        return None, {}, {}
    if force.getNumPerParticleParameters() != 1:
        return None, {}, {}

    tables = {}
    for index in range(force.getNumTabulatedFunctions()):
        name = force.getTabulatedFunctionName(index)
        if name not in {"acoef", "bcoef"}:
            continue
        width, height, values = force.getTabulatedFunction(index).getFunctionParameters()
        tables[name] = np.asarray(values, dtype=float).reshape(int(height), int(width))
    if "acoef" not in tables or "bcoef" not in tables:
        return None, {}, {}

    atom_types = np.asarray(
        [str(int(force.getParticleParameters(index)[0])) for index in range(force.getNumParticles())],
        dtype=str,
    )
    unique_types = sorted({int(value) for value in atom_types})
    overrides = {}
    diagonal = {}
    for type_i in unique_types:
        for type_j in unique_types:
            if type_j < type_i:
                continue
            acoef = float(tables["acoef"][type_j, type_i])
            bcoef = float(tables["bcoef"][type_j, type_i])
            epsilon, sigma = _lj_ab_to_epsilon_sigma(acoef, bcoef, unit)
            key = tuple(sorted((str(type_i), str(type_j))))
            overrides[key] = (epsilon, sigma)
            if type_i == type_j:
                diagonal[str(type_i)] = (epsilon, sigma)
    return atom_types, overrides, diagonal


def _is_harmonic_improper_custom_torsion(force):
    expression = force.getEnergyFunction().replace(" ", "")
    if "(theta-theta0)^2" not in expression:
        return False
    parameter_names = {
        force.getPerTorsionParameterName(index)
        for index in range(force.getNumPerTorsionParameters())
    }
    return {"k", "theta0"}.issubset(parameter_names)


def _custom_torsion_parameters(force, parameters, unit):
    values = {
        force.getPerTorsionParameterName(index): parameters[index]
        for index in range(force.getNumPerTorsionParameters())
    }
    # OpenMM expression is k*(theta-theta0)^2.  PyQED stores harmonic
    # impropers as 0.5*K*(theta-theta0)^2, hence K = 2*k.
    return 2.0 * _energy_au(values["k"], unit), _angle_rad(values["theta0"], unit)


def _lj_ab_to_epsilon_sigma(acoef, bcoef, unit):
    if abs(acoef) < 1.0e-30 or abs(bcoef) < 1.0e-30:
        return 0.0, 1.0
    sigma_nm = (acoef / bcoef) ** (1.0 / 6.0)
    epsilon_kj_mol = bcoef * bcoef / (4.0 * acoef)
    return epsilon_kj_mol / HARTREE_TO_KJMOL, sigma_nm / au2nm


def _is_lj_custom_bond(force):
    expression = force.getEnergyFunction()
    if "epsilon" not in expression or "sigma" not in expression:
        return False
    parameter_names = {
        force.getPerBondParameterName(index)
        for index in range(force.getNumPerBondParameters())
    }
    return {"sigma", "epsilon"}.issubset(parameter_names)


def _openmm_constraints(app, constraints):
    if constraints is None or constraints is False or str(constraints).lower() in {"none", "false"}:
        return None
    name = str(constraints).lower()
    if name in {"hbonds", "h-bonds", "h_bonds"}:
        return app.HBonds
    if name in {"allbonds", "all-bonds", "all_bonds"}:
        return app.AllBonds
    if name in {"hangles", "h-angles", "h_angles"}:
        return app.HAngles
    raise ValueError("constraints must be None, 'HBonds', 'AllBonds', or 'HAngles'.")


def _length_bohr(quantity, unit):
    if not hasattr(quantity, "value_in_unit"):
        return float(quantity) / au2nm
    return float(quantity.value_in_unit(unit.nanometer)) / au2nm


def _energy_au(quantity, unit):
    if not hasattr(quantity, "value_in_unit"):
        return float(quantity) / HARTREE_TO_KJMOL
    return float(quantity.value_in_unit(unit.kilojoule_per_mole)) / HARTREE_TO_KJMOL


def _bond_k_au(quantity, unit):
    value = float(quantity.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
    return value * au2nm * au2nm / HARTREE_TO_KJMOL


def _angle_k_au(quantity, unit):
    value = float(quantity.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))
    return value / HARTREE_TO_KJMOL


def _angle_rad(quantity, unit):
    if not hasattr(quantity, "value_in_unit"):
        return float(quantity)
    return float(quantity.value_in_unit(unit.radian))


def _safe_scale(value, reference):
    value = float(value)
    reference = float(reference)
    if abs(reference) < 1.0e-14:
        return 0.0 if abs(value) < 1.0e-14 else 1.0
    return value / reference


def _molecule_ids_from_residues(topology):
    molecule_ids = np.zeros(topology.getNumAtoms(), dtype=int)
    for molecule_id, residue in enumerate(topology.residues()):
        for atom in residue.atoms():
            molecule_ids[int(atom.index)] = molecule_id
    return molecule_ids


def _orthorhombic_cell_bohr(topology, unit):
    vectors = topology.getPeriodicBoxVectors()
    if vectors is None:
        return None
    matrix = np.asarray(vectors.value_in_unit(unit.nanometer), dtype=float)
    offdiag = matrix - np.diag(np.diag(matrix))
    if not np.allclose(offdiag, 0.0, atol=1.0e-8):
        raise ValueError(
            "Only orthorhombic OpenMM boxes are currently supported for membrane embedding."
        )
    return np.diag(matrix) / au2nm


def _as_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _as_upper_set(value):
    return {str(item).upper() for item in _as_list(value)}
