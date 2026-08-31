#!/usr/bin/env python3
"""OpenMM reference path for the toy hydrated all-atom membrane.

The PyQED membrane example is intentionally small and hand-parameterized.  This
script exports the same model to an OpenMM-style System when OpenMM is
installed, using particle parameters, constraints, and explicit nonbonded
exceptions in the same spirit as ``NonbondedForce.createExceptionsFromBonds``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.md.all_atom_lipid_membrane import build_membrane
from pyqed.md import MolecularMechanics
from pyqed.md.calculators import (
    _add_coulomb_pairs,
    _add_coulomb_scaled_pairs,
    _add_ewald_real,
    _add_pme_reciprocal,
)
from pyqed.md.neighborlist import minimum_image
from pyqed.units import au2angstrom, au2fs, au2kjmol, au2nm, fs

HARTREE_TO_KJ_MOL = au2kjmol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "validation", "hydrated", "longer"),
        help="Use a tested membrane/OpenMM workflow preset; explicit CLI flags override preset values.",
    )
    parser.add_argument("--nx", type=int, default=3)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--waters-per-lipid", type=int, default=3)
    parser.add_argument("--salt-pairs", type=int, default=2)
    parser.add_argument("--lipid-spacing", type=float, default=8.5)
    parser.add_argument("--pme-mesh", type=int, default=24)
    parser.add_argument("--ewald-alpha", type=float, default=0.10, help="Ewald/PME alpha in bohr^-1.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timestep-fs", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--friction-ps", type=float, default=10.0)
    parser.add_argument(
        "--nonbonded-method",
        choices=("pme", "cutoff", "direct-cutoff", "nocutoff"),
        default="pme",
    )
    parser.add_argument("--no-lj-shift", action="store_true", help="Disable PyQED LJ cutoff shift for OpenMM parity checks.")
    parser.add_argument("--output-dir", default="/private/tmp/pyqed_openmm_membrane_reference")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--skip-minimize", action="store_true")
    parser.add_argument("--minimize-tolerance", type=float, default=10.0, help="OpenMM minimization tolerance in kJ/mol/nm.")
    parser.add_argument("--minimize-iterations", type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=0, help="Write OpenMM/PyQED energy snapshots every N MD steps.")
    return parser.parse_args()


PRESETS = {
    "smoke": {
        "nx": 2,
        "ny": 2,
        "waters_per_lipid": 1,
        "salt_pairs": 1,
        "steps": 20,
        "timestep_fs": 0.01,
        "temperature": 50.0,
        "minimize_iterations": 50,
        "snapshot_interval": 5,
        "nonbonded_method": "pme",
        "no_lj_shift": True,
    },
    "validation": {
        "nx": 3,
        "ny": 3,
        "waters_per_lipid": 1,
        "salt_pairs": 1,
        "steps": 20,
        "timestep_fs": 0.01,
        "temperature": 50.0,
        "minimize_iterations": 50,
        "snapshot_interval": 5,
        "nonbonded_method": "pme",
        "no_lj_shift": True,
    },
    "hydrated": {
        "nx": 3,
        "ny": 3,
        "waters_per_lipid": 3,
        "salt_pairs": 2,
        "steps": 50,
        "timestep_fs": 0.02,
        "temperature": 80.0,
        "minimize_iterations": 80,
        "snapshot_interval": 10,
        "nonbonded_method": "pme",
        "no_lj_shift": True,
    },
    "longer": {
        "nx": 3,
        "ny": 3,
        "waters_per_lipid": 2,
        "salt_pairs": 2,
        "steps": 200,
        "timestep_fs": 0.05,
        "temperature": 100.0,
        "minimize_iterations": 100,
        "snapshot_interval": 25,
        "nonbonded_method": "pme",
        "no_lj_shift": True,
    },
}


def apply_preset(args, argv=None):
    if args.preset is None:
        return args
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        explicit.add(token[2:].split("=", 1)[0].replace("-", "_"))
    for name, value in PRESETS[args.preset].items():
        if name not in explicit:
            setattr(args, name, value)
    return args


def write_pdb(atoms, atom_types, path, positions_bohr=None):
    positions = atoms.get_positions() if positions_bohr is None else np.asarray(positions_bohr, dtype=float)
    positions_angstrom = positions * au2angstrom
    regions = atoms.get_array("regions")
    lipid_ids = atoms.get_array("lipid_ids")
    symbols = atoms.atom_symbols()
    cell_angstrom = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    residue_names = {0: "LIP", 1: "LIP", 2: "HOH", 3: "ION"}
    with open(path, "w") as handle:
        handle.write(
            f"CRYST1{cell_angstrom[0]:9.3f}{cell_angstrom[1]:9.3f}{cell_angstrom[2]:9.3f}"
            "  90.00  90.00  90.00 P 1           1\n"
        )
        for index, (symbol, atom_type, xyz, region, resid) in enumerate(
            zip(symbols, atom_types, positions_angstrom, regions, lipid_ids),
            start=1,
        ):
            name = atom_type[:4]
            resname = residue_names.get(int(region), "UNK")
            handle.write(
                f"HETATM{index:5d} {name:<4s} {resname:>3s} A{int(resid) % 10000:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {symbol:>2s}\n"
            )
        handle.write("END\n")


def write_xyz(atoms, path, positions_bohr=None, comment=""):
    positions = atoms.get_positions() if positions_bohr is None else np.asarray(positions_bohr, dtype=float)
    positions_angstrom = positions * au2angstrom
    with open(path, "w") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(f"{comment}\n")
        for symbol, xyz in zip(atoms.atom_symbols(), positions_angstrom):
            handle.write(f"{symbol:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")


def write_manifest(atoms, atom_types, path):
    calc = atoms.calc
    constraints = []
    for constraint in atoms.constraints:
        if hasattr(constraint, "pairs"):
            targets = constraint._targets(atoms)
            constraints.extend(
                {"i": int(i), "j": int(j), "distance_nm": float(r0 * au2nm)}
                for (i, j), r0 in zip(constraint.pairs, targets)
            )

    manifest = {
        "atoms": len(atoms),
        "atom_types": list(atom_types),
        "charges_e": np.asarray(calc.charges, dtype=float).tolist(),
        "lj_epsilon_kj_mol": (np.asarray(calc.lj_epsilon, dtype=float) * HARTREE_TO_KJ_MOL).tolist(),
        "lj_sigma_nm": (np.asarray(calc.lj_sigma, dtype=float) * au2nm).tolist(),
        "constraints": constraints,
        "nonbonded_exclusions": [list(pair) for pair in sorted(calc.nonbonded_exclusions)],
        "lj_pair_scales": {f"{i}-{j}": scale for (i, j), scale in calc.lj_pair_scales.items()},
        "coulomb_pair_scales": {f"{i}-{j}": scale for (i, j), scale in calc.coulomb_pair_scales.items()},
    }
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)


def pyqed_energy_components(atoms, lj_energy_shift=None):
    calc = atoms.calc
    positions = atoms.get_positions()
    if lj_energy_shift is None:
        lj_energy_shift = calc.lj_energy_shift
    common = dict(
        coulomb_constant=calc.coulomb_constant,
        coulomb_method=calc.coulomb_method,
        coulomb_cutoff=calc.coulomb_cutoff,
        ewald_alpha=calc.ewald_alpha,
        pme_mesh=tuple(calc.pme_mesh.tolist()),
        lj_cutoff=calc.lj_cutoff,
        lj_energy_shift=lj_energy_shift,
        exclude_bonded=False,
        exclude_angles=False,
        nonbonded_exclusions=calc.nonbonded_exclusions,
        nonbonded_skin=0.0,
    )
    lj_calc = MolecularMechanics(
        lj_epsilon=calc.lj_epsilon,
        lj_sigma=calc.lj_sigma,
        lj_pair_scales=calc.lj_pair_scales,
        **common,
    )
    coulomb_calc = MolecularMechanics(
        charges=calc.charges,
        coulomb_pair_scales=calc.coulomb_pair_scales,
        **common,
    )
    if lj_energy_shift == calc.lj_energy_shift:
        total = atoms.get_potential_energy()
    else:
        total_calc = MolecularMechanics(
            charges=calc.charges,
            lj_epsilon=calc.lj_epsilon,
            lj_sigma=calc.lj_sigma,
            lj_pair_scales=calc.lj_pair_scales,
            coulomb_pair_scales=calc.coulomb_pair_scales,
            **common,
        )
        total, _ = total_calc.calculate(atoms)
    lj, _ = lj_calc.calculate(atoms)
    coulomb, _ = coulomb_calc.calculate(atoms)
    return {
        "total_hartree": float(total),
        "total_kj_mol": float(total * HARTREE_TO_KJ_MOL),
        "lj_hartree": float(lj),
        "lj_kj_mol": float(lj * HARTREE_TO_KJ_MOL),
        "coulomb_hartree": float(coulomb),
        "coulomb_kj_mol": float(coulomb * HARTREE_TO_KJ_MOL),
        "residual_hartree": float(total - lj - coulomb),
        "residual_kj_mol": float((total - lj - coulomb) * HARTREE_TO_KJ_MOL),
        "positions_shape": list(positions.shape),
    }


def write_pyqed_energy_report(atoms, path, lj_energy_shift=None):
    components = pyqed_energy_components(atoms, lj_energy_shift=lj_energy_shift)
    with open(path, "w") as handle:
        for key, value in components.items():
            if isinstance(value, float):
                handle.write(f"{key}: {value:.12e}\n")
            else:
                handle.write(f"{key}: {value}\n")
    return components


def build_openmm_system(atoms, nonbonded_method="pme", include_lj=True, include_coulomb=True):
    try:
        import openmm
        import openmm.app
        from openmm import unit
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenMM is not installed in this environment.") from exc

    calc = atoms.calc
    system = openmm.System()
    for mass in atoms.get_masses_amu():
        system.addParticle(float(mass) * unit.dalton)

    lengths_nm = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2nm
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(lengths_nm[0], 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, lengths_nm[1], 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 0.0, lengths_nm[2]) * unit.nanometer,
    )

    for constraint in atoms.constraints:
        if hasattr(constraint, "pairs"):
            targets = constraint._targets(atoms)
            for (i, j), distance in zip(constraint.pairs, targets):
                system.addConstraint(int(i), int(j), float(distance * au2nm) * unit.nanometer)

    nonbonded = openmm.NonbondedForce()
    method = nonbonded_method.lower()
    if method == "pme":
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    elif method == "cutoff":
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    elif method == "nocutoff":
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    else:
        raise ValueError("nonbonded_method must be 'pme', 'cutoff', or 'nocutoff'.")
    cutoff_nm = float(calc.coulomb_cutoff * au2nm)
    cutoff_nm = min(cutoff_nm, 0.49 * float(np.min(lengths_nm)))
    nonbonded.setCutoffDistance(cutoff_nm * unit.nanometer)
    nonbonded.setEwaldErrorTolerance(5.0e-4)
    if method == "pme":
        mesh = np.asarray(calc.pme_mesh, dtype=int)
        nonbonded.setPMEParameters(float(calc.ewald_alpha / au2nm), int(mesh[0]), int(mesh[1]), int(mesh[2]))
    nonbonded.setUseDispersionCorrection(False)

    charges = np.asarray(calc.charges, dtype=float)
    epsilon = np.asarray(calc.lj_epsilon, dtype=float) * HARTREE_TO_KJ_MOL
    sigma = np.asarray(calc.lj_sigma, dtype=float) * au2nm
    if not include_coulomb:
        charges = np.zeros_like(charges)
    if not include_lj:
        epsilon = np.zeros_like(epsilon)
    for q, sig, eps in zip(charges, sigma, epsilon):
        nonbonded.addParticle(
            float(q) * unit.elementary_charge,
            float(sig) * unit.nanometer,
            float(eps) * unit.kilojoule_per_mole,
        )

    exception_pairs = set(calc.nonbonded_exclusions)
    exception_pairs.update(calc.lj_pair_scales)
    exception_pairs.update(calc.coulomb_pair_scales)
    for i, j in sorted(exception_pairs):
        coul_scale = calc.coulomb_pair_scales.get((i, j), 0.0)
        lj_scale = calc.lj_pair_scales.get((i, j), 0.0)
        sigma_ij = 0.5 * (sigma[i] + sigma[j])
        epsilon_ij = np.sqrt(epsilon[i] * epsilon[j]) * lj_scale
        charge_product = charges[i] * charges[j] * coul_scale
        nonbonded.addException(
            int(i),
            int(j),
            float(charge_product) * unit.elementary_charge**2,
            float(sigma_ij) * unit.nanometer,
            float(epsilon_ij) * unit.kilojoule_per_mole,
            replace=True,
        )
    system.addForce(nonbonded)
    return system, openmm, unit


def build_openmm_direct_cutoff_system(atoms, include_lj=True, include_coulomb=True):
    try:
        import openmm
        import openmm.app
        from openmm import unit
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenMM is not installed in this environment.") from exc

    calc = atoms.calc
    system = openmm.System()
    for mass in atoms.get_masses_amu():
        system.addParticle(float(mass) * unit.dalton)

    lengths_nm = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2nm
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(lengths_nm[0], 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, lengths_nm[1], 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 0.0, lengths_nm[2]) * unit.nanometer,
    )

    for constraint in atoms.constraints:
        if hasattr(constraint, "pairs"):
            targets = constraint._targets(atoms)
            for (i, j), distance in zip(constraint.pairs, targets):
                system.addConstraint(int(i), int(j), float(distance * au2nm) * unit.nanometer)

    coulomb_constant = HARTREE_TO_KJ_MOL * au2nm
    lj_expr = "4*epsilon*((sigma/r)^12-(sigma/r)^6)" if include_lj else "0"
    coul_expr = f"{coulomb_constant:.16g}*qprod/r" if include_coulomb else "0"
    nonbonded = openmm.CustomNonbondedForce(
        f"{lj_expr} + {coul_expr};"
        "epsilon=sqrt(epsilon1*epsilon2);"
        "sigma=0.5*(sigma1+sigma2);"
        "qprod=charge1*charge2"
    )
    nonbonded.addPerParticleParameter("charge")
    nonbonded.addPerParticleParameter("sigma")
    nonbonded.addPerParticleParameter("epsilon")
    nonbonded.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cutoff_nm = min(float(calc.coulomb_cutoff * au2nm), 0.49 * float(np.min(lengths_nm)))
    nonbonded.setCutoffDistance(cutoff_nm * unit.nanometer)
    charges = np.asarray(calc.charges, dtype=float)
    sigma = np.asarray(calc.lj_sigma, dtype=float) * au2nm
    epsilon = np.asarray(calc.lj_epsilon, dtype=float) * HARTREE_TO_KJ_MOL
    for q, sig, eps in zip(charges, sigma, epsilon):
        nonbonded.addParticle([float(q), float(sig), float(eps)])

    exception_pairs = set(calc.nonbonded_exclusions)
    exception_pairs.update(calc.lj_pair_scales)
    exception_pairs.update(calc.coulomb_pair_scales)
    for i, j in sorted(exception_pairs):
        nonbonded.addExclusion(int(i), int(j))
    system.addForce(nonbonded)

    bond_force = openmm.CustomBondForce(
        f"{lj_expr}*ljscale + {coul_expr}*coulscale;"
        "epsilon=sqrt(epsilon1*epsilon2);"
        "sigma=0.5*(sigma1+sigma2);"
        "qprod=charge1*charge2"
    )
    bond_force.addPerBondParameter("charge1")
    bond_force.addPerBondParameter("charge2")
    bond_force.addPerBondParameter("sigma1")
    bond_force.addPerBondParameter("sigma2")
    bond_force.addPerBondParameter("epsilon1")
    bond_force.addPerBondParameter("epsilon2")
    bond_force.addPerBondParameter("ljscale")
    bond_force.addPerBondParameter("coulscale")
    bond_force.setUsesPeriodicBoundaryConditions(True)
    for i, j in sorted(exception_pairs):
        lj_scale = calc.lj_pair_scales.get((i, j), 0.0)
        coul_scale = calc.coulomb_pair_scales.get((i, j), 0.0)
        if lj_scale == 0.0 and coul_scale == 0.0:
            continue
        bond_force.addBond(
            int(i),
            int(j),
            [
                float(charges[i]),
                float(charges[j]),
                float(sigma[i]),
                float(sigma[j]),
                float(epsilon[i]),
                float(epsilon[j]),
                float(lj_scale),
                float(coul_scale),
            ],
        )
    system.addForce(bond_force)
    return system, openmm, unit


def _state_positions_bohr(state, unit):
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return np.asarray(positions_nm, dtype=float) / au2nm


def _pyqed_energy_at_positions(atoms, positions_bohr, lj_energy_shift=None):
    original = atoms.get_positions()
    try:
        atoms.set_positions(positions_bohr, apply_constraint=False)
        return pyqed_energy_components(atoms, lj_energy_shift=lj_energy_shift)
    finally:
        atoms.set_positions(original, apply_constraint=False)


def _append_xyz_frame(handle, atoms, positions_bohr, comment):
    positions_angstrom = np.asarray(positions_bohr, dtype=float) * au2angstrom
    handle.write(f"{len(atoms)}\n")
    handle.write(f"{comment}\n")
    for symbol, xyz in zip(atoms.atom_symbols(), positions_angstrom):
        handle.write(f"{symbol:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")


def _snapshot_state(simulation, unit):
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    positions = _state_positions_bohr(state, unit)
    return energy, positions


def _write_snapshot_reports(
    atoms,
    simulation,
    unit,
    output_dir,
    steps,
    timestep_fs,
    snapshot_interval,
    lj_energy_shift,
    nonbonded_method,
):
    snapshot_path = output_dir / "openmm_snapshots.xyz"
    energy_path = output_dir / "openmm_pyqed_snapshot_energies.dat"
    remaining = int(steps)
    completed = 0
    interval = int(snapshot_interval)
    with open(snapshot_path, "w") as xyz_handle, open(energy_path, "w") as energy_handle:
        energy_handle.write(
            "step time_fs openmm_kj_mol pyqed_requested_total_kj_mol "
            "pyqed_effective_total_kj_mol pyqed_effective_lj_kj_mol "
            "pyqed_effective_coulomb_kj_mol delta_openmm_minus_pyqed_effective_kj_mol\n"
        )
        openmm_energy, positions_bohr = _snapshot_state(simulation, unit)
        pyqed_requested = _pyqed_energy_at_positions(atoms, positions_bohr, lj_energy_shift=lj_energy_shift)
        pyqed_effective = _pyqed_effective_snapshot_components(
            atoms,
            positions_bohr,
            lj_energy_shift,
            nonbonded_method,
        )
        _append_xyz_frame(
            xyz_handle,
            atoms,
            positions_bohr,
            f"step {completed} time_fs {completed * timestep_fs:.6f} openmm_kj_mol {openmm_energy:.8f}",
        )
        energy_handle.write(
            f"{completed} {completed * timestep_fs:.8f} {openmm_energy:.10f} "
            f"{pyqed_requested['total_kj_mol']:.10f} {pyqed_effective['total_kj_mol']:.10f} "
            f"{pyqed_effective['lj_kj_mol']:.10f} {pyqed_effective['coulomb_kj_mol']:.10f} "
            f"{openmm_energy - pyqed_effective['total_kj_mol']:.10f}\n"
        )
        while remaining > 0:
            chunk = min(interval, remaining)
            simulation.step(chunk)
            remaining -= chunk
            completed += chunk
            openmm_energy, positions_bohr = _snapshot_state(simulation, unit)
            pyqed_requested = _pyqed_energy_at_positions(atoms, positions_bohr, lj_energy_shift=lj_energy_shift)
            pyqed_effective = _pyqed_effective_snapshot_components(
                atoms,
                positions_bohr,
                lj_energy_shift,
                nonbonded_method,
            )
            _append_xyz_frame(
                xyz_handle,
                atoms,
                positions_bohr,
                f"step {completed} time_fs {completed * timestep_fs:.6f} openmm_kj_mol {openmm_energy:.8f}",
            )
            energy_handle.write(
                f"{completed} {completed * timestep_fs:.8f} {openmm_energy:.10f} "
                f"{pyqed_requested['total_kj_mol']:.10f} {pyqed_effective['total_kj_mol']:.10f} "
                f"{pyqed_effective['lj_kj_mol']:.10f} {pyqed_effective['coulomb_kj_mol']:.10f} "
                f"{openmm_energy - pyqed_effective['total_kj_mol']:.10f}\n"
            )
    return snapshot_path, energy_path


def _pyqed_effective_snapshot_components(atoms, positions_bohr, lj_energy_shift, nonbonded_method):
    original = atoms.get_positions()
    try:
        atoms.set_positions(positions_bohr, apply_constraint=False)
        if nonbonded_method == "pme":
            effective_cutoff = _openmm_effective_cutoff_bohr(atoms)
            return pyqed_pme_components(
                atoms,
                lj_energy_shift=lj_energy_shift,
                real_cutoff=effective_cutoff,
                lj_cutoff=effective_cutoff,
            )
        if nonbonded_method == "direct-cutoff":
            return pyqed_direct_cutoff_components(atoms, lj_energy_shift=lj_energy_shift)
        return pyqed_energy_components(atoms, lj_energy_shift=lj_energy_shift)
    finally:
        atoms.set_positions(original, apply_constraint=False)


def run_openmm_reference(
    atoms,
    atom_types,
    steps,
    timestep_fs,
    temperature,
    friction_ps,
    output_dir,
    minimize=True,
    minimize_tolerance=10.0,
    minimize_iterations=200,
    nonbonded_method="pme",
    lj_energy_shift=None,
    snapshot_interval=0,
):
    if nonbonded_method == "direct-cutoff":
        system, openmm, unit = build_openmm_direct_cutoff_system(atoms)
    else:
        system, openmm, unit = build_openmm_system(atoms, nonbonded_method=nonbonded_method)
    integrator = openmm.LangevinMiddleIntegrator(
        float(temperature) * unit.kelvin,
        float(friction_ps) / unit.picosecond,
        float(timestep_fs) * unit.femtosecond,
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = openmm.app.Simulation(_openmm_topology(atoms), system, integrator, platform)
    positions_nm = atoms.get_positions() * au2nm
    simulation.context.setPositions(positions_nm * unit.nanometer)
    state0 = simulation.context.getState(getEnergy=True)
    minimized_openmm = None
    minimized_pyqed = None
    minimized_pdb = None
    minimized_xyz = None
    if minimize:
        openmm.LocalEnergyMinimizer.minimize(
            simulation.context,
            float(minimize_tolerance) * unit.kilojoule_per_mole / unit.nanometer,
            int(minimize_iterations),
        )
        state_min = simulation.context.getState(getEnergy=True, getPositions=True)
        minimized_openmm = state_min.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        minimized_positions_bohr = _state_positions_bohr(state_min, unit)
        minimized_pyqed = _pyqed_energy_at_positions(
            atoms,
            minimized_positions_bohr,
            lj_energy_shift=lj_energy_shift,
        )
        minimized_pdb = output_dir / "pyqed_membrane_openmm_minimized.pdb"
        minimized_xyz = output_dir / "pyqed_membrane_openmm_minimized.xyz"
        write_pdb(atoms, atom_types, minimized_pdb, positions_bohr=minimized_positions_bohr)
        write_xyz(
            atoms,
            minimized_xyz,
            positions_bohr=minimized_positions_bohr,
            comment=f"OpenMM minimized; potential {minimized_openmm:.8f} kJ/mol",
        )
    snapshot_path = None
    snapshot_energy_path = None
    if int(steps) > 0 and int(snapshot_interval) > 0:
        snapshot_path, snapshot_energy_path = _write_snapshot_reports(
            atoms,
            simulation,
            unit,
            output_dir,
            int(steps),
            float(timestep_fs),
            int(snapshot_interval),
            lj_energy_shift,
            nonbonded_method,
        )
    else:
        simulation.step(int(steps))
    state1 = simulation.context.getState(getEnergy=True)
    summary = output_dir / "openmm_summary.txt"
    pyqed_components = pyqed_energy_components(atoms, lj_energy_shift=lj_energy_shift)
    initial_openmm = state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    final_openmm = state1.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    with open(summary, "w") as handle:
        handle.write(f"steps: {int(steps)}\n")
        handle.write(f"time_fs: {int(steps) * float(timestep_fs):.6f}\n")
        handle.write(f"nonbonded_method: {nonbonded_method}\n")
        handle.write(f"pyqed_lj_energy_shift: {bool(lj_energy_shift)}\n")
        handle.write(f"pyqed_total_kj_mol: {pyqed_components['total_kj_mol']:.8f}\n")
        handle.write(f"pyqed_lj_kj_mol: {pyqed_components['lj_kj_mol']:.8f}\n")
        handle.write(f"pyqed_coulomb_kj_mol: {pyqed_components['coulomb_kj_mol']:.8f}\n")
        handle.write(f"initial_potential_kj_mol: {initial_openmm:.8f}\n")
        handle.write(f"initial_delta_openmm_minus_pyqed_kj_mol: {initial_openmm - pyqed_components['total_kj_mol']:.8f}\n")
        if minimized_openmm is not None:
            handle.write(f"minimized_potential_kj_mol: {minimized_openmm:.8f}\n")
            handle.write(f"minimized_pyqed_total_kj_mol: {minimized_pyqed['total_kj_mol']:.8f}\n")
            handle.write(
                "minimized_delta_openmm_minus_pyqed_kj_mol: "
                f"{minimized_openmm - minimized_pyqed['total_kj_mol']:.8f}\n"
            )
            handle.write(f"minimized_pdb: {minimized_pdb}\n")
            handle.write(f"minimized_xyz: {minimized_xyz}\n")
        if snapshot_path is not None:
            handle.write(f"snapshots_xyz: {snapshot_path}\n")
            handle.write(f"snapshot_energies: {snapshot_energy_path}\n")
        handle.write(f"final_potential_kj_mol: {final_openmm:.8f}\n")
    return summary


def _openmm_topology(atoms):
    import openmm.app as app
    from openmm import unit

    top = app.Topology()
    chain = top.addChain("A")
    residue = top.addResidue("MEM", chain)
    for symbol in atoms.atom_symbols():
        element = app.Element.getBySymbol(symbol)
        top.addAtom(symbol, element, residue)
    lengths_nm = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2nm
    top.setPeriodicBoxVectors(
        [
            (lengths_nm[0], 0.0, 0.0) * unit.nanometer,
            (0.0, lengths_nm[1], 0.0) * unit.nanometer,
            (0.0, 0.0, lengths_nm[2]) * unit.nanometer,
        ]
    )
    return top


def openmm_single_point(atoms, nonbonded_method="cutoff", include_lj=True, include_coulomb=True):
    if nonbonded_method == "direct-cutoff":
        system, openmm, unit = build_openmm_direct_cutoff_system(
            atoms,
            include_lj=include_lj,
            include_coulomb=include_coulomb,
        )
    else:
        system, openmm, unit = build_openmm_system(
            atoms,
            nonbonded_method=nonbonded_method,
            include_lj=include_lj,
            include_coulomb=include_coulomb,
        )
    integrator = openmm.VerletIntegrator(0.001 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = openmm.app.Simulation(_openmm_topology(atoms), system, integrator, platform)
    simulation.context.setPositions((atoms.get_positions() * au2nm) * unit.nanometer)
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def write_parity_report(atoms, path, nonbonded_method="cutoff", lj_energy_shift=False):
    if nonbonded_method == "direct-cutoff":
        pyqed = pyqed_direct_cutoff_components(atoms, lj_energy_shift=lj_energy_shift)
    else:
        pyqed = pyqed_energy_components(atoms, lj_energy_shift=lj_energy_shift)
    openmm_total = openmm_single_point(atoms, nonbonded_method=nonbonded_method)
    openmm_lj = openmm_single_point(
        atoms,
        nonbonded_method=nonbonded_method,
        include_lj=True,
        include_coulomb=False,
    )
    openmm_coulomb = openmm_single_point(
        atoms,
        nonbonded_method=nonbonded_method,
        include_lj=False,
        include_coulomb=True,
    )
    rows = {
        "nonbonded_method": nonbonded_method,
        "pyqed_lj_energy_shift": bool(lj_energy_shift),
        "pyqed_total_kj_mol": pyqed["total_kj_mol"],
        "openmm_total_kj_mol": openmm_total,
        "delta_total_openmm_minus_pyqed_kj_mol": openmm_total - pyqed["total_kj_mol"],
        "pyqed_lj_kj_mol": pyqed["lj_kj_mol"],
        "openmm_lj_kj_mol": openmm_lj,
        "delta_lj_openmm_minus_pyqed_kj_mol": openmm_lj - pyqed["lj_kj_mol"],
        "pyqed_coulomb_kj_mol": pyqed["coulomb_kj_mol"],
        "openmm_coulomb_kj_mol": openmm_coulomb,
        "delta_coulomb_openmm_minus_pyqed_kj_mol": openmm_coulomb - pyqed["coulomb_kj_mol"],
    }
    with open(path, "w") as handle:
        for key, value in rows.items():
            if isinstance(value, float):
                handle.write(f"{key}: {value:.10f}\n")
            else:
                handle.write(f"{key}: {value}\n")
    return rows


def write_pme_decomposition_report(atoms, path, lj_energy_shift=False):
    requested = pyqed_pme_components(atoms, lj_energy_shift=lj_energy_shift)
    effective_cutoff = _openmm_effective_cutoff_bohr(atoms)
    effective = pyqed_pme_components(
        atoms,
        lj_energy_shift=lj_energy_shift,
        real_cutoff=effective_cutoff,
        lj_cutoff=effective_cutoff,
    )
    openmm_total = openmm_single_point(atoms, nonbonded_method="pme")
    openmm_lj = openmm_single_point(
        atoms,
        nonbonded_method="pme",
        include_lj=True,
        include_coulomb=False,
    )
    openmm_coulomb = openmm_single_point(
        atoms,
        nonbonded_method="pme",
        include_lj=False,
        include_coulomb=True,
    )
    rows = {
        "requested_cutoff_angstrom": float(atoms.calc.coulomb_cutoff * au2nm * 10.0),
        "openmm_effective_cutoff_angstrom": float(effective_cutoff * au2nm * 10.0),
        "pyqed_requested_total_kj_mol": requested["total_kj_mol"],
        "pyqed_requested_lj_kj_mol": requested["lj_kj_mol"],
        "pyqed_requested_coulomb_kj_mol": requested["coulomb_kj_mol"],
        "pyqed_requested_coulomb_real_kj_mol": requested["coulomb_real_kj_mol"],
        "pyqed_requested_coulomb_reciprocal_kj_mol": requested["coulomb_reciprocal_kj_mol"],
        "pyqed_requested_coulomb_self_kj_mol": requested["coulomb_self_kj_mol"],
        "pyqed_requested_coulomb_exclusion_correction_kj_mol": requested["coulomb_exclusion_correction_kj_mol"],
        "pyqed_requested_coulomb_scaled_pair_correction_kj_mol": requested["coulomb_scaled_pair_correction_kj_mol"],
        "pyqed_effective_total_kj_mol": effective["total_kj_mol"],
        "pyqed_effective_lj_kj_mol": effective["lj_kj_mol"],
        "pyqed_effective_coulomb_kj_mol": effective["coulomb_kj_mol"],
        "pyqed_effective_coulomb_real_kj_mol": effective["coulomb_real_kj_mol"],
        "pyqed_effective_coulomb_reciprocal_kj_mol": effective["coulomb_reciprocal_kj_mol"],
        "pyqed_effective_coulomb_self_kj_mol": effective["coulomb_self_kj_mol"],
        "pyqed_effective_coulomb_exclusion_correction_kj_mol": effective["coulomb_exclusion_correction_kj_mol"],
        "pyqed_effective_coulomb_scaled_pair_correction_kj_mol": effective["coulomb_scaled_pair_correction_kj_mol"],
        "openmm_total_kj_mol": openmm_total,
        "openmm_lj_kj_mol": openmm_lj,
        "openmm_coulomb_kj_mol": openmm_coulomb,
        "delta_total_openmm_minus_pyqed_requested_kj_mol": openmm_total - requested["total_kj_mol"],
        "delta_total_openmm_minus_pyqed_effective_kj_mol": openmm_total - effective["total_kj_mol"],
        "delta_lj_openmm_minus_pyqed_effective_kj_mol": openmm_lj - effective["lj_kj_mol"],
        "delta_coulomb_openmm_minus_pyqed_effective_kj_mol": openmm_coulomb - effective["coulomb_kj_mol"],
        "cutoff_effect_on_pyqed_total_kj_mol": effective["total_kj_mol"] - requested["total_kj_mol"],
        "cutoff_effect_on_pyqed_lj_kj_mol": effective["lj_kj_mol"] - requested["lj_kj_mol"],
        "cutoff_effect_on_pyqed_coulomb_real_kj_mol": effective["coulomb_real_kj_mol"] - requested["coulomb_real_kj_mol"],
    }
    with open(path, "w") as handle:
        for key, value in rows.items():
            if isinstance(value, float):
                handle.write(f"{key}: {value:.10f}\n")
            else:
                handle.write(f"{key}: {value}\n")
    return rows


def _openmm_effective_cutoff_bohr(atoms):
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    return min(float(atoms.calc.coulomb_cutoff), 0.49 * float(np.min(lengths)))


def pyqed_pme_components(
    atoms,
    lj_energy_shift=False,
    real_cutoff=None,
    lj_cutoff=None,
):
    calc = atoms.calc
    positions = atoms.get_positions()
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    charges = np.asarray(calc.charges, dtype=float)
    real_cutoff = float(calc.coulomb_cutoff if real_cutoff is None else real_cutoff)
    lj_cutoff = float(calc.lj_cutoff if lj_cutoff is None else lj_cutoff)
    forces = np.zeros_like(positions)
    real = _add_ewald_real(
        positions,
        lengths,
        forces,
        charges,
        calc.coulomb_constant,
        calc.ewald_alpha,
        real_cutoff,
    )
    forces = np.zeros_like(positions)
    reciprocal = _add_pme_reciprocal(
        positions,
        lengths,
        forces,
        charges,
        calc.coulomb_constant,
        calc.ewald_alpha,
        calc.pme_mesh,
        getattr(calc, "pme_order", 4),
    )
    self_energy = -calc.coulomb_constant * calc.ewald_alpha / np.sqrt(np.pi) * float(np.dot(charges, charges))
    forces = np.zeros_like(positions)
    excluded = _add_coulomb_pairs(
        positions,
        atoms.get_cell(),
        atoms.get_pbc(),
        forces,
        charges,
        calc.coulomb_constant,
        only_pairs=calc.nonbonded_exclusions,
    )
    forces = np.zeros_like(positions)
    scaled_pair_correction = _add_coulomb_scaled_pairs(
        positions,
        atoms.get_cell(),
        atoms.get_pbc(),
        forces,
        charges,
        calc.coulomb_constant,
        calc.coulomb_pair_scales,
    )
    coulomb = real + reciprocal + self_energy - excluded + scaled_pair_correction
    lj = _pyqed_lj_pair_sum(atoms, cutoff=lj_cutoff, energy_shift=lj_energy_shift)
    total = lj + coulomb
    return {
        "total_kj_mol": float(total * HARTREE_TO_KJ_MOL),
        "lj_kj_mol": float(lj * HARTREE_TO_KJ_MOL),
        "coulomb_kj_mol": float(coulomb * HARTREE_TO_KJ_MOL),
        "coulomb_real_kj_mol": float(real * HARTREE_TO_KJ_MOL),
        "coulomb_reciprocal_kj_mol": float(reciprocal * HARTREE_TO_KJ_MOL),
        "coulomb_self_kj_mol": float(self_energy * HARTREE_TO_KJ_MOL),
        "coulomb_exclusion_correction_kj_mol": float((-excluded) * HARTREE_TO_KJ_MOL),
        "coulomb_scaled_pair_correction_kj_mol": float(scaled_pair_correction * HARTREE_TO_KJ_MOL),
    }


def _pyqed_lj_pair_sum(atoms, cutoff, energy_shift=False):
    calc = atoms.calc
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    epsilon = np.asarray(calc.lj_epsilon, dtype=float)
    sigma = np.asarray(calc.lj_sigma, dtype=float)
    base_exclusions = set(calc.nonbonded_exclusions)
    base_exclusions.update(calc.lj_pair_scales)
    lj = 0.0

    def pair_lj(i, j, distance):
        epsilon_ij = np.sqrt(epsilon[i] * epsilon[j])
        if epsilon_ij == 0.0:
            return 0.0
        sigma_ij = 0.5 * (sigma[i] + sigma[j])
        sr6 = (sigma_ij / distance) ** 6
        value = 4.0 * epsilon_ij * (sr6 * sr6 - sr6)
        if energy_shift:
            sr6_cutoff = (sigma_ij / cutoff) ** 6
            value -= 4.0 * epsilon_ij * (sr6_cutoff * sr6_cutoff - sr6_cutoff)
        return value

    for i in range(len(atoms) - 1):
        for j in range(i + 1, len(atoms)):
            pair = (i, j)
            if pair in base_exclusions:
                continue
            distance = float(np.linalg.norm(minimum_image(positions[i] - positions[j], cell, pbc)))
            if distance <= cutoff:
                lj += pair_lj(i, j, distance)
    for pair, scale in calc.lj_pair_scales.items():
        if scale == 0.0:
            continue
        i, j = pair
        distance = float(np.linalg.norm(minimum_image(positions[i] - positions[j], cell, pbc)))
        lj += scale * pair_lj(i, j, distance)
    return lj


def pyqed_direct_cutoff_components(atoms, lj_energy_shift=False):
    calc = atoms.calc
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    charges = np.asarray(calc.charges, dtype=float)
    epsilon = np.asarray(calc.lj_epsilon, dtype=float)
    sigma = np.asarray(calc.lj_sigma, dtype=float)
    cutoff = min(float(calc.coulomb_cutoff), 0.49 * float(np.min(atoms.get_cell().lengths())))
    base_exclusions = set(calc.nonbonded_exclusions)
    base_exclusions.update(calc.lj_pair_scales)
    base_exclusions.update(calc.coulomb_pair_scales)
    lj = 0.0
    coulomb = 0.0

    def pair_lj(i, j, distance):
        epsilon_ij = np.sqrt(epsilon[i] * epsilon[j])
        if epsilon_ij == 0.0:
            return 0.0
        sigma_ij = 0.5 * (sigma[i] + sigma[j])
        sr6 = (sigma_ij / distance) ** 6
        value = 4.0 * epsilon_ij * (sr6 * sr6 - sr6)
        if lj_energy_shift:
            sr6_cutoff = (sigma_ij / cutoff) ** 6
            value -= 4.0 * epsilon_ij * (sr6_cutoff * sr6_cutoff - sr6_cutoff)
        return value

    for i in range(len(atoms) - 1):
        for j in range(i + 1, len(atoms)):
            pair = (i, j)
            if pair in base_exclusions:
                continue
            rij = minimum_image(positions[i] - positions[j], cell, pbc)
            distance = float(np.linalg.norm(rij))
            if distance > cutoff:
                continue
            lj += pair_lj(i, j, distance)
            charge_product = charges[i] * charges[j]
            if charge_product != 0.0:
                coulomb += calc.coulomb_constant * charge_product / distance

    scaled_pairs = set(calc.lj_pair_scales)
    scaled_pairs.update(calc.coulomb_pair_scales)
    for pair in sorted(scaled_pairs):
        i, j = pair
        rij = minimum_image(positions[i] - positions[j], cell, pbc)
        distance = float(np.linalg.norm(rij))
        lj_scale = calc.lj_pair_scales.get(pair, 0.0)
        coulomb_scale = calc.coulomb_pair_scales.get(pair, 0.0)
        if lj_scale != 0.0:
            lj += lj_scale * pair_lj(i, j, distance)
        charge_product = charges[i] * charges[j]
        if coulomb_scale != 0.0 and charge_product != 0.0:
            coulomb += coulomb_scale * calc.coulomb_constant * charge_product / distance

    total = lj + coulomb
    return {
        "total_hartree": float(total),
        "total_kj_mol": float(total * HARTREE_TO_KJ_MOL),
        "lj_hartree": float(lj),
        "lj_kj_mol": float(lj * HARTREE_TO_KJ_MOL),
        "coulomb_hartree": float(coulomb),
        "coulomb_kj_mol": float(coulomb * HARTREE_TO_KJ_MOL),
        "residual_hartree": 0.0,
        "residual_kj_mol": 0.0,
        "positions_shape": list(positions.shape),
    }


def main():
    args = apply_preset(parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms, atom_types = build_membrane(
        args.nx,
        args.ny,
        waters_per_lipid=args.waters_per_lipid,
        salt_pairs=args.salt_pairs,
        coulomb_method="cutoff"
        if args.nonbonded_method in {"cutoff", "direct-cutoff", "nocutoff"}
        else "pme",
        pme_mesh=args.pme_mesh,
        lipid_spacing_angstrom=args.lipid_spacing,
    )
    atoms.calc.ewald_alpha = float(args.ewald_alpha)
    atoms.calc.pme_mesh = np.array([args.pme_mesh, args.pme_mesh, args.pme_mesh], dtype=int)
    pdb_path = output_dir / "pyqed_membrane_reference.pdb"
    manifest_path = output_dir / "pyqed_membrane_openmm_manifest.json"
    write_pdb(atoms, atom_types, pdb_path)
    write_manifest(atoms, atom_types, manifest_path)
    pyqed_energy_path = output_dir / "pyqed_energy_components.txt"
    lj_energy_shift = not args.no_lj_shift
    write_pyqed_energy_report(atoms, pyqed_energy_path, lj_energy_shift=lj_energy_shift)
    if args.no_lj_shift:
        atoms.calc.lj_energy_shift = False
    print(f"pdb: {pdb_path}")
    print(f"manifest: {manifest_path}")
    print(f"pyqed_energy: {pyqed_energy_path}")
    if args.export_only:
        return
    try:
        if args.nonbonded_method in {"cutoff", "direct-cutoff", "nocutoff"}:
            parity_path = output_dir / "openmm_pyqed_parity.txt"
            write_parity_report(
                atoms,
                parity_path,
                nonbonded_method=args.nonbonded_method,
                lj_energy_shift=lj_energy_shift,
            )
            print(f"parity: {parity_path}")
        if args.nonbonded_method == "pme":
            pme_decomposition_path = output_dir / "pme_decomposition.txt"
            write_pme_decomposition_report(
                atoms,
                pme_decomposition_path,
                lj_energy_shift=lj_energy_shift,
            )
            print(f"pme_decomposition: {pme_decomposition_path}")
        summary = run_openmm_reference(
            atoms,
            atom_types,
            args.steps,
            args.timestep_fs,
            args.temperature,
            args.friction_ps,
            output_dir,
            minimize=not args.skip_minimize,
            minimize_tolerance=args.minimize_tolerance,
            minimize_iterations=args.minimize_iterations,
            nonbonded_method=args.nonbonded_method,
            lj_energy_shift=lj_energy_shift,
            snapshot_interval=args.snapshot_interval,
        )
    except RuntimeError as exc:
        print(f"openmm: unavailable ({exc})")
    else:
        print(f"openmm_summary: {summary}")


if __name__ == "__main__":
    main()
