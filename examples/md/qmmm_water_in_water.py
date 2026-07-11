#!/usr/bin/env python3
"""Run a tiny QM water in MM water QM/MM MD example.

The solute is one water molecule treated with PyQED RHF/STO-3G.  The solvent is
a small rigid TIP3P water box treated by the MD ``MM`` calculator.  This is a
workflow and stability smoke test, not a production water simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed import Molecule  # noqa: E402
from pyqed.md import (  # noqa: E402
    Atoms,
    EnergyLogger,
    Langevin,
    MM,
    QMMM,
    Topology,
    XYZTrajectoryWriter,
    set_maxwell_boltzmann_velocities,
    solvate_box,
    tip3p_parameters,
)
from pyqed.units import au2angstrom, au2fs, au2kcalmol, fs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5, help="MD steps to run.")
    parser.add_argument("--waters", type=int, default=2, help="Maximum number of MM waters.")
    parser.add_argument("--temperature", type=float, default=50.0, help="Temperature in K.")
    parser.add_argument("--timestep-fs", type=float, default=0.02, help="Timestep in fs.")
    parser.add_argument("--box-angstrom", type=float, default=9.0, help="Cubic box length in Angstrom.")
    parser.add_argument("--spacing-angstrom", type=float, default=3.2, help="Water grid spacing in Angstrom.")
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0, help="MM nonbonded cutoff in Angstrom.")
    parser.add_argument("--friction", type=float, default=1e-3, help="Langevin friction in inverse atomic time.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for initial velocities.")
    parser.add_argument("--output-dir", default="md_outputs", help="Output directory.")
    return parser.parse_args()


def water_solute(center):
    params = tip3p_parameters()
    center = np.asarray(center, dtype=float)
    theta = np.deg2rad(params["hoh_angle"])
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [params["oh_distance"], 0.0, 0.0],
            [
                params["oh_distance"] * np.cos(theta),
                params["oh_distance"] * np.sin(theta),
                0.0,
            ],
        ]
    )
    # Put the oxygen near the requested center; this keeps the builder simple.
    positions = center + local
    solute = Atoms(
        [
            ["O", tuple(positions[0])],
            ["H", tuple(positions[1])],
            ["H", tuple(positions[2])],
        ]
    )
    solute.topology = Topology(
        charges=[0.0, 0.0, 0.0],
        lj_epsilon=[0.0, 0.0, 0.0],
        lj_sigma=[0.0, 0.0, 0.0],
        molecule_ids=[0, 0, 0],
    )
    solute.set_array("charges", solute.topology.charges, float, ())
    solute.set_array("lj_epsilon", solute.topology.lj_epsilon, float, ())
    solute.set_array("lj_sigma", solute.topology.lj_sigma, float, ())
    solute.set_array("molecule_ids", solute.topology.molecule_ids, int, ())
    return solute, positions


def mm_from_topology(system, cutoff):
    topology = system.topology
    return MM(
        bonds=topology.bonds,
        angles=topology.angles,
        angle_unit="degree",
        charges=topology.charges,
        coulomb_constant=1.0,
        coulomb_cutoff=cutoff,
        lj_epsilon=topology.lj_epsilon,
        lj_sigma=topology.lj_sigma,
        lj_cutoff=cutoff,
        exclude_bonded=True,
        exclude_angles=True,
    )


def qm_water_reference(positions):
    symbols = ("O", "H", "H")
    atom = "; ".join(
        f"{symbol} {x:.16g} {y:.16g} {z:.16g}"
        for symbol, (x, y, z) in zip(symbols, positions)
    )
    mol = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    mol.build(driver="builtin")
    return mol.RHF()


def max_constraint_error(system):
    if not system.constraints:
        return 0.0
    errors = []
    positions = system.get_positions()
    for constraint in system.constraints:
        targets = constraint._targets(system)
        for (i, j), target in zip(constraint.pairs, targets):
            errors.append(abs(np.linalg.norm(positions[i] - positions[j]) - target))
    return float(max(errors, default=0.0))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    box_length = args.box_angstrom / au2angstrom
    spacing = args.spacing_angstrom / au2angstrom
    cutoff = args.cutoff_angstrom / au2angstrom
    timestep = args.timestep_fs * fs

    solute, solute_positions = water_solute(
        center=(0.5 * box_length, 0.5 * box_length, 0.5 * box_length)
    )
    system = solvate_box(
        solute=solute,
        box_size=(box_length, box_length, box_length),
        spacing=spacing,
        min_distance=2.4 / au2angstrom,
        max_waters=args.waters,
        rigid=True,
        lj_cutoff=cutoff,
        coulomb_cutoff=cutoff,
    )

    qm_indices = np.array([0, 1, 2], dtype=int)
    mm_indices = np.arange(3, len(system), dtype=int)
    system.calc = QMMM(
        qm=qm_water_reference(solute_positions),
        mm=mm_from_topology(system, cutoff),
        qm_indices=qm_indices,
        mm_indices=mm_indices,
        electrostatic_embedding=True,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    set_maxwell_boltzmann_velocities(system, temperature=args.temperature, seed=args.seed)

    trajectory_path = output_dir / "qmmm_water_in_water.xyz"
    energy_path = output_dir / "qmmm_water_in_water_energy.dat"
    dynamics = Langevin(
        system,
        timestep=timestep,
        temperature_K=args.temperature,
        friction=args.friction,
    )
    writer = XYZTrajectoryWriter(system, trajectory_path, dynamics=dynamics)
    logger = EnergyLogger(system, energy_path, dynamics=dynamics)
    dynamics.attach(writer)
    dynamics.attach(logger)

    initial_positions = system.get_positions().copy()
    initial_energy = system.get_potential_energy()
    initial_forces = system.get_forces()
    initial_constraint_error = max_constraint_error(system)

    try:
        dynamics.run(args.steps)
    finally:
        writer.close()
        logger.close()

    final_positions = system.get_positions()
    final_energy = system.get_potential_energy()
    final_forces = system.get_forces()
    components = dict(system.calc.results)
    displacement = np.linalg.norm(final_positions - initial_positions, axis=1)
    final_constraint_error = max_constraint_error(system)

    print(f"atoms: {len(system)}")
    print(f"qm_atoms: {len(qm_indices)}")
    print(f"mm_atoms: {len(mm_indices)}")
    print(f"steps: {dynamics.get_number_of_steps()}")
    print(f"time_fs: {dynamics.get_time() * au2fs:.6f}")
    print(f"initial_energy_hartree: {initial_energy:.12e}")
    print(f"final_energy_hartree: {final_energy:.12e}")
    print(f"final_qm_energy_hartree: {components['qm_energy']:.12e}")
    print(f"final_mm_energy_hartree: {components['mm_energy']:.12e}")
    print(f"final_embedding_energy_hartree: {components['embedding_energy']:.12e}")
    print(f"final_energy_kcal_mol: {final_energy * au2kcalmol:.12e}")
    print(f"initial_max_force: {np.max(np.linalg.norm(initial_forces, axis=1)):.12e}")
    print(f"final_max_force: {np.max(np.linalg.norm(final_forces, axis=1)):.12e}")
    print(f"final_qm_force_max: {components['qm_force_max']:.12e}")
    print(f"final_mm_force_max: {components['mm_force_max']:.12e}")
    print(f"final_point_charge_force_max: {components['point_charge_force_max']:.12e}")
    print(f"max_displacement_bohr: {np.max(displacement):.12e}")
    print(f"initial_constraint_error_bohr: {initial_constraint_error:.12e}")
    print(f"final_constraint_error_bohr: {final_constraint_error:.12e}")
    print(f"finite_positions: {bool(np.all(np.isfinite(final_positions)))}")
    print(f"finite_forces: {bool(np.all(np.isfinite(final_forces)))}")
    print(f"trajectory: {trajectory_path}")
    print(f"energy_log: {energy_path}")


if __name__ == "__main__":
    main()
