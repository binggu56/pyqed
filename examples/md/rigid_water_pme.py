#!/usr/bin/env python3
"""Run a tiny rigid TIP3P water PME molecular-dynamics example.

Internal MD units are atomic units:

* length: Bohr
* energy: Hartree
* mass: electron mass
* time: atomic time
* temperature input/output: Kelvin at the public API boundary
"""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.md import (  # noqa: E402
    EnergyLogger,
    Langevin,
    XYZTrajectoryWriter,
    set_maxwell_boltzmann_velocities,
    solvate_box,
)
from pyqed.units import au2angstrom, au2fs, au2kcalmol, fs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10, help="MD steps to run.")
    parser.add_argument("--waters", type=int, default=2, help="Maximum number of waters.")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K.")
    parser.add_argument("--timestep-fs", type=float, default=0.5, help="Timestep in fs.")
    parser.add_argument("--box-angstrom", type=float, default=10.0, help="Cubic box length in Angstrom.")
    parser.add_argument("--spacing-angstrom", type=float, default=3.2, help="Water grid spacing in Angstrom.")
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0, help="PME real-space cutoff in Angstrom.")
    parser.add_argument("--friction", type=float, default=1e-3, help="Langevin friction in inverse atomic time.")
    parser.add_argument("--mesh", type=int, default=16, help="Cubic PME mesh size.")
    parser.add_argument("--seed", type=int, default=3, help="Random seed for initial velocities.")
    parser.add_argument("--output-dir", default="md_outputs", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    box_length = args.box_angstrom / au2angstrom
    spacing = args.spacing_angstrom / au2angstrom
    cutoff = args.cutoff_angstrom / au2angstrom
    timestep = args.timestep_fs * fs

    atoms = solvate_box(
        box_size=(box_length, box_length, box_length),
        spacing=spacing,
        max_waters=args.waters,
        rigid=True,
        coulomb_method="pme",
        coulomb_cutoff=cutoff,
        ewald_alpha=0.35,
        pme_mesh=(args.mesh, args.mesh, args.mesh),
        lj_cutoff=cutoff,
    )
    set_maxwell_boltzmann_velocities(atoms, args.temperature, seed=args.seed)

    trajectory_path = output_dir / "rigid_water_pme.xyz"
    energy_path = output_dir / "rigid_water_pme_energy.dat"
    dynamics = Langevin(
        atoms,
        timestep=timestep,
        temperature_K=args.temperature,
        friction=args.friction,
    )
    writer = XYZTrajectoryWriter(atoms, trajectory_path, dynamics=dynamics)
    logger = EnergyLogger(atoms, energy_path, dynamics=dynamics)
    dynamics.attach(writer)
    dynamics.attach(logger)

    try:
        dynamics.run(args.steps)
    finally:
        writer.close()
        logger.close()

    potential = atoms.get_potential_energy()
    kinetic = atoms.get_kinetic_energy()
    print(f"atoms: {len(atoms)}")
    print(f"steps: {dynamics.get_number_of_steps()}")
    print(f"time_fs: {dynamics.get_time() * au2fs:.6f}")
    print(f"temperature_K: {atoms.get_temperature(remove_center_of_mass=True):.6f}")
    print(f"potential_hartree: {potential:.12e}")
    print(f"kinetic_hartree: {kinetic:.12e}")
    print(f"total_hartree: {potential + kinetic:.12e}")
    print(f"total_kcal_mol: {(potential + kinetic) * au2kcalmol:.12e}")
    print(f"trajectory: {trajectory_path}")
    print(f"energy_log: {energy_path}")


if __name__ == "__main__":
    main()
