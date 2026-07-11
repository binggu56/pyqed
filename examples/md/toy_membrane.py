#!/usr/bin/env python3
"""Run a tiny toy lipid-bilayer MD smoke example.

This is a development smoke test for membrane-shaped systems, not a
production biological membrane simulation.
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
    SemiIsotropicBoxController,
    XYZTrajectoryWriter,
    add_ions_random,
    area_per_lipid,
    leaflet_indices,
    lipid_bilayer,
    set_maxwell_boltzmann_velocities,
    solvate_membrane,
    write_restart,
)
from pyqed.units import au2fs, fs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=2, help="Lipids per leaflet along x.")
    parser.add_argument("--ny", type=int, default=2, help="Lipids per leaflet along y.")
    parser.add_argument("--waters-per-side", type=int, default=4)
    parser.add_argument("--salt-pairs", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument("--friction", type=float, default=1e-2)
    parser.add_argument("--area-per-lipid", type=float, default=60.0, help="Angstrom^2.")
    parser.add_argument("--thickness", type=float, default=36.0, help="Headgroup-headgroup distance in Angstrom.")
    parser.add_argument("--water-padding", type=float, default=18.0, help="Water slab padding in Angstrom.")
    parser.add_argument("--electrostatics", choices=("cutoff", "pme"), default="cutoff")
    parser.add_argument("--cutoff-angstrom", type=float, default=10.0)
    parser.add_argument("--mesh", type=int, default=24)
    parser.add_argument("--box-control", action="store_true", help="Weakly relax xy area and z length toward initial targets.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="md_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    membrane = lipid_bilayer(
        nx=args.nx,
        ny=args.ny,
        area_per_lipid=args.area_per_lipid,
        thickness=args.thickness,
        water_padding=args.water_padding,
        calculator=False,
        seed=args.seed,
    )
    system = solvate_membrane(
        membrane,
        max_waters_per_side=args.waters_per_side,
        rigid=True,
        seed=args.seed,
        coulomb_method=args.electrostatics,
        coulomb_cutoff=args.cutoff_angstrom,
        lj_cutoff=args.cutoff_angstrom,
        pme_mesh=(args.mesh, args.mesh, args.mesh),
    )
    if args.salt_pairs:
        ion_symbols = ["Na", "Cl"] * int(args.salt_pairs)
        system = add_ions_random(
            system,
            ions=ion_symbols,
            seed=args.seed,
            coulomb_method=args.electrostatics,
            coulomb_cutoff=args.cutoff_angstrom,
            lj_cutoff=args.cutoff_angstrom,
            pme_mesh=(args.mesh, args.mesh, args.mesh),
        )
    set_maxwell_boltzmann_velocities(system, args.temperature, seed=args.seed)

    timestep = args.timestep_fs * fs
    dynamics = Langevin(
        system,
        timestep=timestep,
        temperature_K=args.temperature,
        friction=args.friction,
    )
    trajectory_path = output_dir / "toy_membrane.xyz"
    energy_path = output_dir / "toy_membrane_energy.dat"
    restart_path = output_dir / "toy_membrane_restart.npz"
    writer = XYZTrajectoryWriter(system, trajectory_path, dynamics=dynamics)
    logger = EnergyLogger(system, energy_path, dynamics=dynamics)
    dynamics.attach(writer)
    dynamics.attach(logger)
    if args.box_control:
        lengths = system.get_cell().lengths()
        controller = SemiIsotropicBoxController(
            system,
            target_area=lengths[0] * lengths[1],
            target_z=lengths[2],
            coupling=0.02,
        )
        dynamics.attach(controller)

    try:
        dynamics.run(args.steps)
    finally:
        writer.close()
        logger.close()

    write_restart(
        system,
        restart_path,
        step=dynamics.get_number_of_steps(),
        time=dynamics.get_time(),
        metadata={"example": "toy_membrane"},
    )
    print(f"atoms: {len(system)}")
    print(f"lipids: {system.membrane['total_lipids']}")
    print(f"waters: {system.solvation['placed_waters']}")
    print(f"ions: {len(getattr(system, 'ions', {}).get('placed_ions', []))}")
    print(f"upper_leaflet_atoms: {len(leaflet_indices(system, 1))}")
    print(f"lower_leaflet_atoms: {len(leaflet_indices(system, -1))}")
    print(f"area_per_lipid_angstrom2: {area_per_lipid(system):.6f}")
    print(f"steps: {dynamics.get_number_of_steps()}")
    print(f"time_fs: {dynamics.get_time() * au2fs:.6f}")
    print(f"temperature_K: {system.get_temperature(remove_center_of_mass=True):.6f}")
    print(f"potential_hartree: {system.get_potential_energy():.12e}")
    print(f"trajectory: {trajectory_path}")
    print(f"energy_log: {energy_path}")
    print(f"restart: {restart_path}")


if __name__ == "__main__":
    main()
