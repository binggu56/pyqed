#!/usr/bin/env python3
"""Run a small methanol-in-water MD smoke example.

The solute parameters are intentionally compact and example-grade.  They are
enough to exercise the molecule-in-solvent workflow: solute force-field JSON,
rigid TIP3P solvent, PME electrostatics, Kelvin velocity initialization,
trajectory/energy output, and restart writing.
"""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.md import (
    backend_status,
    EnergyLogger,
    Langevin,
    XYZTrajectoryWriter,
    dipole_moment,
    hydrogen_bonds,
    mm_from_topology,
    radial_distribution,
    read_restart,
    run_solvent_equilibration,
    set_maxwell_boltzmann_velocities,
    soft_relaxation,
    solvent_shell_count,
    solute_from_parameters,
    solvate_box,
    steepest_descent,
    water_count_for_density,
    water_density,
    write_minimization_log,
    write_restart,
)
from pyqed.units import au2angstrom, au2debye, au2fs, fs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--waters", type=int, default=3)
    parser.add_argument("--density", type=float, default=None, help="Target water density in g/cm^3; overrides --waters when set.")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--box-angstrom", type=float, default=14.0)
    parser.add_argument("--spacing-angstrom", type=float, default=3.4)
    parser.add_argument("--min-distance-angstrom", type=float, default=2.2)
    parser.add_argument("--water-oxygen-min-angstrom", type=float, default=None)
    parser.add_argument("--placement-relaxation", type=float, default=1.0)
    parser.add_argument("--placement", choices=("grid", "random"), default="grid")
    parser.add_argument("--placement-attempts", type=int, default=10000)
    parser.add_argument("--cutoff-angstrom", type=float, default=6.0)
    parser.add_argument("--mesh", type=int, default=16)
    parser.add_argument("--friction", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--minimize-steps", type=int, default=0)
    parser.add_argument("--minimize-step-angstrom", type=float, default=0.01)
    parser.add_argument("--minimize-fmax", type=float, default=1e-4)
    parser.add_argument("--soft-relax", action="store_true", help="Run staged nonbonded soft relaxation before minimization/MD.")
    parser.add_argument("--preset", choices=("manual", "solvent-smoke"), default="manual")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--restart", default=None, help="Continue from a restart .npz instead of building a fresh box.")
    parser.add_argument("--backend", choices=("python", "openmm"), default="python")
    parser.add_argument("--electrostatics", choices=("pme", "cutoff"), default="pme")
    parser.add_argument("--write-analysis", action="store_true", help="Write analysis.json and rdf.dat.")
    parser.add_argument("--solute", default=str(Path(__file__).with_name("methanol_solute.json")))
    parser.add_argument("--output-dir", default="md_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status = backend_status(args.backend)
    if not status["available"]:
        raise RuntimeError(status["reason"])

    cutoff = args.cutoff_angstrom / au2angstrom
    solute = solute_from_parameters(args.solute, calculator=False)
    solute_atoms = len(solute)
    if args.restart:
        system, restart_metadata = read_restart(args.restart)
        system.calc = mm_from_topology(
            system.topology,
            coulomb_method=args.electrostatics,
            coulomb_cutoff=cutoff,
            lj_cutoff=cutoff,
            pme_mesh=(args.mesh, args.mesh, args.mesh),
        )
        nwaters = max((len(system) - solute_atoms) // 3, 0)
        system.solvation = {
            "placement": "restart",
            "requested_waters": nwaters,
            "placed_waters": nwaters,
            "density_g_cm3": water_density(system, solute_atoms=solute_atoms),
            "restart_metadata": restart_metadata,
        }
    else:
        waters = args.waters
        if args.density is not None:
            waters = water_count_for_density((args.box_angstrom / au2angstrom,) * 3, args.density)
        system = solvate_box(
            solute=solute,
            box_size=(args.box_angstrom / au2angstrom,) * 3,
            spacing=args.spacing_angstrom / au2angstrom,
            min_distance=args.min_distance_angstrom / au2angstrom,
            max_waters=waters,
            rigid=True,
            coulomb_method=args.electrostatics,
            coulomb_cutoff=cutoff,
            lj_cutoff=cutoff,
            pme_mesh=(args.mesh, args.mesh, args.mesh),
            placement=args.placement,
            seed=args.seed,
            max_attempts=args.placement_attempts,
            water_oxygen_min_distance=(
                None if args.water_oxygen_min_angstrom is None
                else args.water_oxygen_min_angstrom / au2angstrom
            ),
            placement_relaxation=args.placement_relaxation,
        )
    timestep = args.timestep_fs * fs
    trajectory_path = output_dir / "methanol_in_water.xyz"
    energy_path = output_dir / "methanol_in_water_energy.dat"
    restart_path = output_dir / "methanol_in_water_restart.npz"
    relaxation = minimization = protocol = None
    final_steps = args.steps
    final_time = 0.0
    if args.preset == "solvent-smoke":
        protocol = run_solvent_equilibration(
            system,
            timestep=timestep,
            temperature_K=args.temperature,
            friction=args.friction,
            production_steps=args.steps,
            warmup_steps=args.warmup_steps,
            minimize_steps=args.minimize_steps,
            minimize_max_step=args.minimize_step_angstrom / au2angstrom,
            minimize_fmax=args.minimize_fmax,
            soft_relax=args.soft_relax,
            soft_relax_stages=((0.1, 0.1, 5), (0.5, 0.5, 5), (1.0, 1.0, 5)),
            output_prefix=output_dir / "methanol_in_water_protocol",
            seed=args.seed,
        )
        production_stage = None
        for index, result in enumerate(protocol["results"]):
            if result["type"] == "soft_relax":
                relaxation = result["stages"]
            elif result["type"] == "minimize":
                minimization = result
            elif result["type"] == "langevin":
                production_stage = index
                final_steps = result["steps"]
                final_time = result["time"]
        if production_stage is not None:
            trajectory_path = output_dir / f"methanol_in_water_protocol_stage{production_stage}.xyz"
            energy_path = output_dir / f"methanol_in_water_protocol_stage{production_stage}_energy.dat"
    else:
        if args.soft_relax:
            relaxation = soft_relaxation(
                system,
                stages=((0.1, 0.1, 5), (0.5, 0.5, 5), (1.0, 1.0, 5)),
                max_step=args.minimize_step_angstrom / au2angstrom,
                fmax=args.minimize_fmax,
            )
            write_minimization_log(output_dir / "methanol_in_water_soft_relax.dat", relaxation)
        if args.minimize_steps:
            minimization = steepest_descent(
                system,
                steps=args.minimize_steps,
                max_step=args.minimize_step_angstrom / au2angstrom,
                fmax=args.minimize_fmax,
            )
            write_minimization_log(output_dir / "methanol_in_water_minimize.dat", minimization)
        if not args.restart or not system.has("momenta"):
            set_maxwell_boltzmann_velocities(system, args.temperature, seed=args.seed)

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

        try:
            dynamics.run(args.steps)
        finally:
            writer.close()
            logger.close()
        final_steps = dynamics.get_number_of_steps()
        final_time = dynamics.get_time()

    write_restart(system, restart_path, step=final_steps, time=final_time, metadata={"example": "methanol_in_water"})
    analysis_path = rdf_path = None
    if args.write_analysis:
        analysis_path = output_dir / "analysis.json"
        rdf_path = output_dir / "rdf.dat"
        _write_analysis(system, solute_atoms, cutoff, analysis_path, rdf_path)
    print(f"atoms: {len(system)}")
    print(f"waters: {system.solvation['placed_waters']}")
    print(f"density_g_cm3: {water_density(system, solute_atoms=len(solute)):.6f}")
    if relaxation is not None:
        print(f"soft_relax_stages: {len(relaxation)}")
    if minimization is not None:
        print(f"minimize_steps: {minimization['steps']}")
        print(f"minimize_fmax: {minimization['fmax']:.12e}")
        print(f"minimize_energy_hartree: {minimization['energy']:.12e}")
    if protocol is not None:
        print(f"protocol_stages: {len(protocol['results'])}")
    print(f"steps: {final_steps}")
    print(f"time_fs: {final_time * au2fs:.6f}")
    print(f"temperature_K: {system.get_temperature(remove_center_of_mass=True):.6f}")
    print(f"potential_hartree: {system.get_potential_energy():.12e}")
    print(f"trajectory: {trajectory_path}")
    print(f"energy_log: {energy_path}")
    print(f"restart: {restart_path}")
    if analysis_path is not None:
        print(f"analysis: {analysis_path}")
        print(f"rdf: {rdf_path}")


def _write_analysis(system, solute_atoms, cutoff, analysis_path, rdf_path):
    water_o = list(range(solute_atoms, len(system), 3))
    solute_indices = list(range(solute_atoms))
    methanol_oxygen = [1] if solute_atoms > 1 else solute_indices
    r, hist = radial_distribution(system, methanol_oxygen, water_o, r_max=cutoff, bins=80)
    with open(rdf_path, "w") as handle:
        handle.write("r_bohr count\n")
        for radius, count in zip(r, hist):
            handle.write(f"{radius:.12e} {int(count)}\n")
    donor_pairs = [(1, 5)] if solute_atoms >= 6 else []
    hbonds = hydrogen_bonds(system, donor_pairs, water_o, distance_cutoff=2.5 / au2angstrom)
    dipole = dipole_moment(system, indices=solute_indices)
    analysis = {
        "solute_atoms": solute_atoms,
        "water_oxygens": len(water_o),
        "shell_water_oxygen_count": solvent_shell_count(
            system, solute_indices, water_o, cutoff=3.5 / au2angstrom
        ),
        "methanol_hbond_count": len(hbonds),
        "solute_dipole_au": dipole.tolist(),
        "solute_dipole_debye": (dipole * au2debye).tolist(),
    }
    with open(analysis_path, "w") as handle:
        json.dump(analysis, handle, indent=2)


if __name__ == "__main__":
    main()
