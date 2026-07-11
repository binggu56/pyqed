"""Run a short OpenMM DPPC membrane simulation from a pre-equilibrated patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=323.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--friction-ps", type=float, default=1.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--surface-tension-bar-nm", type=float, default=0.0)
    parser.add_argument("--barostat-interval", type=int, default=25)
    parser.add_argument("--minimize-iterations", type=int, default=0)
    parser.add_argument("--trajectory-interval", type=int, default=25)
    parser.add_argument("--energy-interval", type=int, default=10)
    parser.add_argument("--no-barostat", action="store_true")
    parser.add_argument("--output-dir", default="/private/tmp/pyqed_openmm_dppc_patch")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openmm, app, unit = _import_openmm()
    data_dir = Path(app.__file__).resolve().parent / "data"
    patch_path = data_dir / "DPPC.pdb"

    pdb = app.PDBFile(str(patch_path))
    forcefield = app.ForceField("charmm36.xml", "charmm36/water.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=5.0e-4,
    )
    if not args.no_barostat:
        system.addForce(
            openmm.MonteCarloMembraneBarostat(
                args.pressure_bar * unit.bar,
                args.surface_tension_bar_nm * unit.bar * unit.nanometer,
                args.temperature * unit.kelvin,
                openmm.MonteCarloMembraneBarostat.XYIsotropic,
                openmm.MonteCarloMembraneBarostat.ZFree,
                int(args.barostat_interval),
            )
        )

    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    initial_state = simulation.context.getState(getEnergy=True)
    initial_potential = _kjmol(initial_state.getPotentialEnergy(), unit)
    if args.minimize_iterations > 0:
        simulation.minimizeEnergy(maxIterations=int(args.minimize_iterations))

    minimized_state = simulation.context.getState(getEnergy=True, getPositions=True)
    minimized_potential = _kjmol(minimized_state.getPotentialEnergy(), unit)
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin)

    log_path = output_dir / "dppc_patch_energy.csv"
    trajectory_path = output_dir / "dppc_patch_trajectory.pdb"
    simulation.reporters.append(
        app.StateDataReporter(
            str(log_path),
            int(args.energy_interval),
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            separator=",",
        )
    )
    if args.trajectory_interval > 0:
        simulation.reporters.append(app.PDBReporter(str(trajectory_path), int(args.trajectory_interval)))

    if args.steps > 0:
        simulation.step(int(args.steps))

    final_state = simulation.context.getState(
        getEnergy=True,
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=True,
    )
    final_pdb_path = output_dir / "dppc_patch_final.pdb"
    with final_pdb_path.open("w") as handle:
        app.PDBFile.writeFile(pdb.topology, final_state.getPositions(), handle)

    minimized_pdb_path = output_dir / "dppc_patch_minimized.pdb"
    with minimized_pdb_path.open("w") as handle:
        app.PDBFile.writeFile(pdb.topology, minimized_state.getPositions(), handle)

    summary = _summary(
        pdb,
        final_state,
        unit,
        args,
        patch_path,
        initial_potential,
        minimized_potential,
        log_path,
        trajectory_path,
        minimized_pdb_path,
        final_pdb_path,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    render_path = output_dir / "dppc_patch_final_views.png"
    _render_views(pdb.topology, final_state.getPositions(), unit, render_path)

    print(summary_path)
    for key in (
        "atoms",
        "lipids",
        "waters",
        "steps",
        "time_ps",
        "initial_potential_kj_mol",
        "minimized_potential_kj_mol",
        "final_potential_kj_mol",
        "final_temperature_K",
        "area_per_lipid_angstrom2",
        "bilayer_thickness_angstrom",
        "barostat",
    ):
        print(f"{key}: {summary[key]}")
    print(f"final_pdb: {final_pdb_path}")
    print(f"trajectory: {trajectory_path}")
    print(f"energy_log: {log_path}")
    print(f"render: {render_path}")


def _import_openmm():
    try:
        import openmm
        import openmm.app as app
        from openmm import unit
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenMM is required for this example.") from exc
    return openmm, app, unit


def _kjmol(quantity, unit):
    return float(quantity.value_in_unit(unit.kilojoule_per_mole))


def _summary(
    pdb,
    final_state,
    unit,
    args,
    patch_path,
    initial_potential,
    minimized_potential,
    log_path,
    trajectory_path,
    minimized_pdb_path,
    final_pdb_path,
):
    residue_counts = {}
    lipid_indices = []
    phosphorus_z = []
    positions = np.asarray(final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    for residue in pdb.topology.residues():
        residue_counts[residue.name] = residue_counts.get(residue.name, 0) + 1
        if residue.name in {"DPP", "DPPC"}:
            atom_indices = [atom.index for atom in residue.atoms()]
            lipid_indices.append(atom_indices)
            for atom in residue.atoms():
                if atom.name == "P":
                    phosphorus_z.append(float(positions[atom.index, 2]))

    box_vectors = final_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    lx = float(np.linalg.norm(box_vectors[0]))
    ly = float(np.linalg.norm(box_vectors[1]))
    lipids = len(lipid_indices)
    area_per_lipid = lx * ly / max(lipids / 2.0, 1.0)
    bilayer_thickness = None
    if phosphorus_z:
        phosphorus_z = np.asarray(phosphorus_z, dtype=float)
        center = float(np.mean(phosphorus_z))
        upper = phosphorus_z[phosphorus_z >= center]
        lower = phosphorus_z[phosphorus_z < center]
        if len(upper) and len(lower):
            bilayer_thickness = float(np.mean(upper) - np.mean(lower))

    final_potential = _kjmol(final_state.getPotentialEnergy(), unit)
    kinetic = _kjmol(final_state.getKineticEnergy(), unit)
    temperature = _last_reported_temperature(log_path)

    return {
        "source_patch": str(patch_path),
        "forcefield": "charmm36.xml + charmm36/water.xml",
        "atoms": int(pdb.topology.getNumAtoms()),
        "residue_counts": residue_counts,
        "lipids": int(lipids),
        "waters": int(residue_counts.get("HOH", 0)),
        "steps": int(args.steps),
        "time_ps": float(args.steps * args.timestep_fs / 1000.0),
        "timestep_fs": float(args.timestep_fs),
        "temperature_target_K": float(args.temperature),
        "friction_ps": float(args.friction_ps),
        "barostat": not bool(args.no_barostat),
        "pressure_bar": float(args.pressure_bar),
        "surface_tension_bar_nm": float(args.surface_tension_bar_nm),
        "barostat_interval": int(args.barostat_interval),
        "initial_potential_kj_mol": float(initial_potential),
        "minimized_potential_kj_mol": float(minimized_potential),
        "final_potential_kj_mol": float(final_potential),
        "final_kinetic_kj_mol": float(kinetic),
        "final_temperature_K": float(temperature),
        "box_angstrom": [lx, ly, float(np.linalg.norm(box_vectors[2]))],
        "area_per_lipid_angstrom2": float(area_per_lipid),
        "bilayer_thickness_angstrom": bilayer_thickness,
        "energy_log": str(log_path),
        "trajectory": str(trajectory_path),
        "minimized_pdb": str(minimized_pdb_path),
        "final_pdb": str(final_pdb_path),
    }


def _last_reported_temperature(log_path):
    if not Path(log_path).exists():
        return float("nan")
    lines = [line.strip() for line in Path(log_path).read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return float("nan")
    header = [field.strip().lstrip("#").strip() for field in lines[0].split(",")]
    fields = [field.strip() for field in lines[-1].split(",")]
    for index, name in enumerate(header):
        if "Temperature" in name:
            return float(fields[index])
    return float("nan")


def _render_views(topology, positions, unit, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = np.asarray(positions.value_in_unit(unit.angstrom), dtype=float)
    residue_names = np.empty(len(coords), dtype=object)
    atom_names = np.empty(len(coords), dtype=object)
    for atom in topology.atoms():
        residue_names[atom.index] = atom.residue.name
        atom_names[atom.index] = atom.name

    lipid = np.isin(residue_names, ["DPP", "DPPC"])
    water = residue_names == "HOH"
    heads = lipid & np.isin(atom_names, ["N", "P", "O11", "O12", "O13", "O14"])
    tails = lipid & ~heads

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    styles = [
        ("tails", tails, "#777777", 1.0, 0.35),
        ("heads", heads, "#7b3294", 7.0, 0.8),
        ("water", water, "#2c7fb8", 1.0, 0.18),
    ]
    for ax, dims, title in ((axes[0], (0, 1), "Top view (x-y)"), (axes[1], (0, 2), "Side view (x-z)")):
        for label, mask, color, size, alpha in styles:
            ax.scatter(
                coords[mask, dims[0]],
                coords[mask, dims[1]],
                s=size,
                c=color,
                alpha=alpha,
                label=label,
                linewidths=0,
            )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Angstrom")
        ax.set_ylabel("Angstrom")
        ax.grid(True, alpha=0.18)
    axes[0].legend(loc="upper right", frameon=False, markerscale=4.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
