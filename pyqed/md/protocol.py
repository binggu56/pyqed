"""Small MD preparation/equilibration protocol helpers."""

from .io import EnergyLogger, XYZTrajectoryWriter
from .langevin import Langevin
from .minimize import soft_relaxation, steepest_descent, write_minimization_log
from .restart import write_restart
from .velocities import set_maxwell_boltzmann_velocities


def equilibrate(
    atoms,
    stages,
    output_prefix=None,
    seed=None,
):
    """Run a simple staged preparation/equilibration protocol.

    Stage dictionaries support:
    ``type='minimize'``, ``type='soft_relax'``, or ``type='langevin'``.
    """
    results = []
    for index, stage in enumerate(stages):
        stage_type = stage.get("type", "langevin")
        if stage_type == "minimize":
            result = steepest_descent(
                atoms,
                steps=stage.get("steps", 100),
                max_step=stage.get("max_step", 0.01),
                fmax=stage.get("fmax", 1e-4),
            )
            result["type"] = stage_type
        elif stage_type == "soft_relax":
            result = {
                "type": stage_type,
                "stages": soft_relaxation(
                    atoms,
                    stages=stage.get("stages", ((0.1, 0.1, 20), (1.0, 1.0, 20))),
                    max_step=stage.get("max_step", 0.01),
                    fmax=stage.get("fmax", 1e-4),
                ),
            }
        elif stage_type == "langevin":
            if stage.get("initialize_velocities", False):
                set_maxwell_boltzmann_velocities(
                    atoms,
                    stage.get("temperature_K", stage.get("temperature", 300.0)),
                    seed=seed,
                )
            dyn = Langevin(
                atoms,
                timestep=stage["timestep"],
                temperature_K=stage.get("temperature_K", stage.get("temperature", 300.0)),
                friction=stage.get("friction", 1e-3),
            )
            writer = logger = None
            if output_prefix is not None:
                writer = XYZTrajectoryWriter(atoms, f"{output_prefix}_stage{index}.xyz", dynamics=dyn)
                logger = EnergyLogger(atoms, f"{output_prefix}_stage{index}_energy.dat", dynamics=dyn)
                dyn.attach(writer)
                dyn.attach(logger)
            try:
                dyn.run(stage.get("steps", 1))
            finally:
                if writer is not None:
                    writer.close()
                if logger is not None:
                    logger.close()
            result = {
                "type": stage_type,
                "steps": dyn.get_number_of_steps(),
                "time": dyn.get_time(),
                "temperature_K": atoms.get_temperature(remove_center_of_mass=True),
                "energy": atoms.get_potential_energy(),
            }
        else:
            raise ValueError("stage type must be 'minimize', 'soft_relax', or 'langevin'.")

        if output_prefix is not None:
            if stage_type in {"minimize"}:
                write_minimization_log(f"{output_prefix}_stage{index}_minimize.dat", result)
            elif stage_type == "soft_relax":
                write_minimization_log(f"{output_prefix}_stage{index}_soft_relax.dat", result["stages"])
            write_restart(
                atoms,
                f"{output_prefix}_stage{index}.npz",
                step=result.get("steps", 0),
                time=result.get("time", 0.0),
                metadata={"stage": index, "type": stage_type},
            )
        results.append(result)
    return results


def solvent_equilibration_stages(
    timestep,
    temperature_K=300.0,
    friction=1e-3,
    production_steps=1000,
    warmup_steps=200,
    minimize_steps=200,
    minimize_max_step=0.01,
    minimize_fmax=1e-4,
    soft_relax=True,
    soft_relax_stages=((0.1, 0.1, 20), (0.5, 0.5, 20), (1.0, 1.0, 20)),
):
    """Return a compact solvent MD preparation/production stage list.

    All times and lengths are expected in atomic units; temperatures are Kelvin.
    """
    stages = []
    if soft_relax:
        stages.append(
            {
                "type": "soft_relax",
                "stages": soft_relax_stages,
                "max_step": minimize_max_step,
                "fmax": minimize_fmax,
            }
        )
    if minimize_steps:
        stages.append(
            {
                "type": "minimize",
                "steps": minimize_steps,
                "max_step": minimize_max_step,
                "fmax": minimize_fmax,
            }
        )
    if warmup_steps:
        stages.append(
            {
                "type": "langevin",
                "label": "warmup",
                "steps": warmup_steps,
                "timestep": timestep,
                "temperature_K": temperature_K,
                "friction": friction,
                "initialize_velocities": True,
            }
        )
    if production_steps:
        stages.append(
            {
                "type": "langevin",
                "label": "production",
                "steps": production_steps,
                "timestep": timestep,
                "temperature_K": temperature_K,
                "friction": friction,
                "initialize_velocities": not warmup_steps,
            }
        )
    return stages


def run_solvent_equilibration(
    atoms,
    timestep,
    temperature_K=300.0,
    friction=1e-3,
    production_steps=1000,
    warmup_steps=200,
    minimize_steps=200,
    minimize_max_step=0.01,
    minimize_fmax=1e-4,
    soft_relax=True,
    soft_relax_stages=((0.1, 0.1, 20), (0.5, 0.5, 20), (1.0, 1.0, 20)),
    output_prefix=None,
    seed=None,
):
    """Run a standard molecule-in-solvent preparation and production preset."""
    stages = solvent_equilibration_stages(
        timestep=timestep,
        temperature_K=temperature_K,
        friction=friction,
        production_steps=production_steps,
        warmup_steps=warmup_steps,
        minimize_steps=minimize_steps,
        minimize_max_step=minimize_max_step,
        minimize_fmax=minimize_fmax,
        soft_relax=soft_relax,
        soft_relax_stages=soft_relax_stages,
    )
    results = equilibrate(atoms, stages, output_prefix=output_prefix, seed=seed)
    return {"stages": stages, "results": results}
