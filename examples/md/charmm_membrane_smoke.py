#!/usr/bin/env python3
"""Smoke-check a local CHARMM-GUI-style membrane folder.

The script expects files such as ``step3_input.psf``, ``step3_input.pdb``, and
``toppar/*.prm`` / ``toppar/*.str``.  It exits successfully with a skip message
when the folder is absent or incomplete, so it can live in automated tests.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.md import (  # noqa: E402
    AU_PRESSURE_TO_BAR,
    EnergyLogger,
    MCBarostatLogger,
    MDEngine,
    MonteCarloSemiIsotropicBarostat,
    OpenMMAdapter,
    XYZTrajectoryWriter,
    atoms_from_charmm,
    backend_status,
    detect_leaflets,
    membrane_diagnostics,
    membrane_summary,
    openmm_available,
    read_charmm_parameters,
    scale_molecule_centers,
    set_maxwell_boltzmann_velocities,
    semi_isotropic_pressure,
    soft_relaxation,
    write_minimization_log,
)
from pyqed.units import au2angstrom, au2fs, fs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default="charmm_gui_membrane")
    parser.add_argument("--psf", default=None)
    parser.add_argument("--pdb", default=None)
    parser.add_argument("--toppar", default=None)
    parser.add_argument("--electrostatics", choices=("cutoff", "pme"), default="pme")
    parser.add_argument("--cutoff-angstrom", type=float, default=12.0)
    parser.add_argument("--switch-angstrom", type=float, default=10.0)
    parser.add_argument("--head-names", default="P,N", help="Comma-separated atom-name prefixes for leaflet references.")
    parser.add_argument("--openmm-check", action="store_true")
    parser.add_argument("--openmm-method", choices=("cutoff", "pme"), default="cutoff")
    parser.add_argument("--component-tolerance-hartree", type=float, default=None)
    parser.add_argument("--fail-on-unsupported", action="store_true")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--fail-on-nonfinite", action="store_true")
    parser.add_argument("--max-force-hartree-per-bohr", type=float, default=None)
    parser.add_argument("--max-abs-pressure-bar", type=float, default=None)
    parser.add_argument("--max-abs-energy-drift-hartree", type=float, default=None)
    parser.add_argument("--max-abs-energy-drift-per-ps-hartree", type=float, default=None)
    parser.add_argument("--box-scale-lateral", type=float, default=1.0)
    parser.add_argument("--box-scale-normal", type=float, default=1.0)
    parser.add_argument("--relax-steps-per-stage", type=int, default=0)
    parser.add_argument("--relax-max-step-angstrom", type=float, default=0.02)
    parser.add_argument("--relax-fmax", type=float, default=5e-4)
    parser.add_argument("--relax-log", default=None)
    parser.add_argument("--md-steps", type=int, default=0)
    parser.add_argument("--md-timestep-fs", type=float, default=0.001)
    parser.add_argument("--md-temperature", type=float, default=10.0)
    parser.add_argument("--md-friction", type=float, default=0.2)
    parser.add_argument("--md-seed", type=int, default=17)
    parser.add_argument("--md-energy-interval", type=int, default=1)
    parser.add_argument("--md-trajectory-interval", type=int, default=1)
    parser.add_argument("--md-energy-log", default=None)
    parser.add_argument("--md-trajectory", default=None)
    parser.add_argument("--mc-barostat", action="store_true", help="Attach native semi-isotropic MC barostat during --md-steps.")
    parser.add_argument("--mc-interval", type=int, default=None, help="MC attempt interval; defaults to --md-energy-interval.")
    parser.add_argument("--mc-max-area-change", type=float, default=0.003)
    parser.add_argument("--mc-max-z-change", type=float, default=0.003)
    parser.add_argument("--mc-log", default=None)
    parser.add_argument("--render-dir", default=None, help="Write PBC-unwrapped membrane PNG diagnostics into this directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    folder = Path(args.folder)
    inputs = _find_inputs(folder, args)
    if inputs is None:
        print(f"skip: CHARMM-GUI inputs not found in {folder}")
        return 0

    cutoff = args.cutoff_angstrom / au2angstrom
    switch_on = args.switch_angstrom / au2angstrom
    atoms = atoms_from_charmm(
        inputs["psf"],
        inputs["parameters"],
        pdb_file=inputs["pdb"],
        coulomb_method=args.electrostatics,
        coulomb_cutoff=cutoff,
        coulomb_energy_shift=args.openmm_check and args.openmm_method == "cutoff",
        lj_cutoff=cutoff,
        lj_switch_on=switch_on,
        pme_mesh=(32, 32, 32),
    )
    applied_box_scale = scale_molecule_centers(
        atoms,
        lateral_scale=args.box_scale_lateral,
        normal_scale=args.box_scale_normal,
    )
    relaxation = []
    if args.relax_steps_per_stage > 0:
        stage_steps = int(args.relax_steps_per_stage)
        relaxation = soft_relaxation(
            atoms,
            stages=((0.1, 0.05, stage_steps), (0.4, 0.3, stage_steps), (1.0, 1.0, stage_steps)),
            max_step=args.relax_max_step_angstrom / au2angstrom,
            fmax=args.relax_fmax,
        )
        if args.relax_log:
            write_minimization_log(args.relax_log, relaxation)
    head_indices = _head_indices(atoms, args.head_names)
    lipids_per_leaflet = _lipids_per_leaflet(atoms, head_indices)
    md_metrics = _run_native_md(atoms, args, lipids_per_leaflet=lipids_per_leaflet)
    render_metrics = _render_outputs(atoms, args.render_dir, head_indices=head_indices)
    parameters = read_charmm_parameters(inputs["parameters"])
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    pyqed_components = _energy_components(atoms)
    charge = float(np.sum(atoms.get_array("charges")))
    print(f"psf: {inputs['psf']}")
    print(f"pdb: {inputs['pdb']}")
    print(f"parameter_files: {len(inputs['parameters'])}")
    print(f"atoms: {len(atoms)}")
    print(f"molecules: {len(np.unique(atoms.get_array('molecule_ids')))}")
    print(f"total_charge: {charge:.8f}")
    print(f"box_scale_lateral: {applied_box_scale[0]:.8f}")
    print(f"box_scale_normal: {applied_box_scale[2]:.8f}")
    _print_relaxation_summary(relaxation)
    _print_md_summary(md_metrics)
    _print_render_summary(render_metrics)
    print(f"potential_hartree: {energy:.12e}")
    print(f"finite_forces: {bool(np.all(np.isfinite(forces)))}")
    lateral_pressure, normal_pressure, pressure_tensor = semi_isotropic_pressure(atoms)
    pressure_diagonal = np.diag(pressure_tensor) * AU_PRESSURE_TO_BAR
    print(f"pressure_lateral_bar: {lateral_pressure * AU_PRESSURE_TO_BAR:.12e}")
    print(f"pressure_normal_bar: {normal_pressure * AU_PRESSURE_TO_BAR:.12e}")
    print(f"pressure_xx_bar: {pressure_diagonal[0]:.12e}")
    print(f"pressure_yy_bar: {pressure_diagonal[1]:.12e}")
    print(f"pressure_zz_bar: {pressure_diagonal[2]:.12e}")
    for name in _component_names():
        if name in pyqed_components:
            print(f"pyqed_component_{name}_hartree: {pyqed_components[name]:.12e}")
    if _has_pyqed_cmaps(atoms):
        print("pyqed_cmap_interpolation: openmm-periodic-bicubic")
    if parameters.unsupported_sections:
        print(f"unsupported_parameter_sections: {','.join(sorted(set(parameters.unsupported_sections)))}")
        if args.fail_on_unsupported:
            print("status: failed-unsupported-parameter-sections")
            return 2
    membrane_metrics = {}
    if len(head_indices):
        labels = detect_leaflets(atoms, head_indices=head_indices)
        membrane_metrics = membrane_summary(atoms, head_indices=head_indices)
        print(f"head_atoms: {len(head_indices)}")
        print(f"upper_atoms: {int(np.count_nonzero(labels > 0))}")
        print(f"lower_atoms: {int(np.count_nonzero(labels < 0))}")
        print(f"area_per_lipid_angstrom2: {membrane_metrics['area_per_lipid_angstrom2']:.6f}")
        print(f"bilayer_thickness_angstrom: {membrane_metrics['bilayer_thickness_angstrom']:.6f}")
    else:
        print("membrane_summary: skipped-no-head-atoms")
    diagnostics = membrane_diagnostics(atoms, head_indices=head_indices if len(head_indices) else None)
    for key in ("max_force_hartree_per_bohr", "molecules"):
        if key in diagnostics:
            print(f"{key}: {diagnostics[key]}")
    status = backend_status("openmm")
    print(f"openmm_available: {status['available']}")
    print(f"openmm_reason: {status['reason']}")
    openmm_metrics = {}
    deltas = {}
    if args.openmm_check and openmm_available():
        adapter = OpenMMAdapter(
            atoms=atoms,
            nonbonded_cutoff=cutoff,
            switch_on=switch_on,
            nonbonded_method=args.openmm_method,
        )
        system = adapter.to_openmm_system()
        openmm_energy = adapter.potential_energy()
        openmm_components = adapter.energy_components()
        print(f"openmm_particles: {system.getNumParticles()}")
        print(f"openmm_forces: {system.getNumForces()}")
        print(f"openmm_potential_hartree: {openmm_energy:.12e}")
        print(f"openmm_minus_pyqed_hartree: {openmm_energy - energy:.12e}")
        openmm_metrics = {
            "openmm_particles": int(system.getNumParticles()),
            "openmm_forces": int(system.getNumForces()),
            "openmm_potential_hartree": float(openmm_energy),
            "openmm_minus_pyqed_hartree": float(openmm_energy - energy),
        }
        for name in _component_names():
            if name in openmm_components:
                print(f"openmm_component_{name}_hartree: {openmm_components[name]:.12e}")
                openmm_metrics[f"openmm_component_{name}_hartree"] = float(openmm_components[name])
            if name in openmm_components and name in pyqed_components:
                delta = openmm_components[name] - pyqed_components[name]
                deltas[name] = delta
                print(f"component_delta_{name}_hartree: {delta:.12e}")
                openmm_metrics[f"component_delta_{name}_hartree"] = float(delta)
        if args.component_tolerance_hartree is not None:
            offenders = {
                name: delta for name, delta in deltas.items()
                if abs(delta) > args.component_tolerance_hartree
            }
            if offenders:
                details = ",".join(f"{name}={delta:.6e}" for name, delta in sorted(offenders.items()))
                print(f"status: failed-component-tolerance {details}")
                return 3
    summary_metrics = _summary_metrics(
        inputs=inputs,
        parameters=parameters,
        atoms=atoms,
        charge=charge,
        applied_box_scale=applied_box_scale,
        relaxation=relaxation,
        md_metrics=md_metrics,
        potential=energy,
        forces=forces,
        pressure=(lateral_pressure, normal_pressure, pressure_diagonal),
        pyqed_components=pyqed_components,
        membrane_metrics=membrane_metrics,
        diagnostics=diagnostics,
        backend_status=status,
        openmm_metrics=openmm_metrics,
    )
    summary_metrics.update(render_metrics)
    if args.summary_json:
        _write_summary_json(args.summary_json, summary_metrics)
        print(f"summary_json: {args.summary_json}")
    gate_failures = _gate_failures(summary_metrics, args)
    if gate_failures:
        print(f"status: failed-gates {','.join(gate_failures)}")
        return 4
    return 0


def _find_inputs(folder, args):
    psf = Path(args.psf) if args.psf else _first_existing(folder, ["step3_input.psf", "step5_input.psf", "input.psf"])
    pdb = Path(args.pdb) if args.pdb else _first_existing(folder, ["step3_input.pdb", "step5_input.pdb", "input.pdb"])
    toppar = Path(args.toppar) if args.toppar else folder / "toppar"
    if psf is None or pdb is None or not toppar.exists():
        return None
    parameters = sorted(toppar.glob("*.prm")) + sorted(toppar.glob("*.str"))
    if not parameters:
        return None
    return {"psf": psf, "pdb": pdb, "parameters": parameters}


def _summary_metrics(
    inputs,
    parameters,
    atoms,
    charge,
    applied_box_scale,
    relaxation,
    md_metrics,
    potential,
    forces,
    pressure,
    pyqed_components,
    membrane_metrics,
    diagnostics,
    backend_status,
    openmm_metrics,
):
    lateral_pressure, normal_pressure, pressure_diagonal = pressure
    metrics = {
        "psf": str(inputs["psf"]),
        "pdb": str(inputs["pdb"]),
        "parameter_files": [str(path) for path in inputs["parameters"]],
        "atoms": int(len(atoms)),
        "molecules": int(len(np.unique(atoms.get_array("molecule_ids")))),
        "total_charge": float(charge),
        "box_scale_lateral": float(applied_box_scale[0]),
        "box_scale_normal": float(applied_box_scale[2]),
        "potential_hartree": float(potential),
        "finite_positions": bool(np.all(np.isfinite(atoms.get_positions()))),
        "finite_forces": bool(np.all(np.isfinite(forces))),
        "pressure_lateral_bar": float(lateral_pressure * AU_PRESSURE_TO_BAR),
        "pressure_normal_bar": float(normal_pressure * AU_PRESSURE_TO_BAR),
        "pressure_xx_bar": float(pressure_diagonal[0]),
        "pressure_yy_bar": float(pressure_diagonal[1]),
        "pressure_zz_bar": float(pressure_diagonal[2]),
        "unsupported_parameter_sections": sorted(set(parameters.unsupported_sections)),
        "openmm_available": bool(backend_status["available"]),
        "openmm_reason": str(backend_status["reason"]),
    }
    metrics.update(_relaxation_metrics(relaxation))
    metrics.update({key: _json_safe(value) for key, value in md_metrics.items()})
    metrics.update(
        {
            f"pyqed_component_{name}_hartree": float(value)
            for name, value in pyqed_components.items()
        }
    )
    metrics.update({key: _json_safe(value) for key, value in membrane_metrics.items()})
    metrics.update({key: _json_safe(value) for key, value in diagnostics.items()})
    metrics.update({key: _json_safe(value) for key, value in openmm_metrics.items()})
    return metrics


def _relaxation_metrics(relaxation):
    if not relaxation:
        return {
            "relaxation_enabled": False,
            "relaxation_total_steps": 0,
        }
    final = relaxation[-1]
    return {
        "relaxation_enabled": True,
        "relaxation_total_steps": int(sum(int(stage.get("steps", 0)) for stage in relaxation)),
        "relaxation_final_fmax_hartree_per_bohr": float(final.get("fmax", np.nan)),
        "relaxation_final_energy_hartree": float(final.get("energy", np.nan)),
        "relaxation_converged": bool(final.get("converged", False)),
    }


def _write_summary_json(path, metrics):
    with open(path, "w") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _gate_failures(metrics, args):
    failures = []
    if args.fail_on_nonfinite:
        for key in ("finite_positions", "finite_forces", "md_finite_positions", "md_finite_forces", "md_energy_log_finite"):
            if key in metrics and not bool(metrics[key]):
                failures.append(key)
    if args.max_force_hartree_per_bohr is not None:
        force = metrics.get("max_force_hartree_per_bohr")
        if force is not None and abs(float(force)) > args.max_force_hartree_per_bohr:
            failures.append("max_force_hartree_per_bohr")
    if args.max_abs_pressure_bar is not None:
        pressure_keys = (
            "pressure_lateral_bar",
            "pressure_normal_bar",
            "pressure_xx_bar",
            "pressure_yy_bar",
            "pressure_zz_bar",
        )
        if any(abs(float(metrics[key])) > args.max_abs_pressure_bar for key in pressure_keys if key in metrics):
            failures.append("pressure_bar")
    if args.max_abs_energy_drift_hartree is not None:
        drift = metrics.get("md_total_energy_drift_hartree")
        if drift is not None and abs(float(drift)) > args.max_abs_energy_drift_hartree:
            failures.append("md_total_energy_drift_hartree")
    if args.max_abs_energy_drift_per_ps_hartree is not None:
        drift = metrics.get("md_total_energy_drift_per_ps_hartree")
        if drift is not None and abs(float(drift)) > args.max_abs_energy_drift_per_ps_hartree:
            failures.append("md_total_energy_drift_per_ps_hartree")
    return failures


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _print_relaxation_summary(relaxation):
    if not relaxation:
        print("relaxation_enabled: False")
        print("relaxation_total_steps: 0")
        return
    final = relaxation[-1]
    total_steps = sum(int(stage.get("steps", 0)) for stage in relaxation)
    print("relaxation_enabled: True")
    print(f"relaxation_total_steps: {total_steps}")
    print(f"relaxation_final_fmax_hartree_per_bohr: {final.get('fmax', float('nan')):.12e}")
    print(f"relaxation_final_energy_hartree: {final.get('energy', float('nan')):.12e}")
    print(f"relaxation_converged: {bool(final.get('converged', False))}")


def _run_native_md(atoms, args, lipids_per_leaflet=None):
    steps = int(args.md_steps)
    if steps <= 0:
        return {
            "md_enabled": False,
            "md_steps": 0,
            "md_time_fs": 0.0,
            "md_finite_positions": bool(np.all(np.isfinite(atoms.get_positions()))),
            "md_finite_forces": bool(np.all(np.isfinite(atoms.get_forces()))),
            "md_energy_log_rows": 0,
        }
    if args.md_energy_interval <= 0:
        raise ValueError("--md-energy-interval must be positive.")
    if args.md_trajectory_interval <= 0:
        raise ValueError("--md-trajectory-interval must be positive.")

    set_maxwell_boltzmann_velocities(atoms, args.md_temperature, seed=args.md_seed)
    engine = MDEngine(
        atoms,
        timestep=args.md_timestep_fs * fs,
        ensemble="langevin",
        temperature_K=args.md_temperature,
        friction=args.md_friction,
    )
    dynamics = engine.dynamics
    writer = None
    logger = None
    mc_logger = None
    mc_barostat = None
    mc_log = args.mc_log
    if args.mc_barostat:
        interval = args.mc_interval or args.md_energy_interval
        if interval <= 0:
            raise ValueError("--mc-interval must be positive.")
        mc_barostat = MonteCarloSemiIsotropicBarostat.from_bar(
            atoms,
            temperature_K=args.md_temperature,
            target_lateral_pressure_bar=1.0,
            target_normal_pressure_bar=1.0,
            max_area_change=args.mc_max_area_change,
            max_z_change=args.mc_max_z_change,
            scale_molecule_centers=True,
            molecule_array="molecule_ids",
            seed=args.md_seed + 1009,
        )
        dynamics.attach(mc_barostat, interval=interval)
        if mc_log:
            mc_logger = MCBarostatLogger(
                mc_barostat,
                mc_log,
                dynamics=dynamics,
                lipids_per_leaflet=lipids_per_leaflet,
            )
            dynamics.attach(mc_logger, interval=interval)
    if args.md_trajectory:
        writer = XYZTrajectoryWriter(atoms, args.md_trajectory, dynamics=dynamics)
        dynamics.attach(writer, interval=args.md_trajectory_interval)
    if args.md_energy_log:
        logger = EnergyLogger(atoms, args.md_energy_log, dynamics=dynamics)
        dynamics.attach(logger, interval=args.md_energy_interval)
    try:
        engine.run(steps)
    finally:
        if writer is not None:
            writer.close()
        if logger is not None:
            logger.close()
        if mc_logger is not None:
            mc_logger.close()

    metrics = {
        "md_enabled": True,
        "md_steps": dynamics.get_number_of_steps(),
        "md_time_fs": float(dynamics.get_time() * au2fs),
        "md_finite_positions": bool(np.all(np.isfinite(atoms.get_positions()))),
        "md_finite_forces": bool(np.all(np.isfinite(atoms.get_forces()))),
        **_energy_log_metrics(args.md_energy_log),
        "md_energy_log": args.md_energy_log,
        "md_trajectory": args.md_trajectory,
    }
    if mc_barostat is not None:
        metrics.update(
            {
                "mc_barostat_enabled": True,
                "mc_barostat_attempts": int(mc_barostat.attempts),
                "mc_barostat_accepted": int(mc_barostat.accepted),
                "mc_barostat_acceptance_rate": float(mc_barostat.acceptance_rate),
                "mc_barostat_log": mc_log,
            }
        )
        metrics.update(_mc_log_metrics(mc_log))
    else:
        metrics["mc_barostat_enabled"] = False
    return metrics


def _print_md_summary(metrics):
    print(f"md_enabled: {metrics['md_enabled']}")
    print(f"md_steps: {metrics['md_steps']}")
    print(f"md_time_fs: {metrics['md_time_fs']:.8f}")
    print(f"md_finite_positions: {metrics['md_finite_positions']}")
    print(f"md_finite_forces: {metrics['md_finite_forces']}")
    print(f"md_energy_log_rows: {metrics['md_energy_log_rows']}")
    print(f"md_energy_log_finite: {metrics.get('md_energy_log_finite', True)}")
    for key in (
        "md_total_energy_drift_hartree",
        "md_energy_log_time_span_ps",
        "md_total_energy_drift_per_ps_hartree",
        "md_temperature_K_min",
        "md_temperature_K_max",
        "md_pressure_lateral_bar_min",
        "md_pressure_lateral_bar_max",
        "md_pressure_normal_bar_min",
        "md_pressure_normal_bar_max",
    ):
        if key in metrics:
            print(f"{key}: {metrics[key]:.12e}")
    if metrics.get("md_energy_log"):
        print(f"md_energy_log: {metrics['md_energy_log']}")
    if metrics.get("md_trajectory"):
        print(f"md_trajectory: {metrics['md_trajectory']}")
    print(f"mc_barostat_enabled: {metrics.get('mc_barostat_enabled', False)}")
    if metrics.get("mc_barostat_enabled"):
        print(f"mc_barostat_attempts: {metrics.get('mc_barostat_attempts', 0)}")
        print(f"mc_barostat_accepted: {metrics.get('mc_barostat_accepted', 0)}")
        print(f"mc_barostat_acceptance_rate: {metrics.get('mc_barostat_acceptance_rate', 0.0):.8f}")
        if metrics.get("mc_barostat_log"):
            print(f"mc_barostat_log: {metrics['mc_barostat_log']}")
        if "mc_log_rows" in metrics:
            print(f"mc_log_rows: {metrics['mc_log_rows']}")


def _mc_log_metrics(path):
    if not path:
        return {"mc_log_rows": 0, "mc_log_finite": True}
    path = Path(path)
    if not path.exists():
        return {"mc_log_rows": 0, "mc_log_finite": False}
    lines = path.read_text().splitlines()
    if len(lines) <= 1:
        return {"mc_log_rows": 0, "mc_log_finite": True}
    header = lines[0].split()
    numeric_indices = [index for index, name in enumerate(header) if name != "move"]
    data = np.loadtxt(lines[1:], ndmin=2, usecols=numeric_indices)
    return {
        "mc_log_rows": int(data.shape[0]),
        "mc_log_finite": bool(np.all(np.isfinite(data))),
    }


def _lipids_per_leaflet(atoms, head_indices):
    if len(head_indices) == 0 or not atoms.has("molecule_ids"):
        return None
    labels = detect_leaflets(atoms, head_indices=head_indices)
    molecule_ids = atoms.get_array("molecule_ids")
    counts = []
    for sign in (-1, 1):
        molecules = np.unique(molecule_ids[head_indices[labels[head_indices] == sign]])
        if len(molecules):
            counts.append(len(molecules))
    if not counts:
        return None
    return int(round(float(np.mean(counts))))


def _data_rows(path):
    if not path:
        return 0
    lines = Path(path).read_text().splitlines()
    return max(len([line for line in lines[1:] if line.strip()]), 0)


def _energy_log_metrics(path):
    if not path:
        return {
            "md_energy_log_rows": 0,
            "md_energy_log_finite": True,
        }
    path = Path(path)
    if not path.exists():
        return {
            "md_energy_log_rows": 0,
            "md_energy_log_finite": False,
        }
    lines = path.read_text().splitlines()
    if not lines:
        return {
            "md_energy_log_rows": 0,
            "md_energy_log_finite": False,
        }
    header = lines[0].split()
    data_lines = [line for line in lines[1:] if line.strip()]
    if not data_lines:
        return {
            "md_energy_log_rows": 0,
            "md_energy_log_finite": True,
        }
    data = np.loadtxt(data_lines, ndmin=2)
    columns = {name: data[:, index] for index, name in enumerate(header)}
    metrics = {
        "md_energy_log_rows": int(data.shape[0]),
        "md_energy_log_finite": bool(np.all(np.isfinite(data))),
    }

    def add_range(source, prefix):
        if source not in columns:
            return
        values = columns[source]
        metrics[f"{prefix}_min"] = float(np.min(values))
        metrics[f"{prefix}_max"] = float(np.max(values))

    add_range("temperature_K", "md_temperature_K")
    add_range("pressure_lateral_bar", "md_pressure_lateral_bar")
    add_range("pressure_normal_bar", "md_pressure_normal_bar")
    add_range("pressure_xx_bar", "md_pressure_xx_bar")
    add_range("pressure_yy_bar", "md_pressure_yy_bar")
    add_range("pressure_zz_bar", "md_pressure_zz_bar")
    if "total" in columns:
        total = columns["total"]
        drift = float(total[-1] - total[0])
        metrics["md_total_energy_drift_hartree"] = drift
        if "time" in columns:
            time_fs = columns["time"] * au2fs
            span_ps = float((time_fs[-1] - time_fs[0]) / 1000.0)
            metrics["md_energy_log_time_span_ps"] = span_ps
            if span_ps > 0.0:
                metrics["md_total_energy_drift_per_ps_hartree"] = float(drift / span_ps)
    return metrics


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _render_outputs(atoms, render_dir, head_indices):
    if not render_dir:
        return {"render_enabled": False}
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    image = render_dir / "charmm_membrane_final.png"
    cross_section = render_dir / "charmm_membrane_cross_section.png"
    density_data = render_dir / "charmm_membrane_density_z.dat"
    density_plot = render_dir / "charmm_membrane_density_z.png"

    positions = _display_positions(atoms)
    masks = _render_masks(atoms, head_indices)
    bonds = _render_bonds(atoms)
    _render_membrane_3d(atoms, positions, masks, bonds, image)
    _render_cross_section(atoms, positions, masks, bonds, cross_section)
    _write_density_profile(atoms, positions, masks, density_data, density_plot)
    return {
        "render_enabled": True,
        "render_image": str(image),
        "render_cross_section": str(cross_section),
        "render_density_profile": str(density_data),
        "render_density_plot": str(density_plot),
    }


def _print_render_summary(metrics):
    print(f"render_enabled: {metrics.get('render_enabled', False)}")
    for key in ("render_image", "render_cross_section", "render_density_profile", "render_density_plot"):
        if key in metrics:
            print(f"{key}: {metrics[key]}")


def _display_positions(atoms):
    positions = np.asarray(atoms.get_positions(), dtype=float)
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    molecule_ids = atoms.get_array("molecule_ids") if atoms.has("molecule_ids") else np.arange(len(atoms))
    display = positions.copy()
    for molecule_id in np.unique(molecule_ids):
        indices = np.flatnonzero(molecule_ids == molecule_id)
        if len(indices) <= 1:
            continue
        reference = positions[indices[0]]
        deltas = positions[indices] - reference
        for axis in range(3):
            if pbc[axis] and lengths[axis] > 0.0:
                deltas[:, axis] -= np.round(deltas[:, axis] / lengths[axis]) * lengths[axis]
        display[indices] = reference + deltas

    center = np.zeros(3)
    center[:2] = 0.5 * lengths[:2]
    center[2] = float(np.mean(display[:, 2]))
    display -= center
    for molecule_id in np.unique(molecule_ids):
        indices = np.flatnonzero(molecule_ids == molecule_id)
        molecule_center = np.mean(display[indices], axis=0)
        shift = np.zeros(3)
        for axis in (0, 1):
            if pbc[axis] and lengths[axis] > 0.0:
                shift[axis] = -np.round(molecule_center[axis] / lengths[axis]) * lengths[axis]
        display[indices] += shift
    return display * au2angstrom


def _render_masks(atoms, head_indices):
    natoms = len(atoms)
    head = np.zeros(natoms, dtype=bool)
    head[np.asarray(head_indices, dtype=int)] = True
    residue_names = _upper_array(atoms.get_array("residue_names")) if atoms.has("residue_names") else np.full(natoms, "")
    atom_names = _upper_array(atoms.get_array("atom_names")) if atoms.has("atom_names") else np.full(natoms, "")
    symbols = _upper_array(atoms.atom_symbols())
    water = np.isin(residue_names, ["TIP3", "TP3", "WAT", "HOH", "SOL", "H2O"]) | np.isin(atom_names, ["OH2", "OW"])
    ions = np.isin(residue_names, ["SOD", "CLA", "NA", "CL", "K", "POT"]) | np.isin(symbols, ["NA", "CL", "K"])
    lipid = ~(water | ions)
    tail = lipid & ~head
    return {
        "head": head,
        "tail": tail,
        "water": water,
        "ion": ions,
        "lipid": lipid,
    }


def _upper_array(values):
    return np.asarray([str(value).upper() for value in values])


def _render_bonds(atoms):
    topology = getattr(atoms, "topology", None)
    if topology is None:
        return []
    return sorted({tuple(sorted((int(i), int(j)))) for i, j, *_ in getattr(topology, "bonds", ())})


def _render_membrane_3d(atoms, positions, masks, bonds, path):
    plt = _pyplot()
    fig = plt.figure(figsize=(8, 6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    _add_3d_bonds(ax, positions, bonds, mask=masks["lipid"] | masks["water"])
    layers = [
        ("head atoms", masks["head"], "#d84a3a", 30),
        ("lipid atoms", masks["tail"], "#2f684e", 12),
        ("water", masks["water"], "#4b8fd8", 12),
        ("ions", masks["ion"], "#7d4ab8", 34),
    ]
    for label, mask, color, size in layers:
        if np.any(mask):
            ax.scatter(
                positions[mask, 0],
                positions[mask, 1],
                positions[mask, 2],
                s=size,
                c=color,
                edgecolors="#222222" if size > 10 else "none",
                linewidths=0.25,
                alpha=0.9,
                depthshade=True,
                label=label,
            )
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    ax.set_xlim(-0.5 * lengths[0], 0.5 * lengths[0])
    ax.set_ylim(-0.5 * lengths[1], 0.5 * lengths[1])
    ax.set_zlim(-0.5 * lengths[2], 0.5 * lengths[2])
    ax.set_xlabel("x / A")
    ax.set_ylabel("y / A")
    ax.set_zlabel("z / A")
    ax.view_init(elev=22, azim=-52)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_cross_section(atoms, positions, masks, bonds, path):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    _add_2d_bonds(ax, positions, bonds, mask=masks["lipid"] | masks["water"])
    layers = [
        ("lipid", masks["tail"], "#2f684e", 12, 0.78),
        ("head", masks["head"], "#d84a3a", 20, 0.9),
        ("water", masks["water"], "#4b8fd8", 12, 0.72),
        ("ions", masks["ion"], "#7d4ab8", 32, 0.95),
    ]
    for label, mask, color, size, alpha in layers:
        if np.any(mask):
            ax.scatter(positions[mask, 0], positions[mask, 2], s=size, c=color, alpha=alpha, label=label, zorder=2)
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-0.5 * lengths[0], 0.5 * lengths[0])
    ax.set_ylim(-0.5 * lengths[2], 0.5 * lengths[2])
    ax.set_xlabel("x / A")
    ax.set_ylabel("z / A")
    ax.legend(frameon=False, ncol=4, loc="upper center")
    ax.set_title("CHARMM membrane cross-section")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_density_profile(atoms, positions, masks, data_path, plot_path, bins=80):
    plt = _pyplot()
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    edges = np.linspace(-0.5 * lengths[2], 0.5 * lengths[2], int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = {
        name: np.histogram(positions[mask, 2], bins=edges)[0]
        for name, mask in masks.items()
        if name in {"head", "tail", "water", "ion"}
    }
    with open(data_path, "w") as handle:
        handle.write("z_angstrom head tail water ion\n")
        for index, z_value in enumerate(centers):
            handle.write(
                f"{z_value:.6f} {counts['head'][index]} {counts['tail'][index]} "
                f"{counts['water'][index]} {counts['ion'][index]}\n"
            )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax.plot(centers, counts["head"], color="#d84a3a", label="head")
    ax.plot(centers, counts["tail"], color="#2f684e", label="lipid")
    ax.plot(centers, counts["water"], color="#4b8fd8", label="water")
    ax.plot(centers, counts["ion"], color="#7d4ab8", label="ions")
    ax.set_xlabel("z / A")
    ax.set_ylabel("atom count per bin")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def _add_3d_bonds(ax, positions, bonds, mask=None):
    if not bonds:
        return
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segments = []
    for i, j in bonds:
        if mask is not None and not (mask[i] and mask[j]):
            continue
        segments.append([positions[i], positions[j]])
    if segments:
        ax.add_collection3d(Line3DCollection(segments, colors="#9b9b9b", linewidths=0.45, alpha=0.45))


def _add_2d_bonds(ax, positions, bonds, mask=None):
    for i, j in bonds:
        if mask is not None and not (mask[i] and mask[j]):
            continue
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 2], positions[j, 2]],
            color="#9b9b9b",
            linewidth=0.45,
            alpha=0.38,
            zorder=1,
        )


def _first_existing(folder, names):
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return None


def _head_indices(atoms, spec):
    prefixes = tuple(item.strip().upper() for item in spec.split(",") if item.strip())
    names = atoms.get_array("atom_names") if atoms.has("atom_names") else []
    indices = []
    for index, name in enumerate(names):
        if str(name).upper().startswith(prefixes):
            indices.append(index)
    return np.asarray(indices, dtype=int)


def _component_names():
    return ("bonds", "angles", "torsions", "impropers", "cmaps", "nonbonded", "total")


def _energy_components(atoms):
    calc = getattr(atoms, "calc", None)
    if calc is not None and hasattr(calc, "energy_components"):
        return calc.energy_components(atoms)
    return {"total": atoms.get_potential_energy()}


def _has_pyqed_cmaps(atoms):
    topology = getattr(atoms, "topology", None)
    return bool(topology is not None and getattr(topology, "cmaps", None))


if __name__ == "__main__":
    raise SystemExit(main())
