#!/usr/bin/env python3
"""Build and smoke-run a native PyQED lipid-template membrane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.md import (  # noqa: E402
    EnergyLogger,
    Langevin,
    XYZTrajectoryWriter,
    available_lipid_templates,
    hydrated_lipid_bilayer_from_template,
    lipid_bilayer_from_template,
    lipid_template,
    membrane_analysis,
    membrane_diagnostics,
    scale_molecule_centers,
    set_maxwell_boltzmann_velocities,
    soft_relaxation,
    write_minimization_log,
    write_pdb,
)
from pyqed.units import au2angstrom, au2fs, fs  # noqa: E402


PRESETS = {
    "build": {
        "nx": 1,
        "ny": 1,
        "waters_per_side": 0,
        "steps": 0,
        "relax_steps_per_stage": 0,
    },
    "smoke": {
        "nx": 1,
        "ny": 1,
        "waters_per_side": 2,
        "steps": 10,
        "timestep_fs": 0.0005,
        "temperature": 10.0,
        "relax_steps_per_stage": 2,
    },
    "gentle-smoke": {
        "nx": 1,
        "ny": 1,
        "waters_per_side": 2,
        "steps": 10,
        "timestep_fs": 0.0001,
        "temperature": 5.0,
        "friction": 5.0,
        "relax_steps_per_stage": 4,
        "relax_max_step_angstrom": 0.01,
        "equilibration_ramp": "1:5,2:5,5:5",
        "temperature_rescale_interval": 1,
    },
    "stability-0.01ps": {
        "nx": 1,
        "ny": 1,
        "waters_per_side": 4,
        "steps": 200,
        "timestep_fs": 0.05,
        "temperature": 30.0,
        "relax_steps_per_stage": 20,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default=None)
    parser.add_argument("--lipid", default="DPPC")
    parser.add_argument("--list-lipids", action="store_true", help="List native lipid templates and exit.")
    parser.add_argument("--openmm-lipid-source", default=None, help="Optional OpenMM lipid force-field XML path.")
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--area-per-lipid", type=float, default=90.0)
    parser.add_argument("--thickness", type=float, default=38.0)
    parser.add_argument("--waters-per-side", type=int, default=0)
    parser.add_argument("--water-spacing", type=float, default=3.2)
    parser.add_argument("--coulomb-method", choices=("cutoff", "pme"), default="cutoff")
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0)
    parser.add_argument("--pme-mesh", type=int, default=12)
    parser.add_argument("--box-scale-lateral", type=float, default=1.0)
    parser.add_argument("--box-scale-normal", type=float, default=1.0)
    parser.add_argument("--relax-steps-per-stage", type=int, default=0)
    parser.add_argument("--relax-max-step-angstrom", type=float, default=0.02)
    parser.add_argument("--relax-fmax", type=float, default=5e-4)
    parser.add_argument(
        "--equilibration-ramp",
        default="",
        help="Comma-separated target_K:steps stages before production, e.g. 1:20,5:20,10:20.",
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--timestep-fs", type=float, default=0.0005)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument(
        "--temperature-rescale-interval",
        type=int,
        default=0,
        help="If positive, rescale velocities to the target temperature every N steps; intended for setup smoke tests.",
    )
    parser.add_argument("--friction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--energy-interval", type=int, default=1)
    parser.add_argument("--trajectory-interval", type=int, default=1)
    parser.add_argument("--output-dir", default="/private/tmp/pyqed_native_lipid_template_membrane")
    args = parser.parse_args(argv)
    return apply_preset(args, sys.argv[1:] if argv is None else argv)


def apply_preset(args, argv):
    if args.preset is None:
        return args
    explicit = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in argv
        if token.startswith("--")
    }
    for key, value in PRESETS[args.preset].items():
        if key not in explicit:
            setattr(args, key, value)
    return args


def main():
    args = parse_args()
    if args.temperature_rescale_interval < 0:
        raise ValueError("temperature rescale interval must be non-negative.")
    if args.list_lipids:
        for name in available_lipid_templates():
            print(name)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        lipid=args.lipid,
        nx=args.nx,
        ny=args.ny,
        area_per_lipid=args.area_per_lipid,
        thickness=args.thickness,
        calculator=True,
        coulomb_method=args.coulomb_method,
        coulomb_cutoff=args.cutoff_angstrom,
        lj_cutoff=args.cutoff_angstrom,
        pme_mesh=(args.pme_mesh, args.pme_mesh, args.pme_mesh),
        seed=args.seed,
        openmm_source=args.openmm_lipid_source,
    )
    if args.waters_per_side > 0:
        atoms = hydrated_lipid_bilayer_from_template(
            waters_per_side=args.waters_per_side,
            water_spacing=args.water_spacing,
            **common,
        )
    else:
        atoms = lipid_bilayer_from_template(**common)

    box_scale = scale_molecule_centers(
        atoms,
        lateral_scale=args.box_scale_lateral,
        normal_scale=args.box_scale_normal,
    )
    relaxation = []
    if args.relax_steps_per_stage > 0:
        nsteps = int(args.relax_steps_per_stage)
        relaxation = soft_relaxation(
            atoms,
            stages=((0.1, 0.05, nsteps), (0.4, 0.3, nsteps), (1.0, 1.0, nsteps)),
            max_step=args.relax_max_step_angstrom / au2angstrom,
            fmax=args.relax_fmax,
        )
        write_minimization_log(output_dir / "native_lipid_template_relax.dat", relaxation)

    energy_path = output_dir / "native_lipid_template_energy.dat"
    equilibration_path = output_dir / "native_lipid_template_equilibration.dat"
    trajectory_path = output_dir / "native_lipid_template.xyz"
    pdb_path = output_dir / "native_lipid_template.pdb"
    analysis_path = output_dir / "analysis.json"
    equilibration_stages = _parse_equilibration_ramp(args.equilibration_ramp)
    equilibration = []
    if equilibration_stages:
        set_maxwell_boltzmann_velocities(atoms, equilibration_stages[0][0], seed=args.seed)
        equilibration = _run_equilibration(
            atoms,
            equilibration_stages,
            timestep=args.timestep_fs * fs,
            friction=args.friction,
            temperature_rescale_interval=args.temperature_rescale_interval,
        )
        _write_equilibration_log(equilibration_path, equilibration)

    if args.steps > 0:
        if not equilibration_stages:
            set_maxwell_boltzmann_velocities(atoms, args.temperature, seed=args.seed)
        else:
            _rescale_temperature(atoms, args.temperature)
        dynamics = Langevin(
            atoms,
            timestep=args.timestep_fs * fs,
            temperature_K=args.temperature,
            friction=args.friction,
        )
        writer = XYZTrajectoryWriter(atoms, trajectory_path, dynamics=dynamics)
        logger = EnergyLogger(atoms, energy_path, dynamics=dynamics)
        production_rescaler = _TemperatureRescaler(atoms, args.temperature)
        if args.temperature_rescale_interval > 0:
            dynamics.insert_observer(
                production_rescaler,
                position=0,
                interval=args.temperature_rescale_interval,
            )
        dynamics.attach(writer, interval=args.trajectory_interval)
        dynamics.attach(logger, interval=args.energy_interval)
        try:
            dynamics.run(args.steps)
        finally:
            writer.close()
            logger.close()
    else:
        dynamics = None

    template = lipid_template(args.lipid, openmm_source=args.openmm_lipid_source)
    write_pdb(atoms, pdb_path)
    head_indices = _head_indices(atoms, template.head_atom_names)
    tail_pairs = _tail_pairs(atoms, template)
    diagnostics = membrane_diagnostics(atoms, head_indices=head_indices)
    analysis = membrane_analysis(atoms, head_indices=head_indices, tail_pairs=tail_pairs)
    analysis_path.write_text(json.dumps(_json_safe(analysis), indent=2, sort_keys=True) + "\n")
    tail_order = analysis.get("tail_order", {})
    topology = atoms.topology
    summary = {
        **diagnostics,
        "preset": args.preset or "custom",
        "lipid_argument": args.lipid,
        "lipid_template": template.name,
        "template_residue_name": template.residue_name,
        "template_description": template.description,
        "template_validated": template.validated,
        "template_forcefield": template.forcefield,
        "template_atoms": int(template.natoms),
        "template_bonds": int(len(template.bonds)),
        "template_angles": int(len(template.angles)),
        "template_torsions": int(len(template.torsions)),
        "template_lj_pair_scales": int(len(template.lj_pair_scales)),
        "template_coulomb_pair_scales": int(len(template.coulomb_pair_scales)),
        "system_bonds": int(len(topology.bonds)),
        "system_angles": int(len(topology.angles)),
        "system_torsions": int(len(topology.torsions)),
        "system_lj_pair_scales": int(len(topology.lj_pair_scales)),
        "system_coulomb_pair_scales": int(len(topology.coulomb_pair_scales)),
        "tail_order_count": int(tail_order.get("count", 0)),
        "tail_order_mean": tail_order.get("mean"),
        "waters_per_side": int(args.waters_per_side),
        "box_scale_lateral": float(box_scale[0]),
        "box_scale_normal": float(box_scale[2]),
        "relaxation_enabled": bool(relaxation),
        "relaxation_total_steps": int(sum(stage.get("steps", 0) for stage in relaxation)),
        "equilibration_enabled": bool(equilibration),
        "equilibration_total_steps": int(sum(stage.get("steps", 0) for stage in equilibration)),
        "temperature_rescale_interval": int(args.temperature_rescale_interval),
        "temperature_rescale_events": int(
            sum(stage.get("temperature_rescale_events", 0) for stage in equilibration)
            + (0 if dynamics is None else production_rescaler.calls)
        ),
        "equilibration_ramp": [
            {"target_temperature_K": float(stage["target_temperature_K"]), "steps": int(stage["steps"])}
            for stage in equilibration
        ],
        "md_steps": 0 if dynamics is None else int(dynamics.get_number_of_steps()),
        "md_time_fs": 0.0 if dynamics is None else float(dynamics.get_time() * au2fs),
        "final_temperature_K": float(atoms.get_temperature()),
        "finite_energy_log": True if args.steps <= 0 else _finite_energy_log(energy_path),
        "energy_log": str(energy_path),
        "equilibration_log": str(equilibration_path),
        "trajectory": str(trajectory_path),
        "pdb": str(pdb_path),
        "analysis": str(analysis_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    print(summary_path)
    for key, value in summary.items():
        print(f"{key}: {value}")


def _head_indices(atoms, names):
    atom_names = atoms.get_array("atom_names")
    molecule_ids = atoms.get_array("molecule_ids")
    lipid_molecules = np.unique(molecule_ids[atom_names != ""])
    result = []
    for molecule_id in lipid_molecules:
        indices = np.nonzero(molecule_ids == molecule_id)[0]
        for index in indices:
            if atom_names[index] in names:
                result.append(int(index))
    return np.asarray(result, dtype=int)


def _tail_pairs(atoms, template):
    if not template.tail_pairs or not atoms.has("molecule_ids"):
        return None
    atom_names = atoms.get_array("atom_names") if atoms.has("atom_names") else np.full(len(atoms), "")
    residue_names = atoms.get_array("residue_names") if atoms.has("residue_names") else np.full(len(atoms), "")
    molecule_ids = atoms.get_array("molecule_ids")
    pairs = []
    for molecule_id in np.unique(molecule_ids[residue_names == template.residue_name]):
        indices = np.nonzero((molecule_ids == molecule_id) & (residue_names == template.residue_name))[0]
        if len(indices) != template.natoms:
            continue
        order = {name: int(index) for name, index in zip(atom_names[indices], indices)}
        for i, j in template.tail_pairs:
            left = template.atom_names[int(i)]
            right = template.atom_names[int(j)]
            if left in order and right in order:
                pairs.append((order[left], order[right]))
    return pairs or None


def _parse_equilibration_ramp(spec):
    spec = str(spec or "").strip()
    if not spec:
        return []
    stages = []
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        try:
            temperature, steps = chunk.split(":", 1)
        except ValueError as exc:
            raise ValueError("equilibration ramp entries must be target_K:steps") from exc
        temperature = float(temperature)
        steps = int(steps)
        if temperature <= 0.0 or steps < 0:
            raise ValueError("equilibration ramp temperatures must be positive and steps non-negative.")
        stages.append((temperature, steps))
    return stages


def _run_equilibration(atoms, stages, timestep, friction, temperature_rescale_interval=0):
    records = []
    elapsed = 0.0
    for stage_index, (target_temperature, steps) in enumerate(stages):
        _rescale_temperature(atoms, target_temperature)
        before = float(atoms.get_temperature())
        dynamics = Langevin(
            atoms,
            timestep=timestep,
            temperature_K=target_temperature,
            friction=friction,
        )
        rescaler = _TemperatureRescaler(atoms, target_temperature)
        if temperature_rescale_interval > 0:
            dynamics.insert_observer(
                rescaler,
                position=0,
                interval=temperature_rescale_interval,
            )
        if steps:
            dynamics.run(steps)
        elapsed += dynamics.get_time()
        records.append(
            {
                "stage": int(stage_index),
                "target_temperature_K": float(target_temperature),
                "steps": int(steps),
                "time_fs": float(elapsed * au2fs),
                "temperature_before_K": before,
                "temperature_after_K": float(atoms.get_temperature()),
                "potential_hartree": float(atoms.get_potential_energy()),
                "kinetic_hartree": float(atoms.get_kinetic_energy()),
                "temperature_rescale_events": int(rescaler.calls),
            }
        )
    return records


def _write_equilibration_log(path, records):
    with open(path, "w") as handle:
        handle.write(
            "stage target_temperature_K steps time_fs temperature_before_K "
            "temperature_after_K potential_hartree kinetic_hartree temperature_rescale_events\n"
        )
        for record in records:
            handle.write(
                f"{record['stage']} {record['target_temperature_K']:.8f} {record['steps']} "
                f"{record['time_fs']:.8f} {record['temperature_before_K']:.8f} "
                f"{record['temperature_after_K']:.8f} {record['potential_hartree']:.12e} "
                f"{record['kinetic_hartree']:.12e} {record['temperature_rescale_events']}\n"
            )


class _TemperatureRescaler:
    def __init__(self, atoms, target_temperature):
        self.atoms = atoms
        self.target_temperature = float(target_temperature)
        self.calls = 0

    def __call__(self):
        _rescale_temperature(self.atoms, self.target_temperature)
        self.calls += 1


def _rescale_temperature(atoms, target_temperature):
    target_temperature = float(target_temperature)
    if target_temperature <= 0.0:
        raise ValueError("target temperature must be positive.")
    current = float(atoms.get_temperature(remove_center_of_mass=True))
    if current <= 0.0 or not np.isfinite(current):
        set_maxwell_boltzmann_velocities(atoms, target_temperature, seed=17)
        return
    momenta = atoms.get_momenta()
    momenta -= momenta.sum(axis=0) / len(atoms)
    atoms.set_momenta(momenta * np.sqrt(target_temperature / current), apply_constraint=False)


def _finite_energy_log(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) <= 1:
        return True
    data = np.loadtxt(lines[1:], ndmin=2)
    return bool(np.all(np.isfinite(data)))


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


if __name__ == "__main__":
    main()
