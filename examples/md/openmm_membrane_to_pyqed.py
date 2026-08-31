#!/usr/bin/env python3
"""Benchmark an OpenMM membrane seed against native PyQED MD.

This is the central production-readiness harness for the membrane route:

1. take an OpenMM membrane PDB, or the installed OpenMM DPPC patch;
2. optionally minimize/run a short OpenMM reference trajectory;
3. import the snapshot into native :mod:`pyqed.md`;
4. compare force-field terms and energies where PyQED has native coverage;
5. write an explicit readiness report with the remaining parity gaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.md import atoms_from_openmm_pdb_system  # noqa: E402
from pyqed.units import au2angstrom, au2kjmol  # noqa: E402


HARTREE_TO_KJMOL = au2kjmol
BOHR_TO_NM = au2angstrom * 0.1
FORCEFIELD_FILES = ("charmm36.xml", "charmm36/water.xml")
SUPPORTED_FORCE_TERMS = {
    "HarmonicBondForce": "full",
    "HarmonicAngleForce": "full",
    "PeriodicTorsionForce": "full",
    "CustomTorsionForce": "harmonic-improper-only",
    "CMAPTorsionForce": "full-when-active",
    "NonbondedForce": "partial",
    "CustomNonbondedForce": "partial-charmm-nbfix",
    "CustomBondForce": "partial-charmm-nbfix-14",
    "CMMotionRemover": "ignored",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", default=None, help="Optional OpenMM-readable membrane PDB. Defaults to OpenMM's DPPC patch.")
    parser.add_argument("--forcefield", nargs="+", default=list(FORCEFIELD_FILES))
    parser.add_argument("--output-dir", default="/private/tmp/pyqed_openmm_membrane_to_pyqed")
    parser.add_argument("--openmm-steps", type=int, default=0)
    parser.add_argument("--minimize-iterations", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=323.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--friction-ps", type=float, default=1.0)
    parser.add_argument("--run-method", choices=("pme", "cutoff"), default="pme")
    parser.add_argument("--compare-method", choices=("cutoff", "pme"), default="cutoff")
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--ewald-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--skip-pyqed-energy", action="store_true")
    parser.add_argument("--skip-force-comparison", action="store_true")
    parser.add_argument("--energy-tolerance-kj-mol", type=float, default=0.1)
    parser.add_argument("--force-rms-tolerance-kj-mol-nm", type=float, default=1.0)
    parser.add_argument("--force-max-tolerance-kj-mol-nm", type=float, default=10.0)
    parser.add_argument("--fail-on-not-ready", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openmm, app, unit = _import_openmm()
    source_pdb = _source_pdb(args, app)
    snapshot_pdb, openmm_run = _write_openmm_snapshot(
        source_pdb=source_pdb,
        output_dir=output_dir,
        args=args,
        openmm=openmm,
        app=app,
        unit=unit,
    )
    pdb = app.PDBFile(str(snapshot_pdb))
    forcefield = app.ForceField(*(str(path) for path in args.forcefield))
    compare_system = _openmm_system(
        pdb,
        forcefield,
        args.compare_method,
        args.cutoff_nm,
        args.ewald_tolerance,
        app,
        unit,
    )
    inventory = _force_inventory(compare_system)
    openmm_components = _openmm_energy_components(
        pdb,
        compare_system,
        openmm,
        unit,
    )
    pyqed_import, pyqed_components, pyqed_forces = _pyqed_import_components_and_forces(
        snapshot_pdb,
        args,
    )
    comparison = _component_comparison(openmm_components, pyqed_components)
    force_comparison = None
    if pyqed_forces is not None and not args.skip_force_comparison:
        openmm_forces = _openmm_forces(pdb, compare_system, openmm, unit)
        force_comparison = _force_comparison(openmm_forces, pyqed_forces)
    readiness = _readiness_report(
        inventory,
        comparison,
        pyqed_components,
        force_comparison,
        args.energy_tolerance_kj_mol,
        args.force_rms_tolerance_kj_mol_nm,
        args.force_max_tolerance_kj_mol_nm,
    )

    summary = {
        "workflow": "openmm_membrane_to_pyqed",
        "source_pdb": str(source_pdb),
        "snapshot_pdb": str(snapshot_pdb),
        "forcefield": list(args.forcefield),
        "openmm_run": openmm_run,
        "openmm_reference": {
            "nonbonded_method": args.compare_method,
            "cutoff_nm": float(args.cutoff_nm),
            "force_inventory": inventory,
            "energy_components_kj_mol": openmm_components,
        },
        "pyqed_import": pyqed_import,
        "pyqed_energy_components_kj_mol": pyqed_components,
        "component_comparison_kj_mol": comparison,
        "force_comparison_kj_mol_nm": force_comparison,
        "readiness": readiness,
    }

    summary_path = output_dir / "openmm_membrane_to_pyqed_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    table_path = output_dir / "energy_component_comparison.dat"
    _write_component_table(table_path, comparison)
    if not args.no_render:
        render_path = output_dir / "openmm_membrane_snapshot_views.png"
        _render_views(pdb.topology, pdb.positions, unit, render_path)
        summary["render"] = str(render_path)
        summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    else:
        render_path = None

    _print_summary(summary_path, table_path, render_path, summary)
    if args.fail_on_not_ready and not readiness["workflow_ready"]:
        return 2
    return 0


def _import_openmm():
    try:
        import openmm
        import openmm.app as app
        from openmm import unit
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenMM is required for this benchmark.") from exc
    return openmm, app, unit


def _source_pdb(args, app):
    if args.pdb is not None:
        return Path(args.pdb)
    return Path(app.__file__).resolve().parent / "data" / "DPPC.pdb"


def _write_openmm_snapshot(source_pdb, output_dir, args, openmm, app, unit):
    pdb = app.PDBFile(str(source_pdb))
    forcefield = app.ForceField(*(str(path) for path in args.forcefield))
    system = _openmm_system(
        pdb,
        forcefield,
        args.run_method,
        args.cutoff_nm,
        args.ewald_tolerance,
        app,
        unit,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    initial_energy = _context_energy_kj_mol(simulation.context, unit)
    if args.minimize_iterations > 0:
        simulation.minimizeEnergy(maxIterations=int(args.minimize_iterations))
    minimized_energy = _context_energy_kj_mol(simulation.context, unit)
    if args.openmm_steps > 0:
        simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin)
        simulation.step(int(args.openmm_steps))
    state = simulation.context.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=True)
    final_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    snapshot_pdb = output_dir / "openmm_membrane_snapshot.pdb"
    with snapshot_pdb.open("w") as handle:
        app.PDBFile.writeFile(pdb.topology, state.getPositions(), handle)
    return snapshot_pdb, {
        "source_atoms": int(pdb.topology.getNumAtoms()),
        "steps": int(args.openmm_steps),
        "time_ps": float(args.openmm_steps * args.timestep_fs / 1000.0),
        "temperature_K": float(args.temperature),
        "run_method": args.run_method,
        "initial_potential_kj_mol": float(initial_energy),
        "minimized_potential_kj_mol": float(minimized_energy),
        "final_potential_kj_mol": float(final_energy),
    }


def _openmm_system(pdb, forcefield, method, cutoff_nm, ewald_tolerance, app, unit):
    nonbonded_method = app.PME if method == "pme" else app.CutoffPeriodic
    return forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=float(cutoff_nm) * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ignoreExternalBonds=True,
        ewaldErrorTolerance=float(ewald_tolerance),
    )


def _force_inventory(system):
    inventory = []
    for index, force in enumerate(system.getForces()):
        cls = type(force).__name__
        entry = {
            "index": int(index),
            "class": cls,
            "support": SUPPORTED_FORCE_TERMS.get(cls, "unknown"),
        }
        for attr, key in (
            ("getNumBonds", "bonds"),
            ("getNumAngles", "angles"),
            ("getNumTorsions", "torsions"),
            ("getNumParticles", "particles"),
            ("getNumExceptions", "exceptions"),
            ("getNumExclusions", "exclusions"),
            ("getNumMaps", "maps"),
        ):
            if hasattr(force, attr):
                try:
                    entry[key] = int(getattr(force, attr)())
                except Exception:
                    pass
        inventory.append(entry)
    return inventory


def _openmm_energy_components(pdb, system, openmm, unit):
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)
    integrator = openmm.VerletIntegrator(0.001 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(pdb.positions)
    total = _context_energy_kj_mol(context, unit)
    by_force = {}
    by_class = {}
    for index, force in enumerate(system.getForces()):
        cls = type(force).__name__
        key = f"{cls}#{index}"
        state = context.getState(getEnergy=True, groups={index})
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        by_force[key] = energy
        by_class[cls] = by_class.get(cls, 0.0) + energy
    by_class["total"] = total
    return {"by_force": by_force, "by_class": by_class, "total": total}


def _openmm_forces(pdb, system, openmm, unit):
    integrator = openmm.VerletIntegrator(0.001 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(pdb.positions)
    state = context.getState(getForces=True)
    return np.asarray(
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
        dtype=float,
    )


def _context_energy_kj_mol(context, unit):
    state = context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def _pyqed_import_components_and_forces(snapshot_pdb, args):
    start = time.perf_counter()
    frame = atoms_from_openmm_pdb_system(
        snapshot_pdb,
        tuple(args.forcefield),
        attach_calculator=not args.skip_pyqed_energy,
        nonbonded_method=args.compare_method,
        nonbonded_cutoff_nm=args.cutoff_nm,
        constraints="HBonds",
        rigid_water=True,
    )
    atoms = frame.atoms
    elapsed = time.perf_counter() - start
    residue_counts = _residue_counts(atoms)
    atom_names = atoms.get_array("atom_names")
    lipid_mask = np.asarray([name in {"DPP", "DPPC", "POPC", "POPE"} for name in atoms.get_array("residue_names")])
    phosphorus = lipid_mask & (atom_names == "P")
    positions_angstrom = atoms.get_positions() * au2angstrom
    lengths_angstrom = atoms.get_cell().lengths() * au2angstrom
    lipids = int(residue_counts.get("DPP", 0) + residue_counts.get("DPPC", 0) + residue_counts.get("POPC", 0))
    pyqed_import = {
        "atoms": int(len(atoms)),
        "residue_counts": residue_counts,
        "charge_sum": float(np.sum(atoms.get_array("charges"))),
        "pbc": [bool(value) for value in atoms.get_pbc()],
        "cell_angstrom": [float(value) for value in lengths_angstrom],
        "constraints": int(len(getattr(atoms, "constraints", []))),
        "topology_terms": _topology_counts(getattr(atoms, "topology", None)),
        "import_seconds": float(elapsed),
    }
    if lipids:
        pyqed_import["lipids"] = lipids
        pyqed_import["area_per_lipid_angstrom2"] = float(lengths_angstrom[0] * lengths_angstrom[1] / max(lipids / 2.0, 1.0))
    if np.any(phosphorus):
        z = positions_angstrom[phosphorus, 2]
        center = float(np.mean(z))
        upper = z[z >= center]
        lower = z[z < center]
        if len(upper) and len(lower):
            pyqed_import["bilayer_thickness_angstrom"] = float(np.mean(upper) - np.mean(lower))

    if args.skip_pyqed_energy:
        return pyqed_import, None, None

    start = time.perf_counter()
    components = atoms.calc.energy_components(atoms)
    pyqed_import["energy_seconds"] = float(time.perf_counter() - start)
    start = time.perf_counter()
    forces = np.asarray(atoms.calc.get_forces(atoms), dtype=float) * HARTREE_TO_KJMOL / BOHR_TO_NM
    pyqed_import["force_seconds"] = float(time.perf_counter() - start)
    return pyqed_import, {
        name: float(value) * HARTREE_TO_KJMOL
        for name, value in components.items()
    }, forces


def _topology_counts(topology):
    if topology is None:
        return {}
    return {
        "bonds": len(getattr(topology, "bonds", ())),
        "angles": len(getattr(topology, "angles", ())),
        "torsions": len(getattr(topology, "torsions", ())),
        "impropers": len(getattr(topology, "impropers", ())),
        "cmaps": len(getattr(topology, "cmaps", ())),
        "nonbonded_exclusions": len(getattr(topology, "nonbonded_exclusions", ())),
        "lj_exclusions": len(getattr(topology, "lj_exclusions", ())),
        "coulomb_exclusions": len(getattr(topology, "coulomb_exclusions", ())),
        "lj_pair_scales": len(getattr(topology, "lj_pair_scales", ())),
        "coulomb_pair_parameters": len(getattr(topology, "coulomb_pair_parameters", ())),
        "lj_pair_overrides": len(getattr(topology, "lj_pair_overrides", ())),
        "lj_pair_parameters": len(getattr(topology, "lj_pair_parameters", ())),
    }


def _component_comparison(openmm_components, pyqed_components):
    if pyqed_components is None:
        return None
    by_class = openmm_components["by_class"]
    mapping = {
        "bonds": ("HarmonicBondForce",),
        "angles": ("HarmonicAngleForce",),
        "torsions": ("PeriodicTorsionForce",),
        "impropers": ("CustomTorsionForce",),
        "cmaps": ("CMAPTorsionForce",),
        "nonbonded": ("NonbondedForce", "CustomNonbondedForce", "CustomBondForce"),
        "total": ("total",),
    }
    comparison = {}
    for pyqed_name, openmm_names in mapping.items():
        openmm_value = sum(float(by_class.get(name, 0.0)) for name in openmm_names)
        pyqed_value = float(pyqed_components.get(pyqed_name, 0.0))
        comparison[pyqed_name] = {
            "openmm": openmm_value,
            "pyqed": pyqed_value,
            "delta_openmm_minus_pyqed": openmm_value - pyqed_value,
        }
    return comparison


def _force_comparison(openmm_forces, pyqed_forces):
    openmm_forces = np.asarray(openmm_forces, dtype=float)
    pyqed_forces = np.asarray(pyqed_forces, dtype=float)
    if openmm_forces.shape != pyqed_forces.shape:
        raise ValueError("OpenMM and PyQED force arrays have different shapes.")
    delta = openmm_forces - pyqed_forces
    delta_norm = np.linalg.norm(delta, axis=1)
    openmm_norm = np.linalg.norm(openmm_forces, axis=1)
    rms_delta = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    rms_openmm = float(np.sqrt(np.mean(np.sum(openmm_forces * openmm_forces, axis=1))))
    max_index = int(np.argmax(delta_norm)) if len(delta_norm) else -1
    return {
        "atoms": int(len(openmm_forces)),
        "rms_delta_kj_mol_nm": rms_delta,
        "max_delta_kj_mol_nm": float(delta_norm[max_index]) if max_index >= 0 else 0.0,
        "mean_delta_kj_mol_nm": float(np.mean(delta_norm)) if len(delta_norm) else 0.0,
        "rms_openmm_kj_mol_nm": rms_openmm,
        "relative_rms_delta": float(rms_delta / rms_openmm) if rms_openmm > 0.0 else 0.0,
        "max_delta_atom": max_index,
        "max_openmm_force_kj_mol_nm": float(np.max(openmm_norm)) if len(openmm_norm) else 0.0,
    }


def _readiness_report(
    inventory,
    comparison,
    pyqed_components,
    force_comparison,
    energy_tolerance,
    force_rms_tolerance,
    force_max_tolerance,
):
    force_gaps = []
    force_warnings = []
    for item in inventory:
        support = item.get("support", "unknown")
        if support == "unknown":
            force_gaps.append({"force": item["class"], "index": item["index"], "support": support})
        elif support in {"partial", "partial-charmm-nbfix", "partial-charmm-nbfix-14", "harmonic-improper-only"}:
            force_warnings.append({"force": item["class"], "index": item["index"], "support": support})
    energy_ready = False
    total_delta = None
    if comparison is not None:
        total_delta = abs(float(comparison["total"]["delta_openmm_minus_pyqed"]))
        energy_ready = bool(total_delta <= float(energy_tolerance))
    force_ready = False
    if force_comparison is not None:
        force_ready = (
            force_comparison["rms_delta_kj_mol_nm"] <= float(force_rms_tolerance)
            and force_comparison["max_delta_kj_mol_nm"] <= float(force_max_tolerance)
        )
    return {
        "import_ready": True,
        "native_energy_evaluated": pyqed_components is not None,
        "energy_parity_ready": energy_ready,
        "force_parity_ready": force_ready,
        "force_comparison_evaluated": force_comparison is not None,
        "energy_tolerance_kj_mol": float(energy_tolerance),
        "force_rms_tolerance_kj_mol_nm": float(force_rms_tolerance),
        "force_max_tolerance_kj_mol_nm": float(force_max_tolerance),
        "total_abs_delta_kj_mol": total_delta,
        "force_rms_delta_kj_mol_nm": None if force_comparison is None else force_comparison["rms_delta_kj_mol_nm"],
        "force_max_delta_kj_mol_nm": None if force_comparison is None else force_comparison["max_delta_kj_mol_nm"],
        "force_gaps": force_gaps,
        "force_warnings": force_warnings,
        "workflow_ready": bool(energy_ready and force_ready and not force_gaps),
        "recommended_next": [
            "add force-vector parity tests for full DPPC/POPC patches",
            "validate PME energy and force parity on OpenMM membrane snapshots",
            "use OpenMM-built snapshots for PyQED QM/MM and embedding while native parity improves",
        ],
    }


def _write_component_table(path, comparison):
    with Path(path).open("w") as handle:
        handle.write("component openmm_kj_mol pyqed_kj_mol delta_openmm_minus_pyqed_kj_mol\n")
        if comparison is None:
            return
        for name, values in comparison.items():
            handle.write(
                f"{name} {values['openmm']:.12e} {values['pyqed']:.12e} "
                f"{values['delta_openmm_minus_pyqed']:.12e}\n"
            )


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
    lipid = np.isin(residue_names, ["DPP", "DPPC", "POPC", "POPE"])
    water = residue_names == "HOH"
    heads = lipid & np.isin(atom_names, ["N", "P", "O11", "O12", "O13", "O14"])
    tails = lipid & ~heads
    styles = [
        ("tails", tails, "#777777", 1.0, 0.32),
        ("heads", heads, "#7b3294", 7.0, 0.82),
        ("water", water, "#2c7fb8", 1.0, 0.16),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    for ax, dims, title in ((axes[0], (0, 1), "Top view (x-y)"), (axes[1], (0, 2), "Side view (x-z)")):
        for label, mask, color, size, alpha in styles:
            ax.scatter(coords[mask, dims[0]], coords[mask, dims[1]], s=size, c=color, alpha=alpha, label=label, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Angstrom")
        ax.set_ylabel("Angstrom")
        ax.grid(True, alpha=0.18)
    axes[0].legend(loc="upper right", frameon=False, markerscale=4.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _residue_counts(atoms):
    residue_names = atoms.get_array("residue_names")
    residue_ids = atoms.get_array("residue_ids")
    chain_ids = atoms.get_array("chain_ids")
    seen = set()
    counts = {}
    for name, resid, chain in zip(residue_names, residue_ids, chain_ids):
        key = (str(name), str(resid), str(chain))
        if key in seen:
            continue
        seen.add(key)
        counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def _counts(values):
    counts = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _print_summary(summary_path, table_path, render_path, summary):
    print(summary_path)
    print(f"snapshot_pdb: {summary['snapshot_pdb']}")
    print(f"atoms: {summary['pyqed_import']['atoms']}")
    print(f"lipids: {summary['pyqed_import'].get('lipids', 0)}")
    print(f"waters: {summary['pyqed_import']['residue_counts'].get('HOH', 0)}")
    print(f"native_energy_evaluated: {summary['readiness']['native_energy_evaluated']}")
    print(f"energy_parity_ready: {summary['readiness']['energy_parity_ready']}")
    print(f"force_parity_ready: {summary['readiness']['force_parity_ready']}")
    print(f"workflow_ready: {summary['readiness']['workflow_ready']}")
    if summary["readiness"]["total_abs_delta_kj_mol"] is not None:
        print(f"total_abs_delta_kj_mol: {summary['readiness']['total_abs_delta_kj_mol']:.12e}")
    if summary["readiness"]["force_rms_delta_kj_mol_nm"] is not None:
        print(f"force_rms_delta_kj_mol_nm: {summary['readiness']['force_rms_delta_kj_mol_nm']:.12e}")
        print(f"force_max_delta_kj_mol_nm: {summary['readiness']['force_max_delta_kj_mol_nm']:.12e}")
    print(f"force_gaps: {len(summary['readiness']['force_gaps'])}")
    print(f"force_warnings: {len(summary['readiness'].get('force_warnings', []))}")
    print(f"component_table: {table_path}")
    if render_path is not None:
        print(f"render: {render_path}")


if __name__ == "__main__":
    raise SystemExit(main())
