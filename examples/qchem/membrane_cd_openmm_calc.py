"""Run molecular CD from OpenMM membrane dynamics snapshots.

This script is the OpenMM-backed version of the membrane CD workflow:

1. Build a small all-atom toy membrane.
2. Append a mobile neutral chiral H2O2 probe to the OpenMM system.
3. Propagate the combined membrane/probe system with OpenMM.
4. Extract membrane point charges and run embedded TDA/TDDFT CD with PyQED.

The toy membrane keeps the runtime modest.  For production, replace
``build_membrane(...)`` with CHARMM-GUI/OpenMM inputs and use the same
``MembraneCD`` call on saved trajectory frames.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.md.all_atom_lipid_membrane import build_membrane
from examples.md.openmm_all_atom_lipid_membrane_reference import (
    _openmm_topology,
    _state_positions_bohr,
    build_openmm_system,
)
from pyqed.md import Atoms, FixBondLengths, MolecularMechanics
from pyqed.md.neighborlist import minimum_image
from pyqed.qchem import MembraneCD
from pyqed.units import au2angstrom, au2ev, au2fs, au2nm


H2O2_LOCAL = np.array(
    [
        [0.000, 0.000, 0.000],
        [2.740, 0.000, 0.000],
        [-0.850, 1.436, 0.000],
        [3.590, 1.436, 1.134],
    ],
    dtype=float,
)
H2O2_SYMBOLS = ("O", "O", "H", "H")
H2O2_CHARGES = np.array([-0.35, -0.35, 0.35, 0.35], dtype=float)
H2O2_LJ_EPSILON = np.array([0.00025, 0.00025, 0.00002, 0.00002], dtype=float)
H2O2_LJ_SIGMA = np.array([5.7, 5.7, 3.5, 3.5], dtype=float)
H2O2_MASSES_AMU = np.array([15.999, 15.999, 1.008, 1.008], dtype=float)


def openmm_membrane_frames(
    nframes=3,
    steps_per_frame=1,
    timestep_fs=0.01,
    temperature_K=50.0,
    friction_ps=10.0,
    minimize_iterations=10,
    nonbonded_method="pme",
):
    """Return PyQED membrane snapshots from a short OpenMM trajectory."""

    atoms, _atom_types = build_membrane(
        nx=1,
        ny=1,
        waters_per_lipid=1,
        salt_pairs=0,
        coulomb_method="pme" if nonbonded_method == "pme" else "cutoff",
        pme_mesh=12,
    )
    atoms.calc.ewald_alpha = 0.10
    atoms.calc.pme_mesh = np.array([12, 12, 12], dtype=int)
    atoms.calc.lj_energy_shift = False
    atoms, qm_indices = _append_mobile_chromophore(atoms)

    system, openmm, unit = build_openmm_system(atoms, nonbonded_method=nonbonded_method)
    integrator = openmm.LangevinMiddleIntegrator(
        float(temperature_K) * unit.kelvin,
        float(friction_ps) / unit.picosecond,
        float(timestep_fs) * unit.femtosecond,
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = openmm.app.Simulation(_openmm_topology(atoms), system, integrator, platform)
    simulation.context.setPositions((atoms.get_positions() * au2nm) * unit.nanometer)

    if minimize_iterations:
        openmm.LocalEnergyMinimizer.minimize(
            simulation.context,
            10.0 * unit.kilojoule_per_mole / unit.nanometer,
            int(minimize_iterations),
        )

    frames = []
    openmm_energies = []
    times_fs = []
    for iframe in range(int(nframes)):
        if iframe:
            simulation.step(int(steps_per_frame))
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        positions_bohr = _state_positions_bohr(state, unit)
        frames.append(_snapshot_from_openmm_state(atoms, positions_bohr))
        openmm_energies.append(float(energy))
        times_fs.append(float(iframe * steps_per_frame * timestep_fs))
    return frames, qm_indices, np.asarray(openmm_energies), np.asarray(times_fs)


def _append_mobile_chromophore(membrane):
    lengths = np.asarray(membrane.get_cell().lengths(), dtype=float)
    chrom_center = _low_overlap_chromophore_center(membrane)
    local_center = np.mean(H2O2_LOCAL, axis=0)
    qm_positions = H2O2_LOCAL - local_center + chrom_center

    atom_records = [
        [symbol, tuple(coord)]
        for symbol, coord in zip(membrane.atom_symbols(), membrane.get_positions())
    ]
    atom_records.extend(
        [symbol, tuple(coord)]
        for symbol, coord in zip(H2O2_SYMBOLS, qm_positions)
    )
    qm_start = len(membrane)
    qm_indices = np.arange(qm_start, qm_start + len(H2O2_SYMBOLS), dtype=int)
    atoms = Atoms(
        atom_records,
        cell=membrane.get_cell(),
        pbc=membrane.get_pbc(),
        constraint=list(membrane.constraints) + [_h2o2_constraints(qm_indices)],
    )

    membrane_calc = membrane.calc
    charges = np.concatenate([membrane.get_array("charges"), H2O2_CHARGES])
    lj_epsilon = np.concatenate([membrane_calc.lj_epsilon, H2O2_LJ_EPSILON])
    lj_sigma = np.concatenate([membrane_calc.lj_sigma, H2O2_LJ_SIGMA])
    membrane_atom_types = membrane_calc.atom_types
    if membrane_atom_types is None:
        membrane_atom_types = np.asarray([f"M{index}" for index in range(len(membrane))], dtype=str)
    atom_types = np.concatenate([
        np.asarray(membrane_atom_types, dtype=str),
        np.asarray(["QO", "QO", "QH", "QH"], dtype=str),
    ])
    masses = np.concatenate([membrane.get_masses_amu(), H2O2_MASSES_AMU])
    nonbonded_exclusions = set(membrane_calc.nonbonded_exclusions)
    for i, index_i in enumerate(qm_indices[:-1]):
        for index_j in qm_indices[i + 1:]:
            nonbonded_exclusions.add(tuple(sorted((int(index_i), int(index_j)))))

    atoms.calc = MolecularMechanics(
        bonds=membrane_calc.bonds,
        angles=membrane_calc.angles,
        torsions=membrane_calc.torsions,
        impropers=membrane_calc.impropers,
        charges=charges,
        coulomb_constant=membrane_calc.coulomb_constant,
        coulomb_method=membrane_calc.coulomb_method,
        coulomb_cutoff=membrane_calc.coulomb_cutoff,
        ewald_alpha=membrane_calc.ewald_alpha,
        ewald_kmax=membrane_calc.ewald_kmax,
        pme_mesh=membrane_calc.pme_mesh,
        pme_order=getattr(membrane_calc, "pme_order", 4),
        lj_epsilon=lj_epsilon,
        lj_sigma=lj_sigma,
        lj_cutoff=membrane_calc.lj_cutoff,
        lj_switch_on=membrane_calc.lj_switch_on,
        lj_energy_shift=membrane_calc.lj_energy_shift,
        atom_types=atom_types,
        lj_pair_overrides=membrane_calc.lj_pair_overrides,
        nonbonded_exclusions=nonbonded_exclusions,
        lj_exclusions=membrane_calc.lj_exclusions,
        coulomb_exclusions=membrane_calc.coulomb_exclusions,
        lj_pair_scales=membrane_calc.lj_pair_scales,
        coulomb_pair_scales=membrane_calc.coulomb_pair_scales,
        nonbonded_skin=membrane_calc.nonbonded_skin,
    )
    atoms.set_array("charges", charges, float, ())
    atoms.set_array("lj_epsilon", lj_epsilon, float, ())
    atoms.set_array("lj_sigma", lj_sigma, float, ())
    atoms.set_array("atom_types", atom_types, str, ())
    atoms.set_array("masses_amu", masses, float, ())

    regions = np.concatenate([
        membrane.get_array("regions") if membrane.has("regions") else np.zeros(len(membrane), dtype=int),
        np.full(len(H2O2_SYMBOLS), 4, dtype=int),
    ])
    atoms.set_array("regions", regions, int, ())
    leaflets = np.zeros(len(atoms), dtype=int)
    membrane_z = membrane.get_positions()[:, 2]
    leaflets[:len(membrane)] = np.where(membrane_z >= np.median(membrane_z), 1, -1)
    atoms.set_array("leaflets", leaflets, int, ())
    return atoms, qm_indices


def _low_overlap_chromophore_center(membrane):
    positions = np.asarray(membrane.get_positions(), dtype=float)
    cell = membrane.get_cell()
    pbc = membrane.get_pbc()
    lengths = np.asarray(cell.lengths(), dtype=float)
    local = H2O2_LOCAL - np.mean(H2O2_LOCAL, axis=0)
    x_grid = np.linspace(0.15 * lengths[0], 0.85 * lengths[0], 8)
    y_grid = np.linspace(0.15 * lengths[1], 0.85 * lengths[1], 8)
    z_grid_angstrom = np.array([-10.0, -8.0, -6.0, -4.0, 0.0, 4.0, 8.0, 10.0])
    best_distance = -np.inf
    best_center = np.array([0.5 * lengths[0], 0.5 * lengths[1], 0.0])
    for x in x_grid:
        for y in y_grid:
            for z_angstrom in z_grid_angstrom:
                center = np.array([x, y, z_angstrom / au2angstrom])
                probe = local + center
                min_distance = np.inf
                for probe_coord in probe:
                    deltas = np.array([
                        minimum_image(probe_coord - env_coord, cell, pbc)
                        for env_coord in positions
                    ])
                    min_distance = min(min_distance, float(np.linalg.norm(deltas, axis=1).min()))
                if min_distance > best_distance:
                    best_distance = min_distance
                    best_center = center
    return best_center


def _h2o2_constraints(qm_indices):
    pairs = [
        (int(qm_indices[0]), int(qm_indices[1])),
        (int(qm_indices[0]), int(qm_indices[2])),
        (int(qm_indices[1]), int(qm_indices[3])),
    ]
    distances = [
        float(np.linalg.norm(H2O2_LOCAL[1] - H2O2_LOCAL[0])),
        float(np.linalg.norm(H2O2_LOCAL[2] - H2O2_LOCAL[0])),
        float(np.linalg.norm(H2O2_LOCAL[3] - H2O2_LOCAL[1])),
    ]
    return FixBondLengths(pairs, distances=distances)


def _snapshot_from_openmm_state(template_atoms, positions_bohr):
    positions_bohr = np.asarray(positions_bohr, dtype=float)
    snapshot = Atoms(
        [
            [symbol, tuple(coord)]
            for symbol, coord in zip(template_atoms.atom_symbols(), positions_bohr)
        ],
        cell=template_atoms.get_cell(),
        pbc=template_atoms.get_pbc(),
    )
    snapshot.set_array(
        "charges",
        template_atoms.get_array("charges"),
        float,
        (),
    )
    if template_atoms.has("leaflets"):
        snapshot.set_array("leaflets", template_atoms.get_array("leaflets"), int, ())
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nframes", type=int, default=3)
    parser.add_argument("--steps-per-frame", type=int, default=1)
    parser.add_argument("--timestep-fs", type=float, default=0.01)
    parser.add_argument("--temperature-K", type=float, default=50.0)
    parser.add_argument("--friction-ps", type=float, default=10.0)
    parser.add_argument("--minimize-iterations", type=int, default=10)
    parser.add_argument("--nonbonded-method", choices=("pme", "cutoff"), default="pme")
    parser.add_argument("--method", choices=("tda", "tddft"), default="tda")
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--output-prefix", default="/private/tmp/pyqed_openmm_membrane_cd")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    frames, qm_indices, openmm_energies, times_fs = openmm_membrane_frames(
        nframes=args.nframes,
        steps_per_frame=args.steps_per_frame,
        timestep_fs=args.timestep_fs,
        temperature_K=args.temperature_K,
        friction_ps=args.friction_ps,
        minimize_iterations=args.minimize_iterations,
        nonbonded_method=args.nonbonded_method,
    )
    workflow = MembraneCD(
        frames,
        qm_indices=qm_indices,
        basis=args.basis,
        method=args.method,
        nstates=args.nstates,
        cutoff=12.0 / au2angstrom,
        embedding_pbc="nearest",
        cap_charge_distance=1.5 / au2angstrom,
        mf_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    result = workflow.run()
    x, signal = result.spectrum(width=0.35, units="ev")

    prefix = Path(args.output_prefix)
    table = prefix.with_suffix(".txt")
    with table.open("w") as handle:
        handle.write("# frame time_fs state excitation_eV rotatory_au depth_A ncharges openmm_kJ_mol\n")
        for iframe, frame in enumerate(result.frames):
            for istate, (energy, strength) in enumerate(
                zip(frame.cd_result.excitation_energies * au2ev, frame.cd_result.rotatory_strengths),
                start=1,
            ):
                handle.write(
                    f"{iframe:5d} {times_fs[iframe]:12.6f} {istate:5d} "
                    f"{energy:16.8f} {strength:16.8e} "
                    f"{frame.snapshot.depth * au2angstrom:12.6f} "
                    f"{len(frame.snapshot.charges):8d} {openmm_energies[iframe]:16.8f}\n"
                )

    spectrum = prefix.with_name(prefix.name + "_spectrum.txt")
    np.savetxt(spectrum, np.column_stack([x, signal]), header="energy_eV averaged_cd_arb")

    print(f"frames: {len(result.frames)}")
    print(f"openmm_time_fs: {times_fs[-1]:.8f}")
    print(f"openmm_energy_kJ_mol_minmax: {openmm_energies.min():.8f} {openmm_energies.max():.8f}")
    print(f"table: {table}")
    print(f"spectrum: {spectrum}")
    for iframe, frame in enumerate(result.frames):
        energies = frame.cd_result.excitation_energies * au2ev
        strengths = frame.cd_result.rotatory_strengths
        print(
            f"frame {iframe}: charges={len(frame.snapshot.charges)} "
            f"E_eV={np.array2string(energies, precision=4)} "
            f"R={np.array2string(strengths, precision=4)}"
        )

    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.plot(x, signal, color="#355c7d", linewidth=2.0)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Averaged CD intensity (arb.)")
        ax.set_title("OpenMM Membrane CD")
        fig.tight_layout()
        figure = prefix.with_suffix(".png")
        fig.savefig(figure, dpi=200)
        print(f"figure: {figure}")


if __name__ == "__main__":
    main()
