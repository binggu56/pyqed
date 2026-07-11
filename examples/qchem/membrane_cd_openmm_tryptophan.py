"""OpenMM membrane CD workflow with a mobile L-tryptophan probe.

Tryptophan is a common membrane-environment reporter because the indole group
is a real UV chromophore.  This example appends a simple mobile neutral
L-tryptophan probe to a toy all-atom membrane, runs a short OpenMM trajectory,
and computes membrane-embedded PyQED CD on the Trp coordinates from each frame.

The force-field parameters below are deliberately lightweight toy parameters.
For production, replace them with the chromophore parameters from the same
force field used for the membrane.
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
from pyqed.units import au2angstrom, au2ev, au2nm


TRP_SYMBOLS = (
    "N", "H", "C", "H", "C", "C", "C", "H", "C", "H", "C", "H",
    "C", "H", "C", "C", "H", "H", "C", "H", "N", "H", "H", "C",
    "O", "O", "H",
)

# Approximate neutral L-tryptophan geometry in Bohr.  The indole group is
# planar; the alpha carbon is displaced out of the indole plane to make a
# chiral amino-acid probe.
TRP_LOCAL = np.array(
    [
        [-1.10, 0.70, 0.00],   # indole N-H
        [-1.75, 1.35, 0.00],
        [0.10, 1.20, 0.00],
        [0.25, 2.27, 0.00],
        [0.95, 0.10, 0.00],
        [0.15, -0.95, 0.00],
        [0.60, -2.25, 0.00],
        [1.65, -2.45, 0.00],
        [-0.35, -3.25, 0.00],
        [-0.05, -4.30, 0.00],
        [-1.70, -2.95, 0.00],
        [-2.45, -3.75, 0.00],
        [-2.10, -1.65, 0.00],
        [-3.15, -1.45, 0.00],
        [-1.15, -0.65, 0.00],
        [2.35, 0.10, 0.10],    # beta carbon
        [2.75, 1.00, 0.55],
        [2.75, -0.80, 0.55],
        [3.10, 0.10, -1.25],   # alpha carbon
        [2.60, 0.90, -1.75],
        [4.45, 0.25, -1.05],   # amino group
        [4.85, 0.95, -1.65],
        [4.90, -0.60, -1.35],
        [2.65, -1.15, -2.00],  # carboxylic acid
        [1.50, -1.45, -2.10],
        [3.55, -1.90, -2.55],
        [3.20, -2.70, -2.95],
    ],
    dtype=float,
) / au2angstrom

TRP_BONDS = (
    (0, 1), (0, 2), (0, 14), (2, 3), (2, 4), (4, 5), (4, 15),
    (5, 6), (5, 14), (6, 7), (6, 8), (8, 9), (8, 10), (10, 11),
    (10, 12), (12, 13), (12, 14), (15, 16), (15, 17), (15, 18),
    (18, 19), (18, 20), (18, 23), (20, 21), (20, 22), (23, 24),
    (23, 25), (25, 26),
)

TRP_CHARGES = np.zeros(len(TRP_SYMBOLS), dtype=float)
TRP_CHARGES[0] = -0.30
TRP_CHARGES[1] = 0.30
TRP_CHARGES[18] = -0.10
TRP_CHARGES[19] = 0.10
TRP_CHARGES[20] = -0.60
TRP_CHARGES[21] = 0.30
TRP_CHARGES[22] = 0.30
TRP_CHARGES[23] = 0.60
TRP_CHARGES[24] = -0.50
TRP_CHARGES[25] = -0.50
TRP_CHARGES[26] = 0.40


def _trp_lj_and_masses():
    epsilon_by_symbol = {"C": 0.00015, "N": 0.00025, "O": 0.00025, "H": 0.00002}
    sigma_by_symbol = {"C": 6.5, "N": 6.0, "O": 5.8, "H": 3.5}
    mass_by_symbol = {"C": 12.011, "N": 14.007, "O": 15.999, "H": 1.008}
    epsilon = np.asarray([epsilon_by_symbol[symbol] for symbol in TRP_SYMBOLS], dtype=float)
    sigma = np.asarray([sigma_by_symbol[symbol] for symbol in TRP_SYMBOLS], dtype=float)
    masses = np.asarray([mass_by_symbol[symbol] for symbol in TRP_SYMBOLS], dtype=float)
    return epsilon, sigma, masses


TRP_LJ_EPSILON, TRP_LJ_SIGMA, TRP_MASSES_AMU = _trp_lj_and_masses()


def openmm_tryptophan_frames(
    nframes=2,
    steps_per_frame=1,
    timestep_fs=0.005,
    temperature_K=50.0,
    friction_ps=10.0,
    minimize_iterations=10,
    nonbonded_method="pme",
):
    atoms, _atom_types = build_membrane(
        nx=2,
        ny=2,
        waters_per_lipid=1,
        salt_pairs=0,
        coulomb_method="pme" if nonbonded_method == "pme" else "cutoff",
        pme_mesh=16,
    )
    atoms.calc.ewald_alpha = 0.10
    atoms.calc.pme_mesh = np.array([16, 16, 16], dtype=int)
    atoms.calc.lj_energy_shift = False
    atoms, qm_indices = _append_tryptophan_probe(atoms)

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
        frames.append(_snapshot_from_state(atoms, _state_positions_bohr(state, unit)))
        openmm_energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        times_fs.append(float(iframe * steps_per_frame * timestep_fs))
    return frames, qm_indices, np.asarray(openmm_energies), np.asarray(times_fs)


def _append_tryptophan_probe(membrane):
    center = _low_overlap_center(membrane)
    local = TRP_LOCAL - np.mean(TRP_LOCAL, axis=0)
    trp_positions = local + center
    records = [[symbol, tuple(coord)] for symbol, coord in zip(membrane.atom_symbols(), membrane.get_positions())]
    records.extend([[symbol, tuple(coord)] for symbol, coord in zip(TRP_SYMBOLS, trp_positions)])

    qm_start = len(membrane)
    qm_indices = np.arange(qm_start, qm_start + len(TRP_SYMBOLS), dtype=int)
    atoms = Atoms(
        records,
        cell=membrane.get_cell(),
        pbc=membrane.get_pbc(),
        constraint=list(membrane.constraints) + [_trp_constraints(qm_indices)],
    )

    calc = membrane.calc
    charges = np.concatenate([membrane.get_array("charges"), TRP_CHARGES])
    lj_epsilon = np.concatenate([calc.lj_epsilon, TRP_LJ_EPSILON])
    lj_sigma = np.concatenate([calc.lj_sigma, TRP_LJ_SIGMA])
    masses = np.concatenate([membrane.get_masses_amu(), TRP_MASSES_AMU])
    atom_types = np.concatenate([
        np.asarray([f"M{index}" for index in range(len(membrane))], dtype=str),
        np.asarray([f"T{index}" for index in range(len(TRP_SYMBOLS))], dtype=str),
    ])
    nonbonded_exclusions = set(calc.nonbonded_exclusions)
    for i, local_i in enumerate(range(len(TRP_SYMBOLS) - 1)):
        for local_j in range(local_i + 1, len(TRP_SYMBOLS)):
            nonbonded_exclusions.add(tuple(sorted((int(qm_indices[local_i]), int(qm_indices[local_j])))))

    atoms.calc = MolecularMechanics(
        bonds=calc.bonds,
        angles=calc.angles,
        torsions=calc.torsions,
        impropers=calc.impropers,
        charges=charges,
        coulomb_constant=calc.coulomb_constant,
        coulomb_method=calc.coulomb_method,
        coulomb_cutoff=calc.coulomb_cutoff,
        ewald_alpha=calc.ewald_alpha,
        ewald_kmax=calc.ewald_kmax,
        pme_mesh=calc.pme_mesh,
        pme_order=getattr(calc, "pme_order", 4),
        lj_epsilon=lj_epsilon,
        lj_sigma=lj_sigma,
        lj_cutoff=calc.lj_cutoff,
        lj_switch_on=calc.lj_switch_on,
        lj_energy_shift=calc.lj_energy_shift,
        atom_types=atom_types,
        lj_pair_overrides=calc.lj_pair_overrides,
        nonbonded_exclusions=nonbonded_exclusions,
        lj_exclusions=calc.lj_exclusions,
        coulomb_exclusions=calc.coulomb_exclusions,
        lj_pair_scales=calc.lj_pair_scales,
        coulomb_pair_scales=calc.coulomb_pair_scales,
        nonbonded_skin=calc.nonbonded_skin,
    )
    atoms.set_array("charges", charges, float, ())
    atoms.set_array("lj_epsilon", lj_epsilon, float, ())
    atoms.set_array("lj_sigma", lj_sigma, float, ())
    atoms.set_array("atom_types", atom_types, str, ())
    atoms.set_array("masses_amu", masses, float, ())

    leaflets = np.zeros(len(atoms), dtype=int)
    z = membrane.get_positions()[:, 2]
    leaflets[:len(membrane)] = np.where(z >= np.median(z), 1, -1)
    atoms.set_array("leaflets", leaflets, int, ())
    return atoms, qm_indices


def _trp_constraints(qm_indices):
    pairs = [(int(qm_indices[i]), int(qm_indices[j])) for i, j in TRP_BONDS]
    distances = [float(np.linalg.norm(TRP_LOCAL[i] - TRP_LOCAL[j])) for i, j in TRP_BONDS]
    return FixBondLengths(pairs, distances=distances)


def _low_overlap_center(membrane):
    positions = np.asarray(membrane.get_positions(), dtype=float)
    cell = membrane.get_cell()
    pbc = membrane.get_pbc()
    lengths = np.asarray(cell.lengths(), dtype=float)
    local = TRP_LOCAL - np.mean(TRP_LOCAL, axis=0)
    x_grid = np.linspace(0.15 * lengths[0], 0.85 * lengths[0], 7)
    y_grid = np.linspace(0.15 * lengths[1], 0.85 * lengths[1], 7)
    z_grid_angstrom = np.array([-12.0, -10.0, -8.0, 8.0, 10.0, 12.0])
    best_distance = -np.inf
    best_center = np.array([0.5 * lengths[0], 0.5 * lengths[1], -10.0 / au2angstrom])
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


def _snapshot_from_state(template_atoms, positions_bohr):
    snapshot = Atoms(
        [
            [symbol, tuple(coord)]
            for symbol, coord in zip(template_atoms.atom_symbols(), positions_bohr)
        ],
        cell=template_atoms.get_cell(),
        pbc=template_atoms.get_pbc(),
    )
    snapshot.set_array("charges", template_atoms.get_array("charges"), float, ())
    if template_atoms.has("leaflets"):
        snapshot.set_array("leaflets", template_atoms.get_array("leaflets"), int, ())
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nframes", type=int, default=2)
    parser.add_argument("--steps-per-frame", type=int, default=1)
    parser.add_argument("--timestep-fs", type=float, default=0.005)
    parser.add_argument("--temperature-K", type=float, default=50.0)
    parser.add_argument("--friction-ps", type=float, default=10.0)
    parser.add_argument("--minimize-iterations", type=int, default=10)
    parser.add_argument("--method", choices=("tda", "tddft"), default="tda")
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--output-prefix", default="/private/tmp/pyqed_openmm_trp_membrane_cd")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    frames, qm_indices, openmm_energies, times_fs = openmm_tryptophan_frames(
        nframes=args.nframes,
        steps_per_frame=args.steps_per_frame,
        timestep_fs=args.timestep_fs,
        temperature_K=args.temperature_K,
        friction_ps=args.friction_ps,
        minimize_iterations=args.minimize_iterations,
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
        build_kwargs={"eri": "s8"},
        mf_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    result = workflow.run()
    x, signal = result.spectrum(width=0.35, units="ev")

    prefix = Path(args.output_prefix)
    table = prefix.with_suffix(".txt")
    with table.open("w") as handle:
        handle.write("# L-tryptophan membrane CD, toy force-field parameters\n")
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

    print(f"chromophore: L-tryptophan")
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
        ax.plot(x, signal, color="#5b6f2a", linewidth=2.0)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Averaged CD intensity (arb.)")
        ax.set_title("OpenMM Membrane CD: L-Tryptophan")
        fig.tight_layout()
        figure = prefix.with_suffix(".png")
        fig.savefig(figure, dpi=200)
        print(f"figure: {figure}")


if __name__ == "__main__":
    main()
