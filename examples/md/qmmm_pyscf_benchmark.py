#!/usr/bin/env python3
"""Benchmark PyQED QM/MM MD against PySCF snapshots.

The trajectory is propagated by PyQED.  If ``--pyscf-every`` is positive, the
embedded STO-3G QM energy and point-charge embedding forces are compared with
PySCF every N MD steps on the same coordinates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pyqed.md import (  # noqa: E402
    EnergyLogger,
    Langevin,
    QMMM,
    XYZTrajectoryWriter,
    set_maxwell_boltzmann_velocities,
    solvate_box,
)
from pyqed import Molecule  # noqa: E402
from pyqed.qchem.dft import RKS  # noqa: E402
from pyqed.units import au2angstrom, au2fs, fs  # noqa: E402
from qmmm_water_box import h2_solute  # noqa: E402
from qmmm_water_in_water import (  # noqa: E402
    max_constraint_error,
    mm_from_topology,
    water_solute,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50, help="MD steps to run.")
    parser.add_argument("--waters", type=int, default=2, help="Maximum number of MM waters.")
    parser.add_argument("--temperature", type=float, default=50.0, help="Temperature in K.")
    parser.add_argument("--timestep-fs", type=float, default=0.02, help="Timestep in fs.")
    parser.add_argument("--box-angstrom", type=float, default=9.0, help="Cubic box length in Angstrom.")
    parser.add_argument("--spacing-angstrom", type=float, default=3.2, help="Water grid spacing in Angstrom.")
    parser.add_argument("--cutoff-angstrom", type=float, default=5.0, help="MM nonbonded cutoff in Angstrom.")
    parser.add_argument("--friction", type=float, default=1e-3, help="Langevin friction in inverse atomic time.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for initial velocities.")
    parser.add_argument(
        "--solute",
        choices=("water", "h2"),
        default="water",
        help="QM solute. Use h2 for the current native RKS-gradient benchmark path.",
    )
    parser.add_argument(
        "--qm-method",
        choices=("rhf", "rks"),
        default="rhf",
        help="QM method for the embedded solute.",
    )
    parser.add_argument("--xc", default="svwn", help="XC functional used when --qm-method=rks.")
    parser.add_argument(
        "--embedding-pbc",
        choices=("none", "nearest", "images"),
        default="none",
        help="Periodic MM-charge embedding mode for the QM Hamiltonian.",
    )
    parser.add_argument(
        "--embedding-cutoff-angstrom",
        type=float,
        default=None,
        help="Real-space image cutoff for --embedding-pbc=images.",
    )
    parser.add_argument(
        "--pyscf-every",
        type=int,
        default=0,
        help="Compare with PySCF every N steps; 0 disables PySCF.",
    )
    parser.add_argument("--output-dir", default="md_outputs", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    box_length = args.box_angstrom / au2angstrom
    spacing = args.spacing_angstrom / au2angstrom
    cutoff = args.cutoff_angstrom / au2angstrom
    embedding_cutoff = (
        None
        if args.embedding_cutoff_angstrom is None
        else args.embedding_cutoff_angstrom / au2angstrom
    )
    timestep = args.timestep_fs * fs

    if args.qm_method == "rks" and args.solute != "h2":
        raise SystemExit(
            "Native RKS QMMM gradients are currently benchmarked for --solute h2. "
            "Use --qm-method rhf for water, or --solute h2 for RKS."
        )

    solute, solute_positions, solute_symbols, min_distance = build_solute(
        args.solute,
        center=(0.5 * box_length, 0.5 * box_length, 0.5 * box_length),
    )
    system = solvate_box(
        solute=solute,
        box_size=(box_length, box_length, box_length),
        spacing=spacing,
        min_distance=min_distance / au2angstrom,
        max_waters=args.waters,
        rigid=True,
        lj_cutoff=cutoff,
        coulomb_cutoff=cutoff,
    )

    qm_indices = np.arange(len(solute_symbols), dtype=int)
    mm_indices = np.arange(len(solute_symbols), len(system), dtype=int)
    system.calc = QMMM(
        qm=qm_method(solute_positions, solute_symbols, method=args.qm_method, xc=args.xc),
        mm=mm_from_topology(system, cutoff),
        qm_indices=qm_indices,
        mm_indices=mm_indices,
        electrostatic_embedding=True,
        embedding_pbc=None if args.embedding_pbc == "none" else args.embedding_pbc,
        embedding_cutoff=embedding_cutoff,
        qm_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    set_maxwell_boltzmann_velocities(system, temperature=args.temperature, seed=args.seed)

    trajectory_path = output_dir / "qmmm_pyscf_benchmark.xyz"
    energy_path = output_dir / "qmmm_pyscf_benchmark_energy.dat"
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

    initial_positions = system.get_positions().copy()
    max_displacement = 0.0
    max_force = 0.0
    max_constraint = 0.0
    max_pyscf_energy_error = 0.0
    max_pyscf_force_error = 0.0
    pyscf_comparisons = 0

    print("step time_fs total_energy qm_energy mm_energy max_force max_displacement constraint_error pyscf_dE pyscf_dF")
    try:
        for step in range(args.steps + 1):
            total_energy = system.get_potential_energy()
            forces = system.get_forces(apply_constraint=False)
            components = dict(system.calc.results)
            positions = system.get_positions()
            displacement = float(np.max(np.linalg.norm(positions - initial_positions, axis=1)))
            constraint_error = max_constraint_error(system)
            force_norm = float(np.max(np.linalg.norm(forces, axis=1)))
            max_displacement = max(max_displacement, displacement)
            max_force = max(max_force, force_norm)
            max_constraint = max(max_constraint, constraint_error)

            pyscf_energy_error = np.nan
            pyscf_force_error = np.nan
            if args.pyscf_every > 0 and step % args.pyscf_every == 0:
                pyscf_energy, pyscf_qm_forces, pyscf_pc_forces = pyscf_embedding(
                    system,
                    qm_indices,
                    components["embedding_coords"],
                    components["embedding_charges"],
                    method=args.qm_method,
                    xc=args.xc,
                )
                pyscf_energy_error = abs(components["qm_energy"] - pyscf_energy)
                qm_force_error = np.max(np.abs(components["qm_forces"] - pyscf_qm_forces))
                pc_force_error = np.max(
                    np.abs(components["embedding_point_charge_forces"] - pyscf_pc_forces)
                )
                pyscf_force_error = float(max(qm_force_error, pc_force_error))
                max_pyscf_energy_error = max(max_pyscf_energy_error, pyscf_energy_error)
                max_pyscf_force_error = max(max_pyscf_force_error, pyscf_force_error)
                pyscf_comparisons += 1

            print(
                f"{step:d} {dynamics.get_time() * au2fs:.8f} "
                f"{total_energy:.12e} {components['qm_energy']:.12e} "
                f"{components['mm_energy']:.12e} {force_norm:.12e} "
                f"{displacement:.12e} {constraint_error:.12e} "
                f"{pyscf_energy_error:.12e} {pyscf_force_error:.12e}"
            )

            if step < args.steps:
                dynamics.run(1)
    finally:
        writer.close()
        logger.close()

    final_positions = system.get_positions()
    final_forces = system.get_forces(apply_constraint=False)
    print(f"atoms: {len(system)}")
    print(f"solute: {args.solute}")
    print(f"qm_atoms: {len(qm_indices)}")
    print(f"qm_method: {args.qm_method}")
    if args.qm_method == "rks":
        print(f"xc: {args.xc}")
    print(f"embedding_pbc: {args.embedding_pbc}")
    if embedding_cutoff is not None:
        print(f"embedding_cutoff_bohr: {embedding_cutoff:.12e}")
    print(f"mm_atoms: {len(mm_indices)}")
    print(f"steps: {dynamics.get_number_of_steps()}")
    print(f"time_fs: {dynamics.get_time() * au2fs:.6f}")
    print(f"max_force: {max_force:.12e}")
    print(f"max_displacement_bohr: {max_displacement:.12e}")
    print(f"max_constraint_error_bohr: {max_constraint:.12e}")
    print(f"pyscf_comparisons: {pyscf_comparisons}")
    print(f"max_pyscf_energy_error_hartree: {max_pyscf_energy_error:.12e}")
    print(f"max_pyscf_force_error: {max_pyscf_force_error:.12e}")
    print(f"finite_positions: {bool(np.all(np.isfinite(final_positions)))}")
    print(f"finite_forces: {bool(np.all(np.isfinite(final_forces)))}")
    print(f"trajectory: {trajectory_path}")
    print(f"energy_log: {energy_path}")


def build_solute(name, center):
    if name == "water":
        solute, positions = water_solute(center=center)
        return solute, positions, ("O", "H", "H"), 2.4
    if name == "h2":
        solute, positions = h2_solute(center=center)
        return solute, positions, ("H", "H"), 2.2
    raise ValueError(f"Unknown solute {name!r}.")


def qm_method(positions, symbols, method="rhf", xc="svwn"):
    atom = "; ".join(
        f"{symbol} {x:.16g} {y:.16g} {z:.16g}"
        for symbol, (x, y, z) in zip(symbols, positions)
    )
    mol = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    mol.build(driver="builtin")
    if method == "rhf":
        return mol.RHF()
    if method == "rks":
        return RKS(mol, xc=xc)
    raise ValueError(f"Unknown QM method {method!r}.")


def pyscf_embedding(system, qm_indices, embedding_coords, embedding_charges, method="rhf", xc="svwn"):
    from pyscf import dft, gto, qmmm, scf

    positions = system.get_positions()
    symbols = system.atom_symbols()
    atom = "; ".join(
        f"{symbols[index]} {x:.16g} {y:.16g} {z:.16g}"
        for index, (x, y, z) in zip(qm_indices, positions[qm_indices])
    )
    pmol = gto.M(atom=atom, unit="Bohr", basis="sto-3g", verbose=0)
    if method == "rhf":
        base = scf.RHF(pmol)
    elif method == "rks":
        base = dft.RKS(pmol)
        base.xc = xc
        base.grids.atom_grid = {"O": (50, 110), "H": (50, 110)}
    else:
        raise ValueError(f"Unknown QM method {method!r}.")
    pyscf_mf = qmmm.mm_charge(
        base,
        embedding_coords,
        embedding_charges,
        unit="Bohr",
    ).run(verbose=0)
    pyscf_grad = pyscf_mf.nuc_grad_method()
    pyscf_qm_grad = pyscf_grad.kernel()
    pyscf_mm_grad = (
        pyscf_grad.grad_hcore_mm(pyscf_mf.make_rdm1())
        + pyscf_grad.grad_nuc_mm()
    )
    return pyscf_mf.e_tot, -pyscf_qm_grad, -pyscf_mm_grad


if __name__ == "__main__":
    main()
