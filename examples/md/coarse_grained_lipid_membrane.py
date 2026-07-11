#!/usr/bin/env python3
"""Small coarse-grained lipid bilayer MD demo.

This is a development-scale membrane calculation, not a validated lipid force
field.  Each lipid is represented by one weakly interacting head bead and
three more attractive tail beads.  The initial geometry is a preassembled
bilayer; short Langevin dynamics probes whether the MD stack can maintain a
membrane-like slab and produce analyzable output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.md import (
    Atoms,
    EnergyLogger,
    Langevin,
    MolecularMechanics,
    XYZTrajectoryWriter,
    set_maxwell_boltzmann_velocities,
    soft_relaxation,
    write_minimization_log,
)
from pyqed.units import au2angstrom, au2fs, fs, kcalmol2au


def angstrom(value):
    return float(value) / au2angstrom


def kcal_per_mol(value):
    return float(value) * kcalmol2au


def kcal_per_mol_angstrom2(value):
    return float(value) * kcalmol2au * au2angstrom**2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=4, help="Lipids along x per leaflet.")
    parser.add_argument("--ny", type=int, default=4, help="Lipids along y per leaflet.")
    parser.add_argument("--steps", type=int, default=200, help="Langevin MD steps.")
    parser.add_argument("--timestep-fs", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=250.0)
    parser.add_argument("--friction", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", default="/private/tmp/pyqed_cg_lipid_membrane")
    parser.add_argument("--trajectory-interval", type=int, default=10)
    parser.add_argument("--energy-interval", type=int, default=5)
    parser.add_argument("--skip-relax", action="store_true")
    return parser.parse_args()


def build_membrane(nx=4, ny=4):
    spacing = angstrom(7.5)
    box = np.array([nx * spacing, ny * spacing, angstrom(40.0)])
    bond = angstrom(3.8)
    head_z = angstrom(10.5)

    atoms = []
    bonds = []
    angles = []
    epsilon = []
    sigma = []
    lipid_ids = []
    bead_types = []

    head_epsilon = kcal_per_mol(0.04)
    tail_epsilon = kcal_per_mol(0.35)
    head_sigma = angstrom(4.6)
    tail_sigma = angstrom(4.1)
    bond_k = kcal_per_mol_angstrom2(8.0)

    lipid = 0
    for leaflet_sign in (1.0, -1.0):
        for ix in range(nx):
            for iy in range(ny):
                x = (ix + 0.5) * spacing
                y = (iy + 0.5) * spacing
                local = []
                for bead in range(4):
                    z = leaflet_sign * (head_z - bead * bond)
                    local.append((x, y, z))

                offset = len(atoms)
                atoms.extend(
                    [
                        ["O", local[0]],
                        ["C", local[1]],
                        ["C", local[2]],
                        ["C", local[3]],
                    ]
                )
                bonds.extend(
                    [
                        (offset, offset + 1, bond_k, bond),
                        (offset + 1, offset + 2, bond_k, bond),
                        (offset + 2, offset + 3, bond_k, bond),
                    ]
                )
                epsilon.extend([head_epsilon, tail_epsilon, tail_epsilon, tail_epsilon])
                sigma.extend([head_sigma, tail_sigma, tail_sigma, tail_sigma])
                bead_types.extend([0, 1, 1, 1])
                lipid_ids.extend([lipid] * 4)
                lipid += 1

    calc = MolecularMechanics(
        bonds=bonds,
        angles=angles,
        angle_unit="degree",
        lj_epsilon=epsilon,
        lj_sigma=sigma,
        lj_cutoff=angstrom(10.0),
        lj_energy_shift=True,
        exclude_bonded=True,
        exclude_angles=True,
        nonbonded_skin=angstrom(1.0),
    )
    membrane = Atoms(atoms, cell=box, pbc=True, calculator=calc)
    membrane.set_array("lipid_ids", lipid_ids, int, ())
    membrane.set_array("bead_types", bead_types, int, ())
    return membrane


def analyze_membrane(atoms):
    positions = atoms.get_positions()
    bead_types = atoms.get_array("bead_types")
    lipid_ids = atoms.get_array("lipid_ids")
    head_positions = positions[bead_types == 0]
    tail_positions = positions[bead_types == 1]

    head_z = head_positions[:, 2]
    upper_heads = head_z[head_z > 0.0]
    lower_heads = head_z[head_z < 0.0]
    box_lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    area_per_lipid = box_lengths[0] * box_lengths[1] / (0.5 * len(np.unique(lipid_ids)))

    order_values = []
    for lipid in np.unique(lipid_ids):
        lipid_positions = positions[lipid_ids == lipid]
        for start in range(1, 3):
            vector = lipid_positions[start + 1] - lipid_positions[start]
            norm = np.linalg.norm(vector)
            if norm > 0.0:
                cos_theta = abs(vector[2]) / norm
                order_values.append(0.5 * (3.0 * cos_theta * cos_theta - 1.0))

    return {
        "lipids": int(len(np.unique(lipid_ids))),
        "atoms": len(atoms),
        "area_per_lipid_angstrom2": area_per_lipid * au2angstrom**2,
        "upper_head_z_angstrom": float(np.mean(upper_heads) * au2angstrom),
        "lower_head_z_angstrom": float(np.mean(lower_heads) * au2angstrom),
        "head_head_thickness_angstrom": float((np.mean(upper_heads) - np.mean(lower_heads)) * au2angstrom),
        "tail_core_half_width_angstrom": float(np.mean(np.abs(tail_positions[:, 2])) * au2angstrom),
        "tail_order_p2": float(np.mean(order_values)),
        "temperature_K": float(atoms.get_temperature(remove_center_of_mass=True)),
        "potential_hartree": float(atoms.get_potential_energy()),
        "kinetic_hartree": float(atoms.get_kinetic_energy()),
    }


def write_summary(path, metrics, dynamics=None, trajectory_path=None, energy_path=None, image_path=None):
    with open(path, "w") as handle:
        if dynamics is not None:
            handle.write(f"steps: {dynamics.get_number_of_steps()}\n")
            handle.write(f"time_fs: {dynamics.get_time() * au2fs:.6f}\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                handle.write(f"{key}: {value:.8f}\n")
            else:
                handle.write(f"{key}: {value}\n")
        if trajectory_path is not None:
            handle.write(f"trajectory: {trajectory_path}\n")
        if energy_path is not None:
            handle.write(f"energy_log: {energy_path}\n")
        if image_path is not None:
            handle.write(f"image: {image_path}\n")


def render_membrane(atoms, path):
    positions = atoms.get_positions() * au2angstrom
    bead_types = atoms.get_array("bead_types")
    lipid_ids = atoms.get_array("lipid_ids")

    fig = plt.figure(figsize=(8, 6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    head = bead_types == 0
    tail = bead_types == 1
    ax.scatter(
        positions[tail, 0],
        positions[tail, 1],
        positions[tail, 2],
        s=16,
        c="#2f6f4f",
        alpha=0.82,
        depthshade=True,
        label="tail beads",
    )
    ax.scatter(
        positions[head, 0],
        positions[head, 1],
        positions[head, 2],
        s=36,
        c="#d94f45",
        edgecolors="#4a1512",
        linewidths=0.35,
        alpha=0.95,
        depthshade=True,
        label="head beads",
    )
    for lipid in np.unique(lipid_ids):
        xyz = positions[lipid_ids == lipid]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#363636", linewidth=0.55, alpha=0.45)

    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float) * au2angstrom
    ax.set_xlim(0, lengths[0])
    ax.set_ylim(0, lengths[1])
    ax.set_zlim(-0.5 * lengths[2], 0.5 * lengths[2])
    ax.set_xlabel("x / A")
    ax.set_ylabel("y / A")
    ax.set_zlabel("z / A")
    ax.view_init(elev=22, azim=-52)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = build_membrane(args.nx, args.ny)
    if not args.skip_relax:
        relaxation = soft_relaxation(
            atoms,
            stages=((0.2, 0.2, 20), (0.6, 0.6, 20), (1.0, 1.0, 20)),
            max_step=angstrom(0.03),
            fmax=2e-4,
        )
        write_minimization_log(output_dir / "cg_lipid_membrane_relax.dat", relaxation)

    set_maxwell_boltzmann_velocities(atoms, args.temperature, seed=args.seed)
    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * fs,
        temperature_K=args.temperature,
        friction=args.friction,
    )
    trajectory_path = output_dir / "cg_lipid_membrane.xyz"
    energy_path = output_dir / "cg_lipid_membrane_energy.dat"
    image_path = output_dir / "cg_lipid_membrane_final.png"
    summary_path = output_dir / "summary.txt"
    writer = XYZTrajectoryWriter(atoms, trajectory_path, dynamics=dynamics)
    logger = EnergyLogger(atoms, energy_path, dynamics=dynamics)
    dynamics.attach(writer, interval=args.trajectory_interval)
    dynamics.attach(logger, interval=args.energy_interval)
    try:
        dynamics.run(args.steps)
    finally:
        writer.close()
        logger.close()

    metrics = analyze_membrane(atoms)
    render_membrane(atoms, image_path)
    write_summary(
        summary_path,
        metrics,
        dynamics=dynamics,
        trajectory_path=trajectory_path,
        energy_path=energy_path,
        image_path=image_path,
    )
    print(summary_path)
    print(summary_path.read_text(), end="")


if __name__ == "__main__":
    main()
