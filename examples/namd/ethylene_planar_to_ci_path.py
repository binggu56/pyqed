#!/usr/bin/env python3
"""Optimize planar ethylene and scan a connecting path to the CI geometry.

The planar endpoint is optimized with state-specific CASSCF(2,2)/6-31G*.
Two equally weighted singlet roots are then evaluated along an aligned
bond-vector geodesic interpolation to the CI-centered geometry used by the 2D
ethylene benchmark.  This is a connecting path, not a constrained
minimum-energy path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr import ethylene_ci_geometry
from pyqed.units import au2angstrom, au2ev


SPECIES = ("C", "H", "H", "C", "H", "H")


def planar_guess():
    cc = 1.335
    ch = 1.087
    hcc = np.deg2rad(121.3)
    radial = ch * np.sin(hcc)
    axial = -ch * np.cos(hcc)
    return np.asarray(
        (
            (0.0, 0.0, 0.5 * cc),
            (radial, 0.0, 0.5 * cc + axial),
            (-radial, 0.0, 0.5 * cc + axial),
            (0.0, 0.0, -0.5 * cc),
            (radial, 0.0, -0.5 * cc - axial),
            (-radial, 0.0, -0.5 * cc - axial),
        ),
        dtype=float,
    )


def molecule(geometry, basis, verbose=0):
    from pyscf import gto

    return gto.M(
        atom=list(zip(SPECIES, np.asarray(geometry, dtype=float))),
        unit="angstrom",
        basis=str(basis),
        charge=0,
        spin=0,
        symmetry=False,
        verbose=int(verbose),
    )


def casscf(mol, *, state_average, mo=None, verbose=0):
    from pyscf import fci, mcscf, scf

    mean_field = scf.RHF(mol)
    mean_field.conv_tol = 1.0e-10
    mean_field.max_cycle = 100
    mean_field.verbose = int(verbose)
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("RHF did not converge")
    active = mcscf.CASSCF(mean_field, 2, 2)
    active.fcisolver = fci.direct_spin0.FCI(mol)
    active.conv_tol = 1.0e-9
    active.max_cycle_macro = 80
    active.verbose = int(verbose)
    if state_average:
        active = active.state_average_((0.5, 0.5))
    active.kernel(mo)
    if not active.converged:
        raise RuntimeError("CASSCF did not converge")
    energies = np.atleast_1d(active.e_states if state_average else active.e_tot)
    return active, np.asarray(energies, dtype=float)


def optimize_planar(guess, basis, maxsteps, verbose):
    from pyscf.geomopt.geometric_solver import optimize

    active, _energies = casscf(
        molecule(guess, basis, verbose), state_average=False, verbose=verbose
    )
    optimized = optimize(active, maxsteps=int(maxsteps), verbose=int(verbose))
    return np.asarray(optimized.atom_coords(unit="angstrom"), dtype=float)


def align(source, target, atoms=(0, 1, 2, 3)):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    selected = np.asarray(atoms, dtype=int)
    source_center = np.mean(source[selected], axis=0)
    target_center = np.mean(target[selected], axis=0)
    covariance = (source[selected] - source_center).T @ (
        target[selected] - target_center
    )
    left, _singular, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return (source - source_center) @ rotation + target_center


def spherical_interpolation(left, right, coordinate):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_length = np.linalg.norm(left)
    right_length = np.linalg.norm(right)
    left_direction = left / left_length
    right_direction = right / right_length
    cosine = float(np.clip(left_direction @ right_direction, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    coordinate = np.asarray(coordinate, dtype=float)
    if angle < 1.0e-10:
        direction = np.broadcast_to(left_direction, (len(coordinate), 3)).copy()
    else:
        direction = (
            np.sin((1.0 - coordinate) * angle)[:, None] * left_direction
            + np.sin(coordinate * angle)[:, None] * right_direction
        ) / np.sin(angle)
    length = (1.0 - coordinate) * left_length + coordinate * right_length
    return length[:, None] * direction


def internal_path(planar, ci, coordinate):
    """Interpolate C--C positions and rotate each C--H bond geodesically."""

    planar = np.asarray(planar, dtype=float)
    ci = np.asarray(ci, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    geometries = np.empty((len(coordinate), len(planar), 3), dtype=float)
    for carbon in (0, 3):
        geometries[:, carbon] = (
            (1.0 - coordinate[:, None]) * planar[carbon]
            + coordinate[:, None] * ci[carbon]
        )
    for hydrogen, carbon in ((1, 0), (2, 0), (4, 3), (5, 3)):
        bond = spherical_interpolation(
            planar[hydrogen] - planar[carbon],
            ci[hydrogen] - ci[carbon],
            coordinate,
        )
        geometries[:, hydrogen] = geometries[:, carbon] + bond
    return geometries


def scan(geometries, basis, verbose):
    from pyqed.ldr import EthyleneCIElectronicDriver
    from pyqed.qchem import Molecule

    driver = EthyleneCIElectronicDriver(
        basis=basis, method="sa-casscf", nroots=2, verbose=verbose
    )
    energies = []
    for number, geometry in enumerate(geometries):
        mol = Molecule(
            atom=list(zip(SPECIES, np.asarray(geometry, dtype=float))),
            unit="angstrom",
            basis=str(basis),
            charge=0,
            spin=0,
        ).build(eri="dense")
        roots = np.asarray(driver._solve(mol, 2).e_tot, dtype=float)
        energies.append(roots)
        print(
            f"path point {number + 1}/{len(geometries)}: "
            f"E0={roots[0]:.10f} E1={roots[1]:.10f}",
            flush=True,
        )
    return np.asarray(energies)


def plot(path_coordinate, energies, output):
    relative = (energies - energies[0, 0]) * au2ev
    gap = (energies[:, 1] - energies[:, 0]) * au2ev
    figure, axis = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    axis.plot(path_coordinate, relative[:, 0], "o-", label=r"$S_0$")
    axis.plot(path_coordinate, relative[:, 1], "o-", label=r"$S_1$")
    axis.set_xlabel(r"path coordinate $s$: planar $S_0\rightarrow$ CI")
    axis.set_ylabel(r"$E-E_{S_0}(s=0)$ / eV")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    gap_axis = axis.twinx()
    gap_axis.plot(path_coordinate, gap, "k--", linewidth=1.2, label="gap")
    gap_axis.set_ylabel(r"$S_1-S_0$ gap / eV")
    axis.set_title(r"C$_2$H$_4$ planar minimum to CI connecting path")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--maxsteps", type=int, default=40)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home()
        / "Library"
        / "CloudStorage"
        / "OneDrive-西湖大学"
        / "data"
        / "pyqed"
        / "ethylene_planar_to_ci",
    )
    args = parser.parse_args()
    if args.points < 3:
        raise ValueError("points must be at least three")

    started = perf_counter()
    planar = optimize_planar(
        planar_guess(), args.basis, args.maxsteps, args.verbose
    )
    ci = np.asarray(ethylene_ci_geometry((0.0, 0.0))) * au2angstrom
    planar = align(planar, ci)
    coordinate = np.linspace(0.0, 1.0, args.points)
    geometries = internal_path(planar, ci, coordinate)
    energies = scan(geometries, args.basis, args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "ethylene_planar_to_ci.npz"
    np.savez_compressed(
        archive,
        path_coordinate=coordinate,
        geometries_angstrom=geometries,
        energies_hartree=energies,
        planar_geometry_angstrom=planar,
        ci_geometry_angstrom=ci,
    )
    figure = args.output_dir / "ethylene_planar_to_ci_pes.png"
    plot(coordinate, energies, figure)
    summary = {
        "system": "C2H4",
        "planar_optimization": "state-specific CASSCF(2,2)/6-31G*",
        "path_electronic_structure": (
            "PyQED ethylene benchmark SA(2)-CASSCF(2,2)/6-31G* driver"
        ),
        "path": (
            "aligned bond-vector geodesic interpolation; "
            "not a minimum-energy path"
        ),
        "points": int(args.points),
        "basis": args.basis,
        "initial_gap_ev": float((energies[0, 1] - energies[0, 0]) * au2ev),
        "final_gap_ev": float((energies[-1, 1] - energies[-1, 0]) * au2ev),
        "ground_state_rise_ev": float((energies[-1, 0] - energies[0, 0]) * au2ev),
        "seconds": perf_counter() - started,
        "archive": str(archive),
        "figure": str(figure),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
