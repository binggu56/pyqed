#!/usr/bin/env python3
"""Generate a reproducible SO2 AM1/MECI energy-and-overlap fitting set."""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2ev
from pyqed.qchem import Molecule

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="AM1 model is under testing")
    from pyqed.qchem.semiempirical.am1 import RAM1


def geometry(r1, r2, theta):
    return np.asarray(
        [
            [r1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [r2 * np.cos(theta), r2 * np.sin(theta), 0.0],
        ]
    )


def electronic_structure(r1, r2, theta, args):
    xyz = geometry(float(r1), float(r2), float(theta))
    atom = [
        [symbol, tuple(position)]
        for symbol, position in zip(("O", "S", "O"), xyz)
    ]
    reference = RAM1(
        Molecule(atom=atom, charge=0, spin=0, unit="bohr")
    ).run(
        conv_tol=args.scf_tol,
        max_cycle=args.max_cycle,
        damping=args.damping,
        verbose=0,
    )
    return reference.MECI(nstates=args.nstates, ncas=args.ncas).run()


def generate(args):
    r1 = np.linspace(args.r_min, args.r_max, args.n_r)
    r2 = np.linspace(args.r_min, args.r_max, args.n_r)
    theta = np.deg2rad(
        np.linspace(args.theta_min_deg, args.theta_max_deg, args.n_theta)
    )
    shape = (len(r1), len(r2), len(theta))
    models = np.empty(shape, dtype=object)
    energies = np.empty((*shape, args.nstates))
    started = time.perf_counter()
    total = int(np.prod(shape))
    for count, index in enumerate(np.ndindex(shape), start=1):
        models[index] = electronic_structure(
            r1[index[0]], r2[index[1]], theta[index[2]], args
        )
        energies[index] = np.asarray(models[index].e_tot)[: args.nstates]
        if count == 1 or count % args.progress_every == 0 or count == total:
            print(
                f"[electronic] {count}/{total}, E0={energies[index][0]:.10f} Eh, "
                f"elapsed={time.perf_counter() - started:.1f} s",
                flush=True,
            )

    links = []
    for axis in range(3):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        values = np.empty((*edge_shape, args.nstates, args.nstates), dtype=complex)
        for left in np.ndindex(tuple(edge_shape)):
            right = list(left)
            right[axis] += 1
            values[left] = models[left].wavefunction_overlap(models[tuple(right)])
        links.append(values)
    return (r1, r2, theta), energies, tuple(links)


def plot_dataset(grids, energies, links, filename):
    r1, _r2, theta = grids
    center = tuple(len(grid) // 2 for grid in grids)
    relative = (energies - energies[..., :1].min()) * au2ev
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), constrained_layout=True)
    for state in range(energies.shape[-1]):
        axes[0].plot(
            r1,
            relative[:, center[1], center[2], state],
            marker="o",
            label=f"S{state}",
        )
        axes[1].plot(
            np.rad2deg(theta),
            relative[center[0], center[1], :, state],
            marker="o",
        )
    axes[0].set(xlabel=r"$r_1$ (bohr)", ylabel="Relative energy (eV)")
    axes[1].set(xlabel=r"$\theta$ (degree)", ylabel="Relative energy (eV)")
    axes[0].legend(frameon=False)
    for axis, values in enumerate(links):
        singular_values = np.linalg.svd(values, compute_uv=False)
        axes[2].hist(
            singular_values.ravel(), bins=24, histtype="step", label=f"axis {axis}"
        )
    axes[2].set(xlabel="Link singular value", ylabel="Count")
    axes[2].legend(frameon=False)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r", type=int, default=5)
    parser.add_argument("--n-theta", type=int, default=5)
    parser.add_argument("--r-min", type=float, default=2.68)
    parser.add_argument("--r-max", type=float, default=2.92)
    parser.add_argument("--theta-min-deg", type=float, default=110.0)
    parser.add_argument("--theta-max-deg", type=float, default=130.0)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--scf-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("/private/tmp/so2_am1_meci_5x5x5.npz")
    )
    args = parser.parse_args()
    grids, energies, links = generate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        r1=grids[0],
        r2=grids[1],
        theta=grids[2],
        energies=energies,
        links_0=links[0],
        links_1=links[1],
        links_2=links[2],
        source=np.asarray("SO2 native AM1/MECI ncas=4 dense fitting grid"),
    )
    figure = args.output.with_suffix(".png")
    plot_dataset(grids, energies, links, figure)
    print(f"dataset: {args.output}")
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
