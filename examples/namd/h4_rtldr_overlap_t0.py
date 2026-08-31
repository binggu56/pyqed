#!/usr/bin/env python3
"""Compute and plot the H4 RT-LDR determinant overlap matrix at t = 0."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR
from pyqed.namd.rtldr.gdvr import GDVRFrame, Solver
from pyqed.qchem.gdvr import AtomicChain


MASS_H = 1836.15267343
Q_MIN = -2.0
Q_MAX = 2.0
NPOINTS = 3
LZ = 15.0
NZ = 50
M = 1
TRANSVERSE_BASIS = "631g"


def atomic_positions(q):
    q1, q2 = q
    sqrt2 = np.sqrt(2.0)
    return np.array(
        [-3.6, -1.2 + (q1 + q2) / sqrt2, 1.2 + (q1 - q2) / sqrt2, 3.6]
    )


def build_frame(q):
    z = atomic_positions(q)
    mol = AtomicChain(
        elements=["H"] * 4,
        coords=[[0.0, 0.0, value] for value in z],
    )
    mol.build(
        Lz=LZ,
        Nz=NZ,
        M=M,
        transverse_basis=TRANSVERSE_BASIS,
        dvr_method="sine",
        verbose=False,
    )
    mf = mol.RHF().run(
        conv=1.0e-8,
        max_iter=100,
        newton=False,
        verbose=False,
    )
    mf.newton(
        max_cycles=90,
        sweeps=1,
        tol=1.0e-6,
        ridge=0.5,
        trust_step=0.5,
        trust_radius=1.0,
        verbose=False,
    )
    return GDVRFrame(mf)


def plot_overlap(overlap, output):
    components = (
        (overlap.real, r"$\mathrm{Re}\,S_{ij}(0)$", "RdBu_r", -1.0, 1.0),
        (overlap.imag, r"$\mathrm{Im}\,S_{ij}(0)$", "RdBu_r", -1.0, 1.0),
        (np.abs(overlap), r"$|S_{ij}(0)|$", "viridis", 0.0, 1.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, components):
        image = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Nuclear grid index $j$")
        ax.set_ylabel("Nuclear grid index $i$")
        ax.set_xticks(range(overlap.shape[0]))
        ax.set_yticks(range(overlap.shape[0]))
        fig.colorbar(image, ax=ax, shrink=0.86)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def main():
    nuclear = DVR(
        domains=[(Q_MIN, Q_MAX)] * 2,
        npts=[NPOINTS] * 2,
        mass=MASS_H,
        names=("q_in_phase", "q_out_of_phase"),
    )
    frames = []
    for index, q in enumerate(nuclear.points):
        print(f"Building geometry {index + 1}/{len(nuclear.points)}: q={q}", flush=True)
        frames.append(build_frame(q))

    overlap = Solver(nuclear=nuclear, electronic=frames).overlap_matrix()
    data_output = Path("h4_overlap_t0.npz")
    plot_output = Path("h4_overlap_t0.png")
    np.savez_compressed(data_output, overlap=overlap, collective_points=nuclear.points)
    plot_overlap(overlap, plot_output)

    print(f"Saved {data_output}")
    print(f"Saved {plot_output}")
    print(f"max |Im S| = {np.max(np.abs(overlap.imag)):.6e}")
    print(
        "max Hermiticity error = "
        f"{np.max(np.abs(overlap - overlap.conj().T)):.6e}"
    )
    print(overlap)


if __name__ == "__main__":
    main()
