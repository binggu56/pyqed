#!/usr/bin/env python3
"""Plot the SO2 electronic potential matrix in the Procrustes gauge."""

from __future__ import annotations
from pyqed.units import au2ev

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt


DEFAULT_REFERENCE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_full_ldr_9x9x9_20fs/"
    "electronic_reference.npz"
)
DEFAULT_GAUGE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_procrustes_gauge_9x9x9/"
    "procrustes_gauge.npz"
)
HARTREE_TO_EV = au2ev


def aligned_energy(energies, gauge):
    """Return ``U(R)^dagger diag(E(R)) U(R)`` relative to the global minimum."""
    energies = np.asarray(energies, dtype=float)
    gauge = np.asarray(gauge, dtype=complex)
    shifted = energies - float(np.min(energies))
    diagonal = np.zeros((*energies.shape[:-1], energies.shape[-1], energies.shape[-1]))
    states = np.arange(energies.shape[-1])
    diagonal[..., states, states] = shifted
    aligned = np.einsum(
        "...ia,...ij,...jb->...ab",
        gauge.conj(),
        diagonal,
        gauge,
        optimize=True,
    )
    return aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gauge", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GAUGE.parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.reference) as archive:
        energies = np.asarray(archive["energies"], dtype=float)
        qs = np.asarray(archive["qs"], dtype=float)
        theta = np.rad2deg(np.asarray(archive["theta"], dtype=float))
        qa = np.asarray(archive["qa"], dtype=float)
    with np.load(args.gauge) as archive:
        gauge = np.asarray(archive["gauge"], dtype=complex)
        center = tuple(int(value) for value in archive["center"])
        patch_boundary = int(
            archive["patch_boundary_theta_index"]
            if "patch_boundary_theta_index" in archive
            else -1
        )
    matrix = aligned_energy(energies, gauge)
    imaginary_max = float(np.max(np.abs(matrix.imag)))
    if imaginary_max > 1.0e-10:
        raise ValueError(f"Aligned energy has imaginary component {imaginary_max:.3e}")
    matrix = matrix.real * HARTREE_TO_EV
    qa_index = int(np.argmin(np.abs(qa)))
    plane = matrix[:, :, qa_index]

    diagonal_scale = float(np.max(plane[..., np.arange(3), np.arange(3)]))
    coupling_scale = max(
        float(np.max(np.abs(plane[..., left, right])))
        for left in range(3)
        for right in range(left + 1, 3)
    )
    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.transparent": False,
        }
    )
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=3,
        width=8.2,
        height=4.75,
        share=True,
        wspace=1.1,
        hspace=2.0,
    )
    diagonal_maps = []
    coupling_maps = []
    for state, axis in enumerate(axes[:3]):
        diagonal_maps.append(
            axis.pcolormesh(
                theta,
                qs,
                plane[..., state, state],
                cmap="viridis",
                vmin=0.0,
                vmax=diagonal_scale,
                shading="nearest",
            )
        )
        axis.format(title=rf"$\bar{{E}}_{{{state}{state}}}$")
    for axis, (left, right) in zip(axes[3:], ((0, 1), (0, 2), (1, 2))):
        coupling_maps.append(
            axis.pcolormesh(
                theta,
                qs,
                plane[..., left, right],
                cmap="ColdHot",
                vmin=-coupling_scale,
                vmax=coupling_scale,
                shading="nearest",
            )
        )
        axis.format(title=rf"$\bar{{E}}_{{{left}{right}}}$")
    for axis in axes:
        axis.plot(theta[center[1]], qs[center[0]], "wo", ms=3.5, mec="black", mew=0.5)
        if 0 <= patch_boundary < len(theta) - 1:
            interface = 0.5 * (
                theta[patch_boundary] + theta[patch_boundary + 1]
            )
            axis.axvline(
                interface,
                color="0.35",
                linestyle="--",
                linewidth=0.65,
                alpha=0.8,
            )
        axis.format(
            xlim=(theta[0], theta[-1]),
            ylim=(qs[0], qs[-1]),
            tickdir="out",
            grid=False,
        )
    axes[0].format(ylabel=r"$q_s$ (bohr)")
    axes[3].format(ylabel=r"$q_s$ (bohr)")
    for axis in axes[3:]:
        axis.format(xlabel=r"$\theta$ (deg)")
    figure.colorbar(
        diagonal_maps[-1],
        loc="r",
        rows=(1,),
        label=r"$\bar E_{\alpha\alpha}$ (eV)",
    )
    figure.colorbar(
        coupling_maps[-1],
        loc="r",
        rows=(2,),
        label=r"$\bar E_{\alpha\beta}$ (eV)",
    )
    for label, axis in zip("abcdef", axes):
        axis.text(
            0.025,
            0.965,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.5,
            },
        )
    stem = args.output_dir / "so2_procrustes_energy_qa0"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"q_a slice = {qa[qa_index]:.8f} bohr")
    print(f"max imaginary component = {imaginary_max:.3e} Eh")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
