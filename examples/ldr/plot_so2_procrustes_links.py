#!/usr/bin/env python3
"""Plot slices of the SO2 forward links in a Procrustes gauge."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import ultraplot as uplt

from pyqed.ldr.overlap import procrustes, unpack


DEFAULT_REFERENCE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_full_ldr_9x9x9_20fs/"
    "electronic_reference.npz"
)
DEFAULT_GAUGE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_procrustes_two_patch_9x9x9/"
    "procrustes_gauge.npz"
)


def aligned_link_fields(shape, links, gauge):
    """Return one aligned forward-link field for each grid axis."""
    gauge = np.asarray(gauge, dtype=complex).reshape(*shape, *gauge.shape[-2:])
    fields = []
    for axis, size in enumerate(shape):
        field_shape = list(shape)
        field_shape[axis] -= 1
        field = np.empty((*field_shape, *gauge.shape[-2:]), dtype=complex)
        for index in np.ndindex(tuple(field_shape)):
            neighbor = list(index)
            neighbor[axis] += 1
            neighbor = tuple(neighbor)
            field[index] = (
                gauge[index].conj().T
                @ links[(axis, index)]
                @ gauge[neighbor]
            )
        fields.append(field)
    return tuple(fields)


def polar_diagnostics(fields):
    rotations = []
    positive_losses = []
    identity = np.eye(fields[0].shape[-1])
    for field in fields:
        unitary, positive, _singular = procrustes(field)
        rotations.append(np.linalg.norm(unitary - identity, axis=(-2, -1)))
        positive_losses.append(np.linalg.norm(positive - identity, axis=(-2, -1)))
    return tuple(rotations), tuple(positive_losses)


def style():
    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.35,
            "savefig.transparent": False,
        }
    )


def save(figure, output_dir, stem):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(png)
    print(pdf)


def plot_norm_slices(
    output_dir,
    grids,
    center,
    patch_boundary,
    rotations,
    positive_losses,
):
    qs, theta_rad, qa = grids
    theta = np.rad2deg(theta_rad)
    qs_mid = 0.5 * (qs[:-1] + qs[1:])
    theta_mid = 0.5 * (theta[:-1] + theta[1:])
    qa_mid = 0.5 * (qa[:-1] + qa[1:])
    qa_link = min(center[2], len(qa_mid) - 1)
    fields = (
        (theta, qs_mid, rotations[0][:, :, center[2]]),
        (theta_mid, qs, rotations[1][:, :, center[2]]),
        (theta, qs, rotations[2][:, :, qa_link]),
        (theta, qs_mid, positive_losses[0][:, :, center[2]]),
        (theta_mid, qs, positive_losses[1][:, :, center[2]]),
        (theta, qs, positive_losses[2][:, :, qa_link]),
    )
    rotation_max = max(float(np.max(field[2])) for field in fields[:3])
    positive_max = max(float(np.max(field[2])) for field in fields[3:])
    rotation_norm = LogNorm(vmin=1.0e-5, vmax=max(rotation_max, 1.0e-4))
    positive_norm = Normalize(vmin=0.0, vmax=max(positive_max, 1.0e-4))
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=3,
        width=8.2,
        height=4.75,
        share=False,
        wspace=1.2,
        hspace=2.0,
    )
    maps = []
    titles = (
        r"$q_s$ links",
        r"$\theta$ links",
        r"$q_a$ links",
    )
    for panel, (axis, (x, y, values)) in enumerate(zip(axes, fields)):
        maps.append(
            axis.pcolormesh(
                x,
                y,
                np.maximum(values, 1.0e-16),
                cmap="viridis" if panel < 3 else "magma",
                norm=rotation_norm if panel < 3 else positive_norm,
                shading="nearest",
            )
        )
        axis.format(
            title=titles[panel] if panel < 3 else "",
            xlabel=r"$\theta$ (deg)" if panel >= 3 else "",
            ylabel=r"$q_s$ (bohr)" if panel % 3 == 0 else "",
            tickdir="out",
            grid=False,
        )
        axis.tick_params(
            labelbottom=panel >= 3,
            labelleft=panel % 3 == 0,
        )
        if 0 <= patch_boundary < len(theta) - 1:
            interface = 0.5 * (
                theta[patch_boundary] + theta[patch_boundary + 1]
            )
            axis.axvline(
                interface,
                color="white",
                linestyle="--",
                linewidth=0.7,
                alpha=0.9,
            )
        axis.text(
            0.025,
            0.965,
            "abcdef"[panel],
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
    figure.colorbar(
        maps[2],
        loc="r",
        rows=(1,),
        label=r"$\|V_\mu-I\|_F$",
    )
    figure.colorbar(
        maps[5],
        loc="r",
        rows=(2,),
        label=r"$\|P_\mu-I\|_F$",
    )
    save(figure, output_dir, "so2_procrustes_link_norm_slices")


def plot_center_lines(output_dir, grids, center, patch_boundary, fields):
    qs, theta_rad, qa = grids
    coordinates = (
        0.5 * (qs[:-1] + qs[1:]),
        np.rad2deg(0.5 * (theta_rad[:-1] + theta_rad[1:])),
        0.5 * (qa[:-1] + qa[1:]),
    )
    cuts = (
        fields[0][:, center[1], center[2]],
        fields[1][center[0], :, center[2]],
        fields[2][center[0], center[1], :],
    )
    imaginary_max = max(float(np.max(np.abs(cut.imag))) for cut in cuts)
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=3,
        width=9.0,
        height=4.45,
        share=False,
        wspace=2.8,
        hspace=2.0,
    )
    titles = (r"$q_s$ links", r"$\theta$ links", r"$q_a$ links")
    xlabels = (r"$q_s$ midpoint (bohr)", r"$\theta$ midpoint (deg)", r"$q_a$ midpoint (bohr)")
    for column, (coordinate, cut) in enumerate(zip(coordinates, cuts)):
        for state, color in enumerate(colors):
            axes[column].plot(
                coordinate,
                cut[:, state, state].real,
                "o-",
                color=color,
                ms=3.2,
                label=rf"$\bar S_{{{state}{state}}}$",
            )
        for (left, right), color in zip(((0, 1), (0, 2), (1, 2)), colors):
            axes[column + 3].plot(
                coordinate,
                cut[:, left, right].real,
                "o-",
                color=color,
                ms=3.2,
                label=rf"$\bar S_{{{left}{right}}}$",
            )
        axes[column].axhline(1.0, color="0.72", linewidth=0.6, zorder=0)
        axes[column + 3].axhline(0.0, color="0.72", linewidth=0.6, zorder=0)
        axes[column].format(title=titles[column], grid=False, tickdir="out")
        axes[column + 3].format(xlabel=xlabels[column], grid=False, tickdir="out")
        axes[column].tick_params(labelbottom=False)
    if 0 <= patch_boundary < len(theta_rad) - 1:
        interface = np.rad2deg(
            0.5 * (theta_rad[patch_boundary] + theta_rad[patch_boundary + 1])
        )
        for axis in (axes[1], axes[4]):
            axis.axvline(interface, color="0.35", linestyle="--", linewidth=0.7)
    axes[0].format(ylabel=r"Diagonal $\Re\,\bar S_{\alpha\alpha}$")
    axes[3].format(ylabel=r"Off-diagonal $\Re\,\bar S_{\alpha\beta}$")
    axes[2].legend(frame=False, loc="best")
    axes[5].legend(frame=False, loc="best")
    for panel, axis in enumerate(axes):
        axis.text(
            0.025,
            0.965,
            "abcdef"[panel],
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )
    save(figure, output_dir, "so2_procrustes_link_center_lines")
    print(f"maximum plotted imaginary component = {imaginary_max:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gauge", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GAUGE.parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.reference) as archive:
        grids = tuple(np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
        shape = tuple(int(value) for value in archive["energies"].shape[:-1])
    with np.load(args.gauge) as archive:
        gauge = np.asarray(archive["gauge"], dtype=complex)
        center = tuple(int(value) for value in archive["center"])
        patch_boundary = int(archive["patch_boundary_theta_index"])
    style()
    fields = aligned_link_fields(shape, links, gauge)
    rotations, positive_losses = polar_diagnostics(fields)
    plot_norm_slices(
        args.output_dir,
        grids,
        center,
        patch_boundary,
        rotations,
        positive_losses,
    )
    plot_center_lines(
        args.output_dir,
        grids,
        center,
        patch_boundary,
        fields,
    )


if __name__ == "__main__":
    main()
