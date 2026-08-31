#!/usr/bin/env python3
"""Diagnose a high-angle third Procrustes patch for the SO2 grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.so2_sparse_link_benchmark import rotate_links
from examples.ldr.so2_procrustes_gauge import stitch_upper
from pyqed.ldr.overlap import procrustes
from pyqed.ldr.ttfit import LinkPath


def raw_links(aligned, gauge):
    output = []
    for axis, values in enumerate(aligned):
        blocks = np.empty_like(values)
        for left in np.ndindex(values.shape[:-2]):
            right = list(left)
            right[axis] += 1
            right = tuple(right)
            blocks[left] = gauge[left] @ values[left] @ gauge[right].conj().T
        output.append(blocks)
    return tuple(output)


def path_patch(links, gauge, anchor):
    shape = gauge.shape[:-2]
    nstates = gauge.shape[-1]
    blocks = {
        (axis, index): values[index]
        for axis, values in enumerate(links)
        for index in np.ndindex(values.shape[:-2])
    }
    path = LinkPath(shape, nstates, blocks, order=(1, 0, 2))
    overlaps = np.empty_like(gauge)
    for index in np.ndindex(shape):
        overlaps[index] = path.between(index, anchor)
    rotation, positive, singular = procrustes(overlaps)
    return gauge @ rotation, positive, singular


def central_theta(values, center):
    return values[center[0], :, center[2]]


def step_norm(values, center):
    return np.linalg.norm(np.diff(central_theta(values, center), axis=0), axis=(1, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fields",
        type=Path,
        default=Path("/private/tmp/so2_one_patch_exact_9x9x9.npz"),
    )
    parser.add_argument(
        "--grids",
        type=Path,
        default=Path("/private/tmp/so2_9x9x9_grids.npz"),
    )
    parser.add_argument(
        "--single-gauge",
        type=Path,
        default=Path(
            "/private/tmp/so2_cas6e6o_631gstar_procrustes_gauge_9x9x9/"
            "procrustes_gauge.npz"
        ),
    )
    parser.add_argument(
        "--two-gauge",
        type=Path,
        default=Path(
            "/private/tmp/so2_cas6e6o_631gstar_procrustes_two_patch_9x9x9/"
            "procrustes_gauge.npz"
        ),
    )
    parser.add_argument(
        "--edge-check",
        type=Path,
        default=Path("/private/tmp/so2_theta_last_edge_5state.npz"),
    )
    parser.add_argument("--high-anchor", type=int, default=7)
    parser.add_argument("--high-boundary", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_three_patch_transport_9x9x9"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.fields) as archive:
        one_energy = np.asarray(archive["energy"])
        one_links = tuple(np.asarray(archive[f"link_{axis}"]) for axis in range(3))
    with np.load(args.grids) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
    with np.load(args.single_gauge) as archive:
        primary = np.asarray(archive["gauge"])
        center = tuple(map(int, archive["center"]))
    with np.load(args.two_gauge) as archive:
        two_gauge = np.asarray(archive["gauge"])
        two_energy = np.asarray(archive["aligned_local_hamiltonian"])
    with np.load(args.edge_check) as archive:
        edge_overlap = np.asarray(archive["overlap"])

    anchor = (center[0], int(args.high_anchor), center[2])
    high_gauge, high_positive, high_singular = path_patch(
        one_links,
        primary,
        anchor,
    )
    physical_links = raw_links(one_links, primary)
    links = {
        (axis, index): values[index]
        for axis, values in enumerate(physical_links)
        for index in np.ndindex(values.shape[:-2])
    }
    gauge, transition = stitch_upper(
        one_energy.shape[:-2],
        links,
        two_gauge,
        high_gauge,
        axis=1,
        boundary=int(args.high_boundary),
    )
    relative = primary.swapaxes(-1, -2).conj() @ gauge
    energy = relative.swapaxes(-1, -2).conj() @ one_energy @ relative
    aligned_links = rotate_links(one_links, primary, gauge)

    eig_error = float(
        np.max(np.abs(np.linalg.eigvalsh(energy) - np.linalg.eigvalsh(one_energy)))
    )
    edge_singular = {
        str(nstates): np.linalg.svd(
            edge_overlap[:nstates, :nstates], compute_uv=False
        ).tolist()
        for nstates in (3, 4, 5)
    }
    summary = {
        "method": "three-patch Procrustes diagnostic with high-angle link transport",
        "grid": list(energy.shape[:-2]),
        "centers": {
            "low_theta_index": 2,
            "middle": list(center),
            "high": list(anchor),
        },
        "high_boundary_theta_index": int(args.high_boundary),
        "max_eigenvalue_change_Eh": eig_error,
        "central_theta_max_step_Eh": {
            "one_patch": float(np.max(step_norm(one_energy, center))),
            "two_patch": float(np.max(step_norm(two_energy, center))),
            "three_patch": float(np.max(step_norm(energy, center))),
        },
        "last_theta_edge_singular_values": edge_singular,
        "three_state_rank_deficient": bool(edge_singular["3"][-1] < 1.0e-10),
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    np.savez(
        args.output_dir / "procrustes_gauge.npz",
        gauge=gauge,
        aligned_local_hamiltonian=energy,
        center=np.asarray(center),
        high_anchor=np.asarray(anchor),
        high_boundary_theta_index=int(args.high_boundary),
        high_positive=high_positive,
        high_singular_values=high_singular,
        high_transition=transition,
        **{f"link_{axis}": values for axis, values in enumerate(aligned_links)},
    )

    theta = np.rad2deg(grids[1])
    edge_theta = 0.5 * (theta[:-1] + theta[1:])
    adiabatic = np.linalg.eigvalsh(central_theta(one_energy, center))
    figure, axes = plt.subplots(1, 3, figsize=(8.7, 2.8), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for state, color in enumerate(colors):
        axes[0].plot(
            theta,
            adiabatic[:, state],
            color=color,
            marker=("o", "s", "^")[state],
            ms=3.5,
            lw=1.25,
            label=rf"$E_{state + 1}$",
        )
    for values, label, color, marker in (
        (one_energy, "one patch", "#999999", "o"),
        (two_energy, "two patches", "#D55E00", "s"),
        (energy, "three patches", "#0072B2", "^"),
    ):
        axes[1].plot(
            edge_theta,
            step_norm(values, center),
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.25,
            label=label,
        )
    positions = np.arange(1, 6)
    for nstates, color, marker in ((3, "#D55E00", "s"), (4, "#0072B2", "o"), (5, "#009E73", "^")):
        singular = np.asarray(edge_singular[str(nstates)])
        axes[2].plot(
            positions[:nstates],
            singular,
            color=color,
            marker=marker,
            ms=4,
            lw=1.25,
            label=rf"{nstates} states",
        )
    axes[0].set(xlabel=r"$\theta$ (degree)", ylabel=r"Adiabatic energy ($E_h$)")
    axes[1].set(
        xlabel=r"$\theta$ edge midpoint (degree)",
        ylabel=r"$\|\Delta\bar E\|_F$ ($E_h$)",
    )
    axes[2].set(
        xlabel="Singular-value index",
        ylabel=r"$\sigma[S(141.8^\circ,149.0^\circ)]$",
        yscale="log",
        ylim=(1.0e-15, 1.0),
        xticks=positions,
    )
    for axis in axes:
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
        axis.legend(frameon=False, fontsize=7.5)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
    figure_path = args.output_dir / "so2_three_patch_diagnostic.png"
    figure.savefig(figure_path, dpi=400, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
