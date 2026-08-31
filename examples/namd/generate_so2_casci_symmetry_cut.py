#!/usr/bin/env python3
"""Generate a symmetry-resolved SO2 CASCI bend cut."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.generate_so2_casci_singlets import (
    electronic_metadata,
    electronic_structure,
    electronic_symmetry_representation,
    require_spin_pure_singlets,
)
from pyqed.units import au2ev
from pyqed.ldr.overlap import procrustes


HARTREE_TO_EV = au2ev
OPERATIONS = {
    "E": (1.0, 1.0, 1.0),
    "C2(x)": (1.0, -1.0, -1.0),
    "sigma_xy": (1.0, 1.0, -1.0),
    "sigma_xz": (1.0, -1.0, 1.0),
}
IRREP_BY_CHARACTERS = {
    (1, 1, 1, 1): "A1",
    (1, 1, -1, -1): "A2",
    (1, -1, 1, -1): "B1",
    (1, -1, -1, 1): "B2",
}
TARGET_IRREPS = ("A1", "B2", "A2")
ROOT_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "<", ">")


def select_lowest_irreps(energies, labels, irreps=TARGET_IRREPS):
    """Select the lowest calculated root of each requested irrep."""
    selected = np.full((energies.shape[0], len(irreps)), np.nan)
    root_indices = np.full((energies.shape[0], len(irreps)), -1, dtype=int)
    for point in range(energies.shape[0]):
        for column, irrep in enumerate(irreps):
            candidates = np.flatnonzero(labels[point] == irrep)
            if candidates.size:
                root = candidates[np.argmin(energies[point, candidates])]
                selected[point, column] = energies[point, root]
                root_indices[point, column] = root
    return selected, root_indices


def neighbor_state_links(frames):
    """Return complete electronic overlap matrices between neighboring points."""
    return np.asarray([
        frames[left].overlap(frames[left + 1])
        for left in range(len(frames) - 1)
    ], dtype=complex)


def select_link_subspace(links, root_indices):
    """Restrict complete links to a geometry-dependent selected state order."""
    links = np.asarray(links, dtype=complex)
    root_indices = np.asarray(root_indices, dtype=int)
    if np.any(root_indices < 0):
        raise ValueError("each target irrep must occur at every geometry")
    return np.asarray([
        links[left][np.ix_(root_indices[left], root_indices[left + 1])]
        for left in range(len(links))
    ])


def positive_link_gauge(links, anchor):
    r"""Build gauges $G_i$ for which $G_i^\dagger S_iG_{i+1}$ is positive."""
    links = np.asarray(links, dtype=complex)
    npoints = links.shape[0] + 1
    nstates = links.shape[-1]
    anchor = int(anchor)
    if links.shape != (npoints - 1, nstates, nstates):
        raise ValueError("links must have shape (npoints - 1, nstates, nstates)")
    if not 0 <= anchor < npoints:
        raise ValueError("anchor is outside the coordinate cut")
    gauges = np.empty((npoints, nstates, nstates), dtype=complex)
    gauges[anchor] = np.eye(nstates)
    for edge in range(anchor, npoints - 1):
        rotation = procrustes(gauges[edge].conj().T @ links[edge])[0]
        gauges[edge + 1] = rotation.conj().T
    for edge in range(anchor - 1, -1, -1):
        rotation = procrustes(links[edge] @ gauges[edge + 1])[0]
        gauges[edge] = rotation
    aligned = np.asarray([
        gauges[edge].conj().T @ links[edge] @ gauges[edge + 1]
        for edge in range(npoints - 1)
    ])
    return gauges, aligned


def rotate_selected_energies(selected, gauges):
    """Rotate selected adiabatic energies into the positive-link gauge."""
    diagonal = np.asarray([np.diag(row) for row in selected], dtype=complex)
    rotated = np.einsum(
        "...ia,...ij,...jb->...ab",
        gauges.conj(), diagonal, gauges, optimize=True,
    )
    return 0.5 * (rotated + rotated.swapaxes(-1, -2).conj())


def plot_cut(theta, energies, labels, selected, output):
    center = int(np.argmin(np.abs(theta - np.deg2rad(120.0))))
    relative = (energies - energies[center, 0]) * HARTREE_TO_EV
    theta_deg = np.rad2deg(theta)
    figure, axes = plt.subplots(
        1, 2, figsize=(10.4, 4.0), sharey=True, constrained_layout=True
    )
    axis, tracked_axis = axes
    colors = dict(zip(sorted(set(labels.reshape(-1))), plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]))
    used = set()
    for point, angle in enumerate(theta_deg):
        for root in range(energies.shape[1]):
            label = labels[point, root]
            axis.scatter(
                angle, relative[point, root], s=34, color=colors[label],
                marker=ROOT_MARKERS[root % len(ROOT_MARKERS)],
                label=label if label not in used else None,
            )
            used.add(label)
    axis.set(
        xlabel=r"$\theta$ (degree)",
        ylabel=r"$E-E_1(120^\circ)$ (eV)",
        title="Energy-ordered roots",
    )
    axis.legend(frameon=False, ncol=min(4, len(used)))
    for column, irrep in enumerate(TARGET_IRREPS):
        tracked_axis.plot(
            theta_deg,
            (selected[:, column] - energies[center, 0]) * HARTREE_TO_EV,
            "o-",
            ms=4,
            color=colors[irrep],
            label=f"lowest {irrep}",
        )
    tracked_axis.set(
        xlabel=r"$\theta$ (degree)",
        title="Symmetry-selected branches",
    )
    tracked_axis.legend(frameon=False)
    figure.suptitle("SO$_2$ CASCI symmetry-resolved bend cut")
    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_p_gauge(theta, hamiltonian, aligned_links, output):
    """Plot the complete three-state Hamiltonian in the positive-link gauge."""
    theta_deg = np.rad2deg(theta)
    center = int(np.argmin(np.abs(theta - np.deg2rad(120.0))))
    reference = float(np.min(np.linalg.eigvalsh(hamiltonian[center])))
    shifted = (hamiltonian - reference * np.eye(hamiltonian.shape[-1])) * HARTREE_TO_EV
    figure, axes = plt.subplots(
        1, 2, figsize=(9.4, 3.8), constrained_layout=True
    )
    for state, irrep in enumerate(TARGET_IRREPS):
        axes[0].plot(
            theta_deg, shifted[:, state, state].real, "o-", ms=4,
            label=rf"$\bar E_{{{irrep},{irrep}}}$",
        )
    for left, right in ((0, 1), (0, 2), (1, 2)):
        axes[1].plot(
            theta_deg,
            shifted[:, left, right].real,
            "o-",
            ms=4,
            label=rf"Re $\bar E_{{{TARGET_IRREPS[left]},{TARGET_IRREPS[right]}}}$",
        )
    link_rotations = procrustes(aligned_links)[0]
    rotation_defect = float(np.max(np.linalg.norm(
        link_rotations - np.eye(hamiltonian.shape[-1]), axis=(-2, -1)
    )))
    imaginary_max = float(np.max(np.abs(shifted.imag)))
    axes[0].set(
        xlabel=r"$\theta$ (degree)",
        ylabel=r"$\bar E-E_{A_1}(120^\circ)$ (eV)",
        title="Diagonal elements",
    )
    axes[1].set(
        xlabel=r"$\theta$ (degree)",
        ylabel=r"Off-diagonal $\bar E$ (eV)",
        title="Off-diagonal elements",
    )
    axes[1].text(
        0.03, 0.05,
        rf"max Im $\bar E={imaginary_max:.1e}$ eV" "\n"
        rf"max $\|V_i-I\|_F={rotation_defect:.1e}$",
        transform=axes[1].transAxes,
    )
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("SO$_2$ CASCI(8e,8o)/6-31G* positive-link gauge")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _anchored_channel_labels(labels):
    totals = {label: list(labels).count(label) for label in set(labels)}
    seen = {label: 0 for label in totals}
    output = []
    for label in labels:
        seen[label] += 1
        output.append(
            label if totals[label] == 1 else f"{label}({seen[label]})"
        )
    return tuple(output)


def plot_full_p_gauge(theta, hamiltonian, anchor_labels, aligned_links, output):
    """Plot the full six-root Hamiltonian in its positive-link gauge."""
    center = int(np.argmin(np.abs(theta - np.deg2rad(120.0))))
    singular_min = np.min(
        np.linalg.svd(aligned_links, compute_uv=False), axis=-1
    )
    lower = center
    while lower > 0 and singular_min[lower - 1] > 1.0e-8:
        lower -= 1
    upper = center
    while upper < len(theta) - 1 and singular_min[upper] > 1.0e-8:
        upper += 1
    valid = slice(lower, upper + 1)
    theta_deg = np.rad2deg(theta[valid])
    reference = float(np.min(np.linalg.eigvalsh(hamiltonian[center])))
    shifted = (hamiltonian - reference * np.eye(hamiltonian.shape[-1])) * HARTREE_TO_EV
    channel_labels = _anchored_channel_labels(anchor_labels)
    figure, axes = plt.subplots(
        1, 2, figsize=(10.6, 4.1), constrained_layout=True
    )
    for state, label in enumerate(channel_labels):
        axes[0].plot(
            theta_deg, shifted[valid, state, state].real, "o-", ms=3.5,
            label=rf"$\bar E_{{{label},{label}}}$",
        )
    allowed = []
    forbidden = []
    for left in range(len(channel_labels)):
        for right in range(left + 1, len(channel_labels)):
            pair = shifted[valid, left, right]
            if anchor_labels[left] == anchor_labels[right]:
                allowed.append(pair)
                axes[1].plot(
                    theta_deg, pair.real, "o-", ms=4,
                    label=rf"Re $\bar E_{{{channel_labels[left]},{channel_labels[right]}}}$",
                )
            else:
                forbidden.append(pair)
    forbidden_max = (
        0.0 if not forbidden else float(np.max(np.abs(np.asarray(forbidden))))
    )
    valid_links = aligned_links[lower:upper]
    link_rotations = procrustes(valid_links)[0]
    rotation_defect = float(np.max(np.linalg.norm(
        link_rotations - np.eye(hamiltonian.shape[-1]), axis=(-2, -1)
    )))
    axes[0].set(
        xlabel=r"$\theta$ (degree)",
        ylabel=r"$\bar E^{(6)}-E_1(120^\circ)$ (eV)",
        title="Six diagonal elements",
    )
    axes[1].set(
        xlabel=r"$\theta$ (degree)",
        ylabel=r"Off-diagonal $\bar E^{(6)}$ (eV)",
        title="Symmetry-allowed couplings",
    )
    axes[1].text(
        0.03, 0.48,
        rf"max forbidden $|\bar E_{{ab}}|={forbidden_max:.1e}$ eV" "\n"
        rf"max $\|V_i-I\|_F={rotation_defect:.1e}$" "\n"
        rf"min $\sigma(L_i)={np.min(singular_min[lower:upper]):.2e}$",
        transform=axes[1].transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    for axis in axes:
        axis.legend(frameon=False, fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "SO$_2$ full six-root CASCI positive-link gauge, "
        f"{theta_deg[0]:.0f}°–{theta_deg[-1]:.0f}°"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r", type=float, default=2.8)
    parser.add_argument("--n-theta", type=int, default=17)
    parser.add_argument("--theta-min-deg", type=float, default=80.0)
    parser.add_argument("--theta-max-deg", type=float, default=160.0)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--nstates", type=int, default=6)
    parser.add_argument("--ncas", type=int, default=8)
    parser.add_argument("--nelecas", type=int, default=8)
    parser.add_argument("--spin-root-cushion", type=int, default=32)
    parser.add_argument("--scf-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/private/tmp/so2_casci_8e8o_631gstar_theta_cut_6roots.npz"
        ),
    )
    args = parser.parse_args()
    theta = np.deg2rad(np.linspace(
        args.theta_min_deg, args.theta_max_deg, args.n_theta
    ))
    energies, spin_square, representations, frames = [], [], [], []
    for index, angle in enumerate(theta, start=1):
        model = electronic_structure(args.r, args.r, angle, args)
        frames.append(model.frame())
        energies.append(np.asarray(model.e_tot))
        spin_square.append([
            model.spin_square(state) for state in range(args.nstates)
        ])
        representations.append([
            electronic_symmetry_representation(model, signs)[0]
            for signs in OPERATIONS.values()
        ])
        print(
            f"[CASCI] {index}/{len(theta)}, E0={energies[-1][0]:.10f} Eh, "
            f"max |S2|={np.max(np.abs(spin_square[-1])):.2e}",
            flush=True,
        )
    energies = np.asarray(energies)
    spin_square = np.asarray(spin_square)
    representations = np.asarray(representations)
    require_spin_pure_singlets(spin_square)
    characters = np.where(
        np.real(np.diagonal(representations, axis1=2, axis2=3)) >= 0.0,
        1,
        -1,
    )
    labels = np.asarray([
        [IRREP_BY_CHARACTERS.get(tuple(characters[point, :, root]), "?")
         for root in range(args.nstates)]
        for point in range(len(theta))
    ])
    selected, selected_root_indices = select_lowest_irreps(energies, labels)
    full_links = neighbor_state_links(frames)
    links = select_link_subspace(full_links, selected_root_indices)
    anchor = int(np.argmin(np.abs(theta - np.deg2rad(120.0))))
    p_gauge, p_links = positive_link_gauge(links, anchor)
    p_hamiltonian = rotate_selected_energies(selected, p_gauge)
    full_p_gauge, full_p_links = positive_link_gauge(full_links, anchor)
    full_p_hamiltonian = rotate_selected_energies(energies, full_p_gauge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        r=np.asarray(args.r),
        theta=theta,
        energies=energies,
        spin_square=spin_square,
        point_group_names=np.asarray(tuple(OPERATIONS)),
        point_group_representations=representations,
        point_group_characters=characters,
        point_group_labels=labels,
        selected_irreps=np.asarray(TARGET_IRREPS),
        selected_energies=selected,
        selected_root_indices=selected_root_indices,
        selected_links=links,
        full_links=full_links,
        p_gauge_anchor=np.asarray(anchor),
        p_gauge=p_gauge,
        p_gauge_links=p_links,
        p_gauge_hamiltonian=p_hamiltonian,
        full_p_gauge=full_p_gauge,
        full_p_gauge_links=full_p_links,
        full_p_gauge_hamiltonian=full_p_hamiltonian,
        full_p_gauge_anchor_labels=labels[anchor],
        **electronic_metadata(args),
    )
    figure = args.output.with_suffix(".png")
    plot_cut(theta, energies, labels, selected, figure)
    p_figure = args.output.with_name(args.output.stem + "_p_gauge.png")
    plot_p_gauge(theta, p_hamiltonian, p_links, p_figure)
    full_p_figure = args.output.with_name(
        args.output.stem + "_full6_p_gauge.png"
    )
    plot_full_p_gauge(
        theta, full_p_hamiltonian, labels[anchor], full_p_links, full_p_figure
    )
    print(f"dataset: {args.output}")
    print(f"figure: {figure}")
    print(f"P-gauge figure: {p_figure}")
    print(f"full-six P-gauge figure: {full_p_figure}")


if __name__ == "__main__":
    main()
