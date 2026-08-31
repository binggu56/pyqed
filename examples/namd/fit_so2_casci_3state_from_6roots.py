#!/usr/bin/env python3
"""Disentangle and fit three SO2 states from a six-root CASCI bend cut."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2ev
from pyqed.ldr.overlap import procrustes
from pyqed.mps.functional import FunctionalTT


HARTREE_TO_EV = au2ev
TARGET_LABELS = ("A1", "B2", "A2")


def transport_subspace(full_links, anchor, anchor_states=(0, 1, 2)):
    """Polar-transport an anchored subspace through a larger raw-state window."""
    full_links = np.asarray(full_links, dtype=complex)
    rotations = procrustes(full_links)[0]
    npoints = len(full_links) + 1
    nraw = full_links.shape[-1]
    anchor_states = np.asarray(anchor_states, dtype=int)
    frames = np.empty((npoints, nraw, len(anchor_states)), dtype=complex)
    frames[int(anchor)] = np.eye(nraw, dtype=complex)[:, anchor_states]
    for edge in range(int(anchor), npoints - 1):
        frames[edge + 1] = rotations[edge].conj().T @ frames[edge]
    for edge in range(int(anchor) - 1, -1, -1):
        frames[edge] = rotations[edge] @ frames[edge + 1]
    links = np.asarray([
        frames[edge].conj().T @ full_links[edge] @ frames[edge + 1]
        for edge in range(npoints - 1)
    ])
    return frames, 0.5 * (links + links.swapaxes(-1, -2).conj())


def projected_hamiltonian(energies, frames):
    """Project the raw diagonal Hamiltonian into transported subspace frames."""
    hamiltonian = np.einsum(
        "...ia,...i,...ib->...ab",
        frames.conj(), energies, frames, optimize=True,
    )
    return 0.5 * (hamiltonian + hamiltonian.swapaxes(-1, -2).conj())


def hamiltonian_residual(energies, frames):
    """Return coupling from the transported subspace to discarded raw states."""
    identity = np.eye(frames.shape[-2])
    residual = np.empty(len(frames))
    for point, frame in enumerate(frames):
        projector = frame @ frame.conj().T
        centered = energies[point] - energies[point, 0]
        coupling = (identity - projector) @ (centered[:, None] * frame)
        residual[point] = np.linalg.norm(coupling)
    return residual


def fit_field(coordinates, values, train, held, degree, bounds, seed):
    model = FunctionalTT(
        degrees=int(degree),
        rank=max(2, int(degree)),
        bounds=(tuple(map(float, bounds)),),
        normalization="frobenius",
        hermitian=True,
        regularization=1.0e-11,
        sweeps=30,
        rtol=1.0e-11,
        random_state=int(seed),
    ).fit(
        coordinates[train],
        values[train],
        validation=(coordinates[held], values[held]),
    )
    return model


def field_metrics(predicted, reference, mask, *, scale=1.0):
    difference = (predicted[mask] - reference[mask]) * scale
    denominator = max(
        float(np.linalg.norm(reference[mask] * scale)), np.finfo(float).tiny
    )
    return {
        "relative_frobenius": float(np.linalg.norm(difference) / denominator),
        "rms": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def plot_fit(
    theta,
    hamiltonian,
    links,
    energy_model,
    link_model,
    point_train,
    point_held,
    link_train,
    link_held,
    residual,
    output,
    metrics,
):
    theta_deg = np.rad2deg(theta)
    midpoint = 0.5 * (theta[:-1] + theta[1:])
    midpoint_deg = np.rad2deg(midpoint)
    dense = np.linspace(theta[0], theta[-1], 401)[:, None]
    fitted_dense = energy_model.predict(dense) * HARTREE_TO_EV
    link_dense = np.linspace(midpoint[0], midpoint[-1], 401)[:, None]
    fitted_link_dense = link_model.predict(link_dense)
    figure, axes = plt.subplots(
        1, 3, figsize=(12.4, 3.75), constrained_layout=True
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for state, label in enumerate(TARGET_LABELS):
        color = colors[state]
        axes[0].plot(
            np.rad2deg(dense[:, 0]), fitted_dense[:, state, state].real,
            color=color, label=rf"fit $\bar E_{{{label},{label}}}$",
        )
        axes[0].scatter(
            theta_deg[point_train],
            (hamiltonian[point_train, state, state] * HARTREE_TO_EV).real,
            s=25, color=color, zorder=3,
        )
        axes[0].scatter(
            theta_deg[point_held],
            (hamiltonian[point_held, state, state] * HARTREE_TO_EV).real,
            s=30, facecolor="white", edgecolor=color, zorder=3,
        )
        axes[1].plot(
            np.rad2deg(link_dense[:, 0]),
            fitted_link_dense[:, state, state].real,
            color=color, label=rf"fit $P_{{{label},{label}}}$",
        )
        axes[1].scatter(
            midpoint_deg[link_train], links[link_train, state, state].real,
            s=25, color=color, zorder=3,
        )
        axes[1].scatter(
            midpoint_deg[link_held], links[link_held, state, state].real,
            s=30, facecolor="white", edgecolor=color, zorder=3,
        )
    singular_min = np.min(np.linalg.svd(links, compute_uv=False), axis=-1)
    axes[2].plot(
        theta_deg, residual * HARTREE_TO_EV, "o-", color=colors[3],
        label=r"$\|(I-WW^\dagger)HW\|_F$",
    )
    twin = axes[2].twinx()
    twin.plot(
        midpoint_deg, singular_min, "s--", color="0.25",
        label=r"$\sigma_{\min}(P_i)$",
    )
    held_energy = metrics["energy_held"]
    held_link = metrics["link_held"]
    axes[2].text(
        0.34, 0.74,
        rf"held max $|\Delta E|$ = {1e3 * held_energy['max_abs']:.2f} meV" "\n"
        rf"held max $|\Delta P|$ = {held_link['max_abs']:.2e}",
        transform=axes[2].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    axes[0].set(
        xlabel=r"$\theta$ (degree)",
        ylabel=r"$H_{\mathrm{eff}}-E_1(120^\circ)$ (eV)",
        title="Three-state Hamiltonian fit",
    )
    axes[1].set(
        xlabel=r"link midpoint $\theta$ (degree)",
        ylabel=r"Positive link $P_i$",
        title="Three-state link fit",
    )
    axes[2].set(
        xlabel=r"$\theta$ (degree)",
        ylabel="Electronic truncation residual (eV)",
        title="Subspace diagnostics",
    )
    twin.set_ylabel(r"Minimum link singular value")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7)
    twin.spines["top"].set_visible(False)
    lines, labels = axes[2].get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    axes[2].legend(lines + twin_lines, labels + twin_labels, frameon=False, loc="center left")
    figure.suptitle("SO$_2$: rank-3 fit disentangled from six CASCI roots")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("/private/tmp/so2_casci_8e8o_631gstar_theta_cut_6roots.npz"),
    )
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/so2_casci_3state_from_6roots_fit.png"),
    )
    args = parser.parse_args()
    with np.load(args.input) as archive:
        theta = np.asarray(archive["theta"], dtype=float)
        energies = np.asarray(archive["energies"], dtype=float)
        full_links = np.asarray(archive["full_links"], dtype=complex)
        anchor = int(archive["p_gauge_anchor"])
    frames, links = transport_subspace(full_links, anchor)
    hamiltonian = projected_hamiltonian(energies, frames)
    reference = float(energies[anchor, 0])
    hamiltonian -= reference * np.eye(hamiltonian.shape[-1])
    residual = hamiltonian_residual(energies, frames)

    if args.holdout_stride < 2:
        raise ValueError("holdout stride must be at least two")
    point_held = np.arange(args.holdout_stride - 1, len(theta), args.holdout_stride)
    point_train = np.setdiff1d(np.arange(len(theta)), point_held)
    edge_midpoint = 0.5 * (theta[:-1] + theta[1:])
    link_held = np.arange(
        args.holdout_stride - 1, len(edge_midpoint), args.holdout_stride
    )
    link_train = np.setdiff1d(np.arange(len(edge_midpoint)), link_held)
    bounds = (theta[0], theta[-1])
    energy_model = fit_field(
        theta[:, None], hamiltonian, point_train, point_held,
        args.degree, bounds, args.seed,
    )
    link_model = fit_field(
        edge_midpoint[:, None], links, link_train, link_held,
        args.degree, bounds, args.seed + 1,
    )
    predicted_energy = energy_model.predict(theta[:, None])
    predicted_links = link_model.predict(edge_midpoint[:, None])
    metrics = {
        "energy_train": field_metrics(
            predicted_energy, hamiltonian, point_train, scale=HARTREE_TO_EV
        ),
        "energy_held": field_metrics(
            predicted_energy, hamiltonian, point_held, scale=HARTREE_TO_EV
        ),
        "link_train": field_metrics(predicted_links, links, link_train),
        "link_held": field_metrics(predicted_links, links, link_held),
        "minimum_link_singular_value": float(np.min(
            np.linalg.svd(links, compute_uv=False)
        )),
        "maximum_hamiltonian_residual_ev": float(
            np.max(residual) * HARTREE_TO_EV
        ),
    }
    stem = args.output.with_suffix("")
    energy_model.save(stem.with_name(stem.name + "_energy.npz"))
    link_model.save(stem.with_name(stem.name + "_links.npz"))
    np.savez(
        stem.with_name(stem.name + "_data.npz"),
        theta=theta,
        frames=frames,
        hamiltonian=hamiltonian,
        links=links,
        predicted_hamiltonian=predicted_energy,
        predicted_links=predicted_links,
        point_train=point_train,
        point_held=point_held,
        link_train=link_train,
        link_held=link_held,
        residual=residual,
    )
    plot_fit(
        theta, hamiltonian, links, energy_model, link_model,
        point_train, point_held, link_train, link_held,
        residual, args.output, metrics,
    )
    for name, values in metrics.items():
        print(f"{name}: {values}")
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
