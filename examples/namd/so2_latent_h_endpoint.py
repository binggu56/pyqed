#!/usr/bin/env python3
"""Fit SO2 endpoint frames as low eigenvectors of a latent Hamiltonian."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr.oracle import synchronize_features
from pyqed.ml import MACE

from examples.namd.so2_mace_ttldr import aligned_fields, split_samples


def synchronize(coordinates, pairs, links, rank, anchor, steps, seed):
    blocks = {}
    for pair, value in zip(pairs, links):
        left, right = (int(pair[0]),), (int(pair[1]),)
        blocks[(left, right)] = value
        blocks[(right, left)] = value.conj().T

    class Samples:
        shape = (len(coordinates),)

        @staticmethod
        def overlap_many(requested):
            return np.asarray([blocks[(tuple(left), tuple(right))] for left, right in requested])

    return synchronize_features(
        Samples(),
        tuple((point,) for point in range(len(coordinates))),
        tuple(((int(left),), (int(right),)) for left, right in pairs),
        rank,
        anchor=(int(anchor),),
        penalty=10.0,
        smoothness=1.0e-4,
        maxiter=steps,
        gtol=1.0e-8,
        seed=seed,
    )


def latent_hamiltonians(frames, nstates):
    rank = frames.shape[-2]
    selected = np.linspace(-1.0, -0.2, nstates)
    complement = 1.0
    projector = frames @ frames.conj().swapaxes(-1, -2)
    resolved = np.einsum(
        "nra,a,nsa->nrs", frames, selected, frames.conj(), optimize=True
    )
    return resolved + complement * (np.eye(rank) - projector), selected, complement


def transport_phases(frames, shape, anchor):
    """Choose continuous eigenvector phases on a nearest-neighbor tree."""

    values = np.asarray(frames, dtype=complex).reshape(*shape, *frames.shape[-2:]).copy()
    anchor_frame = values[anchor]
    diagonal = np.diagonal(anchor_frame[: anchor_frame.shape[-1]])
    phase = diagonal / np.maximum(np.abs(diagonal), np.finfo(float).tiny)
    values[anchor] *= phase.conj()[None, :]
    visited = {anchor}
    queue = deque([anchor])
    while queue:
        parent = queue.popleft()
        for axis, size in enumerate(shape):
            for step in (-1, 1):
                child = list(parent)
                child[axis] += step
                child = tuple(child)
                if child in visited or not 0 <= child[axis] < size:
                    continue
                overlap = np.diagonal(values[parent].conj().T @ values[child])
                phase = overlap / np.maximum(np.abs(overlap), np.finfo(float).tiny)
                values[child] *= phase.conj()[None, :]
                visited.add(child)
                queue.append(child)
    return values


def link_errors(frames, links, masks):
    train, held, pointwise = [], [], []
    magnitude_train, magnitude_held = [], []
    singular_train, singular_held = [], []
    for axis, (reference, mask) in enumerate(zip(links, masks)):
        left = [slice(None)] * len(masks)
        right = [slice(None)] * len(masks)
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        predicted = (
            frames[tuple(left)].conj().swapaxes(-1, -2) @ frames[tuple(right)]
        )
        reference_flat = reference.reshape(-1, *reference.shape[-2:])
        predicted_flat = predicted.reshape(reference_flat.shape)
        error = np.linalg.norm(
            predicted_flat - reference_flat, axis=(-2, -1)
        ) / np.maximum(np.linalg.norm(reference_flat, axis=(-2, -1)), np.finfo(float).tiny)
        train.append(float(np.linalg.norm(predicted_flat[mask] - reference_flat[mask]) / np.linalg.norm(reference_flat[mask])))
        held.append(float(np.linalg.norm(predicted_flat[~mask] - reference_flat[~mask]) / np.linalg.norm(reference_flat[~mask])))
        magnitude_train.append(float(np.linalg.norm(np.abs(predicted_flat[mask]) - np.abs(reference_flat[mask])) / np.linalg.norm(reference_flat[mask])))
        magnitude_held.append(float(np.linalg.norm(np.abs(predicted_flat[~mask]) - np.abs(reference_flat[~mask])) / np.linalg.norm(reference_flat[~mask])))
        predicted_singular = np.linalg.svd(predicted_flat, compute_uv=False)
        reference_singular = np.linalg.svd(reference_flat, compute_uv=False)
        singular_train.append(float(np.linalg.norm(predicted_singular[mask] - reference_singular[mask]) / np.linalg.norm(reference_singular[mask])))
        singular_held.append(float(np.linalg.norm(predicted_singular[~mask] - reference_singular[~mask]) / np.linalg.norm(reference_singular[~mask])))
        pointwise.append(error[~mask])
    return {
        "full": (train, held),
        "magnitude": (magnitude_train, magnitude_held),
        "singular": (singular_train, singular_held),
        "pointwise": pointwise,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--sync-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chart-features", action="store_true")
    parser.add_argument("--spectral", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/so2_latent_h_rank6.png"))
    args = parser.parse_args()

    grids, energy, links, geometry = aligned_fields(args.fixture)
    (
        energy_train,
        _link_train,
        pairs,
        pair_links,
        coordinates,
        train,
        held,
        masks,
    ) = split_samples(grids, energy, links, args.seed, 0.2, "spatial-block")
    nstates = energy.shape[-1]
    if args.rank < nstates + 1:
        raise ValueError("latent rank must exceed the number of electronic states")
    center = np.asarray([grid[len(grid) // 2] for grid in grids])
    anchor = int(np.argmin(np.linalg.norm(coordinates[train] - center, axis=1)))
    targets, sync_info = synchronize(
        coordinates[train], pairs, pair_links, args.rank, anchor, args.sync_steps, args.seed
    )
    latent_h, selected, complement = latent_hamiltonians(targets, nstates)

    model = MACE(
        grids,
        ("O", "S", "O"),
        geometry,
        args.rank,
        chart_features=args.chart_features,
        geometry_units="bohr",
        channels=8,
        max_ell=2,
        interactions=2,
        correlation=2,
        radial_basis=8,
        radial_mlp=(64, 64),
        cutoff=7.0,
    )
    if args.spectral:
        differences = np.abs(
            coordinates[train][pairs[:, 1]] - coordinates[train][pairs[:, 0]]
        )
        pair_axes = np.argmax(differences, axis=1)
        model.fit_spectral(
            coordinates[train],
            pairs,
            pair_links,
            targets,
            pair_axes=pair_axes,
            pretrain_values=latent_h,
            selected_states=nstates,
            hidden=(64, 64),
            epochs=args.epochs,
            pretrain_epochs=args.pretrain_epochs,
            learning_rate=2.0e-3,
            projector_weight=2.0,
            link_weight=5.0,
            spectrum_weight=0.05,
            seed=args.seed,
        )
    else:
        model.fit_h(
            coordinates[train],
            latent_h,
            hidden=(64, 64),
            epochs=args.epochs,
            learning_rate=2.0e-3,
            seed=args.seed,
        )
    prediction = model.neural_energy.predict(coordinates)
    values, vectors = np.linalg.eigh(prediction)
    frames = transport_phases(
        vectors[..., :nstates], energy.shape[:-2], tuple(size // 2 for size in energy.shape[:-2])
    )
    errors = link_errors(frames, links, masks)
    train_error, held_error = errors["full"]
    held_pointwise = errors["pointwise"]

    checkpoint = args.output.with_suffix(".pt")
    model.save(checkpoint)
    baseline_path = Path("/private/tmp/so2_casci6e6o_mace_endpoint_5x5x5.json")
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    direct_train = None if baseline is None else [baseline["link_relative_error"][str(axis)]["train"] for axis in range(3)]
    direct_held = None if baseline is None else [baseline["link_relative_error"][str(axis)]["held_out"] for axis in range(3)]

    figure, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), constrained_layout=True)
    axes[0].semilogy(model.history, color="#0072B2", lw=1.2)
    axes[0].set(xlabel="Epoch", ylabel="Latent-H loss")
    axes[1].scatter(np.tile(selected, len(train)), values[train, :nstates].reshape(-1), s=10, color="#0072B2", alpha=0.7)
    extent = (selected[0] - 0.1, selected[-1] + 0.1)
    axes[1].plot(extent, extent, color="0.55", lw=1.0)
    axes[1].set(xlabel="Target latent eigenvalue", ylabel="Predicted latent eigenvalue")
    labels = (r"$r_1$", r"$r_2$", r"$\theta$")
    x = np.arange(3)
    if direct_held is not None:
        axes[2].bar(x - 0.18, direct_held, width=0.36, color="#999999", label="Direct $Y$")
    axes[2].bar(x + 0.18, held_error, width=0.36, color="#D55E00", label="Latent $H$")
    axes[2].set(xticks=x, xticklabels=labels, ylabel="Held-out link error", ylim=(0.0, max(held_error + (direct_held or [])) * 1.18))
    axes[2].legend(frameon=False)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)

    metrics = {
        "latent_rank": args.rank,
        "nstates": nstates,
        "samples": {"train": len(train), "held_out": len(held)},
        "chart_features": args.chart_features,
        "training": "spectral" if args.spectral else "entrywise",
        "latent_eigenvalues": selected.tolist(),
        "complement_eigenvalue": complement,
        "link_relative_error": {
            str(axis): {"train": train_error[axis], "held_out": held_error[axis]}
            for axis in range(3)
        },
        "link_magnitude_relative_error": {
            str(axis): {
                "train": errors["magnitude"][0][axis],
                "held_out": errors["magnitude"][1][axis],
            }
            for axis in range(3)
        },
        "link_singular_value_relative_error": {
            str(axis): {
                "train": errors["singular"][0][axis],
                "held_out": errors["singular"][1][axis],
            }
            for axis in range(3)
        },
        "direct_y_link_relative_error": None if baseline is None else baseline["link_relative_error"],
        "held_pointwise_rms": [float(np.sqrt(np.mean(item**2))) for item in held_pointwise],
        "synchronization": sync_info,
        "final_loss": model.history[-1],
        "checkpoint": str(checkpoint),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
