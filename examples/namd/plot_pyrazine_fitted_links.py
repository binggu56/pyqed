#!/usr/bin/env python3
"""Compare fitted feature links with pyrazine ab initio overlap blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy as np

from pyqed.ldr import AbInitioFit
from pyqed.ldr.overlap import procrustes
from pyqed.ldr.ttfit import sample_graph


DEFAULT_ROOT = Path(
    "/private/tmp/pyrazine_casci_adaptive_fibers_ttldr_production_final"
)
DEFAULT_DATA = (
    Path.home()
    / "Library/CloudStorage/OneDrive-西湖大学"
    / "manuscripts/SD/calculations/real_smolyak_20260803/product_n11_casci.pkl"
)


def canonical(pair):
    left, right = map(tuple, pair)
    return (left, right) if left < right else (right, left)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/pyrazine_fitted_abinitio_links"),
    )
    args = parser.parse_args()

    with (args.run / "summary.json").open() as stream:
        summary = json.load(stream)
    sampling = summary["abinitio_fit"]
    points = tuple(map(tuple, sampling["points"]))
    shape = tuple(summary["grid"])
    pairs = sample_graph(points, shape, neighbors=int(sampling["neighbors"]))
    if len(pairs) != int(sampling["pairs"]):
        raise RuntimeError("reconstructed sampled graph differs from the saved fit")

    with args.data.open("rb") as stream:
        raw = pickle.load(stream)
    centers = np.asarray(raw["centers"], dtype=float)
    grids = tuple(np.unique(centers[:, axis]) for axis in range(len(shape)))
    overlaps = np.asarray(raw["overlap"], dtype=complex)
    nstates = int(summary["nstates"])
    anchor = tuple(size // 2 for size in shape)
    anchor_flat = int(np.ravel_multi_index(anchor, shape))
    gauges = np.empty((*shape, nstates, nstates), dtype=complex)
    for point in np.ndindex(shape):
        flat = int(np.ravel_multi_index(point, shape))
        gauges[point] = procrustes(overlaps[flat, :, anchor_flat, :])[0]

    fit = AbInitioFit.load(args.run / "fields")
    try:
        feature_rank = int(fit.feature.output_shape_[0])
        all_points = tuple(np.ndindex(shape))
        coordinates = np.asarray(
            [
                [grids[axis][index] for axis, index in enumerate(point)]
                for point in all_points
            ],
            dtype=float,
        )
        features = np.asarray(fit.feature.predict(coordinates)).reshape(
            *shape, feature_rank, nstates
        )
    finally:
        fit.close()

    def blocks(link_pairs):
        reference = []
        fitted = []
        errors = []
        for left, right in link_pairs:
            left_flat = int(np.ravel_multi_index(left, shape))
            right_flat = int(np.ravel_multi_index(right, shape))
            target = (
                gauges[left].conj().T
                @ overlaps[left_flat, :, right_flat, :]
                @ gauges[right]
            )
            predicted = features[left].conj().T @ features[right]
            reference.append(target)
            fitted.append(predicted)
            errors.append(
                np.linalg.norm(predicted - target)
                / max(float(np.linalg.norm(target)), np.finfo(float).tiny)
            )
        return np.asarray(reference), np.asarray(fitted), np.asarray(errors)

    train_reference, train_fitted, train_error = blocks(pairs)
    nearest = []
    for left in np.ndindex(shape):
        for axis, extent in enumerate(shape):
            if left[axis] + 1 >= extent:
                continue
            right = list(left)
            right[axis] += 1
            nearest.append((left, tuple(right)))
    training_edges = {canonical(pair) for pair in pairs}
    unseen = tuple(pair for pair in nearest if canonical(pair) not in training_edges)
    unseen_reference, unseen_fitted, unseen_error = blocks(unseen)

    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.15), constrained_layout=True)
    segments = np.asarray(
        [[[left[0], left[1]], [right[0], right[1]]] for left, right in pairs]
    )
    lower = max(float(np.min(train_error[train_error > 0.0])), 1.0e-5)
    upper = max(float(np.max(train_error)), 10.0 * lower)
    collection = LineCollection(
        segments,
        array=train_error,
        cmap="viridis",
        norm=LogNorm(vmin=lower, vmax=upper),
        linewidths=1.15,
        alpha=0.82,
        zorder=1,
    )
    axes[0].add_collection(collection)
    sampled = np.asarray(points)
    axes[0].scatter(
        sampled[:, 0], sampled[:, 1], s=16, color="0.1", zorder=2
    )
    axes[0].set(
        xlabel=r"$Q_0$ index", ylabel=r"$Q_1$ index", aspect="equal",
        xlim=(-0.5, shape[0] - 0.5), ylim=(-0.5, shape[1] - 0.5),
    )
    fig.colorbar(collection, ax=axes[0], label="Relative link error")

    def parity(axis, reference, fitted, label, color):
        reference = reference.reshape(-1)
        fitted = fitted.reshape(-1)
        axis.scatter(
            reference.real, fitted.real, s=8, alpha=0.42,
            color=color, marker="o", label=label, rasterized=True,
        )

    parity(axes[1], train_reference, train_fitted, "training", "#0072B2")
    parity(axes[1], unseen_reference, unseen_fitted, "unseen", "#D55E00")
    values = np.concatenate(
        (
            train_reference.real.ravel(), train_fitted.real.ravel(),
            unseen_reference.real.ravel(), unseen_fitted.real.ravel(),
        )
    )
    limit = 1.05 * float(np.max(np.abs(values)))
    axes[1].plot([-limit, limit], [-limit, limit], color="0.15", lw=0.8)
    axes[1].set(
        xlabel=r"Ab initio $\bar S_{ij}$",
        ylabel=r"Fitted $Y_i^\dagger Y_j$",
        xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
    )
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left")

    bins = np.logspace(-4, 1, 45)
    axes[2].hist(
        train_error, bins=bins, histtype="step", lw=1.7,
        color="#0072B2", label=f"Training ({len(train_error)})",
    )
    axes[2].hist(
        unseen_error, bins=bins, histtype="step", lw=1.7,
        color="#D55E00", label=f"Unseen nearest ({len(unseen_error)})",
    )
    axes[2].axvline(
        np.sqrt(np.mean(train_error**2)), color="#0072B2", ls="--", lw=1.0
    )
    axes[2].axvline(
        np.sqrt(np.mean(unseen_error**2)), color="#D55E00", ls="--", lw=1.0
    )
    axes[2].set(
        xscale="log", xlabel="Relative link error", ylabel="Number of links"
    )
    axes[2].legend(frameon=False, fontsize=7)

    for label, axis in zip("abc", axes):
        axis.text(
            0.02, 0.97, f"({label})", transform=axis.transAxes,
            va="top", fontweight="bold",
        )
        axis.grid(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=350)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)

    diagnostics = {
        "training_links": len(pairs),
        "unseen_nearest_links": len(unseen),
        "training_rms_relative_error": float(np.sqrt(np.mean(train_error**2))),
        "training_max_relative_error": float(np.max(train_error)),
        "unseen_rms_relative_error": float(np.sqrt(np.mean(unseen_error**2))),
        "unseen_max_relative_error": float(np.max(unseen_error)),
        "maximum_ab_initio_imaginary_part": float(
            max(np.max(np.abs(train_reference.imag)), np.max(np.abs(unseen_reference.imag)))
        ),
        "maximum_fitted_imaginary_part": float(
            max(np.max(np.abs(train_fitted.imag)), np.max(np.abs(unseen_fitted.imag)))
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2))
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
