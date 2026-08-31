#!/usr/bin/env python3
"""Plot direct and invariant diagnostics of a fitted real pyrazine Y field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from pyqed.ldr import AbInitioFit


DEFAULT_RUN = Path("/private/tmp/pyrazine_native_real_rank24_degree10")
DEFAULT_OUTPUT = Path("/private/tmp/pyrazine_native_real_y")


def sampled_points(run):
    summary = json.loads((run / "summary.json").read_text())
    return np.asarray(summary["abinitio_fit"]["points"], dtype=int)


def coordinates(grids):
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1)


def add_samples(axis, grids, points):
    axis.scatter(
        grids[0][points[:, 0]], grids[1][points[:, 1]],
        s=4.0, facecolors="none", edgecolors="black", linewidths=0.25,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--channels", type=int, default=6)
    args = parser.parse_args()

    fit = AbInitioFit.load(args.run / "fields")
    try:
        grids = fit.grids
        shape = tuple(len(grid) for grid in grids)
        feature_rank, nstates = fit.feature.output_shape_
        y = np.asarray(fit.feature.predict(coordinates(grids))).reshape(
            *shape, feature_rank, nstates
        )
    finally:
        fit.close()
    if np.iscomplexobj(y):
        raise ValueError("this diagnostic requires a real fitted Y field")

    points = sampled_points(args.run)
    variance = np.var(y, axis=(0, 1, 3))
    count = min(int(args.channels), feature_rank)
    selected = np.argsort(variance)[-count:][::-1]
    extent = (grids[0][0], grids[0][-1], grids[1][0], grids[1][-1])
    maximum = max(float(np.max(np.abs(y[..., selected, :]))), np.finfo(float).tiny)

    component_figure, component_axes = plt.subplots(
        nstates, count, figsize=(2.0 * count, 1.85 * nstates),
        sharex=True, sharey=True, constrained_layout=True,
    )
    component_axes = np.atleast_2d(component_axes)
    image = None
    for state in range(nstates):
        for column, channel in enumerate(selected):
            axis = component_axes[state, column]
            image = axis.imshow(
                y[:, :, channel, state].T,
                origin="lower", extent=extent, aspect="auto",
                cmap="RdBu_r", vmin=-maximum, vmax=maximum,
                interpolation="nearest",
            )
            add_samples(axis, grids, points)
            if state == 0:
                axis.set_title(rf"$k={channel}$", fontsize=9)
            if column == 0:
                axis.set_ylabel(rf"$Y_{{k,{state}}}$" + "\n" + r"$Q_1$")
            if state == nstates - 1:
                axis.set_xlabel(r"$Q_0$")
            axis.tick_params(labelsize=7)
    component_figure.colorbar(
        image, ax=component_axes, shrink=0.72, label=r"$Y_{k\alpha}(\mathbf{R})$"
    )

    gram = np.einsum("...ra,...rb->...ab", y, y, optimize=True)
    defect = np.linalg.norm(gram - np.eye(nstates), axis=(-2, -1))
    column_norms = np.linalg.norm(y, axis=-2)
    singular_values = np.linalg.svd(y, compute_uv=False)
    fields = [
        *(column_norms[..., state] for state in range(nstates)),
        *(singular_values[..., state] for state in range(nstates)),
        defect,
    ]
    labels = [
        *(rf"$\|Y_{{:{state}}}\|_2$" for state in range(nstates)),
        *(rf"$\sigma_{{{state + 1}}}(Y)$" for state in range(nstates)),
        r"$\|Y^T Y-I\|_F$",
    ]
    summary_figure, summary_axes = plt.subplots(
        3, 3, figsize=(8.6, 8.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for axis, field, label in zip(summary_axes.flat, fields, labels):
        positive = field[field > 0.0]
        use_log = label == r"$\|Y^T Y-I\|_F$" and len(positive)
        norm = (
            LogNorm(vmin=max(float(np.min(positive)), 1.0e-4), vmax=float(np.max(field)))
            if use_log else None
        )
        image = axis.imshow(
            field.T, origin="lower", extent=extent, aspect="auto",
            cmap="viridis", norm=norm, interpolation="nearest",
        )
        add_samples(axis, grids, points)
        axis.set_title(label, fontsize=10)
        summary_figure.colorbar(image, ax=axis, shrink=0.74)
    for axis in summary_axes[-1]:
        axis.set_xlabel(r"$Q_0$")
    for axis in summary_axes[:, 0]:
        axis.set_ylabel(r"$Q_1$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    component_path = args.output.with_name(args.output.name + "_components")
    summary_path = args.output.with_name(args.output.name + "_invariants")
    component_figure.savefig(component_path.with_suffix(".png"), dpi=300)
    component_figure.savefig(component_path.with_suffix(".pdf"))
    summary_figure.savefig(summary_path.with_suffix(".png"), dpi=300)
    summary_figure.savefig(summary_path.with_suffix(".pdf"))
    plt.close(component_figure)
    plt.close(summary_figure)
    np.savez(
        args.output.with_suffix(".npz"),
        grids=np.asarray(grids), y=y, selected_channels=selected,
        column_norms=column_norms, singular_values=singular_values,
        orthogonality_defect=defect,
    )
    print(json.dumps({
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "selected_channels": selected.tolist(),
        "minimum_orthogonality_defect": float(np.min(defect)),
        "median_orthogonality_defect": float(np.median(defect)),
        "maximum_orthogonality_defect": float(np.max(defect)),
        "components": str(component_path.with_suffix(".png")),
        "invariants": str(summary_path.with_suffix(".png")),
        "data": str(args.output.with_suffix(".npz")),
    }, indent=2))


if __name__ == "__main__":
    main()
