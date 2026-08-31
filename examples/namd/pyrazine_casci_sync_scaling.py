#!/usr/bin/env python3
"""Diagnose sparse global-Y synchronization on cached CASCI pyrazine data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.pyrazine_casci_abinitio_ttldr import (
    DEFAULT_DATA,
    CachedCASCIGrid,
    aligned_potential,
    exact_links,
    gauges_from_anchor,
    maximin_points,
)
from pyqed.ldr import AbInitioFit
from pyqed.ldr.ttfit import fit_variational


def evaluate(fit, data, potential, links):
    mesh = np.meshgrid(*data.grids, indexing="ij")
    coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
    fitted_energy = fit.energy.predict(coordinates).reshape(
        *data.shape, data.nstates, data.nstates
    )
    energy_error = np.linalg.norm(fitted_energy - potential, axis=(-2, -1))
    feature_rank = fit.feature.output_shape_[0]
    features = fit.feature.predict(coordinates).reshape(
        *data.shape, feature_rank, data.nstates
    )
    link_error = []
    for (axis, left), target in links.items():
        right = list(left)
        right[axis] += 1
        fitted = features[left].conj().T @ features[tuple(right)]
        link_error.append(
            np.linalg.norm(fitted - target) / max(np.linalg.norm(target), 1.0e-15)
        )
    return {
        "rms_energy_error_Eh": float(np.sqrt(np.mean(energy_error**2))),
        "maximum_energy_error_Eh": float(np.max(energy_error)),
        "rms_heldout_link_error": float(np.sqrt(np.mean(np.square(link_error)))),
        "maximum_heldout_link_error": float(np.max(link_error)),
    }


def sampled_links(shape, count, anchor):
    vertices = maximin_points(shape, count, anchor)
    seed_pairs = []
    for left in vertices:
        for axis, size in enumerate(shape):
            right = list(left)
            if left[axis] + 1 < size:
                right[axis] += 1
                pair = (left, tuple(right))
            else:
                right[axis] -= 1
                pair = (tuple(right), left)
            seed_pairs.append(pair)
    points = set(index for pair in seed_pairs for index in pair)
    pairs = []
    for left in points:
        for axis, size in enumerate(shape):
            if left[axis] + 1 >= size:
                continue
            right = list(left)
            right[axis] += 1
            right = tuple(right)
            if right in points:
                pairs.append((left, right))
    return tuple(dict.fromkeys(pairs))


def fit_case(
    data,
    count,
    *,
    degree,
    rank,
    feature_rank,
    penalty,
    smoothness,
    variational_maxiter,
    seed,
):
    anchor = tuple(size // 2 for size in data.shape)
    pairs = sampled_links(data.shape, count, anchor)
    started = time.perf_counter()
    with AbInitioFit(
        data.grids,
        data.nstates,
        data.build,
        anchor=anchor,
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=data.overlap,
        energy_shift=None,
    ) as fit:
        fit.energy, fit.feature, fit.info = fit_variational(
            fit.oracle,
            data.grids,
            data.nstates,
            pairs,
            max_rank=rank,
            degrees=degree,
            sweeps=8,
            regularization=1.0e-6,
            feature_rank=feature_rank,
            penalty=penalty,
            smoothness=smoothness,
            maxiter=variational_maxiter,
            rtol=1.0e-8,
            seed=seed,
        )
        gauges = gauges_from_anchor(data, anchor)
        potential = aligned_potential(data, gauges, fit.energy_shift)
        metrics = evaluate(fit, data, potential, exact_links(data, gauges))
        variational = fit.info["variational"]
        metrics.update(
            {
                "seed_vertices": int(count),
                "geometries": int(fit.info["unique_geometries"]),
                "geometry_fraction": fit.info["geometry_fraction"],
                "sampled_links": int(fit.info["pairs"]),
                "rms_variational_link_error": float(
                    variational["rms_relative_link_error"]
                ),
                "energy_training_error": float(fit.info["energy_training_error"]),
                "seconds": time.perf_counter() - started,
            }
        )
    return metrics


def full_case(data, *, seed):
    anchor = tuple(size // 2 for size in data.shape)
    started = time.perf_counter()
    with AbInitioFit(
        data.grids,
        data.nstates,
        data.build,
        anchor=anchor,
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=data.overlap,
        energy_shift=None,
    ) as fit:
        fit.run(
            representation="features",
            rank=16,
            degrees=10,
            sweeps=8,
            validation=64,
            rtol=1.0e-9,
            feature_rank=8,
            feature_penalty=50.0,
            feature_maxiter=1000,
            seed=seed,
            start_rank=2,
            kick_rank=2,
        )
        gauges = gauges_from_anchor(data, anchor)
        potential = aligned_potential(data, gauges, fit.energy_shift)
        metrics = evaluate(fit, data, potential, exact_links(data, gauges))
        synchronization = fit.info["feature"]["optimization"]
        metrics.update(
            {
                "geometries": int(np.prod(data.shape)),
                "seed_vertices": int(np.prod(data.shape)),
                "geometry_fraction": 1.0,
                "sampled_links": int(synchronization["links"]),
                "rms_variational_link_error": float(
                    synchronization["rms_relative_link_error"]
                ),
                "feature_training_error": float(
                    fit.info["feature"]["relative_interpolation_error"]
                ),
                "energy_training_error": float(
                    fit.info["energy"]["validation_rms_error"]
                ),
                "seconds": time.perf_counter() - started,
            }
        )
    return metrics


def plot(rows, stem):
    counts = np.asarray([row["geometries"] for row in rows])
    sync = np.asarray([row["rms_variational_link_error"] for row in rows])
    heldout = np.asarray([row["rms_heldout_link_error"] for row in rows])
    energy = np.asarray([row["rms_energy_error_Eh"] for row in rows])
    colors = {"sync": "#0072B2", "heldout": "#D55E00", "energy": "#009E73"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].plot(
        counts, sync, "o-", color=colors["sync"], label="Sampled links"
    )
    axes[0].plot(
        counts, heldout, "s--", color=colors["heldout"], label="All nearest links"
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Electronic geometries")
    axes[0].set_ylabel(r"RMS relative link error")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(counts, energy, "o-", color=colors["energy"])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Electronic geometries")
    axes[1].set_ylabel(r"RMS $\bar E$ error / $E_h$")
    for label, axis in zip("ab", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--counts", type=int, nargs="+", default=(25, 49))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/private/tmp/pyrazine_sync_scaling")
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--feature-rank", type=int, default=16)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--smoothness", type=float, default=1.0e-3)
    parser.add_argument("--variational-maxiter", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = CachedCASCIGrid(args.data)
    rows = [
        fit_case(
            data,
            count,
            degree=args.degree,
            rank=args.rank,
            feature_rank=args.feature_rank,
            penalty=args.penalty,
            smoothness=args.smoothness,
            variational_maxiter=args.variational_maxiter,
            seed=args.seed,
        )
        for count in args.counts
    ]
    rows.append(full_case(data, seed=args.seed))
    rows.sort(key=lambda row: row["geometries"])
    stem = args.output_dir / "pyrazine_sync_scaling"
    (stem.with_suffix(".json")).write_text(json.dumps(rows, indent=2) + "\n")
    plot(rows, stem)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"figure: {stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
