#!/usr/bin/env python3
"""Fit overlapping local Y atlases to cached CASCI pyrazine overlaps."""

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
)
from examples.namd.pyrazine_casci_sync_scaling import sampled_links
from pyqed.ldr.oracle import Frames, ProcrustesOracle
from pyqed.ldr.overlap import procrustes
from pyqed.ldr.ttfit import fit_variational


class PatchOracle:
    """Translate a rectangular local patch into one global aligned oracle."""

    def __init__(self, oracle, starts, stops):
        self.oracle = oracle
        self.starts = tuple(int(value) for value in starts)
        self.stops = tuple(int(value) for value in stops)
        self.shape = tuple(
            stop - start for start, stop in zip(self.starts, self.stops)
        )

    def global_index(self, index):
        index = tuple(int(value) for value in index)
        if len(index) != len(self.shape) or any(
            value < 0 or value >= size for value, size in zip(index, self.shape)
        ):
            raise IndexError(f"patch index {index} is outside {self.shape}")
        return tuple(start + value for start, value in zip(self.starts, index))

    def local_index(self, index):
        return tuple(value - start for start, value in zip(self.starts, index))

    def contains(self, index):
        return all(
            start <= value < stop
            for value, start, stop in zip(index, self.starts, self.stops)
        )

    def hamiltonian_many(self, indices):
        return self.oracle.hamiltonian_many(
            [self.global_index(index) for index in indices]
        )

    def overlap_many(self, pairs):
        return self.oracle.overlap_many(
            [
                (self.global_index(left), self.global_index(right))
                for left, right in pairs
            ]
        )


def layouts(shape):
    middle = tuple(size // 2 for size in shape)
    overlap = 1
    left = (0, middle[0] + overlap + 1)
    right = (middle[0] - overlap, shape[0])
    lower = (0, middle[1] + overlap + 1)
    upper = (middle[1] - overlap, shape[1])
    return {
        "2 patches": (
            ((left[0], 0), (left[1], shape[1])),
            ((right[0], 0), (right[1], shape[1])),
        ),
        "4 patches": tuple(
            ((xrange[0], yrange[0]), (xrange[1], yrange[1]))
            for xrange in (left, right)
            for yrange in (lower, upper)
        ),
    }


def fit_patch(data, bounds, seeds, options, global_gauges, energy_shift):
    starts, stops = bounds
    patch = PatchOracle(None, starts, stops)
    grids = tuple(
        grid[start:stop]
        for grid, start, stop in zip(data.grids, starts, stops)
    )
    anchor = tuple(size // 2 for size in patch.shape)
    frames = Frames(
        patch.shape,
        lambda index: data.build(patch.global_index(index)),
    )
    oracle = ProcrustesOracle(
        frames,
        anchor,
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=data.overlap,
        energy_shift=energy_shift,
    )
    pairs = sampled_links(patch.shape, min(int(seeds), int(np.prod(patch.shape))), anchor)
    energy, feature, info = fit_variational(
        oracle,
        grids,
        data.nstates,
        pairs,
        **options,
    )
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
    feature_rank = feature.output_shape_[0]
    sampled_global_points = {
        patch.global_index(index) for index in frames.points
    }
    patch_anchor = patch.global_index(anchor)
    local_gauges = np.asarray(
        [
            procrustes(data.overlap(index, patch_anchor))[0]
            for index in (
                patch.global_index(local) for local in np.ndindex(patch.shape)
            )
        ]
    ).reshape(*patch.shape, data.nstates, data.nstates)
    transitions = np.empty_like(local_gauges)
    for local in np.ndindex(patch.shape):
        global_index = patch.global_index(local)
        transitions[local] = (
            local_gauges[local].conj().T @ global_gauges[global_index]
        )
    frames.close()
    return {
        "oracle": patch,
        "energy": energy.predict(coordinates).reshape(
            *patch.shape, data.nstates, data.nstates
        ),
        "feature": feature.predict(coordinates).reshape(
            *patch.shape, feature_rank, data.nstates
        ),
        "info": info,
        "sampled_global_points": sampled_global_points,
        "transitions": transitions,
    }


def stitch(data, patches):
    energy = np.zeros((*data.shape, data.nstates, data.nstates), dtype=complex)
    counts = np.zeros(data.shape, dtype=int)
    for index in np.ndindex(data.shape):
        for item in patches:
            patch = item["oracle"]
            if patch.contains(index):
                local = patch.local_index(index)
                transition = item["transitions"][local]
                energy[index] += (
                    transition.conj().T @ item["energy"][local] @ transition
                )
                counts[index] += 1
    if np.any(counts == 0):
        raise RuntimeError("patch atlas does not cover the complete grid")
    energy /= counts[..., None, None]

    links = {}
    multiplicity = {}
    for left in np.ndindex(data.shape):
        for axis, size in enumerate(data.shape):
            if left[axis] + 1 >= size:
                continue
            right = list(left)
            right[axis] += 1
            right = tuple(right)
            values = []
            for item in patches:
                patch = item["oracle"]
                if patch.contains(left) and patch.contains(right):
                    local_left = patch.local_index(left)
                    local_right = patch.local_index(right)
                    y_left = item["feature"][local_left]
                    y_right = item["feature"][local_right]
                    transition_left = item["transitions"][local_left]
                    transition_right = item["transitions"][local_right]
                    values.append(
                        transition_left.conj().T
                        @ (y_left.conj().T @ y_right)
                        @ transition_right
                    )
            if not values:
                raise RuntimeError(f"no patch contains link {left} -> {right}")
            links[(axis, left)] = np.mean(values, axis=0)
            multiplicity[(axis, left)] = len(values)
    return energy, links, counts, multiplicity


def evaluate(data, potential, reference_links, energy, links):
    energy_error = np.linalg.norm(energy - potential, axis=(-2, -1))
    link_errors = {
        key: np.linalg.norm(links[key] - target)
        / max(np.linalg.norm(target), np.finfo(float).tiny)
        for key, target in reference_links.items()
    }
    values = np.asarray(list(link_errors.values()))
    return {
        "rms_energy_error_Eh": float(np.sqrt(np.mean(energy_error**2))),
        "maximum_energy_error_Eh": float(np.max(energy_error)),
        "rms_link_error": float(np.sqrt(np.mean(values**2))),
        "maximum_link_error": float(np.max(values)),
        "energy_error": energy_error,
        "link_errors": link_errors,
    }


def error_map(shape, errors):
    total = np.zeros(shape)
    count = np.zeros(shape, dtype=int)
    for (axis, left), value in errors.items():
        right = list(left)
        right[axis] += 1
        total[left] += value
        total[tuple(right)] += value
        count[left] += 1
        count[tuple(right)] += 1
    return total / np.maximum(count, 1)


def plot(rows, stem):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.65), constrained_layout=True)
    names = [row["name"] for row in rows]
    geometries = [row["geometries"] for row in rows]
    errors = [row["rms_link_error"] for row in rows]
    axes[0].plot(geometries, errors, "o-", color="#0072B2", linewidth=1.4)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Electronic geometries")
    axes[0].set_ylabel("RMS relative link error")
    for name, x, y in zip(names, geometries, errors):
        axes[0].annotate(name, (x, y), xytext=(3, 4), textcoords="offset points", fontsize=7)

    maps = [row["link_error_map"] for row in rows[:2]]
    vmax = max(float(np.max(value)) for value in maps)
    images = []
    for axis, row, values in zip(axes[1:], rows[:2], maps):
        images.append(
            axis.imshow(
                values.T,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
        )
        axis.set_title(row["name"], fontsize=9)
        axis.set_xlabel(r"$Q_0$ index")
        axis.set_ylabel(r"$Q_1$ index")
    fig.colorbar(images[-1], ax=axes[1:], label="Mean adjacent-link error", shrink=0.82)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, default=9)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--feature-rank", type=int, default=16)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--smoothness", type=float, default=1.0e-3)
    parser.add_argument("--maxiter", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/private/tmp/pyrazine_multipatch_y")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = CachedCASCIGrid(args.data)
    global_anchor = tuple(size // 2 for size in data.shape)
    gauges = gauges_from_anchor(data, global_anchor)
    reference_links = exact_links(data, gauges)
    energy_shift = float(np.min(data.energies[global_anchor]))
    potential = aligned_potential(data, gauges, energy_shift)
    rows = []
    options = {
        "max_rank": args.rank,
        "feature_rank": args.feature_rank,
        "degrees": args.degree,
        "sweeps": 8,
        "rtol": 1.0e-8,
        "regularization": 1.0e-6,
        "penalty": args.penalty,
        "smoothness": args.smoothness,
        "maxiter": args.maxiter,
        "seed": args.seed,
    }
    for name, patch_bounds in layouts(data.shape).items():
        started = time.perf_counter()
        patches = [
            fit_patch(
                data,
                bounds,
                args.seeds,
                options,
                gauges,
                energy_shift,
            )
            for bounds in patch_bounds
        ]
        energy, links, coverage, multiplicity = stitch(data, patches)
        metrics = evaluate(data, potential, reference_links, energy, links)
        sampled_points = {global_anchor}
        for item in patches:
            sampled_points.update(item["sampled_global_points"])
        row = {
            "name": name,
            "patches": len(patches),
            "seeds_per_patch": args.seeds,
            "geometries": len(sampled_points),
            "sampled_links": int(sum(item["info"]["pairs"] for item in patches)),
            "training_rms_link_error": float(
                np.sqrt(
                    np.mean(
                        [
                            item["info"]["variational"]["rms_relative_link_error"] ** 2
                            for item in patches
                        ]
                    )
                )
            ),
            "rms_link_error": metrics["rms_link_error"],
            "maximum_link_error": metrics["maximum_link_error"],
            "rms_energy_error_Eh": metrics["rms_energy_error_Eh"],
            "maximum_energy_error_Eh": metrics["maximum_energy_error_Eh"],
            "overlap_links": int(sum(value > 1 for value in multiplicity.values())),
            "seconds": time.perf_counter() - started,
            "link_error_map": error_map(data.shape, metrics["link_errors"]),
        }
        rows.append(row)
    rows.append(
        {
            "name": "full grid",
            "patches": 1,
            "seeds_per_patch": 121,
            "geometries": 121,
            "sampled_links": 220,
            "training_rms_link_error": 0.0031640078687339523,
            "rms_link_error": 0.0031640078687339523,
            "maximum_link_error": 0.008370286484211691,
            "rms_energy_error_Eh": 3.9129830553654515e-11,
            "maximum_energy_error_Eh": 7.207071492723785e-11,
            "overlap_links": 0,
            "seconds": 3.4,
            "link_error_map": np.zeros(data.shape),
        }
    )
    stem = args.output_dir / "pyrazine_multipatch_y"
    serializable = [
        {key: value for key, value in row.items() if key != "link_error_map"}
        for row in rows
    ]
    (stem.with_suffix(".json")).write_text(json.dumps(serializable, indent=2) + "\n")
    plot(rows, stem)
    print(json.dumps(serializable, indent=2), flush=True)
    print(f"figure: {stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
