#!/usr/bin/env python3
"""Benchmark sparse FunctionalTT fits of cached SO2 Procrustes links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr.ttfit import fit_sparse, grid_links


class AlignedOracle:
    def __init__(self, energy, links):
        self.energy = np.asarray(energy)
        self.links = tuple(np.asarray(values) for values in links)

    def hamiltonian_many(self, indices):
        return np.asarray([self.energy[tuple(index)] for index in indices])

    def overlap_many(self, pairs):
        blocks = []
        for left, right in pairs:
            left = tuple(left)
            right = tuple(right)
            delta = np.asarray(right) - left
            axis = int(np.flatnonzero(delta)[0])
            blocks.append(
                self.links[axis][left]
                if delta[axis] > 0
                else self.links[axis][right].conj().T
            )
        return np.asarray(blocks)


def relative(predicted, exact):
    return float(np.linalg.norm(predicted - exact) / np.linalg.norm(exact))


def rotate_links(one_patch, primary_gauge, target_gauge):
    output = []
    for axis, values in enumerate(one_patch):
        rotated = np.empty_like(values)
        for left in np.ndindex(values.shape[:-2]):
            right = list(left)
            right[axis] += 1
            right = tuple(right)
            raw = (
                primary_gauge[left]
                @ values[left]
                @ primary_gauge[right].conj().T
            )
            rotated[left] = (
                target_gauge[left].conj().T @ raw @ target_gauge[right]
            )
        output.append(rotated)
    return tuple(output)


def evaluate(name, energy, links, grids, budgets, args):
    shape = energy.shape[:-2]
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([values.reshape(-1) for values in mesh], axis=1)
    records = []
    for initial, validation, rounds in budgets:
        started = time.perf_counter()
        energy_fit, link_fits, info = fit_sparse(
            AlignedOracle(energy, links),
            grids,
            energy.shape[-1],
            rank=args.rank,
            degrees=args.degree,
            initial=initial,
            validation=validation,
            rounds=rounds,
            rtol=args.rtol,
            sweeps=args.sweeps,
            seed=args.seed,
            regularization=args.regularization,
        )
        predicted_energy = energy_fit.predict(coordinates).reshape(energy.shape)
        predicted_links = grid_links(link_fits, grids)
        errors = []
        for axis, exact in enumerate(links):
            edge_shape = list(shape)
            edge_shape[axis] -= 1
            predicted = np.asarray(
                [
                    predicted_links[(axis, index)]
                    for index in np.ndindex(tuple(edge_shape))
                ]
            ).reshape(exact.shape)
            errors.append(relative(predicted, exact))
        record = {
            "chart": name,
            "budget": [initial, validation, rounds],
            "geometries": info["energy_samples"],
            "link_samples": list(info["link_samples"]),
            "energy_error": relative(predicted_energy, energy),
            "link_errors": errors,
            "worst_link_error": max(errors),
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(record, flush=True)
    return records


def plot(path, records, total):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    styles = {"single patch": ("#0072B2", "o"), "two patch": ("#D55E00", "s")}
    for chart, (color, marker) in styles.items():
        selected = sorted(
            (record for record in records if record["chart"] == chart),
            key=lambda record: record["geometries"],
        )
        samples = [record["geometries"] for record in selected]
        axes[0].plot(
            samples,
            100.0 * np.asarray([record["energy_error"] for record in selected]),
            marker=marker,
            color=color,
            label=chart,
        )
        axes[1].plot(
            samples,
            100.0 * np.asarray([record["worst_link_error"] for record in selected]),
            marker=marker,
            color=color,
            label=chart,
        )
    for label, axis in zip("ab", axes):
        axis.set_xlabel(f"Sampled geometries (of {total})")
        axis.set_yscale("log")
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
    axes[0].set_ylabel(r"Relative $\bar E$ error (%)")
    axes[1].set_ylabel(r"Worst $\bar L_\mu$ error (%)")
    figure.savefig(path, dpi=350)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=Path("/private/tmp/so2_one_patch_exact_9x9x9.npz"))
    parser.add_argument("--grids", type=Path, default=Path("/private/tmp/so2_9x9x9_grids.npz"))
    parser.add_argument("--single-gauge", type=Path, default=Path("/private/tmp/so2_cas6e6o_631gstar_procrustes_gauge_9x9x9/procrustes_gauge.npz"))
    parser.add_argument("--two-gauge", type=Path, default=Path("/private/tmp/so2_cas6e6o_631gstar_procrustes_two_patch_9x9x9/procrustes_gauge.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/so2_sparse_link_benchmark"))
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.fields) as archive:
        single_energy = np.asarray(archive["energy"])
        single_links = tuple(np.asarray(archive[f"link_{axis}"]) for axis in range(3))
    with np.load(args.grids) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
    with np.load(args.single_gauge) as archive:
        primary_gauge = np.asarray(archive["gauge"])
    with np.load(args.two_gauge) as archive:
        two_gauge = np.asarray(archive["gauge"])
        two_energy = np.asarray(archive["aligned_local_hamiltonian"])
    two_links = rotate_links(single_links, primary_gauge, two_gauge)
    budgets = ((48, 24, 4), (96, 32, 4), (128, 32, 4), (128, 48, 4))
    records = evaluate("single patch", single_energy, single_links, grids, budgets, args)
    records += evaluate("two patch", two_energy, two_links, grids, budgets, args)
    summary = {
        "method": "SO2 sparse direct-link FunctionalTT sampling benchmark",
        "grid": list(single_energy.shape[:-2]),
        "rank": args.rank,
        "degree": args.degree,
        "regularization": args.regularization,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure = args.output_dir / "so2_sparse_link_sampling.png"
    plot(figure, records, int(np.prod(single_energy.shape[:-2])))
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
