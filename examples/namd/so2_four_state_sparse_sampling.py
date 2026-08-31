#!/usr/bin/env python3
"""Benchmark sparse matrix TT-cross on the four-state SO2 atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.so2_sparse_link_benchmark import AlignedOracle
from pyqed.ldr.ttfit import fit_aligned, fit_block_cross, fit_svd, grid_links


class CountingOracle(AlignedOracle):
    def __init__(self, energy, links):
        super().__init__(energy, links)
        self.points = set()
        self.pairs = set()

    def hamiltonian_many(self, indices):
        self.points.update(tuple(index) for index in indices)
        return super().hamiltonian_many(indices)

    def overlap_many(self, pairs):
        pairs = tuple((tuple(left), tuple(right)) for left, right in pairs)
        for left, right in pairs:
            self.points.update((left, right))
            self.pairs.add((left, right))
        return super().overlap_many(pairs)


def relative(predicted, exact):
    return float(np.linalg.norm(predicted - exact) / np.linalg.norm(exact))


def permute_fields(energy, links, grids, order):
    energy = np.transpose(energy, (*order, 3, 4))
    grids = tuple(grids[axis] for axis in order)
    permuted_links = [None] * len(order)
    for new_axis, old_axis in enumerate(order):
        permuted_links[new_axis] = np.transpose(
            links[old_axis], (*order, 3, 4)
        )
    return energy, tuple(permuted_links), grids


def evaluate(energy, links, grids, rank, sweeps, validation, seed, method):
    oracle = CountingOracle(energy, links)
    if method == "independent":
        energy_fit, link_fits, info = fit_aligned(
            oracle,
            grids,
            energy.shape[-1],
            max_rank=rank,
            degrees=8,
            sweeps=sweeps,
            rtol=1.0e-3,
            validation=validation,
            seed=seed,
            start_rank=1,
            kick_rank=1,
        )
    elif method == "shared":
        energy_fit, link_fits, info = fit_block_cross(
            oracle,
            grids,
            energy.shape[-1],
            rank=rank,
            degrees=8,
            sweeps=sweeps,
            rtol=1.0e-3,
            validation=validation,
            seed=seed,
            start_rank=1,
            kick_rank=1,
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([values.ravel() for values in mesh], axis=1)
    errors = [
        relative(
            energy_fit.predict(coordinates).reshape(energy.shape),
            energy,
        )
    ]
    predicted = grid_links(link_fits, grids)
    for axis, exact in enumerate(links):
        values = np.asarray(
            [predicted[(axis, index)] for index in np.ndindex(exact.shape[:-2])]
        ).reshape(exact.shape)
        errors.append(relative(values, exact))
    return {
        "method": method,
        "rank": rank,
        "sampled_geometries": len(oracle.points),
        "sampled_pairs": len(oracle.pairs),
        "selected_vertices": info.get("selected_vertices"),
        "energy_error": errors[0],
        "link_errors": errors[1:],
        "worst_link_error": max(errors[1:]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fields",
        type=Path,
        default=Path(
            "/private/tmp/so2_cas4state_three_patch_9x9x9/procrustes_gauge.npz"
        ),
    )
    parser.add_argument(
        "--grids",
        type=Path,
        default=Path("/private/tmp/so2_9x9x9_grids.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_cas4state_sparse_sampling"),
    )
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.fields, allow_pickle=False) as archive:
        energy = np.asarray(archive["aligned_local_hamiltonian"])
        links = tuple(np.asarray(archive[f"link_{axis}"]) for axis in range(3))
    with np.load(args.grids, allow_pickle=False) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
    order = (0, 2, 1)
    energy, links, grids = permute_fields(energy, links, grids, order)

    settings = ((2, 3, 24), (4, 5, 32), (6, 7, 48), (8, 8, 64))
    records = []
    for method in ("independent", "shared"):
        records.extend(
            evaluate(
                energy,
                links,
                grids,
                rank,
                sweeps,
                validation,
                args.seed,
                method,
            )
            for rank, sweeps, validation in settings
        )
    floors = []
    for values in (energy, *links):
        _cores, _fitted, info = fit_svd(
            values.reshape(*values.shape[:-2], -1),
            24,
        )
        floors.append(info["relative_error"])
    summary = {
        "method": "independent versus shared block matrix TT-cross",
        "coordinate_order": ["qs", "qa", "theta"],
        "grid": list(energy.shape[:-2]),
        "full_grid_geometries": int(np.prod(energy.shape[:-2])),
        "records": records,
        "rank24_full_grid_floors": {
            "energy": floors[0],
            "links": floors[1:],
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    styles = {
        "independent": ("Independent", "#999999", "o"),
        "shared": ("Shared block", "#0072B2", "s"),
    }
    for method, (label, color, marker) in styles.items():
        selected = [record for record in records if record["method"] == method]
        samples = np.asarray([record["sampled_geometries"] for record in selected])
        energy_error = 100.0 * np.asarray(
            [record["energy_error"] for record in selected]
        )
        link_error = 100.0 * np.asarray(
            [record["worst_link_error"] for record in selected]
        )
        axes[0].plot(
            samples, energy_error, color=color, marker=marker, lw=1.35, label=label
        )
        axes[1].plot(
            samples, link_error, color=color, marker=marker, lw=1.35, label=label
        )
        for axis, values in zip(axes, (energy_error, link_error)):
            for position, (x, y, record) in enumerate(zip(samples, values, selected)):
                if position not in {0, len(selected) - 1}:
                    continue
                offset = (5, 5) if method == "shared" else (5, -13)
                axis.annotate(
                    rf"$r={record['rank']}$",
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )
    axes[0].axhline(100.0 * floors[0], color="0.35", ls="--", lw=1.0)
    axes[1].axhline(100.0 * max(floors[1:]), color="0.35", ls="--", lw=1.0)
    for axis in axes:
        axis.set(
            xlabel="Unique electronic geometries",
            yscale="log",
        )
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    axes[0].set_ylabel(r"Relative $\bar E$ error (%)")
    axes[1].set_ylabel(r"Worst $\bar L_\mu$ error (%)")
    axes[0].legend(
        frameon=False,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(0.02, 0.28),
    )
    for label, axis in zip("ab", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
    figure_path = args.output_dir / "so2_four_state_sparse_sampling.png"
    figure.savefig(figure_path, dpi=400, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
