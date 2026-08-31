#!/usr/bin/env python3
"""Benchmark matrix-block CUR sampling of two-patch SO2 fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.so2_sparse_link_benchmark import AlignedOracle, rotate_links
from pyqed.ldr.ttfit import fit_cur, fit_svd, grid_links


def relative(predicted, exact):
    return float(np.linalg.norm(predicted - exact) / np.linalg.norm(exact))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=Path("/private/tmp/so2_one_patch_exact_9x9x9.npz"))
    parser.add_argument("--grids", type=Path, default=Path("/private/tmp/so2_9x9x9_grids.npz"))
    parser.add_argument("--single-gauge", type=Path, default=Path("/private/tmp/so2_cas6e6o_631gstar_procrustes_gauge_9x9x9/procrustes_gauge.npz"))
    parser.add_argument("--two-gauge", type=Path, default=Path("/private/tmp/so2_cas6e6o_631gstar_procrustes_two_patch_9x9x9/procrustes_gauge.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/so2_cur_rank24"))
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--slabs", type=int, default=4)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.fields) as archive:
        one_links = tuple(np.asarray(archive[f"link_{axis}"]) for axis in range(3))
    with np.load(args.grids) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
    with np.load(args.single_gauge) as archive:
        single_gauge = np.asarray(archive["gauge"])
    with np.load(args.two_gauge) as archive:
        two_gauge = np.asarray(archive["gauge"])
        energy = np.asarray(archive["aligned_local_hamiltonian"])
    links = rotate_links(one_links, single_gauge, two_gauge)

    energy_fit, link_fits, info = fit_cur(
        AlignedOracle(energy, links),
        grids,
        energy.shape[-1],
        rank=args.rank,
        degrees=8,
        axis=args.axis,
        slabs=args.slabs,
        probes=args.probes,
        seed=args.seed,
    )
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([values.reshape(-1) for values in mesh], axis=1)
    energy_error = relative(energy_fit.predict(coordinates).reshape(energy.shape), energy)
    fitted_links = grid_links(link_fits, grids)
    link_errors = []
    for axis, exact in enumerate(links):
        edge_shape = list(energy.shape[:-2])
        edge_shape[axis] -= 1
        predicted = np.asarray(
            [fitted_links[(axis, index)] for index in np.ndindex(tuple(edge_shape))]
        ).reshape(exact.shape)
        link_errors.append(relative(predicted, exact))

    floor = []
    for values in (energy, *links):
        _cores, _fitted, item = fit_svd(
            values.reshape(*values.shape[:-2], values.shape[-1] ** 2),
            args.rank,
        )
        floor.append(item["relative_error"])
    full_links = [int(np.prod(values.shape[:-2])) for values in links]
    sampled_links = [item["sampled_links"] for item in info["links"]]
    summary = {
        "method": "matrix-block CUR + TT-SVD",
        "chart": "two-patch Procrustes",
        "grid": list(energy.shape[:-2]),
        "rank": args.rank,
        "cur_axis": args.axis,
        "slabs": args.slabs,
        "probes": args.probes,
        "energy_error": energy_error,
        "link_errors": link_errors,
        "full_grid_tt_svd_errors": floor,
        "unique_geometries": info["unique_geometries"],
        "full_geometries": int(np.prod(energy.shape[:-2])),
        "sampled_links": sampled_links,
        "full_links": full_links,
        "selected_slabs": [list(item["slabs"]) for item in info["links"]],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    labels = (r"$\bar E$", r"$\bar L_{q_s}$", r"$\bar L_\theta$", r"$\bar L_{q_a}$")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    positions = np.arange(4)
    width = 0.34
    axes[0].bar(positions - width / 2, 100 * np.asarray([energy_error, *link_errors]), width, color="#0072B2", label="Block CUR")
    axes[0].bar(positions + width / 2, 100 * np.asarray(floor), width, color="#D55E00", label="Full-grid TT-SVD")
    axes[0].set(xticks=positions, xticklabels=labels, yscale="log", ylabel="Relative error (%)")
    axes[0].legend(frameon=False, fontsize=8)

    fractions = [info["unique_geometries"] / np.prod(energy.shape[:-2])]
    fractions += [sample / total for sample, total in zip(sampled_links, full_links)]
    axes[1].bar(positions, 100 * np.asarray(fractions), color=("#555555", "#0072B2", "#D55E00", "#009E73"))
    axes[1].set(xticks=positions, xticklabels=("Geometries", *labels[1:]), ylabel="Sampled fraction (%)", ylim=(0, 105))
    for label, axis in zip("ab", axes):
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
    figure_path = args.output_dir / "so2_cur_sampling.png"
    figure.savefig(figure_path, dpi=350)
    figure.savefig(figure_path.with_suffix(".pdf"))
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
