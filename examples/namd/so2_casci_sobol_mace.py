#!/usr/bin/env python3
"""Benchmark nested Sobol sampling for spin-pure SO2 CASCI/MACE fields.

The first structure is the symmetric equilibrium-like anchor.  The remaining
structures form one scrambled Sobol sequence in the exchange-reduced domain
``r_long >= r_short``.  Consequently every larger training set reuses all
earlier electronic-structure calculations.  Wavefunction overlaps are sampled
only on a sparse nearest-neighbor graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from examples.namd.generate_so2_casci_singlets import (
    electronic_metadata,
    electronic_structure,
    geometry,
    require_spin_pure_singlets,
    validate_electronic_metadata,
)
from pyqed.units import au2ev
from pyqed.ldr.oracle import synchronize_features
from pyqed.ml import MACE, MACEStateModel


EV_PER_HARTREE = au2ev


def sobol_coordinates(count, bounds, seed):
    """Return an anchor plus a nested Sobol prefix on the O-exchange quotient."""
    count = int(count)
    if count < 2:
        raise ValueError("at least two samples are required")
    r_min, r_max, theta_min, theta_max = map(float, bounds)
    needed = count - 1
    power = int(np.ceil(np.log2(needed)))
    unit = qmc.Sobol(3, scramble=True, seed=int(seed)).random_base2(power)[:needed]
    bonds = r_min + (r_max - r_min) * unit[:, :2]
    r_short = np.min(bonds, axis=1)
    r_long = np.max(bonds, axis=1)
    theta = theta_min + (theta_max - theta_min) * unit[:, 2]
    anchor = np.asarray(
        [[0.5 * (r_min + r_max), 0.5 * (r_min + r_max),
          0.5 * (theta_min + theta_max)]]
    )
    return np.concatenate(
        (anchor, np.column_stack((r_long, r_short, theta))), axis=0
    )


def invariant_coordinates(coordinates, bounds):
    """Scale symmetric stretch, unsigned asymmetric stretch, and bend to [0, 1]."""
    coordinates = np.asarray(coordinates, dtype=float)
    r_min, r_max, theta_min, theta_max = map(float, bounds)
    root_two = np.sqrt(2.0)
    values = np.column_stack(
        (
            (coordinates[:, 0] + coordinates[:, 1]) / root_two,
            np.abs(coordinates[:, 0] - coordinates[:, 1]) / root_two,
            coordinates[:, 2],
        )
    )
    lower = np.asarray((root_two * r_min, 0.0, theta_min))
    upper = np.asarray((root_two * r_max, (r_max - r_min) / root_two, theta_max))
    return (values - lower) / (upper - lower)


def sparse_overlap_graph(coordinates, bounds, neighbors):
    """Build a connected local graph with O(kN) overlap blocks."""
    scaled = invariant_coordinates(coordinates, bounds)
    distances = cdist(scaled, scaled)
    np.fill_diagonal(distances, np.inf)
    neighbors = min(int(neighbors), len(coordinates) - 1)
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    pairs = set()
    for left in range(len(coordinates)):
        for right in np.argpartition(distances[left], neighbors - 1)[:neighbors]:
            pairs.add(tuple(sorted((left, int(right)))))
    finite = distances.copy()
    np.fill_diagonal(finite, 0.0)
    tree = minimum_spanning_tree(finite).tocoo()
    pairs.update(tuple(sorted((int(i), int(j)))) for i, j in zip(tree.row, tree.col))
    pairs = np.asarray(sorted(pairs), dtype=int)
    lengths = np.linalg.norm(scaled[pairs[:, 0]] - scaled[pairs[:, 1]], axis=1)
    return pairs, lengths


def generate_dataset(coordinates, pairs, options, output):
    models = []
    energies = []
    spin_square = []
    for count, coordinate in enumerate(coordinates, start=1):
        model = electronic_structure(*coordinate, options)
        models.append(model.frame())
        energies.append(np.asarray(model.e_tot, dtype=float))
        spin_square.append(
            [model.spin_square(state) for state in range(options.nstates)]
        )
        if count == 1 or count % options.progress_every == 0 or count == len(coordinates):
            print(
                f"[CASCI] {count}/{len(coordinates)}, "
                f"E0={energies[-1][0]:.10f} Eh, "
                f"max |S2|={np.max(np.abs(spin_square[-1])):.2e}",
                flush=True,
            )
    overlaps = np.asarray([models[left].overlap(models[right]) for left, right in pairs])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        coordinates=coordinates,
        geometries=np.asarray([geometry(*point) for point in coordinates]),
        energies=np.asarray(energies),
        spin_square=np.asarray(spin_square),
        overlap_pairs=pairs,
        overlap_values=overlaps,
        source=np.asarray(
            f"SO2 nested Sobol, native {options.basis} "
            f"CASCI({options.nelecas}e,{options.ncas}o), Ms=0, multiplicity=1"
        ),
        **electronic_metadata(options),
    )
    return np.asarray(energies), np.asarray(spin_square), overlaps


def validation_data(path, options=None):
    with np.load(path, allow_pickle=False) as archive:
        if options is not None:
            validate_electronic_metadata(archive, options, label="SO2 reference")
        grids = tuple(np.asarray(archive[name]) for name in ("r1", "r2", "theta"))
        energies = np.asarray(archive["energies"])
        spin_square = np.asarray(archive["spin_square"])
        links = tuple(np.asarray(archive[f"links_{axis}"]) for axis in range(3))
    require_spin_pure_singlets(spin_square)
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
    return grids, coordinates, energies.reshape(len(coordinates), -1), links


def graph_link_values(overlap_pairs, overlap_values, pairs):
    lookup = {
        tuple(map(int, pair)): value
        for pair, value in zip(overlap_pairs, overlap_values)
    }
    return np.asarray([lookup[tuple(map(int, pair))] for pair in pairs])


def prune_overlap_graph(pairs, lengths, values, npoints, minimum_singular_value):
    """Keep reliable local overlaps, adding the best edges needed for connectivity."""
    singular_values = np.linalg.svd(values, compute_uv=False).min(axis=1)
    selected = singular_values >= float(minimum_singular_value)
    parent = np.arange(int(npoints))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        left_root, right_root = root(int(left)), root(int(right))
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    for left, right in pairs[selected]:
        join(left, right)
    order = np.lexsort((lengths, -singular_values))
    for edge in order:
        if selected[edge]:
            continue
        if join(*pairs[edge]):
            selected[edge] = True
    if len({root(index) for index in range(npoints)}) != 1:
        raise RuntimeError("overlap graph cannot be connected")
    return pairs[selected], lengths[selected], values[selected], singular_values[selected]


def synchronize_graph(pairs, values, npoints, feature_rank, args):
    blocks = {
        ((int(left),), (int(right),)): value
        for (left, right), value in zip(pairs, values)
    }

    class SampledLinks:
        shape = (int(npoints),)

        @staticmethod
        def overlap_many(requested):
            output = []
            for left, right in requested:
                key = (tuple(left), tuple(right))
                if key in blocks:
                    output.append(blocks[key])
                else:
                    output.append(blocks[(key[1], key[0])].conj().T)
            return np.asarray(output)

    _features, info = synchronize_features(
        SampledLinks(),
        tuple((index,) for index in range(npoints)),
        tuple(((int(left),), (int(right),)) for left, right in pairs),
        feature_rank,
        anchor=(0,),
        penalty=10.0,
        smoothness=args.smoothness,
        maxiter=args.sync_steps,
        seed=args.seed,
    )
    return info


def validation_metrics(fit, coordinates, energies, links, shift, model):
    if model == "energy":
        prediction = fit.predict(
            [geometry(*point) for point in coordinates],
            [(8, 16, 8)] * len(coordinates),
            molecular_charges=np.zeros(len(coordinates)),
            multiplicities=np.ones(len(coordinates)),
        )
        levels = np.asarray(prediction["energies"])
    else:
        prediction = fit.predict_covariant(coordinates)
        levels = np.linalg.eigvalsh(prediction["energy"])
    reference_levels = energies - shift
    level_error = np.abs(levels - reference_levels) * EV_PER_HARTREE
    gap_error = np.abs(np.diff(levels, axis=1) - np.diff(reference_levels, axis=1))
    gap_error *= EV_PER_HARTREE
    metrics = {
        "eigenvalue_mae_ev": float(np.mean(level_error)),
        "eigenvalue_max_ev": float(np.max(level_error)),
        "gap_mae_ev": float(np.mean(gap_error)),
    }
    if model == "energy":
        return metrics
    features = prediction["feature"]
    grid_shape = tuple(reference.shape[axis] + 1 for axis, reference in enumerate(links))
    link_errors = []
    for axis, reference in enumerate(links):
        errors = []
        for left in np.ndindex(reference.shape[:-2]):
            right = list(left)
            right[axis] += 1
            left_flat = np.ravel_multi_index(left, grid_shape)
            right_flat = np.ravel_multi_index(tuple(right), grid_shape)
            predicted = (
                features[left_flat].conj().T @ features[right_flat]
            )
            predicted_sv = np.linalg.svd(predicted, compute_uv=False)
            reference_sv = np.linalg.svd(reference[left], compute_uv=False)
            errors.append(
                np.linalg.norm(predicted_sv - reference_sv)
                / max(np.linalg.norm(reference_sv), np.finfo(float).tiny)
            )
        link_errors.append(np.asarray(errors))
    metrics.update({
        "link_singular_value_rms": float(
            np.sqrt(np.mean(np.concatenate(link_errors) ** 2))
        ),
        "link_singular_value_axis_rms": [
            float(np.sqrt(np.mean(values**2))) for values in link_errors
        ],
    })
    return metrics


def fit_case(
    coordinates, energies, overlap_pairs, overlap_values,
    count, grids, bounds, args,
):
    import torch

    torch.manual_seed(int(args.seed))
    points = coordinates[:count]
    shift = float(energies[0, 0])
    hamiltonians = np.asarray(
        [np.diag(values - shift) for values in energies[:count]]
    )
    pairs, lengths = sparse_overlap_graph(points, bounds, args.neighbors)
    link_values = graph_link_values(overlap_pairs, overlap_values, pairs)
    pairs, lengths, link_values, singular_values = prune_overlap_graph(
        pairs, lengths, link_values, count, args.minimum_link_singular_value
    )
    synchronization = synchronize_graph(
        pairs, link_values, count, args.feature_rank, args
    )
    encoder_options = {
        "channels": args.channels,
        "max_ell": args.max_ell,
        "interactions": args.interactions,
        "correlation": args.correlation,
        "radial_basis": args.radial_basis,
        "radial_mlp": (args.head_width, args.head_width),
        "cutoff": 7.0,
    }
    if args.model == "energy":
        fit = MACEStateModel(
            ("O", "S"), args.nstates,
            hidden=(args.head_width, args.head_width),
            geometry_units="bohr",
            **encoder_options,
        ).fit(
            [geometry(*point) for point in points],
            [(8, 16, 8)] * count,
            hamiltonians,
            molecular_charges=np.zeros(count),
            multiplicities=np.ones(count),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    else:
        fit = MACE(
            grids,
            ("O", "S", "O"),
            lambda coordinate: geometry(*coordinate),
            args.nstates,
            chart_features=args.chart_features,
            geometry_units="bohr",
            **encoder_options,
        ).fit_y(
            (points, hamiltonians),
            points,
            pairs,
            link_values,
            feature_rank=args.feature_rank,
            feature_objective=args.feature_objective,
            hidden=(args.head_width, args.head_width),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            link_weight=args.link_weight,
            isometry_weight=1.0,
            smoothness=args.smoothness,
            sync_steps=args.sync_steps,
            seed=args.seed,
            distill=False,
        )
    return fit, shift, pairs, lengths, singular_values, synchronization


def plot_results(rows, coordinates, bounds, output):
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), constrained_layout=True)
    scaled = invariant_coordinates(coordinates, bounds)
    scatter = axes[0].scatter(
        scaled[:, 0], scaled[:, 1], c=scaled[:, 2], cmap="viridis", s=28,
        edgecolor="white", linewidth=0.25,
    )
    axes[0].scatter(scaled[0, 0], scaled[0, 1], marker="*", s=115,
                    facecolor="none", edgecolor="black", linewidth=1.0)
    axes[0].set(xlabel=r"scaled $q_s$", ylabel=r"scaled $|q_a|$")
    colorbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label(r"scaled $\theta$")

    counts = np.asarray([row["samples"] for row in rows])
    axes[1].plot(
        counts, [row["eigenvalue_mae_ev"] for row in rows], "o-",
        color=colors[0], label="Energy levels",
    )
    axes[1].plot(
        counts, [row["gap_mae_ev"] for row in rows], "s-",
        color=colors[1], label="Energy gaps",
    )
    axes[1].set(xlabel="CASCI geometries", ylabel="Validation MAE (eV)")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        counts, [row["synchronization"]["rms_relative_link_error"] for row in rows], "o-",
        color=colors[2], label=r"$Y$ synchronization",
    )
    axes[2].plot(
        counts, [row["fill_distance"] for row in rows], "s--",
        color=colors[1], label="Fill distance",
    )
    axes[2].set(xlabel="CASCI geometries", ylabel="Dimensionless RMS / distance")
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False, fontsize=8)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes,
                  va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=(17, 33, 65))
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--minimum-link-singular-value", type=float, default=0.4)
    parser.add_argument("--r-min", type=float, default=2.68)
    parser.add_argument("--r-max", type=float, default=2.92)
    parser.add_argument("--theta-min-deg", type=float, default=110.0)
    parser.add_argument("--theta-max-deg", type=float, default=130.0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=6)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument("--spin-root-cushion", type=int, default=8)
    parser.add_argument("--scf-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--feature-rank", type=int, default=15)
    parser.add_argument(
        "--feature-objective", choices=("fixed", "subspace", "links-only"),
        default="fixed",
    )
    parser.add_argument(
        "--chart-features", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--max-ell", type=int, default=2)
    parser.add_argument("--interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--radial-basis", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=64)
    parser.add_argument("--link-weight", type=float, default=1.0)
    parser.add_argument("--smoothness", type=float, default=1.0e-4)
    parser.add_argument("--sync-steps", type=int, default=3000)
    parser.add_argument("--model", choices=("energy", "endpoint"), default="energy")
    parser.add_argument(
        "--reference", type=Path,
        default=Path("/private/tmp/so2_casci_singlet_5x5x5.npz"),
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("/private/tmp/so2_casci_sobol_65.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/so2_casci_sobol_mace.png"),
    )
    args = parser.parse_args()
    counts = tuple(sorted(set(map(int, args.counts))))
    bounds = (
        args.r_min, args.r_max, np.deg2rad(args.theta_min_deg),
        np.deg2rad(args.theta_max_deg),
    )
    coordinates = sobol_coordinates(max(counts), bounds, args.seed)
    graph_pairs = []
    for count in counts:
        pairs, _lengths = sparse_overlap_graph(
            coordinates[:count], bounds, args.neighbors
        )
        graph_pairs.extend(map(tuple, pairs))
    graph_pairs = np.asarray(sorted(set(graph_pairs)), dtype=int)
    options = SimpleNamespace(**vars(args))
    if args.dataset.is_file():
        with np.load(args.dataset, allow_pickle=False) as archive:
            validate_electronic_metadata(archive, args, label="SO2 Sobol cache")
            cached_coordinates = np.asarray(archive["coordinates"])
            if cached_coordinates.shape != coordinates.shape or not np.allclose(
                cached_coordinates, coordinates
            ):
                raise ValueError("cached Sobol dataset does not match this design")
            energies = np.asarray(archive["energies"])
            spin_square = np.asarray(archive["spin_square"])
            overlap_pairs = np.asarray(archive["overlap_pairs"])
            overlap_values = np.asarray(archive["overlap_values"])
            available = {tuple(map(int, pair)) for pair in overlap_pairs}
            if any(tuple(map(int, pair)) not in available for pair in graph_pairs):
                raise ValueError("cached Sobol overlaps do not cover this design")
        print(f"[cache] restored {len(coordinates)} CASCI structures from {args.dataset}")
    else:
        energies, spin_square, overlap_values = generate_dataset(
            coordinates, graph_pairs, options, args.dataset
        )
        overlap_pairs = graph_pairs
    require_spin_pure_singlets(spin_square)
    grids, validation_coordinates, validation_energies, validation_links = (
        validation_data(args.reference, args)
    )
    scaled_validation = invariant_coordinates(validation_coordinates, bounds)
    rows = []
    for count in counts:
        print(f"[fit] nested Sobol prefix N={count}", flush=True)
        fit, shift, pairs, lengths, singular_values, synchronization = fit_case(
            coordinates, energies, overlap_pairs, overlap_values,
            count, grids, bounds, args
        )
        metrics = validation_metrics(
            fit, validation_coordinates, validation_energies,
            validation_links, shift, args.model,
        )
        nearest = cdist(
            scaled_validation, invariant_coordinates(coordinates[:count], bounds)
        ).min(axis=1)
        metrics.update(
            {
                "samples": count,
                "overlap_links": len(pairs),
                "links_per_geometry": len(pairs) / count,
                "maximum_graph_edge": float(np.max(lengths)),
                "rms_graph_edge": float(np.sqrt(np.mean(lengths**2))),
                "minimum_graph_link_singular_value": float(np.min(singular_values)),
                "fill_distance": float(np.max(nearest)),
                "max_abs_spin_square": float(np.max(np.abs(spin_square[:count]))),
                "final_loss": float(
                    fit.history[-1]["loss"]
                    if isinstance(fit.history[-1], dict)
                    else fit.history[-1]
                ),
                "synchronization": synchronization,
            }
        )
        rows.append(metrics)
        fit.save(args.output.with_name(f"{args.output.stem}_n{count}.pt"))
        print(json.dumps(metrics, indent=2))
    plot_results(rows, coordinates, bounds, args.output)
    report = {
        "design": "anchor + nested scrambled Sobol on r_long >= r_short",
        "counts": list(counts),
        "neighbors": args.neighbors,
        "minimum_link_singular_value": args.minimum_link_singular_value,
        "dataset": str(args.dataset),
        "reference": str(args.reference),
        "chart_features": args.chart_features,
        "feature_objective": args.feature_objective,
        "model": args.model,
        "rows": rows,
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"dataset: {args.dataset}")
    print(f"figure: {args.output}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
