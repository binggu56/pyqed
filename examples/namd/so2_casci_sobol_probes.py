#!/usr/bin/env python3
"""Fit SO2 endpoint Y from nested Sobol centers plus local tangent probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.generate_so2_casci_singlets import (
    electronic_metadata,
    electronic_structure,
    geometry,
    require_spin_pure_singlets,
    validate_electronic_metadata,
)
from examples.namd.so2_casci_sobol_mace import (
    EV_PER_HARTREE,
    invariant_coordinates,
    sparse_overlap_graph,
    sobol_coordinates,
    validation_data,
)
from pyqed.ldr.overlap import procrustes
from pyqed.ml import MACE


def probe_design(centers, bounds, steps, *, two_sided_axes=()):
    """Append coordinate-directed probes, optionally on both sides of an axis."""
    centers = np.asarray(centers, dtype=float)
    lower = np.asarray((bounds[0], bounds[0], bounds[2]), dtype=float)
    upper = np.asarray((bounds[1], bounds[1], bounds[3]), dtype=float)
    steps = np.asarray(steps, dtype=float)
    coordinates = list(centers)
    pairs = []
    axes = []
    two_sided_axes = {int(axis) for axis in two_sided_axes}
    for center, coordinate in enumerate(centers):
        for axis, step in enumerate(steps):
            signs = (-1.0, 1.0) if axis in two_sided_axes else (
                1.0 if coordinate[axis] + step <= upper[axis] else -1.0,
            )
            for sign in signs:
                probe = coordinate.copy()
                probe[axis] = np.clip(
                    coordinate[axis] + sign * step, lower[axis], upper[axis]
                )
                if np.isclose(probe[axis], coordinate[axis]):
                    continue
                pairs.append((center, len(coordinates)))
                axes.append(axis)
                coordinates.append(probe)
    return np.asarray(coordinates), np.asarray(pairs, dtype=int), np.asarray(axes, dtype=int)


def landmark_design(centers, bounds, count):
    """Choose a nested farthest-point landmark set starting at the anchor."""
    count = min(int(count), len(centers))
    if count < 0:
        raise ValueError("the electronic landmark count must be nonnegative")
    if count == 0:
        return np.empty(0, dtype=int)
    scaled = invariant_coordinates(centers, bounds)
    selected = [0]
    distance = np.linalg.norm(scaled - scaled[0], axis=1)
    while len(selected) < count:
        point = int(np.argmax(distance))
        selected.append(point)
        distance = np.minimum(
            distance, np.linalg.norm(scaled - scaled[point], axis=1)
        )
    return np.asarray(selected, dtype=int)


def generate(coordinates, pairs, landmark_indices, args):
    models, energies, spin_square = [], [], []
    for count, coordinate in enumerate(coordinates, start=1):
        model = electronic_structure(*coordinate, args)
        models.append(model.frame())
        energies.append(np.asarray(model.e_tot))
        spin_square.append([model.spin_square(state) for state in range(args.nstates)])
        if count == 1 or count % args.progress_every == 0 or count == len(coordinates):
            print(
                f"[CASCI] {count}/{len(coordinates)}, E0={energies[-1][0]:.10f} Eh, "
                f"max |S2|={np.max(np.abs(spin_square[-1])):.2e}",
                flush=True,
            )
    overlaps = np.asarray([models[left].overlap(models[right]) for left, right in pairs])
    anchor_overlaps = np.asarray([model.overlap(models[0]) for model in models])
    landmark_overlaps = np.asarray(
        [
            [models[landmark].overlap(model) for model in models]
            for landmark in landmark_indices
        ],
        dtype=complex,
    ).reshape(len(landmark_indices), len(models), args.nstates, args.nstates)
    return (
        np.asarray(energies), np.asarray(spin_square), overlaps,
        anchor_overlaps, landmark_overlaps,
    )


def align_to_anchor(energies, pairs, overlaps, anchor_overlaps, anchor=0):
    """Express sampled Hamiltonians and links in one smooth electronic gauge."""
    gauges = procrustes(anchor_overlaps)[0]
    shift = float(energies[int(anchor), 0])
    diagonal = np.asarray([np.diag(values - shift) for values in energies])
    hamiltonians = (
        gauges.conj().swapaxes(-1, -2) @ diagonal @ gauges
    )
    aligned_links = np.asarray([
        gauges[left].conj().T @ value @ gauges[right]
        for (left, right), value in zip(pairs, overlaps)
    ])
    return hamiltonians, aligned_links, gauges, shift


def nystrom_features(landmark_overlaps, landmark_indices, gauges, feature_rank):
    """Build isometric global endpoint targets from landmark overlap blocks."""
    landmark_overlaps = np.asarray(landmark_overlaps, dtype=complex)
    landmark_indices = np.asarray(landmark_indices, dtype=int)
    nlandmarks, npoints, nstates, _ = landmark_overlaps.shape
    gram = landmark_overlaps[:, landmark_indices].transpose(0, 2, 1, 3)
    gram = gram.reshape(nlandmarks * nstates, nlandmarks * nstates)
    gram = 0.5 * (gram + gram.conj().T)
    values, vectors = np.linalg.eigh(gram)
    keep = np.flatnonzero(values > 1.0e-10 * max(float(values[-1]), 1.0))
    keep = keep[-min(int(feature_rank), len(keep)):]
    if len(keep) < nstates:
        raise ValueError("electronic landmarks have insufficient numerical rank")
    transform = (vectors[:, keep] / np.sqrt(values[keep])[None, :]).conj().T
    blocks = landmark_overlaps.transpose(1, 0, 2, 3).reshape(
        npoints, nlandmarks * nstates, nstates
    )
    features = np.einsum("ra,nab->nrb", transform, blocks, optimize=True)
    features = features @ np.asarray(gauges)
    left, _singular, right = np.linalg.svd(features, full_matrices=False)
    features = left @ right
    return features, np.asarray(values[keep], dtype=float)


def all_links(coordinates, probe_pairs, ncenters, bounds, neighbors):
    center_pairs, _lengths = sparse_overlap_graph(
        coordinates[:ncenters], bounds, neighbors
    )
    return np.asarray(sorted(set(map(tuple, np.vstack((center_pairs, probe_pairs))))), dtype=int)


def endpoint_metrics(fit, coordinates, energies, links, shift):
    features = fit.neural_feature.predict(coordinates)
    hamiltonians = fit.neural_energy.predict(coordinates)
    predicted_energies, adiabatic_gauges = np.linalg.eigh(hamiltonians)
    adiabatic_features = features @ adiabatic_gauges
    reference_energies = np.asarray(energies).reshape(len(coordinates), -1) - shift
    shape = tuple(reference.shape[axis] + 1 for axis, reference in enumerate(links))
    magnitude_axis_errors = []
    singular_axis_errors = []
    magnitude_pointwise = []
    for axis, reference in enumerate(links):
        magnitude_errors = []
        singular_errors = []
        for left in np.ndindex(reference.shape[:-2]):
            right = list(left)
            right[axis] += 1
            left_flat = np.ravel_multi_index(left, shape)
            right_flat = np.ravel_multi_index(tuple(right), shape)
            predicted = (
                adiabatic_features[left_flat].conj().T
                @ adiabatic_features[right_flat]
            )
            magnitude_errors.append(
                np.linalg.norm(np.abs(predicted) - np.abs(reference[left]))
                / max(np.linalg.norm(reference[left]), np.finfo(float).tiny)
            )
            singular_errors.append(
                np.linalg.norm(
                    np.linalg.svd(predicted, compute_uv=False)
                    - np.linalg.svd(reference[left], compute_uv=False)
                ) / max(np.linalg.norm(reference[left]), np.finfo(float).tiny)
            )
        magnitude_errors = np.asarray(magnitude_errors)
        singular_errors = np.asarray(singular_errors)
        magnitude_pointwise.append(magnitude_errors)
        magnitude_axis_errors.append(float(np.sqrt(np.mean(magnitude_errors**2))))
        singular_axis_errors.append(float(np.sqrt(np.mean(singular_errors**2))))
    isometry = features.conj().swapaxes(-1, -2) @ features
    defect = np.linalg.norm(isometry - np.eye(features.shape[-1]), axis=(1, 2))
    energy_errors = np.abs(predicted_energies - reference_energies)
    return {
        "energy_mae_ev": float(np.mean(energy_errors) * EV_PER_HARTREE),
        "energy_max_ev": float(np.max(energy_errors) * EV_PER_HARTREE),
        "link_magnitude_axis_rms": magnitude_axis_errors,
        "link_magnitude_rms": float(
            np.sqrt(np.mean(np.concatenate(magnitude_pointwise) ** 2))
        ),
        "link_magnitude_axis_max": [
            float(np.max(values)) for values in magnitude_pointwise
        ],
        "link_singular_axis_rms": singular_axis_errors,
        "maximum_isometry_defect": float(np.max(defect)),
    }, magnitude_pointwise


def plot_result(coordinates, ncenters, probe_pairs, metrics, pointwise, output):
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), constrained_layout=True)
    scaled = invariant_coordinates(coordinates, (
        2.68, 2.92, np.deg2rad(110.0), np.deg2rad(130.0)
    ))
    for left, right in probe_pairs:
        axes[0].plot(
            scaled[[left, right], 0], scaled[[left, right], 1],
            color="0.72", lw=0.7, zorder=1,
        )
    axes[0].scatter(
        scaled[ncenters:, 0], scaled[ncenters:, 1], s=14,
        facecolor="none", edgecolor="0.5", label="local probes", zorder=2,
    )
    axes[0].scatter(
        scaled[:ncenters, 0], scaled[:ncenters, 1], s=28,
        c=scaled[:ncenters, 2], cmap="viridis", label="Sobol centers", zorder=3,
    )
    axes[0].set(xlabel=r"scaled $q_s$", ylabel=r"scaled $|q_a|$")
    axes[0].legend(frameon=False, fontsize=8)

    labels = (r"$r_1$", r"$r_2$", r"$\theta$")
    x = np.arange(3)
    axes[1].bar(x, metrics["link_magnitude_axis_rms"], color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set(ylabel=r"Full-grid $|L_{ij}|$ RMS", yscale="log")

    for axis, (values, color, label) in enumerate(zip(pointwise, colors, labels)):
        ordered = np.sort(np.maximum(values, 1.0e-8))
        axes[2].plot(
            ordered, np.arange(1, len(ordered) + 1) / len(ordered),
            color=color, label=label,
        )
    axes[2].set(
        xlabel=r"Full-grid $|L_{ij}|$ relative error",
        ylabel="Cumulative fraction", xscale="log",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centers", type=int, default=17)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--r-min", type=float, default=2.68)
    parser.add_argument("--r-max", type=float, default=2.92)
    parser.add_argument("--theta-min-deg", type=float, default=110.0)
    parser.add_argument("--theta-max-deg", type=float, default=130.0)
    parser.add_argument("--dr", type=float, default=0.06)
    parser.add_argument("--dtheta-deg", type=float, default=5.0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=6)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument("--spin-root-cushion", type=int, default=8)
    parser.add_argument("--scf-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--feature-rank", type=int, default=15)
    parser.add_argument("--landmarks", type=int, default=0)
    parser.add_argument(
        "--feature-objective", choices=("fixed", "subspace", "links-only"),
        default="links-only",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--radial-basis", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=64)
    parser.add_argument("--probe-weight", type=int, default=4)
    parser.add_argument(
        "--two-sided-bend", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--reference", type=Path,
        default=Path("/private/tmp/so2_casci_singlet_5x5x5.npz"),
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("/private/tmp/so2_casci_sobol_probes_17x4.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/so2_casci_sobol_probes_y.png"),
    )
    args = parser.parse_args()
    bounds = (
        args.r_min, args.r_max, np.deg2rad(args.theta_min_deg),
        np.deg2rad(args.theta_max_deg),
    )
    centers = sobol_coordinates(args.centers, bounds, args.seed)
    coordinates, probe_pairs, probe_axes = probe_design(
        centers, bounds, (args.dr, args.dr, np.deg2rad(args.dtheta_deg)),
        two_sided_axes=(2,) if args.two_sided_bend else (),
    )
    landmark_indices = landmark_design(centers, bounds, args.landmarks)
    pairs = all_links(coordinates, probe_pairs, args.centers, bounds, args.neighbors)
    if args.dataset.is_file():
        with np.load(args.dataset, allow_pickle=False) as archive:
            validate_electronic_metadata(archive, args, label="SO2 probe cache")
            cached = np.asarray(archive["coordinates"])
            if cached.shape != coordinates.shape or not np.allclose(cached, coordinates):
                raise ValueError("cached tangent-probe design differs from this request")
            energies = np.asarray(archive["energies"])
            spin_square = np.asarray(archive["spin_square"])
            overlaps = np.asarray(archive["overlaps"])
            anchor_overlaps = np.asarray(archive["anchor_overlaps"])
            landmark_overlaps = np.asarray(archive["landmark_overlaps"])
            if not np.array_equal(
                np.asarray(archive["landmark_indices"]), landmark_indices
            ):
                raise ValueError("cached electronic landmarks differ from this request")
            if not np.array_equal(np.asarray(archive["pairs"]), pairs):
                raise ValueError("cached overlap graph differs from this request")
        print(f"[cache] restored {len(coordinates)} CASCI geometries")
    else:
        energies, spin_square, overlaps, anchor_overlaps, landmark_overlaps = generate(
            coordinates, pairs, landmark_indices, SimpleNamespace(**vars(args))
        )
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.dataset, coordinates=coordinates, centers=args.centers,
            probe_pairs=probe_pairs, probe_axes=probe_axes, pairs=pairs,
            energies=energies, spin_square=spin_square, overlaps=overlaps,
            anchor_overlaps=anchor_overlaps,
            landmark_indices=landmark_indices,
            landmark_overlaps=landmark_overlaps,
            **electronic_metadata(args),
        )
    require_spin_pure_singlets(spin_square)
    if args.probe_weight < 1:
        raise ValueError("probe weight must be positive")
    hamiltonians, aligned_overlaps, gauges, shift = align_to_anchor(
        energies, pairs, overlaps, anchor_overlaps
    )
    if len(landmark_indices):
        feature_targets, landmark_spectrum = nystrom_features(
            landmark_overlaps, landmark_indices, gauges, args.feature_rank
        )
        fitted_rank = feature_targets.shape[1]
    else:
        feature_targets = None
        landmark_spectrum = np.empty(0)
        fitted_rank = args.feature_rank
    lookup = {tuple(pair): value for pair, value in zip(pairs, aligned_overlaps)}
    repeated_probes = np.repeat(probe_pairs, args.probe_weight - 1, axis=0)
    training_pairs = np.vstack((pairs, repeated_probes))
    training_overlaps = np.asarray([lookup[tuple(pair)] for pair in training_pairs])
    grids, validation_coordinates, validation_energies, validation_links = validation_data(
        args.reference, args
    )
    fit = MACE(
        grids, ("O", "S", "O"), lambda coordinate: geometry(*coordinate),
        args.nstates, geometry_units="bohr", channels=args.channels,
        max_ell=2, interactions=2, correlation=2, radial_basis=args.radial_basis,
        radial_mlp=(args.head_width, args.head_width), cutoff=7.0,
    ).fit_y(
        (coordinates, hamiltonians), coordinates, training_pairs, training_overlaps,
        feature_targets=feature_targets, feature_rank=fitted_rank,
        feature_objective=args.feature_objective,
        hidden=(args.head_width, args.head_width), epochs=args.epochs,
        learning_rate=2.0e-3, link_weight=5.0, isometry_weight=1.0,
        smoothness=0.0, sync_steps=5000, seed=args.seed, distill=False,
    )
    metrics, pointwise = endpoint_metrics(
        fit, validation_coordinates, validation_energies, validation_links, shift
    )
    metrics.update({
        "centers": args.centers,
        "probe_geometries": len(coordinates) - args.centers,
        "total_geometries": len(coordinates),
        "overlap_links": len(pairs),
        "training_link_samples": len(training_pairs),
        "probe_weight": args.probe_weight,
        "probe_counts_by_axis": [
            int(np.count_nonzero(probe_axes == axis)) for axis in range(3)
        ],
        "electronic_landmarks": len(landmark_indices),
        "feature_rank": fitted_rank,
        "minimum_retained_landmark_eigenvalue": (
            float(landmark_spectrum[0]) if len(landmark_spectrum) else None
        ),
        "feature_objective": args.feature_objective,
        "max_abs_spin_square": float(np.max(np.abs(spin_square))),
        "training_link_rms": fit.info["synchronization"]["rms_relative_link_error"],
        "final_loss": float(fit.history[-1]),
    })
    fit.save(args.output.with_suffix(".pt"))
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_result(coordinates, args.centers, probe_pairs, metrics, pointwise, args.output)
    print(json.dumps(metrics, indent=2))
    print(f"dataset: {args.dataset}")
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
