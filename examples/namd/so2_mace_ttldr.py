#!/usr/bin/env python3
"""Fit cached ab initio SO2 fields with MACE and validate held-out fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from pyqed.ldr.oracle import Frames, ProcrustesOracle
from pyqed.ldr.ttfit import LinkPath
from pyqed.ml import MACE, frame_projector
from pyqed.mps.functional import FunctionalTT
from pyqed.namd.ttldr import TTLDR
from pyqed.units import amu2au, atomic_mass, au2ev, au2fs


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "so2_am1_meci_3x3x3.npz"
)
O_MASS = atomic_mass["O"] * amu2au
S_MASS = atomic_mass["S"] * amu2au


def geometry_r1_r2_theta(coordinate):
    """Return a body-fixed O-S-O geometry from (r1, r2, theta), in bohr."""
    r1, r2, theta = map(float, coordinate)
    half = 0.5 * theta
    return np.asarray(
        [
            [r1 * np.cos(half), r1 * np.sin(half), 0.0],
            [0.0, 0.0, 0.0],
            [r2 * np.cos(half), -r2 * np.sin(half), 0.0],
        ]
    )


def geometry_qs_theta_qa(coordinate):
    """Return a body-fixed O-S-O geometry from (qs, theta, qa), in bohr."""
    qs, theta, qa = map(float, coordinate)
    root_two = np.sqrt(2.0)
    return geometry_r1_r2_theta(((qs + qa) / root_two, (qs - qa) / root_two, theta))


def edge_data(grids, values, axis):
    edge_grids = list(grids)
    edge_grids[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
    mesh = np.meshgrid(*edge_grids, indexing="ij")
    coordinates = np.stack([item.reshape(-1) for item in mesh], axis=1)
    return coordinates, values.reshape(len(coordinates), *values.shape[-2:])


def aligned_fields(path, grids_path=None):
    with np.load(path, allow_pickle=False) as archive:
        if "energy" in archive:
            if grids_path is None:
                raise ValueError("already aligned fields require --grids")
            energy = np.asarray(archive["energy"])
            links = tuple(np.asarray(archive[f"link_{axis}"]) for axis in range(3))
            with np.load(grids_path, allow_pickle=False) as grid_archive:
                grids = tuple(
                    np.asarray(grid_archive[name]) for name in ("qs", "theta", "qa")
                )
            if energy.shape[:3] != tuple(len(grid) for grid in grids):
                raise ValueError("aligned field and coordinate grid shapes differ")
            return grids, energy, links, geometry_qs_theta_qa
        grids = tuple(np.asarray(archive[name]) for name in ("r1", "r2", "theta"))
        energies = np.asarray(archive["energies"])
        raw_links = tuple(np.asarray(archive[f"links_{axis}"]) for axis in range(3))

    shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    links = {
        (axis, index): values[index]
        for axis, values in enumerate(raw_links)
        for index in np.ndindex(values.shape[:-2])
    }
    transport = LinkPath(shape, nstates, links)
    frames = Frames(shape, lambda index: (index, energies[index]))
    oracle = ProcrustesOracle(
        frames,
        tuple(size // 2 for size in shape),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=transport.between,
        energy_shift=None,
    )
    indices = list(np.ndindex(shape))
    energy = oracle.hamiltonian_many(indices).reshape(*shape, nstates, nstates)
    aligned_links = []
    for axis, values in enumerate(raw_links):
        pairs = []
        for left in np.ndindex(values.shape[:-2]):
            right = list(left)
            right[axis] += 1
            pairs.append((left, tuple(right)))
        aligned_links.append(oracle.overlap_many(pairs).reshape(values.shape))
    return grids, energy, tuple(aligned_links), geometry_r1_r2_theta


def split_samples(grids, energy, links, seed, holdout_fraction, strategy="random"):
    shape = energy.shape[:-2]
    indices = list(np.ndindex(shape))
    center = tuple(size // 2 for size in shape)
    candidates = [position for position, index in enumerate(indices) if index != center]
    rng = np.random.default_rng(seed)
    holdout = max(1, int(round(float(holdout_fraction) * len(indices))))
    if holdout >= len(indices) - 1:
        raise ValueError("holdout fraction leaves too few training geometries")
    if strategy == "random":
        held_positions = set(rng.choice(candidates, holdout, replace=False).tolist())
    elif strategy == "spatial-block":
        mesh = np.meshgrid(*grids, indexing="ij")
        coordinates = np.stack([item.reshape(-1) for item in mesh], axis=1)
        invariant_coordinates = np.column_stack(
            (
                (coordinates[:, 0] + coordinates[:, 1]) / np.sqrt(2.0),
                np.abs(coordinates[:, 0] - coordinates[:, 1]) / np.sqrt(2.0),
                coordinates[:, 2],
            )
        )
        lower = invariant_coordinates.min(axis=0)
        span = np.maximum(
            invariant_coordinates.max(axis=0) - lower, np.finfo(float).eps
        )
        scaled = (invariant_coordinates - lower) / span
        target = np.asarray((0.75, 0.75, 0.75))
        distance = np.linalg.norm(scaled - target, axis=1)
        order = [position for position in np.argsort(distance) if position in candidates]
        held_positions = set(order[:holdout])
    else:
        raise ValueError("split strategy must be 'random' or 'spatial-block'")
    train_positions = np.asarray(
        [position for position in range(len(indices)) if position not in held_positions]
    )
    held_positions = np.asarray(sorted(held_positions))
    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([item.reshape(-1) for item in mesh], axis=1)

    train_links = []
    link_masks = []
    for axis, values in enumerate(links):
        edge_coordinates, edge_values = edge_data(grids, values, axis)
        mask = []
        edge_shape = values.shape[:-2]
        for left in np.ndindex(edge_shape):
            right = list(left)
            right[axis] += 1
            mask.append(
                indices.index(left) not in held_positions
                and indices.index(tuple(right)) not in held_positions
            )
        mask = np.asarray(mask, dtype=bool)
        train_links.append((edge_coordinates[mask], edge_values[mask]))
        link_masks.append(mask)
    train_ids = {position: local for local, position in enumerate(train_positions)}
    feature_pairs = []
    feature_links = []
    for axis, (values, mask) in enumerate(zip(links, link_masks)):
        edge_shape = values.shape[:-2]
        for edge, left in enumerate(np.ndindex(edge_shape)):
            if not mask[edge]:
                continue
            right = list(left)
            right[axis] += 1
            left_flat = indices.index(left)
            right_flat = indices.index(tuple(right))
            feature_pairs.append((train_ids[left_flat], train_ids[right_flat]))
            feature_links.append(values[left])
    return (
        (coordinates[train_positions], energy.reshape(len(indices), *energy.shape[-2:])[train_positions]),
        tuple(train_links),
        np.asarray(feature_pairs, dtype=int),
        np.asarray(feature_links),
        coordinates,
        train_positions,
        held_positions,
        tuple(link_masks),
    )


def kinetic_terms(grids):
    """A light mass-scaled validation KEO in the three valence coordinates."""
    reduced_mass = O_MASS * S_MASS / (O_MASS + S_MASS)
    mean_bond = 0.5 * (np.mean(grids[0]) + np.mean(grids[1]))
    masses = (reduced_mass, reduced_mass, reduced_mass * mean_bond**2)
    identities = tuple(np.eye(len(grid)) for grid in grids)
    terms = []
    for axis, (grid, mass) in enumerate(zip(grids, masses)):
        spacing = float(np.mean(np.diff(grid)))
        laplacian = np.diag(np.full(len(grid), 2.0))
        laplacian += np.diag(np.full(len(grid) - 1, -1.0), 1)
        laplacian += np.diag(np.full(len(grid) - 1, -1.0), -1)
        factors = list(identities)
        factors[axis] = laplacian / (2.0 * mass * spacing**2)
        terms.append((1.0, tuple(factors)))
    return tuple(terms)


def exact_fit(grids, energy, links):
    def field(field_grids, values, hermitian):
        return FunctionalTT(
            degrees=tuple(len(grid) - 1 for grid in field_grids),
            rank=16,
            bounds=tuple((grid[0], grid[-1]) for grid in field_grids),
            normalization="frobenius",
            hermitian=hermitian,
        ).fit_grid(field_grids, values)

    energy_fit = field(grids, energy, True)
    link_fits = []
    for axis, values in enumerate(links):
        edge_grids = list(grids)
        edge_grids[axis] = 0.5 * (grids[axis][:-1] + grids[axis][1:])
        link_fits.append(field(tuple(edge_grids), values, False))
    return SimpleNamespace(
        success=True,
        grids=grids,
        energy=energy_fit,
        links=tuple(link_fits),
        feature=None,
    )


def relative_error(predicted, reference, mask):
    predicted = np.asarray(predicted)[mask]
    reference = np.asarray(reference)[mask]
    return float(np.linalg.norm(predicted - reference) / np.linalg.norm(reference))


def populations(hamiltonian, times, shape, initial_state=2):
    values, vectors = eigh(hamiltonian)
    nuclear_size = int(np.prod(shape))
    nstates = hamiltonian.shape[0] // nuclear_size
    initial = np.zeros(nuclear_size * nstates, dtype=complex)
    center = tuple(size // 2 for size in shape)
    initial[np.ravel_multi_index(center, shape) * nstates + initial_state] = 1.0
    coefficients = vectors.conj().T @ initial
    states = (vectors @ (np.exp(-1j * values[:, None] * times) * coefficients[:, None])).T
    probability = np.abs(states.reshape(len(times), -1, nstates)) ** 2
    return probability.sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--grids", type=Path)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument(
        "--split", choices=("random", "spatial-block"), default="spatial-block"
    )
    parser.add_argument("--feature-rank", type=int, default=9)
    parser.add_argument(
        "--ambient-representation",
        choices=("full", "diagonal"),
        default="full",
        help="Latent Hermitian operator used in H = Y.H @ A @ Y.",
    )
    parser.add_argument("--frame-fraction", type=float, default=0.0)
    parser.add_argument("--ambient-fraction", type=float, default=0.0)
    parser.add_argument("--energy-frame-gradient", type=float, default=1.0)
    parser.add_argument("--link-weight", type=float, default=1.0)
    parser.add_argument(
        "--feature-objective",
        choices=("subspace", "links-only", "fixed"),
        default="links-only",
        help="Gauge-safe endpoint supervision; fixed is the legacy gauge-dependent loss.",
    )
    parser.add_argument(
        "--link-representation",
        choices=("endpoint", "directional"),
        default="endpoint",
        help="Fit links through global endpoint features or direct axis-specific heads.",
    )
    parser.add_argument("--tt-rank", type=int, default=16)
    parser.add_argument("--tt-degree", type=int, default=6)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--max-ell", type=int, default=2)
    parser.add_argument("--interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--radial-basis", type=int, default=4)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Skip FunctionalTT distillation and all LDR construction/dynamics.",
    )
    parser.add_argument("--max-held-energy-error", type=float, default=0.25)
    parser.add_argument("--max-held-link-error", type=float, default=0.35)
    parser.add_argument(
        "--chart-features",
        action="store_true",
        help="Append internal coordinates to the heads; useful as an interpolation baseline but not transferable.",
    )
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/so2_mace_ttldr.png"))
    args = parser.parse_args()

    grids, energy, links, geometry = aligned_fields(args.fixture, args.grids)
    coordinate_labels = (
        (r"$q_s$", r"$\theta$", r"$q_a$")
        if args.grids is not None
        else (r"$r_1$", r"$r_2$", r"$\theta$")
    )
    (
        energy_train,
        link_train,
        feature_pairs,
        feature_links,
        coordinates,
        train,
        held,
        link_masks,
    ) = split_samples(
        grids, energy, links, args.seed, args.holdout, args.split
    )
    fit = MACE(
        grids,
        ("O", "S", "O"),
        geometry,
        energy.shape[-1],
        chart_features=args.chart_features,
        geometry_units="bohr",
        channels=args.channels,
        max_ell=args.max_ell,
        interactions=args.interactions,
        correlation=args.correlation,
        radial_basis=args.radial_basis,
        radial_mlp=(args.head_width, args.head_width),
        cutoff=7.0,
    )
    common_fit = {
        "hidden": (args.head_width, args.head_width),
        "epochs": args.epochs,
        "learning_rate": 2.0e-3,
        "weight_decay": 1.0e-8,
        "seed": args.seed,
        "distill": not args.fit_only,
        "tt_rank": args.tt_rank,
        "tt_degree": args.tt_degree,
    }
    if args.link_representation == "endpoint":
        fit.fit_y(
            energy_train,
            coordinates[train],
            feature_pairs,
            feature_links,
            feature_rank=args.feature_rank,
            feature_objective=args.feature_objective,
            ambient_representation=args.ambient_representation,
            frame_fraction=args.frame_fraction,
            ambient_fraction=args.ambient_fraction,
            energy_frame_gradient=args.energy_frame_gradient,
            link_weight=args.link_weight,
            isometry_weight=1.0,
            smoothness=1.0e-4,
            **common_fit,
        )
    else:
        fit.fit(energy_train, link_train, **common_fit)
    checkpoint = args.output.with_suffix(".pt")
    fit.save(checkpoint)
    link_heads = (
        (fit._feature_head,)
        if args.link_representation == "endpoint"
        else fit._link_heads
    )
    trainable_parameters = int(
        sum(parameter.numel() for parameter in fit.encoder.parameters())
        + sum(parameter.numel() for parameter in fit._energy_head.parameters())
        + sum(
            parameter.numel()
            for head in link_heads
            for parameter in head.parameters()
        )
    )

    reference_energy = energy.reshape(len(coordinates), *energy.shape[-2:])
    predicted_energy = fit.neural_energy.predict(coordinates)
    feature_training_error = None
    anchor_error = None
    predicted_feature = None
    if args.link_representation == "endpoint":
        predicted_training_feature = fit.neural_feature.predict(coordinates[train])
        if args.feature_objective == "fixed":
            feature_training_error = float(
                np.linalg.norm(predicted_training_feature - fit.feature_targets_)
                / np.linalg.norm(fit.feature_targets_)
            )
        elif args.feature_objective == "subspace":
            predicted_projector = frame_projector(predicted_training_feature)
            target_projector = frame_projector(fit.feature_targets_)
            feature_training_error = float(
                np.linalg.norm(predicted_projector - target_projector)
                / np.linalg.norm(target_projector)
            )
        fixed_anchor = np.zeros((fit.feature_rank, energy.shape[-1]), dtype=complex)
        fixed_anchor[: energy.shape[-1]] = np.eye(energy.shape[-1])
        anchor_error = float(
            np.linalg.norm(predicted_training_feature[fit.feature_anchor_] - fixed_anchor)
        )
        predicted_feature = fit.neural_feature.predict(coordinates)
    energy_errors = (
        relative_error(predicted_energy, reference_energy, train),
        relative_error(predicted_energy, reference_energy, held),
    )
    link_errors = []
    held_link_errors = []
    shape = energy.shape[:-2]
    indices = list(np.ndindex(shape))
    for axis, (values, mask) in enumerate(zip(links, link_masks)):
        _edge_coordinates, reference = edge_data(grids, values, axis)
        if args.link_representation == "endpoint":
            predicted = []
            for left in np.ndindex(values.shape[:-2]):
                right = list(left)
                right[axis] += 1
                left_feature = predicted_feature[indices.index(left)]
                right_feature = predicted_feature[indices.index(tuple(right))]
                predicted.append(left_feature.conj().T @ right_feature)
            predicted = np.asarray(predicted)
        else:
            predicted = fit.neural_links[axis].predict(_edge_coordinates)
        pointwise = np.linalg.norm(predicted - reference, axis=(1, 2)) / np.maximum(
            np.linalg.norm(reference, axis=(1, 2)), np.finfo(float).tiny
        )
        held_link_errors.append(pointwise[~mask])
        link_errors.append(
            (
                relative_error(predicted, reference, mask),
                relative_error(predicted, reference, ~mask),
            )
        )

    dimension = int(np.prod(energy.shape[:-2]) * energy.shape[-1])
    predicted_driver = None
    h_error = None
    population_error = None
    reference_populations = None
    predicted_populations = None
    times = None
    if not args.fit_only and dimension <= 300:
        terms = kinetic_terms(grids)
        predicted_driver = TTLDR.from_fit(
            fit, keo=terms, overlap_rank=args.tt_rank, operator_rank=64
        )
        reference_driver = TTLDR.from_fit(
            exact_fit(grids, energy, links),
            keo=terms,
            overlap_rank=args.tt_rank,
            operator_rank=None,
        )
        predicted_h = predicted_driver.hamiltonian.to_dense()
        reference_h = reference_driver.hamiltonian.to_dense()
        h_error = float(
            np.linalg.norm(predicted_h - reference_h) / np.linalg.norm(reference_h)
        )
        times = np.linspace(0.0, args.time_fs / au2fs, 201)
        reference_populations = populations(reference_h, times, energy.shape[:-2])
        predicted_populations = populations(predicted_h, times, energy.shape[:-2])
        population_error = float(
            np.sqrt(np.mean((predicted_populations - reference_populations) ** 2))
        )

    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(fit.history, color=colors[0], lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")

    reference_levels = np.linalg.eigvalsh(reference_energy)
    predicted_levels = np.linalg.eigvalsh(predicted_energy)
    level_error_ev = np.abs(predicted_levels - reference_levels) * au2ev
    reference_gaps = np.diff(reference_levels, axis=1)
    predicted_gaps = np.diff(predicted_levels, axis=1)
    gap_error_ev = np.abs(predicted_gaps - reference_gaps) * au2ev
    axes[1].plot([-0.01, 0.12], [-0.01, 0.12], color="0.65", lw=1.0)
    axes[1].scatter(
        reference_levels[train].ravel(), predicted_levels[train].ravel(),
        s=13, color=colors[0], label="train", alpha=0.75,
    )
    axes[1].scatter(
        reference_levels[held].ravel(), predicted_levels[held].ravel(),
        s=22, facecolor="none", edgecolor=colors[1], label="held out", linewidth=1.0,
    )
    axes[1].set_xlabel(r"Reference $\bar E$ eigenvalue (Eh)")
    axes[1].set_ylabel(r"MACE $\bar E$ eigenvalue (Eh)")
    axes[1].legend(frameon=False, fontsize=8)

    labels = (r"$\bar E$",) + tuple(
        rf"$\bar L_{{{label[1:-1]}}}$" for label in coordinate_labels
    )
    errors = np.asarray((energy_errors, *link_errors))
    x = np.arange(len(labels))
    axes[2].bar(x - 0.18, errors[:, 0], 0.36, color=colors[0], label="train")
    axes[2].bar(x + 0.18, errors[:, 1], 0.36, color=colors[1], label="held out")
    axes[2].set_yscale("log")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Relative Frobenius error")
    axes[2].legend(frameon=False, fontsize=8)

    if times is not None:
        time_fs = times * au2fs
        for state, color in enumerate(colors):
            axes[3].plot(time_fs, reference_populations[:, state], color=color, lw=1.4, label=rf"$P_{state}$")
            axes[3].plot(time_fs, predicted_populations[:, state], color=color, lw=1.1, ls="--")
        axes[3].set_xlabel("Time (fs)")
        axes[3].set_ylabel("Population")
        axes[3].set_ylim(-0.02, 1.02)
        axes[3].legend(frameon=False, fontsize=8, ncol=3)
        axes[3].text(0.98, 0.08, "solid: full\ndashed: MACE", transform=axes[3].transAxes, ha="right", va="bottom", fontsize=8)
    else:
        for axis, (errors_axis, color) in enumerate(zip(held_link_errors, colors)):
            ordered = np.sort(np.maximum(errors_axis, 1.0e-6))
            fraction = np.arange(1, len(ordered) + 1) / len(ordered)
            axes[3].plot(ordered, fraction, color=color, lw=1.4, label=labels[axis + 1])
        axes[3].set_xscale("log")
        axes[3].set_xlabel("Held-out link error")
        axes[3].set_ylabel("Cumulative fraction")
        axes[3].set_ylim(0.0, 1.02)
        axes[3].legend(frameon=False, fontsize=8)

    for label, axis in zip("abcd", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=350)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    metrics = {
        "energy_relative_error": {"train": energy_errors[0], "held_out": energy_errors[1]},
        "eigenvalue_mae_ev": {
            "train": float(np.mean(level_error_ev[train])),
            "held_out": float(np.mean(level_error_ev[held])),
        },
        "gap_mae_ev": {
            "train": float(np.mean(gap_error_ev[train])),
            "held_out": float(np.mean(gap_error_ev[held])),
        },
        "link_relative_error": {
            str(axis): {"train": values[0], "held_out": values[1]}
            for axis, values in enumerate(link_errors)
        },
        "hamiltonian_relative_error": h_error,
        "population_rmse": population_error,
        "feature_training_error": feature_training_error,
        "anchor_error": anchor_error,
        "synchronization": fit.info.get("synchronization"),
        "distillation": fit.info.get("distillation"),
        "ttldr_operator_ranks": (
            None if predicted_driver is None else predicted_driver.operator_ranks
        ),
        "vibronic_dimension": dimension,
        "samples": {
            "energy_train": len(train),
            "energy_total": len(coordinates),
            "link_train": [int(np.sum(mask)) for mask in link_masks],
            "link_total": [len(mask) for mask in link_masks],
            "feature_rank": fit.feature_rank,
            "link_representation": args.link_representation,
            "feature_objective": (
                args.feature_objective
                if args.link_representation == "endpoint"
                else None
            ),
            "split": args.split,
            "held_indices": held.tolist(),
        },
        "final_loss": fit.history[-1],
        "model": {
            "atomistic_only": not args.chart_features,
            "chart_features": args.chart_features,
            "geometry_units": "bohr (converted to Angstrom before ASE/MACE)",
            "checkpoint": str(checkpoint),
            "limited_data_benchmark": True,
            "fit_only": args.fit_only,
            "trainable_parameters": trainable_parameters,
            "structures_per_parameter": len(train) / trainable_parameters,
        },
    }
    metrics_path = args.output.with_suffix(".json")
    validation_passed = bool(
        energy_errors[1] <= args.max_held_energy_error
        and max(values[1] for values in link_errors) <= args.max_held_link_error
    )
    metrics["validation"] = {
        "passed": validation_passed,
        "max_held_energy_error": args.max_held_energy_error,
        "max_held_link_error": args.max_held_link_error,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"figure: {args.output}")
    print(f"metrics: {metrics_path}")
    print(f"checkpoint: {checkpoint}")
    if not validation_passed:
        raise RuntimeError("MACE held-out validation thresholds were not met")


if __name__ == "__main__":
    main()
