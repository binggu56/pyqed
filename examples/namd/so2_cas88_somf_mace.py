#!/usr/bin/env python3
"""Fit Procrustes-gauged SO2 CAS(8,8)/SOMF fields from its database.

The fitted target is the complete complex local Hamiltonian, not separately
fitted adiabatic energies. Raw many-electron overlaps remain nonunitary link
targets. Oxygen exchange and the molecular-plane reflection are imposed as
exact state-space constraints on the MACE Hamiltonian and endpoint fields.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from pyqed.ldr.database import ElectronicDatabase
from pyqed.ldr.so2 import (
    SO2_SPECIES,
    adaptive_points,
    canonical_spin_vibronic_permutation,
    full_spin_overlap,
    geometry,
    invariant_coordinates,
    plane_parities,
    point_group_representations,
    procrustes_fields,
    sparse_overlap_graph,
)
from pyqed.ml import MACE
from pyqed.units import au2ev, au2wavenumber


OVERLAP_PROTOCOL = {
    "algorithm": "pyqed-so2-cas88-full-spin-casci-overlap",
    "version": 1,
    "state_order": "S roots then triplet-root-major Ms=(-1,0,+1)",
    "unitarized": False,
    "inactive_core_factor": True,
}


def specification(coordinate, protocol):
    return {
        "geometry": {
            "species": list(SO2_SPECIES),
            "coordinates_bohr": geometry(coordinate).round(14).tolist(),
        },
        "protocol": protocol,
    }


def load_records(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    database = ElectronicDatabase(summary["database"])
    names, records, identifiers = [], [], []
    for name, item in summary["points"].items():
        key = specification(item["coordinate"], summary["protocol"])
        record = database.get(key)
        if record is None:
            raise FileNotFoundError(f"database record for {name!r} is missing")
        names.append(name)
        records.append(record)
        identifiers.append(database.identifier(key))
    return summary, database, names, records, identifiers


def cached_overlap(database, identifiers, records, left, right):
    value = database.get_overlap(
        identifiers[left], identifiers[right], OVERLAP_PROTOCOL
    )
    if value is None:
        value = full_spin_overlap(records[left], records[right])
        database.put_overlap(
            identifiers[left],
            identifiers[right],
            OVERLAP_PROTOCOL,
            value,
            metadata={"left": int(left), "right": int(right), "raw": True},
        )
    return np.asarray(value, dtype=complex)


def farthest_holdout(coordinates, bounds, count, anchor=0):
    """Choose a space-filling held set while retaining the anchor for training."""

    scaled = invariant_coordinates(coordinates, bounds)
    available = [index for index in range(len(scaled)) if index != int(anchor)]
    center_distance = np.linalg.norm(scaled[available] - scaled[anchor], axis=1)
    selected = [available[int(np.argmax(center_distance))]]
    while len(selected) < int(count):
        candidates = [index for index in available if index not in selected]
        distances = cdist(scaled[candidates], scaled[selected])
        selected.append(candidates[int(np.argmax(np.min(distances, axis=1)))])
    return np.asarray(sorted(selected), dtype=int)


def validation_pairs(coordinates, train, held, bounds, neighbors=2):
    scaled = invariant_coordinates(coordinates, bounds)
    distances = cdist(scaled[held], scaled[train])
    pairs = []
    for row, point in enumerate(held):
        nearest = np.argsort(distances[row])[: int(neighbors)]
        pairs.extend((int(point), int(train[index])) for index in nearest)
    return np.asarray(pairs, dtype=int)


def covariance_metrics(fit, coordinates):
    canonical = np.asarray(coordinates)[
        coordinates[:, 0] > coordinates[:, 1] + 1.0e-12
    ]
    exchanged = canonical.copy()
    exchanged[:, [0, 1]] = exchanged[:, [1, 0]]
    electronic = fit.coordinate_exchange_["electronic_representation"]
    ambient = fit.coordinate_exchange_["ambient_representation"]
    h = fit.neural_energy.predict(canonical)
    h_exchanged = fit.neural_energy.predict(exchanged)
    y = fit.neural_feature.predict(canonical)
    y_exchanged = fit.neural_feature.predict(exchanged)
    expected_h = electronic.conj().T @ h @ electronic
    expected_y = np.einsum(
        "ab,nbi,ij->naj", ambient, y, electronic, optimize=True
    )
    return {
        "exchange_h_max": float(
            np.max(np.linalg.norm(h_exchanged - expected_h, axis=(1, 2)))
        ),
        "exchange_y_max": float(
            np.max(np.linalg.norm(y_exchanged - expected_y, axis=(1, 2)))
        ),
    }


def assess(fit, coordinates, hamiltonians, pairs, links, held):
    predicted_h = fit.neural_energy.predict(coordinates[held])
    reference_h = hamiltonians[held]
    h_error = np.linalg.norm(predicted_h - reference_h, axis=(1, 2))
    identity = np.eye(reference_h.shape[-1])
    predicted_scalar = np.trace(predicted_h, axis1=1, axis2=2) / len(identity)
    reference_scalar = np.trace(reference_h, axis1=1, axis2=2) / len(identity)
    predicted_traceless = predicted_h - predicted_scalar[:, None, None] * identity
    reference_traceless = reference_h - reference_scalar[:, None, None] * identity
    traceless_error = np.linalg.norm(
        predicted_traceless - reference_traceless, axis=(1, 2)
    )
    predicted_levels = np.linalg.eigvalsh(predicted_h)
    reference_levels = np.linalg.eigvalsh(reference_h)
    level_error = np.abs(predicted_levels - reference_levels)
    st_error = np.linalg.norm(
        predicted_h[:, :3, 3:] - reference_h[:, :3, 3:], axis=(1, 2)
    )

    features = fit.neural_feature.predict(coordinates)
    predicted_links = np.asarray(
        [features[left].conj().T @ features[right] for left, right in pairs]
    )
    link_error = np.linalg.norm(predicted_links - links, axis=(1, 2))
    link_scale = np.linalg.norm(links, axis=(1, 2))
    relative_links = link_error / np.maximum(link_scale, np.finfo(float).tiny)
    predicted_singular = np.linalg.svd(predicted_links, compute_uv=False)
    reference_singular = np.linalg.svd(links, compute_uv=False)
    singular_error = np.linalg.norm(
        predicted_singular - reference_singular, axis=1
    ) / np.maximum(
        np.linalg.norm(reference_singular, axis=1), np.finfo(float).tiny
    )
    return {
        "hamiltonian_max_hartree": float(np.max(h_error)),
        "hamiltonian_rms_hartree": float(np.sqrt(np.mean(h_error**2))),
        "traceless_hamiltonian_max_hartree": float(np.max(traceless_error)),
        "eigenvalue_mae_ev": float(np.mean(level_error) * au2ev),
        "eigenvalue_max_ev": float(np.max(level_error) * au2ev),
        "singlet_triplet_block_max_cm-1": float(np.max(st_error) * au2wavenumber),
        "link_relative_rms": float(np.sqrt(np.mean(relative_links**2))),
        "link_relative_max": float(np.max(relative_links)),
        "link_singular_rms": float(np.sqrt(np.mean(singular_error**2))),
        "pointwise_hamiltonian_hartree": h_error.tolist(),
        "pointwise_link_relative": relative_links.tolist(),
    }


def plot_result(coordinates, train, held, bounds, metrics, output):
    scaled = invariant_coordinates(coordinates, bounds)
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.0), constrained_layout=True)
    axes[0].scatter(
        scaled[train, 0], scaled[train, 1], c=scaled[train, 2], cmap="viridis",
        s=28, label="train", edgecolor="white", linewidth=0.3,
    )
    axes[0].scatter(
        scaled[held, 0], scaled[held, 1], facecolor="none", edgecolor="#D55E00",
        s=45, linewidth=1.2, label="held out",
    )
    axes[0].set(xlabel=r"scaled $q_s$", ylabel=r"scaled $|q_a|$")
    axes[0].legend(frameon=False, fontsize=8)

    reference = np.asarray(metrics["held_reference_levels_ev"])
    predicted = np.asarray(metrics["held_predicted_levels_ev"])
    lower = min(float(reference.min()), float(predicted.min()))
    upper = max(float(reference.max()), float(predicted.max()))
    axes[1].plot((lower, upper), (lower, upper), color="0.65", lw=1.0)
    axes[1].scatter(reference, predicted, s=15, color=colors[0], alpha=0.75)
    axes[1].set(xlabel="Reference level (eV)", ylabel="MACE level (eV)")

    h_values = np.asarray(metrics["pointwise_hamiltonian_hartree"])
    axes[2].bar(np.arange(len(h_values)), h_values, color=colors[1])
    axes[2].axhline(metrics["gates"]["hamiltonian_max_hartree"], color="0.2", ls="--", lw=1.0)
    axes[2].set(xlabel="held-out point", ylabel=r"$\|\Delta\bar H\|_F$ (Eh)", yscale="log")

    link_values = np.asarray(metrics["pointwise_link_relative"])
    axes[3].bar(np.arange(len(link_values)), link_values, color=colors[2])
    axes[3].axhline(metrics["gates"]["link_relative_max"], color="0.2", ls="--", lw=1.0)
    axes[3].set(xlabel="held-out link", ylabel="relative raw-link error", yscale="log")
    for label, axis in zip("abcd", axes):
        axis.text(-0.12, 1.02, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_sector_audit(names, parity, compatible, output):
    """Plot the selected-root plane-symmetry content before fitting."""

    compatible = set(map(int, compatible))
    arrays = (
        np.asarray([value[0] for value in parity]),
        np.asarray([value[1] for value in parity]),
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(7.2, max(3.0, 0.25 * len(names))),
        sharey=True, constrained_layout=True,
    )
    for axis, values, title in zip(axes, arrays, ("singlet roots", "triplet roots")):
        axis.imshow(values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        for row in range(len(names)):
            for column in range(values.shape[1]):
                axis.text(
                    column, row, "+" if values[row, column] > 0 else "−",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(values[row, column]) > 0.5 else "black",
                )
            if row not in compatible:
                axis.add_patch(
                    plt.Rectangle(
                        (-0.48, row - 0.48), values.shape[1] - 0.04, 0.96,
                        fill=False, edgecolor="#D55E00", linewidth=1.4,
                    )
                )
        axis.set(
            title=title,
            xlabel="energy-ordered root",
            xticks=np.arange(values.shape[1]),
        )
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set(yticks=np.arange(len(names)), yticklabels=names)
    axes[0].set_ylabel("electronic record")
    figure.suptitle("SO$_2$ molecular-plane reflection sectors; orange = excluded")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--validation-neighbors", type=int, default=2)
    parser.add_argument("--feature-rank", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--acquire", type=int, default=0)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--acquisition-radius", type=float, default=0.30)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    summary, database, names, records, identifiers = load_records(args.summary)
    try:
        anchor = names.index("center")
        parity = [
            plane_parities(record, summary["protocol"]["basis"])
            for record in records
        ]
        target_singlet, target_triplet, _target_info = parity[anchor]
        compatible = []
        excluded = []
        permutations = []
        for index, (singlet, triplet, _info) in enumerate(parity):
            try:
                permutation = canonical_spin_vibronic_permutation(
                    singlet, triplet, target_singlet, target_triplet
                )
            except ValueError:
                excluded.append(
                    {
                        "name": names[index],
                        "singlet_parities": singlet.tolist(),
                        "triplet_parities": triplet.tolist(),
                    }
                )
                continue
            compatible.append(index)
            permutations.append(permutation)
        sector_figure = args.output.with_name(
            args.output.name + "-state-sectors"
        )
        plot_sector_audit(names, parity, compatible, sector_figure)
        raw_records = [records[index] for index in compatible]
        names = [names[index] for index in compatible]
        identifiers = [identifiers[index] for index in compatible]
        records = []
        for raw, permutation in zip(raw_records, permutations):
            record = copy.copy(raw)
            record["h_total"] = np.asarray(raw["h_total"])[
                np.ix_(permutation, permutation)
            ]
            record["labels"] = [raw["labels"][index] for index in permutation]
            records.append(record)
        coordinates = np.asarray([record["coordinate"] for record in records])
        bounds = (
            2.55, 3.05, 0.25, np.deg2rad(100.0), np.deg2rad(140.0)
        )
        anchor = names.index("center")
        held = farthest_holdout(coordinates, bounds, args.holdout, anchor)
        train = np.asarray(
            [index for index in range(len(records)) if index not in set(held)]
        )
        local_pairs, _lengths = sparse_overlap_graph(
            coordinates[train], bounds, args.neighbors
        )
        train_pairs = train[local_pairs]
        held_pairs = validation_pairs(
            coordinates, train, held, bounds, args.validation_neighbors
        )
        needed_pairs = np.asarray(
            sorted(set(map(tuple, np.vstack((train_pairs, held_pairs))))), dtype=int
        )
        overlap_lookup = {
            tuple(pair): cached_overlap(
                database, identifiers, raw_records, int(pair[0]), int(pair[1])
            )[np.ix_(permutations[int(pair[0])], permutations[int(pair[1])])]
            for pair in needed_pairs
        }
        point_group, point_group_info = point_group_representations(
            raw_records[anchor], summary["protocol"]["basis"]
        )
        anchor_permutation = permutations[anchor]
        point_group = {
            name: value[np.ix_(anchor_permutation, anchor_permutation)]
            for name, value in point_group.items()
        }
        center_h = np.asarray(records[anchor]["h_total"])
        commutators = {
            name: float(np.linalg.norm(center_h @ value - value @ center_h))
            for name, value in point_group.items()
        }
        if max(commutators.values()) > 1.0e-8:
            raise RuntimeError(
                "stored SOC Hamiltonian fails C2v covariance at the anchor: "
                + json.dumps(commutators)
            )
        all_overlaps = np.asarray([overlap_lookup[tuple(pair)] for pair in needed_pairs])
        hamiltonians, aligned, gauges, shift, gauge_info = procrustes_fields(
            records,
            needed_pairs,
            all_overlaps,
            point_group,
            anchor,
        )
        aligned_lookup = {
            tuple(pair): value for pair, value in zip(needed_pairs, aligned)
        }
        train_links = np.asarray([aligned_lookup[tuple(pair)] for pair in train_pairs])
        held_links = np.asarray([aligned_lookup[tuple(pair)] for pair in held_pairs])
        train_map = {global_index: local for local, global_index in enumerate(train)}
        local_train_pairs = np.asarray(
            [(train_map[left], train_map[right]) for left, right in train_pairs],
            dtype=int,
        )
        preprocessing = {
            "records": len(records),
            "excluded_state_sector_records": excluded,
            "target_plane_parities": {
                "singlet": target_singlet.tolist(),
                "triplet": target_triplet.tolist(),
            },
            "state_sector_figure": str(sector_figure.with_suffix(".png")),
            "train": train.tolist(),
            "held": held.tolist(),
            "held_names": [names[index] for index in held],
            "training_links": len(train_pairs),
            "validation_links": len(held_pairs),
            "overlap_cache_hits": database.overlap_hits,
            "overlap_cache_writes": database.overlap_writes,
            "minimum_training_link_singular_value": float(
                np.min(np.linalg.svd(train_links, compute_uv=False))
            ),
            "gauge_transport": gauge_info,
            "maximum_gauge_unitarity_defect": float(
                np.max(
                    np.linalg.norm(
                        gauges.conj().swapaxes(-1, -2) @ gauges
                        - np.eye(gauges.shape[-1]),
                        axis=(1, 2),
                    )
                )
            ),
            "point_group": point_group_info,
            "anchor_commutators": commutators,
            "energy_shift_hartree": shift,
        }
        if args.preprocess_only:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".json").write_text(
                json.dumps(preprocessing, indent=2) + "\n"
            )
            print(json.dumps(preprocessing, indent=2))
            return

        grids = (
            np.linspace(float(coordinates[:, 0].min()), float(coordinates[:, 0].max()), 21),
            np.linspace(float(coordinates[:, 1].min()), float(coordinates[:, 1].max()), 21),
            np.linspace(float(coordinates[:, 2].min()), float(coordinates[:, 2].max()), 21),
        )
        fit = MACE(
            grids,
            SO2_SPECIES,
            geometry,
            hamiltonians.shape[-1],
            geometry_units="bohr",
            channels=args.channels,
            max_ell=2,
            interactions=2,
            correlation=2,
            radial_basis=4,
            radial_mlp=(args.head_width, args.head_width),
            cutoff=7.0,
        ).fit_y(
            (coordinates[train], hamiltonians[train]),
            coordinates[train],
            local_train_pairs,
            train_links,
            anchor=int(np.flatnonzero(train == anchor)[0]),
            feature_rank=args.feature_rank,
            feature_objective="links-only",
            hidden=(args.head_width, args.head_width),
            epochs=args.epochs,
            learning_rate=2.0e-3,
            weight_decay=1.0e-8,
            link_weight=5.0,
            smoothness=1.0e-5,
            sync_steps=3000,
            ambient_representation="full",
            energy_representation="direct",
            energy_objective="trace-traceless",
            coordinate_exchange=point_group["C2(x)"],
            fixed_symmetry_representations=(point_group["sigma_xy"],),
            coordinate_exchange_axes=(0, 1),
            frame_fraction=0.30,
            ambient_fraction=0.20,
            energy_frame_gradient=1.0,
            seed=args.seed,
            distill=False,
        )
        metrics = assess(
            fit, coordinates, hamiltonians, held_pairs, held_links, held
        )
        predicted_held = fit.neural_energy.predict(coordinates[held])
        metrics.update(covariance_metrics(fit, coordinates))
        metrics["held_reference_levels_ev"] = (
            np.linalg.eigvalsh(hamiltonians[held]) * au2ev
        ).reshape(-1).tolist()
        metrics["held_predicted_levels_ev"] = (
            np.linalg.eigvalsh(predicted_held) * au2ev
        ).reshape(-1).tolist()
        metrics["gates"] = {
            "hamiltonian_max_hartree": 2.0e-5,
            "link_relative_max": 2.0e-2,
            "exchange_max": 1.0e-10,
        }
        metrics["accepted"] = bool(
            metrics["hamiltonian_max_hartree"]
            <= metrics["gates"]["hamiltonian_max_hartree"]
            and metrics["link_relative_max"]
            <= metrics["gates"]["link_relative_max"]
            and max(metrics["exchange_h_max"], metrics["exchange_y_max"])
            <= metrics["gates"]["exchange_max"]
        )
        metrics["preprocessing"] = preprocessing
        metrics["model"] = {
            "epochs": args.epochs,
            "feature_rank": args.feature_rank,
            "channels": args.channels,
            "head_width": args.head_width,
            "final_loss": float(fit.history[-1]),
            "checkpoint": str(args.output.with_suffix(".pt")),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.acquire:
            acquired, acquisition = adaptive_points(
                fit.neural_feature,
                coordinates,
                bounds,
                args.acquire,
                candidate_pool=args.candidate_pool,
                seed=args.seed + 1,
                max_distance=args.acquisition_radius,
            )
            acquisition["points"] = [
                {"name": name, "coordinate": record["coordinate"].tolist()}
                for name, record in zip(names, records)
            ] + [
                {"name": f"adaptive-{index:03d}", "coordinate": value.tolist()}
                for index, value in enumerate(acquired)
            ]
            acquisition_path = args.output.with_name(
                args.output.name + "-acquire"
            ).with_suffix(".json")
            acquisition_path.write_text(json.dumps(acquisition, indent=2) + "\n")
            metrics["acquisition"] = {
                key: value for key, value in acquisition.items() if key != "points"
            }
            metrics["acquisition"]["file"] = str(acquisition_path)
        fit.save(args.output.with_suffix(".pt"))
        args.output.with_suffix(".json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        plot_result(coordinates, train, held, bounds, metrics, args.output)
        print(json.dumps(metrics, indent=2))
    finally:
        database.close()


if __name__ == "__main__":
    main()
