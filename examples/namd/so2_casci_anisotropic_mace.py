#!/usr/bin/env python3
"""Fit SO2 endpoint fields on a Sobol-bond x bend-line sparse design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from examples.namd.generate_so2_casci_singlets import (
    electronic_metadata,
    electronic_structure,
    geometry,
    require_spin_pure_singlets,
    so2_point_group_representations,
    validate_electronic_metadata,
)
from examples.namd.so2_casci_sobol_mace import (
    invariant_coordinates,
    sparse_overlap_graph,
    validation_data,
)
from examples.namd.so2_casci_sobol_probes import endpoint_metrics
from pyqed.ldr.overlap import procrustes
from pyqed.ml import MACE


def anisotropic_design(bond_count, theta_count, bounds, seed):
    """Return a nested exchange-reduced bond design crossed with a bend line."""
    bond_count = int(bond_count)
    theta_count = int(theta_count)
    if bond_count < 2 or theta_count < 2:
        raise ValueError("the design needs at least two bond and bend samples")
    r_min, r_max, theta_min, theta_max = map(float, bounds)
    needed = bond_count - 1
    unit = qmc.Sobol(2, scramble=True, seed=int(seed)).random_base2(
        int(np.ceil(np.log2(needed)))
    )[:needed]
    sampled = r_min + (r_max - r_min) * unit
    bonds = np.concatenate((
        [[0.5 * (r_min + r_max), 0.5 * (r_min + r_max)]],
        np.column_stack((np.max(sampled, axis=1), np.min(sampled, axis=1))),
    ))
    theta = np.linspace(theta_min, theta_max, theta_count)
    coordinates = np.asarray([
        (r1, r2, angle) for r1, r2 in bonds for angle in theta
    ])
    anchor = theta_count // 2
    return coordinates, bonds, theta, anchor


def anisotropic_links(coordinates, bonds, theta, bounds, neighbors):
    """Connect bend lines and a sparse bond graph at each bend value."""
    nbonds, ntheta = len(bonds), len(theta)
    bend_pairs = [
        (bond * ntheta + angle, bond * ntheta + angle + 1)
        for bond in range(nbonds) for angle in range(ntheta - 1)
    ]
    mid = 0.5 * (bounds[2] + bounds[3])
    representatives = np.column_stack((bonds, np.full(nbonds, mid)))
    bond_graph, _lengths = sparse_overlap_graph(representatives, bounds, neighbors)
    bond_pairs = [
        (left * ntheta + angle, right * ntheta + angle)
        for left, right in bond_graph for angle in range(ntheta)
    ]
    pairs = np.asarray(sorted(set(bend_pairs + bond_pairs)), dtype=int)
    axes = np.asarray([
        2 if (right - left) == 1 else -1 for left, right in pairs
    ], dtype=int)
    return pairs, axes


def generate(coordinates, pairs, anchor, args):
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
    anchor_overlaps = np.asarray([model.overlap(models[anchor]) for model in models])
    return np.asarray(energies), np.asarray(spin_square), overlaps, anchor_overlaps


def symmetry_block_procrustes(value, representation):
    """Take the Procrustes factor independently in each exchange sector."""

    value = np.asarray(value, dtype=complex)
    representation = np.asarray(representation, dtype=complex)
    if representation.ndim == 2:
        representation = representation[None, ...]
    if representation.ndim != 3 or any(
        not np.allclose(value, np.diag(np.diag(value)), atol=1.0e-8)
        for value in representation
    ):
        raise ValueError("SO2 point-group representations must be parity diagonal")
    characters = np.where(
        np.real(np.diagonal(representation, axis1=1, axis2=2)) >= 0.0, 1, -1
    ).T
    rotation = np.zeros_like(value)
    for sector in sorted(set(map(tuple, characters))):
        indices = np.flatnonzero(np.all(characters == sector, axis=1))
        if len(indices):
            rotation[np.ix_(indices, indices)] = procrustes(
                value[np.ix_(indices, indices)]
            )[0]
    return rotation


def align_with_exchange_symmetry(
    energies,
    coordinates,
    pairs,
    overlaps,
    anchor_overlaps,
    exchange_representation,
    fixed_representations=(),
    *,
    anchor,
):
    """Build one anchor gauge while preserving parity on the fixed set."""

    symmetric = np.isclose(coordinates[:, 0], coordinates[:, 1], atol=1.0e-12)
    gauges = procrustes(anchor_overlaps)[0]
    fixed_representations = np.asarray(fixed_representations, dtype=complex)
    if fixed_representations.size == 0:
        fixed_representations = np.empty(
            (0, len(exchange_representation), len(exchange_representation)),
            dtype=complex,
        )
    if len(fixed_representations):
        for point in range(len(coordinates)):
            gauges[point] = symmetry_block_procrustes(
                anchor_overlaps[point], fixed_representations
            )
    full_representations = np.concatenate((
        np.asarray(exchange_representation, dtype=complex)[None, ...],
        fixed_representations,
    ))
    for point in np.flatnonzero(symmetric):
        gauges[point] = symmetry_block_procrustes(
            anchor_overlaps[point], full_representations
        )
    shift = float(energies[int(anchor), 0])
    diagonal = np.asarray([np.diag(values - shift) for values in energies])
    hamiltonians = gauges.conj().swapaxes(-1, -2) @ diagonal @ gauges
    aligned_links = np.asarray([
        gauges[left].conj().T @ value @ gauges[right]
        for (left, right), value in zip(pairs, overlaps)
    ])
    commutators = np.linalg.norm(
        hamiltonians[symmetric] @ exchange_representation
        - exchange_representation @ hamiltonians[symmetric],
        axis=(1, 2),
    )
    fixed_commutators = [
        np.linalg.norm(
            hamiltonians @ representation - representation @ hamiltonians,
            axis=(1, 2),
        )
        for representation in fixed_representations
    ]
    return hamiltonians, aligned_links, gauges, shift, {
        "symmetric_training_points": int(np.count_nonzero(symmetric)),
        "fixed_hamiltonian_commutator_max": float(np.max(commutators)),
        "global_fixed_commutator_max": [
            float(np.max(value)) for value in fixed_commutators
        ],
    }


def exchange_covariance_metrics(fit, coordinates):
    """Measure exact electronic and ambient covariance of fitted endpoints."""

    coordinates = np.asarray(coordinates, dtype=float)
    canonical = coordinates[coordinates[:, 0] > coordinates[:, 1] + 1.0e-12]
    exchanged = canonical.copy()
    exchanged[:, [0, 1]] = exchanged[:, [1, 0]]
    electronic = fit.coordinate_exchange_["electronic_representation"]
    ambient = fit.coordinate_exchange_["ambient_representation"]
    feature = fit.neural_feature.predict(canonical)
    feature_exchanged = fit.neural_feature.predict(exchanged)
    energy = fit.neural_energy.predict(canonical)
    energy_exchanged = fit.neural_energy.predict(exchanged)
    expected_feature = np.einsum(
        "ab,nbi,ij->naj", ambient, feature, electronic, optimize=True
    )
    expected_energy = electronic.conj().T @ energy @ electronic
    feature_error = np.linalg.norm(
        feature_exchanged - expected_feature, axis=(1, 2)
    ) / np.maximum(np.linalg.norm(feature, axis=(1, 2)), np.finfo(float).tiny)
    energy_error = np.linalg.norm(
        energy_exchanged - expected_energy, axis=(1, 2)
    ) / np.maximum(np.linalg.norm(energy, axis=(1, 2)), np.finfo(float).tiny)

    fixed_coordinates = coordinates[
        np.isclose(coordinates[:, 0], coordinates[:, 1], atol=1.0e-12)
    ]
    fixed_feature = fit.neural_feature.predict(fixed_coordinates)
    fixed_energy = fit.neural_energy.predict(fixed_coordinates)
    fixed_feature_error = np.linalg.norm(
        np.einsum(
            "ab,nbi,ij->naj",
            ambient,
            fixed_feature,
            electronic,
            optimize=True,
        ) - fixed_feature,
        axis=(1, 2),
    )
    fixed_energy_error = np.linalg.norm(
        fixed_energy @ electronic - electronic @ fixed_energy,
        axis=(1, 2),
    )
    metrics = {
        "exchange_feature_covariance_rms": float(
            np.sqrt(np.mean(feature_error**2))
        ),
        "exchange_feature_covariance_max": float(np.max(feature_error)),
        "exchange_energy_covariance_rms": float(
            np.sqrt(np.mean(energy_error**2))
        ),
        "exchange_energy_covariance_max": float(np.max(energy_error)),
        "fixed_feature_intertwining_max": float(np.max(fixed_feature_error)),
        "fixed_energy_commutator_max": float(np.max(fixed_energy_error)),
    }
    fixed_electronic = fit.coordinate_exchange_.get(
        "fixed_electronic_representations", ()
    )
    fixed_ambient = fit.coordinate_exchange_.get(
        "fixed_ambient_representations", ()
    )
    all_feature = fit.neural_feature.predict(coordinates)
    all_energy = fit.neural_energy.predict(coordinates)
    fixed_feature_residuals = []
    fixed_energy_residuals = []
    for ambient_value, electronic_value in zip(fixed_ambient, fixed_electronic):
        fixed_feature_residuals.append(float(np.max(np.linalg.norm(
            np.einsum(
                "ab,nbi,ij->naj",
                ambient_value,
                all_feature,
                electronic_value,
                optimize=True,
            ) - all_feature,
            axis=(1, 2),
        ))))
        fixed_energy_residuals.append(float(np.max(np.linalg.norm(
            all_energy @ electronic_value - electronic_value @ all_energy,
            axis=(1, 2),
        ))))
    metrics.update({
        "point_group_fixed_feature_intertwining_max": fixed_feature_residuals,
        "point_group_fixed_energy_commutator_max": fixed_energy_residuals,
    })
    if fixed_electronic:
        ambient_product = ambient @ fixed_ambient[0]
        electronic_product = electronic @ fixed_electronic[0]
        expected_product = np.einsum(
            "ab,nbi,ij->naj",
            ambient_product,
            feature,
            electronic_product,
            optimize=True,
        )
        metrics["point_group_composite_feature_covariance_max"] = float(np.max(
            np.linalg.norm(feature_exchanged - expected_product, axis=(1, 2))
        ))
    return metrics


def plot_result(coordinates, bonds, metrics, pointwise, bounds, output):
    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = (r"$r_1$", r"$r_2$", r"$\theta$")
    figure, axes = plt.subplots(1, 4, figsize=(12.3, 2.9), constrained_layout=True)
    scaled = invariant_coordinates(coordinates, bounds)
    for bond in range(len(bonds)):
        local = slice(bond * (len(coordinates) // len(bonds)), (bond + 1) * (len(coordinates) // len(bonds)))
        axes[0].plot(scaled[local, 0], scaled[local, 2], color="0.78", lw=0.8)
    scatter = axes[0].scatter(
        scaled[:, 0], scaled[:, 2], c=scaled[:, 1], cmap="viridis", s=24,
        edgecolor="white", linewidth=0.3,
    )
    figure.colorbar(scatter, ax=axes[0], label=r"scaled $|q_a|$")
    axes[0].set(xlabel=r"scaled $q_s$", ylabel=r"scaled $\theta$")

    x = np.arange(3)
    axes[1].bar(x, metrics["link_magnitude_axis_rms"], color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set(ylabel=r"Full-grid $|L_{ij}|$ RMS", yscale="log")
    for values, color, label in zip(pointwise, colors, labels):
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
    fixed_energy_values = metrics["point_group_fixed_energy_commutator_max"]
    fixed_feature_values = metrics["point_group_fixed_feature_intertwining_max"]
    symmetry_values = np.asarray([
        metrics["exchange_energy_covariance_max"],
        metrics["exchange_feature_covariance_max"],
        max(fixed_energy_values, default=0.0),
        max(fixed_feature_values, default=0.0),
    ])
    axes[3].bar(
        np.arange(4), np.maximum(symmetry_values, 1.0e-17),
        color=("#0072B2", "#D55E00", "#56B4E9", "#E69F00"),
    )
    axes[3].set_xticks(
        np.arange(4),
        (r"$C_2:H$", r"$C_2:Y$", r"$\sigma:H$", r"$\sigma:Y$"),
    )
    axes[3].set(ylabel="Exchange residual", yscale="log", ylim=(1.0e-17, 1.0e-11))
    for label, axis in zip("abcd", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=350)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond-centers", type=int, default=9)
    parser.add_argument("--theta-nodes", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=19)
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
    parser.add_argument("--feature-rank", type=int, default=45)
    parser.add_argument(
        "--ambient-representation",
        choices=("diagonal", "full"),
        default="diagonal",
    )
    parser.add_argument(
        "--energy-representation",
        choices=("direct", "coupled"),
        default="direct",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--radial-basis", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=64)
    parser.add_argument(
        "--reference", type=Path,
        default=Path("/private/tmp/so2_casci_singlet_5x5x5.npz"),
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("/private/tmp/so2_casci_anisotropic_9x5.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/so2_casci_anisotropic_y.png"),
    )
    args = parser.parse_args()
    bounds = (
        args.r_min, args.r_max, np.deg2rad(args.theta_min_deg),
        np.deg2rad(args.theta_max_deg),
    )
    coordinates, bonds, theta, anchor = anisotropic_design(
        args.bond_centers, args.theta_nodes, bounds, args.seed
    )
    pairs, pair_axes = anisotropic_links(
        coordinates, bonds, theta, bounds, args.neighbors
    )
    if args.dataset.is_file():
        with np.load(args.dataset, allow_pickle=False) as archive:
            validate_electronic_metadata(archive, args, label="SO2 anisotropic cache")
            if not np.allclose(archive["coordinates"], coordinates):
                raise ValueError("cached anisotropic design differs from this request")
            if not np.array_equal(archive["pairs"], pairs):
                raise ValueError("cached anisotropic graph differs from this request")
            energies = np.asarray(archive["energies"])
            spin_square = np.asarray(archive["spin_square"])
            overlaps = np.asarray(archive["overlaps"])
            anchor_overlaps = np.asarray(archive["anchor_overlaps"])
            if "point_group_representations" not in archive:
                raise ValueError(
                    "SO2 anisotropic cache lacks C2v symmetry; regenerate it"
                )
            point_group_names = tuple(map(str, archive["point_group_names"]))
            point_group_representations = np.asarray(
                archive["point_group_representations"], dtype=complex
            )
            point_group_raw = np.asarray(archive["point_group_raw"], dtype=complex)
            point_group_diagnostics = {
                "ao_metric_defects": np.asarray(
                    archive["point_group_ao_metric_defects"]
                ).tolist(),
                "state_involution_defects": np.asarray(
                    archive["point_group_state_involution_defects"]
                ).tolist(),
                "state_off_diagonal_max": np.asarray(
                    archive["point_group_state_off_diagonal_max"]
                ).tolist(),
                "generator_product_defect": float(
                    archive["point_group_generator_product_defect"]
                ),
            }
        print(f"[cache] restored {len(coordinates)} CASCI geometries")
    else:
        energies, spin_square, overlaps, anchor_overlaps = generate(
            coordinates, pairs, anchor, SimpleNamespace(**vars(args))
        )
        (
            point_group_names,
            point_group_representations,
            point_group_raw,
            raw_point_group_diagnostics,
        ) = so2_point_group_representations(
            0.5 * (args.r_min + args.r_max),
            0.5 * (bounds[2] + bounds[3]),
            SimpleNamespace(**vars(args)),
        )
        point_group_diagnostics = {
            "ao_metric_defects": [
                value["ao_metric_defect"]
                for value in raw_point_group_diagnostics["operations"]
            ],
            "state_involution_defects": [
                value["state_involution_defect"]
                for value in raw_point_group_diagnostics["operations"]
            ],
            "state_off_diagonal_max": [
                value["state_off_diagonal_max"]
                for value in raw_point_group_diagnostics["operations"]
            ],
            "generator_product_defect": raw_point_group_diagnostics[
                "generator_product_defect"
            ],
        }
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.dataset, coordinates=coordinates, bonds=bonds, theta=theta,
            anchor=anchor, pairs=pairs, pair_axes=pair_axes, energies=energies,
            spin_square=spin_square, overlaps=overlaps,
            anchor_overlaps=anchor_overlaps,
            point_group_names=np.asarray(point_group_names),
            point_group_representations=point_group_representations,
            point_group_raw=point_group_raw,
            point_group_ao_metric_defects=np.asarray(
                point_group_diagnostics["ao_metric_defects"]
            ),
            point_group_state_involution_defects=np.asarray(
                point_group_diagnostics["state_involution_defects"]
            ),
            point_group_state_off_diagonal_max=np.asarray(
                point_group_diagnostics["state_off_diagonal_max"]
            ),
            point_group_generator_product_defect=np.asarray(
                point_group_diagnostics["generator_product_defect"]
            ),
            **electronic_metadata(args),
        )

    require_spin_pure_singlets(spin_square)
    operation_index = {name: index for index, name in enumerate(point_group_names)}
    exchange_representation = point_group_representations[
        operation_index["C2(x)"]
    ]
    fixed_representations = (
        point_group_representations[operation_index["sigma_xy"]],
    )
    hamiltonians, aligned_links, _gauges, shift, alignment_diagnostics = (
        align_with_exchange_symmetry(
            energies,
            coordinates,
            pairs,
            overlaps,
            anchor_overlaps,
            exchange_representation,
            fixed_representations,
            anchor=anchor,
        )
    )
    reference_grids, validation_coordinates, validation_energies, validation_links = (
        validation_data(args.reference, args)
    )
    grids = (
        np.linspace(args.r_min, args.r_max, len(reference_grids[0])),
        np.linspace(args.r_min, args.r_max, len(reference_grids[1])),
        np.linspace(bounds[2], bounds[3], max(len(reference_grids[2]), len(theta))),
    )
    fit = MACE(
        grids, ("O", "S", "O"), lambda coordinate: geometry(*coordinate),
        args.nstates, geometry_units="bohr", channels=args.channels,
        max_ell=2, interactions=2, correlation=2, radial_basis=args.radial_basis,
        radial_mlp=(args.head_width, args.head_width), cutoff=7.0,
    ).fit_y(
        (coordinates, hamiltonians), coordinates, pairs, aligned_links,
        anchor=anchor, feature_rank=args.feature_rank,
        feature_objective="links-only", hidden=(args.head_width, args.head_width),
        epochs=args.epochs, learning_rate=2.0e-3, link_weight=5.0,
        isometry_weight=1.0, smoothness=0.0, sync_steps=5000,
        coordinate_exchange=exchange_representation,
        fixed_symmetry_representations=fixed_representations,
        coordinate_exchange_axes=(0, 1),
        ambient_representation=args.ambient_representation,
        energy_representation=args.energy_representation,
        seed=args.seed, distill=False,
    )
    metrics, pointwise = endpoint_metrics(
        fit, validation_coordinates, validation_energies, validation_links, shift
    )
    metrics.update(exchange_covariance_metrics(fit, validation_coordinates))
    metrics.update({
        "bond_centers": len(bonds),
        "theta_nodes": len(theta),
        "total_geometries": len(coordinates),
        "overlap_links": len(pairs),
        "bend_links": int(np.count_nonzero(pair_axes == 2)),
        "bond_graph_links": int(np.count_nonzero(pair_axes == -1)),
        "max_abs_spin_square": float(np.max(np.abs(spin_square))),
        "training_link_rms": fit.info["synchronization"]["rms_relative_link_error"],
        "final_loss": float(fit.history[-1]),
        "ambient_representation": args.ambient_representation,
        "energy_representation": args.energy_representation,
        "exchange_parities": np.real(np.diag(exchange_representation)).astype(int).tolist(),
        "point_group_operations": list(point_group_names),
        "point_group_characters": np.real(
            np.diagonal(point_group_representations, axis1=1, axis2=2)
        ).astype(int).tolist(),
        "point_group_raw_real": point_group_raw.real.tolist(),
        "point_group_raw_imag": point_group_raw.imag.tolist(),
        "point_group_ab_initio": point_group_diagnostics,
        "exchange_alignment": alignment_diagnostics,
        "exchange_ambient_odd_dimension": fit.info["coordinate_exchange"][
            "ambient_odd_dimension"
        ],
        "exchange_fixed_frame_rms_before_projection": fit.info[
            "coordinate_exchange"
        ]["fixed_frame_rms_before_projection"],
        "point_group_fixed_ambient_odd_dimensions": [
            value["ambient_odd_dimension"]
            for value in fit.info["coordinate_exchange"]["fixed_generators"]
        ],
    })
    fit.save(args.output.with_suffix(".pt"))
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_result(coordinates, bonds, metrics, pointwise, bounds, args.output)
    print(json.dumps(metrics, indent=2))
    print(f"dataset: {args.dataset}")
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
