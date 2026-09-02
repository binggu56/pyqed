#!/usr/bin/env python3
"""Generate a reproducible expanded-domain H3+ FCI sampling database.

The database contains six spin-pure singlet FCI roots, Procrustes-gauged
Hamiltonians for the S1/S2 manifold, and raw nonunitary overlap links.  The
sampling combines the harmonic packet measure, an ellipsoidal shell, uniform
core coverage, explicit coverage outside the previous chart, and independent
local validation pairs.  S3-equivalent coordinates and links are reduced to
canonical representatives before any electronic-structure calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from jax import numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.special import ndtri
from scipy.stats import qmc

from pyqed.ldr import AbInitioFit, Coord
from pyqed.ldr.sampling_symmetry import FiniteGroupSamplingSymmetry
from pyqed.qchem import Molecule
from pyqed.units import au2ev


SPECIES = ("H", "H", "H")
EQUILIBRIUM_BOHR = 1.7016208760233922
PACKET_WIDTHS_BOHR = np.asarray(
    (0.13127146917565208, 0.15002682626300357, 0.15002692033866494)
)
OLD_BOUNDS = np.asarray(((-0.80, 0.80), (-1.00, 1.00), (-1.00, 1.00)))
EXPANDED_BOUNDS = ((-1.00, 1.20), (-1.25, 1.25), (-1.25, 1.25))


def geometry(q):
    """Non-folding S3-covariant strain chart used by the existing MACE fit."""

    root3 = jnp.sqrt(3.0)
    triangle = jnp.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )
    qs, qx, qy = q
    radius = jnp.sqrt(qx**2 + qy**2 + 1.0e-16)
    distortion_limit = 0.65
    amplitude = distortion_limit * jnp.tanh(radius / distortion_limit)
    traceless = jnp.asarray(((qx, qy), (qy, -qx)))
    strain = (
        jnp.cosh(amplitude / EQUILIBRIUM_BOHR) * jnp.eye(2)
        + jnp.sinh(amplitude / EQUILIBRIUM_BOHR) / radius * traceless
    )
    transform = jnp.exp(qs / EQUILIBRIUM_BOHR) * strain
    return triangle.at[:, :2].set(
        EQUILIBRIUM_BOHR * triangle[:, :2] @ transform
    )


def s3_sampling_symmetry():
    angle = 2.0 * np.pi / 3.0
    rotation = np.eye(3)
    rotation[1:, 1:] = (
        (np.cos(angle), -np.sin(angle)),
        (np.sin(angle), np.cos(angle)),
    )
    reflection = np.diag((1.0, 1.0, -1.0))
    representations = np.asarray(
        [np.linalg.matrix_power(rotation, power) for power in range(3)]
        + [
            reflection @ np.linalg.matrix_power(rotation, power)
            for power in range(3)
        ]
    )
    return FiniteGroupSamplingSymmetry(
        representations,
        name="S3",
        operations=("e", "r", "r2", "s", "sr", "sr2"),
        origin=np.zeros(3),
        tolerance=2.0e-7,
    )


def sobol_normal(seed, count, radius):
    """Truncated Gaussian Sobol points scaled by harmonic packet widths."""

    count = int(count)
    if count == 0:
        return np.empty((0, 3), dtype=float)
    power = int(np.ceil(np.log2(max(32, 6 * count))))
    unit = qmc.Sobol(3, scramble=True, seed=int(seed)).random_base2(power)
    normal = ndtri(np.clip(unit, 1.0e-10, 1.0 - 1.0e-10))
    normal = normal[np.linalg.norm(normal, axis=1) <= float(radius)]
    if len(normal) < count:
        raise RuntimeError("Sobol rejection pool was too small")
    return normal[:count] * PACKET_WIDTHS_BOHR


def ellipsoidal_shell(seed, count, radius):
    if int(count) == 0:
        return np.empty((0, 3), dtype=float)
    random = np.random.default_rng(int(seed))
    directions = random.normal(size=(int(count), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(radius) * directions * PACKET_WIDTHS_BOHR


def uniform_box(seed, count, bounds=EXPANDED_BOUNDS, margin=0.0):
    if int(count) == 0:
        return np.empty((0, 3), dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    lower = bounds[:, 0] + float(margin)
    upper = bounds[:, 1] - float(margin)
    if np.any(upper <= lower):
        raise ValueError("margin leaves an empty sampling box")
    power = int(np.ceil(np.log2(max(32, int(count)))))
    unit = qmc.Sobol(3, scramble=True, seed=int(seed)).random_base2(power)
    return qmc.scale(unit[: int(count)], lower, upper)


def uniform_outer_shell(seed, count, margin=0.0):
    """Uniform points in the expanded box but outside the previous chart."""

    count = int(count)
    if count == 0:
        return np.empty((0, 3), dtype=float)
    expanded = np.asarray(EXPANDED_BOUNDS, dtype=float)
    lower = expanded[:, 0] + float(margin)
    upper = expanded[:, 1] - float(margin)
    power = int(np.ceil(np.log2(max(128, 6 * count))))
    unit = qmc.Sobol(3, scramble=True, seed=int(seed)).random_base2(power)
    candidates = qmc.scale(unit, lower, upper)
    outside = np.any(
        (candidates < OLD_BOUNDS[:, 0]) | (candidates > OLD_BOUNDS[:, 1]),
        axis=1,
    )
    selected = candidates[outside]
    if len(selected) < count:
        raise RuntimeError("outer-shell Sobol pool was too small")
    return selected[:count]


def graph_pairs(coordinates, neighbors=3):
    coordinates = np.asarray(coordinates, dtype=float)
    scale = np.ptp(coordinates, axis=0)
    scale[scale < 1.0e-12] = 1.0
    distances = cdist(coordinates / scale, coordinates / scale)
    tree = minimum_spanning_tree(distances).tocoo()
    pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(tree.row, tree.col)
    }
    np.fill_diagonal(distances, np.inf)
    for left in range(len(coordinates)):
        nearest = np.argpartition(distances[left], neighbors - 1)[:neighbors]
        pairs.update(tuple(sorted((left, int(right)))) for right in nearest)
    return np.asarray(sorted(pairs), dtype=int)


def validation_pairs(seed, count, step=0.08):
    """Independent packet, full-box, and explicit outer-shell local pairs."""

    count = int(count)
    parts = (count // 3, count // 3, count - 2 * (count // 3))
    margin = float(step) + 1.0e-8
    centers = np.vstack(
        (
            sobol_normal(seed, parts[0], radius=5.5),
            uniform_box(seed + 1, parts[1], margin=margin),
            uniform_outer_shell(seed + 2, parts[2], margin=margin),
        )
    )
    bounds = np.asarray(EXPANDED_BOUNDS, dtype=float)
    centers = np.clip(centers, bounds[:, 0] + margin, bounds[:, 1] - margin)
    random = np.random.default_rng(int(seed) + 3)
    directions = random.normal(size=centers.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    endpoints = centers + float(step) * directions
    coordinates = np.empty((2 * count, 3), dtype=float)
    coordinates[0::2] = centers
    coordinates[1::2] = endpoints
    pairs = np.column_stack((2 * np.arange(count), 2 * np.arange(count) + 1))
    return coordinates, pairs


def state_gap_diagnostics(sampler, coordinates):
    roots = []
    for coordinate in np.asarray(coordinates, dtype=float):
        record = sampler.database.get(
            {
                "geometry": sampler.coord.cartesian(tuple(coordinate)),
                "protocol": sampler.protocol,
            }
        )
        if record is None:
            raise RuntimeError("a sampled electronic record is missing")
        values = record[1] if isinstance(record, tuple) else record["energies"]
        roots.append(np.asarray(values, dtype=float))
    roots = np.asarray(roots)
    lower_gap = roots[:, 1] - roots[:, 0]
    upper_gap = roots[:, 3] - roots[:, 2]
    nearest = np.minimum(lower_gap, upper_gap)
    return roots, {
        "minimum_excluded_root_gap_hartree": float(np.min(nearest)),
        "minimum_excluded_root_gap_ev": float(np.min(nearest) * au2ev),
        "one_percent_excluded_root_gap_hartree": float(
            np.quantile(nearest, 0.01)
        ),
        "median_excluded_root_gap_hartree": float(np.median(nearest)),
    }


def link_diagnostics(links):
    singular = np.linalg.svd(np.asarray(links), compute_uv=False)[:, -1]
    return singular, {
        "minimum_link_singular_value": float(np.min(singular)),
        "one_percent_link_singular_value": float(np.quantile(singular, 0.01)),
        "median_link_singular_value": float(np.median(singular)),
        "fraction_links_below_0_9": float(np.mean(singular < 0.9)),
        "maximum_projector_loss": float(np.max(1.0 - singular**2)),
    }


def plot_sampling(training, validation, pairs, singular, output):
    figure, panels = plt.subplots(1, 3, figsize=(9.2, 2.9), constrained_layout=True)
    panels[0].scatter(training[:, 1], training[:, 2], s=9, alpha=0.65, label="training")
    panels[0].scatter(
        validation[:, 1], validation[:, 2], s=10, marker="x", alpha=0.6,
        label="validation",
    )
    panels[0].set(
        xlabel=r"$Q_x$ / bohr", ylabel=r"$Q_y$ / bohr",
        title=r"$S_3$ quotient samples", xlim=EXPANDED_BOUNDS[1],
        ylim=EXPANDED_BOUNDS[2],
    )
    panels[0].legend(frameon=False)
    radius = np.linalg.norm(training[:, 1:], axis=1)
    panels[1].scatter(training[:, 0], radius, s=9, alpha=0.65)
    panels[1].axvline(OLD_BOUNDS[0, 0], color="0.4", linestyle="--", linewidth=0.8)
    panels[1].axvline(OLD_BOUNDS[0, 1], color="0.4", linestyle="--", linewidth=0.8)
    panels[1].set(
        xlabel=r"$Q_s$ / bohr", ylabel=r"$\sqrt{Q_x^2+Q_y^2}$ / bohr",
        title="Core and expanded shell",
    )
    lengths = np.linalg.norm(training[pairs[:, 1]] - training[pairs[:, 0]], axis=1)
    scatter = panels[2].scatter(lengths, singular, c=singular, s=10, cmap="viridis")
    panels[2].axhline(0.9, color="#D55E00", linestyle="--", linewidth=0.9)
    panels[2].set(
        xlabel="link length / bohr", ylabel="minimum singular value",
        title="Training-link retention", ylim=(0.0, 1.02),
    )
    figure.colorbar(scatter, ax=panels[2], shrink=0.8)
    for panel in panels:
        panel.grid(alpha=0.15)
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "data" / "h3plus_fci_augccpvdz" / "expanded_dataset_v1",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--packet-budget", type=int, default=480)
    parser.add_argument("--ellipsoid-shell-budget", type=int, default=180)
    parser.add_argument("--uniform-budget", type=int, default=300)
    parser.add_argument("--outer-shell-budget", type=int, default=600)
    parser.add_argument("--validation-pair-budget", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database = args.output_dir / "electronic.sqlite"

    coord = Coord(to_cartesian=geometry, bounds=EXPANDED_BOUNDS)
    mol = Molecule(
        atom=list(zip(SPECIES, np.asarray(geometry((0.0, 0.0, 0.0))))),
        charge=1, spin=0, unit="bohr", basis="aug-cc-pvdz",
    ).build(eri="dense")
    mean_field = mol.RHF().run()
    electronic = mol.casci(
        mol.nao, 2, nstates=6, ms2=0, multiplicity=1, mf=mean_field
    ).run(nstates=6)
    root_s2 = np.asarray([electronic.spin_square(root) for root in range(6)])
    if np.max(np.abs(root_s2)) > 1.0e-7:
        raise RuntimeError(f"non-singlet CASCI root detected: S^2={root_s2}")
    sampler = AbInitioFit(
        electronic,
        coord=coord,
        states=(1, 2),
        nroots=6,
        database=database,
        symmetry=s3_sampling_symmetry(),
        workers=int(args.workers),
        progress=False,
    )

    reduced = sampler.reduced_size
    calibration = sampler.orbit(np.asarray((0.0, 0.065, 0.027)))
    requested_parts = {
        "origin": np.zeros((1, 3)),
        "calibration_orbit": calibration,
        "packet": sobol_normal(19, reduced(args.packet_budget), radius=5.5),
        "ellipsoid_shell": ellipsoidal_shell(
            29, reduced(args.ellipsoid_shell_budget), radius=5.5
        ),
        "uniform": uniform_box(39, reduced(args.uniform_budget)),
        "outer_shell": uniform_outer_shell(
            49, reduced(args.outer_shell_budget)
        ),
    }
    training = sampler.reduce_coordinates(np.vstack(tuple(requested_parts.values())))
    training_pairs = graph_pairs(training)
    validation, validation_link_pairs = validation_pairs(
        119, reduced(args.validation_pair_budget)
    )
    validation, validation_link_pairs = sampler.reduce_pairs(
        validation, validation_link_pairs
    )

    np.save(args.output_dir / "training_coordinates.npy", training)
    np.save(args.output_dir / "training_pairs.npy", training_pairs)
    np.save(args.output_dir / "validation_coordinates.npy", validation)
    np.save(args.output_dir / "validation_pairs.npy", validation_link_pairs)
    manifest = {
        "system": "H3+",
        "electronic_structure": "spin-pure singlet FCI(2e,27o)/aug-cc-pVDZ",
        "roots_solved": 6,
        "selected_states": [1, 2],
        "coordinate_labels": ["Qs", "Qx", "Qy"],
        "bounds_bohr": [list(value) for value in EXPANDED_BOUNDS],
        "previous_bounds_bohr": OLD_BOUNDS.tolist(),
        "equilibrium_bond_bohr": EQUILIBRIUM_BOHR,
        "packet_widths_bohr": PACKET_WIDTHS_BOHR.tolist(),
        "sampling": {
            key: int(len(value)) for key, value in requested_parts.items()
        },
        "full_domain_budgets": {
            "packet": args.packet_budget,
            "ellipsoid_shell": args.ellipsoid_shell_budget,
            "uniform": args.uniform_budget,
            "outer_shell": args.outer_shell_budget,
            "validation_pairs": args.validation_pair_budget,
        },
        "symmetry": {
            "group": sampler.group,
            "coord_irreps": sampler.coord_irreps,
            "order": 1 if sampler.coord_repr is None else int(len(sampler.coord_repr)),
        },
        "training_coordinates": int(len(training)),
        "training_pairs": int(len(training_pairs)),
        "validation_coordinates": int(len(validation)),
        "validation_pairs": int(len(validation_link_pairs)),
        "seeds": {
            "packet": 19, "ellipsoid_shell": 29, "uniform": 39,
            "outer_shell": 49, "validation": 119,
        },
        "validation_pair_step_bohr": 0.08,
        "database": str(database),
    }
    (args.output_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)

    started = perf_counter()
    print("sampling training Hamiltonians and overlap graph", flush=True)
    training_fields = sampler.continuous_fields(training, training_pairs)
    training_seconds = perf_counter() - started
    started = perf_counter()
    print("sampling independent validation pairs", flush=True)
    validation_fields = sampler.continuous_fields(
        validation, validation_link_pairs
    )
    validation_seconds = perf_counter() - started

    training_singular, training_link_info = link_diagnostics(
        training_fields["links"]
    )
    validation_singular, validation_link_info = link_diagnostics(
        validation_fields["links"]
    )
    training_roots, training_gap_info = state_gap_diagnostics(sampler, training)
    validation_roots, validation_gap_info = state_gap_diagnostics(
        sampler, validation
    )
    np.savez_compressed(
        args.output_dir / "sampled_fields.npz",
        training_coordinates=training,
        training_pairs=training_pairs,
        training_hamiltonians=training_fields["hamiltonians"],
        training_links=training_fields["links"],
        training_root_energies=training_roots,
        validation_coordinates=validation,
        validation_pairs=validation_link_pairs,
        validation_hamiltonians=validation_fields["hamiltonians"],
        validation_links=validation_fields["links"],
        validation_root_energies=validation_roots,
    )
    figure = args.output_dir / "sampling_coverage_and_links.png"
    plot_sampling(training, validation, training_pairs, training_singular, figure)
    report = {
        **manifest,
        "training_link_diagnostics": training_link_info,
        "validation_link_diagnostics": validation_link_info,
        "training_state_gap_diagnostics": training_gap_info,
        "validation_state_gap_diagnostics": validation_gap_info,
        "database_stats": dict(sampler.database.stats),
        "field_sampling_stats": {
            "training": training_fields["stats"],
            "validation": validation_fields["stats"],
        },
        "timings_seconds": {
            "training_fields": training_seconds,
            "validation_fields": validation_seconds,
            "total_fields": training_seconds + validation_seconds,
        },
        "sampling_figure": str(figure),
        "accepted_for_mace_training": bool(
            validation_link_info["minimum_link_singular_value"] >= 0.9
            and validation_gap_info["minimum_excluded_root_gap_hartree"] > 0.0
        ),
    }
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    sampler.close()


if __name__ == "__main__":
    main()
