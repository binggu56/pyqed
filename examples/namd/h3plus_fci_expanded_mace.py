#!/usr/bin/env python3
"""Train and qualify an S3-equivariant MACE-Y model for expanded H3+ FCI data.

The model uses the molecule-specific MACE adaptation implemented by PyQED:
an invariant MACE encoder, symmetry-projected matrix heads, and an isometric
endpoint field trained against raw nonunitary overlaps.  It is not an exact
reproduction of the interatomic-potential model of Batatia et al., NeurIPS 35,
11423 (2022), https://arxiv.org/abs/2206.07697.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr import AbInitioFit, Coord
from pyqed.ml import MACE
from pyqed.qchem import Molecule
from pyqed.units import au2ev

from h3plus_fci_expanded_dataset import (
    EQUILIBRIUM_BOHR,
    EXPANDED_BOUNDS,
    PACKET_WIDTHS_BOHR,
    SPECIES,
    geometry,
    s3_sampling_symmetry,
)


def mace_geometry(q):
    return np.asarray(geometry(np.asarray(q, dtype=float)), dtype=float)


def load_fields(path):
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def finite_group(database, feature_rank):
    """Recover the electronic representation in the database's fixed gauge."""

    coord = Coord(to_cartesian=geometry, bounds=EXPANDED_BOUNDS)
    molecule = Molecule(
        atom=list(zip(SPECIES, mace_geometry((0.0, 0.0, 0.0)))),
        charge=1,
        spin=0,
        unit="bohr",
        basis="aug-cc-pvdz",
    ).build(eri="dense")
    mean_field = molecule.RHF().run()
    electronic = molecule.casci(
        molecule.nao, 2, nstates=6, ms2=0, multiplicity=1, mf=mean_field
    ).run(nstates=6)
    sampler = AbInitioFit(
        electronic,
        coord=coord,
        states=(1, 2),
        nroots=6,
        database=database,
        symmetry=s3_sampling_symmetry(),
        workers=1,
        progress=False,
    )
    sampler._ensure_state_repr(strict=False)
    inferred = sampler.state_repr
    if inferred is None:
        raise RuntimeError(
            "the database Hamiltonians do not determine an S3 state representation"
        )
    coordinate_e = np.asarray(sampler.coord_repr)[:, 1:, 1:]
    intertwiner = sum(
        state @ coordinate.T
        for state, coordinate in zip(inferred, coordinate_e)
    )
    left, _singular, right = np.linalg.svd(intertwiner)
    basis = left @ right
    exact_state = np.asarray(
        [basis @ coordinate @ basis.conj().T for coordinate in coordinate_e]
    )
    group = sampler._symmetry.mace_group(
        exact_state, feature_rank=feature_rank, tolerance=2.0e-7
    )
    validation = {
        "hamiltonian_representation_inferred": True,
        "raw_automatic_guard": dict(sampler.state_validation),
        "maximum_exactification_change": float(
            np.max(np.linalg.norm(exact_state - inferred, axis=(1, 2)))
        ),
    }
    sampler.close()
    return group, validation


def predict(fit, coordinates, pairs):
    hamiltonians = fit.neural_energy.predict(coordinates)
    features = fit.neural_feature.predict(coordinates)
    links = (
        features[pairs[:, 0]].conj().swapaxes(-1, -2)
        @ features[pairs[:, 1]]
    )
    return hamiltonians, links


def metrics(fit, coordinates, pairs, reference_h, reference_links):
    predicted_h, predicted_links = predict(fit, coordinates, pairs)
    h_error = np.linalg.norm(predicted_h - reference_h, axis=(-2, -1))
    reference_e = np.linalg.eigvalsh(reference_h)
    predicted_e = np.linalg.eigvalsh(predicted_h)
    eigen_error = (predicted_e - reference_e) * au2ev * 1.0e3
    link_error = np.linalg.norm(
        predicted_links - reference_links, axis=(-2, -1)
    ) / np.maximum(
        np.linalg.norm(reference_links, axis=(-2, -1)), np.finfo(float).tiny
    )
    return {
        "rms_hamiltonian_error_hartree": float(np.sqrt(np.mean(h_error**2))),
        "maximum_hamiltonian_error_hartree": float(np.max(h_error)),
        "rms_eigenvalue_error_mev": float(np.sqrt(np.mean(eigen_error**2))),
        "maximum_absolute_eigenvalue_error_mev": float(np.max(np.abs(eigen_error))),
        "relative_link_error": float(
            np.linalg.norm(predicted_links - reference_links)
            / np.linalg.norm(reference_links)
        ),
        "rms_relative_link_error": float(np.sqrt(np.mean(link_error**2))),
        "maximum_relative_link_error": float(np.max(link_error)),
        "reference_eigenvalues_hartree": reference_e,
        "predicted_eigenvalues_hartree": predicted_e,
        "eigenvalue_errors_mev": eigen_error,
        "relative_link_errors": link_error,
    }


def symmetry_error(fit, group, coordinates):
    values = fit.neural_energy.predict(coordinates)
    errors = []
    for coordinate, value in zip(coordinates, values):
        for cq, electronic in zip(
            group["coordinate_representations"],
            group["electronic_representations"],
        ):
            transformed = cq @ (coordinate - group["origin"]) + group["origin"]
            prediction = fit.neural_energy.predict(transformed[None])[0]
            reference = electronic @ value @ electronic.conj().T
            errors.append(np.linalg.norm(prediction - reference))
    return float(np.max(errors))


def plot_qualification(fit, fields, validation, output):
    coordinates = fields["validation_coordinates"]
    pairs = fields["validation_pairs"]
    reference_e = validation["reference_eigenvalues_hartree"]
    predicted_e = validation["predicted_eigenvalues_hartree"]
    radius = np.linalg.norm(coordinates / PACKET_WIDTHS_BOHR, axis=1)
    losses = np.asarray(fit.history, dtype=float)

    figure, panels = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True)
    panels[0, 0].semilogy(np.arange(1, len(losses) + 1), losses)
    panels[0, 0].set(xlabel="epoch", ylabel="normalized loss", title="MACE-Y training")
    colors = ("#0072B2", "#D55E00")
    for state, color in enumerate(colors):
        panels[0, 1].scatter(
            reference_e[:, state] * au2ev,
            predicted_e[:, state] * au2ev,
            s=11,
            alpha=0.7,
            color=color,
            label=fr"$S_{state + 1}$",
        )
    limits = np.asarray(panels[0, 1].get_xlim())
    panels[0, 1].plot(limits, limits, color="0.25", linestyle="--", linewidth=0.9)
    panels[0, 1].set(
        xlabel="FCI energy / eV", ylabel="MACE energy / eV",
        title="Held-out energy parity",
    )
    panels[0, 1].legend(frameon=False)
    for state, color in enumerate(colors):
        panels[1, 0].scatter(
            radius,
            validation["eigenvalue_errors_mev"][:, state],
            s=11,
            alpha=0.7,
            color=color,
            label=fr"$S_{state + 1}$",
        )
    panels[1, 0].axhline(0.0, color="0.25", linewidth=0.8)
    panels[1, 0].set(
        xlabel=r"packet-scaled radius $|Q/\sigma|$",
        ylabel="MACE - FCI / meV",
        title="Held-out PES error",
    )
    panels[1, 1].scatter(
        np.arange(len(pairs)), validation["relative_link_errors"], s=13, alpha=0.75
    )
    panels[1, 1].axhline(0.02, color="#D55E00", linestyle="--", linewidth=0.9)
    panels[1, 1].set(
        xlabel="validation-link index (all lengths 0.08 bohr)",
        ylabel="relative Frobenius error",
        title="Held-out raw links",
    )
    for panel in panels.flat:
        panel.grid(alpha=0.15)
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def serializable(metrics):
    return {
        key: value for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def main():
    root = Path(__file__).resolve().parents[3]
    default_data = root / "data/h3plus_fci_augccpvdz/expanded_dataset_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--channels", type=int, default=20)
    parser.add_argument("--feature-rank", type=int, default=20)
    parser.add_argument("--head-width", type=int, default=96)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--warm-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--refine-epochs", type=int, default=2000)
    parser.add_argument("--lbfgs-steps", type=int, default=100)
    args = parser.parse_args()
    output = args.data_dir / "mace_y" if args.output_dir is None else args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    fields = load_fields(args.data_dir / "sampled_fields.npz")
    group, state_validation = finite_group(
        args.data_dir / "electronic.sqlite", args.feature_rank
    )
    axes = tuple(np.linspace(lower, upper, 17) for lower, upper in EXPANDED_BOUNDS)
    started = perf_counter()
    if args.initial_checkpoint is None:
        warm = (
            None if args.warm_checkpoint is None
            else MACE.load(
                args.warm_checkpoint, mace_geometry, device="cpu", distill=False
            )
        )
        stage = (
            "joint MACE-Y training" if warm is None
            else "warm MACE-Y adaptation"
        )
        fit = MACE(
            axes,
            SPECIES,
            mace_geometry,
            2,
            chart_features=True,
            chart_bounds=EXPANDED_BOUNDS,
            geometry_units="bohr",
            channels=args.channels,
            max_ell=2,
            interactions=2,
            correlation=2,
            radial_basis=10,
            radial_mlp=(args.head_width, args.head_width),
            cutoff=4.5,
        ).fit_y(
            (fields["training_coordinates"], fields["training_hamiltonians"]),
            fields["training_coordinates"],
            fields["training_pairs"],
            fields["training_links"],
            feature_rank=args.feature_rank,
            anchor=0,
            feature_objective="links-only",
            ambient_representation="full",
            energy_representation="direct",
            energy_objective="trace-traceless",
            finite_group=group,
            hidden=(args.head_width, args.head_width),
            epochs=args.epochs,
            learning_rate=1.5e-3 if warm is None else 5.0e-4,
            weight_decay=1.0e-8,
            frame_fraction=0.40 if warm is None else 0.0,
            ambient_fraction=0.20 if warm is None else 0.0,
            smoothness=1.0e-5,
            energy_weight=80.0,
            initial_fit=warm,
            seed=args.seed,
            distill=False,
        )
    else:
        stage = "frozen-encoder Hamiltonian refinement"
        fit = MACE.load(
            args.initial_checkpoint, mace_geometry, device="cpu", distill=False
        )
        fit.refine_hamiltonian(
            fields["training_coordinates"],
            fields["training_hamiltonians"],
            epochs=args.refine_epochs,
            learning_rate=1.0e-3,
            weight_decay=1.0e-8,
            lbfgs_steps=args.lbfgs_steps,
            objective="trace-traceless",
            seed=args.seed,
        )
    elapsed = perf_counter() - started

    training = metrics(
        fit,
        fields["training_coordinates"],
        fields["training_pairs"],
        fields["training_hamiltonians"],
        fields["training_links"],
    )
    validation = metrics(
        fit,
        fields["validation_coordinates"],
        fields["validation_pairs"],
        fields["validation_hamiltonians"],
        fields["validation_links"],
    )
    covariance_error = symmetry_error(
        fit, group, fields["validation_coordinates"][::10]
    )
    accepted = bool(
        validation["maximum_hamiltonian_error_hartree"] <= 7.5e-4
        and validation["relative_link_error"] <= 2.0e-2
        and covariance_error <= 2.0e-6
    )
    checkpoint = output / "h3plus_s3_mace_y_expanded.pt"
    fit.save(checkpoint)
    figure = output / "h3plus_s3_mace_y_qualification.png"
    plot_qualification(fit, fields, validation, figure)
    report = {
        "accepted_for_production": accepted,
        "stage": stage,
        "method": "S3-equivariant MACE-Y (direct H, isometric endpoint field)",
        "source_database": str(args.data_dir / "electronic.sqlite"),
        "checkpoint": str(checkpoint),
        "epochs": args.epochs,
        "hamiltonian_refinement": fit.info.get("hamiltonian_refinement"),
        "training_seconds": elapsed,
        "architecture": {
            "channels": args.channels,
            "feature_rank": args.feature_rank,
            "head_width": args.head_width,
            "interactions": 2,
            "max_ell": 2,
            "radial_basis": 10,
        },
        "samples": {
            "training_coordinates": int(len(fields["training_coordinates"])),
            "training_links": int(len(fields["training_pairs"])),
            "validation_coordinates": int(len(fields["validation_coordinates"])),
            "validation_links": int(len(fields["validation_pairs"])),
        },
        "training": serializable(training),
        "validation": serializable(validation),
        "maximum_symmetry_covariance_error_hartree": covariance_error,
        "state_representation_validation": state_validation,
        "acceptance_gates": {
            "maximum_hamiltonian_error_hartree": 7.5e-4,
            "relative_link_error": 2.0e-2,
            "maximum_symmetry_covariance_error_hartree": 2.0e-6,
        },
        "qualification_figure": str(figure),
        "initial_loss": float(fit.history[0]),
        "final_loss": float(fit.history[-1]),
    }
    (output / "qualification_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
