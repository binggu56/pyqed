#!/usr/bin/env python3
"""Fit and validate the scattered three-state phenol SA-CASSCF MACE-Y model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2ev
from pyqed.ml import MACE
from pyqed.models.phenol_coordinates import PHENOL_SPECIES, PhenolReactiveChart


HARTREE_TO_EV = au2ev
COLORS = ("#0072B2", "#D55E00", "#009E73")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def geometry(coordinate):
    chart = PhenolReactiveChart()
    value = np.array(chart.equilibrium, copy=True)
    value[:3] = coordinate
    return chart.geometry(value)


def reflection_group(
    feature_rank,
    electronic,
    coordinate_parities=(1.0, -1.0, 1.0),
):
    feature_rank = int(feature_rank)
    if feature_rank % 3:
        raise ValueError("the reflection-symmetric feature rank must be a multiple of three")
    electronic = np.asarray(electronic, dtype=complex)
    coordinate_parities = np.asarray(coordinate_parities, dtype=float)
    if (
        coordinate_parities.ndim != 1
        or not np.all(np.isin(coordinate_parities, (-1.0, 1.0)))
    ):
        raise ValueError("coordinate_parities must be a one-dimensional sequence of +/-1")
    coordinate = np.diag(coordinate_parities)
    ambient = np.kron(np.eye(feature_rank // 3), electronic)
    return {
        "coordinate_representations": np.asarray(
            (np.eye(len(coordinate_parities)), coordinate)
        ),
        "electronic_representations": np.asarray((np.eye(3), electronic)),
        "ambient_representations": np.asarray((np.eye(feature_rank), ambient)),
        "tolerance": 2.0e-7,
    }


def _predict(field, coordinates, batch=384):
    coordinates = np.asarray(coordinates, dtype=float)
    return np.concatenate(
        [field.predict(coordinates[start : start + batch]) for start in range(0, len(coordinates), batch)]
    )


def _error_metrics(predicted_h, target_h, predicted_links, target_links, mask):
    h_error = np.linalg.norm(predicted_h - target_h, axis=(1, 2)) * HARTREE_TO_EV
    predicted_energy = np.linalg.eigvalsh(predicted_h)
    target_energy = np.linalg.eigvalsh(target_h)
    spectral = np.abs(predicted_energy - target_energy) * HARTREE_TO_EV
    link_error = np.linalg.norm(predicted_links - target_links, axis=(1, 2))
    link_relative = link_error / np.maximum(
        np.linalg.norm(target_links, axis=(1, 2)), np.finfo(float).tiny
    )

    def summarize(values, selected):
        values = np.asarray(values)[np.asarray(selected, dtype=bool)]
        return {
            "count": int(len(values)),
            "mae": float(np.mean(np.abs(values))),
            "rms": float(np.sqrt(np.mean(np.abs(values) ** 2))),
            "maximum": float(np.max(np.abs(values))),
        }

    return {
        "energy_matrix_ev": summarize(h_error, mask["energy"]),
        "energy_spectral_ev": summarize(spectral.reshape(len(spectral), -1), mask["energy"]),
        "link_absolute": summarize(link_error, mask["link"]),
        "link_relative": summarize(link_relative, mask["link"]),
    }, {
        "energy_matrix_ev": h_error,
        "energy_spectral_ev": spectral,
        "link_absolute": link_error,
        "link_relative": link_relative,
    }


def evaluate(fit, data, shifted_hamiltonians):
    coordinates = data["coordinates"]
    pairs = data["pairs"]
    predicted_h = _predict(fit.neural_energy, coordinates)
    features = _predict(fit.neural_feature, coordinates)
    predicted_links = (
        features[pairs[:, 0]].conj().swapaxes(-1, -2) @ features[pairs[:, 1]]
    )
    holdout = {
        "energy": data["energy_holdout"],
        "link": data["link_holdout"],
    }
    training = {name: ~value for name, value in holdout.items()}
    holdout_metrics, pointwise = _error_metrics(
        predicted_h, shifted_hamiltonians, predicted_links, data["p_links"], holdout
    )
    training_metrics, _ = _error_metrics(
        predicted_h, shifted_hamiltonians, predicted_links, data["p_links"], training
    )
    all_metrics, _ = _error_metrics(
        predicted_h,
        shifted_hamiltonians,
        predicted_links,
        data["p_links"],
        {
            "energy": np.ones(len(coordinates), dtype=bool),
            "link": np.ones(len(pairs), dtype=bool),
        },
    )
    gram = features.conj().swapaxes(-1, -2) @ features
    isometry = np.linalg.norm(gram - np.eye(3), axis=(1, 2))
    return {
        "predicted_hamiltonian": predicted_h,
        "predicted_links": predicted_links,
        "features": features,
        "holdout": holdout_metrics,
        "training": training_metrics,
        "all": all_metrics,
        "maximum_isometry_defect": float(np.max(isometry)),
        "pointwise": pointwise,
    }


def dense_checks(fit, data, shift):
    radial_bounds = np.ptp(data["coordinates"][:, 0])
    radial = np.linspace(
        float(np.min(data["coordinates"][:, 0])),
        float(np.max(data["coordinates"][:, 0])),
        max(93, int(np.ceil(radial_bounds / 0.0125)) + 1),
    )
    torsion = np.linspace(-0.4, 0.4, 65)
    bend = np.linspace(np.deg2rad(104.0), np.deg2rad(114.0), 49)
    rr, pp = np.meshgrid(radial, torsion, indexing="ij")
    planar_coordinates = np.column_stack(
        (rr.reshape(-1), pp.reshape(-1), np.full(rr.size, np.deg2rad(108.8)))
    )
    planar_h = _predict(fit.neural_energy, planar_coordinates).reshape(
        len(radial), len(torsion), 3, 3
    )
    rr_b, tt = np.meshgrid(radial, bend, indexing="ij")
    bend_coordinates = np.column_stack(
        (rr_b.reshape(-1), np.zeros(rr_b.size), tt.reshape(-1))
    )
    bend_h = _predict(fit.neural_energy, bend_coordinates).reshape(
        len(radial), len(bend), 3, 3
    )
    reflected = planar_h[:, ::-1]
    representation = data["reflection"]
    transformed = np.einsum(
        "ab,rpbc,cd->rpad", representation, planar_h, representation, optimize=True
    )
    hermiticity = max(
        float(np.max(np.linalg.norm(planar_h - planar_h.conj().swapaxes(-1, -2), axis=(2, 3)))),
        float(np.max(np.linalg.norm(bend_h - bend_h.conj().swapaxes(-1, -2), axis=(2, 3)))),
    )
    reflection = float(np.max(np.linalg.norm(reflected - transformed, axis=(2, 3))))
    planar_energy = np.linalg.eigvalsh(planar_h) + shift
    bend_energy = np.linalg.eigvalsh(bend_h) + shift
    sampled_energy = np.linalg.eigvalsh(data["p_hamiltonian"])
    lower = float(sampled_energy.min() - 0.05)
    upper = float(sampled_energy.max() + 0.05)
    bounded = bool(
        np.min(planar_energy) >= lower
        and np.max(planar_energy) <= upper
        and np.min(bend_energy) >= lower
        and np.max(bend_energy) <= upper
    )
    return {
        "radial": radial,
        "torsion": torsion,
        "bend": bend,
        "planar_energy": planar_energy,
        "bend_energy": bend_energy,
        "maximum_hermiticity_defect": hermiticity,
        "maximum_reflection_covariance_defect": reflection,
        "dense_energies_inside_sample_envelope_plus_1p36_ev": bounded,
        "dense_energy_range_hartree": [
            float(min(np.min(planar_energy), np.min(bend_energy))),
            float(max(np.max(planar_energy), np.max(bend_energy))),
        ],
    }


def plot_diagnostics(output, data, evaluation, history, shift, fit_all=False):
    hold_energy = data["energy_holdout"]
    hold_links = data["link_holdout"]
    target = np.linalg.eigvalsh(data["p_hamiltonian"])[hold_energy]
    predicted = (
        np.linalg.eigvalsh(evaluation["predicted_hamiltonian"][hold_energy]) + shift
    )
    offset = float(np.min(target))
    figure, panels = plt.subplots(2, 2, figsize=(10.3, 7.2), constrained_layout=True)
    for state, color in enumerate(COLORS):
        panels[0, 0].scatter(
            (target[:, state] - offset) * HARTREE_TO_EV,
            (predicted[:, state] - offset) * HARTREE_TO_EV,
            s=24, facecolors="none", edgecolors=color, label=f"P{state}",
        )
    limits = [
        min(panels[0, 0].get_xlim()[0], panels[0, 0].get_ylim()[0]),
        max(panels[0, 0].get_xlim()[1], panels[0, 0].get_ylim()[1]),
    ]
    panels[0, 0].plot(limits, limits, color="0.35", ls="--", lw=1.0)
    panels[0, 0].set(
        xlim=limits, ylim=limits, xlabel="reference energy (eV)",
        ylabel="MACE energy (eV)",
        title=("Qualification-subset energies (included)" if fit_all else "Held-out electronic energies"),
    )
    panels[0, 0].legend(frameon=False, ncol=3)

    radial = data["coordinates"][:, 0]
    spectral = evaluation["pointwise"]["energy_spectral_ev"]
    for state, color in enumerate(COLORS):
        panels[0, 1].scatter(
            radial[~hold_energy], spectral[~hold_energy, state], s=12,
            color=color, alpha=0.45,
        )
        panels[0, 1].scatter(
            radial[hold_energy], spectral[hold_energy, state], s=28,
            marker="x", color=color, label=f"P{state} holdout",
        )
    panels[0, 1].set(
        yscale="log", xlabel=r"$R_{OH}$ ($\AA$)", ylabel="absolute error (eV)",
        title="Pointwise spectral errors",
    )
    panels[0, 1].legend(frameon=False, fontsize=7.5, ncol=2)

    relative = evaluation["pointwise"]["link_relative"]
    markers = ("o", "s", "^")
    labels = (r"$R_{OH}$", r"$\phi$", r"$\theta$")
    for axis, (marker, label) in enumerate(zip(markers, labels)):
        training = (data["pair_axes"] == axis) & ~hold_links
        validation = (data["pair_axes"] == axis) & hold_links
        panels[1, 0].scatter(
            np.flatnonzero(training), relative[training], s=12, marker=marker,
            facecolors="none", edgecolors=COLORS[axis], alpha=0.6,
        )
        panels[1, 0].scatter(
            np.flatnonzero(validation), relative[validation], s=32, marker="x",
            color=COLORS[axis], label=label + " holdout",
        )
    panels[1, 0].set(
        yscale="log", xlabel="overlap-graph edge", ylabel="relative Frobenius error",
        title="Endpoint-overlap reconstruction",
    )
    panels[1, 0].legend(frameon=False, fontsize=8)

    panels[1, 1].semilogy(np.arange(1, len(history) + 1), history, color="#5E3C99", lw=1.1)
    panels[1, 1].set(xlabel="epoch", ylabel="normalized loss", title="MACE-Y optimization")
    for label, panel in zip("abcd", panels.flat):
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
        panel.grid(alpha=0.18)
    png = output / "phenol_sa6_3d_mace_y_validation.png"
    pdf = output / "phenol_sa6_3d_mace_y_validation.pdf"
    figure.savefig(png, dpi=350)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def plot_surfaces(output, dense, data):
    reference = float(np.min(np.linalg.eigvalsh(data["p_hamiltonian"])[0]))
    figure, panels = plt.subplots(2, 3, figsize=(11.8, 6.8), constrained_layout=True)
    for state in range(3):
        values = (dense["planar_energy"][:, :, state] - reference) * HARTREE_TO_EV
        image = panels[0, state].contourf(
            dense["radial"], dense["torsion"], values.T, levels=36, cmap="viridis"
        )
        panels[0, state].set(
            xlabel=r"$R_{OH}$ ($\AA$)", ylabel=r"torsion $\phi$ (rad)",
            title=rf"P{state}: $\theta=108.8^\circ$",
        )
        figure.colorbar(image, ax=panels[0, state], label="relative energy (eV)")
        values = (dense["bend_energy"][:, :, state] - reference) * HARTREE_TO_EV
        image = panels[1, state].contourf(
            dense["radial"], np.rad2deg(dense["bend"]), values.T,
            levels=36, cmap="viridis",
        )
        panels[1, state].set(
            xlabel=r"$R_{OH}$ ($\AA$)", ylabel=r"bend $\theta$ (degree)",
            title=rf"P{state}: $\phi=0$",
        )
        figure.colorbar(image, ax=panels[1, state], label="relative energy (eV)")
    for label, panel in zip("abcdef", panels.flat):
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", color="white", fontweight="bold")
    figure.suptitle("Phenol three-state diagnostic-root P-gauge MACE surfaces")
    png = output / "phenol_sa6_3d_mace_y_surfaces.png"
    pdf = output / "phenol_sa6_3d_mace_y_surfaces.pdf"
    figure.savefig(png, dpi=350)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("/private/tmp/phenol_sa6_mace_dataset_20260820/phenol_sa6_3d_mace_y.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_mace_y_fit_20260820"),
    )
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--sync-steps", type=int, default=4000)
    parser.add_argument("--feature-rank", type=int, default=12)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--energy-weight", type=float, default=5.0)
    parser.add_argument("--link-weight", type=float, default=1.0)
    parser.add_argument("--initial", type=Path)
    parser.add_argument(
        "--reuse-initial-feature-field",
        action="store_true",
        help="reuse the initial model's endpoint frames when refining the same data set",
    )
    parser.add_argument(
        "--fit-all", action="store_true",
        help="train the production checkpoint on all qualified energies and links",
    )
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = _load(args.data)
    coordinates = data["coordinates"]
    hamiltonians = data["p_hamiltonian"]
    anchor = int(data["anchor"])
    shift = float(np.min(np.linalg.eigvalsh(hamiltonians[anchor])))
    shifted = hamiltonians - shift * np.eye(3)
    energy_train = (
        np.ones(len(coordinates), dtype=bool)
        if args.fit_all else ~data["energy_holdout"]
    )
    link_train = (
        np.ones(len(data["pairs"]), dtype=bool)
        if args.fit_all else ~data["link_holdout"]
    )
    grids = tuple(np.unique(coordinates[:, axis]) for axis in range(3))
    group = reflection_group(args.feature_rank, data["reflection"])
    initial_fit = (
        None
        if args.initial is None
        else MACE.load(args.initial, geometry, device="cpu", distill=False)
    )
    if args.reuse_initial_feature_field and initial_fit is None:
        raise ValueError("--reuse-initial-feature-field requires --initial")
    feature_targets = (
        None
        if not args.reuse_initial_feature_field
        else _predict(initial_fit.neural_feature, coordinates)
    )
    fit = MACE(
        grids, PHENOL_SPECIES, geometry, 3,
        chart_features=True, geometry_units="angstrom", channels=args.channels,
        max_ell=2, interactions=2, correlation=2, radial_basis=6,
        radial_mlp=(args.head_width, args.head_width), cutoff=5.0,
    ).fit_y(
        (coordinates[energy_train], shifted[energy_train]),
        coordinates,
        data["pairs"][link_train],
        data["p_links"][link_train],
        feature_targets=feature_targets,
        feature_rank=args.feature_rank,
        anchor=anchor,
        feature_objective="links-only",
        hidden=(args.head_width, args.head_width),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=1.0e-8,
        energy_weight=args.energy_weight,
        link_weight=args.link_weight,
        smoothness=1.0e-5,
        sync_steps=args.sync_steps,
        finite_group=group,
        frame_fraction=0.30 if initial_fit is None else 0.0,
        ambient_fraction=0.20 if initial_fit is None else 0.0,
        energy_representation="direct",
        initial_fit=initial_fit,
        seed=args.seed,
        distill=False,
    )
    evaluation = evaluate(fit, data, shifted)
    dense = dense_checks(fit, data, shift)
    checkpoint = fit.save(args.output / "phenol_sa6_3d_mace_y.pt")
    diagnostic_png, diagnostic_pdf = plot_diagnostics(
        args.output, data, evaluation, fit.history, shift, args.fit_all
    )
    surface_png, surface_pdf = plot_surfaces(args.output, dense, data)
    acceptance = evaluation["all" if args.fit_all else "holdout"]
    prefix = "all_data" if args.fit_all else "heldout"
    gates = {
        f"{prefix}_spectral_rms_below_0p05_ev": acceptance["energy_spectral_ev"]["rms"] <= 0.05,
        f"{prefix}_spectral_max_below_0p15_ev": acceptance["energy_spectral_ev"]["maximum"] <= 0.15,
        f"{prefix}_link_relative_rms_below_0p15": acceptance["link_relative"]["rms"] <= 0.15,
        f"{prefix}_link_relative_max_below_0p35": acceptance["link_relative"]["maximum"] <= 0.35,
        "exact_isometric_endpoint_frames": evaluation["maximum_isometry_defect"] <= 2.0e-5,
        "hermitian_dense_hamiltonian": dense["maximum_hermiticity_defect"] <= 1.0e-10,
        "reflection_covariant_dense_hamiltonian": dense["maximum_reflection_covariance_defect"] <= 1.0e-8,
        "bounded_dense_energy": dense["dense_energies_inside_sample_envelope_plus_1p36_ev"],
    }
    summary = {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "fit_seconds": time.perf_counter() - started,
        "model": {
            "backend": "MACE-Y scattered endpoint field",
            "epochs": args.epochs,
            "feature_rank": args.feature_rank,
            "channels": args.channels,
            "head_width": args.head_width,
            "energy_weight": args.energy_weight,
            "link_weight": args.link_weight,
            "warm_started_from": None if args.initial is None else str(args.initial),
            "reused_initial_feature_field": bool(args.reuse_initial_feature_field),
            "fit_all_qualified_data": args.fit_all,
            "energy_representation": "direct Hermitian P-gauge Hamiltonian",
            "reflection": "exact finite-group covariance",
            "energy_shift_hartree": shift,
            "training_energy_points": int(np.count_nonzero(energy_train)),
            "heldout_energy_points": int(np.count_nonzero(~energy_train)),
            "training_links": int(np.count_nonzero(link_train)),
            "heldout_links": int(np.count_nonzero(~link_train)),
            "synchronization": fit.info["synchronization"],
            "initial_loss": float(fit.history[0]),
            "final_loss": float(fit.history[-1]),
            "final_normalized_parts": fit.losses[-1],
        },
        "validation": {
            "holdout": evaluation["holdout"],
            "training": evaluation["training"],
            "all": evaluation["all"],
            "maximum_isometry_defect": evaluation["maximum_isometry_defect"],
        },
        "dense_checks": {
            key: value for key, value in dense.items()
            if key not in {"radial", "torsion", "bend", "planar_energy", "bend_energy"}
        },
        "artifacts": {
            "data": str(args.data),
            "checkpoint": str(checkpoint),
            "validation_figure": str(diagnostic_png),
            "validation_figure_pdf": str(diagnostic_pdf),
            "surface_figure": str(surface_png),
            "surface_figure_pdf": str(surface_pdf),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
