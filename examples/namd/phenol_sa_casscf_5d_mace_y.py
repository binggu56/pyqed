#!/usr/bin/env python3
"""Fit the corrected five-dimensional phenol SA-CASSCF P-gauge data."""

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
from scipy.stats import qmc

from examples.namd.phenol_sa_casscf_3d_mace_y import (
    HARTREE_TO_EV,
    _jsonable,
    _load,
    _predict,
    evaluate,
    reflection_group,
)
from pyqed.ml import MACE
from pyqed.models.phenol_coordinates import PHENOL_SPECIES, PhenolReactiveChart


COLORS = ("#0072B2", "#D55E00", "#009E73")
AXIS_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
AXIS_LABELS = (r"$R_{OH}$", r"$\phi$", r"$\theta$", r"$Q_{16a}$", r"$Q_{8a}$")
AXIS_MARKERS = ("o", "s", "^", "D", "v")


def anchor_index(data):
    target = np.asarray((0.95, 0.0, np.deg2rad(108.8), 0.0, 0.0))
    scales = np.asarray(data["coordinate_scales"], dtype=float)
    return int(np.argmin(np.linalg.norm((data["coordinates"] - target) / scales, axis=1)))


def dense_checks(fit, data, shift, *, seed=31):
    coordinates = data["coordinates"]
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    center = coordinates[anchor_index(data)].copy()
    radial = np.linspace(lower[0], upper[0], 165)
    torsion = np.linspace(lower[1], upper[1], 81)
    bend = np.linspace(lower[2], upper[2], 81)
    q16 = np.linspace(lower[3], upper[3], 81)
    q8 = np.linspace(lower[4], upper[4], 81)

    def surface(axis, values):
        rr, xx = np.meshgrid(radial, values, indexing="ij")
        points = np.broadcast_to(center, (rr.size, 5)).copy()
        points[:, 0] = rr.reshape(-1)
        points[:, axis] = xx.reshape(-1)
        hamiltonian = _predict(fit.neural_energy, points).reshape(
            len(radial), len(values), 3, 3
        )
        return np.linalg.eigvalsh(hamiltonian) + shift, hamiltonian

    torsion_energy, torsion_h = surface(1, torsion)
    bend_energy, bend_h = surface(2, bend)
    q16_energy, q16_h = surface(3, q16)
    q8_energy, q8_h = surface(4, q8)

    sampler = qmc.Sobol(5, scramble=True, seed=int(seed))
    probes = lower + sampler.random_base2(10) * (upper - lower)
    reflected_probes = probes * data["coordinate_parities"]
    probe_h = _predict(fit.neural_energy, probes)
    reflected_h = _predict(fit.neural_energy, reflected_probes)
    representation = np.asarray(data["reflection"])
    transformed = np.einsum(
        "ab,nbc,cd->nad",
        representation.conj().T,
        probe_h,
        representation,
        optimize=True,
    )
    hermiticity = max(
        float(np.max(np.linalg.norm(values - values.conj().swapaxes(-1, -2), axis=(2, 3))))
        for values in (torsion_h, bend_h, q16_h, q8_h)
    )
    reflection = float(np.max(np.linalg.norm(reflected_h - transformed, axis=(1, 2))))
    sampled_energy = np.linalg.eigvalsh(data["p_hamiltonian"])
    dense_energy = np.concatenate(
        (
            torsion_energy.reshape(-1, 3),
            bend_energy.reshape(-1, 3),
            q16_energy.reshape(-1, 3),
            q8_energy.reshape(-1, 3),
        )
    )
    envelope = (float(sampled_energy.min() - 0.05), float(sampled_energy.max() + 0.05))
    return {
        "radial": radial,
        "torsion": torsion,
        "bend": bend,
        "q16": q16,
        "q8": q8,
        "torsion_energy": torsion_energy,
        "bend_energy": bend_energy,
        "q16_energy": q16_energy,
        "q8_energy": q8_energy,
        "maximum_hermiticity_defect": hermiticity,
        "maximum_reflection_covariance_defect": reflection,
        "dense_energies_inside_sample_envelope_plus_1p36_ev": bool(
            dense_energy.min() >= envelope[0] and dense_energy.max() <= envelope[1]
        ),
        "dense_energy_range_hartree": [float(dense_energy.min()), float(dense_energy.max())],
    }


def plot_diagnostics(output, data, evaluation, history, shift, fit_all):
    hold_energy = data["energy_holdout"]
    hold_links = data["link_holdout"]
    target = np.linalg.eigvalsh(data["p_hamiltonian"])[hold_energy]
    predicted = np.linalg.eigvalsh(evaluation["predicted_hamiltonian"][hold_energy]) + shift
    offset = float(np.min(target))
    figure, panels = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    for state, color in enumerate(COLORS):
        panels[0, 0].scatter(
            (target[:, state] - offset) * HARTREE_TO_EV,
            (predicted[:, state] - offset) * HARTREE_TO_EV,
            s=26,
            facecolors="none",
            edgecolors=color,
            label=f"P{state}",
        )
    limits = [
        min(panels[0, 0].get_xlim()[0], panels[0, 0].get_ylim()[0]),
        max(panels[0, 0].get_xlim()[1], panels[0, 0].get_ylim()[1]),
    ]
    panels[0, 0].plot(limits, limits, color="0.35", ls="--", lw=1.0)
    panels[0, 0].set(
        xlim=limits,
        ylim=limits,
        xlabel="reference energy (eV)",
        ylabel="MACE energy (eV)",
        title="Qualification points (included)" if fit_all else "Held-out 5D energies",
    )
    panels[0, 0].legend(frameon=False, ncol=3)

    radial = data["coordinates"][:, 0]
    spectral = evaluation["pointwise"]["energy_spectral_ev"]
    for state, color in enumerate(COLORS):
        panels[0, 1].scatter(radial[~hold_energy], spectral[~hold_energy, state], s=12, color=color, alpha=0.4)
        panels[0, 1].scatter(
            radial[hold_energy], spectral[hold_energy, state], s=30, marker="x", color=color,
            label=f"P{state} holdout",
        )
    panels[0, 1].set(
        yscale="log", xlabel=r"$R_{OH}$ ($\AA$)", ylabel="absolute error (eV)",
        title="Pointwise spectral errors",
    )
    panels[0, 1].legend(frameon=False, fontsize=8, ncol=2)

    relative = evaluation["pointwise"]["link_relative"]
    for axis, (marker, label, color) in enumerate(zip(AXIS_MARKERS, AXIS_LABELS, AXIS_COLORS)):
        training = (data["pair_axes"] == axis) & ~hold_links
        validation = (data["pair_axes"] == axis) & hold_links
        panels[1, 0].scatter(
            np.flatnonzero(training), relative[training], s=12, marker=marker,
            facecolors="none", edgecolors=color, alpha=0.55,
        )
        panels[1, 0].scatter(
            np.flatnonzero(validation), relative[validation], s=34, marker="x",
            color=color, label=label + " holdout",
        )
    panels[1, 0].set(
        yscale="log", xlabel="overlap-graph edge", ylabel="relative Frobenius error",
        title="Five-dimensional overlap reconstruction",
    )
    panels[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    panels[1, 1].semilogy(np.arange(1, len(history) + 1), history, color="#5E3C99", lw=1.1)
    panels[1, 1].set(xlabel="epoch", ylabel="normalized loss", title="MACE-Y optimization")
    for label, panel in zip("abcd", panels.flat):
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
        panel.grid(alpha=0.18)
    png = output / "phenol_sa6_5d_mace_y_validation.png"
    pdf = output / "phenol_sa6_5d_mace_y_validation.pdf"
    figure.savefig(png, dpi=350)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def plot_surfaces(output, dense, data):
    reference = float(np.min(np.linalg.eigvalsh(data["p_hamiltonian"])[anchor_index(data)]))
    figure, panels = plt.subplots(4, 3, figsize=(11.7, 11.8), constrained_layout=True)
    rows = (
        (dense["torsion"], dense["torsion_energy"], r"$\phi$ (rad)"),
        (np.rad2deg(dense["bend"]), dense["bend_energy"], r"$\theta$ (deg)"),
        (dense["q16"], dense["q16_energy"], r"$Q_{16a}$ ($\AA\sqrt{amu}$)"),
        (dense["q8"], dense["q8_energy"], r"$Q_{8a}$ ($\AA\sqrt{amu}$)"),
    )
    for row, (axis, energy, ylabel) in enumerate(rows):
        for state in range(3):
            values = (energy[:, :, state] - reference) * HARTREE_TO_EV
            image = panels[row, state].contourf(
                dense["radial"], axis, values.T, levels=36, cmap="viridis"
            )
            panels[row, state].set(
                xlabel=r"$R_{OH}$ ($\AA$)", ylabel=ylabel, title=f"P{state}"
            )
            figure.colorbar(image, ax=panels[row, state], label="relative energy (eV)")
    for label, panel in zip("abcdefghijkl", panels.flat):
        panel.text(
            0.02, 0.96, label, transform=panel.transAxes, va="top",
            color="white", fontweight="bold",
        )
    figure.suptitle("Phenol corrected three-state 5D P-gauge MACE surfaces")
    png = output / "phenol_sa6_5d_mace_y_surfaces.png"
    pdf = output / "phenol_sa6_5d_mace_y_surfaces.pdf"
    figure.savefig(png, dpi=350)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("/private/tmp/phenol_sa6_5d_pilot_20260822/phenol_sa6_5d_p_gauge.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa6_5d_mace_y_fit_20260822"),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--sync-steps", type=int, default=3000)
    parser.add_argument("--feature-rank", type=int, default=12)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--head-width", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--energy-weight", type=float, default=50.0)
    parser.add_argument("--link-weight", type=float, default=0.5)
    parser.add_argument("--initial", type=Path)
    parser.add_argument(
        "--chart-bounds",
        choices=("data", "initial"),
        default="data",
        help="normalize chart coordinates to the current data or the initial model",
    )
    parser.add_argument("--reuse-initial-feature-field", action="store_true")
    parser.add_argument(
        "--focus-energy-above-ev", type=float,
        help="repeat training geometries whose initial maximum spectral error exceeds this value",
    )
    parser.add_argument("--focus-repeats", type=int, default=4)
    parser.add_argument("--fit-all", action="store_true")
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = _load(args.data)
    chart = PhenolReactiveChart(modes=data["modes"])
    geometry = chart.geometry
    coordinates = data["coordinates"]
    hamiltonians = data["p_hamiltonian"]
    anchor = anchor_index(data)
    shift = float(np.min(np.linalg.eigvalsh(hamiltonians[anchor])))
    shifted = hamiltonians - shift * np.eye(3)
    energy_train = np.ones(len(coordinates), dtype=bool) if args.fit_all else ~data["energy_holdout"]
    link_train = np.ones(len(data["pairs"]), dtype=bool) if args.fit_all else ~data["link_holdout"]
    grids = tuple(np.unique(coordinates[:, axis]) for axis in range(5))
    group = reflection_group(
        args.feature_rank, data["reflection"], coordinate_parities=data["coordinate_parities"]
    )
    initial_fit = (
        None if args.initial is None
        else MACE.load(args.initial, geometry, device="cpu", distill=False)
    )
    if args.reuse_initial_feature_field and initial_fit is None:
        raise ValueError("--reuse-initial-feature-field requires --initial")
    if args.chart_bounds == "initial" and initial_fit is None:
        raise ValueError("--chart-bounds initial requires --initial")
    feature_targets = (
        None if not args.reuse_initial_feature_field
        else _predict(initial_fit.neural_feature, coordinates)
    )
    focus = np.zeros(len(coordinates), dtype=bool)
    if args.focus_energy_above_ev is not None:
        if initial_fit is None:
            raise ValueError("--focus-energy-above-ev requires --initial")
        if args.focus_repeats < 2:
            raise ValueError("--focus-repeats must be at least two")
        initial_h = _predict(initial_fit.neural_energy, coordinates)
        initial_spectral = np.abs(
            np.linalg.eigvalsh(initial_h) - np.linalg.eigvalsh(shifted)
        ) * HARTREE_TO_EV
        focus = energy_train & (
            np.max(initial_spectral, axis=1) > float(args.focus_energy_above_ev)
        )
    energy_coordinates = coordinates[energy_train]
    energy_targets = shifted[energy_train]
    if np.any(focus):
        energy_coordinates = np.concatenate(
            (energy_coordinates, *([coordinates[focus]] * (args.focus_repeats - 1)))
        )
        energy_targets = np.concatenate(
            (energy_targets, *([shifted[focus]] * (args.focus_repeats - 1)))
        )
    fit = MACE(
        grids, PHENOL_SPECIES, geometry, 3,
        chart_features=True,
        chart_bounds=(
            initial_fit.chart_bounds
            if args.chart_bounds == "initial"
            else None
        ),
        geometry_units="angstrom", channels=args.channels,
        max_ell=2, interactions=2, correlation=2, radial_basis=6,
        radial_mlp=(args.head_width, args.head_width), cutoff=5.0,
    ).fit_y(
        (energy_coordinates, energy_targets),
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
    dense = dense_checks(fit, data, shift, seed=args.seed + 2017)
    checkpoint = fit.save(args.output / "phenol_sa6_5d_mace_y.pt")
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
            "backend": "five-dimensional MACE-Y scattered endpoint field",
            "epochs": args.epochs,
            "feature_rank": args.feature_rank,
            "channels": args.channels,
            "head_width": args.head_width,
            "energy_weight": args.energy_weight,
            "link_weight": args.link_weight,
            "warm_started_from": None if args.initial is None else str(args.initial),
            "chart_bounds": fit.chart_bounds,
            "chart_bounds_source": args.chart_bounds,
            "reused_initial_feature_field": bool(args.reuse_initial_feature_field),
            "fit_all_qualified_data": args.fit_all,
            "energy_representation": "direct Hermitian diagnostic-root P-gauge Hamiltonian",
            "reflection": "exact five-coordinate finite-group covariance",
            "energy_shift_hartree": shift,
            "anchor": anchor,
            "training_energy_points": int(np.count_nonzero(energy_train)),
            "training_energy_rows": int(len(energy_coordinates)),
            "focused_energy_points": int(np.count_nonzero(focus)),
            "focus_energy_above_ev": args.focus_energy_above_ev,
            "focus_repeats": args.focus_repeats if np.any(focus) else 1,
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
            if key not in {
                "radial", "torsion", "bend", "q16", "q8",
                "torsion_energy", "bend_energy", "q16_energy", "q8_energy"
            }
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
