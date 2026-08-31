#!/usr/bin/env python3
"""Train and validate a direct MACE fit of the full planar phenol O-H scan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pyqed.units import au2ev
from pyqed.ml import MACE
from pyqed.models.phenol_coordinates import PHENOL_SPECIES, PhenolReactiveChart


HARTREE_TO_EV = au2ev
ENERGY_HOLDOUT = np.asarray((5, 9, 13, 16, 18))
LINK_HOLDOUT = np.asarray((4, 8, 12, 16, 18))


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


def _geometry(chart, coordinate):
    full = np.array(chart.equilibrium, copy=True)
    full[0] = float(np.asarray(coordinate)[0])
    return chart.geometry(full)


def _new_mace(radii, chart, args):
    return MACE(
        (radii,),
        PHENOL_SPECIES,
        lambda coordinate: _geometry(chart, coordinate),
        3,
        chart_features=True,
        geometry_units="angstrom",
        channels=args.channels,
        max_ell=args.max_ell,
        interactions=args.interactions,
        correlation=args.correlation,
        radial_basis=args.radial_basis,
        radial_mlp=(args.width, args.width),
        cutoff=args.cutoff,
    )


def _train(
    radii,
    hamiltonian,
    link_midpoints,
    links,
    chart,
    args,
    *,
    energy_indices,
    link_indices,
    epochs,
    seed,
):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = _new_mace(radii, chart, args)
    started = time.perf_counter()
    model.fit(
        (radii[energy_indices, None], hamiltonian[energy_indices]),
        ((link_midpoints[link_indices, None], links[link_indices]),),
        hidden=(args.width, args.width),
        epochs=int(epochs),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        energy_weight=1.0,
        link_weight=args.link_weight,
        seed=int(seed),
        distill=False,
    )
    return model, time.perf_counter() - started


def _energy_metrics(prediction, reference):
    error = np.asarray(prediction) - np.asarray(reference)
    spectral = (
        np.linalg.eigvalsh(prediction) - np.linalg.eigvalsh(reference)
    ) * HARTREE_TO_EV
    return {
        "matrix_element_rmse_mev": float(
            np.sqrt(np.mean(np.abs(error) ** 2)) * HARTREE_TO_EV * 1000.0
        ),
        "matrix_frobenius_maximum_ev": float(
            np.max(np.linalg.norm(error, axis=(-2, -1))) * HARTREE_TO_EV
        ),
        "spectral_rmse_mev": float(
            np.sqrt(np.mean(spectral**2)) * 1000.0
        ),
        "spectral_maximum_ev": float(np.max(np.abs(spectral))),
    }


def _link_metrics(prediction, reference):
    error = np.asarray(prediction) - np.asarray(reference)
    return {
        "matrix_element_rmse": float(np.sqrt(np.mean(np.abs(error) ** 2))),
        "frobenius_maximum": float(
            np.max(np.linalg.norm(error, axis=(-2, -1)))
        ),
        "minimum_predicted_singular_value": float(
            np.min(np.linalg.svd(prediction, compute_uv=False))
        ),
        "maximum_predicted_singular_value": float(
            np.max(np.linalg.svd(prediction, compute_uv=False))
        ),
    }


def _plot(
    output,
    radii,
    exact_energy,
    dense_radii,
    dense_energy,
    final_energy_error,
    validation_energy_error,
    link_midpoints,
    exact_link_minimum,
    dense_link_midpoints,
    dense_link_minimum,
    validation_link_error,
    validation_history,
    final_history,
    bridge,
):
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, panels = plt.subplots(
        2, 2, figsize=(11.3, 7.0), constrained_layout=True
    )
    for state, color in enumerate(colors):
        panels[0, 0].plot(
            dense_radii,
            dense_energy[:, state],
            color=color,
            lw=1.5,
            label=f"MACE state {state}",
        )
        panels[0, 0].plot(
            radii,
            exact_energy[:, state],
            "o",
            color=color,
            ms=3.7,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        panels[0, 1].semilogy(
            radii,
            np.maximum(np.abs(final_energy_error[:, state]), 1.0e-5),
            "o-",
            color=color,
            lw=1.1,
            ms=3.0,
            label=f"final state {state}",
        )
    panels[0, 1].semilogy(
        radii[ENERGY_HOLDOUT],
        np.maximum(validation_energy_error, 1.0e-5),
        "kx",
        ms=5.0,
        mew=1.0,
        label="held-out spectral max",
    )
    panels[1, 0].semilogy(
        np.arange(1, len(validation_history) + 1),
        validation_history,
        color="#D55E00",
        lw=1.1,
        label="held-out split training",
    )
    panels[1, 0].semilogy(
        np.arange(1, len(final_history) + 1),
        final_history,
        color="#0072B2",
        lw=1.1,
        label="all-sample training",
    )
    panels[1, 1].plot(
        dense_link_midpoints,
        dense_link_minimum,
        color="#0072B2",
        lw=1.5,
        label="final MACE",
    )
    panels[1, 1].plot(
        link_midpoints,
        exact_link_minimum,
        "ko",
        ms=3.4,
        label="ab initio link",
    )
    panels[1, 1].plot(
        link_midpoints[LINK_HOLDOUT],
        validation_link_error,
        "x",
        color="#D55E00",
        ms=5.0,
        mew=1.0,
        label=r"held-out $\|\Delta S\|_F$",
    )

    for panel in (panels[0, 0], panels[0, 1], panels[1, 1]):
        panel.axvspan(*bridge, color="0.75", alpha=0.28, lw=0.0)
    for panel in panels.flat:
        panel.grid(alpha=0.18, which="both")
    panels[0, 0].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel=r"$E-E_0(R_{eq})$ (eV)",
        title="a  Whole-range MACE potential",
    )
    panels[0, 0].legend(fontsize=7.8, ncol=2)
    panels[0, 1].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel="spectral error (eV)",
        ylim=(1.0e-5, 1.0),
        title="b  Training and held-out errors",
    )
    panels[0, 1].legend(fontsize=7.4, ncol=2)
    panels[1, 0].set(
        xlabel="epoch",
        ylabel="normalized MACE loss",
        title="c  Neural training convergence",
    )
    panels[1, 0].legend(fontsize=8.0)
    panels[1, 1].set(
        xlabel=r"link midpoint $R_{OH}$ ($\AA$)",
        ylabel=r"minimum $\sigma(S^P)$ or held-out error",
        ylim=(0.0, 1.08),
        title="d  MACE retains the low-confidence bridge",
    )
    panels[1, 1].legend(fontsize=7.8, loc="lower right")
    png = output / "phenol_sa6_full_r_mace_diagnostics.png"
    pdf = output / "phenol_sa6_full_r_mace_diagnostics.pdf"
    figure.savefig(png, dpi=320)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_p_gauge_20260820/"
            "phenol_sa6_tracked3_p_gauge.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_full_r_mace_20260820"),
    )
    parser.add_argument("--validation-epochs", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument(
        "--reuse-final",
        action="store_true",
        help="load the existing final checkpoint and only repeat held-out validation",
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-8)
    parser.add_argument("--link-weight", type=float, default=0.25)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-ell", type=int, default=1)
    parser.add_argument("--interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--radial-basis", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--dense-points", type=int, default=3001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", message="Environment variable TORCH_FORCE")
    warnings.filterwarnings("ignore", message="The TorchScript type system")

    with np.load(args.data, allow_pickle=False) as archive:
        radii = np.asarray(archive["radii"], dtype=float)
        anchor = int(archive["anchor"])
        hamiltonian = np.asarray(archive["p_hamiltonian"])
        links = np.asarray(archive["p_links"])
        tracked_singular = np.asarray(archive["tracked_singular_values"])
    shift = float(np.min(np.linalg.eigvalsh(hamiltonian[anchor])))
    hamiltonian = hamiltonian - shift * np.eye(3)
    link_midpoints = 0.5 * (radii[:-1] + radii[1:])
    edge_minimum = tracked_singular[:, -1]
    bridge_index = int(np.argmin(edge_minimum))
    bridge = (float(radii[bridge_index]), float(radii[bridge_index + 1]))
    chart = PhenolReactiveChart()

    energy_training = np.setdiff1d(np.arange(len(radii)), ENERGY_HOLDOUT)
    link_training = np.setdiff1d(np.arange(len(link_midpoints)), LINK_HOLDOUT)
    validation_model, validation_seconds = _train(
        radii,
        hamiltonian,
        link_midpoints,
        links,
        chart,
        args,
        energy_indices=energy_training,
        link_indices=link_training,
        epochs=args.validation_epochs,
        seed=args.seed,
    )
    validation_energy = validation_model.neural_energy.predict(
        radii[ENERGY_HOLDOUT, None]
    )
    validation_links = validation_model.neural_links[0].predict(
        link_midpoints[LINK_HOLDOUT, None]
    )
    validation_energy_metrics = _energy_metrics(
        validation_energy, hamiltonian[ENERGY_HOLDOUT]
    )
    validation_link_metrics = _link_metrics(
        validation_links, links[LINK_HOLDOUT]
    )

    checkpoint = args.output / "phenol_sa6_full_r_mace.pt"
    if args.reuse_final:
        if not checkpoint.exists():
            raise FileNotFoundError(f"final MACE checkpoint not found: {checkpoint}")
        final_model = MACE.load(
            checkpoint,
            lambda coordinate: _geometry(chart, coordinate),
            device="cpu",
            distill=False,
        )
        final_seconds = 0.0
    else:
        final_model, final_seconds = _train(
            radii,
            hamiltonian,
            link_midpoints,
            links,
            chart,
            args,
            energy_indices=np.arange(len(radii)),
            link_indices=np.arange(len(link_midpoints)),
            epochs=args.epochs,
            seed=args.seed,
        )
    final_energy = final_model.neural_energy.predict(radii[:, None])
    final_links = final_model.neural_links[0].predict(link_midpoints[:, None])
    final_energy_metrics = _energy_metrics(final_energy, hamiltonian)
    final_link_metrics = _link_metrics(final_links, links)

    dense_radii = np.linspace(radii[0], radii[-1], args.dense_points)
    dense_link_midpoints = np.linspace(
        link_midpoints[0], link_midpoints[-1], args.dense_points
    )
    dense_hamiltonian = final_model.neural_energy.predict(dense_radii[:, None])
    dense_links = final_model.neural_links[0].predict(
        dense_link_midpoints[:, None]
    )
    dense_energies = np.linalg.eigvalsh(dense_hamiltonian) * HARTREE_TO_EV
    exact_energies = np.linalg.eigvalsh(hamiltonian) * HARTREE_TO_EV
    dense_singular = np.linalg.svd(dense_links, compute_uv=False)
    final_spectral_error = (
        np.linalg.eigvalsh(final_energy) - np.linalg.eigvalsh(hamiltonian)
    ) * HARTREE_TO_EV
    validation_spectral_error = np.max(
        np.abs(
            (
                np.linalg.eigvalsh(validation_energy)
                - np.linalg.eigvalsh(hamiltonian[ENERGY_HOLDOUT])
            )
            * HARTREE_TO_EV
        ),
        axis=1,
    )
    validation_link_error = np.linalg.norm(
        validation_links - links[LINK_HOLDOUT], axis=(-2, -1)
    )
    dense_hermiticity = float(
        np.max(
            np.linalg.norm(
                dense_hamiltonian
                - dense_hamiltonian.conj().swapaxes(-1, -2),
                axis=(-2, -1),
            )
        )
    )

    final_model.save(checkpoint)
    restored = MACE.load(
        checkpoint,
        lambda coordinate: _geometry(chart, coordinate),
        device="cpu",
        distill=False,
    )
    probe = np.linspace(radii[0], radii[-1], 37)[:, None]
    roundtrip_energy_error = float(
        np.max(
            np.abs(
                restored.neural_energy.predict(probe)
                - final_model.neural_energy.predict(probe)
            )
        )
    )
    roundtrip_link_error = float(
        np.max(
            np.abs(
                restored.neural_links[0].predict(probe)
                - final_model.neural_links[0].predict(probe)
            )
        )
    )

    bad_link_singular = np.linalg.svd(
        final_links[bridge_index], compute_uv=False
    )
    png, pdf = _plot(
        args.output,
        radii,
        exact_energies,
        dense_radii,
        dense_energies,
        final_spectral_error,
        validation_spectral_error,
        link_midpoints,
        edge_minimum,
        dense_link_midpoints,
        dense_singular[:, -1],
        validation_link_error,
        np.asarray(validation_model.history),
        np.asarray(final_model.history),
        bridge,
    )
    predictions = args.output / "phenol_sa6_full_r_mace_predictions.npz"
    np.savez_compressed(
        predictions,
        radii=radii,
        exact_hamiltonian=hamiltonian,
        final_hamiltonian=final_energy,
        link_midpoints=link_midpoints,
        exact_links=links,
        final_links=final_links,
        dense_radii=dense_radii,
        dense_hamiltonian=dense_hamiltonian,
        dense_link_midpoints=dense_link_midpoints,
        dense_links=dense_links,
        energy_holdout=ENERGY_HOLDOUT,
        validation_hamiltonian=validation_energy,
        link_holdout=LINK_HOLDOUT,
        validation_links=validation_links,
    )
    summary = {
        "fit_completed": bool(
            np.all(np.isfinite(dense_hamiltonian))
            and np.all(np.isfinite(dense_links))
            and dense_hermiticity <= 1.0e-12
            and roundtrip_energy_error <= 1.0e-12
            and roundtrip_link_error <= 1.0e-12
        ),
        "production_ready": bool(
            validation_energy_metrics["spectral_maximum_ev"] <= 0.05
            and validation_link_metrics["frobenius_maximum"] <= 0.05
        ),
        "validation_gate": {
            "held_out_spectral_maximum_target_ev": 0.05,
            "held_out_spectral_maximum_passed": bool(
                validation_energy_metrics["spectral_maximum_ev"] <= 0.05
            ),
            "held_out_link_frobenius_target": 0.05,
            "held_out_link_frobenius_passed": bool(
                validation_link_metrics["frobenius_maximum"] <= 0.05
            ),
        },
        "backend": "direct-mace-ldr-no-functional-tt-distillation",
        "scope": "planar R_OH = 0.90 to 3.00 angstrom",
        "samples": {"energies": len(radii), "links": len(link_midpoints)},
        "architecture": {
            "channels": args.channels,
            "max_ell": args.max_ell,
            "interactions": args.interactions,
            "correlation": args.correlation,
            "radial_basis": args.radial_basis,
            "head_width": args.width,
            "cutoff_angstrom": args.cutoff,
            "chart_features": True,
        },
        "training": {
            "validation_epochs": args.validation_epochs,
            "final_epochs": args.epochs,
            "validation_seconds": validation_seconds,
            "final_seconds": final_seconds,
            "final_checkpoint_reused": args.reuse_final,
            "validation_initial_loss": float(validation_model.history[0]),
            "validation_final_loss": float(validation_model.history[-1]),
            "final_initial_loss": float(final_model.history[0]),
            "final_loss": float(final_model.history[-1]),
        },
        "held_out": {
            "energy_radii_angstrom": radii[ENERGY_HOLDOUT],
            "link_midpoints_angstrom": link_midpoints[LINK_HOLDOUT],
            "energy": validation_energy_metrics,
            "links": validation_link_metrics,
        },
        "all_sample_final": {
            "energy": final_energy_metrics,
            "links": final_link_metrics,
        },
        "dense": {
            "energy_hermiticity_defect": dense_hermiticity,
            "link_singular_range": [
                float(np.min(dense_singular)),
                float(np.max(dense_singular)),
            ],
        },
        "low_confidence_bridge": {
            "left_radius_angstrom": bridge[0],
            "right_radius_angstrom": bridge[1],
            "sampled_minimum_singular_value": float(edge_minimum[bridge_index]),
            "mace_singular_values": bad_link_singular,
        },
        "roundtrip_maximum_energy_error_hartree": roundtrip_energy_error,
        "roundtrip_maximum_link_error": roundtrip_link_error,
        "checkpoint": str(checkpoint),
        "predictions": str(predictions),
        "figures": {"png": str(png), "pdf": str(pdf)},
        "source": str(args.data),
        "new_quantum_chemistry_calculations": 0,
        "functional_tt_distillation": None,
        "recommended_next_radii_angstrom": [
            1.625,
            1.775,
            2.125,
            2.35,
            2.65,
            2.80,
        ],
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
