#!/usr/bin/env python3
"""Fit the reflection-even scalar P-gauge parent potential for phenol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pyqed.ml import ReflectionScalarMLP


HARTREE_TO_EV = 27.211386245988
PARITIES = np.asarray((1.0, -1.0, 1.0, -1.0, 1.0))


def load_training_data(primary, additional=(), tolerance=1.0e-8):
    """Merge exact scalar-fit records, replacing duplicate coordinates."""

    with np.load(primary, allow_pickle=False) as archive:
        coordinates = np.asarray(archive["coordinates"], dtype=float)
        hamiltonian = np.asarray(archive["p_hamiltonian"], dtype=complex)
        holdout = np.asarray(archive["energy_holdout"], dtype=bool)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 5
        or hamiltonian.shape != (len(coordinates), 3, 3)
        or holdout.shape != (len(coordinates),)
    ):
        raise ValueError("primary scalar-fit data have inconsistent shapes")
    statistics = []
    for path in additional:
        with np.load(path, allow_pickle=False) as archive:
            extra_coordinates = np.asarray(archive["coordinates"], dtype=float)
            extra_hamiltonian = np.asarray(archive["p_hamiltonian"], dtype=complex)
        if (
            extra_coordinates.ndim != 2
            or extra_coordinates.shape[1] != 5
            or extra_hamiltonian.shape != (len(extra_coordinates), 3, 3)
        ):
            raise ValueError(f"additional scalar-fit data {path} have inconsistent shapes")
        added = 0
        replaced = 0
        for coordinate, value in zip(extra_coordinates, extra_hamiltonian):
            difference = coordinates - coordinate
            difference[:, 1] = (difference[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
            matches = np.flatnonzero(np.max(np.abs(difference), axis=1) <= tolerance)
            if matches.size:
                index = int(matches[0])
                hamiltonian[index] = value
                holdout[index] = False
                replaced += 1
            else:
                coordinates = np.vstack((coordinates, coordinate))
                hamiltonian = np.concatenate((hamiltonian, value[None]), axis=0)
                holdout = np.append(holdout, False)
                added += 1
        statistics.append(
            {"path": str(path), "added": added, "replaced": replaced}
        )
    return coordinates, hamiltonian, holdout, statistics


class ParentPotential(torch.nn.Module):
    def __init__(
        self,
        lower,
        upper,
        width=128,
        depth=4,
        periodic_axes=(),
        periodic_harmonics=1,
    ):
        super().__init__()
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.float32))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.float32))
        self.periodic_axes = tuple(sorted({int(axis) for axis in periodic_axes}))
        self.periodic_harmonics = int(periodic_harmonics)
        layers = []
        size = 5 + len(self.periodic_axes) * (2 * self.periodic_harmonics - 1)
        for _ in range(int(depth)):
            layers.extend((torch.nn.Linear(size, width), torch.nn.SiLU()))
            size = width
        layers.append(torch.nn.Linear(size, 1))
        self.network = torch.nn.Sequential(*layers)

    def features(self, coordinates):
        blocks = []
        periodic = set(self.periodic_axes)
        for axis in range(coordinates.shape[1]):
            if axis in periodic:
                angle = (
                    2.0
                    * torch.pi
                    * (coordinates[:, axis] - 0.5 * (self.lower[axis] + self.upper[axis]))
                    / (self.upper[axis] - self.lower[axis])
                )
                for harmonic in range(1, self.periodic_harmonics + 1):
                    blocks.extend(
                        (
                            torch.sin(harmonic * angle),
                            torch.cos(harmonic * angle),
                        )
                    )
            else:
                blocks.append(
                    2.0
                    * (coordinates[:, axis] - self.lower[axis])
                    / (self.upper[axis] - self.lower[axis])
                    - 1.0
                )
        return torch.column_stack(blocks)

    def forward(self, coordinates):
        reflected = coordinates * coordinates.new_tensor(PARITIES)
        return 0.5 * (
            self.network(self.features(coordinates))[:, 0]
            + self.network(self.features(reflected))[:, 0]
        )


def _errors(predicted, target):
    error = np.asarray(predicted) - np.asarray(target)
    return {
        "count": int(len(error)),
        "rms_ev": float(np.sqrt(np.mean(error**2))),
        "maximum_ev": float(np.max(np.abs(error))),
        "mae_ev": float(np.mean(np.abs(error))),
    }


def _train(
    model,
    coordinates,
    target,
    indices,
    *,
    epochs,
    learning_rate,
    weight_decay,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(epochs), eta_min=0.01 * float(learning_rate)
    )
    best = None
    history = []
    for epoch in range(int(epochs)):
        optimizer.zero_grad()
        predicted = model(coordinates)
        loss = torch.mean((predicted[indices] - target[indices]) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        value = float(loss.detach())
        history.append(value)
        if best is None or value < best[0]:
            best = (value, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    return history


def _export(model, lower, upper, shift):
    linear = [
        layer for layer in model.network
        if isinstance(layer, torch.nn.Linear)
    ]
    return ReflectionScalarMLP(
        lower,
        upper,
        PARITIES,
        [layer.weight.detach().cpu().numpy() for layer in linear],
        [layer.bias.detach().cpu().numpy() for layer in linear],
        output_shift=shift,
        output_scale=1.0 / HARTREE_TO_EV,
        periodic_axes=model.periodic_axes,
        periodic_harmonics=model.periodic_harmonics,
    )


def _plot(
    output,
    coordinates,
    target,
    validation_prediction,
    production_prediction,
    holdout,
    history,
):
    figure, panels = plt.subplots(2, 2, figsize=(9.8, 7.4))
    panels = panels.reshape(-1)
    panels[0].plot(history[0], label="held-out fit")
    panels[0].plot(
        np.arange(len(history[1])) + len(history[0]),
        history[1],
        label="all-data polish",
    )
    panels[0].set(yscale="log", xlabel="epoch", ylabel="MSE (eV$^2$)", title="Training")
    panels[0].legend(frameon=False)
    panels[1].scatter(target[~holdout], production_prediction[~holdout], s=10, alpha=0.55)
    panels[1].scatter(
        target[holdout], validation_prediction[holdout], s=20, marker="x", label="holdout"
    )
    limits = (float(np.min(target)), float(np.max(target)))
    panels[1].plot(limits, limits, "k--", lw=1)
    panels[1].set(xlabel="ab initio $H_{11}$ (eV)", ylabel="fit (eV)", title="Scalar parent potential")
    panels[1].legend(frameon=False)
    error = production_prediction - target
    panels[2].hist(error, bins=32, color="C0", alpha=0.8)
    panels[2].set(xlabel="production error (eV)", ylabel="count", title="All-data residual")
    panels[3].scatter(
        coordinates[~holdout, 1], error[~holdout], s=10, alpha=0.55
    )
    panels[3].scatter(
        coordinates[holdout, 1], error[holdout], s=20, marker="x", label="holdout"
    )
    panels[3].axhline(0.0, color="0.2", lw=0.8)
    panels[3].set(
        xlabel=r"torsion $\phi$ (rad)",
        ylabel="production error (eV)",
        title="Periodic-domain residual",
    )
    panels[3].legend(frameon=False)
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    png = output / "phenol_sa6_5d_scalar_parent.png"
    figure.savefig(png, dpi=300)
    figure.savefig(output / "phenol_sa6_5d_scalar_parent.pdf")
    plt.close(figure)
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--additional-data",
        type=Path,
        action="append",
        default=[],
        help="exact P-gauge points appended to training; duplicate coordinates replace base values",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--validation-epochs", type=int, default=10000)
    parser.add_argument("--production-epochs", type=int, default=5000)
    parser.add_argument("--validation-weight-decay", type=float, default=1.0e-7)
    parser.add_argument("--production-weight-decay", type=float, default=1.0e-7)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument(
        "--periodic-torsion",
        action="store_true",
        help="encode torsion with a periodic sine/cosine feature pair",
    )
    parser.add_argument("--periodic-harmonics", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    coordinates, p_hamiltonian, holdout, additions = load_training_data(
        args.data, args.additional_data
    )
    coordinates = np.asarray(coordinates, dtype=np.float32)
    target_hartree = np.asarray(
        p_hamiltonian[:, args.state, args.state].real, dtype=np.float32
    )
    training = np.flatnonzero(~holdout)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    periodic_axes = (1,) if args.periodic_torsion else ()
    shift = float(np.mean(target_hartree[training]))
    target_ev = (target_hartree - shift) * HARTREE_TO_EV
    tensor_coordinates = torch.as_tensor(coordinates)
    tensor_target = torch.as_tensor(target_ev)
    tensor_training = torch.as_tensor(training)

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    model = ParentPotential(
        lower,
        upper,
        args.width,
        args.depth,
        periodic_axes=periodic_axes,
        periodic_harmonics=(args.periodic_harmonics if periodic_axes else 1),
    )
    validation_history = _train(
        model,
        tensor_coordinates,
        tensor_target,
        tensor_training,
        epochs=args.validation_epochs,
        learning_rate=1.5e-3,
        weight_decay=args.validation_weight_decay,
    )
    with torch.no_grad():
        validation_prediction = model(tensor_coordinates).cpu().numpy()
    validation = _errors(validation_prediction[holdout], target_ev[holdout])

    production_history = _train(
        model,
        tensor_coordinates,
        tensor_target,
        torch.arange(len(coordinates)),
        epochs=args.production_epochs,
        learning_rate=5.0e-4,
        weight_decay=args.production_weight_decay,
    )
    field = _export(model, lower, upper, shift)
    production_hartree = field.predict(coordinates)
    production_prediction = (production_hartree - shift) * HARTREE_TO_EV
    production = _errors(production_prediction, target_ev)
    rng = np.random.default_rng(args.seed + 1)
    probe = rng.uniform(lower, upper, size=(256, 5))
    reflection_defect = float(
        np.max(np.abs(field.predict(probe) - field.predict(probe * PARITIES)))
    )
    periodic_defect = 0.0
    for axis in periodic_axes:
        shifted_probe = probe.copy()
        shifted_probe[:, axis] += upper[axis] - lower[axis]
        periodic_defect = max(
            periodic_defect,
            float(np.max(np.abs(field.predict(probe) - field.predict(shifted_probe)))),
        )
    checkpoint = field.save(args.output / "phenol_sa6_5d_scalar_parent.npz")
    figure = _plot(
        args.output,
        coordinates,
        target_ev,
        validation_prediction,
        production_prediction,
        holdout,
        (validation_history, production_history),
    )
    gates = {
        "heldout_rms_below_0p05_ev": validation["rms_ev"] <= 0.05,
        "heldout_maximum_below_0p15_ev": validation["maximum_ev"] <= 0.15,
        "all_data_rms_below_0p03_ev": production["rms_ev"] <= 0.03,
        "all_data_maximum_below_0p15_ev": production["maximum_ev"] <= 0.15,
        "exact_reflection_symmetry": reflection_defect == 0.0,
        "periodicity_to_1e_12_hartree": periodic_defect <= 1.0e-12,
    }
    summary = {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "state": int(args.state),
        "model": {
            "representation": "reflection-symmetrized scalar coordinate MLP",
            "width": int(args.width),
            "depth": int(args.depth),
            "validation_epochs": int(args.validation_epochs),
            "production_epochs": int(args.production_epochs),
            "validation_weight_decay": float(args.validation_weight_decay),
            "production_weight_decay": float(args.production_weight_decay),
            "coordinate_bounds": np.column_stack((lower, upper)).tolist(),
            "periodic_axes": list(periodic_axes),
            "periodic_harmonics": int(model.periodic_harmonics),
        },
        "validation": validation,
        "production": production,
        "maximum_reflection_defect_hartree": reflection_defect,
        "maximum_periodic_defect_hartree": periodic_defect,
        "worst_validation_points": [
            {
                "coordinates": coordinates[index].astype(float).tolist(),
                "target_ev": float(target_ev[index]),
                "prediction_ev": float(validation_prediction[index]),
                "absolute_error_ev": float(
                    abs(validation_prediction[index] - target_ev[index])
                ),
            }
            for index in np.flatnonzero(holdout)[
                np.argsort(
                    -np.abs(
                        validation_prediction[holdout] - target_ev[holdout]
                    )
                )[:5]
            ]
        ],
        "data": str(args.data),
        "additional_data": additions,
        "checkpoint": str(checkpoint),
        "figure": str(figure),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise RuntimeError("scalar parent-potential qualification failed")


if __name__ == "__main__":
    main()
