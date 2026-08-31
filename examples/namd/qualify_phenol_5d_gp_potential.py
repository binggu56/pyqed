#!/usr/bin/env python3
"""Qualify the phenol 5D adiabatic-S1 TT on dynamically relevant support."""

from __future__ import annotations

import argparse
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

from examples.namd.phenol_sa_casscf_5d_gp_control import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FIELD_CACHE,
    DEFAULT_INITIAL_STATE,
    DEFAULT_KEO_CACHE,
    DEFAULT_RADIAL_CORRECTION,
    ProjectedS1Oracle,
    _load_keo,
)
from examples.namd.phenol_sa_casscf_5d_quasibound import (
    load_interpolated_state,
    sample_mps_indices,
)
from pyqed.ml import CorrectedMatrixField, MACE, RadialMatrixCorrection
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.mps import MPS
from pyqed.mps.cross import tt_value


DEFAULT_DYNAMICS = Path(
    "dataset/phenol_5d_production/dynamics/gp_ngp_pilot_phase_only"
)


def _field_cores(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    return tuple(np.load(directory / name) for name in metadata["files"]["potential"])


def _saved_state(archive, mode):
    factors = []
    site = 0
    while f"final_factor_{site}_{mode}" in archive:
        factors.append(archive[f"final_factor_{site}_{mode}"])
        site += 1
    if not factors:
        raise ValueError(f"the dynamics archive has no {mode} final state")
    return MPS(factors)


def _statistics(error):
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "mean_ev": float(np.mean(error)),
        "rms_ev": float(np.sqrt(np.mean(error**2))),
        "p95_absolute_ev": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_ev": float(np.max(absolute)),
    }


def run(args):
    axes, _keo, _metadata = _load_keo(args.keo_cache)
    shape = tuple(map(len, axes))
    cores = _field_cores(args.field_cache)
    fit = MACE.load(
        args.checkpoint, PhenolReactiveChart().geometry, device="cpu", distill=False
    )
    energy = CorrectedMatrixField(
        fit.neural_energy, RadialMatrixCorrection.load(args.radial_correction)
    )
    oracle = ProjectedS1Oracle(
        axes, energy, None, prediction_batch_size=args.prediction_batch_size
    )
    dynamics = np.load(args.dynamics / "phenol_5d_gp_ngp.npz")
    states = {
        "initial": load_interpolated_state(args.initial_state, axes, sites=None),
        "GP at 1 fs": _saved_state(dynamics, "gp"),
        "NGP at 1 fs": _saved_state(dynamics, "ngp"),
    }
    rng = np.random.default_rng(args.seed)
    designs = {
        name: sample_mps_indices(state, args.samples, args.seed + 17 * offset)[:, :5]
        for offset, (name, state) in enumerate(states.items())
    }
    designs["uniform guard"] = np.column_stack(
        [rng.integers(0, size, args.guard_samples) for size in shape]
    )
    all_indices = np.vstack(tuple(designs.values()))
    exact = oracle.potential(all_indices)
    approximate = np.asarray([tt_value(cores, row) for row in all_indices]).real
    errors = 27.211386245988 * (approximate - exact)
    result = {}
    start = 0
    split_errors = {}
    for name, indices in designs.items():
        stop = start + len(indices)
        split_errors[name] = errors[start:stop]
        result[name] = _statistics(split_errors[name])
        start = stop

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "potential_qualification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    figure, panels = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    colors = ("#333333", "#0072B2", "#D55E00", "#6A51A3")
    for (name, values), color in zip(split_errors.items(), colors):
        panels[0].hist(values, bins=50, density=True, histtype="step", linewidth=1.5, color=color, label=name)
    for (name, indices), color in zip(designs.items(), colors):
        values = split_errors[name]
        panels[1].scatter(axes[0][indices[:, 0]], values, s=5, alpha=0.22, color=color, label=name)
    panels[0].set(xlabel=r"$E_{TT}-E_{MACE}$ (eV)", ylabel="density", title="Potential-error distribution")
    panels[1].set(xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)", ylabel=r"$E_{TT}-E_{MACE}$ (eV)", title="Radial localization of error")
    for label, panel in zip("ab", panels):
        panel.axvline(0.0, color="0.65", linewidth=0.8)
        panel.grid(alpha=0.18)
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
    panels[0].legend(frameon=False, fontsize=8)
    panels[1].legend(frameon=False, fontsize=8)
    figure.savefig(args.output / "potential_qualification.png", dpi=350)
    figure.savefig(args.output / "potential_qualification.pdf")
    plt.close(figure)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--radial-correction", type=Path, default=DEFAULT_RADIAL_CORRECTION)
    parser.add_argument("--keo-cache", type=Path, default=DEFAULT_KEO_CACHE)
    parser.add_argument("--field-cache", type=Path, default=DEFAULT_FIELD_CACHE)
    parser.add_argument("--initial-state", type=Path, default=DEFAULT_INITIAL_STATE)
    parser.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--guard-samples", type=int, default=2048)
    parser.add_argument("--prediction-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=811)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
