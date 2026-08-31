#!/usr/bin/env python3
"""Learning-curve and oracle-active-sampling benchmark for the phenol DPEM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

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

from examples.namd import phenol_staged_mace_ftt_ttldr as stage
from pyqed.models.phenol import Phenol3D
from pyqed.units import au2angstrom


def reflected_design(count, bounds, seed):
    """Return an exactly sized nested design respecting ``phi -> -phi``."""

    count = int(count)
    equilibrium = np.asarray((Phenol3D.r_eq * au2angstrom, 0.0))
    if count == 1:
        return equilibrium[None, :]
    pairs = (count - 1) // 2
    positive = stage.sobol_coordinates(
        max(pairs, 1), (bounds[0], (0.0, bounds[1][1])), seed
    )[:pairs]
    values = [equilibrium]
    for point in positive:
        values.extend((point, point * np.asarray((1.0, -1.0))))
    if len(values) < count:
        fraction = (len(values) + 1) / (count + 1)
        values.append(np.asarray((bounds[0][0] + fraction * np.ptp(bounds[0]), 0.0)))
    return np.asarray(values[:count])


def active_extend(coordinates, target_count, bounds, fit, seed):
    """Add reflection-paired candidates using exact-error acquisition.

    This is an oracle benchmark: the real ab-initio workflow uses ensemble
    uncertainty instead, as implemented in ``phenol_abinitio_active.py``.
    """

    target_count = int(target_count)
    candidates = stage.sobol_coordinates(
        4096, (bounds[0], (0.0, bounds[1][1])), int(seed) + 9001
    )
    reduced = fit.neural_energy.predict(candidates)
    predicted = stage.parity_expand(candidates, reduced)
    reference = stage.reference_dpem(candidates)
    error = np.linalg.norm(predicted - reference, axis=(1, 2))
    levels = np.linalg.eigvalsh(reference)
    gap = np.maximum(np.min(np.diff(levels, axis=1), axis=1), 2.0e-3)
    r_eq = Phenol3D.r_eq * au2angstrom
    dynamical = np.exp(-0.5 * ((candidates[:, 0] - r_eq) / 0.65) ** 2)
    dynamical *= 0.35 + 0.65 * np.exp(-0.5 * (candidates[:, 1] / 0.9) ** 2)
    score = error * dynamical / gap
    scale = np.asarray((np.ptp(bounds[0]), np.ptp(bounds[1])))
    existing = coordinates / scale
    selected = []
    active_budget = max(2, (target_count - len(coordinates)) // 2)
    active_budget -= active_budget % 2
    for candidate in np.argsort(score)[::-1]:
        point = candidates[candidate]
        normalized = point / scale
        pool = existing if not selected else np.vstack((existing, np.asarray(selected) / scale))
        if np.min(cdist(normalized[None, :], pool)) < 0.035:
            continue
        selected.extend((point, point * np.asarray((1.0, -1.0))))
        if len(selected) >= active_budget:
            break
    output = np.vstack((coordinates, np.asarray(selected)))
    fallback = reflected_design(2 * target_count, bounds, seed + 1)
    for point in fallback:
        if len(output) >= target_count:
            break
        if np.min(cdist((point / scale)[None, :], output / scale)) >= 0.025:
            output = np.vstack((output, point))
    return output


def options(args, coordinates, output, seed):
    return SimpleNamespace(
        nr=args.nr, ntorsion=args.ntorsion, rmin=args.rmin, rmax=args.rmax,
        samples=len(coordinates), validation_samples=args.validation_samples,
        epochs=args.epochs, learning_rate=args.learning_rate,
        channels=args.channels, head_width=args.width,
        radial_basis=args.radial_basis, cutoff=args.cutoff, target="parity",
        tt_rank=args.tt_rank, tt_degree=args.tt_degree,
        overlap_rank=args.overlap_rank, potential_rank=args.potential_rank,
        operator_rank=args.operator_rank, state_rank=args.state_rank,
        bright_state=2, tmax_fs=args.tmax_fs, steps=args.steps,
        seed=int(seed), output_dir=Path(output),
        training_coordinates=np.asarray(coordinates),
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    counts = tuple(sorted(set(map(int, args.counts.split(",")))))
    bounds = ((args.rmin, args.rmax), (-np.pi, np.pi))
    designs = {
        "Nested Sobol": reflected_design(counts[-1], bounds, args.seed),
        "Active": reflected_design(counts[0], bounds, args.seed),
    }
    records = []
    previous_active_fit = None
    for method in designs:
        coordinates = designs[method]
        for index, count in enumerate(counts):
            if method == "Nested Sobol":
                current = coordinates[:count]
            elif index == 0:
                current = coordinates[:count]
            else:
                current = active_extend(
                    current, count, bounds, previous_active_fit,
                    args.seed + index,
                )
            case_dir = args.output / f"{method.lower().replace(' ', '_')}_{count}"
            result = stage.run(
                options(args, current, case_dir, args.seed)
            )
            metrics = result["metrics"]
            records.append({
                "method": method,
                "samples": int(len(current)),
                "matrix_rmse_mev": metrics["offgrid_validation"]["matrix_rmse_meV"],
                "relative_error": metrics["offgrid_validation"]["relative_frobenius_error"],
                "dynamics_fidelity": metrics["ttldr_final_fidelity_to_reference"],
                "maximum_norm_drift": metrics["maximum_norm_drift"],
            })
            if method == "Active":
                previous_active_fit = result["fit"]
        designs[method] = current
    figure, panels = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    colors = {"Nested Sobol": "#666666", "Active": "#0072b2"}
    for method in designs:
        subset = [record for record in records if record["method"] == method]
        x = [record["samples"] for record in subset]
        panels[0].loglog(x, [record["matrix_rmse_mev"] for record in subset], "o-", color=colors[method], label=method)
        panels[1].semilogx(x, [record["dynamics_fidelity"] for record in subset], "o-", color=colors[method])
        values = designs[method]
        panels[2].scatter(values[:, 0], np.rad2deg(values[:, 1]), s=13, alpha=0.75, color=colors[method], label=method)
    panels[0].set(xlabel="training geometries", ylabel="off-grid matrix RMSE (meV)")
    panels[1].set(xlabel="training geometries", ylabel=f"{args.tmax_fs:g} fs fidelity", ylim=(0.0, 1.02))
    panels[2].set(xlabel=r"$R_{OH}$ (angstrom)", ylabel=r"$\phi_{CCOH}$ (degree)")
    panels[0].legend(frameon=False)
    panels[2].legend(frameon=False)
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
    figure.savefig(args.output / "phenol_learning_curve_active.png", dpi=260)
    figure.savefig(args.output / "phenol_learning_curve_active.pdf")
    plt.close(figure)
    summary = {"counts": counts, "epochs": args.epochs, "records": records}
    (args.output / "phenol_learning_curve_active.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default="32,64,128,210")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--nr", type=int, default=29)
    parser.add_argument("--ntorsion", type=int, default=15)
    parser.add_argument("--rmin", type=float, default=0.82)
    parser.add_argument("--rmax", type=float, default=3.50)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--radial-basis", type=int, default=12)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--tt-rank", type=int, default=16)
    parser.add_argument("--tt-degree", type=int, default=16)
    parser.add_argument("--overlap-rank", type=int, default=12)
    parser.add_argument("--potential-rank", type=int, default=20)
    parser.add_argument("--operator-rank", type=int, default=48)
    parser.add_argument("--state-rank", type=int, default=32)
    parser.add_argument("--tmax-fs", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/phenol_learning_curve_active"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
