#!/usr/bin/env python3
"""Validate the AD-derived five-dimensional phenol Podolsky KEO."""

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

from examples.namd.phenol_5d_mace_ftt_ttldr import build_dvrs
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.namd.phenol import build_phenol_5d_keo_mpo, phenol_metric_evaluators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-points", type=int, default=3)
    parser.add_argument("--cross-rank", type=int, default=5)
    parser.add_argument("--cross-sweeps", type=int, default=4)
    parser.add_argument("--cross-rtol", type=float, default=2.0e-7)
    parser.add_argument("--cross-validation", type=int, default=48)
    parser.add_argument("--mpo-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_5d_numerical_g_validation"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    chart = PhenolReactiveChart()
    axes, dvrs = build_dvrs(args.grid_points, chart)
    started = time.perf_counter()
    components, cross = build_phenol_5d_keo_mpo(
        dvrs,
        chart,
        cross_max_rank=args.cross_rank,
        cross_sweeps=args.cross_sweeps,
        cross_rtol=args.cross_rtol,
        cross_validation=args.cross_validation,
        mpo_max_rank=args.mpo_rank,
        seed=args.seed,
        split=True,
        enforce_reflection=True,
        return_info=True,
    )
    component_dense = [
        _mpo_to_dense_operator(operator) for _active, operator in components
    ]
    dense = sum(component_dense, np.zeros_like(component_dense[0]))
    elapsed = time.perf_counter() - started

    mesh = np.meshgrid(*(dvr.x for dvr in dvrs), indexing="ij")
    atomic_points = np.stack([value.reshape(-1) for value in mesh], axis=1)
    _point, batch = phenol_metric_evaluators(chart)
    metrics, pseudopotential = batch(atomic_points)
    metric_eigenvalues = np.linalg.eigvalsh(metrics)
    metric_condition = np.linalg.cond(metrics)
    shape = tuple(dvr.npts for dvr in dvrs)
    reflected = np.asarray(
        [
            np.ravel_multi_index(
                (
                    index[0],
                    shape[1] - 1 - index[1],
                    index[2],
                    shape[3] - 1 - index[3],
                    index[4],
                ),
                shape,
            )
            for index in np.ndindex(shape)
        ]
    )
    norm = max(float(np.linalg.norm(dense)), np.finfo(float).tiny)
    eigenvalues = np.linalg.eigvalsh(dense)
    summary = {
        "grid_shape": list(shape),
        "seconds": elapsed,
        "kinetic_model": "AD G-matrix J=0 Podolsky KEO",
        "components": len(components),
        "component_active_axes": [list(active) for active, _operator in components],
        "component_ranks": [
            list(map(int, operator.bond_orders())) for _active, operator in components
        ],
        "metric_cross_validation_rms": float(
            cross["cross"]["validation_rms_error"]
        ),
        "metric_cross_validation_max": float(cross["cross"]["validation_error"]),
        "metric_point_samples": int(cross["point_samples"]),
        "minimum_metric_eigenvalue": float(metric_eigenvalues.min()),
        "maximum_metric_condition": float(metric_condition.max()),
        "pseudopotential_range": [
            float(pseudopotential.min()),
            float(pseudopotential.max()),
        ],
        "hermiticity_relative_defect": float(
            np.linalg.norm(dense - dense.conj().T) / norm
        ),
        "reflection_covariance_relative_defect": float(
            np.linalg.norm(dense[np.ix_(reflected, reflected)] - dense) / norm
        ),
        "minimum_keo_eigenvalue": float(eigenvalues[0]),
    }

    public_points = np.stack(
        [
            value.reshape(-1)
            for value in np.meshgrid(*axes, indexing="ij")
        ],
        axis=1,
    )
    labels = tuple(cross["field_labels"])
    component_norms = np.asarray([np.linalg.norm(value) for value in component_dense])
    figure, panels = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    scatter = panels[0, 0].scatter(
        public_points[:, 0],
        metric_eigenvalues[:, 0],
        c=np.log10(metric_condition),
        cmap="viridis",
        s=20,
    )
    figure.colorbar(scatter, ax=panels[0, 0], label=r"$\log_{10}\kappa(g)$")
    panels[0, 0].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel=r"minimum eigenvalue of $g$",
        title="Positive vibrational metric",
    )
    panels[0, 1].scatter(
        public_points[:, 0],
        pseudopotential,
        c=public_points[:, 1],
        cmap="coolwarm",
        s=20,
    )
    panels[0, 1].set(
        xlabel=r"$R_{OH}$ ($\AA$)",
        ylabel=r"$V_{\mathrm{ps}}$ (hartree)",
        title="Podolsky pseudopotential",
    )
    panels[1, 0].bar(np.arange(len(labels)), component_norms, color="#0072B2")
    panels[1, 0].set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        yscale="log",
        ylabel="component Frobenius norm",
        title="Metric KEO components",
    )
    panels[1, 0].tick_params(axis="x", rotation=65, labelsize=7)
    count = min(30, len(eigenvalues))
    panels[1, 1].plot(np.arange(count), eigenvalues[:count], "o-", color="#009E73")
    panels[1, 1].set(
        xlabel="eigenvalue index",
        ylabel="KEO eigenvalue (hartree)",
        title="Lowest numerical-$g$ kinetic levels",
    )
    for label, panel in zip("abcd", panels.flat):
        panel.text(
            0.02,
            0.96,
            label,
            transform=panel.transAxes,
            va="top",
            fontweight="bold",
        )
        panel.grid(alpha=0.18)
    figure_path = args.output / "phenol_5d_numerical_g_validation.png"
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    summary["figure"] = str(figure_path)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
