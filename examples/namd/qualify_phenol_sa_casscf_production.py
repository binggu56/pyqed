#!/usr/bin/env python3
"""Run and audit a full forward/reverse phenol SA-CASSCF qualification scan."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scipy.optimize import linear_sum_assignment

from pyqed.models.phenol_coordinates import PhenolReactiveChart

from phenol_sa_casscf_sequential import (
    HARTREE_TO_EV,
    NCAS,
    NCORE,
    chain_worker,
    load_record,
    pyscf_molecule,
)


DEFAULT_DISTANCES = [
    0.90,
    0.94,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.30,
    1.40,
    1.55,
    1.70,
    1.85,
    1.95,
    2.05,
    2.20,
    2.50,
    3.00,
]


def record_path(output, backend, direction, distance):
    return output / backend / direction / f"r{distance:.5f}.npz"


def load_cut(output, backend, direction, distances):
    return [
        load_record(record_path(output, backend, direction, distance))
        for distance in distances
    ]


def assigned_overlap(overlap):
    overlap = np.asarray(overlap, dtype=float)
    row, col = linear_sum_assignment(-overlap)
    return overlap[row, col]


def same_geometry_active_singular(left, right, basis):
    mol = pyscf_molecule(left["geometry"], basis)
    overlap = mol.intor_symmetric("int1e_ovlp")
    active = (
        left["mo_coeff"][:, NCORE : NCORE + NCAS].T
        @ overlap
        @ right["mo_coeff"][:, NCORE : NCORE + NCAS]
    )
    return np.linalg.svd(active, compute_uv=False)


def _macro_and_restart_count(record):
    history = np.asarray(record["macro_history"], dtype=float).reshape(-1, 3)
    if str(np.asarray(record.get("backend", "")).item()) != "pyscf":
        return len(history), int(record.get("external_restarts", 0))
    macro_labels = history[:, 0]
    macro_count = 1 + int(np.count_nonzero(np.diff(macro_labels)))
    restart_count = int(np.count_nonzero(np.diff(macro_labels) < 0.0))
    return macro_count, restart_count


def audit_cut(records, distances, gradient_threshold, nstates):
    gradients = np.asarray([float(record["orbital_gradient"]) for record in records])
    counts = [_macro_and_restart_count(record) for record in records]
    macros = np.asarray([count[0] for count in counts], dtype=int)
    restarts = np.asarray([count[1] for count in counts], dtype=int)
    recoveries = np.asarray(
        [int(record.get("zero_step_recoveries", 0)) for record in records], dtype=int
    )
    spin = np.asarray([record["spins"] for record in records], dtype=float)
    wall = np.asarray([float(record["wall_seconds"]) for record in records])
    continuity = []
    active_singular = []
    continuity_by_geometry = []
    active_singular_by_geometry = []
    for record in records:
        if "previous_root_overlap" in record:
            assigned = assigned_overlap(record["previous_root_overlap"])
            continuity.extend(assigned.tolist())
            continuity_by_geometry.append(float(np.min(assigned)))
        else:
            continuity_by_geometry.append(None)
        if "active_singular_values" in record:
            singular = np.asarray(record["active_singular_values"], dtype=float)
            active_singular.extend(singular.tolist())
            active_singular_by_geometry.append(float(np.min(singular)))
        else:
            active_singular_by_geometry.append(None)
    failures = []
    for index, record in enumerate(records):
        if not bool(record.get("scf_converged", True)):
            failures.append({"index": index, "reason": "SCF did not converge"})
        if not bool(record.get("orbital_relaxed", False)):
            failures.append({"index": index, "reason": "orbitals not relaxed"})
        if gradients[index] > gradient_threshold:
            failures.append({"index": index, "reason": "gradient above threshold"})
        if np.asarray(record["energies"]).size != nstates:
            failures.append({"index": index, "reason": "incorrect root count"})
    return {
        "failures": failures,
        "max_gradient": float(np.max(gradients)),
        "median_gradient": float(np.median(gradients)),
        "max_macroiterations": int(np.max(macros)),
        "median_macroiterations": float(np.median(macros)),
        "total_external_restarts": int(np.sum(restarts)),
        "max_external_restarts": int(np.max(restarts)),
        "total_zero_step_recoveries": int(np.sum(recoveries)),
        "max_abs_s2": float(np.max(np.abs(spin))),
        "minimum_assigned_adjacent_root_overlap": (
            float(np.min(continuity)) if continuity else None
        ),
        "minimum_adjacent_active_singular_value": (
            float(np.min(active_singular)) if active_singular else None
        ),
        "minimum_assigned_adjacent_root_overlap_by_geometry": continuity_by_geometry,
        "minimum_adjacent_active_singular_value_by_geometry": active_singular_by_geometry,
        "total_wall_seconds": float(np.sum(wall)),
        "median_wall_seconds": float(np.median(wall)),
        "maximum_wall_seconds": float(np.max(wall)),
        "maximum_wall_distance_angstrom": float(distances[int(np.argmax(wall))]),
        "wall_seconds": wall.tolist(),
        "gradients": gradients.tolist(),
        "macroiterations": macros.tolist(),
        "external_restarts": restarts.tolist(),
        "zero_step_recoveries": recoveries.tolist(),
    }


def compare_energies(left, right):
    difference = np.abs(
        np.asarray([record["energies"] for record in left])
        - np.asarray([record["energies"] for record in right])
    ) * HARTREE_TO_EV * 1.0e3
    return {
        "max_mev": float(np.max(difference)),
        "rms_mev": float(np.sqrt(np.mean(difference**2))),
        "per_geometry_max_mev": np.max(difference, axis=1).tolist(),
        "all_state_mev": difference.tolist(),
    }


def plot_qualification(summary, figure_output):
    distances = np.asarray(summary["distances_angstrom"], dtype=float)
    colors = {"pyscf": "#D55E00", "pyqed": "#0072B2"}
    styles = {"forward": "-", "reverse": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.1), constrained_layout=True)

    for direction in ("forward", "reverse"):
        error = summary["backend_comparison"][direction]["per_geometry_max_mev"]
        axes[0, 0].plot(
            distances,
            error,
            marker="o",
            ms=3.2,
            linestyle=styles[direction],
            color="#6A3D9A",
            label=direction,
        )
    axes[0, 0].set(
        ylabel="max state error (meV)",
        title="PyQED versus PySCF",
    )
    axes[0, 0].legend(frameon=False)

    for backend in ("pyscf", "pyqed"):
        axes[0, 1].plot(
            distances,
            summary["hysteresis"][backend]["per_geometry_max_mev"],
            marker="o",
            ms=3.2,
            color=colors[backend],
            label={"pyscf": "PySCF", "pyqed": "PyQED"}[backend],
        )
    axes[0, 1].set(ylabel="forward/reverse difference (meV)", title="Hysteresis")
    axes[0, 1].legend(frameon=False)

    gradient_threshold = summary["gradient_threshold"]
    for backend in ("pyscf", "pyqed"):
        for direction in ("forward", "reverse"):
            values = summary["cuts"][backend][direction]["gradients"]
            axes[1, 0].semilogy(
                distances,
                values,
                marker="o",
                ms=3.0,
                linestyle=styles[direction],
                color=colors[backend],
                label=f"{backend} {direction}",
            )
    axes[1, 0].axhline(gradient_threshold, color="0.25", linestyle=":", linewidth=1)
    axes[1, 0].set(xlabel=r"$R_{\rm OH}$ ($\AA$)", ylabel=r"final $|g_{\rm orb}|_\infty$", title="Orbital convergence")
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    for backend in ("pyscf", "pyqed"):
        axes[1, 1].plot(
            distances,
            summary["direction_active_overlap"][backend],
            marker="s",
            ms=3.0,
            color=colors[backend],
            label={"pyscf": "PySCF", "pyqed": "PyQED"}[backend],
        )
    axes[1, 1].set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel="minimum active-space singular value",
        title="Forward/reverse active-space agreement",
        ylim=(0.0, 1.02),
    )
    axes[1, 1].legend(frameon=False)

    for label, axis in zip(("a", "b", "c", "d"), axes.flat):
        axis.text(-0.15, 1.05, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(color="0.90", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    figure_output.parent.mkdir(parents=True, exist_ok=True)
    png = figure_output.with_suffix(".png")
    pdf = figure_output.with_suffix(".pdf")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_energy_curves(cuts, summary, figure_output):
    distances = np.asarray(summary["distances_angstrom"], dtype=float)
    nstates = int(summary["root_count"])
    anchor = float(PhenolReactiveChart().equilibrium[0])
    anchor_index = int(np.argmin(np.abs(distances - anchor)))
    reference = float(cuts["pyscf"]["forward"][anchor_index]["energies"][0])
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, nstates))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), constrained_layout=True)

    for state, color in enumerate(colors):
        pyscf_energy = (
            np.asarray([row["energies"][state] for row in cuts["pyscf"]["forward"]])
            - reference
        ) * HARTREE_TO_EV
        pyqed_energy = (
            np.asarray([row["energies"][state] for row in cuts["pyqed"]["forward"]])
            - reference
        ) * HARTREE_TO_EV
        axes[0].plot(
            distances,
            pyscf_energy,
            color=color,
            linewidth=1.4,
            marker="o",
            ms=3.0,
            label=f"S{state}",
        )
        axes[0].plot(
            distances,
            pyqed_energy,
            color=color,
            linewidth=0,
            marker="s",
            ms=3.8,
            markerfacecolor="none",
        )
        for direction, linestyle in (("forward", "-"), ("reverse", "--")):
            difference = np.abs(
                np.asarray(
                    [row["energies"][state] for row in cuts["pyqed"][direction]]
                )
                - np.asarray(
                    [row["energies"][state] for row in cuts["pyscf"][direction]]
                )
            ) * HARTREE_TO_EV * 1.0e3
            axes[1].plot(
                distances,
                difference,
                color=color,
                linewidth=1.2,
                linestyle=linestyle,
            )

    axes[0].set(
        ylabel="energy relative to PySCF S0 minimum (eV)",
        title="Complete forward SA-CASSCF curves",
    )
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].text(
        0.99,
        0.02,
        "PySCF: filled circles + lines\nPyQED: open squares",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axes[1].set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel="absolute PyQED−PySCF error (meV)",
        title="All-state backend agreement (solid: forward; dashed: reverse)",
    )
    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.11, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(color="0.90", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    figure_output.parent.mkdir(parents=True, exist_ok=True)
    png = figure_output.with_name(figure_output.name + "_curves").with_suffix(".png")
    pdf = figure_output.with_name(figure_output.name + "_curves").with_suffix(".pdf")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_direction_pes(cuts, summary, figure_output):
    distances = np.asarray(summary["distances_angstrom"], dtype=float)
    nstates = int(summary["root_count"])
    anchor = float(PhenolReactiveChart().equilibrium[0])
    anchor_index = int(np.argmin(np.abs(distances - anchor)))
    reference = float(cuts["pyscf"]["forward"][anchor_index]["energies"][0])
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, nstates))
    fig, axes = plt.subplots(
        1, 2, figsize=(8.2, 3.65), sharex=True, sharey=True, constrained_layout=True
    )

    state_handles = []
    for axis, direction, panel in zip(axes, ("forward", "reverse"), ("a", "b")):
        for state, color in enumerate(colors):
            pyscf_energy = (
                np.asarray(
                    [row["energies"][state] for row in cuts["pyscf"][direction]]
                )
                - reference
            ) * HARTREE_TO_EV
            pyqed_energy = (
                np.asarray(
                    [row["energies"][state] for row in cuts["pyqed"][direction]]
                )
                - reference
            ) * HARTREE_TO_EV
            handle, = axis.plot(
                distances,
                pyscf_energy,
                color=color,
                linewidth=1.35,
                marker="o",
                ms=2.8,
                label=f"S{state}",
            )
            axis.plot(
                distances,
                pyqed_energy,
                color=color,
                linewidth=0,
                marker="s",
                ms=3.7,
                markerfacecolor="none",
                markeredgewidth=0.9,
            )
            if direction == "forward":
                state_handles.append(handle)
        axis.set(
            xlabel=r"$R_{\rm OH}$ ($\AA$)",
            title=f"{direction.capitalize()} branch",
        )
        axis.text(-0.13, 1.04, panel, transform=axis.transAxes, fontweight="bold")
        axis.grid(color="0.90", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("energy relative to forward PySCF S0 minimum (eV)")

    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([], [], color="0.25", marker="o", linewidth=1.35, ms=3.2, label="PySCF"),
        Line2D(
            [], [], color="0.25", marker="s", linewidth=0, ms=4.0,
            markerfacecolor="none", label="PyQED",
        ),
    ]
    fig.legend(
        state_handles + method_handles,
        [handle.get_label() for handle in state_handles + method_handles],
        loc="outside upper center",
        ncol=8,
        frameon=False,
        fontsize=8,
    )

    figure_output.parent.mkdir(parents=True, exist_ok=True)
    png = figure_output.with_name(figure_output.name + "_pes").with_suffix(".png")
    pdf = figure_output.with_name(figure_output.name + "_pes").with_suffix(".pdf")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_continuity(summary, figure_output):
    distances = np.asarray(summary["distances_angstrom"], dtype=float)
    colors = {"pyscf": "#D55E00", "pyqed": "#0072B2"}
    styles = {"forward": "-", "reverse": "--"}
    labels = {"pyscf": "PySCF", "pyqed": "PyQED"}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), constrained_layout=True)
    fields = (
        (
            "minimum_assigned_adjacent_root_overlap_by_geometry",
            "minimum assigned root overlap",
            "Adjacent six-state continuity",
        ),
        (
            "minimum_adjacent_active_singular_value_by_geometry",
            "minimum active-space singular value",
            "Adjacent active-space continuity",
        ),
    )
    for axis, (field, ylabel, title) in zip(axes, fields):
        for backend in ("pyscf", "pyqed"):
            for direction in ("forward", "reverse"):
                values = np.asarray(
                    [np.nan if value is None else value for value in summary["cuts"][backend][direction][field]],
                    dtype=float,
                )
                axis.plot(
                    distances,
                    values,
                    color=colors[backend],
                    linestyle=styles[direction],
                    marker="o",
                    ms=3.0,
                    label=f"{labels[backend]} {direction}",
                )
        axis.set(xlabel=r"$R_{\rm OH}$ ($\AA$)", ylabel=ylabel, title=title, ylim=(-0.02, 1.02))
        axis.grid(color="0.90", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold")

    figure_output.parent.mkdir(parents=True, exist_ok=True)
    png = figure_output.with_name(figure_output.name + "_continuity").with_suffix(".png")
    pdf = figure_output.with_name(figure_output.name + "_continuity").with_suffix(".pdf")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31+g*")
    parser.add_argument("--nstates", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--macro-cycles", type=int, default=50)
    parser.add_argument("--micro-cycles", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--conv-tol", type=float, default=2.0e-7)
    parser.add_argument("--conv-grad", type=float, default=1.0e-5)
    parser.add_argument("--conv-step", type=float, default=1.0e-3)
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--spin-shift", type=float, default=1.0)
    parser.add_argument("--pyqed-max-step", type=float, default=0.025)
    parser.add_argument("--pyqed-ah-cycles", type=int, default=20)
    parser.add_argument("--pyqed-ah-subspace", type=int, default=24)
    parser.add_argument("--pyqed-ah-tol", type=float, default=1.0e-7)
    parser.add_argument("--pyqed-keyframe-interval", type=int, default=4)
    parser.add_argument("--pyqed-keyframe-gradient-trust", type=float, default=3.0)
    parser.add_argument("--pyqed-active-overlap-floor", type=float, default=0.35)
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_oh_scan_native_rdm_7pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_production_qualification"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_production_qualification"),
    )
    parser.add_argument("--distances", type=float, nargs="*", default=DEFAULT_DISTANCES)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="reuse complete cached cuts and regenerate only audits and figures",
    )
    args = parser.parse_args()

    anchor_distance = float(PhenolReactiveChart().equilibrium[0])
    distances = sorted(set(args.distances) | {anchor_distance})
    options = {
        "macro_cycles": args.macro_cycles,
        "micro_cycles": args.micro_cycles,
        "restarts": args.restarts,
        "conv_tol": args.conv_tol,
        "conv_grad": args.conv_grad,
        "conv_step": args.conv_step,
        "scf_tol": args.scf_tol,
        "spin_shift": args.spin_shift,
        "pyqed_optimizer": "AH",
        "pyqed_max_step": args.pyqed_max_step,
        "pyqed_coupling": "qn",
        "pyqed_ah_cycles": args.pyqed_ah_cycles,
        "pyqed_ah_subspace": args.pyqed_ah_subspace,
        "pyqed_ah_tol": args.pyqed_ah_tol,
        "pyqed_keyframe_interval": args.pyqed_keyframe_interval,
        "pyqed_keyframe_gradient_trust": args.pyqed_keyframe_gradient_trust,
        "pyqed_active_overlap_floor": args.pyqed_active_overlap_floor,
        "pyqed_micro_ci_mode": "keyframe",
        "pyqed_reference_dir": None,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    min_distance = distances[0]
    max_distance = distances[-1]

    if not args.analyze_only:
        forward_tasks = []
        for backend in ("pyscf", "pyqed"):
            seed = args.seed_dir / backend / "decreasing" / f"r{min_distance:.5f}.npz"
            if not seed.exists():
                raise FileNotFoundError(seed)
            forward_tasks.append(
                (
                    backend,
                    "forward",
                    distances,
                    str(args.output),
                    str(seed),
                    args.basis,
                    args.nstates,
                    options,
                )
            )
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = [pool.submit(chain_worker, task) for task in forward_tasks]
            for future in as_completed(futures):
                backend, direction, completed = future.result()
                print(f"[{backend}:{direction}] complete ({len(completed)} points)", flush=True)

        reverse_tasks = []
        for backend in ("pyscf", "pyqed"):
            seed = record_path(args.output, backend, "forward", max_distance)
            reverse_tasks.append(
                (
                    backend,
                    "reverse",
                    list(reversed(distances)),
                    str(args.output),
                    str(seed),
                    args.basis,
                    args.nstates,
                    options,
                )
            )
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = [pool.submit(chain_worker, task) for task in reverse_tasks]
            for future in as_completed(futures):
                backend, direction, completed = future.result()
                print(f"[{backend}:{direction}] complete ({len(completed)} points)", flush=True)

    cuts = {
        backend: {
            direction: load_cut(args.output, backend, direction, distances)
            for direction in ("forward", "reverse")
        }
        for backend in ("pyscf", "pyqed")
    }
    summary = {
        "method": f"SA({args.nstates})-CASSCF(10e,10o)/{args.basis}",
        "distances_angstrom": distances,
        "gradient_threshold": args.conv_grad,
        "root_count": args.nstates,
        "spin_constraint": "fix_spin(ss=0)",
        "cuts": {
            backend: {
                direction: audit_cut(
                    cuts[backend][direction], distances, args.conv_grad, args.nstates
                )
                for direction in ("forward", "reverse")
            }
            for backend in ("pyscf", "pyqed")
        },
        "backend_comparison": {
            direction: compare_energies(
                cuts["pyqed"][direction], cuts["pyscf"][direction]
            )
            for direction in ("forward", "reverse")
        },
        "hysteresis": {
            backend: compare_energies(
                cuts[backend]["forward"], cuts[backend]["reverse"]
            )
            for backend in ("pyscf", "pyqed")
        },
        "direction_active_overlap": {},
    }
    for backend in ("pyscf", "pyqed"):
        summary["direction_active_overlap"][backend] = [
            float(
                np.min(
                    same_geometry_active_singular(
                        forward, reverse, args.basis
                    )
                )
            )
            for forward, reverse in zip(
                cuts[backend]["forward"], cuts[backend]["reverse"]
            )
        ]

    all_failures = [
        (backend, direction, failure)
        for backend in ("pyscf", "pyqed")
        for direction in ("forward", "reverse")
        for failure in summary["cuts"][backend][direction]["failures"]
    ]
    summary["all_points_converged"] = not all_failures
    summary["failure_count"] = len(all_failures)
    summary["minimum_direction_active_overlap"] = {
        backend: float(np.min(summary["direction_active_overlap"][backend]))
        for backend in ("pyscf", "pyqed")
    }
    total_wall = {
        backend: sum(
            summary["cuts"][backend][direction]["total_wall_seconds"]
            for direction in ("forward", "reverse")
        )
        for backend in ("pyscf", "pyqed")
    }
    summary["timing"] = {
        "total_wall_seconds": total_wall,
        "pyqed_over_pyscf": total_wall["pyqed"] / total_wall["pyscf"],
    }
    png, pdf = plot_qualification(summary, args.figure_output)
    curves_png, curves_pdf = plot_energy_curves(cuts, summary, args.figure_output)
    pes_png, pes_pdf = plot_direction_pes(cuts, summary, args.figure_output)
    continuity_png, continuity_pdf = plot_continuity(summary, args.figure_output)
    summary["figure"] = str(png)
    summary["figure_pdf"] = str(pdf)
    summary["curves_figure"] = str(curves_png)
    summary["curves_figure_pdf"] = str(curves_pdf)
    summary["pes_figure"] = str(pes_png)
    summary["pes_figure_pdf"] = str(pes_pdf)
    summary["continuity_figure"] = str(continuity_png)
    summary["continuity_figure_pdf"] = str(continuity_pdf)
    summary_path = args.output / "qualification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
