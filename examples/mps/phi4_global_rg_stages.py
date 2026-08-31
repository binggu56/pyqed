"""Benchmark global Wegner--Houghton flow before staged geometric corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt

from pyqed.narg.geometric_rg import (
    Phi4FRG,
    Phi4FunctionalRegulatedQGRF,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
    Phi4WegnerHoughtonLPA,
)


def _global_benchmark():
    extents = [0.5, 0.6, 0.8, 1.0]
    records = []
    flows = []
    for index, extent in enumerate(extents):
        flow = Phi4WegnerHoughtonLPA(3).solve_fixed_point(
            field_maxima=extents[: index + 1],
            mesh_points=151,
            tolerance=1.0e-7,
        )
        if not flow.success:
            raise RuntimeError(flow.message)
        flow.stability_spectrum(points=80)
        records.append(
            {
                "field_max": extent,
                "curvature": flow.fixed_curvature,
                "theta_relevant": flow.relevant_eigenvalue,
                "nu": flow.correlation_exponent,
                "max_residual": flow.fixed_point_history[-1]["max_residual"],
            }
        )
        flows.append(flow)

    reference = flows[2]
    stability = []
    for points in [30, 40, 60, 80]:
        reference.stability_spectrum(points=points)
        stability.append(
            {
                "points": points,
                "theta_relevant": reference.relevant_eigenvalue,
                "nu": reference.correlation_exponent,
            }
        )
    try:
        Phi4WegnerHoughtonLPA(2).solve_fixed_point()
    except ValueError as error:
        d2_diagnostic = str(error)
    else:
        d2_diagnostic = "unexpectedly found an isolated D=2 LPA boundary condition"
    return flows, records, stability, d2_diagnostic


def _smooth_kinetic_benchmark():
    records = []
    flows = []
    for approximation in ["lpa", "lpa_prime", "de2"]:
        flow = Phi4FRG(
            order=4,
            approximation=approximation,
            wavefunction_order=1 if approximation == "de2" else None,
            spacetime_dimension=3,
            radial_order=40,
            angular_order=24,
        ).solve_fixed_point(tolerance=1.0e-7)
        if not flow.success:
            raise RuntimeError(flow.message)
        records.append(
            {
                "approximation": approximation,
                "nu": flow.correlation_exponent,
                "eta": flow.fixed_eta,
                "max_residual": float(np.max(np.abs(flow.fixed_beta))),
            }
        )
        flows.append(flow)
    return flows, records


def _qgrg_correction_benchmark():
    field = np.linspace(-0.8, 0.8, 17)
    flow = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=7,
        quadrature_order=12,
        feshbach_strength=1.0,
        kinetic_strength=0.0,
    )
    initial = Phi4GaussianShell.potential(
        field,
        Phi4GaussianCouplings(mass2=-0.628094322, quartic=4.30942319),
    )
    reverse_strengths = np.linspace(1.0, 0.0, 11)
    flow.continue_feshbach_potential_fixed_point(
        initial,
        reverse_strengths,
        modes=3,
        tolerance=2.0e-7,
        max_evaluations=500,
    )
    if not flow.success:
        raise RuntimeError(flow.message)
    reverse = flow.feshbach_continuation
    forward_strengths = np.linspace(0.0, 1.0, 11)
    flow.continue_feshbach_potential_fixed_point(
        flow.fixed_potential,
        forward_strengths,
        modes=3,
        tolerance=2.0e-7,
        max_evaluations=500,
    )
    if not flow.success:
        raise RuntimeError(flow.message)
    forward = flow.feshbach_continuation
    flow.continue_kinetic_fixed_point(
        flow.fixed_potential,
        np.linspace(0.0, 1.0, 7),
        modes=3,
        tolerance=2.0e-7,
        max_evaluations=500,
    )
    if not flow.success:
        raise RuntimeError(flow.message)
    physical = flow.stability_spectrum(modes=3, project_redundant=True)
    record = {
        "reverse_feshbach": [
            {key: value for key, value in point.items() if key != "potential"}
            for point in reverse
        ],
        "forward_feshbach": [
            {key: value for key, value in point.items() if key != "potential"}
            for point in forward
        ],
        "kinetic": [
            {
                key: value
                for key, value in point.items()
                if key not in {"potential", "inertia", "stiffness"}
            }
            for point in flow.kinetic_continuation
        ],
        "eta_t": flow.geometry["eta_t"],
        "eta_x": flow.geometry["eta_x"],
        "z": flow.geometry["dynamic_exponent"],
        "nu_projected": flow.correlation_exponent,
        "physical_stability_eigenvalues": [
            [float(value.real), float(value.imag)] for value in physical
        ],
        "four_mode_endpoint_reached": False,
    }
    return flow, record


def _plot(output, global_flows, global_records, stability, smooth_records, qgrg):
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=2,
        refwidth=3.0,
        refheight=2.25,
        share=False,
        wspace=4.6,
        hspace=5.8,
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"

    axis = axes[0]
    for flow, record in zip(global_flows, global_records):
        field = np.linspace(0.0, record["field_max"], 300)
        axis.plot(
            field,
            flow.fixed_solution.sol(field)[0],
            label=rf"$\phi_{{\max}}={record['field_max']:.1f}$",
        )
    axis.format(
        xlabel=r"field $\phi$",
        ylabel=r"fixed force $U_*'(\phi)$",
        title="a  Global sharp-cutoff fixed point",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper left", ncols=2)

    axis = axes[1]
    extents = [record["field_max"] for record in global_records]
    refinement = np.arange(len(extents))
    axis.plot(
        refinement,
        [record["nu"] for record in global_records],
        color=blue,
        marker="o",
        label=r"extent convergence",
    )
    axis.axhline(0.689459, color="0.4", linestyle="--", linewidth=0.9)
    axis.plot(
        refinement,
        [point["nu"] for point in stability],
        color=orange,
        marker="s",
        linestyle=":",
        label="stability grid",
    )
    axis.set_xticks(refinement)
    axis.set_xticklabels(
        [
            f"{extent:.1f} / {point['points']}"
            for extent, point in zip(extents, stability)
        ]
    )
    axis.format(
        xlabel=r"field extent $\phi_{\max}$ / collocation points",
        ylabel=r"$\nu$",
        title="b  Boundary and eigenvalue convergence",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper left")

    axis = axes[2]
    positions = np.arange(len(smooth_records))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [record["nu"] for record in smooth_records],
        width=width,
        color=blue,
        label=r"$\nu$",
    )
    axis.bar(
        positions + width / 2,
        [record["eta"] for record in smooth_records],
        width=width,
        color=orange,
        label=r"$\eta$",
    )
    axis.format(
        xticks=positions,
        xticklabels=["LPA", r"LPA$'$", r"local DE2"],
        ylabel="critical exponent",
        title=r"c  Controlled single-$Z$ benchmark",
        grid=False,
    )
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper right", ncols=2)

    axis = axes[3]
    forward = qgrg["forward_feshbach"]
    axis.plot(
        [point["strength"] for point in forward],
        [point["quartic"] for point in forward],
        color=purple,
        marker="o",
        label=r"Feshbach: $U_*''''(0)$",
    )
    metric_axis = axis.twinx()
    kinetic = qgrg["kinetic"]
    metric_axis.plot(
        [point["strength"] for point in kinetic],
        [point["eta_t"] for point in kinetic],
        color=orange,
        marker="s",
        label=r"metric: $\eta_t$",
    )
    metric_axis.plot(
        [point["strength"] for point in kinetic],
        [point["eta_x"] for point in kinetic],
        color=green,
        marker="^",
        linestyle="--",
        label=r"metric: $\eta_x$",
    )
    axis.format(
        xlabel="correction strength",
        ylabel=r"quartic vertex $U_*''''(0)$",
        title="d  QGRG correction homotopies",
        grid=False,
    )
    metric_axis.format(ylabel="anomalous dimension")
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper left")
    metric_axis.legend(frame=False, loc="lower right")

    png = output / "phi4_global_rg_stages.png"
    figure.savefig(png, dpi=400)
    figure.savefig(png.with_suffix(".pdf"))
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_global_rg_stages"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global_flows, global_records, stability, d2 = _global_benchmark()
    _, smooth_records = _smooth_kinetic_benchmark()
    _, qgrg = _qgrg_correction_benchmark()
    figure = _plot(
        args.output_dir,
        global_flows,
        global_records,
        stability,
        smooth_records,
        qgrg,
    )
    summary = {
        "global_wegner_houghton_d3": global_records,
        "stability_grid_convergence": stability,
        "d2_lpa_diagnostic": d2,
        "smooth_single_z_benchmark_d3": smooth_records,
        "qgrg_staged_corrections_d2": qgrg,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
