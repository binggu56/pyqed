"""Continue the clean functional QG-FRG fixed point and extract exponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt

from pyqed.narg.geometric_rg import (
    ExponentialRegulator,
    GaussianRegulator,
    Phi4FunctionalRegulatedQGRF,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
)


def _solve_case(case, kinetic_strengths):
    field = np.linspace(-case["extent"], case["extent"], case["points"])
    flow = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=max(7, 2 * case["modes"] + 1),
        quadrature_order=case["quadrature"],
        regulator=case["regulator"](),
        feshbach_strength=1.0,
        kinetic_strength=0.0,
    )
    initial = Phi4GaussianShell.potential(
        field,
        Phi4GaussianCouplings(
            mass2=-0.628094322,
            quartic=4.30942319,
        ),
    )
    flow.continue_potential_modes(
        initial,
        [3],
        tolerance=case.get("tolerance", 2.0e-7),
        max_evaluations=case.get("max_evaluations", 500),
    )
    if not flow.success:
        raise RuntimeError(
            f"{case['name']} three-mode potential solve failed: "
            f"{flow.message}; residual={np.max(np.abs(flow.fixed_beta)):.3e}"
        )
    flow.continue_kinetic_fixed_point(
        flow.fixed_potential,
        kinetic_strengths,
        modes=3,
        tolerance=case.get("tolerance", 2.0e-7),
        max_evaluations=case.get("max_evaluations", 500),
    )
    if not flow.success:
        raise RuntimeError(
            f"{case['name']} kinetic continuation failed: {flow.message}; "
            f"residual={np.max(np.abs(flow.fixed_beta)):.3e}"
        )
    if case["modes"] > 3:
        flow.continue_coupled_modes(
            flow.fixed_potential,
            np.arange(3, case["modes"] + 1),
            inertia=flow.fixed_inertia,
            stiffness=flow.fixed_stiffness,
            homotopy_steps=case.get("mode_homotopy_steps", 10),
            pseudo_arclength_steps=case.get("pseudo_arclength_steps", 40),
            tolerance=case.get("tolerance", 2.0e-7),
            max_evaluations=case.get("max_evaluations", 500),
        )
    endpoint_reached = bool(flow.success)
    eigenvalues = (
        flow.stability_spectrum(
            modes=case["modes"],
            step=case.get("stability_step", 2.0e-5),
            project_redundant=True,
        )
        if endpoint_reached
        else np.array([], dtype=complex)
    )
    center = flow.normalize_index
    curvature = flow.derivative(flow.fixed_potential, 2)
    quartic = flow.derivative(flow.fixed_potential, 4)
    interacting_branch = bool(endpoint_reached and abs(quartic[center]) > 1.0)
    redundancy = getattr(flow, "redundancy_diagnostics", {})
    redundant_eigenvalue = redundancy.get("dominant_full_mode_eigenvalue")
    if redundant_eigenvalue is not None:
        redundant_eigenvalue = [
            float(np.real(redundant_eigenvalue)),
            float(np.imag(redundant_eigenvalue)),
        ]
    record = {
        "name": case["name"],
        "extent": case["extent"],
        "points": case["points"],
        "quadrature": case["quadrature"],
        "modes": case["modes"],
        "regulator": case["regulator"].__name__,
        "max_residual": float(np.max(np.abs(flow.fixed_beta))),
        "fixed_point_endpoint_reached": endpoint_reached,
        "mass2": float(curvature[center]),
        "quartic": float(quartic[center]),
        "interacting_branch": interacting_branch,
        "minimum_gap2": float(np.min(1.0 + curvature)),
        "eta_t": float(flow.geometry["eta_t"]) if endpoint_reached else None,
        "eta_x": float(flow.geometry["eta_x"]) if endpoint_reached else None,
        "z": (
            float(flow.geometry["dynamic_exponent"])
            if endpoint_reached
            else None
        ),
        "theta_relevant": (
            float(flow.relevant_eigenvalue) if endpoint_reached else None
        ),
        "nu": float(flow.correlation_exponent) if endpoint_reached else None,
        "delta_phi": float(
            0.5
            * (
                flow.geometry["dynamic_exponent"]
                - 1.0
                + flow.geometry["eta_x"]
            )
        ) if endpoint_reached else None,
        "eigenvalues": [
            [float(value.real), float(value.imag)] for value in eigenvalues
        ],
        "full_eigenvalues": [
            [float(value.real), float(value.imag)]
            for value in getattr(flow, "stability_full_eigenvalues", [])
        ],
        "redundancy": {
            key: value
            for key, value in redundancy.items()
            if key != "dominant_full_mode_eigenvalue"
        }
        | ({"dominant_full_mode_eigenvalue": redundant_eigenvalue}
           if redundant_eigenvalue is not None else {}),
        "mode_continuation": [
            dict(point)
            for point in getattr(flow, "coupled_mode_continuation", [])
        ],
        "kinetic_continuation": [
            {
                key: value
                for key, value in point.items()
                if key not in {"potential", "inertia", "stiffness"}
            }
            for point in flow.kinetic_continuation
        ],
    }
    return flow, record


def _cases(quick):
    if quick:
        return [
            {
                "name": "quick",
                "extent": 0.8,
                "points": 17,
                "quadrature": 12,
                "modes": 3,
                "regulator": ExponentialRegulator,
            }
        ]
    return [
        {
            "name": "reference",
            "extent": 0.8,
            "points": 21,
            "quadrature": 14,
            "modes": 3,
            "regulator": ExponentialRegulator,
        },
        {
            "name": "field fine",
            "extent": 0.8,
            "points": 25,
            "quadrature": 14,
            "modes": 3,
            "regulator": ExponentialRegulator,
        },
        {
            "name": "field wide",
            "extent": 1.0,
            "points": 25,
            "quadrature": 14,
            "modes": 3,
            "regulator": ExponentialRegulator,
        },
        {
            "name": "quadrature fine",
            "extent": 0.8,
            "points": 21,
            "quadrature": 18,
            "modes": 3,
            "regulator": ExponentialRegulator,
        },
        {
            "name": "4 modes",
            "extent": 0.8,
            "points": 25,
            "quadrature": 14,
            "modes": 4,
            "regulator": ExponentialRegulator,
            "pseudo_arclength_steps": 20,
        },
        {
            "name": "Gaussian regulator",
            "extent": 0.8,
            "points": 21,
            "quadrature": 14,
            "modes": 3,
            "regulator": GaussianRegulator,
        },
    ]


def _plot(output, flows, records):
    reference = flows[0]
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
        refheight=2.3,
        share=False,
        wspace=4.5,
        hspace=6.0,
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"

    axis = axes[0]
    continuation = reference.kinetic_continuation
    strength = np.array([point["strength"] for point in continuation])
    axis.plot(
        strength,
        [point["eta_t"] for point in continuation],
        color=orange,
        marker="o",
        label=r"$\eta_t$",
    )
    axis.plot(
        strength,
        [point["eta_x"] for point in continuation],
        color=blue,
        marker="s",
        label=r"$\eta_x$",
    )
    axis.axhline(0.25, color="0.4", linestyle="--", linewidth=0.8)
    axis.format(
        xlabel=r"kinetic feedback $\kappa$",
        ylabel="anomalous dimension",
        title="a  Interacting-branch continuation",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper left", ncols=2)

    axis = axes[1]
    center = reference.normalize_index
    axis.plot(
        reference.field,
        reference.fixed_potential - reference.fixed_potential[center],
        color=blue,
        label=r"$U_*(\phi)-U_*(0)$",
    )
    axis.plot(
        reference.field,
        reference.fixed_inertia - 1.0,
        color=orange,
        linestyle="--",
        label=r"$Z_{t,*}(\phi)-1$",
    )
    axis.plot(
        reference.field,
        reference.fixed_stiffness - 1.0,
        color=green,
        linestyle=":",
        label=r"$Z_{x,*}(\phi)-1$",
    )
    axis.format(
        xlabel=r"background $\phi$",
        ylabel="fixed-point function",
        title="b  Functional fixed point",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="best")

    axis = axes[2]
    positions = np.arange(len(records))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [np.nan if record["nu"] is None else record["nu"] for record in records],
        width=width,
        color=blue,
        label=r"$\nu$",
    )
    axis.bar(
        positions + width / 2,
        [
            np.nan if record["eta_x"] is None else record["eta_x"]
            for record in records
        ],
        width=width,
        color=orange,
        label=r"$\eta_x$",
    )
    axis.axhline(1.0, color=blue, linestyle="--", linewidth=0.8)
    axis.axhline(0.25, color=orange, linestyle="--", linewidth=0.8)
    collapsed = [
        index
        for index, record in enumerate(records)
        if not record["interacting_branch"]
    ]
    if collapsed:
        axis.scatter(
            collapsed,
            [0.03 for _ in collapsed],
            marker="x",
            color="black",
            s=28,
            label="branch lost",
            zorder=4,
        )
    axis.format(
        xticks=positions,
        xticklabels=[record["name"] for record in records],
        ylabel="critical exponent",
        title="c  Numerical and regulator dependence",
        grid=False,
    )
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper right", ncols=2)

    axis = axes[3]
    eigenvalues = np.asarray(reference.stability_eigenvalues)
    full_eigenvalues = np.asarray(reference.stability_full_eigenvalues)
    eigenmode = np.arange(1, eigenvalues.size + 1)
    full_mode = np.arange(1, full_eigenvalues.size + 1)
    axis.bar(
        full_mode,
        full_eigenvalues.real,
        color="none",
        edgecolor="0.55",
        linewidth=1.0,
        width=0.78,
        label="unprojected",
    )
    colors = [purple if value > 0.0 else "0.55" for value in eigenvalues.real]
    axis.bar(
        eigenmode,
        eigenvalues.real,
        color=colors,
        width=0.5,
        label="physical projection",
    )
    axis.axhline(0.0, color="0.45", linewidth=0.8)
    axis.format(
        xlabel="stability eigenmode",
        ylabel=r"$\operatorname{Re}\theta$",
        xticks=eigenmode,
        title="d  Gauge-fixed stability spectrum",
        grid=False,
    )
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="lower left")

    png = output / "phi4_clean_qgfrg_exponents.png"
    figure.savefig(png, dpi=400)
    figure.savefig(png.with_suffix(".pdf"))
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_clean_qgfrg"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--continuation-steps", type=int, default=9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kinetic_strengths = np.linspace(0.0, 1.0, args.continuation_steps)

    flows = []
    records = []
    for case in _cases(args.quick):
        flow, record = _solve_case(case, kinetic_strengths)
        flows.append(flow)
        records.append(record)
        if record["fixed_point_endpoint_reached"]:
            print(
                f"{case['name']}: nu={record['nu']:.6f}, "
                f"eta_x={record['eta_x']:.6f}, "
                f"eta_t={record['eta_t']:.6f}, "
                f"residual={record['max_residual']:.3e}, interacting=True"
            )
        else:
            print(
                f"{case['name']}: higher-mode endpoint not reached; "
                f"no exponents extracted"
            )

    figure = _plot(args.output_dir, flows, records)
    summary = {
        "model": "clean smooth-regulator functional QG-FRG for 1+1D phi4",
        "exact_ising": {"nu": 1.0, "eta": 0.25, "z": 1.0},
        "cases": records,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    np.savez(
        args.output_dir / "reference_fixed_point.npz",
        field=flows[0].field,
        potential=flows[0].fixed_potential,
        inertia=flows[0].fixed_inertia,
        stiffness=flows[0].fixed_stiffness,
        stability_eigenvalues=flows[0].stability_eigenvalues,
    )
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
