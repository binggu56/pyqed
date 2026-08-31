"""Run the standard functional FRG workflow for the 3D Ising universality class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt

from pyqed.narg import Phi4FRG


EXACT = {"nu": 0.629971, "eta": 0.036298}


def _driver(order, approximation, args, *, wavefunction_order=None):
    return Phi4FRG(
        order=order,
        approximation=approximation,
        wavefunction_order=wavefunction_order,
        spacetime_dimension=3,
        radial_order=args.radial_order,
        angular_order=args.angular_order,
    )


def _sequence(approximation, args):
    flow = _driver(args.max_order, approximation, args)
    flow.solve_fixed_point(
        tolerance=args.tolerance,
        max_evaluations=args.max_evaluations,
    )
    if not flow.success:
        raise RuntimeError(flow.message)

    records = []
    stages = []
    for point in flow.fixed_point_history:
        stage = _driver(point["order"], approximation, args)
        stage.fixed_state = point["state"].copy()
        stage.fixed_beta = stage.beta(stage.fixed_state)[0]
        stage.fixed_eta = point["eta"]
        stage.stability_spectrum()
        stages.append(stage)
        records.append(
            {
                "approximation": approximation,
                "order": point["order"],
                "wavefunction_order": 0,
                "state": point["state"].tolist(),
                "eta": point["eta"],
                "nu": stage.correlation_exponent,
                "relevant_eigenvalue": stage.relevant_eigenvalue,
                "max_residual": point["max_residual"],
                "stability_eigenvalues": [
                    [float(value.real), float(value.imag)]
                    for value in stage.stability_eigenvalues
                ],
            }
        )
    return flow, stages, records


def _de2_sequence(args):
    flow = _driver(
        args.de2_order,
        "de2",
        args,
        wavefunction_order=args.max_wavefunction_order,
    )
    flow.solve_fixed_point(
        tolerance=args.tolerance,
        max_evaluations=args.max_evaluations,
    )
    if not flow.success:
        raise RuntimeError(flow.message)

    stages = []
    records = []
    for point in flow.fixed_point_history:
        wavefunction_order = point["wavefunction_order"]
        if wavefunction_order == 0:
            continue
        stage = _driver(
            point["order"],
            "de2",
            args,
            wavefunction_order=wavefunction_order,
        )
        stage.fixed_state = point["state"].copy()
        stage.fixed_beta, stage.fixed_eta = stage.beta(stage.fixed_state)
        stage.stability_spectrum()
        stages.append(stage)
        records.append(
            {
                "approximation": "de2",
                "order": point["order"],
                "wavefunction_order": wavefunction_order,
                "state": point["state"].tolist(),
                "eta": point["eta"],
                "nu": stage.correlation_exponent,
                "relevant_eigenvalue": stage.relevant_eigenvalue,
                "max_residual": point["max_residual"],
                "physical_eta": bool(point["eta"] >= 0.0),
                "stability_eigenvalues": [
                    [float(value.real), float(value.imag)]
                    for value in stage.stability_eigenvalues
                ],
            }
        )
    return flow, stages, records


def _plot(output, sequences, de2_stages):
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
        refwidth=2.9,
        refheight=2.2,
        share=False,
        wspace=6.0,
        hspace=5.0,
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#7A5195"

    lpa_prime = sequences["lpa_prime"]
    axis = axes[0]
    field = np.linspace(-0.75, 0.75, 301)
    potential_stages = [lpa_prime[-1], *de2_stages]
    potential_styles = [
        ("0.3", r"LPA$'$"),
        (green, r"local DE2, $M=1$"),
        (purple, r"local DE2, $M=2$"),
    ]
    for stage, (color, label) in zip(potential_stages, potential_styles):
        axis.plot(
            field,
            stage.potential(field),
            color=color,
            label=label,
        )
    axis.format(
        xlabel=r"field $\phi$",
        ylabel=r"$u_*(\phi)-u_*(\phi_0)$",
        title="a  Fixed-point potential",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper center", ncols=1)

    axis = axes[1]
    for approximation, color, marker, label in (
        ("lpa", blue, "o", "LPA"),
        ("lpa_prime", orange, "s", r"LPA$'$"),
    ):
        stages = sequences[approximation]
        axis.plot(
            [stage.order for stage in stages],
            [stage.correlation_exponent for stage in stages],
            color=color,
            marker=marker,
            label=label,
        )
    for stage, color, marker, label in zip(
        de2_stages,
        (green, purple),
        ("^", "D"),
        (r"local DE2, $M=1$", r"local DE2, $M=2$"),
    ):
        axis.scatter(
            [stage.order],
            [stage.correlation_exponent],
            color=color,
            marker=marker,
            s=28,
            zorder=4,
        )
    axis.axhline(EXACT["nu"], color="0.35", linestyle="--", linewidth=1.0)
    axis.format(
        xlabel="potential order $N$",
        ylabel=r"correlation exponent $\nu$",
        title="b  Order convergence",
        xlim=(1.9, max(stage.order for stage in lpa_prime) + 0.15),
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper right", ncols=1)

    axis = axes[2]
    reference_stage = next(
        stage for stage in lpa_prime if stage.order == de2_stages[0].order
    )
    kinetic_stages = [reference_stage, *de2_stages]
    axis.plot(
        range(len(kinetic_stages)),
        [stage.fixed_eta for stage in kinetic_stages],
        color=green,
        marker="o",
        label="local closure",
    )
    axis.axhline(EXACT["eta"], color="0.35", linestyle="--", linewidth=1.0)
    if kinetic_stages[-1].fixed_eta < 0.0:
        axis.scatter(
            [len(kinetic_stages) - 1],
            [kinetic_stages[-1].fixed_eta],
            color="#C44E52",
            marker="x",
            s=40,
            zorder=5,
            label="nonphysical",
        )
    axis.format(
        xlabel="wavefunction order $M$",
        ylabel=r"anomalous dimension $\eta$",
        title="c  Kinetic truncation",
        xticks=range(len(kinetic_stages)),
        xlim=(-0.08, len(kinetic_stages) - 0.85),
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="best")

    axis = axes[3]
    rho = np.linspace(0.0, 0.14, 241)
    for stage, color, label in zip(
        de2_stages,
        (green, purple),
        (r"$M=1$", r"$M=2$"),
    ):
        axis.plot(
            rho,
            stage.wavefunction(np.sqrt(2.0 * rho)),
            color=color,
            label=label,
        )
        axis.scatter(
            [stage.fixed_state[0]], [1.0], color=color, s=18, zorder=4
        )
    axis.axhline(1.0, color="0.55", linestyle="--", linewidth=0.8)
    axis.format(
        xlabel=r"invariant $\rho=\phi^2/2$",
        ylabel=r"$Z_*(\rho)$",
        title="d  Wavefunction profile",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="best")

    png = output / "phi4_standard_frg.png"
    figure.savefig(png, dpi=400)
    figure.savefig(png.with_suffix(".pdf"))
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_standard_frg"),
    )
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--de2-order", type=int, default=4)
    parser.add_argument(
        "--max-wavefunction-order", type=int, choices=(1, 2), default=2
    )
    parser.add_argument("--radial-order", type=int, default=60)
    parser.add_argument("--angular-order", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-evaluations", type=int, default=6000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.max_order = min(args.max_order, 4)
        args.de2_order = min(args.de2_order, 4)
        args.radial_order = min(args.radial_order, 40)
        args.angular_order = min(args.angular_order, 24)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flows = {}
    sequences = {}
    records = []
    for approximation in ("lpa", "lpa_prime"):
        flow, stages, values = _sequence(approximation, args)
        flows[approximation] = flow
        sequences[approximation] = stages
        records.extend(values)
        final = values[-1]
        print(
            f"{approximation}: N={final['order']}, "
            f"nu={final['nu']:.8f}, eta={final['eta']:.8f}, "
            f"residual={final['max_residual']:.3e}"
        )

    de2_flow, de2_stages, de2_records = _de2_sequence(args)
    records.extend(de2_records)
    for value in de2_records:
        print(
            f"de2: N={value['order']}, M={value['wavefunction_order']}, "
            f"nu={value['nu']:.8f}, eta={value['eta']:.8f}, "
            f"residual={value['max_residual']:.3e}, "
            f"physical_eta={value['physical_eta']}"
        )

    figure = _plot(args.output_dir, sequences, de2_stages)
    summary = {
        "model": "smooth-regulator FRG with a local DE2 closure for 3D Ising",
        "de2_scope": (
            "field-dependent Z with local potential-vertex bubbles; "
            "Z-derivative vertices are omitted"
        ),
        "exact_ising": EXACT,
        "records": records,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    arrays = {
        "lpa": flows["lpa"].fixed_state,
        "lpa_prime": flows["lpa_prime"].fixed_state,
        "lpa_prime_eigenvalues": flows["lpa_prime"].stability_eigenvalues,
    }
    for stage in de2_stages:
        label = f"de2_m{stage.wavefunction_order}"
        arrays[label] = stage.fixed_state
        arrays[f"{label}_eigenvalues"] = stage.stability_eigenvalues
    np.savez(args.output_dir / "fixed_points.npz", **arrays)
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
