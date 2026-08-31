"""Test the functional regulator-based QGRF closure for 1+1D phi4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt
from scipy.optimize import root

from pyqed.narg.geometric_rg import (
    Phi4FunctionalRegulatedQGRF,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
)


def _quartic_fixed_point(flow, guess):
    field = flow.field
    center = flow.normalize_index

    def projected_beta(values):
        potential = Phi4GaussianShell.potential(
            field,
            Phi4GaussianCouplings(mass2=values[0], quartic=values[1]),
        )
        rate = flow.potential_rate(potential)
        return np.array(
            [flow.derivative(rate, 2)[center], flow.derivative(rate, 4)[center]]
        )

    solution = root(projected_beta, np.asarray(guess, dtype=float), tol=1.0e-9)
    if not solution.success:
        raise RuntimeError(solution.message)
    projected_beta(solution.x)
    return solution.x, {
        "eta_t": flow.geometry["eta_t"],
        "eta_x": flow.geometry["eta_x"],
        "z": flow.geometry["dynamic_exponent"],
        "residual": projected_beta(solution.x).tolist(),
    }


def _propagate(flow, couplings, *, ell_max, step):
    field = flow.field
    center = flow.normalize_index
    potential = Phi4GaussianShell.potential(field, couplings)
    history = {
        "ell": [0.0],
        "potential": [potential.copy()],
        "mass2": [float(flow.derivative(potential, 2)[center])],
        "quartic": [float(flow.derivative(potential, 4)[center])],
        "minimum_gap2": [float(np.min(1.0 + flow.derivative(potential, 2)))],
    }
    status = "completed"
    message = "reached ell_max"
    failure_ell = None
    failure_gap2 = None
    nsteps = int(np.ceil(ell_max / step))
    for index in range(nsteps):
        try:
            rate = flow.potential_rate(potential)
            candidate = potential + step * rate
            candidate = 0.5 * (candidate + candidate[::-1])
            curvature = flow.derivative(candidate, 2)
            gap2 = 1.0 + curvature
            if np.min(gap2) <= 0.0 or not np.all(np.isfinite(candidate)):
                status = "unstable"
                message = "the local functional frame lost its positive regulator gap"
                failure_ell = (index + 1) * step
                failure_gap2 = float(np.min(gap2))
                break
            potential = candidate
        except ValueError as error:
            status = "unstable"
            message = str(error)
            break
        ell = (index + 1) * step
        history["ell"].append(ell)
        history["potential"].append(potential.copy())
        history["mass2"].append(float(curvature[center]))
        history["quartic"].append(
            float(flow.derivative(potential, 4)[center])
        )
        history["minimum_gap2"].append(float(np.min(gap2)))
    for name in history:
        history[name] = np.asarray(history[name])
    history["status"] = status
    history["message"] = message
    history["failure_ell"] = failure_ell
    history["failure_gap2"] = failure_gap2
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_functional_regulated_qgrf"),
    )
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument("--ell-max", type=float, default=0.01)
    parser.add_argument("--step", type=float, default=1.0e-4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    field = np.linspace(-0.8, 0.8, 33)
    gaussian = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=7,
        quadrature_order=args.quadrature_order,
        include_feshbach=False,
    )
    feshbach = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=7,
        quadrature_order=args.quadrature_order,
        include_feshbach=True,
    )
    gaussian_fixed, gaussian_fixed_data = _quartic_fixed_point(
        gaussian, [-0.22, 3.97]
    )
    feshbach_fixed, feshbach_fixed_data = _quartic_fixed_point(
        feshbach, [-0.61, 4.5]
    )
    gaussian_history = _propagate(
        gaussian,
        Phi4GaussianCouplings(
            mass2=gaussian_fixed[0], quartic=gaussian_fixed[1]
        ),
        ell_max=args.ell_max,
        step=args.step,
    )
    feshbach_history = _propagate(
        feshbach,
        Phi4GaussianCouplings(
            mass2=feshbach_fixed[0], quartic=feshbach_fixed[1]
        ),
        ell_max=args.ell_max,
        step=args.step,
    )

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
        wspace=4.0,
        hspace=6.0,
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"
    center = field.size // 2

    axis = axes[0]
    for history, color, label in (
        (gaussian_history, blue, "regulated Gaussian"),
        (feshbach_history, orange, "Gaussian + local Feshbach"),
    ):
        initial = history["potential"][0]
        final = history["potential"][-1]
        axis.plot(
            field,
            final - final[center],
            color=color,
            label=label,
        )
        axis.plot(
            field,
            initial - initial[center],
            color=color,
            linestyle=":",
            linewidth=0.9,
        )
    axis.format(
        xlabel=r"background $\phi$",
        ylabel=r"$U(\phi)-U(0)$",
        title="a  Functional potential",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper center")

    axis = axes[1]
    for flow, history, color, label in (
        (gaussian, gaussian_history, blue, "regulated Gaussian"),
        (feshbach, feshbach_history, orange, "local Feshbach"),
    ):
        curvature = flow.derivative(history["potential"][-1], 2)
        axis.plot(field, curvature, color=color, label=label)
    axis.axhline(-1.0, color="0.4", linewidth=0.8, linestyle="--")
    axis.format(
        xlabel=r"background $\phi$",
        ylabel=r"$U''(\phi)$",
        title="b  Regulated curvature",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper center")

    axis = axes[2]
    axis.plot(
        gaussian_history["ell"],
        gaussian_history["quartic"],
        color=blue,
        label="regulated Gaussian",
    )
    axis.plot(
        feshbach_history["ell"],
        feshbach_history["quartic"],
        color=orange,
        label="local Feshbach",
    )
    axis.format(
        xlabel=r"RG time $\ell$",
        ylabel=r"$U''''(0)$",
        title="c  Generated local vertex",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="best")

    axis = axes[3]
    axis.plot(
        gaussian_history["ell"],
        gaussian_history["minimum_gap2"],
        color=green,
        label="regulated Gaussian",
    )
    axis.plot(
        feshbach_history["ell"],
        feshbach_history["minimum_gap2"],
        color=purple,
        label="local Feshbach",
    )
    if feshbach_history["failure_ell"] is not None:
        axis.plot(
            [
                feshbach_history["ell"][-1],
                feshbach_history["failure_ell"],
            ],
            [feshbach_history["minimum_gap2"][-1], 0.0],
            color=purple,
            linestyle=":",
            linewidth=1.0,
        )
        axis.scatter(
            feshbach_history["failure_ell"],
            0.0,
            color=purple,
            marker="x",
            zorder=3,
            label="gap lost",
        )
    axis.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    axis.format(
        xlabel=r"RG time $\ell$",
        ylabel=r"$\min_\phi[1+U''(\phi)]$",
        title="d  Stability diagnostic",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="center right", ncols=1)

    png = args.output_dir / "phi4_functional_regulated_qgrf.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png, dpi=400)
    figure.savefig(pdf)
    summary = {
        "model": "functional regulator-based 1+1D phi4 QGRF",
        "field_points": field.size,
        "quadrature_order": args.quadrature_order,
        "gaussian_quartic_projection": {
            "mass2": float(gaussian_fixed[0]),
            "quartic": float(gaussian_fixed[1]),
            **gaussian_fixed_data,
        },
        "feshbach_quartic_projection": {
            "mass2": float(feshbach_fixed[0]),
            "quartic": float(feshbach_fixed[1]),
            **feshbach_fixed_data,
        },
        "gaussian_functional_flow": {
            "status": gaussian_history["status"],
            "message": gaussian_history["message"],
            "last_ell": float(gaussian_history["ell"][-1]),
        },
        "feshbach_functional_flow": {
            "status": feshbach_history["status"],
            "message": feshbach_history["message"],
            "last_stable_ell": float(feshbach_history["ell"][-1]),
            "failure_ell": feshbach_history["failure_ell"],
            "candidate_minimum_gap2": feshbach_history["failure_gap2"],
        },
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    np.savez(
        args.output_dir / "flow.npz",
        field=field,
        gaussian_ell=gaussian_history["ell"],
        gaussian_potential=gaussian_history["potential"],
        gaussian_mass2=gaussian_history["mass2"],
        gaussian_quartic=gaussian_history["quartic"],
        gaussian_gap2=gaussian_history["minimum_gap2"],
        feshbach_ell=feshbach_history["ell"],
        feshbach_potential=feshbach_history["potential"],
        feshbach_mass2=feshbach_history["mass2"],
        feshbach_quartic=feshbach_history["quartic"],
        feshbach_gap2=feshbach_history["minimum_gap2"],
    )
    print(json.dumps(summary, indent=2))
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
