"""Compare continuous variational-Feshbach QGRF with standard sharp-shell RG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import ultraplot as uplt

from pyqed.narg.geometric_rg import (
    Phi4ContinuousQGRF,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
)


def _local_coupling_rates(field, potential_rate, radius=0.1):
    selected = np.abs(field) <= radius
    coefficients = np.polynomial.polynomial.polyfit(
        field[selected], potential_rate[selected], 8
    )
    return float(2.0 * coefficients[2]), float(24.0 * coefficients[4])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_qgrf_standard_rg"),
    )
    parser.add_argument("--field-points", type=int, default=141)
    parser.add_argument("--quadrature-order", type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    field = np.linspace(-0.7, 0.7, args.field_points)
    center = np.argmin(np.abs(field))
    couplings = Phi4GaussianCouplings(mass2=-0.3, quartic=6.0)
    standard = Phi4GaussianShell(spatial_dimension=1)
    standard_potential = standard.beta_potential(field, couplings)
    standard_temporal = np.array(
        [standard.external_kinetic_rates(value, couplings)[0] for value in field]
    )
    qgrf = Phi4ContinuousQGRF(
        quadrature_order=args.quadrature_order,
        derivative_step=1.0e-3,
    )
    qgrf_potential, qgrf_temporal = qgrf.rates(field, couplings)
    components = qgrf.components
    bare_curvature = standard.curvature(field, couplings)
    standard_beta = standard.beta(couplings)
    qgrf_mass_rate, qgrf_quartic_rate = _local_coupling_rates(
        field, qgrf_potential
    )
    minimum = np.sqrt(-6.0 * couplings.mass2 / couplings.quartic)
    minimum_index = np.argmin(np.abs(field - minimum))

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
        refheight=2.35,
        share=False,
        wspace=4.0,
        hspace=7.0,
    )
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"

    axis = axes[0]
    axis.plot(
        field,
        bare_curvature,
        color="0.45",
        linestyle="--",
        label=r"bare $U''(\phi)$",
    )
    axis.plot(
        field,
        components["hartree_mass2"],
        color=blue,
        label=r"variational $\mu^2(\phi)$",
    )
    axis.axhline(0.0, color="0.35", linewidth=0.7)
    axis.format(
        xlabel=r"background $\phi$",
        ylabel="shell mass squared",
        title="(a) Variational frame",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="lower right")

    axis = axes[1]
    axis.plot(
        field,
        standard_potential - standard_potential[center],
        color="0.45",
        linestyle="--",
        label="standard sharp-shell RG",
    )
    axis.plot(
        field,
        qgrf_potential - qgrf_potential[center],
        color=blue,
        label="variational-Feshbach QGRF",
    )
    axis.format(
        xlabel=r"background $\phi$",
        ylabel=r"$\beta_U(\phi)-\beta_U(0)$",
        title="(b) Potential flow",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper center")

    axis = axes[2]
    axis.plot(
        field,
        standard_temporal,
        color="0.45",
        linestyle="--",
        label="standard Gaussian response",
    )
    axis.plot(
        field,
        components["pair_temporal_rate"],
        color=orange,
        linestyle=":",
        label="QGRF pair channel",
    )
    axis.plot(
        field,
        components["triplet_temporal_rate"],
        color=green,
        linestyle="-.",
        label="QGRF triplet channel",
    )
    axis.plot(field, qgrf_temporal, color=blue, label="QGRF total")
    axis.format(
        xlabel=r"background $\phi$",
        ylabel=r"$\beta_{Z_t}(\phi)$",
        title="(c) Temporal response",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="upper center", ncols=2)

    axis = axes[3]
    gaussian_energy = components["gaussian_energy_rate"]
    axis.plot(
        field,
        100.0 * components["three_boson_energy_rate"] / gaussian_energy,
        color=purple,
        label="three-boson",
    )
    axis.plot(
        field,
        100.0 * components["four_boson_energy_rate"] / gaussian_energy,
        color=green,
        label="four-boson",
    )
    axis.axhline(0.0, color="0.35", linewidth=0.7)
    axis.format(
        xlabel=r"background $\phi$",
        ylabel="Feshbach / Gaussian (%)",
        title="(d) Residual shell energy",
        grid=False,
    )
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="center right")

    png = args.output_dir / "phi4_qgrf_standard_rg.png"
    pdf = png.with_suffix(".pdf")
    figure.savefig(png, dpi=400)
    figure.savefig(pdf)
    np.savez(
        args.output_dir / "comparison.npz",
        field=field,
        standard_potential=standard_potential,
        qgrf_potential=qgrf_potential,
        standard_temporal=standard_temporal,
        qgrf_temporal=qgrf_temporal,
        **components,
    )
    summary = {
        "model": "1+1D phi4",
        "couplings": {"mass2": couplings.mass2, "quartic": couplings.quartic},
        "standard_sharp_shell": {
            "beta_mass2": standard_beta.mass2,
            "beta_quartic": standard_beta.quartic,
            "beta_Zt_at_origin": float(standard_temporal[center]),
            "beta_Zt_at_bare_minimum": float(standard_temporal[minimum_index]),
        },
        "variational_feshbach_qgrf": {
            "beta_mass2_local_projection": qgrf_mass_rate,
            "beta_quartic_local_projection": qgrf_quartic_rate,
            "beta_Zt_at_origin": float(qgrf_temporal[center]),
            "beta_Zt_at_bare_minimum": float(qgrf_temporal[minimum_index]),
            "minimum_hartree_mass2": float(
                np.min(components["hartree_mass2"])
            ),
            "maximum_triplet_temporal_rate": float(
                np.max(components["triplet_temporal_rate"])
            ),
            "minimum_three_boson_energy_rate": float(
                np.min(components["three_boson_energy_rate"])
            ),
            "minimum_four_boson_energy_rate": float(
                np.min(components["four_boson_energy_rate"])
            ),
        },
        "spatial_projection": (
            "deferred: the sharp max-momentum projector is nonanalytic in external q"
        ),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure: {png}")


if __name__ == "__main__":
    main()
