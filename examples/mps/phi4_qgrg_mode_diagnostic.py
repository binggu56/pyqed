"""Diagnose relevant QGRG modes and higher-mode continuation in 1+1D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
from matplotlib.patches import Patch
import numpy as np
import ultraplot as uplt

from pyqed.narg.geometric_rg import (
    Phi4FunctionalRegulatedQGRF,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
)


def _solve(points):
    field = np.linspace(-0.8, 0.8, points)
    flow = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=9,
        quadrature_order=12,
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
        initial, [3], tolerance=2.0e-7, max_evaluations=500
    )
    flow.continue_kinetic_fixed_point(
        flow.fixed_potential,
        np.linspace(0.0, 1.0, 7),
        modes=3,
        tolerance=2.0e-7,
        max_evaluations=500,
    )
    if not flow.success:
        raise RuntimeError(flow.message)
    return flow


def _serial_mode(mode):
    value = mode["eigenvalue"]
    return {
        **{key: value for key, value in mode.items() if key != "eigenvalue"},
        "eigenvalue": [float(value.real), float(value.imag)],
    }


def _plot(output, reference_modes, grid_spectra, probe):
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
        ncols=3,
        refwidth=2.25,
        refheight=2.15,
        share=False,
        wspace=4.8,
    )
    colors = {
        "potential": "#0072B2",
        "temporal metric": "#D55E00",
        "spatial metric": "#009E73",
    }

    axis = axes[0]
    indices = np.arange(len(reference_modes))
    values = np.array([mode["eigenvalue"][0] for mode in reference_modes])
    axis.bar(
        indices,
        values,
        color=[colors[mode["dominant_block"]] for mode in reference_modes],
        width=0.62,
    )
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.format(
        xlabel="stability mode",
        ylabel=r"eigenvalue $\theta$",
        title="a  Two relevant directions",
        xticks=indices,
        grid=False,
    )
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.legend(
        handles=[
            Patch(color=colors[label], label=label)
            for label in ("potential", "temporal metric", "spatial metric")
        ],
        frameon=False,
        loc="upper right",
        ncols=1,
        bbox_to_anchor=(0.98, 0.98),
    )

    axis = axes[1]
    for points, spectrum in grid_spectra.items():
        axis.plot(
            np.arange(len(spectrum)),
            spectrum,
            marker="o" if points == 17 else "s",
            label=f"{points} field points",
        )
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.format(
        xlabel="ordered stability mode",
        ylabel=r"$\theta$",
        title="b  Field-grid check",
        grid=False,
    )
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="lower left")

    axis = axes[2]
    accepted = [point for point in probe if point["accepted"]]
    rejected = [point for point in probe if not point["accepted"]]
    axis.plot(
        [point["strength"] for point in accepted],
        [point["quartic"] for point in accepted],
        color="#CC79A7",
        marker="o",
        label=r"accepted $U''''(0)$",
    )
    if rejected:
        axis.scatter(
            [point["strength"] for point in rejected],
            [point["quartic"] for point in rejected],
            color="#CC79A7",
            marker="x",
            label="rejected corrector",
        )
    singular_axis = axis.twinx()
    singular_axis.semilogy(
        [point["strength"] for point in probe],
        [point["smallest_jacobian_singular_value"] for point in probe],
        color="#009E73",
        marker="s",
        linestyle="none",
        label=r"smallest $\sigma(J)$",
    )
    axis.format(
        xlabel="new-equation activation",
        ylabel=r"quartic vertex $U''''(0)$",
        title="c  Four-mode continuation fold",
        grid=False,
    )
    singular_axis.format(ylabel=r"smallest singular value")
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(frame=False, loc="center right")
    singular_axis.legend(frame=False, loc="upper right")

    png = output / "phi4_qgrg_mode_diagnostic.png"
    figure.savefig(png, dpi=400)
    figure.savefig(png.with_suffix(".pdf"))
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_qgrg_mode_diagnostic"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flows = {points: _solve(points) for points in (17, 21)}
    grid_spectra = {}
    for points, flow in flows.items():
        values = flow.stability_spectrum(
            modes=3, step=2.0e-5, project_redundant=True
        )
        grid_spectra[points] = [float(value.real) for value in values]
    reference = flows[17]
    modes = [_serial_mode(mode) for mode in reference.stability_mode_diagnostics]
    stability_steps = {}
    for step in (1.0e-5, 2.0e-5, 5.0e-5):
        values = reference.stability_spectrum(
            modes=3, step=step, project_redundant=True
        )
        stability_steps[str(step)] = [float(value.real) for value in values]
    modes = [_serial_mode(mode) for mode in reference.stability_mode_diagnostics]
    redundancy = dict(reference.redundancy_diagnostics)
    redundant_value = redundancy.pop("dominant_full_mode_eigenvalue")
    redundancy["dominant_full_mode_eigenvalue"] = [
        float(np.real(redundant_value)),
        float(np.imag(redundant_value)),
    ]

    reference.probe_coupled_mode_extension(
        reference.fixed_potential,
        3,
        4,
        inertia=reference.fixed_inertia,
        stiffness=reference.fixed_stiffness,
        initial_step=0.04,
        minimum_step=0.005,
        fold_tolerance=1.0e-3,
        tolerance=2.0e-7,
        max_evaluations=200,
    )
    summary = {
        "model": "functional QGRG for 1+1D phi4",
        "exact_ising_nu": 1.0,
        "grid_spectra": grid_spectra,
        "stability_step_spectra": stability_steps,
        "mode_diagnostics": modes,
        "redundancy_audit": redundancy,
        "mode_extension": reference.mode_extension_diagnostics,
        "mode_extension_history": reference.mode_extension_probe,
        "conclusion": (
            "the second relevant mode is metric dominated but is not a "
            "validated redundant eigenoperator; the interacting branch "
            "encounters a near-singular fold before four-mode activation"
        ),
    }
    with open(
        args.output_dir / "summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    figure = _plot(
        args.output_dir,
        modes,
        grid_spectra,
        reference.mode_extension_probe,
    )
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
