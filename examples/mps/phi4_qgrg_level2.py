"""Diagnose the nonuniform Gaussian response in Level-2 phi4 QGRG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg.geometric_rg import (
    Phi4GaussianCouplings,
    Phi4GaussianShell,
    Phi4SmoothQGRF,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/phi4_qgrg_level2"),
    )
    parser.add_argument("--mass2", type=float, default=0.2)
    parser.add_argument("--quartic", type=float, default=0.8)
    parser.add_argument("--q-max", type=float, default=0.24)
    parser.add_argument("--shell", choices=("sharp", "smooth"), default="smooth")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shell = (
        Phi4GaussianShell(spatial_dimension=1)
        if args.shell == "sharp"
        else Phi4SmoothQGRF(quadrature_order=40)
    )
    couplings = Phi4GaussianCouplings(
        mass2=args.mass2, quartic=args.quartic
    )
    momentum = np.linspace(0.0, args.q_max, 121)
    fields = np.array([0.0, 0.2, 0.4, 0.6])
    responses = [
        shell.level2_response(field, couplings, momentum) for field in fields
    ]

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), constrained_layout=True)
    colors = ["#000000", "#0072B2", "#D55E00", "#009E73"]

    for field, response, color in zip(fields, responses, colors):
        label = rf"$\varphi={field:.1f}$"
        axes[0].plot(
            momentum, response["kernel_response"], color=color, label=label
        )
        difference = np.full_like(momentum, np.nan)
        difference[1:] = (
            response["two_point_rate"][1:] - response["two_point_rate"][0]
        ) / momentum[1:] ** 2
        axes[1].plot(momentum[1:], difference[1:], color=color, label=label)
        axes[1].axhline(
            response["spatial_rate"], color=color, linewidth=0.8, linestyle="--"
        )
        axes[2].plot(
            momentum, response["overlap_metric"], color=color, label=label
        )

    axes[0].set(
        xlabel=r"external momentum $q/\Lambda$",
        ylabel=r"kernel response $\delta\Omega/\delta\varphi$",
    )
    axes[1].set(
        xlabel=r"external momentum $q/\Lambda$",
        ylabel=r"$[\Pi(q)-\Pi(0)]/q^2$",
    )
    axes[2].set(
        xlabel=r"external momentum $q/\Lambda$",
        ylabel=r"overlap metric $\mathcal{G}(q)$",
    )
    axes[2].legend(frameon=False)
    for label, axis in zip("abc", axes):
        axis.text(
            0.02,
            0.97,
            label,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
        axis.grid(color="0.9", linewidth=0.5)

    stem = args.output_dir / f"phi4_qgrg_level2_{args.shell}"
    figure.savefig(stem.with_suffix(".png"))
    figure.savefig(stem.with_suffix(".pdf"))
    np.savez(
        stem.with_suffix(".npz"),
        momentum=momentum,
        fields=fields,
        kernel_response=np.array([item["kernel_response"] for item in responses]),
        two_point_rate=np.array([item["two_point_rate"] for item in responses]),
        overlap_metric=np.array([item["overlap_metric"] for item in responses]),
        spatial_rate=np.array([item["spatial_rate"] for item in responses]),
    )
    summary = {
        "model": f"1+1D Level-2 Gaussian QGRG ({args.shell} shell)",
        "mass2": couplings.mass2,
        "quartic": couplings.quartic,
        "fields": fields.tolist(),
        "spatial_rates": [item["spatial_rate"] for item in responses],
        "zero_momentum_metrics": [
            float(item["overlap_metric"][0]) for item in responses
        ],
    }
    with open(stem.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
