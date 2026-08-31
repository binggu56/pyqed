#!/usr/bin/env python3
"""Compare angular-grid convergence of the H+ + H2 APH dynamics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/private/tmp")
CASES = (
    ("$N_\\theta=9$, $N_\\phi=18$", ROOT / "hplus_h2_aph_scattering"),
    ("$N_\\theta=11$, $N_\\phi=24$", ROOT / "hplus_h2_aph_scattering_fine"),
)
OUTPUT = ROOT / "hplus_h2_aph_scattering" / "hplus_h2_aph_convergence"


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 5.7), sharex=True, constrained_layout=True)
    colors = ("#0072B2", "#D55E00")
    final = []
    for (label, directory), color in zip(CASES, colors):
        data = np.load(directory / "hplus_h2_aph_scattering.npz")
        time = data["times_fs"]
        population = data["populations"]
        product = population[:, 0] + population[:, 1]
        axes[0, 0].plot(time, product, color=color, lw=1.5, label=label)
        axes[0, 1].plot(time, population[:, 2], color=color, lw=1.5)
        axes[1, 0].plot(time, population[:, 3], color=color, lw=1.5)
        axes[1, 1].plot(time, data["norms"], color=color, lw=1.5)
        final.append((label, product[-1], population[-1, 3], data["norms"][-1]))

    axes[0, 0].set(title="exchanged arrangements", ylabel="population")
    axes[0, 1].set(title="reactant arrangement", ylabel="population")
    axes[1, 0].set(title="interaction complex", xlabel="time / fs", ylabel="population")
    axes[1, 1].set(title="surviving norm", xlabel="time / fs", ylabel=r"$\langle\Psi|\Psi\rangle$")
    axes[0, 0].legend(frameon=False, fontsize=9)
    for label, axis in zip("abcd", axes.flat):
        axis.text(0.02, 0.95, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=360)
    figure.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(figure)
    for values in final:
        print(values)
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
