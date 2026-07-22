"""Plot the saved final nuclear probability densities of the CHD LVC model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("chd_c2_lvc_coupled_wavepacket.npz")
OUTPUT = Path("chd_c2_lvc_nuclear_distribution_final")


def main():
    with np.load(DATA) as data:
        grid = data["dimensionless_grid"]
        psi = data["final_wavefunction"]
        final_time = float(data["times_fs"][-1])
        populations = data["populations"][-1]

    state_density = np.abs(psi) ** 2
    total_density = state_density.sum(axis=0)
    panels = [state_density[0], state_density[1], total_density]
    titles = [
        rf"$3p_x$ ($P_x={populations[0]:.3f}$)",
        rf"$3p_y$ ($P_y={populations[1]:.3f}$)",
        r"Total $|Psi_x|^2+|Psi_y|^2$",
    ]
    vmax = max(float(panel.max()) for panel in panels)
    extent = [grid[0], grid[-1], grid[0], grid[-1]]

    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), sharex=True, sharey=True)
    for index, (axis, density, title) in enumerate(zip(axes, panels, titles)):
        image = axis.imshow(
            density.T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        levels = np.linspace(0.15 * vmax, 0.9 * vmax, 6)
        axis.contour(grid, grid, density.T, levels=levels, colors="white", linewidths=0.45, alpha=0.7)
        axis.set_title(title, pad=5)
        axis.set_xlabel(r"Mode 5 coordinate $q_5$")
        axis.text(-0.16, 1.03, chr(ord("a") + index), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")
    axes[0].set_ylabel(r"Mode 8 coordinate $q_8$")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Probability density (arb. units)")
    fig.suptitle(rf"CHD two-mode LVC nuclear distribution at $t={final_time:.0f}$ fs", y=1.01)
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.18, top=0.82, wspace=0.10)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
