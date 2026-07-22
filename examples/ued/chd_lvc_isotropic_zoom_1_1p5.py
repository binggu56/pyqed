"""Zoom the isotropic CHD UED signal over 1.0--1.8 inverse angstrom."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("chd_c2_lvc_isotropic_elastic_total_pd_3mode.npz")
OUTPUT_MAP = Path("chd_c2_lvc_isotropic_pd_zoom_s1_1p8")
OUTPUT_MODULATION = Path("chd_c2_lvc_isotropic_pd_modulation_zoom_s1_1p8")
OUTPUT_TRACES = Path("chd_c2_lvc_isotropic_pd_traces_s1_1p8")


def main():
    with np.load(DATA) as data:
        times = data["times_fs"]
        s = data["s_angstrom_inverse"]
        elastic = data["PD_elastic_percent"]
        total = data["PD_total_percent"]

    mask = (s >= 1.0) & (s <= 1.8)
    s_zoom = s[mask]
    values = [elastic[:, mask], total[:, mask]]
    figure, axes = plt.subplots(
        1, 2, figsize=(8.4, 4.3), sharex=True, sharey=True,
        layout="constrained",
    )
    for panel, (axis, signal, title) in enumerate(zip(
        axes, values, ("Isotropic elastic PD", "Isotropic total PD")
    )):
        vmin, vmax = np.percentile(signal, [0.5, 99.5])
        image = axis.pcolormesh(
            s_zoom, times, signal, shading="auto", cmap="Spectral_r",
            vmin=vmin, vmax=vmax, rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel(r"$s$ ($\mathrm{\AA}^{-1}$)")
        axis.set_xlim(1.0, 1.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.02, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")
        colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.050)
        colorbar.set_label("PD (%)")
    axes[0].set_ylabel("Time delay (fs)")
    figure.savefig(OUTPUT_MAP.with_suffix(".pdf"))
    figure.savefig(OUTPUT_MAP.with_suffix(".png"), dpi=400)
    plt.close(figure)

    modulations = [signal - np.mean(signal, axis=0, keepdims=True) for signal in values]
    figure, axes = plt.subplots(
        1, 2, figsize=(8.4, 4.3), sharex=True, sharey=True,
        layout="constrained",
    )
    for panel, (axis, signal, title) in enumerate(zip(
        axes, modulations,
        ("Elastic oscillatory component", "Total oscillatory component"),
    )):
        limit = np.percentile(np.abs(signal), 99.0)
        image = axis.pcolormesh(
            s_zoom, times, signal, shading="auto", cmap="Spectral_r",
            vmin=-limit, vmax=limit, rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel(r"$s$ ($\mathrm{\AA}^{-1}$)")
        axis.set_xlim(1.0, 1.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.02, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")
        colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.050)
        colorbar.set_label(r"PD$-\langle$PD$\rangle_t$ (%)")
    axes[0].set_ylabel("Time delay (fs)")
    figure.savefig(OUTPUT_MODULATION.with_suffix(".pdf"))
    figure.savefig(OUTPUT_MODULATION.with_suffix(".png"), dpi=400)
    plt.close(figure)

    targets = [1.0, 1.8]
    indices = [int(np.argmin(np.abs(s - target))) for target in targets]
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    for panel, (axis, target, index) in enumerate(zip(axes, targets, indices)):
        axis.plot(times, elastic[:, index], color="#0072B2", lw=1.2,
                  label="elastic")
        axis.plot(times, total[:, index], color="#D55E00", lw=1.2, ls="--",
                  label="elastic + inelastic")
        axis.set_ylabel("PD (%)")
        axis.set_title(rf"$s={s[index]:.3f}\ \mathrm{{\AA}}^{{-1}}$", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.10, 1.02, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Time delay (fs)")
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.96, hspace=0.30)
    figure.savefig(OUTPUT_TRACES.with_suffix(".pdf"))
    figure.savefig(OUTPUT_TRACES.with_suffix(".png"), dpi=400)
    plt.close(figure)


if __name__ == "__main__":
    main()
