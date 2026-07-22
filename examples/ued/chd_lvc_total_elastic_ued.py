"""Build and plot the total elastic CHD UED signal from the three-mode model."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("chd_c2_lvc_level2_coherent_oriented_pd_3mode.npz")
OUTPUT = Path("chd_c2_lvc_total_elastic_ued_3mode")
EXCITATION_FRACTION = 0.06


def main():
    with np.load(INPUT) as data:
        times = data["times_fs"]
        s = data["s_angstrom_inverse"]
        labels = data["direction_labels"].astype(str)
        directions = data["directions_xyz"]
        pd_percent = data["PD_percent"]
        ground_charge_intensity = data["ground_intensity"]

    excited_charge_intensity = ground_charge_intensity[None, :, :] * (
        1.0 + pd_percent / 100.0
    )
    pump_on_charge_intensity = (
        (1.0 - EXCITATION_FRACTION) * ground_charge_intensity[None, :, :]
        + EXCITATION_FRACTION * excited_charge_intensity
    )

    # In the first-Born electron-scattering amplitude the electrostatic
    # potential supplies 1/q^2, hence the differential intensity has a q^-4
    # envelope. Constants common to all curves are omitted.
    s4 = s[None, None, :] ** 4
    ground_reduced_dcs = ground_charge_intensity / s[None, :] ** 4
    excited_reduced_dcs = excited_charge_intensity / s4
    pump_on_reduced_dcs = pump_on_charge_intensity / s4
    difference_reduced_dcs = pump_on_reduced_dcs - ground_reduced_dcs[None, :, :]

    np.savez_compressed(
        f"{OUTPUT}.npz",
        times_fs=times,
        s_angstrom_inverse=s,
        direction_labels=labels,
        directions_xyz=directions,
        excitation_fraction=EXCITATION_FRACTION,
        ground_charge_intensity=ground_charge_intensity,
        excited_charge_intensity=excited_charge_intensity,
        pump_on_charge_intensity=pump_on_charge_intensity,
        ground_reduced_dcs=ground_reduced_dcs,
        excited_reduced_dcs=excited_reduced_dcs,
        pump_on_reduced_dcs=pump_on_reduced_dcs,
        difference_reduced_dcs=difference_reduced_dcs,
    )
    summary = {
        "method": "elastic first-Born charge-density UED, arbitrary common scale",
        "excitation_fraction": EXCITATION_FRACTION,
        "states": ["ground", "paper-frame 3px", "paper-frame 3py"],
        "vibrational_modes": [5, 8, 26],
        "included": [
            "point-nuclear charge amplitude",
            "SA-CASSCF diagonal and 3px/3py transition densities",
            "three-mode LVC populations, coherence, and conditional nuclear motion",
            "ground/excited ensemble mixture",
            "s^-4 electron-diffraction envelope",
        ],
        "not_included": [
            "electronic or vibrational inelastic scattering",
            "rotational averaging and pump photoselection",
            "detector response, lifetime, and temporal convolution",
            "absolute instrument normalization",
        ],
    }
    Path(f"{OUTPUT}.json").write_text(json.dumps(summary, indent=2) + "\n")

    panel_titles = {
        "x_out_of_plane": r"$\mathbf{q}\parallel x$ (out of plane)",
        "y_in_plane": r"$\mathbf{q}\parallel y$ (in plane)",
        "z_C2": r"$\mathbf{q}\parallel z$ ($C_2$ axis)",
        "xy_bisector": r"$\mathbf{q}\parallel(x+y)/\sqrt{2}$",
    }
    normalized = pump_on_reduced_dcs / np.max(
        pump_on_reduced_dcs, axis=(0, 2), keepdims=True
    )
    log_signal = np.log10(np.maximum(normalized, 1.0e-8))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7), sharex=True, sharey=True)
    for panel, (axis, label) in enumerate(zip(axes.flat, labels)):
        image = axis.pcolormesh(
            s, times, log_signal[:, panel], shading="auto", cmap="viridis",
            vmin=-6.0, vmax=0.0, rasterized=True,
        )
        axis.set_title(panel_titles[label], fontsize=10)
        axis.text(-0.13, 1.02, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel(r"$s$ ($\mathrm{\AA}^{-1}$)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Time delay (fs)")
    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.09, top=0.94,
                        wspace=0.16, hspace=0.18)
    colorbar_axis = fig.add_axes([0.88, 0.16, 0.025, 0.70])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"$\log_{10}[I_{\rm on}/I_{\rm on}^{\max}]$")
    fig.savefig(f"{OUTPUT}.pdf")
    fig.savefig(f"{OUTPUT}.png", dpi=400)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
