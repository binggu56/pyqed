"""Plot elastic and inelastic fractions for the inclusive CHD UED model."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("chd_c2_lvc_total_elastic_inelastic_oriented_pd_3mode.npz")
OUTPUT = Path("chd_c2_lvc_elastic_inelastic_strength_3mode")
WINDOWS = [(0.4, 1.6), (1.7, 2.5), (2.5, 8.0), (0.4, 8.0)]


def integrated_strength(elastic, inelastic, mask):
    elastic_sum = float(np.sum(elastic[..., mask]))
    inelastic_sum = float(np.sum(inelastic[..., mask]))
    return {
        "I_inelastic_over_I_elastic": inelastic_sum / elastic_sum,
        "inelastic_fraction_of_total": inelastic_sum / (elastic_sum + inelastic_sum),
    }


def main():
    with np.load(DATA) as data:
        s = data["s_angstrom_inverse"]
        labels = data["direction_labels"].astype(str)
        elastic_ground = data["elastic_ground"]
        inelastic_ground = data["inelastic_ground"]
        elastic_excited = data["elastic_excited"]
        inelastic_excited = data["inelastic_excited"]

    ground_fraction = inelastic_ground / (elastic_ground + inelastic_ground)
    mean_elastic_excited = np.mean(elastic_excited, axis=0)
    mean_inelastic_excited = np.mean(inelastic_excited, axis=0)
    excited_fraction = mean_inelastic_excited / (
        mean_elastic_excited + mean_inelastic_excited
    )
    titles = {
        "x_out_of_plane": r"$\mathbf{q}\parallel x$ (out of plane)",
        "y_in_plane": r"$\mathbf{q}\parallel y$ (in plane)",
        "z_C2": r"$\mathbf{q}\parallel z$ ($C_2$ axis)",
        "xy_bisector": r"$\mathbf{q}\parallel(x+y)/\sqrt{2}$",
    }
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharex=True, sharey=True)
    for panel, (axis, label) in enumerate(zip(axes.flat, labels)):
        axis.plot(s, ground_fraction[panel], color="#0072B2", lw=1.3,
                  label="ground")
        axis.plot(s, excited_fraction[panel], color="#D55E00", lw=1.3,
                  ls="--", label="excited, time averaged")
        axis.set_title(titles[label], fontsize=10)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlim(s[0], s[-1])
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.13, 1.02, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")
    for axis in axes[1]:
        axis.set_xlabel(r"$s$ ($\mathrm{\AA}^{-1}$)")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$I_{\rm inelastic}/I_{\rm total}$")
    axes[0, 0].legend(frameon=False, loc="upper right")
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.95,
                           wspace=0.16, hspace=0.20)
    figure.savefig(f"{OUTPUT}.pdf")
    figure.savefig(f"{OUTPUT}.png", dpi=400)
    plt.close(figure)

    summary = {}
    for direction, label in enumerate(labels):
        summary[label] = {}
        for lower, upper in WINDOWS:
            mask = (s >= lower) & (s <= upper)
            key = f"{lower:.1f}-{upper:.1f}_angstrom^-1"
            summary[label][key] = {
                "ground": integrated_strength(
                    elastic_ground[direction], inelastic_ground[direction], mask
                ),
                "excited_time_integrated": integrated_strength(
                    elastic_excited[:, direction], inelastic_excited[:, direction], mask
                ),
            }
    Path(f"{OUTPUT}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
