"""Six-point octahedral angular average of the CHD elastic and total UED."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("chd_c2_lvc_total_elastic_inelastic_oriented_pd_3mode.npz")
DYNAMICS = Path("chd_c2_lvc_coupled_wavepacket_3mode.npz")
OUTPUT = Path("chd_c2_lvc_isotropic_elastic_total_pd_3mode")


def main():
    with np.load(DATA) as data:
        times = data["times_fs"]
        s = data["s_angstrom_inverse"]
        labels = data["direction_labels"].astype(str)
        elastic_ground_directional = data["elastic_ground"]
        elastic_excited_directional = data["elastic_excited"]
        inelastic_ground_directional = data["inelastic_ground"]
        inelastic_excited_directional = data["inelastic_excited"]

    indices = np.asarray([np.where(labels == label)[0][0] for label in (
        "x_out_of_plane", "y_in_plane", "z_C2"
    )])
    elastic_ground = np.mean(elastic_ground_directional[indices], axis=0)
    elastic_excited = np.mean(elastic_excited_directional[:, indices], axis=1)
    inelastic_ground = np.mean(inelastic_ground_directional[indices], axis=0)
    inelastic_excited = np.mean(inelastic_excited_directional[:, indices], axis=1)
    total_ground = elastic_ground + inelastic_ground
    total_excited = elastic_excited + inelastic_excited
    pd_elastic = 100.0 * (elastic_excited - elastic_ground[None, :]) / elastic_ground[None, :]
    pd_total = 100.0 * (total_excited - total_ground[None, :]) / total_ground[None, :]
    inelastic_fraction_ground = inelastic_ground / total_ground
    inelastic_fraction_excited = inelastic_excited / total_excited

    with np.load(DYNAMICS) as dynamics:
        populations = dynamics["populations"]
        coherence = dynamics["coherence"]
        mean_x = np.sum(
            populations[:, :, None] * dynamics["conditional_mean_x"], axis=1
        )
    low_s = (s >= 0.4) & (s <= 1.6)
    low_s_trace = np.sum(total_excited[:, low_s] - total_ground[None, low_s], axis=1)
    predictors = {
        "Px": populations[:, 0],
        "abs_rho_xy": np.abs(coherence),
        "mean_q5": mean_x[:, 0],
        "mean_q8": mean_x[:, 1],
        "mean_q26": mean_x[:, 2],
    }
    correlations = {
        label: float(np.corrcoef(low_s_trace, values)[0, 1])
        for label, values in predictors.items()
    }

    np.savez_compressed(
        f"{OUTPUT}.npz", times_fs=times, s_angstrom_inverse=s,
        elastic_ground=elastic_ground, elastic_excited=elastic_excited,
        inelastic_ground=inelastic_ground, inelastic_excited=inelastic_excited,
        total_ground=total_ground, total_excited=total_excited,
        PD_elastic_percent=pd_elastic, PD_total_percent=pd_total,
        inelastic_fraction_ground=inelastic_fraction_ground,
        inelastic_fraction_excited=inelastic_fraction_excited,
        low_s_total_difference_trace=low_s_trace,
    )
    summary = {
        "angular_quadrature": "six-point octahedral average; inversion reduces it to +x,+y,+z",
        "exactness": "exact for spherical harmonics through l=3; hence exact for rank-0/rank-2 decomposition",
        "warning": "higher angular orders at large s are not convergence tested",
        "low_s_0.4_1.6_inverse_angstrom_correlations": correlations,
        "ground_inelastic_fraction_windows": {},
        "excited_time_averaged_inelastic_fraction_windows": {},
    }
    for lower, upper in ((0.4, 1.6), (1.7, 2.5), (2.5, 8.0), (0.4, 8.0)):
        mask = (s >= lower) & (s <= upper)
        key = f"{lower:.1f}-{upper:.1f}"
        summary["ground_inelastic_fraction_windows"][key] = float(
            np.sum(inelastic_ground[mask]) / np.sum(total_ground[mask])
        )
        summary["excited_time_averaged_inelastic_fraction_windows"][key] = float(
            np.sum(inelastic_excited[:, mask]) / np.sum(total_excited[:, mask])
        )
    Path(f"{OUTPUT}.json").write_text(json.dumps(summary, indent=2) + "\n")

    reliable = total_ground > 1.0e-3 * np.max(total_ground)
    displays = [
        np.where(reliable[None, :], pd_elastic, np.nan),
        np.where(reliable[None, :], pd_total, np.nan),
    ]
    limit = np.nanpercentile(np.abs(np.concatenate(displays, axis=1)), 98.5)
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), sharex=True, sharey=True)
    titles = ["Isotropic elastic PD", "Isotropic total PD"]
    for panel, (axis, values, title) in enumerate(zip(axes, displays, titles)):
        image = axis.pcolormesh(
            s, times, values, shading="auto", cmap="RdBu_r", vmin=-limit,
            vmax=limit, rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel(r"$s$ ($\mathrm{\AA}^{-1}$)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.13, 1.02, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Time delay (fs)")
    figure.subplots_adjust(left=0.10, right=0.86, bottom=0.13, top=0.90, wspace=0.15)
    colorbar_axis = figure.add_axes([0.89, 0.18, 0.025, 0.64])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("PD (%)")
    figure.savefig(f"{OUTPUT}.pdf")
    figure.savefig(f"{OUTPUT}.png", dpi=400)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
