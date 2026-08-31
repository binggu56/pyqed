#!/usr/bin/env python3
"""Compare the phenol 5D split-CAP trajectory with its no-CAP reference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#0072B2", "#D55E00", "#009E73")


def run(cap_path, reference_path, output):
    cap = np.load(cap_path, allow_pickle=True)
    reference = np.load(reference_path, allow_pickle=True)
    time = np.asarray(cap["times_fs"], dtype=float)
    reference_time = np.asarray(reference["times_fs"], dtype=float)
    radial_axis = np.asarray(tuple(cap["axes"])[0], dtype=float)
    reference_axis = np.asarray(tuple(reference["axes"])[0], dtype=float)
    if not np.array_equal(time, reference_time):
        raise ValueError("CAP and reference trajectories use different time grids")
    if not np.array_equal(radial_axis, reference_axis):
        raise ValueError("CAP and reference trajectories use different radial grids")

    yields = np.asarray(cap["cap_yields"], dtype=float)
    absorbed = np.asarray(cap["absorbed_probabilities"], dtype=float)
    cap_populations = np.asarray(cap["populations"], dtype=float)
    reference_populations = np.asarray(reference["populations"], dtype=float)
    cap_radial = np.asarray(cap["final_radial_absolute"], dtype=float)
    reference_radial = np.asarray(reference["final_radial"], dtype=float)
    initial_radial = np.asarray(cap["initial_radial"], dtype=float)
    cap_profile = np.asarray(cap["cap_profile"], dtype=float)
    active = np.flatnonzero(cap_profile > 0.0)
    cap_start = radial_axis[active[0]] if len(active) else radial_axis[-1]

    figure, panels = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    panels[0].plot(time, absorbed, color="black", lw=2.1, label="total absorbed")
    for channel, color in enumerate(COLORS):
        panels[0].plot(time, yields[:, channel], color=color, lw=1.5, label=fr"$Y_{channel}$")
    panels[0].set(
        xlabel="time (fs)",
        ylabel="probability",
        title="Outgoing CAP flux",
        xlim=(time[0], time[-1]),
    )
    panels[0].legend(frameon=False, fontsize=8)

    for channel, color in enumerate(COLORS):
        panels[1].plot(
            time,
            cap_populations[:, channel],
            color=color,
            lw=1.8,
            label=fr"CAP $P_{channel}$",
        )
        panels[1].plot(
            time,
            reference_populations[:, channel],
            color=color,
            lw=1.0,
            ls="--",
            alpha=0.75,
        )
    panels[1].plot(
        [], [], color="0.35", ls="--", lw=1.0, label="no CAP reference"
    )
    panels[1].set(
        xlabel="time (fs)",
        ylabel="P-gauge population",
        title="Surviving population",
        xlim=(time[0], time[-1]),
        ylim=(-0.02, 1.02),
    )
    panels[1].legend(frameon=False, fontsize=7, ncol=1)

    panels[2].plot(
        radial_axis,
        np.maximum(initial_radial, 1.0e-8),
        color="0.45",
        ls=":",
        lw=1.5,
        label="initial",
    )
    panels[2].plot(
        radial_axis,
        np.maximum(reference_radial, 1.0e-8),
        color="0.25",
        ls="--",
        lw=1.6,
        label="50 fs, no CAP",
    )
    panels[2].plot(
        radial_axis,
        np.maximum(cap_radial, 1.0e-8),
        color=COLORS[0],
        lw=1.8,
        marker="o",
        ms=3.0,
        label="50 fs, surviving",
    )
    panels[2].axvspan(cap_start, radial_axis[-1], color=COLORS[1], alpha=0.10)
    panels[2].set(
        xlabel=r"$R_{OH}$ (angstrom)",
        ylabel="absolute radial probability",
        title="Absorber removes the outgoing tail",
        xlim=(radial_axis[0], radial_axis[-1]),
        ylim=(1.0e-7, 0.5),
        yscale="log",
    )
    panels[2].legend(frameon=False, fontsize=8)

    for label, panel in zip("abc", panels):
        panel.text(
            0.02,
            0.97,
            label,
            transform=panel.transAxes,
            va="top",
            fontweight="bold",
        )
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(alpha=0.16)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=280)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cap",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa_casscf_5d_inward_r49_50fs_cap_tdvp1_20260823/"
            "phenol_sa_casscf_5d_ftt_ttldr.npz"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa_casscf_5d_inward_r49_50fs_tdvp1_20260823/"
            "phenol_sa_casscf_5d_ftt_ttldr.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa_casscf_5d_inward_r49_50fs_cap_tdvp1_20260823/"
            "phenol_5d_cap_vs_no_cap.png"
        ),
    )
    args = parser.parse_args()
    path = run(args.cap, args.reference, args.output)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
