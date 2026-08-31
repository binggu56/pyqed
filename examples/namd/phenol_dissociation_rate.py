#!/usr/bin/env python3
"""Estimate early-time phenol dissociation rates from saved TTLDR fluxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "physical three-state": "#009E73",
    "GP on, projected S1": "#0072B2",
    "GP off, projected S1": "#D55E00",
}


def fit_exponential_rate(times_fs, cumulative_yield, start_fs, stop_fs):
    """Fit ``S(t) = A exp(-k t)`` over one time window."""
    times_fs = np.asarray(times_fs, dtype=float)
    cumulative_yield = np.asarray(cumulative_yield, dtype=float)
    mask = (times_fs >= float(start_fs)) & (times_fs <= float(stop_fs))
    if np.count_nonzero(mask) < 3:
        raise ValueError("the rate-fit window requires at least three samples")
    time = times_fs[mask]
    survival = 1.0 - cumulative_yield[mask]
    if np.any(survival <= 0.0):
        raise ValueError("cumulative yield must remain below one")
    log_decay = -np.log(survival)
    slope, intercept = np.polyfit(time, log_decay, 1)
    fitted = slope * time + intercept
    residual = log_decay - fitted
    variance = np.sum((log_decay - np.mean(log_decay)) ** 2)
    r_squared = 1.0 - np.sum(residual**2) / max(variance, np.finfo(float).tiny)
    k_fs = float(slope)
    return {
        "start_fs": float(start_fs),
        "stop_fs": float(stop_fs),
        "rate_per_fs": k_fs,
        "rate_per_ps": 1000.0 * k_fs,
        "rate_per_second": 1.0e15 * k_fs,
        "lifetime_ps": float(1.0 / (1000.0 * k_fs)),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def load_curves(physical_summary, control_data):
    with Path(physical_summary).open() as handle:
        physical = json.load(handle)["dynamics"][0]
    physical_yield = np.sum(np.asarray(physical["cap_yields"], dtype=float), axis=1)
    control = np.load(control_data)
    return {
        "physical three-state": (
            np.asarray(physical["times_fs"], dtype=float),
            physical_yield,
        ),
        "GP on, projected S1": (
            np.asarray(control["times_fs_gp_on"], dtype=float),
            np.asarray(control["cap_yield_gp_on"], dtype=float),
        ),
        "GP off, projected S1": (
            np.asarray(control["times_fs_gp_off"], dtype=float),
            np.asarray(control["cap_yield_gp_off"], dtype=float),
        ),
    }


def plot_rate_analysis(output, curves, primary_fits, window_fits):
    figure, panels = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    for label, (times, cumulative_yield) in curves.items():
        color = COLORS[label]
        panels[0].plot(times, 100.0 * cumulative_yield, color=color, label=label)
        log_decay = -np.log1p(-cumulative_yield)
        panels[1].plot(times, log_decay, color=color)
        fit = primary_fits[label]
        mask = times >= fit["start_fs"]
        fitted = fit["rate_per_fs"] * times[mask] + fit["intercept"]
        panels[1].plot(times[mask], fitted, "--", color=color, lw=1.4)
        starts = np.asarray([item["start_fs"] for item in window_fits[label]])
        rates = np.asarray([item["rate_per_ps"] for item in window_fits[label]])
        panels[2].plot(starts, rates, "o-", color=color, ms=4, label=label)

    panels[0].set(
        xlabel="time (fs)",
        ylabel="integrated CAP flux (%)",
        title="Dissociation yield",
    )
    panels[1].set(
        xlabel="time (fs)",
        ylabel=r"$-\ln[1-Y(t)]$",
        title="Exponential fit, 50–200 fs",
    )
    panels[2].set(
        xlabel="fit-window start (fs)",
        ylabel=r"$k$ (ps$^{-1}$)",
        title="Window sensitivity (end = 200 fs)",
    )
    panels[0].legend(frameon=False, fontsize=8)
    panels[2].legend(frameon=False, fontsize=8)
    for label, panel in zip("abc", panels):
        panel.text(
            0.02,
            0.96,
            label,
            transform=panel.transAxes,
            va="top",
            fontweight="bold",
        )
        panel.grid(alpha=0.18)
    png = output / "phenol_dissociation_rate.png"
    pdf = output / "phenol_dissociation_rate.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-summary",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_3d_ftt_gp_on_200fs_20260822/summary.json"
        ),
    )
    parser.add_argument(
        "--control-data",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_3d_gp_control_200fs_20260822/"
            "phenol_gp_on_off_200fs.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_dissociation_rate_20260822"),
    )
    parser.add_argument("--fit-start-fs", type=float, default=50.0)
    parser.add_argument("--fit-stop-fs", type=float, default=200.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    curves = load_curves(args.physical_summary, args.control_data)
    primary_fits = {
        label: fit_exponential_rate(times, values, args.fit_start_fs, args.fit_stop_fs)
        for label, (times, values) in curves.items()
    }
    window_starts = (30.0, 50.0, 75.0, 100.0, 125.0, 150.0)
    window_fits = {
        label: [
            fit_exponential_rate(times, values, start, args.fit_stop_fs)
            for start in window_starts
        ]
        for label, (times, values) in curves.items()
    }
    png, pdf = plot_rate_analysis(args.output, curves, primary_fits, window_fits)
    summary = {
        "interpretation": "early-time effective first-order rates, not asymptotic constants",
        "survival_definition": "S(t) = 1 - integrated CAP flux",
        "primary_fits": primary_fits,
        "window_fits": window_fits,
        "figure": str(png),
        "figure_pdf": str(pdf),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
