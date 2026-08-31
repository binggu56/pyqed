#!/usr/bin/env python3
"""Plot the validated 5D phenol propagation speed and accuracy changes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    data = np.load(
        directory / "phenol_sa_casscf_5d_ftt_ttldr.npz", allow_pickle=True
    )
    return summary, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reference, ref = load(args.reference)
    optimized, opt = load(args.optimized)
    x = np.asarray(ref["final_wavefunction"]).reshape(-1)
    y = np.asarray(opt["final_wavefunction"]).reshape(-1)
    overlap = np.vdot(x, y)
    y *= np.exp(-1j * np.angle(overlap))
    errors = {
        "wavefunction relative L2": float(np.linalg.norm(y - x) / np.linalg.norm(x)),
        "populations max abs": float(
            np.max(np.abs(opt["populations"] - ref["populations"]))
        ),
        "norm max abs": float(np.max(np.abs(opt["norms"] - ref["norms"]))),
        "CAP yields max abs": float(
            np.max(np.abs(opt["cap_yields"] - ref["cap_yields"]))
        ),
    }
    baseline_seconds = float(reference["timings_seconds"]["propagation"])
    optimized_seconds = float(optimized["timings_seconds"]["propagation"])
    speedup = baseline_seconds / optimized_seconds

    figure, panels = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)
    panels[0].bar(
        ("baseline\nKrylov 12", "optimized\nKrylov 8"),
        (baseline_seconds, optimized_seconds),
        color=("#9E9E9E", "#0072B2"),
    )
    panels[0].set(ylabel="three-step wall time (s)", title=f"{speedup:.2f}× faster")
    panels[0].grid(axis="y", alpha=0.2)

    panels[1].barh(
        tuple(errors), tuple(errors.values()), color="#009E73"
    )
    panels[1].set(xscale="log", xlabel="error", title="Numerical agreement")
    panels[1].grid(axis="x", alpha=0.2)

    radius = np.asarray(ref["axes"], dtype=object)[0]
    panels[2].plot(
        radius, ref["final_radial"], color="black", lw=1.8, label="rank 12"
    )
    panels[2].plot(
        radius,
        opt["final_radial"],
        "--",
        color="#D55E00",
        lw=1.4,
        label="optimized",
    )
    panels[2].set(
        xlabel=r"$R_{\mathrm{OH}}$ (Å)",
        ylabel="radial probability",
        title="Physical distribution",
    )
    panels[2].legend(frameon=False, fontsize=8)
    panels[2].grid(alpha=0.2)
    for label, panel in zip("abc", panels):
        panel.text(
            0.02, 0.97, label, transform=panel.transAxes,
            va="top", fontweight="bold",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    result = {
        "reference": str(args.reference),
        "optimized": str(args.optimized),
        "baseline_seconds": baseline_seconds,
        "optimized_seconds": optimized_seconds,
        "speedup": speedup,
        "errors": errors,
        "figure": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
