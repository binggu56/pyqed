#!/usr/bin/env python3
"""Plot H3+ independent MACE acceptance metrics from a JSON report."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = json.loads(args.report.read_text())

ratios = np.asarray(
    (
        report["maximum_hamiltonian_error_hartree"]
        / report["hamiltonian_atol_hartree"],
        report["rms_hamiltonian_error_hartree"]
        / report["hamiltonian_rms_hartree"],
        report["relative_link_error"] / report["link_rtol"],
    )
)
coverage = np.asarray(
    (report["hamiltonian_coverage"], report["link_coverage"])
)
colors = ("#0072B2", "#D55E00", "#009E73")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
figure, panels = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
panels[0].bar(
    np.arange(3), ratios, color=colors, edgecolor="black", linewidth=0.7,
    hatch=("", "//", "xx"),
)
panels[0].axhline(1.0, color="0.2", linestyle="--", linewidth=1.2, label="gate")
panels[0].set_yscale("log")
panels[0].set_xticks(np.arange(3), (r"max $\Delta\bar H$", r"RMS $\Delta\bar H$", "links"))
panels[0].set(
    ylabel="measured error / gate",
    title="a  Independent errors",
    ylim=(0.5, 40.0),
)
panels[0].legend(frameon=False)

panels[1].bar(
    np.arange(2), coverage, color=colors[:2], edgecolor="black", linewidth=0.7,
    hatch=("", "//"),
)
panels[1].axhline(
    report["coverage_gate"], color="0.2", linestyle="--", linewidth=1.2,
    label="gate",
)
panels[1].set_xticks(np.arange(2), ("Hamiltonian", "links"))
panels[1].set(
    ylabel="empirical coverage",
    title="b  Calibrated uncertainty",
    ylim=(0.80, 1.00),
)
panels[1].legend(frameon=False)
for panel in panels:
    panel.spines[["top", "right"]].set_visible(False)
    panel.tick_params(direction="out")

args.output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(args.output.with_suffix(".pdf"))
figure.savefig(args.output.with_suffix(".png"), dpi=360)
plt.close(figure)
