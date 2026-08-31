#!/usr/bin/env python3
"""Plot the pyrazine PyQED/block2 pilot and orbital-optimizer diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--optimized-json", type=Path, required=True)
    parser.add_argument("--block2-casci-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = _load(args.pilot_json)
    optimized = _load(args.optimized_json)
    block2_casci = _load(args.block2_casci_json)
    records = [
        pilot["pyqed"],
        pilot["block2"],
        optimized["pyqed"],
    ]
    labels = ["PyQED\n4 RCG steps", "block2\n4 micro steps", "PyQED\n50 RCG steps"]
    colors = ["#3266a8", "#d06b32", "#4b9b67"]
    correlations = [1000.0 * row["correlation_energy_hartree"] for row in records]
    timings = [row["timing_seconds"]["dmrgscf"] for row in records]
    fixed_orbital_difference = (
        pilot["pyqed"]["energy_history_hartree"][0]
        - block2_casci["block2"]["dmrgscf_energy_hartree"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    axes[0].bar(labels, correlations, color=colors)
    axes[0].set(
        ylabel=r"$E-E_{\mathrm{RHF}}$ (m$E_h$)",
        title="Energy after one orbital macroiteration",
    )
    axes[1].bar(labels, timings, color=colors)
    axes[1].set(ylabel="wall time / s", title="DMRG-SCF wall time")
    for index, value in enumerate(timings):
        axes[1].text(index, value, f"{value:.1f}", ha="center", va="bottom")
    for axis in axes:
        axis.tick_params(axis="x", rotation=10)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Pyrazine/aug-cc-pVDZ CAS(10,10), D=32 | "
        f"fixed-orbital DMRG difference {fixed_orbital_difference:.1e} $E_h$"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
