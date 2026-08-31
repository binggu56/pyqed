#!/usr/bin/env python3
"""Plot checkpointed constrained-orbital pyrazine DMRG-SCF convergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    segment_ends = []
    total_time = 0.0
    for segment, path in enumerate(args.json, start=1):
        record = json.loads(path.read_text(encoding="utf-8"))["pyqed"]
        total_time += float(record["timing_seconds"]["dmrgscf"])
        for row in record["macro_diagnostics"]:
            if not row.get("accepted", True) or not row.get("solver", False):
                continue
            rows.append(
                {
                    "segment": segment,
                    "energy": float(row["energy"]),
                    "gradient": float(row["gn"]),
                    "rebuilt": bool(row.get("su2_runtime_rebuilt", False)),
                    "retried": bool(row.get("solver_retried", False)),
                }
            )
        segment_ends.append(len(rows))

    if not rows:
        raise ValueError("No accepted, solver-converged CO macroiterations found.")

    cycles = np.arange(1, len(rows) + 1)
    energies = np.asarray([row["energy"] for row in rows])
    gradients = np.asarray([row["gradient"] for row in rows])
    final_energy = float(np.min(energies))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    axes[0].plot(
        cycles,
        1000.0 * (energies - final_energy),
        "o-",
        color="#3266a8",
        ms=4,
    )
    axes[0].axhline(0.0, color="0.5", lw=0.8)
    axes[0].set(
        xlabel="accepted CO macroiteration",
        ylabel=r"$E-E_{\mathrm{lowest}}$ (m$E_h$)",
        title="Finite-$D$ energy convergence",
    )

    axes[1].semilogy(cycles, gradients, "s-", color="#d06b32", ms=4)
    axes[1].axhline(1.0e-4, color="0.35", ls="--", lw=0.9, label=r"$10^{-4}$ target")
    axes[1].set(
        xlabel="accepted CO macroiteration",
        ylabel="CO orbital-gradient norm",
        title="Orbital stationarity",
    )
    axes[1].legend(fontsize=8)

    for boundary in segment_ends[:-1]:
        for axis in axes:
            axis.axvline(boundary + 0.5, color="0.7", ls=":", lw=0.8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle(
        "Pyrazine/aug-cc-pVDZ CO-DMRG-SCF CAS(10,10), D=32 | "
        f"{total_time:.0f} s checkpointed DMRG-SCF\n"
        rf"lowest $E={final_energy:.10f}$ $E_h$"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
