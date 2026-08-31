#!/usr/bin/env python3
"""Plot checkpointed PyQED and block2 pyrazine DMRG-SCF convergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_many(paths, backend):
    rows = []
    total_time = 0.0
    final = None
    for path in paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))[backend]
        history = (
            record["macro_diagnostics"]
            if backend == "pyqed"
            else record["macro_history"]
        )
        for item in history:
            rows.append(
                (
                    float(item["energy"]),
                    float(item.get("gn", item.get("gradient_norm", np.nan))),
                )
            )
        total_time += float(record["timing_seconds"]["dmrgscf"])
        final = record
    return np.asarray(rows, dtype=float), total_time, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyqed-json", nargs="+", type=Path, required=True)
    parser.add_argument("--block2-json", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pyqed, pyqed_time, pyqed_final = _load_many(args.pyqed_json, "pyqed")
    block2, block2_time, block2_final = _load_many(args.block2_json, "block2")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    colors = {"PyQED": "#3266a8", "block2": "#d06b32"}
    for label, data in (("PyQED", pyqed), ("block2", block2)):
        cycles = np.arange(1, len(data) + 1)
        error = 1000.0 * (data[:, 0] - data[-1, 0])
        axes[0].plot(cycles, error, "o-", ms=3.0, lw=1.2, color=colors[label], label=label)
        finite = np.isfinite(data[:, 1]) & (data[:, 1] > 0.0)
        axes[1].semilogy(
            cycles[finite],
            data[finite, 1],
            "o-",
            ms=3.0,
            lw=1.2,
            color=colors[label],
            label=label,
        )

    axes[0].axhline(0.0, color="0.45", lw=0.8)
    axes[0].set(
        xlabel="checkpointed macro iteration",
        ylabel=r"$E-E_{\mathrm{final}}$ (m$E_h$)",
        title="Energy convergence to each stationary point",
    )
    axes[1].axhline(1.0e-4, color="0.35", ls="--", lw=0.9, label=r"strict $10^{-4}$")
    axes[1].axhline(5.0e-4, color="0.55", ls=":", lw=0.9, label=r"relaxed $5\times10^{-4}$")
    axes[1].set(
        xlabel="checkpointed macro iteration",
        ylabel="nonredundant orbital-gradient norm",
        title="Orbital stationarity",
    )
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Pyrazine/aug-cc-pVDZ CAS(10,10), D=32 | "
        f"PyQED {pyqed_time:.0f} s, block2 {block2_time:.0f} s\n"
        rf"$E_{{\mathrm{{PyQED}}}}={pyqed_final['dmrgscf_energy_hartree']:.9f}$ $E_h$, "
        rf"$E_{{\mathrm{{block2}}}}={block2_final['dmrgscf_energy_hartree']:.9f}$ $E_h$"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
