#!/usr/bin/env python3
"""Plot N=7 targeted-MPS accuracy and cost versus dynamics bond dimension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = (
    HERE / "results" / "channel_targeted_mv_ms_mps_n7"
    / "channel_targeted_mv_ms_data.json",
    HERE / "results" / "channel_targeted_mv_ms_mps_n7_d96"
    / "channel_targeted_mv_ms_data.json",
)
DEFAULT_OUTPUT = (
    HERE / "results" / "channel_targeted_mv_ms_mps_n7_d96"
    / "16_n7_mps_bond_convergence.png"
)


def plot(inputs, output):
    records = [json.loads(Path(path).read_text()) for path in inputs]
    records.sort(key=lambda record: record["dynamics_bond_dim"])
    bond = np.asarray([record["dynamics_bond_dim"] for record in records])
    vector_error = np.asarray(
        [abs(record["vector_mass_mps"] - record["vector_mass_ed"]) for record in records]
    )
    scalar_error = np.asarray(
        [abs(record["scalar_mass_mps"] - record["scalar_mass_ed"]) for record in records]
    )
    vector_time = np.asarray(
        [record["timing_seconds"]["vector_correlation"] for record in records]
    )
    scalar_time = np.asarray(
        [record["timing_seconds"]["scalar_correlation"] for record in records]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    axes[0].semilogy(bond, vector_error, "o-", label=r"$M_V$")
    axes[0].semilogy(bond, scalar_error, "s-", label=r"$M_S$")
    axes[0].set(
        xlabel="dynamics bond dimension",
        ylabel=r"$|M_{\rm MPS}-M_{\rm ED}|/g$",
        title=r"$N=7$ targeted-MPS convergence",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(bond, vector_time, "o-", label="vector correlation")
    axes[1].plot(bond, scalar_time, "s-", label="scalar correlation")
    axes[1].set(
        xlabel="dynamics bond dimension",
        ylabel="wall time (s)",
        title="Accuracy-cost tradeoff",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(bond)
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(direction="in")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.inputs, args.output)


if __name__ == "__main__":
    main()
