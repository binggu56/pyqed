#!/usr/bin/env python3
"""Plot long-time thermofield-SBM dynamics and cost diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.input)
    prefix = f"case{args.case}_"
    times = data["time"]
    sigma_z = data[prefix + "sigma_z"]
    fock_edge = data[prefix + "fock_edge_population"]
    occupation = data[prefix + "max_occupation"]
    norm_error = np.abs(data[prefix + "norm"] - 1.0)
    energy = data[prefix + "energy"]
    energy_error = np.abs(energy - energy[0])
    step_seconds = data[prefix + "step_seconds"]

    fig, axes = plt.subplots(4, 1, figsize=(7.3, 9.7), sharex=True)
    axes[0].plot(times, sigma_z, color="#332288", lw=1.8)
    axes[0].set_ylabel(r"$\langle\sigma_z\rangle$")
    axes[0].set_title(
        r"High-$T$ SBM: $T=1.5$, $N=14$, $d=24$, $D=32$, $\Delta t=0.1$"
    )

    axes[1].semilogy(
        times, np.maximum(fock_edge, 1.0e-16), color="#117733", lw=1.7
    )
    axes[1].set_ylabel("Fock-edge probability", color="#117733")
    occupation_axis = axes[1].twinx()
    occupation_axis.plot(times, occupation, color="#CC6677", lw=1.4, ls="--")
    occupation_axis.set_ylabel("maximum local occupation", color="#CC6677")

    axes[2].semilogy(
        times, np.maximum(norm_error, 1.0e-16), color="#4477AA", lw=1.6,
        label="norm error",
    )
    axes[2].semilogy(
        times, np.maximum(energy_error, 1.0e-16), color="#EE6677", lw=1.5,
        ls="--", label="energy drift",
    )
    axes[2].set_ylabel("conservation error")
    axes[2].legend(frameon=False)

    axes[3].plot(times[1:], step_seconds[1:], color="#AA4499", lw=1.6)
    axes[3].set_ylabel("TDVP seconds / step")
    axes[3].set_xlabel(r"time $t\,\omega_c$")
    for axis in axes:
        axis.grid(alpha=0.2)
    occupation_axis.grid(False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
