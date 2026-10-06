"""Plot saved fixed-orbital Fe(CO)5 SU(2) sweep diagnostics."""

import argparse
import os
from pathlib import Path
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/feco5-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orbitals-label", default="Fixed RHF orbitals")
    parser.add_argument("--cutoff", type=float, default=1e-5)
    args = parser.parse_args()
    with args.history.open("rb") as handle:
        history = pickle.load(handle)
    energy = np.asarray([row["energy"] for row in history])
    residual = np.asarray([row["metric"] for row in history])
    sweep = np.arange(1, len(history)+1) / 2
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    axes[0].plot(sweep, (energy-energy[-1])*1000, "o-")
    axes[0].set(xlabel="Complete sweeps", ylabel="Energy relative to last sweep (mHartree)")
    axes[1].semilogy(sweep, residual, "o-")
    axes[1].axhline(1e-7, color="gray", linestyle="--", label="Sweep tolerance")
    axes[1].set(xlabel="Complete sweeps", ylabel="Maximum local residual (Hartree)")
    axes[1].legend()
    fig.suptitle("Fe(CO)₅ · def2-SVP · CAS(12,12)\n"
                 f"{args.orbitals_label} · D=100/sector · Schmidt cutoff {args.cutoff:g}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
