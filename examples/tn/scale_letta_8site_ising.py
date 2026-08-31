"""Eight-site scale-LETTA validation at the critical Ising point."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.tn import EightSiteScaleLETTA


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/scale_letta_8site_ising.png"),
    )
    parser.add_argument("--maxiter", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    q1 = EightSiteScaleLETTA(q=1).fit_critical_ising(maxiter=args.maxiter)
    q2 = EightSiteScaleLETTA(q=2).fit_critical_ising(maxiter=args.maxiter)
    odd = q2.scaling_dimensions(sector="odd")["dimensions"]
    even = q2.scaling_dimensions(sector="even")["dimensions"]
    sigma = odd[0]
    epsilon = even[1:][np.argmin(np.abs(even[1:] - 1.0))]

    print(f"q=1 TTN energy:       {q1.energy:.12f}")
    print(f"q=2 scale-LETTA:      {q2.energy:.12f}")
    print(f"exact energy:         {q2.exact_energy:.12f}")
    print(f"q=2 energy error:     {q2.energy_error:.6e}")
    print(f"q=2 fidelity:         {q2.fidelity:.8f}")
    print(f"q=2 norm:             {q2.norm():.12f}")
    print(f"Delta_sigma:          {sigma:.8f} (CFT 0.125)")
    print(f"nearest Delta_epsilon:{epsilon: .8f} (CFT 1.0)")
    print("lowest odd spectrum: ", np.array2string(odd[:5], precision=6))
    print("lowest even spectrum:", np.array2string(even[:6], precision=6))

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    labels = ["q=1 TTN", "q=2 scale-LETTA", "exact"]
    energies = [q1.energy, q2.energy, q2.exact_energy]
    axes[0].bar(labels, energies, color=["0.65", "tab:blue", "tab:orange"])
    axes[0].set_ylabel("ground-state energy")
    axes[0].set_title("8-site periodic critical Ising chain")
    axes[0].tick_params(axis="x", rotation=18)

    extracted = [sigma, epsilon]
    targets = [1.0 / 8.0, 1.0]
    positions = np.arange(2)
    axes[1].bar(positions - 0.18, extracted, 0.36, label="scale-LETTA")
    axes[1].bar(positions + 0.18, targets, 0.36, label="Ising CFT")
    axes[1].set_xticks(positions, [r"$\Delta_\sigma$", r"$\Delta_\epsilon$"])
    axes[1].set_ylabel("scaling dimension")
    axes[1].set_title(r"Spectrum of $\mathcal{S}_*$")
    axes[1].legend()
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180)
    print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
