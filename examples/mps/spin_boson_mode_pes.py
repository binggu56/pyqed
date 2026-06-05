"""Plot adiabatic PESs seen when adding Wilson-chain spin-boson modes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg import log_discretized_spin_boson_wilson_chain, spin_boson_mode_pes


def main():
    alpha = 0.335
    chain = log_discretized_spin_boson_wilson_chain(
        16,
        alpha=alpha,
        Lambda=2.0,
        s=0.5,
        omegac=1.0,
        epsilon=0.0,
        delta=0.1,
    )
    q = np.linspace(-8.0, 8.0, 321)
    sites = [0, 1, 3, 5, 7, 9, 11, 13, 15]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.2), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, site in zip(axes, sites):
        pes = spin_boson_mode_pes(
            chain,
            site,
            q,
            nboson=5,
            bond_dim=32,
            nlevels=4,
            basis="sine-dvr",
            displacements=None,
            dvr_qmax=8.0,
        )
        for level in range(pes.surfaces.shape[1]):
            ax.plot(pes.q, pes.surfaces[:, level], linewidth=1.5)
        ax.set_title(
            f"add mode {site + 1}\n"
            f"omega={pes.onsite_frequency:.3e}, |V|={pes.coupling_norm:.3e}",
            fontsize=10,
        )
        ax.set_ylim(0.0, min(3.0, np.nanmax(pes.surfaces[:, :4])))
        ax.grid(True, alpha=0.25)

    for ax in axes[-3:]:
        ax.set_xlabel("new mode coordinate q")
    for ax in axes[::3]:
        ax.set_ylabel("relative adiabatic energy")
    fig.suptitle(f"Spin-boson Wilson-chain adiabatic PESs at alpha={alpha}", fontsize=14)

    out = Path(__file__).with_name("spin_boson_mode_pes.png")
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
