#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Shin-Metiu Ehrenfest dynamics and plot the stored histories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed import proton_mass as mp
from pyqed.models.ShinMetiu import ShinMetiu2
from pyqed.namd import Ehrenfest


OUT = Path("examples/namd/ehrenfest_histories.png")


def main():
    mol = ShinMetiu2()
    mol.build(domain=[[-10, 10]] * 2, npts=[31, 31])

    ed = Ehrenfest(ndim=mol.ndim, ntraj=1, nstates=mol.nstates, mass=[mp] * 2)
    ed.nac_driver = mol.nonadiabatic_coupling
    ed.sample(init_state=2, x0=[0.0, 1.3], ax=18.0)
    ed.run(dt=0.5, nt=400, nout=2)

    populations = np.real(np.diagonal(ed.rho_history, axis1=1, axis2=2))

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    for dim in range(ed.x_history.shape[1]):
        axes[0].plot(ed.times, ed.x_history[:, dim], label=f"x[{dim}]")
    axes[0].set_ylabel("Position (bohr)")
    axes[0].legend(loc="best")

    for state in range(populations.shape[1]):
        axes[1].plot(ed.times, populations[:, state], label=f"pop[{state}]")
    axes[1].set_ylabel("Population")
    axes[1].legend(loc="best")

    axes[2].plot(ed.times, ed.energy_history, label="Ehrenfest energy")
    axes[2].plot(ed.times, ed.norm_history, label="Electronic norm")
    axes[2].set_xlabel("Time (a.u.)")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
