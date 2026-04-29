#!/usr/bin/env python3
"""LiF TDDFT-Ehrenfest population dynamics demo.

This example uses the native pyqed TDDFT backend because the current PySCF
rebuild path can be fragile for LiF along a trajectory.  It initializes the
system in the first excited state, launches a small stretch displacement, runs
overlap-based Ehrenfest dynamics, and saves a population plot.

The resulting population transfer is typically weak for this tiny STO-3G demo,
so treat it as a working dynamics example rather than a converged physical model
of LiF internal conversion.
"""

from pathlib import Path
import io
import sys
import contextlib

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed import au2fs
from pyqed.namd import AbInitioEhrenfest, TDDFTDriver
from pyqed.qchem import Molecule


def main():
    np.random.seed(2)

    mol = Molecule(
        atom="Li 0 0 0; F 0 0 2.9",
        unit="bohr",
        basis="sto-3g",
    )

    # Single Li-F stretch mode in Cartesian coordinates.
    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.08])
    modes = np.zeros((6, 2, 3), dtype=float)
    modes[-1, 0, 2] = -0.5
    modes[-1, 1, 2] = 0.5

    # Silence repeated backend printing from many single-point TDDFT calls.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        driver = TDDFTDriver(mol, nstates=2, xc="lda")

        ed = AbInitioEhrenfest(mol, ntraj=1, nstates=2, nac_driver=driver)
        ed.sample(
            init_state=1,
            frequencies=frequencies,
            normal_modes=modes,
            q0=0.02,
            p0=0.0,
            q_var=0.0,
            p_var=0.0,
        )
        ed.run(
            dt=0.02,
            nt=10,
            nout=1,
            electronic_representation="overlap",
        )

    times_fs = ed.times * au2fs
    populations = np.real(np.diagonal(ed.rho_history, axis1=1, axis2=2))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(times_fs, populations[:, 0], "o-", lw=1.8, ms=4, label="S0")
    ax.plot(times_fs, populations[:, 1], "o-", lw=1.8, ms=4, label="S1")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Population")
    ax.set_title("LiF TDDFT-Ehrenfest Population Dynamics")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    outpath = Path(__file__).with_name("lif_population_dynamics.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print(f"Saved plot to: {outpath}")
    print("Times (fs):")
    print(times_fs)
    print("Populations [S0, S1]:")
    print(populations)


if __name__ == "__main__":
    main()
