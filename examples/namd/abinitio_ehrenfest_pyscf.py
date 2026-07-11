#!/usr/bin/env python3
"""Minimal PySCF-backed ab initio Ehrenfest dynamics example.

This example runs a short TDDFT/RKS Ehrenfest trajectory for H2 using the
overlap-based local-diabatic propagation mode, which is currently the most
robust ab initio path in ``pyqed.namd``.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.namd import AbInitioEhrenfest, TDDFTDriver


def main():
    try:
        from pyscf import gto
    except ImportError as exc:
        raise SystemExit("This example requires PySCF to be installed.") from exc

    np.random.seed(4)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.392",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )

    driver = TDDFTDriver(mol, nstates=2, xc="lda")

    # One vibrational mode for H2: bond stretch along the molecular axis.
    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    modes = np.zeros((6, 2, 3), dtype=float)
    modes[-1, 0, 2] = -0.5
    modes[-1, 1, 2] = 0.5

    ed = AbInitioEhrenfest(mol, ntraj=1, nstates=2, nac_driver=driver)
    ed.sample(
        init_state=0,
        frequencies=frequencies,
        normal_modes=modes,
    )

    ed.run(
        dt=0.05,
        nt=10,
        nout=2,
        electronic_representation="overlap",
    )

    print("Times (a.u.):")
    print(ed.times)
    print()
    print("Energy history (Ha):")
    print(ed.energy_history)
    print()
    print("Final averaged geometry (Bohr):")
    print(ed.x_history[-1].reshape(2, 3))
    print()
    print("Final electronic density matrix:")
    print(ed.rho_history[-1])


if __name__ == "__main__":
    main()
