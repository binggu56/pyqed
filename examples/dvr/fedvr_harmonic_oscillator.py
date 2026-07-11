#!/usr/bin/env python3
"""Finite-element DVR harmonic oscillator smoke example."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import FEDVR


def main():
    dvr = FEDVR(-8.0, 8.0, n_elements=12, n_lobatto=6, mass=1.0)
    energies, _ = dvr.run(lambda x: 0.5 * x**2, num_eigs=8)
    exact = np.arange(8, dtype=float) + 0.5

    print(f"npts = {dvr.npts}")
    print(f"kinetic nnz = {dvr.kinetic_sparse().nnz}")
    print("state  FEDVR energy        exact        error")
    for i, (energy, ref) in enumerate(zip(energies, exact)):
        print(f"{i:5d}  {energy:16.10f}  {ref:10.6f}  {energy - ref: .3e}")


if __name__ == "__main__":
    main()
