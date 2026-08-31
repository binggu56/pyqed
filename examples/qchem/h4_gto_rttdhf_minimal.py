#!/usr/bin/env python3
"""Minimal GTO RT-TDHF calculation for linear H4."""

import numpy as np

from pyqed.qchem import Molecule, RTTDHF
from pyqed.qchem.hf import RHF


mol = Molecule(
    atom="H 0 0 -3.6; H 0 0 -1.2; H 0 0 1.2; H 0 0 3.6",
    unit="bohr",
    basis="d-aug-cc-pvdz",
)
mol.build()  # Native GTO integrals; large ERIs are factorized automatically.
mf = RHF(mol).run()

e0 = 0.05
omega = 0.057
ncycles = 2
duration = ncycles * 2.0 * np.pi / omega
dt = 0.05
nsteps = round(duration / dt)


def field(t):
    ez = 0.0
    if 0.0 <= t <= duration:
        ez = e0 * np.sin(np.pi * t / duration) ** 2 * np.sin(omega * t)
    return np.array([0.0, 0.0, ez])


rt = RTTDHF(mf, field=field).run(dt=dt, nsteps=nsteps, store_dm=False)

np.savez(
    "h4_gto_rttdhf_2cycle.npz",
    time=rt.times,
    field=rt.fields,
    dipole=rt.dipoles,
    energy=rt.energies,
)
