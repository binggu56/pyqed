#!/usr/bin/env python3
"""Compute the linear absorption spectrum of a four-level model."""

import numpy as np

from pyqed import Mol, au2ev


energies_ev = np.array([0.0, 0.5, 1.1, 1.3])
hamiltonian = np.diag(energies_ev / au2ev)

transition_dipole = np.zeros_like(hamiltonian)
for excited_state in (1, 2, 3):
    transition_dipole[0, excited_state] = 1.0
    transition_dipole[excited_state, 0] = 1.0

model = Mol(hamiltonian, edip=transition_dipole)
frequencies = np.linspace(0.0, 2.0, 200) / au2ev
model.absorption(omegas=frequencies)
