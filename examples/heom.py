#!/usr/bin/env python3
"""Propagate a dissipative spin-boson model with HEOM.

The example starts from the second diabatic state and records the expectation
value of sigma-z for 100 time steps.
"""

import numpy as np

from pyqed import pauli
from pyqed.oqs import HEOMSolver


delta = 1.0
bias = 1.0
_, sigma_x, _, sigma_z = pauli()
hamiltonian = -0.5 * delta * sigma_x - 0.5 * bias * sigma_z

rho0 = np.zeros((2, 2), dtype=complex)
rho0[1, 1] = 1.0

solver = HEOMSolver(hamiltonian, c_ops=[sigma_z], e_ops=[sigma_z])
result = solver.run(
    rho0=rho0,
    dt=0.02,
    nt=100,
    temperature=600,
    cutoff=5,
    reorganization=0.2,
    nado=5,
)

print(f"Final <sigma_z>: {np.real(result[0, -1]):.8f}")
