"""Propagate a two-level system with the compact HEOM solver."""

import numpy as np

from pyqed import pauli
from pyqed.oqs import HEOMSolver


_, sx, _, sz = pauli()
hamiltonian = -0.5 * (sx + sz)
rho0 = np.diag([0.0, 1.0])

expect = HEOMSolver(hamiltonian, c_ops=[sz], e_ops=[sz]).run(
    rho0, dt=0.02, nt=100, temperature=600,
    cutoff=5, reorganization=0.2, nado=5,
)

print(f"Final <sigma_z>: {expect[0, -1].real:.8f}")
