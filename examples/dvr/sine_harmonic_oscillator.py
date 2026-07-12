"""Solve the one-dimensional harmonic oscillator with a sine DVR."""

import numpy as np

from pyqed.dvr import SineDVR


dvr = SineDVR(-8.0, 8.0, 80)
hamiltonian = dvr.t() + np.diag(0.5 * dvr.x**2)
energies = np.linalg.eigvalsh(hamiltonian)[:4]

print(np.array2string(energies, precision=8))
