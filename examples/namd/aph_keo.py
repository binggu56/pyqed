"""Minimal $J=0$ APH kinetic-energy construction for a triatom."""

import numpy as np

from pyqed.dvr import ExponentialDVR, LegendreDVR, SineDVR
from pyqed.ldr import keo


coordinates = keo.APH(
    atoms=("H", "H", "H"),
    masses=(1.0, 1.0, 1.0),
    jacobi_atoms=(0, (1, 2)),
)
dvrs = (
    SineDVR(1.5, 8.0, 10, mass=coordinates.mu),
    LegendreDVR(0.01, np.pi / 2.0 - 0.01, 8),
    ExponentialDVR(4, L=2.0 * np.pi, x0=np.pi),
)

T = coordinates.mpo(dvrs, field_max_rank=12, mpo_max_rank=32)
print("physical dimensions:", T.dims)
print("MPO bond dimensions:", T.bond_orders())
