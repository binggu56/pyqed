"""Four-mode, three-state pyrazine vibronic-coupling model."""

from __future__ import annotations

import numpy as np

from pyqed.models.vibronic import Vibronic2
from pyqed.units import au2ev, wavenum2au


class Pyrazine(Vibronic2):
    """Pyrazine $S_0/S_1/S_2$ model in four dimensionless normal modes."""

    frequencies = np.array([1015.0, 596.0, 1230.0, 919.0]) * wavenum2au
    shifts = np.array([3.94, 4.89]) / au2ev
    kappa_1 = np.array([-0.0470, -0.0964, 0.1594, 0.0]) / au2ev
    kappa_2 = np.array([-0.2012, 0.1193, 0.0484, 0.0]) / au2ev
    gamma = -0.018 / au2ev
    coupling = 0.1825 / au2ev

    def __init__(self, x, y, z, q):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.shape = tuple(len(axis) for axis in (self.x, self.y, self.z, self.q))
        self.nx, self.ny, self.nz, self.nq = self.shape
        self.nstates = 3
        self.idm_el = np.eye(self.nstates)
        self.edip = np.zeros((self.nstates, self.nstates))
        self.edip[0, 2] = self.edip[2, 0] = 1.0
        self.mass = tuple(1.0 / frequency for frequency in self.frequencies)
        self.v = None

    def buildV(self):
        """Return the global diabatic potential matrix field."""

        coordinates = np.meshgrid(
            self.x,
            self.y,
            self.z,
            self.q,
            indexing="ij",
        )
        harmonic = sum(
            0.5 * frequency * coordinate**2
            for frequency, coordinate in zip(self.frequencies, coordinates)
        )
        tuning_1 = sum(
            coefficient * coordinate
            for coefficient, coordinate in zip(self.kappa_1, coordinates)
        )
        tuning_2 = sum(
            coefficient * coordinate
            for coefficient, coordinate in zip(self.kappa_2, coordinates)
        )
        quadratic = self.gamma * coordinates[3] ** 2

        potential = np.zeros((*self.shape, self.nstates, self.nstates))
        potential[..., 0, 0] = harmonic
        potential[..., 1, 1] = harmonic + tuning_1 + self.shifts[0] + quadratic
        potential[..., 2, 2] = harmonic + tuning_2 + self.shifts[1] + quadratic
        potential[..., 1, 2] = self.coupling * coordinates[3]
        potential[..., 2, 1] = potential[..., 1, 2]
        self.v = potential
        return potential

    def apes_global(self):
        """Return adiabatic energies and local electronic frames."""

        potential = self.buildV() if self.v is None else self.v
        return np.linalg.eigh(potential)

    def apes(self, x, y, z, q):
        """Return adiabatic data at one coordinate point."""

        return np.linalg.eigh(dpes(x, y, z, q))


def dpes(x, y, z, q):
    """Return the three-state diabatic potential at one point."""

    coordinates = np.asarray([x, y, z, q], dtype=float)
    harmonic = 0.5 * np.dot(Pyrazine.frequencies, coordinates**2)
    potential = np.zeros((3, 3))
    potential[0, 0] = harmonic
    potential[1, 1] = (
        harmonic
        + np.dot(Pyrazine.kappa_1, coordinates)
        + Pyrazine.shifts[0]
        + Pyrazine.gamma * q**2
    )
    potential[2, 2] = (
        harmonic
        + np.dot(Pyrazine.kappa_2, coordinates)
        + Pyrazine.shifts[1]
        + Pyrazine.gamma * q**2
    )
    potential[1, 2] = potential[2, 1] = Pyrazine.coupling * q
    return potential


if __name__ == "__main__":
    from pyqed.dvr import DVR
    from pyqed.ldr import LDR

    grid = DVR([(-6.0, 6.0)] * 4, [3] * 4, mass=Pyrazine.frequencies**-1)
    model = Pyrazine(*grid.x)
    solver = LDR(grid, model.nstates).set_diabatic(
        model.buildV(),
        representation="links",
    )
    print(solver.shape, solver.size, len(solver.links))
