"""Velocity initialization helpers for atomic-unit MD."""

import numpy as np

from pyqed.units import au2k


def set_maxwell_boltzmann_velocities(
    atoms,
    temperature,
    seed=None,
    remove_center_of_mass=True,
):
    """Initialize velocities from a Maxwell-Boltzmann distribution.

    Parameters
    ----------
    atoms
        :class:`pyqed.md.Atoms` object. Masses are read in atomic units.
    temperature
        Temperature in Kelvin.
    seed
        Optional NumPy random seed.
    remove_center_of_mass
        If true, remove net momentum after sampling.
    """
    rng = np.random.default_rng(seed)
    masses = atoms.get_masses()
    kbt = float(temperature) / au2k
    velocities = rng.normal(size=(len(atoms), 3)) * np.sqrt(kbt / masses)[:, None]
    atoms.set_velocities(velocities)
    if remove_center_of_mass:
        momenta = atoms.get_momenta()
        momenta -= momenta.sum(axis=0) / len(atoms)
        atoms.set_momenta(momenta, apply_constraint=False)
    return atoms.get_velocities()
