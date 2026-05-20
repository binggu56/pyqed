#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 21:09:33 2026


@author: Bing Gu (gubing at westlake dot edu dot cn)
"""

import numpy as np
from numpy.random import standard_normal
from pyqed.md.md import MolecularDynamics
from pyqed.units import au2k


from ase.parallel import world


class Langevin(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics.

    Usage: Langevin(atoms, dt, temperature, friction)

    atoms
        The list of atoms.

    dt
        The time step.

    temperature
        The desired ``k_B T`` in Hartree.

    temperature_K
        Optional desired temperature in Kelvin.  If supplied, it is converted
        to Hartree with ``pyqed.units.au2k``.

    friction
        A friction coefficient, typically 1e-4 to 1e-2.

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    The temperature and friction are normally scalars, but in principle one
    quantity per atom could be specified by giving an array.

    This dynamics accesses the atoms using Cartesian coordinates."""

    def __init__(self, atoms, timestep, temperature=None, friction=None, fixcm=True,
                 trajectory=None, logfile=None, loginterval=1,
                 communicator=world, temperature_K=None):
        if friction is None:
            raise TypeError("friction must be supplied.")
        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval)
        self.temp = self._temperature_to_au(temperature, temperature_K)
        self.frict = friction
        self.fixcm = fixcm  # will the center of mass be held fixed?
        self.communicator = communicator
        self.updatevars()

    @staticmethod
    def _temperature_to_au(temperature=None, temperature_K=None):
        if temperature_K is not None:
            return float(temperature_K) / au2k
        if temperature is None:
            raise TypeError("temperature or temperature_K must be supplied.")
        return temperature

    def set_temperature(self, temperature=None, temperature_K=None):
        self.temp = self._temperature_to_au(temperature, temperature_K)
        self.updatevars()

    def set_friction(self, friction):
        self.frict = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        dt = self.dt
        # If the friction is an array some other constants must be arrays too.
        self._localfrict = hasattr(self.frict, 'shape')
        lt = self.frict * dt
        masses = self.masses
        sdpos = dt * np.sqrt(self.temp / masses * (2.0/3.0 - 0.5 * lt) * lt)
        sdpos.shape = (-1, 1)
        sdmom = np.sqrt(self.temp * masses * 2.0 * (1.0 - lt) * lt)
        sdmom.shape = (-1, 1)
        pmcor = np.sqrt(3.0)/2.0 * (1.0 - 0.125 * lt)
        cnst = np.sqrt((1.0 - pmcor) * (1.0 + pmcor))

        act0 = 1.0 - lt + 0.5 * lt * lt
        act1 = (1.0 - 0.5 * lt + (1.0/6.0) * lt * lt)
        act2 = 0.5 - (1.0/6.0) * lt + (1.0/24.0) * lt * lt
        c1 = act1 * dt / masses
        c1.shape = (-1, 1)
        c2 = act2 * dt * dt / masses
        c2.shape = (-1, 1)
        c3 = (act1 - act2) * dt
        c4 = act2 * dt
        del act1, act2
        if self._localfrict:
            # If the friction is an array, so are these
            act0.shape = (-1, 1)
            c3.shape = (-1, 1)
            c4.shape = (-1, 1)
            pmcor.shape = (-1, 1)
            cnst.shape = (-1, 1)
        self.sdpos = sdpos
        self.sdmom = sdmom
        self.c1 = c1
        self.c2 = c2
        self.act0 = act0
        self.c3 = c3
        self.c4 = c4
        self.pmcor = pmcor
        self.cnst = cnst

    def step(self, f):
        atoms = self.atoms
        p = self.atoms.get_momenta()

        random1 = standard_normal(size=(len(atoms), 3))
        random2 = standard_normal(size=(len(atoms), 3))

        self.communicator.broadcast(random1, 0)
        self.communicator.broadcast(random2, 0)

        rrnd = self.sdpos * random1
        prnd = (self.sdmom * self.pmcor * random1 +
                self.sdmom * self.cnst * random2)

        if self.fixcm:
            rrnd = rrnd - np.sum(rrnd, 0) / len(atoms)
            prnd = prnd - np.sum(prnd, 0) / len(atoms)
            n = len(atoms)
            rrnd *= np.sqrt(n / (n - 1.0))
            prnd *= np.sqrt(n / (n - 1.0))

        atoms.set_positions(atoms.get_positions() +
                            self.c1 * p +
                            self.c2 * f + rrnd)
        p *= self.act0
        p += self.c3 * f + prnd
        atoms.set_momenta(p)

        f = atoms.get_forces(md=True)
        atoms.set_momenta(p + self.c4 * f)
        return f
