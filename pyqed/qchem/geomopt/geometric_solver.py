#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 00:05:45 2026

Geometry optimization

Interface to geometric  https://github.com/leeping/geomeTRIC

@author: bingg
"""

from pyqed.md.md import Dynamics
from os.path import isfile
from math import sqrt
import time

try:
    from geometric import internal, optimize, nifty, engine, molecule
    from geometric.errors import GeomOptNotConvergedError
except ImportError:
    msg = ('Geometry optimizer geomeTRIC not found.\ngeomeTRIC library '
           'can be found on github https://github.com/leeping/geomeTRIC.\n'
           'You can install geomeTRIC with "pip install geometric"')
    raise ImportError(msg)



class Optimizer(Dynamics):
    """Base-class for all structure optimization classes."""
    def __init__(self, atoms, restart, logfile, trajectory):
        """Structure optimizer object.

        atoms: Atoms object
            The Atoms object to relax.
        restart: str
            Filename for restart file.  Default value is *None*.
        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            PickleTrajectory will be constructed.  Use *None* for no
            trajectory.
        """
        Dynamics.__init__(self, atoms, logfile, trajectory)
        self.restart = restart

        if restart is None or not isfile(restart):
            self.initialize()
        else:
            self.read()
            # barrier()
    def initialize(self):
        pass

    def run(self, fmax=0.05, steps=100000000):
        """Run structure optimization algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*."""

        self.fmax = fmax
        step = 0
        while step < steps:
            f = self.atoms.get_forces()
            self.log(f)
            self.call_observers()
            if self.converged(f):
                return
            self.step(f)
            self.nsteps += 1
            step += 1

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.atoms.get_forces()
        if hasattr(self.atoms, 'get_curvature'):
            return (forces**2).sum(axis=1).max() < self.fmax**2 and \
                   self.atoms.get_curvature() < 0.0
        return (forces**2).sum(axis=1).max() < self.fmax**2

    def log(self, forces):
        fmax = sqrt((forces**2).sum(axis=1).max())
        e = self.atoms.get_potential_energy()
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            self.logfile.write('%s: %3d  %02d:%02d:%02d %15.6f %12.4f\n' %
                               (name, self.nsteps, T[3], T[4], T[5], e, fmax))
            self.logfile.flush()

    def dump(self, data):
        if rank == 0 and self.restart is not None:
            pickle.dump(data, open(self.restart, 'wb'), protocol=2)

    def load(self):
        return pickle.load(open(self.restart))