#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:26:27 2026

@author: gugroup
"""
from pyqed.md.md import MolecularDynamics

class VelocityVerlet(MolecularDynamics):
    """MD with NVE ensemble and velocity Verlet time integration."""

    def step(self, forces=None):

        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        p = atoms.get_momenta()
        p += 0.5 * self.dt * forces
        masses = atoms.get_masses()[:, None]
        r = atoms.get_positions()

        # if we have constraints then this will do the first part of the
        # RATTLE algorithm:
        atoms.set_positions(r + self.dt * p / masses)
        if atoms.constraints:
            p = (atoms.get_positions() - r) * masses / self.dt

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.
        atoms.set_momenta(p, apply_constraint=False)

        forces = atoms.get_forces(md=True)

        # Second part of RATTLE will be done here:
        atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dt * forces)
        return forces