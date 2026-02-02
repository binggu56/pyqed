#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 00:26:16 2026

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""

from pyqed import Molecule
import numpy as np
from math import pi

#TODO: TO BE REMOVED
from ase.utils import string2index
from ase.cell import Cell

class Atoms(Molecule):
    """
    This is to define Atoms for MD, learn from ASE.
    It should be combined with Molecule class. (or should it?)
    """
    def __init__(self, atom, *args, cell=None, pbc=None, **kwargs):
        super().__init__(atom, *args, **kwargs)

        if cell is None:
            cell = np.zeros((3, 3))
        self.set_cell(cell)

    def set_cell(self, cell, scale_atoms=False, apply_constraint=True):
        """Set unit cell vectors.

        Parameters:

        cell: 3x3 matrix or length 3 or 6 vector
            Unit cell.  A 3x3 matrix (the three unit cell vectors) or
            just three numbers for an orthorhombic cell. Another option is
            6 numbers, which describes unit cell with lengths of unit cell
            vectors and with angles between them (in degrees), in following
            order: [len(a), len(b), len(c), angle(b,c), angle(a,c),
            angle(a,b)].  First vector will lie in x-direction, second in
            xy-plane, and the third one in z-positive subspace.
        scale_atoms: bool
            Fix atomic positions or move atoms with the unit cell?
            Default behavior is to *not* move the atoms (scale_atoms=False).
        apply_constraint: bool
            Whether to apply constraints to the given cell.

        Examples:

        Two equivalent ways to define an orthorhombic cell:

        >>> atoms = Atoms('He')
        >>> a, b, c = 7, 7.5, 8
        >>> atoms.set_cell([a, b, c])
        >>> atoms.set_cell([(a, 0, 0), (0, b, 0), (0, 0, c)])

        FCC unit cell:

        >>> atoms.set_cell([(0, b, b), (b, 0, b), (b, b, 0)])

        Hexagonal unit cell:

        >>> atoms.set_cell([a, a, c, 90, 90, 120])

        Rhombohedral unit cell:

        >>> alpha = 77
        >>> atoms.set_cell([a, a, a, alpha, alpha, alpha])
        """

        # Override pbcs if and only if given a Cell object:
        cell = Cell.new(cell)

        # XXX not working well during initialize due to missing _constraints
        if apply_constraint and hasattr(self, '_constraints'):
            for constraint in self.constraints:
                if hasattr(constraint, 'adjust_cell'):
                    constraint.adjust_cell(self, cell)

        if scale_atoms:
            M = np.linalg.solve(self.cell.complete(), cell.complete())
            self.positions[:] = np.dot(self.positions, M)

        self.cell[:] = cell

    @property
    def pbc(self):
        """Reference to pbc-flags for in-place manipulations."""
        return self._pbc

    @pbc.setter
    def pbc(self, pbc):
        self._pbc[:] = pbc

    def set_pbc(self, pbc):
        """Set periodic boundary condition flags."""
        self.pbc = pbc

    def get_pbc(self):
        """Get periodic boundary condition flags."""
        return self.pbc.copy()

    def set_momenta(self, momenta, apply_constraint=True):
        """Set momenta."""
        if (apply_constraint and len(self.constraints) > 0 and
           momenta is not None):
            momenta = np.array(momenta)  # modify a copy
            for constraint in self.constraints:
                if hasattr(constraint, 'adjust_momenta'):
                    constraint.adjust_momenta(self, momenta)
        self.set_array('momenta', momenta, float, (3,))

    def get_momenta(self):
        """Get array of momenta."""
        if 'momenta' in self.arrays:
            return self.arrays['momenta'].copy()
        else:
            return np.zeros((len(self), 3))

    def get_velocities(self):
        """Get array of velocities."""
        momenta = self.get_momenta()
        masses = self.get_masses()
        return momenta / masses[:, np.newaxis]

    def set_velocities(self, velocities):
        """Set the momenta by specifying the velocities."""
        self.set_momenta(self.get_masses()[:, np.newaxis] * velocities)

    def get_kinetic_energy(self):
        """Get the kinetic energy."""
        momenta = self.arrays.get('momenta')
        if momenta is None:
            return 0.0
        return 0.5 * np.vdot(momenta, self.get_velocities())

    def get_center_of_mass(self, scaled=False, indices=None):
         """Get the center of mass.

         Parameters
         ----------
         scaled : bool
             If True, the center of mass in scaled coordinates is returned.
         indices : list | slice | str, default: None
             If specified, the center of mass of a subset of atoms is returned.
         """
         if indices is None:
             indices = slice(None)
         elif isinstance(indices, str):
             indices = string2index(indices)

         masses = self.get_masses()[indices]
         com = masses @ self.positions[indices] / masses.sum()
         if scaled:
             return self.cell.scaled_positions(com)
         return com  # Cartesian coordinates

    def set_center_of_mass(self, com, scaled=False):
        """Set the center of mass.

        If scaled=True the center of mass is expected in scaled coordinates.
        Constraints are considered for scaled=False.
        """
        old_com = self.get_center_of_mass(scaled=scaled)
        difference = com - old_com
        if scaled:
            self.set_scaled_positions(self.get_scaled_positions() + difference)
        else:
            self.set_positions(self.get_positions() + difference)

    def get_angular_momentum(self):
        """Get total angular momentum with respect to the center of mass."""
        com = self.get_center_of_mass()
        positions = self.get_positions()
        positions -= com  # translate center of mass to origin
        return np.cross(positions, self.get_momenta()).sum(0)

    def get_total_energy(self):
        """Get the total energy - potential plus kinetic energy."""
        return self.get_potential_energy() + self.get_kinetic_energy()

    def get_kinetic_energy(self):
        pass

    @property
    def calc(self):
        """Calculator object."""
        return self._calc

    @calc.setter
    def calc(self, calc):
        self._calc = calc
        if hasattr(calc, 'set_atoms'):
            calc.set_atoms(self)

    def get_magnetic_moment(self):
        """Get calculated total magnetic moment."""
        if self._calc is None:
            raise RuntimeError('Atoms object has no calculator.')
        return self._calc.get_magnetic_moment(self)

    def get_charges(self):
        """Get calculated charges."""
        if self._calc is None:
            raise RuntimeError('Atoms object has no calculator.')
        try:
            return self._calc.get_charges(self)
        except AttributeError:
            # from ase.calculators.calculator import PropertyNotImplementedError
            raise NotImplementedError

    def set_dihedral(self, a1, a2, a3, a4, angle,
                     mask=None, indices=None):
        """Set the dihedral angle (degrees) between vectors a1->a2 and
        a3->a4 by changing the atom indexed by a4.

        If mask is not None, all the atoms described in mask
        (read: the entire subgroup) are moved. Alternatively to the mask,
        the indices of the atoms to be rotated can be supplied. If both
        *mask* and *indices* are given, *indices* overwrites *mask*.

        **Important**: If *mask* or *indices* is given and does not contain
        *a4*, *a4* will NOT be moved. In most cases you therefore want
        to include *a4* in *mask*/*indices*.

        Example: the following defines a very crude
        ethane-like molecule and twists one half of it by 30 degrees.

        >>> atoms = Atoms('HHCCHH', [[-1, 1, 0], [-1, -1, 0], [0, 0, 0],
        ...                          [1, 0, 0], [2, 1, 0], [2, -1, 0]])
        >>> atoms.set_dihedral(1, 2, 3, 4, 210, mask=[0, 0, 0, 1, 1, 1])
        """

        angle *= np.pi / 180

        # if not provided, set mask to the last atom in the
        # dihedral description
        if mask is None and indices is None:
            mask = np.zeros(len(self))
            mask[a4] = 1
        elif indices is not None:
            mask = [index in indices for index in range(len(self))]

        # compute necessary in dihedral change, from current value
        current = self.get_dihedral(a1, a2, a3, a4) * np.pi / 180
        diff = angle - current
        axis = self.positions[a3] - self.positions[a2]
        center = self.positions[a3]
        self._masked_rotate(center, axis, diff, mask)

    def rotate_dihedral(self, a1, a2, a3, a4, angle, mask=None, indices=None):
        """Rotate dihedral angle.

        Same usage as in :meth:`ase.Atoms.set_dihedral`: Rotate a group by a
        predefined dihedral angle, starting from its current configuration.
        """
        start = self.get_dihedral(a1, a2, a3, a4)
        self.set_dihedral(a1, a2, a3, a4, angle + start, mask, indices)

    def get_angle(self, a1, a2, a3, mic=False):
        """Get angle formed by three atoms.

        Calculate angle in degrees between the vectors a2->a1 and
        a2->a3.

        Use mic=True to use the Minimum Image Convention and calculate the
        angle across periodic boundaries.
        """
        return self.get_angles([[a1, a2, a3]], mic=mic)[0]

    def get_angles(self, indices, mic=False):
        """Get angle formed by three atoms for multiple groupings.

        Calculate angle in degrees between vectors between atoms a2->a1
        and a2->a3, where a1, a2, and a3 are in each row of indices.

        Use mic=True to use the Minimum Image Convention and calculate
        the angle across periodic boundaries.
        """
        #TODO: remove
        from ase.geometry import get_angles

        indices = np.array(indices)
        assert indices.shape[1] == 3

        a1s = self.positions[indices[:, 0]]
        a2s = self.positions[indices[:, 1]]
        a3s = self.positions[indices[:, 2]]

        v12 = a1s - a2s
        v32 = a3s - a2s

        cell = None
        pbc = None

        if mic:
            cell = self.cell
            pbc = self.pbc

        return get_angles(v12, v32, cell=cell, pbc=pbc)

    def set_angle(self, a1, a2=None, a3=None, angle=None, mask=None,
                  indices=None, add=False):
        """Set angle (in degrees) formed by three atoms.

        Sets the angle between vectors *a2*->*a1* and *a2*->*a3*.

        If *add* is `True`, the angle will be changed by the value given.

        Same usage as in :meth:`ase.Atoms.set_dihedral`.
        If *mask* and *indices*
        are given, *indices* overwrites *mask*. If *mask* and *indices*
        are not set, only *a3* is moved."""

        if any(a is None for a in [a2, a3, angle]):
            raise ValueError('a2, a3, and angle must not be None')

        # If not provided, set mask to the last atom in the angle description
        if mask is None and indices is None:
            mask = np.zeros(len(self))
            mask[a3] = 1
        elif indices is not None:
            mask = [index in indices for index in range(len(self))]

        if add:
            diff = angle
        else:
            # Compute necessary in angle change, from current value
            diff = angle - self.get_angle(a1, a2, a3)

        diff *= pi / 180
        # Do rotation of subgroup by copying it to temporary atoms object and
        # then rotating that
        v10 = self.positions[a1] - self.positions[a2]
        v12 = self.positions[a3] - self.positions[a2]
        v10 /= np.linalg.norm(v10)
        v12 /= np.linalg.norm(v12)
        axis = np.cross(v10, v12)
        center = self.positions[a2]
        self._masked_rotate(center, axis, diff, mask)

    def get_forces(self, apply_constraint=True, md=False):
        """Calculate atomic forces.

        Ask the attached calculator to calculate the forces and apply
        constraints.  Use *apply_constraint=False* to get the raw
        forces.

        For molecular dynamics (md=True) we don't apply the constraint
        to the forces but to the momenta. When holonomic constraints for
        rigid linear triatomic molecules are present, ask the constraints
        to redistribute the forces within each triple defined in the
        constraints (required for molecular dynamics with this type of
        constraints)."""

        if self._calc is None:
            raise RuntimeError('Atoms object has no calculator.')
        forces = self._calc.get_forces(self)

        if apply_constraint:
            # We need a special md flag here because for MD we want
            # to skip real constraints but include special "constraints"
            # Like Hookean.
            for constraint in self.constraints:
                if md and hasattr(constraint, 'redistribute_forces_md'):
                    constraint.redistribute_forces_md(self, forces)
                if not md or hasattr(constraint, 'adjust_potential_energy'):
                    constraint.adjust_forces(self, forces)
        return forces

    def get_dipole_moment(self):
        """Calculate the electric dipole moment for the atoms object.

        Only available for calculators which has a get_dipole_moment()
        method."""

        if self._calc is None:
            raise RuntimeError('Atoms object has no calculator.')
        return self._calc.get_dipole_moment(self)


if __name__=='__main__':
    mol = Atoms(atom = [
        ['H' , (0. , 0. , 0.91)],
        ['H' , (0. , 0. , -0.91)],
        ['H' , (0. , 0. , 3.6)],
        ['H' , (0. , 0. , -3.6)]])

    print(mol.get_angle(1,2,3))