#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 19:40:15 2025

@author: bingg

Refs:
    https://github.com/jevandezande/zmatrix/blob/master/converter.py#L159
"""

import numpy as np
from pyqed import atomic_mass

def add_first_three_to_cartesian(zmatrix):
     """
     The first three atoms in the zmatrix need to be treated differently
     """
     # First atom
     name, coords, mass = zmatrix[0]
     cartesian = [[name, np.array([0., 0., 0.]), mass]]

     # Second atom
     name, coords, mass = zmatrix[1]
     distance = coords[0][1]
     cartesian.append(
         [name, np.array([distance, 0., 0.]), atomic_mass[name]])

     # Third atom
     name, coords, mass = zmatrix[2]
     atom1, atom2 = coords[:2]
     atom1, distance = atom1
     atom2, angle = atom2
     q = np.array(cartesian[atom1][1], dtype='f8')  # position of atom 1
     r = np.array(cartesian[atom2][1], dtype='f8')  # position of atom 2

     # Vector pointing from q to r
     a = r - q

     # Vector of length distance pointing along the x-axis
     d = distance * a / np.sqrt(np.dot(a, a))

     # Rotate d by the angle around the z-axis
     d = np.dot(rotation_matrix([0, 0, 1], angle), d)

     # Add d to the position of q to get the new coordinates of the atom
     p = q + d
     atom = [name, p, atomic_mass[name]]
     cartesian.append(atom)
     return cartesian


def rotation_matrix(axis, angle):
    """
    Euler-Rodrigues formula for rotation matrix
    """
    # Normalize the axis
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def add_atom_to_cartesian(coords, cartesian):
    """Find the cartesian coordinates of the atom"""
    name, coords, mass = coords
    atom1, distance = coords[0]
    atom2, angle = coords[1]
    atom3, dihedral = coords[2]

    q = cartesian[atom1][1]  # atom 1
    r = cartesian[atom2][1]  # atom 2
    s = cartesian[atom3][1]  # atom 3

    # Vector pointing from q to r
    a = r - q
    # Vector pointing from s to r
    b = r - s

    # Vector of length distance pointing from q to r
    d = distance * a / np.sqrt(np.dot(a, a))

    # Vector normal to plane defined by q, r, s
    normal = np.cross(a, b)
    # Rotate d by the angle around the normal to the plane defined by q, r, s
    d = np.dot(rotation_matrix(normal, angle), d)

    # Rotate d around a by the dihedral
    d = np.dot(rotation_matrix(a, dihedral), d)

    # Add d to the position of q to get the new coordinates of the atom
    p = q + d
    atom = [name, p, mass]

    cartesian.append(atom)
    return cartesian

def remove_dummy_atoms(cartesian):
    """Delete any dummy atoms that may have been placed in the calculated cartesian coordinates"""
    new_cartesian = []
    for atom, xyz, mass in cartesian:
        if not atom == 'X':
            new_cartesian.append((atom, xyz, mass))
    cartesian = new_cartesian
    return cartesian

def center_cartesian(cartesian):
    """Find the center of mass and move it to the origin"""
    total_mass = 0.0
    center_of_mass = np.array([0.0, 0.0, 0.0])
    for atom, xyz, mass in cartesian:
        total_mass += mass
        center_of_mass += xyz * mass
    center_of_mass = center_of_mass / total_mass

    # Translate each atom by the center of mass
    for atom, xyz, mass in cartesian:
        xyz -= center_of_mass

    return cartesian

def zmatrix_to_cartesian(zmatrix):
    """
    Convert the zmartix to cartesian coordinates
    """
    # Deal with first three line separately
    cartesian = add_first_three_to_cartesian(zmatrix)

    for atom in zmatrix[3:]:
        cartesian = add_atom_to_cartesian(atom, cartesian)

    cartesian = remove_dummy_atoms(cartesian)

    center_cartesian(cartesian)

    return cartesian

def read_zmatrix(atoms):
    """
    Read the input zmatrix file (assumes no errors and no variables)
    The zmatrix is a list with each element formatted as follows
    [ name, [[ atom1, distance ], [ atom2, angle ], [ atom3, dihedral ]], mass ]
    The first three atoms have blank lists for the undefined coordinates
    """
    masses = atomic_mass

    zmatrix = []

    # with open(input_file, 'r') as f:


    lines = atoms.split('\n')
    lines = [line for line in lines if line.strip()]

    name = lines[0].strip()
    zmatrix.append([name, [], masses[name]])
    name, atom1, distance = lines[1].split()[:3]
    zmatrix.append([name,
                         [[int(atom1) - 1, float(distance)], [], []],
                         masses[name]])
    name, atom1, distance, atom2, angle = lines[2].split()[:5]
    zmatrix.append([name,
                         [[int(atom1) - 1, float(distance)],
                          [int(atom2) - 1, np.radians(float(angle))], []],
                         masses[name]])
    for line in lines[3:]:
        # Get the components of each line, dropping anything extra
        name, atom1, distance, atom2, angle, atom3, dihedral = line.split()[:7]
        # convert to a base 0 indexing system and use radians
        atom = [name,
                [[int(atom1) - 1, float(distance)],
                 [int(atom2) - 1, np.radians(float(angle))],
                 [int(atom3) - 1, np.radians(float(dihedral))]],
                masses[name]]

        zmatrix.append(atom)

    return zmatrix

def read_zmatrix_from_file(input_file):
    """
    Read the input zmatrix file (assumes no errors and no variables)
    The zmatrix is a list with each element formatted as follows
    [ name, [[ atom1, distance ], [ atom2, angle ], [ atom3, dihedral ]], mass ]
    The first three atoms have blank lists for the undefined coordinates
    """
    masses = atomic_mass

    zmatrix = []


    with open(input_file, 'r') as f:
        next(f)
        next(f)

        name = next(f).strip()
        zmatrix.append([name, [], masses[name]])
        name, atom1, distance = next(f).split()[:3]
        zmatrix.append([name,
                             [[int(atom1) - 1, float(distance)], [], []],
                             masses[name]])
        name, atom1, distance, atom2, angle = next(f).split()[:5]
        zmatrix.append([name,
                             [[int(atom1) - 1, float(distance)],
                              [int(atom2) - 1, np.radians(float(angle))], []],
                             masses[name]])

    for line in f:
        # Get the components of each line, dropping anything extra
        name, atom1, distance, atom2, angle, atom3, dihedral = line.split()[:7]
        # convert to a base 0 indexing system and use radians
        atom = [name,
                [[int(atom1) - 1, float(distance)],
                 [int(atom2) - 1, np.radians(float(angle))],
                 [int(atom3) - 1, np.radians(float(dihedral))]],
                masses[name]]

        zmatrix.append(atom)

    return zmatrix

if __name__=='__main__':
    atom ="""
        O
        H 1 1.0
        H 1 1.0 2 104.5
        """

    # print(lines[1].strip())

    zmatrix = read_zmatrix(atom)
    cart = zmatrix_to_cartesian(zmatrix)

    print(cart)