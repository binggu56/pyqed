# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:28:20 2022

Cartesian to internal coordinates transformation

Based on QCL https://github.com/ben-albrecht/qcl/blob/master/qcl/

@author: Bing Gu
"""

from __future__ import print_function
from __future__ import division

import math
from numpy import sin, cos, pi
from numpy.linalg import norm
import numpy as np

from pyscf import dft, scf, gto

from pyqed import discretize
from pyqed.dvr import SineDVR

# Suppress scientific notation printouts and change default precision
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def build_atom_from_coords(atom_symbol_list, coords):
    """
    construct the atom data format (i.e. xyz format) used in pyscf from coordinates and atom symbols

    Parameters
    ----------
    atom_symbol_list : TYPE
        DESCRIPTION.
    coords : TYPE
        DESCRIPTION.

    Returns
    -------
    atom : TYPE
        DESCRIPTION.

    """
    natm = len(atom_symbol_list)
    atom = []
    for n in range(natm):
        atom.append([atom_symbol_list[n],  coords[n, :].tolist()])

    return atom 

# try:
#     from cclib.parser.data import ccData
#     from cclib.parser.utils import PeriodicTable
# except ImportError:
#     print("Failed to load cclib!")
#     raise


# class ccData_xyz(ccData):
#     """
#     ccData subclass for xyzfiles
#     TODO: Checks for previous steps before continuing,
#     i.e. check for dist_matrix before building conn_vector
#     Includes some hot new attributes and class methods
#     """

#     def __init__(self, attributes={}):
#         """Adding some new attributes for xyzfiles"""

#         self.newcoords = None
#         self.distancematrix = None

#         # Internal Coordinate Connectivity
#         self.connectivity = None
#         self.angleconnectivity = None
#         self.dihedralconnectivity = None

#         # Internal Coordinates
#         self.distances = None
#         self.angles = None
#         self.dihedrals = None

#         self._attrtypes['comment'] = str
#         self._attrlist.append('comment')
#         self._attrtypes['filename'] = str
#         self._attrlist.append('filename')
#         self._attrtypes['elements'] = list
#         self._attrlist.append('elements')

#         #self._attrtypes['distancematrix'] = np.ndarray
#         #self._attrlist.append('distancematrix')
#         #self._attrtypes['connectivity'] = list
#         #self._attrlist.append('connectivity')

#         super(ccData_xyz, self).__init__(attributes=attributes)

#         # Initialize new data types if attributes were parsed as an original ccdata_xyz
#         if not hasattr(self, 'elements'):
#             pt = PeriodicTable()
#             self.comment = '\n'
#             self.filename = ''
#             self.elements = []
#             for atomno in self.atomnos:
#                 self.elements.append(pt.element[atomno])

#     def _build_distance_matrix(self):
#         """Build distance matrix between all atoms
#            TODO: calculate distances only as needed for efficiency"""
#         coords = self.atomcoords[-1]
#         self.distancematrix = np.zeros((len(coords), len(coords)))
#         for i in range(len(coords)):
#             for j in [x for x in range(len(coords)) if x > i]:
#                 self.distancematrix[i][j] = norm(coords[i] - coords[j])
#                 self.distancematrix[j][i] = self.distancematrix[i][j]

#     def build_zmatrix(self):
#         """
#        'Z-Matrix Algorithm'
#         Build main components of zmatrix:
#         Connectivity vector
#         Distances between connected atoms (atom >= 1)
#         Angles between connected atoms (atom >= 2)
#         Dihedral angles between connected atoms (atom >= 3)
#         """
#         self._build_distance_matrix()

#         # self.connectivity[i] tells you the index of 2nd atom connected to atom i
#         self.connectivity = np.zeros(len(self.atomnos)).astype(int)

#         # self.angleconnectivity[i] tells you the index of
#         #    3rd atom connected to atom i and atom self.connectivity[i]
#         self.angleconnectivity = np.zeros(len(self.atomnos)).astype(int)

#         # self.dihedralconnectivity tells you the index of 4th atom connected to
#         #    atom i, atom self.connectivity[i], and atom self.angleconnectivity[i]
#         self.dihedralconnectivity = np.zeros(len(self.atomnos)).astype(int)

#         # Starts with r1
#         self.distances = np.zeros(len(self.atomnos))
#         # Starts with a2
#         self.angles = np.zeros(len(self.atomnos))
#         # Starts with d3
#         self.dihedrals = np.zeros(len(self.atomnos))

#         atoms = range(1, len(self.atomnos))
#         for atom in atoms:
#             # For current atom, find the nearest atom among previous atoms
#             distvector = self.distancematrix[atom][:atom]
#             distmin = np.array(distvector[np.nonzero(distvector)]).min()
#             nearestindices = np.where(distvector == distmin)[0]
#             nearestatom = nearestindices[0]

#             self.connectivity[atom] = nearestatom
#             self.distances[atom] = distmin

#             # Compute Angles
#             if atom >= 2:
#                 atms = [0, 0, 0]
#                 atms[0] = atom
#                 atms[1] = self.connectivity[atms[0]]
#                 atms[2] = self.connectivity[atms[1]]
#                 if atms[2] == atms[1]:
#                     for idx in range(1, len(self.connectivity[:atom])):
#                         if self.connectivity[idx] in atms and not idx in atms:
#                             atms[2] = idx
#                             break

#                 self.angleconnectivity[atom] = atms[2]

#                 self.angles[atom] = self._calc_angle(atms[0], atms[1], atms[2])

#             # Compute Dihedral Angles
#             if atom >= 3:
#                 atms = [0, 0, 0, 0]
#                 atms[0] = atom
#                 atms[1] = self.connectivity[atms[0]]
#                 atms[2] = self.angleconnectivity[atms[0]]
#                 atms[3] = self.angleconnectivity[atms[1]]
#                 if atms[3] in atms[:3]:
#                     for idx in range(1, len(self.connectivity[:atom])):
#                         if self.connectivity[idx] in atms and not idx in atms:
#                             atms[3] = idx
#                             break

#                 self.dihedrals[atom] =\
#                     self._calc_dihedral(atms[0], atms[1], atms[2], atms[3])
#                 if math.isnan(self.dihedrals[atom]):
#                     # TODO: Find explicit way to denote undefined dihedrals
#                     self.dihedrals[atom] = 0.0

#                 self.dihedralconnectivity[atom] = atms[3]

#     def _calc_angle(self, atom1, atom2, atom3):
#         """Calculate angle between 3 atoms"""
#         coords = self.atomcoords[-1]
#         vec1 = coords[atom2] - coords[atom1]
#         uvec1 = vec1 / norm(vec1)
#         vec2 = coords[atom2] - coords[atom3]
#         uvec2 = vec2 / norm(vec2)
#         return np.arccos(np.dot(uvec1, uvec2))*(180.0/pi)

#     def _calc_dihedral(self, atom1, atom2, atom3, atom4):
#         """
#            Calculate dihedral angle between 4 atoms
#            For more information, see:
#                http://math.stackexchange.com/a/47084
#         """
#         coords = self.atomcoords[-1]
#         # Vectors between 4 atoms
#         b1 = coords[atom2] - coords[atom1]
#         b2 = coords[atom2] - coords[atom3]
#         b3 = coords[atom4] - coords[atom3]

#         # Normal vector of plane containing b1,b2
#         n1 = np.cross(b1, b2)
#         un1 = n1 / norm(n1)

#         # Normal vector of plane containing b1,b2
#         n2 = np.cross(b2, b3)
#         un2 = n2 / norm(n2)

#         # un1, ub2, and m1 form orthonormal frame
#         ub2 = b2 / norm(b2)
#         um1 = np.cross(un1, ub2)

#         # dot(ub2, n2) is always zero
#         x = np.dot(un1, un2)
#         y = np.dot(um1, un2)

#         dihedral = np.arctan2(y, x)*(180.0/pi)
#         if dihedral < 0:
#             dihedral = 360.0 + dihedral
#         return dihedral

#     def build_xyz(self):
#         """ Build xyz representation from z-matrix"""
#         coords = self.atomcoords[-1]
#         self.newcoords = np.zeros((len(coords), 3))
#         for i in range(len(coords)):
#             self.newcoords[i] = self._calc_position(i)
#         self.atomcoords[-1] = self.newcoords

#     def _calc_position(self, i):
#         """Calculate position of another atom based on internal coordinates"""

#         if i > 1:
#             j = self.connectivity[i]
#             k = self.angleconnectivity[i]
#             l = self.dihedralconnectivity[i]

#             # Prevent doubles
#             if k == l and i > 0:
#                 for idx in range(1, len(self.connectivity[:i])):
#                     if self.connectivity[idx] in [i, j, k] and not idx in [i, j, k]:
#                         l = idx
#                         break

#             avec = self.newcoords[j]
#             bvec = self.newcoords[k]

#             dst = self.distances[i]
#             ang = self.angles[i] * pi / 180.0

#             if i == 2:
#                 # Third atom will be in same plane as first two
#                 tor = 90.0 * pi / 180.0
#                 cvec = np.array([0, 1, 0])
#             else:
#                 # Fourth + atoms require dihedral (torsional) angle
#                 tor = self.dihedrals[i] * pi / 180.0
#                 cvec = self.newcoords[l]

#             v1 = avec - bvec
#             v2 = avec - cvec

#             n = np.cross(v1, v2)
#             nn = np.cross(v1, n)

#             n /= norm(n)
#             nn /= norm(nn)

#             n *= -sin(tor)
#             nn *= cos(tor)

#             v3 = n + nn
#             v3 /= norm(v3)
#             v3 *= dst * sin(ang)

#             v1 /= norm(v1)
#             v1 *= dst * cos(ang)

#             position = avec + v3 - v1

#         elif i == 1:
#             # Second atom dst away from origin along Z-axis
#             j = self.connectivity[i]
#             dst = self.distances[i]
#             position = np.array([self.newcoords[j][0] + dst, self.newcoords[j][1], self.newcoords[j][2]])

#         elif i == 0:
#             # First atom at the origin
#             position = np.array([0, 0, 0])

#         return position

#     @property
#     def splitatomnos(self):
#         """Returns tuple of atomnos from reactants joined by atoms 0 and 1"""
#         fragments = [[], []]

#         return fragments


#     def print_distance_matrix(self):
#         """Print distance matrix in formatted form"""

#         # Title
#         print("\nDistance Matrix")

#         # Row Indices
#         for i in range(len(self.distancematrix)):
#             print("%3d" % i, end="  ")

#         print("\n", end="")
#         idx = 0
#         for vector in self.distancematrix:

#             # Column indices
#             print(idx, end=" ")

#             # Actual Values
#             for element in vector:
#                 if not element == 0:
#                     print("%1.2f" % element, end=" ")
#                 else:
#                     print("%1s" % " ", end="    ")
#             print("\n", end="")
#             idx += 1

#     def print_xyz(self):
#         """Print Standard XYZ Format"""
#         if not self.newcoords.any():
#             self.build_xyz()

#         print(len(self.newcoords))

#         if self.comment:
#             print(self.comment, end='')
#         else:
#             print(self.filename, end='')

#         atomcoords = [x.tolist() for x in self.newcoords]
#         for i in range(len(atomcoords)):
#             atomcoords[i].insert(0, self.elements[i])

#         for atom in atomcoords:
#             print("  %s %10.5f %10.5f %10.5f" % tuple(atom))

#     def print_gzmat(self):
#         """Print Gaussian Z-Matrix Format
#         e.g.
#         0  3
#         C
#         O  1  r2
#         C  1  r3  2  a3
#         Si 3  r4  1  a4  2  d4
#         ...
#         Variables:
#         r2= 1.1963
#         r3= 1.3054
#         a3= 179.97
#         r4= 1.8426
#         a4= 120.10
#         d4=  96.84
#         ...
#         """
#         pt = PeriodicTable()

#         print('#', self.filename, "\n")
#         print(self.comment)

#         print(self.comment, end='')
#         for i in range(len(self.atomnos)):
#             idx = str(i+1)+" "
#             if i >= 3:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx,
#                       self.angleconnectivity[i]+1, " a"+idx,
#                       self.dihedralconnectivity[i]+1, " d"+idx.rstrip())
#             elif i == 2:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx,
#                       self.angleconnectivity[i]+1, " a"+idx.rstrip())
#             elif i == 1:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx.rstrip())
#             elif i == 0:
#                 print(pt.element[self.atomnos[i]])

#         print("Variables:")

#         for i in range(1, len(self.atomnos)):
#             idx = str(i+1)+"="
#             if i >= 3:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])
#                 print("%s" % "a"+idx, "%6.2f" % self.angles[i])
#                 print("%s" % "d"+idx, "%6.2f" % self.dihedrals[i])
#             elif i == 2:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])
#                 print("%s" % "a"+idx, "%6.2f" % self.angles[i])
#             elif i == 1:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])

#     def print_zmat(self):
#         """Print Standard Z-Matrix Format"""
#         #TODO

#         """
#         0 1
#         O
#         O 1 1.5
#         H 1 1.0 2 120.0
#         H 2 1.0 1 120.0 3 180.0
#         """

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import newton

# from pyscf.lib import logger
import pyscf.ao2mo
import pyscf
from functools import reduce


# from pyqed import eig_asymm, is_positive_def, dag


def intertia_moment(mass, coords):
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    coords = coords - mass_center
    im = np.einsum('i,ij,ik->jk', mass, coords, coords)
    im = np.eye(3) * im.trace() - im
    return im


# class Molecule:
#     def __init__(self, geometry):
#         self.geometry = geometry

#         self.ge = None
#         self.ee = None
#         self.edip = None
#         self.mdip = None

#     # def atom_symbol(self):
#     #     return [mol.atom_symbol(i) for i in range(self.natoms)]
#     def rhf(self):
#         pass

#     def absorption(self, ttype='electron'):
#         # range, uv, ir, xray

#         pass

#     def photoelectron(self):
#         pass

#     def emission(self):
#         pass


# class Molecule(gto.M):
class Mol:
    """
    
    """
    def __init__(self, geometry=None, dvr_type='sine', spin=0, charge=0):
        
        
        self.geometry = geometry # proton position
        
        self.domain = None
        
        self.spin = spin
        self.charge = charge 
        # self.atomic_charge = [1, 1, 1]     # Ion charge
        # self.e = 1     # Electron charge, should be set to actual value in atomic units
        
        # self.L = 10/au2angstrom
        # print(self.L)
        # self.mass = mass  # nuclear mass
        # self.left = np.array([-self.L/2])
        # self.right = np.array([self.L/2])
    
        # self.left = np.array([-self.L/2, 0])
        # self.right = np.array([self.L/2, 0])
    
        self.x = None
        self.y = None
        self.z = None
        
        self.nx = None
        self.ny = None
        self.nz = None
        
        self.X = None
        self.Y = None
        self.Z = None
        
        self.dvr_type = dvr_type
        

        
        # self.atom_coord = mol.atom_coord

        # print(self.atom_coords.shape)
        self.natom = mol.natm
        self.mass = None # mol.atom_mass_list()
        self.atom_symbols = [mol.atom_symbol(i) for i in range(self.natom)]

        self.distmat = None
        # self.e_nuc = None
        
        # DVR basis set
        self.domain = None
        self.level = None
   
    def atom_masses(self):
        pass
    
    def atom_coords(self):
        # shape 3, natoms
        self.atom_coords = None
        pass
        
    def create_grid(self, domain, level):
        """
        create DVR basis set

        Parameters
        ----------
        domain : TYPE
            DESCRIPTION.
        level : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        x = discretize(*domain, level, endpoints=False)
        y = x.copy() 
        
        self.x = x 
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        # self.lx = domain[0][1]-domain[0][0]
        # self.ly = domain[0][1]-domain[0][0]
        # self.dx = self.lx / (self.nx - 1)
        # self.dy = self.ly / (self.ny - 1)
        self.domain = domain 
        self.level = level
    
    def get_hcore(self, R=0):
        """
        single point calculations

        Parameters
        ----------
        R : float
            proton position.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        w : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.

        """
        
        # H(r; R)
        x = self.x
        nx = self.nx

        
        # T 
        # origin method of calculate kinetic term
        # tx = kinetic(x, dvr=self.dvr_type)
        # idx = np.eye(nx)  
        
        # ty = kinetic(y, dvr=self.dvr_type)
        # idy = np.eye(ny)
        
        # T = kron(tx, idy) + kron(idx, ty)
        
        # # # new method of calculate kinetic term
        dvr_x = SineDVR(*self.domain, nx)
        
        # tx = kinetic(self.x, dvr=self.dvr_type)
        T = dvr_x.t()
    
        self.T = T
        
        
        # V
        Ra = self.left
        Rb = self.right 
        v = np.zeros((nx))
        for i in range(nx):   
            r1 = np.array(x[i])
            # Potential from all ions
            v[i] = self.V_en(r1, Ra) + self.V_en(r1, Rb) + self.V_en1(r1, R)
        
        V = np.diag(v)
        # print(V.shape)
        
        # v_sym = self.enforce_spin_symmetry(v)
        # # print(v_sym.shape)
        # V = np.diag(v_sym.ravel())
        
        H = T + V 
        # H = self.imaginary_time_propagation(H)
        
        # if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        #     raise ValueError("H matrix contains NaNs or infs.")
        
        return H

        


    
    def atom_charge(self, atm_id):
        pass
    
    def atom_charges(self):
        pass

    def com(self):
        '''
        return center of mass

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        mass = self.mass
        return np.einsum('i,ij->j', mass, self.atom_coords)/mass.sum()

    def inertia_moment(self):
        mass = self.mass
        coords = self.atom_coords
        return intertia_moment(mass, coords)

    def molecular_frame(self):
        # transfrom to molecular frame
        self.atom_coords -= self.com()
        return self.atom_coords

    def eckart_frame(self, ref):
        """
        transform to the Eckart frame relative to a reference geometry

        Parameters
        ----------
        ref : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.atom_coords = eckart(ref.T, self.atom_coords.T, self.mass)
        return self.atom_coords

    def principle_axes(self):
        pass

    def _build_distance_matrix(self):
        """Build distance matrix between all atoms
           TODO: calculate distances only as needed for efficiency"""
        coords = self.atom_coords()
        natom = self.natm

        distancematrix = np.zeros((natom, natom))

        for i in range(natom):
            for j in range(i+1, natom):
                distancematrix[i, j] = np.linalg.norm(coords[:, i]-coords[:, j])
                distancematrix[j, i] = distancematrix[i, j]

        self.distmat =  distancematrix
        return distancematrix

    def _calc_angle(self, atom1, atom2, atom3):
        """
        Calculate angle in radians between 3 atoms

        Parameters
        ----------
        atom1 : TYPE
            DESCRIPTION.
        atom2 : TYPE
            DESCRIPTION.
        atom3 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        vec1 = self.atom_coord(atom2) - self.atom_coord(atom1)
        uvec1 = vec1 / norm(vec1)
        vec2 = self.atom_coord(atom2) - self.atom_coord(atom3)
        uvec2 = vec2 / norm(vec2)
        return np.arccos(np.dot(uvec1, uvec2))*(180.0/pi)

    def _calc_dihedral(self, atom1, atom2, atom3, atom4):
        """

           Calculate dihedral angle (in radians) between 4 atoms
           For more information, see:
               http://math.stackexchange.com/a/47084

        Parameters
        ----------
        atom1 : TYPE
            DESCRIPTION.
        atom2 : TYPE
            DESCRIPTION.
        atom3 : TYPE
            DESCRIPTION.
        atom4 : TYPE
            DESCRIPTION.

        Returns
        -------
        dihedral : TYPE
            DESCRIPTION.

        """
        r1 = self.atom_coord(atom1)
        r2 = self.atom_coord(atom2)
        r3 = self.atom_coord(atom3)
        r4 = self.atom_coord(atom4)

        # Vectors between 4 atoms
        b1 = r2 - r1
        b2 = r2 - r3
        b3 = r4 - r3

        # Normal vector of plane containing b1,b2
        n1 = np.cross(b1, b2)
        un1 = n1 / norm(n1)

        # Normal vector of plane containing b1,b2
        n2 = np.cross(b2, b3)
        un2 = n2 / norm(n2)

        # un1, ub2, and m1 form orthonormal frame
        ub2 = b2 / norm(b2)
        um1 = np.cross(un1, ub2)

        # dot(ub2, n2) is always zero
        x = np.dot(un1, un2)
        y = np.dot(um1, un2)

        dihedral = np.arctan2(y, x)*(180.0/pi)
        if dihedral < 0:
            dihedral = 360.0 + dihedral
        return dihedral

    def zmat(self, rvar=False, avar=False, dvar=False):
        npart = self.natm

        if self.distmat is None:
            self._build_distance_matrix()

        distmat = self.distmat

        atomnames = self.atom_symbols()

        rlist = []
        alist = []
        dlist = []
        if npart > 0:
            # Write the first atom
            print(atomnames[0])

            if npart > 1:
                # and the second, with distance from first
                n = atomnames[1]
                rlist.append(distmat[0][1])
                if (rvar):
                    r = 'R1'
                else:
                    r = '{:>11.5f}'.format(rlist[0])
                print('{:<3s} {:>4d}  {:11s}'.format(n, 1, r))

                if npart > 2:
                    n = atomnames[2]

                    rlist.append(distmat[0][2])
                    if (rvar):
                        r = 'R2'
                    else:
                        r = '{:>11.5f}'.format(rlist[1])

                    alist.append(self._calc_angle(2, 0, 1))
                    if (avar):
                        t = 'A1'
                    else:
                        t = '{:>11.5f}'.format(alist[0])

                    print('{:<3s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, 1, r, 2, t))

                    if npart > 3:
                        for i in range(3, npart):
                            n = atomnames[i]

                            rlist.append(distmat[i-3][i])
                            if (rvar):
                                r = 'R{:<4d}'.format(i)
                            else:
                                r = '{:>11.5f}'.format(rlist[i-1])

                            alist.append(self._calc_angle(i, i-3, i-2))
                            if (avar):
                                t = 'A{:<4d}'.format(i-1)
                            else:
                                t = '{:>11.5f}'.format(alist[i-2])

                            dlist.append(self._calc_dihedral(i, i-3, i-2, i-1))
                            if (dvar):
                                d = 'D{:<4d}'.format(i-2)
                            else:
                                d = '{:>11.5f}'.format(dlist[i-3])
                            print('{:3s} {:>4d}  {:11s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, i-2, r, i-1, t, i, d))
        if (rvar):
            print(" ")
            for i in range(npart-1):
                print('R{:<4d} = {:>11.5f}'.format(i+1, rlist[i]))
        if (avar):
            print(" ")
            for i in range(npart-2):
                print('A{:<4d} = {:>11.5f}'.format(i+1, alist[i]))
        if (dvar):
            print(" ")
            for i in range(npart-3):
                print('D{:<4d} = {:>11.5f}'.format(i+1, dlist[i]))

        return

    def jacobian(self, q):
        return

    def metric(self):
        pass

    def tofile(self,fname):
        pass
    
    def RHF(self):
        pass
    
    def UHF(self):
        pass
    
    def RKS(self):
        pass
    
    def UKS(self):
        pass
    
    # def energy_nuc(self):
    #     pass
        


def readxyz():
    return

def project_nac():
    pass

def G():
    pass

def quasi_angular_momentum(mass, reference, changed):
    l = 0
    natom = reference.shape[-1]
    for k in range(natom):
        l += mass[k] * np.cross(reference[:,k], changed[:,k])
    return l

def eckart(reference, changed, mass, option=None):
    '''
% Rotates 'changed' to satisfy both Eckart Conditions exactly with respect to 'reference'
% Separate translational and rotational degrees of freedom from internal degrees of freedom
%
% reference: xyz coordinates as (3,NAtom)-matrix
% changed: rotated xyz coordinates as (3,NAtom)-matrix
% masses: 1D array of masses
% option: shifts COM of the returned geometry to origin if it reads 'shiftCOM'
%
% xyz_rot: changed in orientation of reference as (3,NAtom)-matrix
%
%
% Sorting of atoms has to be equal!

    Refs:
% The procedure is following: Dymarsky, Kudin, J. Chem. Phys. 122, 124103 (2005) and
% especially Coutsias, et al., J. Comput. Chem. 25, 1849 (2004).
% According to Kudin, Dymarsky, J. Chem. Phys. 122, 224105 (2005) satisfying Eckart and
% minimizing the RMSD is the same problem!
    '''

    def com(mass, atom_coord):
        '''
        return center of mass

        Params
        ------
        mass: 1d array
            atomic mass
        atom_coord: 2darray
            cartesian coordinates [3, natom]

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return np.einsum('a,ja->j', mass, atom_coord)/mass.sum()


    # Imaginary coordinates are nonsense
    # if (isreal(reference) == 0) && (isreal(changed) == 0):
    #     raise ValueError('Imaginary coordinates in the XYZ-Structures!')

    natoms = len(mass)
# % shift origin to the center of mass
# % Eckart condition of translation (Eckart 1)
    com_ref = com(mass, reference)
    com_changed = com(mass, changed)

    for i in range(natoms):
        reference[:, i] -= com_ref
        changed[:, i] -= com_changed


# % if (abs(max(max(comref))) > 1e-4)
# %     disp('Warning! Translational Eckart Condition for reference not satisfied!');
# % end



    # Quasi Angular Momentum
    # Eckart Condition of rotation (Eckart 2)
    # QAM = 0;
    # for k=1:NAtom
    #     QAM = QAM + masses(k)*cross(reference(:,k),changed(:,k));
    # end



    # Matrix A

    A = np.einsum('a, ia, ja -> ij', mass, changed, reference)

    F = np.zeros((4,4))

    F[0,0] = A[0,0] + A[1,1] + A[2,2]
    F[1,1] = A[0,0] - A[1,1] - A[2,2]
    F[2,2] = -A[0,0] + A[1,1] - A[2,2]
    F[3,3] = -A[0,0] - A[1,1] + A[2,2]

    F[1,0] = A[1,2] - A[2,1]
    F[0,1] = F[1,0]
    F[2,0] = A[2,0] - A[0,2]
    F[0,2] = F[2,0]
    F[3,0] = A[0,1] - A[1,0]
    F[0,3] = F[3,0]
    F[2,1] = A[0,1] + A[1,0]
    F[1,2] = F[2,1]
    F[3,1] = A[0,2] + A[2,0]
    F[1,3] = F[3,1]
    F[3,2] = A[1,2] + A[2,1]
    F[2,3] = F[3,2]


    # The maximum eigenvalue [and its corresponding eigenvector]
    # is the correct choice!!

    # [V,D] = eigh(F)
    # [D_, order] = sort(diag(D),'descend');
    # V = V(:,order);
    D_, V = np.linalg.eigh(F)
    idx = np.argsort(-D_)
    D_ = D_[idx]
    V = V[:,idx]

    # % [V,S,~] = svd(F);
    # % [~, order] = sort(diag(S),'descend');
    # % V = V(:,order);

    if (-D_[3] > D_[0]):
        q = V[:,3]
    else:
        q = V[:,0]


    U = np.zeros((3,3))

    U[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    U[1,1] = q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2
    U[2,2] = q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2

    U[1,0] = 2 * ( q[1] * q[2] + q[0] * q[3])
    U[2,0] = 2 * ( q[1] * q[3] - q[0] * q[2])
    U[0,1] = 2 * ( q[1] * q[2] - q[0] * q[3])
    U[2,1] = 2 * ( q[2] * q[3] + q[0] * q[1])
    U[0,2] = 2 * ( q[1] * q[3] + q[0] * q[2])
    U[1,2] = 2 * ( q[2] * q[3] - q[0] * q[1])

    if (-D_[3] > D_[0]):
        U = -U


    # Transform 'changed' with T to satisfy Eckart 2
    xyz_rot = U @ changed;

    # # Explicit test of Eckart 2
    # QAM3 = 0;
    # for k=1:NAtom
    #     QAM3 = QAM3 + masses(k)*cross(reference(:,k),xyz_rot(:,k));
    # end

    # tmp = 0;
    # for i=1:1:NAtom
    #     tmp = tmp + (norm(xyz_rot(:,i) - reference(:,i)))^2;
    # end
    # RMSD = sqrt(tmp/NAtom);

    # if (nargin < 4)
    #     xyz_rot = xyz_rot + repmat(comref',1,NAtom);
    # else
    #     if ~(strcmp(option,'shiftCOM'))
    #         xyz_rot = xyz_rot + repmat(comref',1,NAtom);

    return xyz_rot


def scan_pes(method='dft'):
    x = np.arange(0.7, 4.01, .1)
    
    mol = gto.Mole()
    if method == 'hf':
        mf_scanner = scf.RHF(mol).as_scanner()
    elif method == 'dft':
        mf_scanner = dft.RKS(mol).set(xc='b3lyp').as_scanner()

    ehf1 = []
    for b in np.arange(0.7, 4.01, 0.1):
        mol = gto.M(verbose = 5,
                    output = 'out_hf-%2.1f' % b,
                    atom = [["F", (0., 0., 0.)],
                            ["H", (0., 0., b)],],
                    basis = 'cc-pvdz')
        ehf1.append(mf_scanner(mol))
        
    import matplotlib.pyplot as plt
    plt.plot(x, ehf1, '-o', label='HF,0.7->4.0')    


if __name__ == '__main__':
    from pyscf import gto, tdscf
    # from lime.units import au2fs, au2ev
    import proplot as plt

    # mol = Molecule()
    mol = gto.Mole()
    mol.verbose = 3
    #mol.atom = [['Ne' , (0., 0., 0.)]]
    #mol.basis = {'Ne': '6-31G'}
    # This is from G2/97 i.e. MP2/6-31G*
    mol.atom = [['H' , (0,      0., 0.)],
                ['H', (1.1, 0., 0.)]]
                # ['F' , (0.91, 0., 0.)]]




    mol.basis = 'STO-3G'
    mol.build()

    geometry2 = [['H' , (0.1,      0., 0.)],
                ['H', (1.3, 0., 0.)]]
    
    mol2 = Molecule(atom=geometry2)

    print(mol2.atom_coords)
    print(mol2.com())
    mol2.molecular_frame()
    print(mol2.eckart_frame(mol.atom_coords()))

    # print(mol.natm)
    
    scan_pes()
    # mole = Molecule(mol)
    # mol.zmat(rvar=True)
    # mf = scf.RHF(mol).run()

    # td = tdscf.TDRHF(mf)
    # td.kernel()




