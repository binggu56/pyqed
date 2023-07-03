#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:19:28 2022

%
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
%
% The procedure is following: Dymarsky, Kudin, J. Chem. Phys. 122, 124103 (2005) and
% especially Coutsias, et al., J. Comput. Chem. 25, 1849 (2004).
% According to Kudin, Dymarsky, J. Chem. Phys. 122, 224105 (2005) satisfying Eckart and
% minimizing the RMSD is the same problem!
%

@author: bing
"""

# function [xyz_rot,U,RMSD] = SatisfyEckart(reference, changed, masses, option)
# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import division

import math
from numpy import sin, cos, pi
from numpy.linalg import norm

# Suppress scientific notation printouts and change default precision
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


import numpy as np
import scipy.linalg as linalg
from scipy.optimize import newton

from pyscf.lib import logger
import pyscf.ao2mo
import pyscf
from functools import reduce


from pyqed.phys import eig_asymm, is_positive_def, dag
from pyqed.optics import Pulse


def intertia_moment(mass, coords):
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    coords = coords - mass_center
    im = np.einsum('i,ij,ik->jk', mass, coords, coords)
    im = np.eye(3) * im.trace() - im
    return im

#class Geometry(pyscf.gto.Mole):
class Geometry:
    def __init__(self, mol, *args):

        super().__init__(*args)

        # self.atom_coord = mol.atom_coord
        self.atom_coords = mol.atom_coords() # shape 3, natoms
        self.natom = mol.natm
        self.mass = mol.atom_mass_list()
        self.atom_symbol = [mol.atom_symbol(i) for i in range(self.natom)]


        self.distmat = None

    # def atom_symbol(self):
    #     return [mol.atom_symbol(i) for i in range(self.natoms)]

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
        # if mass is None:
        mass = self.mass
        # if coords is None:
        coords = self.atom_coords()
        return intertia_moment(mass, coords)

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

if __name__ == '__main__':
    from pyscf import scf, gto, tdscf
    from lime.units import au2fs, au2ev
    import proplot as plt

    mol = Molecule()
    mol.verbose = 3
    #mol.atom = [['Ne' , (0., 0., 0.)]]
    #mol.basis = {'Ne': '6-31G'}
    # This is from G2/97 i.e. MP2/6-31G*
    mol.atom = [['H' , (0,      0., 0.)],
                ['H', (1.1, 0., 0.)]]
                # ['F' , (0.91, 0., 0.)]]


    mol.basis = 'STO-3G'
    mol.build()

    # print(mol.natm)

    # mole = Molecule(mol)
    # mol.zmat(rvar=True)
    mf = scf.RHF(mol).run()

    td = tdscf.TDRHF(mf)
    td.kernel()





def readxyz():
    return

def project_nac():
    pass

def G():
    pass

def eckart(reference, changed, masses, option):

    natom = reference.shape[-1]

# NAtom = size(reference,2);      % Number of atoms

    # Imaginary coordinates are nonsense
    if (isreal(reference) == 0) && (isreal(changed) == 0):
        raise ValueError('Imaginary coordinates in the XYZ-Structures!')


# % shift origin to the center of mass
# % Eckart condition of translation (Eckart 1)
comref = masses(:)' * reference'./ sum( masses(:));
reference = reference - comref

# % if (abs(max(max(comref))) > 1e-4)
# %     disp('Warning! Translational Eckart Condition for reference not satisfied!');
# % end

com = masses(:)' * changed'./ sum( masses(:));
changed = changed - repmat(com',1,NAtom);


% Quasi Angular Momentum
% Eckart Condition of rotation (Eckart 2)
QAM = 0;
for k=1:NAtom
    QAM = QAM + masses(k)*cross(reference(:,k),changed(:,k));
end



% Matrix A

A = zeros(3,3);

for i=1:3   % Loop over Cartesian index
    for j=1:3   % Loop over Cartesian index
        for a=1:NAtom   % Loop over Atoms
            A(i,j) = A(i,j) + masses(a)*changed(i,a)*reference(j,a);
        end
    end
end

F = zeros(4,4);

F(1,1) = A(1,1) + A(2,2) + A(3,3);
F(2,2) = A(1,1) - A(2,2) - A(3,3);
F(3,3) = -A(1,1) + A(2,2) - A(3,3);
F(4,4) = -A(1,1) - A(2,2) + A(3,3);

F(2,1) = A(2,3) - A(3,2);
F(1,2) = F(2,1);
F(3,1) = A(3,1) - A(1,3);
F(1,3) = F(3,1);
F(4,1) = A(1,2) - A(2,1);
F(1,4) = F(4,1);
F(3,2) = A(1,2) + A(2,1);
F(2,3) = F(3,2);
F(4,2) = A(1,3) + A(3,1);
F(2,4) = F(4,2);
F(4,3) = A(2,3) + A(3,2);
F(3,4) = F(4,3);


% The maximum eigenvalue (and its corresponding eigenvector) is the correct choice!!

[V,D] = eig(F);
[D_, order] = sort(diag(D),'descend');
V = V(:,order);

% [V,S,~] = svd(F);
% [~, order] = sort(diag(S),'descend');
% V = V(:,order);

if (-D_(4) > D_(1))
    q = V(:,4);
else
    q = V(:,1);
end

U = zeros(3,3);

U(1,1) = (q(1)^2 + q(2)^2 - q(3)^2 - q(4)^2);
U(2,2) = (q(1)^2 + q(3)^2 - q(2)^2 - q(4)^2);
U(3,3) = (q(1)^2 + q(4)^2 - q(2)^2 - q(3)^2);

U(2,1) = 2 * ( q(2) * q(3) + q(1) * q(4));
U(3,1) = 2 * ( q(2) * q(4) - q(1) * q(3));
U(1,2) = 2 * ( q(2) * q(3) - q(1) * q(4));
U(3,2) = 2 * ( q(3) * q(4) + q(1) * q(2));
U(1,3) = 2 * ( q(2) * q(4) + q(1) * q(3));
U(2,3) = 2 * ( q(3) * q(4) - q(1) * q(2));

if (-D_(4) > D_(1))
    U = -U;
end

% Transform 'changed' with T to satisfy Eckart 2
xyz_rot = U * changed;

% Explicit test of Eckart 2
QAM3 = 0;
for k=1:NAtom
    QAM3 = QAM3 + masses(k)*cross(reference(:,k),xyz_rot(:,k));
end

tmp = 0;
for i=1:1:NAtom
    tmp = tmp + (norm(xyz_rot(:,i) - reference(:,i)))^2;
end
RMSD = sqrt(tmp/NAtom);

if (nargin < 4)
    xyz_rot = xyz_rot + repmat(comref',1,NAtom);
else
    if ~(strcmp(option,'shiftCOM'))
        xyz_rot = xyz_rot + repmat(comref',1,NAtom);
    end
end


end