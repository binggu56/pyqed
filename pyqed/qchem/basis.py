#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:01:55 2024

@author: bingg
"""

import numpy as np
from gbasis.parsers import parse_gbs, make_contractions
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.kinetic_energy import kinetic_energy_integral
from gbasis.integrals.nuclear_electron_attraction import \
nuclear_electron_attraction_integral
from gbasis.integrals.electron_repulsion import electron_repulsion_integral


import os
import pyqed
import re

from numba import njit, vectorize, jit, int64, float64

@njit(float64(int64, int64, int64, float64, float64, float64), parallel=True)
def E(i,j,t,Qx,a,b):
    '''
    Recursive definition of Hermite Gaussian coefficients.

    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    i,j: orbital angular momentum number on Gaussian 'a' and 'b'
    t: number nodes in Hermite (depends on type of integral,
    e.g. always zero for overlap integrals)
    Qx: distance between origins of Gaussian 'a' and 'b'

    Refs
        https://joshuagoings.com/assets/integrals.pdf
    '''
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
        (q*Qx/a)*E(i-1,j,t,Qx,a,b) + \
        (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
        (q*Qx/b)*E(i,j-1,t,Qx,a,b) + \
        (t+1)*E(i,j-1,t+1,Qx,a,b)

@njit
def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
    for Gaussian 'a'
    lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
    A: list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
    B: list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    S3 = E(n1,n2,0,A[2]-B[2],a,b) # Z
    return S1*S2*S3*np.power(np.pi/(a+b),1.5)

def S(a,b):
    '''Evaluates overlap between two contracted Gaussians
    Returns float.
    Arguments:
    a: contracted Gaussian 'a', BasisFunction object
    b: contracted Gaussian 'b', BasisFunction object
    '''
    s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                overlap(a.exps[ia],a.shell,a.origin,
                b.exps[ib],b.shell,b.origin)
    return s


from scipy.special import factorial2
from scipy.special import hyp1f1


# @jit(int64(int64))
def fact2(n: int):
    """
    double factorial n!!

    Parameters
    ----------
    n : int
        int.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if n >= 0:
        return factorial2(n)
    elif n % 2:
        return (-1)**(abs(n+1)//2) * 1/factorial2(abs(n+2))
    else:
        raise ValueError('Factorial2 is not defined for negative even number.')


# @jit
def boys(n,T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)

class ContractedGaussian(object):
    ''' A class that contains all contracted Gaussian basis function data
    Attributes:
    origin: array/list containing the coordinates of the Gaussian origin
    shell: tuple of angular momentum
    exps: list of primitive Gaussian exponents
    coefs: list of primitive Gaussian coefficients
    norm: list of normalization factors for Gaussian primitives
    '''
    def __init__(self,origin=[0.0,0.0,0.0],shell=(0,0,0),exps=[],coefs=[]):
        self.origin = np.asarray(origin)
        self.shell = shell
        self.exps = exps
        self.coefs = coefs
        self.norm = None
        self.normalize()

    def normalize(self):
        ''' Routine to normalize the basis functions, in case they
        do not integrate to unity.
        '''
        l,m,n = self.shell
        L = l+m+n

        # self.norm is a list of length equal to number primitives
        # normalize primitives first (PGBFs)
        self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*np.power(self.exps,l+m+n+1.5)/fact2(2*l-1)/fact2(2*m-1)/fact2(2*n-1)/np.power(np.pi,1.5))
        # now normalize the contracted basis functions (CGBFs)
        # Eq. 1.44 of Valeev integral whitepaper
        prefactor = np.power(np.pi,1.5) * fact2(2*l - 1)*fact2(2*m-1)*fact2(2*n - 1)/np.power(2.0,L)

        N = 0.0
        num_exps = len(self.exps)

        for ia in range(num_exps):
            for ib in range(num_exps):
                N += self.norm[ia]*self.norm[ib]*self.coefs[ia]*self.coefs[ib]/np.power(self.exps[ia] + self.exps[ib],L+1.5)

        # print(prefactor, N)

        N = N * prefactor
        N = np.power(N,-0.5)
        for ia in range(num_exps):
            self.coefs[ia] *= N

@njit
def kinetic(a,lmn1,A,b,lmn2,B):
    ''' Evaluates kinetic energy integral between two Gaussians
    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
    for Gaussian 'a'
    lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
    A: list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
    B: list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*\
    overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
    term1 = -2*np.power(b,2)*\
    (overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) +
    overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) +
    overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) +
        m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) +
        n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))

    return term0+term1+term2

def T(a,b):
    '''Evaluates kinetic energy between two contracted Gaussians
    Returns float.
    Arguments:
    a: contracted Gaussian 'a', BasisFunction object
    b: contracted Gaussian 'b', BasisFunction object
    '''
    t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
            kinetic(a.exps[ia],a.shell,a.origin,\
            b.exps[ib],b.shell,b.origin)
    return t

# @jit
def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    ''' Returns the Coulomb auxiliary Hermite integrals
    Returns a float.
    Arguments:
    t,u,v: order of Coulomb Hermite derivative in x,y,z
    (see defs in Helgaker and Taylor)
    n: order of Boys function
    PCx,y,z: Cartesian vector distance between Gaussian
    composite center P and nuclear center C
    RPC: Distance between P and C
    '''
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

@njit
def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)


def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
    ''' Evaluates kinetic energy integral between two Gaussians
    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
    for Gaussian 'a'
    lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
    A: list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
    B: list containing origin of Gaussian 'b'
    C: list containing origin of nuclear center 'C'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    p = a + b
    P = gaussian_product_center(a,A,b,B) # Gaussian composite center
    RPC = np.linalg.norm(P-C)
    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                val += E(l1,l2,t,A[0]-B[0],a,b) * \
                    E(m1,m2,u,A[1]-B[1],a,b) * \
                    E(n1,n2,v,A[2]-B[2],a,b) * \
                    R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    val *= 2*np.pi/p
    return val




def point_charge(a,b,C):
    '''Evaluates electron-nuclear attraction

    $%overlap between two contracted Gaussians

    Returns float.
    Arguments:
    a: contracted Gaussian 'a', BasisFunction object
    b: contracted Gaussian 'b', BasisFunction object
    C: center of nucleus
    '''
    v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                nuclear_attraction(a.exps[ia],a.shell,a.origin,
                b.exps[ib],b.shell,b.origin,C)
    return v


def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    ''' Evaluates kinetic energy integral between two Gaussians
    Returns a float.
    a,b,c,d: orbital exponent on Gaussian 'a','b','c','d'
    lmn1,lmn2
    lmn3,lmn4: int tuple containing orbital angular momentum
    for Gaussian 'a','b','c','d', respectively
    A,B,C,D: list containing origin of Gaussian 'a','b','c','d'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
    q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
    alpha = p*q/(p+q)
    P = gaussian_product_center(a,A,b,B) # A and B composite center
    Q = gaussian_product_center(c,C,d,D) # C and D composite center
    RPQ = np.linalg.norm(P-Q)
    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            val += E(l1,l2,t,A[0]-B[0],a,b) * \
                                E(m1,m2,u,A[1]-B[1],a,b) * \
                                E(n1,n2,v,A[2]-B[2],a,b) * \
                                E(l3,l4,tau,C[0]-D[0],c,d) * \
                                E(m3,m4,nu ,C[1]-D[1],c,d) * \
                                E(n3,n4,phi,C[2]-D[2],c,d) * \
                                np.power(-1,tau+nu+phi) * \
                                R(t+tau,u+nu,v+phi,0,\
                                alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
    return val

def ERI(a,b,c,d):
    '''Evaluates overlap between two contracted Gaussians
    Returns float.
    Arguments:
    a: contracted Gaussian 'a', BasisFunction object
    b: contracted Gaussian 'b', BasisFunction object
    c: contracted Gaussian 'b', BasisFunction object
    d: contracted Gaussian 'b', BasisFunction object
    '''
    eri = 0.0
    for ja, ca in enumerate(a.coefs):
        for jb, cb in enumerate(b.coefs):
            for jc, cc in enumerate(c.coefs):
                for jd, cd in enumerate(d.coefs):
                    eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                    ca*cb*cc*cd*\
                    electron_repulsion(a.exps[ja],a.shell,a.origin,\
                    b.exps[jb],b.shell,b.origin,\
                    c.exps[jc],c.shell,c.origin,\
                    d.exps[jd],d.shell,d.origin)
    return eri

ALIAS = {
    '631g'       : '6-31g.1.gbs',
    'sto3g'      : "sto-3g.1.gbs",
    'sto6g'      : 'sto-6g.1.gbs',
    '631g**'     : "6-31g_st__st_.0.gbs",
    '6311g**'    : "6-311g_st__st_.0.gbs",
    '6311g'      : "6-311g.0.gbs",
    '631g++'     : "/6-31g++.gbs",
    'ccpvdz'     : 'cc-pvdz.0.gbs'    ,
    'ccpvtz'     : 'cc-pvtz.dat'    ,
    'ccpvqz'     : 'cc-pvqz.dat'    ,
    'ccpv5z'     : 'cc-pv5z.dat'    ,
    'ccpvdpdz'   : 'cc-pvdpdz.dat'  ,
    'augccpvdz'  : 'aug-cc-pvdz.dat',
    'augccpvtz'  : 'aug-cc-pvtz.dat',
    'augccpvqz'  : 'aug-cc-pvqz.dat',
    'augccpv5z'  : 'aug-cc-pv5z.dat',
    'augccpvdpdz': 'aug-cc-pvdpdz.dat',
    'ccpvdzdk'   : 'cc-pvdz-dk.dat' ,
    'ccpvtzdk'   : 'cc-pvtz-dk.dat' ,
    'ccpvqzdk'   : 'cc-pvqz-dk.dat' ,
    'ccpv5zdk'   : 'cc-pv5z-dk.dat' ,
    'ccpvdzdkh'  : 'cc-pvdz-dk.dat' ,
    'ccpvtzdkh'  : 'cc-pvtz-dk.dat' ,
    'ccpvqzdkh'  : 'cc-pvqz-dk.dat' ,
    'ccpv5zdkh'  : 'cc-pv5z-dk.dat' ,
}

def build(mol):
    """
    build electronic integrals in AO using GBasis package

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from gbasis.parsers import parse_gbs, make_contractions

    atoms = mol.atom_symbols()
    atcoords = mol.atom_coords()
    atnums = mol.atom_charges()

    basis_dir = os.path.abspath(f'{pyqed.__file__}/../qchem/basis_set/')

    if isinstance(mol.basis, str):

        basis_dict = parse_gbs(basis_dir + '/' + ALIAS[mol.basis.replace('-','').lower()])
        basis = make_contractions(basis_dict, atoms, atcoords, coord_types="p")
    else:

        raise NotImplementedError('Customized basis not supported yet.')

    # To obtain the total number of AOs we check for each shell its angular momentum and coordinate type
    total_ao = 0
    for shell in basis:
        if shell.coord_type == "cartesian":
            total_ao += shell.angmom_components_cart.shape[0]
        elif shell.coord_type == "spherical":
            total_ao += len(shell.angmom_components_sph)

    mol.nao = total_ao

    print("Number of AOs = ", mol.nao)

    # compute overlap integrals in AO basis
    mol.overlap = overlap_integral(basis)


    # olp_mo = overlap_integral(basis, transform=mo_coeffs.T)

    # compute kinetic energy integrals in AO basis
    k_int1e = kinetic_energy_integral(basis)
    # print("Shape kinetic energy integral: ", k_int1e.shape, "(#AO, #AO)")


    # compute nuclear-electron attraction integrals in AO basis
    # atnums = np.array([1,1])
    nuc_int1e = nuclear_electron_attraction_integral(
            basis, atcoords, atnums)
    # print("Shape Nuclear-electron integral: ", nuc_int1e.shape, "(#AO, #AO)")

    mol.hcore = k_int1e + nuc_int1e

    #Compute e-e repulsion integral in MO basis, shape=(#MO, #MO, #MO, #MO)
    int2e_mo = electron_repulsion_integral(basis, notation='chemist')
    mol.eri = int2e_mo

    mol._bas = basis

    return


def parse_gbs(gbs_basis_file):
    """Parse Gaussian94 basis set file.

    Parameters
    ----------
    gbs_basis_file : str
        Path to the Gaussian94 basis set file.

    Returns
    -------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Dictionary of the element to the list of angular momentum, exponents, and contraction
        coefficients associated with each contraction at the given atom.

    Notes
    -----
    Angular momentum symbol is hard-coded into this function. This means that if the selected basis
    set has an angular momentum greater than "k", an error will be raised.

    Since Gaussian94 basis format does not explicitly state which contractions are generalized, we
    infer that subsequent contractions belong to the same generalized shell if they have the same
    exponents and angular momentum. If two contractions are not one after another or if they are
    associated with more than one angular momentum, they are treated to be segmented contractions.

    """
    # pylint: disable=R0914
    with open(gbs_basis_file) as basis_fh:
        gbs_basis = basis_fh.read()
    # splits file into 'element', 'basis stuff', 'element',' basis stuff'
    # e.g., ['H','stuff with exponents & coefficients\n', 'C', 'stuff with etc\n']
    data = re.split(r"\n\s*(\w[\w]?)\s+\w+\s*\n", gbs_basis)
    dict_angmom = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
    # remove first part
    if "\n" in data[0]:  # pragma: no branch
        data = data[1:]
    # atoms: stride of 2 get the ['H','C', etc]. basis: take strides of 2 to skip elements
    atoms = data[::2]
    basis = data[1::2]
    # trim out headers at the end
    output = {}
    for atom, shells in zip(atoms, basis):
        output.setdefault(atom, [])

        shells = re.split(r"\n?\s*(\w+)\s+\w+\s+\w+\.\w+\s*\n", shells)
        # remove the ends
        atom_basis = shells[1:]
        # get angular momentums
        angmom_shells = atom_basis[::2]
        # get exponents and coefficients
        exps_coeffs_shells = atom_basis[1::2]

        for angmom_seg, exp_coeffs in zip(angmom_shells, exps_coeffs_shells):
            angmom_seg = [dict_angmom[i.lower()] for i in angmom_seg]
            exps = []
            coeffs_seg = []
            exp_coeffs = exp_coeffs.split("\n")
            for line in exp_coeffs:
                test = re.search(
                    r"^\s*([0-9\.DE\+\-]+)\s+((?:(?:[0-9\.DE\+\-]+)\s+)*(?:[0-9\.DE\+\-]+))\s*$",
                    line,
                )
                try:
                    exp, coeff_seg = test.groups()
                    coeff_seg = re.split(r"\s+", coeff_seg)
                except AttributeError:
                    continue
                # clean up
                exp = float(exp.lower().replace("d", "e"))
                coeff_seg = [float(i.lower().replace("d", "e")) for i in coeff_seg if i is not None]
                exps.append(exp)
                coeffs_seg.append(coeff_seg)
            exps = np.array(exps)
            coeffs_seg = np.array(coeffs_seg)
            # if len(angmom_seg) == 1:
            #     coeffs_seg = coeffs_seg[:, None]
            for i, angmom in enumerate(angmom_seg):
                # ensure previous and current exps are same length before using np.allclose()
                if output[atom] and len(output[atom][-1][1]) == len(exps):
                    # check if current exp's should be added to previous generalized contraction
                    hstack = np.allclose(output[atom][-1][1], exps)
                else:
                    hstack = False
                if output[atom] and output[atom][-1][0] == angmom and hstack:
                    output[atom][-1] = (
                        angmom,
                        exps,
                        np.hstack([output[atom][-1][2], coeffs_seg[:, i : i + 1]]),
                    )
                else:
                    output[atom].append((angmom, exps, coeffs_seg[:, i : i + 1]))

    return output


def make_contractions(basis_dict, atoms, coords, coord_types):
    """Return the contractions that correspond to the given atoms for the given basis.

    Parameters
    ----------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Output of the parsers from gbasis.parsers.
    atoms : N-list/tuple of str
        Atoms at which the contractions are centered.
    coords : np.ndarray(N, 3)
        Coordinates of each atom.
    coord_types : {"cartesian"/"c", list/tuple of "cartesian"/"c" or "spherical"/"p", "spherical"/"p"}
        Types of the coordinate system for the contractions.
        If "cartesian" or "c", then all of the contractions are treated as Cartesian contractions.
        If "spherical" or "p", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" (or "c") or "spherical" (or "p") to specify the
        coordinate type of each `GeneralizedContractionShell` instance.
        Default value is "spherical".

    Returns
    -------
    basis : tuple of GeneralizedContractionShell
        Contractions for each atom.
        Contractions are ordered in the same order as in the values of `basis_dict`.

    Raises
    ------
    TypeError
        If `atoms` is not a list or tuple of strings.
        If `coords` is not a two-dimensional `numpy` array with 3 columns.
        If `tol` is not a float.
        If `ovr` is not boolean
    ValueError
        If the length of atoms is not equal to the number of rows of `coords`.

    """
    if not (isinstance(atoms, (list, tuple)) and all(isinstance(i, str) for i in atoms)):
        raise TypeError("Atoms must be provided as a list or tuple.")
    if not (isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3):
        raise TypeError(
            "Coordinates must be provided as a two-dimensional `numpy` array with three columns."
        )

    if len(atoms) != coords.shape[0]:
        raise ValueError("Number of atoms must be equal to the number of rows in the coordinates.")

    basis = []
    # expected number of coordinates
    num_coord_types = sum([len(basis_dict[i]) for i in atoms])

    # check and assign coord_types
    if isinstance(coord_types, str):
        if coord_types not in ["c", "cartesian", "p", "spherical"]:
            raise ValueError(
                f"If coord_types is a string, it must be either 'spherical'/'p' or 'cartesian'/'c'."
                f"got {coord_types}"
            )
        coord_types = [coord_types] * num_coord_types

    if len(coord_types) != num_coord_types:
        raise ValueError(
            f"If coord_types is a list, it must be the same length as the total number of contractions."
            f"got {len(coord_types)}"
        )

    # make shells

    for icenter, (atom, coord) in enumerate(zip(atoms, coords)):
        for angmom, exps, coeffs in basis_dict[atom]:

            for shell in _shell(angmom):

                basis.append(
                    # GeneralizedContractionShell(
                    #     angmom,
                    #     coord,
                    #     coeffs,
                    #     exps,
                    #     coord_types.pop(0),
                    #     icenter=icenter,
                    ContractedGaussian(coord, shell, exps, coeffs)
                    #)
                )
    return tuple(basis)

def _shell(l):
    if l == 0:
        return [(0,0,0)]
    elif l == 1:
        return [(1,0,0), [0,1,0], [0,0,1]]



if __name__=='__main__':

    # kin_e = np.trace(dm.dot(k_int1e))
    # print("Kinetic energy (Hartree):", kin_e)

    import time

    # Define atomic symbols and coordinates (i.e., basis function centers)
    atoms = ["F", "F"]
    atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    basis_dir = os.path.abspath(f'{pyqed.__file__}/../qchem/basis_set/')

    basis_dict = parse_gbs(basis_dir+'/6-31g.1.gbs')
    basis = make_contractions(basis_dict, atoms, atcoords, 'c')

    # print([g.exps for g in basis])

    print(len(basis))

    # # To obtain the total number of AOs we compute the cartesian components for each angular momentum
    # total_ao = 0
    # print(f"Number of generalized shells: {len(basis)}") # Output 6
    # for shell in basis:
    #     total_ao += shell.angmom_components_cart.shape[0]

    # print("Total number of AOs: ", total_ao) # output 10


    # myOrigin = [0.0, 0.0, 0.0]
    # myShell = (0,0,0) # p‐orbitals would be (1,0,0) or (0,1,0) or (0,0,1), etc.
    # myExps = [3.42525091, 0.62391373, 0.16885540]
    # myCoefs = [0.15432897, 0.53532814, 0.44463454]
    # a = ContractedGaussian(origin=myOrigin,shell=myShell,exps=myExps,coefs=myCoefs)


    # H2 = [0.0, 0.0, 1.0]
    # myShell = (0,0,0) # p‐orbitals would be (1,0,0) or (0,1,0) or (0,0,1), etc.
    # myExps = [3.42525091, 0.62391373, 0.16885540]
    # myCoefs = [0.15432897, 0.53532814, 0.44463454]
    # b = ContractedGaussian(origin=H2,shell=myShell,exps=myExps,coefs=myCoefs)

    # basis = [a, b]

    def ao_ints(basis, coords):
        nao = len(basis)
        natom, _ = coords.shape

        s = np.eye(nao)
        for i in range(nao):
            for j in range(i):
                s[i,j] = S(basis[i], basis[j])
                s[j,i] = s[i,j]

        t = np.zeros((nao, nao))
        for i in range(nao):
            for j in range(i+1):
                t[i,j] = T(basis[i], basis[j])
                if i != j: t[j,i] = t[i,j]

        v = np.zeros((nao,nao))
        for i in range(nao):
            for j in range(i+1):
                for C in range(natom):
                    v[i,j] -= point_charge(basis[i], basis[j], coords[C])
                if i != j: v[j,i] = v[i,j]

        eri = np.zeros((nao, nao, nao, nao))
        for p in range(nao):
            for q in range(p):
                for r in range(nao):
                    for s in range(r):
                        eri[p,q,r,s] = ERI(basis[p], basis[q], basis[r], basis[s])

        return s, t, v, eri

    # print(basis[0].exps, basis[0].coefs)
    print(atcoords[1])
    print(point_charge(basis[0], basis[0], atcoords[1]))

    start_time = time.time()
    ao_ints(basis, atcoords)
    end_time = time.time()

    print(end_time-start_time)

    # s,t, v, eri = ao_ints(basis, atcoords)
    # print(t)
    # point_charge(a, a, myOrigin))
    # print(v)

    # from pyqed.qchem import Molecule
    # mol = Molecule(atom = [
    # ['H' , (0. , 0. , 0)],
    # ['H' , (0. , 0. , 1.)], ], basis='631g')
    # mol.build()

    # print(mol._bas[0].)

    # mol.build()
    # print(mol.eri)