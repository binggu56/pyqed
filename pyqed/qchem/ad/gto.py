# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:48:35 2017


Full CI calculation for H2

Main structure:
    1. functions dealing with all integrals
    2. functions transform MO integrals to AO, and then AO to (Gaussian type orbitals) GTOs
    3. Constuct configurations and Hamiltonian matrix

@author: bingg
"""
import numpy as np
from jax.scipy.special import erf
from numpy import sqrt, exp
import scipy

import jax.numpy as jnp
from jax.numpy.linalg import vector_norm as norm
from jax import grad, hessian, value_and_grad, vmap

from pyqed import qchem, dagger
import os
import re

pi = np.pi


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
        return jnp.exp(-q*Qx*Qx) # K_AB
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

def overlap(a,lmn1,A,b,lmn2,B):
    '''
    Evaluates overlap integral between two Gaussians
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
    return S1*S2*S3*jnp.power(jnp.pi/(a+b),1.5)

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

from scipy.special import hyp1f1
# from jax import jit
# from scipy.special import factorial2

def factorial2(n):
    """
    Computes the double factorial of a non-negative integer n.
    n!! = n * (n-2) * (n-4) * ...
    For n=0 or n=-1, n!! = 1 by definition.
    """
    if n < 0:
        return jnp.array(0, dtype=jnp.int32) # Or handle as complex extension if needed

    # Handle base cases
    if n == 0 or n == -1:
        return jnp.array(1, dtype=jnp.int32)

    # Generate a sequence of numbers for multiplication
    # For odd n: n, n-2, ..., 1
    # For even n: n, n-2, ..., 2
    step_val = 2

    # Create the sequence of numbers to multiply
    # JAX does not support dynamic array sizes in JIT, so we need a fixed-size approach.
    # We can use jnp.arange and then filter.

    # Calculate the number of terms in the product
    # num_terms = (n // 2) + (1 if n % 2 == 1 else 0)

    # Generate the sequence of numbers
    # We need to ensure that the sequence is positive
    sequence = jnp.arange(n, 0, -step_val)

    # Compute the product
    return jnp.prod(sequence)



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
    term1 = -2*jnp.power(b,2)*\
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


def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)

def electron_nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
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
                electron_nuclear_attraction(a.exps[ia],a.shell,a.origin,
                b.exps[ib],b.shell,b.origin,C)
    return v

# def point_charge(a,b,C, gradient=False):
#     '''Evaluates point charge integral as in electron-nuclear attraction

#     .. math::

#         v(R_C) = \langle g_A 1/|r - R_C| g_B \rangle

#     Returns float.
#     Arguments:
#     a: contracted Gaussian 'a', BasisFunction object
#     b: contracted Gaussian 'b', BasisFunction object
#     C: center of nucleus
#     '''
#     # if not gradient:

#     v = 0.0
#     for ia, ca in enumerate(a.coefs):
#         for ib, cb in enumerate(b.coefs):
#             v += a.norm[ia]*b.norm[ib]*ca*cb*\
#                 nuclear_attraction(a.exps[ia],a.shell,a.origin,
#                 b.exps[ib],b.shell,b.origin,C)
#     return v

    # else:

    #     dv = value_and_grad(nuclear_attraction, argnums=-1)
    #     ddv = hessian(nuclear_attraction, argnums=-1)

    #     v = 0
    #     f = 0
    #     g = 0

    #     for ia, ca in enumerate(a.coefs):
    #         for ib, cb in enumerate(b.coefs):

    #             _v, _f = dv(a.exps[ia],a.shell,a.origin,
    #                 b.exps[ib],b.shell,b.origin, C)

    #             _g = ddv(a.exps[ia],a.shell,a.origin,
    #                     b.exps[ib],b.shell,b.origin,C)

    #             prefactor = a.norm[ia]*b.norm[ib]*ca*cb
    #             v += prefactor * _v
    #             f += prefactor * _f
    #             g += prefactor * _g

    #     return v, f, g




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

    val *= 2*jnp.power(np.pi,2.5)/(p*q*jnp.sqrt(p+q))
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

# def build(mol):
#     """
#     build electronic integrals in AO using GBasis package

#     Parameters
#     ----------
#     mol : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     from gbasis.parsers import parse_gbs, make_contractions

#     atoms = mol.atom_symbols()
#     atcoords = mol.atom_coords()
#     atnums = mol.atom_charges()

#     basis_dir = os.path.abspath(f'{pyqed.__file__}/../qchem/basis_set/')

#     if isinstance(mol.basis, str):

#         basis_dict = parse_gbs(basis_dir + '/' + ALIAS[mol.basis.replace('-','').lower()])
#         basis = make_contractions(basis_dict, atoms, atcoords, coord_types="p")
#     else:

#         raise NotImplementedError('Customized basis not supported yet.')

#     # To obtain the total number of AOs we check for each shell its angular momentum and coordinate type
#     total_ao = 0
#     for shell in basis:
#         if shell.coord_type == "cartesian":
#             total_ao += shell.angmom_components_cart.shape[0]
#         elif shell.coord_type == "spherical":
#             total_ao += len(shell.angmom_components_sph)

#     mol.nao = total_ao

#     print("Number of AOs = ", mol.nao)

#     # compute overlap integrals in AO basis
#     mol.overlap = overlap_integral(basis)


#     # olp_mo = overlap_integral(basis, transform=mo_coeffs.T)

#     # compute kinetic energy integrals in AO basis
#     k_int1e = kinetic_energy_integral(basis)
#     # print("Shape kinetic energy integral: ", k_int1e.shape, "(#AO, #AO)")


#     # compute nuclear-electron attraction integrals in AO basis
#     # atnums = np.array([1,1])
#     nuc_int1e = nuclear_electron_attraction_integral(
#             basis, atcoords, atnums)
#     # print("Shape Nuclear-electron integral: ", nuc_int1e.shape, "(#AO, #AO)")

#     mol.hcore = k_int1e + nuc_int1e

#     #Compute e-e repulsion integral in MO basis, shape=(#MO, #MO, #MO, #MO)
#     int2e_mo = electron_repulsion_integral(basis, notation='chemist')
#     mol.eri = int2e_mo

#     mol._bas = basis

#     return


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
    """
    Return the contractions that correspond to the given atoms for the given basis.

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
                print('shell', shell)

                basis.append(
                    # GeneralizedContractionShell(
                    #     angmom,
                    #     coord,
                    #     coeffs,
                    #     exps,
                    #     coord_types.pop(0),
                    #     icenter=icenter,
                    ContractedGaussian(origin=coord, shell=shell, exps=exps, coefs=coeffs)
                    #)
                )
    return tuple(basis)

def _shell(l):
    if l == 0:
        return [(0,0,0)]
    elif l == 1:
        return [(1,0,0), [0,1,0], [0,0,1]]







class Molecule(qchem.Molecule):

    def build(self):
        """
        build AO integrals using raw implementation
        """
        import pyqed
        # basis = []
        # nuclear_coor = []
        # for atm_id in range(self.natom):
        #     basis.append(sto3g_hydrogen(center=self.atom_coord(atm_id)))
        #     nuclear_coor.append(self.atom_coord(atm_id))

        # self.basis = basis
        # self.nuclear_coor = nuclear_coor
        # self.nao = self.nbas = len(basis)

        atoms = self.atom_symbols()
        atcoords = self.atom_coords()
        atnums = self.atom_charges()

        basis_dir = os.path.abspath(f'{pyqed.__file__}/../qchem/basis_set/')


        if isinstance(self.basis, str):

            basis_dict = parse_gbs(basis_dir + '/' + ALIAS[self.basis.replace('-','').lower()])
            basis = make_contractions(basis_dict, atoms, atcoords, coord_types="p")
        else:

            raise NotImplementedError('Customized basis not supported yet.')

        s, t, v, eri = build(basis, atcoords, self.atom_charges())

        self.nao = len(basis)
        self.overlap = s
        self.hcore = t + v
        self.eri = eri

        return self

    def nuc_grad(self):
        return Gradient(self)

    def electron_nuclear_attraction(self, atm_id, gradient=(1,2), argnums=0):

        """calculate the nuclear-electron attraction and first-order and second-order derivative

        .. math::

            D_A = \nabla_{R_A} \braket{\mu_j | \sum_C \frac{-Z_C}{\abs(r - R_C)} | \nu_k}

        Parameters
        ----------
        i : number
            atom coordinate index
        j : number
            basis index
        k : number
            basis
        gradient : tuple, optional
            the number in the tuple represent the order of derivative
            e.g. (1,2) represent first- and second- order derivative
            e.g. (0,) represent no derivative
        argnums : int, optional
            _description_, by default 0
            argnums: (Rc, alpha, Ra, beta, Rb)
            argnums = 0 means take derivative with Rc

        Returns
        -------
        V: (nao, nao)
            energy
        D1: (nao, nao, 3) first-order gradients
        D2: (nao, nao, 3,3) second-order gradients
        """

        Rc = self.atom_coord(atm_id)
        Zc = self.atom_charge(atm_id)

        nao = self.nao
        natom = self.natom

        V = np.zeros((nao, nao))

        D1 = np.zeros((nao, nao, 3))
        D2 = np.zeros((nao, nao, 3, 3))

        for j in range(self.nao):
            for k in range(j+1):

                bj = self.basis[j]
                bk = self.basis[k]


                V[j,k], D1[j, k], D2[j,k] = nuclear_attraction_integral(Rc, bj, bk, gradient, argnums)

                if j != k:
                    D1[k, j] = D1[j, k]
                    D2[k, j] = D2[j, k]

        return Zc * V, Zc * D1, Zc * D2


    def overlap_integral(self, gradient=False):

        nao = self.nao

        s = jnp.zeros((nao, nao))
        for i in range(nao):
            for j in range(i):
                tmp = S(self.basis[i], self.basis[j])
                s = s.at[i,j].set(tmp[0])
        s = s + s.T
        s = jnp.fill_diagonal(s, 1, inplace=False)
        return s

    def electron_repulsion_integral(self):
        pass

    def nuclear_repulsion(self):
        pass

def nuclear_repulsion(atcoords, atnums):
    # Compute Nucleus-Nucleus repulsion
    rab = jnp.triu(norm(atcoords[:, None]- atcoords, axis=-1))
    at_charges = jnp.triu(atnums[:, None] * atnums)[jnp.where(rab > 0)]
    nn_e = jnp.sum(at_charges / rab[rab > 0])
    return nn_e

def grad_nuc(mol, atmlst=None):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates
    '''
    z = mol.atom_charges()
    r = mol.atom_coords()
    dr = r[:,None,:] - r
    dist = np.linalg.norm(dr, axis=2)
    diag_idx = np.diag_indices(z.size)
    dist[diag_idx] = 1e100
    rinv = 1./dist
    rinv[diag_idx] = 0.
    gs = np.einsum('i,j,ijx,ij->ix', -z, z, dr, rinv**3)
    if atmlst is not None:
        gs = gs[atmlst]
    return gs

class Gradient:
    def __init__(self, mf_or_mol):
        if isinstance(mf_or_mol, Molecule):
            self.mol = mf_or_mol
        else:
            self.mol = mf_or_mol.mol

        # self.mol = mf_or_mol

        self.atom_coords = self.mol.atom_coords()


    def point_charge(self, C):

        dv = grad(electron_nuclear_attraction)
        ddv = hessian(electron_nuclear_attraction)

        for a in self.basis:
            for b in self.basis:
                v = 0.0
                for ia, ca in enumerate(a.coefs):
                    for ib, cb in enumerate(b.coefs):
                        v += a.norm[ia]*b.norm[ib]*ca*cb*\
                            dv(a.exps[ia],a.shell,a.origin,
                            b.exps[ib],b.shell,b.origin,C)

        return v




    def electron_nuclear_attraction(self):
        """
        .. math::

            F^I_{\mu \nu} = \langle \mu | \nabla V_{eN}(R) | \nu \rangle

        Returns
        -------
        f : TYPE
            DESCRIPTION.
        g : TYPE
            DESCRIPTION.

        """


        R = self.atom_coords

        dv = grad(electron_nuclear_attraction, argnums=-1)

        f = vmap(dv, in_axes=[0, None])(self.x, R)

        ddv = hessian(electron_nuclear_attraction, argnums=-1)
        g = vmap(ddv, in_axes=[0, None])(self.x, R)

        v = jnp.zeros((nao,nao))
        for i in range(nao):
            for j in range(i+1):
                tmp = 0
                for C in range(natom):
                    tmp -= Z[C] * point_charge(basis[i], basis[j], coords[C])[0]

                v = v.at[i,j].set(tmp)
                if i != j:
                    v = v.at[j,i].set(tmp)

        return f, g

    def nuclear_repulsion(self):
        R = self.atom_coords
        Z = self.mol.atom_charges()

        v, dv = value_and_grad(nuclear_repulsion, argnums=0)(R, Z)

        ddv = hessian(nuclear_repulsion, argnums=0)(R, Z)

        return dv, ddv
# class Gaussian:
#     def __init__(self, alpha, center, i=0, j=0, k=0, cartesian=True):
#         """
#         Gaussian
#         .. math::
#             \Phi(x,y,z; \alpha,i,j,k)=\left({\frac {2\alpha }{\pi }}\right)^{3/4}\left[{\frac {(8\alpha )^{i+j+k}i!j!k!}{(2i)!(2j)!(2k)!}}]^{1/2}
#                     x^{i}y^{j}z^{k} e^{-\alpha (x^{2}+y^{2}+z^{2})}}
#         """
#         self.center = center
#         self.alpha = alpha
#         self.i = i
#         self.j = j
#         self.k = k

# class ContractedGaussian:
#     def __init__(self,n,d,g):
#         """
#         contracted Gaussians
#         .. math::
#             \phi = \sum_i=1^n d_i g_i

#         d : contraction coeffiecents
#         g : primative gaussians
#         """
#         self.n = n
#         self.d = d
#         self.g = g

#         return

class STONG:
    def __init__(self,n,d,g):
        """
        Slater Type Orbital fitted with N primative gausians (STO-NG) type basis

        d : contraction coeffiecents
        g : primative gaussians
        """
        self.n = n
        self.d = d
        self.g = g

        return


# def sto3g(center, zeta):
#     """
#     Builds a STO-3G basis that best approximates a single slater type
#     orbital with Slater orbital exponent zeta

#     Parameters
#     ----------
#     center : TYPE
#         DESCRIPTION.
#     zeta : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """

#     scaling = zeta**2
#     return STONG(3,[0.444635, 0.535328, 0.154329],
#             [Gaussian(scaling*.109818, center),
#              Gaussian(scaling*.405771, center),
#              Gaussian(scaling*2.22766, center)])

#STO-3G basis for hydrogen
# def sto3g_hydrogen(center):
#     return sto3g(center, 1.24)

# def sto3g_helium(center):
#     return sto3g(center, 2.0925)



#The overlap integrals describe how the basis functions overlap
#as the atom centered gaussian basis functions are non-orthognal
#they have a non-zero overlap. The integral has the following form:
#S_{ij} = \int \phi_i(r-R_a) \phi_j(r-R_b) \mathrm{d}r
def overlap_integral_sto(b1, b2):
    return two_center_contraction(b1, b2, overlap_integral)

def overlap_integral(alpha, Ra, beta, Rb):
    """
    Integrals  with two GWPs   <g1|g2>

    INPUT:
        g1,g2 : GTO objects
    """

    # Rb = g2.center
    # Ra = g1.center
    # alpha = g1.alpha
    # beta = g2.alpha

    n = (2.*alpha/pi)**(3./4.) * (2.*beta/pi)**(3./4.)

    I  = n * (pi/(alpha+beta))**(3/2)
    I *= np.exp(-alpha*beta/(alpha+beta) * abs(Ra-Rb)**2)

    return I

def nuclear_attraction_gto(Rc, alpha, Ra, beta, Rb):
    """
    Zc - charge of the nuclei
    Rc - postion of the nuclei
    """
    # alpha = g1.alpha
    # beta  = g2.alpha
    Ra = jnp.array(Ra)
    # print('Ra', Ra)
    Rb = jnp.array(Rb)
    Rp = (alpha*Ra + beta*Rb)/(alpha + beta)
    Rc = jnp.array(Rc)

    n = (2*alpha/pi)**(3/4) * (2*beta/pi)**(3/4)
    matrix_element  = n*-2*pi/(alpha+beta)
    matrix_element *= jnp.exp(-alpha*beta/(alpha+beta)*norm(Ra-Rb)**2)

    t = (alpha+beta)*norm(Rp-Rc)**2
    if(abs(t) < 1e-8):
        return matrix_element

    matrix_element *= 0.5 * jnp.sqrt(pi/t) * erf(jnp.sqrt(t))

    return matrix_element

# def nuclear_attraction_gto_with_gradients(Rc, alpha, Ra, beta, Rb, gradient, argnums=0):

#     if not gradient or gradient == (0, ):
#         raise ValueError("NO auto differential")


#     Ra = jnp.array(Ra)
#     Rb = jnp.array(Rb)
#     Rc = jnp.array(Rc)


#     if 1 in gradient and 2 not in gradient:
#         e, f = value_and_grad(nuclear_attraction_gto, argnums)(Rc, alpha, Ra, beta, Rb)
#         return e, f

#     elif 1 in gradient and 2 in gradient:

#         e, f = value_and_grad(nuclear_attraction_gto, argnums)(Rc, alpha, Ra, beta, Rb)
#         h = hessian(nuclear_attraction_gto, argnums)(Rc, alpha, Ra, beta, Rb)

#         return e, f, h

#     elif any(y > 2 for y in gradient):

#         raise NotImplementedError('Gradients bigger than 2 has not been implemented.')

def point_charge_gto(coord, charge, alpha, Ra, beta, Rb):
    """
    Zc - charge of the nuclei
    Rc - postion of the nuclei
    """

    Rp = (alpha*Ra + beta*Rb)/(alpha + beta)

    n = (2*alpha/pi)**(3/4) * (2*beta/pi)**(3/4)
    matrix_element  = n*-2*pi/(alpha+beta)
    matrix_element *= jnp.exp(-alpha*beta/(alpha+beta)*norm(Ra-Rb)**2)

    ss = 0
    for n in range(len(charge)):

        Rc = coord[n, :]
        t = (alpha+beta)*norm(Rp-Rc)**2

        if(abs(t) < 1e-8):
            s = 1
        else:
            s = 0.5 * jnp.sqrt(pi/t) * erf(jnp.sqrt(t)) * charge[n]

        ss += s

    return matrix_element * ss

def nuclear_attraction_integral(Rc, b1, b2, gradient=None, argnums=0):
    """
    b1, b2 : STO orbitals
    """

    Rc = jnp.array(Rc)

    if gradient is None:
        total = 0.0

        for p  in range(b1.n):
            for q in range(b2.n):
                d1 = b1.d[p]
                d2 = b2.d[q]
                total += d1*d2 * nuclear_attraction_gto(Rc, b1.g[p].alpha, b1.g[p].center, b2.g[p].alpha, b2.g[p].center)


        return total

    else:

        assert(argnums == 0)

        total = 0.0
        first_order_diff = 0.
        second_order_diff = 0.

        argnums = 0

        for p  in range(b1.n):
            for q in range(b2.n):
                d1 = b1.d[p]
                d2 = b2.d[q]

                alpha, Ra, beta, Rb = b1.g[p].alpha, b1.g[p].center, b2.g[p].alpha, b2.g[p].center

                e, f = value_and_grad(nuclear_attraction_gto)(Rc, alpha, Ra, beta, Rb)

                # f = grad(nuclear_attraction_gto, argnums)(Rc, alpha, Ra, beta, Rb)

                h = hessian(nuclear_attraction_gto, argnums)(Rc, alpha, Ra, beta, Rb)

                total += e * d1 * d2

                first_order_diff += d1*d2 * jnp.array(f)

                second_order_diff += d1*d2 * jnp.array(h)

        # finite difference check
        # d = jnp.array([0, 0, 0.005])
        # Rc = Rc + d

        # print(Rc)

        # total1 = 0
        # for p  in range(b1.n):
        #     for q in range(b2.n):
        #         d1 = b1.d[p]
        #         d2 = b2.d[q]
        #         total1 += d1*d2 * nuclear_attraction_gto(Rc, b1.g[p].alpha, b1.g[p].center, b2.g[p].alpha, b2.g[p].center)

        # print((total1 - total)/0.005)


        return total, first_order_diff, second_order_diff

def kinetic_energy_gto(alpha, Ra, beta, Rb):

# def kinetic_energy_gto(g1, g2):
    # alpha = g1.alpha
    # beta = g2.alpha
    # Ra = g1.center
    # Rb = g2.center

    n = (2.*alpha/pi)**(3./4.) * (2*beta/pi)**(3./4.)

    gamma = alpha*beta/(alpha + beta)

    matrix_element  = n * gamma
    matrix_element *= (3. - 2. * gamma * abs(Ra-Rb)**2 )
    matrix_element *= (pi/(alpha+beta))**(3./2.)
    matrix_element *= jnp.exp(- gamma * abs(Ra-Rb)**2)

    return matrix_element

def kinetic_energy_integral(b1, b2):
    return two_center_contraction(b1, b2, kinetic_energy_gto)



def two_electron_integral_gto(g1, g2, g3, g4):

    alpha = g1.alpha
    beta  = g2.alpha
    gamma = g3.alpha
    delta = g4.alpha
    Ra = g1.center
    Rb = g2.center
    Rc = g3.center
    Rd = g4.center
    Rp = (alpha*Ra + beta*Rb)/(alpha + beta)
    Rq = (gamma*Rc + delta*Rd)/(gamma + delta)

    n  = (2.*alpha/pi)**(3/4) * (2*beta/pi)**(3/4)
    n *= (2.*gamma/pi)**(3/4) * (2*delta/pi)**(3/4)

    matrix_element  = n*2.*pi**(5./2.)
    matrix_element /= ((alpha+beta)*(gamma+delta) * sqrt(alpha+beta+gamma+delta))
    matrix_element *= exp(-alpha*beta/(alpha+beta)*abs(Ra-Rb)**2 - \
                        gamma*delta/(gamma+delta)*abs(Rc-Rd)**2)
    t = (alpha+beta)*(gamma+delta)/(alpha+beta+gamma+delta)*abs(Rp-Rq)**2

    if abs(t) < 1e-8:
        return matrix_element

    matrix_element *= 0.5 * sqrt(pi/t) * erf(sqrt(t))
    return matrix_element

def two_electron_integral(g1, g2, g3, g4):

    return four_center_contraction(g1, g2, g3, g4, two_electron_integral_gto)

def two_center_contraction(b1, b2, integral):
    """
    b1, b2 : STO orbitals
    """

    total = 0.0
    for p  in range(b1.n):

        d1 = b1.d[p]
        alpha = b1.g[p].alpha
        Ra = b1.g[p].center

        for q in range(b2.n):

            d2 = b2.d[q]
            beta = b2.g[q].alpha
            Rb = b2.g[q].center

            total += d1*d2*integral(alpha, Ra, beta, Rb)

    return total

def four_center_contraction(b1, b2, b3, b4, integral):
    """
    b1, b2, b3, b3 : STO_NG objects
    integral : name of integrals to perform computations
    """
    total = 0.0
    for p in range(b1.n):
        for q in range(b2.n):
            for r in range(b3.n):
                for s in range(b4.n):
                    dp = b1.d[p]
                    dq = b2.d[q]
                    dr = b3.d[r]
                    ds = b4.d[s]
                    total += dp*dq*dr*ds*integral(b1.g[p], b2.g[q], b3.g[r], b4.g[s])
    return total


def hartree_fock(R, Z, CI=False):

    #print("constructing basis set")

    phi = [0] * len(Z)

    for A in range(len(Z)):

        if Z[A] == 1:

            phi[A] = sto3g_hydrogen(R[A])

        elif Z[A] == 2:

            phi[A] = sto3g_helium(R[A])

    # total number of STOs
    K = len(phi)

    print('calculate the AO overlap matrix S')
    #the matrix should be symmetric with diagonal entries equal to one
    #print("building overlap matrix")

    S = jnp.eye(K)

    for i in range(len(phi)):
        for j in range( (i+1),len(phi)):
            s = overlap_integral_sto(phi[i], phi[j])
            S.at[i,j].set(s)
            S.at[j,i].set(s)

    print("S: ", S)


    #calculate the kinetic energy matrix T
    print("building kinetic energy matrix")
    T = jnp.zeros((K,K))

    #print('test', phi[0].g[0].center)
    #print('test', phi[1].g[1].center)

    for i in range(len(phi)):
        for j in range(i, len(phi)):
            T[i,j] = T[j,i] = kinetic_energy_integral(phi[i], phi[j])

    #print("T: ", T)

    #calculate nuclear attraction matrices V_i
    #print("building nuclear attraction matrices")

    V = jnp.zeros((K,K))

    for A in range(K):
        for i in range(K):
            for j in range(i,K):
                v = nuclear_attraction_integral(Z[A], R[A], phi[i], phi[j])
                V[i,j] += v
                if i != j:
                    V[j,i] += v
    #print("V: ", V)

    #build core-Hamiltonian matrix
    #print("building core-Hamiltonian matrix")
    Hcore = T + V

    print("Hcore: ", Hcore)

    #diagonalize overlap matrix to get transformation matrix X
    #print("diagonalizing overlap matrix")
    s, U = jnp.linalg.eigh(S)
    #print("building transformation matrix")
    X = U.dot(jnp.diagflat(s**(-0.5)).dot(dagger(U)))
    #print("X: ", X)


    #calculate all of the two-electron integrals
    #print("building two_electron Coulomb and exchange integrals")

    two_electron = jnp.zeros((K,K,K,K))

    for mu in range(K):
        for v in range(K):
            for lamb in range(K):
                for sigma in range(K):
                    two_electron[mu,v,sigma,lamb] = \
                        two_electron_integral(phi[mu], phi[v], phi[sigma], phi[lamb])

#                    coulomb  = two_electron_integral(phi[mu], phi[v], \
#                                                     phi[sigma], phi[lamb])
#                    two_electron[mu,v,sigma,lamb] = coulomb
                    #print("coulomb  ( ", mu, v, '|', sigma, lamb,"): ",coulomb)
#                    exchange = two_electron_integral(phi[mu], phi[lamb], \
#                                                     phi[sigma], phi[v])
#                    #print("exchange ( ", mu, lamb, '|', sigma, v, "): ",exchange)
#                    two_electron[mu,lamb,sigma,v] = exchange

    P = jnp.zeros((K,K))

    total_energy = 0.0
    old_energy = 0.0
    electronic_energy = 0.0


    # nuclear energy
    nuclear_energy = 0.0
    for A in range(len(Z)):
        for B in range(A+1,len(Z)):
            nuclear_energy += Z[A]*Z[B]/norm(R[A]-R[B])

    print("E_nclr = ", nuclear_energy)

    print("\n {:4s} {:13s} de\n".format("iter", "total energy"))
    for scf_iter in range(100):
        #calculate the two electron part of the Fock matrix
        G = jnp.zeros(Hcore.shape)

        K = len(phi)
        for mu in range(K):
            for v in range(K):
                for lamb in range(K):
                    for sigma in range(K):
                        coulomb  = two_electron[mu,v,sigma,lamb]

                        exchange = two_electron[mu,lamb,sigma,v]
                        #print("coulomb  [ ", mu, v, '|', sigma, lamb,"] : ",coulomb, exchange)

                        G[mu,v] += P[lamb,sigma] * (coulomb - 0.5*exchange)

        F = Hcore + G

        electronic_energy = 0.0

        electronic_energy = jnp.trace(P.dot( Hcore + F))

        electronic_energy *= 0.5

        #test
        #print('one electron energy = ', np.trace(P.dot(Hcore)))
        #print('two electron energy = ', 0.5*np.trace(P.dot(G)))


        #print("E_elec = ", electronic_energy)

        total_energy = electronic_energy + nuclear_energy
        print("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
               total_energy - old_energy))

        if scf_iter > 2 and abs(old_energy - total_energy) < 1e-6:
            break

        #println("F: ", F)
        #Fprime = X' * F * X
        Fprime = dagger(X).dot(F).dot(X)
        #println("F': $Fprime")

        epsilon, Cprime = jnp.linalg.eigh(Fprime)

        print("epsilon: ", epsilon)
        #print("C': ", Cprime)
        C = jnp.real(np.dot(X,Cprime))
        print("C: ", C)


        # new density matrix in original basis
        # P = jnp.zeros(Hcore.shape)
        # for mu in range(len(phi)):
        #     for v in range(len(phi)):
        P = 2. * jnp.outer(C[:,0], C[:,0])

        #print("New density matrix :  \n", P)


        old_energy = total_energy

    print('HF energy = ', total_energy)

    # check if this hartree-fock calculation is for configuration interaction
    # or not, if yes, output the essential information
    if CI == False:
        return total_energy
    else:
        return C, Hcore, nuclear_energy, two_electron

#def energy_functional(Hcore, two_electron, P):
#    """
#    density functional of the one-body density matrix P
#    """
#    G = np.zeros(Hcore.shape)
#    #K = len(phi)
#    K = Hcore.shape[0]
#
#    for mu in range(K):
#        for v in range(K):
#            for lamb in range(K):
#                for sigma in range(K):
#                    coulomb  = two_electron[mu,v,sigma,lamb]
#                    exchange = two_electron[mu,lamb,sigma,v]
#                    G[mu,v] += P[lamb,sigma] * (coulomb - 0.5*exchange)
#
#    F = Hcore + G
#
#    # compute electronic energy
#    electronic_energy = 0.0
#    for mu in range(K):
#        for v in range(K):
#            electronic_energy += P[v,mu]*(Hcore[mu,v]+F[mu,v])
#
#    electronic_energy *= 0.5
#
#    return electronic_energy

def configuration_interaction(R,Z):
    """
    configuration interaction for hydrogen molecule
    INPUT:
        H: Hamiltonian matrix constructed from determinants
        R: inter-nuclear distance
        Z: charge for nuclei
    OUTPUT:
        eigvals: eigenvalues
        eigvecs: eigenvectors (linear combination of determinants)

    """

    # Hartree Fock computations yield a set of MOs
    C, Hcore, nuclear_energy, two_electron = hartree_fock(R, Z, CI=True)

    # number of configurations considered in the calculation
    ND = 2

    P = np.zeros(Hcore.shape)

    K = Hcore.shape[0]
    print('number of MOs = ', K)

    # density matrix
    for mu in range(K):
        for v in range(K):
            P[mu,v] = 2*C[mu,1]*C[v,1]



    coulomb = np.zeros(Hcore.shape)
    exchange = np.zeros(Hcore.shape)

    for i in range(K):
        for j in range(K):

                    for mu in range(K):
                        for v in range(K):
                            for lamb in range(K):
                                for sigma in range(K):
                                    coulomb[i,j] += two_electron[mu, v, sigma, lamb]\
                                                    * C[mu,i] *\
                                            C[v,i] * C[sigma,j] * C[lamb,j]
                                    exchange[i,j] += two_electron[mu, v, sigma, lamb] \
                                                    * C[mu,i] *\
                                            C[v,j] * C[sigma,j] * C[lamb,i]

    F = np.matmul(C.T, np.matmul(Hcore, C))

    electronic_energy = F[0,0]*2 + coulomb[0,0]
    electronic_energy1 = F[1,1]*2 + coulomb[1,1]

    H = np.zeros((ND,ND))
    # construct the Hamiltonian
#    for i in range(1, ND):
#        for j in range(i,ND):
#             H[i,j] =

    H[0,0] = electronic_energy
    H[1,1] = electronic_energy1
    H[0,1] = H[1,0] = exchange[0,1]

    # diagonalizing the matrix
    eigvals, U = scipy.linalg.eigh(H)

    # density matrix represented in terms of Slater Determinants
    Temp = 50000. # K
    # transfer to Hartree
    Temp *= 3.1667909e-6
    print('Temperature  = {} au.'.format(Temp))

    energy_SD = np.array([electronic_energy, electronic_energy1])
    Z = sum(np.exp(-energy_SD/Temp))
    naive_rho = np.diagflat(np.exp(-energy_SD/Temp))
    print('naive density matrix = \n',naive_rho/Z)

    # density matrix represented in terms of Slater Determinants
    Z = sum(np.exp(- eigvals/Temp))
    D = np.diagflat(np.exp(- eigvals/Temp))/Z
    rho = np.matmul(U, np.matmul(D, U.T))

    print('full density matrix = \n', rho)

    total_energy = eigvals + nuclear_energy
    print('nuclear energy = {} \n'.format(nuclear_energy))
    print('total energy = ', total_energy)
    return total_energy





def test_h2(R, method = 'HF'):

    print("TESTING H2")

    if method == 'FCI':
        energy = configuration_interaction([-R/2., R/2.], [1, 1])
    elif method == 'HF':
        hartree_fock([-0.5*R, 0.5*R], [1, 1])

    print(energy[1]-energy[0])

    return

#    print('method = {}, electronic energy = {}'.format(method, \
#          electronic_energy))
    #szabo_energy = -1.8310
    #if abs(electronic_energy - szabo_energy) > 1e-6:
    #    print("TEST FAILED")
    #else:
    #    print("TEST PASSED")





def test_heh():
    print("TESTING HEH+")
    total_energy, electronic_energy = hartree_fock([0., 1.4632], [2, 1])
    szabo_energy = -4.227529
    if abs(electronic_energy - szabo_energy) > 1e-6:
        print("TEST FAILED")
    else:
        print("TEST PASSED")

def heh_pes():
    file = open("heh_pes.dat", "w")
    energy_he, energy_he = hartree_fock([0.0], [2])
    for r in np.linspace(0.7, 3.5, 25):
        total_energy, electronic_energy = hartree_fock([0., r], [2, 1])
        file.write( '{} {} \n'.format(r, (total_energy - energy_he)))

    file.close()

def build(basis, coords, Z, aosym=8):
    nao = len(basis)
    natom, _ = coords.shape

    s = jnp.zeros((nao, nao))
    for i in range(nao):
        for j in range(i):
            tmp = S(basis[i], basis[j])
            s = s.at[i,j].set(tmp[0])
    s = s + s.T
    s = jnp.fill_diagonal(s, 1, inplace=False)


    t = jnp.zeros((nao, nao))
    for i in range(nao):
        for j in range(i+1):
            tmp = T(basis[i], basis[j])[0]
            t = t.at[i,j].set(tmp)
    t = t + t.T
    for i in range(nao):
        t = t.at[i,i].set(T(basis[i], basis[i])[0])


    v = jnp.zeros((nao,nao))
    for i in range(nao):
        for j in range(i+1):
            tmp = 0
            for C in range(natom):
                tmp -= Z[C] * point_charge(basis[i], basis[j], coords[C])[0]

            v = v.at[i,j].set(tmp)
            if i != j:
                v = v.at[j,i].set(tmp)



    eri = jnp.zeros((nao, nao, nao, nao))
    for p in range(nao):
        for q in range(nao):
            for r in range(nao):
                for _s in range(nao):
                    eri = eri.at[p,q,r,_s].set(ERI(basis[p], basis[q], basis[r], basis[_s])[0])

    return s, t, v, eri

if __name__=="__main__":

    # from pyqed.qchem.mol import atomic_chain
    from pyqed.qchem import RHF

    mol = Molecule(atom='H 0 0 0.; H 0. 0 1.4', unit='bohr', basis='631g')
    mol.build()

    dv, ddv = mol.nuc_grad().nuclear_repulsion()

    print(dv.shape, ddv.shape)
    # print(mol.hcore)

    # mol.RHF().run()


    # for i in range(mol.natom):
    #     v, f, g = mol.nuclear_attraction(atm_id=i, gradient=(1,2), argnums=0)

    #     print(f[:,:,2], g[:,:,2,2])



    # kin_e = np.trace(dm.dot(k_int1e))
    # print("Kinetic energy (Hartree):", kin_e)

    # Define atomic symbols and coordinates (i.e., basis function centers)
    atoms = ["H", "H"]
    atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # basis_dict = parse_gbs('../basis_set/sto-3g.1.gbs')

    basis_dict = parse_gbs('../basis_set/6-31g.0.gbs')

    basis = make_contractions(basis_dict, atoms, atcoords, 'c')

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



    # print(len(basis))
    # build(basis, atcoords)
    # nao = len(basis)
    # natom, _ = atcoords.shape

    # print(nao)

    # s = jnp.zeros((nao, nao))

    # for i in range(nao):
    #     for j in range(i):
    #         tmp = S(basis[i], basis[j])
    #         s = s.at[i,j].set(tmp[0])
    # s = s + s.T
    # s = jnp.fill_diagonal(s, 1, inplace=False)
    # print(s)

    # print(S(basis[2], basis[1]))
    from timeit import default_timer as timer
    start = timer()
    Z = [1,1]
    s,t, v, eri = build(basis, atcoords, Z)
    end = timer()
    print(end - start)
    print(t+v)
    # point_charge(a, a, myOrigin))
    # print(eri)



    # mol = qchem.Molecule(atom = [
    # ['H' , (0. , 0. , 0)],
    # ['H' , (0. , 0. , 1.)], ], basis='631g')

    # mol.build()
    # print(mol.hcore)
    # natom = 8
    # z = np.linspace(-10, 10, natom)
    # print('interatomic distance = ', interval(z))
    # mol = atomic_chain(natom, z)

    # mol = mol.topyscf()
    # mol.RHF().run()

    ### finite difference
     # = mol.nuclear_attraction(atm_id=0)

    # from pyqed.qchem.gtos import auto_diff_nuclear_attraction_gto


    # Ra = jnp.array((0, 0, 0.2))
    # alpha = 1.
    # beta = 2.
    # Rb = jnp.array((0.3,0,1))
    # Rc = jnp.array((0.3,0,1))
    # dif1, dif2 = auto_diff_nuclear_attraction_gto(Rc, alpha, Ra, beta, Rb, gradient=(1,2), argnums=0)
    # print('dif', dif1, dif2)

    # from jax import grad, hessian
    # f = grad(nuclear_attraction_gto, argnums=0)(Rc, alpha, Ra, beta, Rb)
    # print('f',f)
    # h = hessian(nuclear_attraction_gto)(Rc, alpha, Ra, beta, Rb)

    # print('h', h)


# if __name__=="__main__":

#     # energy = test_h2(1.6/0.529,'HF')

#     # print(energy)


#     # mol = 'H 0 0 0; H 0 0 0.74'
#     Ra = jnp.array((1., 0, 0))
#     alpha = 1.
#     beta = 2.
#     Rb = jnp.array((0,0,0.3))
#     Rc = jnp.array((2.,0,0))

#     from jax import grad, hessian
#     f = grad(nuclear_attraction_gto, argnums=0)(Rc, alpha, Ra, beta, Rb)
#     print(f)
#     h = hessian(nuclear_attraction_gto)(Rc, alpha, Ra, beta, Rb)

#     print('h', h)
#     # charge = [2]
#     # coord = jnp.array([Rc])
#     # f = grad(point_charge_gto)(coord, charge, alpha, Ra, beta, Rb)

#     # print(f)


#     #h2_pes('HF')
#     #h2_pes('FCI')
#         #h2_pes()
#         #test_heh()

#     #heh_pes()

#     #g1 = GTO_1s(1.0,2.0)
#     #g2 = GTO_1s(1.4,3.2)