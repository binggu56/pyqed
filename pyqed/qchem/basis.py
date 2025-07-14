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

class BasisFunction(object):
    ''' A class that contains all our basis function data
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

        print(prefactor, N)

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




def V(a,b,C):
    '''Evaluates overlap between two contracted Gaussians
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
    atoms = mol.atom_symbols()
    atcoords = mol.atom_coords()
    atnums = mol.atom_charges()

    basis_dir = os.path.abspath(f'{pyqed.__file__}/../qchem/')

    if isinstance(mol.basis, str):

        if mol.basis.replace('-','').lower() == '631g':

            # Obtain basis functions from the basis set files
            basis_dict = parse_gbs(basis_dir + "/6-31g.1.gbs")

        elif mol.basis.replace('-','').lower() == 'sto3g':

            basis_dict = parse_gbs(basis_dir + "/sto-3g.1.gbs")

        elif mol.basis.replace('-','').lower() == 'sto6g':

            basis_dict = parse_gbs(basis_dir + "/sto-6g.1.gbs")

        elif mol.basis.replace('-','').lower() == '6311g':

            basis_dict = parse_gbs(basis_dir + "/6-311g.0.gbs")



        basis = make_contractions(basis_dict, atoms, atcoords, coord_types="p")
    else:

        raise NotImplementedError('Customized basis not supported yet.')

    total_ao = 0
    for shell in basis:
        total_ao += shell.angmom_components_cart.shape[0]

    mol.nao = total_ao

    print("Number of generalized shells:", len(basis))

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




if __name__=='__main__':

    # kin_e = np.trace(dm.dot(k_int1e))
    # print("Kinetic energy (Hartree):", kin_e)

    # Define atomic symbols and coordinates (i.e., basis function centers)
    # atoms = ["H", "H"]
    # atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # # To obtain the total number of AOs we compute the cartesian components for each angular momentum
    # total_ao = 0
    # print(f"Number of generalized shells: {len(basis)}") # Output 6
    # for shell in basis:
    #     total_ao += shell.angmom_components_cart.shape[0]

    # print("Total number of AOs: ", total_ao) # output 10


    myOrigin = [1.0, 2.0, 3.0]
    myShell = (0,0,0) # p‐orbitals would be (1,0,0) or (0,1,0) or (0,0,1), etc.
    myExps = [3.42525091, 0.62391373, 0.16885540]
    myCoefs = [0.15432897, 0.53532814, 0.44463454]
    a = BasisFunction(origin=myOrigin,shell=myShell,exps=myExps,coefs=myCoefs)