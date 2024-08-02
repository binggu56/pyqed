#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:55:10 2018

@author: binggu

Grid representation Restricted DFT for Hydrogen atom

Refs 
    https://pubs.acs.org/doi/10.1021/ed5004788
    
    
NOT FINISHED.
"""
import numpy as np
from scipy.sparse import spdiags, eye, linalg, kron
from numpy import pi

from pyqed import dag, isherm, SineDVR
import logging

from opt_einsum import contract
from numba import vectorize

@vectorize
def regularized_coulomb(r, h):
    """
    
    Refs    
        TASK QUARTERLY vol. 21, No 2, 2017, pp. 177–184

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    epsilon = (3/4*np.pi)**(1/3) * h
    if r <= h:
        v = 3/(2*epsilon) - r/h * (3/(2*epsilon) - h/(h**2 + epsilon**2/3))
    else:
        v = r/(r**2 + epsilon**2/3)
    return v



def get_kinetic(method='dvr'):
    
    if method == 'dvr':
        dvr = SineDVR(-2, 2, npts=10)
        t = dvr.t()
        
    elif method == 'fd':        
        # kinetic energy operator for 1D, with periodic finite difference
        t = -0.5 * spdiags([e, -2*e, e], diags, g, g).toarray()/h**2
    
    
    # outer product to construct matrix representation of the operator
    I = eye(g)

    K = kron(kron(t,I), I) + kron(kron(I,t), I) + kron(kron(I,I), t)

    
    #print(spdiags(Vext, 0, g3, g3).toarray())
    return K


def energy_elec(n, mo_coeff, Vh):
    

    e_kin =  np.trace(dag(mo_coeff) @ K @ mo_coeff) * h**3
    print('e_kin', e_kin)
    
    Eext = np.dot(n, Vext) * h**3 # electron-nuclear Columb energy
    print(Eext)
    Eh = 0.5 * np.dot(n, Vh) * h**3 # Hartree energy
    
    Ex = -(3./4.) * (3./pi)**(1./3.) * sum(n**(4/3)) * h**3


    Etot = e_kin + Eext + Ex + Eh
    
    return Etot 
    
def get_density(mo_coeff):
    
    # mo_coeff = mo_coeff/h**(3/2)

    # construct electron density
    n = 0
    for p in range(nocc):
        n += 2. * np.abs(mo_coeff[:, p])**2/h**(3)
    return n

def get_exchange(n):
    Vx = -(3./pi)**(1/3)*n**(1/3) # LDA exchange potential
    return Vx
            
def rks_solver(max_cycle=30, init_guess='h1e', e_conv=1e-6):
    
    h = K + spdiags(Vext, 0, g3, g3)

    E, mo_coeff = linalg.eigs(h, k=6, which='SR')
    n = get_density(mo_coeff)
    
    print('charge', np.sum(n) * h**3)

    Vh, info = linalg.cgs(L3, -4. * pi * n, maxiter = 400)

    old_energy = energy_elec(n, mo_coeff, Vh)
    
    # construct Hartree potential via Possion eq. L3 Vh = -4*pi*n

    Vx = get_exchange(n)

    # # update total potential
    # Vtot = Vx + Vh + Vext
        
    for scf_iter in range(max_cycle):
        
        # update total potential
        Vtot = Vx + Vh + Vext
        h = K + Vtot
        print(isherm(h))
    
        # compute eigvalues
        E, mo_coeff = linalg.eigsh(h, k=2, which='SA')
        
        print(mo_coeff.shape)
        
        n = get_density(mo_coeff)
        
        Vx = -(3./pi)**(1/3)*n**(1/3) # LDA exchange potential
        # construct Hartree potential via Possion eq. L3 Vh = -4*pi*n
        
        Vh, info = linalg.cgs(L3, -4. * np.pi * n, maxiter = 400)
        
        total_energy = energy_elec(n, mo_coeff[:, :nocc], Vh)
        
        print("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
                   total_energy - old_energy))
        
        if abs(old_energy - total_energy) < e_conv:
            print('SCF Converged.')
            print('Total energy = ', total_energy)
            break
        
        old_energy = total_energy

        # print('Eigenvalue', E)
        # # print('Kinetic energy', T)
        # # print('Exchange energy', Ex)
        # # print('External energy',Eext)
        # #print('Potential energy', Es)
        # print('Total energy for He ', total_energy)


g = 15
g3 = g**3
# p = np.linspace(-2, 2 ,g)

dvr = SineDVR(-2, 2, g)
t = dvr.t()
p = dvr.x



[X, Y, Z] = np.meshgrid(p, p, p)

h = p[1] - p[0]
# X = X[:]; Y = Y[:]; Z = Z[:]

R = np.sqrt(X**2 + Y**2 + Z**2)

Vext = - regularized_coulomb(R, h).flatten()
#Vext = Vext.flatten('C')

e = np.ones(g)
diags = np.array([-1,0,1])

I = eye(g)
K = kron(kron(t,I), I) + kron(kron(I,t), I) + kron(kron(I,I), t)


# Laplacian
L3 = -2 * K 

nelec = 1
nocc = nelec//2
print('occ orbs', nocc)
        
rks_solver()
