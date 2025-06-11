#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:55:10 2018

@author: binggu

Grid representation Restricted DFT for Hydrogen atom

Refs 
    https://pubs.acs.org/doi/10.1021/ed5004788
"""
import numpy as np
from scipy.sparse import spdiags, eye, linalg, kron
from numpy import pi

from pyqed import dag, isherm, SineDVR
import logging


g = 10
g3 = g**3
p = np.linspace(-2, 2 ,g)
domain = [-2, 2]
level = 4
[X, Y, Z] = np.meshgrid(p, p, p)

h = p[1] - p[0]
X = X[:]; Y = Y[:]; Z = Z[:]
R = np.sqrt(X**2 + Y**2 + Z**2)

Vext = -2./R.flatten()
#Vext = Vext.flatten('C')

e = np.ones(g)
diags = np.array([-1,0,1])

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

L3 = get_kinetic()

K = -0.5 * L3

nelec = 2
nocc = nelec//2
print('occ orbs', nocc)

def energy_elec(n, mo_coeff, Vh):
    
    T =  (dag(mo_coeff) @ K @ mo_coeff) * h**3

    
    Eext = np.dot(n, Vext) * h**3 # electron-nuclear Columb energy
    Eh = 0.5 * np.dot(n, Vh) * h**2 # Hartree energy
    Ex = -(3./4.) * (3./pi)**(1./3.) * sum(n**(4/3)) * h**3


    Etot = np.trace(T) + Eext + Ex + Eh
    return Etot 
    
def get_density(mo_coeff):
    
    PSI = mo_coeff/h**(3/2)

    # construct electron density
    n = 0
    for p in range(nocc):
        n += 2. * np.abs(PSI[:, p])**2
    return n

def get_exchange(n):
    Vx = -(3./pi)**(1/3)*n**(1/3) # LDA exchange potential
    return Vx
            
def rks_solver(max_cycle=100, init_guess='h1e', e_conv=1e-6):
    
    h = -0.5*L3 + spdiags(Vext, 0, g3, g3).toarray()

    E, mo_coeff = linalg.eigs(h, k=1, which='SR')
    n = get_density(mo_coeff)

    Vh, info = linalg.cgs(L3, -4. * pi * n, maxiter = 400)

    old_energy = energy_elec(n, mo_coeff, Vh)
    
    # construct Hartree potential via Possion eq. L3 Vh = -4*pi*n

    Vx = get_exchange(n)

    # # update total potential
    # Vtot = Vx + Vh + Vext
        
    for scf_iter in range(max_cycle):
        
        # update total potential
        Vtot = Vx + Vh + Vext
        h = -0.5 * L3 + Vtot
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
        
rks_solver()
