#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:55:10 2018

@author: binggu

Grid representation to solve DFT for Hydrogen atom
"""
import numpy as np
from scipy.sparse import spdiags, eye, linalg, kron
from numpy import pi


g = 10
g3 = g**3
p = np.linspace(-3, 3 ,g)
[X, Y, Z] = np.meshgrid(p, p, p)

h = p[1] - p[0]
X = X[:]; Y = Y[:]; Z = Z[:]
R = np.sqrt(X**2 + Y**2 + Z**2)

Vext = -1./R.flatten()
#Vext = Vext.flatten('C')

e = np.ones(g)
diags = np.array([-1,0,1])

# kinetic energy operator for 1D, with periodic finite difference
L = spdiags([e, -2*e, e], diags, g, g).toarray()/h**2

# outer product to construct matrix representation of the operator
I = eye(g)
print(I.size)
print(L.size)
L3 = kron(kron(L,I), I) + kron(kron(I,L), I) + kron(kron(I,I), L)
print(kron(I,L).size)
#print(spdiags(Vext, 0, g3, g3).toarray())

Vtot = Vext

while True:
    # compute eigvalues
    E, PSI = linalg.eigs(-0.5*L3 + spdiags(Vext, 0, g3, g3).toarray(), k=1, which='SR')

    print(np.size(PSI))
    PSI = PSI/h**(3/2)

    # construct electron density
    n = 2. * PSI**2
    print(n.size)
    print(L3.size)

    Vx = -(3./pi)**(1/3)*n**(1/3) # LDA exchange potential
    # construct Hartree potential via Possion eq. L3 Vh = -4*pi*n
    Vh = linalg.cgs(L3, -4. * pi * n, 1e-7, maxiter = 400)

    # update total potential
    Vtot = Vx + Vh + Vext

    T = 2. * np.dot(PSI, np.dot(L3, PSI)) * h**3
    Eext = np.dot(n, Vext) * h**3 # electron-nuclear Columb energy
    Eh = 0.5 * np.dot(n, Vh) * h**2 # Hartree energy
    Ex = -(3./4.) * (3./pi)**(1./3.) * sum(n**(4/3)) * h**3

    Etot = T + Eext + Ex + Eh
    print('Eigenvalue', E)
    print('Kinetic energy', T)
    print('Exchange energy', Ex)
    print('External energy',Eext)
    #print('Potential energy', Es)
    print('Total energy for Hydrogen atom ', Etot)