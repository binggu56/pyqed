#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:55:10 2018

@author: binggu

Grid representation to solve DFT for Hydrogen atom
"""
import numpy as np
from scipy.sparse import spdiags, eye, linalg, kron
 
g = 30
g3 = g**3
p = np.linspace(-5, 5, g)
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
L3 = kron(kron(L,I), I) + kron(kron(I,L), I) + kron(kron(I,I), L)

#print(spdiags(Vext, 0, g3, g3).toarray())

# compute eigvalues
E, PSI = linalg.eigs(-0.5*L3 + spdiags(Vext, 0, g3, g3).toarray(), k=1, which='SR')
