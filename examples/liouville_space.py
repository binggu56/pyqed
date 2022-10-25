#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 22:54:21 2021

@author: binggu
"""

from lime.superoperator import *

from lime.phys import pauli, ket2dm

s0, sx, sy, sz = pauli()

psi0 = np.array([0.3, 0.5])
rho0 = ket2dm(psi0)

print(np.trace(sz.dot(rho0)))

expect(rho0.flatten(),  sz)