#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:21:07 2023

@author: bing
"""

import numpy as np

import time

from pyqed import basis, ket2dm, mol, pauli
# from lime.superoperator import left
from pyqed.oqs import HEOMSolver, dm2vec
# import proplot as plt
import matplotlib.pyplot as plt




from qutip import basis, sigmax, sigmaz

# The system Hamiltonian:
eps = 0.0  # energy of the 2-level system
Del = 1.0  # tunnelling term
H_sys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()

# Initial state of the system:
rho0 = basis(2,1) * basis(2,1).dag()

# Bath properties:
gamma = 1 # cut off frequency
lam = 0.002  # coupling strength
T = 0.5  # temperature

# System-bath coupling operator:
Q = sigmaz()

from qutip.nonmarkov.heom import DrudeLorentzBath
from qutip.nonmarkov.heom import DrudeLorentzPadeBath

# Number of expansion terms to retain:
Nk = 1

# Matsubara expansion:
bath = DrudeLorentzBath(Q, lam, gamma, T, Nk)

# Padé expansion:
bath = DrudeLorentzPadeBath(Q, lam, gamma, T, Nk)

from qutip.nonmarkov.heom import HEOMSolver
from qutip import Options

max_depth = 5  # maximum hierarchy depth to retain
options = Options(nsteps=15_000)

solver = HEOMSolver(H_sys, bath, max_depth=max_depth, options=options)

# tlist = [0, 10, 20]  # times to evaluate the system state at
# result = solver.run(rho0, tlist)

# Run the solver:
tlist = np.linspace(0, 100, 101)
result = solver.run(rho0, tlist, e_ops={"12": sigmax()})

# Plot the results:
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
# axes.plot(result.times, result.expect["11"], 'b', linewidth=2, label="P11")
axes.plot(result.times, result.expect["12"], 'r', linewidth=2, label="P12")
axes.set_xlabel(r't', fontsize=28)
axes.legend(loc=0, fontsize=12)




# delta = 0.2 
# eps0 = 1.0
# gamma1 = 0.5

# s0, sx, sy, sz = pauli()
# H = - delta/2.0 * sx - eps0/2.0 * sz

# def ohmic_spectrum(w):
#   # if w == 0.0: # dephasing inducing noise
#   #   return gamma1
#   # else: # relaxation inducing noise
#    return gamma1 / 2 * (w / (2 * np.pi))


# # redfield = Redfield_solver(H, c_ops=[sx], spectra=[ohmic_spectrum])
# # R, evecs = redfield.redfield_tensor()
# rho0 = np.zeros((2,2))
# rho0[0, 0] = 1
# sol = HEOMSolver(H, c_ops=[sz])
# nt = 100 
# rho = sol.run(rho0=rho0, dt=0.001, nt=nt, temperature=300, cutoff=5, reorganization=0.2, nado=5)
# print(rho)

# # propagator approach
# rho0 = dm2vec(rho0)

# u = sol.propagator(dt=0.001, nt=nt, temperature=300, cutoff=5, reorganization=0.2, nado=5)

# print(u[0] @ rho0)
