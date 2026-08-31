#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:32:14 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

# if __name__ == "__main__":
import numpy as np
from pyqed import Molecule
from pyqed.dvr import DVR
from pyqed.qchem.ci import CASCI
from pyqed.qchem.ci.casci import overlap

import copy

from pyqed.units import amu2au



# mol = Molecule(atom = [
# ['Li' , (0. , 0. , 0)],
# ['F' , (0. , 0. , 1)], ])

# mol.basis = '631g'
# mol.charge = 0

# mol.molecular_frame()


# print(mol.atom_coords())

nstates = 3
grid = DVR([(2, 10)], npts=[3])

x = grid.x[0] # bond length
nx = grid.shape[0]

print(x)


atom = [
    ['Li' , (0. , 0. , 0)],
    ['F' , (0. , 0. , x[0])]]

mol = Molecule(atom, basis='631g')
mol.molecular_frame()
mol.build()


m1, m2 = mol.atom_mass_list() * amu2au
mass = m1 * m2/(m1+m2)
print(mass)

mf = mol.RHF()
mf.run()

dm = mf.make_rdm1()

ncas, nelecas = (2,2)
casci = CASCI(mf, ncas, nelecas)

casci.run(nstates)


E = np.zeros((nstates, nx))
L = np.zeros((nx-1, nstates, nstates))

E[:,0] = casci.e_tot
casci_old = copy.copy(casci)


for i in range(1, nx):

    R = x[i]
    print('bond length = ', R)

    coord = [[0. , 0. , 0], [0. , 0. , R]]

    mol.set_geom(coord)
    mol.molecular_frame()
    mol.build()

    print(mol.atom_coords())

    mf = mol.RHF()
    mf.run(dm0=dm)

    dm = mf.make_rdm1()

    # ncas, nelecas = (6,2)
    casci = CASCI(mf, ncas, nelecas)

    casci.run(nstates)

    E[:,i] = casci.e_tot

    L[i-1] = overlap(casci_old, casci)

    print(L[i-1])

    casci_old = copy.copy(casci)



np.savez('energy', E)

import ultraplot as plt

fig, ax = plt.subplots()
for n in range(nstates):
    ax.plot(x, E[n])

#### test overlap
# mol2 = Molecule(atom = [
# ['H' , (0. , 0. , 0)],
# ['H' , (0. , 0. , 1.4)], ])
# mol2.basis = '631g'

# # mol.unit = 'b'
# mol2.build()

# mf2 = mol2.RHF().run()


# ncas, nelecas = (4,2)
# casci2 = CASCI(mf2, ncas, nelecas).run(2)

# casci.run()
# S = overlap(casci, casci2)
# print(S)
