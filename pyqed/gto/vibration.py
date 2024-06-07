#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:49:13 2023

@author: bing
"""

import numpy as np
import numpy
from numpy import einsum

from pyqed import norm

from pyscf import gto, scf, dft, tddft


# import pyscf
from pyscf.data import nist 

from pyqed import au2debye, au2amu 
from pyqed import Molecule, build_atom_from_coords

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)








if __name__ == '__main__':

    from pyqed.units import au2amu, au2wavenumber

    mol = gto.Mole()
    # mol.build(
    #     atom = '''
    #  C                  0.000000    1.294496    0.000000
    #  N                  1.191532    0.687931   -0.000000
    #  C                  1.121066   -0.647248   -0.000000
    #  C                 -1.121066   -0.647248   -0.000000
    #  N                 -1.191532    0.687931   -0.000000
    #  H                  2.063953   -1.191624    0.000000
    #  H                  0.000000    2.383248    0.000000
    #  H                 -2.063953   -1.191624   -0.000000
    #  N                 -0.000000   -1.375863   -0.000000''',  # in Angstrom
    #     basis = '631g',
    #     symmetry = True,
    #     verbose = 4,
    # )
    # mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
    #             ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
    #             ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]
    mol.atom = [['O',   [0.000000, -0.000000, 1.271610]],
      ['H',   [0.000000, -0.770445, 1.951195]],
      ['H',   [0.000000, 0.770446, 1.951195]]]

    # mol = gto.M(atom='N 0 0 0; N 0 0 2.100825', basis='def2-svp', verbose=4, unit="bohr")

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    ground_state_energy = mf.e_tot
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    # mass_nuc = mol.atom_mass_list() / au2amu
    # print(mass_nuc)
    # normal modes

    vib = Vibration(mf, optimized=True)
    w, modes, _ = vib.run()

    print(w*au2wavenumber)
    print(modes.shape)
    print(vib.atomic_force())


# #print(vib.atom_coords())

#     td = tddft.TDDFT(mf)
#     td.kernel(nstates = 2)
#     excitation_energy = td.e

#     # # dipole moments
#     # dip = td.transition_dipole() # nstates x 3
#     # medip = td.transition_magnetic_dipole()

#     # print(td.e_tot)
#     grad = td.nuc_grad_method()
#     grad.kernel(state=1) # state = 1 means the first excited state.
#     print(dir(grad))

    # transform to nuc derivative wrt normal modes


    # interstate vibronic coupling: lambda





    # mytd.analyze()

    #

#     from pyscf import gto, scf, eph
#     mol = gto.M()
#     mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
#                 ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
#                 ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]
#     mol.unit = 'Bohr'
#     mol.basis = 'sto3g'
#     mol.verbose=4
#     mol.build() # this is a pre-computed relaxed geometry

#     mf = scf.RHF(mol)
#     mf.conv_tol = 1e-16
#     mf.conv_tol_grad = 1e-10
#     mf.kernel()

#     myeph = eph.EPH(mf)

#     print("Force on the atoms/au:")

#     grad = mf.nuc_grad_method().kernel()
#     print(grad)

#     # the shape of eph is [3N-6, nao, nao]
#     eph, omega = myeph.kernel(mo_rep=True)

#     print(eph, omega)

#!/usr/bin/env python

# '''
# A simple example to run EPH calculation.
# '''
# from pyscf import gto, dft, eph
# mol = gto.M(atom='N 0 0 0; N 0 0 2.100825', basis='def2-svp', verbose=4, unit="bohr")
# # this is a pre-computed relaxed molecule
# # for geometry relaxation, refer to pyscf/example/geomopt
# mf = dft.RKS(mol, xc='pbe,pbe')
# mf.run()

# grad = mf.nuc_grad_method().kernel()
# assert (abs(grad).sum()<1e-5) # making sure the geometry is relaxed

# myeph = eph.EPH(mf)
# mat, omega = myeph.kernel()
# print(mat.shape, omega)
# print(myeph.vec)