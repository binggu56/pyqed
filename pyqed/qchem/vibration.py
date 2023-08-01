#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:49:13 2023

@author: bing
"""

import numpy as np
from numpy import einsum

from pyqed import norm

from pyscf import gto, scf, dft, tddft
from pyqed.coordinates import Molecule

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)




class Vibration(Molecule):
    def __init__(self, mf, optimized=True):
        
        self.natom = mf.mol.natm

        if optimized:

            self.mol = mf.mol
            self.mf = mf

        else:
            from pyscf.geomopt.berny_solver import optimize

            mol = optimize(mf)

            self.mol = mol # optimized geometry

            
            print(mol.atom_coords())
            
            mf_opt = dft.RKS(mol)
            mf_opt.xc = mf.xc

            mf_opt.kernel()

            self.mf = mf_opt

        self.modes = None
        self.freq = None
        
        return
    
    def to_molecular_frame(self):
        pass
    

    def atom_coords(self):
        return self.mol.atom_coords()

    def run(self):
        """
        Compute the normal modes at equilibrium geometry

        Returns
        -------
        w : TYPE
            DESCRIPTION.
        modes : array [3N, N, 3], N is the number of atoms
            The first six modes are translation and rotation.

        """

        from pyscf.prop.freq import rks

        w, modes = rks.Freq(self.mf).kernel()

        self.modes = modes
        self.freq = w
        print(w, modes)
        return w, modes

    def atomic_force(self):
        
        # atomic gradients [natom, 3]
        grad = self.mf.nuc_grad_method().kernel()
        
        de = einsum('mai, ai -> m', self.modes, grad)
        
        return de
    
    # def force(self):
        
        
    
    def dump_xyz(self, fname=None):
        f = open(fname, 'w')
        f.write('{}\n\n'.format(self.natom))
        for i in range(self.natom):
            f.write('{} {} {} {}\n'.format(self.mol.atom_symbol(i), *self.mol.atom_coord(i)))
        return 
    
    def vibronic_coupling(self):
        pass
    
    def scan(mode_id, npts=16, subfolder=False):
        # scan the APES along a normal mode or a few normal modes
        pass
    
    def dip_deriv(self):
        pass
    
    def infrared(self, lw=0.005):
        pass
    
if __name__ == '__main__':
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
    
    vib = Vibration(mf, optimized=True)
    w, modes = vib.run()
    print(modes.shape)
    print(vib.atomic_force())

#print(vib.atom_coords())

# mytd = tddft.TDDFT(mf)


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