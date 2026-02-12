#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:47:49 2023

@author: bing
"""

#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run TDDFT calculation.
'''

from pyscf import gto, scf, dft, tddft

# mol = gto.Mole()
# mol.build(
    
#     atom = 'H 0 0 0; F 0 0 1.1',  
#     # in Angstrom
#     basis = '631g',
#     symmetry = True,
# )

# mf = dft.RKS(mol)
# mf.xc = 'b3lyp'
# mf.kernel()

# mytd = tddft.TDDFT(mf)
# #mytd.nstates = 10
# mytd.kernel()
# mytd.analyze()
# dip = mytd.transition_dipole()
# mdip = mytd.transition_magnetic_dipole()


mol = gto.M(atom="N 0 0 0; N 0 0 1.2", basis="ccpvdz")
ci = mol.apply("CISD").run()
ci.nstates = 5
ci.run()

ci.nstates = 5
force = ci.nuc_grad_method().kernel(state=1)

print(force.shape)

def force(method='tddft'):
    ks = mol.apply("RKS").run()
    td = ks.apply("TDRKS")
    td.nstates = 5
    force = td.nuc_grad_method().kernel(state=1)
    return force
