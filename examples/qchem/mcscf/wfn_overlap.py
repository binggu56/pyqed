#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:26:23 2026

@author: gugroup
"""

if __name__ == "__main__":
    from pyqed import Molecule
    from pyqed.qchem.mcscf.casci import overlap, CASCI
    from pyqed.qchem.mcscf.casscf import CASSCF

    mol = Molecule(atom = [
    ['Li' , (0. , 0. , 0)],
    ['F' , (0. , 0. , 1)], ], basis = 'sto6g')
    mol.build()
    
    # mol.molecular_frame()
    # print(mol.atom_coords())

    nstates = 3
    weights = [1/nstates, ] * nstates 
    
    # Rs = np.linspace(1,4,4)
    # E = np.zeros((nstates, len(Rs)))

    mf = mol.RHF()
    mf.run()

    ncas, nelecas = (2,2)
    mc = CASSCF(mf, ncas, nelecas)
    mc.fix_spin(ss=0, shift=0.2)
    mc.state_average(weights).run(nstates)




    mol2 = Molecule(atom = [
    ['Li' , (0. , 0. , 0)],
    ['F' , (0. , 0. , 1.2)], ], basis = 'sto6g')

    # mol.unit = 'b'
    mol2.build()

    mf2 = mol2.RHF().run()
    mc2 = CASSCF(mf2, ncas, nelecas)
    mc2.fix_spin(ss=0, shift=0.2)

    mc2.state_average(weights).run(nstates)

    # print('Fix spin by penalty')

    # # mc = CASCI(mf2, ncas, nelecas)
    # mc.run(5)

    # casci.run()
    S = overlap(mc, mc2)
    print(S)