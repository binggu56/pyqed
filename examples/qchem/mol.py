#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 09:49:43 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed import Molecule, build_atom_from_coords


atom = [['H' , (0,      0., 0.)],
        ['H', (1.1, 0., 0.)],
        ['H', (1.5, 0, 0)]]

mol = Molecule(atom)

d = mol._build_distance_matrix()

print(d)

# mol.basis = 'sto3G'
# mol.build()
# mol.RHF().run()