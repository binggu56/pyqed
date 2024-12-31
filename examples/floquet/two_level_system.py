#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:29:08 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed import Mol, pauli 

I, X, Y, Z = pauli()
H = 0.5 * Z 
mol = Mol(H, X)

floquet = mol.Floquet(omegad=0.1, E0=0.3, nt=61)

floquet.run()