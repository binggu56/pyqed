#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 22:49:52 2023

@author: bing
"""

from pyqed import Mol, dag, multispin

class Frenkel(Mol):
    def __init__(self, onsite, hopping, nsites):

        H, lowering = multispin(onsite, hopping, nsites)

        if isinstance(lowering, list):
            edip = 0
            for l in lowering:
                edip += l + dag(l)
        else:
            edip = lowering + dag(lowering)

        self.H = H
        self.edip = edip
        self.dim = 2**nsites
        self.lowering = lowering





if __name__ == '__main__':

    from pyqed import au2wavenumber, level_scheme
    from pyqed.models.exciton import Frenkel

    # parameters taken from JCP xxx
    onsite = 26000/au2wavenumber
    J = -260/au2wavenumber


    model = Frenkel(onsite, hopping=J, nsites=2)
    B = model.lowering
    B0 = B[0]
    print(dag(B0) @ B0)
    # print(model.eigenstates())
    # print(len(model.lowering))
    # E, u = model.eigenstates()
    # level_scheme(E)
    # BO spectral density


