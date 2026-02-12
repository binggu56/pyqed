#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 22:49:52 2023

@author: bing, Zi-Hao
"""

from pyqed import Mol, dag, multispin
from scipy.sparse import kron, coo_matrix
from functools import reduce
import numpy as np


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


class Frenkel2(Mol):
    def __init__(self, onsites, hopping, nsites):
        '''
        each site has two excited states, |1> and |2>, and one ground state |0>.
        onsites: list of onsite energies, [onsite1, onsite2, ...], if only one onsite energy is given, all sites have the same onsite energy.
        hopping: list of hopping energies, [j11, j22, j12], if only one hopping energy is given, all sites have the same hopping energy.
        '''

        # 0 is the ground state, 1 is the excited state 1, 2 is the excited state 2
        if type(hopping) is list:
            inter, intra = hopping
        else:
            inter = intra = hopping
        if type(onsites) is list:
            if len(onsites) == 2:
                onsite1, onsite2 = onsites
        else:
            onsite1 = onsite2 = onsites

        # sigma_p1 is the raising operator to the excited state 1
        sigma_p1 = np.zeros((3, 3), dtype=float)
        sigma_p1[0, 1] = 1
        # sigma_p2 is the raising operator to the excited state 2
        sigma_p2 = np.zeros((3, 3), dtype=float)
        sigma_p2[0, 2] = 1
        # unit is the unit matrix
        unit = np.eye(3, dtype=float)

        lowering1 = []
        lowering2 = []
        for i in range(nsites):
            # list of matrices to be kroneckered
            kron_list = [unit]*nsites
            kron_list[i] = sigma_p1
            lowering1.append(reduce(kron, kron_list))
            kron_list = [unit]*nsites
            kron_list[i] = sigma_p2
            lowering2.append(reduce(kron, kron_list))

        dimension = (np.shape(unit)[0])**nsites
        system_h = coo_matrix((dimension, dimension), dtype=np.complex128)
        print(dimension)

        for i in range(nsites):
            system_h += onsite1 * \
                dag(lowering1[i]) @ lowering1[i] + \
                onsite2 * dag(lowering2[i]) @ lowering2[i]
            system_h += inter * \
                dag(lowering1[i]) @ lowering2[i] + \
                inter * dag(lowering2[i]) @ lowering1[i]
        for i in range(nsites-1):
            system_h += intra * \
                dag(lowering1[i]) @ lowering2[i+1] + \
                intra * dag(lowering2[i+1]) @ lowering1[i]
        system_h.eliminate_zeros()

        lowering = lowering1 + lowering2
        edip = 0
        for l in lowering:
            edip += l + dag(l)

        self.H = system_h
        self.edip = edip
        self.dim = 3**nsites
        self.lowering = lowering


class Frenkel2_s(Mol):
    def __init__(self, onsites: list, hopping: list, nsites: int) -> None:
        '''
        each site has two excited states, |1> and |2>, and one ground state |0>.
        only the single excited states are involved in the Hamiltonian.
        onsites: list of onsite energies, [onsite1, onsite2, ...], if only one onsite energy is given, all sites have the same onsite energy.
        hopping: list of hopping energies, [j11, j22, j12], if only one hopping energy is given, all sites have the same hopping energy.
        return: Hamiltonian, dipole operator, dimension of the Hilbert space, lowering operators
        the state is represented by |g, e1, e2, ..., en, e1', e2', ..., en'>, where g is the ground state, en is the excited state 1 on the n-th site, en' is the excited state 2 on the n-th site.
        '''
        # 0 is the ground state, 1 is the excited state 1, 2 is the excited state 2

        self.dim = 2*nsites+1
        if type(hopping) is list:
            inter, intra1 = hopping
        else:
            inter = intra1 = hopping
        if type(onsites) is list:
            if len(onsites) == 2:
                onsite1, onsite2 = onsites
        else:
            onsite1 = onsite2 = onsites

        self.H = np.zeros((self.dim, self.dim), dtype=np.float64)
        lowering1 = [None]*nsites
        lowering2 = [None]*nsites
        for i in range(nsites):
            lowering1[i] = np.zeros((self.dim, self.dim), dtype=np.float64)
            lowering2[i] = np.zeros((self.dim, self.dim), dtype=np.float64)
            lowering1[i][0, i+1] = 1
            lowering2[i][0, nsites+i+1] = 1
        for i in range(nsites):
            self.H += onsite1 *  \
                dag(lowering1[i]) @ lowering1[i] + \
                onsite2 * dag(lowering2[i]) @ lowering2[i]
            self.H += inter * \
                dag(lowering1[i]) @ lowering2[i] + \
                inter * dag(lowering2[i]) @ lowering1[i]
        for i in range(nsites-1):
            self.H += intra1 * \
                dag(lowering1[i]) @ lowering2[i+1] + \
                intra1 * dag(lowering2[i+1]) @ lowering1[i]

        lowering = lowering1 + lowering2
        edip = 0
        for l in lowering:
            edip += l + dag(l)

        self.edip = edip
        self.lowering = lowering


if __name__ == '__main__':

    from pyqed import au2wavenumber, level_scheme
    from pyqed.models.exciton import Frenkel

    # parameters taken from JCP xxx
    onsite = 26000/au2wavenumber
    J = -260/au2wavenumber

    model = Frenkel(onsite, hopping=J, nsites=6)
    B = model.lowering
    B0 = B[0]
    print(dag(B0) @ B0)
    # print(model.eigenstates())
    # print(len(model.lowering))
    E, u = model.eigenstates()
    level_scheme(E)
    # BO spectral density
