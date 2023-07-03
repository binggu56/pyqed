#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:27:59 2023

@author: bing
"""

import numpy as np

from numpy import eye


norb = 1
nt = 10
dt = 0.02


class GF:
    def __init__(self, omega, beta, kind='boson'):
        self.omega = omega
        self.kind = kind

        self.retarded = None
        self.advanced = None
        self.lesser = None
        self.greater = None

    def retarded(self, dt, nt):
        pass

    def advanced(self):
        pass

    def lesser(self):
        pass

    def greater(self):
        pass




h = np.zeros((norb, norb)) # bare Hamiltonian
h[0, 0] = 1


Gret = np.zeros((nt, nt, norb, norb), dtype=complex)
# G0ret =
for k in range(nt):
    Gret[k, k] = -1j * eye(norb)

Gles = np.zeros((nt, nt, norb, norb))
# the lesser GF contains information of the initial state

# self-energy, retarded and lesser

sigma_ret = np.zeros((nt, nt, norb, norb), dtype=complex)
sigma_les = np.zeros((nt, nt, norb, norb), dtype=complex)



def integrate(nt):
    for n in range(nt):
        for m in range(n-1, -1, -1):
            Gret[n, m] = Gret[n, m+1] - h @ Gret[n, m+1] * dt # self-energy term

integrate(nt)

def self_energy():
    pass

