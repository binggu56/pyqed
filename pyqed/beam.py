#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:10:45 2021

@author: bing
"""

from numpy import exp, sqrt, pi
import numpy as np

def beam(rho, phi, z, k, pol=[1, 0, 0], l=0):
    """
    cylindrical vector beam

    Parameters
    ----------
    rho : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    pol : TYPE, optional
        DESCRIPTION. The default is [1, 0, 0].
    l : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # rho = sqrt(x**2 + y**2)
    # A = amplitude(rho, z)

    A = 1.


    return A * exp(1j * k * z + 1j * l *  phi)

phi = np.linspace(0, 2*pi, 128)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.plot(phi, beam(rho=1, phi=phi, z=1, k=1, l=10).real)
