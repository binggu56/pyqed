#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:47:15 2026

@author: gugroup
"""

from pyqed.wpd import SPO2
import numpy as np 

def sho(x, y, k = 1., x0 = 0., y0 = 0.):
    """Usage:
            V = harmosc_factory(**kwargs)

    Return a two-dimensional harmonic oscillator potential V(x, y)
    with wavenumber k.
    i.e. V(x, y) = 1/2 * k * ((x - x0)^2 + (y - y0)^2)

    Keyword arguments
    @param[in] k    wavenumber of the SHO potential (default=1)
    @param[in] x0   x-displacement from origin (default=0)
    @param[in] y0   y-displacement from origin (default=0)
    @returns   V    2-D SHO potential V(x)
    """
    return 0.5 * (x - x0)**2 + 0.5 * (y - y0)**2 + 2*x*y + x**2 * y + x * y**2 + x**2*y**2
    
nx, ny = 15, 15
dvr = DVR2((-6,6), (-6,6), nx, ny)


dvr.v(sho)
E, U = dvr.run(k=3)


