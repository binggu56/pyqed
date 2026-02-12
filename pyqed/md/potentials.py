#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:28:21 2024

@author: bingg
"""

def potentials(epsilon, sigma, r, derivative=False):
    """
    this function returns the derivative of the potential, i.e., the force,


    or the potential energy:



    Here,
     is the distance at which the potential
     is zero,
     is the depth of the potential well, and
     is a cutoff distance. For
    ,
     and
    .



    Parameters
    ----------
    epsilon : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    derivative : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if derivative:
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
    else:
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)