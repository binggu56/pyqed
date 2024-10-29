#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:21:26 2024

sliced bassi sets from STO-NG

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import numpy as np

class Gaussian1s:
    def __init__(self,ax,x):
        """
        Gaussian wavepackets ~exp(-ax/2 *(x-x_0)^2)
        """
        self.center = x
        self.alpha = ax

class STONG:
    """
    Slater Type Orbital fit with N primative gausians (STO-NG) type basis
    """
    def __init__(self,n,d,g):
        """
        n : int
            number of Gaussians
            
        d : contraction coeffiecents
        g : primative gaussians
        """
        self.n = n
        self.d = d
        self.g = g

        return

#Builds a STO-3G basis that best approximates a single slater type
#orbital with Slater orbital exponent zeta
def sto3g(center, zeta):
    scaling = zeta**2
    return STONG(3,[0.444635, 0.535328, 0.154329],
            [Gaussian1s(scaling*.109818, center),
             Gaussian1s(scaling*.405771, center),
             Gaussian1s(scaling*2.22766, center)])

#STO-3G basis for hydrogen
def sto3g_hydrogen(center):
    return sto3g(center, 1.24)

def sto3g_helium(center):
    return sto3g(center, 2.0925)


import jax
import jax.numpy as jnp
from jax import grad

import proplot as plt

x = np.linspace(-10, 10)

def coordinate_map(x):
    b = 0.1
    return x - jnp.arctan(b*x)/b




# df = grad(jnp.arctan)
# df2 = grad(df)

#print(df(2.0)) #(for _x in x)
# ax.plot(x, [df2(_x) for _x in x])
jacobian = jax.grad(coordinate_map)
J = np.array([jacobian(_x) for _x in x])

rho = coordinate_map(x)
# J = jax.jacrev(jnp.arctan)

fig, ax = plt.subplots()
# ax.plot(x, coordinate_map(x))
ax.plot(x, 1/J , '-o')
