#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:09:37 2023

@author: bing
"""

from scipy.interpolate import LinearNDInterpolator
import numpy as np
# import matplotlib.pyplot as plt
import proplot as plt
from numba import vectorize, float64, int32

rng = np.random.default_rng()
x = rng.random(10) - 0.5
y = rng.random(10) - 0.5
z = np.hypot(x, y)
print(z.shape)

X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
interp = LinearNDInterpolator(list(zip(x, y)), z)
Z = interp(X, Y)
# plt.pcolormesh(X, Y, Z, shading='auto')
# plt.plot(x, y, "ok", label="input point")
# plt.legend()
# plt.colorbar()
# plt.axis("equal")
# plt.show()

# @vectorize([float64(float64, int32, int32, float64[:])])
def nodal_basis1D(x, l, j, domain):
    
    h = (domain[1] - domain[0])/2**l
    
    return np.maximum(0, 1. - np.abs((x - domain[0])/h  -  j))

# @vectorize
def nodal_basis(x, l, j, domain=None):
    
    ndim = len(l)

    if domain is None:
        domain=[[0, 1], ] * ndim 
    
    r = 1
    for d in range(ndim):
        r *= nodal_basis1D(x[d], l[d], j[d], domain[d])
    
    return r

from pyqed.phys import gwp

# without boundary, the length of x is 2**l - 1 
# l = 4
l = [4, 4]
x = np.linspace(-4, 4, 2**l[0], endpoint=False)[1:] 
y = np.linspace(-4, 4, 2**l[1], endpoint=False)[1:] 

nx = len(x)
ny = len(y)

j = [2, 3]
b = np.zeros((nx, ny))
for i in range(nx):
    for k in range(ny):
        b[i,k] = nodal_basis([x[i], y[k]], l=l, j=j, domain=[np.array([-4., 4]),] * 2)

fig, ax = plt.subplots()
ax.imshow(b, colorbar='right')
# ax.plot(x, nodal_basis(x, l=4, j=0, domain=[-4, 4]), '-o')    