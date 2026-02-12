#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:44:15 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

## module householder
''' d,c = householder(a).
    Householder similarity transformation of matrix [a] to 
    tridiagonal form].

    p = computeP(a).
    Computes the acccumulated transformation matrix [p]
    after calling householder(a).
'''    
import numpy as np
import math

def householder(a): 
    """
    .. math::
        T = P.T a P

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    """
    n = len(a)
    for k in range(n-2):
        u = a[k+1:n,k]
        uMag = math.sqrt(np.dot(u,u))
        if u[0] < 0.0: uMag = -uMag
        u[0] = u[0] + uMag
        h = np.dot(u,u)/2.0
        v = np.dot(a[k+1:n,k+1:n],u)/h
        g = np.dot(u,v)/(2.0*h)
        v = v - g*u
        a[k+1:n,k+1:n] = a[k+1:n,k+1:n] - np.outer(v,u) \
                         - np.outer(u,v)
        a[k,k+1] = -uMag
    
    # transformation matrix
    p = np.identity(n)*1.0
    for k in range(n-2):
        u = a[k+1:n,k]
        h = np.dot(u,u)/2.0
        v = np.dot(p[1:n,k+1:n],u)/h           
        p[1:n,k+1:n] = p[1:n,k+1:n] - np.outer(v,u)    
    
    return np.diagonal(a),np.diagonal(a,1), p

# def computeP(a): 
#     n = len(a)
#     p = np.identity(n)*1.0
#     for k in range(n-2):
#         u = a[k+1:n,k]
#         h = np.dot(u,u)/2.0
#         v = np.dot(p[1:n,k+1:n],u)/h           
#         p[1:n,k+1:n] = p[1:n,k+1:n] - np.outer(v,u)
#     return p

a = np.random.rand(3, 3)
a = a + a.T 

b = a.copy()

d, c, p = householder(b)
# p = computeP(b)

print(a)
print(p.T @ a @ p)

print(d, c)
