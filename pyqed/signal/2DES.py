#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 22:52:18 2021

2DES taking into account explicitly the laser pulse envelope

@author: bing
"""

import scipy
import numpy as np

# x = np.linspace(-2, 2, 2**11+1)
# y = np.exp(-x**2)
# dx=x[1]-x[0]
# z = scipy.integrate.romb(y, dx=dx)
# print(z)

# print(np.sum(y)*dx)

from scipy.integrate import tplquad

from lime.units import au2ev, au2mev
from lime.optics import Pulse

def f(y,x,z, fa, fb, fc, eta=1e-5):
    return  p1.efield(-z) * p2.efield(y) * p3.efield(x)/(y + x - z - fc) *\
        1./(y-z-fb) * 1./(z+fa+1j*eta)

fmax = 4/au2ev
fa = 1/au2ev
fb = fa
fc = fa


def G(a,b,t, gamma=0.02/au2mev):
    '''
    phenomelogical Green's function in the Liouville space
    
    The dephasing is mimicked by decay constants. 

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    gamma : TYPE, optional
        DESCRIPTION. The default is 0.02/au2mev.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return -1j* np.heaviside(t, 1) * np.exp(-1j*(en[a]-en[b])*t - (decay[a]+decay[b])/2.*t) 

def gsb(dip, t3, t2, t1):
    r = 0
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[0,b] * dip[b,c] * dip[c,d]* dip[d,0] * G(d,c,omega3) * \
                G(d,b,t2) * G(0,d, t1)
    return r 

def response2_freq(dip, omega3, t2, omega1):
    r = 0
    a = 0 # initial state 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(c, nlevel):
                r += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * G(d,c,omega3) * \
                lineshape(d,b,t2) * G(a,b,omega1)
    return r 

def response3_freq(dip, omega3, t2, omega1):
    r = 0
    a = 0 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * G(d,c,omega3) * \
                lineshape(a,c,t2) * G(a,b,omega1)
    return r 

def response4_freq(dip, omega3, t2, omega1):
    r = 0
    a = 0 # initial state, assuming ground state here 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[0,b] * dip[b,c] * dip[c,d]* dip[d,0] * G(d,a,omega3) * \
                lineshape(c,a,t2) * G(d, a,omega1)
    return r 


def ESA(evals, dip, g_idx, e_idx, f_idx, gamma, t1, t2, t3):
    '''
    Excited state absorption component of the photon echo signal.
    In Liouville sapce, gg -> ge -> e'e -> fe -> ee

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    t2 : TYPE
        DESCRIPTION.
    omega1 : TYPE
        DESCRIPTION.
    g_idx: list of integers
        index for ground state (manifold)
    e_idx: list of integers
        index for e-states
    f_idx: list of integers
        index of f-states

    Returns
    -------
    signal : 2d array (len(pump), len(probe))
        DESCRIPTION.

    '''

    signal = 0
    a = 0 # initial state

    # sum-over-states
    for b in e_idx:

        G_ab = G(a, b, t1)

        for c in e_idx:
            G_cb = G(c, b, t2)

            for d in f_idx:

                G_db = G(d, b, t3)

                signal += dip[b,a] * dip[c,a] * dip[d,c]* dip[b,d] * \
                    G_db * G_cb * G_ab

    # 1 interaction in the bra side
    sign = -1
    return sign * signal


def GSB(evals, dip, g_idx, e_idx, gamma, t1, t2, t3):
    '''
    gg -> ge -> gg' -> e'g' -> g'g'

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    t2 : TYPE
        DESCRIPTION.
    omega1 : TYPE
        DESCRIPTION.
    g_idx: list of integers
        indexes for ground manifold
    e_idx: list of integers
        indexes for excited state manifold

    Returns
    -------
    chi : TYPE
        DESCRIPTION.

    '''

    signal = 0
    a = 0

    # sum-over-states
    for b in e_idx:
        G_ab = G(a, b, t1)

        for c in g_idx:
            G_ac = G(a, c, t2)

            for d in e_idx:
                G_dc = G(d, c, t3)

                signal += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * \
                    G_dc * G_ac * G_ab
    return signal


def SE(evals, dip, g_idx, e_idx, t1, t2, t3):
    '''
    Stimulated emission gg -> ge -> e'e -> g'e -> g'g' in the impulsive limit.
    The signal wave vector is ks = -k1 + k2 + k3

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    t2 : TYPE
        DESCRIPTION.
    omega1 : TYPE
        DESCRIPTION.
    g_idx: list of integers
        indexes for ground manifold
    e_idx: list of integers
        indexes for excited state manifold

    Returns
    -------
    chi : TYPE
        DESCRIPTION.

    '''

    signal = 0.0
    a = 0

    # sum-over-states
    for b in e_idx:
        G_ab = G(a, b, t1)

        for c in e_idx:
            G_cb = G(c, b, t2)

            for d in g_idx:

                G_cd = G(c, d, t3)

                signal += dip[a,b] * dip[c,a] * dip[d,c]* dip[b, d] * \
                    G_cd * G_cb * G_ab
    return signal

p1 = Pulse()
p2 = Pulse()
p3 = Pulse()

# scan pump and probe frequencies
# for Omega1:
#     for omega3:
#         # sum over states
#         for a :
#             for b :
#                 for c

z = dip[a, 0] * tplquad(f,0, fmax, lambda z: 0, lambda z:1, lambda z, x: 0, \
            lambda z, x: np.pi/2, args=[fa, fb, fc])
print(z)