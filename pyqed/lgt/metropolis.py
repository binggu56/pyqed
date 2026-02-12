#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:47:15 2025

@author: bingg
"""

import numpy as np
from numpy.random import uniform
from numpy import exp

def update(x):
    for j in range(0,N):
        old_x = x[j] # save original value
        old_Sj = S(j,x)
        x[j] = x[j] + uniform(-eps,eps) # update x[j]
        dS = S(j,x) - old_Sj # change in action
        if dS>0 and np.exp(-dS)<uniform(0,1):
            x[j] = old_x # restore old value

def S(j,x): # harm. osc. S
    jp = (j+1)%N # next site
    jm = (j-1)%N # previous site
    return a*x[j]**2/2 + x[j]*(x[j]-x[jp]-x[jm])/a

# import Numeric
# from whrandom import uniform
# from math import *

def compute_G(x,n):
    g = 0
    for j in range(0,N):
        g = g + x[j]*x[(j+n)%N]
    return g/N

def MCaverage(x,G):
    for j in range(0,N): # initialize x
        x[j] = 0
    for j in range(0,5*N_cor): # thermalize x
        update(x)
    for alpha in range(0,N_cf): # loop on random paths
        for j in range(0,N_cor):
            update(x)
        for n in range(0,N):
            G[alpha][n] = compute_G(x,n)
    for n in range(0,N): # compute MC averages
        avg_G = 0
        for alpha in range(0,N_cf):
            avg_G = avg_G + G[alpha][n]
        avg_G = avg_G/N_cf
        print("G(%d) = %g" % (n,avg_G))

def bootstrap(G):
    """
    returns a single bootstrap copy of ensemble G, consisting of N cf measurement

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.

    Returns
    -------
    G_bootstrap : TYPE
        DESCRIPTION.

    """
    N_cf = len(G)
    G_bootstrap = [] # new ensemble
    for i in range(0,N_cf):
        alpha = int(uniform(0,N_cf)) # choose random config
        G_bootstrap.append(G[alpha]) # keep G[alpha]
    return G_bootstrap

def bin(G,binsize):
    """
    make a binned copy of an ensemble of measurements G. The original ensemble consists of individual measurements G[alpha], one for each
    configuration. The function bin(G,binsize) bins the ensemble into bins of size binsize,
    averages the G’s within each bin, and returns an ensemble consisting of the averages.

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    binsize : TYPE
        DESCRIPTION.

    Returns
    -------
    G_binned : TYPE
        DESCRIPTION.

    """
    G_binned = [] # binned ensemble
    for i in range(0,len(G),binsize): # loop on bins
        G_avg = 0
    for j in range(0,binsize): # loop on bin elements
        G_avg = G_avg + G[i+j]
        G_binned.append(G_avg/binsize) # keep bin avg
    return G_binned

# set parameters:
N = 20
N_cor = 20
N_cf = 100
a = 0.5
eps = 1.4

class QMC:
    def __init__(self, n, ncorr, ncf, eps, a):
        """


        Parameters
        ----------
        n : TYPE
            DESCRIPTION.
        ncorr : TYPE
            DESCRIPTION.
        ncf : TYPE
            DESCRIPTION.
        eps : range for update positions
            Parameter eps determines the size of the update.
            It should be adjusted so that roughly half of all trial updates are accepted.
        a : TYPE
            lattice spacing.

        Returns
        -------
        None.

        """

# create arrays:
x = np.zeros((N,), float)
G = np.zeros((N_cf,N), float)

MCaverage(x, G)

def avg(G): # MC avg of G
    return np.sum(G)/len(G)

def sdev(G): # std dev of G
    g = np.asarray(G)
    return np.abs(avg(g**2)-avg(g)**2)**0.5

print('avg G\n',avg(G))
print('avg G (binned)\n',avg(bin(G,4)))
print('avg G (bootstrap)\n', avg(bootstrap(G)))


def deltaE(G): # Delta E(t)
    avgG = avg(G)
    adE = np.log(np.abs(avgG[:-1]/avgG[1:]))
    return adE/a



print('Delta E\n',deltaE(G))
print('Delta E (bootstrap)\n',deltaE(bootstrap(G)))

def bootstrap_deltaE(G,nbstrap=100): # Delta E + errors
    avgE = deltaE(G) # avg deltaE
    bsE = []

    for i in range(nbstrap): # bs copies of deltaE
        g = bootstrap(G)
        bsE.append(deltaE(g))

    bsE = np.array(bsE)
    sdevE = sdev(bsE) # spread of deltaE’s
    print('\n%2s %10s %10s' % "t","Delta E(t)","error")
    print(26*"-")
    for i in range(len(avgE)/2):
        print("%2d %10g %10g" % (i,avgE[i],sdevE[i]))

bootstrap_deltaE(G)