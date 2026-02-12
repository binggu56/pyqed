#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:13:29 2025

2D Lattice Gauge Theory based on  

https://github.com/julian-urban/lattice-phi4/

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import numpy as np
import copy
from tqdm import tqdm

# import ultraplot as plt 

import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid')

def get_action(phi, k, l):
    """
    compute the action 
    
    .. math::
        
        S = - 2 \kappa \sum_\mu \phi_n \phi_{n+\mu} + (1 - 2\lambda) \phi_n^2 + 
            \lambda \phi_n^4

    Parameters
    ----------
    phi : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    l : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.sum(-2 * k * phi * (np.roll(phi, 1, 0) + np.roll(phi, 1, 1))
                  + (1 - 2 * l) * phi**2 + l * phi**4)

def get_drift(phi, k, l):
    """
    drift term 
    
    .. math::
        
        K = -\frac{\delta S}{\delta \phi} = 2\kappa (\phi_{n + \mu} + \phi_{n - \mu})\
            + 2\phi(x) (2\lambda (1 - \phi(x))^2 - 1)

    Parameters
    ----------
    phi : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    l : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return (2 * k * (np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
                     + np.roll(phi, 1, 1) + np.roll(phi, -1, 1))
            + 2 * phi * (2 * l * (1 - phi**2) - 1))

def get_hamiltonian(chi, action):
    """
    Hamiltonian 
    
    .. math::
        
        H = 1/2 \chi^2 + S
        
        \chi = \partial_0 \phi

    Parameters
    ----------
    chi : TYPE
        DESCRIPTION.
    action : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return 0.5 * np.sum(chi**2) + action


def hmc(phi_0, S_0, k, l, n_steps=100):
    """
    
    Hamiltonian Monte Carlo simulation
    
    Refs:
        PHYSICAL REVIEW D 96, 114505 (2017)

    Parameters
    ----------
    phi_0 : TYPE
        DESCRIPTION.
    S_0 : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    l : TYPE
        DESCRIPTION.
    n_steps : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    bool
        DESCRIPTION.

    """
    dt = 1 / n_steps

    phi = phi_0
    chi = np.random.randn(*phi.shape)
    H_0 = get_hamiltonian(chi, S_0)

    chi += 0.5 * dt * get_drift(phi, k, l)
    for i in range(n_steps-1):
        phi += dt * chi
        chi += dt * get_drift(phi, k, l)
    phi += dt * chi
    chi += 0.5 * dt * get_drift(phi, k, l)

    S = get_action(phi, k, l)
    dH = get_hamiltonian(chi, S) - H_0

    if dH > 0:
        if np.random.rand() >= np.exp(-dH):
            return phi_0, S_0, False
    return phi, S, True

L = 32 # 32 x 32 lattice 
k = 0.3
l = 0.02

phi = np.random.randn(L,L) # a random configuration 
S = get_action(phi, k, l) # action

for _ in tqdm(range(1000)):
    phi, S, accepted = hmc(phi, S, k, l)

cfgs = []

for i in tqdm(range(10000)):
    phi, S, accepted = hmc(phi, S, k, l)
    
    if i % 10 == 0:
        cfgs.append(copy.deepcopy(phi))
        
cfgs = np.array(cfgs)

def jackknife(samples: np.ndarray):
    """Return mean and estimated lower error bound."""
    means = []

    for i in range(samples.shape[0]):
        means.append(np.delete(samples, i, axis=0).mean(axis=0))

    means = np.asarray(means)
    mean = means.mean(axis=0)
    error = np.sqrt((samples.shape[0] - 1) * np.mean(np.square(means - mean), axis=0))
    
    return mean, error

def get_mag(cfgs: np.ndarray):
    """Return mean and error of magnetization."""
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])
    return jackknife(cfgs.mean(axis=axis))

def get_abs_mag(cfgs: np.ndarray):
    """Return mean and error of absolute magnetization."""
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])
    return jackknife(np.abs(cfgs.mean(axis=axis)))

def get_chi2(cfgs: np.ndarray):
    """Return mean and error of suceptibility."""
    V = np.prod(cfgs.shape[1:])
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])
    mags = cfgs.mean(axis=axis)
    return jackknife(V * (mags**2 - mags.mean()**2))

def get_corr_func(cfgs: np.ndarray):
    """Return connected two-point correlation function with errors for symmetric lattices."""
    mag_sq = np.mean(cfgs)**2
    corr_func = []
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])

    for i in range(1, cfgs.shape[1], 1):
        corrs = []

        for mu in range(len(cfgs.shape)-1):
            corrs.append(np.mean(cfgs * np.roll(cfgs, i, mu+1), axis=axis))

        corrs = np.array(corrs).mean(axis=0)
        corr_mean, corr_err = jackknife(corrs - mag_sq)
        corr_func.append([i, corr_mean, corr_err])

    return np.array(corr_func)

M, M_err = get_mag(cfgs)
M_abs, M_abs_err = get_abs_mag(cfgs)
chi2, chi2_err = get_chi2(cfgs)

print("M = %.4f +/- %.4f" % (M, M_err))
print("|M| = %.4f +/- %.4f" % (M_abs, M_abs_err))
print("chi2 = %.4f +/- %.4f" % (chi2, chi2_err))

corr_func = get_corr_func(cfgs)

fig, ax = plt.subplots(1,1, dpi=125, figsize=(8,8))
plt.xticks([i for i in range(1, L, 4)])
ax.errorbar(corr_func[:,0], corr_func[:,1], yerr=corr_func[:,2], label='2-point correlator')
plt.legend(prop={'size': 16})
plt.show()
