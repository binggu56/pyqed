#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:16:49 2020

@author: bing

Entangled two-photon absorption
"""

import numpy as np

from numpy import exp, pi, sqrt

from lime.phys import lorentzian
from lime.units import au2ev, au2angstrom


def ETPA(omegap, E, edip, Te, g_idx=[0], e_idx=[], f_idx=[]):
    """
    ETPA signal with SOS formula
    """
    N = len(E)
    tdm = edip
    # gamma = np.zeros(nstates)
    # for j in range(1, N):
    #    gamma[j] = sum(tdm[:j, j]**2) * 0.0005
    # gamma[1:] = 0.0001

    # print('lifetimes of polariton states =  {} eV'.format(gamma * au2ev))

    omega1 = omegap * 0.5
    omega2 = omegap - omega1
    # flist = [3, 4] # final states list
    i = g_idx[0]

    A = np.zeros(N, dtype=complex)

    signal = 0.0

    for f in f_idx:
        for m in e_idx:
            A[f] += tdm[f, m] * tdm[m, i] * \
                    ((exp(1j * (omega1 - (en[m] - en[i]) + 1j * gamma[m]) * T) - 1.) / (omega1 - (en[m] - en[i]) + 1j * gamma[m]) \
                     + (exp(1j * (omega2 - (en[m] - en[i])) * T) - 1.)/(omega2 - (en[m] - en[i]) + 1j * gamma[m]))

        signal += np.abs(A[f])**2 * lorentzian(omegap - en[f] + en[i], gamma[f])

    return signal

def vacuum_efield(omega):
    '''
    Vacuum electric field fluctuations. The prefactor relating the electric
    field operator to the annihilation operator.

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    n = 1. # reflaction index
    A = (1e4 / au2angstrom)**2 # quantization area, ~ 1 micrometer squared
    c = 1./137.036
    return sqrt(2. * pi * omega / (c * n * A))



def transition_amplitude(E, edip, Te, omegap, sigmap, g_idx=[0], e_idx=[], \
                         f_idx=[], decay=1e-4):
    '''
    Transition amplitude for two-photon transition g -> f  with SOS formula.
    This is useful for TPA signal.

    SPDC type-II quantum light with Gaussian pump envelope and
    sinc phase matching condition is assumed here.

    The Gaussian pump ~ exp(-(w - wp)**2/4/sigmap^2)

    Parameters
    ----------
    E : TYPE
        DESCRIPTION.
    edip : ndarray
        electric dipole.
    Te : TYPE
        DESCRIPTION.
    omegap : TYPE
        DESCRIPTION.
    sigmap : TYPE
        DESCRIPTION.
    g_idx : TYPE, optional
        DESCRIPTION. The default is [0].
    e_idx : TYPE, optional
        DESCRIPTION. The default is [].
    f_idx : TYPE, optional
        DESCRIPTION. The default is [].
    decay : TYPE, optional
        DESCRIPTION. The default is 1e-4.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.

    '''
    # number of states
    N = len(E)
    en = E

    # lifetime of excited states
    gamma = np.zeros(N)
    gamma[1:] = decay
    print('lifetimes of polariton states =  {} eV'.format(decay * au2ev))

    # degenerate quantum light
    omega1 = omegap * 0.5 # signal
    omega2 = omegap - omega1 # idler

    # ground-state index
    i = g_idx[0]

    if len(g_idx) > 1:
        raise NotImplementedError('The ground state index g_idx has to be [0] \
                         Multiple g states are not implemented yet.')

    # transition amplitude for f-states
    A = np.zeros(N, dtype=complex)

    for f in f_idx:
        for m in e_idx:

            # transition amplitude through m with two time-orderings of
            # the incoming photons

            # signal photon interactions first

            A[f] += edip[f, m] * edip[m, i] * \
                    h(omega1 - (en[m]-en[i]) + 1j * gamma[m],  Te)

            # idler first
            A[f] += edip[f, m] * edip[m, i] * \
                    h(omega2 - (en[m] - en[i]) + 1j * gamma[m], Te)

        # pump envelope
        A[f] *= exp(-(E[f] - E[i] - omegap)**2/4./sigmap**2)

    return A * sqrt(pi/Te/sigmap) * vacuum_efield(omega1) *\
        vacuum_efield(omega2) * (2.*pi)**(3/4)

def h(z, a=1):
    '''
    indefinite integral of an exponential function

      exp(i x a) - 1
      --------------
            i * x

    Parameters
    ----------
    z : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return (exp(1j * z * a) - 1.)/(1j * z)

if __name__ == '__main__':
    # nlevel = 3
    # tau = [1.e-5, ] * nlevel
    # H = np.diag([0, 0.5, 0.8])

    from lime.units import au2fs, au2ev

    E = np.array([0., 0.6, 1.1, 1.3])/au2ev
    # gamma = [0, 0.02, 0.02]
    H = np.diag(E)

    from lime.mol import Mol
    from lime.optics import Biphoton

    from matplotlib import cm

    dip = np.zeros((len(E), len(E)))
    dip[1,2] = dip[2,1] = 1.
    dip[1,3] = dip[3,1] = 1.

    dip[0,1] = dip[1, 0] = 1.
    dip[3, 3] = 1.
    dip[0, 3] = dip[3,0] = 1.

    # mol = Mol(H, dip)
    # epp = Biphoton(0, 0.04/au2ev, Te=10./au2fs)
    # p = np.linspace(-4, 4, 256)/au2ev
    # q = p
    # epp.set_grid(p, q)

    # epp.get_jsa()
    # epp.plt_jsa()

    pump = np.linspace(0.5, 1.5, 100)/au2ev
    A = transition_amplitude(E,  edip=dip, Te=1/au2fs, omegap=1.2/au2ev,\
                                  sigmap=0.2/au2ev, e_idx=[1], f_idx=[2, 3])

    print((vacuum_efield(1.2/au2ev)*vacuum_efield(1.2/au2ev))**2)
