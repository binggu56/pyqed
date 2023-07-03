'''
Sum over system eigenstates formula for computing the nonlinear signals.

This is the simpliest method for nonlinear signals without going to
the Liouville space.

'''

import numpy as np
import proplot as plt
from numba import jit
import sys
from numpy import heaviside
from matplotlib import cm

from numpy import conj

from pyqed.phys import lorentzian, dag
from pyqed.units import au2ev, au2mev

from pyqed.style import subplots

# class SOS:
#     def __init__(self, mol):
#         self.mol = mol 
        
#     def absorption():
#         pass
    
# def electronic_polarizability(w, gidx, eidx, vidx, E, d, use_rwa=True):
#     """
#     Compute the vibrational/electronic polarizability using sum-over-states formula

#     \alpha_{ji}(w) = d_{jv} d_{vi}/(E_v - E_i - w)

#     Parameters
#     ----------
#     w : TYPE
#         DESCRIPTION.
#     Er : TYPE
#         eigenenergies of resonant states.
#     Ev : TYPE
#         eigenenergies of virtual states.
#     d : ndarray or list of two arrays [d_cg, d_vc]
#         DESCRIPTION.
#     use_rwa : TYPE, optional
#         DESCRIPTION. The default is True.

#     Returns
#     -------
#     a : TYPE
#         DESCRIPTION.

#     """
#     ng = len(gidx)
#     ne = len(eidx)
#     nv = len(vidx) # virtual states

#     assert d.shape == (nv, ne)

#     # denominator
#     dE = Ev[:, np.newaxis] - Er - w

#     a = dag(d).dot(d/dE)

#     return a


def polarizability(w, Er, Ev, d, use_rwa=True):
    """
    Compute the vibrational/electronic polarizability using sum-over-states formula

    \alpha_{ji}(w) = d_{jv} d_{vi}/(E_v - E_i - w)

    Parameters
    ----------
    w : TYPE
        DESCRIPTION.
    Er : TYPE
        eigenenergies of resonant states.
    Ev : TYPE
        eigenenergies of virtual states.
    d : TYPE
        DESCRIPTION.
    use_rwa : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    a : ndarray
        polarizability.

    """
    ne = len(Er)
    nv = len(Ev)

    assert d.shape == (nv, ne)

    # denominator
    dE = Ev[:, np.newaxis] - Er - w

    a = dag(d).dot(d/dE)

    return a


def absorption(mol, omegas, linewidth=None, plt_signal=True, fname=None, normalize=False, scale=1., yscale=None):
    '''
    SOS for linear absorption signal
    ..math::
        S(\omega) = 2 \pi \sum_f |\mu_{fg}|^2 \delta(\omega - \omega_{fg}).
    The delta function is replaced with a Lorentzian function.

    Parameters
    ----------
    omegas : 1d array
        detection frequency window for the signal
    H : 2D array
        Hamiltonian
    edip : 2d array
        electric dipole moment
    output : TYPE, optional
        DESCRIPTION. The default is None.
    gamma : float, optional
        Lifetime broadening. The default is 1./au2ev.
    normalize : TYPE, optional
        Normalize the maximum intensity of the signal as 1. The default is False.

    Returns
    -------
    signal : 1d array
        linear absorption signal at omegas

    '''

    edip = mol.edip_rms

    # set linewidth, this can be generalized to frequency-dependent linewidth
    if linewidth is None:
        print('Linewidth not specified, using 20 meV.')
        gamma = [20/au2mev, ] * mol.nstates
    else:
        gamma = [linewidth, ] * mol.nstates

    # if mol.gamma is None:
    #     gamma = [10/au2mev, ] * mol.nstates
    # else:
    #     gamma = mol.gamma

    eigenergies = mol.eigvals()
    # set the ground state energy to 0
    eigenergies = eigenergies - eigenergies[0]

    # assume the initial state is from the ground state 0
    signal = 0.0
    for j in range(1, mol.nstates):
        e = eigenergies[j]
        signal += abs(edip[j, 0])**2 * lorentzian(omegas-e, gamma[j])

    if normalize:
        signal /= max(signal)

    if plt_signal:

        fig, ax = plt.subplots(figsize=(4,3))

        ax.plot(omegas * au2ev, signal)

        for j, e in enumerate(eigenergies):
            ax.axvline(e * au2ev, 0., abs(edip[0, j])**2 * scale, color='grey')


        ax.set_xlim(min(omegas) * au2ev, max(omegas) * au2ev)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Absorption')
        if yscale:
            ax.set_yscale(yscale)


        # ax.set_ylim(0, 1)

        # fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, \
                            # left=0.17, top=0.96, right=0.96)

        if fname is not None:
            fig.savefig(fname, dpi=1200, transparent=True)

        return signal, fig, ax
    else:
        return signal

def linear_absorption(omegas, transition_energies, dip, output=None, \
                      gamma=1./au2ev, scale=1, normalize=False):
    '''
    SOS for linear absorption signal S = 2 pi |mu_{fg}|^2 delta(omega - omega_{fg}).
    The delta function is replaced with a Lorentzian function.

    Parameters
    ----------
    omegas : 1d array
        the frequency range for the signal
    transition_energies : TYPE
        DESCRIPTION.
    edip : 1d array
        transtion dipole moment
    output : TYPE, optional
        DESCRIPTION. The default is None.
    gamma : float, optional
        Lifetime broadening. The default is 1./au2ev.
    scale: float
        scale the transition dipole sticks
    normalize : TYPE, optional
        Normalize the maximum intensity of the signal as 1. The default is False.

    Returns
    -------
    signal : 1d array
        linear absorption signal at omegas

    '''

    signal = 0.0


    for j, transition_energy in enumerate(transition_energies):

        signal += dip[j]**2 * lorentzian(omegas-transition_energy, gamma)


    if normalize:
        signal /= max(signal)

    if output is not None:

        fig, ax = plt.subplots(figsize=(4,3))

        ax.plot(omegas * au2ev, signal)

        #scale = 1./np.sum(dip**2)

        for j, e in enumerate(transition_energies):
            ax.axvline(e * au2ev, 0., abs(dip[j])**2 * scale, color='grey')



        # ax.set_xlim(min(omegas) * au2ev, max(omegas) * au2ev)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Absorption')
        #ax.set_yscale('log')
        # ax.set_ylim(0, 1)

        # fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, \
                            # left=0.17, top=0.96, right=0.96)
        fig.savefig(output, dpi=1200, transparent=True)

    return signal

def TPA(E, dip, omegap, g_idx, e_idx, f_idx, gamma, degenerate=True):
    """
    TPA signal with classical light
    """
    if degenerate:
        omega1 = omegap * 0.5
        omega2 = omegap - omega1

    i = 0

    signal = 0

    for f in f_idx:

        tmp = 0.0

        for m in e_idx:

             p1 = dip[f, m] * dip[m, i] / (omega1 - (E[m] - E[i]) + 1j * gamma[m])
             p2 = dip[f, m] * dip[m, i] /(omega2 - (E[m] - E[i]) + 1j * gamma[m])
             # if abs(p1) > 10:
             #      print('0 -> photon a -> {} -> photon b -> {}'.format(m, f), p1)
             #      print('0 -> photon b -> {} -> photon a -> {}'.format(m, f), p2)

             tmp += (p1 + p2)

        signal += np.abs(tmp)**2 * lorentzian(omegap - E[f] + E[i], width=gamma[f])

    return signal


def TPA2D(E, dip, omegaps, omega1s, g_idx, e_idx, f_idx, gamma):
    """
    2D two-photon-absorption signal with classical light scanning the omegap = omega1 + omega2 and omega1
    """

    g = 0

    signal = np.zeros((len(omegaps), len(omega1s)))

    for i, omegap in enumerate(omegaps):

        for j, omega1 in enumerate(omega1s):

            omega2 = omegap - omega1

            for f in f_idx:

                tmp = 0.

                for m in e_idx:

                     tmp += dip[f, m] * dip[m, g] * ( 1./(omega1 - (E[m] - E[g]) + 1j * gamma[m])\
                      + 1./(omega2 - (E[m] - E[g]) + 1j * gamma[m]) )

                signal[i,j] += np.abs(tmp)**2 * lorentzian(omegap - E[f] + E[g], width=gamma[f])

    return signal

def TPA2D_time_order(E, dip, omegaps, omega1s, g_idx, e_idx, f_idx, gamma):
    """
    2D two-photon-absorption signal with classical light scanning the omegap = omega1 + omega2 and omega1
    """

    g = 0

    signal = np.zeros((len(omegaps), len(omega1s)))

    for i in  range(len(omegaps)):
        omegap = omegaps[i]

        for j in range(len(omega1s)):
            omega1 = omega1s[j]

            omega2 = omegap - omega1

            for f in f_idx:

                tmp = 0.
                for m in e_idx:
                     tmp += dip[f, m] * dip[m, g] * 1./(omega1 - (E[m] - E[g]) + 1j * gamma[m])

                signal[i,j] += np.abs(tmp)**2 * lorentzian(omegap - E[f] + E[g], width=gamma[f])

    return signal

def gaussian(x, width):
    return np.exp(-(x/width)**2)

def GF(E, a, b, t):
    '''
    Retarded propagator of the element |a><b| for time t

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # if t >= 0:
        # N = len(evals)
        # propagator = np.zeros((N,N), dtype=complex)

        # for a in range(N):
        #     for b in range(N):
        #         propagator[a, b] = np.exp(-1j * (evals[a] - evals[b]) * t - (gamma[a] + gamma[b])/2. * t)

        # return propagator
    return  -1j * heaviside(t, 0) * np.exp(-1j * (E[a] - E[b]) * t)
    # else:
    #     return 0.

@jit
def G(omega, E, a, b):
    '''
    Green's function in the frequency domain, i.e., FT of the retarded propagator

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    evals : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return 1./(omega - (E[a]- E[b]))



def ESA(evals, dip, omega1, omega3, tau2, g_idx, e_idx, f_idx, gamma):
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

    signal = np.zeros((len(omega1), len(omega3)), dtype=complex)
    a = 0 # initial state

    # for i in range(len(omega1)):
    #     pump = omega1[i]

    #     for j in range(len(omega3)):
    #         probe = omega3[j]

    pump, probe = np.meshgrid(omega1, omega3)

    # sum-over-states
    for b in e_idx:

        G_ab = 1./(pump - (evals[a]-evals[b]) + 1j * (gamma[a] + gamma[b])/2.0)

        for c in e_idx:
            U_cb = -1j * np.exp(-1j * (evals[c] - evals[b]) * tau2 - (gamma[c] + gamma[b])/2. * tau2)

            for d in f_idx:

                G_db = 1./(probe - (evals[d]-evals[b]) + 1j * (gamma[d] + gamma[b])/2.0)

                signal += dip[b,a] * dip[c,a] * dip[d,c]* dip[b,d] * \
                    G_db * U_cb * G_ab

    # 1 interaction in the bra side
    sign = -1
    return sign * signal


def _ESA(evals, dip, omega1, omega2, t3, g_idx, e_idx, f_idx, gamma, \
         dephasing=10/au2mev):
    '''
    Excited state absorption component of the photon echo signal.
    In Liouville sapce, gg -> ge -> e'e -> fe -> ee

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega1 : TYPE
        DESCRIPTION.
    omega2 : TYPE
        DESCRIPTION.
    t2 : TYPE
        DESCRIPTION.
    g_idx: list of integers
        index for ground state (manifold)
    e_idx: list of integers
        index for e-states
    f_idx: list of integers
        index of f-states
    dephasing: float
        pure dephasing rate
    Returns
    -------
    signal : 2d array (len(pump), len(probe))
        DESCRIPTION.

    '''

    signal = np.zeros((len(omega2), len(omega1)), dtype=complex)
    a = 0 # initial state

    pump, probe = np.meshgrid(omega1, omega2)

    N = len(evals)

    # define pure dephasing rate
    gammaD = np.ones((N, N), dtype=float) * dephasing
    np.fill_diagonal(gammaD, 0)

    # sum-over-states
    for b in e_idx:

        G_ab = 1./(pump - (evals[a]-evals[b]) + 1j * ((gamma[a] + gamma[b])/2.0 + gammaD[a, b]))

        for c in e_idx:
            U_cb = 1./(probe - (evals[c] - evals[b]) + 1j*((gamma[c] + gamma[b])/2. + gammaD[c, b]))

            for d in f_idx:

                G_db = -1j * np.exp(-1j * (evals[d]-evals[b])*t3 - \
                                    ((gamma[d] + gamma[b])/2.0 + gammaD[d, b])*t3)

                signal += dip[b,a] * dip[c,a] * dip[d,c]* dip[b,d] * \
                    G_db * U_cb * G_ab

    # 1 interaction in the bra side
    sign = -1
    return sign * signal

def GSB(evals, dip, omega1, omega3, tau2, g_idx, e_idx, gamma):
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
    n1, n3 = len(omega1), len(omega3)
    signal = np.zeros((n1, n3), dtype=complex)

    a = 0
    c = 0
    # for i in range(n1):
    #     pump = omega1[i]

    #     for j in range(n3):
    #         probe = omega3[j]

    pump, probe = np.meshgrid(omega1, omega3)

    # sum-over-states
    for b in e_idx:

        G_ab = 1./(pump - (evals[a]-evals[b]) + 1j * (gamma[a] + gamma[b])/2.0)

        # for c in g_idx:
        #     U_ac = -1j * np.exp(-1j * (evals[a] - evals[c]) * tau2 - (gamma[a] + gamma[c])/2. * tau2)

        for d in e_idx:

            G_dc = 1./(probe - (evals[d]-evals[c]) + 1j * (gamma[d] + gamma[c])/2.0)

            signal += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * \
                G_dc * G_ab
    return signal


# def _GSB(evals, dip, omega1, omega2, t3, g_idx, e_idx, gamma):
#     '''
#     GSB for photon echo
#     gg -> ge -> gg' -> e'g' -> g'g'

#     Parameters
#     ----------
#     evals : TYPE
#         DESCRIPTION.
#     dip : TYPE
#         DESCRIPTION.
#     omega3 : TYPE
#         DESCRIPTION.
#     t2 : TYPE
#         DESCRIPTION.
#     omega1 : TYPE
#         DESCRIPTION.
#     g_idx: list of integers
#         indexes for ground manifold
#     e_idx: list of integers
#         indexes for excited state manifold

#     Returns
#     -------
#     chi : TYPE
#         DESCRIPTION.

#     '''
#     n1, n3 = len(omega1), len(omega2)
#     signal = np.zeros((n1, n3), dtype=complex)

#     a = 0
#     c = 0

#     pump, probe = np.meshgrid(omega1, omega2)

#     # sum-over-states
#     for b in e_idx:

#         G_ab = 1./(pump - (evals[a]-evals[b]) + 1j * (gamma[a] + gamma[b])/2.0)

#         for d in e_idx:

#             G_dc = 1./(probe - (evals[d]-evals[c]) + 1j * (gamma[d] + gamma[c])/2.0)

#             signal += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * \
#                 G_dc * G_ab

#     return signal

def SE(evals, dip, omega1, omega3, tau2, g_idx, e_idx, gamma):
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

    signal = np.zeros((len(omega1), len(omega3)), dtype=complex)
    a = 0
    # for i in range(len(omega1)):
    #     pump = omega1[i]

    #     for j in range(len(omega3)):
    #         probe = omega3[j]

    pump, probe = np.meshgrid(omega1, omega3)

    # sum-over-states
    for b in e_idx:

        G_ab = 1./(pump - (evals[a]-evals[b]) + 1j * (gamma[a] + gamma[b])/2.0)

        for c in e_idx:
            U_cb = -1j * np.exp(-1j * (evals[c] - evals[b]) * tau2 - (gamma[c] + gamma[b])/2. * tau2)

            for d in g_idx:

                G_cd = 1./(probe - (evals[c]-evals[d]) + 1j * (gamma[c] + gamma[d])/2.0)

                signal += dip[a,b] * dip[c,a] * dip[d,c]* dip[b, d] * \
                    G_cd * U_cb * G_ab

    return signal


def _SE(E, dip, omega1, omega2, t3, g_idx, e_idx, gamma, dephasing=10/au2mev):
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

    signal = np.zeros((len(omega2), len(omega1)), dtype=complex)
    a = 0

    pump, probe = np.meshgrid(omega1, omega2)

    N = len(E)
    # define pure dephasing rate
    gammaD = np.ones((N, N)) * dephasing
    np.fill_diagonal(gammaD, 0)

    # sum-over-states
    for b in e_idx:

        G_ab = 1./(pump - (E[a]-E[b]) + 1j * ((gamma[a] + gamma[b])/2.0 + gammaD[a, b]))

        for c in e_idx:
            U_cb = 1./(probe - (E[c]-E[b])  + 1j* ((gamma[c] + gamma[b])/2. + gammaD[c, b]))

            for d in g_idx:

                G_cd = -1j * np.exp(-1j * (E[c]-E[d])*t3 - ((gamma[c] + gamma[d])/2.0 + gammaD[c, d])*t3)

                signal += dip[a,b] * dip[c,a] * dip[d,c]* dip[b, d] * \
                    G_cd * U_cb * G_ab

    return signal


def _photon_echo(evals, edip, omega1, omega3, t2, g_idx, e_idx, f_idx, gamma):
    """
    2D photon echo signal scanning omega1 and omega3 at population time t2.

    Parameters
    ----------
    evals : ndarray
        eigenvalues of system.
    edip : ndarray
        electric dipole matrix.
    omega1 : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    tau2 : TYPE
        DESCRIPTION.
    g_idx : TYPE
        DESCRIPTION.
    e_idx : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    gsb = GSB(evals, edip, omega1, omega3, t2, g_idx, e_idx, gamma)
    se = SE(evals, edip, omega1, omega3, t2, g_idx, e_idx, gamma)
    esa = ESA(evals, edip, omega1, omega3, t2, g_idx, e_idx, f_idx, gamma)

    return gsb + se + esa


def photon_echo_t3(mol, omega1, omega2, t3, g_idx=[0], e_idx=None, f_idx=None,\
                   fname='2DES', plt_signal=False, separate=False):
    """
    2D photon echo signal scanning omega1 and omega2 at detection time t3.

    The current implementation only applies for a single ground state.

    For a manifold of g states, the ground state bleaching neglected here has to be considered.

    Parameters
    ----------
    evals : ndarray
        eigenvalues of system.
    edip : ndarray
        electric dipole matrix.
    omega1 : TYPE
        DESCRIPTION.
    omega3 : TYPE
        DESCRIPTION.
    tau2 : TYPE
        DESCRIPTION.
    g_idx : TYPE
        DESCRIPTION.
    e_idx : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    separate: bool
        separate the ladder diagrams

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    E = mol.eigvals()
    edip = mol.edip_rms

    gamma = mol.gamma
    dephasing = mol.dephasing

    if gamma is None:
        raise ValueError('Please set the decay constants gamma first.')

    N = mol.nstates

    if e_idx is None: e_idx = range(1, N)
    if f_idx is None: f_idx = range(1, N)


    # gsb = _GSB(evals, edip, omega1, omega3, t2, g_idx, e_idx, gamma)
    se = _SE(E, edip, -omega1, omega2, t3, g_idx, e_idx, gamma, dephasing=dephasing)
    esa = _ESA(E, edip, -omega1, omega2, t3, g_idx, e_idx, f_idx, \
               gamma, dephasing=dephasing)

    S = se + esa



    if plt_signal == True:

        # make plots
        fig, ax = plt.subplots(refaspect=2)

        im = ax.contourf(omega1*au2ev, omega2*au2ev, S.real/abs(S).max(), #interpolation='bilinear',
                        cmap=cm.RdBu, lw=0.6,
                        origin='lower') #-abs(SPE).max())

        ax.set_xlabel(r'$-\Omega_1$ (eV)')
        ax.set_ylabel(r'$\Omega_2$ (eV)')

    if separate:
        np.savez(fname, omega1, omega2, se, esa)
        return se, esa
    else:
        np.savez(fname, omega1, omega2, S)
        return S

def photon_echo(mol, pump, probe, t2=0., g_idx=[0], e_idx=None, f_idx=None, fname='signal', \
                plt_signal=False, pol=None):
    """
    Photon echo signal for a multi-level system using SOS expression.

    Approximations:
        1. decay are included phenomelogically
        2. no population relaxation

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.
    pump : TYPE
        Omega1, conjugate variable of t1
    probe : TYPE
        Omega3, conjugate variable of t3
    t2 : TYPE
        population time.
    g_idx : TYPE
        DESCRIPTION.
    e_idx : TYPE
        DESCRIPTION.
    f_idx : TYPE
        DESCRIPTION.
    gamma : float
        decay rates for excited states.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    S : TYPE
        DESCRIPTION.

    """

    E = mol.eigvals()
    dip = mol.edip_rms

    gamma = mol.gamma

    if gamma is None:
        raise ValueError('Please set the decay constants gamma first.')

    N = mol.nstates

    if e_idx is None: e_idx = range(N)
    if f_idx is None: f_idx = range(N)

    # compute the signal
    S = _photon_echo(E, dip, omega1=-pump, omega3=probe, t2=t2, g_idx=g_idx, e_idx=e_idx, f_idx=f_idx,\
              gamma=gamma)

    np.savez(fname, pump, probe, S)

    if plt_signal == True:

        # make plots
        fig, ax = plt.subplots()

        omega_min = min(pump) * au2ev
        omega_max = max(pump) * au2ev

        im = ax.contour(S.real.T/abs(S).max(), #interpolation='bilinear',
                        cmap=cm.RdBu,
                        origin='lower', extent=[omega_min, omega_max, omega_min, omega_max],
                        vmax=1, vmin=-1) #-abs(SPE).max())

        ax.plot(pump*au2ev, probe*au2ev, '--', lw=1, color='grey')
        # ax.axhline(y=1.1, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
        # ax.axhline(y=0.9, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
        #
        # ax.axvline(x=1.1, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
        # ax.axvline(x=0.9, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
        #im = ax.contour(SPE,
        #               origin='lower', extent=[0.8, omega_max, omega_min, omega_max],
        #               vmax=1, vmin=0) #-abs(SPE).max())

        ax.set_xlabel(r'$-\Omega_1$ (eV)')
        ax.set_ylabel(r'$\Omega_3$ (eV)')
        #ax.set_title(r'$T_2 = $' + str(t2))

        # plt.colorbar(im)

        # fig.subplots_adjust(hspace=0.0,wspace=0.0,bottom=0.14,left=0.0,top=0.95,right=0.98)

    return S

def DQC_R1(evals, dip, omega1=None, omega2=[], omega3=None, tau1=None, tau3=None,\
           g_idx=[0], e_idx=None, f_idx=None, gamma=None):
    '''
    Double quantum coherence, diagram 1:
        gg -> eg -> fg -> fe' -> e'e' in the impulsive limit.
    The signal wave vector is ks = k1 + k2 - k3

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

    a = 0
    if omega3 is None and tau3 is not None:

        signal = np.zeros((len(omega1), len(omega2)), dtype=complex)

        for i in range(len(omega1)):
            pump = omega1[i]

            for j in range(len(omega2)):
                probe = omega2[j]

                # sum-over-states
                for b in e_idx:

                    G_ba = 1./(probe - (evals[b]-evals[a]) + 1j * (gamma[b] + gamma[a])/2.0)


                    for c in f_idx:
                        G_ca = 1./(probe - (evals[c]-evals[a]) + 1j * (gamma[c] + gamma[a])/2.0)


                        for d in e_idx:

                            U_cd = -1j * np.exp(-1j * (evals[c] - evals[d]) * tau3 - (gamma[c] + gamma[d])/2. * tau3)


                            signal[i,j] += dip[b, a] * dip[c,b] * dip[d,a]* dip[d,c] * \
                                G_ba * G_ca * U_cd

    elif omega1 is None and tau1 is not None:

        signal = np.zeros((len(omega2), len(omega3)), dtype=complex)


        for i in range(len(omega2)):
            pump = omega2[i]

            for j in range(len(omega3)):
                probe = omega3[j]

                # sum-over-states
                for b in e_idx:

                    U_ba =  -1j * np.exp(-1j * (evals[b] - evals[a]) * tau1 - (gamma[b] + gamma[a])/2. * tau1)


                    for c in f_idx:
                        G_ca = 1./(pump - (evals[c]-evals[a]) + 1j * (gamma[c] + gamma[a])/2.0)


                        for d in e_idx:

                            G_cd = 1./(probe - (evals[c]-evals[d]) + 1j * (gamma[c] + gamma[d])/2.0)


                            signal[i,j] += dip[b, a] * dip[c,b] * dip[d,a]* dip[d,c] * \
                                U_ba * G_ca * G_cd

    # one interaction in the bra side
    sign = -1
    return sign * signal

def DQC_R2(evals, dip, omega1=None, omega2=[], omega3=None, tau1=None, tau3=None,\
           g_idx=[0], e_idx=None, f_idx=None, gamma=None):
    '''
    Double quantum coherence, diagram 2:
        gg -> eg -> fg -> eg -> gg in the impulsive limit.
    The signal wave vector is ks = k1 + k2 - k3

    Parameters
    ----------
    evals : TYPE
        DESCRIPTION.
    dip : TYPE
        DESCRIPTION.
    omega1 : TYPE, optional
        DESCRIPTION. The default is None.
    omega2 : TYPE, optional
        DESCRIPTION. The default is [].
    omega3 : TYPE, optional
        DESCRIPTION. The default is None.
    tau1 : TYPE, optional
        DESCRIPTION. The default is None.
    tau3 : TYPE, optional
        DESCRIPTION. The default is None.
    g_idx : TYPE, optional
        DESCRIPTION. The default is [0].
    e_idx : TYPE, optional
        DESCRIPTION. The default is None.
    f_idx : TYPE, optional
        DESCRIPTION. The default is None.
    gamma : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    signal : TYPE
        DESCRIPTION.

    '''


    a = 0

    if omega3 is None and tau3 is not None:

        signal = np.zeros((len(omega1), len(omega2)), dtype=complex)

        for i in range(len(omega1)):
            pump = omega1[i]
            for j in range(len(omega2)):
                probe = omega2[j]

                # sum-over-states
                for b in e_idx:

                    G_ba = 1./(pump - (evals[b]-evals[a]) + 1j * (gamma[b] + gamma[a])/2.0)


                    for c in f_idx:
                        G_ca = 1./(probe - (evals[c]-evals[a]) + 1j * (gamma[c] + gamma[a])/2.0)


                        for d in e_idx:

                            U_da =  -1j * np.exp(-1j * (evals[d] - evals[a]) * tau3 - (gamma[d] + gamma[a])/2. * tau3)


                            signal[i,j] += dip[b, a] * dip[c,b] * dip[d,c]* dip[a,d] * \
                                G_ba * G_ca * U_da

    elif omega1 is None and tau1 is not None:

        signal = np.zeros((len(omega2), len(omega3)), dtype=complex)

        for i in range(len(omega2)):
            pump = omega2[i]
            for j in range(len(omega3)):
                probe = omega3[j]

                # sum-over-states
                for b in e_idx:

                    U_ba = np.exp(-1j * (evals[b] - evals[a]) * tau1 - (gamma[b] + gamma[a])/2. * tau1)

                    for c in f_idx:
                        G_ca = 1./(pump - (evals[c]-evals[a]) + 1j * (gamma[c] + gamma[a])/2.0)

                        for d in e_idx:

                            G_da = 1./(probe - (evals[d]-evals[a]) + 1j * (gamma[d] + gamma[a])/2.0)

                            signal[i,j] += dip[b, a] * dip[c,b] * dip[d,c]* dip[a,d] * \
                                U_ba * G_ca * G_da
    else:
        raise Exception('Input Error! Please specify either omega1, tau3 or omega3, tau1.')

    # positive sign due to 0 interactions at the bra side
    sign = 1
    return sign * signal

# def spontaneous_photon_echo(E, dip, pump, probe, tau2=0.0, normalize=True):
#     """

#     Compute the spontaneous photon echo signal.

#     Parameters
#     ----------
#     E : TYPE
#         DESCRIPTION.
#     dip : TYPE
#         DESCRIPTION.
#     pump: 1d array
#         pump frequency of the first pulse
#     probe: 1d array
#         probe frequency of the third pulse
#     tau2: float
#         time-delay between the second and third pulses. The default is 0.0.

#     Returns
#     -------
#     None.

#     """

#     signal = np.zeros((len(pump), len(probe)))

#     for i in range(len(pump)):
#         for j in range(len(probe)):

#             signal[i,j] = response2_freq(E, dip, probe[j], tau2, pump[i]) + \
#                           response3_freq(E, dip, probe[j], tau2, pump[i])

#     if normalize:
#         signal /= abs(signal).max() # normalize

#     return signal


def etpa(omegaps, mol, epp, g_idx, e_idx, f_idx):
    """
    ETPA signal with temporal modes (TMs).
    The JSA is reconstructed with the TMs first.

    Parameters
    ----------
    omegaps : TYPE
        DESCRIPTION.
    g_idx : TYPE
        DESCRIPTION.
    e_idx : TYPE
        DESCRIPTION.
    f_idx : TYPE
        DESCRIPTION.

    Returns
    -------
    signal : TYPE
        DESCRIPTION.

    """

    Es = mol.eigenenergies()
    edip = mol.edip

    # joint temporal amplitude
    t1, t2, jta = epp.get_jta()
    return _etpa(omegaps, Es, edip, jta, t1, t2, g_idx, e_idx, f_idx)


# @jit
def _etpa(omegaps, Es, edip, jta, t1, t2, g_idx, e_idx, f_idx):
    """
    internal function to compute the ETPA signal.

    The double time integrals are computed numerically.

    Parameters
    ----------
    omegaps: pump center frequencies
    Es: eigenenergies
    edip: electric dipole operator
    jta: 2d array
        joint temporal amplitude
    t1: 1d array
    t2: 1d array
    g_idx: ground-state manifold
    e_idx: intermediate states
    f_idx: final states

    Returns
    -------
    signal: 1d array

    """
    # setup the temporal grid
    T1, T2 = np.meshgrid(t1, t2)

    # discrete heaviside function
    theta = heaviside(T2 - T1, 0.5)

    signal = np.zeros(len(omegaps), dtype=complex)
    g = g_idx
    for j, omegap in enumerate(omegaps): # loop over pump frequencies

        omega1 = omega2 = omegap/2.

        for f in f_idx: # loop over final states
            for e in e_idx:

                detuning2 = Es[f] - Es[e] - omega2
                detuning1 = Es[e] - Es[g] - omega1

                D = edip[e, g] * edip[f, e]

                signal[j] += D * np.sum(theta * np.exp(1j * detuning2 * T2 +
                                                    1j * detuning1 * T1) * jta)

                detuning2 = Es[f] - Es[e] - omega1
                detuning1 = Es[e] - Es[g] - omega2
                signal[j] += D * np.sum(theta * np.exp(1j * detuning2 * T2 +
                                                    1j * detuning1 * T1) * jta.T)

    return signal

def test_etpa():
    epp = Biphoton(0, 0.04/au2ev, Te=10./au2fs)
    p = np.linspace(-4, 4, 256)/au2ev
    q = p
    epp.set_grid(p, q)

    epp.get_jsa()
    # epp.plt_jsa()

    pump = np.linspace(0.5, 1.5, 100)/au2ev
    signal = etpa(pump, mol, epp, [0], [1, 2, 3], [2, 3])

    fig, ax = subplots()
    ax.plot(pump*au2ev, np.abs(signal)**2)
    plt.show()
    return

def cars(E, edip, shift, omega1, t2=0, gamma=10/au2mev):
    '''
    two pump pulses followed by a stimulated raman probe.

    The first, second, and fourth pulses are assumed impulse,
    the thrid pulse is cw.


    S = \sum_{b,a = e} 2 * pi * delta(Eshift - omega_{ba}) * mu_{bg} *
        mu_{ag} * alpha_{ba}

    Parameters
    ----------
    E : TYPE
        DESCRIPTION.
    edip : TYPE
        DESCRIPTION.
    t1 : TYPE
        time decay between pulses 1 and 2
    t2 : TYPE, optional
        time delay between pulse 2 and 4 (stokes beam). The default is 0.

    Returns
    -------
    S : TYPE
        DESCRIPTION.

    '''

    N = len(E)
    g = 0
    S = 0
    alpha = np.ones((N, N))
    np.fill_diagonal(alpha, 0)

    for a in range(1, N):
        for b in range(1, N):
            S += edip[b, g] * edip[a, g] * alpha[b, a] * np.outer(lorentzian(shift - (E[b] - E[a]), gamma), \
                1./(omega1 - (E[a] - E[g]) + 1j * gamma))

    return S

def mcd(mol, omegas):
    '''
    magentic circular dichroism signal with SOS

    The electronic structure data should contain the B field ready,
    not the bare quantities.

    B = (0, 0, Bz)

    Reference:
        Shichao Sun, David Williams-Young, and Xiaosong Li, JCTC, 2019, 15, 3162-3169

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.
    omegas : TYPE
        DESCRIPTION.

    Returns
    -------
    signal : TYPE
        DESCRIPTION.

    '''
    signal = 0.

    mu = mol.edip[0, :, :]

    E = mol.eigvals()
    gamma = mol.gamma

    nstates = mol.nstates

    for nst in range(1, nstates):
        signal += np.imag(mu[nst, 0] * conj(mu[nst, 1]) - mu[nst, 1] * conj(mu[nst, 0]))\
             * lorentzian(omegas - E[nst], gamma[nst])


    return signal


def test_model():

    E = np.array([0., 0.6, 1.1])/au2ev
    N = len(E)

    gamma = np.array([0, 0.002, 0.002])/au2ev
    H = np.diag(E)

    dip = np.zeros((N, N, 3), dtype=complex)
    dip[1,2, :] = [1.+0.5j, 1.+0.1j, 0]

    dip[2,1, :] = conj(dip[1, 2, :])
    # dip[1,3, :] = dip[3,1] = 1.

    dip[0, 1, :] = [1.+0.2j, 0.5-0.1j, 0]
    dip[1, 0, :] = conj(dip[0, 1, :])

    # dip[3, 3] = 1.
    # dip[0, 3] = dip[3,0] = [0.5, 1, 0]

    mol = Mol(H, dip)
    mol.set_decay(gamma)

    return mol



if __name__ == '__main__':

    from lime.units import au2fs
    from lime.style import subplots

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    E = np.array([0., 0.5, 1.1, 1.3])/au2ev
    gamma = [0, 0.002, 0.002, 0.002]
    H = np.diag(E)

    from lime.mol import Mol
    from lime.optics import Biphoton

    from matplotlib import cm

    dip = np.zeros((len(E), len(E)))
    dip[1,2] = dip[2,1] = 1.
    dip[1,3] = dip[3,1] = 1.

    dip[0,1] = dip[1, 0] = 1.
    dip[0, 3] = dip[3,0] = 1
    dip[0, 2] = dip[2,0] = 1

    mol = Mol(H, edip_rms=dip)
    mol.set_decay_for_all(50/au2mev)

    mol.absorption(omegas=np.linspace(0., 2, 100)/au2ev, plt_signal=True)

    pump = np.linspace(0., 2, 100)/au2ev
    probe = np.linspace(0, 1, 100)/au2ev

    g_idx=[0]
    e_idx= [1, 2, 3]
    f_idx=[2, 3]

    # photon_echo(mol, pump, probe, t2=1e-3, g_idx=g_idx, e_idx=e_idx, f_idx=f_idx, plt_signal=True)

    # photon_echo_t3(mol, omega1=pump, omega2=probe, t3=1e-3, g_idx=g_idx)

