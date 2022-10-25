#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:16:53 2020

@author: Bing Gu

Basic module for many-level systems
"""
from __future__ import absolute_import

from typing import Union, Iterable

import numpy as np
import scipy.integrate
from numpy.core._multiarray_umath import ndarray
from scipy.sparse import csr_matrix, lil_matrix, identity, kron, \
    linalg, issparse
from dataclasses import dataclass

import numba
import sys
import warnings

import proplot as plt

from pyqed.units import au2fs, au2ev
from pyqed.signal import sos
from pyqed.phys import dag, quantum_dynamics, \
    obs, basis, isdiag, jump, multimode, transform, rk4, tdse, \
        isdiag

from pyqed.units import au2wavenumber, au2fs

# def rk4(rho, fun, dt, *args):
#     """
#     Runge-Kutta method
#     """
#     dt2 = dt/2.0

#     k1 = fun(rho, *args )
#     k2 = fun(rho + k1*dt2, *args)
#     k3 = fun(rho + k2*dt2, *args)
#     k4 = fun(rho + k3*dt, *args)

#     rho += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

#     return rho

# def tdse(psi, H):
#     return -1j * H @ psi

def read_input(fname_e, fname_edip,  g_included=True):
    """
    Read input data from quantum chemistry output.

    Parameters
    ----------
    fname_e : str
        filename for the energy levels.
    fname_edip : list
        filenames for the electric dipole moment.
    g_included : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    mol : TYPE
        DESCRIPTION.

    """

    E = np.genfromtxt(fname_e)

    if not g_included:
        E = np.insert(E, 0, 0)

    nstates = len(E)

    # read electric dipole moment g-e

    edip = np.zeros((nstates, nstates, 3))

    for k in range(3):
        edip[:,:, k] = np.genfromtxt(fname_edip[k], unpack=False)


    # H = np.diag(E)
    # mol = Mol(H, edip)
    # return mol
    return E, edip


class Result:
    def __init__(self, description=None, psi0=None, rho0=None, dt=None, \
                 Nt=None, times=None, t0=None, nout=None):
        self.description = description
        self.dt = dt
        self.timesteps = Nt
        self.observables = None
        self.rholist = None
        # self.psilist = [psi0]
        self.psilist = []

        # self._psilist = None
        self.psi = None
        self.rho0 = rho0
        self.psi0 = psi0
        self.nout = nout
        self.times = t0 + np.arange(Nt//nout) * dt * nout
        return

    # @property
    # def psilist(self):
    #     return self.psilist

    # @psilist.setter
    # def psilist(self, psilist):
    #     self.psilist = psilist

    def expect(self):
        return self.observables

    def analyze(self):



        nobs = self.observables.shape[-1]
        if nobs == 0:
            raise ValueError('There are no observables to analyze.')

        import proplot as plt
        fig, ax = plt.subplots()

        for j in range(nobs):
            ax.plot(self.times * au2fs, self.observables[:,j].real)

        ax.format(xlabel='Time (fs)')

        return

    def dump(self, fname):
        """
        save results to disk

        Parameters
        ----------
        fname : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        import pickle
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def save(self, fname):
        self.dump(fname)

    # def times(self):
    #     if dt is not None & Nt is not None:
    #         return np.arange(self.Nt) * self.dt
    #     else:
    #         sys.exit("ERROR: Either dt or Nt is None.")


class Mol:
    def __init__(self, H, edip=None, edip_rms=None, gamma=None):
        """
        Class for multi-level systems.

        All signals computed using SOS formula can be directly called from the
        Mol objective.

        More sophisticated ways to compute the spectra should be done with
        specific method, e.g., MESolver.

        Parameters
        ----------
        H : TYPE
            DESCRIPTION.
        dip : TYPE
            DESCRIPTION.
        rho : TYPE, optional
            DESCRIPTION. The default is None.
        tau: 1d array
            lifetime of energy levels
        Returns
        -------
        None.

        """
        self.H = H
        self.nonhermH = None
        self.h = H
        #        self.initial_state = psi0
        self._edip = edip
        self.dip = self.edip
        # if edip is not None:
        #     self.edip_x = edip[:,:, 0]
        #     self.edip_y = edip[:, :, 1]
        #     self.edip_z = edip[:, :, 2]
        #     self.edip_rms = np.sqrt(np.abs(edip[:, :, 0])**2 + \
        #                             np.abs(edip[:, :, 1])**2 + \
        #                                 np.abs(edip[:,:,2])**2)
        # else:
        #     self.edip_rms = edip_rms

        self._edip_rms = edip_rms

        self.nstates = H.shape[0]
        self.dim = H.shape[0]
        self.size = H.shape[0]

        # self.raising = np.tril(edip)
        # self.lowering = np.triu(edip)

        self.idm = identity(self.dim)

        # if tau is not None:
        #     self.lifetime = tau
        #     self.decay = 1./tau
        self.gamma = gamma
        self.mdip = None
        self.dephasing = 0.

        if isdiag(H):
            self.E = np.diag(H)
        else:
            self.E = self.eigenenergies()

    # def raising(self):
    #     return self.raising

    # def lowering(self):
    #     return self.lowering
    @property
    def edip(self):
        return self._edip

    @edip.setter
    def edip(self, edip):
        self._edip = edip


    def set_dipole(self, dip):
        self.dip = dip
        return

    def set_edip(self, edip, pol=None):
        self.edip_rms = edip
        return

    def set_mdip(self, mdip):
        self.mdip = mdip
        return

    @property
    def edip_rms(self):
        if self._edip_rms is None:
            self._edip_rms = np.sqrt(np.abs(self.edip[:, :, 0])**2 + \
                                    np.abs(self.edip[:, :, 1])**2 + \
                                    np.abs(self.edip[:,:,2])**2)
        return self._edip_rms

    @edip_rms.setter
    def edip_rms(self, edip):
        self._edip_rms = edip


    # @property
    # def H(self):
    #     return self.H

    # @H.setter
    # def H(self, H):
    #     '''
    #     Set model Hamiltonian

    #     Parameters
    #     ----------
    #     H : 2d array
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     '''
    #     self.H = H
    #     return

    def groundstate(self, method='trivial'):

        if method == 'trivial':
            return basis(self.dim, 0).tocsr()

        elif method == "diag":

            eg, evecs = self.eigenstates(k=1)[:]
            print('ground state energy = ', eg)
            return evecs

    def energy(self, psi):
        return obs(self.H, psi)


    def set_decay_for_all(self, gamma):
        """
        Set the decay rate for all excited states.

        Parameters
        ----------
        gamma : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.gamma = [gamma] * self.nstates
        self.gamma[0] = 0  # ground state decay is 0
        return

    def set_dephasing(self, gamma):
        """
        set the pure dephasing rate for all coherences

        Parameters
        ----------
        gamma : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.dephasing = gamma

    def set_decay(self, gamma):
        """
        Set the decay rate for all excited states.

        Parameters
        ----------
        gamma : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.gamma = gamma
        return


    # def get_ham(self):
    #     return self.H


    # def getH(self):
    #     return self.H

    def get_nonhermitianH(self):
        H = self.H

        if self.gamma is not None:

            self.nonhermH = H - 1j * np.diagflat(self.gamma)

        else:
            raise ValueError('Please set gamma first.')

        return self.nonhermH

    def get_nonhermH(self):
        H = self.H

        if self.gamma is not None:

            self.nonhermH = H - 1j * np.diagflat(self.gamma)

        else:
            raise ValueError('Please set gamma first.')

        return self.nonhermH

    def get_dip(self):
        return self.dip

    def get_edip(self):
        return self.edip

    def get_dm(self):
        return self.dm

    def set_lifetime(self, tau):
        self.lifetime = tau

    def eigenenergies(self):
        return np.linalg.eigvals(self.H)

    def eigvals(self):
        if isdiag(self.H):
            return np.diagonal(self.H)
        else:
            return np.linalg.eigvals(self.H)

    def eigenstates(self, k=6):
        """

        Parameters
        ----------
        k: integer
            number of eigenstates to compute, < dim

        Returns
        -------
        eigvals: vector
        eigvecs: 2d array
        """
        if self.H is None:
            raise ValueError('Call getH/calcH to compute H first.')

        if k < self.dim:
            eigvals, eigvecs = linalg.eigs(self.H, k=k, which='SR')

            idx = eigvals.argsort()[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            return eigvals, eigvecs

        if k == self.dim:
            return np.linalg.eigh(self.H.toarray())

    def driven_dynamics(self, pulse, dt=0.001, Nt=1, obs_ops=None, nout=1, t0=0.0):
        '''
        wavepacket dynamics in the presence of laser pulses

        Parameters
        ----------
        pulse : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        obs_ops : TYPE, optional
            DESCRIPTION. The default is None.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0: float
            initial time

        Returns
        -------
        None.

        '''
        H = self.H
        dip = self.dip
        psi0 = self.initial_state

        if psi0 is None:
            sys.exit("Error: Initial wavefunction not specified!")

        driven_dynamics(H, dip, psi0, pulse, dt=dt, Nt=Nt, \
                        e_ops=obs_ops, nout=nout, t0=t0)

        return

    def quantum_dynamics(self, psi0, dt=0.001, Nt=1, obs_ops=None, nout=1, t0=0.0):
        '''
        quantum dynamics under time-independent hamiltonian

        Parameters
        ----------
        pulse : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        obs_ops : TYPE, optional
            DESCRIPTION. The default is None.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0: float
            initial time

        Returns
        -------
        None.

        '''
        ham = self.H

        quantum_dynamics(ham, psi0, dt=dt, Nt=Nt, \
                         obs_ops=obs_ops, nout=nout, t0=t0)

        return

    def evolve(self, psi0=None, dt=0.001, Nt=10, e_ops=None, nout=1, t0=0.0, pulse=None):
        '''
        quantum dynamics under time-independent Hamiltonian

        Parameters
        ----------
        pulse : TYPE
            DESCRIPTION.
        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        obs_ops : TYPE, optional
            DESCRIPTION. The default is None.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0: float
            initial time

        Returns
        -------
        None.

        '''
        # if psi0 is None:
        #     raise ValueError("Please specify initial wavefunction psi0.")

        H = self.H

        if e_ops is None:
            e_ops = []

        # if the electric dipole is not given, use the RMS of edip
        if self.edip is None:
            if self.edip_rms is None:
                raise ValueError('Please provide electric dipole, either edip or edip_rms.')
            else:
                edip = self.edip_rms
        else:
            edip = self.edip

        return SESolver(H).run(psi0=psi0, dt=dt, Nt=Nt,\
                       e_ops=e_ops, nout=nout, t0=t0, edip=edip, pulse=pulse)
        # if pulse is None:
        #     return quantum_dynamics(H, psi0, dt=dt, Nt=Nt, \
        #                             obs_ops=e_ops, nout=nout, t0=t0)
        # else:

        #     edip = self.edip
        #     return driven_dynamics(H, edip, psi0, pulse, dt=dt, Nt=Nt, \
        #                            e_ops=e_ops, nout=nout, t0=t0)


    # def run(self, psi0=None, dt=0.01, e_ops=None, Nt=1, nout=1, t0=0.0, \
    #         edip=None, pulse=None):
    #     '''
    #     quantum dynamics under time-independent Hamiltonian

    #     Parameters
    #     ----------
    #     pulse : TYPE
    #         DESCRIPTION.
    #     dt : float, optional
    #         time interval. The default is 0.001.
    #     Nt : int, optional
    #         int. The default is 1.
    #     obs_ops : TYPE, optional
    #         DESCRIPTION. The default is None.
    #     nout : TYPE, optional
    #         DESCRIPTION. The default is 1.
    #     t0: float
    #         initial time

    #     Returns
    #     -------
    #     None.

    #     '''
    #     if psi0 is None:
    #         raise ValueError("Please specify initial wavefunction psi0.")

    #     H = self.H

    #     if pulse is None:
    #         return quantum_dynamics(H, psi0, dt=dt, Nt=Nt, \
    #                                 obs_ops=e_ops, nout=nout, t0=t0)
    #     else:

    #         if isinstance(pulse, list):
    #             H = [self.H]
    #             for i in range(len(pulse)):
    #                 H.append([edip[i], pulse[i].efield])
    #         else:
    #             H = [self.H, [edip, pulse.efield]]

    #         return driven_dynamics(H, psi0, dt=dt, Nt=Nt, \
    #                                e_ops=e_ops, nout=nout, t0=t0)

    # def heom(self, env, nado=5, c_ops=None, obs_ops=None, fname=None):
    #     nt = self.nt
    #     dt = self.dt
    #     return _heom(self.oqs, env, c_ops, nado, nt, dt, fname)

    # def redfield(self, env, dt, Nt, c_ops, obs_ops, rho0, integrator='SOD'):
    #     nstates = self.nstates

    #     h0 = self.H

    #     _redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env, integrator='SOD')
    #     return

    # def tcl2(self, env, c_ops, obs_ops, rho0, dt, Nt, integrator='SOD'):

    #     nstates = self.nstates
    #     h0 = self.H

    #     redfield(nstates, rho0, c_ops, h0, Nt, dt,obs_ops, env, integrator='SOD')

    #     return

    # def lindblad(self, c_ops, obs_ops, rho0, dt, Nt):
    #     """
    #     lindblad quantum master equations

    #     Parameters
    #     ----------
    #     rho0: 2D array
    #         initial density matrix
    #     """

    #     h0 = self.H
    #     lindblad(rho0, h0, c_ops, Nt, dt, obs_ops)
    #     return

    # def steady_state(self):
    #     return

    # def correlation_2p_1t(self, rho0, ops, dt, Nt, method='lindblad', output='cor.dat'):
    #     '''
    #     two-point correlation function <A(t)B>

    #     Parameters
    #     ----------
    #     rho0 : TYPE
    #         DESCRIPTION.
    #     ops : TYPE
    #         DESCRIPTION.
    #     dt : TYPE
    #         DESCRIPTION.
    #     Nt : TYPE
    #         DESCRIPTION.
    #     method : TYPE, optional
    #         DESCRIPTION. The default is 'lindblad'.
    #     output : TYPE, optional
    #         DESCRIPTION. The default is 'cor.dat'.

    #     Returns
    #     -------
    #     None.

    #     '''

    #     H = self.hamiltonian
    #     c_ops = self.c_ops

    #     correlation_2p_1t(H, rho0, ops=ops, c_ops=c_ops, dt=dt, Nt=Nt, \
    #                       method=method, output=output)

    #     return

    def tcl2(self):
        """
        second-order time-convolutionless quantum master equation
        """
        pass

    def absorption(self, omegas, method='sos', **kwargs):
        '''
        Linear absorption of the model.

        Parameters
        ----------
        omegas : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'SOS'.
        normalize : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if method == 'sos':
            # eigvals = self.eigvals()
            # dip = self.dip
            # gamma = self.gamma

            return sos.absorption(self, omegas, **kwargs)

        elif method == 'superoperator':

            raise NotImplementedError('The method {} has not been implemented. \
                              Try "sos"'.format(method))

    def PE(self, pump, probe, t2=0.0, **kwargs):
        '''
        alias for photon_echo

        '''
        return self.photon_echo(pump, probe, t2=t2, **kwargs)

    def photon_echo(self, pump, probe, t2=0.0, **kwargs):
        '''
        2D photon echo signal at -k1+k2+k3

        Parameters
        ----------
        pump : TYPE
            DESCRIPTION.
        probe : TYPE
            DESCRIPTION.
        t2 : TYPE, optional
            DESCRIPTION. The default is 0.0.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # E = self.eigvals()
        # edip = self.edip

        return sos.photon_echo(self, pump=pump, probe=probe, \
                                t2=t2, **kwargs)

    def PE2(self, omega1, omega2, t3=0.0, **kwargs):
        '''
        2D photon echo signal at -k1+k2+k3
        Transforming t1 and t2 to frequency domain.

        Parameters
        ----------
        pump : TYPE
            DESCRIPTION.
        probe : TYPE
            DESCRIPTION.
        t2 : TYPE, optional
            DESCRIPTION. The default is 0.0.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''

        return sos.photon_echo_t3(self, omega1=omega1, omega2=omega2, \
                                t3=t3, **kwargs)

    def DQC(self):
        pass

    def TPA(self):
        pass

    def ETPA(self):
        pass

    def cars(self, shift, omega1, t2=0., plt_signal=False, fname=None):

        edip = self.edip_rms
        E = self.eigvals()

        S = sos.cars(E, edip, shift, omega1, t2=t2)

        if not plt_signal:

            return S

        else:

            fig, ax = plt.subplots()

            ax.contourf(shift*au2wavenumber, omega1*au2ev, S.imag.T, lw=0.6,\
                        cmap='spectral')

            ax.format(xlabel='Raman shift (cm$^{-1}$)', ylabel=r'$\Omega_1$ (eV)')

            fig.savefig(fname+'.pdf')

            return S, fig, ax

@dataclass
class Mode:
    omega: float
    couplings: list
    truncate: int = 2


class LVC(Mol):
    """
    linear vibronic coupling model in Fock space
    """
    def __init__(self, E, modes):
        """

        Parameters
        ----------
        E : 1d array
            electronic energy at ground-state minimum
        modes : list of Mode objs
            vibrational modes

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if len(E) != 3:
            raise NotImplementedError
        self.e_fc = E
        self.nel = self.nstates = len(E)
        self.nmodes = len(modes)
        self.modes = modes
        self.truncate = None
        self.fock_dims = [m.truncate for m in modes]
        self.nvib = np.prod(self.fock_dims)

        self.idm_vib = identity(self.nvib) # vibrational identity
        self.idm_el = identity(self.nstates) # electronic identity
        self.omegas = [mode.omega for mode in self.modes]


        self.H = None
        self.dim = None

    def buildH(self):
        """
        Calculate the vibronic Hamiltonian.

        Parameters
        ----------
        nums : list of integers
            size for the Fock space of each mode

        Returns
        -------
        2d array
            Hamiltonian

        """

        omegas = self.omegas
        nmodes = self.nmodes

        # identity matrices in each subspace
        nel = self.nstates
        I_el = identity(nel)

        h_el = np.diagflat(self.e_fc)

        # calculate the vibrational Hamiltonian
        hv, xs = multimode(omegas, nmodes, truncate=self.fock_dims[0])

        # bare vibronic H in real e-states
        H = kron(h_el, identity(hv.shape[0])) + kron(I_el, hv)



        # vibronic coupling, tuning + coupling
        for j, mode in enumerate(self.modes):
            # n = mode.truncate

            # # vibrational Hamiltonian
            # hv = boson(mode.omega, n, ZPE=False)

            # H = kron(H, Iv) + kron(identity(H.shape), hv)
            V = 0.
            for c in mode.couplings:
                a, b = c[0]
                strength = c[1]
                V += strength * jump(a, b, nel)

            H += kron(V, xs[j])

        self.H = H
        self.dim = H.shape[0]



        return self.H

    def APES(self, x):

        V = np.diag(self.e_fc)

        # for n in range(self.nmodes):
        V += 0.5 * np.sum(self.omegas * x**2) * self.idm_el

        # V += tmp * self.idm_el

        for j, mode in enumerate(self.modes):
            for c in mode.couplings:
                a, b = c[0]
                strength = c[1]
                V += strength * jump(a, b, self.nstates) * x[j]

        E = np.linalg.eigvals(V)
        return np.sort(E)

    def calc_edip(self):
        pass

    def promote(self, A, which='el'):

        if which in ['el', 'e', 'electronic']:
            A = kron(A, self.idm_vib)
        elif which in ['v', 'vib', 'vibrational']:
            A = kron(self.idm_el, A)

        return A

    def vertical(self, n=1):
        """
        generate the initial state created by vertical excitation

        Parameters
        ----------
        n : int, optional
            initially excited state. The default is 1.

        Returns
        -------
        psi : TYPE
            DESCRIPTION.

        """
        psi = basis(self.nstates, n)

        dims = self.fock_dims

        chi = basis(dims[0], 0)

        for j in range(1, self.nmodes):
            chi = np.kron(chi, basis(dims[j], 0))

        psi = np.kron(psi, chi)

        return psi

    def groundstate(self):
        """
        return the ground state

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.vertical(n=0)

    def buildop(self, i, f=None, isherm=True):
        """
        construct electronic operator

            \ket{f}\bra{i}

        if isherm:
            return \ket{f}\bra{i} + \ket{i}\bra{f}

        Parameters
        -------
        i: int
            initial state.
        f: int, optional
            final state. Default None. If None, set f = i.

        isherm: bool, optional
            indicator of whether the returned matrix is Hermitian or not
            Default: True

        Returns
        -------
        2darray
            DESCRIPTION.

        """
        if f is None:
            f = i

        p = jump(i=i, f=f, dim=self.nstates, isherm=isherm)

        return kron(p, self.idm_vib)

    def dpes(self, q):
        pass


    def wavepacket_dynamics(self, method='RK4'):


        if method == 'RK4':

            sol = SESolver()

            if self.H is None:
                self.buildH()

            sol.H = self.H

            sol.groundstate = self.groundstate()

            return sol

        # elif method == 'SPO':

        #     if self.nmodes == 1:
        #         from pyqed.wpd import SPO

        #         sol = SPO()

        #     elif self.nmodes == 2:
        #         from pyqed.wpd import SPO2

        #         sol = SPO2()

        else:
            raise ValueError('The number of modes {} is not \
                             supported.'.format(self.nmodes) )




            return sol

    def rdm_el(self, psi):
        """
        Compute the electronic reduced density matrix.

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        psi_reshaped = np.reshape(psi, (self.nel, self.nvib))

        return psi_reshaped.dot(dag(psi_reshaped))

    def add_coupling(self, coupling):
        """
        add additional coupling terms to the Hamiltonian such as Stark
        and Zeeman effects

        Parameters
        ----------
        coupling : list, [[a, b], strength]
            describe the coupling, a, b labels the electronic states

        Returns
        -------
        ndarray
            updated H.

        """
        a, b = coupling[0]
        strength = coupling[1]

        self.H += strength * kron(jump(a, b, self.nel),  self.idm_vib)

        return self.H

    def plot_PES(self, x, y):
        """
        plot the 3D adiabatic potential energy surfaces

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        from pyqed.style import plot_surfaces

        if self.nmodes != 2:
            raise ValueError('This function only works for nmodes=2.')

        nx, ny = len(x), len(y)
        E = np.zeros(nx, ny, self.nstates)

        for i in range(nx):
            for j in range(ny):
                E[i, j, :] = self.APES([x[i], y[j]])

        return plot_surfaces(x, y, [E[:,:, 1], E[:,:, 2]])


def polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

class JahnTeller(LVC):
    """
    E \otimes e Jahn-Teller model with two degenerate modes and two degenerate electronic states
    (+ the ground state)
    """
    def __init__(self, E, omega, kappa, truncate=24):
        """


        Parameters
        ----------
        E : TYPE
            DESCRIPTION.
        omega : TYPE
            DESCRIPTION.
        kappa : TYPE
            DESCRIPTION.
        truncate : TYPE, optional
            DESCRIPTION. The default is 24.

        Returns
        -------
        None.

        """
        self.omega = omega
        self.kappa = kappa

        tuning_mode = Mode(omega, couplings=[[[1, 1], kappa], \
                                             [[2, 2], -kappa]], truncate=24)
        coupling_mode = Mode(omega, [[[1, 2], kappa]], truncate=24)

        modes = [tuning_mode, coupling_mode]
        super().__init__(E, modes)


    def APES(self, x, y, B=None):

        V = np.diag(self.e_fc).astype(complex)

        rho, theta = polar(x, y)
        V += 0.5 * self.omega * rho**2 * self.idm_el

        # coupling
        C = self.kappa * rho * np.exp(-1j * theta) * jump(1, 2, dim=3, isherm=False)
        V += C + dag(C)

        # for j, mode in enumerate(self.modes):
        #     for c in mode.couplings:
        #         a, b = c[0]
        #         strength = c[1]
        #         V += strength * jump(a, b, self.nstates) * xvec[j]

        # # reverse transformation from |+/-> to |x/y>
        # R = np.zeros((self.nstates, self.nstates))
        # R[0, 0] = 1.
        # R[1:, 1:] = 1./np.sqrt(2) * np.array([[1, 1], [-1j, 1j]])

        if B is not None:
            V += B * np.diag([0, -0.5, 0.5])

        E = np.linalg.eigvals(V)
        return np.sort(E)

    def buildH(self, B=None):
        H = super().buildH()

        # rotate the electronic states into ring-current carrying eigenstates of Lz ???
        # |+/-> = (|x> +/- |y>)/sqrt(2)
        R = np.zeros((self.nstates, self.nstates), dtype=complex)
        R[0, 0] = 1.
        R[1:, 1:] = 1./np.sqrt(2) * np.array([[1, 1j], [1, -1j]])

        R = kron(R, self.idm_vib)

        self.H = transform(H, R)

        if B is not None:
            H += B * kron(np.diag([0, -0.5, 0.5]), self.idm_vib)

        self.H = H
        return H

# class Vibronic:
#     '''
#     2-dimensional grid based vibronic model
#     '''
#     def __init__(self, nelec, xlim, ylim):
#         self.xlim = xlim
#         self.ylim = ylim






class SESolver:
    def __init__(self, H=None):
        """
        Basic class for time-dependent Schrodinger equation.

        Parameters
        ----------
        H : array
            Hamiltonian.

        Returns
        -------
        None.

        """
        self.H = H
        self.groundstate = None
        # self._isherm = isherm




    def run(self, psi0=None, dt=0.01, Nt=1,\
               e_ops=None, nout=1, t0=0.0, edip=None, pulse=None, use_sparse=True):
        '''
        quantum dynamics under time-independent and time-dependent Hamiltonian

        Parameters
        ----------
        psi0: 1d array
            initial state
        pulse : callable/list of callable
            external laser pulses

        dt : TYPE, optional
            DESCRIPTION. The default is 0.001.
        Nt : TYPE, optional
            DESCRIPTION. The default is 1.
        obs_ops : TYPE, optional
            DESCRIPTION. The default is None.
        nout : TYPE, optional
            DESCRIPTION. The default is 1.
        t0: float
            initial time. The default is 0.

        Returns
        -------
        None.

        '''
        if psi0 is None:
            print("Initial state not specified. Using the ground state.")
            psi0 = self.groundstate

        H0 = self.H

        if pulse is None:

            return _quantum_dynamics(H0, psi0, dt=dt, Nt=Nt,
                                     e_ops=e_ops, nout=nout, t0=t0)
        else:
            if edip is None:
                raise ValueError('Electric dipole must be provided for \
                                 laser-driven dynamics.')

            if isinstance(pulse, list): # multi pulses

                H = [self.H]

                for i in range(len(pulse)):
                    H.append([edip[i], pulse[i].efield])

                return driven_dynamics(H=H, psi0=psi0, dt=dt, Nt=Nt, \
                                   e_ops=e_ops, nout=nout, t0=t0)

            else: # single pulse

                if edip.ndim == 2: # dipole projected on the laser polarization

                    H = [self.H]
                    H.append([edip, pulse.efield])

                    return driven_dynamics(H=H, psi0=psi0, dt=dt, Nt=Nt, \
                                   e_ops=e_ops, nout=nout, t0=t0, \
                                       use_sparse=use_sparse)

                elif edip.ndim == 3: # full dipole
                    return _driven_dynamics(H=H0, psi0=psi0, edip=edip,\
                                            E=pulse.E, dt=dt, Nt=Nt, \
                                   e_ops=e_ops, nout=nout, t0=t0)


    # def run(self, psi0=None, dt=0.001, Nt=1, e_ops=[], nout=1, t0=0.0,\
    #         edip=None, pulse=None):

    #     return self.evolve(psi0=psi0, dt=dt, Nt=Nt, e_ops=e_ops, nout=nout,\
    #                        t0=t0, edip=edip, pulse=pulse)

    def propagator(self, dt, Nt):
        H = self.H
        return _propagator(H, dt, Nt)

    def correlation_2op_1t(self):
        pass

    def correlation_3op_1t(self, psi0, oplist, dt, Nt):
        """
        <AB(t)C>

        Parameters
        ----------
        psi0
        oplist
        dt
        Nt

        Returns
        -------

        """
        H = self.H
        a_op, b_op, c_op = oplist

        corr_vec = np.zeros(Nt, dtype=complex)

        psi_ket = _quantum_dynamics(H, c_op @ psi0, dt=dt, Nt=Nt).psilist
        psi_bra = _quantum_dynamics(H, dag(a_op) @ psi0, dt=dt, Nt=Nt).psilist

        for j in range(Nt):
            corr_vec[j] = np.vdot(psi_bra[j], b_op @ psi_ket[j])

        return corr_vec

    def correlation_3op_2t(self, psi0, oplist, dt, Nt, Ntau):
        """
        <A(t)B(t+tau)C(t)>
        Parameters
        ----------
        oplist: list of arrays
            [a, b, c]
        psi0: array
            initial state
        dt
        nt: integer
            number of time steps for t
        ntau: integer
            time steps for tau

        Returns
        -------

        """
        H = self.H
        psi_t = _quantum_dynamics(H, psi0, dt=dt, Nt=Nt).psilist

        a_op, b_op, c_op = oplist

        corr_mat = np.zeros([Nt, Ntau], dtype=complex)

        for t_idx, psi in enumerate(psi_t):
            psi_tau_ket = _quantum_dynamics(H, c_op @ psi, dt=dt, Nt=Ntau).psilist
            psi_tau_bra = _quantum_dynamics(H, dag(a_op) @ psi, dt=dt, Nt=Ntau).psilist

            corr_mat[t_idx, :] = [np.vdot(psi_tau_bra[j], b_op @ psi_tau_ket[j])
                                  for j in range(Ntau)]

        return corr_mat

    def correlation_4op_1t(self, psi0, oplist, dt=0.005, Nt=1):
        """
        <AB(t)C(t)D>

        Parameters
        ----------
        psi0
        oplist
        dt
        Nt

        Returns
        -------

        """
        a_op, b_op, c_op, d_op = oplist
        return self.correlation_3op_1t(psi0, [a_op, b_op @ c_op, d_op], dt, Nt)

    def correlation_4op_2t(self, psi0, oplist, dt=0.005, Nt=1, Ntau=1):
        """

        Parameters
        ----------
        psi0 : vector
            initial state
        oplist : list of arrays
        """
        a_op, b_op, c_op, d_op = oplist
        return self.correlation_3op_2t(psi0, [a_op, b_op @ c_op, d_op], dt, Nt, Ntau)


def _propagator(H, dt, Nt):
    """
    compute the resolvent for the multi-point correlation function signals
    U(t) = e^{-i H t}

    Parameters
    -----------
    t: float or list
        times
    """

    # propagator
    U = identity(H.shape[-1], dtype=complex)

    # set the ground state energy to 0
    print('Computing the propagator. '
          'Please make sure that the ground-state energy is 0.')
    Ulist = []
    for k in range(Nt):
        Ulist.append(U.copy())
        U = rk4(U, tdse, dt, H)

    return Ulist


def _quantum_dynamics(H, psi0, dt=0.001, Nt=1, e_ops=[], t0=0.0,
                      nout=1, store_states=True, output='obs.dat'):
    """
    Quantum dynamics for a multilevel system.

    Parameters
    ----------
    e_ops: list of arrays
        expectation values to compute.
    H : 2d array
        Hamiltonian of the molecule
    psi0: 1d array
        initial wavefunction
    dt : float
        time step.
    Nt : int
        timesteps.
    e_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    """

    psi = psi0.copy()

    if e_ops is not None:
        fmt = '{} ' * (len(e_ops) + 1) + '\n'

    # f_obs = open(output, 'w')  # observables
    # f_wf = open('psi.dat', 'w')

    t = t0

    # f_obs.close()

    if store_states:

        result = Result(dt=dt, Nt=Nt, psi0=psi0, t0=t0, nout=nout)

        observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)
        # psilist = [psi0.copy()]

        # compute observables for t0

        # observables[0, :] = [obs(psi, e_op).toarray().item() for e_op in e_ops]
        observables[0, :] = [obs(psi, e_op) for e_op in e_ops]

        for k1 in range(1, Nt // nout):

            for k2 in range(nout):
                psi = rk4(psi, tdse, dt, H)

            t += dt * nout

            # compute observables
            observables[k1, :] = [obs(psi, e_op) for e_op in e_ops]

            # f_obs.write(fmt.format(t, *e_list))

            result.psilist.append(psi.copy())

        # result.psilist = psilist
        result.observables = observables

        return result

    else:  # not save states

        f_obs = open(output, 'w')  # observables

        for k1 in range(int(Nt / nout)):
            for k2 in range(nout):
                psi = rk4(psi, tdse, dt, H)

            t += dt * nout

            # compute observables
            e_list = [obs(psi[:,0], e_op) for e_op in e_ops]
            f_obs.write(fmt.format(t, *e_list))

        f_obs.close()

        return psi


def _ode_solver(H, psi0, dt=0.001, Nt=1, e_ops=[], t0=0.0,
                nout=1, store_states=True, output='obs.dat'):
    """
    Integrate the TDSE for a multilevel system using Scipy.

    Parameters
    ----------
    e_ops: list of arrays
        expectation values to compute.
    H : 2d array
        Hamiltonian of the molecule
    psi0: 1d array
        initial wavefunction
    dt : float
        time step.
    Nt : int
        timesteps.
    e_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    """

    psi = psi0.copy()
    t_eval = np.arange(Nt // nout) * dt * nout

    def fun(t, psi): return -1j * H.dot(psi)

    tf = t0 + Nt * dt
    t_span = (t0, tf)
    sol = scipy.integrate.solve_ivp(fun, t_span=t_span, y0=psi,
                                    t_eval=t_eval)

    result = Result(dt=dt, Nt=Nt, psi0=psi0)
    result.times = sol.t
    print(sol.nfev)

    observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)
    for i in range(len(t_eval)):
        psi = sol.y[:, i]
        observables[i, :] = [obs(psi, e_op) for e_op in e_ops]

    result.observables = observables
    # if store_states:
    #
    #     result = Result(dt=dt, Nt=Nt, psi0=psi0)
    #
    #     observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)
    #     psilist = [psi0.copy()]
    #
    #     # compute observables for t0
    #     observables[0, :] = [obs(psi, e_op) for e_op in e_ops]
    #
    #     for k1 in range(1, Nt // nout):
    #
    #         for k2 in range(nout):
    #             psi = rk4(psi, tdse, dt, H)
    #
    #         t += dt * nout
    #
    #         # compute observables
    #         observables[k1, :] = [obs(psi, e_op) for e_op in e_ops]
    #         # f_obs.write(fmt.format(t, *e_list))
    #
    #         psilist.append(psi.copy())
    #
    #     # f_obs.close()
    #
    #     result.psilist = psilist
    #     result.observables = observables

    return result

def _driven_dynamics(H, psi0, edip, E, dt=0.001, Nt=1, e_ops=None, nout=1, \
                    t0=0.0, sparse=True):
    '''
    Laser-driven dynamics in the presence of polarized laser pulses

    Parameters
    ----------
    ham : 2d array
        Hamiltonian of the molecule
    dip : TYPE
        transition dipole moment
    psi0: 1d array
        initial wavefunction
    pulse : TYPE
        laser pulse
    dt : TYPE
        time step.
    Nt : TYPE
        timesteps.
    e_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps
    sparse: bool
        indicater whether to use sparse matrices or not (is there a gain?)

    Returns
    -------
    None.

    '''


    if sparse:
        psi = csr_matrix(psi0.astype(complex)).T.tocsr()

    nstates = H.shape[0]

    def calcH(t):

        # for i in range(1, len(H)):

        ht = H - np.einsum('ija, a -> ij', edip, E(t))
        return ht

    if e_ops is None:
        e_ops = []

    t = t0

    # f_dip = open('dipole'+'{:f}'.format(pulse.delay * au2fs)+'.dat', 'w')

    result = Result(dt=dt, Nt=Nt, psi0=psi0, t0=t0, nout=nout)

    observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)

    # psilist = [psi0.copy()]
    psit = np.zeros((nstates, Nt//nout), dtype=complex)

    # compute observables at t0
    observables[0, :] = [obs(psi, e_op).toarray().item() for e_op in e_ops]

    psit[:,0] = psi.toarray()[:, 0]

    for k1 in range(1, Nt // nout):

        for k2 in range(nout):

            # ht = -pulse.field(t) * edip + H
            ht = csr_matrix(calcH(t))
            psi = rk4(psi, tdse, dt, ht)

        t += dt * nout

        # compute observables
        observables[k1, :] = [obs(psi, e_op).toarray().item() for e_op in e_ops]
        # f_obs.write(fmt.format(t, *e_list))

        result.psilist.append(psi.copy())
        psit[:, k1] = psi.toarray()[:, 0]


    # result.psilist = psilist
    result.psi = psit
    result.observables = observables

    return result



def driven_dynamics(H, psi0, dt=0.01, Nt=1, e_ops=None, nout=1, \
                    t0=0.0, return_result=True, use_sparse=True):
    '''
    Laser-driven dynamics in the presence of laser pulses

    Parameters
    ----------
    ham : 2d array
        Hamiltonian of the molecule
    dip : TYPE
        transition dipole moment
    psi0: 1d array
        initial wavefunction
    pulse : TYPE
        laser pulse
    dt : TYPE
        time step.
    Nt : TYPE
        timesteps.
    e_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps
    sparse: bool
        indicater whether to use sparse matrices or not (is there a gain?)

    Returns
    -------
    None.

    '''


    if use_sparse:
        # note that when using sparse matrix, the psi is of shape [nstates, 1]
        psi = csr_matrix(psi0.astype(complex)).T.tocsr()
    else:
        psi = psi0

    nstates = H[0].shape[0]

    def calcH(t):

        Ht = H[0].copy().astype(complex)

        for i in range(1, len(H)):
            Ht += - H[i][1](t) * H[i][0]

        return csr_matrix(Ht)

    if e_ops is None:
        e_ops = []

    t = t0

    # f_dip = open('dipole'+'{:f}'.format(pulse.delay * au2fs)+'.dat', 'w')
    if return_result:

        result = Result(dt=dt, Nt=Nt, psi0=psi0, t0=t0, nout=nout)

        observables = np.zeros((Nt // nout, len(e_ops)), dtype=complex)

        psilist = [psi0.copy()]
        psit = np.zeros((nstates, Nt//nout), dtype=complex)

        # compute observables at t0
        observables[0, :] = [obs(psi, e_op).toarray().item() for e_op in e_ops]

        psit[:,0] = psi.toarray()[:, 0]
        # psit[:, 0] = psi

        for k1 in range(1, Nt // nout):

            for k2 in range(nout):

                # ht = -pulse.field(t) * edip + H
                ht = calcH(t)

                psi = rk4(psi, tdse, dt, ht)

            t += dt * nout

            # compute observables
            observables[k1, :] = [obs(psi, e_op).toarray().item() for e_op in e_ops]
            # f_obs.write(fmt.format(t, *e_list))

            psilist.append(psi.copy())
            psit[:, k1] = psi.toarray()[:, 0]
            # psit[:, k1] = psi


        result.psilist = psilist
        result.psi = psit
        result.observables = observables

        return result

    else:

        fmt = '{} ' * (len(e_ops) + 1) + '\n'
        fmt_dm = '{} ' * (nstates + 1) + '\n'

        f_dm = open('psi.dat', 'w')  # wavefunction
        f_obs = open('obs.dat', 'w')  # observables

        for k1 in range(int(Nt / nout)):

            for k2 in range(nout):
                ht = calcH(t)
                psi = rk4(psi, tdse, dt, ht)

            t += dt * nout

            # compute observables
            Aave = np.zeros(len(e_ops), dtype=complex)
            for j, A in enumerate(e_ops):
                Aave[j] = obs(psi, A)

            f_dm.write(fmt_dm.format(t, *psi))
            f_obs.write(fmt.format(t, *Aave))

        f_dm.close()
        f_obs.close()

        return

def mls(dim=3):

    E = np.array([0., 0.6, 10.0])/au2ev
    N = len(E)

    gamma = np.array([0, 0.002, 0.002])/au2ev
    H = np.diag(E)

    dip = np.zeros((N, N, 3), dtype=complex)
    dip[1,2, :] = [1.+0.5j, 1.+0.1j, 0]

    dip[2,1, :] = np.conj(dip[1, 2, :])
    # dip[1,3, :] = dip[3,1] = 1.

    dip[0, 1, :] = [1.+0.2j, 0.5-0.1j, 0]
    dip[1, 0, :] = np.conj(dip[0, 1, :])

    # dip[3, 3] = 1.
    dip[0, 2, :] = dip[2, 0, :] = [0.5, 1, 0]

    mol = Mol(H, dip)
    mol.set_decay(gamma)

    return mol

def test_sesolver():
    s0, sx, sy, sz = pauli()
    H = (-0.05) * sz - 0. * sx

    solver = SESolver(H)
    Nt = 1000
    dt = 4
    psi0 = (basis(2, 0) + basis(2, 1)) / np.sqrt(2)
    start_time = time.time()

    result = solver.evolve(psi0, dt=dt, Nt=Nt, e_ops=[sx])
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    times = np.arange(Nt) * dt
    r = _ode_solver(H, psi0, dt=dt, Nt=Nt, nout=2, e_ops=[sx])

    print("--- %s seconds ---" % (time.time() - start_time))

    # test correlation functions
    # corr = solver.correlation_3p_1t(psi0=psi0, oplist=[s0, sx, sx],
    #                                 dt=0.05, Nt=Nt)
    #
    # times = np.arange(Nt)
    #
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(times, result.observables[:, 0])
    ax.plot(r.times, r.observables[:, 0], '--')
    # ax.plot(times, corr.real)
    plt.show()

def high_frequency_drive():

    from pyqed.phys import coh_op
    from pyqed.optics import Pulse

    mol = mls()
    N = mol.dim

    solver = SESolver(mol.H)
    Nt = 400
    dt = 4
    psi0 = basis(N, 0)
    start_time = time.time()
    pulse = Pulse(omegac=1/au2ev)

    # r = mol.evolve(psi0=psi0, dt=dt, Nt=Nt, e_ops=[coh_op(1,1, N)],\
    #                     pulse=pulse)

    r = solver.run(psi0=psi0, dt=dt, Nt=Nt, e_ops=[coh_op(0,0, N), coh_op(1,1, N), coh_op(2,2,N)],\
                          pulse=pulse,edip=mol.edip[:,:,0].real)

    print("--- %s seconds ---" % (time.time() - start_time))

    # test correlation functions
    # corr = solver.correlation_3p_1t(psi0=psi0, oplist=[s0, sx, sx],
    #                                 dt=0.05, Nt=Nt)
    #
    # times = np.arange(Nt)
    #
    import proplot as plt

    fig, ax = plt.subplots()
    # ax.plot(times, result.observables[:, 0])
    for j in range(3):
        ax.plot(r.times, r.observables[:, j], '-', label=str(j))
    # ax.format(ylim=(0, 1e-3))
    # ax.plot(times, corr.real)
    plt.show()

if __name__ == '__main__':
    from pyqed.phys import pauli
    import time
    import proplot as plt

    mol = mls()
    mol.set_decay([0, 0.002, 0.002])
    omegas=np.linspace(0, 2, 200)/au2ev
    shift = np.linspace(0, 1)/au2ev


    high_frequency_drive()
    # mol.absorption(omegas)
    # mol.photon_echo(pump=omegas, probe=omegas, plt_signal=True)
    # S = mol.cars(shift=shift, omega1=omegas, plt_signal=True)

