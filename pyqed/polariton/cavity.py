#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:26:02 2019

@author: binggu
"""

import numpy as np
import scipy
from scipy.sparse import lil_matrix, csr_matrix, kron, identity, linalg

from pyqed.units import au2fs, au2k, au2ev
from pyqed import dag, coth, ket2dm, comm, anticomm, sigmax, sort, Mol

from pyqed.optics import Pulse
from pyqed.wpd import SPO2

import sys
if sys.version_info[1] < 10:
    import proplot as plt
else:
    import matplotlib.pyplot as plt


class Composite(Mol):
    def __init__(self, A, B):
        """

        Parameters
        ----------
        A : mol object
            Quantum system.
        B : object
            Quantum system.

        Returns
        -------
        None.

        """

        self.A = A
        self.B = B
        self.idm = kron(A.idm, B.idm)  # identity matrix
        self.ida = A.idm
        self.idb = B.idm
        self.H = None
        self.nonhermH = None
        self.rdm_a = None
        self.rdm_b = None
        self.dim = A.dim * B.dim
        self.dims = [A.dim, B.dim]
        self.eigvals = None
        self.eigvecs = None

    def getH(self, a_ops=None, b_ops=None, g=0):
        """
        Compute the Hamiltonian for the composite A + B system.

        The interaction between A and B are
            V_AB = sum_i g[i] * a_ops[i] * b_ops[i]
        Parameters
        ----------
        a_ops: list of arrays
            coupling operators for subsystem A
        b_ops: list of arrays
            coupling operators for subsystem B
        g: coupling constants

        Returns
        -------

        """

        H = kron(self.A.H, self.idb) + kron(self.ida, self.B.H)

        if a_ops == None:

            print('Warning: there is no coupling between the two subsystems.')



        elif isinstance(a_ops, list):

            for i, a_op in enumerate(a_ops):
                b_op = b_ops[i]
                H += g[i] * kron(a_op, b_op)


        elif isinstance(a_ops, np.ndarray):

            H += g * a_ops @ b_ops

        self.H = H

        return H

    def get_nonhermH(self, a_ops=None, b_ops=None, g=0):
        """
        The interaction between A and B are
            V_AB = sum_i g[i] * a_ops[i] * b_ops[i]
        Parameters
        ----------
        a_ops: list of arrays
            coupling operators for subsystem A
        b_ops: list of arrays
            coupling operators for subsystem B
        g: coupling constants

        Returns
        -------

        """
        if self.A.nonhermH is None:
            raise ValueError('Call get_nonhermH() for subsystem A first.')

        if self.B.nonhermH is None:
            raise ValueError('Call get_nonhermH() for subsystem B first.')

        H = kron(self.A.nonhermH, self.idb) + kron(self.ida, self.B.nonhermH)

        if a_ops == None:

            print('Warning: there is no coupling between the two subsystems.')

        elif isinstance(a_ops, list):

            for i, a_op in enumerate(a_ops):
                b_op = b_ops[i]
                H += g[i] * kron(a_op, b_op)


        elif isinstance(a_ops, np.ndarray):

            H += g * a_ops @ b_ops

        self.nonhermH = H

        return H

    def promote(self, o, subspace='A'):
        """
        promote an operator in subspace to the full Hilbert space
        E.g. A = A \otimes I_B
        """
        if subspace == 'A':
            return kron(o, self.B.idm)

        elif subspace == 'B':
            return kron(self.A.idm, o)
        else:
            raise ValueError('The subspace option can only be A or B.')

    def promote_ops(self, ops, subspaces=None):
        if subspaces is None:
            subspaces = ['A'] * len(ops)

        new_ops = []
        for i, op in enumerate(ops):
            new_ops.append(self.promote(op, subspaces[i]))

        return new_ops


    def eigenstates(self, k=None):
        """
        compute the polaritonic spectrum

        Parameters
        ----------
        k : int, optional
            number of eigenstates. The default is 1.
        sparse : TYPE, optional
            if the Hamiltonian is sparse. The default is True.

        Returns
        -------
        evals : TYPE
            DESCRIPTION.
        evecs : TYPE
            DESCRIPTION.
        n_ph : TYPE
            photonic fractions in polaritons.

        """

        if self.H is None:
            sys.exit('Please call getH to compute the Hamiltonian first.')

        if k is None:

            # compute the full polariton states
            if issparse(self.H):
                evals, evecs = scipy.linalg.eigh(self.H.toarray())
            else:
                evals, evecs = scipy.linalg.eigh(self.H)

            self.eigvals = evals
            self.eigvecs = evecs
            return evals, evecs

        elif k < self.dim:

            if issparse(self.H):
                evals, evecs = linalg.eigsh(self.H, k, which='SA')
            else:
                raise TypeError('H is not sparse matrix.')

            self.eigvals = evals
            self.eigvecs = evecs
            return evals, evecs

        else:
            raise ValueError('k cannot exceed the size of H.')

    def spectrum(self):
        if self.H is None:
            sys.exit('Call getH() to compute the full Hamiltonian first.')
        else:
            eigvals, eigvecs = np.linalg.eigh(self.H.toarray())

            return eigvals, eigvecs

    def transform_basis(self, a):
        """
        transform the operator a from the direct product basis to polariton basis

        Parameters
        ----------
        a : TYPE
            DESCRIPTION.

        Returns
        -------
        2d array

        """
        if self.eigvecs is None:
            self.eigenstates()

        return basis_transform(a, self.eigvecs)

            # raise ValueError('Call eigenstates() to compute eigvecs first.')

    def rdm(self, psi, which='A'):
        """
        compute the reduced density matrix of A/B from a pure state

        Parameters
        ----------
        psi: array
            pure state
        which: str
            indicator of which rdm A or B. Default 'A'.

        Returns
        -------

        """
        na = self.A.dim
        nb = self.B.dim
        psi_reshaped = psi.reshape((na, nb))
        if which == 'A':

            rdm = psi_reshaped @ dag(psi_reshaped)
            return rdm

        elif which == 'B':
            rdm = psi_reshaped.T @ psi_reshaped.conj()
            return rdm
        else:
            raise ValueError('which option can only be A or B.')

    def purity(self, psi):
        rdm = self.rdm(psi)

        return np.trace(rdm @ rdm)


def ham_ho(freq, n, ZPE=False):
    """
    input:
        freq: fundemental frequency in units of Energy
        n : size of matrix
    output:
        h: hamiltonian of the harmonic oscilator
    """

    if ZPE:
        energy = np.arange(n + 0.5) * freq
    else:
        energy = np.arange(n) * freq

    return np.diagflat(energy)


# class Mol:
#     def __init__(self, ham, dip, rho=None):
#         self.ham = ham
#         #self.initial_state = psi
#         self.dm = rho
#         self.dip = dip
#         self.n_states = ham.shape[0]
#         self.ex = np.tril(dip.toarray())
#         self.deex = np.triu(dip.toarray())
#         self.idm = identity(ham.shape[0])
#         self.size = ham.shape[0]

#     def set_dip(self, dip):
#         self.dip = dip
#         return

#     def set_dipole(self, dip):
#         self.dip = dip
#         return

#     def get_ham(self):
#         return self.ham

#     def get_dip(self):
#         return self.dip

#     def get_dm(self):
#         return self.dm


def fft(t, x, freq=np.linspace(0,0.1)):

    t = t/au2fs

    dt = (t[1] - t[0]).real

    sp = np.zeros(len(freq), dtype=np.complex128)

    for i in range(len(freq)):
        sp[i] = x.dot(np.exp(1j * freq[i] * t - 0.002*t)) * dt

    return sp


def obs(A, rho):
    """
    compute observables
    """
    return A.dot( rho).diagonal().sum()

# def rk4_step(a, fun, dt, *args):

#     dt2 = dt/2.0

#     k1 = fun(a, *args)
#     k2 = fun(a + k1*dt2, *args)
#     k3 = fun(a + k2*dt2, *args)
#     k4 = fun(a + k3*dt, *args)

#     a += (k1 + 2*k2 + 2*k3 + k4)/6. * dt
#     return a

# def comm(a,b):
#     return a.dot(b) - b.dot(a)

# def anticomm(a,b):
#     return a.dot(b) + b.dot(a)


# class Pulse:
#     def __init__(self, delay, sigma, omegac, amplitude=0.01, cep=0.):
#         """
#         Gaussian pulse exp(-(t-T)^2/2 * sigma^2)
#         """
#         self.delay = delay
#         self.sigma = sigma
#         self.omegac = omegac # central frequency
#         self.unit = 'au'
#         self.amplitude = amplitude
#         self.cep = cep

#     def envelop(self, t):
#         return np.exp(-(t-self.delay)**2/2./self.sigma**2)

#     def spectrum(self, omega):
#         omegac = self.omegac
#         sigma = self.sigma
#         return sigma * np.sqrt(2.*np.pi) * np.exp(-(omega-omegac)**2 * sigma**2/2.)

#     def field(self, t):
#         '''
#         electric field
#         '''
#         omegac = self.omegac
#         delay = self.delay
#         a = self.amplitude
#         sigma = self.sigma
#         return a * np.exp(-(t-delay)**2/2./sigma**2)*np.cos(omegac * (t-delay))


class Cavity():
    def __init__(self, freq, n_cav=None, x=None, decay=None):
        """
        class for single-mode cavity

        Parameters
        ----------
        freq : TYPE
            DESCRIPTION.
        n_cav : TYPE, optional
            DESCRIPTION. The default is None.
        x : TYPE, optional
            DESCRIPTION. The default is None.
        decay : float, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.freq = self.omega = freq
        self.resonance = freq
        self.ncav = self.n_cav = n_cav
        self.n = self.dim = n_cav

        self.idm = identity(n_cav)
        # self.create = self.get_create()

        # self.a = self.get_annihilate()
        self.H = self.getH()
        
        # number of grid points to represent quadrature
        if x is not None:
            self.x = x
            self.nx = len(x)
        
        self.decay = decay 
#    @property
#    def hamiltonian(self):
#        return self._hamiltonian
#
#    @hamiltonian.setter
#    def hamiltonian(self):
#        self._hamiltonian = ham_ho(self.resonance, self.n)

    def ground_state(self, sparse=True):
        """
        get initial density matrix for cavity vacuum state
        """
        vac = np.zeros(self.n_cav)
        vac[0] = 1.
        if sparse:
            return csr_matrix(vac)
        else:
            return vac
    
    def vacuum(self, sparse=True):
        """
        get initial density matrix for cavity vacuum state
        """
        vac = np.zeros(self.n_cav)
        vac[0] = 1.
        if sparse:
            return csr_matrix(vac)
        else:
            return vac

    def vacuum_dm(self):
        """
        get initial density matrix for cavity vacuum state
        """
        vac = np.zeros(self.n_cav)
        vac[0] = 1.
        return ket2dm(vac)
    
    def getH(self, zpe=False):
        return ham_ho(self.freq, self.n_cav)

    def create(self):
        n_cav = self.n_cav
        c = lil_matrix((n_cav, n_cav))
        c.setdiag(np.sqrt(np.arange(1, n_cav)), -1)
        return c.tocsr()

    def annihilate(self):
        n_cav = self.n_cav
        a = lil_matrix((n_cav, n_cav))
        a.setdiag(np.sqrt(np.arange(1, n_cav)), 1)

        return a.tocsr()

    def get_dm(self):
        """
        get initial density matrix for cavity
        """
        vac = np.zeros(self.n_cav)
        vac[0] = 1.
        return ket2dm(vac)

    def get_number_operator(self):
        """
        number operator
        """
        ncav = self.n_cav
        a = lil_matrix((ncav, ncav))
        a.setdiag(range(ncav), 0)
        return a.tocsr()

    def quadrature(self):
        """
        quadrature; corresponding to the displacement field D
        
        .. math::
            D = \frac{1}{\sqrt{2}} (a + a^\dag)

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        a = self.annihilate()
        return 1./np.sqrt(2.) * (a + dag(a))

    def get_nonhermitianH(self):
        '''
        non-Hermitian Hamiltonian for the cavity mode

        Params:
            kappa: decay constant

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        ncav = self.n_cav
        if self.quality_factor is not None:
            kappa = self.freq/2./self.quality_factor
        else:
            raise ValueError('The quality factor cannot be None.')

        self.nonhermH = self.H - 1j * kappa * np.identity(ncav)
        return self.nonhermH

    def get_nonhermH(self):
        return self.get_nonhermitianH()

class Polariton(Composite):
    def __init__(self, mol, cav):

        super(Polariton, self).__init__(mol, cav)

        self.mol = mol
        self.cav = cav
        self._ham = self.H = None
        self.dip = None
        self.cav_leak = None
        self.H = None
        self.dims = [mol.dim, cav.n_cav]
        self.dim = mol.dim * cav.n_cav
        #self.dm = kron(mol.dm, cav.get_dm())

    def getH(self, g, RWA=False):
        """


        Parameters
        ----------
        g : float
            single photon electric field strength sqrt(hbar * omegac / 2 episilon_0 V)
        RWA : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            Hamiltonian.

        """

        mol = self.mol
        cav = self.cav

        hmol = mol.getH()
        hcav = cav.getH()

        Icav = cav.idm
        Imol = mol.idm

        if RWA:

            hint = g * (kron(mol.raising, cav.annihilate()) +
                        kron(mol.lowering, cav.create()))

        else:

            hint = g * kron(mol.edip, cav.create() + cav.annihilate())

        self.H = kron(hmol, Icav) + kron(Imol, hcav) + hint

        return self.H

    def get_nonhermitianH(self, g, RWA=False):

        mol = self.mol
        cav = self.cav

        hmol = mol.get_nonhermitianH()
        hcav = cav.get_nonhermitianH()

        Icav = cav.idm
        Imol = mol.idm

        if RWA:

            hint = g * (kron(mol.raising(), cav.get_annihilate()) +
                        kron(mol.lowering(), cav.get_create()))

        else:

            hint = g * kron(mol.dip, cav.get_create() + cav.get_annihilate())

        H = kron(hmol, Icav) + kron(Imol, hcav) + hint

        return H

    def get_ham(self, RWA=False):
        return self.getH(RWA)

    def setH(self, h):
        self.H = h
        return

    def get_edip(self, basis='product'):
        '''
        transition dipole moment in the direct product basis

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return kron(self.mol.edip, self.cav.idm)

    def get_dm(self):
        return kron(self.mol.dm, self.cav.vacuum_dm())

    def get_cav_leak(self):
        """
        damp operator for the cavity mode
        """
        if self.cav_leak == None:
            self.cav_leak = kron(self.mol.idm, self.cav.annihilate)

        return self.cav_leak

    def eigenstates(self, k=None):
        """
        compute the polaritonic spectrum

        Parameters
        ----------
        k : int, optional
            number of eigenstates. The default is 1.
        sparse : TYPE, optional
            if the Hamiltonian is sparse. The default is True.

        Returns
        -------
        evals : TYPE
            DESCRIPTION.
        evecs : TYPE
            DESCRIPTION.
        n_ph : TYPE
            photonic fractions in polaritons.

        """

        if self.H == None:
            sys.exit('Please call getH to compute the Hamiltonian first.')


        if k is None:

            """
            compute the full polariton states with numpy
            """

            h = self.H.toarray()
            # evals, evecs = scipy.linalg.eigh(h, subset_by_index=[0, self.dim])
            evals, evecs = scipy.linalg.eigh(h)
            # number of photons in polariton states
            num_op = self.cav.num()
            num_op = kron(self.mol.idm, num_op)

            n_ph = np.zeros(self.dim)
            for j in range(self.dim):
                n_ph[j] = np.real(evecs[:, j].conj().dot(
                    num_op.dot(evecs[:, j])))

            self.eigvals = evals
            self.eigvecs = evecs

            return evals, evecs

        elif k < self.dim:

            evals, evecs = linalg.eigsh(self.H, k, which='SA')

            # number of photons in polariton states
            num_op = self.cav.num()
            num_op = kron(self.mol.idm, num_op)

            n_ph = np.zeros(k)
            for j in range(k):
                n_ph[j] = np.real(evecs[:, j].conj().dot(
                    num_op.dot(evecs[:, j])))

            return evals, evecs, n_ph

    def promote_op(self, a, kind='mol'):
        """
        promote a local operator to the composite polariton space

        Parameters
        ----------
        a : TYPE
            DESCRIPTION.
        kind : TYPE, optional
            DESCRIPTION. The default is 'mol'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if kind in ['mol', 'm']:

            return kron(a, self.cav.idm)

        elif kind in ['cav', 'c']:

            return kron(self.mol.idm, a)
        
    def rdm_photon(self):
        """
        return the reduced density matrix for the photons
        """
    def transform_basis(self, a):
        """
        transform the operator a from the direct product basis to polariton basis

        Parameters
        ----------
        a : TYPE
            DESCRIPTION.

        Returns
        -------
        2d array

        """
        if self.eigvecs is None:
            self.eigenstates()

        return basis_transform(a, self.eigvecs)

    def driven_dynamics(self, psi0, pulse, dt=0.001, nt=1, obs_ops=None, nout=1, t0=0.0):
        H = self.H

        if self.dip is None:
            edip = self.get_dip()
        else:
            edip = self.dip

        # psi0 = self.initial_state

        if psi0 is None:
            sys.exit("Error: Initial wavefunction not specified!")

        H = [self.H, [edip, pulse]]

        return driven_dynamics(H, psi0, dt=dt, Nt=nt, \
                        e_ops=obs_ops, nout=nout, t0=t0)

# class Polariton:
#     def __init__(self, mol, cav, g=None):
#         self.g = g
#         self.mol = mol
#         self.cav = cav
#         self._ham = None
#         self.dip = None
#         self.cav_leak = None
#         #self.dm = kron(mol.dm, cav.get_dm())
        
#         self.H = None

#     def getH(self, g, RWA=True):
#         mol = self.mol
#         cav = self.cav

#         # g = self.g

#         hmol = mol.get_ham()
#         hcav = cav.get_ham()

#         Icav = identity(self.cav.n_cav)
#         Imol = identity(self.mol.n_states)

#         if RWA == True:
#             hint = g * (kron(mol.ex, cav.get_annihilate()) + kron(mol.deex, cav.get_create()))
#         elif RWA == False:
#             hint = g * kron(mol.dip, cav.get_create() + cav.get_annihilate())

#         H = kron(hmol, Icav) + kron(Imol, hcav) + hint
#         self.H = H
#         return H

#     def get_ham(self, RWA=False):
#         print('Deprecated. Please use getH()')
#         return

#     def get_dip(self):
#         return kron(self.mol.dip, self.cav.idm)

#     def get_dm(self):
#         return kron(self.mol.dm, self.cav.vacuum_dm())

#     def get_cav_leak(self):
#         """
#         damp operator for the cavity mode
#         """
#         if self.cav_leak == None:
#             self.cav_leak = kron(self.mol.idm, self.cav.annihilate)

#         return self.cav_leak

#     def spectrum(self, nstates, RWA=False):
#         """
#         compute the polaritonic spectrum
#         """
#         ham = self.get_ham(RWA=RWA)
#         ham = csr_matrix(ham)
#         return linalg.eigsh(ham, nstates, which='SA')

#     def rdm_photon(self):
#         """
#         return the reduced density matrix for the photons
#         """


class VibronicPolariton:
    """
    a 1D vibronic model coupled to a single-mode cavity
    """
    # def __init__(self, x, y, masses, nstates=2, coords='linear', G=None, abc=False):

    #     super.__init__(x, y, masses, nstates, coords, G, abc)

    def __init__(self, mol, cav):

        self.mol = mol
        self.cav = cav
        self.x = mol.x
        self.nx = mol.nx

        self.nstates = mol.nstates * cav.ncav # number of polariton states

        self.v = None
        self.va = None

    # def build(self):
    #     mol = self.mol
    #     cav = self.cav

    #     nx = mol.nx
    #     nstates = mol.states

    #     nc = cav.nx
    #     v = np.zeros((nx, nc, nstates))

    #     for n in range(nc):
    #         v[:, n, :] = mol.v

    def dpes(self, g, rwa=False, gauge='dipole'):

        # if rwa == False:
        #     raise NotImplementedError('Counterrotating terms are not included yet.\
        #                               Please contact Bing Gu to add.')

        mol = self.mol
        cav = self.cav

        omegac = cav.omega
        nx = mol.nx
        nel = mol.nstates
        ncav = cav.ncav

        nstates = self.nstates

        v = np.zeros((nx, nstates, nstates), dtype=complex)


        for j in range(nstates):
            a, n = np.unravel_index(j, (nel, ncav))
            v[:, j, j] = mol.v[:, a, a] + n * omegac


        # cavity-molecule coupling
        a = cav.annihilate()
        # for i in range(nx):
        #     v[i, :, :] += g * kron(mol.edip, a + dag(a))


        if mol.edip.shape == (nel, nel):
            # Condon approximation
            v += np.tile(g * kron(mol.edip, a + dag(a)).toarray(), (nx, 1, 1))

        elif mol.edip.shape == (nx, nel, nel):
            
            for i in range(nx):
                v[i, :, :] += g * kron(mol.edip[i], a + dag(a)).toarray() 

        self.v = v
        return v

    def add_coupling(self, ops):
        
        nel = self.mol.nstates
        nx = self.nx 
        
        for pair in ops:
            
            mol_op, cav_op = pair
            
            if mol_op.shape == (nel, nel):
                # Condon approximation
                self.v += np.tile( kron(mol_op, cav_op), (nx, 1, 1))
    
            elif mol_op.shape == (nx, nel, nel):
                
                for i in range(nx):
                    self.v[i, :, :] += kron(mol_op[i], cav_op) 

        return self.v        

    def ppes(self):
        """
        Compute polaritonic surfaces

        Returns
        -------
        E : TYPE
            DESCRIPTION.

        """
        E = np.zeros((self.nx, self.nstates))
        
        for i in range(self.nx):
            V = self.v[i, :, :]
            w, u = sort(*np.linalg.eigh(V))
            # E, U = sort(E, U)
            E[i, :] = w

        self.va = E
        return E

    def ground_state(self, rwa=False):
        if rwa:
            v = self.mol.v[:, 0, 0]
        else:
            # construct the ground-state polaritonic PES
            v = self.va[:, 0, 0]

        # DVR
        from pyqed.dvr.dvr_1d import SincDVR
        L = self.x.max() - self.x.min()
        dvr = SincDVR(128, L)


    def draw_surfaces(self, n=None, representation='diabatic'):
        if self.v is None:
            raise ValueError('Call dpes() first.')

        if n is None:
            n = self.nstates

        if representation == 'diabatic':
            v = self.v

            fig, ax = plt.subplots()
            for j in range(n):
                ax.plot(self.x, v[:, j,j].real)

            fig, ax = plt.subplots()
            for j in range(n):
                for i in range(j):
                    ax.plot(self.x, v[:, i, j].real)

        elif representation == 'adiabatic':
            v = self.va

            fig, ax = plt.subplots()
            for j in range(n):
                ax.plot(self.x, v[:, j].real)

        return

    def run(self, psi0, dt, nt=1, t0=0, nout=1):

        from pyqed.namd.diabatic import SPO

        spo = SPO(self.x, mass=self.mol.mass, nstates=self.nstates, v=self.v)
        return spo.run(psi0=psi0, dt=dt, nt=nt)


    def plot_surface(self, state_id=0, representation='adiabatic'):
        # from pyqed.style import plot_surface
        fig, ax = plt.subplots()
        if representation == 'adiabatic':
            ax.plot(self.x, self.va[:, state_id])
        else:
            ax.plot(self.x, self.v[:, state_id, state_id])
            
        return

class VibronicPolariton2:
    """
    2D vibronic model in the diabatic representation coupled to
    a single-mode optical cavity (electronic strong coupling)

    """
    def __init__(self, mol, cav):
        self.mol = mol
        self.cav = cav
        self.x, self.y = mol.x, mol.y
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.nx, self.ny = mol.nx, mol.ny
        self.nel = self.mol.nstates
        self.ncav = self.cav.ncav
        self.nstates = self.nel * self.ncav


        self.v = None
        self.va = None # adiabatic polaritonic PES
        self._transformation = None # diabatic to adiabatic transformation matrix
        self._ground_state = None

    def ground_state(self, representation='adiabatic'):
        # if rwa:
        #     return self.mol.ground_state()
        # else:
        #     # construct the ground-state polaritonic PES

        from pyqed.dvr.dvr_2d import DVR2

        x = self.x
        y = self.y

        if self.va is None:
            self.ppes()

        dvr = DVR2(x, y, mass=self.mol.mass) # for normal coordinates

        if representation == 'adiabatic':
            V = self.va[:, :, 0]
        elif representation == 'diabatic':
            V = self.v[:, :, 0, 0]

        E0, U = dvr.run(V, k=1)

        self._ground_state = U[:, 0].reshape(self.nx, self.ny)

        return E0, self._ground_state


    def dpes(self, g, rwa=False):
        """
        Compute the diabatic potential energy surfaces

        Parameters
        ----------
        g : TYPE
            DESCRIPTION.
        rwa : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        mol = self.mol
        cav = self.cav

        omegac = cav.omega

        nx, ny = self.nx, self.ny

        nel = mol.nstates
        ncav = cav.ncav

        nstates = self.nstates # polariton states

        v = np.zeros((nx, ny, nstates, nstates))
        
        # build the global DPES
        if mol.v is None:
            mol.dpes_global()

        for j in range(nstates):
            a, n = np.unravel_index(j, (nel, ncav))
            v[:, :, j, j] = mol.v[:, :, a, a] + n * omegac


        # cavity-molecule coupling
        a = cav.annihilate()

        v += np.tile(g * kron(mol.edip.real, a + dag(a)).toarray(), (nx, ny, 1, 1))

        self.v = v

        return v

    def ppes(self, return_transformation=False):
        """
        Compute the polaritonic potential energy surfaces by diagonalization

        Parameters
        ----------
        return_transformation : TYPE, optional
            Return transformation matrices. The default is False.

        Returns
        -------
        E : TYPE
            DESCRIPTION.

        """

        nx = self.nx
        ny = self.ny
        N = self.nstates

        E = np.zeros((self.nx, self.ny, self.nstates))

        if not return_transformation:

            for i in range(self.nx):
                for j in range(self.ny):
                    V = self.v[i, j, :, :]
                    w = np.linalg.eigvalsh(V)
                    # E, U = sort(E, U)
                    E[i, j, :] = w
        else:

            T = np.zeros((nx, ny, N, N), dtype=complex)

            for i in range(self.nx):
                for j in range(self.ny):
                    V = self.v[i, j, :, :]
                    w, u = sort(*np.linalg.eigh(V))

                    E[i, j, :] = w
                    T[i, j, :, :] = u

            self._transformation = T

        self.va = E

        return E

    def berry_curvature(self, state_id):
        # compute Berry curvature from the A2D transformation matrix
        pass

    def run(self, psi0=None, dt=0.1, Nt=10, t0=0, nout=1):

        if psi0 is None:
            psi0 = np.zeros((self.nx, self.ny, self.nstates))
            psi0[:, :, 0] = self._ground_state

        spo = SPO2(self.x, self.y, mass=self.mol.mass, nstates=self.nstates)
        spo.V = self.v

        return spo.run(psi0=psi0, dt=dt, Nt=Nt, t0=t0, nout=nout)


    def plot_surface(self, state_id=0, representation='adiabatic'):
        from pyqed.style import plot_surface

        if representation == 'adiabatic':
            plot_surface(self.x, self.y, self.va[:, :, state_id])
        else:
            plot_surface(self.x, self.y, self.v[:, :, state_id, state_id])
            
        return

    def plot_wavepacket(self, psilist, **kwargs):

        if not isinstance(psilist, list): psilist = [psilist]


        for i, psi in enumerate(psilist):
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharey=True)

            ax0.contour(self.X, self.Y, np.abs(psi[:,:, 1])**2)
            ax1.contour(self.X, self.Y, np.abs(psi[:, :,0])**2)
            # ax0.format(**kwargs)
            # ax1.format(**kwargs)
            fig.savefig('psi'+str(i)+'.pdf')
        return ax0, ax1

    def plot_ground_state(self, **kwargs):


        fig, ax = plt.subplots()

        ax.contour(self.X, self.Y, np.real(self._ground_state))
        ax.format(**kwargs)

        fig.savefig('ground_state.pdf')
        return ax

    def promote_op(self, a, kind='mol'):
        """
        promote a local operator to the composite polariton space

        Parameters
        ----------
        a : TYPE
            DESCRIPTION.
        kind : TYPE, optional
            DESCRIPTION. The default is 'mol'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if kind in ['mol', 'm']:

            return kron(a, self.cav.idm)

        elif kind in ['cav', 'c']:

            return kron(self.mol.idm, a)



class VibronicPolaritonNonHermitian(VibronicPolariton2):
    
    def dpes(self, g, decay, rwa=False):
        """
        Compute the non-Hermitian diabatic potential energy surfaces
        
        .. math::
            H = (\omega_\text{c} - \frac{\kappa}{2} ) a^\dag a

        Parameters
        ----------
        g : TYPE
            DESCRIPTION.
        rwa : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        mol = self.mol
        cav = self.cav

        omegac = cav.omega - 0.5j * cav.decay

        nx, ny = self.nx, self.ny

        nel = mol.nstates
        ncav = cav.ncav

        nstates = self.nstates # polariton states

        v = np.zeros((nx, ny, nstates, nstates), dtype=complex)
        
        # build the global DPES
        if mol.v is None:
            mol.dpes_global()

        for j in range(nstates):
            a, n = np.unravel_index(j, (nel, ncav))
            v[:, :, j, j] = mol.v[:, :, a, a] + n * omegac


        # cavity-molecule coupling
        a = cav.annihilate()

        v += np.tile(g * kron(mol.edip.real, a + dag(a)).toarray(), (nx, ny, 1, 1))

        self.v = v

        return v  
    
    def ppes(self, return_transformation=True):
        """
        Compute the polaritonic potential energy surfaces by diagonalization

        Parameters
        ----------
        return_transformation : TYPE, optional
            Return transformation matrices. The default is False.

        Returns
        -------
        E : array [nx, ny, nstates]
            eigenvalues
        
        T : array [nx, ny, nstates, nstates]
            transformation matrix from diabatic states to polaritonic states

        """
        
        if self.cav.decay is None:
            raise ValueError('Please set the cavity decay rate.')
            
        nx = self.nx
        ny = self.ny
        N = self.nstates

        E = np.zeros((self.nx, self.ny, self.nstates))

        if not return_transformation:

            for i in range(self.nx):
                for j in range(self.ny):
                    V = self.v[i, j, :, :]
                    w = np.linalg.eigvals(V)
                    # E, U = sort(E, U)
                    E[i, j, :] = w
            
            self.va = E
            return E
        
        else:

            T = np.zeros((nx, ny, N, N), dtype=complex)

            for i in range(self.nx):
                for j in range(self.ny):
                    V = self.v[i, j, :, :]
                    w, u = sort(*scipy.linalg.eig(V, right=True))

                    E[i, j, :] = w
                    T[i, j, :, :] = u

            self._transformation = T

            self.va = E

            return E, T
    
    def ldr(self):
        pass
        

class DHO2:
    def __init__(self, x, y, nstates=2):
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.nstates = nstates

        self.v = None
        self.edip = sigmax()

    def dpes(self, d=1, E0=1.):
        v = np.zeros((self.nx, self.nstates, self.nstates))
        x = self.x

        v[:, 0, 0] = 0.5 * x**2
        v[:, 1, 1] = 0.5 * (x-d)**2 + E0

        self.v = v
        return v





if __name__ == '__main__':

    from pyqed.models.pyrazine import DHO
    x = np.linspace(-2, 2, 64)
    y = np.linspace(-2, 2, 64)

    mol = DHO(x)
    mol.dpes(d=1, e0=1)

    cav = Cavity(1, 3)

    pol = VibronicPolariton(mol, cav)
    pol.dpes(g=0.05)
    pol.ppes()

    pol.draw_surfaces(n=4, representation='adiabatic')
    
    # pol.product_state(0, 0, 0)

    def test_vp2():
        from pyqed.models.pyrazine import LVC2
    
        mol = LVC2(x, y, mass=[1,1])
        mol.plot_apes()
        
        
        cav = Cavity(3/au2ev, 3)
    
        
        # mol.plot_surface()
    
        
        pol = VibronicPolariton2(mol, cav)
        pol.dpes(g=0.)
        
        # pol.ppes()
        pol.plot_surface(3, representation='diabatic')
    
        psi0 = np.zeros((len(x), len(y), pol.nstates), dtype=complex)
        psi0[:, :, 4] = pol.ground_state()[1]
        
        pol.plot_ground_state()
        
        r = pol.run(psi0=psi0, dt=0.05, Nt=20, nout=2)
    
        # r.plot_wavepacket(r.psilist, 4)


# the diabatic potential energy surface

