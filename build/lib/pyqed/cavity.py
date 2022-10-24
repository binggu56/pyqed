#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:26:02 2019


@author: Bing
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, kron, identity, linalg, issparse
import scipy
import sys


from pyqed import au2fs, au2k, au2ev, dag, coth, ket2dm, comm, anticomm, pauli, destroy,\
    basis, isherm, basis_transform
from pyqed.mol import Mol
from pyqed.oqs import Lindblad_solver





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

def ham_ho(freq, N, ZPE=False):
    """
    input:
        freq: fundamental frequency in units of Energy
        n : size of matrix
    output:
        h: hamiltonian of the harmonic oscillator
    """

    if ZPE:
        energy = np.arange(N + 0.5) * freq
    else:
        energy = np.arange(N) * freq

    H = lil_matrix((N, N))
    H.setdiag(energy)

    return H.tocsr()


def fft(t, x, freq=np.linspace(0, 0.1)):

    t = t/au2fs

    dt = (t[1] - t[0]).real

    sp = np.zeros(len(freq), dtype=np.complex128)

    for i in range(len(freq)):
        sp[i] = x.dot(np.exp(1j * freq[i] * t - 0.002*t)) * dt

    return sp

# def dag(H):
#     return H.conj().T

# def coth(x):
#     return 1./np.tanh(x)

# def ket2dm(psi):
#     return np.einsum("i, j -> ij", psi.conj(), psi)

# def obs(A, rho):
#     """
#     compute observables
#     """
#     return A.dot( rho).diagonal().sum()


def rk4_step(a, fun, dt, *args):

    dt2 = dt/2.0

    k1 = fun(a, *args)
    k2 = fun(a + k1*dt2, *args)
    k3 = fun(a + k2*dt2, *args)
    k4 = fun(a + k3*dt, *args)

    a += (k1 + 2*k2 + 2*k3 + k4)/6. * dt
    return a


class Pulse:
    def __init__(self, delay, sigma, omegac, amplitude=0.01, cep=0.):
        """
        Gaussian pulse A * exp(-(t-T)^2/2 / sigma^2)
        A: amplitude
        T: time delay
        sigma: duration
        """
        self.delay = delay
        self.sigma = sigma
        self.omegac = omegac  # central frequency
        self.unit = 'au'
        self.amplitude = amplitude
        self.cep = cep

    def envelop(self, t):
        return np.exp(-(t-self.delay)**2/2./self.sigma**2)

    def spectrum(self, omega):
        """
        Fourier transform of the Gaussian pulse
        """
        omegac = self.omegac
        sigma = self.sigma
        a = self.amplitude
        return a * sigma * np.sqrt(2.*np.pi) * np.exp(-(omega-omegac)**2 * sigma**2/2.)

    def field(self, t):
        '''
        electric field
        '''
        omegac = self.omegac
        delay = self.delay
        a = self.amplitude
        sigma = self.sigma
        return a * np.exp(-(t-delay)**2/2./sigma**2)*np.cos(omegac * (t-delay))


class Cavity():
    def __init__(self, freq, ncav, Q=None, polarization=None):
        self.frequency = self.freq = self.resonance = freq
        self.n_cav = self.ncav = ncav
        self.n = ncav
        self.dim = ncav # dimension of Fock space

        self.idm = identity(ncav)
        # self.create = self.get_create()
        # self.annihilate = destroy(ncav)
        self.H = self.getH()
        self.nonhermH = None

        self.quality_factor = Q
        self.decay = freq/2./Q
        self.polarization = polarization

#    @property
#    def hamiltonian(self):
#        return self._hamiltonian
#
#    @hamiltonian.setter
#    def hamiltonian(self):
#        self._hamiltonian = ham_ho(self.resonance, self.n)

    def get_ham(self, zpe=False):
        return self.getH(zpe)
    #
    # def H(self, ZPE=False):
    #     self._H = ham_ho(self.freq, self.dim, ZPE=ZPE)
    #     return self._H

    def getH(self, ZPE=False):
        self.H = ham_ho(self.freq, self.n_cav, ZPE=ZPE)
        return self.H

    def setQ(self, Q):
        self.quality_factor = Q

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

    def ham(self, ZPE=False):
        return ham_ho(self.freq, self.n_cav, ZPE=ZPE)

    def get_create(self):
        n_cav = self.n_cav
        c = lil_matrix((n_cav, n_cav))
        c.setdiag(np.sqrt(np.arange(1, n_cav)), -1)
        return c.tocsr()

    def get_annihilate(self):
        n_cav = self.n_cav
        a = lil_matrix((n_cav, n_cav))
        a.setdiag(np.sqrt(np.arange(1, n_cav)), 1)

        return a.tocsr()

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

    def get_num(self):
        """
        number operator
        """
        ncav = self.n_cav
        a = lil_matrix((ncav, ncav))
        a.setdiag(range(ncav), 0)

        return a.tocsr()

    def num(self):
        """
        number operator
        input:
            N: integer
                number of states
        """
        N = self.n_cav
        a = lil_matrix((N, N))
        a.setdiag(range(N), 0)
        return a.tocsr()
    
    # def x(self):
    #     a = self.annihilate()
    #     return 1./np.sqrt(2.) * (a + dag(a))


    
    
class Polariton(Composite):
    def __init__(self, mol, cav):

        super(Polariton, self).__init__(mol, cav)

        self.mol = mol
        self.cav = cav
        self._ham = None
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

            hint = g * kron(mol.dip, cav.create() + cav.annihilate())

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

    def get_dip(self, basis='product'):
        '''
        transition dipole moment in the direct product basis

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return kron(self.mol.dip, self.cav.idm)

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

def QRM(omega0, omegac, ncav=2):
    '''
    Quantum Rabi model / Jaynes-Cummings model

    Parameters
    ----------
    omega0 : float
        atomic transition frequency
    omegac : float
        cavity frequency
    g : float
        cavity-molecule coupling strength

    Returns
    -------
    rabi: object

    '''
    s0, sx, sy, sz = pauli()

    hmol = 0.5 * omega0 * (-sz + s0)

    mol = Mol(hmol, sx)  # atom
    cav = Cavity(omegac, ncav)  # cavity

    return Polariton(mol, cav)

class Dicke(Cavity):

    def __init__(self, onsite, nsites, omegac):
        self.nsites = nsites
        self.onsite = onsite

        self.omegac = omegac

        return

    def Hsubspace(self, g, nexc=1):
        '''
        single-polariton Hamiltonian for N two-level molecules coupled to
        a single cavity mode including the ground state

        input:
            g_cav: single cavity-molecule coupling strength
        '''
        nsites = self.nsites
        onsite = self.onsite
        omegac = self.omegac

        if nexc == 1:
            nstates = nsites + 1 + 1 # number of states in the system

            dip = np.zeros((nstates, nstates))
            # the states are respectively, |000>, |100>, |010>, |001>,
            # the position are ordered as

            H = np.diagflat([0.] + onsite + [omegac])
            H[1:nsites+1, -1] = H[-1, 1:nsites+1] = g

            dip[0,1:nsites+1] = dip[1:nsites+1, 0] =  1

            return csr_matrix(H), csr_matrix(dip)
        else:
            raise NotImplementedError()

    def Hfull(self):
        pass

if __name__ == '__main__':

    #mol = QRM(1, 1)
    from lime.phys import quadrature
    s0, sx, sy, sz = pauli()

    hmol = 0.5 * (-sz + s0)

    mol = Mol(hmol, sx)  # atom
    mol.set_decay_for_all(0.05)
    mol.get_nonhermH()

    cav = Cavity(1, 2, Q=100)  # cavity
    cav.get_nonhermH()

    pol = Polariton(mol, cav)
    H = pol.get_nonhermH(g=0.1)

    print(isherm(H.toarray()))

    # set up the initial state
    psi0 = basis(2, 0)
    rho0 = ket2dm(psi0)

    # mesolver = Lindblad_solver(H, c_ops=[0.2 * sx], e_ops = [sz])
    Nt = 200
    dt = 0.05

    # results = mesolver.evolve(rho0, dt=dt, Nt=Nt)
    # pol.getH([sx], [cav.annihilate() + cav.create()], [0.1])
    # evals, evecs = pol.eigenstates()
    # print(pol.purity(evecs[:,2]))

    # fig, ax = subplots()
    # ax.plot(times, )