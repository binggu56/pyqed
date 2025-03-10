#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:53:56 2018

@author: binggu
"""

import numpy as np
import sys
from scipy import linalg
from pyqed.mol import Mol, dag

class Floquet:
    """
    peridically driven multi-level system with a single frequency

    TODO: add more harmonics so it can treat a second harmonic driving
    """
    def __init__(self, H, edip, omegad, E0, nt):

        # super().__init__(H, edip)
        self.H = H
        self.edip = edip
        self.nt = nt
        self.E0 = E0
        
        self.omegad = omegad # driving freqency
        self.FBZ = [-omegad/2., omegad/2] # first Floquet-BZ

    def momentum_matrix_elements(self):
        """
        get momentum matrix elements by commutation relation
        .. math::
            p_{fi} = i E_{fi} r_{fi}

        Returns
        -------
        p : TYPE
            DESCRIPTION.

        """
        E = self.E
        p = 1j * np.subtract.outer(E, E) * self.edip
        return p

    def run(self, gauge='length', method='Floquet'):
        """
        .. math::
            E(t) = E_0 * \cos(\Omega t)
            A(t) = -\frac{E_0}{\Omega} * \sin(\Omega  t)

        Parameters
        ----------
        E0 : TYPE
            electric field amplitude.
        gauge : TYPE, optional
            DESCRIPTION. The default is 'length'.

        Returns
        -------
        quasienergies : TYPE
            DESCRIPTION.
        floquet_modes : TYPE
            DESCRIPTION.

        """
        H0 = self.H
        E0 = self.E0
        nt = self.nt 
        omegad = self.omegad
        

        if gauge == 'length': # electric dipole

            H1 = -0.5 * self.edip * E0

        elif gauge == 'velocity':

            H1 = 0.5j * self.momentum() * E0/omegad
        
        quasienergies, floquet_modes, G = quasiE(H0, H1, nt, omegad, method=1)

        return quasienergies, floquet_modes, G
    
    def winding_number(self): # To be modified
        pass

    def velocity_to_length(self):
        # transform the truncated velocity gauge Hamiltonian to length gauge
        pass
    
    def propagator(self, t1, t2=0):
        pass


class FloquetGF(Floquet):
    """
    Floquet Green's function formalism
    """
    def __init__(self, mol, amplitude, omegad, nt):
        # super.__init__(self, t, obj)
        self._mol = mol
        self.omegad = omegad
        self.nt = nt
        self.norbs = mol.size

        self.quasienergy, self.floquet_modes = self.spectrum(amplitude, nt, omegad)
        self.eta = 1e-3
        
    def bare_gf(self):
        # bare Green's functions
        quasienergy, fmodes = self.quasienergy, self.floquet_modes
        pass
        
    def retarded(self, omega, representation='floquet'):
        nt = self.nt
        norb = self.norb
        omegad = self.omegad
        
        # nomega = len(omega)
        
        if representation == 'floquet':
            
            # the frequency should be in the first BZ
            # omega = np.linspace(-omegad/2, omegad/2, nomega)
            assert -omegad/2 < omega < omegad/2
                
            gr = np.zeros((nt, nt, norb, norb), dtype=complex)
            
            for m in range(nt):
                for n in range(nt):
                    pass
                            
        elif representation == 'wigner':
            
            gr = np.zeros((nt, norb, norb), dtype=complex)
        
        # else:
        #     raise ValueError('Representation {} does not exist. \
        #                      Allowed values are `floquet`, `wigner`'.format(representation))
            
        return gr
    
    def wigner_to_floquet(self, g, omega):
        # transform from the Wigner representation to the Floquet representation 
        # Ref: Tsuji, Oka, Aoki, PRB 78, 235124 (2008)
        omegad = self.omegad
        
        assert -omegad/2 < omega < omegad/2
        
        # l = m - n 
        # gf[m, n, omega] = g[l, omega + (m + n)/2 * omegad  ] 
        pass
    
    def advanced(self):
        pass
    
    def lesser(self):
        pass
    
    def greater(self):
        pass

def _quasiE(H0, H1, Nt, omega):
    """
    Construct the Floquet hamiltonian of size Norbs * Nt

    The total Hamiltonian is
    .. math::
        H = H_0 + H_1 \cos(\Omega * t)

    INPUT
        Norbs : number of orbitals
        Nt    : number of Fourier components
        E0    : electric field amplitude
    """

    Norbs = H0.shape[-1]

    #print('transition dipoles \n', M)

    # dimensionality of the Floquet matrix
    NF = Norbs * Nt
    F = np.zeros((NF,NF), dtype=complex)

    N0 = -(Nt-1)/2 # starting point for Fourier companent of time exp(i n w t)

    idt = np.identity(Nt)
    idm = np.identity(Norbs)
    # construc the Floquet H for a general tight-binding Hamiltonian
    for n in range(Nt):
        for m in range(Nt):

            # atomic basis index
            for k in range(Norbs):
                for l in range(Norbs):

                # map the index i to double-index (n,k) n : time Fourier component
                # with relationship for this :  Norbs * n + k = i

                    i = Norbs * n + k
                    j = Norbs * m + l
                    F[i,j] = HamiltonFT(H0, H1, n-m)[k,l] + (n + N0) \
                             * omega * idt[n,m] * idm[k,l]


    # for a two-state model

#    for n in range(Nt):
#        for m in range(Nt):
#            F[n * Norbs, m * Norbs] = (N0 + n) * omega * delta(n,m)
#            F[n * Norbs + 1, m * Norbs + 1] = (onsite1 + (N0+n) * omega) * delta(n,m)
#            F[n * Norbs, m * Norbs + 1] = t * delta(n,m+1)
#            F[n * Norbs + 1, m * Norbs] = t * delta(n,m-1)
    #print('\n Floquet matrix \n', F)

    # compute the eigenvalues of the Floquet Hamiltonian,
    eigvals, eigvecs = linalg.eigh(F)

    #print('Floquet quasienergies', eigvals)

    # specify a range to choose the quasienergies, choose the first BZ
    # [-hbar omega/2, hbar * omega/2]
    eigvals_subset = np.zeros(Norbs, dtype=complex)
    eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


    # check if the Floquet states is complete
    j = 0
    for i in range(NF):
        if  eigvals[i] < omega/2.0 and eigvals[i] > -omega/2.0:
            eigvals_subset[j] = eigvals[i]
            eigvecs_subset[:,j] = eigvecs[:,i]
            j += 1
    if j != Norbs:
        print("Error: Number of Floquet states {} is not equal to \
              the number of orbitals {} in the first BZ. \n".format(j, Norbs))
        sys.exit()


    # now we have a complete linear independent set of solutions for the time-dependent problem
    # to compute the coefficients before each Floquet state if we start with |alpha>
    # At time t = 0, constuct the overlap matrix between Floquet modes and system eigenstates
    # G[j,i] = < system eigenstate j | Floquet state i >
    G = np.zeros((Norbs,Norbs), dtype=complex)
    for i in range(Norbs):
        for j in range(Norbs):
            tmp = 0.0
            for m in range(Nt):
                tmp += eigvecs_subset[m * Norbs + j, i]
            G[j,i] = tmp


    # to plot G on site basis, transform it to site-basis representation
    #Gsite = U.dot(G)

    return eigvals_subset, eigvecs_subset


def quasiE(H0, H1, Nt, omega, method=1):
    """
    Construct the Floquet hamiltonian of size Norbs * Nt

    The total Hamiltonian is
    .. math::
        H = H_0 + H_1 (\exp(i*\Omega * t)+\exp(-i*\Omega * t))
        or H = H_0 + 2 * H_1 \cos(\Omega * t)

    INPUT
        Norbs : number of orbitals
        Nt    : number of Fourier components
        E0    : electric field amplitude
    """
    if method == 1:
        Norbs = H0.shape[-1]

        #print('transition dipoles \n', M)

        # dimensionality of the Floquet matrix
        NF = Norbs * Nt
        F = np.zeros((NF,NF), dtype=complex)

        N0 = -(Nt-1)/2 # starting point for Fourier components of time exp(-i n w t)

        idt = np.identity(Nt)
        idm = np.identity(Norbs)
        # construc the Floquet H for a general tight-binding Hamiltonian
        for n in range(Nt):
            for m in range(Nt):

                # atomic basis index
                for k in range(Norbs):
                    for l in range(Norbs):

                    # map the index i to double-index (n,k) n : time Fourier component
                    # with relationship for this :  Norbs * n + k = i

                        i = Norbs * n + k
                        j = Norbs * m + l
                        F[i,j] = HamiltonFT(H0, H1, n-m)[k,l] - (n + N0) \
                                * omega * idt[n,m] * idm[k,l]


        # for a two-state model

    #    for n in range(Nt):
    #        for m in range(Nt):
    #            F[n * Norbs, m * Norbs] = (N0 + n) * omega * delta(n,m)
    #            F[n * Norbs + 1, m * Norbs + 1] = (onsite1 + (N0+n) * omega) * delta(n,m)
    #            F[n * Norbs, m * Norbs + 1] = t * delta(n,m+1)
    #            F[n * Norbs + 1, m * Norbs] = t * delta(n,m-1)
        #print('\n Floquet matrix \n', F)

        # compute the eigenvalues of the Floquet Hamiltonian,
        eigvals, eigvecs = linalg.eigh(F)

        #print('Floquet quasienergies', eigvals)

        # specify a range to choose the quasienergies, choose the first BZ
        # [-hbar omega/2, hbar * omega/2]
        eigvals_subset = np.zeros(Norbs, dtype=complex)
        eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


        # check if the Floquet states is complete
        j = 0
        for i in range(NF):
            if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
                eigvals_subset[j] = eigvals[i]
                eigvecs_subset[:,j] = eigvecs[:,i]
                j += 1
        if j != Norbs:
            print("Error: Number of Floquet states {} is not equal to \
                the number of orbitals {} in the first BZ. \n".format(j, Norbs))
            sys.exit()


        # now we have a complete linear independent set of solutions for the time-dependent problem
        # to compute the coefficients before each Floquet state if we start with |alpha>
        # At time t = 0, constuct the overlap matrix between Floquet modes and system eigenstates
        # G[j,i] = < system eigenstate j | Floquet state i >
        G = np.zeros((Norbs,Norbs), dtype=complex)
        for i in range(Norbs):
            for j in range(Norbs):
                tmp = 0.0
                for m in range(Nt):
                    tmp += eigvecs_subset[m * Norbs + j, i]
                G[j,i] = tmp


        # to plot G on site basis, transform it to site-basis representation
        Gsite = eigvecs_subset.dot(G)

        return eigvals_subset, eigvecs_subset, G
    elif method == 2:
            # Use Diagonalization Propagator method
            time_step = 5000
            dt = 2 * np.pi / (time_step * omega)  # Time step for propagator
            U = np.eye(H0.shape[0], dtype=complex)  # Initialize the propagator
            for t in range(time_step):
                time = t * dt
                H_t = H0 + H1 * (np.exp(1j*omega * time)+np.exp(-1j*omega * time))
                U = linalg.expm(-1j * H_t * dt) @ U  # Update the propagator

            # Diagonalize the propagator to get quasi-energies and modes
            eigvals, eigvecs = np.linalg.eig(U)
            quasi_energies = np.angle(eigvals) * omega / (2 * np.pi)
                    # Compute the overlap matrix G
            Norbs = H0.shape[0]
            G = np.zeros((Norbs, Norbs), dtype=complex)
            for i in range(Norbs):
                for j in range(Norbs):
                    G[j, i] = np.sum(eigvecs[:, i].conjugate() * eigvecs[:, j])

            # Transform G to the site basis
            Gsite = eigvecs.dot(G)
            return quasi_energies, eigvecs, Gsite

    else:
        raise ValueError(f"Method {method} not recognized. Use 1 for Floquet or 2 for Diagonalization_Propagator.")

def HamiltonFT(H0, H1, n):
    """
    Fourier transform of the Hamiltonian matrix, required to construct the
    Floquet Hamiltonian

    INPUT
        n : Fourier component index
        M : dipole matrix
    """
    Norbs = H0.shape[-1]

    if n == 0:
        return H0
    
    elif n == 1:
        return H1

    elif n == -1:
        return dag(H1)

    else:
        return np.zeros((Norbs,Norbs))


def Floquet_Winding_number(H0, H1, Nt, omega, method=1):
    """
    Unfold the Floquet band to aligned phase and calculate its winding number, used for 1D systems

    The total Hamiltonian is
    .. math::
        H = H_0 + H_1 (\exp(i*\Omega * t)+\exp(-i*\Omega * t))
        or H = H_0 + 2 * H_1 \cos(\Omega * t)

    INPUT
        Norbs : number of orbitals
        Nt    : number of Fourier components
        E0    : electric field amplitude
    """
    if method == 1:
        Norbs = H0.shape[-1]

        #print('transition dipoles \n', M)

        # dimensionality of the Floquet matrix
        NF = Norbs * Nt
        F = np.zeros((NF,NF), dtype=complex)

        N0 = -(Nt-1)/2 # starting point for Fourier components of time exp(-i n w t)

        idt = np.identity(Nt)
        idm = np.identity(Norbs)
        # construc the Floquet H for a general tight-binding Hamiltonian
        for n in range(Nt):
            for m in range(Nt):

                # atomic basis index
                for k in range(Norbs):
                    for l in range(Norbs):

                    # map the index i to double-index (n,k) n : time Fourier component
                    # with relationship for this :  Norbs * n + k = i

                        i = Norbs * n + k
                        j = Norbs * m + l
                        F[i,j] = HamiltonFT(H0, H1, n-m)[k,l] - (n + N0) \
                                * omega * idt[n,m] * idm[k,l]


        # for a two-state model

    #    for n in range(Nt):
    #        for m in range(Nt):
    #            F[n * Norbs, m * Norbs] = (N0 + n) * omega * delta(n,m)
    #            F[n * Norbs + 1, m * Norbs + 1] = (onsite1 + (N0+n) * omega) * delta(n,m)
    #            F[n * Norbs, m * Norbs + 1] = t * delta(n,m+1)
    #            F[n * Norbs + 1, m * Norbs] = t * delta(n,m-1)
        #print('\n Floquet matrix \n', F)

        # compute the eigenvalues of the Floquet Hamiltonian,
        eigvals, eigvecs = linalg.eigh(F)

        #print('Floquet quasienergies', eigvals)

        # specify a range to choose the quasienergies, choose the first BZ
        # [-hbar omega/2, hbar * omega/2]
        eigvals_subset = np.zeros(Norbs, dtype=complex)
        eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


        # check if the Floquet states is complete
        j = 0
        for i in range(NF):
            if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
                eigvals_subset[j] = eigvals[i]
                eigvecs_subset[:,j] = eigvecs[:,i]
                j += 1
        if j != Norbs:
            print("Error: Number of Floquet states {} is not equal to \
                the number of orbitals {} in the first BZ. \n".format(j, Norbs))
            sys.exit()


        # now we have a complete linear independent set of solutions for the time-dependent problem
        # to compute the coefficients before each Floquet state if we start with |alpha>
        # At time t = 0, constuct the overlap matrix between Floquet modes and system eigenstates
        # G[j,i] = < system eigenstate j | Floquet state i >
        G = np.zeros((Norbs,Norbs), dtype=complex)
        for i in range(Norbs):
            for j in range(Norbs):
                tmp = 0.0
                for m in range(Nt):
                    tmp += eigvecs_subset[m * Norbs + j, i]
                G[j,i] = tmp


        # to plot G on site basis, transform it to site-basis representation
        Gsite = eigvecs_subset.dot(G)

        return eigvals_subset, eigvecs_subset, G
    
    elif method == 2:
           print('Winding number is not available for Diagonalization Propagator method yet, use method 1.') 
           sys.exit()
            # # Use Diagonalization Propagator method
            # time_step = 5000
            # dt = 2 * np.pi / (time_step * omega)  # Time step for propagator
            # U = np.eye(H0.shape[0], dtype=complex)  # Initialize the propagator
            # for t in range(time_step):
            #     time = t * dt
            #     H_t = H0 + H1 * (np.exp(1j*omega * time)+np.exp(-1j*omega * time))
            #     U = linalg.expm(-1j * H_t * dt) @ U  # Update the propagator

            # # Diagonalize the propagator to get quasi-energies and modes
            # eigvals, eigvecs = np.linalg.eig(U)
            # quasi_energies = np.angle(eigvals) * omega / (2 * np.pi)
            #         # Compute the overlap matrix G
            # Norbs = H0.shape[0]
            # G = np.zeros((Norbs, Norbs), dtype=complex)
            # for i in range(Norbs):
            #     for j in range(Norbs):
            #         G[j, i] = np.sum(eigvecs[:, i].conjugate() * eigvecs[:, j])

            # # Transform G to the site basis
            # Gsite = eigvecs.dot(G)
            # return quasi_energies, eigvecs, Gsite

    else:
        raise ValueError(f"Method {method} not recognized. Use 1 for Floquet or 2 for Diagonalization_Propagator.")



if __name__ == '__main__':
    from pyqed import pauli
    s0, sx, sy, sz = pauli()
    
    mol = Floquet(H=0.5*sz, edip=sx)
    
    qe, fmodes = mol.spectrum(0.4, omegad = 1, nt=10, gauge='length')
    print(qe)
    # qe, fmodes = dmol.spectrum(E0=0.4, Nt=10, gauge='velocity')
    # print(qe)、、



