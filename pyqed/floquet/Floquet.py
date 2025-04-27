#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:53:56 2018

@author: binggu
"""

import numpy as np
import sys
from scipy import linalg
from scipy.special import jv
from pyqed.mol import Mol, dag
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# from tqdm import tqdm
import time
import os
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013

from numpy import exp, eye, zeros, arctan2
from scipy.linalg import eigh


class TightBinding(Mol):
    """
    1D tight-binding chain with Bloch periodic boundary conditions.

    Inherits from Mol so that FloquetBloch can be constructed.

    Parameters
    ----------
    h0 : array_like, shape (norb, norb)
        On-site Hamiltonian matrix for a single unit cell.
    hopping : array_like or float
        If float, uniform scalar hopping between cells. If array_like, matrix of shape (norb, norb)
        representing inter-cell hopping.
    a : float, optional
        Lattice constant (default = 1).
    nk : int, optional
        Number of k-points in the Brillouin zone for band structure.
    mu : float, optional
        On site potential (added to diagonal).
    """
    def __init__(self, h0, hopping=0.0, a=1.0, nk=50, mu=0.0):
        # Store parameters
        self.h0 = np.array(h0, dtype=complex)
        self.norb = self.h0.shape[0]
        # hopping matrix between unit cells (norb x norb)
        self.hop = (np.array(hopping, dtype=complex)
                    if np.shape(hopping) != () else np.eye(self.norb) * hopping)
        self.a = a
        self.mu = mu
        # precompute k-grid
        self.k_vals = np.linspace(-np.pi/a, np.pi/a, nk)
        # placeholders for results
        self._bands = None  # shape (norb, nk)

        # Initialize Mol parent with dummy edip (will override in Floquet)
        super().__init__(H=self.h0, edip=np.zeros_like(self.h0))

    def buildH(self, k):
        """
        Construct the Bloch Hamiltonian H(k) = h0 + hop*e^{i k a} + hop^dag e^{-i k a} + mu*I.

        Parameters
        ----------
        k : float
            Crystal momentum.

        Returns
        -------
        Hk : ndarray, shape (norb, norb)
        """
        phase = np.exp(1j * k * self.a)
        Hk = (self.h0
              + self.hop * phase
              + self.hop.conj().T * np.conj(phase)
              + np.eye(self.norb) * self.mu)
        return Hk

    def run(self, k=None):
        """
        Compute band eigenvalues at one or many k-points.

        Parameters
        ----------
        k : float or array_like, optional
            If None, uses the full internal k-grid. Otherwise compute only at specified k.

        Returns
        -------
        ks : ndarray
            k-points used.
        bands : ndarray, shape (norb, len(ks))
            Eigenvalues sorted ascending along axis 0.
        """
        ks = np.atleast_1d(k) if k is not None else self.k_vals
        bands = np.zeros((self.norb, len(ks)), dtype=float)
        for idx, kpt in enumerate(ks):
            eigs = np.linalg.eigvalsh(self.buildH(kpt))
            bands[:, idx] = np.sort(eigs.real)
        self._bands = bands
        return ks, bands

    def plot(self, k=None):
        """
        Plot the valence and conduction bands over k.
        """
        ks, bands = self.run(k)
        plt.figure(figsize=(8, 4))
        for b in range(self.norb):
            plt.plot(ks, bands[b], label=f'Band {b}')
        plt.xlabel('k')
        plt.ylabel('Energy')
        plt.title('Tight-Binding Band Structure')
        plt.legend()
        plt.grid(True)
        plt.show()
 
    def band_gap(self):
        """
        Compute the minimum gap between the lowest two bands over the k-grid.
        """
        if self._bands is None:
            _, self._bands = self.run()
        # gap = min_{k} [ band1(k) - band0(k) ]
        return np.min(self._bands[1] - self._bands[0])

    def Floquet(self, **kwargs):
        """
        Create a FloquetBloch solver for this TB model.

        Keyword Args
        ------------
        omegad : float
            Driving frequency.
        E0 : float
            Electric field amplitude.
        polarization : array_like of length 3
            Field polarization vector (unused for 1D).
        nt : int
            Number of Floquet harmonics.
        gauge : {'length','velocity'}
            Choice of gauge.

        Returns
        -------
        floq : FloquetBloch
        """
        # Overwrite parent H to be the static H0(k) function
        # We wrap buildH to supply H at each k inside FloquetBloch
        def Hk_func(kpt):
            return self.buildH(kpt)

        # dipole operator in Bloch basis: position * charge
        # for 1D, position operator acts as i d/dk in Bloch basis, but here use real-space site positions
        # approximate by diagonal positions [0, a, 2a,...]
        pos = np.diag(np.arange(self.norb) * self.a)

        floq = FloquetBloch(Hk=Hk_func,
                             Edip=pos,
                             **kwargs)
        return floq


    def density_of_states(self):
        pass

    def zeeman(self):
        # drive with magnetic field
        pass

    def gf(self):
        # surface and bulk Green function
        pass

    def gf_surface(self):
        pass


    def LvN(self, *args, **kwargs):
        # Liouville von Newnman equation solver
        pass
        # return LvN(*args, **kwargs)

class FloquetBloch:
    """
    peridically driven tight-binding system with a single frequency

    TODO: add more harmonics so it can treat a second harmonic driving
    """
    def __init__(self, H , omegad, E0, nt, edip=None):

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

    


    def phase_diagram(self, k_values, E0_values, omega_values, save_dir,
                        save_band_plot=True, nt=61, v=0.2, w=0.15):
        """
        Run Floquet winding number phase diagram over (E0, omega).

        Parameters:
        - k_values: np.array, k points along BZ
        - E0_values: 1D array of E0 field amplitudes
        - omega_values: 1D array of driving frequencies
        - save_dir: str, root directory for data and plots
        - save_band_plot: bool, whether to save band plots
        - nt, v, w: model and time-discretization parameters
        """
        os.makedirs(save_dir, exist_ok=True)
        data_dir = os.path.join(save_dir, "h5")
        band_dir = os.path.join(save_dir, "bands") if save_band_plot else None
        if band_dir: os.makedirs(band_dir, exist_ok=True)

        winding_real = np.zeros((len(E0_values), len(omega_values)))
        winding_int = np.zeros_like(winding_real)

        total = len(E0_values) * len(omega_values)
        start = time.time()
        idx = 0

        for j, omega in enumerate(omega_values):
            T = 2 * np.pi / omega
            pre_occ = pre_con = None
            for i, E0 in enumerate(E0_values):
                self.E0 = E0
                self.omegad = omega

                fname = os.path.join(data_dir, f"data_E0_{E0:.5f}_omega_{omega:.4f}.h5")

                occ, occ_e, con, con_e, draw = track_valence_band(
                    k_values, T, E0, omega, previous_val=pre_occ, previous_con=pre_con,
                    v=v, w=w, nt=nt, filename=fname
                )
                if draw and band_dir:
                    figure(occ_e, con_e, k_values, E0, omega, save_folder=band_dir)


                w_real = berry_phase_winding(k_values, occ)
                winding_real[i, j] = w_real
                winding_int[i, j] = round(w_real) if not np.isnan(w_real) else 0

                pre_occ, pre_con = occ, con

                idx += 1
                elapsed = time.time() - start
                remaining = elapsed / idx * (total - idx)
                progress = f"[{idx}/{total}]---------- E0={E0:.4f}, omega={omega:.4f} | Remaining: {remaining/60:.1f} min"
                print(progress, end="\r")

        # Save final phase diagram
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axs[0].imshow(winding_real, aspect='auto', origin='lower',
                            extent=[omega_values[0], omega_values[-1], E0_values[0], E0_values[-1]])
        axs[0].set_title("Winding (Real)"); axs[0].set_xlabel("omega"); axs[0].set_ylabel("E0")
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(winding_int, aspect='auto', origin='lower',
                            extent=[omega_values[0], omega_values[-1], E0_values[0], E0_values[-1]])
        axs[1].set_title("Winding (Integer)")
        fig.colorbar(im1, ax=axs[1])

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "winding_phase_diagram.png"))
        plt.close(fig)
        
        # return

    def run_phase_diagram_GL2013(self, k_vals, E0_over_omega_vals, b_vals, save_dir,
                             nt=61, t=1.5, omega=100, save_band_plot=False):


        os.makedirs(save_dir, exist_ok=True)
        band_dir = os.path.join(save_dir, "bands") if save_band_plot else None
        data_dir = os.path.join(save_dir, "h5")
        os.makedirs(data_dir, exist_ok=True)
        if save_band_plot:
            os.makedirs(band_dir, exist_ok=True)

        wind_real = np.zeros((len(E0_over_omega_vals), len(b_vals)))
        wind_int = np.zeros_like(wind_real)
        start = time.time()

        for j, b in enumerate(b_vals):
            pre_occ = pre_con = None
            for i, E0o in enumerate(E0_over_omega_vals):
                fname = os.path.join(data_dir, f"data_E0_over_omega_{E0o:.6f}_b_{b:.3f}_t_{t:.2f}.h5")
                occ, occ_e, con, con_e, draw = track_valence_band_GL2013(k_vals, E0o,
                                    previous_val=pre_occ, previous_con=pre_con,
                                    nt=nt, filename=fname, b=b, t=t, omega=omega)

                if draw and save_band_plot:
                    plt.figure(figsize=(8, 6))
                    plt.plot(k_vals, occ_e, label=f'Val_E0_over_omega = {E0o:.6f}, b = {b:.2f},t = {t:.2f}')
                    plt.plot(k_vals, con_e, label=f'Val_E0_over_omega = {E0o:.6f}, b = {b:.2f},t = {t:.2f}')
                    plt.title(f"$E_0/\omega$ = {E0o:.2f}, b = {b:.2f}")
                    plt.xlabel("k")
                    plt.ylabel("Quasienergy")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(band_dir, f"band_E0o_{E0o:.4f}_b_{b:.2f}.png"))
                    plt.close()

                w_real = berry_phase_winding(k_vals, occ)
                wind_real[i, j] = w_real
                wind_int[i, j] = round(w_real) if not np.isnan(w_real) else 0
                pre_occ, pre_con = occ, con

                elapsed = time.time() - start
                remaining = (len(E0_over_omega_vals) * len(b_vals) - (j * len(E0_over_omega_vals) + i)) * elapsed / (i + 1)
                print(f"[{i+1}/{len(E0_over_omega_vals)}], ---------- b = {b:.3f} E0/omega = {E0o:.3f} | Remaining: {remaining/60:.1f} min", end="\r")

        # Plot winding phase diagram
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        im0 = axs[0].imshow(wind_real, aspect='auto', origin='lower',
                            extent=[b_vals[0], b_vals[-1], E0_over_omega_vals[0], E0_over_omega_vals[-1]])
        axs[0].set_title("Winding (Real)")
        axs[0].set_xlabel("b")
        axs[0].set_ylabel("E0/ω")
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(wind_int, aspect='auto', origin='lower',
                            extent=[b_vals[0], b_vals[-1], E0_over_omega_vals[0], E0_over_omega_vals[-1]])
        axs[1].set_title("Winding (Integer)")
        axs[1].set_xlabel("b")
        axs[1].set_ylabel("E0/ω")
        fig.colorbar(im1, ax=axs[1])
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "winding_phase_diagram_GL2013.png"))
        plt.close()



    def winding_number(self, T, quasi_E = None, previous_state = None, gauge='length'): # To be modified
        H0 = self.H
        E0 = self.E0
        nt = self.nt 
        omegad = self.omegad
        nt = self.nt 
        omegad = self.omegad

        if gauge == 'length': # electric dipole

            H1 = -0.5 * self.edip * E0

        elif gauge == 'velocity':

            H1 = 0.5j * self.momentum() * E0/omegad
        
        elif gauge == 'peierls':
        
            occ_state, occ_state_energy = Floquet_Winding_number(H0, H1, nt, omegad, T, E0, quasi_E, previous_state)

        return occ_state, occ_state_energy
    
    def winding_number_Peierls(self, T, k, quasi_E = None, previous_state = None, gauge='length',w=0.2): # To be modified
        H0 = self.H
        E0 = self.E0
        nt = self.nt 
        omegad = self.omegad
        occ_state, occ_state_energy = Floquet_Winding_number_Peierls(H0, k, nt, omegad, T, E0, quasi_E, previous_state, w=w)
        return occ_state, occ_state_energy

    def winding_number_Peierls_GL2013(self, k, quasi_E = None, previous_state = None, gauge='length', b=0.5, t=1, E_over_omega = 1): # To be modified
        H0 = self.H
        E0 = self.E0
        nt = self.nt 
        omegad = self.omegad
        occ_state, occ_state_energy = Floquet_Winding_number_Peierls_GL2013(H0, k, nt, E_over_omega, quasi_E, previous_state, b=b, t=t)
        return occ_state, occ_state_energy

    def winding_number_Peierls_circular(self,k0, nt, omega, T, E0,
                                                    delta_x, delta_y,
                                                    a, t0, xi,
                                                    quasi_E,                 # or None
                                                    previous_state):
        H0 = self.H
        E0 = self.E0
        nt = self.nt 
        occ_state, occ_state_energy = Floquet_Winding_number_Peierls_circular(k0,                     # Bloch momentum
                                            nt, omega, T, E0,      # drive
                                            delta_x, delta_y,      # geometry
                                            a=a, t0=t0, xi=xi, # lattice & decay
                                            quasiE=quasi_E, previous_state=previous_state)
        return occ_state, occ_state_energy
    

    def velocity_to_length(self):
        # transform the truncated velocity gauge Hamiltonian to length gauge
        pass
    
    def propagator(self, t1, t2=0):
        pass


class FloquetGF(FloquetBloch):
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
        H0: time-independent part of the Hamiltonian
        H1: time-dependent part of the Hamiltonian
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




def HamiltonFT_peierls(Hn, n):
    """
    H(t) is SSH model with k replaced by k - E_0/w * sin(wt) by Peierls substitution
    Use Jacobi-Anger expansion to construct the Hamiltonian in extended space
    """

    Norbs = Hn[0].shape[-1]
    if n >= 0:
        return Hn[n]
    
    elif n < 0:
        return dag(Hn[n])

    else:
        return np.zeros((Norbs,Norbs))


def group_floquet_quasienergies(eigvals, eigvecs, omega=1.0, n_bands=2):
    """
    Identify 'n_bands' Floquet bands by clustering eigenvalues based on
    their fractional part mod '1'. Then sort each band and return
    the grouped eigenvalues/eigenvectors.

    Parameters
    ----------
    eigvals : array_like
        Floquet eigenvalues (length = n_bands * N_t for a 2-level system).
    eigvecs : ndarray
        Corresponding eigenvectors (shape = (N_F, N_F)), where columns
        match the order of 'eigvals'.
    omega : float
        Driving frequency (if ~1.0, we do mod 1).
    n_bands : int
        Number of bands to split into (2 for a two-level system).

    Returns
    -------
    band_vals : list of 1D arrays
        A list of length 'n_bands'; each entry is a sorted array of
        eigenvalues belonging to that band.
    band_vecs : list of 2D arrays
        A list of length 'n_bands'; each entry is a 2D array of the
        corresponding eigenvectors (columns match the sorted eigenvalues).
    """
    eigvals = np.asarray(eigvals)
    # Sort globally first
    idx_sort = np.argsort(eigvals)
    eigvals_sorted = eigvals[idx_sort]
    eigvecs_sorted = eigvecs[:, idx_sort]

    # Cluster the fractional parts mod 'omega' in 1D
    frac = np.mod(eigvals_sorted, 1)
    km = KMeans(n_clusters=n_bands, random_state=0).fit(frac.reshape(-1,1))
    labels = km.labels_

    band_vals = []
    band_vecs = []
    for band_idx in range(n_bands):
        # Extract all eigenvalues/vectors belonging to cluster band_idx
        these_vals = eigvals_sorted[labels == band_idx]
        these_vecs = eigvecs_sorted[:, labels == band_idx]
        # Sort them by ascending eigenvalue
        sub_idx = np.argsort(these_vals)
        these_vals = these_vals[sub_idx]
        these_vecs = these_vecs[:, sub_idx]
        band_vals.append(these_vals)
        band_vecs.append(these_vecs)
    return band_vals, band_vecs


def Floquet_Winding_number(H0, H1, Nt, omega, T, E ,quasiE = None, previous_state = None):
    """
    Build and diagonalize the Floquet Hamiltonian for a 1D system,
    then group the 2*N_t eigenvalues/eigenstates into two Floquet bands.
    choose the correct Floquet branch if E = 0 (by comparing with the directly diagonalized energies)
    if E != 0, choose the correct branch by doing overlap with the previous state.

    H(t) = H0 + 2*H1*cos(omega * t)
    """
    if E == 0:
        Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
        NF = Norbs * Nt           # dimension of Floquet matrix
        N0 = -(Nt-1)//2           # shift for Fourier indices

        # Construct the Floquet matrix
        F = np.zeros((NF, NF), dtype=complex)
        for n in range(Nt):
            for m in range(Nt):
                for k in range(Norbs):
                    for l in range(Norbs):
                        i = Norbs*n + k
                        j = Norbs*m + l
                        # Hamiltonian block + photon block
                        F[i, j] = (HamiltonFT(H0, H1, n-m)[k, l]
                                   - (n + N0)*omega*(n==m)*(k==l))

        # Diagonalize
        eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)
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
        eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
        eigvals_copy = np.array(eigvals_copy)
        idx = np.argsort(eigvals_copy.real)
        occ_state = eigvecs[:, idx[0]]
        occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
        return occ_state, occ_state_energy
    else:
        Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
        NF = Norbs * Nt           # dimension of Floquet matrix
        N0 = -(Nt-1)//2           # shift for Fourier indices

        # Construct the Floquet matrix
        F = np.zeros((NF, NF), dtype=complex)
        for n in range(Nt):
            for m in range(Nt):
                for k in range(Norbs):
                    for l in range(Norbs):
                        i = Norbs*n + k
                        j = Norbs*m + l
                        # Hamiltonian block + photon block
                        F[i, j] = (HamiltonFT(H0, H1, n-m)[k, l]
                                   - (n + N0)*omega*(n==m)*(k==l))

        # Diagonalize
        eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)s
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
        overlap = np.zeros(NF)
        for i in range(NF):
            for j in range(NF):
                overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
                # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
            # if np.abs(overlap[i]) < 0.05:
            #     overlap[i]=0
            # else:
            #     print(overlap[i])
        idx = np.argsort(abs(overlap))

        occ_state = eigvecs[:,idx[-1]]
        occ_state_energy = eigvals[idx[-1]]
        # for i in range(len(overlap)):
        #     # occ_state += eigvecs[:,i]*overlap[i]
        #     occ_state_energy += eigvals[i] * overlap[i]**2
        # # occ_state /=np.linalg.norm(occ_state)
        
        return occ_state, occ_state_energy


def Floquet_Winding_number_Peierls(H0, k, Nt, omega, T, E ,quasiE = None, previous_state = None, w = 0.2):
    """
    Build and diagonalize the Floquet Hamiltonian for a 1D system,
    then group the 2*N_t eigenvalues/eigenstates into two Floquet bands.
    choose the correct Floquet branch if E = 0 (by comparing with the directly diagonalized energies)
    if E != 0, choose the correct branch by doing overlap with the previous state.

    H(t) is SSH model with k replaced by k - E_0/w * sin(wt) by Peierls substitution
    """
    a = 1 #lattice constant, need to be modifyed accordingly, later need to be included into the variables
    A = E /omega
    B = a* k /2
    if E == 0:
        Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
        NF = Norbs * Nt           # dimension of Floquet matrix
        N0 = -(Nt-1)//2           # shift for Fourier indices

        # Hn = [np.array([[0, 0], [0, 0]], dtype=complex) for a in range(Nt)]
        # Hn[0] = H0 + np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex)

        Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
        for i in range(Nt):
            Hn[i][0][1] *= jv(-i,A)
            Hn[i][1][0] *= jv(i,A)
        Hn[0] += H0

        # Construct the Floquet matrix
        # need to be modified.
        F = np.zeros((NF, NF), dtype=complex)
        for n in range(Nt):
            for m in range(Nt):
                for k in range(Norbs):
                    for l in range(Norbs):
                        i = Norbs*n + k
                        j = Norbs*m + l
                        # Hamiltonian block + photon block
                        F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
                                   - (n + N0)*omega*(n==m)*(k==l))

        # Diagonalize
        eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)
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
        eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
        eigvals_copy = np.array(eigvals_copy)
        idx = np.argsort(eigvals_copy.real)
        occ_state = eigvecs[:, idx[0]]
        occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
        return occ_state, occ_state_energy
    else:
        Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
        NF = Norbs * Nt           # dimension of Floquet matrix
        N0 = -(Nt-1)//2           # shift for Fourier indices
        Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
        for i in range(Nt):
            Hn[i][0][1] *= jv(-i,A)
            Hn[i][1][0] *= jv(i,A)
        Hn[0] += H0
        # Construct the Floquet matrix
        # need to be modified.
        F = np.zeros((NF, NF), dtype=complex)
        for n in range(Nt):
            for m in range(Nt):
                for k in range(Norbs):
                    for l in range(Norbs):
                        i = Norbs*n + k
                        j = Norbs*m + l
                        # Hamiltonian block + photon block
                        F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
                                   - (n + N0)*omega*(n==m)*(k==l))

        # Diagonalize
        eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)s
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
        overlap = np.zeros(NF)
        for i in range(NF):
            for j in range(NF):
                overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
                # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
            # if np.abs(overlap[i]) < 0.05:
            #     overlap[i]=0
            # else:
            #     print(overlap[i])
        idx = np.argsort(abs(overlap))

        occ_state = eigvecs[:,idx[-1]]
        occ_state_energy = eigvals[idx[-1]]
        # for i in range(len(overlap)):
        #     # occ_state += eigvecs[:,i]*overlap[i]
        #     occ_state_energy += eigvals[i] * overlap[i]**2
        # # occ_state /=np.linalg.norm(occ_state)
        
        return occ_state, occ_state_energy
    
# ==============================================================
#  Circularly polarised Peierls helper  (δx,δy embedding)
# ==============================================================



def Floquet_Winding_number_Peierls_circular(
        k, Nt, omega, T, E0,                        # drive & grid
        delta_x, delta_y,                           # geometry
        a=1.0, t0=1.0, xi=1.0,                      # lattice/decay
        quasiE=None, previous_state=None):
    """
    Construct the extended Floquet matrix for a circularly polarised
    vector potential  A(t)=A0[cosΩt, sinΩt]  and return the valence
    Floquet state at momentum k.

    All array sizes are identical to those used in the linear routine,
    so the outer code stays unchanged.
    """
    Norbs = 2                   # SSH: 2 sites / unit cell
    NF    = Norbs * Nt
    N0    = -(Nt - 1) // 2      # Fourier index shift  (Nt must be odd)

    # ---- static hopping magnitudes  v0, w0  --------------------
    d_v   = (delta_x**2 + delta_y**2)**0.5
    d_w   = np.sqrt((a-delta_x)**2 + delta_y**2)
    v0    = t0 * exp(-d_v / xi)
    w0    = t0 * exp(-d_w / xi)

    # ---- bond angles & drive amplitude -------------------------
    theta_v = arctan2(delta_y,            delta_x)
    theta_w = arctan2(-delta_y,  a - delta_x)
    z_v     = (E0 / omega) * d_v          #  α|d|
    z_w     = (E0 / omega) * d_w

    # # Fourier coeffs  t^{(m)}  for m = N0 … N0+Nt-1
    # m_list  = np.arange(Nt) + N0
    # coeff_v = v0 * (-1j)**m_list * jv(m_list, z_v) * np.exp(-1j*m_list*theta_v)
    # coeff_w = w0 * (-1j)**m_list * jv(m_list, z_w) * np.exp(-1j*m_list*theta_w)
    # ---------- Fourier coefficients t^{(m)} ---------------------
    # need m = -(Nt-1) … +(Nt-1)  →  2*Nt-1 values
    
    #check this part
    m_all  = np.arange(-Nt+1, Nt)           # length 2*Nt-1
    coeff_v = v0 * (-1j)**(-m_all) * jv(-m_all, z_v) * np.exp(1j*m_all*theta_v)
    coeff_w = w0 * (-1j)**(-m_all) * jv(-m_all, z_w) * np.exp(1j*m_all*theta_w)


    # ---- build Floquet matrix  F  --------------------------------
    F = zeros((NF, NF), dtype=complex)
    for n in range(Nt):
        for m in range(Nt):
            mm = n - m                          # harmonic index
            # v_m = coeff_v[mm + (Nt-1)//2]
            # w_m = coeff_w[mm + (Nt-1)//2] * exp(-1j * k)
            v_m = coeff_v[mm + Nt - 1]
            w_m = coeff_w[mm + Nt - 1] * exp(-1j * k * a)

            block = zeros((2, 2), dtype=complex)
            block[0, 1] = v_m + w_m
            block[1, 0] = (v_m + w_m).conjugate()
            if n == m:
                block += eye(2) * (n + N0) * omega
            F[2*n:2*n+2, 2*m:2*m+2] = block

    # ---- diagonalise & pick valence branch -----------------------
    eigvals, eigvecs = eigh(F)
    # zone = np.logical_and(eigvals.real <=  0.5*omega,
    #                       eigvals.real >= -0.5*omega)
    # eps, vec = eigvals[zone], eigvecs[:, zone]
    eigvals_subset = np.zeros(Norbs, dtype=complex)
    eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)
    j=0
    for i in range(NF):
        if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
            eigvals_subset[j] = eigvals[i]
            eigvecs_subset[:,j] = eigvecs[:,i]
            j += 1
    if j != Norbs:
        print("Error: Number of Floquet states {} is not equal to \
            the number of orbitals {} in the first BZ. \n".format(j, Norbs))
        sys.exit()
    if E0 == 0:
        eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
        eigvals_copy = np.array(eigvals_copy)
        idx = np.argsort(eigvals_copy.real)
        occ_state = eigvecs[:, idx[0]]
        occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
        return occ_state, occ_state_energy
    else:
        overlap = np.zeros(NF)
        for i in range(NF):
            for j in range(NF):
                overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
                # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
            # if np.abs(overlap[i]) < 0.05:
            #     overlap[i]=0
            # else:
            #     print(overlap[i])
        idx = np.argsort(abs(overlap))

        occ_state = eigvecs[:,idx[-1]]
        occ_state_energy = eigvals[idx[-1]]
        return occ_state, occ_state_energy
        
    # if previous_state is None or quasiE is not None:
    #     idx = np.argmin(np.abs(eps.real - (quasiE if quasiE is not None else 0.0)))
    # else:
    #     olap = eigvecs.conj().T @ previous_state
    #     idx  = np.argmax(np.abs(olap))

    # return eigvecs[:, idx], eigvecs[idx].real


def Floquet_Winding_number_Peierls_GL2013(H0, k, Nt, E_over_omega ,quasiE = None, previous_state = None, b=0.5, t=1):
    """
    Build and diagonalize the Floquet Hamiltonian for a 1D system,
    then group the 2*N_t eigenvalues/eigenstates into two Floquet bands.
    choose the correct Floquet branch if E = 0 (by comparing with the directly diagonalized energies)
    if E != 0, choose the correct branch by doing overlap with the previous state.

    H(t) is SSH model with k replaced by k - E_0/w * sin(wt) by Peierls substitution
    """
    a = 1 #lattice constant, need to be modifyed accordingly, later need to be included into the variables
    A = E_over_omega
    omega = 100
    if E_over_omega == 0:
        Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
        NF = Norbs * Nt           # dimension of Floquet matrix
        N0 = -(Nt-1)//2           # shift for Fourier indices

        Hn = [np.array([[0, 0], [0, 0]], dtype=complex) for a in range(Nt)]
        # Hn[0] = H0 + np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex)

        # Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
        Hn_b = [np.array([[0, t], [t, 0]], dtype=complex) for a in range(Nt)]
        Hn_a_b = [np.array([[0, np.exp(1j*k)], [np.exp(-1j*k), 0]], dtype=complex) for a in range(Nt)]
        # Hn_b = [np.array([[0, t*np.exp(-1j*k*b)], [t*np.exp(1j*k*b), 0]], dtype=complex) for a in range(Nt)]
        # Hn_a_b = [np.array([[0, np.exp(1j*k*(1-b))], [np.exp(-1j*k*(1-b)), 0]], dtype=complex) for a in range(Nt)]

        for i in range(Nt):
            Hn[i][0][1] = Hn_b[i][0][1] * jv(-i,A*b) + Hn_a_b[i][0][1] * jv(i,A*(1-b)) 
            Hn[i][1][0] = Hn_b[i][1][0] * jv(i,A*b) + Hn_a_b[i][1][0] * jv(-i,A*(1-b))
        Hn[0] = H0
        # Construct the Floquet matrix
        # need to be modified.
        F = np.zeros((NF, NF), dtype=complex)
        for n in range(Nt):
            for m in range(Nt):
                for k in range(Norbs):
                    for l in range(Norbs):
                        i = Norbs*n + k
                        j = Norbs*m + l
                        # Hamiltonian block + photon block
                        F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
                                   - (n + N0)*omega*(n==m)*(k==l))

        # Diagonalize
        eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)
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
        eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
        eigvals_copy = np.array(eigvals_copy)
        idx = np.argsort(eigvals_copy.real)
        occ_state = eigvecs[:, idx[0]]
        occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
        return occ_state, occ_state_energy
    else:
        Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
        NF = Norbs * Nt           # dimension of Floquet matrix
        N0 = -(Nt-1)//2           # shift for Fourier indices
        Hn = [np.array([[0, 0], [0, 0]], dtype=complex) for a in range(Nt)]
        # Hn[0] = H0 + np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex)

        # Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
        Hn_b = [np.array([[0, t], [t, 0]], dtype=complex) for a in range(Nt)]
        Hn_a_b = [np.array([[0, np.exp(1j*k)], [np.exp(-1j*k), 0]], dtype=complex) for a in range(Nt)]
        for i in range(Nt):
            Hn[i][0][1] = Hn_b[i][0][1] * jv(-i,A*b) + Hn_a_b[i][0][1] * jv(i,A*(1-b)) 
            Hn[i][1][0] = Hn_b[i][1][0] * jv(i,A*b) + Hn_a_b[i][1][0] * jv(-i,A*(1-b))
        Hn[0] = H0
        # Construct the Floquet matrix
        # need to be modified.
        F = np.zeros((NF, NF), dtype=complex)
        for n in range(Nt):
            for m in range(Nt):
                for k in range(Norbs):
                    for l in range(Norbs):
                        i = Norbs*n + k
                        j = Norbs*m + l
                        # Hamiltonian block + photon block
                        F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
                                   - (n + N0)*omega*(n==m)*(k==l))

        # Diagonalize
        eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)s
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
        overlap = np.zeros(NF)
        for i in range(NF):
            for j in range(NF):
                overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
                # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
            # if np.abs(overlap[i]) < 0.05:
            #     overlap[i]=0
            # else:
            #     print(overlap[i])
        idx = np.argsort(abs(overlap))

        occ_state = eigvecs[:,idx[-1]]
        occ_state_energy = eigvals[idx[-1]]
        # for i in range(len(overlap)):
        #     # occ_state += eigvecs[:,i]*overlap[i]
        #     occ_state_energy += eigvals[i] * overlap[i]**2
        # # occ_state /=np.linalg.norm(occ_state)
        
        return occ_state, occ_state_energy

# def test():
#     """
#     Test of FB winding number solver aganist GL2024, PRL,

#     Returns
#     -------
#     None.

#     """
    
if __name__ == '__main__':
    # from pyqed import pauli
    # s0, sx, sy, sz = pauli()
    
    tb = TightBinding(norbs=2, coords=[[]])
    
    tb.band_gap()
    tb.band_structure(ks)
    
    
    floquet = tb.Floquet(omegad = 1, E = 1, polarization=[1, 0, 0])
    
    floquet.gauge = 'length'
    floquet.nt = 61

    
    floquet.winding_number(n=0)
    
    floquet.quasienergy()
    floquet.floquet_modes()
    
    
    # phase diagram 
    W = np.zeros((10, 10))
    for i, E in enumerate(np.linspace(0, 0.1, 10)):
        for j, omegad in enumerate(np.linspace(0, 1, 10)):
            floquet.set_amplitude(E)
            floquet.set_driving_frequency(omegad)
            
            W[i, j] = floquet.winding_number()
    
    # plot W
    
    # dynamics 
    
    
    # mol = Floquet(H=0.5*sz, edip=sx)    
    # qe, fmodes = mol.spectrum(0.4, omegad = 1, nt=10, gauge='length')


    # qe, fmodes = dmol.spectrum(E0=0.4, Nt=10, gauge='velocity')
    # print(qe)