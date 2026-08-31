#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 14:37:35 2026

@author: gugroup
"""

# from pyqed.nrg import star_to_chain
# from pyqed import pauli 

# class SBM:
#     """
#     spin-boson model
#     """
#     def __init__(self, epsilon, Delta, omegac=1):
#         """


#         Parameters
#         ----------
#         epsilon : TYPE
#             DESCRIPTION.
#         Delta : TYPE
#             DESCRIPTION.
#         omegac : TYPE, optional
#             cutoff frequency. The default is 1.

#         Returns
#         -------
#         None.

#         """

#         self.omegac = omegac

#         I, X, Y, Z = pauli()

#         self.H = 0.5 * (- epsilon * Z + X * Delta)

#     def spectral_density(self, s=1, alpha=1):
#         pass

#     def discretize(self):
#         pass

#     def to_wilson_chain(self):
#         pass

#     def HEOM(self):
#         pass

#     def Redfield(self):
#         pass
    
#     def DMRG(self):
#         # build MPO 
#         pass



from pyqed import pauli
import numpy as np
from pyqed.nrg.nrg import Boson
from pyqed.models.impurity.spin_boson import (
    SpinBosonWilsonChain,
    _default_op_quadrature_order,
    _normalize_chain_scheme,
    log_discretized_spin_boson_star_bath,
    spin_boson_spectral_density,
)
from pyqed.models.impurity.wilson import quadrature_star_bath, star_to_wilson_chain

from scipy import integrate
from scipy.sparse import lil_matrix, csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh

from pyqed import Cavity, Mol, Composite, dag, SineDVR, pauli, sort, householder

from pyqed.phys import eigh
from opt_einsum import contract
from scipy import sparse
        

#imports for mpo and dmrg calculation
import logging
from pyqed.mps import MPO, MPS
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSHO, BasisHalfSpin, BasisSpin
from pyqed.mps.autompo.light_automatic_mpo import Mpo
from pyqed.mps import DMRG as DMRG_Solver


class SBM:
    """
    spin-boson model for open quantum systems with NRG solver 

    .. math::

        H = -\Delta X + \epsilon Z + \sum_i \omega_i a_i^\dager a_i + Z \sum_i \lambda_i (a_i + a_i^\dagger)

    is mapped to

    .. math::

        H = -\Delta X + \epsilon Z + \sqrt{\eta_0/\pi} Z/2 (b_0+b_0^\dagger) +
        \sum_{n=0}^\infty \epsilon_n b_n^\dagger b_n + t_n(b_n b_{n+1}^\dagger + H.c.)

    where X, Y, Z are spin-half operators.

    The spectral density is defined as

    .. math::
        J(\omega) = \pi \sum_i \lambda_i^2 \delta(\omega - \omega_i)

    for `math`: \omega \in [0, \omega_c]


    """

    def __init__(
        self,
        Himp,
        alpha,
        L=2.0,
        s=1,
        omegac=1,
        epsilon=0,
        delta=0,
        nmodes=None,
        scheme="wilson",
        chain_method="lanczos",
    ):
        """


        Parameters
        ----------
        Himp : TYPE
            impurity Hamiltonian.
        L : TYPE, optional
            DESCRIPTION. The default is 2.0.
        s : TYPE, optional
            DESCRIPTION. The default is 1 corresponding to the Ohmic bath.
        omegac : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        # self.nsites = len(onsite) + 1

        # self.hopping = hopping
        self.L = L # Lambda for log-discretization
        self.Himp = Himp

        self.nmodes = None if nmodes is None else int(nmodes)
        assert(s > -1) # s has to be larger than -1
        self.s = s
        self.omegac = omegac
        self.alpha = alpha
        self.chain_scheme = _normalize_chain_scheme(scheme)
        self.chain_method = chain_method


        self.xi = None
        self.g = None
        ### Wilson chain params
        self.onsite = None
        self.hopping = None
        self.star_to_chain = None
        self.chain = None
        
        self.t0 = None

        ### constant parameters
        self.epsilon = epsilon
        self.delta = delta
        self.H = None
        self.support = None
        self.quadrature_order = None
        
    def add_coupling(self):
        pass

    def spectral_density(self, omega):
        """Return the spin-boson spectral density ``J(omega)``."""
        return spin_boson_spectral_density(
            omega,
            alpha=self.alpha,
            s=self.s,
            omegac=self.omegac,
        )

    def oscillator_energy(self, n):
        """
        n-th mode energy in the log-discretization (n = 0, 1, ...)

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        L = self.L
        s = self.s
        omegac = self.omegac

        return (s+1)/(s+2) * (1. - L**(-s-2))/(1. - L**(-s-1)) * omegac * L**(-n)


    def discretize(self, N=None, *, scheme=None, support=None, quadrature_order=None):
        # H = -\Delta X + \epsilon Z + \sum_i \xi_i a_i^\dagger a_i + \frac{Z}{2\sqrt{\pi}} \sum_i  \gamma_i (a_i + a_i^\dagger)
        """
        logrithmic discretization

        .. math::

            H = H_imp + \sqrt{\eta0/\pi} Z (b_0 + b_0^\dagger)

        Refs:
            PHYSICAL REVIEW B 71, 045122 s2005d

        Parameters
        ----------
        N : TYPE
            number of modes.
        s : TYPE, optional
            exponent in spectral density. The default is 1.
        omegac : TYPE, optional
            cutoff frequency. The default is 1.
        alpha : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        xi : TYPE
            DESCRIPTION.
        g : TYPE
            DESCRIPTION.

        """
        if N is None:
            N = self.nmodes
        if N is None:
            raise ValueError("N must be passed to discretize() or the constructor.")

        scheme = self.chain_scheme if scheme is None else _normalize_chain_scheme(scheme)
        self._set_star_bath(
            int(N),
            scheme=scheme,
            support=support,
            quadrature_order=quadrature_order,
        )
        self.to_chain(scheme=scheme, method=self.chain_method)

        # U = np.zeros((N, nmax))

        # U[0] = g/np.sqrt(eta0)

        # t = np.zeros(N) # hopping
        # epsilon = np.zeros(N, dtype=float) # onsite

        # epsilon[0] = sum(U[0]**2 * xi)

        # t[0] = sum( (xi - epsilon[0])**2 * g2 )/eta0
        # t[0] = np.sqrt(t[0])

        # U[1] = (xi - epsilon[0]) * U[0]/t[0]

        # for m in range(1, N-1):

        #     epsilon[m] = sum(U[m]**2 * xi)


        #     t[m] = np.sqrt( sum( ((xi - epsilon[m]) * U[m] -  t[m-1] * U[m-1] )**2) )


        #     U[m+1] = ((xi - epsilon[m]) * U[m] - t[m-1] * U[m-1])/t[m]
            
        #     print(sum(U[m]**2))


        # epsilon[N-1] = sum(U[N-1]**2 * xi)
        # t[N-1] = np.sqrt( sum( ((xi - epsilon[N-1])* U[N-1] -  t[N-2] * U[N-2])**2) )

        return self

    def _set_star_bath(self, N, *, scheme, support=None, quadrature_order=None):
        if scheme == "wilson":
            self.xi, self.g = log_discretized_spin_boson_star_bath(
                int(N),
                alpha=self.alpha,
                Lambda=self.L,
                s=self.s,
                omegac=self.omegac,
            )
            support = (0.0, self.omegac)
            quadrature_order = None
        elif scheme == "orthogonal-polynomial":
            support = (0.0, self.omegac) if support is None else tuple(map(float, support))
            quadrature_order = _default_op_quadrature_order(int(N), quadrature_order)
            self.xi, self.g = quadrature_star_bath(
                self.spectral_density,
                support,
                quadrature_order,
            )
        else:
            raise ValueError("scheme must be 'wilson' or 'orthogonal-polynomial'.")
        self.nmodes = int(N)
        self.chain_scheme = scheme
        self.support = support
        self.quadrature_order = quadrature_order
        self.chain = None

    def to_chain(
        self,
        *,
        scheme=None,
        N=None,
        nmodes=None,
        method=None,
        support=None,
        quadrature_order=None,
    ):
        """Return the chain representation, building star modes if needed."""
        scheme = self.chain_scheme if scheme is None else _normalize_chain_scheme(scheme)
        if nmodes is None:
            nmodes = N
        if nmodes is None:
            nmodes = self.nmodes
        requested_support = None if support is None else tuple(map(float, support))
        needs_star = self.xi is None or self.g is None or self.chain_scheme != scheme
        if nmodes is not None and self.nmodes != int(nmodes):
            needs_star = True
        if requested_support is not None and self.support != requested_support:
            needs_star = True
        if quadrature_order is not None and self.quadrature_order != int(quadrature_order):
            needs_star = True
        if needs_star:
            if nmodes is None:
                raise ValueError("N/nmodes must be passed to to_chain() or the constructor.")
            self._set_star_bath(
                int(nmodes),
                scheme=scheme,
                support=requested_support,
                quadrature_order=quadrature_order,
            )
        method = self.chain_method if method is None else method
        onsite, hopping, t0, transform = star_to_wilson_chain(
            self.xi,
            self.g,
            method=method,
        )
        self.onsite = onsite[: self.nmodes]
        self.hopping = hopping[: max(0, self.nmodes - 1)]
        self.t0 = t0
        self.star_to_chain = transform[: self.nmodes]
        self.chain_method = method
        self.chain = SpinBosonWilsonChain(
            onsite=self.onsite,
            hopping=self.hopping,
            impurity_coupling=t0,
            epsilon=self.epsilon,
            delta=self.delta,
            star_frequencies=self.xi,
            star_couplings=self.g,
            star_to_chain=self.star_to_chain,
        )
        return self.chain

    def to_wilson_chain(self, *, scheme=None, method=None, support=None, quadrature_order=None):
        """Alias for :meth:`to_chain`."""
        return self.to_chain(
            scheme=scheme,
            method=method,
            support=support,
            quadrature_order=quadrature_order,
        )

    def __iter__(self):
        if self.onsite is None or self.hopping is None:
            self.to_chain()
        return iter((self.onsite, self.hopping))

    def build_H_mpo(self, nb=6):
        
        N = self.nmodes
        # 1. Discretize
        if self.onsite is None or len(self.onsite) < N:
            self.discretize(N)
        
        epsilon_n = self.onsite
        t_n = self.hopping

        # 2. Define Basis
        basis = []
        basis.append(BasisHalfSpin(dof=0))  # Site 0: Impurity
        for n in range(N):
            basis.append(BasisSHO(dof=n+1, nbas=nb, omega=1.0)) # Site n+1: Bath

        # 3. Construct Hamiltonian Terms
        ham_terms = []
        
        # Impurity
        if abs(self.delta) > 1e-12:
            ham_terms.append(Op("sigma_x", 0, factor=-self.delta / 2.0))
        if abs(self.epsilon) > 1e-12:
            ham_terms.append(Op("sigma_z", 0, factor=self.epsilon / 2.0))

        # Coupling (Site 0 <-> Site 1)
        if abs(self.t0) > 1e-12:
            ham_terms.append(Op("sigma_z", 0) * Op("b", 1) * (self.t0 / 2.0))
            ham_terms.append(Op("sigma_z", 0) * Op(r"b^\dagger", 1) * (self.t0 / 2.0))

        # Bath Chain
        for n in range(N):
            site_idx = n + 1
            if abs(epsilon_n[n]) > 1e-12:
                ham_terms.append(Op(r"b^\dagger b", site_idx, factor=epsilon_n[n]))

            if n < N - 1:
                t_val = t_n[n]
                if abs(t_val) > 1e-12:
                    ham_terms.append(Op(r"b^\dagger", site_idx) * Op("b", site_idx + 1) * t_val)
                    ham_terms.append(Op("b", site_idx) * Op(r"b^\dagger", site_idx + 1) * t_val)

        # 4. Generate MPO
        model = Model(basis=basis, ham_terms=ham_terms)
        mpo_raw = Mpo(model, algo="qr")
        
        # Transpose MPO from (L, s, t, R) -> (L, R, s, t)
        # Expected by mps.py contract_from_right
        mpo_fixed = []
        for W in mpo_raw:
            # W shape is (Left, PhysOut, PhysIn, Right) -> (0, 1, 2, 3)
            # Target is (Left, Right, PhysOut, PhysIn) -> (0, 3, 1, 2)
            W_T = W.transpose(0, 3, 1, 2)
            mpo_fixed.append(W_T)
        
        self.H = MPO(mpo_fixed)
        return self.H
        

    def DMRG(self, nb=6, D=20, nsweeps=50):
        """
        Constructs the MPO for the Spin-Boson Model and initializes the DMRG solver.
        """
        if self.H is None:
            self.build_H_mpo()
            
        N = self.nmodes

        # 5. Create Initial Guess (Product State)
        init_mps = []

        # Site 0 (Spin): Shape (L, Phys, R) -> (1, 2, 1)
        A_spin = np.zeros((1, 2, 1))
        A_spin[0, 0, 0] = 1.0  # Up state
        init_mps.append(A_spin)

        # Sites 1..N (Boson): Shape (L, Phys, R) -> (1, nb, 1)
        for _ in range(N):
            A_boson = np.zeros((1, nb, 1))
            A_boson[0, 0, 0] = 1.0 # Vacuum state
            init_mps.append(A_boson)

        # 6. Return DMRG Object
        return DMRG_Solver(
            self.H,
            D=D,
            nsweeps=nsweeps,
            init_guess=MPS(init_mps, sites=self.H.input_sites),
            not_conv_err=True,
        )
        


    def exact_diag(self, N, nb):
        """
        Exact Diagonalization (ED) for benchmarking.
        !!!!Only run this for small N (e.g., N <= 6) and small nb. prevent large system since it would crash your computer. i won't be responsible for that!!!
        """
        if self.onsite is None or len(self.onsite) < N:
            self.discretize(N)
            
        epsilon_n = self.onsite
        t_n = self.hopping

        # define Local Operators
        # Spin operators (2x2)
        sx = sparse.csr_matrix([[0, 1], [1, 0]])
        sz = sparse.csr_matrix([[1, 0], [0, -1]])
        id_spin = sparse.eye(2)
        # Boson operators (nb x nb)
        diag_values = np.sqrt(np.arange(1, nb))
        b_mat = sparse.diags([diag_values], [1], shape=(nb, nb))
        
        bdag_mat = b_mat.T
        num_mat = bdag_mat @ b_mat
        id_b = sparse.eye(nb)
        # Helper to expand local operators to the full Hilbert Space
        # Hilbert Space Structure: Spin (x) Boson_0 (x) Boson_1 ... (x) Boson_{N-1}
        def expand_op(op, site_index):
            """
            site_index: -1 for Spin, 0 to N-1 for Bosons
            """
            # Start list of operators for kron
            ops_list = []
            
            # Spin part
            if site_index == -1:
                ops_list.append(op)
            else:
                ops_list.append(id_spin)
            
            # Boson parts
            for i in range(N):
                if i == site_index:
                    ops_list.append(op)
                else:
                    ops_list.append(id_b)
            
            # Compute Kronecker product
            # Note: sparse.kron can be slow if done sequentially without optimization, 
            # but for small N (ED) it is acceptable.
            full_op = ops_list[0]
            for next_op in ops_list[1:]:
                full_op = sparse.kron(full_op, next_op)
                
            return full_op

        # 4. Construct Hamiltonian Term by Term
        # H_imp = -Delta/2 * Sx + Epsilon/2 * Sz
        H_total = - (self.delta / 2.0) * expand_op(sx, -1) + \
                  (self.epsilon / 2.0) * expand_op(sz, -1)

        # H_coupling = t0/2 * Sz * (b0 + b0^dag)
        # Note: Sz acts on Spin, b0 on site 0. We multiply their expanded forms.
        if abs(self.t0) > 1e-12:
            op_sz_full = expand_op(sz, -1)
            op_x0_full = expand_op(b_mat + bdag_mat, 0)
            H_total += (self.t0 / 2.0) * (op_sz_full @ op_x0_full)

        # H_chain
        for n in range(N):
            # On-site energy: eps_n * b_n^dag * b_n
            if abs(epsilon_n[n]) > 1e-12:
                H_total += epsilon_n[n] * expand_op(num_mat, n)
            
            # Hopping: t_n * (b_n^dag * b_{n+1} + h.c.)
            if n < N - 1:
                t_val = t_n[n]
                if abs(t_val) > 1e-12:
                    # b_n^dag * b_{n+1}
                    hop = expand_op(bdag_mat, n) @ expand_op(b_mat, n+1)
                    # Add h.c.
                    H_total += t_val * (hop + hop.T.conj())

        # 5. Diagonalize
        # Calculate only the lowest eigenvalue (k=1)
        print(f"  [ED] Matrix shape: {H_total.shape}, Non-zeros: {H_total.nnz}")
        vals, vecs = eigsh(H_total, k=1, which='SA') # SA = Smallest Algebraic
        
        return vals[0]
    
    def TDDMRG(self, D=40, nb=6):
        from pyqed.mps import TDMPS 
        if self.H is None:
            self.build_H_mpo(nb=nb)
        return TDMPS(self.H, D)
    
    def build_product_state(self):
        """
        build a product state 
        .. math::
            |+> \otimes \prod_{i=0}^{L-1} |g_i\rangle

        Returns
        -------
        MPS

        """
        pass

    def run(self, N, nb=60, D=10, chain=True):
        """
        NRG solver for Wilson chain 

        Parameters
        ----------
        N : TYPE
            DESCRIPTION.
        nb : TYPE, optional
            DESCRIPTION. The default is 60.
        D: retained eigenstates

        Returns
        -------
        None.

        """


        I, X, Y, Z = pauli()

        # if self.onsite is None:
        #     epsilon, t = self.discretize(N)

        # else:

        epsilon = self.onsite
        t = self.hopping

        # N = self.nmodes

        L = self.L
        # impurity + the first boson site

        # nz = 60 # truncation in the Fock space

        e_tot = np.zeros((N, D))

        site = Boson(1, nb) # the 0th site
        Hb = site.buildH()

        a = site.annihilate()
        # ad = site.create()
        
        Isite = site.identity


        # x = dvr.x
        # dvr.v = x**2/

        if chain:
            
            # for n in range(nz):
            H = kron(self.H, eye(nb)) + kron(I, epsilon[0] * Hb)  + \
                self.t0 * np.sqrt(1/np.pi) * kron(Z/2., a + dag(a))
    
            E, U = eigh(H, k=D)
            E = E - E[0]
            
            
            e_tot[0] = E
            
            I = eye(D)
    
    
            a_tilde = dag(U) @ kron(eye(2), a) @ U
            # ad_tilde = dag(U) @ kron(eye(2), ad) @ U
            num =  U[:,0].conj().T @ kron(eye(2), Hb) @ U[:,0]
            
            Z_tilde = dag(U) @ kron(Z, eye(nb)) @ U
            


            
            for n in range(N-1):
    
    
                H = L * kron(np.diag(E), eye(nb)) + L**(n+1) * (\
                    kron(I, epsilon[n+1] * Hb) + t[n] * (kron(a_tilde, dag(a)) + \
                                                         kron(dag(a_tilde), a)))
    
                E, U = eigh(H, k=D)
                E = E - E[0]
    
                a_tilde = dag(U) @ kron(I, a) @ U
                Z_tilde = dag(U) @ kron(Z_tilde, site.identity) @ U
                
                # ad_tilde = dag(U) @ kron(I, ad) @ U
                
                # num =  contract('i,ij,j->', U[:,0].conj(), kron(I, dag(a) @ a), U[:,0])
                
                num =  U[:,0].conj().T @ kron(I, Hb) @ U[:,0]
                Sz = U[:,0].conj().T @ kron(Z_tilde, eye(nb)) @ U[:,0]
                
                print(num, Sz)
                
    
                e_tot[n+1] = E
    
            return e_tot
        
        else:
            # star 
            
            xi = self.xi 
            g = self.g 
            
            # for n in range(nz):
            H = kron(self.H, eye(nb)) + kron(I, xi[0] * Hb)  + \
                g[0] * np.sqrt(1/np.pi) * kron(Z/2., a + dag(a))
    
            E, U = eigh(H, k=D)
            E = E - E[0]
            
            e_tot[0] = E
            
            I = eye(D)
    
    
            a_tilde = dag(U) @ kron(eye(2), a) @ U
            # ad_tilde = dag(U) @ kron(eye(2), ad) @ U
            num =  U[:,0].conj().T @ kron(eye(2), Hb) @ U[:,0]
            
            Z_tilde = dag(U) @ kron(Z, eye(nb)) @ U
            
            for n in range(N-1):
    
    
                H = L * kron(np.diag(E), eye(nb)) + L**(n+1) * (\
                    kron(I, xi[n+1] * Hb) + g[n+1]/2./np.sqrt(np.pi) * (kron(Z_tilde, dag(a) + a)))
    
                E, U = eigh(H, k=D)
                E = E - E[0]
    

                
                # ad_tilde = dag(U) @ kron(I, ad) @ U
                
                # num =  contract('i,ij,j->', U[:,0].conj(), kron(I, dag(a) @ a), U[:,0])
                
                num =  U[:,0].conj().T @ kron(I, Hb) @ U[:,0]
                Sz = U[:,0].conj().T @ kron(Z_tilde, eye(nb)) @ U[:,0]
                
                a_tilde = dag(U) @ kron(I, a) @ U
                Z_tilde = dag(U) @ kron(Z_tilde, site.identity) @ U
                
                print(num, Sz)
                
    
                e_tot[n+1] = E
    
            return e_tot



    


if __name__=='__main__':
    

    I, X, Y, Z = pauli()
    epsilon = 0.1
    Delta = 0.01
    H = 0.5 * (epsilon * Z - X * Delta)
    nb = 5  #local bosonic dimension

    mol = SBM(H, s=0.8, L=2., alpha=0.05, omegac=1, epsilon=epsilon, 
              delta=Delta)

    eps, t = mol.discretize(N=10)
    dmrg = mol.DMRG(nb=nb, D=20)  
    # D defines maximum bond dimension, nsweep defines maximum dmrg scan rounds,\
    # default dmrg sweep number is 50.
    
    dmrg.run() # ground state stored in dmrg.ground_state as a MPS object

    
    # ######  get the rdm and how to use the rdm

    # ###### to get 1-rdm, call make_rdm, site idx to be calculated could be assigned, as integer, or as list, rdm will return as dictionary
    # # print(dmrg.make_rdm1(idx=0))
    # # print(dmrg.make_rdm1(idx=[0,1]))
    
    dm1 = dmrg.make_rdm1()
    print(dm1)
    
    # # then extract local rdm, for example spin site
    # rho_spin = dm1[0]
    
    # # you may get <sigma_z> from rdm on spin site
    # szAve = np.trace(rho_spin @ Z)
    # print(f"  <sigma_z> = {np.real(szAve):.8f}")

    # # you may get <b^dag b> from rdm on certain bosonic site
    # bath_sites = sorted([k for k in dm1.keys() if k > 0])

    # total_n = 0.0
    
    # for idx in bath_sites:
    #     rho_b = dm1[idx]
    #     n_op = np.diag(np.arange(nb))
        
    #     # Tr(rho * N)
    #     val_n = np.trace(rho_b @ n_op)
    #     val_n_real = np.real(val_n)
    #     total_n += val_n_real
        
    #     # check occupation probability of the highest state in local bosonic site
    #     # if this is too big, then you may need to increase nb
    #     max_fock_prob = np.real(rho_b[nb-1, nb-1])
        
    #     print(f"{idx:^6} | {val_n_real:^12.6f} | {max_fock_prob:^15.2e}")

    # print("-" * 38)
    # print(f"Total Bath Particle Number: {total_n:.6f}")


    # ###### to get 2-rdm, call make_rdm2, site idx to be calculated could be assigned, as list of turple
    # print(dmrg.make_rdm2(idx_pairs=[(0,1)]))
    # print(dmrg.make_rdm2(idx_pairs=[(0,1),(1,2)]))

    ##### Note, here in Spin Boson Model, site 0 and other sites have different local dimension, so assign index carefully 





# ########## below is benchmark test comparing with ED, validating the MPO construction for DMRG run, please keep system small
# if __name__=='__main__':
#     # Physics Parameters
#     epsilon = 0.4
#     Delta = 1
#     I, X, Y, Z = pauli()
#     H_imp = 0.5 * (epsilon * Z - X * Delta)

#     # Initialize Model
#     # s=0.8 (sub-ohmic), alpha=0.1 (coupling)
#     nrg = SBM(H_imp, s=0.8, L=2.0, alpha=0.1, omegac=1.0, epsilon=epsilon, delta=Delta)

#     # Define Benchmark Size (Small!)
#     N_bench = 5   # Only 4 bath sites
#     nb_bench = 4  # Only 4 basis states per boson
    
#     print(f"--- Benchmarking SBM (N={N_bench}, nb={nb_bench}) ---")

#     # Run ED
#     print("Running Exact Diagonalization...")
#     E_exact = nrg.exact_diag(N=N_bench, nb=nb_bench)
#     print(f"Exact Energy: {E_exact:.8f}")

#     # Run DMRG
#     # Note: Pass N and nb explicitly to ensure we simulate the EXACT same system
#     print("\nRunning DMRG...")
#     # Use high bond dimension D to ensure MPS error is negligible compared to N/nb choice
#     dmrg_sim = nrg.DMRG(nb=nb_bench, D=20, nsweeps=10).run()
    
#     # In mps.py, .run() stores the final energy in self.e_tot
#     E_dmrg = dmrg_sim.e_tot
#     print(f"DMRG Energy:  {E_dmrg:.8f}")

#     # Compare
#     diff = abs(E_exact - E_dmrg)
#     print(f"\nDifference:   {diff:.2e}")
