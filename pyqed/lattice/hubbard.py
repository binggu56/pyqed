#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:32:12 2025

Fermi-Hubbard and Bose-Hubbard models

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionChain, annihilate, create
from pyqed import dag, tensor, transform


from pyqed import SpinHalfFermionOperators, eigh


from scipy.sparse.linalg import eigsh
# from scipy.sparse import kron, eye, csr_matrix

from opt_einsum import contract

import numpy as np

from pyqed import TFIM, multispin, Molecule, build_atom_from_coords

# from pyqed.qchem.ci.fci import FCI
# from pyqed.phys import obs, isdiag

class FermiHubbard(SpinHalfFermionChain):
    """
    exact diagonalization of spin-half Fermi Hubbard model (with long-range interactions)
    by Jordan-Wigner transformation

    .. math::

        H = \sum_{<i,j>, i < j} - t (c_{i\sigma}^\dagger c_{j\sigma} + hc) +
                 U n_{i\alpha} n_{i\beta} - \mu (n_i^{tot})

    where <i,j> indicate nearest-neighbors on the chain.
    Electron interactions can be included in the Hamiltonian easily.
    """

    def __init__(self, t, U, nsites, filling=None, nelec=None, mu=None):
        """


        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        U : TYPE
            DESCRIPTION.
        nsites : TYPE
            DESCRIPTION.
        filling : TYPE, optional
            DESCRIPTION. The default is None.
        nelec : TYPE, optional
            DESCRIPTION. The default is None.
        mu : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.t = t # hopping
        self.U = U # Hubbard repulsion
        self.mu = mu

        self.L = self.nsites = nsites

        self.d = 4 # local dimension of each site

        self.filling = filling
        self.nelec = nelec

        ###
        self.H = None
        self.ntot = None

        self.eigvals = None # TBE
        self.e_tot = None
        self.eigvecs = None

    def run(self, nstates=1):

        # # single electron part
        # Ca = mf.mo_coeff[:, :self.ncas]
        # hcore_mo = contract('ia, ij, jb -> ab', Ca.conj(), mf.hcore, Ca)


        # eri = self.mf.eri
        # eri_mo = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

        # # eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

        # self.hcore_mo = hcore_mo

        self.jordan_wigner()

        E, X = eigsh(self.H, k=nstates, which='SA')

        self.e_tot = E
        self.eigvecs = X
        
        for i in range(nstates):
            print('Root', i, E[i])

        return E



    def jordan_wigner(self, forward=True, ao_symm=8):
        """
        apply JWT

        Returns
        -------
        H : TYPE
            DESCRIPTION.
        aosym: int, AO symmetry
            8: eight-fold symmetry for real-valued orbitals
            4: four-fold symmetry for complex-valued orbitals

        """

        # an inefficient implementation without consdiering any syemmetry
        # can be used to compute triplet states

        nelec = self.nelec

        nmo = norb = L = self.L
        t = self.t
        U = self.U

        Cu = annihilate(norb, spin='up', forward=forward)
        Cd = annihilate(norb, spin='down', forward=forward)
        Cdu = create(norb, spin='up', forward=forward)
        Cdd = create(norb, spin='down', forward=forward)

        self.Cu = Cu
        self.Cd = Cd
        self.Cdu = Cdu
        self.Cdd = Cdd

        H = 0
        # for p in range(nmo):
        #     for q in range(p+1):
                # H += jordan_wigner_one_body(q, p, hcore_mo[q, p], hc=True)
        for i in range(L-1):
                h = -t * (Cdu[i] @ Cu[i+1] + Cdd[i] @ Cd[i+1])
                H += h + dag(h)

        # build total number operator
        # number_operator = 0
        Na = 0
        Nb = 0
        for p in range(L):
            Na += Cdu[p] @ Cu[p]
            Nb += Cdd[p] @ Cd[p]


        # poor man's implementation of JWT for 2e operators wihtout exploiting any symmetry
        for i in range(L):
            H += U * Cdu[i] @ Cu[i] @ Cdd[i] @ Cd[i]

        # digonal elements for p = q, r = s
        # I = tensor(Is(L))

        self.ntot = [Na, Nb]

        if self.mu:

            self.H = H - self.mu * (Na + Nb)
        else:
            self.H = H

        # return H + (Na - nelec/2 * I) @ (Na - self.nelec/2 * I) + \
        #     (Nb - self.nelec/2 * I) @ (Nb - self.nelec/2 * I)

        return self.H        

    def DMRG(self):
        pass

    def number_operator(self, site_id, spin='up'):
        """
        number operator for each site

        Parameters
        ----------
        site_id : TYPE
            DESCRIPTION.
        spin : TYPE, optional
            DESCRIPTION. The default is 'up'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if spin == 'up':
            return self.Cdu[site_id] @ self.Cu[site_id]
        elif spin == 'down':
            return self.Cdd[site_id] @ self.Cd[site_id]
        if spin == 'tot':
            return self.Cdu[site_id] @ self.Cu[site_id] + self.Cdd[site_id] @ self.Cd[site_id]


    def gen_mps(self, state='random'):
        if state == 'hf':
            # create a HF MPS
            pass

    def spin_tot(self, psi):
        pass


class BoseHubbard():
    """
    Bose-Hubbard model

    .. math::
        H = -t\sum _{\left\langle i,j\right\rangle }\left({\hat {b}}_{i}^{\dagger }{\hat {b}}_{j}+{\hat {b}}_{j}^{\dagger }{\hat {b}}_{i}\right) + {\frac{U}{2}}\sum_{i} {\hat{n}}_{i} ({\hat{n}}_i -1) -\mu \sum_{i} {\hat{n}}_i
    """
    def __init__(self, t, U, nsites, filling=None, nelec=None, mu=None):
        self.t = t
        self.U = U
        self.nsites = nsites
        self.filling = filling
        self.nelec = nelec
        self.mu = mu

        self.H = None
        self.ntot = None

        self.eigvals = None
        self.eigvecs = None

    def buildH(self):
        pass

    def run(self, nstates=1):
        pass

    def DMRG(self):
        pass

    def NARG(self):
        pass

if __name__=='__main__':
    # pass

    hubbard = FermiHubbard(t=1, U=1, nsites=4)
    hubbard.run(2)