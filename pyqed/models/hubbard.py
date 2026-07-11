#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:58:13 2026

@author: gugroup
"""

from pyqed.mps.fermion import SpinHalfFermionChain, annihilate, create
from pyqed.mps.hubbard import FermiHubbard

from pyqed.qchem.mol import atomic_chain
import numpy as np
from pyqed import dag



class Hubbard(SpinHalfFermionChain):
    """
    exact diagonalization of spin-half Fermi-Hubbard model (with long-range interactions)
    by Jordan-Wigner transformation

    .. math::

        H = \sum_{<i,j>, i < j} - t (c_{i\sigma}^\dagger c_{j\sigma} + hc) +
                 U n_{i\alpha} n_{i\beta} - \mu (n_i^{tot})

    where <i,j> indicate nearest-neighbors on the chain.
    Electron interactions can be included in the Hamiltonian easily.
    """

    def __init__(self, t, U, nsites, filling=None, nelec=None, mu=0):
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

        self.operators = None # basic operators for a chain


    def exact_diag(self, nstates=1):
        """
        exact diag without using any symmetry
        
        Check the number of particles and spin.

        Parameters
        ----------
        nstates : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        E : TYPE
            DESCRIPTION.

        """

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

        # if self.mu:

        self.H = H - self.mu * (Na + Nb)
        # else:
        #     self.H = H

        # return H + (Na - nelec/2 * I) @ (Na - self.nelec/2 * I) + \
        #     (Nb - self.nelec/2 * I) @ (Nb - self.nelec/2 * I)

        return self.H

    def build_h_mpo(self):
        """

        the complete MPO of the Hamiltonian

        Returns
        -------
        None.

        """

        ops = SpinHalfFermionOperators()
        # JWu = ops['JWu']
        # JWd = ops['JWd']
        JW = ops['JW']
        Cu = ops['Cu']
        Cdu = ops['Cdu']
        Cd = ops['Cd']
        Cdd = ops['Cdd']
        Sz = ops['Sz']
        Nu = ops['Nu']
        Nd = ops['Nd']
        Ntot = ops['Ntot']

        L = self.L
        t = - self.t

        U = self.U
        mu = self.mu

        self.I = I = np.eye(4)
        self.Z = Z = np.zeros((4,4))

        W_first = np.array([[U * Nu @ Nd - mu * (Nu + Nd), -t * Cdu @ JW, -t * Cdd @ JW, t * Cu @ JW, t * Cd @ JW, I]])

        # print('w1',W_first)


        W = np.array([[I, Z, Z, Z, Z, Z],
         [Cu, Z, Z, Z, Z, Z],
         [Cd, Z, Z, Z, Z, Z],
         [Cdu, Z, Z, Z, Z, Z],
         [Cdd, Z, Z, Z, Z, Z],
         [U * Nu @ Nd - mu * (Nu + Nd), -t * Cdu @ JW, -t * Cdd @ JW, t * Cu @ JW, t * Cd @ JW, I]])
        W_last = np.array([[I], [Cu], [Cd], [Cdu], [Cdd], [U * Nu @ Nd - mu * (Nu + Nd)]])


        self.W_first = W_first

        self.W = W
        self.W_last = W_last
        # print('w_first',np.shape(W_first))
        # print('w',np.shape(W))

        if self.L >= 3:
            mpo = [self.W_first] + ([self.W] * (self.L-2)) + [self.W_last]
            # result = mpo[0]
            # for i in range(1,self.L):
            #     result = coarse_gain_MPO(result,mpo[i])   # translate MPO form into exact form

        elif self.L == 2:
            mpo = [self.W_first] + [self.W_last]
            # result = coarse_gain_MPO(mpo[0],mpo[1])

        else:
            print("L should be more than 2")
            # result = -1
            mpo = -1

        self.h_mpo = MPO(mpo)
        return

    def DMRG(self, D=10):

        self.build_h_mpo()

        return DMRG(self.h_mpo.cores, D)

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

from pyqed.units import au2ev, au2angstrom, au2fs, eV_per_angstrom, au2wavenumber


# SSH parameters (apart from U, which is not contained in SSH)
t = 2.5/au2ev
U = 0./au2ev 
K = 21/au2ev * au2angstrom**2 # spring constant
M = 1349.14 /au2ev / au2fs**2 * au2angstrom**2
a = 1.22/au2angstrom
alpha = 4.1 * eV_per_angstrom

omega = np.sqrt(K/M) 
print(omega * au2wavenumber)


model = Hubbard(t, U, nsites=8, nelec=8)
model.run(10)




# natom = 4
# z = np.linspace(-3, 3, natom)
# mol = atomic_chain(natom, z)
# mol.basis = 'sto6g'
# mol.build()
# # mf = scf.RHF(mol).run()

# print(type(mol.nelec))

# mf = mol.RHF().run()

# print('number of electrons', mol.nelec)
# print('number of orbs = ', mol.nao)

# # e, fcivec = pyscf.fci.FCI(mf).kernel(verbose=4)
# # print(e)
# # Ca = mf.mo_coeff[0ArithmeticError
# # n = Ca.shape[-1]

# # mo_coeff = mf.mo_coeff
# # get the two-electron integrals as a numpy array
# # eri = get_eri_mo(mol, mo_coeff)

# # n = mol.nao
# # Ca = mo_coeff

# # h1e = get_hcore_mo(mf)
# # eri = get_eri_mo(mf)

# # print(mol.nelec)
# # model = SpinHalfFermionChain(h1e, eri).run(3)


# h1e = mf.get_hcore_mo()
# eri = mf.get_eri_mo()


# model = SpinHalfFermionChain(h1e, eri, mol.nelec)
#    # model = SpinHalfFermionChain(h1e, eri)

# model.run(nstates=10)
#    # narg = NARG(h1e, eri, D=20)

#    # block = model.initialize()