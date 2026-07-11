#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 10:18:41 2025

@author: Bing Gu (gubing@westlake.edu.cn)

NARG for fermionic chain models (e.g. Hubbard model, quantum chemistry)
"""


from pyqed.mps.fermion import SpinHalfFermionChain, annihilate, create
from pyqed import dag, tensor, transform, expect, hadamard, pauli


from pyqed import SpinHalfFermionOperators, eigh, sort, TFIM


from scipy.sparse.linalg import eigsh
from scipy.sparse import kron, eye

from opt_einsum import contract

import numpy as np

class SpinHalfFermionSite:
    pass

#### LDR for TFIM spin chain; but why would the first n sites follow the n + 1 site
#### unless the coupling is super strong?

# magnetic field
g = 1

### spin chains
I, X, Y, Z = pauli()

### H = - J (ZZ + gX)

D = 10
d = 2 # local dimension of a site
# step 1: solve the eigenstates of the first 2 sites
L = 12
nroots = 6




nstart = 6 # initial block
tfim = TFIM(1, g, nsites=nstart)

# H0 = - tensor(Z, Z, I, I) - tensor(I, Z, Z, I) - tensor(I, I, Z, Z) \
#     - g * (tensor(X, I, I, I) +  tensor(I, X, I, I) + tensor(I, I, X, I) + tensor(I, I, I, X))


# initialize

H0 = tfim.build()

E = np.zeros((2, min(D, d**nstart)))
U = np.zeros((d**nstart, min(D, d**nstart), 2))

for k, sz in enumerate([1, -1]):

    # the last term in the Hamiltonian comes from the - Z_2 Z_3
    # h = - tensor(Z, Z, I) - tensor(I, Z, Z) - g * (tensor(X, I, I) +  tensor(I, X, I) + tensor(I, I, X)) - sz * tensor(I, I, Z)
    h = H0 - sz * tensor([I, ] * (nstart - 1) + [Z])

    e, u =  eigh(h, min(D, h.shape[-1]))

    E[k, :] = e
    U[:, :, k] = u

letta = [U.copy()]

# total Hamiltonian for the superblock
S = contract('ibm,  ian -> mbna', U.conj(), U)
H = np.diag(E.reshape((D * 2)))  - g * contract('mbna, mn -> mbna', S, X).reshape((2*D, 2*D))



### add the rest sites
for l in range(L-nstart-1):
    E = np.zeros((2, D))
    U = np.zeros((2 * D, D, 2))
    for k, sz in enumerate([1, -1]):
        h = H - sz * kron(eye(D), Z)
        e, u =  eigh(h, D)
        E[k, :] = e
        U[:, :, k] = u

    S = contract('ibm,  ian -> mbna', U.conj(), U)
    H = np.diag(E.reshape((D * 2)))  - g * contract('mbna, mn -> mbna', S, X).reshape((2*D, 2*D))

    letta.append(U.copy())
    
### add site 6

# E = np.zeros((2, D))
# U = np.zeros((2 * D, D, 2))
# for k, sz in enumerate([1, -1]):
#     h = H - sz * kron(eye(D), Z)

#     e, u =  eigh(h, D)
#     E[k, :] = e
#     U[:, :, k] = u

# S = contract('ibm,  ian -> mbna', U.conj(), U)
# H = np.diag(E.reshape((D * 2)))  - g * contract('mbna, mn -> mbna', S, X).reshape((2*D, 2*D))


# final diagonalization
E, X =  eigh(H, k=nroots)

letta.append(X.copy())
print('TFIM sites', L)
print('NARG', E)


print([A.shape for A in letta])


# tfim = TFIM(1, g, nsites=L)
# tfim.run(nroots=nroots)

# print("Exact", tfim.e_tot)

# exact = [-25.39349675, -23.29495202, -23.02243305,  23.02243305,  23.29495202, 25.39349675]
# print('Error', (E - tfim.e_tot)/L)




# for 2D models, we are now adding one layer by one layer



#### fermion chain
# ops = SpinHalfFermionOperators()
# Cd = ops['Cd']
# Cu = ops['Cu']
# Cdu = ops['Cdu']
# Cdd = ops['Cdd']
# JW = ops['JW']
# Ntot = ops['Ntot']
# Nu = ops['Nu']
# Nd = ops['Nd']

# # eigenstates of c^\dagger_\uparrow + c_\uparrow
# m = Cdu + Cdu

# E, U = np.linalg.eigh(m)

# print(U)


# print(dag(U) @ (Cdd+ Cd) @ U)
# # print(E)


##### Variational LETTA for spin models 
from pyqed.models.heisenberg import Heisenberg
mol = Heisenberg(L=3)
neel = mol.build_neel_state()

H = mol.build_H_mpo()
W0, W1, W2 = H.factors # left, right, out, in 

print(W0.shape)

    


class FermiHubbard(SpinHalfFermionChain):
    """
    exact diagonalization of spin-half Fermi Hubbard model (with long-range interactions)
    by Jordan-Wigner transformation
    """
    # .. math::
    #     H = \sum_{<i,j>, i < j} - t (c_{i\sigma}^\dagger c_{j\sigma} + hc) +
    #         U n_{i\uparrow} n_{i\downarrow} - \mu (n_{i\uparrow} + n_{i\downarrow})
#
    # where <i,j> indicate nearest-neighbors on the chain.
    # Electron interactions can be included in the Hamiltonian easily.
    # """
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

        self.eigvals = None
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

        self.eigvals = E
        self.eigvecs = X

        return E, X



    def jordan_wigner(self):
        """
        apply JWT

        Returns
        -------
        H : TYPE
            DESCRIPTION.
        aosym: int, AO symmetry
            8: eight-fold symmetry

        """

        # an inefficient implementation without consdiering any syemmetry
        # can be used to compute triplet states

        nelec = self.nelec

        nmo = norb = L = self.L
        t = self.t
        U = self.U

        Cu = annihilate(norb, spin='up')
        Cd = annihilate(norb, spin='down')
        Cdu = create(norb, spin='up')
        Cdd = create(norb, spin='down')

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

if __name__=='__main__':
    pass



    # hubbard = FermiHubbard(t=1, U=1, nsites=4)

    # Us = np.linspace(0, 10, 80)
    # Vars = []
    # Es = []

    # for U in Us:

    #     hubbard.U = U

    #     E, X = hubbard.run(nstates=1)

    #     ave, var = expect(X[:,0], hubbard.number_operator(0), variance=True)
    #     Vars.append(var)
    #     Es.append(E)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(Us, Vars)

    # fig, ax = plt.subplots()
    # ax.plot(Us, Es)