#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:36:57 2024

@author: Bing Gu (gubing@westlake.edu.cn)

Jordan-Wigner transformation for spinful (spin-1/2) fermions

Refs:
    TenPy

    https://mareknarozniak.com/2020/10/14/jordan-wigner-transformation/
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:59:10 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""
import numpy as np



from scipy.sparse import identity, kron, csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm

from scipy.special import erf
# from scipy.constants import e, epsilon_0, hbar
import logging
# import warnings

from pyqed import discretize, sort, dag, tensor
from pyqed.mps.mps import DMRG
from pyqed.phys import eigh
from pyqed.davidson import davidson
from pyqed import au2ev, au2angstrom, obs
from pyqed.dvr import SineDVR
# from pyqed import scf

from pyqed.qchem.gto.rhf import make_rdm1, energy_elec

# from numba import vectorize, float64, jit
import sys
from opt_einsum import contract

def SpinHalfFermionOperators(filling=1.):
    d = 4
    states = ['empty', 'up', 'down', 'full']
    # 0) Build the operators.
    Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float64)
    Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float64)

    Nu = np.diag(Nu_diag)
    Nd = np.diag(Nd_diag)
    Ntot = np.diag(Nu_diag + Nd_diag)
    dN = np.diag(Nu_diag + Nd_diag - filling)
    NuNd = np.diag(Nu_diag * Nd_diag)
    JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
    JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
    JW = JWu * JWd  # (-1)^{Nu+Nd}


    Cu = np.zeros((d, d))
    Cu[0, 1] = Cu[2, 3] = 1
    Cdu = np.transpose(Cu)
    # For spin-down annihilation operator: include a Jordan-Wigner string JWu
    # this ensures that Cdu.Cd = - Cd.Cdu
    # c.f. the chapter on the Jordan-Wigner trafo in the userguide
    Cd_noJW = np.zeros((d, d))
    Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
    Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
    Cdd = np.transpose(Cd)

    # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
    # where S^gamma is the 2x2 matrix for spin-half
    Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
    Sp = np.dot(Cdu, Cd)
    Sm = np.dot(Cdd, Cu)
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)

    ops = dict(JW=JW, JWu=JWu, JWd=JWd,
               Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
               Nu=Nu, Nd=Nd, Ntot=Ntot, NuNd=NuNd, dN=dN,
               Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable
    return ops



ops = SpinHalfFermionOperators()
Cd = ops['Cd']
Cu = ops['Cu']
Cdu = ops['Cdu']
Cdd = ops['Cdd']
JW = ops['JW']
Ntot = ops['Ntot']
Nu = ops['Nu']
Nd = ops['Nd']

def Is(l):
    """
    list of identity matrices

    Parameters
    ----------
    l : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if l > 0:
        return [eye(4), ] * l
    else:
        return []

# def Sx(N, i):
#     return tensor(Is(i) + [sigmax()] + Is(N - i - 1))
# def Sy(N, i):
#     return tensor(Is(i) + [sigmay()] + Is(N - i - 1))
# def Sz(N, i):
#     return tensor(Is(i) + [sigmaz()] + Is(N - i - 1))


def annihilate(L, spin='up', forward=True):
    """
    electron annihilation operators under JW transformation for a fermion chain

    Parameters
    ----------
    L : TYPE
        DESCRIPTION.
    spin : TYPE, optional
        DESCRIPTION. The default is 'up'.
    forward: bool
        control JW string starts from the first or the last site. Default: True.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    a : list
        list of {c_i, i=1, 2, ..., L} operators.

    """

    assert(spin in ['up', 'down'])

    if forward:

        a = []

        if spin == 'up':

            for i in range(L):
                ai = [JW, ] * i +  [Cu] + Is(L - i -1)
                a.append(tensor(ai))
            return a

        elif spin == 'down':

            for i in range(L):
                ai = [JW, ] * i +  [Cd] + Is(L - i -1)
                a.append(tensor(ai))
            return a

        else:
            raise ValueError('Spin {} can only be up or down.')

    else:

        a = []

        if spin == 'up':

            for i in range(L):
                ai = Is(i) +  [Cu] + [JW, ] * (L - i -1)
                a.append(tensor(ai))
            return a

        elif spin == 'down':

            for i in range(L):
                ai = Is(i) +  [Cd] + [JW, ] * (L - i -1)
                a.append(tensor(ai))
            return a



def create(L, spin='up', forward=True):

    assert(spin in ['up', 'down'])

    if forward:

        a = []

        if spin == 'up':

            for i in range(L):
                ai = [JW, ] * i +  [Cdu] + Is(L - i -1)
                a.append(tensor(ai))
            return a

        elif spin == 'down':

            for i in range(L):
                ai = [JW, ] * i +  [Cdd] + Is(L - i -1)
                a.append(tensor(ai))
            return a

    else:

        a = []

        if spin == 'up':

            for i in range(L):
                ai = Is(i) +  [Cdu] + [JW, ] * (L - i -1)
                a.append(tensor(ai))
            return a

        elif spin == 'down':

            for i in range(L):
                ai = Is(i) +  [Cdd] + [JW, ] * (L - i -1)
                a.append(tensor(ai))
            return a




def number_operator(L, spin='up'):
    a = []

    if spin == 'up':

        for i in range(L):
            ai = [Is, ] * i +  [Nu] + Is(L - i -1)
            a.append(tensor(ai))
        return a

    elif spin == 'down':

        for i in range(L):
            ai = [Is, ] * i +  [Nd] + Is(L - i -1)
            a.append(tensor(ai))
        return a

    elif spin == 'tot':

        for i in range(L):
            ai = [JW, ] * i +  [Ntot] + Is(L - i -1)
            a.append(tensor(ai))
        return a

    else:
        raise ValueError('Spin {} can only be up, down or tot.')



def jordan_wigner_transform(j, L, coeff=1, hc=False):
    """
    ["JW", ..., "JW", "Cu",  "Id", ..., "Id"]    # for the annihilation operator spin-up
    ["JW", ..., "JW", "Cd",  "Id", ..., "Id"]    # for the annihilation operator spin-down
    ["JW", ..., "JW", "Cdu",  "Id", ..., "Id"]   # for the creation operator spin-up
    ["JW", ..., "JW", "Cdd",  "Id", ..., "Id"]   # for the creation operator spin-down

    Refs
        https://github.com/tenpy/tenpy/blob/main/doc/intro/JordanWigner.rst

    """
    return

def jordan_wigner_one_body(i, j, L, coeff=1, hc=True):
    """
    Compute JWT of :math:`h_{ij} \sum_\sigma c_{i\sigma} ^\dag c_{j\sigma} + H.c.`

    .. math::
        c_i^\dagger c_j \rightarrow \sigma_i^+ (-1)^{\sum i \le k < j n_k} \sigma_j^-

    For exampe, for j = i + 2

    ["JW", ..., "JW", "Cd", "Id", "Id", "Id", ..., "Id"] * ["JW", ..., "JW", "JW", "JW", "C", "Id", ..., "Id"]
    == ["JW JW", ..., "JW JW", "Cd JW",  "Id JW", "Id C", ..., "Id"]
    == ["Id",    ..., "Id",    "Cd JW",  "JW",    "C",    ..., "Id"]


    if j = i - 2
    ["JW", ..., "JW", "JW", "JW", "Cd", "Id", ..., "Id"] * ["JW", ..., "JW", "C", "Id", "Id", "Id", ..., "Id"]
    == ["JW JW", ..., "JW JW", "JW C",  "JW", "Cd Id", ..., "Id"]
    == ["Id",    ..., "Id",    "JW C",  "JW", "Cd",    ..., "Id"]

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.
    j : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    assert  i <= j
    Hu = Is(i-1) +  [coeff * Cdu @ JW] + [JW, ] * (j-i-1) + [Cu] + Is(L-j-1)
    Hd = Is(i-1) +  [coeff * Cdd @ JW] + [JW, ] * (j-i-1) + [Cd] + Is(L-j-1)

    # if hc:
    #     Hu = Is(i-1) +  [coeff.conj() * Cdu @ JW] + Is(j-i-1) + [Cu] + Is(L-j-1)

    H = tensor(Hu) + tensor(Hd)

    if hc and i != j:
        return H + dag(H)
    else:
        return H

def jordan_wigner_two_body(i, j, k, l, coeff, hc=True):
    """

    .. math::
        c^\dag_i c^\dag_j c_k c_l

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.
    j : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    l : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """



# class SpinHalfFermionChain:

#     """
#     exact diagonalization of spin-half open fermion chain with long-range interactions

#     by Jordan-Wigner transformation

#     .. math::

#         H = \sum_{p, q} h_{pq} c_p^\dag c_q + 1/2 v_{pqrs} c^\dagger_p c^\dagger_q c_r c_s−2λ∑rc†rcr,

#     where r and s indicate neighbors on the chain.

#     Electron interactions can be included in the Hamiltonian easily.

#     """
#     def __init__(self, h1e, eri, nelec):
#         # if L is None:
#         L = h1e.shape[-1]
#         self.L = self.nsites = L

#         self.h1e = h1e
#         self.eri = eri
#         self.d = 4 # local dimension of each site
#         # self.filling = filling
#         self.nelec = nelec

#         self.H = None
#         self.Cu = None
#         self.Cd = None
#         self.Cdd = None
#         self.Cdu = None # C^\dag_\uparrow
#         self.Nu_tot = None
#         self.Nd_tot = None

#     def run(self, nstates=1):

#         # # single electron part
#         # Ca = mf.mo_coeff[:, :self.ncas]
#         # hcore_mo = contract('ia, ij, jb -> ab', Ca.conj(), mf.hcore, Ca)


#         # eri = self.mf.eri
#         # eri_mo = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

#         # # eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

#         # self.hcore_mo = hcore_mo

#         # H = self.jordan_wigner()
#         # self.H = H
#         E, X = eigh(self.H, k=nstates, which='SA')

#         # Nu = np.diag(dag(X) @ self.Nu_tot @ X)
#         # Nd = np.diag(dag(X) @ self.Nd_tot @ X)

#         # print('Energy  Nu     Nd')
#         # for i in range(nstates):
#         #     print(E[i], Nu[i], Nd[i])

#         return E, X


#     def build(self):
#         self.jordan_wigner()

#     def jordan_wigner(self, forward=True, aosym='8'):
#         """
#         MOs based on Restricted HF calculations

#         Returns
#         -------
#         H : TYPE
#             DESCRIPTION.
#         aosym: int, AO symmetry
#             8: eight-fold symmetry

#         """
#         h1e = self.h1e
#         v = self.eri

#         # an inefficient implementation without consdiering any syemmetry
#         # can be used to compute triplet states

#         nelec = self.nelec

#         norb = h1e.shape[-1]
#         nmo = L = norb # does not necesarrily have to MOs

#         Cu = annihilate(norb, spin='up', forward=forward)
#         Cd = annihilate(norb, spin='down', forward=forward)
#         Cdu = create(norb, spin='up', forward=forward)
#         Cdd = create(norb, spin='down', forward=forward)


#         self.Cu = Cu
#         self.Cd = Cd
#         self.Cdu = Cdu
#         self.Cdd = Cdd

#         H = 0
#         # for p in range(nmo):
#         #     for q in range(p+1):
#                 # H += jordan_wigner_one_body(q, p, hcore_mo[q, p], hc=True)
#         for p in range(nmo):
#             for q in range(nmo):
#                 H += h1e[p, q] * (Cdu[p] @ Cu[q] + Cdd[p] @ Cd[q])

#         # build total number operator
#         # number_operator = 0
#         Na = 0
#         Nb = 0
#         for p in range(L):
#             Na += Cdu[p] @ Cu[p]
#             Nb += Cdd[p] @ Cd[p]

#         self.Nu_tot = Na
#         self.Nd_tot = Nb


#         # poor man's implementation of JWT for 2e operators wihtout exploiting any symmetry
#         for p in range(nmo):
#             for q in range(nmo):
#                 for r in range(nmo):
#                     for s in range(nmo):
#                         H += 0.5 * v[p, q, r, s] * (\
#                             Cdu[p] @ Cdu[r] @ Cu[s] @ Cu[q] +\
#                             Cdu[p] @ Cdd[r] @ Cd[s] @ Cu[q] +\
#                             Cdd[p] @ Cdu[r] @ Cu[s] @ Cd[q] +
#                             Cdd[p] @ Cdd[r] @ Cd[s] @ Cd[q])
#                         # H += jordan_wigner_two_body(p, q, s, r, )

#         # digonal elements for p = q, r = s
#         self.H = H

#         return H

#     def fix_nelec(self, nelec=None, s=1):

#         if self.H is None:
#             self.build()

#         I = tensor(Is(self.L))

#         Na = self.Nu_tot
#         Nb = self.Nd_tot

#         if nelec is None:
#             nelec = self.nelec

#         self.H += s * (Na - nelec/2 * I) @ (Na - nelec/2 * I) + \
#                 s * (Nb - self.nelec/2 * I) @ (Nb - self.nelec/2 * I)
#         return

#     def DMRG(self):
#         # build the MPO of H and then apply the DMRG algorithm
#         pass
#         # return DMRG(H, D)

#     def gen_mps(self, state='random'):
#         if state == 'hf':
#             # create a HF MPS
#             pass

#     def spin_tot(self, psi):
#         pass