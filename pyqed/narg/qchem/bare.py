#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 10:18:41 2025

@author: Bing Gu (gubing@westlake.edu.cn)

NARG for fermionic chain models (e.g. Hubbard model, quantum chemistry)
"""


from pyqed.qchem.jordan_wigner.spinful import annihilate, create
from pyqed import dag, tensor, transform, expect, hadamard, pauli
from pyqed.mps.fermion import SpinHalfFermionChain

from pyqed import SpinHalfFermionOperators, eigh, sort
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, create

from scipy.sparse.linalg import eigsh
from scipy.sparse import kron, eye, csr_matrix, issparse

from opt_einsum import contract

import numpy as np

from pyqed import TFIM, multispin, transform
from pyqed.qchem import Molecule, build_atom_from_coords
from pyqed.phys import eigh
from pyqed.qchem.ci.fci import FCI
from pyqed.phys import obs, isdiag

import logging as log

from .active_space import CAS_OPTION_DEFAULTS, pop_active_space_options, prepare_active_space

import logging

# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)



# rotate the basis to the new representation

def rotate(A, B, U):
    """
    rotate :math:`V = A \otimes B` to the adiabatic representation
    :math:`|\phi_{\alpha_l n_{l+1}} \rangle \otimes | n_{l+1} \rangle

    Parameters
    ----------
    A : TYPE
        operator in the first l sites
    B : TYPE
        operator in the l+1 site
    U : ndarray [primitive basis index, adiabatic state index, (l+1)th site state index]
        DESCRIPTION.

    Returns
    -------
    v : TYPE
        DESCRIPTION.

    """
    n, D, d = U.shape

    assert(n == A.shape[0])
    assert(d == B.shape[1])

    if issparse(A):
        A = A.toarray()

    _v = contract('ibm, ij, jan -> mbna', U.conj(), A, U)
    v = contract('mbna, mn -> mbna', _v, B).reshape((d*D, d*D))
    return v




def zero_like_operator(reference):
    """Complex zero operator with the same shape/storage family as reference."""
    shape = reference.shape
    if issparse(reference):
        return csr_matrix(shape, dtype=complex)
    return np.zeros(shape, dtype=complex)





# logger = logging.getLogger('foo')
# log.setLevel(log.INFO)
# logger.info(f'active')

#### fermion chain
ops = SpinHalfFermionOperators()
cd = ops['Cd']
cu = ops['Cu']
cdu = ops['Cdu']
cdd = ops['Cdd']
JW = ops['JW']
Ntot = ops['Ntot']
Nu = ops['Nu']
Nd = ops['Nd']

# eigenstates of c^\dagger_\uparrow + c_\uparrow
# m = Cdu + Cdu

def atomic_chain(natom, z, element='H', basis='631g', spin=0):

    # ds = np.linspace(-4, 4, natom)

    elements = [element, ] * natom

    R = np.zeros((natom, 3))
    R[:, 2] = z

    atom = build_atom_from_coords(elements, R)

    mol = Molecule(
        atom = atom,
        basis = basis,
        unit = 'b',
        spin = spin,
        )

    return mol




def kernel(h1e, eri, D=20, n0=4, nstates=1, verbose=False):
    
    # C = mf.mo_coeff
    
    # h1e = mol.hcore
    
    # h1e = dag(C) @ h1e @ C
    
    # eri = mol.eri
    L = h1e.shape[-1]
    
    # # transform to MOs
    # eri = contract('ijkl, ip, jq, kr, ls -> pqrs', eri, C.conj(), C, C.conj(), C)
    v = eri
    
    # initiate the block with l0 Spin-Orbitals
    nstart = n0
    model = SpinHalfFermionChain(h1e[:nstart, :nstart], v[:nstart, :nstart, :nstart, :nstart],
                                 nelec=mol.nelec)
    # model.fix_nelec(s=2)
    
    model.jordan_wigner(forward=False)
    
    # D = 160 # retained adiabatic eigenstates
    E0, U0 = model.brute_force(nstates=D)
    
    # E0 = model.e_tot
    # U0 = model.X
    
    if verbose:
        print('Initial block energy = ', E0)
    
    H0 = model.H
    
    
    def single_site_hamiltonian(n):
        """
        Hamiltonian for a single spin-orbital
    
        Parameters
        ----------
        n : TYPE
            orbital ID. Starting from zero.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
    
        return h1e[n,n] * (cdu @ cu + cdd @ cd) + eri[n, n, n, n] * Nu @ Nd
    
    
    
    p = nstart
    # add the pth site
    h = single_site_hamiltonian(p)
    # assert(isdiag(h))
    if verbose:
        print('site', p, ' H = ', np.diag(h))
    
    # the adaibatic states at |\uparrow>
    # psi = np.array([0, 1., 0, 0])
    
    # nu = obs(psi, Nu) # expect 1
    # nd = obs(psi, Nd) # expect 0
    
    Cdu = model.Cdu
    Cdd = model.Cdd
    Cu = model.Cu
    Cd = model.Cd
    
    nu = 1
    nd = 0
    
    ### add all interaction between previous sites (0,1,...n-1) and the new site (n)
    
    # two-operator \sum_{i, j < p} v[i,j,p,p] - v[i, p, p, j] * (nu + nd)
    H = H0.copy()
    for i in range(nstart):
        for j in range(nstart):
            H += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
            H -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
    E1, U1 = eigh(H, k=D)
    # print(E1)
    
    # the adaibatic states at |\downarrow>
    nd = 1
    nu = 0
    
    H2 = H0.copy()
    for i in range(nstart):
        for j in range(nstart):
            H2 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
            H2 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
    E2, U2 = eigh(H2, k=D)
    # print(E2)
    
    # the adaibatic states at |\uparrow \downarrow>
    nu = 1
    nd = 1
    
    H3 = H0.copy()
    for i in range(nstart):
        for j in range(nstart):
            H3 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
            H3 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
    E3, U3 = eigh(H3, k=D)
    # print(E3)
    
    
    d = 4 # local dim
    E = np.zeros((d, min(D, d**nstart)))
    U = np.zeros((d**nstart, min(D, d**nstart), d), dtype=np.result_type(U0, U1, U2, U3, complex))
    
    E[0, :] = E0 + h[0, 0]
    E[1, :] = E1 + h[1, 1]
    E[2, :] = E2 + h[2, 2]
    E[3, :] = E3 + h[3, 3]
    
    # print('E = ', E)
    
    U[:, :, 0] = U0
    U[:, :, 1] = U1
    U[:, :, 2] = U2
    U[:, :, 3] = U3
    
    
    # build total Hamiltonian for 123 + 4
    
    # adiabatic H + diagonal part of h4
    # S = contract('ibm,  ian -> mbna', U.conj(), U)
    
    # residual interactions including a_p, a_p^dag a_p a_p
    
    Htot = np.diag(E.reshape((D * d))).astype(complex)
    
    # c_p V1, V2, V3
    v1u = zero_like_operator(H0)
    v1d = zero_like_operator(H0)
    
    
    for i in range(nstart):
    
        v1u = v1u + h1e[i, p] * Cdu[i]
        v1d = v1d + h1e[i, p] * Cdd[i]
    
    # print('v1u', v1u, v1d)
    
    for i in range(nstart):
        for j in range(nstart):
            for k in range(nstart):
                v1u += eri[k,p,j,i] * Cdu[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
                v1d += eri[k,p,j,i] * Cdd[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
    
    # jw_string = tensor([JW, ] * n0)
    
    
    # V1u =  contract('ibm, ij, jan -> mbna', U.conj(), (v1u @ jw_string).toarray() , U)
    # V1d =  contract('ibm, ij, jan -> mbna', U.conj(), (v1d @ jw_string).toarray(), U)
    
    V1u =  contract('ibm, ij, jan -> mbna', U.conj(), v1u.toarray() , U)
    V1d =  contract('ibm, ij, jan -> mbna', U.conj(), v1d.toarray(), U)
    
    # Cdu_p = create(p, spin='up')[-1]
    # Cdd_p = create(p, spin='down')[-1]
    
    # print(cu, cd)
    
    V1 = contract('mbna, mn -> mbna', V1u, JW @ cu).reshape((d*D, d*D)) + \
        contract('mbna, mn -> mbna', V1d, JW @ cd).reshape((d*D, d*D))
    
    # print('V1', V1)
    
    Htot += V1 + dag(V1) # this is not correct? I have to consider the JW string for Cp!
    
    # V2 term
    v2a = zero_like_operator(H0)
    for i in range(nstart):
        for j in range(nstart):
            v2a += -eri[i, p, p, j] * Cdd[i] @ Cu[j]
    
    v2b = zero_like_operator(H0)
    for i in range(n0):
        for j in range(n0):
            v2b += 0.5 * eri[p,i,p,j] * (Cd[i] @ Cu[j] - Cu[i] @ Cd[j])
    
    # print(dag(U) @ (Cdd+ Cd) @ U)
    
    V2 = contract('ibm, ij, jan -> mbna', U.conj(), v2a.toarray(), U)
    H2a = contract('mbna, mn -> mbna', V2, cdu @ cd).reshape((d*D, d*D))
    
    V2b = contract('ibm, ij, jan -> mbna', U.conj(), v2b.toarray(), U)
    H2b = contract('mbna, mn -> mbna', V2b, cdu @ cdd).reshape((d*D, d*D))
    
    # print('V2', H2a, H2b)
    Htot += H2a + dag(H2a) + H2b + dag(H2b)
    
    
    ## V3 (V3 can be combined with V1)
    v3u = zero_like_operator(H0)
    v3d = zero_like_operator(H0)
    for i in range(n0):
        v3u += eri[i, p, p, p] * Cdu[i]
        v3d += eri[i, p, p, p] * Cdd[i]
    
    V3u =  contract('ibm, ij, jan -> mbna', U.conj(), v3u.toarray(), U)
    V3d =  contract('ibm, ij, jan -> mbna', U.conj(), v3d.toarray(), U)
    
    # Cu_p = annihilate(p, spin='up')[-1]
    # Cd_p = annihilate(p, spin='down')[-1]
    
    
    H3 = contract('mbna, mn -> mbna', V3u, JW @ Nd @ cu).reshape((d*D, d*D)) + \
        contract('mbna, mn -> mbna', V3d, JW @ Nu @ cd).reshape((d*D, d*D))
    
    Htot += H3 + dag(H3)
    
    # print(Htot)
    # nroots = 10
    
    ######################
    # add the next orbital
    ######################
    
    # E0, U0 = eigsh(Htot, k=D, which='SA')
    
    # # print(E0)
    # log.info('\nTotal energy for {} orbitals = {}'.format(p+1, E0 + mol.energy_nuc()))
    
    
    
    # # l = nstart + 1
    # p += 1 # site id for the new orbital
    # print('\n--- adding the {}th orbital ---'.format(p+1))
    # print('p = ', p)
    
    # # the annihilation are operators \sigma_i Z_{i+1}......Z_l
    
    # H0 = Htot.copy()
    
    # Iblock = eye(d**n0) # block identity
    # Isite = eye(d) # site identity
    
    
    # Cu = [rotate(op, JW, U) for op in Cu] + [rotate(Iblock,  cu, U)]
    # Cd = [rotate(op, JW, U) for op in Cd] + [rotate(Iblock, cd, U)]
    # Cdu = [rotate(op, JW, U) for op in Cdu] + [rotate(Iblock, cdu, U)]
    # Cdd = [rotate(op, JW, U) for op in Cdd] + [rotate(Iblock, cdd, U)]
    
    
    # # print('Cu', Cu)
    # ### add all interaction between previous sites (0,1,...p-1) and the new site (p)
    
    # nu = 1
    # nd = 0
    
    # # two-operator \sum_{i, j < p} v[i,j,p,p] - v[i, p, p, j] * (nu + nd)
    # H1 = H0.copy()
    # for i in range(p):
    #     for j in range(p):
    #         H1 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
    #         H1 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
    # E1, U1 = eigh(H1, k=D)
    # # print(E1)
    
    # # the adaibatic states at |\downarrow>
    # nu = 0
    # nd = 1
    
    # H2 = H0.copy()
    # for i in range(p):
    #     for j in range(p):
    #         H2 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
    #         H2 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
    # E2, U2 = eigh(H2, k=D)
    # # print(r'adiabatic states corresponding to |\uparrow> = \n', E2)
    
    # # the adaibatic states at |\uparrow \downarrow>
    # nu = 1
    # nd = 1
    
    # H3 = H0.copy()
    # for i in range(p):
    #     for j in range(p):
    #         H3 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
    #         H3 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
    # E3, U3 = eigh(H3, k=D)
    # # print(E3)
    
    
    # # build the total H for the superblock of l_0 + 1 + 1 sites
    
    
    # E = np.zeros((d, D))
    # U = np.zeros((D * d, D, d))
    
    # h = single_site_hamiltonian(p)
    # log.info('site', p, 'H = ', np.diag(h))
    
    # E[0, :] = E0 + h[0, 0]
    # E[1, :] = E1 + h[1, 1]
    # E[2, :] = E2 + h[2, 2]
    # E[3, :] = E3 + h[3, 3]
    
    # # print('Enew = ', E)
    
    # U[:, :, 0] = U0
    # U[:, :, 1] = U1
    # U[:, :, 2] = U2
    # U[:, :, 3] = U3
    
    # # add residual interactions including a_p, a_p^dag a_p a_p
    
    # Htot = np.diag(E.reshape((D * d)))
    
    # # c_p V1
    # v1u = 0
    # v1d = 0
    
    # for i in range(p):
    #     v1u += h1e[i, p] * Cdu[i]
    #     v1d += h1e[i, p] * Cdd[i]
    
    
    # for i in range(p):
    #     for j in range(p):
    #         for k in range(p):
    #             v1u += eri[k,p,j,i] * Cdu[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
    #             v1d += eri[k,p,j,i] * Cdd[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
    
    # # jw_string = tensor([JW, ] * n0)
    
    # # V1u =  contract('ibm, ij, jan -> mbna', U.conj(), v1u.toarray() , U)
    # # V1d =  contract('ibm, ij, jan -> mbna', U.conj(), v1d.toarray(), U)
    
    # V1 = rotate(v1u, JW @ cu, U) + rotate(v1d, JW @ cd, U)
    
    
    # Htot += V1 + dag(V1)
    
    # v2a = 0
    # for i in range(p):
    #     for j in range(p):
    #         v2a += -eri[i, p, p, j] * Cdd[i] @ Cu[j]
    
    # v2b = 0
    # for i in range(p):
    #     for j in range(p):
    #         v2b += 0.5 * eri[p,i,p,j] * (Cd[i] @ Cu[j] - Cu[i] @ Cd[j])
    
    
    # # V2 = contract('ibm, ij, jan -> mbna', U.conj(), v2a.toarray(), U)
    # # H2a = contract('mbna, mn -> mbna', V2, cdu @ cd).reshape((d*D, d*D))
    # V2a = rotate(v2a, cdu @ cd, U)
    # # V2b = contract('ibm, ij, jan -> mbna', U.conj(), v2b.toarray(), U)
    # # H2b = contract('mbna, mn -> mbna', V2b, cdu @ cdd).reshape((d*D, d*D))
    # V2b = rotate(v2b, cdu @ cdd, U)
    
    # V2 = V2a + V2b
    
    # Htot += V2 + dag(V2)
    
    
    # ## V3 (V3 can be combined with V1)
    # v3u = 0
    # v3d = 0
    # for i in range(p):
    #     v3u += eri[i, p, p, p] * Cdu[i]
    #     v3d += eri[i, p, p, p] * Cdd[i]
    
    # # V3u =  contract('ibm, ij, jan -> mbna', U.conj(), v3u.toarray(), U)
    # # V3d =  contract('ibm, ij, jan -> mbna', U.conj(), v3d.toarray(), U)
    
    # # H3 = contract('mbna, mn -> mbna', V3u, JW @ Nd @ cu).reshape((d*D, d*D)) + \
    # #     contract('mbna, mn -> mbna', V3d, JW @ Nu @ cd).reshape((d*D, d*D))
    
    # V3 = rotate(v3u, JW @ Nd @ cu, U) + rotate(v3d, JW @ Nu @ cd, U)
    # Htot += V3 + dag(V3)
    
    while p < L-1:
    
        E0, U0 = eigsh(Htot, k=D, which='SA')
    
        log.info('\nTotal energy for {} orbitals = {}'.format(p+1, E0 + mol.energy_nuc()))
    
        p += 1 # site id for the new orbital
        if verbose:
            print('\n--- adding the {}th orbital ---'.format(p+1))
        if verbose:
            print('p = ', p)
    
        # the annihilation are operators \sigma_i Z_{i+1}......Z_l
    
        H0 = Htot.copy()
    
        # Iblock = eye(d*D) # block identity
        Iblock = eye(Cu[0].shape[-1])
        Isite = eye(d) # site identity
    
    
        Cu = [rotate(op, JW, U) for op in Cu] + [rotate(Iblock,  cu, U)]
        Cd = [rotate(op, JW, U) for op in Cd] + [rotate(Iblock, cd, U)]
        Cdu = [rotate(op, JW, U) for op in Cdu] + [rotate(Iblock, cdu, U)]
        Cdd = [rotate(op, JW, U) for op in Cdd] + [rotate(Iblock, cdd, U)]
    
    
        # print('Cu', Cu)
        ### add all interaction between previous sites (0,1,...p-1) and the new site (p)
    
        nu = 1
        nd = 0
    
        # two-operator \sum_{i, j < p} v[i,j,p,p] - v[i, p, p, j] * (nu + nd)
        H1 = H0.copy()
        for i in range(p):
            for j in range(p):
                H1 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
                H1 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
        E1, U1 = eigh(H1, k=D)
        # print(E1)
    
        # the adaibatic states at |\downarrow>
        nu = 0
        nd = 1
    
        H2 = H0.copy()
        for i in range(p):
            for j in range(p):
                H2 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
                H2 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
        E2, U2 = eigh(H2, k=D)
        # print(r'adiabatic states corresponding to |\uparrow> = \n', E2)
    
        # the adaibatic states at |\uparrow \downarrow>
        nu = 1
        nd = 1
    
        H3 = H0.copy()
        for i in range(p):
            for j in range(p):
                H3 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
                H3 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])
    
        E3, U3 = eigh(H3, k=D)
        # print(E3)
    
        #########################
        # build the total H for the superblock of l_0 + 1 + 1 sites
        #########################
        # nstates = min(D, d**l)
    
    
        E = np.zeros((d, D))
        U = np.zeros((D * d, D, d), dtype=np.result_type(U0, U1, U2, U3, complex))
    
        h = single_site_hamiltonian(p)
        log.info('site', p, 'H = ', np.diag(h))
    
        E[0, :] = E0 + h[0, 0]
        E[1, :] = E1 + h[1, 1]
        E[2, :] = E2 + h[2, 2]
        E[3, :] = E3 + h[3, 3]
    
        # print('Enew = ', E)
    
        U[:, :, 0] = U0
        U[:, :, 1] = U1
        U[:, :, 2] = U2
        U[:, :, 3] = U3
    
        # add residual interactions including a_p, a_p^dag a_p a_p
    
        Htot = np.diag(E.reshape((D * d))).astype(complex)
    
        # c_p V1
        v1u = zero_like_operator(H0)
        v1d = zero_like_operator(H0)
    
        for i in range(p):
            v1u += h1e[i, p] * Cdu[i]
            v1d += h1e[i, p] * Cdd[i]
    
    
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    v1u += eri[k,p,j,i] * Cdu[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
                    v1d += eri[k,p,j,i] * Cdd[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
    
        # jw_string = tensor([JW, ] * n0)
    
        # V1u =  contract('ibm, ij, jan -> mbna', U.conj(), v1u.toarray() , U)
        # V1d =  contract('ibm, ij, jan -> mbna', U.conj(), v1d.toarray(), U)
    
        V1 = rotate(v1u, JW @ cu, U) + rotate(v1d, JW @ cd, U)
    
    
        Htot += V1 + dag(V1)
    
        v2a = zero_like_operator(H0)
        for i in range(p):
            for j in range(p):
                v2a += -eri[i, p, p, j] * Cdd[i] @ Cu[j]
    
        v2b = zero_like_operator(H0)
        for i in range(p):
            for j in range(p):
                v2b += 0.5 * eri[p,i,p,j] * (Cd[i] @ Cu[j] - Cu[i] @ Cd[j])
    
    
        # V2 = contract('ibm, ij, jan -> mbna', U.conj(), v2a.toarray(), U)
        # H2a = contract('mbna, mn -> mbna', V2, cdu @ cd).reshape((d*D, d*D))
        V2a = rotate(v2a, cdu @ cd, U)
        # V2b = contract('ibm, ij, jan -> mbna', U.conj(), v2b.toarray(), U)
        # H2b = contract('mbna, mn -> mbna', V2b, cdu @ cdd).reshape((d*D, d*D))
        V2b = rotate(v2b, cdu @ cdd, U)
    
        V2 = V2a + V2b
    
        Htot += V2 + dag(V2)
    
    
        ## V3 (V3 can be combined with V1)
        v3u = zero_like_operator(H0)
        v3d = zero_like_operator(H0)
        for i in range(p):
            v3u += eri[i, p, p, p] * Cdu[i]
            v3d += eri[i, p, p, p] * Cdd[i]
    
        # V3u =  contract('ibm, ij, jan -> mbna', U.conj(), v3u.toarray(), U)
        # V3d =  contract('ibm, ij, jan -> mbna', U.conj(), v3d.toarray(), U)
    
        # H3 = contract('mbna, mn -> mbna', V3u, JW @ Nd @ cu).reshape((d*D, d*D)) + \
        #     contract('mbna, mn -> mbna', V3d, JW @ Nu @ cd).reshape((d*D, d*D))
    
        V3 = rotate(v3u, JW @ Nd @ cu, U) + rotate(v3d, JW @ Nu @ cd, U)
        Htot += V3 + dag(V3)
    
    ###############################
    
    
    ### Final diagonalization
    
    
    # nroots = 20
    E, X = eigsh(Htot, k=nstates, which='SA')
    
    if verbose:
        print('NARG energy = ', E + mol.energy_nuc())
    
    return E + mol.energy_nuc(), X


class NARG:
    """Object API for the bare quantum-chemistry NARG driver.

    This backend does not block diagonalize by symmetry sectors.  Use
    :class:`AbelianNARG` or :class:`SU2NARG` for symmetry-adapted runs.
    """

    DEFAULT_OPTIONS = {
        "D": 20,
        "n0": 4,
        "nstates": 1,
        "verbose": False,
        **CAS_OPTION_DEFAULTS,
    }

    def __init__(self, mf, *, mol=None, h1e=None, eri=None, **options):
        self.mf = mf
        self.mol = mol if mol is not None else getattr(mf, "mol", None)
        self.h1e = h1e
        self.eri = eri
        self.options = dict(self.DEFAULT_OPTIONS)
        self.options.update(options)
        self.e_tot = None
        self.vectors = None
        self.result = None
        self.active_space = None
        self.ncas = None
        self.nelecas = None
        self.ncore = None
        self.mo_core = None
        self.mo_cas = None
        self.e_core = None

    def integrals(self):
        """Return MO one- and two-electron integrals for the wrapped mean field."""
        opts = dict(self.options)
        cas_options = pop_active_space_options(opts)
        h1e, eri, _, _ = prepare_active_space(
            self.mf,
            self.mol,
            h1e=self.h1e,
            eri=self.eri,
            **cas_options,
        )
        return h1e, eri

    def _set_active_space(self, active_space):
        self.active_space = active_space
        if active_space is None:
            self.ncas = self.nelecas = self.ncore = None
            self.mo_core = self.mo_cas = None
            self.e_core = None
            return
        self.ncas = active_space.ncas
        self.nelecas = active_space.nelecas
        self.ncore = active_space.ncore
        self.mo_core = active_space.mo_core
        self.mo_cas = active_space.mo_cas
        self.e_core = active_space.energy_core

    def run(self, **options):
        """Run bare NARG and return ``(e_tot, vectors)``."""
        opts = dict(self.options)
        opts.update(options)
        cas_options = pop_active_space_options(opts)
        h1e = opts.pop("h1e", None)
        eri = opts.pop("eri", None)

        global mol
        active_mol = opts.pop("mol", None)
        if active_mol is not None:
            self.mol = active_mol
        if self.mol is None:
            self.mol = getattr(self.mf, "mol", None)
        if self.mol is None:
            raise ValueError("NARG needs a Molecule; pass NARG(mf, mol=mol) or run(..., mol=mol).")

        h1e, eri, prepared_mol, active_space = prepare_active_space(
            self.mf,
            self.mol,
            h1e=h1e,
            eri=eri,
            **cas_options,
        )
        self.h1e = h1e
        self.eri = eri
        self.mol = prepared_mol
        self._set_active_space(active_space)
        mol = self.mol
        nsites = int(np.asarray(h1e).shape[-1])
        if int(opts.get("n0", self.DEFAULT_OPTIONS["n0"])) >= nsites:
            if nsites < 2:
                raise ValueError("QChem NARG needs at least two spatial orbitals.")
            opts["n0"] = nsites - 1

        self.result = kernel(h1e, eri, **opts)
        self.e_tot, self.vectors = self.result
        return self.result
