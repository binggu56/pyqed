#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:59:10 2024

@author: Bing Gu (gubing@westlake.edu.cn)


# Simple DMRG tutorial.  This code integrates the following concepts:
#  - Infinite system algorithm
#  - Finite system algorithm
#
# Copyright 2013 James R. Garrison and Ryan V. Mishmash.
# Open source under the MIT license.  Source code at
# <https://github.com/simple-dmrg/simple-dmrg/>

# This code will run under any version of Python >= 2.6.  The following line
# provides consistency between python2 and python3.
from __future__ import print_function, division  # requires Python >= 2.6

# numpy and scipy imports


# We will use python's "namedtuple" to represent the Block and EnlargedBlock
# objects
from collections import namedtuple

"""
import numpy as np
import scipy.constants as const
# import scipy.linalg as la
# import scipy
from scipy.sparse.linalg import eigsh
from scipy.linalg import kron, norm, eigh

from scipy.special import erf
from scipy.constants import e, epsilon_0, hbar
import logging
import warnings

from pyqed import discretize, sort, dag, tensor
from pyqed.davidson import davidson
from pyqed.ldr.ldr import kinetic
from pyqed import au2ev, au2angstrom
from pyqed.dvr import SineDVR
# from pyqed import scf
from pyqed.qchem.gto.rhf import make_rdm1, energy_elec
# from pyqed.jordan_wigner import jordan_wigner_one_body, jordan_wigner_two_body

from pyqed.qchem.ci.fci import SpinOuterProduct, givenΛgetB

# from numba import vectorize, float64, jit
# import sys
from opt_einsum import contract

from collections import namedtuple

from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionOperators

from scipy.sparse import identity, kron, csr_matrix, diags

ops = SpinHalfFermionOperators()
Ntot = csr_matrix(ops["Ntot"])
Cdu = ops["Cdu"]
Cdd = ops["Cdd"]
Cu = ops["Cu"]
Cd = ops["Cd"]
JW = ops["JW"]

model_d = 4  # single-site basis size

class SpinHalfFermionChain:

    """
    exact diagonalization of spin-half open fermion chain with long-range interactions

    by Jordan-Wigner transformation

    .. math::

        H = \sum_{<rs>} (c_r^\dagger c_s + c†scr−γ(c†rc†s+cscr))−2λ \sum_r c^\dagger_r c_r,

    where r and s indicate neighbors on the chain.

    Electron interactions can be included in the Hamiltonian easily.

    """
    def __init__(self, h1e, eri, nelec):
        # if L is None:
        L = h1e.shape[-1]
        self.L = self.nsites = L

        self.h1e = h1e
        self.eri = eri
        self.d = 4 # local dimension of each site
        # self.filling = filling
        self.nelec = nelec

        self.H = None

    def run(self, nstates=1):

        # # single electron part
        # Ca = mf.mo_coeff[:, :self.ncas]
        # hcore_mo = contract('ia, ij, jb -> ab', Ca.conj(), mf.hcore, Ca)


        # eri = self.mf.eri
        # eri_mo = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

        # # eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

        # self.hcore_mo = hcore_mo

        H = self.jordan_wigner(self.h1e, self.eri)
        self.H = H
        E, X = eigsh(H, k=nstates, which='SA')

        return E, X



    def jordan_wigner(self, h1e, v):
        """
        MOs based on Restricted HF calculations

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        """
        # an inefficient implementation without consdiering any syemmetry

        from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body

        nelec = self.nelec

        norb = h1e.shape[-1]
        nmo = L = norb # does not necesarrily have to MOs


        Cu = annihilate(norb, spin='up')
        Cd = annihilate(norb, spin='down')
        Cdu = create(norb, spin='up')
        Cdd = create(norb, spin='down')

        H = 0
        # for p in range(nmo):
        #     for q in range(p+1):
                # H += jordan_wigner_one_body(q, p, hcore_mo[q, p], hc=True)
        for p in range(nmo):
            for q in range(nmo):
                H += h1e[p, q] * (Cdu[p] @ Cu[q] + Cdd[p] @ Cd[q])

        # build total number operator
        # number_operator = 0
        Na = 0
        Nb = 0
        for p in range(L):
            Na += Cdu[p] @ Cu[p]
            Nb += Cdd[p] @ Cd[p]


        # poor man's implementation of JWT for 2e operators wihtout exploiting any symmetry
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        H += 0.5 * v[p, q, r, s] * (\
                            Cdu[p] @ Cdu[r] @ Cu[s] @ Cu[q] +\
                            Cdu[p] @ Cdd[r] @ Cd[s] @ Cu[q] +\
                            Cdd[p] @ Cdu[r] @ Cu[s] @ Cd[q] +
                            Cdd[p] @ Cdd[r] @ Cd[s] @ Cd[q])
                        # H += jordan_wigner_two_body(p, q, s, r, )

        # digonal elements for p = q, r = s
        I = tensor(Is(L))

        return H + (Na - nelec/2 * I) @ (Na - self.nelec/2 * I) + \
            (Nb - self.nelec/2 * I) @ (Nb - self.nelec/2 * I)


    def DMRG(self):
        pass

    def gen_mps(self):
        pass




Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    """
    check the operators are of the right size

    Parameters
    ----------
    block : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True

# This function should test the same exact things, so there is no need to
# repeat its definition.
is_valid_enlarged_block = is_valid_block

def H2(Sz1, Sp1, Sz2, Sp2):  # two-site part of H
    """
    Given the operators S^z and S^+ on two sites in different Hilbert spaces
    (e.g. two blocks), returns a Kronecker product representing the
    corresponding two-site term in the Hamiltonian that joins the two sites.
    """
    J = Jz = 1.
    return (
        (J / 2) * (kron(Sp1, Sp2.conjugate().transpose()) +\
                   kron(Sp1.conjugate().transpose(), Sp2)) +
        Jz * kron(Sz1, Sz2)
    )


import scipy
# scipy.linalg.kron(a, b)
def enlarge_block(block, forward=True):
    """
    This function enlarges the provided Block by a single site, returning an
    EnlargedBlock.

    The connection operators need to be updated.
    """
    mblock = block.basis_size
    o = block.operator_dict

    # Create the new operators for the enlarged block.  Our basis becomes a
    # Kronecker product of the Block basis and the single-site basis.  NOTE:
    # `kron` uses the tensor product convention making blocks of the second
    # array scaled by the first.  As such, we adopt this convention for
    # Kronecker products throughout the code.
    l = block.length

    # if block.length == 1:
    #     enlarged_operator_dict = {
    #         "H": kron(o["H"], identity(4)) + kron(identity(mblock), H1[l+1]) \
    #             + H2(o["conn_Sz"], o["conn_Sp"], Sz1, Sp1),
    #         "conn_Sz": kron(identity(mblock), Sz1),
    #         "conn_Sp": kron(identity(mblock), Sp1),
    #     }
    # else:
    if forward:
        site_id = l


        H = kron(o["H"], identity(4)) + kron(identity(mblock), H1[site_id])

        for j in range(l):
            H += eri[j, site_id] * kron(o["Ntot"][j], Ntot)
            H += h1e[j, site_id] * (kron(o["Cdu"][j], Cu) + kron(o["Cdd"][j], Cd))

        Ntot_update = [kron(op, identity(4)) for op in o["Ntot"]] + [kron(identity(mblock), Ntot)]
        Cdu_update = [kron(op, JW) for op in o["Cdu"]] + [kron(identity(mblock), Cdu @ JW)]
        Cdd_update = [kron(op, JW) for op in o["Cdd"]] + [kron(identity(mblock), Cdd @ JW)]
        Cu_update = [kron(op, JW) for op in o["Cu"]] + [kron(identity(mblock), Cu @ JW)]
        Cd_update = [kron(op, JW) for op in o["Cd"]] + [kron(identity(mblock), Cd @ JW)]

    else:
        # env
        site_id = nsites - l - 1
        H = kron(o["H"], identity(4)) + kron(identity(mblock), H1[site_id])

        for j in range(l):
            H += eri[L - 1 - j, site_id] * kron(o["Ntot"][j], Ntot)
            H += h1e[L - 1 - j, site_id] * (kron(o["Cdu"][j], Cu) + kron(o["Cdd"][j], Cd))

    # H += H2(o["conn_Sz"], o["conn_Sp"], Sz1, Sp1)

        Ntot_update = [kron(op, identity(4)) for op in o["Ntot"]] + [kron(identity(mblock), Ntot)]
        Cdu_update = [kron(op, JW) for op in o["Cdu"]] + [kron(identity(mblock), Cdu)]
        Cdd_update = [kron(op, JW) for op in o["Cdd"]] + [kron(identity(mblock), Cdd)]
        Cu_update = [kron(op, JW) for op in o["Cu"]] + [kron(identity(mblock), Cu)]
        Cd_update = [kron(op, JW) for op in o["Cd"]] + [kron(identity(mblock), Cd)]



    enlarged_operator_dict = {
        "H": H,
        "Ntot": Ntot_update,
        "Cdu": Cdu_update,
        "Cdd": Cdd_update,
        "Cu": Cu_update,
        "Cd": Cd_update

    }


    return EnlargedBlock(length=(block.length + 1),
                         basis_size=(block.basis_size * model_d),
                         operator_dict=enlarged_operator_dict)

def rotate_and_truncate(operator, transformation_matrix):
    """Transforms the operator to the new (possibly truncated) basis given by
    `transformation_matrix`.
    """
    if isinstance(operator, list):
        return [transformation_matrix.conjugate().transpose() @ op @ transformation_matrix for op in operator]
    else:
        return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))

def single_dmrg_step(sys, env, m, forward=True):
    """
    Performs a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `m` states in the new basis.
    """
    # assert is_valid_block(sys)
    # assert is_valid_block(env)

    # Enlarge each block by a single site.
    sys_enl = enlarge_block(sys, forward)

    if sys is env:  # no need to recalculate a second time
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env, forward=not forward)

    # assert is_valid_enlarged_block(sys_enl)
    # assert is_valid_enlarged_block(env_enl)

    # Construct the full superblock Hamiltonian.
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict


    superblock_hamiltonian = kron(sys_enl_op["H"], identity(m_env_enl)) + \
                             kron(identity(m_sys_enl), env_enl_op["H"])

    if forward:
        for j in range(sys_enl.length):
            for k in range(env_enl.length):
                print(j, L-k-1, eri[j, L-k-1], h1e[j, L-k-1])
                superblock_hamiltonian += eri[j, L-k-1] * kron(sys_enl_op["Ntot"][j], env_enl_op["Ntot"][k])
                superblock_hamiltonian += h1e[j, L-k-1] * (kron(sys_enl_op["Cdu"][j], env_enl_op["Cu"][k]) +\
                                                kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k]) +\
                                                kron(sys_enl_op["Cu"][j], env_enl_op["Cdu"][k]) +\
                                                kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k])
                                                )
    else:
        for j in range(sys_enl.length):
            for k in range(env_enl.length):
                superblock_hamiltonian += eri[k, L-j-1] * kron(sys_enl_op["Ntot"][j], env_enl_op["Ntot"][k])
                superblock_hamiltonian += h1e[k, L-j-1] * (kron(sys_enl_op["Cdu"][j], env_enl_op["Cu"][k]) +\
                                                kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k]) +\
                                                kron(sys_enl_op["Cu"][j], env_enl_op["Cdu"][k]) +\
                                                kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k])
                                                )


    # Call ARPACK to find the superblock ground state.  ("SA" means find the
    # "smallest in amplitude" eigenvalue.)
    (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")

    # Construct the reduced density matrix of the system by tracing out the
    # environment
    #
    # We want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.  Since the environment (column) index updates most
    # quickly in our Kronecker product structure, psi0 is thus row-major ("C
    # style").


    psi0 = psi0.reshape([sys_enl.basis_size, -1], order="C")
    rho = np.dot(psi0, psi0.conjugate().transpose())
    rho_env = np.dot(psi0.conjugate().transpose(), psi0)

    # Diagonalize the reduced density matrix and sort the eigenvectors by
    # eigenvalue.
    evals, evecs = np.linalg.eigh(rho)
    possible_eigenstates = []
    for eval, evec in zip(evals, evecs.transpose()):
        possible_eigenstates.append((eval, evec))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    evals_env, evecs_env = np.linalg.eigh(rho_env)
    possible_eigenstates_env = []
    for eval, evec in zip(evals_env, evecs_env.transpose()):
        possible_eigenstates_env.append((eval, evec))
    possible_eigenstates_env.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first


    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec

    transformation_matrix_env = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates_env[:my_m]):
        transformation_matrix_env[:, i] = evec

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    print("truncation error:", truncation_error)

    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    new_env_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_env_operator_dict[name] = rotate_and_truncate(op, transformation_matrix_env)

    # for name in ["H", "Ntot"]:
    #     new_operator_dict[name] = rotate_and_truncate(sys_enl.operator_dict[name], transformation_matrix)

    # for name in ["Cu", "Cd", "Cdu", "Cdd"]:
    #     new_operator_dict[name] = [rotate_and_truncate(op, transformation_matrix)\
    #                                for op in sys_enl.operator_dict[name]]

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)

    newblock_env = Block(length=env_enl.length,
                     basis_size=my_m,
                     operator_dict=new_env_operator_dict)


    return newblock, newblock_env, energy

# def single_dmrg_step(sys, env, m):
#     """
#     Performs a single DMRG step using `sys` as the system and `env` as the
#     environment, keeping a maximum of `m` states in the new basis.
#     """
#     # assert is_valid_block(sys)
#     # assert is_valid_block(env)

#     # Enlarge each block by a single site.
#     sys_enl = enlarge_block(sys)

#     if sys is env:  # no need to recalculate a second time
#         env_enl = sys_enl
#     else:
#         env_enl = enlarge_block(env, direction='backward')

#     # assert is_valid_enlarged_block(sys_enl)
#     # assert is_valid_enlarged_block(env_enl)

#     # Construct the full superblock Hamiltonian.
#     m_sys_enl = sys_enl.basis_size
#     m_env_enl = env_enl.basis_size
#     sys_enl_op = sys_enl.operator_dict
#     env_enl_op = env_enl.operator_dict


#     superblock_hamiltonian = kron(sys_enl_op["H"], identity(m_env_enl)) + \
#                              kron(identity(m_sys_enl), env_enl_op["H"])

#     for j in range(sys.length):
#         for k in range(env.length):
#             superblock_hamiltonian += eri[j, k] * kron(sys_enl_op["Ntot"][j], env_enl_op["Ntot"][k])
#             superblock_hamiltonian += h1e[j, k] * (kron(sys_enl_op["Cdu"][j], env_enl_op["Cu"][k]) +\
#                                             kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k]) +\
#                                             kron(sys_enl_op["Cu"][j], env_enl_op["Cdu"][k]) +\
#                                             kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k])
#                                             )

#     # Call ARPACK to find the superblock ground state.  ("SA" means find the
#     # "smallest in amplitude" eigenvalue.)
#     (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")

#     # Construct the reduced density matrix of the system by tracing out the
#     # environment
#     #
#     # We want to make the (sys, env) indices correspond to (row, column) of a
#     # matrix, respectively.  Since the environment (column) index updates most
#     # quickly in our Kronecker product structure, psi0 is thus row-major ("C
#     # style").


#     psi0 = psi0.reshape([sys_enl.basis_size, -1], order="C")
#     rho = np.dot(psi0, psi0.conjugate().transpose())
#     rho_env = np.dot(psi0.conjugate().transpose(), psi0)

#     # Diagonalize the reduced density matrix and sort the eigenvectors by
#     # eigenvalue.
#     evals, evecs = np.linalg.eigh(rho)
#     possible_eigenstates = []
#     for eval, evec in zip(evals, evecs.transpose()):
#         possible_eigenstates.append((eval, evec))
#     possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

#     evals_env, evecs_env = np.linalg.eigh(rho_env)
#     possible_eigenstates_env = []
#     for eval, evec in zip(evals_env, evecs_env.transpose()):
#         possible_eigenstates_env.append((eval, evec))
#     possible_eigenstates_env.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first


#     # Build the transformation matrix from the `m` overall most significant
#     # eigenvectors.
#     my_m = min(len(possible_eigenstates), m)
#     transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
#     for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
#         transformation_matrix[:, i] = evec

#     transformation_matrix_env = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
#     for i, (eval, evec) in enumerate(possible_eigenstates_env[:my_m]):
#         transformation_matrix_env[:, i] = evec

#     truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
#     print("truncation error:", truncation_error)

#     # Rotate and truncate each operator.
#     new_operator_dict = {}
#     for name, op in sys_enl.operator_dict.items():
#         new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

#     new_env_operator_dict = {}
#     for name, op in sys_enl.operator_dict.items():
#         new_env_operator_dict[name] = rotate_and_truncate(op, transformation_matrix_env)

#     # for name in ["H", "Ntot"]:
#     #     new_operator_dict[name] = rotate_and_truncate(sys_enl.operator_dict[name], transformation_matrix)

#     # for name in ["Cu", "Cd", "Cdu", "Cdd"]:
#     #     new_operator_dict[name] = [rotate_and_truncate(op, transformation_matrix)\
#     #                                for op in sys_enl.operator_dict[name]]

#     newblock = Block(length=sys_enl.length,
#                      basis_size=my_m,
#                      operator_dict=new_operator_dict)

#     newblock_env = Block(length=env_enl.length,
#                      basis_size=my_m,
#                      operator_dict=new_env_operator_dict)


#     return newblock, newblock_env, energy

def sweep(sys, env, m, forward=True):
    """

    Performs a single DMRG sweep from left to right using `sys` as the system and `env` as the
    environment, keeping a maximum of `m` states in the new basis.
    """
    # assert is_valid_block(sys)
    # assert is_valid_block(env)

    # Enlarge each block by a single site.
    sys_enl = enlarge_block(sys, forward=forward)

    if sys is env:  # no need to recalculate a second time
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env, forward=not forward)

    # assert is_valid_enlarged_block(sys_enl)
    # assert is_valid_enlarged_block(env_enl)

    # Construct the full superblock Hamiltonian.
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict

    superblock_hamiltonian = kron(sys_enl_op["H"], identity(m_env_enl)) + \
                             kron(identity(m_sys_enl), env_enl_op["H"])

    if forward:
        for j in range(sys.length):
            for k in range(env.length):
                superblock_hamiltonian += eri[j, (L-1-k)] * kron(sys_enl_op["Ntot"][j], env_enl_op["Ntot"][k])
                superblock_hamiltonian += h1e[j, (L-1-k)] * (kron(sys_enl_op["Cdu"][j], env_enl_op["Cu"][k]) +\
                                                kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k]) +\
                                                kron(sys_enl_op["Cu"][j], env_enl_op["Cdu"][k]) +\
                                                kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k])
                                                )
    else:
        for j in range(sys.length):
            for k in range(env.length):
                superblock_hamiltonian += eri[k, int(L-j-1)] * kron(sys_enl_op["Ntot"][j], env_enl_op["Ntot"][k])
                superblock_hamiltonian += h1e[k, int(L-j-1)] * (kron(sys_enl_op["Cdu"][j], env_enl_op["Cu"][k]) +\
                                                kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k]) +\
                                                kron(sys_enl_op["Cu"][j], env_enl_op["Cdu"][k]) +\
                                                kron(sys_enl_op["Cdd"][j], env_enl_op["Cd"][k])
                                                )

    # Call ARPACK to find the superblock ground state.  ("SA" means find the
    # "smallest in amplitude" eigenvalue.)
    (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")

    # Construct the reduced density matrix of the system by tracing out the
    # environment
    #
    # We want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.  Since the environment (column) index updates most
    # quickly in our Kronecker product structure, psi0 is thus row-major ("C
    # style").
    psi0 = psi0.reshape([sys_enl.basis_size, -1], order="C")
    rho = np.dot(psi0, psi0.conjugate().transpose())

    # Diagonalize the reduced density matrix and sort the eigenvectors by
    # eigenvalue.
    evals, evecs = np.linalg.eigh(rho)
    possible_eigenstates = []
    for eval, evec in zip(evals, evecs.transpose()):
        possible_eigenstates.append((eval, evec))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    print("truncation error:", truncation_error)

    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)

    return newblock, energy


def graphic(sys_block, env_block, sys_label="l"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    assert sys_label in ("l", "r")
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        # The system should be on the right and the environment should be on
        # the left, so reverse the graphic.
        graphic = graphic[::-1]
    return graphic

def infinite_system_algorithm(L, m):
    block = initial_block
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.length < L:
        print("L =", block.length * 2 + 2)
        block, energy = single_dmrg_step(block, block, m=m)
        print("E/L =", energy / (block.length * 2))



def finite_system_algorithm(L, m_warmup, m):
    """
    Finite system DMRG

    Parameters
    ----------
    L : TYPE
        DESCRIPTION.
    m_warmup : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    assert L % 2 == 0  # require that L is an even number

    # To keep things simple, this dictionary is not actually saved to disk, but
    # we use it to represent persistent storage.
    block_disk = {}  # "disk" storage for Block objects

    # Use the infinite system algorithm to build up to desired size.  Each time
    # we construct a block, we save it for future reference as both a left
    # ("l") and right ("r") block, as the infinite system algorithm assumes the
    # environment is a mirror image of the system.
    block = initial_block
    block_disk["l", block.length] = block
    block_disk["r", block.length] = block

    while 2 * block.length < L:
        # Perform a single DMRG step and save the new Block to "disk"
        print(graphic(block, block))
        block, energy = single_dmrg_step(block, block, m=m_warmup)
        print("E/L =", energy / (block.length * 2))
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block

    # Now that the system is built up to its full size, we perform sweeps using
    # the finite system algorithm.  At first the left block will act as the
    # system, growing at the expense of the right block (the environment), but
    # once we come to the end of the chain these roles will be reversed.
    sys_label, env_label = "l", "r"
    sys_block = block; del block  # rename the variable

    print(sys_block.basis_size)

    # for m in m_sweep_list:
    while True:
        # Load the appropriate environment block from "disk"
        env_block = block_disk[env_label, L - sys_block.length - 2]

        print(sys_block.length, env_block.length)


        if env_block.length == 1:
            # We've come to the end of the chain, so we reverse course.
            sys_block, env_block = env_block, sys_block
            sys_label, env_label = env_label, sys_label

        # Perform a single DMRG step.
        print(graphic(sys_block, env_block, sys_label))
        sys_block, energy = single_dmrg_step(sys_block, env_block, m=m)

        print("E/L =", energy / L)

        # Save the block from this step to disk.
        block_disk[sys_label, sys_block.length] = sys_block

        # Check whether we just completed a full sweep.
        if sys_label == "l" and 2 * sys_block.length == L:
            break  # escape from the "while True" loop

class DMRG:
    """
    ab initio DRMG/DVR quantum chemistry calculation
    """
    def __init__(self, mf, L, m=None):
        """


        Parameters
        ----------
        d : TYPE
            DESCRIPTION.
        L : TYPE
            DESCRIPTION.
        m : TYPE, optional
            maximum bond dimension. The default is None.

        Returns
        -------
        None.

        """
        self.mf = mf

        self.dim = 4

        self.h1e = mf.hcore
        self.eri = mf.eri

        self.nsites = self.L = L
        self.m = m

    def run(self, initial_block, m_warmup=10):
        L = self.L
        m = self.m

        assert L % 2 == 0  # require that L is an even number

        # To keep things simple, this dictionary is not actually saved to disk, but
        # we use it to represent persistent storage.
        block_disk = {}  # "disk" storage for Block objects

        # Use the infinite system algorithm to build up to desired size.  Each time
        # we construct a block, we save it for future reference as both a left
        # ("l") and right ("r") block, as the infinite system algorithm assumes the
        # environment is a mirror image of the system.
        block = initial_block
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block

        while 2 * block.length < L:
            # Perform a single DMRG step and save the new Block to "disk"
            print(graphic(block, block))
            block, energy = single_dmrg_step(block, block, m=m_warmup)
            print("E/L =", energy / (block.length * 2))
            block_disk["l", block.length] = block
            block_disk["r", block.length] = block

        # Now that the system is built up to its full size, we perform sweeps using
        # the finite system algorithm.  At first the left block will act as the
        # system, growing at the expense of the right block (the environment), but
        # once we come to the end of the chain these roles will be reversed.
        sys_label, env_label = "l", "r"
        sys_block = block; del block  # rename the variable

        # Now that the system is built up to its full size, we perform sweeps using
        # the finite system algorithm.  At first the left block will act as the
        # system, growing at the expense of the right block (the environment), but
        # once we come to the end of the chain these roles will be reversed.
        sys_label, env_label = "l", "r"
        sys_block = block; del block  # rename the variable

        # for m in m_sweep_list:
        while True:
            # Load the appropriate environment block from "disk"
            env_block = block_disk[env_label, L - sys_block.length - 2]

            if env_block.length == 1:
                # We've come to the end of the chain, so we reverse course.
                sys_block, env_block = env_block, sys_block
                sys_label, env_label = env_label, sys_label

            # Perform a single DMRG step.
            print(graphic(sys_block, env_block, sys_label))

            sys_block, energy = single_dmrg_step(sys_block, env_block, m=m)

            print("E =", energy)

            # Save the block from this step to disk.
            block_disk[sys_label, sys_block.length] = sys_block

            # Check whether we just completed a full sweep.
            if sys_label == "l" and 2 * sys_block.length == L:
                break  # escape from the "while True" loop

            # finite_system_algorithm(L, m_warmup, m)

    def warmup(self, m_warmup):
        block_disk = {}  # "disk" storage for Block objects

        L = self.L
        h1e = self.h1e
        eri = self.eri

        assert L % 2 == 0  # require that L is an even number


        # initial system band environment block
        sys = Block(length=1, basis_size=4, operator_dict={
                "H": H1[0],
                "Ntot": [Ntot],
                "Cu": [Cu @ JW],
                "Cd": [Cd @ JW],
                # "Nu": [ops['Nu']],
                # "Nd": [ops['Nd']],
                "Cdu": [Cdu @ JW],
                "Cdd": [Cdd @ JW]
            })

        env =  Block(length=1, basis_size=4, operator_dict={
                "H": H1[-1],
                "Ntot": [Ntot],
                "Cu": [Cu],
                "Cd": [Cd],
                # "Nu": [ops['Nu']],
                # "Nd": [ops['Nd']],
                "Cdu": [Cdu],
                "Cdd": [Cdd]
            })

        block_disk["l", sys.length] = sys
        block_disk["r", env.length] = env

        while 2 * sys.length < L:

            # Perform a single DMRG step and save the new Block to "disk"
            print(graphic(sys, env))

            sys, env, energy = single_dmrg_step(sys, env, m=m_warmup)

            print("E =", energy)

            block_disk["l", sys.length] = sys
            block_disk["r", env.length] = env

        ### Sweep

        # Now that the system is built up to its full size, we perform sweeps using
        # the finite system algorithm.  At first the left block will act as the
        # system, growing at the expense of the right block (the environment), but
        # once we come to the end of the chain these roles will be reversed.
        sys_label, env_label = "l", "r"
        sys_block = sys; del sys  # rename the variable

        # for m in m_sweep_list:
        forward = True
        while False:
            # Load the appropriate environment block from "disk"
            env_block = block_disk[env_label, L - sys_block.length - 2]

            if env_block.length == 1:
                forward = not forward
                # We've come to the end of the chain, so we reverse course.
                sys_block, env_block = env_block, sys_block
                sys_label, env_label = env_label, sys_label

            # Perform a single DMRG step.
            print(graphic(sys_block, env_block, sys_label))

            sys_block, energy = sweep(sys_block, env_block, m=self.m, forward=forward)

            print("E =", energy)

            # Save the block from this step to disk.
            block_disk[sys_label, sys_block.length] = sys_block

            # Check whether we just completed a full sweep.
            if sys_label == "l" and 2 * sys_block.length == L:
                # break  # escape from the "while True" loop
                print('one cycle')

class DMRGSCF(DMRG):
    """
    optimize the orbitals
    """
    pass


if __name__=='__main__':
    from pyscf import gto, scf, dft, tddft, ao2mo
    from pyqed.qchem.mol import get_hcore_mo, get_eri_mo
    from pyqed.qchem.gto.rhf import RHF
    from pyqed.qchem.dvr.rhf import RHF1D
    from pyqed.models.ShinMetiu2e1d import AtomicChain

    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)


    # mol = gto.Mole()
    # mol.atom = [
    #     ['H' , (0. , 0. , .917)],
    #     ['Li' , (0. , 0. , 0.)], ]
    # mol.basis = 'sto3g'
    # mol.build()



    # mf = scf.RHF(mol).run()

    # e, fcivec = pyscf.fci.FCI(mf).kernel(verbose=4)
    # print(e)
    # Ca = mf.mo_coeff[0ArithmeticError
    # n = Ca.shape[-1]

    # mo_coeff = mf.mo_coeff
    # get the two-electron integrals as a numpy array
    # eri = get_eri_mo(mol, mo_coeff)

    # n = mol.nao
    # Ca = mo_coeff


    l = 10/au2angstrom
        # print(self.L)
        # self.mass = mass  # nuclear mass
    # z = np.array([-L/2, -L/4, L/4, L/2])
    z0 = np.linspace(-1, 1, 4) * l/2
    print(z0)

    print('interatomic distance = ', (z0[1] - z0[0])*au2angstrom)


    mol = AtomicChain(z0, charge=0)
    print('number of electrons = ', mol.nelec)
    ###############################################################################
    # mol = ShinMetiu1d(nstates=3, nelec=2)
    # # mol.spin = 0
    # mol.create_grid([-15/au2angstrom, 15/au2angstrom], level=4)

    # # exact
    # R = 0.
    # w, u = mol.single_point(R)
    # print(w)

    # fig, ax = plt.subplots()
    # ax.imshow(u[:, 1].reshape(mol.nx, mol.nx), origin='lower')

    # HF
    mf = RHF1D(mol, dvr_type='sine')
    mf.domain = [-15/au2angstrom, 15/au2angstrom]
    mf.nx = 6

    mf.run()
    # exact
    # w, u = mol.single_point(R)
    # print(w)


    # fig, ax = plt.subplots()
    # ax.imshow(u[:, 1].reshape(mol.nx, mol.nx), origin='lower')


    mo = mf.mo_coeff
    print('orb energies', mf.mo_energy)

    h1e = mf.hcore
    eri = mf.eri

    # print(h1e)
    # print(eri)

    L = nsites = mf.nx


    # onsite Hamiltonian
    H1 = [ (h1e[j, j] + 0.5 * eri[j, j]) * Ntot for j in range(nsites)]

    print('onsite energies', [ (h1e[j, j] + 0.5 * eri[j, j]) for j in range(nsites)])


    ### DMRG

    dmrg = DMRG(mf, L=nsites, m=30)
    dmrg.warmup(20)

    nx = mf.nx
    I = np.eye(nx)
    eri_full = np.zeros((nx, nx, nx, nx))
    for i in range(nx):
        for j in range(nx):
            for k in range(nx):
                for l in range(nx):
                    eri_full[i,j,k,l] = eri[j,k] * I[i, j] * I[k, l]

    E, X = SpinHalfFermionChain(h1e, eri_full, nelec=mol.nelectron).run()
    print(E)
    # Model-specific code for the Heisenberg XXZ chain

    # Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype='d')  # single-site S^z
    # Sp1 = np.array([[0, 1], [0, 0]], dtype='d')  # single-site S^+

    # H1 = np.array([[0, 0], [0, 0]], dtype='d')  # single-site portion of H is zero








    # conn refers to the connection operator, that is, the operator on the edge of
    # the block, on the interior of the chain.  We need to be able to represent S^z
    # and S^+ on that site in the current basis in order to grow the chain.
    # initial_block = Block(length=1, basis_size=model_d, operator_dict={
    #     "H": H1,
    #     "Cu": ops['Cu'],
    #     "Cd": ops['Cd'],
    #     "Nu": ops['Nu'],
    #     "Nd": ops['Nd']
    # })

    #infinite_system_algorithm(L=100, m=20)
    # finite_system_algorithm(L=nsites, m_warmup=10, m=10)