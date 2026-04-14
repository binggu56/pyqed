#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:07:30 2025

@author: bingg
"""
import numpy as np
from scipy.linalg import eigh
# from pyqed.qchem.mcscf.casci import CASCI
from opt_einsum import contract
from pyqed.qchem.mcscf.direct_ci import CASCI
# from pyqed.qchem.mcscf.casci import CASCI
import matplotlib.pyplot as plt
from pyqed import optimize

from pyqed.optimize import minimize
# from pyqed.optimize import Optimize
import time

from scipy.linalg import sqrtm, fractional_matrix_power, logm, expm
from functools import reduce

class CASSCF(CASCI):
    """

    Using the OptOrbFCI algorithm to optimize orbitals
    (better than conventional CASSCF algorithm)



    """
    def __init__(self, mf, ncas, nelecas, max_cycles=30, **kwargs):
        super().__init__(mf, ncas, nelecas, **kwargs)

        self.max_cycles = max_cycles # macroiterations
        self.tol = 1e-6 # energy tol
        self.mo_coeff = None # opt orb


        self.weights = None
        self.nstates = 1

        self.dipole_moment = None

        self.dm1 = None
        self.dm2 = None

        self.electric_dipole = None
        self.magnetic_dipole = None

        hcore = mf.get_hcore()
        if hcore.dtype == complex:
            self.dtype = np.complex128
        else:
            self.dtype = np.float64

    def _make_casci(self, mf, ncas, nelecas):

        mc = CASCI(mf, ncas=ncas, nelecas=nelecas)

        mc.spin_purification = self.spin_purification
        mc.ss = self.ss
        mc.shift = self.shift

        return mc


    def run(self, nstates=1, method='newton'):

        mf = self.mf

        # canonical molecular orbs
        C0 = mf.mo_coeff

        # CASCI roots
        if nstates == None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        if self.weights is not None:
            # self.weights = weights
            if nstates != len(self.weights):
                print('nstates', nstates)
                print('len weight', len(self.weights))
                raise ValueError("the nstates you requires does not align with the nstates indicated by the weights. check input.")

        nmo = self.mf.nao
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore

        # mc = CASCI(mf, ncas=ncas, nelecas=nelecas)
        # # spin
        # mc.spin_purification = self.spin_purification
        # mc.ss = self.ss
        # mc.shift = self.shift

        mc = self._make_casci(mf, ncas=ncas, nelecas=nelecas)


        # shift = self.shift
        # purify_spin = self.spin_purification


        # if self.spin_purification:
        #     mc.fix_spin(ss=self.ss, shift=self.shift)
        # print('nstate.....', nstates)
        mc.run(nstates)


        # matrix elements in CMOs
        h1e = mf.get_hcore_mo()
        eri = mf.get_eri_mo()

        U0 = np.zeros((nmo, ncas+ncore))
        for i in range(ncas+ncore):
            U0[i, i] = 1.


        if nstates == 1: # ground state only
            C, mc, e_table, num_iter, dm1, dm2 = kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=self.max_cycles)

        elif nstates > 1:

            if self.weights is None:
                self.state_average(weights = np.ones(nstates)/nstates)
            if len(self.weights) != nstates: 
                self.state_average(weights = np.ones(nstates)/nstates)

            C, mc, e_table, num_iter, dm1, dm2 = kernel_state_average(mc, weights=self.weights, U0=U0, nelecas=nelecas, ncas=ncas,
                                         C0=C0, h1e=h1e, eri=eri, max_cycles=self.max_cycles)

        self.mo_coeff = C
        self.e_tot = mc.e_tot
        self.ci = mc.ci
        self.e_table = e_table
        self.num_iter = num_iter
        self.dm1 = dm1
        self.dm2 = dm2

        return self

    def state_average(self, weights=None):
        self.nstates = len(weights)
        self.weights = weights
        return self

    def get_electric_dip(self, initial_state, final_state, unit, **kwargs):

        # # mf = self.mf

        # # # canonical molecular orbs
        # # C0 = mf.mo_coeff

        # # CASCI roots
        # nstates = self.nstates

        # nmo = self.mf.nao
        # ncas = self.ncas
        # nelecas = self.nelecas
        # ncore = self.ncore

        # mc = CASCI(mf, ncas=ncas, nelecas=nelecas, electric_field=self.electric_field, vector_potential=self.vector_potential)
        # # spin
        # mc.spin_purification = self.spin_purification
        # mc.ss = self.ss
        # mc.shift = self.shift

        # mc.run(nstates, mo_coeff=self.mo_coeff)

        # # matrix elements in CMOs
        # h1e = mf.get_hcore_mo()
        # eri = mf.get_eri_mo()
        # print('ci',self.ci)
        # mc = CASCI(mf=self, ncas=self.ncas, nelecas=self.nelecas, electric_field=self.electric_field, vector_potential=self.vector_potential)
        # self.electric_dipole =  CASCI.get_electric_dip(self, initial_state, final_state, unit, **kwargs)

        return self.electric_dipole

    # def get_dip_moment(self, nstates, unit='Debye', **kwargs):

    #     dm1 = self.dm1
    #     mo_coeff = self.mo_coeff[:, 0:self.ncas + self.ncore]


    #     if nstates == 1:
    #         print('mo_coeff shape', np.shape(mo_coeff))
    #         dip = dipole_moment(self.mol, dm1, mo_coeff, unit)
    #         print(' dipole moment : {} {}'.format(dip, unit))
    #         self.dipole_moment = dip
    #     else:
    #         self.dipole_moment = []
    #         for state_id in range(nstates):
    #             dip = dipole_moment(self.mol, dm1[state_id], mo_coeff, unit)
    #             print('state : {}; dipole moment : {} {}'.format(state_id, dip, unit))
    #             self.dipole_moment.append(dip)

    #     return self.dipole_moment




def energy(U, h1e, eri, dm1, dm2):
    """
    electronic energy

    Parameters
    ----------
    U : ndarray of (n, p < n/2)
        transformation matrix
    h1e : TYPE
        core Hamiltonian in canonical MO
    eri : TYPE
        DESCRIPTION.
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    e : TYPE
        DESCRIPTION.

    """

    e = contract('pq, pa, qb, ab ->', h1e, U.conj(), U, dm1)
    e += 0.5 * (contract('pqrs, pa, qb, rc, sd, abcd ->', eri, U.conj(), U.conj(), U, U, dm2))
    # e += 0.5 * (contract('pqrs, pa, qb, rc, sd, acdb ->', eri, U.conj(), U.conj(), U, U, dm2))

    return e




def kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=50, tol=1e-6, method='newton', dtype = np.float64, **kwargs):
    """
    complete active space orbital optimization with orthonomality constraint

    .. math::
        U^\top U = I_N

        E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
        1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

    where U is a M x N (M > N) matrix.

    .. math::
        U_{k+1} = orth(U_k - \tau_k G_k)

    where G_k = \nabla P(U_k) is the gradient.

    Parameters
    ----------
    h1e : TYPE
        DESCRIPTION.
    h2e : TYPE
        DESCRIPTION.
    U0: ndarray
        initial guess of orbitals
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    e_table = np.zeros((mc.nstates, max_cycles))

    # diis 
    maxdiis = 6
    diis_U = []
    diis_err = []

    def diis(U, iter):
        diis_U.append(U.copy())

        if len(diis_U) > 1:
            diis_err.append((diis_U[-1] - diis_U[-2]).copy())

        if iter <= 1:
            return U

        if len(diis_err) > maxdiis:
            diis_err.pop(0)
            diis_U.pop(0)

        print('len(u)_1', len(diis_U))
        print('len(err)', len(diis_err))

        bsize = len(diis_err)
        if bsize < 2:
            return U

        bmat = -1.0 * np.ones((bsize + 1, bsize + 1), dtype=float)
        rhs = np.zeros(bsize + 1, dtype=float)
        bmat[bsize, bsize] = 0.0
        rhs[bsize] = -1.0

        for b1 in range(bsize):
            for b2 in range(bsize):
                bmat[b1, b2] = np.vdot(diis_err[b1], diis_err[b2]).real

        try:
            C = np.linalg.solve(bmat, rhs)
        except np.linalg.LinAlgError:
            print("DIIS: singular B matrix, trying regularization")
            try:
                bmat[:-1, :-1] += np.eye(bsize) * 1.0e-10
                C = np.linalg.solve(bmat, rhs)
            except np.linalg.LinAlgError:
                print("DIIS: regularized solve failed")
                return U

        U_new = np.zeros_like(U, dtype=U.dtype)
        for i, k in enumerate(C[:-1]):
            U_new += k * diis_U[i + 1]

        print('len(u)_2', len(diis_U))

        A = U_new.conj().T @ U_new
        A_inv_sqrt = np.linalg.inv(sqrtm(A))

        return U_new @ A_inv_sqrt


    
    if mc.ncore > 0:
        with_core = True
    else:
        with_core = False

    dm1, dm2 = mc.make_rdm12(0, with_core=with_core)

    # eri = mc.eri_so[0, 0] # for spin-restricted calculation
    # nmo = self.nmo

    # U0 = np.zeros((nmo, ncas))
    # for i in range(ncas):
    #     U0[i, i] = 1

    k = 0
    U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))
    U = diis(U, k)
    # U_mix = update_U(U)
    # U, E = minimize_2(energy, U0, args=(h1e, eri, dm1, dm2))


    e_table[:,k] = mc.e_tot
    k += 1

    e_old = mc.e_tot

    converged = False
    while k < max_cycles:
        # print('mo_coeff shape', mo_coeff.shape)
        # print('U shape', U.shape)
        # if k == 1:
        #     mo_coeff = C0 @ U
        # else:
        #     mo_coeff = 0.5 * C0 @ (U + U_old)

        mo_coeff = C0 @ U
        mc.run(mo_coeff=mo_coeff, **kwargs)
        e_table[:,k] = mc.e_tot

        if abs(mc.e_tot - e_old) < tol:
            print('\nCASSCF converged at macroiteration {}'.format(k+1))
            print("E(CASSCF) = {}".format(mc.e_tot))
            converged = True
            e_table[:,k] = mc.e_tot
            k += 1
            break

        e_old = mc.e_tot


        dm1, dm2 = mc.make_rdm12(0, with_core=with_core)
        # U_old = U
        # U0 = orth(U + 0.1 * np.random.randn(nmo, ncas))
   

        U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2), tau=1)
        U = diis(U, k)
        # U, E = minimize_2(energy, U0, args=(h1e, eri, dm1, dm2))
        k += 1
        print('k = ', k)
        # print(E + mol.energy_nuc())


    if not converged:
        # raise RuntimeError('Max macro steps reached. CASSCF not converged.')
        print(f"Max macro steps reached. CASSCF not converged.")

    # if k >= max_cycles:
    #     k -= 1 
    # print('dm1 shape', np.shape(dm1))
    
    return mo_coeff, mc, e_table, k, dm1, dm2




def kernel_state_average(mc, weights, U0, nelecas, ncas, C0, h1e, eri,
                         max_cycles=50, tol=1e-6, **kwargs):

    e_table = np.zeros((mc.nstates, max_cycles))

    # diis 
    maxdiis = 6
    diis_U = []
    diis_err = []

    def diis(U, iter):
        diis_U.append(U.copy())

        if len(diis_U) > 1:
            diis_err.append((diis_U[-1] - diis_U[-2]).copy())

        if iter <= 1:
            return U

        if len(diis_err) > maxdiis:
            diis_err.pop(0)
            diis_U.pop(0)

        print('len(u)_1', len(diis_U))
        print('len(err)', len(diis_err))

        bsize = len(diis_err)
        if bsize < 2:
            return U

        bmat = -1.0 * np.ones((bsize + 1, bsize + 1), dtype=float)
        rhs = np.zeros(bsize + 1, dtype=float)
        bmat[bsize, bsize] = 0.0
        rhs[bsize] = -1.0

        for b1 in range(bsize):
            for b2 in range(bsize):
                bmat[b1, b2] = np.vdot(diis_err[b1], diis_err[b2]).real

        try:
            C = np.linalg.solve(bmat, rhs)
        except np.linalg.LinAlgError:
            print("DIIS: singular B matrix, trying regularization")
            try:
                bmat[:-1, :-1] += np.eye(bsize) * 1.0e-10
                C = np.linalg.solve(bmat, rhs)
            except np.linalg.LinAlgError:
                print("DIIS: regularized solve failed")
                return U

        U_new = np.zeros_like(U, dtype=U.dtype)
        for i, k in enumerate(C[:-1]):
            U_new += k * diis_U[i + 1]

        print('len(u)_2', len(diis_U))

        A = U_new.conj().T @ U_new
        A_inv_sqrt = np.linalg.inv(sqrtm(A))

        return U_new @ A_inv_sqrt
    
    if mc.ncore > 0:
        with_core = True
    else:
        with_core = False

    nstates = mc.nstates

    dm1 = 0
    dm2 = 0
    for n in range(nstates):
        _dm1, _dm2 = mc.make_rdm12(n, with_core=with_core)
        dm1 += _dm1 * weights[n]
        dm2 += _dm2 * weights[n]

    k = 0
    U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))
    U = diis(U, k)
    # U, E = Optimize(energy, U0, args=(h1e, eri, dm1, dm2))

    e_table[:,k] = mc.e_tot
    k += 1

    e_old = sum(weights * mc.e_tot)

    converged = False
    while k < max_cycles:

        mo_coeff = C0 @ U

        mc.run(nstates, mo_coeff=mo_coeff, **kwargs)

        eAve = sum(weights * mc.e_tot)
        e_table[:,k] = mc.e_tot

        if abs(eAve - e_old) < tol:
            print('CASSCF converged at macroiteration {}'.format(k+1))
            print("E(CASSCF) = {}".format(mc.e_tot))
            converged = True
            e_table[:,k] = mc.e_tot
            k += 1
            break

        e_old = eAve

        # update 1- and 2-RDMs
        dm1 = 0
        dm2 = 0
        for n in range(nstates):
            _dm1, _dm2 = mc.make_rdm12(n, with_core=with_core)
            dm1 += _dm1 * weights[n]
            dm2 += _dm2 * weights[n]


        U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))
        U = diis(U, k)
        k += 1
        print('k = ', k)
        # U, E = Optimize(energy, U0, args=(h1e, eri, dm1, dm2))
        # print(E + mol.energy_nuc())


    if not converged:
        # raise RuntimeError('Max macro steps reached. CASSCF not converged.')
        print(f"Max macro steps reached. CASSCF not converged.")

    # if k >= max_cycles:
    #     k -= 1

    dm1 = []
    dm2 = []
    for n in range(nstates):
        _dm1, _dm2 = mc.make_rdm12(n, with_core=with_core)
        dm1.append(_dm1)
        dm2.append(_dm2)
    return mo_coeff, mc, e_table, k, dm1, dm2


# def constrained_optimization(U, h1e, h2e, dm1, dm2, max_steps=50):
#     """
#     complete active space orbital optimization with orthonomality constraint

#     .. math::
#         U^\top U = I_N

#         E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
#         1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

#     where U is a M x N (M > N) matrix.

#     .. math::
#         U_{k+1} = orth(U_k - \tau_k G_k)

#     Parameters
#     ----------
#     h1e : TYPE
#         DESCRIPTION.
#     h2e : TYPE
#         ERI.
#     dm1 : TYPE
#         1RDM.
#     dm2 : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """

#     # orb opt
#     converged = False
#     k = 0

#     # add random noise
#     U += 0.1 * np.random.randn(U.shape)
#     U = orth(U)

#     U_old = U.copy()
#     for k in range(max_steps):

#         G = gradient(U, h1e, h2e, dm1, dm2)
#         U = orth(U - stepsize(k) * G)

#         if 1 - abs(inner(U_old, U)) < 1e-3:
#             converged = True
#             break

#         U_old = U.copy()
#         k += 1

#     if converged:
#         return U
#     else:
#         raise RuntimeError('Constrained optimization not converged.')


def gradient(U, h1e, h2e, dm1, dm2):
    g = h1e @ U @ dm1.T + h1e.T @ U @ dm1  # these two terms are probably the same
    g += 0.5 * (contract('pqrs, qb, rc, sd, abcd -> pa', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, rc, sd, abcd -> qb', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, sd, abcd -> rc', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, rc, abcd -> sd', h2e, U, U, U, dm2) )
    return g



class CASPT2(CASSCF):
    """
    CASSCF
    """
    pass


# import torch

# from expm32 import expm32, differential




if __name__=='__main__':

    from pyqed import Molecule
    # from pyqed.qchem.mcscf.direct_ci import CASCI

    print('-------------------- pyqed --------------------')
    mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='sto3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc = CASSCF(mf, ncas=6, nelecas=6, max_cycles=100)

    nstates = 1
    mc.state_average(weights = np.ones(nstates)/nstates)
    mc.fix_spin(ss=0, shift=0.2)
    mc.run(nstates=nstates)






