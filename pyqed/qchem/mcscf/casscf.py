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


from pyqed.optimize import minimize

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


    def run(self, nstates= None, weights = None):
        mf = self.mf

        # canonical molecular orbs
        C0 = mf.mo_coeff

        # CASCI roots
        if nstates == None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        if weights != None:
            self.weights = weights
            if nstates != len(self.weights):
                raise ValueError("the nstates you requires does not align with the nstates indicated by the weights. check input.")
        nmo = self.mf.nao
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore

        mc = CASCI(mf, ncas=ncas, nelecas=nelecas)
        # spin
        mc.spin_purification = self.spin_purification
        mc.ss = self.ss
        mc.shift = self.shift

        # shift = self.shift
        # purify_spin = self.spin_purification


        # if self.spin_purification:
        #     mc.fix_spin(ss=self.ss, shift=self.shift)

        mc.run(nstates)


        # matrix elements in CMOs
        h1e = mf.get_hcore_mo()
        eri = mf.get_eri_mo()

        U0 = np.zeros((nmo, ncas+ncore))
        for i in range(ncas+ncore):
            U0[i, i] = 1.

        if nstates == 1: # ground state only
            C, mc = kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=self.max_cycles)

        elif nstates > 1:
            if self.weights is None:
                self.state_average(weights = np.ones(nstates)/nstates)
            if len(self.weights) != nstates: 
                self.state_average(weights = np.ones(nstates)/nstates)

            C, mc = kernel_state_average(mc, weights=self.weights, U0=U0, nelecas=nelecas, ncas=ncas,
                                         C0=C0, h1e=h1e, eri=eri)

        self.mo_coeff = C
        self.e_tot = mc.e_tot
        self.ci = mc.ci

        return self

    def state_average(self, weights):
        self.nstates = len(weights)
        self.weights = weights
        return self


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

    e = contract('pq, pa, qb, ab ->', h1e, U, U, dm1)
    e += 0.5 * (contract('pqrs, pa, qb, rc, sd, abcd ->', eri, U, U, U, U, dm2))
    return e



def kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=30, tol=1e-6, **kwargs):
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

    U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))

    k = 0

    e_old = mc.e_tot

    converged = False
    while k < max_cycles:

        mo_coeff = C0 @ U

        mc.run(mo_coeff=mo_coeff, **kwargs)

        if abs(mc.e_tot - e_old) < tol:
            print('\nCASSCF converged at macroiteration {}'.format(k))
            print("E(CASSCF) = {}".format(mc.e_tot))
            converged = True
            break

        e_old = mc.e_tot


        dm1, dm2 = mc.make_rdm12(0, with_core=with_core)

        # U0 = orth(U + 0.1 * np.random.randn(nmo, ncas))

        U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2), tau=1)
        # print(E + mol.energy_nuc())

        k += 1

    if not converged:
        raise RuntimeError('Max macro steps reached. CASSCF not converged.')

    return mo_coeff, mc


def kernel_state_average(mc, weights, U0, nelecas, ncas, C0, h1e, eri,
                         max_cycles=50, tol=1e-6, **kwargs):

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

    U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))


    e_old = sum(weights * mc.e_tot)

    converged = False
    k = 0
    while k < max_cycles:

        mo_coeff = C0 @ U

        mc.run(nstates, mo_coeff=mo_coeff, **kwargs)

        eAve = sum(weights * mc.e_tot)

        if abs(eAve - e_old) < tol:
            print('CASSCF converged at macroiteration {}'.format(k))
            print("E(CASSCF) = {}".format(mc.e_tot))
            converged = True
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
        # print(E + mol.energy_nuc())

        k += 1

    if not converged:
        raise RuntimeError('Max macro steps reached. CASSCF not converged.')

    return mo_coeff, mc


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

    mol = Molecule(atom='Li 0 0 0; H 0 0 1.4', unit='b', basis='6311g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc = CASSCF(mf, ncas=2, nelecas=2, max_cycles=50)

    nstates = 1
    mc.state_average(weights = np.ones(nstates)/nstates)
    mc.fix_spin(ss=0, shift=0.2)
    mc.run()

    # correct result is E(CASSCF) = [-7.67160344]