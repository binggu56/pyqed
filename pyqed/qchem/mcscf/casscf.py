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
# from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.mcscf.casci import CASCI
import matplotlib.pyplot as plt
from pyqed import optimize
from pyqed.optimize import Newton_opt
# from pyqed.qchem.mcscf.casci import dipole_moment
# from pyqed.qchem.mcscf.casci import make_rdm1
# from pyqed.qchem.mcscf.casci import make_rdm2

from pyqed.optimize import minimize
from pyqed.optimize import opt, UnitaryNewtonSolver

import time

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
        nstates = self.nstates

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

        mc.run(nstates)


        # matrix elements in CMOs
        h1e = mf.get_hcore_mo()
        eri = mf.get_eri_mo()

        U0 = np.zeros((nmo, ncas+ncore))
        for i in range(ncas+ncore):
            U0[i, i] = 1.


        if nstates == 1: # ground state only
            C, mc, e_table, num_iter, dm1, dm2 = kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=self.max_cycles, method='newton', dtype=self.dtype)

        elif nstates > 1:

            if self.weights is None:
                raise ValueError('State weights not provided.')

            C, mc, e_table, num_iter, dm1, dm2 = kernel_state_average(mc, weights=self.weights, U0=U0, nelecas=nelecas, ncas=ncas,
                                         C0=C0, h1e=h1e, eri=eri, max_cycles=self.max_cycles, method='newton', dtype=self.dtype)

            # self.electric_dipole = mc.get_electric_dip(self.)

        self.mo_coeff = C
        self.e_tot = mc.e_tot
        self.ci = mc.ci
        self.e_table = e_table
        self.num_iter = num_iter
        self.dm1 = dm1
        self.dm2 = dm2

        # self.electric_dipole = mc.get_electric_dip(initial_state=0, final_state=0, unit='Debye')
        # for state_id in range(nstates):
        #     mc.get_electric_dip(initial_state=0, final_state=state_id, unit='au')

        return self

    def state_average(self, weights):
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


def kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=30, tol=1e-6, method='newton', dtype = np.float64, **kwargs):
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

    # # test newton opt
    # print('*'*100)
    # opt = Newton_opt(U0, h1e, eri, dm1, dm2)
    # opt.get_gradient()
    # opt.get_hessian()
    
    # U, E = opt(energy, U0, h1e, eri, dm1, dm2)
    # newt = UnitaryNewtonSolver(energy, U0, h1e, eri, dm1, dm2)
    # newt.get_gradient()
    # newt.get_hessian()
    # U, E = newt.solve()
    # print('orb_opt_ener', E+mol.energy_nuc())

    U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2), dtype=dtype)

    k = 0

    e_old = mc.e_tot

    converged = False
    while k < max_cycles:

        mo_coeff = C0 @ U

        mc.run(mo_coeff=mo_coeff, **kwargs)

        e_table[:,k] = mc.e_tot

        if abs(mc.e_tot - e_old) < tol:
            print('\nCASSCF converged at macroiteration {}'.format(k+1))
            print("E(CASSCF) = {}".format(mc.e_tot))
            converged = True
            break

        e_old = mc.e_tot


        dm1, dm2 = mc.make_rdm12(0, with_core=with_core)

        U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2), tau=1, dtype=dtype)
        # newt = UnitaryNewtonSolver(energy, U0, h1e, eri, dm1, dm2)
        # newt.get_gradient()
        # newt.get_hessian()
        # U, E = newt.solve()
        # print('orb_opt_ener', E+mol.energy_nuc())

        k += 1

    if not converged:
        # raise RuntimeError('Max macro steps reached. CASSCF not converged.')
        print(f"Max macro steps reached. CASSCF not converged.")

    if k >= max_cycles:
        k -= 1 
    # print('dm1 shape', np.shape(dm1))
    
    return mo_coeff, mc, e_table, k, dm1, dm2



# def kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=30, tol=1e-6, **kwargs):
#     """
#     complete active space orbital optimization with orthonomality constraint

#     .. math::
#         U^\top U = I_N

#         E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
#         1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

#     where U is a M x N (M > N) matrix.

#     .. math::
#         U_{k+1} = orth(U_k - \tau_k G_k)

#     where G_k = \nabla P(U_k) is the gradient.

#     Parameters
#     ----------
#     h1e : TYPE
#         DESCRIPTION.
#     h2e : TYPE
#         DESCRIPTION.
#     U0: ndarray
#         initial guess of orbitals
#     dm1 : TYPE
#         DESCRIPTION.
#     dm2 : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     e_table = np.zeros((mc.nstates, max_cycles))

#     if mc.ncore > 0:
#         with_core = True
#     else:
#         with_core = False

#     dm1, dm2 = mc.make_rdm12(0, with_core=with_core)

#     # eri = mc.eri_so[0, 0] # for spin-restricted calculation
#     # nmo = self.nmo

#     # U0 = np.zeros((nmo, ncas))
#     # for i in range(ncas):
#     #     U0[i, i] = 1

#     U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))

#     k = 0

#     e_old = mc.e_tot

#     converged = False
#     while k < max_cycles:

#         mo_coeff = C0 @ U

#         mc.run(mo_coeff=mo_coeff, **kwargs)

#         e_table[:,k] = mc.e_tot

#         if abs(mc.e_tot - e_old) < tol:
#             print('\nCASSCF converged at macroiteration {}'.format(k))
#             print("E(CASSCF) = {}".format(mc.e_tot))
#             converged = True
#             break

#         e_old = mc.e_tot


#         dm1, dm2 = mc.make_rdm12(0, with_core=with_core)

#         U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2), tau=1)

#         # print(E + mol.energy_nuc())

#         k += 1

#     if not converged:
#         # raise RuntimeError('Max macro steps reached. CASSCF not converged.')
#         print(f"Max macro steps reached. CASSCF not converged.")

#     if k >= max_cycles:
#         k -= 1 
#     print('dm1 shape', np.shape(dm1))
    
#     return mo_coeff, mc, e_table, k, dm1, dm2


def kernel_state_average(mc, weights, U0, nelecas, ncas, C0, h1e, eri,
                         max_cycles=50, tol=1e-6, method='newton', dtype = np.float64, **kwargs):

    e_table = np.zeros((mc.nstates, max_cycles))
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
    print('dm2 type',dm2.dtype)

    U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2), dtype=dtype)


    e_old = sum(weights * mc.e_tot)


    converged = False
    k = 0
    while k < max_cycles:

        mo_coeff = C0 @ U

        mc.run(nstates, mo_coeff=mo_coeff, **kwargs)

        eAve = sum(weights * mc.e_tot)
        e_table[:,k] = mc.e_tot

        if abs(eAve - e_old) < tol:
            print('CASSCF converged at macroiteration {}'.format(k+1))
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

        U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2), dtype=dtype)
        # print('optimize energy', E + mf.e_nuc)

        # print(E + mol.energy_nuc())

        k += 1

    if not converged:
        # raise RuntimeError('Max macro steps reached. CASSCF not converged.')
        print(f"Max macro steps reached. CASSCF not converged.")

    if k >= max_cycles:
        k -= 1

    dm1 = []
    dm2 = []
    for n in range(nstates):
        _dm1, _dm2 = mc.make_rdm12(n, with_core=with_core)
        dm1.append(_dm1)
        dm2.append(_dm2)
        
    # print('dm1 shape', np.shape(dm1))
    return mo_coeff, mc, e_table, k, dm1, dm2


def constrained_optimization(U, h1e, h2e, dm1, dm2, max_steps=50):
    """
    complete active space orbital optimization with orthonomality constraint

    .. math::
        U^\top U = I_N

        E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
        1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

    where U is a M x N (M > N) matrix.

    .. math::
        U_{k+1} = orth(U_k - \tau_k G_k)

    Parameters
    ----------
    h1e : TYPE
        DESCRIPTION.
    h2e : TYPE
        ERI.
    dm1 : TYPE
        1RDM.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # orb opt
    converged = False
    k = 0

    # add random noise
    U += 0.1 * np.random.randn(U.shape)
    U = orth(U)

    U_old = U.copy()
    for k in range(max_steps):

        G = gradient(U, h1e, h2e, dm1, dm2)
        U = orth(U - stepsize(k) * G)

        if 1 - abs(inner(U_old, U)) < 1e-3:
            converged = True
            break

        U_old = U.copy()
        k += 1

    if converged:
        return U
    else:
        raise RuntimeError('Constrained optimization not converged.')


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

# def cayley_map(X):
#     n = X.size(0)
#     Id = torch.eye(n, dtype=X.dtype, device=X.device)
#     return torch.solve(Id - X, Id + X)[0]


if __name__=='__main__':

    from pyqed import Molecule
    from pyqed.qchem.mcscf.casci import add_electric_field, add_vector_potential
    # from pyqed.qchem.mcscf.direct_ci import CASCI



    # atoms = ['N', 'C', 'C', 'N', 'C', 'C', 'H', 'H', 'H', 'H']

    # coords = np.array([
    #     [    0.0000000000,     0.0000000000,     1.5648929680],
    #     [   -0.0000000000,     1.0505512966,     0.7130576060],
    #     [    0.0000000000,     1.0505512966,    -0.7130576060],
    #     [    0.0000000000,     0.0000000000,    -1.5648929680],
    #     [    0.0000000000,    -1.0505512966,    -0.7130576060],
    #     [    0.0000000000,    -1.0505512966,     0.7130576060],
    #     [   -0.0000000000,     2.0619681303,     1.1514435560],
    #     [    0.0000000000,     2.0619681303,    -1.1514435560],
    #     [   -0.0000000000,    -2.0619681303,    -1.1514435560],
    #     [    0.0000000000,    -2.0619681303,     1.1514435560],
    # ])
    # BOHR = 1.8897261246257702
    # coords_bohr = coords * BOHR
    # atom_list = [[a, *r] for a, r in zip(atoms, coords_bohr)]

    # mol = Molecule(atom_list, basis='sto3g')



    mol = Molecule(atom='Li 0 0 0; H 0 0 1.4', unit='b', basis='sto3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()
    # hcore0 = mf.get_hcore()
    # # print('hcore0', hcore0)

    # # add electric field or vector potential
    # hcore_electric_field = add_electric_field(mol, electric_field=np.array([0,0,0]))
    # hcore_vector_potential = add_vector_potential(mol, vector_potential=np.array([0,0,1]))

    # mol.hcore = hcore0 + hcore_electric_field + hcore_vector_potential
    mf = mol.RHF().run()

    mc = CASSCF(mf, ncas=2, nelecas=2, max_cycles=50)

    nstates = 2


    mc.state_average(weights = np.ones(nstates)/nstates)
    mc.fix_spin(ss=0, shift=0.2)


    # print('casscf begin')
    mc.run(nstates=nstates)
    # print(mc.e_table)
    # num_iter = mc.num_iter
    # print('num of iter', num_iter)

    # mc.get_dip_moment(nstates=nstates,unit='Debye')
    # print('casscf dip', mc.dipole_moment)

    # # plot iteration-energy
    # iter = [i+1 for i in range(num_iter+1)]
    # print('iter', iter)
    # e0 = mc.e_table[0, 0:num_iter+1]
    # print('e0', e0)
    # e1 = mc.e_table[1, 0:num_iter+1]
    # print('e1', e1)
    # # e2 = mc.e_table[2, 0:num_iter+1]
    # # print('e2', e2)

    # plt.plot(iter, e0, 'r.-', alpha=0.5, linewidth = 1, label='gs')
    # plt.plot(iter, e1, 'b.-', alpha=0.5, linewidth = 1, label='es1')
    # # plt.plot(iter, e2, 'g.-', alpha=0.5, linewidth = 1, label='es2')
    # plt.legend()
    # plt.xlabel('number of iter')
    # plt.ylabel("energy(au)")
    # plt.tight_layout()
    # # plt.savefig("save/2026_01_12/test.png")
    # plt.savefig("save/2026_01_12/test(2,2).png")

    # plt.clf()

    # plt.plot(iter, e1, 'b.-', alpha=0.5, linewidth = 1, label='es')
    # plt.legend()
    # plt.xlabel('number of iter')
    # plt.ylabel("energy(au)")
    # plt.tight_layout()
    # plt.savefig("save/2026_01_12/LiF_es_cas(2,2)_sto3g.png")
    # # correct result is E(CASSCF) = [-7.67160344]