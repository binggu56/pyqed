#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:38:33 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:05:29 2022

@author: Bing
"""

import logging
import numpy as np
from scipy.linalg import eigh
from pyscf.scf import _vhf
import sys

from pyqed import dagger, dag
from pyqed.gto import RHF

# class RHF:
#     def __init__(self, mol, init_guess='h1e'):
#         self.mol = mol
#         self.max_cycle = 100
#         self.init_guess = init_guess

#         self.mo_occ = None
#         self.mo_coeff = None
#         self.e_tot = None
#         self.e_nuc = None
#         self.e_kin = None
#         self.e_ne = None
#         self.e_j = None
#         self.e_k = None

#     def run(self, **kwargs):
#         self.e_tot, self.e_nuc, self.mo_energy, self.mo_coeff, self.mo_occ =\
#             hartree_fock(self.mol, **kwargs)
#         return

#     def rdm1(self):
#         return make_rdm1(self.mo_coeff, mo_occ)

#     def get_ovlp(self):
#         pass

#     def get_fock(self):
#         pass




class UHF(RHF):
    def __init__(self, mol, init_guess='h1e', max_cycle=50):
        RHF.__init__(self, mol, init_guess=init_guess, max_cycle=max_cycle)

    def run(self):
        scf(self.max_cycle)

def get_hcore(mol):
    '''Core Hamiltonian
    Examples:
    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> scf.hf.get_hcore(mol)
    array([[-0.93767904, -0.59316327],
           [-0.59316327, -0.93767904]])

    From Pyscf.
    '''
    h = mol.intor_symmetric('int1e_kin')

    if mol._pseudo:
        # Although mol._pseudo for GTH PP is only available in Cell, GTH PP
        # may exist if mol is converted from cell object.
        from pyscf.gto import pp_int
        h += pp_int.get_gth_pp(mol)
    else:
        h+= mol.intor_symmetric('int1e_nuc')

    if len(mol._ecpbas) > 0:
        h += mol.intor_symmetric('ECPscalar')
    return h


def get_veff(mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
    '''Unrestricted Hartree-Fock potential matrix for the given density matrix
    
    .. math::

        V_{ij}^\alpha &= \sum_{kl} (ij|kl)(\gamma_{lk}^\alpha+\gamma_{lk}^\beta)
                       - \sum_{kl} (il|kj)\gamma_{lk}^\alpha \\
        V_{ij}^\beta  &= \sum_{kl} (ij|kl)(\gamma_{lk}^\alpha+\gamma_{lk}^\beta)
                       - \sum_{kl} (il|kj)\gamma_{lk}^\beta
                       
    Args:
        mol : an instance of :class:`Mole`
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices
    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference HF potential matrix.
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
    Returns:
        matrix Vhf = 2*J - K.  Vhf can be a list matrices, corresponding to the
        input density matrices.
    Examples:
    >>> import np
    >>> from pyscf import gto, scf
    >>> from pyscf.scf import _vhf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> dm0 = np.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> vhf0 = scf.hf.get_veff(mol, dm0, hermi=0)
    >>> dm1 = np.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> vhf1 = scf.hf.get_veff(mol, dm1, hermi=0)
    >>> vhf2 = scf.hf.get_veff(mol, dm1, dm_last=dm0, vhf_last=vhf0, hermi=0)
    >>> np.allclose(vhf1, vhf2)
    True
    '''
    if dm_last is None:
        vj, vk = get_jk(mol, np.asarray(dm), hermi, vhfopt)
        return vj - vk * .5
    else:
        ddm = np.asarray(dm) - np.asarray(dm_last)
        vj, vk = get_jk(mol, ddm, hermi, vhfopt)
        return vj - vk * .5 + np.asarray(vhf_last)


# def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None):
#     '''Compute J, K matrices for all input density matrices
#     Args:
#         mol : an instance of :class:`Mole`
#         dm : ndarray or list of ndarrays
#             A density matrix or a list of density matrices
#     Kwargs:
#         hermi : int
#             Whether J, K matrix is hermitian
#             | 0 : not hermitian and not symmetric
#             | 1 : hermitian or symmetric
#             | 2 : anti-hermitian
#         vhfopt :
#             A class which holds precomputed quantities to optimize the
#             computation of J, K matrices
#         with_j : boolean
#             Whether to compute J matrices
#         with_k : booleanzenm
#             Whether to compute K matrices
#         omega : float
#             Parameter of range-seperated Coulomb operator: erf( omega * r12 ) / r12.
#             If specified, integration are evaluated based on the long-range
#             part of the range-seperated Coulomb operator.
#     Returns:
#         Depending on the given dm, the function returns one J and one K matrix,
#         or a list of J matrices and a list of K matrices, corresponding to the
#         input density matrices.
#     Examples:
#     >>> from pyscf import gto, scf
#     >>> from pyscf.scf import _vhf
#     >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
#     >>> dms = np.random.random((3,mol.nao_nr(),mol.nao_nr()))
#     >>> j, k = scf.hf.get_jk(mol, dms, hermi=0)
#     >>> print(j.shape)
#     (3, 2, 2)
#     '''
#     dm = np.asarray(dm, order='C')
#     dm_shape = dm.shape
#     dm_dtype = dm.dtype
#     nao = dm_shape[-1]

#     if dm_dtype == np.complex128:
#         dm = np.vstack((dm.real, dm.imag)).reshape(-1,nao,nao)
#         hermi = 0

#     if omega is None:
#         vj, vk = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
#                              vhfopt, hermi, mol.cart, with_j, with_k)
#     else:
#         # The vhfopt of standard Coulomb operator can be used here as an approximate
#         # integral prescreening conditioner since long-range part Coulomb is always
#         # smaller than standard Coulomb.  It's safe to filter LR integrals with the
#         # integral estimation from standard Coulomb.
#         with mol.with_range_coulomb(omega):
#             vj, vk = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
#                                  vhfopt, hermi, mol.cart, with_j, with_k)

#     if dm_dtype == np.complex128:
#         if with_j:
#             vj = vj.reshape((2,) + dm_shape)
#             vj = vj[0] + vj[1] * 1j
#         if with_k:
#             vk = vk.reshape((2,) + dm_shape)
#             vk = vk[0] + vk[1] * 1j
#     else:
#         if with_j:
#             vj = vj.reshape(dm_shape)
#         if with_k:
#             vk = vk.reshape(dm_shape)
#     return vj, vk


def get_jk(g, dm):
    Da, Db = dm

    J = np.einsum("pqrs,rs->pq", g, Da + Db)
    Ka = np.einsum("prqs,rs->pq", g, Da)
    Kb = np.einsum("prqs,rs->pq", g, Db)
    return J, Ka, Kb
        
# def energy_elec(mf, dm=None, h1e=None, vhf=None):
#     '''Electronic energy of Unrestricted Hartree-Fock

#     Note this function has side effects which cause mf.scf_summary updated.

#     Returns:
#         Hartree-Fock electronic energy and the 2-electron part contribution
#     '''
#     if dm is None: dm = mf.make_rdm1()
#     if h1e is None:
#         h1e = mf.get_hcore()
    
#     if isinstance(dm, np.ndarray) and dm.ndim == 2:
#         dm = np.array((dm*.5, dm*.5))
        
#     if vhf is None:
#         vhf = mf.get_veff(mf.mol, dm)
        
#     if h1e[0].ndim < dm[0].ndim:  # get [0] because h1e and dm may not be ndarrays
#         h1e = (h1e, h1e)
    
#     e1 = np.einsum('ij,ji->', h1e[0], dm[0])
#     e1+= np.einsum('ij,ji->', h1e[1], dm[1])
#     e_coul =(np.einsum('ij,ji->', vhf[0], dm[0]) +
#              np.einsum('ij,ji->', vhf[1], dm[1])) * .5
#     e_elec = (e1 + e_coul).real
#     mf.scf_summary['e1'] = e1.real
#     mf.scf_summary['e2'] = e_coul.real
#     logging.debug(mf, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
#     return e_elec, e_coul


def make_rdm1(mo_coeff, mo_occ, **kwargs):
    '''One-particle density matrix in AO representation
    Args:
        mo_coeff : 2D ndarray
            Orbital coefficients. Each column is one orbital.
        mo_occ : 1D ndarray
            Occupancy
    Returns:
        One-particle density matrix, 2D ndarray
    '''
    mocc = mo_coeff[:,mo_occ>0]
# DO NOT make tag_array for dm1 here because this DM array may be modified and
# passed to functions like get_jk, get_vxc.  These functions may take the tags
# (mo_coeff, mo_occ) to compute the potential if tags were found in the DM
# array and modifications to DM array may be ignored.
    return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)




def hartree_fock(mol, max_cycle=50):

    #print("constructing basis set")

    # phi = [0] * len(Z)

    # for A in range(len(Z)):

    #     if Z[A] == 1:

    #         phi[A] = sto3g_hydrogen(R[A])

    #     elif Z[A] == 2:

    #         phi[A] = sto3g_helium(R[A])

    # total number of STOs
    # K = len(phi)

    #calculate the overlap matrix S
    #the matrix should be symmetric with diagonal entries equal to one
    logging.info("building overlap matrix")

    S = mol.intor_symmetric('int1e_ovlp')

    # for i in range(len(phi)):
    #     for j in range( (i+1),len(phi)):
    #         S[i,j] = S[j,i] = overlap_integral_sto(phi[i], phi[j])

    # print("S: ", S)


    # #calculate the kinetic energy matrix T
    # print("building kinetic energy matrix")
    # T = np.zeros((K,K))

    # #print('test', phi[0].g[0].center)
    # #print('test', phi[1].g[1].center)

    # for i in range(len(phi)):
    #     for j in range(i, len(phi)):
    #         T[i,j] = T[j,i] = kinetic_energy_integral(phi[i], phi[j])


    # #print("building nuclear attraction matrices")

    # V = np.zeros((K,K))

    # for A in range(K):
    #     for i in range(K):
    #         for j in range(i,K):
    #             v = nuclear_attraction_integral(Z[A], R[A], phi[i], phi[j])
    #             V[i,j] += v
    #             if i != j:
    #                 V[j,i] += v
    # #print("V: ", V)

    # #build core-Hamiltonian matrix
    # #print("building core-Hamiltonian matrix")
    # Hcore = T + V

    hcore = get_hcore(mol)

    # print("Hcore: ", Hcore)

    #diagonalize overlap matrix to get transformation matrix X
    #print("diagonalizing overlap matrix")
    s, U = eigh(S)
    #print("building transformation matrix")
    X = U.dot(np.diagflat(s**(-0.5)).dot(dagger(U)))


    #calculate all of the two-electron integrals
    #print("building two_electron Coulomb and exchange integrals")

    # two_electron = np.zeros((K,K,K,K))

    # for mu in range(K):
    #     for v in range(K):
    #         for lamb in range(K):
    #             for sigma in range(K):
    #                 two_electron[mu,v,sigma,lamb] = \
    #                     two_electron_integral(phi[mu], phi[v], phi[sigma], phi[lamb])

#                    coulomb  = two_electron_integral(phi[mu], phi[v], \
#                                                     phi[sigma], phi[lamb])
#                    two_electron[mu,v,sigma,lamb] = coulomb
                    #print("coulomb  ( ", mu, v, '|', sigma, lamb,"): ",coulomb)
#                    exchange = two_electron_integral(phi[mu], phi[lamb], \
#                                                     phi[sigma], phi[v])
#                    #print("exchange ( ", mu, lamb, '|', sigma, v, "): ",exchange)
#                    two_electron[mu,lamb,sigma,v] = exchange

    # P = np.zeros((K,K))

    total_energy = 0.0
    old_energy = 0.0
    electronic_energy = 0.0

    nocc = mol.nelectron // 2
    mo_occ = np.zeros(mol.nao)
    mo_occ[:nocc] = 2

    # dm = init_guess_by_hcore(hcore)
    def init_guess_by_h1e(h):
        h = dag(X) @ h @ X
        mo_energy, C = eigh(h)
        return make_rdm1(C, mo_occ)

    dm = init_guess_by_h1e(hcore)

    # nuclear energy
    # nuclear_energy = 0.0
    # for A in range(len(Z)):
    #     for B in range(A+1,len(Z)):
    #         nuclear_energy += Z[A]*Z[B]/abs(R[A]-R[B])

    nuclear_energy = mol.energy_nuc()

    print("E_nclr = ", nuclear_energy)

    print("\n {:4s} {:13s} de\n".format("iter", "total energy"))

    conv = False
    for scf_iter in range(max_cycle):

        # #calculate the two electron part of the Fock matrix

        vhf = get_veff(mol, dm)
        F = hcore + vhf

        electronic_energy = energy_elec(dm, hcore, vhf)


        #print("E_elec = ", electronic_energy)

        total_energy = electronic_energy + nuclear_energy

        print("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
               total_energy - old_energy))

        if scf_iter > 2 and abs(old_energy - total_energy) < 1e-6:
            conv = True
            break

        #println("F: ", F)
        #Fprime = X' * F * X
        Fprime = dagger(X).dot(F).dot(X)
        #println("F': $Fprime")
        mo_energy, Cprime = eigh(Fprime)
        # print("epsilon: ", epsilon)
        #print("C': ", Cprime)
        mo_coeff = np.real(np.dot(X,Cprime))
        # print("C: ", C)


        # new density matrix in original basis
        # P = np.zeros(Hcore.shape)
        # for mu in range(len(phi)):
        #     for v in range(len(phi)):
        #         P[mu,v] = 2. * C[mu,0] * C[v,0]
        dm = make_rdm1(mo_coeff, mo_occ)

        old_energy = total_energy

    if not conv: sys.exit('SCF not converged.')

    print('HF energy = ', total_energy)

    # check if this hartree-fock calculation is for configuration interaction
    # or not, if yes, output the essential information
    # if CI == False:
    return total_energy, nuclear_energy, mo_energy, mo_coeff, mo_occ
    # else:
    # return C, Hcore, nuclear_energy, two_electron



def energy_elec(D, h1e, fock):
    Da, Db = D
    Fa, Fb = fock
    return 0.5 * np.sum( h1e * (Da + Db) + Da * Fa + Db * Fb)


def unrestricted_hartree_fock(mol, max_cycle=50, e_conv = 1e-8, d_conv = 1e-8):

    H = mol.get_hcore()
    S = mol.intor_symmetric('cint1e_ovlp_sph') 
        
    #Get two-e integral (TEI) from PySCF format and convert it to chemist format 
    g = mol.intor('cint2e_sph')

    
    A = S
    A = sp.linalg.fractional_matrix_power(A, -0.5)
    
    # print(A @ S @ A)
    
    
    # Diagonalize Core H
    def diag(F, A):
        Fp = reduce(np.dot, (A.T, F, A))
        eps, Cp = np.linalg.eigh(Fp)
        C = A.dot(Cp)
        return eps, C
    
    
    eps_a, C_a = diag(H, A)
    eps_b, C_b = diag(H, A)
    
    Cocc_a, Cocc_b = C_a[:, :nel_a], C_b[:, :nel_b]
    D_a, D_b = Cocc_a.dot(Cocc_a.T), Cocc_b.dot(Cocc_b.T)

    #SCF procedure
    E_old = 0.0
    Fa_old = None
    Fb_old = None
    maxcyc = None
    
    for iteration in range(ncycl):
    

        J, Ka, Kb = get_jk(g, [D_a, D_b])
        
        Fa_new = H + J - Ka
        Fb_new = H + J - Kb
        
    	
        # conditional iteration > start_damp
        if iteration >= damp_start:
            Fa = damp_value * Fa_old + (1.0 - damp_value) * Fa_new
            Fb = damp_value * Fb_old + (1.0 - damp_value) * Fb_new		
        else:
            Fa = Fa_new
            Fb = Fb_new
    		
        Fa_old = Fa_new
        Fb_old = Fb_new	
    	
    
        # Build the AO gradient
        grada = reduce(np.dot,(Fa, D_a, S)) - reduce(np.dot, (S, D_a, Fa))
        grada_rms = np.mean(grada ** 2) ** 0.5
    
        gradb = reduce(np.dot,(Fb, D_b, S)) - reduce(np.dot, (S, D_b, Fb))
        gradb_rms = np.mean(gradb ** 2) ** 0.5
        grad_rms = 	grada_rms + gradb_rms
    	
        # Build the energy
        E_electric = energy_elec([D_a, D_b], H, [Fa, Fb])
        
        E_total = E_electric + mol.energy_nuc()
    
        E_diff = E_total - E_old
        E_old = E_total
        
        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
                (iteration, E_total, E_diff, grad_rms))
    
        # Break if e_conv and d_conv are met
        if (abs(E_diff) < e_conv) and (grad_rms < d_conv):
            maxcyc = iteration
            break
    		
        if (iteration == ncycl - 1):
            maxcyc = ncycl
    
        eps_a, C_a = diag(Fa, A)
        eps_b, C_b = diag(Fb, A)
        
        Cocc_a, Cocc_b = C_a[:, :nel_a], C_b[:, :nel_b]
        D_a, D_b = Cocc_a.dot(Cocc_a.T), Cocc_b.dot(Cocc_b.T)
    
    if (maxcyc == ncycl):
        	print("SCF has not finished!\n")
    else:
        	print("SCF has finished!\n")
    
    return E_total
        


import unittest
class TestUHF(unittest.TestCase):

    def test(self):
        #PYSCF solution
        '''
        Note: PySCF uses a combination of many SCF techniques, this include DIIS, generating initial guess 1-rdm 
        based on ANO basis'minao',...
        For RHF solution, usually there is no difference when using/not using these techniques.
        However, this is not the case for UHF solution. To compare with simple algorithm UHF above, I used damping technque
        for both codes. This is not the best technique for UHF convergences
        
        Will updated the DIIS in this SCF
        '''

        

        E_total = 0
        
        mf = scf.UHF(mol)
        # mf.init_guess = 'miao'
        mf.max_cycle = 100
        mf.diis_start_cycle=200
        mf.diis = False
        mf.damp = damp_value #using damping instead of DIIS
        mf.kernel()
        
        
        pyscf_energy = mf.e_tot
        # print("Energy matches PySCF %s" % np.allclose(pyscf_energy, E_total))
                               
        self.assertEqual(pyscf_energy, E_total)
        
        
        
if __name__ == '__main__':
    # from pyscf import gto, scf
    # mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    # conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))
    # print('conv = %s, E(HF) = %.12f' % (conv, e))
    # # conv = True, E(HF) = -1.081170784378

    # # hartree_fock(mol)
    # hf = RHF(mol)
    # hf.run()
    # print(hf.e_tot)


    '''
    Refs 
        https://github.com/hungpham2017/Hartree-Fock/blob/master/Python/uhf_pyscf.py

    Unrestricted Hartree-Fock using PySCF intergrals
    Author: Hung Q. Pham (UMN, phamx494@umn.edu)
    
    '''
    
    import numpy as np
    import scipy as sp
    from pyscf import gto, scf, ao2mo
    from functools import reduce
    
    mol = gto.M()
    mol.atom =("""
     O                 -1.82412712    0.02352916    0.00000000
     O                 -0.72539390    0.07630510    0.00000000
     N                 -2.14114302    1.07685766    0.00000000
    """)
    
    mol.spin = 1
    mol.basis = '6-31g'
    mol.build()
    Norb = mol.nao_nr()
    print(Norb)
    
    #UHF convergent criteria

    if (mol.nelectron %2 != 0):
    	nel_a = int(0.5 * (mol.nelectron + 1))
    	nel_b = int(0.5 * (mol.nelectron - 1))
    else:
    	nel_a = int(0.5 * mol.nelectron)
    	nel_b = nel_a
    
    damp_value = 0.20
    damp_start = 1
    ncycl = 100

    

    
    E_total = unrestricted_hartree_fock(mol)
    
    mf = scf.UHF(mol)
    # mf.init_guess = 'miao'
    mf.max_cycle = 100
    mf.diis_start_cycle=200
    mf.diis = False
    mf.damp = damp_value #using damping instead of DIIS
    mf.kernel()
    
    
    pyscf_energy = mf.e_tot
    print("Energy matches PySCF %s" % np.allclose(pyscf_energy, E_total))

        
    	

