#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:04:50 2024

@author: bingg
"""
import numpy as np
import scipy.constants as const
import scipy.linalg as la
import scipy 
from scipy.sparse import identity, kron, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import kron, norm, eigh

from scipy.special import erf
from scipy.constants import e, epsilon_0, hbar
import logging
import warnings

from pyqed import discretize, sort, dag
from pyqed.davidson import davidson
from pyqed.ldr.ldr import kinetic
from pyqed import au2ev, au2angstrom
from pyqed.dvr import SineDVR
from pyqed import scf
from pyqed.scf import make_rdm1, energy_elec
# from pyqed.jordan_wigner import jordan_wigner_one_body, jordan_wigner_two_body

from pyqed.ci.fci import SpinOuterProduct, givenΛgetB

from numba import vectorize, float64, jit
import sys
from opt_einsum import contract
from itertools import combinations


def soft_coulomb(r, R=1):
    if np.isclose(r, 0):
            # if r_R_distance == 0:
        return 2 / (R * np.sqrt(np.pi))
    else:
        return erf(r / R) / r
    
class RKS:
    """
    restricited DVR-HF method in 1D 
    """
    def __init__(self, mol, init_guess='hcore', dvr_type = 'sine'): # nelec, spin):
        # self.spin = spin 
        # self.nelec = nelec
        self.mol = mol        
    
        self.T = None
        self.hcore = None 
        self.fock = None
        
        self.mol = mol
        self.max_cycle = 100
        self.tol = 1e-6
        self.init_guess = init_guess

        ###
        self.nx = self.mol.nx 
        self.x = self.mol.x
        
        self.mo_occ = None
        self.mo_coeff = None
        self.e_tot = None
        self.e_nuc = None
        self.e_kin = None
        self.e_ne = None
        self.e_j = None
        self.e_k = None
        
        self.eri = None
        
    def create_grid(self, domain, level):
            
        x = discretize(domain, level, endpoints=False)
        
        self.x = x 
        self.nx = len(x)

        self.lx = domain[1]-domain[0]
        # self.dx = self.lx / (self.nx - 1)

        self.domain = domain
    
    def get_eri(self):
        """
        electronc repulsion integral in DVR basis

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        nx = self.nx 
        x = self.x
        
        v = soft_coulomb(0, self.mol.Re) * np.eye(nx)
        for i in range(nx):
            for j in range(i):
                d = np.linalg.norm(x[i] - x[j])
                v[i,j] = soft_coulomb(d, self.mol.Re)
                v[j,i] = v[i,j]

        self.eri = v
        return v

    def get_veff(self, dm):
        """
        compute Hartree and Fock potential

        Parameters
        ----------
        dm : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # hartree potential        
        J = contract('ij, jj -> i', self.eri, dm)
        J = np.diag(J)
        
        # exchange 
        K = self.eri * dm 
        
        vHF = J - 0.5 * K
        
        return vHF
    
    def get_hcore(self):
        return self.mol.get_hcore()
        
    def run(self, R):
        # scf cycle
        max_cycle = self.max_cycle
        tol = self.tol
        
        mol = self.mol
        
        # Hcore (kinetic + v_en)
        hcore = mol.get_hcore(R)
        self.hcore = hcore 
        
        # occ number
        nocc = self.mol.nelectron // 2
        mo_occ = np.zeros(self.nx)
        mo_occ[:nocc] = 2
        
        self.mo_occ = np.stack([mo_occ, mo_occ])
        print('mo_occ', self.mo_occ)
        
        self.get_eri()
        
        if self.init_guess == 'hcore':

            mo_energy, mo_coeff = eigh(hcore)
            dm = make_rdm1(mo_coeff, mo_occ)
        
            
            vhf = self.get_veff(dm)            
            old_energy = energy_elec(dm, hcore, vhf)
            
        print("\n {:4s} {:13s} de\n".format("iter", "total energy"))

        nuclear_energy = mol.energy_nuc(R)
        
        conv = False
        for scf_iter in range(max_cycle):

            # calculate the two electron part of the Fock matrix

            vhf = self.get_veff(dm)
            F = hcore + vhf


            mo_energy, mo_coeff = eigh(F)
            # print("epsilon: ", epsilon)
            #print("C': ", Cprime)
            # mo_coeff = C
            # print("C: ", C)


            # new density matrix in original basis
            # P = np.zeros(Hcore.shape)
            # for mu in range(len(phi)):
            #     for v in range(len(phi)):
            #         P[mu,v] = 2. * C[mu,0] * C[v,0]
            dm = make_rdm1(mo_coeff, mo_occ)

            electronic_energy = energy_elec(dm, hcore, vhf)

            

            print("E_elec = ", electronic_energy)

            total_energy = electronic_energy + nuclear_energy

            logging.info("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
                   total_energy - old_energy))

            if scf_iter > 2 and abs(old_energy - total_energy) < tol:
                conv = True
                print('SCF Converged.')
                break
            
            old_energy = total_energy


            #println("F: ", F)
            #Fprime = X' * F * X
            # Fprime = dagger(X).dot(F).dot(X)
            #println("F': $Fprime")
            
        self.mo_coeff = mo_coeff
        self.mo_energy = mo_energy


        if not conv: sys.exit('SCF not converged.')

        print('HF energy = ', total_energy)
        
        return total_energy
    
    # def energy_elec(dm, h1e=None, vhf=None):
    #     r'''
    #     Electronic part of Hartree-Fock energy, for given core hamiltonian and
    #     HF potential
        
    #     ... math::
    #         E = \sum_{ij}h_{ij} \gamma_{ji}
    #           + \frac{1}{2}\sum_{ijkl} \gamma_{ji}\gamma_{lk} \langle ik||jl\rangle
              
    #     Note this function has side effects which cause mf.scf_summary updated.
    #     Args:
    #         mf : an instance of SCF class
    #     Kwargs:
    #         dm : 2D ndarray
    #             one-partical density matrix
    #         h1e : 2D ndarray
    #             Core hamiltonian
    #         vhf : 2D ndarray
    #             HF potential
    #     Returns:
    #         Hartree-Fock electronic energy and the Coulomb energy
    #     Examples:
    #     >>> from pyscf import gto, scf
    #     >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    #     >>> mf = scf.RHF(mol)
    #     >>> mf.scf()
    #     >>> dm = mf.make_rdm1()
    #     >>> scf.hf.energy_elec(mf, dm)
    #     (-1.5176090667746334, 0.60917167853723675)
    #     >>> mf.energy_elec(dm)
    #     (-1.5176090667746334, 0.60917167853723675)
    #     '''
    #     # if dm is None: dm = mf.make_rdm1()
    #     # if h1e is None: h1e = mf.get_hcore()
    #     # if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    #     e1 = np.einsum('ij,ji->', h1e, dm).real
    #     e_coul = np.einsum('ij,ji->', vhf, dm).real * .5
    #     # mf.scf_summary['e1'] = e1
    #     # mf.scf_summary['e2'] = e_coul
    #     # logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    #     return e1+e_coul #, e_coul


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

class RHF2D(scf.RHF):
    """
    restricited DVR-HF method in 2D 
    """
    def __init__(self, mol, init_guess='hcore', dvr_type = 'sine'): # nelec, spin):
        # self.spin = spin 
        # self.nelec = nelec
        self.mol = mol        
    
        self.T = None
        self.hcore = None 
        self.fock = None
        
        self.mol = mol
        self.max_cycle = 100
        self.tol = 1e-6
        self.init_guess = init_guess

        self.mo_occ = None
        self.mo_coeff = None
        self.e_tot = None
        self.e_nuc = None
        self.e_kin = None
        self.e_ne = None
        self.e_j = None
        self.e_k = None
        
    def create_grid(self, domains, levels):
            
        x = discretize(*domains[0], levels[0], endpoints=False)
        y = discretize(*domains[0], levels[1], endpoints=False)
        
        self.x = x 
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.lx = domains[0][1]-domains[0][0]
        self.ly = domains[0][1]-domains[0][0]
        # self.dx = self.lx / (self.nx - 1)
        # self.dy = self.ly / (self.ny - 1)
        self.domains = domains
            
    def run(self, init_guess='hcore'):
        # scf cycle
        max_cycle = self.max_cycle
        tol = self.tol
        
        conv = False
        for scf_iter in range(max_cycle):

            # calculate the two electron part of the Fock matrix

            vhf = get_veff(mol, dm)
            F = hcore + vhf

            mo_energy, mo_coeff = eigh(F)

            dm = make_rdm1(mo_coeff, mo_occ)

            electronic_energy = energy_elec(dm, hcore, vhf)


            #print("E_elec = ", electronic_energy)

            total_energy = electronic_energy + nuclear_energy

            logging.info("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
                   total_energy - old_energy))

            if scf_iter > 2 and abs(old_energy - total_energy) < tol:
                conv = True
                self.mo_coeff = mo_eff
                self.mo_energy = mo_energy
                break

            #println("F: ", F)
            #Fprime = X' * F * X
            # Fprime = dagger(X).dot(F).dot(X)
            #println("F': $Fprime")

            # print("epsilon: ", epsilon)
            #print("C': ", Cprime)
            # mo_coeff = C
            # print("C: ", C)


            # new density matrix in original basis
            # P = np.zeros(Hcore.shape)
            # for mu in range(len(phi)):
            #     for v in range(len(phi)):
            #         P[mu,v] = 2. * C[mu,0] * C[v,0]

            old_energy = total_energy

        if not conv: sys.exit('SCF not converged.')

        print('HF energy = ', total_energy)
        
        return 
    
    def get_veff(self):
        pass





        
    
    # def jordan_wigner(self):
    #     # an inefficient implementation without consdiering any syemmetry 
        
    #     # transform the Hamiltonian in DVR set to (truncated) MOs 
    #     nmo = self.ncas
    #     mf = self.mf 
        
    #     mo = mf.mo_coeff[:self.ncas]
        
    #     # single electron part
    #     hcore_mo = contract('ia, ij, jb -> ab', mo.conj(), mf.hcore, mo)
        
    #     self.hcore_mo = hcore_mo
        
    #     eri = self.mf.eri
    #     eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

    #     H = 0        
    #     for p in range(nmo):
    #         for q in range(p+1):
    #             H += jordan_wigner_one_body(p, q, hcore_mo[p, q], hc=True)
                
    #     for p in range(nmo):
    #         for q in range(nmo):
    #             for r in range(nmo):
    #                 for s in range(nmo):
    #                     H += jordan_wigner_two_body(p, q, s, r, 0.5*eri_mo[p, q, r, s])
    
    #     return H
    
    
    def run(self, nstates=3):
        from pyqed.ci.fci import SlaterCondon, CI_H
        
        mf = self.mf 
        ncas = self.ncas 
        
        mo_occ = mf.mo_occ[:, :ncas]/2
        
        mf.mo_coeff = mf.mo_coeff[:, :ncas]
        
        Binary = get_fci_combos(mo_occ)
        print('Binary shape', Binary.shape)
        
        H1, H2 = self.get_SO_matrix(mf)
        SC1, SC2 = SlaterCondon(Binary)
        H_CI = CI_H(Binary, H1, H2, SC1, SC2)
        
        print('HCI', H_CI)
        
        # E, X = np.linalg.eigh(H_CI)
        E, X = eigsh(H_CI, k=nstates, which='SA')
        return E, X
        
def get_fci_combos(mo_occ):
    # print(mf.mo_occ.shape)
    O_sp = np.asarray(mo_occ, dtype=np.int8)
    
    # number of electrons for each spin 
    N_s = np.einsum("sp -> s", O_sp)
    
    N = O_sp.shape[1]
    Λ_α = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[0] ) ) )
    Λ_β = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[1] ) ) )
    ΛA, ΛB = SpinOuterProduct(Λ_α, Λ_β)
    Binary = givenΛgetB(ΛA, ΛB, N)
    return Binary


class RCISD:
    def __init__(self, mf):
        pass
    
    def buildH(self):
        # build the CI H in determinants {0, ia, ijab}
        pass
        
    def run(self):
        pass

class UCISD(RCISD):
    def __init__(self, mf):
        pass