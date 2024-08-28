#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:52:48 2024

@author: xiaozhu
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

from pyqed import au2ev, au2angstrom, fine_structure, pauli
from pyqed.dvr import SineDVR
# from pyqed import scf
# from pyqed.scf import make_rdm1, energy_elec
# from pyqed.jordan_wigner import jordan_wigner_one_body, jordan_wigner_two_body

# from pyqed.ci.fci import SpinOuterProduct, givenΛgetB

# from numba import vectorize, float64, jit
# import sys
# from opt_einsum import contract
# from itertools import combinations

# @vectorize([float64(float64, float64)])
# @vectorize


def soft_coulomb(r, R=1):
    if np.isclose(r, 0):
            # if r_R_distance == 0:
        return 2 / (R * np.sqrt(np.pi))
    else:
        return erf(r / R) / r

def force(r, R=1):
    
    if np.isclose(r, 0):
        return 0
    else:
        return ((2 * np.exp(-r**2/R**2))/(np.sqrt(np.pi) * R) - erf(r/R))/r**2

soft_coulomb = np.vectorize(soft_coulomb)



        
def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None):
    '''Compute J, K matrices for all input density matrices
    Args:
        mol : an instance of :class:`Mole`
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices
    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : not hermitian and not symmetric
            | 1 : hermitian or symmetric
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
        with_j : boolean
            Whether to compute J matrices
        with_k : boolean
            Whether to compute K matrices
        omega : float
            Parameter of range-seperated Coulomb operator: erf( omega * r12 ) / r12.
            If specified, integration are evaluated based on the long-range
            part of the range-seperated Coulomb operator.
    Returns:
        Depending on the given dm, the function returns one J and one K matrix,
        or a list of J matrices and a list of K matrices, corresponding to the
        input density matrices.
    Examples:
    >>> from pyscf import gto, scf
    >>> from pyscf.scf import _vhf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> dms = np.random.random((3,mol.nao_nr(),mol.nao_nr()))
    >>> j, k = scf.hf.get_jk(mol, dms, hermi=0)
    >>> print(j.shape)
    (3, 2, 2)
    '''
    dm = np.asarray(dm, order='C')
    dm_shape = dm.shape
    dm_dtype = dm.dtype
    nao = dm_shape[-1]

    if dm_dtype == np.complex128:
        dm = np.vstack((dm.real, dm.imag)).reshape(-1,nao,nao)
        hermi = 0

    if omega is None:
        vj, vk = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
                             vhfopt, hermi, mol.cart, with_j, with_k)
    else:
        # The vhfopt of standard Coulomb operator can be used here as an approximate
        # integral prescreening conditioner since long-range part Coulomb is always
        # smaller than standard Coulomb.  It's safe to filter LR integrals with the
        # integral estimation from standard Coulomb.
        with mol.with_range_coulomb(omega):
            vj, vk = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
                                 vhfopt, hermi, mol.cart, with_j, with_k)

    if dm_dtype == np.complex128:
        if with_j:
            vj = vj.reshape((2,) + dm_shape)
            vj = vj[0] + vj[1] * 1j
        if with_k:
            vk = vk.reshape((2,) + dm_shape)
            vk = vk[0] + vk[1] * 1j
    else:
        if with_j:
            vj = vj.reshape(dm_shape)
        if with_k:
            vk = vk.reshape(dm_shape)
    return vj, vk

def get_veff(mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
    '''Hartree-Fock potential matrix for the given density matrix
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






class ShinMetiu1d:
    """
    Shin-Metiu model with N electrons in 1D
    
    Refs 
        
    """
    
    # def __init__(self, method = 'scipy', nstates=4, dvr_type='sinc', mass=1836, spin_state ='antiparallel'):
    def __init__(self, method = 'scipy', nstates=3, nelec=2, dvr_type='sine', spin=0):
        
        self.Rc = 1.5/au2angstrom  # Adjustable parameter in the pseudopotential
        self.Rf = 1.5/au2angstrom  # Adjustable parameter in the pseudopotential
        if spin == 0:
            self.Re = 2.5/au2angstrom  # Adjustable parameter in the pseudopotential
        elif spin == 1:
            self.Re = 1.5/au2angstrom
        else: print('Missing spin_state parameter')
        
        
        # self.R = R # proton position
        
        
        self.domain = None
        
        self.spin = spin
        self.nelec = self.nelectron = nelec
        self.Z = [1, 1, 1]     # Ion charge
        self.e = -1     # Electron charge, should be set to actual value in atomic units
        
        self.L = 10/au2angstrom
        # print(self.L)
        # self.mass = mass  # nuclear mass
        self.left = np.array([-self.L/2])
        self.right = np.array([self.L/2])
    
        # self.left = np.array([-self.L/2, 0])
        # self.right = np.array([self.L/2, 0])
    
        self.x = None
        self.y = None
        self.nx = None
        self.ny = None
        self.u = None
        self.X = None
        self.Y = None
        
        self.dvr_type = dvr_type
        
        self.method = method
        self.v0 = None 
        # self.nv = 4 # davision’s default number of feature vectors is 4
        self.nstates = nstates

        self.e_nuc = None
    

        
        
    def create_grid(self, domain, level):
        
        x = discretize(*domain, level, endpoints=False)
        y = x.copy() 
        
        self.x = x 
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        # self.lx = domain[0][1]-domain[0][0]
        # self.ly = domain[0][1]-domain[0][0]
        # self.dx = self.lx / (self.nx - 1)
        # self.dy = self.ly / (self.ny - 1)
        self.domain = domain 
        self.level = level
    
    def get_hcore(self, R=0):
        """
        single point calculations

        Parameters
        ----------
        R : float
            proton position.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        w : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.

        """
        
        # H(r; R)
        x = self.x
        nx = self.nx

        
        # T 
        # origin method of calculate kinetic term
        # tx = kinetic(x, dvr=self.dvr_type)
        # idx = np.eye(nx)  
        
        # ty = kinetic(y, dvr=self.dvr_type)
        # idy = np.eye(ny)
        
        # T = kron(tx, idy) + kron(idx, ty)
        
        # # # new method of calculate kinetic term
        dvr_x = SineDVR(*self.domain, nx)
        
        # tx = kinetic(self.x, dvr=self.dvr_type)
        T = dvr_x.t()
    
        self.T = T
        
        
        # V
        Ra = self.left
        Rb = self.right 
        v = np.zeros((nx))
        for i in range(nx):   
            r1 = np.array(x[i])
            # Potential from all ions
            v[i] = self.V_en(r1, Ra) + self.V_en(r1, Rb) + self.V_en1(r1, R)
        
        V = np.diag(v)
        # print(V.shape)
        
        # v_sym = self.enforce_spin_symmetry(v)
        # # print(v_sym.shape)
        # V = np.diag(v_sym.ravel())
        
        H = T + V 
        # H = self.imaginary_time_propagation(H)
        
        # if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        #     raise ValueError("H matrix contains NaNs or infs.")
        
        return H
    
    def single_point(self, R):
        """
        single point calculations

        Parameters
        ----------
        R : float
            proton position.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        w : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.

        """
        
        # H(r; R)
        x = self.x
        y = self.y
        nx = self.nx
        ny = self.ny
        
        # T 
        # origin method of calculate kinetic term
        # tx = kinetic(x, dvr=self.dvr_type)
        # idx = np.eye(nx)  
        
        # ty = kinetic(y, dvr=self.dvr_type)
        # idy = np.eye(ny)
        
        # T = kron(tx, idy) + kron(idx, ty)
        
        # # # new method of calculate kinetic term
        dvr_x = SineDVR(*self.domain, nx)
        tx = dvr_x.t()
        idx = np.eye(self.nx)
        
        T = kron(tx, idx) + kron(idx, tx)
        self.T = T
        
        
        # V
        v = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                r1 = np.array([x[i]])
                r2 = np.array([y[j]])
                v[i, j] = self.potential_energy(r1, r2, R)
    
        V = np.diag(v.ravel())
        # print(V.shape)
        
        # v_sym = self.enforce_spin_symmetry(v)
        # # print(v_sym.shape)
        # V = np.diag(v_sym.ravel())
        
        H = T + V 
        # H = self.imaginary_time_propagation(H)
        
        # if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        #     raise ValueError("H matrix contains NaNs or infs.")
        
        if self.method == 'exact':
            w, u = eigh(H)
        elif self.method == 'davidson':
            w, u = davidson(H, neigen=self.nstates)
        elif self.method == 'scipy':
            w, u = scipy.sparse.linalg.eigsh(csr_matrix(H), k=self.nstates, which='SA', v0=self.v0)
            self.v0 = u[:,0] # store the eigenvectors for next calculation
        else:
            raise ValueError("Invalid method specified")
        
        # u_sym = self.enforce_spin_symmetry(np.outer(u, u))
        return w, u
    
    def V_en(self, r, R):
        """
        Electron-nucleus interaction potential.
        """
        
        d = np.linalg.norm(r - R)
        
        # ze2 = self.Z * self.e**2
        return -soft_coulomb(d, self.Rf)
        # return -ze2 * erf(np.linalg.norm(r - R) / self.Rc) / np.linalg.norm(r - R)
        
    def V_en1(self, r, R):
        """
        Electron-nucleus interaction potential.
        """
       
        d = np.linalg.norm(r - R)
        
        return -soft_coulomb(d, self.Rc)

    def V_nn(self, R1, R2):
        """
        Nucleus-nucleus interaction potential.
        """
        
        R2_R1_distance = np.linalg.norm(R2 - R1)
        
        if R2_R1_distance == 0:
            return 0
        # return self.e**2 / np.linalg.norm(R2 - R1)
        return 1 / np.linalg.norm(R2 - R1)
    
    def V_ee(self, r1, r2):
        """
        Electron-electron interaction potential.
        """
        d = np.linalg.norm(r1 - r2)

        return soft_coulomb(d, self.Re)

    def energy_nuc(self, R):   
        Ra = self.left
        Rb = self.right 
        return self.V_nn(R, Ra) + self.V_nn(R, Rb) 

    
    def potential_energy(self, r1, r2, R):
        """
        Calculate the electron-nuclear energy V(x, y) on a grid.
        """     
        # Convert R from atomic units to Ångström for calculations
        Ra = self.left
        Rb = self.right 
        
        # Potential from all ions
        v = self.V_en(r1, Ra) + self.V_en(r1, Rb) + self.V_en1(r1, R) +\
            self.V_en(r2, Ra) + self.V_en(r2, Rb) + self.V_en1(r2, R)

        # nuclei-nuclei interaction
        v_nn = self.V_nn(R, Ra) + self.V_nn(R, Rb) + self.V_nn(Ra, Rb)
        v_ee = self.V_ee(r1, r2)
        
        v += v_nn + v_ee        
        self.e_nuc = v_nn 
        return v
      

    def pes(self, domain=[-2,2], level=5):
        
        # calc PES
        # L = self.L 
        X = discretize(*domain, level) #endpoints=False)
        E = np.zeros((len(X), self.nstates))
        U = np.zeros((len(X), self.nx, self.ny, self.nstates))
        
        print('Scanning the APES')
        for k in range(len(X)):
            
            R = [X[k]]
            # print(R.shape)
            # print(R)
            w, u = self.single_point(R)
            # w, u = sort(*self.single_point(R))
            # print(w.shape)
            # print(u.shape)
            # print(u[:, :self.nstates].shape)
            E[k, :] = w[:self.nstates]
            U[k] = u[:, :self.nstates].reshape(self.nx, self.ny, self.nstates)
            # U[i] = u[:, :self.nstates].reshape(self.nx, self.ny, self.nstates)
            # print(u[:, :self.nstates].shape)
        
        self.u = U  
        self.X = X     
        
        # output_messages = []
        
        # # Enforce symmetry or antisymmetry based on spin state
        # for state in range(self.nstates):
        #     for ix in range(self.nx):
        #         for iy in range(ix, self.ny):  # Only need to loop over half the matrix due to symmetry
        #             if self.spin_state == 'parallel':
        #                 # Antisymmetric spatial part for parallel spins
        #                 if not np.allclose(U[:, ix, iy, state], -U[:, iy, ix, state], equal_nan=True):
        #                     E[:, state] = np.inf
        #                     output_messages.append(f'State {state}, indices ({ix}, {iy}): Not antisymmetric (not equal to negative)')
        #                 else:
        #                     output_messages.append(f'State {state}, indices ({ix}, {iy}): Antisymmetric (equal to negative)')
        #             elif self.spin_state == 'antiparallel':
        #                 # Symmetric spatial part for antiparallel spins
        #                 if not np.allclose(U[:, ix, iy, state], U[:, iy, ix, state]):
        #                     E[:, state] = np.inf
        #                     output_messages.append(f'State {state}, indices ({ix}, {iy}): Not symmetric (not equal)')
        #                 else:
        #                     output_messages.append(f'State {state}, indices ({ix}, {iy}): Symmetric (equal)')
        
        # fig, ax = plt.subplots()
        
        # ax.plot(X, Y, E[:, 0], label='Ground state')
        # ax.plot(X, Y, E[:, 1], label='Excited state')
        # print(E)
        return X, E, U#, output_messages
    
    # def electronic_overlap(self):

    #     U = self.u # adiabatic states
        
    #     A = np.einsum('aijm, cijn -> amcn', U.conj(), U) # Basis set cancellation,  the operation involves a sum over the i dimension
        
    #     # print(A)
    #     return A
    
    def plot_pes(self):
        import matplotlib.pyplot as plt
        import proplot as pplt
        from matplotlib.ticker import MaxNLocator, NullLocator
        from mpl_toolkits.mplot3d import Axes3D
        
        fontsize = 40
        line_width = 4  
        tick_width = 2  
        border_width = 3

        fig = plt.figure(figsize=(8, 10))
        # ax = fig.add_subplot(111, projection='3d')
        for n in range (8):
            plt.plot(X, E[:, n], label=f'E{n}', linewidth=line_width)
            
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        ax = plt.gca()
        ax.xaxis.set_tick_params(width=tick_width)  
        ax.yaxis.set_tick_params(width=tick_width)
        # ax.set_xlim(-7, 7)
        # ax.set_xlim(-4.5/au2angstrom, 4.5/au2angstrom)
        ax.set_ylim(-22, -12)

        for spine in ax.spines.values():
            spine.set_linewidth(border_width)

        plt.xlabel('X Axis', fontsize=40, labelpad=25)
        plt.ylabel('Energy', fontsize=40, labelpad=25)
        plt.title('Potential Energy Surfaces', fontsize=40, pad=15) 

        plt.legend(fontsize=40, loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3)

        plt.tight_layout()
        plt.show()


class RHF_SOC(ShinMetiu1d):
    """
    3D spinful electron in a linear chain of three protons 
    """
    

    def create_grid(self, domain, level):
        """
        create real-space grids (DVR basis sets)

        Parameters
        ----------
        domain : TYPE
            DESCRIPTION.
        level : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if isinstance(level, (int, float)):
            level = [level, ] * 3
            
        x = discretize(*domain[0], level[0], endpoints=False)
        y = discretize(*domain[1], level[1], endpoints=False)
        z = discretize(*domain[2], level[2], endpoints=False)

        
        self.x = x 
        self.y = y
        self.z = z 
        
        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)
        
        # self.lx = domain[0][1]-domain[0][0]
        # self.ly = domain[0][1]-domain[0][0]
        # self.dx = self.lx / (self.nx - 1)
        # self.dy = self.ly / (self.ny - 1)
        self.domain = domain 
        self.level = level
        
    def run(self):
        """
        build electronic Hamiltonian matrix H(r; R) in spin-orbitals and diagonalize
        
        Parameters
        ----------
        R : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        w : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.

        """

        x, y, z = self.x, self.y, self.z
        nx, ny, nz = self.nx, self.ny, self.nz
        
        domain = self.domain
        
        # KEO
        if self.dvr_type == 'sine':
            
            dvr_x = SineDVR(domain[0][0], domain[0][1], nx, mass=self.mass)
            tx = dvr_x.t()
            idx = dvr_x.idm 
            
            dvr_y = SineDVR(domain[1][0], domain[1][1], ny, mass=self.mass)
            ty = dvr_y.t()
            idy = dvr_y.idm 
            
            dvr_z = SineDVR(domain[2][0], domain[2][1], nz, mass=self.mass)
            tz = dvr_z.t()
            idz = dvr_z.idm 
        
        else:
            raise NotImplementedError('DVR type {} has not been \
                                      implemented.'.format(self.dvr_type))

        
        T = kron(tx, kron(idy, idz)) + kron(idx, kron(ty, idz)) + \
            kron(idx, kron(idy, tz))

        
        # PEO
        v = np.zeros((nx, ny, nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r = np.array([x[i], y[j], z[k]])
                    v[i, j, k] = self.potential_energy(r)
        
        V = np.diag(v.ravel())
        

        nao = nx * ny * nz
        nso = nao * 2 # number of spin-orbitals
        
        hcore = T + V # KEO + Coulomb potential
        
        # 1e SOC
        s0, s1, s2, s3 = pauli() 
        
        hso1 = alpha**2/2 * py
        
        # H = np.block([ 
        #       [hcore, hso], 
        #       [hso.conj(), hcore]])

        H = kron(hcore, s0)        
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            raise ValueError("H matrix contains NaNs or infs.")
        
        if self.method == 'exact':
            w, u = eigh(H)
        
        elif self.method == 'davidson':
            w, u = davidson(H, neigen=self.nstates)
            
        elif self.method == 'scipy':
            w, u = scipy.sparse.linalg.eigsh(csr_matrix(H), k=self.nstates, which='SA', v0=self.v0)
            
            self.v0 = u[:,0] # store the eigenvectors for next calculation

        else:
            raise ValueError("Invalid method specified")
    
        
        return w, u
    
class AtomicChain(ShinMetiu1d):
    """
    1D chain of atoms
    """
    def __init__(self, R, nstates=None, charge=0, dvr_type='sine', spin=0, Z=1, diag_method = 'scipy'):

        self.geometry = self.atom_coords = self.R = R
        self.nuc_charge = Z
        self.natom = len(R)
        
        # self.charge = self.nuc_charge * self.natom - nelec
        self.charge = charge 
        self.nelec =  self.nuc_charge * self.natom - self.charge   
        
        super().__init__(method = diag_method, nstates=nstates, nelec=self.nelec, \
                       dvr_type=dvr_type, spin=spin)
        
    
    def v_en(self, r):
        """
        Electron-nucleus interaction potential.
        """
        
        v = 0 
        for a in range(self.natom):
            d = np.linalg.norm(r - self.R[a])
            v += -soft_coulomb(d, self.Rf)
        return v  * self.nuc_charge
    
    def get_hcore(self):
        """
        single point calculations

        Parameters
        ----------
        R : float
            proton position.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        w : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.

        """
        
        # H(r; R)
        x = self.x
        nx = self.nx

        
        # T 
        # origin method of calculate kinetic term
        # tx = kinetic(x, dvr=self.dvr_type)
        # idx = np.eye(nx)  
        
        # ty = kinetic(y, dvr=self.dvr_type)
        # idy = np.eye(ny)
        
        # T = kron(tx, idy) + kron(idx, ty)
        
        # # # new method of calculate kinetic term
        dvr_x = SineDVR(*self.domain, nx)
        
        # tx = kinetic(self.x, dvr=self.dvr_type)
        T = dvr_x.t()
    
        self.T = T
        
        
        # V_en
        # Ra = self.left
        # Rb = self.right 
        v = np.zeros((nx))
        for i in range(nx):   
            r1 = np.array(x[i])
            # Potential from all ions
            v[i] = self.v_en(r1)
        
        V = np.diag(v)
        
        # v_sym = self.enforce_spin_symmetry(v)
        # # print(v_sym.shape)
        # V = np.diag(v_sym.ravel())
        
        H = T + V 
        # H = self.imaginary_time_propagation(H)
        
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            raise ValueError("H matrix contains NaNs or infs.")
        
        return H
    
    def energy_nuc(self):   
        # Ra = self.left
        # Rb = self.right 
        v = 0        
        for a in range(self.natom):
            for b in range(a):
                v += self.V_nn(self.R[a], self.R[b])
        return v 

def plot_mo(mo):



    fig, ax = plt.subplots()
    for j in range(4):
        ax.plot(mol.x, mo[:, j], label= str(j))
    ax.legend(frameon=False, title='MO')
    fig.savefig('MO.pdf')
        
    
def eri_svd(mf):
    # ax.imshow(mf.eri)
    u, a, vh = np.linalg.svd(mf.eri)
    # ax.plot(a)  
    return u, a, vh

if __name__=='__main__':
    
    # import proplot as plt 
    import matplotlib.pyplot as plt
    from pyqed.qchem.dvr.rhf import RHF1D
    from pyqed.qchem.casci import CASCI

    # r = np.linspace(0, 1)
    # # v = [soft_coulomb(_r, 1) for _r in r]
    # v = soft_coulomb(r)

    # fig, ax = plt.subplots()
    # ax.plot(r, v)
    
    L = 10/au2angstrom
        # print(self.L)
        # self.mass = mass  # nuclear mass
    # z = np.array([-L/2, -L/4, L/4, L/2])
    z0 = np.linspace(-1, 1, 4) * L/2
    print(z0)
    
    print('distance = ', (z0[1] - z0[0])*au2angstrom)
    

    mol = AtomicChain(z0, charge=0)
    print('number of electrons = ', mol.nelec)
    ###############################################################################  
    # mol = ShinMetiu1d(nstates=3, nelec=2)
    # # mol.spin = 0
    mol.create_grid([-15/au2angstrom, 15/au2angstrom], level=4)
    
    # # exact 
    # R = 0.
    # w, u = mol.single_point(R)
    # print(w)
    
    # fig, ax = plt.subplots()
    # ax.imshow(u[:, 1].reshape(mol.nx, mol.nx), origin='lower')
    
    # HF 
    mf = RHF1D(mol)
    mf.run()

    # E = mf.e_tot
    
    cas = CASCI(mf, ncas=6, nelecas=4)
    w, X = cas.run(1)
    # e_cas = w
    print("{:.15f}".format(w[0]))
    
 
    ### scan PEC
    
    # ds = np.linspace(-3, 2, 10)
    # E = np.zeros(len(ds))
    
    # nstates = 3
    # e_cas = np.zeros((len(ds), nstates))

    # for i in range(len(ds)):
        
    #     d = ds[i]
    #     z = z0 + np.array([0, d, -d, 0])
    
    #     mol = AtomicChain(z, charge=0)
    #     print('number of electrons = ', mol.nelec)
    #     ###############################################################################  
    #     # mol = ShinMetiu1d(nstates=3, nelec=2)
    #     # # mol.spin = 0
    #     mol.create_grid([-15/au2angstrom, 15/au2angstrom], level=7)
        
    #     # # exact 
    #     # R = 0.
    #     # w, u = mol.single_point(R)
    #     # print(w)
        
    #     # fig, ax = plt.subplots()
    #     # ax.imshow(u[:, 1].reshape(mol.nx, mol.nx), origin='lower')
        
    #     # HF 
    #     mf = RHF1D(mol)    
    #     mf.run()
    
    #     E[i] = mf.e_tot
        
    #     cas = CASCI(mf, ncas=6)
    #     w, X = cas.run(3)
    #     e_cas[i, :] = w
    #     # mo = mf.mo_coeff

    # np.savez('e_cas_mode2_l7', ds, E, e_cas)



    # ax.set_ylim(-3,0)
    

    
    ## CASCI
    

    
    
    # mol.create_grid(5, [[-15/au2angstrom, 0], [0, 15/au2angstrom]])
    # X, E, U = mol.pes(domain=[-4.5/au2angstrom, 4.5/au2angstrom], level=1) #level=7
    # E = au2ev*E
    # print(E)
    # print(U.shape)
    # np.save('E_2e1d_e5n7_rc1.5_re2.5.npy', E)
    # np.save('U_2e1d_e5n7_rc1.5_re2.5.npy', U)
    # mol.plot_pes()
    
    # output_text = "\n".join(output_messages)
    # file_path = '/home/xiaozhu/Pyscf/240116_ShinMetiu_2e/output_messages_antiparallel1.txt'
    # with open(file_path, 'w') as file:
    #     file.write(output_text)
    # print("Output messages saved to:", file_path)
    # A = mol.electronic_overlap()
    # print(A.shape)
    # np.save('E_1d_nuclei10_grid10.npy', E)
    
    # E = np.nan_to_num(E, nan=0.0, posinf=None, neginf=None)
    ###############################################################################
    # PES plotting using matplotlib
    # import matplotlib.pyplot as plt
    # import proplot as pplt
    # from matplotlib.ticker import MaxNLocator, NullLocator
    # from mpl_toolkits.mplot3d import Axes3D
    
    # X_grid = np.meshgrid(X)
    
    # fontsize = 40
    # line_width = 4  
    # tick_width = 2  
    # border_width = 3
    
    # fig = plt.figure(figsize=(8, 10))
    

    # ax = fig.add_subplot(111, projection='3d')
    # for n in range (8):
    #     plt.plot(X, E[:, n], label=f'E{n}', linewidth=line_width)
    
    # if spin_state =='antiparallel':
    #     plt.plot(X, E[:, 0], label='E0', linewidth=line_width)
    #     plt.plot(X, E[:, 2], label='E1', linewidth=line_width)
    #     plt.plot(X, E[:, 4], label='E2', linewidth=line_width)
    #     plt.plot(X, E[:, 6], label='E3', linewidth=line_width)
    # elif spin_state =='parallel':
    #     plt.plot(X, E[:, 1], label='E0', linewidth=line_width)
    #     plt.plot(X, E[:, 3], label='E1', linewidth=line_width)
    #     plt.plot(X, E[:, 5], label='E2', linewidth=line_width)
    #     plt.plot(X, E[:, 7], label='E3', linewidth=line_width)
    
    # for i in range(2):
    # for i in range(E.shape[1]):
    #     plt.plot(X, E[:, i], label=f'State {i+1}', linewidth=line_width)
    
    
    
    
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    
    # ax = plt.gca()
    # ax.xaxis.set_tick_params(width=tick_width)  
    # ax.yaxis.set_tick_params(width=tick_width)
    # # ax.set_xlim(-7, 7)
    # # ax.set_xlim(-4.5/au2angstrom, 4.5/au2angstrom)
    # ax.set_ylim(-22, -12)
    
    # for spine in ax.spines.values():
    #     spine.set_linewidth(border_width)
    
    # plt.xlabel('X Axis', fontsize=40, labelpad=25)
    # plt.ylabel('Energy', fontsize=40, labelpad=25)
    # plt.title('Potential Energy Surfaces', fontsize=40, pad=15) 
    
    # # plt.legend(fontsize=40, loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3)
    
    # plt.tight_layout()
    # plt.show()

 # def enforce_spin_symmetry(self, psi):
 
 #     if self.spin_state == 'antiparallel':
         
 #         psi_sym = psi + psi.T
 #     else:
         
 #         psi_sym = psi - psi.T
 #     return psi_sym / np.linalg.norm(psi_sym)
 
 # def enforce_spin_symmetry(self, psi):
 #     """
 #     Enforce the spin symmetry on the spatial wave function.
 #     """
 #     # Assume psi is a square matrix (NxN)
 #     for i in range(psi.shape[0]):
 #         for j in range(i + 1, psi.shape[1]):
 #             if self.spin_state == 'parallel':
 #                 # For parallel spins, enforce antisymmetry
 #                 psi[j, i] = -psi[i, j]
 #             elif self.spin_state == 'antiparallel':
 #                 # For antiparallel spins, enforce symmetry
 #                 psi[j, i] = psi[i, j]
 #                 # antisym_val = 0.5 * (psi[i, j] - psi[j, i])
 #                 # psi[i, j] = antisym_val
 #                 # psi[j, i] = antisym_val
 #     for k in range(psi.shape[0]):
 #         if self.spin_state == 'parallel':
 #             psi[k, k] = 0
         
 #         elif self.spin_state == 'antiparallel':
 #             psi[k, k] = psi[k, k]
                 
 #     return psi

  # def enforce_spin_symmetry(self, psi):
  #     """
  #     Enforce the spin symmetry on the spatial wave function.
  #     """
  #     # Assume psi is a square matrix (NxN)
  #     for i in range(psi.shape[0]):
  #         for j in range(i + 1, psi.shape[1]):
  #             if self.spin_state == 'parallel':
  #                 # For parallel spins, enforce antisymmetry
  #                 psi[j, i] = -psi[i, j]
  #             elif self.spin_state == 'antiparallel':
  #                 # For antiparallel spins, enforce symmetry
  #                 psi[j, i] = psi[i, j]
  #                 # antisym_val = 0.5 * (psi[i, j] - psi[j, i])
  #                 # psi[i, j] = antisym_val
  #                 # psi[j, i] = antisym_val
  #     for k in range(psi.shape[0]):
  #         if self.spin_state == 'parallel':
  #             psi[k, k] = 0
          
  #         elif self.spin_state == 'antiparallel':
  #             psi[k, k] = psi[k, k]
                  
  #     return psi
  
  # def imaginary_time_propagation(self, dt, max_steps=1000, convergence_threshold=1e-6):
  #    """
  #    Perform imaginary time propagation to find the ground state or excited state wave function.
  #    """
  #    # Initialize the wave function as a random guess
  #    psi = np.random.rand(self.nx, self.ny)
  #    psi /= np.linalg.norm(psi)
     
  #    # Initialize the old wave function for convergence check
  #    psi_old = np.zeros_like(psi)
 
  #    # Precompute the kinetic energy operator
  #    tx = kinetic(self.x, dvr=self.dvr_type)
  #    ty = kinetic(self.y, dvr=self.dvr_type)
  #    T = np.kron(tx, np.eye(self.ny)) + np.kron(np.eye(self.nx), ty)
 
  #    # Start the imaginary time evolution loop
  #    for step in range(max_steps):
  #        # Enforce the correct symmetry for the spin state
  #        psi_symmetric = self.enforce_spin_symmetry(psi)
         
  #        # Compute the potential energy term on the grid for the current R
  #        V = self.potential_energy(self.x[:, None], self.y[None, :], R)
         
  #        # Apply the Hamiltonian to the wave function
  #        H_psi = T @ psi_symmetric.ravel() + V.ravel() * psi_symmetric.ravel()
  #        H_psi = H_psi.reshape(self.nx, self.ny)
         
  #        # Apply the imaginary time evolution operator
  #        psi = np.exp(-H_psi * dt) * psi_symmetric
         
  #        # Renormalize the wave function
  #        psi /= np.linalg.norm(psi)
         
  #        # Check for convergence
  #        if step > 0 and np.linalg.norm(psi - psi_old) < convergence_threshold:
  #            print(f"Converged after {step} iterations.")
  #            break

  #        psi_old = psi.copy()
     
  #    return psi

 # def imaginary_time_propagation(self, dt, max_steps=1000, convergence_threshold=1e-6):
 #     """
 #     Perform imaginary time propagation to find the ground state or excited state wave function.
 #     """
 #     # Initialize the wave function as a random guess
 #     psi = np.random.rand(self.nx, self.ny)
 #     psi /= np.linalg.norm(psi)
     
 #     # Initialize the old wave function for convergence check
 #     psi_old = np.zeros_like(psi)
 
 #     # Start the imaginary time evolution loop
 #     for step in range(max_steps):
 #         # Enforce the correct symmetry for the spin state
 #         psi_symmetric = self.enforce_spin_symmetry(psi)
         
 #         # Apply the Hamiltonian to the wave function
 #         # For this example, we're only applying the potential energy operator
 #         # You would need to include the kinetic energy operator as well
 #         psi = np.exp(-self.potential_energy(self.x[:, None], self.y[None, :], 0) * dt) * psi_symmetric
         
 #         # Renormalize the wave function
 #         psi /= np.linalg.norm(psi)
         
 #         # Check for convergence (this is a simple version, you might need a more sophisticated check)
 #         if step > 0 and np.linalg.norm(psi - psi_old) < convergence_threshold:
 #             print(f"Converged after {step} iterations.")
 #             break

 #         psi_old = psi.copy()
     
 #     return psi
