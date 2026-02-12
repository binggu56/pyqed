#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:04:50 2024

@author: bingg
"""
import numpy as np
import scipy.constants as const
import scipy
from scipy.sparse import identity, kron, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import kron, norm, eigh

from jax.scipy.special import erf
# from scipy.special import erf
import jax.numpy as jnp
from jax.numpy.linalg import vector_norm as norm

# from scipy.constants import e, epsilon_0, hbar
import logging
import warnings

from pyqed import discretize, sort, dag, tensor, pauli
from pyqed.davidson import davidson
from pyqed.ldr.ldr import kinetic
from pyqed import au2ev, au2angstrom
from pyqed.dvr import SineDVR
# from pyqed import scf
from pyqed.qchem.hf.rhf import make_rdm1, energy_elec
# from pyqed.jordan_wigner import jordan_wigner_one_body, jordan_wigner_two_body

# from pyqed.qchem.gto.ci.fci import SpinOuterProduct, givenΛgetB

# from numba import vectorize, float64, jit
import sys
from opt_einsum import contract
from itertools import combinations

def soft_coulomb(r, R=10):
    if np.isclose(r, 0):
            # if r_R_distance == 0:
        return 2 / (R * np.sqrt(np.pi))
    else:
        return erf(r / R) / r

# def soft_coulomb(r, R=1):
#     if np.isclose(r, 0):
#             # if r_R_distance == 0:
#         return 2 / (R * np.sqrt(np.pi))
#     else:
#         return erf(r / R) / r

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
#         with_k : boolean
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

def get_veff(eri, dm):
    """
    compute Hartree and Fock potential in DVR

    Parameters
    ----------
    dm : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # hartree potential
    J = jnp.einsum('ij, jj -> i', eri, dm)
    J = np.diag(J)

    # exchange
    K = eri * dm

    vHF = J - 0.5 * K

    return vHF



class RHF1D:
    """
    restricited DVR-HF method in 1D for soft Columb potential
    """
    def __init__(self, mol, init_guess='hcore'): # nelec, spin):
        # self.spin = spin
        self.nelec = mol.nelec
        self.mol = mol

        self.T = None
        self.hcore = None
        self.fock = None

        self.mol = mol
        self.max_cycle = 100
        self.tol = 1e-6
        self.init_guess = init_guess

        ### dvr setup
        self.x = mol.x
        self.nx = self.nao = self.nmo = mol.nx
        
        self.domain = mol.domain
        self.dvr_type = mol.dvr_type

        self.mo_occ = None
        self.mo_coeff = None
        self.e_tot = None

        self.e_nuc = mol.nuclear_repulsion()

        self.e_kin = None
        self.e_ne = None
        self.e_j = None
        self.e_k = None
        self.hcore = None

        self.eri = None

        self.dvr = None

    # def create_grid(self, domain, level, endpoints=False):

    #     x = discretize(*domain, level, endpoints=endpoints)

    #     self.x = x
    #     self.nx = len(x)

    #     self.lx = domain[1]-domain[0]
    #     # self.dx = self.lx / (self.nx - 1)

    #     self.domain = domain

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
                v[i,j] = v[j,i] = soft_coulomb(d, self.mol.Re)

                # v.at[i,j].set(_v)
                # v.at[j,i].set(_v)

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

        return get_veff(self.eri, dm)


    def get_hcore(self):
        """
        build the DVR basis sets and 1e and 2e integrals

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


        nx = self.nx
        x = self.x
        # T
        # origin method of calculate kinetic term

        # if self.dvr_type == 'sine':
        #     dvr = SineDVR(*self.domain, nx)
        # else:
        #     raise ValueError('DVR {} is not supported yet, use sine.'.format(self.dvr_type))


        # x = dvr.x
        # self.x = x


        # tx = kinetic(self.x, dvr=self.dvr_type)
        # T = dvr.t()

        self.T = self.mol.T

        # V_en
        # Ra = self.left
        # Rb = self.right
        # v = np.zeros((nx))
        # for i in range(nx):
        #     r1 = np.array(x[i])
            # Potential from all ions

        v = self.mol.electron_nuclear_attraction()

        # print("rhf v", v)

        V = np.diag(v)

        # v_sym = self.enforce_spin_symmetry(v)
        # # print(v_sym.shape)
        # V = np.diag(v_sym.ravel())

        H = self.T + V
        # H = self.imaginary_time_propagation(H)

        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            raise ValueError("H matrix contains NaNs or infs.")

        self.hcore = H
        return H

    def energy_nuc(self):
        return self.mol.energy_nuc()

    def run(self, dm0=None):
        # scf cycle
        max_cycle = self.max_cycle
        tol = self.tol

        mol = self.mol

        # Hcore (kinetic + v_en)
        hcore = self.get_hcore()
        # self.hcore = hcore

        # occ number
        nocc = self.mol.nelectron // 2
        mo_occ = np.zeros(self.nx)
        mo_occ[:nocc] = 2

        self.mo_occ = np.stack([mo_occ, mo_occ])
        # print('mo_occ', self.mo_occ)

        eri = self.get_eri()

        if dm0 is not None:

            vhf = get_veff(eri, dm0)
            old_energy = energy_elec(dm0, hcore, vhf)

        else:

            if self.init_guess == 'hcore':

                mo_energy, mo_coeff = eigh(hcore)
                dm = make_rdm1(mo_coeff, mo_occ)


                vhf = get_veff(eri, dm)
                old_energy = energy_elec(dm, hcore, vhf)

        logging.info("\n {:4s} {:13s} de\n".format("iter", "total energy"))

        nuclear_energy = mol.nuclear_repulsion()

        # print('nuclear repulsion', nuclear_energy)

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

            # print("E_elec = ", electronic_energy)

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

        self.e_tot = total_energy
        print('HF energy = ', total_energy)

        return self

    def as_scanner(self):
        """
        return a function for computing the PES

        Returns
        -------
        None.

        """

        dm = self.make_rdm()



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

class RHF2D(RHF1D):
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



class RHF:
    """
    3D RHF/DVR calculation
    """

    def __init__(self, mol, init_guess='hcore', dvr_type = 'sine'):
        # self.spin = spin
        # self.nelec = nelec
        self.mol = mol

        self.natom = mol.natm

        self.T = None
        self.hcore = None
        self.fock = None


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

    def potential(self, r):
        """
        Coulomb potential at position r

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        v = 0
        for i in range(self.natom):
            RA = self.mol.atom_coord(i)
            v += soft_coulomb(r, RA)
        return v


    def qubitization(self):
        pass

    def create_grid(self, domain, level):
        """
        create 3D real-space grids (DVR basis sets)

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

    def get_hcore(self):

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


        T = kron(tx, kron(idy, idz)) + kron(idx, kron(ty, idz)) + kron(idx, kron(idy, tz))

        # Tx = kinetic(self.x, dvr=self.dvr_type)
        # Ty = kinetic(self.y, dvr=self.dvr_type)
        # Tz = kinetic(self.z, dvr=self.dvr_type)

        # PEO
        v = np.zeros((nx, ny, nz))
        f = np.zeros(nx, ny, nz, 3)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r = np.array([x[i], y[j], z[k]])
                    v[i, j, k] = self.potential_energy(r)
                    f[i, j, k] = force(r)

        V = np.diag(v.ravel())

        self.hcore = T + V # KEO + Coulomb potential
        return self.hcore


    def run(self, R):
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



        nao = nx * ny * nz
        nso = nao * 2 # number of spin-orbitals



        # SOC

        # spin operators
        s0, sx, sy, sz = 0.5 * pauli()

        # H = np.block([
        #       [hcore, hso],
        #       [hso.conj(), hcore]])

        H = kron(hcore, s0) + kron(fy @ pz - fz @ py, sx)

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




# def wavefunction_overlap():
#     # wavefunction overlap between two exact many-electron wavefunctions at
#     # different geometries
#     pass


class RCISD:
    def __init__(self, mf, nstates=1):
        self.nstates = nstates
        self.mf = mf
        self.nmo = mf.nmo
        self.nocc = mf.nocc


        self.H = None



    def buildH(self):
        '''
        Return diagonal of CISD hamiltonian in Slater determinant basis.

        Note that a constant has been substracted of all elements.
        The first element is the HF energy (minus the
        constant), the next elements are the diagonal elements with singly
        excited determinants (<D_i^a|H|D_i^a> within the constant), then
        doubly excited determinants (<D_ij^ab|H|D_ij^ab> within the
        constant).

        Args:
            myci : CISD (inheriting) object
            eris : ccsd._ChemistsERIs (inheriting) object (poss diff for df)
                Contains the various (pq|rs) integrals needed.

        Returns:
            numpy array (size: (1, 1 + #single excitations from HF det
                                   + #double excitations from HF det))
                Diagonal elements of hamiltonian matrix within a constant,

        '''

        # restricted HF
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc

        # number of Slater determinants
        nsd = 1 + nocc * nvir + nocc**2*nvir**2
        HCI = np.zeros((nsd, nsd))


        e_hf = self.mf.energy_elec()
        HCI[0, 0] = e_hf



    def run(self, ci0=None, max_cycle=50, tol=1e-8):
        pass

    def vec_to_amplitudes(self, civec, copy=True):
        nmo = self.nmo
        nocc = self.nocc

        nvir = nmo - nocc
        c0 = civec[0]
        cp = lambda x: (x.copy() if copy else x)
        c1 = cp(civec[1:nocc*nvir+1].reshape(nocc,nvir))
        c2 = cp(civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir))
        return c0, c1, c2

    def make_rdm1(self):
        pass

    def make_rdm2(self):
        pass






# class UCISD(RCISD):
#     def __init__(self, mf):
#         pass


# def discrete_cosine_transform_matrix(n):
#     """
#     transform the Cartesian coordinates to collective coordinates
#     (e.g. enter-of-mass and relative motion)

#     .. math::

#         y_k = 2 \sum_{n=0}^{N-1} x_n \cos( \frac{\pi k (2n + 1) }{2N} )


#     Parameters
#     ----------
#     n : TYPE
#         DESCRIPTION.
#     dtype : TYPE, optional
#         DESCRIPTION. The default is None.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     from scipy.fft import dct

#     return dct(np.eye(n), type=2, axis=0, norm='ortho')
#     # return dft(n, scale='sqrtn')




if __name__=='__main__':

    # import proplot as plt
    from pyqed.qchem.dvr import RHF1D
    # from pyqed.qchem.ci.casci import CASCI
    from pyqed.qchem.dvr.casci import CASCI

    from pyqed.models.ShinMetiu2e1d import ShinMetiu1d
    # import matplotlib.pyplot as plt
    # r = np.linspace(0, 1)
    # # v = [soft_coulomb(_r, 1) for _r in r]
    # v = soft_coulomb(r)

    # fig, ax = plt.subplots()
    # ax.plot(r, v)
    from pyqed.qchem.dvr.rhf import RHF1D
    from pyqed.models.ShinMetiu2e1d import AtomicChain
    from pyqed import au2angstrom, discrete_cosine_transform_matrix

    import numpy as np

    natom = 8
    l = 10/au2angstrom

    z0 = np.linspace(-1, 1, natom, endpoint=True) * l/2
    print(z0 * au2angstrom)

    print('interatomic distance = ', (z0[1] - z0[0])*au2angstrom)


    # print('number of electrons = ', mol.nelec)
    ###############################################################################

    elements = ['H'] * natom
    T = discrete_cosine_transform_matrix(natom - 2) # remove the boundary atoms

    print(T)


    N = 8
    ds = np.linspace(-2, 2, N)
    e_hf = np.zeros(N)
    nstates = 3
    e_casci = np.zeros((N, nstates))

    i = 3
    print(T[i])

    Rf = 2
    for n in range(N):

        q = np.zeros(natom-2) # collective coordinates
        q[i] = ds[n]


        z = z0.copy()
        z[1:natom-1] += T.T @ q

        print('geometry', z)

        mol = AtomicChain(z, Rf, charge=0)


        # HF
        mf = RHF1D(mol, dvr_type='sine')
        mf.domain = [-12/au2angstrom, 12/au2angstrom]
        mf.nx = 32

        mf.run()
        e_hf[n] = mf.e_tot

        # CASCI
        mc = CASCI(mf, ncas=mol.nelec//2+2)

        E, X = mc.run(nstates)
        e_casci[n] = E



    import ultraplot as plt
    fig, ax = plt.subplots()

    ax.plot(ds, e_hf, '--')
    for n in range(nstates):
        ax.plot(ds, e_casci[:,n], '-o')

    h1e = mf.hcore
    eri = mf.eri

    L = nsites = mf.nx

    ###############################################################################
    # mol = ShinMetiu1d(method='scipy', nstates=3, nelec=4)
    # mol.spin = 0
    # mol.create_grid([-15/au2angstrom, 15/au2angstrom], level=5)
    # mol.R = 0.

    # # exact
    # # w, u = mol.single_point(R)
    # # print(w)


    # # fig, ax = plt.subplots()
    # # ax.imshow(u[:, 1].reshape(mol.nx, mol.nx), origin='lower')

    # # HF
    # mf = RHF1D(mol)
    # mf.create_grid([-15/au2angstrom, 15/au2angstrom], level=5)
    # mf.run()

    # mo = mf.mo_coeff
    # print('orb energies', mf.mo_energy)
    # e_nuc = mol.energy_nuc(R=0)

    # # print(mol.e_nuc)


    # # fig, ax = plt.subplots()
    # # ax.imshow(mf.eri)

    # # fig, ax = plt.subplots()
    # # for j in range(4):
    # #     ax.plot(mol.x, mo[:, j])




    # # H = cas.qubitization()
    # # E, X = scipy.sparse.linalg.eigsh(H, k=nstates, which='SA')
    # # print(E + e_nuc)

    # # E, X = np.linalg.eigh(H.toarray())
    # # for i in range(len(E)):
    # #     print(E[i] + e_nuc)