#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:57:08 2024

ab initio LDR

@author: Yujuan, Bing
"""

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import LDR

import scipy.sparse as sp
from opt_einsum import contract
from tqdm import tqdm
import string
from scipy.linalg import inv
from scipy.sparse import eye
import logging
from pyscf import gto, scf, ci
from tqdm import tqdm
import pickle
from functools import reduce
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import inv


def neighbor(n, d, dims):
    """
    find the nearest-neighbor of the n-th element in d-direction

    Parameters
    ----------
    n : TYPE
        flattened index.
    d : TYPE
        DESCRIPTION.
    dims : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    idx = np.unravel_index(n, dims)
    # ndim = len(dims)

    e = np.zeros_like(dims)
    e[abs(d)] = np.copysign(1, d)

    nn = idx + e
    if 0 <= nn[d] < dims[d]:
        return np.ravel_multi_index(nn, dims)
    else:
        print("{} is boundary element without neighbor in {} direction".format(idx, d))


# def gen_enisum_string(D):
#     alphabet = list(string.ascii_lowercase)
#     if (D > 10):
#         raise ValueError('Dimension D = {} cannot be larger than 10.'.format(D))

#     ini = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y'
#     einsum_string = ini
#     for n in range(D):
#         einsum_string += ','
#         einsum_string += alphabet[n] + alphabet[n+D]

#     return (einsum_string + ' -> ' + ini)




# class LDRN:
#     """
#     many-dimensional many-state nonadiabatic conical intersection dynamics in
#     DVR + LDR + SPO


#     The required input to run is APES and electronic overlap matrix.


#     This is extremely expansive, the maximum dimension should be < 4.

#     """
#     def __init__(self, domains, levels, ndim=3, nstates=2, x0=None, mass=None, \
#                  dvr_type='sine', ref_geom=None):

#         assert(len(domains) == len(levels) == ndim)

#         self.domains = domains
#         self.levels = levels

#         self.L = [domain[1] - domain[0] for domain in domains]

#         # x = []
#         w = []
#         dvr = []
#         if dvr_type in ['sinc', 'sine']:
#             # uniform grid
#             for d in range(ndim):
#                 l = levels[d]
#                 # x.append(discretize(*domains[d], levels[d], endpoints=True))
#                 _w = [1/(2**l+1), ] * (2**l+1)
#                 w.append(_w)

#         elif dvr_type == 'gauss_hermite':

#             assert x0 is not None

#             for d in range(ndim):
#                 _dvr = HermiteDVR(x0[d], levels[d])
#                 # x.append(_dvr.x)
#                 w.append(_dvr.w)
#                 dvr.append(_dvr.copy())

#         else:
#             raise ValueError('DVR {} is not supported. Please use sinc.')


#         self.x = ref_geom
#         self.w = w # weights
#         self.dvr = dvr
#         self.dx = [interval(_x) for _x in ref_geom]
#         self.nx = [len(_x) for _x in ref_geom]

#         self.dvr_type = [dvr_type, ] * ndim

#         if mass is None:
#             mass = [1, ] * ndim
#         self.mass = mass

#         self.nstates = nstates
#         self.ndim = ndim

#         # all configurations in a vector
#         self.points = np.fliplr(cartesian_product(ref_geom))
#         self.ntot = len(self.points)

#         ###
#         self.H = None
#         self.K = self.T = None
#         # self._V = None

#         self._v = None
#         self.exp_K = None
#         self.exp_V = None
#         self.exp_T = None # KEO in LDR
#         self.wf_overlap = self.A = None
#         self.apes = None


#     @property
#     def v(self):
#         return self._v

#     @v.setter
#     def v(self, v):
#         assert(v.shape == (*self.nx, self.nstates, self.nstates))

#         if abs(np.min(v)) > 0.1:
#             raise ValueError('The PES minimum is not 0. Shift the PES.')

#         self._v = v


#     def buildK(self, dt):
#         """
#         For the kinetic energy operator with Jacobi coordinates

#             K = \frac{p_r^2}{2\mu} + \frac{1}{I(r)} p_\theta^2

#         Since the two KEOs for each dof do not commute, it has to be factorized as

#         e^{-i K \delta t} = e{-i K_1 \delta t} e^{- i K_2 \delta t}

#         where $p_\theta = -i \pa_\theta$ is the momentum operator.


#         Parameters
#         ----------
#         dt : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         TYPE
#             DESCRIPTION.

#         """

#         self.exp_K = []
#         self.K = []

#         for d in range(self.ndim):

#             dvr = SineDVR(*self.domains[d], self.nx[d], mass=self.mass[d])

#             Tx = dvr.t() # np.array(expKx).shape=(nx, ny)
#             expKx = dvr.expT(dt) # np.array(Tx).shape=(nx, ny)

#             self.exp_K.append(expKx.copy())
#             self.K.append(Tx.copy())

#         print('self.exp_K.shape=', np.array(self.exp_K).shape, 'self.K.shape=', np.array(self.K).shape)
#         return self.exp_K, self.K



#     def buildV(self, dt):
#         """
#         Setup the propagators appearing in the split-operator method.



#         Parameters
#         ----------
#         dt : TYPE
#             DESCRIPTION.

#         intertia: func
#             moment of inertia, only used for Jocabi coordinates.

#         Returns
#         -------
#         None.

#         """

#         dt2 = 0.5 * dt
#         self.exp_V = np.exp(-1j * dt * self.apes)

#         self.exp_V_half = np.exp(-1j * dt2 * self.apes)

#         return


#     def gen_enisum_string(self, D):
#         """
#         Generate einsum string for computing the short-time propagator

#         ij...a, ij...a, kl...b, lk...b -> ija, klb

#         Parameters
#         ----------
#         D : TYPE
#             DESCRIPTION.

#         Raises
#         ------
#         ValueError
#             DESCRIPTION.

#         Returns
#         -------
#         TYPE
#             DESCRIPTION.

#         """
#         alphabet = list(string.ascii_lowercase)
#         if (D > 10):
#             raise ValueError('Dimension D = {} cannot be larger than 10.'.format(D))

#         s1 = "".join(alphabet[:D]) + 'x'
#         # s2 = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y'
#         s3 = "".join(alphabet[D:2*D])+'y'
#         s2 = s1 + s3

#         einsum_string = s1 + ',' + s2 + ',' + s3 + '->' + s2

#         return einsum_string


#     def short_time_propagator(self, dt):

#         if self.apes is None:
#             print('building the adibatic potential energy surfaces ...')
#             self.build_apes()

#         self.buildV(dt)

#         print('building the kinetic energy propagator')
#         self.buildK(dt)


#         if self.A is None:
#             logging.info('building the electronic overlap matrix')
#             self.build_ovlp()


#         einsum_string = gen_enisum_string(self.ndim)
#         exp_T = contract(einsum_string, self.A, *self.exp_K)


#         einsum_string = self.gen_enisum_string(self.ndim)
#         U = contract(einsum_string, self.exp_V_half, exp_T, self.exp_V_half)
#         return U


#     def buildH(self, dt):

#         print('building the potential energy propagator')
#         self.buildV(dt)

#         print('building the kinetic energy propagator')
#         self.buildK(dt)

#         if self.A is None:
#             logging.info('building the electronic overlap matrix')
#             self.build_ovlp()

#         size = np.prod(self.nx) * self.nstates # size = nx * ny

#         einsum_string = gen_enisum_string(self.ndim)
#         H = np.diag(self.apes.flatten()) + \
#             contract(einsum_string, self.A, *self.K).reshape(size, size) #einsum_string = abxcdy,ac,bd -> abxcdy, self.A.shape=(63, 63, 1, 63, 63, 1), self.K.shape= (2, 63, 63)

#         self.H = H

#         return H


#     def Hpsi(self, psi, A, K, Va):

#         Kx = K[0]
#         Ky = K[1]

#         Kx_Ky = sp.kron(Kx, np.eye(self.nx[1])) + sp.kron(np.eye(self.nx[0]), Ky)
#         Kx_Ky = Kx_Ky.toarray()
#         K_total = Kx_Ky.reshape(self.nx[0], self.nx[1],self.nx[0], self.nx[1])

#         K_total = contract('abxcdy, abcd -> abxcdy', A, K_total)
#         Kpsi = contract('abxcdy, cdy -> abx', K_total, psi)

#         Vpsi = Va*psi # Va is adiabatic potential energy matrix
#         hpsi = Kpsi + Vpsi

#         return -1j * hpsi



#     def run(self, psi0, dt, nt, nout=1, t0=0, method='spo'):

#         assert(psi0.shape == (*self.nx, self.nstates))

#         r = ResultLDR(dx=self.dx, x=self.x, dt=dt, psi0=psi0, Nt=nt, t0=t0, nout=nout)
#         r.psilist = [psi0]

#         if method == 'spo':
#             # Split-operator method for linear coordinates

#             if self.H is None:
#                 self.buildH(dt)

#             einsum_string = gen_enisum_string(self.ndim)
#             exp_T = contract(einsum_string, self.A, *self.exp_K) #einsum_string = abxcdy,ac,bd -> abxcdy, #sself.exp_K.shape=(2, 63, 63)

#             alphabet = list(string.ascii_lowercase)
#             D = self.ndim
#             _string = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y, ' + \
#                 "".join(alphabet[D:2*D])+'y -> ' + "".join(alphabet[:D]) + 'x'

#             psi = psi0.copy()
#             psi = self.exp_V_half * psi
#             for k in tqdm(range(nt//nout)):
#                 for kk in range(nout):

#                     psi = contract(_string, exp_T, psi)
#                     psi = self.exp_V * psi

#                 r.psilist.append(psi.copy())
#             psi = self.exp_V_half * psi


#         elif method == 'rk4':

#             print('building the kinetic energy propagator')
#             exp_K, K = self.buildK(dt)

#             t = t0
#             psi = psi0.copy()

#             for k in range(nt//nout):
#                 for l in range(nout):

#                     t += dt
#                     psi = rk4(psi, self.Hpsi, dt, self.A, K, self.apes)

#                 r.psilist.append(psi.copy())

#         return r


#     def initialize_wavepacket(self, ref_geom, idx0, idx1, a, A, target_state=1):
#         """
#         Initialize a Gaussian wavepacket psi0 centered at position x0.

#         Parameters:
#         - ref_geom: list of two 1D arrays [x_grid, y_grid]
#         - nx, ny: grid sizes in x and y directions
#         - nstates: number of electronic states
#         - idx0, idx1: index of initial position (e.g. [0.0, 1.71875])
#         - a: 2x2 matrix (numpy array), Gaussian width matrix
#         - target_state: the index of the adiabatic state (default is 1)
#         - A: global electronic overlap matrix

#         Returns:
#         - psi0: complex array of shape (nx, ny, nstates)
#         """
#         ndim = self.ndim
#         nx, ny = len(ref_geom[0]), len(ref_geom[1])
#         psi0 = np.zeros((nx, ny, self.nstates), dtype=complex)
#         for i in range(nx):
#             for j in range(ny):
#                 x = np.array([ref_geom[0][i], ref_geom[1][j]])
#                 a = a * np.eye(ndim)
#                 psi0[i, j, target_state] = gwp(x, a=a, x0=[ref_geom[0][idx0], ref_geom[1][idx1]], ndim=2)
#         # project it into the computational space using the overlap matrix
#         psi0 = np.einsum('ijk,ij->ijk', A[:, :, :, idx0, idx1, target_state], psi0[:,:,target_state])
#         return psi0



class LPA:
    def __init__(self, ci_list, dims, nstates):
        # with open(pes_data_file, 'rb') as f:
        #     self.pes_data = pickle.load(f)

        # self.nx = int(np.sqrt(len(self.pes_data)))
        # self.ny = self.nx

        self.dims = dims
        self.ndim = len(dims)

        self.npts = len(ci_list)

        self.nstates = nstates
        # self.n_total = self.nx * self.ny * self.nstates

        self.link = None

    def rebuild_molecule(self, mol_basis, mol_atom):
        mol = gto.Mole()
        mol.atom = mol_atom
        mol.basis = mol_basis
        mol.spin = 0
        mol.charge = 1
        mol.unit = 'bohr'
        mol.build()
        return mol

    # def overlap(self, data1, data2, state1, state2):

    #     mol1 = self.rebuild_molecule(data1['mol_basis'], data1['mol_atom'])
    #     mol2 = self.rebuild_molecule(data2['mol_basis'], data2['mol_atom'])
    #     mo_coeff1 = data1['mo_coeff']
    #     mo_coeff2 = data2['mo_coeff']

    #     s12 = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
    #     s12 = reduce(np.dot, (mo_coeff1.T, s12, mo_coeff2))
    #     nmo = mo_coeff2.shape[1]
    #     nocc = mol2.nelectron // 2

    #     overlap = ci.cisd.overlap(data1['ci_vector'][state1], data2['ci_vector'][state2], nmo, nocc, s12)
    #     return overlap

    def link(self):

        L = np.zeros((self.npts, self.nstates, self.nstates))

        links = []

        for d in range(self.ndim):

            # for i in range(self.nx):
            #     for j in range(self.ny):
            for n in range(self.npts):

                nn = neighbor(n, d, self.dims)
                            # # Self overlap
                            # self.link[i, j, state1, i, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + j], state1, state2)

                            # if 0 <= i - 1 < self.nx:
                            #     self.link[i - 1, j, state1, i, j, state2] = self.compute_overlap(self.pes_data[(i - 1) * self.ny + j], self.pes_data[i * self.ny + j], state1, state2)
                            #     self.link[i, j, state1, i - 1, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[(i - 1) * self.ny + j], state1, state2)

                            # if 0 <= j - 1 < self.ny:
                            #     self.link[i, j - 1, state1, i, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + (j - 1)], self.pes_data[i * self.ny + j], state1, state2)
                            #     self.link[i, j, state1, i, j - 1, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + (j - 1)], state1, state2)


                        # Self overlap
                        # L[i, j, state1, i, j, state2] = self.overlap(self.pes_data[i * self.ny + j], self.pes_data[i * self.ny + j], state1, state2)

                        # Neighbors in the diagonal directions

                        # if 0 <= i + delta < self.nx:
                if nn is not None:
                    L[n] = self.overlap_matrix(self.ci_list[n], self.ci_list[nn])
                             # L[i, j, state1, i + delta, j, state2] = self.compute_overlap(self.pes_data[i * self.ny + j], self.pes_data[(i + delta) * self.ny + j], state1, state2)

                links.append(L)

        return links

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.A, f)

    def _linked_product_approximation_2D(self, Lx, Ly):
        """原始方法1：linked_product_approximation_2D"""
        A1d = A2d = csr_matrix(np.zeros(Lx.shape))
        I = eye(Lx.shape[0], format="csr")

        A1d = Lx @ (I - Lx**(self.nx - 1)) @ inv(I - Lx)
        A1d = A1d + A1d.conj().T

        for j in range(1, self.ny):
            A2d += A1d @ Ly**j + Ly**j
        A2d = A2d + A2d.conj().T

        Atot = A1d + A2d
        print("Atot.shape:", Atot.shape)
        return Atot

    def _extract_neighbors(self, A):
        """提取最近邻与对角元信息，并构造成稀疏矩阵"""

        nearest_neighbor_x = np.zeros_like(A)
        nearest_neighbor_y = np.zeros_like(A)
        diagonal = np.zeros_like(A)

        for i in range(self.nx):
            for j in range(self.ny):
                diagonal[i, j, :, i, j, :] = A[i, j, :, i, j, :]
                if i + 1 < self.nx:
                    nearest_neighbor_x[i, j, :, i + 1, j, :] = A[i, j, :, i + 1, j, :]
                if j + 1 < self.ny:
                    nearest_neighbor_y[i, j, :, i, j + 1, :] = A[i, j, :, i, j + 1, :]

        self.Lx = csr_matrix(nearest_neighbor_x.reshape(self.n_total, self.n_total))
        self.Ly = csr_matrix(nearest_neighbor_y.reshape(self.n_total, self.n_total))
        self.A_diagonal = csr_matrix(diagonal.reshape(self.n_total, self.n_total))

    def _reshape_and_save(self, A_appro):
        A_appro = A_appro + self.A_diagonal
        A_appro = A_appro.toarray()
        A_appro = A_appro.reshape(self.nx, self.ny, self.nstates, self.nx, self.ny, self.nstates)
        np.save(self.save_file, A_appro)
        return A_appro

    def approximate_overlap(self):
        print('Step 1: Extracting neighbors and diagonals...')
        self._extract_neighbors(self.link)

        print(f'Step 2: Running linked product approximation ...')
        A_linked = self._linked_product_approximation_2D(self.Lx, self.Ly)

        print('Step 3: Reshaping and saving to file...')
        A = self._reshape_and_save(A_linked)

        return A


def initialize_wavepacket(ref_geom, idx0, idx1, a, A, target_state=1):
    """
    Initialize a Gaussian wavepacket psi0 centered at position x0.

    Parameters:
    - ref_geom: list of two 1D arrays [x_grid, y_grid]
    - nx, ny: grid sizes in x and y directions
    - nstates: number of electronic states
    - idx0, idx1: index of initial position (e.g. [0.0, 1.71875])
    - a: 2x2 matrix (numpy array), Gaussian width matrix
    - target_state: the index of the adiabatic state (default is 1)
    - A: global electronic overlap matrix

    Returns:
    - psi0: complex array of shape (nx, ny, nstates)
    """
    ndim = self.ndim
    nx, ny = len(ref_geom[0]), len(ref_geom[1])
    psi0 = np.zeros((nx, ny, self.nstates), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            x = np.array([ref_geom[0][i], ref_geom[1][j]])
            a = a * np.eye(ndim)
            psi0[i, j, target_state] = gwp(x, a=a, x0=[ref_geom[0][idx0], ref_geom[1][idx1]], ndim=2)
    # project it into the computational space using the overlap matrix
    psi0 = np.einsum('ijk,ij->ijk', A[:, :, :, idx0, idx1, target_state], psi0[:,:,target_state])
    return psi0


class QChemDriver:
    def __init__(self, mol, spin=0, nstates=3, method='cisd'):

        self.nstates = nstates
        self.mol = mol
        # self.nx = len(reference_geometry[0])
        # self.ny = len(reference_geometry[1])
        # self.x = reference_geometry[0]
        # self.y = reference_geometry[1]

        self.ci = None

        return

    def run(self, R):

        self.mol.set_geom_(R)

        if self.method == 'cisd':
            return self.cisd()
        else:
            raise ValueError('Method {} is not supported. Try cisd.'.format(self.method))

    def cisd(self, R, *args):

        self.mol.set_geom_(R) # update geometry

        mf = scf.RHF(self.mol, *args)
        # mf.init_guess = 'atom'  # Use 'atom' initial guess method
        mf.kernel()

        myci = ci.CISD(mf)
        myci.nstates = self.nstates
        myci.run()

        self.ci = myci
        self.e_tot = myci.e_tot
        self.ci = myci.ci

        return self

        # return {
        #     'e_s0': myci.e_tot[0],
        #     'e_s1': myci.e_tot[1],
        #     'e_s2': myci.e_tot[2],
        #     'mo_coeff': mf.mo_coeff,
        #     'mo_energy': mf.mo_energy,
        #     'mol_basis': mf.mol.basis,
        #     'mol_atom': mf.mol.atom,
        #     'ci_vector': myci.ci
        # }


def global_overlap(ci_list, nstates, dtype=float):

    # mol1 = self.rebuild_molecule(data1['mol_basis'], data1['mol_atom'])
    # mol2 = self.rebuild_molecule(data2['mol_basis'], data2['mol_atom'])

    N = len(ci_list)

    A = np.zeros((N, N, nstates, nstates), dtype=dtype)

    for n in range(N):
        A[n,n] = eye(nstates)

    for n in range(N):

        cibra = ci_list[n]
        # mo_coeff1 = cibra.mo_coeff
        # mol1 = cibra.mol
        # ci1 = cibra.ci

        for m in range(n):
            ciket = ci_list[m]
            # mo_coeff2 = ciket.mo_coeff
            # mol2 = ciket.mol

            # s12 = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
            # s12 = reduce(np.dot, (mo_coeff1.T, s12, mo_coeff2))
            # nmo = mo_coeff2.shape[1]
            # nocc = mol2.nelectron // 2

    # nstates = self.nstates

    # for i in range(1, nstates):
    #     S[i-1, i-1] = ci.cisd.overlap(ci1.ci[i-1], ci2.ci[i-1], nmo, nocc)

    # print('<CISD-mol1|CISD-mol2> = ',

            A[n, m] = overlap_matrix(cibra, ciket)
            A[m, n] = A[n, m].conj().T

    return A



def overlap_matrix(cibra, ciket):

    # mol1 = self.rebuild_molecule(data1['mol_basis'], data1['mol_atom'])
    # mol2 = self.rebuild_molecule(data2['mol_basis'], data2['mol_atom'])

    # mo_coeff1 = data1['mo_coeff']

    mo_coeff1 = cibra.mo_coeff
    mo_coeff2 = ciket.mo_coeff

    # ci1 = self.ci
    mol1 = cibra.mol
    mol2 = ciket.mol
    # mo_coeff2 = data2['mo_coeff']


    s12 = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
    s12 = reduce(np.dot, (mo_coeff1.T, s12, mo_coeff2))
    nmo = mo_coeff2.shape[1]
    nocc = mol2.nelectron // 2

    S = np.zeros((nstates, nstates))

    for i in range(nstates):
        S[i, i] = ci.cisd.overlap(cibra.ci[i], ciket.ci[i], nmo, nocc, s12)

    # print('<CISD-mol1|CISD-mol2> = ',
    for i in range(nstates):
        for j in range(i):
            S[i, j] = ci.cisd.overlap(cibra.ci[i-1], ciket.ci[j-1], nmo, nocc, s12)
            S[j, i] = S[i, j]

    return S

if __name__=='__main__':
    from pyqed.phys import gwp2
    from pyqed.units import au2fs
    from pyqed.qchem import Molecule

    def coord_transform(x, t=None, x0=None):
        """
        transform from reactive coords to Cartesian coords for quantum chemistry

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        t : 2darray
            transformation matrix of shape (3Natom, d)

        Returns
        -------
        None.

        """
        l = 2
        return np.array([-l/2, 0, 0], [x[0], x[1], 0], [ l/2, 0, 0])

    ndim = 2 # Number of nuclear degrees of freedom
    nstates = 3 # Number of electronic states
    mass = [1836, 1836] # Nuclear mass


    domains = [[-5.5, 5.5], [0, 11]] # Nuclear coordinate domains
    levels = [5, 5] # Discretization levels

    grid = DVR(domains, [2**level - 1 for level in levels], mass=mass)
    ldr = LDR(grid, nstates)

    # x, y = ldr.x
    nx, ny = ldr.shape
    npts = ldr.ngrid

    # psi0 = initialize_wavepacket(ref_geom=ref_geom, idx0=32, idx1=10, a=30, A=overlap_matrix, target_state=1) # x[idx0], y[idx1]是波包初始位置的中心点


    # def transform2Cartesian_coord(domains, levels):
    #     reference_geometries = []
    #     for d in range(len(domains)):
    #         reference_geometries.append(discretize(*domains[d], levels[d], endpoints=True))
    #     return reference_geometries
    # ref_geom = transform2Cartesian_coord(domains, levels)




     ### quantum chemistry

    atom = """
            H -1.0, 0, 0;
            H 1., 0, 0;
            H 0,0,0
            """
    mol = Molecule(atom, charge=1, basis='631g')
    mol.build()

    driver = QChemDriver(mol, nstates=nstates)
    # apes = mol.scan_pes()



    # quantum chemistry

    v = np.zeros((npts, nstates))

    ci_list = []

    for i in range(npts):
            # q = [x[i], y[j]] # reactive coordinates

        R = coord_transform(ldr.points[i]) # transform to Cartesian coordinates

        myci = driver.cisd(R)

        v[i] = myci.e_tot

        ci_list.append(myci)  # Store all energy states for the configuration

    # with open('qchem_data_total.pkl', 'wb') as f:
    #     pickle.dump(pes_data, f)


    # build the global electronic overlap matrix

    lpa = LPA(ci_list, dims=ldr.shape, nstates=nstates)

    links = lpa.link()

    A = lpa.global_overlap()

  #psi0 = ldr.initialize_wavepacket(ref_geom=ref_geom, idx0=0, idx1=0, a=10, A=overlap_matrix, target_state=1)



    # load electronic structure data into the quantum dynamics solver

    ldr.energies = v.reshape((nx, ny, nstates))
    ldr.set_overlaps(overlaps=A)


    psi0 = np.zeros((nx, ny, nstates), dtype=complex)
    x, y = ldr.x 
    psi0[:,:,1] = gwp2(x, y, a=np.eye(2) * 30)
    
    # project it into the computational space using the overlap matrix
    psi0 = np.einsum('ijk,ij->ijk', A[:, :, :, 10, 10, 1], psi0[:,:,1])

    ldr.run(psi0, dt=0.2, nsteps=1000)

    # result.dump('result')
