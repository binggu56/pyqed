#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:53:56 2018

@author: binggu
"""

import numpy as np
import sys
from scipy import linalg
from scipy.special import jv
from pyqed.mol import Mol, dag
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# from tqdm import tqdm
import time
import os
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013, save_data_to_hdf5, load_data_from_hdf5
from numpy import exp, eye, zeros, arctan2
from scipy.linalg import eigh
import os
import h5py

class TightBinding(Mol):
    """
    Generic tight-binding model parameterized by orbital coordinates and exponential-decay hopping.

    Parameters
    ----------
    coords : list of list of float
        Coordinates of each orbital within the unit cell; inner-list length = spatial dimension.
    lambda_decay : float
        Characteristic decay length: hopping t_ij = exp(-|r_i - r_j| / lambda_decay).
    a : float or array_like
        Lattice constant(s). Scalar or array of length = dimension.
    nk : int or list of int, optional
        Number of k-points per dimension for Brillouin zone sampling.
    mu : float, optional
        On-site energy added to diagonal.
    """
    def __init__(self, coords, relative_Hopping = None, lambda_decay=1.0, lattice_constant=[1.0], nk=50, mu=0.0):
        # Orbital positions and dimensionality
        self.coords = np.array(coords, dtype=float)
        self.norb = len(self.coords)
        self.dim = len(self.coords[0])
        self.lambda_decay = float(lambda_decay)
        self.direction_of_position_vector = np.zeros(self.dim) #this is stored for calculating dot product with A (magnetic vector potential) later
        # Lattice vectors
        # check if lattice_constant is a set up correctly, if it is a scalar, then extend in one direction
        if np.shape(lattice_constant) == ():
            if self.dim == 1:
                self.lattice_constant = np.ones(self.dim) * float(lattice_constant)
            else:   
                raise ValueError("dimension of lattice_constant and dimension of the system does not match, if you want to set up higher dimension system only extend in one direction, please provide a list of lattice_constant and set the non-extension direction to 0, put non-extension direction to end of the list")
        else:
            self.lattice_constant = np.array(lattice_constant, dtype=float)
            if self.lattice_constant.size != self.dim:
                raise ValueError("dimension of lattice_constant and dimension of the system does not match, if you want to set up higher dimension system only extend in one direction, please provide a list of lattice_constant and set the non-extension direction to 0, put non-extension direction to end of the list")


        # count the non-zero elements in the lattice_constant as the extension direction
        self.extension_direction_number = np.count_nonzero(self.lattice_constant)
        # check if the zeros in lattice_constant is located at the end of the list
        non_extension_direction_number = len(self.lattice_constant) - self.extension_direction_number
        for i in range(non_extension_direction_number):
            if self.lattice_constant[-i-1] != 0:
                raise ValueError("the zeros in lattice_constant is not located at the end of the list, please set the non-extension direction to 0, put non-extension direction to end of the list")
        
        # Check if relative_Hopping is provided
        if relative_Hopping is not None:
            if len(relative_Hopping) != self.norb*(self.norb-1)*2**(self.extension_direction_number-1):
                raise ValueError("Length of `relative_Hopping` must match the existing pairs of orbitals, for example, in two dimension, for 2 orbitals, the length of `relative_Hopping` must be 2 (tao_AB, tao_BA'), for 3 orbitals, the length of `relative_Hopping` must be 6 (tao_AB, tao_BA', tao_AC, tao_CA', tao_BC, tao_CB'), in three dimension, for 2 orbitals, the length of `relative_Hopping` must be 4 (tao_AB, tao_BAx, tao_BAy, tao_BAxy)")
            self.relative_Hopping = np.array(relative_Hopping, dtype=float)
            self.direction_of_position_vector = []
            self.modified_position_vector = []
            if self.extension_direction_number == 1:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.direction_of_position_vector.append(self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
            elif self.extension_direction_number == 2:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for a in range(self.dim):
                    for i in range(self.norb):
                        for j in range(self.norb):
                            if j > i:
                                modified_position_vector = np.zeros(self.dim)
                                modified_position_vector[a] = self.lattice_constant[a]
                                self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                self.modified_position_vector.append(modified_position_vector)
                # current_dim = 2
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            modified_position_vector[1] = self.lattice_constant[1]
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
            elif self.extension_direction_number == 3:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.direction_of_position_vector.append(self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for a in range(self.dim):
                    for i in range(self.norb):
                        for j in range(self.norb):
                            if j > i:
                                modified_position_vector = np.zeros(self.dim)
                                modified_position_vector[a] = self.lattice_constant[a]
                                self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                self.modified_position_vector.append(modified_position_vector)
                # current_dim = 2
                for a in range(self.dim):
                    for b in range(a+1, self.dim):
                        for i in range(self.norb):
                            for j in range(self.norb):
                                if j > i:
                                    modified_position_vector = np.zeros(self.dim)
                                    modified_position_vector[a] = self.lattice_constant[a]
                                    modified_position_vector[b] = self.lattice_constant[b]
                                    self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                    self.modified_position_vector.append(modified_position_vector)
                # current_dim = 3
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            modified_position_vector[1] = self.lattice_constant[1]
                            modified_position_vector[2] = self.lattice_constant[2]
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
        else:
            self.relative_Hopping = []
            self.direction_of_position_vector = []
            self.modified_position_vector = []
            if self.extension_direction_number == 1:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(self.coords[j] - self.coords[i]) / self.lambda_decay))
                            self.direction_of_position_vector.append(self.coords[j] - self.coords[i])
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
            elif self.extension_direction_number == 2:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(self.coords[j] - self.coords[i]) / self.lambda_decay))
                            self.direction_of_position_vector.append(self.coords[j] - self.coords[i])
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for a in range(self.dim):
                    for i in range(self.norb):
                        for j in range(self.norb):
                            if j > i:
                                modified_position_vector = np.zeros(self.dim)
                                modified_position_vector[a] = self.lattice_constant[a]
                                self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                                self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                self.modified_position_vector.append(modified_position_vector)
                # current_dim = 2
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            modified_position_vector[1] = self.lattice_constant[1]
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
            elif self.extension_direction_number == 3:
                # current_dim = 0
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(self.coords[j] - self.coords[i]) / self.lambda_decay))
                            self.direction_of_position_vector.append(self.coords[j] - self.coords[i])
                            self.modified_position_vector.append(np.zeros(self.dim))
                # current_dim = 1
                for a in range(self.dim):
                    for i in range(self.norb):
                        for j in range(self.norb):
                            if j > i:
                                modified_position_vector = np.zeros(self.dim)
                                modified_position_vector[a] = self.lattice_constant[a]
                                self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                                self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                self.modified_position_vector.append(modified_position_vector)
                # current_dim = 2
                for a in range(self.dim):
                    for b in range(a+1, self.dim):
                        for i in range(self.norb):
                            for j in range(self.norb):
                                if j > i:
                                    modified_position_vector = np.zeros(self.dim)
                                    modified_position_vector[a] = self.lattice_constant[a]
                                    modified_position_vector[b] = self.lattice_constant[b]
                                    self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                                    self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                                    self.modified_position_vector.append(modified_position_vector)
                # current_dim = 3
                for i in range(self.norb):
                    for j in range(self.norb):
                        if j > i:
                            modified_position_vector = np.zeros(self.dim)
                            modified_position_vector[0] = self.lattice_constant[0]
                            modified_position_vector[1] = self.lattice_constant[1]
                            modified_position_vector[2] = self.lattice_constant[2]
                            self.relative_Hopping.append(np.exp(-np.linalg.norm(modified_position_vector + self.coords[i] - self.coords[j]) / self.lambda_decay))
                            self.direction_of_position_vector.append(modified_position_vector + self.coords[i] - self.coords[j])
                            self.modified_position_vector.append(modified_position_vector)
        self.mu = mu

        # number of unique (i<j) pairs
        pairs = [(i, j)
                 for i in range(self.norb)
                 for j in range(self.norb)
                 if j > i]
        Npairs = len(pairs)

        # allocate matrices
        self.intra       = np.zeros((self.norb, self.norb), dtype=complex)
        self.inter_upper = np.zeros_like(self.intra)
        self.inter_lower = np.zeros_like(self.intra)

        # 1) intracell hoppings (R = 0)
        for idx, (i, j) in enumerate(pairs):
            t = self.relative_Hopping[idx]
            # fill both (i→j) and its Hermitian partner (j→i)
            self.intra[i, j] = t
            self.intra[j, i] = np.conj(t)

        # 2) intercell hoppings to the “+a” neighbor (R = +a)
        for idx, (i, j) in enumerate(pairs, start=Npairs):
            t = self.relative_Hopping[idx]
            # by convention we put the +a hop in (i←j), i.e. H_{i,j}(+a) = t
            # and its Hermitian partner in (j←i) for R = -a
            i0, j0 = pairs[idx - Npairs]
            self.inter_upper[i0, j0] = t
            self.inter_lower[j0, i0] = np.conj(t)


        # Build k-point grid for Brillouin zone
        if isinstance(nk, int):
            nk_list = [nk] * self.dim
        else:
            nk_list = list(nk)
            if len(nk_list) != self.dim:
                raise ValueError("`nk` must be int or list of length dim")

        axes = [np.linspace(-np.pi/self.lattice_constant[d], np.pi/self.lattice_constant[d], nk_list[d])
                for d in range(self.dim)]
        grids = np.meshgrid(*axes, indexing='ij')
        pts = np.stack([g.flatten() for g in grids], axis=-1)
        self.k_vals = pts  # shape: (prod(nk_list), dim)

        # Placeholder for computed band energies
        self._bands = None


    def buildH(self, k):
        # flatten k to a scalar phase if 1D
        if self.dim == 1:
            φ = float(k) * self.lattice_constant[0]
        else:
            φ = np.dot(k, self.lattice_constant)

        return ( self.intra
               + self.inter_upper * np.exp(+1j*φ)
               + self.inter_lower * np.exp(-1j*φ)
               + np.eye(self.norb) * self.mu )

    def run(self, k=None):
        """
        Diagonalize H(k) at one or many k-points.

        Parameters
        ----------
        k : array_like, shape (M, dim) or (dim,) or None
            If None, uses precomputed self.k_vals.

        Returns
        -------
        ks : ndarray, shape (M, dim)
            k-points used.
        bands : ndarray, shape (norb, M)
            Sorted eigenvalues at each k.
        """
        if self.dim == 1:
            if isinstance(k, (float, int)):
                k_vec = np.array([k])
        else:
            if isinstance(k, (list, np.ndarray)):
                if len(k[0]) != self.dim:
                    raise ValueError(f"each k point must have length {self.dim}")

        M = len(k)
        bands = np.zeros((self.norb, M), dtype=float)
        for idx, kpt in enumerate(k):
            eigs = np.linalg.eigvalsh(self.buildH(kpt))
            bands[:, idx] = np.sort(eigs.real)
        self._bands = bands
        self.k_vals = k
        return k, bands

    def plot(self, k=None):
        """
        Plot band energies vs k for 1D models only.
        """
        if self.dim != 1:
            raise NotImplementedError("Plotting only supported in 1D.")
        if self._bands is None:
            ks, bands = self.run(k)
        else:
            plt.figure(figsize=(8, 4))
            for b in range(self.norb):
                plt.plot(self.k_vals, self._bands[b, :], label=f'Band {b}')
            plt.xlabel('k')
            plt.ylabel('Energy')
            plt.title('Tight-Binding Band Structure')
            plt.legend()
            plt.grid(True)
            plt.show()

    def band_gap(self):
        """
        Compute minimum gap between first two bands over the k-grid.
        """
        if self._bands is None:
            self.run()
        if self.norb < 2:
            return 0.0
        gap = self._bands[1] - self._bands[0]
        return np.min(gap)

    def Floquet(self, data_path, **kwargs):
        """
        Return a FloquetBloch instance with coordinate info.
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        Hk_func = lambda kpt: self.buildH(kpt)
        # pos = np.diag(np.arange(self.norb) * self.a_vec[0])
        floq = FloquetBloch(Hk_func=Hk_func, **kwargs, coords=self.coords, a_vec=self.lattice_constant,
                            norbs=self.norb, data_path=data_path, relative_Hopping=self.relative_Hopping, direction_of_position_vector=self.direction_of_position_vector, translation_vectors=self.modified_position_vector)
        # floq = FloquetBloch(Hk=Hk_func, Edip=pos, **kwargs)
        # Attach coordinate info for extended H build
        return floq
    


class FloquetBloch:
    """
    Periodically driven tight-binding system in Bloch basis.

    Parameters
    ----------
    Hk_func : callable
        Function Hk_func(kpt) returning the static Bloch Hamiltonian H(k).
    omegad : float
        Driving frequency ωₙ.
    E0 : float or array_like
        Field amplitude(s). If list/array, builds one Floquet matrix per E₀.
    nt : int
        Number of Floquet harmonics (matrix size = norbs * nt).
    coords : array_like, shape (norbs, dim)
        Orbital positions within the unit cell.
    a_vec : array_like, shape (dim,)
        Lattice vector(s).
    norbs : int
        Number of orbitals per cell.
    relative_Hopping : list of float
        Exponential-decay hoppings t₍ᵢⱼ₎ in the same order as the pairs list.
    direction_of_position_vector : list of array_like
        Full displacement vectors d₍ᵢⱼ₎ = (rⱼ-rᵢ) + lattice shifts.
    translation_vectors : list of array_like
        Pure lattice-shift R for each hop (i.e. d₍ᵢⱼ₎ - (rⱼ-rᵢ)).
    gauge : str, optional
        Driving gauge (only "Peierls" supported here).
    polarization : array_like, optional
        Field polarization vector (dim-length).
    data_path : str, optional
        Path for caching or output.
    initial_band : int, optional
        Which static band to track by default.
    """
    def __init__(self,
                 Hk_func,
                 omegad,
                 E0,
                 nt,
                 coords,
                 a_vec,
                 norbs,
                 relative_Hopping,
                 direction_of_position_vector,
                 translation_vectors,
                 gauge='Peierls',
                 polarization=None,
                 data_path='MacBook_local_data/floquet_data'):

        # store core parameters
        self.Hk_func  = Hk_func
        self.omegad   = float(omegad)
        self.nt       = int(nt)
        self.coords   = np.array(coords, float)    # (norbs, dim)
        self.a_vec    = np.array(a_vec, float)     # (dim,)
        self.norbs    = int(norbs)
        self.gauge    = gauge
        self.dim      = len(self.a_vec)

        # --- E0 handling: scalar or list ---
        E_arr = np.atleast_1d(E0).astype(float)
        if E_arr.ndim != 1:
            raise ValueError("E0 must be a scalar or 1D array of floats")
        if E_arr.size == 1:
            self._E_list   = None
            self.E0_scalar = float(E_arr[0])
        else:
            self._E_list   = list(E_arr)
            self.E0_scalar = None

        # hopping data (must match length = 2 × #unique pairs)
        self.relative_Hopping            = list(relative_Hopping)
        self.direction_of_position_vector = [np.array(d, float)
                                            for d in direction_of_position_vector]
        self.translation_vectors         = [np.array(R, float)
                                            for R in translation_vectors]

        # build list of unique orbital pairs i<j
        self.pairs = [(i, j)
                      for i in range(self.norbs)
                      for j in range(self.norbs) if j > i]
        expected_len = 2 * len(self.pairs)

        if not (len(self.relative_Hopping) == len(self.direction_of_position_vector)
                == len(self.translation_vectors)
                == expected_len):
            print('len of direction of position vector', len(self.direction_of_position_vector))
            print('len of realtive hopping', len(self.relative_Hopping))
            print('len of translation_vectors', len(self.translation_vectors))
            raise ValueError(f"Expected 2*#pairs={expected_len} hops, "
                             f"got {len(self.relative_Hopping)}")

        # polarization vector
        if polarization is None:
            self.polarization = np.zeros(self.coords.shape[1], float)
        else:
            self.polarization = np.array(polarization, float)
        if self.polarization.size != self.coords.shape[1]:
            raise ValueError("polarization must match coordinate dimension")

        # Floquet block indices range
        self.N0    = self.nt // 2
        self.all_p = np.arange(-self.N0, self.N0 + 1, dtype=int)

        # optional metadata
        self.data_path    = data_path
        self.k = None #placeholder

    # def build_extendedH(self, kpt, Ecur=None):
    #     """
    #     Construct the Floquet Hamiltonian(s) F(k) for one or many E₀.

    #     Parameters
    #     ----------
    #     kpt : float or array_like
    #         Crystal momentum (1D scalar or dim-vector).
    #     Ecur : float, optional
    #         If provided, override the built-in amplitude(s).

    #     Returns
    #     -------
    #     F : (norbs*nt, norbs*nt) complex ndarray, if single E₀  
    #     [F₁, F₂, ...] : list of such arrays, if multiple E₀
    #     """
    #     kpt = np.atleast_1d(kpt).astype(float)
    #     Norbs, nt, ω = self.norbs, self.nt, self.omegad

    #     # decide which E₀'s to loop over
    #     if Ecur is None:
    #         raise ValueError("Please Provide E")
    #     # 1) build each Fourier block H^(p)
    #     Hn = {p: np.zeros((Norbs, Norbs), complex)
    #             for p in self.all_p}

    #     # loop all hops: first intracell, then intercell
    #     for idx, ((i, j), t, dvec, R) in enumerate(zip(
    #             self.pairs + self.pairs,
    #             self.relative_Hopping,
    #             self.direction_of_position_vector,
    #             self.translation_vectors)):
    #         base = self.coords[j] - self.coords[i]
    #         dvec = base + R
    #         # compute Bessel argument
    #         arg = np.dot(self.polarization, dvec) * (Ecur / ω)
    #         # momentum-space phase from the pure lattice shift R
    #         phase = np.exp(1j * np.dot(kpt, R))

    #         # fill each Fourier harmonic block
    #         for p in self.all_p:
    #             J = jv(p, arg)
    #             Hn[p][i, j] += t * J * phase
    #             Hn[p][j, i] += np.conj(t * J * phase)

    #     # 2) assemble the full Floquet matrix F
    #     NF = Norbs * nt
    #     F  = np.zeros((NF, NF), complex)
    #     for n in range(nt):
    #         for m in range(nt):
    #             p = n - m
    #             block = Hn.get(p, np.zeros((Norbs, Norbs), complex))
    #             if n == m:
    #                 # add (n-N0)*ω identity on the diagonal block
    #                 block = block + np.eye(Norbs) * ((n - self.N0) * ω)
    #             i0, i1 = n*Norbs, (n+1)*Norbs
    #             j0, j1 = m*Norbs, (m+1)*Norbs
    #             F[i0:i1, j0:j1] = block

    #     # results.append(F)

    #     # return single array if only one E₀ requested
    #     return F
    #     # return results[0] if len(results) == 1 else results
    def build_extendedH(self, kpt, Ecur=None):

        kpt = np.atleast_1d(kpt).astype(float)
        Norbs, nt, ω = self.norbs, self.nt, self.omegad

        # decide which E0 list to use
        if Ecur is not None:
            E_list = [float(Ecur)]
        elif self._E_list is None:
            E_list = [self.E0_scalar]
        else:
            E_list = list(self._E_list)

        # set up empty Fourier blocks
        Hn = {p: np.zeros((Norbs, Norbs), dtype=complex)
            for p in self.all_p}

        # static orbital positions
        R0 = np.array(self.coords)  # shape (norbs, dim)

        # directional polarization vector
        Avec = np.array(self.polarization)

        # shorthand for second index of pairs
        pairs = self.pairs

        # 1D case: two shifts R=0 and R=+a
        if self.dim == 1:
            a = self.a_vec[0]
            for (i, j), t0, dij in zip(pairs + pairs,
                                    self.relative_Hopping,
                                    self.direction_of_position_vector):
                # note: pairs+pairs so we still consume two entries per pair
                # but here dij was built = (r_j-r_i)+R already
                # R = dij - (r_j-r_i):
                base = R0[j,0] - R0[i,0]
                Rshift = dij[0] - base

                for E0 in E_list:
                    arg0 = (E0/ω) * (Avec[0] * base)
                    arg1 = (E0/ω) * (Avec[0] * (base + a))
                    phase1 = np.exp(1j * kpt * a)

                    for p in self.all_p:
                        J0 = jv(p, arg0)
                        J1 = jv(p, arg1)
                        # intracell hop (R=0)
                        Hn[p][i, j] += t0 * J0
                        Hn[p][j, i] += np.conjugate(t0 * J0)
                        # intercell hop (R=+a)
                        Hn[p][i, j] += t0 * J1 * phase1
                        Hn[p][j, i] += np.conjugate(t0 * J1 * phase1)

        # 2D case: shifts R in { (0,0), (a,0), (0,b), (a,b) }
        elif self.dim == 2:
            a, b = self.a_vec
            shifts = [
                np.array([0.0, 0.0]),
                np.array([a,   0.0]),
                np.array([0.0, b  ]),
                np.array([a,   b  ]),
            ]
            for (i, j), t in zip(pairs * len(shifts), self.relative_Hopping):
                # figure out which shift this t corresponds to
                idx = len(self.pairs)  # reused as counter
                pidx = self.relative_Hopping.index(t) // len(self.pairs)
                R = shifts[pidx]
                base_vec = R0[j] - R0[i]

                for E0 in E_list:
                    # precompute phase and arg for this R
                    arg = (E0/ω) * np.dot(Avec, base_vec + R)
                    phase = np.exp(1j * (kpt[0]*R[0] + kpt[1]*R[1]))

                    for p in self.all_p:
                        J = jv(p, arg)
                        Hn[p][i, j] += t * J * phase
                        Hn[p][j, i] += np.conjugate(t * J * phase)

        # 3D case: R ∈ { (0,0,0), (a,0,0), (0,b,0), (0,0,c),
        #                 (a,b,0), (a,0,c), (0,b,c), (a,b,c) }
        elif self.dim == 3:
            a, b, c = self.a_vec
            shifts = []
            for da in (0.0, a):
                for db in (0.0, b):
                    for dc in (0.0, c):
                        shifts.append(np.array([da, db, dc]))
            for (i, j), t in zip(pairs * len(shifts), self.relative_Hopping):
                pidx = self.relative_Hopping.index(t) // len(self.pairs)
                R = shifts[pidx]
                base_vec = R0[j] - R0[i]

                for E0 in E_list:
                    arg   = (E0/ω) * np.dot(Avec, base_vec + R)
                    phase = np.exp(1j * (kpt[0]*R[0] + kpt[1]*R[1] + kpt[2]*R[2]))
                    for p in self.all_p:
                        J = jv(p, arg)
                        Hn[p][i, j] += t * J * phase
                        Hn[p][j, i] += np.conjugate(t * J * phase)

        else:
            raise NotImplementedError("Floquet build only for dim=1,2,3")

        # Finally assemble the big (Norbs*nt)x(Norbs*nt) Floquet matrix F
        NF = Norbs * nt
        F = np.zeros((NF, NF), dtype=complex)
        for m in range(nt):
            for n in range(nt):
                p = m - n
                block = Hn.get(p, np.zeros((Norbs, Norbs), complex))
                if m == n:
                    block = block + np.eye(Norbs) * ((m - self.N0) * ω)
                i0, i1 = m*Norbs, (m+1)*Norbs
                j0, j1 = n*Norbs, (n+1)*Norbs
                F[i0:i1, j0:j1] = block

        return F



    def track_band(self, k_values, E0=None, quasienergy = None, previous_state = None, filename=None, band_index=None):
        '''
        Compute the energy and corresponding eigenstate of the assigned band for the given k list, stored and saved in local file.
        this is a helper function for .run, usually should not be directly called.
        parameters
        ----------
        k : array_like, shape (M, dim)
            k-point(s) to compute.
        quasienergy: np.array, shape (len(k),)
        previous_state: list of np.array, each np.array element in the list have shape (len(k), Norbs * nt)

        Improve:
        band_index has not been considered yet
        '''
        nt = self.nt
        # if previous_state is None and quasienergy is None:
        #     raise ValueError("One of the information (quasienergy, previous_state) is required to track the band.")
        if E0 != 0 and previous_state is None:
            raise ValueError("previous_state is required to track the band at when external field is applied.")
            
        if filename and os.path.exists(filename):
            print(f"Loading data from {filename}...")   
            return load_data_from_hdf5(filename)    

        NF = self.norbs * nt
        band_energy = np.zeros((len(k_values), self.norbs), dtype=complex)
        band_eigenstates = [np.zeros((len(k_values), NF), dtype=complex) for a in range(self.norbs)]
        Norbs = self.norbs
        omega = self.omegad
        for i, k0 in enumerate(k_values):
            extended_Floquet_hamiltonian = self.build_extendedH(k0, E0)
            if E0 == 0:
                filename = os.path.join(self.data_path, f"band_E{E0:.6f}.h5")
                Real_band_energy, _ = linalg.eigh(self.Hk_func(k0))
                        # Diagonalize
                eigvals, eigvecs = linalg.eigh(extended_Floquet_hamiltonian)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)
                # specify a range to choose the quasienergies, choose the first BZ
                # [-hbar omega/2, hbar * omega/2]
                eigvals_subset = np.zeros(Norbs, dtype=complex)
                eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)
                # check if the Floquet states is complete
                j = 0
                for m in range(NF):
                    if -omega/2 <= eigvals[m].real <= omega/2:
                        eigvals_subset[j]   = eigvals[m]
                        eigvecs_subset[:,j] = eigvecs[:,m]
                        j += 1
                if j != Norbs:
                    print("Error: Number of Floquet states {} is not equal to \
                        the number of orbitals {} in the first BZ. \n".format(j, Norbs))
                    sys.exit()
                # now track one Floquet level per real band
                for band_idx in range(self.norbs):
                    # 1) pick out the ONE real-band energy
                    target_energy = Real_band_energy[band_idx]           # a scalar

                    # 2) form a 1-D distance array
                    distances = np.abs(eigvals - target_energy)          # shape = (NF,)

                    # 3) get the single best Floquet index
                    floq_idx  = np.argmin(distances)                     # integer

                    # 4) assign that ONE scalar into your storage
                    band_energy[i, band_idx]            = eigvals[floq_idx]
                    band_eigenstates[band_idx][i, :]    = eigvecs[:, floq_idx]
                # return band_energy, band_eigenstates
            else:
                filename = os.path.join(self.data_path, f"band_E{E0:.6f}.h5")
                # Diagonalize
                eigvals, eigvecs = linalg.eigh(extended_Floquet_hamiltonian)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)s
                # specify a range to choose the quasienergies, choose the first BZ
                # [-hbar omega/2, hbar * omega/2]
                eigvals_subset = np.zeros(Norbs, dtype=complex)
                eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)
                # check if the Floquet states is complete
                j = 0
                for m in range(NF):
                    if -omega/2 <= eigvals[m].real <= omega/2:
                        eigvals_subset[j]   = eigvals[m]
                        eigvecs_subset[:,j] = eigvecs[:,m]
                        j += 1
                if j != Norbs:
                    print("Error: Number of Floquet states {} is not equal to \
                        the number of orbitals {} in the first BZ. \n".format(j, Norbs))
                    sys.exit()
                # overlap = np.zeros(NF)
                # for i in range(NF):
                #     overlap[i] = np.abs(np.dot(previous_state.conj(), eigvecs[:, i]))**2

                # idx = np.argsort(overlap)[::-1]  # Sort indices by descending overlap

                # for band_idx in range(self.norb):
                #     band_eigenstates[band_idx][:, i] = eigvecs[:, idx[band_idx]]
                #     band_energy[i][band_idx] = eigvals[idx[band_idx]]

                # return band_energy, band_eigenstates
                    # — build overlap matrix: shape (Norbs, NF) —
                overlap = np.zeros((Norbs, NF), dtype=float)
                for band_idx in range(Norbs):
                    prev_vec = previous_state[band_idx][i, :]            # shape (NF,)
                    # dot prev_vec* with every eigvecs[:,m] -> scalar, then abs²
                    # np.vdot does conj(prev_vec)·eigvecs[:,m]
                    for m in range(NF):
                        overlap[band_idx, m] = np.abs(np.vdot(prev_vec, eigvecs[:, m]))**2

                # — greedy assignment so each band picks its best *unused* Floquet level —
                assigned = set()
                assignment = {}  # band_idx -> chosen index m
                for band_idx in range(Norbs):
                    # look through this band’s overlaps in descending order
                    for m in np.argsort(overlap[band_idx])[::-1]:
                        if m not in assigned:
                            assigned.add(m)
                            assignment[band_idx] = m
                            break

                # — store each band’s matched eigenvalue & eigenvector —
                for band_idx, m in assignment.items():
                    band_energy[i, band_idx]         = eigvals[m]
                    band_eigenstates[band_idx][i, :] = eigvecs[:, m]
                # — ensure our N_orbs quasi‐energies are ascending; if not, reorder — 
                # get the real parts for comparison
                vals = band_energy[i].real    # shape (Norbs,)
                # check monotonic non‐decreasing
                if not np.all(np.diff(vals) >= 0):
                    # compute the ascending sort‐order
                    order = np.argsort(vals)
                    # reorder the energies row
                    band_energy[i, :] = band_energy[i, order]
                    # reorder the eigenstates for this k‐point
                    # stash the old row of each band
                    old_states = [ band_eigenstates[b][i, :].copy()
                                for b in range(Norbs) ]
                    for new_b, old_b in enumerate(order):
                        band_eigenstates[new_b][i, :] = old_states[old_b]

                
        save_data_to_hdf5(filename, band_energy, band_eigenstates)
        return band_energy, band_eigenstates



    def run(self, k, nE_steps = None, calculated_bands=None):
        """
        Compute and cache Floquet bands over k and E values.

        k : array_like, shape (M, dim)
        nE_steps : int, ignored if E0 is list/array
        calculated_bands : list[int] or None
        """
        self.k = k
        time_start = time.time()
        if self._E_list is None and nE_steps is None:
            raise ValueError('please specify the steps you wish to take in reaching the final E field strength, or directly provide a list of E when setting up the FloquetBloch class')
        if self._E_list is None:
            E_list = np.linspace(0,self.E0_scalar,nE_steps)
        if self.E0_scalar is None:
            E_list = self._E_list
            if E_list[0] != 0:
                E_list.insert(0, 0)
        
        for i, E_current in enumerate(E_list):
            filename = os.path.join(self.data_path, f"band_E{E_current:.6f}.h5")
            if i == 0:
                # For the first E value, no previous state or quasienergy
                quasienergy, previous_state = self.track_band(k, E_current, filename=filename, band_index=calculated_bands)
            else:
                # Use the previous state and quasienergy to track the band
                quasienergy, previous_state = self.track_band(k, E_current, quasienergy=quasienergy, previous_state=previous_state, filename=filename, band_index=calculated_bands)
            print("current E is ", E_current, " time_past ", time.time()-time_start)
        return quasienergy, previous_state

    def plot_band_structure(self, k=None, E=None,
                            save_band_structure=True, outdir=None):
        """
        Plot (or save) Floquet bands at specified E or list of E.

        E : float or list
            Field value(s) to plot; default last.
        save_band_structure : bool
            If True, save plots instead of showing.
        outdir : str or None
            Directory under data_path to save plots.
        """

        # 1) Make sure we have self.k and data on disk
        if self.k is None:
            # first time: run to generate data for this k-grid
            self.run(k)
        else:
            # if user passes k manually, it must match
            if k is not None and list(k) != list(self.k):
                raise ValueError(
                    "Cannot plot on a different k-grid than was used in .run()."
                )

        # 2) Build list of E values to plot
        if E is None:
            # take last E from your run-list
            if self._E_list is not None:
                E_list = self._E_list
            else:
                E_list = [self.E0_scalar]
        elif isinstance(E, (float, int)):
            E_list = [float(E)]
        else:
            E_list = list(E)

        # 3) Loop over each E, load & plot
        for E_val in E_list:
            fname = os.path.join(self.data_path, f"band_E{E_val:.6f}.h5")
            print(f"Loading band data from {fname}…")
            band_energy, _ = load_data_from_hdf5(fname)
            # band_energy shape: (len(k), Norbs)

            fig, ax = plt.subplots()
            for band_idx in range(band_energy.shape[1]):
                ax.plot(self.k, band_energy[:, band_idx].real,
                        label=f"band {band_idx}")
            ax.set_xlabel("k")
            ax.set_ylabel("Energy")
            ax.set_title(f"Floquet bands at E = {E_val:.6f}")
            ax.legend()

            # 4) either save to disk or show interactively
            if save_band_structure:
                # decide output folder
                if outdir:
                    target = os.path.join(self.data_path, outdir)
                else:
                    target = os.path.join(self.data_path, "Floquet_Band_Structure")
                os.makedirs(target, exist_ok=True)
                out_png = os.path.join(target, f"band_E{E_val:.6f}.png")
                fig.savefig(out_png)
                plt.close(fig)
                print(f"Saved plot to {out_png}")
            else:
                plt.show()


    def winding_number(self, band, E=None):
        """
        Compute the Floquet winding number for a single tracked band by
        reading the cached eigenstates from HDF5 files.

        Parameters
        ----------
        band : int
            Index of the Floquet band to compute (0 ≤ band < self.norbs).
        E : float or list of float, optional
            Field amplitude(s) to evaluate.  If None, uses all E's from the last run.

        Returns
        -------
        float or list of float
            The winding number(s) for the requested band and field value(s).
        """

        # 1) must have run() first
        if self.k is None:
            raise RuntimeError("Call run() before computing winding number.")

        Norbs = self.norbs
        # 2) validate band index
        if not isinstance(band, int) or not (0 <= band < Norbs):
            raise ValueError(f"`band` must be an integer in [0, {Norbs-1}].")

        # 3) build list of E values
        if E is None:
            E_list = self._E_list if self._E_list is not None else [self.E0_scalar]
        elif isinstance(E, (float, int)):
            E_list = [float(E)]
        else:
            E_list = list(E)

        results = []
        # 4) loop over each field amplitude
        for E_val in E_list:
            fname = os.path.join(self.data_path, f"band_E{E_val:.6f}.h5")
            print(f"Loading Floquet data from {fname}…")
            _, all_states = load_data_from_hdf5(fname)
            # all_states is a list of length Norbs, each shape (Nk, NF)
            psi_k_NF = all_states[band]      # shape = (Nk, NF)
            # Transpose to get shape (NF, Nk)
            vecs = psi_k_NF.T
            Nk = vecs.shape[1]

            # 5) build the projector chain P = |ψ₀⟩⟨ψ₀|⋯|ψ_{Nk-1}⟩⟨ψ_{Nk-1}|
            #    then extract the total phase from Tr P
            P = np.outer(vecs[:, 0], np.conjugate(vecs[:, 0]))
            for j in range(1, Nk):
                v = vecs[:, j]
                v = v / np.linalg.norm(v)
                P = P @ np.outer(v, np.conjugate(v))

            angle = np.angle(np.round(np.trace(P)),5)
            # original normalization was (angle mod 2π) / π
            winding = (angle % (2 * np.pi)) / np.pi
            results.append(winding)

        # 6) return scalar if one E, else list
        return results[0] if len(results) == 1 else results


def test_Gomez_Leon_2013(E0 = 20, number_of_step_in_b = 10, nt = 21):
    """
    Test the Gomez-Leon 2013 model but using TightBinding and FloquetBloch classes, calculate the winding number and finally plot the heatmap
    """
    # Parameters

    omega = 100
    E_over_omega = np.linspace(0, E0/omega, 200)
    E = [e * omega for e in E_over_omega]
    k_vals = np.linspace(-np.pi, np.pi, 100)
    b_grid = np.linspace(0, 1.0, number_of_step_in_b)
    for b in b_grid:
        # Create tight-binding model
        coords = [[0], [b]]
        tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], nk=50, mu=0.0, relative_Hopping=[1,1.5])
        
        # Run Floquet analysis
        floquet_model = tb_model.Floquet(omegad=omega, E0=E, nt=nt, polarization=[1], data_path=f'MacBook_local_data/floquet_data_Gomez_Leon_test_b={b:.2f}')
        energies, states = floquet_model.run(k_vals)
        
        # Plot band structure
        floquet_model.plot_band_structure(k_vals)

def Floquet_Winding_number_Peierls_GL2013(H0, k, Nt, E_over_omega ,quasiE = None, previous_state = None, b=0.5, t=1):
    """
    Build and diagonalize the Floquet Hamiltonian for a 1D system,
    then group the 2*N_t eigenvalues/eigenstates into two Floquet bands.
    choose the correct Floquet branch if E = 0 (by comparing with the directly diagonalized energies)
    if E != 0, choose the correct branch by doing overlap with the previous state.

    H(t) is SSH model with k replaced by k - E_0/w * sin(wt) by Peierls substitution
    """
    a = 1 #lattice constant, need to be modifyed accordingly, later need to be included into the variables
    A = E_over_omega
    omega = 100
    if E_over_omega == 0:
        Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
        NF = Norbs * Nt           # dimension of Floquet matrix
        N0 = -(Nt-1)//2           # shift for Fourier indices

        Hn = [np.array([[0, 0], [0, 0]], dtype=complex) for a in range(Nt)]
        # Hn[0] = H0 + np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex)

        # Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
        Hn_b = [np.array([[0, t], [t, 0]], dtype=complex) for a in range(Nt)]
        Hn_a_b = [np.array([[0, np.exp(1j*k)], [np.exp(-1j*k), 0]], dtype=complex) for a in range(Nt)]
        # Hn_b = [np.array([[0, t*np.exp(-1j*k*b)], [t*np.exp(1j*k*b), 0]], dtype=complex) for a in range(Nt)]
        # Hn_a_b = [np.array([[0, np.exp(1j*k*(1-b))], [np.exp(-1j*k*(1-b)), 0]], dtype=complex) for a in range(Nt)]

        for i in range(Nt):
            Hn[i][0][1] = Hn_b[i][0][1] * jv(-i,A*b) + Hn_a_b[i][0][1] * jv(i,A*(1-b)) 
            Hn[i][1][0] = Hn_b[i][1][0] * jv(i,A*b) + Hn_a_b[i][1][0] * jv(-i,A*(1-b))
        Hn[0] = H0
        # Construct the Floquet matrix
        # need to be modified.
        F = np.zeros((NF, NF), dtype=complex)
        for n in range(Nt):
            for m in range(Nt):
                for k in range(Norbs):
                    for l in range(Norbs):
                        i = Norbs*n + k
                        j = Norbs*m + l
                        # Hamiltonian block + photon block
                        F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
                                   - (n + N0)*omega*(n==m)*(k==l))

        # Diagonalize
        eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)
        # specify a range to choose the quasienergies, choose the first BZ
        # [-hbar omega/2, hbar * omega/2]
        eigvals_subset = np.zeros(Norbs, dtype=complex)
        eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)
        # check if the Floquet states is complete
        j = 0
        for i in range(NF):
            if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
                eigvals_subset[j] = eigvals[i]
                eigvecs_subset[:,j] = eigvecs[:,i]
                j += 1
        if j != Norbs:
            print("Error: Number of Floquet states {} is not equal to \
                the number of orbitals {} in the first BZ. \n".format(j, Norbs))
            sys.exit()
        eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
        eigvals_copy = np.array(eigvals_copy)
        idx = np.argsort(eigvals_copy.real)
        occ_state = eigvecs[:, idx[0]]
        occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
        return occ_state, occ_state_energy
    else:
        Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
        NF = Norbs * Nt           # dimension of Floquet matrix
        N0 = -(Nt-1)//2           # shift for Fourier indices
        Hn = [np.array([[0, 0], [0, 0]], dtype=complex) for a in range(Nt)]
        # Hn[0] = H0 + np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex)

        # Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
        Hn_b = [np.array([[0, t], [t, 0]], dtype=complex) for a in range(Nt)]
        Hn_a_b = [np.array([[0, np.exp(1j*k)], [np.exp(-1j*k), 0]], dtype=complex) for a in range(Nt)]
        for i in range(Nt):
            Hn[i][0][1] = Hn_b[i][0][1] * jv(-i,A*b) + Hn_a_b[i][0][1] * jv(i,A*(1-b)) 
            Hn[i][1][0] = Hn_b[i][1][0] * jv(i,A*b) + Hn_a_b[i][1][0] * jv(-i,A*(1-b))
        Hn[0] = H0
        # Construct the Floquet matrix
        # need to be modified.
        F = np.zeros((NF, NF), dtype=complex)
        for n in range(Nt):
            for m in range(Nt):
                for k in range(Norbs):
                    for l in range(Norbs):
                        i = Norbs*n + k
                        j = Norbs*m + l
                        # Hamiltonian block + photon block
                        F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
                                   - (n + N0)*omega*(n==m)*(k==l))
                        
        # Diagonalize
        eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)s
        # specify a range to choose the quasienergies, choose the first BZ
        # [-hbar omega/2, hbar * omega/2]
        eigvals_subset = np.zeros(Norbs, dtype=complex)
        eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


        # check if the Floquet states is complete
        j = 0
        for i in range(NF):
            if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
                eigvals_subset[j] = eigvals[i]
                eigvecs_subset[:,j] = eigvecs[:,i]
                j += 1
        if j != Norbs:
            print("Error: Number of Floquet states {} is not equal to \
                the number of orbitals {} in the first BZ. \n".format(j, Norbs))
            sys.exit()
        overlap = np.zeros(NF)
        for i in range(NF):
            for j in range(NF):
                overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
                # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
            # if np.abs(overlap[i]) < 0.05:
            #     overlap[i]=0
            # else:
            #     print(overlap[i])
        idx = np.argsort(abs(overlap))

        occ_state = eigvecs[:,idx[-1]]
        occ_state_energy = eigvals[idx[-1]]
        # for i in range(len(overlap)):
        #     # occ_state += eigvecs[:,i]*overlap[i]
        #     occ_state_energy += eigvals[i] * overlap[i]**2
        # # occ_state /=np.linalg.norm(occ_state)
        
        return occ_state, occ_state_energy



# class Floquet:
#     """
#     peridically driven tight-binding system with a single frequency

#     TODO: add more harmonics so it can treat a second harmonic driving
#     """
#     def __init__(self, H , omegad, E0, nt, gauge='Peierls', polarization='linear'):

#         # super().__init__(H, edip)
#         self.H = H # a function of k,
#         self.nt = nt
#         self.E0 = E0
#         self.gauge = gauge
#         self.polarization = polarization     
#         self.omegad = omegad # driving freqency

#     def momentum_matrix_elements(self):
#         """
#         get momentum matrix elements by commutation relation
#         .. math::
#             p_{fi} = i E_{fi} r_{fi}

#         Returns
#         -------
#         p : TYPE
#             DESCRIPTION.

#         """
#         E = self.E
#         p = 1j * np.subtract.outer(E, E) * self.edip
#         return p

#     def phase_diagram(self, k_values, E0_values, omega_values, save_dir,
#                         save_band_plot=True, nt=61, v=0.2, w=0.15):
#         """
#         Run Floquet winding number phase diagram over (E0, omega).

#         Parameters:
#         - k_values: np.array, k points along BZ
#         - E0_values: 1D array of E0 field amplitudes
#         - omega_values: 1D array of driving frequencies
#         - save_dir: str, root directory for data and plots
#         - save_band_plot: bool, whether to save band plots
#         - nt, v, w: model and time-discretization parameters
#         """
#         os.makedirs(save_dir, exist_ok=True)
#         data_dir = os.path.join(save_dir, "h5")
#         band_dir = os.path.join(save_dir, "bands") if save_band_plot else None
#         if band_dir: os.makedirs(band_dir, exist_ok=True)

#         winding_real = np.zeros((len(E0_values), len(omega_values)))
#         winding_int = np.zeros_like(winding_real)

#         total = len(E0_values) * len(omega_values)
#         start = time.time()
#         idx = 0

#         for j, omega in enumerate(omega_values):
#             T = 2 * np.pi / omega
#             pre_occ = pre_con = None
#             for i, E0 in enumerate(E0_values):
#                 self.E0 = E0
#                 self.omegad = omega

#                 fname = os.path.join(data_dir, f"data_E0_{E0:.5f}_omega_{omega:.4f}.h5")

#                 occ, occ_e, con, con_e, draw = track_valence_band(
#                     k_values, T, E0, omega, previous_val=pre_occ, previous_con=pre_con,
#                     v=v, w=w, nt=nt, filename=fname
#                 )
#                 if draw and band_dir:
#                     figure(occ_e, con_e, k_values, E0, omega, save_folder=band_dir)


#                 w_real = berry_phase_winding(k_values, occ)
#                 winding_real[i, j] = w_real
#                 winding_int[i, j] = round(w_real) if not np.isnan(w_real) else 0

#                 pre_occ, pre_con = occ, con

#                 idx += 1
#                 elapsed = time.time() - start
#                 remaining = elapsed / idx * (total - idx)
#                 progress = f"[{idx}/{total}]---------- E0={E0:.4f}, omega={omega:.4f} | Remaining: {remaining/60:.1f} min"
#                 print(progress, end="\r")

#         # Save final phase diagram
#         fig, axs = plt.subplots(1, 2, figsize=(14, 5))
#         im0 = axs[0].imshow(winding_real, aspect='auto', origin='lower',
#                             extent=[omega_values[0], omega_values[-1], E0_values[0], E0_values[-1]])
#         axs[0].set_title("Winding (Real)"); axs[0].set_xlabel("omega"); axs[0].set_ylabel("E0")
#         fig.colorbar(im0, ax=axs[0])

#         im1 = axs[1].imshow(winding_int, aspect='auto', origin='lower',
#                             extent=[omega_values[0], omega_values[-1], E0_values[0], E0_values[-1]])
#         axs[1].set_title("Winding (Integer)")
#         fig.colorbar(im1, ax=axs[1])

#         fig.tight_layout()
#         fig.savefig(os.path.join(save_dir, "winding_phase_diagram.png"))
#         plt.close(fig)
        
#         # return

#     def run_phase_diagram_GL2013(self, k_vals, E0_over_omega_vals, b_vals, save_dir,
#                              nt=61, t=1.5, omega=100, save_band_plot=False):


#         os.makedirs(save_dir, exist_ok=True)
#         band_dir = os.path.join(save_dir, "bands") if save_band_plot else None
#         data_dir = os.path.join(save_dir, "h5")
#         os.makedirs(data_dir, exist_ok=True)
#         if save_band_plot:
#             os.makedirs(band_dir, exist_ok=True)

#         wind_real = np.zeros((len(E0_over_omega_vals), len(b_vals)))
#         wind_int = np.zeros_like(wind_real)
#         start = time.time()

#         for j, b in enumerate(b_vals):
#             pre_occ = pre_con = None
#             for i, E0o in enumerate(E0_over_omega_vals):
#                 fname = os.path.join(data_dir, f"data_E0_over_omega_{E0o:.6f}_b_{b:.3f}_t_{t:.2f}.h5")
#                 occ, occ_e, con, con_e, draw = track_valence_band_GL2013(k_vals, E0o,
#                                     previous_val=pre_occ, previous_con=pre_con,
#                                     nt=nt, filename=fname, b=b, t=t, omega=omega)

#                 if draw and save_band_plot:
#                     plt.figure(figsize=(8, 6))
#                     plt.plot(k_vals, occ_e, label=f'Val_E0_over_omega = {E0o:.6f}, b = {b:.2f},t = {t:.2f}')
#                     plt.plot(k_vals, con_e, label=f'Val_E0_over_omega = {E0o:.6f}, b = {b:.2f},t = {t:.2f}')
#                     plt.title(f"$E_0/\omega$ = {E0o:.2f}, b = {b:.2f}")
#                     plt.xlabel("k")
#                     plt.ylabel("Quasienergy")
#                     plt.legend()
#                     plt.grid(True)
#                     plt.savefig(os.path.join(band_dir, f"band_E0o_{E0o:.4f}_b_{b:.2f}.png"))
#                     plt.close()

#                 w_real = berry_phase_winding(k_vals, occ)
#                 wind_real[i, j] = w_real
#                 wind_int[i, j] = round(w_real) if not np.isnan(w_real) else 0
#                 pre_occ, pre_con = occ, con

#                 elapsed = time.time() - start
#                 remaining = (len(E0_over_omega_vals) * len(b_vals) - (j * len(E0_over_omega_vals) + i)) * elapsed / (i + 1)
#                 print(f"[{i+1}/{len(E0_over_omega_vals)}], ---------- b = {b:.3f} E0/omega = {E0o:.3f} | Remaining: {remaining/60:.1f} min", end="\r")

#         # Plot winding phase diagram
#         fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#         im0 = axs[0].imshow(wind_real, aspect='auto', origin='lower',
#                             extent=[b_vals[0], b_vals[-1], E0_over_omega_vals[0], E0_over_omega_vals[-1]])
#         axs[0].set_title("Winding (Real)")
#         axs[0].set_xlabel("b")
#         axs[0].set_ylabel("E0/ω")
#         fig.colorbar(im0, ax=axs[0])

#         im1 = axs[1].imshow(wind_int, aspect='auto', origin='lower',
#                             extent=[b_vals[0], b_vals[-1], E0_over_omega_vals[0], E0_over_omega_vals[-1]])
#         axs[1].set_title("Winding (Integer)")
#         axs[1].set_xlabel("b")
#         axs[1].set_ylabel("E0/ω")
#         fig.colorbar(im1, ax=axs[1])
#         fig.tight_layout()
#         fig.savefig(os.path.join(save_dir, "winding_phase_diagram_GL2013.png"))
#         plt.close()



#     def winding_number(self, T, quasi_E = None, previous_state = None, gauge='length'): # To be modified
#         H0 = self.H
#         E0 = self.E0
#         nt = self.nt 
#         omegad = self.omegad
#         nt = self.nt 
#         omegad = self.omegad

#         if gauge == 'length': # electric dipole

#             H1 = -0.5 * self.edip * E0

#         elif gauge == 'velocity':

#             H1 = 0.5j * self.momentum() * E0/omegad
        
#         elif gauge == 'peierls':
        
#             occ_state, occ_state_energy = Floquet_Winding_number(H0, H1, nt, omegad, T, E0, quasi_E, previous_state)

#         return occ_state, occ_state_energy
    
#     def winding_number_Peierls(self, T, k, quasi_E = None, previous_state = None, gauge='length',w=0.2): # To be modified
#         H0 = self.H
#         E0 = self.E0
#         nt = self.nt 
#         omegad = self.omegad
#         occ_state, occ_state_energy = Floquet_Winding_number_Peierls(H0, k, nt, omegad, T, E0, quasi_E, previous_state, w=w)
#         return occ_state, occ_state_energy

#     def winding_number_Peierls_GL2013(self, k, quasi_E = None, previous_state = None, gauge='length', b=0.5, t=1, E_over_omega = 1): # To be modified
#         H0 = self.H
#         E0 = self.E0
#         nt = self.nt 
#         omegad = self.omegad
#         occ_state, occ_state_energy = Floquet_Winding_number_Peierls_GL2013(H0, k, nt, E_over_omega, quasi_E, previous_state, b=b, t=t)
#         return occ_state, occ_state_energy

#     def winding_number_Peierls_circular(self,k0, nt, omega, T, E0,
#                                                     delta_x, delta_y,
#                                                     a, t0, xi,
#                                                     quasi_E,                 # or None
#                                                     previous_state):
#         H0 = self.H
#         E0 = self.E0
#         nt = self.nt 
#         occ_state, occ_state_energy = Floquet_Winding_number_Peierls_circular(k0,                     # Bloch momentum
#                                             nt, omega, T, E0,      # drive
#                                             delta_x, delta_y,      # geometry
#                                             a=a, t0=t0, xi=xi, # lattice & decay
#                                             quasiE=quasi_E, previous_state=previous_state)
#         return occ_state, occ_state_energy
    

#     def velocity_to_length(self):
#         # transform the truncated velocity gauge Hamiltonian to length gauge
#         pass
    
#     def propagator(self, t1, t2=0):
#         pass


# class FloquetGF(FloquetBloch):
#     """
#     Floquet Green's function formalism
#     """
#     def __init__(self, mol, amplitude, omegad, nt):
#         # super.__init__(self, t, obj)
#         self._mol = mol
#         self.omegad = omegad
#         self.nt = nt
#         self.norbs = mol.size

#         self.quasienergy, self.floquet_modes = self.spectrum(amplitude, nt, omegad)
#         self.eta = 1e-3
        
#     def bare_gf(self):
#         # bare Green's functions
#         quasienergy, fmodes = self.quasienergy, self.floquet_modes
#         pass
        
#     def retarded(self, omega, representation='floquet'):
#         nt = self.nt
#         norb = self.norb
#         omegad = self.omegad
        
#         # nomega = len(omega)
        
#         if representation == 'floquet':
            
#             # the frequency should be in the first BZ
#             # omega = np.linspace(-omegad/2, omegad/2, nomega)
#             assert -omegad/2 < omega < omegad/2
                
#             gr = np.zeros((nt, nt, norb, norb), dtype=complex)
            
#             for m in range(nt):
#                 for n in range(nt):
#                     pass
                            
#         elif representation == 'wigner':
            
#             gr = np.zeros((nt, norb, norb), dtype=complex)
        
#         # else:
#         #     raise ValueError('Representation {} does not exist. \
#         #                      Allowed values are `floquet`, `wigner`'.format(representation))
            
#         return gr
    
#     def wigner_to_floquet(self, g, omega):
#         # transform from the Wigner representation to the Floquet representation 
#         # Ref: Tsuji, Oka, Aoki, PRB 78, 235124 (2008)
#         omegad = self.omegad
        
#         assert -omegad/2 < omega < omegad/2
        
#         # l = m - n 
#         # gf[m, n, omega] = g[l, omega + (m + n)/2 * omegad  ] 
#         pass
    
#     def advanced(self):
#         pass
    
#     def lesser(self):
#         pass
    
#     def greater(self):
#         pass

# def _quasiE(H0, H1, Nt, omega):
#     """
#     Construct the Floquet hamiltonian of size Norbs * Nt

#     The total Hamiltonian is
#     .. math::
#         H = H_0 + H_1 \cos(\Omega * t)

#     INPUT
#         Norbs : number of orbitals
#         Nt    : number of Fourier components
#         E0    : electric field amplitude
#     """

#     Norbs = H0.shape[-1]

#     #print('transition dipoles \n', M)

#     # dimensionality of the Floquet matrix
#     NF = Norbs * Nt
#     F = np.zeros((NF,NF), dtype=complex)

#     N0 = -(Nt-1)/2 # starting point for Fourier companent of time exp(i n w t)

#     idt = np.identity(Nt)
#     idm = np.identity(Norbs)
#     # construc the Floquet H for a general tight-binding Hamiltonian
#     for n in range(Nt):
#         for m in range(Nt):

#             # atomic basis index
#             for k in range(Norbs):
#                 for l in range(Norbs):

#                 # map the index i to double-index (n,k) n : time Fourier component
#                 # with relationship for this :  Norbs * n + k = i

#                     i = Norbs * n + k
#                     j = Norbs * m + l
#                     F[i,j] = HamiltonFT(H0, H1, n-m)[k,l] + (n + N0) \
#                              * omega * idt[n,m] * idm[k,l]


#     # for a two-state model

# #    for n in range(Nt):
# #        for m in range(Nt):
# #            F[n * Norbs, m * Norbs] = (N0 + n) * omega * delta(n,m)
# #            F[n * Norbs + 1, m * Norbs + 1] = (onsite1 + (N0+n) * omega) * delta(n,m)
# #            F[n * Norbs, m * Norbs + 1] = t * delta(n,m+1)
# #            F[n * Norbs + 1, m * Norbs] = t * delta(n,m-1)
#     #print('\n Floquet matrix \n', F)

#     # compute the eigenvalues of the Floquet Hamiltonian,
#     eigvals, eigvecs = linalg.eigh(F)

#     #print('Floquet quasienergies', eigvals)

#     # specify a range to choose the quasienergies, choose the first BZ
#     # [-hbar omega/2, hbar * omega/2]
#     eigvals_subset = np.zeros(Norbs, dtype=complex)
#     eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


#     # check if the Floquet states is complete
#     j = 0
#     for i in range(NF):
#         if  eigvals[i] < omega/2.0 and eigvals[i] > -omega/2.0:
#             eigvals_subset[j] = eigvals[i]
#             eigvecs_subset[:,j] = eigvecs[:,i]
#             j += 1
#     if j != Norbs:
#         print("Error: Number of Floquet states {} is not equal to \
#               the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#         sys.exit()


#     # now we have a complete linear independent set of solutions for the time-dependent problem
#     # to compute the coefficients before each Floquet state if we start with |alpha>
#     # At time t = 0, constuct the overlap matrix between Floquet modes and system eigenstates
#     # G[j,i] = < system eigenstate j | Floquet state i >
#     G = np.zeros((Norbs,Norbs), dtype=complex)
#     for i in range(Norbs):
#         for j in range(Norbs):
#             tmp = 0.0
#             for m in range(Nt):
#                 tmp += eigvecs_subset[m * Norbs + j, i]
#             G[j,i] = tmp


#     # to plot G on site basis, transform it to site-basis representation
#     #Gsite = U.dot(G)

#     return eigvals_subset, eigvecs_subset


# def quasiE(H0, H1, Nt, omega, method=1):
#     """
#     Construct the Floquet hamiltonian of size Norbs * Nt

#     The total Hamiltonian is
#     .. math::
#         H = H_0 + H_1 (\exp(i*\Omega * t)+\exp(-i*\Omega * t))
#         or H = H_0 + 2 * H_1 \cos(\Omega * t)

#     INPUT
#         Norbs : number of orbitals
#         Nt    : number of Fourier components
#         E0    : electric field amplitude
#     """
#     if method == 1:
#         Norbs = H0.shape[-1]

#         #print('transition dipoles \n', M)

#         # dimensionality of the Floquet matrix
#         NF = Norbs * Nt
#         F = np.zeros((NF,NF), dtype=complex)

#         N0 = -(Nt-1)/2 # starting point for Fourier components of time exp(-i n w t)

#         idt = np.identity(Nt)
#         idm = np.identity(Norbs)
#         # construc the Floquet H for a general tight-binding Hamiltonian
#         for n in range(Nt):
#             for m in range(Nt):

#                 # atomic basis index
#                 for k in range(Norbs):
#                     for l in range(Norbs):

#                     # map the index i to double-index (n,k) n : time Fourier component
#                     # with relationship for this :  Norbs * n + k = i

#                         i = Norbs * n + k
#                         j = Norbs * m + l
#                         F[i,j] = HamiltonFT(H0, H1, n-m)[k,l] - (n + N0) \
#                                 * omega * idt[n,m] * idm[k,l]


#         # for a two-state model

#     #    for n in range(Nt):
#     #        for m in range(Nt):
#     #            F[n * Norbs, m * Norbs] = (N0 + n) * omega * delta(n,m)
#     #            F[n * Norbs + 1, m * Norbs + 1] = (onsite1 + (N0+n) * omega) * delta(n,m)
#     #            F[n * Norbs, m * Norbs + 1] = t * delta(n,m+1)
#     #            F[n * Norbs + 1, m * Norbs] = t * delta(n,m-1)
#         #print('\n Floquet matrix \n', F)

#         # compute the eigenvalues of the Floquet Hamiltonian,
#         eigvals, eigvecs = linalg.eigh(F)

#         #print('Floquet quasienergies', eigvals)

#         # specify a range to choose the quasienergies, choose the first BZ
#         # [-hbar omega/2, hbar * omega/2]
#         eigvals_subset = np.zeros(Norbs, dtype=complex)
#         eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


#         # check if the Floquet states is complete
#         j = 0
#         for i in range(NF):
#             if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
#                 eigvals_subset[j] = eigvals[i]
#                 eigvecs_subset[:,j] = eigvecs[:,i]
#                 j += 1
#         if j != Norbs:
#             print("Error: Number of Floquet states {} is not equal to \
#                 the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#             sys.exit()


#         # now we have a complete linear independent set of solutions for the time-dependent problem
#         # to compute the coefficients before each Floquet state if we start with |alpha>
#         # At time t = 0, constuct the overlap matrix between Floquet modes and system eigenstates
#         # G[j,i] = < system eigenstate j | Floquet state i >
#         G = np.zeros((Norbs,Norbs), dtype=complex)
#         for i in range(Norbs):
#             for j in range(Norbs):
#                 tmp = 0.0
#                 for m in range(Nt):
#                     tmp += eigvecs_subset[m * Norbs + j, i]
#                 G[j,i] = tmp


#         # to plot G on site basis, transform it to site-basis representation
#         Gsite = eigvecs_subset.dot(G)

#         return eigvals_subset, eigvecs_subset, G
#     elif method == 2:
#             # Use Diagonalization Propagator method
#             time_step = 5000
#             dt = 2 * np.pi / (time_step * omega)  # Time step for propagator
#             U = np.eye(H0.shape[0], dtype=complex)  # Initialize the propagator
#             for t in range(time_step):
#                 time = t * dt
#                 H_t = H0 + H1 * (np.exp(1j*omega * time)+np.exp(-1j*omega * time))
#                 U = linalg.expm(-1j * H_t * dt) @ U  # Update the propagator

#             # Diagonalize the propagator to get quasi-energies and modes
#             eigvals, eigvecs = np.linalg.eig(U)
#             quasi_energies = np.angle(eigvals) * omega / (2 * np.pi)
#                     # Compute the overlap matrix G
#             Norbs = H0.shape[0]
#             G = np.zeros((Norbs, Norbs), dtype=complex)
#             for i in range(Norbs):
#                 for j in range(Norbs):
#                     G[j, i] = np.sum(eigvecs[:, i].conjugate() * eigvecs[:, j])

#             # Transform G to the site basis
#             Gsite = eigvecs.dot(G)
#             return quasi_energies, eigvecs, Gsite

#     else:
#         raise ValueError(f"Method {method} not recognized. Use 1 for Floquet or 2 for Diagonalization_Propagator.")

# def HamiltonFT(H0, H1, n):
#     """
#     Fourier transform of the Hamiltonian matrix, required to construct the
#     Floquet Hamiltonian

#     INPUT
#         n : Fourier component index
#         H0: time-independent part of the Hamiltonian
#         H1: time-dependent part of the Hamiltonian
#     """
#     Norbs = H0.shape[-1]

#     if n == 0:
#         return H0
    
#     elif n == 1:
#         return H1

#     elif n == -1:
#         return dag(H1)

#     else:
#         return np.zeros((Norbs,Norbs))




# def HamiltonFT_peierls(Hn, n):
#     """
#     H(t) is SSH model with k replaced by k - E_0/w * sin(wt) by Peierls substitution
#     Use Jacobi-Anger expansion to construct the Hamiltonian in extended space
#     """

#     Norbs = Hn[0].shape[-1]
#     if n >= 0:
#         return Hn[n]
    
#     elif n < 0:
#         return dag(Hn[n])

#     else:
#         return np.zeros((Norbs,Norbs))


# def group_floquet_quasienergies(eigvals, eigvecs, omega=1.0, n_bands=2):
#     """
#     Identify 'n_bands' Floquet bands by clustering eigenvalues based on
#     their fractional part mod '1'. Then sort each band and return
#     the grouped eigenvalues/eigenvectors.

#     Parameters
#     ----------
#     eigvals : array_like
#         Floquet eigenvalues (length = n_bands * N_t for a 2-level system).
#     eigvecs : ndarray
#         Corresponding eigenvectors (shape = (N_F, N_F)), where columns
#         match the order of 'eigvals'.
#     omega : float
#         Driving frequency (if ~1.0, we do mod 1).
#     n_bands : int
#         Number of bands to split into (2 for a two-level system).

#     Returns
#     -------
#     band_vals : list of 1D arrays
#         A list of length 'n_bands'; each entry is a sorted array of
#         eigenvalues belonging to that band.
#     band_vecs : list of 2D arrays
#         A list of length 'n_bands'; each entry is a 2D array of the
#         corresponding eigenvectors (columns match the sorted eigenvalues).
#     """
#     eigvals = np.asarray(eigvals)
#     # Sort globally first
#     idx_sort = np.argsort(eigvals)
#     eigvals_sorted = eigvals[idx_sort]
#     eigvecs_sorted = eigvecs[:, idx_sort]

#     # Cluster the fractional parts mod 'omega' in 1D
#     frac = np.mod(eigvals_sorted, 1)
#     km = KMeans(n_clusters=n_bands, random_state=0).fit(frac.reshape(-1,1))
#     labels = km.labels_

#     band_vals = []
#     band_vecs = []
#     for band_idx in range(n_bands):
#         # Extract all eigenvalues/vectors belonging to cluster band_idx
#         these_vals = eigvals_sorted[labels == band_idx]
#         these_vecs = eigvecs_sorted[:, labels == band_idx]
#         # Sort them by ascending eigenvalue
#         sub_idx = np.argsort(these_vals)
#         these_vals = these_vals[sub_idx]
#         these_vecs = these_vecs[:, sub_idx]
#         band_vals.append(these_vals)
#         band_vecs.append(these_vecs)
#     return band_vals, band_vecs


# def Floquet_Winding_number(H0, H1, Nt, omega, T, E ,quasiE = None, previous_state = None):
#     """
#     Build and diagonalize the Floquet Hamiltonian for a 1D system,
#     then group the 2*N_t eigenvalues/eigenstates into two Floquet bands.
#     choose the correct Floquet branch if E = 0 (by comparing with the directly diagonalized energies)
#     if E != 0, choose the correct branch by doing overlap with the previous state.

#     H(t) = H0 + 2*H1*cos(omega * t)
#     """
#     if E == 0:
#         Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
#         NF = Norbs * Nt           # dimension of Floquet matrix
#         N0 = -(Nt-1)//2           # shift for Fourier indices

#         # Construct the Floquet matrix
#         F = np.zeros((NF, NF), dtype=complex)
#         for n in range(Nt):
#             for m in range(Nt):
#                 for k in range(Norbs):
#                     for l in range(Norbs):
#                         i = Norbs*n + k
#                         j = Norbs*m + l
#                         # Hamiltonian block + photon block
#                         F[i, j] = (HamiltonFT(H0, H1, n-m)[k, l]
#                                    - (n + N0)*omega*(n==m)*(k==l))

#         # Diagonalize
#         eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)
#         # specify a range to choose the quasienergies, choose the first BZ
#         # [-hbar omega/2, hbar * omega/2]
#         eigvals_subset = np.zeros(Norbs, dtype=complex)
#         eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)
#         # check if the Floquet states is complete
#         j = 0
#         for i in range(NF):
#             if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
#                 eigvals_subset[j] = eigvals[i]
#                 eigvecs_subset[:,j] = eigvecs[:,i]
#                 j += 1
#         if j != Norbs:
#             print("Error: Number of Floquet states {} is not equal to \
#                 the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#             sys.exit()
#         eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
#         eigvals_copy = np.array(eigvals_copy)
#         idx = np.argsort(eigvals_copy.real)
#         occ_state = eigvecs[:, idx[0]]
#         occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
#         return occ_state, occ_state_energy
#     else:
#         Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
#         NF = Norbs * Nt           # dimension of Floquet matrix
#         N0 = -(Nt-1)//2           # shift for Fourier indices

#         # Construct the Floquet matrix
#         F = np.zeros((NF, NF), dtype=complex)
#         for n in range(Nt):
#             for m in range(Nt):
#                 for k in range(Norbs):
#                     for l in range(Norbs):
#                         i = Norbs*n + k
#                         j = Norbs*m + l
#                         # Hamiltonian block + photon block
#                         F[i, j] = (HamiltonFT(H0, H1, n-m)[k, l]
#                                    - (n + N0)*omega*(n==m)*(k==l))

#         # Diagonalize
#         eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)s
#         # specify a range to choose the quasienergies, choose the first BZ
#         # [-hbar omega/2, hbar * omega/2]
#         eigvals_subset = np.zeros(Norbs, dtype=complex)
#         eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


#         # check if the Floquet states is complete
#         j = 0
#         for i in range(NF):
#             if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
#                 eigvals_subset[j] = eigvals[i]
#                 eigvecs_subset[:,j] = eigvecs[:,i]
#                 j += 1
#         if j != Norbs:
#             print("Error: Number of Floquet states {} is not equal to \
#                 the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#             sys.exit()
#         overlap = np.zeros(NF)
#         for i in range(NF):
#             for j in range(NF):
#                 overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
#                 # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
#             # if np.abs(overlap[i]) < 0.05:
#             #     overlap[i]=0
#             # else:
#             #     print(overlap[i])
#         idx = np.argsort(abs(overlap))

#         occ_state = eigvecs[:,idx[-1]]
#         occ_state_energy = eigvals[idx[-1]]
#         # for i in range(len(overlap)):
#         #     # occ_state += eigvecs[:,i]*overlap[i]
#         #     occ_state_energy += eigvals[i] * overlap[i]**2
#         # # occ_state /=np.linalg.norm(occ_state)
        
#         return occ_state, occ_state_energy


# def Floquet_Winding_number_Peierls(H0, k, Nt, omega, T, E ,quasiE = None, previous_state = None, w = 0.2):
#     """
#     Build and diagonalize the Floquet Hamiltonian for a 1D system,
#     then group the 2*N_t eigenvalues/eigenstates into two Floquet bands.
#     choose the correct Floquet branch if E = 0 (by comparing with the directly diagonalized energies)
#     if E != 0, choose the correct branch by doing overlap with the previous state.

#     H(t) is SSH model with k replaced by k - E_0/w * sin(wt) by Peierls substitution
#     """
#     a = 1 #lattice constant, need to be modifyed accordingly, later need to be included into the variables
#     A = E /omega
#     if E == 0:
#         Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
#         NF = Norbs * Nt           # dimension of Floquet matrix
#         N0 = -(Nt-1)//2           # shift for Fourier indices

#         # Hn = [np.array([[0, 0], [0, 0]], dtype=complex) for a in range(Nt)]
#         # Hn[0] = H0 + np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex)

#         Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
#         for i in range(Nt):
#             Hn[i][0][1] *= jv(-i,A)
#             Hn[i][1][0] *= jv(i,A)
#         Hn[0] += H0

#         # Construct the Floquet matrix
#         # need to be modified.
#         F = np.zeros((NF, NF), dtype=complex)
#         for n in range(Nt):
#             for m in range(Nt):
#                 for k in range(Norbs):
#                     for l in range(Norbs):
#                         i = Norbs*n + k
#                         j = Norbs*m + l
#                         # Hamiltonian block + photon block
#                         F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
#                                    - (n + N0)*omega*(n==m)*(k==l))

#         # Diagonalize
#         eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)
#         # specify a range to choose the quasienergies, choose the first BZ
#         # [-hbar omega/2, hbar * omega/2]
#         eigvals_subset = np.zeros(Norbs, dtype=complex)
#         eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)
#         # check if the Floquet states is complete
#         j = 0
#         for i in range(NF):
#             if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
#                 eigvals_subset[j] = eigvals[i]
#                 eigvecs_subset[:,j] = eigvecs[:,i]
#                 j += 1
#         if j != Norbs:
#             print("Error: Number of Floquet states {} is not equal to \
#                 the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#             sys.exit()
#         eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
#         eigvals_copy = np.array(eigvals_copy)
#         idx = np.argsort(eigvals_copy.real)
#         occ_state = eigvecs[:, idx[0]]
#         occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
#         return occ_state, occ_state_energy
#     else:
#         Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
#         NF = Norbs * Nt           # dimension of Floquet matrix
#         N0 = -(Nt-1)//2           # shift for Fourier indices
#         Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
#         for i in range(Nt):
#             Hn[i][0][1] *= jv(-i,A)
#             Hn[i][1][0] *= jv(i,A)
#         Hn[0] += H0
#         # Construct the Floquet matrix
#         # need to be modified.
#         F = np.zeros((NF, NF), dtype=complex)
#         for n in range(Nt):
#             for m in range(Nt):
#                 for k in range(Norbs):
#                     for l in range(Norbs):
#                         i = Norbs*n + k
#                         j = Norbs*m + l
#                         # Hamiltonian block + photon block
#                         F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
#                                    - (n + N0)*omega*(n==m)*(k==l))

#         # Diagonalize
#         eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)s
#         # specify a range to choose the quasienergies, choose the first BZ
#         # [-hbar omega/2, hbar * omega/2]
#         eigvals_subset = np.zeros(Norbs, dtype=complex)
#         eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


#         # check if the Floquet states is complete
#         j = 0
#         for i in range(NF):
#             if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
#                 eigvals_subset[j] = eigvals[i]
#                 eigvecs_subset[:,j] = eigvecs[:,i]
#                 j += 1
#         if j != Norbs:
#             print("Error: Number of Floquet states {} is not equal to \
#                 the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#             sys.exit()
#         overlap = np.zeros(NF)
#         for i in range(NF):
#             for j in range(NF):
#                 overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
#                 # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
#             # if np.abs(overlap[i]) < 0.05:
#             #     overlap[i]=0
#             # else:
#             #     print(overlap[i])
#         idx = np.argsort(abs(overlap))

#         occ_state = eigvecs[:,idx[-1]]
#         occ_state_energy = eigvals[idx[-1]]
#         # for i in range(len(overlap)):
#         #     # occ_state += eigvecs[:,i]*overlap[i]
#         #     occ_state_energy += eigvals[i] * overlap[i]**2
#         # # occ_state /=np.linalg.norm(occ_state)
        
#         return occ_state, occ_state_energy
    
# # ==============================================================
# #  Circularly polarised Peierls helper  (δx,δy embedding)
# # ==============================================================



# def Floquet_Winding_number_Peierls_circular(
#         k, Nt, omega, T, E0,                        # drive & grid
#         delta_x, delta_y,                           # geometry
#         a=1.0, t0=1.0, xi=1.0,                      # lattice/decay
#         quasiE=None, previous_state=None):
#     """
#     Construct the extended Floquet matrix for a circularly polarised
#     vector potential  A(t)=A0[cosΩt, sinΩt]  and return the valence
#     Floquet state at momentum k.

#     All array sizes are identical to those used in the linear routine,
#     so the outer code stays unchanged.
#     """
#     Norbs = 2                   # SSH: 2 sites / unit cell
#     NF    = Norbs * Nt
#     N0    = -(Nt - 1) // 2      # Fourier index shift  (Nt must be odd)

#     # ---- static hopping magnitudes  v0, w0  --------------------
#     d_v   = (delta_x**2 + delta_y**2)**0.5
#     d_w   = np.sqrt((a-delta_x)**2 + delta_y**2)
#     v0    = t0 * exp(-d_v / xi)
#     w0    = t0 * exp(-d_w / xi)

#     # ---- bond angles & drive amplitude -------------------------
#     theta_v = arctan2(delta_y,            delta_x)
#     theta_w = arctan2(-delta_y,  a - delta_x)
#     z_v     = (E0 / omega) * d_v          #  α|d|
#     z_w     = (E0 / omega) * d_w

#     # # Fourier coeffs  t^{(m)}  for m = N0 … N0+Nt-1
#     # m_list  = np.arange(Nt) + N0
#     # coeff_v = v0 * (-1j)**m_list * jv(m_list, z_v) * np.exp(-1j*m_list*theta_v)
#     # coeff_w = w0 * (-1j)**m_list * jv(m_list, z_w) * np.exp(-1j*m_list*theta_w)
#     # ---------- Fourier coefficients t^{(m)} ---------------------
#     # need m = -(Nt-1) … +(Nt-1)  →  2*Nt-1 values
    
#     #check this part
#     m_all  = np.arange(-Nt+1, Nt)           # length 2*Nt-1
#     coeff_v = v0 * (-1j)**(-m_all) * jv(-m_all, z_v) * np.exp(1j*m_all*theta_v)
#     coeff_w = w0 * (-1j)**(-m_all) * jv(-m_all, z_w) * np.exp(1j*m_all*theta_w)


#     # ---- build Floquet matrix  F  --------------------------------
#     F = zeros((NF, NF), dtype=complex)
#     for n in range(Nt):
#         for m in range(Nt):
#             mm = n - m                          # harmonic index
#             # v_m = coeff_v[mm + (Nt-1)//2]
#             # w_m = coeff_w[mm + (Nt-1)//2] * exp(-1j * k)
#             v_m = coeff_v[mm + Nt - 1]
#             w_m = coeff_w[mm + Nt - 1] * exp(-1j * k * a)

#             block = zeros((2, 2), dtype=complex)
#             block[0, 1] = v_m + w_m
#             block[1, 0] = (v_m + w_m).conjugate()
#             if n == m:
#                 block += eye(2) * (n + N0) * omega
#             F[2*n:2*n+2, 2*m:2*m+2] = block

#     # ---- diagonalise & pick valence branch -----------------------
#     eigvals, eigvecs = eigh(F)
#     # zone = np.logical_and(eigvals.real <=  0.5*omega,
#     #                       eigvals.real >= -0.5*omega)
#     # eps, vec = eigvals[zone], eigvecs[:, zone]
#     eigvals_subset = np.zeros(Norbs, dtype=complex)
#     eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)
#     j=0
#     for i in range(NF):
#         if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
#             eigvals_subset[j] = eigvals[i]
#             eigvecs_subset[:,j] = eigvecs[:,i]
#             j += 1
#     if j != Norbs:
#         print("Error: Number of Floquet states {} is not equal to \
#             the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#         sys.exit()
#     if E0 == 0:
#         eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
#         eigvals_copy = np.array(eigvals_copy)
#         idx = np.argsort(eigvals_copy.real)
#         occ_state = eigvecs[:, idx[0]]
#         occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
#         return occ_state, occ_state_energy
#     else:
#         overlap = np.zeros(NF)
#         for i in range(NF):
#             for j in range(NF):
#                 overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
#                 # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
#             # if np.abs(overlap[i]) < 0.05:
#             #     overlap[i]=0
#             # else:
#             #     print(overlap[i])
#         idx = np.argsort(abs(overlap))

#         occ_state = eigvecs[:,idx[-1]]
#         occ_state_energy = eigvals[idx[-1]]
#         return occ_state, occ_state_energy
        
#     # if previous_state is None or quasiE is not None:
#     #     idx = np.argmin(np.abs(eps.real - (quasiE if quasiE is not None else 0.0)))
#     # else:
#     #     olap = eigvecs.conj().T @ previous_state
#     #     idx  = np.argmax(np.abs(olap))

#     # return eigvecs[:, idx], eigvecs[idx].real


# def Floquet_Winding_number_Peierls_GL2013(H0, k, Nt, E_over_omega ,quasiE = None, previous_state = None, b=0.5, t=1):
#     """
#     Build and diagonalize the Floquet Hamiltonian for a 1D system,
#     then group the 2*N_t eigenvalues/eigenstates into two Floquet bands.
#     choose the correct Floquet branch if E = 0 (by comparing with the directly diagonalized energies)
#     if E != 0, choose the correct branch by doing overlap with the previous state.

#     H(t) is SSH model with k replaced by k - E_0/w * sin(wt) by Peierls substitution
#     """
#     a = 1 #lattice constant, need to be modifyed accordingly, later need to be included into the variables
#     A = E_over_omega
#     omega = 100
#     if E_over_omega == 0:
#         Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
#         NF = Norbs * Nt           # dimension of Floquet matrix
#         N0 = -(Nt-1)//2           # shift for Fourier indices

#         Hn = [np.array([[0, 0], [0, 0]], dtype=complex) for a in range(Nt)]
#         # Hn[0] = H0 + np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex)

#         # Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
#         Hn_b = [np.array([[0, t], [t, 0]], dtype=complex) for a in range(Nt)]
#         Hn_a_b = [np.array([[0, np.exp(1j*k)], [np.exp(-1j*k), 0]], dtype=complex) for a in range(Nt)]
#         # Hn_b = [np.array([[0, t*np.exp(-1j*k*b)], [t*np.exp(1j*k*b), 0]], dtype=complex) for a in range(Nt)]
#         # Hn_a_b = [np.array([[0, np.exp(1j*k*(1-b))], [np.exp(-1j*k*(1-b)), 0]], dtype=complex) for a in range(Nt)]

#         for i in range(Nt):
#             Hn[i][0][1] = Hn_b[i][0][1] * jv(-i,A*b) + Hn_a_b[i][0][1] * jv(i,A*(1-b)) 
#             Hn[i][1][0] = Hn_b[i][1][0] * jv(i,A*b) + Hn_a_b[i][1][0] * jv(-i,A*(1-b))
#         Hn[0] = H0
#         # Construct the Floquet matrix
#         # need to be modified.
#         F = np.zeros((NF, NF), dtype=complex)
#         for n in range(Nt):
#             for m in range(Nt):
#                 for k in range(Norbs):
#                     for l in range(Norbs):
#                         i = Norbs*n + k
#                         j = Norbs*m + l
#                         # Hamiltonian block + photon block
#                         F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
#                                    - (n + N0)*omega*(n==m)*(k==l))

#         # Diagonalize
#         eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)
#         # specify a range to choose the quasienergies, choose the first BZ
#         # [-hbar omega/2, hbar * omega/2]
#         eigvals_subset = np.zeros(Norbs, dtype=complex)
#         eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)
#         # check if the Floquet states is complete
#         j = 0
#         for i in range(NF):
#             if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
#                 eigvals_subset[j] = eigvals[i]
#                 eigvecs_subset[:,j] = eigvecs[:,i]
#                 j += 1
#         if j != Norbs:
#             print("Error: Number of Floquet states {} is not equal to \
#                 the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#             sys.exit()
#         eigvals_copy = [np.abs(x - quasiE) for x in eigvals]
#         eigvals_copy = np.array(eigvals_copy)
#         idx = np.argsort(eigvals_copy.real)
#         occ_state = eigvecs[:, idx[0]]
#         occ_state_energy = eigvals[idx[0]]  # might needed for winding number calculation
#         return occ_state, occ_state_energy
#     else:
#         Norbs = H0.shape[-1]      # e.g., 2 for a two-level system
#         NF = Norbs * Nt           # dimension of Floquet matrix
#         N0 = -(Nt-1)//2           # shift for Fourier indices
#         Hn = [np.array([[0, 0], [0, 0]], dtype=complex) for a in range(Nt)]
#         # Hn[0] = H0 + np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex)

#         # Hn = [np.array([[0, w*np.exp(-1j*k)], [w*np.exp(1j*k), 0]], dtype=complex) for a in range(Nt)]
#         Hn_b = [np.array([[0, t], [t, 0]], dtype=complex) for a in range(Nt)]
#         Hn_a_b = [np.array([[0, np.exp(1j*k)], [np.exp(-1j*k), 0]], dtype=complex) for a in range(Nt)]
#         for i in range(Nt):
#             Hn[i][0][1] = Hn_b[i][0][1] * jv(-i,A*b) + Hn_a_b[i][0][1] * jv(i,A*(1-b)) 
#             Hn[i][1][0] = Hn_b[i][1][0] * jv(i,A*b) + Hn_a_b[i][1][0] * jv(-i,A*(1-b))
#         Hn[0] = H0
#         # Construct the Floquet matrix
#         # need to be modified.
#         F = np.zeros((NF, NF), dtype=complex)
#         for n in range(Nt):
#             for m in range(Nt):
#                 for k in range(Norbs):
#                     for l in range(Norbs):
#                         i = Norbs*n + k
#                         j = Norbs*m + l
#                         # Hamiltonian block + photon block
#                         F[i, j] = (HamiltonFT_peierls(Hn, n-m)[k, l]
#                                    - (n + N0)*omega*(n==m)*(k==l))

#         # Diagonalize
#         eigvals, eigvecs = linalg.eigh(F)  # shape(eigvals)=(NF,), shape(eigvecs)=(NF,NF)s
#         # specify a range to choose the quasienergies, choose the first BZ
#         # [-hbar omega/2, hbar * omega/2]
#         eigvals_subset = np.zeros(Norbs, dtype=complex)
#         eigvecs_subset = np.zeros((NF , Norbs), dtype=complex)


#         # check if the Floquet states is complete
#         j = 0
#         for i in range(NF):
#             if  eigvals[i] <= omega/2.0 and eigvals[i] >= -omega/2.0:
#                 eigvals_subset[j] = eigvals[i]
#                 eigvecs_subset[:,j] = eigvecs[:,i]
#                 j += 1
#         if j != Norbs:
#             print("Error: Number of Floquet states {} is not equal to \
#                 the number of orbitals {} in the first BZ. \n".format(j, Norbs))
#             sys.exit()
#         overlap = np.zeros(NF)
#         for i in range(NF):
#             for j in range(NF):
#                 overlap[i] += eigvecs[j,i] *np.conjugate(previous_state[j])
#                 # overlap[i] += np.conjugate(eigvecs[i,j]) * previous_state[j]
#             # if np.abs(overlap[i]) < 0.05:
#             #     overlap[i]=0
#             # else:
#             #     print(overlap[i])
#         idx = np.argsort(abs(overlap))

#         occ_state = eigvecs[:,idx[-1]]
#         occ_state_energy = eigvals[idx[-1]]
#         # for i in range(len(overlap)):
#         #     # occ_state += eigvecs[:,i]*overlap[i]
#         #     occ_state_energy += eigvals[i] * overlap[i]**2
#         # # occ_state /=np.linalg.norm(occ_state)
        
#         return occ_state, occ_state_energy

# # def test():
# #     """
# #     Test of FB winding number solver aganist GL2024, PRL,

# #     Returns
# #     -------
# #     None.

# #     """
    
# if __name__ == '__main__':
#     # from pyqed import pauli
#     # s0, sx, sy, sz = pauli()
    
#     tb = TightBinding(norbs=2, coords=[[]])
    
#     tb.band_gap()
#     tb.band_structure(ks)
    
    
#     floquet = tb.Floquet(omegad = 1, E = 1, polarization=[1, 0, 0])
    
#     floquet.gauge = 'length'
#     floquet.nt = 61

    
#     floquet.winding_number(n=0)
    
#     floquet.quasienergy()
#     floquet.floquet_modes()
    
    
#     # phase diagram 
#     W = np.zeros((10, 10))
#     for i, E in enumerate(np.linspace(0, 0.1, 10)):
#         for j, omegad in enumerate(np.linspace(0, 1, 10)):
#             floquet.set_amplitude(E)
#             floquet.set_driving_frequency(omegad)
            
#             W[i, j] = floquet.winding_number()
    
#     # plot W
    
#     # dynamics 
    
    
#     # mol = Floquet(H=0.5*sz, edip=sx)    
#     # qe, fmodes = mol.spectrum(0.4, omegad = 1, nt=10, gauge='length')


#     # qe, fmodes = dmol.spectrum(E0=0.4, Nt=10, gauge='velocity')
#     # print(qe)

if __name__ == "__main__":
    test_Gomez_Leon_2013()
#     # 1D two-orbital example: orbitals at x=0 and x=0.5
#     coords        = [[0.0], [0.4]]
#     lambda_decay = 0.5
#     a             = 1.0
#     mu            = 0

#     # instantiate and build hopping matrices
#     tb = TightBinding(
#         coords,
#         lambda_decay=lambda_decay,
#         lattice_constant=a,
#         nk=10,
#         mu=mu
#     )

#     # sample H(k) at k=0 and k=π
#     for k in [0.0, np.pi]:
#         Hk = tb.buildH(k)
#         print(f"H(k = {k:.3f}):\n{Hk}\n")
#         eigs = np.linalg.eigvalsh(Hk)
#         print(f"eigenvalues at k={k:.3f}: {eigs}\n{'-'*40}\n")
#     tb.run(k=np.linspace(-np.pi, np.pi, 100))
#     tb.plot(k=np.linspace(-np.pi, np.pi, 100))
#     print("Band gap =", tb.band_gap())

#     floquet = tb.Floquet(omegad=5.0, E0=np.linspace(0, 1,101), nt=21, gauge='Peierls', polarization=[1], data_path='MacBook_local_data/floquet_data')
#     floquet.run(k=np.linspace(-np.pi, np.pi, 100))
#     floquet.plot_band_structure(k=np.linspace(-np.pi, np.pi, 100),save_band_structure=True)
#     a = floquet.winding_number(band=0)
#     print(a)