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
from pyqed.floquet.utils import track_valence_band, berry_phase_winding, figure, track_valence_band_GL2013
from numpy import exp, eye, zeros, arctan2
from scipy.linalg import eigh

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
    def __init__(self, coords, lambda_decay=1.0, a=1.0, nk=50, mu=0.0):
        # Orbital positions and dimensionality
        self.coords = np.array(coords, dtype=float)
        self.norb = len(self.coords)
        self.dim = len(self.coords[0])
        print("dim", self.dim)
        # Lattice vectors
        if np.shape(a) == ():
            self.a_vec = np.ones(self.dim) * float(a)
        else:
            self.a_vec = np.array(a, dtype=float)
            if self.a_vec.size != self.dim:
                raise ValueError("Length of `a` must equal spatial dimension")

        self.mu = mu
        self.lambda_decay = float(lambda_decay)

        # Build intracell and intercell hopping matrices
        self.intra = np.zeros((self.norb, self.norb), dtype=complex)
        self.inter_upper = np.zeros_like(self.intra)
        self.inter_lower = np.zeros_like(self.intra)
        for i in range(self.norb):
            for j in range(self.norb):
                # Intracell hopping
                delta = self.coords[j] - self.coords[i]
                dist = np.linalg.norm(delta)
                if i > j:
                    self.intra[i, j] = np.exp(-dist / self.lambda_decay)
                    self.intra[j, i] = np.conj(self.intra[i, j])
                # Intercell hopping (to next unit cell)
                delta_p = (self.coords[j] + self.a_vec) - self.coords[i]
                dist_p = np.linalg.norm(delta_p)
                if i > j:
                    self.inter_lower[i, j] = np.exp(dist_p / self.lambda_decay)
                    self.inter_upper[j, i] = np.conj(self.inter_lower[i, j])

        # Build k-point grid for Brillouin zone
        if isinstance(nk, int):
            nk_list = [nk] * self.dim
        else:
            nk_list = list(nk)
            if len(nk_list) != self.dim:
                raise ValueError("`nk` must be int or list of length dim")

        axes = [np.linspace(-np.pi/self.a_vec[d], np.pi/self.a_vec[d], nk_list[d])
                for d in range(self.dim)]
        grids = np.meshgrid(*axes, indexing='ij')
        pts = np.stack([g.flatten() for g in grids], axis=-1)
        self.k_vals = pts  # shape: (prod(nk_list), dim)

        # Placeholder for computed band energies
        self._bands = None


    def buildH(self, k):
        """
        Construct Bloch Hamiltonian H(k) = intra + inter_upper*e^{i k·a} + inter_lower*e^{-i k·a} + mu*I.

        Parameters
        ----------
        k : array_like, length = dim
            Crystal momentum vector.
        """
        #check if k is a list of k points, if self.dim == 1, then k is a list of floats
        if self.dim == 1:
            if isinstance(k, (float, int)):
                k_vec = np.array([k])
        else:
            if isinstance(k, (list, np.ndarray)):
                if len(k[0]) != self.dim:
                    raise ValueError(f"each k point must have length {self.dim}")

        Hk = (self.intra
              + self.inter_upper * np.exp(1j * np.dot(k, self.a_vec))
              + self.inter_lower * np.exp(-1j * np.dot(k, self.a_vec))
              + np.eye(self.norb) * self.mu)

        return Hk

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

    def Floquet(self, **kwargs):
        """
        Return a FloquetBloch instance with coordinate info.
        """
        Hk_func = lambda kpt: self.buildH(kpt)
        # pos = np.diag(np.arange(self.norb) * self.a_vec[0])
        floq = FloquetBloch(Hk_func=Hk_func, **kwargs, coords=self.coords, a_vec=self.a_vec,
                            norbs=self.norb)
        # floq = FloquetBloch(Hk=Hk_func, Edip=pos, **kwargs)
        # Attach coordinate info for extended H build
        return floq


    def density_of_states(self):
        pass

    def zeeman(self):
        # drive with magnetic field
        pass

    def gf(self):
        # surface and bulk Green function
        pass

    def gf_surface(self):
        pass


    def LvN(self, *args, **kwargs):
        # Liouville von Newnman equation solver
        pass
        # return LvN(*args, **kwargs)


# class FloquetBloch:
#     """
#     Periodically driven tight-binding system in Bloch basis.
#     Performs Floquet analysis via Jacobi-Anger expansion and tracks specified bands.

#     Parameters
#     ----------
#     Hk_func : callable
#         Function Hk_func(k) returning the static Bloch Hamiltonian at momentum k.
#     omegad : float
#         Driving angular frequency.
#     E0 : float
#         Electric field amplitude.
#     nt : int
#         Number of Floquet harmonics (Fourier components).
#     coords : array_like, shape (norbs, dim)
#         Orbital coordinates within unit cell (for dipole projection).
#     a_vec : array_like, shape (dim,)
#         Lattice vector(s) for phase factors.
#     norbs : int
#         Number of orbitals per cell.
#     gauge : {'Peierls', 'length', 'velocity'}
#         Gauge choice.
#     polarization : array_like, shape (dim,)
#         Jones vector specifying field polarization.
#     data_path : str or None
#         Directory to cache computed band data (.npz). If None, no caching.
#     initial_band : int
#         Band index at E=0 to track if calculated_bands not given.
#     """
#     def __init__(self, Hk_func, omegad, E0, nt,
#                  coords, a_vec, norbs,
#                  gauge='Peierls', polarization=None,
#                  data_path=None, initial_band=0):
#         self.Hk_func = Hk_func
#         self.omegad = omegad
#         self.E0 = E0
#         self.nt = nt
#         self.coords = np.array(coords)
#         self.a_vec = np.array(a_vec)
#         self.norbs = norbs
#         self.gauge = gauge
#         if polarization is None:
#             self.polarization = np.zeros(self.coords.shape[1])
#         else:
#             self.polarization = np.array(polarization)
#         if self.polarization.size != self.coords.shape[1]:
#             raise ValueError("polarization must match coordinate dimension")
#         self.data_path = data_path
#         self.initial_band = initial_band
#         # storage for last results
#         self.k_vals = None
#         self._tracked_bands = None
#         self._energies = None
#         self._states = None
#         self.dim = len(coords[0])

#     def build_extendedH(self, kpt):
#         """Build the Floquet extended Hamiltonian via Jacobi-Anger expansion."""
#         if isinstance(self.E0, (float, int)):
#             omegad, E0 = self.omegad, self.E0
#         if isinstance(self.E0, (list, np.ndarray)):
#             omegad, E0 = self.omegad, self.E0[-1]
#             print('Note: Here in .build_extendedH, Only the extendedH for largesr H built and shown, but list input type is allowed in .run()')
#         Norbs, nt = self.norbs, self.nt
#         coords, eps = self.coords, self.polarization
#         H0 = self.Hk_func(kpt)
#         dmat = np.dot(coords[:, None, :] - coords[None, :, :], eps)
#         Hn = {}
#         for p in range(-nt//2, nt//2 + 1):
#             Hn[p] = H0 * jv(p, E0/omegad * dmat)
#         NF = Norbs * nt
#         F = np.zeros((NF, NF), dtype=complex)
#         N0 = -(nt - 1) // 2
#         for n in range(nt):
#             for m in range(nt):
#                 p = n - m
#                 block = Hn.get(p, np.zeros((Norbs, Norbs)))
#                 if n == m:
#                     block = block + np.eye(Norbs) * ((n + N0) * omegad)
#                 i0, i1 = n*Norbs, (n+1)*Norbs
#                 j0, j1 = m*Norbs, (m+1)*Norbs
#                 F[i0:i1, j0:j1] = block
#         return F

#     def run(self, k, nE_steps=10, calculated_bands=None):
#         """
#         Diagonalize for each k, tracking specified Floquet bands.

#         Parameters
#         ----------
#         k : array_like
#             k-points of shape (M, dim).
#         nE_steps : int
#             Number of field ramp steps.
#         calculated_bands : list of int, optional
#             Bands to track; defaults to [initial_band].

#         Returns
#         -------
#         ks : ndarray, shape (M, dim)
#         energies : ndarray, shape (len(calculated_bands), M)
#         states : ndarray, shape (len(calculated_bands), norbs*nt, M)
#         """
#         # reshape for 1D
#         if self.dim == 1:
#             if isinstance(k, (float, int)):
#                 k = np.array([k])
#             if isinstance(k[0], (float, int)):
#                 k = np.array(k).reshape(-1, 1)
#         ks = np.atleast_2d(k)
#         M = ks.shape[0]
#         Norbs, nt = self.norbs, self.nt
#         NF = Norbs * nt

#         # choose bands
#         bands_to_track = list(calculated_bands) if calculated_bands is not None else [self.initial_band]
#         self._tracked_bands = bands_to_track
#         nb = len(bands_to_track)

#         # caching
#         cache_file = None
#         if self.data_path:
#             os.makedirs(self.data_path, exist_ok=True)
#             pol = '_'.join(map(str, self.polarization))
#             cache_file = os.path.join(self.data_path,
#                 f"floquet_E{self.E0:.6f}_w{self.omegad:.6f}_nt{nt}_pol{pol}.npz")
#             if os.path.exists(cache_file):
#                 data = np.load(cache_file)
#                 energies = data['energies']
#                 states = data['states']
#                 self.k_vals = ks
#                 self._energies = energies
#                 self._states = states
#                 for j, b in enumerate(bands_to_track):
#                     setattr(self, f"_bands_{b}", energies[j])
#                     setattr(self, f"_states_{b}", states[j])
#                 return ks, energies, states

#         # outputs
#         energies = np.zeros((nb, M), dtype=float)
#         states = np.zeros((nb, NF, M), dtype=complex)
#         prev = np.zeros((NF, M, nb), dtype=complex)
#         E_list = np.linspace(0, self.E0, nE_steps) if self.E0 != 0 else [0.]

#         for ie, Ecur in enumerate(E_list):
#             self.E0 = Ecur
#             for ik, kpt in enumerate(ks):
#                 Fk = self.build_extendedH(kpt)
#                 eigvals, eigvecs = np.linalg.eigh(Fk)
#                 if ie == 0:
#                     static = np.linalg.eigvalsh(self.Hk_func(kpt))
#                     for j, b in enumerate(bands_to_track):
#                         target = static[b]
#                         idx = np.argmin(np.abs(eigvals - target))
#                         prev[:, ik, j] = eigvecs[:, idx]
#                         if len(E_list) == 1:
#                             energies[j, ik] = eigvals[idx]
#                             states[j, :, ik] = eigvecs[:, idx]
#                 else:
#                     for j in range(nb):
#                         ov = np.abs(prev[:, ik, j].conj() @ eigvecs)
#                         idx = np.argmax(ov)
#                         prev[:, ik, j] = eigvecs[:, idx]
#                         if ie == len(E_list) - 1:
#                             energies[j, ik] = eigvals[idx]
#                             states[j, :, ik] = eigvecs[:, idx]

#         # store
#         self.k_vals = ks
#         self._energies = energies
#         self._states = states
#         for j, b in enumerate(bands_to_track):
#             setattr(self, f"_bands_{b}", energies[j])
#             setattr(self, f"_states_{b}", states[j])
#         if cache_file:
#             np.savez(cache_file, energies=energies, states=states)
#         return ks, energies, states

#     def plot_band_structure(self, k=None):
#         """Plot tracked band energies vs k in 1D."""
#         if self.dim != 1:
#             raise NotImplementedError("Plotting only supported in 1D.")
#         if self._energies is None:
#             ks, energies, _ = self.run(k)
#         else:
#             ks = self.k_vals
#             energies = self._energies
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(8, 4))
#         for j, b in enumerate(self._tracked_bands):
#             plt.plot(ks, energies[j], label=f'Band {b}')
#         plt.xlabel('k')
#         plt.ylabel('Energy')
#         plt.title('Floquet-Bloch Band Structure')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     def winding_number(self, band=None):
#         """
#         Compute winding number via Berry phase of a tracked Floquet state.

#         Parameters
#         ----------
#         band : int, optional
#             Band index to compute for; if None and only one tracked, uses that.
#         """
#         if self._states is None or self.k_vals is None:
#             raise RuntimeError("Run first before computing winding.")
#         if band is None:
#             if len(self._tracked_bands) == 1:
#                 band = self._tracked_bands[0]
#             else:
#                 raise ValueError("Must specify band if multiple tracked.")
#         if band not in self._tracked_bands:
#             raise ValueError(f"Band {band} not tracked (tracked: {self._tracked_bands})")
#         j = self._tracked_bands.index(band)
#         vecs = self._states[j]  # shape (NF, M)
#         N = vecs.shape[1]
#         P = np.outer(vecs[:, 0], vecs[:, 0].conj())
#         for i in range(1, N):
#             psi = vecs[:, i] / np.linalg.norm(vecs[:, i])
#             P = P @ np.outer(psi, psi.conj())
#         angle = np.angle(np.trace(P))
#         return (angle % (2*np.pi)) / np.pi

import numpy as np
import os
import h5py
from scipy.special import jv

class FloquetBloch:
    """
    Periodically driven tight-binding system in Bloch basis with caching.

    Parameters
    ----------
    Hk_func : callable
        Returns static Bloch Hamiltonian H(k).
    omegad : float
        Driving frequency.
    E0 : float or list/ndarray
        Field amplitudes; if list/array, must start at zero.
    nt : int
        Number of Floquet harmonics.
    coords : array_like, shape (norbs, dim)
        Orbital positions.
    a_vec : array_like, shape (dim,)
        Lattice vectors.
    norbs : int
        Number of orbitals per cell.
    data_path : str or None
        Directory for HDF5 caching and plots.
    initial_band : int
        Default static band to track.
    """
    def __init__(self, Hk_func, omegad, E0, nt,
                 coords, a_vec, norbs,
                 gauge='Peierls', polarization=None,
                 data_path=None, initial_band=0):
        self.Hk_func = Hk_func
        self.omegad = omegad
        if isinstance(E0, (list, np.ndarray)):
            self.E0 = E0[-1]
            self._E_list = None             # array of E values
        if isinstance(E0, (float, int)):
            self.E0 = E0
            self._E_list = None
        self.nt = nt
        self.coords = np.array(coords)
        self.a_vec = np.array(a_vec)
        self.norbs = norbs
        self.gauge = gauge
        self.polarization = (np.zeros(self.coords.shape[1]) if polarization is None
                             else np.array(polarization))
        if self.polarization.size != self.coords.shape[1]:
            raise ValueError("polarization must match coordinate dimension")
        self.data_path = data_path
        self.initial_band = initial_band

        # storage
        self.k_vals = None
        self._tracked_bands = None      # list of bands
        self._energies_all = None       # shape (nb, nE, M)
        self._states_all = None         # shape (nb, nE, NF, M)
        self.dim = len(coords[0])

    def build_extendedH(self, kpt, Ecur=None):
        if Ecur is None:
            Ecur = self.E0
            print('Note: Here only the extendedH for largesr E built and shown if your input is a list')
        Norbs, nt = self.norbs, self.nt
        omegad = self.omegad
        H0 = self.Hk_func(kpt)
        dmat = np.dot(self.coords[:,None,:] - self.coords[None,:,:],
                      self.polarization)
        Hn = {p: H0 * jv(p, Ecur/omegad * dmat)
              for p in range(-nt//2, nt//2+1)}
        NF = Norbs*nt
        F = np.zeros((NF, NF), dtype=complex)
        N0 = -(nt-1)//2
        for n in range(nt):
            for m in range(nt):
                p = n - m
                block = Hn.get(p, np.zeros((Norbs, Norbs)))
                if n == m:
                    block += np.eye(Norbs) * ((n+N0)*omegad)
                i0, i1 = n*Norbs, (n+1)*Norbs
                j0, j1 = m*Norbs, (m+1)*Norbs
                F[i0:i1, j0:j1] = block
        return F

    def run(self, k, nE_steps=10, calculated_bands=None):
        """
        Compute and cache Floquet bands per-field.

        k : array_like, shape (M, dim)
        nE_steps : int, ignored if E0 is list/array
        calculated_bands : list[int] or None
        """
        # reshape for 1D
        if self.dim == 1:
            arr = np.atleast_1d(k)
            if np.isscalar(arr[0]):
                k = arr.reshape(-1,1)
        ks = np.atleast_2d(k)
        M = ks.shape[0]
        Norbs, nt = self.norbs, self.nt
        NF = Norbs*nt

        # choose bands
        bands = list(calculated_bands) if calculated_bands is not None else [self.initial_band]
        nb = len(bands)
        self._tracked_bands = bands

        # setup E_list
        if isinstance(self.E0, (list, np.ndarray)):
            E_list = np.array(self.E0, float)
            E_list.sort()
            if E_list[0] != 0:
                raise ValueError("If E0 is list, must start at 0.")
        else:
            E_list = np.linspace(0, self.E0, nE_steps) if self.E0 != 0 else np.array([0.])
        self._E_list = E_list

        # prepare results
        energies = np.zeros((nb, len(E_list), M), dtype=float)
        states = np.zeros((nb, len(E_list), NF, M), dtype=complex)

        # caching per E
        if self.data_path:
            os.makedirs(self.data_path, exist_ok=True)
            pol = '_'.join(map(str, self.polarization))
            bands_str = '_'.join(map(str, bands))
        else:
            pol = bands_str = None

        for ie, Ecur in enumerate(E_list):
            # per-field cache filename
            if self.data_path:
                fname = f"floquet_E{Ecur:.6f}_w{self.omegad:.6f}_nt{nt}_pol{pol}_bands{bands_str}.h5"
                cache = os.path.join(self.data_path, fname)
            else:
                cache = None

            if cache and os.path.isfile(cache):
                with h5py.File(cache, 'r') as f:
                    # load
                    if self.k_vals is None:
                        self.k_vals = f['k_vals'][:]
                    energies[:, ie, :] = f['energies'][:]
                    states[:, ie, :, :] = f['states'][:]
                continue

            # compute this Ecur
            prev = np.zeros((NF, M, nb), dtype=complex)
            for ik, kpt in enumerate(ks):
                Fk = self.build_extendedH(kpt, Ecur)
                eigvals, eigvecs = np.linalg.eigh(Fk)
                if ie == 0:
                    static = np.linalg.eigvalsh(self.Hk_func(kpt))
                    for j,b in enumerate(bands):
                        tgt = static[b]
                        idx = np.argmin(np.abs(eigvals - tgt))
                        prev[:,ik,j] = eigvecs[:,idx]
                        energies[j,ie,ik] = eigvals[idx]
                        states[j,ie,:,ik] = eigvecs[:,idx]
                else:
                    for j in range(nb):
                        ov = np.abs(prev[:,ik,j].conj() @ eigvecs)
                        idx = np.argmax(ov)
                        prev[:,ik,j] = eigvecs[:,idx]
                        energies[j,ie,ik] = eigvals[idx]
                        states[j,ie,:,ik] = eigvecs[:,idx]

            # save cache
            if cache:
                with h5py.File(cache, 'w') as f:
                    f.create_dataset('k_vals', data=ks)
                    f.create_dataset('energies', data=energies[:,ie,:])
                    f.create_dataset('states', data=states[:,ie,:,:])

        # store all
        self.k_vals = ks
        self._energies_all = energies
        self._states_all = states

        return ks, energies, states

    def plot_band_structure(self, k=None, E=None,
                            save_band_structure=False, outdir=None):
        """
        Plot (or save) Floquet bands at specified E or list of E.

        E : float or list
            Field value(s) to plot; default last.
        save_band_structure : bool
            If True, save plots instead of showing.
        outdir : str or None
            Directory under data_path to save plots.
        """
        import matplotlib.pyplot as plt
        if self._energies_all is None:
            self.run(k)
        ks = self.k_vals
        E_list = self._E_list
        # handle E list
        E_vals = E_list if E is None else np.atleast_1d(E)
        if len(E_vals) > 1:
            if not save_band_structure:
                print("Warning: multiple E provided, forcing save_band_structure=True")
                save_band_structure = True
        # prepare save dir
        if save_band_structure:
            base = (outdir if outdir else self.data_path)
            plotdir = os.path.join(base, 'Floquet_Band_Structure')
            os.makedirs(plotdir, exist_ok=True)
        # loop
        for Ecur in E_vals:
            # find index
            idx = np.argmin(np.abs(E_list - Ecur))
            data = self._energies_all[:, idx, :]
            plt.figure(figsize=(8,4))
            for j,b in enumerate(self._tracked_bands):
                plt.plot(ks, data[j], label=f'Band {b}')
            plt.xlabel('k'); plt.ylabel('Energy')
            plt.title(f'Floquet Bands @ E={Ecur:.3f}, ω={self.omegad:.3f}')
            plt.legend(); plt.grid(True)
            if save_band_structure:
                fname = f"band_E{Ecur:.6f}_w{self.omegad:.6f}.png"
                plt.savefig(os.path.join(plotdir, fname))
                plt.close()
            else:
                plt.show()

    def winding_number(self, band=None, E=None):
        """
        Compute winding number for given band and field E or list of E.

        Returns float or list of floats.
        """
        if self._states_all is None or self.k_vals is None:
            raise RuntimeError("Call run() before winding_number().")
        bands = self._tracked_bands
        if band is None:
            if len(bands)==1:
                band = bands[0]
            else:
                raise ValueError("Specify band when tracking multiple.")
        if band not in bands:
            raise ValueError(f"Band {band} not tracked.")
        j = bands.index(band)
        E_list = self._E_list
        E_vals = E_list if E is None else np.atleast_1d(E)
        results = []
        for Ecur in E_vals:
            ie = np.argmin(np.abs(E_list - Ecur))
            vecs = self._states_all[j, ie]  # (NF, M)
            M = vecs.shape[1]
            P = np.outer(vecs[:,0], vecs[:,0].conj())
            for i in range(1, M):
                psi = vecs[:,i]/np.linalg.norm(vecs[:,i])
                P = P @ np.outer(psi, psi.conj())
            angle = np.angle(np.trace(P))
            results.append((angle % (2*np.pi))/np.pi)
        return results if len(results)>1 else results[0]

def test_Gomez_Leon_2013():
    """
    Test the Gomez-Leon 2013 model but using TightBinding and FloquetBloch classes, calculate the winding number and finally plot the heatmap
    """
    # Parameters

    omega = 1.0
    nt = 61
    k_vals = np.linspace(-np.pi, np.pi, 100)
    b_grid = np.linspace(0.1, 1.0, 10)
    for b in b_grid:
        # Create tight-binding model
        coords = [[0], [b]]
        tb_model = TightBinding(coords, lambda_decay=1.0, a=1.0, nk=50, mu=0.0)
        
        # Run Floquet analysis
        floquet_model = tb_model.Floquet(omegad=omega, E0=E0, nt=nt)
        ks, energies, states = floquet_model.run(k_vals)
        
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