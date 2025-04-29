# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:38:48 2022

Tight-binding models

@author: Bing Gu

"""
import numpy as np
from numpy.linalg import inv
from pyqed import dagger, dag
import scipy
from pyqed.floquet.Floquet import FloquetBloch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from pyqed.mol import Mol

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


# if __name__=="__main__":
#     tb = TightBinding()
#     print(tb.BZ)

# Example usage
if __name__ == '__main__':
    # Define 2-orbital unit cell at coords [0], [0.5]
    coords = [[0.0], [0.5]]
    tb = TightBinding(coords, lambda_decay=1.0, a=1.0, nk=100, mu=0.0)
    print('ongoing')
    ks, bands = tb.run(k=np.linspace(-np.pi, np.pi, 100))
    print("Band gap =", tb.band_gap())
    tb.plot()

    # Floquet example
    floq = tb.Floquet(omegad=5.0, E0=0.2, nt=21, gauge='Peierls', polarization=[1])
    print("Floquet hamiltonian", floq.build_extendedH(0))
    floq.run(k=np.linspace(-np.pi, np.pi, 100))
    floq.plot_band_structure(k=np.linspace(-np.pi, np.pi, 100))
    print("Floquet winding #", floq.winding_number(band = 0))


# To improve:
# 1. mu now only take in float, but can be a matrix
# 2. add more options for hopping, e.g. nearest neighbor, next nearest neighbor
# 3. not dealing with the case user input different k grid when calling the function for the class