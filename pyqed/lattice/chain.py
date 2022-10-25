# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:38:48 2022

Tight-binding models

From pyqula.

@author: Bing Gu

"""
import numpy as np
from numpy.linalg import inv
from lime.phys import dagger, dag
import scipy
from scipy.sparse import csr_matrix

from pyqed.Floquet import Floquet
from pyqed import Mol

class Chain(Mol):
    """
    open 1D tight-binding chain
    """
    def __init__(self, nsite, onsite, hopping, norb=1, \
                 boundary_condition='open', disorder=False):
        """

        Parameters
        ----------
        nsite : TYPE
            DESCRIPTION.
        onsite : TYPE
            DESCRIPTION.
        hopping : TYPE
            DESCRIPTION.
        norb : int, optional
            number of orbitals/wannier functions per unit cell.
            The default is 1.
        boundary_condition: str
            'periodic' or 'open'

        Returns
        -------
        None.

        """
        self.nsite = nsite
        self.norb = norb
        self.size = nsite * norb
        self.onsite = onsite
        self.hopping = hopping

        self.H = None
        self.boundary_condtion = boundary_condition

    def position(self):
        """
        position matrix elements in localized basis/Wannier basis

        .. math::
            x = \sum_{n=1}^N \sum_A n (\ket{m A}\bra{m A})


        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        norb = self.norb

        if self.norb == 1:
            return np.diag(range(1, self.nsite))
        else:
            r = np.zeros((self.size, self.size))
            for n in range(self.nsite):
                for i in range(self.norb):
                    r[n * norb + i, n * norb + i] = n+1

    def buildH(self):
        """

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        """
        nsite = self.nsite
        norb = self.norb

        H = csr_matrix(nsite * norb)

        onsite = self.onsite
        hopping = self.hopping

        if self.norb == 1:

            H.setdiag(self.onsite)
            H.setdiag(self.hopping, 1)
            H.setdiag(self.hopping, -1)

        else:
            assert(hopping.shape == (norb, norb))
            assert(len(onsite) == norb)

            for i in range(nsite):
                for j in range(norb):
                    H[norb*i + j, norb*i + j] = self.onsite[j]

            # nearest hopping
            # TODO: add long-range hopping
            for n in range(nsite-1):
                for j in range(norb):
                    for k in range(norb):
                        H[norb*n, j, norb*(n+1) + k] = hopping[j, k]

        self.H = H
        return H

    def floquet(self):
        # drive_with_efield

        return Floquet(self.H, -self.position())

    def zeeman(self):
        # drive with magnetic field
        pass




def draw_points(points, colors):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    fig, ax = plt.subplots()
    ax.scatter(x, y, color=colors)
    # ax.format(xlim=(-1, 2))
    return

class Lattice:
    def __init__(self, size=[2, 2], norb=1, lattice_vectors=None, \
                 orb_coords=None, nspin=1):
        """
        2D lattice model

        Parameters
        ----------
        size : TYPE, optional
            DESCRIPTION. The default is [2, 2].
        norb : TYPE, optional
            DESCRIPTION. The default is 1.
        lattice_vectors : TYPE, optional
            DESCRIPTION. The default is None.
        orb_coords : TYPE, optional
            DESCRIPTION. The default is None.
        nspin : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        self.nx, self.ny = size
        self.norb = norb # number of orbitals in a unit cell
        self._dim = self.norb * np.prod(size)
        self.nspin = nspin
        if orb_coords is None:
            orb_coords = [[0.5, 0.5], ] * norb
        self.orb_coords = orb_coords

        if lattice_vectors is None:
            lattice_vectors = [np.array([1., 0]), np.array([0., 1])]
        self.lattice_vectors = lattice_vectors
        self.a1 = lattice_vectors[0]
        self.a2 = lattice_vectors[1]


        self.H = np.zeros((self._dim, self._dim))

    def index(self, i, j, n):
        """
        return the index of the n-th orbital at cell (i, j) in the basis set.

        The basis set is ordered as :math:`\ket{\phi_{ijn}}` where i, j are the
        cell indexes and :math:`n` is the orbital index.

        (0, 0, :) (0, 1, :) (0, 2, :) ... (i, j, :)

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.
        j : TYPE
            DESCRIPTION.
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return (i * self.ny + j) * self.norb + n

    def set_hop(self, J, i, j, R, boundary_condition='open'):
        """
        set hopping parameter J between i-orbital  and j-orbital separated by
        :math:`R = (R_0, R_1)` unit cells


        Parameters
        ----------
        J : TYPE
            DESCRIPTION.
        i : TYPE
            DESCRIPTION.
        j : TYPE
            DESCRIPTION.
        R : list of two integers
            distance between hopping

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if boundary_condition == 'periodic':
            raise NotImplementedError('Periodic boundary condition is not implemented.\
                                      Use open instead.')

        for n in range(self.nx-R[0]):
            for m in range(self.ny-R[1]):
                ind_i = self.index(n, m, i)
                ind_j = self.index(n + R[0], m+R[1], j)
                self.H[ind_i, ind_j] = J
                self.H[ind_j, ind_i] = np.conj(J)

        return self.H

    def set_onsite(self):
        pass

    def solve(self):
        self.evals, self.evecs = scipy.linalg.eigh(self.H)
        return self.evals, self.evecs

    def draw(self, psi):
        from lime.style import tocolor

        points = []
        colors = []
        for i in range(self.nx):
            for j in range(self.ny):
                for n in range(self.norb):
                    p = self.orb_coords[n] + i * self.a1 + j * self.a2
                    points.append(p)
                    prob = np.abs(psi)**2
                    color = tocolor(np.abs(psi[self.index(i,j,n)])**2, vmin=min(prob), \
                                    vmax=max(prob))
                    print(color)
                    colors.append(color)



        draw_points(points, colors)
        return

class RiceMele:
    def __init__(self, v, w, nsite=None):
        self.intra = v
        self.inter = w
        self.H = None
        self.nsite = nsite
        self.evecs = None
        self.evals = None

    def band_structure(self):
        pass

    def buildH(self):

        H = np.zeros((self.nsite, self.nsite))
        for i in range(0, self.nsite-1, 2):
            H[i, i+1] = H[i+1, i] = self.intra

        for i in range(1, self.nsite-1, 2):
            H[i, i+1] = H[i+1, i] = self.inter

        self.H = H
        return H

    def solve(self):
        self.evals, self.evecs = scipy.linalg.eigh(self.H)
        return self.evals, self.evecs

    def plot_state(self, index=0):
        if self.evecs is None:
            self.solve()

        if not isinstance(index, list):
            index = [index]

        fig, axs = plt.subplots(nrows=len(index), figsize=(4,6))
        for i, n in enumerate(index):
            axs[i].bar(self.evecs[:,n]**2)
        return

    def ldos(self, omega, eta=1e-4):

        if self.H is None:
            # raise ValueError('H is none. Call chain() to build the H first.')
            self.buildH()

        g = scipy.linalg.inv((omega - 1j*eta) * np.identity(self.nsite) - self.H)
        return g[0, 0].imag

    def gf(self, omega, eta=1e-4, method='sos'):
        """
        Calculate the retarded Green's function in the frequency domain
        ..math::
            (\omega - i \eta - \mathbf{H}) \mathbf{G} = \mathbf{I}

        Parameters
        ----------
        omega : float
            DESCRIPTION.
        eta : float, optional
            infinitesimal number to ensure causality. The default is 1e-4.
        method : str, optional
            'sos': Sum-over-states expansion
            ..math::
                G = \sum_j \frac{\ket{\psi_j}\bra{\psi_j}}{\omega - i\eta - \omega_j}
            where j runs over all eigenstates.

            'diag':
                direction diagonalization of the Hamiltonian
                ..math::
                    G = \del{\omega - i \eta - \mathbf{H}}^{-1}

            The default is 'sos'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if self.H is None:
            # raise ValueError('H is none. Call chain() to build the H first.')
            self.buildH()

        if method == 'diag':
            g = scipy.linalg.inv((omega - 1j*eta) * np.identity(self.nsite) - self.H)

        elif method == 'sos':
            if self.evals is None:
                self.solve()
            if isinstance(omega, (float, complex)):
                U = self.evecs
                g = U @ np.diag(1./(omega - 1j*eta - self.evals)) @ dag(U)



        return g[0, 0].imag

def green_renormalization(intra,inter,energy=0.0,nite=None,
                            info=False,delta=0.001,**kwargs):
    """ Calculates bulk and surface Green function by a renormalization
    algorithm, as described in I. Phys. F: Met. Phys. 15 (1985) 851-858 """
    # intra = algebra.todense(intra)
    # inter = algebra.todense(inter)
    error = np.abs(delta)*1e-6 # overwrite error

    e = np.matrix(np.identity(intra.shape[0])) * (energy + 1j*delta)
    ite = 0
    alpha = inter.copy()
    beta = dagger(inter).copy()
    epsilon = intra.copy()
    epsilon_s = intra.copy()
    while True: # implementation of Eq 11
      einv = inv(e - epsilon) # inverse
      epsilon_s = epsilon_s + alpha @ einv @ beta
      epsilon = epsilon + alpha @ einv @ beta + beta @ einv @ alpha
      alpha = alpha @ einv @ alpha  # new alpha
      beta = beta @ einv @ beta  # new beta
      ite += 1
      # stop conditions
      if not nite is None:
        if ite > nite:  break
      else:
        if np.max(np.abs(alpha))<error and np.max(np.abs(beta))<error: break
    if info:
      print("Converged in ",ite,"iterations")
    g_surf = inv(e - epsilon_s) # surface green function
    g_bulk = inv(e - epsilon)  # bulk green function
    return g_bulk, g_surf


if __name__ == '__main__':
    import proplot as plt

    model = RiceMele(0.1, 0.4, 12)
    model.buildH()

    omegas = np.linspace(-1,1, 100)

    ldos = np.zeros(len(omegas))
    for i in range(len(omegas)):
        ldos[i] = model.gf(omegas[i], method='sos')

    fig, ax = plt.subplots()
    ax.plot(omegas, ldos)

    # model.plot_state([1,2,3,4,5,6])
    # print(model.evals)

    lattice = Lattice(size=(20,1), norb=2, orb_coords=[[0.4, 0.4], [0.6,0.6]])
    # for i in range(3):
    #     for j in range(2):
    #         print(lattice.index(i, j, 0))
    lattice.set_hop(0.05, 0, 1, [0, 0])
    lattice.set_hop(0.08, 1, 0, [1, 0])

    evals, evecs = lattice.solve()
    from lime.style import level_scheme
    level_scheme(evals)

    from lime.phys import get_index
    I = get_index(evals, 0)
    # for j in range(0, 10):
    lattice.draw(evecs[:,I])

# def green_renormalization_jit(intra,inter,energy=0.0,delta=1e-4,**kwargs):
#     intra = algebra.todense(intra)*(1.0+0j)
#     inter = algebra.todense(inter)*(1.0+0j)
#     g0 = intra*0.0j
#     g1 = intra*0.0j
#     nite = int(10/delta)
#     error = delta*1e-3
#     energyz = energy + 1j*delta
#     e = np.array(np.identity(intra.shape[0]),dtype=np.complex) * energyz
#     return green_renormalization_jit_core(g0,g1,intra,inter,e,nite,
#                                                 error)
