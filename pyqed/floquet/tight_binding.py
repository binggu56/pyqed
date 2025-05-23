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
from scipy.sparse import csr_matrix

from pyqed.floquet.Floquet import Floquet
from pyqed import Mol

class TightBinding(Mol):
    """
    1D tight-binding chain
    """
    def __init__(self, h, hopping=0, L=None, a=1, mu=0, nk=50, disorder=False):
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
        mu: float
            chemical potential
        a : lattice constant

        Returns
        -------
        None.

        """
        self.L = L
        self.norb = h.shape[0]
        # self.size = nsite * norb
        # self.onsite = onsite
        self.hopping = hopping

        self.H = None
        self.a = a
         
        self.BZ = np.linspace(-np.pi/a, np.pi/a, nk) # first BZ

    def run(self):
        """Compute the band structure of the tight-binding model
        """
        E = np.zeros((nk, norb)) # band structure
        bloch_states = np.zeros((nk, norb, norb))
        for k in range(self.BZ):
            
            
            pass

    def position(self):
        """
        position matrix elements in localized basis/Wannier basis

        valid for open boundary condition

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
        
        return r

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

    def Floquet(self, **args):
        # drive_with_efield

        return FloquetBloch(self.H, -self.position(), **args)

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
        return LvN(*args, **kwargs)


if __name__=="__main__":
    tb = TightBinding()
    print(tb.BZ)