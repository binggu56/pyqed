#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:26:00 2023

@author: bingg
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, kron, identity, linalg

from pyqed.units import au2fs, au2k, au2ev
from pyqed import dag, coth, ket2dm, comm, anticomm, sigmax, sort, Composite
from pyqed.optics import Pulse
from pyqed.namd.diabatic import SPO2, SPO3
from pyqed.polariton.cavity import Cavity

import sys
if sys.version_info[1] < 10:
    import proplot as plt
else:
    import matplotlib.pyplot as plt


class VSC:
    """
    2D vibronic model in the diabatic/adiabatic representation coupled to
    a single-mode IR cavity (vibrational strong coupling)
    
    The photon mode is treated the same as a vibrational mode, although the cavity mode
    is not directly coupled to the electronic motion.

    """
    def __init__(self, mol, cav):
        """
        

        Parameters
        ----------
        mol : LVC obj
            vibronic model
        cav : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.mol = mol
        self.cav = cav
        self.x, self.y = mol.x, mol.y
        # self.X, self.Y, self.Q = np.meshgrid(mol.x, mol.y, cav.x)
        self.nx, self.ny = mol.nx, mol.ny
        self.nel = self.mol.nstates
        self.ncav = self.cav.ncav
        self.nstates = self.nel

        if cav.x is None:
            raise ValueError('Please specify the x varible in Cavity.')
        
        self.q = cav.x
        self.nq = len(cav.x)

        

        self.v = self.vd = None
        self.va = None # adiabatic polaritonic PES
        self.diabatic_to_adiabatic = self._transformation = None # diabatic to adiabatic transformation matrix
        self._ground_state = None

    def ground_state(self, representation='adiabatic'):
        # if rwa:
        #     return self.mol.ground_state()
        # else:
        #     # construct the ground-state polaritonic PES

        from pyqed.dvr.dvr_2d import DVR2

        x = self.x
        y = self.y

        if self.va is None:
            self.ppes()

        dvr = DVR2(x, y, mass=self.mol.mass) # for normal coordinates

        if representation == 'adiabatic':
            V = self.va[:, :, 0]
        elif representation == 'diabatic':
            V = self.v[:, :, 0, 0]

        E0, U = dvr.run(V, k=1)

        self._ground_state = U[:, 0].reshape(self.nx, self.ny)

        return E0, self._ground_state


    def dpes(self, g, rwa=False):
        """
        Compute the diabatic potential energy surfaces with cavity

        Parameters
        ----------
        g : TYPE
            DESCRIPTION.
        rwa : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        
        gx, gy = g
        
        mol = self.mol
        cav = self.cav

        omegac = cav.omega

        nx, ny, nq = self.nx, self.ny, self.nq
        x, y, q = self.x, self.y, self.q
        
        nel = mol.nstates

        vd = np.zeros((nx, ny, nq, nel, nel))
        
        # build the global DPES
        if mol.v is None:
            mol.dpes_global()
        

        # surfaces
        for n in range(nel):
            for i in range(nx):
                for j in range(ny):
                    for k in range(nq):
                        vd[i, j, k, n, n] = mol.v[i, j, n, n] + 0.5*omegac**2*q[k]**2\
                            + (gx * x[i] + gy * y[j]) * q[k]

        # diabatic couplings
        for n in range(nel):
            for m in range(n):
                for k in range(nq):
                    vd[:, :, k, n, m] = mol.v[:, :, n, m]
                    vd[:, :, k, m, n] = vd[:, :, k, n, m].conj()
                        


        # # cavity-molecule coupling
        # a = cav.annihilate()

        # v += np.tile(g * kron(mol.edip.real, a + dag(a)).toarray(), (nx, ny, 1, 1))

        self.vd = self.v = vd

        return vd

    def ppes(self, return_transformation=False):
        """
        Compute the polaritonic potential energy surfaces by diagonalization

        Parameters
        ----------
        return_transformation : TYPE, optional
            Return transformation matrices. The default is False.

        Returns
        -------
        E : TYPE
            DESCRIPTION.

        """

        nx = self.nx
        ny = self.ny
        nq = self.nq
        nstates = self.nstates

        E = np.zeros((self.nx, self.ny, nq, nstates))

        if not return_transformation:

            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(nq):
                        V = self.v[i, j, k, :, :]
                        w = np.linalg.eigvalsh(V)
                    # E, U = sort(E, U)
                        E[i, j, k, :] = w
        else:

            T = np.zeros((nx, ny, nq, nstates, nstates), dtype=complex)

            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(nq):
                        V = self.v[i, j, :, :]
                        w, u = sort(*np.linalg.eigh(V))
    
                        E[i, j, k, :] = w
                        T[i, j, k, :, :] = u

            self._transformation = T

        self.va = E

        return E

    def berry_curvature(self, state_id):
        # compute Berry curvature from the A2D transformation matrix
        pass

    def run(self, psi0=None, dt=0.1, Nt=10, t0=0, nout=1):

        if psi0 is None:
            psi0 = np.zeros((self.nx, self.ny, self.nstates))
            psi0[:, :, 0] = self._ground_state

        spo = SPO3(self.x, self.y, self.q, mass=[*self.mol.mass, 1], \
                   nstates=self.nstates)
        spo.V = self.v

        return spo.run(psi0=psi0, dt=dt, nt=Nt, t0=t0, nout=nout)


    def plot_surface(self, state_id=0, representation='adiabatic'):
        from pyqed.style import plot_surface

        if representation == 'adiabatic':
            plot_surface(self.x, self.y, self.va[:, :, state_id])
        else:
            plot_surface(self.x, self.y, self.v[:, :, state_id, state_id])
            
        return

    def plot_wavepacket(self, psilist, **kwargs):

        if not isinstance(psilist, list): psilist = [psilist]


        for i, psi in enumerate(psilist):
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharey=True)

            ax0.contour(self.X, self.Y, np.abs(psi[:,:, 1])**2)
            ax1.contour(self.X, self.Y, np.abs(psi[:, :,0])**2)
            # ax0.format(**kwargs)
            # ax1.format(**kwargs)
            fig.savefig('psi'+str(i)+'.pdf')
        return ax0, ax1

    def plot_ground_state(self, **kwargs):


        fig, ax = plt.subplots()

        ax.contour(self.X, self.Y, np.real(self._ground_state))
        # ax.format(**kwargs)

        fig.savefig('ground_state.pdf')
        return ax

    def promote_op(self, a, kind='mol'):
        """
        promote a local operator to the composite polariton space

        Parameters
        ----------
        a : TYPE
            DESCRIPTION.
        kind : TYPE, optional
            DESCRIPTION. The default is 'mol'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if kind in ['mol', 'm']:

            return kron(a, self.cav.idm)

        elif kind in ['cav', 'c']:

            return kron(self.mol.idm, a)

if __name__ == '__main__':

    from pyqed.phys import gwp
    
    x = np.linspace(-2, 2, 64)
    y = np.linspace(-2, 2, 64)
    q = np.linspace(-2, 2, 64)

    nx = len(x)
    ny = len(y)
    nq = len(q)
    # mol = DHO(x)
    # mol.dpes(d=2, E0=2)


    # pol = VibronicPolariton(mol, cav)
    # pol.dpes(g=0.05)
    # pol.ppes()

    # pol.draw_surfaces(n=4, representation='adiabatic')
    # pol.product_state(0, 0, 0)

    from pyqed.models.pyrazine import LVC2

    mol = LVC2(x, y, mass=[1,1])
    # mol.plot_apes()
    
    
    cav = Cavity(1/au2ev, 3, x)

    
    # VSC
    pol = VSC(mol, cav)
    pol.dpes(g=[0.1, 0.1])
    
    pol.ppes()
    # pol.plot_surface(1, representation='diabatic')

    psi0 = np.zeros((len(x), len(y), len(q), pol.nstates), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            for k in range(nq):
                psi0[i, j, k, 1] = gwp([x[i], y[j], q[k]])
    
    # pol.plot_ground_state()
    
    r = pol.run(psi0=psi0, dt=0.05, Nt=10, nout=2)