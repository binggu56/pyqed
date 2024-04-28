import numpy as np
from scipy.sparse import identity, kron
from numpy import meshgrid
from scipy.linalg import eigh
# from cmath import log

import sys
import matplotlib.pyplot as plt


from pyqed import boson, interval, sigmax, sort, ket2dm, overlap,\
    polar2cartesian, Mol, SESolver, dag, SPO, SPO2, SPO3
from pyqed.style import set_style
from pyqed import wavenumber, au2ev
# from pyqed.units import au2ev, wavenumber2hartree

class Triazine:
    """
    2D linear vibronic coupling model
    """
    def __init__(self, x=None, y=None, mass=[1, 1], nstates=3):
        # super().__init__(x, y, mass, nstates=nstates)
    
        # self.dpes()
        self.edip = sigmax()
#        assert(self.edip.ndim == nstates)
        self.mass = mass
        self.nstates = nstates
    
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
    
        self.va = None
        self.v = None
    
    
    def dpes_global(self):
        """
        compute the glocal DPES
    
        Returns
        -------
        None.
    
        """
        x, y = self.x, self.y
        nx = self.nx
        ny = self.ny
        nstates = self.nstates
        omega = 660 * wavenumber
    
        X, Y = np.meshgrid(x, y)

        v = np.zeros((nx, ny, nstates, nstates), dtype=complex)
    
        v[:, :, 0, 0] = omega * (X**2/2. + (Y)**2/2.)
        v[:, :, 1, 1] = omega * (X**2/2. + (Y)**2/2.) + 7.0/au2ev
        v[:, :, 2, 2] = omega * (X**2/2. + (Y)**2/2.) + 7.0/au2ev        
        v[:, :, 1, 2] = 2.2 * omega * (X - 1j * Y)
        v[:, :, 2, 1] = 2.2 * omega * (X + 1j * Y)       
           
        self.v = v
        return

    def apes(self, x):
    
    
        v = self.dpes(x)
    
        # self.v = v
        w, u = eigh(v)
        return w, u
    
    def wilson_loop(self, n=0, r=1):
    
    
        l = identity(self.nstates)
    
    
        for theta in np.linspace(0, 2 * np.pi, 800):
    
            x, y = polar2cartesian(r, theta)
    
    
            w, u = self.apes([x, y])
    
            # print(u[:,0])
            # ground state projection operator
            p = ket2dm(u[:,n])
    
            l = l @ p
    
        return np.trace(l)
    
    def berry_phase(self, n=0, r=1):
    
    
        phase = 0
        z  = 1
    
        w, u = self.apes([r, 0])
        u0 = u[:, n]
    
        uold = u0
        loop = np.linspace(0, 2 * np.pi)
        for i in range(1, len(loop)):
    
            theta = loop[i]
            x, y = polar2cartesian(r, theta)
    
            w, u = self.apes([x, y])
            unew = u[:,n]
            z *= overlap(uold, unew)
    
            uold = unew
    
        z *= overlap(unew, u0)
    
        return z
        # return -np.angle(z)
    
    def apes_global(self):
    
        x = self.x
        y = self.y
        assert(x is not None)
    
        nstates = self.nstates
    
        nx = len(x)
        ny = len(y)
    
        v = np.zeros((nx, ny, nstates))
    
        for i in range(nx):
            for j in range(ny):
                w, u = self.apes([x[i], y[j]])
                v[i, j, :] = w
    
        return v
    
#    def plot_apes(self):
    
#        v = self.apes_global()
#        mayavi([v[:,:,k] for k in range(self.nstates)])
    
    def run(self):
        pass
