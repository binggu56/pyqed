#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:37:59 2023

Electronic structure solver with sparse grids

@author: Bing Gu
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.sparse.linalg as sla
import scipy.sparse as sp
import scipy.special.orthogonal as ortho
from scipy.sparse import identity, csr_matrix
from functools import reduce
import warnings

from pyqed import meshgrid, interval
from pyqed.phys import discretize, tensor
from scipy.io import savemat


def _cartesian_product(arrays):
    """
    A fast cartesion product function
    """
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
    out = np.empty(rows * cols, dtype=broadcasted[0].dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T

class DVRn():
    
    def __init__(self, mol, domains, levels, ndim=3, mass=None): #xlim=None, nx, ylim, ny, mx=1, my=1):
        # self.dvr1d = dvr1d

        # self.nx = nx
        # self.xmin, self.xmax = xlim
        # self.ymin, self.ymax = ylim
        # self.Lx = self.xmax - self.xmin
        # self.ny = ny
        # self.Ly = self.ymax - self.ymin
        # self.dx = self.Lx/nx
        # self.dy = self.Ly/ny
        # self.x0 = (self.xmin + self.xmax)/2
        # self.y0 = (self.ymin + self.ymax)/2

        # self.x = self.x0 + np.arange(nx) * self.dx - self.Lx/2. + self.dx/2.
        # self.y = self.y0 + np.arange(ny) * self.dy - self.Ly/2. + self.dy/2.

        # self.x = x
        # self.y = y
        # self.nx = len(x)
        # self.ny = len(y)
        # self.xmax = max(x)
        # self.xmin = min(x)
        # self.ymax = max(y)
        # self.ymin = min(y)
        self.mol = mol
        self.ndim = ndim
        if mass is None:
            mass = [1, ] * ndim
        self.mass = mass
        
        assert(len(domains) == len(levels) == ndim)
        
        self.L = [domain[1] - domain[0] for domain in domains]

        
        x = []
        for d in range(ndim):
            x.append(discretize(*domains[d], levels[d]))
        self.x = x
        
        self.dx = [interval(_x) for _x in self.x]
        
        self.n = [len(_x) for _x in x] 
        
        if mass is None:
            mass = [1, ] * ndim

        # self.X, self.Y = meshgrid(self.x, self.y)
        self.points = np.fliplr(_cartesian_product(x))
        self.npts = len(self.points)

        ###
        self.H = None
        self._K = None
        self._V = None
        # self.size = self.nx * self.ny

    def nuclear_attraction(self, r):
        assert(len(r) == 3)
        mol = self.mol 
        
        v = 0
        for a in range(mol.natm):
            Ra = mol.atom_coord(a)
            Za = mol.atom_charge(a)
            
            d = np.linalg.norm(r - Ra)

            if d < 1e-10:
                v += - Za * self.L[0]**3
            else:
                v += - Za/d              

    
        return v
    
    def v(self):
        """Return the Coulomb potential matrix.
        
        .. math::
            
            V(\mathbf{r}) =  \sum_A - \frac{Z_A}{\abs{\mathbf{r} - \mathbf{R}_A}}
            
        Usage:
            v_matrix = self.v(V)

        @param[in] V potential function
        @returns v_matrix potential matrix
        """
        # if self.ndim == 2:
        #     x, y = self.x
        #     X, Y = np.meshgrid(x, y)

        #     self._V = V(X, Y)
        
        # else:
            
        v = []
        for i, point in enumerate(self.points):
            v.append(self.nuclear_attraction(point))
        
        # v = np.reshape(v, self.n + [self.ndim])
        
        self._V = v 
            
        return v
                    

        # return sp.diags(self._V.flatten())

    def t(self, boundary_conditions=None, coords='linear',\
          inertia=None):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t()

        For Jacobi coordinates (x = r, y = theta), the kinetic energy operator is given by

        T =  \frac{p_r^2}{2\mu} + \frac{1}{2I(r)} p_\theta^2


        @returns T kinetic energy matrix
        """
        if boundary_conditions is None:
            boundary_conditions = ['periodic', ] * self.ndim 
            

        if coords == 'linear':

            T = 0

            for d in range(self.ndim):
                
                idm = [sp.eye(n) for n in self.n]    

                tx = _kmat(self.n[d], self.L[d], self.mass[d], boundary_condition=boundary_conditions[d])

                idm[d] = tx
                
                T += tensor(idm)
                
            return T

        elif coords == 'jacobi':
            
            bc_x, bc_y = boundary_conditions


            tx = _kmat(self.nx, self.Lx, self.mx, boundary_condition=bc_x)
            ty = _kmat(self.ny, self.Ly, self.my, boundary_condition=bc_y)

            idy = sp.identity(self.ny)
            # moment of inertia I(r_i)\delta_{ij}
            I = sp.diags(1./(inertia(self.x)))

            self._K = sp.kron(tx, idy) + sp.kron(I, ty)

            return self._K

    def buildH(self, coords='linear', **kwargs):
        """Return the hamiltonian matrix with the given potential.
        Usage:
            H = self.h(V)

        @param[in] 
        V : callable or array
            potential function
        @returns 
        H: array
            potential matrix
        kwargs: list
            kwargs for computing kinetic energy matrix
        """

        # if hasattr(V, '__call__'):
        #     u = self.v(V)
        # else:
        #     u = sp.diags(V.flatten())


        self.H = self.t(coords=coords, **kwargs)  + sp.diags(self.v())

        return (self.H)

    # def plot(self, V, E, U, **kwargs):
    #     doshow = kwargs.get('doshow', False)
    #     nplot = kwargs.get('nplot', 5)
    #     uscale = kwargs.get('uscale', 1.)
    #     xmin = kwargs.get('xmin', self.xy[:,0].min())
    #     xmax = kwargs.get('xmax', self.xy[:,0].max())
    #     ymin = kwargs.get('ymin', self.xy[:,1].min())
    #     ymax = kwargs.get('ymax', self.xy[:,1].max())
    #     zmin = kwargs.get('zmin', np.ceil(V(self.xy).min() - 1.))
    #     zmax = kwargs.get('zmax',
    #                       np.floor(max(U.max()+E.max()+1.,
    #                                V(self.xy).max()+1.)))

    #     npts = self.dvr1d.npts
    #     xy = self.xy.reshape((npts, npts, 2))
    #     vp = V(self.xy).reshape((npts, npts))

    #     colors = tableau20
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     ax.plot_surface(xy[:,:,0], xy[:,:,1], vp,
    #             alpha=0.15, rstride=2, cstride=2)
    #     for i in range(nplot):
    #         if i == 0:
    #             ax.plot_surface(xy[:,:,0], xy[:,:,1],
    #                 uscale * abs(U[:, i].reshape((npts, npts))) + E[i],
    #                 alpha=0.3, color=colors[i],
    #                 rstride=2, cstride=2)
    #         else:
    #             ax.plot_surface(xy[:,:,0], xy[:,:,1],
    #                 uscale * U[:, i].reshape((npts, npts)) + E[i],
    #                 alpha=0.3, color=colors[i],
    #                 rstride=2, cstride=2)
    #     ax.set_xlim3d(xmin, xmax)
    #     ax.set_ylim3d(ymin, ymax)
    #     ax.set_zlim3d(zmin, zmax)
    #     if doshow: plt.show()
    #     return

    def run(self, k=6, **kwargs):

        if self.H is None:
            self.buildH(**kwargs)

        # If all eigenvalues are required, then we use np.linalg.eigh()
        if k >= self.npts or k is None:
            E, U = np.linalg.eigh(self.H.toarray())

        # But if we don't need all eigenvalues, only the smallest ones,
        # then when the size of the H matrix becomes large enough, it is
        # better to use sla.eigsh() with a shift-invert method. Here we
        # have to have a good guess for the smallest eigenvalue so we
        # ask for eigenvalues closest to the minimum of the potential.
        else:
            E, U = sp.linalg.eigsh(self.H, k=k, which='SM')
            
        # nuclear repulsion energy
        E += self.mol.energy_nuc()


        # doshow = kwargs.get('doshow', False)
        # if doshow:
        #     uscale = kwargs.get('uscale', 1.)
        #     xmin = kwargs.get('xmin', self.xy[:,0].min())
        #     xmax = kwargs.get('xmax', self.xy[:,0].max())
        #     ymin = kwargs.get('ymin', self.xy[:,1].min())
        #     ymax = kwargs.get('ymax', self.xy[:,1].max())
        #     zmin = kwargs.get('zmin', np.ceil(V(self.xy).min() - 1.))
        #     zmax = kwargs.get('zmax',
        #                       np.floor(max(U.max()+E.max()+1.,
        #                                V(self.xy).max()+1.)))

        #     self.plot(V, E, U, nplot=num_eigs,
        #               xmin=xmin, xmax=xmax,
        #               ymin=ymin, ymax=ymax,
        #               zmin=zmin, zmax=zmax,
        #               uscale=uscale, doshow=doshow)
        return E, U

    def plot(self, U, **kwargs):

        nx, ny = self.nx, self.ny

        for k in range(U.shape[-1]):

            chi = U[:, k].reshape(nx, ny)
            fig, ax = plt.subplots()
            ax.contourf(self.X, self.Y, chi, **kwargs)

        return

    def plot_surface(self, **kwargs):


        fig, ax = plt.subplots()
        ax.contour(self.X, self.Y, self._V, **kwargs)

        return

    def test_potential(self, V, num_eigs = 5, **kwargs):

        h = self.buildH(V)
        # Get the eigenpairs
        # There are multiple options here.
        # If the user is asking for all of the eigenvalues,
        # then we need to use np.linalg.eigh()
        if num_eigs == h.shape[0]:
            E, U = np.linalg.eigh(h)
        # But if we don't need all eigenvalues, only the smallest ones,
        # then when the size of the H matrix becomes large enough, it is
        # better to use sla.eigsh() with a shift-invert method. Here we
        # have to have a good guess for the smallest eigenvalue so we
        # ask for eigenvalues closest to the minimum of the potential.
        else:
            E, U = sla.eigsh(h, k=num_eigs, which='SM')

        precision = kwargs.get('precision', 8)

        # Print and plot stuff
        print('The first {n:d} energies are:'.format(n=num_eigs))
        print(np.array_str(E[:num_eigs], precision=precision))

        doshow = kwargs.get('doshow', False)
        if doshow:
            uscale = kwargs.get('uscale', 1.)
            xmin = kwargs.get('xmin', self.xy[:,0].min())
            xmax = kwargs.get('xmax', self.xy[:,0].max())
            ymin = kwargs.get('ymin', self.xy[:,1].min())
            ymax = kwargs.get('ymax', self.xy[:,1].max())
            zmin = kwargs.get('zmin', np.ceil(V(self.xy).min() - 1.))
            zmax = kwargs.get('zmax',
                              np.floor(max(U.max()+E.max()+1.,
                                       V(self.xy).max()+1.)))

            self.plot(V, E, U, nplot=num_eigs,
                      xmin=xmin, xmax=xmax,
                      ymin=ymin, ymax=ymax,
                      zmin=zmin, zmax=zmax,
                      uscale=uscale, doshow=doshow)
        return E, U

    def sho_test(self, k = 1., num_eigs=8, precision=8,
                 uscale=1., doshow=False):
        print('Testing 2-D DVR with an SHO potential')
        # vF = VFactory()
        # V = vF.sho(k=k)
        def V(x, y):
            return 0.5 * (x**2 + y**2)
        E, U = self.test_potential(V, doshow=doshow, num_eigs=num_eigs,
                                   precision=precision, uscale=uscale,
                                   xmin=-3.5, xmax=3.5,
                                   ymin=-3.5, ymax=3.5,
                                   zmin=-0.05, zmax=4.)

        return E, U

def _kmat(N, L, mass, boundary_condition='vanishing'):
    """
    Sinc function basis for periodic functions over an interval
    `[x0 - L/2, x0 + L/2]` with `N` points

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    mass : TYPE
        DESCRIPTION.

    Returns
    -------
    T : TYPE
        DESCRIPTION.

    """
    n = np.arange(N)

    _m = n[:, np.newaxis]
    _n = n[np.newaxis, :]

    # if boundary_condition == 'vanishing':

    dx = L/N
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2.

    T[n, n] = np.pi**2. / 3. / dx**2.
    T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

    # elif boundary_condition == 'periodic':

    #     _arg = np.pi*(_m-_n)/N

    #     if (0 == N // 2):
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             T = 2.*(-1.)**(_m-_n)/np.sin(_arg)**2.
    #         T[n, n] = (N**2. + 2.)/3.
    #     else:
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             T = 2.*(-1.)**(_m-_n)*np.cos(_arg)/np.sin(_arg)**2.
    #         T[n, n] = (N**2. - 1.)/3.

    #     T *= (np.pi/L)**2.
    #     T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

    return T

class Triatomic:
    def __init__(self, coords, Z):
        self.coords = coords 
        self.charge = self.Z = Z
        
    def run(self):
        pass

if __name__ == '__main__':
    from pyscf import gto 
    mol = gto.M(atom='H 0 0 0', spin=1)
    dvr = DVRn(mol, domains=[[-6, 6],] * 3, levels=[5, ] * 3)   
    dvr.buildH()
    # print(dvr._V)
    e, u = dvr.run(k=6)
    print('Energy = ', e)

        

