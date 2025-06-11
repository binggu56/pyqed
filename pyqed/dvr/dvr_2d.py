"""
Use Discrete Variable Representation method to solve
two-dimensional potentials.

A good general introduction to DVR methods is
Light and Carrington, Adv. Chem. Phys. 114, 263 (2000)
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.sparse.linalg as sla
import scipy.sparse as sp
import scipy.special.orthogonal as ortho
from scipy.sparse import identity
from functools import reduce
import warnings

from pyqed import meshgrid, interval, cartesian_product, SineDVR
from pyqed.phys import discretize
from scipy.io import savemat


def export_to_matlab(fname, psi, fmt='matlab'):

    mdic = {'wavefunction': psi}
    savemat(fname, mdic)
    return


    
class DVRN(object):
    
    def __init__(self, domains, levels, ndim=2, mass=None): #xlim=None, nx, ylim, ny, mx=1, my=1):
        
        self.dvr = []
        for d in range(ndim):
            dvr = SineDVR(domains[d], levels[d])
            self.dvr.append(dvr.copy())
        

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
        assert(len(domains) == len(levels) == ndim)
        
        self.L = [domain[1] - domain[0] for domain in domains]

        
        x = []
        for d in range(ndim):
            x.append(discretize(*domains[d], levels[d]))
        self.x = x
        
        self.dx = [interval[_x] for _x in x]
        
        self.n = [len(_x) for _x in x] 
        
        if mass is None:
            mass = [1, ] * ndim

        # self.X, self.Y = meshgrid(self.x, self.y)
        self.points = np.fliplr(cartesian_product(x))
        self.npts = len(self.points)

        ###
        self.H = None
        self._K = None
        self._V = None
        # self.size = self.nx * self.ny

    def v(self, V):
        """Return the potential matrix with the given potential.
        Usage:
            v_matrix = self.v(V)

        @param[in] V potential function
        @returns v_matrix potential matrix
        """
        if self.ndim == 2:
            x, y = self.x
            X, Y = np.meshgrid(x, y)

            self._V = V(X, Y)
        
        else:
            
            v = []
            for i, point in enumerate(self.points):
                v.append(V(point))
            
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
            boundary_conditions = ['vanishing', ] * self.ndim 
            

        if coords == 'linear':

            T = 0

            for d in self.ndim:
                
                idm = [sp.eye(n) for n in self.n]    

                tx = _kmat(self.n[d], self.L[d], self.mass[d], boundary_condition=boundary_conditions[d])

                idm[d] = tx
                
                T += sp.kron(idm)
                
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

    def buildH(self, V, coords='linear', **kwargs):
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

        if hasattr(V, '__call__'):
            u = self.v(V)
        else:
            u = sp.diags(V.flatten())


        self.H = self.t(coords=coords, **kwargs)  + u

        return self.H

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

    def run(self, V, k=6, **kwargs):

        if self.H is None:
            self.buildH(V, **kwargs)

        # If all eigenvalues are required, then we use np.linalg.eigh()
        if k == self.size:
            E, U = np.linalg.eigh(self.H.toarray())
        # But if we don't need all eigenvalues, only the smallest ones,
        # then when the size of the H matrix becomes large enough, it is
        # better to use sla.eigsh() with a shift-invert method. Here we
        # have to have a good guess for the smallest eigenvalue so we
        # ask for eigenvalues closest to the minimum of the potential.
        else:
            E, U = sp.linalg.eigsh(self.H, k=k, which='SM')


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

class DVR2(object):
    def __cartesian_product(self, arrays):
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


    def __init__(self, xlim, ylim, nx, ny, dvr_type='sine', mass=None):
        # self.dvr1d = dvr1d

        self.nx = nx
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.Lx = self.xmax - self.xmin
        self.ny = ny
        self.Ly = self.ymax - self.ymin
        self.dx = self.Lx/nx
        self.dy = self.Ly/ny
        self.x0 = (self.xmin + self.xmax)/2
        self.y0 = (self.ymin + self.ymax)/2

        # self.x = self.x0 + np.arange(nx) * self.dx - self.Lx/2. + self.dx/2.
        # self.y = self.y0 + np.arange(ny) * self.dy - self.Ly/2. + self.dy/2.

        
        if dvr_type == 'sine':
            dvr_x = SineDVR(*xlim, nx)
            dvr_y = SineDVR(*ylim, ny)
            self.x = dvr_x.x
            self.y = dvr_y.x
            
            self.dvr = [dvr_x, dvr_y]
            
        # self.x = x
        # self.y = y
        # self.nx = len(x)
        # self.ny = len(y)
        # self.xmax = max(x)
        # self.xmin = min(x)
        # self.ymax = max(y)
        # self.ymin = min(y)
        # self.Lx = self.xmax - self.xmin
        # self.Ly = self.ymax - self.ymin
        # self.dx = x[1] - x[0]
        # self.dy = y[1] - y[0]

        if mass is None:
            mass = [1, 1]
        self.mx, self.my = mass

        self.X, self.Y = meshgrid(self.x, self.y)
        self.xy = np.fliplr(self.__cartesian_product([self.x, self.y]))

        self.H = None
        self._K = None
        self._V = None
        self.size = self.nx * self.ny

    def v(self, V, *args):
        """Return the potential matrix with the given potential.
        Usage:
            v_matrix = self.v(V)

        @param[in] 
            V : callable
                potential function
        @returns v_matrix potential matrix
        """

        if callable(V):

            self._V = V(self.X, self.Y, *args)

            return self._V
        
        else:
            self._V = V

    def t(self, boundary_conditions=['vanishing', 'vanishing'], coords='linear',\
          inertia=None):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t()

        For Jacobi coordinates (x = r, y = theta), the kinetic energy operator is given by

        T =  \frac{p_r^2}{2\mu} + \frac{1}{2I(r)} p_\theta^2


        @returns T kinetic energy matrix
        """

        bc_x, bc_y = boundary_conditions

        if coords == 'linear':

            tx = self.dvr[0].t()
            ty = self.dvr[1].t()
            
            idx = sp.identity(self.nx)
            idy = sp.identity(self.ny)
            return sp.kron(idx, ty) + sp.kron(tx, idy)

        elif coords == 'jacobi':

            # tx = _kmat(self.nx, self.Lx, self.mx, boundary_condition=bc_x)
            # ty = _kmat(self.ny, self.Ly, self.my, boundary_condition=bc_y)
            
            tx = self.dvr[0].t()
            ty = self.dvr[1].t()

            idy = sp.identity(self.ny)
            # moment of inertia I(r_i)\delta_{ij}
            I = sp.diags(1./(inertia(self.x)))

            self._K = sp.kron(tx, idy) + sp.kron(I, ty)

            return self._K

    def buildH(self, coords='linear', **kwargs):
        """Return the hamiltonian matrix with the given potential.
        Usage:
            H = self.h(V)

        @param[in] V potential function
        @returns H potential matrix
        kwargs: list
            kwargs for computing kinetic energy matrix
        """

        # if hasattr(V, '__call__'):
        #     u = self.v(V)
        # else:
        u = sp.diags(self._V.flatten())


        self.H = self.t(coords=coords, **kwargs)  + u

        return self.H

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
        if k == self.size:
            E, U = np.linalg.eigh(self.H.toarray())
        # But if we don't need all eigenvalues, only the smallest ones,
        # then when the size of the H matrix becomes large enough, it is
        # better to use sla.eigsh() with a shift-invert method. Here we
        # have to have a good guess for the smallest eigenvalue so we
        # ask for eigenvalues closest to the minimum of the potential.
        else:
            E, U = sp.linalg.eigsh(self.H, k=k, which='SA')


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
        # print('eigenvalues = ', E)
        
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

def _KEO(method='sinc'):
    if method == 'sinc':
        return sincDVR()
    else:
        pass

def sincDVR(N, L, mass):
    n = np.arange(N)

    _m = n[:, np.newaxis]
    _n = n[np.newaxis, :]

    dx = L/N
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2.

    T[n, n] = np.pi**2. / 3. / dx**2.
    T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

    return T

def sincDVRPeriodic():
    pass


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

    if boundary_condition == 'vanishing':

        dx = L/N
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2.

        T[n, n] = np.pi**2. / 3. / dx**2.
        T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

    elif boundary_condition == 'periodic':

        _arg = np.pi*(_m-_n)/N

        if (0 == N // 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = 2.*(-1.)**(_m-_n)/np.sin(_arg)**2.
            T[n, n] = (N**2. + 2.)/3.
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = 2.*(-1.)**(_m-_n)*np.cos(_arg)/np.sin(_arg)**2.
            T[n, n] = (N**2. - 1.)/3.

        T *= (np.pi/L)**2.
        T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

    return T

class VFactory(object):
    """Factory functions to build different potentials
    A factory is a function that returns other functions.
    """
    # def square_well(self, depth = 1., width = 1.,
    #                 origin = 0., o_val = 0.):
    #     """Usage:
    #             V = square_well_factory(**kwargs)

    #     Returns a function of a single variable V(x),
    #     representing the square-well potential:

    #          (-A/2, V0)            (A/2, V0)
    #     ------------       +       ----------------
    #                |               |
    #                |               |
    #                |               |
    #                |               |
    #      (-A/2, 0) |-------+-------| (A/2, 0)
    #                      (0, 0)

    #     Keyword arguments:
    #     @param[in] depth    Depth of the potential well (default=1)
    #     @param[in] width    Width of the potential well (default=1)
    #     @param[in] origin   Location of the well's center (default=0)
    #     @param[in] o_val    Value of the potential at origin (default=0)
    #     @returns   V        The square well potential function V(x)
    #     """
    #     def V(x):
    #         interior_idx = np.abs(x - origin) < width / 2.
    #         V = np.ones_like(x) * (depth + o_val)
    #         V[interior_idx] = o_val
    #         return V
    #     return V

    # def double_well(self, x1 = -2., x2 = -1., x3 = 1.,
    #                 x4 = 2., V1 = 1., V2 = 0.,
    #                 V3 = 1., V4 = 0., V5 = 1.):
    #     """Usage:
    #             V = double_square_well_factory(**kwargs)

    #     Returns a one-dimensional potential function that represents
    #     a double-square-well potential. The potential looks like

    #        (x1, V1)      (x2, V3)   (x3, V3)      (x4, V5)
    #     ----------            ---------            ----------
    #              |            |       |            |
    #              |            |       |            |
    #              |            |       |            |
    #              |            |       |            |
    #              |____________|       |____________|
    #        (x1, V2)      (x2, V2)   (x3, V4)      (x4, V4)

    #     Keywork arguments
    #     @param[in] x1    x-coordinate x1 above (default=-2)
    #     @param[in] x2    x-coordinate x2 above (default=-1)
    #     @param[in] x3    x-coordinate x3 above (default=1)
    #     @param[in] x4    x-coordinate x4 above (default=2)
    #     @param[in] V1    constant V1 above (default=1)
    #     @param[in] V2    constant V2 above (default=0)
    #     @param[in] V3    constant V3 above (default=1)
    #     @param[in] V4    constant V4 above (default=0)
    #     @param[in] V5    constant V5 above (default=1)
    #     @returns   V     double square-well potential V(x)
    #     """
    #     assert (x1 < x2 < x3 < x4), \
    #         "x-coordinates do not satisfy x1 < x2 < x3 < x4"
    #     def V(x):
    #         l_well_idx = np.logical_and(x < x2, x > x1)
    #         r_well_idx = np.logical_and(x < x4, x > x3)
    #         middle_idx = np.logical_and(x >= x2, x <= x3)
    #         far_rt_idx = np.greater_equal(x, x4)
    #         V = np.ones_like(x) * V1
    #         V[l_well_idx] = V2
    #         V[middle_idx] = V3
    #         V[r_well_idx] = V4
    #         V[far_rt_idx] = V5
    #         return V
    #     return V

    def sho(self, k = 1., x0 = 0., y0 = 0.):
        """Usage:
                V = harmosc_factory(**kwargs)

        Return a two-dimensional harmonic oscillator potential V(x, y)
        with wavenumber k.
        i.e. V(x, y) = 1/2 * k * ((x - x0)^2 + (y - y0)^2)

        Keyword arguments
        @param[in] k    wavenumber of the SHO potential (default=1)
        @param[in] x0   x-displacement from origin (default=0)
        @param[in] y0   y-displacement from origin (default=0)
        @returns   V    2-D SHO potential V(x)
        """
        def V(xy): return 0.5 * k * (np.square(xy[:,0] - x0)
                                   + np.square(xy[:,1] - y0))
        return V

    # def power(self, a = 1., p=1., x0 = 0.):
    #     """Usage:
    #             V = self.power(**kwargs)

    #     Return a potential V(x) = a * (x - x0)^p

    #     Keyword arguments
    #     @param[in] a    coefficient (default=1)
    #     @param[in] p    power to raise x (default=1)
    #     @param[in] x0   displacement from origin (default=0)
    #     @returns   V    1-D cubic potential V(x)
    #     """
    #     def V(x): return a * np.power(x - x0, p)
    #     return V

    # def morse(self, D = 1., a = 1., x0 = 0.):
    #     """Usage:
    #             V = morse_factory(**kwargs)

    #     Return a one-dimensional Morse potential V(x)
    #     i.e. V(x) = D * (1 - exp(-a * (x - x0)))^2 - D

    #     Keyword arguments
    #     @param[in] D    dissociation depth
    #     @param[in] a    inverse "width" of the potential
    #     @param[in] x0   equilibrium bond distance
    #     @returns   V    Morse potential V(x)
    #     """
    #     def V(x):
    #         return D * np.power(1. - np.exp(-a * (x - x0)), 2.) - D
    #     return V

    # def sombrero(self, a = -10., b = 1.):
    #     """Usage:
    #             V = sombrero_factory(**kwargs)

    #     Return a one-dimensional version of the sombrero potential
    #     i.e. V(x) = a * x^2 + b * x^4
    #     This function asserts a < 0 and b > 0

    #     Keyword arguments
    #     @param[in] a    coefficient of the x^2 term (default=-10)
    #     @param[in] b    coefficient of the x^4 term (default=1)
    #     @returns   V    1-D Mexican hat potential V(x)
    #     """
    #     assert (a < 0), "Coefficient a must be negative"
    #     assert (b > 0), "Coefficient b must be positive"
    #     def V(x):
    #         return a * np.square(x) + b * np.power(x, 4)
    #     return V

    # def woods_saxon(self, V0 = 50., z = 0.5, r0 = 1.2, A = 16):
    #     """Usage:
    #             V = woods_saxon_factory(**kwargs)

    #     Return a Woods-Saxon potential
    #     i.e. V(r) = - V0 / (1. + exp((r - R) / z))
    #     where R = r0 * A^(1/3)

    #     Keyword arguments
    #     @param[in] V0   potential depth (default=50.)
    #     @param[in] z    surface thickness (default=0.5)
    #     @param[in] r0   rms nuclear radius (default=1.2)
    #     @param[in] A    mass number (default=16)
    #     @returns   V    Woods-Saxon potential V(r)
    #     """
    #     def V(r):
    #         x0 = r0 * np.power(A, 1. / 3.)
    #         return -V0 / (1. + np.exp((r - x0)/ z))
    #     return V



if __name__ == '__main__':
    def sho(x, y, k = 1., x0 = 0., y0 = 0.):
        """Usage:
                V = harmosc_factory(**kwargs)

        Return a two-dimensional harmonic oscillator potential V(x, y)
        with wavenumber k.
        i.e. V(x, y) = 1/2 * k * ((x - x0)^2 + (y - y0)^2)

        Keyword arguments
        @param[in] k    wavenumber of the SHO potential (default=1)
        @param[in] x0   x-displacement from origin (default=0)
        @param[in] y0   y-displacement from origin (default=0)
        @returns   V    2-D SHO potential V(x)
        """
        return 0.5 * (x - x0)**2 + 0.5 * (y - y0)**2 + 2*x*y + x**2 * y + x * y**2 + x**2*y**2
        
    nx, ny = 15, 15
    dvr = DVR2((-6,6), nx, (-6,6), ny)
    
    
    dvr.v(sho)
    E, U = dvr.run(k=3)
    
    
    for j in range(3):
        u, s, vh = np.linalg.svd(U[:, j].reshape(nx, ny))
        
        print(s)
        
        fig, ax = plt.subplots()
        ax.plot(s, 'o')
    
    
    # x = discretize(l=4)
    # print(x)
    # nx = len(x)
    # xy = np.fliplr(__cartesian_product([x, x]))
    # print(len(xy))
    # xy = np.reshape(xy, (nx, nx, 2))
    # X, Y = xy[:, :, 0], xy[:, :, 1]
    
    # X1, Y1 = np.meshgrid(x,x)
    # print(X - X1)    