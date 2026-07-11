"""
Use a simple Discrete Variable Representation method to solve
one-dimensional potentials.

A good general introduction to DVR methods is
Light and Carrington, Adv. Chem. Phys. 114, 263 (2000)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.special.orthogonal as ortho
import scipy
# import bessel
import warnings
from opt_einsum import contract

from pyqed import interval


def kinetic(x, mass=1, dvr='sinc'):
    """
    kinetic enegy operator for the DVR set

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    mass : TYPE, optional
        DESCRIPTION. The default is 1.
    dvr : TYPE, optional
        DESCRIPTION. The default is 'sinc'.

    Returns
    -------
    Tx : TYPE
        DESCRIPTION.


    Refs:

        M.H. Beck et al. Physics Reports 324 (2000) 1-105


    """

    # L = xmax - xmin
    # a = L / npts
    nx = len(x)
        # self.n = np.arange(npts)
        # self.x = self.x0 + self.n * self.a - self.L / 2. + self.a / 2.
        # self.w = np.ones(npts, dtype=np.float64) * self.a
        # self.k_max = np.pi/self.a

    L = x[-1] - x[0]
    dx = interval(x)
    n = np.arange(nx)
    nx = npts = len(x)


    if dvr == 'sinc':

        # Colbert-Miller DVR 1992

        _m = n[:, np.newaxis]
        _n = n[np.newaxis, :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / dx**2

        T[n, n] = np.pi**2. / 3. / dx**2
        T *= 0.5/mass   # (pc)^2 / (2 mc^2)

    elif dvr == 'sine':

        # Sine DVR (particle in-a-box)
        # n = np.arange(1, npts + 1)
        # dx = (xmax - xmin)/(npts + 1)
        # x = float(xmin) + self.a * self.n

        npts = N = len(x)
        n = np.arange(1, npts + 1)


        _i = n[:, np.newaxis]
        _j = n[np.newaxis, :]

        L = dx * (npts + 1)

        m = npts + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = ((-1.)**(_i-_j)
                * (1./np.square(np.sin(np.pi / (2. * m) * (_i-_j)))
                - 1./np.square(np.sin(np.pi / (2. * m) * (_i+_j)))))

        T[n - 1, n - 1] = 0.
        T += np.diag((2. * m**2. + 1.) / 3.
                      - 1./np.square(np.sin(np.pi * n / m)))
        T *= np.pi**2. / 2. / L**2. #prefactor common to all of T
        T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     T = 2 * (-1.)**(_i-_j)/(N+1)**2 * \
        #         np.sin(np.pi * _i/(N+1)) * np.sin(np.pi * _j/(N+1))\
        #         /(np.cos(np.pi * _i /(N+1)) - np.cos(_j * np.pi/(N+1)))**2

        # T[n - 1, n - 1] = 0.
        # T += np.diag(-1/3 + 1/(6 * (N+1)**2) - 1/(2 * (N+1)**2 * np.sin(n * np.pi/(N+1))**2))

        # T *= np.pi**2. / (2. * mass * dx**2) #prefactor common to all of T

    elif dvr == 'SincPeriodic':

        _m = n[:, np.newaxis]
        _n = n[np.newaxis, :]

        _arg = np.pi*(_m-_n)/nx

        if (0 == nx % 2):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = 2.*(-1.)**(_m-_n)/np.sin(_arg)**2.

            T[n, n] = (nx**2. + 2.)/3.
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = 2.*(-1.)**(_m-_n)*np.cos(_arg)/np.sin(_arg)**2.
            T[n, n] = (nx**2. - 1.)/3.

        T *= (np.pi/L)**2.
        T *= 0.5 / mass   # (pc)^2 / (2 mc^2)

    return T

class DVR(object):


    def v(self, V):
        """
        Return the potential matrix with the given potential.
        Usage:
            v_matrix = self.v(V)

        @param[in] V potential function
        @returns v_matrix potential matrix
        """
        v_matrix = np.diag(V(self.x))
        return v_matrix

    def h(self, V):
        """Return the hamiltonian matrix with the given potential.
        Usage:
            H = self.h(V)

        @param[in] V potential function
        @returns H potential matrix
        """
        return self.t() + self.v(V)

    def dvr2fbr(DVR, T):
        """Transform a matrix from the discrete variable representation
        to the finite basis representation"""
        return np.dot(T, np.dot(DVR, np.transpose(T)))

    def fbr2dvr(FBR, T):
        """Transform a matrix from the finite basis representation to the
        discrete variable representation."""
        return np.dot(np.transpose(T), np.dot(FBR, T))

    def plot(self, V, E, U, **kwargs):
        doshow = kwargs.get('doshow', False)
        nplot = kwargs.get('nplot', 5)
        xmin = kwargs.get('xmin', self.x.min())
        xmax = kwargs.get('xmax', self.x.max())
        ymin = kwargs.get('ymin', np.ceil(V(self.x).min() - 1.))
        ymax = kwargs.get('ymax',
                          np.floor(max(U.max()+E.max()+1., V(self.x).max()+1.)))
        plt.plot(self.x, V(self.x))
        for i in range(nplot):
            if i == 0:
                plt.plot(self.x, abs(U[:, i])+E[i])
            else:
                plt.plot(self.x, U[:, i]+E[i])
        plt.axis(ymax=ymax, ymin=ymin)
        plt.axis(xmax=xmax, xmin=xmin)
        if doshow: plt.show()
        return

    def run(self, V, num_eigs = 5, **kwargs):
        h = self.h(V)
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
            E, U = sla.eigsh(h, k=num_eigs, which='LM',
                             sigma=V(self.x).min())

        self.eigvals = E
        self.eigvecs = U
        self.potential = V
        return E, U

    def draw_states(self, **kwargs):
        E, U = self.eigvals, self.eigvecs
        V = self.potential

        num_eigs = len(E)

        xmin = kwargs.get('xmin', self.x.min())
        xmax = kwargs.get('xmax', self.x.max())
        ymin = kwargs.get('ymin', np.ceil(V(self.x).min() - 1.))
        ymax = kwargs.get('ymax',
                          np.floor(max(U.max()+E.max()+1., V(self.x).max()+1.)))
        precision = kwargs.get('precision', 8)

        # Print and plot stuff
        print('The first {n:d} energies are:'.format(n=num_eigs))
        print(np.array_str(E[:num_eigs], precision=precision))
        self.plot(V, E, U, nplot=num_eigs,
                  xmin=xmin, xmax=xmax,
                  ymin=ymin, ymax=ymax,
                  doshow=True)
        return

    def inf_square_well_test(self, precision=8):
        print('Testing 1-D DVR with an infinite square-well potential')
        vF = VFactory()
        V = vF.square_well(depth=1e30, width=10.)
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=-10., xmax=10.,
                            ymin=-0.25, ymax=2.)
        e_exact = np.square(np.arange(1,6)) * np.pi**2. / 2. / 10.**2.
        print("Compare to the exact energies:")
        print(np.array_str(e_exact, precision=precision))
        print
        return

    def square_well_test(self, precision=8):
        #print'Testing 1-D DVR with a finite square-well potential'
        vF = VFactory()
        V = vF.square_well(depth=9./2., width=10.)
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=-10., xmax=10.,
                            ymin=-0.25, ymax=2.)
        e_exact = 9./2. * np.array([0.009636, 0.038522, 0.086582,
                                    0.153683, 0.239608])
        print("Compare to these energies:")
        print(np.array_str(e_exact, precision=precision))
        #print "from: http://pilotscholars.up.edu/phy_facpubs/8\n"
        return

    def double_well_test(self, precision=8):
        print('Testing 1-D DVR with a double-well potential')
        vF = VFactory()
        V = vF.double_well()
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=-3.5, xmax=3.5,
                            ymin=-0.5, ymax=4.)
        print
        return

    def sho_test(self, k=1., num_eigs=5, precision=8,
            xmin=-3.5, xmax=3.5, ymin=0., ymax=6.):
        print('Testing 1-D DVR with an SHO potential')
        vF = VFactory()
        V = vF.sho(k=k)
        self.run(V, num_eigs=num_eigs,
                            precision=precision,
                            xmin=xmin, xmax=xmax,
                            ymin=ymin, ymax=ymax)
        print
        return

    def morse_test(self, precision=8, xmin=0., xmax=32., ymin=-3., ymax=1.):
        print('Testing 1-D DVR with a Morse potential')
        vF = VFactory()
        V = vF.morse(D=3., a=0.5)
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=xmin, xmax=xmax,
                            ymin=ymin, ymax=ymax)
        print
        return

    def sombrero_test(self, precision=8):
        print('Testing 1-D DVR with a sombrero potential')
        vF = VFactory()
        V = vF.sombrero(a=-5.)
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=-5., xmax=5., ymax=5.)
        print
        return

    def woods_saxon_test(self, precision=8):
        print('Testing 1-D DVR with a Woods-Saxon potential')
        vF = VFactory()
        V = vF.woods_saxon(A=4)
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=0., xmax=5.,
                            ymin=-50., ymax=0.)
        print
        return

    def test_all(self, precision=8):
        self.square_well_test(precision=precision)
        self.double_well_test(precision=precision)
        self.sho_test(precision=precision)
        self.morse_test(precision=precision)
        self.sombrero_test(precision=precision)
        self.woods_saxon_test(precision=precision)



class SincDVR(DVR):
    r"""Sinc function basis for non-periodic functions over an interval
    `x0 +- L/2` with `npts` points.
    Usage:
        d = sincDVR1D(npts, L, [x0])

    @param[in] npts number of points
    @param[in] L size of interval
    @param[in] x0 origin offset (default=0)
    @attribute a step size
    @attribute n vector of x-domain indices
    @attribute x discretized x-domain
    @attribute k_max cutoff frequency
    @method h return hamiltonian matrix
    @method f return DVR basis vectors
    """
    def __init__(self,  L, npts, x0=0.):

        self.npts = npts
        # self.L = x.max() - x.min()
        self.L = L
        # self.x0 = x[self.npts//2]
        self.a = self.dx = L / npts
        self.x0 = x0
        # self.a = interval(x)
        self.n = np.arange(self.npts)
        # self.x = self.x0 + self.n * self.a - self.L / 2. + self.a / 2.
        self.x = self.x0 + self.n * self.a - self.L / 2.

        # self.x = x
        self.w = np.ones(self.npts, dtype=np.float64) * self.a
        self.k_max = np.pi/self.a

        self.potential = None
        self.eigvals = None
        self.eigvecs = None

    def t(self, hc=1., mc2=1.):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t()

        @returns T kinetic energy matrix
        """
        _m = self.n[:, np.newaxis]
        _n = self.n[np.newaxis, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / self.a**2.
        T[self.n, self.n] = np.pi**2. / 3. / self.a**2.
        T *= 0.5 * hc**2. / mc2   # (pc)^2 / (2 mc^2)
        return T

    def ip(self, hbar=1.):
        """Return the momentum matrix times i (imaginary number)
        i.e. ip = hbar frac{d}{dx}
        Usage:
            iP = self.p()

        @returns iP momentum matrix times i (imaginary number)
        """
        _m = self.n[:, np.newaxis]
        _n = self.n[np.newaxis, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iP = (-1.)**(_m-_n) / (_m-_n) / self.a
        iP[self.n, self.n] = 0.
        iP *= hbar
        return iP

    def momentum(self, hbar=1.):
        """Return the momentum matrix times i (imaginary number)
        i.e. ip = hbar frac{d}{dx}
        Usage:
            iP = self.p()

        @returns iP momentum matrix times i (imaginary number)
        """
        return -1j * self.ip()

    def f(self, x=None):
        """Return the DVR basis vectors"""
        if x is None:
            x_m = self.x[:, np.newaxis]
        else:
            x_m = np.asarray(x)[:, np.newaxis]
        x_n = self.x[np.newaxis, :]
        return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)
    
    def run(self, num_eigs = 5, **kwargs):
        
        assert self.v is not None
        
        h = self.t() + np.diag(self.v)
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
            E, U = sla.eigsh(h, k=num_eigs, which='LM',
                             sigma=self.v.min())

        self.eigvals = E
        self.eigvecs = U
        # self.potential = V
        return E, U

# class SincDVRPeriodic(SincDVR):
class ExponentialDVR(SincDVR):
    r"""
    Sinc function basis for periodic functions over an interval
    `x0 +- L/2` with `N = 2n + 1` points.

    Refs
        M.H. Beck et al. Physics Reports 324 (2000) 1-105, P94

    """
    def __init__(self, n, L=1 ,x0=0, *v, **kw):
        # Small shift here for consistent abscissa
        # SincDVR.__init__(self, *v, **kw)
        # self.x -= self.a/2.
        self.npts = self.N = 2*n + 1
        self.L = L
        self.n = np.arange(self.npts)
        self.x0 = x0
        self.a = self.L/self.npts
        self.x = self.x0 + self.n * self.a - self.L / 2.

        self.kx = (self.n - n) * 2 * np.pi/self.L
        # scipy.fftpack.fftfreq

    def t(self, hc=1., mc2=1.):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t(V)

        @returns T kinetic energy matrix
        """
        _m = self.n[:, np.newaxis]
        _n = self.n[np.newaxis, :]
        _arg = np.pi*(_m-_n)/self.npts
        if (0 == self.npts % 2):
            T = 2.*(-1.)**(_m-_n)/np.sin(_arg)**2.
            T[self.n, self.n] = (self.npts**2. + 2.)/3.
        else:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = 2.*(-1.)**(_m-_n)*np.cos(_arg)/np.sin(_arg)**2.

            T[self.n, self.n] = (self.npts**2. - 1.)/3.

        T *= (np.pi/self.L)**2.
        T *= 0.5 * hc**2. / mc2   # (pc)^2 / (2 mc^2)
        return T

    def derivative(self):
        """
        DVR expression for derivative operator d/dx

        Returns
        -------
        D : TYPE
            DESCRIPTION.

        """
        _m = self.n[:, np.newaxis]
        _n = self.n[np.newaxis, :]


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            D = np.pi/self.L * (-1.)**(_m-_n)/np.sin(np.pi * (_m - _n)/self.npts)

        return D


    def f(self, x=None):
        """Return the DVR basis vectors"""
        if x is None:
            x_m = self.x[:, np.newaxis]
        else:
            x_m = np.asarray(x)[:, np.newaxis]
        x_n = self.x[np.newaxis, :]
        f = np.sinc((x_m-x_n)/self.a)/np.sinc((x_m-x_n)/self.L)/np.sqrt(self.a)

        if (0 == self.npts % 2):
            f *= np.exp(-1j*np.pi*(x_m-x_n)/self.L)
        return f

    def run(self, v, k=6):
        if callable(v):
            V = np.diag(v(self.x))
        else:
            V = np.diag(v)

        h = V + self.t()


        # Get the eigenpairs
        # There are multiple options here.
        # If the user is asking for all of the eigenvalues,
        # then we need to use np.linalg.eigh()
        if k == h.shape[0]:
            E, U = np.linalg.eigh(h)
        # But if we don't need all eigenvalues, only the smallest ones,
        # then when the size of the H matrix becomes large enough, it is
        # better to use sla.eigsh() with a shift-invert method. Here we
        # have to have a good guess for the smallest eigenvalue so we
        # ask for eigenvalues closest to the minimum of the potential.
        else:
            E, U = sla.eigsh(h, k=k, which='LM',
                             sigma=V.min())

        self.eigvals = E
        self.eigvecs = U
        # self.potential = V
        return E, U



class SineDVR(DVR):
    r"""Sine function basis for non-periodic functions over an interval
    `x_min ... x_max` with `npts` points.
    Usage:
        d = sincDVR1D(npts, xmin, xmax)

        @param[in] npts number of points
        @param[in] xmin "left" end of interval
        @param[in] xmax "right" end of interval
        @attribute a step size
        @attribute n vector of x-domain indices
        @attribute x discretized x-domain
        @attribute k_max cutoff frequency
        @attribute L size of x-domain
        @method h return hamiltonian matrix
        @method f return DVR basis vectors


    """
    def __init__(self, xmin, xmax, npts, mass=1):
        """


        Parameters
        ----------
        xmin : TYPE
            DESCRIPTION.
        xmax : TYPE
            DESCRIPTION.
        npts : int
            number of basis functions/grids points (excluding boundary, 2^l -1).

        Returns
        -------
        None.

        """
        self.npts = npts
        self.xmin = xmin
        self.xmax = xmax
        self.L = float(xmax - xmin)
        self.dx = self.L /(npts + 1)
        self.n = np.arange(1, npts + 1)
        self.x = float(xmin) + self.dx * self.n
        self.k_max = None




        ###
        self.T = None
        self.U = None

        self._mass = mass
        self.v = None

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = value



    def t_fbr(self):
        m = self.mass
        l = self.L

        return (0.5 / m) * (np.pi / l)**2 * np.arange(1, self.npts + 1)**2

    def t(self, hc=1., mc2=1.):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t(V)

        @returns T kinetic energy matrix
        """
        _i = self.n[:, np.newaxis]
        _j = self.n[np.newaxis, :]
        m = self.npts + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            T = ((-1.)**(_i-_j)
                * (1./np.square(np.sin(np.pi / (2. * m) * (_i-_j)))
                - 1./np.square(np.sin(np.pi / (2. * m) * (_i+_j)))))

        T[self.n - 1, self.n - 1] = 0.
        T += np.diag((2. * m**2. + 1.) / 3.
                     - 1./np.square(np.sin(np.pi * self.n / m)))

        T *= np.pi**2. / 2. / self.L**2 #prefactor common to all of T
        T *= 0.5 * hc**2. / self.mass   # (pc)^2 / (2 mc^2)

        self.T = T
        return T

    def momentum(self):
        """
        momentum operator matrix elements in DVR

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if self.U is None:
            self.fbr2dvr()

        U = self.U
        p = np.zeros((self.npts, self.npts))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            p = (np.subtract.outer(self.n, self.n) % 2) * np.outer(self.n, self.n)/np.subtract.outer(self.n**2, self.n**2)

        p[np.isnan(p)] = 0
        p = p * (-4j)/self.L

        return contract('ia, ij, jb -> ab', U.conj(), p, U)

    def expT(self, dt=1):
        """
        kinetic energy propagator

        .. math::

            e^{-i T t}

        Parameters
        ----------
        dt : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        U = self.fbr2dvr()

        _t = np.exp(-1j * dt/(2 * self.mass) * self.n**2 * np.pi**2/self.L**2)

        return contract('ia, i, ib -> ab', U.conj(), _t, U)
        # return U.conj().T @ np.diag(_t) @ U


    def fbr2dvr(self):
        """
        transformation matrix from FBR to DVR

        .. math::

            U_{j\alpha} = \sqrt{2/(n+1)} \sin(j \alpha \pi/(n+1))

        for j labels the FBR and :math:`\alpha = 1, 2, ..., n` labels the DVR set.

        Returns
        -------
        U : TYPE
            DESCRIPTION.

        """
        n = self.npts

        U = np.sin(np.outer(self.n, self.n) * np.pi/(n+1)) * np.sqrt(2./(n+1))

        self.U = U
        return U


    def basis(self, x, a=0):
        """
        Return the DVR basis vectors

        Parameters
        ----------
        a : int
            a-th DVR basis set.

        Returns
        -------
        None.

        """

        # a += 1
        center = self.xmin + (a + 1) * self.dx
        n = self.npts
        L = self.L
        _x = (x - center)/L
        _y = (x + center)/L


        chi = 1/(2 * np.sqrt(L* (n+1))) * \
            (np.sin((2*n+1) * np.pi/2 * _x)/np.sin(np.pi/2*_x) - \
              np.sin((2*n+1) * np.pi/2 * _y)/np.sin(np.pi/2*_y))


        return chi

    def run(self, num_eigs = 5, **kwargs):
        
        assert self.v is not None
        
        h = self.t() + np.diag(self.v)
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
            E, U = sla.eigsh(h, k=num_eigs, which='LM',
                             sigma=self.v.min())

        self.eigvals = E
        self.eigvecs = U
        # self.potential = V
        return E, U

#         if x is None:
#             x_m = self.x[:, np.newaxis]
#         else:
#             x_m = np.asarray(x)[:, np.newaxis]
#         x_n = self.x[np.newaxis, :]
#         return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)


class PODVR(DVR):
    r"""Potential-optimized DVR built from a primitive sine DVR.

    The DVR is built by:

    1. constructing a primitive sine DVR on ``[xmin, xmax]``;
    2. diagonalizing a reference Hamiltonian in that primitive basis;
    3. retaining the lowest ``npts`` reference eigenfunctions as an FBR;
    4. diagonalizing the coordinate operator in that truncated FBR.

    The resulting grid points are nonuniform and concentrated in the region
    important for the chosen reference potential.  If ``v_ref`` is omitted,
    a Morse-like reference potential is used, which is convenient for
    stretching coordinates.

    This class exposes the same basic interface as :class:`SineDVR`: ``x``,
    ``dx``, ``w``, ``t()``, and ``momentum()``.  The kinetic and momentum
    matrices are the primitive operators projected into the optimized DVR.
    """

    def __init__(
        self,
        xmin,
        xmax,
        npts,
        v_ref=None,
        De=0.2,
        a=1.0,
        re=None,
        mass=1.0,
        primitive_npts=None,
    ):
        if npts < 1:
            raise ValueError("npts must be positive.")
        if xmax <= xmin:
            raise ValueError("xmax must be larger than xmin.")
        if v_ref is None and De <= 0:
            raise ValueError("De must be positive.")
        if v_ref is None and a <= 0:
            raise ValueError("a must be positive.")

        self.npts = int(npts)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.L = self.xmax - self.xmin
        self.v_ref = v_ref
        self.De = None if De is None else float(De)
        self.a = None if a is None else float(a)
        self.re = 0.5 * (self.xmin + self.xmax) if re is None else float(re)
        self._mass = float(mass)
        self.primitive_npts = int(
            primitive_npts if primitive_npts is not None else max(4 * self.npts + 20, self.npts + 8)
        )
        if self.primitive_npts < self.npts:
            raise ValueError("primitive_npts must be >= npts.")

        primitive = SineDVR(self.xmin, self.xmax, self.primitive_npts, mass=self.mass)
        V_ref = np.diag(self.reference_potential(primitive.x))
        H_ref = primitive.t() + V_ref
        evals, evecs = np.linalg.eigh(H_ref)

        C = evecs[:, :self.npts]
        x_fbr = C.conj().T @ np.diag(primitive.x) @ C
        x_grid, U = np.linalg.eigh(x_fbr)
        order = np.argsort(x_grid)

        self.x = np.asarray(x_grid[order], dtype=float)
        self.n = np.arange(self.npts)
        self.U = U[:, order]
        self.reference_energies = evals[:self.npts]
        self.primitive = primitive
        self.fbr = C

        T_fbr = C.conj().T @ primitive.t() @ C
        P_fbr = C.conj().T @ primitive.momentum() @ C
        self.T = self.U.conj().T @ T_fbr @ self.U
        self.P = self.U.conj().T @ P_fbr @ self.U
        self.T = 0.5 * (self.T + self.T.conj().T)
        self.P = 0.5 * (self.P + self.P.conj().T)

        self.w = self._voronoi_weights()
        self.dx = float(np.mean(self.w))
        self.k_max = None
        self.v = None

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = float(value)

    def reference_potential(self, x):
        x = np.asarray(x, dtype=float)
        if self.v_ref is not None:
            if callable(self.v_ref):
                values = self.v_ref(x)
            else:
                values = self.v_ref
            values = np.asarray(values, dtype=float)
            if values.shape != x.shape:
                raise ValueError("v_ref must return one value for each grid point.")
            return values
        return self.De * (1.0 - np.exp(-self.a * (x - self.re))) ** 2

    def _voronoi_weights(self):
        if self.npts == 1:
            return np.asarray([self.L], dtype=float)
        edges = np.empty(self.npts + 1, dtype=float)
        edges[0] = self.xmin
        edges[-1] = self.xmax
        edges[1:-1] = 0.5 * (self.x[:-1] + self.x[1:])
        return np.diff(edges)

    def t(self, hc=1.0, mc2=None):
        if hc != 1.0 or mc2 is not None:
            scale = hc ** 2
            if mc2 is not None:
                scale *= self.mass / mc2
            return scale * self.T
        return self.T

    def momentum(self):
        return self.P

    def fbr2dvr(self):
        return self.U

    def f(self, x=None):
        if x is None:
            x = self.x
        x = np.asarray(x, dtype=float)
        primitive_values = np.column_stack([
            self.primitive.basis(x, a=i) for i in range(self.primitive_npts)
        ])
        return primitive_values @ self.fbr @ self.U

    def run(self, v=None, num_eigs=5):
        if v is None:
            if self.v is None:
                raise ValueError("Provide v or set self.v before run().")
            V = np.diag(self.v)
        elif callable(v):
            V = np.diag(v(self.x))
        else:
            V = np.diag(np.asarray(v, dtype=float))

        H = self.t() + V
        if num_eigs == H.shape[0]:
            E, U = np.linalg.eigh(H)
        else:
            E, U = sla.eigsh(H, k=num_eigs, which='LM', sigma=np.diag(V).min())
        self.eigvals = E
        self.eigvecs = U
        return E, U


class FEDVR(DVR):
    r"""One-dimensional finite-element DVR with Gauss-Lobatto points.

    The interval ``[xmin, xmax]`` is split into ``n_elements`` finite elements.
    Each element uses ``n_lobatto`` Gauss-Lobatto-Legendre nodes and local
    Lagrange cardinal functions.  Shared element-boundary nodes are merged into
    bridge functions.  Local potentials are diagonal under Lobatto quadrature,
    while the kinetic-energy matrix is sparse and block-local.

    By default the two outer boundary nodes are removed, corresponding to
    Dirichlet boundary conditions on a finite box.
    """

    def __init__(
        self,
        xmin,
        xmax,
        n_elements,
        n_lobatto=5,
        mass=1.0,
        boundary="dirichlet",
    ):
        if xmax <= xmin:
            raise ValueError("xmax must be larger than xmin.")
        if n_elements < 1:
            raise ValueError("n_elements must be positive.")
        if n_lobatto < 2:
            raise ValueError("n_lobatto must be at least 2.")
        boundary = boundary.lower()
        if boundary not in ("dirichlet", "none"):
            raise ValueError("boundary must be 'dirichlet' or 'none'.")

        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.L = self.xmax - self.xmin
        self.n_elements = int(n_elements)
        self.n_lobatto = int(n_lobatto)
        self._mass = float(mass)
        self.boundary = boundary
        self.v = None

        self.ref_x, self.ref_w = self._lobatto_nodes_weights(self.n_lobatto)
        self.ref_D = self._differentiation_matrix(self.ref_x)

        self._assemble()
        self.T = None
        self.P = None
        self.eigvals = None
        self.eigvecs = None

    @staticmethod
    def _lobatto_nodes_weights(n):
        if n == 2:
            x = np.array([-1.0, 1.0])
            w = np.array([1.0, 1.0])
            return x, w

        poly = np.polynomial.legendre.Legendre.basis(n - 1)
        interior = poly.deriv().roots()
        x = np.concatenate(([-1.0], interior, [1.0]))
        pvals = poly(x)
        w = 2.0 / (n * (n - 1) * pvals**2)
        return x, w

    @staticmethod
    def _differentiation_matrix(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        bary = np.ones(n, dtype=float)
        for j in range(n):
            bary[j] = 1.0 / np.prod(x[j] - np.delete(x, j))

        D = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = bary[j] / bary[i] / (x[i] - x[j])
            D[i, i] = -np.sum(D[i, np.arange(n) != i])
        return D

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = float(value)
        self.T = None
        self.P = None

    def _assemble(self):
        n_full = self.n_elements * (self.n_lobatto - 1) + 1
        h = self.L / self.n_elements
        scale = 0.5 * h

        full_x = np.empty(n_full, dtype=float)
        full_w = np.zeros(n_full, dtype=float)
        stiffness = sp.lil_matrix((n_full, n_full), dtype=float)
        derivative = sp.lil_matrix((n_full, n_full), dtype=float)

        for elem in range(self.n_elements):
            left = self.xmin + elem * h
            center = left + scale
            local_x = center + scale * self.ref_x
            local_w = scale * self.ref_w
            local_ids = elem * (self.n_lobatto - 1) + np.arange(self.n_lobatto)
            full_x[local_ids] = local_x
            full_w[local_ids] += local_w

            D_x = self.ref_D / scale
            K_local = D_x.T @ np.diag(local_w) @ D_x
            for a, ia in enumerate(local_ids):
                for b, ib in enumerate(local_ids):
                    stiffness[ia, ib] += K_local[a, b]
                    derivative[ia, ib] += local_w[a] * D_x[a, b]

        if self.boundary == "dirichlet":
            active = np.arange(1, n_full - 1)
        else:
            active = np.arange(n_full)

        self.full_x = full_x
        self.full_w = full_w
        self.active = active
        self.x = full_x[active]
        self.w = full_w[active]
        self.npts = len(self.x)
        self.n = np.arange(self.npts)
        self.dx = float(np.mean(np.diff(self.x))) if self.npts > 1 else self.L
        self.k_max = None

        W_inv_sqrt = sp.diags(1.0 / np.sqrt(full_w[active]))
        K_active = stiffness.tocsr()[active][:, active]
        D_active = derivative.tocsr()[active][:, active]
        self._T_base = W_inv_sqrt @ K_active @ W_inv_sqrt
        self._D_base = W_inv_sqrt @ D_active @ W_inv_sqrt
        self._T_base = self._T_base.tocsr()
        self._D_base = self._D_base.tocsr()

    def t(self, hc=1.0, mc2=None, sparse=False):
        scale = 0.5 / self.mass
        if hc != 1.0 or mc2 is not None:
            scale *= hc**2
            if mc2 is not None:
                scale *= self.mass / mc2
        T = scale * self._T_base
        self.T = T
        return T if sparse else T.toarray()

    def kinetic_sparse(self, hc=1.0, mc2=None):
        return self.t(hc=hc, mc2=mc2, sparse=True)

    def momentum(self, sparse=False):
        P = -1j * self._D_base
        self.P = P
        return P if sparse else P.toarray()

    def h(self, V, sparse=False):
        values = V(self.x) if callable(V) else np.asarray(V, dtype=float)
        H = self.kinetic_sparse() + sp.diags(values)
        return H if sparse else H.toarray()

    def run(self, V=None, num_eigs=5):
        if V is None:
            if self.v is None:
                raise ValueError("Provide V or set self.v before run().")
            values = np.asarray(self.v, dtype=float)
        elif callable(V):
            values = np.asarray(V(self.x), dtype=float)
        else:
            values = np.asarray(V, dtype=float)
        if values.shape != (self.npts,):
            raise ValueError("Potential values must have shape (npts,).")

        H = self.kinetic_sparse() + sp.diags(values)
        if num_eigs >= self.npts:
            E, U = np.linalg.eigh(H.toarray())
        else:
            E, U = sla.eigsh(H, k=num_eigs, which="SA")
            order = np.argsort(E)
            E, U = E[order], U[:, order]
        self.eigvals = E
        self.eigvecs = U
        return E, U


class LegendreDVR(DVR):
    r"""Gauss-Legendre DVR on a finite interval.

    This DVR is useful for bounded angular coordinates.  The grid points are
    the Gauss-Legendre quadrature nodes mapped from ``[-1, 1]`` to
    ``[xmin, xmax]``.  Coordinate-dependent operators are diagonal on this
    grid.  The derivative/momentum operators are represented with the
    Lagrange-cardinal differentiation matrix in the quadrature-normalized DVR
    basis.
    """

    def __init__(self, xmin, xmax, npts, mass=1.0):
        if npts < 1:
            raise ValueError("npts must be positive.")
        if xmax <= xmin:
            raise ValueError("xmax must be larger than xmin.")

        self.npts = int(npts)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.L = self.xmax - self.xmin
        self._mass = float(mass)
        self.n = np.arange(self.npts)

        y, wy = np.polynomial.legendre.leggauss(self.npts)
        scale = 0.5 * self.L
        shift = 0.5 * (self.xmin + self.xmax)
        self.y = y
        self.x = shift + scale * y
        self.w = scale * wy
        self.dx = float(np.mean(self.w))
        self.k_max = None
        self.v = None
        self._D = None
        self._D2 = None
        self.T = None

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = float(value)

    def _differentiation_matrix(self):
        if self._D is not None:
            return self._D

        x = self.x
        n = self.npts
        bary = np.ones(n, dtype=float)
        for j in range(n):
            bary[j] = 1.0 / np.prod(x[j] - np.delete(x, j))

        D_lagrange = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    D_lagrange[i, j] = bary[j] / bary[i] / (x[i] - x[j])
            D_lagrange[i, i] = -np.sum(D_lagrange[i, np.arange(n) != i])

        root_w = np.sqrt(self.w)
        self._D = (root_w[:, None] / root_w[None, :]) * D_lagrange
        return self._D

    def t(self, hc=1.0, mc2=None):
        D = self._differentiation_matrix()
        D2 = D @ D
        self._D2 = D2
        T = -0.5 / self.mass * D2
        if hc != 1.0 or mc2 is not None:
            scale = hc ** 2
            if mc2 is not None:
                scale *= self.mass / mc2
            T = scale * T
        self.T = 0.5 * (T + T.conj().T)
        return self.T

    def momentum(self):
        D = self._differentiation_matrix()
        return -1j * D

    def f(self, x=None):
        if x is None:
            x = self.x
        x = np.asarray(x, dtype=float)
        basis = np.empty((x.size, self.npts), dtype=float)
        for j in range(self.npts):
            roots = np.delete(self.x, j)
            denom = np.prod(self.x[j] - roots)
            basis[:, j] = np.prod(x[:, None] - roots[None, :], axis=1) / denom / np.sqrt(self.w[j])
        return basis


class HermiteDVR(DVR):
    r"""Hermite function basis for non-periodic functions over an interval
    `-x_max ... x_max` with `npts` points.
    Usage:
        d = sincDVR1D(npts, xmax, [x0])

    @param[in] npts number of points
    @param[in] xmax "right" end of interval
    @param[in] x0 shifted center of interval
    @attribute a step size
    @attribute n vector of x-domain indices
    @attribute x discretized x-domain
    @attribute w quadrature weights
    @attribute k_max cutoff frequency
    @attribute L size of x-domain
    @method h return hamiltonian matrix
    @method f return DVR basis vectors
    """
    def __init__(self, npts, xmax=None, x0=0.):
        assert (npts < 269), \
            "Must make npts < 269 for numpy to find quadrature points."
        self.npts = npts
        self.x0 = float(x0)
        self.n = np.arange(npts)
        c = np.zeros(npts+1)
        c[-1] = 1.
        self.x = np.polynomial.hermite.hermroots(c)
        if xmax is None:
            self.gamma = 1.
        else:
            assert xmax is None, "Sorry, xmax is currently broken"
            self.gamma = self.x.max() / float(xmax)

        self.x = self.x0 + self.x / self.gamma
        self.w = np.exp(-np.square(self.x))
        self.L = self.x.max() - self.x.min()
        self.a = None
        self.k_max = None

    def t(self, hc=1., mc2=1.):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t(V)

        @returns T kinetic energy matrix
        """
        _i = self.n[:, np.newaxis]
        _j = self.n[np.newaxis, :]
        _xi = self.x[:, np.newaxis]
        _xj = self.x[np.newaxis, :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = 2.*(-1.)**(_i-_j)/(_xi-_xj)**2.

        T[self.n, self.n] = 0.
        T += np.diag((2. * self.npts + 1.
                      - np.square(self.x)) / 3.)
        T *= self.gamma
        T *= 0.5 * hc**2. / mc2   # (pc)^2 / (2 mc^2)
        return T

#     def f(self, x=None):
#         """Return the DVR basis vectors"""
#         if x is None:
#             x_m = self.x[:, np.newaxis]
#         else:
#             x_m = np.asarray(x)[:, np.newaxis]
#         x_n = self.x[np.newaxis, :]
#         return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)

class BesselDVR(DVR):
    r"""Bessel function basis for non-periodic functions over an interval
    `0 ... R` with `npts` points, `dim` dimensions, `lam` angular
    momentum number.
    Usage:
        d = BesselDVR(npts, R, dim, lam)

    @param[in] npts number of points
    @param[in] R max radius
    @param[in] dim dimension of the Bessel representation
    @param[in] lam angular momentum quantum number
    @attribute n vector of domain indices
    @attribute z discretized domain
    @attribute nu
    @attribute K
    @attribute r
    @method h return hamiltonian matrix
    @method f return DVR basis vectors
    """
    def __init__(self, npts, R, l=0, dim=2):
        assert type(dim) == int, "dim must be an integer."
        assert dim > 1, "dim must be 2 or more."
        self.N = self.npts = npts
        self.n = np.arange(npts)
        self.R = R
        self.dim = dim
        self.lam = self.l = l 
        self.__init_private()
        

    def __init_private(self):
        
        nu = self.lam + self.dim/2. - 1.
        # self.z = bessel.j_root(nu=self.nu, N=self.npts)
        self.z = scipy.special.jn_zeros(nu, self.npts)
        
        self.K = self.z[-1] / self.R
        self.x = self.z / self.K
        
    def nu(self, l):
        return l + self.dim/2.0 - 1
    
    def get_J(self, nu, d=0):
        r"""Return the `d`'th derivative of the bessel functions J_{\nu}(z)."""
        nu2 = 2*nu
        if 0 == d:
            def j(z):
                return scipy.special.jn(nu, z)
        else:
            # Compute derivatives using recurrence relations.  Not
            # efficient for high orders!
            def j(z):
                return (self.get_J(nu - 1, d - 1)(z) - self.get_J(nu + 1, d - 1)(z))/2.0
        return j
    
    def get_abscissa(self, l=0):
        nu = self.nu(l)
        zn = scipy.special.jn_zeros(nu,self.N) # Only works for even dim
        return zn/self.K

    def get_weights(self, l):
        """Return the integration weights"""
        nu = self.nu(l=l)
        rn = self.get_abscissa(l=l)
        n = np.arange(len(rn))
        zn = rn * self.K
        
        dJ = self.get_J(nu, d=1)
        
        return np.divide(2.0, self.K * zn * dJ(zn)**2)

    def t(self, l, hc=1., mc2=1.):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t(V)

        @returns T kinetic energy matrix
        """
        n = self.npts
        nu = self.nu(l)
        K = self.K

        _i = self.n[:, np.newaxis]
        _j = self.n[np.newaxis, :]
        _xi = self.z[:, np.newaxis]
        _xj = self.z[np.newaxis, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = 8. * K**2. * (-1.)**(_i-_j) * _xi * _xj /(_xi**2. - _xj**2.)**2.
        T[self.n, self.n] = 0.
        T += np.diag(K**2. / 3. * (1. + 2. * (nu**2. - 1.) / self.z**2.))
        T *= 0.5 * hc**2. / mc2   # (pc)^2 / (2 mc^2)
        return T
    
    def run(self, v, k=6):
        if callable(v):
            V = np.diag(v(self.x))
        else:
            V = np.diag(v)

        h = V + self.t(self.l)


        # Get the eigenpairs
        # There are multiple options here.
        # If the user is asking for all of the eigenvalues,
        # then we need to use np.linalg.eigh()
        if k == h.shape[0]:
            E, U = np.linalg.eigh(h)
        # But if we don't need all eigenvalues, only the smallest ones,
        # then when the size of the H matrix becomes large enough, it is
        # better to use sla.eigsh() with a shift-invert method. Here we
        # have to have a good guess for the smallest eigenvalue so we
        # ask for eigenvalues closest to the minimum of the potential.
        else:
            E, U = sla.eigsh(h, k=k, which='LM',
                             sigma=V.min())

        self.eigvals = E
        self.eigvecs = U
        # self.potential = V
        return E, U

    def get_F(self, n, l, normalize=False):
        """Return the DVR basis functions."""
        nu = self.nu(l=l)
        rn = self.get_abscissa(l=l)[n]
        zn = rn * self.k
        F_n = 1./np.sqrt(self.get_weights(l)[n]) if normalize else 1.0
        def F(r):
            z = self.k*r
            J = self.get_J(nu)
            return np.divide((-1)**(n+1)*self.k*zn*np.sqrt(2*r)*J(z),
                              (z**2 - zn**2)) / F_n
        return F

class LaguerreDVR(DVR):
    def __init__(self, N, alpha=0):
        """
        for radial coordinate x \in [0, \infty]
        
        .. math::
            
            w(x) = x^\alpha e^{-x}, x \ge 0 

        Parameters
        ----------
        N : TYPE
            DESCRIPTION.
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        
        pass

class ChebDVR(DVR):
    pass

# Factory functions to build different potentials:
# A factory is a function that makes a function.
class VFactory(object):
    """Factory functions to build different potentials
    A factory is a function that returns other functions.
    """
    def square_well(self, depth = 1., width = 1.,
                    origin = 0., o_val = 0.):
        """Usage:
                V = square_well_factory(**kwargs)

        Returns a function of a single variable V(x),
        representing the square-well potential:

             (-A/2, V0)            (A/2, V0)
        ------------       +       ----------------
                   |               |
                   |               |
                   |               |
                   |               |
         (-A/2, 0) |-------+-------| (A/2, 0)
                         (0, 0)

        Keyword arguments:
        @param[in] depth    Depth of the potential well (default=1)
        @param[in] width    Width of the potential well (default=1)
        @param[in] origin   Location of the well's center (default=0)
        @param[in] o_val    Value of the potential at origin (default=0)
        @returns   V        The square well potential function V(x)
        """
        def V(x):
            interior_idx = np.abs(x - origin) < width / 2.
            V = np.ones_like(x) * (depth + o_val)
            V[interior_idx] = o_val
            return V
        return V

    def double_well(self, x1 = -2., x2 = -1., x3 = 1.,
                    x4 = 2., V1 = 1., V2 = 0.,
                    V3 = 1., V4 = 0., V5 = 1.):
        """Usage:
                V = double_square_well_factory(**kwargs)

        Returns a one-dimensional potential function that represents
        a double-square-well potential. The potential looks like

           (x1, V1)      (x2, V3)   (x3, V3)      (x4, V5)
        ----------            ---------            ----------
                 |            |       |            |
                 |            |       |            |
                 |            |       |            |
                 |            |       |            |
                 |____________|       |____________|
           (x1, V2)      (x2, V2)   (x3, V4)      (x4, V4)

        Keywork arguments
        @param[in] x1    x-coordinate x1 above (default=-2)
        @param[in] x2    x-coordinate x2 above (default=-1)
        @param[in] x3    x-coordinate x3 above (default=1)
        @param[in] x4    x-coordinate x4 above (default=2)
        @param[in] V1    constant V1 above (default=1)
        @param[in] V2    constant V2 above (default=0)
        @param[in] V3    constant V3 above (default=1)
        @param[in] V4    constant V4 above (default=0)
        @param[in] V5    constant V5 above (default=1)
        @returns   V     double square-well potential V(x)
        """
        assert (x1 < x2 < x3 < x4), \
            "x-coordinates do not satisfy x1 < x2 < x3 < x4"
        def V(x):
            l_well_idx = np.logical_and(x < x2, x > x1)
            r_well_idx = np.logical_and(x < x4, x > x3)
            middle_idx = np.logical_and(x >= x2, x <= x3)
            far_rt_idx = np.greater_equal(x, x4)
            V = np.ones_like(x) * V1
            V[l_well_idx] = V2
            V[middle_idx] = V3
            V[r_well_idx] = V4
            V[far_rt_idx] = V5
            return V
        return V

    def sho(self, k = 1., x0 = 0.):
        """Usage:
                V = harmosc_factory(**kwargs)

        Return a one-dimensional harmonic oscillator potential V(x)
        with wavenumber k. i.e. V(x) = 1/2 * k * (x - x0)^2

        Keyword arguments
        @param[in] k    wavenumber of the SHO potential (default=1)
        @param[in] x0   displacement from origin (default=0)
        @returns   V    1-D SHO potential V(x)
        """
        def V(x): return 0.5 * k * np.square(x - x0)
        return V

    def power(self, a = 1., p=1., x0 = 0.):
        """Usage:
                V = self.power(**kwargs)

        Return a potential V(x) = a * (x - x0)^p

        Keyword arguments
        @param[in] a    coefficient (default=1)
        @param[in] p    power to raise x (default=1)
        @param[in] x0   displacement from origin (default=0)
        @returns   V    1-D cubic potential V(x)
        """
        def V(x): return a * np.power(x - x0, p)
        return V

    def morse(self, D = 1., a = 1., x0 = 0.):
        """Usage:
                V = morse_factory(**kwargs)

        Return a one-dimensional Morse potential V(x)
        i.e. V(x) = D * (1 - exp(-a * (x - x0)))^2 - D

        Keyword arguments
        @param[in] D    dissociation depth
        @param[in] a    inverse "width" of the potential
        @param[in] x0   equilibrium bond distance
        @returns   V    Morse potential V(x)
        """
        def V(x):
            return D * np.power(1. - np.exp(-a * (x - x0)), 2.) - D
        return V

    def sombrero(self, a = -10., b = 1.):
        """Usage:
                V = sombrero_factory(**kwargs)

        Return a one-dimensional version of the sombrero potential
        i.e. V(x) = a * x^2 + b * x^4
        This function asserts a < 0 and b > 0

        Keyword arguments
        @param[in] a    coefficient of the x^2 term (default=-10)
        @param[in] b    coefficient of the x^4 term (default=1)
        @returns   V    1-D Mexican hat potential V(x)
        """
        assert (a < 0), "Coefficient a must be negative"
        assert (b > 0), "Coefficient b must be positive"
        def V(x):
            return a * np.square(x) + b * np.power(x, 4)
        return V

    def woods_saxon(self, V0 = 50., z = 0.5, r0 = 1.2, A = 16):
        """Usage:
                V = woods_saxon_factory(**kwargs)

        Return a Woods-Saxon potential
        i.e. V(r) = - V0 / (1. + exp((r - R) / z))
        where R = r0 * A^(1/3)

        Keyword arguments
        @param[in] V0   potential depth (default=50.)
        @param[in] z    surface thickness (default=0.5)
        @param[in] r0   rms nuclear radius (default=1.2)
        @param[in] A    mass number (default=16)
        @returns   V    Woods-Saxon potential V(r)
        """
        def V(r):
            x0 = r0 * np.power(A, 1. / 3.)
            return -V0 / (1. + np.exp((r - x0)/ z))
        return V


if __name__ == '__main__':
    import time
    # x = np.linspace(-7, 7, 200)
    def v(x):
        return x**2/2

    def test_sincdvr():
        # dvr = SincDVR(npts=20, L=10)

        dvr = HermiteDVR(npts=10)
        x = dvr.x


        w, u = dvr.run(v, num_eigs=10)
        dvr.draw_states()

    def test_expdvr():

        dvr = ExponentialDVR(n=6, L=10)
        E, U = dvr.run(v, k=5)

        print(E)

    def test_sineDVR():
        dvr = SineDVR(-5, 5, 20)
        E, U = dvr.run(v, k=5)

        print(E)


    # test_sineDVR()


    dvr_z = SineDVR(-4, 4, npts=16)

    z = dvr_z.x
    print(z)
    
    E, U = dvr_z.run(v, 6)
    print(E)
    # x = np.linspace(-6,6,200)

    # fig, ax = plt.subplots()
    # ax.plot(x, dvr.basis(x, 32))

    
    for N in [128, 256]:
        dvr = BesselDVR(npts=N, R=8, l=0)
        r = dvr.x
        # print(dvr.get_weights(l=0))
        # def v(x):
        #     return -1/np.sqrt(x**2 + 0.00001) 
        # for n in range(dvr_z.npts):
        a = 0.05
        # _v = -1/np.sqrt(r**2 + a**2)
        _v = r**2/2
        E, U = dvr.run(_v, k=1)
    
        print(E)
    
    fig, ax = plt.subplots()
    ax.plot(r, -U[:,0], '-o')
    ax.plot(r, np.sqrt(r)*np.exp(-r))

    # U = dvr.fbr2dvr()

    # # print((np.subtract.outer(dvr.n, dvr.n) % 2))

    # start = time.time()
    # T = dvr.t()
    # print(T)

    # print(U.conj().T @ np.diagflat(dvr.t_fbr()) @ U)
    # print(kinetic(dvr.x, dvr='sine'))

    # # p = dvr.momentum()
    # # print(p)
    # # scipy.linalg.expm(-1j * T * 0.2)

    # time1 = time.time()
    # # print('expm', time1 - start)

    # dvr.expT(0.2)

    # time2 = time.time()
    # print(time2 - time1)

    # print(np.sin(1 * 3 * np.pi/4) * np.sqrt(2/4 ))
