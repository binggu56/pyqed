from __future__ import absolute_import

import numpy as np
from numpy import exp, pi, sqrt
from scipy.sparse import csr_matrix, lil_matrix, identity, kron, linalg,\
                        spdiags, issparse
from scipy.special import hermite, jv
from math import factorial
import scipy as sp
# import numba
# from numba import jit
import sys

def jacobi_anger(n, z=1):
    """
    Jacobi-Anger expansion
    .. math::
        e^{iz \cos(\theta)} = \sum_{n=-\infty}^\infty i^n J_n(z) e^{in\theta}

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    z : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    return 1j**n * jv(n, z)


def is_positive_def(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def polarization_vector(pol='x'):
    """
    unit length polarization vector

    Parameters
    ----------
    d : TYPE, optional
        DESCRIPTION. The default is 'x'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if pol == 'x':
        return np.array([1., 0., 0.])
    elif pol == 'y':
        return np.array([0., 1., 0.])
    elif pol == 'lcp':
        return 1./np.sqrt(2.) * np.array([1., -1j, 0])
    elif pol == 'rcp':
        return 1./np.sqrt(2.) * np.array([1., 1j, 0])
    else:
        raise ValueError('There is no {} polarization'.format(pol))


def spin_ops(m):
    """
    Spin operators represented by Sz eigenstates

    Parameters
    ----------
    m : int
        spin multiplicity :math:`m = 2S + 1`

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if m == 1:
        return 0.5 * pauli()
    elif m == 3:
        Sz = np.zeros((m,m))
        Sz[0, 0] = 1; Sz[2, 2] = -1
        return csr_matrix(Sz)
    else:
        raise NotImplementedError('Spin operators with multiplicity {} \
                                  has not been implemented yet'.format(m))


def rotate(angle):
    return np.array()

class HarmonicOscillator:
    """
    basic class for harmonic oscillator
    """
    def __init__(self, omega, mass=1, x0=0):
        self.mass = mass
        self.omega = omega
        self.x0 = 0

    def eigenstate(self, x, n=0):
        x = x - self.x0
        alpha = (self.mass * self.omega)
        phi = 1/sqrt(2**n * factorial(n)) * (alpha/np.pi)**(1/4) * np.exp(-alpha * x**2/2.) * \
            hermite(n=n)(np.sqrt(alpha) * x)
        return phi

    def potential(self, x):
        return 1/2 * self.omega * (x-self.x0)**2



class Morse:
    """
    basic class for Morse oscillator
    """
    def __init__(self, D, a, re, mass=1):
        self.D = D
        self.a = a
        self.re = re
        self.mass = mass
        self.omega = a * sqrt(2.*D/mass)

    def eigval(self, n):
        """

        Given an (integer) quantum number v, return the vibrational energy.

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        D = self.D

        return (n+0.5)* self.omega - (self.omega * (n+0.5))**2/(4. * D)

    def eigenstate(self, x, n=0):

        from scipy.special import genlaguerre, gamma
        from math import factorial

        l = sqrt(2.*self.mass * self.D)/self.a
        alpha = 2*l - 2*n - 1

        z = 2*l*exp(-(x-self.re))

        # normalization coeff
        C = sqrt(factorial(n) * alpha /gamma(2*l - n))

        psi = C * z**(alpha/2.) * exp(-0.5 * z) * genlaguerre(n, alpha)(z)

        return psi


    def potential(self, x):
        return morse(x, self.D, self.a, self.re)



def morse(r, D, a, re):
    """
    Morse potential
        D * (1. - e^{-a * (r - r_e)})**2

    Parameters
    ----------
    r : float/1darray
        DESCRIPTION.
    D : float
        well depth
    a : TYPE
        'width' of the potential.
    re : TYPE
        equilibrium bond distance

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return D * (1. - exp(-a * (r - re)))**2


def gwp2(x, y, sigma=np.identity(2), xc=[0, 0], kc=[0, 0]):
    """
    generate a 2D Gaussian wavepacket in grid
    :param x0: float, mean value of gaussian wavepacket along x
    :param y0: float, mean value of gaussian wavepacket along y
    :param sigma: float array, covariance matrix with 2X2 dimension
    :param kx0: float, initial momentum along x
    :param ky0: float, initial momentum along y
    :return: float 2darray, the gaussian distribution in 2D grid
    """
    gauss_2d = np.zeros((len(x), len(y)), dtype=np.complex128)
    x0, y0 = xc
    kx, ky = kc

    A = inv(sigma)

    delta = A[0, 0] * (x-x0)**2 + A[1, 1] * (y-y0)**2 + 2. * A[0, 1]*(x-x0)*(y-y0)

    gauss_2d = (sqrt(det(sigma)) * sqrt(pi) ** 2) ** (-0.5) \
                              * exp(-0.5 * delta + 1j * kx * (x-x0) + 1j * ky * (y-y0))

    return gauss_2d


def meshgrid(*args):
    """
    fix the indexing of the Numpy meshgrid

    Parameters
    ----------
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.meshgrid(*args, indexing='ij')

def jump(f, i, dim=2, isherm=True):

    A = lil_matrix((dim, dim))

    if i == f:
        A[i, i] = 1.
    else:
        if isherm:
            A[f, i] = A[i, f] = 1.
        else:
            A[f, i] = 1.

    return A.tocsr()

def eig_asymm(h):
    '''Diagonalize a real, *asymmetrix* matrix and return sorted results.

    Return the eigenvalues and eigenvectors (column matrix)
    sorted from lowest to highest eigenvalue.
    '''
    e, c = np.linalg.eig(h)
    if np.allclose(e.imag, 0*e.imag):
        e = np.real(e)
    else:
        print("WARNING: Eigenvalues are complex, will be returned as such.")

    idx = e.argsort()
    e = e[idx]
    c = c[:,idx]

    return e, c


def is_positive_def(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def sort(eigvals, eigvecs):
    """
    sort eigenvalues and eigenvectors from low to high

    Parameters
    ----------
    eigvals : TYPE
        DESCRIPTION.
    eigvecs : TYPE
        DESCRIPTION.

    Returns
    -------
    eigvals : TYPE
        DESCRIPTION.
    eigvecs : TYPE
        DESCRIPTION.

    """
    idx = np.argsort(eigvals)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    return eigvals, eigvecs

def coh_op(j, i, d):
    """
    return a matrix representing the coherence :math:`|j\rangle \langle i|`

    Parameters
    ----------
    j : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    a = lil_matrix((d,d), dtype=float)
    a[j, i] = 1.
    return a.tocsr()


def rect(x):
    return np.heaviside(x+0.5, 0.5) - np.heaviside(x - 0.5, 0.5)

def interval(x):
    return x[1] - x[0]

def fftfreq(times):
    """
    get the spectral range corresponding to the temporal grid (a.u.)

    Parameters
    ----------
    times : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.fft.fftshift(np.fft.fftfreq(len(times), interval(times)))

def tensor(*args):
    """Calculates the tensor product of input operators.

    Build from QuTip.

    Parameters
    ----------
    args : array_like
        ``list`` or ``array`` of quantum objects for tensor product.

    Returns
    -------


    """

    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        # this is the case when tensor is called on the form:
        # tensor([q1, q2, q3, ...])
        qlist = args[0]

    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        qlist = args

    for n, q in enumerate(qlist):
        if n == 0:
            out = q
        else:
            out = sp.kron(out, q, format='csr')

    return out


def ptrace(rho, dims, which='B'):
    """
    partial trace of subsystems in a density matrix defined in a composite space

    Parameters
    ----------
    rho : ndarray
        DESCRIPTION.
    which : TYPE, optional
        DESCRIPTION. The default is 'B'.

    Returns
    -------
    rhoA : TYPE
        DESCRIPTION.

    """
    dimA, dimB = dims

    if rho.shape[0] != dimA * dimB:
        raise ValueError('Size of density matrix does not match dimensions.')

    if issparse(rho):
        rho = rho.toarray()

    rho_reshaped = np.reshape(rho, (dimA, dimB, dimA, dimB))

    if which=='B':
        rhoA = np.einsum('injn -> ij', rho_reshaped)
        return rhoA

    elif which == 'A':

        rhoB = np.einsum('inim -> nm', rho_reshaped)

        return rhoB
    else:
        raise ValueError('which can only be A or B.')


def tracedist(A, B):
    """
    Calculates the trace distance between two density matrices..
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Parameters
    ----------!=
    A : 2D array (N,N)
        Density matrix or state vector.
    B : 2D array (N,N)
        Density matrix or state vector with same dimensions as A.
    tol : float
        Tolerance used by sparse eigensolver, if used. (0=Machine precision)
    sparse : {False, True}
        Use sparse eigensolver.

    Returns
    -------
    tracedist : float
        Trace distance between A and B.

    Examples
    --------
    >>> x=fock_dm(5,3)
    >>> y=coherent_dm(5,1)
    >>> tracedist(x,y)
    0.9705143161472971

    """

    if A.dims != B.dims:
        raise TypeError("A and B do not have same dimensions.")

    diff = A - B
    diff = diff.dag() * diff
    vals = sp.linalg.eigs(diff)
    return float(np.real(0.5 * np.sum(np.sqrt(np.abs(vals)))))


def hilbert_dist(A, B):
    """
    Returns the Hilbert-Schmidt distance between two density matrices A & B.

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    dist : float
        Hilbert-Schmidt distance between density matrices.

    Notes
    -----
    See V. Vedral and M. B. Plenio, Phys. Rev. A 57, 1619 (1998).

    """

    if A.shape != B.shape:
        raise TypeError('A and B do not have same dimensions.')

    return ((A - B)**2).tr()

def lowering(dims=2):
    if dims == 2:
        sm = csr_matrix(np.array([[0.0, 1.0],[0.0,0.0]], dtype=np.complex128))
    else:
        raise ValueError('dims can only be 2.')
    return sm


def raising(dims=2):
    """
    raising operator for spin-1/2
    Parameters
    ----------
    dims: integer
        Hilbert space dimension

    Returns
    -------
    sp: 2x2 array
        raising operator
    """
    if dims == 2:
        sp = csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128))
    else:
        raise ValueError('dims can only be 2.')
    return sp


def sinc(x):
    '''
    .. math::
    sinc(x) = \frac{\sin(x)}{x}

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.sinc(x/np.pi)

def norm2(f, dx=1, dy=1):
    '''
    L2 norm of the 2D array f

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    dy : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.trace(dag(f).dot(f))*dx*dy

def get_index(array, value):
    '''
    get the index of element in array closest to value
    '''
    if value < np.min(array) or value > np.max(array):
        print('Warning: the value is out of the range of the array!')

    return np.argmin(np.abs(array-value))


def rgwp(x, x0=0., sigma=1.):
    '''
    real Gaussian wavepacket

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    x0 : float
        central position
    sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    psi = 1./np.sqrt(np.sqrt(np.pi) * sigma) * np.exp(-(x-x0)**2/2./sigma**2)
    return psi

def gwp(x, sigma=1., x0=0., p0=0.):
    '''
    complex Gaussian wavepacket

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    sigma : TYPE, optional
        DESCRIPTION. The default is 1..
    x0 : TYPE, optional
        DESCRIPTION. The default is 0..
    p0 : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    psi : TYPE
        DESCRIPTION.

    '''
    psi = np.sqrt(np.sqrt(1./np.pi/sigma**2)) * \
        np.exp(-(x-x0)**2/2./sigma**2 + 1j * p0 * (x-x0))
    return psi

def gwp_k(k, sigma, x0,k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    a = 1./sigma**2

    return ((a / np.sqrt(np.pi))**0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))

def thermal_dm(n, u):
    """
    return the thermal density matrix for a boson
    n: integer
        dimension of the Fock space
    u: float
        reduced temperature, omega/k_B T
    """
    nlist = np.arange(n)
    diags = exp(- nlist * u)
    diags /= np.sum(diags)
    rho = lil_matrix(n)
    rho.setdiag(diags)
    return rho.tocsr()

def liouvillian(rho, H, c_ops):
    """
    lindblad quantum master eqution
    """
    rhs = -1j * comm(H, rho)
    for c_op in c_ops:
        rhs += lindbladian(c_op, rho)
    return rhs

def lindbladian(l, rho):
    """
    lindblad superoperator: l rho l^\dag - 1/2 * {l^\dag l, rho}
    l is the operator corresponding to the disired physical process
    e.g. l = a, for the cavity decay and
    l = sm for polarization decay
    """
    return l.dot(rho.dot(dag(l))) - 0.5 * anticomm(dag(l).dot(l), rho)

def ket2dm(psi):
    """
    convert a ket into a density matrix

    Parameters
    ----------
    psi : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.einsum("i, j -> ij", psi, psi.conj())

def norm(psi, dx=1):
    '''
    normalization of the wavefunction

    N = \int dx \psi^*(x) \psi(x)

    Parameters
    ----------
    psi : 1d array, complex
        DESCRIPTION.

    Returns
    -------
    float, L2 norm

    '''
    return dag(psi).dot(psi).real * dx


def destroy(N):
    """
    Annihilation operator for bosons.

    Parameters
    ----------
    N : int
        Size of Hilbert space.

    Returns
    -------
    2d array complex

    """

    a = lil_matrix((N, N))
    a.setdiag(np.sqrt(np.arange(1, N)), 1)

    return a.tocsr()


def rk4(rho, fun, dt, *args):
    """
    Runge-Kutta method
    """
    dt2 = dt/2.0

    k1 = fun(rho, *args )
    k2 = fun(rho + k1*dt2, *args)
    k3 = fun(rho + k2*dt2, *args)
    k4 = fun(rho + k3*dt, *args)

    rho += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

    return rho

def fermi(E, Ef = 0.0, T = 1e-4):
    """
    Fermi-Dirac distribution function
    INPUT:
        E : Energy
        Ef : Fermi energy
        T : temperture (in units of energy, i.e., kT)
    OUTPUT:
        f(E): Fermi-Dirac distribution function at energy E

    """
#    if E > Ef:
#        return 0.0
#    else:
#        return 1.0
    #else:
    return 1./(1. + np.exp((E-Ef)/T))

def lorentzian(x, width=1.):
    '''


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    x0 : float
        center of the Lorentzian

    width : float
        Half-wdith half-maximum

    Returns
    -------
    None.

    '''
    return 1./np.pi * width/(width**2 + (x)**2)


def gaussian(x, sigma=1.):
    '''
    normalized Gaussian distribution

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return 1/sigma/sqrt(2.*pi) * exp(-x**2/2./sigma**2)




def transform(A, v):
    """
    transformation rule: A_{ab} = <a|i><i|A|j><j|b> = Anew = v^\dag A v
    input:
        A: matrix of operator A in old basis
        v: basis transformation matrix
    output:
        Anew: matrix A in the new basis
    """
    Anew = dag(v).dot(A.dot(v))
    #Anew = csr_matrix(A)

    return Anew

def basis_transform(A, v):
    """
    transformation rule: A_{ab} = <a|i><i|A|j><j|b> = Anew = v^\dag A v
    input:
        A: matrix of operator A in old basis
        v: basis transformation matrix
    output:
        Anew: matrix A in the new basis
    """
    Anew = dag(v).dot(A.dot(v))

    return csr_matrix(Anew)

def heaviside(x):
    return 0.5 * (np.sign(x) + 1)

def commutator(A,B):
    assert(A.shape == B.shape)
    return A.dot(B) - B.dot(A)

# @jit
def comm(A,B):
    assert(A.shape == B.shape)
    return np.dot(A, B) - np.dot(B, A)

# @jit
def anticomm(A,B):
    assert(A.shape == B.shape)
    return np.dot(A, B) + np.dot(B, A)

def anticommutator(A,B):
    assert(A.shape == B.shape)
    return A.dot(B) + B.dot(A)

def dagger(a):
    return a.conjugate().transpose()

# @jit
def dag(a):
    return a.conjugate().transpose()

def coth(x):
    return 1./np.tanh(x)

def sigmaz():
     return np.array([[1.0,0.0],[0.0,-1.0]], dtype=np.complex128)

def sigmax():
    return np.array([[0.0,1.0],[1.0,0.0]], dtype=np.complex128)

def sigmay():
    return np.array([[0.0,-1j],[1j,0.0]], dtype=np.complex128)

def pauli():
    # spin-half matrices
    sz = np.array([[1.0,0.0],[0.0,-1.0]])

    sx = np.array([[0.0,1.0],[1.0,0.0]])

    sy = np.array([[0.0,-1j],[1j,0.0]])

    s0 = np.identity(2)

    for _ in [s0, sx, sy, sz]:
        _ = csr_matrix(_)

    return s0, sx, sy, sz


def ham_ho(freq, n, ZPE=False):
    """
    Hamiltonian for harmonic oscilator

    input:
        freq: fundemental frequency in units of Energy
        n : size of matrix
        ZPE: boolean, if ZPE is included in the Hamiltonian
    output:
        h: hamiltonian of the harmonic oscilator
    """
    if ZPE:
        h = lil_matrix((n,n))
        h = h.setdiag((np.arange(n) + 0.5) * freq)
    else:
        h = lil_matrix((n, n)).setdiag(np.arange(n) * freq)

    return h

def boson(omega, n, ZPE=False):
    if ZPE:
        h = lil_matrix((n,n))
        h.setdiag((np.arange(n) + 0.5) * omega)
    else:
        h = lil_matrix((n, n))
        h.setdiag(np.arange(n) * omega)
    return h

def quadrature(n):
    """
    Quadrature operator of a photon mode
    X = (a + a+)/sqrt{2}

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    a = destroy(n)
    return 1./np.sqrt(2) * (a + dag(a))

# @jit
def obs_dm(rho, d):
    """
    observables for operator d
    """

    dAve = d.dot(rho).diagonal().sum()

    return dAve

def obs(psi, a):
    """


    Parameters
    ----------
    psi : 1d array
        wavefunction.
    a : 2d array
        operator a.

    Returns
    -------
    complex
        Expectation of operator a.

    """
    return dag(psi) @ a @ psi

def resolvent(omega, Ulist, dt):
    """
    compute the resolvent 1/(omega - H) from the Fourier transform of the propagator
    omega: float
        frequency to evaluate the resolvent
    Ulist: list of matrices
        propagators
    dt: time-step used in the computation of U
    """
    N = len(Ulist)
    t = np.array(np.arange(N) * dt)
    return sum(np.exp(1j * omega * t) * Ulist)


def basis(N, j):
    """
    Parameters
    ----------
    N: int
        Size of Hilbert space for a multi-level system.
    j: int
        The j-th basis function.

    Returns
    -------
    1d complex array
        j-th basis function for the Hilbert space.
    """
    b = np.zeros(N, dtype=complex)
    if N < j:
        sys.exit('Increase the size of the Hilbert space.')
    else:
        b[j] = 1.0

    return b


def tdse(wf, h):
    return -1j * h @ wf

def quantum_dynamics(ham, psi0, dt=0.001, Nt=1, obs_ops=None, nout=1,\
                    t0=0.0, output='obs.dat'):
    '''
    Laser-driven dynamics in the presence of laser pulses

    Parameters
    ----------
    ham : 2d array
        Hamiltonian of the molecule
    psi0: 1d array
        initial wavefunction
    dt : float
        time step.
    Nt : int
        timesteps.
    obs_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    '''

    # initialize the density matrix
    #wf = csr_matrix(wf0).transpose()
    psi = psi0

    #nstates = len(psi0)

    #f = open(fname,'w')
    if obs_ops is not None:
        fmt = '{} '* (len(obs_ops) + 1)  + '\n'
    #fmt_dm = '{} '* (nstates + 1)  + '\n'

    #f_dm = open('psi.dat', 'w') # wavefunction
    f_obs = open(output, 'w') # observables

    t = t0

    #f_dip = open('dipole'+'{:f}'.format(pulse.delay * au2fs)+'.dat', 'w')

    for k1 in range(int(Nt/nout)):

        for k2 in range(nout):
            psi = rk4(psi, tdse, dt, ham)

        t += dt * nout

        # compute observables
        Aave = np.zeros(len(obs_ops), dtype=complex)

        for j, A in enumerate(obs_ops):
            Aave[j] = obs(psi, A)

        #print(Aave)

#        f_dm.write(fmt_dm.format(t, *psi))
        f_obs.write(fmt.format(t, *Aave))

    np.savez('psi', psi)
    #f_dm.close()
    f_obs.close()

    return

def driven_dynamics(ham, dip, psi0, pulse, dt=0.001, Nt=1, obs_ops=None, nout=1,\
                    t0=0.0):
    '''
    Laser-driven dynamics in the presence of laser pulses

    Parameters
    ----------
    ham : 2d array
        Hamiltonian of the molecule
    dip : TYPE
        transition dipole moment
    psi0: 1d array
        initial wavefunction
    pulse : TYPE
        laser pulse
    dt : TYPE
        time step.
    Nt : TYPE
        timesteps.
    obs_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    '''

    # initialize the density matrix
    #wf = csr_matrix(wf0).transpose()
    psi = psi0

    nstates = len(psi0)

    #f = open(fname,'w')
    fmt = '{} '* (len(obs_ops) + 1)  + '\n'
    fmt_dm = '{} '* (nstates + 1)  + '\n'

    f_dm = open('psi.dat', 'w') # wavefunction
    f_obs = open('obs.dat', 'w') # observables

    t = t0

    #f_dip = open('dipole'+'{:f}'.format(pulse.delay * au2fs)+'.dat', 'w')

    for k1 in range(int(Nt/nout)):

        for k2 in range(nout):

            ht = pulse.field(t) * dip + ham
            psi = rk4(psi, tdse, dt, ht)

        t += dt * nout

        # compute observables
        Aave = np.zeros(len(obs_ops), dtype=complex)
        for j, A in enumerate(obs_ops):
            Aave[j] = obs(psi, A)

        #print(Aave)

        f_dm.write(fmt_dm.format(t, *psi))
        f_obs.write(fmt.format(t, *Aave))

    f_dm.close()
    f_obs.close()

    return

def driven_dissipative_dynamics(ham, dip, rho0, pulse, dt=0.001, Nt=1, \
                                obs_ops=None, nout=1):
    '''
    Laser-driven dynamics in the presence of laser pulses

    Parameters
    ----------
    ham : 2d array
        Hamiltonian of the molecule
    dip : TYPE
        transition dipole moment
    rho0: 2d array complex
        initial density matrix
    pulse : TYPE
        laser pulse
    dt : float
        DESCRIPTION.
    Nt : TYPE
        DESCRIPTION.
    obs_ops: list
        observable operators to compute
    nout: int
        Store data every nout steps

    Returns
    -------
    None.

    '''
    return


####################
# spin chains
####################
def multi_spin(onsite, nsites):
    """
    construct the hamiltonian for a multi-spin system
    params:
        onsite: array, transition energy for each spin
        nsites: number of spins
    """

    s0, sx, sy, sz = pauli()
    sm = lowering()

    head = onsite[0] * kron(sz, tensor_power(s0, nsites-1))
    tail = onsite[-1] * kron(tensor_power(s0, nsites-1), sz)
    ham = head + tail

    for i in range(1, nsites-1):
        ham += onsite[i] * kron(tensor_power(s0, i), kron(sz, tensor_power(s0, nsites-i-1)))

    lower_head = kron(sm, tensor_power(s0, nsites-1))
    lower_tail = kron(tensor_power(s0, nsites-1), sm)
    lower = lower_head + lower_tail

    for i in range(1, nsites-1):
        lower += kron(tensor_power(s0, i), kron(sm, tensor_power(s0, nsites-i-1)))



    return ham, lower

def multiboson(omega, nmodes, J=0, truncate=2):
    """
    construct the hamiltonian for a multi-spin system

    Parameters
    ----------
    omegas : 1D array
        resonance frequenies of the boson modes
    nmodes : integer
        number of boson modes
    J : float
        hopping constant
    truncation : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    ham : TYPE
        DESCRIPTION.
    lower : TYPE
        DESCRIPTION.

    """

    N = truncate
    h0 = boson(omega, N)
    idm = identity(N)
    a = destroy(N)
    adag = dag(a)
    x = a  + adag


    if nmodes == 1:

        return h0

    elif nmodes == 2:

        ham = kron(idm, h0) + kron(h0, idm) + J * kron(x, x)
        return ham

    elif nmodes > 2:

        head = kron(h0, tensor_power(idm, nmodes-1))
        tail = kron(tensor_power(idm, nmodes-1), h0)
        ham = head + tail

        for i in range(1, nmodes-1):
             ham += kron(tensor_power(idm, i), \
                                     kron(h0, tensor_power(idm, nmodes-i-1)))

        hop_head = J * kron(kron(x, x), tensor_power(idm, nmodes-2))
        hop_tail = J * kron(tensor_power(idm, nmodes-2), kron(x, x))

        ham += hop_head + hop_tail

        for i in range(1, nmodes-2):
            ham += J * kron(tensor_power(idm, i), \
                                kron(kron(x, x), tensor_power(idm, nmodes-i-2)))

        # connect the last mode to the first mode

    # lower_head = kron(a, tensor_power(idm, nmodes-1))
    # lower_tail = kron(tensor_power(idm, nmodes-1), a)
    # lower = lower_head + lower_tail

    # for i in range(1, nmodes-1):
    #     lower += kron(tensor_power(idm, i), kron(a, tensor_power(idm, nmodes-i-1)))


        return ham


def multimode(omegas, nmodes, J=0, truncate=2):
    """
    construct the direct tensor-product Hamiltonian for a multi-mode system

    Parameters
    ----------
    omegas : 1D array
        resonance frequenies of the boson modes
    nmodes : integer
        number of boson modes
    J : float
        nearest-neighbour hopping constant
    truncate : list
        size of Fock space for each mode

    Returns
    -------
    ham : TYPE
        DESCRIPTION.
    xs : list
        position operators in the composite space for each mode

    """

    N = truncate
    h0 = boson(omegas[0], N)
    idm = identity(N)
    x = quadrature(N)

    assert len(omegas) == nmodes

    if nmodes == 1:

        return h0, x

    elif nmodes == 2:

        hf = boson(omegas[-1], N)
        ham = kron(idm, hf) + kron(h0, idm) + J * kron(x, x)

        xs = [kron(x, idm), kron(idm, x)]
        return ham, xs

    elif nmodes > 2:
        h0 = boson(omegas[0], N)
        hf = boson(omegas[-1], N)

        head = kron(h0, tensor_power(idm, nmodes-1))
        tail = kron(tensor_power(idm, nmodes-1), hf)
        ham = head + tail

        for i in range(1, nmodes-1):
            h = boson(omegas[i], N)
            ham += kron(tensor_power(idm, i), \
                                     kron(h, tensor_power(idm, nmodes-i-1)))

        hop_head = J * kron(kron(x, x), tensor_power(idm, nmodes-2))
        hop_tail = J * kron(tensor_power(idm, nmodes-2), kron(x, x))

        ham += hop_head + hop_tail

        for i in range(1, nmodes-2):
            ham += J * kron(tensor_power(idm, i), \
                                kron(kron(x, x), tensor_power(idm, nmodes-i-2)))

        # connect the last mode to the first mode

        lower_head = kron(x, tensor_power(idm, nmodes-1))
        xs = []
        xs.append(lower_head)

        for i in range(1, nmodes-1):
            # x = quadrature(dims[i])
            lower = kron(tensor_power(idm, i), kron(x, tensor_power(idm, nmodes-i-1)))
            xs.append(lower.copy())

        lower_tail = kron(tensor_power(idm, nmodes-1), x)
        xs.append(lower_tail)

        return ham, xs

def project(P, a):
    """
    reduce the representation of operators to a subspace defined by the
    projection operator P

    Parameters
    ----------
    P : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

def tensor_power(a, n:int):
    """
    kron(a, kron(a, ...))
    """
    if n == 1:
        return csr_matrix(a)
    else:
        tmp = a
        for i in range(n-1):
            tmp = kron(tmp, a)

        return tmp

#def exact_diagonalization(nsites):
#    """
#    exact-diagonalization of the Dicke model
#    """
#    nsites = 10
#    omega0 = 2.0
#
#    g = 0.1
#
#    onsite = np.random.randn(nsites) * 0.08 + omega0
#    print('onsite energies', onsite)
#
#    ham, dip = multi_spin(onsite, nsites)
#
#    cav = Cavity(n_cav=2, freq = omega0)
#    print(cav.get_ham())
#
#    mol = Mol(ham, dip=dip)
#
#    pol = Polariton(mol, cav, g)
#
#
#
#    # spectrum of the sytem
#
#    #fig, ax = plt.subplots()
#    #w = linalg.eigsh(ham, nsites+1, which='SA')[0]
#    #for i in range(1, len(w)):
#    #    ax.axvline(w[i] - w[0])
#
#
#    # polaritonic spectrum
#    fig, ax = plt.subplots()
#
#    set_style()
#
#    nstates = nsites + 2
#    w, v = pol.spectrum(nstates)
#
#    num_op = cav.get_num()
#    print(num_op)
#    num_op = kron(identity(ham.shape[0]), num_op)
#
#    n_ph = np.zeros(nstates)
#    for j in range(nstates):
#        n_ph[j] = v[:,j].conj().dot(num_op.dot(v[:,j]))
#
#    print(n_ph)
#
#    for i in range(1, len(w)):
#        ax.axvline(w[i] - w[0], ymin=0 , ymax = n_ph[i], lw=3)
#
#    #ax.set_xlim(1,3)
#    ax.axvline(omega0 + g * np.sqrt(nsites), c='r', lw=1)
#    ax.axvline(omega0 - g * np.sqrt(nsites), color='r', lw=1)
#    ax.set_ylim(0, 0.5)
#
#    fig.savefig('polariton_spectrum.eps', transparent=True)

def expm(A, t, method='EOM'):
    """
    exponentiate a matrix at t
        U(t) = e^{A t}

    Parameters
    -----------
    A : TYPE
        DESCRIPTION.

    t: float or list
        times

    method : TYPE, optional
        DESCRIPTION. The default is 'EOM'.

        EOM: equation of motion approach.
            d/dt U(t) = A U(t)
            This can be generalized for time-dependent Hamiltonians A(t)

        diagonalization: diagonalize A

            for Hermitian matrices only, this is prefered


    Returns
    -------
    Ulist : TYPE
        DESCRIPTION.

    """

    if method == 'EOM':

        # identity matrix at t = 0
        U = identity(A.shape[-1], dtype=complex)

        # set the ground state energy to 0
        print('Computing the propagator. '
              'Please make sure that the ground-state energy is 0.')

        Ulist = []
        Nt = len(t)
        dt = interval(t)

        for k in range(Nt):
            Ulist.append(U.copy())
            U = rk4(U, ldo, dt, A)

        return Ulist

    elif method == 'SOS':

        raise NotImplementedError('Method of {} has not been implemented.\
                                  Choose from EOM'.format(method))


def ldo(b, A):
    '''
    linear differential operator Ab

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return A.dot(b)


def isherm(a):
    return np.allclose(a, dag(a))

def isdiag(M):
    """
    Check if a matrix is diagonal.

    Parameters
    ----------
    M : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.all(M == np.diag(np.diagonal(M)))


def pdf_normal(x, mu=0, sigma=1.):
    return 1. / (sigma * np.sqrt(2 * np.pi)) * \
            np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

if __name__ == '__main__':
    a = sigmaz() - 1j * sigmax()
    print(isherm(a))

    import matplotlib.pyplot as plt
    x = np.linspace(-1, 1)
    plt.plot(x, pdf_normal(x))
    plt.show()