'''
modules for open quantum systems



@author: Bing Gu
@email: bingg@uci.edu

Credits:
    QuTip
'''

import numpy as np
from numpy import tensordot
from numba import jit, njit
import sys
import scipy
from scipy.linalg import eigh, eig
from scipy.sparse import csr_matrix, issparse, identity
import scipy.sparse.linalg as la
from scipy.integrate import solve_ivp
# from scipy.sparse.linalg import eigs
import opt_einsum as oe

from lime.phys import anticommutator, comm, commutator, anticomm, dag, ket2dm, \
    obs_dm, destroy, rk4, basis, transform, isherm, expm

# from lime.superoperator import lindblad_dissipator
from lime.superoperator import op2sop, dm2vec, obs, left, right, operator_to_superoperator
from lime.liouville import sort
from lime.units import au2fs, au2k, au2wavenumber
from lime.mol import Mol, Result



import lime.superoperator as superop



class Redfield_solver:
    def __init__(self, H, c_ops=None, spectra=None, e_ops=None):
        self.H = H
        self.c_ops = c_ops
        self.R = None
        self.spectra = spectra
        self.evecs = None
        self.dim = H.shape[0] # size of Hilbert space
        self.U = None
        self.G = None # Green's function for Redfield equation

        # deprcated variables, use e_ops as function variable
        self.e_ops = e_ops

    def idm(self, sp=True):
        if sp:
            return dm2vec(identity(self.dim)) # identity operator in Liouville space
        else:
            return dm2vec(identity(self.dim).toarray())

    def configure(self, H, c_ops, e_ops):
        self.c_ops = c_ops
        self.e_ops = e_ops
        self.H = H
        return

    def evolve(self, rho0, dt, Nt, evecs=None, e_ops=[], store_states=False, t0=0, nout=1):
        '''
        propogate the open quantum dynamics

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        Nt : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        if self.R is None:
            R, evecs = self.redfield_tensor()

        result = _redfield(self.R, rho0, evecs=self.evecs, Nt=Nt, dt=dt, t0=t0, e_ops=e_ops)

        return result

    def redfield_tensor(self, secular=False):
        """
        compute the Redfield tensor represented in the eigenbasis of H

        The returned Redfield tensor contains the unitary dynamics and is defined as

        i d/dt rho =  R rho

        Parameters
        ----------
        spectra : list of callback functions
            DESCRIPTION.
        secular : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        H = self.H
        c_ops = self.c_ops

        if self.spectra is None:
            raise TypeError('Specify the bath spectral function.')
        else:
            spectra = self.spectra

        R, evecs = redfield_tensor(H, c_ops, spectra, secular)
        self.R = R
        self.evecs = evecs

        return R, evecs

    # def liouvillian(self):
    #     """
    #     compute the Liouvillian superoperator
    #     i d/dt rho = L rho

    #     Returns
    #     -------
    #     TYPE
    #         DESCRIPTION.

    #     """
    #     if self.R is None:
    #         raise('Call redfield_tensor first.')

    #     return  op2sop(self.H) - 1j * self.R
    def steady_state():
        pass

    def gf(self, t, w=None, secular=False, k=1, domain='time', method='EOM'):
        # compute the Liouville-space Green's function with Redfield dissipation

        # H = self.H
        # c_ops = self.c_ops

        if method == 'EOM':

            # construct total liouvillian
            if self.R is None:
                R, evecs = self.redfield_tensor(secular=secular)

            # Liouvillian

            return -1j * expm(R, t)

        elif method in ['eseries', 'diag', 'diagonalization']:

            if self.R is None:
                R, evecs = self.redfield_tensor(secular=secular)

            return getG(1j * self.R, t)


    def propagator(self, t, method='SOS'):
        """
        compute the Liouville-space propagator with Redfield dissipation

        Parameters
        ----------
        t : list
            time values to compute the propagator.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        list
            list of propagators at t.

        """


        # construct total liouvillian
        if self.R is None:
            raise TypeError('Redfield tensor is not computed. Please call redfield_tensor()')


        if method == 'EOM':

            U = expm(self.R, t)

            # store the Green's function for future use G(t) = -1j * (t>0) * U(t)
            # G = np.zeros((self.dim**2, self.dim**2, len(t)), dtype=complex)
            # for n in range(len(t)):
            #     G[:,:, n] = -1j * U[n].toarray()

            # G = -1j * np.dstack(U)

            U = [_.toarray() for _ in U]

            self.U = np.dstack(U)

        elif method in ['eseries', 'SOS']:

            evals1, U1 = eig(self.R.toarray())

            U2 = scipy.linalg.inv(U1)

            E = np.exp(evals1[:,np.newaxis] * t[np.newaxis,:])
            # self.U =  np.einsum('aj, jk, jb -> abk', U1, E, U2)
            self.U =  oe.contract('aj, jk, jb -> abk', U1, E, U2)

        self.G = -1j * self.U

        return self.U

    def expect(self, rho0, e_ops):

        evecs = self.evecs
        U = self.U
        # transform the intial density matrix to eigenbasis
        rho0_eb = dm2vec(transform(rho0, evecs))
        e_ops = [transform(e, evecs) for e in e_ops]

        if isinstance(U, list):
            # if the propagator is given in list form
            Nt = len(U)
            out = np.zeros((Nt, len(e_ops)), dtype=complex)

            for j, e in enumerate(e_ops):

                rholist = [u.dot(rho0_eb) for u in U]

                out[:, j] = [np.vdot(e, rho) for rho in rholist]

        else: # if propagator is ndarray
            Nt = U.shape[-1]
            out = np.zeros((Nt, len(e_ops)), dtype=complex)

            rho = np.tensordot(U, rho0_eb, axes=([1], [0]))
            for j, e in enumerate(e_ops):
                out[:, j] = dm2vec(e).dot(rho)

        return out

    def correlation_2op_1t(self, rho0, a, b, tau):
        # 2-point correlation function <<I|a G(tau) b|rho0>>

        if self.G is None:
            self.propagator(tau)

        G = self.G

        # unit operator in Liouville space
        idm = dm2vec(identity(self.dim))

        # for _ in [a, b, rho0]:
        #     if _.ndim == 2:
        #         _ = dm2vec(_)

        if rho0.ndim == 2:
            rho0 = dm2vec(rho0)

        corr = idm.dot(a.dot(np.tensordot(G, b.dot(rho0), axes=([1], [0]))))

        return corr


    def correlation_4op_3t(self, rho0, oplist, signature, tau):
        """
        4-point correlation function <<I|A G(tau3) B G(tau2) C G(tau1) D|rho0>>

        make sure all the operators, density matrix, and the GF are represented
        in the eigenbasis of H.

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        oplist : TYPE
            DESCRIPTION.
        signature : str of (lr+-)
            signifies the superoperator transform the operators in the oplist
        tau : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """


        if len(oplist) != 4:
            raise ValueError('Number of operators is not 4.')

        a = operator_to_superoperator(oplist[0], signature[0])
        b = operator_to_superoperator(oplist[1], signature[1])
        c = operator_to_superoperator(oplist[2], signature[2])
        d = operator_to_superoperator(oplist[3], signature[3])

        for _ in oplist:
            if issparse(_):
                _ = _.toarray()


        # nmax = max(len(tau3), len(tau2), len(tau1))

        # g = -1j * (tau > 0) * self.propagator(tau)
        # G = np.zeros((self.dim, self.dim, len(tau)))

        # for n, elem in enumerate(g):
        #     G[:,:, n] = g[n]

        if self.G is None:
            self.propagator(tau)

        G = self.G

        # unit operator in Liouville space
        N = self.dim
        idm = self.idm(sp=False)

        # print(d.shape, dm2vec(rho0.toarray()).shape)
        if issparse(rho0):
            rho = d.dot(dm2vec(rho0.toarray()))
        else:
            rho = d.dot(dm2vec(rho0))

        # print(type(rho), rho.shape)

        if issparse(rho): rho = rho.toarray()

        tmp = np.tensordot(G, rho, axes=((1), (0)))
        tmp = c.dot(tmp)
        tmp = np.tensordot(G, tmp, axes=([1], [0])) # ajk

        # tmp = np.einsum('dcj, ca, abk, b -> djk', G, c, G, rho)


        '''
        Scipy sparse matrix does not support dimensions more than 2, so
        ndarray has to be used for the tensor products.

        This can be improved by using sparse package.
        '''
        tmp = np.tensordot(b.todense(), tmp, axes=([1], [0]))
        # tmp = b.todense().dot(tmp)

        tmp = np.tensordot(G, tmp, axes=([1], [0]))

        return oe.contract('a, ab, bijk -> ijk', idm, a.todense(), tmp)

        # corr = np.einsum('a, ab, bck, bc, cdj, de, efi, f ->kji', idm, \
        #                  left(a), G, left(b), left(c), G, left(d).dot(dm2vec(rho0))

        # return

def _redfield(R, rho0, evecs=None, Nt=1, dt=0.005, t0=0, e_ops=[], return_result=True):

    """
    time propagation of the Redfield quantum master equation with RK4

    Input
    -------
    R: 2d array
        Redfield tensor

    rho0: 2d array
        initial density matrix

    Nt: total number of time steps

    dt: time step
    e_ops: list of observable operators

    Returns
    =========
    rho: 2D array
        density matrix at time t = Nt * dt
    """

    N = rho0.shape[0]
    # initialize the density matrix
    if e_ops is None:
        e_ops = []

    # basis transformation
    if evecs is not None:
        rho0 = transform(rho0, evecs)
        e_ops = [transform(e, evecs) for e in e_ops]

    rho = rho0.copy()
    rho = dm2vec(rho).astype(complex)

    # tf = t0 + dt * Nt
    # result = solve_ivp(rhs, t_span=(t0, tf), y0=rho0, vectorized=True, args=(R, ))

    t = t0

    if return_result == False:

        f_obs = open('obs.dat', 'w')
        fmt = '{} '* (len(e_ops) + 1) + '\n'

        for k in range(Nt):

            # compute observables
            # observables = np.zeros(len(e_ops), dtype=complex)
            # for i, obs_op in enumerate(e_ops):

            observables = [obs_dm(rho, e) for e in e_ops]

            t += dt
            rho = rk4(rho, rhs, dt, R)

            # dipole-dipole auto-corrlation function
            #cor = np.trace(np.matmul(d, rho))

            # take a partial trace to obtain the rho_el


            f_obs.write(fmt.format(t, *observables))


        f_obs.close()
        # f_dm.close()

        return rho

    else:

        rholist = [] # store density matries

        result = Result(dt=dt, Nt=Nt, rho0=rho0)

        observables = np.zeros((Nt, len(e_ops)), dtype=complex)

        for k in range(Nt):

            t += dt
            rho = rk4(rho, rhs, dt, R)

            tmp = np.reshape(rho, (N, N))
            rholist.append(transform(tmp, dag(evecs)))


            observables[k, :] = [obs_dm(tmp, e) for e in e_ops]


        result.observables = observables
        result.rholist = rholist

        return result


def rhs(psi, H):
    return H.dot(psi)

def getG(L, t, w=None, k=6, domain='time'):
    """
    return the Green's function of linear differential operator (id/dt - L)

    time domain
        G(t) = -1j * theta(t) * exp(-i L t)
    frequency domain
        G(w) = 1/(w - L)

    Parameters
    ----------
    l : TYPE
        DESCRIPTION.

    Returns
    -------
    G : TYPE
        DESCRIPTION.

    """
    # evals1, U1 = eigs(L, k=k, which='LR')
    # evals1, U1 = sort(evals1, U1)

    # eigvals2, U2 = eigs(dag(L), k=k, which='LR')
    # eigvals2, U2 = sort(eigvals2, U2)
    evals1, U1 = eig(L.todense())
    # evals1, U1 = sort(evals1, U1)

    # U2 = U1.conj()
    # eigvals2, U2 = eig(dag(L).todense())
    # eigvals2, U2 = sort(eigvals2, U2)

    U2 = scipy.linalg.inv(U1)

    # norm = np.array([np.vdot(U2[:,n], U1[:,n]) for n in range(len(evals1))])

    # print('normalization', norm)

    if domain == 'time':

        U = -1j * np.exp(-1j * evals1[:,np.newaxis] * t[np.newaxis,:])
       # tmp = np.exp(-1j * evals1[:,np.newaxis] * t[np.newaxis,:])

        # U = np.einsum('ab, a -> ab', tmp, 1./norm)

        G =  np.einsum('aj, jk, jb -> abk', U1, U, U2)

    elif domain == 'freq':

        W = 1./((w[:, np.newaxis] - evals1[np.newaxis, :]))
        G = np.einsum('an, nk, bn ->abk', U1, W, U2.conj())

    return G

def redfield_tensor(H, a_ops, spectra, secular=False):

    for a in a_ops:
        if issparse(a):
            if not isherm(a.todense()):
                raise TypeError("Operators in a_ops must be Hermitian.")
        elif not isherm(a):
            raise TypeError("Operators in a_ops must be Hermitian.")

    if issparse(H):
        evals, evecs = eigh(H.todense())
    else:
        evals, evecs = eigh(H)

    print('eigenvalues', evals)

    # pre-calculate matrix elements and spectral densities
    # W[m,n] = real(evals[m] - evals[n])
    W = np.real(evals[:,np.newaxis] - evals[np.newaxis,:])

    # bath spectral density at transition energies

    K = len(a_ops)
    N = len(evals)

    C = []
    for k in range(K):

        c = np.zeros((N,N))
        for n in range(N):
            for m in range(N):
                c[n, m] = spectra[k](-W[n, m])

        C.append(c)

        # C = [s(-W) for s in spectra]

    # transfrom a_ops to eigenbasis
    A = [transform(a, evecs) for a in a_ops]


    # calc the lambda operators
    L = [C[k] * A[k]  for k in range(K)]

    R = 0
    # superoperator
    for k in range(K):
        a = A[k]
        l = L[k]
        R += op2sop(a).dot(left(l) - right(dag(l)))

    return csr_matrix(-1j * op2sop(np.diag(evals))  - R), evecs



class OQS(Mol):
    def __init__(self, H, c_ops=None):
        '''
        open quantum systems class

        Returns
        -------
        None.

        '''
        self.hamiltonian = H
        self.h_sys = H
        self.H = H
        self.nstates = H.shape[-1]
        #self.rho = rho0
        self.e_ops = None
        self.c_ops = None

    def set_hamiltonian(self, h):
        self.H = H
        return

    def setH(self, h):
        self.H = H
        return

    def set_c_ops(self, c_ops):
        self.c_ops = c_ops
        return

    def set_e_ops(self, e_ops):
        """
        set observable operators
        """
        self.e_ops = e_ops
        return

    def configure(self, c_ops, e_ops):
        self.c_ops = c_ops
        self.e_ops = e_ops
        return

    # def heom(self, env, nado=5, fname=None):
    #     nt = self.nt
    #     dt = self.dt
    #     return _heom(self.oqs, env, self.c_ops, nado, nt, dt, fname)

    # def redfield(self, env, dt, Nt, integrator='SOD'):
    #     nstates = self.nstates
    #     rho0 = self.rho0
    #     c_ops = self.c_ops
    #     h0 = self.hamiltonian
    #     e_ops = self.e_ops

    #     redfield(nstates, rho0, c_ops, h0, Nt, dt,e_ops, env, integrator='SOD')

    # def tcl2(self, env, rho0, dt, Nt, integrator='SOD'):
    #     nstates = self.nstates
    #     c_ops = self.c_ops
    #     h0 = self.hamiltonian
    #     e_ops = self.e_ops

    #     redfield(nstates, rho0, c_ops, h0, Nt, dt,e_ops, env, integrator='SOD')

    #     return

    # def lindblad(self, rho0, dt, Nt):
    #     """
    #     lindblad quantum master equations

    #     Parameters
    #     ----------
    #     rho0: 2D array
    #         initial density matrix
    #     """
    #     c_ops = self.c_ops
    #     e_ops = self.e_ops
    #     h0 = self.hamiltonian
    #     lindblad(rho0, h0, c_ops, Nt, dt, e_ops)
    #     return



    def correlation_2p_1t(self, rho0, ops, dt, Nt, method='lindblad', output='cor.dat'):
        '''
        two-point correlation function <A(t)B>

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        ops : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        Nt : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'lindblad'.
        output : TYPE, optional
            DESCRIPTION. The default is 'cor.dat'.

        Returns
        -------
        None.

        '''

        H = self.hamiltonian
        c_ops = self.c_ops

        correlation_2p_1t(H, rho0, ops=ops, c_ops=c_ops, dt=dt,\
                          Nt=Nt, method=method, output=output)

        return

    # def tcl2(self):
    #     """
    #     second-order time-convolutionless quantum master equation
    #     """
    #     pass

# @jit
def liouvillian(rho, H, c_ops):
    """
    lindblad quantum master eqution
    """
    rhs = -1j * comm(H, rho)
    for c_op in c_ops:
        rhs += lindbladian(c_op, rho)
    return rhs

# @jit
def lindbladian(l, rho):
    """
    lindblad superoperator: l rho l^\dag - 1/2 * {l^\dag l, rho}
    l is the operator corresponding to the disired physical process
    e.g. l = a, for the cavity decay and
    l = sm for polarization decay
    """
    return np.dot(l, np.dot(rho, dag(l))) - 0.5 * anticomm(np.dot(dag(l), l), rho)

#@numba.jit
def _correlation_2p_1t(H, rho0, ops, c_ops, dt, Nt, method='lindblad', output='cor.dat'):
    """
    compute the time-translation invariant two-point correlation function in the
    density matrix formalism using quantum regression theorem

        <A(t)B> = Tr[ A U(t) (B rho0)  U^\dag(t)]

    input:
    ========
    H: 2d array
        full Hamiltonian

    rho0: initial density matrix

    ops: list of operators [A, B] for computing the correlation functions

    method: str
        dynamics method e.g. lindblad, redfield, heom

    args: dictionary of parameters for dynamics

    Returns
    ========
    the density matrix is stored in 'dm.dat'
    'cor.dat': the correlation function is stored
    """
    #nstates =  H.shape[-1] # number of states in the system

    # initialize the density matrix
    A, B = ops
    rho = B.dot(rho0)

    f = open(output, 'w')
    # f_dm = open('dm.dat', 'w')
    # fmt = '{} ' * (H.size + 1) + '\n' # format to store the density matrix

    # dynamics

    t = 0.0
    cor = np.zeros(Nt, dtype=complex)

    # sparse matrix
    H = csr_matrix(H)
    rho = csr_matrix(rho)

    A = csr_matrix(A)

    c_ops_sparse = [csr_matrix(c_op) for c_op in c_ops]

    if method == 'lindblad':

        for k in range(Nt):

            t += dt

            rho = rk4(rho, liouvillian, dt, H, c_ops_sparse)

            # cor = A.dot(rho).diagonal().sum()
            tmp = obs_dm(rho, A)
            cor[k] = tmp
            # store the reduced density matrix
            f.write('{} {} \n'.format(t, tmp))

            # f_dm.write(fmt.format(t, *np.ravel(rho)))

            # f_dm.write(fmt.format(t, *np.ravel(rho.toarray())))

    else:
        sys.exit('The method {} has not been implemented yet! Please \
                 try lindblad.'.format(method))

    f.close()
    # f_dm.close()

    return cor

class Env:
    def __init__(self, temperature, cutoff, reorg):
        self.temperature = temperature
        self.cutoff = cutoff
        self.reorg = reorg

    def set_bath_ops(self, bath_ops):
        """


        Parameters
        ----------
        bath_ops : list of 2d arrays
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.bath_ops = bath_ops
        return

    def corr(self):
        """
        compute the correlation function for bath operators
        """
        pass

    def spectral_density(self):
        """
        spectral density
        """
        pass



@jit
def func(rho, h0, c_ops, l_ops):
    """
    right-hand side of the master equation
    """
    rhs = -1j * commutator(h0, rho)

    for i in range(len(c_ops)):
        c_op = c_ops[i]
        l_op = l_ops[i]
        rhs -=  commutator(c_op, l_op.dot(rho) - rho.dot(dag(l_op)))
    return rhs





def coherent(N, alpha):
    """Generates a coherent state with eigenvalue alpha.

    Constructed using displacement operator on vacuum state.

    Modified from Qutip.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue of coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state. Using a non-zero offset will make the
        default method 'analytic'.

    method : string {'operator', 'analytic'}
        Method for generating coherent state.

    Returns
    -------
    state : qobj
        Qobj quantum object for coherent state

    Examples
    --------
    >>> coherent(5,0.25j)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data =
    [[  9.69233235e-01+0.j        ]
     [  0.00000000e+00+0.24230831j]
     [ -4.28344935e-02+0.j        ]
     [  0.00000000e+00-0.00618204j]
     [  7.80904967e-04+0.j        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent state is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting state is normalized. With 'analytic' method the coherent state
    is generated using the analytical formula for the coherent state
    coefficients in the Fock basis. This method does not guarantee that the
    state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """

    x = basis(N, 0)
    a = destroy(N)
    D = la.expm(alpha * dag(a) - np.conj(alpha) * a)

    return D @ x

    # elif method == "analytic" or offset > 0:

    #     data = np.zeros([N, 1], dtype=complex)
    #     n = arange(N) + offset
    #     data[:, 0] = np.exp(-(abs(alpha) ** 2) / 2.0) * (alpha ** (n)) / \
    #         _sqrt_factorial(n)
    #     return Qobj(data)

    # else:
    #     raise TypeError(
    #         "The method option can only take values 'operator' or 'analytic'")



def coherent_dm(N, alpha):
    """Density matrix representation of a coherent state.

    Constructed via outer product of :func:`qutip.states.coherent`

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue for coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state.

    method : string {'operator', 'analytic'}
        Method for generating coherent density matrix.

    Returns
    -------
    dm : qobj
        Density matrix representation of coherent state.

    Examples
    --------
    >>> coherent_dm(3,0.25j)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.93941695+0.j          0.00000000-0.23480733j -0.04216943+0.j        ]
     [ 0.00000000+0.23480733j  0.05869011+0.j          0.00000000-0.01054025j]
     [-0.04216943+0.j          0.00000000+0.01054025j  0.00189294+0.j\
        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent density matrix is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting density matrix is normalized. With 'analytic' method the coherent
    density matrix is generated using the analytical formula for the coherent
    state coefficients in the Fock basis. This method does not guarantee that
    the state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """
    #if method == "operator":
    psi = coherent(N, alpha)

    return ket2dm(psi)

    # elif method == "analytic":
    #     psi = coherent(N, alpha, offset=offset, method='analytic')
    #     return psi * psi.dag()

    # else:
    #     raise TypeError(
    #         "The method option can only take values 'operator' or 'analytic'")



def make_lambda(ns, h0, S, T, cutfreq, reorg):

    tmax = 1000.0
    print('time range to numericall compute the bath correlation function = [0, {}]'.format(tmax))
    print('Decay of correlation function at {} fs = {} \n'.format(tmax * au2fs, \
          corr(tmax, T, cutfreq, reorg)/corr(0., T, cutfreq, reorg)))
    print('!!! Please make sure this is much less than 1 !!!')

    time = np.linspace(0, tmax, 10000)
    dt = time[1] - time[0]
    dt2 = dt/2.0


    l = np.zeros((ns, ns), dtype=np.complex128)

    #t = time[0]
    #phi = hop * t - Delta * np.sin(omegad * t)/omegad
    #Lambda += corr(t) * (np.sin(2. * phi) * sigmay + np.cos(2. * phi) * sigmaz) * dt2

    #h0 = Hamiltonian()
    Sint = S.copy()

    for k in range(len(time)):

        t = time[k]

        Sint += -1j * commutator(S, h0) * (-dt)
        l += dt * Sint * corr(t, T, cutfreq, reorg)

#    t = time[len(time)-1]
#    phi = hop * t + Delta * np.sin(omegad * t)/omegad
#    Lambda += corr(t) * (np.sin(2. * phi) * sigmay + np.cos(2. * phi) * sigmaz) * dt2
#    Lambda = cy * sigmay + cz * sigmaz

    return l



def _redfield_old(nstates, rho0, c_ops, h0, Nt, dt,e_ops, env):
    """
    time propagation of the Redfield equation with second-order differencing
    input:
        nstate: total number of states
        h0: system Hamiltonian
        Nt: total number of time steps
        dt: tiem step
        c_ops: list of collapse operators
        e_ops: list of observable operators
        rho0: initial density matrix
    """
    t = 0.0

    print('Total number of states in the system = {}'.format(nstates))

    # initialize the density matrix
    rho = rho0

    # properties of the environment
    T = env.T
    cutfreq = env.cutfreq
    reorg = env.reorg

    #f = open(fname,'w')
    fmt = '{} '* (len(e_ops) + 1) + '\n'

    # construct system-bath operators in H_SB

    # short time approximation
    # Lambda = 0.5 * reorg * T * ((hop - Delta)/cutfreq**2 * sigmay + 1./cutfreq * sigmaz)

    # constuct the Lambda operators needed in Redfield equation
    l_ops = []
    for c_op in c_ops:
        l_ops.append(getLambda(nstates, h0, c_op, T, cutfreq, reorg))

    f_dm = open('den_mat.dat', 'w')
    f_obs = open('obs.dat', 'w')

    t = 0.0
    dt2 = dt/2.0

    # first-step
    rho_half = rho0 + func(rho0, h0, c_ops, l_ops) * dt2
    rho1 = rho0 + func(rho_half, h0, c_ops, l_ops) * dt

    rho_old = rho0
    rho = rho1

    for k in range(Nt):

        t += dt

        rho_new = rho_old + func(rho, h0, c_ops, l_ops) * 2. * dt

        # update rho_old
        rho_old = rho
        rho = rho_new

        # dipole-dipole auto-corrlation function
        #cor = np.trace(np.matmul(d, rho))

        # store the reduced density matrix
        f_dm.write('{} '* (nstates**2 + 1) + '\n'.format(t, *rho))

        # take a partial trace to obtain the rho_el
        obs = np.zeros(len(e_ops))
        for i, obs_op in enumerate(e_ops):
            obs[i] = observe(obs_op, rho)

        f_obs.write(fmt.format(t * au2fs, *obs))


    f_obs.close()
    f_dm.close()

    return rho

@jit
def observe(A, rho):
    """
    compute expectation value of the operator A
    """
    return A.dot(rho).diagonal().sum()

class Lindblad_solver():
    def __init__(self, H=None, c_ops=None, e_ops=None):
        self.c_ops = c_ops
        self.e_ops = e_ops
        self.H = H
        return

    def set_c_ops(self, c_ops):
        self.c_ops = c_ops
        return

    def set_e_ops(self, e_ops):
        """
        set observable operators
        """
        self.e_ops = e_ops
        return

    def setH(self, H):
        self.H = H
        return

    def configure(self, c_ops, e_ops):
        self.c_ops = c_ops
        self.e_ops = e_ops
        return

    def liouvillian(self):
        H = self.H
        c_ops = self.c_ops
        return superop.liouvillian(H, c_ops)

    def steady_state(self):
        pass

    def evolve(self, rho0, dt, Nt, t0=0., e_ops=None, return_result=True):
        """
        propagate the dynamics

        if H is ndarray, H is time-independent
        if H = [H0, [f(t), H1]], H = H0 - f(t) * H1 is time-dependent

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        Nt : TYPE
            DESCRIPTION.
        t0 : TYPE, optional
            DESCRIPTION. The default is 0..
        e_ops : TYPE, optional
            DESCRIPTION. The default is None.
        return_result : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if isinstance(self.H, list):

            return _lindblad_driven(self.H, rho0=rho0, c_ops=self.c_ops,
                                    e_ops=e_ops,
                                    Nt=Nt, dt=dt, t0=t0)

        else:

            return _lindblad(self.H, rho0, c_ops=self.c_ops, e_ops=e_ops, \
                  Nt=Nt, dt=dt, return_result=return_result)

    def evolve_fast(self, rho0, dt, Nt, t0=0., e_ops=None, return_result=True, fast=True):
            return _fast_lindblad(self.H, rho0, c_ops=self.c_ops, e_ops=e_ops, \
                   Nt=Nt, dt=dt)

    def correlation_2op_1t(self, rho0, a_op, b_op, dt, Nt, output='cor.dat'):
        '''
        two-point correlation function <A(t)B>

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        ops : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        Nt : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'lindblad'.
        output : TYPE, optional
            DESCRIPTION. The default is 'cor.dat'.

        Returns
        -------
        None.

        '''

        c_ops = self.c_ops
        H = self.H

        return _correlation_2p_1t(H, rho0, ops=[a_op, b_op], c_ops=c_ops, dt=dt,\
                          Nt=Nt, output=output)

    def correlation_3op_1t(self, rho0, oplist, dt=0.005, Nt=1):
        """
        <AB(t)C>

        Parameters
        ----------
        psi0
        oplist
        dt
        Nt

        Returns
        -------
        cor: 1D array
        """
        a_op, b_op, c_op = oplist
        cor = _lindblad(self.H, rho0 = c_op @ rho0 @ a_op, c_ops=self.c_ops, \
                        e_ops=[b_op], dt=dt, Nt=Nt).observables[:,0]

        return cor

    def correlation_4op_1t(self, rho0, oplist, dt=0.005, Nt=1):
        """
        <AB(t)C(t)D>

        Parameters
        ----------
        psi0
        oplist
        dt
        Nt

        Returns
        -------

        """
        a_op, b_op, c_op, d_op = oplist
        return self.correlation_3p_1t(rho0=rho0, oplist=[a_op, b_op @ c_op, d_op],\
                                      dt=dt, Nt=Nt)


    def correlation_3op_2t(self, rho0, ops, dt, Nt, Ntau):
        """
        Internal function for calculating the three-operator two-time
        correlation function:
        <A(t)B(t+tau)C(t)>
        using the Linblad master equation solver.
        """

        # the solvers only work for positive time differences and the correlators
        # require positive tau
        # if state0 is None:
        #     rho0 = steadystate(H, c_ops)
        #     tlist = [0]
        # elif isket(state0):
        #     rho0 = ket2dm(state0)
        # else:
        #     rho0 = state0
        H = self.H
        c_ops = self.c_ops
        rho_t = _lindblad(H, rho0, c_ops, dt=dt, Nt=Nt, return_result=True).rholist

        a_op, b_op, c_op = ops

        corr_mat = np.zeros([Nt, Ntau], dtype=complex)

        for t_idx, rho in enumerate(rho_t):

            corr_mat[t_idx, :] = _lindblad(H, rho0=c_op @ rho @ a_op, \
                                           dt=dt, Nt=Ntau, c_ops=c_ops,\
                e_ops=[b_op], return_result=True).observables[:,0]

        return corr_mat

    def correlation_4op_1t(self, rho0, ops, dt, nt):
        """
        Internal function for calculating the four-operator two-time
        correlation function:
        <A B(t)C(t) D>
        using the Linblad master equation solver.
        """

        if len(ops) != 4:
            raise ValueError('Number of operators is not 4.')
        else:
            a, b, c, d = ops

        corr = self.correlation_3op_1t(rho0, [a, b@c, d], dt, nt)
        return corr

    def correlation_4op_2t(self, rho0, ops, dt, nt, ntau):
        """
        Internal function for calculating the four-operator two-time
        correlation function:
        <A(t)B(t+tau)C(t+tau)D(t)>
        using the Linblad master equation solver.
        """

        if len(ops) != 4:
            raise ValueError('Number of operators is not 4.')
        else:
            a, b, c, d = ops

        corr = self.correlation_3op_2t(rho0, [a, b@c, d], dt, nt, ntau)
        return corr



class HEOMSolverDL():
    def __init__(self, H=None, c_ops=None, e_ops=None):
        self.c_ops = c_ops
        self.e_ops = e_ops
        self.e_ops = e_ops
        self.H = H
        return

    def set_c_ops(self, c_ops):
        self.c_ops = c_ops
        return

    def set_e_ops(self, e_ops):
        """
        set observable operators
        """
        self.e_ops = e_ops

        return

    def setH(self, H):
        self.H = H
        return

    def configure(self, c_ops, e_ops):
        self.c_ops = c_ops
        self.e_ops = e_ops
        return

    def solve(self, rho0, dt, Nt, return_result):
        return _lindblad(self.H, rho0, self.c_ops, e_ops=self.e_ops, \
                  Nt=Nt, dt=dt, return_result=return_result)

    def correlation_2op_1t(self, rho0, a_op, b_op, dt, Nt, output='cor.dat'):
        '''
        two-point correlation function <A(t)B>

        Parameters
        ----------
        rho0 : TYPE
            DESCRIPTION.
        ops : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.
        Nt : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'lindblad'.
        output : TYPE, optional
            DESCRIPTION. The default is 'cor.dat'.

        Returns
        -------
        None.

        '''

        c_ops = self.c_ops
        H = self.H

        return _correlation_2p_1t(H, rho0, ops=[a_op, b_op], c_ops=c_ops, dt=dt,\
                          Nt=Nt, output=output)


    def correlation_3op_2t(self, rho0, ops, dt, Nt, Ntau):
        """
        Internal function for calculating the three-operator two-time
        correlation function:
        <A(t)B(t+tau)C(t)>
        using the Linblad master equation solver.
        """

        # the solvers only work for positive time differences and the correlators
        # require positive tau
        # if state0 is None:
        #     rho0 = steadystate(H, c_ops)
        #     tlist = [0]
        # elif isket(state0):
        #     rho0 = ket2dm(state0)
        # else:
        #     rho0 = state0
        H = self.H
        c_ops = self.c_ops
        rho_t = _lindblad(H, rho0, c_ops, dt=dt, Nt=Nt, return_result=True).rholist

        a_op, b_op, c_op = ops

        corr_mat = np.zeros([Nt, Ntau], dtype=complex)

        for t_idx, rho in enumerate(rho_t):

            corr_mat[t_idx, :] = _lindblad(H, rho0=c_op @ rho @ a_op, \
                                           dt=dt, Nt=Ntau, c_ops=c_ops,\
                e_ops=[b_op], return_result=True).observables[:,0]

        return corr_mat
# exponential series solvers

# def _correlation_es_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op):
#     """
#     Internal function for calculating the three-operator two-time
#     correlation function:
#     <A(t)B(t+tau)C(t)>
#     using an exponential series solver.
#     """

#     # the solvers only work for positive time differences and the correlators
#     # require positive tau
#     if state0 is None:
#         rho0 = steadystate(H, c_ops)
#         tlist = [0]
#     elif isket(state0):
#         rho0 = ket2dm(state0)
#     else:
#         rho0 = state0

#     if debug:
#         print(inspect.stack()[0][3])

#     # contruct the Liouvillian
#     L = liouvillian(H, c_ops)

#     corr_mat = np.zeros([np.size(tlist), np.size(taulist)], dtype=complex)
#     solES_t = ode2es(L, rho0)

#     # evaluate the correlation function
#     for t_idx in range(len(tlist)):
#         rho_t = esval(solES_t, [tlist[t_idx]])
#         solES_tau = ode2es(L, c_op * rho_t * a_op)
#         corr_mat[t_idx, :] = esval(expect(b_op, solES_tau), taulist)

#     return corr_mat

# @jit
# def _rk4(rho, fun, dt, H, c_ops):
#     """
#     Runge-Kutta method
#     """
#     dt2 = dt/2.0

#     k1 = fun(rho,  H, c_ops )
#     k2 = fun(rho + k1*dt2,  H, c_ops)
#     k3 = fun(rho + k2*dt2,  H, c_ops)
#     k4 = fun(rho + k3*dt,  H, c_ops)

#     rho += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

#     return rho

@njit
def _fast_lindblad(H, rho0, c_ops, e_ops=None, Nt=1, dt=0.005):

    """
    fast time propagation of the lindblad quantum master equation

    (not completed yet.)

    Input
    -------
    h0: 2d array
            system Hamiltonian
    Nt: total number of time steps

    dt: time step
        c_ops: list of collapse operators
        e_ops: list of observable operators
        rho0: initial density matrix

    Returns
    =========
    rho: 2D array
        density matrix at time t = Nt * dt
    """

    # initialize the density matrix
    rho = rho0

    # if e_ops is None:
    #     e_ops = []

    t = 0.0
    dt2 = dt/2.
    # first-step
    # rho_half = rho0 + liouvillian(rho0, h0, c_ops) * dt2
    # rho1 = rho0 + liouvillian(rho_half, h0, c_ops) * dt

    # rho_old = rho0
    # rho = rho1

        # f_dm = open('den_mat.dat', 'w')
        # fmt_dm = '{} ' * (nstates**2 + 1) + '\n'

        # f_obs = open('obs.dat', 'w')
        # fmt = '{} '* (len(e_ops) + 1) + '\n'

    for k in range(Nt):

        t += dt

        # rho_new = rho_old + liouvillian(rho, h0, c_ops) * 2. * dt
        # # update rho_old
        # rho_old = rho
        # rho = rho_new

        # rho = rk4(rho, liouvillian, dt, H, c_ops)
        k1 = liouvillian(rho, H, c_ops )
        k2 = liouvillian(rho + k1*dt2,  H, c_ops)
        k3 = liouvillian(rho + k2*dt2,  H, c_ops)
        k4 = liouvillian(rho + k3*dt,  H, c_ops)

        rho += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

        # dipole-dipole auto-corrlation function
        #cor = np.trace(np.matmul(d, rho))

        # take a partial trace to obtain the rho_el
        # compute observables
        observables = np.zeros(len(e_ops), dtype=complex)

        # for i, obs_op in enumerate(e_ops):
        #     observables[i] = obs_dm(rho, obs_op)

        # f_obs.write(fmt.format(t, *observables))


    # f_obs.close()
    # f_dm.close()

    return rho

    # else:

    #     rholist = [] # store density matries

    #     result = Result(dt=dt, Nt=Nt, rho0=rho0)

    #     observables = np.zeros((Nt, len(e_ops)), dtype=complex)

    #     for k in range(Nt):

    #         t += dt
    #         rho = rk4(rho, liouvillian, dt, H, c_ops)

    #         rholist.append(rho.copy())


    #         observables[k, :] = [obs_dm(rho, op) for op in e_ops]


    #     result.observables = observables
    #     result.rholist = rholist

    #     return result

def _lindblad(H, rho0, c_ops, e_ops=None, Nt=1, dt=0.005, return_result=True):

    """
    time propagation of the lindblad quantum master equation
    with second-order differencing

    Input
    -------
    h0: 2d array
            system Hamiltonian
    Nt: total number of time steps

    dt: time step
        c_ops: list of collapse operators
        e_ops: list of observable operators
        rho0: initial density matrix

    Returns
    =========
    rho: 2D array
        density matrix at time t = Nt * dt
    """

    # initialize the density matrix
    rho = rho0.copy()
    rho = rho.astype(complex)

    if e_ops is None:
        e_ops = []

    t = 0.0
    # first-step
    # rho_half = rho0 + liouvillian(rho0, h0, c_ops) * dt2
    # rho1 = rho0 + liouvillian(rho_half, h0, c_ops) * dt

    # rho_old = rho0
    # rho = rho1
    if return_result == False:

        # f_dm = open('den_mat.dat', 'w')
        # fmt_dm = '{} ' * (nstates**2 + 1) + '\n'

        f_obs = open('obs.dat', 'w')
        fmt = '{} '* (len(e_ops) + 1) + '\n'

        for k in range(Nt):

            # compute observables
            observables = np.zeros(len(e_ops), dtype=complex)

            for i, obs_op in enumerate(e_ops):
                observables[i] = obs_dm(rho, obs_op)

            t += dt

            # rho_new = rho_old + liouvillian(rho, h0, c_ops) * 2. * dt
            # # update rho_old
            # rho_old = rho
            # rho = rho_new

            rho = rk4(rho, liouvillian, dt, H, c_ops)

            # dipole-dipole auto-corrlation function
            #cor = np.trace(np.matmul(d, rho))

            # take a partial trace to obtain the rho_el


            f_obs.write(fmt.format(t, *observables))


        f_obs.close()
        # f_dm.close()

        return rho

    else:

        rholist = [] # store density matries

        result = Result(dt=dt, Nt=Nt, rho0=rho0)

        observables = np.zeros((Nt, len(e_ops)), dtype=complex)

        for k in range(Nt):

            t += dt
            rho = rk4(rho, liouvillian, dt, H, c_ops)

            rholist.append(rho.copy())


            observables[k, :] = [obs_dm(rho, op) for op in e_ops]


        result.observables = observables
        result.rholist = rholist

        return result


def _lindblad_driven(H, rho0, c_ops=None, e_ops=None, Nt=1, dt=0.005, t0=0.,
                     return_result=True):

    """
    time propagation of the lindblad quantum master equation with time-dependent Hamiltonian
    H = H0 + f(t) * H1 - ...

    Input
    -------
    H:  list [H0, [H1, f1(t)]]
            system Hamiltonian
    pulse: Pulse object
        externel pulse
    Nt: total number of time steps

    dt: time step
    c_ops: list of collapse operators
    e_ops: list of observable operators
    rho0: initial density matrix

    Returns
    =========
    rho: 2D array
        density matrix at time t = Nt * dt
    """

    def calculateH(t):

        Ht = H[0]

        for i in range(1, len(H)):
            Ht += - H[i][1](t) * H[i][0]

        return Ht

    nstates = H[0].shape[-1]

    if c_ops is None:
        c_ops = []
    if e_ops is None:
        e_ops = []


    # initialize the density matrix
    rho = rho0.copy()
    rho = rho.astype(complex)



    t = t0

    if return_result == False:

        f_dm = open('den_mat.dat', 'w')
        fmt_dm = '{} ' * (nstates**2 + 1) + '\n'

        f_obs = open('obs.dat', 'w')
        fmt = '{} '* (len(e_ops) + 1) + '\n'

        for k in range(Nt):

            t += dt

            Ht = calculateH(t)

            rho = rk4(rho, liouvillian, dt, Ht, c_ops)

            # dipole-dipole auto-corrlation function
            #cor = np.trace(np.matmul(d, rho))

            # take a partial trace to obtain the rho_el
            # compute observables
            observables = np.zeros(len(e_ops), dtype=complex)

            for i, obs_op in enumerate(e_ops):
                observables[i] = obs_dm(rho, obs_op)

            f_obs.write(fmt.format(t, *observables))


        f_obs.close()
        f_dm.close()

        return rho

    else:

        rholist = [] # store density matries

        result = Result(dt=dt, Nt=Nt, rho0=rho0)

        observables = np.zeros((Nt, len(e_ops)), dtype=complex)

        for k in range(Nt):

            t += dt

            Ht = calculateH(t)

            rho = rk4(rho, liouvillian, dt, Ht, c_ops)

            rholist.append(rho.copy())

            observables[k, :] = [obs_dm(rho, op) for op in e_ops]


        result.observables = observables
        result.rholist = rholist

        return result

def _heom_dl(H, rho0, c_ops, e_ops, temperature, cutoff, reorganization,\
             nado, dt, nt, fname=None, return_result=True):
    '''

    terminator : ado[:,:,nado] = 0

    INPUT:
        T: in units of energy, kB * T, temperature of the bath
        reorg: reorganization energy
        nado : auxiliary density operators, truncation of the hierachy
        fname: file name for output

    '''
    nst = H.shape[0]
    ado = np.zeros((nst, nst, nado), dtype=np.complex128)     # auxiliary density operators
    ado[:,:,0] = rho0 # initial density matrix



    gamma = cutoff # cutoff frequency of the environment, larger gamma --> more Makovian
    T = temperature/au2k
    reorg = reorganization
    print('Temperature of the environment = {}'.format(T))
    print('High-Temperature check gamma/(kT) = {}'.format(gamma/T))

    if gamma/T > 0.8:
        print('WARNING: High-Temperature Approximation may fail.')

    print('Reorganization energy = {}'.format(reorg))

    # D(t) = (a + ib) * exp(- gamma * t)
    a = np.pi * reorg * T  # initial value of the correlation function D(0) = pi * lambda * kB * T
    b = 0.0
    print('Amplitude of the fluctuations = {}'.format(a))

    #sz = np.zeros((nstate, nstate), dtype=np.complex128)
    sz = c_ops # collapse opeartor


    f = open(fname,'w')
    fmt = '{} '* 5 + '\n'

    # propagation time loop - HEOM
    t = 0.0
    for k in range(nt):

        t += dt # time increments

        ado[:,:,0] += -1j * commutator(H, ado[:,:,0]) * dt - \
            commutator(sz, ado[:,:,1]) * dt

        for n in range(nado-1):
            ado[:,:,n] += -1j * commutator(H, ado[:,:,n]) * dt + \
                        (- commutator(sz, ado[:,:,n+1]) - n * gamma * ado[:,:,n] + n * \
                        (a * commutator(sz, ado[:,:,n-1]) + \
                         1j * b * anticommutator(sz, ado[:,:,n-1]))) * dt

        # store the reduced density matrix
        f.write(fmt.format(t, ado[0,0,0], ado[0,1,0], ado[1,0,0], ado[1,1,0]))

        #sz += -1j * commutator(sz, H) * dt

    f.close()
    return ado[:,:,0]

def test_lindblad():
    mesolver = Lindblad_solver(H, c_ops=[sz.astype(complex)])
    Nt = 800
    dt = 0.5

    import time
    start_time = time.time()
    # L = mesolver.liouvillian()
    t0=-10/au2fs
    result = mesolver.evolve(rho0, dt=dt, Nt=Nt, e_ops = [H1], return_result=True, t0=t0)
    #corr = mesolver.correlation_3op_2t(rho0, [sz, sz, sz], dt, Nt, Ntau=Nt)

    # print(result.observables)
    import proplot as plt
    fig, ax = plt.subplots()
    times = np.arange(Nt) * dt + t0
    ax.plot(times, result.observables[:,0])
    ax.format(ylabel='Coherence')

    print('Execution time = {} s'.format(time.time() - start_time))

    fig, ax = plt.subplots()
    ax.plot(times, pulse.field(times))
    ax.set_ylabel('E(t)')
    return


if __name__ == '__main__':


    from lime.phys import pauli
    from lime.optics import Pulse
    from lime.units import au2ev

    import time
    start_time = time.time()

    s0, sx, sy, sz = pauli()

    # set up the molecule
    H0 =  1./au2ev * 0.5 * (s0 - sz)
    H1 = sx.astype(complex)

    # def coeff1(t):
    #     return np.exp(-t**2)*np.cos(1./au2ev * t)

    pulse = Pulse(tau=2/au2fs, omegac=1./au2ev, delay=0, amplitude=0.05)


    # H = [H0.astype(complex), [H1.astype(complex), pulse.field]]
    H = H0.astype(complex)

    psi0 = basis(2, 0)
    rho0 = ket2dm(psi0).astype(complex)

    mesolver = Redfield_solver(H, c_ops=[sz.astype(complex)])
    Nt = 800
    dt = 0.5


    result = mesolver.evolve(rho0, dt=dt, Nt=Nt, e_ops = [H1])
    #corr = mesolver.correlation_3op_2t(rho0, [sz, sz, sz], dt, Nt, Ntau=Nt)

    # print(result.observables)
    import proplot as plt
    fig, ax = plt.subplots()
    times = np.arange(Nt) * dt
    ax.plot(times, result.observables[:,0])
    ax.format(ylabel='Coherence')

    print('Execution time = {} s'.format(time.time() - start_time))

    # fig, ax = plt.subplots()
    # ax.plot(times, pulse.field(times))
    # ax.set_ylabel('E(t)')
