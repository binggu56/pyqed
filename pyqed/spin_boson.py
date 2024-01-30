import numpy as np
import sys
from scipy.sparse import csr_matrix
import numba

from pyqed import pauli, dag, comm, au2k
from pyqed.phys import rk4
from pyqed.oqs import HEOMSolver, op2sop

from tqdm import tqdm
from numba import vectorize

s0, sx, sy, sz = pauli()

class SpinBoson(object):
    def __init__(self, beta, delta=1, omegac=1.0, reorg=2.0):
        """
        Spin-Boson model 
        
        .. math::
            H = \Delta \sigma_x/2 + \sigma_z B + H_B
            
            B = \sum_j c_j x_j 
            H_B = \sum_j \omega_j a_j^\dag a_j
            
        Note that the excited states is 0. 
        
        Parameters
        ----------
        beta : TYPE
            DESCRIPTION.
        delta : TYPE, optional
            DESCRIPTION. The default is 1.
        omegac : TYPE, optional
            DESCRIPTION. The default is 1.0.
        reorg : TYPE, optional
            DESCRIPTION. The default is 2.0.

        Returns
        -------
        None.

        """
        self.beta = beta # temperature
        self.omegac = omegac
        self.reorg = reorg
        self.delta = delta
        
        self.H = delta * sx/2       

    def pure_dephasing(self, t):
        """
        Compute the exact decoherence function for the pure-dephasing spin_boson
            model H = eps0* sigmaz/2 + sigmaz *(g_k a^\dag_k + g_k^* a_k) +
            omega_k a_k^\dag a_k

        INPUT:
            beta: inverse temperature
            omegac: cutoff frequency
            reorg: reorganization energy

        OUTPUT: Decoherence function at time t
            ln Phi(t) = - int_0^\infty J(omega)coth(beta*omega/2) domega
            J(omega) = sum_k |g_k|^2 delta(omega - omega_k)

        """
        omegac = self.omegac
        reorg = self.reorg
        beta = self.beta

        freqmax = 20.0
        N = 2**11
        freq = np.linspace(1e-4, freqmax, N)

        # parameters for spectral density Drude
        name = 'Drude'

        # J(w) * coth(beta * w/2)
        J = self.spectral_density(freq, omegac, reorg, name) / np.tanh(0.5 \
                                 * beta * freq)

        if J[-1]/freqmax**2 > 1e-3:
            print('J[-1]/freqmax^2 = {} \n Max frequency not big enough \
                  for integration.'.format(J[-1]/freqmax**2))

        # decoherence function
        dw = freq[1] - freq[0]
        tmp = - np.sum(J * (1. - np.cos(freq * t)) / freq**2) * dw

        return tmp

    def drude_spectral_density(self, omega, omegac, reorg, name):
        """
        Drude spectral density
        """
        if name == 'Drude':
            return 2. * reorg * omegac * omega/(omega**2 + omegac**2)
        else:
            sys.exit('Error: There is no such spectral denisty.')

    def update_temp(self, beta):
        """
        update inverse temperature
        INPUT:
            beta: 1/(k_B T) in units of Energy
        """
        self.beta = beta

        return
    
    def sz_interaction_picture(self, t):
        """
        Compute the operator :math:`\sigma_z` in the interaction picture of\
            :math:`\Delta \sigma_x/2`
    
        .. math::
            \sigma_z(t) = e^{+ i H t} \sigma_z e^{-i H t}
            
        Parameters
        ----------
        t : float
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        
        
        return sz * np.cos(self.delta * t) + sy * np.sin(self.delta * t)
    
    
    def HEOM(self):
        sol = HEOMSolver(self.H, c_ops=[sz])
        return sol
    
    def TCL2(self):
        sol = TCL2(self.H, c_ops=[sz])
        return sol


def corr(t, gamma=1, reorg=1, T=500/au2k):
    """
    bath correlation function for Drude spectral density at high-T

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    reorg : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return reorg * (2 * T  - 1j * gamma) * np.exp(-gamma * t)


def sz_int(t):
    """
    Compute the operator :math:`\sigma_z` in the interaction picture of\
        :math:`\sigma_x/2`

    .. math::
        \sigma_z(t) = e^{+ i H t} \sigma_z e^{-i H t}
        
    Parameters
    ----------
    t : float
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    return sz * np.cos(t) + sy * np.sin(t)


def time_local_generator(t):
    """
    .. math::
        
        G(t) = -  S^-(t)  \int_0^t \dif t_1  D_K(t,t_1) S^-(t_1) + \
            i D_C(t,t_1) S^+(t_1)

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    G : TYPE
        DESCRIPTION.

    """

    D = corr(t[:, np.newaxis] - t[np.newaxis, :])
    D -= np.tril(D, k=-1)
    
    K = np.real(D)  # keldysh correlation function
    C = np.imag(D)  # commutator correlation function
    
    S = [sz_int(_t) for _t in t]
    Sm = [op2sop(_S, kind='commutator').toarray() for _S in S]
    Sp = [op2sop(_S, kind='anticommutator').toarray() for _S in S]
    
    Sm = np.array(Sm)
    Sp = np.array(Sp)
    
    G = np.tensordot(K, Sm, axes=(1, 0)) + 1j * np.tensordot(C, Sp, axes=(1,0)) # shape [nt, N^2, N^2] 
    G = - np.einsum('tij, tjk -> tik', Sm, G)
    return G

class TCL2:
    def __init__(self, H, c_ops, e_ops=None):
        self.H = H
        self.n = H.shape[0] # size of system Hilbert space
        assert(len(c_ops) == 1) # only work for one-c_op for now
        self.c_ops = c_ops
        self.e_ops = e_ops
        
    
    # def bath_correlation_function(self):
    #     pass
        
    def run(self, rho0, dt, nt, corr=None):
        """
        time propagation of the Redfield equation
        """
        t = 0.0
        c_ops = self.c_ops
        e_ops = self.e_ops
    
        # ns = n_el * n_vt * n_vc * n_cav  # number of states in the system
    
        # # initialize the density matrix
        # rho_el = np.zeros((n_el, n_el), dtype=np.complex128)
        # rho_el[1, 1] = 1.0
        # rho_el[0, 0] = 0.
    
    
        # rho_cav = np.zeros((n_cav, n_cav), dtype=np.complex128)
        # rho_cav[0, 0] = 1.0
        # rho_vc = np.zeros((n_vc, n_vc), dtype=np.complex128)
        # rho_vc[0, 0] = 1.0
        # rho_vt = np.zeros((n_vt, n_vt), dtype=np.complex128)
        # rho_vt[0, 0] = 1.0
    
        # rho0 = kron(rho_el, np.kron(rho_cav, np.kron(rho_vc, rho_vt)))
    
        rho = rho0.copy()
    
        #f = open(fname,'w')
        #fmt = '{} '* 5 + '\n'
    
        # construct system-bath operators in H_SB
    
        # short time approximation
        # Lambda = 0.5 * reorg * T * ((hop - Delta)/cutfreq**2 * sigmay + 1./cutfreq * sigmaz)
    
        # constuct system Hamiltonian and system collapse operators
        # h0, S1, S2 = ham_sys(n_el, n_cav, n_vc, n_vt)
    
    #    file_lambda = 'lambda.npz'
    #
    #    if read_lambda == True:
    #
    #        data = np.load(file_lambda)
    #        Lambda1 = data['arr_0']
    #        Lambda2 = data['arr_1']
    #
    #    else:
    #        # constuct the operators needed in Redfield equation
    #        Lambda1 = getLambda(ns, h0, S1, T, cutfreq, reorg)
    #        Lambda2 = getLambda(ns, h0, S2, T, cutfreq, reorg)
    #
    #        np.savez(file_lambda, Lambda1, Lambda2)
    
    
        #print(Lambda1, '\n', Lambda2)
    
        f_dm = open('den_mat.dat', 'w')
        f_pos = open('position.dat', 'w')
    
        t = 0.0
        dt2 = dt/2.0
        H = self.H 
        # observables
        # Ag, Ae, A_coh = pop(n_el, n_cav, n_vc, n_vt)
    
    
        print('Time propagation start ... \n')
        # ns = self.n
        # Lambda1 = csr_matrix((ns, ns), dtype=np.complex128)
        # Lambda2 = csr_matrix((ns, ns), dtype=np.complex128)
        
        G = time_local_generator(t)
    
        for k in tqdm(range(nt)):
            

            rho += G[k] @ rho * dt
            # rho = rk4(rho, func, dt, H, c_ops, l_ops)


            # Lambda1 += s1_int * corr(t) * dt2

            # Sm = op2sop( sz_int(t) )
            
            # for c_op in c_ops:
            #     c_op = rk4(c_op, liouville, -dt, H)
            # # s2_int = rk4_step(s2_int, liouville, -dt, H)
            # for l_op in l_ops:
            #     l_op += c_ops[i] * corr(t) * dt2
            
            # Lambda2 += s2_int * corr(t) * dt2
    
    
            # dipole-dipole auto-corrlation function
            #cor = np.trace(np.matmul(d, rho))
    
            # store the reduced density matrix
            #f.write('{} {} \n'.format(t, cor))
    
            # take a partial trace to obtain the rho_el

            observables = [obs(e_op, rho) for e_op in e_ops]

    
            # position expectation
            # qc = obs(S1, rho)
            # qt = obs(S2, rho)
    
            t += dt
    
    
            # f_dm.write('{} {} {} {} \n'.format(t, P1, P2, coh))
            # f_pos.write('{} {} {} \n'.format(t, qc, qt))
    
    
        f_pos.close()
        f_dm.close()
    
        return rho

def liouville(a, h):
    return 1j * comm(h, a)

def obs(A, rho):

    return A.dot( rho).diagonal().sum()

# @numba.autojit
# def _liouvillian_tcl2(rho, h0, c_ops, Lambda):
#     """
#     right-hand side of the master equation
#     """
#     rhs = -1j * comm(h0, rho) 
#     for i, c_op in enumerate(c_ops):
#         l = Lambda[i]
#         rhs -=  comm(c_op, l.dot(rho) - rho.dot(dag(l))) 
            
#     return csr_matrix(rhs)


def liouvillian_tcl2(rho, h0, c_ops, l_ops):
    """
    right-hand side of the Redfield and time-convolutionless master equation in the following \
        form 
    
    ..math::
        \mathcal{L} \rho = -i [H_0, \rho] + [C, L \rho - \rho L^\dag]   
    
    """
    rhs = -1j * comm(h0, rho)

    for i in range(len(c_ops)):
        c_op = c_ops[i]
        l_op = l_ops[i]
        rhs -=  comm(c_op, l_op.dot(rho) - rho.dot(dag(l_op)))
        
    return csr_matrix(rhs)
        
if __name__=='__main__':


    # def ohmic_spectrum(w):
    #   # if w == 0.0: # dephasing inducing noise
    #   #   return gamma1
    #   # else: # relaxation inducing noise
    #    return gamma1 / 2 * (w / (2 * np.pi))


    # redfield = Redfield_solver(H, c_ops=[sx], spectra=[ohmic_spectrum])
    # R, evecs = redfield.redfield_tensor()
    rho0 = np.zeros((2,2))
    rho0[0, 0] = 1
    
    
    # sol = HEOMSolver(H, c_ops=[sz])
    
    sbm = SpinBoson(beta=0.1)
    # sol = sbm.HEOM()
    
    # nt = 100 
    # rho = sol.run(rho0=rho0, dt=0.001, nt=nt, temperature=3e5, cutoff=5, reorganization=0.2, nado=5)
    # print(rho)
    dt = 0.05
    t = dt * np.arange(10)
    G = time_local_generator(t)
    print(G.shape)
    
    
    





