import numpy as np
import sys
from scipy.sparse import csr_matrix
import numba

from pyqed import pauli, dag, comm
from pyqed.oqs import HEOMSolver



s0, sx, sy, sz = pauli()

class SpinBoson(object):
    def __init__(self, beta, delta=1, omegac=1.0, reorg=2.0):
        """
        Spin-Boson model 
        
        .. math::
            H = \Delta \sigma_x + \sigma_z B + H_B
            
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
        
        self.H = delta/2 * sx       

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

    def spectral_density(self, omega, omegac, reorg, name):
        """
        Spectral density
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
            :math:`\Delta \sigma_x`
    
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
    
    
    def HEOM(self):
        sol = HEOMSolver(self.H, c_ops=[sz])
        return sol
    
    def TCL2(self):
        sol = TCL2(self.H, c_ops=[sz])


class TCL2:
    def __init__(self, H, c_ops, e_ops=None):
        self.H = H
        self.c_ops = c_ops
        self.e_ops = e_ops
        
    def run(self, dt, nt):
        pass
    
    def bath_correlation_function(self):
        pass
        
    
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
    sol = sbm.HEOM()
    
    nt = 100 
    rho = sol.run(rho0=rho0, dt=0.001, nt=nt, temperature=3e5, cutoff=5, reorganization=0.2, nado=5)
    print(rho)
        
        
    
    
    





