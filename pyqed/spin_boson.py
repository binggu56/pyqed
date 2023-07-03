import numpy as np
import sys

class Spin_boson(object):
    def __init__(self, beta, omegac=1.0, reorg=2.0):
        self.beta = beta # temperature
        self.omegac = omegac
        self.reorg = reorg

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





