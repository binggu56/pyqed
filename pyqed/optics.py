#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:26:02 2019

@author: binggu
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, kron, identity, linalg
from numpy import sqrt, exp, pi
import proplot as plt

from pyqed.units import au2k, au2ev, alpha, \
    au2watt_per_centimeter_squared, au2fs
from pyqed.fft import fft, fft2
from pyqed.phys import rect, sinc, dag, interval
from pyqed.wigner import spectrogram

from numba import jit

def intensity_to_field(I):
    """
    transform intensity to electric field

    E = \sqrt(\frac{2I}{c \epsilon_0})

    Parameters
    ----------
    I : TYPE
        intensity, in units of W/cm^2

    Returns
    -------
    None.
        electric field in atomic units

    """
    return np.sqrt(2.* I * 4. * np.pi/au2watt_per_centimeter_squared/alpha)

def field_to_intensity(E):
    pass

def std_to_fwhm(tau):
    """
    Transformation between standard deviation to FWHM for a Gaussian pulse

    Parameters
    ----------
    tau : float
        std

    Returns
    -------
    float
        FWHM.

    """
    return (2.* np.sqrt(2. * np.log(2.))) * tau

def fwhm_to_std(fwhm):
    return fwhm/(2.* np.sqrt(2. * np.log(2.)))


class Analyser:
    def __init__(self, E, t):
        self.E = E
        self.t = t
        self.dt = interval(t)
        self.I = None # Wigner transform of the signal I(\omega, t)
        self.omegas = None # frequency domain

    def FROG(self, w=None, use_fft=False):
        # FROG spectrogram of the field
        # I(\omega, \tau) = \abs{ \int_{-\infty}^\infty E(t) E(t-\tau) e^{-i\omega t} dt}^2

        E = self.E
        N = len(E)

        # I, J = np.meshgrid(np.arrange(N), np.arange(N))
        Esig = np.zeros((N, N))
        for i in range(N):
            tau = np.arange(i, dtype=int)
            Esig[i, tau] = E[i] * E[i-tau]

        if use_fft:
            I, w = fft(Esig, self.t)

            return np.abs(I)**2, w
        else:
            I = Esig.dot(np.exp(-1j * np.outer(w, t)).T)

            return np.abs(I)**2

    def spectrogram(self):
        Ew, w = spectrogram(self.E, self.dt)

        self.I = Ew
        self.omegas = w

        return Ew, w

    def plot_spectrogram(self):
        fig, ax = plt.subplots()
        ax.contourf(self.t*au2fs, self.omegas*au2ev, np.abs(self.I)**2, origin='lower')
        ax.format(xlabel='Time (fs)', ylabel='Frequency (eV)', ylim=(0, None))
        return fig, ax


class Pulse:
    def __init__(self, omegac=3./au2ev, tau=5./au2fs, tc=0, delay=0., \
                 amplitude=0.001, intensity=None, cep=0., beta=0, polarization=None):
        """
        (linearly chirped) Gaussian pulse

        The (approximate) positive frequency component of the electric field reads

        ..math::
            E^{(+)} = \frac{1}{2} * A  exp(-(t-t0)^2/2/T^2) * exp[-i w (t-t0)]

        This is only true when the bandwidth is much smaller than the central frequency.

        A: electric field amplitude
        T: time delay
        sigma: duration
        intensity: if intensity is given, it will overwrite the amplitude

        DEPRECATED, Use GaussianPulse instead.
        """
        self.delay = delay # deprecated, use tc
        self.tc = tc

        self.tau = tau
        self.fwhm = tau * 2.3548200450309493 # from tau to fwhm
        self.sigma = tau # for compatibility only
        self.omegac = omegac # central frequency
        self.unit = 'au'
        
        if intensity is None:
            self.amplitude = amplitude
        else:
            self.amplitude = intensity_to_field(intensity)
            
        self.cep = cep
        self.bandwidth = 1./tau
        self.duration = 2. * tau
        self.beta = beta  # linear chirping rate, dimensionless
        self.ndim = 1
        self.polarization = polarization


    def envelop(self, t):
        tc = self.tc
        return self.amplitude * np.exp(-(t-tc)**2/2./self.tau**2)

    def spectrum(self, omega):
        """
        Fourier transform of the Gaussian pulse
        """
        omega0 = self.omegac
        tau = self.tau
        A0 = self.amplitude

        return A0 * tau * np.sqrt(2.*np.pi) * np.exp(-(omega-omega0)**2 * tau**2/2.)

    def field(self, t):
        '''
        electric field
        '''
        return self.efield(t)

    def efield(self, t, return_complex=False):
        """

        Parameters
        ----------
        t : TYPE
            time.

        Returns
        -------
        complex
            complex-valued  electric field at time t. Take the real part for the
            electric field.

        """
        omegac = self.omegac
        t0 = self.tc
        a = self.amplitude
        tau = self.sigma


        # E = a * np.exp(-(t-t0)**2/2./tau**2)*np.exp(-1j * omegac * (t-t0))\
        #     * np.exp(-1j * beta * omegac * (t-t0)**2/tau)

        E = a * np.exp(-(t-t0)**2/2./tau**2) * np.exp(-1j * omegac * (t-t0))

        return np.real(E)

    def E(self, t):
        omegac = self.omegac
        t0 = self.tc
        a = self.amplitude
        tau = self.sigma
        pol = self.polarization

        F = a * pol * np.exp(-(t-t0)**2/2./tau**2)*np.exp(-1j * omegac * (t-t0))

        return np.real(F)

    def spectrogram(self, efield):

        # from tftb.processing import WignerVilleDistribution

        # wvd = WignerVilleDistribution(z)
        # w, ts, fs = wvd.run()
        return

    def plt_efield(self):

        import proplot as plt
        from pyqed.units import au2volt_per_angstrom

        fig, ax = plt.subplots()

        t = np.linspace(self.tc - 2.*self.fwhm, self.tc + 2 * self.fwhm, 100)

        ax.plot(t*au2fs, self.efield(t) * au2volt_per_angstrom)
        ax.format(xlabel='Time (fs)', ylabel='$E(t)$')

        return

class GaussianPulse:
    def __init__(self, omegac=3./au2ev, tau=5./au2fs, tc=0, delay=0., \
                 amplitude=0.001, cep=0., beta=0, polarization=None):
        """
        (linearly chirped) Gaussian pulse

        The positive frequency component reads

        E = Re A * exp(-(t-t_c)^2/2/\tau^2) * e^{-i w (t-t_c)}

        A: electric field amplitude
        T: time delay
        sigma: duration

        """
        self.delay = delay # deprecated, use tc
        self.tc = tc

        self.tau = tau
        self.fwhm = tau * 2.3548200450309493 # from tau to fwhm
        self.sigma = tau # for compatibility only
        self.omegac = omegac # central frequency
        self.unit = 'au'
        self.amplitude = amplitude
        self.cep = cep
        self.bandwidth = 1./tau
        self.duration = 2. * tau
        self.beta = beta  # linear chirping rate, dimensionless
        self.ndim = 1



    def envelop(self, t):
        tc = self.tc
        return self.amplitude * np.exp(-(t-tc)**2/2./self.tau**2)

    def spectrum(self, omega):
        """
        Fourier transform of the Gaussian pulse
        """
        omega0 = self.omegac
        tau = self.tau
        A0 = self.amplitude

        return A0 * tau * np.sqrt(2.*np.pi) * np.exp(-(omega-omega0)**2 * tau**2/2.)

    def field(self, t):
        '''
        electric field
        '''
        return self.efield(t)

    def efield(self, t):
        """

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.

        Returns
        -------
        electric field at time t.

        """
        omegac = self.omegac
        t0 = self.tc
        a = self.amplitude
        tau = self.sigma
        beta = self.beta

        return a * np.exp(-(t-t0)**2/2./tau**2)*np.cos(omegac * (t-t0))


        # E = a * np.exp(-(t-t0)**2/2./tau**2)*np.exp(-1j * omegac * (t-t0))\
        #     * np.exp(-1j * beta * omegac * (t-t0)**2/tau)

        # return E.real

    def spectrogram(self, efield):

        # from tftb.processing import WignerVilleDistribution

        # wvd = WignerVilleDistribution(z)
        # w, ts, fs = wvd.run()
        return

    def plt_efield(self):

        import proplot as plt
        from lime.units import au2volt_per_angstrom

        fig, ax = plt.subplots()

        t = np.linspace(self.tc - 2.*self.fwhm, self.tc + 2 * self.fwhm, 100)

        ax.plot(t*au2fs, self.efield(t) * au2volt_per_angstrom)
        ax.format(xlabel='Time (fs)', ylabel='$E(t)$')

        return

class ChirpedPulse(Pulse):
    def __init__(self, omegac=3./au2ev, tau=5./au2fs, tc=0, amplitude=0.001, cep=0., beta=0):
        """
        (linearly chirped) Gaussian pulse

        The positive frequency component reads

        E = A/2 * exp(-(t-t0)^2/2/T^2) * exp[-i w (t-t0)(1 + beta (t-t0)/T)]

        A: electric field amplitude
        T: time delay
        sigma: duration

        """
        self.tc = tc

        self.tau = tau
        self._fwhm = tau * 2.3548200450309493 # from tau to fwhm
        self.sigma = tau # for compatibility only
        self.omegac = omegac # central frequency
        self.unit = 'au'
        self.amplitude = amplitude
        self.cep = cep
        self.bandwidth = 1./tau
        self.duration = 2. * tau
        self.beta = beta  # linear chirping rate, dimensionless
        self.ndim = 1

    def efield(self, t):
        """

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.

        Returns
        -------
        electric field at time t.

        """
        omegac = self.omegac
        t0 = self.tc
        a = self.amplitude
        tau = self.sigma
        beta = self.beta
#
#        if beta is None:
#            return a * np.exp(-(t-delay)**2/2./sigma**2)*np.cos(omegac * (t-delay))
#        else:

        E = a * np.exp(-(t-t0)**2/2./tau**2)*np.exp(-1j * omegac * (t-t0))\
            * np.exp(-1j * beta * omegac * (t-t0)**2/tau)

        return E.real

    def spectrum(self, omega):
        """
        Fourier transform of the Gaussian pulse
        """
        omega0 = self.omegac
        T = self.tau
        A0 = self.amplitude
        beta = self.beta

#        if beta is None:
#            return A0 * sigma * np.sqrt(2.*np.pi) * np.exp(-(omega-omega0)**2 * sigma**2/2.)
#        else:
        a = 0.5/T**2 + 1j * beta * omega0/T
        return A0 * np.sqrt(np.pi/a) * np.exp(-(omega - omega0)**2/4./a)

    # def set_fwhm(self, fwhm):
    #     self.tau = fwhm/(2.* np.sqrt(2. * np.log(2.)))
    #     return

    # @property
    # def fwhm(self):
    #     return self._fwhm


# def heaviside(x):
#     """
#     Heaviside function defined in a grid.
#       returns 0 if x<=0, and 1 if x>0
#     """
#     x = np.asarray(x)
#     y = np.zeros(x.shape)
#     y[x > 0] = 1.0
#     return y


class Biphoton:
    def __init__(self, omegap, bw, Te, p=None, q=None, phase_matching='sinc'):
        """
        Class for entangled photon pair.
        Parameters
        ----------
        omegap: float
            pump carrier frequency
        bw: float
            pump bandwidth
        p: signal grid
        q: idler grid
        phase_matching: str
            type of phase matching. Default is 'sinc'. A narrowband approxmation is invoked.
        """
        self.omegap = omegap
        self.pump_bandwidth = bw
        self.phase_matching = phase_matching
        self.signal_center_frequency = omegap / 2.
        self.idler_center_frequency = omegap / 2.
        self.entanglement_time = Te
        self.jsa = None
        self.jta = None
        self.p = p
        self.q = q
        if p is not None:
            self.dp = interval(p)
            self.dq = interval(q)
        self.grid = [p, q]

    def pump(self, bandwidth):
        """
        pump pulse envelope
        Parameters
        ----------
        bandwidth

        Returns
        -------

        """
        alpha = np.sqrt(1. / (np.sqrt(2. * np.pi) * bandwidth)) * \
                np.exp(-(p + q) ** 2 / 4. / bandwidth ** 2)
        return alpha

    def set_grid(self, p, q):
        self.p = p
        self.q = q
        return

    def get_jsa(self):
        """

        Returns
        -------
        jsa: array
            joint spectral amplitude

        """
        p = self.p
        q = self.q
        bw = self.pump_bandwidth

        self.jsa = _jsa(p, q, bw, model=self.phase_matching,
                          Te=self.entanglement_time)
        return self.jsa

    def get_jta(self):
        """
        Compute the joint temporal amplitude J(ts, ti) over a temporal meshgrid.

        Returns
        -------
        ts: 1d array
            signal time grid
        ti: 1d array
            idler temporal grid
        jta: 2d array
            joint temporal amplitude
        """
        p = self.p
        q = self.q
        dp = p[1] - p[0]
        dq = q[1] - q[0]
        if self.jsa is not None:

            ts, ti, jta = fft2(self.jsa, dp, dq)
            self.jta = jta
            return ts, ti, jta

        else:
            raise ValueError('jsa is None. Call get_jsa() first.')

    def jta(self, ts, ti):
        return

    def detect(self):
        """
        two-photon detection amplitude in a temporal grid defined by
        the spectral grid.

        Returns
        -------
        t1: 1d array
        t2: 1d array
        d: detection amplitude in the temporal grid (t1, t2)

        """

        if self.jsa is None:
            raise ValueError('Please call get_jsa() to compute the jsa first.')

        bw = self.pump_bandwidth
        omega_s = self.signal_center_frequency
        omega_i = self.idler_center_frequency
        p = self.p
        q = self.q
        dp = p[1] - p[0]
        dq = q[1] - q[0]
        return _detection_amplitude(self.jsa, omega_s, omega_i, dp, dq)

    def detect_si(self):
        pass

    def detect_is(self):
        pass

    def g2(self):
        pass

    def bandwidth(self, which='signal'):
        """
        Compute the bandwidth of the signal/idler mode

        Parameters
        ----------
        which : TYPE, optional
            DESCRIPTION. The default is 'signal'.

        Returns
        -------
        None.

        """
        p, q = self.p, self.q
        dp = interval(p)
        dq = interval(q)

        f = self.jsa

        if which == 'signal':
            rho = rdm(f, dq, which='x')
            sigma = sqrt(rho.diagonal().dot(p**2) * dp)

        elif which == 'idler':

            rho = rdm(f, dp, which='y')

            sigma = sqrt(rho.diagonal().dot(q**2) * dq)

        return sigma

    def plt_jsa(self, xlabel=None, ylabel=None, fname=None):

        if self.jsa is None:
            self.get_jsa()

        plt, ax = imshow(self.p * au2ev, self.q * au2ev, np.abs(self.jsa))

        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_xlabel(ylabel)


        if fname is not None:
            plt.savefig(fname)

        plt.show()

        return ax

    def rdm(self, which='signal'):
        if which == 'signal':
            return rdm(self.jsa, dy=self.dq, which='x')


def jta(t2, t1, omegap, sigmap, Te):
    """
    Analytical form for the joint temporal amplitude for SPDC type-II
    two-photon state.

    Note that two single-photon electric field prefactors are neglected.

    Parameters
    ----------
    t2 : TYPE
        DESCRIPTION.
    t1 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    omegas = omegap/2.
    omegai = omegap/2.

    tau = t2 - t1
    amp = sqrt(sigmap/Te) * (2.*pi)**(3./4) * \
        rect(tau/2./Te) * exp(-sigmap**2*(t1+t2)**2/4.) *\
            exp(-1j * omegas * t1 - 1j*omegai * t2)

    return amp


def rdm(f, dx=1, dy=1, which='x'):
    '''
    Compute the reduced density matrix by tracing out the other dof for a 2D wavefunction

    Parameters
    ----------
    f : 2D array
        2D wavefunction
    dx : float, optional
        DESCRIPTION. The default is 1.
    dy : float, optional
        DESCRIPTION. The default is 1.
    which: str
        indicator which rdm is required. Default is 'x'.

    Returns
    -------
    rho1 : TYPE
        Reduced density matrix

    '''
    if which == 'x':
        rho = f.dot(dag(f)) * dy
    elif which == 'y':
        rho = f.T.dot(np.conj(f)) * dx
    else:
        raise ValueError('The argument which can only be x or y.')
    return rho


def _jsa(p, q, pump_bw, model='sinc', Te=None):
    '''
    Construct the joint spectral amplitude

    Parameters
    ----------
    p : 1d array
        signal frequency (detuning from the center frequency)
    q : 1d array
        idler frequency
    pump_bw : float
        pump bandwidth
    sm : float
        1/entanglement time
    Te : float
        Entanglement time.

    Returns
    -------
    jsa : TYPE
        DESCRIPTION.

    '''
    P, Q = np.meshgrid(p, q)
    sigma_plus = pump_bw
    sigma_minus = 1. / Te

    # pump envelope
    alpha = np.sqrt(1. / (np.sqrt(2. * np.pi) * sigma_plus)) * \
            np.exp(-(P + Q) ** 2 / 4. / sigma_plus ** 2)

    # phase-matching function

    if model == 'Gaussian':
        beta = np.sqrt(1. / np.sqrt(2. * np.pi) / sigma_minus) * \
               np.exp(-(P - Q) ** 2 / 4. / sigma_minus ** 2)

        jsa = sqrt(2) * alpha * beta

    elif model == 'sinc':

        beta = sqrt(0.5 * Te / np.pi) * sinc(Te * (P - Q) / 4.)

        # const =  np.trace(dag(f).dot(f))*dq*dp

        jsa = alpha * beta

    return jsa





def hom(p, q, f, tau):
    """
    HOM coincidence probability

    Parameters
    ----------
    p
    q
    f
    tau
    method: str
        "brute": directly integrating the JSA over the frequency grid
        "schmidt": compute the signal using the Schmidt modes of the
            entangled light
    nmodes

    Returns
    -------
    prob: 1d array
        coincidence probability

    """
    dp = interval(p)
    dq = interval(q)

    P, Q = np.meshgrid(p, q)

    prob = np.zeros(len(tau))

    for j in range(len(tau)):
        t = tau[j]
        prob[j] = 0.5 - 0.5 * np.sum(f.conj() * f.T *
                                     np.exp(1j * (P - Q) * t)).real * dq*dp

    return prob


def hom_schmidt(p, q, f, method='rdm', nmodes=5):
    """
    HOM signal with Schmidt modes

    Parameters
    ----------
    p
    q
    f
    nmodes

    Returns
    -------

    """
    dp = interval(p)
    dq = interval(q)

    # schmidt decompose the JSA
    s, phi, chi = schmidt_decompose(f, dp, dq, method=method,
                                    nmodes=nmodes)

    prob = np.zeros(len(tau))

    for j in range(len(tau)):
        t = tau[j]

        for a in range(nmodes):
            for b in range(nmodes):

                tmp1 = (phi[:,a].conj() * chi[:, b] * np.exp(1j * p * t)).sum() * dp
                tmp2 = (phi[:,b] * chi[:, a].conj() * np.exp(-1j * q * t)).sum() * dq

                prob[j] += -2. * np.real(s[a] * s[b] * tmp1 * tmp2)

    prob = 0.5 + prob/4.
    return prob


def schmidt_decompose(f, dp, dq, nmodes=5, method='rdm'):
    """
    kernel method
    f: 2D array,
        input function to be decomposed
    nmodes: int
        number of modes to be kept
    method: str
        rdm or svd
    """
    if method == 'rdm':
        kernel1 = f.dot(dag(f)) * dq * dp
        kernel2 = f.T.dot(f.conj()) * dp * dq


        print('c: Schmidt coefficients')
        s, phi = np.linalg.eig(kernel1)
        s1, psi = np.linalg.eig(kernel2)

        phi /= np.sqrt(dp)
        psi /= np.sqrt(dq)
    elif method == 'svd':
        raise NotImplementedError

    return np.sqrt(s[:nmodes]), phi[:, :nmodes], psi[:, :nmodes]

def _detection_amplitude(jsa, omega1, omega2, dp, dq):
    '''
    Detection amplitude <0|E(t)E(t')|Phi>
    t, t' are defined on a 2D grid used in the FFT,
    E(t) = Es(t) + Ei(t) is the total electric field operator.
    This contains two amplitudes corresponding to two different
    ordering of photon interactions
        <0|T Ei(t)Es(t')|Phi> + <0|T Es(t)Ei(t')|Phi>

    The t, t' are defined relative to t0, i.e, they are temporal durations from t0.

    Parameters
    ----------
    jsa : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    omega1 : float
        central frequency of signal beam
    omega2 : float
        central frequency of idler beam

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    '''

    t1, t2, jta = fft2(jsa, dp, dq)

    dt2 = t2[1] - t2[0]

    T1, T2 = np.meshgrid(t1, t2)

    # detection amplitude d(t1, t2) ~ JTA(t2, t1)
    d = np.exp(-1j * omega2 * T1 - 1j * omega1 * T2) * \
        np.sqrt(omega1 * omega2) * jta.T + \
        np.exp(-1j * omega1 * T1 - 1j * omega2 * T2) * \
        np.sqrt(omega1 * omega2) * jta

    #   amp = np.einsum('ij, ij -> i', d, heaviside(T1 - T2) * \
    #                   np.exp(-1j * gap20 * (T1-T2))) * dt2

    return t1, t2, d


if __name__ == '__main__':

    def test_biphoton():
        from lime.units import au2ev, au2fs

        p = np.linspace(-2, 2, 128) / au2ev
        q = p
        epp = Biphoton(omegap=3 / au2ev, bw=0.2 / au2ev, Te=10/au2fs,
                       p=p, q=q)

        JSA = epp.get_jsa()

        # epp.plt_jsa()
        # t1, t2, d = epp.detect()
        tau = np.linspace(-10, 10)/au2fs

        prob = hom(p, q, JSA, tau)

        fig, ax = plt.subplots()
        ax.plot(tau, prob)

        plt.show()

    # pulse = GaussianPulse()
    pulse = Pulse(omegac=4/au2ev, tau=8/au2fs, beta=0.2)

    t = np.linspace(-12, 12, 256)/au2fs

    analyser = Analyser(pulse.efield(t), t)

    I, w = analyser.spectrogram()
    analyser.plot_spectrogram()



