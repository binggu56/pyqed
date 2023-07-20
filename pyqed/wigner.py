#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 22:23:15 2021

@author: bing

@Source:
https://www.frank-zalkow.de/en/the-wigner-ville-distribution-with-python.html

"""
from __future__ import absolute_import

# from time import clock
from numpy import abs, arange, shape, array, ceil, zeros, conj, ix_,\
 transpose, append, real, float64, linspace, sqrt, pi

import numpy as np
from scipy.signal import hilbert
from scipy import interpolate
from scipy.fft import fft, fftfreq, fftshift, ifft
from math import log, ceil, floor
import sys

import pyqed

import matplotlib.pyplot as plt

def nextpow2(p):
    n = 2
    while p < n:
        n *= 2
    return n

def wvd(audioFile, t=None, N=None, trace=0, make_analytic=True):
    # if make_analytic:
    #     x = hilbert(audioFile[1])
    # else:
    #     x = array(audioFile[1])

    x = array(audioFile)
    if x.ndim == 1:
        [xrow, xcol] = shape(array([x]))
    else: raise ValueError("Signal x must be one-dimensional.")

    if t is None: t = arange(len(x))
    if N is None: N = len(x)

    if (N <= 0 ):
        raise ValueError("Number of Frequency bins N must be greater than zero.")

    if t.ndim == 1:
        [trow, tcol] = shape(array([t]))
    else:
        raise ValueError("Time indices t must be one-dimensional.")

    if xrow != 1:
        raise ValueError("Signal x must have one row.")
    elif trow != 1:
        raise ValueError("Time indicies t must have one row.")
    # elif nextpow2(N) != N:
    #     print("For a faster computation, number of Frequency bins N should be a power of two.")

    tfr = zeros([N, tcol], dtype='complex')
    # if trace: print "Wigner-Ville distribution",
    for icol in range(0, tcol):

        ti = t[icol]

        taumax = min([ti, xcol-ti-1, int(round(N/2))-1])

        tau = arange(-taumax, taumax+1)

        indices = ((N+tau)%N)
        tfr[ix_(indices, [icol])] = transpose(array(x[ti+tau] * conj(x[ti-tau]), ndmin=2))
        tau=int(round(N/2))+1
        if ((ti+1) <= (xcol-tau)) and ((ti+1) >= (tau+1)):
            if(tau >= tfr.shape[0]): tfr = append(tfr, zeros([1, tcol]), axis=0)
            tfr[ix_([tau], [icol])] = array(0.5 * (x[ti+tau] * conj(x[ti-tau]) + x[ti-tau] * conj(x[ti+tau])))
        # if trace: disprog(icol, tcol, 10)

    tfr = real(fft.fft(tfr, axis=0))
    f = 0.5*arange(N)/float(N)
    return transpose(tfr), t, f


# def wigner(signal):
#     """
#     Wigner transform of an input signal with FFT.
#     W(t, w) = int dtau x(t + tau/2) x^*(t - tau/2) e^{i w tau}

#     Parameters
#     ----------
#     x : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.
#     TYPE
#         DESCRIPTION.
#     TYPE
#         DESCRIPTION.

#     """

#     N = len(signal)
#     tausec = N//2
#     winlength = tausec - 1

#     taulens = np.min(np.c_[np.arange(N),
#                             N - np.arange(N) - 1,
#                       winlength * np.ones(N)], axis=1)

#     # complex conjugate
#     conj_signal = np.conj(signal)

#     # the wigner function, axis 0 is the new axis
#     tfr = zeros((N, N), dtype=complex)

#     for icol in range(N):

#         taumax = taulens[icol]

#         tau = np.arange(-taumax, taumax + 1).astype(int)

#         indices = np.remainder(N + tau, N).astype(int)
#         # print(tau)
#         # print(indices)
#         # if icol == 2: sys.exit()

#         tfr[indices, icol] = signal[icol + tau] * conj_signal[icol - tau]


#         # if (icol <= N - tausec) and (icol >= tausec + 1):
#         #     tfr[tausec, icol] = signal[icol + tausec, 0] * \
#         #         np.conj(signal[icol - tausec, 0]) + \
#         #         signal[icol - tausec, 0] * conj_signal[icol + tausec, 0]

#     fig, ax = plt.subplots()
#     ax.matshow(tfr.real)

#     tfr = np.real(fft(tfr, axis=0))

#     # freqs = 0.5 * np.arange(N, dtype=float) / N
#     freqs = fftfreq(N)

#     return tfr, freqs

def spectrogram(x, d=1):
    """
    Wigner transform of an input signal with FFT.

    W(w, t) = \int d\tau x(t + \tau/2) x^*(t - \tau/2) e^{i w tau}

    Parameters
    ----------
    x : 1d array
        The time-domain signal.

    d : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        spectrogram in (f, t)
    freqs : 1d array
        sample frequencies.

    """

    N = len(x)
    tausec = N//2

    # taus = np.linspace(-N//2*dt)
    winlength = tausec - 1

    taulens = np.min(np.c_[np.arange(N),
                            N - np.arange(N) - 1,
                      winlength * np.ones(N)], axis=1)

    taus = np.arange(-tausec, tausec) * d

    # complex conjugate
    xc = np.conj(x)

    # the wigner function, axis 0 is the new axis
    w = zeros((N, N), dtype=complex)

    for j in range(N):

        taumax = taulens[j]

        tau = np.arange(-taumax, taumax + 1).astype(int)

        indices = (tau + tausec).astype(int)

        # if j == 2: sys.exit()

        w[indices, j] = x[j + tau] * xc[j - tau]


        # if (icol <= N - tausec) and (icol >= tausec + 1):
        #     tfr[tausec, icol] = signal[icol + tausec, 0] * \
        #         np.conj(signal[icol - tausec, 0]) + \
        #         signal[icol - tausec, 0] * conj_signal[icol + tausec, 0]
        w[:, j], freqs = pyqed.fft.ifft(w[:, j], taus)

    return w, freqs/2


def wigner(x, d=1):
    """
    Wigner transform of an input signal with FFT.
    W(t, w) = int dtau x(t + tau/2) x^*(t - tau/2) e^{i w tau}

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    N = len(x)
    tausec = N//2

    # taus = np.linspace(-N//2*dt)
    winlength = tausec - 1

    taulens = np.min(np.c_[np.arange(N),
                            N - np.arange(N) - 1,
                      winlength * np.ones(N)], axis=1)

    taus = np.arange(-tausec, tausec) * d

    # complex conjugate
    xc = np.conj(x)

    # the wigner function, axis 0 is the new axis
    w = zeros((N, N), dtype=complex)

    for j in range(N):

        taumax = taulens[j]

        tau = np.arange(-taumax, taumax + 1).astype(int)

        indices = (tau + tausec).astype(int)

        # if j == 2: sys.exit()

        w[indices, j] = x[j + tau] * xc[j - tau]


        # if (icol <= N - tausec) and (icol >= tausec + 1):
        #     tfr[tausec, icol] = signal[icol + tausec, 0] * \
        #         np.conj(signal[icol - tausec, 0]) + \
        #         signal[icol - tausec, 0] * conj_signal[icol + tausec, 0]
        freqs, w[:, j] = pyqed.fft.fft(w[:, j], taus)

    # fig, ax = plt.subplots()
    # ax.matshow(w.real)

    # tfr = np.real(fft(w, axis=0))

    # freqs = 0.5 * np.arange(N, dtype=float) / N
    # freqs = pi * fftfreq(N, d=d)
    return np.real(w.T), freqs

if __name__=='__main__':

    from pyqed.optics import Pulse, interval
    from pyqed.units import au2fs, au2ev
    #from pyqed.phys import

    pulse = Pulse(tau=4/au2fs, omegac=2/au2ev, beta=0.1)
    sigma=4/au2fs
    omegac=2/au2ev
    t = np.linspace(-14, 14, 256)/au2fs
    N = len(t)

    efield = pulse.efield(t)
    # efield = np.exp(-t**2/sigma**2-0.1j*omegac*t**2)
    dt = interval(t)

    # w = wvd(efield.real, N=None, trace=0, make_analytic=False)[0]
    # w = wigner(efield)
    wvd, freqs = spectrogram(efield, dt)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(wvd)

    from pyqed.style import imshow

    imshow(freqs, t, wvd.T.real, xlabel=r'$\omega$', ylabel='$t$')

    # from pyqed.style import surf
    # ax = surf(wvd, t, freqs)
    # ax.set_ylim(-0.1, 0)
