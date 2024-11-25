#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:00:40 2024


Prony decomposition of time-series data

@author: Zi-Hao Chen 


"""



import math
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt


def fit_J(w, res, expn, etal, sigma):
    for i in range(len(etal)):
        res += etal[i] / (expn[i] + sigma * 1.j * w)


def fit_t(t, expn, etal):
    res = 0
    for i in range(len(etal)):
        res += etal[i] * np.exp(-expn[i] * t)
    return res





# fft_ct = np.exp(-fft_t)

def prony_decomposition(x, fft_ct, nexp):
    """
    decompose a function into a sum of exponentials 
    
    Refs
        Zi-Hao's thesis
        
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    fft_ct : TYPE
        DESCRIPTION.
    nexp : TYPE
        DESCRIPTION.

    Returns
    -------
    etal1 : TYPE
        DESCRIPTION.
    expn1 : TYPE
        DESCRIPTION.
    err : TYPE
        DESCRIPTION.
        

    """
    
    n = (len(x)-1)//2
    
    n_sample = n + 1 
    n_gamma_l2 = [nexp] # number of exponentials
    
    h = np.real(fft_ct)
    H = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        H[i, :] = h[i:n_sample + i]
    sing_vs, Q = LA.eigh(H)
    
    # del H
    phase_mat = np.diag(
        [np.exp(-1j * np.angle(sing_v) / 2.0) for sing_v in sing_vs])
    vs = np.array([np.abs(sing_v) for sing_v in sing_vs])
    Qp = np.dot(Q, phase_mat)
    sort_array = np.argsort(vs)[::-1]
    vs = vs[sort_array]
    Qp = (Qp[:, sort_array])
    
    nroots = 20
    # vs = vs[:20]
    # Qp = Qp[:, :20]
    
    vs = vs[:nroots]
    Qp = Qp[:, :nroots]
    
    for n_gamma in n_gamma_l2:
        
        print("len of gamma", n_gamma)
        
        gamma = np.roots(Qp[:, n_gamma][::-1])
        gamma_new = gamma[np.argsort(np.abs(gamma))[:n_gamma]]
        t_real = 2 * n * np.log(gamma_new)
        gamma_m = np.zeros((n_sample * 2 - 1, n_gamma), dtype=complex)
        for i in range(n_gamma):
            for j in range(n_sample * 2 - 1):
                gamma_m[j, i] = gamma_new[i]**j
        omega_real = np.dot(LA.inv(np.dot(np.transpose(gamma_m), gamma_m)),
                            np.dot(np.transpose(gamma_m), np.transpose(h)))
    
        # res_t = np.zeros(len(t), dtype=complex)
        # fit_t(fft_t, res_t, -t_real / scale, omega_real)
        # plt.plot(fft_t, np.real(fft_ct) - res_t)
        # plt.savefig("real_{}.pdf".format(n_gamma))
        # plt.clf()
    
    
    etal1 = omega_real
    expn1 = -t_real / scale
    
    # fft_t = x 
    plt.plot(x, np.real(fft_ct))
    
    plt.plot(x, fft_ct, label='Exact')
    # plt.plot(fft_t, fft_ct.imag)
    
    # res_t = np.zeros(len(fft_t), dtype=complex)
    res_t = fit_t(x, expn1, etal1)
    
    plt.plot(x, res_t, '--', label='Fit')
    # plt.plot(fft_t, res_t.imag, '--')
    
    plt.xlim(0, 1)
    plt.legend()
    
    err = (np.abs(fft_ct - res_t) ** 2).sum()/len(x) # sum of squared residues
    
    return etal1, expn1, err

if __name__=='__main__':
    
    n = 4000 # 2N + 1 points 
    scale = 20 # range [0, 80]
    
    # scale_fft = 1000
    # n_fft = 10000000
    
    # n_rate = (scale_fft * scale/ (4 * n)) # print data every n_rate points
    # print(n_rate)
    # n_rate = int(n_rate)
    
    # w = np.linspace(0, scale_fft * np.pi, n_fft + 1)[:-1]
    # dw = w[1] - w[0]
    # print(dw)
    
    # fft_t = 2 * np.pi * np.fft.fftfreq(len(w), dw)
    # fft_t = fft_t[(scale>=fft_t) & (fft_t >= 0)][::n_rate]
    
    # fft_ct = 1 / fft_t
    
    # fft_ct[0] = fft_ct[1]
    
    
    x = np.linspace(0, scale, 2*n+1)
    fft_ct = 1/x
    fft_ct[0] = fft_ct[1]
    
    etal1, expn1, err = prony_decomposition(x, fft_ct, 15)
    
    print(err)


# plt.plot(fft_t, fft_ct - res_t)




