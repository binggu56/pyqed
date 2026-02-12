#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:12:08 2020

Third-order susceptibility for a generic multi-level system 

@author: Bing Gu 
"""
import numpy as np 
from phys import heaviside

def lineshape(a,b,t):
    return heaviside(t) * np.exp(-1j*(en[a]-en[b])*t - (decay[a]+decay[b])/2.*t) 
    

def G(a, b, omega):
    return 1./(omega - (en[a]-en[b]) + 1j * (decay[a]+decay[b])/2.0)
    
def response1_time(dip, t3, t2, t1):
    r = 0
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[0,b] * dip[b,c] * dip[c,d]* dip[d,0] * lineshape(d,c,t3) * \
                lineshape(d,b,t2) * lineshape(d,0,t1)
    return r 

def response1_freq(dip, omega3, t2, omega1):
    r = 0
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[0,b] * dip[b,c] * dip[c,d]* dip[d,0] * G(d,c,omega3) * \
                lineshape(d,b,t2) * G(d,0,omega1)
    return r 

def response2_freq(dip, omega3, t2, omega1):
    r = 0
    a = 0 # initial state 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(c, nlevel):
                r += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * G(d,c,omega3) * \
                lineshape(d,b,t2) * G(a,b,omega1)
    return r 

def response3_freq(dip, omega3, t2, omega1):
    r = 0
    a = 0 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * G(d,c,omega3) * \
                lineshape(a,c,t2) * G(a,b,omega1)
    return r 

def response4_freq(dip, omega3, t2, omega1):
    r = 0
    a = 0 # initial state, assuming ground state here 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[0,b] * dip[b,c] * dip[c,d]* dip[d,0] * G(d,a,omega3) * \
                lineshape(c,a,t2) * G(d, a,omega1)
    return r 

def response1_fd(omega3, omega2, omega1):
    """
    frequency domain response functions 
    """
    r = 0
    a = 0 # ground state 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[0,b] * dip[b,c] * dip[c,d]* dip[d,0] * G(d,c,omega3) * \
                G(d,b,omega2) * G(d,0,omega1)
    return r 

def response2_fd(omega3, omega2, omega1):
    r = 0
    a = 0 # initial state 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(c, nlevel):
                r += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * G(d,c,omega3) * \
                G(d,b,omega2) * G(a,b,omega1)
    return r 

def response3_fd(omega3, omega2, omega1):
    r = 0
    a = 0 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[a,b] * dip[b,c] * dip[c,d]* dip[d,a] * G(d,c,omega3) * \
                G(a,c,omega2) * G(a,b,omega1)
    return r 

def response4_fd(omega3, omega2, omega1):
    r = 0
    a = 0 # initial state, assuming ground state here 
    for b in range(nlevel):
        for c in range(nlevel):
            for d in range(nlevel):
                r += dip[0,b] * dip[b,c] * dip[c,d]* dip[d,0] * G(d,a,omega3) * \
                G(c,a, omega2) * G(d, a,omega1)
    return r 

def susceptibility(omega_in):
    """
    compute the susceptibility for a multilevel system 
    chi^3(-\omega)_s; omega_2, omega_2, omega_3) = -1/3! * \sum_{permutation of omega_1, omega_2, omega_3} \
        S^{3}(omega_1 + omega_2 + \omega_3, \omega_1 + \omega_2, \omega_1)
    
    signal frequency omega_s = \sum_{n=1}^3 \omega_n 
    
    S^{3} is the third-order response functions [Shaul's book, page 122]
    
    S^{(3)}(\omega_1 + \omega_2 + \omega_3, \omega_1 + \omega_2, \omega_1) = \
        -1 * \sum_{\alpha = 1}^4 R_\alpha(\omega_1 + \omega_2 + \omega_3, \omega_1 + \omega_2, \omega_1) + 
        R_^*\alpha( - \omega_1 - \omega_2 - \omega_3, - \omega_1 - \omega_2, -\omega_1)
    
    R_\alpha(\omega_1 + \omega_2 + \omega_3, \omega_1 + \omega_2, \omega_1): one of the \
        eight components of the third-order response functions (4 pair of complex conjugates)
    
    input: 
        mol: the material, multi-level system 
        omega_in: vector, frequencies of incoming pulses  
    
    """    
    if len(omega_in) != 3:
        sys.exit('The number of incoming pulses = {}, should be 3.'.format(len(omega_in)))
    
    omega1, omega2, omega3 = omega_in[:]
#    print('incoming frequencies = ', omega1, omega2, omega3)
    
    chi = response1_fd(omega1 + omega2 + omega3, omega1 + omega2, omega1) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega1).conjugate()
    
    chi += response2_fd(omega1 + omega2 + omega3, omega1 + omega2, omega1) + \
        response2_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega1).conjugate()
        
    chi += response3_fd(omega1 + omega2 + omega3, omega1 + omega2, omega1) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega1).conjugate()
        
    chi += response4_fd(omega1 + omega2 + omega3, omega1 + omega2, omega1) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega1).conjugate()
        
    # permute all incoming frequencies
    # 1, 2, 3 -> 2, 1, 3
    chi = response1_fd(omega1 + omega2 + omega3, omega1 + omega2, omega2) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega2).conjugate()
    
    chi += response2_fd(omega1 + omega2 + omega3, omega1 + omega2, omega2) + \
        response2_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega2).conjugate()
        
    chi += response3_fd(omega1 + omega2 + omega3, omega1 + omega2, omega2) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega2).conjugate()
        
    chi += response4_fd(omega1 + omega2 + omega3, omega1 + omega2, omega2) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega2).conjugate()    

    # 1, 2, 3 -> 3, 2, 1
    chi = response1_fd(omega1 + omega2 + omega3, omega3 + omega2, omega3) + \
        response1_fd( -omega1 - omega2 - omega3, - omega3 - omega2, -omega3).conjugate()
    
    chi += response2_fd(omega1 + omega2 + omega3, omega3 + omega2, omega3) + \
        response2_fd( -omega1 - omega2 - omega3, - omega3 - omega2, -omega3).conjugate()
        
    chi += response3_fd(omega1 + omega2 + omega3, omega3 + omega2, omega3) + \
        response1_fd( -omega1 - omega2 - omega3, - omega3 - omega2, -omega3).conjugate()
        
    chi += response4_fd(omega1 + omega2 + omega3, omega3 + omega2, omega3) + \
        response1_fd( -omega1 - omega2 - omega3, - omega3 - omega2, -omega3).conjugate()
        
    # 1, 2, 3 -> 1, 3, 2
    chi = response1_fd(omega1 + omega2 + omega3, omega1 + omega3, omega1) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega3, -omega1).conjugate()
    
    chi += response2_fd(omega1 + omega2 + omega3, omega1 + omega3, omega1) + \
        response2_fd( -omega1 - omega2 - omega3, - omega1 - omega3, -omega1).conjugate()
        
    chi += response3_fd(omega1 + omega2 + omega3, omega1 + omega3, omega1) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega3, -omega1).conjugate()
        
    chi += response4_fd(omega1 + omega2 + omega3, omega1 + omega3, omega1) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega3, -omega1).conjugate()
        
    # 1, 2, 3 -> 3, 1, 2
    chi = response1_fd(omega1 + omega2 + omega3, omega3 + omega1, omega3) + \
        response1_fd( -omega1 - omega2 - omega3, - omega3 - omega1, -omega3).conjugate()
    
    chi += response2_fd(omega1 + omega2 + omega3, omega3 + omega1, omega3) + \
        response2_fd( -omega1 - omega2 - omega3, - omega3 - omega1, -omega3).conjugate()
        
    chi += response3_fd(omega1 + omega2 + omega3, omega3 + omega1, omega3) + \
        response1_fd( -omega1 - omega2 - omega3, - omega1 - omega2, -omega1).conjugate()
        
    chi += response4_fd(omega1 + omega2 + omega3, omega3 + omega1, omega3) + \
        response1_fd( -omega1 - omega2 - omega3, - omega3 - omega1, -omega3).conjugate()

    # 1, 2, 3 -> 2, 3, 1
    chi = response1_fd(omega1 + omega2 + omega3, omega2 + omega3, omega2) + \
        response1_fd( -omega1 - omega2 - omega3, - omega2 - omega3, -omega2).conjugate()
    
    chi += response2_fd(omega1 + omega2 + omega3, omega2 + omega3, omega2) + \
        response2_fd( -omega1 - omega2 - omega3, - omega2 - omega3, -omega2).conjugate()
        
    chi += response3_fd(omega1 + omega2 + omega3, omega2 + omega3, omega2) + \
        response1_fd( -omega1 - omega2 - omega3, - omega2 - omega3, -omega2).conjugate()
        
    chi += response4_fd(omega1 + omega2 + omega3, omega2 + omega3, omega2) + \
        response1_fd( -omega1 - omega2 - omega3, - omega2 - omega3, -omega2).conjugate()
        
    chi *= 1./6.
    
    return chi 