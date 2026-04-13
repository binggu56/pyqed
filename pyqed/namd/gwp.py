#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:32:54 2026

simultaneous diagonalization of randomly distributed multidimensional Gaussian wavepackets 


@author: Bing Gu (gubing@westlake.edu.cn)

"""


import numpy as np
from numpy import exp, pi, sqrt
from pyqed import transform, dag, isunitary, rk4, isdiag


import scipy
from scipy.linalg import inv
from scipy.sparse import kron, eye


class CGWP:
    def __init__(self, q, p=0, a=1, phase=0, ndim=1):
        """
        normalized complex multidimensional Gaussian wavepackets
        .. math::
            
            g(x) = N e^{- 1/2 (x-q)^T A (x-q) + ip(x-q) + i p_0 \cdot (x - x_0) + i\theta}

        Parameters
        ----------
        q : TYPE
            DESCRIPTION.
        p : TYPE, optional
            DESCRIPTION. The default is 0.
        a : TYPE, optional
            DESCRIPTION. The default is 1.
        phase : TYPE, optional
            DESCRIPTION. The default is 0.
        ndim : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        self.q = self.x = q
        self.p = p

        self.phase = phase
        # self.coeff = coeff # electronic coefficients
        self.ndim = ndim
        self.a = a
        
        if ndim == 1:
            self.var = 1./sqrt(a)
            self.fwhm = 2.*sqrt(2. * np.log(2)) / sqrt(a)

        elif ndim > 1:

            if isinstance(a, float):
                self.a = a * eye(ndim) # homogenous width
            
            if isinstance(p, float):
                self.p = np.array([p] * ndim)
                
            if isinstance(q, float):
                self.q = np.array([q] * ndim) 
                
            assert(a.shape == (ndim, ndim))
            assert(len(q) == ndim)
            assert(len(p) == ndim)
        
        self.params = (a, q, p)


    def evaluate(self, x):
        a, q, p = self.params
        if self.ndim == 1:
            return (a/pi)**(1/4) * exp(-0.5 * a * (x-q)**2 + 1j * p * (x-q))
        else: 
            pass

    def __mult__(self, other):
        pass
        # return GWP(a, q)
        
    def overlap(self, other):
        return overlap(self, other)


def overlap(gl, gr):
    # compute overlap between two GWPs with different covariance matrix
    
    n = gl.ndim 
    assert n == gr.ndim 
    
    A1, q1, p1 = gl.params
    A2, q2, p2 = gr.params
    
    (2*pi)**n/np.linalg.det(A1 + A2)
    
    return 

    
    