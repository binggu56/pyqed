#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:38:51 2023

@author: bing
"""

import numpy as np 
from pyqed import interval

import proplot as plt

# LippmanSchwingerSolver


a = -1.5 # l e f t b o rde r o f sim ul a ti o n box
b = 1.5 # r i g h t b o rde r
N = 160 # number o f s u b d i v i s i o n s
pot_id = 'square' # i d o f p o t e n t i a l t o u se
k_vec = np.linspace(0.1, 8, 20)
# wave number f o r incoming wave

def V(x):
    L = 2
    V0 = 4
    z = np.zeros( x.shape ) 
    z[abs(x) <= L / 2] = -V0
    return z 

def G(x , y, k): 
    """
    1D Green's function
    .. math::
        
        (\Delta + k^2) G(x-y) = \delta(x-y)
    
    Returns
    =======
    complex number
    """
    return -1j / k * np.exp( 1j * k * abs ( x - y ) )

class LippmannSchwingerSolver:
    """
    Refs:
        https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/puschnig_peter/unigrazform/Theses/Hirtler_Bachelorarbeit_final.pdf
    """
    def __init__(self, a, b, n):
        self.a = a 
        self.b = b 
        self.n = n
        
    
    def run(self, k_vec):
        N = self.n
        
        K = np.zeros( (N+1, N+1) , dtype=complex )
        # se tup K(x , t ) ma trix
        x = np.linspace(a, b, N+1)
        t = np.linspace(a, b, N+1)
        h = interval(x)
        
        transmission = np.zeros(len(k_vec))
        fig, ax = plt.subplots()
        ax.plot(x, V(x), '--')
        
        for i, k in enumerate(k_vec):
            
            [xx , tt] = np.meshgrid( x , t )  # g e t a l l (x , t ) c ombin a ti on s
            K = (G(xx, tt, k) * V(tt) ).transpose()  # c a l c u l a t e v al u e s o f K
            A = np.identity(N+1) - h * K
            
            phi = np.exp(1j * k * x) # homogeneous function phi
            psi = np.linalg.solve(A, phi) # scattering states
        
            transmission[i] = np.abs(psi[-1])
        
        fig, ax = plt.subplots()
        ax.plot(k_vec, transmission, '-o')
        
sol = LippmannSchwingerSolver(a, b, N)
sol.run(k_vec)

class LippmannSchwinger2DSolver:
    def __init__(self):
        x1 = -2;
        x2 = 2 ;
        y1 = -2;
        y2 = 2 ;
        Nx = 40; # number o f p oi n t s i n x d i r e c t i o n
        Ny = 40 # number o f p oi n t s i n y d i r e c t i o n
        # p o t i d = ’ Trench ’ ; # type o f p o t e n t i a l t o u se
        angle = np.zeros((16 , 1 ) ) ; # a n gl e between k and x−axis
        k_mag = np.linspace( 1 , 16 , 16 ) ; # magnitude o f k v e c t o r
        
        X, Y = np.meshgrid( x , y ) ; # g e t a l l (x , y ) c ombin a ti on s
        eps = 1e-4; # f o r a v oi di n g 0 i n h ankel f u n c ti o n
        coords = numpy.array( [np.ravel(X), np.ravel(Y ) ] ).transpose( ) ;
        S = scipy.spatial.distance.cdist(coords , coords + eps,'euclidean') + 0j
        
        scipy.special.hankel1( 0 , k * S , out=S) ;
        S *= -1j/4
        
        h = ( x[ 1 ] - x[ 0 ] ) * ( y[ 1 ]- y[ 0 ] ) ;
        S *= np.ravel(V(X + eps, Y + eps) ) ;
        S *= -h ;
        S += np.identity(N)
        psi = np.linalg.solve(S, phi ) 
