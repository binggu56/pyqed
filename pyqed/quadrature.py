# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:43:40 2021

@author: Bing

Computing multidimensional Gaussian integral with Gaussian-Hermite quadrature
"""

from scipy.special import roots_hermite
import numpy as np 
from math import sqrt, pi
import itertools

class Quadrature():
    def __init__(self):

        self.x = None
        self.w = None
    
    def gauss_hermite(self, n, mu=None, sigma=None):
        """
        Compute int f(x) exp(-x^2) dx = sum_{i=0}^n w[i] * f(x[i]) 
        using Gaussian-Hermite quadrature
    
        Parameters
        ----------
        n : TYPE
            DESCRIPTION.
        mu : TYPE, optional
            DESCRIPTION. The default is None.
        sigma : TYPE, optional
            DESCRIPTION. The default is None.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
        w : TYPE
            DESCRIPTION.
    
        """
        return gauss_hermite_quadrature(n, mu=mu, sigma=sigma)
        
    def integrate(self, f):
        
        x = self.x 
        w = self.w 
        
        const = 1./sqrt(pi)
        
        return np.sum(w * const * f(x)) 

def gauss_hermite_quadrature(n, mu=None, sigma=None):
    """
    Compute int f(x) exp(-x^2) dx = sum_{i=0}^n w[i] * f(x[i]) 
    using Gaussian-Hermite quadrature

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    mu : TYPE, optional
        DESCRIPTION. The default is None.
    sigma : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    """
    x, w = roots_hermite(n)
    if mu is not None:
        y = sqrt(2.) * sigma * x + mu  
        
        return y, w 
    else:
        return x, w 

# x, w = roots_hermite(10)
const = 1./sqrt(pi)

x, w = gauss_hermite_quadrature(20, mu=0, sigma=1)
def f(x):
    return np.sin(x)


ps = np.linspace(-5,5)
# for p in ps:
g = [np.sum( w * const * np.exp(-1j * p * x)) for p in ps]

import proplot as plt
fig, ax = plt.subplots()

ax.plot(ps, np.real(g))

# multidimensional
mu = np.array([1, 0])
Sigma = np.array([[1.3, -0.213], [-0.213, 1.2]])
N = len(mu)
const = np.pi**(-0.5*N)
xn = np.array(list(itertools.product(*(x,)*N)))

print(xn.shape)

wn = np.prod(np.array(list(itertools.product(*(w,)*N))), 1)
yn = 2.0**0.5*np.dot(np.linalg.cholesky(Sigma), xn.T).T + mu[None, :]
print("Normalising constant: %f" % np.sum(wn * const))
print("Mean:")
print(np.sum((wn * const)[:, None] * yn, 0))
print("Covariance:")
covfunc = lambda x: np.outer(x - mu, x - mu)
print(np.sum((wn * const)[:, None, None] * np.array(list(map(covfunc, yn))), 0))