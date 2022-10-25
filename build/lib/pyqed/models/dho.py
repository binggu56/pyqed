#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:47:03 2021

Numerically evaluating Franck-Condon factors for displaced harmonic oscilators

@author: Bing Gu
"""

from scipy.special import hermite
from scipy.special import eval_hermite as Hermite
from lime.units import au2wavenumber

# configuration
deltaX = 0.2e-10 # Distance between the two potentials in SI units
omega1 = 4.33e14 # angular frequency of the lower potential in SI units
omega2 = 4.14e14 # angular frequency of the upper potential in SI units
m = 1.66e-27*14  # vibrating masses

# -- code --
from scipy import integrate
from numpy.polynomial.hermite import hermval
import numpy as np
# import math
from math import factorial, pi, sqrt, exp
# from numpy import pi, sqrt, exp



        

def FranckCondon(n1, omega1, n2, omega2, d):
    '''
    Numerically computing the Franck-Condon-Factors.

    This can be generalized to anharmonic surfaces. It then requires
    the vibrational eigenstates of both surfaces.

    This can be used for non-Condon approximations.

    Parameters
    ----------
    omega1 : TYPE
        DESCRIPTION.
    omega2 : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    def psi(n, x, omega, mass=1):
        '''
        eigenstates of the harmonic oscillator

        Parameters
        ----------
        n : int
            quantum number. n = 0 is ground state.
        x : TYPE
            coordinates
        omega : TYPE
            fundemental frequency, au.
        mass : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        a = mass*omega
        return pow(a/pi, 0.25)/sqrt(pow(2, n)*factorial(n))* \
            hermite(n)(sqrt(a)*x) * exp(-0.5*a*x*x)

    # overlap product for transition n1 -> n2 at x0, where x0=(x=0) for the lower potential
    def transition(x, n1, n2):
        return psi(n1, x, omega1)*psi(n2, x-d, omega2)

    #for i in range(-500, 500):
    #    print psi(2, float(i)/10e11, omega1)

    print('{0:3}{1:3}{2}'.format('n1', 'n2', 'FC factor'))

    # for n1 in range(0, 10):
    #     for n2 in range(0, n1):
        # the integration bounds are to be determined by hand usually, they go from
        # -numpy.inf to numpy.inf - biggest influence is the mass m
    fc = integrate.quad(transition, -np.inf, np.inf, args=(n1, n2))
    return fc[0]


if __name__ == '__main__':
    
    # hbar = 1.05e-34
    # Hermite polynomial of n-th grade at x
    # def Hermite(n, x):
    #     # coeff = [0]*(n)
    #     # coeff.append(1)
    #     # return hermval(x, coeff)
    #     return
    
    omega1 = 499/au2wavenumber
    omega2 = 501/au2wavenumber
    d = 1. # a.u.
    
    # fc = fc[0]*fc[0]
    # print('{0:^3}{1:^3}{2}'.format(n1, n2, fc[0]))
    print(FranckCondon(2, omega1, 3, omega2, d))

