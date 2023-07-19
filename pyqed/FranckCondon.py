# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:50:39 2021

TODO:
    multidimensional;
    Duchinsky rotation;

@author: Bing
"""

import math

from math import factorial
from math import sqrt, exp
from scipy.special import hermite, binom
# import numba

def dfactortial(n):
    return math.prod(range(n, 0, -2))

# @numba.jit
def FranckCondon(Ln, Lm, d):
    '''
    Analytical formula for the Franck-Condon factors from

    Chang, J.-L. Journal of Molecular Spectroscopy 232, 102–104 (2005).

    Parameters
    ----------
    Ln : TYPE
        DESCRIPTION.
    Lm : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    float
        Franck-Condon overlap.

    '''

    # If the excited state frequency (Ln[1]) is greater than the ground state
    # frequency (Lm[1]) then we must swap Ln and Lm for the program, but then
    # take the absolute value of the result.
    if (Ln[1] > Lm[1]):
        Ln, Lm = Lm, Ln

    n = Ln[0]
    m = Lm[0]
    wn_wavenumbers = Ln[1]
    wm_wavenumbers = Lm[1]

    wn = wn_wavenumbers/8065.5/27.2116
    wm = wm_wavenumbers/8065.5/27.2116
    # f = float(wn)/wm
    # w = wm

    # The formula is used for (x+d)^2 whereas I use (x-d)^2 for
    # the excited-state surface
    d = -d
    # F is the (massless) force constant for the mode. But which w?
    # F = w ** 2

    #convertedQSquared = deltaQ**2/(6.02214*(10**23) * 9.1094*(10**-28))
    # convertedQSquared = deltaQ**2


    # X is defined as such in Siders, Marcus 1981 Average frequency?
    # X = convertedQSquared / 2
    A = 2. * sqrt(wn * wm)/(wn + wm)
    S = d**2 * wn*wm/(wn + wm)

    p = sqrt(A*exp(-S)/(factorial(n) * factorial(m))) / 2.**((n+m)/2)

    def I(i,j):
        if (i+j) % 2 == 0:
            K = (i+j)//2
            return dfactortial(i+j-1)/(wn + wm)**K
        else:
            return 0

    fc = 0
    for i in range(n+1):
        F = hermite(n-i)
        bn = - wm * sqrt(wn)* d/(wn + wm)

        for j in range(m+1):

            G = hermite(m-j)
            bm = wn * sqrt(wm) * d/(wn + wm)

            fc += binom(n, i) * binom(m, j) * F(bn) * G(bm) * (2*sqrt(wn))**i *\
                (2*sqrt(wm))**j * I(i, j)

    return fc * p




# def genIntensities( deltaE, deltaQ, w_wavenumbers, wprime_wavenumbers):
#     """ wprime must be greater than w"""
#     wprime = wprime_wavenumbers/8065.5/27.2116
#     w = w_wavenumbers/8065.5/27.2116
#     intensityFunction = lambda n: (diffFreqOverlap([n, wprime_wavenumbers], [0, w_wavenumbers], deltaQ))**2
#     intensities = map(intensityFunction, range(0,11))
#     return intensities

# def genEnergies(deltaE, w_wavenumbers, wprime_wavenumbers):
#     wprime = wprime_wavenumbers/8065.5/27.2116
#     w = w_wavenumbers/8065.5/27.2116
#     energyFunction = lambda n: (deltaE + (n+0.5)*(wprime) - 0.5*w)
#     energies = map(energyFunction, range(0, 11))
#     return energies


if __name__ == '__main__':

    fc = FranckCondon([2, 500], [2, 500], 0)
    print(fc)