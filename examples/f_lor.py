#! /usr/bin/env python

""" Lorentz oscillator function """

from numpy import *

def f_lor ( t, y ):
    """
    Right hand side for the Lorenz system
    """
    a=10.
    r=70.
    r=28
    b=8./3.

    f=zeros(3)
    f[0]=-a*y[0]+a*y[1]
    f[1]=r*y[0]-y[1]-y[0]*y[2]
    f[2]=-b*y[2]+y[0]*y[1]
    return(f)
