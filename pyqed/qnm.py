"""
One-sided cavity (quasi)normal modes
    -d ----medium1--- 0 ---medium0--- L
"""
import numpy as np
from numpy import sin, cos, tan, exp

from lime.units import c

def tmm(omega, r, t):
    """
    transfer matrix method for layered materials
    """

def propagation(omega, n, l):
    """
    propagation matrix  
    """
    
    phase = exp(-1j * omega * l * n/c)
    
    return np.array([[phase, 0], [0, phase.conj()]]) 

def interface(n1, n2):
    eta = n1/n2 
    D = 0.5 * np.array([[1. + eta, 1. - eta], [1. - eta, 1. + eta]])
    return D

def single_layer(omega, n1, n2, l):
    """
    transmission function of single layer
    n1 | n2| n1


    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    n1 : TYPE
        DESCRIPTION.
    n2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    D12 = interface(n1, n2)
    P = propagation(omega, n2, l)
    D21 = interface(n2, n1)
    

    return 

def resonance(omega):
    """
    determine the resonance of the cavity mode.
    Resonance frequency is the zeros of the returned determinantal function.

    Parameters
    ----------
    omega

    Returns
    -------

    """
    k1 = omega / n1
    k0 = omega / n0
    return tan(k0 * L) + (k0/k1) * tan(k1 * d)

def mode(z):
    if -d < z < 0:
        A = sin(k1 * (d + z))
    elif L > z > 0:
        A = (k1 / k0) * cos(k1 * d) * sin(k0 * z) + sin(k1 * d) * cos(k0 * z)
    else:
        A = 0.

    return A


if __name__ == '__main__':
    from lime.style import subplots
    fig, ax = subplots()
    
    from scipy import optimize

    for n1 in np.linspace(0.3, 1.5):
    #for n1 in [0.3, 0.41]:
        n0 = 1
        d = 1
        L = 2
    
        root = optimize.newton(resonance, 4)
        print('root = ', root)
    
        # if np.isclose(root, 4, atol=0.2):
        omega = root
        k1 = omega / n1
        k0 = omega / n0
    
    
    
        z = np.linspace(-d , L, 150)
    
    
        ax.plot(z, [mode(x) for x in z], label=r'$n_1$ = {}'.format(n1))

    # ax.legend()
    fig.show()
    ax.axhline(0, ls='--')


