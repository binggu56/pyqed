# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:18:29 2022

@author: Bing
"""

import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp

mass = 1.
e = 1.
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
ez = np.array([0, 0, 1])

def efield(t):
    cep = 0.
    omega = 1.
    E0 = 1

    return E0 * (cos(omega * t + cep) * ex - sin(omega * t + cep) * ey)

def force(t, z):
    omega = 2
    q, p = z[0:3], z[3:6]
    f0 = - mass * omega**2 * q
    f1 = - e * efield(t)
    F = np.concatenate((p, f0+f1))
    return F

def light_driven_free_electron(t, q0=(0.4, 0, 0), p0=(0, 0, 0), E0=1, omega=1, cep=0.):
    """
    a free electron in AC elctric field

    .. math::
        \dot{q} = p/m
        \dot{p} = f(q) = - e E(q, t) - m \omega_0^2 q

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    x0 : TYPE, optional
        DESCRIPTION. The default is (0, 0, 0).
    v0 : TYPE, optional
        DESCRIPTION. The default is (0, 0, 0).
    E0 : TYPE, optional
        DESCRIPTION. The default is 1.
    omega : TYPE, optional
        DESCRIPTION. The default is 1.
    cep : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    """
    z0 = np.concatenate((q0, p0))

    sol = solve_ivp(force, [0, 100], z0, method='RK45')

    # x =  E0 * np.sin(cep)/omega * t \
    #     + E0/omega**2 * (cos(omega * t + cep) - cos(cep))

    # y = - E0 * np.cos(cep)/omega * t \
    #     + E0/omega**2 * (sin(omega * t + cep) - sin(cep))

    return sol


t = np.linspace(0, 20, 200)
import proplot as plt
fig, ax = plt.subplots()
sol = light_driven_free_electron(t, cep=np.pi/4)
# print(sol.y.shape)
# print(sol.t.shape)


ax.plot(sol.y[0, :], sol.y[1, :])