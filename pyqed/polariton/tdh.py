#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:47:19 2023


N molecules coupled with M cavity modes

@author: bing
"""

import numpy as np
from pyqed import Mol, sigmax, comm
from pyqed.units import au2ev, au2fs

class TDH:
    def __init__(self, mol, cav):
        self.nstates = mol.nstates
        self.mol = mol
        self.cav = cav
        
        self.coupling = None
        return 
    
    def set_coupling(self, g):
        """
        The molecule-field interaction is given by 
        .. math::
            
            H_{CM} = g \mu (a + a^\dagger)
            
        where g is the cooperative coupling strength.

        Parameters
        ----------
        g : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.coupling = g * self.mol.edip
        return 
    
    def run(self, rho0, dt=0.01, nt=1, drive=None):
        nstates = self.nstates
        
        M = self.coupling

        omegac = self.cav.omega
        gamma = self.cav.decay
        # equations of motion

        dt2 = dt/2

        # molecular density matrix
        rho = np.zeros((nt, nstates, nstates), dtype=complex)
        rho[0] = rho0
        rho[1] =  rho[0] + (-1j * comm(H, rho[0])) * dt

        vh = 0. # t0

        for k in range(1, nt-1):
            t = k * dt

            rho[k+1] =  rho[k-1] + (-1j * comm(H + vh - pulse(t) * edip, rho[k])) * dt * 2

            # vh = M * D(t, omegac) * np.trace(rho[0] @ M) * dt2\
            vh = 0
            for n in range(k):
                vh += M * D((k-n)*dt, omegac, gamma) * np.trace(rho[n] @ M) * dt

        return 
    
    
def D(t, omega0=1/au2ev, gamma=0.0/au2ev):
    """
    propagator for a single boson (photon) mode

    if gamma < \omega,
    .. math::

        D^{-1}_0(t) = - \frac{1}{\omega_0} (\partial_t^2 + 2\gamma\partial_t + \omega_0^2)

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    omega : TYPE, optional
        DESCRIPTION. The default is 1.
    gamma : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if gamma < omega0:
        omega = np.sqrt(omega0**2 - gamma**2)
        return -(omega0) * np.heaviside(t, 0) * np.sin(omega * t)/omega * np.exp(-gamma * t)
    elif gamma > omega0:
        omega = np.sqrt(-omega0**2 + gamma**2)
        return omega0 * np.heaviside(t, 0) * np.sinh(omega * t)/omega * np.exp(-gamma * t)

def pulse(t, tau=2/au2fs):
    return 0.001 * np.exp(-(t-8/au2fs)**2/2/tau**2) * np.cos(omegac * t)




if __name__ == '__main__':
    
    E = [0, 1/au2ev]
    H = np.diagflat(E)
    mol = Mol(H, edip=sigmax())
    edip=sigmax()
    
    nstates = len(E)
    
    # initial density matrix, ground state
    
    # rho = np.zeros((nstates, nstates))
    # rho[0, 0] = 1.0
    
    # collective coupling constant
    sx = sigmax()
    M = 0.1/au2ev * sx
    
    omegac = 1/au2ev
    # equations of motion
    
    dt = 0.04/au2fs
    dt2 = dt/2
    nt = 1500
    
    rho = np.zeros((nt, nstates, nstates), dtype=complex)
    rho[0, 0, 0] = 1.
    rho[1] =  rho[0] + (-1j * comm(H, rho[0])) * dt
    
    vh = 0. # t0
    
    for k in range(1, nt-1):
        t = k * dt
    
        rho[k+1] =  rho[k-1] + (-1j * comm(H + vh - pulse(t) * edip, rho[k])) * dt * 2
    
        # vh = M * D(t, omegac) * np.trace(rho[0] @ M) * dt2\
        vh = 0
        for n in range(k):
            vh += M * D((k-n)*dt, omegac, gamma=0.02/au2ev) * np.trace(rho[n] @ M) * dt
    
    
    # save results
    ts = np.arange(nt) * dt * au2fs
    np.savez('negf_leak_0.02', ts, rho)




