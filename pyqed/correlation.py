#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:36:55 2019

@author: bing
"""

import numpy as np 
import sys 
from .phys import rk4
 
def correlation_4p_2t():
    # <V(tau3+tau2+tau1) V(tau2+tau1) V(tau1) V>
    return 

def correlation_3p_1t(H, rho0, ops, c_ops, tlist, dyn, *args):
    """
    compute the time-translation invariant two-point correlation function in the 
    density matrix formalism using quantum regression theorem 
        <AB(t)C> = Tr[ AU(t) B rho_0 C  U^\dag(t)]
   
    the density matrix is stored in 'dm.dat'
    the correlation function is stored in 'corr.dat'
        
    input:
        H: full Hamiltonian 
        rho0: initial wavepacket 
        ops: list of operators [A, B] 
        dyn: dynamics method e.g. lindblad, redfield, heom
        args: dictionary of parameters for dynamics 
        
    output: 
        t:
        
    """
    nstates =  H.toarray().shape[-1] # number of states in the system

    # initialize the density matrix
    A, B, C = ops
    rho = C.dot(rho0.dot(A))
    
    f = open('cor.dat', 'w')
    f_dm = open('dm.dat', 'w')

    # dynamics 

    t = 0.0
    Nt = len(tlist)
    dt = tlist[1] - tlist[0] 
    
    fmt = '{} ' * (nstates**2 + 1) + '\n' # format to store the density matrix
    
    for k in range(Nt):

        t += dt

        rho = rk4(rho, dyn, dt, H, c_ops) 

        cor = B.dot(rho).diagonal().sum()
        
        # store the reduced density matrix
        f.write('{} {} \n'.format(t, cor))
        f_dm.write(fmt.format(t, *np.ravel(rho.toarray())))


    f.close()
    f_dm.close()

    return 