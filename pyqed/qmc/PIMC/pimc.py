# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:01:33 2016

@author: bing
"""

import numpy as np 
from math import * 


def V(x):
    """
    potential energy function
    use units such that m = 1 and omega_0 = 1
    """
    return 0.5 * pow(x, 2.0) 

def dVdx(x):
    """
    derivative dV(x)/dx used in virial theorem
    """
    return x

def OneMetropolisStep():
    
    global x, x_new, P, E_sum, E_sqd_sum, acceptances   
    
    for i in range(M):
        
        # choose a time slice at random
        j = int(np.random.rand() * M)
        
        # indexes of neighbors periodic in tau
        j_minus = j - 1
        j_plus = j + 1
        if j_minus < 0:
            j_minus = M - 1
        if j_plus > M - 1: 
            j_plus = 0
        
        # choose a random trial displacement
        x_trial = x[j] + (2.0 * np.random.rand() - 1.0) * delta
        
        # compute change in energy
        Delta_E = V(x_trial) - V(x[j])        \
                + 0.5 * pow((x[j_plus] - x_trial) / Delta_tau, 2.0) \
                + 0.5 * pow((x_trial - x[j_minus]) / Delta_tau, 2.0) \
                - 0.5 * pow((x[j_plus] - x[j]) / Delta_tau, 2.0) \
                - 0.5 * pow((x[j] - x[j_minus]) / Delta_tau, 2.0)
    
        if Delta_E < 0.0 or exp(- Delta_tau * Delta_E) > np.random.rand():
            acceptances += 1 
            x_new = x_trial
            x[j] = x_trial 
        else:
            x_new = x[j]
            
        # add x_new to histogram bin
        bin = int((x_new - x_min) / (x_max - x_min) * n_bins)
        if bin >= 0 and bin < M:
            P[bin] += 1
            
            
        # compute Energy using virial theorem formula and accumulate
        E = V(x_new) + 0.5 * x_new * dVdx(x_new)
        E_sum += E
        E_sqd_sum += E * E

if __name__ == '__main__':
    
    print(" Path Integral Monte Carlo for the Harmonic Oscillator\n")
    print(" -----------------------------------------------------\n")

    # set simulation parameters
    tau = 10.0        # time period 
    M = 100           # number of time slices 
    x_max = 4.0 
    n_bins = 100       #  number of bins for psi histogram
    delta = 1.0        # Metropolis step size in x
    MC_Steps = 20000  # number of Monte Carlo steps in simulation
    
    print(" Imaginary time period tau =  {} \n " \
          "\n Number of time slices M = {} \n" \
          "\n Maximum displacement to bin x_max = {} \n " \
          "\n Number of histogram bins in x = {} \n" \
          "\n Metropolis step size delta = {} \n" \
          "\n Number of Monte Carlo steps = ".format(tau, M, x_max,n_bins, delta, MC_Steps))
    
    
    P = np.zeros(n_bins)           # histogram for |psi|^2

    Delta_tau = tau / M  # imaginary time step and period 

    x_min = -x_max
    dx = (x_max - x_min) / n_bins

    print(' Initializing atom positions using uniform sampling \n')
    x = (2.0 * np.random.rand(M) - 1.0) * x_max   # displacements from equilibrium of M "atoms"
    
    therm_steps = MC_Steps / 5
    acceptances = 0
    x_new = 0
    print(" Doing {} thermalization steps ... \n".format(therm_steps))
    for step in range(therm_steps):
        OneMetropolisStep()
    
    print("Percentage of accepted steps = {} % \n".format(acceptances / float(M * therm_steps) * 100.0))

    E_sum = 0
    E_sqd_sum = 0
    acceptances = 0
    P = np.zeros(n_bins)
    
    print(" Doing {} production steps ... \n".format(MC_Steps))
    
    for step in range(MC_Steps):
        OneMetropolisStep()


    # compute averages
    values = MC_Steps * M
    E_ave = E_sum / values
    E_var = E_sqd_sum / values - E_ave * E_ave
    print("\n <E> = {} +/- {} \n".format(E_ave, sqrt(E_var / values)))
    print("<E^2> - <E>^2 = {} \n".format(E_var))
    
    f = open("pimc.out","w")
    E_ave = 0
    for bin in range(n_bins):
        x = x_min + dx * (bin + 0.5)
        f.write('{} {} \n'.format(x, P[bin] / values)) 
        E_ave += P[bin] / values * (0.5 * x * dVdx(x) + V(x))
    f.close() 
    
    print(" <E> from P(x) = {} \n".format(E_ave)) 
    print(" Probability histogram written to file pimc.out") 

