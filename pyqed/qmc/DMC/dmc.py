# Diffusion Monte Carlo program for the 3-D harmonic oscillator

# author: Bing Gu 
# date:  May 9, 2016 


# import modules 
import numpy as np 
from math import *  

DIM = 3             # dimensionality of space

def V(r):         # harmonic oscillator in DIM dimensions
    rSqd = np.dot(r,r) 
    return 0.5 * rSqd 

def oneMonteCarloStep(n):

    global N, r, alive , E_T, dt  
    
    # Diffusive step
    r[n] += np.random.randn(DIM) * sqrt(dt)

    # Branching step
    q = exp(- dt * (V(r[n]) - E_T))
    survivors = int(q)

    if q - survivors > np.random.rand():
        survivors += 1 

    # if survivors is zero, then kill the walker
    if survivors == 0:
        alive[n] = False
    
    # append survivors-1 copies of the walker to the end of the array
    elif survivors > 1:
        
        newN = N + survivors - 1

        r.resize((newN,DIM))
        alive.resize(newN) 

        r[N:newN] = r[n]
        alive[N:newN] = True

        N = newN 
    

def oneTimeStep():

    global N, ESum, ESqdSum, psi, E_T,r, alive 
    
    # DMC step for each walker
    N_0 = N
    
    for n in range(N_0):
        oneMonteCarloStep(n)

    # remove all dead walkers from the arrays
    newN = 0
    for n in range(N):
        if alive[n]:
            if n != newN:
                 r[newN] = r[n]
                 alive[newN] = True
            newN += 1 
    N = newN

    r.resize((N,DIM))
    alive.resize(N)

    # adjust E_T
    E_T += log(float(N_T) / float(N)) / 10.0  
    
    # measure energy, wave function
    ESum += E_T
    ESqdSum += E_T * E_T

    for n in range(N):
        rSqd = np.dot(r[n],r[n]) 
        i = int( sqrt(rSqd) / rMax * NPSI)
        if i < NPSI:
            psi[i] += 1

def zeroAccumulator(): 
    global ESum, ESqdSum, psi 

    ESum = ESqdSum = 0.0 
    psi = np.zeros(NPSI)


if __name__ == '__main__':

    print " Diffusion Monte Carlo for the 3-D Harmonic Oscillator\n"
    print " -----------------------------------------------------\n"
    N_T =  input(" Enter desired target number of walkers: ")
    
    dt = input( " Enter time step dt: " ) 

    timeSteps = input( " Enter total number of time steps: ")

    # do 20% of timeSteps as thermalization steps
    thermSteps = int(0.2 * timeSteps) 
    
    rMax = 4.0               # max value of r to measure psi
    NPSI = 100               # number of bins for wave function
    psi = np.zeros(NPSI)     # wave function histogram

    N = N_T                   # set N to target number specified by user

    r = np.random.rand(N,DIM) - 0.5 # uniform sampling for n-th walker 
    alive = np.array([True] * N)
        
    E_T = 0.0                   # initial guess for the ground state energy
   
    zeroAccumulator() 
    for i in range(thermSteps):
        oneTimeStep()

    # production steps
    
    zeroAccumulator() 
    for i in range(timeSteps):
        oneTimeStep()

    # compute averages
    EAve = ESum / timeSteps
    EVar = ESqdSum / timeSteps - EAve * EAve
    print(" <E> = {} +/-  {} ".format( EAve, sqrt(EVar / timeSteps)))
    print(" <E^2> - <E>^2 =  {} ".format(EVar))
    
    psiNorm = 0.0
    psiExactNorm = 0.0
    dr = rMax / NPSI
     
    # there is a bug associated with normalization constant for DMC, but the energy should be corrent 

    for i in range(NPSI):
        r = i * dr
        psiNorm += psi[i] * psi[i]     
        psiExactNorm += pow(r, DIM-1) * exp(- r * r)
        
    psiNorm = sqrt(psiNorm)
    psiExactNorm = sqrt(psiExactNorm)

    # store the wavefunction 
    f = open("psi.data", 'w')
    for i in range(NPSI):
        r = i * dr
        f.write( '{} {} {} \n'.format(r, psi[i] / psiNorm, \
                pow(r, (DIM-1)) * exp(- r * r / 2) / psiExactNorm))
    f.close() 

