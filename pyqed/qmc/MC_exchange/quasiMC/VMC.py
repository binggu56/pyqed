# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:43:36 2016

Diffusion Monte Carlo 

@author: bing
"""
import numpy as np 


xmin = -6. 
xmax = 6.
dx = 0.1 



delta = 1

def p(y, x):
    """
    trial function is exp(-alpha*x^2)
    compute the ratio of rho(x') / rho(x)
    """    
    return np.exp(- alpha/2.0 * ((y-2)**2 - (x-2)**2))

def eLocal(x):
    """compute the local energy"""
    
    return np.sqrt(1./2./np.pi) * np.exp(-alpha/2. * (x+2)**2)





def MetropolisStep():
    
    global eSum, eSqdSum, nAccept 

    # chose a walker at random
    n = int(np.random.rand() * N)

    # make a trial move
    y = x[n] + delta * np.random.randn();

    # Metropolis test
    if p(y, x[n]) > np.random.rand():
        x[n] = y
        nAccept += 1 

    # accumulate energy and wave function
    e = eLocal(x[n])
    eSum += e
    eSqdSum += e * e
    i = int((x[n] - xmin) / dx)
    if i >= 0 and i < nPsiSqd:
        psiSqd[i] += 1
        
def oneMonteCarloStep():

    # perform N Metropolis steps
    for i in range(N):
        MetropolisStep()



print(" Variational Monte Carlo for Harmonic Oscillator\n")
print(" -----------------------------------------------\n")

N = input( " Enter number of walkers: ")

#alpha =  input(" Enter parameter alpha: ")
alpha = 1.0 

#MCSteps = input( " Enter number of Monte Carlo steps: ") 

MCSteps = 20000

#def init(N):
#    """initialize the random walkers"""
x = np.random.rand(N)
nPsiSqd = int((xmax-xmin)/dx)
psiSqd = np.zeros(nPsiSqd)

# perform 20% of MCSteps as thermalization steps
# and adjust step size so acceptance ratio ~50%
thermSteps = int(0.2 * MCSteps)
adjustInterval = int(0.1 * thermSteps) + 1
nAccept = 0
print(" Performing {} thermalization steps ...".format(thermSteps))

for i in range(thermSteps):
    oneMonteCarloStep()
    if (i+1) % adjustInterval == 0:
        delta *= nAccept / (0.5 * N * adjustInterval)
        nAccept = 0 
        
print("\n Adjusted Gaussian step size = {} ".format(delta)) 



nAccept = 0  # accumulator for number of accepted steps
eSum = 0. 
eSqdSum = 0.

nAccept = 0;
print( " Performing {} production steps ...".format(MCSteps)) 


 
for i in range(MCSteps):
    oneMonteCarloStep()


# compute and print energy

eAve = eSum / N / MCSteps

eVar = eSqdSum / N / MCSteps - eAve * eAve

error = np.sqrt(eVar/N / MCSteps)

print("\n <Energy> = {} +/- {} ".format(eAve, error)) 
print("\n Variance = {}".format(eVar))




# write wave function squared in file
f = open("psiSqd.data",'w')


psiNorm = sum(psiSqd) * dx
for i in range(nPsiSqd):
    z = xmin + i * dx;
    f.write('{} {} \n'.format(z, psiSqd[i] / psiNorm))
    
f.close()
print( " Probability density written in file psiSqd.data")


