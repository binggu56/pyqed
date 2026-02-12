# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:35:39 2016

@author: bing
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:42:22 2016

@author: bing
"""
import numpy as np
import scipy  
import numba 
import sys 


@numba.autojit
def derivs(x):
    """
    Morse potential     
    """ 
    
    PES = 'Morse' 
    
    if PES == 'Morse':
        
        a, x0 = 1.02, 1.4 
        De = 0.176 / 100.0 
    
        d = (1.0-np.exp(-a*x))
        
        v0 = De*d**2
            
        dv = 2. * De * d * a * np.exp(-a*x)
        
    elif PES == 'HO':
        
        v0 = x**2/2.0 
        dv = x 
    
        #ddv = 2.0 * De * (-d*np.exp(-a*((x-x0)))*a**2 + (np.exp(-a*(x-x0)))**2*a**2)
    
    return v0,dv

@numba.autojit
def qpot(x,p,r,w):

    """
    Linear Quantum Force : direct polynomial fitting of derivative-log density (amplitude)    
    curve_fit : randomly choose M points and do a nonlinear least-square fitting to a 
            predefined functional form      
    """
    
    #tau = (max(xdata) - min(xdata))/(max(x) - min(x))
    #if tau > 0.6:
    #    pass 
    #else: 
    #    print('Data points are not sampled well.'
    
    Nb = 4 
    S = np.zeros((Nb,Nb))
    
    for j in range(Nb):
        for k in range(Nb):
            S[j,k] = np.dot(x**(j+k), w)  
    
    bp = np.zeros(Nb)
    br = np.zeros(Nb)
    
    for n in range(Nb):
        bp[n] = np.dot(x**n * p, w)
        br[n] = np.dot(x**n * r, w)
        
        
    cp = np.linalg.solve(S,bp)
    cr = np.linalg.solve(S,br)  

    #unit = np.identity(Nb)
    #r_approx = cr[0] * unit + cr[1] * x + cr[2] * x**2 + cr[3] * x**3 
    #p_approx = cp[0] * unit + cp[1] * x + cp[2] * x**2 + cp[3] * x**3

    dr = cr[1] + 2. * cr[2] * x + 3. * cr[3] * x**2 
    dp = cp[1] + 2. * cp[2] * x + 3. * cp[3] * x**2
    
    ddr = 2. * cr[2] + 6. * cr[3] * x 
    ddp = 2. * cp[2] + 6. * cp[3] * x 
    
    fr =  -1./2./am * (2. * r * dp + ddp)
    fq = 1./2./am * (2. * r * dr + ddr)  
    
    Eu = -1./2./am * np.dot(r**2 + dr,w)
        
    return Eu,fq,fr 


    
def sym(V):

    n = V.shape[-1] 
    
    for i in range(n):
        for j in range(i):
            V[j,i] = V[i,j] 
    return V 






# initialization    
Ntraj = 8000 
a0 = 9.16 * 2  
x0 = 1.3  


x = np.random.randn(Ntraj) 

#x = np.zeros(Ntraj)  
#for k in range(Ntraj):
#    x[k] = np.random.randn() 
#    while x[k] > 3.0:
#        x[k] = np.random.randn()
    
        
x = x / np.sqrt(2.0 * a0) 

p = np.zeros(Ntraj)
r = - a0 * x

w = np.array([1./Ntraj]*Ntraj)
am = 916. 
Nt = 8000
dt = 0.1

dt2 = dt/2.0 
t = 0.0 


f = open('traj.dat','w')
fe = open('en.out','w')
f_MSE = open('rMSE.out','w')
nout = 20       # number of trajectories to print 
fmt =  ' {}' * (nout+1)  + '\n'  
Eu = 0.  

Ndim = 1        # dimensionality of the system  
fric_cons = 0.08       # friction constant  

v0, dv = derivs(x)
Eu,fq,fr = qpot(x,p,r,w)

for k in range(Nt):
    t = t + dt 

    p += (- dv + fq) * dt2 - fric_cons * p * dt2   
    r += fr * dt2
    
    x +=  p*dt/am

    # force field 
    Eu, fq, fr = qpot(x,p,r,w)
    if Eu < 0:
        print('Error: U = {} should not be negative. \n'.format(Eu))
        sys.exit()
        
    v0, dv = derivs(x)

    p += (- dv + fq) * dt2 - fric_cons * p * dt2 
    r += fr * dt2 
       
    f.write(fmt.format(t,*x[0:nout]))
    Ek = np.dot(p*p,w)/2./am 
    Ev = np.dot(v0,w)
    Etot = Ek + Ev + Eu
    
    fe.write('{} {} {} {} {} \n'.format(t,Ek,Ev,Eu,Etot))
    
    if k == Nt-1:
        print('The total energy = {}. \n'.format(Etot))

fe.close()
f.close() 

hartree_wavenumber = scipy.constants.value(u'hartree-inverse meter relationship') / 1e2 


a, x0, De = 1.02, 1.4, 0.176/100 
print('The well depth = {} cm-1. \n'.format(De * hartree_wavenumber))

omega  = a * np.sqrt(2. * De / am )
E0 = omega/2. - omega**2/16./De
dE = (Etot-E0) * hartree_wavenumber 
print('Exact ground-state energy = {} Hartree. \nEnergy deviation = {} cm-1. \n'.format(E0,dE))
    


    