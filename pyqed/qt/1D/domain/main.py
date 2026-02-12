# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:35:39 2016

@author: bing

Double well ground state computation with quantum trajectories
Nonclassical momentum is approximated using subspaces 

Several questions to answer: 

1. nonlinearity can be introduced into the approximation by domain function 
2. subspaces can improve the accuracy of results 
3. in particular, it should be the best way to approximate double-well problems 
 
# alternatively - Nonlinear curve fitting with Pade approximants 

"""

import numpy as np
#import scipy  
import numba 
import sys

import constants as paras 
from vpot import derivs
#from fit import linear_fit_domain_nor 

from scipy.optimize import curve_fit


bohr_angstrom = 0.52917721092
hartree_wavenumber = 219474.63 

#hartree_wavenumber = scipy.constants.value(u'hartree-inverse meter relationship') / 1e2 




    
@numba.autojit
def rational3_2(x,a0,a1,a2,a3,b1,b2):
    
    return (a0 + a1 * x + a2 * x**2 + a3 * x**3) /(1.0 + b1 * x + b2 * x**2)  
     

def df3_2(x,a0,a1,a2,a3,b1,b2):
    P = (a0 + a1 * x + a2 * x**2 + a3 * x**3)
    Q = 1.0 + b1 * x + b2 * x**2 
    
    dP = a1 + 2.0 * a2 * x + 3.0 * a3 * x**2 
    dQ = b1 + 2.0 * b2 * x
    
    ddP = 2. * a2 + 6. * a3 * x 
    ddQ = 2.0 * b2 
    
    return dP/Q - P * dQ / Q**2, ddP/Q - 2.0 * dP * dQ / Q**2 + 2.0 * P * dQ**2 / Q**3 - P * ddQ / Q**2

@numba.autojit
def rational(x,a0,a1,a2,a3,a4,a5,b1,b2,b3,b4):
    
    return (a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5) / \
            (1.0 + b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4)  
     

def df(x,a0,a1,a2,a3,a4,a5,b1,b2,b3,b4):
    P = (a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5)
    Q = 1.0 + b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4 
    
    dP = a1 + 2.0 * a2 * x + 3.0 * a3 * x**2 + 4. * a4 * x**3 + 5. * a5 * x**4 
    dQ = b1 + 2.0 * b2 * x + 3. * b3 * x**2 + 4. * b4 * x**3 
    
    ddP = 2. * a2 + 6. * a3 * x + 12.0 * a4 * x**2 + 20.0 * a5 * x**3 
    ddQ = 2.0 * b2 + 6.0 * b3 * x + 12.0 * b4 * x**2 
    
    return dP/Q - P * dQ / Q**2, ddP/Q - 2.0 * dP * dQ / Q**2 + 2.0 * P * dQ**2 / Q**3 - P * ddQ / Q**2


def qpot_nonlinear(x,p,r,w):
    
    global f_MSE, p0_p, p0_r 
    
#    popt, pcov = curve_fit(rational3_2, x, p)    
#    a0, a1, a2, a3, b1, b2 = popt 
#    
#    #p_approx = func(x,*popt)       
#
#    dp, ddp = df3_2(x,a0, a1, a2, a3, b1, b2)
#      
#    # curve_fit of r with poly 
#    popt, pcov = curve_fit(rational3_2, x, r)
#    a0, a1, a2, a3, b1, b2 = popt 
#    
#    r_approx = rational3_2(x, a0, a1, a2, a3, b1, b2)
#    dr, ddr = df3_2(x, a0, a1, a2, a3, b1, b2)
#    
#    rMSE = np.dot(r-r_approx, r-r_approx)/Ntraj 
    
    if True : 
        
        popt, pcov = curve_fit(rational, x, p)    

        #p0_p = popt 
        
        a0, a1, a2, a3, a4, a5, b1, b2, b3, b4 = popt 
        
        #p_approx = func(x,*popt)       
    
        dp, ddp = df(x,a0, a1, a2, a3, a4, a5, b1, b2, b3, b4 )
          
        # curve_fit of r with poly 
        
        popt, pcov = curve_fit(rational, x, r)
        a0, a1, a2, a3, a4, a5, b1, b2, b3, b4 = popt 
        
        #p0_r = popt 
        
        r_approx = rational(x, a0, a1, a2, a3, a4, a5, b1, b2, b3, b4)
        dr, ddr = df(x, a0, a1, a2, a3, a4, a5, b1, b2,b3,b4)
        
        rMSE = np.dot(r-r_approx, r-r_approx)/Ntraj 
        if rMSE > 0.1: 
            print('Fitting of r fails. Mean Square Error = {} \n'.format(rMSE))
            
            f = open('r.out', 'w')
            for i in range(Ntraj):
                f.write('{} {} {} \n'.format(x[i], r[i], r_approx[i]))
            f.close()
            
            sys.exit() 
        
    f_MSE.write('{} {} \n'.format(t, rMSE))
    
    fr = -1./2./am * (2. * dp * r + ddp)
    fq = 1./2./am * (2. * r * dr + ddr)  
    
    Eu = -1./2./am * np.dot(r**2 + dr,w)
        
    return Eu,fq,fr, rMSE


@numba.autojit
def qpot_linear(x,p,r,w):
    
    global f_MSE

    """
    Linear Quantum Force : direct polynomial fitting of derivative-log density (amplitude)    
    curve_fit : randomly choose M points and do a nonlinear least-square fitting to a 
            predefined functional form      
    """
    
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
    r_approx = cr[0] + cr[1] * x + cr[2] * x**2 + cr[3] * x**3 
    #p_approx = cp[0] + cp[1] * x + cp[2] * x**2 + cp[3] * x**3
    
    rMSE = np.dot(r-r_approx, r-r_approx)/Ntraj
    
    f_MSE.write('{} {} \n'.format(t,rMSE))
    

    dr = cr[1] + 2. * cr[2] * x + 3. * cr[3] * x**2 
    dp = cp[1] + 2. * cp[2] * x + 3. * cp[3] * x**2
    
    ddr = 2. * cr[2] + 6. * cr[3] * x 
    ddp = 2. * cp[2] + 6. * cp[3] * x 
    
    fr =  -1./2./am * (2. * r * dp + ddp)
    fq = 1./2./am * (2. * r * dr + ddr)  
    
    Eu = -1./2./am * np.dot(r**2 + dr,w)
    
    
        
    return Eu,fq,fr, rMSE 

@numba.autojit
def qpot_linear_domain(x,p,r,w,L=6):
    
    global f_MSE

    """
    Linear Quantum Force : direct polynomial fitting of derivative-log density (amplitude)   
    LQF with predefined domains 
    
    curve_fit : randomly choose M points and do a nonlinear least-square fitting to a 
            predefined functional form      
    """
    
    for l in range(L):

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
    r_approx = cr[0] + cr[1] * x + cr[2] * x**2 + cr[3] * x**3 
    #p_approx = cp[0] + cp[1] * x + cp[2] * x**2 + cp[3] * x**3
    
    rMSE = np.dot(r-r_approx, r-r_approx)/Ntraj
    
    f_MSE.write('{} {} \n'.format(t,rMSE))
    

    dr = cr[1] + 2. * cr[2] * x + 3. * cr[3] * x**2 
    dp = cp[1] + 2. * cp[2] * x + 3. * cp[3] * x**2
    
    ddr = 2. * cr[2] + 6. * cr[3] * x 
    ddp = 2. * cp[2] + 6. * cp[3] * x 
    
    fr =  -1./2./am * (2. * r * dp + ddp)
    fq = 1./2./am * (2. * r * dr + ddr)  
    
    Eu = -1./2./am * np.dot(r**2 + dr,w)
    
    
        
    return Eu,fq,fr, rMSE 

    
def sym(V):

    n = V.shape[-1] 
    
    for i in range(n):
        for j in range(i):
            V[j,i] = V[i,j] 
    return V 

def trial(x,a):
    """
    trial function for each domain 
    """
    if len(a) != 2:
        sys.exit('length of coefficients does not match linear function')
        
    return a[0] + a[1]*x

def sech(x):
    return 1.0/np.cosh(x)
    
def linear_fit_domain(x, w, L=3):
    """
    linear fit with spacial domains 
    
    input: 
        L : number of domains 
    """
    # define domain functions 
    d = 2.0
    xdom = [-0.6, 0.6]
    domFunc = [] 
    dDomFunc = [] 
    ddDomFunc = [] 

    # define the first domain function 
    
    x0 = xdom[0] 

    func = 0.5 * (1. - np.tanh(d * (x - x0)))

    domFunc.append(func)   
    dDomFunc.append( - 0.5 * d * sech(d * (x-x0))**2 ) 
    ddDomFunc.append( d**2 * np.tanh(d*(x-x0)) * sech(d*(x-x0))**2)


    for i in range(L-2):
        
        xl, xr = xdom[i], xdom[i+1] 
        
        domFunc.append(0.5 * (np.tanh(d*(x - xl)) - np.tanh(d * (x - xr))))
        dDomFunc.append(0.5 * ( d * sech( d * (x-xl))**2 - d * sech(d * (x-xr))**2) )
        ddDomFunc.append(- d**2 * (np.tanh(d*(x-xl)) * sech(d*(x-xl))**2 - np.tanh(d*(x-xr)) * sech(d*(x-xr))**2) )
    

    lastDom = 1. - sum(domFunc)    
    dLastDom = - sum(dDomFunc)
    ddLastDom = - sum(ddDomFunc)

    domFunc.append(lastDom)
    dDomFunc.append(dLastDom)
    ddDomFunc.append(ddLastDom)
    
    if len(domFunc) != L:
        print('number of domain funcions',len(domFunc))
        sys.exit('the number of domains does not match domain functions')
    

    U  = np.zeros(len(x))
    fq = np.zeros(len(x)) 
      
    Nb = 2 # number of basis 
    
    for k in range(L):
        
        
        
        S = np.zeros((Nb,Nb))
        
        S[0,0] = np.dot(domFunc[k],w)

        print('number of trajectories in {} domain = {} \n'.format(k,S[0,0]))
        
        S[0,1] = S[1,0] = np.dot(x * domFunc[k], w)
        S[1,1] = np.dot(x**2 * domFunc[k], w)
    
        b = np.zeros(Nb)
        b[0] = np.dot(dDomFunc[k], w)
        b[1] = np.dot(domFunc[k] + x*dDomFunc[k], w)
    
        b = - 0.5*b
        
        a = np.linalg.solve(S,b)
        
        rk = trial(x,a)
        
        U += (rk**2 + a[1]) * domFunc[k] + rk * dDomFunc[k] 
        fq += 2.0 * rk * a[1] * domFunc[k] + (rk**2 + 2.0 * a[1]) * dDomFunc[k] + rk * dDomFunc[k]
        
    fq = fq/2.0/am 
    U = - U/2.0/am 
    
    # expectation value of quantum potential 
    Eu = np.dot(U,w)
    
    return Eu, fq 

# initialization    
Ntraj = 2048*4 
a0 = 2.0 
x0 = 0. 


x = np.random.randn(Ntraj) 

#x = np.zeros(Ntraj)  
#for k in range(Ntraj):
#    x[k] = np.random.randn() 
#    while x[k] > 3.0:
#        x[k] = np.random.randn()
    
        
x = x / np.sqrt(2.0 * a0) + x0 

print('initial configuration ranage {} {} \n'.format(min(x),max(x)))

p = np.zeros(Ntraj)
r = - a0 * (x-x0) 

w = np.array([1./Ntraj]*Ntraj)
am = paras.am 
print(' mass = {}'.format(am)) 

Nt = 1500 
dt = 0.002

dt2 = dt/2.0 
t = 0.0 


# initial guess for nonlinear optimization 
#p0_p = np.array((1.0, 1.0, 1.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0))
#p0_r = np.array((1.0, 1.0, 1.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0))

f = open('traj.dat','w')
fe = open('en.dat','w')
f_MSE = open('rMSE.dat','w')
f_ave = open('xAve.dat', 'w')

nout = 20       # number of trajectories to print 
fmt =  ' {}' * (nout+1)  + '\n'  
Eu = 0.  

L = 3 #  number of domains 

Ndim = 1        # dimensionality of the system  
fric_cons = 2.0     # friction constant  

v0, dv = derivs(x)
print('Initial potential energy = {} Hartree.'.format(np.dot(v0,w)))

Eu,fq = linear_fit_domain(x,w,L=L)

print('Start propagate the trajectories ...\n')

for k in range(Nt):
    t = t + dt 

    #print('force field')
    #print('classical force',-dv)
    #print('quantum force',fq)
    
    p += (- dv + fq) * dt2 - fric_cons * p * dt2   
    #r += fr * dt2
    
    x +=  p*dt/am

    # force field 
    
    Eu, fq = linear_fit_domain(x,w,L=L)

    if Eu < 0:
        pass 
# print('Error: U = {} should not be negative. \n'.format(Eu))
        #print('MSE = {}.'.format(rMSE))
                
# sys.exit()
        
    v0, dv = derivs(x)

    p += (- dv + fq) * dt2 - fric_cons * p * dt2 
    #r += fr * dt2 
       
    f.write(fmt.format(t,*x[0:nout]))
    Ek = np.dot(p*p,w)/2./am 
    Ev = np.dot(v0,w)
    Etot = Ek + Ev + Eu
    
    fe.write('{} {} {} {} {} \n'.format(t,Ek,Ev,Eu,Etot))
    f_ave.write('{} {} \n'.format(t, np.dot(x,w)))
    
print('Finish propagation ... \n')
print('The kinetic energy = {} Hartree \n'.format(Ek))
print('The potential energy = {} Hartree \n'.format(Ev))
print('Quantum potential energy = {} Hartree \n'.format(Eu))
print('The total energy = {} Hartree. \n'.format(Etot))

fe.close()
f.close() 
f_MSE.close() 

f = open('r.out', 'w')
for i in range(Ntraj):
    f.write('{} {} \n'.format(x[i], p[i]))
f.close()

#a, x0, De = 1.02, 1.4, 0.176/100 
#print('The well depth = {} cm-1. \n'.format(De * hartree_wavenumber))
#
#omega  = a * np.sqrt(2. * De / am )
#E0 = omega/2. - omega**2/16./De
#dE = (Etot-E0) * hartree_wavenumber 
#print('Exact ground-state energy = {} Hartree. \nEnergy deviation = {} cm-1. \n'.format(E0,dE))
#    


    
