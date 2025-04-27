#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 23:25:51 2017


Ehrenfest dynamics with a single mixed-classical trajectory 

@author: binggu

@status: not finished, work on single Ehrenfest trajectory first 
"""


import numpy as np
#import scipy  
import numba 
import sys 
import math 



bohr_angstrom = 0.52917721092
hartree_wavenumber = 219474.63 

#hartree_wavenumber = scipy.constants.value(u'hartree-inverse meter relationship') / 1e2 


def M1mat(a, Nb):
    
    M1 = np.zeros((Nb,Nb)) 

    for m in range(Nb-1):
        M1[m,m+1] = np.sqrt(float(m+1)/2.0/a)

    M1 = Sym(M1) 

    return M1 

def M2mat(a, Nb):
    
    M2 = np.zeros((Nb,Nb)) 

    for m in range(Nb):
        M2[m,m] = (float(m) + 0.5)/a 
    
    if Nb > 1: 
        for m in range(Nb-2):
            M2[m,m+2] = np.sqrt(float((m+1)*(m+2)))/2.0/a 

    M2 = Sym(M2)

    return M2 

def M3mat(a, Nb):
    
    M3 = np.zeros((Nb,Nb)) 

    for m in range(Nb-1):
        M3[m,m+1] = 3.0 * (float(m+1)/2./a)**1.5 

    if Nb > 2:
        for m in range(Nb-3):
            M3[m,m+3] = np.sqrt(float((m+1)*(m+2)*(m+3))) / (2.0*a)**1.5 
    
    M3 = Sym(M3) 

    return M3 

def M4mat(a, Nb):
    
    M4 = np.zeros((Nb,Nb))

    for m in range(Nb):
        M4[m,m] =  float(3.0 * m**2 + 3.0 * (m+1)**2) / (2.*a)**2
    
    if Nb > 1: 
        for m in range(Nb-2):
            M4[m,m+2] = (4.0*m + 6.0) * np.sqrt(float((m+1)*(m+2))) / (2.*a)**2
            
    if Nb > 3: 
        for m in range(Nb-4):
            M4[m,m+4] = np.sqrt(float((m+1)*(m+2)*(m+3)*(m+4))) / (2.0*a)**2

    M4 = Sym(M4) 

    if Nb > 1:
        if not M4[0,1] == M4[1,0]: 
            print(M4) 
            print('\n ERROR: Not symmetric matrix M4.\n')
            sys.exit() 
    return M4


def Hermite(x):

    cons = np.array([1. / np.sqrt(float(2**n) * float(math.factorial(n))) for n in range(Nb)])
    
    H = [] 
    H.append(1.0) 
    H.append( x * 2.0 ) 
    if Nb > 2:
        for n in range(2,Nb):
            Hn = 2.0 * x * H[n-1] - 2.0*(n-1) * H[n-2]
            H.append(Hn)
    
    for n in range(Nb):
        H[n] = H[n]*cons[n] 

    return H

#    if n == 0: 
#        H.append(1.)  
#    elif n == 1: 
#        return 2. * x * cons 
#    elif n == 2: 
#        return (4. * x**2 - 2.) * cons   
#    elif n == 3: 
#        return (8.0 * x**3 - 12.0 * x) * cons 
#    elif n == 4:
#        return (16.0 * x**4 - 48.0 * x**2 + 12.0) * cons 
#    elif n == 5:
#        return (32.0*x**5 - 160.0*x**3 + 120.0*x) * cons 
#    elif n == 6: 
#        return ()

def Vx(x):
    
    g = 0.1    
    return  x**2/2.0 + g * x**4 / 4.0

def Kmat(alpha,pAve, Nb):

    K = np.zeros((Nb,Nb),dtype=complex)

    ar = alpha.real 

    for j in range(Nb): 
        K[j,j] = np.abs(alpha)**2 / ar * (2. * j + 1.)/2. +  pAve**2 
    
    for j in range(1,Nb):
        K[j-1,j] = -1j*np.conj(alpha) * pAve * np.sqrt(2. * j / ar)
        K[j,j-1] = np.conj(K[j-1,j])

    if Nb > 2: 
        for j in range(2,Nb):
            K[j-2,j] = - np.sqrt(float((j-1)*j)) * np.conj(alpha)**2 / 2. / ar  
            K[j,j-2] = np.conj(K[j-2,j])
    

    #K[0,0] = np.abs(alpha)**2/alpha.real / 2. + pAve**2
    #K[1,1] = np.abs(alpha)**2/alpha.real * 3.0 / 2. + pAve**2 

    #K[0,1] = -1j*np.conj(alpha) * pAve * np.sqrt(2.*j/alpha.real)
    #K[1,0] = np.conj(K[0,1])
    K = K / (2.*amx) 

    return K 

def Sym(V):
    n = V.shape[-1]
    
    for i in range(n):
        for j in range(i):
            V[i,j] = V[j,i] 
    return V 

# @numba.autojit
def Vint(x,y):
    """
    interaction potential between x and y     
    """ 
    
    PES = 'HO' 
    
    if PES == 'Morse':
        
        a, x0 = 1.02, 1.4 
        De = 0.176 / 100.0 
    
        d = (1.0-np.exp(-a*x))
        
        v0 = De*d**2
            
        dv = 2. * De * d * a * np.exp(-a*x)
        
    elif PES == 'HO':
        
        v0 = x**2/2.0  + y**2/2.0 
         

    elif PES == 'AHO':
        
        eps = 0.4 
        
        v0 = x**2/2.0 + eps * x**4/4.0 
        dv = x + eps * x**3  
        #ddv = 2.0 * De * (-d*np.exp(-a*((x-x0)))*a**2 + (np.exp(-a*(x-x0)))**2*a**2)

#    elif PES == 'pH2':
#        
#        dx = 1e-4
#        
#        v0 = np.zeros(Ntraj)
#        dv = np.zeros(Ntraj)
#        
#        for i in range(Ntraj):
#            v0[i] = vpot(x[i])
#            dv[i] = ( vpot(x[i] + dx) - v0[i])/dx
        
    return v0 


def ground(x):
    return 0.5 * np.sum(x**2), x

def excited(x):
    return 0.5 * np.sum((x-1.0)**2) + 1.0, x - 1.0

# @numba.autojit 
def MeanField(y,c):
    
    V0, dV0 = ground(y) 
    V1, dV1 = excited(y)
    
    Vmf = abs(c[:,0])**2 * V0 + abs(c[:, 1])**2 * V1
    dVmf = abs(c[:, 0])**2 * dV0 + abs(c[:, 1])**2 * dV1
    
    return Vmf, dVmf 


class Ehrenfest:
    def __init__(self, ntraj, ndim, nstates):
        self.ntraj = ntraj
        self.ndim = ndim
        self.nstates = nstates
        self.c = np.zeros((ntraj,nstates),dtype=np.complex128)

        self.x = None # nuclear position    
        self.p = None # nuclear momentum
        self.w = None # weight of each trajectory
        
    def sample(self, temperature=300, unit='K'):
        
        if unit == 'K':
            temperature = temperature/au2k
        elif unit == 'au':
            temperature = temperature
        else:
            raise ValueError(f"Invalid unit: {unit}")
        
        self.x = np.random.randn(self.ntraj, self.ndim)
        self.x = self.x / np.sqrt(2.0 * self.ax) + self.x0

        self.p = np.zeros(self.ntraj, self.ndim)

        self.w = np.array([1./self.ntraj]*self.ntraj)
    
    def run(self, dt=0.002, nt=200):
        pass        
        
# initialization 
# for nuclear DOF  : an ensemble of trajectories 
# for electronic DOF  : for each trajectory associate a complex vector c of dimension M 

ntraj = Ntraj = 10
M = nstates = 2 
#nfit = 5
#ax = 1.0 # width of the GH basis 
ay0 = 16.0  
y0 = 0.1 

# initial conditions for c 
c = np.zeros((Ntraj,M),dtype=np.complex128)

# mixture of ground and first excited state

c[:,0] = 1.0/np.sqrt(2.0)+0j
c[:,1] = 1.0/np.sqrt(2.0)+0j
#for i in range(2,M):
#    c[:,i] = 0.0+0.0j

# coherent state 
#z = 1.0/np.sqrt(2.0) * x0 * np.sqrt(ax) 
#for i in range(M):
#    c[:,i] = np.exp(-0.5 * np.abs(z)**2) * z**i / np.sqrt(math.factorial(i))

print('initial occupation \n',c[0,:])
print('trace of density matrix',np.vdot(c[0,:], c[0,:]))
# ---------------------------------
# initial conditions for nuclear trajectory   

# ensemble of trajectories    
y = np.random.randn(ntraj)             
y = y / np.sqrt(2.0 * ay0) + y0
print('trajectory range {}, {}'.format(min(y),max(y)))

print('intial nuclear position',y)
py = np.zeros(Ntraj)
# ry = - ay0 * (y-y0) 

w = np.array([1./Ntraj]*Ntraj)

# -------------------------------

amx = 1.0 
amy = 1836.15 

f_MSE = open('rMSE.out','w')
nout = 1       # number of trajectories to print 
fmt =  ' {}' * (nout+1)  + '\n'  
#Eu = 0.  

Ndim = 1           # dimensionality of the nuclei    
fric_cons = 0.0      # friction constant  


Nt = 20000
dt = 0.002
dt2 = dt/2.0 
t = 0.0 

print('time range for propagation is [0,{}]'.format(Nt*dt))
print('timestep  = {}'.format(dt))
    
# construct the Hamiltonian matrix for anharmonic oscilator 
#g = 0.0 
#V = 0.5 * M2mat(ax,M) + g* M4mat(ax,M)
#K = Kmat(ax,0.0,M)
#H = K+V

#print('Hamiltonian matrix in DOF x = \n')
#print(H)
#print('\n')

#eps = 0.5 # nonlinear coupling Vint = eps*x**2*y**2

# @numba.autojit 
def den(c,w):
    """
        compute reduced density matrix elements 
    """
    rho = np.zeros((M,M),dtype=np.complex128)
    for k in range(Ntraj):
        for i in range(M):
            for j in range(M):
                rho[i,j] += c[k,i] * np.conjugate(c[k,j]) * w[k]
    
    rho2 = np.dot(rho,rho)
        
    purity = 0.0+0.0j
    for i in range(M):
        purity += rho2[i,i]
        
    return rho[0,1], purity.real  
        
# @numba.autojit 
def norm(c,w):
    
    anm = 0.0 

    for k in range(Ntraj):
        anm += np.vdot(c[k,:], c[k,:]).real * w[k]
    return anm

# @numba.autojit
def fit_c(c,y):
    """
    global approximation of c vs y to obtain the derivative c'',c'     
    """
    dc = np.zeros((Ntraj,M),dtype=np.complex128)
    ddc = np.zeros((Ntraj,M),dtype=np.complex128)
    
    for j in range(M):

        z = c[:,j]
        pars = np.polyfit(y,z,nfit)
        p0 = np.poly1d(pars)
        p1 = np.polyder(p0)
        p2 = np.polyder(p1)
#for k in range(Ntraj):
        dc[:,j] = p1(y)
        ddc[:,j] = p2(y)
            
    return dc, ddc
    
# @numba.autojit 
def prop_c(y):
    
    # dc, ddc = fit_c(c,y)

    dcdt = np.zeros([ntraj,M],dtype=np.complex128)
    
    
    #X1 = M1mat(ax,M)
    for k in range(ntraj):
        
        H = np.zeros((nstates, nstates))
        H[0,0] = ground(y[k])[0]
        H[0,1] = H[1,0] = 0.0 
        H[1,1] = excited(y[k])[0]
         
        # anharmonic term in the bath potential 
        #Va = y[k]**4 * 1.0
        
        tmp = H.dot(c[k,:])

        dcdt[k,:] = -1j * tmp
       
    return dcdt
    
# @numba.autojit 
def xAve(c,y,w):
    """
    compute expectation value of x     
    """
    Xmat = M1mat(ax,M)

    x_ave = 0.0+0.0j    
    for k in range(Ntraj):
        for m in range(M):
            for n in range(M):
                x_ave += Xmat[m,n] * np.conjugate(c[k,m]) * c[k,n] * w[k]
    
    return x_ave.real 
    
# propagate the QTs for y 


# update the coeffcients for each trajectory 
fmt_c = ' {} '* (M+1)
  
f = open('traj.dat','w')
fe = open('en.out','w')
fc = open('c.dat','w')
fx = open('xAve.dat','w')
fnorm = open('norm.dat', 'w')
fden = open('den.dat','w')


v0, dv = MeanField(y,c)

cold = c 
dcdt = prop_c(y)
c = c + dcdt * dt

for k in range(Nt):
    
    t = t + dt 

    py += - dv * dt2 - fric_cons * py * dt2   
    
    y +=  py*dt/amy

    # force field 
        
    # x_ave = xAve(c,y,w)
    v0, dv = MeanField(y,c)

    py += - dv * dt2 - fric_cons * py * dt2 
    
    # renormalization 

    #anm = norm(c,w)
    #c /= np.sqrt(anm)
    
    # update c 
   
    dcdt = prop_c(y)
    cnew = cold + dcdt * dt * 2.0
    cold = c 
    c = cnew

    
    #  output data for each timestep 
#    d = c
#    for k in range(Ntraj):
#        for i in range(M):
#            d[k,i] = np.exp(-1j*t*H[i,i])*c[k,i]


    # fx.write('{} {} \n'.format(t,x_ave))
           
    f.write(fmt.format(t,*y[0:nout]))

    #fnorm.write(' {} {} \n'.format(t,anm))

    # output density matrix elements 
    rho, purity = den(c,w)
    fden.write(' {} {} {} \n'.format(t,rho, purity))
    
    Ek = np.dot(py*py,w)/2./amy  
    Ev = np.dot(v0,w) 
    #Eu = Eu 
    Etot = Ek + Ev 
    
    fe.write('{} {} {} {} \n'.format(t,Ek,Ev,Etot))


print('The total energy = {} Hartree. \n'.format(Etot))

# print trajectory and coefficients 
for k in range(Ntraj):
    fc.write( '{} {} {} \n'.format(y[k], c[k,0],c[k,-1]))

fe.close()
f.close() 
fc.close()
fx.close()


#a, x0, De = 1.02, 1.4, 0.176/100 
#print('The well depth = {} cm-1. \n'.format(De * hartree_wavenumber))
#
#omega  = a * np.sqrt(2. * De / am )
#E0 = omega/2. - omega**2/16./De
#dE = (Etot-E0) * hartree_wavenumber 
#print('Exact ground-state energy = {} Hartree. \nEnergy deviation = {} cm-1. \n'.format(E0,dE))
#    


    
