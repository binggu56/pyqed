# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:54:11 2017

Floquet theory for periodic Hamiltonian 

@author: bingg
"""

import numpy as np 
import matplotlib.pyplot as plt 

import sys 

def delta(i,j):
    if i==j: 
        return 1
    else:
        return 0 
        
def Hamilton(Norbs):
    """
    Norbs: number of atomic orbitals 

    """

    
    h = np.zeros((Norbs,Norbs))
    # diagonal elements     
    #for i in range(Norbs):
    #    h[i,i] = E0 

    h[0,0] = onsite0 
    h[1,1] = onsite1
    
    # off-diagonal elements     
    for i in range(Norbs-1):
        h[i,i+1] = 0.5 
        h[i+1,i] = 0.5
        
    sys_eigvals, sys_eigvecs = np.linalg.eigh(h)
    
    print('system eigenvalues',sys_eigvals)
    
    return sys_eigvecs, sys_eigvals
    
def basisTransform(U):
    """
    transform the electric dipole interaction from atomic basis to eigenstates of H_S
    M = - e E0 n a |n><n|
        a : atomic distance [A]
        E0 : electric field amplitude [V/A]
    """


    M = np.zeros((Norbs, Norbs))
    for i in range(Norbs):
        M[i,i] = - E0 * a * (i - 0.5 * (Norbs-1))
    
    Uh = np.conj(M.transpose())
    
    return np.matmul(Uh, np.matmul(M,U)) 

def FloquetHamilton(Norbs, Nt):
    """
    construct a Floquet hamiltonian of the size Norbs * Nt  
    """
    global omega # frequency of the radiation  

    
    NF = Norbs * Nt 
    F = np.zeros((NF,NF))
    
#    for n in range(Nt):
#        for k in range(Norbs):
#        # we need to map the index i to double-index (n,k) n : time Fourier component 
#        # k : atomic bassi index
#        
#        # a relationship for this is Norbs * n + k ---> i
#            i = Norbs * n + k 
#            for m in range(Nt):
#                for l in range(Norbs):
#                    j = Norbs * m + l 
#                    
#                    F[i,j] =  
    
    # for a two-state model 
    
    # starting point of Fourier component exp(i n\omega t)
    N0 = -(Nt-1)/2 
    for n in range(Nt):
        for m in range(Nt):
            F[n * Norbs, m * Norbs] = (N0 + n) * omega * delta(n,m)
            F[n * Norbs + 1, m * Norbs + 1] = (onsite1 + (N0+n) * omega) * delta(n,m)
            F[n * Norbs, m * Norbs + 1] = t * delta(n,m+1)
            F[n * Norbs + 1, m * Norbs] = t * delta(n,m-1)
            
    # after construct the Floquet Hamiltonian, the eigenvalues are computed 
    eigvals, eigvecs = np.linalg.eigh(F)
    
    print(eigvals)
    
    # specify a range to choose the quasienergies, choose [-hbar omega/2, hbar * omega/2]
    eigvals_subset = np.zeros(Norbs)
    eigvecs_subset = np.zeros((NF , Norbs))


    j = 0     
    for i in range(NF):
        if  eigvals[i] < omega/2.0 and eigvals[i] > -omega/2.0:
            eigvals_subset[j] = eigvals[i]
            eigvecs_subset[:,j] = eigvecs[:,i]
            j += 1 
    if j != Norbs: 
        print("Error: nuber of Floquet states is not equal to the number of orbitals. {} \n".format(j))
        sys.exit() 
        
        
    # now we have a complete linear independent set of solutions for the TD problem 
    # to compute the coefficients before each Floquet state if we start with |alpha>
    # at time 0, this is done by solving a matrix equation CG = 1 
    G = np.zeros((Norbs,Norbs))
    for i in range(Norbs):
        for j in range(Norbs):
            tmp = 0.0 
            for m in range(Nt):
                tmp += eigvecs_subset[m * Norbs + j, i]
            G[i,j] = tmp
            
    C = np.linalg.inv(G)
    
    return eigvals_subset, eigvecs_subset, C   

def Gamma(i,j,ii,jj, sys_eigvals):
    
    # fermi energy 
    fermi = 1.0
    
    tmp = delta(i,j) * delta(ii,jj) * heaviside(fermi - sys_eigvals[i]) * heaviside(fermi - sys_eigvals[ii])
    
    tmp +=  delta(ii,j) * delta(i,jj) * heaviside(fermi - sys_eigvals[i]) * heaviside(sys_eigvals[ii] - fermi)
    
    return tmp 

def heaviside(x):
    return 0.5 * (np.sign(x) + 1)
    
def spectrum():
    """
        C : expansion coefficients 
        F : eigenstates of Floquet Hamiltonian (a subset of it)
        M : dipole operator matrix element 
    """
    quasiE,F, C = FloquetHamilton(Norbs, Nt)
    
    U, sysEigvals = Hamilton(Norbs)

    M = basisTransform(U)
    
    fIntra = open('intra.dat','w')
    fInter = open('inter.dat','w')
    
    D = np.zeros((Nt, Nt, Norbs, Norbs, Norbs, Norbs))
    
    for ii in range(Norbs):
        for jj in range(Norbs):
            for n in range(Nt):
                for nn in range(Nt):
                    for k in range(Norbs):
                        for kk in range(Norbs):
                            tmp = 0.0 
                            for i in range(Norbs):
                                for j in range(Norbs):
                                    tmp += M[i,j] * C[i,ii] * np.conj(C[j,jj]) * F[n * Norbs + k, ii] * \
                                            np.conj(F[nn * Norbs + kk, jj])
                                            
                            D[n,nn,ii,jj,k,kk] = tmp 

        
    for l in range(Nt):

       # ll = l - (Nt-1)/2
        
        tmp  = 0.0 
        for nn in range(Nt-l):
            for n in range(Nt-l):
                for ii in range(Norbs):
                    for jj in range(Norbs):
                        for k in range(Norbs):
                            for kk in range(Norbs):
                                for kkk in range(Norbs):
                                    for kkkk in range(Norbs):
                                        tmp += D[nn+l,nn,ii,ii,k,kk] * D[n,n+l,jj,jj,kkk,kkkk] * Gamma(k,kk,kkk,kkkk,sysEigvals)
        fIntra.write('{} {} \n'.format(l * omega, tmp))
        
    for l in range(-Nt,Nt):
        
        for ii in range(Norbs):
            for jj in range(Norbs):
                
                if ii != jj:

                    tmp  = 0.0 
                    
                    for n in range(max(0,-l), min(Nt-l,Nt)):
                        for nn in range(max(0,-l), min(Nt-l,Nt)):
            
                            for k in range(Norbs):
                                for kk in range(Norbs):
                                    for kkk in range(Norbs):
                                        for kkkk in range(Norbs):
                                            tmp += D[n+l,n,ii,jj,k,kk] * D[nn,nn+l,jj,ii,kkk,kkkk] * Gamma(k,kk,kkk,kkkk,sysEigvals)
                                                    
                    fInter.write('{} {} \n'.format(quasiE[ii] - quasiE[jj] + l * omega, tmp))
    
    
   
Nt = 9 # has to be odd integer 
Norbs = 2 

omega = 1.0
 
onsite0 = 0.0
onsite1 = 2.0

E0 = 0.5 
a = 2. 
t = 1.0


quasiE0 = (1.0-np.sqrt(5.0))/2.0 + 1.0
quasiE1 = (1.0+np.sqrt(5.0))/2.0 - 2.0

print('Quasienergies')
print(quasiE0, quasiE1)






#x = range(Norbs * Nt)
#plt.plot(x , eigvals,'-o',markersize=8)
#plt.axhline(y = quasiE0,lw=1)
#plt.axhline(y = quasiE1,lw=1)
#
#plt.show()




                    