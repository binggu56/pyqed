# -*- coding: utf-8 -*-
"""

Created on Fri Jan 13 14:54:11 2017

Floquet theory for periodic Hamiltonian 

a tight-binding model with two-bands 

@author: bingg

"""

import numpy as np 
import matplotlib.pyplot as plt 
import numba 

from pyqed import heaviside

import sys 

def delta(i,j):
    if i==j: 
        return 1
    else:
        return 0 
        


# @numba.jit        
def Hamilton(Norbs):
    """
    Norbs: number of atomic orbitals 

    """
    
    NPS = 2 # norbs per site 
    
    ta = 1.0 
    tb = 0.7
  
    onsite1 = -2.0 
    onsite2 = 2.0  
    
    fock0 = np.zeros((Norbs,Norbs))

    Nsites = int(Norbs/NPS)     
    
    for i in range(Nsites):
        fock0[2*i, 2*i] = onsite1
        fock0[2*i+1,2*i+1] = onsite2
    
    for i in range(Nsites):
        fock0[2*i,2*i+1] = -ta
        fock0[2*i+1, 2*i] = -ta
        
    for i in range(Nsites-1):
        fock0[2*i+2, 2*i+1]  = -tb
        fock0[2*i+1, 2*i+2] = -tb  

#    two-level system    
#    h = np.zeros((Norbs,Norbs))
#    # diagonal elements     
#
#    h[0,0] = onsite0 
#    h[1,1] = onsite1
#    
#    # off-diagonal elements     
#    for i in range(Norbs-1):
#        h[i,i+1] = 0.5 
#        h[i+1,i] = 0.5
        
    sys_eigvals, sys_eigvecs = np.linalg.eigh(fock0)
    
    
    return sys_eigvecs, sys_eigvals

#@numba.autojit     
def basisTransform(U):
    """
    transform the electric dipole interaction from atomic basis to eigenstates of H_S
    M = - e E0 n a |n><n|
        a : atomic distance [A]
        E0 : electric field amplitude [V/A]
    """

    M = np.zeros((Norbs, Norbs))
    
    Nsites = int(Norbs/2)
    
    for i in range(Nsites):
        
        M[2*i, 2*i] = (i - 0.5 * (Nsites - 1) ) * unitLength                                                                       
        M[2*i+1, 2*i+1] = M[2*i, 2*i] 

    M = - E0 * M
    
    return np.matmul(U.conj().T, np.matmul(M,U)) 

#@numba.autojit 
def HamiltonFT(sysEigvals, M, n):
    

    
    if n == 0:
        
        H = np.zeros((Norbs,Norbs))
        
        for i in range(Norbs):
            H[i,i] = sysEigvals[i]
            
        return H 
        
    elif n == 1 or n == -1:
        return 0.5 * M
    
    else:
        
        return np.zeros((Norbs,Norbs))

    
def FloquetHamilton(Norbs, Nt):
    """
    construct a Floquet hamiltonian of the size Norbs * Nt  
    """
    global omega # frequency of the radiation  


    U, sysEigvals = Hamilton(Norbs)
    M = basisTransform(U)
    
    NF = Norbs * Nt 
    F = np.zeros((NF,NF))
    
    N0 = -(Nt-1)/2 # starting point for Fourier companent of time exp(i n w t)
    
    # for a general tight-binding Hamiltonian 
    
    for n in range(Nt):
        for k in range(Norbs):
        # we need to map the index i to double-index (n,k) n : time Fourier component 
        # k : atomic bassi index
        
        # a relationship for this is Norbs * n + k ---> i
            i = Norbs * n + k 
            
            for m in range(Nt):
                for l in range(Norbs):
                    j = Norbs * m + l 
                    
                    F[i,j] = HamiltonFT(sysEigvals, M, n-m)[k,l] + (n + N0) * omega * delta(n,m) * delta(k,l) 
                    
    
    # for a two-state model 
 
#    for n in range(Nt):
#        for m in range(Nt):
#            F[n * Norbs, m * Norbs] = (N0 + n) * omega * delta(n,m)
#            F[n * Norbs + 1, m * Norbs + 1] = (onsite1 + (N0+n) * omega) * delta(n,m)
#            F[n * Norbs, m * Norbs + 1] = t * delta(n,m+1)
#            F[n * Norbs + 1, m * Norbs] = t * delta(n,m-1)
            
    # after construct the Floquet Hamiltonian, the eigenvalues are computed 
    eigvals, eigvecs = np.linalg.eigh(F)
    
    print('Floquet states', eigvals)
    
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
        print("Error: number of Floquet states is not equal to the number of orbitals. {} \n".format(j))
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

# def heaviside(x):
#     return 0.5 * (np.sign(x) + 1)
    
def spectrum():
    """
        C : expansion coefficients 
        F : eigenstates of Floquet Hamiltonian (a subset of it)
        M : dipole operator matrix element 
    """
    quasiE,F, C = FloquetHamilton(Norbs, Nt)
    
    U, sys_eigvals = Hamilton(Norbs)

    M = basisTransform(U)
    
    fermi = 0.0 
    
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
        
        tmp = 0.0 
        
        for nn in range(Nt-l):
            for n in range(Nt-l):
                for ii in range(Norbs):
                    for jj in range(Norbs):
                        for k in range(Norbs):
                            #for kk in range(Norbs):
                            for kkk in range(Norbs):
                                    #for kkkk in range(Norbs):
                                tmp += D[nn+l,nn,ii,ii,k,k] * D[n,n+l,jj,jj,kkk,kkk] * heaviside(fermi - sys_eigvals[k]) * heaviside(fermi - sys_eigvals[kkk]) \
                                     + D[nn+l,nn,ii,ii,k,kkk] * D[n,n+l,jj,jj,kkk,k] * heaviside(fermi - sys_eigvals[k]) * heaviside(sys_eigvals[kkk] - fermi) 
                                    
                                    
        fIntra.write('{} {} \n'.format(l * omega, tmp))
        
    for l in range(-Nt,Nt):
        
        for ii in range(Norbs):
            for jj in range(Norbs):
                
                if ii != jj:

                    tmp = 0.0  
                    
                    for n in range(max(0,-l), min(Nt-l,Nt)):
                        for nn in range(max(0,-l), min(Nt-l,Nt)):
            
                            for k in range(Norbs):
                                #for kk in range(Norbs):
                                for kkk in range(Norbs):
                                       # for kkkk in range(Norbs):
                                            
                                    tmp += D[n+l,n,ii,jj,k,k] * D[nn,nn+l,jj,ii,kkk,kkk] * heaviside(fermi - sys_eigvals[k]) * heaviside(fermi - sys_eigvals[kkk]) \
                                          +D[n+l,n,ii,jj,k,kkk] * D[nn,nn+l,jj,ii,kkk,k] * heaviside(fermi - sys_eigvals[k]) * heaviside(sys_eigvals[kkk] - fermi)
                                                    
                                                    
                    fInter.write('{} {} \n'.format(quasiE[ii] - quasiE[jj] + l * omega, tmp))
                    
#    for l in range(-Nt,0):
#        
#        for ii in range(Norbs):
#            for jj in range(Norbs):
#
#                tmp  = 0.0 
#                
#                for nn in range(-l,Nt):
#                    for n in range(-l,Nt):
#        
#                        for k in range(Norbs):
#                            for kk in range(Norbs):
#                                for kkk in range(Norbs):
#                                    for kkkk in range(Norbs):
#                                        tmp += D[nn+l,nn,ii,jj,k,kk] * D[n,n+l,jj,ii,kkk,kkkk] * Gamma(k,kk,kkk,kkkk,sysEigvals)
#                                                
#                fInter.write('{} {} \n'.format(quasiE[ii] - quasiE[jj] + l * omega, tmp))
#                        
    fInter.close()
    fIntra.close() 
    
    return 
    
    
    
    
   


Nt = 21 # has to be odd integer 
Norbs = 12 

omega = 0.5
 
#onsite0 = 0.0
#onsite1 = 2.0

E0 = 0.5
 
unitLength = 2. 


# spectrum()






#x = range(Norbs * Nt)
#plt.plot(x , eigvals,'-o',markersize=8)
#plt.axhline(y = quasiE0,lw=1)
#plt.axhline(y = quasiE1,lw=1)
#
#plt.show()




                    
