#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:32:53 2024

Src: 
    https://github.com/NLESC-JCER/DavidsonPython/blob/master/davidson.py


"""

import numpy as np
import math
import time 
import logging


def digaonal_dominant(n,sparsity=1E-4):
    
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i,i] = 1E3*np.random.rand() 
        #A[i,i] = i+1
    A = A + sparsity*np.random.randn(n,n) 
    A = (A.T + A)/2 
    return A

def diag_non_tda(n,sparsity=1E-4):

    A = digaonal_dominant(n)
    C = sparsity*np.random.rand(n,n)

    return np.block([ [A,C],[-C.T,-A.T] ])



def jacobi_correction(uj,A,thetaj):
    I = np.eye(A.shape[0])
    Pj = I-np.dot(uj,uj.T)
    rj = np.dot((A - thetaj*I),uj) 

    w = np.dot(Pj,np.dot((A-thetaj*I),Pj))
    return np.linalg.solve(w,rj)


def get_initial_guess(A,neigen):
    nrows, ncols = A.shape
    d = np.diag(A)
    index = np.argsort(d)
    guess = np.zeros((nrows,neigen))
    for i in range(neigen):
        guess[index[i],i] = 1
    
    return guess


def reorder_matrix(A):
    
    n = A.shape[0]
    tmp = np.zeros((n,n))

    index = np.argsort(np.diagonal(A))

    for i in range(n):
        for j in range(i,n):
            tmp[i,j] = A[index[i],index[j]]
            tmp[j,i] = tmp[i,j]
    return tmp

def davidson_solver(A, neigen, tol=1E-6, itermax = 1000, jacobi=False):
    """Davidosn solver for eigenvalue problem
    
    Seems quite slow!

    Args :
        A (numpy matrix) : the matrix to diagonalize
        neigen (int)     : the number of eigenvalue requied
        tol (float)      : the rpecision required
        itermax (int)    : the maximum number of iteration
        jacobi (bool)    : do the jacobi correction
    Returns :
        eigenvalues (array) : lowest eigenvalues
        eigenvectors (numpy.array) : eigenvectors
    """
    n = A.shape[0]
    k = 2*neigen            # number of initial guess vectors 
    V = np.eye(n,k)         # set of k unit vectors as guess
    I = np.eye(n)           # identity matrix same dimen as A
    Adiag = np.diag(A)

    V = get_initial_guess(A,k)
    
    print('\n'+'='*20)
    print("= Davidson Solver ")
    print('='*20)

    #invA = np.linalg.inv(A)
    #inv_approx_0 = 2*I - A
    #invA2 = np.dot(invA,invA)
    #invA3 = np.dot(invA2,invA)

    norm = np.zeros(neigen)

    # Begin block Davidson routine
    print("iter size norm (%e)" %tol)
    for i in range(itermax):
    
        # QR of V t oorthonormalize the V matrix
        # this uses GrahmShmidtd in the back
        V,R = np.linalg.qr(V)

        # form the projected matrix 
        T = np.dot(V.T,np.dot(A,V))


        # Diagonalize the projected matrix
        theta,s = np.linalg.eigh(T)

        # Ritz eigenvector
        q = np.dot(V,s)

        # compute the residual append append it to the 
        # set of eigenvectors
        
        for j in range(neigen):

            # residue vetor
            res = np.dot((A - theta[j]*I),q[:,j]) 
            norm[j] = np.linalg.norm(res)

            # correction vector
            if(jacobi):
            	delta = jacobi_correction(q[:,j],A,theta[j])
            else:
            	delta = res / (theta[j]-Adiag+1E-16)
                #C = inv_approx_0 + theta[j]*I
                #delta = -np.dot(C,res)

            delta /= np.linalg.norm(delta)

            # expand the basis
            V = np.hstack((V,delta.reshape(-1,1)))

        # comute the norm to se if eigenvalue converge
        logging.info(" %03d %03d %e" %(i,V.shape[1],np.max(norm)))
        if np.all(norm < tol):
            print("= Davidson has converged")
            break
        
    return theta[:neigen], q[:,:neigen]




def block_davidson(A, neig=3, max_iterations=20, tol = 1e-9):
    """
    Bloch davidson algorithm

    Refs
        [1] https://github.com/sreeganb/davidson_algorithm/blob/master/davidson.py

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    neig : TYPE, optional
        DESCRIPTION. The default is 3.
    max_iterations: TYPE, optional
        Maximum number of iterations. The default is 20.
    tol : TYPE, optional
        Convergence tolerance. The default is 1e-9.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

             
    # Setup the subspace trial vectors
    k = neig + 1 
    logging.info('No. of start vectors:',k)

    logging.info('No. of desired Eigenvalues:',neig)
    
    n = A.shape[0]
    
    t = np.eye(n,k) # initial trial vectors
    v = np.zeros((n,n)) # holder for trial vectors as iterations progress
    I = np.eye(n) # n*n identity matrix
    
    ritz = np.zeros((n,n))
    f = np.zeros((n,n))
    #-------------------------------------------------------------------------------
    # Begin iterations  
    #-------------------------------------------------------------------------------
    # start = time.time()
    iter = 0
    for m in range(k, max_iterations, k):
        iter = iter + 1
        logging.info("Iteration no:", iter)
        if iter==1:  # for first iteration add normalized guess vectors to matrix v
            for l in range(m):
                v[:,l] = t[:,l]/(np.linalg.norm(t[:,l]))
                
        # Matrix-vector products, form the projected Hamiltonian in the subspace
        T = np.linalg.multi_dot([v[:,:m].T,A,v[:,:m]]) # selects fastest evaluation order
        
        w, vects = sort(*np.linalg.eig(T)) # Diagonalize the subspace Hamiltonian
        jj = 0
        # s = w.argsort()
        # ss = w[s]
        # vects = vects[:, s]
        #***************************************************************************
        # For each eigenvector of T build a Ritz vector, precondition it and check
        # if the norm is greater than a set threshold.
        #***************************************************************************
        for ii in range(m): #for each new eigenvector of T
            f = np.diag(1./ np.diag((np.diag(np.diag(A)) - w[ii]*I)))
    #        print (f)
            
            ritz[:,ii] = np.dot(f,np.linalg.multi_dot([(A-w[ii]*I),v[:,:m],vects[:,ii]]))
            if np.linalg.norm(ritz[:,ii]) > 1e-7 :
                ritz[:,ii] = ritz[:,ii]/(np.linalg.norm(ritz[:,ii]))
                v[:,m+jj] = ritz[:,ii]
                jj = jj + 1
        
        eigvecs = v[:, :m] @ vects
        
        q, r = np.linalg.qr(v[:,:m+jj-1])
        
        for kk in range(m+jj-1):
            v[:,kk] = q[:,kk]
            
        # for ii in range(neig):
        #     print (ss[ii])
        
        if iter==1: 
            check_old = w[:neig]
            check_new = 1
        elif iter==2:
            check_new = w[:neig]
        else: 
            check_old = check_new
            check_new = w[:neig]
            
        check = np.linalg.norm(check_new - check_old)
        if check < tol:
            logging.info('Block Davidson converged at iteration no.:',iter)
            break
    
    # end = time.time()
    # print('Block Davidson time:',end-start)
    
    # print(ritz.shape)
    
    return w[:neig], eigvecs

if __name__=='__main__':
    
    from pyqed import sort
    import proplot as plt
    # Build a fake sparse symmetric matrix 
    n = 1000
    print('Dimension of the matrix',n,'*',n)
    sparsity = 0.01
    A = np.zeros((n,n))
    for i in range(0,n) : 
        A[i,i] = i-9
    A = A + sparsity*np.random.randn(n,n)
    A = (A.T + A)/2
    
    neig = 3
    start = time.time()
    w, u = davidson_solver(A, neig)
    print(w)
    end = time.time() 
    print ('Davidson Diagonalization time:',end-start)
    
    start = time.time()
    eig, eigvecs = sort(*np.linalg.eig(A))
    end = time.time() 
    # s = eig.argsort()
    # ss = eig[s]
    print(np.allclose(np.abs(u[:, 1]), np.abs(eigvecs[:, 1]), atol=1e-6, rtol=1e-5))
    # fig, ax = plt.subplots()
    # ax.plot(u[:, 0], '--')
    # ax.plot(eigvecs[:, 0])

    print('Exact Diagonalization:')

    print(eig[:neig])
    #print(ss[:neig])
    print ('Exact Diagonalization time:',end-start)