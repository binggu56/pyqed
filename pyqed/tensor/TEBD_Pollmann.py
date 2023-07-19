""" Comparison of the entanglement growth S(t) following a global quench using
the first order MPO based time evolution for the XX model with the TEBD algorithm.

The TEBD algorithm is also used to recompress the MPS after the MPO time evolution.
This is simpler to code but less efficient than a variational optimization.
See arXiv:1407.1832 for details and how to extend to higher orders.

Frank Pollmann, frankp@pks.mpg.de 

Ref: SciPost Phys. Lect. Notes 5 (2018) · published 8 October 2018


"""

import numpy as np
import pylab as pl
from scipy.linalg import expm


s0 = np.eye(2)
sp = np.array([[0.,1.],[0.,0.]])
sm = np.array([[0.,0.],[1.,0.]])

def initial_state_af(d,chi_max,L,dtype=complex):
    """ 
    Create an antiferromagnetc product state. 
    The MPS is put in the right canonical form, 
    MPS = S[0]--B[0]--B[1]--...--B[L-1]
    """
    B_list = []
    s_list = []
    for i in range(L):
        B = np.zeros((d,1,1),dtype=dtype)
        B[np.mod(i,2),0,0] = 1.
        s = np.zeros(1)
        s[0] = 1.
        B_list.append(B)
        s_list.append(s)
    s_list.append(s)
    return B_list,s_list

def make_U_xx_bond(L,delta):
    " Create the bond evolution operator used by the TEBD algorithm."
    d = 2
    H = np.real(np.kron(sp,sm) + np.kron(sm,sp))
    u_list = (L-1)*[np.reshape(expm(-delta*H),(2,2,2,2))]
    return u_list,d

def make_U_xx_mpo(L,dt,dtype=float):
    """ 
    Create the MPO of the time evolution operator.  
    
    w =  I      S+    S-
        -dt*S- 
        -dt*S+
        
    This MPO generates 
    ..math::
        H_{xx} = \sum_{<i,j>} S_i S_j^\dagger + H.c.
        
    """

    w = np.zeros((3,3,2,2),dtype=type(dt))
    w[0,:] = [s0,sp,sm]
    w[1:,0] = [-dt*sm,-dt*sp]
    w_list = [w]*L
    return w_list

def apply_mpo_svd(B_list,s_list,w_list,chi_max):
    '''
    Apply the MPO to an MPS.
    '''
    d = B_list[0].shape[0] # Hilbert space dimension
    
    D = w_list[0].shape[0] # bond dimension for MPO
    
    L = len(B_list) # nsites


    # first site, only use the first row of the W[0, :]
    
    chi1 = B_list[0].shape[1] # MPS bond dimension left
    chi2 = B_list[0].shape[2] # MPS bond dimension right
    
    # B = W[0]_{ij, cd} B_{i, ab} 
    B = np.tensordot(B_list[0],w_list[0][0,:,:,:],axes=(0,1))
    B = np.reshape(np.transpose(B,(3,0,1,2)),(d,chi1,chi2*D))
    
    B_list[0] = B

    # for sites l = 2 to L-1
    for i_site in range(1,L-1):
        chi1 = B_list[i_site].shape[1]
        chi2 = B_list[i_site].shape[2]
        B = np.tensordot(B_list[i_site],w_list[i_site][:,:,:,:],axes=(0,2))
        B = np.reshape(np.transpose(B,(4,0,2,1,3)),(d,chi1*D,chi2*D))
        B_list[i_site] = B
        s_list[i_site] = np.reshape(np.tensordot(s_list[i_site],np.ones(D),axes=0),D*chi1)

    # the last site 
    chi1 = B_list[L-1].shape[1]
    chi2 = B_list[L-1].shape[2]

    B = np.tensordot(B_list[L-1],w_list[L-1][:,0,:,:],axes=(0,1))
    B = np.reshape(np.transpose(B,(3,0,2,1)),(d,D*chi1,chi2))
    s_list[L-1] = np.reshape(np.tensordot(s_list[L-1],np.ones(D),axes=0),D*chi1)
    B_list[L-1] = B

    # reduce bond dimension
    tebd(B_list,s_list,(L-1)*[np.reshape(np.eye(d**2),[d,d,d,d])],chi_max)

    return 

def tebd(B_list,s_list,U_list,chi_max):
    " Use TEBD to optmize the MPS and to project it back. "
    d = B_list[0].shape[0]
    L = len(B_list)

    for p in [0,1]:
        for i_bond in np.arange(p,L-1,2):
            i1=i_bond
            i2=i_bond+1

            chi1 = B_list[i1].shape[1]
            chi3 = B_list[i2].shape[2]

            # Construct theta matrix #
            C = np.tensordot(B_list[i1],B_list[i2],axes=(2,1))
            C = np.tensordot(C,U_list[i_bond],axes=([0,2],[0,1]))

            theta = np.reshape(np.transpose(np.transpose(C)*s_list[i1],(1,3,0,2)),(d*chi1,d*chi3))
            C = np.reshape(np.transpose(C,(2,0,3,1)),(d*chi1,d*chi3))
            # Schmidt decomposition #
            X, Y, Z = np.linalg.svd(theta)
            Z=Z.T

            W = np.dot(C,Z.conj())
            chi2 = np.min([np.sum(Y>10.**(-8)), chi_max])

            # Obtain the new values for B and l #
            invsq = np.sqrt(sum(Y[:chi2]**2))
            s_list[i2] = Y[:chi2]/invsq
            B_list[i1] = np.reshape(W[:,:chi2],(d,chi1,chi2))/invsq
            B_list[i2] = np.transpose(np.reshape(Z[:,:chi2],(d,chi3,chi2)),(0,2,1))

if __name__ == "__main__":
    # Define Pararemeter here
    delta = 0.02
    L = 20
    chi_max = 40
    N_steps = 100

    # MPO based time evolution (this one can be extended to also include long range interactions by using a different MPO)
    w_list = make_U_xx_mpo(L, 1j*delta)
    B_list,s_list = initial_state_af(2,chi_max,L,dtype=complex)
    S = [0]

    for step in range(N_steps):
        # evolve dt
        apply_mpo_svd(B_list,s_list,w_list,chi_max)

        # compute vN entropy
        s2 = np.array(s_list[L//2])**2
        S.append(-np.sum(s2*np.log(s2)))

    pl.plot(delta*np.arange(N_steps+1),S)

    # TEBD algorithm
    u_list,d =  make_U_xx_bond(L,1j*delta)
    B_list,s_list = initial_state_af(2,chi_max,L,dtype=complex)
    S = [0]
    for step in range(N_steps):
        tebd(B_list,s_list,u_list,chi_max)
        s2 = np.array(s_list[L//2])**2
        S.append(-np.sum(s2*np.log(s2)))

    pl.plot(delta*np.arange(N_steps+1),S)
    pl.xlabel('$t$')
    pl.ylabel('$S$')
    pl.legend(['MPO','TEBD'],loc='upper left')
    pl.show()