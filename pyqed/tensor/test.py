# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 09:43:38 2019

@author: Bing
"""

##### Lets initialize some tensors in Python/Numpy
import numpy as np
from numpy import linalg as LA
from ncon import ncon


# tensor with randomly generated entries, order 3, dims: 2-by-3-by-4
B = np.random.rand(2,3,4)

# reshape, transpose 
Bt = B.transpose(2,1,0)

# contraction

##### Ex.1.5(b): Contraction using ncon
d = 10
A = np.random.rand(d,d,d); B = np.random.rand(d,d,d,d)
C = np.random.rand(d,d,d); D = np.random.rand(d,d)

TensorArray = [A,B,C,D]
IndexArray = [[1,-2,2],[-1,1,3,4],[5,3,2],[4,5]]

#E = ncon(TensorArray,IndexArray,cont_order = [5,3,4,1,2])
E = ncon(TensorArray,IndexArray)


# Decomposition 

##### Ex2.2(a): SVD of matrix
def test_svd():
    d1 = 10; d2 = 6
    A = np.random.rand(d1,d2)
    [U,S,Vh] = LA.svd(A,full_matrices=False)
    # check result
    Af = U @ np.diag(S) @ Vh
    dA = LA.norm(Af-A) 


##### Ex2.2(b): SVD of tensor
def svd_decomposition():
    d = 10; A = np.random.rand(d,d,d)
    Am = A.reshape(d**2,d)
    Um,Sm,Vh = LA.svd(Am,full_matrices=False)
    U = Um.reshape(d,d,d); S = np.diag(Sm)
    # check result
    Af = ncon([U,S,Vh],[[-1,-2,1],[1,2],[2,-3]])
    dA = LA.norm(Af-A)
    print(dA) 

def random_unitary():
    """
    generate random unitary matrices via SVD
    """
    ##### Initialize unitaries and isometries
    d1 = 10; d2 = 6;
    
    # d1-by-d1 random unitary matrix U
    U,_,_ = LA.svd(np.random.rand(d1,d1))
    # d1-by-d2 random isometric matrix W
    A = np.random.rand(d1,d2);
    W,_,_ = LA.svd(A,full_matrices=False)
    print(W)

def qr_decomposition():
    ##### Ex2.2(g): QR decomp of tensor
    d = 10
    A = np.random.rand(d,d,d)
    Qm,R = LA.qr(A.reshape(d**2,d))
    Q = Qm.reshape(d,d,d)
    # check result
    Af = ncon([Q,R],[[-1,-2,1],[1,-3]]);
    dA = LA.norm(Af-A)
    print(dA)

def eigen_decomposition():
    ##### Ex2.2(d): spect. decomp. of tensor
    d = 2; A = np.random.rand(d,d,d,d)
    H = 0.5*(A + A.transpose(2,3,0,1))
    D,U = LA.eigh(H.reshape(d**2,d**2))
    U = U.reshape(d,d,d**2)
    # check result
    Hf = ncon([U,np.diag(D),U],
               [[-1,-2,1],[1,2],[-3,-4,2]])
    dH = LA.norm(Hf-H)
    print(dH)
    

def tensor_decomposition(chi):
    ##### SVD decomposition with restrited rank 
    d = 6; A = np.random.rand(d,d,d,d,d)
    Um,S,Vhm = LA.svd(A.reshape(d**3,d**2),full_matrices=False)
    U = Um.reshape(d,d,d,d**2)
    Vh = Vhm.reshape(d**2,d,d)
    ##### truncation
    Vhtilda = Vh[:chi,:,:]
    Stilda = np.diag(S[:chi])
    Utilda = U[:,:,:,:chi]
    B = ncon([Utilda,Stilda,Vhtilda],[[-1,-2,-3,1],[1,2],[2,-4,-5]])
    ##### compare
    epsAB = LA.norm(A-B) / LA.norm(A)
    return epsAB

#import matplotlib.pyplot as plt 
#
#fig, ax = plt.subplots()
#N = range(10, 40, 4)
#
#ax.plot(N, [tensor_decomposition(n) for n in N], 'd-')
#
#plt.show()

def effective_rank():
    ##### Ex2.4(d): effective rank
    # Generate toeplitz matrix
    d = 500;
    A = (np.diag(np.ones(d-1),-1) + 
         np.diag(np.ones(d-1), 1))
    A = A / LA.norm(A) #normalize
    
    # compute effective rank to accuracy 'deltaval'
    deltaval = 1e-2
    Um, Sm, Vhm = LA.svd(A)
    r_delta = sum(Sm > deltaval)
    eps_err = np.sqrt(sum(Sm[r_delta:]**2))
    print(eps_err) 
    
    # -*- coding: utf-8 -*-
# doTEBD.py
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs

def doTEBD(hamAB,hamBA,A,B,sAB,sBA,chi,tau,evotype="imag",numiter=1000,dispon=True,midsteps=10,midtol=1e-8,E0=0.0):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 21/1/2019
------------------------
Implementation of time evolution (real or imaginary) for MPS with 2-site unit \
cell (A-B), based on TEBD algorithm. Nearest neighbor input Hamiltonian is \
specified by 'hamAB, hamBA', while 'A' and 'B' are the MPS tensors, 'sAB' and \
'sBA' are the index weights (i.e. Schmidt coefficients). Maximum bond \
dimension specified by 'chi', and timestep by 'tau'.

Optional arguments:
`evotype="imag"`: real or imaginary time evolution ["real","imag"]
`dispon::Bool=true`: display convergence data
`midsteps::Int=10`: number of evolution steps between MPS re-orthogonalization
`midtol::Float=1e-8`: smallest singular values utilised in truncation step
`E0::Float=0.0`: specify exact ground energy (if known)
"""
    
    ##### Exponentiate Hamiltonian
    d = A.shape[1]
    if evotype == "real":
        G_AB = expm(1j*tau*hamAB.reshape(d**2,d**2)).reshape(d,d,d,d)
        G_BA = expm(1j*tau*hamBA.reshape(d**2,d**2)).reshape(d,d,d,d)
    elif evotype == "imag":
        G_AB = expm(-tau*hamAB.reshape(d**2,d**2)).reshape(d,d,d,d)
        G_BA = expm(-tau*hamBA.reshape(d**2,d**2)).reshape(d,d,d,d)
                
    ##### Edge matrices
    rhoLB = np.eye(A.shape[0])/A.shape[0]
    rhoRB = np.eye(A.shape[2])/A.shape[2]

    ##### Functions for contracting MPS from left / right
    def leftboundMPS(rhoB):
        chitemp = spA.shape[0]
        return ncon([rhoB.reshape(chitemp,chitemp),spA,np.conj(spA),spB,np.conj(spB)],
                     [[1,2],[1,3,4],[2,3,5],[4,6,-1],[5,6,-2]]).reshape(chitemp**2)
    
    def rightboundMPS(rhoB):
        chitemp = Aps.shape[2]
        return ncon([rhoB.reshape(chitemp,chitemp),Aps,np.conj(Aps),Bps,np.conj(Bps)],
                     [[1,3],[4,2,1],[6,2,3],[-1,5,4],[-2,5,6]]).reshape(chitemp**2)


    for k in range(numiter+1):
        ##### Bring MPS to normal form
        if np.mod(k,midsteps) == 0 or (k == numiter):
            
            ##### Contract from the infinite left (using Arpack)
            chitemp = A.shape[0]
            if rhoLB.shape[0] == chitemp:
                rho0 = rhoLB.reshape(np.prod(rhoLB.shape))
            else:
                rho0 = (np.eye(chitemp)/chitemp).reshape(chitemp**2)
            
            spA = ncon([np.diag(sBA),A],[[-1,1],[1,-2,-3]])
            spB = ncon([np.diag(sAB),B],[[-1,1],[1,-2,-3]])
            Dtemp, rhoLB = eigs(LinearOperator((chitemp**2,chitemp**2),matvec=leftboundMPS),k=1,which='LM',v0=rho0)
            
            ##### Normalize the l.h.s. density matrices
            if not isinstance(A[0,0,0],complex):
                rhoLB = np.real(rhoLB)
            rhoLB = rhoLB.reshape(chitemp,chitemp)
            rhoLB = 0.5*(rhoLB + np.conj(rhoLB.T))
            rhoLB = rhoLB/np.trace(rhoLB)
            rhoLA = ncon([rhoLB,np.diag(sBA),np.diag(sBA),A,np.conj(A)],[[1,2],[1,3],[2,4],[3,5,-1],[4,5,-2]])
            rhoLA = rhoLA/np.trace(rhoLA)
            
            ##### Contract from the infinite right (using Arpack)
            chitemp = A.shape[2]
            if rhoRB.shape[0] == chitemp:
                rho0 = rhoRB.reshape(np.prod(rhoRB.shape))
            else:
                rho0 = (np.eye(chitemp)/chitemp).reshape(chitemp**2)
            
            Aps = ncon([A,np.diag(sAB)],[[-1,-2,1],[1,-3]])
            Bps = ncon([B,np.diag(sBA)],[[-1,-2,1],[1,-3]])
            Dtemp, rhoRB = eigs(LinearOperator((chitemp**2,chitemp**2),matvec=rightboundMPS),k=1,which='LM',v0=rho0)
            
            ##### Normalize the r.h.s. density matrices
            if not isinstance(A[0,0,0],complex):
                rhoRB = np.real(rhoRB)
            rhoRB = rhoRB.reshape(chitemp,chitemp)
            rhoRB = 0.5*(rhoRB + np.conj(rhoRB.T))
            rhoRB = rhoRB/np.trace(rhoRB)
            rhoRA = ncon([rhoRB,np.diag(sAB),np.diag(sAB),A,np.conj(A)],[[1,2],[3,1],[5,2],[-1,4,3],[-2,4,5]])
            rhoRA = rhoRA/np.trace(rhoRA)
            
            ##### Orthogonalise
            A, B, sAB, sBA = orthogMPS(rhoLA,rhoLB,rhoRA,rhoRB,A,B,sAB,sBA, dtol = 1e-14)
    
            ##### Compute energy and display
            if dispon:
                rholocAB = ncon([np.diag(sBA**2),A,np.conj(A),np.diag(sAB),np.diag(sAB),B,np.conj(B),np.diag(sBA**2)],
                                 [[3,4],[3,-3,1],[4,-1,2],[1,7],[2,8],[7,-4,5],[8,-2,6],[5,6]]).reshape(d**2,d**2)
                
                rholocBA = ncon([np.diag(sAB**2),B,np.conj(B),np.diag(sBA),np.diag(sBA),A,np.conj(A),np.diag(sAB**2)],
                                 [[3,4],[3,-3,1],[4,-1,2],[1,7],[2,8],[7,-4,5],[8,-2,6],[5,6]]).reshape(d**2,d**2)
                                
                EnergyAB = np.real(np.trace(hamAB.reshape(d**2,d**2)@np.conj(rholocAB)))
                EnergyBA = np.real(np.trace(hamBA.reshape(d**2,d**2)@np.conj(rholocBA)))
                Energy = 0.5*(EnergyAB + EnergyBA)
                
                stemp = sAB[range(sum(sAB>midtol))]**2
                Entropy = -sum(stemp*np.log2(stemp))
                
                print('Iteration: %d of %d, Bond dim: %d, Timestep: %f, Energy: %f, Energy Error: %e, Entropy: %f, Cut-err %f'
                      % (k,numiter,min(A.shape[0],B.shape[0]),tau,Energy,Energy-E0,Entropy ,max(min(sAB),min(sBA))))
                                
        if k < numiter:
            ##### Implement A-B update
            sBAtr = sBA*(sBA > midtol) + midtol*(sBA < midtol)
            ut,st,vht = LA.svd(ncon([np.diag(sBAtr),A,np.diag(sAB),B,np.diag(sBAtr),G_AB],[[-1,1],[1,5,2],[2,4],
                                     [4,6,3],[3,-4],[-2,-3,5,6]]).reshape(d*sBAtr.shape[0],d*sBAtr.shape[0]),full_matrices=False);
            chitemp = min(chi,len(st))
            A = (np.diag(1/sBAtr) @ ut[:,range(chitemp)].reshape(sBAtr.shape[0],d*chitemp)).reshape(sBAtr.shape[0],d,chitemp)
            B = (vht[range(chitemp),:].reshape(chitemp*d,sBAtr.shape[0]) @ np.diag(1/sBAtr)).reshape(chitemp,d,sBAtr.shape[0])
            sAB = st[range(chitemp)]/LA.norm(st[range(chitemp)])
            
            ##### Implement B-A update
            sABtr = sAB*(sAB > midtol) + midtol*(sAB < midtol)
            ut,st,vht = LA.svd(ncon([np.diag(sABtr),B,np.diag(sBA),A,np.diag(sABtr),G_BA],[[-1,1],[1,5,2],[2,4],
                                     [4,6,3],[3,-4],[-2,-3,5,6]]).reshape(d*sABtr.shape[0],d*sABtr.shape[0]))
            chitemp = min(chi,len(st))
            B = (np.diag(1/sABtr) @ ut[:,range(chitemp)].reshape(sABtr.shape[0],d*chitemp)).reshape(sABtr.shape[0],d,chitemp)
            A = (vht[range(chitemp),:].reshape(chitemp*d,sABtr.shape[0]) @ np.diag(1/sABtr)).reshape(chitemp,d,sABtr.shape[0])
            sBA = st[range(chitemp)]/LA.norm(st[range(chitemp)])
        
        
    rholocAB = ncon([np.diag(sBA**2),A,np.conj(A),np.diag(sAB),np.diag(sAB),B,np.conj(B),np.diag(sBA**2)],
        [[3,4],[3,-3,1],[4,-1,2],[1,7],[2,8],[7,-4,5],[8,-2,6],[5,6]]).reshape(d**2,d**2)
    rholocBA = ncon([np.diag(sAB**2),B,np.conj(B),np.diag(sBA),np.diag(sBA),A,np.conj(A),np.diag(sAB**2)],
        [[3,4],[3,-3,1],[4,-1,2],[1,7],[2,8],[7,-4,5],[8,-2,6],[5,6]]).reshape(d**2,d**2)

    return A, B, sAB, sBA, rholocAB, rholocBA

#----------------------------------------------------------------------------------------
def eigCut(rho, dtol = 1e-10):
    """ eigCut: truncated eigendecomposition for Hermitian, positive \
    semi-definite matrices. Keeps only eigenvalues greater than 'dtol' and \
    sorts eigenvalues in descending order """
    
    dtemp,utemp = LA.eigh(rho)
    chitemp = sum(dtemp>dtol)
    
    return dtemp[range(-1,-chitemp-1,-1)], utemp[:,range(-1,-chitemp-1,-1)]

#----------------------------------------------------------------------------------------
def orthogMPS(rhoLA,rhoLB,rhoRA,rhoRB,A,B,sAB,sBA, dtol = 1e-14):
    """ orthogMPS: bring a matrix product state to normal form """

    DLB, ULB = eigCut(rhoLB, dtol = dtol)
    DRA, URA = eigCut(rhoRA, dtol = dtol)
    uBA,sBA,vhBA = LA.svd(np.diag(np.sqrt(DLB))@ULB.T@np.diag(sBA)@URA@np.diag(np.sqrt(DRA)),full_matrices=False)
    sBA = sBA/LA.norm(sBA)
    
    DLA, ULA = eigCut(rhoLA, dtol = dtol)
    DRB, URB = eigCut(rhoRB, dtol = dtol)
    uAB,sAB,vhAB = LA.svd(np.diag(np.sqrt(DLA))@ULA.T@np.diag(sAB)@URB@np.diag(np.sqrt(DRB)),full_matrices=False)
    sAB = sAB/LA.norm(sAB)
    
    xL = np.conj(ULB) @ np.diag(1/np.sqrt(DLB)) @ uBA
    xR = np.conj(URA) @ np.diag(1/np.sqrt(DRA)) @ vhBA.T
    yL = np.conj(ULA) @ np.diag(1/np.sqrt(DLA)) @ uAB
    yR = np.conj(URB) @ np.diag(1/np.sqrt(DRB)) @ vhAB.T 
    A = ncon([xR,A,yL],[[1,-1],[1,-2,2],[2,-3]])
    B = ncon([yR,B,xL],[[1,-1],[1,-2,2],[2,-3]])
    
    A = A/np.sqrt(ncon([np.diag(sBA**2),A,np.conj(A),np.diag(sAB**2)],[[1,3],[1,4,2],[3,4,5],[2,5]]))
    B = B/np.sqrt(ncon([np.diag(sAB**2),B,np.conj(B),np.diag(sBA**2)],[[1,3],[1,4,2],[3,4,5],[2,5]]))

    return A, B, sAB, sBA

# -*- coding: utf-8 -*-
"""
mainTEBD.py
---------------------------------------------------------------------
Script file for initializing the Hamiltonian and MPS tensors before passing \
to the TEBD routine.

    by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
"""


##### Example 1: XX model #############
#######################################

##### Set bond dimensions and simulation options
chi = 16 # bond dimension
tau = 0.1 # timestep

OPTS_numiter = 500 # number of timesteps
OPTS_dispon = True # print simulation data
OPTS_evotype = "imag" # real or imaginary time evolution
OPTS_midtol = 1e-6 # smallest singular values utilised in truncation step
OPTS_E0 = -4/np.pi # specify exact ground energy (if known)
OPTS_midsteps = int(1/tau); # timesteps between MPS re-orthogonalization

#### Define Hamiltonian (quantum XX model)
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0,-1]])
hamAB = (np.real(np.kron(sX,sX) + np.kron(sY,sY))).reshape(2,2,2,2)
hamBA = (np.real(np.kron(sX,sX) + np.kron(sY,sY))).reshape(2,2,2,2)

##### Initialize tensors
d = hamAB.shape[0]
sAB = np.ones(chi)/np.sqrt(chi);
sBA = np.ones(chi)/np.sqrt(chi);
A = np.random.rand(chi,d,chi);
B = np.random.rand(chi,d,chi);

##### Imaginary time evolution with TEBD ######
###############################################

##### Run TEBD routine
A, B, sAB, sBA, rholocAB, rholocBA = doTEBD(hamAB,hamBA,A,B,sAB,sBA,chi,tau, evotype = OPTS_evotype,
    numiter = OPTS_numiter, dispon = OPTS_dispon, midsteps = OPTS_midsteps, midtol = OPTS_midtol,E0 = OPTS_E0);

##### continute running TEBD routine with reduced timestep
tau = 0.01;
OPTS_numiter = 2000;
OPTS_midsteps = 100;
A, B, sAB, sBA, rholocAB, rholocBA = doTEBD(hamAB,hamBA,A,B,sAB,sBA,chi,tau, evotype = OPTS_evotype,
    numiter = OPTS_numiter, dispon = OPTS_dispon, midsteps = OPTS_midsteps, midtol = OPTS_midtol, E0 = OPTS_E0);

##### continute running TEBD routine with reduced timestep and increased bond dim
#chi = 32;
#tau = 0.001;
#OPTS_numiter = 16000;
#OPTS_midsteps = 1000;
#A, B, sAB, sBA, rholocAB, rholocBA = doTEBD(hamAB,hamBA,A,B,sAB,sBA,chi,tau, evotype = OPTS_evotype,
#    numiter = OPTS_numiter, dispon = OPTS_dispon, midsteps = OPTS_midsteps, midtol = OPTS_midtol, E0 = OPTS_E0);

##### Compare with exact results
EnergyAB = np.real(np.trace(hamAB.reshape(4,4) @ np.conj(rholocAB)))
EnergyBA = np.real(np.trace(hamBA.reshape(4,4) @ np.conj(rholocBA)))
EnergyMPS = 0.5*(EnergyAB + EnergyBA)
EnErr = abs(EnergyMPS + 4/np.pi)
print('Final results => Bond dim: %d, Energy: %f, Energy Error: %e' % (chi, EnergyMPS, EnErr))

