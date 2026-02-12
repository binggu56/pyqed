#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 23:03:15 2025

@author: bingg
"""

import numpy as np
from opt_einsum import contract
import time
from scipy import linalg
from typing import Tuple, Union, Optional
import scipy.linalg as la

def minimize(f, X0, args=(), tau=2, taum=1e-15, tauM=1e15, eta=0.85, 
             rho1=0.5, delta=0.2, epsilon=1e-5, max_steps=50, dtype=np.float64):
    """
    Implicit Steepest Descent Method for Optimization
    with Orthogonality Constraints (Implicit–SD)

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    X0 : TYPE
        DESCRIPTION.
    tau : TYPE, optional
        DESCRIPTION. The default is 1.
    eta : TYPE, optional
        DESCRIPTION. The default is 0.5.
    delta : TYPE, optional
        DESCRIPTION. The default is 0.5.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 1e-5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    References
    ----------

    Optimization Lett. 2022, 16:1773

    """


    n, p = X0.shape

    # mean = 0
    # std = 0.1
    # X0 = X0 + np.random.normal(loc=mean, scale=std, size=(n, p))

    Q0 = 1
    k = 0
    C = f(X0, *args)
    Id = np.identity(n, dtype=dtype)


    X = X0.astype(dtype)


    # taum = 1
    # tauM = 2

    Q = Q0
    G = gradient(X0, *args)
    # print('gradient', G)

    df = grad(X, G)

    while norm(df) > epsilon:
        # print('orb opt')
        A = G @ X.T.conj() - X @ G.T.conj()
        Y = project(np.linalg.inv(Id + tau * A) @ X)

        while f(Y, *args) > C + rho1 * tau * (-1/2 * norm(A)**2):
            
            tau = tau * delta
            Y = project(np.linalg.inv(Id + tau * A) @ X)

        Xnew = Y
        Qnew = eta * Q + 1

        v = f(Xnew, *args)
        # print('energy = ', v)

        Cnew = (eta * Q * C + v)/Qnew
        Gnew = gradient(Xnew, *args)

        df_new = grad(Xnew, Gnew)

        tau = stepsize(k+1, Xnew-X, df_new-df)
        tau = max(min(tau, tauM), taum)

        k += 1

        # update
        X = Xnew
        Q = Qnew
        C = Cnew
        G = Gnew
        df = df_new

    # print('overlap between orbital', np.dot(X0.T, X))

    return X, v



    

def norm(A):
    """
    Frobenius norm of matrix
    .. math::

        ||A||_F = \sqrt{ A^\dagger A }
    """
    return np.sqrt(np.trace(A.T.conj() @ A))

def grad(X, G=None):
    """
    Riemmann gradient

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if G is None:
        G = gradient(X)
    return G - X @ G.T.conj() @ X

def project(V):
    """
    projection to Siefel manifold by orthonormalization

    .. math::

        orth(V) = VQ\Lambda^{-1/2}

    where Q and \Lambda are eigenvectors and eigenvalues of :math:`V^T V`.

    Refs
    ----
    JCTC 2020, 12, 6207

    Parameters
    ----------
    V : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    w, Q = np.linalg.eigh(V.T.conj() @ V)
    
    return V @ Q @ np.diag(1/np.sqrt(w)) @ Q.T.conj()

# def orth(V):
#     """
#     projection to Siefel manifold by orthonormalization

#     .. math::

#         orth(V) = VQ\Lambda^{-1/2}

#     where Q and \Lambda are eigenvectors and eigenvalues of :math:`V^T V`.

#     Refs
#     ----
#     JCTC 2020, 12, 6207

#     Parameters
#     ----------
#     V : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     w, Q = np.linalg.eigh(V.T @ V)
#     return V @ Q @ np.diag(1/np.sqrt(w))


def orth(V):
    """
    Alternative projection to complex Stiefel manifold
    
    Parameters
    ----------
    V : array_like
        Complex matrix
    
    Returns
    -------
    array
        Orthonormalized matrix
    """
    # QR decomposition preserves complex structure
    Q, R = np.linalg.qr(V, mode='reduced')
    return Q


def stepsize(k, dU, dG):
    """
    Barzilai−Borwein stepsize for update U matrix

    Parameters
    ----------
    k : TYPE
        DESCRIPTION.
    U : TYPE
        DESCRIPTION.
    Uprev : TYPE
        DESCRIPTION.
    G : TYPE
        DESCRIPTION.
    Gprev : TYPE
        DESCRIPTION.

    Returns
    -------
    tau : TYPE
        DESCRIPTION.

    """
    if k % 2 == 0:
        # even
        # dU = U - Uprev
        # dG = G - Gprev
        tau = abs(inner(dU, dG))/inner(dG, dG)

    else:
        # odd
        # dU = U - Uprev
        # dG = G - Gprev
        # print('abs(inner(dU, dG))',abs(inner(dU, dG)))
        tau = inner(dU, dU)/abs(inner(dU, dG))

    return tau

def inner(a, b):
    return np.trace(a.T.conj() @ b)


def gradient(U, h1e, h2e, dm1, dm2):

    """gradient of energy(U, h1e, eri, dm1, dm2) func.
        U is complex

    Returns
    -------
    _type_
        _description_
    """    
    # print('gradient U type', U.dtype)
    g = contract('pq, qb, ab -> pa', h1e, U, dm1)
    g += 0.5 * (contract('pqrs, qb, rc, sd, abcd -> pa', h2e, U.conj(), U, U, dm2) + \
        contract('pqrs, pa, rc, sd, abcd -> qb', h2e, U.conj(), U, U, dm2) )

    return g

# def gradient(U, h1e, h2e, dm1, dm2):

#     # start_time = time.time()

#     # g = h1e @ U @ dm1.T + h1e.T @ U @ dm1  # these two terms are probably the same
#     g = contract('pq, qb, ab -> pa', h1e, U, dm1) + contract('pq, pa, ab -> qb', h1e, U, dm1)
#     g += 0.5 * (contract('pqrs, qb, rc, sd, abcd -> pa', h2e, U, U, U, dm2) + \
#         contract('pqrs, pa, rc, sd, abcd -> qb', h2e, U, U, U, dm2) + \
#         contract('pqrs, pa, qb, sd, abcd -> rc', h2e, U, U, U, dm2) + \
#         contract('pqrs, pa, qb, rc, abcd -> sd', h2e, U, U, U, dm2) )
    
#     # end_time = time.time()
#     # execution_time = end_time - start_time
    
#     # print(f"gradient contract time : {execution_time:.6f} seconds")
#     return g



# def hessian(U, h1e, h2e, dm1, dm2):

#     n, p = U.shape
#     h = 2 * contract('pq, ab -> paqb', h1e, dm1) 
#     h += 0.5 * (contract('pqrs, rc, sd, abcd -> paqb', h2e, U, U, dm2) + \
#         contract('pqrs, qb, sd, abcd -> parc', h2e, U, U, dm2) + \
#         contract('pqrs, qb, rc, abcd -> pasd', h2e, U, U, dm2) + \
#         contract('pqrs, rc, sd, abcd -> qbpa', h2e, U, U, dm2) + \
#         contract('pqrs, pa, sd, abcd -> qbrc', h2e, U, U, dm2) + \
#         contract('pqrs, pa, rc, abcd -> qbsd', h2e, U, U, dm2) + \
#         contract('pqrs, qb, sd, abcd -> rcpa', h2e, U, U, dm2) + \
#         contract('pqrs, pa, sd, abcd -> rcqb', h2e, U, U, dm2) + \
#         contract('pqrs, pa, qb, abcd -> rcsd', h2e, U, U, dm2) + \
#         contract('pqrs, qb, rc, abcd -> sdpa', h2e, U, U, dm2) + \
#         contract('pqrs, pa, rc, abcd -> sdpb', h2e, U, U, dm2) + \
#         contract('pqrs, pa, qb, abcd -> sdrc', h2e, U, U, dm2))
#     h = h.reshape((n*p, n*p))

#     return h

def hessian(U, h1e, h2e, dm1, dm2):

    n, p = U.shape
    h = contract('pq, ab -> paqb', h1e, dm1) + contract('pq, ab -> qbpa', h1e, dm1)
    h += 0.5 * (contract('pqrs, rc, sd, abcd -> paqb', h2e, U, U, dm2) + \
        contract('pqrs, qb, sd, abcd -> parc', h2e, U, U, dm2) + \
        contract('pqrs, qb, rc, abcd -> pasd', h2e, U, U, dm2) + \
        contract('pqrs, rc, sd, abcd -> qbpa', h2e, U, U, dm2) + \
        contract('pqrs, pa, sd, abcd -> qbrc', h2e, U, U, dm2) + \
        contract('pqrs, pa, rc, abcd -> qbsd', h2e, U, U, dm2) + \
        contract('pqrs, qb, sd, abcd -> rcpa', h2e, U, U, dm2) + \
        contract('pqrs, pa, sd, abcd -> rcqb', h2e, U, U, dm2) + \
        contract('pqrs, pa, qb, abcd -> rcsd', h2e, U, U, dm2) + \
        contract('pqrs, qb, rc, abcd -> sdpa', h2e, U, U, dm2) + \
        contract('pqrs, pa, rc, abcd -> sdpb', h2e, U, U, dm2) + \
        contract('pqrs, pa, qb, abcd -> sdrc', h2e, U, U, dm2))
    h = h.reshape((n*p, n*p))
    return h


# def hessian(U, h1e, h2e, dm1, dm2):
#     n, p = U.shape

#     h = 2 * contract('pq, ab -> paqb', h1e, dm1)

#     h2e_UU = contract('pqrs, rc, sd -> pqcd', h2e, U, U)
#     base_term = contract('pqcd, abcd -> paqb', h2e_UU, dm2)  # shape: (n, p, n, p)

#     terms = []

#     terms.append(base_term)  # paqb

#     terms.append(base_term.transpose(2, 3, 0, 1))  # qbpa - shape: (n, p, n, p)

#     h2e_Ur_sd = contract('pqrs, rc, sd -> pqcd', h2e, U, U)  # [pqcd]
#     h2e_Uq_sd = contract('pqrs, qb, sd -> prbd', h2e, U, U)  # [prbd]
#     h2e_Uq_rc = contract('pqrs, qb, rc -> psbc', h2e, U, U)  # [psbc]

#     term1 = contract('pqcd, abcd -> paqb', h2e_Ur_sd, dm2)
#     term2 = contract('prbd, abcd -> parc', h2e_Uq_sd, dm2)
#     term3 = contract('psbc, abcd -> pasd', h2e_Uq_rc, dm2)
#     term4 = term1.transpose(2, 3, 0, 1)
#     h2e_Up_sd = contract('pqrs, pa, sd -> qrsa', h2e, U, U)  # [qrsa]
#     term5 = contract('qrsa, abcd -> qbrc', h2e_Up_sd, dm2)
#     h2e_Up_rc = contract('pqrs, pa, rc -> qrsa', h2e, U, U)  # [qrsa]
#     term6 = contract('qrsa, abcd -> qbsd', h2e_Up_rc, dm2)
#     term7 = term2.transpose(2, 3, 0, 1)
#     term8 = term5.transpose(2, 3, 0, 1)
#     h2e_Up_Uq = contract('pqrs, pa, qb -> rsab', h2e, U, U)  # [rsab]
#     term9 = contract('rsab, abcd -> rcsd', h2e_Up_Uq, dm2)
#     term10 = term3.transpose(2, 3, 0, 1)
#     term11 = term6.transpose(2, 3, 0, 1)
#     term12 = term9.transpose(2, 3, 0, 1)
    
#     all_terms = [term1, term2, term3, term4, term5, term6,
#                  term7, term8, term9, term10, term11, term12]
    

#     for i, term in enumerate(all_terms):
#         if term.shape != (n, p, n, p):
#             print(f"Warning: term {i+1} has shape {term.shape}, expected ({n}, {p}, {n}, {p})")
#             all_terms[i] = term.transpose(0, 1, 2, 3)
    
#     h += 0.5 * sum(all_terms)
    # h = h.reshape((n*p, n*p))
    
    return h

class Newton_opt:
    def __init__(self, U, h1e, eri, dm1, dm2):

        self.U = U
        self.n, self.p = U.shape
        
        self.h1e = h1e
        self.eri = eri
        self.dm1 = dm1
        self.dm2 = dm2

        self.grad = None
        self.hess = None

        # print('init U is unitary', U.T @ U)
        

    def get_gradient(self):

        grad = gradient(self.U, self.h1e, self.eri, self.dm1, self.dm2)
        grad = grad.reshape((self.n*self.p,1))
        self.grad = grad
        return grad

    def get_hessian(self):

        hess = hessian(self.U, self.h1e, self.eri, self.dm1, self.dm2)
        hess = hess.reshape((self.n*self.p, self.n*self.p))
        self.hess = hess
        return hess

    # def make_KKT_matrix(self):

    #     # shape test
    #     U1 = self.U
    #     U1 = U1.reshape((self.n*self.p,1))

    #     zero = np.zeros((1,1))

    #     top_row = np.hstack((self.hess, U1))
    #     bottom_row = np.hstack((U1.T, zero))
    #     KKT = np.vstack((top_row, bottom_row))

    #     vec = - np.vstack((self.grad, zero))


    #     x = linalg.solve(KKT, vec)
    #     delta_U = x[0:self.n*self.p,:].reshape((self.n,self.p))
    #     U_new = self.U + delta_U
    #     self.U = U_new
    #     print('after KKT U is unitary', self.U.T @ self.U)

    #     return self.U

    def build_constraint_matrices_correct(U):

        n, p = U.shape
        m = p * (p + 1) // 2
        n_vars = n * p
        
        C = np.zeros((m, n_vars))
        B = np.zeros((n_vars, m))
        
        # 1. 首先构建C矩阵
        constraint_idx = 0
        for i in range(p):          # i 是较大的索引
            for j in range(i + 1):  # j <= i
                # C矩阵第constraint_idx行
                if i == j:
                    # 对角线约束
                    for k in range(n):
                        C[constraint_idx, i*n + k] = 2.0 * U[k, i]
                else:
                    # 非对角线约束
                    for k in range(n):
                        C[constraint_idx, i*n + k] = U[k, j]   # ΔU_{k,i} 项
                        C[constraint_idx, j*n + k] = U[k, i]   # ΔU_{k,j} 项
                
                # 2. 构建B矩阵（同一循环中）
                for k in range(n):
                    if i == j:
                        B[i*n + k, constraint_idx] = U[k, i]
                    else:
                        B[i*n + k, constraint_idx] = U[k, j]   # ΔU_{k,i} 项
                        B[j*n + k, constraint_idx] = U[k, i]   # ΔU_{k,j} 项
                
                constraint_idx += 1
        
        return B, C

    # def build_constraint_matrices(U):

    #     n, p = U.shape
    #     m = p * (p + 1) // 2

    #     n_vars = n * p

    #     C = np.zeros((m, n_vars))
    #     B = np.zeros((n_vars, m)) 

    #     for j in range(p):
    #         for i in range(j,p):
    #             if i == j:
    #                 C[(2*p-j+1)*j//2, j*n:(j+1)*n] = [2 * U[k,i] for k in range(n)]
    #             else:
    #                 C[(2*p-j+1)*j//2+i, j*n:(j+1)*n] = [U[k,i] for k in range(n)]
    #                 C[(2*p-j+1)*j//2+i, i*n:(i+1)*n] = [U[k,j] for k in range(n)]

    #     for i in range(n):
    #         for j in range(p):
    #             for k in range(p):
    #                 if k >= j:
    #                     B[n*j+i,(2*p-j+1)*j//2:(2*p-j+1)*j//2+p-j] = [U[i,l] for l in range(j,p)]
    #                 #     print('{},{}:{}={}'.format(n*j+i,(2*p-j+1)*j//2,(2*p-j+1)*j//2+p-j,[U[i,l] for l in range(j,p)] ))
    #                 else:
    #                     a = U[i, k]
    #                     B[n*j+i,(2*p-k-1)*k//2+j-k] = a

    #     return B, C

    def make_KKT_matrix(self):

        # shape test
        U1 = self.U
        U1 = U1.reshape((self.n*self.p,1))

        B, C = Newton_opt.build_constraint_matrices_correct(self.U)



        zero = np.zeros((self.p*(self.p+1)//2, self.p*(self.p+1)//2))

        top_row = np.hstack((self.hess, B))
        bottom_row = np.hstack((C, zero))
        KKT = np.vstack((top_row, bottom_row))

        vec = - np.vstack((self.grad, np.zeros((self.p*(self.p+1)//2,1))))


        x = linalg.solve(KKT, vec)
        delta_U = x[0:self.n*self.p,:].reshape((self.n,self.p))
        # print('AAAAA',delta_U.T @ delta_U)
        U_new = self.U + delta_U
        self.U, _ = la.qr(U_new, mode='economic')
        # print('after KKT U is unitary', self.U.T @ self.U)

        return self.U

def opt(f, U, h1e, eri, dm1, dm2, epsilon = 1e-6):

    old_ener = f(U, h1e, eri, dm1, dm2)
    print('orb_opt_ener', old_ener)

    newton = Newton_opt(U, h1e, eri, dm1, dm2)
    newton.get_gradient()
    newton.get_hessian()
    newton.make_KKT_matrix()
    U = newton.U

    new_ener = f(U, h1e, eri, dm1, dm2)
    print('orb_opt_ener', new_ener)

    k = 0

    while abs(new_ener - old_ener) > epsilon:
        print(' ------ opt begin ------')

        newton = Newton_opt(U, h1e, eri, dm1, dm2)
        newton.get_gradient()
        newton.get_hessian()
        newton.make_KKT_matrix()
        U = newton.U

        old_ener = new_ener
        new_ener = f(U, h1e, eri, dm1, dm2)
        k += 1
        print('orb_opt_ener', new_ener)
    E = new_ener
    # print(' ------ opt end ------')

    return U, E


def energy(U, h1e, eri, dm1, dm2):
    """
    electronic energy

    Parameters
    ----------
    U : ndarray of (n, p < n/2)
        transformation matrix
    h1e : TYPE
        core Hamiltonian in canonical MO
    eri : TYPE
        DESCRIPTION.
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    e : TYPE
        DESCRIPTION.

    """

    e = contract('pq, pa, qb, ab ->', h1e, U, U, dm1)
    e += 0.5 * (contract('pqrs, pa, qb, rc, sd, abcd ->', eri, U, U, U, U, dm2))
    return e





def kernel(mf, U0, max_steps=50, tol=1e-6):
    """
    complete active space orbital optimization with orthonomality constraint

    .. math::
        U^\top U = I_N

        E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
        1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

    where U is a M x N (M > N) matrix.

    .. math::
        U_{k+1} = orth(U_k - \tau_k G_k)

    where G_k = \nabla P(U_k) is the gradient.

    Parameters
    ----------
    h1e : TYPE
        DESCRIPTION.
    h2e : TYPE
        DESCRIPTION.
    U0: ndarray
        initial guess of orbitals
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    k = 0
    U = U0 # initial guess for U0

    # first FCI calculation
    mc = CASCI(mf, ncas=2, nelecas=2)
    mc.run(U)
    e_old = mc.e_tot


    while k < max_steps:

        # update CI coeff
        mc.run(U)

        if abs(mc.e_tot - e_old) < tol:
            print("E(CASSCF) = {}".format(mc.e_tot))
            break

        dm1, dm2 = mc.make_rdm12(0)
        h1e = mc.hcore
        eri = mc.eri_so[0, 0] # for spin-restricted calculation

        # update the MOs by updating U
        U, E = minimize(energy, U, args=(h1e, eri, dm1, dm2))

        k += 1

    return mc


class UnitaryNewtonSolver:
    """
    Generic Newton-Raphson solver for minimizing f(U) subject to U^T U = I.
    """
    def __init__(self, f, U, h1e, eri, dm1, dm2, max_iter=15, tol=1e-7):

        self.max_iter = max_iter
        self.tol = tol
        self.U = U
        self.n, self.p = U.shape

        self.f = f
        
        self.h1e = h1e
        self.eri = eri
        self.dm1 = dm1
        self.dm2 = dm2

        self.grad = None
        self.hess = None

    def get_gradient(self):

        grad = gradient(self.U, self.h1e, self.eri, self.dm1, self.dm2)
        # grad = grad.reshape((self.n*self.p,1))
        self.grad = grad
        return grad

    def get_hessian(self):

        hess = hessian(self.U, self.h1e, self.eri, self.dm1, self.dm2)
        hess = hess.reshape((self.n*self.p, self.n*self.p))
        self.hess = hess
        return hess


    def solve(self):
        """
        Args:
            U_init: Initial guess (n x p).
            func_E: Callable returning float (Energy).
            func_Grad: Callable returning (n x p) array (Gradient).
            func_Hess: Callable returning (dim x dim) array (Energy Hessian).
        """

        U = self.U.copy()
        n, p = U.shape
        dim = n * p
        
        # print(f"--- Starting Newton Engine (n={n}, p={p}) ---")
        # print(f"Initial Value: {func_E(U):.8f}\n")
        
        # Initialize Multipliers (Lambda)
        Lambda = np.zeros((p, p))
        
        for step in range(1, self.max_iter + 1):
            # 1. Get Physics Data (Generic calls)
            E_curr = self.f(self.U, self.h1e, self.eri, self.dm1, self.dm2)
            Grad = self.grad
            H_Energy = self.hess
            
            # 2. Update Lagrange Multipliers (Projection)
            # Lambda ~ U.T @ Grad (Symmetrized)
            L_proxy = U.T @ Grad
            Lambda = 0.5 * (L_proxy + L_proxy.T)
            
            # 3. Build Lagrangian Hessian
            # H_Lag = H_Energy - H_Constraint
            # H_Constraint = 2 * Lambda (x) I_n (Manifold curvature)
            H_Constraint = 2.0 * np.kron(Lambda, np.eye(n))
            H_Total = H_Energy - H_Constraint
            
            # 4. Build KKT Constraints (Matrix B)
            # Enforces: U.T * dU + dU.T * U = 0
            num_cons = p * (p + 1) // 2
            B = np.zeros((num_cons, dim))
            idx = 0
            for j in range(p):
                for i in range(j + 1):
                    # Place U_j at block i, U_i at block j
                    B[idx, i*n : (i+1)*n] += U[:, j]
                    B[idx, j*n : (j+1)*n] += U[:, i]
                    idx += 1
            
            # 5. Assemble Full KKT System
            # [ H   B.T ] [ dU ] = [ -g ]
            # [ B    0  ] [ dL ]   [  0 ]
            zeros_block = np.zeros((num_cons, num_cons))
            top = np.hstack([H_Total, B.T])
            bot = np.hstack([B, zeros_block])
            KKT = np.vstack([top, bot])
            
            rhs = np.concatenate([-Grad.flatten(), np.zeros(num_cons)])
            
            # 6. Solve Linear System
            try:
                sol = la.solve(KKT, rhs)
                delta_U = sol[:dim].reshape(n, p)
            except la.LinAlgError:
                print("  [!] Singular Matrix. Adding Regularization.")
                KKT[0:dim, 0:dim] += 1e-3 * np.eye(dim)
                sol = la.solve(KKT, rhs)
                delta_U = sol[:dim].reshape(n, p)
            
            # 7. Line Search & Retraction
            step_norm = la.norm(delta_U)
            alpha = 1.0
            

            U_trial_raw = U + alpha * delta_U
            U_trial, _ = la.qr(U_trial_raw, mode='economic')
            
            # Check Descent
            E_trial = self.f(self.U, self.h1e, self.eri, self.dm1, self.dm2)
            


            for _ in range(5):
                # Retract: U_new = QR(U + alpha*dU)
                U_trial_raw = U + alpha * delta_U
                U_trial, _ = la.qr(U_trial_raw, mode='economic')
                
                # Check Descent
                E_trial = self.f(self.U, self.h1e, self.eri, self.dm1, self.dm2)
                # print('orb_opt_ener', E_trial)
                if E_trial < E_curr + 1e-9:
                    U = U_trial
                    E_curr = E_trial
                    break
                alpha *= 0.5
            print('orb_opt_energy', E_curr)
                
            # print(f"Step {step}: Value = {E_curr:.8f} | |Step| = {step_norm:.6e} | Alpha = {alpha}")
            
            if step_norm < self.tol:
                print("--> Converged!")
                break
                
        return U, E_curr


if __name__=='__main__':


    from pyqed import Molecule
    from pyqed.qchem.mcscf.casci import CASCI

    mol = Molecule(atom='Li 0 0 0; H 0 0 1.4', unit='b', basis='sto3g')
    mol.build()

    mf = mol.RHF().run()
    C = mf.mo_coeff

    ncas=2
    nelecas = 2
    mc = CASCI(mf, ncas=ncas, nelecas=nelecas)
    nstates = 1
    mc.run(nstates)

    weights = np.ones(nstates)/nstates
    dm1 = 0
    dm2 = 0
    for n in range(nstates):
        _dm1, _dm2 = mc.make_rdm12(n)
        dm1 += _dm1 * weights[n]
        dm2 += _dm2 * weights[n]
        
            
    

    h1e = mf.get_hcore_mo()
    eri = mf.get_eri_mo()

    # h1e_ = mc.hcore
    # print(h1e_[0])
    # print(h1e.shape)

    # eri = mc.eri_so[0, 0] # for spin-restricted calculation
    nmo = mol.nao
    # print('# MO = ', nmo)

    U0 = np.zeros((nmo, ncas))
    for i in range(ncas):
        U0[i, i] = 1

    # print('E= ',energy(U0, h1e, eri, dm1, dm2))

    # U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))
    U, E = opt(energy, U0, h1e, eri, dm1, dm2)


    # # test newton opt
    # print('*'*100)
    # opt = Newton_opt(U0, h1e, eri, dm1, dm2)
    # opt.get_gradient()
    # opt.get_hessian()
    # opt.make_KKT_matrix()

    k = 0
    max_cycles = 3
    e_old = sum(weights * mc.e_tot) 
    tol = 1e-6

    converged = False
    while k < max_cycles:

        mo_coeff = C @ U 
        mc.run(nstates, mo_coeff=mo_coeff)
        
        eAve = sum(weights * mc.e_tot) 

        if abs(eAve - e_old) < tol:
            print('CASSCF converged at macroiteration {}'.format(k))
            print("E(CASSCF) = {}".format(mc.e_tot))
            converged = True
            break

        e_old = eAve

        # dm1, dm2 = mc.make_rdm12(0)
        dm1 = 0
        dm2 = 0
        for n in range(nstates):
            _dm1, _dm2 = mc.make_rdm12(n)
            dm1 += _dm1 * weights[n]
            dm2 += _dm2 * weights[n]
                
        # U0 = orth(U + 0.1 * np.random.randn(nmo, ncas))

        # U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))
        U, E = opt(energy, U0, h1e, eri, dm1, dm2)

        print('U is unitary', U.T @ U)
        # print(E + mol.energy_nuc())

        k += 1

    if not converged:
        raise RuntimeError('Max macro steps reached. CASSCF not converged.')
        
        

    # # diis storage
    # maxdiis = 6
    # diis_error_convergence = 1.0e-5
    
    # diis_error_matrices = np.zeros((maxdiis, nmo, ncas))
    # diis_fock_matrices = np.zeros_like(diis_error_matrices)
    
    # def diis(fock, dens, overlap, orth, iter, diis_min=1):
    #     """
    #     Extrapolate new fock matrix based on input fock matrix
    #         and previous fock-matrices.
    
    #     Arguments:
    #         fock -- current fock matrix
    
    #     Returns:
    #         (fock, error) -- interpolated fock matrix and diis-error
    #     """
    #     diis_fock = np.zeros_like(fock)
    
    #     if iter <= diis_min:
    #         return fock, 0.0
    
    #     # copy data down to lower storage
    #     for k in reversed(range(1, min(iter, maxdiis))):
    
    #         diis_error_matrices[k] = diis_error_matrices[k-1][:]
    #         diis_fock_matrices[k] = diis_fock_matrices[k-1][:]
    
    #     # calculate error matrix
        
    #     # error_mat = reduce(np.dot, (fock, dens, overlap))
    #     # error_mat -= error_mat.T
    
    #     # # put orthogonal error matrix in storage
    #     # # pulay use S^(-1/2) but here we choose whatever the user has defined
        
    #     # diis_error_matrices[0]  = reduce(np.dot, (orth.T, error_mat, orth))
        
    #     diis_error_matrices[0]  = fock - 

    #     diis_fock_matrices[0] = fock[:]
    #     diis_error_index = np.abs(diis_error_matrices[0]).argmax()
    #     diis_error = math.fabs(np.ravel(diis_error_matrices[0])[diis_error_index])
    
    #     # calculate B-matrix and solve for coefficients that reduces error
    #     bsize = min(iter, maxdiis)-1
    #     bmat = -1.0 * np.ones((bsize+1,bsize+1))
    #     rhs = np.zeros(bsize+1)
    #     bmat[bsize, bsize] = 0
    #     rhs[bsize] = -1
    #     for b1 in range(bsize):
    #         for b2 in range(bsize):
    #             bmat[b1, b2] = np.trace(diis_error_matrices[b1].dot(diis_error_matrices[b2]))
    #     C =  np.linalg.solve(bmat, rhs)
    
    #     # form new interpolated diis fock matrix
    #     for i, k in enumerate(C[:-1]):
    #         diis_fock += k*diis_fock_matrices[i]
    
    #     return diis_fock, diis_error