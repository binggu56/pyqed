#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 23:03:15 2025

@author: bingg
"""

import numpy as np
from opt_einsum import contract


def _factorized_two_electron_energy(U, pair_factors, dm2):
    """Two-electron orbital objective from Cholesky pair factors."""
    transformed = contract('Ppq,pa,qb->Pab', pair_factors, U, U)
    return 0.5 * contract('Pab,Pcd,abcd->', transformed, transformed, dm2)


def _factorized_two_electron_gradient(U, pair_factors, dm2):
    """Two-electron orbital gradient from Cholesky pair factors."""
    transformed = contract('Ppq,pa,qb->Pab', pair_factors, U, U)
    left = contract('Pcd,abcd->Pab', transformed, dm2)
    right = contract('Pab,abcd->Pcd', transformed, dm2)
    return 0.5 * (
        contract('Ppq,qb,Pab->pa', pair_factors, U, left)
        + contract('Ppq,pa,Pab->qb', pair_factors, U, left)
        + contract('Prs,sd,Pcd->rc', pair_factors, U, right)
        + contract('Prs,rc,Pcd->sd', pair_factors, U, right)
    )


def minimize(f, X0, args=(), tau=2, taum=1e-15, tauM=1e15, eta=0.85,
             rho1=0.5, delta=0.2, epsilon=1e-5, algorithm='RCG',
             history_size=7):
    """
    Minimize ``f(X)`` subject to orthonormal columns ``X.T @ X = I``.

    The optimizer keeps the original ``U``-matrix formulation used by the local
    CASSCF code, but upgrades the search direction from plain steepest descent
    to a Riemannian conjugate-gradient (RCG) step on the Stiefel manifold.
    ``algorithm='SD'`` is kept as a fallback for debugging or comparison.

    Parameters
    ----------
    f : callable
        Objective function.
    X0 : ndarray
        Initial orthonormal-column guess.
    args : tuple, optional
        Extra arguments forwarded to ``f`` and ``gradient``.
    tau : float, optional
        Initial step size.
    taum : float, optional
        Minimum allowed step size.
    tauM : float, optional
        Maximum allowed step size.
    eta : float, optional
        Weight for the non-monotone line-search reference energy.
    rho1 : float, optional
        Armijo factor.
    delta : float, optional
        Backtracking reduction factor.
    epsilon : float, optional
        Convergence threshold on the Riemannian gradient norm.
    algorithm : {'RCG', 'SD', 'LBFGS'}, optional
        Optimization algorithm on the Stiefel manifold.
    history_size : int, optional
        Number of secant pairs kept by the limited-memory BFGS backend.

    Returns
    -------
    X : ndarray
        Optimized matrix with orthonormal columns.
    v : float
        Objective value at ``X``.

    References
    ----------
    Optimization Lett. 2022, 16:1773
    """
    algorithm = algorithm.upper()
    if algorithm not in ('RCG', 'SD', 'LBFGS'):
        raise ValueError(
            "Unknown orthogonality-constrained optimizer '{}'. Use 'RCG', 'SD' or 'LBFGS'.".format(
                algorithm
            )
        )

    # Start from a projected point so the optimizer can be called with slightly
    # noisy guesses without violating the manifold constraint.
    X = project(X0)
    C = f(X, *args)
    Q = 1.0
    G = gradient(X, *args)
    df = grad(X, G)
    direction = -df
    v = C
    k = 0
    lbfgs_s = []
    lbfgs_y = []

    while norm(df) > epsilon:
        if algorithm == 'LBFGS':
            direction = -lbfgs_direction(df, lbfgs_s, lbfgs_y)

        directional_derivative = np.real(inner(df, direction))
        if directional_derivative >= 0:
            # Restart if conjugacy was lost numerically and the direction no
            # longer points downhill.
            direction = -df
            directional_derivative = -np.real(inner(df, df))

        step = max(min(tau, tauM), taum)
        Y = retract(X, step * direction)
        trial_value = f(Y, *args)

        # Non-monotone Armijo backtracking: use the weighted reference energy
        # ``C`` so the optimizer can occasionally accept small uphill moves
        # while still converging more aggressively than strict monotone descent.
        while trial_value > C + rho1 * step * directional_derivative:
            step *= delta
            if step < taum:
                step = taum
                Y = retract(X, step * direction)
                trial_value = f(Y, *args)
                break
            Y = retract(X, step * direction)
            trial_value = f(Y, *args)

        Xnew = Y
        Qnew = eta * Q + 1.0
        v = trial_value
        Cnew = (eta * Q * C + v) / Qnew
        Gnew = gradient(Xnew, *args)
        df_new = grad(Xnew, Gnew)

        transported_grad = transport(Xnew, df)
        if algorithm == 'RCG':
            beta_num = np.real(inner(df_new, df_new - transported_grad))
            beta_den = max(abs(np.real(inner(df, df))), 1e-16)
            beta = max(0.0, beta_num / beta_den)
            transported_dir = transport(Xnew, direction)
            direction = -df_new + beta * transported_dir
        elif algorithm == 'LBFGS':
            update_lbfgs_history(lbfgs_s, lbfgs_y, Xnew, Xnew - X, df_new - transported_grad, history_size)
            direction = -lbfgs_direction(df_new, lbfgs_s, lbfgs_y)
        else:
            direction = -df_new

        tau = safe_stepsize(k + 1, Xnew - X, df_new - transported_grad, step)
        tau = max(min(tau, tauM), taum)

        k += 1
        X = Xnew
        Q = Qnew
        C = Cnew
        G = Gnew
        df = df_new

    return X, v



def norm(A):
    """
    Frobenius norm of matrix
    .. math::

        ||A||_F = \sqrt{ A^\dagger A }
    """
    return np.sqrt(np.real(np.trace(A.T.conj() @ A)))


def sym(A):
    """Hermitian/symmetric part of a square matrix."""
    return 0.5 * (A + A.T.conj())

def grad(X, G=None):
    """
    Project the Euclidean gradient onto the Stiefel tangent space at ``X``.

    The tangent-space projection is the Riemannian gradient used by the
    manifold optimizer.
    """
    if G is None:
        G = gradient(X)
    return tangent_projection(X, G)


def tangent_projection(X, Z):
    """Project ``Z`` onto the tangent space of the Stiefel manifold at ``X``."""
    return Z - X @ sym(X.T.conj() @ Z)


def transport(X, Z):
    """
    Transport a tangent vector to the tangent space at a new point ``X``.

    For the current projected-retraction optimizer, a simple tangent-space
    reprojection is a practical vector transport.
    """
    return tangent_projection(X, Z)

def project(V):
    """
    Project ``V`` onto the Stiefel manifold by polar orthonormalization.

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
    w, Q = np.linalg.eigh(V.T @ V)
    return V @ Q @ np.diag(1/np.sqrt(w)) @ Q.T


def retract(X, D):
    """
    Retract a tangent step ``D`` back to the Stiefel manifold.

    Keeping retraction as a small helper makes the line search and the search
    direction logic easier to read in ``minimize``.
    """
    return project(X + D)

def orth(V):
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
    w, Q = np.linalg.eigh(V.T @ V)
    return V @ Q @ np.diag(1/np.sqrt(w))

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
        tau = inner(dU, dU)/abs(inner(dU, dG))

    return tau


def safe_stepsize(k, dU, dG, fallback):
    """
    Barzilai-Borwein step with a safe fallback when the secant data degenerates.
    """
    denom1 = abs(inner(dG, dG))
    denom2 = abs(inner(dU, dG))
    if denom1 < 1e-16 or denom2 < 1e-16:
        return fallback

    tau = stepsize(k, dU, dG)
    if not np.isfinite(tau) or tau <= 0:
        return fallback
    return tau


def lbfgs_direction(grad_vec, s_history, y_history):
    """
    Apply the standard two-loop recursion for limited-memory BFGS.

    The secant pairs are tangent vectors on the Stiefel manifold.  We use the
    usual Euclidean algebra on those tangent coordinates because the optimizer
    already transports them back to the current tangent space before they enter
    the history.
    """
    if len(s_history) == 0:
        return grad_vec.copy()

    q = grad_vec.copy()
    alpha = []
    rho = []

    for s_vec, y_vec in zip(reversed(s_history), reversed(y_history)):
        sy = np.real(inner(s_vec, y_vec))
        if abs(sy) < 1e-16:
            alpha.append(0.0)
            rho.append(0.0)
            continue
        rho_i = 1.0 / sy
        alpha_i = rho_i * np.real(inner(s_vec, q))
        q = q - alpha_i * y_vec
        alpha.append(alpha_i)
        rho.append(rho_i)

    s_last = s_history[-1]
    y_last = y_history[-1]
    yy = np.real(inner(y_last, y_last))
    sy = np.real(inner(s_last, y_last))
    gamma = sy / yy if yy > 1e-16 else 1.0
    r = gamma * q

    for idx, (s_vec, y_vec) in enumerate(zip(s_history, y_history)):
        rho_i = rho[-1 - idx]
        alpha_i = alpha[-1 - idx]
        if rho_i == 0.0:
            continue
        beta = rho_i * np.real(inner(y_vec, r))
        r = r + s_vec * (alpha_i - beta)

    return r


def update_lbfgs_history(s_history, y_history, Xnew, raw_step, raw_grad_diff, history_size):
    """
    Store one transported secant pair for the manifold L-BFGS update.

    ``raw_step`` and ``raw_grad_diff`` are first built in ambient coordinates
    and then projected to the new tangent space before they are stored.
    """
    s_vec = transport(Xnew, raw_step)
    y_vec = transport(Xnew, raw_grad_diff)
    curvature = np.real(inner(s_vec, y_vec))
    if curvature <= 1e-12:
        return

    s_history.append(s_vec.copy())
    y_history.append(y_vec.copy())

    if len(s_history) > history_size:
        del s_history[0]
        del y_history[0]

def inner(a, b):
    return np.trace(a.T.conj() @ b)

def gradient(U, h1e, h2e, dm1, dm2):
    g = h1e @ U @ dm1.T + h1e.T @ U @ dm1  # these two terms are probably the same
    if np.ndim(h2e) == 3:
        g += _factorized_two_electron_gradient(U, h2e, dm2)
    else:
        g += 0.5 * (contract('pqrs, qb, rc, sd, abcd -> pa', h2e, U, U, U, dm2) + \
            contract('pqrs, pa, rc, sd, abcd -> qb', h2e, U, U, U, dm2) + \
            contract('pqrs, pa, qb, sd, abcd -> rc', h2e, U, U, U, dm2) + \
            contract('pqrs, pa, qb, rc, abcd -> sd', h2e, U, U, U, dm2) )
    return g


def energy(U, h1e, eri, dm1, dm2):
    """Orbital objective supporting dense ERIs or factorized pair factors."""
    e = contract('pq, pa, qb, ab ->', h1e, U, U, dm1)
    if np.ndim(eri) == 3:
        e += _factorized_two_electron_energy(U, eri, dm2)
    else:
        e += 0.5 * contract('pqrs, pa, qb, rc, sd, abcd ->', eri, U, U, U, U, dm2)
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





if __name__=='__main__':


    from pyqed import Molecule
    from pyqed.qchem.mcscf import CASCI

    mol = Molecule(atom='Li 0 0 0; H 0 0 1.4', unit='b', basis='631g')
    mol.build()

    mf = mol.RHF().run()
    C = mf.mo_coeff

    ncas=2
    nelecas = 2
    mc = CASCI(mf, ncas=ncas, nelecas=nelecas)
    nstates = 3
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

    U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))

    k = 0
    max_cycles = 20
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

        U, E = minimize(energy, U0, args=(h1e, eri, dm1, dm2))
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
