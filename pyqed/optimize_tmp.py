'''
A feasible method for optimization with orthogonality constraints

ref: https://link.springer.com/article/10.1007/s10107-012-0584-1. DOI: 10.1007/s10107-012-0584-1
'''

import numpy as np
from opt_einsum import contract

# algorithm 1: a gradient descent method with curvilinear search

def minimize_1():

    k = 0
    

    pass




# algorithm 2: a curvilinear search method with Barzilai-Borwein step

def minimize_2(f, X0, args=(), tau, taum, tauM, eta, rho1, delta, epsilon):

    n, p = X0.shape

    Id = np.identity(n)
    k = 0
    C = f(X0, *args)
    Q0 = Id

    X = X0
    Q = Q0

    G = gradient(X0, *args)
    df = grad(X, G)

    while norm(df) > epsilon:

        A = G @ X.T - X @ G.TDM
        Y = np.linalg.inv(Id + tau/2 * A) @ (Id - tau/2 * A) @ X

        while f(Y, *args) >= C + rho1 * tau * (-0.5 * norm(A)**2):
            tau = delta * tau

        X_new = Y
        Q_new = eta * Q + 1

        v = f(X_new, *args)

        C_new = (eta * Q * C + v) / Q_new
        G_new = gradient(X_new, *args)
        df_new = grad(X_new, G_new)


        tau = step_size(X_new - X, df_new - df, k, parameter=1)
        tau = max(min(tau, tauM), taum)

        k += 1

        X = X_new
        Q = Q_new
        C = C_new
        G = G_new
        df = df_new

    return X, v

def step_size(S, Y, k, parameter):

    if parameter == 1:
        tau = np.trace(S.T @ S) / np.abs(np.trace(S.T @ Y))
    elif parameter == 2:
        tau = np.abs(np.trace(S.T @ Y)) / np.trace(Y.T @ Y)
    else:
        raise ValueError("There is no parameter = {} choice for step_size. Use 1 or 2".format(parameter))

    return tau

def grad(X, G):
    """
    Riemmann gradient

    Parameters
    ----------
    X : _type_
        _description_
    G : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    return G - X @ G.T @ X

def inner(A, B):
    """
    Euclidean inner product between two matrices A and B

    Parameters
    ----------
    A : _type_
        _description_
    B : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    return np.trace(A.T @ B)

def norm(A):
    """
    Frobenius norm of A

    Parameters
    ----------
    A : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    return np.sqrt(inner(A, A))

def gradient(U, h1e, h2e, dm1, dm2):
    g = h1e @ U @ dm1.T + h1e.T @ U @ dm1  # these two terms are probably the same
    g += 0.5 * (contract('pqrs, qb, rc, sd, abcd -> pa', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, rc, sd, abcd -> qb', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, sd, abcd -> rc', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, rc, abcd -> sd', h2e, U, U, U, dm2) )
    return g