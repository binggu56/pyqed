#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:56:48 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import numpy as np 
from time import perf_counter as tpc
from opt_einsum import contract
import teneva 

def func(X):
    """Schaffer function."""
    Z = X[:, :-1]**2 + X[:, 1:]**2
    y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
    return np.sum(y, axis=1)


d = 7      # Dimension of the function
a = -5.    # Lower bounds for spatial grid
b = +6.    # Upper bounds for spatial grid

m_trn  = 1.E+5  # Train data size (number of function calls)
m_vld  = 1.E+3  # Validation data size
m_tst  = 1.E+5  # Test data size
nswp   = 6      # Sweep number for ALS iterations

r      = 5      # TT-rank of the initial random tensor
n      = 2      # Initial shape of the coefficients' tensor
n_max  = 20     # Maximum shape of the coefficients' tensor

X_trn = np.vstack([np.random.uniform(a, b, int(m_trn)) for k in range(d)]).T
y_trn = func(X_trn)

X_vld = np.vstack([np.random.uniform(a, b, int(m_vld)) for k in range(d)]).T
y_vld = func(X_vld)

X_tst = np.vstack([np.random.uniform(a, b, int(m_trn)) for k in range(d)]).T
y_tst = func(X_tst)

t = tpc()
A0 = teneva.rand([n]*d, r)
A = teneva.als_func(X_trn, y_trn, A0, a, b, nswp, e=None,
    X_vld=X_vld, y_vld=y_vld, n_max=n_max, log=True)
t = tpc() - t

print(f'Build time     : {t:-10.2f}')

# >>> ----------------------------------------
# >>> Output:

# # pre | time:      0.153 | rank:   5.0 | e_vld: 1.7e+00 |
# #   1 | time:      1.129 | rank:   5.0 | e_vld: 2.7e-01 | e: 1.0e+00 |
# #   2 | time:      2.853 | rank:   5.0 | e_vld: 2.3e-01 | e: 6.6e-01 |
# #   3 | time:      5.373 | rank:   5.0 | e_vld: 1.8e-01 | e: 5.4e-01 |
# #   4 | time:      8.340 | rank:   5.0 | e_vld: 1.4e-01 | e: 3.3e-01 |
# #   5 | time:     11.770 | rank:   5.0 | e_vld: 8.3e-02 | e: 2.5e-01 |
# #   6 | time:     15.886 | rank:   5.0 | e_vld: 6.9e-02 | e: 1.3e-01 | stop: nswp |
# Build time     :      15.90
#


t = tpc()

y_our = teneva.func_get(X_tst, A, a, b)
e = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)

t = tpc() - t
print(f'Relative error : {e:-10.1e}')
print(f'Check time     : {t:-10.2f}')

# >>> ----------------------------------------
# >>> Output:

# Relative error :    7.1e-02
# Check time     :       4.76
#

teneva.show(A)