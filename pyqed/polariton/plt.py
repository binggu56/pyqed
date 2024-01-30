#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:21:01 2023

@author: bing
"""
import numpy as np
import proplot as plt
from pyqed import load_result
from pyqed import Mol, sigmax, comm
from pyqed.units import au2ev, au2fs



dat = np.load('negf_0.1.dat.npz')
ts = dat['arr_0']
rho = dat['arr_1']

# fig, ax = plt.subplots()
# ax.plot(ts, pulse(ts))

fig, ax = plt.subplots(figsize=(4,3))

# ax.plot(ts, rho[:, 0, 1].real)
ax.plot(ts, rho[:, 1, 1].real, label='PP-NEGF')

# reference
dat = np.load('negf_0.npz')
ts = dat['arr_0']
rho = dat['arr_1']

# fig, ax = plt.subplots()
# ax.plot(ts, pulse(ts))

# ax.plot(ts, rho[:, 0, 1].real)
ax.plot(ts, rho[:, 1, 1].real, 'grey', label='Bare')

# reference
dat = np.load('negf_leak_0.02.npz')
ts = dat['arr_0']
rho = dat['arr_1']
ax.plot(ts, rho[:, 1, 1].real, 'r', label='PP-NEGF w/ leak')

########## exact with SC
result = load_result('exact.dat')
ax.plot(result.times*au2fs, result.observables[:,0].real, 'k--', label='Exact')

ax.format(xlabel='Time (fs)', ylabel=r'$\rho_{ee}(t)$', ylim=(0, 0.013), grid=False)
ax.legend(loc=2, ncols=2, frameon=False)

fig.savefig('pp.pdf')
# ax.plot(ts, rho[:, 0, 0].real)
# ax.format(ylim=(0,1))