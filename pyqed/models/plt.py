#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:20:33 2024

@author: bingg
"""

import numpy as np
import matplotlib.pyplot as plt

from pyqed.style import set_style

set_style(fontsize=11) 

dat = np.load('e_cas_mode2_l7.npz')

ds = dat['arr_0']
E = dat['arr_1']
e_cas = dat['arr_2']

# print(E)
# ds = np.linspace(-1, 1, 4)


fig, ax = plt.subplots()
# ax.plot(ds, E, '-s', color='C2', label='HF')
ax.plot(ds, e_cas[:,0], '-o', label='CASCI')
ax.plot(ds, e_cas[:,1], '-d', label='CASCI S$_1$')
ax.plot(ds, e_cas[:,2], '-s', label='CASCI S$_2$')

ax.legend(frameon=False)
ax.set_xlabel(r'$u_2\ (a_0)$')
ax.set_ylabel('Energy (a.u.)')
# ax.set_ylim(-1.55, -1.2)
# ax.set_xlim(-3, 2)

# plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
# fig.savefig('PEC_mode2_l7.pdf', bbox_inches='tight')