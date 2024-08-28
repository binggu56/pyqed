#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:57:46 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""


from pyqed.lattice.chain import RiceMele
import matplotlib.pyplot as plt 

from pyqed import level_scheme

model = RiceMele(0.22, 0.2, 18)
e, _ = model.run()

print(e)
# level_scheme(e)

floquet = model.Floquet(omegad=0.1, E0=0.3, nt=61)



E, U, G = floquet.run()

print(G.shape)
# level_scheme(E)

for j in range(36):
    fig, ax = plt.subplots()
    ax.plot(G[j,:])




