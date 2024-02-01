#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:33:25 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.models.ShinMetiu import ShinMetiu

if __name__=='__main__':
    import time
    from pyqed.ldr.ldr import LDRN
    
    import proplot as plt
    
    
    # Example usage:
    mol = ShinMetiu(dvr_type='sine')
    mol.create_grid(6, domain=[[-15, 15]]) # check whether the domain enough big
    X, E, U = mol.pes(domain=[-8,8], level = 5)
    print(E)
    
    print(U.shape)
    
    fig, ax = plt.subplots()

    for i in range(3):
        ax.plot(X, E[:,i])