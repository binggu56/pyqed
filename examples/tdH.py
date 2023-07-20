# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:21:32 2021

@author: Bing
"""
from lime.phys import pauli
import numpy as np 

s0, sx, sy, sz = pauli()

def coeff1(t, **args):
    return np.sin(t) * sx

H = [sz, [sx, coeff1]]
print(len(H))

def calculateH(t):
    
    Ht = H[0]

    for i in range(1, len(H)):
        Ht += + H[i][1](t)
    
    return Ht

print(calculateH(0.1))
    