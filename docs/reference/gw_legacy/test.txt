#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:57:46 2023

@author: bing
"""
import numpy as np

# a = np.random.rand(2,2)

# print(a[1][0], a[1,0])
def scalar2array(a, d=1):
    if d == 1:
        return np.atleast_1d(a)
    elif d == 2:
        return np.reshape(a, (1,1))
    else:
        raise ValueError('A scalar cannot be transformed to {} dim array'.format(d))

a = 0.1
a = np.reshape(a, (1,1))
print(a)
b = 0.5
b = scalar2array(b)
print(b)