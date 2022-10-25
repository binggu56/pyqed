#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:30:04 2022

Some trivial computations for the group theory

@author: bing
"""

import numpy as np

a = np.zeros((4,2))
a[0, 0] = 1
a[3, 0] = 1
a[1, 1] = 1
a[2, 1] = 1
u, v, w = np.linalg.svd(a)

print(v)
print(u)
print(w)