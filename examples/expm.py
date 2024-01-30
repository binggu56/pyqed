#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 00:07:58 2024

@author: bingg
"""

from pyqed import expm
import numpy as np
import time 

a = np.random.rand(3000, 3000)
a = a + a.T
start = time.time()
expm(a, t=0.02, method='scipy')
end = time.time()

print(end-start)

start = time.time()
expm(a, t=0.02, method='diag')
end = time.time()

print(end-start)