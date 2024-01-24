#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:06:11 2023

@author: bing
"""

import numpy as np 
import scipy 
from pyqed.nonherm import eig
a = np.array([[0+0.3j, 0.2j], [0.5-0.2j, 0.6]])
w, ul, ur = scipy.linalg.eig(a, left=True)
print(ur)
print(ul.conj().T @ ur)

w, vr, vl = eig(a)
print(vr)
print(vl @ vr)