# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:11:32 2019

@author: Bing
"""

import numpy as np 

A = np.random.randn(2,3,4)
print(A)
print(np.transpose(A) == np.transpose(A, (2,1,0)))