# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:01:48 2016

@author: bing
"""

import numpy as np 


s = 2000000
x = np.random.rand(s)

x = 8.0*(x-0.5)

anm = np.sqrt(1./2./np.pi)

print 8*anm*sum(np.exp(-0.5*x**2))/s 

