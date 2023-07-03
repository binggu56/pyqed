# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:53:05 2016

@author: Bing Gu 
"""

import matplotlib.pyplot as plt 
import numpy as np 

data = np.genfromtxt('pimc.out')
x = data[:,0]
plt.plot(data[:,0],data[:,1]*10, label='PIMC')
plt.plot(x, np.exp(-x*x) * np.sqrt(1.0/2.0/np.pi),label='Exact')

plt.legend() 
plt.show() 

