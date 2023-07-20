#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:07:44 2020

@author: bing
"""

import sys
sys.path.append(r'C:\Users\Bing\Google Drive\lime')
sys.path.append(r'/Users/bing/Google Drive/lime')


import numpy as np 

from lime.fft import fft, ifft 

x = np.linspace(-20, 20, 200)

f = np.exp(-x**2/2)

freq, g = fft(x, f)

y, h = ifft(freq, g)

print(freq)

import matplotlib.pyplot as plt 

fig, ax = plt.subplots()
ax.plot(freq, g)
nx = len(x)
dx = x[1] - x[0] 
#ax.plot(2. * np.pi * np.fft.fftfreq(nx, dx), np.fft.fft(f))

fig, ax = plt.subplots()
ax.plot(y, h)
ax.plot(x, f)
