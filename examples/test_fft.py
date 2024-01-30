#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 18:01:28 2021

@author: bing
"""

import numpy as np
from numpy import pi

from lime.fft import fft, ifft, dft

t = np.linspace(-80, 80, 1024)
dt = t[1] - t[0]

# y = np.fft.ifft(np.exp(-1j * t))
x = np.exp(1j * t)
freq, y = fft(x, t)

y1 = dft(t, x, freq)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(freq, y.real)
# ax.plot(freq, y1.real, 'r')
ax.set_xlim(-2,2)