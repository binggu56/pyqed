#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:05:22 2021

@author: bing
"""

from lime.phys import norm2, interval
from lime.optics import jta, Biphoton
from lime.units import au2ev, au2fs
from lime.style import imshow

import numpy as np
from numpy import pi

omegap=2.0/au2ev
sigmap=0.4/au2ev
Te=2/au2fs

w = np.linspace(-4, 4, 128)/au2ev

biphoton = Biphoton(omegap=omegap, bw=sigmap, Te=Te, p=w, q=w)
biphoton.get_jsa()

t1, t2, J1 = biphoton.get_jta()
dt = interval(t1)

print('norm = ', norm2(J1, dt, dt)/(2*pi)**2)


# analytical expression
t1 = np.linspace(-10, 10, 512)/au2fs
t2 = t1
T1, T2 = np.meshgrid(t1, t2)

J = jta(T2, T1, omegap=2.0/au2ev, sigmap=0.4/au2ev, Te=2/au2fs)

# plt, ax = subplots()

imshow(t1, t2, np.real(J.T))

dt = interval(t1)
print(norm2(J, dt, dt)/(2*pi)**2)