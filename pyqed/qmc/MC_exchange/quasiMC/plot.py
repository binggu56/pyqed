# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:28:23 2016

@author: bing
"""

import numpy as np 
import matplotlib.pyplot as plt  

import matplotlib as mpl 

#plt.style.use('ggplot')

font = {'family' : 'Times New Roman',
        'weight' : 'normal', 
        'size'   : '12'}

mpl.rc('font', **font)  # pass in the font dict as kwargs

#dat = np.genfromtxt('sobol/random.dat')
#
#fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(ncols=2,nrows=2) 
#
#ax4.plot(dat[:,0],dat[:,1],'o',markersize=3)
#
#dat2 = np.genfromtxt('ran1/normal.dat')
#
#ax3.plot(dat2[:,0],dat2[:,1],'o',markersize=3)
#
#x,y = np.genfromtxt('random.dat',unpack=True)
#
#ax2.plot(x,y,'o',markersize=3)
#
#x,y = np.genfromtxt('ran2.dat',unpack=True)
#
#ax1.plot(x,y,'o',markersize=3)

x = np.array([10000, 20000, 100000 , 200000, 500000]) 
x = 1./np.sqrt(x)

y = np.array([1.0489922729470280, 1.0151243791486708, 1.0081307660222230, 
     1.0036717974790175, 1.0014285885836964]) 
y = y-1. 

z = np.array([1.0145262113813669,0.98850412318143310, 1.0001542763276223,0.99939065597754062, 
                1.0000680719061754  ]) 
z = np.abs(z-1.) 

fig, ax = plt.subplots() 
ax.plot(x,y,'g-o',label='Normal')
ax.plot(x,z,'r-s',label='Uniform')   
ax.set_xlabel('$1/\sqrt{N}$')
ax.set_ylabel('$\gamma -1 $')
ax.set_xlim(0,0.012)
ax.set_ylim(-0.002,0.05)

plt.legend(loc=0)

#ax.set_title('$\int e^{-x_1^2/2}dx$')
plt.show() 
plt.grid(False)
plt.savefig('/home/bing/quasiMonteCarlo/integration.pdf')