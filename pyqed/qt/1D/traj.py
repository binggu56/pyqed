#!/usr/bin/python

import numpy as np
import pylab as plt

#with open("traj.dat") as f:
#    data = f.read()
#
#    data = data.split('\n')
#
#    x = [row.split(' ')[0] for row in data]
#    y = [row.split(' ')[1] for row in data]
#
#    fig = plt.figure()
#
#    ax1 = fig.add_subplot(111)
#
#    ax1.set_title("Plot title...")    
#    ax1.set_xlabel('your x label..')
#    ax1.set_ylabel('your y label...')
#
#    ax1.plot(x,y, c='r', label='the data')
#
#    leg = ax1.legend()
#fig = plt.figure()
plt.subplot(111)
#plt.ylim(-8,8)
data = np.genfromtxt(fname='traj.dat') 
#data = np.loadtxt('traj.dat')
for x in range(1,data.shape[1]):
    plt.plot(data[:,x],data[:,0],lw=1)

#plt.figure(1) 
#plt.plot(x,y1,'-')
#plt.plot(x,y2,'g-')
plt.xlabel('time')
#plt.ylabel('position')
#plt.title('traj')

#plt.subplot(212)
#data = np.genfromtxt(fname='../2.0.2/traj.dat') 
#data = np.loadtxt('traj.dat')




plt.savefig('traj.pdf')

plt.show() 



