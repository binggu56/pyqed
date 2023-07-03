import numpy as np
import pylab as plt
import matplotlib.pyplot as plt 
import matplotlib as mpl 

#data = np.genfromtxt(fname='/home/bing/dissipation/energy.dat')

data = np.genfromtxt(fname='energy.dat') 

fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)

#font = {'family' : 'ubuntu',
#        'weight' : 'normal',
#        'size'   : '16'}

#mpl.rc('font', **font)  # pass in the font dict as kwargs

mpl.rcParams['font.size'] = 12
#mpl.rcParams['figure.figsize'] = 8,6

#pl.title('two-steps fitting alg')
ax1.set_ylabel('Energy [hartree]')
ax1.plot(data[:,0],data[:,2],'b--',linewidth=2,label='Potential')
#pl.plot(dat[:,0],dat[:,2],'r-',linewidth=2)
ax1.plot(data[:,0],data[:,3],'g-.',linewidth=2,label='Quantum Potential')
ax1.plot(data[:,0],data[:,4],'k-',linewidth=2,label='Energy')
#pl.legend(bbox_to_anchor=(0.5, 0.38, 0.42, .302), loc=3,ncol=1, mode="expand", borderaxespad=0.)
#ax1.set_yticks((0.4,0.6,0.8))
ax1.legend(loc=0)
ax1.set_ylim(0,5)

ax2.set_xlabel('time [a.u.]')
ax2.set_ylabel('Energy [hartree]')
ax2.plot(data[:,0],data[:,1],'r--',linewidth=2,label='$Kinetic$')
#pl.plot(dat[:,0],dat[:,1],'k-',linewidth=2)
ax2.set_yscale('log')
#ax2.set_xticks((0,4,8))
#ax2.set_yticks((1e-7,1e-5,1e-3))
plt.legend(loc=0)

plt.subplots_adjust(hspace=0.)
plt.show() 

