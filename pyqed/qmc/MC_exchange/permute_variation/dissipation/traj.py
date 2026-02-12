import numpy as np
import pylab as plt
import matplotlib.pyplot as plt 
import matplotlib as mpl 

#data = np.genfromtxt(fname='/home/bing/dissipation/energy.dat')

data = np.genfromtxt(fname='x.dat') 

fig, ax = plt.subplots(ncols=1, nrows=1)

#font = {'family' : 'ubuntu',
#        'weight' : 'normal',
#        'size'   : '16'}

#mpl.rc('font', **font)  # pass in the font dict as kwargs

mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = 8,6

print data.shape 

#for i in range(data.shape[-1]-2):
ax.plot(data[:,0],data[:1]) 
#ax.legend(loc=0)


#plt.subplots_adjust(hspace=0.)
plt.show() 

