import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
import pylab 

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

font = {'family' : 'Times New Roman', 'weight' : 'regular', 'size'   : '18'}

mpl.rc('font', **font)  # pass in the font dict as kwargs

mpl.rcParams['xtick.major.pad']='4'
mpl.rcParams['ytick.major.pad']='4'
#plt.figure(figsize=(14,9))

fig, ax = plt.subplots()

minorLocatorX = MultipleLocator(0.25)
minorLocatorY = MultipleLocator(0.2)

# for the minor ticks, use no labels; default NullFormatter


# data
x, y, yerr, exact = np.genfromtxt('energy.dat',unpack=True,skip_header=1, 
                                  comments = '#')

# First illustrate basic pyplot interface, using defaults where possible.


#plt.errorbar(x, y, yerr=yerr, ecolor='g',capsize=6,elinewidth=2, capthick=2,label='Approximate')

ax.set_ylabel('Ground-state Energy [$E_h$]')
ax.set_xlabel('R$_0$ [$a_0$]',labelpad=12)
plt.xlim(0.8,3.3)
plt.ylim(2,4.2)
plt.yticks((2.4,2.8,3.2,3.6,4.0))

plt.minorticks_on()
ax.tick_params(axis='both',which='minor',length=5,width=2,labelsize=18)
ax.tick_params(axis='both',which='major',length=8,width=2,labelsize=18)

plt.hlines(4.0, 0.8,3.5,linewidth=2,linestyles='dashed', colors='r')


#zpe = (0.318953,0.343397,0.351372)

#zpe += np.sqrt(2.)/4.0 
ax.plot(x, exact,'k--o', lw=2,markersize=8,label='Exact')

x = np.resize(x,4)
y = np.resize(y,4) 
x[-1] = 3.0 
y[-1] = 3.523319

ax.plot(x,y,'g-s',linewidth=2, markersize=8,label='Approximate')

ax.xaxis.set_minor_locator(minorLocatorX)
ax.yaxis.set_minor_locator(minorLocatorY)

#plt.annotate('$E_{local}$',(2,3.0))
plt.legend(loc=4, frameon=False)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.14)


plt.savefig('GSE_3D.pdf')


plt.show()
