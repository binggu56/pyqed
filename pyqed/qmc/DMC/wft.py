import matplotlib.pyplot as plt 
import numpy as np 

data = np.genfromtxt('psi.data')

r = np.linspace(0,4) 

plt.plot(data[:,0],data[:,1]*1.2,label='DMC')
plt.plot(data[:,0],data[:,2],label='Exact')


plt.legend() 
plt.show()


