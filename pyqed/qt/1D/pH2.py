# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:11:40 2016

@author: bingg
"""
import numpy as np 
import numba

bohr_angstrom = 0.52917721092
hartree_wavenumber = 219474.63 

Vmin = -24.2288 

b = np.array([-6.631e-02, 1.346e-01, -3.300e-02, 6e0, -1.4e01, -1.193e02, 2.290e02, \
            1.110e03, -1.850e03, -3.5e03, 6.0e03])
            
@numba.autojit 
def vpot(r):
    
    re = 3.47005
    De = 24.2288 
	
    r = r * bohr_angstrom 
    
    beta_inf = np.log(2.0 * De / u_LR(re)) 
    
    s = 0.0        
    for j in range(11):
        s += b[j] * y_ref(r,1)**j
    
      
    beta = y_ref(r,6) * beta_inf + (1.0 - y_ref(r,6))  * s  
    
    vpot = De * (1.0 - u_LR(r)/u_LR(re) * np.exp(- beta * y_eq(r,6)))**2
    
    vpot = vpot + Vmin 
    
    vpot = vpot / hartree_wavenumber 
    
    return vpot 
	

def y_eq(r,n):
    
    re = 3.47005
     
    y_eq = (r**n - re**n)/(r**n + re**n) 
 
    return y_eq  
    
def y_ref(r,n):
    
    r_ref = 4.60
     
    z = (r**n - r_ref**n)/(r**n + r_ref**n)     

    return z
    
def u_LR(r):
    
    C6 = 5.820364e04
    C8 = 2.87052154e05 
    C10 = 1.80757343e06 
    
    z = damp(r,6) * C6/r**6 + damp(r,8) * C8/r**8 + damp(r,10) * C10 / r**10 
      
    return z
	
def damp(r,n):
    
   den = 1.10 
		 
   z = (1.0 - np.exp(-3.30 * den * r / n - 0.423 * (den * r)**2/np.sqrt(float(n))))**(n-1) 
   
   return z 
