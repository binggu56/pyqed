import numba 

@numba.autojit
def derivs(x):
    """
    Morse potential     
    """ 
    
    PES = 'AHO' 
    
    if PES == 'Morse':
        
        a, x0 = 1.02, 1.4 
        De = 0.176 / 100.0 
    
        d = (1.0-np.exp(-a*x))
        
        v0 = De*d**2
            
        dv = 2. * De * d * a * np.exp(-a*x)
        
    elif PES == 'HO':
        
        v0 = x**2/2.0 
        dv = x 
    
    elif PES == 'AHO':
        
        eps = 0.4 
        
        v0 = x**2/2.0 + eps * x**4/4.0 
        dv = x + eps * x**3  
        
        #ddv = 2.0 * De * (-d*np.exp(-a*((x-x0)))*a**2 + (np.exp(-a*(x-x0)))**2*a**2)

    elif PES == 'pH2':
        
        dx = 1e-4
        
        v0 = np.zeros(Ntraj)
        dv = np.zeros(Ntraj)
        
        for i in range(Ntraj):
            v0[i] = vpot(x[i])
            dv[i] = ( vpot(x[i] + dx) - v0[i])/dx
        
    elif PES == 'double_well':

        a = 4.0 
        b = 12.0 
        v0 = x**2 * (a * x**2 - b) + b**2/4./a 
        dv = 4. * a * x**3 - b * 2. * x 
        
    return v0,dv
