import numpy as np 
import constants as paras 

def trial(x,a):
    return a[0] + a[1] * x

def linear_fit_domain_nor(x, w, L=2):
    """
    linear fit with spacial domains without r 
    
    input: 
        L : number of domains 
    """
    
    am = paras.am 

    # define domain functions 
    d = 16.0
    #xdom = [-1.0, 1.0]
    xdom = [0.0] 
    
    if abs(len(xdom)+1-L) > 0:
        print('Error : size of xdom does not match number of domains.')

    domFunc = dDomFunc = ddDomFunc = []
    
    y = d * (x - xdom[0])  
    #domFunc.append(0.5 * (1. - np.tanh(y))) 
#dDomFunc.append( - 0.5 * d / np.cosh(y)**2 ) 
#   ddDomFunc.append( d**2 * np.tanh(y)/np.cosh(y)**2 )

    if L > 2:
        for i in range(L-2):
            domFunc.append(0.5 * (np.tanh(d*(x - xdom[i])) - np.tanh(d * (x - xdom[i+1]))))
            dDomFunc.append(0.5 * (d / np.cosh(d*(x-xdom[i]))**2 - d / np.cosh(d * (x-xdom[i+1])**2)))
    
    # for two domains only 
    lastDom = 0.5 * (1. + np.tanh(y))    
    dLastDom = 0.5 * d / np.cosh(y)**2
    ddLastDom = d**2 * np.tanh(y)/np.cosh(y)**2

    domFunc = [ 0.5 * (1. - np.tanh(y)), 0.5 * (1. + np.tanh(y))]
    dDomFunc = [ - 0.5 * d / np.cosh(y)**2, 0.5 * d / np.cosh(y)**2 ] 
    ddDomFunc = [ d**2 * np.tanh(y)/np.cosh(y)**2, d**2 * np.tanh(y)/np.cosh(y)**2]

#domFunc.append(lastDom)
#dDomFunc.append(dLastDom)
#ddDomFunc.append(ddLastDom) 

    
    u = r = dr = ddr = fq = np.zeros(len(x))
       
    Nb = 2 # number of basis 

    print('domain Function',domFunc[0],'\n',domFunc[1])

    for k in range(L):
        
        S = np.zeros((Nb,Nb))
        S[0,0] = np.dot(domFunc[k],w)
        S[0,1] = S[1,0] = np.dot(x * domFunc[k], w)
        S[1,1] = np.dot(x**2 * domFunc[k], w)
    
        b = np.zeros(Nb)
        b[0] = np.dot(dDomFunc[k],  w)
        b[1] = np.dot(x *  dDomFunc[k] + domFunc[k], w)

        b = - 0.5 * b   
  
        print('domain ',k,'\n',S,'\n',b)

        a = np.linalg.solve(S,b)

        print('fitting coefficients ',a)
        
        r = trial(x,a) 
        dr = a[1]  
        ul = r*r + dr 
        dul = 2. * r * dr 
        u += - 1./2./am * (ul * domFunc[k] + r * dDomFunc[k]) 
        fq += 1./2./am * (ul * dDomFunc[k] + dul * domFunc[k] + dr * dDomFunc[k] + r * ddDomFunc[k])

    Eu =  np.dot(u, w) 

    return Eu, fq 
