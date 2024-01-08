#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 22:09:04 2023

@author: Bing Gu

system-bath quantum dynamics 

A Morse oscillator coupled with a set of harmonic oscilators 

"""
import numpy as np
#import scipy  
import numba 
import sys 
import math 

from pyqed import Result
import matplotlib.pyplot as plt




bohr_angstrom = 0.52917721092
hartree_wavenumber = 219474.63 

#hartree_wavenumber = scipy.constants.value(u'hartree-inverse meter relationship') / 1e2 

class ResultQT(Result):
    def __init__(self, ntraj, ndim, nstates, **kwargs):
        super(ResultQT, self).__init__(**kwargs)
        self.ntraj = ntraj
        self.ndim = ndim
        self.nstates = nstates
        
        self.c = None
        self.xlist = None
        self.p = None
        self.r = None
        self.rholist = None # electronic density matrices
    
    def expect(self):
        pass
    
    def plot_traj(self, d=0):
        

        
        if self.xlist is None:
            raise ValueError('No data for trajectories.')
        
        fig, ax = plt.subplots()
        for n in range(10):
            ax.plot(self.times, [x[n, d] for x in self.xlist], 'k', lw=1)
        
        return 
    
    def plot_rdm(self):
        # plot the electronic density matrices including populations and coherences
        fig, ax = plt.subplots()
        
        for j in range(self.nstates):
            ax.plot(self.times, [rho[j, j] for rho in self.rholist])
    

def M1mat(a, Nb):
    """
    matrix representation of position operators in harmonic oscilator eigenstates
    
    .. math::
        [x^n]_{ij} = \braket{i| x^n | j}

    Parameters
    ----------
    a : float
        Gaussian width parameter, :math:`\alpha = m \omega`
    Nb : int
        DESCRIPTION.

    Returns
    -------
    M1 : TYPE
        DESCRIPTION.

    """
    
    M1 = np.zeros((Nb,Nb)) 

    for m in range(Nb-1):
        M1[m,m+1] = np.sqrt(float(m+1)/2.0/a)

    M1 = Sym(M1) 

    return M1 

def M2mat(a, Nb):
    
    M2 = np.zeros((Nb,Nb)) 

    for m in range(Nb):
        M2[m,m] = (float(m) + 0.5)/a 
    
    if Nb > 1: 
        for m in range(Nb-2):
            M2[m,m+2] = np.sqrt(float((m+1)*(m+2)))/2.0/a 

    M2 = Sym(M2)

    return M2 

def M3mat(a, Nb):
    
    M3 = np.zeros((Nb,Nb)) 

    for m in range(Nb-1):
        M3[m,m+1] = 3.0 * (float(m+1)/2./a)**1.5 

    if Nb > 2:
        for m in range(Nb-3):
            M3[m,m+3] = np.sqrt(float((m+1)*(m+2)*(m+3))) / (2.0*a)**1.5 
    
    M3 = Sym(M3) 

    return M3 

def M4mat(a, Nb):
    
    M4 = np.zeros((Nb,Nb))

    for m in range(Nb):
        M4[m,m] =  float(3.0 * m**2 + 3.0 * (m+1)**2) / (2.*a)**2
    
    if Nb > 1: 
        for m in range(Nb-2):
            M4[m,m+2] = (4.0*m + 6.0) * np.sqrt(float((m+1)*(m+2))) / (2.*a)**2
            
    if Nb > 3: 
        for m in range(Nb-4):
            M4[m,m+4] = np.sqrt(float((m+1)*(m+2)*(m+3)*(m+4))) / (2.0*a)**2

    M4 = Sym(M4) 

    if Nb > 1:
        if not M4[0,1] == M4[1,0]: 
            print(M4) 
            print('\n ERROR: Not symmetric matrix M4.\n')
            sys.exit() 
    return M4


def Hermite(x, Nb):

    cons = np.array([1. / np.sqrt(float(2**n) * float(math.factorial(n))) for n in range(Nb)])
    
    H = [] 
    H.append(1.0) 
    H.append( x * 2.0 ) 
    if Nb > 2:
        for n in range(2,Nb):
            Hn = 2.0 * x * H[n-1] - 2.0*(n-1) * H[n-2]
            H.append(Hn)
    
    for n in range(Nb):
        H[n] = H[n]*cons[n] 

    return H

#    if n == 0: 
#        H.append(1.)  
#    elif n == 1: 
#        return 2. * x * cons 
#    elif n == 2: 
#        return (4. * x**2 - 2.) * cons   
#    elif n == 3: 
#        return (8.0 * x**3 - 12.0 * x) * cons 
#    elif n == 4:
#        return (16.0 * x**4 - 48.0 * x**2 + 12.0) * cons 
#    elif n == 5:
#        return (32.0*x**5 - 160.0*x**3 + 120.0*x) * cons 
#    elif n == 6: 
#        return ()


# Pauli matrices 

# def sigma_z():
    
#     S = np.zeros((2,2))
#     S[0,0] = 1.0
#     S[1,1] = -1.0 
    
#     return S 

# def sigma_x():
    
#     S = np.zeros((2,2))
#     S[1,0] = 1.0
#     S[0,1] = 1.0 
    
#     return S
    
    

def Vx(x):
    
    g = 0.1    
    return  x**2/2.0 + g * x**4 / 4.0

def Kmat(alpha,pAve, Nb):

    K = np.zeros((Nb,Nb),dtype=complex)

    ar = alpha.real 

    for j in range(Nb): 
        K[j,j] = np.abs(alpha)**2 / ar * (2. * j + 1.)/2. +  pAve**2 
    
    for j in range(1,Nb):
        K[j-1,j] = -1j*np.conj(alpha) * pAve * np.sqrt(2. * j / ar)
        K[j,j-1] = np.conj(K[j-1,j])

    if Nb > 2: 
        for j in range(2,Nb):
            K[j-2,j] = - np.sqrt(float((j-1)*j)) * np.conj(alpha)**2 / 2. / ar  
            K[j,j-2] = np.conj(K[j-2,j])
    

    #K[0,0] = np.abs(alpha)**2/alpha.real / 2. + pAve**2
    #K[1,1] = np.abs(alpha)**2/alpha.real * 3.0 / 2. + pAve**2 

    #K[0,1] = -1j*np.conj(alpha) * pAve * np.sqrt(2.*j/alpha.real)
    #K[1,0] = np.conj(K[0,1])
    K = K / (2.*amx) 

    return K 

def Sym(V):
    n = V.shape[-1]
    
    for i in range(n):
        for j in range(i):
            V[i,j] = V[j,i] 
    return V 

# @numba.jit
def Vint(a,Q):
    """
    bath configuration dependent interaction energy matrix 
    V_SB = x * (sum_k C[k] * Q[k])
    """ 
    global Nb 
#    PES = 'HO' 
#    
#    if PES == 'Morse':
#        
#        a, x0 = 1.02, 1.4 
#        De = 0.176 / 100.0 
#    
#        d = (1.0-np.exp(-a*x))
#        
#        v0 = De*d**2
#            
#        dv = 2. * De * d * a * np.exp(-a*x)
#        
#    elif PES == 'HO':
#        
#        v0 = x**2/2.0  + y**2/2.0 
#         
#
#    elif PES == 'AHO':
#        
#        eps = 0.4 
#        
#        v0 = x**2/2.0 + eps * x**4/4.0 
#        dv = x + eps * x**3  
        #ddv = 2.0 * De * (-d*np.exp(-a*((x-x0)))*a**2 + (np.exp(-a*(x-x0)))**2*a**2)

    # bilinear coupling 

    C = np.array([0.1] * Ndim) # coupling constant 
    
    Vx = M1mat(a, Nb)  

    tmp = np.zeros(Nb,Nb)
    
    for j in range(Ndim):
        tmp += Vx * C[j] * Q[j]
        
    return tmp 
    
#    elif PES == 'pH2':
#        
#        dx = 1e-4
#        
#        v0 = np.zeros(Ntraj)
#        dv = np.zeros(Ntraj)
#        
#        for i in range(Ntraj):
#            v0[i] = vpot(x[i])
#            dv[i] = ( vpot(x[i] + dx) - v0[i])/dx
        
    return v0 

def Vb(Q, szAve, xAve=None):
    """
    bath potential and force for one configuration 
    harmonic bath

    Parameters
    ==========
    omega: array of ndim
        bath frquencies     
    """
    # omegac = 2.5 # cutoff frequency 
    
    # omega = np.linspace(0.2, omegac, Ndim)

    v0 = np.zeros(ntraj)
    dv = np.zeros((ntraj, ndim))
    
    for i in range(ntraj):
        for j in range(ndim):
            v0 += mass[j] * omega[j]**2 * Q[i,j]**2/2.0 + szAve * g[j] * Q[i, j]  
            dv[i,j] = mass[j] * omega[j]**2 * Q[i,j] + szAve * g[j]

    return v0,dv 

#def LQF(x,w):
#    
#    xAve = np.dot(x,w) 
#    xSqdAve = np.dot(x*x,w) 
#    
#    var = (xSqdAve - xAve**2)
#
#    a = 1. / 2. / var 
#    
#    r = - a * (x-xAve) 
#    
#    dr = - a 
#
#    uAve =  (np.dot(r**2,w))/2./amy 
#                
#    du = -1./amy * (r*dr)
#
#    return r, du, uAve
 
# @numba.jit
def LQF(x,w):
    
    ntraj, ndim = x.shape
    Ndim = ndim
    Ntraj = ntraj 
    
    # C_{j \mu} = -1/2 \pa_\mu f_j
    c = np.zeros((ndim+1, ndim))
    for j in range(ndim):
        c[j, j] = -0.5
    # c = np.array([-0.5, ] * ndim + [0.0]) 
     
    S = np.zeros((Ndim+1,Ndim+1))
 
    for i in range(ntraj):
 
        f = np.array([*x[i,:],1.0])
 
        for m in range(Ndim+1):
            for n in range(m+1):
 
                S[m,n] += w[i] * f[m] * f[n]
    S = Sym(S)
 
    c = np.linalg.solve(S,c)
 
    #u = np.zeros(Ntraj)
    du = np.zeros((Ntraj,Ndim))
    r = np.zeros((Ntraj,Ndim))
    dr = np.zeros((Ntraj,Ndim))
 
    for i in range(Ntraj):
        
        f = np.array([*x[i,:],1.0])
        
        for j in range(Ndim):
            for k in range(Ndim+1):

                r[i,j] += c[k,j] * f[k] 
    
    for j in range(Ndim):
        for k in range(Ndim):
            dr[j,k] = c[j,k] 
 
    for i in range(Ntraj):
 
       # calculate quantum potential
        #u[i] = - r**2/(2.0*am) - c[1]/(2.*am)
        for j in range(Ndim):
            for k in range(Ndim):
        
                du[i,j] -= (2.0 * r[i,k] * dr[j,k])/ (2.0 * mass[k])
  
    return r,du

# @numba.jit
def qpot(x,p,r,w, ndim=1):

    """
    Linear Quantum Force : direct polynomial fitting of derivative-log density (amplitude)    
    curve_fit : randomly choose M points and do a nonlinear least-square fitting to a 
            predefined functional form      
    """
    
    #tau = (max(xdata) - min(xdata))/(max(x) - min(x))
    #if tau > 0.6:
    #    pass 
    #else: 
    #    print('Data points are not sampled well.'
    
    if ndim == 1:

        am= amy 
        
        Nb = 2
        S = np.zeros((Nb,Nb))
        
        for j in range(Nb):
            for k in range(Nb):
                S[j,k] = np.dot(x**(j+k), w)  
        
        bp = np.zeros(Nb)
        br = np.zeros(Nb)
        
        for n in range(Nb):
            bp[n] = np.dot(x**n * p, w)
            br[n] = np.dot(x**n * r, w)
            
            
        cp = np.linalg.solve(S,bp)
        cr = np.linalg.solve(S,br)  
    
        #unit = np.identity(Nb)
        #r_approx = cr[0] * unit + cr[1] * x + cr[2] * x**2 + cr[3] * x**3 
        #p_approx = cp[0] * unit + cp[1] * x + cp[2] * x**2 + cp[3] * x**3
    
        N = len(x)
        
        dr = np.zeros(N)
        dp = np.zeros(N)
        ddr = np.zeros(N)
        ddp = np.zeros(N)
        
        for k in range(1,Nb):    
            dr += float(k) * cr[k] * x**(k-1) 
            dp += float(k) * cp[k] * x**(k-1)  
        
        for k in range(2,Nb-1):
            ddr += float(k * (k-1)) * cr[k] * x**(k-2) 
            ddp += float(k * (k-1)) * cp[k] * x**(k-2) 
        
        fr =  -1./2./am * (2. * r * dp + ddp)
        fq = 1./2./am * (2. * r * dr + ddr)  
        
        Eu = -1./2./am * np.dot(r**2 + dr,w)
    
    elif ndim > 1:
        pass
        
    return Eu,fq,fr 

class Adiabatic:
    pass

class NAQT:
    """
    non-adiabatic quantum trajectory method. 
    The sys-bath state is represented by an ensemble of quantum trajectories (x, p, r)
    and expansion coefficients C(R)
    """
    def __init__(self, ntraj, ndim, nstates=1, mass=None):
        self.ntraj = ntraj
        self.ndim = ndim
        self.nstates = nstates

        # self.weights = self.w = None
        self.w = np.array([1./ntraj] * ntraj)
        
        if mass is None:
            mass = np.array([1] * ndim)
        self.mass = mass
    
    def sample(self, a, x0=0, p0=0):
        ndim = self.ndim 
        ntraj = self.ntraj
        
        # if isinstance(x0, float):
        #     x0 = [x0] * ndim
        
        x = np.random.randn(ntraj, ndim) 

        for j in range(ndim):            
            x[:,j] = x[:,j] / np.sqrt(2.0 * a[j]) + x0[j]
            
        # print('trajectory range {}, {}'.format(min(x),max(x)))
        
        p = np.zeros((ntraj, ndim))
        # p = p0

        r = np.zeros((ntraj, ndim))

        for j in range(ndim):
            r[:,j] = - a[j] * (x[:,j] - x0[j]) 
        
        # weights

        
        c = np.zeros((ntraj, nstates), dtype=complex)
        c[:, 1] = 1. 
        
        return x, p, r, c
    
    def norm(self, c):
        
        anm = 0.0 
    
        for k in range(self.ntraj):
            anm += np.vdot(c[k,:], c[k,:]).real * self.w[k]
        return anm

    def obs(self, s, c):
        
        # confirm s is an system operator
        assert(s.shape == (self.nstates, self.nstates))
        
        return np.einsum('km, mn, kn, k ->', c.conj(), s, c, self.w).real


    def expect(self, op, c):
        """
        compute expectation value of an system operator      
        """
        w = self.w 
        # nstates = self.nstates        
        # Xmat = M1mat(ax, M)
        # x_ave = 0.0+0.0j    
        # for k in range(self.ntraj):
        #     for m in range(M):
        #         for n in range(M):
        #             x_ave += Xmat[m,n] * np.conjugate(c[k,m]) * c[k,n] * w[k]
        xAve = np.einsum('mn, km, kn, k', op, c.conj(), c, w)
        return np.real_if_close(xAve)
    
    def xAve(self, x):
        pass
    
    # @numba.autojit 
    def rdm(self, c):
        """
            compute electronic reduced density matrix elements 
        """
        w = self.w 
        # nstates = self.nstates
        # rho = np.zeros((nstates, nstates),dtype=np.complex128)
        
        rho = np.einsum('ka, kb, k -> ab', c, c.conj(), w)
        
        return rho 

    def run(self, dt=0.001, nt=10, nout=1, e_ops=[], friction=0,\
            t0=0):
        
        # H = self.H  
        
        mass = self.mass 
        a = omega * mass
        w = self.w
        
        # initial conditions
        # x, p, c = x0.copy(), p0.copy(), c0.copy()
        x0, p0, r0, c0 = self.sample(a, x0=-1/omega, p0=[0]*ndim)
        #TODO: use mass-scaled coordinates
        x = x0.copy()
        p = p0.copy()
        r = r0.copy()
        c = c0.copy()        
        # classical and quantum force 
        
        szAve = self.obs(sz, c)
        v0, dv = Vb(x, szAve)
        
        ry, du = LQF(x, w)

        cold = c0
        dcdt = get_dcdt(H, c, x,ry,p, w, mass, szAve)
        c  += dcdt * dt
        
        rho = self.rdm(c)
        xlist = [x0]
        plist = [p0]
        rholist = [rho.copy()]
        clist = [c0]
        
        result = ResultQT(ntraj, ndim, nstates=nstates, dt=dt, Nt=nt)
        
        t = t0
        dt2 = dt/2
        
        for k in range(nt//nout):
            
            for k1 in range(nout):    
                
                t += dt 
    
                p += (- dv - du) * dt2 - friction * p * dt2   
                
                for j in range(ndim):
                    x[:,j] +=  p[:,j] * dt/mass[j] 
    
                # force field 
                ry, du = LQF(x, w)
                
                szAve = self.obs(sz, c)
                v0, dv = Vb(x, szAve)
    
                p += (- dv - du) * dt2 - friction * p * dt2 
                
                # renormalization 
    
                # anm = norm(c,w)
                # c /= np.sqrt(anm)
                
                # update c 
               
                dcdt = get_dcdt(H, c, x, ry, p, w, mass, szAve)
                cnew = cold + dcdt * dt * 2.0
                cold = c 
                c = cnew
    
                
                #  output data for each timestep 
            #    d = c
            #    for k in range(Ntraj):
            #        for i in range(M):
            #            d[k,i] = np.exp(-1j*t*H[i,i])*c[k,i]
    
    
                # fx.write('{} {} \n'.format(t, szAve))
                       
                # f.write(fmt.format(t,*x[0:nout,0]))
    
                # fnorm.write(' {} {} \n'.format(t,anm))
    
                # output density matrix elements 
                rho = self.rdm(c)
                rholist.append(rho.copy())
                xlist.append(x.copy())
                clist.append(c.copy())
                plist.append(p.copy())
            

            # fden.write(' {} {} {} {} {}\n'.format(t,*rho.flatten()))
            
        #    Ek = np.dot(py*py,w)/2./amy  
        #    Ev = np.dot(v0,w) 
        #    Eu = Eu 
        #    Etot = Ek + Ev + Eu
        #    
        #    fe.write('{} {} {} {} {} \n'.format(t,Ek,Ev,Eu,Etot))
        #
        #
        #print('The total energy = {} Hartree. \n'.format(Etot))

        # print trajectory and coefficients 
        # for k in range(ntraj):
        #     fc.write( '{} {} {} {} \n'.format(x[k], c[k,0],c[k,-2],c[k,-1]))

        # fe.close()
        # f.close() 
        # fc.close()
        # fx.close()
        
        result.xlist = xlist
        result.p = plist 
        result.rholist = rholist 
        
        return result
    
    # def run(self, dt=0.001, nt=10):
        
    #     a = omega * self.mass
    #     w = self.w 
    #     mass = self.mass
    #     ndim = self.ndim
        
    #     x, p, r, c = self.sample(a, x0=[0] * ndim, p0=[0]*ndim)

    #     szAve = self.expect(sz, c)
    #     v0, dv = Vb(x, szAve)
    #     r, du = LQF(x, w)

    #     cold = c 
    #     dcdt = prop_c(c, x, r, p, w, szAve)
    #     c = c + dcdt * dt
        
    #     result = ResultQT(ntraj, ndim, nstates=nstates, dt=dt, Nt=nt)

        
    #     print('time range for propagation is [0,{}]'.format(nt*dt))
    #     print('timestep  = {}'.format(dt))
        
    #     print(dv.shape, du.shape, fric_cons, p.shape)
        
    #     t = 0
    #     dt2 = dt/2
    #     for k in range(nt):
            
    #         t += dt 

    #         p += (- dv - du) * dt2 - fric_cons * p * dt2   
            
    #         # for j in range(ndim):
    #         #     x[:,j] +=  p[:,j] * dt/mass[j] 
            
    #         x += np.einsum('nj, j -> nj', p, 1/mass) * dt
            
    #         # force field 
    #         r, du = LQF(x, w)
                
    #         szAve = self.expect(sz, c)
    #         v0, dv = Vb(x, szAve)
            


    #         p += (- dv - du) * dt2 - fric_cons * p * dt2 
            
    #         # renormalization 

    #         anm = norm(c,w)
    #         c /= np.sqrt(anm)
            
    #         # update c 
           
    #         dcdt = prop_c(c, x, r, p, w, szAve)

    #         cnew = cold + dcdt * dt * 2.0
    #         cold = c 
    #         c = cnew

            
    #         #  output data for each timestep 
    #     #    d = c
    #     #    for k in range(Ntraj):
    #     #        for i in range(M):
    #     #            d[k,i] = np.exp(-1j*t*H[i,i])*c[k,i]


    #         fx.write('{} {} \n'.format(t, szAve))
                   
    #         f.write(fmt.format(t, *x[0:nout,0]))

    #         fnorm.write(' {} {} \n'.format(t,anm))

    #         # output density matrix elements 
    #         rho = qt.rdm(c)
    #         fden.write(' {} {} \n'.format(t,rho))
            
    #     #    Ek = np.dot(py*py,w)/2./amy  
    #     #    Ev = np.dot(v0,w) 
    #     #    Eu = Eu 
    #     #    Etot = Ek + Ev + Eu
    #     #    
    #     #    fe.write('{} {} {} {} {} \n'.format(t,Ek,Ev,Eu,Etot))
    #     #
    #     #
    #     #print('The total energy = {} Hartree. \n'.format(Etot))

    #     # print trajectory and coefficients 
    #     for k in range(ntraj):
    #         fc.write( '{} {} {} {} \n'.format(x[k], c[k,0],c[k,-2],c[k,-1]))

    #     fe.close()
    #     f.close() 
    #     fc.close()
    #     fx.close()
        

def norm(c, w):
    
    anm = 0.0 

    for k in range(ntraj):
        anm += np.vdot(c[k,:], c[k,:]).real * w[k]
    return anm

# initialization 
# for nuclear DOF  : an ensemble of trajectories 
# for electron DOF : for each trajectory associate a complex vector C of dimension nstates 
   

# nfit = 4
# ax = 1.0 # width of the GH basis 
   
  
# x0 = 0.5
# initial conditions for c 
# c = np.zeros((Ntraj,M),dtype=np.complex128)

# mixture of ground and first excited state

#c[:,0] = 1.0/np.sqrt(2.0)+0j
#c[:,1] = 1.0/np.sqrt(2.0)+0j
#for i in range(2,M):
#    c[:,i] = 0.0+0.0j

# coherent state 
# z = 1.0/np.sqrt(2.0) * x0 
# for i in range(M):
#     c[:,i] = np.exp(-0.5 * np.abs(z)**2) * z**i / np.sqrt(math.factorial(i))

# print('initial occupation \n',c[0,:])
# print('trace of density matrix',np.vdot(c[0,:], c[0,:]))
# ---------------------------------
# initial conditions for QTs     
# ndim = Ndim = 2 # dimensionality of bath 
# y0 = np.zeros(Ndim)
# ay0 = np.array([4.0, 4.0])

# amy = np.array([10.0, 10.0])

# y = np.random.randn(ntraj, ndim) 

# for j in range(Ndim):            
#     y[:,j] = y[:,j] / np.sqrt(2.0 * ay0[j]) + y0[j]
    
# #print('trajectory range {}, {}'.format(min(y),max(y)))
# py = np.zeros((Ntraj,Ndim))
# ry = np.zeros((Ntraj,Ndim))

# for j in range(Ndim):
#     ry[:,j] = - ay0[j] * (y[:,j] - y0[j]) 

# w = np.array([1./Ntraj]*Ntraj)

# -------------------------------

# amx = 1.0 

# f_MSE = open('rMSE.out','w')
# nout = 20       # number of trajectories to print 
# fmt =  ' {}' * (nout+1)  + '\n'  
# Eu = 0.  


# fric_cons = 0.0      # friction constant  


# Nt = 200
# dt = 0.001
# dt2 = dt/2.0 
# t = 0.0 


    
# construct the Hamiltonian matrix for the system
# In the diabatic representation, H is the same for all trajectories whereas
# for adiabatic representation, H parameterically depends on the positions of each trajectory


# omega0 = 1 * electronvolt
# H = -0.5 * omega0 * sz

# print('Hamiltonian matrix in DOF x = \n')
# print(H)
# print('\n')

  
        
# @numba.autojit 
# def norm(c,w):
    
#     anm = 0.0 

#     for k in range(Ntraj):
#         anm += np.vdot(c[k,:], c[k,:]).real * w[k]
#     return anm

# @numba.autojit
def fit_c(c, x, w, deg=1):
    """
    global approximation of :math:`C_\alpha(R)` to obtain the derivative C'', C'    
    
    output fitting coefficients 
    
    params
    =======
    c: complex array (ntraj, nstates) 
        expansion coefficients
    
    x: array (ntraj, ndim)
        trajectories.
        
    """
    ntraj, ndim = x.shape
    nstates = c.shape[-1]
    
    if ndim == 1:
        
        dc = np.zeros((ntraj, nstates), dtype=np.complex128)
        ddc = np.zeros((ntraj, nstates), dtype=np.complex128)
    
        for j in range(nstates): # states 
    
            z = c[:,j]
            pars = np.polyfit(y,z,deg)
            p0 = np.poly1d(pars)
            p1 = np.polyder(p0)
            p2 = np.polyder(p1)
            
    #for k in range(Ntraj):
            dc[:,j] = p1(x)
            ddc[:,j] = p2(x)

    elif ndim > 1:

        dc = np.zeros((ntraj, nstates, ndim), dtype=np.complex128)
        ddc = np.zeros((ntraj, nstates, ndim), dtype=np.complex128)
        
        if deg == 1:
            nbasis = ndim + 1

            S = np.zeros((ndim + 1, ndim+1))
            b = np.zeros((ndim+1, nstates), dtype=complex)
            
            for i in range(ntraj):
         
                f = np.array([*x[i,:],1.0]) # fitting basis
         
                for m in range(ndim+1):
                    
                    for a in range(nstates):
                        b[m, a] += w[i] * c[i, a] * f[m]
                    
                    for n in range(m+1):
         
                        S[m,n] += w[i] * f[m] * f[n]
            S = Sym(S)
            
            coeff = np.linalg.solve(S, b)
            
            
            for a in range(nstates):
                for m in range(ndim):
                    dc[:, a, m] = coeff[m, a]

        elif deg == 2:
            nbasis = 1 + ndim + ndim**2
            pass 

        elif deg > 2:
            raise NotImplementedError('Only linear basis has been implemented. Set deg=1.')

    # error analysis
    # err = np.sum(c - x )
    
    return dc, ddc

def linear(x):
    return np.append(x, [1.0])

def H(x, szAve):
    h = 0.5 * sx 
    for i in range(ndim):
        h += (sz  - szAve * s0)* g[i] * x[i] #- np.eye(nstates) * g[i] * x[i] * szAve
    return h

# @numba.autojit 
def prop_c(c, x, r, p, w, szAve):
    
    ntraj, nstates = c.shape
    
    dc, ddc = fit_c(c, x, w)
    
    # M = nstates

    dcdt = np.zeros([ntraj, nstates], dtype=np.complex128)
        
    # Vcoup = Vint(a=1, Q)
    # X2 = M2mat(1, nstates)
    
    for k in range(ntraj):
        
        hk = Hs(x[k, :], szAve)
        
        # tmp = (H + Vp).dot(c[k,:]) - ddc[k,:]/2.0/amy - dc[k,:] * ry[k]/amy + Va * c[k,:]
        tmp = hk.dot(c[k,:])  - dc[k,:,:] @ (r[k,:]/mass)

        dcdt[k,:] = -1j * tmp
       
    return dcdt

def get_dcdt(H,c,y,ry,py, w, mass, *args):
    
    ntraj, nstates = c.shape
    
    dc, ddc = fit_c(c,y, w)
    
    # M = nstates

    dcdt = np.zeros([ntraj, nstates], dtype=np.complex128)
    
    # eps = 0.20 # nonlinear coupling Vint = eps*x**2*y
    
    # Vcoup = Vint(a=1, Q)
    # X2 = M2mat(1, nstates)
    
    for k in range(ntraj):
        
        hk = H(y[k], args)
        
        # tmp = (H + Vp).dot(c[k,:]) - ddc[k,:]/2.0/amy - dc[k,:] * ry[k]/amy + Va * c[k,:]
        
        tmp = hk.dot(c[k])
        
        for d in range(ndim):
            tmp += (- ddc[k, :, d]/2. - dc[k,:, d] * ry[k,d])/mass[d] 

        dcdt[k,:] = -1j * tmp
       
    return dcdt
    
# # @numba.autojit 
# def xAve(c,y,w):
#     """
#     compute expectation value of x     
#     """
#     Xmat = M1mat(ax,M)

#     x_ave = 0.0+0.0j    
#     for k in range(Ntraj):
#         for m in range(M):
#             for n in range(M):
#                 x_ave += Xmat[m,n] * np.conjugate(c[k,m]) * c[k,n] * w[k]
    
#     return x_ave.real 
    





# update the coeffcients for each trajectory 
# fmt_c = ' {} '* (M+1)
  
# f = open('traj.dat','w')
# fe = open('en.out','w')
# fc = open('c.dat','w')
# fx = open('xAve.dat','w')
# fnorm = open('norm.dat', 'w')
# fden = open('den.dat','w')


from pyqed.phys import discretize

def J(omega, reorg=0, cutoff=1):
    return 2 * reorg * omega * cutoff/(omega**2 + cutoff**2) 

def ohmic(omega, alpha=1, cutoff=2):
    """
    

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    alpha : TYPE
        dimensionless Kondo parameter that characterizes the system–bath 
        coupling strength
    cutoff : float
        the characteristic frequency of the bath

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.pi/2 * alpha * omega * np.exp(-omega/cutoff)

if __name__=='__main__':
    
    from pyqed import pauli 
    from pyqed.units import * 
    
    s0, sx, sy, sz = pauli()
    
    # spin-boson model
    
    ntraj = 256
    nstates = M = 2
    ndim = 32
    
    # alpha = 0.4 # 1? is the critial point
    cutoff = 2
    omega, g = discretize(ohmic, a=0, b= 8 * cutoff, nmodes=ndim, mesh='log')
    mass = [1] * ndim
    print('mode frequencies = ', omega)
    print('coupling strength = ', g)
    
    print(ntraj, ndim)
    qt = NAQT(ntraj=ntraj, ndim=ndim, nstates=nstates)
    
    result = qt.run(dt=0.005, nt=400)
    result.plot_traj(d=0)
    result.plot_rdm()


#a, x0, De = 1.02, 1.4, 0.176/100 
#print('The well depth = {} cm-1. \n'.format(De * hartree_wavenumber))
#
#omega  = a * np.sqrt(2. * De / am )
#E0 = omega/2. - omega**2/16./De
#dE = (Etot-E0) * hartree_wavenumber 
#print('Exact ground-state energy = {} Hartree. \nEnergy deviation = {} cm-1. \n'.format(E0,dE))
#    


    
