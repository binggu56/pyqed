# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:42:22 2016

@author: Bing Gu

@Description: spin-boson model with partial hydrodynamics

"""
import numpy as np
import scipy
import numba
import sys
import math

from opt_einsum import contract

from pyqed import au2angstrom, au2wavenumber

bohr_angstrom = 0.52917721092
hartree_wavenumber = 219474.63

#hartree_wavenumber = scipy.constants.value(u'hartree-inverse meter relationship') / 1e2
class QT:
    """
    Quantum trajectory method with approximate quantum force
    """
    def __init__(self, ntraj, ndim, mass):
        
        self.ntraj = ntraj 
        self.ndim = ndim
        self.w = 1./ntraj
        self.mass = mass

        # self.p = np.zeros((ntraj, ndim))
        # self.x = np.zeros((ntraj, ndim))
        self.x = None
        self.p = None

    def sample(self, x0=None, p0=None):
        """
        Monte Carlo sampling of the quantum trajectories 

        Parameters
        ----------
        x0 : TYPE, optional
            DESCRIPTION. The default is None.
        p0 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        ndim = self.ndim 
        ntraj = self.ntraj 
        
        a = np.array([4.0, ] * ndim)
        
        if x0 is None:
            x0 = np.zeros(ndim)
        assert(len(x0) == ndim)        
        
        x = np.random.randn(ntraj, ndim)
        
        for j in range(ndim):
            x[:,j] = x[:,j] / np.sqrt(2.0 * a[j]) + x0[j]
        
        p = p0 * np.ones((ntraj, ndim))
        
        self.x = x 
        self.p = p

        return x, p 
            
    # def set_potential(self, V):
        
    #     self.potential = V
        
    def _force(self, x):
        """
        Compute the classical force exerting on the trajectories

        .. math::
            \mathbf{F}(x) = - \mathbm{\nabla} V(\mathbf{x}) 
            
        Returns
        -------
        None.

        """
        
        ntraj, ndim = self.ntraj, self.ndim 
        
        v = np.zeros(ntraj)
        force = np.zeros((ntraj, ndim))
        
        for n in range(ntraj):
            y = x[i, :]
            v[i], force[i, :] = potential(y)

        return v, force
    
    def quantum_force(self):
        pass
    
    def run(self, nt, dt, friction=0):
        
        ndim = self.ndim
        mass = self.mass
        
        dt2 = dt/2. 
        
        x, p = self.sample()
        
        # force
        v0, force = self._force(x)    
        r, du, Eu = LQF(x, w)


        t = 0
        for k in range(nt):

            t += dt
            
            p += (force - du) * dt2 - friction * p * dt2

            for j in range(ndim):
                x[:,j] +=  p[:,j]*dt/mass[j]

            # force field
            r, du, Eu = LQF(y,w)
            # x_ave = xAve(c,y,w)
            v0, force = self._force(x)

            p += (force - du) * dt2 - friction * p * dt2

            #  output data for each timestep
        #    d = c
        #    for k in range(Ntraj):
        #        for i in range(M):
        #            d[k,i] = np.exp(-1j*t*H[i,i])*c[k,i]


            fx.write('{} {} \n'.format(t,x_ave))

            f.write(fmt.format(t,*y[0:nout]))

            fnorm.write(' {} {} \n'.format(t,anm))

            # output density matrix elements
            rho = den(c,w)
            fden.write(' {} {} \n'.format(t,rho))

            Ek = np.dot(py*py,w)/2./amy
            Ev = np.dot(v0,w)
            Eu = Eu
            Etot = Ek + Ev + Eu

            fe.write('{} {} {} {} {} \n'.format(t,Ek,Ev,Eu,Etot))
            
            

class NAQT:
    """
    nonadiabatic dynamics with quantum trajectory method 
    
    valid for slow variables with internal space
    """
    def __init__(self, ntraj, ndim, nstates, mass=None):
        self.c = np.zeros((ntraj, nstates),dtype=np.complex128)
        self.ntraj = ntraj
        self.ndim = ndim
        self.nstates = nstates
        
        self.w = 1./ntraj # weights
        
        if mass is None:
            mass = [1, ] * ndim
        self.mass = mass

        self.p = np.zeros((ntraj, ndim))
        self.x = np.zeros((ntraj, ndim))

    def sample(self, a, x0=None, p0=None):
        """
        Monte Carlo sampling for a Gaussian distribution
        
        Warning: This is only valid for Gaussian.

        Parameters
        ----------
        a : array of ndim
            Gaussian width parameters
        x0 : TYPE, optional
            DESCRIPTION. The default is None.
        p0 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        ndim = self.ndim 
        ntraj = self.ntraj 
        
        # a = np.array([4.0, ] * ndim)
        
        if x0 is None:
            x0 = np.zeros(ndim)
        assert(len(x0) == ndim)
        
        if p0 is None:
            p0 = np.zeros(ndim)
        
        x = np.random.randn(ntraj, ndim)
        
        for j in range(ndim):
            x[:,j] = x[:,j] / np.sqrt(2.0 * a[j]) + x0[j]
        
        p = p0 * np.ones((ntraj, ndim))
        
        self.x = x 
        self.p = p

        return x, p 
            


    def quantum_force(self):
        pass

    def run(self, nt, dt, friction):

        ndim = self.ndim
        mass = self.mass

        dt2 = dt/2.

        c, x, p = self.initialize()

        # force
        v0, dv = pes(c, x)
        r, du, Eu = LQF(x, w)

        cold = c
        dcdt = prop_c(H,c,x,r,p)
        c = c + dcdt * dt

        t = 0
        for k in range(nt):

            t += dt

            p += (-dv - du) * dt2 - friction * p * dt2

            for j in range(ndim):
                x[:,j] +=  p[:,j]*dt/mass[j]

            # force field
            r, du, Eu = LQF(y,w)

            # x_ave = xAve(c,y,w)
            v0, dv = pes(x)

            p += (- dv - du) * dt2 - friction * p * dt2

            # renormalization

            # update c

            dcdt = prop_c(H,c,x,r,p)
            cnew = cold + dcdt * dt * 2.0
            cold = c
            c = cnew


            #  output data for each timestep
        #    d = c
        #    for k in range(Ntraj):
        #        for i in range(M):
        #            d[k,i] = np.exp(-1j*t*H[i,i])*c[k,i]


            fx.write('{} {} \n'.format(t,x_ave))

            f.write(fmt.format(t,*y[0:nout]))

            fnorm.write(' {} {} \n'.format(t,anm))

            # output density matrix elements
            rho = den(c,w)
            fden.write(' {} {} \n'.format(t,rho))

            Ek = np.dot(py*py,w)/2./amy
            Ev = np.dot(v0,w)
            Eu = Eu
            Etot = Ek + Ev + Eu

            fe.write('{} {} {} {} {} \n'.format(t,Ek,Ev,Eu,Etot))
        return


def M1mat(a, Nb):

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
    """
    Hermite polynormials up to order nb evaluated at x 

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    Nb : TYPE
        DESCRIPTION.

    Returns
    -------
    H : TYPE
        DESCRIPTION.

    """
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

def sigma_z():

    S = np.zeros((2,2))
    S[0,0] = 1.0
    S[1,1] = -1.0

    return S

def sigma_x():

    S = np.zeros((2,2))
    S[1,0] = 1.0
    S[0,1] = 1.0

    return S



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

@numba.jit
def Vint(g,Q):
    """
    g[Ndim] : coupling constants
    Q : bath configuration
    """

    Sz = sigma_z()

    tmp = 0.0
    for i in range(Ndim):
        tmp += g[i] * Q[i]

    Vsb = tmp * Sz

    return Vsb

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
def LQF(x, w):
    """
    linearized quantum force, exact for a Gaussian function

    Ref:
        Garashchuk, S. & Rassolov, V. A. Energy conserving approximations to 
        the quantum potential: Dynamics with linearized quantum force. 
        J. Chem. Phys. 120, 1181–1190 (2004).

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.
    du : TYPE
        DESCRIPTION.

    """
    ntraj, ndim = x.shape 
    
    # w = [1/ntraj, ] * ntraj
    
    b = np.zeros((ndim+1, ndim)) 
    b[:ndim, :ndim] = -0.5 * np.eye(ndim)
    
    S = np.zeros((Ndim+1,Ndim+1))

    for i in range(Ntraj):
        
        f = np.append(x[i, :], 1.0)
        
        # f[0:ndim] = x[i, :]
        # f[ndim] = 1.
        
        
        for m in range(Ndim+1):
            for n in range(m+1):

                S[m,n] += w[i] * f[m] * f[n]
                
    S = Sym(S)

    c = np.linalg.solve(S,b)

    #u = np.zeros(Ntraj)
    du = np.zeros((Ntraj,Ndim))
    r = np.zeros((Ntraj,Ndim))
    dr = np.zeros((Ntraj,Ndim))

    for i in range(Ntraj):

        # f = np.array([x[i,:],1.0])
        f = np.append(x[i, :], 1.0)

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

                du[i,j] -= (2.0 * r[i,k] * dr[j,k])/ (2.0 * amy[k])

    # quantum potential
    Q = 0
    return r,du, Q

# @numba.jit
def qpot(x,p,r,w):

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

    return Eu,fq,fr




def pes(c, x):
    # time-dependent potential energy surface
    pass



#print('Hamiltonian matrix in DOF x = \n')
#print(H)


@numba.jit
def den(c,w):
    """
        compute density matrix elements
    """
    rho = np.zeros((M,M),dtype=np.complex128)
    for k in range(Ntraj):
        rho[0,1] += c[k,0] * np.conjugate(c[k,1]) * w[k]

    return rho[0,1]

@numba.jit
def norm(c,w):

    anm = 0.0

    for k in range(Ntraj):
        anm += np.vdot(c[k,:], c[k,:]).real * w[k]
    return anm

# @numba.jit
# def fit_c(c,y):
#     """
#     global approximation of c vs y to obtain the derivative c'',c'
#     """
#     dc = np.zeros((Ntraj,M),dtype=np.complex128)
#     ddc = np.zeros((Ntraj,M),dtype=np.complex128)

#     for j in range(M):

#         z = c[:,j]
#         pars = np.polyfit(y,z,nfit)
#         p0 = np.poly1d(pars)
#         p1 = np.polyder(p0)
#         p2 = np.polyder(p1)
# #for k in range(Ntraj):
#         dc[:,j] = p1(y)
#         ddc[:,j] = p2(y)

#     return dc, ddc

def fit_c(c, x, deg=1):
    """
    global approximation of :math:`C_\alpha(R)` to obtain the derivative C'', C'    
    
    output fitting coefficients 
    
    params
    =======
    c: complex array (ntraj, nstates) 
        expansion coefficients
    
    x: array (ntraj, ndim)
        trajectories.
        
    deg: int 
        Default = 1 (linear fit). 
        
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
            nb = ndim + 1

            S = np.zeros((ndim + 1, ndim+1))
            b = np.zeros((ndim+1, nstates), dtype=complex)
            
            f = np.zeros((ntraj, nb))
            f[:,:-1] = x
            f[:, -1] = 1.0

            b = contract('i, ia, im->ma', w, c, f)
            S = contract('im, in, i -> mn', f, f, w)

            # for i in range(ntraj):
         
            #     f = np.array([*x[i,:],1.0]) # fitting basis
         
            #     for m in range(ndim+1):
                    
            #         for a in range(nstates):
            #             b[m, a] += w[i] * c[i, a] * f[m]
                    
            #         for n in range(m+1):
         
            #             S[m,n] += w[i] * f[m] * f[n]
            # S = Sym(S)
            
            coeff = np.linalg.solve(S, b)
            
        
            
            for a in range(nstates):
                for m in range(ndim):
                    dc[:, a, m] = coeff[m, a]
            
                return dc, ddc


        elif deg == 2:
            
            # number of basis sets 
            nb = 1 + ndim + ndim*(ndim +1)//2
            

            S = np.zeros((nb, nb))
            b = np.zeros((nb, nstates), dtype=complex)
            
            f = np.zeros((ntraj, nb))
            f[:,1:] = x
            f[:, 0] = 1.0
            
            n = ndim + 1
            for j in range(ndim):
                for k in range(j, ndim):
                    f[:, n] = x[:, j] * x[:, k]
                    n += 1
                    
            # f[:, ndim + 1:] = np.reshape(contract('ni, nj -> nij', x, x), ntraj, ndim**2)
            

            b = contract('i, ia, im -> ma', w, c, f)
            S = contract('im, in, i -> mn', f, f, w)
            
            coeff = np.linalg.solve(S, b)
            
            
            for a in range(nstates):
                a = coeff[0, a]
                b = coeff[1:ndim+1, a]
                c = np.reshape(coeff[ndim+1:, a], (ndim, ndim))
                
                for d in range(ndim):
                    dc[:, a, d] = b[d, a] + x[:, d]
                    ddc[:, a, d] = 2 * c[d, d]
            
                return dc, ddc
            raise NotImplementedError('Only linear basis has been implemented. Set deg=1.') 

        elif deg > 2:
            raise NotImplementedError('Only linear basis has been implemented. Set deg=1.')

    # error analysis
    # err = np.sum(c - x )
    


def prop_c(H,c,y,ry,py):
    """Propagate the coefficients

    Parameters
    ----------
    H : 2d array (nstates, nstates)
        Hamiltonian matrix
    c : 2d array (ntraj, nstates)
        coefficients
    y : 2d array (ntraj, ndim)
        coordinates
    ry : 2d array (ntraj, ndim)
        quantum force
    py : 2d array (ntraj, ndim)
        momentum
    x_ave : float
        average position

    Returns
    -------
    dcdt : 2d array (ntraj, nstates)
        time derivative of coefficients
    """

    dc, ddc = fit_c(c,y)

    dcdt = np.zeros([Ntraj,M],dtype=np.complex128)

    SzAve = AveSigma_z(c, w)
    Sz = sigma_z()

    for k in range(ntraj):

        Vp = sum(g * y[k]) * (Sz - SzAve * np.eye(nstates))

        # anharmonicity in the bath potential
        #Va = y[k]**4*0.1
       
        tmp = (H + Vp).dot(c[k,:]) - c[k] -  contract('aj,j->a', ddc[k,:,:], 0.5/mass) - contract('aj, j -> a', dc[k], ry[k]/mass) #+ Va * c[k,:]


        dcdt[k,:] = -1j * tmp

    return dcdt

def AveSigma_z(c,w):

    tmp = 0.0
    for k in range(Ntraj):
        tmp += abs(c[k,0])**2 - abs(c[k,1])**2 * w[k]

    return tmp

# @numba.jit
def xAve(c,y,w):
    """
    compute expectation value of x
    """
    Xmat = M1mat(ax,M)

    x_ave = 0.0+0.0j
    for k in range(Ntraj):
        for m in range(M):
            for n in range(M):
                x_ave += Xmat[m,n] * np.conjugate(c[k,m]) * c[k,n] * w[k]

    return x_ave.real

def Vy(y, SzAve):
    ntraj, ndim = y.shape
    v0 = np.zeros(ntraj)
    dv = np.zeros((ntraj, ndim))

    for n in range(ntraj):
        v0[n] = np.sum(y[n]**2 * omegas/2.0)
        dv[n] = omegas * y[n] + g * SzAve

    return v0, dv


# initialization
# for DOF y : an ensemble of trajectories
# for DOF x : for each trajectory associate a complex vector c of dimension M

Ntraj = ntraj = 512
nstates = M = 2 # number of states
nfit = 3 # degree of the polynomial for fitting the coefficients

ax = 1.0 # width of the Gaussian basis
x0 = 0.5 # initial position of the Gaussian basis   

# initial conditions for c
c = np.zeros((Ntraj, M), dtype=np.complex128)

# mixture of ground and first excited state

#c[:,0] = 1.0/np.sqrt(2.0)+0j
# c[:,1] = 1.0/np.sqrt(2.0)+0j
#for i in range(2,M):
#    c[:,i] = 0.0+0.0j

c[:,1] = 1.0 # initial state is the excited state

# coherent state
# z = 1.0/np.sqrt(2.0) * x0
# for i in range(M):
#     c[:,i] = np.exp(-0.5 * np.abs(z)**2) * z**i / np.sqrt(math.factorial(i))

print('initial occupation \n',c[0,:])
print('trace of density matrix',np.vdot(c[0,:], c[0,:]))
# ---------------------------------
# initial conditions for QTs
ndim = Ndim = 8
print('dimensionality of nuclei = {} \n'.format(ndim))

omegas = np.linspace(0.5, 2, ndim)

print('vibrational frequency', omegas)

y0 = np.array([0., ] * ndim)
ay0 = np.array([1.0, ] * ndim)

mass = amy = 1/omegas

assert(len(amy) == ndim)


y = np.random.randn(Ntraj, ndim)
for j in range(ndim):
    y[:,j] = y[:,j] / np.sqrt(2.0 * ay0[j]) + y0[j]

print('trajectory range {}, {}'.format(min(y[:, 0]),max(y[:, 0])))
py = np.zeros((Ntraj,Ndim))
ry = np.zeros((Ntraj,Ndim))


# If the initial wavefunction of y is a Gaussian wave packet, the quantum momentum is 
# a linear function of coordinates
for j in range(Ndim):
    ry[:,j] = - ay0[j] * (y[:,j] - y0[j])

# trajectory weigths 
w = np.array([1./ntraj] * ntraj)

# -------------------------------

amx = 1.0

f_MSE = open('rMSE.out','w')
nout = 20       # number of trajectories to print
fmt =  ' {}' * (nout+1)  + '\n'
Eu = 0.


fric_cons = 0.2      # friction constant


Nt = 400
dt = 0.004
dt2 = dt/2.0
t = 0.0

print('time range for propagation is [0,{}]'.format(Nt*dt))
print('timestep  = {}'.format(dt))

# construct the Hamiltonian matrix for anharmonic oscilator
g = np.array([0.05, ] * ndim)

if len(g) != ndim:
    raise ValueError('Error: coupling constants has wrong dimensionality.')
#V = 0.5 * M2mat(ax,M) + g/4.0 * M4mat(ax,M)
#K = Kmat(ax,0.0,M)
Delta = 1
H = Delta * sigma_x() / 2.0

# propagate the QTs for y
# update the coeffcients for each trajectory
fmt_c = ' {} '* (M+1)

f = open('traj.dat','w')
fe = open('en.out','w')
fc = open('c.dat','w')
fx = open('xAve.dat','w')
fnorm = open('norm.dat', 'w')
fden = open('den.dat','w')


v0, dv = Vy(y, -1)
ry, du, Eu = LQF(y,w)

print('classical and quantum potential energy', sum(v0 * w), sum(Eu * w))

cold = c
dcdt = prop_c(H,c,y,ry,py)
c = c + dcdt * dt

population = np.zeros(Nt)

for k in range(Nt):

    t = t + dt

    py += (- dv - du) * dt2 - fric_cons * py * dt2

    for j in range(Ndim):
        y[:,j] +=  py[:,j]*dt/amy[j]

    # force field
    ry, du, Eu = LQF(y,w)

    SzAve = AveSigma_z(c, w)
    v0, dv = Vy(y, SzAve)


    py += (- dv - du) * dt2 - fric_cons * py * dt2

    # renormalization

    anm = norm(c,w)
    c /= np.sqrt(anm)

    # update c

    dcdt = prop_c(H,c,y,ry,py)
    cnew = cold + dcdt * dt * 2.0
    cold = c
    c = cnew
    
    population[k] = np.sum(np.abs(c[:,1])**2 * w)

    #  output data for each timestep
#    d = c
#    for k in range(Ntraj):
#        for i in range(M):
#            d[k,i] = np.exp(-1j*t*H[i,i])*c[k,i]


    fx.write('{} {} \n'.format(t,SzAve))

    f.write(fmt.format(t,*y[0:nout]))

    # fnorm.write(' {} {} \n'.format(t,anm))

    # output density matrix elements
    rho = den(c,w)
    fden.write(' {} {} \n'.format(t,rho))

    Ek = contract('ij,ij,i, j->', py, py, w, 1/mass)/2.
    Ev = np.dot(v0,w)
    Eu = Eu
    Etot = Ek + Ev + Eu

    fe.write('{} {} {} {} {} \n'.format(t,Ek,Ev,Eu,Etot))


print('The total energy = {} Hartree. \n'.format(Etot))

# print trajectory and coefficients
for k in range(Ntraj):
    fc.write( '{} {} {} {} \n'.format(y[k], c[k,0],c[k,-2],c[k,-1]))

fe.close()
f.close()
fc.close()
fx.close()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(population)
plt.show()


#a, x0, De = 1.02, 1.4, 0.176/100
#print('The well depth = {} cm-1. \n'.format(De * hartree_wavenumber))
#
#omega  = a * np.sqrt(2. * De / am )
#E0 = omega/2. - omega**2/16./De
#dE = (Etot-E0) * hartree_wavenumber
#print('Exact ground-state energy = {} Hartree. \nEnergy deviation = {} cm-1. \n'.format(E0,dE))
#



