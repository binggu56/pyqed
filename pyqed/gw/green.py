#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:09:34 2023

@author: Bing Gu

The initial state has to be Gaussian such that the Wick's theorem can be invoked.

*----------------------------------------------------------------------------

   author: Naoto Tsuji <tsuji@cms.phys.s.u-tokyo.ac.jp>

           Department of Physics, University of Tokyo

   date:   February 28, 2013

----------------------------------------------------------------------------*/
#include "green.h"
#include "integral.h"
"""
import numpy as np
from numpy import exp, diag, conj, eye, sqrt, conj, cos, sin, trace
from numba import vectorize

from scipy.linalg import eigh

from pyqed import dag, propagator_H_const, propagator, interval

# from scipy.integrate import trapezoid

EXPMAX = 100

def scalar2array(a, d=1):
    """
    transform a scalar to one or two dimensional array

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    d : TYPE, optional
        DESCRIPTION. The default is 1.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if d == 1:
        return np.atleast_1d(a)
    elif d == 2:
        return np.reshape(a, (1,1))
    else:
        raise ValueError('A scalar cannot be transformed to {} dim array'.format(d))

def fermi_exp(beta, tau, omega):
     if (omega < 0):
         return exp(omega * tau) * fermi(beta, omega) # exp(w*t)/(1+exp(b*w)) always OK for w<0
     else:
         return exp((tau - beta) * omega) * fermi(beta, -omega) #  exp((t-b)*w)/(1+exp(-w*b))


@vectorize
def fermi(beta, omega):

    arg = omega * beta

    if abs(arg) > EXPMAX:
        return 0. if arg > 0.0 else 1.0
    else:
        return 1.0 / (1.0 + exp(arg))

@vectorize
def bose(beta, omega):
    # assert(omega > 0)
    arg = omega * beta
    # if (arg < 0):
    #      return (-1.0 - bose(beta, -omega))
    if arg < -EXPMAX:
        return -1.0
    elif -EXPMAX < arg < -1e-10:
        return -1.0 - 1./(exp(-arg) - 1.)
    elif -1e-10 < arg < 0:
        return -1.0 + 1.0/arg


    if (abs(arg) > EXPMAX):
         return 0.0
    elif (arg < 1e-10):
         return 1.0 / arg
    else:
         return 1.0 / (exp(arg) - 1.0)



 # # exp(tau*w)b(w) ... actually assuming positive tau
 # template <typename T>
def bose_exp(beta, tau, omega):
    if (omega < 0):
        return exp(tau * omega) * bose(beta, omega)
    else:
        return -exp((tau - beta) * omega) * bose(beta, -omega)


# class CntrFunc(self, ):
#     def __init__(self, dt=None, nt=None, beta=None, ntau=1, size=1, contour):
#         if contour == 'm':
#             self.matsubara = np.zeros((ntau+1, size, size), dtype=complex)


class NEGF:
    def __init__(self, nt, ntau=1, size=1, sign=-1, dt=None, beta=1e6):
        """
        The time domain used to represent the GF are

        Matsubara component C^M(iΔτ) for i=0,...,ntau,
        retarded component CR(iΔt,jΔt) for i=0,...,nt, j=0,...,i ,
        lesser component C<(iΔt,jΔt) for j=0,...,nt, i=0,...,j,
        left-mixing component C⌉(iΔt,jΔτ) for i=0,...,nt, j=0,...,ntau

        Parameters
        ----------
        nt : TYPE
            DESCRIPTION.
        ntau : TYPE, optional
            DESCRIPTION. The default is 1.
        size : TYPE, optional
            DESCRIPTION. The default is 1.
        sign : TYPE, optional
            DESCRIPTION. The default is -1.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.omega = omega
        self.nt = nt
        self.size = size
        # self.ptype = ptype
        self.sign = sign
        self.beta = beta
        self.dtau = beta/ntau
        self.ntau = ntau
        self.isherm = True
        self.dt = dt

        # self.contour = contour
        # if size > 1:

        self.retarded = np.zeros((nt+1, nt+1, size, size), dtype=complex)
        self.lesser = np.zeros((nt+1, nt+1, size, size), dtype=complex)
        self.left_mixing = np.zeros((nt+1, nt+1, size, size), dtype=complex)
        self.right_mixing = np.zeros((nt+1, nt+1, size, size), dtype=complex)
        
        # G^M satisfy the time-translation invariance
        self.matsubara = np.zeros((ntau+1, size, size), dtype=complex) 
        
        # elif size == 1:

        #     self.retarded = np.zeros((nt+1, nt+1), dtype=complex)
        #     self.lesser = np.zeros((nt+1, nt+1), dtype=complex)
        #     self.left_mixing = np.zeros((nt+1, nt+1), dtype=complex)
        #     self.right_mixing = np.zeros((nt+1, nt+1), dtype=complex)
        #     self.matsubara = np.zeros((ntau+1), dtype=complex)
        # else:
        #     raise ValueError('size {} should be positive integer'.format(size))

    # def retarded(self, dt, nt):
    #     pass

    # def advanced(self):
    #     pass

    # def lesser(self):
    #     pass

    def get_ret(self, n, m):
        return self.retarded[n, m]
    
    def get_mat(self, m):
        return self.matsubara[m]

    def get_gtr(self, n, m):
        """
        Using the equality 
        
        .. math::
            G^> - G^< =  G^R - G^A

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.
        m : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.get_les(n, m) + self.get_ret(n, m) - self.get_adv(n, m)

    def get_vt(self, m, j):
        return dag(self.get_tv(j, self.ntau - m))

    def get_tv(self, i, m):
        return self.left_mixing[i, m]

    def get_les(self, i, j):
        # if i > j:
        #     return self.lesser[i, j]
        # else:
        #     return - dag(self.lesser[j, i])
        return self.lesser[i, j]
    
    def get_adv(self, n, m):
        """
        .. math::

            G^a(z, z') = [G^r(z', z)]^\dag

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.
        m : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # if n > m:
        #     return 0
        # else:
        #     return dag(self.get_ret(m, n))
        return dag(self.get_ret(m, n))

    def set_ret(self, i, j, value):
        # if isinstance(value, (int, float, complex)):
        #     value = value * eye(self.size)

        # assert(value.shape == (self.size, self.size))
        assert(i >= j)

        self.retarded[i, j] = value
        return

    def set_les(self, i, j, value):
        # assert(value.shape == (self.size, self.size))
        self.lesser[i, j] = value
        return

    # def set_gtr(self, i, j, value):
    #     self.greater[i, j] = value

    def set_mat(self, m, value):
        # assert(value.shape == (self.size, self.size))
        assert(m <= self.ntau)

        self.matsubara[m] =  value

    def set_tv(self, n, m, value):
        self.left_mixing[n, m] = value

    # def set_vt(self, n, m, value):
    #     pass


    def hermitian_conjugate(self):
        """
        return the Hermitian conjugate of G, as defined by
        .. math::
            [C‡]R(t,t′)=(CA(t′,t))† ,
            [C‡]≷(t,t′)=−(C≷(t′,t))† ,
            [C‡]⌉(t,τ)=−ξ(C⌈(β−τ,t))† ,
            [C‡]M(τ)=(CM(τ))

        Returns
        -------
        None.

        """
        Gcc = NEGF(self.nt, self.ntau, self.size, self.sign)
        Gcc.matsubara = self.matsubara
        # Gcc.retarded = conj(np.transpose(G.advanced, (1032)))
        return Gcc

    def density_matrix(self, tstp):

        # assert(M.rows() == size1_ && M.cols() == size2_)
        assert(tstp >= -1 & tstp <= self.nt)

        if (tstp == -1):
            M = -1.0 * self.get_mat(ntau)
        else:
            M = 1j * self.sign * self.get_les(tstp, tstp)
        return M

    def print_to_file(self):
        pass

    def read_from_file(self):
        pass

    def __add__(self, B, x=1):
        """
        .. math::

            C(t, t') = A(t,t') + x B(t,t')

        Parameters
        ----------
        B : NEGF object
            DESCRIPTION.

        Returns
        -------
        C : TYPE
            DESCRIPTION.

        """


        C = NEGF(self.nt, self.ntau, self.size, self.sign)

        if B.isherm & self.isherm:
            C.isherm = True
        else:
            C.isherm = False

        C.retarded = self.retarded

        for i in range(nt+1):

            for j in range(i+1):
                C.set_ret(i, j, self.get_ret(i, j) + B.get_ret(i, j))

            for j in range(i, nt+1):
                C.set_les(i, j, self.get_les(i, j) + B.get_les(i, j))

        for m in range(ntau+1):
            for n in range(nt+1):
                C.set_tv(n, m, self.get_tv(n, m) + B.get_tv(n, m))

            C.set_mat(m, self.get_mat(m) + B.get_mat(m))

        return C
    
    def __matmul__(self, B):
        C = convolute(self, B)
        return C



def green_boson_XX_timestep(tstp, beta, ntau, w, dt):
    """
    Equilibrium Green's function for boson quadrature

    .. math::

        X = (a + a^\dag)/sqrt(2)
        D(t, t') = - i \braket{ \mathcal{T}_C X(t) X(t')}

    where t, t' are contour variables.

    Parameters
    ----------
    tstp : TYPE
        DESCRIPTION.
    ntau : TYPE
        DESCRIPTION.
    size1 : TYPE
        DESCRIPTION.
    ret : TYPE
        DESCRIPTION.
    tv : TYPE
        DESCRIPTION.
    les : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    """
    bw = bose(beta, w)

    # s2=size*size
     # std::complex<T> d_les, d_gtr, d_ret, d_tv;
     # std::complex<T> i = std::complex<T>(0.0, 1.0);

    dtau = beta/ntau
    assert(w*beta>0)

    nt = tstp

    D = NEGF(dt=dt, nt=nt, beta=beta, ntau=ntau)


    # # for(int tp=0;tp<=tstp;tp++):
    for tp in range(tstp+1):

        c = cos(w*(tstp-tp)* dt)
        s = sin(w*(tstp-tp)* dt)

        d_les = -1j *  ( (c-1j*s) + 2.0 * c * bw ) #   # D_les(tp, t)
        d_gtr = -1j *  ( (c-1j*s) + 2.0 * c * bw ) #  # D_gtr(t, tp)

        d_ret = d_gtr + conj(d_les)            #   # D_ret(t, tp)

        D.lesser[tp, tstp] = 0.5*d_les
        D.retarded[tstp, tp] = 0.5*d_ret

    # for l in range(nt+1):

    #     c = cos(w* l * dt)
    #     s = sin(w* l * dt)

    #     res = -0.5j *  ( (c-1j*s) + 2.0 * c * bw ) #   # D_les(tp, t)

    #     for i in range(nt+1):
    #         D.set_les(i-l, i, res)
    #         D.set_ret(i, i-l, res + conj(res))



    for v in range(ntau+1):
        c = cos(w*tstp*dt)
        s = sin(w*tstp*dt)
        ewv  = exp(w*v*dtau)
        emwv = exp(-w*v*dtau)
        d_tv = -0.5j * ( (c + 1j*s)*emwv + ((c+1j*s)*emwv + (c-1j*s)*ewv)*bw ) # ok
        D.set_tv(tstp, v, d_tv)


    for v in range(ntau+1):
        ewv = bose_exp(beta, v*dtau, w) # # exp(tau*w)b(w)
        emwv=-bose_exp(beta, v*dtau, -w) # # -exp(-tau*w)b(-w)
        d_mat = -0.5*(ewv+emwv)
        # mat[v*s2]=d_mat;
        D.set_mat(v, d_mat)

    return D

def green_boson_XX(nt, beta, ntau, w, dt):
    """
    Equilibrium Green's function for boson quadrature

    .. math::

        X = (a + a^\dag)/sqrt(2)
        D(t, t') = - i \braket{ \mathcal{T}_C X(t) X(t')}

    where t, t' are contour variables.

    Parameters
    ----------
    tstp : TYPE
        DESCRIPTION.
    ntau : TYPE
        DESCRIPTION.
    size1 : TYPE
        DESCRIPTION.
    ret : TYPE
        DESCRIPTION.
    tv : TYPE
        DESCRIPTION.
    les : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    """
    bw = bose(beta, w)
    print(bw)

    # s2=size*size
     # std::complex<T> d_les, d_gtr, d_ret, d_tv;
     # std::complex<T> i = std::complex<T>(0.0, 1.0);

    dtau = beta/ntau
    assert(w*beta>0)


    D = NEGF(dt=dt, nt=nt, beta=beta, ntau=ntau)


    # # for(int tp=0;tp<=tstp;tp++):
    for tstp in range(nt+1):
        for tp in range(tstp+1):

            c = cos(w*(tstp-tp)* dt)
            s = sin(w*(tstp-tp)* dt)

            d_les = -1j *  ( (c-1j*s) + 2.0 * c * bw ) #   # D_les(tp, t)
            d_gtr = -1j *  ( (c-1j*s) + 2.0 * c * bw ) #  # D_gtr(t, tp)

            d_ret = d_gtr + conj(d_les)            #   # D_ret(t, tp)

            D.set_les(tp, tstp, 0.5*d_les)
            D.set_ret(tstp, tp, 0.5*d_ret)

    # for l in range(nt+1):

    #     c = cos(w* l * dt)
    #     s = sin(w* l * dt)

    #     res = -0.5j *  ( (c-1j*s) + 2.0 * c * bw ) #   # D_les(tp, t)

    #     for i in range(nt+1):
    #         D.set_les(i-l, i, res)
    #         D.set_ret(i, i-l, res + conj(res))


    for tstp in range(nt+1):
        for v in range(ntau+1):

            c = cos(w*tstp*dt)
            s = sin(w*tstp*dt)
            ewv  = exp(w*v*dtau)
            emwv = exp(-w*v*dtau)
            d_tv = -0.5j * ( (c + 1j*s)*emwv + ((c+1j*s)*emwv + (c-1j*s)*ewv)*bw ) # ok
            D.set_tv(tstp, v, d_tv)


    for v in range(ntau+1):
        ewv = bose_exp(beta, v*dtau, w) # # exp(tau*w)b(w)
        emwv=-bose_exp(beta, v*dtau, -w) # # -exp(-tau*w)b(-w)
        d_mat = -0.5*(ewv+emwv)
        # mat[v*s2]=d_mat;
        D.set_mat(v, d_mat)

    return D

# def green_single_pole_XX_mat(int ntau,int size1,std::complex<T> *mat,T w,T beta) {
#      double ewv,emwv;
#      int s2=size1*size1;
#      std::complex<T> d_mat;
#      std::complex<T> i = std::complex<T>(0.0, 1.0);
#      double mat_axisunit = beta/ntau;
#      assert(w*beta>0);

#      for(int v=0;v<=ntau;v++) {
#          ewv=bose_exp(beta,v*mat_axisunit,w); # exp(tau*w)b(w)
#          emwv=-bose_exp(beta,v*mat_axisunit,-w); # -exp(-tau*w)b(-w)
#          d_mat = -0.5*(ewv+emwv);
#          mat[v*s2]=d_mat;
#      }
#  }

class DOS:
    # def __init__(self):
    #     self.beta =
    #     self.mu = mu
    #     self.low = None
    #     self.high = None
    def sample(self, n):
        return np.linspace(self.low, self.high, n)

class Bethe:
    def __init__(self):
        self.V_ = 1
        self.low = -2
        self.high = 2

    def dos(self, x):
        V_ = self.V_
        arg = 4.0 * V_ * V_ - x * x
        num = V_ * V_ * np.pi * 2
        return  0.0  if arg < 0 else  sqrt(arg) / num

class Ohmic(DOS):
    def __init__(self, omegac=1):
        self.high = 20 * omegac
        self.low = 0.01
        self.omegac = omegac

    def dos(self, x):
        omegac = self.omegac
        return x*x*exp(-x/omegac)/(2.0*omegac *omegac * omegac)

# class KB(GF):
#     def __init__(self, nt, size):
#         super().__init__(nt, size)
#         self.left_mixing = None
#         self.right_mixing = None

# def green_equilibrium_ret(G, dos, h, limit, nn, mu):
#    typedef std::complex<double> cplx;
#    nt=G.nt
#    size1=G.size
#    sign=G.sign
#    # double t;
#    # cplx res,err,
#    # cplx_i=cplx(0,1);
#    cplx_i = 1j
#    fourier::adft_func adft;
#    dos_wrapper<dos_function> dos1(dos,sign,mu);
#    dos1.mu_=mu;
#    dos1.x_=ret;
#    adft.sample(0.0,dos.lo_,dos.hi_,dos1,nn,limit);
#    for(l=0;l<=nt;l++){ # l = t-t'
#     t=h*l;
#     adft.dft(-t,res,err);
#     res *= std::complex<double>(0,-1.0);
#     for(i=l;i<=nt;i++) element_set<T,LARGESIZE>(size1,G.retptr(i,i-l),(std::complex<T>)(res))


 # template <typename T,class dos_function>
 # void green_equilibrium_mat(herm_matrix<T> &G,dos_function &dos,double beta,int limit,int nn,double mu)
 # {
 #   typedef std::complex<double> cplx;
 #   int ntau=G.ntau(),m,size1=G.size1();
 #   int sign=G.sig();
 #   double dtau;
 #   cplx res,err;
 #   fourier::adft_func adft;
 #   dos_wrapper<dos_function> dos1(dos,sign,mu);
 #   dos1.beta_=beta;
 #   dtau=beta/ntau;
 #   dos1.mu_=mu;
 #   dos1.x_=mat;
 #   for(m=0;m<=ntau;m++){
 #     dos1.tau_=m*dtau;
 #     adft.sample(0.0,dos.lo_,dos.hi_,dos1,nn,limit);
 #     adft.dft(0.0,res,err);
 #     element_set<T,LARGESIZE>(size1,G.matptr(m),(std::complex<T>)(res));
 #   }
 # }
 # template <typename T,class dos_function>
 # void green_equilibrium_tv(herm_matrix<T> &G,dos_function &dos,double beta,double h,int limit,int nn,double mu)
 # {
 #   typedef std::complex<double> cplx;
 #   int ntau=G.ntau(),nt=G.nt(),n,m,size1=G.size1();
 #   double dtau;
 #   int sign=G.sig();
 #   cplx res,err,cplx_i=cplx(0,1);
 #   fourier::adft_func adft;
 #   dos_wrapper<dos_function> dos1(dos,sign,mu);
 #   dos1.beta_=beta;
 #   dtau=beta/ntau;
 #   dos1.x_=tv;
 #   dos1.mu_=mu;
 #   for(m=0;m<=ntau;m++){
 #     dos1.tau_=m*dtau;
 #     adft.sample(0.0,dos.lo_,dos.hi_,dos1,nn,limit);
 #     for(n=0;n<=nt;n++){
 #       adft.dft(-n*h,res,err);
 #       res *= std::complex<double>(0,-1.0);
 #       element_set<T,LARGESIZE>(size1,G.tvptr(n,m),(std::complex<T>)(res));
 #     }
 #   }
 # }
 # template <typename T,class dos_function>
 # void green_equilibrium_les(herm_matrix<T> &G,dos_function &dos,double beta,double h,int limit,int nn,double mu)
 # {
 #   typedef std::complex<double> cplx;
 #   int nt=G.nt(),i,l,size1=G.size1();
 #   double t;
 #   int sign=G.sig();
 #   cplx res,err,cplx_i=cplx(0,1);
 #   fourier::adft_func adft
 #   dos_wrapper<dos_function> dos1(dos,sign,mu);
 #   dos1.mu_=mu;
 #   dos1.x_=les;
 #   dos1.beta_=beta;
 #   adft.sample(0.0,dos.lo_-0.0,dos.hi_-0.0,dos1,nn,limit);
 #   for(l=0;l<=nt;l++){ # l = t'-t
 #    t=h*l;
 #    adft.dft(t,res,err);
 #    res *= std::complex<double>(0,-1.0);
 #    for(i=l;i<=nt;i++) element_set<T,LARGESIZE>(size1,G.lesptr(i-l,i),(std::complex<T>)res);


def distribution_eq(beta, omega, sign): #distribution_eq(T beta,T omega,int sign)

    if(sign==-1):
        return fermi(beta,omega);
    elif(sign==1):
     return bose(beta,omega)


def distribution_exp_eq(beta, tau, omega, sign):
    if(sign==-1):
        return fermi_exp(beta,tau,omega)
    elif(sign==1):
        return bose_exp(beta,tau,omega)


 # template <typename T,class dos_function>
def green_single_mode(omega0, dt, nt, beta, ntau=0, sign=-1, mu=0):
    """
    Equilibrium Green's functions (time-translation invariant) for
    a single fermion orbital or boson mode

    .. math::
        H = \omega_0 c^\dag c
        g^R(t,t') = -i e^{-i \omega (t-t')}


    assume the chemical potential is included in epsilon

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    limit : TYPE
        DESCRIPTION.
    nn : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    
        # g^<(t,t') = -i \xi e^{-i \omega (t-t')} f_\xi(\omega - \mu)
        # g^M(\tau) = -e^{-\tau (\omega - \mu)} f_\xi(-(\omega - \mu))
        # g^\rceil(t, \tau) = -i \xi e^{-i \omega t} e^{\tau (\omega - \mu)} f_\xi(\omega - \mu)
        


    G = NEGF(dt=dt, nt=nt, ntau=ntau, beta=beta, sign=sign,  size=1)
    dtau = G.dtau

    f = distribution_eq(omega0 - mu, beta, sign) # distribution function

    for l in range(nt+1): ## l = t-t'
        t = dt * l
        # adft.dft(-t,res,err)
        res = -1j * exp(-1j * t * omega0)

        for i in range(l, nt+1):
            G.set_ret(i, i-l, res)

        les = -1j * sign * exp(-1j * t * omega0) * f
        for i in range(l, nt+1):
            # element_set<T,LARGESIZE>(size1,G.lesptr(i-l,i),(std::complex<T>)res);
            G.set_les(i-l, i, les)


    for m in range(ntau+1):
        tau = m*dtau
      # adft.sample(0.0,dos.lo_,dos.hi_,dos1,nn,limit);
        for n in range(nt+1):
            t = n * dt
        # adft.dft(-n*h,res,err);
            res = -1j * sign  * exp(-1j * omega0 * t) * \
                exp(tau * (omega0 - mu)) * f

            G.set_tv(n, m, res)
        # element_set<T,LARGESIZE>(size1,G.tvptr(n,m),(std::complex<T>)(res));


    # green_equilibrium_mat(G,dos,beta,limit,nn,mu);

    for m in range(ntau+1):
        tau = m*dtau
        res = - exp(- tau * (omega0 - mu)) * \
                         distribution_eq(beta, -(omega0-mu), sign)
        G.set_mat(m, res)

    return G

 # template <typename T,class dos_function>
def green_equilibrium(dos, beta, dt, nt, ntau, limit=256, mu=0, sign=-1):
    """
    Equilibrium Green's functions (time-translation invariant) for
    a given density of states

    .. math::
        G(t, t') = \int d\omega A(\omega) g_\omega(t,t')

        # g^R_\omega(t,t') = -i e^{-i \omega (t-t')}


    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    dos : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    limit : int
        discretize the frequency domain.
    nn : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
        # g^<_\omega(t,t') = -i \xi e^{-i \omega (t-t')} f_\xi(\omega - \mu)
        # g^M_\omega(\tau) = - e^{-\tau (\omega - \mu)} f_\xi(-(\omega - \mu))
        # g^\rceil_\omega(t, \tau) = -i \xi e^{-i \omega t} e^{\tau (\omega - \mu)}
        #     f_\xi(\omega - \mu)

    # TODO: the integration here can be improved by using FFT or matrix multiplication.

    G = NEGF(dt=dt, nt=nt, ntau=ntau, beta=beta, size=1)
    dtau = G.dtau

    # omega = np.linspace(dos.low, dos.high, limit)
    omega = dos.sample(limit)

    domega = interval(omega)
    A = dos.dos(omega) # density of states
    f = distribution_eq(omega - mu, beta, sign) # distribution function
    print(f)

    for l in range(nt+1): ## l = t-t'
        t = dt * l
        # adft.dft(-t,res,err)
        res = np.trapz(A * exp(-1j * t * omega)) * domega
        res *= -1j
        # for(i=l;i<=nt;i++):
        for i in range(nt+1):
            G.set_ret(i, i-l, res)

        les = -1j * sign * np.trapz(A * exp(-1j * t * omega) * f) * domega
        for i in range(l, nt+1):
            # element_set<T,LARGESIZE>(size1,G.lesptr(i-l,i),(std::complex<T>)res);
            G.set_les(i-l, i, les)


    for m in range(ntau+1):
        tau = m*dtau
      # adft.sample(0.0,dos.lo_,dos.hi_,dos1,nn,limit);
        for n in range(nt+1):
            t = n * dt
        # adft.dft(-n*h,res,err);
            res = -1j * sign * np.trapz(A * exp(-1j * omega * t) * \
                                        exp(tau * (omega - mu)) * f) * domega
            G.set_tv(n, m, res)
        # element_set<T,LARGESIZE>(size1,G.tvptr(n,m),(std::complex<T>)(res));



    # green_equilibrium_mat(G,dos,beta,limit,nn,mu);

    for m in range(ntau+1):
        tau = m*dtau
        res = - np.trapz(exp(- tau * (omega - mu)) * \
                         distribution_eq(beta, -(omega-mu), sign)) * domega
        G.set_mat(m, res)
    # green_equilibrium_ret(G,dos,h,limit,nn,mu);
    # green_equilibrium_tv(G,dos,beta,h,limit,nn,mu);
    # green_equilibrium_les(G,dos,beta,h,limit,nn,mu);

    return G




# def green_from_H(H0, beta, nt, ntau, h, mu=0):
#     """
#     Compute the Green's function from a time-dependent H
#     .. math::
#         G(z, z') = -i \braket{0|c(1) c^\dag(1') |0}

#     Parameters
#     ----------
#     G : TYPE
#         DESCRIPTION.
#     mu : TYPE
#         DESCRIPTION.
#     H : list
#         time-dependent H.
#     beta : TYPE
#         DESCRIPTION.
#     nt : TYPE
#         DESCRIPTION.
#     ntau : TYPE
#         DESCRIPTION.
#     h : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
# # std::complex<T> iu = std::complex<T>(0.0, 1.0);
# #    int nt=G.nt(),ntau=G.ntau();
# #    int size=G.size1();
# #    int sign=G.sig();
#    # double tau,t,
#     dtau=beta/ntau
#     size = H0.shape[-1]
#     # cdmatrix H0(size,size),H1(size,size);
#     eps.get_value(-1,H0);
#     H1 = mu*eye(size)-H0
#     # cntr::function<T> Ut(nt,size);
#     # cdmatrix evec0(size,size),value(size,size);
#     # dvector eval0(size),eval0m(size);
#     # Eigen::SelfAdjointEigenSolver<cdmatrix> eigensolver(H1);
#     # evec0=eigensolver.eigenvectors();
#     # eval0=eigensolver.eigenvalues();
#     # eval0m=(-1.0)*eval0;


#    for(int m=0;m<=ntau;m++){
#      tau=m*dtau;
#      if(sign==-1){
#        value=(-1.0)*evec0*fermi_exp(beta,tau,eval0).asDiagonal()*evec0.adjoint();
#      }else if(sign==1){
#        value=evec0*bose_exp(beta,tau,eval0).asDiagonal()*evec0.adjoint();
#      }
#      G.set_mat(m,value);
#    }

#    if(nt >= 0){
#      cdmatrix idm(size,size);
#      idm = MatrixXcd::Identity(size,size);
#      Ut.set_value(-1,idm);
#      Ut.set_value(0,idm);
#      for(int tstp=1;tstp<=nt;tstp++){
#        propagator_exp(tstp,Ut,eps,h,order,kt,fixHam);
#      }
#      for(int tstp=0;tstp<=nt;tstp++){
#        cdmatrix tmp;
#        Ut.get_value(tstp,tmp);
#        tmp = tmp * std::complex<T>(cos(mu * h * tstp),sin(mu * h * tstp));
#        Ut.set_value(tstp,tmp);
#      }

#      cdmatrix expp(size,size);
#      for(int m=0;m<=ntau;m++){
#        tau=m*dtau;
#        for(int n=0;n<=nt;n++){
#          Ut.get_value(n,expp);
#          if(sign==-1){
#            value=iu*expp*evec0*fermi_exp(beta,tau,eval0m).asDiagonal()*evec0.adjoint();
#          }else if(sign==1){
#            value=-iu*expp*evec0*bose_exp(beta,tau,eval0m).asDiagonal()*evec0.adjoint();
#          }
#          G.set_tv(n,m,value);
#        }
#      }
#      if(sign==-1){
#        value=evec0*fermi(beta,eval0m).asDiagonal()*evec0.adjoint();
#      }else if(sign==1){
#        value=-1.0*evec0*bose(beta,eval0m).asDiagonal()*evec0.adjoint();
#      }
#      cdmatrix exppt1(size,size);
#      cdmatrix exppt2(size,size);
#      for(int m=0;m<=nt;m++){
#        for(int n=0;n<=m;n++){
#            cdmatrix tmp(size,size);
#            Ut.get_value(m,exppt1);
#            Ut.get_value(n,exppt2);
#            tmp = -iu*exppt1*exppt2.adjoint();
#            G.set_ret(m,n,tmp);
#            tmp=iu*exppt2*value*exppt1.adjoint();
#            G.set_les(n,m,tmp);


# def propagator(tstp, U, H, dt, order, kt, fixHam=false){):
# # int nt=U.nt_;
# #    int size=U.size1_;
# #    cdmatrix prop(size,size);

#    # assert(tstp<=nt);
#    # assert(order==2 || order==4);
#    # assert(size==H.size1_);

#    if tstp==0:
#      # prop.setIdentity()
#      # U.set_value(tstp,prop)
#      U[0] = idm
#    else:
#        if(order==2){
#          cdmatrix arg(size,size);
#           # Get H(t+dt/2)-> Extrapolate and interpolate
#          interpolate_CF2(tstp,H,arg,kt,fixHam)
#          arg = -1j*arg*dt
#          # U.get_value(tstp-1,prop);
#          # prop=arg.exp()*prop;
#          # U.set_value(tstp,prop);
#          U[tstp] = exp(arg) @ U[tstp-1]

#      }else if(order==4){
#         cdmatrix H1(size,size),H2(size,size);
#         cdmatrix arg1(size,size),arg2(size,size);
#         # Get H(t+dt*c1) and H(t+dt*c2) -> Extrapolate and interpolate
#         interpolate_CF4(tstp,H,H1,H2,kt,fixHam);
#         double a1=(3.0-2.0*sqrt(3.0))/12.0;
#         double a2=(3.0+2.0*sqrt(3.0))/12.0;
#         arg1=std::complex<double>(0.0,-1.0)*dt*(a1*H1+a2*H2);
#         arg2=std::complex<double>(0.0,-1.0)*dt*(a2*H1+a1*H2);
#         U.get_value(tstp-1,prop);
#         prop=arg1.exp()*arg2.exp()*prop;
#         U.set_value(tstp,prop);
#      }
#    }
#  }

def green_from_H_const(H0, beta, nt, ntau, dt, sign=1, mu=0):
    # std::complex<T> iu = std::complex<T>(0.0, 1.0);
    # int nt=G.nt(),ntau=G.ntau();
    # int size=G.size1();
    # int sign=G.sig();
    # double tau,t,

    dtau = beta/ntau
    size = H0.shape[-1]

    idm = np.eye(size)
    Hmu = -H0 + mu * idm
    eval0, evec0 = eigh(Hmu)

    eval0m=(-1.0)*eval0


    # for m in range(ntau+1):

    #     tau=m*dtau

    #  # if(sign==-1){
    #     if sign == -1:

    #         value=(-1.0)*evec0*np.diag(fermi_exp(beta,tau,eval0)) * dag(evec0)

    #     elif sign == 1:

    #         value=(1.0)*evec0 * np.diag(bose_exp(beta,tau,eval0)) *dag(evec0)

    #     G.set_mat(m,value)


   # # if(nt >=0 ){
   #   # IHdt = -iu * h * H0;
   #  if nt >= 0:
   #      IHdt = iu * h * Hmu
   #      Udt = np.exp(IHdt)

   #   # cntr::function<T> Ut(nt,size);
   #   # cdmatrix Un(size,size);
   #   # Ut.set_value(-1,idm);

   #  # Ut.set_value(0,idm)
   #  U = idm
   #  Ut = [U.copy()]
   #   # for(int n=1;n<=nt;n++)
   #     # Ut.get_value(n-1,Un);
   #     # Un = Un @ Udt
   #     # Ut.set_value(n,Un);
   #  for n in range(nt+1):
   #      U = U @ Udt
   #      Ut.append(U.copy())

    Ut = propagator_H_const(-Hmu, dt, nt)
    # for i in range(nt):
    #     print(Ut[i][0,0])

     # cdmatrix expp(size,size);
     # for(int m=0;m<=ntau;m++){
     #   tau=m*dtau;
     #   for(int n=0;n<=nt;n++){
     #     Ut.get_value(n,expp);
     #     if(sign==-1){
     #       value=iu*expp*evec0*fermi_exp(beta,tau,eval0m).asDiagonal()*evec0.adjoint();
     #     }else if(sign==1){
     #       value=-iu*expp*evec0*bose_exp(beta,tau,eval0m).asDiagonal()*evec0.adjoint();
     #     }
     #     G.set_tv(n,m,value);
     #   }
     # }

    # f(H, beta) fermi/bose distribution function
    if sign==-1:
        value = evec0 @ diag(fermi(beta,eval0m)) @ dag(evec0)
    elif(sign==1):
        value = -1.0*evec0 @ diag(bose(beta,eval0m)) @ dag(evec0)

    print('value=', value)

    G = NEGF(dt=dt, nt=nt, ntau=ntau, size=size, beta=beta)


    for m in range(nt+1):
        for n in range(m+1):
        # for(int n=0;n<=m;n++){
           # cdmatrix tmp(size,size);
           # Ut.get_value(m,exppt1);
           # Ut.get_value(n,exppt2);
           exppt1 = Ut[m]
           exppt2 = Ut[n]

           tmp = -1j * exppt1 @ dag(exppt2)

           G.set_ret(m, n, tmp)

           tmp = 1j * exppt2 @ value @ dag(exppt1)
           G.set_les(n, m, tmp)
    return G

def green_from_H(H0, beta, nt, ntau, dt, sign=-1, mu=0):
    # std::complex<T> iu = std::complex<T>(0.0, 1.0);
    # int nt=G.nt(),ntau=G.ntau();
    # int size=G.size1();
    # int sign=G.sig();
    # double tau,t,
    # h = dt
    if isinstance(H0, np.ndarray):
        return green_from_H_const(H0, beta, nt, ntau, dt)

    dtau = beta/ntau
    size = H0[0].shape[0]

    idm = np.eye(size)
    Hmu = -H0 + mu * idm
    eval0, evec0 = eigh(Hmu)

    eval0m=(-1.0)*eval0


    # for m in range(ntau+1):

    #     tau=m*dtau

    #  # if(sign==-1){
    #     if sign == -1:

    #         value=(-1.0)*evec0*np.diag(fermi_exp(beta,tau,eval0)) * dag(evec0)

    #     elif sign == 1:

    #         value=(1.0)*evec0 * np.diag(bose_exp(beta,tau,eval0)) *dag(evec0)

    #     G.set_mat(m,value)


   # # if(nt >=0 ){
   #   # IHdt = -iu * h * H0;
   #  if nt >= 0:
   #      IHdt = iu * h * Hmu
   #      Udt = np.exp(IHdt)

   #   # cntr::function<T> Ut(nt,size);
   #   # cdmatrix Un(size,size);
   #   # Ut.set_value(-1,idm);

   #  # Ut.set_value(0,idm)
   #  U = idm
   #  Ut = [U.copy()]
   #   # for(int n=1;n<=nt;n++)
   #     # Ut.get_value(n-1,Un);
   #     # Un = Un @ Udt
   #     # Ut.set_value(n,Un);
   #  for n in range(nt+1):
   #      U = U @ Udt
   #      Ut.append(U.copy())

    Ut = propagator(-Hmu, dt, nt)

     # cdmatrix expp(size,size);
     # for(int m=0;m<=ntau;m++){
     #   tau=m*dtau;
     #   for(int n=0;n<=nt;n++){
     #     Ut.get_value(n,expp);
     #     if(sign==-1){
     #       value=iu*expp*evec0*fermi_exp(beta,tau,eval0m).asDiagonal()*evec0.adjoint();
     #     }else if(sign==1){
     #       value=-iu*expp*evec0*bose_exp(beta,tau,eval0m).asDiagonal()*evec0.adjoint();
     #     }
     #     G.set_tv(n,m,value);
     #   }
     # }

    # f(H, beta) fermi/bose distribution function
    if sign==-1:
        value = evec0 @ diag(fermi(beta,eval0m)) @ dag(evec0)
    elif(sign==1):
        value = -1.0*evec0 @ diag(bose(beta,eval0m)) @ dag(evec0)

    G = NEGF(nt=nt, ntau=ntau, size=size, beta=beta)

    for m in range(nt+1):
        for n in range(m+1):
        # for(int n=0;n<=m;n++){
           # cdmatrix tmp(size,size);
           # Ut.get_value(m,exppt1);
           # Ut.get_value(n,exppt2);
            exppt1 = Ut[m]
            exppt2 = Ut[n]

            tmp = -1j * exppt1 @ dag(exppt2)
            # print(tmp.shape)
            G.set_ret(m, n, tmp)

            tmp = 1j * exppt2 @ value @ dag(exppt1)
            G.set_les(n, m, tmp)
    return G


def hartree(G, D):
    """
    Hartree self-energy
    .. math::
        \Sigma_ij(t', t) = -i \int d t_2 \mu_{ij} \delta(t', t) D(t_1, t_2) Tr G(t_2, t_2^+)

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    D : complex
        photon GF.

    Returns
    -------
    None.

    """

def fock(G, D):
    pass


# #  BUBBLE 1 :  C(t,t') = ii * A(t,t') * B(t',t)
#  #  BUBBLE 2 :  C(t,t') = ii * A(t,t') * B(t,t')



#  # -------------  Auxiliary routines: -------------


def get_bubble_1_mat(sc, c1, c2, amat, sa, a1, a2, bmat, sb, b1, b2, sigb, ntau):
    """
    Matsubara
    .. math::

        C_{c1,c2}(tau) = - A_{a1,a2}(tau) * B_{b2,b1}(-tau)
                       = - A_{a1,a2}(tau) * B_{b2,b1}(beta-tau) (no cc needed !!)


    Parameters
    ----------
    sc : TYPE
        DESCRIPTION.
    c1 : TYPE
        DESCRIPTION.
    c2 : TYPE
        DESCRIPTION.
    amat : TYPE
        DESCRIPTION.
    sa : TYPE
        DESCRIPTION.
    a1 : TYPE
        DESCRIPTION.
    a2 : TYPE
        DESCRIPTION.
    bmat : TYPE
        DESCRIPTION.
    sb : TYPE
        DESCRIPTION.
    b1 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.
    sigb : TYPE
        DESCRIPTION.
    ntau : TYPE
        DESCRIPTION.

    Returns
    -------
    cmat : TYPE
        DESCRIPTION.

    """
    cmat = [0, ] * (ntau+1)

    a12 = a1 * sa + a2
    sa2 = sa * sa
    b21 = b2 * sb + b1
    sb2 = sb * sb
    c12 = c1 * sc + c2
    sc2 = sc * sc
    sig = -1.0 * sigb
    for m in range(ntau+1):
        cmat[c12 + m * sc2] = sig * amat[m * sa2 + a12] @ bmat[(ntau - m) * sb2 + b21]

    return cmat

def get_bubble_1_timestep(tstp, cret, ctv, cles, sc, c1, c2, aret, atv, ales, accret,
                          acctv, accles, sa, a1, a2, bret, btv, bles, bccret, bcctv,
                          bccles, sb, b1, b2, sigb, ntau):

    # compute C[i<=tstp, tstp]
    a12 = a1 * sa + a2
    a21 = a2 * sa + a1
    sa2 = sa * sa
    b12 = b1 * sb + b2
    b21 = b2 * sb + b1
    sb2 = sb * sb
    c12 = c1 * sc + c2
    sc2 = sc * sc
    msigb = -1.0 * sigb
    ii = 1j
    # std::complex<T> bgtr21_tt1, cgtr_tt1, cles_tt1, ales12_tt1, bgtr21_t1t, agtr12_tt1,
        # bles21_t1t, bvt21, bles21_tt1;
    # for (m = 0; m <= tstp; m++) {
    for m in range(tstp + 1):
        # Bles_{21}(tstp,m) = - Bccles_{12}(m,tstp)^*
        bles21_tt1 = -conj(bccles[m * sb2 + b12])
        # Bgtr_{21}(tstp,m) = Bret_{21}(tstp,m) - bles21_tt1;
        bgtr21_tt1 = bret[m * sb2 + b21] + bles21_tt1
        # bgtr21_t1t = - Bccgtr_{12}(t,t1)^*
        #            = - [ Bccret_{12}(t,t1) + Bccles_{12}(t,t1)  ]^*
        #            = - Bccret_{12}(t,t1)^* + Bles_{21}(t1,t)
        bles21_t1t = bles[m * sb2 + b21];
        bgtr21_t1t = -conj(bccret[m * sb2 + b12]) + bles21_t1t
        # Ales_{12}(tstp,m) = -Accles_{21}(m,tstp)^*
        ales12_tt1 = -conj(accles[sa2 * m + a21])
        # Agtr_{a1,a2}(tstp,m) = Aret_{a1,a2}(tstp,m) - Ales_{a1,a2}(tstp,m)
        agtr12_tt1 = aret[m * sa2 + a12] + ales12_tt1
        # Cgtr_{12}(tstp,m) = ii * Agtr_{12}(tstp,m)*Bles_{21}(m,tstp)
        cgtr_tt1 = ii * agtr12_tt1 * bles21_t1t
        # Cles_{12}(tstp,m) = ii * Ales_{12}(tstp,m)*Bgtr_{21}(m,tstp)
        cles_tt1 = ii * ales12_tt1 * bgtr21_t1t
        # Cret_{12}(tstp,m) = Cgtr_{12}(tstp,m) - Cles_{12}(tstp,m)
        cret[m * sc2 + c12] = cgtr_tt1 - cles_tt1
        # Cles_{12}(m,tstp) = ii * Ales_{12}(m,tstp)*Bgtr_{21}(tstp,m)
        cles[m * sc2 + c12] = ii * ales[m * sa2 + a12] * bgtr21_tt1

     # for (m = 0; m <= ntau; m++):
    for m in range(ntau + 1):
        bvt21 = msigb * conj(bcctv[(ntau - m) * sb2 + b12])
        ctv[m * sc2 + c12] = ii * atv[m * sa2 + a12] * bvt21

    return cret, cles, ctv

 # #  BUBBLE 2 :
 # #  C(t,t') = ii * A(t,t') * B(t,t')

 # # Matsubara:
 # # C_{c1,c2}(tau) = - A_{a1,a2}(tau) * B_{b1,b2}(tau)
 # template <typename T>
 # void get_bubble_2_mat(std::complex<T> *cmat, int sc, int c1, int c2, std::complex<T> *amat,
 #                       int sa, int a1, int a2, std::complex<T> *bmat, int sb, int b1, int b2,
 #                       int ntau) {
 #     int m;
 #     int a12 = a1 * sa + a2, sa2 = sa * sa;
 #     int b12 = b1 * sb + b2, sb2 = sb * sb;
 #     int c12 = c1 * sc + c2, sc2 = sc * sc;
 #     for (m = 0; m <= ntau; m++)
 #         cmat[c12 + m * sc2] = -amat[m * sa2 + a12] * bmat[m * sb2 + b12];
 # }
 # template <typename T>
 # void
 # get_bubble_2_timestep(int tstp, std::complex<T> *cret, std::complex<T> *ctv,
 #                       std::complex<T> *cles, int sc, int c1, int c2, std::complex<T> *aret,
 #                       std::complex<T> *atv, std::complex<T> *ales, std::complex<T> *accret,
 #                       std::complex<T> *acctv, std::complex<T> *accles, int sa, int a1,
 #                       int a2, std::complex<T> *bret, std::complex<T> *btv,
 #                       std::complex<T> *bles, std::complex<T> *bccret, std::complex<T> *bcctv,
 #                       std::complex<T> *bccles, int sb, int b1, int b2, int ntau) {
 #     int m;
 #     int a12 = a1 * sa + a2, a21 = a2 * sa + a1, sa2 = sa * sa;
 #     int b12 = b1 * sb + b2, b21 = b2 * sb + b1, sb2 = sb * sb;
 #     int c12 = c1 * sc + c2, sc2 = sc * sc;
 #     std::complex<T> ii = std::complex<T>(0, 1.0);
 #     std::complex<T> bgtr12_tt1, agtr12_tt1, cgtr_tt1, cles_tt1;
 #     for (m = 0; m <= tstp; m++) {
 #         bgtr12_tt1 = bret[m * sb2 + b12] - conj(bccles[m * sb2 + b21]);
 #         agtr12_tt1 = aret[m * sa2 + a12] - conj(accles[m * sa2 + a21]);
 #         # Cgtr_{12}(tstp,m) = ii * Agtr_{12}(tstp,m)*Bles_{21}(m,tstp)
 #         cgtr_tt1 = ii * agtr12_tt1 * bgtr12_tt1;
 #         # Cles_{12}(tstp,m) = ii * Ales_{12}(tstp,m)*Bles_{12}(tstp,m)
 #         cles_tt1 = ii * conj(accles[m * sa2 + a21]) * conj(bccles[m * sb2 + b21]);
 #         # Cret_{12}(tstp,m) = Cgtr_{12}(tstp,m) - Cles_{12}(tstp,m)
 #         cret[m * sc2 + c12] = cgtr_tt1 - cles_tt1;
 #         # Cles_{12}(m,tstp) = ii * Ales_{12}(m,tstp)*Bgtr_{12}(m,tstp)
 #         cles[m * sc2 + c12] = ii * ales[m * sa2 + a12] * bles[m * sb2 + b12];
 #     }
 #     for (m = 0; m <= ntau; m++) {
 #         ctv[m * sc2 + c12] = ii * atv[m * sa2 + a12] * btv[m * sb2 + b12];
 #     }
 # }


 # # A,B hermitian, orbital dimension > 1:
 # # C_{c1,c2}(t1,t2) = ii * A_{a1,a2}(t1,t2) * B_{b2,b1}(t2,t1)


def bubble(A, B, vertex):
    """
    bubble diagram consisting of A ->, and B <-
    .. math::

        C_{c1, c2}(t1,t2) = -i  V^{c1}_{b1, a1) A_{a1,a2}(t1,t2)  B_{b2,b1}(t2,t1) V^{c2}_{b2, a2}

    TODO: multimodes. Right now only a single mode is supported.

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    vertex : ndarray c, b (outgoing), a (incoming)
        v^c_{ba} = v^c_{ab}^*.

    Returns
    -------
    C : TYPE
        DESCRIPTION.

    """
    nt, ntau, size = A.nt, A.ntau, A.size

    assert(nt == B.nt)
    assert(ntau == B.ntau)
    assert(size == B.size)

    # vertex is a 2D array (boson index is suppressed since 
    # we only have a single mode
    
    assert(vertex.ndim == 2) 

    c = NEGF(nt=nt, ntau=ntau, size=1)

    v = vertex
    
    for i in range(nt+1):

        for j in range(i+1):
            # retarded
            # C^r(t,t') = A^r(t, t') B^<(t', t) + A^<(t,t') B^a(t', t)

            ales = A.get_les(i, j)
            aret = A.get_ret(i, j)

            badv = B.get_adv(j, i)
            bles = B.get_les(j, i)

            print('ar, bles', aret, bles, ales, badv)


            l1 = v @ aret # b1 a2
            r1 = v.T @ bles # a2 b1
            value1 = trace(l1 @ r1)
            print('value1', value1)

            # l2 = vertex @ A.get_les(i, j) # b1 a2
            # r2 = conj(vertex) @ B.get_adv(j, i) # a2 b1

            value2 = trace(v @ ales @ (v.T @ badv))
            print('value 2', value2)

            c.set_ret(i, j, -1j * (value1 + value2))

        # lesser
        # C^< = a^< b^>
        for j in range(i+1, nt+1):

            ales = A.get_les(i, j)
            bgtr = B.get_gtr(j, i)

            c.set_les(i, j, -1j * trace(v @ ales @ conj(v) @ bgtr))


    # left-mixing
    # C^{tv} = A^{tv} B^{vt}

    for i in range(nt+1):
        for n in range(ntau+1):

            atv = A.get_tv(i, n)
            bvt = B.get_vt(n, i)

            c.set_tv(i, j, -1j * trace(v @ atv @ conj(v) @ bvt))


    # elif vertex.ndim == 3:

    #     sc = vertex.shape[0]
    #     C = NEGF(nt, ntau, sc)

    #     l = vertex @ A.get_ret(i, j) # c1 b1 a2
    #     r = conj(vertex) @ B.get_les(j, i) # c2 a2 b1

    #     value = np.einsum('cba, dab ->cd', l, r)
    #     C.set_ret(i, j, value)

    # else:
    #     raise ValueError('Vertex dim {} can only be 2 or 3'.format(vertex.ndim))


    return c

def Bubble1(tstp, C, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2):
     ntau = C.ntau
     # assert(ntau == A.ntau());
     # assert(ntau == B.ntau());
     # assert(ntau == Acc.ntau());
     # assert(ntau == Bcc.ntau());
     # assert(tstp == C.tstp_);
     # assert(tstp == B.tstp_);
     # assert(tstp == A.tstp_);
     # assert(tstp == Acc.tstp_);
     # assert(tstp == Bcc.tstp_);
     # assert(a1 <= A.size1());
     # assert(a2 <= A.size1());
     # assert(b1 <= B.size1());
     # assert(b2 <= B.size1());
     # assert(c1 <= C.size1());
     # assert(c2 <= C.size1());
     # assert(Acc.size1() ==  A.size1());
     # assert(Bcc.size1() == B.size1());

     if tstp == -1:
         get_bubble_1_mat(C.matptr(0), C.size1(), c1, c2, A.matptr(0), A.size1(), a1, a2,
                          B.matptr(0), B.size1(), b1, b2, B.sig(), ntau)
     else:
         get_bubble_1_timestep(tstp, C.retptr(0), C.tvptr(0), C.lesptr(0), C.size1(), c1, c2,
                               A.retptr(0), A.tvptr(0), A.lesptr(0), Acc.retptr(0),
                               Acc.tvptr(0), Acc.lesptr(0), A.size1(), a1, a2, B.retptr(0),
                               B.tvptr(0), B.lesptr(0), Bcc.retptr(0), Bcc.tvptr(0),
                               Bcc.lesptr(0), B.size1(), b1, b2, B.sig(), ntau)


# def Bubble1(int tstp, herm_matrix_timestep_view<T> &C, int c1, int c2,
#               herm_matrix_timestep_view<T> &A, int a1, int a2,
#               herm_matrix_timestep_view<T> &B, int b1, int b2)
#      return Bubble1(tstp, C, c1, c2, A, A, a1, a2, B, B, b1, b2)
 # template <typename T>
 # void Bubble1(int tstp, herm_matrix_timestep_view<T> &C, herm_matrix_timestep_view<T> &A,
 #              herm_matrix_timestep_view<T> &Acc, herm_matrix_timestep_view<T> &B,
 #              herm_matrix_timestep_view<T> &Bcc) {
 #     return Bubble1(tstp, C, 0, 0, A, Acc, 0, 0, B, Bcc, 0, 0);
 # }
 # template <typename T>
 # void Bubble1(int tstp, herm_matrix_timestep_view<T> &C, herm_matrix_timestep_view<T> &A,
 #              herm_matrix_timestep_view<T> &B) {
 #     return Bubble1(tstp, C, A, A, B, B);
 # }
 # template <class GGC, class GGA, class GGB>
 # void Bubble1(int tstp, GGC &C, int c1, int c2, GGA &A, GGA &Acc, int a1, int a2, GGB &B,
 #              GGB &Bcc, int b1, int b2) {
 #     herm_matrix_timestep_view<typename GGC::scalar_type> ctmp(tstp, C);
 #     herm_matrix_timestep_view<typename GGA::scalar_type> atmp(tstp, A);
 #     herm_matrix_timestep_view<typename GGA::scalar_type> acctmp(tstp, Acc);
 #     herm_matrix_timestep_view<typename GGB::scalar_type> btmp(tstp, B);
 #     herm_matrix_timestep_view<typename GGB::scalar_type> bcctmp(tstp, Bcc);
 #     Bubble1(tstp, ctmp, c1, c2, atmp, acctmp, a1, a2, btmp, bcctmp, b1, b2);
 # }
 # template <class GGC, class GGA, class GGB>
 # void Bubble1(int tstp, GGC &C, int c1, int c2, GGA &A, int a1, int a2, GGB &B, int b1,
 #              int b2) {
 #     herm_matrix_timestep_view<typename GGC::scalar_type> ctmp(tstp, C);
 #     herm_matrix_timestep_view<typename GGA::scalar_type> atmp(tstp, A);
 #     herm_matrix_timestep_view<typename GGB::scalar_type> btmp(tstp, B);
 #     Bubble1(tstp, ctmp, c1, c2, atmp, a1, a2, btmp, b1, b2);
 # }
 # template <class GGC, class GGA, class GGB>
 # void Bubble1(int tstp, GGC &C, GGA &A, GGA &Acc, GGB &B, GGB &Bcc) {
 #     herm_matrix_timestep_view<typename GGC::scalar_type> ctmp(tstp, C);
 #     herm_matrix_timestep_view<typename GGA::scalar_type> atmp(tstp, A);
 #     herm_matrix_timestep_view<typename GGA::scalar_type> acctmp(tstp, Acc);
 #     herm_matrix_timestep_view<typename GGB::scalar_type> btmp(tstp, B);
 #     herm_matrix_timestep_view<typename GGB::scalar_type> bcctmp(tstp, Bcc);
 #     Bubble1(tstp, ctmp, atmp, acctmp, btmp, bcctmp);
 # }
 # template <class GGC, class GGA, class GGB> void Bubble1(int tstp, GGC &C, GGA &A, GGB &B) {
 #     herm_matrix_timestep_view<typename GGC::scalar_type> ctmp(tstp, C);
 #     herm_matrix_timestep_view<typename GGA::scalar_type> atmp(tstp, A);
 #     herm_matrix_timestep_view<typename GGB::scalar_type> btmp(tstp, B);
 #     Bubble1(tstp, ctmp, atmp, btmp);
 # }



def self_energy(G, D, diagram='hartree'):
    """
    electron-photon/phonon coupling self-energy

    The second order contains the Hartree and Fock diagrams.

    Fock term
    .. math::
        \Sigma(t,t') = i G(t,t') D(t', t)

        \Sigma^\text{R}(t,t') = i \del{ G^\text{r}(t,t') D^<(t',t) + \
                                        G^\text{<}(t,t') D^\text{a}(t',t) }

    The fourth-order contains the second-Born diagrams.

    For self-consistancy, use dressed G and D.


    Parameters
    ----------
    G : TYPE
        electron GF.
    D : TYPE
        boson GF.
        .. math::
            D(1, 2) = -i \braket{\phi(1) \phi(2)}

    order : TYPE, optional
        DESCRIPTION. The default is 2.


        Fock
        .. math::
            \Sigma(1', 1) = D(1, 1')G(1', 1)

    Returns
    -------
    None.

    """
    if diagram == 'hartree':
        return hartree(G, D)

    elif diagram == 'fock':
        pass
    elif diagram == '2B':
        pass
    elif diagram == 'bubble':
        pass
    elif diagram == 'GW':
        pass
    else:
        raise NotImplementedError('Diagram {} has not been implemented.'.format(diagram))


def trapezoid(f):
    """
  #----------------------------------------------------------------------------
  #  This function calculates the sum
  #
  #    \sum_{k=i}^{j} w_{k}^{i,j} f_{k}.
  #
  #    w_{k}^{i,j} = 1/2  for k=i,j
  #                = 1    for i<k<j
  #----------------------------------------------------------------------------
    """
    # if j == -1:
    #     j = len(f) - 1
    
    # if j == i:
    #   return 0.0

    # else:
    #   integral = 0
    #   integral += 0.5*f[i]
    #   for k in range(i+1, j, 1):
    #       integral+=f[k]
    #   integral += 0.5*f[j]
    
    I = 0.5 * (f[0] + f[-1])
    for k in range(1, len(f)-1):
        I += f[k]
        
    return I

def trapezoid_half_edge(f, i, j):
  # #----------------------------------------------------------------------------
  # #  This function calculates the sum
  # #
  # #    \sum_{k=i}^{j} w_{k}^{i,j} f_{k}.
  # #
  # #    w_{k}^{i,j} = 1/2  for k=i
  # #                = 1    for i<k<=j
  # #----------------------------------------------------------------------------
    integral = 0

    integral+=0.5*f[i];
    for k in range(i+1, j+1, 1): integral+=f[k]
    return integral


# def KB_derivative::update(parm parm_, int n, KB_derivative &G_der)
# {
#   # d/dt G_old <= d/dt G_new
#   for (int i=0;i<=n;i++){
#     retarded[i]=G_der.retarded[i];
#     lesser[i]=G_der.lesser[i];
#   }
#   for (int j=0;j<=parm_.N_tau;j++){
#     left_mixing[j]=G_der.left_mixing[j];
#   }
# }




def convolute_timestep(n, A, B):
    """
    /*----------------------------------------------------------------------------
    This subroutine calculates a convolution of two Kadanoff-Baym function A
    and B,
                C(t,t')=(A*B)(t,t')

    for t=n*dt or t'=n*dt. Integrals are evaluated by the trapezoidal rule.
    ----------------------------------------------------------------------------*/

    retarded component
    .. math::
    C^{R}(t,t')=\int_{t'}^{t} ds A^{R}(t,s)*B^{R}(s,t')


    Ref.
    1. Aoki, H. et al. Nonequilibrium dynamical mean-field theory and its applications.
        Rev. Mod. Phys. 86, 779–837 (2014).


    """
    # vector<complex<double> > AxB(max(parm_.N_tau+1,n+1));

    nt, ntau, size = A.nt, A.ntau, A.size
    dtau = A.dtau
    dt = A.dt

    assert(nt == B.nt)
    assert(ntau == B.ntau)
    assert(size == B.size)

    C = NEGF(nt=nt, ntau=ntau, size=size, dt=dt)


    # retarded
    for j in range(n+1):

        # C.retarded[n, j]=0.0

        AxB = [A.retarded[n][k] @ B.retarded[k][j] for k in range(j, n+1)]


        C.retarded[n, j] += dt * trapezoid(AxB) #trapezoid(AxB, j,n)


  # left-mixing component
  # C^{Left}(t,tau')=\int_0^{beta} dtau A^{Left}(t,tau)*B^{M}(tau,tau')
  #                  +\int_0^{t} ds A^{R}(t,s)*B^{Left}(s,tau')
    for j in range(ntau+1):

        AxB = [A.left_mixing[n][k] @ B.matsubara[ntau+k-j] for k in range(j+1)]
        C.left_mixing[n][j] += - dtau * trapezoid(AxB)

        AxB = [A.left_mixing[n][k] @ B.matsubara[k-j] for k in range(j, ntau+1)]
        C.left_mixing[n][j] += dtau*  trapezoid(AxB) # trapezoid(AxB,j,ntau)

        AxB = [A.retarded[n][k] @ B.left_mixing[k][j] for k in range(n+1)]
        C.left_mixing[n][j] += dt*trapezoid(AxB)

  #   left_mixing[n][j]=0.0;
  #   for (int k=0;k<=j;k++) AxB[k]=A.left_mixing[n][k]*B.matsubara_t[parm_.N_tau+k-j];

  #   left_mixing[n][j]+=-parm_.dtau*trapezoid(AxB,0,j);
  #   for (int k=j;k<=parm_.N_tau;k++) AxB[k]=A.left_mixing[n][k]*B.matsubara_t[k-j];
  #   left_mixing[n][j]+=parm_.dtau*trapezoid(AxB,j,parm_.N_tau);
  #   for (int k=0;k<=n;k++) AxB[k]=A.retarded[n][k]*B.left_mixing[k][j];
  #   left_mixing[n][j]+=parm_.dt*trapezoid(AxB,0,n)

  # lesser component
  # # C^{<}(t,t')=-i\int_0^{beta} dtau A^{Left}(t,tau)*B^{Right}(tau,t')
  # #             +\int_0^{t'} ds A^{<}(t,s)*B^{A}(s,t')
  # #             +\int_0^{t} ds A^{R}(t,s)*B^{<}(s,t')
  #/ (i,j)=(n,j)
    for j in range(n):

        # C^< = A^r \cdot B^< + B^< \cdot C^a + B^{\rceil} \star C^{\lceil}

        # for (int k=0;k<=parm_.N_tau;k++) AxB[k]=A.left_mixing[n][k]*conj(B.left_mixing[j][parm_.N_tau-k]);

        AxB = [A.left_mixing[n][k] @ dag(B.left_mixing[j][ntau-k]) \
               for k in range(ntau+1)]

        # lesser[n][j]+=-xj*parm_.dtau*trapezoid(AxB,0,parm_.N_tau)
        C.lesser[n][j] += -1j * dtau * trapezoid(AxB)

        # for k in range(j):
            # AxB[k] = A.lesser[n, k] * conj(B.retarded[j, k])
        AxB = [A.lesser[n, k] @ dag(B.retarded[j, k]) for k in range(j+1)]
        C.lesser[n, j] += dt * trapezoid(AxB)

        AxB = [A.retarded[n, k] @ B.lesser[k, j] for k in range(0, n+1)]
        C.lesser[n, j] += dt*trapezoid(AxB)

  # (i,j)=(i,n)
    for i in range(n+1):

        # for (int k=0;k<=parm_.N_tau;k++) AxB[k]=A.left_mixing[i][k]*conj(B.left_mixing[n][parm_.N_tau-k]);
        AxB = [A.left_mixing[i][k] @ dag(B.left_mixing[n][ntau-k]) \
                                         for k in range(ntau+1)]

        C.lesser[i][n] += -1j* dtau * trapezoid(AxB)

        AxB = [A.lesser[i, k] @ dag(B.retarded[n, k]) for k in range(n+1)]
        C.lesser[i, n] += dt * trapezoid(AxB)


        AxB = [A.retarded[i, k] @ B.lesser[k, n] for k in range(i+1)]
        C.lesser[i, n] += dt*trapezoid(AxB)

    return C

def convolute(A, B):
    """
    /*----------------------------------------------------------------------------
    This subroutine calculates a convolution of two Kadanoff-Baym function A
    and B,
                C(t,t')=(A*B)(t,t')

    for t=n*dt or t'=n*dt. Integrals are evaluated by the trapezoidal rule.
    ----------------------------------------------------------------------------*/

    retarded component

    .. math::
        
        C^{R}(t,t')=\int_{t'}^{t} ds A^{R}(t,s) B^{R}(s,t')


    Ref.
    1. Aoki, H. et al. Nonequilibrium dynamical mean-field theory and its applications.
        Rev. Mod. Phys. 86, 779–837 (2014).


    """
    # vector<complex<double> > AxB(max(parm_.N_tau+1,n+1));

    nt, ntau, size = A.nt, A.ntau, A.size
    dtau = A.dtau
    dt = A.dt

    assert(nt == B.nt)
    assert(ntau == B.ntau)
    assert(size == B.size)

    C = NEGF(nt=nt, ntau=ntau, size=size, dt=dt)
    

    # retarded
    for n in range(nt+1):
        for j in range(n+1):

            # res = 0

            AxB = [A.retarded[n, k] @ B.retarded[k, j] for k in range(j, n+1)]

            C.retarded[n,j] = dt * trapezoid(AxB) #trapezoid(AxB, j,n)
                        


  # left-mixing component
  # C^{Left}(t,tau')=\int_0^{beta} dtau A^{Left}(t,tau)*B^{M}(tau,tau')
  #                  +\int_0^{t} ds A^{R}(t,s)*B^{Left}(s,tau')
    for n in range(nt+1):
        for j in range(ntau+1):

            AxB = [A.left_mixing[n][k] @ B.matsubara[ntau+k-j] for k in range(j+1)]
            C.left_mixing[n][j] += - dtau * trapezoid(AxB)

            AxB = [A.left_mixing[n][k] @ B.matsubara[k-j] for k in range(j, ntau+1)]
            C.left_mixing[n][j] += dtau*  trapezoid(AxB) # trapezoid(AxB,j,ntau)

            AxB = [A.retarded[n][k] @ B.left_mixing[k][j] for k in range(n+1)]
            C.left_mixing[n][j] += dt*trapezoid(AxB)

  #   left_mixing[n][j]=0.0;
  #   for (int k=0;k<=j;k++) AxB[k]=A.left_mixing[n][k]*B.matsubara_t[parm_.N_tau+k-j];

  #   left_mixing[n][j]+=-parm_.dtau*trapezoid(AxB,0,j);
  #   for (int k=j;k<=parm_.N_tau;k++) AxB[k]=A.left_mixing[n][k]*B.matsubara_t[k-j];
  #   left_mixing[n][j]+=parm_.dtau*trapezoid(AxB,j,parm_.N_tau);
  #   for (int k=0;k<=n;k++) AxB[k]=A.retarded[n][k]*B.left_mixing[k][j];
  #   left_mixing[n][j]+=parm_.dt*trapezoid(AxB,0,n)

  # lesser component
  # # C^{<}(t,t')=-i\int_0^{beta} dtau A^{Left}(t,tau)*B^{Right}(tau,t')
  # #             +\int_0^{t'} ds A^{<}(t,s)*B^{A}(s,t')
  # #             +\int_0^{t} ds A^{R}(t,s)*B^{<}(s,t')
  #/ (i,j)=(n,j)
    for n in range(nt+1):
        for j in range(n):

            # C^< = A^r \cdot B^< + B^< \cdot C^a + B^{\rceil} \star C^{\lceil}

            # for (int k=0;k<=parm_.N_tau;k++) AxB[k]=A.left_mixing[n][k]*conj(B.left_mixing[j][parm_.N_tau-k]);

            AxB = [A.left_mixing[n][k] @ dag(B.left_mixing[j][ntau-k]) \
                   for k in range(ntau+1)]

            # lesser[n][j]+=-xj*parm_.dtau*trapezoid(AxB,0,parm_.N_tau)
            C.lesser[n][j] += -1j * dtau * trapezoid(AxB)

            # for k in range(j):
                # AxB[k] = A.lesser[n, k] * conj(B.retarded[j, k])
            AxB = [A.lesser[n, k] @ dag(B.retarded[j, k]) for k in range(j+1)]
            C.lesser[n, j] += dt * trapezoid(AxB)

            AxB = [A.retarded[n, k] @ B.lesser[k, j] for k in range(0, n+1)]
            C.lesser[n, j] += dt*trapezoid(AxB)

  # (i,j)=(i,n)
    for n in range(nt+1):
        for i in range(n+1):

            # for (int k=0;k<=parm_.N_tau;k++) AxB[k]=A.left_mixing[i][k]*conj(B.left_mixing[n][parm_.N_tau-k]);
            AxB = [A.left_mixing[i][k] @ dag(B.left_mixing[n][ntau-k]) \
                                             for k in range(ntau+1)]

            C.lesser[i][n] += -1j* dtau * trapezoid(AxB)

            AxB = [A.lesser[i, k] @ dag(B.retarded[n, k]) for k in range(n+1)]
            C.lesser[i, n] += dt * trapezoid(AxB)


            AxB = [A.retarded[i, k] @ B.lesser[k, n] for k in range(i+1)]
            C.lesser[i, n] += dt*trapezoid(AxB)

    return C

def volterra_int(n, G0, K):
    """
{
  /*----------------------------------------------------------------------------
      This subroutine solves a Volterra integral equation of the second kind,

                  G(t,t')=G0(t,t')+(K*G)(t,t'),

      for t=n*dt or t'=n*dt. The integral equation is solved by the second-order
      implicit Runge-Kutta method.
  ----------------------------------------------------------------------------*/
  vector<complex<double> > KxG(max(parm_.N_tau+1,n+1));

    Parameters
    ----------
    parm parm_ : TYPE
        DESCRIPTION.
    int n : TYPE
        DESCRIPTION.
    G0 : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """


    # retarded Green function
    # G^{R}(t,t')-\int_{t'}^{t} ds K^{R}(t,s)*G^{R}(s,t') = G0^{R}(t,t')
    G = NEGF()
    G.retarded[n, n] = G0.retarded[n, n]
    for j in range(n):
        retarded[n, j] = G0.retarded[n, j]
        for l in range(j):
            KxG[l] = K.retarded[n, l] * retarded[l, j]

        retarded[n, j] += dt * trapezoid_half_edge(KxG,j,n-1)
        retarded[n, j]/=1.0-0.5*dt*K.retarded[n, n]


  # # left-mixing Green function
  # # G^{Left}(t,tau')-\int_0^{t} ds K^{R}(t,s)*G^{Left}(s,tau')
  # #    = G0^{Left}(t,tau')+\int_0^{beta} dtau K^{Left}(t,tau)*G^{M}(tau,tau')
  # for (int j=0;j<=parm_.N_tau;j++){
  #   left_mixing[n][j]=G0.left_mixing[n][j];
  #   for (int l=0;l<=j;l++) KxG[l]=K.left_mixing[n][l]*matsubara_t[parm_.N_tau+l-j];
  #   left_mixing[n][j]+=-parm_.dtau*trapezoid(KxG,0,j);
  #   for (int l=j;l<=parm_.N_tau;l++) KxG[l]=K.left_mixing[n][l]*matsubara_t[l-j];
  #   left_mixing[n][j]+=parm_.dtau*trapezoid(KxG,j,parm_.N_tau);
  #   for (int l=0;l<=n-1;l++) KxG[l]=K.retarded[n][l]*left_mixing[l][j];
  #   left_mixing[n][j]+=parm_.dt*trapezoid_half_edge(KxG,0,n-1);
  #   left_mixing[n][j]/=1.0-0.5*parm_.dt*K.retarded[n][n];
  # }

  # # lesser Green function
  # # G^{<}(t,t')-\int_0^{t} ds K^{R}(t,s)*G^{<}(s,t')
  # #    = G0^{<}(t,t')-i\int_0^{beta} dtau K^{Left}(t,tau)*G^{Right}(tau,t')
  # #      +\int_0^{t'} ds K^{<}(t,s)*G^{A}(s,t')
  # # G^{<}(t_{n},t_{j})
    for j in range(n):
      lesser[n][j]=G0.lesser[n][j];
      # for (int l=0;l<=parm_.N_tau;l++) KxG[l]=K.left_mixing[n][l]*conj(left_mixing[j][parm_.N_tau-l]);
      # lesser[n][j]+=-xj*parm_.dtau*trapezoid(KxG,0,parm_.N_tau);
      for l in range(j+1): KxG[l]=K.lesser[n, l]*conj(retarded[j, l])
      lesser[n, j]+= dt*trapezoid(KxG,0,j)

      for l in range(n): KxG[l]=K.retarded[n, l]*lesser[l, j]
      lesser[n, j]+= dt*trapezoid_half_edge(KxG,0,n-1)
      lesser[n, j]/=1.0-0.5*dt*K.retarded[n, n]

    # # Hermite conjugate
    # # G^{<}(t_{i},t_{n})
    for i in range(n): lesser[i, n]=-conj(lesser[n, i])
     # G^{<}(t_{n},t_{n})
    lesser[n, n] = G0.lesser[n, n]
    # for (int l=0;l<=parm_.N_tau;l++) KxG[l]=K.left_mixing[n][l]*conj(left_mixing[n][parm_.N_tau-l]);
    # lesser[n][n]+=-xj*parm_.dtau*trapezoid(KxG,0,parm_.N_tau);
    for l in range(n+1): KxG[l]=K.lesser[n, l]*conj(retarded[n, l])
    lesser[n, n]+= dt*trapezoid(KxG,0,n)

    for l in range(n): KxG[l]=K.retarded[n, l]*lesser[l, n]
    lesser[n, n] += dt*trapezoid_half_edge(KxG,0,n-1)
    lesser[n, n] /= 1.0-0.5*dt*K.retarded[n, n]


class KBSolver:
    def __init__(self, h, dt, nt, beta, ntau, omegac):
        """
        Kadanoff-Baym equation solver for coupled fermion-boson systems.

        Only a single boson mode is considered.

        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.
        nt : TYPE
            DESCRIPTION.
        beta : TYPE
            DESCRIPTION.
        ntau : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.
        sign : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        None.

        """
        self.dt = dt
        self.nt = nt
        self.ntau = ntau
        # self.norb =
        self.size = h.shape[-1]
        self.beta = beta
        # self.nmodes = self.size_b = hb.shape[-1]
        self.h = h # mean-field H
        # self.hb = hb
        self.omegac = omegac

        self.G0 = None
        self.D0 = None

    def initiate(self):
        """
        Compute the bare electron and photon Green functions

        Returns
        -------
        G0 : TYPE
            DESCRIPTION.
        D0 : TYPE
            DESCRIPTION.

        """
        h = self.h
        dt=self.dt
        nt=self.nt
        beta=self.beta
        ntau=self.ntau
        w = self.omegac
        
        G0 = green_from_H(h, dt=self.dt, nt=self.nt, beta=self.beta, \
                          ntau=ntau, sign=-1)

        D0 = green_single_mode(w, dt=dt, nt=nt, beta=beta, ntau=ntau, sign=1, mu=10)
        # D0 = green_from_H(, dt=dt, nt=nt, beta=beta, ntau=ntau)

        # D0 = green_boson_XX(nt=nt, beta=beta, ntau=ntau, w=w, dt=dt)

        # set up the Green's function at t = 0, t' = 0
        return G0, D0

    def sigma(self):
        # set up the self-energy
        pass

    def run(self, sigma):
        pass
        # volterra_intdiff()


def volterra_intdiff(param_, n, h, K, G_der, G_der_new):
    """
    /*----------------------------------------------------------------------------
        This subroutine solves a Volterra integrodifferential equation of
        the second kind,

                    [i*d/dt-h(t)]G(t,t')=delta(t,t')+(K*G)(t,t'),

       for t=n*dt or t'=n*dt. The integrodifferential equation is solved by
       the second-order implicit Runge-Kutta method.
    ----------------------------------------------------------------------------*/
    vector<complex<double> > KxG(max(parm_.N_tau+1,n+1));

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    G_der : TYPE
        DESCRIPTION.
    G_der_new : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    xj = 1j


  # # retarded Green function
  # # d/dt G^{R}(t,t') = -i*delta(t,t')-i*h(t)*G^{R}(t,t')
  # #                    -i\int_{t'}^{t} ds K^{R}(t,s) G^{R}(s,t')
    G = NEGF(nt, ntau) # the following code only works for size == 1 now

    G.retarded[n, n] = -1j
    
 
    G_der_new.retarded[n] = -1j* h[n] @ G.retarded[n, n]

    for j in range(n):

        G.retarded[n, j] = G.retarded[n-1, j] + 0.5 * dt*G_der.retarded[j]

        KxG = [K.retarded[n][l] @ G.retarded[l][j] for l in range(n)]

        G_der_new.retarded[j] = -1j*dt * trapezoid_half_edge(KxG,j,n-1)

        G.retarded[n][j] += 0.5*dt*G_der_new.retarded[j]

        # Solve Eq. A7 in Ref. 1
        G.retarded[n][j] = np.linalg.solve(1.0+0.5j*dt*h[n] + 0.25j*dt*dt * K.retarded[n][n], G.retarded[n, j])

        G_der_new.retarded[j] += -1j*h[n] @ G.retarded[n][j]-0.5j*dt* K.retarded[n][n] @ G.retarded[n][j]


  # # left-mixing Green function
  # # d/dt G^{Left}(t,tau') = -i*h(t)*G^{Left}(t,tau')
  # #                         -i\int_0^{beta} dtau K^{Left}(t,tau)*G^{M}(tau,tau')
  # for (int j=0;j<=parm_.N_tau;j++){
  #   left_mixing[n][j]=left_mixing[n-1][j]+0.5*parm_.dt*G_der.left_mixing[j];
  #   for (int l=0;l<=j;l++) KxG[l]=K.left_mixing[n][l]*matsubara_t[parm_.N_tau+l-j];
  #   G_der_new.left_mixing[j]=xj*parm_.dtau*trapezoid(KxG,0,j);
  #   for (int l=j;l<=parm_.N_tau;l++) KxG[l]=K.left_mixing[n][l]*matsubara_t[l-j];
  #   G_der_new.left_mixing[j]+=-xj*parm_.dtau*trapezoid(KxG,j,parm_.N_tau);
  #   for (int l=0;l<=n-1;l++) KxG[l]=K.retarded[n][l]*left_mixing[l][j];
  #   G_der_new.left_mixing[j]+=-xj*parm_.dt*trapezoid_half_edge(KxG,0,n-1);
  #   left_mixing[n][j]+=0.5*parm_.dt*G_der_new.left_mixing[j];
  #   left_mixing[n][j]/=1.0+0.5*xj*parm_.dt*h[n]+0.25*xj*parm_.dt*parm_.dt*K.retarded[n][n];
  #   G_der_new.left_mixing[j]+=-xj*h[n]*left_mixing[n][j]-0.5*xj*parm_.dt*K.retarded[n][n]*left_mixing[n][j];
  # }

  # # lesser Green function
  # # d/dt G^{<}(t,t') = -i*h(t)*G^{<}(t,t')
  # #                    -i*(-i)*\int_0^{beta} dtau K^{Left}(t,tau)*G^{Right}(tau,t')
  # #                    -i*\int_0^{t'} ds K^{<}(t,s)*G^{A}(s,t')
  # #                    -i*\int_0^{t} ds K^{R}(t,s)*G^{<}(s,t')
  # #
  # # G^{<}(t_{n},t_{j}), d/dt G^{<}(t_{n},t_{j})

    for j in range(n):
        G.lesser[n][j] = lesser[n-1][j] + 0.5 * dt * G_der.lesser[j]

        for l in range(ntau+1):
            KxG[l] = K.left_mixing[n][l] @ conj(left_mixing[j][ntau-l])

        G_der_new.lesser[j]= -1j*(-xj) * dtau * trapezoid(KxG,0, ntau)

        KxG = [K.lesser[n][l]*conj(retarded[j][l]) for l in range(j+1)]
        G_der_new.lesser[j] += -1j * dt * trapezoid(KxG,0,j)

        KxG[l] = [K.retarded[n][l]*lesser[l][j] for l in range(n)]
        G_der_new.lesser[j] += -sf1j*parm_.dt*trapezoid_half_edge(KxG,0,n-1);
        lesser[n][j]+=0.5*parm_.dt*G_der_new.lesser[j];
        lesser[n][j]/=1.0+0.5*xj*parm_.dt*h[n]+0.25*xj*parm_.dt*parm_.dt*K.retarded[n][n];
        G_der_new.lesser[j]+=-xj*h[n]*lesser[n][j]-0.5*xj*parm_.dt*K.retarded[n][n]*lesser[n][j];

  # Hermite conjugate
    for i in range(n): lesser[i][n]=-conj(lesser[n][i]);
     # d/dt G^{<}(t_{n-1},t_{n})
    G_der_lesser=-1j*h[n-1]*lesser[n-1][n];
    for l in range(parm_.N_tau+1): KxG[l]=K.left_mixing[n-1][l]*conj(left_mixing[n][parm_.N_tau-l]);
    G_der_lesser+=-1j*(-xj)*parm_.dtau*trapezoid(KxG,0,parm_.N_tau);
    for l in range(n+1): KxG[l]=K.lesser[n-1][l]*conj(retarded[n][l]);
    G_der_lesser+=-1j*parm_.dt*trapezoid(KxG,0,n);
    for l in range(n): KxG[l]=K.retarded[n-1][l]*lesser[l][n];
    G_der_lesser+=-1j*parm_.dt*trapezoid(KxG,0,n-1);
    # / G^{<}(t_{n},t_{n}), d/dt G^{<}(t_{n},t_{n})
    lesser[n][n]=lesser[n-1][n]+0.5*parm_.dt*G_der_lesser;
    for l in range(parm_.N_tau+1): KxG[l]=K.left_mixing[n][l]*conj(left_mixing[n][parm_.N_tau-l]);
    G_der_new.lesser[n]=-xj*(-xj)*parm_.dtau*trapezoid(KxG,0,parm_.N_tau);
    for l in range(n+1): KxG[l]=K.lesser[n][l]*conj(retarded[n][l]);
    G_der_new.lesser[n]+=-xj*parm_.dt*trapezoid(KxG,0,n);
    for l in range(n): KxG[l]=K.retarded[n][l]*lesser[l][n];
    G_der_new.lesser[n]+=-xj*parm_.dt*trapezoid_half_edge(KxG,0,n-1);
    lesser[n][n]+=0.5*parm_.dt*G_der_new.lesser[n];
    lesser[n][n]/=1.0+0.5*xj*parm_.dt*h[n]+0.25*xj*parm_.dt*parm_.dt*K.retarded[n][n];
    G_der_new.lesser[n]+=-xj*h[n]*lesser[n][n]-0.5*xj*parm_.dt*K.retarded[n][n]*lesser[n][n];

if __name__ == '__main__':
    from pyqed import quadrature, pauli
    # import proplot as plt
    import matplotlib.pyplot as plt

    s0, sx, sy, sz = pauli()

    H = -0.5 * sz



    # omega = np.linspace(1e-8, 1, 6)
    # # f = bose(10, omega)
    # # print(f)

    # U = propagator_H_const(H, dt=0.01, nt=10)
    # print(U[-1])

    # G = green_from_H_const(H, mu=1e-10, beta=100, nt=100, ntau=1, h=0.01)
    # print(1j*G.retarded[-1, 0])

    # def H(t):
    #     return 0.5 * (-sz + np.cos(t)*s0)

    # U = propagator(H, dt=0.01, nt=10)

    def test_gf_eq():
        dt = 0.5
        nt = 100
        ntau = 10
        limit = 256
        beta = 10
        dos  = Ohmic()
        G = green_equilibrium(dos, beta, dt, nt, ntau, limit, mu=1)

        import proplot as plt
        fig, ax = plt.subplots()
        ax.plot(G.lesser[0, :].imag)

        fig, ax = plt.subplots()
        ax.plot(dos.dos(dos.sample(limit)))
        # ax.format(xlim=(0,2))

    # test_gf_eq()
    dt = 0.2
    nt = 100
    ntau = 1
    limit = 256
    beta = 10
    sol = KBSolver(h=H, dt=dt, nt=nt, beta=beta, ntau=ntau, omegac=1)

    G0, D0 = sol.initiate()
    print(D0.retarded.shape)
    
    print('G0', G0.retarded[1, 1], G0.lesser[1,1])
    print('D0', D0.retarded[1, 1], D0.lesser[1, 1])

    fig, ax = plt.subplots()
    # ax.plot(G0.retarded[nt,:, 0, 0].imag)

    # print(G0.lesser[0, 0])
    # print(G0.retarded[0, 0])



    ax.plot(D0.retarded[nt,:, 0, 0].imag, label=r'$D_0^R$')
    ax.plot(D0.retarded[nt,:, 0, 0].real, label='D0')
    ax.plot(G0.lesser[0,:, 0, 0].imag, label=r'Im $G_0^<$')
    ax.plot(G0.retarded[nt, :, 0, 0].imag, label=r'$G_0^r$')

    ax.legend()


    # polarization function
    chi = bubble(G0, G0, vertex=0.2*sx)

    print('chi', chi.retarded)
    
    fig, ax = plt.subplots()
    # ax.plot(G0.retarded[nt,:, 0, 0].imag)

    # ax.plot(D0.retarded[nt,:, 0, 0].imag)
    ax.plot(chi.lesser[0, :, 0, 0].real, label='$\chi$')
    ax.plot(chi.retarded[:, 0, 0, 0].imag, label='$\chi$')

    ax.plot(chi.lesser[:, 0, 0, 0].imag, label='$\chi$')
    
    # print(chi.retarded)
    ax.legend()

    print(chi.retarded.shape)
    print('D0 size', D0.retarded.shape)

    
    
    # Dyson equation with bare self-energy
    D = D0 @ chi @ D0 # convolute(D0, chi)
    # D =  convolute(tmp, D0)

    

    fig, ax = plt.subplots()
    # ax.plot(G0.retarded[nt,:, 0, 0].imag)
    t = np.arange(nt+1) * dt
    ax.plot(t, D.retarded[nt,:, 0, 0].imag)
    ax.plot(t, D.retarded[nt,:, 0, 0].real, label='DR')
    ax.plot(t, D.lesser[nt,:, 0, 0].imag, label=r'Im $D^R(t,0)$')
    ax.plot(t, D.lesser[nt,:, 0, 0].real, label=r'Re $D^R(t,0)$')
    ax.legend()


