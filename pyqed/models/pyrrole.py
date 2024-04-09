# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:41:15 2022

vibronic model for pyrrole

@author: Bing Gu (gubing@westlake.edu.cn)

Refs:
    Domcke JCP 2005
"""

from numpy import sqrt, exp, tanh, pi, cos, sin
import numpy as np
from numpy.linalg import inv, det

from pyqed.phys import heaviside, meshgrid, morse, Morse
from pyqed.units import au2ev, atomic_mass, au2amu
import numba

from scipy.io import savemat

def export_to_matlab(fname, psi, fmt='matlab'):
    mdic = {'wavefunction': psi}
    savemat(fname, mdic)
    return

def toarray(psilist):

    return np.array(psilist)


class Pyrrole:
    """
    S2/S0 conical intersection model for photodissociation of pyrrole
    
    Refs
        Domcke
    """
    def __init__(self):

        self.r0 = 1.959 # equilibrium bond length at theta = 0

        self.reduced_mass = self._reduced_mass()

    def _reduced_mass(self):
        mH = atomic_mass['H'] /au2amu
        mN = atomic_mass['N'] /au2amu
        mM = 4. *(atomic_mass['C']/au2amu + mH)
        mu = mH*(mM + mN)/(mH + mM + mN)
        self.reduced_mass = mu
        return mu

    def v11(self, r):
        D1 = 5.117/au2ev
        r1 = 1.959
        a1 = 1.196
        return morse(r, D1, a1, r1)

    def v21(self, r):
        D21 = 8.07/au2ev
        r21 = 1.922
        a21 = 0.882
        E2 = 5.584/au2ev

        return morse(r, D21, a21, r21) + E2

    def v22(self, r):
        A22 = 0.091/au2ev
        D22 = 4.092/au2ev
        r22 = 5.203
        a22 = 1.290
        return A22 * exp(-a22 * (r - r22)) + D22

    def omegac1(self, r):
        d2 = 2.696
        alpha1 = 0.00015

        B11 = 5.147/au2ev
        B12 = -1.344/au2ev
        B13 = 0.884/au2ev
        B14 = 1.2910
        d1 = 3.1

        def f1(r):
            return 0.5 * (1. + tanh((r-d2)/alpha1))

        return (B11 + B12 * r) * (1. - f1(r)) + B13 * exp(-B14 * (r - d1)) * f1(r)



    def omegac2(self, r):
        B21 = 3.819/au2ev
        B22 = -1.219/au2ev

        B23 = 2.335/(au2ev)
        B24 = 0.226/(au2ev)

        return (0.5 * (B21 + B22*r) - 0.5 * sqrt((B23 + B22*r)**2 + 4*B24**2)) \
            * heaviside(2.55-r)

    def l12(self, r):
        lmax = 2.4/au2ev

        beta12 = 1.942

        d12 = 3.454

        return 0.5 * lmax * (1 - tanh((r - d12)/beta12))


    def DPES(self, r, qc):

        l22 = 1.669/au2ev

        nx = len(r)
        ny = len(qc)

        V = np.zeros((nx, ny, 2, 2))

        R, Qc = np.meshgrid(r, qc, indexing='ij')

        # transform to rNH, theta
        R, Qc = self.transform(R, Qc)

        # diabatic surfaces
        V[:, :, 0, 0] = self.v11(R) + 0.5 * self.omegac1(R) * Qc**2

        V[:, :, 1, 1] = 0.5 * (self.v21(R) + self.v22(R)) - 0.5 * sqrt((self.v21(R) - self.v22(R))**2 + 4*l22**2)\
            + 0.5 * self.omegac2(R) * Qc**2

        # diabatic couplings
        V[:, :, 0, 1] = V[:, :, 1, 0] = self.l12(R) * Qc

        # v = morse(r, D, a1, r1) + 0.5 * omegac(r)**2 * qc**2

        return V

    def APES(self, r, q, n=0):
        
        return V

    def S0(self, r, qc):

        # nx = len(r)
        # ny = len(qc)

        # V = np.zeros((nx, ny))

        # R, Qc = np.meshgrid(r, qc, indexing='ij')

        # transform to rNH, theta
        rNH, theta = self.transform(r, qc)

        # diabatic surfaces
        V = self.v11(rNH) + 0.5 * self.omegac1(rNH) * theta**2

        return V

    def transform(self, r, q):
        """
        transform between Jacobi coordinates to internal coordinates

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        q : TYPE
            DESCRIPTION.

        Returns
        -------
        rNH : TYPE
            DESCRIPTION.
        theta : TYPE
            DESCRIPTION.

        """
        rNH = sqrt(r**2*sin(q)**2 + (r*cos(q) - 2.168)**2)
        theta = np.arcsin(r/rNH * sin(q))
        return rNH, theta

    def plot_surface(self, r, q, **kwargs):
        import proplot as plt
        X, Y = meshgrid(r, q)

        fig, ax = plt.subplots()
        ax.contourf(X, Y, self.S0(X, Y), **kwargs)

        return
    
    def inertia(self, r):
        mH = atomic_mass['H'] /au2amu
        mN = atomic_mass['N'] /au2amu
        mM = 4. *(atomic_mass['C']/au2amu + mH)
        mu = mH*(mM + mN)/(mH + mM + mN)
        mu_MN = mM * mN/(mM + mN)

        rMN = 2.7512
        I = 1./(mu * r**2) + 1./(mu_MN * rMN**2)

        return 1./I

    def eigenstates(self, nstates=0, method='dvr'):
        
        # from pyqed.dvr.dvr_2d import DVR2
        # DVR2()    
        pass
    

class PyrroleCation:
    """
    Adiabatic potential energy surfaces fitted by Feng Chen with
    quantum chemistry data from Shichao Sun.
    """
    def __init__(self):

        self.r0 = 1.9404 # equilibrium bond length at theta = 0
        self.E0 = 0.2999

        self.reduced_mass = self._reduced_mass()
        self._V = None

    def _reduced_mass(self):
        mH = atomic_mass['H'] /au2amu
        mN = atomic_mass['N'] /au2amu
        mM = 4. *(atomic_mass['C']/au2amu + mH)
        mu = mH*(mM + mN)/(mH + mM + mN)
        self.reduced_mass = mu
        return mu

    def v11(self, r):
        D1 = 0.2167
        r1 = self.r0
        a1 = 1.055
        return morse(r, D1, a1, r1)

    # def v21(self, r):
    #     D21 = 8.07/au2ev
    #     r21 = 1.922
    #     a21 = 0.882
    #     E2 = 5.584/au2ev

    #     return morse(r, D21, a21, r21) + E2

    # def v22(self, r):
    #     A22 = 0.091/au2ev
    #     D22 = 4.092/au2ev
    #     r22 = 5.203
    #     a22 = 1.290
    #     return A22 * exp(-a22 * (r - r22)) + D22

    def omegac(self, r):
        d2 = 4.6353
        alpha1 = 2.0202

        B11 = 0.0851
        B12 = -0.0126
        B13 = 6.1015
        B14 = 1.9383

        def f1(r):
            return 0.5 * (1. + tanh((r-d2)/alpha1))

        return (B11 + B12 * r) * (1. - f1(r)) + B13 * exp(-B14 * r) * f1(r)


    def D0(self, r, qc):

        # R, Qc = np.meshgrid(r, qc, indexing='ij')

        # transform to rNH, theta
        rNH, theta = self.transform(r, qc)

        # diabatic surfaces
        V = self.v11(rNH) + 0.5 * self.omegac(rNH) * theta**2

        # self._V = V
        return V
    
    def D1(self, r, q):
        
        def v(r):
            D = 0.2028
            a = 1.0732
            r0 = 1.9537
            return morse(r, D, a, r0)
        
        def omegac(r):
            d2 = 4.4689
            alpha1 = 0.5077
    
            B11 = 0.1278
            B12 = -0.0257
            B13 = 36.7638
            B14 = 1.6474
    
            f1 = lambda r : 0.5 * (1. + tanh((r-d2)/alpha1))
    
            return (B11 + B12 * r) * (1. - f1(r)) + B13 * exp(-B14 * r) * f1(r)      
            
        rNH, theta = self.transform(r, q)
        
        V = v(rNH) + 0.5 * omegac(rNH)**2 * theta**2
        return V

    def inertia(self, r):
        mH = atomic_mass['H'] /au2amu
        mN = atomic_mass['N'] /au2amu
        mM = 4. *(atomic_mass['C']/au2amu + mH)
        mu = mH*(mM + mN)/(mH + mM + mN)
        mu_MN = mM * mN/(mM + mN)

        rMN = 2.7512
        I = 1./(mu * r**2) + 1./(mu_MN * rMN**2)

        return 1./I

    def transform(self, r, q):
        """
        transform between Jacobi coordinates to internal coordinates

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        q : TYPE
            DESCRIPTION.

        Returns
        -------
        rNH : TYPE
            DESCRIPTION.
        theta : TYPE
            DESCRIPTION.

        """
        rNH = sqrt(r**2*sin(q)**2 + (r*cos(q) - 2.168)**2)
        theta = np.arcsin(r/rNH * sin(q))
        return rNH, theta

    def plot_surface(self, r, q, **kwargs):
        import proplot as plt
        X, Y = meshgrid(r, q)

        fig, ax = plt.subplots()
        ax.contourf(X, Y, self.D0(X, Y), **kwargs)

        return


if __name__=='__main__':

    from lime.wpd import SPO2
    from lime.units import au2fs
    from lime.phys import gwp
    import proplot as plt


    mol = Pyrrole()



    r = np.linspace(3, 10, 256)
    q = np.linspace(-2, 2, 128)

    R, Q = np.meshgrid(r, q, indexing='ij')
    mu = mol.reduced_mass
    sol = SPO2(x=r, y=q, masses=[mu, mol.inertia], coords='jacobi')

    V, wq = mol.DPES(r, q)
    sol.V = V

    fig, ax = plt.subplots()
    ax.plot(r, V[:, np.abs(q).argmin(), 0, 0])
    ax.plot(r, V[:, np.abs(q).argmin(), 1, 1])
    ax.format(xlim=(3.8,4))

    fig, ax = plt.subplots()
    ax.plot(q, V[np.abs(r-1.959).argmin(), :,  0, 0])
    ax.plot(q, V[np.abs(r-1.959).argmin(), :, 1, 1])

    nx, ny = len(r), len(q)

    mo = Morse(D=D1, re=r1, a=a1, mass=mu)
    phi = mo.eigenstate(r, 1)


    # run 2d nonadiabatic wavepacket dynamics, there is problem with the initial density
    # state, which can be solved by imaginary time propagator

    psi0 = np.zeros((nx, ny, 2), dtype=complex)
    sigma = 1./(inertia(transform(r1, q=0)[0]) * wq)**0.5

    # psi0[:, :, 1] = np.outer(phi, gwp(q, sigma=1./(inertia(r1) * wq)**0.5))
    R, Q = meshgrid(r, q)
    RNH, Theta = transform(R, Q)

    # for i in range(nx):
    #     for j in range(ny):
    #         rNH, theta = transform(r[i], q[j])
    #         psi0[i, j, 1] = mo.eigenstate(rNH, 1) * gwp(theta, sigma=sigma)

    psi0[:, :, 1] =  mo.eigenstate(RNH, 1) * gwp(Theta, sigma=sigma)

    ax0, ax1 = sol.plt_wp([psi0])
    ax0.format(xlim=(4, 5))
    # sol.plot_surface()

    # r = sol.run(psi0, dt=0.01/au2fs, Nt=1200, nout=10)




    # for j in range(len(r.times)):
    #     export(str(r.times[j])+'.mat', r.psilist[j])

    # ax0, ax1 = sol.plt_wp([r.psi])
    # ax0.format(ylim=(-0.4, 0.4))
    # ax1.format(ylim=(-0.4, 0.4))




    # from lime.wpd import SPO
    # rNH = np.linspace(1, 12, 1024)
    # spo = SPO(rNH, mass=mH)
    # spo.V = S1(rNH)

    # r = spo.run(psi0=mo.eigenstate(rNH, 1), dt=0.02/au2fs, Nt=800, nout=5)

    # fig, ax = plt.subplots()
    # ax.plot(rNH, r.psi)
    # ax.plot(rNH, r.psi0)



    # psi = toarray(r.psilist)
    # export('psi.mat', psi)

    # for j in range(len(r.times)):
    #     export(str('{:f2}'.format(r.times[j]*au2fs))+'.mat', r.psilist[j])



