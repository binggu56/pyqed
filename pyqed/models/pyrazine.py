#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:04:43 2021

Three-state two-mode linear vibronic model of pyrazine (S0/S1/S2)

@author: bing
"""


import numpy as np
# import numba
from scipy.sparse import identity, coo_matrix, lil_matrix, csr_matrix, kron
from numpy import meshgrid
from scipy.linalg import eigh
from cmath import log

import sys
# import proplot as plt
import matplotlib.pyplot as plt

from pyqed import boson, interval, sigmax, sort, ket2dm, overlap,\
    polar2cartesian, SPO2
    
from pyqed.models.vibronic import Vibronic2
    
from pyqed.style import set_style
from pyqed.units import au2ev, wavenumber2hartree, wavenum2au






def pos(n_vib):
    """
    position matrix elements <n|Q|n'>
    """
    X = np.zeros((n_vib, n_vib))

    for i in range(1, n_vib):
        X[i, i-1] = np.sqrt(i/2.)
    for i in range(n_vib-1):
        X[i, i+1] = np.sqrt((i+1)/2.)

    return X


def polariton_hamiltonian(n_el, n_vc, n_vt, cav, g):
    """
    contruct vibronic-cavity basis for polaritonic states, that is a
    direct product of electron-vibration-photon space
    """

    #n_basis = n_el * n_cav * n_vc * n_vt

    freq_vc = 952. * wavenumber2hartree
    freq_vt = 597. * wavenumber2hartree

    Eshift = np.array([0.0, 31800.0, 39000]) * wavenumber2hartree
    kappa = np.array([0.0, -847.0, 1202.]) * wavenumber2hartree
    coup = 2110.0 * wavenumber2hartree # inter-state coupling lambda


    # indentity matrices in each subspace
    I_el = identity(n_el)
    I_cav =  cav.idm
    I_vc = identity(n_vc)
    I_vt = identity(n_vt)

    h_cav = cav.ham()
    h_vt = boson(freq_vt, n_vt, ZPE=False)
    h_vc = boson(freq_vc, n_vc, ZPE=False)

    h_el = np.diagflat(Eshift)

    # the bare term in the system Hamiltonian
    h0 = kron(h_el, kron(I_cav, kron(I_vc, I_vt))) + \
        kron(I_el, kron(h_cav, kron(I_vc, I_vt))) \
        + kron(I_el, kron(I_cav, kron(h_vc, I_vt))) +\
        kron(I_el, kron(I_cav, kron(I_vc, h_vt))) \

    X = pos(n_vt)


    h1 = kron(np.diagflat(kappa), kron(I_cav, kron(I_vc, X)))


    Xc = pos(n_vc)

    trans_el = np.zeros((n_el, n_el)) # electronic excitation operator
    #deex = np.zeros((n_el, n_el)) # deexcitation
    #deex[2, 1] = 1.0
    #ex[1, 2] = 1.0
    trans_el[1,2] = trans_el[2,1] = 1.0 #= ex + deex

    #h_fake = kron(np.diagflat(kappa), kron(I_cav, kron(Xc, I_vt)))

    ###
#    h_m = np.zeros((n_basis, n_basis))
#
#    for m, b0 in enumerate(basis_set):
#        for n, b1 in enumerate(basis_set):
##            h_m[m,n] = h_el[b0.n_el, b1.n_el] * h_cav[b0.n_cav, b1.n_cav] * h_vc[b0.n_vc, b1.n_vc]
#            h_m[m,n] = trans_el[b0.n_el, b1.n_el] * I_cav[b0.n_cav, b1.n_cav] \
#                    * Xc[b0.n_vc, b1.n_vc] * I_vt[b0.n_vt, b1.n_vt]

    h2 = coup * kron(trans_el, kron(I_cav, kron(Xc, I_vt)), format='csr')

    # if n_cav = n _el
    deex_cav = cav.get_annihilate()
    ex_cav = cav.get_create()

    d_ex = np.zeros((n_el, n_el)) # electronic excitation operator
    d_deex = np.zeros((n_el, n_el)) # deexcitation
    d_deex[0, 2] = 1.0
    d_ex[2, 0] = 1.0

    dip = d_deex + d_ex

    h3 = g * kron(dip, kron(deex_cav + ex_cav, kron(I_vc, I_vt)))

    h_s = h0 + h1 + h2 + h3

    # polaritonic states can be obtained by diagonalizing H_S
    # v is the basis transformation matrix, v[i,j] = <old basis i| polaritonic state j>
    #eigvals, v = np.linalg.eigh(h_s)

    #h_s = csr_matrix(h_s)

    # collapse operators in dissipative dynamics
    Sc = kron(I_el, kron(I_cav, kron(Xc, I_vt)), format='csr')
    St = kron(I_el, kron(I_cav, kron(I_vc, X)), format='csr')

#    St = csr_matrix(St)
#    Sc = csr_matrix(Sc)

    return h_s, Sc, St

def vibronic_hamiltonian(n_el, n_vc, n_vt):
    """
    contruct vibronic-cavity basis for polaritonic states, that is a
    direct product of electron-vibration-photon space
    """

    #n_basis = n_el * n_cav * n_vc * n_vt

    freq_vc = 952. * wavenumber2hartree
    freq_vt = 597. * wavenumber2hartree

    Eshift = np.array([0.0, 31800.0, 39000]) * wavenumber2hartree
    kappa = np.array([0.0, -847.0, 1202.]) * wavenumber2hartree
    coup = 2110.0 * wavenumber2hartree # inter-state coupling lambda


    # indentity matrices in each subspace
    I_el = identity(n_el)
    I_vc = identity(n_vc)
    I_vt = identity(n_vt)

    h_vt = boson(freq_vt, n_vt, ZPE=False)
    h_vc = boson(freq_vc, n_vc, ZPE=False)

    h_el = np.diagflat(Eshift)

    # the bare term in the system Hamiltonian
    h0 = kron(h_el, kron(I_vc, I_vt)) + kron(I_el, kron(h_vc, I_vt)) +\
         kron(I_el, kron(I_vc, h_vt))

    X = pos(n_vt)

    h1 = kron(np.diagflat(kappa), kron(I_vc, X))

    Xc = pos(n_vc)

    trans_el = np.zeros((n_el, n_el)) # electronic excitation operator
    #deex = np.zeros((n_el, n_el)) # deexcitation
    #deex[2, 1] = 1.0
    #ex[1, 2] = 1.0
    trans_el[1,2] = trans_el[2,1] = 1.0 #= ex + deex

    #h_fake = kron(np.diagflat(kappa), kron(I_cav, kron(Xc, I_vt)))

    ###
#    h_m = np.zeros((n_basis, n_basis))
#
#    for m, b0 in enumerate(basis_set):
#        for n, b1 in enumerate(basis_set):
##            h_m[m,n] = h_el[b0.n_el, b1.n_el] * h_cav[b0.n_cav, b1.n_cav] * h_vc[b0.n_vc, b1.n_vc]
#            h_m[m,n] = trans_el[b0.n_el, b1.n_el] * I_cav[b0.n_cav, b1.n_cav] \
#                    * Xc[b0.n_vc, b1.n_vc] * I_vt[b0.n_vt, b1.n_vt]

    h2 = coup * kron(trans_el, kron(Xc, I_vt), format='csr')


    h_s = h0 + h1 + h2

    # polaritonic states can be obtained by diagonalizing H_S
    # v is the basis transformation matrix, v[i,j] = <old basis i| polaritonic state j>
    #eigvals, v = np.linalg.eigh(h_s)

    # collapse operators in dissipative dynamics
    # Sc = kron(I_el, kron(Xc, I_vt), format='csr')
    # St = kron(I_el, kron(I_vc, X), format='csr')

    return h_s




class Pyrazine(Vibronic2):
    """
    vibronic coupling model for pyrazine S0/S1/S2 conical intersection
    """
    def __init__(self, x=None, y=None):
        self.x = x 
        self.y = y
        
        self.nx = len(x)
        self.ny = len(y)
        
        self.nstates = 3
        
        self.idm_el = np.eye(self.nstates)
        
        self.edip = np.zeros((self.nstates, self.nstates))
        self.edip[0, 2] = self.edip[2, 0] = 1.
        
        self.mass = [1/(952. * wavenum2au), 1./(597. * wavenum2au)]
        
        self.v = None

    def apes(self, x):

        x = np.atleast_1d(x)
        nx = len(x)

        wlist = []
        ulist = []

        for i in range(nx):
            # v = np.zeros((ns, ns))

            w, u = get_apes(*x[i])
            w, u = sort(w, u)

            wlist.append(w.copy())
            ulist.append(u.copy())


        return wlist, ulist
    
    def buildV(self):
        """
        Build the diabatic PES

        Returns
        -------
        None.

        """
        nx, ny = self.nx, self.ny 
        nstates = self.nstates
        
        x, y = self.x, self.y
        
        v = np.zeros((nx, ny, nstates, nstates))
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        freq_vc = 952. * wavenum2au
        freq_vt = 597. * wavenum2au

        Eshift = np.array([31800.0, 39000]) * wavenum2au
        kappa = np.array([-847.0, 1202.]) * wavenum2au

        v0 = freq_vc * X**2/2. + freq_vt * Y**2/2 + kappa[0] * Y + Eshift[0]
        v1 = freq_vc * X**2/2 + freq_vt * Y**2/2 + kappa[1] * Y + Eshift[1]

        coup = 2110 * X * wavenum2au

        vg = freq_vc * X**2/2. + freq_vt * Y**2/2.
        
        v[:, :, 0, 0] = vg 
        v[:, :, 1, 1] = v0
        v[:, :, 2, 2] = v1
        v[:, :, 2, 1] = coup 
        v[:, :, 1, 2] = coup 
        
        self.v = v
        
        return v
        
    def dpes(self, x, y):
        """
        Diabatic PES
    
        Parameters
        ----------
        x : TYPE
            qc coupling mode coordinate
        y : TYPE
            qt tuning mode coordinate
    
        Returns
        -------
        2D array
            molecular Hamiltonian
    
        """
        nstates = self.nstates
        
        freq_vc = 952. * wavenumber2hartree
        freq_vt = 597. * wavenumber2hartree
    
        Eshift = np.array([31800.0, 39000]) * wavenumber2hartree
        kappa = np.array([-847.0, 1202.]) * wavenumber2hartree
    
        V0 = freq_vc * x**2/2. + freq_vt * y**2/2 + kappa[0] * y + Eshift[0]
        V1 = freq_vc * x**2/2 + freq_vt * y**2/2 + kappa[1] * y + Eshift[1]
    
        coup = 2110 * x * wavenumber2hartree
    
        Vg = freq_vc * x**2/2. + freq_vt * y**2/2
    
        hmol = np.zeros((nstates, nstates))
        hmol[0, 0] = Vg
        hmol[1, 1] = V0
        hmol[2, 2] = V1
        hmol[1,2] = hmol[2,1] = coup
    
        return hmol


def get_apes(x, y):
    """
    diabatic PES
    input:
        R: 1d array with length n_dof
    output:
        V: same size as R, potential energy
    """

    freq_vc = 952. * wavenum2au
    freq_vt = 597. * wavenum2au

    Eshift = np.array([31800.0, 39000]) * wavenum2au
    kappa = np.array([-847.0, 1202.]) * wavenum2au

    V0 = freq_vc * x**2/2. + freq_vt * y**2/2 + kappa[0] * y + Eshift[0]
    V1 = freq_vc * x**2/2 + freq_vt * y**2/2 + kappa[1] * y + Eshift[1]

    coup = 2110 * x * wavenum2au

    Vg = freq_vc * x**2/2. + freq_vt * y**2/2.

    #A0 = np.zeros(len(x))
    #A1 = np.zeros(len(x))

    #for i in range(len(x)):
    V = np.array([[V0, coup], [coup, V1]])

    w, u = np.linalg.eigh(V)

    return w, u


def cut():
    x = 0
    y = np.linspace(-8,6,100)

    dpes = DPES(x, y)

    fig, ax = plt.subplots(figsize=(4,4))
    # set_style(13)

    for surface in dpes:
        ax.plot(y, surface * au2ev, lw=2)
    #ax.plot(y, (dpes[1] - dpes[0]) * au2ev, label='0-1')
    #ax.plot(y, (dpes[2]- dpes[0]) * au2ev, label='0-2')

    #ax.legend()
    #ax.set_ylim(4.31, 4.32)
    #ax.grid()
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel('Tuning mode')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig('dpes.pdf', dpi=1200, transparent=True)
    plt.show()


def plot3d():

    #data = [go.Surface(z=apes)]
    #fig = go.Figure(data = data)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5,4))
    set_style(fontsize=14)

    ax = fig.add_subplot(111, projection='3d')
    #py.iplot(fig)

    x = np.linspace(-4, 4)
    y = np.linspace(-4, 4)

    apes = np.zeros((len(x), len(y)))
    apes1 = np.zeros((len(x), len(y)))
    apes2 = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            apes[i,j], [apes1[i,j], apes2[i,j]] = get_apes(x[i], y[j])

    X, Y = np.meshgrid(x, y)

    for surface in [apes, apes1, apes2]:
        ax.plot_surface(X, Y, surface * au2ev, rstride=1, cstride=1, cmap='viridis',\
                    edgecolor='k',
                    linewidth=0.1)

    #surf(ground)
#    ax.plot_surface(X, Y, apes1 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)
#
#    ax.plot_surface(X, Y, apes2 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)

    ax.view_init(10, -60)
    ax.set_zlim(0, 7)
    ax.set_xlabel(r'Couping mode')
    ax.set_ylabel(r'Tuning mode')

    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Energy (eV)', rotation=90)

    #fig.subplots_adjust(top=0.95, bottom=0.16,left=0.16, right=0.9)

    plt.savefig('apes_3d.pdf')

    plt.show()


def contour():

    #data = [go.Surface(z=apes)]
    #fig = go.Figure(data = data)


    x = np.linspace(-6, 6, 200)
    y = np.linspace(-4, 4, 200)

    apes = np.zeros((len(x), len(y)))
    apes1 = np.zeros((len(x), len(y)))
    apes2 = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            apes[i,j], [apes1[i,j], apes2[i,j]] = get_apes(x[i], y[j])

    X, Y = np.meshgrid(x, y)

    for j, surface in enumerate([apes, apes1, apes2]):

        # fig, ax = plt.subplots()

        fig, ax = matplot(x, y, surface.T * au2ev, cmap='inferno')

    #ax.contour(apes1)
    #surf(ground)
#    ax.plot_surface(X, Y, apes1 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)
#
#    ax.plot_surface(X, Y, apes2 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)

        ax.set_xlabel(r'Tuning mode $Q_\mathrm{t}$')
        ax.set_ylabel(r'Coupling mode $Q_\mathrm{c}$')

        #ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        #ax.set_zlabel('Energy (eV)', rotation=90)

        fig.subplots_adjust(top=0.95, bottom=0.16,left=0.16, right=0.95)

        plt.savefig('apes{}_contour.pdf'.format(j))

    return

def mayavi(surfaces):


    from mayavi import mlab

    # apes, apes1 = surfaces

    fig = mlab.figure()
    for surface in surfaces:
        mlab.surf(surface * au2ev)
    # surf3 = mlab.surf(apes1 * au2ev, warp_scale=20)
    #mlab.surf(ground * au2ev, warp_scale=20)



    mlab.axes(xlabel = 'Coupling mode', ylabel = 'Tuning mode')
    #mlab.view(60, 74, 17, [-2.5, -4.6, -0.3])

    mlab.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import rcParams

    rcParams['axes.labelpad'] = 6
    rcParams['xtick.major.pad']='2'
    rcParams['ytick.major.pad']='2'

    x = np.linspace(-1, 1)
    y = np.linspace(-1, 1)
    deltas = [-0.5, 0, 0.5, 1, 2]

    # loop_integral = np.zeros(len(deltas))
    # for i, delta in enumerate(deltas):
    mol = LVC2(x, y, delta=0)
    mol.plot_apes()

    #     loop_integral[i] = mol.berry_phase(n=1, r=3)

    # fig, ax = plt.subplots()
    # ax.plot(deltas, loop_integral)


    # mol.plot_apes()

    

    # from pyqed.style import plot_surface
    # plot_surface(x, y, F)

    # mayavi()
    # contour()
    # cut()
    # plot3d()



