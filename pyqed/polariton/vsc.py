#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:26:00 2023

@author: bingg
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, kron, identity, linalg

from pyqed.units import au2fs, au2k, au2ev
from pyqed import dag, coth, ket2dm, comm, anticomm, sigmax, sort, Composite, rk4,\
    interval
from pyqed.optics import Pulse
from pyqed.namd.diabatic import SPO2, SPO3
from pyqed.polariton.cavity import Cavity

import sys
if sys.version_info[1] < 10:
    import proplot as plt
else:
    import matplotlib.pyplot as plt




class VSC:
    """
    2D vibronic model in the diabatic/adiabatic representation coupled to
    a single-mode IR cavity (vibrational strong coupling)
    
    The photon mode is treated the same as a vibrational mode, although the cavity mode
    is not directly coupled to the electronic motion.

    """
    def __init__(self, mol, cav):
        """
        

        Parameters
        ----------
        mol : LVC obj
            vibronic model
        cav : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.mol = mol
        self.cav = cav
        self.x, self.y = mol.x, mol.y
        # self.X, self.Y, self.Q = np.meshgrid(mol.x, mol.y, cav.x)
        self.nx, self.ny = mol.nx, mol.ny
        self.nel = self.mol.nstates
        self.ncav = self.cav.ncav
        self.nstates = self.nel

        if cav.x is None:
            raise ValueError('Please specify the x varible in Cavity.')
        
        self.q = cav.x
        self.nq = len(cav.x)
        self.dq = interval(self.q)
        self.dx = interval(self.x)
        self.dy = interval(self.y)
        


        self.v = self.vd = None
        self.va = None # adiabatic polaritonic PES
        self.diabatic_to_adiabatic = self._transformation = None # diabatic to adiabatic transformation matrix
        self._ground_state = None
        self.g = None

    def ground_state(self, representation='adiabatic'):
        # if rwa:
        #     return self.mol.ground_state()
        # else:
        #     # construct the ground-state polaritonic PES

        from pyqed.dvr.dvr_2d import DVR2

        x = self.x
        y = self.y

        if self.va is None:
            self.ppes()

        dvr = DVR2(x, y, mass=self.mol.mass) # for normal coordinates

        if representation == 'adiabatic':
            V = self.va[:, :, 0]
        elif representation == 'diabatic':
            V = self.v[:, :, 0, 0]

        E0, U = dvr.run(V, k=1)

        self._ground_state = U[:, 0].reshape(self.nx, self.ny)

        return E0, self._ground_state


    def build_dpes(self, g, rwa=False):
        """
        Compute the diabatic potential energy surfaces with cavity
        
        .. math::
            
            H_{int} = \frac{1}{2} \omega_c (p_c - g \mu(R))^2

        Parameters
        ----------
        g : TYPE
            DESCRIPTION.
        dms: callable 
            dipole moment (projected on the cavity polarization) surface
        rwa : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        v : TYPE
            DESCRIPTION.

        """
        
        gx, gy = g
        
        self.g = g
        
        mol = self.mol
        cav = self.cav

        omegac = cav.omega

        nx, ny, nq = self.nx, self.ny, self.nq
        x, y, q = self.x, self.y, self.q
        
        nel = mol.nstates

        vd = np.zeros((nx, ny, nq, nel, nel))
        
        # build the global DPES
        if mol.v is None:
            mol.dpes_global()
        
        idm = np.eye(nel) # electronic identity
        
        # diabatic polariton surfaces
        for i in range(nx):
            for j in range(ny):
                for k in range(nq):   

        # for n in range(nel):
            # for i in range(nx):
            #     for j in range(ny):
                    
                    vd[i, j, k] = mol.v[i, j] + (0.5 * omegac**2 * q[k] \
                            + gy * y[j] * q[k]) * idm

        # diabatic couplings
        for n in range(nel):
            for m in range(n):
                for k in range(nq):
                    vd[:, :, k, n, m] = mol.v[:, :, n, m]
                    vd[:, :, k, m, n] = vd[:, :, k, n, m].conj()
                        


        # # cavity-molecule coupling
        # a = cav.annihilate()

        # v += np.tile(g * kron(mol.edip.real, a + dag(a)).toarray(), (nx, ny, 1, 1))

        self.vd = self.v = vd

        return vd

    def ppes(self, return_transformation=False):
        """
        Compute the polaritonic potential energy surfaces by diagonalization

        Parameters
        ----------
        return_transformation : TYPE, optional
            Return transformation matrices. The default is False.

        Returns
        -------
        E : TYPE
            DESCRIPTION.

        """

        nx = self.nx
        ny = self.ny
        nq = self.nq
        nstates = self.nstates

        E = np.zeros((self.nx, self.ny, nq, nstates))

        if not return_transformation:

            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(nq):
                        V = self.v[i, j, k, :, :]
                        w = np.linalg.eigvalsh(V)
                    # E, U = sort(E, U)
                        E[i, j, k, :] = w
        else:

            T = np.zeros((nx, ny, nq, nstates, nstates), dtype=complex)

            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(nq):
                        V = self.v[i, j, :, :]
                        w, u = sort(*np.linalg.eigh(V))
    
                        E[i, j, k, :] = w
                        T[i, j, k, :, :] = u

            self._transformation = T

        self.va = E

        return E

    def berry_curvature(self, state_id):
        # compute Berry curvature from the A2D transformation matrix
        pass

    def run(self, psi0=None, dt=0.1, nt=10, t0=0, nout=1, method='spo'):

        dx = interval(self.x)
        dy = interval(self.y)
        dq = interval(self.q)
        
        kx = 2. * np.pi * np.fft.fftfreq(self.nx, dx)
        ky = 2. * np.pi * np.fft.fftfreq(self.ny, dy)
        kz = 2. * np.pi * np.fft.fftfreq(self.nq, dq)
        
        if psi0 is None:
            psi0 = np.zeros((self.nx, self.ny, self.nstates))
            psi0[:, :, 0] = self._ground_state
        
        if method == 'spo':
            
            spo = SPO3(self.x, self.y, self.q, mass=[*self.mol.mass, 1], \
                       nstates=self.nstates)
            spo.V = self.v
    
            return spo.run(psi0=psi0, dt=dt, nt=nt, t0=t0, nout=nout)
        
        elif method == 'rk4':
            
            x, y, q = self.x, self.y, self.q
            
            gx, gy = self.g
            
            t = t0
            psi = psi0.copy()
            
            psilist = [psi]
            
            for k in range(nt//nout):

                for l in range(nout):

                    t += dt 
                    psi = rk4(psi, hpsi, dt, x, kx, ky, kz, self.v, gx)
                
                psilist.append(psi.copy())
                
            results = {}
            results['dt'] = dt
            results['nt'] = nt
            results['psi'] = psilist
            
            
            return results
    
    def rdm_el(self, psi):
        """
        
        compute the electronic reduced density matrix 

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        rho : TYPE
            DESCRIPTION.

        """
        
        rho = np.einsum('ijkm, ijkn -> mn', psi, psi.conj()) * self.dx * self.dy * self.dq
        return rho 

    def rdm_cav(self, psi):
        """
        
        compute the cavity reduced density matrix 

        Parameters
        ----------
        psi : TYPE
            DESCRIPTION.

        Returns
        -------
        rho : TYPE
            DESCRIPTION.

        """
        
        rho = np.einsum('ijkn, ijln -> kl', psi, psi.conj()) 
        return rho 

    def plot_surface(self, state_id=0, representation='adiabatic'):
        from pyqed.style import plot_surface

        if representation == 'adiabatic':
            plot_surface(self.x, self.y, self.va[:, :, state_id])
        else:
            plot_surface(self.x, self.y, self.v[:, :, state_id, state_id])
            
        return

    def plot_wavepacket(self, psilist, **kwargs):

        if not isinstance(psilist, list): psilist = [psilist]


        for i, psi in enumerate(psilist):
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharey=True)

            ax0.contour(self.X, self.Y, np.abs(psi[:,:, 1])**2)
            ax1.contour(self.X, self.Y, np.abs(psi[:, :,0])**2)
            # ax0.format(**kwargs)
            # ax1.format(**kwargs)
            fig.savefig('psi'+str(i)+'.pdf')
        return ax0, ax1

    def plot_ground_state(self, **kwargs):


        fig, ax = plt.subplots()

        ax.contour(self.X, self.Y, np.real(self._ground_state))
        # ax.format(**kwargs)

        fig.savefig('ground_state.pdf')
        return ax

    def promote_op(self, a, kind='mol'):
        """
        promote a local operator to the composite polariton space

        Parameters
        ----------
        a : TYPE
            DESCRIPTION.
        kind : TYPE, optional
            DESCRIPTION. The default is 'mol'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if kind in ['mol', 'm']:

            return kron(a, self.cav.idm)

        elif kind in ['cav', 'c']:

            return kron(self.mol.idm, a)


def hpsi(psi, x, kx, ky, kz, vmat, g, coordinates='linear', mass=None, G=None):
    """
    evaluate :math:`H \ket{psi}` for a vibromic model coupled to a chiral cavity mode
    
    .. math::
        
        H = T_\text{N} + V(x, y) + \frac{1}{2} (p_c^2 + \omega_c^2 q_c^2) + \
            g (x p_c + y q_c) 
    
    where we have neglected the dipole self-energy.
    
    The momentum operator is done with FFT.
    
    input:
        v: 1d array, adiabatic surfaces
        d: nonadiabatic couplings, matrix
        G: 4d array N, N, Nx, Ny (N: # of states)
    output:
        hpsi: H operates on psi
    """
    ndim = 3
    
    if mass is None:
        # raise Warning('Mass not given, set to 1.')
        mass = [1., ] * ndim
        
    nx, ny, nz, nstates = psi.shape
    
    # kx = 2. * np.pi * fftfreq(nx, dx)
    # ky = 2. * np.pi * fftfreq(ny, dy)
    # kz = 2. * np.pi * fftfreq(nz, dz)

    # dx = interval(x)
    # dy = interval(y)
    # dz = interval(z)

    #vpsi = [vmat[i] * psi[i] for i in range(nstates)]

    # T |psi> = - \grad^2/2m * psi(x) = k**2/2m * psi(k)
    # D\grad |psi> = D(x) * F^{-1} F
    psi_k = np.zeros((nx, ny, nz, nstates), dtype=complex)
    tpsi = np.zeros((nx, ny, nz, nstates), dtype=complex)
    # xpc_psi = np.zeros((nx, ny, nz, nstates), dtype=complex)

    # FFT
    # for i in range(nstates):
    #     tmp = psi[:, :, :, i]
    #     psi_k[:,:,i] = (tmp)
    
    psi_k = np.fft.fftn(psi, axes=(0, 1, 2)) 
    
    # cavity momentum operator operate on the WF
    kzpsi = np.einsum('k, ijkn -> ijkn', kz, psi_k)
    
    # kypsi = np.einsum('j, ijn -> ijn', ky, psi_k)

    # dxpsi = np.zeros((nx, ny, nstates), dtype=complex)
    dzpsi = np.fft.ifftn(kzpsi, axes=(0,1,2))

    # for i in range(nstates):
    #     dxpsi[:,:,i] = ifft2(kxpsi[:,:,i])
    #     dypsi[:,:,i] = ifft2(kypsi[:,:,i])

    # kinetic energy operator
    if coordinates == 'linear':

        mx, my, mz = mass

        # T = np.einsum('i, j -> ij', kx**2/2./mx, ky**2/2./my)
        Kx, Ky, Kz = np.meshgrid(kx, ky, kz)

        T = Kx**2/2./mx + Ky**2/2./my + Kz**2/2/mz

        tpsi_k = np.einsum('ijk, ijkn -> ijkn', T, psi_k)

        # for i in range(nstates):
        #     tpsi[:,:,i] = ifft2(tpsi_k[:,:,i])
        
        tpsi = np.fft.ifftn(tpsi_k, axes=(0,1,2))
        
        # X, Y, Z = np.mgrids(x, y, z)
        x_pc_psi = g * np.einsum('i, ijkn -> ijkn', x, dzpsi)
        # np.einsum('ijmn, ijn -> ijm', nac_y, dypsi) * 1j/my # array with size nstates

    elif coordinates == 'curvilinear':


        # for i in range(nx):
        #     for j in range(ny):
        #         #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

        #         for k in range(nstates):
        #             tpsi[k][i, j] = G.dot(np.array([dxpsi[k][i, j], dypsi[k][i, j]]))

        # for n in range(nstates):
            # tpsi[:, :, n] = KEO(psi[:, :, n], kx, ky, G)
        raise NotImplementedError('Curvilinear coordinates not implemented.')


    # NACs operate on the WF


    # nac_x, nac_y = nonadiabatic_couplings(X, Y, nstates)


    # for i in range(nx):
    #     for j in range(ny):
            # tmp1 = np.array([dxpsi[k][i,j] for k in range(nstates)])
            # tmp2 = np.array([dypsi[k][i,j] for k in range(nstates)])

    vpsi = np.einsum('ijkmn, ijkn -> ijkm', vmat, psi)

    # kinetic energy operator for linear coordinates

    # ... NAC_x * G11 * P_x + NAC_y * G22 * P_y + cross terms
    hpsi = tpsi + vpsi + x_pc_psi    

    return -1j * hpsi
    
    
# def hpsi(psi, vmat, coordinates='linear', \
#          mass=None, G=None):
#     """
#     evaluate H \psi with the full vibronic Hamiltonian
#     input:
#         v: 1d array, adiabatic surfaces
#         d: nonadiabatic couplings, matrix
#         G: 4d array N, N, Nx, Ny (N: # of states)
#     output:
#         hpsi: H operates on psi
#     """
    
#     nx, ny, nz, nel = psi.shape
    
#     kx = 2. * np.pi * fftfreq(nx, dx)
#     ky = 2. * np.pi * fftfreq(ny, dy)
#     kz = 2. * np.pi * fftfreq(ny, dy)
#     # v |psi>
# #    for i in range(len(x)):
# #        for j in range(len(y)):
# #            v_tmp = np.diagflat(vmat[:][i,j])
# #            array_tmp = np.array([psi[0][i, j], psi[1][i, j]])
# #            vpsi = vmat.dot(array_tmp)
#     # if nstates != len(vmat):
#     #     sys.exit('Error: number of electronic states does not match the length of PPES matrix!')



#     #vpsi = [vmat[i] * psi[i] for i in range(nstates)]

#     # T |psi> = - \grad^2/2m * psi(x) = k**2/2m * psi(k)
#     # D\grad |psi> = D(x) * F^{-1} F
#     psi_k = np.zeros((nx, ny, nstates), dtype=complex)

#     tpsi = np.zeros((nx, ny, nstates), dtype=complex)
#     nacpsi = np.zeros((nx, ny, nstates), dtype=complex)

#     # FFT
#     for i in range(nstates):
#         tmp = psi[:,:,i]
#         psi_k[:,:,i] = fft2(tmp)

#     # momentum operator operate on the WF
#     kxpsi = np.einsum('i, ijn -> ijn', kx, psi_k)
#     kypsi = np.einsum('j, ijn -> ijn', ky, psi_k)

#     dxpsi = np.zeros((nx, ny, nstates), dtype=complex)
#     dypsi = np.zeros((nx, ny, nstates), dtype=complex)

#     for i in range(nstates):
#         dxpsi[:,:,i] = ifft2(kxpsi[:,:,i])
#         dypsi[:,:,i] = ifft2(kypsi[:,:,i])

#     # kinetic energy operator
#     if coordinates == 'linear':

#         mx, my = mass

#         # T = np.einsum('i, j -> ij', kx**2/2./mx, ky**2/2./my)
#         Kx, Ky = np.meshgrid(kx, ky)

#         T = Kx**2/2./mx + Ky**2/2./my

#         tpsi_k = np.einsum('ij, ijn -> ijn', T, psi_k)

#         for i in range(nstates):
#             tpsi[:,:,i] = ifft2(tpsi_k[:,:,i])

#         nacpsi = np.einsum('ijmn, ijn -> ijm', nac_x, dxpsi) * 1j/mx + \
#         np.einsum('ijmn, ijn -> ijm', nac_y, dypsi) * 1j/my # array with size nstates

#     elif coordinates == 'curvilinear':


#         # for i in range(nx):
#         #     for j in range(ny):
#         #         #G = metric_tensor(x[i], y[j]) # 2 x 2 matrix metric tensor at (x, y)

#         #         for k in range(nstates):
#         #             tpsi[k][i, j] = G.dot(np.array([dxpsi[k][i, j], dypsi[k][i, j]]))

#         for n in range(nstates):
#             tpsi[:, :, n] = KEO(psi[:, :, n], kx, ky, G)


#     # NACs operate on the WF


#     # nac_x, nac_y = nonadiabatic_couplings(X, Y, nstates)


#     # for i in range(nx):
#     #     for j in range(ny):
#             # tmp1 = np.array([dxpsi[k][i,j] for k in range(nstates)])
#             # tmp2 = np.array([dypsi[k][i,j] for k in range(nstates)])





#     # kinetic energy operator for linear coordinates






#     # ... NAC_x * G11 * P_x + NAC_y * G22 * P_y + cross terms
#     hpsi = tpsi + vpsi + nacpsi

#     return -1j * hpsi


# def dms(x, y, nel):
#     # dipole moment surface
#     dip = np.zeros((nel, nel))
    
#     pass

if __name__ == '__main__':

    from pyqed.phys import gwp
    from pyqed import wavenumber
    
    x = np.linspace(-4, 4, 64)
    y = np.linspace(-4, 4, 64)
    q = np.linspace(-4, 4, 64)

    nx = len(x)
    ny = len(y)
    nq = len(q)
    # mol = DHO(x)
    # mol.dpes(d=2, E0=2)


    # pol = VibronicPolariton(mol, cav)
    # pol.dpes(g=0.05)
    # pol.ppes()

    # pol.draw_surfaces(n=4, representation='adiabatic')
    # pol.product_state(0, 0, 0)

    # from pyqed.models.vibronic import LVC2

    # mol = LVC2(x, y, mass=[1,1])
    from pyqed import wavenumber
    from pyqed.models.jahn_teller import JahnTeller
    
    omega = 660. * wavenumber
    Eshift = np.array([0.0, 7.0, 7.0])/au2ev
    kappa = 2.2 * omega # inter-state coupling lambda

    # tuning_mode = Mode(omega, couplings=[[[1, 1], kappa], \
    #                                      [[2, 2], -kappa]], truncate=24)

    # coupling_mode = Mode(omega, [[[1, 2], kappa]], truncate=24)

    # modes =  [tuning_mode, coupling_mode]

    mol = JahnTeller(Eshift, omega, kappa)
                       
    # mol.plot_apes()
    
    
    cav = Cavity(1000 * wavenumber, 3, x=x)

    
    # VSC
    pol = VSC(mol, cav)
    pol.build_dpes(g=[0.1, 0.1])
    
    pol.ppes()
    # pol.plot_surface(1, representation='diabatic')

    psi0 = np.zeros((len(x), len(y), len(q), pol.nstates), dtype=complex)
    for i in range(nx):
        for j in range(ny):
            for k in range(nq):
                psi0[i, j, k, 1] = gwp([x[i], y[j], q[k]])
    
    # pol.plot_ground_state()
    
    r = pol.run(psi0=psi0, dt=0.001, nt=20, method='rk4')
    
    print(pol.rdm_el(r['psi'][0]))

    print(pol.rdm_el(r['psi'][-1]))
    