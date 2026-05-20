#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:04:15 2024

@author: Bing Gu (gubing@westlake.edu.cn)

NARG for interacting bosons
"""

from pyqed.dvr.dvr_1d import SineDVR, BesselDVR, SincDVR
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, kron, csr_matrix, diags_array
from pyqed import pauli




def direct_product():
    nz = 128*2
    dvr_z = SineDVR(-6, 6, npts=nz)
    # dvr_z = SincDVR(20, nz)
    z = dvr_z.x
    Kz = dvr_z.t()
    Iz = eye(nz)

    # E, U = dvr.run(v, 6)
    # print(z)
    # x = np.linspace(-6,6,200)

    # fig, ax = plt.subplots()
    # ax.plot(x, dvr.basis(x, 32))

    nr = 100
    # nstates = 2
    dvr = BesselDVR(npts=nr, R=10, l=0)
    Kr = dvr.t(l=0)

    # print(Kr.shape)
    Ir = eye(nr)

    def v(r, z):
        return -1/np.sqrt(r**2 + z**2)

    r = dvr.x

    R, Z = np.meshgrid(r, z, indexing='ij')

    _v = v(R, Z)



    H = kron(Kr, Iz) + kron(Ir, Kz) + diags_array(_v.flatten())


    E, U = eigsh(csr_matrix(H), k=3, which='SA')

    print(E)

    fig, ax = plt.subplots()
    ax.imshow(U[:,0].reshape(nr, nz))


def ldr():
    nz = 80
    dvr_z = SineDVR(-6, 6, npts=nz)
    # dvr_z = SincDVR(20, nz)
    z = dvr_z.x
    Kz = dvr_z.t()
    nstates = 10

    # print(dvr.get_weights(l=0))
    # def v(x):
    #     return -1/np.sqrt(x**2 + 0.00001)

    nr = 128
    # nstates = 2
    dvr = BesselDVR(npts=nr, R=10, l=0)
    r = dvr.x

    orb_energy = np.zeros((nz, nstates))
    orb = np.zeros((nz, nr, nstates))

    for n in range(nz):

        _v = -1/np.sqrt(r**2 + z[n]**2)
        E, U = dvr.run(_v, k=nstates)

        # print(np.vdot(U[:, 0], U[:, 0]))
        # print(E)
        orb_energy[n] = E
        orb[n] = U

        if n % 20 == 0:
            fig, ax = plt.subplots()
            ax.plot(r, U[:,1], '-o')
    # # ax.plot(r, np.sqrt(r)*np.exp(-r))



    #### build electronic integrals

    Kz = dvr_z.t()

    V = np.diag(orb_energy.flatten())


    # transversal overlap matrix
    S = np.zeros((nz, nz, nstates, nstates))

    for n in range(nz):
        S[n, n] = np.eye(nstates)

    for i in range(nz):
        for j in range(i):

            for a in range(nstates):
                for b in range(nstates):
                    S[i, j, a, b] = np.vdot(orb[i, :, a], orb[j, :, b])


            S[j, i] = S[i, j].conj().T

    print('E orb', orb_energy)

    # print(S)

    size = nz * nstates
    H = np.einsum('mn, mnba -> mbna', Kz, S).reshape(size, size) + V
    # print(H)


    E, U = eigsh(H, k=4, which='SA')
    print(E)

    return E


def NARG(
        L,
        D=10,
        nz=16,
        nstates=1,
        omegas=None,
        g=None,
        xmax=6.0,
        adaptive_box=False,
        box_safety=1.25,
        min_xmax=6.0):
    """
    Nonadiabatic RG for the coupled boson benchmark.

    Parameters
    ----------
    L : int
        Number of sites or modes.
    D : int, optional
        Number of adiabatic states kept at each iteration.
    nz : int, optional
        Number of DVR grid points for each mode.
    nstates : int, optional
        Number of final many-body eigenstates to return.
    omegas : array_like, optional
        Mode frequencies. Must contain at least ``L`` entries. If omitted,
        the module-level ``omegas`` value is used for backward compatibility.
    g : float, optional
        Coupling strength. If omitted, the module-level ``g`` value is used
        when available; otherwise 0.05 is used.
    xmax : float, optional
        Half-width of the SineDVR box when ``adaptive_box`` is false.
    adaptive_box : bool, optional
        If true, choose each DVR half-width from the harmonic turning point
        implied by the local quadratic PES and DVR mass.
    box_safety : float, optional
        Safety factor applied to the adaptive turning-point estimate.
    min_xmax : float, optional
        Minimum half-width used by the adaptive box.

    Returns
    -------
    E : ndarray
        Lowest many-body eigenvalues.
    U : ndarray
        Corresponding eigenvectors in the final retained basis.

    """

    if omegas is None:
        omegas = globals().get("omegas")
    if omegas is None:
        raise ValueError("omegas must be provided when no module-level omegas is defined.")
    omegas = np.asarray(omegas, dtype=float)
    if L < 3:
        raise ValueError("L must be at least 3 for the current recursive boson NARG implementation.")
    if len(omegas) < L:
        raise ValueError(f"omegas must contain at least L={L} entries.")
    if D > nz:
        raise ValueError(f"D={D} cannot exceed nz={nz} for this DVR implementation.")
    if g is None:
        g = globals().get("g", 0.05)
    g = float(g)

    def dvr_extent(omega, mass=1.0):
        if not adaptive_box:
            return float(xmax)
        nlevel = max(1, min(D, nz) - 1)
        # For H = p^2/(2m) + 0.5*k*x^2, with k represented here by omega,
        # E_n = (n + 1/2) sqrt(k/m), so x_turn^2 = (2n + 1) sqrt(k/m) / k.
        force_constant = float(omega)
        mass = float(mass)
        freq = np.sqrt(force_constant / mass)
        turning_point = np.sqrt((2 * nlevel + 1) * freq / force_constant)
        return max(float(min_xmax), float(box_safety) * turning_point)

    # discrete variable representation of the slow dof
    z_extent = dvr_extent(omegas[1], mass=1/omegas[1])
    dvr_z = SineDVR(-z_extent, z_extent, nz, mass=1/omegas[1])
    z = dvr_z.x
    Kz = dvr_z.t()

    # adiabatic representation of the fast dof
    x_extent = dvr_extent(omegas[0], mass=1.0)
    dvr = SineDVR(-x_extent, x_extent, npts=nz)
    x = dvr.x

    orb_energy = np.zeros((nz, D))
    orb = np.zeros((nz, nz, D))
    
    def potential(x, z):
        """
        x: fast
        z: slow

        The potential energy for the slow dof at a fixed slow dof.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return 0.5 * omegas[0] * x**2 + g * x**2 * z**2 + 0.5 * omegas[1] * z**2


    # solve the eigenstates for x at different z
    for n in range(nz):

        _v = potential(x, z[n])
        dvr.v = _v
        E, U = dvr.run(D)

        orb_energy[n] = E
        orb[n] = U



    #### build electronic integrals

    Kz = dvr_z.t()

    # build the Hamiltonian for the composte system (x, z)
    H = buildH(Kz, orb_energy, orb)

    # transform the coupling operators Z to the adiabatic representation
    # Z = np.einsum('mn, mnba -> mbna', Kz, S).reshape(size, size)

    # add another dof
    X = kron(np.diag(z), np.eye(D))

    z2_extent = dvr_extent(omegas[2], mass=1/omegas[2])
    dvr2 = SineDVR(-z2_extent, z2_extent, nz, mass=1/omegas[2])
    z2 = dvr2.x
    K2 = dvr2.t()


    E2 = np.zeros((nz, D))
    U2 = np.zeros((nz, nz * D, D))

    for n in range(nz):
        h = H + g * X**2 * z2[n]**2

        E2[n], U2[n] = eigsh(h, k=D, which='SA')

        E2[n] += 0.5 * omegas[2] * z2[n]**2 # analog of nuclear repulsion energy

    H = buildH(K2, E2, U2)

    for k in range(3, L):
        # add another dof
        zk_extent = dvr_extent(omegas[k], mass=1/omegas[k])
        dvr2 = SineDVR(-zk_extent, zk_extent, nz, mass=1/omegas[k])
        z = dvr2.x
        K2 = dvr2.t()
    
        X = kron(np.diag(z), np.eye(D))
    
    
        E2 = np.zeros((nz, D))
        U2 = np.zeros((nz, nz * D, D))
    
        for n in range(nz):
            h = H + g * X**2 * z[n]**2
    
            E2[n], U2[n] = eigsh(h, k=D, which='SA')
    
            E2[n] += 0.5 * omegas[k] * z[n]**2
    
        H = buildH(K2, E2, U2)

    # # add the fifth dof
    # dvr2 = SineDVR(-6, 6, nz, mass=1/omegas[4])
    # z = dvr2.x
    # K2 = dvr2.t()

    # X = kron(np.diag(z), np.eye(D))

    # E2 = np.zeros((nz, D))
    # U2 = np.zeros((nz, nz * D, D))

    # for n in range(nz):
    #     h = H + g * X**2 * z[n]**2

    #     E2[n], U2[n] = eigsh(h, k=D, which='SA')

    #     E2[n] += 0.5 * omegas[4] * z[n]**2

    # H = buildH(K2, E2, U2)


    # final diagonalization
    E, U = eigsh(H, k=nstates, which='SA')
    # print(E)

    return E, U

def buildH(Kz, E, U):
    """
    build the LDR Hamiltonian

    Parameters
    ----------
    Kz : TYPE
        kinetic energy operator of the slow dof
    E : array [nz, nstates]
        adiabatic energies
    U : TYPE
        DESCRIPTION.

    Returns
    -------
    H : TYPE
        DESCRIPTION.

    """

    nz, nstates = E.shape
    V = np.diag(E.flatten())

    # print('E orb', orb_energy)

    S = overlap(U)

    size = nz * nstates
    H = np.einsum('mn, mnba -> mbna', Kz, S).reshape(size, size) + V

    return H

def overlap(U):
    orb = U
    nz, _, nstates = U.shape
    # transversal overlap matrix
    S = np.zeros((nz, nz, nstates, nstates))

    for n in range(nz):
        S[n, n] = np.eye(nstates)

    for i in range(nz):
        for j in range(i):

            for a in range(nstates):
                for b in range(nstates):
                    S[i, j, a, b] = np.vdot(orb[i, :, a], orb[j, :, b])


            S[j, i] = S[i, j].conj().T
    return S


# class Site:
#     def __init__(self, H=None):
#         self.H = H
#         self.operator_dict = {'H': H}


if __name__=="__main__":
    
    from pyqed import logarithmic_discretize
    
    try:
        import ultraplot as plt
    except:
        import matplotlib.pyplot as plt

    
    s = 1.2
    nmodes = 10
    omegas = np.array(logarithmic_discretize(n=nmodes, base=s))
    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(6,3))
    # ax.bar(omegas, [1,]*len(omegas), width=0.01, edgecolor='None',color='k',align='center')
    # plt.show()
    
    # omegas = [1, 1/s, , 0.125, 0.0625, 0.03125]
    
    print('omegas = ', omegas)
    
    g = 0.1
    # NARG(D=1, nz=16, nstates=6)
    
    # for D in [2, 4, 8]:
    D = 20
    E1, U1 = NARG(L=nmodes, D=D, nz=32, nstates=16, omegas=omegas, g=g)
    np.savez('D{}_nmodes{}'.format(D, nmodes), E1, U1)
    # np.savez('normal_modes_nmodes{}'.format(D, nmodes), E1, U1)
    
    print('Eigenvalues = ', E1)
    
    ZPE = sum(omegas) * 0.5
    # E0 = [ZPE]
    # for j in range(6):
    #     E0 += list(j*omegas + ZPE)
    
    
    fig, (ax0, ax1) = plt.subplots(figsize=(5,4), nrows=2, sharex=True, hspace=0.)
    
    # ax0.bar(E0, [1, ] * len(E0), width=0.01)
    
    ax1.bar(E1, [1, ] * len(E1), width=0.01)
    
