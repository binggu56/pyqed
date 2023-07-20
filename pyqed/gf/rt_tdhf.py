#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:05:25 2022

Real time time-dependent Hartree-Fock

@author: Bing Gu
"""
import numpy as np
import scipy.linalg as linalg
from scipy.optimize import newton

from pyscf.lib import logger
import pyscf.ao2mo
import pyscf
from functools import reduce


from lime.phys import eig_asymm, is_positive_def, dag
from lime.optics import Pulse


def center_of_mass(mol):
    mass = mol.atom_mass_list()
    coords = mol.atom_coords()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    # coords = coords - mass_center
    return mass_center


def charge_center(mol):
    charge_center = (np.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
                 / mol.atom_charges().sum())
    return charge_center

def _tdhf(mo_coeff, hcore, r, pulse):
    pass

def self_energy_hf(eri, rdm1):
    """
    HF self energy

    The Hartree term reads
        \Sigma_{H, pq} = v_{pq, rs} \sum_\sigma P^\sigma_{rs}

    Make sure both quantities are represented by the same basis set.

    Parameters
    ----------
    eri : TYPE
        DESCRIPTION.
    rdm1 : TYPE
        DESCRIPTION.

    Returns
    -------
    S : TYPE
        DESCRIPTION.

    """

    sigmaH = 2. * np.einsum('ijkl, kl -> ij', eri, rdm1)
    sigmax = - np.einsum('ijkl, kl -> ik', eri, rdm1)

    return sigmaH + sigmax

class TDHF:
    def __init__(mf, pulse):
        pass

if __name__ == '__main__':
    from pyscf import scf, gto
    from lime.units import au2fs, au2ev
    import proplot as plt

    mol = gto.Mole()
    mol.verbose = 3
    #mol.atom = [['Ne' , (0., 0., 0.)]]
    #mol.basis = {'Ne': '6-31G'}
    # This is from G2/97 i.e. MP2/6-31G*
    mol.atom = [['H' , (0,      0., 0.)],
                ['H', (1.1, 0., 0.)]]
                # ['F' , (0.91, 0., 0.)]]


    mol.basis = 'STO-3G'
    mol.build()

    mol.set_common_origin(charge_center(mol))

    mf = scf.RHF(mol)

    mf.kernel()

    # 1-particle RDM in AOs
    C = mf.mo_coeff[:, mf.mo_occ > 0]
    rdm1 = np.conj(C).dot(C.T)

    print(mf.mo_energy)
    hcore = mf.get_hcore()

    r = mol.intor('int1e_r') # AO-matrix elements of r
    eri = mol.intor('int2e')


    Nt = 2000
    dt = 0.02/au2fs
    t0 = -8/au2fs
    t = t0
    ts = t0 + np.arange(Nt) * dt
    pulse = Pulse(omegac=0.1, tau=2/au2fs, amplitude=0.01)

    out = np.zeros(Nt, dtype=complex)
    for k in range(Nt):

        sigma = self_energy_hf(eri, rdm1)

        # fock matrix including the drive in the electric dipole gauge
        f = hcore + sigma + r[0, :, :] * pulse.efield(t)

        # propagate
        u = linalg.expm(-1j * f * dt)
        rdm1 = u.dot(rdm1.dot(dag(u)))

        t += dt
        out[k] = rdm1[0, 1]

    fig, ax = plt.subplots()
    ax.plot(ts, out.real)
    ax.plot(ts, out.imag)

    # omega = np.linspace(-1.5, 1.5, 100)
    from lime.fft import dft, fft
    # g = dft(ts, out, omega)
    omega, g = fft(out, ts)

    fig, ax = plt.subplots()
    ax.plot(omega, np.abs(g))
    ax.format(xlim=(0.6, 1.5), ylim=(0, 50))




