#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared KS-SCF helpers for AO-based DFT.
"""

import logging
import numpy as np
from scipy.linalg import eigh

from .grid import (
    AOGrid,
    build_gga_potential_matrix,
    build_local_potential_matrix,
    density_gradient_on_grid,
    density_on_grid,
    xc_energy_from_grid,
)
from .xc import eval_xc, hybrid_coeff, needs_gradients, xc_type


def make_rdm1(mo_coeff, mo_occ):
    """
    One-particle density matrix in the AO basis.
    """
    mo_occ = np.asarray(mo_occ)
    mocc = mo_coeff[:, mo_occ > 0]
    return np.dot(mocc * mo_occ[mo_occ > 0], mocc.conj().T)


def get_j(mol, dm):
    """
    Coulomb matrix in the AO basis.
    """
    from pyqed.qchem.hf.rhf import get_jk

    return get_jk(mol, dm, eri_factors=getattr(mol, 'eri_factors', None))[0]


def get_k(mol, dm):
    """
    Exchange matrix in the AO basis.
    """
    from pyqed.qchem.hf.rhf import get_jk

    return get_jk(mol, dm, eri_factors=getattr(mol, 'eri_factors', None))[1]


def build_xc(dm, grid, xc='lda_x'):
    """
    Build grid density, XC potential matrix, and XC energy.
    """
    rho = density_on_grid(dm, grid.ao)
    if xc_type(xc) == 'LDA':
        eps_xc, v_xc = eval_xc(rho, xc=xc)
        vxc_mat = build_local_potential_matrix(v_xc, grid.weights, grid.ao)
    else:
        if getattr(grid, 'ao_grad', None) is None:
            raise ValueError(f"XC functional '{xc}' requires AO gradients on the grid.")
        rho_grad = density_gradient_on_grid(dm, grid.ao, grid.ao_grad)
        eps_xc, (vrho, vsigma) = eval_xc(rho, xc=xc, grad_rho=rho_grad)
        vxc_mat = build_gga_potential_matrix(
            vrho,
            vsigma,
            rho_grad,
            grid.weights,
            grid.ao,
            grid.ao_grad,
        )
    exc = xc_energy_from_grid(rho, eps_xc, grid.weights)
    return rho, exc, vxc_mat


def ensure_grid_for_xc(mol, grid, xc):
    """
    Make sure the numerical grid carries all information required by ``xc``.
    """
    if not needs_gradients(xc):
        return grid

    if getattr(grid, 'ao_grad', None) is not None:
        return grid

    if getattr(grid, 'coords', None) is not None:
        grid.attach_gradients(mol)
        return grid

    return AOGrid.atom_centered(mol, with_grad=True)


def diagonalize(fock, overlap):
    """
    Solve the generalized eigenvalue problem FC = SCE.
    """
    return eigh(fock, overlap)


def init_guess_by_hcore(hcore, overlap, mo_occ):
    """
    Core Hamiltonian initial guess.
    """
    mo_energy, mo_coeff = diagonalize(hcore, overlap)
    dm = make_rdm1(mo_coeff, mo_occ)
    return mo_energy, mo_coeff, dm


def ks_energy(dm, hcore, j, exc, vxc_mat, e_nuc=0.0, k=None, hyb=0.0):
    """
    Restricted Kohn-Sham total energy.

    The direct KS expression is

        E = Tr[D H_core] + 1/2 Tr[D J] - hyb/4 Tr[D K] + E_xc + E_nuc

    so the explicit XC potential matrix is not part of the final energy.
    ``vxc_mat`` is kept in the signature for API compatibility with callers
    that already build it alongside ``exc``.
    """
    e_one = np.einsum('ij,ji->', hcore, dm).real
    e_coul = 0.5 * np.einsum('ij,ji->', j, dm).real
    e_hyb = 0.0
    if k is not None and hyb != 0.0:
        e_hyb = -0.25 * hyb * np.einsum('ij,ji->', k, dm).real
    return e_one + e_coul + e_hyb + exc + e_nuc


def run_rks(mol, grid, dm0=None, init_guess='hcore', xc='lda_x',
            max_cycle=50, conv_tol=1e-8, damping=0.25, verbose=0):
    """
    Restricted Kohn-Sham SCF using AO integrals and a precomputed numerical grid.
    """
    if grid.nao != mol.nao:
        raise ValueError("grid.nao must match mol.nao.")

    grid = ensure_grid_for_xc(mol, grid, xc)

    overlap = mol.overlap
    hcore = mol.hcore
    e_nuc = getattr(mol, 'e_nuc', None)
    if e_nuc is None and hasattr(mol, 'energy_nuc'):
        e_nuc = mol.energy_nuc()
    if e_nuc is None:
        e_nuc = 0.0
    hyb = hybrid_coeff(xc)
    nocc = mol.nelec // 2

    mo_occ = np.zeros(mol.nao)
    mo_occ[:nocc] = 2.0

    if dm0 is not None:
        dm = np.asarray(dm0)
        mo_energy, mo_coeff = diagonalize(hcore, overlap)
    elif init_guess == 'hcore':
        mo_energy, mo_coeff, dm = init_guess_by_hcore(hcore, overlap, mo_occ)
    else:
        raise ValueError("Only init_guess='hcore' is currently supported.")

    converged = False
    e_last = None

    if verbose:
        logging.info("\n {:4s} {:16s} {:12s}".format('iter', 'total energy', 'de'))

    for scf_iter in range(max_cycle):
        j = get_j(mol, dm)
        k = get_k(mol, dm) if hyb != 0.0 else None
        rho, exc, vxc_mat = build_xc(dm, grid, xc=xc)
        fock = hcore + j + vxc_mat
        if k is not None:
            fock = fock - 0.5 * hyb * k

        mo_energy, mo_coeff = diagonalize(fock, overlap)
        dm_new = make_rdm1(mo_coeff, mo_occ)
        j_new = get_j(mol, dm_new)
        k_new = get_k(mol, dm_new) if hyb != 0.0 else None
        rho_new, exc_new, vxc_mat_new = build_xc(dm_new, grid, xc=xc)
        e_tot = ks_energy(
            dm_new,
            hcore,
            j_new,
            exc_new,
            vxc_mat_new,
            e_nuc=e_nuc,
            k=k_new,
            hyb=hyb,
        )
        de = None if e_last is None else e_tot - e_last

        if verbose:
            logging.info("{:3d} {:16.10f} {:12.4e}".format(
                scf_iter, e_tot, 0.0 if de is None else de))

        if e_last is not None and abs(de) < conv_tol:
            converged = True
            dm = dm_new
            j = j_new
            k = k_new
            rho = rho_new
            exc = exc_new
            vxc_mat = vxc_mat_new
            fock = hcore + j + vxc_mat
            if k is not None:
                fock = fock - 0.5 * hyb * k
            break

        dm = (1.0 - damping) * dm_new + damping * dm
        e_last = e_tot

    return {
        'converged': converged,
        'e_tot': e_tot,
        'mo_energy': mo_energy,
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
        'dm': dm,
        'hcore': hcore,
        'j': j,
        'k': k,
        'vxc': vxc_mat,
        'fock': fock,
        'rho': rho,
        'exc': exc,
        'hyb': hyb,
        'grid': grid,
    }
