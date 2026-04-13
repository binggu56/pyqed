#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native AO-based unrestricted Hartree-Fock.
"""

import logging
import math
from functools import reduce

import numpy as np
from opt_einsum import contract
from scipy.linalg import eigh

from pyqed import dag, dagger


def _split_nelec(nelec, spin):
    spin = int(round(spin))
    nelec = int(round(nelec))
    if (nelec + spin) % 2 != 0:
        raise ValueError(
            f"Incompatible electron count/spin: nelec={nelec}, spin={spin}."
        )

    nalpha = (nelec + spin) // 2
    nbeta = nelec - nalpha
    if nalpha < 0 or nbeta < 0:
        raise ValueError(
            f"Invalid unrestricted occupation: nalpha={nalpha}, nbeta={nbeta}."
        )
    return nalpha, nbeta


class UHF:
    def __init__(self, mol, init_guess='h1e'):
        self.mol = mol
        self.max_cycle = 100
        self.init_guess = init_guess
        self.conv_tol = 1e-8
        self.conv_tol_dm = 1e-6
        self.damping = 0.0

        self.na = None
        self.nb = None
        self.nocc = None
        self.nvir = None

        self.nao = self.nmo = mol.nao
        self.nso = 2 * mol.nao
        self.nelec = mol.nelec

        self.mo_occ = None
        self.mo_coeff = None
        self.e_tot = None
        self.e_nuc = None
        self.hcore = None
        self.vhf = None
        self.dm = None
        self.mo_energy = None
        self.converged = False

        self.eri = mol.eri

    def run(self, dm0=None, **kwargs):
        out = unrestricted_hartree_fock(
            self.mol,
            dm0=dm0,
            init_guess=kwargs.get('init_guess', self.init_guess),
            max_cycle=kwargs.get('max_cycle', self.max_cycle),
            tol=kwargs.get('tol', kwargs.get('conv_tol', self.conv_tol)),
            d_tol=kwargs.get('d_tol', kwargs.get('conv_tol_dm', self.conv_tol_dm)),
            damping=kwargs.get('damping', self.damping),
            verbose=kwargs.get('verbose', 0),
        )

        self.e_tot = out['e_tot']
        self.e_nuc = out['e_nuc']
        self.mo_energy = out['mo_energy']
        self.mo_coeff = out['mo_coeff']
        self.mo_occ = out['mo_occ']
        self.hcore = out['hcore']
        self.vhf = out['vhf']
        self.dm = out['dm']
        self.converged = out['converged']
        self.na = out['na']
        self.nb = out['nb']
        self.nocc = (self.na, self.nb)
        self.nvir = (self.nao - self.na, self.nao - self.nb)
        return self

    def make_rdm1(self):
        if self.mo_coeff is None or self.mo_occ is None:
            raise ValueError("No converged orbitals are available yet.")
        return np.array((
            make_rdm1(self.mo_coeff[0], self.mo_occ[0]),
            make_rdm1(self.mo_coeff[1], self.mo_occ[1]),
        ))

    def get_ovlp(self):
        return self.mol.overlap

    def get_hcore(self):
        hcore = self.mol.hcore
        return (hcore, hcore)

    def get_hcore_mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        hcore = self.mol.hcore
        return (
            dag(mo_coeff[0]) @ hcore @ mo_coeff[0],
            dag(mo_coeff[1]) @ hcore @ mo_coeff[1],
        )

    def get_veff(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        return get_veff(self.mol, dm)

    def get_fock(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        veff = self.get_veff(dm)
        hcore = self.mol.hcore
        return np.array((hcore + veff[0], hcore + veff[1]))

    def get_j(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        dm = np.asarray(dm)
        return get_j(self.mol, dm[0] + dm[1])

    def get_k(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        dm = np.asarray(dm)
        return np.array((get_k(self.mol, dm[0]), get_k(self.mol, dm[1])))

    def get_jk(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        dm = np.asarray(dm)
        return self.get_j(dm), self.get_k(dm)

    def energy_elec(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        vhf = self.get_veff(dm)
        return energy_elec(dm, self.mol.hcore, vhf)

    def energy_nuc(self):
        if self.e_nuc is not None:
            return self.e_nuc
        return self.mol.energy_nuc()

    def get_eri(self, representation='ao'):
        if representation == 'ao':
            return self.eri
        if representation == 'mo':
            if self.mo_coeff is None:
                raise ValueError("Run UHF before requesting MO integrals.")
            return self.get_eri_mo()
        raise ValueError("representation must be 'ao' or 'mo'.")

    def get_eri_mo(self, mo_coeff=None, notation='chem'):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise ValueError("Run UHF before requesting MO integrals.")

        ca, cb = mo_coeff
        aa = contract('pqrs,pi,qj,rk,sl->ijkl', self.eri, ca.conj(), ca, ca.conj(), ca)
        bb = contract('pqrs,pi,qj,rk,sl->ijkl', self.eri, cb.conj(), cb, cb.conj(), cb)
        ab = contract('pqrs,pi,qj,rk,sl->ijkl', self.eri, ca.conj(), ca, cb.conj(), cb)
        ba = contract('pqrs,pi,qj,rk,sl->ijkl', self.eri, cb.conj(), cb, ca.conj(), ca)

        if notation == 'phys':
            aa = np.transpose(aa, (0, 2, 1, 3))
            bb = np.transpose(bb, (0, 2, 1, 3))
            ab = np.transpose(ab, (0, 2, 1, 3))
            ba = np.transpose(ba, (0, 2, 1, 3))
        elif notation != 'chem':
            raise ValueError("notation must be 'chem' or 'phys'.")

        return np.array(((aa, ab), (ba, bb)), dtype=complex)


def get_j(mol, dm):
    return np.einsum('rs,pqrs->pq', dm, mol.eri, optimize=True)


def get_k(mol, dm):
    return np.einsum('rs,psrq->pq', dm, mol.eri, optimize=True)


def get_veff(mol, dm, dm_last=None, vhf_last=None):
    dm = np.asarray(dm)
    if dm.shape[0] != 2:
        raise ValueError("UHF density matrix must have shape (2, nao, nao).")

    if dm_last is not None:
        ddm = dm - np.asarray(dm_last)
        dvhf = get_veff(mol, ddm)
        return dvhf + np.asarray(vhf_last)

    j_tot = get_j(mol, dm[0] + dm[1])
    k_a = get_k(mol, dm[0])
    k_b = get_k(mol, dm[1])
    return np.array((j_tot - k_a, j_tot - k_b))


def energy_elec(dm, h1e=None, vhf=None):
    dm = np.asarray(dm)
    if dm.shape[0] != 2:
        raise ValueError("UHF density matrix must have shape (2, nao, nao).")

    if h1e is None:
        raise ValueError("h1e is required.")
    if vhf is None:
        raise ValueError("vhf is required.")

    if np.asarray(h1e).ndim == 2:
        h1e = np.array((h1e, h1e))
    vhf = np.asarray(vhf)

    e1 = (
        np.einsum('ij,ji->', h1e[0], dm[0]).real
        + np.einsum('ij,ji->', h1e[1], dm[1]).real
    )
    e2 = 0.5 * (
        np.einsum('ij,ji->', vhf[0], dm[0]).real
        + np.einsum('ij,ji->', vhf[1], dm[1]).real
    )
    return e1 + e2


def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mocc = mo_coeff[:, mo_occ > 0]
    return np.dot(mocc * mo_occ[mo_occ > 0], mocc.conj().T)


def _orthogonalizer(overlap):
    s, u = eigh(overlap)
    if np.min(s) <= 0.0:
        raise ValueError("Overlap matrix is not positive definite.")
    return u @ np.diagflat(s ** (-0.5)) @ dagger(u)


def _diagonalize(fock, orth):
    fock_orth = reduce(np.dot, (dagger(orth), fock, orth))
    mo_energy, c_orth = eigh(fock_orth)
    mo_coeff = np.real_if_close(orth @ c_orth)
    return mo_energy, mo_coeff


def _initial_guess(hcore, orth, na, nb, dm0=None, init_guess='h1e'):
    mo_occ_a = np.zeros(hcore.shape[0])
    mo_occ_b = np.zeros(hcore.shape[0])
    mo_occ_a[:na] = 1.0
    mo_occ_b[:nb] = 1.0

    if dm0 is not None:
        dm = np.asarray(dm0, dtype=float)
        if dm.shape != (2, hcore.shape[0], hcore.shape[0]):
            raise ValueError("dm0 must have shape (2, nao, nao).")
        return dm, (mo_occ_a, mo_occ_b)

    if init_guess not in ('h1e', 'hcore'):
        raise ValueError("Only init_guess='h1e'/'hcore' is currently supported.")

    mo_energy, mo_coeff = _diagonalize(hcore, orth)
    dm_a = make_rdm1(mo_coeff, mo_occ_a)
    dm_b = make_rdm1(mo_coeff, mo_occ_b)
    return np.array((dm_a, dm_b)), (mo_occ_a, mo_occ_b)


def _make_diis_extrapolator(max_diis=6):
    error_a = np.zeros((max_diis, 1, 1))
    error_b = np.zeros((max_diis, 1, 1))
    fock_a = np.zeros((max_diis, 1, 1))
    fock_b = np.zeros((max_diis, 1, 1))

    def diis(fock_pair, dens_pair, overlap, orth, iteration):
        fa, fb = fock_pair
        da, db = dens_pair

        if iteration <= 1:
            return np.array((fa, fb)), 0.0

        nonlocal error_a, error_b, fock_a, fock_b
        if error_a.shape[1:] != fa.shape:
            error_a = np.zeros((max_diis,) + fa.shape)
            error_b = np.zeros((max_diis,) + fb.shape)
            fock_a = np.zeros((max_diis,) + fa.shape)
            fock_b = np.zeros((max_diis,) + fb.shape)

        for k in reversed(range(1, min(iteration, max_diis))):
            error_a[k] = error_a[k - 1]
            error_b[k] = error_b[k - 1]
            fock_a[k] = fock_a[k - 1]
            fock_b[k] = fock_b[k - 1]

        ea = fa @ da @ overlap - overlap @ da @ fa
        eb = fb @ db @ overlap - overlap @ db @ fb
        error_a[0] = orth.T @ ea @ orth
        error_b[0] = orth.T @ eb @ orth
        fock_a[0] = fa
        fock_b[0] = fb

        diis_error = max(
            np.max(np.abs(error_a[0])),
            np.max(np.abs(error_b[0])),
        )

        bsize = min(iteration, max_diis) - 1
        bmat = -1.0 * np.ones((bsize + 1, bsize + 1))
        rhs = np.zeros(bsize + 1)
        bmat[bsize, bsize] = 0.0
        rhs[bsize] = -1.0
        for i in range(bsize):
            for j in range(bsize):
                bmat[i, j] = (
                    np.trace(error_a[i] @ error_a[j])
                    + np.trace(error_b[i] @ error_b[j])
                )

        try:
            coeff = np.linalg.solve(bmat, rhs)
        except np.linalg.LinAlgError:
            return np.array((fa, fb)), diis_error

        fa_diis = np.zeros_like(fa)
        fb_diis = np.zeros_like(fb)
        for c, fa_hist, fb_hist in zip(coeff[:-1], fock_a, fock_b):
            fa_diis += c * fa_hist
            fb_diis += c * fb_hist

        return np.array((fa_diis, fb_diis)), diis_error

    return diis


def unrestricted_hartree_fock(
    mol,
    dm0=None,
    init_guess='h1e',
    max_cycle=50,
    tol=1e-8,
    d_tol=1e-6,
    damping=0.0,
    verbose=0,
):
    overlap = mol.overlap
    hcore = mol.hcore
    e_nuc = mol.energy_nuc()
    orth = _orthogonalizer(overlap)
    na, nb = _split_nelec(mol.nelec, getattr(mol, 'spin', 0))

    dm, mo_occ = _initial_guess(hcore, orth, na, nb, dm0=dm0, init_guess=init_guess)
    diis = _make_diis_extrapolator()

    e_last = None
    converged = False

    if verbose:
        logging.info("\n {:4s} {:16s} {:12s} {:12s}".format(
            'iter', 'total energy', 'de', 'ddm'))

    for scf_iter in range(max_cycle):
        vhf = get_veff(mol, dm)
        fock = np.array((hcore + vhf[0], hcore + vhf[1]))
        fock, diis_error = diis(fock, dm, overlap, orth, scf_iter)

        if damping:
            fock = damping * fock + (1.0 - damping) * np.array((hcore + vhf[0], hcore + vhf[1]))

        mo_energy_a, mo_coeff_a = _diagonalize(fock[0], orth)
        mo_energy_b, mo_coeff_b = _diagonalize(fock[1], orth)

        dm_new = np.array((
            make_rdm1(mo_coeff_a, mo_occ[0]),
            make_rdm1(mo_coeff_b, mo_occ[1]),
        ))

        vhf_new = get_veff(mol, dm_new)
        e_elec = energy_elec(dm_new, hcore, vhf_new)
        e_tot = e_elec + e_nuc
        de = None if e_last is None else e_tot - e_last
        ddm = np.linalg.norm(dm_new - dm)

        if verbose:
            logging.info("{:3d} {:16.10f} {:12.4e} {:12.4e}".format(
                scf_iter,
                e_tot,
                0.0 if de is None else de,
                ddm,
            ))

        dm = dm_new
        vhf = vhf_new

        if e_last is not None and abs(de) < tol and ddm < d_tol:
            converged = True
            break

        e_last = e_tot

        if diis_error < max(math.sqrt(tol), 1e-10):
            damping = 0.0

    return {
        'converged': converged,
        'e_tot': e_tot,
        'e_nuc': e_nuc,
        'mo_energy': (mo_energy_a, mo_energy_b),
        'mo_coeff': (mo_coeff_a, mo_coeff_b),
        'mo_occ': mo_occ,
        'hcore': (hcore, hcore),
        'vhf': vhf,
        'dm': dm,
        'na': na,
        'nb': nb,
    }
