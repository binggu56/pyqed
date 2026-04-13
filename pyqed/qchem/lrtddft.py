#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear-response restricted TDA/TDDFT for native pyqed mean-field objects.
"""

import numpy as np
from scipy.linalg import eigh

from .dft.scf import ensure_grid_for_xc
from .dft.xc import eval_fxc, hybrid_coeff, xc_type


def _eig_hermitian(a, nstates=None):
    e, x = eigh(a)
    if nstates is not None:
        e = e[:nstates]
        x = x[:, :nstates]
    return e, x


def _ov_blocks(mf):
    mo_energy = np.asarray(mf.mo_energy)
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)

    occidx = np.where(mo_occ > 0)[0]
    viridx = np.where(mo_occ == 0)[0]

    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:, viridx]
    return mo_energy, mo_coeff, occidx, viridx, orbo, orbv


def _lda_kernel_ovov(mf, orbo, orbv):
    grid = ensure_grid_for_xc(mf.mol, mf.grid, mf.xc)
    rho = np.einsum('gu,uv,gv->g', grid.ao, mf.dm, grid.ao, optimize=True).real
    fxc = eval_fxc(rho, mf.xc)

    rho_o = np.einsum('gu,ui->gi', grid.ao, orbo, optimize=True)
    rho_v = np.einsum('gu,ua->ga', grid.ao, orbv, optimize=True)
    rho_ov = np.einsum('gi,ga->gia', rho_o, rho_v, optimize=True)
    w_ov = np.einsum('gia,g->gia', rho_ov, 2.0 * grid.weights * fxc, optimize=True)
    return np.einsum('gia,gjb->iajb', rho_ov, w_ov, optimize=True)


def get_ab(mf):
    """
    Restricted singlet A/B matrices for linear-response TDDFT.

    Notes
    -----
    This initial native implementation supports:
    - RHF references as TDHF (no XC kernel)
    - RKS references with LDA-family kernels (`lda`, `svwn`, etc.)

    Hybrid GGA kernels such as B3LYP are not included yet in linear-response.
    """
    mo_energy, mo_coeff, occidx, viridx, orbo, orbv = _ov_blocks(mf)
    nocc = len(occidx)
    nvir = len(viridx)

    e_ia = mo_energy[viridx] - mo_energy[occidx, None]
    a = np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
    b = np.zeros_like(a)

    eri = mf.mol.eri
    eri_iajb = np.einsum(
        'pqrs,pi,qa,rj,sb->iajb',
        eri,
        orbo,
        orbv,
        orbo,
        orbv,
        optimize=True,
    )
    a += 2.0 * eri_iajb
    b += 2.0 * eri_iajb

    hyb = 0.0
    if hasattr(mf, 'xc'):
        hyb = hybrid_coeff(mf.xc)
    elif mf.__class__.__name__.lower() == 'rhf':
        hyb = 1.0

    if hyb != 0.0:
        eri_ijab = np.einsum(
            'pqrs,pi,qj,ra,sb->ijab',
            eri,
            orbo,
            orbo,
            orbv,
            orbv,
            optimize=True,
        )
        a -= hyb * np.transpose(eri_ijab, (0, 2, 1, 3))

        eri_jaib = np.einsum(
            'pqrs,pj,qa,ri,sb->jaib',
            eri,
            orbo,
            orbv,
            orbo,
            orbv,
            optimize=True,
        )
        b -= hyb * np.transpose(eri_jaib, (2, 1, 0, 3))

    if hasattr(mf, 'xc'):
        xctype = xc_type(mf.xc)
        if xctype == 'LDA':
            kxc = _lda_kernel_ovov(mf, orbo, orbv)
            a += kxc
            b += kxc
        else:
            raise NotImplementedError(
                f"Linear-response TDDFT currently supports only LDA-family kernels, got '{mf.xc}'."
            )

    return a, b


class TDA:
    """
    Restricted singlet TDA on top of RHF or native RKS references.
    """

    def __init__(self, mf):
        self._scf = mf
        self.mol = mf.mol
        self.a = None
        self.b = None
        self.e = None
        self.xy = None

        _, _, occidx, viridx, _, _ = _ov_blocks(mf)
        self.nocc = len(occidx)
        self.nvir = len(viridx)

    def get_ab(self):
        a, b = get_ab(self._scf)
        self.a = a
        self.b = b
        return a, b

    def run(self, nstates=None):
        a, _ = self.get_ab()
        dim = self.nocc * self.nvir
        e, x = _eig_hermitian(a.reshape(dim, dim), nstates=nstates)
        self.e = e
        self.xy = [
            (x[:, i].reshape(self.nocc, self.nvir), np.zeros((self.nocc, self.nvir)))
            for i in range(x.shape[1])
        ]
        return self


class TDDFT(TDA):
    """
    Restricted singlet linear-response TDDFT.
    """

    def run(self, nstates=None, using_tda=False):
        if using_tda:
            return super().run(nstates=nstates)

        a, b = self.get_ab()
        dim = self.nocc * self.nvir
        a2 = a.reshape(dim, dim)
        b2 = b.reshape(dim, dim)
        ham = np.block([[a2, b2], [-b2, -a2]])
        e, vec = np.linalg.eig(ham)

        mask = (e.real > 1e-8) & (np.abs(e.imag) < 1e-7)
        e = e.real[mask]
        vec = vec[:, mask].real
        order = np.argsort(e)
        e = e[order]
        vec = vec[:, order]

        if nstates is not None:
            e = e[:nstates]
            vec = vec[:, :nstates]

        self.e = e
        self.xy = []
        for i in range(vec.shape[1]):
            x = vec[:dim, i].reshape(self.nocc, self.nvir)
            y = vec[dim:, i].reshape(self.nocc, self.nvir)
            self.xy.append((x, y))
        return self
