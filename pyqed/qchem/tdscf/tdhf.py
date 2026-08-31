#!/usr/bin/env python3
"""PySCF-backed restricted TDH/TDHF response helpers."""

import logging

import numpy as np
from pyscf import ao2mo, scf
from scipy.linalg import eigh, sqrtm
from scipy.sparse.linalg import eigsh

from pyqed.units import au2ev as AU2EV

au2ev = AU2EV


def is_positive_def(a, tol=0.0):
    vals = np.linalg.eigvalsh(np.asarray(a))
    return bool(np.all(vals > tol))


def eig_asymm(h):
    """Diagonalize a real non-Hermitian matrix and sort by eigenvalue."""
    e, c = np.linalg.eig(h)
    if np.allclose(e.imag, 0.0):
        e = e.real
        c = c.real
    else:
        logging.warning("TDHF eigenvalues have non-negligible imaginary parts.")

    idx = np.argsort(e.real)
    return e[idx], c[:, idx]


def eig(a, k=None, **kwargs):
    """Hermitian eigensolver with sparse fallback for small requested roots."""
    a = np.asarray(a)
    if isinstance(k, int):
        if k <= 0:
            raise ValueError("k must be positive.")
        if k < a.shape[0] - 1:
            e, x = eigsh(a, k=k, **kwargs)
        else:
            e, x = eigh(a)
            e = e[:k]
            x = x[:, :k]
    else:
        e, x = eigh(a)
    idx = np.argsort(e)
    return e[idx], x[:, idx]


def _validate_method(method):
    key = str(method).upper()
    if key not in {"TDH", "TDHF"}:
        raise NotImplementedError("This legacy module supports only TDH and TDHF.")
    return key


def _as_pyscf_rhf(mf):
    if not isinstance(mf, scf.rhf.RHF):
        raise NotImplementedError("Only PySCF restricted HF references are supported.")
    if mf.mo_coeff is None or mf.mo_energy is None or mf.mo_occ is None:
        raise ValueError("Run the PySCF RHF reference before TDHF.")
    return mf


def _ov_blocks(td):
    mf = _as_pyscf_rhf(td._scf)
    mo_energy = np.asarray(mf.mo_energy)
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)

    occidx = np.where(mo_occ > 0)[0]
    viridx = np.where(mo_occ == 0)[0]
    if occidx.size == 0 or viridx.size == 0:
        raise ValueError("TDHF requires at least one occupied and one virtual orbital.")

    return mo_energy, occidx, viridx, mo_coeff[:, occidx], mo_coeff[:, viridx]


def get_ab(td, method="TDH", singlet=True):
    """Compute restricted PySCF RHF A/B matrices in the OV response space."""
    method = _validate_method(method)
    mo_energy, occidx, viridx, orbo, orbv = _ov_blocks(td)
    nocc = len(occidx)
    nvir = len(viridx)
    dim = nocc * nvir
    logging.info("TDHF response dimension = %d", dim)

    e_ia = mo_energy[viridx] - mo_energy[occidx, None]
    a = np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
    b = np.zeros_like(a)

    eri_iajb = ao2mo.general(
        td.mol,
        (orbo, orbv, orbo, orbv),
        compact=False,
    ).reshape(nocc, nvir, nocc, nvir)

    eri_ijab = ao2mo.general(
        td.mol,
        (orbo, orbo, orbv, orbv),
        compact=False,
    ).reshape(nocc, nocc, nvir, nvir)
    k_a = np.transpose(eri_ijab, (0, 2, 1, 3))

    eri_jaib = ao2mo.general(
        td.mol,
        (orbo, orbv, orbo, orbv),
        compact=False,
    ).reshape(nocc, nvir, nocc, nvir)
    k_b = np.transpose(eri_jaib, (2, 1, 0, 3))

    if singlet:
        a += 2.0 * eri_iajb
        b += 2.0 * eri_iajb

    if method == "TDHF":
        a -= k_a
        b -= k_b

    a = a.reshape(dim, dim)
    b = b.reshape(dim, dim)
    if not np.allclose(a, a.T):
        raise ValueError("TDHF A matrix is not symmetric.")
    if not np.allclose(b, b.T):
        raise ValueError("TDHF B matrix is not symmetric.")
    return a, b


def rpa(td, using_tda=False, using_casida=True, method="TDH", singlet=True):
    """Solve the restricted TDH/TDHF RPA problem."""
    a, b = get_ab(td, method=method, singlet=singlet)

    if using_tda:
        return eig(a)

    if using_casida:
        a_minus_b = a - b
        if not is_positive_def(a_minus_b):
            raise ValueError("Casida TDHF requires A-B to be positive definite.")
        sqrt_a_minus_b = sqrtm(a_minus_b)
        ham = sqrt_a_minus_b @ (a + b) @ sqrt_a_minus_b
        esq, vec = eigh(ham)
        return np.sqrt(np.clip(esq, 0.0, None)), vec

    ham = np.block([[a, b], [-b, -a]])
    e, xy = eig_asymm(ham)
    mask = e.real > 1.0e-8
    return e[mask], xy[:, mask]


class TDHF:
    """Restricted TDH/TDHF adapter for a converged PySCF RHF object."""

    def __init__(self, mf):
        mf = _as_pyscf_rhf(mf)
        self.mol = mf.mol
        self._scf = mf
        self.verbose = getattr(self.mol, "verbose", 0)
        self.stdout = getattr(self.mol, "stdout", None)
        self.max_memory = getattr(mf, "max_memory", None)
        self.spin = 0
        self.singlet = True

        self._a = None
        self._b = None
        self.e = None
        self.xy = None

        self.e_mf = np.asarray(mf.mo_energy)
        self.nocc = int(self.mol.nelectron // 2)
        self.nso = len(self.e_mf)

    def get_ab(self, method="TDHF", singlet=None):
        if singlet is None:
            singlet = self.singlet
        self._a, self._b = get_ab(self, method=method, singlet=singlet)
        return self._a, self._b

    def run(self, nstates=None, using_tda=False, using_casida=True, method="TDHF", singlet=None):
        if singlet is None:
            singlet = self.singlet

        a, b = self.get_ab(method=method, singlet=singlet)
        if using_tda:
            self.e, self.xy = eig(a, k=nstates, which="SA" if isinstance(nstates, int) else None)
            return self.e, self.xy

        if using_casida:
            a_minus_b = a - b
            if not is_positive_def(a_minus_b):
                raise ValueError("Casida TDHF requires A-B to be positive definite.")
            sqrt_a_minus_b = sqrtm(a_minus_b)
            ham = sqrt_a_minus_b @ (a + b) @ sqrt_a_minus_b
            if isinstance(nstates, int):
                esq, vec = eig(ham, k=nstates, which="SA")
            else:
                esq, vec = eigh(ham)
            self.e = np.sqrt(np.clip(esq, 0.0, None))
            self.xy = vec
            return self.e, self.xy

        ham = np.block([[a, b], [-b, -a]])
        e, xy = eig_asymm(ham)
        mask = e.real > 1.0e-8
        e = e[mask]
        xy = xy[:, mask]
        if isinstance(nstates, int):
            e = e[:nstates]
            xy = xy[:, :nstates]
        self.e = e
        self.xy = xy
        return self.e, self.xy
