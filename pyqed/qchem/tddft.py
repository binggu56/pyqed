#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear-response restricted TDA/TDDFT for native pyqed mean-field objects.
"""

import numpy as np
from scipy.linalg import eigh

from .dft.scf import ensure_grid_for_xc
from .dft.xc import eval_fxc, hybrid_coeff, xc_type


class Gradients:
    """
    Analytic nuclear gradients for native ``pyqed.qchem.tddft`` excited states.

    Notes
    -----
    The linear-response eigenproblem is solved natively in ``pyqed``.  Excited-state
    analytic gradients are delegated to a mirrored PySCF TDA/TDDFT calculation on the
    same molecule and XC functional because the native codebase does not yet contain
    the TDDFT response/Z-vector machinery needed for a fully native implementation.
    """

    def __init__(self, td, backend='pyscf'):
        self.base = td
        self.backend = backend
        self._mirror_scf = None
        self._mirror_td = None

    def dump_flags(self):
        return self

    def _build_mirror(self):
        if self.backend != 'pyscf':
            raise NotImplementedError(
                f"TDDFT excited-state analytic gradients with backend='{self.backend}' "
                "are not implemented. Use backend='pyscf'."
            )
        if self._mirror_td is not None:
            return self._mirror_scf, self._mirror_td

        try:
            from pyscf import dft as pyscf_dft
            from pyscf import tdscf as pyscf_tdscf
        except Exception as exc:
            raise NotImplementedError(
                "Analytic excited-state gradients for pyqed TDDFT currently require "
                "a local PySCF installation."
            ) from exc

        mol = self.base.mol
        if not hasattr(mol, 'topyscf'):
            raise NotImplementedError(
                "Analytic excited-state gradients require a pyqed Molecule with topyscf()."
            )

        pmol = mol.topyscf()
        mf = pyscf_dft.RKS(pmol)
        mf.xc = getattr(self.base._scf, 'xc', 'lda')

        dm0 = getattr(self.base._scf, 'dm', None)
        if dm0 is not None:
            mf.kernel(dm0=np.asarray(dm0))
        else:
            mf.kernel()

        if type(self.base) is TDA:
            td = pyscf_tdscf.TDA(mf)
        else:
            td = pyscf_tdscf.TDDFT(mf)

        target_states = len(self.base.e) if getattr(self.base, 'e', None) is not None else 1
        td.nstates = target_states
        td.kernel()

        self._mirror_scf = mf
        self._mirror_td = td
        return mf, td

    def kernel(self, state=1, atmlst=None):
        if state == 0:
            grad = self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)
            return np.asarray(grad, dtype=float)

        if state < 0:
            raise ValueError("state must be >= 0.")

        mf, td = self._build_mirror()
        if state > len(td.e):
            raise ValueError(
                f"Requested excited state {state} but only {len(td.e)} states are available."
            )
        grad = td.nuc_grad_method().kernel(state=state, atmlst=atmlst)
        return np.asarray(grad, dtype=float)

    run = kernel


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


def _dense_eri(mol):
    eri = getattr(mol, 'eri', None)
    if eri is not None:
        return eri

    eri_s4 = getattr(mol, 'eri_s4', None)
    if eri_s4 is not None:
        from pyqed.qchem.basis import unpack_eri_s4

        return unpack_eri_s4(eri_s4, mol.nao)

    eri_s8 = getattr(mol, 'eri_s8', None)
    if eri_s8 is not None:
        from pyqed.qchem.basis import unpack_eri_s8

        return unpack_eri_s8(eri_s8, mol.nao)

    raise ValueError("TDDFT requires mol.eri, mol.eri_s4, or mol.eri_s8.")


def _pcm_kernel_ovov(mf, solvent):
    """
    Singlet PCM response kernel in the occupied-virtual basis.

    The native TDDFT matrices use the same spin-adapted singlet convention as
    the Coulomb term in ``get_ab``.  The PCM response is a direct density
    response and therefore contributes the same ``ovov`` block to A and B.
    """
    _, _, _, _, orbo, orbv = _ov_blocks(mf)
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    kernel = np.empty((nocc, nvir, nocc, nvir), dtype=float)
    for j in range(nocc):
        for b in range(nvir):
            dm_jb = np.einsum("p,q->pq", orbo[:, j], orbv[:, b].conj(), optimize=True)
            v_jb = solvent._B_dot_x(dm_jb)
            kernel[:, :, j, b] = np.einsum(
                "pi,pq,qa->ia",
                orbo.conj(),
                v_jb,
                orbv,
                optimize=True,
            ).real
    return 2.0 * kernel


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

    eri = _dense_eri(mf.mol)
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
        solvent = getattr(self, "with_solvent", None)
        if solvent is not None:
            pcm_kernel = _pcm_kernel_ovov(self._scf, solvent)
            a = a + pcm_kernel
            b = b + pcm_kernel
            self.pcm_response_kernel = pcm_kernel
        self.a = a
        self.b = b
        return a, b

    def PCM(self, solvent_obj=None, dm=None, equilibrium_solvation=False, **kwargs):
        """
        Attach PCM linear response to TDA/TDDFT.

        By default this follows the non-equilibrium vertical-excitation
        convention and uses an optical dielectric for the fast solvent
        response.  Pass ``equilibrium_solvation=True`` to use the equilibrium
        dielectric from the supplied/reference solvent object.
        """
        from pyqed.qchem import solvent

        td = solvent.PCM(
            self,
            solvent_obj=solvent_obj,
            dm=dm,
            equilibrium_solvation=equilibrium_solvation,
        )
        for key, value in kwargs.items():
            setattr(td.with_solvent, key, value)
        return td

    def nuc_grad_method(self, backend='pyscf'):
        return Gradients(self, backend=backend)

    def _contract_multipole(self, ints, hermi=True, xy=None):
        """
        Contract a spin-independent one-electron operator with TD amplitudes.

        This follows the restricted singlet convention used by PySCF:
        Hermitian operators contract with ``X + Y`` and anti-Hermitian
        operators with ``X - Y``.  The factor of two is the spin trace.
        """
        if xy is None:
            xy = self.xy
        if xy is None:
            raise ValueError("Run TDDFT/TDA before requesting transition moments.")

        ints = np.asarray(ints)
        if ints.ndim == 2:
            ints = ints.reshape((1,) + ints.shape)
        elif ints.ndim == 3 and ints.shape[0] != 3 and ints.shape[-1] == 3:
            ints = np.moveaxis(ints, -1, 0)
        if ints.ndim != 3:
            raise ValueError("Operator integrals must have shape (nao, nao) or (ncomp, nao, nao).")

        _, _, _, _, orbo, orbv = _ov_blocks(self._scf)
        ints_ov = np.einsum('xpq,pi,qa->xia', ints, orbo, orbv.conj(), optimize=True)

        values = []
        for x, y in xy:
            x = np.asarray(x)
            y = np.asarray(y)
            norm = np.vdot(x, x).real - np.vdot(y, y).real
            if norm <= 0.0:
                raise ValueError("TD amplitudes have non-positive RPA norm.")
            scale = np.sqrt(0.5 / norm)
            amp = scale * (x + y if hermi else x - y)
            values.append(2.0 * np.einsum('xia,ia->x', ints_ov, amp, optimize=True))
        values = np.asarray(values)
        return values[:, 0] if values.shape[1] == 1 else values

    def transition_dipole(self, center=None):
        """
        Length-gauge transition dipole moments in the PySCF convention.

        The returned operator is the position operator ``r - center`` rather
        than the electronic dipole ``-r``.
        """
        if center is None:
            center = self.mol.nuc_charge_center()
        ints = self.mol.moment_integral(center=np.asarray(center, dtype=float))
        return self._contract_multipole(ints, hermi=True)

    def transition_magnetic_dipole(self, center=None, convention='standard'):
        """
        Transition magnetic dipole moments.

        By default this follows the standard orbital magnetic-dipole
        convention, ``-0.5 * r x grad``.  Use ``convention='pyscf'`` or
        ``'raw'`` to reproduce PySCF's unhalved transition-vector convention.
        """
        if center is None:
            center = self.mol.nuc_charge_center()
        ints = self.mol.magnetic_dipole_integral(
            center=np.asarray(center, dtype=float),
            convention=convention,
        )
        return self._contract_multipole(ints, hermi=False)

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
