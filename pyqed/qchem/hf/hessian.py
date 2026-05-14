#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native RHF analytic Hessian from builtin derivative integrals.
"""

import numpy as np

from pyqed.qchem.basis_derivatives import (
    compact_eri_veff,
    compact_eri_veff_many,
    eri_derivative_veff_scalar,
    eri_derivatives,
    one_electron_derivatives,
    position_derivatives,
)


def _nuclear_hessian(mol):
    coords = np.asarray(mol.atom_coords(), dtype=float)
    charges = np.asarray(mol.atom_charges(), dtype=float)
    natm = coords.shape[0]
    hess = np.zeros((natm, 3, natm, 3), dtype=float)
    eye = np.eye(3)

    for ia in range(natm):
        for ja in range(natm):
            if ia == ja:
                continue
            rij = coords[ia] - coords[ja]
            r = np.linalg.norm(rij)
            block = charges[ia] * charges[ja] * (
                eye / r**3 - 3.0 * np.outer(rij, rij) / r**5
            )
            hess[ia, :, ja, :] = block
            hess[ia, :, ia, :] -= block
    return hess


def _jk_from_eri(eri, dm):
    vj = np.einsum("rs,pqrs->pq", dm, eri, optimize=True)
    vk = np.einsum("rs,prqs->pq", dm, eri, optimize=True)
    return vj, vk


def _veff_from_eri(eri, dm):
    vj, vk = _jk_from_eri(eri, dm)
    return vj - 0.5 * vk


def _eri_veff(eri, dm, *index):
    if hasattr(eri, "veff"):
        return compact_eri_veff(eri, dm, *index)
    block = eri if not index else eri[index]
    return _veff_from_eri(block, dm)


class RHFHessian:
    """
    Analytic Cartesian Hessian for native ``pyqed.qchem.hf.RHF``.

    The implementation uses PyQED's builtin derivative integral layer for all
    explicit one- and two-electron nuclear derivatives.  The CPHF equations are
    solved as a dense occupied-virtual linear system.
    """

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.coords = np.asarray(self.mol.atom_coords(), dtype=float)
        self.hess = None
        self.hess4 = None
        self.cphf_amplitudes = None
        self.first_order_density = None
        self.first_order_energy_weighted_density = None
        self.first_order_orbital_energy = None
        self.dipole_derivative = None

    def _require_scf(self):
        if self.base.mo_coeff is None or self.base.mo_occ is None or self.base.mo_energy is None:
            raise ValueError("Run RHF before requesting its analytic Hessian.")
        if getattr(self.base, "_pyscf_mf", None) is not None or getattr(self.base, "density_fit", False):
            raise NotImplementedError(
                "Native RHF Hessian requires a builtin RHF reference, not density_fit=True."
            )
        if getattr(self.base, "low_rank_jk", False) or getattr(self.base, "cholesky_jk", False):
            raise NotImplementedError(
                "Native RHF Hessian currently requires the exact builtin J/K reference."
            )

    @property
    def _npert(self):
        return self.mol.natom * 3

    def _pack_pert(self, arr):
        return np.asarray(arr, dtype=float).reshape(self._npert, *arr.shape[2:])

    def _estimate_compact_eri2_bytes(self):
        basis = getattr(self.mol, "_bas_cart", None)
        if basis is None:
            basis = getattr(self.mol, "_bas", None)
        if basis is None:
            return np.inf
        nao_cart = len(basis)
        npair = nao_cart * (nao_cart + 1) // 2
        npert = self._npert
        return int(npert * npert * npair * npair * np.dtype(float).itemsize)

    def _second_derivative_veff_scalar(self, dm0, max_compact_bytes):
        estimated_bytes = self._estimate_compact_eri2_bytes()
        if estimated_bytes <= int(max_compact_bytes):
            g2 = eri_derivatives(self.mol, order=2, compact=True)
            g2_veff = compact_eri_veff_many(g2, dm0)
            return np.einsum("pq,xypq->xy", dm0, g2_veff, optimize=True)
        return eri_derivative_veff_scalar(self.mol, dm0, dm0, order=2)

    def _explicit_fock_derivatives(self, h1, eri1, dm0):
        npert = self._npert
        h1f = h1.reshape(npert, *h1.shape[2:])
        if hasattr(eri1, "data"):
            f1 = h1f + compact_eri_veff_many(eri1, dm0).reshape(h1f.shape)
            return f1.reshape(h1.shape)

        f1 = np.empty_like(h1f)
        for x in range(npert):
            f1[x] = h1f[x] + _eri_veff(eri1, dm0, x)
        return f1.reshape(h1.shape)

    def _response_veff_mo(self, u_mo, mo_coeff, cocc):
        c1 = mo_coeff @ u_mo
        dm1 = 2.0 * (c1 @ cocc.T + cocc @ c1.T)
        v1 = self.base.get_veff(dm1)
        return mo_coeff.T @ v1 @ cocc, dm1, v1

    def _build_cphf_matrix(self, mo_coeff, cocc, occidx, viridx, e_occ, e_vir):
        nocc = occidx.size
        nvir = viridx.size
        nvar = nvir * nocc
        amat = np.zeros((nvar, nvar), dtype=float)
        for col, (bpos, j) in enumerate(np.ndindex(nvir, nocc)):
            trial = np.zeros((mo_coeff.shape[1], nocc), dtype=float)
            trial[viridx[bpos], j] = 1.0
            v_trial, _dm_trial, _ = self._response_veff_mo(trial, mo_coeff, cocc)
            amat[:, col] = -v_trial[viridx].reshape(-1)
            row = bpos * nocc + j
            amat[row, col] += e_occ[j] - e_vir[bpos]
        return amat

    def _solve_cphf(self, f1, s1):
        mo_coeff = np.asarray(self.base.mo_coeff, dtype=float)
        mo_energy = np.asarray(self.base.mo_energy, dtype=float)
        mo_occ = np.asarray(self.base.mo_occ)
        occidx = np.flatnonzero(mo_occ > 0)
        viridx = np.flatnonzero(mo_occ == 0)
        cocc = mo_coeff[:, occidx]
        nmo = mo_coeff.shape[1]
        nocc = occidx.size
        nvir = viridx.size
        npert = self._npert
        e_occ = mo_energy[occidx]
        e_vir = mo_energy[viridx]

        f1f = f1.reshape(npert, *f1.shape[2:])
        s1f = s1.reshape(npert, *s1.shape[2:])

        u_all = np.zeros((npert, nmo, nocc), dtype=float)
        e1_all = np.zeros((npert, nocc, nocc), dtype=float)
        dm1_all = np.zeros((npert, self.mol.nao, self.mol.nao), dtype=float)
        w1_all = np.zeros_like(dm1_all)

        amat = self._build_cphf_matrix(mo_coeff, cocc, occidx, viridx, e_occ, e_vir)
        rhs_all = np.zeros((npert, nvir, nocc), dtype=float)
        u_fixed_all = np.zeros((npert, nmo, nocc), dtype=float)
        h_vo_all = np.zeros((npert, nmo, nocc), dtype=float)

        for x in range(npert):
            f1_mo = mo_coeff.T @ f1f[x] @ mo_coeff
            s1_mo = mo_coeff.T @ s1f[x] @ mo_coeff
            h_vo = f1_mo[:, occidx] - s1_mo[:, occidx] * e_occ

            u_fixed = np.zeros((nmo, nocc), dtype=float)
            u_fixed[occidx, :] = -0.5 * s1_mo[np.ix_(occidx, occidx)]
            v_fixed, _dm_fixed, _ = self._response_veff_mo(u_fixed, mo_coeff, cocc)
            rhs_all[x] = h_vo[viridx] + v_fixed[viridx]
            u_fixed_all[x] = u_fixed
            h_vo_all[x] = h_vo

        u_vo_all = np.linalg.solve(amat, rhs_all.reshape(npert, -1).T).T.reshape(npert, nvir, nocc)

        for x in range(npert):
            h_vo = h_vo_all[x]
            u_vo = u_vo_all[x]
            u = u_fixed_all[x].copy()
            u[viridx] = u_vo

            v_total, dm1, _ = self._response_veff_mo(u, mo_coeff, cocc)
            hs = h_vo + v_total
            e1 = hs[occidx] + u[occidx] * (e_occ[:, None] - e_occ[None, :])

            c1 = mo_coeff @ u
            w1 = 2.0 * (
                c1 @ np.diag(e_occ) @ cocc.T
                + cocc @ np.diag(e_occ) @ c1.T
                + cocc @ e1 @ cocc.T
            )

            u_all[x] = u
            e1_all[x] = e1
            dm1_all[x] = dm1
            w1_all[x] = w1

        self.cphf_amplitudes = u_all.reshape(self.mol.natom, 3, nmo, nocc)
        self.first_order_density = dm1_all.reshape(self.mol.natom, 3, self.mol.nao, self.mol.nao)
        self.first_order_energy_weighted_density = w1_all.reshape(
            self.mol.natom, 3, self.mol.nao, self.mol.nao
        )
        self.first_order_orbital_energy = e1_all.reshape(self.mol.natom, 3, nocc, nocc)
        return dm1_all, w1_all

    def run(self, symmetrize=True, max_compact_eri2_bytes=512 * 1024**2):
        self._require_scf()

        dm0 = np.asarray(self.base.make_rdm1(), dtype=float)
        mo_coeff = np.asarray(self.base.mo_coeff, dtype=float)
        mo_occ = np.asarray(self.base.mo_occ)
        occidx = mo_occ > 0
        cocc = mo_coeff[:, occidx]
        w0 = np.einsum(
            "pi,qi,i->pq",
            cocc,
            cocc,
            np.asarray(self.base.mo_energy)[occidx] * mo_occ[occidx],
            optimize=True,
        )

        s1 = one_electron_derivatives(self.mol, "overlap", order=1)
        h1 = one_electron_derivatives(self.mol, "hcore", order=1)
        g1 = eri_derivatives(self.mol, order=1, compact=True)
        s2 = one_electron_derivatives(self.mol, "overlap", order=2)
        h2 = one_electron_derivatives(self.mol, "hcore", order=2)
        g2_scalar = self._second_derivative_veff_scalar(dm0, max_compact_eri2_bytes)

        f1 = self._explicit_fock_derivatives(h1, g1, dm0)
        dm1, w1 = self._solve_cphf(f1, s1)

        natm = self.mol.natom
        npert = self._npert
        s1f = s1.reshape(npert, self.mol.nao, self.mol.nao)
        h1f = h1.reshape(npert, self.mol.nao, self.mol.nao)
        s2f = s2.reshape(npert, npert, self.mol.nao, self.mol.nao)
        h2f = h2.reshape(npert, npert, self.mol.nao, self.mol.nao)
        hnuc = _nuclear_hessian(self.mol).reshape(npert, npert)
        g1_dm0 = compact_eri_veff_many(g1, dm0).reshape(npert, self.mol.nao, self.mol.nao)
        g1_dm1 = np.asarray(
            [
                compact_eri_veff_many(g1, dm1_y).reshape(npert, self.mol.nao, self.mol.nao)
                for dm1_y in dm1
            ]
        )

        hess = np.zeros((npert, npert), dtype=float)
        for x in range(npert):
            for y in range(npert):
                value = np.einsum("pq,pq->", dm0, h2f[x, y], optimize=True)
                value += np.einsum("pq,pq->", dm1[y], h1f[x], optimize=True)
                value += 0.5 * g2_scalar[x, y]
                value += 0.5 * np.einsum(
                    "pq,pq->", dm1[y], g1_dm0[x], optimize=True
                )
                value += 0.5 * np.einsum(
                    "pq,pq->", dm0, g1_dm1[y, x], optimize=True
                )
                value -= np.einsum("pq,pq->", w0, s2f[x, y], optimize=True)
                value -= np.einsum("pq,pq->", w1[y], s1f[x], optimize=True)
                hess[x, y] = value + hnuc[x, y]

        if symmetrize:
            hess = 0.5 * (hess + hess.T)
        self.hess = hess
        self.hess4 = hess.reshape(natm, 3, natm, 3)
        return self.hess

    def kernel(self, **kwargs):
        return self.run(**kwargs)

    def cartesian_dipole_derivatives(self, center=None):
        """
        Analytic Cartesian derivatives of the total RHF dipole moment.

        Returns
        -------
        np.ndarray
            Tensor with shape ``(natm, 3, 3)`` indexed as
            ``(atom, nuclear_axis, dipole_axis)``.
        """
        if self.first_order_density is None:
            self.run()
        if center is None:
            center = np.zeros(3)
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must be a length-3 Cartesian vector.")

        dm0 = np.asarray(self.base.make_rdm1(), dtype=float)
        dm1 = np.asarray(self.first_order_density, dtype=float)
        r_ao = np.asarray(self.mol.position_integral(center=center), dtype=float)
        r1_ao = position_derivatives(self.mol, center=center)

        charges = np.asarray(self.mol.atom_charges(), dtype=float)
        coords = np.asarray(self.mol.atom_coords(), dtype=float)
        natm = coords.shape[0]

        deriv = np.zeros((natm, 3, 3), dtype=float)
        for atom in range(natm):
            deriv[atom] += charges[atom] * np.eye(3)
        deriv -= np.einsum("Axtpq,qp->Axt", r1_ao, dm0, optimize=True)
        deriv -= np.einsum("tpq,Axqp->Axt", r_ao, dm1, optimize=True)

        self.dipole_derivative = deriv
        return deriv

    def normal_mode_dipole_derivatives(self, modes, center=None):
        """
        Contract Cartesian dipole derivatives with normal modes.
        """
        modes = np.asarray(modes, dtype=float)
        return np.einsum(
            "kAx,Axt->kt",
            modes,
            self.cartesian_dipole_derivatives(center=center),
            optimize=True,
        )

    def vibrational_analysis(
        self,
        remove_translation_rotation=True,
        negative_imaginary=True,
        zero_tol=1e-7,
    ):
        if self.hess is None:
            raise ValueError("Run the Hessian calculation before requesting vibrational analysis.")
        from pyqed.qchem.dft.hessian import analyze_cartesian_hessian

        return analyze_cartesian_hessian(
            self.hess,
            self.coords,
            self.mol.atom_mass_list(),
            remove_translation_rotation=remove_translation_rotation,
            negative_imaginary=negative_imaginary,
            zero_tol=zero_tol,
        )

    def frequencies(self, unit="cm^-1", **kwargs):
        data = self.vibrational_analysis(**kwargs)
        unit = unit.lower()
        if unit in ("cm^-1", "cm-1", "wavenumber", "wavenumbers"):
            return data["freq_cm1"]
        if unit in ("au", "a.u.", "hartree"):
            return data["freq_au"]
        raise ValueError("unit must be 'cm^-1' or 'au'.")
