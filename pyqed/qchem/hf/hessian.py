#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native RHF analytic Hessian from builtin derivative integrals.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from pyqed.qchem.basis_derivatives import (
    compact_eri_veff,
    compact_eri_veff_many,
    eri_derivative_veff_scalar,
    eri_derivatives,
    one_electron_derivatives,
    position_derivatives,
)
from pyqed.units import amu_to_au


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

    def _second_derivative_veff_scalar(
        self,
        dm0,
        max_compact_bytes,
        workers=None,
    ):
        estimated_bytes = self._estimate_compact_eri2_bytes()
        if estimated_bytes <= int(max_compact_bytes):
            g2 = eri_derivatives(self.mol, order=2, compact=True)
            g2_veff = compact_eri_veff_many(g2, dm0)
            return np.einsum("pq,xypq->xy", dm0, g2_veff, optimize=True)
        return eri_derivative_veff_scalar(
            self.mol,
            dm0,
            dm0,
            order=2,
            workers=workers,
        )

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

    def _cphf_orbital_columns(self, f1, s1):
        """Solve occupied MO response columns for arbitrary perturbations."""
        mo_coeff = np.asarray(self.base.mo_coeff, dtype=float)
        mo_energy = np.asarray(self.base.mo_energy, dtype=float)
        mo_occ = np.asarray(self.base.mo_occ)
        occidx = np.flatnonzero(mo_occ > 0)
        viridx = np.flatnonzero(mo_occ == 0)
        cocc = mo_coeff[:, occidx]
        nmo = mo_coeff.shape[1]
        nocc = occidx.size
        nvir = viridx.size
        e_occ = mo_energy[occidx]
        e_vir = mo_energy[viridx]

        f1 = np.asarray(f1, dtype=float)
        s1 = np.asarray(s1, dtype=float)
        if f1.shape != s1.shape or f1.shape[-2:] != (self.mol.nao, self.mol.nao):
            raise ValueError(
                "fock and overlap derivatives must have matching shape "
                "(..., nao, nao)"
            )
        perturbation_shape = f1.shape[:-2]
        npert = int(np.prod(perturbation_shape))
        f1f = f1.reshape(npert, self.mol.nao, self.mol.nao)
        s1f = s1.reshape(npert, self.mol.nao, self.mol.nao)

        u_all = np.zeros((npert, nmo, nocc), dtype=float)
        h_vo_all = np.zeros_like(u_all)
        s1_mo_all = np.empty((npert, nmo, nmo), dtype=float)
        rhs_all = np.zeros((npert, nvir, nocc), dtype=float)
        u_fixed_all = np.zeros_like(u_all)
        amat = self._build_cphf_matrix(
            mo_coeff, cocc, occidx, viridx, e_occ, e_vir
        )

        for x in range(npert):
            f1_mo = mo_coeff.T @ f1f[x] @ mo_coeff
            s1_mo = mo_coeff.T @ s1f[x] @ mo_coeff
            h_vo = f1_mo[:, occidx] - s1_mo[:, occidx] * e_occ

            u_fixed = np.zeros((nmo, nocc), dtype=float)
            u_fixed[occidx, :] = -0.5 * s1_mo[np.ix_(occidx, occidx)]
            v_fixed, _dm_fixed, _ = self._response_veff_mo(
                u_fixed, mo_coeff, cocc
            )
            rhs_all[x] = h_vo[viridx] + v_fixed[viridx]
            u_fixed_all[x] = u_fixed
            h_vo_all[x] = h_vo
            s1_mo_all[x] = s1_mo

        if nvir:
            u_vo_all = np.linalg.solve(
                amat, rhs_all.reshape(npert, -1).T
            ).T.reshape(npert, nvir, nocc)
        else:
            u_vo_all = np.empty((npert, 0, nocc), dtype=float)
        u_all[:] = u_fixed_all
        u_all[:, viridx, :] = u_vo_all
        return {
            "amplitudes": u_all,
            "h_vo": h_vo_all,
            "overlap_mo": s1_mo_all,
            "perturbation_shape": perturbation_shape,
            "mo_coeff": mo_coeff,
            "mo_energy": mo_energy,
            "cocc": cocc,
            "occidx": occidx,
            "viridx": viridx,
            "e_occ": e_occ,
        }

    def solve_mo_response(self, fock_derivatives, overlap_derivatives):
        """Return full RHF MO coefficient derivatives in canonical MO gauge.

        CPHF first determines the occupied response and therefore the relaxed
        first-order Fock matrix. The response of every canonical MO then follows
        from the differentiated generalized Fock eigenproblem. This includes
        occupied-occupied and virtual-virtual rotations needed by truncated
        active spaces.
        """
        response = self._first_order_mo_response_data(
            fock_derivatives,
            overlap_derivatives,
        )
        full = response["full_response"]
        return full.reshape(
            *response["perturbation_shape"],
            full.shape[-2],
            full.shape[-1],
        )

    def _first_order_mo_response_data(self, fock_derivatives, overlap_derivatives):
        self._require_scf()
        response = self._cphf_orbital_columns(
            fock_derivatives, overlap_derivatives
        )
        amplitudes = response["amplitudes"]
        overlap_mo = response["overlap_mo"]
        mo_coeff = response["mo_coeff"]
        mo_energy = response["mo_energy"]
        cocc = response["cocc"]
        occidx = response["occidx"]
        e_occ = response["e_occ"]
        f1 = np.asarray(fock_derivatives, dtype=float).reshape(
            len(amplitudes), self.mol.nao, self.mol.nao
        )
        full = -0.5 * overlap_mo
        dm1_all = np.empty(
            (len(amplitudes), self.mol.nao, self.mol.nao),
            dtype=float,
        )
        f1_total_ao = np.empty_like(dm1_all)
        f1_total_mo = np.empty_like(full)
        e1_occ = np.empty((len(amplitudes), len(e_occ), len(e_occ)))
        e1 = np.empty((len(amplitudes), len(mo_energy)))
        for perturbation in range(len(full)):
            v_occ, dm1, v1_ao = self._response_veff_mo(
                amplitudes[perturbation], mo_coeff, cocc
            )
            total_ao = f1[perturbation] + v1_ao
            relaxed_fock = mo_coeff.T @ total_ao @ mo_coeff
            h_vo = response["h_vo"][perturbation]
            e1_occ[perturbation] = (
                h_vo[occidx]
                + v_occ[occidx]
                + amplitudes[perturbation, occidx]
                * (e_occ[:, None] - e_occ[None, :])
            )
            e1[perturbation] = (
                np.diag(relaxed_fock)
                - np.diag(overlap_mo[perturbation]) * mo_energy
            )
            dm1_all[perturbation] = dm1
            f1_total_ao[perturbation] = total_ao
            f1_total_mo[perturbation] = relaxed_fock
            for column in range(len(mo_energy)):
                for row in range(len(mo_energy)):
                    if row == column:
                        continue
                    denominator = mo_energy[column] - mo_energy[row]
                    if abs(denominator) < 1.0e-10:
                        continue
                    full[perturbation, row, column] = (
                        relaxed_fock[row, column]
                        - overlap_mo[perturbation, row, column]
                        * mo_energy[column]
                    ) / denominator
        response.update({
            "full_response": full,
            "density_response": dm1_all,
            "fock_total_ao": f1_total_ao,
            "fock_total_mo": f1_total_mo,
            "orbital_energy_response": e1,
            "occupied_energy_response": e1_occ,
        })
        return response

    @staticmethod
    def _second_normalization(ux, uy, sx, sy, sxy, columns=None):
        if columns is None:
            left_x = ux.T
            left_y = uy.T
            sx_right = sx
            sy_right = sy
            sxy_block = sxy
        else:
            columns = np.asarray(columns, dtype=int)
            left_x = ux.T
            left_y = uy.T
            sx_right = sx[:, columns]
            sy_right = sy[:, columns]
            sxy_block = sxy[np.ix_(columns, columns)]
        return (
            sxy_block
            + left_x @ sy_right
            + left_x @ uy
            + left_y @ sx_right
            + left_y @ ux
            + sx_right.T @ uy
            + sy_right.T @ ux
        )

    def _second_order_mo_response_data(
        self,
        first,
        fock_second_explicit,
        overlap_second,
    ):
        mo_coeff = first["mo_coeff"]
        mo_energy = first["mo_energy"]
        occidx = first["occidx"]
        viridx = first["viridx"]
        cocc = first["cocc"]
        e_occ = first["e_occ"]
        u_occ = first["amplitudes"]
        u_full = first["full_response"]
        s1_mo = first["overlap_mo"]
        f1_mo = first["fock_total_mo"]
        e1_occ = first["occupied_energy_response"]
        e1 = first["orbital_energy_response"]
        ncoord, nmo, nocc = u_occ.shape
        nvir = len(viridx)
        f2_explicit = np.asarray(fock_second_explicit, dtype=float).reshape(
            ncoord, ncoord, self.mol.nao, self.mol.nao
        )
        s2 = np.asarray(overlap_second, dtype=float).reshape(
            ncoord, ncoord, self.mol.nao, self.mol.nao
        )
        s2_mo = np.einsum(
            "pi,xypq,qj->xyij",
            mo_coeff,
            s2,
            mo_coeff,
            optimize=True,
        )
        e0_occ = np.diag(e_occ)
        fixed = np.zeros((ncoord, ncoord, nmo, nocc), dtype=float)
        rhs = np.zeros((ncoord, ncoord, nvir, nocc), dtype=float)

        for x in range(ncoord):
            for y in range(ncoord):
                normalization = self._second_normalization(
                    u_occ[x],
                    u_occ[y],
                    s1_mo[x],
                    s1_mo[y],
                    s2_mo[x, y],
                    columns=occidx,
                )
                fixed[x, y, occidx] = -0.5 * normalization
                c1x = mo_coeff @ u_occ[x]
                c1y = mo_coeff @ u_occ[y]
                c2_fixed = mo_coeff @ fixed[x, y]
                dm2_fixed = 2.0 * (
                    c2_fixed @ cocc.T
                    + cocc @ c2_fixed.T
                    + c1x @ c1y.T
                    + c1y @ c1x.T
                )
                f2_known = f2_explicit[x, y] + self.base.get_veff(dm2_fixed)
                f2_known_mo = mo_coeff.T @ f2_known @ mo_coeff
                known = (
                    f2_known_mo[:, occidx]
                    + f1_mo[x] @ u_occ[y]
                    + f1_mo[y] @ u_occ[x]
                    - s2_mo[x, y][:, occidx] @ e0_occ
                    - s1_mo[x] @ u_occ[y] @ e0_occ
                    - s1_mo[x][:, occidx] @ e1_occ[y]
                    - s1_mo[y] @ u_occ[x] @ e0_occ
                    - u_occ[x] @ e1_occ[y]
                    - s1_mo[y][:, occidx] @ e1_occ[x]
                    - u_occ[y] @ e1_occ[x]
                )
                rhs[x, y] = known[viridx]

        if nvir:
            amat = self._build_cphf_matrix(
                mo_coeff,
                cocc,
                occidx,
                viridx,
                e_occ,
                mo_energy[viridx],
            )
            solved = np.linalg.solve(
                amat,
                rhs.reshape(ncoord * ncoord, -1).T,
            ).T.reshape(ncoord, ncoord, nvir, nocc)
        else:
            solved = np.empty((ncoord, ncoord, 0, nocc), dtype=float)

        u2_occ = fixed
        u2_occ[:, :, viridx] = solved
        dm2 = np.empty(
            (ncoord, ncoord, self.mol.nao, self.mol.nao),
            dtype=float,
        )
        f2_total_mo = np.empty((ncoord, ncoord, nmo, nmo), dtype=float)
        for x in range(ncoord):
            for y in range(ncoord):
                c1x = mo_coeff @ u_occ[x]
                c1y = mo_coeff @ u_occ[y]
                c2 = mo_coeff @ u2_occ[x, y]
                dm2[x, y] = 2.0 * (
                    c2 @ cocc.T
                    + cocc @ c2.T
                    + c1x @ c1y.T
                    + c1y @ c1x.T
                )
                f2_total = f2_explicit[x, y] + self.base.get_veff(dm2[x, y])
                f2_total_mo[x, y] = mo_coeff.T @ f2_total @ mo_coeff

        e0 = np.diag(mo_energy)
        u2_full = np.empty((ncoord, ncoord, nmo, nmo), dtype=float)
        for x in range(ncoord):
            for y in range(ncoord):
                normalization = self._second_normalization(
                    u_full[x],
                    u_full[y],
                    s1_mo[x],
                    s1_mo[y],
                    s2_mo[x, y],
                )
                response = -0.5 * normalization
                known = (
                    f2_total_mo[x, y]
                    + f1_mo[x] @ u_full[y]
                    + f1_mo[y] @ u_full[x]
                    - s2_mo[x, y] @ e0
                    - s1_mo[x] @ u_full[y] @ e0
                    - s1_mo[x] @ np.diag(e1[y])
                    - s1_mo[y] @ u_full[x] @ e0
                    - u_full[x] @ np.diag(e1[y])
                    - s1_mo[y] @ np.diag(e1[x])
                    - u_full[y] @ np.diag(e1[x])
                )
                for column in range(nmo):
                    for row in range(nmo):
                        if row == column:
                            continue
                        denominator = mo_energy[column] - mo_energy[row]
                        if abs(denominator) < 1.0e-10:
                            continue
                        response[row, column] = known[row, column] / denominator
                u2_full[x, y] = response

        u2_full = 0.5 * (u2_full + u2_full.swapaxes(0, 1))
        return {
            "full_response": u2_full,
            "occupied_response": u2_occ,
            "density_response": dm2,
            "fock_total_mo": f2_total_mo,
            "overlap_mo": s2_mo,
        }

    def orbital_second_response(
        self,
        hcore_first,
        eri_first,
        overlap_first,
        hcore_second,
        eri_second,
        overlap_second,
    ):
        """Return first and second canonical RHF MO coefficient responses."""
        self._require_scf()
        h1 = np.asarray(hcore_first, dtype=float)
        eri1 = np.asarray(eri_first, dtype=float)
        s1 = np.asarray(overlap_first, dtype=float)
        h2 = np.asarray(hcore_second, dtype=float)
        eri2 = np.asarray(eri_second, dtype=float)
        s2 = np.asarray(overlap_second, dtype=float)
        ncoord = h1.shape[0]
        one_shape = (ncoord, self.mol.nao, self.mol.nao)
        two_shape = (ncoord, ncoord, self.mol.nao, self.mol.nao)
        if h1.shape != one_shape or s1.shape != one_shape:
            raise ValueError(f"first one-electron derivatives must have shape {one_shape}")
        if h2.shape != two_shape or s2.shape != two_shape:
            raise ValueError(f"second one-electron derivatives must have shape {two_shape}")
        if eri1.shape != (ncoord,) + (self.mol.nao,) * 4:
            raise ValueError("first ERI derivative shape is inconsistent")
        if eri2.shape != (ncoord, ncoord) + (self.mol.nao,) * 4:
            raise ValueError("second ERI derivative shape is inconsistent")

        dm0 = np.asarray(self.base.make_rdm1(), dtype=float)
        f1 = np.empty_like(h1)
        for x in range(ncoord):
            f1[x] = h1[x] + _veff_from_eri(eri1[x], dm0)
        first = self._first_order_mo_response_data(f1, s1)
        dm1 = first["density_response"]
        f2_explicit = np.empty_like(h2)
        for x in range(ncoord):
            for y in range(ncoord):
                f2_explicit[x, y] = (
                    h2[x, y]
                    + _veff_from_eri(eri2[x, y], dm0)
                    + _veff_from_eri(eri1[x], dm1[y])
                    + _veff_from_eri(eri1[y], dm1[x])
                )
        second = self._second_order_mo_response_data(first, f2_explicit, s2)
        return first["full_response"], second["full_response"]

    def orbital_response(self, hcore_derivatives, eri_derivatives, overlap_derivatives):
        """Solve RHF MO response from explicit AO integral derivatives."""
        self._require_scf()
        h1 = np.asarray(hcore_derivatives, dtype=float)
        eri1 = np.asarray(eri_derivatives, dtype=float)
        s1 = np.asarray(overlap_derivatives, dtype=float)
        if h1.shape != s1.shape or h1.shape[-2:] != (self.mol.nao, self.mol.nao):
            raise ValueError(
                "hcore and overlap derivatives must have matching shape "
                "(..., nao, nao)"
            )
        expected_eri = (*h1.shape[:-2],) + (self.mol.nao,) * 4
        if eri1.shape != expected_eri:
            raise ValueError(
                f"eri_derivatives shape {eri1.shape} != {expected_eri}"
            )
        dm0 = np.asarray(self.base.make_rdm1(), dtype=float)
        f1 = np.empty_like(h1)
        for index in np.ndindex(h1.shape[:-2]):
            f1[index] = h1[index] + _veff_from_eri(eri1[index], dm0)
        return self.solve_mo_response(f1, s1)

    def _solve_cphf(self, f1, s1):
        response = self._cphf_orbital_columns(f1, s1)
        mo_coeff = response["mo_coeff"]
        cocc = response["cocc"]
        occidx = response["occidx"]
        e_occ = response["e_occ"]
        u_all = response["amplitudes"]
        h_vo_all = response["h_vo"]
        npert, nmo, nocc = u_all.shape
        e1_all = np.zeros((npert, nocc, nocc), dtype=float)
        dm1_all = np.zeros((npert, self.mol.nao, self.mol.nao), dtype=float)
        w1_all = np.zeros_like(dm1_all)

        for x in range(npert):
            h_vo = h_vo_all[x]
            u = u_all[x]

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

    def run(
        self,
        symmetrize=True,
        max_compact_eri2_bytes=512 * 1024**2,
        workers=None,
    ):
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
        g2_scalar = self._second_derivative_veff_scalar(
            dm0,
            max_compact_eri2_bytes,
            workers=workers,
        )

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
        try:
            from pyqed.qchem.dft.hessian import analyze_cartesian_hessian
        except ModuleNotFoundError as error:
            if error.name != "pyqed.qchem.dft":
                raise
            from pyqed.qchem.DFT.hessian import analyze_cartesian_hessian

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

    def normal_modes(
        self,
        targets=None,
        *,
        target_unit="cm^-1",
        dimensionless=False,
        **kwargs,
    ):
        """Return positive-frequency modes, optionally matched to targets.

        Parameters
        ----------
        targets : array_like, optional
            Frequencies used to select distinct nearest modes. By default all
            positive-frequency vibrational modes are returned.
        target_unit : {'cm^-1', 'au'}, optional
            Unit of ``targets``.
        dimensionless : bool, optional
            Scale Cartesian modes for displacements by dimensionless normal
            coordinates.

        Returns
        -------
        omega, modes : ndarray
            Selected frequencies in atomic units and Cartesian mode vectors.
        """

        data = self.vibrational_analysis(**kwargs)
        omega = np.asarray(data["freq_au"], dtype=float)
        modes = np.asarray(data["modes"], dtype=float)
        valid = np.flatnonzero(np.isfinite(omega) & (omega > 0.0))

        if targets is not None:
            targets = np.atleast_1d(np.asarray(targets, dtype=float))
            if targets.ndim != 1 or not np.all(np.isfinite(targets)):
                raise ValueError("targets must be a finite one-dimensional array")
            if targets.size > valid.size:
                raise ValueError("more target frequencies than positive normal modes")
            unit = str(target_unit).lower()
            if unit in ("cm^-1", "cm-1", "wavenumber", "wavenumbers"):
                values = np.asarray(data["freq_cm1"], dtype=float)[valid]
            elif unit in ("au", "a.u.", "hartree"):
                values = omega[valid]
            else:
                raise ValueError("target_unit must be 'cm^-1' or 'au'")
            rows, columns = linear_sum_assignment(
                abs(values[:, None] - targets[None, :])
            )
            selected = np.empty(targets.size, dtype=int)
            selected[columns] = valid[rows]
        else:
            selected = valid

        omega = omega[selected]
        modes = np.array(modes[selected], copy=True)
        if dimensionless:
            modes /= np.sqrt(amu_to_au * omega)[:, None, None]
        return omega, modes
