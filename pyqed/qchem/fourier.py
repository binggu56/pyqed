#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fourier-space Gaussian AO integrals."""

import math

import numpy as np

from pyqed.qchem.basis import E

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional acceleration
    njit = None


def _validate_gvec(gvec):
    gvec = np.asarray(gvec, dtype=float)
    if gvec.shape != (3,):
        raise ValueError("gvec must have shape (3,).")
    return gvec


def _validate_gvecs(gvecs):
    gvecs = np.asarray(gvecs, dtype=float)
    if gvecs.ndim != 2 or gvecs.shape[1] != 3:
        raise ValueError("gvecs must have shape (ng, 3).")
    return np.ascontiguousarray(gvecs)


def _pack_basis_for_compiled(basis):
    basis = tuple(basis)
    nao = len(basis)
    max_prim = max((len(fn.exps) for fn in basis), default=0)
    shells = np.empty((nao, 3), dtype=np.int64)
    origins = np.empty((nao, 3), dtype=float)
    exps = np.zeros((nao, max_prim), dtype=float)
    weights = np.zeros((nao, max_prim), dtype=float)
    nprim = np.empty(nao, dtype=np.int64)

    for i, fn in enumerate(basis):
        shells[i] = np.asarray(fn.shell, dtype=np.int64)
        origins[i] = np.asarray(fn.origin, dtype=float)
        fn_exps = np.asarray(fn.exps, dtype=float)
        fn_weights = np.asarray(fn.prim_weights, dtype=float)
        n = len(fn_exps)
        nprim[i] = n
        exps[i, :n] = fn_exps
        weights[i, :n] = fn_weights

    return (
        np.ascontiguousarray(shells),
        np.ascontiguousarray(origins),
        np.ascontiguousarray(exps),
        np.ascontiguousarray(weights),
        np.ascontiguousarray(nprim),
    )


def _upper_pair_indices(nao):
    pair_p = []
    pair_q = []
    for p in range(nao):
        for q in range(p, nao):
            pair_p.append(p)
            pair_q.append(q)
    return (
        np.ascontiguousarray(pair_p, dtype=np.int64),
        np.ascontiguousarray(pair_q, dtype=np.int64),
    )


def _pack_pair_primitives(pair_p, pair_q, exps, weights, nprim):
    starts = [0]
    alpha = []
    beta = []
    alpha_over_p = []
    beta_over_p = []
    inv_4p = []
    prefactor = []

    for pidx, qidx in zip(pair_p, pair_q):
        for ia in range(nprim[pidx]):
            a = float(exps[pidx, ia])
            wa = float(weights[pidx, ia])
            for ib in range(nprim[qidx]):
                b = float(exps[qidx, ib])
                wb = float(weights[qidx, ib])
                pp = a + b
                alpha.append(a)
                beta.append(b)
                alpha_over_p.append(a / pp)
                beta_over_p.append(b / pp)
                inv_4p.append(1.0 / (4.0 * pp))
                prefactor.append(wa * wb * (math.pi / pp) ** 1.5)
        starts.append(len(alpha))

    return (
        np.ascontiguousarray(starts, dtype=np.int64),
        np.ascontiguousarray(alpha, dtype=float),
        np.ascontiguousarray(beta, dtype=float),
        np.ascontiguousarray(alpha_over_p, dtype=float),
        np.ascontiguousarray(beta_over_p, dtype=float),
        np.ascontiguousarray(inv_4p, dtype=float),
        np.ascontiguousarray(prefactor, dtype=float),
    )


def _plane_gvecs(gvecs, plane_z=None, plane_tol=0.0):
    gvecs = _validate_gvecs(gvecs)
    if plane_z is None:
        plane_z = bool(np.all(np.abs(gvecs[:, 2]) <= float(plane_tol)))
    else:
        plane_z = bool(plane_z)
    if plane_z:
        gvecs = np.array(gvecs, copy=True)
        gvecs[:, 2] = 0.0
    return np.ascontiguousarray(gvecs), plane_z


def _as_density_arrays(dm1_ao, tdm1_ao, nao):
    dm1_ao = np.ascontiguousarray(dm1_ao, dtype=np.complex128)
    tdm1_ao = np.ascontiguousarray(tdm1_ao, dtype=np.complex128)
    if dm1_ao.ndim != 3 or dm1_ao.shape[1:] != (nao, nao):
        raise ValueError(f"dm1_ao must have shape (nstates, {nao}, {nao}).")
    if tdm1_ao.ndim != 4 or tdm1_ao.shape[2:] != (nao, nao):
        raise ValueError(f"tdm1_ao must have shape (nstates, nstates, {nao}, {nao}).")
    if tdm1_ao.shape[:2] != (dm1_ao.shape[0], dm1_ao.shape[0]):
        raise ValueError("tdm1_ao leading dimensions must match dm1_ao.")
    return dm1_ao, tdm1_ao


def _cartesian_density_arrays(dm1_ao, tdm1_ao, transform):
    if transform is None:
        return (
            np.ascontiguousarray(dm1_ao, dtype=np.complex128),
            np.ascontiguousarray(tdm1_ao, dtype=np.complex128),
        )
    transform = np.asarray(transform)
    dm1_cart = np.einsum("ci,sij,dj->scd", transform, dm1_ao, transform, optimize=True)
    tdm1_cart = np.einsum(
        "ci,stij,dj->stcd", transform, tdm1_ao, transform, optimize=True
    )
    return (
        np.ascontiguousarray(dm1_cart, dtype=np.complex128),
        np.ascontiguousarray(tdm1_cart, dtype=np.complex128),
    )


def _basis_atom_indices(basis, atom_coords, tol=1e-8):
    atom_coords = np.asarray(atom_coords, dtype=float)
    indices = np.empty(len(basis), dtype=np.int64)
    for i, fn in enumerate(basis):
        origin = np.asarray(fn.origin, dtype=float)
        distances = np.linalg.norm(atom_coords - origin[None, :], axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] > tol:
            raise ValueError("AO origin does not match any atom coordinate.")
        indices[i] = idx
    return np.ascontiguousarray(indices)


def has_compiled_ao_ft():
    """Return whether the optional compiled AO Fourier backend is available."""
    return _ao_pair_ft_matrices_numba is not None


if njit is not None:

    @njit(cache=True)
    def _E_compiled(i, j, t, qx, a, b):
        p = a + b
        q = a * b / p
        if t < 0 or t > i + j:
            return 0.0
        if i == 0 and j == 0 and t == 0:
            return math.exp(-q * qx * qx)
        if j == 0:
            return (
                (1.0 / (2.0 * p)) * _E_compiled(i - 1, j, t - 1, qx, a, b)
                - (q * qx / a) * _E_compiled(i - 1, j, t, qx, a, b)
                + (t + 1.0) * _E_compiled(i - 1, j, t + 1, qx, a, b)
            )
        return (
            (1.0 / (2.0 * p)) * _E_compiled(i, j - 1, t - 1, qx, a, b)
            + (q * qx / b) * _E_compiled(i, j - 1, t, qx, a, b)
            + (t + 1.0) * _E_compiled(i, j - 1, t + 1, qx, a, b)
        )

    @njit(cache=True)
    def _complex_ipow(z, n):
        out = 1.0 + 0.0j
        for _ in range(n):
            out *= z
        return out

    @njit(cache=True, parallel=True)
    def _ao_pair_ft_matrices_numba(
        shells,
        origins,
        gvecs,
        pair_p,
        pair_q,
        prim_start,
        prim_alpha,
        prim_beta,
        prim_alpha_over_p,
        prim_beta_over_p,
        prim_inv_4p,
        prim_prefactor,
        plane_z,
    ):
        ng = gvecs.shape[0]
        nao = shells.shape[0]
        npair = pair_p.shape[0]
        out = np.empty((ng, nao, nao), dtype=np.complex128)

        for flat in prange(npair * ng):
            pair_idx = flat // ng
            gidx = flat - pair_idx * ng
            pidx = pair_p[pair_idx]
            qidx = pair_q[pair_idx]

            l1 = shells[pidx, 0]
            m1 = shells[pidx, 1]
            n1 = shells[pidx, 2]
            ax = origins[pidx, 0]
            ay = origins[pidx, 1]
            az = origins[pidx, 2]

            l2 = shells[qidx, 0]
            m2 = shells[qidx, 1]
            n2 = shells[qidx, 2]
            bx = origins[qidx, 0]
            by = origins[qidx, 1]
            bz = origins[qidx, 2]
            abx = ax - bx
            aby = ay - by
            abz = az - bz

            gxv = gvecs[gidx, 0]
            gyv = gvecs[gidx, 1]
            gzv = gvecs[gidx, 2]
            g2 = gxv * gxv + gyv * gyv + gzv * gzv
            gx = -1j * gxv
            gy = -1j * gyv
            gz = -1j * gzv
            value = 0.0 + 0.0j

            v_stop = n1 + n2 + 1
            if plane_z:
                v_stop = 1

            for iprim in range(prim_start[pair_idx], prim_start[pair_idx + 1]):
                alpha = prim_alpha[iprim]
                beta = prim_beta[iprim]
                cx = prim_alpha_over_p[iprim] * ax + prim_beta_over_p[iprim] * bx
                cy = prim_alpha_over_p[iprim] * ay + prim_beta_over_p[iprim] * by
                cz = prim_alpha_over_p[iprim] * az + prim_beta_over_p[iprim] * bz
                damping = math.exp(-g2 * prim_inv_4p[iprim])
                phase_arg = gxv * cx + gyv * cy + gzv * cz
                phase = math.cos(phase_arg) - 1j * math.sin(phase_arg)
                primitive = 0.0 + 0.0j

                for t in range(l1 + l2 + 1):
                    ex = _E_compiled(l1, l2, t, abx, alpha, beta)
                    gx_t = _complex_ipow(gx, t)
                    for u in range(m1 + m2 + 1):
                        exy = ex * _E_compiled(m1, m2, u, aby, alpha, beta)
                        gxy = gx_t * _complex_ipow(gy, u)
                        for v in range(v_stop):
                            primitive += (
                                exy
                                * _E_compiled(n1, n2, v, abz, alpha, beta)
                                * gxy
                                * _complex_ipow(gz, v)
                            )

                value += prim_prefactor[iprim] * damping * phase * primitive

            out[gidx, pidx, qidx] = value
            out[gidx, qidx, pidx] = value

        return out

    @njit(cache=True)
    def _pair_ft_value_numba(
        pair_idx,
        gidx,
        shells,
        origins,
        gvecs,
        pair_p,
        pair_q,
        prim_start,
        prim_alpha,
        prim_beta,
        prim_alpha_over_p,
        prim_beta_over_p,
        prim_inv_4p,
        prim_prefactor,
        plane_z,
    ):
        pidx = pair_p[pair_idx]
        qidx = pair_q[pair_idx]
        l1 = shells[pidx, 0]
        m1 = shells[pidx, 1]
        n1 = shells[pidx, 2]
        l2 = shells[qidx, 0]
        m2 = shells[qidx, 1]
        n2 = shells[qidx, 2]
        ax = origins[pidx, 0]
        ay = origins[pidx, 1]
        az = origins[pidx, 2]
        bx = origins[qidx, 0]
        by = origins[qidx, 1]
        bz = origins[qidx, 2]
        abx = ax - bx
        aby = ay - by
        abz = az - bz
        gxv = gvecs[gidx, 0]
        gyv = gvecs[gidx, 1]
        gzv = gvecs[gidx, 2]
        g2 = gxv * gxv + gyv * gyv + gzv * gzv
        gx = -1j * gxv
        gy = -1j * gyv
        gz = -1j * gzv
        v_stop = n1 + n2 + 1
        if plane_z:
            v_stop = 1
        value = 0.0 + 0.0j

        for iprim in range(prim_start[pair_idx], prim_start[pair_idx + 1]):
            alpha = prim_alpha[iprim]
            beta = prim_beta[iprim]
            cx = prim_alpha_over_p[iprim] * ax + prim_beta_over_p[iprim] * bx
            cy = prim_alpha_over_p[iprim] * ay + prim_beta_over_p[iprim] * by
            cz = prim_alpha_over_p[iprim] * az + prim_beta_over_p[iprim] * bz
            damping = math.exp(-g2 * prim_inv_4p[iprim])
            phase_arg = gxv * cx + gyv * cy + gzv * cz
            phase = math.cos(phase_arg) - 1j * math.sin(phase_arg)
            primitive = 0.0 + 0.0j

            for t in range(l1 + l2 + 1):
                ex = _E_compiled(l1, l2, t, abx, alpha, beta)
                gx_t = _complex_ipow(gx, t)
                for u in range(m1 + m2 + 1):
                    exy = ex * _E_compiled(m1, m2, u, aby, alpha, beta)
                    gxy = gx_t * _complex_ipow(gy, u)
                    for v in range(v_stop):
                        primitive += (
                            exy
                            * _E_compiled(n1, n2, v, abz, alpha, beta)
                            * gxy
                            * _complex_ipow(gz, v)
                        )

            value += prim_prefactor[iprim] * damping * phase * primitive

        return value

    @njit(cache=True, parallel=True)
    def _density_fts_numba(
        shells,
        origins,
        gvecs,
        pair_p,
        pair_q,
        prim_start,
        prim_alpha,
        prim_beta,
        prim_alpha_over_p,
        prim_beta_over_p,
        prim_inv_4p,
        prim_prefactor,
        plane_z,
        dm1,
        tdm1,
    ):
        ng = gvecs.shape[0]
        nstates = dm1.shape[0]
        npair = pair_p.shape[0]
        ft_ii = np.zeros((nstates, ng), dtype=np.complex128)
        ft_ij = np.zeros((nstates, nstates, ng), dtype=np.complex128)

        for gidx in prange(ng):
            for pair_idx in range(npair):
                pidx = pair_p[pair_idx]
                qidx = pair_q[pair_idx]
                pair_value = _pair_ft_value_numba(
                    pair_idx,
                    gidx,
                    shells,
                    origins,
                    gvecs,
                    pair_p,
                    pair_q,
                    prim_start,
                    prim_alpha,
                    prim_beta,
                    prim_alpha_over_p,
                    prim_beta_over_p,
                    prim_inv_4p,
                    prim_prefactor,
                    plane_z,
                )
                if pidx == qidx:
                    for i in range(nstates):
                        ft_ii[i, gidx] += dm1[i, pidx, pidx] * pair_value
                        for j in range(nstates):
                            ft_ij[i, j, gidx] += tdm1[i, j, pidx, pidx] * pair_value
                else:
                    for i in range(nstates):
                        ft_ii[i, gidx] += (
                            dm1[i, pidx, qidx] + dm1[i, qidx, pidx]
                        ) * pair_value
                        for j in range(nstates):
                            ft_ij[i, j, gidx] += (
                                tdm1[i, j, pidx, qidx] + tdm1[i, j, qidx, pidx]
                            ) * pair_value
        return ft_ii, ft_ij

    @njit(cache=True, parallel=True)
    def _density_fts_batch_numba(
        shells,
        origins_batch,
        gvecs,
        pair_p,
        pair_q,
        prim_start,
        prim_alpha,
        prim_beta,
        prim_alpha_over_p,
        prim_beta_over_p,
        prim_inv_4p,
        prim_prefactor,
        plane_z,
        dm1,
        tdm1,
    ):
        ngeom = origins_batch.shape[0]
        ng = gvecs.shape[0]
        nstates = dm1.shape[1]
        npair = pair_p.shape[0]
        ft_ii = np.zeros((ngeom, nstates, ng), dtype=np.complex128)
        ft_ij = np.zeros((ngeom, nstates, nstates, ng), dtype=np.complex128)

        for flat in prange(ngeom * ng):
            geom = flat // ng
            gidx = flat - geom * ng
            origins = origins_batch[geom]
            for pair_idx in range(npair):
                pidx = pair_p[pair_idx]
                qidx = pair_q[pair_idx]
                pair_value = _pair_ft_value_numba(
                    pair_idx,
                    gidx,
                    shells,
                    origins,
                    gvecs,
                    pair_p,
                    pair_q,
                    prim_start,
                    prim_alpha,
                    prim_beta,
                    prim_alpha_over_p,
                    prim_beta_over_p,
                    prim_inv_4p,
                    prim_prefactor,
                    plane_z,
                )
                if pidx == qidx:
                    for i in range(nstates):
                        ft_ii[geom, i, gidx] += dm1[geom, i, pidx, pidx] * pair_value
                        for j in range(nstates):
                            ft_ij[geom, i, j, gidx] += (
                                tdm1[geom, i, j, pidx, pidx] * pair_value
                            )
                else:
                    for i in range(nstates):
                        ft_ii[geom, i, gidx] += (
                            dm1[geom, i, pidx, qidx] + dm1[geom, i, qidx, pidx]
                        ) * pair_value
                        for j in range(nstates):
                            ft_ij[geom, i, j, gidx] += (
                                tdm1[geom, i, j, pidx, qidx]
                                + tdm1[geom, i, j, qidx, pidx]
                            ) * pair_value
        return ft_ii, ft_ij

else:
    _ao_pair_ft_matrices_numba = None
    _density_fts_numba = None
    _density_fts_batch_numba = None


def gaussian_pair_ft(a, b, gvec):
    """
    Fourier transform of a Cartesian Gaussian AO pair density.

    Returns int chi_a(r) chi_b(r) exp(-i G.r) dr for contracted, normalized
    Cartesian Gaussians from the builtin basis representation.
    """
    gvec = _validate_gvec(gvec)
    a_origin = np.asarray(a.origin, dtype=float)
    b_origin = np.asarray(b.origin, dtype=float)
    g2 = float(np.dot(gvec, gvec))
    l1, m1, n1 = tuple(int(x) for x in a.shell)
    l2, m2, n2 = tuple(int(x) for x in b.shell)

    value = 0.0j
    for ia, wa in enumerate(a.prim_weights):
        alpha = float(a.exps[ia])
        for ib, wb in enumerate(b.prim_weights):
            beta = float(b.exps[ib])
            p = alpha + beta
            center = (alpha * a_origin + beta * b_origin) / p
            prefactor = wa * wb * (math.pi / p) ** 1.5
            damping = math.exp(-g2 / (4.0 * p))
            phase = np.exp(-1j * np.dot(gvec, center))
            primitive = 0.0j
            ab = a_origin - b_origin
            for t in range(l1 + l2 + 1):
                ex = E(l1, l2, t, float(ab[0]), alpha, beta)
                gx = (-1j * gvec[0]) ** t
                for u in range(m1 + m2 + 1):
                    exy = ex * E(m1, m2, u, float(ab[1]), alpha, beta)
                    gxy = gx * ((-1j * gvec[1]) ** u)
                    for v in range(n1 + n2 + 1):
                        primitive += (
                            exy
                            * E(n1, n2, v, float(ab[2]), alpha, beta)
                            * gxy
                            * ((-1j * gvec[2]) ** v)
                        )
            value += prefactor * damping * phase * primitive
    return complex(value)


def gaussian_pair_ft_batch(a, b, gvecs):
    """
    Fourier transform of a Cartesian Gaussian AO pair for many G vectors.

    Returns
    -------
    ndarray, shape (ng,)
        ``int chi_a(r) chi_b(r) exp(-i G.r) dr`` for every row of ``gvecs``.
    """
    gvecs = _validate_gvecs(gvecs)
    a_origin = np.asarray(a.origin, dtype=float)
    b_origin = np.asarray(b.origin, dtype=float)
    g2 = np.einsum("gi,gi->g", gvecs, gvecs)
    l1, m1, n1 = tuple(int(x) for x in a.shell)
    l2, m2, n2 = tuple(int(x) for x in b.shell)

    value = np.zeros(len(gvecs), dtype=np.complex128)
    ab = a_origin - b_origin
    gx = -1j * gvecs[:, 0]
    gy = -1j * gvecs[:, 1]
    gz = -1j * gvecs[:, 2]

    for ia, wa in enumerate(a.prim_weights):
        alpha = float(a.exps[ia])
        for ib, wb in enumerate(b.prim_weights):
            beta = float(b.exps[ib])
            p = alpha + beta
            center = (alpha * a_origin + beta * b_origin) / p
            prefactor = wa * wb * (math.pi / p) ** 1.5
            damping = np.exp(-g2 / (4.0 * p))
            phase = np.exp(-1j * (gvecs @ center))
            primitive = np.zeros(len(gvecs), dtype=np.complex128)

            for t in range(l1 + l2 + 1):
                ex = E(l1, l2, t, float(ab[0]), alpha, beta)
                gx_t = gx**t
                for u in range(m1 + m2 + 1):
                    exy = ex * E(m1, m2, u, float(ab[1]), alpha, beta)
                    gxy = gx_t * (gy**u)
                    for v in range(n1 + n2 + 1):
                        primitive += (
                            exy
                            * E(n1, n2, v, float(ab[2]), alpha, beta)
                            * gxy
                            * (gz**v)
                        )
            value += prefactor * damping * phase * primitive
    return value


def gaussian_pair_ft_s(a, b, gvec):
    """Backward-compatible alias for Gaussian AO pair Fourier factors."""
    return gaussian_pair_ft(a, b, gvec)


def ao_pair_ft_matrix_s(basis, gvec):
    """Return all AO pair Fourier factors for a builtin Cartesian AO basis."""
    basis = tuple(basis)

    nao = len(basis)
    out = np.empty((nao, nao), dtype=np.complex128)
    for p, bp in enumerate(basis):
        for q, bq in enumerate(basis):
            out[p, q] = gaussian_pair_ft_s(bp, bq, gvec)
    return out


class AOPairFTPlan:
    """Reusable native AO-pair Fourier plan for one Gaussian basis topology."""

    def __init__(self, basis, transform=None, atom_coords=None):
        self.basis = tuple(basis)
        self.transform = None if transform is None else np.asarray(transform)
        self.shells, self.origins, exps, weights, nprim = _pack_basis_for_compiled(
            self.basis
        )
        self.pair_p, self.pair_q = _upper_pair_indices(len(self.basis))
        (
            self.prim_start,
            self.prim_alpha,
            self.prim_beta,
            self.prim_alpha_over_p,
            self.prim_beta_over_p,
            self.prim_inv_4p,
            self.prim_prefactor,
        ) = _pack_pair_primitives(self.pair_p, self.pair_q, exps, weights, nprim)
        self.atom_indices = None
        if atom_coords is not None:
            self.atom_indices = _basis_atom_indices(self.basis, atom_coords)

    @classmethod
    def from_molecule(cls, mol):
        basis, transform = mol._cart_basis()
        atom_coords = mol.atom_coords() if hasattr(mol, "atom_coords") else None
        return cls(basis, transform=transform, atom_coords=atom_coords)

    @property
    def ncart(self):
        return len(self.basis)

    @property
    def nao(self):
        if self.transform is None:
            return self.ncart
        return self.transform.shape[1]

    def origins_from_atom_coords(self, atom_coords):
        if self.atom_indices is None:
            raise ValueError("This AO FT plan was not initialized with atom coordinates.")
        atom_coords = np.asarray(atom_coords, dtype=float)
        return np.ascontiguousarray(atom_coords[self.atom_indices])

    def _origin_array(self, origins=None):
        if origins is None:
            return self.origins
        origins = np.asarray(origins, dtype=float)
        if origins.shape != self.origins.shape:
            raise ValueError(f"origins must have shape {self.origins.shape}.")
        return np.ascontiguousarray(origins)

    def matrices(self, gvecs, origins=None, compiled=False, plane_z=None, plane_tol=0.0):
        gvecs, plane_z = _plane_gvecs(gvecs, plane_z=plane_z, plane_tol=plane_tol)
        origins = self._origin_array(origins)
        if compiled is True:
            if _ao_pair_ft_matrices_numba is None:
                if origins is not self.origins:
                    raise RuntimeError("Numba is required when overriding AO origins.")
                return ao_pair_ft_matrices(self.basis, gvecs, compiled=False)
            return _ao_pair_ft_matrices_numba(
                self.shells,
                origins,
                gvecs,
                self.pair_p,
                self.pair_q,
                self.prim_start,
                self.prim_alpha,
                self.prim_beta,
                self.prim_alpha_over_p,
                self.prim_beta_over_p,
                self.prim_inv_4p,
                self.prim_prefactor,
                plane_z,
            )
        if origins is not self.origins:
            raise RuntimeError("compiled=True is required when overriding AO origins.")
        return ao_pair_ft_matrices(self.basis, gvecs, compiled=False)

    def contract(
        self,
        dm1_ao,
        tdm1_ao,
        gvecs,
        origins=None,
        compiled=True,
        plane_z=None,
        plane_tol=0.0,
    ):
        dm1_ao, tdm1_ao = _as_density_arrays(dm1_ao, tdm1_ao, self.nao)
        dm1_cart, tdm1_cart = _cartesian_density_arrays(
            dm1_ao, tdm1_ao, self.transform
        )
        gvecs, plane_z = _plane_gvecs(gvecs, plane_z=plane_z, plane_tol=plane_tol)
        origins = self._origin_array(origins)

        if compiled is True and _density_fts_numba is not None:
            return _density_fts_numba(
                self.shells,
                origins,
                gvecs,
                self.pair_p,
                self.pair_q,
                self.prim_start,
                self.prim_alpha,
                self.prim_beta,
                self.prim_alpha_over_p,
                self.prim_beta_over_p,
                self.prim_inv_4p,
                self.prim_prefactor,
                plane_z,
                dm1_cart,
                tdm1_cart,
            )

        pair_cart = self.matrices(gvecs, origins=origins, compiled=False)
        ft_ii = np.einsum("imn,qmn->iq", dm1_cart, pair_cart, optimize=True)
        ft_ij = np.einsum("ijmn,qmn->ijq", tdm1_cart, pair_cart, optimize=True)
        return ft_ii, ft_ij

    def contract_batch(
        self,
        dm1_ao,
        tdm1_ao,
        gvecs,
        origins_batch,
        compiled=True,
        plane_z=None,
        plane_tol=0.0,
    ):
        dm1_ao = np.asarray(dm1_ao)
        tdm1_ao = np.asarray(tdm1_ao)
        origins_batch = np.ascontiguousarray(origins_batch, dtype=float)
        if origins_batch.ndim != 3 or origins_batch.shape[1:] != self.origins.shape:
            raise ValueError(
                f"origins_batch must have shape (ngeom, {self.ncart}, 3)."
            )
        if dm1_ao.ndim != 4 or dm1_ao.shape[2:] != (self.nao, self.nao):
            raise ValueError(
                f"dm1_ao must have shape (ngeom, nstates, {self.nao}, {self.nao})."
            )
        if tdm1_ao.ndim != 5 or tdm1_ao.shape[3:] != (self.nao, self.nao):
            raise ValueError(
                "tdm1_ao must have shape "
                f"(ngeom, nstates, nstates, {self.nao}, {self.nao})."
            )
        if dm1_ao.shape[0] != origins_batch.shape[0] or tdm1_ao.shape[0] != origins_batch.shape[0]:
            raise ValueError("Density and origin batches must have the same length.")

        dm1_cart = []
        tdm1_cart = []
        for dm1, tdm1 in zip(dm1_ao, tdm1_ao):
            dm1_i, tdm1_i = _cartesian_density_arrays(dm1, tdm1, self.transform)
            dm1_cart.append(dm1_i)
            tdm1_cart.append(tdm1_i)
        dm1_cart = np.ascontiguousarray(dm1_cart, dtype=np.complex128)
        tdm1_cart = np.ascontiguousarray(tdm1_cart, dtype=np.complex128)
        gvecs, plane_z = _plane_gvecs(gvecs, plane_z=plane_z, plane_tol=plane_tol)

        if compiled is not True or _density_fts_batch_numba is None:
            ft_ii = []
            ft_ij = []
            for dm1, tdm1, origins in zip(dm1_ao, tdm1_ao, origins_batch):
                a, b = self.contract(
                    dm1,
                    tdm1,
                    gvecs,
                    origins=origins,
                    compiled=compiled,
                    plane_z=plane_z,
                )
                ft_ii.append(a)
                ft_ij.append(b)
            return np.asarray(ft_ii), np.asarray(ft_ij)

        return _density_fts_batch_numba(
            self.shells,
            origins_batch,
            gvecs,
            self.pair_p,
            self.pair_q,
            self.prim_start,
            self.prim_alpha,
            self.prim_beta,
            self.prim_alpha_over_p,
            self.prim_beta_over_p,
            self.prim_inv_4p,
            self.prim_prefactor,
            plane_z,
            dm1_cart,
            tdm1_cart,
        )


def ao_pair_ft_matrices_compiled(basis, gvecs, plane_z=None, plane_tol=0.0):
    """Return AO-pair Fourier factors using the optional compiled backend."""
    return AOPairFTPlan(basis).matrices(
        gvecs, compiled=True, plane_z=plane_z, plane_tol=plane_tol
    )


def ao_density_ft(
    basis,
    dm1_ao,
    tdm1_ao,
    gvecs,
    transform=None,
    compiled=True,
    plane_z=None,
    plane_tol=0.0,
):
    """Contract AO-pair Fourier factors with AO density matrices."""
    return AOPairFTPlan(basis, transform=transform).contract(
        dm1_ao,
        tdm1_ao,
        gvecs,
        compiled=compiled,
        plane_z=plane_z,
        plane_tol=plane_tol,
    )


def ao_pair_ft_matrices(basis, gvecs, compiled=False, plane_z=None, plane_tol=0.0):
    """Return AO-pair Fourier factors for many G vectors.

    The AO product is symmetric for real Gaussian AOs, so only the upper
    triangle is evaluated explicitly.

    Parameters
    ----------
    compiled : bool, optional
        Use the optional Numba backend. The vectorized NumPy backend remains
        the default to avoid first-call JIT latency in small jobs.
    """
    basis = tuple(basis)
    gvecs = _validate_gvecs(gvecs)
    if compiled is True:
        return ao_pair_ft_matrices_compiled(
            basis, gvecs, plane_z=plane_z, plane_tol=plane_tol
        )
    if compiled not in (False, None):
        raise ValueError("compiled must be True or False.")

    nao = len(basis)
    out = np.empty((len(gvecs), nao, nao), dtype=np.complex128)
    for p, bp in enumerate(basis):
        for q in range(p, nao):
            values = gaussian_pair_ft_batch(bp, basis[q], gvecs)
            out[:, p, q] = values
            if q != p:
                out[:, q, p] = values
    return out
