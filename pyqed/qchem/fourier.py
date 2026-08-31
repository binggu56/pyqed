#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fourier-space Gaussian AO integrals."""

import math
from importlib import import_module

import numpy as np

from pyqed.qchem.basis import E, _basis_cy, _cart_shell_blocks

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional acceleration
    njit = None

try:  # pragma: no cover - optional C++ acceleration
    _gdf_cpp = import_module("pyqed.qchem._gdf_cpp")
except Exception:  # pragma: no cover - optional acceleration
    _gdf_cpp = None


def has_periodic_pair_ft_backend():
    """Return whether the optional C++ single-phase periodic pair-FT is available."""
    return _gdf_cpp is not None and hasattr(
        _gdf_cpp,
        "periodic_pair_ft_primitive_sum",
    )


def has_periodic_pair_ft_many_backend():
    """Return whether the optional C++ batched periodic pair-FT is available."""
    return _gdf_cpp is not None and hasattr(
        _gdf_cpp,
        "periodic_pair_ft_primitive_sum_many",
    )


def has_periodic_pair_ft_image_group_backend():
    """Return whether image-resolved C++ periodic pair-FT sums are available."""
    return _gdf_cpp is not None and hasattr(
        _gdf_cpp,
        "periodic_pair_ft_primitive_image_group_sum",
    )


def has_periodic_pair_ft_contract_backend():
    """Return whether the optional C++ fused pair-FT contraction is available."""
    return _gdf_cpp is not None and hasattr(
        _gdf_cpp,
        "periodic_pair_ft_primitive_contract_many",
    )


def has_periodic_pair_ft_product_backend():
    """Return whether the optional C++ product-driven periodic pair-FT is available."""
    return _gdf_cpp is not None and hasattr(
        _gdf_cpp,
        "periodic_pair_ft_product_sum_many",
    )


def has_gaussian_ft_backend():
    """Return whether the optional C++ one-center Gaussian FT is available."""
    return _gdf_cpp is not None and hasattr(_gdf_cpp, "gaussian_ft_batch")


def has_cartesian_shell_transform_backend():
    """Return whether the optional C++ shell-block transform is available."""
    return _gdf_cpp is not None and hasattr(_gdf_cpp, "cartesian_shell_transform")


def gaussian_ft_batch_compiled(
    gvecs,
    shells,
    origins,
    exps,
    weights,
    nprim,
    threads=None,
):
    """Return Cartesian one-center Gaussian FTs from the optional C++ backend."""
    if not has_gaussian_ft_backend():
        raise RuntimeError("C++ Gaussian FT backend is not available.")
    args = (
        np.ascontiguousarray(gvecs, dtype=float),
        np.ascontiguousarray(shells, dtype=np.int64),
        np.ascontiguousarray(origins, dtype=float),
        np.ascontiguousarray(exps, dtype=float),
        np.ascontiguousarray(weights, dtype=float),
        np.ascontiguousarray(nprim, dtype=np.int64),
    )
    if threads is None:
        return _gdf_cpp.gaussian_ft_batch(*args)
    return _gdf_cpp.gaussian_ft_batch(*args, int(threads))


def cartesian_shell_transform_compiled(
    cart_ft,
    transform,
    cart_start,
    cart_stop,
    aux_start,
    aux_stop,
    threads=None,
):
    """Apply a shell-block Cartesian-to-auxiliary transform in C++."""
    if not has_cartesian_shell_transform_backend():
        raise RuntimeError("C++ Cartesian shell-transform backend is not available.")
    args = (
        np.ascontiguousarray(cart_ft, dtype=np.complex128),
        np.ascontiguousarray(transform, dtype=float),
        np.ascontiguousarray(cart_start, dtype=np.int64),
        np.ascontiguousarray(cart_stop, dtype=np.int64),
        np.ascontiguousarray(aux_start, dtype=np.int64),
        np.ascontiguousarray(aux_stop, dtype=np.int64),
    )
    if threads is None:
        return _gdf_cpp.cartesian_shell_transform(*args)
    return _gdf_cpp.cartesian_shell_transform(*args, int(threads))


def _pair_ft_cart_shell_blocks(basis):
    try:
        return _cart_shell_blocks(basis)
    except ValueError:
        return [(index, index + 1, int(sum(fn.shell))) for index, fn in enumerate(basis)]


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


def gaussian_ft_moment(order, gcoord, exponent):
    """Fourier transform of one Cartesian Gaussian moment."""

    order = int(order)
    gcoord = np.asarray(gcoord, dtype=float)
    exponent = float(exponent)
    if order < 0:
        raise ValueError("order must be non-negative.")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive.")
    moments = [
        np.sqrt(np.pi / exponent)
        * np.exp(-(gcoord * gcoord) / (4.0 * exponent))
    ]
    if order == 0:
        return moments[0]
    moments.append((-1.0j * gcoord * moments[0]) / (2.0 * exponent))
    for n in range(1, order):
        moments.append(
            (n * moments[n - 1] - 1.0j * gcoord * moments[n])
            / (2.0 * exponent)
        )
    return moments[order]


def contracted_gaussian_ft_batch(function, gvecs):
    """Evaluate one normalized contracted Cartesian Gaussian on a G grid."""

    gvecs = _validate_gvecs(gvecs)
    lx, ly, lz = (int(value) for value in function.shell)
    origin = np.asarray(function.origin, dtype=float)
    phase = np.exp(-1.0j * (gvecs @ origin))
    values = np.zeros(len(gvecs), dtype=np.complex128)
    for exponent, weight in zip(function.exps, function.prim_weights):
        exponent = float(exponent)
        values += (
            float(weight)
            * phase
            * gaussian_ft_moment(lx, gvecs[:, 0], exponent)
            * gaussian_ft_moment(ly, gvecs[:, 1], exponent)
            * gaussian_ft_moment(lz, gvecs[:, 2], exponent)
        )
    return values


def gaussian_basis_ft_batch(basis, gvecs, transform=None, threads=None):
    """Evaluate a Cartesian or transformed Gaussian AO basis on a G grid."""

    basis = tuple(basis)
    gvecs = _validate_gvecs(gvecs)
    if not basis:
        cartesian = np.empty((len(gvecs), 0), dtype=np.complex128)
    elif has_gaussian_ft_backend():
        shells, origins, exps, weights, nprim = _pack_basis_for_compiled(basis)
        cartesian = gaussian_ft_batch_compiled(
            gvecs,
            shells,
            origins,
            exps,
            weights,
            nprim,
            threads=threads,
        )
    else:
        cartesian = np.vstack(
            [contracted_gaussian_ft_batch(function, gvecs) for function in basis]
        ).T
    cartesian = np.asarray(cartesian, dtype=np.complex128)
    if transform is None:
        return np.ascontiguousarray(cartesian)
    transform = np.asarray(transform, dtype=float)
    if transform.ndim != 2 or transform.shape[0] != cartesian.shape[1]:
        raise ValueError(
            "transform must have shape (ncartesian, nao); got "
            f"{transform.shape}."
        )
    return np.ascontiguousarray(cartesian @ transform)


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

    @njit(cache=True, parallel=True)
    def _ao_block_pair_ft_sum_numba(
        shells,
        left_origins,
        right_origins_batch,
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
        phases,
        pair_image_starts,
        pair_image_indices,
        plane_z,
    ):
        nleft = left_origins.shape[0]
        ng = gvecs.shape[0]
        npair = pair_p.shape[0]
        out = np.zeros((ng, nleft, right_origins_batch.shape[1]), dtype=np.complex128)

        for flat in prange(npair * ng):
            pair_idx = flat // ng
            gidx = flat - pair_idx * ng
            pidx = pair_p[pair_idx]
            qidx = pair_q[pair_idx]
            qlocal = qidx - nleft

            l1 = shells[pidx, 0]
            m1 = shells[pidx, 1]
            n1 = shells[pidx, 2]
            ax = left_origins[pidx, 0]
            ay = left_origins[pidx, 1]
            az = left_origins[pidx, 2]

            l2 = shells[qidx, 0]
            m2 = shells[qidx, 1]
            n2 = shells[qidx, 2]

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
            for cursor in range(pair_image_starts[pair_idx], pair_image_starts[pair_idx + 1]):
                image = pair_image_indices[cursor]
                bx = right_origins_batch[image, qlocal, 0]
                by = right_origins_batch[image, qlocal, 1]
                bz = right_origins_batch[image, qlocal, 2]
                abx = ax - bx
                aby = ay - by
                abz = az - bz
                image_value = 0.0 + 0.0j

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

                    image_value += prim_prefactor[iprim] * damping * phase * primitive

                value += phases[image] * image_value

            out[gidx, pidx, qlocal] = value

        return out

    @njit(cache=True, parallel=True)
    def _ao_block_pair_ft_primitive_sum_numba(
        gvecs,
        pair_p,
        pair_q,
        nleft,
        nright,
        pair_term_starts,
        term_image,
        term_center,
        term_inv_4p,
        term_coeff,
        term_power,
        phases,
        plane_z,
    ):
        ng = gvecs.shape[0]
        npair = pair_p.shape[0]
        out = np.zeros((ng, nleft, nright), dtype=np.complex128)

        for flat in prange(npair * ng):
            pair_idx = flat // ng
            gidx = flat - pair_idx * ng
            pidx = pair_p[pair_idx]
            qlocal = pair_q[pair_idx] - nleft

            gxv = gvecs[gidx, 0]
            gyv = gvecs[gidx, 1]
            gzv = gvecs[gidx, 2]
            g2 = gxv * gxv + gyv * gyv + gzv * gzv
            gx = -1j * gxv
            gy = -1j * gyv
            gz = -1j * gzv

            value = 0.0 + 0.0j
            for term in range(pair_term_starts[pair_idx], pair_term_starts[pair_idx + 1]):
                tv = term_power[term, 2]
                if plane_z and tv != 0:
                    continue
                phase_arg = (
                    gxv * term_center[term, 0]
                    + gyv * term_center[term, 1]
                    + gzv * term_center[term, 2]
                )
                phase = math.cos(phase_arg) - 1j * math.sin(phase_arg)
                value += (
                    phases[term_image[term]]
                    * term_coeff[term]
                    * math.exp(-g2 * term_inv_4p[term])
                    * phase
                    * _complex_ipow(gx, term_power[term, 0])
                    * _complex_ipow(gy, term_power[term, 1])
                    * _complex_ipow(gz, tv)
                )

            out[gidx, pidx, qlocal] = value

        return out

else:
    _ao_pair_ft_matrices_numba = None
    _density_fts_numba = None
    _density_fts_batch_numba = None
    _ao_block_pair_ft_sum_numba = None
    _ao_block_pair_ft_primitive_sum_numba = None


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


class AOBlockPairFTPlan:
    """Reusable AO-pair Fourier plan for all pairs between two Gaussian bases."""

    def __init__(self, left_basis, right_basis):
        self.left_basis = tuple(left_basis)
        self.right_basis = tuple(right_basis)
        self.nleft = len(self.left_basis)
        self.nright = len(self.right_basis)
        self.basis = self.left_basis + self.right_basis
        self.shells, self.origins, exps, weights, nprim = _pack_basis_for_compiled(
            self.basis
        )
        left = np.repeat(np.arange(self.nleft, dtype=np.int64), self.nright)
        right = self.nleft + np.tile(np.arange(self.nright, dtype=np.int64), self.nleft)
        self.pair_p = np.ascontiguousarray(left, dtype=np.int64)
        self.pair_q = np.ascontiguousarray(right, dtype=np.int64)
        (
            self.prim_start,
            self.prim_alpha,
            self.prim_beta,
            self.prim_alpha_over_p,
            self.prim_beta_over_p,
            self.prim_inv_4p,
            self.prim_prefactor,
        ) = _pack_pair_primitives(self.pair_p, self.pair_q, exps, weights, nprim)

    @property
    def left_origins(self):
        return self.origins[: self.nleft]

    @property
    def right_origins(self):
        return self.origins[self.nleft :]

    def _origin_array(self, left_origins=None, right_origins=None):
        if left_origins is None:
            left_origins = self.left_origins
        else:
            left_origins = np.asarray(left_origins, dtype=float)
            if left_origins.shape != self.left_origins.shape:
                raise ValueError(f"left_origins must have shape {self.left_origins.shape}.")
        if right_origins is None:
            right_origins = self.right_origins
        else:
            right_origins = np.asarray(right_origins, dtype=float)
            if right_origins.shape != self.right_origins.shape:
                raise ValueError(f"right_origins must have shape {self.right_origins.shape}.")
        return np.ascontiguousarray(np.vstack([left_origins, right_origins]))

    def matrices(
        self,
        gvecs,
        left_origins=None,
        right_origins=None,
        compiled=True,
        plane_z=None,
        plane_tol=0.0,
    ):
        gvecs, plane_z = _plane_gvecs(gvecs, plane_z=plane_z, plane_tol=plane_tol)
        origins = self._origin_array(left_origins, right_origins)
        origins_overridden = left_origins is not None or right_origins is not None

        if compiled is True and _ao_pair_ft_matrices_numba is not None:
            values = _ao_pair_ft_matrices_numba(
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
            return np.ascontiguousarray(values[:, : self.nleft, self.nleft :])

        if origins_overridden:
            raise RuntimeError("compiled=True is required when overriding AO origins.")
        if compiled not in (False, None, True):
            raise ValueError("compiled must be True or False.")

        out = np.empty((len(gvecs), self.nleft, self.nright), dtype=np.complex128)
        for p, bp in enumerate(self.left_basis):
            for q, bq in enumerate(self.right_basis):
                out[:, p, q] = gaussian_pair_ft_batch(bp, bq, gvecs)
        return out

    def periodic_primitive_terms(
        self,
        left_origins,
        right_origins_batch,
        image_pair_mask=None,
        coeff_tol=0.0,
        compiled=True,
    ):
        left_origins = np.ascontiguousarray(left_origins, dtype=float)
        right_origins_batch = np.ascontiguousarray(right_origins_batch, dtype=float)
        if left_origins.shape != self.left_origins.shape:
            raise ValueError(f"left_origins must have shape {self.left_origins.shape}.")
        if (
            right_origins_batch.ndim != 3
            or right_origins_batch.shape[1:] != self.right_origins.shape
        ):
            raise ValueError(
                "right_origins_batch must have shape "
                f"(nimage, {self.nright}, 3)."
            )
        nimage = right_origins_batch.shape[0]
        npair = len(self.pair_p)
        if image_pair_mask is None:
            image_pair_mask = np.ones((nimage, npair), dtype=np.bool_)
        else:
            image_pair_mask = np.ascontiguousarray(image_pair_mask, dtype=np.bool_)
            if image_pair_mask.shape != (nimage, npair):
                raise ValueError(
                    "image_pair_mask must have shape "
                    f"(nimage, {npair})."
                )

        if compiled and hasattr(_basis_cy, "compute_periodic_pair_ft_primitive_terms"):
            try:
                result = _basis_cy.compute_periodic_pair_ft_primitive_terms(
                    np.ascontiguousarray(self.shells, dtype=np.int64),
                    left_origins,
                    right_origins_batch,
                    np.ascontiguousarray(self.pair_p, dtype=np.int64),
                    np.ascontiguousarray(self.pair_q, dtype=np.int64),
                    self.nleft,
                    np.ascontiguousarray(self.prim_start, dtype=np.int64),
                    np.ascontiguousarray(self.prim_alpha, dtype=float),
                    np.ascontiguousarray(self.prim_beta, dtype=float),
                    np.ascontiguousarray(self.prim_alpha_over_p, dtype=float),
                    np.ascontiguousarray(self.prim_beta_over_p, dtype=float),
                    np.ascontiguousarray(self.prim_inv_4p, dtype=float),
                    np.ascontiguousarray(self.prim_prefactor, dtype=float),
                    np.ascontiguousarray(image_pair_mask, dtype=np.uint8),
                    float(coeff_tol),
                )
            except NotImplementedError:
                pass
            else:
                result["builder_backend"] = "compiled"
                return result
        if compiled not in (False, None, True):
            raise ValueError("compiled must be True or False.")

        starts = [0]
        image_group_starts = [0]
        image_group_images = []
        image_group_term_starts = []
        image_group_term_stops = []
        image_group_product_starts = []
        image_group_product_stops = []
        product_group_term_starts = []
        product_group_term_stops = []
        product_group_factor_ids = []
        product_factor_ids = {}
        images = []
        centers = []
        inv_4p = []
        coeffs = []
        powers = []
        coeff_tol = max(float(coeff_tol), 0.0)

        for pair_idx, (pidx, qidx) in enumerate(zip(self.pair_p, self.pair_q)):
            qlocal = int(qidx - self.nleft)
            bp = self.left_basis[int(pidx)]
            bq = self.right_basis[qlocal]
            l1, m1, n1 = (int(x) for x in bp.shell)
            l2, m2, n2 = (int(x) for x in bq.shell)
            ax, ay, az = left_origins[int(pidx)]

            for image in np.nonzero(image_pair_mask[:, pair_idx])[0]:
                image_term_start = len(images)
                image_product_start = len(product_group_term_starts)
                bx, by, bz = right_origins_batch[int(image), qlocal]
                abx = ax - bx
                aby = ay - by
                abz = az - bz
                for ia, wa in enumerate(bp.prim_weights):
                    alpha = float(bp.exps[ia])
                    wa = float(wa)
                    for ib, wb in enumerate(bq.prim_weights):
                        beta = float(bq.exps[ib])
                        pexp = alpha + beta
                        alpha_over_p = alpha / pexp
                        beta_over_p = beta / pexp
                        center = (
                            alpha_over_p * ax + beta_over_p * bx,
                            alpha_over_p * ay + beta_over_p * by,
                            alpha_over_p * az + beta_over_p * bz,
                        )
                        prefactor = float(wa) * float(wb) * (math.pi / pexp) ** 1.5
                        primitive_inv_4p = 1.0 / (4.0 * pexp)
                        product_term_start = len(images)
                        for t in range(l1 + l2 + 1):
                            ex = E(l1, l2, t, float(abx), alpha, beta)
                            if ex == 0.0:
                                continue
                            for u in range(m1 + m2 + 1):
                                exy = ex * E(m1, m2, u, float(aby), alpha, beta)
                                if exy == 0.0:
                                    continue
                                for v in range(n1 + n2 + 1):
                                    coeff = exy * E(n1, n2, v, float(abz), alpha, beta)
                                    coeff *= prefactor
                                    if abs(coeff) <= coeff_tol:
                                        continue
                                    images.append(int(image))
                                    centers.append(center)
                                    inv_4p.append(primitive_inv_4p)
                                    coeffs.append(float(coeff))
                                    powers.append((t, u, v))
                        if len(images) > product_term_start:
                            product_factor_key = (
                                center[0],
                                center[1],
                                center[2],
                                primitive_inv_4p,
                            )
                            product_factor_id = product_factor_ids.setdefault(
                                product_factor_key,
                                len(product_factor_ids),
                            )
                            product_group_term_starts.append(product_term_start)
                            product_group_term_stops.append(len(images))
                            product_group_factor_ids.append(product_factor_id)
                if len(images) > image_term_start:
                    image_group_images.append(int(image))
                    image_group_term_starts.append(image_term_start)
                    image_group_term_stops.append(len(images))
                    image_group_product_starts.append(image_product_start)
                    image_group_product_stops.append(len(product_group_term_starts))
            starts.append(len(images))
            image_group_starts.append(len(image_group_images))

        result = {
            "pair_term_starts": np.ascontiguousarray(starts, dtype=np.int64),
            "term_image": np.ascontiguousarray(images, dtype=np.int64),
            "term_center": np.ascontiguousarray(centers, dtype=float).reshape(-1, 3),
            "term_inv_4p": np.ascontiguousarray(inv_4p, dtype=float),
            "term_coeff": np.ascontiguousarray(coeffs, dtype=float),
            "term_power": np.ascontiguousarray(powers, dtype=np.int64).reshape(-1, 3),
            "pair_image_group_starts": np.ascontiguousarray(
                image_group_starts,
                dtype=np.int64,
            ),
            "image_group_image": np.ascontiguousarray(
                image_group_images,
                dtype=np.int64,
            ),
            "image_group_term_start": np.ascontiguousarray(
                image_group_term_starts,
                dtype=np.int64,
            ),
            "image_group_term_stop": np.ascontiguousarray(
                image_group_term_stops,
                dtype=np.int64,
            ),
            "image_group_product_start": np.ascontiguousarray(
                image_group_product_starts,
                dtype=np.int64,
            ),
            "image_group_product_stop": np.ascontiguousarray(
                image_group_product_stops,
                dtype=np.int64,
            ),
            "product_group_term_start": np.ascontiguousarray(
                product_group_term_starts,
                dtype=np.int64,
            ),
            "product_group_term_stop": np.ascontiguousarray(
                product_group_term_stops,
                dtype=np.int64,
            ),
            "product_group_factor_id": np.ascontiguousarray(
                product_group_factor_ids,
                dtype=np.int64,
            ),
            "product_group_factor_count": int(len(product_factor_ids)),
        }
        result["builder_backend"] = "python"
        return result

    def periodic_product_terms(
        self,
        left_origins,
        right_origins_batch,
        image_pair_mask=None,
        coeff_tol=0.0,
    ):
        left_origins = np.ascontiguousarray(left_origins, dtype=float)
        right_origins_batch = np.ascontiguousarray(right_origins_batch, dtype=float)
        if left_origins.shape != self.left_origins.shape:
            raise ValueError(f"left_origins must have shape {self.left_origins.shape}.")
        if (
            right_origins_batch.ndim != 3
            or right_origins_batch.shape[1:] != self.right_origins.shape
        ):
            raise ValueError(
                "right_origins_batch must have shape "
                f"(nimage, {self.nright}, 3)."
            )
        nimage = right_origins_batch.shape[0]
        npair = len(self.pair_p)
        if image_pair_mask is None:
            image_pair_mask = np.ones((nimage, npair), dtype=np.bool_)
        else:
            image_pair_mask = np.ascontiguousarray(image_pair_mask, dtype=np.bool_)
            if image_pair_mask.shape != (nimage, npair):
                raise ValueError(
                    "image_pair_mask must have shape "
                    f"(nimage, {npair})."
                )

        left_blocks = _pair_ft_cart_shell_blocks(self.left_basis)
        right_blocks = _pair_ft_cart_shell_blocks(self.right_basis)
        factor_ids = {}
        factor_centers = []
        factor_inv_4p = []
        product_images = []
        product_factor_ids = []
        product_entry_starts = []
        product_entry_stops = []
        entry_pairs = []
        entry_coeffs = []
        entry_powers = []
        coeff_tol = max(float(coeff_tol), 0.0)

        def factor_id(center, inv_4p_value):
            key = (
                center[0],
                center[1],
                center[2],
                inv_4p_value,
            )
            found = factor_ids.get(key)
            if found is not None:
                return found
            found = len(factor_ids)
            factor_ids[key] = found
            factor_centers.append(center)
            factor_inv_4p.append(float(inv_4p_value))
            return found

        for p0, p1, _lleft in left_blocks:
            for q0, q1, _lright in right_blocks:
                component_pairs = [
                    (p, q, p * self.nright + q)
                    for p in range(p0, p1)
                    for q in range(q0, q1)
                ]
                bp0 = self.left_basis[p0]
                bq0 = self.right_basis[q0]
                ax, ay, az = left_origins[p0]
                l_components = [
                    tuple(int(x) for x in self.left_basis[p].shell)
                    for p in range(p0, p1)
                ]
                r_components = [
                    tuple(int(x) for x in self.right_basis[q].shell)
                    for q in range(q0, q1)
                ]
                for image in range(nimage):
                    active_pairs = [
                        (p, q, pair_idx)
                        for p, q, pair_idx in component_pairs
                        if image_pair_mask[image, pair_idx]
                    ]
                    if not active_pairs:
                        continue
                    bx, by, bz = right_origins_batch[int(image), q0]
                    abx = ax - bx
                    aby = ay - by
                    abz = az - bz
                    for ia, wa in enumerate(bp0.prim_weights):
                        alpha = float(bp0.exps[ia])
                        wa = float(wa)
                        for ib, wb in enumerate(bq0.prim_weights):
                            beta = float(bq0.exps[ib])
                            pexp = alpha + beta
                            alpha_over_p = alpha / pexp
                            beta_over_p = beta / pexp
                            center = (
                                alpha_over_p * ax + beta_over_p * bx,
                                alpha_over_p * ay + beta_over_p * by,
                                alpha_over_p * az + beta_over_p * bz,
                            )
                            prefactor = float(wa) * float(wb) * (math.pi / pexp) ** 1.5
                            primitive_inv_4p = 1.0 / (4.0 * pexp)
                            entry_start = len(entry_pairs)
                            for p, q, pair_idx in active_pairs:
                                l1, m1, n1 = l_components[p - p0]
                                l2, m2, n2 = r_components[q - q0]
                                for t in range(l1 + l2 + 1):
                                    ex = E(l1, l2, t, float(abx), alpha, beta)
                                    if ex == 0.0:
                                        continue
                                    for u in range(m1 + m2 + 1):
                                        exy = ex * E(m1, m2, u, float(aby), alpha, beta)
                                        if exy == 0.0:
                                            continue
                                        for v in range(n1 + n2 + 1):
                                            coeff = exy * E(n1, n2, v, float(abz), alpha, beta)
                                            coeff *= prefactor
                                            if abs(coeff) <= coeff_tol:
                                                continue
                                            entry_pairs.append(int(pair_idx))
                                            entry_coeffs.append(float(coeff))
                                            entry_powers.append((t, u, v))
                            if len(entry_pairs) == entry_start:
                                continue
                            product_images.append(int(image))
                            product_factor_ids.append(factor_id(center, primitive_inv_4p))
                            product_entry_starts.append(entry_start)
                            product_entry_stops.append(len(entry_pairs))

        return {
            "factor_center": np.ascontiguousarray(factor_centers, dtype=float).reshape(-1, 3),
            "factor_inv_4p": np.ascontiguousarray(factor_inv_4p, dtype=float),
            "product_image": np.ascontiguousarray(product_images, dtype=np.int64),
            "product_factor_id": np.ascontiguousarray(product_factor_ids, dtype=np.int64),
            "product_entry_start": np.ascontiguousarray(product_entry_starts, dtype=np.int64),
            "product_entry_stop": np.ascontiguousarray(product_entry_stops, dtype=np.int64),
            "entry_pair": np.ascontiguousarray(entry_pairs, dtype=np.int64),
            "entry_coeff": np.ascontiguousarray(entry_coeffs, dtype=float),
            "entry_power": np.ascontiguousarray(entry_powers, dtype=np.int64).reshape(-1, 3),
            "left_shell_blocks": np.ascontiguousarray(left_blocks, dtype=np.int64).reshape(-1, 3),
            "right_shell_blocks": np.ascontiguousarray(right_blocks, dtype=np.int64).reshape(-1, 3),
        }

    def periodic_sum(
        self,
        gvecs,
        left_origins,
        right_origins_batch,
        phases,
        image_pair_mask=None,
        pair_image_starts=None,
        pair_image_indices=None,
        primitive_terms=None,
        compiled=True,
        plane_z=None,
        plane_tol=0.0,
        threads=None,
    ):
        gvecs, plane_z = _plane_gvecs(gvecs, plane_z=plane_z, plane_tol=plane_tol)
        left_origins = np.ascontiguousarray(left_origins, dtype=float)
        right_origins_batch = np.ascontiguousarray(right_origins_batch, dtype=float)
        phases = np.ascontiguousarray(phases, dtype=np.complex128)
        if left_origins.shape != self.left_origins.shape:
            raise ValueError(f"left_origins must have shape {self.left_origins.shape}.")
        if (
            right_origins_batch.ndim != 3
            or right_origins_batch.shape[1:] != self.right_origins.shape
        ):
            raise ValueError(
                "right_origins_batch must have shape "
                f"(nimage, {self.nright}, 3)."
            )
        if phases.shape != (right_origins_batch.shape[0],):
            raise ValueError("phases must have shape (nimage,).")
        nimage = right_origins_batch.shape[0]
        npair = len(self.pair_p)
        if image_pair_mask is None:
            image_pair_mask = np.ones(
                (nimage, npair),
                dtype=np.bool_,
            )
        else:
            image_pair_mask = np.ascontiguousarray(image_pair_mask, dtype=np.bool_)
            if image_pair_mask.shape != (nimage, npair):
                raise ValueError(
                    "image_pair_mask must have shape "
                    f"(nimage, {npair})."
                )
        if pair_image_starts is None or pair_image_indices is None:
            if pair_image_starts is not None or pair_image_indices is not None:
                raise ValueError(
                    "pair_image_starts and pair_image_indices must be provided together."
                )
            pair_image_starts = np.empty(npair + 1, dtype=np.int64)
            pair_image_starts[0] = 0
            counts = np.count_nonzero(image_pair_mask, axis=0).astype(np.int64)
            np.cumsum(counts, out=pair_image_starts[1:])
            pair_image_indices = np.empty(int(pair_image_starts[-1]), dtype=np.int64)
            for pair_idx in range(npair):
                start = int(pair_image_starts[pair_idx])
                stop = int(pair_image_starts[pair_idx + 1])
                pair_image_indices[start:stop] = np.nonzero(image_pair_mask[:, pair_idx])[0]
        else:
            pair_image_starts = np.ascontiguousarray(pair_image_starts, dtype=np.int64)
            pair_image_indices = np.ascontiguousarray(pair_image_indices, dtype=np.int64)
            if pair_image_starts.shape != (npair + 1,):
                raise ValueError(f"pair_image_starts must have shape ({npair + 1},).")
            if pair_image_starts[0] != 0 or pair_image_starts[-1] != len(pair_image_indices):
                raise ValueError("pair_image_starts must span pair_image_indices.")
            if pair_image_indices.size and (
                np.min(pair_image_indices) < 0 or np.max(pair_image_indices) >= nimage
            ):
                raise ValueError("pair_image_indices contains an out-of-range image index.")

        if (
            primitive_terms is not None
            and compiled is True
            and has_periodic_pair_ft_backend()
        ):
            args = self._primitive_sum_args(
                gvecs,
                primitive_terms,
                phases,
                plane_z,
                threads=threads,
            )
            return _gdf_cpp.periodic_pair_ft_primitive_sum(*args)

        if (
            primitive_terms is not None
            and compiled is True
            and _ao_block_pair_ft_primitive_sum_numba is not None
        ):
            return _ao_block_pair_ft_primitive_sum_numba(
                *self._primitive_sum_args(
                    gvecs,
                    primitive_terms,
                    phases,
                    plane_z,
                    include_image_groups=False,
                )
            )

        if compiled is True and _ao_block_pair_ft_sum_numba is not None:
            return _ao_block_pair_ft_sum_numba(
                self.shells,
                left_origins,
                right_origins_batch,
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
                phases,
                pair_image_starts,
                pair_image_indices,
                plane_z,
            )

        if compiled not in (False, None, True):
            raise ValueError("compiled must be True or False.")

        out = np.zeros((len(gvecs), self.nleft, self.nright), dtype=np.complex128)
        for image, (right_origins, phase) in enumerate(zip(right_origins_batch, phases)):
            if not np.any(image_pair_mask[image]):
                continue
            block = self.matrices(
                gvecs,
                left_origins=left_origins,
                right_origins=right_origins,
                compiled=True,
                plane_z=plane_z,
            )
            block *= image_pair_mask[image].reshape(self.nleft, self.nright)
            out += phase * block
        return out

    def periodic_sum_many(
        self,
        gvecs,
        left_origins,
        right_origins_batch,
        phases,
        image_pair_mask=None,
        pair_image_starts=None,
        pair_image_indices=None,
        primitive_terms=None,
        product_terms=None,
        compiled=True,
        plane_z=None,
        plane_tol=0.0,
        threads=None,
    ):
        gvecs, plane_z = _plane_gvecs(gvecs, plane_z=plane_z, plane_tol=plane_tol)
        phases = np.ascontiguousarray(phases, dtype=np.complex128)
        if phases.ndim != 2:
            raise ValueError("phases must have shape (nphase, nimage).")

        if (
            product_terms is not None
            and compiled is True
            and has_periodic_pair_ft_product_backend()
        ):
            args = self._product_sum_args(
                gvecs,
                product_terms,
                phases,
                plane_z,
                threads=threads,
            )
            return _gdf_cpp.periodic_pair_ft_product_sum_many(*args)

        if (
            primitive_terms is not None
            and compiled is True
            and has_periodic_pair_ft_many_backend()
        ):
            args = self._primitive_sum_args(
                gvecs,
                primitive_terms,
                phases,
                plane_z,
                threads=threads,
            )
            return _gdf_cpp.periodic_pair_ft_primitive_sum_many(*args)

        return np.stack(
            [
                self.periodic_sum(
                    gvecs,
                    left_origins=left_origins,
                    right_origins_batch=right_origins_batch,
                    phases=phase_row,
                    image_pair_mask=image_pair_mask,
                    pair_image_starts=pair_image_starts,
                    pair_image_indices=pair_image_indices,
                    primitive_terms=primitive_terms,
                    compiled=compiled,
                    plane_z=plane_z,
                    threads=threads,
                )
                for phase_row in phases
            ],
            axis=0,
        )

    def periodic_sum_many_phase_blas(
        self,
        gvecs,
        left_origins,
        right_origins_batch,
        phases,
        image_pair_mask=None,
        pair_image_starts=None,
        pair_image_indices=None,
        primitive_terms=None,
        compiled=True,
        plane_z=None,
        plane_tol=0.0,
        threads=None,
    ):
        gvecs, plane_z = _plane_gvecs(gvecs, plane_z=plane_z, plane_tol=plane_tol)
        phases = np.ascontiguousarray(phases, dtype=np.complex128)
        if phases.ndim != 2:
            raise ValueError("phases must have shape (nphase, nimage).")
        if (
            primitive_terms is None
            or compiled is not True
            or not has_periodic_pair_ft_image_group_backend()
        ):
            return self.periodic_sum_many(
                gvecs,
                left_origins=left_origins,
                right_origins_batch=right_origins_batch,
                phases=phases,
                image_pair_mask=image_pair_mask,
                pair_image_starts=pair_image_starts,
                pair_image_indices=pair_image_indices,
                primitive_terms=primitive_terms,
                compiled=compiled,
                plane_z=plane_z,
                threads=threads,
            )

        group_sums = _gdf_cpp.periodic_pair_ft_primitive_image_group_sum(
            *self._primitive_image_group_sum_args(
                gvecs,
                primitive_terms,
                plane_z,
                threads=threads,
            )
        )
        group_starts = np.asarray(
            primitive_terms["pair_image_group_starts"],
            dtype=np.int64,
        )
        group_images = np.asarray(
            primitive_terms["image_group_image"],
            dtype=np.int64,
        )
        out = np.zeros(
            (phases.shape[0], len(gvecs), self.nleft, self.nright),
            dtype=np.complex128,
        )
        for pair_idx, (pidx, qidx) in enumerate(zip(self.pair_p, self.pair_q)):
            start = int(group_starts[pair_idx])
            stop = int(group_starts[pair_idx + 1])
            if start == stop:
                continue
            images = group_images[start:stop]
            values = phases[:, images] @ group_sums[:, start:stop].T
            out[:, :, int(pidx), int(qidx - self.nleft)] = values
        return np.ascontiguousarray(out)

    def periodic_contract_many(
        self,
        gvecs,
        weighted_aux,
        left_origins,
        right_origins_batch,
        phases,
        image_pair_mask=None,
        pair_image_starts=None,
        pair_image_indices=None,
        primitive_terms=None,
        compiled=True,
        plane_z=None,
        plane_tol=0.0,
        threads=None,
    ):
        gvecs, plane_z = _plane_gvecs(gvecs, plane_z=plane_z, plane_tol=plane_tol)
        weighted_aux = np.ascontiguousarray(weighted_aux, dtype=np.complex128)
        if weighted_aux.ndim != 2 or weighted_aux.shape[0] != len(gvecs):
            raise ValueError("weighted_aux must have shape (ng, naux).")
        phases = np.ascontiguousarray(phases, dtype=np.complex128)
        if phases.ndim != 2:
            raise ValueError("phases must have shape (nphase, nimage).")

        if (
            primitive_terms is not None
            and compiled is True
            and has_periodic_pair_ft_contract_backend()
        ):
            args = self._primitive_sum_args(
                gvecs,
                primitive_terms,
                phases,
                plane_z,
                weighted_aux=weighted_aux,
                threads=threads,
            )
            return _gdf_cpp.periodic_pair_ft_primitive_contract_many(*args)

        pair_batch = self.periodic_sum_many(
            gvecs,
            left_origins=left_origins,
            right_origins_batch=right_origins_batch,
            phases=phases,
            image_pair_mask=image_pair_mask,
            pair_image_starts=pair_image_starts,
            pair_image_indices=pair_image_indices,
            primitive_terms=primitive_terms,
            compiled=compiled,
            plane_z=plane_z,
            threads=threads,
        )
        return np.einsum("ga,xgmn->xamn", weighted_aux, pair_batch, optimize=True)

    def _primitive_sum_args(
        self,
        gvecs,
        primitive_terms,
        phases,
        plane_z,
        weighted_aux=None,
        threads=None,
        include_image_groups=True,
    ):
        args = [
            np.ascontiguousarray(gvecs, dtype=float),
            self.pair_p,
            self.pair_q,
            int(self.nleft),
            int(self.nright),
            np.ascontiguousarray(primitive_terms["pair_term_starts"], dtype=np.int64),
            np.ascontiguousarray(primitive_terms["term_image"], dtype=np.int64),
            np.ascontiguousarray(primitive_terms["term_center"], dtype=float),
            np.ascontiguousarray(primitive_terms["term_inv_4p"], dtype=float),
            np.ascontiguousarray(primitive_terms["term_coeff"], dtype=float),
            np.ascontiguousarray(primitive_terms["term_power"], dtype=np.int64),
        ]
        if include_image_groups:
            args.extend(
                [
                    np.ascontiguousarray(
                        primitive_terms["pair_image_group_starts"],
                        dtype=np.int64,
                    ),
                    np.ascontiguousarray(
                        primitive_terms["image_group_image"],
                        dtype=np.int64,
                    ),
                    np.ascontiguousarray(
                        primitive_terms["image_group_term_start"],
                        dtype=np.int64,
                    ),
                    np.ascontiguousarray(
                        primitive_terms["image_group_term_stop"],
                        dtype=np.int64,
                    ),
                    np.ascontiguousarray(
                        primitive_terms["image_group_product_start"],
                        dtype=np.int64,
                    ),
                    np.ascontiguousarray(
                        primitive_terms["image_group_product_stop"],
                        dtype=np.int64,
                    ),
                    np.ascontiguousarray(
                        primitive_terms["product_group_term_start"],
                        dtype=np.int64,
                    ),
                    np.ascontiguousarray(
                        primitive_terms["product_group_term_stop"],
                        dtype=np.int64,
                    ),
                    np.ascontiguousarray(
                        primitive_terms["product_group_factor_id"],
                        dtype=np.int64,
                    ),
                ]
            )
        args.append(
            np.ascontiguousarray(phases, dtype=np.complex128),
        )
        if weighted_aux is not None:
            args.append(np.ascontiguousarray(weighted_aux, dtype=np.complex128))
        if (
            include_image_groups
            and weighted_aux is None
            and np.asarray(phases).ndim == 2
        ):
            args.append(float(primitive_terms.get("factor_screen_tol", 0.0)))
        args.append(bool(plane_z))
        if threads is not None:
            args.append(int(threads))
        return tuple(args)

    def _primitive_image_group_sum_args(
        self,
        gvecs,
        primitive_terms,
        plane_z,
        threads=None,
    ):
        args = [
            np.ascontiguousarray(gvecs, dtype=float),
            self.pair_p,
            self.pair_q,
            int(self.nleft),
            int(self.nright),
            np.ascontiguousarray(primitive_terms["pair_term_starts"], dtype=np.int64),
            np.ascontiguousarray(primitive_terms["term_image"], dtype=np.int64),
            np.ascontiguousarray(primitive_terms["term_center"], dtype=float),
            np.ascontiguousarray(primitive_terms["term_inv_4p"], dtype=float),
            np.ascontiguousarray(primitive_terms["term_coeff"], dtype=float),
            np.ascontiguousarray(primitive_terms["term_power"], dtype=np.int64),
            np.ascontiguousarray(
                primitive_terms["pair_image_group_starts"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                primitive_terms["image_group_image"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                primitive_terms["image_group_term_start"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                primitive_terms["image_group_term_stop"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                primitive_terms["image_group_product_start"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                primitive_terms["image_group_product_stop"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                primitive_terms["product_group_term_start"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                primitive_terms["product_group_term_stop"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                primitive_terms["product_group_factor_id"],
                dtype=np.int64,
            ),
            bool(plane_z),
        ]
        if threads is not None:
            args.append(int(threads))
        return tuple(args)

    def _product_sum_args(
        self,
        gvecs,
        product_terms,
        phases,
        plane_z,
        threads=None,
    ):
        args = [
            np.ascontiguousarray(gvecs, dtype=float),
            self.pair_p,
            self.pair_q,
            int(self.nleft),
            int(self.nright),
            np.ascontiguousarray(product_terms["factor_center"], dtype=float),
            np.ascontiguousarray(product_terms["factor_inv_4p"], dtype=float),
            np.ascontiguousarray(product_terms["product_image"], dtype=np.int64),
            np.ascontiguousarray(product_terms["product_factor_id"], dtype=np.int64),
            np.ascontiguousarray(product_terms["product_entry_start"], dtype=np.int64),
            np.ascontiguousarray(product_terms["product_entry_stop"], dtype=np.int64),
            np.ascontiguousarray(product_terms["entry_pair"], dtype=np.int64),
            np.ascontiguousarray(product_terms["entry_coeff"], dtype=float),
            np.ascontiguousarray(product_terms["entry_power"], dtype=np.int64),
            np.ascontiguousarray(phases, dtype=np.complex128),
            bool(plane_z),
        ]
        if threads is not None:
            args.append(int(threads))
        return tuple(args)


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
