#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np

from pyqed.qchem.basis import E, R
from pyqed.qchem.fourier import (
    ao_pair_ft_matrix_s,
    ao_pair_ft_matrices_compiled,
    ao_pair_ft_matrices,
    gaussian_pair_ft,
    gaussian_pair_ft_batch,
    gaussian_pair_ft_s,
)


def _require_s_gaussians(basis):
    for fn in basis:
        if tuple(int(x) for x in fn.shell) != (0, 0, 0):
            raise NotImplementedError(
                "Reciprocal-space AO pair Fourier factors currently support only s Gaussians."
            )


def _validate_gvec(gvec):
    gvec = np.asarray(gvec, dtype=float)
    if gvec.shape != (3,):
        raise ValueError("gvec must have shape (3,).")
    return gvec


def _validate_center(center):
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,).")
    return center


def _boys0(t):
    t = float(t)
    if t < 1e-12:
        return 1.0
    root = math.sqrt(t)
    return 0.5 * math.sqrt(math.pi / t) * math.erf(root)


def reciprocal_lattice(lattice_vectors):
    """Return reciprocal lattice vectors as rows, with a_i dot b_j = 2*pi delta_ij."""
    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3).")
    return 2.0 * math.pi * np.linalg.inv(lattice).T


def reciprocal_vectors(lattice_vectors, recip_cut, include_zero=False):
    """Enumerate reciprocal vectors up to integer index cutoff."""
    recip = reciprocal_lattice(lattice_vectors)
    recip_cut = int(recip_cut)
    if recip_cut < 0:
        raise ValueError("recip_cut must be non-negative.")

    out = []
    for h in range(-recip_cut, recip_cut + 1):
        for k in range(-recip_cut, recip_cut + 1):
            for l in range(-recip_cut, recip_cut + 1):
                if not include_zero and h == 0 and k == 0 and l == 0:
                    continue
                gvec = h * recip[0] + k * recip[1] + l * recip[2]
                out.append((h, k, l, gvec))
    return out


def gauss_chebyshev_grid(n):
    """Gauss-Chebyshev radial grid used for low-dimensional inf_vacuum G grids."""
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    ln2 = 1.0 / math.log(2.0)
    fac = 16.0 / 3.0 / (n + 1)
    x1 = np.arange(1, n + 1, dtype=float) * math.pi / (n + 1)
    idx = np.arange(n, dtype=float)
    xi = (
        (n - 1 - idx * 2.0) / (n + 1.0)
        + (1.0 + 2.0 / 3.0 * np.sin(x1) ** 2)
        * np.sin(2.0 * x1)
        / math.pi
    )
    xi = (xi - xi[::-1]) / 2.0
    r = 1.0 - np.log(1.0 + xi) * ln2
    dr = fac * np.sin(x1) ** 4 * ln2 / (1.0 + xi)
    return r, dr


def _non_uniform_gv_base(n):
    rs, ws = gauss_chebyshev_grid(n)
    return np.hstack((rs, -rs[::-1])), np.hstack((ws, ws[::-1]))


def inf_vacuum_1d_gv_weights(lattice_vectors, mesh):
    """
    Native clone of PySCF's dimension=1, low_dim_ft_type='inf_vacuum' G grid.

    The periodic direction is sampled on FFT integer G indices. The two
    transverse directions use nonuniform quadrature for infinite vacuum.
    """
    lattice = np.asarray(lattice_vectors, dtype=float)
    mesh = np.asarray(mesh, dtype=int)
    if lattice.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3).")
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError("mesh must be a length-3 positive integer array.")
    if mesh[1] < 2 or mesh[2] < 2:
        raise ValueError("transverse mesh entries must be at least 2.")

    b = reciprocal_lattice(lattice)
    rx = np.fft.fftfreq(int(mesh[0]), 1.0 / int(mesh[0]))
    wx = np.repeat(np.linalg.norm(b[0]), len(rx))
    ry, wy = _non_uniform_gv_base(int(mesh[1]) // 2)
    rz, wz = _non_uniform_gv_base(int(mesh[2]) // 2)
    ry = ry / np.linalg.norm(b[1])
    rz = rz / np.linalg.norm(b[2])
    weights = np.einsum("i,j,k->ijk", wx, wy, wz).reshape(-1)
    weights *= 1.0 / (2.0 * math.pi) ** 3

    gx, gy, gz = np.meshgrid(rx, ry, rz, indexing="ij")
    coeffs = np.stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)), axis=1)
    gvecs = coeffs @ b
    return gvecs, weights


def short_range_point_charge(a, b, center, eta):
    """
    Integral of a Gaussian AO pair against erfc(eta*r)/r.

    Returns int chi_a(r) chi_b(r) erfc(eta*|r-center|)/|r-center| dr.
    ``eta=0`` recovers the full Coulomb point-charge attraction integral.
    """
    eta = float(eta)
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")

    center = _validate_center(center)
    a_origin = np.asarray(a.origin, dtype=float)
    b_origin = np.asarray(b.origin, dtype=float)
    l1, m1, n1 = tuple(int(x) for x in a.shell)
    l2, m2, n2 = tuple(int(x) for x in b.shell)

    value = 0.0
    for ia, wa in enumerate(a.prim_weights):
        alpha = float(a.exps[ia])
        for ib, wb in enumerate(b.prim_weights):
            beta = float(b.exps[ib])
            p = alpha + beta
            product_center = (alpha * a_origin + beta * b_origin) / p
            diff = product_center - center
            rpc = float(np.linalg.norm(diff))
            ab = a_origin - b_origin
            primitive_full = 0.0
            primitive_lr = 0.0
            p_lr = None if eta == 0.0 else p * eta * eta / (p + eta * eta)
            lr_scale = 0.0 if eta == 0.0 else eta / math.sqrt(p + eta * eta)

            for t in range(l1 + l2 + 1):
                ex = E(l1, l2, t, float(ab[0]), alpha, beta)
                for u in range(m1 + m2 + 1):
                    exy = ex * E(m1, m2, u, float(ab[1]), alpha, beta)
                    for v in range(n1 + n2 + 1):
                        coeff = exy * E(n1, n2, v, float(ab[2]), alpha, beta)
                        primitive_full += coeff * R(
                            t, u, v, 0, p, float(diff[0]), float(diff[1]), float(diff[2]), rpc
                        )
                        if p_lr is not None:
                            primitive_lr += coeff * R(
                                t, u, v, 0, p_lr, float(diff[0]), float(diff[1]), float(diff[2]), rpc
                            )

            value += wa * wb * (2.0 * math.pi / p) * (primitive_full - lr_scale * primitive_lr)
    return float(value)


def short_range_point_charge_s(a, b, center, eta):
    """Backward-compatible alias for range-separated point-charge attraction."""
    return short_range_point_charge(a, b, center, eta)


def short_range_nuclear_attraction_matrix_s(
    charges,
    coords,
    basis,
    lattice_vectors,
    eta,
    real_cut=4,
):
    """
    Real-space short-range electron-nuclear attraction matrix for Cartesian AOs.

    This is the erfc(eta*r)/r complement to the damped reciprocal nuclear
    attraction matrix.
    """
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    lattice = np.asarray(lattice_vectors, dtype=float)
    if coords.shape != (len(charges), 3):
        raise ValueError("coords must have shape (ncharge, 3).")
    if lattice.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3).")

    eta = float(eta)
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    real_cut = int(real_cut)
    if real_cut < 0:
        raise ValueError("real_cut must be non-negative.")

    basis = tuple(basis)

    mat = np.zeros((len(basis), len(basis)), dtype=float)
    image_range = range(-real_cut, real_cut + 1)
    for p, bp in enumerate(basis):
        for q, bq in enumerate(basis):
            value = 0.0
            for charge, coord in zip(charges, coords):
                for nx in image_range:
                    for ny in image_range:
                        for nz in image_range:
                            shift = nx * lattice[0] + ny * lattice[1] + nz * lattice[2]
                            value -= charge * short_range_point_charge_s(
                                bp,
                                bq,
                                coord + shift,
                                eta,
                            )
            mat[p, q] = value
    return 0.5 * (mat + mat.T)


def short_range_eri(a, b, c, d, eta):
    """
    Two-electron Cartesian Gaussian integral with erfc(eta*r12)/r12.

    Returns (ab|cd)_SR. ``eta=0`` recovers the full Coulomb ERI.
    """
    eta = float(eta)
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")

    a_origin = np.asarray(a.origin, dtype=float)
    b_origin = np.asarray(b.origin, dtype=float)
    c_origin = np.asarray(c.origin, dtype=float)
    d_origin = np.asarray(d.origin, dtype=float)
    l1, m1, n1 = tuple(int(x) for x in a.shell)
    l2, m2, n2 = tuple(int(x) for x in b.shell)
    l3, m3, n3 = tuple(int(x) for x in c.shell)
    l4, m4, n4 = tuple(int(x) for x in d.shell)

    value = 0.0
    for ia, wa in enumerate(a.prim_weights):
        alpha = float(a.exps[ia])
        for ib, wb in enumerate(b.prim_weights):
            beta = float(b.exps[ib])
            p = alpha + beta
            p_center = (alpha * a_origin + beta * b_origin) / p
            ab = a_origin - b_origin
            for ic, wc in enumerate(c.prim_weights):
                gamma = float(c.exps[ic])
                for id_, wd in enumerate(d.prim_weights):
                    delta = float(d.exps[id_])
                    q = gamma + delta
                    q_center = (gamma * c_origin + delta * d_origin) / q
                    cd = c_origin - d_origin
                    diff = p_center - q_center
                    rpq = float(np.linalg.norm(diff))
                    theta = p * q / (p + q)
                    prefactor = 2.0 * math.pi ** 2.5 / (p * q * math.sqrt(p + q))
                    theta_lr = None if eta == 0.0 else theta * eta * eta / (theta + eta * eta)
                    lr_scale = 0.0 if eta == 0.0 else eta / math.sqrt(theta + eta * eta)

                    primitive_full = 0.0
                    primitive_lr = 0.0
                    for t in range(l1 + l2 + 1):
                        ex_ab = E(l1, l2, t, float(ab[0]), alpha, beta)
                        for u in range(m1 + m2 + 1):
                            exy_ab = ex_ab * E(m1, m2, u, float(ab[1]), alpha, beta)
                            for v in range(n1 + n2 + 1):
                                xyz_ab = exy_ab * E(n1, n2, v, float(ab[2]), alpha, beta)
                                for tau in range(l3 + l4 + 1):
                                    ex_cd = E(l3, l4, tau, float(cd[0]), gamma, delta)
                                    for nu in range(m3 + m4 + 1):
                                        exy_cd = ex_cd * E(m3, m4, nu, float(cd[1]), gamma, delta)
                                        for phi in range(n3 + n4 + 1):
                                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                                            coeff = (
                                                xyz_ab
                                                * exy_cd
                                                * E(n3, n4, phi, float(cd[2]), gamma, delta)
                                                * sign
                                            )
                                            primitive_full += coeff * R(
                                                t + tau,
                                                u + nu,
                                                v + phi,
                                                0,
                                                theta,
                                                float(diff[0]),
                                                float(diff[1]),
                                                float(diff[2]),
                                                rpq,
                                            )
                                            if theta_lr is not None:
                                                primitive_lr += coeff * R(
                                                    t + tau,
                                                    u + nu,
                                                    v + phi,
                                                    0,
                                                    theta_lr,
                                                    float(diff[0]),
                                                    float(diff[1]),
                                                    float(diff[2]),
                                                    rpq,
                                                )

                    value += wa * wb * wc * wd * prefactor * (
                        primitive_full - lr_scale * primitive_lr
                    )
    return float(value)


def short_range_eri_s(a, b, c, d, eta):
    """Backward-compatible alias for range-separated ERIs."""
    return short_range_eri(a, b, c, d, eta)


def short_range_eri_tensor_s(basis, eta):
    """Dense chemist-notation short-range ERI tensor for a Cartesian AO basis."""
    basis = tuple(basis)
    nao = len(basis)
    eri = np.empty((nao, nao, nao, nao), dtype=float)
    for p, bp in enumerate(basis):
        for q, bq in enumerate(basis):
            for r, br in enumerate(basis):
                for s, bs in enumerate(basis):
                    eri[p, q, r, s] = short_range_eri_s(bp, bq, br, bs, eta)
    return eri


def reciprocal_nuclear_attraction_matrix_s(
    charges,
    coords,
    basis,
    lattice_vectors,
    recip_cut=8,
    eta=None,
):
    """
    Reciprocal-space electron-nuclear attraction matrix for a Cartesian AO basis.

    The G=0 Coulomb component is omitted, corresponding to the usual neutral
    periodic Poisson convention. If ``eta`` is provided, the reciprocal kernel is
    Ewald damped and represents only the smooth long-range component.
    """
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    lattice = np.asarray(lattice_vectors, dtype=float)
    if coords.shape != (len(charges), 3):
        raise ValueError("coords must have shape (ncharge, 3).")

    volume = abs(float(np.linalg.det(lattice)))
    if volume <= 0.0:
        raise ValueError("lattice volume must be positive.")

    basis = tuple(basis)

    mat = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for _h, _k, _l, gvec in reciprocal_vectors(lattice, recip_cut, include_zero=False):
        g2 = float(np.dot(gvec, gvec))
        if g2 <= 0.0:
            continue
        damping = 1.0
        if eta is not None:
            eta_f = float(eta)
            if eta_f <= 0.0:
                raise ValueError("eta must be positive.")
            damping = math.exp(-g2 / (4.0 * eta_f * eta_f))
        rho_nuc = np.sum(charges * np.exp(-1j * coords @ gvec))
        pair_plus_g = ao_pair_ft_matrix_s(basis, -gvec)
        mat += -(4.0 * math.pi / volume) * damping * rho_nuc * pair_plus_g / g2
    return 0.5 * (mat + mat.conj().T)


def reciprocal_hartree_matrix_s(dm, basis, lattice_vectors, recip_cut=8, eta=None):
    """
    Reciprocal-space Hartree/J matrix for a Cartesian AO basis and density matrix.

    The G=0 Coulomb component is omitted, matching
    reciprocal_nuclear_attraction_matrix_s. If ``eta`` is provided, the
    reciprocal kernel is Ewald damped and represents only the smooth long-range
    component.
    """
    lattice = np.asarray(lattice_vectors, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    if volume <= 0.0:
        raise ValueError("lattice volume must be positive.")

    basis = tuple(basis)

    dm = np.asarray(dm, dtype=np.complex128)
    if dm.shape != (len(basis), len(basis)):
        raise ValueError("dm must have shape (nao, nao).")

    mat = np.zeros_like(dm, dtype=np.complex128)
    for _h, _k, _l, gvec in reciprocal_vectors(lattice, recip_cut, include_zero=False):
        g2 = float(np.dot(gvec, gvec))
        if g2 <= 0.0:
            continue
        damping = 1.0
        if eta is not None:
            eta_f = float(eta)
            if eta_f <= 0.0:
                raise ValueError("eta must be positive.")
            damping = math.exp(-g2 / (4.0 * eta_f * eta_f))
        pair_minus_g = ao_pair_ft_matrix_s(basis, gvec)
        rho_e = np.einsum("pq,pq->", dm, pair_minus_g, optimize=True)
        pair_plus_g = ao_pair_ft_matrix_s(basis, -gvec)
        mat += (4.0 * math.pi / volume) * damping * rho_e * pair_plus_g / g2
    return 0.5 * (mat + mat.conj().T)


def reciprocal_eri_tensor_s(basis, lattice_vectors, recip_cut=8, eta=None):
    """
    Reciprocal-space long-range ERI tensor for a Cartesian AO basis.

    Returns chemist notation (pq|rs). The G=0 Coulomb component is omitted.
    If ``eta`` is provided, the kernel is Ewald damped and represents the
    smooth long-range erf(eta*r12)/r12 component.
    """
    lattice = np.asarray(lattice_vectors, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    if volume <= 0.0:
        raise ValueError("lattice volume must be positive.")

    basis = tuple(basis)

    nao = len(basis)
    eri = np.zeros((nao, nao, nao, nao), dtype=np.complex128)
    for _h, _k, _l, gvec in reciprocal_vectors(lattice, recip_cut, include_zero=False):
        g2 = float(np.dot(gvec, gvec))
        if g2 <= 0.0:
            continue
        damping = 1.0
        if eta is not None:
            eta_f = float(eta)
            if eta_f <= 0.0:
                raise ValueError("eta must be positive.")
            damping = math.exp(-g2 / (4.0 * eta_f * eta_f))
        pair_g = ao_pair_ft_matrix_s(basis, gvec)
        pair_minus_g = ao_pair_ft_matrix_s(basis, -gvec)
        eri += (
            (4.0 * math.pi / volume)
            * damping
            / g2
            * np.einsum("pq,rs->pqrs", pair_g, pair_minus_g, optimize=True)
        )
    eri = 0.5 * (eri + eri.transpose(1, 0, 3, 2).conj())
    return np.asarray(eri.real, dtype=float)


def ewald_nuclear_repulsion(
    charges,
    coords,
    lattice_vectors,
    eta=None,
    real_cut=4,
    recip_cut=8,
    neutralizing_background=True,
):
    """
    Point-charge Ewald lattice energy for a 3D supercell.

    The current native PBC cell is 1D, but it is embedded in a 3D supercell with
    transverse vacuum. This routine evaluates the corresponding 3D-supercell
    Ewald sum without depending on PySCF.
    """
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    lattice = np.asarray(lattice_vectors, dtype=float)
    if coords.shape != (len(charges), 3):
        raise ValueError("coords must have shape (ncharge, 3).")
    if lattice.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3).")

    volume = abs(float(np.linalg.det(lattice)))
    if volume <= 0.0:
        raise ValueError("lattice volume must be positive.")

    lengths = np.linalg.norm(lattice, axis=1)
    if eta is None:
        eta = math.sqrt(math.pi) / float(np.min(lengths))
    eta = float(eta)
    real_cut = int(real_cut)
    recip_cut = int(recip_cut)
    if eta <= 0.0:
        raise ValueError("eta must be positive.")
    if real_cut < 0 or recip_cut < 0:
        raise ValueError("real_cut and recip_cut must be non-negative.")

    e_real = 0.0
    image_range = range(-real_cut, real_cut + 1)
    for ia, (za, ra) in enumerate(zip(charges, coords)):
        for ib, (zb, rb) in enumerate(zip(charges, coords)):
            zz = za * zb
            for nx in image_range:
                for ny in image_range:
                    for nz in image_range:
                        if ia == ib and nx == 0 and ny == 0 and nz == 0:
                            continue
                        shift = nx * lattice[0] + ny * lattice[1] + nz * lattice[2]
                        dist = np.linalg.norm(ra - rb + shift)
                        if dist > 1e-14:
                            e_real += zz * math.erfc(eta * dist) / dist
    e_real *= 0.5

    e_recip = 0.0
    for _h, _k, _l, gvec in reciprocal_vectors(lattice, recip_cut, include_zero=False):
        g2 = float(np.dot(gvec, gvec))
        if g2 <= 0.0:
            continue
        rho = np.sum(charges * np.exp(1j * coords @ gvec))
        e_recip += math.exp(-g2 / (4.0 * eta * eta)) * (abs(rho) ** 2) / g2
    e_recip *= 2.0 * math.pi / volume

    e_self = -eta / math.sqrt(math.pi) * float(np.dot(charges, charges))
    e_background = 0.0
    total_charge = float(np.sum(charges))
    if neutralizing_background and abs(total_charge) > 1e-14:
        e_background = -math.pi * total_charge * total_charge / (2.0 * eta * eta * volume)

    return float(e_real + e_recip + e_self + e_background)


def ewald_nuclear_repulsion_1d_inf_vacuum(
    charges,
    coords,
    lattice_vectors,
    eta,
    real_cut=4,
    mesh=(31, 38, 38),
):
    """
    1D periodic Ewald nuclear energy with infinite transverse vacuum convention.
    """
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    lattice = np.asarray(lattice_vectors, dtype=float)
    if coords.shape != (len(charges), 3):
        raise ValueError("coords must have shape (ncharge, 3).")
    if lattice.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3).")
    eta = float(eta)
    if eta <= 0.0:
        raise ValueError("eta must be positive.")

    a1 = lattice[0]
    e_real = 0.0
    for ia, (za, ra) in enumerate(zip(charges, coords)):
        for ib, (zb, rb) in enumerate(zip(charges, coords)):
            for n in range(-int(real_cut), int(real_cut) + 1):
                if ia == ib and n == 0:
                    continue
                diff = ra - rb + float(n) * a1
                dist = np.linalg.norm(diff)
                if dist > 1e-14:
                    e_real += za * zb * math.erfc(eta * dist) / dist
    e_real *= 0.5

    gvecs, weights = inf_vacuum_1d_gv_weights(lattice, mesh)
    g2 = np.einsum("gi,gi->g", gvecs, gvecs)
    mask = g2 > 1e-16
    rho = np.einsum("z,zg->g", charges, np.exp(-1j * coords @ gvecs[mask].T))
    coul = 4.0 * math.pi * weights[mask] / g2[mask]
    e_recip = 0.5 * np.einsum(
        "g,g,g->",
        rho.conj(),
        rho * np.exp(-g2[mask] / (4.0 * eta * eta)),
        coul,
    ).real
    e_self = -eta / math.sqrt(math.pi) * float(np.dot(charges, charges))
    return float(e_real + e_recip + e_self)
