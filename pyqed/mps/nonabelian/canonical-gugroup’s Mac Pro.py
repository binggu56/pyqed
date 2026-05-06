#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonicalization helpers for fixed-layout non-Abelian MPS tensors.
"""

from __future__ import annotations

import numpy as np

from .contraction import merge_mps_sites
from .coupling import couple_two_sectors_matrix
from .decompose import svd_two_site
from .tensor import NonabelianTensor


def _site_dense_matrix(site, *, mode):
    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        raise ValueError("_site_dense_matrix expects a rank-3 NonabelianTensor site tensor.")
    if mode not in {"left", "right"}:
        raise ValueError("mode must be 'left' or 'right'.")

    grouped = {}
    for (q_left, q_phys, q_right), block in site.data.items():
        arr = np.asarray(block)
        if mode == "left":
            grouped.setdefault(q_right, []).append(arr.reshape(-1, arr.shape[2]))
        else:
            grouped.setdefault(q_left, []).append(arr.reshape(arr.shape[0], -1))
    return grouped


def left_canonical_error(site):
    """
    Return the maximum isometry error for a left-canonical site tensor.
    """
    grouped = _site_dense_matrix(site, mode="left")
    err = 0.0
    for mats in grouped.values():
        M = np.concatenate(mats, axis=0)
        gram = M.conj().T @ M
        err = max(err, float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=gram.dtype))))
    return err


def right_canonical_error(site):
    """
    Return the maximum isometry error for a right-canonical site tensor.
    """
    grouped = _site_dense_matrix(site, mode="right")
    err = 0.0
    for mats in grouped.values():
        M = np.concatenate(mats, axis=1)
        gram = M @ M.conj().T
        err = max(err, float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=gram.dtype))))
    return err


def _irrep_dim(sector):
    irrep = getattr(sector, "irrep", None)
    if irrep is not None and hasattr(irrep, "dim"):
        return int(irrep.dim)
    dim = getattr(sector, "dim", None)
    if dim is not None:
        return int(dim)
    return 1


def _site_full_cg_matrix_blocks(site, *, mode):
    """
    Expand one site into the full CG-coupled Hilbert-space matrix blocks.

    The helper supports both the legacy explicit-physical-m layout and the new
    degeneracy-only physical layout.  In the latter case the physical irrep
    components are supplied entirely by the CG map.
    """
    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        raise ValueError("_site_full_cg_matrix_blocks expects a rank-3 NonabelianTensor.")
    if mode not in {"left", "right"}:
        raise ValueError("mode must be 'left' or 'right'.")

    grouped = {}
    for (q_left, q_phys, q_right), block in site.data.items():
        arr = np.asarray(block)
        left_irrep_dim = _irrep_dim(q_left)
        phys_irrep_dim = _irrep_dim(q_phys)
        right_irrep_dim = _irrep_dim(q_right)
        cg = couple_two_sectors_matrix(q_left, q_phys, q_right)

        if arr.shape[1] == phys_irrep_dim:
            full = np.zeros(
                (
                    arr.shape[0],
                    left_irrep_dim,
                    phys_irrep_dim,
                    arr.shape[2],
                    right_irrep_dim,
                ),
                dtype=arr.dtype,
            )
            for m_left in range(left_irrep_dim):
                for m_phys in range(phys_irrep_dim):
                    row = m_left * phys_irrep_dim + m_phys
                    for m_right in range(right_irrep_dim):
                        full[:, m_left, m_phys, :, m_right] = (
                            arr[:, m_phys, :] * cg[row, m_right]
                        )
        elif arr.shape[1] == 1:
            full = np.zeros(
                (
                    arr.shape[0],
                    left_irrep_dim,
                    phys_irrep_dim,
                    arr.shape[2],
                    right_irrep_dim,
                ),
                dtype=arr.dtype,
            )
            reduced = arr[:, 0, :]
            for m_left in range(left_irrep_dim):
                for m_phys in range(phys_irrep_dim):
                    row = m_left * phys_irrep_dim + m_phys
                    for m_right in range(right_irrep_dim):
                        full[:, m_left, m_phys, :, m_right] = reduced * cg[row, m_right]
        else:
            raise ValueError(
                f"Cannot CG-expand physical block {(q_left, q_phys, q_right)!r}: "
                f"physical axis has dimension {arr.shape[1]}, expected 1 or {phys_irrep_dim}."
            )

        if mode == "left":
            grouped.setdefault(q_right, []).append(
                full.reshape(
                    arr.shape[0] * left_irrep_dim * phys_irrep_dim,
                    arr.shape[2] * right_irrep_dim,
                )
            )
        else:
            grouped.setdefault(q_left, []).append(
                full.reshape(
                    arr.shape[0] * left_irrep_dim,
                    phys_irrep_dim * arr.shape[2] * right_irrep_dim,
                )
            )
    return grouped


def left_cg_canonical_error(site):
    """
    Return the left-isometry error after full CG Hilbert-space expansion.
    """
    grouped = _site_full_cg_matrix_blocks(site, mode="left")
    err = 0.0
    for mats in grouped.values():
        M = np.concatenate(mats, axis=0)
        gram = M.conj().T @ M
        err = max(err, float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=gram.dtype))))
    return err


def right_cg_canonical_error(site):
    """
    Return the right-isometry error after full CG Hilbert-space expansion.
    """
    grouped = _site_full_cg_matrix_blocks(site, mode="right")
    err = 0.0
    for mats in grouped.values():
        M = np.concatenate(mats, axis=1)
        gram = M @ M.conj().T
        err = max(err, float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=gram.dtype))))
    return err


def mixed_cg_canonical_errors(sites, center):
    """
    Return full-CG left/right canonical errors around a mixed-canonical center.
    """
    if not 0 <= int(center) < len(sites):
        raise IndexError(f"Center {center} out of range for chain length {len(sites)}.")
    left_err = 0.0
    right_err = 0.0
    for i, site in enumerate(sites):
        if i < int(center):
            left_err = max(left_err, left_cg_canonical_error(site))
        elif i > int(center):
            right_err = max(right_err, right_cg_canonical_error(site))
    return left_err, right_err


def left_canonicalize_sites(
    sites,
    *,
    cutoff=0.0,
    max_bond=None,
    max_bond_mode="states",
    bond_coupling="left",
):
    """
    Put a chain into left-canonical form by exact two-site gauge moves.
    """
    if len(sites) < 2:
        return [site.copy() for site in sites]
    current = [site.copy() for site in sites]
    for i in range(len(current) - 1):
        merged = merge_mps_sites(current[i], current[i + 1])
        left, right, _s, trunc_err, _kept = svd_two_site(
            merged,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb="right",
            bond_coupling=bond_coupling,
            max_bond_mode=max_bond_mode,
        )
        if trunc_err > 1e-12:
            raise ValueError("left_canonicalize_sites would truncate the state; increase max_bond or lower cutoff.")
        current[i], current[i + 1] = left, right
    return current


def right_canonicalize_sites(
    sites,
    *,
    cutoff=0.0,
    max_bond=None,
    max_bond_mode="states",
    bond_coupling="left",
):
    """
    Put a chain into right-canonical form by exact two-site gauge moves.
    """
    if len(sites) < 2:
        return [site.copy() for site in sites]
    current = [site.copy() for site in sites]
    for i in range(len(current) - 1, 0, -1):
        merged = merge_mps_sites(current[i - 1], current[i])
        left, right, _s, trunc_err, _kept = svd_two_site(
            merged,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb="left",
            bond_coupling=bond_coupling,
            max_bond_mode=max_bond_mode,
        )
        if trunc_err > 1e-12:
            raise ValueError("right_canonicalize_sites would truncate the state; increase max_bond or lower cutoff.")
        current[i - 1], current[i] = left, right
    return current


def mixed_canonicalize_sites(
    sites,
    center,
    *,
    cutoff=0.0,
    max_bond=None,
    max_bond_mode="states",
    bond_coupling="left",
):
    """
    Put a chain into mixed canonical form with orthogonality center at ``center``.
    """
    if not 0 <= int(center) < len(sites):
        raise IndexError(f"Center {center} out of range for chain length {len(sites)}.")
    current = [site.copy() for site in sites]
    for i in range(int(center)):
        merged = merge_mps_sites(current[i], current[i + 1])
        left, right, _s, trunc_err, _kept = svd_two_site(
            merged,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb="right",
            bond_coupling=bond_coupling,
            max_bond_mode=max_bond_mode,
        )
        if trunc_err > 1e-12:
            raise ValueError("mixed_canonicalize_sites would truncate the state on the left pass.")
        current[i], current[i + 1] = left, right
    for i in range(len(current) - 1, int(center), -1):
        merged = merge_mps_sites(current[i - 1], current[i])
        left, right, _s, trunc_err, _kept = svd_two_site(
            merged,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb="left",
            bond_coupling=bond_coupling,
            max_bond_mode=max_bond_mode,
        )
        if trunc_err > 1e-12:
            raise ValueError("mixed_canonicalize_sites would truncate the state on the right pass.")
        current[i - 1], current[i] = left, right
    return current


def mixed_canonical_errors(sites, center):
    """
    Return the maximum left/right canonical errors around a mixed-canonical center.

    Sites strictly left of ``center`` must be left-canonical and sites strictly
    right of ``center`` must be right-canonical. The center tensor itself is not
    constrained.
    """
    if not 0 <= int(center) < len(sites):
        raise IndexError(f"Center {center} out of range for chain length {len(sites)}.")
    left_err = 0.0
    right_err = 0.0
    for i, site in enumerate(sites):
        if i < int(center):
            if getattr(site, "metadata", {}).get("physical_basis") == "reduced_spatial":
                left_err = max(left_err, left_cg_canonical_error(site))
            else:
                left_err = max(left_err, left_canonical_error(site))
        elif i > int(center):
            if getattr(site, "metadata", {}).get("physical_basis") == "reduced_spatial":
                right_err = max(right_err, right_cg_canonical_error(site))
            else:
                right_err = max(right_err, right_canonical_error(site))
    return left_err, right_err


def assert_mixed_canonical_sites(sites, center, *, tol=1e-10):
    """
    Raise if a chain is not mixed-canonical about ``center`` within ``tol``.
    """
    left_err, right_err = mixed_canonical_errors(sites, center)
    if left_err > tol or right_err > tol:
        raise ValueError(
            "Chain is not in mixed canonical gauge: "
            f"center={center}, left_error={left_err:.3e}, right_error={right_err:.3e}, tol={tol:.3e}."
        )
