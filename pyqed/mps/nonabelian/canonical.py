#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonicalization helpers for fixed-layout non-Abelian MPS tensors.
"""

from __future__ import annotations

import numpy as np

from .contraction import merge_mps_sites
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
            left_err = max(left_err, left_canonical_error(site))
        elif i > int(center):
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
