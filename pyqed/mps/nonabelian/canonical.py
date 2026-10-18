#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonicalization helpers for fixed-layout non-Abelian MPS tensors.
"""

from __future__ import annotations

import numpy as np

from .contraction import merge_mps_sites
from .decompose import svd_two_site
from pyqed.symmetry import IrrepTensor


def left_qr_sites(sites):
    """Exactly left-orthogonalize fully reduced sites without bond growth.

    Sectorwise single-site QR and absorption, with no singular-value
    truncation. Operates on copies, including their sector multiplicities.
    """
    current = [site.copy() for site in sites]
    for i in range(len(current)-1):
        left, right = current[i:i+2]
        if left.metadata.get('physical_basis') != 'fully_reduced_su2':
            raise ValueError('left_qr_sites requires fully reduced SU2 tensors')
        ld, rd = dict(left.data), dict(right.data)
        qns = []
        for sector in dict.fromkeys(left.qns[2]):
            keys = [key for key in ld if key[2] == sector]
            if not keys:
                continue
            width = ld[keys[0]].shape[2]
            matrix = np.concatenate([ld[key].reshape(-1,width) for key in keys])
            q, r = np.linalg.qr(matrix,mode='reduced')
            rank = q.shape[1]
            qns.extend([sector]*rank)
            offset = 0
            for key in keys:
                shape = ld[key].shape
                rows = shape[0]*shape[1]
                ld[key] = q[offset:offset+rows].reshape(shape[0],shape[1],rank)
                offset += rows
            for key in rd:
                if key[0] == sector:
                    rd[key] = np.einsum('ab,bpr->apr',r,rd[key],optimize=True)
        current[i] = IrrepTensor(data=ld,qns=[left.qns[0],left.qns[1],qns],
                                 dirs=left.dirs,metadata=dict(left.metadata))
        current[i+1] = IrrepTensor(data=rd,qns=[qns,right.qns[1],right.qns[2]],
                                   dirs=right.dirs,metadata=dict(right.metadata))
    return current


def right_qr_sites(sites):
    """Exactly right-orthogonalize reduced SU(2) sites without bond growth.

    The QR uses the spin-dimension weights of the right canonical metric.
    Absorbing its triangular factor into the preceding site preserves the
    complete state, including complex phases and Abelian spatial sectors.
    """
    current = [site.copy() for site in sites]
    for i in range(len(current)-1, 0, -1):
        left, right = current[i-1:i+1]
        if right.metadata.get('physical_basis') != 'fully_reduced_su2':
            raise ValueError('right_qr_sites requires fully reduced SU2 tensors')
        ld, rd = dict(left.data), dict(right.data)
        qns = []
        for sector in dict.fromkeys(right.qns[0]):
            keys = [key for key in rd if key[0] == sector]
            if not keys:
                continue
            weights = [np.sqrt(_irrep_dim(key[2]) / _irrep_dim(sector)) for key in keys]
            matrix = np.concatenate([weight*rd[key].reshape(rd[key].shape[0], -1)
                                     for key, weight in zip(keys, weights)], axis=1)
            q, r = np.linalg.qr(matrix.conj().T, mode='reduced')
            rank = q.shape[1]
            qns.extend([sector]*rank)
            offset = 0
            for key, weight in zip(keys, weights):
                shape = rd[key].shape
                cols = shape[1]*shape[2]
                rd[key] = (q[offset:offset+cols].conj().T/weight).reshape(rank,shape[1],shape[2])
                offset += cols
            for key in ld:
                if key[2] == sector:
                    ld[key] = np.einsum('lpa,ab->lpb',ld[key],r.conj().T,optimize=True)
        current[i-1] = IrrepTensor(data=ld,qns=[left.qns[0],left.qns[1],qns],
                                    dirs=left.dirs,metadata=dict(left.metadata))
        current[i] = IrrepTensor(data=rd,qns=[qns,right.qns[1],right.qns[2]],
                                  dirs=right.dirs,metadata=dict(right.metadata))
    return current


def _exact_reduced_gauge(sites, cutoff, max_bond):
    return (max_bond is None and cutoff == 0
            and all(site.metadata.get('physical_basis') == 'fully_reduced_su2'
                    and all(block.shape[1] == 1 for block in site.data.values()) for site in sites))


def _site_dense_matrix(site, *, mode):
    if not isinstance(site, IrrepTensor) or site.rank != 3:
        raise ValueError("_site_dense_matrix expects a rank-3 IrrepTensor site tensor.")
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


def _irrep_dim(sector):
    irrep = getattr(sector, "irrep", None)
    if irrep is not None and hasattr(irrep, "dim"):
        return int(irrep.dim)
    labels = getattr(sector, "labels", ())
    components = getattr(sector, "components", ())
    if "su2" in labels:
        irrep = components[labels.index("su2")]
        if hasattr(irrep, "dim"):
            return int(irrep.dim)
    dim = getattr(sector, "dim", None)
    if dim is not None:
        return int(dim)
    return 1


def _site_full_dense_matrix(site, *, mode):
    if not isinstance(site, IrrepTensor) or site.rank != 3:
        raise ValueError("_site_full_dense_matrix expects a rank-3 IrrepTensor site tensor.")
    if mode not in {"left", "right"}:
        raise ValueError("mode must be 'left' or 'right'.")

    left_sectors = tuple(dict.fromkeys(site.qns[0]))
    phys_sectors = tuple(dict.fromkeys(site.qns[1]))
    right_sectors = tuple(dict.fromkeys(site.qns[2]))
    left_dims = {
        sector: max(
            [np.asarray(block).shape[0] for key, block in site.data.items() if key[0] == sector]
            or [sum(1 for item in site.qns[0] if item == sector)]
        )
        for sector in left_sectors
    }
    phys_dims = {
        sector: max(
            [np.asarray(block).shape[1] for key, block in site.data.items() if key[1] == sector]
            or [getattr(sector, "dim", sum(1 for item in site.qns[1] if item == sector))]
        )
        for sector in phys_sectors
    }
    right_dims = {
        sector: max(
            [np.asarray(block).shape[2] for key, block in site.data.items() if key[2] == sector]
            or [sum(1 for item in site.qns[2] if item == sector)]
        )
        for sector in right_sectors
    }

    if mode == "left":
        row_offsets = {}
        offset = 0
        for q_left in left_sectors:
            for q_phys in phys_sectors:
                row_offsets[(q_left, q_phys)] = offset
                offset += left_dims[q_left] * phys_dims[q_phys]
        col_offsets = {}
        col_offset = 0
        for q_right in right_sectors:
            col_offsets[q_right] = col_offset
            col_offset += right_dims[q_right]
        matrix = np.zeros((offset, col_offset), dtype=np.result_type(*[np.asarray(block).dtype for block in site.data.values()], float))
        for (q_left, q_phys, q_right), block in site.data.items():
            arr = np.asarray(block)
            row0 = row_offsets[(q_left, q_phys)]
            col0 = col_offsets[q_right]
            rows = arr.shape[0] * arr.shape[1]
            matrix[row0:row0 + rows, col0:col0 + arr.shape[2]] = arr.reshape(rows, arr.shape[2])
        return matrix

    row_offsets = {}
    offset = 0
    for q_left in left_sectors:
        row_offsets[q_left] = offset
        offset += left_dims[q_left]
    col_offsets = {}
    col_offset = 0
    for q_phys in phys_sectors:
        for q_right in right_sectors:
            col_offsets[(q_phys, q_right)] = col_offset
            col_offset += phys_dims[q_phys] * right_dims[q_right]
    matrix = np.zeros((offset, col_offset), dtype=np.result_type(*[np.asarray(block).dtype for block in site.data.values()], float))
    for (q_left, q_phys, q_right), block in site.data.items():
        arr = np.asarray(block)
        row0 = row_offsets[q_left]
        col0 = col_offsets[(q_phys, q_right)]
        cols = arr.shape[1] * arr.shape[2]
        matrix[row0:row0 + arr.shape[0], col0:col0 + cols] = arr.reshape(arr.shape[0], cols)
    return matrix


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


def left_identity_metric_error(site):
    """
    Return the full explicit-basis left-isometry error.

    Unlike :func:`left_canonical_error`, this includes cross-sector overlaps.
    It is the diagnostic relevant to whether an identity-MPO environment will
    expose an identity local norm in the current explicit/reduced mixed basis.
    """
    matrix = _site_full_dense_matrix(site, mode="left")
    gram = matrix.conj().T @ matrix
    return float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=gram.dtype)))


def right_canonical_error(site):
    """
    Return the maximum isometry error for a right-canonical site tensor.
    """
    if not isinstance(site, IrrepTensor) or site.rank != 3:
        raise ValueError("right_canonical_error expects a rank-3 IrrepTensor site tensor.")
    grouped = {}
    for (q_left, _q_phys, q_right), block in site.data.items():
        arr = np.asarray(block)
        matrix = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] == 1:
            weight = _irrep_dim(q_right) / max(_irrep_dim(q_left), 1)
        else:
            weight = 1.0
        grouped.setdefault(q_left, []).append((matrix, float(weight)))
    err = 0.0
    for weighted_mats in grouped.values():
        gram = None
        for matrix, weight in weighted_mats:
            block_gram = weight * (matrix @ matrix.conj().T)
            gram = block_gram if gram is None else gram + block_gram
        if gram is None:
            continue
        err = max(err, float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=gram.dtype))))
    return err


def right_identity_metric_error(site):
    """
    Return the full explicit-basis right-isometry error including cross sectors.
    """
    matrix = _site_full_dense_matrix(site, mode="right")
    gram = matrix @ matrix.conj().T
    return float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=gram.dtype)))


def mixed_identity_metric_errors(sites, center):
    if not 0 <= int(center) < len(sites):
        raise IndexError(f"Center {center} out of range for chain length {len(sites)}.")
    left_err = 0.0
    right_err = 0.0
    for i, site in enumerate(sites):
        if i < int(center):
            left_err = max(left_err, left_identity_metric_error(site))
        elif i > int(center):
            right_err = max(right_err, right_identity_metric_error(site))
    return left_err, right_err


def left_canonicalize_sites(
    sites,
    *,
    cutoff=0.0,
    max_bond=None,
    max_bond_mode="states",
    bond_coupling="left",
    retain_sector_topology=False,
):
    """
    Put a chain into left-canonical form by exact gauge moves.

    Untruncated fully reduced SU(2) chains use sectorwise single-site QR,
    preserving existing virtual spaces; other settings use two-site SVD.
    """
    if _exact_reduced_gauge(sites, cutoff, max_bond):
        return left_qr_sites(sites)
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
            retain_sector_topology=retain_sector_topology,
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
    retain_sector_topology=False,
):
    """
    Put a chain into right-canonical form by exact gauge moves.

    Untruncated fully reduced SU(2) chains use weighted single-site QR,
    preserving existing virtual spaces; other settings use two-site SVD.
    """
    if _exact_reduced_gauge(sites, cutoff, max_bond):
        return right_qr_sites(sites)
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
            retain_sector_topology=retain_sector_topology,
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
    retain_sector_topology=False,
):
    """
    Put a chain into mixed canonical form with orthogonality center at ``center``.

    Untruncated fully reduced SU(2) chains use single-site QR, so exact gauge
    changes cannot inflate bond multiplicities by retaining null SVD vectors.
    """
    if not 0 <= int(center) < len(sites):
        raise IndexError(f"Center {center} out of range for chain length {len(sites)}.")
    if _exact_reduced_gauge(sites, cutoff, max_bond):
        # Two-site SVD at zero cutoff retains null singular directions and can
        # inflate a restart bond far beyond its saved multiplicity. Single-site
        # QR moves the gauge within the existing virtual spaces instead.
        current = left_qr_sites(sites[:int(center)+1]) + [site.copy() for site in sites[int(center)+1:]]
        current[int(center):] = right_qr_sites(current[int(center):])
        return current
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
            retain_sector_topology=retain_sector_topology,
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
            retain_sector_topology=retain_sector_topology,
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
