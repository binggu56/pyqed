"""Finite MPS/MPO data structures and the legacy finite-system DMRG engine.

Dense MPS tensors use ``(left bond, physical, right bond)`` ordering.  The
large Abelian moving-environment implementation remains in this module because
the legacy DMRG kernels share its internal data structures directly; small
independent compatibility helpers live in dedicated sibling modules.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy

import numpy as np
import scipy.sparse as sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from tensorly.decomposition import tensor_train_matrix

from pyqed.lattice import Site
from pyqed.mps.decompose import compress, decompose
from pyqed.mps.dense_canonical import LeftCanonical, RightCanonical
from pyqed.mps.umps import UniformMPS
from pyqed.tn import MPO

logger = logging.getLogger(__name__)

try:
    from numba import njit as _numba_njit
    from numba import prange as _numba_prange
    from numba import get_num_threads as _numba_get_num_threads
except Exception:  # pragma: no cover - numba is an optional accelerator.
    _numba_njit = None
    _numba_prange = range
    _numba_get_num_threads = None
try:
    from pyqed.mps import packed_cython as _packed_cython
except Exception:  # pragma: no cover - optional compiled backend.
    _packed_cython = None
try:
    from pyqed.mps import cpp_davidson as _cpp_davidson
except Exception:  # pragma: no cover - optional compiled backend.
    _cpp_davidson = None
try:
    from pyqed.mps.symmetry import (
        BlockTensor,
        tensordot,
        solve_davidson,
        solve_davidson_block,
        QN,
        Sector,
        SymmetryManager,
        is_sector_like,
        zero_like_sector,
    )
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None
from pyqed.mps.abelian_direct import (
    AbelianCompactBlockDataTable,
    AbelianCompactRenormalizedDataTable,
    AbelianContextualComponentStore,
    AbelianContextualDirectFamilyBuilder,
    AbelianContextualFamilyDispatchPlan,
    AbelianContextualFamilyBuildOptions,
    AbelianCompositePackedDirectFamilyEntries,
    AbelianDirectFamilyDispatchPlan,
    AbelianDirectFamilyLiteralPlan,
    AbelianDirectRoutePlan,
    AbelianDenseBoundaryActionDataTable,
    AbelianGroupedRenormalizedDataTable,
    AbelianEnvironmentTensorData,
    AbelianLocalActionPlan,
    AbelianLocalActionPlanCache,
    AbelianMovingEnvironmentFlatMatvec,
    AbelianMovingEnvironmentTables,
    AbelianNativeExactPatternComponentTable,
    AbelianNativeExactPatternOperatorTable,
    AbelianNativeGeneratorOperatorTable,
    AbelianNativePairBoundaryOperatorTable,
    AbelianLocalVectorLayout,
    AbelianOperatorFamilyPlan,
    AbelianPackedDirectFamilyEntries,
    AbelianPackedBoundaryTensor,
    AbelianPackedContextualBoundaryTable,
    AbelianPackedIdentityLocalEntry,
    AbelianPackedLocalGeneratorEntry,
    AbelianPackedLocalStateProto,
    AbelianPackedTensorViewCache,
    AbelianPlannedPackedDirectFamilyEntries,
    AbelianRenormalizedActionDataTable,
    AbelianSiteTensorData,
    AbelianSparseBoundaryActionDataTable,
    AbelianSameSidePBoundaryValueTable,
    AbelianSameSidePRouteIdentityEntries,
    AbelianSameSidePRoutePlan,
    AbelianSpatialLocalOperatorBuilder,
    AbelianTwoSiteUpdateData,
    abelian_axis_sector_dims,
    abelian_contract_from_left_data,
    abelian_contract_from_right_data,
    _coalesced_packed_identity_local_entries,
    _coalesced_packed_local_generator_entries,
    _pack_two_site_split_layout_integer_sector_ids,
    abelian_apply_block_preconditioner,
    abelian_apply_jacobi_preconditioner,
    abelian_block_data_dtype,
    abelian_build_block_preconditioner_blocks,
    abelian_flat_qchem_jacobi_diagonal,
    abelian_flatten_to_layout,
    abelian_local_layout_from_tensor,
    abelian_local_layout_size,
    abelian_layout_from_map,
    abelian_layout_offsets,
    abelian_merge_layout_tensor,
    abelian_merge_adjacent_site_tensors,
    abelian_merge_normalize_adjacent_site_tensors,
    abelian_merge_normalize_flatten_adjacent_site_tensors,
    abelian_environment_advance_payload_stats,
    abelian_packed_boundary_advance_payload_stats,
    abelian_svd_kernel_stats,
    abelian_project_tensor_to_layout,
    abelian_project_tensor_to_layout_with_stats,
    abelian_qns_from_layout,
    abelian_remap_flat_layout,
    abelian_safe_two_site_layout_map,
    abelian_sector_signature,
    abelian_tensor_data_tensordot,
    abelian_transpose_tensor_data,
    abelian_truncate_layout_map_by_norm,
    abelian_unflatten_data_from_layout,
    abelian_zero_data_from_layout,
    abelian_extend_projected_hamiltonian,
    abelian_lowest_ritz_state,
    abelian_multiply_s_v_data,
    abelian_multiply_u_s_data,
    abelian_normalize_flat_vector,
    abelian_orthogonalize_candidate,
    abelian_packed_local_action_apply_clean,
    abelian_packed_local_action_matches_reference,
    abelian_packed_local_action_probe_reference,
    abelian_restart_basis_from_vector,
    abelian_state_averaged_two_site_svd_from_permuted_data,
    abelian_site_tensors_from_split,
    abelian_split_state_averaged_two_site_svd_data,
    abelian_split_flat_two_site_svd_data_from_kernel,
    abelian_split_flat_two_site_svd_data,
    abelian_split_two_site_svd_data,
    abelian_two_site_svd_from_permuted_data,
    abelian_two_site_mps_flow_valid,
    advance_abelian_packed_left_identity_boundary,
    advance_abelian_packed_left_boundary,
    advance_abelian_packed_right_identity_boundary,
    advance_abelian_packed_right_boundary,
    abelian_typed_direct_entry_buckets,
    abelian_generator_owner_from_support,
    abelian_generator_region_from_support,
    compare_abelian_packed_boundary_tensors,
    compose_abelian_packed_boundary_operators,
    make_contextual_family_records,
    native_p_owner_records,
    abelian_packed_tensor_axis_qns,
    make_abelian_packed_initial_left_environment,
    make_abelian_packed_initial_right_environment,
    make_abelian_packed_local_generator_pair,
    is_abelian_packed_boundary_tensor,
    merge_abelian_same_side_p_route_plan,
    pack_abelian_boundary_tensor,
    packed_same_side_p_product_correction,
    scale_abelian_boundary_tensor,
    sum_abelian_packed_boundary_terms,
    unpack_abelian_packed_boundary_tensor,
)
from pyqed.mps.abelian_storage import (
    abelian_environment_scalar,
    make_initial_left_environment,
    make_initial_right_environment,
    make_identity_mpo_site_from_mps_site,
    make_abelian_site_tensor,
    to_native_abelian_site_tensor,
)


if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _numba_matrix_chain_e_a_accum(e_blk, a_blk, out):
        na, ni, nj = e_blk.shape
        _nj_a, nk, nx, ny = a_blk.shape
        for a in range(na):
            for i in range(ni):
                for k in range(nk):
                    for x in range(nx):
                        for y in range(ny):
                            total = out[a, i, k, x, y] * 0.0
                            for j in range(nj):
                                total += e_blk[a, i, j] * a_blk[j, k, x, y]
                            out[a, i, k, x, y] += total

    @_numba_njit(cache=True)
    def _numba_matrix_chain_r_w1_accum(r_blk, w_blk, out):
        na, ni, nk, nx, ny = r_blk.shape
        _na_w, nb, nu, _nx_w = w_blk.shape
        for i in range(ni):
            for k in range(nk):
                for y in range(ny):
                    for b in range(nb):
                        for u in range(nu):
                            total = out[i, k, y, b, u] * 0.0
                            for a in range(na):
                                for x in range(nx):
                                    total += r_blk[a, i, k, x, y] * w_blk[a, b, u, x]
                            out[i, k, y, b, u] += total

    @_numba_njit(cache=True)
    def _numba_matrix_chain_t2_w2_accum(t2_blk, w_blk, out):
        ni, nk, ny, nb, nu = t2_blk.shape
        _nb_w, nc, nv, _ny_w = w_blk.shape
        for i in range(ni):
            for k in range(nk):
                for u in range(nu):
                    for c in range(nc):
                        for v in range(nv):
                            total = out[i, k, u, c, v] * 0.0
                            for b in range(nb):
                                for y in range(ny):
                                    total += t2_blk[i, k, y, b, u] * w_blk[b, c, v, y]
                            out[i, k, u, c, v] += total

    @_numba_njit(cache=True)
    def _numba_matrix_chain_t3_f_accum(t3_blk, f_blk, out):
        ni, nk, nu, nc, nv = t3_blk.shape
        _nc_f, nl, _nk_f = f_blk.shape
        for i in range(ni):
            for l in range(nl):
                for u in range(nu):
                    for v in range(nv):
                            total = out[i, l, u, v] * 0.0
                            for c in range(nc):
                                for k in range(nk):
                                    total += t3_blk[i, k, u, c, v] * f_blk[c, l, k]
                            out[i, l, u, v] += total

    @_numba_njit(cache=True)
    def _numba_batched_matrix_chain_e_a_accum(e_stack, a_buf, a_pos, out_buf, out_pos):
        batch, na, ni, nj = e_stack.shape
        _nblocks, _nj_a, nk, nx, ny = a_buf.shape
        for entry in range(batch):
            ai = a_pos[entry]
            oi = out_pos[entry]
            for a in range(na):
                for i in range(ni):
                    for k in range(nk):
                        for x in range(nx):
                            for y in range(ny):
                                total = out_buf[oi, a, i, k, x, y] * 0.0
                                for j in range(nj):
                                    total += e_stack[entry, a, i, j] * a_buf[ai, j, k, x, y]
                                out_buf[oi, a, i, k, x, y] += total

    @_numba_njit(cache=True)
    def _numba_batched_matrix_chain_r_w_accum(r_buf, r_pos, w_stack, out_buf, out_pos):
        batch, na, nb, nu, nx = w_stack.shape
        _nr, _na_r, ni, nk, _nx_r, ny = r_buf.shape
        for entry in range(batch):
            ri = r_pos[entry]
            oi = out_pos[entry]
            for i in range(ni):
                for k in range(nk):
                    for y in range(ny):
                        for b in range(nb):
                            for u in range(nu):
                                total = out_buf[oi, i, k, y, b, u] * 0.0
                                for a in range(na):
                                    for x in range(nx):
                                        total += r_buf[ri, a, i, k, x, y] * w_stack[entry, a, b, u, x]
                                out_buf[oi, i, k, y, b, u] += total

    @_numba_njit(cache=True)
    def _numba_batched_matrix_chain_t2_w_accum(t2_buf, t2_pos, w_stack, out_buf, out_pos):
        batch, nb, nc, nv, ny = w_stack.shape
        _nt2, ni, nk, _ny_t2, _nb_t2, nu = t2_buf.shape
        for entry in range(batch):
            ti = t2_pos[entry]
            oi = out_pos[entry]
            for i in range(ni):
                for k in range(nk):
                    for u in range(nu):
                        for c in range(nc):
                            for v in range(nv):
                                total = out_buf[oi, i, k, u, c, v] * 0.0
                                for b in range(nb):
                                    for y in range(ny):
                                        total += t2_buf[ti, i, k, y, b, u] * w_stack[entry, b, c, v, y]
                                out_buf[oi, i, k, u, c, v] += total

    @_numba_njit(cache=True)
    def _numba_batched_matrix_chain_t3_f_accum(t3_buf, t3_pos, f_stack, out_buf, out_pos):
        batch, nc, nl, nk = f_stack.shape
        _nt3, ni, _nk_t3, nu, _nc_t3, nv = t3_buf.shape
        for entry in range(batch):
            ti = t3_pos[entry]
            oi = out_pos[entry]
            for i in range(ni):
                for l in range(nl):
                    for u in range(nu):
                        for v in range(nv):
                            total = out_buf[oi, i, l, u, v] * 0.0
                            for c in range(nc):
                                for k in range(nk):
                                    total += t3_buf[ti, i, k, u, c, v] * f_stack[entry, c, l, k]
                            out_buf[oi, i, l, u, v] += total

    @_numba_njit(parallel=True, fastmath=True, cache=True)
    def _numba_batched_matrix_chain_e_a_accum_parallel(e_stack, a_buf, a_pos, out_buf, out_pos):
        batch, na, ni, nj = e_stack.shape
        _nblocks, _nj_a, nk, nx, ny = a_buf.shape
        total_size = na * ni * nk * nx * ny
        for entry in range(batch):
            ai = a_pos[entry]
            oi = out_pos[entry]
            for flat in _numba_prange(total_size):
                y = flat % ny
                tmp = flat // ny
                x = tmp % nx
                tmp = tmp // nx
                k = tmp % nk
                tmp = tmp // nk
                i = tmp % ni
                a = tmp // ni
                total = out_buf[oi, a, i, k, x, y] * 0.0
                for j in range(nj):
                    total += e_stack[entry, a, i, j] * a_buf[ai, j, k, x, y]
                out_buf[oi, a, i, k, x, y] += total

    @_numba_njit(parallel=True, fastmath=True, cache=True)
    def _numba_batched_matrix_chain_r_w_accum_parallel(r_buf, r_pos, w_stack, out_buf, out_pos):
        batch, na, nb, nu, nx = w_stack.shape
        _nr, _na_r, ni, nk, _nx_r, ny = r_buf.shape
        total_size = ni * nk * ny * nb * nu
        for entry in range(batch):
            ri = r_pos[entry]
            oi = out_pos[entry]
            for flat in _numba_prange(total_size):
                u = flat % nu
                tmp = flat // nu
                b = tmp % nb
                tmp = tmp // nb
                y = tmp % ny
                tmp = tmp // ny
                k = tmp % nk
                i = tmp // nk
                total = out_buf[oi, i, k, y, b, u] * 0.0
                for a in range(na):
                    for x in range(nx):
                        total += r_buf[ri, a, i, k, x, y] * w_stack[entry, a, b, u, x]
                out_buf[oi, i, k, y, b, u] += total

    @_numba_njit(parallel=True, fastmath=True, cache=True)
    def _numba_batched_matrix_chain_t2_w_accum_parallel(t2_buf, t2_pos, w_stack, out_buf, out_pos):
        batch, nb, nc, nv, ny = w_stack.shape
        _nt2, ni, nk, _ny_t2, _nb_t2, nu = t2_buf.shape
        total_size = ni * nk * nu * nc * nv
        for entry in range(batch):
            ti = t2_pos[entry]
            oi = out_pos[entry]
            for flat in _numba_prange(total_size):
                v = flat % nv
                tmp = flat // nv
                c = tmp % nc
                tmp = tmp // nc
                u = tmp % nu
                tmp = tmp // nu
                k = tmp % nk
                i = tmp // nk
                total = out_buf[oi, i, k, u, c, v] * 0.0
                for b in range(nb):
                    for y in range(ny):
                        total += t2_buf[ti, i, k, y, b, u] * w_stack[entry, b, c, v, y]
                out_buf[oi, i, k, u, c, v] += total

    @_numba_njit(parallel=True, fastmath=True, cache=True)
    def _numba_batched_matrix_chain_t3_f_accum_parallel(t3_buf, t3_pos, f_stack, out_buf, out_pos):
        batch, nc, nl, nk = f_stack.shape
        _nt3, ni, _nk_t3, nu, _nc_t3, nv = t3_buf.shape
        total_size = ni * nl * nu * nv
        for entry in range(batch):
            ti = t3_pos[entry]
            oi = out_pos[entry]
            for flat in _numba_prange(total_size):
                v = flat % nv
                tmp = flat // nv
                u = tmp % nu
                tmp = tmp // nu
                l = tmp % nl
                i = tmp // nl
                total = out_buf[oi, i, l, u, v] * 0.0
                for c in range(nc):
                    for k in range(nk):
                        total += t3_buf[ti, i, k, u, c, v] * f_stack[entry, c, l, k]
                out_buf[oi, i, l, u, v] += total
else:
    _numba_matrix_chain_e_a_accum = None
    _numba_matrix_chain_r_w1_accum = None
    _numba_matrix_chain_t2_w2_accum = None
    _numba_matrix_chain_t3_f_accum = None
    _numba_batched_matrix_chain_e_a_accum = None
    _numba_batched_matrix_chain_r_w_accum = None
    _numba_batched_matrix_chain_t2_w_accum = None
    _numba_batched_matrix_chain_t3_f_accum = None
    _numba_batched_matrix_chain_e_a_accum_parallel = None
    _numba_batched_matrix_chain_r_w_accum_parallel = None
    _numba_batched_matrix_chain_t2_w_accum_parallel = None
    _numba_batched_matrix_chain_t3_f_accum_parallel = None

def SpinHalfFermionOperators(filling=1.):
    d = 4
    states = ['empty', 'up', 'down', 'full']
    # 0) Build the operators.
    Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float64)
    Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float64)

    Nu = np.diag(Nu_diag)
    Nd = np.diag(Nd_diag)
    Ntot = np.diag(Nu_diag + Nd_diag)
    dN = np.diag(Nu_diag + Nd_diag - filling)
    NuNd = np.diag(Nu_diag * Nd_diag)
    JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
    JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
    JW = JWu * JWd  # (-1)^{Nu+Nd}


    Cu = np.zeros((d, d))
    Cu[0, 1] = Cu[2, 3] = 1
    Cdu = np.transpose(Cu)
    # For spin-down annihilation operator: include a Jordan-Wigner string JWu
    # this ensures that Cdu.Cd = - Cd.Cdu
    # c.f. the chapter on the Jordan-Wigner trafo in the userguide
    Cd_noJW = np.zeros((d, d))
    Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
    Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
    Cdd = np.transpose(Cd)

    # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
    # where S^gamma is the 2x2 matrix for spin-half
    Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
    Sp = np.dot(Cdu, Cd)
    Sm = np.dot(Cdd, Cu)
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)

    ops = dict(JW=JW, JWu=JWu, JWd=JWd,
               Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
               Nu=Nu, Nd=Nd, Ntot=Ntot, NuNd=NuNd, dN=dN,
               Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable
    return ops

# below is some helper functions for the dmrg sweep with U(1) symmetry.
_DETERMINISTIC_TRUNCATION_RTOL = 1e-10
_DETERMINISTIC_TRUNCATION_ATOL = 1e-12
_DETERMINISTIC_SUBSPACE_RTOL = 1e-10
_DETERMINISTIC_SUBSPACE_ATOL = 1e-12


def _stable_sector_sort_key(value):
    if hasattr(value, "labels") and hasattr(value, "components"):
        return (
            type(value).__name__,
            tuple(str(label) for label in value.labels),
            tuple(_stable_sector_sort_key(component) for component in value.components),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_stable_sector_sort_key(item) for item in value))
    if isinstance(value, (np.integer, int)):
        return ("int", int(value))
    if isinstance(value, (np.floating, float)):
        return ("float", float(value))
    if isinstance(value, str):
        return ("str", value)
    return (type(value).__name__, repr(value))


def _cluster_sorted_values(values, rtol, atol):
    values = np.asarray(values, dtype=float)
    clusters = []
    start = 0
    n = len(values)
    while start < n:
        stop = start + 1
        reference = float(abs(values[start]))
        while stop < n:
            current = float(abs(values[stop]))
            tol = max(float(atol), float(rtol) * max(reference, current))
            if abs(float(values[stop]) - float(values[start])) > tol:
                break
            stop += 1
        clusters.append((start, stop))
        start = stop
    return clusters


def _sort_singular_entries(entries, rtol=None, atol=None):
    """Sort singular values with deterministic tie-breaking across sectors."""
    if not entries:
        return []
    rtol = _DETERMINISTIC_TRUNCATION_RTOL if rtol is None else float(rtol)
    atol = _DETERMINISTIC_TRUNCATION_ATOL if atol is None else float(atol)
    ordered = sorted(
        entries,
        key=lambda item: (-float(np.real(item[0])), _stable_sector_sort_key(item[1]), int(item[2])),
    )
    values = [float(np.real(item[0])) for item in ordered]
    stable = []
    for start, stop in _cluster_sorted_values(values, rtol=rtol, atol=atol):
        cluster = sorted(
            ordered[start:stop],
            key=lambda item: (_stable_sector_sort_key(item[1]), int(item[2])),
        )
        stable.extend(cluster)
    return stable


def _fix_vector_phase_inplace(vec):
    if vec.size == 0:
        return vec
    idx = int(np.argmax(np.abs(vec)))
    ref = vec[idx]
    if abs(ref) <= 0.0:
        return vec
    if np.iscomplexobj(vec):
        vec *= np.conj(ref) / abs(ref)
    elif ref < 0:
        vec *= -1.0
    return vec


def _canonical_subspace_rotation(left_basis):
    """Return a deterministic unitary rotation for a near-degenerate subspace."""
    ncol = left_basis.shape[1]
    if ncol <= 1:
        return np.eye(ncol, dtype=left_basis.dtype)
    dtype = np.result_type(left_basis.dtype, np.complex128 if np.iscomplexobj(left_basis) else np.float64)
    vectors = []
    tol = 100.0 * np.finfo(float).eps * max(1, ncol)
    for row in range(left_basis.shape[0]):
        candidate = np.asarray(left_basis[row, :].conj(), dtype=dtype).copy()
        for _ in range(2):
            for basis_vec in vectors:
                candidate -= basis_vec * np.vdot(basis_vec, candidate)
        norm = np.linalg.norm(candidate)
        if norm <= tol:
            continue
        candidate /= norm
        _fix_vector_phase_inplace(candidate)
        vectors.append(candidate)
        if len(vectors) == ncol:
            break
    if len(vectors) != ncol:
        return np.eye(ncol, dtype=dtype)
    return np.column_stack(vectors)


def _canonicalize_basis_phases(basis):
    for i in range(basis.shape[1]):
        _fix_vector_phase_inplace(basis[:, i])
    return basis


def _canonicalize_svd_pair(U, S, Vt, rtol=None, atol=None):
    """Choose a deterministic gauge for SVD columns in tied singular spaces."""
    if len(S) == 0:
        return U, S, Vt
    rtol = _DETERMINISTIC_SUBSPACE_RTOL if rtol is None else float(rtol)
    atol = _DETERMINISTIC_SUBSPACE_ATOL if atol is None else float(atol)
    for start, stop in _cluster_sorted_values(S, rtol=rtol, atol=atol):
        if stop - start <= 1:
            continue
        rotation = _canonical_subspace_rotation(U[:, start:stop])
        U[:, start:stop] = U[:, start:stop] @ rotation
        Vt[start:stop, :] = rotation.conj().T @ Vt[start:stop, :]
    for i in range(min(U.shape[1], Vt.shape[0])):
        idx = int(np.argmax(np.abs(U[:, i])))
        ref = U[idx, i]
        if abs(ref) <= 0.0:
            continue
        if np.iscomplexobj(U) or np.iscomplexobj(Vt):
            phase = np.conj(ref) / abs(ref)
            U[:, i] *= phase
            Vt[i, :] *= np.conj(phase)
        elif ref < 0:
            U[:, i] *= -1.0
            Vt[i, :] *= -1.0
    return U, S, Vt


def _canonicalize_density_basis(basis, strengths, rtol=None, atol=None):
    """Choose a deterministic gauge for eigenvectors of tied density roots."""
    if len(strengths) == 0:
        return basis
    rtol = _DETERMINISTIC_SUBSPACE_RTOL if rtol is None else float(rtol)
    atol = _DETERMINISTIC_SUBSPACE_ATOL if atol is None else float(atol)
    for start, stop in _cluster_sorted_values(strengths, rtol=rtol, atol=atol):
        if stop - start <= 1:
            continue
        rotation = _canonical_subspace_rotation(basis[:, start:stop])
        basis[:, start:stop] = basis[:, start:stop] @ rotation
    return _canonicalize_basis_phases(basis)


def svd_symmetric(AA, cutoff=1e-10, m_max=None):

    AA_perm = AA.transpose(0, 2, 1, 3)
    svd_result = abelian_two_site_svd_from_permuted_data(
        AA_perm.data,
        m_max=m_max,
    )

    # Original qns from AA
    qns_L      = AA_perm.qns[0]
    qns_pL     = AA_perm.qns[1]
    qns_R      = AA_perm.qns[2]
    qns_pR     = AA_perm.qns[3]


    carrier = AbelianSiteTensorData if isinstance(AA, AbelianSiteTensorData) else BlockTensor
    U = carrier(
        svd_result.u_data,
        qns=[qns_L, qns_pL, svd_result.bond_qns],
        dirs=[AA_perm.dirs[0], AA_perm.dirs[1], 1]
    )

    V = carrier(
        svd_result.v_data,
        qns=[svd_result.bond_qns, qns_R, qns_pR],
        dirs=[-1, AA_perm.dirs[2], AA_perm.dirs[3]]
    )

    return (
        U,
        V,
        svd_result.s_data,
        svd_result.truncation_error,
        svd_result.kept_states,
    )




def _make_complementary_boundary_stack(families, nsites):
    if families is None:
        return None, {}
    try:
        from pyqed.mps.nonabelian.renormalized import (
            ComplementaryFamilyRenormalizedOperatorBlock,
            ComplementaryFamilyRenormalizedOperatorTable,
            ComplementaryRenormalizedOperatorStack,
        )
    except Exception:
        return None, {}

    stack = ComplementaryRenormalizedOperatorStack(families=families)
    payloads = {}

    def _attach_family_table(entry):
        blocks = {}
        for channel, name in enumerate(entry.family_names):
            payload = entry.family_payloads.get(str(name))
            n_terms = 0 if payload is None else int(payload.n_terms)
            cross_terms = 0 if payload is None else int(payload.cross_terms)
            channels = (int(channel),) if cross_terms > 0 else ()
            payload_keys = () if payload is None else tuple(payload.entries)
            blocks[str(name)] = ComplementaryFamilyRenormalizedOperatorBlock(
                family_name=str(name),
                channels=channels,
                symbolic_terms=int(cross_terms),
                payload_keys=payload_keys,
                stored_elements=int(n_terms),
                payload_norm=0.0 if payload is None else float(payload.coefficient_norm),
                coefficient_terms=int(n_terms),
                coefficient_cross_terms=int(cross_terms),
            )
        object.__setattr__(
            entry,
            "family_operator_table",
            ComplementaryFamilyRenormalizedOperatorTable(
                side=entry.side,
                bond=entry.bond,
                family_blocks=blocks,
                source="abelian_complementary_payload_table",
            ),
        )
        return entry

    parent = None
    for bond in range(max(0, int(nsites) - 1)):
        entry = _attach_family_table(stack.put(
            "left",
            bond,
            signature=("abelian_complementary_boundary", "left", int(bond)),
            source="abelian_sweep_boundary",
            parent_key=parent,
        ))
        payloads[("left", int(bond))] = entry
        parent = entry.key

    parent = None
    for bond in range(max(0, int(nsites) - 1), 0, -1):
        entry = _attach_family_table(stack.put(
            "right",
            bond,
            signature=("abelian_complementary_boundary", "right", int(bond)),
            source="abelian_sweep_boundary",
            parent_key=parent,
        ))
        payloads[("right", int(bond))] = entry
        parent = entry.key

    return stack, payloads


def _blocktensor_from_abelian_layout_data(vector_layout, data):
    return BlockTensor(
        data,
        [list(q) for q in vector_layout.qns],
        list(vector_layout.dirs),
    )


def _tensor_from_abelian_layout_data_like(proto, vector_layout, data):
    if isinstance(proto, AbelianSiteTensorData):
        return AbelianSiteTensorData(
            data,
            [list(q) for q in vector_layout.qns],
            list(vector_layout.dirs),
        )
    return _blocktensor_from_abelian_layout_data(vector_layout, data)


class AbelianComplementaryBoundaryActionTable(AbelianDenseBoundaryActionDataTable):
    """Dense boundary action table with native-aware compatibility adapters."""

    def flatten(self, A):
        return self.flatten_data(getattr(A, "data", {}) or {})

    def unflatten(self, vec):
        return _blocktensor_from_abelian_layout_data(
            self.vector_layout,
            self.unflatten_data(vec),
        )

    def apply(self, A):
        return _tensor_from_abelian_layout_data_like(
            A,
            self.vector_layout,
            self.apply_data(getattr(A, "data", {}) or {}),
        )

    def apply_channels(self, A):
        return {
            name: _tensor_from_abelian_layout_data_like(A, self.vector_layout, data)
            for name, data in self.apply_channels_data(
                getattr(A, "data", {}) or {}
            ).items()
        }


class AbelianSparseComplementaryBoundaryActionTable(AbelianSparseBoundaryActionDataTable):
    """Sparse action table with native-aware compatibility adapters."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("kernel_backend", _packed_cython)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_csr(cls, *args, **kwargs):
        kwargs.setdefault("kernel_backend", _packed_cython)
        return super().from_csr(*args, **kwargs)

    def flatten(self, A):
        return self.flatten_data(getattr(A, "data", {}) or {})

    def unflatten(self, vec):
        return _blocktensor_from_abelian_layout_data(
            self.vector_layout,
            self.unflatten_data(vec),
        )

    def apply(self, A):
        return _tensor_from_abelian_layout_data_like(
            A,
            self.vector_layout,
            self.apply_data(getattr(A, "data", {}) or {}),
        )

    def apply_channels(self, A):
        return {}


class AbelianRenormalizedOperatorActionTable(AbelianRenormalizedActionDataTable):
    """Renormalized action table with native-aware compatibility adapters."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("kernel_backend", _packed_cython)
        super().__init__(*args, **kwargs)

    def flatten(self, A):
        return self.flatten_data(getattr(A, "data", {}) or {})

    def unflatten(self, vec):
        return _blocktensor_from_abelian_layout_data(
            self.vector_layout,
            self.unflatten_data(vec),
        )

    def apply(self, A):
        return _tensor_from_abelian_layout_data_like(
            A,
            self.vector_layout,
            self.apply_data(getattr(A, "data", {}) or {}),
        )

    def apply_channels(self, A):
        return {}


__all__ = [name for name in globals() if not name.startswith("__")]
