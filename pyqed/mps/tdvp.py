"""TDVP propagation for dense-layout MPS/MPO objects."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal, expm

from pyqed.mps.dense_canonical import left_qr as _left_qr
from pyqed.mps.dense_canonical import right_rq as _right_rq
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.mps import (
    MPS,
    MPO,
    _release_free_numeric_pages,
    contract_from_left,
    contract_from_right,
    dense_to_symmetric,
    dense_to_symmetric_mpo,
    initial_E,
    initial_F,
    symmetric_to_dense,  # noqa: F401 - retained as a TDVP module helper
)
from pyqed.mps.abelian_direct import (
    AbelianEnvironmentTensorData,
    AbelianSiteTensorData,
    _cpp_table_kernel,
    abelian_merge_adjacent_site_tensors,
    abelian_site_tensors_from_split,
    abelian_split_two_site_svd_data,
    abelian_tensor_data_tensordot,
    abelian_transpose_tensor_data,
)
from pyqed.mps.abelian_storage import (
    make_identity_mpo_site_from_mps_site,
    to_native_abelian_site_tensor,
)
from pyqed.mps.symmetry import AbelianSector, zero_like_sector

_tdvp_cpp = None
_tdvp_cpp_tried = False
_dense_tdvp_cpp = None
_dense_tdvp_cpp_tried = False
_dense_tdvp_cpp_last_error = None
_BLOCK_HEFF_PLAN_CACHE = OrderedDict()
_BLOCK_HEFF_PLAN_CACHE_MAX = 512
_BLOCK_HEFF_BACKEND_DECISION_CACHE = OrderedDict()
_BLOCK_HEFF_BACKEND_DECISION_CACHE_MAX = 1024
_BLOCK_HEFF_PLAN_MIN_ROUTE_ESTIMATE = 128
_BLOCK_MOVING_ENV_COUNTERS = (
    "environment_plan_builds",
    "environment_plan_replacements",
    "environment_plan_cache_hits",
    "environment_plan_advance_calls",
    "environment_plan_failures",
    "sweep_environment_step_calls",
    "sweep_environment_step_updates",
    "sweep_environment_step_auto_calls",
    "sweep_environment_step_failures",
    "one_site_tdvp_sweep_calls",
    "one_site_tdvp_sweep_failures",
    "one_site_tdvp_sweep_site_evolutions",
    "one_site_tdvp_sweep_bond_evolutions",
    "one_site_tdvp_sweep_site_matvecs",
    "one_site_tdvp_sweep_bond_matvecs",
    "one_site_tdvp_sweep_left_qr_calls",
    "one_site_tdvp_sweep_right_rq_calls",
    "one_site_tdvp_sweep_environment_advances",
    "two_site_tdvp_sweep_calls",
    "two_site_tdvp_sweep_failures",
    "two_site_tdvp_sweep_two_site_evolutions",
    "two_site_tdvp_sweep_site_evolutions",
    "two_site_tdvp_sweep_two_site_matvecs",
    "two_site_tdvp_sweep_site_matvecs",
    "two_site_tdvp_sweep_merges",
    "two_site_tdvp_sweep_splits",
    "two_site_tdvp_sweep_environment_advances",
)
_BLOCK_MOVING_ENV_TIMERS = (
    "environment_plan_build_seconds",
    "environment_plan_advance_seconds",
    "one_site_tdvp_sweep_seconds",
    "two_site_tdvp_sweep_seconds",
)
_BLOCK_MOVING_ENV_ABSOLUTE = (
    "environment_plan_records",
    "environment_plan_last_routes",
    "environment_plan_last_blocks",
    "one_site_tdvp_sweep_last_nsites",
    "two_site_tdvp_sweep_last_nsites",
)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


_BLOCK_HEFF_CPP_MAX_ROUTE_ESTIMATE = _env_int(
    "PYQED_TDVP_BLOCK_HEFF_CPP_MAX_ROUTE", 20_000_000
)
_BLOCK_HEFF_AUTOTUNE_MAX_ROUTE_ESTIMATE = _env_int(
    "PYQED_TDVP_BLOCK_HEFF_AUTOTUNE_MAX_ROUTE", 20_000_000
)
_BLOCK_QR_CPP_MIN_ELEMENTS = _env_int(
    "PYQED_TDVP_BLOCK_QR_CPP_MIN_ELEMENTS", 1_000_000_000
)
_BLOCK_ONE_SITE_CPP_ENGINE = _env_int("PYQED_TDVP_CPP_ONE_SITE_ENGINE", 1)
_BLOCK_TWO_SITE_CPP_ENGINE = _env_int("PYQED_TDVP_CPP_TWO_SITE_ENGINE", 1)
_SUM_TDVP_CPP_MAX_DIRECT_SUM_ELEMENTS = _env_int(
    "PYQED_SUM_TDVP_CPP_MAX_DIRECT_SUM_ELEMENTS", 20_000_000
)
_AFFINE_BLOCK_SPARSE_MPO_CACHE = OrderedDict()
_AFFINE_BLOCK_SPARSE_MPO_CACHE_MAX = 64
_BLOCK_SPARSE_MPO_CACHE = OrderedDict()
_BLOCK_SPARSE_MPO_CACHE_MAX = _env_int("PYQED_TDVP_BLOCK_SPARSE_MPO_CACHE_MAX", 16)


def _is_lanczos_method(method):
    return str(method).lower().replace("_", "-") in {
        "lanczos",
        "hermitian",
        "hermitian-lanczos",
    }


def _cpp_tdvp_available():
    global _tdvp_cpp
    global _tdvp_cpp_tried

    if _tdvp_cpp is None and not _tdvp_cpp_tried:
        _tdvp_cpp_tried = True
        try:
            from . import tdvp_cpp as module

            _tdvp_cpp = module
        except Exception:
            _tdvp_cpp = None
    return (
        _tdvp_cpp is not None
        and getattr(_tdvp_cpp, "CPP_TDVP_AVAILABLE", False)
        and getattr(_tdvp_cpp, "CPP_TDVP_HAS_BLAS", False)
        and getattr(_tdvp_cpp, "site_lanczos", None) is not None
        and getattr(_tdvp_cpp, "two_site_lanczos", None) is not None
        and getattr(_tdvp_cpp, "bond_lanczos", None) is not None
    )


def _cpp_dense_tdvp_workspace_type():
    """Return the cached dense local-exp workspace type when native kernels exist."""
    global _dense_tdvp_cpp
    global _dense_tdvp_cpp_tried
    if _dense_tdvp_cpp is None and not _dense_tdvp_cpp_tried:
        _dense_tdvp_cpp_tried = True
        try:
            from . import cpp_davidson as module

            _dense_tdvp_cpp = module
        except Exception:
            _dense_tdvp_cpp = None
    if _dense_tdvp_cpp is None or not getattr(
        _dense_tdvp_cpp, "CPP_DAVIDSON_AVAILABLE", False
    ):
        return None
    workspace_type = getattr(_dense_tdvp_cpp, "DenseSweepWorkspace", None)
    if workspace_type is None or not hasattr(workspace_type, "evolve_two_site"):
        return None
    return workspace_type


def _new_cpp_dense_tdvp_workspace():
    workspace_type = _cpp_dense_tdvp_workspace_type()
    if workspace_type is None:
        return None
    try:
        return workspace_type()
    except Exception:
        return None


def _dense_mpo_pair_for_workspace(W_left, W_right):
    """Whether a dense cached plan beats the physical-transition kernel."""
    total = W_left.size + W_right.size
    if total == 0:
        return False
    nnz = np.count_nonzero(np.abs(W_left) > 1.0e-14)
    nnz += np.count_nonzero(np.abs(W_right) > 1.0e-14)
    return 4 * int(nnz) >= 3 * int(total)


def _dense_mpo_for_workspace(mpo):
    total = sum(np.asarray(W).size for W in mpo)
    if total == 0:
        return False
    nnz = sum(np.count_nonzero(np.abs(W) > 1.0e-14) for W in mpo)
    return 4 * int(nnz) >= 3 * int(total)


def _arnoldi_expm_apply(vec, shape, apply_heff, dt, *, krylov_dim=12, tol=1.0e-13):
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm <= tol:
        return vec.reshape(shape)

    size = vec.size
    mmax = min(int(krylov_dim), size)
    basis = np.zeros((size, mmax), dtype=complex)
    h_krylov = np.zeros((mmax, mmax), dtype=complex)
    basis[:, 0] = vec / norm
    actual_dim = mmax

    for j in range(mmax):
        trial = apply_heff(basis[:, j].reshape(shape)).reshape(-1)
        for i in range(j + 1):
            coeff = np.vdot(basis[:, i], trial)
            h_krylov[i, j] += coeff
            trial -= coeff * basis[:, i]
        for i in range(j + 1):
            coeff = np.vdot(basis[:, i], trial)
            h_krylov[i, j] += coeff
            trial -= coeff * basis[:, i]

        beta = np.linalg.norm(trial)
        actual_dim = j + 1
        if beta <= tol or j + 1 == mmax:
            break
        h_krylov[j + 1, j] = beta
        basis[:, j + 1] = trial / beta

    h_small = h_krylov[:actual_dim, :actual_dim]
    e1 = np.zeros(actual_dim, dtype=complex)
    e1[0] = norm
    evolved = basis[:, :actual_dim] @ (expm(-1j * dt * h_small) @ e1)
    return evolved.reshape(shape)


def _lanczos_expm_apply(vec, shape, apply_heff, dt, *, krylov_dim=12, tol=1.0e-13):
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm <= tol:
        return vec.reshape(shape)

    size = vec.size
    mmax = min(int(krylov_dim), size)
    basis = np.zeros((size, mmax), dtype=complex)
    alpha = np.zeros(mmax, dtype=float)
    beta = np.zeros(max(mmax - 1, 0), dtype=float)
    basis[:, 0] = vec / norm

    actual_dim = mmax
    q_prev = None
    beta_prev = 0.0
    previous_action = None
    small_action = None
    for j in range(mmax):
        q = basis[:, j]
        trial = apply_heff(q.reshape(shape)).reshape(-1)
        if q_prev is not None:
            trial -= beta_prev * q_prev
        alpha_j = np.vdot(q, trial)
        alpha[j] = float(np.real(alpha_j))
        trial -= alpha_j * q

        beta_j = np.linalg.norm(trial)
        actual_dim = j + 1
        if actual_dim == 1:
            current_action = np.array(
                [norm * np.exp(-1j * dt * alpha[0])], dtype=complex
            )
        else:
            evals, evecs = eigh_tridiagonal(
                alpha[:actual_dim], beta[: actual_dim - 1]
            )
            e1 = np.zeros(actual_dim, dtype=complex)
            e1[0] = norm
            current_action = evecs @ (
                np.exp(-1j * dt * evals) * (evecs.T.conj() @ e1)
            )
        if previous_action is not None and tol > 0.0:
            action_delta = np.linalg.norm(
                current_action
                - np.pad(previous_action, (0, 1))
            )
            if action_delta <= tol * max(1.0, norm):
                small_action = current_action
                break
        previous_action = current_action
        if beta_j <= tol or j + 1 == mmax:
            break
        beta[j] = beta_j
        q_prev = q
        beta_prev = beta_j
        basis[:, j + 1] = trial / beta_j

    if small_action is None:
        small_action = previous_action
    evolved = basis[:, :actual_dim] @ small_action
    return evolved.reshape(shape)


def _krylov_expm_apply(
    vec,
    shape,
    apply_heff,
    dt,
    *,
    krylov_dim=12,
    tol=1.0e-13,
    method="lanczos",
):
    key = str(method).lower().replace("_", "-")
    if key in {"lanczos", "hermitian", "hermitian-lanczos"}:
        return _lanczos_expm_apply(
            vec,
            shape,
            apply_heff,
            dt,
            krylov_dim=krylov_dim,
            tol=tol,
        )
    if key in {"arnoldi", "generic"}:
        return _arnoldi_expm_apply(
            vec,
            shape,
            apply_heff,
            dt,
            krylov_dim=krylov_dim,
            tol=tol,
        )
    raise ValueError("krylov_method must be 'lanczos' or 'arnoldi'.")


def _mpo_factors(H):
    return H.factors if isinstance(H, MPO) else list(H)


def _copy_mpo_factor_for_tdvp(factor):
    if hasattr(factor, "qns") and hasattr(factor, "copy"):
        return factor.copy()
    return np.asarray(factor)


def _standard_mps_factors(psi):
    return [np.asarray(psi._get_std_B(i), dtype=complex).copy() for i in range(psi.L)]


def _mps_factors_norm2(factors):
    env = np.ones((1, 1), dtype=complex)
    for factor in factors:
        A = np.asarray(factor, dtype=complex)
        env = np.einsum("ab,api,bpj->ij", env, A.conj(), A, optimize=True)
    return float(np.real_if_close(env[0, 0]).real)


def _normalize_mps_factors_inplace(psi, norm2):
    norm = float(np.sqrt(max(float(norm2), 0.0)))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-norm MPS.")
    psi.factors[0] = psi.factors[0] / norm
    return psi


def _sector_labels_and_components(sector):
    if hasattr(sector, "labels") and hasattr(sector, "components"):
        return tuple(str(label).lower() for label in sector.labels), tuple(
            sector.components
        )
    if isinstance(sector, (tuple, list, np.ndarray)):
        return None, tuple(sector)
    return None, (sector,)


def _add_sector_components(left, right, labels=None):
    if len(left) != len(right):
        raise ValueError("All sector labels must have the same component count.")
    out = []
    for idx, (a, b) in enumerate(zip(left, right)):
        label = None if labels is None else labels[idx]
        if label in {"pg", "point_group", "abelianpg"}:
            out.append(int(a) ^ int(b))
        else:
            out.append(a + b)
    return tuple(out)


def _sector_components_equal(left, right, *, atol=1.0e-12):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(
            a, (float, complex, np.floating, np.complexfloating)
        ) or isinstance(b, (float, complex, np.floating, np.complexfloating)):
            if not np.isclose(a, b, atol=atol, rtol=0.0):
                return False
        elif a != b:
            return False
    return True


def _site_sector_table(local_sectors, nsites, phys_dims):
    if local_sectors is None:
        raise ValueError("local_sectors must be supplied for SymmetricTDVP.")

    local_sectors = list(local_sectors)
    if not local_sectors:
        raise ValueError("local_sectors cannot be empty.")

    first = local_sectors[0]
    first_is_site_table = (
        isinstance(first, list) and first and not hasattr(first, "components")
    ) or (
        isinstance(first, tuple)
        and first
        and not hasattr(first, "components")
        and not all(np.isscalar(item) for item in first)
    )

    if len(local_sectors) == nsites and first_is_site_table:
        tables = [list(site_sectors) for site_sectors in local_sectors]
    else:
        tables = [local_sectors for _ in range(nsites)]

    for i, (table, phys_dim) in enumerate(zip(tables, phys_dims)):
        if len(table) != phys_dim:
            raise ValueError(
                f"local_sectors for site {i} has length {len(table)}, expected physical dimension {phys_dim}."
            )
    return tables


def _local_sector_phys_dims(local_sectors, nsites):
    if local_sectors is None:
        return None

    local_sectors = list(local_sectors)
    if not local_sectors:
        raise ValueError("local_sectors cannot be empty.")

    first = local_sectors[0]
    first_is_site_table = (
        isinstance(first, list) and first and not hasattr(first, "components")
    ) or (
        isinstance(first, tuple)
        and first
        and not hasattr(first, "components")
        and not all(np.isscalar(item) for item in first)
    )
    if len(local_sectors) == nsites and first_is_site_table:
        return tuple(len(site_sectors) for site_sectors in local_sectors)
    return tuple(len(local_sectors) for _ in range(nsites))


def _dense_sector_mask(shape, local_sectors, target_sector):
    nsites = len(shape)
    labels, target_components, normalized_tables = _normalized_sector_tables(
        local_sectors,
        nsites,
        shape,
        target_sector,
    )

    mask = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        total = tuple(0 for _ in target_components)
        for site, phys_index in enumerate(index):
            components = normalized_tables[site][phys_index]
            total = _add_sector_components(total, components, labels=labels)
        if _sector_components_equal(total, target_components):
            mask[index] = True
    return mask


def _normalized_sector_tables(local_sectors, nsites, phys_dims, target_sector):
    tables = _site_sector_table(local_sectors, nsites, phys_dims)
    target_labels, target_components = _sector_labels_and_components(target_sector)
    normalized_tables = []
    for table in tables:
        normalized = []
        for sector in table:
            labels, components = _sector_labels_and_components(sector)
            if (
                target_labels is not None
                and labels is not None
                and labels != target_labels
            ):
                raise ValueError(
                    "local_sectors and target_sector use different Abelian labels."
                )
            normalized.append(components)
        normalized_tables.append(normalized)

    labels = target_labels
    if labels is None:
        for table in tables:
            for sector in table:
                local_labels, _ = _sector_labels_and_components(sector)
                if local_labels is not None:
                    labels = local_labels
                    break
            if labels is not None:
                break
    return labels, target_components, normalized_tables


def _can_finish_sector(prefix, suffixes, target, labels):
    for suffix in suffixes:
        if _sector_components_equal(
            _add_sector_components(prefix, suffix, labels=labels), target
        ):
            return True
    return False


def _sector_projector_mpo(shape, local_sectors, target_sector):
    nsites = len(shape)
    labels, target, tables = _normalized_sector_tables(
        local_sectors, nsites, shape, target_sector
    )
    zero = tuple(0 for _ in target)

    prefix_possible = [set() for _ in range(nsites + 1)]
    prefix_possible[0].add(zero)
    for site, table in enumerate(tables):
        for prefix in prefix_possible[site]:
            for components in table:
                prefix_possible[site + 1].add(
                    _add_sector_components(prefix, components, labels=labels)
                )

    suffix_possible = [set() for _ in range(nsites + 1)]
    suffix_possible[nsites].add(zero)
    for site in range(nsites - 1, -1, -1):
        for components in tables[site]:
            for suffix in suffix_possible[site + 1]:
                suffix_possible[site].add(
                    _add_sector_components(components, suffix, labels=labels)
                )

    bond_states = []
    for site in range(nsites + 1):
        states = [
            state
            for state in prefix_possible[site]
            if _can_finish_sector(state, suffix_possible[site], target, labels)
        ]
        if site == 0:
            states = [
                state for state in states if _sector_components_equal(state, zero)
            ]
        if site == nsites:
            states = [
                state for state in states if _sector_components_equal(state, target)
            ]
        states = sorted(states)
        if not states:
            raise ValueError("No product states exist in the requested target sector.")
        bond_states.append(states)

    factors = []
    for site, phys_dim in enumerate(shape):
        left_states = bond_states[site]
        right_states = bond_states[site + 1]
        right_lookup = {state: idx for idx, state in enumerate(right_states)}
        W = np.zeros(
            (len(left_states), len(right_states), phys_dim, phys_dim), dtype=complex
        )
        for left_idx, left_state in enumerate(left_states):
            for phys_index, components in enumerate(tables[site]):
                right_state = _add_sector_components(
                    left_state, components, labels=labels
                )
                right_idx = right_lookup.get(right_state)
                if right_idx is not None:
                    W[left_idx, right_idx, phys_index, phys_index] = 1.0
        factors.append(W)
    return MPO(factors, homogeneous=False), tuple(len(states) for states in bond_states)


def _full_tt_rank(shape):
    if len(shape) <= 1:
        return 1
    ranks = []
    for split in range(1, len(shape)):
        left = int(np.prod(shape[:split], dtype=np.int64))
        right = int(np.prod(shape[split:], dtype=np.int64))
        ranks.append(min(left, right))
    return max(ranks)


def spatial_fermion_number_sz_sectors():
    """Return local spatial-orbital sectors ``(N, 2*Sz)``."""
    return [(0, 0), (1, 1), (1, -1), (2, 0)]


def _update_left_env(left, A, W, *, dense_cpp=False):
    if dense_cpp:
        workspace_type = _cpp_dense_tdvp_workspace_type()
        updater = (
            None
            if _dense_tdvp_cpp is None
            else getattr(_dense_tdvp_cpp, "dense_environment_update_left", None)
        )
        if workspace_type is not None and updater is not None:
            try:
                out = updater(
                    np.asarray(W, dtype=np.complex128),
                    np.asarray(A, dtype=np.complex128),
                    np.asarray(left.transpose(1, 0, 2), dtype=np.complex128),
                    np.asarray(A, dtype=np.complex128),
                )
                return np.asarray(out, dtype=complex).transpose(1, 0, 2)
            except Exception:
                pass
    tmp = np.einsum("amb,bqs->amqs", left, A, optimize=True)
    tmp = np.einsum("mnpq,amqs->anps", W, tmp, optimize=True)
    return np.einsum("apr,anps->rns", A.conj(), tmp, optimize=True)


def _update_right_env(right, A, W, *, dense_cpp=False):
    if dense_cpp:
        workspace_type = _cpp_dense_tdvp_workspace_type()
        updater = (
            None
            if _dense_tdvp_cpp is None
            else getattr(_dense_tdvp_cpp, "dense_environment_update_right", None)
        )
        if workspace_type is not None and updater is not None:
            try:
                out = updater(
                    np.asarray(W, dtype=np.complex128),
                    np.asarray(A, dtype=np.complex128),
                    np.asarray(right.transpose(1, 0, 2), dtype=np.complex128),
                    np.asarray(A, dtype=np.complex128),
                )
                return np.asarray(out, dtype=complex).transpose(1, 0, 2)
            except Exception:
                pass
    tmp = np.einsum("bqs,rns->bqrn", A, right, optimize=True)
    tmp = np.einsum("mnpq,bqrn->bmpr", W, tmp, optimize=True)
    return np.einsum("apr,bmpr->amb", A.conj(), tmp, optimize=True)


def _build_right_envs(factors, mpo, *, dense_cpp=False):
    nsites = len(factors)
    dtype = np.result_type(*(factors + mpo), complex)
    right_envs = [None] * (nsites + 1)
    right_envs[nsites] = np.ones((1, 1, 1), dtype=dtype)
    for i in range(nsites - 1, -1, -1):
        right_envs[i] = _update_right_env(
            right_envs[i + 1], factors[i], mpo[i], dense_cpp=dense_cpp
        )
    return right_envs


def _build_left_envs(factors, mpo):
    nsites = len(factors)
    dtype = np.result_type(*(factors + mpo), complex)
    left_envs = [None] * (nsites + 1)
    left_envs[0] = np.ones((1, 1, 1), dtype=dtype)
    for i in range(nsites):
        left_envs[i + 1] = _update_left_env(left_envs[i], factors[i], mpo[i])
    return left_envs


def _sector_qn_from_components(components, labels):
    components = tuple(components)
    if labels is None:
        labels = tuple(f"q{i}" for i in range(len(components)))
    return AbelianSector(labels, components)


def _block_sparse_site_qn_maps(local_sectors, nsites, phys_dims, target_sector):
    labels, target_components, normalized_tables = _normalized_sector_tables(
        local_sectors,
        nsites,
        phys_dims,
        target_sector,
    )
    qn_maps = []
    for table in normalized_tables:
        qn_maps.append(
            {
                idx: _sector_qn_from_components(components, labels)
                for idx, components in enumerate(table)
            }
        )
    target_qn = _sector_qn_from_components(target_components, labels)
    return qn_maps, target_qn


def _block_sparse_uniform_phys_qns(site_qn_maps):
    phys_qns = [site_qn_maps[0][idx] for idx in sorted(site_qn_maps[0])]
    for site, qn_map in enumerate(site_qn_maps[1:], start=1):
        other = [qn_map[idx] for idx in sorted(qn_map)]
        if other != phys_qns:
            raise NotImplementedError(
                "block-sparse TDVP currently requires identical local sector tables on all sites; "
                f"site 0 and site {site} differ."
            )
    return phys_qns


def _block_sparse_phys_dims(psi):
    dims = []
    for site in range(psi.L):
        tensor = psi.factors[site]
        if hasattr(tensor, "qns"):
            dims.append(len(tensor.qns[2]))
        else:
            dims.append(int(np.asarray(psi._get_std_B(site)).shape[1]))
    return tuple(int(dim) for dim in dims)


def _as_block_sparse_factors(psi, site_qn_maps, *, copy=True):
    if not isinstance(psi, MPS):
        raise TypeError("block-sparse TDVP expects an MPS initial state.")
    if psi.factors and hasattr(psi.factors[0], "qns"):
        return [to_native_abelian_site_tensor(site, copy=copy) for site in psi.factors]
    phys_qns = _block_sparse_uniform_phys_qns(site_qn_maps)
    dense_factors = _standard_mps_factors(psi)
    return dense_to_symmetric(
        dense_factors,
        phys_qns=phys_qns,
        native_site_storage=True,
    )


def _site_qn_maps_signature(site_qn_maps):
    return tuple(
        tuple((int(idx), repr(qn)) for idx, qn in sorted(qn_map.items()))
        for qn_map in site_qn_maps
    )


def _stable_mpo_cache_key(H, factors):
    key = getattr(H, "_pyqed_cache_key", None)
    if key is None:
        key = tuple(id(factor) for factor in factors)
    try:
        hash(key)
    except TypeError:
        key = repr(key)
    return key


def _affine_mpo_metadata(H):
    return getattr(H, "_pyqed_affine_mpo", None)


def _affine_first_factor(meta):
    first_blocks = [meta["base_first"]] + [
        coeff * block for coeff, block in zip(meta["coeffs"], meta["term_first"])
    ]
    return np.concatenate(first_blocks, axis=1)


def _affine_first_site_template(meta, site_qn_map):
    first_blocks = [np.asarray(meta["base_first"])] + [
        np.asarray(block) for block in meta["term_first"]
    ]
    if not first_blocks:
        return None
    if any(block.ndim != 4 or block.shape[0] != 1 for block in first_blocks):
        return None
    phys_shape = first_blocks[0].shape[2:]
    if any(block.shape[2:] != phys_shape for block in first_blocks[1:]):
        return None

    first_val = list(site_qn_map.values())[0]
    q_left = zero_like_sector(first_val)
    phys_by_q = defaultdict(list)
    for state, qn in site_qn_map.items():
        phys_by_q[qn].append(int(state))
    phys_by_q = {qn: sorted(states) for qn, states in phys_by_q.items()}

    entries_by_component = []
    next_nodes = set()
    block_keys = set()
    right_offset = 0
    for component, block in enumerate(first_blocks):
        idxs = np.nonzero(np.abs(block) > 1.0e-12)
        entries = []
        for pos in range(len(idxs[0])):
            left_idx = int(idxs[0][pos])
            if left_idx != 0:
                return None
            right_idx = int(idxs[1][pos]) + right_offset
            out_s = int(idxs[2][pos])
            in_s = int(idxs[3][pos])
            q_out = site_qn_map[out_s]
            q_in = site_qn_map[in_s]
            try:
                q_right = q_left - (q_out - q_in)
            except TypeError:
                return None
            key = (q_left, q_right, q_out, q_in)
            value = block[0, int(idxs[1][pos]), out_s, in_s]
            entries.append((right_idx, out_s, in_s, key, value))
            next_nodes.add((right_idx, q_right))
            block_keys.add(key)
        entries_by_component.append(entries)
        right_offset += int(block.shape[1])

    if not next_nodes:
        return None
    r_map = {
        qn: sorted([node for node in next_nodes if node[1] == qn])
        for qn in set(qn for _, qn in next_nodes)
    }
    col_lookup = {
        node: idx for nodes in r_map.values() for idx, node in enumerate(nodes)
    }
    out_lookup = {
        qn: {state: idx for idx, state in enumerate(states)}
        for qn, states in phys_by_q.items()
    }

    block_shapes = {}
    for q_l, q_r, q_o, q_i in block_keys:
        block_shapes[(q_l, q_r, q_o, q_i)] = (
            1,
            len(r_map[q_r]),
            len(phys_by_q[q_o]),
            len(phys_by_q[q_i]),
        )

    component_data = []
    dtype = np.result_type(*first_blocks)
    ordered_keys = sorted(block_keys)
    for entries in entries_by_component:
        data = OrderedDict(
            (key, np.zeros(block_shapes[key], dtype=dtype)) for key in ordered_keys
        )
        for right_idx, out_s, in_s, key, value in entries:
            _q_l, q_r, q_o, q_i = key
            data[key][
                0,
                col_lookup[(right_idx, q_r)],
                out_lookup[q_o][out_s],
                out_lookup[q_i][in_s],
            ] = value
        component_data.append(data)

    qns = (
        (q_left,),
        tuple(sorted(r_map)),
        tuple(sorted(phys_by_q)),
        tuple(sorted(phys_by_q)),
    )
    return {
        "component_data": component_data,
        "qns": qns,
        "dirs": (-1, 1, 1, -1),
    }


def _affine_first_site_from_template(template, coeffs):
    all_coeffs = (1.0 + 0.0j, *tuple(coeffs))
    component_data = template["component_data"]
    if len(all_coeffs) != len(component_data):
        return None
    data = OrderedDict()
    for key in component_data[0]:
        block = np.zeros_like(
            component_data[0][key],
            dtype=np.result_type(*all_coeffs, component_data[0][key]),
        )
        for coeff, basis_data in zip(all_coeffs, component_data):
            if coeff != 0:
                block = block + coeff * basis_data[key]
        data[key] = block
    return AbelianSiteTensorData(data, template["qns"], template["dirs"], copy=False)


def _as_affine_block_sparse_mpo(H, site_qn_maps):
    meta = _affine_mpo_metadata(H)
    if meta is None:
        return None
    shared = meta.get("shared")
    if shared is None:
        return None

    key = (
        meta.get("cache_id", int(meta["template_id"])),
        _site_qn_maps_signature(site_qn_maps),
    )
    first = _affine_first_factor(meta)
    cached = _AFFINE_BLOCK_SPARSE_MPO_CACHE.get(key)
    if cached is None:
        dense_factors = [first] + [np.asarray(w) for w in shared[1:]]
        converted = dense_to_symmetric_mpo(
            dense_factors,
            site_qn_maps,
            native_site_storage=True,
        )
        _AFFINE_BLOCK_SPARSE_MPO_CACHE[key] = {
            "shared_tail": converted[1:],
            "site1_left_qns": tuple(converted[1].qns[0]) if len(converted) > 1 else (),
            "site0_template": _affine_first_site_template(meta, site_qn_maps[0]),
        }
        _AFFINE_BLOCK_SPARSE_MPO_CACHE.move_to_end(key)
        if len(_AFFINE_BLOCK_SPARSE_MPO_CACHE) > _AFFINE_BLOCK_SPARSE_MPO_CACHE_MAX:
            _AFFINE_BLOCK_SPARSE_MPO_CACHE.popitem(last=False)
        return converted

    template = cached.get("site0_template")
    site0 = (
        _affine_first_site_from_template(template, meta["coeffs"])
        if template is not None
        else None
    )
    if site0 is None:
        site0 = dense_to_symmetric_mpo(
            [first],
            site_qn_maps[:1],
            native_site_storage=True,
        )[0]
    if tuple(site0.qns[1]) != cached["site1_left_qns"]:
        dense_factors = [first] + [np.asarray(w) for w in shared[1:]]
        converted = dense_to_symmetric_mpo(
            dense_factors,
            site_qn_maps,
            native_site_storage=True,
        )
        cached["shared_tail"] = converted[1:]
        cached["site1_left_qns"] = (
            tuple(converted[1].qns[0]) if len(converted) > 1 else ()
        )
        cached["site0_template"] = _affine_first_site_template(meta, site_qn_maps[0])
        return converted
    _AFFINE_BLOCK_SPARSE_MPO_CACHE.move_to_end(key)
    return [site0] + list(cached["shared_tail"])


def _as_block_sparse_mpo(H, site_qn_maps):
    affine = _as_affine_block_sparse_mpo(H, site_qn_maps)
    if affine is not None:
        return affine
    factors = _mpo_factors(H)
    if factors and hasattr(factors[0], "qns"):
        return [to_native_abelian_site_tensor(site, copy=True) for site in factors]
    cache_key = (
        _stable_mpo_cache_key(H, factors),
        _site_qn_maps_signature(site_qn_maps),
    )
    cached = _BLOCK_SPARSE_MPO_CACHE.get(cache_key)
    if cached is not None:
        _BLOCK_SPARSE_MPO_CACHE.move_to_end(cache_key)
        return list(cached)
    converted = dense_to_symmetric_mpo(
        [np.asarray(w) for w in factors],
        site_qn_maps,
        native_site_storage=True,
    )
    _BLOCK_SPARSE_MPO_CACHE[cache_key] = tuple(converted)
    _BLOCK_SPARSE_MPO_CACHE.move_to_end(cache_key)
    if len(_BLOCK_SPARSE_MPO_CACHE) > _BLOCK_SPARSE_MPO_CACHE_MAX:
        _BLOCK_SPARSE_MPO_CACHE.popitem(last=False)
    return converted


def _block_mps_norm2(factors):
    if not factors:
        return 1.0
    identity = [make_identity_mpo_site_from_mps_site(site) for site in factors]
    env = initial_E(identity[0])
    for site, W in zip(factors, identity):
        env = contract_from_left(W, site, env, site)
    if not getattr(env, "data", None):
        return 0.0
    total = 0.0
    for block in env.data.values():
        total += np.asarray(block).reshape(-1).sum()
    total = np.real_if_close(total)
    return float(np.real(total))


def _right_canonical_block_mps_norm2(factors):
    if not factors:
        return 1.0
    return float(sum(np.vdot(block, block).real for block in factors[0].data.values()))


def _normalize_block_factors_inplace(factors, norm2=None):
    if norm2 is None:
        norm2 = _block_mps_norm2(factors)
    if norm2 <= 0.0:
        raise ValueError("Cannot normalize a zero-norm block-sparse MPS.")
    factors[0] = factors[0] * (1.0 / np.sqrt(norm2))
    return factors, norm2


def _build_block_right_envs(
    factors,
    mpo,
    target_qn,
    *,
    moving_environment=None,
    env_plan_prefix="",
):
    nsites = len(factors)
    right_envs = [None] * (nsites + 1)
    right_envs[nsites] = initial_F(mpo[-1], target_qn=target_qn)
    if (
        moving_environment is not None
        and hasattr(moving_environment, "sweep_environment_step_auto")
        and env_plan_prefix
    ):
        stack = [right_envs[nsites]]
        update_rows = [
            (
                f"{env_plan_prefix}:right-build:{i}",
                mpo[i],
                factors[i],
                factors[i],
                stack,
            )
            for i in range(nsites - 1, -1, -1)
        ]
        try:
            moving_environment.sweep_environment_step_auto(
                "right",
                AbelianEnvironmentTensorData,
                update_rows,
                [],
            )
            if len(stack) == nsites + 1:
                for offset, env in enumerate(stack):
                    right_envs[nsites - offset] = env
                return right_envs
        except Exception:
            pass

    for i in range(nsites - 1, -1, -1):
        plan_key = f"{env_plan_prefix}:right-build:{i}" if env_plan_prefix else None
        right_envs[i] = _advance_block_environment(
            "right",
            mpo[i],
            factors[i],
            right_envs[i + 1],
            factors[i],
            moving_environment=moving_environment,
            plan_key=plan_key,
        )
    return right_envs


def _new_cpp_moving_environment():
    owner_cls = _cpp_table_kernel("MovingEnvironment")
    if owner_cls is None:
        return None
    try:
        return owner_cls()
    except Exception:
        return None


def _moving_environment_stats(moving_environment):
    if moving_environment is None or not hasattr(moving_environment, "stats"):
        return {}
    try:
        return dict(moving_environment.stats())
    except Exception:
        return {}


def _moving_environment_delta_info(moving_environment, before):
    info = {"cpp_moving_environment": moving_environment is not None}
    after = _moving_environment_stats(moving_environment)
    if not after:
        return info
    before = before or {}
    for key in _BLOCK_MOVING_ENV_COUNTERS:
        value = after.get(key)
        if value is not None:
            info[f"cpp_{key}"] = int(value) - int(before.get(key, 0))
    for key in _BLOCK_MOVING_ENV_TIMERS:
        value = after.get(key)
        if value is not None:
            info[f"cpp_{key}"] = float(value) - float(before.get(key, 0.0))
    for key in _BLOCK_MOVING_ENV_ABSOLUTE:
        value = after.get(key)
        if value is not None:
            info[f"cpp_{key}"] = int(value)
    return info


def _block_environment_advance_signature(direction, W, A, env, B):
    def route_signature(tensor):
        data = getattr(tensor, "data", None)
        if data is None:
            return None
        return (
            tuple(getattr(tensor, "dirs", ())),
            tuple(tuple(key) for key in data.keys()),
        )

    signatures = tuple(route_signature(tensor) for tensor in (W, A, env, B))
    if any(signature is None for signature in signatures):
        return None
    return repr((str(direction), signatures))


def _advance_block_environment(
    direction,
    W,
    A,
    env,
    B,
    *,
    moving_environment=None,
    plan_key=None,
):
    direction = str(direction).lower()
    if direction not in {"left", "right"}:
        raise ValueError(
            "Block environment advance direction must be 'left' or 'right'."
        )

    if moving_environment is not None:
        if plan_key is None:
            plan_key = f"tdvp-block-environment:{direction}"
        if hasattr(moving_environment, "environment_advance_auto"):
            try:
                payload = moving_environment.environment_advance_auto(
                    str(plan_key),
                    direction,
                    W,
                    A,
                    env,
                    B,
                )
                return _cpp_payload_to_abelian_tensor(
                    payload,
                    carrier=AbelianEnvironmentTensorData,
                )
            except Exception:
                pass
        signature = _block_environment_advance_signature(direction, W, A, env, B)
        if signature is not None:
            try:
                payload = moving_environment.environment_advance(
                    str(plan_key),
                    direction,
                    W,
                    A,
                    env,
                    B,
                    signature,
                )
                return _cpp_payload_to_abelian_tensor(
                    payload,
                    carrier=AbelianEnvironmentTensorData,
                )
            except Exception:
                pass

    if direction == "left":
        return contract_from_left(W, A, env, B)
    return contract_from_right(W, A, env, B)


def _cpp_one_site_tdvp_sweep(
    factors,
    mpo,
    target_qn,
    dt,
    *,
    moving_environment=None,
    env_plan_prefix="tdvp-block",
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
):
    if not bool(_BLOCK_ONE_SITE_CPP_ENGINE):
        return None
    if moving_environment is None or not hasattr(
        moving_environment, "one_site_tdvp_sweep"
    ):
        return None

    krylov_key = str(krylov_method).lower().replace("_", "-")
    if krylov_key in {"lanczos", "hermitian", "hermitian-lanczos"}:
        callbacks = {}
    else:

        def evolve_site(theta, left, W, right, local_dt):
            return _evolve_block_site(
                theta,
                left,
                W,
                right,
                float(local_dt),
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
            )

        def evolve_bond(center, left, right, local_dt):
            return _evolve_block_bond(
                center,
                left,
                right,
                float(local_dt),
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
            )

        callbacks = {
            "evolve_site": evolve_site,
            "evolve_bond": evolve_bond,
            "left_qr": _block_left_qr,
            "right_rq": _block_right_rq,
            "absorb_center_left": _block_absorb_center_left,
            "absorb_center_right": _block_absorb_center_right,
        }
    try:
        out_factors, info = moving_environment.one_site_tdvp_sweep(
            list(factors),
            list(mpo),
            initial_E(mpo[0]),
            initial_F(mpo[-1], target_qn=target_qn),
            float(dt),
            AbelianEnvironmentTensorData,
            callbacks,
            int(krylov_dim),
            float(krylov_tol),
            str(krylov_method),
            str(env_plan_prefix or "tdvp-block"),
        )
    except Exception:
        return None
    info = dict(info)
    info.setdefault("cpp_one_site_engine", True)
    return list(out_factors), info


def _cpp_two_site_tdvp_sweep(
    factors,
    mpo,
    target_qn,
    dt,
    *,
    max_bond=None,
    cutoff=0.0,
    moving_environment=None,
    env_plan_prefix="tdvp2-block",
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
):
    if not bool(_BLOCK_TWO_SITE_CPP_ENGINE):
        return None
    if moving_environment is None or not hasattr(
        moving_environment, "two_site_tdvp_sweep"
    ):
        return None
    krylov_key = str(krylov_method).lower().replace("_", "-")
    if krylov_key not in {"lanczos", "hermitian", "hermitian-lanczos"}:
        return None
    try:
        out_factors, info = moving_environment.two_site_tdvp_sweep(
            list(factors),
            list(mpo),
            initial_E(mpo[0]),
            initial_F(mpo[-1], target_qn=target_qn),
            float(dt),
            AbelianEnvironmentTensorData,
            max_bond,
            float(cutoff),
            int(krylov_dim),
            float(krylov_tol),
            str(krylov_method),
            str(env_plan_prefix or "tdvp2-block"),
        )
    except Exception:
        return None
    info = dict(info)
    info.setdefault("cpp_two_site_engine", True)
    return list(out_factors), info


def _python_one_site_tdvp_sweep(
    factors,
    mpo,
    target_qn,
    dt,
    *,
    moving_environment=None,
    env_plan_prefix="tdvp-block",
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
):
    nsites = len(factors)
    if nsites == 1:
        factors[0] = _evolve_block_site(
            factors[0],
            initial_E(mpo[0]),
            mpo[0],
            initial_F(mpo[0], target_qn=target_qn),
            dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        return factors

    half_dt = 0.5 * dt
    right_envs = _build_block_right_envs(
        factors,
        mpo,
        target_qn,
        moving_environment=moving_environment,
        env_plan_prefix=env_plan_prefix,
    )
    left_envs = [None] * nsites
    left_envs[0] = initial_E(mpo[0])

    left = left_envs[0]
    for i in range(nsites - 1):
        factors[i] = _evolve_block_site(
            factors[i],
            left,
            mpo[i],
            right_envs[i + 1],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        q, center = _block_left_qr(factors[i])
        factors[i] = q
        left = _advance_block_environment(
            "left",
            mpo[i],
            q,
            left,
            q,
            moving_environment=moving_environment,
            plan_key=f"{env_plan_prefix}:left-sweep:{i}",
        )
        left_envs[i + 1] = left
        center = _evolve_block_bond(
            center,
            left,
            right_envs[i + 1],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        factors[i + 1] = _block_absorb_center_left(center, factors[i + 1])

    factors[-1] = _evolve_block_site(
        factors[-1],
        left_envs[-1],
        mpo[-1],
        initial_F(mpo[-1], target_qn=target_qn),
        half_dt,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        krylov_method=krylov_method,
    )

    right = initial_F(mpo[-1], target_qn=target_qn)
    for i in range(nsites - 1, 0, -1):
        factors[i] = _evolve_block_site(
            factors[i],
            left_envs[i],
            mpo[i],
            right,
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        center, q = _block_right_rq(factors[i])
        factors[i] = q
        right = _advance_block_environment(
            "right",
            mpo[i],
            q,
            right,
            q,
            moving_environment=moving_environment,
            plan_key=f"{env_plan_prefix}:right-sweep:{i}",
        )
        center = _evolve_block_bond(
            center,
            left_envs[i],
            right,
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        factors[i - 1] = _block_absorb_center_right(factors[i - 1], center)

    factors[0] = _evolve_block_site(
        factors[0],
        initial_E(mpo[0]),
        mpo[0],
        right,
        half_dt,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        krylov_method=krylov_method,
    )
    return factors


def _cpp_payload_to_abelian_tensor(payload, carrier=AbelianSiteTensorData):
    keys, blocks, qns, dirs = payload
    if carrier is AbelianSiteTensorData:
        out = AbelianSiteTensorData.__new__(AbelianSiteTensorData)
        out.data = OrderedDict(
            (tuple(key), np.asarray(block))
            for key, block in zip(tuple(keys), tuple(blocks))
        )
        out.qns = tuple(tuple(axis_qns) for axis_qns in (qns or ()))
        out.dirs = tuple(int(d) for d in (dirs or ()))
        out._layout_signature = None
        return out
    data = {
        tuple(key): np.asarray(block) for key, block in zip(tuple(keys), tuple(blocks))
    }
    return carrier(data, qns, dirs, copy=False)


def _cpp_lapack_qr(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.size < _BLOCK_QR_CPP_MIN_ELEMENTS:
        return None
    kernel = _cpp_table_kernel("lapack_qr")
    if kernel is None:
        return None
    try:
        q, r = kernel(matrix)
    except Exception:
        return None
    return np.asarray(q, dtype=complex), np.asarray(r, dtype=complex)


def _abelian_block_layout_signature(tensor):
    data = getattr(tensor, "data", None)
    if data is None:
        return None
    cached = getattr(tensor, "_layout_signature", None)
    if cached is not None:
        return cached
    signature = (
        tuple(getattr(tensor, "dirs", ())),
        tuple(
            (tuple(key), tuple(int(dim) for dim in np.asarray(block).shape))
            for key, block in data.items()
        ),
    )
    try:
        tensor._layout_signature = signature
    except Exception:
        pass
    return signature


def _cached_block_heff_plan(kind, *tensors):
    signatures = tuple(_abelian_block_layout_signature(tensor) for tensor in tensors)
    return _cached_block_heff_plan_for_signatures(kind, signatures, *tensors)


def _cached_block_heff_plan_for_signatures(kind, signatures, *tensors):
    class_name = {
        "site": "AbelianTDVPSiteHeffPlan",
        "bond": "AbelianTDVPBondHeffPlan",
    }[kind]
    plan_cls = _cpp_table_kernel(class_name)
    if plan_cls is None:
        return None
    if any(signature is None for signature in signatures):
        return None
    key = (kind, signatures)
    plan = _BLOCK_HEFF_PLAN_CACHE.get(key)
    if plan is not None:
        _BLOCK_HEFF_PLAN_CACHE.move_to_end(key)
        return plan
    plan = plan_cls.from_tensors(*tensors)
    _BLOCK_HEFF_PLAN_CACHE[key] = plan
    if len(_BLOCK_HEFF_PLAN_CACHE) > _BLOCK_HEFF_PLAN_CACHE_MAX:
        _BLOCK_HEFF_PLAN_CACHE.popitem(last=False)
    return plan


def _block_count(tensor):
    data = getattr(tensor, "data", None)
    return len(data) if data is not None else 0


def _block_heff_route_estimate(*tensors):
    estimate = 1
    for tensor in tensors:
        estimate *= max(1, int(_block_count(tensor)))
    return int(estimate)


def _should_try_block_heff_cpp(*tensors):
    return _block_heff_route_estimate(*tensors) <= _BLOCK_HEFF_CPP_MAX_ROUTE_ESTIMATE


def _should_try_block_heff_plan(*tensors):
    estimate = _block_heff_route_estimate(*tensors)
    return (
        _BLOCK_HEFF_PLAN_MIN_ROUTE_ESTIMATE
        <= estimate
        <= _BLOCK_HEFF_CPP_MAX_ROUTE_ESTIMATE
    )


def _block_tensor_max_abs_diff(a, b):
    data_a = getattr(a, "data", None)
    data_b = getattr(b, "data", None)
    if data_a is None or data_b is None or set(data_a) != set(data_b):
        return np.inf
    max_diff = 0.0
    for key, block_a in data_a.items():
        block_b = data_b[key]
        if np.asarray(block_a).shape != np.asarray(block_b).shape:
            return np.inf
        if np.asarray(block_a).size:
            diff = float(np.max(np.abs(np.asarray(block_a) - np.asarray(block_b))))
            max_diff = max(max_diff, diff)
    return max_diff


def _cache_block_heff_backend_decision(key, value):
    _BLOCK_HEFF_BACKEND_DECISION_CACHE[key] = value
    _BLOCK_HEFF_BACKEND_DECISION_CACHE.move_to_end(key)
    if len(_BLOCK_HEFF_BACKEND_DECISION_CACHE) > _BLOCK_HEFF_BACKEND_DECISION_CACHE_MAX:
        _BLOCK_HEFF_BACKEND_DECISION_CACHE.popitem(last=False)


def _apply_block_site_heff_python(theta, left, W, right):
    tmp = abelian_tensor_data_tensordot(left, theta, ([2], [0]))
    tmp = abelian_tensor_data_tensordot(tmp, W, ([0, 3], [0, 3]))
    tmp = abelian_tensor_data_tensordot(tmp, right, ([2, 1], [0, 2]))
    return abelian_transpose_tensor_data(
        tmp,
        (0, 2, 1),
        carrier=AbelianSiteTensorData,
    )


def _apply_block_site_heff(theta, left, W, right):
    if _should_try_block_heff_cpp(theta, left, W, right):
        if _should_try_block_heff_plan(theta, left, W, right):
            plan = _cached_block_heff_plan("site", theta, left, W, right)
            if plan is not None:
                try:
                    return _cpp_payload_to_abelian_tensor(
                        plan.apply(theta, left, W, right)
                    )
                except Exception:
                    pass
        kernel = _cpp_table_kernel("abelian_tdvp_site_heff_data")
        if kernel is not None:
            try:
                return _cpp_payload_to_abelian_tensor(kernel(theta, left, W, right))
            except Exception:
                pass
    return _apply_block_site_heff_python(theta, left, W, right)


def _apply_block_two_site_heff(theta, left, W_left, W_right, right):
    """Apply a two-site effective Hamiltonian to native Abelian blocks."""

    tmp = abelian_tensor_data_tensordot(left, theta, ([2], [0]))
    tmp = abelian_tensor_data_tensordot(tmp, W_left, ([0, 3], [0, 3]))
    tmp = abelian_tensor_data_tensordot(tmp, W_right, ([3, 2], [0, 3]))
    tmp = abelian_tensor_data_tensordot(tmp, right, ([3, 1], [0, 2]))
    return abelian_transpose_tensor_data(
        tmp,
        (0, 3, 1, 2),
        carrier=AbelianSiteTensorData,
    )


def _apply_block_bond_heff(center, left, right):
    if _should_try_block_heff_cpp(center, left, right):
        if _should_try_block_heff_plan(center, left, right):
            plan = _cached_block_heff_plan("bond", center, left, right)
            if plan is not None:
                try:
                    return _cpp_payload_to_abelian_tensor(
                        plan.apply(center, left, right)
                    )
                except Exception:
                    pass
        kernel = _cpp_table_kernel("abelian_tdvp_bond_heff_data")
        if kernel is not None:
            try:
                return _cpp_payload_to_abelian_tensor(kernel(center, left, right))
            except Exception:
                pass
    tmp = abelian_tensor_data_tensordot(left, center, ([2], [0]))
    return abelian_tensor_data_tensordot(tmp, right, ([0, 2], [0, 2]))


def _make_planned_block_site_heff(theta, left, W, right):
    route_estimate = _block_heff_route_estimate(theta, left, W, right)
    if route_estimate > _BLOCK_HEFF_AUTOTUNE_MAX_ROUTE_ESTIMATE:
        return lambda local: _apply_block_site_heff(local, left, W, right)
    expected_signature = _abelian_block_layout_signature(theta)
    fixed_signatures = (
        _abelian_block_layout_signature(left),
        _abelian_block_layout_signature(W),
        _abelian_block_layout_signature(right),
    )
    plan = _cached_block_heff_plan_for_signatures(
        "site",
        (expected_signature, *fixed_signatures),
        theta,
        left,
        W,
        right,
    )
    if plan is None:
        return lambda local: _apply_block_site_heff(local, left, W, right)
    backend_key = ("site", expected_signature, *fixed_signatures)
    local_plan_cache = {expected_signature: plan}

    preferred = "cpp" if route_estimate <= _BLOCK_HEFF_CPP_MAX_ROUTE_ESTIMATE else None

    def apply(local):
        local_signature = _abelian_block_layout_signature(local)
        if local_signature == expected_signature:
            decision = preferred
            if decision is None:
                decision = _BLOCK_HEFF_BACKEND_DECISION_CACHE.get(backend_key)
            if decision is None:
                try:
                    start = time.perf_counter()
                    python_out = _apply_block_site_heff_python(local, left, W, right)
                    python_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    cpp_out = _cpp_payload_to_abelian_tensor(
                        plan.apply(local, left, W, right)
                    )
                    cpp_seconds = time.perf_counter() - start
                    if _block_tensor_max_abs_diff(python_out, cpp_out) <= 1.0e-9:
                        decision = "cpp" if cpp_seconds < python_seconds else "python"
                    else:
                        decision = "python"
                    _cache_block_heff_backend_decision(backend_key, decision)
                    return cpp_out if decision == "cpp" else python_out
                except Exception:
                    _cache_block_heff_backend_decision(backend_key, "python")
                    return _apply_block_site_heff_python(local, left, W, right)
            if decision == "cpp":
                try:
                    return _cpp_payload_to_abelian_tensor(
                        plan.apply(local, left, W, right)
                    )
                except Exception:
                    _cache_block_heff_backend_decision(backend_key, "python")
            return _apply_block_site_heff_python(local, left, W, right)

        if _should_try_block_heff_cpp(local, left, W, right):
            local_plan = local_plan_cache.get(local_signature)
            if local_plan is None:
                local_plan = _cached_block_heff_plan_for_signatures(
                    "site",
                    (local_signature, *fixed_signatures),
                    local,
                    left,
                    W,
                    right,
                )
                local_plan_cache[local_signature] = local_plan
            if local_plan is not None:
                try:
                    return _cpp_payload_to_abelian_tensor(
                        local_plan.apply(local, left, W, right)
                    )
                except Exception:
                    pass
            kernel = _cpp_table_kernel("abelian_tdvp_site_heff_data")
            if kernel is not None:
                try:
                    return _cpp_payload_to_abelian_tensor(kernel(local, left, W, right))
                except Exception:
                    pass
        return _apply_block_site_heff_python(local, left, W, right)

    return apply


def _make_planned_block_bond_heff(center, left, right):
    if not _should_try_block_heff_cpp(
        center, left, right
    ) or not _should_try_block_heff_plan(center, left, right):
        return lambda local: _apply_block_bond_heff(local, left, right)
    expected_signature = _abelian_block_layout_signature(center)
    fixed_signatures = (
        _abelian_block_layout_signature(left),
        _abelian_block_layout_signature(right),
    )
    plan = _cached_block_heff_plan_for_signatures(
        "bond",
        (expected_signature, *fixed_signatures),
        center,
        left,
        right,
    )
    if plan is None:
        return lambda local: _apply_block_bond_heff(local, left, right)

    def apply(local):
        if _abelian_block_layout_signature(local) == expected_signature:
            try:
                return _cpp_payload_to_abelian_tensor(plan.apply(local, left, right))
            except Exception:
                pass
        return _apply_block_bond_heff(local, left, right)

    return apply


def _block_linear_combination(coeffs, basis):
    if basis and all(isinstance(vec, AbelianSiteTensorData) for vec in basis):
        out = None
        qns = basis[0].qns
        dirs = basis[0].dirs
        for coeff, vec in zip(coeffs, basis):
            if abs(coeff) <= 0.0:
                continue
            if vec.qns != qns or vec.dirs != dirs:
                out = None
                break
            if out is None:
                out = OrderedDict(
                    (key, np.asarray(block) * coeff) for key, block in vec.data.items()
                )
                continue
            for key, block in vec.data.items():
                contrib = np.asarray(block) * coeff
                old = out.get(key)
                out[key] = contrib if old is None else old + contrib
        if out is not None:
            if not out:
                return basis[0] * 0.0
            return AbelianSiteTensorData(out, qns, dirs, copy=False)

    out = None
    for coeff, vec in zip(coeffs, basis):
        if abs(coeff) <= 0.0:
            continue
        term = vec * coeff
        out = term if out is None else out + term
    if out is None:
        return basis[0] * 0.0
    return out


def _block_krylov_expm_apply(
    vec,
    apply_heff,
    dt,
    *,
    krylov_dim=12,
    tol=1.0e-13,
    method="lanczos",
):
    norm = vec.norm()
    if norm <= tol:
        return vec.copy()

    key = str(method).lower().replace("_", "-")
    mmax = max(1, int(krylov_dim))
    if key in {"lanczos", "hermitian", "hermitian-lanczos"}:
        basis = [vec * (1.0 / norm)]
        alpha = np.zeros(mmax, dtype=float)
        beta = np.zeros(max(mmax - 1, 0), dtype=float)
        q_prev = None
        beta_prev = 0.0
        actual_dim = 1
        previous_coeffs = None
        coeffs = None
        for j in range(mmax):
            q = basis[j]
            trial = apply_heff(q)
            if q_prev is not None:
                trial = trial - q_prev * beta_prev
            alpha_j = q.dot(trial)
            alpha[j] = float(np.real(alpha_j))
            trial = trial - q * alpha_j
            beta_j = trial.norm()
            actual_dim = j + 1
            if actual_dim == 1:
                current_coeffs = np.array(
                    [norm * np.exp(-1j * dt * alpha[0])],
                    dtype=complex,
                )
            else:
                evals, evecs = eigh_tridiagonal(
                    alpha[:actual_dim],
                    beta[: actual_dim - 1],
                )
                e1 = np.zeros(actual_dim, dtype=complex)
                e1[0] = norm
                current_coeffs = evecs @ (
                    np.exp(-1j * dt * evals) * (evecs.T.conj() @ e1)
                )
            if previous_coeffs is not None:
                delta2 = np.sum(np.abs(current_coeffs[:-1] - previous_coeffs) ** 2)
                delta2 += abs(current_coeffs[-1]) ** 2
                if np.sqrt(delta2) <= tol * max(1.0, norm):
                    coeffs = current_coeffs
                    break
            previous_coeffs = current_coeffs
            if beta_j <= tol or j + 1 == mmax:
                break
            beta[j] = beta_j
            q_prev = q
            beta_prev = beta_j
            basis.append(trial * (1.0 / beta_j))

        if coeffs is None:
            coeffs = current_coeffs
        return _block_linear_combination(coeffs, basis[:actual_dim])

    if key not in {"arnoldi", "generic"}:
        raise ValueError("krylov_method must be 'lanczos' or 'arnoldi'.")

    basis = [vec * (1.0 / norm)]
    h_krylov = np.zeros((mmax, mmax), dtype=complex)
    actual_dim = 1
    for j in range(mmax):
        trial = apply_heff(basis[j])
        for i in range(j + 1):
            coeff = basis[i].dot(trial)
            h_krylov[i, j] += coeff
            trial = trial - basis[i] * coeff
        for i in range(j + 1):
            coeff = basis[i].dot(trial)
            h_krylov[i, j] += coeff
            trial = trial - basis[i] * coeff
        beta = trial.norm()
        actual_dim = j + 1
        if beta <= tol or j + 1 == mmax:
            break
        h_krylov[j + 1, j] = beta
        basis.append(trial * (1.0 / beta))

    h_small = h_krylov[:actual_dim, :actual_dim]
    e1 = np.zeros(actual_dim, dtype=complex)
    e1[0] = norm
    coeffs = expm(-1j * dt * h_small) @ e1
    return _block_linear_combination(coeffs, basis[:actual_dim])


def _evolve_block_site(
    theta,
    left,
    W,
    right,
    dt,
    *,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
):
    apply_heff = _make_planned_block_site_heff(theta, left, W, right)
    return _block_krylov_expm_apply(
        theta,
        apply_heff,
        dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _evolve_block_two_site(
    theta,
    left,
    W_left,
    W_right,
    right,
    dt,
    *,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
):
    def apply_heff(local):
        return _apply_block_two_site_heff(
            local,
            left,
            W_left,
            W_right,
            right,
        )

    return _block_krylov_expm_apply(
        theta,
        apply_heff,
        dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _evolve_block_bond(
    center,
    left,
    right,
    dt,
    *,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
):
    apply_heff = _make_planned_block_bond_heff(center, left, right)
    return _block_krylov_expm_apply(
        center,
        apply_heff,
        -dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _apply_block_two_site_heff(theta, left, W_left, W_right, right):
    tmp = abelian_tensor_data_tensordot(left, theta, ([2], [0]))
    tmp = abelian_tensor_data_tensordot(tmp, W_left, ([0, 3], [0, 3]))
    tmp = abelian_tensor_data_tensordot(tmp, W_right, ([3, 2], [0, 3]))
    tmp = abelian_tensor_data_tensordot(tmp, right, ([3, 1], [0, 2]))
    return abelian_transpose_tensor_data(
        tmp,
        (0, 3, 1, 2),
        carrier=AbelianSiteTensorData,
    )


def _evolve_block_two_site(
    theta,
    left,
    W_left,
    W_right,
    right,
    dt,
    *,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
):
    theta_elements = sum(
        int(np.asarray(block).size)
        for block in (getattr(theta, "data", {}) or {}).values()
    )
    if _is_lanczos_method(krylov_method) and theta_elements >= 128:
        kernel = _cpp_table_kernel("abelian_tdvp_two_site_lanczos")
        if kernel is not None:
            try:
                return kernel(
                    theta,
                    left,
                    W_left,
                    W_right,
                    right,
                    float(dt),
                    int(krylov_dim),
                    float(krylov_tol),
                    AbelianSiteTensorData,
                )
            except Exception:
                pass

    def apply_heff(local):
        return _apply_block_two_site_heff(
            local,
            left,
            W_left,
            W_right,
            right,
        )

    return _block_krylov_expm_apply(
        theta,
        apply_heff,
        dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _block_left_qr(theta):
    data_q = {}
    data_center = {}
    new_right_qns = []
    dtype = np.result_type(
        *[np.asarray(block).dtype for block in theta.data.values()],
        complex,
    )
    by_right = {}
    for key, block in theta.data.items():
        by_right.setdefault(key[1], []).append((key, np.asarray(block)))

    for q_right, entries in by_right.items():
        rows = []
        cols = None
        for key, block in entries:
            left_dim, right_dim, phys_dim = block.shape
            if cols is None:
                cols = right_dim
            elif cols != right_dim:
                raise ValueError("Inconsistent right-sector degeneracy in block QR.")
            rows.append((key, left_dim, phys_dim))
        if cols is None:
            continue
        mat = np.zeros(
            (sum(left_dim * phys_dim for _, left_dim, phys_dim in rows), cols),
            dtype=dtype,
        )
        offset = 0
        for key, left_dim, phys_dim in rows:
            block = theta.data[key]
            size = left_dim * phys_dim
            mat[offset : offset + size] = (
                np.asarray(block).transpose(0, 2, 1).reshape(size, cols)
            )
            offset += size
        qr = _cpp_lapack_qr(mat)
        if qr is None:
            q_mat, r_mat = np.linalg.qr(mat, mode="reduced")
        else:
            q_mat, r_mat = qr
        chi = int(q_mat.shape[1])
        if chi == 0:
            continue
        new_right_qns.append(q_right)
        data_center[(q_right, q_right)] = r_mat
        offset = 0
        for key, left_dim, phys_dim in rows:
            size = left_dim * phys_dim
            q_block = (
                q_mat[offset : offset + size]
                .reshape(left_dim, phys_dim, chi)
                .transpose(0, 2, 1)
            )
            data_q[(key[0], q_right, key[2])] = q_block
            offset += size

    q_tensor = AbelianSiteTensorData(
        data_q,
        [list(theta.qns[0]), new_right_qns, list(theta.qns[2])],
        theta.dirs,
        copy=False,
    )
    center = AbelianSiteTensorData(
        data_center,
        [new_right_qns, list(theta.qns[1])],
        [-1, 1],
        copy=False,
    )
    return q_tensor, center


def _block_right_rq(theta):
    data_q = {}
    data_center = {}
    new_left_qns = []
    dtype = np.result_type(
        *[np.asarray(block).dtype for block in theta.data.values()],
        complex,
    )
    by_left = {}
    for key, block in theta.data.items():
        by_left.setdefault(key[0], []).append((key, np.asarray(block)))

    for q_left, entries in by_left.items():
        cols = []
        left_dim = None
        for key, block in entries:
            block_left_dim, right_dim, phys_dim = block.shape
            if left_dim is None:
                left_dim = block_left_dim
            elif left_dim != block_left_dim:
                raise ValueError("Inconsistent left-sector degeneracy in block RQ.")
            cols.append((key, right_dim, phys_dim))
        if left_dim is None:
            continue
        mat = np.zeros(
            (left_dim, sum(right_dim * phys_dim for _, right_dim, phys_dim in cols)),
            dtype=dtype,
        )
        offset = 0
        for key, right_dim, phys_dim in cols:
            block = theta.data[key]
            size = right_dim * phys_dim
            mat[:, offset : offset + size] = (
                np.asarray(block).transpose(0, 2, 1).reshape(left_dim, size)
            )
            offset += size
        qr = _cpp_lapack_qr(mat.T)
        if qr is None:
            q_t, r_t = np.linalg.qr(mat.T, mode="reduced")
        else:
            q_t, r_t = qr
        chi = int(q_t.shape[1])
        if chi == 0:
            continue
        center = r_t.T
        q_mat = q_t.T
        new_left_qns.append(q_left)
        data_center[(q_left, q_left)] = center
        offset = 0
        for key, right_dim, phys_dim in cols:
            size = right_dim * phys_dim
            q_block = (
                q_mat[:, offset : offset + size]
                .reshape(chi, phys_dim, right_dim)
                .transpose(0, 2, 1)
            )
            data_q[(q_left, key[1], key[2])] = q_block
            offset += size

    center = AbelianSiteTensorData(
        data_center,
        [list(theta.qns[0]), new_left_qns],
        [-1, 1],
        copy=False,
    )
    q_tensor = AbelianSiteTensorData(
        data_q,
        [new_left_qns, list(theta.qns[1]), list(theta.qns[2])],
        theta.dirs,
        copy=False,
    )
    return center, q_tensor


def _block_absorb_center_left(center, right_site):
    return abelian_tensor_data_tensordot(center, right_site, ([1], [0]))


def _block_absorb_center_right(left_site, center):
    tmp = abelian_tensor_data_tensordot(left_site, center, ([1], [0]))
    return abelian_transpose_tensor_data(
        tmp,
        (0, 2, 1),
        carrier=AbelianSiteTensorData,
    )


def _block_right_canonicalize_qr(factors):
    """Right-canonicalize fixed-rank block MPS factors without SVD growth."""
    for site in range(len(factors) - 1, 0, -1):
        center, right = _block_right_rq(factors[site])
        factors[site] = right
        factors[site - 1] = _block_absorb_center_right(
            factors[site - 1],
            center,
        )
    return factors


def _physical_diagonal_blocks(W, *, cutoff=1.0e-14):
    W = np.asarray(W)
    if W.ndim != 4 or W.shape[2] != W.shape[3]:
        return None
    phys_dim = W.shape[2]
    offdiag = ~np.eye(phys_dim, dtype=bool)
    if np.any(np.abs(W[:, :, offdiag]) > cutoff):
        return None
    return np.diagonal(W, axis1=2, axis2=3)


def _apply_site_heff(theta, left, W, right):
    tmp = np.einsum("bqs,rns->bqrn", theta, right, optimize=True)
    tmp = np.einsum("mnpq,bqrn->bmpr", W, tmp, optimize=True)
    return np.einsum("amb,bmpr->apr", left, tmp, optimize=True)


def _evolve_site(
    theta,
    left,
    W,
    right,
    dt,
    *,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
    diagonal_fast_path=False,
):
    shape = theta.shape
    if (
        not diagonal_fast_path
        and _is_lanczos_method(krylov_method)
        and _cpp_tdvp_available()
    ):
        try:
            return _tdvp_cpp.site_lanczos(
                np.asarray(theta, dtype=complex),
                np.asarray(left, dtype=complex),
                np.asarray(W, dtype=complex),
                np.asarray(right, dtype=complex),
                float(dt),
                int(krylov_dim),
                float(krylov_tol),
            )
        except Exception:
            pass

    W_diag = _physical_diagonal_blocks(W) if diagonal_fast_path else None
    if W_diag is not None:
        left_kernels = [
            np.einsum("amb,mn->abn", left, W_diag[:, :, p], optimize=True)
            for p in range(shape[1])
        ]

        def apply_heff(local):
            out = np.zeros(shape, dtype=np.result_type(local, left, W, right, complex))
            for p, left_kernel in enumerate(left_kernels):
                out[:, p, :] = np.einsum(
                    "abn,bs,rns->ar",
                    left_kernel,
                    local[:, p, :],
                    right,
                    optimize=True,
                )
            return out
    else:
        left_kernel = np.einsum("amb,mnpq->abnpq", left, W, optimize=True)

        def apply_heff(local):
            tmp = np.tensordot(local, right, axes=([2], [2]))
            return np.tensordot(left_kernel, tmp, axes=([1, 2, 4], [0, 3, 1]))

    return _krylov_expm_apply(
        theta,
        shape,
        apply_heff,
        dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _apply_two_site_heff(theta, left, W_left, W_right, right):
    tmp = np.einsum("bqsd,cod->bqsco", theta, right, optimize=True)
    tmp = np.einsum("nors,bqsco->bqnrc", W_right, tmp, optimize=True)
    tmp = np.einsum("mnpq,bqnrc->bmprc", W_left, tmp, optimize=True)
    return np.einsum("amb,bmprc->aprc", left, tmp, optimize=True)


def _build_sparse_two_site_kernel(left, W_left, W_right, right, *, cutoff=1.0e-14):
    nshared = W_left.shape[1]
    d_left_out, d_left_in = W_left.shape[2], W_left.shape[3]
    d_right_out, d_right_in = W_right.shape[2], W_right.shape[3]
    kernels = []
    pair_count = 0

    for n in range(nshared):
        raw_left_by_q = [[] for _ in range(d_left_in)]
        for p in range(d_left_out):
            for q in range(d_left_in):
                coeff = W_left[:, n, p, q]
                if not np.any(np.abs(coeff) > cutoff):
                    continue
                block = np.einsum("amb,m->ab", left, coeff, optimize=True)
                if np.any(np.abs(block) > cutoff):
                    raw_left_by_q[q].append((p, block))

        raw_right_by_s = [[] for _ in range(d_right_in)]
        for r in range(d_right_out):
            for s in range(d_right_in):
                coeff = W_right[n, :, r, s]
                if not np.any(np.abs(coeff) > cutoff):
                    continue
                block = np.einsum("cod,o->cd", right, coeff, optimize=True)
                if np.any(np.abs(block) > cutoff):
                    raw_right_by_s[s].append((r, block))

        if not any(raw_left_by_q) or not any(raw_right_by_s):
            continue

        left_by_q = []
        for terms in raw_left_by_q:
            if terms:
                left_by_q.append(
                    (
                        np.asarray([p for p, _ in terms], dtype=int),
                        np.stack([block for _, block in terms], axis=0),
                    )
                )
            else:
                left_by_q.append(None)

        right_by_s = []
        for terms in raw_right_by_s:
            if terms:
                right_by_s.append(
                    (
                        np.asarray([r for r, _ in terms], dtype=int),
                        np.stack([block for _, block in terms], axis=0),
                    )
                )
            else:
                right_by_s.append(None)

        for left_terms in left_by_q:
            if left_terms is None:
                continue
            for right_terms in right_by_s:
                if right_terms is not None:
                    pair_count += left_terms[0].size * right_terms[0].size
        kernels.append((left_by_q, right_by_s))

    dense_pair_count = nshared * d_left_out * d_left_in * d_right_out * d_right_in
    return kernels, pair_count, dense_pair_count


def _estimate_sparse_two_site_pairs(W_left, W_right, *, cutoff=1.0e-14):
    nshared = W_left.shape[1]
    d_left_out, d_left_in = W_left.shape[2], W_left.shape[3]
    d_right_out, d_right_in = W_right.shape[2], W_right.shape[3]
    pair_count = 0
    for n in range(nshared):
        left_count = np.count_nonzero(np.any(np.abs(W_left[:, n]) > cutoff, axis=0))
        right_count = np.count_nonzero(np.any(np.abs(W_right[n]) > cutoff, axis=0))
        pair_count += int(left_count) * int(right_count)
    dense_pair_count = nshared * d_left_out * d_left_in * d_right_out * d_right_in
    return pair_count, dense_pair_count


def _apply_sparse_two_site_kernel(theta, kernels, shape):
    out = np.zeros(shape, dtype=np.result_type(theta, complex))
    for left_by_q, right_by_s in kernels:
        for q, left_terms in enumerate(left_by_q):
            if left_terms is None:
                continue
            p_indices, left_stack = left_terms
            for s, right_terms in enumerate(right_by_s):
                if right_terms is None:
                    continue
                r_indices, right_stack = right_terms
                local_block = theta[:, q, s, :]
                projected = np.einsum(
                    "xab,bd->xad", left_stack, local_block, optimize=True
                )
                contribution = np.einsum(
                    "xad,ycd->xayc", projected, right_stack, optimize=True
                )
                for ix, p in enumerate(p_indices):
                    for iy, r in enumerate(r_indices):
                        out[:, p, r, :] += contribution[ix, :, iy, :]
    return out


def _apply_sparse_two_site_kernel_vectorized(theta, kernels, shape):
    out = np.zeros(shape, dtype=np.result_type(theta, complex))
    for left_by_q, right_by_s in kernels:
        for q, left_terms in enumerate(left_by_q):
            if left_terms is None:
                continue
            p_indices, left_stack = left_terms
            for s, right_terms in enumerate(right_by_s):
                if right_terms is None:
                    continue
                r_indices, right_stack = right_terms
                local_block = theta[:, q, s, :]
                projected = np.einsum(
                    "xab,bd->xad", left_stack, local_block, optimize=True
                )
                contribution = np.einsum(
                    "xad,ycd->xayc", projected, right_stack, optimize=True
                )
                out[:, p_indices[:, None], r_indices[None, :], :] += (
                    contribution.transpose(1, 0, 2, 3)
                )
    return out


def _evolve_two_site(
    theta,
    left,
    W_left,
    W_right,
    right,
    dt,
    *,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
    diagonal_fast_path=False,
    sparse_threshold=0.0,
    sparse_vectorized=True,
    dense_workspace=None,
    workspace_key=None,
):
    global _dense_tdvp_cpp_last_error
    shape = theta.shape
    if (
        dense_workspace is not None
        and not diagonal_fast_path
        and float(sparse_threshold) <= 0.0
        and _is_lanczos_method(krylov_method)
        and _dense_mpo_pair_for_workspace(W_left, W_right)
    ):
        try:
            evolved = dense_workspace.evolve_two_site(
                str(workspace_key),
                np.asarray(left.transpose(1, 0, 2), dtype=np.complex128),
                np.asarray(W_left, dtype=np.complex128),
                np.asarray(W_right, dtype=np.complex128),
                np.asarray(right.transpose(1, 0, 2), dtype=np.complex128),
                np.asarray(theta, dtype=np.complex128),
                float(dt),
                int(krylov_dim),
                float(krylov_tol),
                "blas",
                True,
            )
            return np.asarray(evolved, dtype=complex).reshape(shape)
        except Exception as exc:
            _dense_tdvp_cpp_last_error = str(exc)
            pass
    if (
        not diagonal_fast_path
        and float(sparse_threshold) <= 0.0
        and _is_lanczos_method(krylov_method)
        and _cpp_tdvp_available()
    ):
        try:
            return _tdvp_cpp.two_site_lanczos(
                np.asarray(theta, dtype=complex),
                np.asarray(left, dtype=complex),
                np.asarray(W_left, dtype=complex),
                np.asarray(W_right, dtype=complex),
                np.asarray(right, dtype=complex),
                float(dt),
                int(krylov_dim),
                float(krylov_tol),
            )
        except Exception:
            pass

    W_left_diag = _physical_diagonal_blocks(W_left) if diagonal_fast_path else None
    W_right_diag = _physical_diagonal_blocks(W_right) if diagonal_fast_path else None
    if W_left_diag is not None and W_right_diag is not None:
        left_kernels = [
            np.einsum("amb,mn->abn", left, W_left_diag[:, :, p], optimize=True)
            for p in range(shape[1])
        ]
        right_kernels = [
            np.einsum("no,cod->ncd", W_right_diag[:, :, r], right, optimize=True)
            for r in range(shape[2])
        ]

        def apply_heff(local):
            out = np.zeros(
                shape,
                dtype=np.result_type(local, left, W_left, W_right, right, complex),
            )
            for p, left_kernel in enumerate(left_kernels):
                for r, right_kernel in enumerate(right_kernels):
                    out[:, p, r, :] = np.einsum(
                        "abn,bd,ncd->ac",
                        left_kernel,
                        local[:, p, r, :],
                        right_kernel,
                        optimize=True,
                    )
            return out
    else:
        estimated_sparse_pairs, dense_pairs = _estimate_sparse_two_site_pairs(
            W_left, W_right
        )
        sparse_kernel = None
        threshold = float(sparse_threshold)
        if threshold > 0.0 and estimated_sparse_pairs <= threshold * dense_pairs:
            sparse_kernel, sparse_pairs, dense_pairs = _build_sparse_two_site_kernel(
                left,
                W_left,
                W_right,
                right,
            )
            use_sparse = bool(sparse_kernel) and sparse_pairs <= threshold * dense_pairs
        else:
            use_sparse = False

        if use_sparse:

            def apply_heff(local):
                if sparse_vectorized:
                    return _apply_sparse_two_site_kernel_vectorized(
                        local, sparse_kernel, shape
                    )
                return _apply_sparse_two_site_kernel(local, sparse_kernel, shape)
        else:
            left_kernel = np.einsum("amb,mnpq->abnpq", left, W_left, optimize=True)
            right_kernel = np.einsum("nors,cod->nrscd", W_right, right, optimize=True)

            def apply_heff(local):
                tmp = np.tensordot(left_kernel, local, axes=([1, 4], [0, 1]))
                return np.tensordot(tmp, right_kernel, axes=([1, 3, 4], [0, 2, 4]))

    return _krylov_expm_apply(
        theta,
        shape,
        apply_heff,
        dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _apply_bond_heff(center, left, right):
    tmp = np.einsum("bs,rms->brm", center, right, optimize=True)
    return np.einsum("amb,brm->ar", left, tmp, optimize=True)


def _evolve_bond(
    center,
    left,
    right,
    dt,
    *,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
):
    shape = center.shape
    if _is_lanczos_method(krylov_method) and _cpp_tdvp_available():
        try:
            return _tdvp_cpp.bond_lanczos(
                np.asarray(center, dtype=complex),
                np.asarray(left, dtype=complex),
                np.asarray(right, dtype=complex),
                float(-dt),
                int(krylov_dim),
                float(krylov_tol),
            )
        except Exception:
            pass

    def apply_heff(local):
        return _apply_bond_heff(local, left, right)

    return _krylov_expm_apply(
        center,
        shape,
        apply_heff,
        -dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _merge_two_site(left_site, right_site):
    return np.tensordot(left_site, right_site, axes=([2], [0]))


def _svd_keep_count(s, max_bond=None, cutoff=0.0):
    keep = len(s)
    if cutoff and cutoff > 0.0:
        keep = int(np.count_nonzero(s > cutoff))
        keep = max(1, keep)
    if max_bond is not None:
        keep = min(keep, int(max_bond))
    return max(1, keep)


def _split_two_site_left(theta, max_bond=None, cutoff=0.0):
    left_dim, d_left, d_right, right_dim = theta.shape
    mat = theta.reshape(left_dim * d_left, d_right * right_dim)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    keep = _svd_keep_count(s, max_bond=max_bond, cutoff=cutoff)
    discarded = float(np.sum(np.abs(s[keep:]) ** 2))
    u = u[:, :keep]
    s_keep = s[:keep]
    vh = vh[:keep]
    left_site = u.reshape(left_dim, d_left, keep)
    right_center = (s_keep[:, None] * vh).reshape(keep, d_right, right_dim)
    return left_site, right_center, discarded


def _split_two_site_right(theta, max_bond=None, cutoff=0.0):
    left_dim, d_left, d_right, right_dim = theta.shape
    mat = theta.reshape(left_dim * d_left, d_right * right_dim)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    keep = _svd_keep_count(s, max_bond=max_bond, cutoff=cutoff)
    discarded = float(np.sum(np.abs(s[keep:]) ** 2))
    u = u[:, :keep]
    s_keep = s[:keep]
    vh = vh[:keep]
    left_center = (u * s_keep[None, :]).reshape(left_dim, d_left, keep)
    right_site = vh.reshape(keep, d_right, right_dim)
    return left_center, right_site, discarded


def one_site_tdvp_step(
    psi,
    H,
    dt,
    *,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
    diagonal_fast_path=False,
    canonicalize=True,
    normalize=True,
    return_info=False,
):
    """
    Propagate an MPS by one second-order one-site TDVP step.

    The implementation uses projector splitting: local site tensors are evolved
    by ``exp(-i H_eff dt/2)`` and bond-center matrices by the compensating
    ``exp(+i K_eff dt/2)``.  It keeps the MPS bond dimensions fixed.
    """
    if not isinstance(psi, MPS):
        raise TypeError("one_site_tdvp_step expects an MPS initial state.")

    mpo = [np.asarray(w) for w in _mpo_factors(H)]
    if len(mpo) != psi.L:
        raise ValueError("MPS and MPO lengths must match.")

    work = psi.copy().to_order(["lv", "p", "rv"])
    if canonicalize:
        work = work.right_canonicalize()
    factors = _standard_mps_factors(work)
    nsites = len(factors)
    if nsites == 0:
        raise ValueError("Cannot propagate an empty MPS.")

    for i, (A, W) in enumerate(zip(factors, mpo)):
        if A.shape[1] != W.shape[2] or A.shape[1] != W.shape[3]:
            raise ValueError(f"Physical dimension mismatch at site {i}.")

    dtype = np.result_type(*(factors + mpo), complex)
    left_identity = np.ones((1, 1, 1), dtype=dtype)
    right_identity = np.ones((1, 1, 1), dtype=dtype)

    if nsites == 1:
        factors[0] = _evolve_site(
            factors[0],
            left_identity,
            mpo[0],
            right_identity,
            dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            diagonal_fast_path=diagonal_fast_path,
        )
        out = MPS(factors, labels=["lv", "p", "rv"], sites=psi.sites)
        norm2 = out.norm_squared()
        if normalize:
            out.normalize()
        info = {
            "pre_normalization_norm2": float(np.real(norm2)),
            "pre_normalization_norm": float(np.sqrt(max(float(np.real(norm2)), 0.0))),
        }
        return (out, info) if return_info else out

    half_dt = 0.5 * dt
    right_envs = _build_right_envs(factors, mpo)
    left_envs = [None] * nsites
    left_envs[0] = left_identity

    left = left_identity
    for i in range(nsites - 1):
        factors[i] = _evolve_site(
            factors[i],
            left,
            mpo[i],
            right_envs[i + 1],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            diagonal_fast_path=diagonal_fast_path,
        )
        q, center = _left_qr(factors[i])
        factors[i] = q
        left = _update_left_env(left, q, mpo[i])
        left_envs[i + 1] = left
        center = _evolve_bond(
            center,
            left,
            right_envs[i + 1],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        factors[i + 1] = np.tensordot(center, factors[i + 1], axes=([1], [0]))

    factors[-1] = _evolve_site(
        factors[-1],
        left_envs[-1],
        mpo[-1],
        right_identity,
        dt,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        krylov_method=krylov_method,
        diagonal_fast_path=diagonal_fast_path,
    )

    right = right_identity
    for i in range(nsites - 1, 0, -1):
        if i != nsites - 1:
            factors[i] = _evolve_site(
                factors[i],
                left_envs[i],
                mpo[i],
                right,
                half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                diagonal_fast_path=diagonal_fast_path,
            )
        center, q = _right_rq(factors[i])
        factors[i] = q
        right = _update_right_env(right, q, mpo[i])
        center = _evolve_bond(
            center,
            left_envs[i],
            right,
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        factors[i - 1] = np.tensordot(factors[i - 1], center, axes=([2], [0]))

    factors[0] = _evolve_site(
        factors[0],
        left_identity,
        mpo[0],
        right,
        half_dt,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        krylov_method=krylov_method,
        diagonal_fast_path=diagonal_fast_path,
    )

    out = MPS(factors, labels=["lv", "p", "rv"], sites=psi.sites)
    norm2 = out.norm_squared()
    if normalize:
        out.normalize()
    info = {
        "pre_normalization_norm2": float(np.real(norm2)),
        "pre_normalization_norm": float(np.sqrt(max(float(np.real(norm2)), 0.0))),
    }
    return (out, info) if return_info else out


def block_sparse_one_site_tdvp_step(
    psi,
    H,
    dt,
    *,
    local_sectors,
    target_sector,
    site_qn_maps=None,
    target_qn=None,
    block_mpo=None,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
    canonicalize=True,
    normalize=True,
    copy_state=True,
    moving_environment=None,
    env_plan_prefix="tdvp-block",
    return_info=False,
):
    """
    Propagate a fixed-sector MPS by one one-site TDVP step using Abelian blocks.

    The returned MPS stores native ``AbelianSiteTensorData`` tensors in
    ``["lv", "rv", "p"]`` layout.  No dense sector projector is applied during
    propagation; local site and bond-center evolutions act directly inside the
    block layouts selected by the Abelian quantum numbers.
    """
    if not isinstance(psi, MPS):
        raise TypeError("block_sparse_one_site_tdvp_step expects an MPS initial state.")

    nsites = psi.L
    env_plan_prefix = str(env_plan_prefix or "tdvp-block")
    dense_mpo = None if block_mpo is not None else _mpo_factors(H)
    mpo_length = len(block_mpo) if block_mpo is not None else len(dense_mpo)
    if mpo_length != nsites:
        raise ValueError("MPS and MPO lengths must match.")
    if site_qn_maps is None or target_qn is None:
        phys_dims = _block_sparse_phys_dims(psi)
        site_qn_maps, target_qn = _block_sparse_site_qn_maps(
            local_sectors,
            nsites,
            phys_dims,
            target_sector,
        )
    factors = _as_block_sparse_factors(psi, site_qn_maps, copy=copy_state)
    mpo_cached = block_mpo is not None
    mpo = block_mpo if mpo_cached else _as_block_sparse_mpo(dense_mpo, site_qn_maps)
    moving_stats_before = _moving_environment_stats(moving_environment)
    if canonicalize:
        factors = _block_right_canonicalize_qr(factors)
        factors, _ = _normalize_block_factors_inplace(factors)

    if nsites == 0:
        raise ValueError("Cannot propagate an empty MPS.")

    sweep_info = {"cpp_one_site_engine": False}
    cpp_sweep = _cpp_one_site_tdvp_sweep(
        factors,
        mpo,
        target_qn,
        dt,
        moving_environment=moving_environment,
        env_plan_prefix=env_plan_prefix,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        krylov_method=krylov_method,
    )
    if cpp_sweep is None:
        factors = _python_one_site_tdvp_sweep(
            factors,
            mpo,
            target_qn,
            dt,
            moving_environment=moving_environment,
            env_plan_prefix=env_plan_prefix,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
    else:
        factors, sweep_info = cpp_sweep

    pre_norm2 = _right_canonical_block_mps_norm2(factors)
    if normalize:
        factors, pre_norm2 = _normalize_block_factors_inplace(factors, pre_norm2)
    out = MPS(factors, labels=["lv", "rv", "p"], sites=psi.sites)
    info = {
        "backend": "block-sparse",
        "projection_backend": "block-sparse",
        "integrator": "tdvp",
        "target_sector": target_sector,
        "target_qn": target_qn,
        "pre_normalization_norm2": float(pre_norm2),
        "pre_normalization_norm": float(np.sqrt(max(float(pre_norm2), 0.0))),
        "input_sector_weight": 1.0,
        "output_sector_weight": 1.0,
        "input_discarded_sector_weight": 0.0,
        "output_discarded_sector_weight": 0.0,
        "mps_blocks": int(sum(len(site.data) for site in factors)),
        "mpo_blocks": int(sum(len(site.data) for site in mpo)),
        "mpo_cached": bool(mpo_cached),
        "state_copied": bool(copy_state),
    }
    info.update(sweep_info)
    info.update(_moving_environment_delta_info(moving_environment, moving_stats_before))
    return (out, info) if return_info else out


def block_sparse_two_site_tdvp_step(
    psi,
    H,
    dt,
    *,
    local_sectors,
    target_sector,
    site_qn_maps=None,
    target_qn=None,
    block_mpo=None,
    max_bond=None,
    cutoff=0.0,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
    canonicalize=True,
    normalize=True,
    copy_state=True,
    moving_environment=None,
    env_plan_prefix="tdvp2-block",
    return_info=False,
):
    """Propagate a fixed-sector MPS by one block-sparse two-site TDVP step."""
    if not isinstance(psi, MPS):
        raise TypeError("block_sparse_two_site_tdvp_step expects an MPS initial state.")

    nsites = psi.L
    dense_mpo = None if block_mpo is not None else _mpo_factors(H)
    mpo_length = len(block_mpo) if block_mpo is not None else len(dense_mpo)
    if mpo_length != nsites:
        raise ValueError("MPS and MPO lengths must match.")
    if site_qn_maps is None or target_qn is None:
        phys_dims = _block_sparse_phys_dims(psi)
        site_qn_maps, target_qn = _block_sparse_site_qn_maps(
            local_sectors,
            nsites,
            phys_dims,
            target_sector,
        )

    factors = _as_block_sparse_factors(psi, site_qn_maps, copy=copy_state)
    mpo_cached = block_mpo is not None
    mpo = block_mpo if mpo_cached else _as_block_sparse_mpo(dense_mpo, site_qn_maps)
    moving_stats_before = _moving_environment_stats(moving_environment)
    env_plan_prefix = str(env_plan_prefix or "tdvp2-block")

    if canonicalize:
        factors = _block_right_canonicalize_qr(factors)
        factors, _ = _normalize_block_factors_inplace(factors)
    if nsites == 0:
        raise ValueError("Cannot propagate an empty MPS.")
    if nsites == 1:
        out, info = block_sparse_one_site_tdvp_step(
            psi,
            H,
            dt,
            local_sectors=local_sectors,
            target_sector=target_sector,
            site_qn_maps=site_qn_maps,
            target_qn=target_qn,
            block_mpo=mpo,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            canonicalize=canonicalize,
            normalize=normalize,
            copy_state=copy_state,
            moving_environment=moving_environment,
            env_plan_prefix=env_plan_prefix,
            return_info=True,
        )
        info["integrator"] = "tdvp2"
        info["truncation_error"] = 0.0
        return (out, info) if return_info else out

    cpp_sweep = None
    if nsites > 2:
        cpp_sweep = _cpp_two_site_tdvp_sweep(
            factors,
            mpo,
            target_qn,
            dt,
            max_bond=max_bond,
            cutoff=cutoff,
            moving_environment=moving_environment,
            env_plan_prefix=env_plan_prefix,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
    if cpp_sweep is not None:
        factors, sweep_info = cpp_sweep
        pre_norm2 = _right_canonical_block_mps_norm2(factors)
        if normalize:
            factors, pre_norm2 = _normalize_block_factors_inplace(
                factors,
                pre_norm2,
            )
        out = MPS(factors, labels=["lv", "rv", "p"], sites=psi.sites)
        info = {
            "backend": "block-sparse",
            "projection_backend": "block-sparse",
            "integrator": "tdvp2",
            "target_sector": target_sector,
            "target_qn": target_qn,
            "pre_normalization_norm2": float(pre_norm2),
            "pre_normalization_norm": float(np.sqrt(max(float(pre_norm2), 0.0))),
            "truncation_error": float(sweep_info["truncation_error"]),
            "max_kept_states": int(sweep_info["max_kept_states"]),
            "input_sector_weight": 1.0,
            "output_sector_weight": 1.0,
            "input_discarded_sector_weight": 0.0,
            "output_discarded_sector_weight": 0.0,
            "mps_blocks": int(sum(len(site.data) for site in factors)),
            "mpo_blocks": int(sum(len(site.data) for site in mpo)),
            "mpo_cached": bool(mpo_cached),
            "state_copied": bool(copy_state),
        }
        info.update(sweep_info)
        info.update(
            _moving_environment_delta_info(
                moving_environment,
                moving_stats_before,
            )
        )
        return (out, info) if return_info else out

    half_dt = 0.5 * dt
    truncation_error = 0.0
    kept_states = []
    right_envs = _build_block_right_envs(
        factors,
        mpo,
        target_qn,
        moving_environment=moving_environment,
        env_plan_prefix=env_plan_prefix,
    )
    left_envs = [None] * nsites
    left_envs[0] = initial_E(mpo[0])

    left = left_envs[0]
    for site in range(nsites - 1):
        theta = abelian_merge_adjacent_site_tensors(
            factors[site],
            factors[site + 1],
        )
        theta = _evolve_block_two_site(
            theta,
            left,
            mpo[site],
            mpo[site + 1],
            right_envs[site + 2],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        split = abelian_split_two_site_svd_data(
            theta.data,
            qns=theta.qns,
            dirs=theta.dirs,
            direction="right",
            m_max=max_bond,
            cutoff=cutoff,
        )
        update = abelian_site_tensors_from_split(split)
        factors[site] = update.left
        factors[site + 1] = update.right
        truncation_error += float(update.truncation_error)
        kept_states.append(int(update.kept_states))
        left = _advance_block_environment(
            "left",
            mpo[site],
            factors[site],
            left,
            factors[site],
            moving_environment=moving_environment,
            plan_key=f"{env_plan_prefix}:left-sweep:{site}",
        )
        left_envs[site + 1] = left
        if site < nsites - 2:
            factors[site + 1] = _evolve_block_site(
                factors[site + 1],
                left,
                mpo[site + 1],
                right_envs[site + 2],
                -half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
            )

    right = initial_F(mpo[-1], target_qn=target_qn)
    for site in range(nsites - 2, -1, -1):
        theta = abelian_merge_adjacent_site_tensors(
            factors[site],
            factors[site + 1],
        )
        theta = _evolve_block_two_site(
            theta,
            left_envs[site],
            mpo[site],
            mpo[site + 1],
            right,
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
        )
        split = abelian_split_two_site_svd_data(
            theta.data,
            qns=theta.qns,
            dirs=theta.dirs,
            direction="left",
            m_max=max_bond,
            cutoff=cutoff,
        )
        update = abelian_site_tensors_from_split(split)
        factors[site] = update.left
        factors[site + 1] = update.right
        truncation_error += float(update.truncation_error)
        kept_states.append(int(update.kept_states))
        right = _advance_block_environment(
            "right",
            mpo[site + 1],
            factors[site + 1],
            right,
            factors[site + 1],
            moving_environment=moving_environment,
            plan_key=f"{env_plan_prefix}:right-sweep:{site + 1}",
        )
        if site > 0:
            factors[site] = _evolve_block_site(
                factors[site],
                left_envs[site],
                mpo[site],
                right,
                -half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
            )

    pre_norm2 = _right_canonical_block_mps_norm2(factors)
    if normalize:
        factors, pre_norm2 = _normalize_block_factors_inplace(factors, pre_norm2)
    out = MPS(factors, labels=["lv", "rv", "p"], sites=psi.sites)
    info = {
        "backend": "block-sparse",
        "projection_backend": "block-sparse",
        "integrator": "tdvp2",
        "target_sector": target_sector,
        "target_qn": target_qn,
        "pre_normalization_norm2": float(pre_norm2),
        "pre_normalization_norm": float(np.sqrt(max(float(pre_norm2), 0.0))),
        "truncation_error": float(truncation_error),
        "max_kept_states": int(max(kept_states, default=1)),
        "input_sector_weight": 1.0,
        "output_sector_weight": 1.0,
        "input_discarded_sector_weight": 0.0,
        "output_discarded_sector_weight": 0.0,
        "mps_blocks": int(sum(len(site.data) for site in factors)),
        "mpo_blocks": int(sum(len(site.data) for site in mpo)),
        "mpo_cached": bool(mpo_cached),
        "state_copied": bool(copy_state),
    }
    info.update(_moving_environment_delta_info(moving_environment, moving_stats_before))
    return (out, info) if return_info else out


def two_site_tdvp_step(
    psi,
    H,
    dt,
    *,
    max_bond=None,
    cutoff=0.0,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
    diagonal_fast_path=False,
    sparse_threshold=0.0,
    sparse_vectorized=True,
    canonicalize=True,
    normalize=True,
    return_info=False,
    _dense_workspace=None,
    _dense_cpp_env=False,
):
    """
    Propagate an MPS by one second-order two-site TDVP step.

    Unlike one-site TDVP, this can enlarge bonds up to ``max_bond`` during the
    SVD splits.  The discarded singular-value weight is reported as
    ``truncation_error`` when ``return_info`` is true.
    """
    if not isinstance(psi, MPS):
        raise TypeError("two_site_tdvp_step expects an MPS initial state.")

    mpo = [np.asarray(w) for w in _mpo_factors(H)]
    if len(mpo) != psi.L:
        raise ValueError("MPS and MPO lengths must match.")

    work = psi.copy().to_order(["lv", "p", "rv"])
    if canonicalize:
        work = work.right_canonicalize()
    factors = _standard_mps_factors(work)
    nsites = len(factors)
    if nsites == 0:
        raise ValueError("Cannot propagate an empty MPS.")
    if nsites == 1:
        return one_site_tdvp_step(
            psi,
            H,
            dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            diagonal_fast_path=diagonal_fast_path,
            canonicalize=canonicalize,
            normalize=normalize,
            return_info=return_info,
        )

    for i, (A, W) in enumerate(zip(factors, mpo)):
        if A.shape[1] != W.shape[2] or A.shape[1] != W.shape[3]:
            raise ValueError(f"Physical dimension mismatch at site {i}.")

    half_dt = 0.5 * dt
    truncation_error = 0.0
    right_envs = _build_right_envs(factors, mpo, dense_cpp=_dense_cpp_env)
    left_envs = [None] * nsites
    left_envs[0] = np.ones((1, 1, 1), dtype=np.result_type(*(factors + mpo), complex))

    left = left_envs[0]
    for i in range(nsites - 1):
        theta = _merge_two_site(factors[i], factors[i + 1])
        theta = _evolve_two_site(
            theta,
            left,
            mpo[i],
            mpo[i + 1],
            right_envs[i + 2],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            diagonal_fast_path=diagonal_fast_path,
            sparse_threshold=sparse_threshold,
            sparse_vectorized=sparse_vectorized,
            dense_workspace=_dense_workspace,
            workspace_key=f"tdvp2:{i}",
        )
        factors[i], right_center, discarded = _split_two_site_left(
            theta,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        truncation_error += discarded
        left = _update_left_env(left, factors[i], mpo[i], dense_cpp=_dense_cpp_env)
        left_envs[i + 1] = left
        if i < nsites - 2:
            right_center = _evolve_site(
                right_center,
                left,
                mpo[i + 1],
                right_envs[i + 2],
                -half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                diagonal_fast_path=diagonal_fast_path,
            )
        factors[i + 1] = right_center

    right = np.ones((1, 1, 1), dtype=np.result_type(*(factors + mpo), complex))
    for i in range(nsites - 2, -1, -1):
        theta = _merge_two_site(factors[i], factors[i + 1])
        theta = _evolve_two_site(
            theta,
            left_envs[i],
            mpo[i],
            mpo[i + 1],
            right,
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            diagonal_fast_path=diagonal_fast_path,
            sparse_threshold=sparse_threshold,
            sparse_vectorized=sparse_vectorized,
            dense_workspace=_dense_workspace,
            workspace_key=f"tdvp2:{i}",
        )
        left_center, factors[i + 1], discarded = _split_two_site_right(
            theta,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        truncation_error += discarded
        right = _update_right_env(
            right, factors[i + 1], mpo[i + 1], dense_cpp=_dense_cpp_env
        )
        if i > 0:
            left_center = _evolve_site(
                left_center,
                left_envs[i],
                mpo[i],
                right,
                -half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                diagonal_fast_path=diagonal_fast_path,
            )
        factors[i] = left_center

    out = MPS(factors, labels=["lv", "p", "rv"], sites=psi.sites)
    norm2 = out.norm_squared()
    if normalize:
        out.normalize()
    info = {
        "pre_normalization_norm2": float(np.real(norm2)),
        "pre_normalization_norm": float(np.sqrt(max(float(np.real(norm2)), 0.0))),
        "truncation_error": truncation_error,
    }
    return (out, info) if return_info else out


def _evolve_site_sum(
    theta,
    environments,
    dt,
    *,
    krylov_dim,
    krylov_tol,
    krylov_method,
    executor=None,
):
    shape = theta.shape
    environments = tuple(environments)
    native = (
        _tdvp_cpp
        if _cpp_tdvp_available()
        and _is_lanczos_method(krylov_method)
        and getattr(_tdvp_cpp, "site_lanczos_sum", None) is not None
        else None
    )
    if native is not None:
        try:
            result = native.site_lanczos_sum(
                np.asarray(theta, dtype=complex),
                [np.asarray(item[0], dtype=complex) for item in environments],
                [np.asarray(item[1], dtype=complex) for item in environments],
                [np.asarray(item[2], dtype=complex) for item in environments],
                float(dt),
                int(krylov_dim),
                float(krylov_tol),
                int(_SUM_TDVP_CPP_MAX_DIRECT_SUM_ELEMENTS),
                int(getattr(executor, "_max_workers", 1)),
            )
            _release_free_numeric_pages()
            if os.environ.get("PYQED_TDVP_MEMORY_PROFILE"):
                try:
                    import platform
                    import psutil
                    import resource

                    stats = dict(native.kernel_stats())
                    rss = psutil.Process().memory_info().rss / 2**30
                    peak_scale = 1.0 if platform.system() == "Darwin" else 1024.0
                    peak = (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        * peak_scale / 2**30
                    )
                    print(
                        "[sum-TDVP memory] site "
                        f"shape={shape} rss={rss:.4f} GiB peak={peak:.4f} GiB "
                        f"stats={stats}",
                        flush=True,
                    )
                except Exception:
                    pass
            return result
        except Exception as exc:
            if os.environ.get("PYQED_TDVP_MEMORY_PROFILE"):
                print(
                    f"[sum-TDVP native fallback] shape={shape}: {exc}",
                    flush=True,
                )

    def build_kernel(environment):
        left, operator, right = environment
        return (
            np.einsum("amb,mnpq->abnpq", left, operator, optimize=True),
            right,
        )

    kernels = list(map(build_kernel, environments)) if executor is None else list(
        executor.map(build_kernel, environments)
    )

    def apply_kernel(kernel, local):
        left_kernel, right = kernel
        tmp = np.tensordot(local, right, axes=([2], [2]))
        return np.tensordot(
            left_kernel,
            tmp,
            axes=([1, 2, 4], [0, 3, 1]),
        )

    def apply_heff(local):
        if executor is None:
            values = iter(apply_kernel(kernel, local) for kernel in kernels)
        else:
            values = iter(
                executor.map(
                    apply_kernel,
                    kernels,
                    (local,) * len(kernels),
                )
            )
        result = next(values)
        for value in values:
            result += value
        return result

    return _krylov_expm_apply(
        theta,
        shape,
        apply_heff,
        dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _evolve_two_site_sum(
    theta,
    environments,
    dt,
    *,
    krylov_dim,
    krylov_tol,
    krylov_method,
    executor=None,
):
    shape = theta.shape
    environments = tuple(environments)
    native = (
        _tdvp_cpp
        if _cpp_tdvp_available()
        and _is_lanczos_method(krylov_method)
        and getattr(_tdvp_cpp, "two_site_lanczos_sum", None) is not None
        else None
    )
    if native is not None:
        try:
            return native.two_site_lanczos_sum(
                np.asarray(theta, dtype=complex),
                [np.asarray(item[0], dtype=complex) for item in environments],
                [np.asarray(item[1], dtype=complex) for item in environments],
                [np.asarray(item[2], dtype=complex) for item in environments],
                [np.asarray(item[3], dtype=complex) for item in environments],
                float(dt),
                int(krylov_dim),
                float(krylov_tol),
                int(_SUM_TDVP_CPP_MAX_DIRECT_SUM_ELEMENTS),
                int(getattr(executor, "_max_workers", 1)),
            )
        except Exception:
            pass

    def build_kernel(environment):
        left, first, second, right = environment
        return (
            np.einsum("amb,mnpq->abnpq", left, first, optimize=True),
            np.einsum("nors,cod->nrscd", second, right, optimize=True),
        )

    kernels = list(map(build_kernel, environments)) if executor is None else list(
        executor.map(build_kernel, environments)
    )

    def apply_kernel(kernel, local):
        left_kernel, right_kernel = kernel
        tmp = np.tensordot(
            left_kernel,
            local,
            axes=([1, 4], [0, 1]),
        )
        return np.tensordot(
            tmp,
            right_kernel,
            axes=([1, 3, 4], [0, 2, 4]),
        )

    def apply_heff(local):
        if executor is None:
            values = iter(apply_kernel(kernel, local) for kernel in kernels)
        else:
            values = iter(
                executor.map(
                    apply_kernel,
                    kernels,
                    (local,) * len(kernels),
                )
            )
        result = next(values)
        for value in values:
            result += value
        return result

    return _krylov_expm_apply(
        theta,
        shape,
        apply_heff,
        dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _evolve_bond_sum(
    center,
    environments,
    dt,
    *,
    krylov_dim,
    krylov_tol,
    krylov_method,
):
    environments = tuple(environments)
    native = (
        _tdvp_cpp
        if _cpp_tdvp_available()
        and _is_lanczos_method(krylov_method)
        and getattr(_tdvp_cpp, "bond_lanczos_sum", None) is not None
        else None
    )
    if native is not None:
        try:
            return native.bond_lanczos_sum(
                np.asarray(center, dtype=complex),
                [np.asarray(item[0], dtype=complex) for item in environments],
                [np.asarray(item[1], dtype=complex) for item in environments],
                float(-dt),
                int(krylov_dim),
                float(krylov_tol),
                int(_SUM_TDVP_CPP_MAX_DIRECT_SUM_ELEMENTS),
            )
        except Exception:
            pass

    def apply_heff(local):
        result = np.zeros_like(local, dtype=complex)
        for left, right in environments:
            result += _apply_bond_heff(local, left, right)
        return result

    return _krylov_expm_apply(
        center,
        center.shape,
        apply_heff,
        -dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def one_site_tdvp_sum_step(
    psi,
    operators,
    dt,
    *,
    max_bond=None,
    cutoff=0.0,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
    canonicalize=True,
    normalize=True,
    imaginary_time=False,
    return_info=False,
    _executor=None,
    _right_environments=None,
    _cache_right_environments=False,
):
    r"""Propagate at fixed MPS ranks with a sum of compact MPO components.

    Set ``imaginary_time=True`` to apply the normalized projector
    :math:`\exp(-H\,dt)` instead of real-time evolution.
    """
    if not isinstance(psi, MPS):
        raise TypeError("one_site_tdvp_sum_step expects an MPS initial state")
    operators = tuple(operators)
    if not operators or not all(isinstance(operator, MPO) for operator in operators):
        raise TypeError("operators must be a non-empty sequence of MPOs")
    mpos = [
        [np.asarray(factor) for factor in operator.factors] for operator in operators
    ]
    if any(len(mpo) != psi.L for mpo in mpos):
        raise ValueError("MPS and all MPO lengths must match")

    work = psi.copy().to_order(["lv", "p", "rv"])
    if canonicalize:
        work = work.right_canonicalize()
    factors = _standard_mps_factors(work)
    nsites = len(factors)
    if nsites == 0:
        raise ValueError("sum TDVP requires at least one site")
    for mpo in mpos:
        for site, (state_factor, operator_factor) in enumerate(zip(factors, mpo)):
            if (
                state_factor.shape[1] != operator_factor.shape[2]
                or state_factor.shape[1] != operator_factor.shape[3]
            ):
                raise ValueError(f"physical dimension mismatch at site {site}")

    native_sweep = (
        _tdvp_cpp
        if _cpp_tdvp_available()
        and _is_lanczos_method(krylov_method)
        and getattr(_tdvp_cpp, "one_site_lanczos_sum_sweep", None) is not None
        and _env_int("PYQED_SUM_TDVP_CPP_SWEEP", 1) != 0
        else None
    )
    if native_sweep is not None:
        try:
            result = native_sweep.one_site_lanczos_sum_sweep(
                [np.asarray(factor, dtype=complex) for factor in factors],
                [
                    [np.asarray(factor, dtype=complex) for factor in mpo]
                    for mpo in mpos
                ],
                [] if _right_environments is None else _right_environments,
                float(dt),
                int(krylov_dim),
                float(krylov_tol),
                int(_SUM_TDVP_CPP_MAX_DIRECT_SUM_ELEMENTS),
                int(getattr(_executor, "_max_workers", 1)),
                bool(normalize),
                bool(imaginary_time),
            )
            factors = [np.asarray(factor) for factor in result["factors"]]
            norm2 = float(result["pre_normalization_norm2"])
            out = MPS(factors, labels=["lv", "p", "rv"], sites=psi.sites)
            out.gauge = "right_canonical"
            out.center = 0
            info = {
                "pre_normalization_norm2": norm2,
                "pre_normalization_norm": float(np.sqrt(max(norm2, 0.0))),
                "truncation_error": 0.0,
                "components": len(mpos),
                "integrator": "tdvp",
                "backend": "compiled-sum-tdvp-sweep",
                "imaginary_time": bool(imaginary_time),
                "right_environments_reused": bool(
                    result["reused_right_environments"]
                ),
            }
            if _cache_right_environments:
                info["_right_environments"] = result["right_environments"]
            _release_free_numeric_pages()
            return (out, info) if return_info else out
        except Exception as exc:
            if os.environ.get("PYQED_TDVP_MEMORY_PROFILE"):
                print(f"[sum-TDVP compiled sweep fallback] {exc}", flush=True)

    if imaginary_time:
        dt = -1j * float(dt)

    identity = np.ones((1, 1, 1), dtype=complex)
    if nsites == 1:
        factors[0] = _evolve_site_sum(
            factors[0],
            [(identity, mpo[0], identity) for mpo in mpos],
            dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            executor=_executor,
        )
    else:
        half_dt = 0.5 * dt

        def build_right_environments(mpo):
            return _build_right_envs(factors, mpo)

        right_envs = (
            list(map(build_right_environments, mpos))
            if _executor is None
            else list(_executor.map(build_right_environments, mpos))
        )
        left_envs = [[None] * nsites for _ in mpos]
        for storage in left_envs:
            storage[0] = identity

        for site in range(nsites - 1):
            factors[site] = _evolve_site_sum(
                factors[site],
                [
                    (
                        left_envs[term][site],
                        mpo[site],
                        right_envs[term][site + 1],
                    )
                    for term, mpo in enumerate(mpos)
                ],
                half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                executor=_executor,
            )
            factors[site], center = _left_qr(factors[site])

            def update_left_environment(item):
                term, mpo = item
                return term, _update_left_env(
                    left_envs[term][site], factors[site], mpo[site]
                )

            updates = (
                map(update_left_environment, enumerate(mpos))
                if _executor is None
                else _executor.map(update_left_environment, enumerate(mpos))
            )
            for term, environment in updates:
                left_envs[term][site + 1] = environment
            center = _evolve_bond_sum(
                center,
                [
                    (
                        left_envs[term][site + 1],
                        right_envs[term][site + 1],
                    )
                    for term in range(len(mpos))
                ],
                half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
            )
            factors[site + 1] = np.tensordot(
                center, factors[site + 1], axes=([1], [0])
            )

        factors[-1] = _evolve_site_sum(
            factors[-1],
            [
                (left_envs[term][-1], mpo[-1], identity)
                for term, mpo in enumerate(mpos)
            ],
            dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            executor=_executor,
        )

        rights = [identity for _ in mpos]
        for site in range(nsites - 1, 0, -1):
            if site != nsites - 1:
                factors[site] = _evolve_site_sum(
                    factors[site],
                    [
                        (left_envs[term][site], mpo[site], rights[term])
                        for term, mpo in enumerate(mpos)
                    ],
                    half_dt,
                    krylov_dim=krylov_dim,
                    krylov_tol=krylov_tol,
                    krylov_method=krylov_method,
                    executor=_executor,
                )
            center, factors[site] = _right_rq(factors[site])

            def update_right_environment(item):
                term, mpo = item
                return term, _update_right_env(
                    rights[term], factors[site], mpo[site]
                )

            updates = (
                map(update_right_environment, enumerate(mpos))
                if _executor is None
                else _executor.map(update_right_environment, enumerate(mpos))
            )
            for term, environment in updates:
                rights[term] = environment
            center = _evolve_bond_sum(
                center,
                [
                    (left_envs[term][site], rights[term])
                    for term in range(len(mpos))
                ],
                half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
            )
            factors[site - 1] = np.tensordot(
                factors[site - 1], center, axes=([2], [0])
            )

        factors[0] = _evolve_site_sum(
            factors[0],
            [(identity, mpo[0], rights[term]) for term, mpo in enumerate(mpos)],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            executor=_executor,
        )

    out = MPS(factors, labels=["lv", "p", "rv"], sites=psi.sites)
    out.gauge = "right_canonical"
    out.center = 0
    norm2 = float(np.real(out.norm_squared()))
    if normalize:
        out.normalize()
    info = {
        "pre_normalization_norm2": norm2,
        "pre_normalization_norm": float(np.sqrt(max(norm2, 0.0))),
        "truncation_error": 0.0,
        "components": len(mpos),
        "integrator": "tdvp",
        "imaginary_time": bool(imaginary_time),
    }
    return (out, info) if return_info else out


def two_site_tdvp_sum_step(
    psi,
    operators,
    dt,
    *,
    max_bond=None,
    cutoff=0.0,
    krylov_dim=12,
    krylov_tol=1.0e-13,
    krylov_method="lanczos",
    canonicalize=True,
    normalize=True,
    return_info=False,
    _executor=None,
):
    """Propagate with a sum of MPOs without forming their direct-sum bonds."""
    if not isinstance(psi, MPS):
        raise TypeError("two_site_tdvp_sum_step expects an MPS initial state")
    operators = tuple(operators)
    if not operators or not all(isinstance(operator, MPO) for operator in operators):
        raise TypeError("operators must be a non-empty sequence of MPOs")
    mpos = [
        [np.asarray(factor) for factor in operator.factors] for operator in operators
    ]
    if any(len(mpo) != psi.L for mpo in mpos):
        raise ValueError("MPS and all MPO lengths must match")

    work = psi.copy().to_order(["lv", "p", "rv"])
    if canonicalize:
        work = work.right_canonicalize()
    factors = _standard_mps_factors(work)
    nsites = len(factors)
    if nsites < 2:
        raise ValueError("sum TDVP2 requires at least two sites")
    for mpo in mpos:
        for site, (state_factor, operator_factor) in enumerate(zip(factors, mpo)):
            if (
                state_factor.shape[1] != operator_factor.shape[2]
                or state_factor.shape[1] != operator_factor.shape[3]
            ):
                raise ValueError(f"physical dimension mismatch at site {site}")

    half_dt = 0.5 * dt
    truncation_error = 0.0

    def build_right_environments(mpo):
        return _build_right_envs(factors, mpo)

    right_envs = (
        list(map(build_right_environments, mpos))
        if _executor is None
        else list(_executor.map(build_right_environments, mpos))
    )
    left_envs = [[None] * nsites for _ in mpos]
    for term, (mpo, storage) in enumerate(zip(mpos, left_envs)):
        storage[0] = np.ones(
            (1, 1, 1),
            dtype=np.result_type(*(factors + mpo), complex),
        )

    for site in range(nsites - 1):
        theta = _merge_two_site(factors[site], factors[site + 1])
        theta = _evolve_two_site_sum(
            theta,
            [
                (
                    left_envs[term][site],
                    mpo[site],
                    mpo[site + 1],
                    right_envs[term][site + 2],
                )
                for term, mpo in enumerate(mpos)
            ],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            executor=_executor,
        )
        factors[site], center, discarded = _split_two_site_left(
            theta,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        truncation_error += discarded

        def update_left_environment(item):
            term, mpo = item
            return term, _update_left_env(
                left_envs[term][site], factors[site], mpo[site]
            )

        updates = (
            map(update_left_environment, enumerate(mpos))
            if _executor is None
            else _executor.map(update_left_environment, enumerate(mpos))
        )
        for term, environment in updates:
            left_envs[term][site + 1] = environment
        if site < nsites - 2:
            center = _evolve_site_sum(
                center,
                [
                    (
                        left_envs[term][site + 1],
                        mpo[site + 1],
                        right_envs[term][site + 2],
                    )
                    for term, mpo in enumerate(mpos)
                ],
                -half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                executor=_executor,
            )
        factors[site + 1] = center

    right = [
        np.ones((1, 1, 1), dtype=np.result_type(*(factors + mpo), complex))
        for mpo in mpos
    ]
    for site in range(nsites - 2, -1, -1):
        theta = _merge_two_site(factors[site], factors[site + 1])
        theta = _evolve_two_site_sum(
            theta,
            [
                (
                    left_envs[term][site],
                    mpo[site],
                    mpo[site + 1],
                    right[term],
                )
                for term, mpo in enumerate(mpos)
            ],
            half_dt,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            krylov_method=krylov_method,
            executor=_executor,
        )
        center, factors[site + 1], discarded = _split_two_site_right(
            theta,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        truncation_error += discarded

        def update_right_environment(item):
            term, mpo = item
            return term, _update_right_env(
                right[term], factors[site + 1], mpo[site + 1]
            )

        updates = (
            map(update_right_environment, enumerate(mpos))
            if _executor is None
            else _executor.map(update_right_environment, enumerate(mpos))
        )
        for term, environment in updates:
            right[term] = environment
        if site > 0:
            center = _evolve_site_sum(
                center,
                [
                    (
                        left_envs[term][site],
                        mpo[site],
                        right[term],
                    )
                    for term, mpo in enumerate(mpos)
                ],
                -half_dt,
                krylov_dim=krylov_dim,
                krylov_tol=krylov_tol,
                krylov_method=krylov_method,
                executor=_executor,
            )
        factors[site] = center

    out = MPS(factors, labels=["lv", "p", "rv"], sites=psi.sites)
    out.gauge = "right_canonical"
    out.center = 0
    norm2 = float(np.real(out.norm_squared()))
    if normalize:
        out.normalize()
    info = {
        "pre_normalization_norm2": norm2,
        "pre_normalization_norm": float(np.sqrt(max(norm2, 0.0))),
        "truncation_error": float(truncation_error),
        "components": len(mpos),
    }
    return (out, info) if return_info else out


def _sum_tdvp_right_signature(state):
    """Fingerprint sites right of the TDVP center for safe environment reuse."""
    factors = _standard_mps_factors(state)
    return tuple(
        (factor.shape, hash(np.ascontiguousarray(factor).tobytes()))
        for factor in factors[1:]
    )


class TDVPEngine:
    """Reusable TDVP engine for one MPO or a compact sum of MPO components."""

    def __init__(
        self,
        H,
        *,
        integrator="tdvp2",
        max_bond=None,
        cutoff=0.0,
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        sparse_threshold=0.0,
        sparse_vectorized=True,
        canonicalize_first=True,
        canonicalize_each_step=False,
        dense_cpp_workspace="auto",
        workers=1,
    ):
        global _dense_tdvp_cpp_last_error
        key = str(integrator).lower().replace("_", "-")
        if key in {"tdvp", "tdvp1", "1tdvp", "one-site-tdvp", "1site-tdvp"}:
            self.integrator = "tdvp"
        elif key in {"tdvp2", "2tdvp", "two-site-tdvp", "2site-tdvp"}:
            self.integrator = "tdvp2"
        else:
            raise ValueError("integrator must be 'tdvp' or 'tdvp2'.")
        if isinstance(H, MPO):
            self.components = (H,)
        else:
            try:
                self.components = tuple(H)
            except TypeError as exc:
                raise TypeError(
                    "TDVPEngine requires an MPO or a non-empty sequence of MPOs."
                ) from exc
            if not self.components or not all(
                isinstance(operator, MPO) for operator in self.components
            ):
                raise TypeError(
                    "TDVPEngine requires an MPO or a non-empty sequence of MPOs."
                )
        self.operator_mode = "single" if len(self.components) == 1 else "sum"
        self.mpo = (
            [np.asarray(w) for w in _mpo_factors(self.components[0])]
            if self.operator_mode == "single"
            else None
        )
        self.max_bond = max_bond
        self.cutoff = float(cutoff)
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        self.krylov_method = str(krylov_method).lower().replace("_", "-")
        self.diagonal_fast_path = diagonal_fast_path
        self.sparse_threshold = sparse_threshold
        self.sparse_vectorized = sparse_vectorized
        self.canonicalize_first = bool(canonicalize_first)
        self.canonicalize_each_step = bool(canonicalize_each_step)
        workspace_mode = str(dense_cpp_workspace).lower().replace("_", "-")
        if workspace_mode not in {"auto", "on", "off"}:
            raise ValueError("dense_cpp_workspace must be 'auto', 'on', or 'off'.")
        self.dense_cpp_workspace = (
            None
            if workspace_mode == "off" or self.operator_mode == "sum"
            else _new_cpp_dense_tdvp_workspace()
        )
        self._dense_cpp_workspace_dense_mpo = (
            _dense_mpo_for_workspace(self.mpo)
            if self.operator_mode == "single"
            else False
        )
        self.dense_cpp_env = (
            self.operator_mode == "single"
            and workspace_mode == "on"
            and _cpp_dense_tdvp_workspace_type() is not None
        )
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError("workers must be positive")
        self._executor = (
            ThreadPoolExecutor(max_workers=self.workers)
            if self.operator_mode == "sum" and self.workers > 1
            else None
        )
        _dense_tdvp_cpp_last_error = None
        self._prepared = False
        self._right_environments = None
        self._right_signature = None

    def close(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None
        self._right_environments = None
        self._right_signature = None

    def __del__(self):
        self.close()

    def reset(self):
        self._prepared = False
        self._right_environments = None
        self._right_signature = None

    def step(self, psi, dt, *, normalize=True, return_info=True):
        canonicalize = self.canonicalize_each_step or (
            self.canonicalize_first and not self._prepared
        )
        if self.operator_mode == "sum":
            stepper = (
                one_site_tdvp_sum_step
                if self.integrator == "tdvp"
                else two_site_tdvp_sum_step
            )
            arguments = {
                "max_bond": self.max_bond,
                "cutoff": self.cutoff,
                "krylov_dim": self.krylov_dim,
                "krylov_tol": self.krylov_tol,
                "krylov_method": self.krylov_method,
                "canonicalize": canonicalize,
                "normalize": normalize,
                "return_info": True,
                "_executor": self._executor,
            }
            if self.integrator == "tdvp":
                signature = _sum_tdvp_right_signature(psi)
                arguments["_right_environments"] = (
                    self._right_environments
                    if signature == self._right_signature
                    else None
                )
                arguments["_cache_right_environments"] = True
            out, info = stepper(psi, self.components, dt, **arguments)
            if self.integrator == "tdvp":
                self._right_environments = info.pop("_right_environments", None)
                self._right_signature = (
                    _sum_tdvp_right_signature(out)
                    if self._right_environments is not None
                    else None
                )
        elif self.integrator == "tdvp2":
            out, info = two_site_tdvp_step(
                psi,
                self.mpo,
                dt,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
                krylov_dim=self.krylov_dim,
                krylov_tol=self.krylov_tol,
                krylov_method=self.krylov_method,
                diagonal_fast_path=self.diagonal_fast_path,
                sparse_threshold=self.sparse_threshold,
                sparse_vectorized=self.sparse_vectorized,
                canonicalize=canonicalize,
                normalize=normalize,
                return_info=True,
                _dense_workspace=(
                    self.dense_cpp_workspace
                    if self._dense_cpp_workspace_dense_mpo
                    else None
                ),
                _dense_cpp_env=self.dense_cpp_env,
            )
        else:
            out, info = one_site_tdvp_step(
                psi,
                self.mpo,
                dt,
                krylov_dim=self.krylov_dim,
                krylov_tol=self.krylov_tol,
                krylov_method=self.krylov_method,
                diagonal_fast_path=self.diagonal_fast_path,
                canonicalize=canonicalize,
                normalize=normalize,
                return_info=True,
            )
        self._prepared = True
        info["integrator"] = self.integrator
        info["operator_mode"] = self.operator_mode
        info["components"] = len(self.components)
        info["workers"] = self.workers if self.operator_mode == "sum" else 1
        if self.dense_cpp_workspace is not None:
            try:
                native_stats = dict(self.dense_cpp_workspace.stats())
            except Exception:
                native_stats = {}
            info["dense_cpp_workspace"] = bool(native_stats)
            info["dense_cpp_workspace_two_site_evolutions"] = int(
                native_stats.get("two_site_evolve_calls", 0)
            )
            info["dense_cpp_workspace_static_w_reuses"] = int(
                native_stats.get("two_site_static_w_reuses", 0)
            )
            info["dense_cpp_workspace_last_error"] = _dense_tdvp_cpp_last_error
            info["dense_cpp_environment"] = self.dense_cpp_env
        return (out, info) if return_info else out


class SymmetricTDVP:
    """
    Sector-preserving one-site or two-site TDVP driver.

    The default projector is a diagonal finite-state MPO whose virtual bond
    carries the running Abelian quantum number.  A bounded dense projector is
    still available through :meth:`project_dense` as a small-system reference.
    """

    def __init__(
        self,
        H,
        *,
        local_sectors,
        target_sector,
        integrator="tdvp",
        max_bond=None,
        cutoff=0.0,
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
        diagonal_fast_path=False,
        canonicalize_first=None,
        canonicalize_each_step=False,
        projection_backend="mpo",
        max_dense_sites=12,
        max_dense_size=1_000_000,
    ):
        key = str(integrator).lower().replace("_", "-")
        if key in {"tdvp", "tdvp1", "1tdvp", "one-site-tdvp", "1site-tdvp"}:
            self.integrator = "tdvp"
        elif key in {"tdvp2", "2tdvp", "two-site-tdvp", "2site-tdvp"}:
            self.integrator = "tdvp2"
        else:
            raise ValueError("integrator must be 'tdvp' or 'tdvp2'.")
        self._mpo_source = H
        self.mpo = [_copy_mpo_factor_for_tdvp(w) for w in _mpo_factors(H)]
        self.local_sectors = local_sectors
        self.target_sector = target_sector
        self.max_bond = max_bond
        self.cutoff = float(cutoff)
        self.krylov_dim = krylov_dim
        self.krylov_tol = krylov_tol
        self.krylov_method = str(krylov_method).lower().replace("_", "-")
        self.diagonal_fast_path = diagonal_fast_path
        projection_backend = str(projection_backend).lower().replace("_", "-")
        if projection_backend not in {
            "mpo",
            "sector-mpo",
            "dense",
            "dense-sector",
            "block",
            "blocks",
            "block-sparse",
            "abelian",
            "abelian-block",
        }:
            raise ValueError(
                "projection_backend must be 'mpo', 'dense', or 'block-sparse'."
            )
        if projection_backend in {"mpo", "sector-mpo"}:
            self.projection_backend = "mpo"
        elif projection_backend in {"dense", "dense-sector"}:
            self.projection_backend = "dense"
        else:
            self.projection_backend = "block-sparse"
        if canonicalize_first is None:
            canonicalize_first = True
        self.canonicalize_first = bool(canonicalize_first)
        self.canonicalize_each_step = bool(canonicalize_each_step)
        self.max_dense_sites = int(max_dense_sites)
        self.max_dense_size = int(max_dense_size)
        self._prepared = False
        self._mask_cache = {}
        self._projector_cache = {}
        self._block_sparse_sector_cache = {}
        self._block_sparse_mpo_cache = {}
        self._block_sparse_cpp_moving_environment = None
        self._block_sparse_cpp_moving_environment_disabled = False
        self._block_sparse_env_plan_prefix = f"sym{self.integrator}-block"

    def reset(self):
        self._prepared = False

    def _block_sparse_moving_environment(self):
        if self._block_sparse_cpp_moving_environment_disabled:
            return None
        if self._block_sparse_cpp_moving_environment is None:
            owner = _new_cpp_moving_environment()
            if owner is None:
                self._block_sparse_cpp_moving_environment_disabled = True
                return None
            self._block_sparse_cpp_moving_environment = owner
        return self._block_sparse_cpp_moving_environment

    def _block_sparse_sector_data(self, psi):
        nsites = psi.L
        if len(self.mpo) != nsites:
            raise ValueError("MPS and MPO lengths must match.")
        phys_dims = _local_sector_phys_dims(self.local_sectors, nsites)
        if phys_dims is None:
            phys_dims = _block_sparse_phys_dims(psi)
        key = (nsites, phys_dims)
        cached = self._block_sparse_sector_cache.get(key)
        if cached is None:
            cached = _block_sparse_site_qn_maps(
                self.local_sectors,
                nsites,
                phys_dims,
                self.target_sector,
            )
            self._block_sparse_sector_cache[key] = cached
        site_qn_maps, target_qn = cached
        return phys_dims, site_qn_maps, target_qn

    def _block_sparse_cached_mpo(self, phys_dims, site_qn_maps):
        if _affine_mpo_metadata(self._mpo_source) is not None:
            return _as_block_sparse_mpo(self._mpo_source, site_qn_maps)
        key = tuple(phys_dims)
        mpo = self._block_sparse_mpo_cache.get(key)
        if mpo is None:
            mpo = _as_block_sparse_mpo(self.mpo, site_qn_maps)
            self._block_sparse_mpo_cache[key] = mpo
        return mpo

    def update_mpo_source(self, H):
        """Refresh the source MPO for affine Hamiltonians without rebuilding the engine."""
        self._mpo_source = H
        self.mpo = [_copy_mpo_factor_for_tdvp(w) for w in _mpo_factors(H)]

    def sector_mask(self, shape):
        shape = tuple(int(dim) for dim in shape)
        if len(shape) > self.max_dense_sites:
            raise ValueError(
                f"SymmetricTDVP dense-sector projection is limited to {self.max_dense_sites} sites; "
                f"got {len(shape)}."
            )
        dense_size = int(np.prod(shape, dtype=np.int64))
        if dense_size > self.max_dense_size:
            raise ValueError(
                f"SymmetricTDVP dense-sector projection would materialize {dense_size} amplitudes; "
                f"limit is {self.max_dense_size}."
            )
        key = shape
        if key not in self._mask_cache:
            self._mask_cache[key] = _dense_sector_mask(
                shape, self.local_sectors, self.target_sector
            )
        return self._mask_cache[key]

    def projector(self, shape):
        shape = tuple(int(dim) for dim in shape)
        if shape not in self._projector_cache:
            self._projector_cache[shape] = _sector_projector_mpo(
                shape,
                self.local_sectors,
                self.target_sector,
            )
        return self._projector_cache[shape]

    def project_dense(self, psi, *, normalize=True, return_info=False):
        if not isinstance(psi, MPS):
            raise TypeError("SymmetricTDVP.project_dense expects an MPS.")
        factors = _standard_mps_factors(psi)
        shape = tuple(factor.shape[1] for factor in factors)
        mask = self.sector_mask(shape)
        tensor = np.asarray(tt_to_tensor(factors), dtype=complex).reshape(shape)
        norm2 = float(np.vdot(tensor.reshape(-1), tensor.reshape(-1)).real)
        projected = np.where(mask, tensor, 0.0)
        projected_norm2 = float(
            np.vdot(projected.reshape(-1), projected.reshape(-1)).real
        )
        if projected_norm2 <= 0.0:
            raise ValueError(
                "The requested target sector has zero weight in the supplied state."
            )

        rank = _full_tt_rank(shape) if self.max_bond is None else int(self.max_bond)
        out = MPS(
            decompose(projected, rank=rank),
            labels=["lv", "p", "rv"],
            sites=psi.sites,
        )
        if normalize:
            out.normalize()
        info = {
            "backend": "dense-sector",
            "target_sector": self.target_sector,
            "sector_weight": 0.0 if norm2 <= 0.0 else projected_norm2 / norm2,
            "discarded_sector_weight": 0.0
            if norm2 <= 0.0
            else max(0.0, 1.0 - projected_norm2 / norm2),
            "pre_projection_norm2": norm2,
            "projected_norm2": projected_norm2,
            "rank": rank,
        }
        return (out, info) if return_info else out

    def project(self, psi, *, normalize=True, return_info=False):
        if self.projection_backend == "block-sparse":
            if not isinstance(psi, MPS):
                raise TypeError("SymmetricTDVP.project expects an MPS.")
            _phys_dims, site_qn_maps, target_qn = self._block_sparse_sector_data(psi)
            block_factors = _as_block_sparse_factors(psi, site_qn_maps)
            norm2 = _block_mps_norm2(block_factors)
            if norm2 <= 0.0:
                raise ValueError(
                    "The requested target sector has zero weight in the supplied state."
                )
            if normalize:
                block_factors, norm2 = _normalize_block_factors_inplace(
                    block_factors, norm2
                )
            out = MPS(
                block_factors,
                labels=["lv", "rv", "p"],
                sites=psi.sites,
            )
            info = {
                "backend": "block-sparse",
                "target_sector": self.target_sector,
                "target_qn": target_qn,
                "sector_weight": 1.0,
                "discarded_sector_weight": 0.0,
                "pre_projection_norm2": norm2,
                "projected_norm2": norm2,
                "mps_blocks": int(sum(len(site.data) for site in block_factors)),
            }
            return (out, info) if return_info else out

        if self.projection_backend == "dense":
            return self.project_dense(psi, normalize=normalize, return_info=return_info)
        if not isinstance(psi, MPS):
            raise TypeError("SymmetricTDVP.project expects an MPS.")

        work = psi.copy().to_order(["lv", "p", "rv"])
        shape = tuple(np.asarray(factor).shape[1] for factor in work.factors)
        projector, bond_state_counts = self.projector(shape)
        norm2 = _mps_factors_norm2(work.factors)
        out = projector @ work
        if self.max_bond is not None and max(out.bond_orders()) > int(self.max_bond):
            if not normalize:
                raise ValueError(
                    "SymmetricTDVP cannot compress a projected state while normalize=False."
                )
            out = out.compress(int(self.max_bond))
            out = projector @ out
            if max(out.bond_orders()) > int(self.max_bond):
                out = out.compress(int(self.max_bond))
            post_compression_projection = True
        else:
            post_compression_projection = False
        projected_norm2 = _mps_factors_norm2(out.factors)
        if projected_norm2 <= 0.0:
            raise ValueError(
                "The requested target sector has zero weight in the supplied state."
            )
        compressed_norm2 = projected_norm2
        projected_norm2_after_compress = projected_norm2
        if normalize:
            _normalize_mps_factors_inplace(out, projected_norm2_after_compress)

        sector_weight = 0.0 if norm2 <= 0.0 else projected_norm2 / norm2
        info = {
            "backend": "sector-mpo",
            "target_sector": self.target_sector,
            "sector_weight": sector_weight,
            "discarded_sector_weight": 0.0
            if norm2 <= 0.0
            else max(0.0, 1.0 - sector_weight),
            "pre_projection_norm2": norm2,
            "projected_norm2": projected_norm2,
            "compressed_norm2": compressed_norm2,
            "projected_norm2_after_compress": projected_norm2_after_compress,
            "post_compression_projection": post_compression_projection,
            "projector_bond_state_counts": bond_state_counts,
            "max_projector_bond": int(max(bond_state_counts)),
        }
        return (out, info) if return_info else out

    def step(self, psi, dt, *, normalize=True, return_info=True):
        canonicalize = self.canonicalize_each_step or (
            self.canonicalize_first and not self._prepared
        )
        if self.projection_backend == "block-sparse":
            phys_dims, site_qn_maps, target_qn = self._block_sparse_sector_data(psi)
            block_mpo = self._block_sparse_cached_mpo(phys_dims, site_qn_maps)
            if self.integrator == "tdvp2":
                out, info = block_sparse_two_site_tdvp_step(
                    psi,
                    self.mpo,
                    dt,
                    local_sectors=self.local_sectors,
                    target_sector=self.target_sector,
                    max_bond=self.max_bond,
                    cutoff=self.cutoff,
                    site_qn_maps=site_qn_maps,
                    target_qn=target_qn,
                    block_mpo=block_mpo,
                    krylov_dim=self.krylov_dim,
                    krylov_tol=self.krylov_tol,
                    krylov_method=self.krylov_method,
                    canonicalize=canonicalize,
                    normalize=normalize,
                    return_info=True,
                )
                self._prepared = True
                return (out, info) if return_info else out
            moving_environment = self._block_sparse_moving_environment()
            if self.integrator == "tdvp2":
                out, info = block_sparse_two_site_tdvp_step(
                    psi,
                    self.mpo,
                    dt,
                    local_sectors=self.local_sectors,
                    target_sector=self.target_sector,
                    site_qn_maps=site_qn_maps,
                    target_qn=target_qn,
                    block_mpo=block_mpo,
                    max_bond=self.max_bond,
                    cutoff=self.cutoff,
                    krylov_dim=self.krylov_dim,
                    krylov_tol=self.krylov_tol,
                    krylov_method=self.krylov_method,
                    canonicalize=canonicalize,
                    normalize=normalize,
                    copy_state=not self._prepared,
                    moving_environment=moving_environment,
                    env_plan_prefix=self._block_sparse_env_plan_prefix,
                    return_info=True,
                )
            else:
                out, info = block_sparse_one_site_tdvp_step(
                    psi,
                    self.mpo,
                    dt,
                    local_sectors=self.local_sectors,
                    target_sector=self.target_sector,
                    site_qn_maps=site_qn_maps,
                    target_qn=target_qn,
                    block_mpo=block_mpo,
                    krylov_dim=self.krylov_dim,
                    krylov_tol=self.krylov_tol,
                    krylov_method=self.krylov_method,
                    canonicalize=canonicalize,
                    normalize=normalize,
                    copy_state=not self._prepared,
                    moving_environment=moving_environment,
                    env_plan_prefix=self._block_sparse_env_plan_prefix,
                    return_info=True,
                )
            self._prepared = True
            return (out, info) if return_info else out

        projected_in, input_info = self.project(
            psi, normalize=normalize, return_info=True
        )
        if self.integrator == "tdvp2":
            out, step_info = two_site_tdvp_step(
                projected_in,
                self.mpo,
                dt,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
                krylov_dim=self.krylov_dim,
                krylov_tol=self.krylov_tol,
                krylov_method=self.krylov_method,
                diagonal_fast_path=self.diagonal_fast_path,
                canonicalize=canonicalize,
                normalize=normalize,
                return_info=True,
            )
        else:
            out, step_info = one_site_tdvp_step(
                projected_in,
                self.mpo,
                dt,
                krylov_dim=self.krylov_dim,
                krylov_tol=self.krylov_tol,
                krylov_method=self.krylov_method,
                diagonal_fast_path=self.diagonal_fast_path,
                canonicalize=canonicalize,
                normalize=normalize,
                return_info=True,
            )
        out, output_info = self.project(out, normalize=normalize, return_info=True)
        self._prepared = True
        info = dict(step_info)
        info.update(
            {
                "backend": input_info["backend"],
                "projection_backend": input_info["backend"],
                "integrator": self.integrator,
                "target_sector": self.target_sector,
                "input_sector_weight": input_info["sector_weight"],
                "input_discarded_sector_weight": input_info["discarded_sector_weight"],
                "output_sector_weight": output_info["sector_weight"],
                "output_discarded_sector_weight": output_info[
                    "discarded_sector_weight"
                ],
                "max_projector_bond": max(
                    int(input_info.get("max_projector_bond", 1)),
                    int(output_info.get("max_projector_bond", 1)),
                ),
            }
        )
        return (out, info) if return_info else out
