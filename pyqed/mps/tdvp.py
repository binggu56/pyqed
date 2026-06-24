"""TDVP propagation for dense-layout MPS/MPO objects."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal, expm

from pyqed.mps.mps import MPS, MPO

_tdvp_cpp = None
_tdvp_cpp_tried = False


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
        and getattr(_tdvp_cpp, "bond_lanczos", None) is not None
    )


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
        if beta_j <= tol or j + 1 == mmax:
            break
        beta[j] = beta_j
        q_prev = q
        beta_prev = beta_j
        basis[:, j + 1] = trial / beta_j

    if actual_dim == 1:
        small_action = np.array([norm * np.exp(-1j * dt * alpha[0])], dtype=complex)
    else:
        evals, evecs = eigh_tridiagonal(alpha[:actual_dim], beta[:actual_dim - 1])
        e1 = np.zeros(actual_dim, dtype=complex)
        e1[0] = norm
        small_action = evecs @ (np.exp(-1j * dt * evals) * (evecs.T.conj() @ e1))
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


def _standard_mps_factors(psi):
    return [np.asarray(psi._get_std_B(i), dtype=complex).copy() for i in range(psi.L)]


def _update_left_env(left, A, W):
    tmp = np.einsum("amb,bqs->amqs", left, A, optimize=True)
    tmp = np.einsum("mnpq,amqs->anps", W, tmp, optimize=True)
    return np.einsum("apr,anps->rns", A.conj(), tmp, optimize=True)


def _update_right_env(right, A, W):
    tmp = np.einsum("bqs,rns->bqrn", A, right, optimize=True)
    tmp = np.einsum("mnpq,bqrn->bmpr", W, tmp, optimize=True)
    return np.einsum("apr,bmpr->amb", A.conj(), tmp, optimize=True)


def _build_right_envs(factors, mpo):
    nsites = len(factors)
    dtype = np.result_type(*(factors + mpo), complex)
    right_envs = [None] * (nsites + 1)
    right_envs[nsites] = np.ones((1, 1, 1), dtype=dtype)
    for i in range(nsites - 1, -1, -1):
        right_envs[i] = _update_right_env(right_envs[i + 1], factors[i], mpo[i])
    return right_envs


def _build_left_envs(factors, mpo):
    nsites = len(factors)
    dtype = np.result_type(*(factors + mpo), complex)
    left_envs = [None] * (nsites + 1)
    left_envs[0] = np.ones((1, 1, 1), dtype=dtype)
    for i in range(nsites):
        left_envs[i + 1] = _update_left_env(left_envs[i], factors[i], mpo[i])
    return left_envs


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
                left_by_q.append((
                    np.asarray([p for p, _ in terms], dtype=int),
                    np.stack([block for _, block in terms], axis=0),
                ))
            else:
                left_by_q.append(None)

        right_by_s = []
        for terms in raw_right_by_s:
            if terms:
                right_by_s.append((
                    np.asarray([r for r, _ in terms], dtype=int),
                    np.stack([block for _, block in terms], axis=0),
                ))
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
                projected = np.einsum("xab,bd->xad", left_stack, local_block, optimize=True)
                contribution = np.einsum("xad,ycd->xayc", projected, right_stack, optimize=True)
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
                projected = np.einsum("xab,bd->xad", left_stack, local_block, optimize=True)
                contribution = np.einsum("xad,ycd->xayc", projected, right_stack, optimize=True)
                out[:, p_indices[:, None], r_indices[None, :], :] += contribution.transpose(1, 0, 2, 3)
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
):
    shape = theta.shape
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
            out = np.zeros(shape, dtype=np.result_type(local, left, W_left, W_right, right, complex))
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
        estimated_sparse_pairs, dense_pairs = _estimate_sparse_two_site_pairs(W_left, W_right)
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
                    return _apply_sparse_two_site_kernel_vectorized(local, sparse_kernel, shape)
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

    kernel = np.einsum("amb,rms->abrs", left, right, optimize=True)

    def apply_heff(local):
        return np.tensordot(kernel, local, axes=([1, 3], [0, 1]))

    return _krylov_expm_apply(
        center,
        shape,
        apply_heff,
        -dt,
        krylov_dim=krylov_dim,
        tol=krylov_tol,
        method=krylov_method,
    )


def _left_qr(theta):
    left_dim, phys_dim, right_dim = theta.shape
    q, r = np.linalg.qr(theta.reshape(left_dim * phys_dim, right_dim), mode="reduced")
    chi = q.shape[1]
    return q.reshape(left_dim, phys_dim, chi), r


def _right_rq(theta):
    left_dim, phys_dim, right_dim = theta.shape
    q_t, r_t = np.linalg.qr(theta.reshape(left_dim, phys_dim * right_dim).T, mode="reduced")
    chi = q_t.shape[1]
    center = r_t.T
    q = q_t.T.reshape(chi, phys_dim, right_dim)
    return center, q


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
        out = MPS(factors, labels=["lv", "p", "rv"])
        norm2 = out.norm()
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
        half_dt,
        krylov_dim=krylov_dim,
        krylov_tol=krylov_tol,
        krylov_method=krylov_method,
        diagonal_fast_path=diagonal_fast_path,
    )

    right = right_identity
    for i in range(nsites - 1, 0, -1):
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

    out = MPS(factors, labels=["lv", "p", "rv"])
    norm2 = out.norm()
    if normalize:
        out.normalize()
    info = {
        "pre_normalization_norm2": float(np.real(norm2)),
        "pre_normalization_norm": float(np.sqrt(max(float(np.real(norm2)), 0.0))),
    }
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
    right_envs = _build_right_envs(factors, mpo)
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
        )
        factors[i], right_center, discarded = _split_two_site_left(
            theta,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        truncation_error += discarded
        left = _update_left_env(left, factors[i], mpo[i])
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
        )
        left_center, factors[i + 1], discarded = _split_two_site_right(
            theta,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        truncation_error += discarded
        right = _update_right_env(right, factors[i + 1], mpo[i + 1])
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

    out = MPS(factors, labels=["lv", "p", "rv"])
    norm2 = out.norm()
    if normalize:
        out.normalize()
    info = {
        "pre_normalization_norm2": float(np.real(norm2)),
        "pre_normalization_norm": float(np.sqrt(max(float(np.real(norm2)), 0.0))),
        "truncation_error": truncation_error,
    }
    return (out, info) if return_info else out


class TDVPEngine:
    """Small reusable TDVP engine for repeated steps with one fixed MPO."""

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
    ):
        key = str(integrator).lower().replace("_", "-")
        if key in {"tdvp", "tdvp1", "1tdvp", "one-site-tdvp", "1site-tdvp"}:
            self.integrator = "tdvp"
        elif key in {"tdvp2", "2tdvp", "two-site-tdvp", "2site-tdvp"}:
            self.integrator = "tdvp2"
        else:
            raise ValueError("integrator must be 'tdvp' or 'tdvp2'.")
        self.mpo = [np.asarray(w) for w in _mpo_factors(H)]
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.krylov_dim = krylov_dim
        self.krylov_tol = krylov_tol
        self.krylov_method = str(krylov_method).lower().replace("_", "-")
        self.diagonal_fast_path = diagonal_fast_path
        self.sparse_threshold = sparse_threshold
        self.sparse_vectorized = sparse_vectorized
        self.canonicalize_first = bool(canonicalize_first)
        self.canonicalize_each_step = bool(canonicalize_each_step)
        self._prepared = False

    def reset(self):
        self._prepared = False

    def step(self, psi, dt, *, normalize=True, return_info=True):
        canonicalize = self.canonicalize_each_step or (self.canonicalize_first and not self._prepared)
        if self.integrator == "tdvp2":
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
        return (out, info) if return_info else out
