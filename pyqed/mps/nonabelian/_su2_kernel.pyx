# cython: language_level=3
"""Optional Cython entry point for SU(2) block2-style local actions."""

import numpy as np
cimport numpy as cnp


def factorize_rank_coupled_left(object E, object W):
    """
    Build a left qchem factor block ``L[l,k,w,a,b]`` from ``E[x,l,k]``
    and ``W[x,w,a,b]``.
    """

    if np.iscomplexobj(E) or np.iscomplexobj(W):
        return _factorize_rank_coupled_left_complex(E, W)
    return _factorize_rank_coupled_left_real(E, W)


def factorize_rank_coupled_left_real(object E, object W):
    """Real-valued fast path for ``factorize_rank_coupled_left``."""

    return _factorize_rank_coupled_left_real(E, W)


def factorize_rank_coupled_right(object W, object F):
    """
    Build a right qchem factor block ``R[w,q,r,d,c]`` from ``W[w,y,d,c]``
    and ``F[y,q,r]``.
    """

    if np.iscomplexobj(W) or np.iscomplexobj(F):
        return _factorize_rank_coupled_right_complex(W, F)
    return _factorize_rank_coupled_right_real(W, F)


def factorize_rank_coupled_right_real(object W, object F):
    """Real-valued fast path for ``factorize_rank_coupled_right``."""

    return _factorize_rank_coupled_right_real(W, F)


cdef object _factorize_rank_coupled_left_real(object E, object W):
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t l_dim = E_arr.shape[1]
    cdef Py_ssize_t k_dim = E_arr.shape[2]
    cdef Py_ssize_t w_dim = W_arr.shape[1]
    cdef Py_ssize_t a_dim = W_arr.shape[2]
    cdef Py_ssize_t b_dim = W_arr.shape[3]
    cdef cnp.ndarray[cnp.double_t, ndim=5] out
    cdef Py_ssize_t x, l, k, w, a, b
    cdef double e_val
    cdef double w_val

    if W_arr.shape[0] != x_dim:
        return None
    out = np.zeros((l_dim, k_dim, w_dim, a_dim, b_dim), dtype=np.float64)
    for x in range(x_dim):
        for l in range(l_dim):
            for k in range(k_dim):
                e_val = E_arr[x, l, k]
                if e_val == 0.0:
                    continue
                for w in range(w_dim):
                    for a in range(a_dim):
                        for b in range(b_dim):
                            w_val = W_arr[x, w, a, b]
                            if w_val != 0.0:
                                out[l, k, w, a, b] += e_val * w_val
    return out


cdef object _factorize_rank_coupled_right_real(object W, object F):
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.float64)
    cdef Py_ssize_t w_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t d_dim = W_arr.shape[2]
    cdef Py_ssize_t c_dim = W_arr.shape[3]
    cdef Py_ssize_t q_dim = F_arr.shape[1]
    cdef Py_ssize_t r_dim = F_arr.shape[2]
    cdef cnp.ndarray[cnp.double_t, ndim=5] out
    cdef Py_ssize_t w, y, q, r, d, c
    cdef double f_val
    cdef double w_val

    if F_arr.shape[0] != y_dim:
        return None
    out = np.zeros((w_dim, q_dim, r_dim, d_dim, c_dim), dtype=np.float64)
    for w in range(w_dim):
        for y in range(y_dim):
            for d in range(d_dim):
                for c in range(c_dim):
                    w_val = W_arr[w, y, d, c]
                    if w_val == 0.0:
                        continue
                    for q in range(q_dim):
                        for r in range(r_dim):
                            f_val = F_arr[y, q, r]
                            if f_val != 0.0:
                                out[w, q, r, d, c] += w_val * f_val
    return out


cdef object _factorize_rank_coupled_left_complex(object E, object W):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t l_dim = E_arr.shape[1]
    cdef Py_ssize_t k_dim = E_arr.shape[2]
    cdef Py_ssize_t w_dim = W_arr.shape[1]
    cdef Py_ssize_t a_dim = W_arr.shape[2]
    cdef Py_ssize_t b_dim = W_arr.shape[3]
    cdef cnp.ndarray[cnp.complex128_t, ndim=5] out
    cdef Py_ssize_t x, l, k, w, a, b
    cdef double complex e_val
    cdef double complex w_val

    if W_arr.shape[0] != x_dim:
        return None
    out = np.zeros((l_dim, k_dim, w_dim, a_dim, b_dim), dtype=np.complex128)
    for x in range(x_dim):
        for l in range(l_dim):
            for k in range(k_dim):
                e_val = E_arr[x, l, k]
                if e_val == 0.0:
                    continue
                for w in range(w_dim):
                    for a in range(a_dim):
                        for b in range(b_dim):
                            w_val = W_arr[x, w, a, b]
                            if w_val != 0.0:
                                out[l, k, w, a, b] += e_val * w_val
    return out


cdef object _factorize_rank_coupled_right_complex(object W, object F):
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.complex128)
    cdef Py_ssize_t w_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t d_dim = W_arr.shape[2]
    cdef Py_ssize_t c_dim = W_arr.shape[3]
    cdef Py_ssize_t q_dim = F_arr.shape[1]
    cdef Py_ssize_t r_dim = F_arr.shape[2]
    cdef cnp.ndarray[cnp.complex128_t, ndim=5] out
    cdef Py_ssize_t w, y, q, r, d, c
    cdef double complex f_val
    cdef double complex w_val

    if F_arr.shape[0] != y_dim:
        return None
    out = np.zeros((w_dim, q_dim, r_dim, d_dim, c_dim), dtype=np.complex128)
    for w in range(w_dim):
        for y in range(y_dim):
            for d in range(d_dim):
                for c in range(c_dim):
                    w_val = W_arr[w, y, d, c]
                    if w_val == 0.0:
                        continue
                    for q in range(q_dim):
                        for r in range(r_dim):
                            f_val = F_arr[y, q, r]
                            if f_val != 0.0:
                                out[w, q, r, d, c] += w_val * f_val
    return out


def build_su2_qchem_factor_matches(
    object basis_left_ids,
    object basis_p1_ids,
    object basis_p2_ids,
    object basis_right_ids,
    object left_key_map,
    object right_key_map,
    object out_map,
    object left_entry_offsets,
    object left_out_boundary_ids,
    object left_out_physical_ids,
    object left_middle_ids,
    object right_entry_offsets,
    object right_out_boundary_ids,
    object right_out_physical_ids,
    object right_middle_ids,
):
    """
    Build packed SU(2) qchem factor matches from integer table metadata.

    Returns ``(input_indices, output_indices, left_entry_indices,
    right_entry_indices)`` as int64 arrays.  Python owns sector encoding and
    dense lookup-table construction; this helper keeps the heavy nested
    left/right schedule matching out of Python.
    """

    cdef cnp.ndarray[cnp.int64_t, ndim=1] b_l = np.ascontiguousarray(basis_left_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] b_p1 = np.ascontiguousarray(basis_p1_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] b_p2 = np.ascontiguousarray(basis_p2_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] b_r = np.ascontiguousarray(basis_right_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] l_key = np.ascontiguousarray(left_key_map, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] r_key = np.ascontiguousarray(right_key_map, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=4] out_lookup = np.ascontiguousarray(out_map, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_offsets = np.ascontiguousarray(left_entry_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_out_b = np.ascontiguousarray(left_out_boundary_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_out_p = np.ascontiguousarray(left_out_physical_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_mid = np.ascontiguousarray(left_middle_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_offsets = np.ascontiguousarray(right_entry_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_out_b = np.ascontiguousarray(right_out_boundary_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_out_p = np.ascontiguousarray(right_out_physical_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_mid = np.ascontiguousarray(right_middle_ids, dtype=np.int64)
    cdef Py_ssize_t n_basis = b_l.shape[0]
    cdef Py_ssize_t in_idx
    cdef Py_ssize_t lrow
    cdef Py_ssize_t rrow
    cdef Py_ssize_t lstart
    cdef Py_ssize_t lstop
    cdef Py_ssize_t rstart
    cdef Py_ssize_t rstop
    cdef Py_ssize_t lidx
    cdef Py_ssize_t ridx
    cdef long out_idx
    cdef list in_indices = []
    cdef list out_indices = []
    cdef list left_indices = []
    cdef list right_indices = []

    for in_idx in range(n_basis):
        if b_l[in_idx] < 0 or b_p1[in_idx] < 0 or b_p2[in_idx] < 0 or b_r[in_idx] < 0:
            continue
        lrow = l_key[b_l[in_idx], b_p1[in_idx]]
        if lrow < 0:
            continue
        rrow = r_key[b_r[in_idx], b_p2[in_idx]]
        if rrow < 0:
            continue
        lstart = l_offsets[lrow]
        lstop = l_offsets[lrow + 1]
        rstart = r_offsets[rrow]
        rstop = r_offsets[rrow + 1]
        for lidx in range(lstart, lstop):
            for ridx in range(rstart, rstop):
                if l_mid[lidx] != r_mid[ridx]:
                    continue
                out_idx = out_lookup[
                    l_out_b[lidx],
                    l_out_p[lidx],
                    r_out_p[ridx],
                    r_out_b[ridx],
                ]
                if out_idx < 0:
                    continue
                in_indices.append(in_idx)
                out_indices.append(out_idx)
                left_indices.append(lidx)
                right_indices.append(ridx)
    return (
        np.asarray(in_indices, dtype=np.int64),
        np.asarray(out_indices, dtype=np.int64),
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
    )


def build_su2_qchem_parent_blocks_from_matches(
    object basis_shapes,
    object entry_comp_ids,
    object entry_comp_starts,
    object component_dims,
    object in_indices,
    object out_indices,
    object left_indices,
    object right_indices,
    object left_factor_data,
    object left_factor_offsets,
    object left_factor_shape_offsets,
    object left_factor_shapes,
    object left_factor_indices,
    object right_factor_data,
    object right_factor_offsets,
    object right_factor_shape_offsets,
    object right_factor_shapes,
    object right_factor_indices,
):
    """
    Assemble component parent blocks from packed SU(2) qchem matches.

    This keeps the high-volume factor contractions and block updates in Cython.
    Factor payloads are consumed from raw packed arrays owned by the Python
    packed factor tables.
    """

    cdef cnp.ndarray[cnp.int64_t, ndim=2] shapes = np.ascontiguousarray(basis_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] comps = np.ascontiguousarray(entry_comp_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] starts = np.ascontiguousarray(entry_comp_starts, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] comp_dims = np.ascontiguousarray(component_dims, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_arr = np.ascontiguousarray(in_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_arr = np.ascontiguousarray(out_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_arr = np.ascontiguousarray(left_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_arr = np.ascontiguousarray(right_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] l_data = np.ascontiguousarray(left_factor_data, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_offsets = np.ascontiguousarray(left_factor_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_shape_offsets = np.ascontiguousarray(left_factor_shape_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_shapes = np.ascontiguousarray(left_factor_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_factor_indices = np.ascontiguousarray(left_factor_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] r_data = np.ascontiguousarray(right_factor_data, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_offsets = np.ascontiguousarray(right_factor_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_shape_offsets = np.ascontiguousarray(right_factor_shape_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_shapes = np.ascontiguousarray(right_factor_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_factor_indices = np.ascontiguousarray(right_factor_indices, dtype=np.int64)
    cdef Py_ssize_t n_matches = in_arr.shape[0]
    cdef Py_ssize_t match_idx
    cdef Py_ssize_t in_idx
    cdef Py_ssize_t out_idx
    cdef Py_ssize_t left_entry
    cdef Py_ssize_t right_entry
    cdef Py_ssize_t left_factor
    cdef Py_ssize_t right_factor
    cdef Py_ssize_t left_base
    cdef Py_ssize_t right_base
    cdef Py_ssize_t left_shape_base
    cdef Py_ssize_t right_shape_base
    cdef int in_comp
    cdef int out_comp
    cdef int kdim
    cdef int bdim
    cdef int cdim
    cdef int rdim
    cdef int ldim
    cdef int adim
    cdef int ddim
    cdef int qdim
    cdef int wdim
    cdef int l, a, d, q, k, b, c, r, w
    cdef Py_ssize_t row
    cdef Py_ssize_t col
    cdef Py_ssize_t out_start
    cdef Py_ssize_t in_start
    cdef double val
    cdef object key
    cdef object block_obj
    cdef dict blocks = {}
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] block
    cdef Py_ssize_t left_pos
    cdef Py_ssize_t right_pos

    for match_idx in range(n_matches):
        in_idx = in_arr[match_idx]
        out_idx = out_arr[match_idx]
        in_comp = <int>comps[in_idx]
        out_comp = <int>comps[out_idx]
        if in_comp < 0 or out_comp < 0:
            return None
        key = (in_comp, out_comp)
        block_obj = blocks.get(key)
        if block_obj is None:
            block_obj = np.zeros(
                (int(comp_dims[out_comp]), int(comp_dims[in_comp])),
                dtype=np.complex128,
            )
            blocks[key] = block_obj
        block = block_obj

        left_entry = left_arr[match_idx]
        right_entry = right_arr[match_idx]
        if left_entry < 0 or right_entry < 0:
            return None
        left_factor = l_factor_indices[left_entry]
        right_factor = r_factor_indices[right_entry]
        if left_factor < 0 or right_factor < 0:
            return None
        left_base = l_offsets[left_factor]
        right_base = r_offsets[right_factor]
        left_shape_base = l_shape_offsets[left_factor]
        right_shape_base = r_shape_offsets[right_factor]
        if l_shape_offsets[left_factor + 1] - left_shape_base != 5:
            return None
        if r_shape_offsets[right_factor + 1] - right_shape_base != 5:
            return None
        ldim = <int>l_shapes[left_shape_base]
        kdim = <int>l_shapes[left_shape_base + 1]
        wdim = <int>l_shapes[left_shape_base + 2]
        adim = <int>l_shapes[left_shape_base + 3]
        bdim = <int>l_shapes[left_shape_base + 4]
        if <int>r_shapes[right_shape_base] != wdim:
            return None
        qdim = <int>r_shapes[right_shape_base + 1]
        rdim = <int>r_shapes[right_shape_base + 2]
        ddim = <int>r_shapes[right_shape_base + 3]
        cdim = <int>r_shapes[right_shape_base + 4]
        if (
            shapes[in_idx, 0] != kdim
            or shapes[in_idx, 1] != bdim
            or shapes[in_idx, 2] != cdim
            or shapes[in_idx, 3] != rdim
            or shapes[out_idx, 0] != ldim
            or shapes[out_idx, 1] != adim
            or shapes[out_idx, 2] != ddim
            or shapes[out_idx, 3] != qdim
        ):
            return None
        in_start = starts[in_idx]
        out_start = starts[out_idx]
        for l in range(ldim):
            for a in range(adim):
                for d in range(ddim):
                    for q in range(qdim):
                        row = out_start + (((l * adim + a) * ddim + d) * qdim + q)
                        for k in range(kdim):
                            for b in range(bdim):
                                for c in range(cdim):
                                    for r in range(rdim):
                                        val = 0.0
                                        for w in range(wdim):
                                            left_pos = left_base + (((l * kdim + k) * wdim + w) * adim + a) * bdim + b
                                            right_pos = right_base + (((w * qdim + q) * rdim + r) * ddim + d) * cdim + c
                                            val += l_data[left_pos] * r_data[right_pos]
                                        if val != 0.0:
                                            col = in_start + (((k * bdim + b) * cdim + c) * rdim + r)
                                            block[row, col] += val
    return tuple(
        (int(key[0]), int(key[1]), np.ascontiguousarray(block))
        for key, block in sorted(blocks.items())
    )


def contract_rank_coupled_left_scalar_channel(object E, object A, object W, object B):
    """
    Contract ``x=1, y=1`` rank-coupled left boundary update blocks.

    Shapes are ``E[1,i,j]``, ``A[i,p,r]``, ``W[1,1,p,q]``,
    ``B[j,q,s]``; the result has shape ``(1,r,s)``.
    """

    if np.iscomplexobj(E) or np.iscomplexobj(A) or np.iscomplexobj(W) or np.iscomplexobj(B):
        return _contract_rank_coupled_left_scalar_channel_complex(E, A, W, B)
    return _contract_rank_coupled_left_scalar_channel_real(E, A, W, B)


def contract_rank_coupled_right_scalar_channel(object A, object W, object F, object B):
    """
    Contract ``x=1, y=1`` rank-coupled right boundary update blocks.

    Shapes are ``A[i,p,r]``, ``W[1,1,p,q]``, ``F[1,r,s]``,
    ``B[j,q,s]``; the result has shape ``(1,i,j)``.
    """

    if np.iscomplexobj(A) or np.iscomplexobj(W) or np.iscomplexobj(F) or np.iscomplexobj(B):
        return _contract_rank_coupled_right_scalar_channel_complex(A, W, F, B)
    return _contract_rank_coupled_right_scalar_channel_real(A, W, F, B)


def contract_rank_coupled_left_general(object E, object A, object W, object B):
    """
    Contract a small rank-coupled left boundary update block.

    Shapes are ``E[x,i,j]``, ``A[i,p,r]``, ``W[x,y,p,q]``,
    ``B[j,q,s]``; the result has shape ``(y,r,s)``.  This is the packed
    qchem sweep's small-block replacement for the three Python-dispatched
    ``einsum`` calls used by the reference path.
    """

    if np.iscomplexobj(E) or np.iscomplexobj(A) or np.iscomplexobj(W) or np.iscomplexobj(B):
        return _contract_rank_coupled_left_general_complex(E, A, W, B)
    return _contract_rank_coupled_left_general_real(E, A, W, B)


def contract_rank_coupled_right_general(object A, object W, object F, object B):
    """
    Contract a small rank-coupled right boundary update block.

    Shapes are ``A[i,p,r]``, ``W[x,y,p,q]``, ``F[y,r,s]``,
    ``B[j,q,s]``; the result has shape ``(x,i,j)``.
    """

    if np.iscomplexobj(A) or np.iscomplexobj(W) or np.iscomplexobj(F) or np.iscomplexobj(B):
        return _contract_rank_coupled_right_general_complex(A, W, F, B)
    return _contract_rank_coupled_right_general_real(A, W, F, B)


def accumulate_rank_coupled_left_terms(
    object target_blocks,
    object e_blocks,
    object A,
    object B,
    object reduced_terms,
    long max_work,
):
    """
    Accumulate a batch of small left-boundary rank-coupled terms in place.

    ``reduced_terms`` contains ``(left_channel, right_channel, W_block)``
    entries.  The function returns ``True`` when the batch was handled by the
    real-valued Cython path and ``False`` when the Python reference path should
    be used instead.
    """

    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] out_arr
    cdef object term
    cdef object W_obj
    cdef object item
    cdef list validated = []
    cdef Py_ssize_t left_idx
    cdef Py_ssize_t right_idx
    cdef long work

    if np.iscomplexobj(A) or np.iscomplexobj(B):
        return False
    A_arr = np.ascontiguousarray(A, dtype=np.float64)
    B_arr = np.ascontiguousarray(B, dtype=np.float64)
    for term in reduced_terms:
        left_idx = int(term[0])
        right_idx = int(term[1])
        if left_idx < 0 or right_idx < 0:
            return False
        if left_idx >= len(e_blocks) or right_idx >= len(target_blocks):
            continue
        W_obj = term[2]
        if (
            np.iscomplexobj(W_obj)
            or np.iscomplexobj(e_blocks[left_idx])
            or np.iscomplexobj(target_blocks[right_idx])
        ):
            return False
        E_arr = np.ascontiguousarray(e_blocks[left_idx], dtype=np.float64)
        W_arr = np.ascontiguousarray(W_obj, dtype=np.float64)
        out_arr = np.asarray(target_blocks[right_idx], dtype=np.float64)
        work = (
            <long>E_arr.shape[0]
            * <long>W_arr.shape[1]
            * <long>E_arr.shape[1]
            * <long>E_arr.shape[2]
            * <long>A_arr.shape[1]
            * <long>W_arr.shape[3]
            * <long>A_arr.shape[2]
            * <long>B_arr.shape[2]
        )
        if max_work > 0 and work > max_work:
            return False
        if A_arr.shape[0] != E_arr.shape[1] or W_arr.shape[0] != E_arr.shape[0] or W_arr.shape[2] != A_arr.shape[1]:
            return False
        if B_arr.shape[0] != E_arr.shape[2] or B_arr.shape[1] != W_arr.shape[3]:
            return False
        if out_arr.shape[0] != W_arr.shape[1] or out_arr.shape[1] != A_arr.shape[2] or out_arr.shape[2] != B_arr.shape[2]:
            return False
        validated.append((left_idx, right_idx, W_obj))
    for item in validated:
        left_idx = int(item[0])
        right_idx = int(item[1])
        W_obj = item[2]
        E_arr = np.ascontiguousarray(e_blocks[left_idx], dtype=np.float64)
        W_arr = np.ascontiguousarray(W_obj, dtype=np.float64)
        out_arr = np.asarray(target_blocks[right_idx], dtype=np.float64)
        if not _accumulate_rank_coupled_left_general_real(E_arr, A_arr, W_arr, B_arr, out_arr):
            return False
    return True


def accumulate_rank_coupled_right_terms(
    object target_blocks,
    object A,
    object B,
    object f_blocks,
    object reduced_terms,
    long max_work,
):
    """
    Accumulate a batch of small right-boundary rank-coupled terms in place.
    """

    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] out_arr
    cdef object term
    cdef object W_obj
    cdef object item
    cdef list validated = []
    cdef Py_ssize_t left_idx
    cdef Py_ssize_t right_idx
    cdef long work

    if np.iscomplexobj(A) or np.iscomplexobj(B):
        return False
    A_arr = np.ascontiguousarray(A, dtype=np.float64)
    B_arr = np.ascontiguousarray(B, dtype=np.float64)
    for term in reduced_terms:
        left_idx = int(term[0])
        right_idx = int(term[1])
        if left_idx < 0 or right_idx < 0:
            return False
        if right_idx >= len(f_blocks) or left_idx >= len(target_blocks):
            continue
        W_obj = term[2]
        if (
            np.iscomplexobj(W_obj)
            or np.iscomplexobj(f_blocks[right_idx])
            or np.iscomplexobj(target_blocks[left_idx])
        ):
            return False
        F_arr = np.ascontiguousarray(f_blocks[right_idx], dtype=np.float64)
        W_arr = np.ascontiguousarray(W_obj, dtype=np.float64)
        out_arr = np.asarray(target_blocks[left_idx], dtype=np.float64)
        work = (
            <long>W_arr.shape[0]
            * <long>F_arr.shape[0]
            * <long>A_arr.shape[0]
            * <long>B_arr.shape[0]
            * <long>A_arr.shape[1]
            * <long>W_arr.shape[3]
            * <long>A_arr.shape[2]
            * <long>F_arr.shape[2]
        )
        if max_work > 0 and work > max_work:
            return False
        if W_arr.shape[2] != A_arr.shape[1] or F_arr.shape[0] != W_arr.shape[1] or F_arr.shape[1] != A_arr.shape[2]:
            return False
        if B_arr.shape[1] != W_arr.shape[3] or B_arr.shape[2] != F_arr.shape[2]:
            return False
        if out_arr.shape[0] != W_arr.shape[0] or out_arr.shape[1] != A_arr.shape[0] or out_arr.shape[2] != B_arr.shape[0]:
            return False
        validated.append((left_idx, right_idx, W_obj))
    for item in validated:
        left_idx = int(item[0])
        right_idx = int(item[1])
        W_obj = item[2]
        F_arr = np.ascontiguousarray(f_blocks[right_idx], dtype=np.float64)
        W_arr = np.ascontiguousarray(W_obj, dtype=np.float64)
        out_arr = np.asarray(target_blocks[left_idx], dtype=np.float64)
        if not _accumulate_rank_coupled_right_general_real(A_arr, W_arr, F_arr, B_arr, out_arr):
            return False
    return True


def accumulate_rank_coupled_left_real_terms(
    object target_blocks,
    object e_blocks,
    object A,
    object B,
    object left_indices,
    object right_indices,
    object w_blocks,
    long max_work,
):
    """
    Accumulate prevalidated real left-boundary rank-coupled terms in place.

    This entry point is used by the qchem sweep after Python has already
    selected real-valued contiguous blocks.  It intentionally avoids the
    per-term ``np.iscomplexobj`` and conversion work in the generic fallback.
    """

    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] out_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_arr = np.asarray(left_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_arr = np.asarray(right_indices, dtype=np.int64)
    cdef Py_ssize_t n_terms = left_arr.shape[0]
    cdef Py_ssize_t t
    cdef Py_ssize_t left_idx
    cdef Py_ssize_t right_idx
    cdef long work

    if right_arr.shape[0] != n_terms or len(w_blocks) != n_terms:
        return False
    try:
        A_arr = A
        B_arr = B
    except (TypeError, ValueError):
        return False
    for t in range(n_terms):
        left_idx = <Py_ssize_t>left_arr[t]
        right_idx = <Py_ssize_t>right_arr[t]
        if left_idx < 0 or right_idx < 0:
            return False
        if left_idx >= len(e_blocks) or right_idx >= len(target_blocks):
            continue
        try:
            E_arr = e_blocks[left_idx]
            W_arr = w_blocks[t]
            out_arr = target_blocks[right_idx]
        except (TypeError, ValueError):
            return False
        work = (
            <long>E_arr.shape[0]
            * <long>W_arr.shape[1]
            * <long>E_arr.shape[1]
            * <long>E_arr.shape[2]
            * <long>A_arr.shape[1]
            * <long>W_arr.shape[3]
            * <long>A_arr.shape[2]
            * <long>B_arr.shape[2]
        )
        if max_work > 0 and work > max_work:
            return False
        if A_arr.shape[0] != E_arr.shape[1] or W_arr.shape[0] != E_arr.shape[0] or W_arr.shape[2] != A_arr.shape[1]:
            return False
        if B_arr.shape[0] != E_arr.shape[2] or B_arr.shape[1] != W_arr.shape[3]:
            return False
        if out_arr.shape[0] != W_arr.shape[1] or out_arr.shape[1] != A_arr.shape[2] or out_arr.shape[2] != B_arr.shape[2]:
            return False
    for t in range(n_terms):
        left_idx = <Py_ssize_t>left_arr[t]
        right_idx = <Py_ssize_t>right_arr[t]
        if left_idx >= len(e_blocks) or right_idx >= len(target_blocks):
            continue
        E_arr = e_blocks[left_idx]
        W_arr = w_blocks[t]
        out_arr = target_blocks[right_idx]
        if not _accumulate_rank_coupled_left_general_real(E_arr, A_arr, W_arr, B_arr, out_arr):
            return False
    return True


def accumulate_rank_coupled_right_real_terms(
    object target_blocks,
    object A,
    object B,
    object f_blocks,
    object left_indices,
    object right_indices,
    object w_blocks,
    long max_work,
):
    """Accumulate prevalidated real right-boundary rank-coupled terms."""

    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] out_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_arr = np.asarray(left_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_arr = np.asarray(right_indices, dtype=np.int64)
    cdef Py_ssize_t n_terms = left_arr.shape[0]
    cdef Py_ssize_t t
    cdef Py_ssize_t left_idx
    cdef Py_ssize_t right_idx
    cdef long work

    if right_arr.shape[0] != n_terms or len(w_blocks) != n_terms:
        return False
    try:
        A_arr = A
        B_arr = B
    except (TypeError, ValueError):
        return False
    for t in range(n_terms):
        left_idx = <Py_ssize_t>left_arr[t]
        right_idx = <Py_ssize_t>right_arr[t]
        if left_idx < 0 or right_idx < 0:
            return False
        if right_idx >= len(f_blocks) or left_idx >= len(target_blocks):
            continue
        try:
            F_arr = f_blocks[right_idx]
            W_arr = w_blocks[t]
            out_arr = target_blocks[left_idx]
        except (TypeError, ValueError):
            return False
        work = (
            <long>W_arr.shape[0]
            * <long>F_arr.shape[0]
            * <long>A_arr.shape[0]
            * <long>B_arr.shape[0]
            * <long>A_arr.shape[1]
            * <long>W_arr.shape[3]
            * <long>A_arr.shape[2]
            * <long>F_arr.shape[2]
        )
        if max_work > 0 and work > max_work:
            return False
        if W_arr.shape[2] != A_arr.shape[1] or F_arr.shape[0] != W_arr.shape[1] or F_arr.shape[1] != A_arr.shape[2]:
            return False
        if B_arr.shape[1] != W_arr.shape[3] or B_arr.shape[2] != F_arr.shape[2]:
            return False
        if out_arr.shape[0] != W_arr.shape[0] or out_arr.shape[1] != A_arr.shape[0] or out_arr.shape[2] != B_arr.shape[0]:
            return False
    for t in range(n_terms):
        left_idx = <Py_ssize_t>left_arr[t]
        right_idx = <Py_ssize_t>right_arr[t]
        if right_idx >= len(f_blocks) or left_idx >= len(target_blocks):
            continue
        F_arr = f_blocks[right_idx]
        W_arr = w_blocks[t]
        out_arr = target_blocks[left_idx]
        if not _accumulate_rank_coupled_right_general_real(A_arr, W_arr, F_arr, B_arr, out_arr):
            return False
    return True


cdef bint _apply_parent_block_batch_impl(
    object blocks,
    object in_comps,
    object out_comps,
    object parent_inputs,
    object parent_outputs,
    bint apply,
):
    """
    Apply same-shape parent component blocks without per-matvec stack buffers.

    ``blocks[n]`` maps ``parent_inputs[in_comps[n]]`` into
    ``parent_outputs[out_comps[n]]``.  This is the hot path for the packed SU2
    qchem action when direct parent blocks are present.
    """

    cdef cnp.ndarray[cnp.complex128_t, ndim=3] block_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_arr = np.asarray(in_comps, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_arr = np.asarray(out_comps, dtype=np.int64)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] inp
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] out
    cdef Py_ssize_t n_terms
    cdef Py_ssize_t t
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t rows
    cdef Py_ssize_t cols
    cdef Py_ssize_t in_idx
    cdef Py_ssize_t out_idx
    cdef double complex acc

    try:
        block_arr = blocks
    except (TypeError, ValueError):
        return False
    n_terms = block_arr.shape[0]
    if in_arr.shape[0] != n_terms or out_arr.shape[0] != n_terms:
        return False
    rows = block_arr.shape[1]
    cols = block_arr.shape[2]
    for t in range(n_terms):
        in_idx = <Py_ssize_t>in_arr[t]
        out_idx = <Py_ssize_t>out_arr[t]
        if in_idx < 0 or out_idx < 0:
            return False
        if in_idx >= len(parent_inputs) or out_idx >= len(parent_outputs):
            return False
        try:
            inp = parent_inputs[in_idx]
            out = parent_outputs[out_idx]
        except (TypeError, ValueError):
            return False
        if inp.shape[0] != cols or out.shape[0] != rows:
            return False
        if not apply:
            continue
        for i in range(rows):
            acc = 0.0
            for j in range(cols):
                acc += block_arr[t, i, j] * inp[j]
            out[i] += acc
    return True


def apply_parent_block_batch(
    object blocks,
    object in_comps,
    object out_comps,
    object parent_inputs,
    object parent_outputs,
):
    """
    Apply one same-shape parent component block batch.

    Returns ``False`` before mutating outputs when the layout is unsupported.
    """

    if not _apply_parent_block_batch_impl(
        blocks,
        in_comps,
        out_comps,
        parent_inputs,
        parent_outputs,
        False,
    ):
        return False
    return _apply_parent_block_batch_impl(
        blocks,
        in_comps,
        out_comps,
        parent_inputs,
        parent_outputs,
        True,
    )


def apply_parent_block_batches(
    object batches,
    object parent_inputs,
    object parent_outputs,
):
    """
    Apply all parent-block batches in one Cython call.

    ``batches`` is the tuple of Python ``_ParentBlockBatch`` objects owned by
    ``SU2LocalAction``.  Validation is separated from mutation so Python can
    safely fall back to its reference path when any batch is unsupported.
    """

    cdef object batch
    for batch in batches:
        if not _apply_parent_block_batch_impl(
            batch.blocks,
            batch.in_comps,
            batch.out_comps,
            parent_inputs,
            parent_outputs,
            False,
        ):
            return False
    for batch in batches:
        if not _apply_parent_block_batch_impl(
            batch.blocks,
            batch.in_comps,
            batch.out_comps,
            parent_inputs,
            parent_outputs,
            True,
        ):
            return False
    return True


cdef object _contract_rank_coupled_left_scalar_channel_real(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.float64)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t q_dim = B_arr.shape[1]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((1, r_dim, s_dim), dtype=np.float64)
    cdef Py_ssize_t i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double e_val

    for p in range(p_dim):
        for q in range(q_dim):
            coeff = W_arr[0, 0, p, q]
            if coeff == 0.0:
                continue
            for i in range(i_dim):
                for r in range(r_dim):
                    a_val = A_arr[i, p, r] * coeff
                    if a_val == 0.0:
                        continue
                    for j in range(j_dim):
                        e_val = E_arr[0, i, j] * a_val
                        if e_val == 0.0:
                            continue
                        for s in range(s_dim):
                            out[0, r, s] += e_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_right_scalar_channel_real(object A, object W, object F, object B):
    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.float64)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t q_dim = B_arr.shape[1]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((1, i_dim, j_dim), dtype=np.float64)
    cdef Py_ssize_t i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double f_val

    for p in range(p_dim):
        for q in range(q_dim):
            coeff = W_arr[0, 0, p, q]
            if coeff == 0.0:
                continue
            for i in range(i_dim):
                for r in range(r_dim):
                    a_val = A_arr[i, p, r] * coeff
                    if a_val == 0.0:
                        continue
                    for s in range(s_dim):
                        f_val = F_arr[0, r, s] * a_val
                        if f_val == 0.0:
                            continue
                        for j in range(j_dim):
                            out[0, i, j] += f_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_left_general_real(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.float64)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t i_dim = E_arr.shape[1]
    cdef Py_ssize_t j_dim = E_arr.shape[2]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.double_t, ndim=3] out
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double e_val

    if A_arr.shape[0] != i_dim or W_arr.shape[0] != x_dim or W_arr.shape[2] != p_dim:
        return None
    if B_arr.shape[0] != j_dim or B_arr.shape[1] != q_dim:
        return None
    out = np.zeros((y_dim, r_dim, s_dim), dtype=np.float64)
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for j in range(j_dim):
                                e_val = E_arr[x, i, j] * a_val
                                if e_val == 0.0:
                                    continue
                                for s in range(s_dim):
                                    out[y, r, s] += e_val * B_arr[j, q, s]
    return out


cdef bint _accumulate_rank_coupled_left_general_real(
    cnp.ndarray[cnp.double_t, ndim=3] E_arr,
    cnp.ndarray[cnp.double_t, ndim=3] A_arr,
    cnp.ndarray[cnp.double_t, ndim=4] W_arr,
    cnp.ndarray[cnp.double_t, ndim=3] B_arr,
    cnp.ndarray[cnp.double_t, ndim=3] out,
):
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t i_dim = E_arr.shape[1]
    cdef Py_ssize_t j_dim = E_arr.shape[2]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double e_val

    if A_arr.shape[0] != i_dim or W_arr.shape[0] != x_dim or W_arr.shape[2] != p_dim:
        return False
    if B_arr.shape[0] != j_dim or B_arr.shape[1] != q_dim:
        return False
    if out.shape[0] != y_dim or out.shape[1] != r_dim or out.shape[2] != s_dim:
        return False
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for j in range(j_dim):
                                e_val = E_arr[x, i, j] * a_val
                                if e_val == 0.0:
                                    continue
                                for s in range(s_dim):
                                    out[y, r, s] += e_val * B_arr[j, q, s]
    return True


cdef object _contract_rank_coupled_right_general_real(object A, object W, object F, object B):
    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.float64)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t x_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = F_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=3] out
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double f_val

    if W_arr.shape[2] != p_dim or F_arr.shape[0] != y_dim or F_arr.shape[1] != r_dim:
        return None
    if B_arr.shape[1] != q_dim or B_arr.shape[2] != s_dim:
        return None
    out = np.zeros((x_dim, i_dim, j_dim), dtype=np.float64)
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for s in range(s_dim):
                                f_val = F_arr[y, r, s] * a_val
                                if f_val == 0.0:
                                    continue
                                for j in range(j_dim):
                                    out[x, i, j] += f_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_left_scalar_channel_complex(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t q_dim = B_arr.shape[1]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out = np.zeros((1, r_dim, s_dim), dtype=np.complex128)
    cdef Py_ssize_t i, j, p, q, r, s
    cdef double complex coeff
    cdef double complex a_val
    cdef double complex e_val

    for p in range(p_dim):
        for q in range(q_dim):
            coeff = W_arr[0, 0, p, q]
            if coeff == 0.0:
                continue
            for i in range(i_dim):
                for r in range(r_dim):
                    a_val = A_arr[i, p, r] * coeff
                    if a_val == 0.0:
                        continue
                    for j in range(j_dim):
                        e_val = E_arr[0, i, j] * a_val
                        if e_val == 0.0:
                            continue
                        for s in range(s_dim):
                            out[0, r, s] += e_val * B_arr[j, q, s]
    return out


cdef bint _accumulate_rank_coupled_right_general_real(
    cnp.ndarray[cnp.double_t, ndim=3] A_arr,
    cnp.ndarray[cnp.double_t, ndim=4] W_arr,
    cnp.ndarray[cnp.double_t, ndim=3] F_arr,
    cnp.ndarray[cnp.double_t, ndim=3] B_arr,
    cnp.ndarray[cnp.double_t, ndim=3] out,
):
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t x_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = F_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double f_val

    if W_arr.shape[2] != p_dim or F_arr.shape[0] != y_dim or F_arr.shape[1] != r_dim:
        return False
    if B_arr.shape[1] != q_dim or B_arr.shape[2] != s_dim:
        return False
    if out.shape[0] != x_dim or out.shape[1] != i_dim or out.shape[2] != j_dim:
        return False
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for s in range(s_dim):
                                f_val = F_arr[y, r, s] * a_val
                                if f_val == 0.0:
                                    continue
                                for j in range(j_dim):
                                    out[x, i, j] += f_val * B_arr[j, q, s]
    return True


cdef object _contract_rank_coupled_left_general_complex(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t i_dim = E_arr.shape[1]
    cdef Py_ssize_t j_dim = E_arr.shape[2]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double complex coeff
    cdef double complex a_val
    cdef double complex e_val

    if A_arr.shape[0] != i_dim or W_arr.shape[0] != x_dim or W_arr.shape[2] != p_dim:
        return None
    if B_arr.shape[0] != j_dim or B_arr.shape[1] != q_dim:
        return None
    out = np.zeros((y_dim, r_dim, s_dim), dtype=np.complex128)
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for j in range(j_dim):
                                e_val = E_arr[x, i, j] * a_val
                                if e_val == 0.0:
                                    continue
                                for s in range(s_dim):
                                    out[y, r, s] += e_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_right_general_complex(object A, object W, object F, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t x_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = F_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double complex coeff
    cdef double complex a_val
    cdef double complex f_val

    if W_arr.shape[2] != p_dim or F_arr.shape[0] != y_dim or F_arr.shape[1] != r_dim:
        return None
    if B_arr.shape[1] != q_dim or B_arr.shape[2] != s_dim:
        return None
    out = np.zeros((x_dim, i_dim, j_dim), dtype=np.complex128)
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for s in range(s_dim):
                                f_val = F_arr[y, r, s] * a_val
                                if f_val == 0.0:
                                    continue
                                for j in range(j_dim):
                                    out[x, i, j] += f_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_right_scalar_channel_complex(object A, object W, object F, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t q_dim = B_arr.shape[1]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out = np.zeros((1, i_dim, j_dim), dtype=np.complex128)
    cdef Py_ssize_t i, j, p, q, r, s
    cdef double complex coeff
    cdef double complex a_val
    cdef double complex f_val

    for p in range(p_dim):
        for q in range(q_dim):
            coeff = W_arr[0, 0, p, q]
            if coeff == 0.0:
                continue
            for i in range(i_dim):
                for r in range(r_dim):
                    a_val = A_arr[i, p, r] * coeff
                    if a_val == 0.0:
                        continue
                    for s in range(s_dim):
                        f_val = F_arr[0, r, s] * a_val
                        if f_val == 0.0:
                            continue
                        for j in range(j_dim):
                            out[0, i, j] += f_val * B_arr[j, q, s]
    return out


def apply_su2_block2_action(object plan, object x, object out):
    """
    Apply a packed SU(2) local action into ``out``.

    This entry keeps the Python ``SU2LocalAction`` object as the plan ABI while
    executing the packed local-action schedule directly.
    """

    cdef Py_ssize_t idx
    cdef object vector = np.asarray(x, dtype=complex).reshape(plan.dim)
    cdef list parent_inputs = []
    cdef list parent_outputs = []
    cdef object transform
    cdef object slc
    cdef object batch
    cdef object entries
    cdef object entry
    cdef object input_mats
    cdef object tmp
    cdef object tmp_mats
    cdef object contribs
    cdef int ldim
    cdef int adim
    cdef int ddim
    cdef int qdim
    cdef int in_comp
    cdef int out_comp
    cdef object block_in

    out[...] = 0.0
    for idx in range(len(plan.transforms)):
        transform = plan.transforms[idx]
        slc = plan.orth_slices[idx]
        parent_inputs.append(transform @ vector[slc])
        parent_outputs.append(np.zeros(plan.parent_dims[idx], dtype=complex))

    for entry in plan.parent_blocks:
        in_comp = int(entry[0])
        out_comp = int(entry[1])
        parent_outputs[out_comp] += entry[2] @ parent_inputs[in_comp]

    for batch in plan.batch_plans:
        entries = batch.entries
        input_mats = np.stack(
            [
                parent_inputs[int(entry.in_comp)][entry.in_slice]
                .reshape(entry.input_entry.shape)
                .reshape(
                    int(np.prod(entry.input_entry.shape[:2], dtype=int)),
                    int(np.prod(entry.input_entry.shape[2:], dtype=int)),
                )
                for entry in entries
            ],
            axis=0,
        )
        tmp = np.matmul(batch.left_mats, input_mats).reshape(
            (len(entries),) + tuple(batch.tmp_shape)
        )
        ldim, adim, ddim, qdim = (int(dim) for dim in batch.output_shape)
        tmp_mats = np.ascontiguousarray(
            tmp.transpose(0, 2, 4, 1, 3, 6, 5).reshape(
                len(entries),
                ldim * adim,
                -1,
            )
        )
        contribs = np.matmul(tmp_mats, batch.right_mats).reshape(
            len(entries),
            ldim * adim * ddim * qdim,
        )
        for idx, entry in enumerate(entries):
            parent_outputs[int(entry.out_comp)][entry.out_slice] += contribs[idx]

    for entry in plan.single_entries:
        in_comp = int(entry.in_comp)
        out_comp = int(entry.out_comp)
        block_in = parent_inputs[in_comp][entry.in_slice].reshape(
            entry.input_entry.shape
        )
        parent_outputs[out_comp][entry.out_slice] += entry.apply_block(block_in)

    for idx in range(len(plan.transforms)):
        transform = plan.transforms[idx]
        out[plan.orth_slices[idx]] = transform.conj().T @ parent_outputs[idx]
    return out


def build_component_parent_blocks(object plan, object component_dims):
    """
    Build component parent blocks from a component-direct factorized plan.

    This mirrors ``DirectOrthonormalFactorizedTable._build_component_parent_blocks``
    while keeping the outer loop and dictionary assembly in the extension.
    """

    cdef object blocks = {}
    cdef object entry
    cdef object term
    cdef object key
    cdef object block
    cdef object kernel
    cdef int in_comp
    cdef int out_comp
    cdef object in_slice
    cdef object out_slice

    if plan is None:
        return None
    for entry in plan:
        in_comp = int(entry[0])
        out_comp = int(entry[1])
        in_slice = entry[2]
        out_slice = entry[3]
        term = entry[4]
        key = (in_comp, out_comp)
        block = blocks.get(key)
        if block is None:
            block = np.zeros(
                (int(component_dims[out_comp]), int(component_dims[in_comp])),
                dtype=complex,
            )
            blocks[key] = block
        kernel = term.kernel_matrix(
            term.input_entry.shape,
            max_elements=max(
                int(term.input_entry.size) * int(term.output_entry.size),
                1,
            ),
        )
        if kernel is None:
            return None
        block[out_slice, in_slice] += np.asarray(kernel, dtype=complex)
    return tuple(
        (int(key[0]), int(key[1]), np.ascontiguousarray(block))
        for key, block in sorted(blocks.items())
    )


def project_component_orthonormal_blocks(
    object parent_blocks,
    object transforms,
    long max_elements,
):
    """
    Project parent component blocks into orthonormal component coordinates.
    """

    cdef object entry
    cdef object X_in
    cdef object X_out
    cdef object parent_block
    cdef object transformed
    cdef list out = []
    cdef long total_elements = 0
    cdef int in_comp
    cdef int out_comp

    if parent_blocks is None:
        return None
    for entry in parent_blocks:
        in_comp = int(entry[0])
        out_comp = int(entry[1])
        total_elements += (
            int(transforms[out_comp].shape[1])
            * int(transforms[in_comp].shape[1])
        )
        if total_elements > max_elements:
            return None
    for entry in parent_blocks:
        in_comp = int(entry[0])
        out_comp = int(entry[1])
        parent_block = entry[2]
        X_in = np.asarray(transforms[in_comp], dtype=complex)
        X_out = np.asarray(transforms[out_comp], dtype=complex)
        transformed = X_out.conj().T @ np.asarray(parent_block, dtype=complex) @ X_in
        if np.linalg.norm(transformed.reshape(-1)) > 1.0e-15:
            out.append((in_comp, out_comp, np.ascontiguousarray(transformed)))
    return tuple(out)
