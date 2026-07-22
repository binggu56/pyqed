# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp


def packed_left_boundary_block(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t i_dim = E_arr.shape[1]
    cdef Py_ssize_t j_dim = E_arr.shape[2]
    cdef Py_ssize_t a_dim = A_arr.shape[1]
    cdef Py_ssize_t u_dim = A_arr.shape[2]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t v_dim = W_arr.shape[3]
    cdef Py_ssize_t b_dim = B_arr.shape[1]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    cdef Py_ssize_t x, i, j, a, u, y, v, b
    cdef cnp.complex128_t ea
    cdef cnp.complex128_t eaw

    if A_arr.shape[0] != i_dim or W_arr.shape[0] != x_dim or W_arr.shape[2] != u_dim:
        raise ValueError("left boundary block shapes are incompatible")
    if B_arr.shape[0] != j_dim or B_arr.shape[2] != v_dim:
        raise ValueError("left boundary block shapes are incompatible")
    out = np.zeros((y_dim, a_dim, b_dim), dtype=np.complex128)
    for x in range(x_dim):
        for i in range(i_dim):
            for j in range(j_dim):
                for a in range(a_dim):
                    for u in range(u_dim):
                        ea = E_arr[x, i, j] * A_arr[i, a, u]
                        for y in range(y_dim):
                            for v in range(v_dim):
                                eaw = ea * W_arr[x, y, u, v]
                                for b in range(b_dim):
                                    out[y, a, b] += eaw * B_arr[j, b, v]
    return out


def packed_right_boundary_block(object A, object F, object W, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t a_dim = A_arr.shape[0]
    cdef Py_ssize_t i_dim = A_arr.shape[1]
    cdef Py_ssize_t p_dim = A_arr.shape[2]
    cdef Py_ssize_t x_dim = F_arr.shape[0]
    cdef Py_ssize_t j_dim = F_arr.shape[2]
    cdef Py_ssize_t y_dim = W_arr.shape[0]
    cdef Py_ssize_t v_dim = W_arr.shape[3]
    cdef Py_ssize_t b_dim = B_arr.shape[0]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    cdef Py_ssize_t x, i, j, a, p, y, v, b
    cdef cnp.complex128_t af
    cdef cnp.complex128_t afw

    if F_arr.shape[1] != i_dim or W_arr.shape[1] != x_dim or W_arr.shape[2] != p_dim:
        raise ValueError("right boundary block shapes are incompatible")
    if B_arr.shape[1] != j_dim or B_arr.shape[2] != v_dim:
        raise ValueError("right boundary block shapes are incompatible")
    out = np.zeros((y_dim, a_dim, b_dim), dtype=np.complex128)
    for x in range(x_dim):
        for i in range(i_dim):
            for j in range(j_dim):
                for a in range(a_dim):
                    for p in range(p_dim):
                        af = A_arr[a, i, p] * F_arr[x, i, j]
                        for y in range(y_dim):
                            for v in range(v_dim):
                                afw = af * W_arr[y, x, p, v]
                                for b in range(b_dim):
                                    out[y, a, b] += afw * B_arr[b, j, v]
    return out


def packed_left_identity_boundary_block(object A, object E, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t i_dim = E_arr.shape[1]
    cdef Py_ssize_t j_dim = E_arr.shape[2]
    cdef Py_ssize_t a_dim = A_arr.shape[1]
    cdef Py_ssize_t u_dim = A_arr.shape[2]
    cdef Py_ssize_t b_dim = B_arr.shape[1]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    cdef Py_ssize_t x, i, j, a, u, b
    cdef cnp.complex128_t ea

    if A_arr.shape[0] != i_dim:
        raise ValueError("left identity boundary block shapes are incompatible")
    if B_arr.shape[0] != j_dim or B_arr.shape[2] != u_dim:
        raise ValueError("left identity boundary block shapes are incompatible")
    out = np.zeros((x_dim, a_dim, b_dim), dtype=np.complex128)
    for x in range(x_dim):
        for i in range(i_dim):
            for j in range(j_dim):
                for a in range(a_dim):
                    for u in range(u_dim):
                        ea = E_arr[x, i, j] * A_arr[i, a, u]
                        for b in range(b_dim):
                            out[x, a, b] += ea * B_arr[j, b, u]
    return out


def packed_right_identity_boundary_block(object A, object F, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t a_dim = A_arr.shape[0]
    cdef Py_ssize_t i_dim = A_arr.shape[1]
    cdef Py_ssize_t p_dim = A_arr.shape[2]
    cdef Py_ssize_t x_dim = F_arr.shape[0]
    cdef Py_ssize_t j_dim = F_arr.shape[2]
    cdef Py_ssize_t b_dim = B_arr.shape[0]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    cdef Py_ssize_t x, i, j, a, p, b
    cdef cnp.complex128_t af

    if F_arr.shape[1] != i_dim:
        raise ValueError("right identity boundary block shapes are incompatible")
    if B_arr.shape[1] != j_dim or B_arr.shape[2] != p_dim:
        raise ValueError("right identity boundary block shapes are incompatible")
    out = np.zeros((x_dim, a_dim, b_dim), dtype=np.complex128)
    for x in range(x_dim):
        for i in range(i_dim):
            for j in range(j_dim):
                for a in range(a_dim):
                    for p in range(p_dim):
                        af = A_arr[a, i, p] * F_arr[x, i, j]
                        for b in range(b_dim):
                            out[x, a, b] += af * B_arr[b, j, p]
    return out


def batched_matrix_chain_e_a_accum(
    cnp.ndarray[cnp.complex128_t, ndim=4] e_stack,
    cnp.ndarray[cnp.complex128_t, ndim=5] a_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] a_pos,
    cnp.ndarray[cnp.complex128_t, ndim=6] out_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
):
    cdef Py_ssize_t batch = e_stack.shape[0]
    cdef Py_ssize_t na = e_stack.shape[1]
    cdef Py_ssize_t ni = e_stack.shape[2]
    cdef Py_ssize_t nj = e_stack.shape[3]
    cdef Py_ssize_t nk = a_buf.shape[2]
    cdef Py_ssize_t nx = a_buf.shape[3]
    cdef Py_ssize_t ny = a_buf.shape[4]
    cdef Py_ssize_t entry, ai, oi, a, i, j, k, x, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ai = a_pos[entry]
        oi = out_pos[entry]
        for a in range(na):
            for i in range(ni):
                for k in range(nk):
                    for x in range(nx):
                        for y in range(ny):
                            total = 0.0
                            for j in range(nj):
                                total += e_stack[entry, a, i, j] * a_buf[ai, j, k, x, y]
                            out_buf[oi, a, i, k, x, y] += total


def batched_matrix_chain_r_w_accum(
    cnp.ndarray[cnp.complex128_t, ndim=6] r_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] r_pos,
    cnp.ndarray[cnp.complex128_t, ndim=5] w_stack,
    cnp.ndarray[cnp.complex128_t, ndim=6] out_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
):
    cdef Py_ssize_t batch = w_stack.shape[0]
    cdef Py_ssize_t na = w_stack.shape[1]
    cdef Py_ssize_t nb = w_stack.shape[2]
    cdef Py_ssize_t nu = w_stack.shape[3]
    cdef Py_ssize_t nx = w_stack.shape[4]
    cdef Py_ssize_t ni = r_buf.shape[2]
    cdef Py_ssize_t nk = r_buf.shape[3]
    cdef Py_ssize_t ny = r_buf.shape[5]
    cdef Py_ssize_t entry, ri, oi, a, b, i, k, u, x, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ri = r_pos[entry]
        oi = out_pos[entry]
        for i in range(ni):
            for k in range(nk):
                for y in range(ny):
                    for b in range(nb):
                        for u in range(nu):
                            total = 0.0
                            for a in range(na):
                                for x in range(nx):
                                    total += r_buf[ri, a, i, k, x, y] * w_stack[entry, a, b, u, x]
                            out_buf[oi, i, k, y, b, u] += total


def batched_matrix_chain_t2_w_accum(
    cnp.ndarray[cnp.complex128_t, ndim=6] t2_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_pos,
    cnp.ndarray[cnp.complex128_t, ndim=5] w_stack,
    cnp.ndarray[cnp.complex128_t, ndim=6] out_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
):
    cdef Py_ssize_t batch = w_stack.shape[0]
    cdef Py_ssize_t nb = w_stack.shape[1]
    cdef Py_ssize_t nc = w_stack.shape[2]
    cdef Py_ssize_t nv = w_stack.shape[3]
    cdef Py_ssize_t ny = w_stack.shape[4]
    cdef Py_ssize_t ni = t2_buf.shape[1]
    cdef Py_ssize_t nk = t2_buf.shape[2]
    cdef Py_ssize_t nu = t2_buf.shape[5]
    cdef Py_ssize_t entry, ti, oi, b, c, i, k, u, v, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ti = t2_pos[entry]
        oi = out_pos[entry]
        for i in range(ni):
            for k in range(nk):
                for u in range(nu):
                    for c in range(nc):
                        for v in range(nv):
                            total = 0.0
                            for b in range(nb):
                                for y in range(ny):
                                    total += t2_buf[ti, i, k, y, b, u] * w_stack[entry, b, c, v, y]
                            out_buf[oi, i, k, u, c, v] += total


def batched_matrix_chain_t3_f_accum(
    cnp.ndarray[cnp.complex128_t, ndim=6] t3_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_pos,
    cnp.ndarray[cnp.complex128_t, ndim=4] f_stack,
    cnp.ndarray[cnp.complex128_t, ndim=5] out_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
):
    cdef Py_ssize_t batch = f_stack.shape[0]
    cdef Py_ssize_t nc = f_stack.shape[1]
    cdef Py_ssize_t nl = f_stack.shape[2]
    cdef Py_ssize_t nk = f_stack.shape[3]
    cdef Py_ssize_t ni = t3_buf.shape[1]
    cdef Py_ssize_t nu = t3_buf.shape[3]
    cdef Py_ssize_t nv = t3_buf.shape[5]
    cdef Py_ssize_t entry, ti, oi, c, i, k, l, u, v
    cdef cnp.complex128_t total

    for entry in range(batch):
        ti = t3_pos[entry]
        oi = out_pos[entry]
        for i in range(ni):
            for l in range(nl):
                for u in range(nu):
                    for v in range(nv):
                        total = 0.0
                        for c in range(nc):
                            for k in range(nk):
                                total += t3_buf[ti, i, k, u, c, v] * f_stack[entry, c, l, k]
                        out_buf[oi, i, l, u, v] += total


cdef void _e_a_accum(
    cnp.ndarray[cnp.complex128_t, ndim=4] e_stack,
    cnp.ndarray[cnp.complex128_t, ndim=5] a_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] a_pos,
    cnp.ndarray[cnp.complex128_t, ndim=6] out_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
):
    cdef Py_ssize_t batch = e_stack.shape[0]
    cdef Py_ssize_t na = e_stack.shape[1]
    cdef Py_ssize_t ni = e_stack.shape[2]
    cdef Py_ssize_t nj = e_stack.shape[3]
    cdef Py_ssize_t nk = a_buf.shape[2]
    cdef Py_ssize_t nx = a_buf.shape[3]
    cdef Py_ssize_t ny = a_buf.shape[4]
    cdef Py_ssize_t entry, ai, oi, a, i, j, k, x, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ai = a_pos[entry]
        oi = out_pos[entry]
        for a in range(na):
            for i in range(ni):
                for k in range(nk):
                    for x in range(nx):
                        for y in range(ny):
                            total = 0.0
                            for j in range(nj):
                                total += e_stack[entry, a, i, j] * a_buf[ai, j, k, x, y]
                            out_buf[oi, a, i, k, x, y] += total


cdef void _r_w_accum(
    cnp.ndarray[cnp.complex128_t, ndim=6] r_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] r_pos,
    cnp.ndarray[cnp.complex128_t, ndim=5] w_stack,
    cnp.ndarray[cnp.complex128_t, ndim=6] out_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
):
    cdef Py_ssize_t batch = w_stack.shape[0]
    cdef Py_ssize_t na = w_stack.shape[1]
    cdef Py_ssize_t nb = w_stack.shape[2]
    cdef Py_ssize_t nu = w_stack.shape[3]
    cdef Py_ssize_t nx = w_stack.shape[4]
    cdef Py_ssize_t ni = r_buf.shape[2]
    cdef Py_ssize_t nk = r_buf.shape[3]
    cdef Py_ssize_t ny = r_buf.shape[5]
    cdef Py_ssize_t entry, ri, oi, a, b, i, k, u, x, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ri = r_pos[entry]
        oi = out_pos[entry]
        for i in range(ni):
            for k in range(nk):
                for y in range(ny):
                    for b in range(nb):
                        for u in range(nu):
                            total = 0.0
                            for a in range(na):
                                for x in range(nx):
                                    total += r_buf[ri, a, i, k, x, y] * w_stack[entry, a, b, u, x]
                            out_buf[oi, i, k, y, b, u] += total


cdef void _t2_w_accum(
    cnp.ndarray[cnp.complex128_t, ndim=6] t2_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_pos,
    cnp.ndarray[cnp.complex128_t, ndim=5] w_stack,
    cnp.ndarray[cnp.complex128_t, ndim=6] out_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
):
    cdef Py_ssize_t batch = w_stack.shape[0]
    cdef Py_ssize_t nb = w_stack.shape[1]
    cdef Py_ssize_t nc = w_stack.shape[2]
    cdef Py_ssize_t nv = w_stack.shape[3]
    cdef Py_ssize_t ny = w_stack.shape[4]
    cdef Py_ssize_t ni = t2_buf.shape[1]
    cdef Py_ssize_t nk = t2_buf.shape[2]
    cdef Py_ssize_t nu = t2_buf.shape[5]
    cdef Py_ssize_t entry, ti, oi, b, c, i, k, u, v, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ti = t2_pos[entry]
        oi = out_pos[entry]
        for i in range(ni):
            for k in range(nk):
                for u in range(nu):
                    for c in range(nc):
                        for v in range(nv):
                            total = 0.0
                            for b in range(nb):
                                for y in range(ny):
                                    total += t2_buf[ti, i, k, y, b, u] * w_stack[entry, b, c, v, y]
                            out_buf[oi, i, k, u, c, v] += total


cdef void _t3_f_accum(
    cnp.ndarray[cnp.complex128_t, ndim=6] t3_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_pos,
    cnp.ndarray[cnp.complex128_t, ndim=4] f_stack,
    cnp.ndarray[cnp.complex128_t, ndim=5] out_buf,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
):
    cdef Py_ssize_t batch = f_stack.shape[0]
    cdef Py_ssize_t nc = f_stack.shape[1]
    cdef Py_ssize_t nl = f_stack.shape[2]
    cdef Py_ssize_t nk = f_stack.shape[3]
    cdef Py_ssize_t ni = t3_buf.shape[1]
    cdef Py_ssize_t nu = t3_buf.shape[3]
    cdef Py_ssize_t nv = t3_buf.shape[5]
    cdef Py_ssize_t entry, ti, oi, c, i, k, l, u, v
    cdef cnp.complex128_t total

    for entry in range(batch):
        ti = t3_pos[entry]
        oi = out_pos[entry]
        for i in range(ni):
            for l in range(nl):
                for u in range(nu):
                    for v in range(nv):
                        total = 0.0
                        for c in range(nc):
                            for k in range(nk):
                                total += t3_buf[ti, i, k, u, c, v] * f_stack[entry, c, l, k]
                        out_buf[oi, i, l, u, v] += total


def run_batched_matrix_chain(
    object r_e_stacks,
    object t2_w_stacks,
    object t3_w_stacks,
    object out_f_stacks,
    object r_specs,
    object t2_specs,
    object t3_specs,
    object out_specs,
    object a_data,
    object r_data,
    object t2_data,
    object t3_data,
    object out_data,
):
    cdef Py_ssize_t group_index
    cdef object spec

    for group_index in range(len(r_specs)):
        spec = r_specs[group_index]
        _e_a_accum(
            r_e_stacks[group_index],
            a_data[spec[0]],
            spec[1],
            r_data[spec[2]],
            spec[3],
        )
    for group_index in range(len(t2_specs)):
        spec = t2_specs[group_index]
        _r_w_accum(
            r_data[spec[0]],
            spec[1],
            t2_w_stacks[group_index],
            t2_data[spec[2]],
            spec[3],
        )
    for group_index in range(len(t3_specs)):
        spec = t3_specs[group_index]
        _t2_w_accum(
            t2_data[spec[0]],
            spec[1],
            t3_w_stacks[group_index],
            t3_data[spec[2]],
            spec[3],
        )
    for group_index in range(len(out_specs)):
        spec = out_specs[group_index]
        _t3_f_accum(
            t3_data[spec[0]],
            spec[1],
            out_f_stacks[group_index],
            out_data[spec[2]],
            spec[3],
        )


cdef inline Py_ssize_t _block_base(
    cnp.ndarray[cnp.int64_t, ndim=1] offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] sizes,
    Py_ssize_t group,
    Py_ssize_t pos,
):
    return offsets[group] + pos * sizes[group]


cdef void _e_a_arena(
    cnp.ndarray[cnp.complex128_t, ndim=4] e_stack,
    cnp.ndarray[cnp.complex128_t, ndim=1] a_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] a_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] a_sizes,
    Py_ssize_t a_group,
    cnp.ndarray[cnp.int64_t, ndim=1] a_pos,
    cnp.ndarray[cnp.complex128_t, ndim=1] r_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] r_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] r_sizes,
    Py_ssize_t r_group,
    cnp.ndarray[cnp.int64_t, ndim=1] r_pos,
    cnp.ndarray[cnp.int64_t, ndim=1] dims,
):
    cdef Py_ssize_t batch = e_stack.shape[0]
    cdef Py_ssize_t nj = dims[0]
    cdef Py_ssize_t nk = dims[1]
    cdef Py_ssize_t nx = dims[2]
    cdef Py_ssize_t ny = dims[3]
    cdef Py_ssize_t na = dims[4]
    cdef Py_ssize_t ni = dims[5]
    cdef Py_ssize_t entry, ai_base, ri_base, a, i, j, k, x, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ai_base = _block_base(a_offsets, a_sizes, a_group, a_pos[entry])
        ri_base = _block_base(r_offsets, r_sizes, r_group, r_pos[entry])
        for a in range(na):
            for i in range(ni):
                for k in range(nk):
                    for x in range(nx):
                        for y in range(ny):
                            total = 0.0
                            for j in range(nj):
                                total += (
                                    e_stack[entry, a, i, j]
                                    * a_arena[ai_base + (((j * nk + k) * nx + x) * ny + y)]
                                )
                            r_arena[ri_base + ((((a * ni + i) * nk + k) * nx + x) * ny + y)] += total


cdef void _r_w_arena(
    cnp.ndarray[cnp.complex128_t, ndim=1] r_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] r_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] r_sizes,
    Py_ssize_t r_group,
    cnp.ndarray[cnp.int64_t, ndim=1] r_pos,
    cnp.ndarray[cnp.complex128_t, ndim=5] w_stack,
    cnp.ndarray[cnp.complex128_t, ndim=1] t2_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_sizes,
    Py_ssize_t t2_group,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_pos,
    cnp.ndarray[cnp.int64_t, ndim=1] dims,
):
    cdef Py_ssize_t batch = w_stack.shape[0]
    cdef Py_ssize_t na = dims[0]
    cdef Py_ssize_t ni = dims[1]
    cdef Py_ssize_t nk = dims[2]
    cdef Py_ssize_t nx = dims[3]
    cdef Py_ssize_t ny = dims[4]
    cdef Py_ssize_t nb = dims[8]
    cdef Py_ssize_t nu = dims[9]
    cdef Py_ssize_t entry, ri_base, ti_base, a, b, i, k, u, x, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ri_base = _block_base(r_offsets, r_sizes, r_group, r_pos[entry])
        ti_base = _block_base(t2_offsets, t2_sizes, t2_group, t2_pos[entry])
        for i in range(ni):
            for k in range(nk):
                for y in range(ny):
                    for b in range(nb):
                        for u in range(nu):
                            total = 0.0
                            for a in range(na):
                                for x in range(nx):
                                    total += (
                                        r_arena[ri_base + ((((a * ni + i) * nk + k) * nx + x) * ny + y)]
                                        * w_stack[entry, a, b, u, x]
                                    )
                            t2_arena[ti_base + ((((i * nk + k) * ny + y) * nb + b) * nu + u)] += total


cdef void _t2_w_arena(
    cnp.ndarray[cnp.complex128_t, ndim=1] t2_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_sizes,
    Py_ssize_t t2_group,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_pos,
    cnp.ndarray[cnp.complex128_t, ndim=5] w_stack,
    cnp.ndarray[cnp.complex128_t, ndim=1] t3_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_sizes,
    Py_ssize_t t3_group,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_pos,
    cnp.ndarray[cnp.int64_t, ndim=1] dims,
):
    cdef Py_ssize_t batch = w_stack.shape[0]
    cdef Py_ssize_t ni = dims[0]
    cdef Py_ssize_t nk = dims[1]
    cdef Py_ssize_t ny = dims[2]
    cdef Py_ssize_t nb = dims[3]
    cdef Py_ssize_t nu = dims[4]
    cdef Py_ssize_t nc = dims[8]
    cdef Py_ssize_t nv = dims[9]
    cdef Py_ssize_t entry, ti_base, t3_base, i, k, b, c, u, v, y
    cdef cnp.complex128_t total

    for entry in range(batch):
        ti_base = _block_base(t2_offsets, t2_sizes, t2_group, t2_pos[entry])
        t3_base = _block_base(t3_offsets, t3_sizes, t3_group, t3_pos[entry])
        for i in range(ni):
            for k in range(nk):
                for u in range(nu):
                    for c in range(nc):
                        for v in range(nv):
                            total = 0.0
                            for b in range(nb):
                                for y in range(ny):
                                    total += (
                                        t2_arena[ti_base + ((((i * nk + k) * ny + y) * nb + b) * nu + u)]
                                        * w_stack[entry, b, c, v, y]
                                    )
                            t3_arena[t3_base + ((((i * nk + k) * nu + u) * nc + c) * nv + v)] += total


cdef void _t3_f_arena(
    cnp.ndarray[cnp.complex128_t, ndim=1] t3_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_sizes,
    Py_ssize_t t3_group,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_pos,
    cnp.ndarray[cnp.complex128_t, ndim=4] f_stack,
    cnp.ndarray[cnp.complex128_t, ndim=1] out_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] out_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] out_sizes,
    Py_ssize_t out_group,
    cnp.ndarray[cnp.int64_t, ndim=1] out_pos,
    cnp.ndarray[cnp.int64_t, ndim=1] dims,
):
    cdef Py_ssize_t batch = f_stack.shape[0]
    cdef Py_ssize_t ni = dims[0]
    cdef Py_ssize_t nk = dims[1]
    cdef Py_ssize_t nu = dims[2]
    cdef Py_ssize_t nc = dims[3]
    cdef Py_ssize_t nv = dims[4]
    cdef Py_ssize_t nl = dims[6]
    cdef Py_ssize_t entry, t3_base, out_base, i, c, k, l, u, v
    cdef cnp.complex128_t total

    for entry in range(batch):
        t3_base = _block_base(t3_offsets, t3_sizes, t3_group, t3_pos[entry])
        out_base = _block_base(out_offsets, out_sizes, out_group, out_pos[entry])
        for i in range(ni):
            for l in range(nl):
                for u in range(nu):
                    for v in range(nv):
                        total = 0.0
                        for c in range(nc):
                            for k in range(nk):
                                total += (
                                    t3_arena[t3_base + ((((i * nk + k) * nu + u) * nc + c) * nv + v)]
                                    * f_stack[entry, c, l, k]
                                )
                        out_arena[out_base + (((i * nl + l) * nu + u) * nv + v)] += total


def run_batched_matrix_chain_arenas(
    object r_e_stacks,
    object t2_w_stacks,
    object t3_w_stacks,
    object out_f_stacks,
    object r_specs,
    object t2_specs,
    object t3_specs,
    object out_specs,
    cnp.ndarray[cnp.complex128_t, ndim=1] a_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] a_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] a_sizes,
    cnp.ndarray[cnp.complex128_t, ndim=1] r_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] r_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] r_sizes,
    cnp.ndarray[cnp.complex128_t, ndim=1] t2_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] t2_sizes,
    cnp.ndarray[cnp.complex128_t, ndim=1] t3_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] t3_sizes,
    cnp.ndarray[cnp.complex128_t, ndim=1] out_arena,
    cnp.ndarray[cnp.int64_t, ndim=1] out_offsets,
    cnp.ndarray[cnp.int64_t, ndim=1] out_sizes,
):
    cdef Py_ssize_t group_index
    cdef object spec

    for group_index in range(len(r_specs)):
        spec = r_specs[group_index]
        _e_a_arena(
            r_e_stacks[group_index],
            a_arena, a_offsets, a_sizes, spec[0], spec[1],
            r_arena, r_offsets, r_sizes, spec[2], spec[3], spec[4],
        )
    for group_index in range(len(t2_specs)):
        spec = t2_specs[group_index]
        _r_w_arena(
            r_arena, r_offsets, r_sizes, spec[0], spec[1],
            t2_w_stacks[group_index],
            t2_arena, t2_offsets, t2_sizes, spec[2], spec[3], spec[4],
        )
    for group_index in range(len(t3_specs)):
        spec = t3_specs[group_index]
        _t2_w_arena(
            t2_arena, t2_offsets, t2_sizes, spec[0], spec[1],
            t3_w_stacks[group_index],
            t3_arena, t3_offsets, t3_sizes, spec[2], spec[3], spec[4],
        )
    for group_index in range(len(out_specs)):
        spec = out_specs[group_index]
        _t3_f_arena(
            t3_arena, t3_offsets, t3_sizes, spec[0], spec[1],
            out_f_stacks[group_index],
            out_arena, out_offsets, out_sizes, spec[2], spec[3], spec[4],
        )


def sparse_coo_matvec(
    cnp.ndarray[cnp.int64_t, ndim=1] rows,
    cnp.ndarray[cnp.int64_t, ndim=1] cols,
    cnp.ndarray[cnp.complex128_t, ndim=1] values,
    cnp.ndarray[cnp.complex128_t, ndim=1] vector,
    Py_ssize_t dim,
):
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] out = np.zeros(dim, dtype=np.complex128)
    cdef Py_ssize_t nnz = values.shape[0]
    cdef Py_ssize_t i
    for i in range(nnz):
        out[rows[i]] += values[i] * vector[cols[i]]
    return out


def sparse_csr_matvec(
    cnp.ndarray[cnp.int64_t, ndim=1] indptr,
    cnp.ndarray[cnp.int64_t, ndim=1] indices,
    cnp.ndarray[cnp.complex128_t, ndim=1] values,
    cnp.ndarray[cnp.complex128_t, ndim=1] vector,
    Py_ssize_t dim,
):
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] out = np.zeros(dim, dtype=np.complex128)
    cdef Py_ssize_t row, ptr, start, stop
    cdef cnp.complex128_t total
    for row in range(dim):
        start = indptr[row]
        stop = indptr[row + 1]
        total = 0.0
        for ptr in range(start, stop):
            total += values[ptr] * vector[indices[ptr]]
        out[row] = total
    return out


def direct_operator_entry_coo(
    cnp.ndarray[cnp.complex128_t, ndim=3] left_stack,
    cnp.ndarray[cnp.complex128_t, ndim=3] right_stack,
    cnp.ndarray[cnp.int64_t, ndim=1] dims,
    Py_ssize_t in_start,
    Py_ssize_t out_start,
    double tol,
):
    cdef Py_ssize_t nb = left_stack.shape[0]
    cdef Py_ssize_t ni = dims[0]
    cdef Py_ssize_t nl = dims[1]
    cdef Py_ssize_t nu = dims[2]
    cdef Py_ssize_t nv = dims[3]
    cdef Py_ssize_t nj = dims[4]
    cdef Py_ssize_t nx = dims[5]
    cdef Py_ssize_t nk = dims[6]
    cdef Py_ssize_t ny = dims[7]
    cdef Py_ssize_t in_size = nj * nk * nx * ny
    cdef Py_ssize_t out_size = ni * nl * nu * nv
    cdef Py_ssize_t max_nnz = in_size * out_size
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rows = np.empty(max_nnz, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cols = np.empty(max_nnz, dtype=np.int64)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] values = np.empty(max_nnz, dtype=np.complex128)
    cdef double tol2 = tol * tol
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t local_col, local_row, tmp, j, k, x, y, i, l, u, v
    cdef Py_ssize_t left_col, right_row, left_row, right_col, b
    cdef cnp.complex128_t total
    cdef double mag2

    for local_col in range(in_size):
        y = local_col % ny
        tmp = local_col // ny
        x = tmp % nx
        tmp = tmp // nx
        k = tmp % nk
        j = tmp // nk
        left_col = j * nx + x
        right_row = k * ny + y
        for local_row in range(out_size):
            v = local_row % nv
            tmp = local_row // nv
            u = tmp % nu
            tmp = tmp // nu
            l = tmp % nl
            i = tmp // nl
            left_row = i * nu + u
            right_col = l * nv + v
            total = 0.0
            for b in range(nb):
                total += left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
            mag2 = total.real * total.real + total.imag * total.imag
            if mag2 > tol2:
                rows[count] = out_start + local_row
                cols[count] = in_start + local_col
                values[count] = total
                count += 1
    return rows[:count].copy(), cols[:count].copy(), values[:count].copy()


def direct_operator_entry_sparse_product_coo(
    cnp.ndarray[cnp.complex128_t, ndim=3] left_stack,
    cnp.ndarray[cnp.complex128_t, ndim=3] right_stack,
    cnp.ndarray[cnp.int64_t, ndim=1] dims,
    Py_ssize_t in_start,
    Py_ssize_t out_start,
    double tol,
    Py_ssize_t max_entries,
):
    cdef Py_ssize_t nb = left_stack.shape[0]
    cdef Py_ssize_t ni = dims[0]
    cdef Py_ssize_t nl = dims[1]
    cdef Py_ssize_t nu = dims[2]
    cdef Py_ssize_t nv = dims[3]
    cdef Py_ssize_t nj = dims[4]
    cdef Py_ssize_t nx = dims[5]
    cdef Py_ssize_t nk = dims[6]
    cdef Py_ssize_t ny = dims[7]
    cdef Py_ssize_t left_rows = ni * nu
    cdef Py_ssize_t left_cols = nj * nx
    cdef Py_ssize_t right_rows = nk * ny
    cdef Py_ssize_t right_cols = nl * nv
    cdef double tol2 = tol * tol
    cdef Py_ssize_t b, lr, lc, rr, rc, tmp
    cdef Py_ssize_t left_count, right_count, count = 0
    cdef double mag2
    cdef cnp.complex128_t lv, rv

    for b in range(nb):
        left_count = 0
        for lr in range(left_rows):
            for lc in range(left_cols):
                lv = left_stack[b, lr, lc]
                mag2 = lv.real * lv.real + lv.imag * lv.imag
                if mag2 > tol2:
                    left_count += 1
        if left_count == 0:
            continue
        right_count = 0
        for rr in range(right_rows):
            for rc in range(right_cols):
                rv = right_stack[b, rr, rc]
                mag2 = rv.real * rv.real + rv.imag * rv.imag
                if mag2 > tol2:
                    right_count += 1
        count += left_count * right_count
        if max_entries > 0 and count > max_entries:
            return None

    cdef cnp.ndarray[cnp.int64_t, ndim=1] rows = np.empty(count, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cols = np.empty(count, dtype=np.int64)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] values = np.empty(count, dtype=np.complex128)
    cdef Py_ssize_t pos = 0
    cdef Py_ssize_t i, u, j, x, k, y, l, v, local_row, local_col

    for b in range(nb):
        for lr in range(left_rows):
            lv = 0.0
            u = lr % nu
            i = lr // nu
            for lc in range(left_cols):
                lv = left_stack[b, lr, lc]
                mag2 = lv.real * lv.real + lv.imag * lv.imag
                if mag2 <= tol2:
                    continue
                x = lc % nx
                j = lc // nx
                for rr in range(right_rows):
                    y = rr % ny
                    k = rr // ny
                    local_col = (((j * nk + k) * nx + x) * ny + y)
                    for rc in range(right_cols):
                        rv = right_stack[b, rr, rc]
                        mag2 = rv.real * rv.real + rv.imag * rv.imag
                        if mag2 <= tol2:
                            continue
                        v = rc % nv
                        l = rc // nv
                        local_row = (((i * nl + l) * nu + u) * nv + v)
                        rows[pos] = out_start + local_row
                        cols[pos] = in_start + local_col
                        values[pos] = lv * rv
                        pos += 1
    return rows, cols, values


def direct_operator_entries_coo(
    object left_stacks,
    object right_stacks,
    cnp.ndarray[cnp.int64_t, ndim=2] dims_all,
    cnp.ndarray[cnp.int64_t, ndim=1] in_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] out_starts,
    double tol,
):
    cdef Py_ssize_t n_entries = len(left_stacks)
    cdef list row_chunks = []
    cdef list col_chunks = []
    cdef list value_chunks = []
    cdef Py_ssize_t entry
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] left_stack
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] right_stack
    cdef Py_ssize_t nb, ni, nl, nu, nv, nj, nx, nk, ny
    cdef Py_ssize_t in_size, out_size, max_nnz
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cols
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] values
    cdef double tol2 = tol * tol
    cdef Py_ssize_t count
    cdef Py_ssize_t local_col, local_row, tmp, j, k, x, y, i, l, u, v
    cdef Py_ssize_t left_col, right_row, left_row, right_col, b
    cdef cnp.complex128_t total
    cdef double mag2

    for entry in range(n_entries):
        left_stack = left_stacks[entry]
        right_stack = right_stacks[entry]
        nb = left_stack.shape[0]
        ni = dims_all[entry, 0]
        nl = dims_all[entry, 1]
        nu = dims_all[entry, 2]
        nv = dims_all[entry, 3]
        nj = dims_all[entry, 4]
        nx = dims_all[entry, 5]
        nk = dims_all[entry, 6]
        ny = dims_all[entry, 7]
        in_size = nj * nk * nx * ny
        out_size = ni * nl * nu * nv
        max_nnz = in_size * out_size
        rows = np.empty(max_nnz, dtype=np.int64)
        cols = np.empty(max_nnz, dtype=np.int64)
        values = np.empty(max_nnz, dtype=np.complex128)
        count = 0
        for local_col in range(in_size):
            y = local_col % ny
            tmp = local_col // ny
            x = tmp % nx
            tmp = tmp // nx
            k = tmp % nk
            j = tmp // nk
            left_col = j * nx + x
            right_row = k * ny + y
            for local_row in range(out_size):
                v = local_row % nv
                tmp = local_row // nv
                u = tmp % nu
                tmp = tmp // nu
                l = tmp % nl
                i = tmp // nl
                left_row = i * nu + u
                right_col = l * nv + v
                total = 0.0
                for b in range(nb):
                    total += left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                mag2 = total.real * total.real + total.imag * total.imag
                if mag2 > tol2:
                    rows[count] = out_starts[entry] + local_row
                    cols[count] = in_starts[entry] + local_col
                    values[count] = total
                    count += 1
        if count:
            row_chunks.append(rows[:count].copy())
            col_chunks.append(cols[:count].copy())
            value_chunks.append(values[:count].copy())
    if not row_chunks:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.complex128),
        )
    return (
        np.concatenate(row_chunks),
        np.concatenate(col_chunks),
        np.concatenate(value_chunks),
    )


def direct_operator_entries_csr(
    object left_stacks,
    object right_stacks,
    cnp.ndarray[cnp.int64_t, ndim=2] dims_all,
    cnp.ndarray[cnp.int64_t, ndim=1] in_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] out_starts,
    Py_ssize_t dim,
    double tol,
    object scales_all=None,
):
    cdef Py_ssize_t n_entries = len(left_stacks)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] dense = np.zeros((dim, dim), dtype=np.complex128)
    cdef Py_ssize_t entry
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] left_stack
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] right_stack
    cdef Py_ssize_t nb, ni, nl, nu, nv, nj, nx, nk, ny
    cdef Py_ssize_t in_size, out_size
    cdef Py_ssize_t raw_count = 0
    cdef double tol2 = tol * tol
    cdef Py_ssize_t local_col, local_row, tmp, j, k, x, y, i, l, u, v
    cdef Py_ssize_t left_col, right_row, left_row, right_col, b
    cdef cnp.complex128_t total, value, scale
    cdef double mag2
    cdef bint use_scales = scales_all is not None
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] scales_array
    if use_scales:
        scales_array = scales_all

    for entry in range(n_entries):
        scale = 1.0
        if use_scales:
            scale = scales_array[entry]
        left_stack = left_stacks[entry]
        right_stack = right_stacks[entry]
        nb = left_stack.shape[0]
        ni = dims_all[entry, 0]
        nl = dims_all[entry, 1]
        nu = dims_all[entry, 2]
        nv = dims_all[entry, 3]
        nj = dims_all[entry, 4]
        nx = dims_all[entry, 5]
        nk = dims_all[entry, 6]
        ny = dims_all[entry, 7]
        in_size = nj * nk * nx * ny
        out_size = ni * nl * nu * nv
        for local_col in range(in_size):
            y = local_col % ny
            tmp = local_col // ny
            x = tmp % nx
            tmp = tmp // nx
            k = tmp % nk
            j = tmp // nk
            left_col = j * nx + x
            right_row = k * ny + y
            for local_row in range(out_size):
                v = local_row % nv
                tmp = local_row // nv
                u = tmp % nu
                tmp = tmp // nu
                l = tmp % nl
                i = tmp // nl
                left_row = i * nu + u
                right_col = l * nv + v
                total = 0.0
                for b in range(nb):
                    total += left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                total = total * scale
                mag2 = total.real * total.real + total.imag * total.imag
                if mag2 > tol2:
                    dense[out_starts[entry] + local_row, in_starts[entry] + local_col] += total
                    raw_count += 1

    cdef cnp.ndarray[cnp.int64_t, ndim=1] indptr = np.zeros(dim + 1, dtype=np.int64)
    cdef Py_ssize_t row, col, nnz = 0
    for row in range(dim):
        for col in range(dim):
            value = dense[row, col]
            mag2 = value.real * value.real + value.imag * value.imag
            if mag2 > tol2:
                nnz += 1
        indptr[row + 1] = nnz

    cdef cnp.ndarray[cnp.int64_t, ndim=1] indices = np.empty(nnz, dtype=np.int64)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] values = np.empty(nnz, dtype=np.complex128)
    cdef Py_ssize_t pos = 0
    for row in range(dim):
        for col in range(dim):
            value = dense[row, col]
            mag2 = value.real * value.real + value.imag * value.imag
            if mag2 > tol2:
                indices[pos] = col
                values[pos] = value
                pos += 1
    return indptr, indices, values, raw_count


def direct_operator_entries_csr_np_extract(
    object left_stacks,
    object right_stacks,
    cnp.ndarray[cnp.int64_t, ndim=2] dims_all,
    cnp.ndarray[cnp.int64_t, ndim=1] in_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] out_starts,
    Py_ssize_t dim,
    double tol,
    object scales_all=None,
):
    cdef Py_ssize_t n_entries = len(left_stacks)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] dense = np.zeros((dim, dim), dtype=np.complex128)
    cdef Py_ssize_t entry
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] left_stack
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] right_stack
    cdef Py_ssize_t nb, ni, nl, nu, nv, nj, nx, nk, ny
    cdef Py_ssize_t in_size, out_size
    cdef Py_ssize_t raw_count = 0
    cdef double tol2 = tol * tol
    cdef Py_ssize_t local_col, local_row, tmp, j, k, x, y, i, l, u, v
    cdef Py_ssize_t left_col, right_row, left_row, right_col, b
    cdef cnp.complex128_t total, scale
    cdef double mag2
    cdef bint use_scales = scales_all is not None
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] scales_array
    if use_scales:
        scales_array = scales_all

    for entry in range(n_entries):
        scale = 1.0
        if use_scales:
            scale = scales_array[entry]
        left_stack = left_stacks[entry]
        right_stack = right_stacks[entry]
        nb = left_stack.shape[0]
        ni = dims_all[entry, 0]
        nl = dims_all[entry, 1]
        nu = dims_all[entry, 2]
        nv = dims_all[entry, 3]
        nj = dims_all[entry, 4]
        nx = dims_all[entry, 5]
        nk = dims_all[entry, 6]
        ny = dims_all[entry, 7]
        in_size = nj * nk * nx * ny
        out_size = ni * nl * nu * nv
        for local_col in range(in_size):
            y = local_col % ny
            tmp = local_col // ny
            x = tmp % nx
            tmp = tmp // nx
            k = tmp % nk
            j = tmp // nk
            left_col = j * nx + x
            right_row = k * ny + y
            for local_row in range(out_size):
                v = local_row % nv
                tmp = local_row // nv
                u = tmp % nu
                tmp = tmp // nu
                l = tmp % nl
                i = tmp // nl
                left_row = i * nu + u
                right_col = l * nv + v
                total = 0.0
                for b in range(nb):
                    total += left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                total = total * scale
                mag2 = total.real * total.real + total.imag * total.imag
                if mag2 > tol2:
                    dense[out_starts[entry] + local_row, in_starts[entry] + local_col] += total
                    raw_count += 1

    rows, cols = np.nonzero(np.abs(dense) > tol)
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    values = np.ascontiguousarray(dense[rows, cols], dtype=np.complex128)
    indptr = np.zeros(dim + 1, dtype=np.int64)
    if rows.size:
        np.add.at(indptr, rows + 1, 1)
        np.cumsum(indptr, out=indptr)
    return indptr, cols, values, raw_count


def csr_dense_lookup(
    cnp.ndarray[cnp.int64_t, ndim=1] indptr,
    cnp.ndarray[cnp.int64_t, ndim=1] indices,
    Py_ssize_t dim,
):
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lookup = np.empty(dim * dim, dtype=np.int64)
    cdef Py_ssize_t row, ptr, start, stop
    lookup.fill(-1)
    for row in range(dim):
        start = indptr[row]
        stop = indptr[row + 1]
        for ptr in range(start, stop):
            lookup[row * dim + indices[ptr]] = ptr
    return lookup


def direct_operator_entries_csr_refill(
    object left_stacks,
    object right_stacks,
    cnp.ndarray[cnp.int64_t, ndim=2] dims_all,
    cnp.ndarray[cnp.int64_t, ndim=1] in_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] out_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] lookup,
    Py_ssize_t dim,
    Py_ssize_t nnz,
    double tol,
    object scales_all=None,
):
    cdef Py_ssize_t n_entries = len(left_stacks)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] values = np.zeros(nnz, dtype=np.complex128)
    cdef Py_ssize_t entry
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] left_stack
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] right_stack
    cdef Py_ssize_t nb, ni, nl, nu, nv, nj, nx, nk, ny
    cdef Py_ssize_t in_size, out_size
    cdef Py_ssize_t raw_count = 0
    cdef Py_ssize_t missing_count = 0
    cdef double tol2 = tol * tol
    cdef Py_ssize_t local_col, local_row, tmp, j, k, x, y, i, l, u, v
    cdef Py_ssize_t left_col, right_row, left_row, right_col, b
    cdef Py_ssize_t global_row, global_col, pos
    cdef cnp.complex128_t total, scale
    cdef double mag2
    cdef bint use_scales = scales_all is not None
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] scales_array
    if use_scales:
        scales_array = scales_all

    if lookup.shape[0] != dim * dim:
        raise ValueError("CSR lookup size does not match dimension")

    for entry in range(n_entries):
        scale = 1.0
        if use_scales:
            scale = scales_array[entry]
        left_stack = left_stacks[entry]
        right_stack = right_stacks[entry]
        nb = left_stack.shape[0]
        ni = dims_all[entry, 0]
        nl = dims_all[entry, 1]
        nu = dims_all[entry, 2]
        nv = dims_all[entry, 3]
        nj = dims_all[entry, 4]
        nx = dims_all[entry, 5]
        nk = dims_all[entry, 6]
        ny = dims_all[entry, 7]
        in_size = nj * nk * nx * ny
        out_size = ni * nl * nu * nv
        for local_col in range(in_size):
            y = local_col % ny
            tmp = local_col // ny
            x = tmp % nx
            tmp = tmp // nx
            k = tmp % nk
            j = tmp // nk
            left_col = j * nx + x
            right_row = k * ny + y
            global_col = in_starts[entry] + local_col
            for local_row in range(out_size):
                v = local_row % nv
                tmp = local_row // nv
                u = tmp % nu
                tmp = tmp // nu
                l = tmp % nl
                i = tmp // nl
                left_row = i * nu + u
                right_col = l * nv + v
                total = 0.0
                for b in range(nb):
                    total += left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                total = total * scale
                mag2 = total.real * total.real + total.imag * total.imag
                if mag2 > tol2:
                    raw_count += 1
                    global_row = out_starts[entry] + local_row
                    pos = lookup[global_row * dim + global_col]
                    if pos >= 0:
                        values[pos] += total
                    else:
                        missing_count += 1
    return values, raw_count, missing_count


def direct_operator_entries_matvec(
    object left_stacks,
    object right_stacks,
    cnp.ndarray[cnp.int64_t, ndim=2] dims_all,
    cnp.ndarray[cnp.int64_t, ndim=1] in_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] out_starts,
    cnp.ndarray[cnp.complex128_t, ndim=1] vector,
    Py_ssize_t dim,
    object scales_all=None,
):
    cdef Py_ssize_t n_entries = len(left_stacks)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] out = np.zeros(dim, dtype=np.complex128)
    cdef Py_ssize_t entry
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] left_stack
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] right_stack
    cdef Py_ssize_t nb, ni, nl, nu, nv, nj, nx, nk, ny
    cdef Py_ssize_t in_size, out_size
    cdef Py_ssize_t local_col, local_row, tmp, j, k, x, y, i, l, u, v
    cdef Py_ssize_t left_col, right_row, left_row, right_col, b
    cdef Py_ssize_t global_col, global_row
    cdef cnp.complex128_t total, vin, scale
    cdef bint use_scales = scales_all is not None
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] scales_array

    for entry in range(n_entries):
        scale = 1.0
        if use_scales:
            scales_array = scales_all
            scale = scales_array[entry]
        left_stack = left_stacks[entry]
        right_stack = right_stacks[entry]
        nb = left_stack.shape[0]
        ni = dims_all[entry, 0]
        nl = dims_all[entry, 1]
        nu = dims_all[entry, 2]
        nv = dims_all[entry, 3]
        nj = dims_all[entry, 4]
        nx = dims_all[entry, 5]
        nk = dims_all[entry, 6]
        ny = dims_all[entry, 7]
        in_size = nj * nk * nx * ny
        out_size = ni * nl * nu * nv
        for local_col in range(in_size):
            global_col = in_starts[entry] + local_col
            vin = vector[global_col]
            if vin.real == 0.0 and vin.imag == 0.0:
                continue
            y = local_col % ny
            tmp = local_col // ny
            x = tmp % nx
            tmp = tmp // nx
            k = tmp % nk
            j = tmp // nk
            left_col = j * nx + x
            right_row = k * ny + y
            for local_row in range(out_size):
                v = local_row % nv
                tmp = local_row // nv
                u = tmp % nu
                tmp = tmp // nu
                l = tmp % nl
                i = tmp // nl
                left_row = i * nu + u
                right_col = l * nv + v
                total = 0.0
                for b in range(nb):
                    total += left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                global_row = out_starts[entry] + local_row
                out[global_row] += total * scale * vin
    return out


def direct_operator_groups_matvec(
    object left_groups,
    object right_groups,
    cnp.ndarray[cnp.int64_t, ndim=2] dims_all,
    cnp.ndarray[cnp.int64_t, ndim=1] in_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] out_starts,
    cnp.ndarray[cnp.complex128_t, ndim=1] vector,
    Py_ssize_t dim,
    object group_scales=None,
):
    cdef Py_ssize_t n_groups = len(left_groups)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] out = np.zeros(dim, dtype=np.complex128)
    cdef Py_ssize_t group
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] left_stack
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] right_stack
    cdef Py_ssize_t nb, ni, nl, nu, nv, nj, nx, nk, ny
    cdef Py_ssize_t in_size, out_size
    cdef Py_ssize_t local_col, local_row, tmp, j, k, x, y, i, l, u, v
    cdef Py_ssize_t left_col, right_row, left_row, right_col, b
    cdef Py_ssize_t global_col, global_row
    cdef cnp.complex128_t total, vin
    cdef bint use_group_scales = group_scales is not None
    cdef bint use_scales
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] scales
    cdef object scale_obj

    for group in range(n_groups):
        left_stack = left_groups[group]
        right_stack = right_groups[group]
        nb = left_stack.shape[0]
        use_scales = False
        if use_group_scales:
            scale_obj = group_scales[group]
            if scale_obj is not None:
                scales = scale_obj
                use_scales = scales.shape[0] == nb
        ni = dims_all[group, 0]
        nl = dims_all[group, 1]
        nu = dims_all[group, 2]
        nv = dims_all[group, 3]
        nj = dims_all[group, 4]
        nx = dims_all[group, 5]
        nk = dims_all[group, 6]
        ny = dims_all[group, 7]
        in_size = nj * nk * nx * ny
        out_size = ni * nl * nu * nv
        for local_col in range(in_size):
            global_col = in_starts[group] + local_col
            vin = vector[global_col]
            if vin.real == 0.0 and vin.imag == 0.0:
                continue
            y = local_col % ny
            tmp = local_col // ny
            x = tmp % nx
            tmp = tmp // nx
            k = tmp % nk
            j = tmp // nk
            left_col = j * nx + x
            right_row = k * ny + y
            for local_row in range(out_size):
                v = local_row % nv
                tmp = local_row // nv
                u = tmp % nu
                tmp = tmp // nu
                l = tmp % nl
                i = tmp // nl
                left_row = i * nu + u
                right_col = l * nv + v
                total = 0.0
                if use_scales:
                    for b in range(nb):
                        total += scales[b] * left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                else:
                    for b in range(nb):
                        total += left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                global_row = out_starts[group] + local_row
                out[global_row] += total * vin
    return out


def direct_operator_groups_dense_blocks(
    object left_groups,
    object right_groups,
    cnp.ndarray[cnp.int64_t, ndim=2] dims_all,
    object group_scales=None,
):
    cdef Py_ssize_t n_groups = len(left_groups)
    cdef list blocks = []
    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_sizes = np.empty(n_groups, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_sizes = np.empty(n_groups, dtype=np.int64)
    cdef Py_ssize_t group
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] left_stack
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] right_stack
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] block
    cdef Py_ssize_t nb, ni, nl, nu, nv, nj, nx, nk, ny
    cdef Py_ssize_t in_size, out_size
    cdef Py_ssize_t local_col, local_row, tmp, j, k, x, y, i, l, u, v
    cdef Py_ssize_t left_col, right_row, left_row, right_col, b
    cdef cnp.complex128_t total
    cdef bint use_group_scales = group_scales is not None
    cdef bint use_scales
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] scales
    cdef object scale_obj

    for group in range(n_groups):
        left_stack = left_groups[group]
        right_stack = right_groups[group]
        nb = left_stack.shape[0]
        use_scales = False
        if use_group_scales:
            scale_obj = group_scales[group]
            if scale_obj is not None:
                scales = scale_obj
                use_scales = scales.shape[0] == nb
        ni = dims_all[group, 0]
        nl = dims_all[group, 1]
        nu = dims_all[group, 2]
        nv = dims_all[group, 3]
        nj = dims_all[group, 4]
        nx = dims_all[group, 5]
        nk = dims_all[group, 6]
        ny = dims_all[group, 7]
        in_size = nj * nk * nx * ny
        out_size = ni * nl * nu * nv
        in_sizes[group] = in_size
        out_sizes[group] = out_size
        block = np.zeros((out_size, in_size), dtype=np.complex128)
        for local_col in range(in_size):
            y = local_col % ny
            tmp = local_col // ny
            x = tmp % nx
            tmp = tmp // nx
            k = tmp % nk
            j = tmp // nk
            left_col = j * nx + x
            right_row = k * ny + y
            for local_row in range(out_size):
                v = local_row % nv
                tmp = local_row // nv
                u = tmp % nu
                tmp = tmp // nu
                l = tmp % nl
                i = tmp // nl
                left_row = i * nu + u
                right_col = l * nv + v
                total = 0.0
                if use_scales:
                    for b in range(nb):
                        total += scales[b] * left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                else:
                    for b in range(nb):
                        total += left_stack[b, left_row, left_col] * right_stack[b, right_row, right_col]
                block[local_row, local_col] = total
        blocks.append(block)
    return blocks, in_sizes, out_sizes


def direct_operator_block_matrices_matvec(
    object blocks,
    cnp.ndarray[cnp.int64_t, ndim=1] in_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] out_starts,
    cnp.ndarray[cnp.complex128_t, ndim=1] vector,
    Py_ssize_t dim,
):
    cdef Py_ssize_t n_blocks = len(blocks)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] out = np.zeros(dim, dtype=np.complex128)
    cdef Py_ssize_t block_index, row, col, in_start, out_start
    cdef Py_ssize_t n_rows, n_cols
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] block
    cdef cnp.complex128_t total

    for block_index in range(n_blocks):
        block = blocks[block_index]
        n_rows = block.shape[0]
        n_cols = block.shape[1]
        in_start = in_starts[block_index]
        out_start = out_starts[block_index]
        for row in range(n_rows):
            total = 0.0
            for col in range(n_cols):
                total += block[row, col] * vector[in_start + col]
            out[out_start + row] += total
    return out


def direct_operator_block_sparse_matvec(
    object block_rows,
    object block_cols,
    object block_values,
    cnp.ndarray[cnp.int64_t, ndim=1] in_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] out_starts,
    cnp.ndarray[cnp.complex128_t, ndim=1] vector,
    Py_ssize_t dim,
):
    cdef Py_ssize_t n_blocks = len(block_values)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] out = np.zeros(dim, dtype=np.complex128)
    cdef Py_ssize_t block_index, ptr, nnz, in_start, out_start
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rows
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cols
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] values

    for block_index in range(n_blocks):
        rows = block_rows[block_index]
        cols = block_cols[block_index]
        values = block_values[block_index]
        nnz = values.shape[0]
        in_start = in_starts[block_index]
        out_start = out_starts[block_index]
        for ptr in range(nnz):
            out[out_start + rows[ptr]] += values[ptr] * vector[in_start + cols[ptr]]
    return out
