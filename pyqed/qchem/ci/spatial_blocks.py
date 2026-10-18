"""Matrix-free spin/spatial CI actions with cached representative rows.

Adaptation of finite-group matrix-unit projection (F. A. Cotton, Chemical
Applications of Group Theory, 3rd ed., Wiley, 1990, ISBN 9780471510949), with
exact representative-row reconstruction for an invariant Hamiltonian. The
reduced action contracts only independent output rows; no Hamiltonian block
is assembled. Sparse row bases and reconstruction plans are cached. Rank
and zero tolerances limit numerical exactness. Dense mixing or ill-conditioned
row selection uses assembled/projected fallbacks. The assembled helper is
retained as a validation reference and bounded fallback, not the default.
Row selection adapts pivoted Gram-Schmidt with reorthogonalization (G. H. Golub
and C. F. Van Loan, Matrix Computations, 4th ed., Johns Hopkins, 2013). It checks
inverse residuals and a Frobenius condition bound, with LAPACK QR fallback;
this is not a claim of identical pivots to Householder QR.
"""
from collections import OrderedDict
import hashlib

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import connected_components

_PLAN_CACHE = OrderedDict()
_MAX_PLAN_BYTES = 64 * 1024**2
_MAX_BLOCK_ELEMENTS = 8_000_000


def _trim(matrix, tol=1e-13):
    matrix = matrix.tocsr()
    matrix.data[np.abs(matrix.data) < tol] = 0
    matrix.eliminate_zeros()
    return matrix


def _component_eigenspace(matrix, target, *, projector=False):
    """Batch small disconnected Hermitian blocks into bounded LAPACK calls."""
    if matrix.shape[0] == 0:
        return sparse.csr_matrix(matrix.shape, dtype=matrix.dtype)
    count, labels = connected_components(abs(matrix), directed=False)
    sizes = np.bincount(labels, minlength=count)
    if sizes.max(initial=0) > 512:
        return None
    order = np.argsort(labels, kind='stable')
    offsets = np.r_[0, np.cumsum(sizes)]
    local = np.empty(len(labels), dtype=np.int64)
    local[order] = np.arange(len(labels)) - np.repeat(offsets[:-1], sizes)
    coo = matrix.tocoo()
    entry_components = labels[coo.row]
    entry_order = np.argsort(entry_components, kind='stable')
    entry_offsets = np.r_[0, np.cumsum(np.bincount(entry_components, minlength=count))]
    rows, cols, values = [], [], []
    dimension = 0
    for size in np.unique(sizes):
        components = np.flatnonzero(sizes == size)
        # Limit temporary dense panels and eigenvectors to roughly 32 MiB each.
        batch_size = max(1, 2_000_000 // (size*size))
        for start in range(0, len(components), batch_size):
            batch = components[start:start+batch_size]
            indices = order[offsets[batch, None] + np.arange(size)]
            lengths = entry_offsets[batch+1] - entry_offsets[batch]
            positions = np.repeat(entry_offsets[batch], lengths)
            positions += np.arange(lengths.sum()) - np.repeat(np.r_[0, np.cumsum(lengths)[:-1]], lengths)
            entries = entry_order[positions]
            panels = np.zeros((len(batch), size, size), dtype=matrix.dtype)
            panels[np.repeat(np.arange(len(batch)), lengths), local[coo.row[entries]],
                   local[coo.col[entries]]] = coo.data[entries]
            if size == 1:
                eigenvalues, vectors = panels.real[:, 0, :], np.ones_like(panels)
            else:
                eigenvalues, vectors = np.linalg.eigh(panels)
            if projector and np.max(np.minimum(abs(eigenvalues), abs(eigenvalues-1)), initial=0) > 1e-8:
                raise ValueError('Spatial row matrix is not an orthogonal projector.')
            keep = abs(eigenvalues-target) < 1e-8
            numbering = np.cumsum(keep.ravel()).reshape(keep.shape)-1 + dimension
            b, i, j = np.nonzero((abs(vectors) > 1e-13) & keep[:, None, :])
            rows.append(indices[b, i]); cols.append(numbering[b, j]); values.append(vectors[b, i, j])
            dimension += np.count_nonzero(keep)
    return sparse.csr_matrix((np.real_if_close(np.concatenate(values)),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(matrix.shape[0], dimension))


@njit(cache=True, nogil=True)
def _projector_rows(ap, ai, av, bp, bi, bv, allowed, capacity):
    na, nb = ap.shape[1]-1, bp.shape[1]-1
    ndet = na*nb
    data = np.empty(capacity, dtype=av.dtype)
    indices = np.empty(capacity, dtype=np.int64)
    ptr = np.zeros(ndet+1, dtype=np.int64)
    work = np.zeros(ndet, dtype=av.dtype)
    marks = np.full(ndet, -1, dtype=np.int64)
    touched = np.empty(ndet, dtype=np.int64)
    used = 0
    for row in range(ndet):
        if not allowed[row]:
            ptr[row+1] = used
            continue
        ia, ib = row//nb, row%nb
        count = 0
        for op in range(len(ap)):
            for a in range(ap[op, ia], ap[op, ia+1]):
                for b in range(bp[op, ib], bp[op, ib+1]):
                    col = ai[a]*nb+bi[b]
                    if not allowed[col]:
                        continue
                    if marks[col] != row:
                        marks[col] = row
                        touched[count] = col
                        count += 1
                        work[col] = 0
                    work[col] += av[a]*bv[b]
        for j in range(count):
            col = touched[j]
            if abs(work[col]) >= 1e-13:
                if used == capacity:
                    return data, indices, ptr, False
                data[used], indices[used] = work[col], col
                used += 1
        ptr[row+1] = used
    return data[:used], indices[:used], ptr, True


def _row_projector(action):
    """Accumulate weighted Kronecker entries once per determinant row."""
    left, right = [], []
    capacity = 0
    terms = getattr(action, "_sparse_projector_terms", action._projector_terms)
    for weights, (a, b) in terms:
        if abs(weights[0, 0]) < 1e-14:
            continue
        if not sparse.issparse(a):
            same_strings = a is b
            a = _trim(sparse.csr_matrix(np.real_if_close(a)))
            b = a if same_strings else _trim(sparse.csr_matrix(np.real_if_close(b)))
        a = a*np.real_if_close(weights[0, 0])
        left.append(a); right.append(b)
        capacity += a.nnz*b.nnz
    if not left:
        return sparse.csr_matrix((action.ndet, action.ndet))
    dtype = np.result_type(*[m.dtype for m in left+right])
    packed = []
    for matrices in (left, right):
        offsets = np.r_[0, np.cumsum([m.nnz for m in matrices])]
        packed.extend((np.stack([m.indptr+offset for m, offset in zip(matrices, offsets)]),
                       np.concatenate([m.indices for m in matrices]),
                       np.asarray(np.concatenate([m.data for m in matrices]), dtype=dtype)))
    allowed = np.ones(action.ndet, dtype=bool)
    if hasattr(action, 'string_matrices'):
        for name, orbital, (a, b) in zip(action.names, action.matrices, action.string_matrices):
            target = action.irrep_matrices[name][:, 0]
            if (abs(abs(target[0])-1) < 1e-14 and np.max(abs(target[1:]), initial=0) < 1e-14
                    and np.max(abs(orbital-np.diag(orbital.diagonal())), initial=0) < 1e-14):
                allowed &= (abs(a.diagonal()[:, None]*b.diagonal()[None, :]-target[0]) < 1e-10).ravel()
    data, indices, ptr, success = _projector_rows(*packed, allowed, min(capacity, _MAX_BLOCK_ELEMENTS))
    if not success:
        return None
    matrix = sparse.csr_matrix((data, indices, ptr), shape=(action.ndet, action.ndet))
    matrix.sort_indices()
    return matrix


def row_basis(action, *, spin=None, conn=None, binary=None):
    """Return a content-keyed sparse orthonormal row basis, or a bounded fallback.

    All operation matrices, occupations, determinant ordering and projector
    weights enter the key; rotated orbitals cannot reuse stale transforms.
    """
    digest = hashlib.blake2b(digest_size=24)
    for array in (*action.matrices, action.alpha, action.beta, action.ia,
                  action.ib, action.weights):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(str(spin).encode())
    key = digest.digest()
    if key in _PLAN_CACHE:
        _PLAN_CACHE.move_to_end(key)
        return _PLAN_CACHE[key], True
    projector = _row_projector(action)
    if projector is None:
        return None, False
    order = action.ia * len(action.beta) + action.ib
    if not np.array_equal(order, np.arange(action.ndet)):
        projector = projector[order][:, order]
    projector = _trim(projector)
    projector.data = np.real_if_close(projector.data)
    basis = _component_eigenspace(projector, 1, projector=True)
    if basis is None:
        return None, False
    if basis.shape[1] != action.multiplicity:
        raise ValueError('Spatial row basis has the wrong irrep multiplicity.')
    if spin is not None:
        basis = _spin_basis(basis, conn, binary, spin)
        if basis is None:
            return None, False
    size = basis.data.nbytes + basis.indices.nbytes + basis.indptr.nbytes
    if size > _MAX_PLAN_BYTES:
        return None, False
    while _PLAN_CACHE and (len(_PLAN_CACHE) >= 8 or size + sum(
            q.data.nbytes + q.indices.nbytes + q.indptr.nbytes
            for q in _PLAN_CACHE.values()) > _MAX_PLAN_BYTES):
        _PLAN_CACHE.popitem(last=False)
    _PLAN_CACHE[key] = basis
    return basis, False


def _links(conn, suffix, size, double=False):
    names = ('J', 'p', 'q', 'r', 's', 'phase') if double else ('J', 'p', 'q', 'phase')
    destination = getattr(conn, 'I_' + suffix)
    order = np.argsort(destination, kind='stable')
    entries = np.column_stack([getattr(conn, name + '_' + suffix)[order] for name in names])
    offsets = np.r_[0, np.cumsum(np.bincount(destination, minlength=size))]
    return np.asarray(offsets, dtype=np.int64), np.asarray(entries, dtype=np.int64)


@njit(cache=True)
def _hamiltonian_tile(start, stop, h, same, eri, diagonal, oa, ob, a, b, aa, bb):
    """Emit bounded CSR rows using the existing compact-CI sign convention."""
    na, nb, norb = len(oa), len(ob), h.shape[0]
    ap, av = a; bp, bv = b; aap, aav = aa; bbp, bbv = bb
    ptr = np.zeros(stop-start+1, dtype=np.int64)
    for i in range(start, stop):
        ia, ib = i // nb, i % nb
        la, lb = ap[ia+1]-ap[ia], bp[ib+1]-bp[ib]
        ptr[i-start+1] = ptr[i-start] + 1 + la + lb + la*lb + aap[ia+1]-aap[ia] + bbp[ib+1]-bbp[ib]
    indices = np.empty(ptr[-1], dtype=np.int64)
    data = np.empty(ptr[-1], dtype=np.float64)
    for i in range(start, stop):
        ia, ib = i // nb, i % nb
        k = ptr[i-start]
        indices[k], data[k] = i, diagonal[i]; k += 1
        for l in range(ap[ia], ap[ia+1]):
            ja, p, q, sign = av[l]
            value = h[p,q]
            for r in range(norb):
                if oa[ja,r] and r != q: value += same[p,q,r,r]
                if ob[ib,r]: value += eri[p,q,r,r]
            indices[k], data[k] = ja*nb+ib, -sign*value; k += 1
        for l in range(bp[ib], bp[ib+1]):
            jb, p, q, sign = bv[l]
            value = h[p,q]
            for r in range(norb):
                if ob[jb,r] and r != q: value += same[p,q,r,r]
                if oa[ia,r]: value += eri[p,q,r,r]
            indices[k], data[k] = ia*nb+jb, -sign*value; k += 1
        for l in range(aap[ia], aap[ia+1]):
            ja, p, q, r, s, sign = aav[l]
            indices[k], data[k] = ja*nb+ib, sign*same[p,q,r,s]; k += 1
        for l in range(bbp[ib], bbp[ib+1]):
            jb, p, q, r, s, sign = bbv[l]
            indices[k], data[k] = ia*nb+jb, sign*same[p,q,r,s]; k += 1
        for l in range(ap[ia], ap[ia+1]):
            ja, p, q, sign = av[l]
            for m in range(bp[ib], bp[ib+1]):
                jb, r, s, signb = bv[m]
                indices[k], data[k] = ja*nb+jb, sign*signb*eri[p,q,r,s]; k += 1
    return data, indices, ptr


def assemble_row_hamiltonian(basis, conn, h, same, eri, diagonal):
    """Contract determinant row tiles directly into a selected sparse irrep block.

    Numeric values are rebuilt on every solve; only orbital transformation
    plans are cached. Returns None if reduced fill exceeds the storage budget.
    """
    na, nb = len(conn.alpha_occ), len(conn.beta_occ)
    a, b = _links(conn, 'A', na), _links(conn, 'B', nb)
    aa, bb = _links(conn, 'AA', na, True), _links(conn, 'BB', nb, True)
    result = sparse.csr_matrix((basis.shape[1], basis.shape[1]), dtype=basis.dtype)
    # Bound each Slater–Condon tile to roughly 16 MiB of values and indices.
    max_a, max_b = np.diff(a[0]).max(initial=0), np.diff(b[0]).max(initial=0)
    per_row = 1 + max_a + max_b + max_a*max_b + np.diff(aa[0]).max(initial=0) + np.diff(bb[0]).max(initial=0)
    tile_size = max(1, min(256, 1_000_000 // per_row))
    for start in range(0, na*nb, tile_size):
        stop = min(start+tile_size, na*nb)
        left = basis[start:stop].conj().T
        if not left.nnz:
            continue
        data, indices, ptr = _hamiltonian_tile(start, stop, np.asarray(h.real),
            np.asarray(same.real), np.asarray(eri.real), diagonal,
            conn.alpha_occ, conn.beta_occ, a, b, aa, bb)
        tile = sparse.csr_matrix((data, indices, ptr), shape=(stop-start, na*nb))
        tile.eliminate_zeros()
        result += left @ (tile @ basis)
        if result.nnz > _MAX_BLOCK_ELEMENTS:
            return None
    error = result - result.conj().T
    if np.max(abs(error.data), initial=0) > 1e-9:
        raise ValueError('Symmetry-blocked CI Hamiltonian is not Hermitian.')
    return (result + result.conj().T) * .5


def _spin_basis(basis, conn, binary, spin):
    """Resolve S² inside sparse spatial blocks using the existing CI operator."""
    alpha, beta = conn.alpha_occ.astype(float), conn.beta_occ.astype(float)
    na_e, nb_e = alpha.sum(axis=1)[:, None], beta.sum(axis=1)[None, :]
    diagonal = (.25*(na_e-nb_e)**2 + .5*(na_e+nb_e) - alpha @ beta.T).ravel()
    # S² has only diagonal and opposite-spin exchange terms. Reuse the
    # grouped string links instead of visiting every two-electron excitation.
    active = np.flatnonzero(np.diff(basis.indptr))
    row_map = np.full(len(diagonal), -1, dtype=np.int64)
    row_map[active] = np.arange(len(active))
    compact = basis[active]
    rows = [np.arange(len(active))]
    cols = [rows[0]]
    values = [diagonal[active]]
    entries = len(active)
    na, nb = len(conn.alpha_occ), len(conn.beta_occ)
    norb = conn.alpha_occ.shape[1]
    for p in range(norb):
        for q in range(norb):
            if p == q:
                continue
            a = conn.alpha_order[conn.alpha_offsets[p*norb+q]:conn.alpha_offsets[p*norb+q+1]]
            b = conn.beta_order[conn.beta_offsets[q*norb+p]:conn.beta_offsets[q*norb+p+1]]
            if len(a) and len(b):
                r = row_map[(conn.I_A[a, None]*nb + conn.I_B[b]).ravel()]
                c = row_map[(conn.J_A[a, None]*nb + conn.J_B[b]).ravel()]
                keep = (r >= 0) & (c >= 0)
                entries += np.count_nonzero(keep)
                if entries > _MAX_BLOCK_ELEMENTS:
                    return None
                rows.append(r[keep]); cols.append(c[keep])
                values.append((-conn.phase_A[a, None]*conn.phase_B[b]).ravel()[keep])
    spin_operator = sparse.csr_matrix((np.concatenate(values),
        (np.concatenate(rows), np.concatenate(cols))), shape=(len(active), len(active)))
    operator = _trim(compact.conj().T @ (spin_operator @ compact))
    transform = _component_eigenspace(operator, spin*(spin+1))
    if transform is None:
        return None
    return _trim(basis @ transform)


_ROW_PLAN_CACHE = OrderedDict()


@njit(cache=True, nogil=True)
def _select_rows(data, row, col, entry_order, entry_offsets, row_order,
                 row_offsets, local_rows, local_cols, order, offsets):
    """Pivoted reorthogonalization and checked row inverses for sparse blocks."""
    dimension = len(order)
    capacity = 0
    for i in range(len(offsets)-1):
        size = offsets[i+1]-offsets[i]
        if size > 512:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, dtype=data.dtype), 0., False
        capacity += size*size
        if capacity > 8_000_000:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, dtype=data.dtype), 0., False
    selected = np.empty(dimension, np.int64)
    ir, ic = np.empty(capacity, np.int64), np.empty(capacity, np.int64)
    values = np.empty(capacity, dtype=data.dtype)
    used, condition = 0, 1.
    for group in range(len(offsets)-1):
        start, stop = offsets[group], offsets[group+1]
        size = stop-start
        nrows = row_offsets[group+1]-row_offsets[group]
        dense = np.zeros((nrows, size), dtype=data.dtype)
        for e in range(entry_offsets[group], entry_offsets[group+1]):
            p = entry_order[e]
            dense[local_rows[row[p]], local_cols[col[p]]] = data[p]
        work = dense.copy()
        chosen = np.empty(size, np.int64)
        for k in range(size):
            best, norm = -1, 0.
            for i in range(nrows):
                current = 0.
                for j in range(size):
                    current += abs(work[i,j])**2
                if current > norm:
                    best, norm = i, current
            if best < 0 or norm < 1e-24:
                return selected, ir, ic, values, condition, False
            chosen[k] = best
            pivot = work[best].copy()/np.sqrt(norm)
            # Two passes control loss of orthogonality in pivot selection.
            for repeat in range(2):
                for i in range(nrows):
                    overlap = data[0]*0
                    for j in range(size):
                        overlap += work[i,j]*np.conj(pivot[j])
                    for j in range(size):
                        work[i,j] -= overlap*pivot[j]
            work[best] = 0
        square = dense[chosen].copy()
        inverse = np.linalg.inv(square)
        norm, invnorm, error = 0., 0., 0.
        for i in range(size):
            for j in range(size):
                norm += abs(square[i,j])**2
                invnorm += abs(inverse[i,j])**2
                value = data[0]*0
                for k in range(size):
                    value += square[i,k]*inverse[k,j]
                error = max(error, abs(value-(1 if i == j else 0)))
        bound = np.sqrt(norm*invnorm)
        if bound > 100 or error > 1e-10:
            return selected, ir, ic, values, bound, False
        condition = max(condition, bound)
        for i in range(size):
            selected[start+i] = row_order[row_offsets[group]+chosen[i]]
            for j in range(size):
                if abs(inverse[i,j]) > 1e-14:
                    ir[used], ic[used] = order[start+i], start+j
                    values[used] = inverse[i,j]
                    used += 1
    return selected, ir[:used], ic[:used], values[:used], condition, True


def representative_rows(basis, conn):
    """Cache a stable left inverse of selected rows of an orthonormal basis.

    For invariant H and basis B, H B = B K. If S selects independent rows,
    K x = (S B)^-1 S H B x. This is exact row reconstruction, not sampling or
    an approximate Hamiltonian projection. Pivoted, twice-reorthogonalized
    Gram-Schmidt selects rows within small connected components. Inverse
    residuals and a Frobenius condition bound are checked; rejected selections
    use LAPACK pivoted QR with a spectral condition check.
    """
    from scipy.linalg import qr
    basis.sum_duplicates()
    basis.eliminate_zeros()
    basis.sort_indices()
    digest = hashlib.blake2b(digest_size=24)
    digest.update(str(basis.shape).encode())
    for array in (basis.data, basis.indices, basis.indptr, conn.alpha_occ, conn.beta_occ):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    key = digest.digest()
    if key in _ROW_PLAN_CACHE:
        _ROW_PLAN_CACHE.move_to_end(key)
        return _ROW_PLAN_CACHE[key][0], True
    if basis.nnz > _MAX_BLOCK_ELEMENTS:
        return None, False
    # A star per occupied row has the same column connectivity as B†B,
    # with at most nnz(B) edges rather than quadratic intermediate fill.
    row_sizes = np.diff(basis.indptr)
    active_rows = np.flatnonzero(row_sizes)
    anchors = np.repeat(basis.indices[basis.indptr[active_rows]], row_sizes[active_rows])
    graph = sparse.csr_matrix((np.ones(basis.nnz, dtype=bool), (basis.indices, anchors)),
                              shape=(basis.shape[1], basis.shape[1]))
    count, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
    if np.sum(np.diff(offsets).astype(np.int64)**2) > _MAX_BLOCK_ELEMENTS:
        return None, False
    coo = basis.tocoo()
    entry_components = labels[coo.col]
    entry_order = np.argsort(entry_components, kind='stable')
    entry_offsets = np.r_[0, np.cumsum(np.bincount(entry_components, minlength=count))]
    active_rows = np.flatnonzero(np.diff(basis.indptr))
    row_components = labels[basis.indices[basis.indptr[active_rows]]]
    row_order = active_rows[np.argsort(row_components, kind='stable')]
    row_sizes = np.bincount(row_components, minlength=count)
    row_offsets = np.r_[0, np.cumsum(row_sizes)]
    local_rows = np.empty(basis.shape[0], dtype=np.int64)
    local_rows[row_order] = np.arange(len(row_order)) - np.repeat(row_offsets[:-1], row_sizes)
    local_cols = np.empty(basis.shape[1], dtype=np.int64)
    local_cols[order] = np.arange(len(order)) - np.repeat(offsets[:-1], np.diff(offsets))
    try:
        selected, rows, cols, values, condition, success = _select_rows(
            coo.data, coo.row, coo.col, entry_order, entry_offsets, row_order,
            row_offsets, local_rows, local_cols, order, offsets)
    except np.linalg.LinAlgError:
        success = False
    if not success:
        selected, rows, cols, values = [], [], [], []
        condition = 1.0
        local_plans = {}
        local_plan_bytes = 0
        for group, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            component = order[lo:hi]
            if len(component) > 512:
                return None, False
            entries = entry_order[entry_offsets[group]:entry_offsets[group+1]]
            if len(component) == 1:
                if not len(entries):
                    return None, False
                pivot = entries[np.argmax(abs(coo.data[entries]))]
                rows.append(component[0]); cols.append(len(selected))
                values.append(1 / coo.data[pivot])
                selected.append(coo.row[pivot])
                continue
            active = row_order[row_offsets[group]:row_offsets[group+1]]
            dense = np.zeros((len(active), len(component)), dtype=basis.dtype)
            dense[local_rows[coo.row[entries]], local_cols[coo.col[entries]]] = coo.data[entries]
            panel_key = (dense.shape, dense.tobytes())
            plan = local_plans.get(panel_key)
            if plan is None:
                _, pivots = qr(dense.T, mode='r', pivoting=True)
                chosen = pivots[:len(component)]
                square = dense[chosen]
                block_condition = float(np.linalg.cond(square))
                if block_condition > 100:
                    return None, False
                inverse = np.linalg.solve(square, np.eye(len(component)))
                plan = chosen, inverse, block_condition
                plan_bytes = len(panel_key[1]) + inverse.nbytes + chosen.nbytes
                if local_plan_bytes + plan_bytes <= 8*1024**2:
                    local_plans[panel_key] = plan
                    local_plan_bytes += plan_bytes
            chosen, inverse, block_condition = plan
            condition = max(condition, block_condition)
            i, j = np.nonzero(abs(inverse) > 1e-14)
            rows.extend(component[i]); cols.extend(len(selected)+j); values.extend(inverse[i,j])
            selected.extend(active[chosen])
    inverse = sparse.csr_matrix((np.real_if_close(values), (rows, cols)),
                               shape=(basis.shape[1], basis.shape[1]))
    selected = np.asarray(selected, dtype=np.int64)
    na, nb = len(conn.alpha_occ), len(conn.beta_occ)
    links = (_links(conn, 'A', na), _links(conn, 'B', nb),
             _links(conn, 'AA', na, True), _links(conn, 'BB', nb, True))
    plan = (selected, inverse, links, condition)
    size = selected.nbytes + inverse.data.nbytes + inverse.indices.nbytes + inverse.indptr.nbytes
    size += sum(array.nbytes for link in links for array in link)
    if size <= _MAX_PLAN_BYTES:
        while _ROW_PLAN_CACHE and (len(_ROW_PLAN_CACHE) >= 8 or
                size + sum(entry[1] for entry in _ROW_PLAN_CACHE.values()) > _MAX_PLAN_BYTES):
            _ROW_PLAN_CACHE.popitem(last=False)
        _ROW_PLAN_CACHE[key] = (plan, size)
    return plan, False


@njit(cache=True, nogil=True)
def _representative_sigma(c, selected, nb, norb, diagonal, eri, cross_a, cross_b,
                          a, b, aa, bb, va, vb, vaa, vbb):
    """Contract only representative output rows, without determinant pair tables."""
    ap, av = a; bp, bv = b; aap, aav = aa; bbp, bbv = bb
    out = np.empty(len(selected), dtype=c.dtype)
    for index in range(len(selected)):
        row = selected[index]
        ia, ib = row // nb, row % nb
        value = diagonal[row]*c[row]
        for l in range(ap[ia], ap[ia+1]):
            ja, p, q, sign = av[l]
            value += (va[l]-sign*cross_b[p*norb+q, ib])*c[ja*nb+ib]
        for l in range(bp[ib], bp[ib+1]):
            jb, p, q, sign = bv[l]
            value += (vb[l]-sign*cross_a[p*norb+q, ia])*c[ia*nb+jb]
        for l in range(aap[ia], aap[ia+1]):
            value += vaa[l]*c[aav[l,0]*nb+ib]
        for l in range(bbp[ib], bbp[ib+1]):
            value += vbb[l]*c[ia*nb+bbv[l,0]]
        for l in range(ap[ia], ap[ia+1]):
            ja, p, q, sign = av[l]
            for m in range(bp[ib], bp[ib+1]):
                jb, r, s, signb = bv[m]
                value += sign*signb*eri[p*norb+q, r*norb+s]*c[ja*nb+jb]
        out[index] = value
    return out


def matrix_free_row_action(basis, conn, h, same, eri, diagonal):
    """Prepare an exact representative-row action for a spin/spatial invariant H.

    Only spin-string link coefficients and pair integrals are retained. Neither
    determinant-pair Hamiltonian entries nor a reduced Hamiltonian are built.
    A full source vector is expanded through sparse B; only dim(B) output rows
    are contracted. The preconditioner is diag(B† diag(H) B), not diag(B† H B).
    This adapts the matrix-unit basis in Cotton (1990; module reference) with
    exact row reconstruction. It requires H B to remain in range(B).
    """
    from pyqed.qchem.mcscf.direct_ci import _spin_string_cross_diagonal
    plan, hit = representative_rows(basis, conn)
    if plan is None:
        return None
    selected, inverse, links, condition = plan
    norb = h.shape[0]
    h, same, eri = (np.ascontiguousarray(array.real) for array in (h, same, eri))
    cross_a = _spin_string_cross_diagonal(eri, conn.alpha_occ)
    cross_b = _spin_string_cross_diagonal(eri, conn.beta_occ)
    same_diagonal = np.diagonal(same, axis1=2, axis2=3)
    values = []
    for (_, entries), occupations in zip(links[:2], (conn.alpha_occ, conn.beta_occ)):
        j, p, q, sign = entries.T
        occupation = occupations[j].copy()
        occupation[np.arange(len(j)), q] = 0
        same_value = np.einsum('lr,lr->l', same_diagonal[p,q], occupation)
        values.append(np.ascontiguousarray(-sign*(h[p,q]+same_value)))
    for _, entries in links[2:]:
        j, p, q, r, s, sign = entries.T
        values.append(np.ascontiguousarray(sign*same[p,q,r,s]))
    preconditioner = np.asarray(basis.multiply(basis.conj()).real.T @ diagonal).ravel()

    def matvec(vector):
        vector = np.asarray(vector)
        if vector.ndim == 2:
            return np.column_stack([matvec(vector[:,i]) for i in range(vector.shape[1])])
        source = np.ascontiguousarray(basis @ vector)
        contracted = _representative_sigma(source, selected, len(conn.beta_occ), norb,
            diagonal, eri.reshape(norb*norb, norb*norb), cross_a, cross_b, *links, *values)
        return inverse @ contracted

    return matvec, preconditioner, dict(contraction_plan_cache_hit=hit,
        representative_rows=len(selected), reconstruction_condition=condition)


def full_space_validation_action(h, same, eri, diagonal, conn, binary, workers=None):
    """Reuse the compiled RHF workspace for independent full residual checks."""
    from pyqed.qchem.mcscf.direct_ci import (_rectangular_spin0_pairs,
        _spin0_pair_arrays, _make_sigma_compact_rhf_blas_cpp_matvec,
        _spin_string_cross_diagonal)
    pairs = _rectangular_spin0_pairs(binary)
    if pairs is None:
        return None
    left, right, _ = _spin0_pair_arrays(pairs)
    names = ['alpha_occ', 'beta_occ']
    for suffix in ('A', 'B', 'AA', 'BB'):
        names.extend(f'{name}_{suffix}' for name in
                     (('I', 'J', 'p', 'q', 'phase') if len(suffix)==1
                      else ('I', 'J', 'p', 'q', 'r', 's', 'phase')))
    names.extend(f'{spin}_{name}' for spin in ('alpha', 'beta') for name in
                 ('offsets', 'order', 'ordered_I', 'ordered_J', 'ordered_phase'))
    return _make_sigma_compact_rhf_blas_cpp_matvec(h, same, eri, diagonal, left, right,
        **{name: getattr(conn, name) for name in names},
        alpha_cross_diag=_spin_string_cross_diagonal(eri, conn.alpha_occ),
        beta_cross_diag=_spin_string_cross_diagonal(eri, conn.beta_occ), workers=workers)
