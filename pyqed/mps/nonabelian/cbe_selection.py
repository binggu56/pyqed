"""Reduced shrewd selection for controlled bond expansion.

Gleis, Li and von Delft, PRL 130, 246402 (2023),
https://doi.org/10.1103/PhysRevLett.130.246402, supplement Fig. S-2/Table I.
The four SVD stages operate on multiplicities, with global sector budgets.
Non-Abelian channel dimensions make D'=ceil(D/w) an approximate budget.
Operator halves are supplied by the existing reduced contraction routes;
no full two-site residual or explicit discarded-space projector is formed.
"""
import numpy as np


def _svd(matrix, cutoff):
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(s > cutoff * max(s.max(initial=0), 1e-30)))
    return u[:, :rank], s[:rank], vh[:rank]


def _largest(spectra, count):
    candidates = [(float(s), q, i) for q, values in spectra.items()
                  for i, s in enumerate(values)]
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = {}
    for _, q, i in candidates[:count]:
        selected.setdefault(q, []).append(i)
    return selected


def shrewd_subspace(matrices, halves, *, direction, expansion_size,
                    preselection_size=None, cutoff=1e-12, state_factors=None):
    """Select whole multiplets using the four SVDs of Fig. S-2.

    ``halves[(out_sector, in_sector, channel)] = (L, R)`` means
    H(M)[out] += L @ M[in] @ R.T, in orthonormal coordinates. ``channel``
    retains the shared MPO bond. Preselection is over input multiplicities;
    final selection is over output multiplicities. Exact DD selection is
    recovered when preselection is untruncated, up to numerical rank cutoff.
    """
    if direction not in ('lr', 'rl') or expansion_size < 1:
        raise ValueError('Invalid CBE direction or expansion size')
    if preselection_size is not None and (
            int(preselection_size) != preselection_size or preselection_size < 1):
        raise ValueError('cbe_preselection must be a positive multiplet count')
    if direction == 'lr':
        matrices = {q: m.conj().T for q, m in matrices.items()}
        halves = {key: (r.conj(), l.conj()) for key, (l, r) in halves.items()}
        if state_factors is not None:
            state_factors = {q: (r.conj().T, l.conj().T) for q, (l,r) in state_factors.items()}
    states = ({q: _svd(m, cutoff) for q, m in matrices.items()} if state_factors is None
              else {q: canonical_factors(l, r, cutoff) for q, (l,r) in state_factors.items()})
    left, right, groups = {}, {}, {}
    for key, (l, r) in halves.items():
        out, inp, _ = key
        u, s, vh = states[inp]
        if not len(s):
            continue
        uo, _, vo = states[out]
        x = l @ u
        x -= uo @ (uo.conj().T @ x)
        y = (s[:, None] * vh) @ r.T
        y -= (y @ vo.conj().T) @ vo
        # Relative tests reject numerical remnants of exact kept-space terms.
        if (np.linalg.norm(x) <= cutoff * max(np.linalg.norm(l), 1e-30) or
                np.linalg.norm(y) <= cutoff * max(np.linalg.norm(r)*np.linalg.norm(s), 1e-30)):
            continue
        left[key], right[key] = x, y
        groups.setdefault(inp, []).append(key)
    w = max(1, len({key[2] for key in left}))
    bond_size = sum(len(s) for _, s, _ in states.values())
    budget = (max(1, int(np.ceil(max(bond_size, expansion_size)/w)))
              if preselection_size is None else int(preselection_size))
    pieces, decompositions, spectra = {}, {}, {}
    for inp, keys in groups.items():
        # (a): transfer weights of the right discarded half to the MPS bond.
        u, s, _ = _svd(np.concatenate([right[key] for key in keys], axis=1), cutoff)
        us = u * s
        blocks = [left[key] @ us for key in keys]
        # (b): truncate the MPS bond with the MPO leg still open.
        u, s, _ = _svd(np.concatenate(blocks, axis=0), cutoff)
        decompositions[inp] = (u, s)
        spectra[inp] = s
        pieces[inp] = [(key, block.shape[0]) for key, block in zip(keys, blocks)]
    chosen = _largest(spectra, budget)
    pre = {q: [] for q in matrices}
    for inp, (u, s) in decompositions.items():
        keep = chosen.get(inp, [])
        us = u[:, keep] * s[keep]
        offset = 0
        for key, size in pieces[inp]:
            pre[key[0]].append(us[offset:offset+size])
            offset += size
    complements, spectra = {}, {}
    pre_dimension = 0
    for out, matrix in matrices.items():
        old = states[out][0]
        if not pre[out]:
            complements[out] = np.empty((matrix.shape[0], 0), dtype=matrix.dtype)
            spectra[out] = np.empty(0)
            continue
        # (c): redirect the MPO leg, remove null vectors, safely reorthogonalize.
        candidate = np.concatenate(pre[out], axis=1)
        candidate -= old @ (old.conj().T @ candidate)
        basis, _, _ = _svd(candidate, cutoff)
        pre_dimension += basis.shape[1]
        # (d): close the MPO bond only after projecting to the preselected space.
        projected = np.zeros((basis.shape[1], matrix.shape[1]), dtype=complex)
        for key in left:
            if key[0] == out:
                projected += (basis.conj().T @ left[key]) @ right[key]
        u, s, _ = _svd(projected, cutoff)
        complements[out], spectra[out] = basis @ u, s
    chosen = _largest(spectra, int(expansion_size))
    bases = {}
    for q in matrices:
        columns = np.concatenate((states[q][0], complements[q][:, chosen.get(q, [])]), axis=1)
        bases[q] = np.linalg.qr(columns, mode='reduced')[0]
    return bases, sum(map(len, chosen.values())), {
        'cbe_selection': 'shrewd', 'cbe_preselection_budget': budget,
        'cbe_preselected_multiplets': pre_dimension, 'cbe_operator_channels': w,
    }


def whiten_halves(halves, weights):
    """Apply the factorized canonical metric on both operator indices."""
    scales = {}
    for q, w in weights.items():
        l, r = w[:, 0], w[0]/w[0, 0]
        if not np.allclose(w, l[:, None]*r[None, :], atol=1e-11, rtol=1e-10):
            raise ValueError('Shrewd selection requires a factorized canonical metric')
        scales[q] = l, r
    return {key: (l/scales[key[0]][0][:, None]/scales[key[1]][0][None, :],
                  r/scales[key[0]][1][:, None]/scales[key[1]][1][None, :])
            for key, (l, r) in halves.items()}


def packed_halves(operator, indices):
    """Recover shared one-sided operators from contextual reduced NC routes."""
    action = operator.aux_packed_matvec or operator.packed_matvec
    compiled = getattr(action, 'compiled_factorized_terms', None)
    if compiled is not None and hasattr(compiled, 'one_sided_factors'):
        compiled = compiled.one_sided_factors()
    if compiled is None or not hasattr(compiled, 'middle_channels'):
        raise NotImplementedError('Shrewd CBE requires contextual reduced operator factors')
    basis = operator.basis
    positions = _entry_positions(indices, basis.size)
    halves, seen_left, seen_right = {}, set(), set()
    tables = compiled.plan.left_factor_table, compiled.plan.right_factor_table
    caches = {}, {}
    storage = 0
    for ii, oi, li, ri, channel in zip(compiled.in_indices, compiled.out_indices,
            compiled.left_indices, compiled.right_indices, compiled.middle_channels):
        ie, oe = basis[int(ii)], basis[int(oi)]
        iq, oq = ie.key[2], oe.key[2]
        ir, ic = positions[ie.offset]; orow, oc = positions[oe.offset]
        for cache, table, index in zip(caches, tables, (li, ri)):
            if int(index) not in cache:
                cache[int(index)] = table.factor(int(index))
        lf, rf = caches[0][int(li)], caches[1][int(ri)]
        for component in range(lf.shape[2]):
            key = oq, iq, (int(channel), component)
            if key not in halves:
                shapes = ((indices[oq].shape[0], indices[iq].shape[0]),
                          (indices[oq].shape[1], indices[iq].shape[1]))
                storage += sum(int(np.prod(s)) for s in shapes)*16
                if storage > 256*1024**2:
                    raise MemoryError('CBE one-sided operator workspace exceeds 256 MiB')
                halves[key] = tuple(np.zeros(s, dtype=complex) for s in shapes)
            l, r = halves[key]
            lk = key, orow, ir, int(li)
            rk = key, oc, ic, int(ri)
            if lk not in seen_left:
                value = lf[:, :, component, :, :].transpose(0, 2, 1, 3).reshape(
                    oe.shape[0]*oe.shape[1], ie.shape[0]*ie.shape[1])
                l[orow:orow+value.shape[0], ir:ir+value.shape[1]] += value
                seen_left.add(lk)
            if rk not in seen_right:
                value = rf[component].transpose(2, 0, 3, 1).reshape(
                    oe.shape[2]*oe.shape[3], ie.shape[2]*ie.shape[3])
                r[oc:oc+value.shape[0], ic:ic+value.shape[1]] += value
                seen_right.add(rk)
    return {key: pair for key, pair in halves.items() if np.any(pair[0]) and np.any(pair[1])}


def multiplet_halves(driver, bond, matrices, layouts, lm, rm, left_env, right_env):
    """One-sided reduced CG contractions, retaining the open spatial MPO Leg."""
    from .multiplet_operator import routes, dimension
    sym = driver.mpo.symmetry
    w1, w2 = driver.mpo[bond:bond+2]
    halves, storage = {}, 0
    def get(key):
        nonlocal storage
        if key not in halves:
            oq, iq, _ = key
            shapes = ((matrices[oq].shape[0], matrices[iq].shape[0]),
                      (matrices[oq].shape[1], matrices[iq].shape[1]))
            storage += sum(int(np.prod(s)) for s in shapes)*16
            if storage > 256*1024**2:
                raise MemoryError('CBE one-sided operator workspace exceeds 256 MiB')
            halves[key] = tuple(np.zeros(s, complex) for s in shapes)
        return halves[key]
    for ak, bk, wk, lk, rk, weight in routes(sym, tuple(lm.blocks), tuple(lm.blocks), tuple(w1)):
        if lk not in left_env:
            continue
        oq, iq = ak[2], bk[2]
        rs, ashape = layouts[oq][0][ak]; cs, bshape = layouts[iq][0][bk]
        values = np.einsum('axb,xypq->yapbq', left_env[lk], w1[wk], optimize=True)
        for j, value in enumerate(values):
            get((oq, iq, (rk[1], j)))[0][rs, cs] += weight*value.reshape(
                ashape[0]*ashape[1], bshape[0]*bshape[1])
    for ak, bk, wk, lk, rk, weight in routes(sym, tuple(rm.blocks), tuple(rm.blocks), tuple(w2)):
        if rk not in right_env:
            continue
        oq, iq = ak[0], bk[0]
        rs, ashape = layouts[oq][1][ak]; cs, bshape = layouts[iq][1][bk]
        values = np.einsum('yzrs,czd->yrcsd', w2[wk], right_env[rk], optimize=True)
        for j, value in enumerate(values):
            get((oq, iq, (lk[1], j)))[1][rs, cs] += weight/dimension(sym, oq)*value.reshape(
                ashape[1]*ashape[2], bshape[1]*bshape[2])
    return {key: pair for key, pair in halves.items() if np.any(pair[0]) and np.any(pair[1])}


def state_factors_from_blocks(layouts, left_blocks, right_blocks, weights):
    """Pack the two existing site tensors without forming or SVDing their pair."""
    factors = {}
    for q, (rows, cols) in layouts.items():
        rank = next((v.shape[2] for key, v in left_blocks.items() if key[2] == q), 0)
        nr, nc = weights[q].shape
        left, right = np.zeros((nr, rank), complex), np.zeros((rank, nc), complex)
        if rank:
            for key, (span, _) in rows.items():
                if key in left_blocks:
                    left[span] = left_blocks[key].reshape(span.stop-span.start, rank)
            for key, (span, _) in cols.items():
                if key in right_blocks:
                    right[:, span] = right_blocks[key].reshape(rank, span.stop-span.start)
        w = weights[q]
        factors[q] = (w[:, 0, None]*left, right*(w[0]/w[0, 0])[None, :])
    return factors


def canonical_factors(left, right, cutoff=1e-12):
    """Bond-canonical factors from two thin QRs and a bond-size SVD."""
    ql, rl = np.linalg.qr(left, mode='reduced')
    qr, rr = np.linalg.qr(right.conj().T, mode='reduced')
    u, s, vh = _svd(rl@rr.conj().T, cutoff)
    return ql@u, s, vh@qr.conj().T


def packed_state_factors(sites, basis, indices, weights):
    positions = _entry_positions(indices, basis.size)
    layouts = {q: ({}, {}) for q in indices}
    for e in basis:
        l, p, mid, q, r = e.key
        row, col = positions[e.offset]
        layouts[mid][0][l,p,mid] = (slice(row,row+e.shape[0]*e.shape[1]), None)
        layouts[mid][1][mid,q,r] = (slice(col,col+e.shape[2]*e.shape[3]), None)
    return state_factors_from_blocks(layouts, sites[0].data, sites[1].data, weights)


def _entry_positions(indices, size):
    positions = np.empty((size, 2), dtype=np.int64)
    for ix in indices.values():
        positions[ix, 0] = np.arange(ix.shape[0])[:, None]
        positions[ix, 1] = np.arange(ix.shape[1])[None, :]
    return positions


def one_site_action(halves, bases, sections, direction):
    """Contract the fixed site once and compile the reduced one-site action.

    Reuses the factor-family BLAS executor with identity coordinate maps.
    Its workspace contains only one-site vectors, never expanded pair vectors.
    """
    from pyqed.mps.cpp_davidson import SU2FactorizedFamilyTable
    if SU2FactorizedFamilyTable is None:
        raise RuntimeError('CBE one-site contractions require the compiled factor executor')
    sectors = list(sections)
    numbers = {q: i for i, q in enumerate(sectors)}
    transforms, entries = [], []
    dimension = max(span.stop for span, _ in sections.values())
    diagonal = np.zeros(dimension, complex)
    for q, (span, shape) in sections.items():
        size = span.stop-span.start
        transforms.append(('diagonal', span.start, size,
                           np.arange(size, dtype=np.int64), np.ones(size, complex)))
    for (oq, iq, _), (left, right) in halves.items():
        if direction == 'lr':
            right = bases[oq].T @ right @ bases[iq].conj()
        else:
            left = bases[oq].conj().T @ left @ bases[iq]
        ins, ishape = sections[iq]; outs, oshape = sections[oq]
        if not np.prod(ishape) or not np.prod(oshape):
            continue
        nr, nc = oshape; kr, kc = ishape
        entries.append((numbers[iq], numbers[oq], 0, kr*kc, 0, nr*nc,
                        np.ascontiguousarray(left), np.ascontiguousarray(right.T),
                        (1, nr, 1, 1, 1, kc), (nr, 1, 1, nc),
                        (kr, 1, 1, kc), nr*nc))
        if oq == iq:
            diagonal[outs] += np.outer(np.diag(left), np.diag(right)).ravel()
    table = SU2FactorizedFamilyTable(transforms, entries, dimension)
    return table, diagonal.real


def split_factors(vector, bases, sections, indices, scale, basis, direction):
    """Return unwhitened site factors for truncation without forming their pair."""
    positions = _entry_positions(indices, basis.size)
    factors = {}
    for q, ix in indices.items():
        span, shape = sections[q]
        center = vector[span].reshape(shape)
        left, right = ((center, bases[q].conj().T) if direction == 'lr'
                       else (bases[q], center))
        weights = scale[ix]
        left = left / weights[:, :1]
        right = right / (weights[0:1] / weights[0, 0])
        factors[q] = ({}, {})
        for entry in basis:
            if entry.key[2] != q:
                continue
            l, p, _, r, t = entry.key
            row, col = positions[entry.offset]
            nr = entry.shape[0]*entry.shape[1]
            nc = entry.shape[2]*entry.shape[3]
            factors[q][0][(l, p)] = left[row:row+nr]
            factors[q][1][(r, t)] = right[:, col:col+nc]
    return factors


def projected_site_factors(projection, factors, inverse_right_scale):
    """Map individual site factors to the canonical reduced truncation metric."""
    from .tensor import IdentityBasisTransform
    blocks = []
    for side, entries, basis_map in (
            (0, projection.left_entries, projection.left_basis_map),
            (1, projection.right_entries, projection.right_basis_map)):
        pieces = []
        for entry in entries:
            value = factors[side][entry.child_sectors]
            transform = basis_map[(entry.child_sectors, entry.selected_shape, entry.slot)]
            if not isinstance(transform, IdentityBasisTransform):
                value = transform.T @ value if side == 0 else value @ transform
            pieces.append(value)
        blocks.append(np.concatenate(pieces, axis=side))
    left, right = blocks
    if inverse_right_scale is not None:
        right = right / inverse_right_scale[None, :]
    return left, right
