"""Reduced orbital-overlap operators and LETTA frontier contractions.

All spins are doubled integers. Operator cores store normalized irreducible
local operators, not magnetic components. Their charge is shifted by two per
site, equivalent to the input-hole convention for an operator ket.
"""

from collections import defaultdict
from functools import lru_cache
from math import exp, fsum, lgamma, sqrt

import numpy as np


def _fuse(a, b):
    return range(abs(a - b), a + b + 1, 2)


def _triangle(a, b, c):
    return min(a, b, c) >= 0 and c in _fuse(a, b)


@lru_cache(maxsize=131072)
def _six(a, b, c, d, e, f):
    """Racah finite sum, NIST DLMF 34.4.2; no magnetic indices."""
    triples = ((a, b, c), (a, e, f), (d, b, f), (d, e, c))
    if not all(_triangle(*t) for t in triples):
        return 0.0
    log_prefactor = 0.0
    for x, y, z in triples:
        log_prefactor += 0.5 * (
            lgamma((x + y - z) // 2 + 1) + lgamma((x - y + z) // 2 + 1)
            + lgamma((-x + y + z) // 2 + 1) - lgamma((x + y + z) // 2 + 2)
        )
    lower = tuple(sum(t) // 2 for t in triples)
    upper = ((a + b + d + e) // 2, (b + c + e + f) // 2, (c + a + f + d) // 2)
    return fsum((-1)**z * exp(log_prefactor + lgamma(z + 2)
                             - sum(lgamma(z - x + 1) for x in lower)
                             - sum(lgamma(x - z + 1) for x in upper))
                for z in range(max(lower), min(upper) + 1))


@lru_cache(maxsize=131072)
def _nine(a, b, c, d, e, f, g, h, i):
    """9j as three 6j symbols, NIST DLMF 34.6.2."""
    triples = ((a, b, c), (d, e, f), (g, h, i), (a, d, g), (b, e, h), (c, f, i))
    if not all(_triangle(*t) for t in triples):
        return 0.0
    return fsum((-1)**x * (x + 1) * _six(a, b, c, f, i, x)
                * _six(d, e, f, b, x, h) * _six(g, h, i, x, a, d)
                for x in _fuse(a, i) if x in _fuse(h, d) and x in _fuse(b, f))


_SPIN = (0, 1, 0)
_OPS = tuple((p, q, k) for p in range(3) for q in range(3) for k in _fuse(_SPIN[q], _SPIN[p]))
_GROUPS = defaultdict(list)
for _op in _OPS:
    _GROUPS[(_op[0] - _op[1] + 2, _op[2])].append(_op)
_GROUPS = dict(_GROUPS)
_LOCATION = {op: (sector, i) for sector, ops in _GROUPS.items() for i, op in enumerate(ops)}


@lru_cache(maxsize=None)
def _pairs(charge, spin):
    return tuple((a, b) for a in _OPS for b in _OPS
                 if a[0] - a[1] + b[0] - b[1] + 4 == charge
                 and spin in _fuse(a[2], b[2]))


@lru_cache(maxsize=None)
def _operator_recoupling(a, b, total, jin, jout):
    p, q, k = a
    r, s, l = b
    return sqrt((k + 1) * (l + 1) * (jin + 1) * (jout + 1)) * _nine(
        _SPIN[q], k, _SPIN[p], _SPIN[s], l, _SPIN[r], jin, total, jout,
    )


def _two_orbital_gate(g):
    """Analytic reduced exterior powers in the two-orbital coupled basis."""
    a, b, c, d = np.asarray(g).ravel()
    det = a * d - b * c
    root2 = sqrt(2.0)
    sectors = {
        (0, 0): ([(0, 0)], [[1]]),
        (1, 1): ([(1, 0), (0, 1)], g),
        (2, 0): ([(2, 0), (1, 1), (0, 2)],
                 [[a*a, root2*a*b, b*b], [root2*a*c, a*d+b*c, root2*b*d],
                  [c*c, root2*c*d, d*d]]),
        (2, 2): ([(1, 1)], [[det]]),
        (3, 1): ([(2, 1), (1, 2)], det * np.array([[a, -b], [-c, d]])),
        (4, 0): ([(2, 2)], [[det*det]]),
    }
    return {(spin, out, inp): complex(value)
            for (_, spin), (basis, matrix) in sectors.items()
            for out, row in zip(basis, np.asarray(matrix))
            for inp, value in zip(basis, row)}


def _pair_action(g, charge, spin):
    basis = _pairs(charge, spin)
    gate = _two_orbital_gate(g)
    action = np.zeros((len(basis), len(basis)), dtype=complex)
    for i, (a, b) in enumerate(basis):
        for j, (c, d) in enumerate(basis):
            if a[1] != c[1] or b[1] != d[1]:
                continue
            for jin in _fuse(_SPIN[a[1]], _SPIN[b[1]]):
                for jout in _fuse(_SPIN[a[0]], _SPIN[b[0]]):
                    value = gate.get((jout, (a[0], b[0]), (c[0], d[0])), 0.0)
                    if value:
                        action[i, j] += (_operator_recoupling(a, b, spin, jin, jout) * value
                                         * _operator_recoupling(c, d, spin, jin, jout))
    return action


@lru_cache(maxsize=131072)
def _recouple(left, first, middle, second, right, total):
    return ((-1)**((left + first + second + right)//2)
            * sqrt((middle + 1) * (total + 1)) * _six(left, first, middle, second, right, total))


def _size_guard(size, limit):
    if limit is not None and size > limit:
        raise MemoryError(f"Reduced LETTA overlap needs {size} elements; memory_limit={limit}.")


def _bond_dim(core):
    return sum(next(block.shape[2] for (l, p, r), block in core.items() if r == sector)
               for sector in {key[2] for key in core})


def _left_gauge(cores, i, memory_limit=None):
    site, neighbor = cores[i], cores[i + 1]
    output, transfers = {}, {}
    for sector in sorted({key[2] for key in site}):
        keys = [key for key in site if key[2] == sector]
        shapes = [site[key].shape for key in keys]
        _size_guard(sum(site[key].size for key in keys), memory_limit)
        matrix = np.concatenate([site[key].reshape(-1, site[key].shape[2]) for key in keys])
        q, r = np.linalg.qr(matrix, mode="reduced")
        offset = 0
        for key, shape in zip(keys, shapes):
            size = shape[0] * shape[1]
            output[key] = q[offset:offset + size].reshape(shape[0], shape[1], -1)
            offset += size
        transfers[sector] = r
    cores[i] = output
    cores[i + 1] = {key: np.einsum("ab,bpc->apc", transfers[key[0]], value)
                    for key, value in neighbor.items() if key[0] in transfers}


def _right_gauge(cores, i, memory_limit=None):
    site, neighbor = cores[i], cores[i - 1]
    output, transfers = {}, {}
    for sector in sorted({key[0] for key in site}):
        keys = [key for key in site if key[0] == sector]
        shapes = [site[key].shape for key in keys]
        _size_guard(sum(site[key].size for key in keys), memory_limit)
        weights = [sqrt((key[2][1] + 1) / (sector[1] + 1)) for key in keys]
        matrix = np.concatenate([site[key].reshape(site[key].shape[0], -1) * w
                                 for key, w in zip(keys, weights)], axis=1)
        q, r = np.linalg.qr(matrix.conj().T, mode="reduced")
        right = q.conj().T
        offset = 0
        for key, shape, weight in zip(keys, shapes, weights):
            size = shape[1] * shape[2]
            output[key] = (right[:, offset:offset + size] / weight).reshape(-1, shape[1], shape[2])
            offset += size
        transfers[sector] = r.conj().T
    cores[i] = output
    cores[i - 1] = {key: np.einsum("apb,bc->apc", value, transfers[key[2]])
                    for key, value in neighbor.items() if key[2] in transfers}


def _select_ranks(candidates, cutoff, max_bond, discarded_budget=None):
    """Select whole multiplets by weighted tail, optionally under a hard cap."""
    candidates.sort(reverse=True)
    if discarded_budget is None:
        peak = candidates[0][0] if candidates else 0.0
        selected = [entry for entry in candidates if entry[0] > cutoff**2*peak]
    else:
        # Accumulate the small tail directly; total-minus-prefix loses precision.
        tail = 0.0
        count = len(candidates)
        for weight, _, _ in reversed(candidates):
            if tail + weight > discarded_budget:
                break
            tail += weight
            count -= 1
        selected = candidates[:max(1, count)]
    if max_bond is not None:
        selected = selected[:max_bond]
    ranks = defaultdict(int)
    for _, sector, j in selected or candidates[:1]:
        ranks[sector] = max(ranks[sector], j+1)
    return ranks


def _split(channels, cutoff, max_bond, memory_limit, discarded_budget=None):
    decompositions = {}
    candidates = []
    total_norm = 0.0
    for middle in sorted({key[2] for key in channels}):
        entries = {key: value for key, value in channels.items() if key[2] == middle}
        rows, cols = {}, {}
        for (l, p, _, q, r), block in entries.items():
            rows[(l, p)] = block.shape[:2]
            cols[(q, r)] = block.shape[2:]
        ro, co = {}, {}
        nrow = ncol = 0
        for key, shape in rows.items():
            ro[key] = slice(nrow, nrow + int(np.prod(shape)))
            nrow = ro[key].stop
        for key, shape in cols.items():
            co[key] = slice(ncol, ncol + int(np.prod(shape)))
            ncol = co[key].stop
        _size_guard(nrow * ncol, memory_limit)
        matrix = np.zeros((nrow, ncol), dtype=complex)
        for (l, p, _, q, r), block in entries.items():
            weight = sqrt((r[1] + 1) / (middle[1] + 1))
            matrix[ro[(l, p)], co[(q, r)]] = block.reshape(ro[(l, p)].stop-ro[(l, p)].start, -1) * weight
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        weights = (middle[1] + 1) * s**2
        total_norm += float(weights.sum())
        candidates.extend((float(weight), middle, j) for j, weight in enumerate(weights))
        decompositions[middle] = (u, s, vh, rows, cols, ro, co)
    ranks = _select_ranks(candidates, cutoff, max_bond, discarded_budget)
    left, right = {}, {}
    discarded = 0.0
    for sector, (u, s, vh, rows, cols, ro, co) in decompositions.items():
        rank = ranks[sector]
        discarded += float((sector[1] + 1) * np.dot(s[rank:], s[rank:]))
        if not rank:
            continue
        for (l, p), shape in rows.items():
            left[(l, p, sector)] = u[ro[(l, p)], :rank].reshape(*shape, rank)
        for (q, r), shape in cols.items():
            block = s[:rank, None] * vh[:rank, co[(q, r)]]
            block /= sqrt((r[1] + 1) / (sector[1] + 1))
            right[(sector, q, r)] = block.reshape(rank, *shape)
    return left, right, discarded, total_norm


def _apply_gate(cores, i, gate, cutoff, max_bond, memory_limit):
    coupled = {}
    for (l, p, mid), left in cores[i].items():
        for (mid2, q, r), right in cores[i + 1].items():
            if mid != mid2:
                continue
            block = np.einsum("apb,bqc->apqc", left, right)
            charge = p[0] + q[0]
            for total in _fuse(p[1], q[1]):
                if r[1] not in _fuse(l[1], total):
                    continue
                factor = _recouple(l[1], p[1], mid[1], q[1], r[1], total)
                key = (l, r, charge, total)
                basis = _pairs(charge, total)
                if key not in coupled:
                    _size_guard(left.shape[0] * len(basis) * right.shape[2], memory_limit)
                    coupled[key] = np.zeros((left.shape[0], len(basis), right.shape[2]), complex)
                positions = {pair: j for j, pair in enumerate(basis)}
                for a, op1 in enumerate(_GROUPS[p]):
                    for b, op2 in enumerate(_GROUPS[q]):
                        coupled[key][:, positions[(op1, op2)], :] += factor * block[:, a, b, :]
    channels = {}
    actions = {}
    for (l, r, charge, total), value in coupled.items():
        if (charge, total) not in actions:
            actions[(charge, total)] = _pair_action(gate, charge, total)
        out = np.einsum("ij,ajr->air", actions[(charge, total)], value)
        for index, (op1, op2) in enumerate(_pairs(charge, total)):
            p, a = _LOCATION[op1]
            q, b = _LOCATION[op2]
            for spin in _fuse(l[1], p[1]):
                if r[1] not in _fuse(spin, q[1]):
                    continue
                mid = (l[0] + p[0], spin)
                factor = _recouple(l[1], p[1], spin, q[1], r[1], total)
                if not factor:
                    continue
                key = (l, p, mid, q, r)
                if key not in channels:
                    shape = (out.shape[0], len(_GROUPS[p]), len(_GROUPS[q]), out.shape[2])
                    _size_guard(int(np.prod(shape)), memory_limit)
                    channels[key] = np.zeros(shape, complex)
                channels[key][:, a, b, :] += factor * out[:, index, :]
    cores[i], cores[i + 1], discarded, norm = _split(channels, cutoff, max_bond, memory_limit)
    return discarded, norm


def _unitary_circuit(matrix):
    """Pivot on the diagonal, restoring orbital order after each rotation.

    Adjacent QR elimination can turn weak distant mixing into long sequences
    of large rotations. Diagonal pivots keep those rotations small; swap
    conjugation implements each one without leaving a permuted operator.
    """
    work = np.asarray(matrix, complex).copy()
    n = len(work)
    eliminations = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            a, b = work[col, col], work[row, col]
            if b == 0:
                continue
            norm = np.hypot(abs(a), abs(b))
            c, t = a / norm, b / norm
            gate = np.array([[c.conjugate(), t.conjugate()], [-t, c]])
            work[[col, row], :] = gate @ work[[col, row], :]
            eliminations.append((col, row, gate.conj().T))
    circuit = [("diagonal", np.diag(work).copy())]
    swap = np.array([[0, 1], [1, 0]], complex)
    for col, row, gate in reversed(eliminations):
        circuit.extend(("gate", bond, swap) for bond in range(row-1, col, -1))
        circuit.append(("gate", col, gate))
        circuit.extend(("gate", bond, swap) for bond in range(col+1, row))
    return circuit


def orbital_operator(matrix, *, cutoff=1e-12, max_bond=128, memory_limit=2**24):
    """Reference helper: build a reduced MPO for the fermionic exterior action.

    Used by validation tests, not the biorthogonal LETTA overlap path.

    Operator-Schmidt SVDs truncate whole charge/spin multiplets. The reported
    Frobenius error bound accumulates discarded norms with subsequent gate
    amplification; it excludes floating-point roundoff. No LETTA state is
    transformed. Recoupling uses NIST DLMF 34.4/34.6, an operator-space
    adaptation of the nonorthogonal orbital circuit (Knecht et al., JCTC 12,
    5881, 2016, doi:10.1021/acs.jctc.6b00889), not their MPS algorithm.
    """
    matrix = np.asarray(matrix, complex)
    n = len(matrix)
    if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)) or n < 1:
        raise ValueError("orbital_overlap must be a finite square spatial-orbital matrix.")
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be finite and nonnegative.")
    if max_bond is not None and (isinstance(max_bond, (bool, np.bool_)) or int(max_bond) != max_bond or max_bond < 1):
        raise ValueError("max_bond must be a positive integer or None.")
    cores = [{((2*i, 0), (2, 0), (2*i + 2, 0)): np.array([1, sqrt(2), 1], complex)[None, :, None]}
             for i in range(n)]
    if np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0:
        circuit = [("diagonal", np.diag(matrix))]
    elif np.linalg.norm(matrix.conj().T @ matrix - np.eye(n)) < 1e-14:
        circuit = _unitary_circuit(matrix)
    else:
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        circuit = _unitary_circuit(vh) + [("diagonal", s)] + _unitary_circuit(u)
    center = None
    error_bound = 0.0
    discarded_weights = []
    for kind, *payload in circuit:
        if kind == "diagonal":
            for i, scale in enumerate(payload[0]):
                cores[i] = {key: block * np.array([scale**op[0] for op in _GROUPS[key[1]]])[None, :, None]
                            for key, block in cores[i].items()}
                error_bound *= max(1.0, abs(scale))**2
            center = None
        else:
            i, gate = payload
            if center is None:
                for j in range(n - 1, 0, -1):
                    _right_gauge(cores, j)
                center = 0
            while center < i:
                _left_gauge(cores, center)
                center += 1
            while center > i:
                _right_gauge(cores, center)
                center -= 1
            discarded, norm = _apply_gate(cores, i, gate, cutoff, max_bond, memory_limit)
            center = i + 1
            amplification = float(np.prod(np.maximum(1.0, np.linalg.svd(gate, compute_uv=False)))**2)
            error_bound = amplification * error_bound + sqrt(discarded)
            discarded_weights.append(discarded / norm if norm else 0.0)
    return cores, dict(backend="letta_reduced", fully_reduced=True,
                       mps_conversion=False, determinant_expansion=False,
                       local_magnetic_expansion=False, adjacent_gate_count=len(discarded_weights),
                       max_bond=max_bond, cutoff=cutoff,
                       operator_bond_dimensions=[_bond_dim(core) for core in cores[:-1]],
                       operator_frobenius_error_bound=float(error_bound),
                       gate_discarded_weights=discarded_weights,
                       exact=bool(error_bound == 0.0))


def _advance(state, site, key, previous):
    values = dict(zip(state.frontiers[site], previous))
    ql, qp, qr, condition = key
    endpoint = state._tie_index_by_site[site][qp if state.tie == "physical" else ql]
    if site in values and values[site] != endpoint:
        return None
    for parent, value in zip(state.parent_sets[site], condition):
        if parent in values and values[parent] != value:
            return None
        values[parent] = value
    return tuple(values[parent] for parent in state.frontiers[site + 1])


@lru_cache(maxsize=131072)
def _environment_factor(a, p, b, c, q, d, left, local, right):
    return sqrt((a + 1) * (p + 1) * (d + 1) * (right + 1)) * _nine(
        c, left, a, q, local, p, d, right, b,
    ) * sqrt((local + 1) / (p + 1))


def contract_overlap(bra, ket, cores, *, memory_limit=2**24):
    """Advance independent LETTA tie frontiers using reduced 9j routes."""
    vacuum_b = bra._base_sites[0].legs[0].sectors[0]
    vacuum_k = ket._base_sites[0].legs[0].sectors[0]
    env = {(vacuum_b, vacuum_k, (0, 0), (), ()): np.ones((1, 1, 1), complex)}
    peak = 1
    for i, core in enumerate(cores):
        next_env = {}
        bra_by_left, ket_by_left, ops_by_left = defaultdict(list), defaultdict(list), defaultdict(list)
        for key, block in bra.tensors[i].items():
            bra_by_left[key[0]].append((key, block[:, 0, :]))
        for key, block in ket.tensors[i].items():
            ket_by_left[key[0]].append((key, block[:, 0, :]))
        for (left, physical, right), block in core.items():
            for index, op in enumerate(_GROUPS[physical]):
                ops_by_left[left].append((physical, right, op, block[:, index, :]))
        for (qb, qk, qo, cb, ck), value in env.items():
            for kb, tb in bra_by_left[qb]:
                nb = _advance(bra, i, kb, cb)
                if nb is None:
                    continue
                for kk, tk in ket_by_left[qk]:
                    nk = _advance(ket, i, kk, ck)
                    if nk is None:
                        continue
                    for physical, right, (p, q, rank), op in ops_by_left[qo]:
                        if kb[1].charge != p or kk[1].charge != q:
                            continue
                        factor = _environment_factor(
                            qb.irrep.two_j, kb[1].irrep.two_j, kb[2].irrep.two_j,
                            qk.irrep.two_j, kk[1].irrep.two_j, kk[2].irrep.two_j,
                            qo[1], rank, right[1],
                        )
                        if not factor:
                            continue
                        key = (kb[2], kk[2], right, nb, nk)
                        size = tb.shape[1] * op.shape[1] * tk.shape[1]
                        _size_guard(size, memory_limit)
                        if op.shape == (1, 1):
                            # Identity-metric contraction has a scalar operator bond.
                            contribution = (factor*op[0, 0]) * (tb.conj().T @ value[:, 0, :] @ tk)[:, None, :]
                        else:
                            contribution = factor * np.einsum("abc,ad,be,cf->def", value, tb.conj(), op, tk, optimize=True)
                        if key in next_env:
                            next_env[key] += contribution
                        else:
                            next_env[key] = contribution
        env = next_env
        size = sum(value.size for value in env.values())
        _size_guard(size, memory_limit)
        peak = max(peak, size)
    nbra = int(bra._base_sites[-1].legs[2].dims[bra.target_sector])
    nket = int(ket._base_sites[-1].legs[2].dims[ket.target_sector])
    value = np.zeros((nbra, nket), complex)
    if bra.target_sector == ket.target_sector:
        for (qb, qk, qo, cb, ck), block in env.items():
            if qb == bra.target_sector and qk == ket.target_sector and qo == (2*bra.nsites, 0):
                value += block[:, 0, :]
    return value, peak
