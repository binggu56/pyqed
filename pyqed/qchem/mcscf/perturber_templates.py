"""Shared active-space templates for fully internally contracted Fock PT2.

External orbital labels change integral coefficients and orbital-energy shifts,
but not the active excitation vectors.  Store each active template once rather
than expanding it over every inactive-hole/virtual-particle combination.
This is an algebraic factorization of the spin-free FIC space, not an additional
contraction or an external-orbital cutoff. RAS references retain their ordinary
external first-order space. See Malmqvist et al., JCP 128, 204109 (2008),
doi:10.1063/1.2920188. IPEA and multistate corrections are not provided here.
"""

from functools import lru_cache
from itertools import combinations_with_replacement
from time import perf_counter
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from scipy import sparse


class ReferenceIntegrals(NamedTuple):
    """Exact excitation ERIs and the closed-inactive-shell potential."""

    pair: np.ndarray  # (p i|q j), i,j inactive or active
    core_potential: np.ndarray


def _pattern(holes, particles):
    def relabel(items):
        labels = {orb: i for i, orb in enumerate(sorted(set(items)))}
        return tuple(labels[orb] for orb in items)

    return relabel(holes), relabel(particles)


def _external_key(operator, ncore, ncas):
    return (
        tuple(sorted(q for p, q in operator if q < ncore)),
        tuple(sorted(p for p, q in operator if p >= ncore + ncas)),
    )


def _operators(ncore, ncas, nmo):
    transitions = [(p, q) for q in range(ncore + ncas) for p in range(ncore, nmo)]
    for t in transitions:
        yield (t,)
    for i, right in enumerate(transitions):
        for left in transitions[: i + 1]:
            yield left, right


def _integrals(eri, p, q, r, s):
    if eri.ndim == 4:
        return eri[p, q, r, s]
    return np.einsum("L...,L...->...", eri[:, p, q], eri[:, r, s])


def _hamiltonian_coefficients(h1, eri, ncore, ncas, frozen_core=0):
    """Coefficients of H|0> in the ordered spin-free excitation basis.

    Normal ordering over the closed inactive shell gives its Fock contribution.
    The E_pq E_rs contraction and commutator account for the selected ordering
    of unlike excitation operators; omitting either changes singly external RHSs.
    """
    nmo = len(h1)
    transitions = [
        (p, q) for q in range(frozen_core, ncore + ncas) for p in range(ncore, nmo)
    ]
    p, q = np.asarray(transitions, dtype=int).reshape(-1, 2).T
    if isinstance(eri, ReferenceIntegrals):
        pair = eri.pair[p[:, None], q[:, None], p, q]
        single = h1 + eri.core_potential
        for i in range(ncore, ncore + ncas):
            single -= 0.5 * eri.pair[:, i, :, i]
    else:
        pair = _integrals(eri, p[:, None], q[:, None], p, q)
        single = h1.copy()
        a, b = np.indices(h1.shape)
        for i in range(ncore):
            single += 2 * _integrals(eri, a, b, i, i) - _integrals(eri, a, i, i, b)
        for i in range(ncore, ncore + ncas):
            single -= 0.5 * _integrals(eri, a, i, i, b)
    left, right = np.triu_indices(len(transitions), 1)
    values = 0.5 * pair[left, right]
    mask = q[right] == p[left]
    np.add.at(single, (p[right[mask]], q[left[mask]]), values[mask])
    mask = q[left] == p[right]
    np.add.at(single, (p[left[mask]], q[right[mask]]), -values[mask])
    ids = np.full((nmo, nmo), -1, dtype=int)
    ids[p, q] = np.arange(len(transitions))
    return single, pair, ids


def _spin_excitation(rows, p, q):
    """Source indices, destination bitstrings and phases for a_p^dagger a_q."""
    occupied = (rows & np.uint64(1 << q)) != 0
    if p != q:
        occupied &= (rows & np.uint64(1 << p)) == 0
    indices = np.flatnonzero(occupied)
    selected = rows[indices]
    between = 0 if p == q else ((1 << max(p, q)) - (1 << (min(p, q) + 1)))
    signs = 1 - 2 * (np.bitwise_count(selected & np.uint64(between)) & 1).astype(
        np.int8
    )
    return indices, selected ^ np.uint64((1 << p) ^ (1 << q)), signs


def _excitation_action(state, p, q, nmo):
    """Apply E_pq to sparse arrays, discovering rows outside the reference.

    Square determinant-space operator builders cannot discover these rows.
    Use the same fermionic excitation rules in array form, with the existing
    arbitrary-width dictionary action for bitstrings beyond one word.
    """
    rows, values = state
    if 2 * nmo > 64:
        from .caspt2 import _apply_spatial_one_body

        result = _apply_spatial_one_body(dict(zip(map(int, rows), values)), p, q, nmo)
        return np.array(list(result), dtype=object), np.array(
            list(result.values()), dtype=float
        )
    pieces, amplitudes = [], []
    for spin in (0, 1):
        indices, destinations, signs = _spin_excitation(
            rows, p + spin * nmo, q + spin * nmo
        )
        pieces.append(destinations)
        amplitudes.append(values[indices] * signs)
    bits, inverse = np.unique(np.concatenate(pieces), return_inverse=True)
    values = np.bincount(
        inverse, weights=np.concatenate(amplitudes), minlength=len(bits)
    )
    nonzero = values != 0
    return bits[nonzero], values[nonzero]


class _ActiveTemplate:
    def __init__(
        self,
        pattern,
        binary,
        ci,
        fock_active,
        screen_tol,
        lindep_tol,
        memory_bytes=None,
        *,
        ic_convention="projected",
        norm_tol=0.0,
        fock_contraction="operator",
    ):
        from .caspt2 import (
            _embed_active_determinants,
            _one_body_sparse_matrix_in_determinant_space,
        )

        self.pattern = pattern
        holes, particles = pattern
        self.ncore = len(set(holes))
        self.ncas = binary.shape[-1]
        self.nmo = self.ncore + self.ncas + len(set(particles))
        key = holes, tuple(self.ncore + self.ncas + p for p in particles)
        bits = np.array(
            _embed_active_determinants(binary, self.ncore, self.ncas, self.nmo),
            dtype=np.uint64 if 2 * self.nmo <= 64 else object,
        )
        ci = np.asarray(ci, dtype=float)
        selected = abs(ci) > screen_tol
        reference = bits[selected], ci[selected]
        self.operators, columns = [], []
        nnz = 0
        last_right, first_action = None, None
        for operator in _operators(self.ncore, self.ncas, self.nmo):
            if _external_key(operator, self.ncore, self.ncas) != key:
                continue
            right = operator[-1]
            if right != last_right:
                first_action = _excitation_action(reference, *right, self.nmo)
                last_right = right
            state = (
                first_action
                if len(operator) == 1
                else _excitation_action(first_action, *operator[0], self.nmo)
            )
            keep = abs(state[1]) > screen_tol
            state = state[0][keep], state[1][keep]
            self.operators.append(operator)
            nnz += len(state[0])
            # Sparse columns, assembly copies and canonical metric work arrays.
            estimate = 128 * nnz + 64 * len(self.operators) ** 2
            if memory_bytes is not None and estimate > memory_bytes:
                raise MemoryError(
                    "Shared active template exceeds max_memory_mb during assembly."
                )
            columns.append(state)
        rows = (
            np.unique(np.concatenate([c[0] for c in columns])) if columns else bits[:0]
        )
        rr = (
            np.concatenate([np.searchsorted(rows, c[0]) for c in columns])
            if columns
            else []
        )
        cc = np.repeat(np.arange(len(columns)), [len(c[0]) for c in columns])
        vv = np.concatenate([c[1] for c in columns]) if columns else []
        raw = sparse.coo_matrix((vv, (rr, cc)), shape=(len(rows), len(columns))).tocsc()
        norms = np.sqrt(np.asarray(raw.power(2).sum(axis=0)).ravel())
        nonzero = norms > screen_tol
        original = raw[:, nonzero].multiply(1 / norms[nonzero]).tocsr()
        self.norms = norms
        self.nonzero = nonzero
        metric_raw = raw
        if ic_convention == "molcas" and np.sum(binary[0]) > 0:
            # Molcas uses the ordered double generators, with singles folded
            # into their number-operator traces (SPEC1A/C/D). Keep the original
            # Hamiltonian expansion independently: changing its raw basis must
            # not drop the single-excitation part of H|0>.
            if (len(holes), len(particles)) in ((1, 0), (0, 1), (1, 1)):
                selected = [j for j, op in enumerate(self.operators) if len(op) == 2]
                columns = []
                lookup = {op: j for j, op in enumerate(self.operators)}
                for j in selected:
                    op = self.operators[j]
                    column = raw[:, j]
                    if not holes and op[0][0] < self.ncore + self.ncas:
                        # E_rt E_uv = E_uv E_rt + delta_tu E_rv.
                        (u, v), (r, t) = op
                        if t == u:
                            column = column + raw[:, lookup[((r, v),)]]
                    columns.append(column)
                metric_raw = sparse.hstack(columns, format="csr")
        metric_norms2 = np.asarray(metric_raw.power(2).sum(axis=0)).ravel()
        selected = metric_norms2 > max(norm_tol, screen_tol**2)
        scale = 1 / np.sqrt(metric_norms2[selected])
        if ic_convention == "molcas":
            # SBDIAG_SER's small deterministic diagonal scaling.
            scale *= 1 + (np.flatnonzero(selected) + 1) * 3e-6
        self.raw = metric_raw[:, selected].multiply(scale).tocsr()
        self.transpose = self.raw.T.tocsr()
        metric = (self.transpose @ self.raw).toarray()
        values, vectors = np.linalg.eigh((metric + metric.T) * 0.5)
        keep = values > max(lindep_tol, np.finfo(float).eps * len(metric))
        self.transform = vectors[:, keep] / np.sqrt(values[keep])
        self.rows = rows.tolist()
        self.rank = int(np.count_nonzero(keep))
        # Q^T raw gives the exact Hamiltonian RHS without determinant couplings.
        self.rhs_transform = self.transform.T @ (self.transpose @ original).toarray()
        fock = np.zeros((self.nmo, self.nmo))
        active = slice(self.ncore, self.ncore + self.ncas)
        fock[active, active] = fock_active
        if 2 * self.nmo <= 64 and np.array_equal(
            fock_active, np.diag(np.diag(fock_active))
        ):
            energies = np.zeros(len(rows))
            for a, energy in enumerate(np.diag(fock_active)):
                for spin in (0, 1):
                    energies += energy * (
                        (rows >> np.uint64(spin * self.nmo + self.ncore + a))
                        & np.uint64(1)
                    )
            projected = (
                self.transpose @ self.raw.multiply(energies[:, None]).tocsr()
            ).toarray()
        elif fock_contraction == "density":
            from .transition_density import one_body_moment

            projected = one_body_moment(np.asarray(rows), self.raw, fock)
        else:
            action, _ = _one_body_sparse_matrix_in_determinant_space(
                self.rows, fock, fock, self.nmo
            )
            projected = (self.transpose @ action @ self.raw).toarray()
        self.fock = self.transform.T @ projected @ self.transform
        self.fock = (self.fock + self.fock.T) * 0.5
        self.values, self.vectors = np.linalg.eigh(self.fock)

    def orbital_map(self, holes, particles, ncore):
        return np.r_[
            sorted(set(holes)),
            np.arange(ncore, ncore + self.ncas),
            sorted(set(particles)),
        ].astype(int)

    def rhs(self, groups, ncore, single, pair, transition_ids):
        result = np.zeros((len(groups), self.rank))
        for start in range(0, len(groups), 1024):
            batch = groups[start : start + 1024]
            maps = np.empty((len(batch), self.nmo), dtype=int)
            if self.ncore:
                maps[:, : self.ncore] = np.array([h for h, v in batch])[:, : self.ncore]
            maps[:, self.ncore : self.ncore + self.ncas] = ncore + np.arange(self.ncas)
            nv = self.nmo - self.ncore - self.ncas
            if nv:
                maps[:, self.ncore + self.ncas :] = np.array([v for h, v in batch])[
                    :, :nv
                ]
            coeff = np.empty((len(batch), len(self.operators)))
            for j, operator in enumerate(self.operators):
                mapped = [(maps[:, p], maps[:, q]) for p, q in operator]
                if len(operator) == 1:
                    coeff[:, j] = single[mapped[0]]
                else:
                    ids = [transition_ids[t] for t in mapped]
                    coeff[:, j] = pair[ids[0], ids[1]]
                    if operator[0] == operator[1]:
                        coeff[:, j] *= 0.5
            coeff = coeff[:, self.nonzero] * self.norms[self.nonzero]
            result[start : start + len(batch)] = coeff @ self.rhs_transform.T
        return result


class _RDMTemplate(_ActiveTemplate):
    """Internally contracted coordinates built entirely from active RDMs."""

    def __init__(
        self,
        pattern,
        engine,
        energies,
        screen_tol,
        lindep_tol,
        memory_bytes=None,
        *,
        ic_convention="projected",
        norm_tol=0.0,
    ):
        self.pattern = pattern
        self.engine = engine
        holes, particles = pattern
        self.ncore = len(set(holes))
        self.ncas = engine.ncas
        self.nmo = self.ncore + self.ncas + len(set(particles))
        key = holes, tuple(self.ncore + self.ncas + p for p in particles)
        self.operators = [
            op
            for op in _operators(self.ncore, self.ncas, self.nmo)
            if _external_key(op, self.ncore, self.ncas) == key
        ]
        count = len(self.operators)
        if memory_bytes is not None and 96 * count**2 > memory_bytes:
            raise MemoryError("RDM template matrices exceed max_memory_mb.")
        gram = engine.matrix(self.operators, self.operators, self.ncore)
        gram = (gram + gram.T) * 0.5
        self.norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
        self.nonzero = self.norms > screen_tol
        coefficients = sparse.eye(count, format="csc")
        if (
            ic_convention == "molcas"
            and (len(holes), len(particles)) in ((1, 0), (0, 1), (1, 1))
            and self.ncas
        ):
            indices = [j for j, op in enumerate(self.operators) if len(op) == 2]
            coefficients = coefficients[:, indices].tolil()
            lookup = {op: j for j, op in enumerate(self.operators)}
            for col, j in enumerate(indices):
                op = self.operators[j]
                if not holes and op[0][0] < self.ncore + self.ncas:
                    (u, v), (r, t) = op
                    if t == u:
                        coefficients[lookup[((r, v),)], col] += 1
            coefficients = coefficients.tocsc()
        metric = np.asarray(coefficients.T @ gram @ coefficients)
        norms2 = np.diag(metric)
        selected = norms2 > max(norm_tol, screen_tol**2)
        scale = 1 / np.sqrt(norms2[selected])
        if ic_convention == "molcas":
            scale *= 1 + (np.flatnonzero(selected) + 1) * 3e-6
        coefficients = coefficients[:, selected].multiply(scale).tocsr()
        metric = metric[np.ix_(selected, selected)] * scale[:, None] * scale[None, :]
        values, vectors = np.linalg.eigh((metric + metric.T) * 0.5)
        keep = values > max(lindep_tol, np.finfo(float).eps * len(metric))
        self.transform = vectors[:, keep] / np.sqrt(values[keep])
        self.projector = coefficients @ self.transform
        self.rank = int(np.count_nonzero(keep))
        self.rhs_transform = (
            self.projector.T @ gram[:, self.nonzero] / self.norms[self.nonzero]
        )
        eps = np.r_[
            np.zeros(self.ncore), energies, np.zeros(self.nmo - self.ncore - self.ncas)
        ]
        shifts = np.array(
            [sum(eps[p] - eps[q] for p, q in op) for op in self.operators]
        )
        moment = engine.matrix(
            self.operators, self.operators, self.ncore, weighted=True, centered=True
        )
        moment += 0.5 * (shifts[:, None] + shifts[None, :]) * gram
        self.fock = self.projector.T @ moment @ self.projector + engine.center * np.eye(
            self.rank
        )
        self.fock = (self.fock + self.fock.T) * 0.5
        self.values, self.vectors = np.linalg.eigh(self.fock)
        self.rows = []
        self.raw = sparse.csr_matrix((0, coefficients.shape[1]))
        self.transpose = self.raw.T.tocsr()


@lru_cache(maxsize=1)
def _grouped_external_spaces(ncore, nmo, ncas, frozen_core):
    """Immutable external labels, shared across roots of one orbital layout."""
    global _fock_topology_cache
    _fock_topology_cache = None
    groups = {}
    for nh, nv in ((2, 2), (2, 1), (1, 2), (2, 0), (0, 2), (1, 1), (1, 0), (0, 1)):
        hs = list(combinations_with_replacement(range(frozen_core, ncore), nh))
        vs = list(combinations_with_replacement(range(ncore + ncas, nmo), nv))
        for h in hs:
            for v in vs:
                groups.setdefault(_pattern(h, v), []).append((h, v))
    return MappingProxyType({p: tuple(g) for p, g in groups.items()})


def _mapped_rows(template, holes, particles, ncore, nmo):
    """Embed a template and its fermion phases in a larger orbital space."""
    mapping = template.orbital_map(holes, particles, ncore)
    selected = set(holes)
    spectators = [i for i in range(ncore) if i not in selected]
    filled = sum((1 << i) | (1 << (nmo + i)) for i in spectators)
    if 2 * nmo <= 64:
        source = np.asarray(template.rows, dtype=np.uint64)
        mapped = np.full(len(source), filled, dtype=np.uint64)
        parity = np.zeros(len(source), dtype=np.int8)
        for spin in (0, 1):
            for i, orbital in enumerate(mapping):
                occupied = (source >> np.uint64(spin * template.nmo + i)) & np.uint64(1)
                mapped |= occupied << np.uint64(spin * nmo + int(orbital))
                if i < template.ncore and sum(j > orbital for j in spectators) % 2:
                    parity ^= 1 - occupied.astype(np.int8)
        return mapped.tolist(), 1.0 - 2 * parity
    rows, phases = [], []
    for det in template.rows:
        mapped = filled
        parity = 0
        for spin in (0, 1):
            for i, orbital in enumerate(mapping):
                if (det >> (spin * template.nmo + i)) & 1:
                    mapped |= 1 << (spin * nmo + int(orbital))
                elif i < template.ncore:
                    parity += sum(j > orbital for j in spectators)
        rows.append(mapped)
        phases.append(1.0 if parity % 2 == 0 else -1.0)
    return rows, np.asarray(phases)


def _transition_key(source, target, p, q, ncore, ncas):
    hs = sorted(set(source[0] + target[0]))
    vs = sorted(set(source[1] + target[1]))
    mapping = {orb: i for i, orb in enumerate(hs)}
    mapping.update((ncore + a, len(hs) + a) for a in range(ncas))
    mapping.update((orb, len(hs) + ncas + i) for i, orb in enumerate(vs))
    s = tuple(mapping[i] for i in source[0]), tuple(mapping[i] for i in source[1])
    t = tuple(mapping[i] for i in target[0]), tuple(mapping[i] for i in target[1])
    return s, t, mapping[p], mapping[q]


def _transition_embedding(template, group, nc, nmo):
    if isinstance(template, _RDMTemplate):
        mapping = template.orbital_map(*group, nc)
        words = [
            tuple((int(mapping[p]), int(mapping[q])) for p, q in op)
            for op in template.operators
        ]
        return template.engine.prepare(words, nc, True), template.engine.prepare(
            words, nc
        )
    rows, phases = _mapped_rows(template, *group, nc, nmo)
    rows = np.asarray(rows, dtype=np.uint64 if 2 * nmo <= 64 else object)
    return rows, phases, np.argsort(rows)


def _transition_matrix_reference(source, target, key, embedding=None):
    from .casci import _annihilate_bit, _create_bit

    s, t, p, q = key
    nc = len(set(s[0] + t[0]))
    nmo = nc + source.ncas + len(set(s[1] + t[1]))
    embedding = embedding or _transition_embedding
    sr, sp, _ = embedding(source, s, nc, nmo)
    tr, tp, order = embedding(target, t, nc, nmo)
    rr, cc, vv = [], [], []
    if 2 * nmo <= 64:
        sorted_rows = tr[order]
        for spin in (0, 1):
            indices, bra, signs = _spin_excitation(sr, p + spin * nmo, q + spin * nmo)
            positions = np.searchsorted(sorted_rows, bra)
            found = positions < len(tr)
            indices, bra, signs, positions = (
                x[found] for x in (indices, bra, signs, positions)
            )
            found = sorted_rows[positions] == bra
            i, j = order[positions[found]], indices[found]
            rr.append(i)
            cc.append(j)
            vv.append(signs[found] * sp[j] * tp[i])
        rr, cc, vv = (np.concatenate(parts) for parts in (rr, cc, vv))
    else:
        target_index = {int(det): i for i, det in enumerate(tr)}
        for j, det in enumerate(sr):
            for spin in (0, 1):
                ket, sign1 = _annihilate_bit(int(det), spin * nmo + q)
                if not sign1:
                    continue
                bra, sign2 = _create_bit(ket, spin * nmo + p)
                i = target_index.get(bra)
                if sign2 and i is not None:
                    rr.append(i)
                    cc.append(j)
                    vv.append(sign1 * sign2 * sp[j] * tp[i])
    operator = sparse.coo_matrix((vv, (rr, cc)), shape=(len(tr), len(sr))).tocsr()
    projected = (target.transpose @ operator @ source.raw).toarray()
    return target.transform.T @ projected @ source.transform


def _transition_matrix(source, target, key, embedding=None, factor=None):
    if isinstance(source, _RDMTemplate):
        s, t, p, q = key
        nc = len(set(s[0] + t[0]))
        nmo = nc + source.ncas + len(set(s[1] + t[1]))
        embedding = embedding or _transition_embedding
        gram = source.engine.contract(
            embedding(target, t, nc, nmo)[0],
            embedding(source, s, nc, nmo)[1],
            nc,
            middle=((p, q),),
        )
        return target.projector.T @ gram @ source.projector
    from .transition_density import annihilation_factor, contract_factors

    s, t, p, q = key
    nc = len(set(s[0] + t[0]))
    nmo = nc + source.ncas + len(set(s[1] + t[1]))
    embedding = embedding or _transition_embedding
    if factor is None:

        def factor(template, group, nc, nmo, orbital):
            rows, phases, _ = embedding(template, group, nc, nmo)
            return annihilation_factor(rows, template.raw, orbital, phases)

    projected = sum(
        contract_factors(
            factor(target, t, nc, nmo, p + spin * nmo),
            factor(source, s, nc, nmo, q + spin * nmo),
        )
        for spin in (0, 1)
    )
    return target.transform.T @ projected @ source.transform


_fock_topology_cache = None


def _fock_topology(groups, fock, ncore, ncas, memory_bytes=None, cache_bytes=0):
    """Reuse only external indices; refresh every Fock coefficient per root."""
    global _fock_topology_cache
    key = (
        ncore,
        ncas,
        len(fock),
        tuple(groups.items()),
        (abs(fock) >= 1e-14).tobytes(),
    )
    cached = _fock_topology_cache
    if cached is None or cached[0] != key:
        _fock_topology_cache = None
        cached = None
        indices, entries, peak = _build_fock_topology(
            groups, fock, ncore, ncas, memory_bytes, cache_bytes
        )
        for rr, cc, external, transition in indices.values():
            rr.flags.writeable = cc.flags.writeable = external.flags.writeable = False
        cached = (key, indices, entries, peak)
        _fock_topology_cache = cached
    _, indices, entries, peak = cached
    if memory_bytes is not None and peak + cache_bytes > memory_bytes:
        raise MemoryError(
            "Shared FIC Fock graph exceeds max_memory_mb during assembly."
        )
    result = {}
    for k, (rr, cc, external, transition) in indices.items():
        values = fock[external[0], external[1]] if k[2] == 2 else external
        result[k] = (rr, cc, values, transition)
    return result, entries


def _build_fock_topology(groups, fock, ncore, ncas, memory_bytes=None, cache_bytes=0):
    """Batch external insertions against sorted, packed orbital-label keys."""
    if not groups:
        return {}, 0, 0
    patterns = list(groups)
    sizes = np.array([len(groups[p]) for p in patterns])
    base = len(fock) + 1
    if base**4 > np.iinfo(np.int64).max:
        raise OverflowError("External orbital keys exceed int64.")
    powers = np.array([base**3, base**2, base, 1], dtype=np.int64)
    lookup_bytes = int(sizes.sum()) * 128
    peak = 0

    def guard(count):
        nonlocal peak
        peak = max(peak, lookup_bytes + count * 128)
        if (
            memory_bytes is not None
            and lookup_bytes + cache_bytes + count * 128 > memory_bytes
        ):
            raise MemoryError(
                "Shared FIC Fock graph exceeds max_memory_mb during assembly."
            )

    guard(0)
    labels = [
        np.array(
            [h + (-1,) * (2 - len(h)) + v + (-1,) * (2 - len(v)) for h, v in groups[p]],
            dtype=np.int64,
        )
        for p in patterns
    ]
    codes = np.concatenate([(a + 1) @ powers for a in labels])
    order = np.argsort(codes)
    codes = codes[order]
    ids = np.repeat(np.arange(len(patterns)), sizes)[order]
    offsets = np.r_[0, np.cumsum(sizes)]
    correlated = np.unique(np.concatenate([a[:, :2].ravel() for a in labels]))
    correlated = correlated[correlated >= 0]
    virtuals = np.arange(ncore + ncas, len(fock))
    active_cores = correlated[
        np.any(abs(fock[ncore : ncore + ncas, correlated]) >= 1e-14, axis=0)
    ]
    active_virtuals = virtuals[
        np.any(abs(fock[virtuals, ncore : ncore + ncas]) >= 1e-14, axis=1)
    ]
    i, r = np.nonzero(abs(fock[np.ix_(virtuals, correlated)].T) >= 1e-14)
    insertions = [
        (np.full(len(active_cores), ncore), active_cores),
        (active_virtuals, np.full(len(active_virtuals), ncore)),
        (virtuals[r], correlated[i]),
    ]
    result = {}
    entries = 0
    for pattern, source in zip(patterns, labels):
        nh, nv = map(len, pattern)
        for kind, (ps, qs) in enumerate(insertions):
            if (
                (kind != 1 and nh == 2)
                or (kind != 0 and nv == 2)
                or (kind < 2 and not ncas)
                or not len(ps)
            ):
                continue
            count = len(source) * len(ps)
            guard(entries + count)
            cc = np.repeat(np.arange(len(source)), len(ps))
            p = np.tile(ps, len(source))
            q = np.tile(qs, len(source))
            target = source[cc].copy()
            ho = np.zeros(count, dtype=int)
            vo = ho.copy()
            if kind != 1:
                if nh:
                    ho = np.sign(q - target[:, 0])
                    target[:, 1] = np.maximum(q, target[:, 0])
                    target[:, 0] = np.minimum(q, target[:, 0])
                else:
                    target[:, 0] = q
            if kind != 0:
                if nv:
                    vo = np.sign(p - target[:, 2])
                    target[:, 3] = np.maximum(p, target[:, 2])
                    target[:, 2] = np.minimum(p, target[:, 2])
                else:
                    target[:, 2] = p
            wanted = (target + 1) @ powers
            positions = np.searchsorted(codes, wanted)
            keep = positions < len(codes)
            positions = np.minimum(positions, len(codes) - 1)
            keep &= codes[positions] == wanted
            cc, p, q, ho, vo, positions = (
                a[keep] for a in (cc, p, q, ho, vo, positions)
            )
            target_ids = ids[positions]
            rr = order[positions] - offsets[target_ids]
            cases = (target_ids * 3 + ho + 1) * 3 + vo + 1
            for case in np.unique(cases):
                selected = np.flatnonzero(cases == case)
                first = selected[0]
                dest = patterns[target_ids[first]]
                key = pattern, dest, kind, int(ho[first]), int(vo[first])
                transition = _transition_key(
                    groups[pattern][cc[first]],
                    groups[dest][rr[first]],
                    int(p[first]),
                    int(q[first]),
                    ncore,
                    ncas,
                )
                external = (
                    q[selected]
                    if kind == 0
                    else (
                        p[selected]
                        if kind == 1
                        else np.array([p[selected], q[selected]])
                    )
                )
                result[key] = (rr[selected], cc[selected], external, transition)
                entries += len(selected)
    return result, entries, peak


def _fock_connections(
    groups,
    templates,
    fock,
    ncore,
    ncas,
    memory_bytes=None,
    *,
    ic_convention="projected",
    fock_contraction="operator",
):
    """Factor the external-label graph from shared active transition matrices."""
    from functools import lru_cache

    # All active-index transitions of a template pair use the same determinant
    # embedding. Reuse its phases and lookup rather than reconstructing them
    # for each Fock matrix element. Bound the temporary cache explicitly.
    cache_bytes = (
        160
        * max((len(t.rows) for t in templates.values()), default=0)
        * min(8, len(templates))
    )
    if fock_contraction == "rdm":
        cache_bytes = (
            512 * max((len(t.operators) for t in templates.values()), default=0) * 8
        )

    @lru_cache(maxsize=8)
    def embedding(template, group, nc, nmo):
        return _transition_embedding(template, group, nc, nmo)

    cache_bytes += (8 if fock_contraction == "density" else 0) * max(
        (
            t.raw.data.nbytes
            + t.raw.indices.nbytes
            + t.raw.indptr.nbytes
            + 16 * len(t.rows)
            for t in templates.values()
        ),
        default=0,
    )

    @lru_cache(maxsize=4)
    def factor(template, group, nc, nmo, orbital):
        from .transition_density import annihilation_factor

        rows, phases, _ = embedding(template, group, nc, nmo)
        return annihilation_factor(rows, template.raw, orbital, phases)

    connections, entries = _fock_topology(
        groups, fock, ncore, ncas, memory_bytes, cache_bytes
    )
    result = []
    graph_bytes = 0
    for (s, t, kind, _, _), (rr, cc, external, base) in connections.items():
        rr, cc = np.asarray(rr), np.asarray(cc)
        external = np.asarray(external, dtype=float if kind == 2 else int)
        # The external topology is independent of the active index. Expand
        # only its Fock coefficients and shared active transition matrix.
        for a in range(ncas if kind < 2 else 1):
            key = (
                *base[:2],
                base[2] + (a if kind == 0 else 0),
                base[3] + (a if kind == 1 else 0),
            )
            values = (
                fock[ncore + a, external]
                if kind == 0
                else fock[external, ncore + a] if kind == 1 else external
            )
            keep = abs(values) >= 1e-14
            if not np.any(keep):
                continue
            estimate = (
                16 * np.count_nonzero(keep) + 8 * templates[s].rank * templates[t].rank
            )
            if (
                memory_bytes is not None
                and entries * 128 + cache_bytes + graph_bytes + estimate > memory_bytes
            ):
                raise MemoryError(
                    "Shared FIC Fock graph exceeds max_memory_mb during assembly."
                )
            matrix = (
                _transition_matrix(templates[s], templates[t], key, embedding, factor)
                if fock_contraction in ("density", "rdm")
                else _transition_matrix_reference(
                    templates[s], templates[t], key, embedding
                )
            )
            if ic_convention == "molcas" and (
                len(s[0]),
                len(s[1]),
                len(t[0]),
                len(t[1]),
            ) == (1, 0, 2, 1):
                # OpenMolcas SGM cases 4/5, MKLIST 14/15 (A <-> E+/-).
                matrix *= np.sqrt(2.0)
            if np.max(abs(matrix), initial=0) < 1e-13:
                continue
            matrix = templates[t].vectors.T @ matrix @ templates[s].vectors
            graph = sparse.coo_matrix(
                (values[keep], (rr[keep], cc[keep])),
                shape=(len(groups[t]), len(groups[s])),
            ).tocsr()
            graph_bytes += (
                graph.data.nbytes
                + graph.indices.nbytes
                + graph.indptr.nbytes
                + matrix.nbytes
            )
            result.append((s, t, graph, matrix))
    return result


def solve_shared_fois(
    binary,
    ci,
    ncore,
    ncas,
    h1,
    eri,
    fock,
    reference_active_fock,
    *,
    frozen_core=0,
    imaginary_shift=0.0,
    real_shift=0.0,
    screen_tol=1e-12,
    lindep_tol=1e-10,
    solver_tol=1e-10,
    max_iterations=500,
    max_memory_mb=2048.0,
    progress=None,
    ic_convention="projected",
    norm_tol=0.0,
    fock_contraction="operator",
    transition_ci=None,
):
    """Solve the complete spin-free external FIC space with shared templates.

    The blockwise imaginary shift follows Forsberg and Malmqvist, CPL 274,
    196 (1997), doi:10.1016/S0009-2614(97)00669-6, as in the direct solver.
    Inactive and virtual blocks must be semicanonical. ``projected`` is the
    exact Galerkin projection of F; ``molcas`` instead follows OpenMolcas's
    ordered metrics and A/E coupling convention (SGM cases 4/5), which differs
    from that projection. See https://github.com/Molcas/OpenMolcas/tree/master/src/caspt2.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    from .caspt2 import CASPT2_PERTURBER_CLASSES, CASPT2Component

    if ic_convention not in ("projected", "molcas"):
        raise ValueError("ic_convention must be 'projected' or 'molcas'.")
    if fock_contraction not in ("operator", "density", "rdm"):
        raise ValueError("fock_contraction must be 'operator', 'density', or 'rdm'.")
    if fock_contraction == "rdm" and norm_tol < 1e-12:
        raise ValueError("RDM contractions require norm_tol >= 1e-12.")
    started = perf_counter()
    nmo = len(h1)
    groups = _grouped_external_spaces(ncore, nmo, ncas, frozen_core)
    templates = {}
    retained_bytes = 0
    budget = None if max_memory_mb is None else max_memory_mb * 1024**2
    engine = None
    if fock_contraction == "rdm":
        from .rdm_contractions import RDMContractions

        active = fock[ncore : ncore + ncas, ncore : ncore + ncas]
        if not np.array_equal(active, np.diag(np.diag(active))):
            raise ValueError(
                "RDM contractions require a pseudocanonical active Fock operator."
            )
        engine = RDMContractions(
            np.where(abs(ci) > screen_tol, ci, 0.0),
            binary,
            np.diag(active),
            max_memory_mb,
        )
        # Reserve the packed density copies used by fused contraction plans.
        retained_bytes = 2 * sum(
            getattr(d, "nbytes", 8) for d in (*engine.rdms, *engine.weighted)
        )
    for pattern in groups:
        if progress:
            progress(
                f"Building active template {pattern}, external groups={len(groups[pattern])}"
            )
        if engine is not None:
            templates[pattern] = _RDMTemplate(
                pattern,
                engine,
                np.diag(active),
                screen_tol,
                lindep_tol,
                None if budget is None else budget - retained_bytes,
                ic_convention=ic_convention,
                norm_tol=norm_tol,
            )
            retained_bytes += templates[pattern].projector.nbytes
        else:
            templates[pattern] = _ActiveTemplate(
                pattern,
                binary,
                ci,
                fock[ncore : ncore + ncas, ncore : ncore + ncas],
                screen_tol,
                lindep_tol,
                None if budget is None else budget - retained_bytes,
                ic_convention=ic_convention,
                norm_tol=norm_tol,
                fock_contraction=fock_contraction,
            )
        t = templates[pattern]
        retained_bytes += (
            t.raw.data.nbytes
            + t.raw.indices.nbytes
            + t.raw.indptr.nbytes
            + t.transpose.data.nbytes
            + t.transpose.indices.nbytes
            + t.transpose.indptr.nbytes
            + t.transform.nbytes
            + t.rhs_transform.nbytes
            + t.fock.nbytes
            + t.vectors.nbytes
            + 64 * len(t.rows)
        )
        if progress:
            progress(
                f"Template {pattern}: rows={len(templates[pattern].rows)}, rank={templates[pattern].rank}"
            )
    groups = {p: g for p, g in groups.items() if templates[p].rank}
    templates = {p: templates[p] for p in groups}
    build_seconds = perf_counter() - started
    size = sum(len(g) * templates[p].rank for p, g in groups.items())
    # GMRES basis, RHS, diagonals, amplitudes, and temporary matrix products.
    planned_bytes = (
        size * 8 * 64 + retained_bytes + 512 * sum(map(len, groups.values()))
    )
    if max_memory_mb is not None and planned_bytes > max_memory_mb * 1024**2:
        raise MemoryError(
            f"Shared FIC workspace requires at least {planned_bytes / 1024**2:.1f} MiB; "
            f"max_memory_mb={max_memory_mb}."
        )
    rhs_start = perf_counter()
    if progress:
        progress(f"Building Hamiltonian RHS, contracted rank={size}")
    single, pair, transition_ids = _hamiltonian_coefficients(
        h1, eri, ncore, ncas, frozen_core
    )
    rhs, diagonal, slices = {}, {}, {}
    offset = 0
    for pattern, keys in groups.items():
        template = templates[pattern]
        rhs[pattern] = (
            template.rhs(keys, ncore, single, pair, transition_ids) @ template.vectors
        )
        shifts = np.array(
            [sum(fock[i, i] for i in h) - sum(fock[r, r] for r in v) for h, v in keys]
        )
        diagonal[pattern] = reference_active_fock + shifts[:, None] - template.values
        slices[pattern] = slice(offset, offset + rhs[pattern].size)
        offset += rhs[pattern].size
    rhs_seconds = perf_counter() - rhs_start
    graph_start = perf_counter()
    if progress:
        progress("Building shared inter-space Fock graph")
    connections = _fock_connections(
        groups,
        templates,
        fock,
        ncore,
        ncas,
        None if budget is None else budget - planned_bytes,
        ic_convention=ic_convention,
        fock_contraction=fock_contraction,
    )
    graph_bytes = sum(
        g.data.nbytes + g.indices.nbytes + g.indptr.nbytes + m.nbytes
        for s, t, g, m in connections
    )
    if (
        max_memory_mb is not None
        and planned_bytes + graph_bytes > max_memory_mb * 1024**2
    ):
        raise MemoryError("Shared FIC Fock graph exceeds max_memory_mb.")
    graph_seconds = perf_counter() - graph_start

    def unpack(vector):
        return {p: vector[slices[p]].reshape(rhs[p].shape) for p in groups}

    def pack(parts):
        return (
            np.concatenate([parts[p].ravel() for p in groups])
            if groups
            else np.empty(0)
        )

    shifted = {}
    for p, d in diagonal.items():
        values = d - real_shift
        near = abs(values) < screen_tol
        if imaginary_shift:
            values = np.where(near, np.copysign(screen_tol, values), values)
            shifted[p] = values + imaginary_shift**2 / values
        else:
            if np.any(near & (abs(rhs[p]) > screen_tol)):
                raise ZeroDivisionError(
                    "Near-zero shared FIC-CASPT2 denominator; use an imaginary shift."
                )
            shifted[p] = values
    from .caspt2 import _cpp_attr

    graph_action = _cpp_attr("apply_fic_graph")
    if graph_action is not None:
        edges = [
            (
                slices[s].start,
                slices[t].start,
                *graph.shape[::-1],
                matrix.shape[1],
                matrix.shape[0],
                np.asarray(graph.indptr, dtype=np.int64),
                np.asarray(graph.indices, dtype=np.int64),
                np.ascontiguousarray(graph.data),
                np.ascontiguousarray(matrix),
            )
            for s, t, graph, matrix in connections
        ]
        scratch = max((max(e[2], e[3]) * min(e[4], e[5]) for e in edges), default=0)
        graph_bytes += sum(
            e[6].nbytes + e[7].nbytes + e[9].nbytes for e in edges
        ) + 16 * (scratch + size)
        if budget is not None and planned_bytes + graph_bytes > budget:
            raise MemoryError("Shared FIC action workspace exceeds max_memory_mb.")
        workspace = np.empty(scratch)
        transpose_workspace = np.empty(scratch)
        action_diagonals = (pack(diagonal), pack(shifted))

    def action(vector, shifted_equation=True):
        if graph_action is not None:
            return graph_action(
                vector,
                action_diagonals[int(shifted_equation)],
                edges,
                workspace,
                transpose_workspace,
            )
        x = unpack(vector)
        d = shifted if shifted_equation else diagonal
        y = {p: d[p] * x[p] for p in groups}
        for s, t, graph, matrix in connections:
            # Project on the smaller rank before multiplying the external graph.
            if matrix.shape[0] <= matrix.shape[1]:
                y[t] -= graph @ (x[s] @ matrix.T)
                y[s] -= (graph.T @ x[t]) @ matrix
            else:
                y[t] -= (graph @ x[s]) @ matrix.T
                y[s] -= graph.T @ (x[t] @ matrix)
        return pack(y)

    b = pack(rhs)
    denominators = pack(shifted)
    # An uncoupled zero-denominator mode has zero amplitude. Keep its exact
    # diagonal in the action, using a finite preconditioner only.
    denominators = np.where(abs(denominators) < screen_tol, 1.0, denominators)
    initial = b / denominators
    solve_start = perf_counter()
    history = []
    if progress:
        progress(f"Solving coupled equation: graph terms={len(connections)}")
    if size and connections:
        lhs = LinearOperator((size, size), matvec=action, dtype=float)
        pre = LinearOperator(
            (size, size), matvec=lambda x: x / denominators, dtype=float
        )
        solution, info = gmres(
            lhs,
            b,
            x0=initial,
            M=pre,
            rtol=solver_tol,
            atol=1e-13,
            restart=min(40, size),
            maxiter=max(1, (max_iterations + 39) // 40),
            callback=lambda r: history.append(float(r)),
            callback_type="pr_norm",
        )
        relative = np.linalg.norm(action(solution) - b) / max(np.linalg.norm(b), 1e-300)
        if info or relative > max(10 * solver_tol, 1e-11):
            raise RuntimeError(
                f"Shared FIC-CASPT2 did not converge: GMRES info={info}, residual={relative:g}"
            )
    else:
        solution = initial
    applied = action(solution, False)
    energy = 2 * b @ solution - solution @ applied
    unshifted_residual = applied - real_shift * solution - b
    equation_residual = action(solution) - b
    x, ax = unpack(solution), unpack(applied)
    transition_couplings = None
    if transition_ci is not None:
        if engine is None:
            raise NotImplementedError(
                "Shared multistate couplings require RDM contractions."
            )
        import copy

        from .rdm_contractions import RDMContractions

        transition_couplings = []
        for bra in transition_ci:
            density = RDMContractions.transition(ci, bra, binary)
            value = 0.0
            for pattern, keys in groups.items():
                template = templates[pattern]
                gram = density.matrix(
                    template.operators, template.operators, template.ncore
                )
                proxy = copy.copy(template)
                # All Hamiltonian generators must survive even when a ket's
                # corresponding generator vanishes (transition density).
                proxy.nonzero = np.ones(len(template.operators), dtype=bool)
                proxy.norms = np.ones(len(template.operators))
                proxy.rhs_transform = template.projector.T @ gram
                rhs_bra = (
                    proxy.rhs(keys, ncore, single, pair, transition_ids)
                    @ template.vectors
                )
                value += float(np.sum(rhs_bra * x[pattern]))
            transition_couplings.append(value)
    components = {}
    nonvariational_components = {}
    classes = {
        (2, 2): 0,
        (2, 1): 1,
        (1, 2): 2,
        (2, 0): 3,
        (0, 2): 4,
        (1, 1): 5,
        (1, 0): 6,
        (0, 1): 7,
    }
    for i, name in enumerate(CASPT2_PERTURBER_CLASSES):
        pats = [p for p in groups if classes[len(p[0]), len(p[1])] == i]
        nonvariational_components[name] = sum(
            float(np.sum(rhs[p] * x[p])) for p in pats
        )
        components[name] = CASPT2Component(
            name,
            sum(
                len(groups[p])
                * (templates[p].rank if engine is not None else len(templates[p].rows))
                for p in pats
            ),
            sum(float(np.sum(2 * rhs[p] * x[p] - x[p] * ax[p])) for p in pats),
            sum(float(np.sum(x[p] ** 2)) for p in pats),
        )
    return dict(
        energy=float(energy),
        nonvariational=float(b @ solution),
        transition_couplings=transition_couplings,
        nonvariational_components=nonvariational_components,
        norm=float(solution @ solution),
        components=components,
        rank=size,
        raw_count=sum(
            len(groups[p]) * np.count_nonzero(templates[p].nonzero) for p in groups
        ),
        template_count=len(templates),
        template_rows=sum(len(t.rows) for t in templates.values()),
        logical_rows=sum(len(groups[p]) * len(templates[p].rows) for p in groups),
        planned_bytes=max(
            planned_bytes + graph_bytes,
            engine.workspace_bytes if engine is not None else 0,
        ),
        groups=sum(map(len, groups.values())),
        graph_terms=len(connections),
        iterations=len(history),
        history=history,
        residual=float(np.linalg.norm(unshifted_residual)),
        relative_residual=float(
            np.linalg.norm(unshifted_residual) / max(np.linalg.norm(b), 1e-300)
        ),
        equation_residual=float(
            np.linalg.norm(equation_residual) / max(np.linalg.norm(b), 1e-300)
        ),
        timings=dict(
            template_build_s=build_seconds,
            rhs_s=rhs_seconds,
            fock_graph_s=graph_seconds,
            solve_s=perf_counter() - solve_start,
            total_s=perf_counter() - started,
        ),
    )
