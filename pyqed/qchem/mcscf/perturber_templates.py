"""Shared active-space templates for fully internally contracted Fock PT2.

External orbital labels change integral coefficients and orbital-energy shifts,
but not the active excitation vectors.  Store each active template once rather
than expanding it over every inactive-hole/virtual-particle combination.
This is an algebraic factorization of the spin-free FIC space, not an additional
contraction or an external-orbital cutoff. RAS references retain their ordinary
external first-order space. See Malmqvist et al., JCP 128, 204109 (2008),
doi:10.1063/1.2920188. The multistate driver uses ordered transition contractions from this module;
IPEA shifts and correlated transition properties are not implemented.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement, product
from time import perf_counter
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from scipy import sparse

_DOUBLE_TILE_PAIRS = 1 << 16
# Includes caller RHS, four recurrence vectors, solution, and staging headroom.
_MINRES_WORK_VECTORS = 10
_FOCK_EDGE_BATCH = 1 << 18
_FOCK_CACHE_BYTES = 16 << 20
_DIAGONAL_BLOCK_SIZE = 1 << 16


class ReferenceIntegrals(NamedTuple):
    """Exact excitation ERIs and the closed-inactive-shell potential."""

    pair: np.ndarray  # (p i|q j), i,j inactive or active
    core_potential: np.ndarray


class ReferencePairFactors(NamedTuple):
    factors: np.ndarray  # (P|i p), i correlated inactive or active, p any MO
    core_potential: np.ndarray
    first_orbital: int = 0


class _ActivePairIntegrals(NamedTuple):
    rows: np.ndarray
    values: np.ndarray

    @property
    def nbytes(self):
        return self.rows.nbytes + self.values.nbytes


class _ExcitationCoefficients:
    """Hamiltonian coefficients evaluated only for requested MO pairs."""

    def __init__(self, h1, eri, ncore, ncas, frozen_core):
        self.eri = eri
        self.frozen_core = frozen_core
        self.ncore = ncore
        self.nocc = ncore + ncas
        self.nmo = len(h1)
        self.active_pair_integrals = None
        p = np.arange(ncore, self.nmo)[:, None]
        q = np.arange(frozen_core, self.nocc)
        self.single = h1[p, q]
        if isinstance(eri, (ReferenceIntegrals, ReferencePairFactors)):
            self.single += eri.core_potential[p, q]
        else:
            for i in range(ncore):
                self.single += 2 * _integrals(eri, p, q, i, i) - self.pair(p, i, q, i)
        for i in range(ncore, self.nocc):
            # The commutator compares (i,q) and (p,i) in q-major order.
            sign = np.where(q == i, np.sign(p - i), np.sign(i - q))
            self.single += 0.5 * (sign - 1) * self.block(p[:, 0], q, i, i)

    def pair(self, p, q, r, s):
        eri = self.eri
        if isinstance(eri, ReferenceIntegrals):
            return eri.pair[p, q, r, s]
        if isinstance(eri, ReferencePairFactors):
            return np.einsum(
                "P...,P...->...",
                eri.factors[:, q - eri.first_orbital, p],
                eri.factors[:, s - eri.first_orbital, r],
            )
        return _integrals(eri, p, q, r, s)

    def block(self, left, right, i, j):
        """A rectangle of (a i|b j)."""
        eri = self.eri
        if isinstance(eri, ReferencePairFactors):
            return (
                eri.factors[:, i - eri.first_orbital, left].T
                @ eri.factors[:, j - eri.first_orbital, right]
            )
        return self.pair(left[:, None], i, right, j)

    def prepare_active_pair_integrals(self, memory_bytes, cache):
        """Build the rectangular Gram block needed by the retained classes."""
        if not isinstance(self.eri, ReferencePairFactors) or self.ncore == self.nocc:
            return 0, False
        cached = None if cache is None else cache.get("active_pair_integrals")
        if cached is not None:
            if memory_bytes is None or cached.nbytes <= memory_bytes:
                self.active_pair_integrals = cached
                return cached.nbytes, True
            del cache["active_pair_integrals"]
            self.active_pair_integrals = None
            return 0, False

        npart = self.nmo - self.ncore
        nactive = self.nocc - self.ncore
        nholes = self.ncore - self.frozen_core
        ntrans = (nholes + nactive) * npart
        nrows = nholes * nactive + nactive * npart
        eri = self.eri
        naux = len(eri.factors)
        storage_bytes = 8 * ntrans * (nrows + 1)
        fixed_work = 16 * nrows * naux + 24 * (ntrans + nrows)
        panel_column_bytes = 8 * (2 * naux + nrows)
        panel_bytes = 16 << 20
        if memory_bytes is not None:
            panel_bytes = min(panel_bytes, memory_bytes - storage_bytes - fixed_work)
        width = min(ntrans, int(panel_bytes // panel_column_bytes))
        if width < 1:
            return 0, False

        q, p = np.divmod(np.arange(ntrans), npart)
        q += self.frozen_core - eri.first_orbital
        p += self.ncore
        active = (p < self.nocc) | (q >= self.ncore - eri.first_orbital)
        rows = np.full(ntrans, -1, dtype=np.intp)
        rows[active] = np.arange(nrows)
        active_factors = np.ascontiguousarray(eri.factors[:, q[active], p[active]].T)
        values = np.empty((nrows, ntrans))
        for start in range(0, ntrans, width):
            panel = slice(start, start + width)
            np.matmul(
                active_factors, eri.factors[:, q[panel], p[panel]], out=values[:, panel]
            )
        self.active_pair_integrals = _ActivePairIntegrals(rows, values)
        if cache is not None:
            cache["active_pair_integrals"] = self.active_pair_integrals
        return storage_bytes, False

    def retained_pair(self, p, q, r, s):
        integrals = self.active_pair_integrals
        if integrals is None:
            return self.pair(p, q, r, s)
        npart = self.nmo - self.ncore
        left = (q - self.frozen_core) * npart + p - self.ncore
        right = (s - self.frozen_core) * npart + r - self.ncore
        left_row = integrals.rows[left]
        use_left = left_row >= 0
        # In a retained class at least one of the two pairs has an active index.
        return integrals.values[
            np.where(use_left, left_row, integrals.rows[right]),
            np.where(use_left, right, left),
        ]


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
            holes, particles = groups.orbital_batch(start, start + 1024)
            maps = np.empty((len(holes), self.nmo), dtype=int)
            if self.ncore:
                maps[:, : self.ncore] = holes[:, : self.ncore]
            maps[:, self.ncore : self.ncore + self.ncas] = ncore + np.arange(self.ncas)
            nv = self.nmo - self.ncore - self.ncas
            if nv:
                maps[:, self.ncore + self.ncas :] = particles[:, :nv]
            coeff = np.empty((len(holes), len(self.operators)))
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
            result[start : start + len(holes)] = coeff @ self.rhs_transform.T
        return result

    def rhs_batches(self, groups, coefficients):
        active = coefficients.ncore + np.arange(self.ncas)
        nv = self.nmo - self.ncore - self.ncas
        factors = (
            getattr(coefficients.eri, "factors", None)
            if coefficients.active_pair_integrals is None
            else None
        )
        block = (
            1024
            if factors is None
            else min(1024, max(1, (8 << 20) // (16 * len(factors))))
        )
        for start in range(0, len(groups), block):
            holes, particles = groups.orbital_batch(start, start + block)
            maps = np.empty((self.nmo, len(holes)), dtype=int)
            if self.ncore:
                maps[: self.ncore] = holes[:, : self.ncore].T
            maps[self.ncore : self.ncore + self.ncas] = active[:, None]
            if nv:
                maps[-nv:] = particles[:, :nv].T
            raw = np.empty((len(holes), len(self.operators)))
            for j, op in enumerate(self.operators):
                p, q = maps[list(op[0])]
                if len(op) == 1:
                    raw[:, j] = coefficients.single[
                        p - coefficients.ncore, q - coefficients.frozen_core
                    ]
                else:
                    r, s = maps[list(op[1])]
                    raw[:, j] = coefficients.retained_pair(p, q, r, s)
                    if op[0] == op[1]:
                        raw[:, j] *= 0.5
            yield slice(start, start + len(holes)), raw

    def tiled_rhs(self, groups, coefficients, out=None):
        result = np.empty((len(groups), self.rank)) if out is None else out
        projection = self.rhs_projection()
        for rows, raw in self.rhs_batches(groups, coefficients):
            result[rows] = raw[:, self.nonzero] @ projection
        return result

    def rhs_projection(self):
        return (self.rhs_transform.T @ self.vectors) * self.norms[self.nonzero, None]

    def project_rhs(self, raw):
        return raw[:, self.nonzero] @ self.rhs_projection()


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


def _external_axis_counts(n):
    return {(): 1, (0,): n, (0, 0): n, (0, 1): n * (n - 1) // 2}


def _external_pattern_counts(nholes, nvirtuals):
    return {
        (h, v): nh * nv
        for h, nh in _external_axis_counts(nholes).items()
        for v, nv in _external_axis_counts(nvirtuals).items()
        if nh and nv and (h or v)
    }


@dataclass(frozen=True, eq=False)
class _ExternalGroups:
    """Cartesian product of hole and virtual labels with implicit row order."""

    holes: np.ndarray
    virtuals: np.ndarray

    def __len__(self):
        return len(self.holes) * len(self.virtuals)

    def __iter__(self):
        return product(
            map(tuple, self.holes.tolist()), map(tuple, self.virtuals.tolist())
        )

    def __getitem__(self, index):
        h, v = divmod(range(len(self))[index], len(self.virtuals))
        return tuple(self.holes[h].tolist()), tuple(self.virtuals[v].tolist())

    def orbital_batch(self, start, stop):
        h, v = np.divmod(np.arange(start, min(stop, len(self))), len(self.virtuals))
        return self.holes[h], self.virtuals[v]

    def rows(self, holes, virtuals, pattern, ncore, nvirt):
        def axis_rows(values, axis, labels, norbitals):
            if not axis:
                return 0
            first = values[:, 0].astype(np.int64) - int(labels[0, 0])
            if axis == (0, 1):
                second = values[:, 1] - int(labels[0, 0])
                return first * (2 * norbitals - first - 1) // 2 + second - first - 1
            return first

        return axis_rows(holes, pattern[0], self.holes, ncore) * len(
            self.virtuals
        ) + axis_rows(virtuals, pattern[1], self.virtuals, nvirt)


@lru_cache(maxsize=1)
def _grouped_external_spaces(ncore, nmo, ncas, frozen_core, patterns=None):
    """Immutable external labels, shared across roots of one orbital layout."""
    global _fock_topology_cache
    _fock_topology_cache = None

    def labels(space, pattern):
        if pattern == (0, 1):
            left, right = np.triu_indices(len(space), 1)
            rows = (np.column_stack((left, right)) + space.start).astype(np.int32)
        elif not pattern:
            rows = np.empty((1, 0), dtype=np.int32)
        else:
            rows = np.repeat(
                np.arange(space.start, space.stop, dtype=np.int32)[:, None],
                len(pattern),
                axis=1,
            )
        rows.flags.writeable = False
        return rows

    nh, nv = ncore - frozen_core, nmo - ncore - ncas
    if patterns is None:
        patterns = _external_pattern_counts(nh, nv)
    holes = {h: labels(range(frozen_core, ncore), h) for h in {p[0] for p in patterns}}
    virtuals = {
        v: labels(range(ncore + ncas, nmo), v) for v in {p[1] for p in patterns}
    }
    return MappingProxyType(
        {(h, v): _ExternalGroups(holes[h], virtuals[v]) for h, v in patterns}
    )


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


def _fock_topology_indices(groups, fock, ncore, ncas, memory_bytes=None, cache_bytes=0):
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
        indices, entries, _ = _build_fock_topology(
            groups, fock, ncore, ncas, memory_bytes, cache_bytes
        )
        for rr, cc, external, transition in indices.values():
            rr.flags.writeable = cc.flags.writeable = external.flags.writeable = False
        stored = 8 * len(groups) + sum(
            rr.nbytes + cc.nbytes + external.nbytes
            for rr, cc, external, _ in indices.values()
        )
        cached = (key, indices, entries, stored)
        _fock_topology_cache = cached
    _, indices, entries, stored = cached
    if memory_bytes is not None and stored + cache_bytes > memory_bytes:
        raise MemoryError(
            "Shared FIC Fock graph exceeds max_memory_mb during assembly."
        )
    return indices, entries


def _fock_topology(groups, fock, ncore, ncas, memory_bytes=None, cache_bytes=0):
    indices, entries = _fock_topology_indices(
        groups, fock, ncore, ncas, memory_bytes, cache_bytes
    )
    result = {}
    for k, (rr, cc, external, transition) in indices.items():
        values = fock[external[0], external[1]] if k[2] == 2 else external
        result[k] = (rr, cc, values, transition)
    return result, entries


def _iter_fock_edges(groups, fock, ncore, ncas, batch_size, guard=lambda _: None):
    """Yield bounded external-index batches without retaining an edge graph."""
    if not groups:
        return
    classes = {tuple(map(len, pattern)) for pattern in groups}
    correlated = np.unique(np.concatenate([g.holes.ravel() for g in groups.values()]))
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
    transitions = {}
    limit = np.iinfo(np.int32).max
    for pattern, source in groups.items():
        nh, nv = map(len, pattern)
        for kind, (ps, qs) in enumerate(insertions):
            destination = (nh + (kind != 1), nv + (kind != 0))
            # The eliminated external-double class has no rows in this graph.
            if destination not in classes or (kind < 2 and not ncas) or not len(ps):
                continue
            # Bound insertions as well as source rows, including a dense F_ai block.
            insertion_width = min(len(ps), batch_size)
            source_width = max(1, batch_size // insertion_width)
            for start in range(0, len(source), source_width):
                stop = min(len(source), start + source_width)
                source_holes, source_virtuals = source.orbital_batch(start, stop)
                for begin in range(0, len(ps), insertion_width):
                    end = min(len(ps), begin + insertion_width)
                    width = end - begin
                    count = (stop - start) * width
                    guard(count * 128)
                    cc = np.repeat(np.arange(start, stop), width)
                    p, q = np.tile(ps[begin:end], stop - start), np.tile(
                        qs[begin:end], stop - start
                    )
                    holes = np.repeat(source_holes, width, axis=0)
                    virtual = np.repeat(source_virtuals, width, axis=0)
                    ho = np.zeros(count, dtype=np.int8)
                    vo = np.zeros_like(ho)
                    if kind != 1:
                        if nh:
                            ho = np.sign(q - holes[:, 0]).astype(np.int8)
                            holes = np.column_stack(
                                (np.minimum(q, holes[:, 0]), np.maximum(q, holes[:, 0]))
                            )
                        else:
                            holes = q[:, None]
                    if kind != 0:
                        if nv:
                            vo = np.sign(p - virtual[:, 0]).astype(np.int8)
                            virtual = np.column_stack(
                                (
                                    np.minimum(p, virtual[:, 0]),
                                    np.maximum(p, virtual[:, 0]),
                                )
                            )
                        else:
                            virtual = p[:, None]
                    cases = (ho + 1) * 3 + vo + 1
                    for case in np.unique(cases):
                        hsign, vsign = divmod(int(case), 3)
                        hsign, vsign = hsign - 1, vsign - 1
                        hpattern, vpattern = pattern
                        if kind != 1:
                            hpattern = (0, int(hsign != 0)) if nh else (0,)
                        if kind != 0:
                            vpattern = (0, int(vsign != 0)) if nv else (0,)
                        dest = hpattern, vpattern
                        if dest not in groups:
                            continue
                        selected = np.flatnonzero(cases == case)
                        target = groups[dest]
                        rr = target.rows(
                            holes[selected],
                            virtual[selected],
                            dest,
                            len(correlated),
                            len(virtuals),
                        )
                        rows = (
                            rr.astype(np.int32, copy=False)
                            if len(target) <= limit
                            else rr
                        )
                        cols = cc[selected].astype(
                            np.int32 if len(source) <= limit else np.int64
                        )
                        key = pattern, dest, kind, hsign, vsign
                        if key not in transitions:
                            first = selected[0]
                            transitions[key] = _transition_key(
                                source[cc[first]],
                                target[rr[0]],
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
                        ).astype(np.int32)
                        rows.flags.writeable = cols.flags.writeable = (
                            external.flags.writeable
                        ) = False
                        yield key, rows, cols, external, transitions[key]
                    guard(count * 128)


def _build_fock_topology(groups, fock, ncore, ncas, memory_bytes=None, cache_bytes=0):
    """Generate graph edges in bounded batches and rank destination pairs directly."""
    if not groups:
        return {}, 0, 0
    peak = 0
    graph_bytes = 0

    def guard(workspace):
        nonlocal peak
        peak = max(peak, graph_bytes + workspace)
        if (
            memory_bytes is not None
            and graph_bytes + workspace + cache_bytes > memory_bytes
        ):
            raise MemoryError(
                "Shared FIC Fock graph exceeds max_memory_mb during assembly."
            )

    guard(8 * len(groups))
    parts, transitions = {}, {}
    entries = 0
    for key, rows, cols, external, base in _iter_fock_edges(
        groups, fock, ncore, ncas, _FOCK_EDGE_BATCH, guard
    ):
        parts.setdefault(key, []).append((rows, cols, external))
        transitions[key] = base
        graph_bytes += rows.nbytes + cols.nbytes + external.nbytes
        entries += len(rows)
    result = {}
    for key, chunks in parts.items():
        if len(chunks) == 1:
            rows, cols, external = chunks[0]
        else:
            guard(sum(array.nbytes for chunk in chunks for array in chunk))
            rows, cols, external = (
                np.concatenate(axis, axis=-1) for axis in zip(*chunks)
            )
        result[key] = (rows, cols, external, transitions[key])
        chunks.clear()
    return result, entries, peak


class _FockGraph:
    """Active transition matrices with bounded caching of external indices."""

    def __init__(self, groups, fock, ncore, ncas, batch_size, matrices, indices=None):
        self.groups, self.fock = groups, fock
        self.ncore, self.ncas, self.batch_size = ncore, ncas, batch_size
        self.matrices, self.indices = matrices, indices

    def __len__(self):
        return sum(map(len, self.matrices.values()))

    def __iter__(self):
        batches = (
            _iter_fock_edges(
                self.groups, self.fock, self.ncore, self.ncas, self.batch_size
            )
            if self.indices is None
            else ((key, *arrays, None) for key, arrays in self.indices.items())
        )
        for key, rows, cols, labels, _ in batches:
            s, t = key[:2]
            for weights, matrix in self.matrices.get(key, ()):
                yield s, t, rows, cols, labels.reshape(-1, len(rows)), weights, matrix


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
    compact=False,
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

    if compact:
        global _fock_topology_cache
        _fock_topology_cache = None
        batch_size = min(_FOCK_EDGE_BATCH, 1 << 14)
        topology_bytes = 0
        graph_bytes = 128 * batch_size
        if memory_bytes is not None and graph_bytes + cache_bytes > memory_bytes:
            raise MemoryError(
                "Shared FIC Fock graph exceeds max_memory_mb during assembly."
            )
        cache_limit = _FOCK_CACHE_BYTES
        if memory_bytes is not None:
            cache_limit = min(
                cache_limit, max(0, (memory_bytes - cache_bytes - graph_bytes) // 4)
            )
        parts, stored_bytes = ({} if cache_limit else None), 0
        connections = {}
        for key, rr, cc, labels, base in _iter_fock_edges(
            groups, fock, ncore, ncas, batch_size
        ):
            if key not in connections:
                connections[key] = (base, np.zeros(len(fock), dtype=bool))
            if key[2] < 2:
                connections[key][1][labels] = True
            if parts is not None:
                stored_bytes += rr.nbytes + cc.nbytes + labels.nbytes
                if stored_bytes > cache_limit:
                    parts = None
                else:
                    parts.setdefault(key, []).append((rr, cc, labels))
        indices = None
        if parts is not None:
            indices = {}
            for key, chunks in parts.items():
                arrays = (
                    chunks[0]
                    if len(chunks) == 1
                    else tuple(np.concatenate(axis, axis=-1) for axis in zip(*chunks))
                )
                for array in arrays:
                    array.flags.writeable = False
                indices[key] = arrays
            topology_bytes = stored_bytes
            graph_bytes += topology_bytes
            parts.clear()
            if indices:
                del chunks, arrays, array
        # Do not keep the final yielded batch alive through matrix construction.
        if connections:
            del rr, cc, labels
        descriptors = (
            (key, base, present) for key, (base, present) in connections.items()
        )
        matrices = {}
    else:
        connections, _ = _fock_topology_indices(
            groups, fock, ncore, ncas, memory_bytes, cache_bytes
        )
        topology_bytes = sum(
            rr.nbytes + cc.nbytes + labels.nbytes
            for rr, cc, labels, _ in connections.values()
        )
        graph_bytes = topology_bytes
        descriptors = ((key, data[3], None) for key, data in connections.items())
    result = []
    for topology_key, base, present in descriptors:
        s, t, kind, _, _ = topology_key
        if not compact:
            rr, cc, labels, _ = connections[topology_key]
        # The external topology is independent of the active index. Expand
        # only the active transition matrices on the compact path.
        for a in range(ncas if kind < 2 else 1):
            key = (
                *base[:2],
                base[2] + (a if kind == 0 else 0),
                base[3] + (a if kind == 1 else 0),
            )
            if kind == 0:
                weights = fock[ncore + a, :, None]
            elif kind == 1:
                weights = fock[:, ncore + a, None]
            else:
                weights = fock
            if compact:
                if kind < 2 and not np.any(abs(weights[present, 0]) >= 1e-14):
                    continue
                edge_bytes = weights.nbytes if kind < 2 else 0
            else:
                values = (
                    weights[labels, 0] if kind < 2 else weights[labels[0], labels[1]]
                )
                keep = abs(values) >= 1e-14
                if not np.any(keep):
                    continue
                edge_bytes = 16 * np.count_nonzero(keep) + 8 * (len(groups[t]) + 1)
            estimate = edge_bytes + 8 * templates[s].rank * templates[t].rank
            if (
                memory_bytes is not None
                and cache_bytes + graph_bytes + estimate > memory_bytes
            ):
                if compact and indices is not None:
                    indices = None
                    graph_bytes -= topology_bytes
                    topology_bytes = 0
                if cache_bytes + graph_bytes + estimate > memory_bytes:
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
            if compact:
                matrices.setdefault(topology_key, []).append((weights, matrix))
                graph_bytes += edge_bytes + matrix.nbytes
            else:
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
    if compact:
        result = _FockGraph(groups, fock, ncore, ncas, batch_size, matrices, indices)
    return result, graph_bytes, topology_bytes


class _ExternalDoubles:
    """Schur elimination of the two-inactive-hole/two-virtual class.

    Only the repeated/distinct-index templates and their Fock transition
    matrices persist. Each tile fixes the inactive pair and spans ordered
    virtual pairs. Parent rows have the same Cartesian-product order as
    _grouped_external_spaces.
    """

    def __init__(
        self,
        templates,
        parents,
        groups,
        fock,
        ncore,
        ncas,
        frozen_core,
        reference_active_fock,
        real_shift,
        imaginary_shift,
        screen_tol,
    ):
        self.templates = templates
        self.fock = fock
        self.eps = np.diag(fock)
        self.ncore, self.nocc = ncore, ncore + ncas
        self.frozen_core = frozen_core
        self.nholes, self.nvirt = ncore - frozen_core, len(fock) - self.nocc
        self.reference_active_fock = reference_active_fock
        self.real_shift, self.imaginary_shift, self.screen_tol = (
            real_shift,
            imaginary_shift,
            screen_tol,
        )
        self.edges = {}
        active = np.arange(self.ncore, self.nocc)
        correlated = np.arange(self.frozen_core, self.ncore)
        external = np.arange(self.nocc, len(fock))
        hole_fock = fock[np.ix_(active, correlated)].T
        virtual_fock = fock[np.ix_(external, active)]
        hole_fock[abs(hole_fock) < 1e-14] = 0.0
        virtual_fock[abs(virtual_fock) < 1e-14] = 0.0
        self.counts = {
            pattern: self._count(self.nholes, pattern[0])
            * self._count(self.nvirt, pattern[1])
            for pattern in templates
        }
        for pattern, target in templates.items():
            holes, virtuals = pattern[0], tuple(self.nocc + v for v in pattern[1])
            edges = []
            for drop_h in (*range(len(set(holes))), None):
                for drop_v in (*range(len(set(virtuals))), None):
                    if drop_h is None and drop_v is None:
                        continue
                    h = holes if drop_h is None else (holes[1 - drop_h],)
                    v = virtuals if drop_v is None else (virtuals[1 - drop_v],)
                    source_pattern = _pattern(h, v)
                    if source_pattern not in groups:
                        continue
                    source = parents[source_pattern]
                    ps = active if drop_v is None else (virtuals[drop_v],)
                    qs = active if drop_h is None else (holes[drop_h],)
                    matrices = []
                    for p in ps:
                        for q in qs:
                            key = _transition_key(
                                (h, v), (holes, virtuals), p, q, ncore, ncas
                            )
                            matrix = _transition_matrix(source, target, key)
                            if np.max(abs(matrix), initial=0) >= 1e-13:
                                matrix = -(target.vectors.T @ matrix @ source.vectors)
                            else:
                                matrix = np.zeros_like(matrix)
                            matrices.append(matrix)
                    if not matrices or not np.any(matrices):
                        continue
                    if drop_v is None:
                        block = np.einsum("pa,aij->pij", hole_fock, matrices)
                    elif drop_h is None:
                        block = np.einsum("pa,aij->pij", virtual_fock, matrices)
                    else:
                        block = (
                            matrices[0]
                            if np.any(
                                abs(fock[self.nocc :, frozen_core:ncore]) >= 1e-14
                            )
                            else None
                        )
                    if block is not None and np.any(block):
                        edges.append((source_pattern, drop_h, drop_v, block))
            self.edges[pattern] = edges

    @staticmethod
    def _count(n, pattern):
        return n * (n - 1) // 2 if pattern == (0, 1) else n

    @staticmethod
    def _index(n, first, second, pattern):
        if pattern == (0, 1):
            return first * (2 * n - first - 1) // 2 + second - first - 1
        return first

    def tiles(self, coefficients=None):
        width = max(1, _DOUBLE_TILE_PAIRS // max(1, self.nvirt))
        all_virtuals = np.arange(self.nocc, len(self.eps))
        rectangles = []
        for start in range(0, self.nvirt, width):
            left = all_virtuals[start : start + width]
            positions = tuple(
                np.nonzero(mask)
                for mask in (
                    left[:, None] == all_virtuals,
                    left[:, None] < all_virtuals,
                )
            )
            rectangles.append((left, positions))
        for i, j in combinations_with_replacement(
            range(self.frozen_core, self.ncore), 2
        ):
            hp = (0, int(i != j))
            for left, positions in rectangles:
                if coefficients is not None:
                    direct = coefficients.block(left, all_virtuals, i, j)
                    exchange = (
                        direct
                        if i == j
                        else coefficients.block(left, all_virtuals, j, i)
                    )
                for distinct, (row, col) in enumerate(positions):
                    pattern = hp, (0, distinct)
                    if pattern not in self.templates:
                        continue
                    if not len(row):
                        continue
                    a, b = left[row], all_virtuals[col]
                    template = self.templates[pattern]
                    diag = (
                        self.reference_active_fock
                        + self.eps[i]
                        + self.eps[j]
                        - self.eps[a, None]
                        - self.eps[b, None]
                        - template.values
                    )
                    shifted = diag - self.real_shift
                    near = abs(shifted) < self.screen_tol
                    if self.imaginary_shift:
                        shifted = np.where(
                            near, np.copysign(self.screen_tol, shifted), shifted
                        )
                        shifted += self.imaginary_shift**2 / shifted
                    raw = None
                    if coefficients is not None:
                        raw = np.empty((len(row), len(template.operators)))
                        for k, (first, second) in enumerate(template.operators):
                            p, q = first
                            use_exchange = p - template.ncore - template.ncas != q
                            raw[:, k] = (exchange if use_exchange else direct)[row, col]
                            if first == second:
                                raw[:, k] *= 0.5
                    yield pattern, (i, j), (
                        (left, all_virtuals),
                        (row, col),
                        (a, b),
                    ), diag, shifted, raw

    def coupling(self, pattern, holes, tile, rectangular):
        axes, _, pairs = tile
        for parent, drop_h, drop_v, block in self.edges[pattern]:
            h, v = holes, axes if rectangular and drop_v is not None else pairs
            weight = 1.0
            if drop_h is not None:
                q, h = h[drop_h], (h[1 - drop_h],) * 2
            if drop_v is not None:
                p, v = v[drop_v], (v[1 - drop_v],) * 2
            if drop_v is None:
                matrix = block[q - self.frozen_core]
            elif drop_h is None:
                matrix = block[p - self.nocc]
            else:
                matrix = block
                weight = self.fock[p, q, None]
                weight = np.where(abs(weight) >= 1e-14, weight, 0.0)
            hi = self._index(
                self.nholes, h[0] - self.frozen_core, h[1] - self.frozen_core, parent[0]
            )
            vi = self._index(self.nvirt, v[0] - self.nocc, v[1] - self.nocc, parent[1])
            rows = hi * self._count(self.nvirt, parent[1]) + vi
            yield parent, rows, weight, matrix, drop_v

    def inverse(self, shifted, rhs):
        near = abs(shifted) < self.screen_tol
        if np.any(near & (abs(rhs) > self.screen_tol)):
            raise ZeroDivisionError(
                "Near-zero external-doubles denominator; use an imaginary shift."
            )
        return np.divide(rhs, shifted, out=np.zeros_like(rhs), where=~near)

    def feedback_arrays(self, slices, parents):
        """Pack the shared transition blocks for the iterative Schur action."""
        offsets = [0]
        blocks, edges = [], []
        ranks = np.zeros(4, dtype=np.int64)
        energies = np.zeros(
            (4, max(template.rank for template in self.templates.values()))
        )
        start = 0
        for p in product(((0, 0), (0, 1)), repeat=2):
            k = len(offsets) - 1
            if p in self.templates:
                template = self.templates[p]
                ranks[k] = template.rank
                energies[k, : template.rank] = (
                    self.reference_active_fock - template.values - self.real_shift
                )
            for parent, drop_h, drop_v, block in self.edges.get(p, ()):
                edges.append(
                    (
                        -1 if drop_h is None else drop_h,
                        -1 if drop_v is None else drop_v,
                        slices[parent].start,
                        parents[parent].rank,
                        self._count(self.nvirt, parent[1]),
                        start,
                    )
                )
                blocks.append(block.ravel())
                start += block.size
            offsets.append(len(edges))
        cross = self.fock[self.nocc :, self.frozen_core : self.ncore].copy()
        cross[abs(cross) < 1e-14] = 0.0
        return (
            self.eps[self.frozen_core : self.ncore],
            self.eps[self.nocc :],
            cross,
            np.asarray(offsets, dtype=np.int64),
            ranks,
            energies,
            np.asarray(edges, dtype=np.int64),
            np.concatenate(blocks),
            self.imaginary_shift**2,
            self.screen_tol,
        )

    def apply(self, pattern, holes, tile, x, result, rhs, shifted):
        """Apply the local inverse and scatter its feedback to retained classes."""
        near = abs(shifted) < self.screen_tol
        singular = np.any(near)
        rectangular = pattern[1] == (0, 1) and not singular
        edges = list(self.coupling(pattern, holes, tile, rectangular))
        if singular:
            for _, _, weight, matrix, _ in edges:
                if np.any(near & (np.any(matrix != 0.0, axis=-1) & (weight != 0.0))):
                    raise ZeroDivisionError(
                        "Singular coupled external-doubles block; use an imaginary shift."
                    )
        axes, indices, _ = tile
        applied = np.zeros_like(shifted)
        if x is not None:
            for parent, rows, weight, matrix, drop_v in edges:
                values = x[parent][rows]
                if rectangular and drop_v is not None:
                    # Parent amplitudes depend on one virtual index. Contract
                    # their active rank before selecting the ordered pairs.
                    keep, drop = indices[1 - drop_v], indices[drop_v]
                    if matrix.ndim == 3:
                        for k in range(applied.shape[1]):
                            applied[:, k] += (values @ matrix[:, k, :].T)[keep, drop]
                    else:
                        applied += (values @ matrix.T)[keep] * weight[drop]
                else:
                    applied += np.einsum("...ij,...j->...i", matrix, values) * weight
        amplitudes = self.inverse(shifted, rhs - applied)
        if rectangular and any(edge[-1] is not None for edge in edges):
            # The reverse contraction sums over the dropped virtual axis.
            grid = np.zeros((amplitudes.shape[1], len(axes[0]), len(axes[1])))
            grid[:, indices[0], indices[1]] = amplitudes.T
        for parent, rows, weight, matrix, drop_v in edges:
            if rectangular and drop_v is not None:
                oriented = grid if drop_v == 1 else grid.transpose(0, 2, 1)
                if matrix.ndim == 3:
                    contribution = sum(
                        oriented[k] @ matrix[:, k, :] for k in range(oriented.shape[0])
                    )
                else:
                    contribution = (oriented @ weight[:, 0]).T @ matrix
                result[parent][rows] += contribution
            else:
                contribution = (
                    np.einsum("...i,...ij->...j", amplitudes, matrix) * weight
                )
                if drop_v is None or pattern[1] == (0, 0):
                    result[parent][rows] += contribution
                else:
                    np.add.at(result[parent], rows, contribution)
        return amplitudes, applied


def _solve_shared_minres(
    operator, rhs, diagonal, initial, tolerance, max_iterations, *, accumulate=None
):
    """Real symmetric MINRES with reusable vectors and true-residual acceptance.

    Adaptation of the Paige--Saunders short recurrence, SIAM J. Numer.
    Anal. 12, 617--629 (1975), doi:10.1137/0712047. Uses positive diagonal
    preconditioning, bounded array temporaries, and residual restarts.
    The diagonal is an array or a callable yielding (slice, positive values)
    blocks. Consumes the initial guess and array diagonal. No least-squares
    acceptance is used for inconsistent singular systems. If supplied,
    ``accumulate(vector, out)`` adds the operator action into ``out`` so
    the oldest Lanczos residual can hold the next one. The current residual
    is reconstructed as beta * diagonal * v after updating the solution,
    eliminating a separately stored preconditioned-residual vector.
    """

    def axpy(destination, scale, source):
        for start in range(0, len(source), 1 << 16):
            part = slice(start, start + (1 << 16))
            destination[part] += scale * source[part]

    if callable(diagonal):
        preconditioner_blocks = diagonal
    else:
        np.abs(diagonal, out=diagonal)

        def preconditioner_blocks():
            for start in range(0, len(diagonal), 1 << 16):
                part = slice(start, start + (1 << 16))
                yield part, diagonal[part]

    def precondition(source, destination):
        for part, values in preconditioner_blocks():
            np.divide(source[part], values, out=destination[part])

    solution = np.zeros_like(rhs) if initial is None else initial
    rhs_norm = max(np.linalg.norm(rhs), 1e-300)
    target = max(tolerance * rhs_norm, 1e-13)
    residual = operator @ solution
    np.subtract(rhs, residual, out=residual)
    history = []
    iterations = 0
    relative = np.linalg.norm(residual) / rhs_norm
    if np.linalg.norm(residual) <= target:
        return solution, iterations, [float(relative)]
    previous = np.empty_like(rhs)
    w = np.zeros_like(rhs)
    old_w = np.zeros_like(rhs)
    eps = np.finfo(float).eps
    while iterations < max_iterations:
        beta_squared = 0.0
        for part, values in preconditioner_blocks():
            beta_squared += np.dot(residual[part], residual[part] / values)
        beta = np.sqrt(beta_squared)
        v = residual
        del residual
        precondition(v, v)
        if not np.isfinite(beta) or beta == 0:
            break
        phibar = beta
        cs, sn, dbar, epsln, norm_squared = -1.0, 0.0, 0.0, 0.0, 0.0
        w.fill(0.0)
        old_w.fill(0.0)
        first = True
        old_beta = 0.0
        while iterations < max_iterations:
            v /= beta
            if accumulate is None:
                applied = operator @ v
                if not first:
                    axpy(applied, -beta / old_beta, previous)
                previous = applied
                del applied
            else:
                if first:
                    previous.fill(0.0)
                else:
                    previous *= -beta / old_beta
                accumulate(v, previous)
            alpha = np.dot(v, previous)
            old_beta = beta
            beta_squared = 0.0
            for part, values in preconditioner_blocks():
                previous[part] -= alpha * values * v[part]
                beta_squared += np.dot(previous[part], previous[part] / values)
            beta = np.sqrt(beta_squared)
            norm_squared += alpha**2 + old_beta**2 + beta**2
            old_eps = epsln
            delta = cs * dbar + sn * alpha
            gbar = sn * dbar - cs * alpha
            epsln, dbar = sn * beta, -cs * beta
            gamma = max(np.hypot(gbar, beta), eps)
            cs, sn = gbar / gamma, beta / gamma
            phi, phibar = cs * phibar, sn * phibar
            old_w *= -old_eps
            axpy(old_w, -delta, w)
            old_w += v
            old_w /= gamma
            w, old_w = old_w, w
            axpy(solution, phi, w)
            iterations += 1
            estimate = (
                np.sqrt(norm_squared) * np.linalg.norm(solution) * tolerance * 0.1
            )
            if (
                abs(phibar) <= estimate
                or beta <= eps * old_beta
                or iterations == max_iterations
            ):
                break
            # v is no longer needed by the direction update. Restore the
            # current unpreconditioned residual in place, then rotate buffers.
            for part, values in preconditioner_blocks():
                v[part] *= values * old_beta
                previous[part] /= values
            previous, v = v, previous
            first = False
        # Release Lanczos buffers before allocating the true residual.
        del previous, v
        residual = operator @ solution
        np.subtract(rhs, residual, out=residual)
        residual_norm = np.linalg.norm(residual)
        relative = float(residual_norm / rhs_norm)
        if residual_norm <= target:
            history.append(relative)
            return solution, iterations, history
        if not np.isfinite(relative) or (history and relative >= history[-1]):
            break
        history.append(relative)
        previous = np.empty_like(rhs)
    raise RuntimeError(
        f"Shared FIC-CASPT2 MINRES residual is {relative:g} after {iterations} iterations."
    )


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
    eliminate_external_doubles=False,
    direct_workers=1,
    excitation_pair_cache=None,
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

    from .caspt2 import (
        CASPT2_PERTURBER_CLASSES,
        CASPT2Component,
        _resolve_direct_workers,
    )

    if ic_convention not in ("projected", "molcas"):
        raise ValueError("ic_convention must be 'projected' or 'molcas'.")
    if fock_contraction not in ("operator", "density", "rdm"):
        raise ValueError("fock_contraction must be 'operator', 'density', or 'rdm'.")
    if fock_contraction == "rdm" and norm_tol < 1e-12:
        raise ValueError("RDM contractions require norm_tol >= 1e-12.")
    if eliminate_external_doubles and fock_contraction != "rdm":
        raise ValueError(
            "External-doubles elimination requires shared RDM contractions."
        )
    started = perf_counter()
    nmo = len(h1)
    pattern_counts = _external_pattern_counts(ncore - frozen_core, nmo - ncore - ncas)
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
    for pattern, count in pattern_counts.items():
        if progress:
            progress(f"Building active template {pattern}, external groups={count}")
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
    templates = {p: t for p, t in templates.items() if t.rank}
    double_templates = (
        {p: t for p, t in templates.items() if len(p[0]) == len(p[1]) == 2}
        if eliminate_external_doubles
        else {}
    )
    templates = {p: t for p, t in templates.items() if p not in double_templates}
    build_seconds = perf_counter() - started
    size = sum(pattern_counts[p] * t.rank for p, t in templates.items())
    axis_counts = (
        _external_axis_counts(ncore - frozen_core),
        _external_axis_counts(nmo - ncore - ncas),
    )
    orbital_label_bytes = 4 * sum(
        len(axis) * axis_counts[k][axis]
        for k in range(2)
        for axis in {p[k] for p in templates}
    )
    work_vectors = _MINRES_WORK_VECTORS if eliminate_external_doubles else 64
    # Krylov storage, RHS, diagonals, amplitudes and full-length temporaries.
    planned_bytes = size * 8 * work_vectors + retained_bytes + orbital_label_bytes
    if eliminate_external_doubles:
        nocc = ncore + ncas
        planned_bytes += 64 * (nmo - ncore) * (nocc - frozen_core)
        if double_templates:
            nv = nmo - nocc
            width = min(nv, max(1, _DOUBLE_TILE_PAIRS // nv))
            parent_classes = {(1, 1), (1, 2), (2, 1)}
            parent_rank = max(
                (
                    t.rank
                    for p, t in templates.items()
                    if tuple(map(len, p)) in parent_classes
                ),
                default=0,
            )
            double_rank = max(t.rank for t in double_templates.values())
            planned_bytes += width * nv * (256 + 32 * parent_rank * (1 + double_rank))
            planned_bytes += 8 * nv * (nv + 1)
            nh = ncore - frozen_core
            planned_bytes += 16 * nh * (nh + 1)
            if isinstance(eri, ReferencePairFactors):
                planned_bytes += 16 * len(eri.factors) * (nv + width)
    if max_memory_mb is not None and planned_bytes > max_memory_mb * 1024**2:
        raise MemoryError(
            f"Shared FIC workspace requires at least {planned_bytes / 1024**2:.1f} MiB; "
            f"max_memory_mb={max_memory_mb}."
        )
    groups = _grouped_external_spaces(ncore, nmo, ncas, frozen_core, tuple(templates))
    diagonal_start = perf_counter()
    diagonal, slices = {}, {}
    eps = np.diag(fock)
    offset = 0
    for pattern, keys in groups.items():
        template = templates[pattern]
        hole_shifts = eps[keys.holes].sum(axis=1)
        virtual_shifts = eps[keys.virtuals].sum(axis=1)
        count = len(keys) * template.rank
        if eliminate_external_doubles:
            diagonal[pattern] = (hole_shifts, virtual_shifts, template.values)
        else:
            shifts = (hole_shifts[:, None] - virtual_shifts).ravel()
            diagonal[pattern] = (
                reference_active_fock + shifts[:, None] - template.values
            )
        slices[pattern] = slice(offset, offset + count)
        offset += count
    diagonal_seconds = perf_counter() - diagonal_start
    graph_start = perf_counter()
    if progress:
        progress("Building shared inter-space Fock graph")
    graph_budget = None if budget is None else budget - planned_bytes
    pair_cache = excitation_pair_cache if excitation_pair_cache is not None else {}
    cached_pairs = pair_cache.get("active_pair_integrals")
    cached_bytes = cached_pairs.nbytes if cached_pairs is not None else 0
    del cached_pairs
    compact_graph = eliminate_external_doubles
    try:
        connections, graph_bytes, topology_bytes = _fock_connections(
            groups,
            templates,
            fock,
            ncore,
            ncas,
            None if graph_budget is None else graph_budget - cached_bytes,
            ic_convention=ic_convention,
            fock_contraction=fock_contraction,
            compact=compact_graph,
        )
    except MemoryError:
        if not cached_bytes:
            raise
        del pair_cache["active_pair_integrals"]
        connections, graph_bytes, topology_bytes = _fock_connections(
            groups,
            templates,
            fock,
            ncore,
            ncas,
            graph_budget,
            ic_convention=ic_convention,
            fock_contraction=fock_contraction,
            compact=compact_graph,
        )
    if (
        max_memory_mb is not None
        and planned_bytes + graph_bytes > max_memory_mb * 1024**2
    ):
        raise MemoryError("Shared FIC Fock graph exceeds max_memory_mb.")
    graph_seconds = perf_counter() - graph_start
    doubles = None
    if double_templates:
        doubles = _ExternalDoubles(
            double_templates,
            templates,
            groups,
            fock,
            ncore,
            ncas,
            frozen_core,
            reference_active_fock,
            real_shift,
            imaginary_shift,
            screen_tol,
        )

    def unpack(vector):
        return {
            p: vector[slices[p]].reshape(len(keys), templates[p].rank)
            for p, keys in groups.items()
        }

    def pack(parts):
        return (
            np.concatenate([parts[p].ravel() for p in groups])
            if groups
            else np.empty(0)
        )

    from .caspt2 import _cpp_attr

    graph_action = None if compact_graph else _cpp_attr("apply_fic_graph")
    if compact_graph:
        from .fic_fock_action import apply_compact_fock_graph

        max_rank = max((t.rank for t in templates.values()), default=0)
        scratch = max(max_rank, 1 << 16) if connections else 0
        # np.take can buffer its output, so reserve three bounded panels.
        graph_bytes += 24 * scratch
        if budget is not None and planned_bytes + graph_bytes > budget:
            raise MemoryError("Shared FIC action workspace exceeds max_memory_mb.")
        workspace, transpose_workspace = np.empty(scratch), np.empty(scratch)
    elif graph_action is not None:
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
        graph_bytes += (
            sum(e[6].nbytes + e[7].nbytes + e[9].nbytes for e in edges) + 16 * scratch
        )
        if budget is not None and planned_bytes + graph_bytes > budget:
            raise MemoryError("Shared FIC action workspace exceeds max_memory_mb.")
        workspace = np.empty(scratch)
        transpose_workspace = np.empty(scratch)

    rhs_start = perf_counter()
    if progress:
        progress(f"Building Hamiltonian RHS, contracted rank={size}")
    coefficients = None
    active_pair_bytes, active_pair_reused = 0, False
    if eliminate_external_doubles:
        coefficients = _ExcitationCoefficients(h1, eri, ncore, ncas, frozen_core)
        if groups:
            active_pair_bytes, active_pair_reused = (
                coefficients.prepare_active_pair_integrals(
                    None if budget is None else budget - planned_bytes - graph_bytes,
                    pair_cache,
                )
            )
        b = np.empty(size)
        rhs = unpack(b)
        for p, keys in groups.items():
            templates[p].tiled_rhs(keys, coefficients, out=rhs[p])
    else:
        single, pair, transition_ids = _hamiltonian_coefficients(
            h1, eri, ncore, ncas, frozen_core
        )
        rhs = {
            p: templates[p].rhs(keys, ncore, single, pair, transition_ids)
            @ templates[p].vectors
            for p, keys in groups.items()
        }
    rhs_seconds = diagonal_seconds + perf_counter() - rhs_start

    def diagonal_blocks(shifted_equation=True):
        for pattern, (holes, virtuals, active_values) in diagonal.items():
            rank, nv = len(active_values), len(virtuals)
            rows_per_block = max(1, _DIAGONAL_BLOCK_SIZE // rank)
            for first in range(0, len(groups[pattern]), rows_per_block):
                rows = np.arange(
                    first, min(first + rows_per_block, len(groups[pattern]))
                )
                values = (
                    reference_active_fock
                    + holes[rows // nv, None]
                    - virtuals[rows % nv, None]
                    - active_values
                ).ravel()
                if shifted_equation:
                    values -= real_shift
                    if imaginary_shift:
                        near = abs(values) < screen_tol
                        values[near] = np.copysign(screen_tol, values[near])
                        values += imaginary_shift**2 / values
                start = slices[pattern].start + first * rank
                yield slice(start, start + len(values)), values

    if not compact_graph:
        b = pack(rhs)
        rhs = unpack(b)
    if compact_graph:
        if not imaginary_shift:
            for rows, values in diagonal_blocks():
                if np.any((abs(values) < screen_tol) & (abs(b[rows]) > screen_tol)):
                    raise ZeroDivisionError(
                        "Near-zero shared FIC-CASPT2 denominator; use an imaginary shift."
                    )
    else:
        shifted = {}
        for p, d in diagonal.items():
            values = d - real_shift
            near = abs(values) < screen_tol
            if imaginary_shift:
                values = np.where(near, np.copysign(screen_tol, values), values)
                values += imaginary_shift**2 / values
            elif np.any(near & (abs(rhs[p]) > screen_tol)):
                raise ZeroDivisionError(
                    "Near-zero shared FIC-CASPT2 denominator; use an imaginary shift."
                )
            shifted[p] = values
        if diagonal:
            del d, near, values
        if graph_action is not None:
            action_diagonals = pack(diagonal), pack(shifted)
            diagonal.clear()
            shifted.clear()

    def action(vector, shifted_equation=True, out=None):
        if compact_graph:
            add = out is not None
            if out is None:
                out = np.empty_like(vector)
            for rows, values in diagonal_blocks(shifted_equation):
                if add:
                    out[rows] += vector[rows] * values
                else:
                    np.multiply(vector[rows], values, out=out[rows])
            apply_compact_fock_graph(
                unpack(vector), unpack(out), connections, workspace, transpose_workspace
            )
            return out
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

    def double_rhs_feedback(result):
        output = unpack(result)
        for pattern, holes, virtuals, _, shifted, raw in doubles.tiles(coefficients):
            tile_rhs = doubles.templates[pattern].project_rhs(raw)
            doubles.apply(pattern, holes, virtuals, None, output, tile_rhs, shifted)
        return result

    feedback_apply_seconds = 0.0

    def reduced_action(vector, out=None):
        nonlocal feedback_apply_seconds
        result = action(vector, out=out)
        if coupled_doubles:
            started_feedback = perf_counter()
            if executor is None:
                apply_external_double_feedback(
                    vector, result, hole_pairs, *feedback_arrays
                )
            else:

                def apply_chunk(pairs):
                    apply_external_double_feedback(
                        vector, result, pairs, *feedback_arrays
                    )

                for batch in parallel_rounds:
                    list(executor.map(apply_chunk, batch))
            feedback_apply_seconds += perf_counter() - started_feedback
        return result

    coupled_doubles = doubles is not None and any(doubles.edges.values())
    workers = 1
    if coupled_doubles:
        from .external_double_feedback import (
            apply_external_double_feedback,
            disjoint_hole_pair_rounds,
        )

        feedback_arrays = doubles.feedback_arrays(slices, templates)
        hole_pairs = np.column_stack(np.triu_indices(doubles.nholes))
        count = sum(doubles.counts.values())
        if count >= _DOUBLE_TILE_PAIRS:
            workers = _resolve_direct_workers(
                direct_workers, doubles.nholes // 2, count
            )
        if workers > 1:
            parallel_rounds = [
                np.array_split(pairs, min(workers, len(pairs)))
                for pairs in disjoint_hole_pair_rounds(doubles.nholes)
                if len(pairs)
            ]
            # Compile the workers' signature before their first concurrent call.
            apply_external_double_feedback(
                b, np.empty(0), hole_pairs[:0], *feedback_arrays
            )
    feedback_start = perf_counter()
    solve_rhs = b
    if coupled_doubles:
        # The original RHS is regenerated after MINRES into this buffer.
        np.negative(b, out=b)
        double_rhs_feedback(b)
        np.negative(b, out=b)
    feedback_seconds = perf_counter() - feedback_start

    def preconditioner_blocks(positive=True):
        for rows, values in diagonal_blocks():
            # Preserve the exact action diagonal. Regularize only the
            # preconditioner for uncoupled zero-denominator modes.
            values[abs(values) < screen_tol] = 1.0
            if positive:
                np.abs(values, out=values)
            yield rows, values

    if compact_graph:
        denominators = preconditioner_blocks
        initial = np.empty_like(solve_rhs)
        for rows, values in preconditioner_blocks(positive=False):
            np.divide(solve_rhs[rows], values, out=initial[rows])
    else:
        denominators = (
            action_diagonals[1].copy() if graph_action is not None else pack(shifted)
        )
        denominators[abs(denominators) < screen_tol] = 1.0
        initial = solve_rhs / denominators
    # This Gram cache accelerates RHS and bra projections, but the Schur
    # action uses only RI factors. Do not retain it alongside Krylov vectors.
    rebuild_pairs = bool(
        active_pair_bytes and size and (connections or coupled_doubles)
    )
    if rebuild_pairs:
        coefficients.active_pair_integrals = None
        pair_cache.pop("active_pair_integrals", None)
    solve_start = perf_counter()
    history = []
    iterations = 0
    solver = "shared_block_direct"
    if progress:
        progress(f"Solving coupled equation: graph terms={len(connections)}")
    if size and (connections or coupled_doubles):
        lhs = LinearOperator((size, size), matvec=reduced_action, dtype=float)
        with (
            ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()
        ) as executor:
            if eliminate_external_doubles:
                solver = "coupled_template_minres"
                solution, iterations, history = _solve_shared_minres(
                    lhs,
                    solve_rhs,
                    denominators,
                    initial,
                    solver_tol,
                    max_iterations,
                    accumulate=reduced_action,
                )
            else:
                solver = "coupled_template_gmres"
                pre = LinearOperator(
                    (size, size), matvec=lambda x, d=denominators: x / d, dtype=float
                )
                solution, info = gmres(
                    lhs,
                    solve_rhs,
                    x0=initial,
                    M=pre,
                    rtol=solver_tol,
                    atol=1e-13,
                    restart=min(40, size),
                    maxiter=max(1, (max_iterations + 39) // 40),
                    callback=lambda r: history.append(float(r)),
                    callback_type="pr_norm",
                )
                iterations = len(history)
                if info:
                    raise RuntimeError(
                        f"Shared FIC-CASPT2 did not converge: GMRES info={info}."
                    )
    else:
        solution = initial
    del denominators, initial, solve_rhs
    if rebuild_pairs:
        pair_restore_start = perf_counter()
        coefficients.prepare_active_pair_integrals(
            None if budget is None else budget - planned_bytes - graph_bytes, pair_cache
        )
        rhs_seconds += perf_counter() - pair_restore_start
    if coupled_doubles:
        rhs_restore_start = perf_counter()
        for p, keys in groups.items():
            templates[p].tiled_rhs(keys, coefficients, out=rhs[p])
        rhs_seconds += perf_counter() - rhs_restore_start
    transition_couplings = None
    bras = []
    if transition_ci is not None:
        if engine is None:
            raise NotImplementedError(
                "Shared multistate couplings require RDM contractions."
            )
        import copy

        from .rdm_contractions import RDMContractions

        transition_couplings = np.zeros(len(transition_ci))
        for bra in transition_ci:
            density = RDMContractions.transition(ci, bra, binary)
            bra_templates = {}
            for pattern, template in (templates | double_templates).items():
                gram = density.matrix(
                    template.operators, template.operators, template.ncore
                )
                proxy = copy.copy(template)
                # All Hamiltonian generators must survive even when a ket's
                # corresponding generator vanishes (transition density).
                proxy.nonzero = np.ones(len(template.operators), dtype=bool)
                proxy.norms = np.ones(len(template.operators))
                proxy.rhs_transform = template.projector.T @ gram
                bra_templates[pattern] = proxy
            bras.append(bra_templates)

    x = unpack(solution)
    double_energy = double_nonvariational = double_norm = double_rhs_norm = 0.0
    double_residual = double_equation_residual = 0.0
    cross = np.zeros_like(solution) if coupled_doubles else 0.0
    reconstruction_start = perf_counter()
    if doubles is not None:
        tile_feedback = unpack(cross) if coupled_doubles else {}
        for pattern, holes, virtuals, diag, shifted, raw in doubles.tiles(coefficients):
            tile_rhs = double_templates[pattern].project_rhs(raw)
            amp, tile_cross = doubles.apply(
                pattern, holes, virtuals, x, tile_feedback, tile_rhs, shifted
            )
            tile_applied = diag * amp + tile_cross
            double_energy += float(np.sum(amp * (2 * tile_rhs - tile_applied)))
            double_nonvariational += float(np.sum(tile_rhs * amp))
            double_norm += float(np.sum(amp**2))
            double_rhs_norm += float(np.sum(tile_rhs**2))
            double_residual += float(
                np.sum((tile_applied - real_shift * amp - tile_rhs) ** 2)
            )
            double_equation_residual += float(
                np.sum((shifted * amp + tile_cross - tile_rhs) ** 2)
            )
            for k, bra in enumerate(bras):
                transition_couplings[k] += float(
                    np.sum(bra[pattern].project_rhs(raw) * amp)
                )
    reconstruction_seconds = perf_counter() - reconstruction_start
    if doubles is not None:
        del tile_feedback
    applied = action(solution, False)
    applied += cross
    energy = 2 * b @ solution - solution @ applied + double_energy
    if compact_graph:
        # The graph and external-double contribution are identical in both
        # equations. Only the diagonal changes, so reuse the unshifted action.
        unshifted_residual_sq = equation_residual_sq = 0.0
        for (rows, raw), (_, shifted_values) in zip(
            diagonal_blocks(False), diagonal_blocks(True)
        ):
            residual_block = applied[rows] - b[rows]
            equation_block = residual_block + (shifted_values - raw) * solution[rows]
            if real_shift:
                residual_block -= real_shift * solution[rows]
            unshifted_residual_sq += np.dot(residual_block, residual_block)
            equation_residual_sq += np.dot(equation_block, equation_block)
        del cross
    else:
        unshifted_residual = applied - b
        if real_shift:
            unshifted_residual -= real_shift * solution
        unshifted_residual_sq = np.dot(unshifted_residual, unshifted_residual)
        del unshifted_residual
        equation_residual = action(solution)
        equation_residual += cross
        equation_residual -= b
        equation_residual_sq = np.dot(equation_residual, equation_residual)
        del equation_residual, cross
    ax = unpack(applied)
    if bras:
        for pattern, keys in groups.items():
            ket = x[pattern]
            if coefficients is None:
                for k, bra in enumerate(bras):
                    proxy = bra[pattern]
                    rhs_bra = (
                        proxy.rhs(keys, ncore, single, pair, transition_ids)
                        @ proxy.vectors
                    )
                    transition_couplings[k] += float(np.sum(rhs_bra * ket))
            else:
                projections = [
                    (bra[pattern].nonzero, bra[pattern].rhs_projection())
                    for bra in bras
                ]
                for rows, raw in templates[pattern].rhs_batches(keys, coefficients):
                    for k, (columns, projection) in enumerate(projections):
                        transition_couplings[k] += float(
                            np.sum((raw[:, columns] @ projection) * ket[rows])
                        )
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
    rank_doubles = 0
    raw_doubles = 0
    if doubles is not None:
        rank_doubles = sum(
            doubles.counts[p] * t.rank for p, t in double_templates.items()
        )
        raw_doubles = sum(
            doubles.counts[p] * np.count_nonzero(t.nonzero)
            for p, t in double_templates.items()
        )
        name = CASPT2_PERTURBER_CLASSES[0]
        nonvariational_components[name] = double_nonvariational
        components[name] = CASPT2Component(
            name, rank_doubles, double_energy, double_norm
        )
    rhs_norm = max(np.sqrt(np.dot(b, b) + double_rhs_norm), 1e-300)
    residual = np.sqrt(unshifted_residual_sq + double_residual)
    regularized = np.sqrt(equation_residual_sq + double_equation_residual) / rhs_norm
    if (eliminate_external_doubles or connections) and regularized > max(
        10 * solver_tol, 1e-11
    ):
        raise RuntimeError(
            f"Shared FIC-CASPT2 full equation residual is {regularized:g}."
        )
    # RHS/cache construction and post-solve diagnostics reserve six
    # full vectors. The ten-vector Krylov reserve runs with no Gram cache.
    phase_bytes = (
        max(
            planned_bytes,
            planned_bytes - size * 8 * (work_vectors - 6) + active_pair_bytes,
        )
        if rebuild_pairs
        else planned_bytes + active_pair_bytes
    )
    return dict(
        energy=float(energy),
        nonvariational=float(b @ solution + double_nonvariational),
        transition_couplings=transition_couplings,
        nonvariational_components=nonvariational_components,
        norm=float(solution @ solution + double_norm),
        components=components,
        rank=size + rank_doubles,
        iterative_rank=size,
        eliminated_groups=sum(doubles.counts.values()) if doubles else 0,
        external_double_workers=workers,
        raw_count=sum(
            len(groups[p]) * np.count_nonzero(templates[p].nonzero) for p in groups
        )
        + raw_doubles,
        template_count=len(templates) + len(double_templates),
        template_rows=sum(len(t.rows) for t in templates.values()),
        logical_rows=sum(len(groups[p]) * len(templates[p].rows) for p in groups),
        active_pair_bytes=active_pair_bytes,
        active_pair_reused=active_pair_reused,
        solver=solver,
        work_vectors=work_vectors,
        orbital_label_bytes=orbital_label_bytes,
        fock_topology_bytes=topology_bytes,
        fock_graph_bytes=graph_bytes,
        planned_bytes=max(
            phase_bytes + graph_bytes,
            engine.workspace_bytes if engine is not None else 0,
        ),
        groups=sum(map(len, groups.values())),
        graph_terms=len(connections),
        iterations=iterations,
        history=history,
        residual=float(residual),
        relative_residual=float(residual / rhs_norm),
        equation_residual=float(regularized),
        timings=dict(
            template_build_s=build_seconds,
            rhs_s=rhs_seconds,
            external_doubles_feedback_s=feedback_seconds if coupled_doubles else 0.0,
            external_doubles_matvec_s=feedback_apply_seconds,
            external_doubles_reconstruction_s=(
                reconstruction_seconds if doubles else 0.0
            ),
            fock_graph_s=graph_seconds,
            solve_s=perf_counter() - solve_start,
            total_s=perf_counter() - started,
        ),
    )
