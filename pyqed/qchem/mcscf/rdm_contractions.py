"""Exact spin-free contractions after tracing closed and empty external shells.

External Wick contractions and active normal ordering generate reusable tensor
index plans. Active intermediates are represented by RAS-safe RDMs, never by
projection into the reference RAS space. The supported Fock moment uses a
pseudocanonical active operator and symmetric energy-weighted 1--3 RDMs.
This is an algebraic evaluation of the FIC matrix elements in Malmqvist et al.,
JCP 128, 204109 (2008), doi:10.1063/1.2920188; the surrounding solver retains
its documented single-state, zero-IPEA and optional Molcas conventions.
"""

from collections import defaultdict
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2048)
def _plan(signature):
    fields = []
    slot = 0
    for spin, (p, q) in enumerate(signature):
        for create, orb in ((True, p), (False, q)):
            if orb == "a":
                orb = ("a", slot)
                slot += 1
            fields.append((create, orb, spin))
    result = defaultdict(int)

    def union(parent, a, b):
        a, b = parent[a], parent[b]
        return tuple(a if x == b else x for x in parent)

    def finish(word, parent, deltas, coefficient):
        for i in range(len(word) - 1):
            if not word[i][0] and word[i + 1][0]:
                a, b = word[i : i + 2]
                finish(
                    word[:i] + word[i + 2 :],
                    union(parent, a[2], b[2]),
                    deltas + ((a[1][1], b[1][1]),),
                    coefficient,
                )
                finish(word[:i] + (b, a) + word[i + 2 :], parent, deltas, -coefficient)
                return
        creators = [f for f in word if f[0]]
        annih = [f for f in word if not f[0]]
        paired = []
        indices = []
        for c in creators:
            matches = [j for j, a in enumerate(annih) if parent[a[2]] == parent[c[2]]]
            if len(matches) != 1:
                raise RuntimeError("Invalid spin-free contraction topology.")
            j = matches[0]
            paired.append(j)
            indices.extend((c[1][1], annih[j][1][1]))
        permutation = paired[::-1]
        parity = sum(
            permutation[i] > permutation[j]
            for i in range(len(permutation))
            for j in range(i + 1, len(permutation))
        )
        closed = len(set(parent) - {parent[f[2]] for f in word})
        result[
            tuple(indices), tuple(sorted(set(tuple(sorted(d)) for d in deltas)))
        ] += (coefficient * (-1) ** parity * 2**closed)

    def external(word, parent, coefficient):
        i = next((i for i, f in enumerate(word) if f[1][0] != "a"), None)
        if i is None:
            finish(word, parent, (), coefficient)
            return
        a = word[i]
        if (a[1][0] == "c") != a[0]:
            return
        for j in range(i + 1, len(word)):
            b = word[j]
            if b[1] == a[1] and b[0] != a[0]:
                external(
                    word[:i] + word[i + 1 : j] + word[j + 1 :],
                    union(parent, a[2], b[2]),
                    coefficient * (-1) ** (j - i - 1),
                )

    external(tuple(fields), tuple(range(len(signature))), 1)
    return tuple((indices, deltas, c) for (indices, deltas), c in result.items() if c)


def _normal_rdms(ci, binary):
    from .nevpt2 import _spin_free_rdms123

    return _normal_order(float(np.dot(ci, ci)), *(_spin_free_rdms123(ci, binary)))


def _normal_order(norm, d1, d2, d3):
    d1, d2, d3 = (x.copy() for x in (d1, d2, d3))
    n = len(d1)
    for q in range(n):
        d2[:, q, q, :] -= d1
    for q in range(n):
        d3[:, q, q, :, :, :] -= d2
        d3[:, :, :, q, q, :] -= d2
        d3[:, q, :, :, q, :] -= d2.transpose(0, 2, 3, 1)
        for s in range(n):
            d3[:, q, q, s, s, :] -= d1
    return (norm, d1, d2, d3)


@lru_cache(maxsize=2048)
def _encoded_plan(signature):
    terms = _plan(signature)
    width = 8 + 2 * max((len(d) for i, d, c in terms), default=0)
    result = np.full((len(terms), width), -1, dtype=np.int64)
    for row, (indices, deltas, c) in zip(result, terms):
        if len(indices) > 6:
            raise NotImplementedError(
                "Contraction requires more than the active three-RDM."
            )
        row[:2] = c, len(indices)
        row[2 : 2 + len(indices)] = indices
        row[8 : 8 + 2 * len(deltas)] = np.asarray(deltas, dtype=np.int64).ravel()
    return result


class RDMContractions:
    """Shared real reference densities and external-shell contraction plans."""

    @classmethod
    def transition(cls, bra, ket, binary):
        """Ordered real transition 1--3 RDMs, retaining the RAS closure.

        Reuses the excitation-pair kernels; unlike polarization of diagonal
        densities this retains the antisymmetric transition contribution.
        Weighted Fock moments are deliberately unavailable on this object.
        """
        from .nevpt2 import _rdm_excitation_closure, _spin_free_excitation_pairs

        bra, rows = _rdm_excitation_closure(bra, binary)
        ket, other = _rdm_excitation_closure(ket, binary)
        if not np.array_equal(rows, other):
            raise ValueError(
                "Transition references must share the same excitation closure."
            )
        n = binary.shape[-1]
        left, _ = _spin_free_excitation_pairs(bra, rows, packed=True)
        right, pairs = _spin_free_excitation_pairs(ket, rows, packed=True)
        left = left.transpose(1, 0, 2).reshape(n * n, -1)
        right = right.reshape(n * n, -1)
        d1 = right @ bra
        d2 = left @ right.T
        a, b = np.triu_indices(n * n)
        d3 = np.empty((n * n, n * n, n * n))
        d3[:, a, b] = left @ pairs.T
        d3[:, b, a] = d3[:, a, b]
        mask = (a != b) & (a % n == b // n)
        d3[:, b[mask], a[mask]] -= d2[:, a[mask] // n * n + b[mask] % n]
        mask = (a != b) & (a // n == b % n)
        d3[:, b[mask], a[mask]] += d2[:, b[mask] // n * n + a[mask] % n]
        obj = cls.__new__(cls)
        obj.ncas = n
        obj.rdms = _normal_order(
            float(bra @ ket),
            d1.reshape(n, n),
            d2.reshape((n,) * 4),
            d3.reshape((n,) * 6),
        )
        obj.weighted = tuple(np.zeros_like(d) for d in obj.rdms)
        obj.center = 0.0
        obj._transition_only = True
        return obj

    def __init__(self, ci, binary, energies, max_memory_mb=None):
        self.ncas = binary.shape[2]
        from math import comb

        na, nb = binary[0].sum(axis=1)
        determinants = comb(self.ncas, int(na)) * comb(self.ncas, int(nb))
        n2 = self.ncas**2
        estimate = (4 * n2 * (n2 + 1) + 80 * n2) * determinants + 128 * self.ncas**6
        self.workspace_bytes = estimate
        if max_memory_mb is not None and estimate > max_memory_mb * 1024**2:
            raise MemoryError("RDM contraction workspace exceeds max_memory_mb.")
        if not self.ncas or not na + nb:
            self.rdms = (float(np.dot(ci, ci)),) + tuple(
                np.zeros((self.ncas,) * (2 * k)) for k in (1, 2, 3)
            )
            self.weighted = tuple(d * 0 for d in self.rdms)
            self.center = 0.0
            return
        from .nevpt2 import _rdm_excitation_closure, _spin_free_excitation_pairs

        ci, binary = _rdm_excitation_closure(ci, binary)
        n = self.ncas
        ndet = len(ci)
        singles, pair_rows = _spin_free_excitation_pairs(ci, binary, packed=True)
        left = singles.transpose(1, 0, 2).reshape(n * n, ndet)
        right = singles.reshape(n * n, ndet)
        a, b = np.triu_indices(n * n)

        def expand(values, second):
            result = np.empty((n * n, n * n, n * n))
            result[:, a, b] = values
            result[:, b, a] = values
            # [E_rs,E_tu] = delta_st E_ru - delta_ru E_ts. This also
            # holds after projection onto the RAS-safe intermediate closure.
            mask = (a != b) & (a % n == b // n)
            result[:, b[mask], a[mask]] -= second[:, a[mask] // n * n + b[mask] % n]
            mask = (a != b) & (a // n == b % n)
            result[:, b[mask], a[mask]] += second[:, b[mask] // n * n + a[mask] % n]
            return result.reshape(n * n, n**4)

        d1 = right @ ci
        d2 = left @ right.T
        d3 = expand(left @ pair_rows.T, d2)
        diagonal = np.asarray(binary).sum(axis=1) @ energies
        center = float(np.dot(ci * ci, diagonal) / np.dot(ci, ci))
        diagonal -= center
        weighted_left = left * diagonal
        delta = (energies[:, None] - energies).ravel()
        pair_delta = (delta[:, None] + delta).ravel()
        # [F,E_pq]=(epsilon_p-epsilon_q)E_pq supplies both sides of the
        # symmetric transition density from one excitation-pair tensor.
        w1 = right @ (diagonal * ci) - 0.5 * delta * d1
        w2 = weighted_left @ right.T
        w3 = (
            expand(weighted_left @ pair_rows.T, w2)
            - 0.5 * (-delta[:, None] + pair_delta) * d3
        )
        w2 -= 0.5 * (-delta[:, None] + delta) * d2
        shape = lambda x, k: x.reshape((n,) * (2 * k))
        self.rdms = _normal_order(
            float(ci @ ci), shape(d1, 1), shape(d2, 2), shape(d3, 3)
        )
        self.center = center
        self.weighted = _normal_order(
            float(ci @ (diagonal * ci)), shape(w1, 1), shape(w2, 2), shape(w3, 3)
        )

    def prepare(self, words, ncore, bra=False):
        """Group operator words into reusable external signatures and active indices."""
        groups = defaultdict(list)
        for i, word in enumerate(words):
            if bra:
                word = tuple((q, p) for p, q in reversed(word))
            signature = []
            values = []
            for pair in word:
                converted = []
                for x in pair:
                    if ncore <= x < ncore + self.ncas:
                        converted.append("a")
                        values.append(x - ncore)
                    else:
                        converted.append(
                            ("c", x) if x < ncore else ("v", x - ncore - self.ncas)
                        )
                signature.append(tuple(converted))
            groups[tuple(signature)].append((i, values))
        return tuple(
            (
                s,
                np.array([i for i, v in rows]),
                np.array([v for i, v in rows], dtype=int),
            )
            for s, rows in groups.items()
        )

    def matrix(self, left, right, ncore, middle=(), weighted=False, centered=False):
        """Return <left† middle right> for lists of spin-free operator words."""
        return self.contract(
            self.prepare(left, ncore, True),
            self.prepare(right, ncore),
            ncore,
            middle,
            weighted,
            centered,
        )

    def contract(self, lg, rg, ncore, middle=(), weighted=False, centered=False):
        """Contract prepared bra/ket words, reusing their classification across Fock indices."""
        if weighted and getattr(self, "_transition_only", False):
            raise NotImplementedError(
                "Transition Fock-weighted RDMs are not constructed."
            )
        ms, _, mv = self.prepare([middle], ncore)[0]
        middle_values = list(mv[0])
        dms = self.weighted if weighted else self.rdms
        from .caspt2 import _cpp_attr

        kernel = _cpp_attr("contract_rdm_plan")
        if kernel is not None:
            if not hasattr(self, "_packed_densities"):
                self._packed_densities = tuple(
                    np.concatenate([np.asarray(d).ravel() for d in densities])
                    for densities in (self.rdms, self.weighted)
                )
            packed = self._packed_densities[int(weighted)]
        result = np.zeros(
            (sum(len(i) for s, i, a in lg), sum(len(i) for s, i, a in rg))
        )
        for ls, li, la in lg:
            for rs, ri, ra in rg:
                if kernel is not None:
                    result[np.ix_(li, ri)] = kernel(
                        la,
                        ra,
                        np.asarray(middle_values, dtype=np.int64),
                        _encoded_plan(ls + ms + rs),
                        packed,
                        self.ncas,
                    )
                    continue
                variables = (
                    [la[:, i, None] for i in range(la.shape[1])]
                    + middle_values
                    + [ra[None, :, i] for i in range(ra.shape[1])]
                )
                block = np.zeros((len(li), len(ri)))
                for indices, deltas, c in _plan(ls + ms + rs):
                    rank = len(indices) // 2
                    if rank > 3:
                        raise NotImplementedError(
                            "Contraction requires more than the active three-RDM."
                        )
                    value = (
                        dms[rank][tuple(variables[i] for i in indices)]
                        if indices
                        else dms[0]
                    )
                    for i, j in deltas:
                        value = value * (variables[i] == variables[j])
                    block += c * value
                result[np.ix_(li, ri)] = block
        if weighted and not centered:
            result += self.center * self.contract(lg, rg, ncore, middle)
        return result
