"""Exact RI contractions in an orbital frame without a full MO-factor tensor.

This reorders the existing PyQED frozen-density CASSCF contractions. It adds
no screening, fitting approximation or modified Hessian model. AO factors
remain packed; only core/active MO columns may be cached on local scratch.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
from threadpoolctl import threadpool_limits

from pyqed.parallel import resolve_workers
from ..basis import PackedRIFactors, _ri_storage, _pair_factors_to_full


class OrbitalRIFactors:
    """Real orbitals over packed AO factors, with a shared panel-memory budget.

    ``memory_mb`` is a total panel-workspace target in MiB across all workers,
    not a process RSS limit. A single panel is the minimum. Outputs and the
    optional disk-backed occupied-column cache are accounted separately.
    No full MO factor array or core-inclusive four-index RDM is constructed.
    """

    def __init__(
        self, source, coefficients, *, memory_mb=384, workers=None, cache_occupied=True
    ):
        if not isinstance(source, PackedRIFactors):
            raise TypeError("Orbital RI factors require packed real AO factors.")
        c = np.asarray(coefficients)
        if c.ndim != 2 or c.shape[0] != source.nao or np.iscomplexobj(c):
            raise ValueError(
                "Expected real coefficients matching the AO factor dimension."
            )
        if not np.isfinite(memory_mb) or memory_mb <= 0:
            raise ValueError("memory_mb must be finite and positive.")
        self.source = source
        self.coefficients = np.array(c, copy=True)
        self.coefficients.flags.writeable = False
        self.memory_bytes = max(1, int(memory_mb * 2**20))
        self.workers = resolve_workers(workers)
        self.cache_occupied = bool(cache_occupied)
        self._occupied = None
        self.shape = (len(source), c.shape[1], c.shape[1])
        self.ndim, self.dtype = 3, source.dtype
        self.last_plan = {}

    def __len__(self):
        return self.shape[0]

    def rotate(self, rotation):
        return type(self)(
            self.source,
            self.coefficients @ rotation,
            memory_mb=self.memory_bytes / 2**20,
            workers=self.workers,
            cache_occupied=False,
        )

    def _run(self, bytes_per_aux, function, fixed_bytes=0):
        # Partition into at most W tasks, rather than queueing one result per
        # panel. Futures cannot retain unbounded panel outputs.
        count = max(
            1,
            min(
                self.workers,
                len(self),
                self.memory_bytes // max(1, bytes_per_aux + fixed_bytes),
            ),
        )
        block = max(
            1, (self.memory_bytes // count - fixed_bytes) // max(1, bytes_per_aux)
        )
        self.last_plan = dict(
            workers=count,
            auxiliary_block_size=block,
            planned_workspace_bytes=count * (block * bytes_per_aux + fixed_bytes),
            target_bytes=self.memory_bytes,
        )
        ranges = [
            (len(self) * i // count, len(self) * (i + 1) // count, block)
            for i in range(count)
        ]
        # One BLAS thread per worker; this context also prevents nested fanout
        # when the caller has temporarily enabled multithreaded BLAS for CI.
        with threadpool_limits(limits=1, user_api="blas"):
            if count == 1:
                return [function(ranges[0])]
            with ThreadPoolExecutor(max_workers=count) as pool:
                return list(pool.map(function, ranges))

    def _occupied_array(self, nocc):
        if self._occupied is not None and self._occupied.shape[2] == nocc:
            return self._occupied
        return None

    def active_integrals(self, h1, ncore, ncas, energy_nuc=0.0):
        """Return active h1, active RI factors and core energy in one AO pass."""
        c, a = int(ncore), int(ncas)
        o, n, m = c + a, self.shape[1], self.source.nao
        if c < 0 or a < 1 or o > n:
            raise ValueError("Invalid core/active orbital partition.")
        active = _ri_storage((len(self), a, a))
        cached = self._occupied_array(o)
        building = self.cache_occupied and cached is None
        occupied = _ri_storage((len(self), n, o)) if building else cached
        u = self.coefficients

        def work(interval):
            begin, end, block = interval
            potential, energy = np.zeros((a, a)), 0.0
            for start in range(begin, end, block):
                stop = min(start + block, end)
                if cached is None:
                    b = self.source.block(start, stop)
                    right = b @ u[:, :o]
                    del b
                    if building:
                        left = u.T @ right
                        occupied[start:stop] = left
                        boo = left[:, :o]
                    else:
                        boo = u[:, :o].T @ right
                    del right
                else:
                    boo = cached[start:stop, :o]
                cc, ac, aa = boo[:, :c, :c], boo[:, c:, :c], boo[:, c:, c:]
                trace = np.trace(cc, axis1=1, axis2=2)
                energy += 2 * np.dot(trace, trace) - np.einsum("Pij,Pji->", cc, cc)
                potential += 2 * np.einsum("Pab,P->ab", aa, trace, optimize=True)
                potential -= np.einsum("Pai,Pbi->ab", ac, ac, optimize=True)
                active[start:stop] = aa
                del boo, cc, ac, aa, trace
                if building:
                    del left
            return potential, energy

        bytes_per_aux = 8 * (m * m + m * o + (n * o if building else o * o) + 3 * o * o)
        rows = self._run(bytes_per_aux, work, fixed_bytes=8 * a * a)
        if building:
            occupied.flags.writeable = False
            self._occupied = occupied
        h = np.array(h1[c:o, c:o], copy=True)
        energy = energy_nuc + 2 * np.trace(h1[:c, :c])
        for potential, contribution in rows:
            h += potential
            energy += contribution
        return h, active, float(np.real(energy))

    def mo_factors(self, left, right=None):
        right = left if right is None else right
        l, r = self.coefficients @ left, self.coefficients @ right
        m, nl, nr = len(l), l.shape[1], r.shape[1]
        result = _ri_storage(
            (len(self), nl, nr), dtype=np.result_type(l, r, self.dtype)
        )

        def work(interval):
            begin, end, block = interval
            for start in range(begin, end, block):
                stop = min(start + block, end)
                b = self.source.block(start, stop)
                result[start:stop] = l.conj().T @ (b @ r)
                del b

        self._run(result.dtype.itemsize * (m * m + m * nr + nl * nr), work)
        return result

    def _prepare_occupied(self, nocc):
        cached = self._occupied_array(nocc)
        if cached is None and self.cache_occupied:
            # The rectangular transform is all that the Hessian/gradient uses.
            cached = self.mo_factors(
                np.eye(self.shape[1]), np.eye(self.shape[1])[:, :nocc]
            )
            cached.flags.writeable = False
            self._occupied = cached
        return cached

    @staticmethod
    def _contract(dm2, factors):
        return (
            dm2.contract(factors)
            if hasattr(dm2, "contract")
            else np.einsum("Pst,rqst->Prq", factors, dm2, optimize=True)
        )

    def generalized_fock(self, h1, dm1, dm2):
        n, o, m = self.shape[1], len(dm1), self.source.nao
        cached = self._prepare_occupied(o)
        u = self.coefficients
        dtype = np.result_type(h1, dm1, dm2.dtype)

        def work(interval):
            begin, end, block = interval
            result = np.zeros((n, o), dtype=dtype)
            for start in range(begin, end, block):
                stop = min(start + block, end)
                if cached is None:
                    b = self.source.block(start, stop)
                    right = b @ u[:, :o]
                    del b
                    left = u.T @ right
                    del right
                else:
                    left = cached[start:stop]
                contracted = self._contract(dm2, left[:, :o])
                result += np.einsum("Ppr,Prq->pq", left, contracted, optimize=True)
                del left, contracted
            return result

        item = np.dtype(dtype).itemsize
        rows = self._run(
            item * ((m * m + m * o if cached is None else 0) + 2 * n * o + 3 * o * o),
            work,
            fixed_bytes=item * n * o,
        )
        fock = np.zeros((n, n), dtype=dtype)
        fock[:, :o] = h1[:, :o] @ dm1
        for row in rows:
            fock[:, :o] += row
        return fock

    def hessian_action(self, h1, dm1, dm2, kappas):
        """Frozen-density analytic action, sharing each AO panel across vectors.

        dB[:, :, occ] = k.T @ B[:, :, occ] + C.T @ B_AO @ (C @ k[:, occ]).
        Only occupied columns and their responses enter the compact RDM
        contractions. This is the same action as the full-MO reference.
        """
        k = np.asarray(kappas)
        single = k.ndim == 2
        vectors = k[None] if single else k
        n, o, m = self.shape[1], len(dm1), self.source.nao
        if vectors.ndim != 3 or vectors.shape[1:] != (n, n) or np.iscomplexobj(k):
            raise ValueError(
                "Expected real orbital generators of shape (nmo,nmo) or (nvec,nmo,nmo)."
            )
        cached = self._prepare_occupied(o)
        u = self.coefficients
        dtype = np.result_type(h1, dm1, dm2.dtype, k)
        directions = _ri_storage((len(vectors), m, o), dtype=dtype)
        df = np.empty((len(vectors), n, o), dtype=dtype)
        for v, generator in enumerate(vectors):
            directions[v] = u @ generator[:, :o]
            df[v] = (h1 @ generator[:, :o] - generator @ h1[:, :o]) @ dm1
        lock = Lock()

        def work(interval):
            begin, end, block = interval
            for start in range(begin, end, block):
                stop = min(start + block, end)
                b = self.source.block(start, stop)
                left = (
                    cached[start:stop] if cached is not None else u.T @ (b @ u[:, :o])
                )
                base = self._contract(dm2, left[:, :o])
                for v, generator in enumerate(vectors):
                    derivative = generator.T @ left
                    right = b @ directions[v]
                    derivative += u.T @ right
                    del right
                    response = self._contract(dm2, derivative[:, :o])
                    contribution = np.einsum(
                        "Ppr,Prq->pq", derivative, base, optimize=True
                    )
                    contribution += np.einsum(
                        "Ppr,Prq->pq", left, response, optimize=True
                    )
                    with lock:
                        df[v] += contribution
                    del derivative, response, contribution
                del b, left, base

        item = np.dtype(dtype).itemsize
        self._run(
            item * (m * m + 2 * m * o + 5 * n * o + 6 * o * o),
            work,
            fixed_bytes=item * 2 * n * o,
        )
        full = np.zeros((len(vectors), n, n), dtype=dtype)
        full[:, :, :o] = df
        result = 2 * (full - full.transpose(0, 2, 1).conj())
        return result[0] if single else result

    def veff(self, dm):
        """General density fallback; direct CI uses active_integrals instead."""
        occupied = np.flatnonzero(np.any(dm != 0, axis=0) | np.any(dm != 0, axis=1))
        n, m, o = self.shape[1], self.source.nao, len(occupied)
        dtype = np.result_type(dm, self.dtype)
        if not o:
            return np.zeros((n, n), dtype=dtype)
        u, density = self.coefficients, dm[np.ix_(occupied, occupied)]

        def work(interval):
            begin, end, block = interval
            j = np.zeros(self.source.pair_shape[1], dtype=dtype)
            k = np.zeros((n, n), dtype=dtype)
            for start in range(begin, end, block):
                stop = min(start + block, end)
                b = self.source.block(start, stop)
                left = u.T @ (b @ u[:, occupied])
                del b
                charge = np.einsum(
                    "Pkl,lk->P", left[:, occupied], density, optimize=True
                )
                j += charge @ self.source.pair_factors[start:stop]
                k += np.einsum("Pil,lk,Pjk->ij", left, density, left, optimize=True)
                del left, charge
            return j, k

        item = np.dtype(dtype).itemsize
        rows = self._run(
            item * (m * m + m * o + 3 * n * o + o * o),
            work,
            fixed_bytes=item * (self.source.pair_shape[1] + 2 * n * n),
        )
        result = np.zeros((n, n), dtype=dtype)
        for j, k in rows:
            result += u.T @ _pair_factors_to_full(j[None], m)[0] @ u - 0.5 * k
        return result

    def storage_info(self):
        return dict(
            ao_factor_bytes=self.source.pair_factors.nbytes,
            full_mo_factor_bytes=0,
            occupied_factor_bytes=(
                0 if self._occupied is None else self._occupied.nbytes
            ),
            **self.last_plan,
        )
