"""Prepared reduced-MPO actions in tied local parameter coordinates."""
import numpy as np

from pyqed.mps.nonabelian import environment as env
from .parameter_operator import _positive_integer


class MPOParameterOperator:
    """Apply a frozen local MPO without assembling its parameter-space matrix.

    Wigner--Eckart contractions adapt Weichselbaum, Phys. Rev. B 86, 245124
    (2012), https://doi.org/10.1103/PhysRevB.86.245124 to tied site parameters.
    MPO/right-environment factors are computed on demand per sector pair
    and left channel and selectively retained in a bounded cache. Each action lifts parameters, contracts left factors and
    these prepared right factors, then applies the tied adjoint. No determinant
    or magnetic-state expansion is used. This implementation requires scalar
    physical multiplicities; unsupported terms raise explicitly.

    Channel regrouping is algebraically exact but changes floating-point sums.
    Both preparation and output accumulation use compensation. Independent
    reduced-action checks remain necessary; no convergence guarantee is added.
    No prepared factors are written to disk. ``storage_bytes`` bounds cached
    payload; ``factor_bytes`` limits one active factor (compensated construction
    uses up to four such arrays). Route/recoupling data are released per sector
    pair. Metadata, boundaries, lifted tensors and contraction outputs are
    outside those budgets; this is not a total process memory bound.
    Boundaries and the route plan must remain unchanged while this object lives.
    """

    def __init__(self, plan, core, left, right, *, storage_bytes=32*2**20,
                 factor_bytes=16*2**20):
        self.storage_bytes = _positive_integer(storage_bytes, 'storage_bytes')
        self.factor_bytes = _positive_integer(factor_bytes, 'factor_bytes')
        self.plan, self.core, self.left, self.right = plan, core, left, right
        self.dimension = plan.nparameters
        self.groups = [(key, shape) for key, (_, _, shape) in plan._group_by_key.items()]
        if any(shape[1] != 1 for _, shape in self.groups):
            raise NotImplementedError('Prepared MPO action requires scalar physical multiplicities')
        self.cache = {}
        self.resident_bytes = self.disk_bytes = 0
        self.cache_hits = self.factor_builds = 0
        self.closed = False

    def _factor(self, bra_key, ket_key, lc, terms, right):
        key = (bra_key, ket_key, lc)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        rb, rk = bra_key[2], ket_key[2]
        total = correction = None
        for rc, w in terms:
            block = right[rc]
            shape = (w.shape[0], block.shape[1], block.shape[2])
            if int(np.prod(shape))*16 > self.factor_bytes:
                raise MemoryError('Active MPO factor exceeds factor_bytes')
            irrep = self.core.right_channel_irreps[rc]
            weights = np.array([env._component_basis_norm(rb, rk, irrep, m)
                                for m in env.ordered_two_m_values(irrep)])
            value = np.einsum('xy,yrs,y->xrs', w[:, :, 0, 0], block, weights).astype(complex, copy=False)
            if total is None:
                total = np.zeros(shape, dtype=complex)
                correction = np.zeros_like(total)
            value -= correction
            updated = total+value
            np.subtract(updated, total, out=correction)
            correction -= value
            total = updated
        self.factor_builds += 1
        # Keep a bounded working set across matvecs; a full sequential scan
        # must not evict every previously reusable factor.
        if self.resident_bytes+total.nbytes <= self.storage_bytes:
            self.cache[key] = total
            self.resident_bytes += total.nbytes
        return total

    def matvec(self, vector):
        if self.closed:
            raise RuntimeError('Prepared MPO operator is closed')
        vector = np.asarray(vector)
        if vector.shape != (self.dimension,):
            raise ValueError('Parameter vector has the wrong shape')
        ket = self.plan.tensor(vector)
        def blocks():
            for bra_key, shape in self.groups:
                lb, pb, rb = bra_key
                out = np.zeros(shape, dtype=complex)
                correction = np.zeros_like(out)
                for ket_key, _ in self.groups:
                    lk, pk, rk = ket_key
                    e, f = self.left.get((lb, lk)), self.right.get((rb, rk))
                    if e is None or f is None:
                        continue
                    terms = env._left_reduced_rank_coupled_block(self.core, lb, lk, pb, pk, rb, rk)
                    channels = {}
                    try:
                        for (lc, rc), w in (terms or {}).items():
                            if lc in e and rc in f:
                                if w.shape[2:] != (1, 1):
                                    raise NotImplementedError('Prepared MPO action requires scalar physical multiplicities')
                                channels.setdefault(lc, []).append((rc, w))
                        for lc, routes in channels.items():
                            factor = self._factor(bra_key, ket_key, lc, routes, f)
                            product = e[lc]@ket.data[ket_key][:, 0, :]
                            value = np.sum(product@factor.transpose(0, 2, 1), axis=0)[:, None, :]
                            adjusted = value-correction
                            updated = out+adjusted
                            correction[:] = (updated-out)-adjusted
                            out[:] = updated
                            del factor, product, value, adjusted, updated
                    finally:
                        self.core._environment_reduced_block_cache.clear()
                        channels.clear()
                        del terms
                yield bra_key, out
        return self.plan.adjoint(blocks())

    def close(self):
        self.cache.clear()
        self.groups.clear()
        self.resident_bytes = 0
        self.left = self.right = self.core = self.plan = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
