"""Exact real-orbital CASSCF integral blocks, without a full MO tensor."""

import numpy as np


class IntegralBlocks:
    """Store (pq|ij) and (pi|qj), with i,j in core+active orbitals.

    This is an exact contraction reordering, not density fitting. Packed AO
    integrals remain quartic, but are unpacked one cubic slab at a time.
    MO storage and intermediates scale quadratically in the full-space size
    for fixed occupied-space size. Only real orbitals are supported.
    """

    def __init__(self, source, coefficients, nocc):
        source = np.asarray(source)
        self.source = source
        self.coefficients = np.array(coefficients, copy=True)
        if self.coefficients.ndim != 2 or not 0 < nocc <= self.coefficients.shape[1]:
            raise ValueError('Expected a coefficient matrix and valid occupied-space size.')
        self.nocc = nocc
        c, o = self.coefficients, self.coefficients[:, :nocc]
        nao = c.shape[0]
        npair = nao * (nao+1) // 2
        if source.shape not in ((npair*(npair+1)//2,), (nao,)*4):
            raise ValueError('AO integrals must be packed-s8 or a full AO tensor.')
        if np.iscomplexobj(c) or np.iscomplexobj(source):
            raise ValueError('IntegralBlocks requires real integrals and orbitals.')
        ids = np.arange(nao)
        hi, lo = np.maximum(ids[:, None], ids), np.minimum(ids[:, None], ids)
        pairs = hi * (hi+1) // 2 + lo
        pp = np.empty((nao, nao, nocc, nocc))
        po = np.empty((nao, nocc, nao, nocc))
        for p in range(nao):
            if source.ndim == 1:
                a, b = pairs[p, :, None, None], pairs[None, :, :]
                hi, lo = np.maximum(a, b), np.minimum(a, b)
                slab = source[hi * (hi + 1) // 2 + lo]
            else:
                slab = source[p]
            pp[p] = np.einsum('qrs,ri,sj->qij', slab, o, o, optimize=True)
            po[p] = np.einsum('qrs,qi,sj->irj', slab, o, o, optimize=True)
        self.ppoo = np.einsum('pa,qb,pqij->abij', c, c, pp, optimize=True)
        self.popo = np.einsum('pa,rb,pirj->aibj', c, c, po, optimize=True)
        self.ppoo.flags.writeable = self.popo.flags.writeable = False
        self.coefficients.flags.writeable = False

    def rotate(self, u):
        return type(self)(self.source, self.coefficients @ u, self.nocc)

    @property
    def pooo(self):
        return self.ppoo[:, :self.nocc]

    def transform(self, *coefficients):
        if any(np.any(c[self.nocc:]) for c in coefficients):
            raise ValueError('This integral view only transforms core+active orbitals.')
        a, b, c, d = (v[:self.nocc] for v in coefficients)
        return np.einsum('pi,qj,pqrs,rk,sl->ijkl', a.conj(), b,
                         self.ppoo[:self.nocc, :self.nocc], c.conj(), d,
                         optimize=True)

    def veff(self, dm):
        if np.any(dm[self.nocc:]) or np.any(dm[:, self.nocc:]):
            raise ValueError('This integral view requires a core+active density.')
        d = dm[:self.nocc, :self.nocc]
        return (np.einsum('rs,pqrs->pq', d, self.ppoo, optimize=True)
                - .5 * np.einsum('rs,prqs->pq', d, self.popo, optimize=True))


class FactorIntegralBlocks(IntegralBlocks):
    """Occupied integral blocks accumulated from batches of supplied CD factors.

    No full-rank MO factor tensor is stored. Contractions are exact for the
    input factors; the CD approximation is not changed. Each batch contains
    at most 32 auxiliary indices. Real orbitals/factors only.
    """

    def __init__(self, source, coefficients, nocc, gradient_only=False):
        self.source = np.asarray(source)
        self.coefficients = np.array(coefficients, copy=True)
        if self.coefficients.ndim != 2 or not 0 < nocc <= self.coefficients.shape[1]:
            raise ValueError('Expected a coefficient matrix and valid occupied-space size.')
        self.nocc = nocc
        self.gradient_only = gradient_only
        c = self.coefficients
        if np.iscomplexobj(c) or np.iscomplexobj(self.source):
            raise ValueError('FactorIntegralBlocks requires real inputs.')
        nao, nmo = c.shape
        if self.source.ndim not in (2, 3) or self.source.shape[1:] not in (
                (nao*(nao+1)//2,), (nao, nao)):
            raise ValueError('Expected packed or full AO factors.')
        self.ppoo = np.zeros((nmo, nocc if gradient_only else nmo, nocc, nocc))
        self.popo = None if gradient_only else np.zeros((nmo, nocc, nmo, nocc))
        ids = np.arange(nao)
        hi, lo = np.maximum(ids[:, None], ids), np.minimum(ids[:, None], ids)
        pairs = hi*(hi+1)//2+lo
        batch_size = min(32, max(1, (8*1024**2)//(8*(nao**2+2*nao*nmo+nmo**2))))
        for start in range(0, self.source.shape[0], batch_size):
            batch = self.source[start:start+batch_size]
            if batch.ndim == 2:
                batch = batch[:, pairs]
            occupied = c.T @ (batch @ c[:, :nocc])
            if gradient_only:
                self.ppoo += np.einsum('Ppi,Pjk->pijk', occupied,
                                      occupied[:, :nocc], optimize=True)
                continue
            # The Coulomb block needs both free indices, only for this batch.
            full = c.T @ (batch @ c)
            self.ppoo += np.einsum('Ppq,Pij->pqij', full,
                                  occupied[:, :nocc], optimize=True)
            self.popo += np.einsum('Ppi,Pqj->piqj', occupied, occupied, optimize=True)
        self.ppoo.flags.writeable = False
        if self.popo is not None:
            self.popo.flags.writeable = False
        self.coefficients.flags.writeable = False

    def trial_rotate(self, u):
        return type(self)(self.source, self.coefficients @ u, self.nocc, gradient_only=True)

    def veff(self, dm):
        if not self.gradient_only:
            return super().veff(dm)
        if np.any(dm[self.nocc:]) or np.any(dm[:, self.nocc:]):
            raise ValueError('Trial CI requires a core+active density.')
        d = dm[:self.nocc, :self.nocc]
        eri = self.ppoo[:self.nocc]
        potential = np.zeros_like(dm)
        potential[:self.nocc, :self.nocc] = (
            np.einsum('rs,pqrs->pq', d, eri, optimize=True)
            -.5*np.einsum('rs,prqs->pq', d, eri, optimize=True))
        return potential
