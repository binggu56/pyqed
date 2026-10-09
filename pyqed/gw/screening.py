"""Exact factorized storage for real molecular spectral screening."""

import numpy as np
import scipy.linalg as la


class StaticScreening:
    """Direct-RPA static BSE screening from the auxiliary dielectric factor.

    Exact evaluation of R = (I - epsilon(0)^-1)/4, equivalent to the complete
    spectral sum M M / omega in the existing static BSE kernel (Bruneval,
    JCP 136, 194107 (2012), doi:10.1063/1.4718428). Real restricted CD/RI
    only. This changes the evaluation, not the static approximation: it is
    neither dynamic BSE nor a screening-pole truncation.
    """

    def __init__(self, factors, cholesky, max_memory=4000):
        self.factors = factors
        self.cholesky = cholesky
        self.dtype = factors.dtype
        self.block = max(1, int(max(1e-6, max_memory) * 1e6 / (64 * len(factors))))
        self._occupied = self._ov = None

    def _transform(self, left, right):
        pairs = self.factors[:, left, right]
        out = np.empty(pairs.shape, dtype=self.dtype)
        for i in range(pairs.shape[1]):
            for start in range(0, pairs.shape[2], self.block):
                s = slice(start, start + self.block)
                rhs = pairs[:, i, s]
                out[:, i, s] = (
                    rhs - la.cho_solve(self.cholesky, rhs, check_finite=False)
                ) / 4
        return out

    def static_occupied(self, nocc):
        if self._occupied is None or self._occupied.shape[1] != nocc:
            self._occupied = self._transform(slice(None, nocc), slice(None, nocc))
        return self._occupied

    def static_ov(self, nocc):
        if self._ov is None or self._ov.shape[1] != nocc:
            self._ov = self._transform(slice(None, nocc), slice(nocc, None))
        return self._ov


class FactorizedCouplings:
    """Store M[p,q,L] = sum_P factors[P,p,q] projection[P,L].

    Exact reassociation of the RI spectral GW/BSE equations; all charge
    poles are retained, with no additional rank approximation. See Bruneval,
    J. Chem. Phys. 136, 194107 (2012), doi:10.1063/1.4718428.
    Indexing forms only the requested block. Explicit array conversion
    materializes the full tensor and is intended for small references.
    """

    def __init__(self, factors, projection, poles):
        self.factors = factors
        self.projection = projection
        self.poles = np.array(poles, copy=True)
        self.shape = (*factors.shape[1:], len(poles))
        self.dtype = np.dtype(np.result_type(factors, projection))
        self._occupied = self._ov = self._static_metric = None

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        key = key + (slice(None),) * (3 - len(key))
        return np.tensordot(
            self.factors[(slice(None),) + key[:2]],
            self.projection[:, key[2]],
            axes=(0, 0),
        )

    def __array__(self, dtype=None, copy=None):
        out = np.empty(self.shape, dtype=self.dtype if dtype is None else dtype)
        for p in range(self.shape[0]):
            out[p] = self[p]
        return out

    def static_occupied(self, nocc):
        """Return sum_QL projection[P,L] projection[Q,L] F[Q,i,j]/omega[L]."""
        if self._occupied is None or self._occupied.shape[1] != nocc:
            occupied = self.factors[:, :nocc, :nocc].reshape(len(self.factors), -1)
            self._occupied = (self._metric() @ occupied).reshape(-1, nocc, nocc)
        return self._occupied

    def _metric(self):
        if self._static_metric is None:
            self._static_metric = (self.projection / self.poles) @ self.projection.T
        return self._static_metric

    def static_ov(self, nocc):
        if self._ov is None or self._ov.shape[1] != nocc:
            pairs = self.factors[:, :nocc, nocc:].reshape(len(self.factors), -1)
            self._ov = (self._metric() @ pairs).reshape(-1, nocc, self.shape[0] - nocc)
        return self._ov
