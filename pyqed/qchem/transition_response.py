"""Blocked singlet response actions; no occupied-virtual Hessian storage."""

import numpy as np

from .basis import PackedRIFactors, mo_pair_factors
from .dft.scf import ensure_grid_for_xc
from .dft.xc import eval_fxc, xc_type, hybrid_coeff


class TransitionResponse:
    """Real closed-shell LDA/TDHF response in occupied-virtual order.

    Implements the adiabatic Casida equations (Casida, 1995,
    doi:10.1142/9789812830586_0005), with blocked quadrature and the
    reference's exact or CD/RI integrals. This is an algebraic reorganization,
    not a new XC approximation. GGA/hybrid DFT kernels are unsupported.
    Pure-LDA A-B is the orbital-gap diagonal; TDHF retains exchange.
    All MO pair factors are transformed and consumed in auxiliary blocks;
    no complete occupied-virtual or occupied-occupied factor tensor is cached.
    This trades repeated transformations for bounded response storage.
    """

    def __init__(self, td, block_size=512):
        from .tddft import _ov_blocks

        mf = td._scf
        if np.iscomplexobj(mf.mo_coeff):
            raise NotImplementedError("Response requires real orbitals.")
        energies, _, occ, vir, self.co, self.cv = _ov_blocks(mf)
        self.gap = (energies[vir] - energies[occ, None]).ravel()
        self.no, self.nv = len(occ), len(vir)
        self.mol, self.solvent = mf.mol, getattr(td, "with_solvent", None)
        self.exchange = 0.0 if hasattr(mf, "xc") else 1.0
        self.block_size = block_size
        self.grid = None
        if hasattr(mf, "xc"):
            if xc_type(mf.xc) != "LDA" or hybrid_coeff(mf.xc):
                raise NotImplementedError("Linear-response DFT requires pure LDA.")
            self.grid = ensure_grid_for_xc(mf.mol, mf.grid, mf.xc)
            self.go = self.grid.ao @ self.co
            self.gv = self.grid.ao @ self.cv
            self.weight = np.empty(len(self.grid.weights))
            for start in range(0, len(self.weight), block_size):
                s = slice(start, start + block_size)
                phi = self.grid.ao[s]
                rho = np.einsum("gp,pq,gq->g", phi, mf.dm, phi, optimize=True)
                self.weight[s] = 2 * self.grid.weights[s] * eval_fxc(rho, mf.xc)
        self.factors = getattr(mf.mol, "eri_factors", None)
        if self.factors is None:
            n = mf.mol.nao
            indices = np.arange(n)
            self.pairs = np.maximum(indices[:, None], indices) * (
                np.maximum(indices[:, None], indices) + 1
            ) // 2 + np.minimum(indices[:, None], indices)

    def _exact(self, vectors, include_b=True):
        """AO-index blocks also accept nonsymmetric transition densities."""
        density = np.einsum("pi,iak,qa->pqk", self.co, vectors, self.cv, optimize=True)
        j = np.empty_like(density)
        ka, kb = np.zeros_like(density), np.zeros_like(density)
        mol, pairs = self.mol, self.pairs
        for p in range(mol.nao):
            if getattr(mol, "eri", None) is not None:
                block = mol.eri[p]
            elif getattr(mol, "eri_s4", None) is not None:
                block = mol.eri_s4[pairs[p, :, None, None], pairs[None, :, :]]
            elif getattr(mol, "eri_s8", None) is not None:
                left, right = pairs[p, :, None, None], pairs[None, :, :]
                hi, lo = np.maximum(left, right), np.minimum(left, right)
                block = mol.eri_s8[hi * (hi + 1) // 2 + lo]
            else:
                raise NotImplementedError(
                    "Response requires stored exact or CD/RI integrals."
                )
            j[p] = np.einsum("qrs,rsk->qk", block, density, optimize=True)
            if self.exchange:
                ka[p] = np.einsum("qrs,qsk->rk", block, density, optimize=True)
                if include_b:
                    kb[p] = np.einsum("qrs,sqk->rk", block, density, optimize=True)
        project = lambda value: np.einsum(
            "pi,pqk,qa->iak", self.co, value, self.cv, optimize=True
        )
        return 2 * project(j), project(ka), project(kb)

    def kernels(self, vectors, *, include_b=True):
        v = vectors.reshape(self.no, self.nv, -1)
        if self.factors is None:
            common, ka, kb = self._exact(v, include_b)
        else:
            dtype = np.dtype(np.result_type(self.factors.dtype, v))
            common = np.zeros(v.shape, dtype=dtype)
            ka, kb = np.zeros_like(common), np.zeros_like(common)
            itemsize = dtype.itemsize
            work = (
                2 * self.mol.nao**2
                + self.no**2
                + self.nv**2
                + 2 * self.no * self.nv
                + (3 * self.no * self.nv + self.no**2) * v.shape[-1]
            )
            block = max(1, min(64, (32 << 20) // (itemsize * work)))
            for start in range(0, len(self.factors), block):
                s = slice(start, start + block)
                factors = (
                    self.factors.pair_factors[s]
                    if isinstance(self.factors, PackedRIFactors)
                    else self.factors[s]
                )
                ov = mo_pair_factors(factors, self.co, self.cv)
                pairs = ov.reshape(len(ov), -1)
                common += (2 * pairs.T @ (pairs @ vectors)).reshape(v.shape)
                if self.exchange:
                    oo = mo_pair_factors(factors, self.co)
                    vv = mo_pair_factors(factors, self.cv)
                    projected = np.einsum("Pij,jbk->Pibk", oo, v, optimize=True)
                    ka += np.einsum("Pab,Pibk->iak", vv, projected, optimize=True)
                    if include_b:
                        projected = np.einsum("Pib,jbk->Pijk", ov, v, optimize=True)
                        kb += np.einsum("Pja,Pijk->iak", ov, projected, optimize=True)
        common = common.reshape(vectors.shape)
        if self.grid is not None:
            for start in range(0, len(self.weight), self.block_size):
                s = slice(start, start + self.block_size)
                pairs = (self.go[s, :, None] * self.gv[s, None, :]).reshape(
                    len(self.go[s]), -1
                )
                common += pairs.T @ (self.weight[s, None] * (pairs @ vectors))
        if self.solvent is not None:
            for k in range(v.shape[-1]):
                density = self.co @ v[:, :, k] @ self.cv.T
                common[:, k] += (
                    2 * self.co.T @ self.solvent._B_dot_x(density) @ self.cv
                ).ravel()
        return common - ka.reshape(vectors.shape), (
            common - kb.reshape(vectors.shape) if include_b else None
        )

    def tda(self, vectors):
        return self.gap[:, None] * vectors + self.kernels(vectors, include_b=False)[0]

    def casida(self, vectors):
        root = np.sqrt(self.gap)[:, None]
        return (
            self.gap[:, None] ** 2 * vectors
            + 2 * root * self.kernels(root * vectors, include_b=False)[0]
        )

    def rpa(self, vectors):
        x, y = np.split(vectors, 2)
        if not self.exchange:
            common = self.kernels(x + y, include_b=False)[0]
            return np.vstack(
                (self.gap[:, None] * x + common, -self.gap[:, None] * y - common)
            )
        ax, bx = self.kernels(x)
        ay, by = self.kernels(y)
        return np.vstack(
            (self.gap[:, None] * x + ax + by, -self.gap[:, None] * y - ay - bx)
        )
