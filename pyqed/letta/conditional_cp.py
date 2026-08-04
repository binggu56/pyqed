"""CP compression restricted to graph-tied parent labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cp import cp_als


@dataclass(frozen=True)
class ConditionalCPDecomposition:
    r"""Low-rank dependence on parent labels with a dense owned-site core.

    The represented tensor is

    .. math::

        A(o, p_1,\ldots,p_k)
        = \sum_r B(o,r)\prod_j U^{(j)}(r,p_j),

    where ``o`` contains both virtual bonds and the owned physical site.
    """

    core: np.ndarray
    parent_factors: tuple[np.ndarray, ...]
    original_shape: tuple[int, ...]
    relative_error: float
    n_iter: int
    converged: bool

    @property
    def rank(self) -> int:
        return int(self.core.shape[-1])

    @property
    def nparameters(self) -> int:
        return int(
            self.core.size + sum(factor.size for factor in self.parent_factors)
        )

    def reconstruct(self) -> np.ndarray:
        """Expand the conditional factors to the original dense tensor."""
        dtype = np.result_type(
            self.core.dtype,
            *[factor.dtype for factor in self.parent_factors],
        )
        result = np.zeros(self.original_shape, dtype=dtype)
        for component in range(self.rank):
            term = self.core[..., component]
            for factor in self.parent_factors:
                term = np.multiply.outer(term, factor[component])
            result += term
        return result


def _exact_label_decomposition(tensor, nparents):
    owned_shape = tensor.shape[:-nparents]
    parent_shape = tensor.shape[-nparents:]
    configurations = tuple(np.ndindex(*parent_shape))
    rank = len(configurations)
    core = np.empty(owned_shape + (rank,), dtype=tensor.dtype)
    factors = [
        np.zeros((rank, dimension), dtype=tensor.dtype)
        for dimension in parent_shape
    ]
    for component, configuration in enumerate(configurations):
        core[..., component] = tensor[(..., *configuration)]
        for factor, value in zip(factors, configuration):
            factor[component, value] = 1
    return ConditionalCPDecomposition(
        core=core,
        parent_factors=tuple(factors),
        original_shape=tuple(tensor.shape),
        relative_error=0.0,
        n_iter=0,
        converged=True,
    )


def conditional_cp_decompose(
    tensor,
    nparents: int,
    rank: int,
    *,
    max_iter: int = 500,
    tol: float = 1.0e-11,
    seeds=(0,),
) -> ConditionalCPDecomposition:
    """Compress only the trailing parent-label axes of a LETTA tensor."""
    tensor = np.asarray(tensor)
    nparents = int(nparents)
    rank = int(rank)
    if nparents < 0 or nparents > tensor.ndim:
        raise ValueError("nparents is inconsistent with the tensor order.")
    if rank < 1:
        raise ValueError("rank must be positive.")
    if nparents == 0:
        return ConditionalCPDecomposition(
            core=tensor[..., None].copy(),
            parent_factors=(),
            original_shape=tuple(tensor.shape),
            relative_error=0.0,
            n_iter=0,
            converged=True,
        )

    owned_shape = tensor.shape[:-nparents]
    parent_shape = tensor.shape[-nparents:]
    maximum_rank = int(np.prod(parent_shape, dtype=int))
    rank = min(rank, maximum_rank)
    if rank == maximum_rank:
        return _exact_label_decomposition(tensor, nparents)

    owned_size = int(np.prod(owned_shape, dtype=int))
    grouped = tensor.reshape((owned_size, *parent_shape))
    best = None
    for seed in tuple(seeds):
        candidate = cp_als(
            grouped,
            rank,
            max_iter=max_iter,
            tol=tol,
            seed=None if seed is None else int(seed),
        )
        if best is None or candidate.relative_error < best.relative_error:
            best = candidate
    if best is None:
        raise ValueError("seeds must contain at least one initialization.")

    core = (
        best.factors[0] * best.weights[None, :]
    ).reshape(owned_shape + (rank,))
    parent_factors = tuple(
        np.asarray(factor.T).copy() for factor in best.factors[1:]
    )
    return ConditionalCPDecomposition(
        core=core,
        parent_factors=parent_factors,
        original_shape=tuple(tensor.shape),
        relative_error=float(best.relative_error),
        n_iter=int(best.n_iter),
        converged=bool(best.converged),
    )


__all__ = ["ConditionalCPDecomposition", "conditional_cp_decompose"]
