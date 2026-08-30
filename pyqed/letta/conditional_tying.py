"""Dense-reference LETTA with matrix-valued conditional tie factors.

This module validates a local tensor-train factorization of a graph-tied
LETTA tensor.  It deliberately uses dense many-body projectors for the local
variational solves, so it is intended for small tests before the same local
factorization is taught to the frontier contraction backends.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse

from .core import _lowest_generalized_eigenpair
from .cp_tying import (
    _expanded_ints,
    _normalized_fidelity,
    _validated_dims,
    _validated_parent_sets,
)


@dataclass(frozen=True)
class ConditionalTTUpdate:
    """Diagnostics for one matrix-valued conditional-factor update."""

    site: int
    factor: int
    raw_dim: int
    metric_rank: int
    energy_before: float
    energy: float
    accepted: bool


def _random_array(rng, shape, dtype):
    array = rng.normal(size=shape)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        array = (array + 1.0j * rng.normal(size=shape)) / np.sqrt(2.0)
    return np.asarray(array / np.sqrt(max(1, np.prod(shape))), dtype=dtype)


def _conditional_tt_svd(tensor, local_dim, parent_dims, chi):
    """Factor one dense tied tensor into conditional matrix factors."""
    tensor = np.asarray(tensor)
    local_dim = int(local_dim)
    parent_dims = tuple(int(dim) for dim in parent_dims)
    chi = int(chi)
    left_dim, right_dim = tensor.shape[:2]
    if not parent_dims:
        return [tensor.transpose(0, 2, 1).copy()]

    # Move the right backbone bond behind all physical indices:
    # (D_l, D_r, s_i, s_p...) -> (D_l, s_i, s_p..., D_r).
    order = (0,) + tuple(range(2, tensor.ndim)) + (1,)
    residual = tensor.transpose(order)
    factors = []

    matrix = residual.reshape(left_dim * local_dim, -1)
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    rank = min(chi, singular_values.size)
    factors.append(left[:, :rank].reshape(left_dim, local_dim, rank))
    residual = (singular_values[:rank, None] * right[:rank]).reshape(
        rank,
        *parent_dims,
        right_dim,
    )

    for parent_dim in parent_dims[:-1]:
        matrix = residual.reshape(rank * parent_dim, -1)
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        next_rank = min(chi, singular_values.size)
        base = left[:, :next_rank].reshape(rank, parent_dim, next_rank)
        factors.append(
            np.broadcast_to(base[:, None, :, :], (rank, local_dim, parent_dim, next_rank)).copy()
        )
        residual = (singular_values[:next_rank, None] * right[:next_rank]).reshape(
            next_rank,
            *residual.shape[2:],
        )
        rank = next_rank

    last = residual.reshape(rank, parent_dims[-1], right_dim)
    factors.append(
        np.broadcast_to(
            last[:, None, :, :],
            (rank, local_dim, parent_dims[-1], right_dim),
        ).copy()
    )
    return factors


def _materialize_conditional_factors(factors, local_dim):
    if len(factors) == 1:
        return factors[0].transpose(0, 2, 1).copy()
    pieces = []
    for physical in range(local_dim):
        value = factors[0][:, physical, :]
        for factor in factors[1:]:
            value = np.tensordot(
                value,
                factor[:, physical, :, :],
                axes=([-1], [0]),
            )
        pieces.append(np.moveaxis(value, -1, 1))
    return np.stack(pieces, axis=2)


def _bond_dimensions(nsites, D):
    if np.isscalar(D):
        value = int(D)
        dimensions = (1,) + (value,) * max(0, nsites - 1) + (1,)
    else:
        dimensions = tuple(int(value) for value in D)
        if len(dimensions) == max(0, nsites - 1):
            dimensions = (1,) + dimensions + (1,)
        elif len(dimensions) != nsites + 1:
            raise ValueError(
                "D must be scalar, contain the internal dimensions, or "
                "include both boundaries."
            )
    if dimensions[0] != 1 or dimensions[-1] != 1:
        raise ValueError("open-boundary LETTA requires unit boundary dimensions.")
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("D must contain only positive dimensions.")
    return dimensions


class ConditionalTTLETTA:
    r"""LETTA with a matrix-valued conditional tensor train at every site.

    For ``P_i=(p_1,...,p_m)``, a local tensor is represented as

    .. math::

        A^{s_i,s_{p_1}\ldots s_{p_m}}_{ab}
        = B^{s_i}_{a\gamma_1}
          C_1^{s_i s_{p_1}}{}_{\gamma_1\gamma_2}\cdots
          C_m^{s_i s_{p_m}}{}_{\gamma_m b}.

    The backbone indices ``a,b`` have dimension ``D`` and the internal
    conditional indices have maximum dimension ``chi``.  At ``chi=1`` this
    is the pair-product form ``B(s_i) prod_j C_ij(s_i,s_j)``.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        parent_sets,
        *,
        D=1,
        chi=1,
        factors=None,
        seed: int | None = None,
    ):
        self.dims = _validated_dims(dims)
        self.parent_sets = _validated_parent_sets(self.dims, parent_sets)
        self._bonds = _bond_dimensions(len(self.dims), D)
        self.D = max(self._bonds)
        self.chi = _expanded_ints(chi, len(self.dims), name="chi")

        dimension = int(np.prod(self.dims))
        self.hamiltonian = (
            hamiltonian.tocsr() if issparse(hamiltonian) else np.asarray(hamiltonian)
        )
        if self.hamiltonian.shape != (dimension, dimension):
            raise ValueError("hamiltonian shape is inconsistent with dims.")
        dtype = np.result_type(self.hamiltonian.dtype, np.complex128)
        self.rng = np.random.default_rng(seed)
        self._configs = np.asarray(list(np.ndindex(*self.dims)), dtype=np.intp)
        self.physical_groups = tuple(
            (site,) + parents for site, parents in enumerate(self.parent_sets)
        )
        if factors is None:
            self.factors = [
                self._random_site_factors(site, dtype)
                for site in range(len(self.dims))
            ]
        else:
            if len(factors) != len(self.dims):
                raise ValueError("factors must contain one factor sequence per site.")
            self.factors = [
                [
                    np.asarray(
                        factor,
                        dtype=np.result_type(np.asarray(factor).dtype, dtype),
                    ).copy()
                    for factor in site
                ]
                for site in factors
            ]
            self._validate_factors()

        self.history: list[dict] = []
        self.energy: float | None = None
        self.converged = False
        self.factorization_errors: tuple[float, ...] | None = None
        self.balance_gauges()
        self.energy = self.expectation()

    def _random_site_factors(self, site, dtype):
        group = self.physical_groups[site]
        left_dim, right_dim = self._bonds[site : site + 2]
        local_dim = self.dims[site]
        if len(group) == 1:
            return [_random_array(self.rng, (left_dim, local_dim, right_dim), dtype)]

        rank = self.chi[site]
        result = [_random_array(self.rng, (left_dim, local_dim, rank), dtype)]
        for parent in group[1:-1]:
            result.append(
                _random_array(
                    self.rng,
                    (rank, local_dim, self.dims[parent], rank),
                    dtype,
                )
            )
        result.append(
            _random_array(
                self.rng,
                (rank, local_dim, self.dims[group[-1]], right_dim),
                dtype,
            )
        )
        return result

    def _validate_factors(self):
        for site, (group, factors) in enumerate(zip(self.physical_groups, self.factors)):
            if len(factors) != len(group):
                raise ValueError(f"site {site} must contain one B/C factor per physical leg.")
            local_dim = self.dims[site]
            left_dim, right_dim = self._bonds[site : site + 2]
            if len(group) == 1:
                expected = (left_dim, local_dim, right_dim)
                if factors[0].shape != expected:
                    raise ValueError(f"factor 0 at site {site} must have shape {expected}.")
                continue
            if factors[0].ndim != 3 or factors[0].shape[:2] != (left_dim, local_dim):
                raise ValueError(f"B factor at site {site} has inconsistent dimensions.")
            if factors[0].shape[-1] > self.chi[site]:
                raise ValueError(f"B factor at site {site} exceeds chi.")
            previous = factors[0].shape[-1]
            for mode, (parent, factor) in enumerate(zip(group[1:], factors[1:]), start=1):
                expected_prefix = (previous, local_dim, self.dims[parent])
                if factor.ndim != 4 or factor.shape[:3] != expected_prefix:
                    raise ValueError(f"C factor {mode} at site {site} has inconsistent dimensions.")
                expected_right = right_dim if mode == len(group) - 1 else None
                if expected_right is not None and factor.shape[-1] != expected_right:
                    raise ValueError(f"last C factor at site {site} must end on the right D bond.")
                if expected_right is None and factor.shape[-1] > self.chi[site]:
                    raise ValueError(f"C factor {mode} at site {site} exceeds chi.")
                previous = factor.shape[-1]

    @classmethod
    def from_dense(
        cls,
        hamiltonian,
        dims,
        parent_sets,
        tensors,
        *,
        chi,
        seed: int | None = None,
    ) -> "ConditionalTTLETTA":
        """Compress unrestricted local tied tensors with conditional TT-SVD."""
        dims = _validated_dims(dims)
        parent_sets = _validated_parent_sets(dims, parent_sets)
        ranks = _expanded_ints(chi, len(dims), name="chi")
        if len(tensors) != len(dims):
            raise ValueError("tensors must contain one entry per site.")
        left_dims = tuple(int(np.asarray(tensor).shape[0]) for tensor in tensors)
        right_dims = tuple(int(np.asarray(tensor).shape[1]) for tensor in tensors)
        if left_dims[0] != 1 or right_dims[-1] != 1:
            raise ValueError("dense tensors must have unit boundary dimensions.")
        if left_dims[1:] != right_dims[:-1]:
            raise ValueError("neighboring dense tensors have inconsistent virtual bonds.")
        bonds = left_dims + (right_dims[-1],)

        factors = []
        for site, (tensor, parents, rank) in enumerate(zip(tensors, parent_sets, ranks)):
            tensor = np.asarray(tensor)
            expected = (left_dims[site], right_dims[site], dims[site]) + tuple(
                dims[parent] for parent in parents
            )
            if tensor.shape != expected:
                raise ValueError(f"tensor {site} shape must be {expected}.")
            factors.append(
                _conditional_tt_svd(
                    tensor,
                    dims[site],
                    tuple(dims[parent] for parent in parents),
                    rank,
                )
            )
        errors = []
        for tensor, site_factors in zip(tensors, factors):
            reconstructed = _materialize_conditional_factors(
                site_factors,
                tensor.shape[2],
            )
            denominator = np.linalg.norm(tensor)
            error = np.linalg.norm(reconstructed - tensor)
            errors.append(float(error / denominator) if denominator else float(error))

        result = cls(
            hamiltonian,
            dims,
            parent_sets,
            D=bonds,
            chi=ranks,
            factors=factors,
            seed=seed,
        )
        result.factorization_errors = tuple(errors)
        return result

    def copy(self):
        result = type(self)(
            self.hamiltonian,
            self.dims,
            self.parent_sets,
            D=self._bonds,
            chi=self.chi,
            factors=[[factor.copy() for factor in site] for site in self.factors],
        )
        result.history = list(self.history)
        result.energy = self.energy
        result.converged = self.converged
        result.factorization_errors = self.factorization_errors
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return result

    @property
    def nparameters(self):
        return int(sum(factor.size for site in self.factors for factor in site))

    @property
    def dense_nparameters(self):
        return int(
            sum(
                self._bonds[site]
                * self._bonds[site + 1]
                * np.prod([self.dims[index] for index in group])
                for site, group in enumerate(self.physical_groups)
            )
        )

    @property
    def compression_ratio(self):
        return float(self.nparameters / self.dense_nparameters)

    @property
    def local_ranks(self):
        return tuple(
            tuple(factor.shape[-1] for factor in factors[:-1])
            for factors in self.factors
        )

    def materialize_tensor(self, site):
        """Return one full ``(D_l,D_r,s_i,s_P...)`` local tensor."""
        site = int(site)
        factors = self.factors[site]
        return _materialize_conditional_factors(factors, self.dims[site])

    def _site_transfer(self, site):
        factors = self.factors[site]
        group = self.physical_groups[site]
        physical = self._configs[:, site]
        if len(group) == 1:
            return factors[0][:, physical, :].transpose(1, 0, 2)
        transfer = factors[0][:, physical, :].transpose(1, 0, 2)
        for parent, factor in zip(group[1:], factors[1:]):
            selected = factor[:, physical, self._configs[:, parent], :].transpose(1, 0, 2)
            transfer = np.einsum("cab,cbf->caf", transfer, selected, optimize=True)
        return transfer

    def state_vector(self, *, normalize=False):
        dtype = np.result_type(*[factor.dtype for site in self.factors for factor in site])
        environment = np.ones((len(self._configs), 1), dtype=dtype)
        for site in range(len(self.dims)):
            environment = np.einsum(
                "ca,cab->cb",
                environment,
                self._site_transfer(site),
                optimize=True,
            )
        vector = environment[:, 0]
        if normalize:
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("conditional-TT LETTA state is zero or nonfinite.")
            vector = vector / norm
        return vector

    def norm(self):
        vector = self.state_vector()
        return float(np.vdot(vector, vector).real)

    def expectation(self):
        vector = self.state_vector()
        norm = np.vdot(vector, vector)
        if abs(norm) <= 1.0e-28:
            raise ValueError("conditional-TT LETTA state is numerically zero.")
        return float(np.real(np.vdot(vector, self.hamiltonian @ vector) / norm))

    def fidelity(self, state):
        return _normalized_fidelity(self.state_vector(), state)

    def balance_gauges(self):
        """Balance scalar factor norms and normalize the represented state."""
        for factors in self.factors:
            for factor in factors[1:]:
                norm = float(np.linalg.norm(factor))
                if norm > 0.0:
                    factor /= norm
                    factors[0] *= norm
        state_norm = np.sqrt(self.norm())
        if not np.isfinite(state_norm) or state_norm <= 0.0:
            raise ValueError("conditional-TT LETTA state cannot be normalized.")
        self.factors[0][0] /= state_norm
        return self

    def _block_projector(self, site, factor_index):
        block = self.factors[site][factor_index]
        original = block.copy()
        projector = np.empty((len(self._configs), block.size), dtype=complex)
        try:
            block.fill(0)
            for column in range(block.size):
                block.flat[column] = 1.0
                projector[:, column] = self.state_vector()
                block.flat[column] = 0.0
        finally:
            block[...] = original
        return projector

    def optimize_factor(self, site, factor, *, metric_tol=1.0e-12):
        """Optimize one B/C factor exactly inside its current linear span."""
        site = int(site)
        factor = int(factor)
        if site < 0 or site >= len(self.dims):
            raise IndexError("site is out of range.")
        if factor < 0 or factor >= len(self.factors[site]):
            raise IndexError("factor is out of range.")
        block = self.factors[site][factor]
        old_block = block.copy()
        energy_before = self.expectation()
        projector = self._block_projector(site, factor)
        metric = projector.T.conj() @ projector
        effective = projector.T.conj() @ (self.hamiltonian @ projector)
        metric = 0.5 * (metric + metric.T.conj())
        effective = 0.5 * (effective + effective.T.conj())
        eigenvalues = np.linalg.eigvalsh(metric)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        metric_rank = int(np.count_nonzero(eigenvalues > metric_tol * scale))
        accepted = False
        energy = energy_before
        try:
            _local_energy, vector = _lowest_generalized_eigenpair(
                effective,
                metric,
                metric_tol=metric_tol,
            )
            block[...] = vector.reshape(block.shape)
            energy = self.expectation()
            tolerance = 128.0 * np.finfo(float).eps * max(1.0, abs(energy_before))
            accepted = np.isfinite(energy) and energy <= energy_before + tolerance
        except (ValueError, np.linalg.LinAlgError):
            accepted = False
        if not accepted:
            block[...] = old_block
            energy = energy_before
        self.energy = float(energy)
        return ConditionalTTUpdate(
            site=site,
            factor=factor,
            raw_dim=block.size,
            metric_rank=metric_rank,
            energy_before=float(energy_before),
            energy=float(energy),
            accepted=bool(accepted),
        )

    def run(self, *, nsweeps=4, tol=1.0e-10, metric_tol=1.0e-12, verbose=False):
        """Alternately optimize the matrix-valued B/C factors."""
        nsweeps = int(nsweeps)
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        previous = self.expectation()
        self.history = []
        self.converged = False
        for sweep in range(nsweeps):
            updates = []
            sites = tuple(range(len(self.dims)))
            if sweep % 2:
                sites = sites[::-1]
            for site in sites:
                factors = tuple(range(len(self.factors[site])))
                if sweep % 2:
                    factors = factors[::-1]
                for factor in factors:
                    updates.append(
                        self.optimize_factor(site, factor, metric_tol=metric_tol)
                    )
            self.balance_gauges()
            energy = self.expectation()
            delta = abs(energy - previous)
            self.energy = energy
            self.history.append(
                {
                    "sweep": sweep,
                    "energy": energy,
                    "delta_energy": delta,
                    "accepted_factors": sum(update.accepted for update in updates),
                    "updates": updates,
                }
            )
            if verbose:
                print(
                    f"conditional-TT sweep={sweep:2d} E={energy: .12f} "
                    f"dE={delta:.3e} accepted={sum(update.accepted for update in updates)}"
                )
            if delta <= float(tol):
                self.converged = True
                break
            previous = energy
        return self


__all__ = ["ConditionalTTLETTA", "ConditionalTTUpdate"]
