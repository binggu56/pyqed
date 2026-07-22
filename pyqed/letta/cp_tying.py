"""Dense-reference LETTA with CP-compressed physical ties.

This module is deliberately a small validation implementation.  Each site
has an MPS-like virtual core and one CP factor for every physical leg tied to
that site.  Dense state reconstruction and dense local projectors keep the
variational equations transparent for small benchmarks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse

from .core import _lowest_generalized_eigenpair
from .cp import cp_als
from .physical_tying import PhysicalTieState


def _validated_dims(dims) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in dims)
    if not dims or any(dim < 1 for dim in dims):
        raise ValueError("dims must contain positive integers.")
    return dims


def _validated_parent_sets(dims, parent_sets) -> tuple[tuple[int, ...], ...]:
    nsites = len(dims)
    parent_sets = tuple(tuple(sorted({int(parent) for parent in parents})) for parents in parent_sets)
    if len(parent_sets) == nsites - 1:
        parent_sets = parent_sets + ((),)
    if len(parent_sets) != nsites:
        raise ValueError("parent_sets must contain one set per site, or omit the terminal empty set.")
    for site, parents in enumerate(parent_sets):
        if any(parent <= site or parent >= nsites for parent in parents):
            raise ValueError(f"parents for site {site} must be future site indices.")
    if parent_sets[-1]:
        raise ValueError("the terminal site cannot have future physical parents.")
    return parent_sets


def _expanded_ints(value, size: int, *, name: str) -> tuple[int, ...]:
    if np.isscalar(value):
        values = (int(value),) * size
    else:
        values = tuple(int(item) for item in value)
    if len(values) != size or any(item < 1 for item in values):
        raise ValueError(f"{name} must contain one positive integer per site.")
    return values


def _normalized_fidelity(left, right) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    denominator = np.vdot(left, left).real * np.vdot(right, right).real
    if denominator <= 0.0:
        raise ValueError("fidelity requires nonzero states.")
    return float(abs(np.vdot(left, right)) ** 2 / denominator)


@dataclass(frozen=True)
class CPBlockUpdate:
    """Diagnostic record for one multilinear Rayleigh--Ritz update."""

    site: int
    kind: str
    mode: int | None
    raw_dim: int
    metric_rank: int
    energy_before: float
    energy: float
    accepted: bool


class CPTiedLETTA:
    r"""LETTA with CP-compressed physical dependence and virtual bond ``D``.

    The local tied tensor is

    .. math::

        T^{[k]}_{a b}(x_k, x_{P_k}) =
        \sum_{\mu=1}^{r_k} G^{[k]}_{a b \mu}
        \prod_{j\in\{k\}\cup P_k} U^{[k,j]}_{\mu x_j}.

    Contracting the virtual indices ``a,b`` along the chain gives the full
    wavefunction.  ``r_k`` compresses one physical tie and is independent of
    the propagated virtual dimension ``D``.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        parent_sets,
        *,
        bond_dim: int = 1,
        tie_ranks=2,
        cores=None,
        physical_factors=None,
        seed: int | None = None,
    ):
        self.dims = _validated_dims(dims)
        self.parent_sets = _validated_parent_sets(self.dims, parent_sets)
        self.tie_ranks = _expanded_ints(tie_ranks, len(self.dims), name="tie_ranks")
        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")

        dimension = int(np.prod(self.dims))
        self.hamiltonian = hamiltonian.tocsr() if issparse(hamiltonian) else np.asarray(hamiltonian)
        if self.hamiltonian.shape != (dimension, dimension):
            raise ValueError("hamiltonian shape is inconsistent with dims.")
        # Complex parameters avoid losing phase directions in local residual
        # and generalized-eigenvector updates, even for a real Hamiltonian.
        parameter_dtype = np.result_type(self.hamiltonian.dtype, np.complex128)
        self._configs = np.asarray(list(np.ndindex(*self.dims)), dtype=np.intp)
        self.rng = np.random.default_rng(seed)
        self.physical_sites = tuple(
            (site,) + parents for site, parents in enumerate(self.parent_sets)
        )
        bonds = (1,) + (self.bond_dim,) * max(0, len(self.dims) - 1) + (1,)

        if (cores is None) != (physical_factors is None):
            raise ValueError("cores and physical_factors must be supplied together.")
        if cores is None:
            self.cores = []
            self.physical_factors = []
            for site, rank in enumerate(self.tie_ranks):
                shape = (bonds[site], bonds[site + 1], rank)
                core = self.rng.normal(size=shape) / np.sqrt(max(1, np.prod(shape)))
                factors = []
                for physical_site in self.physical_sites[site]:
                    factor = self.rng.normal(size=(rank, self.dims[physical_site]))
                    factor /= np.maximum(np.linalg.norm(factor, axis=1, keepdims=True), 1.0e-15)
                    factors.append(factor)
                self.cores.append(core.astype(parameter_dtype))
                self.physical_factors.append(
                    [factor.astype(parameter_dtype) for factor in factors]
                )
        else:
            self.cores = [
                np.asarray(
                    core,
                    dtype=np.result_type(np.asarray(core).dtype, parameter_dtype),
                ).copy()
                for core in cores
            ]
            self.physical_factors = [
                [
                    np.asarray(
                        factor,
                        dtype=np.result_type(np.asarray(factor).dtype, parameter_dtype),
                    ).copy()
                    for factor in factors
                ]
                for factors in physical_factors
            ]
            self._validate_parameter_shapes(bonds)

        self.history: list[dict] = []
        self.adaptive_history: list[dict] = []
        self.energy: float | None = None
        self.converged = False
        self._initial_dense_factors: tuple[np.ndarray, ...] | None = None
        self.balance_gauges()
        self.energy = self.expectation()

    def _validate_parameter_shapes(self, bonds) -> None:
        if len(self.cores) != len(self.dims) or len(self.physical_factors) != len(self.dims):
            raise ValueError("cores and physical_factors must contain one entry per site.")
        for site, rank in enumerate(self.tie_ranks):
            expected_core = (bonds[site], bonds[site + 1], rank)
            if self.cores[site].shape != expected_core:
                raise ValueError(f"core {site} shape must be {expected_core}.")
            factors = self.physical_factors[site]
            if len(factors) != len(self.physical_sites[site]):
                raise ValueError(f"site {site} has the wrong number of CP leg factors.")
            for factor, physical_site in zip(factors, self.physical_sites[site]):
                expected = (rank, self.dims[physical_site])
                if factor.shape != expected:
                    raise ValueError(f"physical factor at site {site} must have shape {expected}.")

    @classmethod
    def from_physical_tie_state(
        cls,
        state: PhysicalTieState,
        hamiltonian,
        *,
        tie_ranks=2,
        bond_dim: int = 1,
        cp_max_iter: int = 300,
        cp_tol: float = 1.0e-11,
        virtual_noise: float | None = None,
        seed: int | None = 0,
    ) -> "CPTiedLETTA":
        """Initialize from dense physical-tie factors using local CP-ALS."""
        ranks = _expanded_ints(tie_ranks, len(state.dims), name="tie_ranks")
        bond_dim = int(bond_dim)
        if bond_dim < 1:
            raise ValueError("bond_dim must be positive.")
        if virtual_noise is None:
            virtual_noise = 1.0e-6 if bond_dim > 1 else 0.0
        virtual_noise = float(virtual_noise)
        if not np.isfinite(virtual_noise) or virtual_noise < 0.0:
            raise ValueError("virtual_noise must be finite and nonnegative.")
        rng = np.random.default_rng(seed)
        dense_factors = tuple(np.asarray(factor) for factor in (*state.factors, state.terminal))
        bonds = (1,) + (bond_dim,) * max(0, len(state.dims) - 1) + (1,)
        cores = []
        physical_factors = []
        for site, (tensor, rank) in enumerate(zip(dense_factors, ranks)):
            decomposition = cp_als(
                tensor,
                rank,
                max_iter=cp_max_iter,
                tol=cp_tol,
                seed=None if seed is None else seed + site,
            )
            core = np.zeros((bonds[site], bonds[site + 1], rank), dtype=tensor.dtype)
            core[0, 0, :] = decomposition.weights
            if virtual_noise:
                noise = rng.normal(size=core.shape)
                if np.iscomplexobj(tensor):
                    noise = noise + 1.0j * rng.normal(size=core.shape)
                    noise /= np.sqrt(2.0)
                core += virtual_noise * noise
            cores.append(core)
            physical_factors.append(
                [np.asarray(factor.T).copy() for factor in decomposition.factors]
            )
        obj = cls(
            hamiltonian,
            state.dims,
            state.parent_sets,
            bond_dim=bond_dim,
            tie_ranks=ranks,
            cores=cores,
            physical_factors=physical_factors,
            seed=seed,
        )
        obj._initial_dense_factors = tuple(factor.copy() for factor in dense_factors)
        return obj

    def copy(self) -> "CPTiedLETTA":
        result = type(self)(
            self.hamiltonian,
            self.dims,
            self.parent_sets,
            bond_dim=self.bond_dim,
            tie_ranks=self.tie_ranks,
            cores=[core.copy() for core in self.cores],
            physical_factors=[
                [factor.copy() for factor in factors]
                for factors in self.physical_factors
            ],
        )
        if self._initial_dense_factors is not None:
            result._initial_dense_factors = tuple(
                factor.copy() for factor in self._initial_dense_factors
            )
        result.history = list(self.history)
        result.adaptive_history = list(self.adaptive_history)
        result.energy = self.energy
        result.converged = self.converged
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return result

    @property
    def nparameters(self) -> int:
        return int(
            sum(core.size for core in self.cores)
            + sum(factor.size for factors in self.physical_factors for factor in factors)
        )

    def state_vector(self, *, normalize: bool = False) -> np.ndarray:
        """Return the dense vector, evaluating CP factors without materializing ties."""
        nconfigs = len(self._configs)
        dtype = np.result_type(
            *[core.dtype for core in self.cores],
            *[
                factor.dtype
                for factors in self.physical_factors
                for factor in factors
            ],
        )
        environment = np.ones((nconfigs, 1), dtype=dtype)
        for site, (core, factors, physical_sites) in enumerate(
            zip(self.cores, self.physical_factors, self.physical_sites)
        ):
            weights = np.ones((nconfigs, self.tie_ranks[site]), dtype=dtype)
            for factor, physical_site in zip(factors, physical_sites):
                weights *= factor[:, self._configs[:, physical_site]].T
            transfer = np.einsum("abm,cm->cab", core, weights, optimize=True)
            environment = np.einsum("ca,cab->cb", environment, transfer, optimize=True)
        vector = environment[:, 0]
        if normalize:
            norm = np.linalg.norm(vector)
            if norm <= 0.0:
                raise ValueError("CP-tied LETTA state is zero.")
            vector = vector / norm
        return vector

    def norm(self) -> float:
        vector = self.state_vector()
        return float(np.vdot(vector, vector).real)

    def expectation(self) -> float:
        vector = self.state_vector()
        norm = np.vdot(vector, vector)
        if abs(norm) <= 1.0e-28:
            raise ValueError("CP-tied LETTA state is numerically zero.")
        return float(np.real(np.vdot(vector, self.hamiltonian @ vector) / norm))

    def fidelity(self, state) -> float:
        return _normalized_fidelity(self.state_vector(), state)

    def energy_residual(self) -> tuple[float, np.ndarray, float]:
        r"""Return ``(energy, (H-E)|psi>, ||(H-E)|psi>||)`` for normalized ``psi``."""
        vector = self.state_vector(normalize=True)
        h_vector = self.hamiltonian @ vector
        energy = float(np.real(np.vdot(vector, h_vector)))
        residual = h_vector - energy * vector
        residual_norm = float(np.linalg.norm(residual))
        return energy, residual, residual_norm

    def _site_transfer(self, site: int) -> np.ndarray:
        """Return all configuration-dependent virtual transfer matrices at one site."""
        core = self.cores[site]
        dtype = np.result_type(
            core.dtype,
            *[factor.dtype for factor in self.physical_factors[site]],
        )
        weights = np.ones((len(self._configs), self.tie_ranks[site]), dtype=dtype)
        for factor, physical_site in zip(
            self.physical_factors[site],
            self.physical_sites[site],
        ):
            weights *= factor[:, self._configs[:, physical_site]].T
        return np.einsum("abm,cm->cab", core, weights, optimize=True)

    def _local_dense_projector(
        self,
        site: int,
        *,
        parents=None,
        max_projector_dim: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Map an unrestricted local tied tensor to the full state vector."""
        site = int(site)
        if site < 0 or site >= len(self.dims):
            raise IndexError("site is out of range.")
        if parents is None:
            selected_parents = self.parent_sets[site]
        else:
            selected_parents = tuple(sorted({int(parent) for parent in parents}))
            if any(parent <= site or parent >= len(self.dims) for parent in selected_parents):
                raise ValueError("candidate parents must be future physical sites.")
        physical_sites = (site,) + selected_parents
        left_dim, right_dim = self.cores[site].shape[:2]
        shape = (left_dim, right_dim) + tuple(
            self.dims[physical_site] for physical_site in physical_sites
        )
        width = int(np.prod(shape))
        if max_projector_dim is not None and width > int(max_projector_dim):
            raise ValueError("candidate local projector exceeds max_projector_dim.")

        dtype = np.result_type(
            *[core.dtype for core in self.cores],
            *[
                factor.dtype
                for factors in self.physical_factors
                for factor in factors
            ],
            complex,
        )
        left = np.ones((len(self._configs), 1), dtype=dtype)
        for other_site in range(site):
            left = np.einsum(
                "ca,cab->cb",
                left,
                self._site_transfer(other_site),
                optimize=True,
            )
        right = np.ones((len(self._configs), 1), dtype=dtype)
        for other_site in range(len(self.dims) - 1, site, -1):
            right = np.einsum(
                "cab,cb->ca",
                self._site_transfer(other_site),
                right,
                optimize=True,
            )

        projector = np.zeros((len(self._configs), width), dtype=dtype)
        for row, config in enumerate(self._configs):
            physical = tuple(config[physical_site] for physical_site in physical_sites)
            coefficients = left[row, :, None] * right[row, None, :]
            for left_index in range(left_dim):
                for right_index in range(right_dim):
                    column = np.ravel_multi_index(
                        (left_index, right_index) + physical,
                        shape,
                    )
                    projector[row, column] = coefficients[left_index, right_index]
        return projector, shape

    @staticmethod
    def _residual_projection_score(
        projector: np.ndarray,
        residual: np.ndarray,
        *,
        metric_tol: float,
    ) -> tuple[float, int]:
        """Return the squared norm of the residual projected into ``range(P)``."""
        metric = projector.T.conj() @ projector
        metric = 0.5 * (metric + metric.T.conj())
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        support = eigenvalues > float(metric_tol) * scale
        if not np.any(support):
            return 0.0, 0
        gradient = projector.T.conj() @ residual
        reduced = eigenvectors[:, support].T.conj() @ gradient
        score = np.sum(np.abs(reduced) ** 2 / eigenvalues[support])
        return float(np.real(score)), int(np.count_nonzero(support))

    @staticmethod
    def _orthonormal_range(
        projector: np.ndarray,
        *,
        metric_tol: float,
    ) -> np.ndarray:
        """Return an orthonormal basis for a projector's supported range."""
        left, singular_values, _right = np.linalg.svd(
            projector,
            full_matrices=False,
        )
        if singular_values.size == 0:
            return left[:, :0]
        scale = max(float(singular_values[0]), np.finfo(float).tiny)
        support = singular_values > np.sqrt(float(metric_tol)) * scale
        return left[:, support]

    def balance_gauges(self) -> "CPTiedLETTA":
        """Normalize CP rows, absorbing their scales into the virtual core."""
        for site, factors in enumerate(self.physical_factors):
            for component in range(self.tie_ranks[site]):
                for factor in factors:
                    norm = float(np.linalg.norm(factor[component]))
                    if norm > 0.0:
                        factor[component] /= norm
                        self.cores[site][:, :, component] *= norm
        state_norm = np.sqrt(self.norm())
        if not np.isfinite(state_norm) or state_norm <= 0.0:
            raise ValueError("CP-tied LETTA state cannot be normalized.")
        self.cores[0] /= state_norm
        return self

    def perturb(
        self,
        scale: float,
        *,
        seed: int | None = None,
    ) -> "CPTiedLETTA":
        """Add small parameter noise while retaining the current graph and ranks."""
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("perturbation scale must be finite and nonnegative.")
        if scale == 0.0:
            return self
        rng = np.random.default_rng(seed)

        def noise_like(array):
            noise = rng.normal(size=array.shape)
            if np.iscomplexobj(array):
                noise = (noise + 1.0j * rng.normal(size=array.shape)) / np.sqrt(2.0)
            return noise

        self.cores = [
            core + scale * noise_like(core)
            for core in self.cores
        ]
        self.physical_factors = [
            [
                factor + scale * noise_like(factor)
                for factor in factors
            ]
            for factors in self.physical_factors
        ]
        self.balance_gauges()
        self.energy = self.expectation()
        self.converged = False
        return self

    def _block(self, site: int, kind: str, mode: int | None):
        if kind == "core":
            if mode is not None:
                raise ValueError("core blocks do not have a physical mode.")
            return self.cores[site]
        if kind == "physical":
            if mode is None or mode < 0 or mode >= len(self.physical_factors[site]):
                raise IndexError("physical CP mode is out of range.")
            return self.physical_factors[site][mode]
        raise ValueError("kind must be 'core' or 'physical'.")

    def _block_projector(self, site: int, kind: str, mode: int | None) -> np.ndarray:
        block = self._block(site, kind, mode)
        original = block.copy()
        dtype = np.result_type(original.dtype, complex)
        projector = np.empty((int(np.prod(self.dims)), block.size), dtype=dtype)
        try:
            block.fill(0)
            for column in range(block.size):
                block.flat[column] = 1.0
                projector[:, column] = self.state_vector()
                block.flat[column] = 0.0
        finally:
            block[...] = original
        return projector

    def optimize_block(
        self,
        site: int,
        kind: str,
        mode: int | None = None,
        *,
        metric_tol: float = 1.0e-12,
    ) -> CPBlockUpdate:
        """Optimize one CP leg matrix or one virtual core exactly in its span."""
        site = int(site)
        if site < 0 or site >= len(self.dims):
            raise IndexError("site is out of range.")
        block = self._block(site, kind, mode)
        old_block = block.copy()
        energy_before = self.expectation()
        projector = self._block_projector(site, kind, mode)
        metric = projector.T.conj() @ projector
        h_projector = self.hamiltonian @ projector
        effective = projector.T.conj() @ h_projector
        metric = 0.5 * (metric + metric.T.conj())
        effective = 0.5 * (effective + effective.T.conj())
        eigenvalues = np.linalg.eigvalsh(metric)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        metric_rank = int(np.count_nonzero(eigenvalues > metric_tol * scale))
        accepted = False
        energy_after = energy_before
        try:
            _local_energy, vector = _lowest_generalized_eigenpair(
                effective,
                metric,
                metric_tol=metric_tol,
            )
            block[...] = vector.reshape(block.shape)
            energy_after = self.expectation()
            tolerance = 128.0 * np.finfo(float).eps * max(1.0, abs(energy_before))
            accepted = np.isfinite(energy_after) and energy_after <= energy_before + tolerance
        except (ValueError, np.linalg.LinAlgError):
            accepted = False
        if not accepted:
            block[...] = old_block
            energy_after = energy_before
        self.energy = float(energy_after)
        return CPBlockUpdate(
            site=site,
            kind=kind,
            mode=mode,
            raw_dim=block.size,
            metric_rank=metric_rank,
            energy_before=float(energy_before),
            energy=float(energy_after),
            accepted=bool(accepted),
        )

    def run(
        self,
        *,
        nsweeps: int = 4,
        tol: float = 1.0e-10,
        metric_tol: float = 1.0e-12,
        verbose: bool = False,
    ) -> "CPTiedLETTA":
        """Alternately optimize every CP factor and every virtual core."""
        nsweeps = int(nsweeps)
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        previous = self.expectation()
        self.history = []
        self.converged = False
        for sweep in range(nsweeps):
            updates = []
            sites = range(len(self.dims)) if sweep % 2 == 0 else range(len(self.dims) - 1, -1, -1)
            for site in sites:
                modes = range(len(self.physical_factors[site]))
                if sweep % 2:
                    modes = reversed(tuple(modes))
                    updates.append(self.optimize_block(site, "core", metric_tol=metric_tol))
                for mode in modes:
                    updates.append(
                        self.optimize_block(site, "physical", mode, metric_tol=metric_tol)
                    )
                if sweep % 2 == 0:
                    updates.append(self.optimize_block(site, "core", metric_tol=metric_tol))
            self.balance_gauges()
            energy = self.expectation()
            delta = abs(energy - previous)
            self.energy = energy
            self.history.append(
                {
                    "sweep": sweep,
                    "energy": energy,
                    "delta_energy": delta,
                    "accepted_blocks": sum(update.accepted for update in updates),
                    "updates": updates,
                }
            )
            if verbose:
                print(
                    f"CP-LETTA sweep={sweep:2d} E={energy: .12f} "
                    f"dE={delta:.3e} accepted={sum(update.accepted for update in updates)}"
                )
            if delta <= tol:
                self.converged = True
                break
            previous = energy
        return self

    def set_parent_set(self, site: int, parents) -> "CPTiedLETTA":
        """Replace one physical-parent set, preserving common CP leg factors."""
        site = int(site)
        if site < 0 or site >= len(self.dims) - 1:
            raise IndexError("site must be a nonterminal physical site.")
        parents = tuple(sorted({int(parent) for parent in parents}))
        if any(parent <= site or parent >= len(self.dims) for parent in parents):
            raise ValueError("parents must be future physical sites.")
        old_sites = self.physical_sites[site]
        old_factors = {
            physical_site: factor
            for physical_site, factor in zip(
                old_sites,
                self.physical_factors[site],
            )
        }
        dtype = np.result_type(
            self.cores[site].dtype,
            *[factor.dtype for factor in self.physical_factors[site]],
        )
        new_sites = (site,) + parents
        new_factors = []
        for physical_site in new_sites:
            if physical_site in old_factors:
                new_factors.append(old_factors[physical_site].copy())
            else:
                new_factors.append(
                    np.ones(
                        (self.tie_ranks[site], self.dims[physical_site]),
                        dtype=dtype,
                    )
                )
        parent_sets = list(self.parent_sets)
        parent_sets[site] = parents
        self.parent_sets = tuple(parent_sets)
        physical_sites = list(self.physical_sites)
        physical_sites[site] = new_sites
        self.physical_sites = tuple(physical_sites)
        self.physical_factors[site] = new_factors
        self._initial_dense_factors = None
        self.balance_gauges()
        self.energy = self.expectation()
        self.converged = False
        return self

    def _rank_residual_direction(
        self,
        site: int,
        residual: np.ndarray,
        *,
        max_projector_dim: int | None,
        metric_tol: float,
    ) -> tuple[float, np.ndarray, tuple[np.ndarray, ...]]:
        projector, dense_shape = self._local_dense_projector(
            site,
            max_projector_dim=max_projector_dim,
        )
        tangent_projectors = [self._block_projector(site, "core", None)]
        tangent_projectors.extend(
            self._block_projector(site, "physical", mode)
            for mode in range(len(self.physical_factors[site]))
        )
        tangent = np.concatenate(tangent_projectors, axis=1)
        tangent_basis = self._orthonormal_range(
            tangent,
            metric_tol=metric_tol,
        )
        incremental = projector - tangent_basis @ (
            tangent_basis.T.conj() @ projector
        )
        gradient = (incremental.T.conj() @ residual).reshape(dense_shape)
        grouped_shape = (dense_shape[0] * dense_shape[1],) + dense_shape[2:]
        decomposition = cp_als(
            gradient.reshape(grouped_shape),
            1,
            max_iter=200,
            tol=1.0e-11,
            seed=site,
        )
        direction_tensor = decomposition.reconstruct().reshape(dense_shape)
        direction = incremental @ direction_tensor.reshape(-1)
        norm = float(np.vdot(direction, direction).real)
        score = 0.0 if norm <= 0.0 else float(
            abs(np.vdot(direction, residual)) ** 2 / norm
        )
        physical_rows = tuple(
            np.asarray(factor[:, 0]).copy()
            for factor in decomposition.factors[1:]
        )
        core_direction = (
            decomposition.weights[0] * decomposition.factors[0][:, 0]
        ).reshape(dense_shape[:2])
        return score, core_direction, physical_rows

    def grow_site_rank_from_residual(
        self,
        site: int,
        residual: np.ndarray | None = None,
        *,
        max_projector_dim: int | None = 512,
        activation: float = 0.0,
        metric_tol: float = 1.0e-12,
    ) -> dict:
        """Append one residual-informed CP component.

        ``activation=0`` preserves the represented state.  A positive value
        takes a small step opposite the projected energy residual.
        """
        site = int(site)
        if site < 0 or site >= len(self.dims):
            raise IndexError("site is out of range.")
        if residual is None:
            _energy, residual, _norm = self.energy_residual()
        residual = np.asarray(residual).reshape(-1)
        if residual.size != int(np.prod(self.dims)):
            raise ValueError("residual size is inconsistent with dims.")
        activation = float(activation)
        if not np.isfinite(activation) or activation < 0.0:
            raise ValueError("activation must be finite and nonnegative.")
        score, core_direction, physical_rows = self._rank_residual_direction(
            site,
            residual,
            max_projector_dim=max_projector_dim,
            metric_tol=metric_tol,
        )
        old_rank = self.tie_ranks[site]
        core = np.zeros(
            self.cores[site].shape[:2] + (old_rank + 1,),
            dtype=self.cores[site].dtype,
        )
        core[:, :, :old_rank] = self.cores[site]
        core[:, :, old_rank] = -activation * core_direction
        self.cores[site] = core
        grown_factors = []
        for factor, row in zip(self.physical_factors[site], physical_rows):
            grown = np.empty(
                (old_rank + 1, factor.shape[1]),
                dtype=np.result_type(factor.dtype, row.dtype),
            )
            grown[:old_rank] = factor
            row_norm = float(np.linalg.norm(row))
            if row_norm <= 0.0:
                row = self.rng.normal(size=factor.shape[1])
                if np.iscomplexobj(grown):
                    row = row + 1.0j * self.rng.normal(size=factor.shape[1])
                row_norm = float(np.linalg.norm(row))
            grown[old_rank] = row / row_norm
            grown_factors.append(grown)
        self.physical_factors[site] = grown_factors
        ranks = list(self.tie_ranks)
        ranks[site] = old_rank + 1
        self.tie_ranks = tuple(ranks)
        self._initial_dense_factors = None
        self.balance_gauges()
        self.energy = self.expectation()
        self.converged = False
        return {
            "site": site,
            "rank_before": old_rank,
            "rank": old_rank + 1,
            "residual_score": score,
        }

    def residual_rank_proposals(
        self,
        max_tie_rank: int,
        residual: np.ndarray | None = None,
        *,
        max_projector_dim: int | None = 512,
        metric_tol: float = 1.0e-12,
    ) -> list[dict]:
        """Score one-component CP-rank growth from the current residual."""
        max_tie_rank = int(max_tie_rank)
        if max_tie_rank < 1:
            raise ValueError("max_tie_rank must be positive.")
        if residual is None:
            _energy, residual, _norm = self.energy_residual()
        proposals = []
        for site, rank in enumerate(self.tie_ranks):
            if rank >= max_tie_rank:
                continue
            try:
                score, _core_direction, _rows = self._rank_residual_direction(
                    site,
                    residual,
                    max_projector_dim=max_projector_dim,
                    metric_tol=metric_tol,
                )
            except ValueError:
                continue
            added_parameters = (
                self.cores[site].shape[0] * self.cores[site].shape[1]
                + sum(self.dims[physical_site] for physical_site in self.physical_sites[site])
            )
            proposals.append(
                {
                    "kind": "rank",
                    "site": site,
                    "rank_before": rank,
                    "rank": rank + 1,
                    "residual_score": score,
                    "added_parameters": int(added_parameters),
                    "efficiency": score / max(1, added_parameters),
                }
            )
        return sorted(
            proposals,
            key=lambda proposal: (
                -proposal["efficiency"],
                -proposal["residual_score"],
                proposal["site"],
            ),
        )

    def residual_parent_proposals(
        self,
        max_parents: int,
        residual: np.ndarray | None = None,
        *,
        per_site_candidates: int = 1,
        metric_tol: float = 1.0e-12,
        max_projector_dim: int | None = 512,
    ) -> list[dict]:
        """Score additions or swaps of physical parents using residual tangents."""
        max_parents = int(max_parents)
        per_site_candidates = int(per_site_candidates)
        if max_parents < 0:
            raise ValueError("max_parents must be nonnegative.")
        if per_site_candidates < 1:
            raise ValueError("per_site_candidates must be positive.")
        if residual is None:
            _energy, residual, _norm = self.energy_residual()
        proposals = []
        for site in range(len(self.dims) - 1):
            current = self.parent_sets[site]
            if len(current) > max_parents:
                raise ValueError("the current graph exceeds max_parents.")
            try:
                current_projector, _shape = self._local_dense_projector(
                    site,
                    max_projector_dim=max_projector_dim,
                )
            except ValueError:
                continue
            current_basis = self._orthonormal_range(
                current_projector,
                metric_tol=metric_tol,
            )
            current_rank = int(current_basis.shape[1])
            future = tuple(range(site + 1, len(self.dims)))
            missing = tuple(parent for parent in future if parent not in current)
            candidates = []
            if len(current) < max_parents:
                candidates.extend(
                    tuple(sorted(current + (parent,)))
                    for parent in missing
                )
            elif current:
                candidates.extend(
                    tuple(sorted((set(current) - {removed}) | {parent}))
                    for removed in current
                    for parent in missing
                )
            site_trials = []
            for parents in dict.fromkeys(candidates):
                try:
                    projector, _candidate_shape = self._local_dense_projector(
                        site,
                        parents=parents,
                        max_projector_dim=max_projector_dim,
                    )
                except ValueError:
                    continue
                incremental = projector - current_basis @ (
                    current_basis.T.conj() @ projector
                )
                gain, metric_rank = self._residual_projection_score(
                    incremental,
                    residual,
                    metric_tol=metric_tol,
                )
                site_trials.append(
                    {
                        "kind": "graph",
                        "site": site,
                        "parents_before": current,
                        "parents": parents,
                        "residual_score": gain,
                        "residual_gain": gain,
                        "metric_rank": metric_rank,
                        "current_metric_rank": current_rank,
                        "raw_dim": int(projector.shape[1]),
                    }
                )
            site_trials.sort(
                key=lambda proposal: (
                    -proposal["residual_gain"],
                    -proposal["residual_score"],
                    proposal["parents"],
                )
            )
            proposals.extend(site_trials[:per_site_candidates])
        return sorted(
            proposals,
            key=lambda proposal: (
                -proposal["residual_gain"],
                -proposal["residual_score"],
                proposal["site"],
                proposal["parents"],
            ),
        )

    def _adopt(self, other: "CPTiedLETTA") -> None:
        self.parent_sets = other.parent_sets
        self.physical_sites = other.physical_sites
        self.tie_ranks = other.tie_ranks
        self.cores = [core.copy() for core in other.cores]
        self.physical_factors = [
            [factor.copy() for factor in factors]
            for factors in other.physical_factors
        ]
        self.energy = float(other.energy)
        self.history = list(other.history)
        self.converged = other.converged
        self._initial_dense_factors = None

    def run_residual_adaptive(
        self,
        *,
        max_parents: int,
        max_tie_rank: int,
        ncycles: int = 4,
        initial_sweeps: int = 2,
        branch_sweeps: int = 2,
        candidate_budget: int = 2,
        per_site_graph_candidates: int = 1,
        residual_tol: float = 1.0e-8,
        energy_tol: float = 1.0e-10,
        metric_tol: float = 1.0e-12,
        max_projector_dim: int | None = 512,
        probe_noise: float = 1.0e-2,
        rank_activation: float = 1.0e-3,
        exploration_metric_tol: float = 1.0e-16,
        proposal_kinds: tuple[str, ...] = ("graph", "rank"),
        seed: int | None = 0,
        verbose: bool = False,
    ) -> "CPTiedLETTA":
        """Alternate residual-guided graph/rank branches with matched relaxation.

        ``candidate_budget`` is global across the enabled proposal kinds.  The
        first kind rotates with cycle parity, so budget one gives strict
        primary graph/rank alternation and larger budgets interleave the two
        families.  At CP rank one, a graph branch may also include the
        recorded compound rank activation needed to open a nonseparable tie.
        """
        ncycles = int(ncycles)
        initial_sweeps = int(initial_sweeps)
        branch_sweeps = int(branch_sweeps)
        candidate_budget = int(candidate_budget)
        if min(ncycles, initial_sweeps, branch_sweeps) < 0:
            raise ValueError("cycle and sweep counts must be nonnegative.")
        if candidate_budget < 1:
            raise ValueError("candidate_budget must be positive.")
        if isinstance(proposal_kinds, str):
            proposal_kinds = (proposal_kinds,)
        proposal_kinds = tuple(
            dict.fromkeys(str(kind).strip().lower() for kind in proposal_kinds)
        )
        unknown_kinds = set(proposal_kinds) - {"graph", "rank"}
        if not proposal_kinds or unknown_kinds:
            raise ValueError(
                "proposal_kinds must be a nonempty subset of {'graph', 'rank'}."
            )
        residual_tol = float(residual_tol)
        energy_tol = float(energy_tol)
        metric_tol = float(metric_tol)
        for name, value in (
            ("residual_tol", residual_tol),
            ("energy_tol", energy_tol),
            ("metric_tol", metric_tol),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        probe_noise = float(probe_noise)
        if not np.isfinite(probe_noise) or probe_noise < 0.0:
            raise ValueError("probe_noise must be finite and nonnegative.")
        rank_activation = float(rank_activation)
        if not np.isfinite(rank_activation) or rank_activation < 0.0:
            raise ValueError("rank_activation must be finite and nonnegative.")
        exploration_metric_tol = float(exploration_metric_tol)
        if not np.isfinite(exploration_metric_tol) or exploration_metric_tol < 0.0:
            raise ValueError("exploration_metric_tol must be finite and nonnegative.")
        if any(len(parents) > int(max_parents) for parents in self.parent_sets):
            raise ValueError("the current graph exceeds max_parents.")
        if any(rank > int(max_tie_rank) for rank in self.tie_ranks):
            raise ValueError("a current tie rank exceeds max_tie_rank.")
        if initial_sweeps:
            self.run(
                nsweeps=initial_sweeps,
                tol=0.0,
                metric_tol=metric_tol,
            )
        self.adaptive_history = []
        stalled_kinds: set[str] = set()
        for cycle in range(ncycles):
            energy_before, residual, residual_norm = self.energy_residual()
            if residual_norm <= residual_tol:
                self.converged = True
                break
            exploration = self.copy()
            exploration.perturb(
                probe_noise,
                seed=None if seed is None else seed + 1000 * cycle,
            )
            # Keep the incumbent residual as the desired correction.  Scoring
            # it in the common perturbed tangent is a support-probe heuristic
            # for otherwise hidden nodal directions; actual trial energies are
            # judged against both the incumbent and a matched perturbed control.
            proposal_residual = residual
            control = exploration.copy()
            if branch_sweeps:
                control.run(
                    nsweeps=branch_sweeps,
                    tol=0.0,
                    metric_tol=metric_tol,
                )
            control_energy = float(control.energy)
            preferred_order = (
                ("graph", "rank") if cycle % 2 == 0 else ("rank", "graph")
            )
            proposal_order = tuple(
                kind for kind in preferred_order if kind in proposal_kinds
            )
            proposal_probe = exploration
            proposals_by_kind = {}

            def proposals_for(kind):
                if kind not in proposals_by_kind:
                    if kind == "graph":
                        proposals_by_kind[kind] = (
                            proposal_probe.residual_parent_proposals(
                                max_parents,
                                proposal_residual,
                                per_site_candidates=per_site_graph_candidates,
                                metric_tol=metric_tol,
                                max_projector_dim=max_projector_dim,
                            )
                        )
                    else:
                        proposals_by_kind[kind] = (
                            proposal_probe.residual_rank_proposals(
                                max_tie_rank,
                                proposal_residual,
                                max_projector_dim=max_projector_dim,
                                metric_tol=metric_tol,
                            )
                        )
                return proposals_by_kind[kind]

            scheduled_proposals = []
            proposal_indices = {kind: 0 for kind in proposal_order}
            while len(scheduled_proposals) < candidate_budget:
                added = False
                for kind in proposal_order:
                    index = proposal_indices[kind]
                    kind_proposals = proposals_for(kind)
                    if index >= len(kind_proposals):
                        continue
                    scheduled_proposals.append(
                        (kind, kind_proposals[index])
                    )
                    proposal_indices[kind] = index + 1
                    added = True
                    if len(scheduled_proposals) >= candidate_budget:
                        break
                if not added:
                    break
            trials = []
            branches = []
            for kind, proposal in scheduled_proposals:
                branch = exploration.copy()
                compound_rank = False
                try:
                    if kind == "graph":
                        branch.set_parent_set(
                            proposal["site"],
                            proposal["parents"],
                        )
                        if (
                            "rank" in proposal_kinds
                            and branch.tie_ranks[proposal["site"]] == 1
                            and branch.tie_ranks[proposal["site"]] < int(max_tie_rank)
                        ):
                            branch.grow_site_rank_from_residual(
                                proposal["site"],
                                proposal_residual,
                                max_projector_dim=max_projector_dim,
                                activation=rank_activation,
                                metric_tol=metric_tol,
                            )
                            branch.optimize_block(
                                proposal["site"],
                                "core",
                                metric_tol=exploration_metric_tol,
                            )
                            compound_rank = True
                    else:
                        branch.grow_site_rank_from_residual(
                            proposal["site"],
                            proposal_residual,
                            max_projector_dim=max_projector_dim,
                            activation=rank_activation,
                            metric_tol=metric_tol,
                        )
                        branch.optimize_block(
                            proposal["site"],
                            "core",
                            metric_tol=exploration_metric_tol,
                        )
                    if branch_sweeps:
                        branch.run(
                            nsweeps=branch_sweeps,
                            tol=0.0,
                            metric_tol=metric_tol,
                        )
                    branch_energy = float(branch.energy)
                    valid = np.isfinite(branch_energy)
                except (ValueError, np.linalg.LinAlgError):
                    branch_energy = np.inf
                    valid = False
                record = {
                    **proposal,
                    "compound_rank": compound_rank,
                    "energy": branch_energy,
                    "valid": valid,
                }
                trials.append(record)
                if valid:
                    branches.append((branch_energy, record, branch))
            chosen = None
            threshold_scale = max(1.0, abs(energy_before), abs(control_energy))
            threshold = energy_tol * threshold_scale
            if branches:
                best_energy, best_record, best_branch = min(
                    branches,
                    key=lambda item: (
                        item[0],
                        item[1]["kind"],
                        item[1]["site"],
                    ),
                )
                if best_energy < min(control_energy, energy_before) - threshold:
                    self._adopt(best_branch)
                    chosen = {
                        key: value
                        for key, value in best_record.items()
                        if key not in {"valid"}
                    }
            control_only = False
            if chosen is None and control_energy < energy_before - threshold:
                self._adopt(control)
                control_only = True
            energy_after = self.expectation()
            record = {
                "cycle": cycle,
                "energy_before": energy_before,
                "control_energy": control_energy,
                "energy": energy_after,
                "residual_norm": residual_norm,
                "accepted": chosen is not None or control_only,
                "structural_accepted": chosen is not None,
                "control_only": control_only,
                "proposal_order": proposal_order,
                "scheduled_kinds": tuple(kind for kind, _ in scheduled_proposals),
                "chosen": chosen,
                "trials": trials,
                "parent_sets": self.parent_sets,
                "tie_ranks": self.tie_ranks,
            }
            self.adaptive_history.append(record)
            if verbose:
                label = "control" if control_only else None if chosen is None else chosen["kind"]
                print(
                    f"residual-adaptive cycle={cycle:2d} E={energy_after: .12f} "
                    f"||r||={residual_norm:.3e} accepted={label}"
                )
            if chosen is None and not control_only:
                attempted_kinds = set(proposals_by_kind)
                stalled_kinds.update(attempted_kinds or proposal_kinds)
                if set(proposal_kinds) <= stalled_kinds:
                    break
            else:
                stalled_kinds.clear()
        self.energy = self.expectation()
        return self

    def expand_tie_ranks(
        self,
        tie_ranks,
        *,
        seed: int | None = None,
        cp_max_iter: int = 300,
        cp_tol: float = 1.0e-11,
    ) -> "CPTiedLETTA":
        """Grow CP ranks without changing the represented state initially."""
        new_ranks = _expanded_ints(tie_ranks, len(self.dims), name="tie_ranks")
        if any(new < old for new, old in zip(new_ranks, self.tie_ranks)):
            raise ValueError("tie ranks can only be increased.")
        rng = np.random.default_rng(seed)
        for site, (old_rank, new_rank) in enumerate(zip(self.tie_ranks, new_ranks)):
            if new_rank == old_rank:
                continue
            old_core = self.cores[site]
            core = np.zeros(old_core.shape[:2] + (new_rank,), dtype=old_core.dtype)
            core[:, :, :old_rank] = old_core
            self.cores[site] = core

            initializer = None
            if self._initial_dense_factors is not None:
                initializer = cp_als(
                    self._initial_dense_factors[site],
                    new_rank,
                    max_iter=cp_max_iter,
                    tol=cp_tol,
                    seed=None if seed is None else seed + site,
                )
            grown_factors = []
            for mode, (factor, physical_site) in enumerate(
                zip(self.physical_factors[site], self.physical_sites[site])
            ):
                grown = np.empty((new_rank, self.dims[physical_site]), dtype=factor.dtype)
                grown[:old_rank] = factor
                if initializer is None:
                    added = rng.normal(size=(new_rank - old_rank, self.dims[physical_site]))
                    if np.iscomplexobj(factor):
                        added = added + 1.0j * rng.normal(size=added.shape)
                    added /= np.maximum(np.linalg.norm(added, axis=1, keepdims=True), 1.0e-15)
                    grown[old_rank:] = added
                else:
                    grown[old_rank:] = initializer.factors[mode][:, old_rank:new_rank].T
                    zero_rows = np.linalg.norm(grown[old_rank:], axis=1) == 0.0
                    if np.any(zero_rows):
                        added = rng.normal(size=(np.count_nonzero(zero_rows), self.dims[physical_site]))
                        added /= np.maximum(np.linalg.norm(added, axis=1, keepdims=True), 1.0e-15)
                        added_rows = grown[old_rank:]
                        added_rows[zero_rows] = added
                grown_factors.append(grown)
            self.physical_factors[site] = grown_factors
        self.tie_ranks = new_ranks
        self.balance_gauges()
        self.energy = self.expectation()
        self.converged = False
        return self


__all__ = ["CPBlockUpdate", "CPTiedLETTA"]
