"""Dense-reference LETTA with arbitrary physical ties and virtual bonds."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse

from .core import _lowest_generalized_eigenpair
from .cp_tying import (
    _normalized_fidelity,
    _validated_dims,
    _validated_parent_sets,
)


_DENSE_PROJECTOR_WORK_MAX_BYTES = 384 * 1024**2


@dataclass(frozen=True)
class DenseSiteUpdate:
    """Diagnostic record for one unrestricted local tensor update."""

    site: int
    raw_dim: int
    metric_rank: int
    energy_before: float
    energy: float
    accepted: bool


class DenseTiedLETTA:
    r"""LETTA with unrestricted physical-context tensors.

    For physical parents ``P_k``, the represented wavefunction is

    .. math::

        \Psi(\mathbf{x}) = \sum_{a_1\ldots a_{L-1}}
        \prod_{k=0}^{L-1} T^{[k]}_{a_k a_{k+1}}
        (x_k, x_{P_k}),

    where the boundary virtual dimensions are one.  Unlike
    :class:`~pyqed.letta.cp_tying.CPTiedLETTA`, each ``T[k]`` is stored as a
    full tensor rather than a CP decomposition over its physical legs.

    This is a dense small-system reference implementation.  Local projectors
    have one row per many-body configuration, so it is intended for validation
    and modest benchmarks rather than large lattices.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        parent_sets,
        *,
        bond_dim: int = 1,
        tensors=None,
        seed: int | None = None,
    ):
        self.dims = _validated_dims(dims)
        self.parent_sets = _validated_parent_sets(self.dims, parent_sets)
        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")

        dimension = int(np.prod(self.dims))
        self.hamiltonian = (
            hamiltonian.tocsr()
            if issparse(hamiltonian)
            else np.asarray(hamiltonian)
        )
        if self.hamiltonian.shape != (dimension, dimension):
            raise ValueError("hamiltonian shape is inconsistent with dims.")

        self.rng = np.random.default_rng(seed)
        self._configs = np.asarray(list(np.ndindex(*self.dims)), dtype=np.intp)
        self.physical_sites = tuple(
            (site,) + parents for site, parents in enumerate(self.parent_sets)
        )
        bonds = self._bond_dims()
        shapes = tuple(
            (bonds[site], bonds[site + 1])
            + tuple(self.dims[index] for index in physical_sites)
            for site, physical_sites in enumerate(self.physical_sites)
        )
        parameter_dtype = np.result_type(self.hamiltonian.dtype, np.float64)
        if tensors is None:
            self.tensors = []
            for shape in shapes:
                tensor = self.rng.normal(size=shape) / np.sqrt(np.prod(shape))
                self.tensors.append(tensor.astype(parameter_dtype))
        else:
            if len(tensors) != len(self.dims):
                raise ValueError("tensors must contain one entry per site.")
            self.tensors = [
                np.asarray(
                    tensor,
                    dtype=np.result_type(np.asarray(tensor).dtype, parameter_dtype),
                ).copy()
                for tensor in tensors
            ]
            for site, (tensor, shape) in enumerate(zip(self.tensors, shapes)):
                if tensor.shape != shape:
                    raise ValueError(f"tensor {site} shape must be {shape}.")

        self._physical_columns = tuple(
            np.ravel_multi_index(
                tuple(self._configs[:, index] for index in physical_sites),
                tuple(self.dims[index] for index in physical_sites),
            ).astype(np.int32, copy=False)
            for physical_sites in self.physical_sites
        )
        self.history: list[dict] = []
        self.energy: float | None = None
        self.converged = False
        self.balance_gauges()
        self.energy = self.expectation()

    def _bond_dims(self) -> tuple[int, ...]:
        return (
            (1,)
            + (self.bond_dim,) * max(0, len(self.dims) - 1)
            + (1,)
        )

    @classmethod
    def from_cp(cls, state) -> "DenseTiedLETTA":
        """Expand a CP-tied LETTA exactly into unrestricted local tensors."""
        tensors = []
        for core, factors in zip(state.cores, state.physical_factors):
            shape = core.shape[:2] + tuple(factor.shape[1] for factor in factors)
            tensor = np.zeros(shape, dtype=np.result_type(core.dtype, *[f.dtype for f in factors]))
            for component in range(core.shape[2]):
                term = core[:, :, component]
                for factor in factors:
                    term = term[..., None] * factor[component]
                tensor += term
            tensors.append(tensor)
        result = cls(
            state.hamiltonian,
            state.dims,
            state.parent_sets,
            bond_dim=state.bond_dim,
            tensors=tensors,
        )
        result.rng.bit_generator.state = deepcopy(state.rng.bit_generator.state)
        return result

    def copy(self) -> "DenseTiedLETTA":
        result = type(self)(
            self.hamiltonian,
            self.dims,
            self.parent_sets,
            bond_dim=self.bond_dim,
            tensors=[tensor.copy() for tensor in self.tensors],
        )
        result.history = list(self.history)
        result.energy = self.energy
        result.converged = self.converged
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return result

    @property
    def nparameters(self) -> int:
        return int(sum(tensor.size for tensor in self.tensors))

    def _site_transfer(self, site: int) -> np.ndarray:
        tensor = self.tensors[site]
        left_dim, right_dim = tensor.shape[:2]
        flat = tensor.reshape(left_dim, right_dim, -1)
        return flat[:, :, self._physical_columns[site]].transpose(2, 0, 1)

    def state_vector(self, *, normalize: bool = False) -> np.ndarray:
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
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
                raise ValueError("dense-tied LETTA state is zero or nonfinite.")
            vector = vector / norm
        return vector

    def norm(self) -> float:
        vector = self.state_vector()
        return float(np.vdot(vector, vector).real)

    def normalize(self) -> "DenseTiedLETTA":
        norm = np.sqrt(self.norm())
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("dense-tied LETTA state cannot be normalized.")
        self.tensors[0] /= norm
        return self

    def balance_gauges(
        self,
        *,
        state_norm: float | None = None,
    ) -> "DenseTiedLETTA":
        """Remove harmless scalar imbalances and normalize the full state."""
        tensor_norms = np.asarray(
            [float(np.linalg.norm(tensor)) for tensor in self.tensors]
        )
        if np.any(~np.isfinite(tensor_norms)) or np.any(tensor_norms <= 0.0):
            bad_site = int(
                np.flatnonzero(
                    (~np.isfinite(tensor_norms)) | (tensor_norms <= 0.0)
                )[0]
            )
            raise ValueError(f"tensor {bad_site} is zero or nonfinite.")
        if state_norm is None:
            state_norm = np.sqrt(self.norm())
        state_norm = float(state_norm)
        if not np.isfinite(state_norm) or state_norm <= 0.0:
            raise ValueError("dense-tied LETTA state cannot be normalized.")
        common_scale = float(
            np.exp(
                (np.sum(np.log(tensor_norms)) - np.log(state_norm))
                / len(self.tensors)
            )
        )
        for site, (tensor, tensor_norm) in enumerate(
            zip(self.tensors, tensor_norms)
        ):
            if not np.isfinite(tensor_norm) or tensor_norm <= 0.0:
                raise ValueError(f"tensor {site} is zero or nonfinite.")
            self.tensors[site] = tensor * (common_scale / tensor_norm)
        return self

    def expectation(self) -> float:
        vector = self.state_vector()
        norm = np.vdot(vector, vector)
        if abs(norm) <= 1.0e-28:
            raise ValueError("dense-tied LETTA state is numerically zero.")
        return float(
            np.real(np.vdot(vector, self.hamiltonian @ vector) / norm)
        )

    def fidelity(self, state) -> float:
        return _normalized_fidelity(self.state_vector(), state)

    def energy_residual(self) -> tuple[float, np.ndarray, float]:
        vector = self.state_vector(normalize=True)
        h_vector = self.hamiltonian @ vector
        energy = float(np.real(np.vdot(vector, h_vector)))
        residual = h_vector - energy * vector
        return energy, residual, float(np.linalg.norm(residual))

    def local_projector(
        self,
        site: int,
        *,
        left_environment=None,
        right_environment=None,
        sparse: bool = False,
    ):
        """Map one unrestricted local tensor to the full state vector."""
        site = int(site)
        if site < 0 or site >= len(self.dims):
            raise IndexError("site is out of range.")
        tensor = self.tensors[site]
        left_dim, right_dim = tensor.shape[:2]
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        if left_environment is None:
            left = np.ones((len(self._configs), 1), dtype=dtype)
            for other_site in range(site):
                left = np.einsum(
                    "ca,cab->cb",
                    left,
                    self._site_transfer(other_site),
                    optimize=True,
                )
        else:
            left = np.asarray(left_environment)
            if left.shape != (len(self._configs), left_dim):
                raise ValueError("left_environment has the wrong shape.")
        if right_environment is None:
            right = np.ones((len(self._configs), 1), dtype=dtype)
            for other_site in range(len(self.dims) - 1, site, -1):
                right = np.einsum(
                    "cab,cb->ca",
                    self._site_transfer(other_site),
                    right,
                    optimize=True,
                )
        else:
            right = np.asarray(right_environment)
            if right.shape != (len(self._configs), right_dim):
                raise ValueError("right_environment has the wrong shape.")

        physical_dim = int(np.prod(tensor.shape[2:]))
        width = tensor.size
        coefficients = (
            left[:, :, None] * right[:, None, :]
        ).reshape(len(self._configs), left_dim * right_dim)
        columns = (
            np.arange(left_dim * right_dim, dtype=np.int32)[None, :]
            * physical_dim
            + self._physical_columns[site][:, None]
        )
        if sparse:
            entries_per_row = left_dim * right_dim
            indptr = np.arange(
                0,
                (len(self._configs) + 1) * entries_per_row,
                entries_per_row,
                dtype=np.int32,
            )
            return csr_matrix(
                (
                    coefficients.reshape(-1),
                    columns.reshape(-1),
                    indptr,
                ),
                shape=(len(self._configs), width),
            )
        projector = np.zeros((len(self._configs), width), dtype=dtype)
        projector[np.arange(len(self._configs))[:, None], columns] = coefficients
        return projector

    def optimize_site(
        self,
        site: int,
        *,
        metric_tol: float = 1.0e-12,
        left_environment=None,
        right_environment=None,
        energy_before: float | None = None,
    ) -> DenseSiteUpdate:
        """Minimize the energy over one full local tensor."""
        site = int(site)
        if site < 0 or site >= len(self.dims):
            raise IndexError("site is out of range.")
        metric_tol = float(metric_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")

        old_tensor = self.tensors[site].copy()
        if energy_before is None:
            energy_before = self.expectation()
        energy_before = float(energy_before)
        projector_dtype = np.result_type(
            *[tensor.dtype for tensor in self.tensors]
        )
        projector_bytes = (
            len(self._configs)
            * old_tensor.size
            * np.dtype(projector_dtype).itemsize
        )
        dense_work_bytes = 3 * projector_bytes
        use_sparse = (
            issparse(self.hamiltonian)
            and dense_work_bytes > _DENSE_PROJECTOR_WORK_MAX_BYTES
        )
        projector = self.local_projector(
            site,
            left_environment=left_environment,
            right_environment=right_environment,
            sparse=use_sparse,
        )
        if issparse(projector):
            adjoint = projector.getH()
            metric = (adjoint @ projector).toarray()
            h_projector = self.hamiltonian @ projector
            effective = (adjoint @ h_projector).toarray()
        else:
            metric = projector.T.conj() @ projector
            h_projector = self.hamiltonian @ projector
            effective = projector.T.conj() @ h_projector
        metric = 0.5 * (metric + metric.T.conj())
        effective = 0.5 * (effective + effective.T.conj())
        eigenvalues = np.linalg.eigvalsh(metric)
        scale = max(
            float(np.linalg.norm(metric, ord=np.inf)),
            np.finfo(float).tiny,
        )
        metric_rank = int(np.count_nonzero(eigenvalues > metric_tol * scale))

        accepted = False
        energy_after = energy_before
        try:
            _local_energy, vector = _lowest_generalized_eigenpair(
                effective,
                metric,
                metric_tol=metric_tol,
            )
            denominator = np.vdot(vector, metric @ vector)
            if abs(denominator) <= np.finfo(float).tiny:
                raise ValueError("local tensor has zero metric norm.")
            energy_after = float(
                np.real(
                    np.vdot(vector, effective @ vector) / denominator
                )
            )
            tolerance = (
                128.0
                * np.finfo(float).eps
                * max(1.0, abs(energy_before))
            )
            accepted = (
                np.isfinite(energy_after)
                and energy_after <= energy_before + tolerance
            )
            if accepted:
                self.tensors[site][...] = vector.reshape(old_tensor.shape)
        except (ValueError, np.linalg.LinAlgError):
            accepted = False
        if not accepted:
            self.tensors[site][...] = old_tensor
            energy_after = energy_before
        self.energy = float(energy_after)
        return DenseSiteUpdate(
            site=site,
            raw_dim=old_tensor.size,
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
    ) -> "DenseTiedLETTA":
        """Optimize all unrestricted tensors in alternating sweep order."""
        nsweeps = int(nsweeps)
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        previous = self.expectation()
        self.energy = previous
        self.history = []
        self.converged = False
        for sweep in range(nsweeps):
            dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
            if sweep % 2 == 0:
                right_environments = [None] * (len(self.dims) + 1)
                right_environments[-1] = np.ones(
                    (len(self._configs), 1), dtype=dtype
                )
                for site in range(len(self.dims) - 1, -1, -1):
                    right_environments[site] = np.einsum(
                        "cab,cb->ca",
                        self._site_transfer(site),
                        right_environments[site + 1],
                        optimize=True,
                    )
                left = np.ones((len(self._configs), 1), dtype=dtype)
                updates = []
                for site in range(len(self.dims)):
                    updates.append(
                        self.optimize_site(
                            site,
                            metric_tol=metric_tol,
                            left_environment=left,
                            right_environment=right_environments[site + 1],
                            energy_before=self.energy,
                        )
                    )
                    left = np.einsum(
                        "ca,cab->cb",
                        left,
                        self._site_transfer(site),
                        optimize=True,
                    )
            else:
                left_environments = [None] * (len(self.dims) + 1)
                left_environments[0] = np.ones(
                    (len(self._configs), 1), dtype=dtype
                )
                for site in range(len(self.dims)):
                    left_environments[site + 1] = np.einsum(
                        "ca,cab->cb",
                        left_environments[site],
                        self._site_transfer(site),
                        optimize=True,
                    )
                right = np.ones((len(self._configs), 1), dtype=dtype)
                updates = []
                for site in range(len(self.dims) - 1, -1, -1):
                    updates.append(
                        self.optimize_site(
                            site,
                            metric_tol=metric_tol,
                            left_environment=left_environments[site],
                            right_environment=right,
                            energy_before=self.energy,
                        )
                    )
                    right = np.einsum(
                        "cab,cb->ca",
                        self._site_transfer(site),
                        right,
                        optimize=True,
                    )
            vector = self.state_vector()
            norm = np.vdot(vector, vector)
            if abs(norm) <= np.finfo(float).tiny:
                raise ValueError("dense-tied LETTA state is numerically zero.")
            energy = float(
                np.real(np.vdot(vector, self.hamiltonian @ vector) / norm)
            )
            self.balance_gauges(state_norm=np.sqrt(float(np.real(norm))))
            delta = abs(energy - previous)
            self.energy = energy
            self.history.append(
                {
                    "sweep": sweep,
                    "energy": energy,
                    "delta_energy": delta,
                    "accepted_sites": sum(update.accepted for update in updates),
                    "updates": updates,
                }
            )
            if verbose:
                print(
                    f"dense-LETTA sweep={sweep:2d} E={energy: .12f} "
                    f"dE={delta:.3e} "
                    f"accepted={sum(update.accepted for update in updates)}",
                    flush=True,
                )
            if delta <= tol:
                self.converged = True
                break
            previous = energy
        return self


__all__ = ["DenseSiteUpdate", "DenseTiedLETTA"]
