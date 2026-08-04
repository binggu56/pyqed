"""Exact variation-after-projection for unrestricted graph LETTA states."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from .frontier_tying import FrontierTiedLETTA
from .local_terms import (
    LocalMPOProduct,
    local_charges_from_sites,
    fixed_charge_projector_mpo,
    validate_charge_conservation,
)
from .renormalized_frontier import renormalized_operator_mpo


def _charge_tuple(value) -> tuple[int, ...]:
    if hasattr(value, "charge"):
        value = value.charge
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return tuple(int(component) for component in value)
    return (int(value),)


def _normalized_projection_data(dims, local_charges, target, left_boundary):
    dims = tuple(int(dim) for dim in dims)
    local_charges = tuple(
        tuple(_charge_tuple(charge) for charge in site_charges)
        for site_charges in local_charges
    )
    if len(local_charges) != len(dims):
        raise ValueError("local_charges must contain one entry per physical site.")
    if any(
        len(site_charges) != dim
        for site_charges, dim in zip(local_charges, dims)
    ):
        raise ValueError(
            "each local_charges entry must contain one charge per local state."
        )
    target = _charge_tuple(target)
    if any(
        len(charge) != len(target)
        for site_charges in local_charges
        for charge in site_charges
    ):
        raise ValueError("all local charges must have the target charge rank.")
    if left_boundary is None:
        left_boundary = tuple(0 for _ in target)
    left_boundary = _charge_tuple(left_boundary)
    if len(left_boundary) != len(target):
        raise ValueError("left_boundary and target must have the same charge rank.")
    return local_charges, target, left_boundary


@dataclass(frozen=True)
class SectorProjection:
    """Fixed, zero-parameter Abelian projector attached to a LETTA state."""

    local_charges: tuple[tuple[tuple[int, ...], ...], ...]
    target: tuple[int, ...]
    left_boundary: tuple[int, ...]
    mpo_bond_dims: tuple[int, ...]

    @property
    def max_mpo_bond(self) -> int:
        return max(self.mpo_bond_dims)


class SectorProjectedLETTA(FrontierTiedLETTA):
    r"""Unrestricted graph LETTA optimized after exact charge projection.

    For the ordinary unrestricted LETTA state ``|Psi(A)>``, this class
    represents

    .. math::

        |\Phi_Q(A)\rangle = P_Q |\Psi(A)\rangle .

    Every entry of every LETTA tensor remains variational.  ``P_Q`` is a
    fixed finite-state MPO that counts each unique physical site once; it does
    not allocate the LETTA bond dimension among charge sectors and does not
    impose local tensor masks.
    """

    _preserve_pair_metric_null_components = True

    def __init__(
        self,
        hamiltonian,
        dims,
        parent_sets,
        *,
        local_charges=None,
        target=0,
        left_boundary=None,
        conservation_atol=None,
        **kwargs,
    ):
        if local_charges is None:
            local_charges = local_charges_from_sites(
                hamiltonian.sites,
                require=True,
            )
        local_charges, target, left_boundary = _normalized_projection_data(
            dims,
            local_charges,
            target,
            left_boundary,
        )
        if kwargs.get("tt_norm_backend", "exact") != "exact":
            raise ValueError(
                "SectorProjectedLETTA currently requires tt_norm_backend='exact' "
                "so metric-null tensor coordinates can be retained."
            )
        validate_charge_conservation(
            hamiltonian,
            local_charges,
            atol=conservation_atol,
        )
        projector = fixed_charge_projector_mpo(
            dims,
            local_charges,
            target,
            left_boundary=left_boundary,
        )
        if all(len(term.sites) <= 2 for term in hamiltonian.terms):
            hamiltonian_mpo, hamiltonian_mpo_diagnostics = (
                renormalized_operator_mpo(
                    hamiltonian,
                    local_qns=local_charges,
                )
            )
        else:
            hamiltonian_mpo = hamiltonian.to_mpo()
            hamiltonian_mpo_diagnostics = {
                "bond_dims": hamiltonian_mpo.bond_dims,
                "max_bond_dim": max(hamiltonian_mpo.bond_dims),
                "representation": "finite_state_local_terms",
            }
        objective = LocalMPOProduct(hamiltonian_mpo, projector)
        kwargs.setdefault("frontier_backend", "identity_block")
        if any(
            name in kwargs
            for name in (
                "_norm_mpo",
                "_objective_mpo",
                "_objective_is_hermitian",
            )
        ):
            raise TypeError("projected operator MPOs are constructed internally.")
        self.local_charges = local_charges
        self.target_charge = target
        self.left_boundary_charge = left_boundary
        self.conservation_atol = conservation_atol
        self.projected_hamiltonian_mpo_diagnostics = (
            hamiltonian_mpo_diagnostics
        )
        super().__init__(
            hamiltonian,
            dims,
            parent_sets,
            _norm_mpo=projector,
            _objective_mpo=objective,
            _objective_is_hermitian=True,
            **kwargs,
        )
        self.symmetry = "u1"
        self.projection = SectorProjection(
            local_charges=local_charges,
            target=target,
            left_boundary=left_boundary,
            mpo_bond_dims=projector.bond_dims,
        )

    @property
    def nparameters(self) -> int:
        """Number of unrestricted LETTA tensor parameters."""
        return int(sum(tensor.size for tensor in self.tensors))

    @property
    def dense_nparameters(self) -> int:
        return self.nparameters

    def local_support_sizes(self):
        """Report that every local tensor coordinate remains variational."""
        return tuple((tensor.size, tensor.size) for tensor in self.tensors)

    def raw_state_vector(self, *, normalize=False):
        """Build the unprojected LETTA vector for small-system validation."""
        return super().state_vector(normalize=normalize)

    def state_vector(self, *, normalize=False):
        """Build ``P_Q|Psi>`` explicitly for small-system validation only."""
        vector = self.raw_state_vector(normalize=False)
        mask = np.zeros(vector.size, dtype=bool)
        for row, configuration in enumerate(np.ndindex(*self.dims)):
            charge = list(self.left_boundary_charge)
            for site, physical in enumerate(configuration):
                for component, value in enumerate(
                    self.local_charges[site][physical]
                ):
                    charge[component] += value
            mask[row] = tuple(charge) == self.target_charge
        vector = np.where(mask, vector, 0)
        if normalize:
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("the projected LETTA state is zero or nonfinite.")
            vector = vector / norm
        return vector

    def _complete_local_solution(
        self,
        site,
        old_vector,
        vector,
        *,
        metric,
        metric_tol,
        environment,
    ):
        """Retain the incumbent tensor component invisible to ``P_Q``."""
        if metric is None:
            metric = self.local_metric(site, environment=environment)
        metric = 0.5 * (np.asarray(metric) + np.asarray(metric).T.conj())
        values, vectors = np.linalg.eigh(metric)
        scale = max(
            float(np.max(np.abs(values), initial=0.0)),
            np.finfo(float).tiny,
        )
        active = values > float(metric_tol) * scale
        if not np.any(active):
            raise ValueError("projected local overlap metric is numerically singular.")
        active_vectors = vectors[:, active]
        old_vector = np.asarray(old_vector).reshape(-1)
        vector = np.asarray(vector).reshape(-1)
        old_active = active_vectors @ (active_vectors.T.conj() @ old_vector)
        new_active = active_vectors @ (active_vectors.T.conj() @ vector)
        overlap = np.vdot(old_active, metric @ new_active)
        if abs(overlap) > np.finfo(float).tiny:
            new_active *= overlap.conjugate() / abs(overlap)
        return new_active + (old_vector - old_active)

    @staticmethod
    def _complete_from_metric_eigenspaces(
        old_vector,
        vector,
        eigenspaces,
        metric_action,
    ):
        old_vector = np.asarray(old_vector).reshape(-1)
        vector = np.asarray(vector).reshape(-1)
        eigenspaces = tuple(eigenspaces)
        active_rank = sum(basis.shape[1] for _indices, basis in eigenspaces)
        if active_rank == old_vector.size:
            return None
        if active_rank == 0:
            raise ValueError("projected pair overlap metric is numerically singular.")
        old_active = np.zeros_like(
            old_vector,
            dtype=np.result_type(old_vector, vector),
        )
        new_active = np.zeros_like(old_active)
        for indices, basis in eigenspaces:
            old_block = old_vector[indices]
            new_block = vector[indices]
            old_active[indices] = basis @ (basis.T.conj() @ old_block)
            new_active[indices] = basis @ (basis.T.conj() @ new_block)
        overlap = np.vdot(old_active, metric_action(new_active))
        if abs(overlap) > np.finfo(float).tiny:
            new_active *= overlap.conjugate() / abs(overlap)
        return new_active + (old_vector - old_active)

    @classmethod
    def _complete_from_dense_metric(
        cls,
        old_vector,
        vector,
        metric,
        *,
        metric_eigensystem=None,
    ):
        metric = 0.5 * (np.asarray(metric) + np.asarray(metric).T.conj())
        if metric_eigensystem is None:
            values, vectors = np.linalg.eigh(metric)
        else:
            values, vectors = metric_eigensystem
            values = np.asarray(values)
            vectors = np.asarray(vectors)
            if values.shape != (metric.shape[0],) or vectors.shape != metric.shape:
                raise ValueError(
                    "pair metric eigensystem has incompatible shapes."
                )
        scale = max(
            float(np.linalg.norm(metric, ord=np.inf)),
            float(np.max(np.abs(values), initial=0.0)),
            np.finfo(float).tiny,
        )
        active = values > 64.0 * np.finfo(float).eps * scale
        indices = np.arange(metric.shape[0], dtype=np.intp)
        return cls._complete_from_metric_eigenspaces(
            old_vector,
            vector,
            ((indices, vectors[:, active]),),
            lambda trial: metric @ trial,
        )

    @classmethod
    def _complete_from_block_metric(cls, old_vector, vector, metric_operator):
        layout = metric_operator.layout
        scale = max(
            (
                float(np.linalg.norm(block, ord=np.inf))
                for block in metric_operator.blocks.values()
            ),
            default=0.0,
        )
        scale = max(scale, np.finfo(float).tiny)
        floor = 64.0 * np.finfo(float).eps * scale
        eigenspaces = []
        for block, indices in enumerate(layout.block_indices):
            metric = metric_operator.blocks[(block, block)]
            metric = 0.5 * (metric + metric.T.conj())
            values, vectors = np.linalg.eigh(metric)
            eigenspaces.append((indices, vectors[:, values > floor]))
        return cls._complete_from_metric_eigenspaces(
            old_vector,
            vector,
            eigenspaces,
            metric_operator.matvec,
        )

    @staticmethod
    def _complete_from_metric_action(old_vector, vector, metric_action):
        old_vector = np.asarray(old_vector).reshape(-1)
        vector = np.asarray(vector).reshape(-1)
        size = old_vector.size
        dtype = np.dtype(np.result_type(old_vector, vector))

        def action(trial):
            return np.asarray(metric_action(trial)).reshape(-1)

        operator = LinearOperator(
            (size, size),
            matvec=action,
            rmatvec=action,
            dtype=dtype,
        )
        tolerance = 64.0 * np.finfo(float).eps

        def active_projection(source):
            right_hand_side = action(source)
            scale = max(float(np.linalg.norm(right_hand_side)), np.finfo(float).tiny)
            if scale <= np.finfo(float).tiny:
                return np.zeros_like(source, dtype=dtype), True
            result = lsmr(
                operator,
                right_hand_side,
                atol=tolerance,
                btol=tolerance,
                maxiter=max(32, 4 * size),
            )
            projected = np.asarray(result[0], dtype=dtype)
            relative_residual = float(
                np.linalg.norm(action(projected) - right_hand_side) / scale
            )
            return projected, relative_residual <= 1.0e-10

        old_active, old_converged = active_projection(old_vector)
        new_active, new_converged = active_projection(vector)
        if not old_converged or not new_converged:
            return old_vector.copy()
        old_null = old_vector - old_active
        new_null = vector - new_active
        null_scale = max(
            float(np.linalg.norm(old_vector)),
            float(np.linalg.norm(vector)),
            1.0,
        )
        if max(np.linalg.norm(old_null), np.linalg.norm(new_null)) <= (
            2048.0 * np.finfo(float).eps * null_scale
        ):
            return None
        overlap = np.vdot(old_active, action(new_active))
        if abs(overlap) > np.finfo(float).tiny:
            new_active *= overlap.conjugate() / abs(overlap)
        return new_active + old_null

    def _complete_pair_metric_solution(
        self,
        old_vector,
        vector,
        *,
        metric=None,
        metric_operator=None,
        metric_action=None,
        metric_eigensystem=None,
    ):
        """Retain incumbent coordinates invisible to the exact projector."""
        if metric is not None:
            return self._complete_from_dense_metric(
                old_vector,
                vector,
                metric,
                metric_eigensystem=metric_eigensystem,
            )
        if metric_operator is not None:
            return self._complete_from_block_metric(
                old_vector,
                vector,
                metric_operator,
            )
        if metric_action is None:
            raise ValueError("a pair metric representation is required.")
        return self._complete_from_metric_action(
            old_vector,
            vector,
            metric_action,
        )

    @classmethod
    def from_unrestricted(
        cls,
        state: FrontierTiedLETTA,
        *,
        local_charges=None,
        target=0,
        left_boundary=None,
        frontier_backend=None,
        **kwargs,
    ):
        """Project an existing unrestricted LETTA without changing its tensors."""
        if type(state) is not FrontierTiedLETTA:
            raise TypeError("state must be an unrestricted FrontierTiedLETTA.")
        if local_charges is None:
            local_charges = local_charges_from_sites(
                state.hamiltonian.sites,
                require=True,
            )
        kwargs.setdefault(
            "frontier_backend",
            (
                "identity_block"
                if frontier_backend is None
                else frontier_backend
            ),
        )
        kwargs.setdefault("_balance_initial_gauges", False)
        result = cls(
            state.hamiltonian,
            state.dims,
            state.parent_sets,
            local_charges=local_charges,
            target=target,
            left_boundary=left_boundary,
            bond_dims=state.bond_dims,
            tensors=[tensor.copy() for tensor in state.tensors],
            path_optimizer=state.path_optimizer,
            **kwargs,
        )
        result.rng.bit_generator.state = deepcopy(state.rng.bit_generator.state)
        return result

    def copy(self):
        result = type(self)(
            self.hamiltonian,
            self.dims,
            self.parent_sets,
            local_charges=self.local_charges,
            target=self.target_charge,
            left_boundary=self.left_boundary_charge,
            conservation_atol=self.conservation_atol,
            bond_dim=self.bond_dim,
            bond_dims=self.bond_dims,
            tensors=[tensor.copy() for tensor in self.tensors],
            frontier_backend=self.frontier_backend,
            path_optimizer=self.path_optimizer,
            local_backend=self.local_backend,
            local_rank=self.local_options["rank"],
            local_rtol=self.local_options["rtol"],
            local_atol=self.local_options["atol"],
            max_rank=self.tt_options["max_rank"],
            rtol=self.tt_options["rtol"],
            atol=self.tt_options["atol"],
            transfer_max_rank=self.tt_options["transfer_max_rank"],
            transfer_rtol=self.tt_options["transfer_rtol"],
            transfer_atol=self.tt_options["transfer_atol"],
            tt_absorption=self.tt_options["absorption"],
            tt_norm_backend=self.tt_norm_backend,
            tt_hermitize=self.tt_hermitize,
            _balance_initial_gauges=False,
        )
        result.history = list(self.history)
        result.converged = self.converged
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return result


# Backward-compatible shorter alias names.
ProjectedLETTA = SectorProjectedLETTA


__all__ = ["SectorProjectedLETTA", "ProjectedLETTA", "SectorProjection"]
