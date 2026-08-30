"""Frontier contraction for matrix-valued conditional tie factors."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from .conditional_tying import (
    ConditionalTTUpdate,
    _bond_dimensions,
    _materialize_conditional_factors,
    _random_array,
)
from .core import _lowest_generalized_eigenpair
from .cp_tying import _expanded_ints, _validated_parent_sets
from .frontier_abelian import AbelianFrontierTiedLETTA, FrontierAbelianLayout
from .frontier_tying import FrontierSiteEnvironment, FrontierTiedLETTA


class _ConditionalTensorView:
    """Read-only lazy sequence of materialized conditional tensors."""

    def __init__(self, state):
        self.state = state

    def __len__(self):
        return len(self.state.dims)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.state.materialize_tensor(site) for site in range(*index.indices(len(self)))]
        return self.state.materialize_tensor(index)

    def __iter__(self):
        for site in range(len(self)):
            yield self.state.materialize_tensor(site)


def _ranked_charge_labels(labels, dimension):
    """Choose a compact internal charge layout from a backbone bond."""
    labels = tuple(tuple(label) for label in labels)
    if dimension >= len(labels):
        selected = list(labels)
        while len(selected) < dimension:
            selected.append(labels[len(selected) % len(labels)])
        return tuple(selected)
    counts = Counter(tuple(label) for label in labels)
    ranked = sorted(
        counts,
        key=lambda label: (
            -counts[label],
            sum(abs(value) for value in label),
            label,
        ),
    )
    selected = ranked[:dimension]
    while len(selected) < dimension:
        selected.append(ranked[len(selected) % len(ranked)])
    return tuple(selected)


class _ConditionalTTFrontierMixin:
    """Factor storage and factor-level Rayleigh--Ritz sweeps."""

    def __init__(
        self,
        hamiltonian,
        parent_sets,
        *legacy_parent_sets,
        chi=1,
        factors=None,
        init="mps",
        parent_group_size=1,
        **kwargs,
    ):
        if legacy_parent_sets:
            if len(legacy_parent_sets) > 1:
                raise TypeError("only one legacy parent_sets argument is accepted.")
            parent_sets = legacy_parent_sets[0]
        dims = hamiltonian.dims
        parents = _validated_parent_sets(dims, parent_sets)
        self.parent_group_size = int(parent_group_size)
        if self.parent_group_size < 1:
            raise ValueError("parent_group_size must be positive.")
        self.chi = _expanded_ints(chi, len(dims), name="chi")
        layout = kwargs.get("abelian_layout")
        if layout is not None and not isinstance(layout, FrontierAbelianLayout):
            raise TypeError("abelian_layout must be a FrontierAbelianLayout.")
        if "tensors" in kwargs:
            raise TypeError("supply conditional factors instead of dense tensors.")

        if layout is not None:
            bonds = layout.bond_dims
        else:
            bonds = _bond_dimensions(
                len(dims),
                kwargs.get("bond_dims", kwargs.get("bond_dim", 1)),
            )
        self._conditional_bonds = tuple(bonds)
        self._conditional_dims = tuple(dims)
        self._conditional_groups = tuple(
            (site,) + site_parents
            for site, site_parents in enumerate(parents)
        )
        self._conditional_parent_groups = tuple(
            tuple(
                tuple(site_parents[start : start + self.parent_group_size])
                for start in range(0, len(site_parents), self.parent_group_size)
            )
            for site_parents in parents
        )
        self.internal_qns = self._internal_charge_layout(layout)
        init = str(init).lower().replace("-", "_")
        if init not in {"mps", "random"}:
            raise ValueError("init must be 'mps' or 'random'.")
        self.factor_init = init
        dtype = np.result_type(hamiltonian.dtype, np.complex128)
        factor_rng = np.random.default_rng(kwargs.get("seed"))
        if factors is None:
            self.factors = [
                self._random_factors(site, factor_rng, dtype)
                for site in range(len(dims))
            ]
            if init == "mps":
                self._initialize_mps_controls(layout)
        else:
            if len(factors) != len(dims):
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
            self._validate_conditional_factors(dims)
        self.factor_masks = self._factor_support_masks(layout)
        self._apply_factor_masks()

        initial = [self.materialize_tensor(site) for site in range(len(dims))]
        super().__init__(
            hamiltonian,
            parents,
            tensors=initial,
            **kwargs,
        )

        # The base constructor applies only state-preserving scalar balancing.
        # Transfer those scalars back into B before discarding persistent dense
        # local tensors.
        balanced = self.tensors
        for site, (before, after) in enumerate(zip(initial, balanced)):
            denominator = np.vdot(before, before)
            if abs(denominator) <= np.finfo(float).tiny:
                raise ValueError(f"conditional tensor {site} is numerically zero.")
            scale = np.vdot(before, after) / denominator
            residual = np.linalg.norm(after - scale * before)
            tolerance = 1.0e-11 * max(float(np.linalg.norm(after)), 1.0)
            if residual > tolerance:
                raise ValueError(
                    "symmetry projection changed a conditional tensor; "
                    "its factor charge support is inconsistent."
                )
            self.factors[site][0] *= scale
        self.tensors = _ConditionalTensorView(self)
        self.energy = self.expectation()

    def _internal_charge_layout(self, layout):
        if layout is None:
            return (None,) * len(self._conditional_groups)
        return tuple(
            (
                None
                if len(group) == 1
                else _ranked_charge_labels(
                    layout.bond_qns[site + 1],
                    self.chi[site],
                )
            )
            for site, group in enumerate(self._conditional_groups)
        )

    def _random_factors(self, site, rng, dtype):
        dims = self._conditional_dims
        parent_groups = self._conditional_parent_groups[site]
        left_dim, right_dim = self._conditional_bonds[site : site + 2]
        local_dim = dims[site]
        if not parent_groups:
            return [_random_array(rng, (left_dim, local_dim, right_dim), dtype)]
        rank = self.chi[site]
        result = [_random_array(rng, (left_dim, local_dim, rank), dtype)]
        for parent_group in parent_groups[:-1]:
            result.append(
                _random_array(
                    rng,
                    (rank, local_dim, *(dims[parent] for parent in parent_group), rank),
                    dtype,
                )
            )
        parent_group = parent_groups[-1]
        result.append(
            _random_array(
                rng,
                (
                    rank,
                    local_dim,
                    *(dims[parent] for parent in parent_group),
                    right_dim,
                ),
                dtype,
            )
        )
        return result

    def _validate_conditional_factors(self, dims):
        for site, (parent_groups, factors) in enumerate(
            zip(self._conditional_parent_groups, self.factors)
        ):
            if len(factors) != 1 + len(parent_groups):
                raise ValueError(
                    f"site {site} must contain one B factor and one factor per "
                    "parent group."
                )
            left_dim, right_dim = self._conditional_bonds[site : site + 2]
            local_dim = dims[site]
            if not parent_groups:
                if factors[0].shape != (left_dim, local_dim, right_dim):
                    raise ValueError(f"untied B factor at site {site} has an invalid shape.")
                continue
            if factors[0].shape != (left_dim, local_dim, self.chi[site]):
                raise ValueError(f"B factor at site {site} has an invalid shape.")
            for mode, (parent_group, factor) in enumerate(
                zip(parent_groups, factors[1:]), start=1
            ):
                right = right_dim if mode == len(parent_groups) else self.chi[site]
                expected = (
                    self.chi[site],
                    local_dim,
                    *(dims[parent] for parent in parent_group),
                    right,
                )
                if factor.shape != expected:
                    raise ValueError(
                        f"C factor {mode} at site {site} must have shape {expected}."
                    )

    def _initialize_mps_controls(self, layout):
        """Initialize tied C factors as neutral identity transfers."""
        for site, factors in enumerate(self.factors):
            if len(factors) == 1:
                continue
            for factor in factors[1:-1]:
                factor.fill(0)
                for gamma in range(factor.shape[0]):
                    index = (gamma,) + (slice(None),) * (factor.ndim - 2) + (gamma,)
                    factor[index] = 1
            last = factors[-1]
            last.fill(0)
            if layout is None:
                for gamma in range(last.shape[0]):
                    index = (gamma,) + (slice(None),) * (last.ndim - 2)
                    index += (gamma % last.shape[-1],)
                    last[index] = 1
                continue
            used = Counter()
            right_labels = layout.bond_qns[site + 1]
            by_charge = {
                charge: tuple(
                    index for index, label in enumerate(right_labels) if label == charge
                )
                for charge in dict.fromkeys(right_labels)
            }
            for gamma, charge in enumerate(self.internal_qns[site]):
                candidates = by_charge[charge]
                right = candidates[used[charge] % len(candidates)]
                used[charge] += 1
                index = (gamma,) + (slice(None),) * (last.ndim - 2) + (right,)
                last[index] = 1

    def _factor_support_masks(self, layout):
        if layout is None:
            return tuple(
                tuple(np.ones(factor.shape, dtype=bool) for factor in factors)
                for factors in self.factors
            )
        masks = []
        for site, (group, parent_groups, factors) in enumerate(
            zip(
                self._conditional_groups,
                self._conditional_parent_groups,
                self.factors,
            )
        ):
            if not parent_groups:
                mask = np.zeros(factors[0].shape, dtype=bool)
                for left, q_left in enumerate(layout.bond_qns[site]):
                    for physical, q_site in enumerate(layout.local_qns[site]):
                        needed = tuple(a + b for a, b in zip(q_left, q_site))
                        for right, q_right in enumerate(layout.bond_qns[site + 1]):
                            if q_right == needed:
                                mask[left, physical, right] = True
                masks.append((mask,))
                continue
            gamma_qns = self.internal_qns[site]
            B_mask = np.zeros(factors[0].shape, dtype=bool)
            for left, q_left in enumerate(layout.bond_qns[site]):
                for physical, q_site in enumerate(layout.local_qns[site]):
                    needed = tuple(a + b for a, b in zip(q_left, q_site))
                    for gamma, q_gamma in enumerate(gamma_qns):
                        if q_gamma == needed:
                            B_mask[left, physical, gamma] = True
            site_masks = [B_mask]
            for mode, factor in enumerate(factors[1:], start=1):
                mask = np.zeros(factor.shape, dtype=bool)
                if mode == len(parent_groups):
                    for gamma, q_gamma in enumerate(gamma_qns):
                        for right, q_right in enumerate(layout.bond_qns[site + 1]):
                            if q_gamma == q_right:
                                index = (gamma,) + (slice(None),) * (factor.ndim - 2)
                                index += (right,)
                                mask[index] = True
                else:
                    for left, q_left in enumerate(gamma_qns):
                        for right, q_right in enumerate(gamma_qns):
                            if q_left == q_right:
                                index = (left,) + (slice(None),) * (factor.ndim - 2)
                                index += (right,)
                                mask[index] = True
                site_masks.append(mask)
            if any(not np.any(mask) for mask in site_masks):
                raise ValueError(
                    f"chi={self.chi[site]} disconnects the U(1) factor chain at site {site}."
                )
            masks.append(tuple(site_masks))
        return tuple(masks)

    def _apply_factor_masks(self):
        for factors, masks in zip(self.factors, self.factor_masks):
            for factor, mask in zip(factors, masks):
                factor[~mask] = 0

    @property
    def nparameters(self):
        return int(sum(np.count_nonzero(mask) for site in self.factor_masks for mask in site))

    @property
    def dense_nparameters(self):
        """Parameter count of full, symmetry-unrestricted tied tensors."""
        return int(
            sum(
                self._conditional_bonds[site]
                * self._conditional_bonds[site + 1]
                * np.prod([self.dims[index] for index in group])
                for site, group in enumerate(self._conditional_groups)
            )
        )

    @property
    def unfactorized_nparameters(self):
        """Parameter count of the corresponding dense (possibly U(1)) state."""
        if hasattr(self, "local_masks"):
            return int(sum(np.count_nonzero(mask) for mask in self.local_masks))
        return self.dense_nparameters

    @property
    def stored_tensor_elements(self):
        return self.nparameters

    @property
    def peak_materialized_tensor_elements(self):
        return max(self.materialize_tensor(site).size for site in range(len(self.dims)))

    @property
    def compression_ratio(self):
        return float(self.nparameters / self.unfactorized_nparameters)

    def materialize_tensor(self, site):
        site = int(site)
        if site < 0 or site >= len(self._conditional_groups):
            raise IndexError("site is out of range.")
        parent_groups = self._conditional_parent_groups[site]
        if self.parent_group_size == 1:
            return _materialize_conditional_factors(
                self.factors[site], self._conditional_dims[site]
            )
        factors = self.factors[site]
        if not parent_groups:
            return factors[0].transpose(0, 2, 1).copy()
        pieces = []
        for physical in range(self._conditional_dims[site]):
            value = factors[0][:, physical, :]
            for factor in factors[1:]:
                value = np.tensordot(
                    value,
                    factor[:, physical, ...],
                    axes=([-1], [0]),
                )
            pieces.append(np.moveaxis(value, -1, 1))
        return np.stack(pieces, axis=2)

    def materialize_tensors(self):
        return tuple(self.materialize_tensor(site) for site in range(len(self.dims)))

    def selected_matrix(self, site, configuration):
        """Contract one site's factors for a single physical configuration."""
        site = int(site)
        if site < 0 or site >= len(self._conditional_groups):
            raise IndexError("site is out of range.")
        configuration = np.asarray(configuration, dtype=np.intp)
        if configuration.shape != (len(self.dims),):
            raise ValueError(
                f"configuration must have shape {(len(self.dims),)}."
            )
        factors = self.factors[site]
        physical = int(configuration[site])
        if len(factors) == 1:
            return factors[0][:, physical, :]
        value = factors[0][:, physical, :]
        for parent_group, factor in zip(
            self._conditional_parent_groups[site], factors[1:]
        ):
            index = (slice(None), physical)
            index += tuple(int(configuration[parent]) for parent in parent_group)
            index += (slice(None),)
            value = value @ factor[index]
        return value

    def amplitude(self, configuration):
        """Evaluate one amplitude without materializing tied local tensors."""
        if getattr(self, "autoregressive", False):
            return self._autoregressive_amplitude(configuration)
        matrices = [
            self.selected_matrix(site, configuration)
            for site in range(len(self.dims))
        ]
        value = matrices[0]
        for matrix in matrices[1:]:
            value = value @ matrix
        return np.asarray(value).reshape(()).item()

    def amplitudes(self, configurations):
        configurations = np.asarray(configurations, dtype=np.intp)
        if configurations.ndim != 2 or configurations.shape[1] != len(self.dims):
            raise ValueError(
                f"configurations must have shape (nsamples, {len(self.dims)})."
            )
        if any(
            np.any(configurations[:, site] < 0)
            or np.any(configurations[:, site] >= dim)
            for site, dim in enumerate(self.dims)
        ):
            raise ValueError("configurations contain an out-of-range local state.")
        if getattr(self, "autoregressive", False):
            return np.asarray(
                [self._autoregressive_amplitude(config) for config in configurations]
            )
        dtype = np.result_type(
            *[
                factor.dtype
                for site_factors in self.factors
                for factor in site_factors
            ]
        )
        environment = np.ones((len(configurations), 1), dtype=dtype)
        for site, (group, parent_groups, factors) in enumerate(
            zip(
                self._conditional_groups,
                self._conditional_parent_groups,
                self.factors,
            )
        ):
            physical = configurations[:, site]
            transfer = factors[0].transpose(1, 0, 2)[physical]
            for parent_group, factor in zip(parent_groups, factors[1:]):
                control = np.stack(
                    [
                        factor[
                            (slice(None), int(config[site]))
                            + tuple(int(config[parent]) for parent in parent_group)
                            + (slice(None),)
                        ]
                        for config in configurations
                    ]
                )
                transfer = transfer @ control
            environment = np.einsum(
                "ma,mab->mb",
                environment,
                transfer,
                optimize=True,
            )
        return environment[:, 0]

    def _autoregressive_candidates(self, site, configuration, right_state):
        """Return normalized-context candidate vectors for all values of a site."""
        configuration = np.asarray(configuration, dtype=np.intp)
        candidates = []
        trial = configuration.copy()
        for physical in range(self.dims[site]):
            trial[site] = physical
            candidates.append(self.selected_matrix(site, trial) @ right_state)
        return tuple(candidates)

    def _autoregressive_amplitude(self, configuration):
        r"""Evaluate the normalized suffix-conditional wavefunction.

        For a normalized right context ``r_{i+1}``, the conditional weight is

        .. math::

            p(s_i\mid s_{>i}) =
            \frac{\lVert M_i(s_i,s_{>i})r_{i+1}\rVert^2}
                 {\sum_t\lVert M_i(t,s_{>i})r_{i+1}\rVert^2}.
        """
        configuration = np.asarray(configuration, dtype=np.intp)
        if configuration.shape != (len(self.dims),):
            raise ValueError(
                f"configuration must have shape {(len(self.dims),)}."
            )
        if any(
            configuration[site] < 0 or configuration[site] >= dim
            for site, dim in enumerate(self.dims)
        ):
            raise ValueError("configuration contains an out-of-range local state.")
        dtype = np.result_type(
            *[
                factor.dtype
                for site_factors in self.factors
                for factor in site_factors
            ]
        )
        right_state = np.ones(1, dtype=dtype)
        log_probability = 0.0
        tiny = np.finfo(float).tiny
        for site in range(len(self.dims) - 1, -1, -1):
            candidates = self._autoregressive_candidates(
                site,
                configuration,
                right_state,
            )
            weights = np.asarray(
                [float(np.real(np.vdot(value, value))) for value in candidates]
            )
            total = float(np.sum(weights))
            chosen_weight = float(weights[configuration[site]])
            if not np.isfinite(total) or total <= tiny:
                return np.asarray(0, dtype=dtype).item()
            if not np.isfinite(chosen_weight) or chosen_weight <= tiny:
                return np.asarray(0, dtype=dtype).item()
            log_probability += np.log(chosen_weight / total)
            right_state = candidates[configuration[site]] / np.sqrt(chosen_weight)
        phase = right_state.reshape(()).item()
        magnitude = abs(phase)
        if magnitude <= tiny or not np.isfinite(magnitude):
            return np.asarray(0, dtype=dtype).item()
        return np.exp(0.5 * log_probability) * phase / magnitude

    def normalize(self):
        self._normalize_factor_state(self.norm())
        return self

    def balance_gauges(self, *, state_norm=None):
        # The base constructor calls this before installing the lazy tensor
        # view.  Thereafter scalar normalization must be absorbed into B.
        if not isinstance(getattr(self, "tensors", None), _ConditionalTensorView):
            return super().balance_gauges(state_norm=state_norm)
        if state_norm is None:
            norm_squared = self.norm()
        else:
            norm_squared = float(state_norm) ** 2
        self._normalize_factor_state(norm_squared)
        return self

    def canonicalize_virtual(self, direction="left"):
        raise NotImplementedError(
            "backbone canonicalization for conditional factors is not yet implemented."
        )

    def canonicalize_frontier_gauge(self, **kwargs):
        raise NotImplementedError(
            "frontier gauge transformations must be pushed into conditional factors."
        )

    def copy(self):
        kwargs = {
            "bond_dims": self.bond_dims,
            "chi": self.chi,
            "factors": [
                [factor.copy() for factor in factors]
                for factors in self.factors
            ],
            "init": self.factor_init,
            "parent_group_size": self.parent_group_size,
            "frontier_backend": self.frontier_backend,
            "chunk_size": self.chunk_size,
            "chunk_memory": self.chunk_memory,
            "chunk_span": self.chunk_span,
            "workers": self.workers,
            "path_optimizer": self.path_optimizer,
            "max_rank": self.tt_options["max_rank"],
            "rtol": self.tt_options["rtol"],
            "atol": self.tt_options["atol"],
            "transfer_max_rank": self.tt_options["transfer_max_rank"],
            "transfer_rtol": self.tt_options["transfer_rtol"],
            "transfer_atol": self.tt_options["transfer_atol"],
            "tt_absorption": self.tt_options["absorption"],
            "tt_norm_backend": self.tt_norm_backend,
            "tt_hermitize": self.tt_hermitize,
            "tt_channels": self.tt_channels,
            "tt_gauge": self.tt_gauge,
        }
        if hasattr(self, "abelian_layout"):
            kwargs["abelian_layout"] = self.abelian_layout
        result = type(self)(self.hamiltonian, self.parent_sets, **kwargs)
        result.history = deepcopy(self.history)
        result.energy = self.energy
        result.converged = self.converged
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return self._copy_public_settings_to(result)

    def _factor_jacobian(self, site, factor_index):
        block = self.factors[site][factor_index]
        mask = self.factor_masks[site][factor_index]
        active = np.flatnonzero(mask.reshape(-1))
        original = block.copy()
        local_size = self.materialize_tensor(site).size
        jacobian = np.empty(
            (local_size, active.size),
            dtype=np.result_type(block.dtype, complex),
        )
        try:
            block.fill(0)
            for column, flat_index in enumerate(active):
                block.flat[flat_index] = 1
                jacobian[:, column] = self.materialize_tensor(site).reshape(-1)
                block.flat[flat_index] = 0
        finally:
            block[...] = original
        return jacobian, active

    def optimize_factor(
        self,
        site,
        factor,
        *,
        environment=None,
        metric_tol=1.0e-12,
    ):
        """Optimize one B/C factor using exact matrix-free frontier actions."""
        site = self._validated_site(site)
        factor = int(factor)
        if factor < 0 or factor >= len(self.factors[site]):
            raise IndexError("factor is out of range.")
        environment = self._resolved_environment(site, environment)
        block = self.factors[site][factor]
        old_block = block.copy()
        jacobian, active = self._factor_jacobian(site, factor)
        vectors = jacobian.T
        if hasattr(self._norm_frontier, "hole_actions"):
            metric_columns = self._norm_frontier.hole_actions(
                site,
                environment.norm_left,
                environment.norm_right,
                vectors,
            ).T
        else:
            metric_columns = np.column_stack(
                [
                    self.metric_action(site, column, environment=environment)
                    for column in vectors
                ]
            )
        if hasattr(self._hamiltonian_frontier, "hole_actions"):
            hamiltonian_columns = self._hamiltonian_frontier.hole_actions(
                site,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
                vectors,
            ).T
        else:
            hamiltonian_columns = np.column_stack(
                [
                    self.hamiltonian_action(site, column, environment=environment)
                    for column in vectors
                ]
            )
        metric = jacobian.T.conj() @ metric_columns
        effective = jacobian.T.conj() @ hamiltonian_columns
        metric = self._hermitian_part(metric)
        effective = self._hermitian_part(effective)
        old_vector = old_block.reshape(-1)[active]
        old_norm = np.vdot(old_vector, metric @ old_vector)
        if abs(old_norm) <= np.finfo(float).tiny:
            raise ValueError("conditional factor has zero frontier norm.")
        energy_before = float(
            np.real(np.vdot(old_vector, effective @ old_vector) / old_norm)
        )
        eigenvalues = np.linalg.eigvalsh(metric)
        scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), np.finfo(float).tiny)
        metric_rank = int(np.count_nonzero(eigenvalues > metric_tol * scale))
        accepted = False
        energy = energy_before
        try:
            _root, vector = _lowest_generalized_eigenpair(
                effective,
                metric,
                metric_tol=metric_tol,
            )
            candidate_norm = np.vdot(vector, metric @ vector)
            candidate_energy = float(
                np.real(np.vdot(vector, effective @ vector) / candidate_norm)
            )
            tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(energy_before))
            accepted = np.isfinite(candidate_energy) and candidate_energy <= energy_before + tolerance
            if accepted:
                block.fill(0)
                block.reshape(-1)[active] = vector
                energy = candidate_energy
        except (ValueError, np.linalg.LinAlgError):
            accepted = False
        if not accepted:
            block[...] = old_block
        self.energy = float(energy)
        return ConditionalTTUpdate(
            site=site,
            factor=factor,
            raw_dim=active.size,
            metric_rank=metric_rank,
            energy_before=energy_before,
            energy=float(energy),
            accepted=accepted,
        )

    def _optimize_site_factors(self, site, environment, *, reverse, metric_tol):
        indices = tuple(range(len(self.factors[site])))
        if reverse:
            indices = indices[::-1]
        return tuple(
            self.optimize_factor(
                site,
                factor,
                environment=environment,
                metric_tol=metric_tol,
            )
            for factor in indices
        )

    def _normalize_factor_state(self, norm_squared):
        norm_squared = float(np.real(norm_squared))
        if not np.isfinite(norm_squared) or norm_squared <= 0.0:
            raise ValueError("conditional frontier LETTA state is numerically zero.")
        self.factors[0][0] /= np.sqrt(norm_squared)

    def run(
        self,
        *,
        nsweeps=4,
        sweep_offset=0,
        tol=1.0e-10,
        metric_tol=1.0e-12,
        environment_cache="checkpointed",
        environment_checkpoint_interval=None,
        verbose=False,
        **unsupported,
    ):
        """Sweep over B/C factors using exact frontier environments."""
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"unsupported conditional-TT sweep options: {names}.")
        nsweeps = int(nsweeps)
        sweep_offset = int(sweep_offset)
        if nsweeps < 0 or sweep_offset < 0:
            raise ValueError("nsweeps and sweep_offset must be nonnegative.")
        if not self.norm_contraction_is_exact:
            raise ValueError("factor sweeps require an exact norm contraction.")
        if not self.hamiltonian_action_is_hermitian:
            raise ValueError("factor sweeps require a Hermitian Hamiltonian action.")
        environment_cache = str(environment_cache).lower().replace("-", "_")
        if environment_cache in {"checkpoint", "recompute"}:
            environment_cache = "checkpointed"
        if environment_cache not in {"checkpointed", "full"}:
            raise ValueError("environment_cache must be 'checkpointed' or 'full'.")
        if environment_checkpoint_interval is None:
            environment_checkpoint_interval = max(1, int(np.ceil(np.sqrt(len(self.dims)))))
        environment_checkpoint_interval = int(environment_checkpoint_interval)
        if environment_checkpoint_interval < 1:
            raise ValueError("environment_checkpoint_interval must be positive.")

        previous = self.expectation()
        self.energy = previous
        self.history = []
        self.converged = False
        for sweep in range(nsweeps):
            direction = (sweep_offset + sweep) % 2
            updates = []
            cuts = self._environment_checkpoint_cuts(environment_checkpoint_interval)
            if direction == 0:
                if environment_cache == "full":
                    norm_fixed = self._norm_frontier.build_right(self.tensors)
                    hamiltonian_fixed = self._hamiltonian_frontier.build_right(self.tensors)
                    checkpoints = None
                else:
                    norm_fixed = hamiltonian_fixed = None
                    checkpoints = (
                        self._build_environment_checkpoints(
                            self._norm_frontier,
                            direction="right",
                            cuts=cuts,
                        ),
                        self._build_environment_checkpoints(
                            self._hamiltonian_frontier,
                            direction="right",
                            cuts=cuts,
                        ),
                    )
                moving_norm = self._norm_frontier.left_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.left_boundary()
                for start, end in zip(cuts[:-1], cuts[1:]):
                    if checkpoints is not None:
                        norm_fixed = self._recompute_environment_block(
                            self._norm_frontier,
                            direction="right",
                            start=start,
                            end=end,
                            checkpoint=checkpoints[0][end],
                        )
                        hamiltonian_fixed = self._recompute_environment_block(
                            self._hamiltonian_frontier,
                            direction="right",
                            start=start,
                            end=end,
                            checkpoint=checkpoints[1][end],
                        )
                    for site in range(start, end):
                        environment = FrontierSiteEnvironment(
                            site,
                            moving_norm,
                            norm_fixed[site + 1],
                            moving_hamiltonian,
                            hamiltonian_fixed[site + 1],
                        )
                        updates.extend(
                            self._optimize_site_factors(
                                site,
                                environment,
                                reverse=False,
                                metric_tol=metric_tol,
                            )
                        )
                        moving_norm = self._norm_frontier.advance_left(
                            moving_norm, self.tensors, site
                        )
                        moving_hamiltonian = self._hamiltonian_frontier.advance_left(
                            moving_hamiltonian, self.tensors, site
                        )
                boundary = len(self.dims)
            else:
                if environment_cache == "full":
                    norm_fixed = self._norm_frontier.build_left(self.tensors)
                    hamiltonian_fixed = self._hamiltonian_frontier.build_left(self.tensors)
                    checkpoints = None
                else:
                    norm_fixed = hamiltonian_fixed = None
                    checkpoints = (
                        self._build_environment_checkpoints(
                            self._norm_frontier,
                            direction="left",
                            cuts=cuts,
                        ),
                        self._build_environment_checkpoints(
                            self._hamiltonian_frontier,
                            direction="left",
                            cuts=cuts,
                        ),
                    )
                moving_norm = self._norm_frontier.right_boundary()
                moving_hamiltonian = self._hamiltonian_frontier.right_boundary()
                for start, end in reversed(tuple(zip(cuts[:-1], cuts[1:]))):
                    if checkpoints is not None:
                        norm_fixed = self._recompute_environment_block(
                            self._norm_frontier,
                            direction="left",
                            start=start,
                            end=end,
                            checkpoint=checkpoints[0][start],
                        )
                        hamiltonian_fixed = self._recompute_environment_block(
                            self._hamiltonian_frontier,
                            direction="left",
                            start=start,
                            end=end,
                            checkpoint=checkpoints[1][start],
                        )
                    for site in range(end - 1, start - 1, -1):
                        environment = FrontierSiteEnvironment(
                            site,
                            norm_fixed[site],
                            moving_norm,
                            hamiltonian_fixed[site],
                            moving_hamiltonian,
                        )
                        updates.extend(
                            self._optimize_site_factors(
                                site,
                                environment,
                                reverse=True,
                                metric_tol=metric_tol,
                            )
                        )
                        moving_norm = self._norm_frontier.advance_right(
                            moving_norm, self.tensors, site
                        )
                        moving_hamiltonian = self._hamiltonian_frontier.advance_right(
                            moving_hamiltonian, self.tensors, site
                        )
                boundary = 0

            norm = self._completed_frontier_scalar(
                self._norm_frontier,
                moving_norm,
                boundary,
            )
            numerator = self._completed_frontier_scalar(
                self._hamiltonian_frontier,
                moving_hamiltonian,
                boundary,
            )
            energy = float(np.real(numerator / norm))
            self._normalize_factor_state(norm)
            delta = abs(energy - previous)
            self.energy = energy
            self.history.append(
                {
                    "sweep": sweep_offset + sweep,
                    "direction": "left_to_right" if direction == 0 else "right_to_left",
                    "energy": energy,
                    "delta_energy": delta,
                    "accepted_factors": sum(update.accepted for update in updates),
                    "updates": tuple(updates),
                    "parameters": self.nparameters,
                    "dense_parameters": self.dense_nparameters,
                    "unfactorized_parameters": self.unfactorized_nparameters,
                    "environment_cache": environment_cache,
                }
            )
            if verbose:
                print(
                    f"conditional-frontier sweep={sweep_offset + sweep:2d} "
                    f"E={energy: .12f} dE={delta:.3e} "
                    f"accepted={sum(update.accepted for update in updates)}"
                )
            if delta <= float(tol):
                self.converged = True
                break
            previous = energy
        return self


class ConditionalFrontierLETTA(_ConditionalTTFrontierMixin, FrontierTiedLETTA):
    """Unrestricted conditional-TT LETTA with exact frontier contraction."""


class U1ConditionalFrontierLETTA(
    _ConditionalTTFrontierMixin,
    AbelianFrontierTiedLETTA,
):
    """U(1)-blocked conditional-TT LETTA with neutral tied controls."""


class FactorizedFutureLETTA(_ConditionalTTFrontierMixin):
    r"""Graph/future-tied LETTA evaluated entirely through local factors.

    Site ``i`` carries conditional factors for every physical variable
    ``s_j`` with ``j > i``.  The local tensor is never materialized:

    .. math::

        A_i(s_i, s_{i+1}, \ldots, s_{N-1})
        = B_i(s_i)\prod_{j=i+1}^{N-1} C_{ij}(s_i, s_j).

    This class intentionally has no exact frontier contraction.  Explicit
    configuration enumeration is available as a small-system reference;
    scalable energies and optimization use :class:`pyqed.letta.VMC`.
    """

    def __init__(
        self,
        hamiltonian,
        *,
        bond_dims,
        parent_sets=None,
        chi=1,
        factors=None,
        init="mps",
        abelian_layout=None,
        autoregressive=False,
        seed=None,
    ):
        dims = tuple(hamiltonian.dims)
        nsites = len(dims)
        if parent_sets is None:
            parents = tuple(
                tuple(range(site + 1, nsites)) for site in range(nsites)
            )
        else:
            parents = _validated_parent_sets(dims, parent_sets)
        bonds = tuple(int(value) for value in bond_dims)
        if len(bonds) != nsites + 1 or bonds[0] != 1 or bonds[-1] != 1:
            raise ValueError("bond_dims must include unit open boundaries.")
        if any(value < 1 for value in bonds):
            raise ValueError("bond_dims must contain only positive dimensions.")
        if abelian_layout is not None and not isinstance(
            abelian_layout, FrontierAbelianLayout
        ):
            raise TypeError("abelian_layout must be a FrontierAbelianLayout.")

        self.hamiltonian = hamiltonian
        self.sites = tuple(hamiltonian.sites)
        self.dims = dims
        self.parent_sets = parents
        self.physical_groups = tuple(
            (site,) + site_parents
            for site, site_parents in enumerate(parents)
        )
        self.graph = tuple(
            (site, parent)
            for site, site_parents in enumerate(parents)
            for parent in site_parents
        )
        self.ordering = tuple(range(nsites))
        self.bond_dims = bonds
        self._conditional_bonds = bonds
        self._conditional_dims = dims
        self._conditional_groups = self.physical_groups
        self.parent_group_size = 1
        self._conditional_parent_groups = tuple(
            tuple((parent,) for parent in group[1:])
            for group in self.physical_groups
        )
        self.chi = _expanded_ints(chi, nsites, name="chi")
        self.abelian_layout = abelian_layout
        self.internal_qns = self._internal_charge_layout(abelian_layout)
        init = str(init).lower().replace("-", "_")
        if init not in {"mps", "random"}:
            raise ValueError("init must be 'mps' or 'random'.")
        self.factor_init = init
        self.autoregressive = bool(autoregressive)
        self.rng = np.random.default_rng(seed)
        dtype = np.result_type(hamiltonian.dtype, np.complex128)
        if factors is None:
            self.factors = [
                self._random_factors(site, self.rng, dtype)
                for site in range(nsites)
            ]
            if init == "mps":
                self._initialize_mps_controls(abelian_layout)
        else:
            if len(factors) != nsites:
                raise ValueError("factors must contain one factor sequence per site.")
            self.factors = [
                [np.asarray(factor, dtype=dtype).copy() for factor in site_factors]
                for site_factors in factors
            ]
            self._validate_conditional_factors(dims)
        self.factor_masks = self._factor_support_masks(abelian_layout)
        self._apply_factor_masks()
        self.energy = None
        self.history = []
        self.adaptation_history = []
        self.converged = False
        self.contraction = "vmc"

    @property
    def D(self):
        return max(self.bond_dims)

    def copy(self):
        result = type(self)(
            self.hamiltonian,
            bond_dims=self.bond_dims,
            parent_sets=self.parent_sets,
            chi=self.chi,
            factors=[
                [factor.copy() for factor in site_factors]
                for site_factors in self.factors
            ],
            init=self.factor_init,
            abelian_layout=self.abelian_layout,
            autoregressive=self.autoregressive,
        )
        result.target_charge = getattr(self, "target_charge", None)
        result.adaptation_history = deepcopy(
            getattr(self, "adaptation_history", [])
        )
        return result

    def _right_internal_embedding(self, site):
        """Map every right-bond column into a matching internal direction."""
        right_dim = self.bond_dims[site + 1]
        if self.chi[site] < right_dim:
            raise ValueError(
                f"adding the first/latest tie at site {site} without changing "
                f"the state requires chi >= {right_dim}."
            )
        if self.abelian_layout is None:
            return tuple(range(right_dim))
        gamma_qns = _ranked_charge_labels(
            self.abelian_layout.bond_qns[site + 1],
            self.chi[site],
        )
        used = set()
        embedding = []
        for charge in self.abelian_layout.bond_qns[site + 1]:
            candidates = [
                gamma
                for gamma, gamma_charge in enumerate(gamma_qns)
                if gamma_charge == charge and gamma not in used
            ]
            if not candidates:
                raise ValueError(
                    f"chi={self.chi[site]} lacks a copy of charge {charge} "
                    f"needed to add a tie at site {site}."
                )
            gamma = candidates[0]
            used.add(gamma)
            embedding.append(gamma)
        return tuple(embedding)

    def with_added_ties(self, edges):
        """Return a state-preserving copy with neutral new conditional ties."""
        nsites = len(self.dims)
        additions = set()
        for edge in edges:
            values = tuple(int(site) for site in edge)
            if len(values) != 2:
                raise ValueError("each tie must contain two site indices.")
            left, right = sorted(values)
            if left < 0 or right >= nsites or left == right:
                raise ValueError("ties must join distinct valid sites.")
            if right not in self.parent_sets[left]:
                additions.add((left, right))
        parents = [list(site_parents) for site_parents in self.parent_sets]
        factors = [
            [factor.copy() for factor in site_factors]
            for site_factors in self.factors
        ]
        for site, parent in sorted(additions):
            old_parents = parents[site]
            position = int(np.searchsorted(old_parents, parent))
            local_dim = self.dims[site]
            parent_dim = self.dims[parent]
            dtype = factors[site][0].dtype
            if not old_parents:
                old = factors[site][0]
                embedding = self._right_internal_embedding(site)
                first = np.zeros(
                    (old.shape[0], local_dim, self.chi[site]),
                    dtype=dtype,
                )
                last = np.zeros(
                    (self.chi[site], local_dim, parent_dim, old.shape[-1]),
                    dtype=dtype,
                )
                for right, gamma in enumerate(embedding):
                    first[:, :, gamma] = old[:, :, right]
                    last[gamma, :, :, right] = 1
                factors[site] = [first, last]
            elif position < len(old_parents):
                identity = np.zeros(
                    (
                        self.chi[site],
                        local_dim,
                        parent_dim,
                        self.chi[site],
                    ),
                    dtype=dtype,
                )
                for gamma in range(self.chi[site]):
                    identity[gamma, :, :, gamma] = 1
                factors[site].insert(position + 1, identity)
            else:
                old_last = factors[site][-1]
                embedding = self._right_internal_embedding(site)
                intermediate = np.zeros(
                    old_last.shape[:-1] + (self.chi[site],),
                    dtype=dtype,
                )
                last = np.zeros(
                    (
                        self.chi[site],
                        local_dim,
                        parent_dim,
                        old_last.shape[-1],
                    ),
                    dtype=dtype,
                )
                for right, gamma in enumerate(embedding):
                    intermediate[..., gamma] = old_last[..., right]
                    last[gamma, :, :, right] = 1
                factors[site][-1] = intermediate
                factors[site].append(last)
            old_parents.insert(position, parent)

        result = type(self)(
            self.hamiltonian,
            bond_dims=self.bond_dims,
            parent_sets=tuple(tuple(site_parents) for site_parents in parents),
            chi=self.chi,
            factors=factors,
            init=self.factor_init,
            abelian_layout=self.abelian_layout,
            autoregressive=self.autoregressive,
        )
        result.target_charge = getattr(self, "target_charge", None)
        result.adaptation_history = deepcopy(
            getattr(self, "adaptation_history", [])
        )
        return result

    def adapt_ties(
        self,
        *,
        n_ties=1,
        nsamples=4096,
        candidate_edges=None,
        min_span=2,
        seed=None,
    ):
        """Add high-correlation absent ties using independent current-state samples."""
        if not self.autoregressive:
            raise ValueError("sample-driven tie adaptation requires autoregressive=True.")
        n_ties = int(n_ties)
        nsamples = int(nsamples)
        min_span = int(min_span)
        if n_ties < 0 or nsamples < 1 or min_span < 1:
            raise ValueError(
                "n_ties must be nonnegative, nsamples positive, and min_span positive."
            )
        existing = set(self.graph)
        if candidate_edges is None:
            candidates = tuple(
                (left, right)
                for left in range(len(self.dims))
                for right in range(left + min_span, len(self.dims))
                if (left, right) not in existing
            )
        else:
            normalized = set()
            for edge in candidate_edges:
                values = tuple(int(site) for site in edge)
                if len(values) != 2:
                    raise ValueError("each candidate edge must contain two sites.")
                left, right = sorted(values)
                if left < 0 or right >= len(self.dims) or left == right:
                    raise ValueError("candidate edges must join distinct valid sites.")
                if (left, right) not in existing:
                    normalized.add((left, right))
            candidates = tuple(sorted(normalized))
        samples = self.sample(nsamples, seed=seed)
        scored = []
        for left, right in candidates:
            flat = samples[:, left] * self.dims[right] + samples[:, right]
            joint = np.bincount(
                flat,
                minlength=self.dims[left] * self.dims[right],
            ).reshape(self.dims[left], self.dims[right]) / nsamples
            independent = np.outer(np.sum(joint, axis=1), np.sum(joint, axis=0))
            score = 0.5 * float(np.sum(np.abs(joint - independent)))
            scored.append(((left, right), score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        selected = tuple(edge for edge, _score in scored[:n_ties])
        result = self.with_added_ties(selected)
        result.adaptation_history.append(
            {
                "added_ties": selected,
                "scores": tuple(scored[:n_ties]),
                "nsamples": nsamples,
                "candidate_count": len(candidates),
            }
        )
        return result

    def sample(self, nsamples, *, seed=None, rng=None, return_amplitudes=False):
        """Draw independent suffix-to-prefix samples from an autoregressive state."""
        if not self.autoregressive:
            raise ValueError("independent sampling requires autoregressive=True.")
        nsamples = int(nsamples)
        if nsamples < 1:
            raise ValueError("nsamples must be positive.")
        if rng is not None and seed is not None:
            raise TypeError("supply either seed or rng, not both.")
        if rng is None:
            rng = np.random.default_rng(seed)
        dtype = np.result_type(
            *[
                factor.dtype
                for site_factors in self.factors
                for factor in site_factors
            ]
        )
        configurations = np.zeros((nsamples, len(self.dims)), dtype=np.intp)
        amplitudes = np.empty(nsamples, dtype=dtype)
        tiny = np.finfo(float).tiny
        for sample in range(nsamples):
            configuration = configurations[sample]
            right_state = np.ones(1, dtype=dtype)
            log_probability = 0.0
            for site in range(len(self.dims) - 1, -1, -1):
                candidates = self._autoregressive_candidates(
                    site,
                    configuration,
                    right_state,
                )
                weights = np.asarray(
                    [float(np.real(np.vdot(value, value))) for value in candidates]
                )
                total = float(np.sum(weights))
                if not np.isfinite(total) or total <= tiny:
                    raise ValueError(
                        "autoregressive factors contain a zero-probability suffix."
                    )
                probabilities = weights / total
                physical = int(rng.choice(self.dims[site], p=probabilities))
                configuration[site] = physical
                chosen_weight = float(weights[physical])
                log_probability += np.log(chosen_weight / total)
                right_state = candidates[physical] / np.sqrt(chosen_weight)
            phase = right_state.reshape(()).item()
            amplitudes[sample] = (
                np.exp(0.5 * log_probability) * phase / abs(phase)
            )
        return (configurations, amplitudes) if return_amplitudes else configurations

    def state_vector(
        self,
        *,
        normalize=False,
        max_states=1_000_000,
        batch_size=4096,
    ):
        """Enumerate the exact state vector without materializing local tensors."""
        dimension = int(np.prod(self.dims, dtype=int))
        if max_states is not None and dimension > int(max_states):
            raise ValueError(
                f"exact enumeration needs {dimension} configurations, exceeding "
                f"max_states={int(max_states)}; use VMC for scalable evaluation."
            )
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        dtype = np.result_type(
            *[
                factor.dtype
                for site_factors in self.factors
                for factor in site_factors
            ]
        )
        vector = np.empty(dimension, dtype=dtype)
        for start in range(0, dimension, batch_size):
            stop = min(start + batch_size, dimension)
            configurations = np.column_stack(
                np.unravel_index(np.arange(start, stop), self.dims)
            )
            vector[start:stop] = self.amplitudes(configurations)
        if normalize:
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("all-future LETTA state is zero or nonfinite.")
            vector /= norm
        return vector

    def expectation(self, *, max_states=1_000_000, batch_size=4096):
        """Return an exact small-system energy by explicit enumeration."""
        vector = self.state_vector(
            max_states=max_states,
            batch_size=batch_size,
        )
        return self.hamiltonian.expectation(vector)

    def norm(self, *, max_states=1_000_000, batch_size=4096):
        """Return the exact squared norm by explicit enumeration."""
        vector = self.state_vector(
            max_states=max_states,
            batch_size=batch_size,
        )
        return float(np.real(np.vdot(vector, vector)))

    def normalize(self, *, max_states=1_000_000, batch_size=4096):
        norm_squared = self.norm(
            max_states=max_states,
            batch_size=batch_size,
        )
        if not np.isfinite(norm_squared) or norm_squared <= 0.0:
            raise ValueError("all-future LETTA state is zero or nonfinite.")
        self.factors[0][0] /= np.sqrt(norm_squared)
        return self

    def balance_gauges(self, *, state_norm=None):
        raise NotImplementedError(
            "global gauge balancing requires an exact all-future contraction."
        )

    def run(self, **kwargs):
        raise NotImplementedError(
            "FactorizedFutureLETTA is optimized with VMC(state), not an exact sweep."
        )


__all__ = [
    "ConditionalFrontierLETTA",
    "FactorizedFutureLETTA",
    "U1ConditionalFrontierLETTA",
]
