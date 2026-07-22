"""Adaptive physical-context truncation of recursive exact factorizations.

For a wavefunction residual ``psi[x_k, z]``, where ``z`` is the complete
future configuration, a chosen physical parent set partitions ``z`` into
context cells.  Inside each cell the optimal physical-only conditional factor
is the leading left singular vector of the corresponding residual matrix.  Its
right singular coefficients form the residual passed to the next site.

Repeating this construction produces a normalized conditional-product state
without summed virtual indices.  Parent sets may be fixed or selected greedily
from the local discarded Gram weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


def _validate_dims(dims) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in dims)
    if not dims or any(dim < 1 for dim in dims):
        raise ValueError("dims must contain positive integers.")
    return dims


def _normalized_vector(state, full_dim: int) -> np.ndarray:
    state = np.asarray(state)
    if state.size != int(full_dim):
        raise ValueError("state size is inconsistent with dims.")
    vector = state.reshape(-1).astype(np.result_type(state.dtype, complex), copy=True)
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("state must be nonzero.")
    return vector / norm


def fixed_range_parent_sets(nsites: int, tie_range: int) -> tuple[tuple[int, ...], ...]:
    """Return forward contiguous parent sets for a range-``tie_range`` XF."""
    nsites = int(nsites)
    tie_range = int(tie_range)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    if tie_range < 0:
        raise ValueError("tie_range must be nonnegative.")
    return tuple(
        tuple(range(site + 1, min(nsites, site + 1 + tie_range)))
        for site in range(nsites - 1)
    )


def _validate_parent_set(site: int, parents, nsites: int) -> tuple[int, ...]:
    parents = tuple(sorted({int(parent) for parent in parents}))
    if any(parent <= site or parent >= nsites for parent in parents):
        raise ValueError(f"parents for site {site} must be future site indices.")
    return parents


def _phase_fix_conditional(conditional: np.ndarray, coefficients: np.ndarray):
    """Make the largest conditional component real without changing its product."""
    pivot = int(np.argmax(np.abs(conditional)))
    if abs(conditional[pivot]) == 0.0:
        return conditional, coefficients
    phase = np.exp(-1.0j * np.angle(conditional[pivot]))
    return conditional * phase, coefficients * phase.conjugate()


def _principal_gram_factor(matrix: np.ndarray):
    """Return a deterministic principal Gram vector and projection coefficients."""
    gram = matrix @ matrix.T.conj()
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = max(0.0, float(np.real(eigenvalues[-1])))
    scale = max(largest, np.finfo(float).tiny)
    degeneracy_tol = 64.0 * np.finfo(float).eps * max(1, matrix.shape[0]) * scale
    leading = eigenvalues >= eigenvalues[-1] - degeneracy_tol
    subspace = eigenvectors[:, leading]
    projector = subspace @ subspace.T.conj()

    conditional = None
    vector_tol = np.sqrt(np.finfo(float).eps)
    for pivot in range(matrix.shape[0]):
        candidate = projector[:, pivot]
        norm = np.linalg.norm(candidate)
        if norm > vector_tol:
            conditional = candidate / norm
            break
    if conditional is None:  # pragma: no cover - a nonempty eigenspace always spans something
        conditional = eigenvectors[:, -1]

    coefficients = conditional.conj() @ matrix
    conditional, coefficients = _phase_fix_conditional(conditional, coefficients)
    discarded = max(0.0, float(np.trace(gram).real - largest))
    return conditional, coefficients, discarded


@dataclass(frozen=True)
class PhysicalTieStep:
    """Diagnostics for one conditional rank-one factorization step."""

    site: int
    parents: tuple[int, ...]
    discarded_weight: float
    input_weight: float
    retained_weight: float

    @property
    def relative_discarded_weight(self) -> float:
        if self.input_weight <= 0.0:
            return 0.0
        return float(self.discarded_weight / self.input_weight)


@dataclass
class PhysicalTieState:
    """Conditional-product state produced by physical-context truncation."""

    dims: tuple[int, ...]
    factors: tuple[np.ndarray, ...]
    parent_sets: tuple[tuple[int, ...], ...]
    terminal: np.ndarray
    steps: tuple[PhysicalTieStep, ...]

    @property
    def nsites(self) -> int:
        return len(self.dims)

    @property
    def discarded_weight(self) -> float:
        return float(sum(step.discarded_weight for step in self.steps))

    @property
    def retained_weight(self) -> float:
        return float(np.vdot(self.terminal, self.terminal).real)

    def state_vector(self, *, normalize: bool = True) -> np.ndarray:
        """Return the represented dense product-basis vector."""
        return _physical_tie_vector(
            self.dims,
            self.factors,
            self.parent_sets,
            self.terminal,
            normalize=normalize,
        )

    def fidelity(self, state) -> float:
        """Return normalized squared overlap with ``state``."""
        target = _normalized_vector(state, int(np.prod(self.dims)))
        approximate = self.state_vector(normalize=True)
        return float(abs(np.vdot(target, approximate)) ** 2)

    def variational_ansatz(self, hamiltonian) -> "VariationalPhysicalTie":
        """Return an independent dense variational ansatz seeded from this state."""
        return VariationalPhysicalTie.from_compressed(self, hamiltonian)


def _physical_tie_vector(dims, factors, parent_sets, terminal, *, normalize: bool):
    dtype = np.result_type(np.asarray(terminal).dtype, *[factor.dtype for factor in factors])
    state = np.empty(dims, dtype=dtype)
    for config in np.ndindex(*dims):
        amplitude = terminal[config[-1]]
        for site, (factor, parents) in enumerate(zip(factors, parent_sets)):
            index = (config[site],) + tuple(config[parent] for parent in parents)
            amplitude *= factor[index]
        state[config] = amplitude
    vector = state.reshape(-1)
    if normalize:
        norm = np.linalg.norm(vector)
        if norm <= 0.0:
            raise ValueError("physical-tie state is numerically zero.")
        vector = vector / norm
    return vector


@dataclass
class VariationalPhysicalTie:
    """Dense variational relaxation with physical-parent ties.

    Tensor updates keep every physical tie fixed.  ``run_parent_search`` can
    replace ties by exhaustive local Rayleigh--Ritz comparisons, while
    ``run_relaxed_parent_search`` judges a screened set of temporarily uphill
    graph trials after independent tensor relaxation.  Individual context
    columns need not remain normalized; the represented state is normalized
    globally, so this is a variational relaxation of the recursive-XF seed
    rather than another exact-factorization gauge.
    """

    hamiltonian: np.ndarray
    dims: tuple[int, ...]
    factors: list[np.ndarray]
    parent_sets: tuple[tuple[int, ...], ...]
    terminal: np.ndarray
    history: list[dict] = field(default_factory=list)
    graph_history: list[dict] = field(default_factory=list)
    relaxed_graph_history: list[dict] = field(default_factory=list)
    energy: float | None = None
    converged: bool = False
    _configs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.dims = _validate_dims(self.dims)
        full_dim = int(np.prod(self.dims))
        matrix = self.hamiltonian.toarray() if hasattr(self.hamiltonian, "toarray") else self.hamiltonian
        self.hamiltonian = np.asarray(matrix)
        if self.hamiltonian.shape != (full_dim, full_dim):
            raise ValueError("hamiltonian shape is inconsistent with dims.")
        if not np.all(np.isfinite(self.hamiltonian)):
            raise ValueError("hamiltonian must contain only finite values.")
        if not np.allclose(self.hamiltonian, self.hamiltonian.T.conj(), rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("hamiltonian must be Hermitian.")
        if len(self.factors) != len(self.dims) - 1:
            raise ValueError("factors must contain one entry per nonterminal site.")
        if len(self.parent_sets) != len(self.factors):
            raise ValueError("parent_sets must contain one entry per factor.")
        self.parent_sets = tuple(
            _validate_parent_set(site, parents, len(self.dims))
            for site, parents in enumerate(self.parent_sets)
        )
        self._configs = np.array(list(np.ndindex(*self.dims)), dtype=np.intp)
        self.factors = [np.asarray(factor, dtype=complex).copy() for factor in self.factors]
        self.terminal = np.asarray(self.terminal, dtype=complex).copy()
        for site, (factor, parents) in enumerate(zip(self.factors, self.parent_sets)):
            expected = (self.dims[site],) + tuple(self.dims[parent] for parent in parents)
            if factor.shape != expected:
                raise ValueError(f"factor {site} shape {factor.shape} does not match {expected}.")
        if self.terminal.shape != (self.dims[-1],):
            raise ValueError("terminal shape must match the final local dimension.")
        self._balance_scalar_gauges()
        self.normalize()
        self.energy = self.expectation()

    @classmethod
    def from_compressed(cls, state: PhysicalTieState, hamiltonian):
        """Copy a compressed physical-tie state into a variational ansatz."""
        return cls(
            hamiltonian=hamiltonian,
            dims=state.dims,
            factors=[factor.copy() for factor in state.factors],
            parent_sets=state.parent_sets,
            terminal=state.terminal.copy(),
        )

    def copy(self) -> "VariationalPhysicalTie":
        """Return an independent ansatz with the same physical state and graph."""
        return type(self)(
            hamiltonian=self.hamiltonian,
            dims=self.dims,
            factors=[factor.copy() for factor in self.factors],
            parent_sets=self.parent_sets,
            terminal=self.terminal.copy(),
        )

    @property
    def nvariables(self) -> int:
        return len(self.factors) + 1

    def state_vector(self, *, normalize: bool = False) -> np.ndarray:
        return _physical_tie_vector(
            self.dims,
            self.factors,
            self.parent_sets,
            self.terminal,
            normalize=normalize,
        )

    def norm(self) -> float:
        vector = self.state_vector(normalize=False)
        return float(np.vdot(vector, vector).real)

    @staticmethod
    def _unit_array(array: np.ndarray):
        largest = float(np.max(np.abs(array))) if array.size else 0.0
        if largest == 0.0:
            return None
        scaled = array / largest
        scaled_norm = float(np.linalg.norm(scaled))
        return scaled / scaled_norm

    def _balance_scalar_gauges(self) -> None:
        """Remove all scalar tensor gauges without forming their product."""
        arrays = [*self.factors, self.terminal]
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("factors and terminal must contain only finite values.")
        units = []
        contains_zero = False
        for array in arrays:
            unit = self._unit_array(array)
            contains_zero = contains_zero or unit is None
            units.append(unit)
        if contains_zero:
            return
        self.factors = units[:-1]
        self.terminal = units[-1]

    @staticmethod
    def _scaled_norm_parts(vector: np.ndarray) -> tuple[float, float]:
        if not np.all(np.isfinite(vector)):
            raise ValueError("physical-tie state contains nonfinite values.")
        largest = float(np.max(np.abs(vector))) if vector.size else 0.0
        if largest == 0.0:
            raise ValueError("physical-tie state is zero in floating-point arithmetic.")
        scaled_norm = float(np.linalg.norm(vector / largest))
        if not np.isfinite(scaled_norm) or scaled_norm == 0.0:
            raise ValueError("physical-tie state has an invalid norm.")
        return largest, scaled_norm

    def normalize(self) -> "VariationalPhysicalTie":
        self._balance_scalar_gauges()
        largest, scaled_norm = self._scaled_norm_parts(self.state_vector(normalize=False))
        log_norm = float(np.log(largest) + np.log(scaled_norm))
        common_scale = float(np.exp(-log_norm / self.nvariables))
        if not np.isfinite(common_scale) or common_scale == 0.0:
            raise ValueError("physical-tie state cannot be normalized in floating point.")
        self.factors = [factor * common_scale for factor in self.factors]
        self.terminal *= common_scale

        largest, scaled_norm = self._scaled_norm_parts(self.state_vector(normalize=False))
        correction = float(np.exp(-(np.log(largest) + np.log(scaled_norm))))
        self.terminal *= correction
        return self

    def expectation(self) -> float:
        vector = self.state_vector(normalize=False)
        largest, _ = self._scaled_norm_parts(vector)
        scaled = vector / largest
        norm = np.vdot(scaled, scaled)
        return float(np.real(np.vdot(scaled, self.hamiltonian @ scaled) / norm))

    def perturb(self, scale: float, *, seed: int | None = None) -> "VariationalPhysicalTie":
        """Add small complex noise to escape symmetry-locked coordinate saddles."""
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("perturbation scale must be finite and nonnegative.")
        if scale == 0.0:
            return self
        rng = np.random.default_rng(seed)
        for index, factor in enumerate(self.factors):
            noise = rng.normal(size=factor.shape) + 1.0j * rng.normal(size=factor.shape)
            self.factors[index] = factor + (scale / np.sqrt(2.0)) * noise
        noise = rng.normal(size=self.terminal.shape) + 1.0j * rng.normal(size=self.terminal.shape)
        self.terminal += (scale / np.sqrt(2.0)) * noise
        self.normalize()
        self.energy = self.expectation()
        return self

    def _variable_shape(
        self,
        variable: int,
        *,
        parents: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        if variable < len(self.factors):
            if parents is None:
                return self.factors[variable].shape
            return (self.dims[variable],) + tuple(
                self.dims[parent] for parent in parents
            )
        return self.terminal.shape

    def _local_environment(self, variable: int) -> np.ndarray:
        terminal_variable = variable == len(self.factors)
        if terminal_variable:
            environment = np.ones(len(self._configs), dtype=complex)
        else:
            environment = self.terminal[self._configs[:, -1]].astype(complex, copy=True)
        for site, (factor, parents) in enumerate(zip(self.factors, self.parent_sets)):
            if site == variable:
                continue
            index = (self._configs[:, site],) + tuple(
                self._configs[:, parent] for parent in parents
            )
            environment *= factor[index]
        largest = float(np.max(np.abs(environment))) if environment.size else 0.0
        if not np.isfinite(largest) or largest == 0.0:
            raise ValueError("local physical-tie environment is numerically zero or nonfinite.")
        return environment / largest

    def local_projector(
        self,
        variable: int,
        *,
        parents=None,
        environment: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the dense linear map from one variable to the full state."""
        variable = int(variable)
        if variable < 0 or variable >= self.nvariables:
            raise IndexError("variable index out of range.")
        terminal_variable = variable == len(self.factors)
        if terminal_variable:
            if parents is not None:
                raise ValueError("the terminal variable has no physical parents.")
            selected_parents = None
        else:
            selected_parents = (
                self.parent_sets[variable]
                if parents is None
                else _validate_parent_set(variable, parents, len(self.dims))
            )
        shape = self._variable_shape(variable, parents=selected_parents)
        dtype = np.result_type(self.hamiltonian.dtype, self.terminal.dtype, complex)
        projector = np.zeros((int(np.prod(self.dims)), int(np.prod(shape))), dtype=dtype)
        if environment is None:
            environment = self._local_environment(variable)
        else:
            environment = np.asarray(environment)
            if environment.shape != (len(self._configs),):
                raise ValueError("local environment shape is inconsistent with dims.")
        if terminal_variable:
            local_index = (self._configs[:, -1],)
        else:
            local_index = (self._configs[:, variable],) + tuple(
                self._configs[:, parent] for parent in selected_parents
            )
        columns = np.ravel_multi_index(local_index, shape)
        projector[np.arange(len(self._configs)), columns] = environment
        return projector

    def _solve_local_projector(
        self,
        projector: np.ndarray,
        *,
        metric_tol: float,
        h_projector: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, int]:
        """Solve one local RR problem using the diagonal physical-context metric."""
        weights = np.sum(np.abs(projector) ** 2, axis=0).real
        threshold = metric_tol * float(np.sum(weights))
        support = np.flatnonzero(weights > threshold)
        if support.size == 0:
            raise ValueError("effective overlap metric is numerically singular.")
        if h_projector is None:
            h_projector = self.hamiltonian @ projector
        heff = projector.T.conj() @ h_projector
        inverse_norms = 1.0 / np.sqrt(weights[support])
        reduced = heff[np.ix_(support, support)]
        reduced = inverse_norms[:, None] * reduced * inverse_norms[None, :]
        reduced = 0.5 * (reduced + reduced.T.conj())
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        vector = np.zeros(projector.shape[1], dtype=projector.dtype)
        vector[support] = inverse_norms * eigenvectors[:, 0]
        return float(np.real(eigenvalues[0])), vector, int(support.size)

    def optimize_variable(self, variable: int, *, metric_tol: float = 1.0e-12) -> dict:
        """Minimize the energy over one factor with all other factors fixed."""
        variable = int(variable)
        if variable < 0 or variable >= self.nvariables:
            raise IndexError("variable index out of range.")
        metric_tol = float(metric_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        energy_before = self.expectation()
        old_terminal = self.terminal.copy()
        old_factors = [factor.copy() for factor in self.factors]
        projector = self.local_projector(variable)
        local_energy, vector, metric_rank = self._solve_local_projector(
            projector,
            metric_tol=metric_tol,
        )
        if variable < len(self.factors):
            self.factors[variable] = vector.reshape(self.factors[variable].shape)
        else:
            self.terminal = vector.reshape(self.terminal.shape)
        accepted = True
        try:
            self._balance_scalar_gauges()
            self.normalize()
            energy_after = self.expectation()
        except ValueError:
            accepted = False
            energy_after = energy_before
        if not np.isfinite(energy_after) or energy_after > energy_before:
            accepted = False
        if not accepted:
            self.terminal = old_terminal
            self.factors = old_factors
            energy_after = energy_before
        self.energy = energy_after
        return {
            "variable": variable,
            "local_energy": float(local_energy),
            "energy_before": float(energy_before),
            "energy": float(energy_after),
            "accepted": bool(accepted),
            "raw_dim": int(projector.shape[1]),
            "metric_rank": metric_rank,
        }

    def _parent_rr_trials(
        self,
        site: int,
        candidates: list[tuple[int, ...]],
        *,
        metric_tol: float,
    ) -> list[dict]:
        environment = self._local_environment(site)
        projectors = [
            self.local_projector(site, parents=parents, environment=environment)
            for parents in candidates
        ]
        widths = [projector.shape[1] for projector in projectors]
        combined = np.concatenate(projectors, axis=1)
        h_combined = self.hamiltonian @ combined

        trials = []
        offset = 0
        for parents, projector, width in zip(candidates, projectors, widths):
            h_projector = h_combined[:, offset : offset + width]
            offset += width
            try:
                local_energy, vector, metric_rank = self._solve_local_projector(
                    projector,
                    metric_tol=metric_tol,
                    h_projector=h_projector,
                )
            except (ValueError, np.linalg.LinAlgError):
                trials.append(
                    {
                        "parents": parents,
                        "energy": np.inf,
                        "raw_dim": width,
                        "metric_rank": 0,
                        "valid": False,
                    }
                )
                continue
            trials.append(
                {
                    "parents": parents,
                    "energy": local_energy,
                    "vector": vector,
                    "raw_dim": width,
                    "metric_rank": metric_rank,
                    "valid": True,
                }
            )
        return trials

    def optimize_parent_set(
        self,
        site: int,
        candidate_parent_sets,
        *,
        metric_tol: float = 1.0e-12,
        graph_tol: float = 1.0e-10,
    ) -> dict:
        """Select one site's physical parents by local Rayleigh--Ritz energy."""
        site = int(site)
        if site < 0 or site >= len(self.factors):
            raise IndexError("site index must refer to a nonterminal factor.")
        metric_tol = float(metric_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        graph_tol = float(graph_tol)
        if not np.isfinite(graph_tol) or graph_tol < 0.0:
            raise ValueError("graph_tol must be finite and nonnegative.")

        parents_before = self.parent_sets[site]
        candidates = [parents_before]
        candidates.extend(
            _validate_parent_set(site, parents, len(self.dims))
            for parents in candidate_parent_sets
        )
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise ValueError("candidate_parent_sets must not be empty.")

        energy_before = self.expectation()
        old_terminal = self.terminal.copy()
        old_factors = [factor.copy() for factor in self.factors]
        old_parent_sets = self.parent_sets
        trials = self._parent_rr_trials(
            site,
            candidates,
            metric_tol=metric_tol,
        )

        valid_trials = [trial for trial in trials if trial["valid"]]
        if not valid_trials:
            raise ValueError("all candidate parent sets have singular overlap metrics.")
        current_trial = next(
            (
                trial
                for trial in valid_trials
                if trial["parents"] == parents_before
            ),
            None,
        )
        alternate_trials = [
            trial for trial in valid_trials if trial["parents"] != parents_before
        ]
        chosen = None
        if alternate_trials:
            best_alternate = min(
                alternate_trials,
                key=lambda trial: (trial["energy"], trial["parents"]),
            )
            reference_energy = (
                energy_before
                if current_trial is None
                else float(current_trial["energy"])
            )
            comparison_scale = max(
                1.0,
                abs(reference_energy),
                abs(float(best_alternate["energy"])),
            )
            if (
                best_alternate["energy"]
                < reference_energy - graph_tol * comparison_scale
            ):
                chosen = best_alternate

        public_trials = [
            {key: value for key, value in trial.items() if key != "vector"}
            for trial in trials
        ]
        if chosen is None:
            self.energy = energy_before
            return {
                "site": site,
                "parents_before": parents_before,
                "parents": parents_before,
                "graph_changed": False,
                "energy_before": float(energy_before),
                "energy": float(energy_before),
                "accepted": False,
                "candidate_count": len(candidates),
                "valid_candidates": len(valid_trials),
                "trials": public_trials,
            }

        chosen_parents = chosen["parents"]
        parent_sets = list(self.parent_sets)
        parent_sets[site] = chosen_parents
        self.parent_sets = tuple(parent_sets)
        shape = self._variable_shape(site, parents=chosen_parents)
        self.factors[site] = chosen["vector"].reshape(shape)
        accepted = True
        try:
            self.normalize()
            energy_after = self.expectation()
        except ValueError:
            accepted = False
            energy_after = energy_before

        graph_changed = chosen_parents != parents_before
        scale = max(1.0, abs(energy_before))
        if not np.isfinite(energy_after) or energy_after > energy_before:
            accepted = False
        if graph_changed and energy_after >= energy_before - graph_tol * scale:
            accepted = False
        if not accepted:
            self.terminal = old_terminal
            self.factors = old_factors
            self.parent_sets = old_parent_sets
            energy_after = energy_before
            graph_changed = False
            chosen_parents = parents_before
        self.energy = energy_after

        return {
            "site": site,
            "parents_before": parents_before,
            "parents": chosen_parents,
            "graph_changed": bool(graph_changed),
            "energy_before": float(energy_before),
            "energy": float(energy_after),
            "accepted": bool(accepted),
            "candidate_count": len(candidates),
            "valid_candidates": len(valid_trials),
            "trials": public_trials,
        }

    def _parent_candidates(
        self,
        site: int,
        max_parents: int,
    ) -> tuple[tuple[int, ...], ...]:
        future = tuple(range(site + 1, len(self.dims)))
        cardinality = min(max_parents, len(future))
        return tuple(
            parents
            for size in range(cardinality + 1)
            for parents in combinations(future, size)
        )

    def _relaxed_parent_branch(
        self,
        site: int,
        trial: dict,
        *,
        tensor_sweeps: int,
    ) -> tuple["VariationalPhysicalTie", float]:
        branch = self.copy()
        parents = trial["parents"]
        parent_sets = list(branch.parent_sets)
        parent_sets[site] = parents
        branch.parent_sets = tuple(parent_sets)
        shape = branch._variable_shape(site, parents=parents)
        branch.factors[site] = trial["vector"].reshape(shape).copy()
        branch.normalize()
        forced_energy = branch.expectation()
        branch.energy = forced_energy
        if tensor_sweeps:
            branch.run(
                nsweeps=tensor_sweeps,
                tol=0.0,
                noise=0.0,
            )
        return branch, float(forced_energy)

    def optimize_parent_graph_relaxed(
        self,
        max_parents: int,
        *,
        candidate_budget: int = 6,
        per_site_candidates: int = 2,
        tensor_sweeps: int = 2,
        metric_tol: float = 1.0e-12,
        graph_tol: float = 1.0e-10,
    ) -> dict:
        """Try locally screened parent moves after independent tensor relaxation."""
        max_parents = int(max_parents)
        candidate_budget = int(candidate_budget)
        per_site_candidates = int(per_site_candidates)
        tensor_sweeps = int(tensor_sweeps)
        if max_parents < 0:
            raise ValueError("max_parents must be nonnegative.")
        if any(len(parents) > max_parents for parents in self.parent_sets):
            raise ValueError("the current graph exceeds max_parents.")
        if candidate_budget < 1:
            raise ValueError("candidate_budget must be positive.")
        if per_site_candidates < 1:
            raise ValueError("per_site_candidates must be positive.")
        if tensor_sweeps < 0:
            raise ValueError("tensor_sweeps must be nonnegative.")
        metric_tol = float(metric_tol)
        graph_tol = float(graph_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if not np.isfinite(graph_tol) or graph_tol < 0.0:
            raise ValueError("graph_tol must be finite and nonnegative.")

        energy_before = self.expectation()
        screened = []
        screened_count = 0
        for site in range(len(self.factors)):
            parents_before = self.parent_sets[site]
            candidates = list(
                dict.fromkeys(
                    [parents_before, *self._parent_candidates(site, max_parents)]
                )
            )
            trials = self._parent_rr_trials(
                site,
                candidates,
                metric_tol=metric_tol,
            )
            current = next(
                (
                    trial
                    for trial in trials
                    if trial["parents"] == parents_before and trial["valid"]
                ),
                None,
            )
            reference_energy = energy_before if current is None else current["energy"]
            alternatives = []
            for trial in trials:
                if not trial["valid"] or trial["parents"] == parents_before:
                    continue
                screened_count += 1
                candidate = {
                    **trial,
                    "site": site,
                    "parents_before": parents_before,
                    "screen_delta": float(trial["energy"] - reference_energy),
                }
                alternatives.append(candidate)
            alternatives.sort(
                key=lambda trial: (
                    trial["screen_delta"],
                    trial["energy"],
                    trial["parents"],
                )
            )
            screened.extend(alternatives[:per_site_candidates])

        screened.sort(
            key=lambda trial: (
                trial["screen_delta"],
                trial["energy"],
                trial["site"],
                trial["parents"],
            )
        )
        selected = screened[:candidate_budget]

        control = self.copy()
        if tensor_sweeps:
            control.run(nsweeps=tensor_sweeps, tol=0.0, noise=0.0)
        control_energy = float(control.energy)
        branch_records = []
        valid_branches = []
        for trial in selected:
            public = {
                key: value
                for key, value in trial.items()
                if key != "vector"
            }
            try:
                branch, forced_energy = self._relaxed_parent_branch(
                    trial["site"],
                    trial,
                    tensor_sweeps=tensor_sweeps,
                )
                relaxed_energy = float(branch.energy)
            except (ValueError, np.linalg.LinAlgError):
                branch_records.append(
                    {
                        **public,
                        "forced_energy": np.inf,
                        "relaxed_energy": np.inf,
                        "valid_branch": False,
                    }
                )
                continue
            record = {
                **public,
                "forced_energy": forced_energy,
                "relaxed_energy": relaxed_energy,
                "valid_branch": True,
            }
            branch_records.append(record)
            valid_branches.append((relaxed_energy, trial, branch, record))

        accepted = False
        chosen_site = None
        chosen_parents = None
        if valid_branches:
            _best_energy, best_trial, best_branch, _record = min(
                valid_branches,
                key=lambda item: (
                    item[0],
                    item[1]["site"],
                    item[1]["parents"],
                ),
            )
            validated_energy = best_branch.expectation()
            comparison_scale = max(
                1.0,
                abs(energy_before),
                abs(control_energy),
                abs(validated_energy),
            )
            threshold = graph_tol * comparison_scale
            if (
                validated_energy < energy_before - threshold
                and validated_energy < control_energy - threshold
            ):
                self.factors = [factor.copy() for factor in best_branch.factors]
                self.terminal = best_branch.terminal.copy()
                self.parent_sets = best_branch.parent_sets
                self.energy = float(validated_energy)
                self.history = list(best_branch.history)
                self.converged = False
                accepted = True
                chosen_site = int(best_trial["site"])
                chosen_parents = best_trial["parents"]

        if not accepted:
            self.energy = energy_before
        return {
            "energy_before": float(energy_before),
            "control_energy": control_energy,
            "energy": float(self.energy),
            "accepted": accepted,
            "graph_changed": accepted,
            "site": chosen_site,
            "parents": chosen_parents,
            "screened_candidates": screened_count,
            "trial_candidates": len(selected),
            "branches": branch_records,
        }

    def run_relaxed_parent_search(
        self,
        max_parents: int,
        *,
        nsweeps: int = 2,
        candidate_budget: int = 6,
        per_site_candidates: int = 2,
        tensor_sweeps: int = 2,
        metric_tol: float = 1.0e-12,
        graph_tol: float = 1.0e-10,
        verbose: bool = False,
    ) -> "VariationalPhysicalTie":
        """Repeat relaxed parent trials until no improving branch remains."""
        max_parents = int(max_parents)
        nsweeps = int(nsweeps)
        candidate_budget = int(candidate_budget)
        per_site_candidates = int(per_site_candidates)
        tensor_sweeps = int(tensor_sweeps)
        metric_tol = float(metric_tol)
        graph_tol = float(graph_tol)
        if max_parents < 0:
            raise ValueError("max_parents must be nonnegative.")
        if any(len(parents) > max_parents for parents in self.parent_sets):
            raise ValueError("the current graph exceeds max_parents.")
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        if candidate_budget < 1:
            raise ValueError("candidate_budget must be positive.")
        if per_site_candidates < 1:
            raise ValueError("per_site_candidates must be positive.")
        if tensor_sweeps < 0:
            raise ValueError("tensor_sweeps must be nonnegative.")
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if not np.isfinite(graph_tol) or graph_tol < 0.0:
            raise ValueError("graph_tol must be finite and nonnegative.")
        original_energy = self.expectation()
        original_factors = [factor.copy() for factor in self.factors]
        original_terminal = self.terminal.copy()
        original_parent_sets = self.parent_sets
        original_history = list(self.history)
        original_relaxed_history = list(self.relaxed_graph_history)
        original_converged = self.converged
        self.relaxed_graph_history = []
        self.converged = False
        try:
            for sweep in range(nsweeps):
                update = self.optimize_parent_graph_relaxed(
                    max_parents,
                    candidate_budget=candidate_budget,
                    per_site_candidates=per_site_candidates,
                    tensor_sweeps=tensor_sweeps,
                    metric_tol=metric_tol,
                    graph_tol=graph_tol,
                )
                self.relaxed_graph_history.append(
                    {
                        "sweep": sweep,
                        **update,
                    }
                )
                if verbose:
                    print(
                        f"relaxed-parent sweep={sweep:2d} "
                        f"E={self.energy: .12f} accepted={update['accepted']}"
                    )
                if not update["accepted"]:
                    self.converged = True
                    break
        except Exception:
            self.factors = original_factors
            self.terminal = original_terminal
            self.parent_sets = original_parent_sets
            self.energy = original_energy
            self.history = original_history
            self.relaxed_graph_history = original_relaxed_history
            self.converged = original_converged
            raise
        return self

    def run_parent_search(
        self,
        max_parents: int,
        *,
        nsweeps: int = 2,
        tensor_sweeps: int = 1,
        tol: float = 1.0e-10,
        metric_tol: float = 1.0e-12,
        graph_tol: float = 1.0e-10,
        verbose: bool = False,
    ) -> "VariationalPhysicalTie":
        """Alternate exhaustive RR parent selection and fixed-graph sweeps."""
        max_parents = int(max_parents)
        nsweeps = int(nsweeps)
        tensor_sweeps = int(tensor_sweeps)
        if max_parents < 0:
            raise ValueError("max_parents must be nonnegative.")
        if any(len(parents) > max_parents for parents in self.parent_sets):
            raise ValueError("the current graph exceeds max_parents.")
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        if tensor_sweeps < 0:
            raise ValueError("tensor_sweeps must be nonnegative.")
        tol = float(tol)
        metric_tol = float(metric_tol)
        graph_tol = float(graph_tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if not np.isfinite(graph_tol) or graph_tol < 0.0:
            raise ValueError("graph_tol must be finite and nonnegative.")

        original_energy = self.expectation()
        original_factors = [factor.copy() for factor in self.factors]
        original_terminal = self.terminal.copy()
        original_parent_sets = self.parent_sets
        tensor_order = list(range(self.nvariables)) + list(
            reversed(range(self.nvariables - 1))
        )
        self.graph_history = []
        self.converged = False
        previous = original_energy

        try:
            for sweep in range(nsweeps):
                sites = list(range(len(self.factors)))
                if sweep % 2:
                    sites.reverse()
                graph_updates = [
                    self.optimize_parent_set(
                        site,
                        self._parent_candidates(site, max_parents),
                        metric_tol=metric_tol,
                        graph_tol=graph_tol,
                    )
                    for site in sites
                ]
                tensor_updates = []
                for _ in range(tensor_sweeps):
                    tensor_updates.append(
                        [
                            self.optimize_variable(variable, metric_tol=metric_tol)
                            for variable in tensor_order
                        ]
                    )
                energy = self.expectation()
                delta = abs(energy - previous)
                graph_changes = sum(
                    update["graph_changed"] for update in graph_updates
                )
                self.graph_history.append(
                    {
                        "sweep": sweep,
                        "energy": energy,
                        "delta_energy": delta,
                        "graph_changes": graph_changes,
                        "graph_updates": graph_updates,
                        "tensor_updates": tensor_updates,
                    }
                )
                self.energy = energy
                if verbose:
                    print(
                        f"parent-RR sweep={sweep:2d} E={energy: .12f} "
                        f"dE={delta:.3e} changes={graph_changes}"
                    )
                if graph_changes == 0 and delta <= tol:
                    self.converged = True
                    break
                previous = energy
        except Exception:
            self.factors = original_factors
            self.terminal = original_terminal
            self.parent_sets = original_parent_sets
            self.energy = original_energy
            self.graph_history = []
            self.converged = False
            raise

        rollback_scale = max(1.0, abs(original_energy))
        if self.energy > original_energy + 64.0 * np.finfo(float).eps * rollback_scale:
            self.factors = original_factors
            self.terminal = original_terminal
            self.parent_sets = original_parent_sets
            self.energy = original_energy
            self.converged = False
            if self.graph_history:
                self.graph_history[-1]["rolled_back"] = True
                self.graph_history[-1]["energy"] = original_energy
                self.graph_history[-1]["delta_energy"] = 0.0
        return self

    def run(
        self,
        *,
        nsweeps: int = 4,
        tol: float = 1.0e-10,
        metric_tol: float = 1.0e-12,
        noise: float = 0.0,
        seed: int | None = None,
        verbose: bool = False,
    ) -> "VariationalPhysicalTie":
        """Run alternating dense one-variable variational sweeps."""
        nsweeps = int(nsweeps)
        if nsweeps < 0:
            raise ValueError("nsweeps must be nonnegative.")
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        metric_tol = float(metric_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        original_energy = self.expectation()
        original_factors = [factor.copy() for factor in self.factors]
        original_terminal = self.terminal.copy()
        if nsweeps > 0:
            self.perturb(noise, seed=seed)
        order = list(range(self.nvariables)) + list(reversed(range(self.nvariables - 1)))
        previous = self.expectation()
        self.energy = previous
        self.history = []
        self.converged = False
        for sweep in range(nsweeps):
            updates = [
                self.optimize_variable(variable, metric_tol=metric_tol)
                for variable in order
            ]
            energy = self.expectation()
            delta = abs(energy - previous)
            self.history.append(
                {
                    "sweep": sweep,
                    "energy": energy,
                    "delta_energy": delta,
                    "updates": updates,
                }
            )
            if verbose:
                print(f"physical-tie sweep={sweep:2d} E={energy: .12f} dE={delta:.3e}")
            self.energy = energy
            if delta <= tol:
                self.converged = True
                break
            previous = energy
        if self.energy > original_energy + 1.0e-11 * max(1.0, abs(original_energy)):
            self.factors = original_factors
            self.terminal = original_terminal
            self.energy = original_energy
            self.converged = False
            if self.history:
                self.history[-1]["rolled_back"] = True
                self.history[-1]["energy"] = original_energy
                self.history[-1]["delta_energy"] = 0.0
        return self


def _context_rank_one_factor(
    residual: np.ndarray,
    *,
    site: int,
    parents: tuple[int, ...],
    dims: tuple[int, ...],
):
    """Return the optimal conditional rank-one factor for fixed parents."""
    local_dim = dims[site]
    suffix_shape = dims[site + 1 :]
    expected = (local_dim,) + suffix_shape
    if residual.shape != expected:
        raise ValueError(f"residual shape {residual.shape} does not match {expected}.")

    parent_offsets = tuple(parent - site - 1 for parent in parents)
    context_shape = tuple(dims[parent] for parent in parents)
    factor = np.zeros((local_dim,) + context_shape, dtype=residual.dtype)
    next_residual = np.zeros(suffix_shape, dtype=residual.dtype)

    suffixes = list(np.ndindex(*suffix_shape)) if suffix_shape else [()]
    groups: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for suffix in suffixes:
        context = tuple(suffix[offset] for offset in parent_offsets)
        groups.setdefault(context, []).append(suffix)

    discarded = 0.0
    for context, columns in groups.items():
        matrix = np.column_stack([residual[(slice(None),) + suffix] for suffix in columns])
        if not np.any(matrix):
            conditional = np.zeros(local_dim, dtype=residual.dtype)
            conditional[0] = 1.0
            coefficients = np.zeros(len(columns), dtype=residual.dtype)
        else:
            conditional, coefficients, context_discarded = _principal_gram_factor(matrix)
            discarded += context_discarded
        factor[(slice(None),) + context] = conditional
        for suffix, coefficient in zip(columns, coefficients):
            next_residual[suffix] = coefficient

    input_weight = float(np.vdot(residual, residual).real)
    retained_weight = float(np.vdot(next_residual, next_residual).real)
    roundoff = input_weight - retained_weight - discarded
    if abs(roundoff) > 1.0e-10 * max(1.0, input_weight):
        raise RuntimeError("conditional SVD weights do not close numerically.")
    return factor, next_residual, discarded, input_weight, retained_weight


def compress_physical_ties(
    state,
    dims,
    *,
    parent_sets=None,
    max_parents: int | None = None,
    relative_tolerance: float = 0.0,
    mandatory_nearest: bool = False,
) -> PhysicalTieState:
    """Compress ``state`` into a physical-context conditional product.

    Exactly one of ``parent_sets`` and ``max_parents`` must be supplied.
    With ``parent_sets``, those physical contexts are used directly.  With
    ``max_parents``, sites are added greedily to minimize the local discarded
    Gram weight until the budget or ``relative_tolerance`` is reached.
    """
    dims = _validate_dims(dims)
    nsites = len(dims)
    vector = _normalized_vector(state, int(np.prod(dims)))
    residual = vector.reshape(dims)
    relative_tolerance = float(relative_tolerance)
    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be nonnegative.")
    if (parent_sets is None) == (max_parents is None):
        raise ValueError("supply exactly one of parent_sets or max_parents.")

    if parent_sets is not None:
        if len(parent_sets) != nsites - 1:
            raise ValueError("parent_sets must contain one entry per nonterminal site.")
        requested = tuple(
            _validate_parent_set(site, parents, nsites)
            for site, parents in enumerate(parent_sets)
        )
    else:
        max_parents = int(max_parents)
        if max_parents < 0:
            raise ValueError("max_parents must be nonnegative.")
        if mandatory_nearest and max_parents < 1:
            raise ValueError("mandatory_nearest requires max_parents to be at least one.")
        requested = None

    factors = []
    selected_parent_sets = []
    steps = []
    for site in range(nsites - 1):
        if requested is not None:
            parents = requested[site]
            result = _context_rank_one_factor(
                residual, site=site, parents=parents, dims=dims
            )
        else:
            parents = (site + 1,) if mandatory_nearest else ()
            cache = {}

            def evaluate(candidate_parents):
                candidate_parents = tuple(sorted(candidate_parents))
                if candidate_parents not in cache:
                    cache[candidate_parents] = _context_rank_one_factor(
                        residual,
                        site=site,
                        parents=candidate_parents,
                        dims=dims,
                    )
                return cache[candidate_parents]

            result = evaluate(parents)
            while len(parents) < max_parents:
                discarded = result[2]
                input_weight = result[3]
                relative = 0.0 if input_weight <= 0.0 else discarded / input_weight
                if relative <= relative_tolerance:
                    break
                candidates = [
                    parent
                    for parent in range(site + 1, nsites)
                    if parent not in parents
                ]
                if not candidates:
                    break
                trials = []
                current_context_count = int(np.prod([dims[parent] for parent in parents]))
                for parent in candidates:
                    trial_parents = tuple(sorted(parents + (parent,)))
                    trial = evaluate(trial_parents)
                    gain = discarded - trial[2]
                    added_parameters = dims[site] * current_context_count * (dims[parent] - 1)
                    score = gain / max(1, added_parameters)
                    trials.append((score, gain, parent, trial_parents, trial))
                _score, improvement, _parent, best_parents, best_result = max(
                    trials, key=lambda item: (item[0], item[1], -item[2])
                )
                relative_improvement = 0.0 if input_weight <= 0.0 else improvement / input_weight
                if relative_improvement <= 64.0 * np.finfo(float).eps:
                    break
                parents = best_parents
                result = best_result

        factor, next_residual, discarded, input_weight, retained_weight = result
        factors.append(factor)
        selected_parent_sets.append(parents)
        steps.append(
            PhysicalTieStep(
                site=site,
                parents=parents,
                discarded_weight=discarded,
                input_weight=input_weight,
                retained_weight=retained_weight,
            )
        )
        residual = next_residual

    return PhysicalTieState(
        dims=dims,
        factors=tuple(factors),
        parent_sets=tuple(selected_parent_sets),
        terminal=np.asarray(residual),
        steps=tuple(steps),
    )


__all__ = [
    "PhysicalTieState",
    "PhysicalTieStep",
    "VariationalPhysicalTie",
    "compress_physical_ties",
    "fixed_range_parent_sets",
]
