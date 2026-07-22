"""Current-state-driven adaptive physical-tie graphs for frontier LETTA.

The routines in this module change only the physical dependency graph.  They
do not require a dense target state or an exact ground state.  Candidate ties
are ranked from samples of the current wavefunction using

* a connected configuration-correlation signal, and
* the non-additive pair component of the current local-energy residual.

The ranking includes an explicit penalty for the exact tie-frontier width.
Shortlisted graph changes are built as independent ``FrontierTiedLETTA``
states, relaxed for a small number of variational sweeps, and accepted only
after a fresh global energy contraction shows no increase.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import log

import numpy as np

from .cp_tying import _validated_dims, _validated_parent_sets
from .frontier_tying import FrontierTiedLETTA
from .vmc import LETTAVMC


@dataclass(frozen=True)
class TieFrontierCut:
    """Tie-frontier cost proxy at one chain cut."""

    cut: int
    frontier_sites: tuple[int, ...]
    width: int
    physical_states: int
    norm_elements: int


@dataclass(frozen=True)
class TieGraphCost:
    """Exact physical-frontier profile, excluding fixed MPO channel factors."""

    peak_width: int
    peak_physical_states: int
    peak_norm_elements: int
    total_norm_elements: int
    log_objective: float
    cuts: tuple[TieFrontierCut, ...]


@dataclass(frozen=True)
class TieSignal:
    """Current-state evidence that one physical pair deserves a direct tie."""

    edge: tuple[int, int]
    connected_correlation: float
    residual_coupling: float
    score: float


@dataclass(frozen=True)
class TieSignalBatch:
    """Pair signals and sampling diagnostics from one current state."""

    signals: tuple[TieSignal, ...]
    nsamples: int
    mean_local_energy: complex
    local_energy_variance: float
    acceptance_rate: float


@dataclass(frozen=True)
class TieGraphProposal:
    """One add/remove proposal before variational relaxation."""

    operation: str
    edge: tuple[int, int]
    parent_sets: tuple[tuple[int, ...], ...]
    signal: TieSignal
    cost_before: TieGraphCost
    cost_after: TieGraphCost
    delta_log_cost: float
    proxy_utility: float


@dataclass(frozen=True)
class TieGraphEvaluation:
    """Energy and complexity diagnostics for a relaxed graph proposal."""

    proposal: TieGraphProposal
    energy_before: float
    migrated_energy: float
    energy_after: float
    energy_gain: float
    penalized_gain: float
    fresh_energy_check_passed: bool
    accepted: bool
    relaxation_energies: tuple[float, ...]
    candidate_state: FrontierTiedLETTA = field(repr=False, compare=False)


@dataclass(frozen=True)
class AdaptiveTieGraphStep:
    """One sample--propose--relax graph-selection step."""

    state: FrontierTiedLETTA = field(repr=False, compare=False)
    signal_batch: TieSignalBatch
    proposals: tuple[TieGraphProposal, ...]
    evaluations: tuple[TieGraphEvaluation, ...]
    selected: TieGraphEvaluation | None


@dataclass(frozen=True)
class AdaptiveTieGraphRun:
    """Result of several adaptive tie-graph steps."""

    state: FrontierTiedLETTA = field(repr=False, compare=False)
    steps: tuple[AdaptiveTieGraphStep, ...]


def _canonical_edge(edge, nsites: int) -> tuple[int, int]:
    if len(edge) != 2:
        raise ValueError("an edge must contain exactly two sites.")
    left, right = sorted((int(edge[0]), int(edge[1])))
    if left == right or left < 0 or right >= nsites:
        raise ValueError("edges must join two distinct valid sites.")
    return left, right


def tie_edges(parent_sets) -> tuple[tuple[int, int], ...]:
    """Return canonical undirected edges represented by ``parent_sets``."""
    return tuple(
        (site, parent)
        for site, parents in enumerate(parent_sets)
        for parent in parents
    )


def tie_frontier_cost(
    dims,
    parent_sets,
    *,
    bond_dim: int = 1,
    bond_dims=None,
) -> TieGraphCost:
    r"""Return an exact cost proxy for tied variables crossing every cut.

    At cut ``k``, the retained physical frontier is

    .. math::

        F_k=\{j\ge k:\;\exists i<k,\ j\in P_i\}.

    The norm message therefore carries ``prod(d_j, j in F_k)`` physical
    configurations and two virtual indices of the exact dimension at that
    cut.  Fixed Hamiltonian-MPO channel factors are deliberately omitted: when
    comparing tie graphs for the same Hamiltonian, they do not change which
    new physical variables cross a cut.

    ``log_objective`` combines peak and accumulated storage, so a proposal is
    penalized even when it lengthens a costly frontier without changing its
    maximum width.
    """
    dims = _validated_dims(dims)
    parent_sets = _validated_parent_sets(dims, parent_sets)
    bond_dim = int(bond_dim)
    if bond_dim < 1:
        raise ValueError("bond_dim must be positive.")
    if bond_dims is None:
        virtual_dims = (1,) + (bond_dim,) * max(0, len(dims) - 1) + (1,)
    else:
        virtual_dims = tuple(int(dimension) for dimension in bond_dims)
        if len(virtual_dims) == max(0, len(dims) - 1):
            virtual_dims = (1,) + virtual_dims + (1,)
        elif len(virtual_dims) != len(dims) + 1:
            raise ValueError(
                "bond_dims must contain the internal dimensions or all "
                "dimensions including the two boundaries."
            )
        if virtual_dims[0] != 1 or virtual_dims[-1] != 1:
            raise ValueError("the boundary bond dimensions must be one.")
        if any(dimension < 1 for dimension in virtual_dims):
            raise ValueError("bond_dims must contain positive dimensions.")

    cuts = []
    for cut in range(1, len(dims)):
        frontier = tuple(
            site
            for site in range(cut, len(dims))
            if any(site in parent_sets[left] for left in range(cut))
        )
        physical_states = int(np.prod([dims[site] for site in frontier], dtype=int))
        norm_elements = int(virtual_dims[cut] ** 2 * physical_states)
        cuts.append(
            TieFrontierCut(
                cut=cut,
                frontier_sites=frontier,
                width=len(frontier),
                physical_states=physical_states,
                norm_elements=norm_elements,
            )
        )
    if cuts:
        peak_width = max(item.width for item in cuts)
        peak_states = max(item.physical_states for item in cuts)
        peak_elements = max(item.norm_elements for item in cuts)
        total_elements = sum(item.norm_elements for item in cuts)
    else:
        peak_width = 0
        peak_states = 1
        peak_elements = 1
        total_elements = 1
    log_objective = log(max(peak_elements, 1)) + log(max(total_elements, 1))
    return TieGraphCost(
        peak_width=peak_width,
        peak_physical_states=peak_states,
        peak_norm_elements=peak_elements,
        total_norm_elements=total_elements,
        log_objective=float(log_objective),
        cuts=tuple(cuts),
    )


def _validated_samples(configurations, local_energies, dims):
    dims = _validated_dims(dims)
    configurations = np.asarray(configurations, dtype=np.intp)
    local_energies = np.asarray(local_energies)
    if configurations.ndim != 2 or configurations.shape[1] != len(dims):
        raise ValueError(
            f"configurations must have shape (nsamples, {len(dims)})."
        )
    if len(configurations) < 1 or local_energies.shape != (len(configurations),):
        raise ValueError("local_energies must contain one value per sample.")
    if np.any(~np.isfinite(local_energies)):
        raise ValueError("local_energies must be finite.")
    for site, dim in enumerate(dims):
        if np.any(configurations[:, site] < 0) or np.any(
            configurations[:, site] >= dim
        ):
            raise ValueError("configuration contains an out-of-range state.")
    return configurations, local_energies, dims


def _pair_signal(configurations, local_energies, dims, edge):
    left, right = edge
    left_dim, right_dim = dims[left], dims[right]
    nsamples = len(configurations)
    flat = configurations[:, left] * right_dim + configurations[:, right]
    counts = np.bincount(flat, minlength=left_dim * right_dim).reshape(
        left_dim, right_dim
    )
    probability = counts / nsamples
    left_probability = np.sum(probability, axis=1)
    right_probability = np.sum(probability, axis=0)
    independent = np.outer(left_probability, right_probability)
    # Total-variation distance is bounded and remains meaningful when some
    # sampled categories are absent.
    connected = 0.5 * float(np.sum(np.abs(probability - independent)))

    mean_energy = np.mean(local_energies)
    centered_energy = local_energies - mean_energy
    global_scale = float(np.sqrt(np.mean(np.abs(centered_energy) ** 2)))
    tiny = np.finfo(float).tiny
    if global_scale <= tiny:
        residual = 0.0
    else:
        pair_mean = np.zeros((left_dim, right_dim), dtype=local_energies.dtype)
        left_mean = np.zeros(left_dim, dtype=local_energies.dtype)
        right_mean = np.zeros(right_dim, dtype=local_energies.dtype)
        for a in range(left_dim):
            selected = configurations[:, left] == a
            if np.any(selected):
                left_mean[a] = np.mean(local_energies[selected])
        for b in range(right_dim):
            selected = configurations[:, right] == b
            if np.any(selected):
                right_mean[b] = np.mean(local_energies[selected])
        for a in range(left_dim):
            for b in range(right_dim):
                selected = (configurations[:, left] == a) & (
                    configurations[:, right] == b
                )
                if np.any(selected):
                    pair_mean[a, b] = np.mean(local_energies[selected])
        nonadditive = pair_mean - left_mean[:, None] - right_mean[None, :] + mean_energy
        # Unobserved cells have zero probability and do not contribute.
        residual = float(
            np.sqrt(np.sum(probability * np.abs(nonadditive) ** 2)) / global_scale
        )
    return connected, residual


def graph_signals_from_samples(
    configurations,
    local_energies,
    dims,
    candidate_edges,
    *,
    correlation_weight: float = 1.0,
    residual_weight: float = 1.0,
    acceptance_rate: float = float("nan"),
) -> TieSignalBatch:
    r"""Measure correlation and residual signals from the current state.

    For a pair ``(i,j)``, the correlation signal is the total-variation
    distance between the sampled joint distribution and the product of its
    marginals.  The residual signal is the normalized RMS non-additive part

    .. math::

        \bar E_{ab}-\bar E_{a\cdot}-\bar E_{\cdot b}+\bar E

    of the current local energy conditioned on ``s_i=a, s_j=b``.  It is a
    current-state variational residual proxy, not an oracle-state comparison.
    """
    configurations, local_energies, dims = _validated_samples(
        configurations, local_energies, dims
    )
    correlation_weight = float(correlation_weight)
    residual_weight = float(residual_weight)
    if not np.isfinite(correlation_weight) or correlation_weight < 0.0:
        raise ValueError("correlation_weight must be finite and nonnegative.")
    if not np.isfinite(residual_weight) or residual_weight < 0.0:
        raise ValueError("residual_weight must be finite and nonnegative.")
    edges = tuple(
        dict.fromkeys(_canonical_edge(edge, len(dims)) for edge in candidate_edges)
    )
    signals = []
    for edge in edges:
        connected, residual = _pair_signal(
            configurations, local_energies, dims, edge
        )
        signals.append(
            TieSignal(
                edge=edge,
                connected_correlation=connected,
                residual_coupling=residual,
                score=float(
                    correlation_weight * connected + residual_weight * residual
                ),
            )
        )
    mean_energy = np.mean(local_energies)
    variance = float(np.mean(np.abs(local_energies - mean_energy) ** 2))
    return TieSignalBatch(
        signals=tuple(signals),
        nsamples=len(configurations),
        mean_local_energy=complex(mean_energy),
        local_energy_variance=variance,
        acceptance_rate=float(acceptance_rate),
    )


def sample_tie_signals(
    state: FrontierTiedLETTA,
    candidate_edges,
    *,
    nsamples: int = 1024,
    burn_in: int = 100,
    sweeps_between: int = 1,
    seed: int | None = None,
    proposal="mixed",
    exchange_probability: float = 0.5,
    correlation_weight: float = 1.0,
    residual_weight: float = 1.0,
) -> TieSignalBatch:
    """Sample current-state tie signals without constructing a dense state."""
    if not isinstance(state, FrontierTiedLETTA):
        raise TypeError("state must be a FrontierTiedLETTA.")
    vmc = LETTAVMC(
        state,
        state.hamiltonian,
        seed=seed,
        proposal=proposal,
        exchange_probability=exchange_probability,
    )
    samples = vmc.sample(
        nsamples,
        burn_in=burn_in,
        sweeps_between=sweeps_between,
    )
    return graph_signals_from_samples(
        samples.configurations,
        samples.local_energies,
        state.dims,
        candidate_edges,
        correlation_weight=correlation_weight,
        residual_weight=residual_weight,
        acceptance_rate=samples.diagnostics.acceptance_rate,
    )


def _changed_parent_sets(parent_sets, edge, operation):
    left, right = edge
    result = [set(parents) for parents in parent_sets]
    operation = str(operation).lower()
    if operation == "add":
        result[left].add(right)
    elif operation == "remove":
        result[left].discard(right)
    else:
        raise ValueError("operation must be 'add' or 'remove'.")
    return tuple(tuple(sorted(parents)) for parents in result)


def rank_tie_graph_proposals(
    state: FrontierTiedLETTA,
    signal_batch: TieSignalBatch,
    *,
    operations=("add",),
    cost_weight: float = 0.25,
    max_frontier_width: int | None = None,
) -> tuple[TieGraphProposal, ...]:
    r"""Rank graph changes by current-state signal minus frontier cost.

    For additions, ``utility = signal - lambda * Delta(log C)``.  For
    removals the signal changes sign, so a weak existing tie can be exchanged
    for a sufficient cost reduction.  The cost proxy ``C`` combines the peak
    and total exact norm-frontier storage.
    """
    if not isinstance(state, FrontierTiedLETTA):
        raise TypeError("state must be a FrontierTiedLETTA.")
    if not isinstance(signal_batch, TieSignalBatch):
        raise TypeError("signal_batch must be a TieSignalBatch.")
    operations = tuple(str(operation).lower() for operation in operations)
    if not operations or any(item not in {"add", "remove"} for item in operations):
        raise ValueError("operations must contain 'add' and/or 'remove'.")
    cost_weight = float(cost_weight)
    if not np.isfinite(cost_weight) or cost_weight < 0.0:
        raise ValueError("cost_weight must be finite and nonnegative.")
    if max_frontier_width is not None:
        max_frontier_width = int(max_frontier_width)
        if max_frontier_width < 0:
            raise ValueError("max_frontier_width must be nonnegative.")

    existing = set(tie_edges(state.parent_sets))
    cost_before = tie_frontier_cost(
        state.dims,
        state.parent_sets,
        bond_dims=state.bond_dims,
    )
    proposals = []
    for signal in signal_batch.signals:
        for operation in operations:
            if operation == "add" and signal.edge in existing:
                continue
            if operation == "remove" and signal.edge not in existing:
                continue
            parent_sets = _changed_parent_sets(
                state.parent_sets, signal.edge, operation
            )
            cost_after = tie_frontier_cost(
                state.dims,
                parent_sets,
                bond_dims=state.bond_dims,
            )
            if (
                max_frontier_width is not None
                and cost_after.peak_width > max_frontier_width
            ):
                continue
            delta_log_cost = cost_after.log_objective - cost_before.log_objective
            signed_signal = signal.score if operation == "add" else -signal.score
            utility = signed_signal - cost_weight * delta_log_cost
            proposals.append(
                TieGraphProposal(
                    operation=operation,
                    edge=signal.edge,
                    parent_sets=parent_sets,
                    signal=signal,
                    cost_before=cost_before,
                    cost_after=cost_after,
                    delta_log_cost=float(delta_log_cost),
                    proxy_utility=float(utility),
                )
            )
    return tuple(
        sorted(
            proposals,
            key=lambda item: (-item.proxy_utility, item.operation, item.edge),
        )
    )


def _constructor_options(state):
    return {
        "frontier_backend": state.frontier_backend,
        "path_optimizer": state.path_optimizer,
        "tt_max_rank": state.tt_options["max_rank"],
        "tt_rtol": state.tt_options["rtol"],
        "tt_atol": state.tt_options["atol"],
        "tt_transfer_max_rank": state.tt_options["transfer_max_rank"],
        "tt_transfer_rtol": state.tt_options["transfer_rtol"],
        "tt_transfer_atol": state.tt_options["transfer_atol"],
        "tt_absorption": state.tt_options["absorption"],
        "tt_norm_backend": state.tt_norm_backend,
        "tt_hermitize": state.tt_hermitize,
    }


def state_with_tie_graph_proposal(
    state: FrontierTiedLETTA,
    proposal: TieGraphProposal,
) -> FrontierTiedLETTA:
    """Build an independent candidate state for one graph proposal.

    Adding ``(i,j)`` repeats tensor ``i`` along its new ``s_j`` axis.  The
    represented wavefunction is therefore exactly unchanged before
    relaxation.  Removing a tie uses the uniform least-squares constant slice;
    it is not state preserving and is accepted only after the global energy
    safeguard in :func:`evaluate_tie_graph_proposal`.
    """
    # Import lazily so ``frontier_abelian`` can continue to depend on the base
    # frontier implementation without creating a module-import cycle.
    from .frontier_abelian import AbelianFrontierTiedLETTA

    if type(state) not in {FrontierTiedLETTA, AbelianFrontierTiedLETTA}:
        raise TypeError(
            "graph migration supports FrontierTiedLETTA and "
            "AbelianFrontierTiedLETTA exactly; another subclass must provide "
            "its own structure-preserving tensor migration."
        )
    if not isinstance(proposal, TieGraphProposal):
        raise TypeError("proposal must be a TieGraphProposal.")
    left, right = proposal.edge
    tensors = [tensor.copy() for tensor in state.tensors]
    old_physical = (left,) + tuple(state.parent_sets[left])
    physical_axis = 2 + old_physical.index(right) if right in old_physical else None
    if proposal.operation == "add":
        new_physical = (left,) + tuple(proposal.parent_sets[left])
        physical_axis = 2 + new_physical.index(right)
        tensors[left] = np.repeat(
            np.expand_dims(tensors[left], axis=physical_axis),
            state.dims[right],
            axis=physical_axis,
        )
    elif proposal.operation == "remove":
        if physical_axis is None:
            raise ValueError("cannot remove an absent tie.")
        tensors[left] = np.mean(tensors[left], axis=physical_axis)
    else:
        raise ValueError("proposal operation must be 'add' or 'remove'.")

    constructor_options = _constructor_options(state)
    if isinstance(state, AbelianFrontierTiedLETTA):
        constructor_options["abelian_layout"] = state.abelian_layout
    candidate = type(state)(
        state.hamiltonian,
        state.dims,
        proposal.parent_sets,
        bond_dim=state.bond_dim,
        bond_dims=state.bond_dims,
        tensors=tensors,
        **constructor_options,
    )
    # Construction may balance tensor magnitudes.  Restore the intended
    # migrated tensors exactly, as FrontierTiedLETTA.copy() does.
    candidate.tensors = [tensor.copy() for tensor in tensors]
    candidate.energy = candidate.expectation()
    candidate.history = []
    candidate.converged = False
    candidate.rng.bit_generator.state = deepcopy(state.rng.bit_generator.state)
    return candidate


def evaluate_tie_graph_proposal(
    state: FrontierTiedLETTA,
    proposal: TieGraphProposal,
    *,
    relaxation_sweeps: int = 1,
    minimum_energy_gain: float = 0.0,
    energy_cost_weight: float = 0.0,
    run_options: dict | None = None,
) -> TieGraphEvaluation:
    """Relax one graph on a copy and apply a monotone global-energy safeguard."""
    relaxation_sweeps = int(relaxation_sweeps)
    if relaxation_sweeps < 0:
        raise ValueError("relaxation_sweeps must be nonnegative.")
    minimum_energy_gain = float(minimum_energy_gain)
    energy_cost_weight = float(energy_cost_weight)
    if not np.isfinite(minimum_energy_gain) or minimum_energy_gain < 0.0:
        raise ValueError("minimum_energy_gain must be finite and nonnegative.")
    if not np.isfinite(energy_cost_weight) or energy_cost_weight < 0.0:
        raise ValueError("energy_cost_weight must be finite and nonnegative.")
    options = {"solver": "whitened"}
    if run_options is not None:
        options.update(dict(run_options))
    options.pop("nsweeps", None)

    energy_before = float(state.expectation())
    candidate = state_with_tie_graph_proposal(state, proposal)
    migrated_energy = float(candidate.expectation())
    if relaxation_sweeps:
        candidate.run(nsweeps=relaxation_sweeps, **options)
    energy_after = float(candidate.expectation())
    relaxation_energies = tuple(float(item["energy"]) for item in candidate.history)
    energy_gain = energy_before - energy_after
    penalized_gain = energy_gain - energy_cost_weight * proposal.delta_log_cost
    numerical_tolerance = 512.0 * np.finfo(float).eps * max(
        1.0, abs(energy_before)
    )
    fresh_energy_check_passed = bool(
        np.isfinite(energy_after)
        and energy_after <= energy_before + numerical_tolerance
    )
    accepted = (
        fresh_energy_check_passed
        and penalized_gain > minimum_energy_gain
    )
    return TieGraphEvaluation(
        proposal=proposal,
        energy_before=energy_before,
        migrated_energy=migrated_energy,
        energy_after=energy_after,
        energy_gain=float(energy_gain),
        penalized_gain=float(penalized_gain),
        fresh_energy_check_passed=fresh_energy_check_passed,
        accepted=bool(accepted),
        relaxation_energies=relaxation_energies,
        candidate_state=candidate,
    )


def adaptive_tie_graph_step(
    state: FrontierTiedLETTA,
    *,
    candidate_edges=None,
    signal_batch: TieSignalBatch | None = None,
    operations=("add",),
    shortlist: int = 3,
    cost_weight: float = 0.25,
    energy_cost_weight: float = 0.0,
    max_frontier_width: int | None = None,
    relaxation_sweeps: int = 1,
    minimum_energy_gain: float = 0.0,
    run_options: dict | None = None,
    nsamples: int = 1024,
    burn_in: int = 100,
    sweeps_between: int = 1,
    seed: int | None = None,
    sampler_proposal="mixed",
    exchange_probability: float = 0.5,
    correlation_weight: float = 1.0,
    residual_weight: float = 1.0,
) -> AdaptiveTieGraphStep:
    """Perform one oracle-free adaptive graph selection step."""
    from .frontier_abelian import AbelianFrontierTiedLETTA

    if type(state) not in {FrontierTiedLETTA, AbelianFrontierTiedLETTA}:
        raise TypeError(
            "adaptive graph migration supports FrontierTiedLETTA and "
            "AbelianFrontierTiedLETTA exactly."
        )
    shortlist = int(shortlist)
    if shortlist < 1:
        raise ValueError("shortlist must be positive.")
    if candidate_edges is None:
        candidate_edges = tuple(
            (left, right)
            for left in range(len(state.dims))
            for right in range(left + 1, len(state.dims))
        )
    else:
        candidate_edges = tuple(candidate_edges)
    if signal_batch is None:
        signal_batch = sample_tie_signals(
            state,
            candidate_edges,
            nsamples=nsamples,
            burn_in=burn_in,
            sweeps_between=sweeps_between,
            seed=seed,
            proposal=sampler_proposal,
            exchange_probability=exchange_probability,
            correlation_weight=correlation_weight,
            residual_weight=residual_weight,
        )
    proposals = rank_tie_graph_proposals(
        state,
        signal_batch,
        operations=operations,
        cost_weight=cost_weight,
        max_frontier_width=max_frontier_width,
    )
    evaluations = tuple(
        evaluate_tie_graph_proposal(
            state,
            proposal,
            relaxation_sweeps=relaxation_sweeps,
            minimum_energy_gain=minimum_energy_gain,
            energy_cost_weight=energy_cost_weight,
            run_options=run_options,
        )
        for proposal in proposals[:shortlist]
    )
    accepted = tuple(item for item in evaluations if item.accepted)
    selected = (
        max(accepted, key=lambda item: (item.penalized_gain, item.energy_gain))
        if accepted
        else None
    )
    selected_state = selected.candidate_state if selected is not None else state
    return AdaptiveTieGraphStep(
        state=selected_state,
        signal_batch=signal_batch,
        proposals=proposals,
        evaluations=evaluations,
        selected=selected,
    )


def adapt_tie_graph(
    state: FrontierTiedLETTA,
    *,
    nsteps: int = 1,
    seed: int | None = None,
    **step_options,
) -> AdaptiveTieGraphRun:
    """Repeat adaptive graph steps, resampling after every accepted graph."""
    nsteps = int(nsteps)
    if nsteps < 0:
        raise ValueError("nsteps must be nonnegative.")
    current = state
    steps = []
    for step in range(nsteps):
        result = adaptive_tie_graph_step(
            current,
            seed=None if seed is None else int(seed) + step,
            **step_options,
        )
        steps.append(result)
        if result.selected is None:
            break
        current = result.state
    return AdaptiveTieGraphRun(state=current, steps=tuple(steps))


__all__ = [
    "AdaptiveTieGraphRun",
    "AdaptiveTieGraphStep",
    "TieFrontierCut",
    "TieGraphCost",
    "TieGraphEvaluation",
    "TieGraphProposal",
    "TieSignal",
    "TieSignalBatch",
    "adapt_tie_graph",
    "adaptive_tie_graph_step",
    "evaluate_tie_graph_proposal",
    "graph_signals_from_samples",
    "rank_tie_graph_proposals",
    "sample_tie_signals",
    "state_with_tie_graph_proposal",
    "tie_edges",
    "tie_frontier_cost",
]
