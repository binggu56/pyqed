"""Greedy, energy-checked invariant-tie growth for reduced SU(2) LETTA.

Information-guided correlator selection: Kovyrshin and Reiher, J. Chem.
Phys. 147, 214111 (2017), https://doi.org/10.1063/1.5004693. This adaptation
uses classical mutual information of invariant-sector projectors, not full
orbital reduced-density-matrix entropy or the paper's SATNS growth rule.
Exact reduced norm contractions rank candidates; bounded variational trials
select ties. No determinant expansion, magnetic-label tying, tie removal,
or guarantee of an optimal graph is provided.
"""
from copy import deepcopy
from itertools import combinations, product
from math import prod
from numbers import Integral

import numpy as np

from pyqed.mps.nonabelian.environment import contract_chain_expectation
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from .su2_qchem import _canonical_graph, _frontiers


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _graph_cost(state, graph):
    parents = tuple(tuple(j for i, j in graph if i == site)
                    for site in range(state.nsites))
    frontiers = _frontiers(state.nsites, parents)
    assignments = tuple(prod(len(state.tie_domains[j]) for j in cut)
                        for cut in frontiers)
    parameters = sum(
        sum(block.size for block in base.data.values())
        * prod(len(state.tie_domains[j]) for j in parents[i])
        for i, base in enumerate(state._base_sites)
    )
    return parents, max(map(len, frontiers)), max(assignments), sum(assignments), parameters


def with_tie_graph(state, graph, *, max_frontier_states=None, evaluate_energy=True):
    """Copy reduced blocks across new labels, exactly preserving state.

    ``evaluate_energy=False`` prepares a checkpoint without expanding its
    frontier for contraction; the returned energy is unset until evaluated.
    """
    graph = _canonical_graph(state.nsites, graph)
    if not set(state.graph).issubset(graph):
        raise ValueError('with_tie_graph supports additions only')
    cap = _positive_int(state.max_frontier_states if max_frontier_states is None
                        else max_frontier_states, 'max_frontier_states')
    parents, _, peak, _, _ = _graph_cost(state, graph)
    if peak > cap:
        raise MemoryError(f'proposed graph has {peak} invariant frontier states; cap is {cap}')
    common = dict(graph=graph, D=state.D, base_sites=state._base_sites,
                  virtual_boundary=state.virtual_boundary,
                  compact_frontier=state.compact_frontier,
                  tie=state.tie, init='mps', workers=state.workers,
                  n_threads=state.n_threads, we_route_memory=state.we_route_memory,
                  max_frontier_states=cap, _evaluate_initial_state=False)
    if state._hamiltonian_spec is not None:
        candidate = type(state).from_integrals(**deepcopy(state._hamiltonian_spec), **common)
    else:
        candidate = type(state)(None if state._state_only else state.mpo,
                               target_sector=state.target_sector, nelec=state.nelec,
                               spin=state.spin, ecore=state.ecore, **common)
    try:
        tensors = []
        for site, tensor in enumerate(candidate.tensors):
            positions = tuple(parents[site].index(j) for j in state.parent_sets[site])
            tensors.append({key: state.tensors[site][
                key[:3] + (tuple(key[3][p] for p in positions),)
            ].copy() for key in tensor})
        candidate.tensors = tensors
        candidate._materialized_site_cache.clear()
        candidate.core_energy = state.core_energy
        candidate.ecore = state.ecore
        candidate.mpo_includes_core_energy = state.mpo_includes_core_energy
        candidate._hamiltonian_representation_policy = state._hamiltonian_representation_policy
        candidate._representation_decision = deepcopy(state._representation_decision)
        # Preserve the active runtime representation after an auto-policy switch.
        if state.hamiltonian_representation == 'mpo' and candidate.hamiltonian_representation != 'mpo':
            candidate._materialize_explicit_hamiltonian()
        candidate.history = deepcopy(state.history)
        candidate.tie_history = deepcopy(state.tie_history)
        for name in ('mf', 'mol', 'active_space', 'ncas', 'nelecas', 'ncore'):
            if hasattr(state, name):
                setattr(candidate, name, getattr(state, name))
        candidate.energy = (candidate.expectation()
                            if evaluate_energy and not state._state_only else None)
        candidate.converged = False
        candidate.success = None
        return candidate
    except BaseException:
        candidate.close()
        raise


def _sector_information(state, edges):
    """Joint probabilities from reduced invariant-sector projectors, in nats."""
    sites = state.materialize()
    identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
    norm = float(np.real(contract_chain_expectation(sites, identity)))
    if not np.isfinite(norm) or norm <= 0:
        raise FloatingPointError('invalid norm during adaptive tie ranking')
    axis = 1 if state.tie == 'physical' else 0
    scores = {}
    for i, j in edges:
        probabilities = np.zeros((len(state.tie_domains[i]), len(state.tie_domains[j])))
        for a, b in product(range(probabilities.shape[0]), range(probabilities.shape[1])):
            filtered = list(sites)
            for site, label in ((i, a), (j, b)):
                filtered[site] = sites[site].copy()
                sector = state.tie_domains[site][label]
                for key, block in filtered[site].data.items():
                    if key[axis] != sector:
                        filtered[site].data[key] = np.zeros_like(block)
            probability = float(np.real(contract_chain_expectation(filtered, identity))) / norm
            if not np.isfinite(probability) or probability < -1e-10:
                raise FloatingPointError('invalid sector probability during adaptive tie ranking')
            probabilities[a, b] = max(0., probability)
        total = probabilities.sum()
        if not np.isclose(total, 1., atol=1e-8, rtol=1e-8):
            raise FloatingPointError('sector projectors do not resolve the current state norm')
        probabilities /= total
        independent = probabilities.sum(axis=1)[:, None] * probabilities.sum(axis=0)[None, :]
        mask = probabilities > 0
        scores[i, j] = max(0., float(np.sum(probabilities[mask] *
                                          np.log(probabilities[mask] / independent[mask]))))
    return scores


def adapt_ties(state, *, candidate_edges=None, shortlist=2, relaxation_sweeps=1,
               max_frontier_width=2, max_frontier_states=None,
               max_parents=2, max_parameters=100000, cost_weight=0.05,
               minimum_energy_gain=1e-8, run_options=None):
    """Adopt at most one addition after independent reduced variational trials.

    Returns the same solver and appends proposal diagnostics to tie_history.
    See module documentation for references and fidelity. Rejected trials do
    not alter its tensors, history, or graph. Correlation scores are heuristics;
    fresh global energy gains, not these scores, decide acceptance.
    """
    if state._state_only:
        raise ValueError('adaptive ties require a Hamiltonian')
    shortlist = _positive_int(shortlist, 'shortlist')
    relaxation_sweeps = _positive_int(relaxation_sweeps, 'relaxation_sweeps')
    max_frontier_width = _positive_int(max_frontier_width, 'max_frontier_width')
    max_parents = _positive_int(max_parents, 'max_parents')
    max_parameters = _positive_int(max_parameters, 'max_parameters')
    cap = _positive_int(state.max_frontier_states if max_frontier_states is None
                        else max_frontier_states, 'max_frontier_states')
    for value, name in ((cost_weight, 'cost_weight'), (minimum_energy_gain, 'minimum_energy_gain')):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    options = dict(run_options or {})
    if set(options) & {'adaptive_ties', 'tie_options', 'checkpoint', 'checkpoint_every', 'nsweeps'}:
        raise ValueError('trial run_options cannot control graph adaptation, checkpoints, or nsweeps')
    options['reset_history'] = False
    edges = tuple(combinations(range(state.nsites), 2)) if candidate_edges is None else candidate_edges
    edges = _canonical_graph(state.nsites, edges)
    before = float(state.expectation())
    if not np.isfinite(before):
        raise FloatingPointError('nonfinite energy before tie adaptation')
    baseline = _graph_cost(state, state.graph)
    feasible = {}
    for edge in edges:
        if edge in state.graph:
            continue
        graph = tuple(sorted((*state.graph, edge)))
        cost = _graph_cost(state, graph)
        parents, width, peak, total, parameters = cost
        if (width > max_frontier_width or peak > cap or parameters > max_parameters
                or max(map(len, parents)) > max_parents):
            continue
        penalty = np.log(parameters / baseline[-1]) + np.log(total / baseline[-2])
        feasible[edge] = (graph, cost, float(penalty))
    information = _sector_information(state, feasible)
    ranked = sorted(feasible, key=lambda e: (-(information[e] - cost_weight * feasible[e][2]), e))
    record = dict(energy_before=before, graph_before=state.graph, proposals=[], accepted_edge=None)
    comparison = control_energy = before
    if ranked:
        control = with_tie_graph(state, state.graph, max_frontier_states=cap)
        try:
            control.run(nsweeps=relaxation_sweeps, **options)
            control_energy = float(control.expectation())
            if not np.isfinite(control_energy):
                raise FloatingPointError('nonfinite fixed-graph control energy')
            comparison = min(before, control_energy)
        finally:
            control.close()
    record['control_energy'] = control_energy
    record['comparison_energy'] = comparison
    best = None
    best_gain = max(float(minimum_energy_gain),
                    128 * np.finfo(float).eps * max(1., abs(comparison)))
    try:
        for edge in ranked[:shortlist]:
            graph, cost, penalty = feasible[edge]
            candidate = None
            try:
                candidate = with_tie_graph(state, graph, max_frontier_states=cap)
                migrated = float(candidate.expectation())
                if not np.isclose(migrated, before, atol=1e-9, rtol=1e-11):
                    raise FloatingPointError('tie migration changed the energy')
                candidate.run(nsweeps=relaxation_sweeps, **options)
                after = float(candidate.expectation())
                gain = comparison - after
                accepted = np.isfinite(after) and gain > best_gain
                record['proposals'].append(dict(
                    edge=edge, sector_information=information[edge], cost_penalty=penalty,
                    frontier_width=cost[1], frontier_states=cost[2], nparameters=cost[-1],
                    migrated_energy=migrated, energy_after=after, energy_gain=before - after,
                    tie_energy_gain=gain, status='evaluated',
                    solver_summary=candidate.convergence_summary,
                ))
                if accepted:
                    if best is not None:
                        best.close()
                    best, best_gain = candidate, gain
                    candidate = None
                    record['accepted_edge'] = edge
            except MemoryError as error:
                record['proposals'].append(dict(
                    edge=edge, sector_information=information[edge], cost_penalty=penalty,
                    frontier_width=cost[1], frontier_states=cost[2], nparameters=cost[-1],
                    energy_after=None, status='memory_limit', error=str(error),
                ))
            finally:
                if candidate is not None:
                    candidate.close()
        if best is not None:
            fresh = float(best.expectation())
            if not np.isfinite(fresh) or comparison - fresh <= max(
                    float(minimum_energy_gain),
                    128 * np.finfo(float).eps * max(1., abs(comparison))):
                raise FloatingPointError('accepted tie failed fresh energy verification')
            best.energy = fresh
            best.converged = False
            best.success = False
            best.message = 'accepted adaptive invariant tie; further cycles required'
            state.close()
            state.__dict__.update(best.__dict__)
            best._solver_executor = None
        state.tie_history.append(record)
        return state
    finally:
        if best is not None:
            best.close()


def run_adaptive(state, *, nsweeps, tie_options=None, run_options=None):
    """Alternate ordinary cycles and bounded graph trials on the same solver."""
    nsweeps = _positive_int(nsweeps, 'nsweeps')
    ties = dict(tie_options or {})
    every = _positive_int(ties.pop('every', 2), 'every')
    options = dict(run_options or {})
    checkpoint = options.pop('checkpoint', None)
    options.pop('checkpoint_every', None)
    reset = options.pop('reset_history', True)
    trial_options = dict(ties.pop('run_options', options))
    for start in range(0, nsweeps, every):
        state.run(nsweeps=min(every, nsweeps - start), reset_history=reset and start == 0,
                  **options)
        state.adapt_ties(run_options=trial_options, **ties)
        if checkpoint is not None:
            state.save_checkpoint(checkpoint)
    return state
