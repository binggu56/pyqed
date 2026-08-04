"""Site-order optimization for physically tied Heisenberg graphs."""

from __future__ import annotations

import numpy as np


def _validated_edges(nsites, edges, *, weighted):
    result = []
    for edge in edges:
        if weighted:
            if len(edge) != 3:
                raise ValueError("weighted edges must be (left, right, coupling).")
            left, right, coupling = edge
            coupling = float(coupling)
            if not np.isfinite(coupling):
                raise ValueError("edge couplings must be finite.")
        else:
            if len(edge) != 2:
                raise ValueError("tie edges must be (left, right).")
            left, right = edge
            coupling = 1.0
        left, right = int(left), int(right)
        if left == right or min(left, right) < 0 or max(left, right) >= nsites:
            raise ValueError("edges must join distinct valid sites.")
        result.append((left, right, coupling))
    return tuple(result)


def _heisenberg_graph_data(nsites, tie_edges, weighted_bonds):
    tie_neighbors = [0] * nsites
    for left, right, _coupling in tie_edges:
        tie_neighbors[left] |= 1 << right
        tie_neighbors[right] |= 1 << left

    coupling = np.zeros((nsites, nsites), dtype=float)
    for left, right, value in weighted_bonds:
        coupling[left, right] += value
        coupling[right, left] += value
    interactions = tuple(
        (left, right)
        for left in range(nsites)
        for right in range(left + 1, nsites)
        if coupling[left, right] != 0.0
    )
    interaction_neighbors = [0] * nsites
    for left, right in interactions:
        interaction_neighbors[left] |= 1 << right
        interaction_neighbors[right] |= 1 << left
    return tuple(tie_neighbors), coupling, interactions, tuple(interaction_neighbors)


def _heisenberg_cut_diagnostic(
    nsites,
    mask,
    tie_neighbors,
    coupling,
    interactions,
    *,
    local_dim,
):
    full = (1 << nsites) - 1
    right_mask = full ^ mask
    left_sites = [site for site in range(nsites) if mask & (1 << site)]
    right_sites = [site for site in range(nsites) if right_mask & (1 << site)]
    frontier = tuple(site for site in right_sites if tie_neighbors[site] & mask)
    crossing_rank = int(
        np.linalg.matrix_rank(
            coupling[np.ix_(left_sites, right_sites)],
            tol=1.0e-12,
        )
    )
    has_left = any(
        mask & (1 << left) and mask & (1 << right)
        for left, right in interactions
    )
    has_right = any(
        right_mask & (1 << left) and right_mask & (1 << right)
        for left, right in interactions
    )
    rank = 3 * crossing_rank + int(has_left) + int(has_right)
    return {
        "frontier_sites": frontier,
        "frontier_width": len(frontier),
        "operator_rank": rank,
        "score": float(rank * local_dim ** (2 * len(frontier))),
    }


def _heisenberg_block_cut_diagnostic(
    nsites,
    mask,
    tie_neighbors,
    interactions,
    interaction_neighbors,
    *,
    local_dim,
):
    full = (1 << nsites) - 1
    right_mask = full ^ mask
    frontier = tuple(
        site
        for site in range(nsites)
        if right_mask & (1 << site) and tie_neighbors[site] & mask
    )
    frontier_mask = sum(1 << site for site in frontier)
    idle_paired = tuple(
        site for site in frontier if interaction_neighbors[site] & right_mask
    )
    crossing = 0
    crossing_on_frontier = 0
    has_left = False
    has_right = False
    for left, right in interactions:
        left_in_prefix = bool(mask & (1 << left))
        right_in_prefix = bool(mask & (1 << right))
        if left_in_prefix and right_in_prefix:
            has_left = True
            continue
        if not left_in_prefix and not right_in_prefix:
            has_right = True
            continue
        crossing += 1
        future_site = right if left_in_prefix else left
        crossing_on_frontier += bool(frontier_mask & (1 << future_site))

    frontier_width = len(frontier)
    base = float(local_dim**frontier_width)
    block_factor = (
        int(has_left)
        + int(has_right) * local_dim ** len(idle_paired)
        + 3 * crossing
        + 2 * (local_dim - 1) * crossing_on_frontier
    )
    return {
        "frontier_sites": frontier,
        "frontier_width": frontier_width,
        "mpo_channels": 2 + 3 * crossing,
        "active_mpo_channels": int(has_left) + int(has_right) + 3 * crossing,
        "idle_paired_sites": idle_paired,
        "crossing_bonds": crossing,
        "crossing_bonds_on_tied_frontier": crossing_on_frontier,
        "score": float(base * block_factor),
    }


def _heisenberg_cut_tables(
    nsites,
    tie_edges,
    weighted_bonds,
    *,
    local_dim,
):
    tie_neighbors = [0] * nsites
    for left, right, _coupling in tie_edges:
        tie_neighbors[left] |= 1 << right
        tie_neighbors[right] |= 1 << left

    coupling = np.zeros((nsites, nsites), dtype=float)
    for left, right, value in weighted_bonds:
        coupling[left, right] += value
        coupling[right, left] += value
    interactions = tuple(
        (left, right)
        for left in range(nsites)
        for right in range(left + 1, nsites)
        if coupling[left, right] != 0.0
    )

    size = 1 << nsites
    full = size - 1
    frontier_width = np.zeros(size, dtype=np.int16)
    operator_rank = np.ones(size, dtype=np.int16)
    score = np.zeros(size, dtype=float)
    frontier_sites = [()] * size
    for mask in range(1, full):
        left_sites = [site for site in range(nsites) if mask & (1 << site)]
        right_sites = [site for site in range(nsites) if not mask & (1 << site)]
        frontier = tuple(site for site in right_sites if tie_neighbors[site] & mask)
        crossing_rank = int(
            np.linalg.matrix_rank(
                coupling[np.ix_(left_sites, right_sites)],
                tol=1.0e-12,
            )
        )
        has_left = any(
            mask & (1 << left) and mask & (1 << right) for left, right in interactions
        )
        has_right = any(
            not mask & (1 << left) and not mask & (1 << right)
            for left, right in interactions
        )
        rank = 3 * crossing_rank + int(has_left) + int(has_right)
        frontier_sites[mask] = frontier
        frontier_width[mask] = len(frontier)
        operator_rank[mask] = rank
        score[mask] = rank * float(local_dim ** (2 * len(frontier)))
    return frontier_sites, frontier_width, operator_rank, score


def _heisenberg_block_cut_tables(
    nsites,
    tie_edges,
    weighted_bonds,
    *,
    local_dim,
):
    tie_neighbors = [0] * nsites
    for left, right, _coupling in tie_edges:
        tie_neighbors[left] |= 1 << right
        tie_neighbors[right] |= 1 << left

    coupling = np.zeros((nsites, nsites), dtype=float)
    for left, right, value in weighted_bonds:
        coupling[left, right] += value
        coupling[right, left] += value
    interactions = tuple(
        (left, right)
        for left in range(nsites)
        for right in range(left + 1, nsites)
        if coupling[left, right] != 0.0
    )
    interaction_neighbors = [0] * nsites
    for left, right in interactions:
        interaction_neighbors[left] |= 1 << right
        interaction_neighbors[right] |= 1 << left

    size = 1 << nsites
    full = size - 1
    diagnostics = [None] * size
    score = np.zeros(size, dtype=float)
    for mask in range(1, full):
        right_mask = full ^ mask
        frontier = tuple(
            site
            for site in range(nsites)
            if right_mask & (1 << site) and tie_neighbors[site] & mask
        )
        frontier_mask = sum(1 << site for site in frontier)
        idle_paired = tuple(
            site for site in frontier if interaction_neighbors[site] & right_mask
        )
        crossing = 0
        crossing_on_frontier = 0
        has_left = False
        has_right = False
        for left, right in interactions:
            left_in_prefix = bool(mask & (1 << left))
            right_in_prefix = bool(mask & (1 << right))
            if left_in_prefix and right_in_prefix:
                has_left = True
                continue
            if not left_in_prefix and not right_in_prefix:
                has_right = True
                continue
            crossing += 1
            future_site = right if left_in_prefix else left
            crossing_on_frontier += bool(frontier_mask & (1 << future_site))

        frontier_width = len(frontier)
        base = float(local_dim**frontier_width)
        # The raw Heisenberg automaton has idle/done channels and three
        # channels per crossing bond.  Reachability pruning keeps idle only
        # while a term remains wholly on the right and done only after a term
        # has completed on the left.  In each spin-vector triplet, exactly two
        # suffix operators are off-diagonal in the computational basis.
        block_factor = (
            int(has_left)
            + int(has_right) * local_dim ** len(idle_paired)
            + 3 * crossing
            + 2 * (local_dim - 1) * crossing_on_frontier
        )
        score[mask] = base * block_factor
        diagnostics[mask] = {
            "frontier_sites": frontier,
            "frontier_width": frontier_width,
            "mpo_channels": 2 + 3 * crossing,
            "active_mpo_channels": int(has_left) + int(has_right) + 3 * crossing,
            "idle_paired_sites": idle_paired,
            "crossing_bonds": crossing,
            "crossing_bonds_on_tied_frontier": crossing_on_frontier,
            "score": float(score[mask]),
        }
    return diagnostics, score


def _optimize_order_from_scores(nsites, score):
    size = 1 << nsites
    full = size - 1
    best_peak = np.full(size, np.inf)
    best_total = np.full(size, np.inf)
    previous = np.full(size, -1, dtype=np.int64)
    appended_site = np.full(size, -1, dtype=np.int16)
    best_peak[0] = 0.0
    best_total[0] = 0.0
    for mask in range(size):
        if not np.isfinite(best_peak[mask]):
            continue
        remaining = full ^ mask
        while remaining:
            bit = remaining & -remaining
            site = bit.bit_length() - 1
            new_mask = mask | bit
            candidate_peak = max(best_peak[mask], score[new_mask])
            candidate_total = best_total[mask] + score[new_mask]
            if candidate_peak < best_peak[new_mask] or (
                candidate_peak == best_peak[new_mask]
                and candidate_total < best_total[new_mask]
            ):
                best_peak[new_mask] = candidate_peak
                best_total[new_mask] = candidate_total
                previous[new_mask] = mask
                appended_site[new_mask] = site
            remaining -= bit

    order = []
    mask = full
    while mask:
        order.append(int(appended_site[mask]))
        mask = int(previous[mask])
    return tuple(reversed(order))


def heisenberg_frontier_profile(
    nsites,
    tie_edges,
    weighted_bonds,
    order,
    *,
    local_dim=2,
):
    r"""Return exact cut diagnostics for an isotropic Heisenberg term graph.

    For a cut with coupling matrix ``J_LR``, the exact operator-Schmidt rank is

    .. math::

        \chi = 3\,\mathrm{rank}(J_{LR}) + I(H_L\ne0) + I(H_R\ne0).

    The reported score omits the order-independent virtual factor ``D^2`` and
    equals ``chi * local_dim**(2 * frontier_width)``.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    local_dim = int(local_dim)
    if local_dim < 1:
        raise ValueError("local_dim must be positive.")
    order = tuple(int(site) for site in order)
    if sorted(order) != list(range(nsites)):
        raise ValueError("order must be a permutation of all sites.")
    ties = _validated_edges(nsites, tie_edges, weighted=False)
    bonds = _validated_edges(nsites, weighted_bonds, weighted=True)
    tie_neighbors, coupling, interactions, _interaction_neighbors = (
        _heisenberg_graph_data(nsites, ties, bonds)
    )
    mask = 0
    profile = []
    for cut, site in enumerate(order[:-1], start=1):
        mask |= 1 << site
        profile.append(
            {
                "cut": cut,
                **_heisenberg_cut_diagnostic(
                    nsites,
                    mask,
                    tie_neighbors,
                    coupling,
                    interactions,
                    local_dim=local_dim,
                ),
            }
        )
    return tuple(profile)


def optimize_heisenberg_order(
    nsites,
    tie_edges,
    weighted_bonds,
    *,
    local_dim=2,
    max_exact_sites=20,
):
    """Find the exact minimum-peak order by subset dynamic programming.

    Orders with equal peak scores are resolved by the sum of all cut scores,
    reducing total fixed-side frontier storage.  The dynamic program costs
    ``O(nsites * 2**nsites)`` after polynomial-per-cut score construction and
    is therefore restricted to modest graphs.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    if nsites > int(max_exact_sites):
        raise ValueError(
            f"exact order optimization is limited to {int(max_exact_sites)} sites."
        )
    local_dim = int(local_dim)
    if local_dim < 1:
        raise ValueError("local_dim must be positive.")
    ties = _validated_edges(nsites, tie_edges, weighted=False)
    bonds = _validated_edges(nsites, weighted_bonds, weighted=True)
    _frontier_sites, _frontier_width, _operator_rank, score = _heisenberg_cut_tables(
        nsites,
        ties,
        bonds,
        local_dim=local_dim,
    )

    return _optimize_order_from_scores(nsites, score)


def _heuristic_heisenberg_order(
    nsites,
    ties,
    bonds,
    *,
    local_dim,
    beam_width,
    block,
):
    beam_width = int(beam_width)
    if beam_width < 1:
        raise ValueError("beam_width must be positive.")
    tie_neighbors, coupling, interactions, interaction_neighbors = (
        _heisenberg_graph_data(nsites, ties, bonds)
    )
    full = (1 << nsites) - 1
    log_local_dim = float(np.log(local_dim))
    score_cache = {}

    def cut_score(mask):
        cached = score_cache.get(mask)
        if cached is not None:
            return cached
        if mask == 0 or mask == full:
            result = (-np.inf, 0.0)
        else:
            left_sites = [site for site in range(nsites) if mask & (1 << site)]
            right_sites = [
                site for site in range(nsites) if not mask & (1 << site)
            ]
            crossing_weight = float(
                np.sum(np.abs(coupling[np.ix_(left_sites, right_sites)]))
            )
            if block:
                diagnostic = _heisenberg_block_cut_diagnostic(
                    nsites,
                    mask,
                    tie_neighbors,
                    interactions,
                    interaction_neighbors,
                    local_dim=local_dim,
                )
                log_score = (
                    -np.inf
                    if diagnostic["score"] == 0.0
                    else float(np.log(diagnostic["score"]))
                )
            else:
                diagnostic = _heisenberg_cut_diagnostic(
                    nsites,
                    mask,
                    tie_neighbors,
                    coupling,
                    interactions,
                    local_dim=local_dim,
                )
                width = diagnostic["frontier_width"]
                rank = diagnostic["operator_rank"]
                # Optimize the combined norm/Hamiltonian footprint.  The norm
                # frontier still costs d**width when a disconnected cut has
                # zero Hamiltonian rank.
                log_score = (
                    width * log_local_dim
                    if rank == 0
                    else float(np.log(rank)) + 2 * width * log_local_dim
                )
            result = (log_score, crossing_weight)
        score_cache[mask] = result
        return result

    # Entries are (mask, peak-log-score, total-log-score, crossing-weight,
    # order).  Paths reaching the same mask have identical possible futures,
    # so only the lexicographically best path for that mask is retained.
    beam = ((0, -np.inf, -np.inf, 0.0, ()),)
    for _depth in range(nsites):
        # Masks from an earlier depth cannot recur, so retain only this
        # layer's lazy cut scores and keep memory linear in the beam size.
        score_cache.clear()
        candidates = {}
        for mask, peak, total, weight, order in beam:
            remaining = full ^ mask
            while remaining:
                bit = remaining & -remaining
                site = bit.bit_length() - 1
                new_mask = mask | bit
                log_score, crossing_weight = cut_score(new_mask)
                candidate = (
                    new_mask,
                    max(peak, log_score),
                    float(np.logaddexp(total, log_score)),
                    weight + crossing_weight,
                    order + (site,),
                )
                incumbent = candidates.get(new_mask)
                candidate_key = candidate[1:]
                if incumbent is None or candidate_key < incumbent[1:]:
                    candidates[new_mask] = candidate
                remaining -= bit
        beam = tuple(
            sorted(
                candidates.values(),
                key=lambda entry: (entry[1], entry[2], entry[3], entry[4]),
            )[:beam_width]
        )
    return beam[0][4]


def heuristic_heisenberg_order(
    nsites,
    tie_edges,
    weighted_bonds,
    *,
    local_dim=2,
    beam_width=64,
):
    """Deterministically optimize a large graph with weighted beam search.

    The primary cut score combines the tied-physical frontier with the exact
    operator-Schmidt rank of the weighted Heisenberg coupling matrix.  Peak
    storage is minimized first, followed by total storage and the accumulated
    absolute coupling weight crossing all cuts.  Unlike the exact subset
    dynamic program, memory is ``O(beam_width * nsites)``.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    local_dim = int(local_dim)
    if local_dim < 1:
        raise ValueError("local_dim must be positive.")
    ties = _validated_edges(nsites, tie_edges, weighted=False)
    bonds = _validated_edges(nsites, weighted_bonds, weighted=True)
    return _heuristic_heisenberg_order(
        nsites,
        ties,
        bonds,
        local_dim=local_dim,
        beam_width=beam_width,
        block=False,
    )


def heisenberg_block_frontier_profile(
    nsites,
    tie_edges,
    weighted_bonds,
    order,
    *,
    local_dim=2,
):
    r"""Return exact cut sizes for the identity-aware Heisenberg backend.

    The score is the number of Hamiltonian block-message elements with the
    common virtual factor $D^2$ omitted.  It applies to the raw finite-state
    MPO produced for two-site isotropic Heisenberg interactions.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    local_dim = int(local_dim)
    if local_dim < 1:
        raise ValueError("local_dim must be positive.")
    order = tuple(int(site) for site in order)
    if sorted(order) != list(range(nsites)):
        raise ValueError("order must be a permutation of all sites.")
    ties = _validated_edges(nsites, tie_edges, weighted=False)
    bonds = _validated_edges(nsites, weighted_bonds, weighted=True)
    tie_neighbors, _coupling, interactions, interaction_neighbors = (
        _heisenberg_graph_data(nsites, ties, bonds)
    )
    mask = 0
    profile = []
    for cut, site in enumerate(order[:-1], start=1):
        mask |= 1 << site
        profile.append(
            {
                "cut": cut,
                **_heisenberg_block_cut_diagnostic(
                    nsites,
                    mask,
                    tie_neighbors,
                    interactions,
                    interaction_neighbors,
                    local_dim=local_dim,
                ),
            }
        )
    return tuple(profile)


def optimize_heisenberg_block_order(
    nsites,
    tie_edges,
    weighted_bonds,
    *,
    local_dim=2,
    max_exact_sites=20,
):
    """Exactly optimize peak, then total identity-aware block storage."""
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    if nsites > int(max_exact_sites):
        raise ValueError(
            f"exact order optimization is limited to {int(max_exact_sites)} sites."
        )
    local_dim = int(local_dim)
    if local_dim < 1:
        raise ValueError("local_dim must be positive.")
    ties = _validated_edges(nsites, tie_edges, weighted=False)
    bonds = _validated_edges(nsites, weighted_bonds, weighted=True)
    _diagnostics, score = _heisenberg_block_cut_tables(
        nsites,
        ties,
        bonds,
        local_dim=local_dim,
    )
    return _optimize_order_from_scores(nsites, score)


def heuristic_heisenberg_block_order(
    nsites,
    tie_edges,
    weighted_bonds,
    *,
    local_dim=2,
    beam_width=64,
):
    """Deterministic large-graph ordering for identity-aware block messages."""
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    local_dim = int(local_dim)
    if local_dim < 1:
        raise ValueError("local_dim must be positive.")
    ties = _validated_edges(nsites, tie_edges, weighted=False)
    bonds = _validated_edges(nsites, weighted_bonds, weighted=True)
    return _heuristic_heisenberg_order(
        nsites,
        ties,
        bonds,
        local_dim=local_dim,
        beam_width=beam_width,
        block=True,
    )


__all__ = [
    "heuristic_heisenberg_block_order",
    "heuristic_heisenberg_order",
    "heisenberg_block_frontier_profile",
    "heisenberg_frontier_profile",
    "optimize_heisenberg_block_order",
    "optimize_heisenberg_order",
]
