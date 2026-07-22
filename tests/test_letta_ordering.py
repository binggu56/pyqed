from itertools import permutations

import numpy as np
import pytest

from pyqed.letta import (
    FrontierTiedLETTA,
    LocalHamiltonian,
    LocalTerm,
    heisenberg_block_frontier_profile,
    heisenberg_frontier_profile,
    optimize_heisenberg_block_order,
    optimize_heisenberg_order,
)
from pyqed.letta.ordering import (
    heuristic_heisenberg_block_order,
    heuristic_heisenberg_order,
)


def _objective(nsites, tie_edges, weighted_bonds, order):
    profile = heisenberg_frontier_profile(
        nsites,
        tie_edges,
        weighted_bonds,
        order,
    )
    scores = tuple(entry["score"] for entry in profile)
    return max(scores, default=0.0), sum(scores)


def _block_objective(nsites, tie_edges, weighted_bonds, order):
    profile = heisenberg_block_frontier_profile(
        nsites,
        tie_edges,
        weighted_bonds,
        order,
    )
    scores = tuple(entry["score"] for entry in profile)
    return max(scores, default=0.0), sum(scores)


def test_heisenberg_frontier_profile_reports_exact_chain_cut_data():
    ties = ((0, 1), (1, 2))
    bonds = ((0, 1, 1.0), (1, 2, 1.0))

    profile = heisenberg_frontier_profile(3, ties, bonds, (0, 1, 2))

    assert profile == (
        {
            "cut": 1,
            "frontier_sites": (1,),
            "frontier_width": 1,
            "operator_rank": 4,
            "score": 16.0,
        },
        {
            "cut": 2,
            "frontier_sites": (2,),
            "frontier_width": 1,
            "operator_rank": 4,
            "score": 16.0,
        },
    )


def test_exact_order_optimizer_matches_brute_force_peak_and_total():
    ties = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4))
    bonds = (
        (0, 1, 1.0),
        (1, 2, -0.7),
        (2, 3, 0.4),
        (3, 4, 1.2),
        (0, 4, 0.3),
        (1, 3, 0.5),
    )

    order = optimize_heisenberg_order(5, ties, bonds)
    brute_objective = min(
        _objective(5, ties, bonds, candidate) for candidate in permutations(range(5))
    )

    assert sorted(order) == list(range(5))
    assert _objective(5, ties, bonds, order) == brute_objective


def test_exact_block_order_optimizer_matches_brute_force_peak_and_total():
    ties = ((0, 1), (1, 2), (2, 3), (0, 3))
    bonds = (
        (0, 1, 1.0),
        (1, 2, 0.7),
        (2, 3, 1.2),
        (0, 3, 0.3),
        (0, 2, 0.5),
    )

    order = optimize_heisenberg_block_order(4, ties, bonds)
    brute_objective = min(
        _block_objective(4, ties, bonds, candidate)
        for candidate in permutations(range(4))
    )

    assert sorted(order) == list(range(4))
    assert _block_objective(4, ties, bonds, order) == brute_objective


@pytest.mark.parametrize(
    ("exact_optimizer", "heuristic_optimizer", "objective"),
    [
        (optimize_heisenberg_order, heuristic_heisenberg_order, _objective),
        (
            optimize_heisenberg_block_order,
            heuristic_heisenberg_block_order,
            _block_objective,
        ),
    ],
)
def test_weighted_beam_order_matches_exact_when_all_subset_masks_fit(
    exact_optimizer,
    heuristic_optimizer,
    objective,
):
    ties = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (0, 6),
        (1, 5),
    )
    bonds = tuple(
        (left, right, 1.0 + 0.1 * index)
        for index, (left, right) in enumerate(ties)
    ) + ((0, 3, 0.4), (2, 6, -0.3))

    exact = exact_optimizer(7, ties, bonds)
    heuristic = heuristic_optimizer(7, ties, bonds, beam_width=64)

    assert sorted(heuristic) == list(range(7))
    assert objective(7, ties, bonds, heuristic) == objective(7, ties, bonds, exact)


def test_large_graph_weighted_order_and_profile_are_deterministic():
    nsites = 32
    ties = tuple((site, site + 1) for site in range(nsites - 1))
    ties += tuple((site, site + 8) for site in range(nsites - 8))
    bonds = tuple(
        (left, right, 1.0 if right == left + 1 else 0.35)
        for left, right in ties
    )
    bonds += tuple((site, site + 9, -0.2) for site in range(nsites - 9))

    first = heuristic_heisenberg_order(
        nsites, ties, bonds, beam_width=16
    )
    second = heuristic_heisenberg_order(
        nsites, ties, bonds, beam_width=16
    )
    profile = heisenberg_frontier_profile(nsites, ties, bonds, first)

    assert first == second
    assert sorted(first) == list(range(nsites))
    assert len(profile) == nsites - 1
    assert all(np.isfinite(entry["score"]) for entry in profile)


def test_block_profile_matches_constructed_frontier_message_sizes():
    ties = ((0, 1), (1, 2), (2, 3), (0, 3))
    bonds = (
        (0, 1, 1.0),
        (1, 2, 0.7),
        (2, 3, 1.2),
        (0, 3, 0.3),
        (0, 2, 0.5),
    )
    order = (3, 1, 0, 2)
    position = {old: new for new, old in enumerate(order)}

    def remap(edge):
        left, right, *payload = edge
        return (*sorted((position[left], position[right])), *payload)

    mapped_ties = tuple(remap(edge) for edge in ties)
    mapped_bonds = tuple(remap(edge) for edge in bonds)
    parents = [set() for _ in order]
    for left, right in mapped_ties:
        left, right = sorted((left, right))
        parents[left].add(right)
    exchange = np.array(
        [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, 0.0],
            [0.0, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ]
    )
    hamiltonian = LocalHamiltonian(
        (2,) * 4,
        [
            LocalTerm((left, right), coupling * exchange)
            for left, right, coupling in mapped_bonds
        ],
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        tuple(tuple(sorted(values)) for values in parents),
        bond_dim=3,
        seed=4,
        frontier_backend="identity_block",
    )
    profile = heisenberg_block_frontier_profile(4, ties, bonds, order)

    assert [
        state._hamiltonian_frontier.message_elements(entry["cut"]) for entry in profile
    ] == [9 * entry["score"] for entry in profile]

    compressed_state = FrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        tuple(tuple(sorted(values)) for values in parents),
        bond_dim=3,
        seed=4,
        frontier_backend="compressed",
    )
    compressed_profile = heisenberg_frontier_profile(4, ties, bonds, order)
    assert [
        compressed_state._hamiltonian_frontier.message_elements(entry["cut"])
        for entry in compressed_profile
    ] == [9 * entry["score"] for entry in compressed_profile]


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((0, (), ()), "positive"),
        ((3, ((0, 3),), ()), "valid sites"),
        ((3, (), ((0, 1),)), "weighted edges"),
    ],
)
def test_ordering_input_validation(args, match):
    with pytest.raises(ValueError, match=match):
        optimize_heisenberg_order(*args)


def test_ordering_rejects_nonpermutation_and_large_exact_problem():
    with pytest.raises(ValueError, match="permutation"):
        heisenberg_frontier_profile(3, (), (), (0, 0, 2))
    with pytest.raises(ValueError, match="limited"):
        optimize_heisenberg_order(21, (), ())
