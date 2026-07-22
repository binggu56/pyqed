import numpy as np
import pytest

from pyqed.letta.physical_tying import (
    compress_physical_ties,
    fixed_range_parent_sets,
)


def _crossed_bell_pair_state():
    """Return |Phi+>_(0,2) tensor |Phi+>_(1,3) in chain order."""
    state = np.zeros((2, 2, 2, 2), dtype=complex)
    for x0 in range(2):
        for x1 in range(2):
            state[x0, x1, x0, x1] = 0.5
    return state


def test_greedy_physical_ties_find_crossed_bell_partners_exactly():
    state = _crossed_bell_pair_state()

    compressed = compress_physical_ties(
        state,
        state.shape,
        max_parents=1,
        relative_tolerance=1.0e-14,
    )

    # In one-based XF notation the nontrivial contexts are P_1={3} and
    # P_2={4}; no context is needed for the uniform third-site factor.
    assert compressed.parent_sets == ((2,), (3,), ())
    np.testing.assert_allclose(compressed.state_vector(), state.reshape(-1), atol=1.0e-14)
    np.testing.assert_allclose(compressed.fidelity(state), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(compressed.discarded_weight, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(compressed.retained_weight, 1.0, atol=1.0e-14)


def test_full_suffix_physical_ties_exactly_factor_random_complex_state():
    rng = np.random.default_rng(17)
    dims = (2, 3, 2, 2)
    state = rng.normal(size=dims) + 1.0j * rng.normal(size=dims)
    expected = state.reshape(-1) / np.linalg.norm(state)
    full_suffix = tuple(
        tuple(range(site + 1, len(dims)))
        for site in range(len(dims) - 1)
    )

    compressed = compress_physical_ties(
        state,
        dims,
        parent_sets=full_suffix,
    )

    assert compressed.parent_sets == ((1, 2, 3), (2, 3), (3,))
    np.testing.assert_allclose(compressed.state_vector(), expected, atol=1.0e-13)
    np.testing.assert_allclose(compressed.fidelity(state), 1.0, atol=1.0e-13)
    np.testing.assert_allclose(compressed.discarded_weight, 0.0, atol=1.0e-13)


def test_nearest_physical_ties_miss_crossed_bell_correlations():
    state = _crossed_bell_pair_state()
    nearest = fixed_range_parent_sets(nsites=4, tie_range=1)

    compressed = compress_physical_ties(
        state,
        state.shape,
        parent_sets=nearest,
    )

    assert compressed.parent_sets == ((1,), (2,), (3,))
    np.testing.assert_allclose(compressed.fidelity(state), 0.25, atol=1.0e-14)
    np.testing.assert_allclose(compressed.retained_weight, 0.25, atol=1.0e-14)
    np.testing.assert_allclose(compressed.discarded_weight, 0.75, atol=1.0e-14)
    np.testing.assert_allclose(
        compressed.discarded_weight + compressed.retained_weight,
        1.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        compressed.fidelity(state),
        compressed.retained_weight,
        atol=1.0e-14,
    )


def test_adaptive_physical_tying_is_covariant_under_global_phase():
    rng = np.random.default_rng(31)
    dims = (2, 2, 2, 2, 2)
    state = rng.normal(size=dims) + 1.0j * rng.normal(size=dims)
    phase = np.exp(0.73j)

    reference = compress_physical_ties(
        state,
        dims,
        max_parents=2,
        relative_tolerance=1.0e-14,
    )
    rotated = compress_physical_ties(
        phase * state,
        dims,
        max_parents=2,
        relative_tolerance=1.0e-14,
    )

    assert rotated.parent_sets == reference.parent_sets
    np.testing.assert_allclose(rotated.retained_weight, reference.retained_weight, atol=1.0e-14)
    np.testing.assert_allclose(
        rotated.state_vector(),
        phase * reference.state_vector(),
        atol=1.0e-13,
    )


def test_degenerate_ghz_gram_space_has_deterministic_product_projection():
    state = np.zeros((2, 2, 2), dtype=complex)
    state[0, 0, 0] = 1.0 / np.sqrt(2.0)
    state[1, 1, 1] = 1.0 / np.sqrt(2.0)
    no_parents = fixed_range_parent_sets(nsites=3, tie_range=0)

    compressed = compress_physical_ties(state, state.shape, parent_sets=no_parents)

    np.testing.assert_allclose(compressed.fidelity(state), 0.5, atol=1.0e-14)
    np.testing.assert_allclose(compressed.retained_weight, 0.5, atol=1.0e-14)
    np.testing.assert_allclose(compressed.discarded_weight, 0.5, atol=1.0e-14)
    np.testing.assert_allclose(
        compressed.state_vector(),
        np.eye(1, 8, dtype=complex).reshape(-1),
        atol=1.0e-14,
    )


def test_mandatory_nearest_requires_a_parent_slot():
    with pytest.raises(ValueError, match="at least one"):
        compress_physical_ties(
            np.ones(4),
            (2, 2),
            max_parents=0,
            mandatory_nearest=True,
        )
