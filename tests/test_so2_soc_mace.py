from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem.ci.fci import get_fci_string_basis
from pyqed.qchem.mcscf.soc_si import (
    align_triplet_multiplet_phases,
    spin_lower,
)
from pyqed.ldr.so2 import (
    adaptive_points,
    canonical_spin_vibronic_permutation,
    full_spin_overlap,
    procrustes_fields,
    select_root_sectors,
    spin_one_representation,
    symmetry_block_procrustes,
)
from examples.namd.home_so2_cas88_somf_v7 import (
    candidate_choices,
    home_choices_on_tree,
)


def test_so2_root_selection_uses_lowest_candidate_in_each_fixed_sector():
    selected = select_root_sectors((1, -1, 1, -1, -1, 1), (1, -1, -1))
    np.testing.assert_array_equal(selected, (0, 1, 3))
    with pytest.raises(ValueError, match="do not span"):
        select_root_sectors((1, -1, 1), (1, -1, -1))


def test_so2_candidate_choices_keep_every_fixed_sector_subspace():
    choices = candidate_choices((1, -1, -1, 1, -1, 1))
    assert len(choices) == 9
    assert choices[0] == (0, 1, 2)
    assert choices[-1] == (5, 2, 4)
    assert all(choice[0] in (0, 3, 5) for choice in choices)


def test_so2_tree_homing_maximizes_raw_overlap_globally():
    choices = (((0, 1, 2), (0, 1, 3)),) * 3
    scores = (
        np.asarray(((0.9, 0.1), (0.2, 0.8))),
        np.asarray(((0.1, 0.9), (0.8, 0.2))),
    )
    selected = home_choices_on_tree(
        choices,
        np.asarray(((0, 1), (1, 2))),
        scores,
        np.asarray((0, 1)),
        0,
        (0, 1, 2),
    )
    assert selected == ((0, 1, 2), (0, 1, 2), (0, 1, 3))


def test_spin_vibronic_permutation_canonicalizes_root_sector_order():
    permutation = canonical_spin_vibronic_permutation(
        (1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, -1, -1)
    )
    np.testing.assert_array_equal(permutation, (0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11))
    with pytest.raises(ValueError, match="target symmetry sectors"):
        canonical_spin_vibronic_permutation(
            (1, -1, 1), (1, -1, -1), (1, -1, -1), (1, -1, -1)
        )


def test_so2_procrustes_fields_transport_over_graph_without_anchor_links():
    phase01 = np.diag(np.exp(1j * np.asarray((0.3, -0.2))))
    phase12 = np.diag(np.exp(1j * np.asarray((-0.4, 0.5))))
    overlaps = np.asarray(
        (phase01 @ np.diag((0.8, 0.7)), phase12 @ np.diag((0.6, 0.5)))
    )
    records = [
        {
            "coordinate": np.asarray((2.7 + 0.03 * point, 2.6, 2.08)),
            "labels": ["S0", "T0(Ms=+0)"],
            "h_total": np.diag((point, point + 1.0)),
        }
        for point in range(3)
    ]
    identity = np.eye(2)
    hamiltonians, links, gauges, _shift, info = procrustes_fields(
        records,
        np.asarray(((0, 1), (1, 2))),
        overlaps,
        {"sigma_xy": identity, "C2(x)": identity},
        0,
    )
    np.testing.assert_allclose(
        gauges.conj().swapaxes(-1, -2) @ gauges,
        np.broadcast_to(identity, gauges.shape),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.linalg.svd(links, compute_uv=False),
        np.linalg.svd(overlaps, compute_uv=False),
    )
    assert hamiltonians.shape == (3, 2, 2)
    assert info["minimum_tree_singular_value"] == pytest.approx(0.5)


def test_so2_adaptive_points_are_continuous_canonical_and_reproducible():
    class Feature:
        def predict(self, coordinates):
            coordinates = np.asarray(coordinates)
            values = np.broadcast_to(np.eye(2), (len(coordinates), 2, 2)).copy()
            values[:, 0, 0] += 0.1 * coordinates[:, 2]
            return values

    sampled = np.asarray(((2.7, 2.7, 2.08), (2.8, 2.6, 2.0)))
    bounds = (2.55, 3.05, 0.25, 1.75, 2.45)
    first, info = adaptive_points(
        Feature(), sampled, bounds, 4, candidate_pool=64, seed=9,
        max_distance=2.0,
    )
    second, _ = adaptive_points(
        Feature(), sampled, bounds, 4, candidate_pool=64, seed=9,
        max_distance=2.0,
    )
    np.testing.assert_allclose(first, second)
    assert np.all(first[:, 0] >= first[:, 1])
    assert len(np.unique(first, axis=0)) == 4
    assert info["candidate_pool"] == 64


def test_triplet_phases_are_canonicalized_by_spin_lowering():
    plus_basis = get_fci_string_basis(np.asarray(((1, 1), (0, 0))))
    zero_basis = get_fci_string_basis(np.asarray(((1, 0), (1, 0))))
    minus_basis = get_fci_string_basis(np.asarray(((0, 0), (1, 1))))
    plus = np.ones(1)
    zero = spin_lower(plus, plus_basis, zero_basis) / np.sqrt(2.0)
    minus = spin_lower(zero, zero_basis, minus_basis) / np.sqrt(2.0)
    triplets = {
        1: SimpleNamespace(ci=[np.exp(0.37j) * plus], binary=plus_basis),
        0: SimpleNamespace(ci=[zero], binary=zero_basis),
        -1: SimpleNamespace(ci=[np.exp(-0.81j) * minus], binary=minus_basis),
    }

    diagnostics = align_triplet_multiplet_phases(triplets)

    plus_ladder = np.vdot(
        triplets[0].ci[0],
        spin_lower(triplets[1].ci[0], plus_basis, zero_basis),
    )
    minus_ladder = np.vdot(
        triplets[-1].ci[0],
        spin_lower(triplets[0].ci[0], zero_basis, minus_basis),
    )
    assert plus_ladder == pytest.approx(np.sqrt(2.0))
    assert minus_ladder == pytest.approx(np.sqrt(2.0))
    assert diagnostics["off_diagonal"] < 1.0e-14
    assert diagnostics["amplitude_error"] < 1.0e-14


class _Frame:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.ci = tuple(range(len(value)))

    def overlap(self, other):
        return self.value.conj().T @ other.value


def test_full_so2_spin_overlap_preserves_triplet_root_major_order(monkeypatch):
    monkeypatch.setattr(
        "pyqed.ldr.so2.casci_overlap", lambda left, right: left.overlap(right)
    )
    singlet_left = _Frame(np.eye(2))
    singlet_right = _Frame(np.asarray([[0.8, -0.2], [0.2, 0.8]]))
    triplet_left = {ms: _Frame(np.eye(2)) for ms in (-1, 0, 1)}
    triplet_right = {
        ms: _Frame((1.0 + 0.1 * ms) * np.eye(2)) for ms in (-1, 0, 1)
    }
    left = {"singlet_frame": singlet_left, "triplet_frames": triplet_left}
    right = {"singlet_frame": singlet_right, "triplet_frames": triplet_right}
    value = full_spin_overlap(left, right)
    np.testing.assert_allclose(value[:2, :2], singlet_right.value)
    np.testing.assert_allclose(np.diag(value)[2:], [0.9, 1.0, 1.1] * 2)
    np.testing.assert_allclose(value[:2, 2:], 0.0)


def test_spin_one_so2_point_group_is_unitary_and_closed():
    c2x = spin_one_representation((1.0, -1.0, -1.0))
    sigma_xy = spin_one_representation((1.0, 1.0, -1.0))
    sigma_xz = spin_one_representation((1.0, -1.0, 1.0))
    identity = np.eye(3)
    for value in (c2x, sigma_xy, sigma_xz):
        np.testing.assert_allclose(value.conj().T @ value, identity, atol=1.0e-13)
        np.testing.assert_allclose(value @ value, identity, atol=1.0e-13)
    np.testing.assert_allclose(c2x @ sigma_xy, sigma_xz, atol=1.0e-13)


def test_symmetry_block_procrustes_commutes_with_nondiagonal_involutions():
    exchange = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    tag = np.diag([1.0, 1.0])
    value = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=complex)
    rotation = symmetry_block_procrustes(value, (tag, exchange))
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(2), atol=1.0e-13)
    np.testing.assert_allclose(rotation @ exchange, exchange @ rotation, atol=1.0e-13)
