import numpy as np
import pytest

from pyqed.mps.nonabelian import (
    CouplingChannel,
    FusionLeg,
    ReducedBondSpace,
    clebsch_gordan,
    clebsch_gordan_tensor,
    couple_two_sectors_matrix,
    enumerate_sector_couplings,
    normalize_coupling_scheme,
    recoupling_matrix,
    reduced_bond_space,
    two_m_values,
    combine_legs,
    recouple_fused_leg,
    split_legs,
    NonabelianTensor,
)
from pyqed.mps.symmetry import Sector
from pyqed.mps.su2 import SU2Irrep


def _charge_spin_sector(charge, two_j):
    return Sector(("charge", "su2"), (charge, SU2Irrep(two_j)))


def test_clebsch_gordan_half_times_half_matches_known_singlet_triplet_values():
    half = SU2Irrep(1)
    singlet = SU2Irrep(0)
    triplet = SU2Irrep(2)

    assert clebsch_gordan(half, half, singlet, 1, -1, 0) == pytest.approx(1.0 / np.sqrt(2.0))
    assert clebsch_gordan(half, half, singlet, -1, 1, 0) == pytest.approx(-1.0 / np.sqrt(2.0))
    assert clebsch_gordan(half, half, triplet, 1, 1, 2) == pytest.approx(1.0)
    assert clebsch_gordan(half, half, triplet, 1, -1, 0) == pytest.approx(1.0 / np.sqrt(2.0))


def test_clebsch_gordan_tables_are_normalized_per_output_multiplet():
    half = SU2Irrep(1)
    triplet = SU2Irrep(2)

    table = clebsch_gordan_tensor(half, half, triplet)
    for two_m_fused in two_m_values(triplet):
        coeffs = [
            coeff
            for (two_m_left, two_m_right, two_m), coeff in table.items()
            if two_m == two_m_fused
        ]
        assert sum(abs(coeff) ** 2 for coeff in coeffs) == pytest.approx(1.0)


def test_enumerate_sector_couplings_preserves_three_doublet_multiplicity():
    doublet = _charge_spin_sector(1, 1)
    channels = enumerate_sector_couplings((doublet, doublet, doublet))

    final_doublet = _charge_spin_sector(3, 1)
    final_quartet = _charge_spin_sector(3, 3)

    doublet_channels = [channel for channel in channels if channel.fused_sector == final_doublet]
    quartet_channels = [channel for channel in channels if channel.fused_sector == final_quartet]

    assert len(doublet_channels) == 2
    assert len(quartet_channels) == 1
    assert all(isinstance(channel, CouplingChannel) for channel in channels)
    assert {channel.slot for channel in doublet_channels} == {0, 1}
    assert {channel.intermediate_sectors[0] for channel in doublet_channels} == {
        _charge_spin_sector(2, 0),
        _charge_spin_sector(2, 2),
    }


def test_enumerate_sector_couplings_supports_right_associative_scheme():
    doublet = _charge_spin_sector(1, 1)
    final_doublet = _charge_spin_sector(3, 1)

    channels = enumerate_sector_couplings((doublet, doublet, doublet), scheme="right")
    doublet_channels = [channel for channel in channels if channel.fused_sector == final_doublet]

    assert len(doublet_channels) == 2
    assert {channel.slot for channel in doublet_channels} == {0, 1}
    assert {channel.intermediate_sectors[0] for channel in doublet_channels} == {
        _charge_spin_sector(2, 0),
        _charge_spin_sector(2, 2),
    }


def test_fusion_leg_from_children_carries_explicit_coupling_channels():
    doublet = _charge_spin_sector(1, 1)
    final_doublet = _charge_spin_sector(3, 1)

    leg = FusionLeg.from_children(
        child_legs=(0, 1, 2),
        child_sector_lists=((doublet,), (doublet,), (doublet,)),
        child_dirs=(-1, 1, 1),
        orientation=1,
    )

    channels = leg.channels_for((doublet, doublet, doublet), final_doublet)
    assert len(channels) == 2
    assert {channel.slot for channel in channels} == {0, 1}
    with pytest.raises(ValueError, match="Ambiguous slot"):
        leg.slot_for((doublet, doublet, doublet), final_doublet)


def test_fusion_leg_bond_space_matches_requested_coupling_scheme():
    doublet = _charge_spin_sector(1, 1)
    final_doublet = _charge_spin_sector(3, 1)

    leg = FusionLeg.from_children(
        child_legs=(0, 1, 2),
        child_sector_lists=((doublet,), (doublet,), (doublet,)),
        child_dirs=(-1, 1, 1),
        orientation=1,
        coupling="right",
    )

    bond_space = leg.bond_space((doublet, doublet, doublet), final_doublet)
    assert isinstance(bond_space, ReducedBondSpace)
    assert bond_space.scheme == "right"
    assert bond_space.multiplicity == 2
    np.testing.assert_allclose(
        leg.recoupling_matrix((doublet, doublet, doublet), final_doublet, target_scheme="left")
        @ bond_space.recouple_to(reduced_bond_space((doublet, doublet, doublet), final_doublet, scheme="left")),
        np.eye(2),
        atol=1.0e-12,
    )


def test_reduced_bond_space_recouples_left_and_right_three_doublets():
    doublet = _charge_spin_sector(1, 1)
    final_doublet = _charge_spin_sector(3, 1)

    left_space = reduced_bond_space((doublet, doublet, doublet), final_doublet, scheme="left")
    right_space = reduced_bond_space((doublet, doublet, doublet), final_doublet, scheme="right")
    recouple = left_space.recouple_to(right_space)

    assert left_space.multiplicity == 2
    assert right_space.multiplicity == 2
    assert recouple.shape == (2, 2)
    np.testing.assert_allclose(recouple.T @ recouple, np.eye(2), atol=1.0e-12)
    assert not np.allclose(recouple, np.eye(2), atol=1.0e-12)

    fused_dim = left_space.fused_dim
    left_basis = left_space.concatenated_basis()
    right_basis = right_space.concatenated_basis()
    lifted = np.kron(recouple.T, np.eye(fused_dim))
    np.testing.assert_allclose(left_basis @ lifted, right_basis, atol=1.0e-12)


def test_public_recoupling_matrix_matches_bond_space_overlap():
    doublet = _charge_spin_sector(1, 1)
    final_doublet = _charge_spin_sector(3, 1)

    direct = recoupling_matrix(
        (doublet, doublet, doublet),
        final_doublet,
        source_scheme="left",
        target_scheme="right",
    )
    via_space = reduced_bond_space(
        (doublet, doublet, doublet), final_doublet, scheme="left"
    ).recouple_to(
        reduced_bond_space((doublet, doublet, doublet), final_doublet, scheme="right")
    )
    np.testing.assert_allclose(direct, via_space, atol=1.0e-12)


def test_combine_legs_with_cg_projects_two_doublets_into_singlet():
    left_boundary = _charge_spin_sector(0, 0)
    half = _charge_spin_sector(1, 1)
    singlet = _charge_spin_sector(2, 0)

    # |S=0, M=0> = (|up down> - |down up>) / sqrt(2)
    singlet_block = np.array([[[[0.0], [1.0 / np.sqrt(2.0)]], [[-1.0 / np.sqrt(2.0)], [0.0]]]])
    tensor = NonabelianTensor(
        data={(left_boundary, half, half, singlet): singlet_block},
        qns=[[left_boundary], [half], [half], [singlet]],
        dirs=[-1, 1, 1, 1],
    )
    fusion_leg = FusionLeg.from_children(
        child_legs=(1, 2),
        child_sector_lists=((half,), (half,)),
        child_dirs=(1, 1),
        orientation=1,
        selected_channel=singlet,
    )

    combined = combine_legs(tensor, (1, 2), fusion_leg=fusion_leg, use_cg=True)

    assert combined.data[(left_boundary, singlet, singlet)].shape == (1, 1, 1)
    assert combined.fusion_legs[1].pipe.coupling == "left"
    np.testing.assert_allclose(
        combined.data[(left_boundary, singlet, singlet)],
        np.array([[[1.0]]]),
        atol=1e-12,
    )

    recovered = split_legs(combined, 1)
    np.testing.assert_allclose(
        recovered.data[(left_boundary, half, half, singlet)],
        singlet_block,
        atol=1e-12,
    )


def test_combine_legs_with_cg_builds_singlet_and_triplet_blocks():
    left_boundary = _charge_spin_sector(0, 0)
    half = _charge_spin_sector(1, 1)
    right_boundary = _charge_spin_sector(0, 0)
    singlet = _charge_spin_sector(2, 0)
    triplet = _charge_spin_sector(2, 2)

    tensor = NonabelianTensor(
        data={(left_boundary, half, half, right_boundary): np.arange(4.0).reshape(1, 2, 2, 1)},
        qns=[[left_boundary], [half], [half], [right_boundary]],
        dirs=[-1, 1, 1, 1],
    )
    fusion_leg = FusionLeg.from_children(
        child_legs=(1, 2),
        child_sector_lists=((half,), (half,)),
        child_dirs=(1, 1),
        orientation=1,
    )

    combined = combine_legs(tensor, (1, 2), fusion_leg=fusion_leg, use_cg=True)

    assert set(combined.data) == {
        (left_boundary, singlet, right_boundary),
        (left_boundary, triplet, right_boundary),
    }
    assert combined.data[(left_boundary, singlet, right_boundary)].shape == (1, 1, 1)
    assert combined.data[(left_boundary, triplet, right_boundary)].shape == (1, 3, 1)


def test_combine_three_doublets_builds_reduced_multiplet_axis_and_recouples():
    left_boundary = _charge_spin_sector(0, 0)
    half = _charge_spin_sector(1, 1)
    right_boundary = _charge_spin_sector(0, 0)
    final_doublet = _charge_spin_sector(3, 1)
    final_quartet = _charge_spin_sector(3, 3)

    tensor = NonabelianTensor(
        data={
            (left_boundary, half, half, half, right_boundary): np.arange(8.0).reshape(1, 2, 2, 2, 1)
        },
        qns=[[left_boundary], [half], [half], [half], [right_boundary]],
        dirs=[-1, 1, 1, 1, 1],
    )
    left_leg = FusionLeg.from_children(
        child_legs=(1, 2, 3),
        child_sector_lists=((half,), (half,), (half,)),
        child_dirs=(1, 1, 1),
        orientation=1,
        coupling="left",
    )

    combined_left = combine_legs(tensor, (1, 2, 3), fusion_leg=left_leg, use_cg=True)

    assert set(combined_left.data) == {
        (left_boundary, final_doublet, right_boundary),
        (left_boundary, final_quartet, right_boundary),
    }
    assert combined_left.data[(left_boundary, final_doublet, right_boundary)].shape == (1, 4, 1)
    assert combined_left.data[(left_boundary, final_quartet, right_boundary)].shape == (1, 4, 1)
    assert combined_left.fusion_legs[1].pipe.coupling == "left"

    recovered_left = split_legs(combined_left, 1)
    np.testing.assert_allclose(
        recovered_left.data[(left_boundary, half, half, half, right_boundary)],
        tensor.data[(left_boundary, half, half, half, right_boundary)],
        atol=1.0e-12,
    )

    combined_right = recouple_fused_leg(combined_left, 1, "right")
    assert combined_right.fusion_legs[1].pipe.coupling == "right"
    recovered_right = split_legs(combined_right, 1)
    np.testing.assert_allclose(
        recovered_right.data[(left_boundary, half, half, half, right_boundary)],
        tensor.data[(left_boundary, half, half, half, right_boundary)],
        atol=1.0e-12,
    )
