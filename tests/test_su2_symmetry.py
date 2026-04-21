import pytest

from pyqed.mps.su2 import (
    SU2Irrep,
    ChargeSpinSector,
    SpatialOrbitalSite,
    SpinOrbitalSite,
    fuse_irreps,
    fuse_charge_spin_sectors,
)


def test_su2_irrep_dimension_and_labels():
    singlet = SU2Irrep(0)
    doublet = SU2Irrep(1)
    triplet = SU2Irrep(2)

    assert singlet.j == pytest.approx(0.0)
    assert singlet.dim == 1
    assert str(singlet) == "S=0"

    assert doublet.j == pytest.approx(0.5)
    assert doublet.dim == 2
    assert str(doublet) == "S=1/2"

    assert triplet.j == pytest.approx(1.0)
    assert triplet.dim == 3
    assert str(triplet) == "S=1"


def test_su2_fusion_rules_match_clebsch_gordan_series():
    half = SU2Irrep(1)
    one = SU2Irrep(2)

    fused_half_half = fuse_irreps(half, half)
    assert fused_half_half == (SU2Irrep(0), SU2Irrep(2))

    fused_one_half = fuse_irreps(one, half)
    assert fused_one_half == (SU2Irrep(1), SU2Irrep(3))


def test_charge_spin_sector_fusion_adds_charge_and_fuses_spin():
    left = ChargeSpinSector(1, SU2Irrep(1))
    right = ChargeSpinSector(1, SU2Irrep(1))

    fused = fuse_charge_spin_sectors(left, right)
    assert fused == (
        ChargeSpinSector(2, SU2Irrep(0)),
        ChargeSpinSector(2, SU2Irrep(2)),
    )


def test_spatial_orbital_su2_site_has_expected_sector_structure():
    site = SpatialOrbitalSite()

    assert site.d == 4
    assert site.labels == ("empty", "up", "down", "double")
    assert site.qn == (
        ChargeSpinSector(0, SU2Irrep(0)),
        ChargeSpinSector(1, SU2Irrep(1)),
        ChargeSpinSector(2, SU2Irrep(0)),
    )
    assert site.degeneracy == (1, 2, 1)
    assert site.state_index == ((0,), (1, 2), (3,))


def test_spin_orbital_site_has_expected_abelian_labels():
    up = SpinOrbitalSite("up")
    down = SpinOrbitalSite("down")

    assert up.d == 2
    assert up.labels == ("empty", "occupied")
    assert up.qn == ((0, 0), (1, 1))
    assert up.degeneracy == (1, 1)
    assert up.state_index == ((0,), (1,))

    assert down.qn == ((0, 0), (1, -1))
