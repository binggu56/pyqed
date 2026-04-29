import pytest

from pyqed.mps.symmetry import AbelianSector, Sector, SymmetryManager, QN, zero_like_sector
from pyqed.mps.su2 import (
    SU2Irrep,
    SpinChargeSector,
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
    left = SpinChargeSector(1, SU2Irrep(1))
    right = SpinChargeSector(1, SU2Irrep(1))

    fused = fuse_charge_spin_sectors(left, right)
    assert fused == (
        SpinChargeSector(2, SU2Irrep(0)),
        SpinChargeSector(2, SU2Irrep(2)),
    )


def test_charge_spin_sector_multiplicity_is_implicit():
    sector = SpinChargeSector(1, SU2Irrep(1))

    assert "multiplicity" not in repr(sector)
    assert sector.multiplicity == 1
    assert sector.dim == 2
    with pytest.raises(ValueError, match="implicit"):
        SpinChargeSector(1, SU2Irrep(1), multiplicity=2)


def test_spatial_orbital_su2_site_has_expected_sector_structure():
    site = SpatialOrbitalSite()

    assert site.d == 4
    assert site.labels == ("empty", "up", "down", "double")
    assert site.qn == (
        SpinChargeSector(0, SU2Irrep(0)),
        SpinChargeSector(1, SU2Irrep(1)),
        SpinChargeSector(2, SU2Irrep(0)),
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


def test_abelian_sector_preserves_qn_like_arithmetic():
    left = AbelianSector(("charge", "sz"), (1, 1))
    right = AbelianSector(("charge", "sz"), (1, -1))

    assert left + right == AbelianSector(("charge", "sz"), (2, 0))
    assert left - right == AbelianSector(("charge", "sz"), (0, 2))
    assert left * 0 == AbelianSector(("charge", "sz"), (0, 0))
    assert zero_like_sector(left) == AbelianSector(("charge", "sz"), (0, 0))
    assert QN(1, 0) + QN(0, 1) == QN(1, 1)


def test_generic_sector_can_host_charge_times_su2():
    left = Sector(("charge", "su2"), (1, SU2Irrep(1)))
    right = Sector(("charge", "su2"), (1, SU2Irrep(1)))

    fused = left.fuse(right)

    assert fused == (
        Sector(("charge", "su2"), (2, SU2Irrep(0))),
        Sector(("charge", "su2"), (2, SU2Irrep(2))),
    )


def test_symmetry_manager_can_emit_abelian_and_su2_sectors():
    abelian = SymmetryManager(["charge", "sz"])
    q_occ_up = abelian.get_phys_qn(0, "occ")
    q_occ_down = abelian.get_phys_qn(1, "occ")

    assert q_occ_up == AbelianSector(("charge", "sz"), (1, 1))
    assert q_occ_down == AbelianSector(("charge", "sz"), (1, -1))
    assert abelian.get_target_qn(6, 0) == AbelianSector(("charge", "sz"), (6, 0))

    su2 = SymmetryManager(["charge", "su2"])
    assert su2.get_vac_qn() == Sector(("charge", "su2"), (0, SU2Irrep(0)))
    assert su2.get_phys_qn(0, "occ") == Sector(("charge", "su2"), (1, SU2Irrep(1)))
    assert su2.get_target_qn(4, 2) == Sector(("charge", "su2"), (4, SU2Irrep(2)))
