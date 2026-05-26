import pytest
import numpy as np

from pyqed.mps.mps import HamiltonianMultiplyU1, _make_complementary_boundary_stack
from pyqed.mps.symmetry import AbelianSector, BlockTensor, Sector, SymmetryManager, QN, zero_like_sector
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


def test_symmetry_manager_can_emit_abelian_point_group_sectors():
    sym = SymmetryManager(["charge", "sz", "pg"], orb_sym=(0, 3))

    assert sym.get_phys_qn(1, "up", site_model="spatial") == AbelianSector(
        ("charge", "sz", "pg"),
        (1, 1, 3),
    )
    assert sym.get_phys_qn(1, "double", site_model="spatial") == AbelianSector(
        ("charge", "sz", "pg"),
        (2, 0, 0),
    )
    left = AbelianSector(("charge", "sz", "pg"), (1, 1, 3))
    right = AbelianSector(("charge", "sz", "pg"), (1, -1, 3))
    assert left + right == AbelianSector(("charge", "sz", "pg"), (2, 0, 0))
    assert sym.get_target_qn(2, 0) == AbelianSector(("charge", "sz", "pg"), (2, 0, 0))


def test_compiled_abelian_two_site_matvec_matches_generic_path():
    rng = np.random.default_rng(7)

    def r(shape):
        return rng.standard_normal(shape)

    E = BlockTensor(
        {
            (0, 10, 20): r((2, 3, 4)),
            (1, 11, 21): r((5, 2, 3)),
        },
        qns=[[0, 1], [10, 11], [20, 21]],
        dirs=[1, 1, -1],
    )
    W1 = BlockTensor(
        {
            (0, 2, 30, 40): r((2, 6, 7, 8)),
            (1, 3, 31, 41): r((5, 4, 6, 9)),
        },
        qns=[[0, 1], [2, 3], [30, 31], [40, 41]],
        dirs=[1, -1, 1, -1],
    )
    W2 = BlockTensor(
        {
            (2, 4, 50, 60): r((6, 3, 2, 5)),
            (3, 5, 51, 61): r((4, 7, 3, 2)),
        },
        qns=[[2, 3], [4, 5], [50, 51], [60, 61]],
        dirs=[1, -1, 1, -1],
    )
    F = BlockTensor(
        {
            (4, 70, 80): r((3, 4, 9)),
            (5, 71, 81): r((7, 5, 6)),
        },
        qns=[[4, 5], [70, 71], [80, 81]],
        dirs=[1, 1, -1],
    )
    A = BlockTensor(
        {
            (20, 80, 40, 60): r((4, 9, 8, 5)),
            (21, 81, 41, 61): r((3, 6, 9, 2)),
        },
        qns=[[20, 21], [80, 81], [40, 41], [60, 61]],
        dirs=[1, -1, 1, 1],
    )

    H = HamiltonianMultiplyU1(E, [W1, W2], F)
    generic = H._matvec_generic(A)
    fused = H._matvec_fused_mpo(A)
    compiled = H.matvec(A)

    assert sorted(compiled.data) == sorted(generic.data)
    for key, block in generic.data.items():
        np.testing.assert_allclose(fused.data[key], block, atol=1e-12)
        np.testing.assert_allclose(compiled.data[key], block, atol=1e-12)


def test_abelian_complementary_boundary_stack_tracks_payloads():
    class Family:
        def __init__(self, entries):
            self.entries = entries

    class Families:
        n_sites = 4
        names = ("R", "P")
        families = {
            "R": Family({(0, 1): 1.0, (2, 3): 2.0}),
            "P": Family({(0, 1, 2, 3): 0.5}),
        }

        def as_metadata(self):
            return {"enabled": True, "family_names": self.names}

    stack, payloads = _make_complementary_boundary_stack(Families(), 4)

    assert stack is not None
    assert payloads[("left", 1)].family_payloads["R"].cross_terms == 1
    assert payloads[("right", 2)].family_payloads["P"].cross_terms == 1
    assert stack.stats["family_names"] == ("R", "P")
    assert stack.stats["n_entries"] == 6
    assert stack.stats["numeric_payload_terms"] > 0
    assert stack.stats["family_operator_tables"] == 6
    assert payloads[("left", 1)].family_operator_table.active_family_names == ("R", "P")


def test_abelian_complementary_split_preserves_full_local_action():
    class Family:
        def __init__(self, entries):
            self.entries = entries

    class Families:
        n_sites = 2
        names = ("R", "P")
        prefer_complementary_payload_tensor_matvec = True
        debug_boundary_channel_matrices = True
        families = {
            "R": Family({(0, 1): 0.2, (1, 0): 0.2}),
            "P": Family({}),
        }

        def get(self, name, default=None):
            return self.families.get(name, default)

    q0 = AbelianSector(("charge", "sz"), (0, 0))
    qu = AbelianSector(("charge", "sz"), (1, 1))
    qd = AbelianSector(("charge", "sz"), (1, -1))
    q2 = AbelianSector(("charge", "sz"), (2, 0))
    phys = [q0, qu, qd, q2]
    rng = np.random.default_rng(11)
    data = {
        (0, 0, p1, p2): rng.standard_normal((1, 1, 1, 1))
        for p1 in phys
        for p2 in phys
    }
    A = BlockTensor(data, [[0], [0], phys, phys], [1, -1, 1, 1])
    E = BlockTensor({(0, 0, 0): np.ones((1, 1, 1))}, [[0], [0], [0]], [1, 1, -1])
    F = BlockTensor({(0, 0, 0): np.ones((1, 1, 1))}, [[0], [0], [0]], [1, 1, -1])
    W_data = {
        (0, 0, p, p): np.ones((1, 1, 1, 1))
        for p in phys
    }
    W = BlockTensor(W_data, [[0], [0], phys, phys], [1, -1, 1, -1])

    split_stats = {"calls": 0, "modes": {}, "bonds": {}}
    H = HamiltonianMultiplyU1(
        E,
        [W, W],
        F,
        complementary_operator_families=Families(),
        bond=0,
        complementary_split_stats=split_stats,
    )
    split = H.split_local_action(A)
    generic = H._matvec_generic(A)
    local = split["local"]

    assert split["mode"] == "local_RP_plus_boundary_direct_operator_table"
    assert local is not None
    assert local.norm() > 0
    assert tuple(split["local_channels"]) == ("R",)
    np.testing.assert_allclose(split["local_channels"]["R"].norm(), local.norm(), atol=1e-12)
    assert "R" in H.complementary_split_metadata["local"]["channels"]
    assert split_stats["calls"] == 1
    assert split_stats["modes"]["local_RP_plus_boundary_direct_operator_table"] == 1
    assert split_stats["bonds"][0]["last"]["boundary_operator"]["n_channels"] > 0
    assert split_stats["bonds"][0]["last"]["boundary_operator"]["channels_materialized"] is False
    assert set(split_stats["bonds"][0]["last"]["local_channels"]) == {"R"}
    assert split["boundary_channels"] == {}
    table = H._boundary_table(A)
    assert table.stats["source"] == "exact_full_mpo_minus_local_RP"
    channel_sum = sum(
        table.channel_matrices.values(),
        np.zeros_like(table.matrix),
    )
    np.testing.assert_allclose(channel_sum, table.matrix, atol=1e-12)
    assert "subtract_local:R" in table.channel_matrices
    for key, block in generic.data.items():
        np.testing.assert_allclose(split["total"].data[key], block, atol=1.0e-12)


def test_abelian_boundary_action_table_reports_family_ownership():
    class Family:
        def __init__(self, entries):
            self.entries = entries

    class Families:
        n_sites = 4
        names = ("R", "P")
        debug_boundary_channel_matrices = True
        families = {
            "R": Family({(0, 1): 0.1, (1, 2): 0.2, (2, 3): 0.3}),
            "P": Family({}),
        }

        def get(self, name, default=None):
            return self.families.get(name, default)

        def as_metadata(self):
            return {"enabled": True, "family_names": self.names}

    stack, payloads = _make_complementary_boundary_stack(Families(), 4)
    q0 = AbelianSector(("charge", "sz"), (0, 0))
    qu = AbelianSector(("charge", "sz"), (1, 1))
    qd = AbelianSector(("charge", "sz"), (1, -1))
    q2 = AbelianSector(("charge", "sz"), (2, 0))
    phys = [q0, qu, qd, q2]
    rng = np.random.default_rng(13)
    A = BlockTensor(
        {
            (0, 0, p1, p2): rng.standard_normal((1, 1, 1, 1))
            for p1 in phys
            for p2 in phys
        },
        [[0], [0], phys, phys],
        [1, -1, 1, 1],
    )
    E = BlockTensor({(0, 0, 0): np.ones((1, 1, 1))}, [[0], [0], [0]], [1, 1, -1])
    F = BlockTensor({(0, 0, 0): np.ones((1, 1, 1))}, [[0], [0], [0]], [1, 1, -1])
    W = BlockTensor(
        {(0, 0, p, p): np.ones((1, 1, 1, 1)) for p in phys},
        [[0], [0], phys, phys],
        [1, -1, 1, -1],
    )
    H = HamiltonianMultiplyU1(
        E,
        [W, W],
        F,
        complementary_operator_families=Families(),
        bond=1,
        complementary_boundary_payloads={
            "stack": stack,
            "left": payloads[("left", 1)],
            "right": payloads[("right", 2)],
        },
    )

    assert H._boundary_family_action_table(A) is None

    table = H._boundary_table(A)
    stats = table.stats

    assert stats["active_family_names"] == ("R",)
    assert tuple(item["side"] for item in stats["boundary_family_tables"]) == ("left", "right")
    assert all(item["active_family_names"] == ("R",) for item in stats["boundary_family_tables"])
    assert any(
        item["name"].startswith("mpo_middle:")
        for item in stats["boundary_channel_tables"]
    )
    assert any(
        item["name"] == "subtract_local:R"
        for item in stats["boundary_channel_tables"]
    )
