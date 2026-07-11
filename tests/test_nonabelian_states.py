import numpy as np

from pyqed.mps.nonabelian import (
    build_product_state,
    build_product_spatial_mps,
    build_random_spatial_mps,
    build_spin_spatial_mps,
    half_filled_singlet_sector,
    spatial_target_sector,
)
from pyqed.mps.su2 import SpatialOrbitalSite, fuse_charge_spin_sectors


def test_half_filled_singlet_sector_has_expected_labels():
    sector = half_filled_singlet_sector(4)
    assert sector.charge == 4
    assert sector.two_j == 0


def test_build_random_spatial_mps_uses_cumulative_targeted_bond_sectors():
    tensors = build_random_spatial_mps(
        4,
        target_sector=half_filled_singlet_sector(4),
        bond_multiplicity=2,
        seed=123,
    )
    site = SpatialOrbitalSite()
    vacuum = site.qn[0]
    target = half_filled_singlet_sector(4)

    assert len(tensors) == 4
    assert tensors[0].qns[0] == [vacuum]
    assert tensors[-1].qns[2] == [target]

    for tensor in tensors[1:-1]:
        assert len(set(tensor.qns[0])) >= 2
        assert len(set(tensor.qns[2])) >= 2

    for tensor in tensors:
        for (q_left, q_phys, q_right), block in tensor.data.items():
            assert q_right in fuse_charge_spin_sectors(q_left, q_phys)
            assert block.shape[1] == len(site.state_index[site.qn.index(q_phys)])


def test_build_random_spatial_mps_is_reproducible_with_seed():
    a = build_random_spatial_mps(4, seed=7)
    b = build_random_spatial_mps(4, seed=7)
    for ta, tb in zip(a, b):
        assert ta.qns == tb.qns
        assert ta.dirs == tb.dirs
        assert set(ta.data) == set(tb.data)
        for key in ta.data:
            np.testing.assert_allclose(ta.data[key], tb.data[key])


def test_custom_target_sector_is_supported():
    target = spatial_target_sector(3, 1)
    tensors = build_random_spatial_mps(3, target_sector=target, seed=5)
    assert tensors[-1].qns[2] == [target]


def test_build_product_spatial_mps_tracks_cumulative_sector_flow():
    tensors = build_product_spatial_mps(["full", "empty", "up", "down"])

    assert len(tensors) == 4
    assert tensors[0].qns[0][0].charge == 0
    assert tensors[-1].qns[2][0].charge == 4
    assert tensors[-1].qns[2][0].two_j == 0

    def nonzero_key(tensor):
        for key, block in tensor.data.items():
            if np.linalg.norm(np.asarray(block)) > 1e-12:
                return key
        raise AssertionError("Expected a nonzero product-state block.")

    first = nonzero_key(tensors[0])
    second = nonzero_key(tensors[1])
    third = nonzero_key(tensors[2])
    fourth = nonzero_key(tensors[3])

    assert first[1].charge == 2
    assert second[1].charge == 0
    assert third[1].charge == 1
    assert fourth[1].charge == 1


def test_build_product_spatial_mps_can_expose_enriched_bond_sector_skeleton():
    tensors = build_product_spatial_mps(["full", "empty", "full", "empty"])

    assert len(tensors[1].qns[2]) > 1
    assert len(tensors[2].qns[0]) > 1

    second = next(iter(tensors[1].data))
    third = next(iter(tensors[2].data))
    assert second[0] in tensors[1].qns[0]
    assert second[2] in tensors[1].qns[2]
    assert third[0] in tensors[2].qns[0]
    assert third[2] in tensors[2].qns[2]


def test_build_product_state_keeps_multiple_total_spin_channels():
    tensors = build_product_state(["up", "down", "up"])

    assert len(tensors) == 3
    total_sectors = {(sector.charge, sector.two_j) for sector in tensors[-1].qns[2]}
    assert total_sectors == {(3, 1), (3, 3)}
    assert len(tensors[1].qns[2]) >= 2


def test_build_spin_spatial_mps_aliases_build_product_state():
    direct = build_product_state(["up", "down", "up"])
    alias = build_spin_spatial_mps(["up", "down", "up"])

    assert len(alias) == len(direct)
    for tensor_alias, tensor_direct in zip(alias, direct):
        assert tensor_alias.qns == tensor_direct.qns
        assert tensor_alias.dirs == tensor_direct.dirs
        assert set(tensor_alias.data) == set(tensor_direct.data)
        for key in tensor_alias.data:
            np.testing.assert_allclose(tensor_alias.data[key], tensor_direct.data[key])


def test_nonabelian_tensor_exposes_blocks_alias():
    tensor = build_product_state(["up"])[0]
    assert tensor.blocks is tensor.data
    assert tensor.nblocks == len(tensor.data)
