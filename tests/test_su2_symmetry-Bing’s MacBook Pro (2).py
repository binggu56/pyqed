import pytest
import numpy as np
import pickle
from types import SimpleNamespace

import pyqed.mps.mps as mps_module
from pyqed.mps.mps import (
    AbelianPackedIdentityLocalEntry,
    AbelianPackedLocalGeneratorEntry,
    AbelianRenormalizedOperatorActionTable,
    HamiltonianMultiplyU1,
    MovingEnvironment,
    _abelian_data_factor_list,
    _make_complementary_boundary_stack,
    dense_to_symmetric,
    dense_to_symmetric_mpo,
    contract_from_left,
    contract_from_right,
    initial_E,
    initial_F,
    optimize_two_sites,
    svd_symmetric,
    symmetric_to_dense,
    two_site_dmrg,
)
from pyqed.mps.dmrg import (
    DMRG,
    _right_canonicalize_symmetric_factors,
    dmrg_matvec_options,
)
from pyqed.mps.abelian_direct import (
    AbelianContextualBoundaryBatch,
    AbelianContextualDirectFamilyBuilder,
    AbelianContextualFamilyBuildOptions,
    AbelianCompositePackedDirectFamilyEntries,
    AbelianDirectFamilyLiteralPlan,
    AbelianDirectRoutePlan,
    AbelianEnvironmentTensorData,
    AbelianPackedBoundaryTensor,
    AbelianPackedContextualBoundaryTable,
    AbelianPackedDirectFamilyEntries,
    AbelianPlannedPackedDirectFamilyEntries,
    AbelianPackedTensorViewCache,
    AbelianSameSidePRouteIdentityEntries,
    AbelianSiteTensorData,
    AbelianSpatialLocalOperatorBuilder,
    AbelianTwoSiteSplitResult,
    AbelianTwoSiteUpdateData,
    abelian_merge_adjacent_site_tensors,
    abelian_site_tensors_from_split,
    abelian_split_flat_two_site_svd_data,
)
from pyqed.mps.abelian_storage import (
    abelian_environment_scalar,
    make_identity_mpo_site_from_mps_site,
)
from pyqed.mps import cpp_davidson
from pyqed.mps import packed_cython
from pyqed.mps.symmetry import AbelianSector, BlockTensor, Sector, SymmetryManager, QN, tensordot, zero_like_sector
from pyqed.mps.su2 import (
    SU2Irrep,
    SpinChargeSector,
    SpatialOrbitalSite,
    SpinOrbitalSite,
    fuse_irreps,
    fuse_charge_spin_sectors,
)


def test_abelian_sector_pickle_roundtrip():
    sector = AbelianSector(("charge", "sz"), (3, -1))
    restored = pickle.loads(pickle.dumps(sector))

    assert restored == sector
    assert restored.labels == sector.labels
    assert restored.components == sector.components


def test_qn_pickle_roundtrip():
    qn = QN(2, 0)
    restored = pickle.loads(pickle.dumps(qn))

    assert restored == qn
    assert isinstance(restored, QN)


def test_contextual_family_options_precompute_for_packed_route_default():
    opts = AbelianContextualFamilyBuildOptions.from_matvec_options(
        {"generator_table_packed_route_table": "auto"}
    )

    assert opts.precompute_boundaries is True
    assert opts.precompute_min_records == 0

    opts = AbelianContextualFamilyBuildOptions.from_matvec_options(
        {"generator_table_packed_route_table": "python"}
    )
    assert opts.precompute_boundaries is False

    opts = AbelianContextualFamilyBuildOptions.from_matvec_options(
        {
            "generator_table_packed_route_table": "auto",
            "generator_table_precompute_contextual_boundaries": False,
        }
    )
    assert opts.precompute_boundaries is False


def test_contextual_family_precompute_routes_through_owner():
    class Owner:
        def __init__(self):
            self.calls = 0
            self.left_keys = None
            self.right_keys = None

        def precompute_contextual_boundaries(
            self,
            left_keys,
            right_keys,
            *_args,
            **_kwargs,
        ):
            self.calls += 1
            self.left_keys = tuple(left_keys)
            self.right_keys = tuple(right_keys)
            return (
                (("left-value",),),
                (("right-value",),),
                (7,),
                (8,),
                2,
                3,
                4,
                5,
                0.1,
                0.2,
                False,
                True,
            )

    def unexpected_builder(*_args, **_kwargs):
        raise AssertionError("owner precompute should bypass Python builders")

    phases = []
    owner = Owner()
    builder = AbelianContextualDirectFamilyBuilder(
        stats={},
        record_phase=lambda *args, **kwargs: phases.append((args, kwargs)),
        left_builder=unexpected_builder,
        right_builder=unexpected_builder,
        left_batch_builder=None,
        right_batch_builder=None,
        boundary_batch_owner=owner,
        fallback_builder=unexpected_builder,
    )
    records = [((0,), "A", "B", (1,), 1.0)]

    batch = builder.precompute_boundaries("P", records)

    assert owner.calls == 1
    assert owner.left_keys == (((0,), "A"),)
    assert owner.right_keys == (((1,), "B"),)
    assert batch.left[((0,), "A")] == ("left-value",)
    assert batch.right[((1,), "B")] == ("right-value",)
    assert batch.left_table_ids == (7,)
    assert batch.right_table_ids == (8,)
    stats = builder.stats["contextual_boundary_precompute_owner"]
    assert int(stats["successes"]) == 1
    assert stats["last_used"] is True
    precompute_stats = builder.stats["contextual_boundary_precompute"]
    assert int(precompute_stats["owner_successes"]) == 1
    assert precompute_stats["last_owner_used"] is True
    assert phases[-1][0][0] == "contextual_boundary_precompute"
    assert phases[-1][1]["owner"] == 1


def test_contextual_family_build_entries_accepts_table_id_only_precompute():
    def unexpected_builder(*_args, **_kwargs):
        raise AssertionError("table-id-only precompute should not call builders")

    builder = AbelianContextualDirectFamilyBuilder(
        stats={},
        record_phase=lambda *_args, **_kwargs: None,
        left_builder=unexpected_builder,
        right_builder=unexpected_builder,
        left_batch_builder=None,
        right_batch_builder=None,
        enable_packed_boundary_tables=True,
        left_packed_boundary_table=AbelianPackedContextualBoundaryTable(side="left"),
        right_packed_boundary_table=AbelianPackedContextualBoundaryTable(side="right"),
        fallback_builder=unexpected_builder,
    )
    records = [((0,), "A", "B", (1,), 1.0)]
    route_plan = AbelianDirectRoutePlan.from_records("P", records, bond=0)
    boundary_batch = AbelianContextualBoundaryBatch(
        {},
        {},
        (),
        (),
        (0,),
        (0,),
        {"packed": 1},
        {"packed": 1},
    )
    options = AbelianContextualFamilyBuildOptions(
        precompute_boundaries=True,
        pack_entries=True,
        packed_buffer=True,
    )

    result = builder.build_entries("P", route_plan, options=options, boundary_batch=boundary_batch)

    assert result.entries._pyqed_planned_direct_family_table_ids is True
    assert tuple(result.entries.left_values) == ()
    assert tuple(result.entries.right_values) == ()
    fast = builder.stats["contextual_route_fast_pack"]
    assert int(fast["table_backed_calls"]) == 1
    assert int(fast["packed_boundary_calls"]) == 1
    assert int(builder.stats["contextual_recursive_terms"]) == 1


def test_contextual_route_plan_signature_is_content_stable():
    records_a = (
        (("I",), "A", "B", ("I",), 1.0 + 0.0j),
        (("X",), "C", "D", ("Y",), -0.25 + 0.5j),
    )
    records_b = tuple(tuple(row) for row in records_a)

    plan_a = AbelianDirectRoutePlan.from_records("P", records_a, bond=1)
    plan_b = AbelianDirectRoutePlan.from_records("P", records_b, bond=1)

    assert id(records_a) != id(records_b)
    assert plan_a.signature == plan_b.signature


def _dense_mps_norm(factors):
    env = np.ones((1, 1), dtype=np.result_type(*[np.asarray(f).dtype for f in factors], complex))
    for factor in factors:
        factor = np.asarray(factor)
        env = np.einsum("ab,asr,bsk->rk", env, factor.conj(), factor, optimize=True)
    return float(np.real(env.reshape(-1)[0]))


def _dense_mps_expectation_mpo(factors, mpo):
    factors = [np.asarray(factor) for factor in factors]
    mpo = [np.asarray(site) for site in mpo]
    env = np.ones((1, 1, 1), dtype=np.result_type(*[f.dtype for f in factors], *[w.dtype for w in mpo], complex))
    for factor, site in zip(factors, mpo):
        tmp = np.tensordot(env, factor.conj(), axes=([0], [0]))
        tmp = np.tensordot(tmp, site, axes=([1, 2], [0, 2]))
        tmp = np.tensordot(tmp, factor, axes=([0, 3], [0, 1]))
        env = np.transpose(tmp, (0, 2, 1))
    return float(np.real(env.reshape(-1)[0] / _dense_mps_norm(factors)))


def _dense_mps_state(factors):
    factors = [np.asarray(factor) for factor in factors]
    state = factors[0][0]
    for factor in factors[1:]:
        state = np.tensordot(state, factor, axes=([-1], [0]))
    return state[..., 0]


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


def _two_sector_tied_split(delta):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    data = {
        (q0, q0, q0, q0): np.array([[[[1.0]]]], dtype=float),
        (q0, q0, q1, q0): np.array([[[[1.0 + delta]]]], dtype=float),
    }
    tensor = BlockTensor(
        data,
        qns=[[q0], [q0], [q0, q1], [q0]],
        dirs=[1, -1, 1, 1],
    )
    return tensor, q0, q1


def test_svd_symmetric_truncation_uses_deterministic_sector_tie_break():
    tensor, q0, _q1 = _two_sector_tied_split(delta=5e-11)

    _U, _V, singular_blocks, _trunc_err, kept = svd_symmetric(tensor, m_max=1)

    assert kept == 1
    assert tuple(singular_blocks) == (q0,)


def test_svd_symmetric_truncation_respects_resolved_sector_gap():
    tensor, _q0, q1 = _two_sector_tied_split(delta=5e-9)

    _U, _V, singular_blocks, _trunc_err, kept = svd_symmetric(tensor, m_max=1)

    assert kept == 1
    assert tuple(singular_blocks) == (q1,)


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
    matrix_chain = H._matvec_generic_matrix_chain(A)
    compact_chain = H._matvec_compact_matrix_chain(A)
    batched_compact_chain = H._matvec_batched_compact_matrix_chain(A)
    H._batched_compact_matrix_chain_compiled_kernel = True
    compiled_batched_compact_chain = H._matvec_batched_compact_matrix_chain(A)
    H._batched_compact_matrix_chain_compiled_kernel = False
    H._batched_compact_matrix_chain_compiled_parallel_kernel = True
    parallel_compiled_batched_compact_chain = H._matvec_batched_compact_matrix_chain(A)
    parallel_mode = H.profile_stats["batched_compact_matrix_chain"]["last"]["compiled_kernel_mode"]
    H._batched_compact_matrix_chain_compiled_parallel_min_work = 10**18
    thresholded_batched_compact_chain = H._matvec_batched_compact_matrix_chain(A)
    thresholded_mode = H.profile_stats["batched_compact_matrix_chain"]["last"]["compiled_kernel_mode"]
    native_compact_chain = H._matvec_native_compact_matrix_chain(A)
    compiled = H.matvec(A)

    assert sorted(compiled.data) == sorted(generic.data)
    assert sorted(matrix_chain.data) == sorted(generic.data)
    assert sorted(compact_chain.data) == sorted(generic.data)
    assert sorted(batched_compact_chain.data) == sorted(generic.data)
    assert sorted(compiled_batched_compact_chain.data) == sorted(generic.data)
    assert sorted(parallel_compiled_batched_compact_chain.data) == sorted(generic.data)
    assert sorted(thresholded_batched_compact_chain.data) == sorted(generic.data)
    assert parallel_mode == "parallel"
    assert thresholded_mode == "blas"
    if native_compact_chain is not None:
        assert sorted(native_compact_chain.data) == sorted(generic.data)
    for key, block in generic.data.items():
        np.testing.assert_allclose(fused.data[key], block, atol=1e-12)
        np.testing.assert_allclose(matrix_chain.data[key], block, atol=1e-12)
        np.testing.assert_allclose(compact_chain.data[key], block, atol=1e-12)
        np.testing.assert_allclose(batched_compact_chain.data[key], block, atol=1e-12)
        np.testing.assert_allclose(compiled_batched_compact_chain.data[key], block, atol=1e-12)
        np.testing.assert_allclose(parallel_compiled_batched_compact_chain.data[key], block, atol=1e-12)
        np.testing.assert_allclose(thresholded_batched_compact_chain.data[key], block, atol=1e-12)
        if native_compact_chain is not None:
            np.testing.assert_allclose(native_compact_chain.data[key], block, atol=1e-12)
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

    native_A = AbelianSiteTensorData(A.data, A.qns, A.dirs)
    native_local = H._matvec_local_complementary(native_A)
    native_channels = H._matvec_local_complementary_channels(native_A)
    assert isinstance(native_local, AbelianSiteTensorData)
    assert tuple(native_channels) == ("R",)
    assert isinstance(native_channels["R"], AbelianSiteTensorData)
    assert native_local.qns == tuple(tuple(axis) for axis in local.qns)
    assert native_local.dirs == tuple(local.dirs)
    assert set(native_local.data) == set(local.data)
    for key, block in local.data.items():
        np.testing.assert_allclose(native_local.data[key], block, atol=1.0e-12)

    legacy_middle = H._matvec_middle_mpo_channel(A, 0)
    native_middle = H._matvec_middle_mpo_channel(native_A, 0)
    assert isinstance(native_middle, AbelianSiteTensorData)
    assert native_middle.qns == tuple(tuple(axis) for axis in legacy_middle.qns)
    assert native_middle.dirs == tuple(legacy_middle.dirs)
    assert set(native_middle.data) == set(legacy_middle.data)
    for key, block in legacy_middle.data.items():
        np.testing.assert_allclose(native_middle.data[key], block, atol=1.0e-12)

    legacy_boundary_table = H._matvec_boundary_table(A)
    native_boundary_table = H._matvec_boundary_table(native_A)
    assert isinstance(native_boundary_table, AbelianSiteTensorData)
    assert native_boundary_table.qns == tuple(tuple(axis) for axis in legacy_boundary_table.qns)
    assert native_boundary_table.dirs == tuple(legacy_boundary_table.dirs)
    assert set(native_boundary_table.data) == set(legacy_boundary_table.data)
    for key, block in legacy_boundary_table.data.items():
        np.testing.assert_allclose(
            native_boundary_table.data[key],
            block,
            atol=1.0e-12,
        )

    native_plan = H._build_boundary_direct_operator_plan(native_A)
    native_direct, _direct_channels, _direct_entries = H._apply_direct_operator_plan(
        native_A,
        native_plan,
    )
    assert isinstance(native_direct, AbelianSiteTensorData)

    native_boundary = H._matvec_boundary_factorized(
        native_A,
        local_channels=native_channels,
        collect_channels=True,
    )
    if native_boundary is not None:
        assert isinstance(native_boundary["total"], AbelianSiteTensorData)
        for channel_tensor in native_boundary["channels"].values():
            assert isinstance(channel_tensor, AbelianSiteTensorData)


def test_dense_to_symmetric_mpo_preserves_degenerate_physical_sectors():
    q0 = AbelianSector(("charge", "sz"), (0, 0))
    q1 = AbelianSector(("charge", "sz"), (1, 1))
    ndeg = 3
    phys = [q0] * ndeg + [q1] * ndeg
    site_qn_maps = [dict(enumerate(phys))]

    local = np.zeros((2 * ndeg, 2 * ndeg))
    phonon_block = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.5, 2.0], [0.0, 2.0, 1.0]]
    )
    local[:ndeg, :ndeg] = phonon_block
    local[ndeg:, ndeg:] = 2.0 * phonon_block
    W = local.reshape(1, 1, 2 * ndeg, 2 * ndeg)

    (sym_W,) = dense_to_symmetric_mpo([W], site_qn_maps)

    assert sym_W.data[(q0 * 0, q0 * 0, q0, q0)].shape == (1, 1, ndeg, ndeg)
    assert sym_W.data[(q0 * 0, q0 * 0, q1, q1)].shape == (1, 1, ndeg, ndeg)
    np.testing.assert_allclose(
        sym_W.data[(q0 * 0, q0 * 0, q0, q0)][0, 0],
        phonon_block,
    )
    np.testing.assert_allclose(
        sym_W.data[(q0 * 0, q0 * 0, q1, q1)][0, 0],
        2.0 * phonon_block,
    )


def test_dense_to_symmetric_roundtrips_entangled_fixed_charge_mps():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    A0 = np.zeros((1, 2, 2), dtype=complex)
    A1 = np.zeros((2, 2, 1), dtype=complex)
    A0[0, 0, 0] = 1.0
    A0[0, 1, 1] = 1.0
    A1[0, 1, 0] = inv_sqrt2
    A1[1, 0, 0] = inv_sqrt2

    sym = dense_to_symmetric([A0, A1], phys_qns=[q0, q1])

    assert sym[0].qns[1] == [q0, q1]
    assert set(sym[0].data) == {(q0, q0, q0), (q0, q1, q1)}
    assert set(sym[1].data) == {(q0, q1, q1), (q1, q1, q0)}

    class State:
        factors = sym
        labels = ["lv", "rv", "p"]

    dense = symmetric_to_dense(State()).factors
    np.testing.assert_allclose(_dense_mps_state(dense), _dense_mps_state([A0, A1]))


def test_dense_to_symmetric_can_emit_native_site_data():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    A0 = np.zeros((1, 2, 2), dtype=complex)
    A1 = np.zeros((2, 2, 1), dtype=complex)
    A0[0, 0, 0] = 1.0
    A0[0, 1, 1] = 1.0
    A1[0, 1, 0] = inv_sqrt2
    A1[1, 0, 0] = inv_sqrt2

    legacy = dense_to_symmetric([A0, A1], phys_qns=[q0, q1])
    native = dense_to_symmetric(
        [A0, A1],
        phys_qns=[q0, q1],
        native_site_storage=True,
    )

    assert all(isinstance(site, AbelianSiteTensorData) for site in native)
    assert not any(isinstance(site, BlockTensor) for site in native)
    for legacy_site, native_site in zip(legacy, native):
        assert [list(axis) for axis in native_site.qns] == [
            list(axis) for axis in legacy_site.qns
        ]
        assert list(native_site.dirs) == list(legacy_site.dirs)
        assert set(native_site.data) == set(legacy_site.data)
        for key, block in legacy_site.data.items():
            np.testing.assert_allclose(native_site.data[key], block)


def test_identity_mpo_site_helper_preserves_native_storage():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    psi0 = [
        np.array([0.0, 1.0]).reshape(1, 2, 1),
        np.array([1.0, 0.0]).reshape(1, 2, 1),
    ]

    legacy_site = dense_to_symmetric(psi0, phys_qns=[q0, q1])[0]
    native_site = dense_to_symmetric(
        psi0,
        phys_qns=[q0, q1],
        native_site_storage=True,
    )[0]

    legacy_identity = make_identity_mpo_site_from_mps_site(legacy_site)
    native_identity = make_identity_mpo_site_from_mps_site(native_site)

    assert isinstance(legacy_identity, BlockTensor)
    assert isinstance(native_identity, AbelianSiteTensorData)
    assert not isinstance(native_identity, BlockTensor)
    assert [list(axis) for axis in native_identity.qns] == [
        list(axis) for axis in legacy_identity.qns
    ]
    assert list(native_identity.dirs) == list(legacy_identity.dirs)
    assert set(native_identity.data) == set(legacy_identity.data)
    for key, block in legacy_identity.data.items():
        np.testing.assert_allclose(native_identity.data[key], block)


def test_abelian_environment_scalar_matches_native_legacy_and_dense():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    data = {
        (q0, q0, q0): np.array([[[1.25]]]),
        (q0, q1, q1): np.array([[[2.5]]]),
    }
    legacy = BlockTensor(data, [[q0], [q0, q1], [q0, q1]], [1, -1, 1])
    native = AbelianEnvironmentTensorData(legacy.data, legacy.qns, legacy.dirs)

    assert abelian_environment_scalar(native) == pytest.approx(3.75)
    assert abelian_environment_scalar(legacy) == pytest.approx(3.75)
    assert abelian_environment_scalar(np.array([[[3.75]]])) == pytest.approx(3.75)


def test_abelian_packed_tensor_view_cache_discards_changed_tensor():
    q0 = AbelianSector(("charge",), (0,))
    tensor = AbelianSiteTensorData(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, -1, 1],
    )
    cache = AbelianPackedTensorViewCache(source_prefix="test")

    first = cache.view(tensor, "site")
    assert cache.view(tensor, "site-again") is first
    first_conj = cache.conj(tensor, "site-conj")
    assert cache.conj(tensor, "site-conj-again") is first_conj
    assert cache.stats["view_cache"] == 1
    assert cache.stats["conj_cache"] == 1

    assert cache.discard(tensor) == 2
    assert cache.stats["view_cache"] == 0
    assert cache.stats["conj_cache"] == 0
    assert cache.stats["discarded"] == 2
    rebuilt = cache.view(tensor, "site-rebuilt")
    assert rebuilt is not first


def test_initial_environments_preserve_native_storage():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}]
    W = np.eye(2, dtype=complex).reshape(1, 1, 2, 2)

    legacy_mpo = dense_to_symmetric_mpo([W], site_qn_maps)
    native_mpo = dense_to_symmetric_mpo(
        [W],
        site_qn_maps,
        native_site_storage=True,
    )

    legacy_E = initial_E(legacy_mpo[0])
    legacy_F = initial_F(legacy_mpo[0], target_qn=q1)
    native_E = initial_E(native_mpo[0])
    native_F = initial_F(native_mpo[0], target_qn=q1)

    assert isinstance(legacy_E, BlockTensor)
    assert isinstance(legacy_F, BlockTensor)
    assert isinstance(native_E, AbelianEnvironmentTensorData)
    assert isinstance(native_F, AbelianEnvironmentTensorData)
    assert not isinstance(native_E, BlockTensor)
    assert not isinstance(native_F, BlockTensor)
    assert [list(axis) for axis in native_E.qns] == [list(axis) for axis in legacy_E.qns]
    assert [list(axis) for axis in native_F.qns] == [list(axis) for axis in legacy_F.qns]
    assert set(native_E.data) == set(legacy_E.data)
    assert set(native_F.data) == set(legacy_F.data)
    np.testing.assert_allclose(initial_E(W), np.ones((1, 1, 1)))
    np.testing.assert_allclose(initial_F(W), np.ones((1, 1, 1)))


def test_right_canonicalize_symmetric_factors_preserves_state():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    inv_sqrt3 = 1.0 / np.sqrt(3.0)

    A0 = np.zeros((1, 2, 2), dtype=complex)
    A1 = np.zeros((2, 2, 2), dtype=complex)
    A2 = np.zeros((2, 2, 1), dtype=complex)
    A0[0, 0, 0] = 1.0
    A0[0, 1, 1] = 1.0
    A1[0, 0, 0] = 1.0
    A1[0, 1, 1] = 1.0
    A1[1, 0, 1] = 1.0
    A2[0, 1, 0] = inv_sqrt3
    A2[1, 0, 0] = inv_sqrt3
    dense_state = _dense_mps_state([A0, A1, A2])
    dense_state = dense_state / np.linalg.norm(dense_state)

    sym = dense_to_symmetric([A0, A1, A2], phys_qns=[q0, q1])
    canonical = _right_canonicalize_symmetric_factors(sym)

    class State:
        factors = canonical
        labels = ["lv", "rv", "p"]

    dense = symmetric_to_dense(State()).factors
    canonical_state = _dense_mps_state(dense)
    canonical_state = canonical_state / np.linalg.norm(canonical_state)
    phase = np.vdot(dense_state.reshape(-1), canonical_state.reshape(-1))
    np.testing.assert_allclose(
        canonical_state * np.exp(-1j * np.angle(phase)),
        dense_state,
        atol=1.0e-12,
    )


def test_dense_to_symmetric_roundtrips_entangled_spatial_charge_sz_mps():
    q_empty = AbelianSector(("charge", "sz"), (0, 0))
    q_up = AbelianSector(("charge", "sz"), (1, 1))
    q_down = AbelianSector(("charge", "sz"), (1, -1))
    q_full = AbelianSector(("charge", "sz"), (2, 0))
    phys = [q_empty, q_up, q_down, q_full]
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    A0 = np.zeros((1, 4, 2), dtype=complex)
    A1 = np.zeros((2, 4, 1), dtype=complex)
    A0[0, 0, 0] = 1.0
    A0[0, 3, 1] = 1.0
    A1[0, 3, 0] = inv_sqrt2
    A1[1, 0, 0] = inv_sqrt2

    sym = dense_to_symmetric([A0, A1], phys_qns=phys)

    assert sym[0].qns[1] == [q_empty, q_full]
    assert set(sym[0].data) == {
        (q_empty, q_empty, q_empty),
        (q_empty, q_full, q_full),
    }
    assert set(sym[1].data) == {
        (q_empty, q_full, q_full),
        (q_full, q_full, q_empty),
    }

    class State:
        factors = sym
        labels = ["lv", "rv", "p"]

    site_qn_maps = [dict(enumerate(phys)), dict(enumerate(phys))]
    dense = symmetric_to_dense(State(), site_qn_maps=site_qn_maps).factors
    np.testing.assert_allclose(_dense_mps_state(dense), _dense_mps_state([A0, A1]))


def test_symmetric_to_dense_expands_degenerate_physical_sectors():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    data = {
        (q0, q0, q0): np.array([[[1.0, 2.0]]]),
        (q0, q1, q1): np.array([[[3.0, 4.0]]]),
    }
    tensor = BlockTensor(data, [[q0], [q0, q1], [q0, q1]], [-1, 1, 1])

    class State:
        factors = [tensor]
        labels = ["lv", "rv", "p"]

    dense = symmetric_to_dense(State()).factors[0]

    assert dense.shape == (1, 4, 2)
    np.testing.assert_allclose(dense[0, :, 0], [1.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(dense[0, :, 1], [0.0, 0.0, 3.0, 4.0])


def test_symmetric_to_dense_uses_site_qn_map_physical_order():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    data = {
        (q0, q0, q0): np.array([[[1.0, 2.0]]]),
        (q0, q1, q1): np.array([[[3.0, 4.0]]]),
    }
    tensor = BlockTensor(data, [[q0], [q0, q1], [q0, q1]], [-1, 1, 1])

    class State:
        factors = [tensor]
        labels = ["lv", "rv", "p"]

    site_qn_maps = [{0: q1, 1: q1, 2: q0, 3: q0}]
    dense = symmetric_to_dense(State(), site_qn_maps=site_qn_maps).factors[0]

    assert dense.shape == (1, 4, 2)
    np.testing.assert_allclose(dense[0, :, 0], [0.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(dense[0, :, 1], [3.0, 4.0, 0.0, 0.0])


def test_abelian_dmrg_uses_davidson_not_dense_local_matrix(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W0[0, 3] = np.zeros((2, 2))
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(terms, start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    psi0 = [np.array([0.0, 1.0]).reshape(1, 2, 1), np.array([1.0, 0.0]).reshape(1, 2, 1)]
    phys_qns = [q0, q1]
    init = dense_to_symmetric(psi0, phys_qns=phys_qns)

    def fail_dense_matrix(self, proto, max_dim=256):
        raise AssertionError("Abelian DMRG should use Davidson, not dense local diagonalization.")

    monkeypatch.setattr(HamiltonianMultiplyU1, "dense_matrix", fail_dense_matrix)

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
    ).run()

    assert dmrg.e_tot == pytest.approx(-1.0, abs=1.0e-8)


def test_abelian_dmrg_packed_local_davidson_matches_two_site_hopping(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W0[0, 3] = np.zeros((2, 2))
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(terms, start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    psi0 = [
        np.array([0.0, 1.0]).reshape(1, 2, 1),
        np.array([1.0, 0.0]).reshape(1, 2, 1),
    ]
    init = dense_to_symmetric(psi0, phys_qns=[q0, q1])

    calls = {"n": 0}
    original = HamiltonianMultiplyU1.solve_packed_davidson

    def wrapped(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HamiltonianMultiplyU1, "solve_packed_davidson", wrapped)

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
        abelian_matvec_options={
            "packed_local_davidson": True,
            "packed_local_davidson_restart_dim": 8,
            "packed_local_block_preconditioner": True,
            "packed_local_block_preconditioner_max_block_dim": 8,
            "packed_local_block_preconditioner_max_total_dim": 64,
        },
    ).run()

    assert calls["n"] > 0
    assert getattr(optimize_two_sites, "last_AA_flat", None) is not None
    assert getattr(optimize_two_sites, "last_AA_layout", None) is not None
    assert isinstance(
        getattr(optimize_two_sites, "last_split_result", None),
        AbelianTwoSiteSplitResult,
    )
    assert isinstance(
        getattr(optimize_two_sites, "last_native_site_tensors", None),
        AbelianTwoSiteUpdateData,
    )
    assert dmrg.abelian_matvec_options["native_site_storage"] is True
    assert not getattr(optimize_two_sites, "last_split_legacy_wrapped", True)
    assert dmrg.e_tot == pytest.approx(-1.0, abs=1.0e-8)


def test_abelian_dmrg_native_site_storage_keeps_native_factors():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(
        [(-cd @ parity, c), (-parity @ c, cd)],
        start=1,
    ):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
        abelian_matvec_options={
            "native_site_storage": True,
            "packed_local_davidson": True,
            "packed_local_davidson_restart_dim": 8,
            "packed_local_block_preconditioner": True,
            "packed_local_block_preconditioner_max_block_dim": 8,
            "packed_local_block_preconditioner_max_total_dim": 64,
        },
    ).run()

    assert dmrg.e_tot == pytest.approx(-1.0, abs=1.0e-8)
    assert all(
        isinstance(site, AbelianSiteTensorData)
        for site in dmrg.ground_state.factors
    )
    assert not getattr(mps_module.optimize_two_sites, "last_split_legacy_wrapped", True)


def test_abelian_dmrg_native_dense_guess_reaches_solver_without_blocktensor(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    ident = np.eye(2)
    number = np.diag([0.0, 1.0])
    W0 = np.zeros((1, 2, 2, 2))
    W1 = np.zeros((2, 1, 2, 2))
    W0[0, 0] = 0.25 * number
    W0[0, 1] = ident
    W1[0, 0] = ident
    W1[1, 0] = 0.75 * number
    mpo = dense_to_symmetric_mpo(
        [W0, W1],
        site_qn_maps,
        native_site_storage=True,
    )

    captured = {}

    def fail_blocktensor_init(self, *args, **kwargs):
        raise AssertionError("native DMRG dense-guess path constructed BlockTensor")

    def fake_optimize(A, B, W_left, W_right, E, F, m, direction, *args, **kwargs):
        captured["site_tensors"] = (
            isinstance(A, AbelianSiteTensorData),
            isinstance(B, AbelianSiteTensorData),
        )
        captured["mpo_tensors"] = (
            isinstance(W_left, AbelianSiteTensorData),
            isinstance(W_right, AbelianSiteTensorData),
        )
        captured["environment_tensors"] = (
            isinstance(E, AbelianEnvironmentTensorData),
            isinstance(F, AbelianEnvironmentTensorData),
        )
        captured["native_site_storage"] = kwargs["matvec_options"][
            "native_site_storage"
        ]
        return 0.0, A, B, 0.0, 1

    monkeypatch.setattr(BlockTensor, "__init__", fail_blocktensor_init)
    monkeypatch.setattr(mps_module, "optimize_two_sites", fake_optimize)

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=[
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        nsweeps=1,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        site_qn_maps=site_qn_maps,
        performance="block2-like",
        final_expectation=False,
    ).run()

    assert captured["site_tensors"] == (True, True)
    assert captured["mpo_tensors"] == (True, True)
    assert captured["environment_tensors"] == (True, True)
    assert captured["native_site_storage"] is True
    assert dmrg.e_tot == pytest.approx(0.0)
    assert all(isinstance(site, AbelianSiteTensorData) for site in dmrg.ground_state.factors)


def test_mps_dmrg_main_does_not_expose_legacy_blocktensor_surface():
    import importlib

    mps_dmrg_module = importlib.import_module("pyqed.mps.dmrg")
    storage_module = importlib.import_module("pyqed.mps.abelian_storage")

    assert not hasattr(mps_dmrg_module, "BlockTensor")
    assert not hasattr(mps_dmrg_module, "SYMMETRY_AVAILABLE")
    assert not hasattr(mps_dmrg_module, "make_legacy_abelian_tensor")
    assert (
        mps_dmrg_module.make_identity_mpo_site_from_mps_site
        is storage_module.make_identity_mpo_site_from_mps_site
    )


def test_abelian_dmrg_native_site_storage_checkpoint_keeps_native_factors(tmp_path):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(
        [(-cd @ parity, c), (-parity @ c, cd)],
        start=1,
    ):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)
    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )

    checkpoint = tmp_path / "native_dmrg.pkl"
    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
        checkpoint_path=checkpoint,
        performance="packed-block-fast",
    ).run()

    payload = DMRG.load_checkpoint(checkpoint)

    assert dmrg.e_tot == pytest.approx(-1.0, abs=1.0e-8)
    assert payload["final"] is True
    assert payload["params"]["performance"] == "packed-block-fast"
    assert payload["params"]["native_site_storage"] is True
    assert all(isinstance(site, AbelianSiteTensorData) for site in payload["mps"])
    assert not any(isinstance(site, BlockTensor) for site in payload["mps"])


def test_abelian_data_factor_list_converts_only_when_enabled():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    factors = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )

    disabled = _abelian_data_factor_list(factors, native_site_storage=False)
    enabled = _abelian_data_factor_list(factors, native_site_storage=True)
    enabled_again = _abelian_data_factor_list(enabled, native_site_storage=True)

    assert disabled is factors
    assert all(isinstance(site, BlockTensor) for site in factors)
    assert all(isinstance(site, AbelianSiteTensorData) for site in enabled)
    assert not any(isinstance(site, BlockTensor) for site in enabled)
    assert enabled_again is not enabled
    assert all(isinstance(site, AbelianSiteTensorData) for site in enabled_again)
    assert enabled_again[0] is enabled[0]


def test_abelian_packed_policy_defaults_to_native_site_storage(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(
        [(-cd @ parity, c), (-parity @ c, cd)],
        start=1,
    ):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)
    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )

    sym_mgr = SymmetryManager(["charge"])
    first_local = {}
    original_optimize = mps_module.optimize_two_sites

    def wrapped_optimize(A, B, W1, W2, *args, **kwargs):
        first_local.setdefault(
            "mps_data",
            (
                isinstance(A, AbelianSiteTensorData),
                isinstance(B, AbelianSiteTensorData),
            ),
        )
        first_local.setdefault(
            "mpo_data",
            (
                isinstance(W1, AbelianSiteTensorData),
                isinstance(W2, AbelianSiteTensorData),
            ),
        )
        first_local.setdefault(
            "mpo_blocktensor",
            (
                isinstance(W1, BlockTensor),
                isinstance(W2, BlockTensor),
            ),
        )
        return original_optimize(A, B, W1, W2, *args, **kwargs)

    monkeypatch.setattr(mps_module, "optimize_two_sites", wrapped_optimize)

    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
        performance="packed-block-fast",
    ).run()

    assert dmrg.abelian_matvec_options["native_site_storage"] is True
    assert first_local["mps_data"] == (True, True)
    assert first_local["mpo_data"] == (True, True)
    assert first_local["mpo_blocktensor"] == (False, False)
    assert dmrg.e_tot == pytest.approx(-1.0, abs=1.0e-8)
    assert all(
        isinstance(site, AbelianSiteTensorData)
        for site in dmrg.ground_state.factors
    )
    assert not getattr(mps_module.optimize_two_sites, "last_split_legacy_wrapped", True)


def test_abelian_fast_policy_uses_native_site_storage(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(
        [(-cd @ parity, c), (-parity @ c, cd)],
        start=1,
    ):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)
    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )

    first_local = {}
    original_optimize = mps_module.optimize_two_sites

    def wrapped_optimize(A, B, W1_local, W2_local, *args, **kwargs):
        first_local.setdefault(
            "mps",
            (
                isinstance(A, AbelianSiteTensorData),
                isinstance(B, AbelianSiteTensorData),
            ),
        )
        first_local.setdefault(
            "mpo",
            (
                isinstance(W1_local, AbelianSiteTensorData),
                isinstance(W2_local, AbelianSiteTensorData),
            ),
        )
        return original_optimize(A, B, W1_local, W2_local, *args, **kwargs)

    monkeypatch.setattr(mps_module, "optimize_two_sites", wrapped_optimize)

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
        performance="fast",
    ).run()

    assert dmrg.abelian_matvec_options["native_site_storage"] is True
    assert first_local["mps"] == (True, True)
    assert first_local["mpo"] == (True, True)
    assert dmrg.e_tot == pytest.approx(-1.0, abs=1.0e-8)


def test_abelian_two_site_dmrg_native_storage_converts_complementary_mpos(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(
        [(-cd @ parity, c), (-parity @ c, cd)],
        start=1,
    ):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)
    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )

    first_local = {}
    original_optimize = mps_module.optimize_two_sites

    def wrapped_optimize(A, B, W1_local, W2_local, *args, **kwargs):
        first_local.setdefault(
            "main_mps",
            (
                isinstance(A, AbelianSiteTensorData),
                isinstance(B, AbelianSiteTensorData),
            ),
        )
        first_local.setdefault(
            "main_mpo",
            (
                isinstance(W1_local, AbelianSiteTensorData),
                isinstance(W2_local, AbelianSiteTensorData),
            ),
        )
        envs = kwargs.get("complementary_family_environments") or {}
        if "aux" in envs:
            E_aux, W_aux, F_aux = envs["aux"]
            first_local.setdefault(
                "aux_env",
                (
                    isinstance(E_aux, AbelianEnvironmentTensorData),
                    isinstance(F_aux, AbelianEnvironmentTensorData),
                ),
            )
            first_local.setdefault(
                "aux_mpo",
                tuple(isinstance(site, AbelianSiteTensorData) for site in W_aux),
            )
            first_local.setdefault(
                "aux_blocktensor",
                tuple(isinstance(site, BlockTensor) for site in W_aux),
            )
        return original_optimize(A, B, W1_local, W2_local, *args, **kwargs)

    monkeypatch.setattr(mps_module, "optimize_two_sites", wrapped_optimize)

    energy, _mps_out, _gauge, _converged = two_site_dmrg(
        init,
        mpo,
        4,
        sweeps=2,
        U1=True,
        target_qn=q1,
        not_conv_err=False,
        sym_mgr=SymmetryManager(["charge"]),
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        complementary_operator_mpos={"aux": mpo},
        abelian_matvec_options={
            "native_site_storage": True,
            "packed_local_davidson": True,
            "packed_local_davidson_restart_dim": 8,
        },
    )

    assert first_local["main_mps"] == (True, True)
    assert first_local["main_mpo"] == (True, True)
    assert first_local["aux_env"] == (True, True)
    assert first_local["aux_mpo"] == (True, True)
    assert first_local["aux_blocktensor"] == (False, False)
    assert float(np.real(energy)) == pytest.approx(-1.0, abs=1.0e-8)


def test_abelian_generator_table_contextual_builders_stay_native(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    q2 = AbelianSector(("charge",), (2,))
    phys = [q0, q1, q1, q2]
    site_qn_maps = [{state: qn for state, qn in enumerate(phys)} for _ in range(2)]

    ident = np.eye(4)
    mpo = dense_to_symmetric_mpo(
        [ident.reshape(1, 1, 4, 4), ident.reshape(1, 1, 4, 4)],
        site_qn_maps,
    )
    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0, 0.0, 0.0]).reshape(1, 4, 1),
            np.array([1.0, 0.0, 0.0, 0.0]).reshape(1, 4, 1),
        ],
        phys_qns=phys,
    )

    captured = {}

    class FakeContextualBuilder:
        def __init__(self, **kwargs):
            self.left_builder = kwargs["left_builder"]
            self.right_builder = kwargs["right_builder"]

        def build_entries(self, family_name, route_plan, options, boundary_batch=None):
            left = self.left_builder((), "I", family_name=family_name)
            right = self.right_builder((), "I", family_name=family_name)
            captured["left"] = left
            captured["right"] = right
            return SimpleNamespace(entries=(), seconds=0.0)

    def fake_optimize(A, B, W1, W2, E, F, m, direction, *args, **kwargs):
        captured["main"] = (
            isinstance(A, AbelianSiteTensorData),
            isinstance(B, AbelianSiteTensorData),
            isinstance(W1, AbelianSiteTensorData),
            isinstance(W2, AbelianSiteTensorData),
            isinstance(E, AbelianEnvironmentTensorData),
            isinstance(F, AbelianEnvironmentTensorData),
        )
        return 0.0, A, B, 0.0, 1

    monkeypatch.setattr(
        mps_module,
        "AbelianContextualDirectFamilyBuilder",
        FakeContextualBuilder,
    )
    monkeypatch.setattr(mps_module, "optimize_two_sites", fake_optimize)

    two_site_dmrg(
        init,
        mpo,
        4,
        sweeps=1,
        U1=True,
        target_qn=q1,
        not_conv_err=False,
        sym_mgr=SymmetryManager(["charge"]),
        site_qn_maps=site_qn_maps,
        complementary_operator_generator_entries={"R": {(0, 1): 0.5}},
        abelian_matvec_options={
            "native_site_storage": True,
            "generator_table_packed_boundary_tensors": False,
            "generator_table_precompute_contextual_boundaries": False,
        },
    )

    assert captured["main"] == (True, True, True, True, True, True)
    left_E, left_W = captured["left"]
    right_W, right_F = captured["right"]
    assert isinstance(left_E, AbelianEnvironmentTensorData)
    assert isinstance(right_F, AbelianEnvironmentTensorData)
    assert isinstance(left_W, AbelianSiteTensorData)
    assert isinstance(right_W, AbelianSiteTensorData)
    assert not any(
        isinstance(tensor, BlockTensor)
        for tensor in (left_E, left_W, right_W, right_F)
    )


def test_abelian_generator_table_native_build_avoids_blocktensor_constructor(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    q2 = AbelianSector(("charge",), (2,))
    phys = [q0, q1, q1, q2]
    site_qn_maps = [{state: qn for state, qn in enumerate(phys)} for _ in range(2)]

    ident = np.eye(4)
    mpo = dense_to_symmetric_mpo(
        [ident.reshape(1, 1, 4, 4), ident.reshape(1, 1, 4, 4)],
        site_qn_maps,
    )
    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0, 0.0, 0.0]).reshape(1, 4, 1),
            np.array([1.0, 0.0, 0.0, 0.0]).reshape(1, 4, 1),
        ],
        phys_qns=phys,
    )

    captured = {}

    def fake_optimize(A, B, W1, W2, E, F, m, direction, *args, **kwargs):
        direct_envs = kwargs.get("complementary_direct_family_environments") or {}
        captured["direct_envs"] = direct_envs
        entries = tuple(direct_envs.get("R") or ())
        captured["entry_types"] = tuple(type(entry) for entry in entries)
        for entry in entries:
            if isinstance(entry, AbelianPackedLocalGeneratorEntry):
                fields = (entry.E, entry.W_left, entry.W_right, entry.F)
                assert all(
                    getattr(field, "_pyqed_packed_boundary_tensor", False)
                    or isinstance(
                        field,
                        (AbelianEnvironmentTensorData, AbelianSiteTensorData),
                    )
                    for field in fields
                )
                assert not any(isinstance(field, BlockTensor) for field in fields)
                continue
            E_term, W_pair, F_term = entry
            assert isinstance(E_term, AbelianEnvironmentTensorData)
            assert isinstance(F_term, AbelianEnvironmentTensorData)
            assert all(isinstance(W, AbelianSiteTensorData) for W in W_pair)
        return 0.0, A, B, 0.0, 1

    def fail_blocktensor_init(self, *args, **kwargs):
        raise AssertionError("BlockTensor constructor reached in native generator table")

    monkeypatch.setattr(mps_module, "optimize_two_sites", fake_optimize)
    monkeypatch.setattr(mps_module.BlockTensor, "__init__", fail_blocktensor_init)

    two_site_dmrg(
        init,
        mpo,
        4,
        sweeps=1,
        U1=True,
        target_qn=q1,
        not_conv_err=False,
        sym_mgr=SymmetryManager(["charge"]),
        site_qn_maps=site_qn_maps,
        complementary_operator_generator_entries={"R": {(0, 1): 0.5}},
        abelian_matvec_options={
            "native_site_storage": True,
            "generator_table_enable_native_boundary_r": True,
            "generator_table_allow_reference_validation_fallback": True,
            "generator_table_packed_boundary_tensors": False,
        },
    )

    assert "R" in captured["direct_envs"]
    assert captured["entry_types"]


def test_abelian_state_averaged_native_guess_generation_stays_native(monkeypatch):
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    ident = np.eye(2)
    number = np.diag([0.0, 1.0])
    W0 = np.zeros((1, 2, 2, 2))
    W1 = np.zeros((2, 1, 2, 2))
    W0[0, 0] = 0.2 * number
    W0[0, 1] = ident
    W1[0, 0] = ident
    W1[1, 0] = 0.3 * number
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    legacy = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    native = [
        AbelianSiteTensorData(site.data, site.qns, site.dirs)
        for site in legacy
    ]
    captured = {}

    def fake_solve_davidson_block(
        _operator,
        guess_list,
        *,
        n_eig,
        tol,
        max_iter,
        max_subspace,
        preconditioner,
    ):
        captured["guess_types"] = tuple(type(guess) for guess in guess_list)
        captured["guess_is_native"] = tuple(
            isinstance(guess, AbelianSiteTensorData)
            for guess in guess_list
        )
        return np.asarray([0.0, 0.1]), tuple(guess_list[:n_eig])

    monkeypatch.setattr(
        mps_module,
        "solve_davidson_block",
        fake_solve_davidson_block,
    )

    result = optimize_two_sites(
        native[0],
        native[1],
        mpo[0],
        mpo[1],
        initial_E(mpo[0]),
        initial_F(mpo[1], target_qn=q1),
        m=4,
        dir="right",
        U1=True,
        sym_mgr=SymmetryManager(["charge"]),
        nstates=2,
        weights=[0.5, 0.5],
        noise=0.0,
        davidson_tol=1.0e-10,
        davidson_max_iter=4,
        matvec_options={
            "native_site_storage": True,
            "moving_environment": False,
        },
        moving_environment=False,
    )

    energies, left, right, _trunc, _kept, local_states = result
    assert np.allclose(energies, [0.0, 0.1])
    assert captured["guess_is_native"] == (True, True)
    assert all(isinstance(state, AbelianSiteTensorData) for state in local_states)
    assert isinstance(left, AbelianSiteTensorData)
    assert isinstance(right, AbelianSiteTensorData)
    assert not isinstance(left, BlockTensor)
    assert not isinstance(right, BlockTensor)
    assert not getattr(optimize_two_sites, "last_split_legacy_wrapped", True)


def test_abelian_dmrg_native_site_storage_multibond_moving_environment():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1} for _ in range(3)]
    ident = np.eye(2)
    number = np.diag([0.0, 1.0])
    W0 = np.zeros((1, 2, 2, 2))
    W1 = np.zeros((2, 2, 2, 2))
    W2 = np.zeros((2, 1, 2, 2))
    W0[0, 0] = 0.2 * number
    W0[0, 1] = ident
    W1[0, 0] = ident
    W1[1, 0] = 0.3 * number
    W1[1, 1] = ident
    W2[0, 0] = ident
    W2[1, 0] = 0.5 * number
    mpo = dense_to_symmetric_mpo([W0, W1, W2], site_qn_maps)

    legacy_init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    native_init = [
        AbelianSiteTensorData(site.data, site.qns, site.dirs)
        for site in legacy_init
    ]

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=native_init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
        abelian_matvec_options={
            "native_site_storage": True,
            "moving_environment": True,
            "packed_local_davidson": True,
            "packed_local_davidson_restart_dim": 8,
            "packed_local_block_preconditioner": True,
            "packed_local_block_preconditioner_max_block_dim": 8,
            "packed_local_block_preconditioner_max_total_dim": 64,
        },
    ).run()

    assert dmrg.e_tot == pytest.approx(0.2, abs=1.0e-8)
    assert all(
        isinstance(site, AbelianSiteTensorData)
        for site in dmrg.ground_state.factors
    )
    assert not getattr(optimize_two_sites, "last_split_legacy_wrapped", True)
    moving = dmrg.environment_profile["moving_environment"]
    assert moving["owner_half_sweep_calls"] >= 1
    assert moving["owner_half_sweep_bonds"] >= 1
    assert moving["owner_half_sweep_accepts"] == moving["owner_half_sweep_calls"]
    assert moving["owner_half_sweep_failures"] == 0


def test_abelian_native_site_environment_contractions_match_legacy():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W0[0, 3] = np.zeros((2, 2))
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(terms, start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    psi0 = [
        np.array([0.0, 1.0]).reshape(1, 2, 1),
        np.array([1.0, 0.0]).reshape(1, 2, 1),
    ]
    init = dense_to_symmetric(psi0, phys_qns=[q0, q1])
    native_sites = [
        AbelianSiteTensorData(site.data, site.qns, site.dirs)
        for site in init
    ]

    left_ref = contract_from_left(
        mpo[0],
        init[0],
        initial_E(mpo[0]),
        init[0],
    )
    left_native = contract_from_left(
        mpo[0],
        native_sites[0],
        initial_E(mpo[0]),
        native_sites[0],
    )
    right_ref = contract_from_right(
        mpo[1],
        init[1],
        initial_F(mpo[1], target_qn=q1),
        init[1],
    )
    right_native = contract_from_right(
        mpo[1],
        native_sites[1],
        initial_F(mpo[1], target_qn=q1),
        native_sites[1],
    )

    assert isinstance(left_native, AbelianEnvironmentTensorData)
    assert isinstance(right_native, AbelianEnvironmentTensorData)
    assert left_native.qns == tuple(tuple(axis) for axis in left_ref.qns)
    assert right_native.qns == tuple(tuple(axis) for axis in right_ref.qns)
    assert left_native.dirs == tuple(left_ref.dirs)
    assert right_native.dirs == tuple(right_ref.dirs)
    assert set(left_native.data) == set(left_ref.data)
    assert set(right_native.data) == set(right_ref.data)
    for key, block in left_ref.data.items():
        np.testing.assert_allclose(left_native.data[key], block, atol=1.0e-12)
    for key, block in right_ref.data.items():
        np.testing.assert_allclose(right_native.data[key], block, atol=1.0e-12)


def test_moving_environment_cpp_update_returns_native_environment_data():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    if (
        getattr(cpp_davidson, "abelian_left_environment_advance_data", None) is None
        or getattr(cpp_davidson, "abelian_right_environment_advance_data", None) is None
    ):
        pytest.skip("native environment advance payloads are unavailable")
    if getattr(cpp_davidson, "AbelianEnvironmentAdvancePlan", None) is None:
        pytest.skip("native environment advance plans are unavailable")
    if not all(
        hasattr(cpp_davidson.MovingEnvironment(), name)
        for name in (
            "set_environment_stack",
            "environment_stack_push",
            "environment_stack_pop",
            "environment_stack_depth",
        )
    ):
        pytest.skip("native environment stack owner is unavailable")

    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W0[0, 3] = np.zeros((2, 2))
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(terms, start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    psi0 = [
        np.array([0.0, 1.0]).reshape(1, 2, 1),
        np.array([1.0, 0.0]).reshape(1, 2, 1),
    ]
    init = dense_to_symmetric(psi0, phys_qns=[q0, q1])
    native_sites = [
        AbelianSiteTensorData(site.data, site.qns, site.dirs)
        for site in init
    ]

    moving = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_environment_update": True,
            "moving_environment_cpp_state_owner": True,
        }
    )
    left_ref = contract_from_left(mpo[0], init[0], initial_E(mpo[0]), init[0])
    right_ref = contract_from_right(
        mpo[1],
        init[1],
        initial_F(mpo[1], target_qn=q1),
        init[1],
    )

    left_native = moving.compiled_backend.update_left_environment(
        mpo[0],
        native_sites[0],
        initial_E(mpo[0]),
        native_sites[0],
    )
    right_native = moving.compiled_backend.update_right_environment(
        mpo[1],
        native_sites[1],
        initial_F(mpo[1], target_qn=q1),
        native_sites[1],
    )
    left_cached = moving.compiled_backend.update_left_environment(
        mpo[0],
        native_sites[0],
        initial_E(mpo[0]),
        native_sites[0],
    )
    right_cached = moving.compiled_backend.update_right_environment(
        mpo[1],
        native_sites[1],
        initial_F(mpo[1], target_qn=q1),
        native_sites[1],
    )

    assert isinstance(left_native, AbelianEnvironmentTensorData)
    assert isinstance(right_native, AbelianEnvironmentTensorData)
    assert isinstance(left_cached, AbelianEnvironmentTensorData)
    assert isinstance(right_cached, AbelianEnvironmentTensorData)
    assert moving.moving_profile_stats["cpp_environment_update_left_calls"] == 2
    assert moving.moving_profile_stats["cpp_environment_update_right_calls"] == 2
    assert moving.moving_profile_stats["cpp_environment_update_failures"] == 0
    assert moving.moving_profile_stats["cpp_environment_plan_builds"] == 2
    assert moving.moving_profile_stats["cpp_environment_plan_cache_hits"] == 2
    assert moving.moving_profile_stats["cpp_environment_plan_advance_calls"] == 4
    assert moving.moving_profile_stats["cpp_environment_plan_failures"] == 0
    assert moving.moving_profile_stats["cpp_environment_plan_owner_failures"] == 0
    assert moving.moving_profile_stats["cpp_environment_plan_owner_records"] == 2
    assert moving.moving_profile_stats["cpp_environment_update_backend_actual"] == (
        "cpp_environment_plan"
    )
    assert moving.moving_profile_stats["cpp_environment_plan_backend_actual"] == (
        "cpp_moving_environment"
    )
    assert set(left_native.data) == set(left_ref.data)
    assert set(right_native.data) == set(right_ref.data)
    for key, block in left_ref.data.items():
        np.testing.assert_allclose(left_native.data[key], block, atol=1.0e-12)
        np.testing.assert_allclose(left_cached.data[key], block, atol=1.0e-12)
    for key, block in right_ref.data.items():
        np.testing.assert_allclose(right_native.data[key], block, atol=1.0e-12)
        np.testing.assert_allclose(right_cached.data[key], block, atol=1.0e-12)


def test_moving_environment_cpp_owner_tracks_environment_stack():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    if getattr(cpp_davidson, "AbelianEnvironmentAdvancePlan", None) is None:
        pytest.skip("native environment advance plans are unavailable")

    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    ident = np.eye(2)
    W0 = np.zeros((1, 1, 2, 2))
    W0[0, 0] = ident
    mpo = dense_to_symmetric_mpo([W0, W0], site_qn_maps)
    init = dense_to_symmetric(
        [
            np.array([1.0, 0.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
        native_site_storage=True,
    )

    left_stack = [initial_E(mpo[0])]
    moving = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_environment_update": True,
            "moving_environment_cpp_state_owner": True,
        }
    ).bind_sweep_stacks(left_environments=left_stack)

    updated = moving.update_left_stack(mpo[0], init[0], init[0])
    owner = moving._cpp_moving_environment
    stack_key = moving._cpp_environment_stack_key("left", "hamiltonian")

    assert isinstance(updated, AbelianEnvironmentTensorData)
    assert int(owner.environment_stack_depth(stack_key)) == len(left_stack)
    assert moving.moving_profile_stats["cpp_environment_stack_backend_actual"] == (
        "cpp_moving_environment"
    )
    assert moving.moving_profile_stats[
        "cpp_moving_environment_environment_stack_pushes"
    ] == 1
    assert moving.moving_profile_stats[
        "cpp_moving_environment_environment_stack_apply_calls"
    ] >= 2
    assert moving.moving_profile_stats[
        "cpp_moving_environment_environment_stack_apply_pushes"
    ] == 1

    popped = moving.pop_left_stack()
    assert popped is updated
    assert int(owner.environment_stack_depth(stack_key)) == len(left_stack)
    assert moving.moving_profile_stats[
        "cpp_moving_environment_environment_stack_pops"
    ] == 1
    assert moving.moving_profile_stats[
        "cpp_moving_environment_environment_stack_apply_pops"
    ] == 1


def test_moving_environment_cpp_owner_applies_environment_stack_actions():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "environment_stack_apply"):
        pytest.skip("C++ MovingEnvironment stack apply owner is not available")

    key = "stack-apply-test"
    result = owner.environment_stack_apply(key, "sync", ("E0",))
    assert tuple(bool(x) for x in result[:4]) == (True, False, False, False)
    assert int(result[4]) == 1
    assert owner.environment_stack_depth(key) == 1

    result = owner.environment_stack_apply(key, "push", None, "E1")
    assert tuple(bool(x) for x in result[:4]) == (False, True, False, False)
    assert int(result[4]) == 2

    result = owner.environment_stack_apply(key, "pop", ("E0", "E1"))
    assert tuple(bool(x) for x in result[:4]) == (False, False, True, False)
    assert int(result[4]) == 1
    assert result[5] == "E1"

    result = owner.environment_stack_apply(key, "replace", None, "E2")
    assert tuple(bool(x) for x in result[:4]) == (False, False, False, True)
    assert int(result[4]) == 1
    assert owner.environment_stack_pop(key) == "E2"

    stats = owner.stats()
    assert int(stats["environment_stack_apply_calls"]) == 4
    assert int(stats["environment_stack_apply_syncs"]) == 1
    assert int(stats["environment_stack_apply_pushes"]) == 1
    assert int(stats["environment_stack_apply_pops"]) == 1
    assert int(stats["environment_stack_apply_replaces"]) == 1
    assert int(stats["environment_stack_apply_failures"]) == 0


def test_moving_environment_cpp_owner_fuses_sweep_environment_step():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None or not hasattr(owner_cls(), "sweep_environment_step_auto"):
        pytest.skip("C++ MovingEnvironment fused sweep-step owner is not available")

    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]
    ident = np.eye(2)
    mpo = dense_to_symmetric_mpo(
        [
            ident.reshape(1, 1, 2, 2),
            ident.reshape(1, 1, 2, 2),
        ],
        site_qn_maps,
    )
    mps = dense_to_symmetric(
        [
            np.array([1.0, 0.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
        native_site_storage=True,
    )
    left_stack = [initial_E(mpo[0])]
    right_env = initial_F(mpo[1], target_qn=q0)
    right_stack = [right_env, right_env]
    moving = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_state_owner": True,
            "moving_environment_cpp_sweep_environment_step": True,
        }
    ).bind_sweep_stacks(
        left_environments=left_stack,
        right_environments=right_stack,
    )

    expected = contract_from_left(mpo[0], mps[0], left_stack[-1], mps[0])
    result = moving.sweep_environment_step(
        "left",
        [("hamiltonian", left_stack, mpo[0], mps[0], mps[0])],
        [("right", "hamiltonian", right_stack)],
    )

    assert result["updates"] == 1
    assert result["pops"] == 1
    assert len(left_stack) == 2
    assert len(right_stack) == 1
    actual = left_stack[-1]
    assert isinstance(actual, AbelianEnvironmentTensorData)
    assert set(actual.data) == set(expected.data)
    for key, block in expected.data.items():
        np.testing.assert_allclose(actual.data[key], block, atol=1.0e-12)
    moving._sync_cpp_moving_environment_stats()
    stats = moving.moving_profile_stats
    assert stats["cpp_sweep_environment_step_calls"] == 1
    assert stats["cpp_sweep_environment_step_updates"] == 1
    assert stats["cpp_sweep_environment_step_pops"] == 1
    assert stats["cpp_sweep_environment_step_auto_calls"] == 1
    assert stats["cpp_sweep_environment_step_failures"] == 0
    owner = moving._cpp_moving_environment
    assert int(
        owner.environment_stack_depth(
            moving._cpp_environment_stack_key("left", "hamiltonian")
        )
    ) == len(left_stack)
    assert int(
        owner.environment_stack_depth(
            moving._cpp_environment_stack_key("right", "hamiltonian")
        )
    ) == len(right_stack)


def test_abelian_native_local_operator_probes_preserve_native_type():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(
        [(-cd @ parity, c), (-parity @ c, cd)],
        start=1,
    ):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)
    legacy = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    native = [
        AbelianSiteTensorData(site.data, site.qns, site.dirs)
        for site in legacy
    ]
    aa_legacy = tensordot(legacy[0], legacy[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    aa_native = abelian_merge_adjacent_site_tensors(native[0], native[1])
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
    )

    image_native = H._matvec_generic(aa_native)
    image_legacy = H._matvec_generic(aa_legacy)
    assert isinstance(image_native, AbelianSiteTensorData)
    assert image_native.qns == tuple(tuple(axis) for axis in image_legacy.qns)
    assert image_native.dirs == tuple(image_legacy.dirs)
    assert set(image_native.data) == set(image_legacy.data)
    for key, block in image_legacy.data.items():
        np.testing.assert_allclose(image_native.data[key], block, atol=1.0e-12)

    fallback_images = [
        H._matvec_generic_matrix_chain(aa_native),
        H._matvec_generic_cached_chain(aa_native),
        H._matvec_family_components_chain(H.E, H.W, H.F, aa_native),
        H._matvec_compact_matrix_chain(aa_native),
        H._matvec_batched_compact_matrix_chain(aa_native),
    ]
    if H._native_compact_matrix_chain_available():
        fallback_images.append(H._matvec_native_compact_matrix_chain(aa_native))
    for fallback in fallback_images:
        assert isinstance(fallback, AbelianSiteTensorData)
        assert fallback.qns == tuple(tuple(axis) for axis in image_legacy.qns)
        assert fallback.dirs == tuple(image_legacy.dirs)
        assert set(fallback.data) == set(image_legacy.data)
        for key, block in image_legacy.data.items():
            np.testing.assert_allclose(fallback.data[key], block, atol=1.0e-12)

    layout = H._layout(aa_native)
    restored = H._unflatten(H._flatten(aa_native, layout), aa_native, layout)
    assert isinstance(restored, AbelianSiteTensorData)
    closed_layout = H._closed_layout(aa_native, 8)
    assert closed_layout is not None
    assert H._layout_is_closed(aa_native, closed_layout)
    H_dense, dense_layout = H.dense_matrix(
        aa_native,
        max_dim=8,
        allow_layout_expansion=True,
    )
    assert dense_layout == closed_layout
    assert H_dense.shape == (H._size(closed_layout), H._size(closed_layout))


def test_abelian_fused_family_environment_uses_native_boundaries():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    ident = np.eye(2)
    number = np.diag([0.0, 1.0])
    W0 = np.zeros((1, 2, 2, 2))
    W1 = np.zeros((2, 1, 2, 2))
    W0[0, 0] = 0.2 * number
    W0[0, 1] = ident
    W1[0, 0] = ident
    W1[1, 0] = 0.3 * number
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)
    E = initial_E(mpo[0])
    F = initial_F(mpo[1], target_qn=q1)
    native_E = AbelianEnvironmentTensorData(E.data, E.qns, E.dirs)
    native_F = AbelianEnvironmentTensorData(F.data, F.qns, F.dirs)
    H = HamiltonianMultiplyU1(
        E,
        mpo,
        F,
        complementary_family_environments={
            "n": (native_E, mpo, native_F),
        },
    )

    fused_E, fused_W, fused_F, n_families = H._fused_named_family_environment()

    assert n_families == 1
    assert isinstance(fused_E, AbelianEnvironmentTensorData)
    assert isinstance(fused_F, AbelianEnvironmentTensorData)
    assert not isinstance(fused_E, BlockTensor)
    assert not isinstance(fused_F, BlockTensor)
    assert all(isinstance(site, AbelianSiteTensorData) for site in fused_W)
    assert not any(isinstance(site, BlockTensor) for site in fused_W)
    assert fused_E.rank == 3
    assert fused_F.rank == 3
    assert all(site.rank == 4 for site in fused_W)
    assert set(fused_E.data)
    assert set(fused_F.data)
    assert H._fused_named_family_environment()[0] is fused_E


def test_abelian_native_right_canonicalization_matches_legacy():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    psi0 = [
        np.array([0.0, 1.0]).reshape(1, 2, 1),
        np.array([1.0, 0.0]).reshape(1, 2, 1),
    ]
    legacy = dense_to_symmetric(psi0, phys_qns=[q0, q1])
    native = [
        AbelianSiteTensorData(site.data, site.qns, site.dirs)
        for site in legacy
    ]

    legacy_out = _right_canonicalize_symmetric_factors(legacy, max_bond_dim=4)
    native_out = _right_canonicalize_symmetric_factors(native, max_bond_dim=4)

    assert len(native_out) == len(legacy_out)
    for native_site, legacy_site in zip(native_out, legacy_out):
        assert isinstance(native_site, AbelianSiteTensorData)
        assert native_site.qns == tuple(tuple(axis) for axis in legacy_site.qns)
        assert native_site.dirs == tuple(legacy_site.dirs)
        assert set(native_site.data) == set(legacy_site.data)
        for key, block in legacy_site.data.items():
            np.testing.assert_allclose(native_site.data[key], block, atol=1.0e-12)


def test_hubbard_ladder_compiled_policy_uses_full_safe_layout():
    fast = dmrg_matvec_options("fast")
    assert fast["native_site_storage"] is True
    assert fast["batched_compact_matrix_chain_force"] is True

    compiled_fast = dmrg_matvec_options("compiled-fast")
    assert compiled_fast["native_site_storage"] is True
    assert compiled_fast["batched_compact_matrix_chain_compiled_kernel"] is True

    opts = dmrg_matvec_options("packed-compiled-fast")

    assert opts["native_site_storage"] is True
    assert opts["packed_local_davidson"] is True
    assert opts["packed_local_project_current_support"] is False
    assert opts["packed_local_project_current_support_truncate"] is False
    assert opts["packed_local_accept_projected_unconverged"] is False
    assert opts["packed_local_davidson_max_dim"] >= 1048576
    assert opts["packed_local_large_safe_max_dim"] >= 4194304
    assert opts["packed_local_large_safe_restart_dim"] == 10
    assert opts["packed_local_large_safe_require_flat"] is True
    assert opts["packed_local_flat_matvec"] is True
    assert opts["packed_local_flat_preconditioner"] is True
    assert opts["packed_local_disable_generic_fallback"] is True
    assert opts["moving_environment"] is True
    assert opts["moving_environment_flat_preconditioner"] is True
    assert opts["packed_local_family_flat_direct_matvec"] is True
    assert opts["packed_local_family_flat_direct_matvec_backend"] == "renormalized_table"
    assert opts["generator_table_precompute_contextual_boundaries"] is False
    assert opts["batched_compact_matrix_chain_compiled_kernel"] is True
    assert opts["batched_compact_matrix_chain_compiled_parallel_kernel"] is True

    projected = dmrg_matvec_options("packed-projector-fast")
    assert projected["native_site_storage"] is True
    assert projected["packed_local_project_current_support"] is True

    cpp = dmrg_matvec_options("packed-cpp-fast")
    assert dmrg_matvec_options("auto") == cpp
    assert dmrg_matvec_options("default") == cpp
    assert dmrg_matvec_options("safe") == cpp
    assert dmrg_matvec_options("block2-like") == cpp
    assert cpp["native_site_storage"] is True
    assert cpp["moving_environment_cpp_davidson"] is True
    assert cpp["moving_environment_cpp_accept_unconverged"] is False
    assert cpp["moving_environment_cpp_validate_solution"] is True
    assert cpp["moving_environment_cpp_solution_residual_tol_factor"] >= 1.0
    assert cpp["moving_environment_cpp_solution_residual_abs_tol"] <= 1.0e-8
    assert cpp["moving_environment_cpp_validate_matvec"] is False
    assert cpp["generator_table_packed_boundary_tensors"] is True
    assert cpp["generator_table_allow_legacy_blocktensor_boundary_tables"] is False
    assert cpp["generator_table_allow_unpacked_boundary_tensor_fallback"] is False
    assert cpp["generator_table_prebuild_same_side_native_p"] is True
    assert cpp["generator_table_incremental_same_side_pair_prebuild"] is True
    legacy_auto = dmrg_matvec_options("legacy-auto")
    assert "native_site_storage" not in legacy_auto
    assert legacy_auto["batched_compact_matrix_chain_selector_enabled"] is True
    assert cpp["moving_environment_cpp_validate_matvec_random_vectors"] == 0
    assert cpp["moving_environment_cpp_compact_plan"] is True
    assert cpp["moving_environment_cpp_compact_plan_matvec"] is True
    assert cpp["moving_environment_cpp_compact_plan_bond_slots"] is True
    assert cpp["moving_environment_cpp_state_owner"] is True
    assert cpp["moving_environment_cpp_site_split_owner"] is True
    assert cpp["moving_environment_cpp_sweep_cursor"] is True
    assert cpp["moving_environment_compact_block_table"] is True
    assert cpp["moving_environment_compact_block_table_max_dim"] >= 4096
    assert cpp["moving_environment_cpp_grouped_renormalized_table"] is True
    assert cpp["moving_environment_cpp_grouped_factorized_table"] is False
    assert cpp["moving_environment_cpp_grouped_raw_table"] is True
    assert cpp["moving_environment_cpp_raw_payload_builder"] is True
    assert cpp["moving_environment_cpp_raw_payload_stack_kernels"] is True
    assert cpp["moving_environment_cpp_named_raw_payload_builder"] is True
    assert cpp["moving_environment_cpp_named_raw_payload_plan"] is True
    assert cpp["moving_environment_cpp_raw_route_plan"] is True
    assert cpp["moving_environment_cpp_raw_route_plan_rebind_layout"] is True
    assert cpp["generator_table_packed_direct_family_entries"] is True
    assert cpp["generator_table_precompute_contextual_boundaries"] is True
    assert cpp["generator_table_precompute_contextual_boundaries_min_records"] == 0
    assert cpp["generator_table_planned_contextual_without_precompute_table_lookup"] is True
    assert cpp["generator_table_use_disjoint_same_side_native_p"] is False
    assert cpp["generator_table_use_true_packed_identity_entries"] is False
    assert cpp["generator_table_planned_native_p_identity_entries"] is True
    assert cpp["moving_environment_cpp_grouped_bond_slots"] is True
    assert cpp["moving_environment_cpp_environment_update"] is True
    assert cpp["packed_local_family_flat_group_identity_csr"] is True
    assert cpp["packed_local_family_flat_group_local_generator_csr"] is True
    without_cpp = dict(cpp)
    without_cpp.pop("moving_environment_cpp_davidson")
    without_cpp.pop("moving_environment_cpp_accept_unconverged")
    without_cpp.pop("moving_environment_cpp_validate_solution")
    without_cpp.pop("moving_environment_cpp_solution_residual_tol_factor")
    without_cpp.pop("moving_environment_cpp_solution_residual_abs_tol")
    without_cpp.pop("moving_environment_cpp_validate_matvec")
    without_cpp.pop("moving_environment_cpp_validate_matvec_random_vectors")
    without_cpp.pop("moving_environment_cpp_compact_plan")
    without_cpp.pop("moving_environment_cpp_compact_plan_matvec")
    without_cpp.pop("moving_environment_cpp_compact_plan_bond_slots")
    without_cpp.pop("moving_environment_cpp_state_owner")
    without_cpp.pop("moving_environment_cpp_site_split_owner")
    without_cpp.pop("moving_environment_cpp_sweep_cursor")
    without_cpp.pop("moving_environment_compact_block_table")
    without_cpp.pop("moving_environment_compact_block_table_max_dim")
    without_cpp.pop("moving_environment_cpp_grouped_renormalized_table")
    without_cpp.pop("moving_environment_cpp_grouped_factorized_table")
    without_cpp.pop("moving_environment_cpp_grouped_raw_table")
    without_cpp.pop("moving_environment_cpp_raw_payload_builder")
    without_cpp.pop("moving_environment_cpp_raw_payload_stack_kernels")
    without_cpp.pop("moving_environment_cpp_named_raw_payload_builder")
    without_cpp.pop("moving_environment_cpp_named_raw_payload_plan")
    without_cpp.pop("moving_environment_cpp_raw_route_plan")
    without_cpp.pop("moving_environment_cpp_raw_route_plan_rebind_layout")
    without_cpp.pop("generator_table_precompute_contextual_boundaries")
    without_cpp.pop("generator_table_precompute_contextual_boundaries_min_records")
    without_cpp.pop("generator_table_planned_contextual_without_precompute")
    without_cpp.pop("generator_table_planned_contextual_without_precompute_table_lookup")
    without_cpp.pop("generator_table_packed_direct_family_entries")
    without_cpp.pop("generator_table_packed_boundary_tensors")
    without_cpp.pop("generator_table_allow_legacy_blocktensor_boundary_tables")
    without_cpp.pop("generator_table_allow_unpacked_boundary_tensor_fallback")
    without_cpp.pop("generator_table_allow_reference_validation_fallback")
    without_cpp.pop("generator_table_prebuild_same_side_native_p")
    without_cpp.pop("generator_table_incremental_same_side_pair_prebuild")
    without_cpp.pop("generator_table_use_disjoint_same_side_native_p")
    without_cpp.pop("generator_table_use_true_packed_identity_entries")
    without_cpp.pop("generator_table_planned_native_p_identity_entries")
    without_cpp["generator_table_precompute_contextual_boundaries"] = False
    without_cpp.pop("moving_environment_cpp_grouped_bond_slots")
    without_cpp.pop("moving_environment_cpp_environment_update")
    without_cpp.pop("packed_local_family_flat_group_identity_csr")
    without_cpp.pop("packed_local_family_flat_group_local_generator_csr")
    assert without_cpp == opts
    assert dmrg_matvec_options("block2-cpp") == cpp


def test_cpp_direct_family_payload_accepts_packed_boundary_tensors():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")

    one3 = np.ones((1, 1, 1), dtype=complex)
    one4 = np.ones((1, 1, 1, 1), dtype=complex)
    E = AbelianPackedBoundaryTensor([("m0", "lo", "li")], [one3])
    W_left = AbelianPackedBoundaryTensor(
        [
            ("m0", "m1", "po", "pi"),
            ("m0", "m1", "po", "pi"),
        ],
        [one4, one4],
    )
    W_right = AbelianPackedBoundaryTensor([("m1", "m2", "qo", "qi")], [one4])
    F = AbelianPackedBoundaryTensor([("m2", "ro", "ri")], [one3])
    entries = AbelianPackedDirectFamilyEntries()
    entries.append_local_generator(
        2.0,
        E,
        W_left,
        W_right,
        F,
        source="packed_boundary_test",
    )
    layout = (
        (("li", "ri", "pi", "qi"), (1, 1, 1, 1)),
        (("lo", "ro", "po", "qo"), (1, 1, 1, 1)),
    )

    builder = cpp_davidson.build_direct_family_payload_fastkeys(
        {"P": entries},
        {},
        layout,
        True,
    )

    assert builder.size() == 1
    assert len(W_left.keys) == 1
    stats = builder.stats()
    assert stats["direct_components"] == 1
    assert stats["direct_w1_misses"] == 0
    assert stats["direct_w2_pair_misses"] == 0
    assert stats["direct_f_pair_misses"] == 0
    assert stats["direct_layout_pair_misses"] == 0


def test_packed_cpp_fast_full_sweep_matches_compiled_policy_for_tiny_chain():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1} for _ in range(3)]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    dense_mpo = []
    for site in range(3):
        W = np.zeros((4, 4, 2, 2))
        W[0, 0] = ident
        W[3, 3] = ident
        for channel, (left, right) in enumerate(terms, start=1):
            W[0, channel] = left
            W[channel, 3] = right
        if site == 0:
            W = W[0:1]
        if site == 2:
            W = W[:, 3:4]
        dense_mpo.append(W)
    mpo = dense_to_symmetric_mpo(dense_mpo, site_qn_maps)
    init = dense_to_symmetric(
        [
            np.array([1.0, 0.0]).reshape(1, 2, 1),
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    sym_mgr = SymmetryManager(["charge"])

    def run(policy):
        return DMRG(
            mpo,
            D=4,
            init_guess=[tensor.copy() for tensor in init],
            nsweeps=3,
            symmetry=True,
            target_qn=sym_mgr.get_target_qn(1),
            sym_mgr=sym_mgr,
            not_conv_err=False,
            davidson_tol=1.0e-10,
            davidson_max_iter=20,
            noise=0.0,
            sweep_tol=0.0,
            site_qn_maps=site_qn_maps,
            performance=policy,
        ).run()

    compiled = run("packed-compiled-fast")
    cpp = run("packed-cpp-fast")

    assert cpp.e_tot == pytest.approx(compiled.e_tot, abs=1.0e-10)
    moving = cpp.environment_profile["moving_environment"]
    assert (
        moving["compact_plan_matvec_calls"] > 0
        or moving.get("cpp_davidson_calls", 0) > 0
    )


def test_abelian_dmrg_records_environment_profile():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1} for _ in range(3)]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    dense_mpo = []
    for site in range(3):
        W = np.zeros((4, 4, 2, 2))
        W[0, 0] = ident
        W[3, 3] = ident
        for channel, (left, right) in enumerate(terms, start=1):
            W[0, channel] = left
            W[channel, 3] = right
        if site == 0:
            W = W[0:1]
        if site == 2:
            W = W[:, 3:4]
        dense_mpo.append(W)
    mpo = dense_to_symmetric_mpo(dense_mpo, site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([1.0, 0.0]).reshape(1, 2, 1),
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
    ).run()

    profile = dmrg.environment_profile
    assert isinstance(profile, dict)
    assert profile["build_left"]["calls"] == 1
    assert profile["build_right"]["calls"] == 1
    assert profile["update_left"]["calls"] >= 1
    assert profile["update_right"]["calls"] >= 1
    assert all(
        entry["seconds"] >= 0.0
        for entry in profile.values()
        if isinstance(entry, dict) and "seconds" in entry
    )
    assert "moving_environment" in profile
    assert dmrg.sweep_history[-1]["environment_profile"] == profile


def test_packed_local_davidson_safely_expands_two_site_charge_layout():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(terms, start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    psi0 = [
        np.array([0.0, 1.0]).reshape(1, 2, 1),
        np.array([1.0, 0.0]).reshape(1, 2, 1),
    ]
    init = dense_to_symmetric(psi0, phys_qns=[q0, q1])
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)

    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_davidson_restart_dim": 8,
        },
    )
    energy, state = H.solve_packed_davidson(
        AA,
        tol=1.0e-10,
        max_iter=30,
        preconditioner=H.jacobi_preconditioner(AA),
    )

    assert energy == pytest.approx(-1.0, abs=1.0e-10)
    stats = H.profile_stats["packed_local_davidson"]
    assert stats["converged"] is True
    assert stats["safe_layout_blocks"] == 2
    assert stats["safe_layout_dimension"] == 2
    assert set(state.data) == {
        (q0, q1, q0, q1),
        (q0, q1, q1, q0),
    }
    for q_left, q_right, q_phys_left, q_phys_right in state.data:
        assert q_left == q0
        assert q_right == q1
        assert q_left + q_phys_left + q_phys_right == q_right


def test_packed_local_davidson_large_safe_layout_uses_flat_compact_path():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(terms, start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
            "packed_local_davidson": True,
            "packed_local_davidson_max_dim": 1,
            "packed_local_large_safe_max_dim": 2,
            "packed_local_large_safe_restart_dim": 2,
            "packed_local_large_safe_require_flat": True,
            "packed_local_project_current_support": False,
            "packed_local_flat_matvec": True,
            "packed_local_flat_preconditioner": True,
        },
    )

    safe_layout = H._layout_from_map(H._safe_two_site_layout_map(AA))
    assert H._size(safe_layout) == 2
    assert H._flatten(AA, safe_layout).shape == (2,)

    energy, state = H.solve_packed_davidson(
        AA,
        tol=1.0e-10,
        max_iter=8,
        preconditioner=H.jacobi_preconditioner(AA),
    )

    assert energy == pytest.approx(-1.0, abs=1.0e-10)
    assert state.norm() == pytest.approx(1.0, abs=1.0e-10)
    stats = H.profile_stats["packed_local_davidson"]
    assert stats["converged"] is True
    assert stats["dimension"] == 2
    assert stats["active_max_dim"] == 2
    assert stats["active_restart_dim"] == 2
    assert stats["large_safe_layout"] is True
    assert stats["projected_current_support"] is False
    flat_stats = H.profile_stats["packed_flat_batched_compact_matrix_chain"]["last"]
    assert flat_stats["dimension"] == 2
    assert flat_stats["project_output"] is False


def test_packed_flat_compact_matvec_matches_blocktensor_matvec():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    number = np.diag([0.0, 1.0])
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W0[0, 3] = 0.2 * number
    W1[0, 0] = 0.4 * number
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
            "packed_local_flat_matvec": True,
        },
    )
    layout = H._layout_from_map(H._safe_two_site_layout_map(AA))
    rng = np.random.default_rng(123)
    vec = rng.standard_normal(H._size(layout))

    flat = H._flat_batched_compact_matrix_chain(vec, AA, layout)
    reference = H._flatten(H.matvec(H._unflatten(vec, AA, layout)), layout)

    np.testing.assert_allclose(flat, reference, atol=1.0e-12)


def test_moving_environment_compact_block_table_matches_flat_compact_matvec():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    number = np.diag([0.0, 1.0])
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W0[0, 3] = 0.2 * number
    W1[0, 0] = 0.4 * number
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    options = {
        "batched_compact_matrix_chain_selector_enabled": True,
        "batched_compact_matrix_chain_force": True,
        "packed_local_flat_matvec": True,
        "moving_environment_compact_block_table": True,
        "moving_environment_compact_block_table_max_dim": 64,
    }
    env = MovingEnvironment(matvec_options=options)
    H = env.set_bond(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        bond=0,
        matvec_options=options,
    ).local_operator()
    layout = H._layout_from_map(H._safe_two_site_layout_map(AA))
    rng = np.random.default_rng(124)
    vec = rng.standard_normal(H._size(layout)).astype(np.complex128)

    table = env.compact_block_table(H, AA, layout)
    assert table is not None
    flat = table.matvec(vec)
    reference = H._flat_batched_compact_matrix_chain(vec, AA, layout)

    np.testing.assert_allclose(flat, reference, atol=1.0e-12)
    assert env.moving_profile_stats["compact_block_table_builds"] == 1

    if getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        cpp_options = dict(options)
        cpp_options["moving_environment_cpp_compact_plan"] = True
        cpp_options["moving_environment_cpp_compact_plan_bond_slots"] = True
        cpp_options["moving_environment_cpp_state_owner"] = True
        cpp_env = MovingEnvironment(matvec_options=cpp_options)
        cpp_H = cpp_env.set_bond(
            initial_E(mpo[0]),
            mpo,
            initial_F(mpo[1], target_qn=q1),
            bond=0,
            matvec_options=cpp_options,
        ).local_operator()
        direct = cpp_env.compact_renormalized_table(cpp_H, AA, layout)
        assert direct is not None
        assert direct.storage == "compact_renormalized_table"
        assert getattr(direct, "build_backend", None) == "cpp_block_constructor"
        cpp_flat = direct.matvec(vec)
        np.testing.assert_allclose(cpp_flat, reference, atol=1.0e-12)
        proto_full = cpp_H._zero_proto_from_layout(AA, layout, complex)
        reference_diag = cpp_H._flatten(cpp_H.diagonal(proto_full), layout)
        np.testing.assert_allclose(direct.diagonal_flat(), reference_diag, atol=1.0e-12)
        assert cpp_env.moving_profile_stats["compact_plan_builds"] == 1
        assert cpp_env.moving_profile_stats["compact_renormalized_table_builds"] == 1
        assert cpp_env.moving_profile_stats["compact_plan_bond_slot_stores"] == 1
        assert (
            cpp_env.moving_profile_stats[
                "compact_renormalized_table_cpp_block_constructor_builds"
            ]
            == 1
        )
        assert (
            cpp_env.moving_profile_stats[
                "compact_renormalized_table_python_stack_constructor_builds"
            ]
            == 0
        )

        scaled_H = cpp_env.set_bond(
            1.25 * initial_E(mpo[0]),
            mpo,
            initial_F(mpo[1], target_qn=q1),
            bond=0,
            matvec_options=cpp_options,
        ).local_operator()
        direct_refreshed = cpp_env.compact_renormalized_table(scaled_H, AA, layout)
        assert direct_refreshed is direct
        refreshed_flat = direct_refreshed.matvec(vec)
        refreshed_reference = scaled_H._flat_batched_compact_matrix_chain(vec, AA, layout)
        np.testing.assert_allclose(refreshed_flat, refreshed_reference, atol=1.0e-12)
        scaled_proto_full = scaled_H._zero_proto_from_layout(AA, layout, complex)
        scaled_reference_diag = scaled_H._flatten(
            scaled_H.diagonal(scaled_proto_full),
            layout,
        )
        np.testing.assert_allclose(
            direct_refreshed.diagonal_flat(),
            scaled_reference_diag,
            atol=1.0e-12,
        )
        assert not np.allclose(refreshed_flat, cpp_flat)
        assert cpp_env.moving_profile_stats["compact_plan_builds"] == 1
        assert cpp_env.moving_profile_stats["compact_renormalized_table_builds"] == 1
        assert cpp_env.moving_profile_stats["compact_renormalized_table_refreshes"] == 1
        assert cpp_env.moving_profile_stats["compact_renormalized_table_cache_hits"] == 1
        if hasattr(direct_refreshed.cpp_plan, "update_stacks_from_blocks"):
            assert (
                cpp_env.moving_profile_stats[
                    "compact_renormalized_table_cpp_block_refreshes"
                ]
                == 1
            )
        assert (
            cpp_env.moving_profile_stats[
                "compact_renormalized_table_python_stack_refreshes"
            ]
            == 0
        )


def test_moving_environment_owns_sweep_stack_updates_and_invalidations():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]
    ident = np.eye(2)
    mpo = dense_to_symmetric_mpo(
        [
            ident.reshape(1, 1, 2, 2),
            ident.reshape(1, 1, 2, 2),
        ],
        site_qn_maps,
    )
    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    left_stack = [initial_E(mpo[0])]
    right_stack = [initial_F(mpo[1], target_qn=q1)]
    invalidations = []
    env = MovingEnvironment(matvec_options={}).bind_sweep_stacks(
        left_environments=left_stack,
        right_environments=right_stack,
        direct_family_cache_invalidator=lambda: invalidations.append("stale"),
    )

    expected_left = contract_from_left(mpo[0], init[0], left_stack[-1], init[0])
    actual_left = env.update_left_stack(mpo[0], init[0], init[0])
    assert left_stack[-1] is actual_left
    assert set(actual_left.data) == set(expected_left.data)
    for key, block in expected_left.data.items():
        np.testing.assert_allclose(actual_left.data[key], block, atol=1.0e-12)
    assert env.pop_left_stack() is actual_left
    assert len(left_stack) == 1
    direct_left = env.update_left(mpo[0], init[0], left_stack[-1], init[0])
    assert set(direct_left.data) == set(expected_left.data)
    for key, block in expected_left.data.items():
        np.testing.assert_allclose(direct_left.data[key], block, atol=1.0e-12)

    expected_right = contract_from_right(mpo[1], init[1], right_stack[-1], init[1])
    actual_right = env.update_right_stack(mpo[1], init[1], init[1])
    assert right_stack[-1] is actual_right
    assert set(actual_right.data) == set(expected_right.data)
    for key, block in expected_right.data.items():
        np.testing.assert_allclose(actual_right.data[key], block, atol=1.0e-12)
    assert env.pop_right_stack() is actual_right
    assert len(right_stack) == 1
    direct_right = env.update_right(mpo[1], init[1], right_stack[-1], init[1])
    assert set(direct_right.data) == set(expected_right.data)
    for key, block in expected_right.data.items():
        np.testing.assert_allclose(direct_right.data[key], block, atol=1.0e-12)

    env.invalidate_direct_family_caches()
    stats = env.moving_profile_stats
    assert invalidations == ["stale"]
    assert stats["direct_family_cache_invalidations"] == 1
    assert stats["sweep_stack_bindings"] == 1
    assert stats["environment_stack_updates"]["push_left"]["calls"] == 1
    assert stats["environment_stack_updates"]["pop_left"]["calls"] == 1
    assert stats["environment_stack_updates"]["push_right"]["calls"] == 1
    assert stats["environment_stack_updates"]["pop_right"]["calls"] == 1
    assert stats["environment_updates"]["update_left"]["calls"] == 2
    assert stats["environment_updates"]["update_right"]["calls"] == 2
    assert stats["environment_update_backend"] == "python_contract"

    revision = [3]
    cache_a = {"a": 1}
    cache_b = {"b": 2}
    env.bind_sweep_stacks(
        left_environments=left_stack,
        right_environments=right_stack,
        direct_family_revision_ref=revision,
        direct_family_cache_maps=(cache_a, cache_b),
    )
    env.invalidate_direct_family_caches()
    assert revision == [4]
    assert cache_a == {}
    assert cache_b == {}
    assert env.moving_profile_stats["direct_family_cache_revision"] == 4
    assert env.moving_profile_stats["direct_family_cache_maps_cleared"] == 2


def test_moving_environment_cpp_renormalized_table_backend_matches_python_table():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    if getattr(cpp_davidson, "RenormalizedTable", None) is None:
        pytest.skip("C++ RenormalizedTable backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    layout = (((q0, q0, q0, q0), (1, 1, 2, 1)),)
    qns = [[q0], [q0], [q0], [q0]]
    dirs = [-1, 1, 1, 1]
    left = np.asarray(
        [
            [[0.5, 0.1], [0.1, -0.2]],
            [[0.2, -0.3], [0.4, 0.7]],
        ],
        dtype=np.complex128,
    )
    right = np.asarray([[[1.0]], [[0.5]]], dtype=np.complex128)
    group_scales = (np.asarray([1.0, -0.25], dtype=np.complex128),)
    collected = {
        "matvec_groups": (
            {
                "left": left,
                "right": right,
                "dims": (1, 1, 2, 1, 1, 2, 1, 1),
                "in_start": 0,
                "out_start": 0,
                "channels": 2,
                "scales": group_scales[0],
            },
        ),
        "group_left": (left,),
        "group_right": (right,),
        "group_scales": group_scales,
        "group_dims_array": np.asarray(
            [[1, 1, 2, 1, 1, 2, 1, 1]],
            dtype=np.int64,
        ),
        "group_in_starts_array": np.asarray([0], dtype=np.int64),
        "group_out_starts_array": np.asarray([0], dtype=np.int64),
        "left": (),
        "right": (),
        "dims_array": np.zeros((0, 8), dtype=np.int64),
        "in_starts_array": np.zeros(0, dtype=np.int64),
        "out_starts_array": np.zeros(0, dtype=np.int64),
    }
    table = AbelianRenormalizedOperatorActionTable(
        collected,
        dim=2,
        layout=layout,
        qns=qns,
        dirs=dirs,
        max_dense_block_elements=16,
        sparse_density_threshold=0.0,
    )
    if table.block_matrices is None:
        pytest.skip("optional dense block-table builder is not available")
    env = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_matvec": True,
            "moving_environment_cpp_validate_matvec": True,
        }
    )
    vec = np.asarray([0.3, -0.7], dtype=np.complex128)

    cpp_table = env.cpp_renormalized_table(table, validation_vector=vec)
    assert cpp_table is not None
    assert type(cpp_table).__name__ == "RenormalizedTable"
    np.testing.assert_allclose(cpp_table.matvec(vec), table.matvec(vec), atol=1.0e-12)
    np.testing.assert_allclose(cpp_table.diagonal(), table.diagonal_flat(), atol=1.0e-12)
    out = env.compiled_backend.apply_renormalized_operator_table(table, vec)
    np.testing.assert_allclose(out, table.matvec(vec), atol=1.0e-12)
    stats = env.moving_profile_stats
    assert stats["cpp_renormalized_table_builds"] == 1
    assert stats["cpp_renormalized_table_validation_calls"] == 1
    assert stats["cpp_renormalized_table_validation_failures"] == 0
    assert stats["cpp_renormalized_table_matvec_calls"] == 1


def test_moving_environment_cpp_sparse_renormalized_table_backend_matches_python_table():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    if getattr(cpp_davidson, "SparseRenormalizedTable", None) is None:
        pytest.skip("C++ SparseRenormalizedTable backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    layout = (((q0, q0, q0, q0), (1, 1, 2, 1)),)
    qns = [[q0], [q0], [q0], [q0]]
    dirs = [-1, 1, 1, 1]
    left = np.asarray([[[0.5, 0.0], [0.0, -0.2]]], dtype=np.complex128)
    right = np.asarray([[[1.0]]], dtype=np.complex128)
    collected = {
        "matvec_groups": (
            {
                "left": left,
                "right": right,
                "dims": (1, 1, 2, 1, 1, 2, 1, 1),
                "in_start": 0,
                "out_start": 0,
                "channels": 1,
            },
        ),
        "group_left": (left,),
        "group_right": (right,),
        "group_dims_array": np.asarray(
            [[1, 1, 2, 1, 1, 2, 1, 1]],
            dtype=np.int64,
        ),
        "group_in_starts_array": np.asarray([0], dtype=np.int64),
        "group_out_starts_array": np.asarray([0], dtype=np.int64),
        "left": (),
        "right": (),
        "dims_array": np.zeros((0, 8), dtype=np.int64),
        "in_starts_array": np.zeros(0, dtype=np.int64),
        "out_starts_array": np.zeros(0, dtype=np.int64),
    }
    table = AbelianRenormalizedOperatorActionTable(
        collected,
        dim=2,
        layout=layout,
        qns=qns,
        dirs=dirs,
        max_dense_block_elements=16,
        sparse_density_threshold=1.0,
    )
    if table.block_sparse_values is None:
        pytest.skip("optional sparse block-table builder is not available")
    env = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_matvec": True,
            "moving_environment_cpp_validate_matvec": True,
        }
    )
    vec = np.asarray([0.3, -0.7], dtype=np.complex128)

    cpp_table = env.cpp_renormalized_table(table, validation_vector=vec)
    assert cpp_table is not None
    assert type(cpp_table).__name__ == "SparseRenormalizedTable"
    np.testing.assert_allclose(cpp_table.matvec(vec), table.matvec(vec), atol=1.0e-12)
    np.testing.assert_allclose(cpp_table.diagonal(), table.diagonal_flat(), atol=1.0e-12)
    out = env.compiled_backend.apply_renormalized_operator_table(table, vec)
    np.testing.assert_allclose(out, table.matvec(vec), atol=1.0e-12)
    stats = env.moving_profile_stats
    assert stats["cpp_renormalized_table_builds"] == 1
    assert stats["cpp_sparse_renormalized_table_builds"] == 1
    assert stats["cpp_renormalized_table_storage"] == "renormalized_operator_block_sparse_table"
    assert stats["cpp_renormalized_table_validation_calls"] == 1
    assert stats["cpp_renormalized_table_validation_failures"] == 0
    assert stats["cpp_renormalized_table_matvec_calls"] == 1


def test_moving_environment_builds_and_reuses_cpp_grouped_renormalized_table():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    if getattr(cpp_davidson, "GroupedRenormalizedTable", None) is None:
        pytest.skip("C++ GroupedRenormalizedTable backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    layout = (((q0, q0, q0, q0), (1, 1, 2, 1)),)
    qns = [[q0], [q0], [q0], [q0]]
    dirs = [-1, 1, 1, 1]
    left = np.asarray(
        [
            [[0.5, 0.1], [0.1, -0.2]],
            [[0.2, -0.3], [0.4, 0.7]],
        ],
        dtype=np.complex128,
    )
    right = np.asarray([[[1.0]], [[0.5]]], dtype=np.complex128)
    group_scales = (np.asarray([1.0, -0.25], dtype=np.complex128),)
    raw_dims = np.asarray(
        [
            [1, 1, 2, 1, 1, 2, 1, 1],
            [1, 1, 2, 1, 1, 2, 1, 1],
        ],
        dtype=np.int64,
    )

    def make_collected(active_left, *, build_groups=True):
        raw_left = (active_left[:1], active_left[1:2])
        raw_right = (right[:1], right[1:2])
        raw_in_starts = np.asarray([0, 0], dtype=np.int64)
        raw_out_starts = np.asarray([0, 0], dtype=np.int64)
        raw_scales = np.asarray([1.0, -0.25], dtype=np.complex128)
        collected = {
            "left": raw_left,
            "right": raw_right,
            "dims": [tuple(row) for row in raw_dims],
            "in_starts": [0, 0],
            "out_starts": [0, 0],
            "scales": list(raw_scales),
            "dims_array": raw_dims,
            "in_starts_array": raw_in_starts,
            "out_starts_array": raw_out_starts,
            "scales_array": raw_scales,
            "family_names": ("test",),
            "matvec_groups": None,
        }
        if not build_groups:
            return collected
        collected.update({
            "matvec_groups": (
                {
                    "left": active_left,
                    "right": right,
                    "dims": (1, 1, 2, 1, 1, 2, 1, 1),
                    "in_start": 0,
                    "out_start": 0,
                    "channels": 2,
                    "scales": group_scales[0],
                },
            ),
            "group_left": (active_left,),
            "group_right": (right,),
            "group_scales": group_scales,
            "group_dims_array": np.asarray(
                [[1, 1, 2, 1, 1, 2, 1, 1]],
                dtype=np.int64,
            ),
            "group_in_starts_array": np.asarray([0], dtype=np.int64),
            "group_out_starts_array": np.asarray([0], dtype=np.int64),
        })
        return collected

    def make_builder_collected(active_left):
        builder = cpp_davidson.RawPayloadBuilder()
        builder.add(active_left[:1], right[:1], raw_dims[0], 0, 0, 1.0 + 0.0j)
        builder.add(active_left[1:2], right[1:2], raw_dims[1], 0, 0, -0.25 + 0.0j)
        return {
            "raw_builder": builder,
            "left": [],
            "right": [],
            "dims": [],
            "in_starts": [],
            "out_starts": [],
            "scales": [],
            "entry_count": 2,
            "family_names": ("test",),
            "matvec_groups": None,
        }

    collected = make_collected(left)
    collected_refreshed = make_collected(left + 0.25)
    holder = {"left": left, "builder": False}
    reference = AbelianRenormalizedOperatorActionTable(
        collected,
        dim=2,
        layout=layout,
        qns=qns,
        dirs=dirs,
        max_dense_block_elements=16,
        sparse_density_threshold=0.0,
    )
    refreshed_reference = AbelianRenormalizedOperatorActionTable(
        collected_refreshed,
        dim=2,
        layout=layout,
        qns=qns,
        dirs=dirs,
        max_dense_block_elements=16,
        sparse_density_threshold=0.0,
    )
    assert not np.allclose(reference.matvec([0.3, -0.7]), refreshed_reference.matvec([0.3, -0.7]))

    class Proto:
        pass

    Proto.dirs = dirs

    class Operator:
        bond = 0
        complementary_operator_families = object()
        complementary_boundary_payloads = {}
        complementary_family_environments = {}
        complementary_direct_family_environments = {}
        _packed_local_family_flat_matvec = True
        _packed_local_family_flat_direct_matvec = True
        _packed_local_family_flat_direct_matvec_backend = "renormalized_table"
        _packed_local_family_flat_direct_matvec_min_dim = 0
        _renormalized_operator_table_dense_block_max_elements = 16
        _renormalized_operator_table_sparse_density_threshold = 0.0
        profile_stats = {}

        def _flat_named_family_csr_kernels(
            self,
            proto,
            active_layout,
            *,
            build_groups=True,
        ):
            assert tuple(active_layout) == layout
            if holder.get("builder") and not build_groups:
                return make_builder_collected(holder["left"])
            return make_collected(holder["left"], build_groups=build_groups)

        def _size(self, active_layout):
            assert tuple(active_layout) == layout
            return 2

        def _qns_from_layout_with_proto(self, active_layout, proto):
            assert tuple(active_layout) == layout
            return qns

        def _boundary_family_tables(self):
            return ()

    env = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_davidson": True,
            "moving_environment_cpp_grouped_renormalized_table": True,
            "moving_environment_cpp_grouped_factorized_table": False,
            "moving_environment_cpp_grouped_raw_table": False,
            "moving_environment_cpp_state_owner": True,
        }
    )
    vec = np.asarray([0.3, -0.7], dtype=np.complex128)

    table = env.renormalized_operator_table(Operator(), Proto(), layout)
    assert table is not None
    assert table.storage == "cpp_grouped_renormalized_table_dense"
    assert env.cpp_renormalized_table(table) is table.cpp_table
    np.testing.assert_allclose(table.matvec(vec), reference.matvec(vec), atol=1.0e-12)
    np.testing.assert_allclose(
        table.diagonal_flat(),
        reference.diagonal_flat(),
        atol=1.0e-12,
    )
    assert table.cpp_moving_environment is env._cpp_moving_environment
    result = table.davidson(
        table.diagonal_flat(),
        vec,
        1.0e-12,
        12,
        4,
        True,
    )
    assert result["accepted"] is True
    env._sync_cpp_moving_environment_stats()
    assert env.moving_profile_stats["cpp_moving_environment_grouped_table_records"] == 1
    assert env.moving_profile_stats["cpp_moving_environment_grouped_table_davidson_calls"] == 1

    holder["left"] = left + 0.25
    reused = env.renormalized_operator_table(Operator(), Proto(), layout)
    assert reused is table
    assert reused.last_refresh_kind == "dense_in_place"
    np.testing.assert_allclose(
        reused.matvec(vec),
        refreshed_reference.matvec(vec),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        reused.diagonal_flat(),
        refreshed_reference.diagonal_flat(),
        atol=1.0e-12,
    )
    compiled = env.compiled_flat_matvec(Operator(), Proto(), layout)
    compiled_reused = env.compiled_flat_matvec(Operator(), Proto(), layout)
    assert compiled_reused is compiled
    stats = env.moving_profile_stats
    assert stats["renormalized_operator_table_builds"] == 1
    assert stats["renormalized_operator_table_refreshes"] >= 1
    assert stats["renormalized_operator_table_slot_reuses"] >= 1
    assert stats["compiled_flat_matvec_builds"] == 1
    assert stats["compiled_flat_matvec_cache_hits"] >= 1
    assert stats["cpp_grouped_renormalized_table_builds"] == 1
    assert stats["cpp_grouped_renormalized_table_refreshes"] >= 1
    assert stats["cpp_grouped_renormalized_table_slot_reuses"] >= 1
    assert stats["cpp_grouped_renormalized_table_fast_refreshes"] >= 1
    assert stats["cpp_grouped_renormalized_table_rebuild_refreshes"] == 0
    assert stats["cpp_grouped_renormalized_table_last_refresh_kind"] == (
        "dense_in_place"
    )
    assert stats["cpp_grouped_renormalized_table_last_blocks"] == 1
    assert stats["cpp_grouped_renormalized_table_last_storage"] == (
        "cpp_grouped_renormalized_table_dense"
    )

    holder["left"] = left
    raw_env = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_davidson": True,
            "moving_environment_cpp_grouped_renormalized_table": True,
            "moving_environment_cpp_grouped_factorized_table": False,
            "moving_environment_cpp_grouped_raw_table": True,
            "moving_environment_cpp_state_owner": True,
        }
    )
    raw_table = raw_env.renormalized_operator_table(Operator(), Proto(), layout)
    assert raw_table is not None
    assert raw_table.storage == "cpp_grouped_renormalized_table_dense"
    assert raw_table.last_refresh_kind == "raw_build"
    assert raw_table.n_entries == 2
    assert "group_dims_array" not in raw_table.collected
    np.testing.assert_allclose(raw_table.matvec(vec), reference.matvec(vec), atol=1.0e-12)
    np.testing.assert_allclose(
        raw_table.diagonal_flat(),
        reference.diagonal_flat(),
        atol=1.0e-12,
    )

    holder["left"] = left + 0.25
    raw_reused = raw_env.renormalized_operator_table(Operator(), Proto(), layout)
    assert raw_reused is raw_table
    assert raw_reused.last_refresh_kind == "raw_rebuild"
    np.testing.assert_allclose(
        raw_reused.matvec(vec),
        refreshed_reference.matvec(vec),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        raw_reused.diagonal_flat(),
        refreshed_reference.diagonal_flat(),
        atol=1.0e-12,
    )
    raw_stats = raw_env.moving_profile_stats
    assert raw_stats["cpp_grouped_renormalized_table_builds"] == 1
    assert raw_stats["cpp_grouped_renormalized_table_raw_builds"] == 1
    assert raw_stats["cpp_grouped_renormalized_table_refreshes"] >= 1
    assert raw_stats["cpp_grouped_renormalized_table_rebuild_refreshes"] >= 1
    assert raw_stats["cpp_grouped_renormalized_table_last_refresh_kind"] == "raw_rebuild"

    if getattr(cpp_davidson, "RawPayloadBuilder", None) is not None:
        holder["left"] = left
        holder["builder"] = True
        builder_env = MovingEnvironment(
            matvec_options={
                "moving_environment_cpp_davidson": True,
                "moving_environment_cpp_grouped_renormalized_table": True,
                "moving_environment_cpp_grouped_factorized_table": False,
                "moving_environment_cpp_grouped_raw_table": True,
                "moving_environment_cpp_raw_payload_builder": True,
            }
        )
        builder_table = builder_env.renormalized_operator_table(
            Operator(),
            Proto(),
            layout,
        )
        assert builder_table is not None
        assert "raw_builder" in builder_table.collected
        assert builder_table.n_entries == 2
        assert builder_table.last_refresh_kind == "raw_build"
        np.testing.assert_allclose(
            builder_table.matvec(vec),
            reference.matvec(vec),
            atol=1.0e-12,
        )

        holder["left"] = left + 0.25
        builder_reused = builder_env.renormalized_operator_table(
            Operator(),
            Proto(),
            layout,
        )
        assert builder_reused is builder_table
        assert builder_reused.last_refresh_kind == "raw_rebuild"
        np.testing.assert_allclose(
            builder_reused.matvec(vec),
            refreshed_reference.matvec(vec),
            atol=1.0e-12,
        )
        builder_stats = builder_env.moving_profile_stats
        assert builder_stats["cpp_grouped_renormalized_table_raw_builds"] == 1
        assert builder_stats["cpp_grouped_renormalized_table_raw_builder_builds"] == 1

    holder["builder"] = False
    if getattr(cpp_davidson, "GroupedFactorizedTable", None) is not None:
        holder["left"] = left
        factorized_env = MovingEnvironment(
            matvec_options={
                "moving_environment_cpp_davidson": True,
                "moving_environment_cpp_grouped_renormalized_table": True,
                "moving_environment_cpp_grouped_factorized_table": True,
            }
        )
        factorized = factorized_env.renormalized_operator_table(
            Operator(),
            Proto(),
            layout,
        )
        assert factorized is not None
        assert factorized.storage == "cpp_grouped_factorized_table"
        np.testing.assert_allclose(
            factorized.matvec(vec),
            reference.matvec(vec),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            factorized.diagonal_flat(),
            reference.diagonal_flat(),
            atol=1.0e-12,
        )

        holder["left"] = left + 0.25
        factorized_reused = factorized_env.renormalized_operator_table(
            Operator(),
            Proto(),
            layout,
        )
        assert factorized_reused is factorized
        assert factorized_reused.last_refresh_kind == "factorized_refresh"
        np.testing.assert_allclose(
            factorized_reused.matvec(vec),
            refreshed_reference.matvec(vec),
            atol=1.0e-12,
        )

        np.testing.assert_allclose(
            factorized_reused.diagonal_flat(),
            refreshed_reference.diagonal_flat(),
            atol=1.0e-12,
        )
        factorized_compiled = factorized_env.compiled_flat_matvec(
            Operator(),
            Proto(),
            layout,
        )
        factorized_compiled_reused = factorized_env.compiled_flat_matvec(
            Operator(),
            Proto(),
            layout,
        )
        assert factorized_compiled_reused is factorized_compiled
        factorized_stats = factorized_env.moving_profile_stats
        assert factorized_stats["cpp_grouped_renormalized_table_builds"] == 1
        assert factorized_stats["cpp_grouped_renormalized_table_refreshes"] >= 1
        assert factorized_stats["cpp_grouped_renormalized_table_fast_refreshes"] >= 1
        assert factorized_stats["cpp_grouped_renormalized_table_rebuild_refreshes"] == 0
        assert factorized_stats["cpp_grouped_renormalized_table_last_storage"] == (
            "cpp_grouped_factorized_table"
        )


def test_moving_environment_cpp_owner_splits_flat_two_site_update():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "split_flat_two_site_svd_data"):
        pytest.skip("C++ MovingEnvironment split owner is not available")

    q0 = AbelianSector(("charge",), (0,))
    layout = (((q0, q0, q0, q0), (1, 1, 2, 1)),)
    qns = [[q0], [q0], [q0], [q0]]
    dirs = [-1, 1, 1, 1]
    vec = np.asarray([0.6, -0.8], dtype=np.complex128)

    env = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_state_owner": True,
            "moving_environment_cpp_site_split_owner": True,
        }
    )
    split = env.split_flat_two_site_svd_data(
        vec,
        layout,
        qns=qns,
        dirs=dirs,
        direction="right",
        m_max=2,
    )
    ref = abelian_split_flat_two_site_svd_data(
        vec,
        layout,
        qns=qns,
        dirs=dirs,
        direction="right",
        m_max=2,
    )

    assert split.kept_states == ref.kept_states
    assert split.bond_qns == ref.bond_qns
    assert split.truncation_error == pytest.approx(ref.truncation_error, abs=1.0e-14)
    for actual, expected in (
        (split.a_data, ref.a_data),
        (split.b_data, ref.b_data),
        (split.s_data, ref.s_data),
    ):
        assert tuple(actual) == tuple(expected)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1.0e-12)

    env._sync_cpp_moving_environment_stats()
    stats = env.moving_profile_stats
    assert stats["cpp_moving_environment_site_split_backend"] == "cpp_moving_environment"
    assert stats["cpp_moving_environment_site_split_flat_calls"] == 1
    assert stats["cpp_moving_environment_site_split_flat_failures"] == 0
    assert stats["cpp_moving_environment_site_split_flat_dim"] == 2


def test_moving_environment_cpp_owner_materializes_flat_two_site_update():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "split_flat_two_site_update"):
        pytest.skip("C++ MovingEnvironment site update owner is not available")

    q0 = AbelianSector(("charge",), (0,))
    layout = (((q0, q0, q0, q0), (1, 1, 2, 1)),)
    qns = [[q0], [q0], [q0], [q0]]
    dirs = [-1, 1, 1, 1]
    vec = np.asarray([0.6, -0.8], dtype=np.complex128)

    env = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_state_owner": True,
            "moving_environment_cpp_site_update_owner": True,
        }
    )
    update = env.split_flat_two_site_update(
        vec,
        layout,
        qns=qns,
        dirs=dirs,
        direction="right",
        m_max=2,
    )
    ref = abelian_site_tensors_from_split(
        abelian_split_flat_two_site_svd_data(
            vec,
            layout,
            qns=qns,
            dirs=dirs,
            direction="right",
            m_max=2,
        )
    )

    assert update.kept_states == ref.kept_states
    assert update.bond_qns == ref.bond_qns
    assert update.truncation_error == pytest.approx(
        ref.truncation_error, abs=1.0e-14
    )
    for actual_site, expected_site in (
        (update.left, ref.left),
        (update.right, ref.right),
    ):
        assert actual_site.qns == expected_site.qns
        assert actual_site.dirs == expected_site.dirs
        assert tuple(actual_site.data) == tuple(expected_site.data)
        for key in actual_site.data:
            np.testing.assert_allclose(
                actual_site.data[key], expected_site.data[key], atol=1.0e-12
            )
    assert tuple(update.s_data) == tuple(ref.s_data)
    for key in update.s_data:
        np.testing.assert_allclose(update.s_data[key], ref.s_data[key], atol=1.0e-12)

    env._sync_cpp_moving_environment_stats()
    stats = env.moving_profile_stats
    assert stats["cpp_moving_environment_site_update_backend"] == (
        "cpp_moving_environment"
    )
    assert stats["cpp_moving_environment_site_update_flat_calls"] == 1
    assert stats["cpp_moving_environment_site_update_flat_failures"] == 0
    assert stats["cpp_moving_environment_site_update_flat_dim"] == 2


def test_moving_environment_owner_local_optimize_rejects_non_native_sites():
    env = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_state_owner": True,
        }
    )
    result = env.optimize_single_state_two_site_native(
        np.zeros((1, 1, 1), dtype=complex),
        np.zeros((1, 1, 1), dtype=complex),
        None,
        None,
        None,
        None,
        1,
        "right",
    )

    assert result is None
    assert env.moving_profile_stats["owner_local_optimize_calls"] == 1
    assert env.moving_profile_stats["owner_local_optimize_rejections"] == 1
    assert (
        env.moving_profile_stats["owner_local_optimize_rejected_reason"]
        == "non_native_site_tensor"
    )


def test_moving_environment_cpp_owner_provides_sweep_cursor():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "sweep_bonds"):
        pytest.skip("C++ MovingEnvironment sweep cursor is not available")

    env = MovingEnvironment(
        matvec_options={
            "moving_environment_cpp_state_owner": True,
            "moving_environment_cpp_sweep_cursor": True,
        }
    )

    assert env.sweep_bonds("lr", 5) == (0, 1, 2)
    assert env.sweep_bonds("rl", 5) == (3, 2, 1)
    assert env.sweep_bonds("recenter_left", 5, center_i=1) == (3, 2, 1)
    assert env.sweep_bonds("recenter_right", 5, center_i=1) == (0, 1)

    env._sync_cpp_moving_environment_stats()
    stats = env.moving_profile_stats
    assert stats["cpp_moving_environment_sweep_cursor_backend"] == (
        "cpp_moving_environment"
    )
    assert stats["cpp_moving_environment_sweep_cursor_plan_calls"] == 4
    assert stats["cpp_moving_environment_sweep_cursor_lr_calls"] == 1
    assert stats["cpp_moving_environment_sweep_cursor_rl_calls"] == 1
    assert stats["cpp_moving_environment_sweep_cursor_recenter_calls"] == 2
    assert stats["cpp_moving_environment_sweep_cursor_steps"] == 11
    assert stats["cpp_moving_environment_sweep_cursor_failures"] == 0

    events = []

    def make_step(bond):
        should_move = int(bond) > 1

        def record(name, value):
            events.append((name, int(value)))
            return True

        return {
            "prepare": (
                lambda bond=bond: record("prepare", bond)
                if should_move
                else False
            ),
            "optimize": lambda bond=bond: (float(bond), None, None, 0.0, 1),
            "assign": lambda _result: None,
            "invalidate": lambda: None,
            "cache_guess": lambda: None,
            "move_environment": (
                lambda bond=bond: record("move", bond)
                if should_move
                else False
            ),
            "fallback_environment": None,
        }

    summary = env.run_single_state_half_sweep(
        direction="recenter_left",
        step_direction="rl",
        bonds=(3, 2, 1),
        make_step=make_step,
        make_update=lambda bond, result, seconds: {
            "bond": int(bond),
            "energy": float(result[0]),
            "seconds": float(seconds),
        },
    )
    assert summary["last_bond"] == 1
    assert [entry["bond"] for entry in summary["updates"]] == [3, 2, 1]
    assert events == [
        ("prepare", 3),
        ("move", 3),
        ("prepare", 2),
        ("move", 2),
    ]
    assert env.moving_profile_stats["owner_half_sweep_calls"] == 1
    assert env.moving_profile_stats["owner_half_sweep_bonds"] == 3
    assert env.moving_profile_stats["owner_bond_step_calls"] == 3
    assert env.moving_profile_stats["owner_bond_step_environment_moves"] == 2

    payload_env = MovingEnvironment()
    payload_events = []
    payload_env.run_single_state_bond_step(
        sweep_direction="lr",
        bond=0,
        prepare=lambda: payload_events.append("prepare") or True,
        prepare_payload=lambda: payload_events.append("payload") or {"P": ()},
        optimize=lambda: payload_events.append("optimize") or (0.0, None, None, 0.0, 1),
        assign=lambda _result: payload_events.append("assign"),
        invalidate=lambda: payload_events.append("invalidate"),
        cache_guess=lambda: payload_events.append("cache"),
        move_environment=lambda: payload_events.append("move") or True,
    )
    assert payload_events == [
        "prepare",
        "payload",
        "optimize",
        "assign",
        "invalidate",
        "cache",
        "move",
    ]
    assert payload_env.moving_profile_stats[
        "owner_bond_step_payload_prepares"
    ] == 1
    assert payload_env.moving_profile_stats[
        "owner_bond_step_payload_prepare_seconds"
    ] >= 0.0

    cpp_payload_env = MovingEnvironment(
        matvec_options={"moving_environment_cpp_state_owner": True}
    )
    cpp_payload_owner = getattr(cpp_payload_env, "_cpp_moving_environment", None)
    if cpp_payload_owner is not None and hasattr(cpp_payload_owner, "run_owner_bond_step"):
        cpp_payload_events = []
        cpp_payload_env.run_single_state_bond_step(
            sweep_direction="lr",
            bond=0,
            prepare=lambda: cpp_payload_events.append("prepare") or True,
            prepare_payload=lambda: cpp_payload_events.append("payload") or {"P": ()},
            optimize=lambda: cpp_payload_events.append("optimize")
            or (0.0, None, None, 0.0, 1),
            assign=lambda _result: cpp_payload_events.append("assign"),
            invalidate=lambda: cpp_payload_events.append("invalidate"),
            cache_guess=lambda: cpp_payload_events.append("cache"),
            move_environment=lambda: cpp_payload_events.append("move") or True,
        )
        assert cpp_payload_events == payload_events
        assert cpp_payload_env.moving_profile_stats[
            "owner_bond_step_orchestrator_actual"
        ] == "cpp_moving_environment"
        assert cpp_payload_env.moving_profile_stats[
            "cpp_moving_environment_owner_bond_step_runner_calls"
        ] == 1
        assert cpp_payload_env.moving_profile_stats[
            "cpp_moving_environment_owner_bond_step_runner_payload_prepares"
        ] == 1

    cpp_half_env = MovingEnvironment(
        matvec_options={"moving_environment_cpp_state_owner": True}
    )
    cpp_half_owner = getattr(cpp_half_env, "_cpp_moving_environment", None)
    if cpp_half_owner is not None and hasattr(cpp_half_owner, "run_owner_half_sweep"):
        cpp_half_events = []

        def cpp_make_step(bond):
            bond = int(bond)
            cpp_half_events.append(("make", bond))
            return {
                "prepare": lambda: cpp_half_events.append(("prepare", bond)) or True,
                "prepare_payload": (
                    lambda: cpp_half_events.append(("payload", bond)) or {"P": ()}
                ),
                "optimize": (
                    lambda: cpp_half_events.append(("optimize", bond))
                    or (float(bond), None, None, 0.0, 1)
                ),
                "assign": lambda _result: cpp_half_events.append(("assign", bond)),
                "invalidate": lambda: cpp_half_events.append(("invalidate", bond)),
                "cache_guess": lambda: cpp_half_events.append(("cache", bond)),
                "move_environment": (
                    lambda: cpp_half_events.append(("move", bond)) or True
                ),
                "fallback_environment": None,
            }

        def cpp_make_update(bond, result, _seconds):
            cpp_half_events.append(("update", int(bond)))
            return {"bond": int(bond), "energy": float(result[0])}

        summary = cpp_half_env.run_single_state_half_sweep(
            direction="lr",
            step_direction="lr",
            bonds=(0, 1),
            make_step=cpp_make_step,
            make_update=cpp_make_update,
            after_step=lambda bond, _result, _update: cpp_half_events.append(
                ("after", int(bond))
            ),
        )
        assert summary["last_bond"] == 1
        assert [entry["bond"] for entry in summary["updates"]] == [0, 1]
        assert cpp_half_events == [
            ("make", 0),
            ("prepare", 0),
            ("payload", 0),
            ("optimize", 0),
            ("assign", 0),
            ("invalidate", 0),
            ("cache", 0),
            ("move", 0),
            ("update", 0),
            ("after", 0),
            ("make", 1),
            ("prepare", 1),
            ("payload", 1),
            ("optimize", 1),
            ("assign", 1),
            ("invalidate", 1),
            ("cache", 1),
            ("move", 1),
            ("update", 1),
            ("after", 1),
        ]
        assert cpp_half_env.moving_profile_stats[
            "owner_half_sweep_backend_actual"
        ] == "cpp_owner_half_sweep_runner"
        assert cpp_half_env.moving_profile_stats[
            "cpp_moving_environment_owner_half_sweep_runner_calls"
        ] == 1
        assert cpp_half_env.moving_profile_stats[
            "cpp_moving_environment_owner_half_sweep_runner_bonds"
        ] == 2
        assert cpp_half_env.moving_profile_stats["owner_bond_step_calls"] == 2

    direct_env = MovingEnvironment()
    direct_builds = []

    def build_direct():
        direct_builds.append(1)
        return {"D": (object(), object())}

    first = direct_env.direct_family_environment_for_bond(
        1,
        build_direct,
        cache_key=("bond", 1),
    )
    second = direct_env.direct_family_environment_for_bond(
        1,
        build_direct,
        cache_key=("bond", 1),
    )
    assert first is second
    assert len(direct_builds) == 1
    assert direct_env.moving_profile_stats[
        "owner_direct_family_environment_calls"
    ] == 2
    assert direct_env.moving_profile_stats[
        "owner_direct_family_environment_cache_hits"
    ] == 1
    prepared = direct_env.prepare_direct_family_environment_for_bond(
        1,
        build_direct,
        cache_key=("bond", 1),
    )
    consumed = direct_env.direct_family_prepared_environment_for_bond(
        1,
        build_direct,
        cache_key=("bond", 1),
    )
    assert prepared is consumed
    assert len(direct_builds) == 1
    assert direct_env.moving_profile_stats[
        "owner_direct_family_environment_prepared_payloads"
    ] == 1
    assert direct_env.moving_profile_stats[
        "owner_direct_family_environment_prepared_hits"
    ] == 1
    missing = direct_env.direct_family_prepared_environment_for_bond(
        2,
        build_direct,
        cache_key=("bond", 2),
    )
    assert missing is not None
    assert len(direct_builds) == 2
    assert direct_env.moving_profile_stats[
        "owner_direct_family_environment_prepared_misses"
    ] == 1
    direct_env.clear_owner_direct_family_environment_cache()
    assert direct_env.moving_profile_stats[
        "owner_direct_family_environment_cache_size"
    ] == 0
    assert direct_env.moving_profile_stats[
        "owner_direct_family_environment_prepared_cache_size"
    ] == 0

    cpp_direct_env = MovingEnvironment(
        matvec_options={"moving_environment_cpp_state_owner": True}
    )
    cpp_owner = getattr(cpp_direct_env, "_cpp_moving_environment", None)
    if (
        cpp_owner is not None
        and hasattr(cpp_owner, "install_direct_family_payload_builder")
    ):
        cpp_builds = []

        def build_cpp_direct():
            cpp_builds.append(1)
            return {"C": (object(),)}

        cpp_prepared = cpp_direct_env.prepare_direct_family_environment_for_bond(
            0,
            build_cpp_direct,
            cache_key=("cpp-bond", 0),
        )
        cpp_consumed = cpp_direct_env.direct_family_prepared_environment_for_bond(
            0,
            build_cpp_direct,
            cache_key=("cpp-bond", 0),
        )
        assert cpp_prepared is cpp_consumed
        assert len(cpp_builds) == 1
        assert cpp_direct_env.moving_profile_stats[
            "owner_direct_family_environment_payload_owner"
        ] == "cpp_moving_environment"
        assert cpp_direct_env.moving_profile_stats[
            "cpp_moving_environment_direct_family_payload_builder_builds"
        ] == 1
        assert cpp_direct_env.moving_profile_stats[
            "cpp_moving_environment_direct_family_payload_hits"
        ] == 1


def test_cpp_moving_environment_stores_direct_family_payload_handles():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "install_direct_family_payload"):
        pytest.skip("C++ direct-family payload owner is not available")

    payload = {"family": (object(),)}
    owner.install_direct_family_payload("bond-1", payload)
    assert owner.direct_family_payload("bond-1") is payload
    assert owner.direct_family_payload("missing") is None
    stats = owner.stats()
    assert int(stats["direct_family_payload_records"]) == 1
    assert int(stats["direct_family_payload_installs"]) == 1
    assert int(stats["direct_family_payload_hits"]) == 1
    assert int(stats["direct_family_payload_misses"]) == 1
    owner.clear_direct_family_payloads()
    stats = owner.stats()
    assert int(stats["direct_family_payload_records"]) == 0
    assert int(stats["direct_family_payload_clears"]) == 1
    assert int(stats["direct_family_payload_cleared_entries"]) == 1
    if not hasattr(owner, "install_direct_family_payload_builder"):
        pytest.skip("C++ direct-family payload builder owner is not available")

    builder_calls = []

    def build_payload():
        builder_calls.append(1)
        return {"built": object()}

    owner.install_direct_family_payload_builder("builder-1", build_payload)
    built = owner.prepare_direct_family_payload_from_builder(
        "payload-1",
        "builder-1",
    )
    cached = owner.prepare_direct_family_payload_from_builder(
        "payload-1",
        "builder-1",
    )
    assert built is cached
    assert len(builder_calls) == 1
    stats = owner.stats()
    assert int(stats["direct_family_payload_builder_records"]) == 1
    assert int(stats["direct_family_payload_builder_installs"]) == 1
    assert int(stats["direct_family_payload_builder_prepare_calls"]) == 2
    assert int(stats["direct_family_payload_builder_builds"]) == 1
    assert int(stats["direct_family_payload_builder_cache_hits"]) == 1
    assert int(stats["direct_family_payload_builder_failures"]) == 0
    owner.clear_direct_family_payload_builders()
    stats = owner.stats()
    assert int(stats["direct_family_payload_builder_records"]) == 0
    assert int(stats["direct_family_payload_builder_clears"]) == 1
    assert int(stats["direct_family_payload_builder_cleared_entries"]) == 1
    if not hasattr(owner, "run_owner_bond_step"):
        pytest.skip("C++ owner bond-step runner is not available")

    events = []
    result = owner.run_owner_bond_step(
        lambda: events.append("prepare") or True,
        lambda: events.append("payload") or {"P": ()},
        lambda: events.append("optimize") or ("energy",),
        lambda _result: events.append("assign"),
        lambda: events.append("invalidate"),
        lambda: events.append("cache"),
        lambda: events.append("move") or True,
        None,
    )
    assert events == [
        "prepare",
        "payload",
        "optimize",
        "assign",
        "invalidate",
        "cache",
        "move",
    ]
    assert result["result"] == ("energy",)
    assert result["prepared"] is True
    assert result["payload_prepared"] is True
    assert result["moved"] is True
    stats = owner.stats()
    assert int(stats["owner_bond_step_runner_calls"]) == 1
    assert int(stats["owner_bond_step_runner_accepted"]) == 1
    assert int(stats["owner_bond_step_runner_payload_prepares"]) == 1
    assert int(stats["owner_bond_step_runner_environment_moves"]) == 1
    assert int(stats["owner_bond_step_runner_failures"]) == 0
    if not hasattr(owner, "run_owner_half_sweep"):
        pytest.skip("C++ owner half-sweep runner is not available")

    half_events = []

    def make_step(bond):
        bond = int(bond)
        half_events.append(("make", bond))
        return {
            "prepare": lambda: half_events.append(("prepare", bond)) or True,
            "prepare_payload": lambda: half_events.append(("payload", bond)) or None,
            "optimize": lambda: half_events.append(("optimize", bond))
            or (float(bond),),
            "assign": lambda _result: half_events.append(("assign", bond)),
            "invalidate": lambda: half_events.append(("invalidate", bond)),
            "cache_guess": lambda: half_events.append(("cache", bond)),
            "move_environment": lambda: half_events.append(("move", bond)) or True,
            "fallback_environment": None,
        }

    def make_update(bond, result, _seconds):
        half_events.append(("update", int(bond)))
        return {"bond": int(bond), "energy": float(result[0])}

    before_bond_calls = int(stats["owner_bond_step_runner_calls"])
    half = owner.run_owner_half_sweep(
        "lr",
        (0, 1),
        make_step,
        make_update,
        lambda bond, _result, _update: half_events.append(("after", int(bond))),
        "lr",
    )
    assert half["last_bond"] == 1
    assert [entry["bond"] for entry in half["updates"]] == [0, 1]
    assert half_events == [
        ("make", 0),
        ("prepare", 0),
        ("payload", 0),
        ("optimize", 0),
        ("assign", 0),
        ("invalidate", 0),
        ("cache", 0),
        ("move", 0),
        ("update", 0),
        ("after", 0),
        ("make", 1),
        ("prepare", 1),
        ("payload", 1),
        ("optimize", 1),
        ("assign", 1),
        ("invalidate", 1),
        ("cache", 1),
        ("move", 1),
        ("update", 1),
        ("after", 1),
    ]
    stats = owner.stats()
    assert int(stats["owner_half_sweep_runner_calls"]) == 1
    assert int(stats["owner_half_sweep_runner_accepted"]) == 1
    assert int(stats["owner_half_sweep_runner_bonds"]) == 2
    assert int(stats["owner_half_sweep_runner_failures"]) == 0
    assert int(stats["owner_bond_step_runner_calls"]) == before_bond_calls + 2

    keyed_owner = owner_cls()
    keyed_events = []
    keyed_builder_calls = []

    def make_keyed_step(bond):
        bond = int(bond)
        keyed_events.append(("make", bond))
        payload_key = f"keyed-payload-{bond}"
        builder_key = f"keyed-builder-{bond}"
        keyed_owner.install_direct_family_payload_builder(
            builder_key,
            lambda bond=bond: keyed_builder_calls.append(bond)
            or {"keyed": (bond,)},
        )
        return {
            "prepare": lambda: keyed_events.append(("prepare", bond)) or True,
            "prepare_payload": (
                lambda: keyed_events.append(("payload-callback", bond)) or None
            ),
            "direct_family_payload_key": payload_key,
            "direct_family_builder_key": builder_key,
            "optimize": lambda: keyed_events.append(("optimize", bond))
            or (float(bond),),
            "assign": lambda _result: keyed_events.append(("assign", bond)),
            "invalidate": lambda: keyed_events.append(("invalidate", bond)),
            "cache_guess": lambda: keyed_events.append(("cache", bond)),
            "move_environment": lambda: keyed_events.append(("move", bond)) or True,
            "fallback_environment": None,
        }

    keyed_half = keyed_owner.run_owner_half_sweep(
        "lr",
        (0, 1),
        make_keyed_step,
        None,
        None,
        "lr",
    )
    assert keyed_half["direct_family_payload_prepares"] == 2
    assert keyed_builder_calls == [0, 1]
    assert keyed_events == [
        ("make", 0),
        ("prepare", 0),
        ("optimize", 0),
        ("assign", 0),
        ("invalidate", 0),
        ("cache", 0),
        ("move", 0),
        ("make", 1),
        ("prepare", 1),
        ("optimize", 1),
        ("assign", 1),
        ("invalidate", 1),
        ("cache", 1),
        ("move", 1),
    ]
    keyed_stats = keyed_owner.stats()
    assert int(keyed_stats["direct_family_payload_builder_prepare_calls"]) == 2
    assert int(keyed_stats["direct_family_payload_builder_builds"]) == 2
    assert int(keyed_stats["direct_family_payload_builder_entries"]) == 2
    assert int(keyed_stats["direct_family_payload_builder_last_entries"]) == 1
    assert float(keyed_stats["direct_family_payload_builder_build_seconds"]) >= 0.0
    assert int(keyed_stats["owner_bond_step_runner_payload_prepares"]) == 2

    plan_owner = owner_cls()
    plan_events = []
    plan_calls = []

    def make_plan_piece(source):
        entries = AbelianPackedDirectFamilyEntries()
        entries.append_identity(
            1.0,
            ("E", source),
            ("F", source),
            source=source,
        )
        return entries

    def make_plan_step(bond):
        bond = int(bond)
        plan_events.append(("make", bond))
        payload_key = f"plan-payload-{bond}"
        plan_key = f"plan-key-{bond}"

        def first_plan():
            plan_calls.append(("first", bond))
            return AbelianDirectFamilyLiteralPlan(
                ("P",),
                (make_plan_piece(f"first-{bond}"),),
            )

        def second_plan():
            plan_calls.append(("second", bond))
            return AbelianDirectFamilyLiteralPlan(("Q",), (None,))

        plan_owner.install_direct_family_two_phase_dispatch_plan(
            plan_key,
            first_plan,
            second_plan,
            AbelianCompositePackedDirectFamilyEntries,
            AbelianPackedDirectFamilyEntries,
        )
        return {
            "prepare": lambda: plan_events.append(("prepare", bond)) or True,
            "prepare_payload": (
                lambda: plan_events.append(("payload-callback", bond)) or None
            ),
            "direct_family_payload_key": payload_key,
            "direct_family_plan_key": plan_key,
            "optimize": lambda: plan_events.append(("optimize", bond))
            or (float(bond),),
            "assign": lambda _result: plan_events.append(("assign", bond)),
            "invalidate": lambda: plan_events.append(("invalidate", bond)),
            "cache_guess": lambda: plan_events.append(("cache", bond)),
            "move_environment": lambda: plan_events.append(("move", bond)) or True,
            "fallback_environment": None,
        }

    plan_half = plan_owner.run_owner_half_sweep(
        "lr",
        (0, 1),
        make_plan_step,
        None,
        None,
        "lr",
    )
    assert plan_half["direct_family_payload_prepares"] == 2
    assert plan_calls == [
        ("first", 0),
        ("second", 0),
        ("first", 1),
        ("second", 1),
    ]
    assert plan_events == [
        ("make", 0),
        ("prepare", 0),
        ("optimize", 0),
        ("assign", 0),
        ("invalidate", 0),
        ("cache", 0),
        ("move", 0),
        ("make", 1),
        ("prepare", 1),
        ("optimize", 1),
        ("assign", 1),
        ("invalidate", 1),
        ("cache", 1),
        ("move", 1),
    ]
    plan_stats = plan_owner.stats()
    assert int(plan_stats["direct_family_two_phase_dispatch_plan_prepare_calls"]) == 2
    assert int(plan_stats["direct_family_two_phase_dispatch_plan_dispatch_calls"]) == 2
    assert int(plan_stats["direct_family_payload_builder_builds"]) == 0
    assert int(plan_stats["owner_bond_step_runner_payload_prepares"]) == 2

    if not hasattr(owner, "install_owner_bond_step"):
        pytest.skip("C++ owner keyed bond-step records are not available")
    if not hasattr(owner, "run_owner_half_sweep_from_step_keys"):
        pytest.skip("C++ owner keyed half-sweep runner is not available")

    record_owner = owner_cls()
    record_events = []
    for bond in (0, 1):
        record_owner.install_owner_bond_step(
            f"record-step-{bond}",
            lambda bond=bond: record_events.append(("prepare", bond)) or True,
            lambda bond=bond: record_events.append(("payload", bond)) or None,
            lambda bond=bond: record_events.append(("optimize", bond))
            or (float(bond),),
            lambda _result, bond=bond: record_events.append(("assign", bond)),
            lambda bond=bond: record_events.append(("invalidate", bond)),
            lambda bond=bond: record_events.append(("cache", bond)),
            lambda bond=bond: record_events.append(("move", bond)) or True,
            None,
        )

    def record_update(bond, result, _seconds):
        record_events.append(("update", int(bond)))
        return {"bond": int(bond), "energy": float(result[0])}

    record_half = record_owner.run_owner_half_sweep_from_step_keys(
        "local",
        ((0, "record-step-0"), (1, "record-step-1")),
        record_update,
        lambda bond, _result, _update: record_events.append(("after", int(bond))),
        "local",
    )
    assert record_half["backend"] == "cpp_owner_half_sweep_step_records"
    assert record_half["last_bond"] == 1
    assert [entry["bond"] for entry in record_half["updates"]] == [0, 1]
    assert record_events == [
        ("prepare", 0),
        ("payload", 0),
        ("optimize", 0),
        ("assign", 0),
        ("invalidate", 0),
        ("cache", 0),
        ("move", 0),
        ("update", 0),
        ("after", 0),
        ("prepare", 1),
        ("payload", 1),
        ("optimize", 1),
        ("assign", 1),
        ("invalidate", 1),
        ("cache", 1),
        ("move", 1),
        ("update", 1),
        ("after", 1),
    ]
    record_stats = record_owner.stats()
    assert int(record_stats["owner_bond_step_record_records"]) == 2
    assert int(record_stats["owner_bond_step_record_installs"]) == 2
    assert int(record_stats["owner_bond_step_record_hits"]) == 2
    assert int(record_stats["owner_bond_step_record_misses"]) == 0
    assert int(record_stats["owner_half_sweep_runner_bonds"]) == 2


def test_cpp_moving_environment_assembles_direct_family_payload_pieces():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")

    def packed(source):
        entries = AbelianPackedDirectFamilyEntries()
        entries.append_identity(1.0, ("E", source), ("F", source), source=source)
        return entries

    p0 = packed("p0")
    p1 = packed("p1")
    r0 = packed("r0")

    owner = owner_cls()
    payload = owner.assemble_direct_family_payload(
        "payload-key",
        ("P", "P", "R"),
        (p0, p1, r0),
        AbelianCompositePackedDirectFamilyEntries,
        AbelianPackedDirectFamilyEntries,
        True,
    )

    assert set(payload) == {"P", "R"}
    assert payload["P"].identity_count == 2
    assert payload["R"].identity_count == 1
    cached = owner.direct_family_payload("payload-key")
    assert set(cached) == {"P", "R"}
    assert cached["P"].identity_count == 2
    stats = owner.stats()
    assert int(stats["direct_family_payload_assembler_calls"]) == 1
    assert int(stats["direct_family_payload_assembler_builds"]) == 1
    assert int(stats["direct_family_payload_assembler_pieces"]) == 3
    assert int(stats["direct_family_payload_assembler_merges"]) == 1
    assert int(stats["direct_family_payload_records"]) == 1


def test_cpp_moving_environment_builds_direct_family_payload_from_piece_builders():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")

    def packed(source):
        entries = AbelianPackedDirectFamilyEntries()
        entries.append_identity(1.0, ("E", source), ("F", source), source=source)
        return entries

    calls = []

    def make_builder(source):
        def build():
            calls.append(source)
            return packed(source)

        return build

    owner = owner_cls()
    payload = owner.build_direct_family_payload_from_piece_builders(
        "piece-plan-payload",
        ("P",),
        (packed("initial"),),
        ("P", "R"),
        (make_builder("p-built"), make_builder("r-built")),
        AbelianCompositePackedDirectFamilyEntries,
        AbelianPackedDirectFamilyEntries,
        True,
    )

    assert calls == ["p-built", "r-built"]
    assert set(payload) == {"P", "R"}
    assert payload["P"].identity_count == 2
    assert payload["R"].identity_count == 1
    cached = owner.direct_family_payload("piece-plan-payload")
    assert cached["P"].identity_count == 2
    stats = owner.stats()
    assert int(stats["direct_family_piece_builder_plan_calls"]) == 1
    assert int(stats["direct_family_piece_builder_plan_builds"]) == 1
    assert int(stats["direct_family_piece_builder_plan_families"]) == 2
    assert int(stats["direct_family_piece_builder_plan_pieces"]) == 2
    assert int(stats["direct_family_piece_builder_plan_entries"]) == 2
    assert int(stats["direct_family_piece_builder_plan_failures"]) == 0
    assert int(stats["direct_family_payload_assembler_builds"]) == 1


def test_cpp_moving_environment_builds_phased_direct_family_payload():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")

    def packed(source):
        entries = AbelianPackedDirectFamilyEntries()
        entries.append_identity(1.0, ("E", source), ("F", source), source=source)
        return entries

    calls = []

    def first_p():
        calls.append("first-p")
        return packed("first-p")

    def first_r():
        calls.append("first-r")
        return None

    def make_second(source):
        def build():
            calls.append(source)
            return packed(source)

        return build

    def second_factory():
        calls.append("factory")
        assert calls == ["first-r", "first-p", "factory"]
        return ("P", "Q"), (make_second("second-p"), make_second("second-q"))

    owner = owner_cls()
    payload = owner.build_direct_family_payload_from_phased_piece_builders(
        "phased-piece-plan-payload",
        ("R", "P"),
        (first_r, first_p),
        second_factory,
        AbelianCompositePackedDirectFamilyEntries,
        AbelianPackedDirectFamilyEntries,
        True,
    )

    assert calls == ["first-r", "first-p", "factory", "second-p", "second-q"]
    assert set(payload) == {"P", "Q"}
    assert payload["P"].identity_count == 2
    assert payload["Q"].identity_count == 1
    cached = owner.direct_family_payload("phased-piece-plan-payload")
    assert cached["P"].identity_count == 2
    stats = owner.stats()
    assert int(stats["direct_family_piece_builder_plan_calls"]) == 1
    assert int(stats["direct_family_piece_builder_plan_builds"]) == 1
    assert int(stats["direct_family_piece_builder_plan_families"]) == 4
    assert int(stats["direct_family_piece_builder_plan_pieces"]) == 3
    assert int(stats["direct_family_piece_builder_plan_entries"]) == 3
    assert int(stats["direct_family_piece_builder_plan_empty_pieces"]) == 1
    assert int(stats["direct_family_piece_builder_plan_failures"]) == 0


def test_cpp_moving_environment_prepares_phased_direct_family_payload_plan():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "install_direct_family_phased_piece_plan"):
        pytest.skip("C++ phased piece-plan owner handles are not available")

    def packed(source):
        entries = AbelianPackedDirectFamilyEntries()
        entries.append_identity(1.0, ("E", source), ("F", source), source=source)
        return entries

    calls = []

    def first_p():
        calls.append("first-p")
        return packed("first-p")

    def first_r():
        calls.append("first-r")
        return None

    def make_second(source):
        def build():
            calls.append(source)
            return packed(source)

        return build

    def second_factory():
        calls.append("factory")
        assert calls == ["first-r", "first-p", "factory"]
        return ("P", "Q"), (make_second("second-p"), make_second("second-q"))

    owner = owner_cls()
    owner.install_direct_family_phased_piece_plan(
        "phased-plan",
        ("R", "P"),
        (first_r, first_p),
        second_factory,
        AbelianCompositePackedDirectFamilyEntries,
        AbelianPackedDirectFamilyEntries,
    )
    payload = owner.prepare_direct_family_payload_from_phased_piece_plan(
        "phased-payload",
        "phased-plan",
        False,
        True,
    )

    assert calls == ["first-r", "first-p", "factory", "second-p", "second-q"]
    assert set(payload) == {"P", "Q"}
    assert payload["P"].identity_count == 2
    assert payload["Q"].identity_count == 1
    cached = owner.prepare_direct_family_payload_from_phased_piece_plan(
        "phased-payload",
        "phased-plan",
        False,
        True,
    )
    assert cached is payload

    stats = owner.stats()
    assert int(stats["direct_family_phased_piece_plan_records"]) == 1
    assert int(stats["direct_family_phased_piece_plan_installs"]) == 1
    assert int(stats["direct_family_phased_piece_plan_prepare_calls"]) == 2
    assert int(stats["direct_family_phased_piece_plan_cache_hits"]) == 1
    assert int(stats["direct_family_phased_piece_plan_misses"]) == 0
    assert int(stats["direct_family_phased_piece_plan_failures"]) == 0
    assert int(stats["direct_family_piece_builder_plan_builds"]) == 1
    assert int(stats["direct_family_piece_builder_plan_entries"]) == 3
    assert int(stats["direct_family_payload_records"]) == 1


def test_cpp_moving_environment_prepares_phased_direct_family_family_plan():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "install_direct_family_phased_family_plan"):
        pytest.skip("C++ phased family-plan owner handles are not available")

    def packed(source):
        entries = AbelianPackedDirectFamilyEntries()
        entries.append_identity(1.0, ("E", source), ("F", source), source=source)
        return entries

    calls = []

    def first_p():
        calls.append("first-p")
        return packed("first-p")

    def first_r():
        calls.append("first-r")
        return None

    family_entries = {
        "P": packed("second-p"),
        "Q": packed("second-q"),
    }

    def family_plan_factory():
        calls.append("factory")
        assert calls == ["first-r", "first-p", "factory"]

        def build_family(family_name):
            family_name = str(family_name)
            calls.append(f"family-{family_name}")
            return family_entries[family_name]

        return SimpleNamespace(family_names=("P", "Q"), build_piece=build_family)

    owner = owner_cls()
    owner.install_direct_family_phased_family_plan(
        "phased-family-plan",
        ("R", "P"),
        (first_r, first_p),
        family_plan_factory,
        AbelianCompositePackedDirectFamilyEntries,
        AbelianPackedDirectFamilyEntries,
    )
    payload = owner.prepare_direct_family_payload_from_phased_family_plan(
        "phased-family-payload",
        "phased-family-plan",
        False,
        True,
    )

    assert calls == [
        "first-r",
        "first-p",
        "factory",
        "family-P",
        "family-Q",
    ]
    assert set(payload) == {"P", "Q"}
    assert payload["P"].identity_count == 2
    assert payload["Q"].identity_count == 1
    cached = owner.prepare_direct_family_payload_from_phased_family_plan(
        "phased-family-payload",
        "phased-family-plan",
        False,
        True,
    )
    assert cached is payload

    stats = owner.stats()
    assert int(stats["direct_family_phased_family_plan_records"]) == 1
    assert int(stats["direct_family_phased_family_plan_installs"]) == 1
    assert int(stats["direct_family_phased_family_plan_prepare_calls"]) == 2
    assert int(stats["direct_family_phased_family_plan_cache_hits"]) == 1
    assert int(stats["direct_family_phased_family_plan_misses"]) == 0
    assert int(stats["direct_family_phased_family_plan_failures"]) == 0
    assert int(stats["direct_family_phased_family_plan_dispatch_calls"]) == 1
    assert int(stats["direct_family_phased_family_plan_dispatch_families"]) == 2
    assert int(stats["direct_family_phased_family_plan_dispatch_pieces"]) == 2
    assert int(stats["direct_family_phased_family_plan_dispatch_entries"]) == 2
    assert int(stats["direct_family_phased_family_plan_dispatch_empty_pieces"]) == 0
    assert int(stats["direct_family_piece_builder_plan_builds"]) == 1
    assert int(stats["direct_family_piece_builder_plan_entries"]) == 3
    assert int(stats["direct_family_payload_records"]) == 1


def test_cpp_moving_environment_prepares_two_phase_direct_family_dispatch_plan():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "install_direct_family_two_phase_dispatch_plan"):
        pytest.skip("C++ two-phase dispatch owner handles are not available")

    def packed(source):
        entries = AbelianPackedDirectFamilyEntries()
        entries.append_identity(1.0, ("E", source), ("F", source), source=source)
        return entries

    calls = []

    def first_plan_factory():
        calls.append("first-plan")

        def build_piece(family_name):
            family_name = str(family_name)
            calls.append(f"first-{family_name}")
            return None if family_name == "R" else packed("first-p")

        return SimpleNamespace(family_names=("R", "P"), build_piece=build_piece)

    def second_plan_factory():
        calls.append("second-plan")
        assert calls == ["first-plan", "first-R", "first-P", "second-plan"]

        def build_piece(family_name):
            family_name = str(family_name)
            calls.append(f"second-{family_name}")
            return packed(f"second-{family_name.lower()}")

        return SimpleNamespace(family_names=("P", "Q"), build_piece=build_piece)

    owner = owner_cls()
    owner.install_direct_family_two_phase_dispatch_plan(
        "two-phase-plan",
        first_plan_factory,
        second_plan_factory,
        AbelianCompositePackedDirectFamilyEntries,
        AbelianPackedDirectFamilyEntries,
    )
    payload = owner.prepare_direct_family_payload_from_two_phase_dispatch_plan(
        "two-phase-payload",
        "two-phase-plan",
        False,
        True,
    )

    assert calls == [
        "first-plan",
        "first-R",
        "first-P",
        "second-plan",
        "second-P",
        "second-Q",
    ]
    assert set(payload) == {"P", "Q"}
    assert payload["P"].identity_count == 2
    assert payload["Q"].identity_count == 1
    cached = owner.prepare_direct_family_payload_from_two_phase_dispatch_plan(
        "two-phase-payload",
        "two-phase-plan",
        False,
        True,
    )
    assert cached is payload

    stats = owner.stats()
    assert int(stats["direct_family_two_phase_dispatch_plan_records"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_installs"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_prepare_calls"]) == 2
    assert int(stats["direct_family_two_phase_dispatch_plan_cache_hits"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_misses"]) == 0
    assert int(stats["direct_family_two_phase_dispatch_plan_failures"]) == 0
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_calls"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_families"]) == 4
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_pieces"]) == 3
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_entries"]) == 3
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_empty_pieces"]) == 1
    assert int(stats["direct_family_piece_builder_plan_builds"]) == 1
    assert int(stats["direct_family_piece_builder_plan_entries"]) == 3
    assert int(stats["direct_family_payload_records"]) == 1


def test_cpp_moving_environment_prepares_two_phase_literal_direct_family_plan():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "install_direct_family_two_phase_dispatch_plan"):
        pytest.skip("C++ two-phase dispatch owner handles are not available")

    def packed(source):
        entries = AbelianPackedDirectFamilyEntries()
        entries.append_identity(1.0, ("E", source), ("F", source), source=source)
        return entries

    calls = []

    def first_plan_factory():
        calls.append("first-plan")
        return AbelianDirectFamilyLiteralPlan(
            ("R", "P"),
            (None, packed("first-p")),
        )

    def second_plan_factory():
        calls.append("second-plan")
        assert calls == ["first-plan", "second-plan"]
        return AbelianDirectFamilyLiteralPlan(
            ("P", "Q"),
            (packed("second-p"), packed("second-q")),
        )

    owner = owner_cls()
    owner.install_direct_family_two_phase_dispatch_plan(
        "two-phase-literal-plan",
        first_plan_factory,
        second_plan_factory,
        AbelianCompositePackedDirectFamilyEntries,
        AbelianPackedDirectFamilyEntries,
    )
    payload = owner.prepare_direct_family_payload_from_two_phase_dispatch_plan(
        "two-phase-literal-payload",
        "two-phase-literal-plan",
        False,
        True,
    )

    assert calls == ["first-plan", "second-plan"]
    assert set(payload) == {"P", "Q"}
    assert payload["P"].identity_count == 2
    assert payload["Q"].identity_count == 1
    cached = owner.prepare_direct_family_payload_from_two_phase_dispatch_plan(
        "two-phase-literal-payload",
        "two-phase-literal-plan",
        False,
        True,
    )
    assert cached is payload

    stats = owner.stats()
    assert int(stats["direct_family_two_phase_dispatch_plan_records"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_installs"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_prepare_calls"]) == 2
    assert int(stats["direct_family_two_phase_dispatch_plan_cache_hits"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_misses"]) == 0
    assert int(stats["direct_family_two_phase_dispatch_plan_failures"]) == 0
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_calls"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_families"]) == 4
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_pieces"]) == 3
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_entries"]) == 3
    assert int(stats["direct_family_two_phase_dispatch_plan_dispatch_empty_pieces"]) == 1
    assert int(stats["direct_family_two_phase_dispatch_plan_literal_families"]) == 4
    assert int(stats["direct_family_two_phase_dispatch_plan_literal_pieces"]) == 3
    assert int(stats["direct_family_two_phase_dispatch_plan_literal_entries"]) == 3
    assert int(stats["direct_family_two_phase_dispatch_plan_literal_empty_pieces"]) == 1
    assert int(stats["direct_family_piece_builder_plan_builds"]) == 1
    assert int(stats["direct_family_piece_builder_plan_entries"]) == 3
    assert int(stats["direct_family_payload_records"]) == 1


def test_cpp_moving_environment_builds_planned_direct_family_entries_from_route():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "build_planned_direct_family_entries_from_route"):
        pytest.skip("C++ planned direct-family entry builder is not available")

    def tensor(label):
        return AbelianPackedBoundaryTensor(
            ((label,),),
            (np.ones((1, 1, 1), dtype=complex),),
            dirs=(1, -1, 1),
            source=f"packed-{label}",
        )

    records = (
        ((0,), "A", "B", (2,), 1.5),
        ((1,), "A", "C", (3,), -0.25),
        ((0,), "A", "B", (2,), 0.5),
    )
    route_plan = AbelianDirectRoutePlan.from_records("Q", records, bond=0)
    left_table = AbelianPackedContextualBoundaryTable(
        side="left",
        bond=0,
        revision=1,
    )
    right_table = AbelianPackedContextualBoundaryTable(
        side="right",
        bond=1,
        revision=1,
    )
    left_payloads = tuple((tensor(f"L{idx}E"), tensor(f"L{idx}W")) for idx in range(route_plan.left_count))
    right_payloads = tuple((tensor(f"R{idx}W"), tensor(f"R{idx}F")) for idx in range(route_plan.right_count))
    assert left_table.put_many(route_plan.left_keys, left_payloads, family_name="Q", normalized=True) == route_plan.left_count
    assert right_table.put_many(route_plan.right_keys, right_payloads, family_name="Q", normalized=True) == route_plan.right_count
    left_ids, *_ = left_table.resolve_current_ids_many(route_plan.left_keys, normalized=True)
    right_ids, *_ = right_table.resolve_current_ids_many(route_plan.right_keys, normalized=True)
    boundary_batch = AbelianContextualBoundaryBatch(
        {},
        {},
        (),
        (),
        tuple(left_ids),
        tuple(right_ids),
        {"packed": route_plan.left_count},
        {"packed": route_plan.right_count},
    )

    reference = AbelianPlannedPackedDirectFamilyEntries.from_route_plan(
        route_plan,
        boundary_batch,
        left_table=left_table,
        right_table=right_table,
        source="contextual_Q_local_generator_csr",
    )
    owner = owner_cls()
    planned = owner.build_planned_direct_family_entries_from_route(
        AbelianPlannedPackedDirectFamilyEntries,
        route_plan,
        boundary_batch,
        left_table,
        right_table,
        "contextual_Q_local_generator_csr",
    )

    assert planned._pyqed_planned_direct_family_table_ids is True
    assert len(planned) == len(reference) == route_plan.pair_count
    np.testing.assert_allclose(planned.local_coeffs, reference.local_coeffs)
    np.testing.assert_array_equal(planned.local_left_ids, reference.local_left_ids)
    np.testing.assert_array_equal(planned.local_right_ids, reference.local_right_ids)
    np.testing.assert_array_equal(planned.left_table_ids, reference.left_table_ids)
    np.testing.assert_array_equal(planned.right_table_ids, reference.right_table_ids)
    assert planned.left_table is left_table
    assert planned.right_table is right_table
    assert planned.source == reference.source

    stats = owner.stats()
    assert int(stats["planned_direct_family_entry_build_calls"]) == 1
    assert int(stats["planned_direct_family_entry_build_successes"]) == 1
    assert int(stats["planned_direct_family_entry_build_failures"]) == 0
    assert int(stats["planned_direct_family_entry_build_entries"]) == route_plan.pair_count
    assert int(stats["planned_direct_family_entry_build_table_backed"]) == 1


def test_cpp_moving_environment_builds_direct_route_plan_from_records():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "build_direct_route_plan_from_records"):
        pytest.skip("C++ direct route-plan builder is not available")

    records = (
        ((0,), "A", "B", (2,), 1.5),
        ((1,), "A", "C", (3,), -0.25),
        ((0,), "A", "B", (2,), 0.5),
        ((1,), "A", "C", (3,), 0.25),
        ((4,), "D", "E", (), -2.0j),
    )
    reference = AbelianDirectRoutePlan.from_records("Q", records, bond=1)
    owner = owner_cls()
    plan = owner.build_direct_route_plan_from_records(
        AbelianDirectRoutePlan,
        "Q",
        records,
        1,
        False,
    )

    assert plan.family_name == reference.family_name
    assert plan.bond == reference.bond
    assert plan.left_keys == reference.left_keys
    assert plan.right_keys == reference.right_keys
    assert plan.records == reference.records
    assert plan.signature == reference.signature
    np.testing.assert_array_equal(plan.left_ids, reference.left_ids)
    np.testing.assert_array_equal(plan.right_ids, reference.right_ids)
    np.testing.assert_allclose(plan.coeffs, reference.coeffs)
    np.testing.assert_array_equal(plan.pair_left_ids, reference.pair_left_ids)
    np.testing.assert_array_equal(plan.pair_right_ids, reference.pair_right_ids)
    np.testing.assert_allclose(plan.pair_coeffs, reference.pair_coeffs)

    stats = owner.stats()
    assert int(stats["direct_route_plan_build_calls"]) == 1
    assert int(stats["direct_route_plan_build_successes"]) == 1
    assert int(stats["direct_route_plan_build_failures"]) == 0
    assert int(stats["direct_route_plan_build_records"]) == reference.record_count
    assert int(stats["direct_route_plan_build_pairs"]) == reference.pair_count
    assert int(stats["direct_route_plan_build_left_keys"]) == reference.left_count
    assert int(stats["direct_route_plan_build_right_keys"]) == reference.right_count


def test_cpp_moving_environment_builds_contextual_family_records_from_terms():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "build_contextual_family_records_from_terms"):
        pytest.skip("C++ contextual record builder is not available")

    from pyqed.mps.abelian_direct import make_contextual_family_records

    terms = (
        (("I", "C", "D", "I"), 1.5),
        (("A", "B", "C", "D"), -0.25j),
        (("I", "N", "N", "I"), 0.75),
    )
    reference = make_contextual_family_records(terms, 1)
    owner = owner_cls()
    records = owner.build_contextual_family_records_from_terms(terms, 1)

    assert tuple(records) == reference
    stats = owner.stats()
    assert int(stats["contextual_record_build_calls"]) == 1
    assert int(stats["contextual_record_build_successes"]) == 1
    assert int(stats["contextual_record_build_failures"]) == 0
    assert int(stats["contextual_record_build_terms"]) == len(terms)


def test_cpp_moving_environment_prebuilds_contextual_local_piece_entries():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "prebuild_contextual_local_piece_entries"):
        pytest.skip("C++ contextual local-entry prebuilder is not available")

    site_qn_maps = ({0: 0, 1: 1}, {0: -1, 1: 0})
    local_ops = {
        "X": np.asarray([[0.0, 2.0], [3.0, 0.0]], dtype=complex),
        "N": np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    }
    pieces = tuple(sorted(local_ops))
    reference_builder = AbelianSpatialLocalOperatorBuilder(
        site_qn_maps,
        local_ops=local_ops,
    )
    reference_cache = {}
    for site in range(len(site_qn_maps)):
        for piece in pieces:
            reference_cache[(piece, site)] = reference_builder.local_piece_entries(
                piece,
                site,
            )

    owner = owner_cls()
    cache = {}
    result = owner.prebuild_contextual_local_piece_entries(
        cache,
        local_ops,
        site_qn_maps,
        pieces,
        len(site_qn_maps),
        1.0e-14,
    )

    assert bool(result["complete"]) is True
    assert int(result["built"]) == len(pieces) * len(site_qn_maps)
    assert int(result["skipped"]) == 0
    assert int(result["failures"]) == 0
    assert set(cache) == set(reference_cache)
    for key, reference in reference_cache.items():
        entries, phys_qns = cache[key]
        ref_entries, ref_phys_qns = reference
        assert tuple(entries) == tuple(ref_entries)
        assert tuple(phys_qns) == tuple(ref_phys_qns)

    stats = owner.stats()
    assert int(stats["contextual_local_entry_prebuild_calls"]) == 1
    assert int(stats["contextual_local_entry_prebuild_successes"]) == 1
    assert int(stats["contextual_local_entry_prebuild_failures"]) == 0
    assert int(stats["contextual_local_entry_prebuild_built"]) == len(reference_cache)


def test_cpp_moving_environment_selects_same_side_route_identity_rows():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "select_same_side_route_identity_rows"):
        pytest.skip("C++ same-side route row selector is not available")

    route_plan = SimpleNamespace(
        raw_key_tuples=((0, 1, 0, 1), (0, 1, 2, 3), (2, 3, 2, 3)),
        term_counts=np.asarray((2, 3, 4), dtype=np.int64),
        raw_keys=np.asarray(
            ((0, 1, 0, 1), (0, 1, 2, 3), (2, 3, 2, 3)),
            dtype=np.int64,
        ),
        offsets=np.asarray((0, 2, 5, 9), dtype=np.int64),
        boundary_ids=np.asarray((0, 1, 1, 2, 3, 0, 2, 4, 5), dtype=np.int64),
        factors=np.ones(9, dtype=np.complex128),
    )
    p_entries = {
        (0, 1, 0, 1): 1.5,
        (0, 1, 2, 3): 0.0,
        (2, 3, 2, 3): -0.25j,
    }
    consumed = {(0, 1, 0, 1)}

    owner = owner_cls()
    selected = owner.select_same_side_route_identity_rows(
        route_plan,
        p_entries,
        consumed,
    )

    assert np.asarray(selected["rows"], dtype=np.int64).tolist() == [2]
    assert np.allclose(np.asarray(selected["coeffs"]), np.asarray([-0.25j]))
    assert selected["raw_keys"] == ((2, 3, 2, 3),)
    assert int(selected["terms"]) == 4
    assert int(selected["scanned"]) == 3
    assert int(selected["skipped_consumed"]) == 1
    assert int(selected["skipped_zero"]) == 1

    entries = owner.build_same_side_route_identity_entries(
        AbelianSameSidePRouteIdentityEntries,
        "left",
        selected["rows"],
        selected["coeffs"],
        int(selected["terms"]),
        route_plan,
        np.arange(6, dtype=np.int64),
        tuple(("B", idx) for idx in range(6)),
        None,
        ("I",),
        "test_same_side_route_identity",
    )
    assert entries.identity_count == 4
    assert entries.local_generator_count == 0
    assert len(entries) == 4

    stats = owner.stats()
    assert int(stats["same_side_route_identity_select_calls"]) == 1
    assert int(stats["same_side_route_identity_select_failures"]) == 0
    assert int(stats["same_side_route_identity_select_rows"]) == 1
    assert int(stats["same_side_route_identity_select_terms"]) == 4
    assert int(stats["same_side_route_identity_select_scanned"]) == 3
    assert int(stats["same_side_route_identity_entry_build_calls"]) == 1
    assert int(stats["same_side_route_identity_entry_build_failures"]) == 0
    assert int(stats["same_side_route_identity_entry_build_rows"]) == 1
    assert int(stats["same_side_route_identity_entry_build_terms"]) == 4


def test_cpp_moving_environment_prepares_same_side_route_identity_info():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "prepare_same_side_route_identity_info"):
        pytest.skip("C++ same-side route info preparation is not available")

    route_plan = SimpleNamespace(
        raw_key_tuples=((0, 1, 0, 1), (0, 1, 2, 3)),
        raw_keys=np.asarray(((0, 1, 0, 1), (0, 1, 2, 3)), dtype=np.int64),
        offsets=np.asarray((0, 2, 5), dtype=np.int64),
        records=2,
        terms=5,
    )
    boundary_value_table = SimpleNamespace(payloads=("p0", "p1", "p2"))
    table = SimpleNamespace(
        _pyqed_same_side_route_columns=route_plan,
        _pyqed_same_side_route_boundary_results=(),
        _pyqed_same_side_route_boundary_table_ids=np.asarray(
            (0, 1, 2, 1, 0),
            dtype=np.int64,
        ),
        _pyqed_same_side_route_boundary_value_table=boundary_value_table,
        _pyqed_same_side_route_boundary_table_complete=True,
    )

    owner = owner_cls()
    info = owner.prepare_same_side_route_identity_info(table, 10)

    assert bool(info["supported"])
    assert info["route_plan"] is route_plan
    assert info["boundary_value_table"] is boundary_value_table
    assert np.asarray(info["boundary_table_ids"], dtype=np.int64).tolist() == [
        0,
        1,
        2,
        1,
        0,
    ]
    assert info["boundary_payloads"] == ("p0", "p1", "p2")
    assert info["row_map"] == {
        (0, 1, 0, 1): 0,
        (0, 1, 2, 3): 1,
    }
    assert table._pyqed_same_side_route_row_map is info["row_map"]

    info_again = owner.prepare_same_side_route_identity_info(table, 10)
    assert bool(info_again["supported"])
    assert info_again["row_map"] is info["row_map"]

    too_large = owner.prepare_same_side_route_identity_info(table, 4)
    assert not bool(too_large["supported"])
    assert str(too_large["reason"]) == "too_many_terms"

    stats = owner.stats()
    assert int(stats["same_side_route_identity_info_calls"]) == 3
    assert int(stats["same_side_route_identity_info_successes"]) == 2
    assert int(stats["same_side_route_identity_info_unsupported"]) == 1
    assert int(stats["same_side_route_identity_info_failures"]) == 0
    assert int(stats["same_side_route_identity_info_records"]) == 4
    assert int(stats["same_side_route_identity_info_terms"]) == 10
    assert int(stats["same_side_route_identity_info_rows"]) == 4
    assert int(stats["same_side_route_identity_info_row_map_builds"]) == 1
    assert int(stats["same_side_route_identity_info_row_map_hits"]) == 1
    assert str(stats["same_side_route_identity_info_last_reason"]) == "too_many_terms"


def test_cpp_moving_environment_prepares_same_side_route_boundary_batch():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "prepare_same_side_route_boundary_batch"):
        pytest.skip("C++ same-side route boundary batch is not available")

    boundary_keys = (
        (("I",), "I"),
        (("C", "D"), "I"),
    )
    route_plan = SimpleNamespace(boundary_keys=boundary_keys)
    value_table = SimpleNamespace(
        entries={boundary_keys[0]: "left-boundary"},
        ids={boundary_keys[0]: 3},
        payloads=["empty", "a", "b", "left-boundary"],
        batch_resolves=0,
        hits=0,
        misses=0,
        last_batch_size=0,
        last_batch_hits=0,
        last_batch_misses=0,
        cpp_resolves=0,
    )
    table = SimpleNamespace()

    owner = owner_cls()
    first = owner.prepare_same_side_route_boundary_batch(
        table,
        route_plan,
        value_table,
    )

    assert np.asarray(first["table_ids"], dtype=np.int64).tolist() == [3, -1]
    assert first["missing_keys"] == (boundary_keys[1],)
    assert first["missing_positions"] == (1,)
    assert int(first["hits"]) == 1
    assert int(first["misses"]) == 1
    assert not bool(first["complete"])
    assert table._pyqed_same_side_route_boundary_value_table is value_table
    assert table._pyqed_same_side_route_boundary_payloads is value_table.payloads
    assert np.asarray(
        table._pyqed_same_side_route_boundary_table_ids,
        dtype=np.int64,
    ).tolist() == [3, -1]
    assert not table._pyqed_same_side_route_boundary_table_complete
    assert value_table.batch_resolves == 1
    assert value_table.cpp_resolves == 1
    assert value_table.hits == 1
    assert value_table.misses == 1

    value_table.entries[boundary_keys[1]] = "right-boundary"
    value_table.ids[boundary_keys[1]] = 4
    value_table.payloads.append("right-boundary")
    second = owner.prepare_same_side_route_boundary_batch(
        table,
        route_plan,
        value_table,
    )

    assert np.asarray(second["table_ids"], dtype=np.int64).tolist() == [3, 4]
    assert second["missing_keys"] == ()
    assert second["missing_positions"] == ()
    assert bool(second["complete"])
    assert table._pyqed_same_side_route_boundary_table_complete
    assert value_table.batch_resolves == 2
    assert value_table.cpp_resolves == 2
    assert value_table.hits == 3
    assert value_table.misses == 1

    stats = owner.stats()
    assert int(stats["same_side_route_boundary_batch_calls"]) == 2
    assert int(stats["same_side_route_boundary_batch_successes"]) == 2
    assert int(stats["same_side_route_boundary_batch_failures"]) == 0
    assert int(stats["same_side_route_boundary_batch_keys"]) == 4
    assert int(stats["same_side_route_boundary_batch_hits"]) == 3
    assert int(stats["same_side_route_boundary_batch_misses"]) == 1
    assert int(stats["same_side_route_boundary_batch_complete"]) == 1


def test_cpp_moving_environment_prepares_same_side_route_boundary_parent_rows():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "prepare_same_side_route_boundary_parent_rows"):
        pytest.skip("C++ same-side route boundary parent rows are not available")

    route_plan = SimpleNamespace(
        boundary_parent_ids=np.asarray((0, 1, 0), dtype=np.int64),
        boundary_parent_keys=(
            (("I",), "I"),
            (("I", "C"), "I"),
        ),
        boundary_local_pieces=("C", "D", "C"),
    )
    missing_keys = (
        (("I", "C"), "I"),
        (("I", "C", "D"), "I"),
        (("I", "C"), "I"),
    )
    missing_positions = (0, 1, 2)

    owner = owner_cls()
    planned = owner.prepare_same_side_route_boundary_parent_rows(
        "left",
        route_plan,
        missing_keys,
        missing_positions,
    )

    assert bool(planned["used_route_layout"])
    assert planned["unique_parent_keys"] == (
        (("I",), "I"),
        (("I", "C"), "I"),
    )
    assert planned["parent_rows"] == (
        (missing_keys[0], 0, 0, (("I",), "I"), "C"),
        (missing_keys[1], 1, 1, (("I", "C"), "I"), "D"),
        (missing_keys[2], 2, 0, (("I",), "I"), "C"),
    )

    fallback_plan = SimpleNamespace(
        boundary_parent_ids=np.asarray((), dtype=np.int64),
        boundary_parent_keys=(),
        boundary_local_pieces=(),
    )
    fallback_key = (("A", "B"), "I")
    fallback_parent_key = (("B",), "I")
    fallback = owner.prepare_same_side_route_boundary_parent_rows(
        "right",
        fallback_plan,
        (fallback_key,),
        (0,),
    )

    assert not bool(fallback["used_route_layout"])
    assert fallback["unique_parent_keys"] == (fallback_parent_key,)
    assert fallback["parent_rows"] == (
        (fallback_key, 0, 0, fallback_parent_key, "A"),
    )

    stats = owner.stats()
    assert int(stats["same_side_route_boundary_parent_plan_calls"]) == 2
    assert int(stats["same_side_route_boundary_parent_plan_successes"]) == 2
    assert int(stats["same_side_route_boundary_parent_plan_failures"]) == 0
    assert int(stats["same_side_route_boundary_parent_plan_rows"]) == 4
    assert int(stats["same_side_route_boundary_parent_plan_unique"]) == 3
    assert int(stats["same_side_route_boundary_parent_plan_route_layout"]) == 1
    assert int(stats["same_side_route_boundary_parent_plan_fallback"]) == 1


def test_cpp_moving_environment_prepares_same_side_route_boundary_parent_values():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "prepare_same_side_route_boundary_parent_values"):
        pytest.skip("C++ same-side route boundary parent values are not available")

    parent_keys = (
        (("I",), "I"),
        (("I", "C"), "I"),
    )
    parent_rows = (
        ((("I", "C"), "I"), 0, 0, parent_keys[0], "C"),
        ((("I", "C", "D"), "I"), 1, 1, parent_keys[1], "D"),
        ((("I", "C"), "I"), 2, 0, parent_keys[0], "C"),
    )
    prev_value_table = SimpleNamespace(
        entries={parent_keys[0]: "parent-0"},
        ids={parent_keys[0]: 7},
        batch_resolves=0,
        hits=0,
        misses=0,
        last_batch_size=0,
        last_batch_hits=0,
        last_batch_misses=0,
        cpp_resolves=0,
    )

    owner = owner_cls()
    values = owner.prepare_same_side_route_boundary_parent_values(
        prev_value_table,
        parent_keys,
        parent_rows,
    )

    assert values["parent_values"] == ("parent-0", None)
    assert values["available_rows"] == (
        ((("I", "C"), "I"), 0, parent_keys[0], "C", "parent-0"),
        ((("I", "C"), "I"), 2, parent_keys[0], "C", "parent-0"),
    )
    assert values["missing_rows"] == (
        ((("I", "C", "D"), "I"), 1, parent_keys[1], "D"),
    )
    assert values["missing_parent_keys"] == (parent_keys[1],)
    assert values["missing_parent_positions"] == (1,)
    assert int(values["hits"]) == 1
    assert int(values["misses"]) == 1
    assert int(values["rows"]) == 3
    assert int(values["available"]) == 2
    assert int(values["missing"]) == 1
    assert prev_value_table.batch_resolves == 1
    assert prev_value_table.cpp_resolves == 1
    assert prev_value_table.hits == 1
    assert prev_value_table.misses == 1

    stats = owner.stats()
    assert int(stats["same_side_route_boundary_parent_value_calls"]) == 1
    assert int(stats["same_side_route_boundary_parent_value_successes"]) == 1
    assert int(stats["same_side_route_boundary_parent_value_failures"]) == 0
    assert int(stats["same_side_route_boundary_parent_value_rows"]) == 3
    assert int(stats["same_side_route_boundary_parent_value_available"]) == 2
    assert int(stats["same_side_route_boundary_parent_value_missing"]) == 1
    assert int(stats["same_side_route_boundary_parent_value_hits"]) == 1
    assert int(stats["same_side_route_boundary_parent_value_misses"]) == 1


def test_cpp_moving_environment_prepares_same_side_route_missing_parent_builds():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "prepare_same_side_route_missing_parent_builds"):
        pytest.skip("C++ same-side missing parent builds are not available")

    parent_keys = (
        (("I",), "I"),
        (("I", "C"), "I"),
    )
    rows = (
        ((("I", "C"), "I"), 0, parent_keys[0], "C"),
        ((("I", "C", "D"), "I"), 1, parent_keys[1], "D"),
        ((("I", "C"), "I"), 2, parent_keys[0], "C"),
    )

    owner = owner_cls()
    planned = owner.prepare_same_side_route_missing_parent_builds(rows)

    assert planned["unique_parent_keys"] == parent_keys
    assert planned["parent_patterns"] == (("I",), ("I", "C"))
    assert planned["parent_rows"] == (
        (rows[0][0], 0, 0, parent_keys[0], "C"),
        (rows[1][0], 1, 1, parent_keys[1], "D"),
        (rows[2][0], 2, 0, parent_keys[0], "C"),
    )
    assert int(planned["rows"]) == 3
    assert int(planned["unique"]) == 2

    stats = owner.stats()
    assert int(stats["same_side_route_missing_parent_build_plan_calls"]) == 1
    assert int(stats["same_side_route_missing_parent_build_plan_successes"]) == 1
    assert int(stats["same_side_route_missing_parent_build_plan_failures"]) == 0
    assert int(stats["same_side_route_missing_parent_build_plan_rows"]) == 3
    assert int(stats["same_side_route_missing_parent_build_plan_unique"]) == 2


def test_cpp_moving_environment_prepares_same_side_route_built_parent_advances():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "prepare_same_side_route_built_parent_advances"):
        pytest.skip("C++ same-side built parent advances are not available")

    parent_keys = (
        (("I",), "I"),
        (("I", "C"), "I"),
    )
    parent_rows = (
        ((("I", "C"), "I"), 0, 0, parent_keys[0], "C"),
        ((("I", "C", "D"), "I"), 1, 1, parent_keys[1], "D"),
        ((("I", "C"), "I"), 2, 0, parent_keys[0], "C"),
    )

    owner = owner_cls()
    planned = owner.prepare_same_side_route_built_parent_advances(
        parent_rows,
        parent_keys,
        ("parent-0", None),
    )

    assert planned["parent_put_keys"] == (parent_keys[0],)
    assert planned["parent_put_values"] == ("parent-0",)
    assert planned["available_rows"] == (
        (parent_rows[0][0], 0, parent_keys[0], "C", "parent-0"),
        (parent_rows[2][0], 2, parent_keys[0], "C", "parent-0"),
    )
    assert planned["remaining_rows"] == (
        (parent_rows[1][0], 1, parent_keys[1], "D"),
    )
    assert int(planned["rows"]) == 3
    assert int(planned["available"]) == 2
    assert int(planned["missing"]) == 1
    assert int(planned["puts"]) == 1

    stats = owner.stats()
    assert int(stats["same_side_route_built_parent_advance_plan_calls"]) == 1
    assert int(stats["same_side_route_built_parent_advance_plan_successes"]) == 1
    assert int(stats["same_side_route_built_parent_advance_plan_failures"]) == 0
    assert int(stats["same_side_route_built_parent_advance_plan_rows"]) == 3
    assert int(stats["same_side_route_built_parent_advance_plan_available"]) == 2
    assert int(stats["same_side_route_built_parent_advance_plan_missing"]) == 1
    assert int(stats["same_side_route_built_parent_advance_plan_puts"]) == 1


def test_cpp_moving_environment_applies_same_side_route_boundary_parent_advances():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    if not hasattr(owner_cls, "apply_same_side_route_boundary_parent_advances"):
        pytest.skip("C++ same-side route boundary parent advances are not available")

    parent_value = ("parent", 0)
    other_parent_value = ("parent", 1)
    parent_key = (("I",), "I")
    other_parent_key = (("I", "C"), "I")
    rows = (
        ((("I", "C"), "I"), 0, parent_key, "C", parent_value),
        ((("I", "C"), "I"), 1, parent_key, "C", parent_value),
        ((("I", "D"), "I"), 2, other_parent_key, "D", other_parent_value),
    )
    calls = []

    def advance(pattern, value):
        calls.append((tuple(pattern), value))
        if tuple(pattern) == ("I", "D"):
            return None
        return ("advanced", tuple(pattern), value)

    owner = owner_cls()
    result = owner.apply_same_side_route_boundary_parent_advances(rows, advance)

    assert result["advanced_keys"] == (rows[0][0], rows[1][0])
    assert result["advanced_positions"] == (0, 1)
    assert result["advanced_values"] == (
        ("advanced", ("I", "C"), parent_value),
        ("advanced", ("I", "C"), parent_value),
    )
    assert result["remaining_keys"] == (rows[2][0],)
    assert result["remaining_positions"] == (2,)
    assert int(result["rows"]) == 3
    assert int(result["advanced"]) == 2
    assert int(result["remaining"]) == 1
    assert int(result["cache_hits"]) == 1
    assert int(result["cache_builds"]) == 2
    assert int(result["none"]) == 1
    assert calls == [
        (("I", "C"), parent_value),
        (("I", "D"), other_parent_value),
    ]

    stats = owner.stats()
    assert int(stats["same_side_route_boundary_parent_advance_calls"]) == 1
    assert int(stats["same_side_route_boundary_parent_advance_successes"]) == 1
    assert int(stats["same_side_route_boundary_parent_advance_failures"]) == 0
    assert int(stats["same_side_route_boundary_parent_advance_rows"]) == 3
    assert int(stats["same_side_route_boundary_parent_advance_advanced"]) == 2
    assert int(stats["same_side_route_boundary_parent_advance_remaining"]) == 1
    assert int(stats["same_side_route_boundary_parent_advance_cache_hits"]) == 1
    assert int(stats["same_side_route_boundary_parent_advance_cache_builds"]) == 2
    assert int(stats["same_side_route_boundary_parent_advance_none"]) == 1


def test_cpp_contextual_prepare_accepts_shared_key_specs():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    if (
        cpp_davidson.contextual_left_prepare_local_table_batch is None
        or cpp_davidson.contextual_right_prepare_local_table_batch is None
    ):
        pytest.skip("C++ contextual batch helpers are not available")

    revision = 7
    token = ("side", 1, 0)
    site_revisions = (3, 5, 9)
    pattern = ("X", "Y")
    left_rows = ((0, "L", ("left-cache",), ("left-table",)),)
    right_rows = ((0, "R", ("right-cache",), ("right-table",)),)
    qns = ("q",)

    prefix_cache = {
        ("shared_left_prefix", revision, pattern, site_revisions[:2]): (
            2,
            "left-env",
            qns,
        )
    }
    rows, shared, missing, unique = cpp_davidson.contextual_left_prepare_local_table_batch(
        {pattern: left_rows},
        prefix_cache,
        site_revisions,
        revision,
        token,
        2,
        "zero",
    )
    assert int(shared) == 1
    assert int(missing) == 0
    assert int(unique) == 1
    assert tuple(rows) == ((("L", 2, qns), "L", 2, qns),)
    assert (revision, token, pattern) in prefix_cache

    suffix_cache = {
        ("shared_right_suffix", revision, pattern, site_revisions[1:], "target"): (
            2,
            "right-env",
            qns,
        )
    }
    rows, shared, missing, unique = cpp_davidson.contextual_right_prepare_local_table_batch(
        {pattern: right_rows},
        suffix_cache,
        (site_revisions, "target"),
        revision,
        token,
        2,
        "zero",
    )
    assert int(shared) == 1
    assert int(missing) == 0
    assert int(unique) == 1
    assert tuple(rows) == ((("R", 2, qns), "R", 2, qns),)
    assert (revision, token, pattern) in suffix_cache


def test_moving_environment_contextual_prebuilt_finalizer_owns_prepare_and_finalize():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "finalize_contextual_prebuilt_batch"):
        pytest.skip("C++ MovingEnvironment prebuilt finalizer owner is not available")

    revision = 7
    token = ("side", 1, 0)
    site_revisions = (3, 5, 9)
    pattern = ("X", "Y")
    qns = ("q",)

    left_rows = ((0, "L", ("left-cache",), ("left-table",)),)
    prefix_cache = {
        ("shared_left_prefix", revision, pattern, site_revisions[:2]): (
            2,
            "left-env",
            qns,
        )
    }
    owner.install_contextual_prebuilt_finalizer(
        "prebuilt-left",
        site_revisions,
        revision,
        token,
        "left",
        2,
        "zero",
    )
    rows, shared, missing, unique = owner.prepare_contextual_prebuilt_local_table(
        "prebuilt-left",
        {pattern: left_rows},
        prefix_cache,
    )
    assert int(shared) == 1
    assert int(missing) == 0
    assert int(unique) == 1
    assert tuple(rows) == ((("L", 2, qns), "L", 2, qns),)

    left_results = [None]
    left_env_cache = {}
    left_table_keys = []
    left_table_values = []
    built, shared, hits, misses = owner.finalize_contextual_prebuilt_batch(
        "prebuilt-left",
        {pattern: left_rows},
        left_results,
        prefix_cache,
        {("L", 2, qns): ("WL", "IL")},
        left_env_cache,
        left_table_keys,
        left_table_values,
    )
    assert int(built) == 1
    assert int(shared) == 0
    assert int(hits) == 1
    assert int(misses) == 0
    assert left_results == [("left-env", "WL")]
    assert left_env_cache[("left-cache",)] == ("left-env", "WL")
    assert left_table_keys == [("left-table",)]
    assert left_table_values == [("left-env", "WL")]

    right_rows = ((0, "R", ("right-cache",), ("right-table",)),)
    suffix_cache = {
        ("shared_right_suffix", revision, pattern, site_revisions[1:], "target"): (
            2,
            "right-env",
            qns,
        )
    }
    owner.install_contextual_prebuilt_finalizer(
        "prebuilt-right",
        (site_revisions, "target"),
        revision,
        token,
        "right",
        2,
        "zero",
    )
    rows, shared, missing, unique = owner.prepare_contextual_prebuilt_local_table(
        "prebuilt-right",
        {pattern: right_rows},
        suffix_cache,
    )
    assert int(shared) == 1
    assert int(missing) == 0
    assert int(unique) == 1
    assert tuple(rows) == ((("R", 2, qns), "R", 2, qns),)

    right_results = [None]
    right_env_cache = {}
    right_table_keys = []
    right_table_values = []
    built, shared, hits, misses = owner.finalize_contextual_prebuilt_batch(
        "prebuilt-right",
        {pattern: right_rows},
        right_results,
        suffix_cache,
        {("R", 2, qns): ("WR", "IR")},
        right_env_cache,
        right_table_keys,
        right_table_values,
    )
    assert int(built) == 1
    assert int(shared) == 0
    assert int(hits) == 1
    assert int(misses) == 0
    assert right_results == [("WR", "right-env")]
    assert right_env_cache[("right-cache",)] == ("WR", "right-env")
    assert right_table_keys == [("right-table",)]
    assert right_table_values == [("WR", "right-env")]

    stats = owner.stats()
    assert int(stats["contextual_prebuilt_installs"]) == 2
    assert int(stats["contextual_prebuilt_prepare_calls"]) == 2
    assert int(stats["contextual_prebuilt_finalize_calls"]) == 2
    assert int(stats["contextual_prebuilt_built_results"]) == 2
    assert int(stats["contextual_prebuilt_local_hits"]) == 2
    assert int(stats["contextual_prebuilt_local_misses"]) == 0
    assert int(stats["contextual_prebuilt_failures"]) == 0


def test_moving_environment_contextual_boundary_batch_resolver_uses_owner_cache():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "resolve_contextual_boundary_batch"):
        pytest.skip("C++ contextual boundary batch resolver is not available")

    calls = []

    def builder(pattern, piece, family_name=None):
        calls.append(("single", tuple(pattern), str(piece), family_name))
        return ("single", tuple(pattern), str(piece), family_name)

    def batch_builder(keys, family_name=None):
        calls.append(("batch", tuple(keys), family_name))
        return tuple(
            ("batch", tuple(pattern), str(piece), family_name)
            for pattern, piece in keys
        )

    keys = (((1, 2), "A"), ((2, 3), "B"))
    cache = {}
    debug = {"calls": 0}

    values, table_ids, hits, misses, build_seconds, batch_used = (
        owner.resolve_contextual_boundary_batch(
            keys,
            cache,
            builder,
            batch_builder,
            "fam",
            None,
            False,
            debug,
            "left",
        )
    )

    assert hits == 0
    assert misses == 2
    assert build_seconds >= 0.0
    assert batch_used
    assert tuple(table_ids) == (-1, -1)
    assert tuple(values) == (
        ("batch", (1, 2), "A", "fam"),
        ("batch", (2, 3), "B", "fam"),
    )
    assert calls == [("batch", keys, "fam")]
    assert int(debug["left_batch_attempts"]) == 1
    assert int(debug["left_batch_returned_keys"]) == 2

    values, table_ids, hits, misses, _build_seconds, batch_used = (
        owner.resolve_contextual_boundary_batch(
            keys,
            cache,
            builder,
            batch_builder,
            "fam",
            None,
            False,
            debug,
            "left",
        )
    )

    assert hits == 2
    assert misses == 0
    assert not batch_used
    assert tuple(table_ids) == (-1, -1)
    assert tuple(values) == (
        ("batch", (1, 2), "A", "fam"),
        ("batch", (2, 3), "B", "fam"),
    )
    assert calls == [("batch", keys, "fam")]
    stats = owner.stats()
    assert int(stats["contextual_boundary_batch_resolve_calls"]) == 2
    assert int(stats["contextual_boundary_batch_resolve_batch_used"]) == 1
    assert int(stats["contextual_boundary_batch_resolve_failures"]) == 0


def test_moving_environment_contextual_boundary_precompute_owner_resolves_both_sides():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "precompute_contextual_boundaries"):
        pytest.skip("C++ contextual boundary precompute owner is not available")

    calls = []

    def left_builder(pattern, piece, family_name=None):
        calls.append(("left-single", tuple(pattern), str(piece), family_name))
        return ("left-single", tuple(pattern), str(piece), family_name)

    def right_builder(pattern, piece, family_name=None):
        calls.append(("right-single", tuple(pattern), str(piece), family_name))
        return ("right-single", tuple(pattern), str(piece), family_name)

    def left_batch(keys, family_name=None):
        calls.append(("left-batch", tuple(keys), family_name))
        return tuple(
            ("left-batch", tuple(pattern), str(piece), family_name)
            for pattern, piece in keys
        )

    def right_batch(keys, family_name=None):
        calls.append(("right-batch", tuple(keys), family_name))
        return tuple(
            ("right-batch", tuple(pattern), str(piece), family_name)
            for pattern, piece in keys
        )

    left_keys = (((1,), "A"),)
    right_keys = (((2,), "B"), ((3,), "C"))
    left_cache = {}
    right_cache = {}
    debug = {"calls": 0}

    result = owner.precompute_contextual_boundaries(
        left_keys,
        right_keys,
        left_cache,
        right_cache,
        left_builder,
        right_builder,
        left_batch,
        right_batch,
        "fam",
        None,
        None,
        debug,
    )
    (
        left_values,
        right_values,
        left_table_ids,
        right_table_ids,
        left_hits,
        left_misses,
        right_hits,
        right_misses,
        _left_seconds,
        _right_seconds,
        left_batch_used,
        right_batch_used,
    ) = result

    assert tuple(left_values) == (("left-batch", (1,), "A", "fam"),)
    assert tuple(right_values) == (
        ("right-batch", (2,), "B", "fam"),
        ("right-batch", (3,), "C", "fam"),
    )
    assert tuple(left_table_ids) == (-1,)
    assert tuple(right_table_ids) == (-1, -1)
    assert (left_hits, left_misses, right_hits, right_misses) == (0, 1, 0, 2)
    assert left_batch_used and right_batch_used
    assert calls == [
        ("left-batch", left_keys, "fam"),
        ("right-batch", right_keys, "fam"),
    ]

    result = owner.precompute_contextual_boundaries(
        left_keys,
        right_keys,
        left_cache,
        right_cache,
        left_builder,
        right_builder,
        left_batch,
        right_batch,
        "fam",
        None,
        None,
        debug,
    )
    assert tuple(result[0]) == (("left-batch", (1,), "A", "fam"),)
    assert tuple(result[1]) == (
        ("right-batch", (2,), "B", "fam"),
        ("right-batch", (3,), "C", "fam"),
    )
    assert tuple(result[4:8]) == (1, 0, 2, 0)
    assert not bool(result[10])
    assert not bool(result[11])
    assert calls == [
        ("left-batch", left_keys, "fam"),
        ("right-batch", right_keys, "fam"),
    ]

    stats = owner.stats()
    assert int(stats["contextual_boundary_precompute_calls"]) == 2
    assert int(stats["contextual_boundary_precompute_left_keys"]) == 2
    assert int(stats["contextual_boundary_precompute_right_keys"]) == 4
    assert int(stats["contextual_boundary_precompute_left_batch"]) == 1
    assert int(stats["contextual_boundary_precompute_right_batch"]) == 1
    assert int(stats["contextual_boundary_precompute_failures"]) == 0


def test_moving_environment_contextual_boundary_precompute_uses_installed_builders():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "precompute_contextual_boundaries_from_builders"):
        pytest.skip("C++ contextual boundary keyed precompute is not available")

    calls = []

    def left_builder(pattern, piece, family_name=None):
        calls.append(("left-single", tuple(pattern), str(piece), family_name))
        return ("left-single", tuple(pattern), str(piece), family_name)

    def right_builder(pattern, piece, family_name=None):
        calls.append(("right-single", tuple(pattern), str(piece), family_name))
        return ("right-single", tuple(pattern), str(piece), family_name)

    def left_batch(keys, family_name=None):
        calls.append(("left-batch", tuple(keys), family_name))
        return tuple(
            ("left-batch", tuple(pattern), str(piece), family_name)
            for pattern, piece in keys
        )

    def right_batch(keys, family_name=None):
        calls.append(("right-batch", tuple(keys), family_name))
        return tuple(
            ("right-batch", tuple(pattern), str(piece), family_name)
            for pattern, piece in keys
        )

    left_cache = {}
    right_cache = {}
    owner.install_contextual_boundary_batch_builder(
        "left-batch-builder",
        left_cache,
        left_builder,
        left_batch,
        "fam",
        None,
        "left",
    )
    owner.install_contextual_boundary_batch_builder(
        "right-batch-builder",
        right_cache,
        right_builder,
        right_batch,
        "fam",
        None,
        "right",
    )

    left_keys = (((1,), "A"),)
    right_keys = (((2,), "B"), ((3,), "C"))
    debug = {"calls": 0}
    result = owner.precompute_contextual_boundaries_from_builders(
        "left-batch-builder",
        "right-batch-builder",
        left_keys,
        right_keys,
        debug,
    )

    assert tuple(result[0]) == (("left-batch", (1,), "A", "fam"),)
    assert tuple(result[1]) == (
        ("right-batch", (2,), "B", "fam"),
        ("right-batch", (3,), "C", "fam"),
    )
    assert tuple(result[4:8]) == (0, 1, 0, 2)
    assert calls == [
        ("left-batch", left_keys, "fam"),
        ("right-batch", right_keys, "fam"),
    ]

    result = owner.precompute_contextual_boundaries_from_builders(
        "left-batch-builder",
        "right-batch-builder",
        left_keys,
        right_keys,
        debug,
    )
    assert tuple(result[4:8]) == (1, 0, 2, 0)
    assert calls == [
        ("left-batch", left_keys, "fam"),
        ("right-batch", right_keys, "fam"),
    ]
    stats = owner.stats()
    assert int(stats["contextual_boundary_batch_builder_records"]) == 2
    assert int(stats["contextual_boundary_batch_builder_installs"]) == 2
    assert int(stats["contextual_boundary_batch_builder_precompute_calls"]) == 2
    assert int(stats["contextual_boundary_batch_builder_resolve_calls"]) == 4
    assert int(stats["contextual_boundary_batch_builder_failures"]) == 0


def test_moving_environment_contextual_boundary_batch_plan_auto_installs_key():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "install_contextual_boundary_batch_plan_auto"):
        pytest.skip("C++ contextual batch plan auto installer is not available")

    token = ("left", 0, 0)
    shared_key_spec = (0, 0)
    boundary_cache = {}
    env_cache = {}
    local_table_cache = {}
    site_operator_cache = {}
    local_entries_cache = {}
    site_a_conj = ()
    site_b = ()

    key = owner.install_contextual_boundary_batch_plan_auto(
        "left",
        1,
        token,
        0,
        4,
        0,
        0,
        None,
        shared_key_spec,
        boundary_cache,
        env_cache,
        local_table_cache,
        site_operator_cache,
        local_entries_cache,
        object,
        site_a_conj,
        site_b,
        None,
        None,
    )
    key_again = owner.install_contextual_boundary_batch_plan_auto(
        "left",
        1,
        token,
        0,
        4,
        0,
        0,
        None,
        shared_key_spec,
        boundary_cache,
        env_cache,
        local_table_cache,
        site_operator_cache,
        local_entries_cache,
        object,
        site_a_conj,
        site_b,
        None,
        None,
    )

    assert isinstance(key, str)
    assert key == key_again
    assert "contextual_boundary_batch_plan|left" in key
    stats = owner.stats()
    assert int(stats["contextual_boundary_batch_plan_records"]) == 1
    assert int(stats["contextual_boundary_batch_plan_installs"]) == 2
    assert int(stats["contextual_boundary_batch_plan_replacements"]) == 0


def test_moving_environment_contextual_boundary_batch_plan_prefetches_local_entries():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "build_contextual_boundary_batch_from_plan"):
        pytest.skip("C++ contextual batch plan builder is not available")

    site_tensor = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_contextual_batch_plan_prefetch",
        assume_unique=True,
    )
    local_entries_cache = {}
    calls = []

    def fill_local_entries(piece, site):
        key = (str(piece), int(site))
        calls.append(key)
        local_entries_cache[key] = (
            ((0, 0, 0, 2.0 + 0.0j),),
            (0,),
        )
        return local_entries_cache[key]

    key = owner.install_contextual_boundary_batch_plan_auto(
        "left",
        5,
        ("tok-prefetch",),
        0,
        2,
        0,
        0,
        None,
        (3, 5),
        {},
        {},
        {},
        {},
        local_entries_cache,
        AbelianPackedBoundaryTensor,
        (site_tensor, site_tensor),
        (site_tensor, site_tensor),
        lambda qn: 0,
        fill_local_entries,
    )

    result = owner.build_contextual_boundary_batch_from_plan(
        key,
        (((), "X"),),
        "P",
    )

    assert len(result) == 1
    assert getattr(result[0][0], "_pyqed_packed_boundary_tensor", False)
    assert getattr(result[0][1], "_pyqed_packed_boundary_tensor", False)
    assert calls == [("X", 0)]
    stats = owner.stats()
    assert int(stats["contextual_boundary_batch_plan_successes"]) == 1
    assert int(stats["contextual_boundary_batch_plan_fallbacks"]) == 0
    assert int(stats["contextual_local_table_entry_prefetch"]) == 1
    assert int(stats["contextual_boundary_batch_plan_local_builds"]) == 1


def test_moving_environment_contextual_boundary_batch_plan_uses_preseeded_local_entries():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "build_contextual_boundary_batch_from_plan"):
        pytest.skip("C++ contextual batch plan builder is not available")

    site_tensor = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_contextual_batch_plan_preseeded",
        assume_unique=True,
    )
    local_entries_cache = {
        ("X", 0): (
            ((0, 0, 0, 2.0 + 0.0j),),
            (0,),
        )
    }

    key = owner.install_contextual_boundary_batch_plan_auto(
        "left",
        5,
        ("tok-preseeded",),
        0,
        2,
        0,
        0,
        None,
        (3, 5),
        {},
        {},
        {},
        {},
        local_entries_cache,
        AbelianPackedBoundaryTensor,
        (site_tensor, site_tensor),
        (site_tensor, site_tensor),
        None,
        None,
    )

    result = owner.build_contextual_boundary_batch_from_plan(
        key,
        (((), "X"),),
        "P",
    )

    assert len(result) == 1
    assert getattr(result[0][0], "_pyqed_packed_boundary_tensor", False)
    assert getattr(result[0][1], "_pyqed_packed_boundary_tensor", False)
    stats = owner.stats()
    assert int(stats["contextual_boundary_batch_plan_successes"]) == 1
    assert int(stats["contextual_boundary_batch_plan_fallbacks"]) == 0
    assert int(stats.get("contextual_local_table_entry_prefetch") or 0) == 0
    assert int(stats["contextual_boundary_batch_plan_local_builds"]) == 1


def test_moving_environment_contextual_boundary_batch_plan_reuses_local_entries():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "build_contextual_boundary_batch_from_plan"):
        pytest.skip("C++ contextual batch plan builder is not available")

    site_tensor = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_contextual_batch_plan_reuse",
        assume_unique=True,
    )
    local_entries_cache = {}
    calls = []

    def fill_local_entries(piece, site):
        key = (str(piece), int(site))
        calls.append(key)
        local_entries_cache[key] = (
            ((0, 0, 0, 2.0 + 0.0j),),
            (0,),
        )
        return local_entries_cache[key]

    def install(revision):
        return owner.install_contextual_boundary_batch_plan_auto(
            "left",
            int(revision),
            ("tok-reuse", int(revision)),
            0,
            2,
            0,
            0,
            None,
            (3, 5),
            {},
            {},
            {},
            {},
            local_entries_cache,
            AbelianPackedBoundaryTensor,
            (site_tensor, site_tensor),
            (site_tensor, site_tensor),
            lambda qn: 0,
            fill_local_entries,
        )

    first = owner.build_contextual_boundary_batch_from_plan(
        install(5),
        (((), "X"),),
        "P",
    )
    second = owner.build_contextual_boundary_batch_from_plan(
        install(6),
        (((), "X"),),
        "P",
    )

    assert len(first) == len(second) == 1
    assert calls == [("X", 0)]
    stats = owner.stats()
    assert int(stats["contextual_boundary_batch_plan_successes"]) == 2
    assert int(stats["contextual_boundary_batch_plan_fallbacks"]) == 0
    assert int(stats["contextual_local_table_entry_prefetch"]) == 1
    assert int(stats["contextual_boundary_batch_plan_local_builds"]) == 2


def test_moving_environment_contextual_boundary_batch_builder_auto_installs_key():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "install_contextual_boundary_batch_builder_auto"):
        pytest.skip("C++ contextual batch builder auto installer is not available")

    def builder(pattern, piece, family_name=None):
        return (tuple(pattern), str(piece), family_name)

    def batch_builder(keys, family_name=None):
        return tuple(
            (tuple(pattern), str(piece), family_name)
            for pattern, piece in keys
        )

    cache = {}
    key = owner.install_contextual_boundary_batch_builder_auto(
        cache,
        builder,
        batch_builder,
        "P",
        None,
        "left",
        "native-plan-key",
    )
    key_again = owner.install_contextual_boundary_batch_builder_auto(
        cache,
        builder,
        batch_builder,
        "P",
        None,
        "left",
        "native-plan-key",
    )

    assert isinstance(key, str)
    assert key == key_again
    assert "contextual_boundary_batch_builder|left" in key
    result = owner.resolve_contextual_boundary_batch_from_builder(
        key,
        (((1,), "A"),),
        False,
        {},
    )
    assert tuple(result[0]) == (((1,), "A", "P"),)
    stats = owner.stats()
    assert int(stats["contextual_boundary_batch_builder_records"]) == 1
    assert int(stats["contextual_boundary_batch_builder_installs"]) == 2
    assert int(stats["contextual_boundary_batch_builder_replacements"]) == 0


def test_moving_environment_contextual_boundary_batch_builder_auto_prefers_native_plan_key():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "install_contextual_boundary_batch_builder_auto"):
        pytest.skip("C++ contextual batch builder auto installer is not available")

    def builder_a(pattern, piece, family_name=None):
        return ("a", tuple(pattern), str(piece), family_name)

    def builder_b(pattern, piece, family_name=None):
        return ("b", tuple(pattern), str(piece), family_name)

    def batch_a(keys, family_name=None):
        return tuple(("a", tuple(pattern), str(piece), family_name) for pattern, piece in keys)

    def batch_b(keys, family_name=None):
        return tuple(("b", tuple(pattern), str(piece), family_name) for pattern, piece in keys)

    key_a = owner.install_contextual_boundary_batch_builder_auto(
        {},
        builder_a,
        batch_a,
        "P",
        None,
        "left",
        "native-plan-key",
    )
    key_b = owner.install_contextual_boundary_batch_builder_auto(
        {},
        builder_b,
        batch_b,
        "P",
        None,
        "left",
        "native-plan-key",
    )

    assert key_a == key_b
    assert "plan=native-plan-key" in key_a
    assert "|builder=" not in key_a
    assert "|batch=" not in key_a
    stats = owner.stats()
    assert int(stats["contextual_boundary_batch_builder_records"]) == 1
    assert int(stats["contextual_boundary_batch_builder_replacements"]) == 1


def test_moving_environment_contextual_prebuilt_finalizer_fuses_local_table():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "run_contextual_prebuilt_finalizer"):
        pytest.skip("C++ MovingEnvironment fused prebuilt finalizer is not available")

    def site_operator(piece, site, qns):
        return ("W", piece, int(site), tuple(qns))

    def initial_operator(W):
        return ("I", W)

    revision = 7
    token = ("side", 1, 0)
    site_revisions = (3, 5, 9)
    pattern = ("X", "Y")
    qns = ("q",)

    left_rows = ((0, "L", ("left-cache",), ("left-table",)),)
    prefix_cache = {
        ("shared_left_prefix", revision, pattern, site_revisions[:2]): (
            2,
            "left-env",
            qns,
        )
    }
    left_results = [None]
    left_env_cache = {}
    left_table_keys = []
    left_table_values = []
    owner.install_contextual_prebuilt_finalizer(
        "prebuilt-left-fused",
        site_revisions,
        revision,
        token,
        "left",
        2,
        "zero",
    )
    owner.install_contextual_local_table_cache(
        "ltable-left-fused",
        {},
        "left",
        None,
        site_operator,
        initial_operator,
    )
    fused = owner.run_contextual_prebuilt_finalizer(
        "prebuilt-left-fused",
        "ltable-left-fused",
        {pattern: left_rows},
        left_results,
        prefix_cache,
        left_env_cache,
        left_table_keys,
        left_table_values,
    )
    assert tuple(int(x) for x in fused[:7]) == (1, 1, 0, 1, 1, 0, 1)
    assert bool(fused[7])
    assert tuple(int(x) for x in fused[8:14]) == (0, 1, 0, 0, 1, 0)
    assert left_results == [("left-env", ("W", "L", 2, qns))]
    assert left_env_cache[("left-cache",)] == left_results[0]
    assert left_table_keys == [("left-table",)]
    assert left_table_values == left_results

    right_rows = ((0, "R", ("right-cache",), ("right-table",)),)
    suffix_cache = {
        ("shared_right_suffix", revision, pattern, site_revisions[1:], "target"): (
            2,
            "right-env",
            qns,
        )
    }
    right_results = [None]
    right_env_cache = {}
    right_table_keys = []
    right_table_values = []
    owner.install_contextual_prebuilt_finalizer(
        "prebuilt-right-fused",
        (site_revisions, "target"),
        revision,
        token,
        "right",
        2,
        "zero",
    )
    owner.install_contextual_local_table_cache(
        "ltable-right-fused",
        {},
        "right",
        "target",
        site_operator,
        initial_operator,
    )
    fused = owner.run_contextual_prebuilt_finalizer(
        "prebuilt-right-fused",
        "ltable-right-fused",
        {pattern: right_rows},
        right_results,
        suffix_cache,
        right_env_cache,
        right_table_keys,
        right_table_values,
    )
    assert tuple(int(x) for x in fused[:7]) == (1, 1, 0, 1, 1, 0, 1)
    assert bool(fused[7])
    assert tuple(int(x) for x in fused[8:14]) == (0, 1, 0, 0, 1, 0)
    assert right_results == [(("W", "R", 2, qns), "right-env")]
    assert right_env_cache[("right-cache",)] == right_results[0]
    assert right_table_keys == [("right-table",)]
    assert right_table_values == right_results

    stats = owner.stats()
    assert int(stats["contextual_prebuilt_fused_calls"]) == 2
    assert int(stats["contextual_prebuilt_fused_successes"]) == 2
    assert int(stats["contextual_prebuilt_fused_failures"]) == 0
    assert int(stats["contextual_prebuilt_prepare_calls"]) == 2
    assert int(stats["contextual_prebuilt_finalize_calls"]) == 2
    assert int(stats["contextual_local_table_probe_calls"]) == 2
    assert int(stats["contextual_local_table_fill_calls"]) == 2


def test_cpp_contextual_probe_local_table_cache_splits_hits_and_misses():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    probe = getattr(cpp_davidson, "contextual_probe_local_table_cache", None)
    if probe is None:
        pytest.skip("C++ contextual local-table probe is not available")

    qns = ("q",)
    rows = (
        (("L", 0, qns), "L", 0, qns),
        (("M", 0, qns), "M", 0, qns),
        (("L", 0, qns), "L", 0, qns),
    )
    cache = {("left", "L", 0, qns): ("WL", "EL")}

    table, missing, hits, misses, duplicates = probe(rows, cache, "left")

    assert dict(table) == {("L", 0, qns): ("WL", "EL")}
    assert tuple(missing) == ((("M", 0, qns), "M", 0, qns),)
    assert int(hits) == 1
    assert int(misses) == 1
    assert int(duplicates) == 1

    right_rows = ((("R", 1, qns), "R", 1, qns),)
    right_cache = {("right", "R", 1, qns, "target"): ("WR", "FR")}

    table, missing, hits, misses, duplicates = probe(
        right_rows,
        right_cache,
        "right",
        "target",
    )

    assert dict(table) == {("R", 1, qns): ("WR", "FR")}
    assert tuple(missing) == ()
    assert int(hits) == 1
    assert int(misses) == 0
    assert int(duplicates) == 0


def test_cpp_contextual_fill_local_table_cache_misses_updates_cache():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    fill = getattr(cpp_davidson, "contextual_fill_local_table_cache_misses", None)
    if fill is None:
        pytest.skip("C++ contextual local-table fill is not available")

    def site_operator(piece, site, qns):
        return ("W", piece, int(site), tuple(qns))

    def initial_operator(W):
        return ("I", W)

    qns = ("q",)
    rows = (
        (("L", 0, qns), "L", 0, qns),
        (("M", 0, qns), "M", 0, qns),
    )
    cache = {("left", "L", 0, qns): ("WL", "EL")}

    table, complete, hits, builds, skipped = fill(
        rows,
        cache,
        {},
        "left",
        site_operator,
        initial_operator,
    )

    assert bool(complete)
    assert int(hits) == 1
    assert int(builds) == 1
    assert int(skipped) == 0
    assert dict(table)[("L", 0, qns)] == ("WL", "EL")
    assert dict(table)[("M", 0, qns)] == (
        ("W", "M", 0, qns),
        ("I", ("W", "M", 0, qns)),
    )
    assert cache[("left", "M", 0, qns)] == dict(table)[("M", 0, qns)]

    right_rows = ((("R", 1, qns), "R", 1, qns),)
    right_cache = {}
    table, complete, hits, builds, skipped = fill(
        right_rows,
        right_cache,
        {},
        "right",
        site_operator,
        initial_operator,
        "target",
    )

    assert bool(complete)
    assert int(hits) == 0
    assert int(builds) == 1
    assert int(skipped) == 0
    assert right_cache[("right", "R", 1, qns, "target")] == dict(table)[
        ("R", 1, qns)
    ]


def test_moving_environment_contextual_local_table_owns_probe_and_fill():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None or not hasattr(owner_cls(), "probe_contextual_local_table_cache"):
        pytest.skip("C++ MovingEnvironment contextual local-table owner is not available")

    def site_operator(piece, site, qns):
        return ("W", piece, int(site), tuple(qns))

    def initial_operator(W):
        return ("I", W)

    qns = ("q",)
    rows = (
        (("L", 0, qns), "L", 0, qns),
        (("M", 0, qns), "M", 0, qns),
        (("L", 0, qns), "L", 0, qns),
    )
    cache = {("left", "L", 0, qns): ("WL", "EL")}
    owner = owner_cls()
    owner.install_contextual_local_table_cache(
        "ltable-left",
        cache,
        "left",
    )

    table, missing, hits, misses, duplicates = (
        owner.probe_contextual_local_table_cache("ltable-left", rows)
    )

    assert dict(table) == {("L", 0, qns): ("WL", "EL")}
    assert tuple(missing) == ((("M", 0, qns), "M", 0, qns),)
    assert int(hits) == 1
    assert int(misses) == 1
    assert int(duplicates) == 1

    owner.install_contextual_local_table_cache(
        "ltable-left",
        cache,
        "left",
        None,
        site_operator,
        initial_operator,
    )
    filled, complete, hits, builds, skipped = (
        owner.fill_contextual_local_table_cache_misses(
            "ltable-left",
            missing,
            dict(table),
        )
    )

    assert bool(complete)
    assert int(hits) == 0
    assert int(builds) == 1
    assert int(skipped) == 0
    assert dict(filled)[("M", 0, qns)] == (
        ("W", "M", 0, qns),
        ("I", ("W", "M", 0, qns)),
    )
    assert cache[("left", "M", 0, qns)] == dict(filled)[("M", 0, qns)]

    stats = owner.stats()
    assert int(stats["contextual_local_table_records"]) == 1
    assert int(stats["contextual_local_table_installs"]) == 2
    assert int(stats["contextual_local_table_probe_calls"]) == 1
    assert int(stats["contextual_local_table_fill_calls"]) == 1
    assert int(stats["contextual_local_table_hits"]) == 1
    assert int(stats["contextual_local_table_misses"]) == 1
    assert int(stats["contextual_local_table_builds"]) == 1


def test_moving_environment_contextual_local_table_builds_packed_entries():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None or not hasattr(owner_cls(), "fill_contextual_local_table_cache_misses"):
        pytest.skip("C++ MovingEnvironment contextual local-table owner is not available")

    qns = (0,)
    rows = ((("X", 0, qns), "X", 0, qns),)
    local_entries_cache = {
        ("X", 0): (
            ((0, 0, 0, 2.0 + 0.0j),),
            (0,),
        )
    }
    cache = {}
    owner = owner_cls()
    owner.install_contextual_local_table_cache(
        "packed-ltable-left",
        cache,
        "left",
        None,
        None,
        None,
        AbelianPackedBoundaryTensor,
        local_entries_cache,
        None,
        None,
        0,
    )

    table, complete, hits, builds, skipped = (
        owner.fill_contextual_local_table_cache_misses(
            "packed-ltable-left",
            rows,
            {},
        )
    )

    assert bool(complete)
    assert int(hits) == 0
    assert int(builds) == 1
    assert int(skipped) == 0
    local = dict(table)[("X", 0, qns)]
    assert len(local) == 2
    assert getattr(local[0], "_pyqed_packed_boundary_tensor", False)
    assert getattr(local[1], "_pyqed_packed_boundary_tensor", False)
    assert cache[("left", "X", 0, qns)] == local
    stats = owner.stats()
    assert int(stats["contextual_local_table_packed_builds"]) == 1
    assert int(stats["contextual_local_table_entry_prefetch"]) == 0
    assert int(stats["contextual_local_table_packed_failures"]) == 0


def test_cpp_contextual_partition_pending_rows_buckets_native_state():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    part = getattr(cpp_davidson, "contextual_partition_pending_rows", None)
    if part is None:
        pytest.skip("C++ contextual pending partition is not available")

    revision = 3
    token = ("tok",)
    cached_key = (revision, token, ("A",), "L")
    pending = (
        (0, ("A",), "L"),
        (1, ("B", "C"), "M"),
        (2, (), "N"),
    )
    env_cache = {cached_key: "cached-left"}
    results = [None, None, None]
    pattern_items = {}
    advance_rows = []
    table_put_keys = []
    table_put_values = []

    env_hits, advance, pattern_rows, buckets = part(
        pending,
        env_cache,
        results,
        pattern_items,
        advance_rows,
        table_put_keys,
        table_put_values,
        "left",
        revision,
        token,
        True,
        True,
    )

    assert int(env_hits) == 1
    assert int(advance) == 1
    assert int(pattern_rows) == 1
    assert int(buckets) == 1
    assert results == ["cached-left", None, None]
    assert table_put_keys == [(("A",), "L")]
    assert table_put_values == ["cached-left"]
    assert advance_rows == [
        (
            1,
            ("B", "C"),
            "M",
            (revision, token, ("B", "C"), "M"),
            (("B", "C"), "M"),
            (("B",), "C"),
        )
    ]
    assert pattern_items == {
        (): [(2, "N", (revision, token, (), "N"), ((), "N"))]
    }

    right_results = [None]
    right_pattern_items = {}
    right_advance_rows = []
    env_hits, advance, pattern_rows, buckets = part(
        ((0, ("R0", "R1"), "X"),),
        {},
        right_results,
        right_pattern_items,
        right_advance_rows,
        [],
        [],
        "right",
        revision,
        token,
        True,
        False,
    )

    assert int(env_hits) == 0
    assert int(advance) == 1
    assert int(pattern_rows) == 0
    assert int(buckets) == 0
    assert right_advance_rows == [
        (
            0,
            ("R0", "R1"),
            "X",
            (revision, token, ("R0", "R1"), "X"),
            (("R0", "R1"), "X"),
            (("R1",), "R0"),
        )
    ]


def test_moving_environment_contextual_partition_owns_native_state():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None or not hasattr(owner_cls(), "partition_contextual_pending_rows"):
        pytest.skip("C++ MovingEnvironment contextual partition owner is not available")

    revision = 31
    token = ("tok",)
    cached_key = (revision, token, ("A",), "L")
    pending = (
        (0, ("A",), "L"),
        (1, ("B", "C"), "M"),
        (2, (), "N"),
    )
    env_cache = {cached_key: "cached-left"}
    results = [None, None, None]
    pattern_items = {}
    advance_rows = []
    table_put_keys = []
    table_put_values = []
    owner = owner_cls()
    owner.install_contextual_pending_partition(
        "partition-left",
        revision,
        token,
        "left",
        True,
        True,
    )

    env_hits, advance, pattern_rows, buckets = owner.partition_contextual_pending_rows(
        "partition-left",
        pending,
        env_cache,
        results,
        pattern_items,
        advance_rows,
        table_put_keys,
        table_put_values,
    )

    assert int(env_hits) == 1
    assert int(advance) == 1
    assert int(pattern_rows) == 1
    assert int(buckets) == 1
    assert results == ["cached-left", None, None]
    assert table_put_keys == [(("A",), "L")]
    assert table_put_values == ["cached-left"]
    assert advance_rows == [
        (
            1,
            ("B", "C"),
            "M",
            (revision, token, ("B", "C"), "M"),
            (("B", "C"), "M"),
            (("B",), "C"),
        )
    ]
    assert pattern_items == {
        (): [(2, "N", (revision, token, (), "N"), ((), "N"))]
    }

    right_results = [None]
    right_pattern_items = {}
    right_advance_rows = []
    owner.install_contextual_pending_partition(
        "partition-right",
        revision,
        token,
        "right",
        True,
        False,
    )
    env_hits, advance, pattern_rows, buckets = owner.partition_contextual_pending_rows(
        "partition-right",
        ((0, ("R0", "R1"), "X"),),
        {},
        right_results,
        right_pattern_items,
        right_advance_rows,
        [],
        [],
    )

    assert int(env_hits) == 0
    assert int(advance) == 1
    assert int(pattern_rows) == 0
    assert int(buckets) == 0
    assert right_advance_rows == [
        (
            0,
            ("R0", "R1"),
            "X",
            (revision, token, ("R0", "R1"), "X"),
            (("R0", "R1"), "X"),
            (("R1",), "R0"),
        )
    ]
    stats = owner.stats()
    assert int(stats["contextual_partition_records"]) == 2
    assert int(stats["contextual_partition_installs"]) == 2
    assert int(stats["contextual_partition_calls"]) == 2
    assert int(stats["contextual_partition_env_hits"]) == 1
    assert int(stats["contextual_partition_advance"]) == 2
    assert int(stats["contextual_partition_pattern_rows"]) == 1
    assert int(stats["contextual_partition_buckets"]) == 1


def test_cpp_contextual_prepare_boundary_build_wave_uses_live_cache():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    planner = getattr(cpp_davidson, "contextual_prepare_boundary_build_wave", None)
    if planner is None:
        pytest.skip("C++ contextual boundary wave planner is not available")

    revision = 11
    token = ("tok",)
    site_revisions = (3, 5, 7)
    base = (0, None, ("zero",))
    left_cache = {
        (revision, token, ()): base,
        ("shared_left_prefix", revision, ("C",), site_revisions[:1]): (
            1,
            "shared-C",
            ("qc",),
        ),
    }
    failed = set()

    rows, shared, cached, deferred, inherited, site_skips, failed_skips, closure = (
        planner(
            (("A", "B"), ("C",)),
            left_cache,
            site_revisions,
            revision,
            token,
            failed,
            "left",
            3,
        )
    )

    assert int(shared) == 1
    assert int(cached) == 0
    assert int(deferred) == 1
    assert int(inherited) == 0
    assert int(site_skips) == 0
    assert int(failed_skips) == 0
    assert int(closure) == 3
    assert left_cache[(revision, token, ("C",))] == (
        1,
        "shared-C",
        ("qc",),
    )
    assert tuple(rows) == (
        (
            ("A",),
            (revision, token, ("A",)),
            ("shared_left_prefix", revision, ("A",), site_revisions[:1]),
            (),
            base,
            0,
            "A",
        ),
    )

    left_cache[(revision, token, ("A",))] = (1, "env-A", ("qa",))
    rows, shared, cached, deferred, inherited, site_skips, failed_skips, closure = (
        planner(
            (("A", "B"), ("C",)),
            left_cache,
            site_revisions,
            revision,
            token,
            failed,
            "left",
            3,
        )
    )

    assert int(shared) == 0
    assert int(cached) == 2
    assert int(deferred) == 0
    assert tuple(rows) == (
        (
            ("A", "B"),
            (revision, token, ("A", "B")),
            ("shared_left_prefix", revision, ("A", "B"), site_revisions[:2]),
            ("A",),
            (1, "env-A", ("qa",)),
            1,
            "B",
        ),
    )

    right_cache = {(revision, token, ()): (3, None, ("zero",))}
    rows, shared, cached, deferred, inherited, site_skips, failed_skips, closure = (
        planner(
            (("R0", "R1"),),
            right_cache,
            (site_revisions, "target"),
            revision,
            token,
            set(),
            "right",
            3,
            1,
        )
    )

    assert int(shared) == 0
    assert int(deferred) == 1
    assert tuple(rows) == (
        (
            ("R1",),
            (revision, token, ("R1",)),
            (
                "shared_right_suffix",
                revision,
                ("R1",),
                site_revisions[2:],
                "target",
            ),
            (),
            (3, None, ("zero",)),
            2,
            "R1",
        ),
    )


def test_moving_environment_contextual_planner_owns_key_specs():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None or not hasattr(owner_cls(), "plan_contextual_boundary_wave"):
        pytest.skip("C++ MovingEnvironment contextual planner owner is not available")

    owner = owner_cls()
    revision = 23
    token = ("tok",)
    site_revisions = (3, 5, 7)
    base = (0, None, ("zero",))
    left_cache = {
        (revision, token, ()): base,
        ("shared_left_prefix", revision, ("C",), site_revisions[:1]): (
            1,
            "shared-C",
            ("qc",),
        ),
    }
    owner.install_contextual_boundary_planner(
        "planner-left",
        site_revisions,
        revision,
        token,
        "left",
        3,
        0,
    )
    rows, shared, cached, deferred, inherited, site_skips, failed_skips, closure = (
        owner.plan_contextual_boundary_wave(
            "planner-left",
            (("A", "B"), ("C",)),
            left_cache,
            set(),
        )
    )

    assert int(shared) == 1
    assert int(deferred) == 1
    assert int(inherited) == 0
    assert int(site_skips) == 0
    assert int(failed_skips) == 0
    assert int(closure) == 3
    assert tuple(rows) == (
        (
            ("A",),
            (revision, token, ("A",)),
            ("shared_left_prefix", revision, ("A",), site_revisions[:1]),
            (),
            base,
            0,
            "A",
        ),
    )

    right_cache = {(revision, token, ()): (3, None, ("zero",))}
    owner.install_contextual_boundary_planner(
        "planner-right",
        (site_revisions, "target"),
        revision,
        token,
        "right",
        3,
        1,
    )
    rows, shared, cached, deferred, inherited, site_skips, failed_skips, closure = (
        owner.plan_contextual_boundary_wave(
            "planner-right",
            (("R0", "R1"),),
            right_cache,
            set(),
        )
    )

    assert int(shared) == 0
    assert int(deferred) == 1
    assert tuple(rows) == (
        (
            ("R1",),
            (revision, token, ("R1",)),
            (
                "shared_right_suffix",
                revision,
                ("R1",),
                site_revisions[2:],
                "target",
            ),
            (),
            (3, None, ("zero",)),
            2,
            "R1",
        ),
    )
    stats = owner.stats()
    assert int(stats["contextual_planner_records"]) == 2
    assert int(stats["contextual_planner_installs"]) == 2
    assert int(stats["contextual_planner_calls"]) == 2
    assert int(stats["contextual_planner_rows"]) == 2
    assert int(stats["contextual_planner_deferred"]) == 2


def test_cpp_contextual_execute_boundary_build_wave_updates_cache():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    execute = getattr(cpp_davidson, "contextual_execute_boundary_build_wave", None)
    if execute is None:
        pytest.skip("C++ contextual boundary wave executor is not available")

    def identity(site, env):
        return SimpleNamespace(qns=((f"ident-{int(site)}",),))

    def site_operator(piece, site, qns):
        if str(piece) == "BAD":
            return None
        return ("W", str(piece), int(site), tuple(qns))

    def initial(W):
        return SimpleNamespace(qns=(("initial",),))

    def contract(W, site, env):
        return SimpleNamespace(qns=((f"{W[1]}-{int(site)}",),))

    revision = 5
    token = ("tok",)
    env = SimpleNamespace(qns=(("parent",),))
    rows = (
        (
            ("A",),
            (revision, token, ("A",)),
            ("shared-left", ("A",)),
            (),
            (0, None, ("zero",)),
            0,
            "A",
        ),
        (
            ("I",),
            (revision, token, ("I",)),
            ("shared-left", ("I",)),
            (),
            (1, env, ("parent",)),
            1,
            "I",
        ),
        (
            ("BAD",),
            (revision, token, ("BAD",)),
            ("shared-left", ("BAD",)),
            (),
            (1, env, ("parent",)),
            1,
            "BAD",
        ),
    )
    cache = {}
    failed = set()

    built, identity_built, generic_built, failures, n_rows = execute(
        rows,
        cache,
        failed,
        "left",
        "zero",
        identity,
        site_operator,
        initial,
        contract,
    )

    assert int(n_rows) == 3
    assert int(built) == 2
    assert int(identity_built) == 1
    assert int(generic_built) == 1
    assert int(failures) == 1
    assert failed == {("BAD",)}
    assert cache[(revision, token, ("A",))][0] == 1
    assert cache[(revision, token, ("A",))][2] == ("A-0",)
    assert cache[("shared-left", ("A",))] == cache[(revision, token, ("A",))]
    assert cache[(revision, token, ("I",))][0] == 2
    assert cache[(revision, token, ("I",))][2] == ("ident-1",)

    right_cache = {}
    built, identity_built, generic_built, failures, n_rows = execute(
        (
            (
                ("R",),
                (revision, token, ("R",)),
                ("shared-right", ("R",)),
                (),
                (3, None, ("zero",)),
                2,
                "R",
            ),
        ),
        right_cache,
        set(),
        "right",
        "zero",
        identity,
        site_operator,
        initial,
        contract,
    )

    assert int(built) == 1
    assert int(identity_built) == 0
    assert int(generic_built) == 1
    assert int(failures) == 0
    assert right_cache[(revision, token, ("R",))][0] == 2
    assert right_cache[(revision, token, ("R",))][2] == ("R-2",)


def test_cpp_contextual_execute_boundary_build_wave_packed_builds_payloads():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    execute = getattr(
        cpp_davidson,
        "contextual_execute_boundary_build_wave_packed",
        None,
    )
    if execute is None:
        pytest.skip("C++ contextual packed boundary wave executor is not available")

    site_tensor = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_site",
        assume_unique=True,
    )
    env = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_env",
        assume_unique=True,
    )
    revision = 13
    token = ("tok",)
    rows = (
        (
            ("X",),
            (revision, token, ("X",)),
            ("shared-left", ("X",)),
            (),
            (0, None, (0,)),
            0,
            "X",
        ),
        (
            ("I",),
            (revision, token, ("I",)),
            ("shared-left", ("I",)),
            (),
            (1, env, (0,)),
            0,
            "I",
        ),
    )
    local_entries_cache = {
        ("X", 0): (
            ((0, 0, 0, 2.0 + 0.0j),),
            (0,),
        )
    }
    op_cache = {}
    boundary_cache = {}
    failed = set()

    built, identity_built, generic_built, failures, n_rows, unsupported, hits, builds = (
        execute(
            rows,
            boundary_cache,
            failed,
            "left",
            0,
            0,
            op_cache,
            local_entries_cache,
            AbelianPackedBoundaryTensor,
            (site_tensor,),
            (site_tensor,),
            lambda qn: 0,
        )
    )

    assert int(n_rows) == 2
    assert int(unsupported) == 0
    assert int(built) == 2
    assert int(identity_built) == 1
    assert int(generic_built) == 1
    assert int(failures) == 0
    assert int(hits) == 0
    assert int(builds) == 1
    assert failed == set()
    assert boundary_cache[(revision, token, ("X",))][0] == 1
    assert boundary_cache[(revision, token, ("I",))][0] == 1
    assert boundary_cache[("shared-left", ("X",))] == boundary_cache[
        (revision, token, ("X",))
    ]
    assert ("packed_left", "X", 0, (0,), "direct_family_site_operator_left") in op_cache


def test_moving_environment_contextual_wave_owner_builds_payloads():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None or not hasattr(owner_cls(), "execute_contextual_wave"):
        pytest.skip("C++ MovingEnvironment contextual wave owner is not available")

    site_tensor = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_site_owner",
        assume_unique=True,
    )
    env = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_env_owner",
        assume_unique=True,
    )
    revision = 17
    token = ("tok",)
    rows = (
        (
            ("X",),
            (revision, token, ("X",)),
            ("shared-left", ("X",)),
            (),
            (0, None, (0,)),
            0,
            "X",
        ),
        (
            ("I",),
            (revision, token, ("I",)),
            ("shared-left", ("I",)),
            (),
            (1, env, (0,)),
            0,
            "I",
        ),
    )
    local_entries_cache = {}

    def fill_local_entries(piece, site):
        key = (str(piece), int(site))
        local_entries_cache[key] = (
            ((0, 0, 0, 2.0 + 0.0j),),
            (0,),
        )
        return local_entries_cache[key]

    op_cache = {}
    boundary_cache = {}
    failed = set()
    owner = owner_cls()
    owner.install_contextual_wave_executor(
        "owner-packed-test",
        AbelianPackedBoundaryTensor,
        op_cache,
        local_entries_cache,
        (site_tensor,),
        (site_tensor,),
        lambda qn: 0,
        fill_local_entries,
    )

    built, identity_built, generic_built, failures, n_rows, unsupported, hits, builds = (
        owner.execute_contextual_wave(
            "owner-packed-test",
            rows,
            boundary_cache,
            failed,
            "left",
            0,
            0,
        )
    )

    assert int(n_rows) == 2
    assert int(unsupported) == 0
    assert int(built) == 2
    assert int(identity_built) == 1
    assert int(generic_built) == 1
    assert int(failures) == 0
    assert int(hits) == 0
    assert int(builds) == 1
    assert failed == set()
    assert boundary_cache[(revision, token, ("X",))][0] == 1
    assert boundary_cache[(revision, token, ("I",))][0] == 1
    assert ("packed_left", "X", 0, (0,), "direct_family_site_operator_left") in op_cache
    stats = owner.stats()
    assert int(stats["contextual_wave_records"]) == 1
    assert int(stats["contextual_wave_installs"]) == 1
    assert int(stats["contextual_wave_execute_calls"]) == 1
    assert int(stats["contextual_wave_rows"]) == 2
    assert int(stats["contextual_wave_built"]) == 2
    assert int(stats["contextual_wave_op_builds"]) == 1
    assert int(stats["contextual_wave_prefetch_entries"]) == 1
    assert int(stats["contextual_wave_prefetch_failures"]) == 0


def test_moving_environment_contextual_wave_owner_fuses_plan_and_execute():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment contextual wave owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "run_contextual_boundary_wave"):
        pytest.skip("C++ MovingEnvironment fused contextual wave owner is not available")

    site_tensor = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_site_fused_owner",
        assume_unique=True,
    )
    revision = 19
    token = ("tok",)
    site_revisions = (3, 5)
    local_entries_cache = {}

    def fill_local_entries(piece, site):
        key = (str(piece), int(site))
        local_entries_cache[key] = (
            ((0, 0, 0, 2.0 + 0.0j),),
            (0,),
        )
        return local_entries_cache[key]

    op_cache = {}
    boundary_cache = {(revision, token, ()): (0, None, (0,))}
    failed = set()
    owner.install_contextual_boundary_planner(
        "planner-fused-left",
        site_revisions,
        revision,
        token,
        "left",
        2,
        0,
    )
    owner.install_contextual_wave_executor(
        "wave-fused-left",
        AbelianPackedBoundaryTensor,
        op_cache,
        local_entries_cache,
        (site_tensor, site_tensor),
        (site_tensor, site_tensor),
        lambda qn: 0,
        fill_local_entries,
    )

    first = owner.run_contextual_boundary_wave(
        "planner-fused-left",
        "wave-fused-left",
        (("X", "I"),),
        boundary_cache,
        failed,
        0,
        0,
    )
    assert tuple(int(x) for x in first[:15]) == (
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
    )
    assert boundary_cache[(revision, token, ("X",))][0] == 1
    assert (revision, token, ("X", "I")) not in boundary_cache

    second = owner.run_contextual_boundary_wave(
        "planner-fused-left",
        "wave-fused-left",
        (("X", "I"),),
        boundary_cache,
        failed,
        0,
        0,
    )
    assert tuple(int(x) for x in second[:15]) == (
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        2,
    )
    assert boundary_cache[(revision, token, ("X", "I"))][0] == 2
    assert failed == set()

    stats = owner.stats()
    assert int(stats["contextual_wave_fused_calls"]) == 2
    assert int(stats["contextual_wave_fused_successes"]) == 2
    assert int(stats["contextual_wave_fused_failures"]) == 0
    assert int(stats["contextual_planner_calls"]) == 2
    assert int(stats["contextual_wave_execute_calls"]) == 2
    assert int(stats["contextual_wave_built"]) == 2
    assert int(stats["contextual_wave_identity"]) == 1
    assert int(stats["contextual_wave_generic"]) == 1


def test_moving_environment_contextual_wave_owner_runs_fixed_point_loop():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")
    owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
    if owner_cls is None:
        pytest.skip("C++ MovingEnvironment contextual wave owner is not available")
    owner = owner_cls()
    if not hasattr(owner, "run_contextual_boundary_wave_loop"):
        pytest.skip("C++ MovingEnvironment contextual wave loop is not available")

    site_tensor = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="test_site_fused_owner_loop",
        assume_unique=True,
    )
    revision = 23
    token = ("tok-loop",)
    site_revisions = (3, 5)
    local_entries_cache = {
        ("X", 0): (
            ((0, 0, 0, 2.0 + 0.0j),),
            (0,),
        )
    }
    op_cache = {}
    boundary_cache = {(revision, token, ()): (0, None, (0,))}
    failed = set()
    owner.install_contextual_boundary_planner(
        "planner-loop-left",
        site_revisions,
        revision,
        token,
        "left",
        2,
        0,
    )
    owner.install_contextual_wave_executor(
        "wave-loop-left",
        AbelianPackedBoundaryTensor,
        op_cache,
        local_entries_cache,
        (site_tensor, site_tensor),
        (site_tensor, site_tensor),
        lambda qn: 0,
        None,
    )

    result = owner.run_contextual_boundary_wave_loop(
        "planner-loop-left",
        "wave-loop-left",
        (("X", "I"),),
        boundary_cache,
        failed,
        0,
        0,
        8,
    )
    assert tuple(int(x) for x in result[:17]) == (
        2,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        3,
        1,
        0,
        0,
        0,
        2,
        2,
        3,
    )
    assert boundary_cache[(revision, token, ("X",))][0] == 1
    assert boundary_cache[(revision, token, ("X", "I"))][0] == 2
    assert failed == set()

    stats = owner.stats()
    assert int(stats["contextual_wave_loop_calls"]) == 1
    assert int(stats["contextual_wave_loop_iterations"]) == 3
    assert int(stats["contextual_wave_loop_successes"]) == 1
    assert int(stats["contextual_wave_loop_failures"]) == 0
    assert int(stats["contextual_wave_prefetch_entries"]) == 0


def test_moving_environment_reuses_cpp_raw_route_plan_for_named_family():
    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "RawRoutePlan", None) is None
    ):
        pytest.skip("C++ raw route plan backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    h = np.array([[2.0, 0.3], [0.3, -1.0]], dtype=np.complex128)
    E = BlockTensor(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, 1, -1],
    )
    W1 = BlockTensor(
        {(q0, q0, q0, q0): h.reshape(1, 1, 2, 2)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    W2 = BlockTensor(
        {(q0, q0, q0, q0): np.ones((1, 1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    F = BlockTensor(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, 1, -1],
    )
    A = BlockTensor(
        {(q0, q0, q0, q0): np.zeros((1, 1, 2, 1), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, 1],
    )
    options = {
        "packed_local_family_flat_matvec": True,
        "packed_local_family_flat_direct_matvec": True,
        "packed_local_family_flat_direct_matvec_backend": "renormalized_table",
        "packed_local_family_flat_group_identity_csr": True,
    }
    family_envs = {"test": (E, [W1, W2], F)}
    families = object()
    operator = HamiltonianMultiplyU1(
        E,
        [W1, W2],
        F,
        complementary_operator_families=families,
        complementary_family_environments=family_envs,
        bond=0,
        matvec_options=options,
    )
    environment = MovingEnvironment(
        complementary_operator_families=families,
        matvec_options={
            "moving_environment_cpp_davidson": True,
            "moving_environment_cpp_grouped_renormalized_table": True,
            "moving_environment_cpp_grouped_factorized_table": False,
            "moving_environment_cpp_grouped_raw_table": True,
            "moving_environment_cpp_raw_payload_builder": True,
            "moving_environment_cpp_named_raw_payload_builder": False,
            "moving_environment_cpp_raw_route_plan": True,
            "generator_table_packed_route_table": "route_plan",
        },
    )
    operator._moving_environment = environment
    layout = operator._layout(A)
    vec = np.asarray([0.25, -0.5], dtype=np.complex128)

    table = environment.renormalized_operator_table(operator, A, layout)
    reused = environment.renormalized_operator_table(operator, A, layout)

    assert reused is table
    np.testing.assert_allclose(table.matvec(vec), h @ vec, atol=1.0e-12)
    stats = environment.moving_profile_stats
    assert stats["cpp_raw_route_plan_builds"] >= 1
    assert stats["cpp_raw_route_plan_refresh_calls"] >= 1
    assert stats["renormalized_operator_payload_collect_calls"] == 1


def test_moving_environment_reuses_cpp_raw_route_plan_for_direct_family():
    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "RawRoutePlan", None) is None
    ):
        pytest.skip("C++ raw route plan backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    h = np.array([[1.5, -0.4], [-0.4, 0.2]], dtype=np.complex128)
    E = BlockTensor(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, 1, -1],
    )
    W1 = BlockTensor(
        {(q0, q0, q0, q0): h.reshape(1, 1, 2, 2)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    W2 = BlockTensor(
        {(q0, q0, q0, q0): np.ones((1, 1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    F = BlockTensor(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, 1, -1],
    )
    A = BlockTensor(
        {(q0, q0, q0, q0): np.zeros((1, 1, 2, 1), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, 1],
    )
    options = {
        "packed_local_family_flat_matvec": True,
        "packed_local_family_flat_direct_matvec": True,
        "packed_local_family_flat_direct_matvec_backend": "renormalized_table",
        "packed_local_family_flat_group_local_generator_csr": True,
    }
    families = object()
    direct_envs = {"direct": ((E, [W1, W2], F),)}
    operator = HamiltonianMultiplyU1(
        E,
        [W1, W2],
        F,
        complementary_operator_families=families,
        complementary_direct_family_environments=direct_envs,
        bond=0,
        matvec_options=options,
    )
    environment = MovingEnvironment(
        complementary_operator_families=families,
        matvec_options={
            "moving_environment_cpp_davidson": True,
            "moving_environment_cpp_grouped_renormalized_table": True,
            "moving_environment_cpp_grouped_factorized_table": False,
            "moving_environment_cpp_grouped_raw_table": True,
            "moving_environment_cpp_raw_payload_builder": True,
            "moving_environment_cpp_raw_route_plan": True,
            "generator_table_packed_route_table": "route_plan",
        },
    )
    operator._moving_environment = environment
    layout = operator._layout(A)
    vec = np.asarray([0.1, -0.3], dtype=np.complex128)

    table = environment.renormalized_operator_table(operator, A, layout)
    reused = environment.renormalized_operator_table(operator, A, layout)

    assert reused is table
    np.testing.assert_allclose(table.matvec(vec), h @ vec, atol=1.0e-12)
    stats = environment.moving_profile_stats
    assert stats["cpp_raw_route_plan_builds"] >= 1
    assert stats["cpp_raw_route_plan_refresh_calls"] >= 1
    assert stats["renormalized_operator_payload_collect_calls"] >= 1


def test_moving_environment_reuses_cpp_raw_route_plan_for_packed_identity():
    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "RawRoutePlan", None) is None
    ):
        pytest.skip("C++ raw route plan backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    coeff = -0.75 + 0.0j
    E = BlockTensor(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, 1, -1],
    )
    F = BlockTensor(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, 1, -1],
    )
    W1 = BlockTensor(
        {(q0, q0, q0, q0): np.ones((1, 1, 2, 2), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    W2 = BlockTensor(
        {(q0, q0, q0, q0): np.ones((1, 1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    A = BlockTensor(
        {(q0, q0, q0, q0): np.zeros((1, 1, 2, 1), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, 1],
    )
    options = {
        "packed_local_family_flat_matvec": True,
        "packed_local_family_flat_direct_matvec": True,
        "packed_local_family_flat_direct_matvec_backend": "renormalized_table",
        "packed_local_family_flat_group_identity_csr": True,
    }
    families = object()
    direct_envs = {"identity": (AbelianPackedIdentityLocalEntry(coeff, E, F),)}
    operator = HamiltonianMultiplyU1(
        E,
        [W1, W2],
        F,
        complementary_operator_families=families,
        complementary_direct_family_environments=direct_envs,
        bond=0,
        matvec_options=options,
    )
    environment = MovingEnvironment(
        complementary_operator_families=families,
        matvec_options={
            "moving_environment_cpp_davidson": True,
            "moving_environment_cpp_grouped_renormalized_table": True,
            "moving_environment_cpp_grouped_factorized_table": False,
            "moving_environment_cpp_grouped_raw_table": True,
            "moving_environment_cpp_raw_payload_builder": True,
            "moving_environment_cpp_raw_route_plan": True,
            "generator_table_packed_route_table": "route_plan",
        },
    )
    operator._moving_environment = environment
    layout = operator._layout(A)
    vec = np.asarray([0.2, -0.6], dtype=np.complex128)

    table = environment.renormalized_operator_table(operator, A, layout)
    reused = environment.renormalized_operator_table(operator, A, layout)

    assert reused is table
    np.testing.assert_allclose(table.matvec(vec), coeff * vec, atol=1.0e-12)
    stats = environment.moving_profile_stats
    assert stats["cpp_raw_route_plan_builds"] >= 1
    assert stats["renormalized_operator_payload_collect_calls"] >= 1


def test_moving_environment_reuses_cpp_raw_route_plan_for_packed_local_generator():
    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "RawRoutePlan", None) is None
    ):
        pytest.skip("C++ raw route plan backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    coeff = 0.4 + 0.0j
    h = np.array([[0.8, 0.1], [0.1, -0.3]], dtype=np.complex128)
    E = BlockTensor(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, 1, -1],
    )
    W1 = BlockTensor(
        {(q0, q0, q0, q0): h.reshape(1, 1, 2, 2)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    W2 = BlockTensor(
        {(q0, q0, q0, q0): np.ones((1, 1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    F = BlockTensor(
        {(q0, q0, q0): np.ones((1, 1, 1), dtype=np.complex128)},
        [[q0], [q0], [q0]],
        [1, 1, -1],
    )
    A = BlockTensor(
        {(q0, q0, q0, q0): np.zeros((1, 1, 2, 1), dtype=np.complex128)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, 1],
    )
    options = {
        "packed_local_family_flat_matvec": True,
        "packed_local_family_flat_direct_matvec": True,
        "packed_local_family_flat_direct_matvec_backend": "renormalized_table",
        "packed_local_family_flat_group_local_generator_csr": True,
    }
    families = object()
    direct_envs = {
        "local": (AbelianPackedLocalGeneratorEntry(coeff, E, W1, W2, F),)
    }
    operator = HamiltonianMultiplyU1(
        E,
        [W1, W2],
        F,
        complementary_operator_families=families,
        complementary_direct_family_environments=direct_envs,
        bond=0,
        matvec_options=options,
    )
    environment = MovingEnvironment(
        complementary_operator_families=families,
        matvec_options={
            "moving_environment_cpp_davidson": True,
            "moving_environment_cpp_grouped_renormalized_table": True,
            "moving_environment_cpp_grouped_factorized_table": False,
            "moving_environment_cpp_grouped_raw_table": True,
            "moving_environment_cpp_raw_payload_builder": True,
            "moving_environment_cpp_raw_route_plan": True,
            "generator_table_packed_route_table": "route_plan",
        },
    )
    operator._moving_environment = environment
    layout = operator._layout(A)
    vec = np.asarray([0.25, -0.7], dtype=np.complex128)

    table = environment.renormalized_operator_table(operator, A, layout)
    reused = environment.renormalized_operator_table(operator, A, layout)

    assert reused is table
    np.testing.assert_allclose(table.matvec(vec), coeff * (h @ vec), atol=1.0e-12)
    stats = environment.moving_profile_stats
    assert stats["cpp_raw_route_plan_builds"] >= 1
    assert stats["renormalized_operator_payload_collect_calls"] >= 1


def test_packed_cpp_davidson_can_use_compact_plan():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    number = np.diag([0.0, 1.0])
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W0[0, 3] = 0.2 * number
    W1[0, 0] = 0.4 * number
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    options = {
        "batched_compact_matrix_chain_selector_enabled": True,
        "batched_compact_matrix_chain_force": True,
        "packed_local_davidson": True,
        "packed_local_flat_matvec": True,
        "packed_local_flat_preconditioner": True,
        "moving_environment_cpp_davidson": True,
        "moving_environment_cpp_compact_plan": True,
        "moving_environment_cpp_state_owner": True,
        "moving_environment_compact_block_table": True,
        "moving_environment_compact_block_table_max_dim": 64,
    }
    env = MovingEnvironment(matvec_options=options)
    H = env.set_bond(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        bond=0,
        matvec_options=options,
    ).local_operator()

    energy, state = H.solve_packed_davidson(AA, tol=1.0e-10, max_iter=12)

    expected = np.linalg.eigvalsh(np.array([[0.4, -1.0], [-1.0, 0.2]]))[0]
    assert energy == pytest.approx(expected, abs=1.0e-10)
    assert state.norm() == pytest.approx(1.0, abs=1.0e-10)
    stats = H.profile_stats["packed_local_davidson"]
    assert stats["cpp_davidson"] is True
    moving = H.profile_summary()["moving_environment"]
    assert moving["cpp_davidson_table_source"] == "compact_renormalized_table"
    assert moving["compact_plan_builds"] == 1
    assert moving["compact_renormalized_table_builds"] == 1
    assert (
        moving["compact_renormalized_table_diagonal_backend"]
        == "cpp_moving_environment_routes"
    )
    assert moving["compact_renormalized_table_diagonal_calls"] == 1
    assert moving["cpp_moving_environment_enabled"] is True
    assert moving["cpp_moving_environment_compact_plan_records"] == 1
    assert moving["cpp_moving_environment_compact_plan_davidson_calls"] == 1
    assert moving["compact_block_table_builds"] == 0


def test_packed_flat_compact_matvec_cython_backend_matches_blocktensor_matvec():
    if not getattr(packed_cython, "CYTHON_AVAILABLE", False):
        pytest.skip("optional packed Cython backend is not available")

    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
            "batched_compact_matrix_chain_compiled_kernel": True,
            "batched_compact_matrix_chain_cython_kernel": True,
            "packed_local_flat_matvec": True,
        },
    )
    layout = H._layout_from_map(H._safe_two_site_layout_map(AA))
    rng = np.random.default_rng(123)
    vec = rng.standard_normal(H._size(layout)).astype(np.complex128)

    flat = H._flat_batched_compact_matrix_chain(vec, AA, layout)
    reference = H._flatten(H.matvec(H._unflatten(vec, AA, layout)), layout)

    np.testing.assert_allclose(flat, reference, atol=1.0e-12)
    stats = H.profile_stats["packed_flat_batched_compact_matrix_chain"]["last"]
    assert stats["compiled_kernel_mode"] == "cython"


def test_packed_flat_projected_compact_matvec_matches_projected_blocktensor_matvec():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "batched_compact_matrix_chain_selector_enabled": True,
            "batched_compact_matrix_chain_force": True,
            "packed_local_flat_matvec": True,
        },
    )
    layout = H._layout(AA)
    rng = np.random.default_rng(321)
    vec = rng.standard_normal(H._size(layout))

    flat = H._flat_batched_compact_matrix_chain(vec, AA, layout, project_output=True)
    reference_tensor = H.matvec(H._unflatten(vec, AA, layout))
    reference = H._flatten(reference_tensor, layout)

    np.testing.assert_allclose(flat, reference, atol=1.0e-12)
    stats = H.profile_stats["packed_flat_batched_compact_matrix_chain"]["last"]
    assert stats["project_output"] is True
    assert stats["projected_plan"] is True
    assert stats["projected_output_blocks"] >= 0
    assert stats["projected_output_dim"] >= 0


def test_packed_flat_jacobi_preconditioner_matches_blocktensor_jacobi():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    ident = np.eye(2)
    number = np.diag([0.0, 1.0])
    W0 = np.zeros((1, 2, 2, 2))
    W1 = np.zeros((2, 1, 2, 2))
    W0[0, 0] = 0.3 * number
    W0[0, 1] = ident
    W1[0, 0] = ident
    W1[1, 0] = 0.7 * number
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "packed_local_flat_preconditioner": True,
        },
    )
    layout = H._layout_from_map(H._safe_two_site_layout_map(AA))
    proto = H._zero_proto_from_layout(AA, layout, float)
    rng = np.random.default_rng(456)
    resid = rng.standard_normal(H._size(layout))
    theta = -0.2

    diagonal = H._flat_jacobi_diagonal(AA, layout)
    assert H.profile_stats["preconditioner"]["backend"] == "block_data"
    assert (
        H.profile_stats.get("packed_flat_preconditioner", {}).get(
            "blocktensor_diagonal_fallbacks",
            0,
        )
        == 0
    )
    denom = theta - diagonal
    expected_flat = resid / denom
    tensor_apply = H.jacobi_preconditioner(proto)
    reference = H._flatten(tensor_apply(H._unflatten(resid, proto, layout), theta), layout)

    np.testing.assert_allclose(expected_flat, reference, atol=1.0e-12)


def test_packed_unflatten_preserves_repeated_virtual_qns():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    layout = (((q0, q1, q1, q0), (1, 2, 1, 1)),)
    proto = BlockTensor(
        {(q0, q1, q1, q0): np.ones((1, 2, 1, 1))},
        [[q0], [q1, q1], [q1], [q0]],
        [-1, 1, 1, 1],
    )

    restored = HamiltonianMultiplyU1._unflatten(np.arange(2.0), proto, layout)

    assert restored.qns[1] == [q1, q1]
    np.testing.assert_allclose(
        restored.data[(q0, q1, q1, q0)].reshape(-1),
        [0.0, 1.0],
    )


def test_packed_local_davidson_preflights_large_safe_layout_before_matvec():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_davidson_max_dim": 1,
        },
    )

    assert H.solve_packed_davidson(AA, tol=1.0e-10, max_iter=30) is None
    stats = H.profile_stats["packed_local_davidson"]
    assert stats["rejected_reason"] == "safe_layout_too_large"
    assert stats["rejected_dimension"] == 2
    assert H.profile_stats["matvec_calls"] == 0


def test_packed_local_davidson_projects_current_support_when_safe_layout_is_large():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_davidson_max_dim": 1,
            "packed_local_project_current_support": True,
        },
    )

    solution = H.solve_packed_davidson(AA, tol=1.0e-10, max_iter=4)

    assert solution is not None
    energy, state = solution
    assert float(np.real(energy)) == pytest.approx(0.0, abs=1.0e-12)
    assert set(state.data) == {(q0, q1, q1, q0)}
    stats = H.profile_stats["packed_local_davidson"]
    assert stats["converged"] is True
    assert stats["dimension"] == 1
    assert stats["safe_layout_dimension"] == 2
    assert stats["projected_current_support"] is True
    assert stats["projected_from_safe_layout_too_large"] is True
    assert stats["projected_pack_calls"] > 0
    if "projected_retained_blocks" in stats:
        assert stats["projected_original_blocks"] >= stats["projected_retained_blocks"]
    else:
        assert stats["projected_original_blocks"] == 1
    assert stats["projected_discarded_blocks"] >= 0


def test_packed_local_davidson_truncates_projected_current_support_by_block_norm():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    AA = BlockTensor(
        {
            (q0, q1, q1, q0): np.array([[[[2.0]]]]),
            (q0, q1, q0, q1): np.array([[[[0.1]]]]),
        },
        [[q0], [q1], [q0, q1], [q0, q1]],
        [-1, 1, 1, 1],
    )
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_davidson_max_dim": 1,
            "packed_local_project_current_support": True,
            "packed_local_project_current_support_truncate": True,
        },
    )

    solution = H.solve_packed_davidson(AA, tol=1.0e-10, max_iter=4)

    assert solution is not None
    _energy, state = solution
    assert set(state.data) == {(q0, q1, q1, q0)}
    stats = H.profile_stats["packed_local_davidson"]
    assert stats["dimension"] == 1
    assert stats["current_layout_dimension"] == 2
    assert stats["projected_truncated_current_support"] is True
    assert stats["projected_retained_blocks"] == 1
    assert stats["projected_retained_norm"] == pytest.approx(
        2.0 / np.sqrt(2.0**2 + 0.1**2)
    )


def test_packed_local_davidson_rejects_low_retained_projected_update():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    AA = BlockTensor(
        {
            (q0, q1, q1, q0): np.array([[[[2.0]]]]),
            (q0, q1, q0, q1): np.array([[[[0.1]]]]),
        },
        [[q0], [q1], [q0, q1], [q0, q1]],
        [-1, 1, 1, 1],
    )
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_davidson_max_dim": 1,
            "packed_local_project_current_support": True,
            "packed_local_project_current_support_truncate": True,
            "packed_local_accept_projected_unconverged": True,
            "packed_local_projected_accept_min_retained_norm": 0.999,
            "packed_local_return_current_on_rejected_projected": True,
        },
    )

    solution = H.solve_packed_davidson(AA, tol=1.0e-10, max_iter=4, current=AA)

    assert solution is not None
    _energy, state = solution
    assert set(state.data) == set(AA.data)
    stats = H.profile_stats["packed_local_davidson"]
    assert stats["returned_current_state"] is True
    assert stats["projected_accept_rejected_reasons"] == [
        "projected_retained_norm_too_small"
    ]
    assert stats["projected_retained_norm"] < 0.999


def test_packed_local_davidson_exports_capped_warm_start_candidate():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_fallback_warm_start": True,
            "packed_local_fallback_warm_start_max_dim": 2,
        },
    )

    assert H.solve_packed_davidson(AA, tol=1.0e-12, max_iter=1) is None
    candidate = H.last_packed_davidson_candidate
    stats = H.profile_stats["packed_local_davidson"]

    assert stats["rejected_reason"] == "not_converged"
    assert stats["warm_start_candidate"]["available"] is True
    assert stats["warm_start_candidate"]["dimension"] == 2
    assert candidate is not None
    assert candidate.norm() == pytest.approx(1.0, abs=1.0e-12)
    assert H.last_packed_davidson_candidate_flat is not None
    assert H.last_packed_davidson_candidate_layout is not None
    assert np.linalg.norm(H.last_packed_davidson_candidate_flat) == pytest.approx(
        1.0,
        abs=1.0e-12,
    )
    assert set(candidate.data) == {
        (q0, q1, q0, q1),
        (q0, q1, q1, q0),
    }


def test_packed_local_davidson_accepts_layout_matched_flat_initial_guess():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate([(-cd @ parity, c), (-parity @ c, cd)], start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    init = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    AA = tensordot(init[0], init[1], axes=([1], [0])).transpose(0, 2, 1, 3)
    H = HamiltonianMultiplyU1(
        initial_E(mpo[0]),
        mpo,
        initial_F(mpo[1], target_qn=q1),
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_safe_layout_expansion": False,
        },
    )
    layout = H._layout(AA)
    flat = H._flatten(AA, layout)

    H.solve_packed_davidson(
        AA,
        tol=1.0e-12,
        max_iter=1,
        initial_flat=flat,
        initial_layout=layout,
        return_flat=True,
    )

    stats = H.profile_stats["packed_local_davidson"]
    assert stats["initial_flat_guess_present"] is True
    assert stats["initial_flat_guess_used"] is True

    H.solve_packed_davidson(
        AA,
        tol=1.0e-12,
        max_iter=1,
        initial_flat=np.asarray(flat, dtype=np.complex128),
        initial_layout=layout,
        initial_is_current=True,
        return_flat=True,
    )

    stats = H.profile_stats["packed_local_davidson"]
    assert stats["initial_current_flat_present"] is True
    assert stats["initial_current_flat_used"] is True
    assert stats["initial_current_flat_nocopy_used"] is True


def test_packed_local_davidson_block_preconditioner_closed_layout():
    q0 = AbelianSector(("charge",), (0,))
    h = np.array([[2.0, 0.3], [0.3, -1.0]])
    E = BlockTensor({(q0, q0, q0): np.ones((1, 1, 1))}, [[q0], [q0], [q0]], [1, 1, -1])
    W1 = BlockTensor(
        {(q0, q0, q0, q0): h.reshape(1, 1, 2, 2)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    W2 = BlockTensor(
        {(q0, q0, q0, q0): np.ones((1, 1, 1, 1))},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    F = BlockTensor({(q0, q0, q0): np.ones((1, 1, 1))}, [[q0], [q0], [q0]], [1, 1, -1])
    A = BlockTensor(
        {(q0, q0, q0, q0): np.array([1.0, 0.2]).reshape(1, 1, 2, 1)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, 1],
    )
    H_plain = HamiltonianMultiplyU1(
        E,
        [W1, W2],
        F,
        matvec_options={"packed_local_davidson": True},
    )
    energy, state = H_plain.solve_packed_davidson(A, tol=1.0e-10, max_iter=20)

    assert energy == pytest.approx(np.linalg.eigvalsh(h)[0], abs=1.0e-10)
    assert state.norm() == pytest.approx(1.0, abs=1.0e-10)
    assert H_plain.profile_stats["packed_local_davidson"]["converged"] is True
    assert H_plain.last_packed_davidson_solution_flat is not None
    assert H_plain.last_packed_davidson_solution_converged is True
    assert np.linalg.norm(H_plain.last_packed_davidson_solution_flat) == pytest.approx(
        1.0,
        abs=1.0e-10,
    )

    H_block = HamiltonianMultiplyU1(
        E,
        [W1, W2],
        F,
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_block_preconditioner": True,
            "packed_local_block_preconditioner_max_block_dim": 4,
            "packed_local_block_preconditioner_max_total_dim": 4,
            "packed_local_davidson_restart_dim": 4,
        },
    )
    blocked = H_block.solve_packed_davidson(A, tol=1.0e-10, max_iter=20)

    assert blocked is None
    stats = H_block.profile_stats["packed_local_davidson"]
    assert stats["converged"] is False
    assert stats["rejected_reason"] == "not_converged"
    assert stats["block_preconditioner"]["last_blocks"] == 1


def test_packed_local_davidson_accepts_unconverged_best_vector_when_requested():
    q0 = AbelianSector(("charge",), (0,))
    h = np.array([[2.0, 0.3], [0.3, -1.0]])
    E = BlockTensor({(q0, q0, q0): np.ones((1, 1, 1))}, [[q0], [q0], [q0]], [1, 1, -1])
    W1 = BlockTensor(
        {(q0, q0, q0, q0): h.reshape(1, 1, 2, 2)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    W2 = BlockTensor(
        {(q0, q0, q0, q0): np.ones((1, 1, 1, 1))},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, -1],
    )
    F = BlockTensor({(q0, q0, q0): np.ones((1, 1, 1))}, [[q0], [q0], [q0]], [1, 1, -1])
    A = BlockTensor(
        {(q0, q0, q0, q0): np.array([1.0, 0.0]).reshape(1, 1, 2, 1)},
        [[q0], [q0], [q0], [q0]],
        [1, -1, 1, 1],
    )
    H = HamiltonianMultiplyU1(
        E,
        [W1, W2],
        F,
        matvec_options={
            "packed_local_davidson": True,
            "packed_local_accept_unconverged": True,
        },
    )

    energy, state = H.solve_packed_davidson(A, tol=1.0e-14, max_iter=1)

    assert float(np.real(energy)) == pytest.approx(2.0, abs=1.0e-12)
    assert state.norm() == pytest.approx(1.0, abs=1.0e-12)
    stats = H.profile_stats["packed_local_davidson"]
    assert stats["converged"] is False
    assert stats["accepted_unconverged"] is True
    assert stats["accepted_reason"] == "packed_local_davidson"
    assert H.last_packed_davidson_solution_flat is not None
    assert H.last_packed_davidson_solution_converged is False
    assert np.linalg.norm(H.last_packed_davidson_solution_flat) == pytest.approx(
        1.0,
        abs=1.0e-12,
    )


def test_abelian_dmrg_accepts_entangled_dense_initial_guess():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(terms, start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    mpo = dense_to_symmetric_mpo([W0, W1], site_qn_maps)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    A0 = np.zeros((1, 2, 2), dtype=complex)
    A1 = np.zeros((2, 2, 1), dtype=complex)
    A0[0, 0, 0] = 1.0
    A0[0, 1, 1] = 1.0
    A1[0, 1, 0] = inv_sqrt2
    A1[1, 0, 0] = inv_sqrt2

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=4,
        init_guess=[A0, A1],
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
    ).run()

    assert dmrg.e_tot == pytest.approx(-1.0, abs=1.0e-8)


def test_abelian_dmrg_reports_post_truncation_state_energy():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    site_qn_maps = [{0: q0, 1: q1}, {0: q0, 1: q1}]

    c = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = c.T
    parity = np.diag([1.0, -1.0])
    ident = np.eye(2)
    terms = [(-cd @ parity, c), (-parity @ c, cd)]
    W0 = np.zeros((1, 4, 2, 2))
    W1 = np.zeros((4, 1, 2, 2))
    W0[0, 0] = ident
    W1[3, 0] = ident
    for channel, (left, right) in enumerate(terms, start=1):
        W0[0, channel] = left
        W1[channel, 0] = right
    dense_mpo = [W0, W1]
    mpo = dense_to_symmetric_mpo(dense_mpo, site_qn_maps)

    psi0 = [np.array([0.0, 1.0]).reshape(1, 2, 1), np.array([1.0, 0.0]).reshape(1, 2, 1)]
    init = dense_to_symmetric(psi0, phys_qns=[q0, q1])

    sym_mgr = SymmetryManager(["charge"])
    dmrg = DMRG(
        mpo,
        D=1,
        init_guess=init,
        nsweeps=2,
        symmetry=True,
        target_qn=sym_mgr.get_target_qn(1),
        sym_mgr=sym_mgr,
        not_conv_err=False,
        davidson_tol=1.0e-10,
        davidson_max_iter=20,
        noise=0.0,
        site_qn_maps=site_qn_maps,
    ).run()

    dense = symmetric_to_dense(dmrg.ground_state, site_qn_maps=site_qn_maps).to_order(["lv", "p", "rv"])
    state_energy = _dense_mps_expectation_mpo(dense.factors, dense_mpo)

    assert dmrg.sweep_history[-1]["energy"] == pytest.approx(-1.0, abs=1.0e-10)
    assert not hasattr(dmrg, "local_e_tot")
    assert state_energy == pytest.approx(0.0, abs=1.0e-10)
    assert dmrg.e_tot == pytest.approx(state_energy, abs=1.0e-10)


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
