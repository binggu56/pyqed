import numpy as np
import pytest

import pyqed.mps.nonabelian.environment as env_mod
import pyqed.mps.nonabelian.solver as solver_mod
import pyqed.mps.nonabelian.sweep as sweep_mod
import pyqed.mps.nonabelian.update as update_mod
from pyqed.mps.nonabelian import (
    MPO,
    PhysicalLeg,
    SiteOperator,
    AutoMPO,
    identity_operator,
    MPS,
    FusionLeg,
    FusionEdge,
    FusionPipe,
    FusionPipeEntry,
    NonabelianTensor,
    tensordot,
    merge_mps_sites,
    combine_legs,
    split_legs,
    svd_two_site,
    left_canonicalize_sites,
    left_canonical_error,
    mixed_canonicalize_sites,
    right_canonical_error,
    right_canonicalize_sites,
    ReducedProjectedSector,
    truncate_reduced_svds,
    LocalOperator,
    TwoSiteEffectiveH,
    ReducedStateLayout,
    ReducedStateVector,
    PackedBlockPreconditioner,
    pack_two_site_state,
    unpack_two_site_state,
    solve_local_two_site,
    DenseEnvironmentChain,
    DenseEnvironmentSweep,
    BlockSparseEnvironmentChain,
    BlockSparseEnvironmentSweep,
    build_dense_bond_operator,
    build_block_sparse_bond_operator,
    build_random_spatial_mps,
    build_random_reduced_spatial_mps,
    build_product_state,
    build_product_spatial_mps,
    build_reduced_product_spatial_mps,
    build_spatial_hubbard_mpo,
    contract_chain_expectation,
    half_filled_singlet_sector,
    mixed_cg_canonical_errors,
    reduced_physical_leg_from_spatial_orbital,
    spatial_identity,
    spatial_number,
    two_site_update,
    sweep_once,
    run_sweeps,
    SweepDriver,
)
from pyqed.mps.symmetry import Sector
from pyqed.mps.su2 import SU2Irrep
from pyqed.mps.nonabelian.update import _expand_two_site_support
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


def _charge_spin_sector(charge, two_j):
    return Sector(("charge", "su2"), (charge, SU2Irrep(two_j)))


def test_nonabelian_tensor_accepts_charge_su2_sector_labels():
    vac = _charge_spin_sector(0, 0)
    dbl = _charge_spin_sector(1, 1)

    tensor = NonabelianTensor(
        data={
            (vac, dbl): np.array([[1.0], [2.0]]),
        },
        qns=[[vac], [dbl]],
        dirs=[-1, 1],
    )

    assert tensor.rank == 2
    assert tensor.shape == (1, 1)
    assert tensor.has_nonabelian_symmetry is True
    assert (vac, dbl) in tensor.data


def test_fusion_leg_exposes_richer_per_leg_metadata():
    left = _charge_spin_sector(0, 0)
    mid = _charge_spin_sector(1, 1)
    leg = FusionLeg(
        child_legs=(0, 2),
        sectors=(mid,),
        orientation=-1,
        coupling="left_associative",
        fusion_map=(((left, mid), mid),),
        selected_channel=mid,
    )

    assert leg.child_legs == (0, 2)
    assert leg.parents == (0, 2)
    assert leg.sectors == (mid,)
    assert leg.orientation == -1
    assert leg.channel == mid


def test_fusion_edge_alias_still_constructs_fusion_leg():
    mid = _charge_spin_sector(1, 1)
    edge = FusionEdge((0, 1), channel=mid)

    assert isinstance(edge, FusionLeg)
    assert edge.parents == (0, 1)
    assert edge.channel == mid


def test_fusion_leg_from_children_builds_first_fusion_map():
    vac = _charge_spin_sector(0, 0)
    spin = _charge_spin_sector(1, 1)

    leg = FusionLeg.from_children(
        child_legs=(0, 1),
        child_sector_lists=((vac, spin), (vac, spin)),
        child_dirs=(-1, 1),
        orientation=1,
    )

    assert leg.child_legs == (0, 1)
    assert leg.child_dirs == (-1, 1)
    assert vac in leg.sectors
    assert spin in leg.sectors
    assert leg.resolve_sector((vac, vac)) == vac
    assert leg.resolve_sector((vac, spin)) == spin
    assert leg.slot_for((vac, spin), spin) == 0


def test_fusion_pipe_tracks_sector_local_packing_layout():
    vac = _charge_spin_sector(0, 0)
    spin = _charge_spin_sector(1, 1)
    pipe = FusionPipe.from_entries(
        child_legs=(0, 1),
        child_sector_lists=((vac, spin), (vac, spin)),
        child_dirs=(-1, 1),
        fused_sectors=(vac, spin),
        entries=(
            FusionPipeEntry((vac, vac), vac, 0, 0, 1, (1, 1)),
            FusionPipeEntry((vac, spin), spin, 0, 0, 2, (1, 2)),
        ),
        orientation=1,
    )

    assert pipe.entry_for_child_sectors((vac, vac)).fused_sector == vac
    assert pipe.entry_for_child_sectors((vac, spin), spin).selected_shape == (1, 2)
    assert pipe.total_dim(spin) == 2


def test_nonabelian_tensor_transpose_and_conjugation_preserve_metadata():
    left = _charge_spin_sector(0, 0)
    right = _charge_spin_sector(1, 1)
    edge = FusionLeg(child_legs=(0, 1), selected_channel=right)
    block = np.array([[1.0 + 2.0j], [3.0 + 4.0j]])

    tensor = NonabelianTensor(
        data={(left, right): block},
        qns=[[left], [right]],
        dirs=[-1, 1],
        fusion_edges=[None, edge],
        metadata={"site_type": "spatial_orbital"},
    )

    transposed = tensor.transpose(1, 0)
    conj = tensor.conj()

    assert transposed.dirs == [1, -1]
    assert transposed.fusion_edges == [edge, None]
    np.testing.assert_allclose(transposed.data[(right, left)], block.T)

    assert conj.dirs == [1, -1]
    np.testing.assert_allclose(conj.data[(left, right)], block.conj())
    assert conj.metadata["site_type"] == "spatial_orbital"


def test_nonabelian_tensor_addition_requires_matching_metadata():
    left = _charge_spin_sector(0, 0)
    right = _charge_spin_sector(1, 1)

    a = NonabelianTensor(
        data={(left, right): np.array([[1.0], [0.0]])},
        qns=[[left], [right]],
        dirs=[-1, 1],
    )
    b = NonabelianTensor(
        data={(left, right): np.array([[0.0], [1.0]])},
        qns=[[left], [right]],
        dirs=[-1, 1],
    )
    c = a + b

    np.testing.assert_allclose(c.data[(left, right)], np.array([[1.0], [1.0]]))

    mismatch = NonabelianTensor(
        data={(left, right): np.array([[1.0], [1.0]])},
        qns=[[left], [right]],
        dirs=[1, -1],
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        _ = a + mismatch


def test_nonabelian_tensor_rejects_unknown_sector_keys():
    left = _charge_spin_sector(0, 0)
    right = _charge_spin_sector(1, 1)
    rogue = _charge_spin_sector(2, 0)

    with pytest.raises(ValueError, match="not present in declared leg sectors"):
        NonabelianTensor(
            data={(left, rogue): np.array([[1.0]])},
            qns=[[left], [right]],
            dirs=[-1, 1],
        )


def test_tensordot_nonabelian_contracts_fixed_layout_blocks():
    left = _charge_spin_sector(0, 0)
    mid = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    fusion = FusionLeg(child_legs=(0, 1), selected_channel=mid)

    A = NonabelianTensor(
        data={(left, mid): np.array([[1.0, 2.0], [3.0, 4.0]])},
        qns=[[left], [mid]],
        dirs=[-1, 1],
        fusion_legs=[None, fusion],
        metadata={"layout": "fixed_tree"},
    )
    B = NonabelianTensor(
        data={(mid, right): np.array([[5.0, 6.0], [7.0, 8.0]])},
        qns=[[mid], [right]],
        dirs=[-1, 1],
        fusion_legs=[fusion, None],
        metadata={"layout": "fixed_tree"},
    )

    C = tensordot(A, B, axes=([1], [0]))

    assert C.qns == [[left], [right]]
    assert C.dirs == [-1, 1]
    assert C.fusion_edges == [None, None]
    assert C.metadata["layout"] == "fixed_tree"
    assert C.metadata["contracted_channels"][(left, right)] == (mid,)
    np.testing.assert_allclose(
        C.data[(left, right)],
        np.array([[19.0, 22.0], [43.0, 50.0]]),
    )


def test_tensordot_nonabelian_rejects_incompatible_fusion_trees():
    left = _charge_spin_sector(0, 0)
    mid = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)

    A = NonabelianTensor(
        data={(left, mid): np.array([[1.0]])},
        qns=[[left], [mid]],
        dirs=[-1, 1],
        fusion_legs=[None, FusionLeg(child_legs=(0, 1), selected_channel=mid)],
    )
    B = NonabelianTensor(
        data={(mid, right): np.array([[2.0]])},
        qns=[[mid], [right]],
        dirs=[-1, 1],
        fusion_legs=[FusionLeg(child_legs=(9, 10), selected_channel=mid), None],
    )

    with pytest.raises(ValueError, match="matching fixed fusion-tree metadata"):
        tensordot(A, B, axes=([1], [0]))


def test_tensordot_nonabelian_requires_opposite_leg_directions():
    left = _charge_spin_sector(0, 0)
    mid = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)

    A = NonabelianTensor(
        data={(left, mid): np.array([[1.0]])},
        qns=[[left], [mid]],
        dirs=[-1, 1],
    )
    B = NonabelianTensor(
        data={(mid, right): np.array([[2.0]])},
        qns=[[mid], [right]],
        dirs=[1, 1],
    )

    with pytest.raises(ValueError, match="opposite leg directions"):
        tensordot(A, B, axes=([1], [0]))


def test_merge_and_svd_two_site_nonabelian_round_trip():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 2.0], [3.0, 4.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[5.0], [6.0]], [[7.0], [8.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    AA = merge_mps_sites(A, B)
    assert AA.metadata["contracted_fusion_leg"].pipe is not None
    A_new, B_new, singular_values, trunc_err, kept = svd_two_site(
        AA, absorb="right"
    )
    AA_rebuilt = merge_mps_sites(A_new, B_new)

    assert AA.qns == [[left], [phys_left], [phys_right], [right]]
    assert kept == 2
    assert trunc_err == pytest.approx(0.0)
    assert bond in singular_values
    assert A_new.fusion_legs[2].pipe is not None
    assert B_new.fusion_legs[0].pipe is not None
    np.testing.assert_allclose(
        AA_rebuilt.data[(left, phys_left, phys_right, right)],
        AA.data[(left, phys_left, phys_right, right)],
    )


def test_svd_two_site_respects_requested_bond_coupling():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 2.0], [3.0, 4.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[5.0], [6.0]], [[7.0], [8.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    A_new, B_new, _, _, _ = svd_two_site(merged, absorb="right", bond_coupling="right")

    assert A_new.fusion_legs[2].pipe.coupling == "right"
    assert B_new.fusion_legs[0].pipe.coupling == "right"


def test_svd_two_site_handles_multi_channel_reduced_bases(monkeypatch):
    import pyqed.mps.nonabelian.decompose as decompose_module
    from pyqed.mps.nonabelian import CouplingChannel, ReducedBondSpace

    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    key = (left, phys_left, phys_right, right)

    merged = NonabelianTensor(
        data={
            key: np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 2, 2, 1),
        },
        qns=[[left], [phys_left], [phys_right], [right]],
        dirs=[-1, 1, 1, 1],
        metadata={
            "contracted_fusion_leg": FusionLeg(
                child_legs=(0, 1, 2, 3),
                child_sector_lists=((left,), (phys_left,), (phys_right,), (right,)),
                child_dirs=(-1, 1, 1, 1),
                sectors=(bond,),
                orientation=1,
                coupling="contracted",
                pipe=FusionPipe.from_entries(
                    child_legs=(0, 1, 2, 3),
                    child_sector_lists=((left,), (phys_left,), (phys_right,), (right,)),
                    child_dirs=(-1, 1, 1, 1),
                    fused_sectors=(bond,),
                    entries=(
                        FusionPipeEntry(
                            key,
                            bond,
                            0,
                            0,
                            4,
                            (1, 2, 2, 1),
                        ),
                    ),
                    orientation=1,
                    coupling="contracted",
                ),
            ),
        },
    )

    def fake_reduced_bond_space(child_sectors, fused_sector, scheme="left"):
        child_sectors = tuple(child_sectors)
        channels = (
            CouplingChannel(
                child_sectors=child_sectors,
                fused_sector=fused_sector,
                intermediate_sectors=(fused_sector,),
                slot=0,
            ),
            CouplingChannel(
                child_sectors=child_sectors,
                fused_sector=fused_sector,
                intermediate_sectors=(fused_sector,),
                slot=1,
            ),
        )
        basis_matrices = (
            np.array([[1.0], [0.0]]),
            np.array([[0.0], [1.0]]),
        )
        return ReducedBondSpace(
            child_sectors=child_sectors,
            fused_sector=fused_sector,
            scheme=scheme,
            channels=channels,
            basis_matrices=basis_matrices,
        )

    monkeypatch.setattr(decompose_module, "reduced_bond_space", fake_reduced_bond_space)

    left_new, right_new, _, trunc_err, kept = svd_two_site(merged, absorb="right")
    rebuilt = merge_mps_sites(left_new, right_new)

    assert kept == 2
    assert trunc_err == pytest.approx(0.0)
    np.testing.assert_allclose(rebuilt.data[key], merged.data[key], atol=1e-12)


def test_two_site_update_reuses_current_merged_tensor_by_default():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 2.0], [3.0, 4.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[5.0], [6.0]], [[7.0], [8.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    update = two_site_update(A, B, absorb="right")
    rebuilt = merge_mps_sites(update["left"], update["right"])

    assert update["kept"] == 2
    assert update["optimized"] is update["merged"]
    np.testing.assert_allclose(
        rebuilt.data[(left, phys_left, phys_right, right)],
        update["merged"].data[(left, phys_left, phys_right, right)],
    )


def test_two_site_update_threads_bond_coupling_into_svd():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 2.0], [3.0, 4.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[5.0], [6.0]], [[7.0], [8.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    update = two_site_update(A, B, absorb="right", bond_coupling="right")

    assert update["left"].fusion_legs[2].pipe.coupling == "right"
    assert update["right"].fusion_legs[0].pipe.coupling == "right"


def test_two_site_update_accepts_solver_callback():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    def solver(merged):
        scaled = {
            key: 2.0 * block
            for key, block in merged.data.items()
        }
        return NonabelianTensor(
            scaled,
            [leg[:] for leg in merged.qns],
            merged.dirs[:],
            fusion_legs=merged.fusion_legs[:],
            metadata=merged.metadata.copy(),
        )

    update = two_site_update(A, B, solver=solver, absorb="right")
    rebuilt = merge_mps_sites(update["left"], update["right"])

    np.testing.assert_allclose(
        rebuilt.data[(left, phys_left, phys_right, right)],
        update["optimized"].data[(left, phys_left, phys_right, right)],
    )


def test_solve_local_two_site_with_explicit_matrix_reports_energy_and_residual():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    vec, _ = pack_two_site_state(merged)
    matrix = np.diag(np.arange(vec.size, dtype=float))

    optimized, objective = solve_local_two_site(
        merged,
        matrix,
        tol=1e-10,
        itermax=50,
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["davidson_converged"] is True
    assert objective["residual"] < 1e-6


def test_two_site_update_captures_local_objective_payload():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    def solver(merged):
        return {
            "optimized": merged,
            "energy": -1.25,
            "metric": 0.03,
        }

    update = two_site_update(A, B, solver=solver, absorb="right")

    assert update["local_objective"]["energy"] == pytest.approx(-1.25)
    assert update["local_objective"]["metric"] == pytest.approx(0.03)


def test_two_site_update_accepts_local_operator():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    vec, _ = pack_two_site_state(merged)
    operator = LocalOperator(matrix=np.diag(np.arange(vec.size, dtype=float)))

    update = two_site_update(
        A,
        B,
        local_operator=operator,
        local_solver_kwargs={"tol": 1e-10, "itermax": 50},
        absorb="right",
    )

    assert update["local_objective"]["energy"] == pytest.approx(0.0)
    assert update["local_objective"]["davidson_converged"] is True


def test_two_site_update_can_prefer_aux_reduced_local_operator():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    vec, layout = pack_two_site_state(merged)
    diagonal = np.arange(vec.size, dtype=float)
    state_layout = ReducedStateLayout(tuple(layout))

    def tensor_matvec(_tensor):
        raise AssertionError("tensor_matvec should not be used when prefer_reduced_local_operator=True")

    def reduced_matvec(state):
        return ReducedStateVector(
            layout=state.layout,
            blocks={
                entry.key: diagonal[entry.offset:entry.offset + entry.size].reshape(entry.shape)
                * np.asarray(state.blocks.get(entry.key, np.zeros(entry.shape)))
                for entry in state_layout.entries
            },
        )

    operator = LocalOperator(
        tensor_matvec=tensor_matvec,
        aux_reduced_matvec=reduced_matvec,
        diag=diagonal,
        name="diag-with-aux-reduced",
    )

    update = two_site_update(
        A,
        B,
        local_operator=operator,
        local_solver_kwargs={"tol": 1e-10, "itermax": 50},
        prefer_reduced_local_operator=True,
        absorb="right",
    )

    assert update["local_objective"]["energy"] == pytest.approx(0.0)
    assert update["local_objective"]["operator_representation"] == "reduced"


def test_solve_local_two_site_coupled_auto_can_use_aux_reduced_operator():
    sites = build_random_spatial_mps(
        nsites=2,
        target_sector=half_filled_singlet_sector(2),
        bond_multiplicity=2,
        seed=3,
        scale=0.2,
    )
    merged = merge_mps_sites(sites[0], sites[1])
    _, layout = pack_two_site_state(merged)
    state_layout = ReducedStateLayout(tuple(layout))
    diagonal = np.arange(state_layout.size, dtype=float)

    def tensor_matvec(_tensor):
        raise AssertionError("tensor_matvec should not be used when aux reduced path is available")

    def reduced_matvec(state):
        return ReducedStateVector(
            layout=state.layout,
            blocks={
                entry.key: diagonal[entry.offset:entry.offset + entry.size].reshape(entry.shape)
                * np.asarray(state.blocks.get(entry.key, np.zeros(entry.shape)))
                for entry in state_layout.entries
            },
        )

    operator = LocalOperator(
        tensor_matvec=tensor_matvec,
        aux_reduced_matvec=reduced_matvec,
        diag=diagonal,
        name="diag-with-aux-reduced",
    )

    optimized, objective = solve_local_two_site(
        merged,
        operator,
        tol=1e-10,
        itermax=50,
        couple_physical="auto",
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["operator_representation"] == "reduced"
    assert objective["coupled_physical_used"] is True


def test_solve_local_two_site_uses_dense_fallback_for_small_problem():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    vec, _ = pack_two_site_state(merged)
    diagonal = np.arange(vec.size, dtype=float)
    operator = LocalOperator(matvec=lambda x: diagonal * x, diag=diagonal)

    optimized, objective = solve_local_two_site(
        merged,
        operator,
        tol=1e-12,
        itermax=0,
        dense_fallback_dim=64,
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["dense_fallback"] is True


def test_solve_local_two_site_keeps_generalized_norm_when_dense_cap_is_exceeded():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    vec, _ = pack_two_site_state(merged)
    matrix = np.diag(np.arange(vec.size, dtype=float))
    norm = np.eye(vec.size, dtype=float)

    optimized, objective = solve_local_two_site(
        merged,
        matrix,
        norm_operator=norm,
        tol=1e-10,
        itermax=50,
        dense_fallback_dim=1,
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["generalized_norm"] is True
    assert "generalized_norm_skipped" not in objective
    assert objective["dense_fallback"] is True


def test_solve_local_two_site_tensor_generalized_norm_uses_tensor_davidson():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    diagonal = np.arange(pack_two_site_state(merged)[0].size, dtype=float)
    norm_diag = np.ones_like(diagonal)

    def _diag_tensor_matvec(diag):
        def apply(tensor):
            vec, layout = pack_two_site_state(tensor)
            return unpack_two_site_state(diag * vec, tensor, layout=layout)
        return apply

    operator = LocalOperator(
        tensor_matvec=_diag_tensor_matvec(diagonal),
        diag=diagonal,
    )
    norm_operator = LocalOperator(
        tensor_matvec=_diag_tensor_matvec(norm_diag),
        diag=norm_diag,
    )

    optimized, objective = solve_local_two_site(
        merged,
        operator,
        norm_operator=norm_operator,
        tol=1e-10,
        itermax=20,
        dense_fallback_dim=1,
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["generalized_norm"] is True
    assert objective["tensor_davidson"] is True
    assert objective["reduced_krylov"] is True
    assert objective["preconditioner_mode"] == "reduced_diagonal"


def test_packed_block_preconditioner_solves_per_block_systems():
    layout = (
        solver_mod.PackedEntry(("a",), (2,), 0, 2),
        solver_mod.PackedEntry(("b",), (1,), 2, 1),
    )
    preconditioner = PackedBlockPreconditioner.from_layout_blocks(
        layout,
        h_blocks=(
            np.array([[2.0, 0.5], [0.5, 1.0]]),
            np.array([[3.0]]),
        ),
        n_blocks=(
            np.eye(2),
            np.array([[2.0]]),
        ),
    )
    resid = np.array([1.0, -2.0, 4.0], dtype=float)
    theta = 5.0

    corrected = preconditioner.apply(resid, theta)

    expected0 = np.linalg.solve(theta * np.eye(2) - np.array([[2.0, 0.5], [0.5, 1.0]]) + 1e-10 * np.eye(2), resid[:2])
    expected1 = np.linalg.solve(theta * np.array([[2.0]]) - np.array([[3.0]]) + 1e-10 * np.eye(1), resid[2:])
    np.testing.assert_allclose(corrected[:2], expected0)
    np.testing.assert_allclose(corrected[2:], expected1)


def test_solve_local_two_site_accepts_reduced_local_operator():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    diagonal = np.arange(pack_two_site_state(merged)[0].size, dtype=float)
    norm_diag = np.ones_like(diagonal)

    def _diag_reduced_matvec(diag):
        cache = {}

        def apply(state):
            if state.layout not in cache:
                cache[state.layout] = {
                    entry.key: diag[entry.offset:entry.offset + entry.size].reshape(entry.shape)
                    for entry in state.layout.entries
                }
            diag_blocks = cache[state.layout]
            blocks = {
                key: np.asarray(block) * diag_blocks[key]
                for key, block in state.blocks.items()
            }
            return ReducedStateVector(layout=state.layout, blocks=blocks)

        return apply

    operator = LocalOperator(
        reduced_matvec=_diag_reduced_matvec(diagonal),
        diag=diagonal,
    )
    norm_operator = LocalOperator(
        reduced_matvec=_diag_reduced_matvec(norm_diag),
        diag=norm_diag,
    )

    optimized, objective = solve_local_two_site(
        merged,
        operator,
        norm_operator=norm_operator,
        tol=1e-10,
        itermax=20,
        dense_fallback_dim=1,
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["operator_representation"] == "reduced"
    assert objective["norm_operator_representation"] == "reduced"
    assert objective["reduced_krylov"] is True
    assert objective["dense_fallback"] is False


def test_solve_local_two_site_effective_h_can_skip_identity_norm_operator():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    diagonal = np.arange(pack_two_site_state(merged)[0].size, dtype=float)

    def _diag_reduced_matvec(diag):
        cache = {}

        def apply(state):
            if state.layout not in cache:
                cache[state.layout] = {
                    entry.key: diag[entry.offset:entry.offset + entry.size].reshape(entry.shape)
                    for entry in state.layout.entries
                }
            diag_blocks = cache[state.layout]
            blocks = {
                key: np.asarray(block) * diag_blocks[key]
                for key, block in state.blocks.items()
            }
            return ReducedStateVector(layout=state.layout, blocks=blocks)

        return apply

    operator = LocalOperator(
        reduced_matvec=_diag_reduced_matvec(diagonal),
        diag=diagonal,
    )
    norm_operator = LocalOperator(
        reduced_matvec=_diag_reduced_matvec(np.ones_like(diagonal)),
        diag=np.ones_like(diagonal),
        identity_like=True,
    )

    optimized, objective = solve_local_two_site(
        merged,
        TwoSiteEffectiveH(
            operator=operator,
            norm_operator=norm_operator,
            canonical_norm=True,
        ),
        tol=1e-10,
        itermax=20,
        dense_fallback_dim=1,
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["canonical_norm"] is True
    assert objective["effective_local_problem"] == "standard"
    assert objective["operator_representation"] == "reduced"
    assert "norm_operator_representation" not in objective
    assert objective.get("generalized_norm", False) is False


def test_solve_local_two_site_can_optimize_in_cg_coupled_basis():
    left = _charge_spin_sector(0, 0)
    phys_left = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(0, 0)
    phys_right = _charge_spin_sector(1, 1)
    singlet_vec = np.array([0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0])
    singlet_block = singlet_vec.reshape(1, 2, 2, 1)

    merged = NonabelianTensor(
        data={(left, phys_left, phys_right, right): singlet_block},
        qns=[[left], [phys_left], [phys_right], [right]],
        dirs=[-1, 1, 1, 1],
    )
    matrix = -np.outer(singlet_vec, singlet_vec)

    optimized_uncoupled, objective_uncoupled = solve_local_two_site(
        merged,
        matrix,
        tol=1e-10,
        itermax=50,
    )
    optimized_coupled, objective_coupled = solve_local_two_site(
        merged,
        matrix,
        tol=1e-10,
        itermax=50,
        couple_physical=True,
    )

    assert optimized_coupled.rank == 4
    assert objective_coupled["coupled_physical"] is True
    assert objective_coupled["coupled_physical_used"] is True
    assert objective_coupled["tensor_davidson"] is True
    assert objective_coupled["reduced_krylov"] is True
    assert objective_coupled["preconditioner_mode"] == "reduced_diagonal"
    assert objective_coupled["reduced_preconditioner"] is True
    assert objective_coupled["dense_fallback"] is False
    assert objective_coupled["energy"] == pytest.approx(objective_uncoupled["energy"])
    np.testing.assert_allclose(
        optimized_coupled.data[(left, phys_left, phys_right, right)],
        singlet_block,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized_uncoupled.data[(left, phys_left, phys_right, right)],
        singlet_block,
        atol=1e-12,
    )


def test_solve_local_two_site_auto_coupling_falls_back_when_cg_basis_is_not_available():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)
    bond_leg = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, bond, phys_left): np.array([[[1.0, 0.0], [0.0, 1.0]]])},
        qns=[[left], [bond], [phys_left]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, bond_leg, None],
    )
    B = NonabelianTensor(
        data={(bond, right, phys_right): np.array([[[1.0], [0.0]], [[0.0], [1.0]]])},
        qns=[[bond], [right], [phys_right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg, None, None],
    )

    merged = merge_mps_sites(A, B)
    vec, _ = pack_two_site_state(merged)
    matrix = np.diag(np.arange(vec.size, dtype=float))

    optimized, objective = solve_local_two_site(
        merged,
        matrix,
        tol=1e-10,
        itermax=50,
        couple_physical="auto",
    )

    assert optimized.rank == 4
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["coupled_physical_used"] is False
    assert "coupled_physical_skipped" in objective


def _assert_same_tensor(a, b):
    assert a.qns == b.qns
    assert a.dirs == b.dirs
    assert a.fusion_legs == b.fusion_legs
    assert set(a.data) == set(b.data)
    for key in a.data:
        np.testing.assert_allclose(a.data[key], b.data[key], atol=1e-12, rtol=1e-12)


def _three_site_chain():
    left = _charge_spin_sector(0, 0)
    bond1 = _charge_spin_sector(1, 1)
    bond2 = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys1 = _charge_spin_sector(1, 1)
    phys2 = _charge_spin_sector(1, 1)
    phys3 = _charge_spin_sector(1, 1)

    bond_leg_01 = FusionLeg(child_legs=(0, 2))
    bond_leg_12 = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={(left, phys1, bond1): np.array([[[1.0, 3.0], [2.0, 4.0]]])},
        qns=[[left], [phys1], [bond1]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, None, bond_leg_01],
    )
    B = NonabelianTensor(
        data={
            (bond1, phys2, bond2): np.array(
                [
                    [[1.0, 0.5], [0.0, 1.5]],
                    [[0.0, 1.0], [1.0, 0.5]],
                ]
            )
        },
        qns=[[bond1], [phys2], [bond2]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg_01, None, bond_leg_12],
    )
    C = NonabelianTensor(
        data={
            (bond2, phys3, right): np.array(
                [
                    [[2.0], [1.0]],
                    [[0.5], [3.0]],
                ]
            )
        },
        qns=[[bond2], [phys3], [right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg_12, None, None],
    )
    return A, B, C


def _three_site_dense_mpo():
    h = np.diag([0.0, 1.0])
    return [
        h.reshape(1, 1, 2, 2).copy(),
        h.reshape(1, 1, 2, 2).copy(),
        h.reshape(1, 1, 2, 2).copy(),
    ]


def _three_site_dense_onsite_sum_mpo():
    h = np.diag([0.0, 1.0])
    ident = np.eye(2)
    first = np.zeros((1, 2, 2, 2))
    middle = np.zeros((2, 2, 2, 2))
    last = np.zeros((2, 1, 2, 2))

    first[0, 0] = ident
    first[0, 1] = h
    middle[0, 0] = ident
    middle[0, 1] = h
    middle[1, 1] = ident
    last[0, 0] = h
    last[1, 0] = ident
    return [first, middle, last]


def _three_site_dense_nn_mpo():
    n = np.diag([0.0, 1.0])
    ident = np.eye(2)
    first = np.zeros((1, 3, 2, 2))
    middle = np.zeros((3, 3, 2, 2))
    last = np.zeros((3, 1, 2, 2))

    first[0, 0] = ident
    first[0, 1] = n
    middle[0, 0] = ident
    middle[0, 1] = n
    middle[1, 2] = n
    middle[2, 2] = ident
    last[1, 0] = n
    last[2, 0] = ident
    return [first, middle, last]


def _three_site_dense_long_range_mpo():
    n = np.diag([0.0, 1.0])
    ident = np.eye(2)
    first = np.zeros((1, 3, 2, 2))
    middle = np.zeros((3, 3, 2, 2))
    last = np.zeros((3, 1, 2, 2))

    first[0, 0] = ident
    first[0, 1] = n
    middle[0, 0] = ident
    middle[1, 1] = ident
    middle[2, 2] = ident
    last[1, 0] = n
    last[2, 0] = ident
    return [first, middle, last]


def _block_sparse_mpo_for_sites(sites, mpo_factors=None):
    if mpo_factors is None:
        mpo_factors = _three_site_dense_mpo()
    phys_slice_maps = []
    for site in sites:
        dims = {}
        for key, block in site.data.items():
            sector = key[1]
            dims.setdefault(sector, block.shape[1])
        offset = 0
        slices = {}
        for sector in site.qns[1]:
            if sector in slices:
                continue
            dim = dims[sector]
            slices[sector] = slice(offset, offset + dim)
            offset += dim
        phys_slice_maps.append(slices)
    return [
        MPO.from_dense(
            core,
            phys_out_dims={
                sector: int(slice_.stop - slice_.start)
                for sector, slice_ in phys_slices.items()
            },
        )
        for core, phys_slices in zip(mpo_factors, phys_slice_maps)
    ]


def _site_operator_mpo_for_sites(sites):
    mpo = []
    for site in sites:
        dims = {}
        for key, block in site.data.items():
            dims.setdefault(key[1], block.shape[1])
        phys_leg = PhysicalLeg.from_dims(dims, sectors=tuple(dict.fromkeys(site.qns[1])))
        op_blocks = {}
        offset = 0
        for sector in phys_leg.sectors:
            dim = phys_leg.dim(sector)
            op_blocks[(sector, sector)] = np.diag(np.arange(offset, offset + dim, dtype=float))
            offset += dim
        site_op = SiteOperator(
            blocks=op_blocks,
            phys_out_leg=phys_leg,
            phys_in_leg=phys_leg,
        )
        mpo.append(MPO.from_site_operator(site_op))
    return mpo


def _identity_mpo_for_sites(sites):
    mpo = []
    for site in sites:
        dims = {}
        for key, block in site.data.items():
            dims.setdefault(key[1], block.shape[1])
        phys_leg = PhysicalLeg.from_dims(dims, sectors=tuple(dict.fromkeys(site.qns[1])))
        mpo.append(MPO.from_site_operator(identity_operator(phys_leg)))
    return mpo


def _number_operator_for_site(site):
    dims = {}
    for key, block in site.data.items():
        dims.setdefault(key[1], block.shape[1])
    phys_leg = PhysicalLeg.from_dims(dims, sectors=tuple(dict.fromkeys(site.qns[1])))
    blocks = {}
    offset = 0
    for sector in phys_leg.sectors:
        dim = phys_leg.dim(sector)
        blocks[(sector, sector)] = np.diag(np.arange(offset, offset + dim, dtype=float))
        offset += dim
    return SiteOperator(blocks=blocks, phys_out_leg=phys_leg, phys_in_leg=phys_leg)


def _three_site_multiblock_chain():
    left = _charge_spin_sector(0, 0)
    bond1_a = _charge_spin_sector(1, 1)
    bond1_b = _charge_spin_sector(3, 1)
    bond2_a = _charge_spin_sector(1, 1)
    bond2_b = _charge_spin_sector(3, 1)
    right = _charge_spin_sector(2, 0)
    phys_a = _charge_spin_sector(1, 1)
    phys_b = _charge_spin_sector(1, 3)

    bond_leg_01 = FusionLeg(child_legs=(0, 2))
    bond_leg_12 = FusionLeg(child_legs=(0, 2))

    A = NonabelianTensor(
        data={
            (left, phys_a, bond1_a): np.array([[[1.0]]]),
            (left, phys_b, bond1_b): np.array([[[1.5]]]),
        },
        qns=[[left], [phys_a, phys_b], [bond1_a, bond1_b]],
        dirs=[-1, 1, 1],
        fusion_legs=[None, None, bond_leg_01],
    )
    B = NonabelianTensor(
        data={
            (bond1_a, phys_a, bond2_a): np.array([[[1.0]]]),
            (bond1_b, phys_b, bond2_b): np.array([[[1.0]]]),
        },
        qns=[[bond1_a, bond1_b], [phys_a, phys_b], [bond2_a, bond2_b]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg_01, None, bond_leg_12],
    )
    C = NonabelianTensor(
        data={
            (bond2_a, phys_a, right): np.array([[[1.0]]]),
            (bond2_b, phys_b, right): np.array([[[1.0]]]),
        },
        qns=[[bond2_a, bond2_b], [phys_a, phys_b], [right]],
        dirs=[-1, 1, 1],
        fusion_legs=[bond_leg_12, None, None],
    )
    return A, B, C


def test_block_sparse_mpo_core_round_trips_dense_reference():
    A, B, C = _three_site_chain()
    dense_mpo = _three_site_dense_mpo()
    sparse_mpo = _block_sparse_mpo_for_sites([A, B, C], dense_mpo)

    for dense_core, sparse_core, site in zip(dense_mpo, sparse_mpo, [A, B, C]):
        dims = {}
        for key, block in site.data.items():
            dims.setdefault(key[1], block.shape[1])
        offset = 0
        phys_slices = {}
        for sector in site.qns[1]:
            if sector in phys_slices:
                continue
            dim = dims[sector]
            phys_slices[sector] = slice(offset, offset + dim)
            offset += dim
        np.testing.assert_allclose(
            sparse_core.as_dense(phys_slices),
            dense_core,
        )


def test_block_sparse_mpo_core_carries_intrinsic_physical_leg_metadata():
    A, _, _ = _three_site_chain()
    dense_core = _three_site_dense_mpo()[0]
    sparse_core = _block_sparse_mpo_for_sites([A], [dense_core])[0]

    assert isinstance(sparse_core.phys_out_leg, PhysicalLeg)
    assert sparse_core.phys_out_leg == sparse_core.phys_in_leg
    assert sparse_core.phys_out_leg.sectors == tuple(dict.fromkeys(A.qns[1]))
    assert sparse_core.phys_out_leg.total_dim == dense_core.shape[2]
    np.testing.assert_allclose(sparse_core.as_dense(), dense_core)


def test_block_sparse_site_operator_builds_mpo_core_directly():
    A, _, _ = _three_site_chain()
    phys_leg = PhysicalLeg.from_dims({A.qns[1][0]: 2}, sectors=(A.qns[1][0],))
    site_op = SiteOperator(
        blocks={(A.qns[1][0], A.qns[1][0]): np.diag([0.0, 1.0])},
        phys_out_leg=phys_leg,
        phys_in_leg=phys_leg,
    )

    mpo_core = MPO.from_site_operator(site_op)

    assert mpo_core.left_dim == 1
    assert mpo_core.right_dim == 1
    np.testing.assert_allclose(
        mpo_core.as_dense(),
        np.diag([0.0, 1.0]).reshape(1, 1, 2, 2),
    )


def test_site_operator_mpo_path_matches_dense_reference():
    A, B, C = _three_site_chain()
    dense_mpo = _three_site_dense_mpo()
    direct_mpo = _site_operator_mpo_for_sites([A, B, C])
    merged = merge_mps_sites(B, C)

    op_direct = build_block_sparse_bond_operator([A, B, C], direct_mpo, 1, merged)
    op_dense = build_dense_bond_operator([A, B, C], dense_mpo, 1, merged)

    optimized_direct, objective_direct = solve_local_two_site(
        merged, op_direct, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_direct, optimized_dense)
    assert objective_direct["energy"] == pytest.approx(objective_dense["energy"])


def test_site_operator_mpo_path_matches_dense_reference_for_multiblock_case():
    A, B, C = _three_site_multiblock_chain()
    dense_mpo = _three_site_dense_mpo()
    direct_mpo = _site_operator_mpo_for_sites([A, B, C])
    merged = merge_mps_sites(B, C)

    op_direct = build_block_sparse_bond_operator([A, B, C], direct_mpo, 1, merged)
    op_dense = build_dense_bond_operator([A, B, C], dense_mpo, 1, merged)

    optimized_direct, objective_direct = solve_local_two_site(
        merged, op_direct, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_direct, optimized_dense)
    assert objective_direct["energy"] == pytest.approx(objective_dense["energy"])


def test_chain_mpo_builder_reproduces_simple_onsite_chain():
    A, B, C = _three_site_chain()
    dense_mpo = _three_site_dense_onsite_sum_mpo()
    builder = AutoMPO.from_sites([A, B, C])

    for site, tensor in enumerate([A, B, C]):
        builder.add_onsite(site, _number_operator_for_site(tensor))
    built_mpo = builder.build()

    merged = merge_mps_sites(B, C)
    op_built = build_block_sparse_bond_operator([A, B, C], built_mpo, 1, merged)
    op_dense = build_dense_bond_operator([A, B, C], dense_mpo, 1, merged)

    optimized_built, objective_built = solve_local_two_site(
        merged, op_built, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    assert abs(float(objective_built["energy"])) <= 1e-8
    assert abs(float(objective_dense["energy"])) <= 1e-8
    assert float(objective_built["residual"]) <= float(objective_dense["residual"]) + 1e-8
    assert sorted(optimized_built.data) == sorted(optimized_dense.data)


def test_chain_mpo_builder_builds_nearest_neighbor_couplings():
    A, B, C = _three_site_chain()
    dense_mpo = _three_site_dense_nn_mpo()
    builder = AutoMPO.from_sites([A, B, C])
    nA = _number_operator_for_site(A)
    nB = _number_operator_for_site(B)
    nC = _number_operator_for_site(C)
    builder.add_nearest_neighbor(0, nA, nB)
    builder.add_nearest_neighbor(1, nB, nC)
    built_mpo = builder.build()

    merged = merge_mps_sites(B, C)
    op_built = build_block_sparse_bond_operator([A, B, C], built_mpo, 1, merged)
    op_dense = build_dense_bond_operator([A, B, C], dense_mpo, 1, merged)

    optimized_built, objective_built = solve_local_two_site(
        merged, op_built, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_built, optimized_dense)
    assert objective_built["energy"] == pytest.approx(objective_dense["energy"])


def test_autompo_add_term_builds_long_range_product():
    A, B, C = _three_site_chain()
    dense_mpo = _three_site_dense_long_range_mpo()
    builder = AutoMPO.from_sites([A, B, C])
    nA = _number_operator_for_site(A)
    nC = _number_operator_for_site(C)
    builder.add_term((2, nC), (0, nA))
    built_mpo = builder.build()

    merged = merge_mps_sites(B, C)
    op_built = build_block_sparse_bond_operator([A, B, C], built_mpo, 1, merged)
    op_dense = build_dense_bond_operator([A, B, C], dense_mpo, 1, merged)

    optimized_built, objective_built = solve_local_two_site(
        merged, op_built, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_built, optimized_dense)
    assert objective_built["energy"] == pytest.approx(objective_dense["energy"])


def test_sweep_once_left_to_right_updates_bonds_in_order():
    A, B, C = _three_site_chain()
    result = sweep_once([A, B, C], direction="lr")

    assert result["direction"] == "lr"
    assert [update["bond"] for update in result["updates"]] == [0, 1]
    _assert_same_tensor(result["sites"][0], result["updates"][0]["left"])
    _assert_same_tensor(result["sites"][1], result["updates"][1]["left"])
    _assert_same_tensor(result["sites"][2], result["updates"][1]["right"])


def test_dense_environment_chain_builds_bond_operator():
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    env = DenseEnvironmentChain.build([A, B, C], mpo)
    merged = merge_mps_sites(A, B)
    operator = env.bond_operator(0, merged)
    optimized, objective = solve_local_two_site(
        merged,
        operator,
        tol=1e-10,
        itermax=50,
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["energy"] == pytest.approx(0.0)
    assert objective["davidson_converged"] is True


def test_dense_environment_chain_accepts_block_sparse_mpo_cores():
    A, B, C = _three_site_chain()
    dense_mpo = _three_site_dense_mpo()
    sparse_mpo = _block_sparse_mpo_for_sites([A, B, C], dense_mpo)
    merged = merge_mps_sites(A, B)

    op_dense = DenseEnvironmentChain.build([A, B, C], dense_mpo).bond_operator(0, merged)
    op_sparse_mpo = DenseEnvironmentChain.build([A, B, C], sparse_mpo).bond_operator(0, merged)

    optimized_dense, objective_dense = solve_local_two_site(merged, op_dense, tol=1e-10, itermax=50)
    optimized_sparse, objective_sparse = solve_local_two_site(
        merged, op_sparse_mpo, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_sparse, optimized_dense)
    assert objective_sparse["energy"] == pytest.approx(objective_dense["energy"])


def test_block_sparse_identity_norm_operator_can_be_marked_canonical():
    sites = build_product_spatial_mps(["full", "empty", "full", "empty"])
    hubbard_mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=2.0,
    )
    sites = sweep_once(
        sites,
        direction="lr",
        mpo_factors=hubbard_mpo,
        max_bond=32,
        local_solver_kwargs={"itermax": 30},
    )["sites"]
    identity_mpo = _identity_mpo_factors_for_sites_and_mpo(sites, hubbard_mpo)
    env = BlockSparseEnvironmentChain.build(sites, identity_mpo)
    merged = _expand_two_site_support(sites[2], sites[3], merge_mps_sites(sites[2], sites[3]))
    operator = env.bond_operator(2, merged)

    assert operator.identity_like is True


def test_right_canonicalize_sites_preserves_state_energy_and_gauge():
    sites = build_product_spatial_mps(["full", "empty", "full", "empty"])
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=2.0,
    )
    before_num = contract_chain_expectation(sites, mpo)
    before_den = contract_chain_expectation(sites, _identity_mpo_factors_for_sites_and_mpo(sites, mpo))

    canonical = right_canonicalize_sites(sites)

    after_num = contract_chain_expectation(canonical, mpo)
    after_den = contract_chain_expectation(canonical, _identity_mpo_factors_for_sites_and_mpo(canonical, mpo))

    assert np.real(after_num / after_den) == pytest.approx(np.real(before_num / before_den))
    assert right_canonical_error(canonical[0]) < 1e-10
    assert right_canonical_error(canonical[1]) < 1e-10
    assert right_canonical_error(canonical[2]) < 1e-10


def test_reduced_spatial_site_uses_degeneracy_only_physical_leg():
    leg = reduced_physical_leg_from_spatial_orbital()
    assert [leg.dim(sector) for sector in leg.sectors] == [1, 1, 1]

    sites = build_reduced_product_spatial_mps(["full", "empty", "full", "empty"])
    assert all(site.metadata.get("physical_basis") == "reduced_spatial" for site in sites)
    assert all(block.shape[1] == 1 for site in sites for block in site.data.values())


def test_reduced_spatial_canonicalization_is_full_cg_orthonormal():
    sites = build_random_reduced_spatial_mps(
        6,
        target_sector=half_filled_singlet_sector(6),
        bond_multiplicity=4,
        seed=7,
    )
    canonical = mixed_canonicalize_sites(
        sites,
        3,
        cutoff=0.0,
        max_bond=None,
        max_bond_mode="reduced",
    )

    left_err, right_err = mixed_cg_canonical_errors(canonical, 3)
    assert left_err < 1e-10
    assert right_err < 1e-10
    assert all(site.metadata.get("physical_basis") == "reduced_spatial" for site in canonical)


def test_right_canonicalize_sites_exposes_identity_like_norm_on_edge_bond():
    sites = build_product_spatial_mps(["full", "empty", "full", "empty"])
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=2.0,
    )
    canonical = right_canonicalize_sites(sites)
    identity_env = BlockSparseEnvironmentChain.build(
        canonical,
        _identity_mpo_factors_for_sites_and_mpo(canonical, mpo),
    )
    merged = _expand_two_site_support(canonical[0], canonical[1], merge_mps_sites(canonical[0], canonical[1]))
    norm_operator = identity_env.bond_operator(0, merged)

    assert norm_operator.identity_like is True


def test_canonical_edge_bond_can_use_orthonormalized_coupled_local_solve():
    sites = build_product_spatial_mps(["full", "empty", "full", "empty"])
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=2.0,
    )
    canonical = right_canonicalize_sites(sites)
    h_env = BlockSparseEnvironmentChain.build(canonical, mpo)
    n_env = BlockSparseEnvironmentChain.build(
        canonical,
        _identity_mpo_factors_for_sites_and_mpo(canonical, mpo),
    )
    merged = _expand_two_site_support(canonical[0], canonical[1], merge_mps_sites(canonical[0], canonical[1]))
    operator = h_env.bond_operator(0, merged)
    norm_operator = n_env.bond_operator(0, merged)

    optimized_gen, objective_gen = solve_local_two_site(
        merged,
        operator,
        norm_operator=norm_operator,
        couple_physical="auto",
        tol=1e-10,
        itermax=30,
    )
    optimized_ortho, objective_ortho = solve_local_two_site(
        merged,
        TwoSiteEffectiveH(
            operator=operator,
            norm_operator=norm_operator,
            canonical_norm=True,
        ),
        couple_physical="auto",
        tol=1e-10,
        itermax=30,
        dense_fallback_dim=256,
    )

    assert objective_ortho["effective_local_problem"] == "orthonormalized_standard"
    assert objective_ortho["canonical_norm_used"] is True
    assert objective_ortho["coupled_physical_used"] is True
    assert objective_ortho["energy"] == pytest.approx(objective_gen["energy"])
    _assert_same_tensor(optimized_ortho, optimized_gen)


def test_large_canonical_bond_can_use_uncoupled_standard_path():
    sites = build_product_spatial_mps(["full", "empty"] * 3)
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=2.0,
    )
    first_sweep = sweep_once(
        sites,
        direction="lr",
        mpo_factors=mpo,
        max_bond=128,
        max_bond_mode="reduced",
        local_solver_kwargs={"itermax": 30, "dense_fallback_dim": 512},
    )
    canonical = left_canonicalize_sites(
        first_sweep["sites"],
        max_bond=None,
        cutoff=0.0,
        max_bond_mode="reduced",
        bond_coupling="left",
    )
    h_env = BlockSparseEnvironmentChain.build(canonical, mpo)
    n_env = BlockSparseEnvironmentChain.build(
        canonical,
        _identity_mpo_factors_for_sites_and_mpo(canonical, mpo),
    )
    merged = _expand_two_site_support(canonical[4], canonical[5], merge_mps_sites(canonical[4], canonical[5]))
    operator = h_env.bond_operator(4, merged)
    norm_operator = n_env.bond_operator(4, merged)

    optimized_gen, objective_gen = solve_local_two_site(
        merged,
        operator,
        norm_operator=norm_operator,
        couple_physical=False,
        tol=1e-10,
        itermax=30,
        dense_fallback_dim=512,
    )
    optimized_std, objective_std = solve_local_two_site(
        merged,
        TwoSiteEffectiveH(
            operator=operator,
            norm_operator=norm_operator,
            canonical_norm=True,
        ),
        couple_physical=False,
        tol=1e-10,
        itermax=30,
        dense_fallback_dim=512,
        orthonormalized_dense_dim=16,
    )

    assert objective_std["effective_local_problem"] == "standard"
    assert objective_std["coupled_physical_used"] is False
    assert "coupled_physical_skipped" not in objective_std
    assert objective_std["energy"] == pytest.approx(objective_gen["energy"])
    _assert_same_tensor(optimized_std, optimized_gen)


def test_small_coupled_norm_problem_can_use_orthonormalized_dense_path():
    left = _charge_spin_sector(0, 0)
    phys_left = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(0, 0)
    phys_right = _charge_spin_sector(1, 1)
    singlet_vec = np.array([0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0])
    singlet_block = singlet_vec.reshape(1, 2, 2, 1)

    merged = NonabelianTensor(
        data={(left, phys_left, phys_right, right): singlet_block},
        qns=[[left], [phys_left], [phys_right], [right]],
        dirs=[-1, 1, 1, 1],
    )
    matrix = -np.outer(singlet_vec, singlet_vec)
    norm = np.eye(matrix.shape[0], dtype=float)

    optimized_generalized, objective_generalized = solve_local_two_site(
        merged,
        matrix,
        norm_operator=norm,
        tol=1e-10,
        itermax=20,
        dense_fallback_dim=1,
        couple_physical=True,
    )
    optimized_ortho, objective_ortho = solve_local_two_site(
        merged,
        matrix,
        norm_operator=norm,
        tol=1e-10,
        itermax=20,
        dense_fallback_dim=64,
        couple_physical=True,
    )

    assert objective_ortho["effective_local_problem"] == "orthonormalized_dense"
    assert objective_ortho["coupled_physical_used"] is True
    assert objective_ortho["energy"] == pytest.approx(objective_generalized["energy"])
    _assert_same_tensor(optimized_ortho, optimized_generalized)


def test_mixed_canonical_interior_bond_can_use_uncoupled_orthonormalized_dense_path():
    sites = build_random_spatial_mps(
        6,
        target_sector=half_filled_singlet_sector(6),
        bond_multiplicity=4,
        seed=7,
        scale=0.3,
    )
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=2.0,
    )
    canonical = mixed_canonicalize_sites(
        sites,
        3,
        max_bond=None,
        cutoff=0.0,
        max_bond_mode="reduced",
        bond_coupling="left",
    )
    h_env = BlockSparseEnvironmentChain.build(canonical, mpo)
    n_env = BlockSparseEnvironmentChain.build(
        canonical,
        _identity_mpo_factors_for_sites_and_mpo(canonical, mpo),
    )
    merged = merge_mps_sites(canonical[2], canonical[3])
    operator = h_env.bond_operator(2, merged)
    norm_operator = n_env.bond_operator(2, merged)

    optimized_gen, objective_gen = solve_local_two_site(
        merged,
        operator,
        norm_operator=norm_operator,
        couple_physical="auto",
        tol=1e-10,
        itermax=30,
        dense_fallback_dim=512,
        orthonormalized_dense_dim=96,
    )
    optimized_ortho, objective_ortho = solve_local_two_site(
        merged,
        operator,
        norm_operator=norm_operator,
        couple_physical="auto",
        tol=1e-10,
        itermax=30,
        dense_fallback_dim=512,
        orthonormalized_dense_dim=2048,
    )
    optimized_reduced, objective_reduced = solve_local_two_site(
        merged,
        TwoSiteEffectiveH(
            operator=operator,
            norm_operator=norm_operator,
            canonical_norm=True,
        ),
        couple_physical="auto",
        tol=1e-10,
        itermax=60,
        dense_fallback_dim=512,
        orthonormalized_dense_dim=16,
    )
    optimized_recoupled, objective_recoupled = solve_local_two_site(
        merged,
        operator,
        norm_operator=norm_operator,
        couple_physical="auto",
        recoupled_reduced=True,
        recoupled_standard_metric_dim=1536,
        tol=1e-10,
        itermax=60,
        dense_fallback_dim=512,
        orthonormalized_dense_dim=16,
    )

    assert objective_gen["effective_local_problem"] == "generalized"
    assert objective_ortho["effective_local_problem"] == "orthonormalized_dense"
    assert objective_ortho["coupled_physical_used"] is False
    assert objective_ortho["coupled_physical_skipped"] == "uncoupled_orthonormalized_path"
    assert objective_ortho["energy"] == pytest.approx(objective_gen["energy"])
    assert objective_reduced["effective_local_problem"] == "generalized"
    assert objective_reduced["energy"] == pytest.approx(objective_gen["energy"], abs=1e-7)
    assert objective_recoupled["effective_local_problem"] == "recoupled_standard"
    assert objective_recoupled["recoupled_reduced"] is True
    assert objective_recoupled["recoupled_metric_whitened"] is True
    assert objective_recoupled["energy"] == pytest.approx(objective_gen["energy"], abs=1e-7)
    assert optimized_reduced.qns == optimized_ortho.qns
    assert optimized_reduced.dirs == optimized_ortho.dirs
    assert optimized_reduced.fusion_legs == optimized_ortho.fusion_legs
    assert set(optimized_reduced.data) == set(optimized_ortho.data)
    assert optimized_ortho.qns == optimized_gen.qns
    assert optimized_ortho.dirs == optimized_gen.dirs
    assert optimized_ortho.fusion_legs == optimized_gen.fusion_legs
    assert set(optimized_ortho.data) == set(optimized_gen.data)
    for key in optimized_ortho.data:
        np.testing.assert_allclose(
            optimized_ortho.data[key],
            optimized_gen.data[key],
            atol=1e-6,
            rtol=1e-6,
        )


def test_mpo_sweep_defaults_to_uncoupled_local_problem_on_product_edge():
    sites = build_product_spatial_mps(["full", "empty", "full", "empty"])
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=2.0,
    )
    result = sweep_once(
        sites,
        direction="lr",
        mpo_factors=mpo,
        max_bond=64,
        local_solver_kwargs={"itermax": 20, "dense_fallback_dim": 256},
    )

    problems = [
        update["local_objective"]["effective_local_problem"]
        for update in result["updates"]
    ]
    assert problems == ["standard", "standard", "standard"]
    assert all(
        update["local_objective"]["coupled_physical_used"] is False
        for update in result["updates"]
    )


def test_dense_environment_chain_supports_multiblock_sites():
    A, B, C = _three_site_multiblock_chain()
    mpo = _three_site_dense_mpo()
    env = DenseEnvironmentChain.build([A, B, C], mpo)
    merged = merge_mps_sites(A, B)
    operator = env.bond_operator(0, merged)

    optimized, objective = solve_local_two_site(
        merged,
        operator,
        tol=1e-10,
        itermax=50,
    )

    assert isinstance(optimized, NonabelianTensor)
    assert objective["davidson_converged"] is True
    assert objective["energy"] >= -1e-10


def test_build_dense_bond_operator_matches_environment_chain():
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    merged = merge_mps_sites(B, C)
    op_a = build_dense_bond_operator([A, B, C], mpo, 1, merged)
    op_b = DenseEnvironmentChain.build([A, B, C], mpo).bond_operator(1, merged)

    optimized_a, objective_a = solve_local_two_site(merged, op_a, tol=1e-10, itermax=50)
    optimized_b, objective_b = solve_local_two_site(merged, op_b, tol=1e-10, itermax=50)

    _assert_same_tensor(optimized_a, optimized_b)
    assert objective_a["energy"] == pytest.approx(objective_b["energy"])


def test_block_sparse_bond_operator_matches_dense_reference():
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    merged = merge_mps_sites(B, C)
    op_sparse = build_block_sparse_bond_operator([A, B, C], mpo, 1, merged)
    op_dense = build_dense_bond_operator([A, B, C], mpo, 1, merged)
    assert op_sparse.diag == pytest.approx(op_dense.diag)

    optimized_sparse, objective_sparse = solve_local_two_site(
        merged, op_sparse, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_sparse, optimized_dense)
    assert objective_sparse["energy"] == pytest.approx(objective_dense["energy"])


def test_block_sparse_bond_operator_accepts_block_sparse_mpo_cores():
    A, B, C = _three_site_chain()
    dense_mpo = _three_site_dense_mpo()
    sparse_mpo = _block_sparse_mpo_for_sites([A, B, C], dense_mpo)
    merged = merge_mps_sites(B, C)

    op_sparse = build_block_sparse_bond_operator([A, B, C], sparse_mpo, 1, merged)
    op_dense = build_dense_bond_operator([A, B, C], dense_mpo, 1, merged)
    assert op_sparse.diag == pytest.approx(op_dense.diag)

    optimized_sparse, objective_sparse = solve_local_two_site(
        merged, op_sparse, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_sparse, optimized_dense)
    assert objective_sparse["energy"] == pytest.approx(objective_dense["energy"])


def test_block_sparse_environment_chain_matches_dense_reference():
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    merged = merge_mps_sites(B, C)
    op_sparse = BlockSparseEnvironmentChain.build([A, B, C], mpo).bond_operator(1, merged)
    op_dense = DenseEnvironmentChain.build([A, B, C], mpo).bond_operator(1, merged)
    assert op_sparse.diag == pytest.approx(op_dense.diag)

    optimized_sparse, objective_sparse = solve_local_two_site(
        merged, op_sparse, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_sparse, optimized_dense)
    assert objective_sparse["energy"] == pytest.approx(objective_dense["energy"])


def _local_operator_packed_matvec(operator):
    return operator.aux_packed_matvec or operator.packed_matvec


def test_block_sparse_hubbard_compiled_packed_backend_matches_dense_reference(monkeypatch):
    monkeypatch.setattr(env_mod, "_FACTORIZED_PACKED_LOCAL_DIM", 10**9)
    sites = build_random_spatial_mps(4, seed=7, bond_multiplicity=4)
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=0.0,
    )
    merged = merge_mps_sites(sites[1], sites[2])
    op_sparse = BlockSparseEnvironmentChain.build(sites, mpo).bond_operator(1, merged)
    op_dense = DenseEnvironmentChain.build(sites, mpo).bond_operator(1, merged)
    packed, layout = pack_two_site_state(merged)
    packed_matvec = _local_operator_packed_matvec(op_sparse)

    assert getattr(packed_matvec, "backend", None) in {"compiled", "compiled-dense", "compiled-csr"}
    dense_ref, _ = pack_two_site_state(
        op_dense.tensor_matvec(unpack_two_site_state(packed, merged, layout=layout)),
        layout=layout,
    )
    np.testing.assert_allclose(
        packed_matvec(packed),
        dense_ref,
        atol=1e-10,
        rtol=1e-10,
    )


def test_block_sparse_hubbard_packed_backend_matches_dense_reference(monkeypatch):
    monkeypatch.setattr(env_mod, "_FACTORIZED_PACKED_LOCAL_DIM", 0)
    sites = build_random_spatial_mps(4, seed=7, bond_multiplicity=4)
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=0.0,
    )
    merged = merge_mps_sites(sites[1], sites[2])
    op_sparse = BlockSparseEnvironmentChain.build(sites, mpo).bond_operator(1, merged)
    op_dense = DenseEnvironmentChain.build(sites, mpo).bond_operator(1, merged)
    packed, layout = pack_two_site_state(merged)
    packed_matvec = _local_operator_packed_matvec(op_sparse)

    assert getattr(packed_matvec, "backend", None) == "factorized-batched"
    dense_ref, _ = pack_two_site_state(
        op_dense.tensor_matvec(unpack_two_site_state(packed, merged, layout=layout)),
        layout=layout,
    )
    np.testing.assert_allclose(
        packed_matvec(packed),
        dense_ref,
        atol=1e-10,
        rtol=1e-10,
    )


def test_dense_environment_sweep_advance_matches_rebuilt_chain():
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    chain = DenseEnvironmentChain.build([A, B, C], mpo)
    sweep_env = chain.start_sweep("lr")

    update = two_site_update(A, B, cutoff=0.0)
    sweep_env.advance_after_update(0, update["left"], update["right"])

    updated_sites = [update["left"], update["right"], C]
    merged = merge_mps_sites(updated_sites[1], updated_sites[2])
    op_incremental = sweep_env.bond_operator(1, merged)
    op_rebuilt = DenseEnvironmentChain.build(updated_sites, mpo).bond_operator(1, merged)

    optimized_inc, objective_inc = solve_local_two_site(merged, op_incremental, tol=1e-10, itermax=50)
    optimized_ref, objective_ref = solve_local_two_site(merged, op_rebuilt, tol=1e-10, itermax=50)

    _assert_same_tensor(optimized_inc, optimized_ref)
    assert objective_inc["energy"] == pytest.approx(objective_ref["energy"])


def test_sweep_once_right_to_left_accepts_bond_aware_solver():
    A, B, C = _three_site_chain()

    def solver(bond, merged):
        scaled = {
            key: (bond + 2.0) * block
            for key, block in merged.data.items()
        }
        return NonabelianTensor(
            scaled,
            [leg[:] for leg in merged.qns],
            merged.dirs[:],
            fusion_legs=merged.fusion_legs[:],
            metadata=merged.metadata.copy(),
        )

    result = sweep_once([A, B, C], direction="rl", solver=solver)

    assert result["direction"] == "rl"
    assert [update["bond"] for update in result["updates"]] == [1, 0]
    _assert_same_tensor(result["sites"][0], result["updates"][1]["left"])
    _assert_same_tensor(result["sites"][1], result["updates"][1]["right"])
    _assert_same_tensor(result["sites"][2], result["updates"][0]["right"])


def test_sweep_once_builds_hamiltonian_and_norm_environments_once_per_sweep(monkeypatch):
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    calls = []
    original_build = BlockSparseEnvironmentChain.build.__func__

    def spy_build(cls, sites, mpo_factors):
        calls.append(len(sites))
        return original_build(cls, sites, mpo_factors)

    monkeypatch.setattr(BlockSparseEnvironmentChain, "build", classmethod(spy_build))
    result = sweep_once(
        [A, B, C],
        direction="lr",
        mpo_factors=mpo,
        local_solver_kwargs={"tol": 1e-10, "itermax": 50},
    )

    assert len(calls) == 2
    assert calls == [3, 3]
    assert result["direction"] == "lr"
    assert len(result["updates"]) == 2


def test_run_sweeps_alternates_direction_and_records_history():
    A, B, C = _three_site_chain()
    result = run_sweeps([A, B, C], nsweeps=3, start_direction="lr")

    assert result["converged"] is False
    assert result["ncompleted"] == 3
    assert [entry["direction"] for entry in result["history"]] == ["lr", "rl", "lr"]
    assert result["last_direction"] == "lr"
    assert len(result["history"][0]["updates"]) == 2


def test_run_sweeps_can_mark_mixer_application_in_history():
    A, B, C = _three_site_chain()
    result = run_sweeps(
        [A, B, C],
        nsweeps=2,
        start_direction="lr",
        mixer_zero_block_noise_scale=1e-8,
        mixer_zero_block_noise_seed=7,
        mixer_nsweeps=1,
    )

    assert result["history"][0]["mixer_applied"] is True
    assert result["history"][1]["mixer_applied"] is False


def test_sweep_once_mixer_seeds_local_guess_without_perturbing_chain():
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    result = sweep_once(
        [A, B, C],
        direction="lr",
        mpo_factors=mpo,
        local_solver_kwargs={"tol": 1e-10, "itermax": 30},
        mixer_zero_block_noise_scale=1e-8,
        mixer_rng=np.random.default_rng(7),
    )

    assert any(
        update.get("local_objective", {}).get("local_guess_used") is True
        for update in result["updates"]
    )


def test_run_sweeps_stops_early_on_convergence_metric():
    A, B, C = _three_site_chain()
    result = run_sweeps(
        [A, B, C],
        nsweeps=5,
        start_direction="rl",
        conv_tol=0.25,
        measure=lambda sweep_result: 0.2,
    )

    assert result["converged"] is True
    assert result["ncompleted"] == 1
    assert result["history"][0]["direction"] == "rl"
    assert result["history"][0]["metric"] == pytest.approx(0.2)


def test_run_sweeps_records_bond_objectives_and_energy_summary():
    A, B, C = _three_site_chain()

    def solver(bond, merged):
        return {
            "optimized": merged,
            "energy": -10.0 - bond,
            "metric": 0.1 * (bond + 1),
        }

    result = run_sweeps([A, B, C], nsweeps=1, start_direction="lr", solver=solver)

    history = result["history"][0]
    bond_objectives = history["bond_objectives"]
    assert len(bond_objectives) == 2
    assert bond_objectives[0]["bond"] == 0
    assert bond_objectives[0]["energy"] == pytest.approx(-10.0)
    assert bond_objectives[0]["metric"] == pytest.approx(0.1)
    assert bond_objectives[1]["bond"] == 1
    assert bond_objectives[1]["energy"] == pytest.approx(-11.0)
    assert bond_objectives[1]["metric"] == pytest.approx(0.2)
    assert history["energy"] == pytest.approx(-10.5)
    assert history["objective_metric"] == pytest.approx(0.15)


def test_run_sweeps_verbose_can_print_sweep_and_bond_updates(capsys):
    A, B, C = _three_site_chain()

    def solver(bond, merged):
        return {
            "optimized": merged,
            "energy": -10.0 - bond,
            "metric": 0.1 * (bond + 1),
        }

    run_sweeps([A, B, C], nsweeps=1, start_direction="lr", solver=solver, verbose=2)

    out = capsys.readouterr().out
    assert "bond  0 |" in out
    assert "bond  1 |" in out
    assert "sweep  0 |" in out
    assert "problem=" in out
    assert "E_post=" in out


def test_run_sweeps_infers_convergence_from_stable_objectives():
    A, B, C = _three_site_chain()

    def solver(bond, merged):
        return {
            "optimized": merged,
            "energy": -2.0,
            "metric": 1e-12,
        }

    result = run_sweeps([A, B, C], nsweeps=3, start_direction="lr", solver=solver)

    assert result["converged"] is True
    assert result["ncompleted"] == 3
    assert [entry["direction"] for entry in result["history"]] == ["lr", "rl", "lr"]


def test_run_sweeps_applies_local_solver_schedule(monkeypatch):
    A, B, C = _three_site_chain()
    seen = []
    original_sweep_once = sweep_mod.sweep_once

    def spy_sweep_once(*args, **kwargs):
        seen.append(dict(kwargs.get("local_solver_kwargs") or {}))
        return original_sweep_once(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "sweep_once", spy_sweep_once)
    result = run_sweeps(
        [A, B, C],
        nsweeps=3,
        start_direction="lr",
        local_solver_kwargs={"tol": 1e-4},
        local_solver_schedule=[
            {"itermax": 4},
            {"itermax": 8, "tol": 1e-6},
        ],
    )

    assert seen == [
        {"tol": 1e-4, "itermax": 4},
        {"tol": 1e-6, "itermax": 8},
        {"tol": 1e-6, "itermax": 8},
    ]
    assert [entry["local_solver_kwargs"] for entry in result["history"]] == seen


def test_run_sweeps_reuses_same_bond_warm_start_guesses(monkeypatch):
    A, B, C = _three_site_chain()
    seen = []

    def fake_solve_local_two_site(merged, operator_spec, *, norm_operator=None, canonical_norm=False, **kwargs):
        _ = operator_spec, norm_operator, canonical_norm
        seen.append(isinstance(kwargs.get("guess"), NonabelianTensor))
        return merged.copy(), {"energy": 0.0, "metric": 0.0}

    monkeypatch.setattr(update_mod, "solve_local_two_site", fake_solve_local_two_site)
    result = run_sweeps(
        [A, B, C],
        nsweeps=2,
        start_direction="lr",
        local_operator=lambda bond, merged: f"bond-{bond}-op",
        warm_start_bonds=True,
    )

    assert seen == [False, False, True, True]
    assert [entry["warm_start_bonds"] for entry in result["history"]] == [True, True]
    assert all("warm_start" not in item for item in result["history"][0]["bond_objectives"])
    assert all(
        item.get("warm_start") == "bond_cache"
        for item in result["history"][1]["bond_objectives"]
    )


def test_run_sweeps_can_disable_same_bond_warm_start_guesses(monkeypatch):
    A, B, C = _three_site_chain()
    seen = []

    def fake_solve_local_two_site(merged, operator_spec, *, norm_operator=None, canonical_norm=False, **kwargs):
        _ = operator_spec, norm_operator, canonical_norm
        seen.append(isinstance(kwargs.get("guess"), NonabelianTensor))
        return merged.copy(), {"energy": 0.0, "metric": 0.0}

    monkeypatch.setattr(update_mod, "solve_local_two_site", fake_solve_local_two_site)
    result = run_sweeps(
        [A, B, C],
        nsweeps=2,
        start_direction="lr",
        local_operator=lambda bond, merged: f"bond-{bond}-op",
        warm_start_bonds=False,
    )

    assert seen == [False, False, False, False]
    assert [entry["warm_start_bonds"] for entry in result["history"]] == [False, False]
    assert all(
        "warm_start" not in item
        for entry in result["history"]
        for item in entry["bond_objectives"]
    )


def test_run_sweeps_uses_default_adaptive_schedule_for_mpo_path(monkeypatch):
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    seen = []
    original_sweep_once = sweep_mod.sweep_once

    def spy_sweep_once(*args, **kwargs):
        seen.append(dict(kwargs.get("local_solver_kwargs") or {}))
        return original_sweep_once(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "sweep_once", spy_sweep_once)
    result = run_sweeps(
        [A, B, C],
        nsweeps=2,
        start_direction="lr",
        mpo_factors=mpo,
    )

    assert seen == [
        {"tol": 1e-10, "itermax": 80, "max_space": 128},
        {"tol": 1e-10, "itermax": 80, "max_space": 128},
    ]
    assert [entry["local_solver_kwargs"] for entry in result["history"]] == seen


def test_run_sweeps_mpo_default_schedule_respects_explicit_solver_kwargs(monkeypatch):
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    seen = []
    original_sweep_once = sweep_mod.sweep_once

    def spy_sweep_once(*args, **kwargs):
        seen.append(dict(kwargs.get("local_solver_kwargs") or {}))
        return original_sweep_once(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "sweep_once", spy_sweep_once)
    result = run_sweeps(
        [A, B, C],
        nsweeps=2,
        start_direction="lr",
        mpo_factors=mpo,
        local_solver_kwargs={"itermax": 12, "tol": 1e-5},
    )

    assert seen == [
        {"tol": 1e-5, "itermax": 12, "max_space": 128},
        {"tol": 1e-5, "itermax": 12, "max_space": 128},
    ]
    assert [entry["local_solver_kwargs"] for entry in result["history"]] == seen


def test_sweep_driver_stores_run_state():
    A, B, C = _three_site_chain()
    driver = SweepDriver([A, B, C], nsweeps=3, start_direction="lr")

    returned = driver.run()

    assert returned is driver
    assert driver.converged is False
    assert driver.ncompleted == 3
    assert driver.last_direction == "lr"
    assert [entry["direction"] for entry in driver.history] == ["lr", "rl", "lr"]
    _assert_same_tensor(driver.sites[0], driver.history[-1]["updates"][0]["left"])


def test_mps_wrapper_owns_sites_and_can_merge_bonds():
    A, B, C = _three_site_chain()
    mps = MPS([A, B, C], target_sector=_charge_spin_sector(2, 0))

    copied = mps.copy()
    merged = mps.merge_bond(1)

    assert len(mps) == 3
    assert copied is not mps
    assert copied.sites is not mps.sites
    assert copied.target_sector == mps.target_sector
    assert merged.rank == 4


def test_run_sweeps_accepts_mps_wrapper():
    A, B, C = _three_site_chain()
    mps = MPS([A, B, C])

    result = run_sweeps(mps, nsweeps=1, start_direction="lr")

    assert isinstance(result["mps"], MPS)
    assert result["mps"].sites == result["sites"]
    assert result["history"][0]["direction"] == "lr"


def test_sweep_driver_accepts_mps_wrapper_and_keeps_sites_property():
    A, B, C = _three_site_chain()
    mps = MPS([A, B, C])

    driver = SweepDriver(mps, nsweeps=1, start_direction="lr").run()

    assert isinstance(driver.mps, MPS)
    assert driver.sites is driver.mps.sites
    assert driver.ncompleted == 1


def test_sweep_driver_marks_objective_stable_run_as_converged():
    A, B, C = _three_site_chain()

    def solver(bond, merged):
        return {
            "optimized": merged,
            "energy": -3.0,
            "metric": 1e-12,
        }

    driver = SweepDriver([A, B, C], nsweeps=3, start_direction="lr", solver=solver)
    driver.run()

    assert driver.converged is True
    assert driver.ncompleted == 3


def test_sweep_driver_reset_and_run_override():
    A, B, C = _three_site_chain()
    driver = SweepDriver([A, B, C], nsweeps=4, start_direction="lr", conv_tol=0.1)
    driver.run(measure=lambda sweep_result: 0.05)

    assert driver.converged is True
    assert driver.ncompleted == 1

    driver.reset()
    assert driver.history == []
    assert driver.converged is False
    assert driver.ncompleted == 0

    driver.run(nsweeps=2, start_direction="rl", measure=lambda sweep_result: 1.0)
    assert driver.converged is False
    assert driver.ncompleted == 2
    assert [entry["direction"] for entry in driver.history] == ["rl", "lr"]


def test_sweep_driver_exposes_last_energy_and_objective_metric():
    A, B, C = _three_site_chain()

    def solver(bond, merged):
        return {
            "optimized": merged,
            "energy": -2.0 - bond,
            "metric": 0.05 * (bond + 1),
        }

    driver = SweepDriver([A, B, C], nsweeps=1, solver=solver)
    driver.run()

    assert driver.last_energy == pytest.approx(-2.5)
    assert driver.last_objective_energy == pytest.approx(-2.5)
    assert driver.last_objective_metric == pytest.approx(0.075)


def test_sweep_driver_accepts_local_operator_callback():
    A, B, C = _three_site_chain()

    def local_operator(bond, merged):
        vec, _ = pack_two_site_state(merged)
        return LocalOperator(matrix=np.diag(np.arange(vec.size, dtype=float) + bond))

    driver = SweepDriver(
        [A, B, C],
        nsweeps=1,
        local_operator=local_operator,
        local_solver_kwargs={"tol": 1e-10, "itermax": 50},
    )
    driver.run()

    assert driver.last_energy == pytest.approx(0.5)
    assert driver.last_objective_energy == pytest.approx(0.5)
    assert driver.last_objective_metric is not None
    assert driver.history[0]["bond_objectives"][0]["energy"] == pytest.approx(0.0)
    assert driver.history[0]["bond_objectives"][1]["energy"] == pytest.approx(1.0)


def test_sweep_driver_forwards_local_solver_schedule():
    A, B, C = _three_site_chain()
    driver = SweepDriver(
        [A, B, C],
        nsweeps=2,
        local_solver_schedule=[{"itermax": 4}, {"itermax": 8}],
    )

    driver.run(local_solver_kwargs={"tol": 1e-5})

    assert [entry["local_solver_kwargs"] for entry in driver.history] == [
        {"tol": 1e-5, "itermax": 4},
        {"tol": 1e-5, "itermax": 8},
    ]


def test_sweep_driver_accepts_mpo_factors_for_effective_local_h():
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()

    driver = SweepDriver(
        [A, B, C],
        nsweeps=1,
        mpo_factors=mpo,
        local_solver_kwargs={"tol": 1e-10, "itermax": 50},
    )
    driver.run()

    numerator = contract_chain_expectation(driver.sites, mpo)
    denominator = contract_chain_expectation(driver.sites, _identity_mpo_for_sites(driver.sites))
    assert driver.last_energy == pytest.approx(float(np.real(numerator / denominator)))
    assert driver.last_objective_energy == pytest.approx(driver.history[0]["objective_energy"])
    assert "objective_energy" in driver.history[0]
    assert driver.last_objective_metric is not None
    assert len(driver.history[0]["bond_objectives"]) == 2


def test_sweep_driver_respects_requested_start_direction_for_mpo_path():
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()

    driver = SweepDriver(
        [A, B, C],
        nsweeps=1,
        start_direction="lr",
        mpo_factors=mpo,
        local_solver_kwargs={"tol": 1e-10, "itermax": 50},
    )
    driver.run()

    assert driver.history[0]["direction"] == "lr"


def test_mpo_sweep_canonicalization_ignores_bond_cap(monkeypatch):
    A, B, C = _three_site_chain()
    mpo = _three_site_dense_mpo()
    seen = []

    original_mixed = sweep_mod.mixed_canonicalize_sites

    def record_mixed(*args, **kwargs):
        seen.append(("mixed", kwargs.get("max_bond"), kwargs.get("cutoff")))
        return original_mixed(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "mixed_canonicalize_sites", record_mixed)

    sweep_once(
        [A, B, C],
        direction="lr",
        mpo_factors=mpo,
        max_bond=1,
        local_solver_kwargs={"tol": 1e-10, "itermax": 20},
    )

    assert seen
    assert seen[0] == ("mixed", None, 0.0)


def test_sweep_driver_accepts_multiblock_mpo_environment_path():
    A, B, C = _three_site_multiblock_chain()
    mpo = _three_site_dense_mpo()

    driver = SweepDriver(
        [A, B, C],
        nsweeps=1,
        mpo_factors=mpo,
        local_solver_kwargs={"tol": 1e-10, "itermax": 50},
    )
    driver.run()

    assert driver.last_energy is not None
    assert driver.last_energy >= -1e-10
    assert driver.last_objective_metric is not None
    assert len(driver.history[0]["bond_objectives"]) == 2


def test_sweep_driver_coupled_local_solver_projects_forbidden_blocks():
    sites = build_random_spatial_mps(4, seed=7, bond_multiplicity=4)
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        chemical_potential=0.0,
        onsite_u=4.0,
    )

    driver = SweepDriver(
        [site.copy() for site in sites],
        nsweeps=2,
        mpo_factors=mpo,
        max_bond=128,
        cutoff=0.0,
        local_solver_kwargs={
            "tol": 1e-10,
            "itermax": 30,
            "couple_physical": "auto",
        },
    )
    driver.run()

    assert driver.last_energy is not None
    assert len(driver.history) == 2
    for sweep in driver.history:
        for update in sweep["updates"]:
            optimized = update["optimized"]
            channels = optimized.metadata.get("contracted_channels", {})
            assert set(optimized.data) <= set(channels)
            for key in optimized.data:
                assert len(channels[key]) >= 1


def test_sweep_driver_mpo_path_retains_best_state_across_nonmonotone_sweeps():
    sites = build_random_spatial_mps(4, seed=7, bond_multiplicity=4)
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        chemical_potential=0.0,
        onsite_u=4.0,
    )

    driver = SweepDriver(
        [site.copy() for site in sites],
        nsweeps=4,
        mpo_factors=mpo,
        max_bond=128,
        cutoff=0.0,
    )
    driver.run()

    history_energies = [entry["energy"] for entry in driver.history]
    assert driver.last_energy == pytest.approx(min(history_energies))
    assert driver.last_objective_energy == pytest.approx(driver.history[-1]["objective_energy"])

    numerator = contract_chain_expectation(driver.sites, mpo)
    denominator = contract_chain_expectation(driver.sites, _identity_mpo_for_sites(driver.sites))
    assert driver.last_energy == pytest.approx(float(np.real(numerator / denominator)))


def test_sweep_driver_default_hubbard_path_returns_half_filled_singlet_sector():
    sites = build_random_spatial_mps(4, seed=7, bond_multiplicity=4)
    mpo = build_spatial_hubbard_mpo(
        4,
        hopping_t=1.0,
        chemical_potential=0.0,
        onsite_u=4.0,
    )

    driver = SweepDriver(
        [site.copy() for site in sites],
        nsweeps=4,
        mpo_factors=mpo,
        max_bond=128,
        cutoff=0.0,
    )
    driver.run()

    identity_mpo = [MPO.from_site_operator(spatial_identity(site)) for site in driver.sites]
    denominator = contract_chain_expectation(driver.sites, identity_mpo)
    total_charge = 0.0
    for site_index, site in enumerate(driver.sites):
        ops = [MPO.from_site_operator(spatial_identity(other)) for other in driver.sites]
        ops[site_index] = MPO.from_site_operator(spatial_number(site))
        total_charge += float(np.real(contract_chain_expectation(driver.sites, ops) / denominator))

    assert driver.last_energy == pytest.approx(-1.9531453086845532)
    assert total_charge == pytest.approx(4.0)
    assert driver.sites[-1].qns[2] == [half_filled_singlet_sector(4)]


def test_expand_two_site_support_from_product_state_adds_missing_hubbard_channels():
    left, right = build_product_spatial_mps(["full", "empty"], enrich_bond_sectors=False)
    merged = merge_mps_sites(left, right)
    expanded = _expand_two_site_support(left, right, merged)

    assert len(merged.data) == 1
    assert len(expanded.data) > len(merged.data)
    assert any(
        key[1].charge == 1 and key[2].charge == 1
        for key in expanded.data
    )


def test_mpo_sweep_canonicalizes_product_state_before_first_update():
    sites = build_product_spatial_mps(["up", "down", "up"])
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=0.0,
    )

    sweep = sweep_once(
        sites,
        direction="lr",
        mpo_factors=mpo,
        max_bond=128,
        cutoff=0.0,
        local_solver_kwargs={"tol": 1e-10, "itermax": 30, "max_space": 64},
    )

    first_update = sweep["updates"][0]
    right_bond_sectors = {key[3] for key in first_update["merged"].data}

    assert len(first_update["merged"].data) == 1
    assert len(right_bond_sectors) == 1


def test_mpo_sweep_can_record_post_update_chain_energy():
    sites = build_product_spatial_mps(["up", "down", "up"])
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=0.0,
    )

    sweep = sweep_once(
        sites,
        direction="lr",
        mpo_factors=mpo,
        max_bond=128,
        cutoff=0.0,
        local_solver_kwargs={"tol": 1e-10, "itermax": 30, "max_space": 64},
        record_post_update_energy=True,
    )

    updates = sweep["updates"]
    assert updates[0]["local_objective"]["post_update_energy"] == pytest.approx(0.0)
    assert updates[1]["local_objective"]["post_update_energy"] == pytest.approx(
        -0.8284271247461902
    )


def test_mpo_sweep_product_state_uses_canonical_gauge_from_start():
    sites = build_product_state(["up", "down", "up"])
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=0.0,
    )

    sweep = sweep_once(
        sites,
        direction="lr",
        mpo_factors=mpo,
        max_bond=128,
        cutoff=0.0,
        local_solver_kwargs={"tol": 1e-10, "itermax": 30, "max_space": 64},
        record_post_update_energy=True,
    )

    updates = sweep["updates"]
    assert updates[0]["local_objective"]["effective_local_problem"] == "standard"
    assert updates[1]["local_objective"]["effective_local_problem"] == "standard"
    assert updates[0]["local_objective"]["post_update_energy"] == pytest.approx(0.0)
    assert updates[1]["local_objective"]["post_update_energy"] == pytest.approx(
        -0.8284271247461902
    )


def test_build_product_state_does_not_request_input_gauge_preservation():
    sites = build_product_state(["up", "down", "up"])

    assert all(not site.metadata.get("preserve_mpo_input_gauge") for site in sites)


def test_mpo_sweep_final_environment_energy_matches_direct_contraction():
    sites = build_product_state(["up", "down", "up"])
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=4.0,
        chemical_potential=0.0,
    )

    sweep = sweep_once(
        sites,
        direction="lr",
        mpo_factors=mpo,
        max_bond=128,
        cutoff=0.0,
        local_solver_kwargs={"tol": 1e-10, "itermax": 30, "max_space": 64},
    )

    numerator = contract_chain_expectation(sweep["sites"], mpo)
    denominator = contract_chain_expectation(
        sweep["sites"],
        _identity_mpo_factors_for_sites_and_mpo(sweep["sites"], mpo),
    )

    assert sweep["final_mpo_numerator"] == pytest.approx(numerator)
    assert sweep["final_mpo_denominator"] == pytest.approx(denominator)


def test_block_sparse_bond_operator_matches_dense_reference_for_multiblock_case():
    A, B, C = _three_site_multiblock_chain()
    mpo = _three_site_dense_mpo()
    merged = merge_mps_sites(B, C)
    op_sparse = build_block_sparse_bond_operator([A, B, C], mpo, 1, merged)
    op_dense = build_dense_bond_operator([A, B, C], mpo, 1, merged)

    optimized_sparse, objective_sparse = solve_local_two_site(
        merged, op_sparse, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_sparse, optimized_dense)
    assert objective_sparse["energy"] == pytest.approx(objective_dense["energy"])


def test_block_sparse_environment_chain_matches_dense_reference_for_multiblock_case():
    A, B, C = _three_site_multiblock_chain()
    mpo = _three_site_dense_mpo()
    merged = merge_mps_sites(B, C)
    op_sparse = BlockSparseEnvironmentChain.build([A, B, C], mpo).bond_operator(1, merged)
    op_dense = DenseEnvironmentChain.build([A, B, C], mpo).bond_operator(1, merged)

    optimized_sparse, objective_sparse = solve_local_two_site(
        merged, op_sparse, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_sparse, optimized_dense)
    assert objective_sparse["energy"] == pytest.approx(objective_dense["energy"])


def test_block_sparse_environment_chain_accepts_block_sparse_mpo_cores_for_multiblock_case():
    A, B, C = _three_site_multiblock_chain()
    dense_mpo = _three_site_dense_mpo()
    sparse_mpo = _block_sparse_mpo_for_sites([A, B, C], dense_mpo)
    merged = merge_mps_sites(B, C)

    op_sparse = BlockSparseEnvironmentChain.build([A, B, C], sparse_mpo).bond_operator(1, merged)
    op_dense = DenseEnvironmentChain.build([A, B, C], dense_mpo).bond_operator(1, merged)

    optimized_sparse, objective_sparse = solve_local_two_site(
        merged, op_sparse, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_sparse, optimized_dense)
    assert objective_sparse["energy"] == pytest.approx(objective_dense["energy"])


def test_svd_two_site_nonabelian_truncates_globally():
    left = _charge_spin_sector(0, 0)
    bond = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    phys_left = _charge_spin_sector(1, 1)
    phys_right = _charge_spin_sector(1, 1)

    AA = NonabelianTensor(
        data={
            (left, phys_left, phys_right, right): np.diag([3.0, 1.0]).reshape(1, 2, 2, 1)
        },
        qns=[[left], [phys_left], [phys_right], [right]],
        dirs=[-1, 1, 1, 1],
        metadata={
            "contracted_fusion_leg": FusionLeg(
                child_legs=(0, 1, 2, 3),
                child_sector_lists=((left,), (phys_left,), (phys_right,), (right,)),
                child_dirs=(-1, 1, 1, 1),
                sectors=(bond,),
                orientation=1,
                coupling="contracted",
                pipe=FusionPipe.from_entries(
                    child_legs=(0, 1, 2, 3),
                    child_sector_lists=((left,), (phys_left,), (phys_right,), (right,)),
                    child_dirs=(-1, 1, 1, 1),
                    fused_sectors=(bond,),
                    entries=(
                        FusionPipeEntry(
                            (left, phys_left, phys_right, right),
                            bond,
                            0,
                            0,
                            4,
                            (1, 2, 2, 1),
                        ),
                    ),
                    orientation=1,
                    coupling="contracted",
                ),
            ),
        },
    )

    A_new, B_new, singular_values, trunc_err, kept = svd_two_site(
        AA, max_bond=1, absorb="right"
    )

    assert kept == 1
    assert trunc_err == pytest.approx(0.1)
    assert singular_values[bond].shape == (1, 1)
    assert A_new.rank == 3
    assert B_new.rank == 3


def test_svd_two_site_irrep_aware_max_bond_uses_su2_state_budget():
    singlet = _charge_spin_sector(0, 0)
    doublet = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(0, 0)
    phys = _charge_spin_sector(0, 0)

    merged = NonabelianTensor(
        data={
            (singlet, phys, right, phys): np.array([[[[0.79]]]]),
            (doublet, phys, right, phys): np.array([[[[0.80]]]]),
        },
        qns=[[singlet, doublet], [phys], [right], [phys]],
        dirs=[-1, 1, 1, 1],
        metadata={
            "contracted_channels": {
                (singlet, phys, right, phys): (singlet,),
                (doublet, phys, right, phys): (doublet,),
            },
        },
    )

    _, _, singular_values_reduced, _, kept_reduced = svd_two_site(
        merged,
        max_bond=1,
        max_bond_mode="reduced",
        absorb="right",
    )
    _, _, singular_values_states, _, kept_states = svd_two_site(
        merged,
        max_bond=1,
        max_bond_mode="states",
        absorb="right",
    )

    assert kept_reduced == 1
    assert kept_states == 1
    assert doublet in singular_values_reduced
    assert singlet not in singular_values_reduced
    assert singlet in singular_values_states
    assert doublet not in singular_values_states


def test_reduced_truncation_helper_uses_shared_su2_state_budget():
    singlet = _charge_spin_sector(0, 0)
    doublet = _charge_spin_sector(1, 1)

    def _single_channel_projection(sector, value):
        key = ((sector,), (1,), 0)
        pipe = FusionPipe.from_entries(
            child_legs=(0,),
            child_sector_lists=((sector,),),
            child_dirs=(1,),
            fused_sectors=(sector,),
            entries=(
                FusionPipeEntry(
                    child_sectors=(sector,),
                    fused_sector=sector,
                    slot=0,
                    offset=0,
                    local_dim=1,
                    selected_shape=(1,),
                ),
            ),
            orientation=1,
            coupling="left",
        )
        return ReducedProjectedSector(
            sector=sector,
            left_pipe=pipe,
            right_pipe=pipe,
            left_basis_map={key: np.eye(1)},
            right_basis_map={key: np.eye(1)},
            blocks={(key, key): np.array([[value]])},
            dtype=float,
        )

    singlet_svd = _single_channel_projection(singlet, 0.79).svd(full_matrices=False)
    doublet_svd = _single_channel_projection(doublet, 0.80).svd(full_matrices=False)

    reduced = truncate_reduced_svds(
        {singlet: singlet_svd, doublet: doublet_svd},
        max_bond=1,
        mode="reduced",
    )
    states = truncate_reduced_svds(
        {singlet: singlet_svd, doublet: doublet_svd},
        max_bond=1,
        mode="states",
    )

    assert reduced.kept == 1
    assert states.kept == 1
    assert tuple(reduced.singular_values_by_sector()) == (doublet,)
    assert tuple(states.singular_values_by_sector()) == (singlet,)


def test_combine_and_split_legs_round_trip():
    left = _charge_spin_sector(0, 0)
    phys_up = _charge_spin_sector(1, 1)
    phys_dn = _charge_spin_sector(1, 1)
    right = _charge_spin_sector(2, 0)
    singlet = _charge_spin_sector(2, 0)

    tensor = NonabelianTensor(
        data={
            (left, phys_up, phys_dn, right): np.arange(8.0).reshape(1, 2, 2, 2),
        },
        qns=[[left], [phys_up], [phys_dn], [right]],
        dirs=[-1, 1, 1, 1],
    )

    fusion_leg = FusionLeg.from_children(
        child_legs=(1, 2),
        child_sector_lists=(tuple(tensor.qns[1]), tuple(tensor.qns[2])),
        child_dirs=(tensor.dirs[1], tensor.dirs[2]),
        orientation=1,
        selected_channel=singlet,
    )

    combined = combine_legs(tensor, (1, 2), fusion_leg=fusion_leg)
    combined = NonabelianTensor(
        combined.data,
        combined.qns,
        combined.dirs,
        fusion_legs=combined.fusion_legs,
        metadata={},
    )
    recovered = split_legs(combined, 1)

    assert combined.rank == 3
    assert combined.fusion_legs[1].selected_channel == singlet
    assert combined.fusion_legs[1].pipe is not None
    np.testing.assert_allclose(
        recovered.data[(left, phys_up, phys_dn, right)],
        tensor.data[(left, phys_up, phys_dn, right)],
    )
