from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrg import (
    DMRG,
    SymmetryManager,
    _build_spatial_active_hamiltonian_matrix,
    _normalize_spatial_family_environment_backend,
    _normalize_spatial_native_p_grouping,
    build_mps_from_configs,
    build_spatial_mps_from_configs,
)
from pyqed.qchem.dmrg.spatial_mpo import build_spatial_block2_carrier_mpo
from pyqed.qchem.dmrg.backends.reduced import (
    build_spatial_complementary_operator_families,
    build_spatial_reduced_hamiltonian_mpo,
)
from pyqed.qchem.dmrg.spatial_terms import (
    accumulate_symbolic_term,
    merge_term_maps,
    spatial_local_ops,
    spatial_complementary_local_matrices,
    spatial_complementary_local_matrix,
    spatial_complementary_family_hamiltonian_term_map,
    spatial_complementary_family_term_maps,
    spatial_one_body_term_map,
    spatial_two_generator_family_term_map,
    spatial_two_body_term_map,
    spatial_two_body_spinfree_term_map,
)
from pyqed.qchem.mcscf.cocas import _fresh_casci_like
from pyqed.mps.mps import (
    AbelianComplementaryBoundaryActionTable,
    AbelianRenormalizedOperatorActionTable,
    AbelianSparseComplementaryBoundaryActionTable,
    dense_to_symmetric_mpo,
    multiply_S_V,
    multiply_U_S,
    sa_svd_symmetric,
    svd_symmetric,
)
from pyqed.mps.abelian_direct import AbelianSiteTensorData
from pyqed.mps.nonabelian import (
    AutoMPO,
    FullyReducedSpatialOrbitalSite,
    RankCoupledMPO,
    SpatialSpinFreeERIBuilder,
    physical_leg_from_spatial_orbital,
)
from pyqed.mps.symmetry import BlockTensor, QN
from pyqed.mps.nonabelian.models import (
    _dense_matrix_from_local_mpo,
    add_spatial_one_body_terms,
    add_spatial_two_generator_product_terms,
)
from pyqed.mps.nonabelian.renormalized import (
    ComplementaryFamilyRenormalizedOperatorTable,
)
from pyqed.mps.abelian_direct import (
    AbelianCompactBlockDataTable,
    AbelianCompactRenormalizedDataTable,
    AbelianContextualComponentStore,
    AbelianContextualDirectFamilyBuilder,
    AbelianContextualFamilyBuildOptions,
    AbelianCompositePackedDirectFamilyEntries,
    AbelianDenseBoundaryActionDataTable,
    AbelianDirectRoutePlan,
    AbelianGroupedRenormalizedDataTable,
    AbelianLocalActionPlan,
    AbelianLocalActionPlanCache,
    AbelianLocalVectorLayout,
    AbelianMovingEnvironmentTables,
    AbelianMovingEnvironmentFlatMatvec,
    AbelianNativeExactPatternComponentTable,
    AbelianNativeExactPatternOperatorTable,
    AbelianNativeGeneratorOperatorTable,
    AbelianNativePairBoundaryOperatorTable,
    AbelianOperatorFamilyPlan,
    AbelianPackedBoundaryTensor,
    AbelianPackedContextualBoundaryTable,
    AbelianPackedDirectFamilyEntries,
    AbelianPackedIdentityLocalEntry,
    AbelianPackedLocalGeneratorEntry,
    AbelianPackedLocalStateProto,
    AbelianPlannedPackedDirectFamilyEntries,
    AbelianPackedTensorViewCache,
    AbelianSameSidePRoutePlan,
    AbelianSameSidePRouteIdentityEntries,
    AbelianSameSidePBoundaryValueTable,
    AbelianSparseBoundaryActionDataTable,
    AbelianSpatialLocalOperatorBuilder,
    AbelianSymmetryAdapter,
    abelian_apply_block_preconditioner,
    abelian_apply_jacobi_preconditioner,
    abelian_axis_sector_dims,
    abelian_block_data_dtype,
    abelian_build_block_preconditioner_blocks,
    abelian_extend_projected_hamiltonian,
    abelian_flat_qchem_jacobi_diagonal,
    abelian_flatten_to_layout,
    abelian_layout_from_map,
    abelian_layout_offsets,
    abelian_merge_adjacent_site_tensors,
    abelian_merge_normalize_adjacent_site_tensors,
    abelian_merge_normalize_flatten_adjacent_site_tensors,
    abelian_merge_layout_tensor,
    abelian_packed_local_action_apply_clean,
    abelian_packed_local_action_matches_reference,
    abelian_packed_local_action_probe_reference,
    abelian_project_block_data_to_layout,
    abelian_project_tensor_to_layout_with_stats,
    abelian_remap_flat_layout,
    abelian_safe_two_site_layout_map,
    abelian_sector_signature,
    abelian_lowest_ritz_state,
    abelian_multiply_s_v_data,
    abelian_multiply_u_s_data,
    abelian_normalize_flat_vector,
    abelian_orthogonalize_candidate,
    abelian_restart_basis_from_vector,
    abelian_state_averaged_two_site_svd_from_permuted_data,
    abelian_site_tensors_from_split,
    abelian_split_state_averaged_two_site_svd_data,
    abelian_split_flat_two_site_svd_data,
    abelian_split_two_site_svd_data,
    _abelian_merge_adjacent_site_tensors_python,
    _abelian_split_two_site_svd_data_python,
    abelian_truncate_layout_map_by_norm,
    abelian_two_site_svd_from_permuted_data,
    abelian_two_site_mps_flow_valid,
    _abelian_two_site_svd_from_permuted_data_python,
    abelian_unflatten_data_from_layout,
    abelian_zero_data_from_layout,
    advance_abelian_packed_left_identity_boundary,
    advance_abelian_packed_left_boundary,
    advance_abelian_packed_right_identity_boundary,
    advance_abelian_packed_right_boundary,
    abelian_packed_tensor_axis_map,
    abelian_packed_tensor_axis_qns,
    abelian_packed_tensor_items,
    abelian_generator_owner_from_support,
    abelian_generator_region_from_support,
    apply_abelian_packed_local_action_entries,
    compare_abelian_packed_boundary_tensors,
    compose_abelian_packed_boundary_operators,
    contextual_boundary_keys,
    conjugate_abelian_packed_boundary_tensor,
    filter_abelian_packed_boundary_tensor_axis,
    is_abelian_packed_boundary_tensor,
    make_abelian_packed_initial_left_environment,
    make_abelian_packed_initial_right_environment,
    make_abelian_packed_local_generator_pair,
    make_abelian_packed_site_operator_from_left,
    make_abelian_packed_site_operator_from_right,
    make_contextual_family_records,
    merge_abelian_same_side_p_route_plan,
    native_p_owner_records,
    pack_abelian_boundary_tensor,
    packed_same_side_p_product_correction,
    scale_abelian_boundary_tensor,
    sum_abelian_packed_boundary_terms,
    tensordot_abelian_packed_boundary_tensors,
    transpose_abelian_packed_boundary_tensor,
    unpack_abelian_packed_boundary_tensor,
)


def _kron_all(operators):
    out = np.asarray(operators[0], dtype=complex)
    for operator in operators[1:]:
        out = np.kron(out, np.asarray(operator, dtype=complex))
    return out


def _dense_from_spatial_term_map(term_map, nsites):
    ops = spatial_local_ops()
    ident = ops["I"]
    dense = np.zeros((4**nsites, 4**nsites), dtype=complex)
    for (symbol, dofs), factor in term_map.items():
        local = [ident.copy() for _ in range(nsites)]
        for piece, site in zip(symbol.split(), dofs):
            local[site] = ops[piece]
        dense += factor * _kron_all(local)
    return dense


def test_spatial_family_environment_backend_block2_aliases_use_family_mpos():
    assert _normalize_spatial_family_environment_backend(None) == "block2_table"
    assert _normalize_spatial_family_environment_backend("block2") == "block2"
    assert (
        _normalize_spatial_family_environment_backend("operator_table")
        == "block2_table"
    )
    assert (
        _normalize_spatial_family_environment_backend("generator_table")
        == "generator_table"
    )
    assert _normalize_spatial_family_environment_backend("autompo") == "block2"
    assert (
        _normalize_spatial_family_environment_backend("native_generators")
        == "block2_native"
    )
    assert (
        _normalize_spatial_family_environment_backend("adaptive_block2")
        == "block2_adaptive"
    )
    assert (
        _normalize_spatial_family_environment_backend("renormalized_generators")
        == "block2"
    )
    assert _normalize_spatial_family_environment_backend("none") == "none"


def test_spatial_native_p_grouping_aliases():
    assert _normalize_spatial_native_p_grouping(None) == "first_site_order"
    assert _normalize_spatial_native_p_grouping("balanced") == "first_site_order"
    assert _normalize_spatial_native_p_grouping("all") == "none"
    assert (
        _normalize_spatial_native_p_grouping("first_two_sites")
        == "first_two_site_order"
    )
    assert _normalize_spatial_native_p_grouping("full_site_order") == "site_order"


def test_qchem_spatial_abelian_auto_defaults_to_block2_carrier_when_possible():
    class Mol:
        spin = 0

    class MF:
        nelec = 2
        mol = Mol()

    dmrg = DMRG(MF(), ncas=2, nelecas=2, D=4, site="spatial", symmetry="sz")

    assert dmrg.spatial_abelian_mpo == "spatial"
    assert dmrg.spatial_family_environment_backend == "block2_table"
    assert dmrg.spatial_block2_table_p_split_metric == "auto"
    assert dmrg.spatial_block2_table_p_split_groups == "auto"
    assert dmrg.spatial_block2_table_native_p is False
    assert dmrg.spatial_complementary_payload_tensor_matvec is True
    assert dmrg.spatial_precontracted_family_environment is True
    assert dmrg.spatial_boundary_table_max_dim == 32
    assert dmrg.spatial_exact_component_compression_policy == "auto"
    assert dmrg.spatial_exact_component_compression_validate is True
    assert dmrg.spatial_exact_component_compression_validation_vectors == 1
    assert dmrg.spatial_exact_component_compression_min_reduction == 1
    assert dmrg.spatial_exact_component_compression_max_group_size == 64
    assert dmrg.spatial_enable_cpp_boundary_p is True
    assert dmrg.spatial_validate_cpp_boundary_p is False
    assert dmrg.spatial_cpp_boundary_p_validation_policy == "off"
    assert dmrg.spatial_direct_operator_batch_min_entries == 2
    assert dmrg.spatial_reduced_mpo is False
    assert dmrg._can_use_spatial_block2_carrier() is True

    explicit_grouped = DMRG(
        MF(),
        ncas=2,
        nelecas=2,
        D=4,
        site="spatial",
        symmetry="sz",
        spatial_abelian_mpo="grouped",
    )
    assert explicit_grouped.spatial_abelian_mpo == "grouped"

    dense = DMRG(MF(), ncas=2, nelecas=2, D=4, site="spatial", symmetry=None)
    assert dense.spatial_abelian_mpo == "grouped"
    assert dense._can_use_spatial_block2_carrier() is False


def test_spatial_block2_carrier_is_d4_scaffold_not_grouped_spin_orbital():
    carrier = build_spatial_block2_carrier_mpo(3)

    assert carrier.info["representation"] == "spatial_block2_table_carrier_mpo"
    assert carrier.info["replaces_grouped_spin_orbital_carrier"] is True
    assert [factor.shape for factor in carrier.factors] == [(1, 1, 4, 4)] * 3
    for factor in carrier.factors:
        np.testing.assert_allclose(factor[0, 0], np.eye(4))


def test_spatial_block2_carrier_uses_family_sweep_energy_for_qchem():
    geom = "; ".join(f"H 0 0 {1.8 * i}" for i in range(4))
    mol = Molecule(atom=geom, unit="b", basis="sto3g")
    mol.build(driver="builtin", eri="factors")
    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)

    solver = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=30,
        init_guess="cid",
        site="spatial",
        symmetry=("charge", "sz"),
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        spatial_abelian_mpo="spatial",
        dmrg_performance="packed-compiled-fast",
    )

    low = solver.run(
        nsweeps=8,
        sweep_tol=1.0e-8,
        davidson_tol=1.0e-9,
        davidson_max_iter=64,
        noise=0.0,
    )

    assert solver.energy == pytest.approx(-2.1754111431673824, abs=1.0e-8)
    assert low.e_tot == pytest.approx(low.sweep_history[-1]["energy"], abs=1.0e-12)
    info = solver._active_integral_build_info
    assert info["representation"] == "spatial_block2_table_carrier_mpo"
    assert info["carrier_only_sweep_energy_final"] is True
    assert info["carrier_only_forced_family_flat_csr"] is True
    resolved = info["resolved_abelian_matvec_options"]
    assert resolved["packed_local_flat_matvec"] is False
    assert resolved["packed_local_family_flat_matvec"] is True
    assert resolved["packed_local_family_flat_matvec_max_dim"] == 10**18
    assert resolved["packed_local_flat_preconditioner"] is False
    family_flat_calls = 0
    carrier_flat_calls = 0
    family_flat_backends = set()
    family_flat_storage = set()
    renormalized_table_calls = 0
    for row in low.sweep_history:
        for update in row.get("updates", ()):
            profile = update.get("matvec_profile") or {}
            family_flat = profile.get("packed_flat_complementary_family_action") or {}
            family_flat_calls += int(
                family_flat.get("calls")
                or 0
            )
            backend = family_flat.get("compiled_direct_matvec_backend")
            if backend is not None:
                family_flat_backends.add(str(backend))
            storage = family_flat.get("renormalized_operator_table_storage")
            if storage is not None:
                family_flat_storage.add(str(storage))
            renormalized_table_calls += int(
                family_flat.get("renormalized_operator_table_calls")
                or 0
            )
            carrier_flat_calls += int(
                (
                    profile.get("packed_flat_batched_compact_matrix_chain")
                    or {}
                ).get("calls")
                or 0
            )
    assert family_flat_calls > 0
    assert renormalized_table_calls > 0
    assert family_flat_backends == {"renormalized_table"}
    assert family_flat_storage <= {
        "renormalized_operator_table",
        "renormalized_operator_block_matrix_table",
        "renormalized_operator_block_sparse_table",
        "cpp_grouped_renormalized_table_dense",
        "cpp_grouped_renormalized_table_sparse",
    }
    assert family_flat_storage
    assert carrier_flat_calls == 0


def test_spatial_block2_carrier_fused_compact_chain_projects_plan():
    geom = "; ".join(f"H 0 0 {1.8 * i}" for i in range(4))
    mol = Molecule(atom=geom, unit="b", basis="sto3g")
    mol.build(driver="builtin", eri="factors")
    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)

    solver = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=30,
        init_guess="cid",
        site="spatial",
        symmetry=("charge", "sz"),
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        spatial_abelian_mpo="spatial",
        dmrg_performance="packed-compiled-fast",
    )

    low = solver.run(
        nsweeps=8,
        sweep_tol=1.0e-8,
        davidson_tol=1.0e-9,
        davidson_max_iter=64,
        noise=0.0,
        abelian_matvec_options={
            "packed_local_family_flat_direct_matvec": True,
            "packed_local_family_flat_direct_matvec_backend": "fused_compact_chain",
        },
    )

    assert solver.energy == pytest.approx(-2.1754111431673824, abs=1.0e-8)

    family_profiles = []
    compact_profiles = []
    for row in low.sweep_history:
        for update in row.get("updates", ()):
            profile = update.get("matvec_profile") or {}
            family_profiles.append(
                profile.get("packed_flat_complementary_family_action") or {}
            )
            compact_profiles.append(
                profile.get("packed_flat_batched_compact_matrix_chain") or {}
            )

    family_profile = next(
        profile
        for profile in reversed(family_profiles)
        if profile.get("fused_compact_chain_calls")
    )
    assert family_profile["compiled_direct_matvec_backend"] == "fused_compact_chain"
    assert family_profile["last"]["source"] == "direct_compiled_named_family_matvec"

    compact_last = next(
        profile.get("last")
        for profile in reversed(compact_profiles)
        if (profile.get("last") or {}).get("projected_plan")
    )
    assert compact_last["project_output"] is True
    assert compact_last["projected_plan"] is True
    assert compact_last["projected_output_blocks"] == 0
    assert compact_last["kept_t3_entries"] < compact_last["full_t3_entries"]


@pytest.mark.parametrize(
    "backend",
    ("grouped_compiled", "renormalized_table", "block2_like"),
)
def test_spatial_block2_carrier_grouped_matvec_backends_match_qchem(backend):
    from pyqed.mps import packed_cython

    if (
        not getattr(packed_cython, "CYTHON_AVAILABLE", False)
        or getattr(packed_cython, "direct_operator_groups_matvec", None) is None
    ):
        pytest.skip("optional packed Cython grouped matvec is unavailable")

    geom = "; ".join(f"H 0 0 {1.8 * i}" for i in range(4))
    mol = Molecule(atom=geom, unit="b", basis="sto3g")
    mol.build(driver="builtin", eri="factors")
    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)

    solver = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=30,
        init_guess="cid",
        site="spatial",
        symmetry=("charge", "sz"),
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        spatial_abelian_mpo="spatial",
        dmrg_performance="packed-compiled-fast",
    )

    low = solver.run(
        nsweeps=8,
        sweep_tol=1.0e-8,
        davidson_tol=1.0e-9,
        davidson_max_iter=64,
        noise=0.0,
        abelian_matvec_options={
            "packed_local_family_flat_direct_matvec": True,
            "packed_local_family_flat_direct_matvec_backend": backend,
        },
    )

    assert solver.energy == pytest.approx(-2.1754111431673824, abs=1.0e-8)

    family_profile = None
    moving_profile = None
    for row in low.sweep_history:
        for update in row.get("updates", ()):
            profile = update.get("matvec_profile") or {}
            moving = profile.get("moving_environment") or {}
            if moving:
                moving_profile = moving
            family = profile.get("packed_flat_complementary_family_action") or {}
            expected_backend = (
                "renormalized_table" if backend == "block2_like" else backend
            )
            if family.get("compiled_direct_matvec_backend") == expected_backend:
                family_profile = family
    assert moving_profile is not None
    assert moving_profile["local_operator_builds"] > 0
    assert family_profile is not None
    assert family_profile["compiled_direct_matvec_groups"] > 0
    assert family_profile["compiled_direct_matvec_group_channels"] > 0
    if backend in {"renormalized_table", "block2_like"}:
        assert family_profile["renormalized_operator_table_calls"] > 0
        assert family_profile["renormalized_operator_table_storage"] in {
            "renormalized_operator_table",
            "renormalized_operator_block_matrix_table",
            "renormalized_operator_block_sparse_table",
        }
        if (
            family_profile["renormalized_operator_table_storage"]
            in {
                "renormalized_operator_block_matrix_table",
                "renormalized_operator_block_sparse_table",
            }
        ):
            assert family_profile["renormalized_operator_table_block_matrices_last"] > 0
            assert (
                family_profile[
                    "renormalized_operator_table_block_matrix_elements_last"
                ]
                > 0
            )
        if (
            family_profile["renormalized_operator_table_storage"]
            == "renormalized_operator_block_sparse_table"
        ):
            assert family_profile["renormalized_operator_table_block_sparse_nnz_last"] > 0


def test_spatial_moving_environment_matches_old_local_operator_energy():
    geom = "; ".join(f"H 0 0 {1.8 * i}" for i in range(4))
    mol = Molecule(atom=geom, unit="b", basis="sto3g")
    mol.build(driver="builtin", eri="factors")
    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)

    def _run(use_moving_environment):
        solver = DMRG(
            mf,
            ncas=4,
            nelecas=4,
            D=12,
            init_guess="cid",
            site="spatial",
            symmetry=("charge", "sz"),
            spin=0,
            verbose=0,
            integral_backend="cholesky",
            spatial_abelian_mpo="spatial",
            dmrg_performance="packed-compiled-fast",
        )
        low = solver.run(
            nsweeps=2,
            sweep_tol=1.0e-8,
            davidson_tol=1.0e-9,
            davidson_max_iter=64,
            noise=0.0,
            abelian_matvec_options={
                "moving_environment": bool(use_moving_environment),
                "packed_local_family_flat_direct_matvec": True,
                "packed_local_family_flat_direct_matvec_backend": "renormalized_table",
            },
        )
        return solver.energy, low

    old_energy, old_low = _run(False)
    moving_energy, moving_low = _run(True)
    assert moving_energy == pytest.approx(old_energy, abs=1.0e-10)
    assert not any(
        (update.get("matvec_profile") or {}).get("moving_environment")
        for row in old_low.sweep_history
        for update in row.get("updates", ())
    )
    assert any(
        (update.get("matvec_profile") or {}).get("moving_environment")
        for row in moving_low.sweep_history
        for update in row.get("updates", ())
    )
    assert any(
        (
            (update.get("matvec_profile") or {})
            .get("moving_environment", {})
            .get("compiled_flat_matvec_calls", 0)
        )
        for row in moving_low.sweep_history
        for update in row.get("updates", ())
    )


def test_spatial_moving_environment_cpp_davidson_matches_python():
    from pyqed.mps import cpp_davidson

    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson backend is unavailable")

    geom = "; ".join(f"H 0 0 {1.8 * i}" for i in range(4))
    mol = Molecule(atom=geom, unit="b", basis="sto3g")
    mol.build(driver="builtin", eri="factors")
    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)

    def _run(use_cpp):
        solver = DMRG(
            mf,
            ncas=4,
            nelecas=4,
            D=12,
            init_guess="cid",
            site="spatial",
            symmetry=("charge", "sz"),
            spin=0,
            verbose=0,
            integral_backend="cholesky",
            spatial_abelian_mpo="spatial",
            dmrg_performance="packed-compiled-fast",
        )
        low = solver.run(
            nsweeps=2,
            sweep_tol=1.0e-8,
            davidson_tol=1.0e-9,
            davidson_max_iter=64,
            noise=0.0,
            abelian_matvec_options={
                "moving_environment_cpp_davidson": bool(use_cpp),
            },
        )
        return solver.energy, low

    python_energy, _python_low = _run(False)
    cpp_energy, cpp_low = _run(True)
    assert cpp_energy == pytest.approx(python_energy, abs=1.0e-10)
    assert any(
        ((update.get("matvec_profile") or {}).get("packed_local_davidson") or {}).get(
            "cpp_davidson"
        )
        for row in cpp_low.sweep_history
        for update in row.get("updates", ())
    )
    assert not any(
        ((update.get("matvec_profile") or {}).get("moving_environment") or {}).get(
            "cpp_davidson_failures"
        )
        for row in cpp_low.sweep_history
        for update in row.get("updates", ())
    )


def test_spatial_moving_environment_cpp_matvec_matches_cython_matvec():
    from pyqed.mps import cpp_davidson

    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "block_table_matvec", None) is None
    ):
        pytest.skip("optional C++ block-table matvec backend is unavailable")

    geom = "; ".join(f"H 0 0 {1.8 * i}" for i in range(4))
    mol = Molecule(atom=geom, unit="b", basis="sto3g")
    mol.build(driver="builtin", eri="factors")
    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)

    def _run(use_cpp_matvec):
        solver = DMRG(
            mf,
            ncas=4,
            nelecas=4,
            D=12,
            init_guess="cid",
            site="spatial",
            symmetry=("charge", "sz"),
            spin=0,
            verbose=0,
            integral_backend="cholesky",
            spatial_abelian_mpo="spatial",
            dmrg_performance="packed-compiled-fast",
        )
        low = solver.run(
            nsweeps=2,
            sweep_tol=1.0e-8,
            davidson_tol=1.0e-9,
            davidson_max_iter=64,
            noise=0.0,
            abelian_matvec_options={
                "moving_environment_cpp_matvec": bool(use_cpp_matvec),
                "moving_environment_cpp_davidson": False,
            },
        )
        return solver.energy, low

    cython_energy, _cython_low = _run(False)
    cpp_energy, cpp_low = _run(True)
    assert cpp_energy == pytest.approx(cython_energy, abs=1.0e-10)
    assert any(
        ((update.get("matvec_profile") or {}).get("moving_environment") or {}).get(
            "cpp_block_matvec_calls"
        )
        for row in cpp_low.sweep_history
        for update in row.get("updates", ())
    )
    assert not any(
        ((update.get("matvec_profile") or {}).get("moving_environment") or {}).get(
            "cpp_block_matvec_failures"
        )
        for row in cpp_low.sweep_history
        for update in row.get("updates", ())
    )


def test_block2_table_backend_uses_family_mpos_not_direct_term_maps():
    h1 = np.array([[0.1, 0.02], [0.02, -0.1]])
    eri = np.zeros((2, 2, 2, 2, 2, 2))
    eri[:, :, 0, 0, 1, 1] = 0.3
    families = build_spatial_complementary_operator_families(h1, eri, cutoff=1.0e-12)
    term_maps = spatial_complementary_family_term_maps(families, cutoff=1.0e-12)

    dmrg = DMRG.__new__(DMRG)
    dmrg.ncas = 2
    dmrg.spatial_family_environment_backend = "block2_table"
    dmrg.spatial_abelian_symbolic_algo = "Hopcroft-Karp"
    dmrg.spatial_native_p_grouping = "first_site_order"

    family_mpos, family_info = DMRG._build_spatial_family_environment_mpos(
        dmrg,
        families,
        term_maps,
        cutoff=1.0e-12,
    )
    DMRG._expose_spatial_family_environment(
        dmrg,
        families,
        term_maps,
        family_mpos,
        expose_direct_terms=True,
    )

    p_mpos = {name for name in family_mpos if name.split(":", 1)[0] == "P"}
    assert "R" in family_mpos
    assert p_mpos
    assert "P" in family_info
    if p_mpos != {"P"}:
        assert family_info["P"]["source"] == "symbolic_spatial_term_map_split_summary"
        assert set(family_info["P"]["split_family_names"]) == p_mpos
    assert dmrg.complementary_operator_mpos is family_mpos
    assert dmrg.complementary_operator_term_maps is None


def test_block2_table_native_p_replaces_only_p_family_mpo():
    h1 = np.array([[0.1, 0.02], [0.02, -0.1]])
    eri = np.zeros((2, 2, 2, 2, 2, 2))
    eri[:, :, 0, 0, 1, 1] = 0.3
    families = build_spatial_complementary_operator_families(h1, eri, cutoff=1.0e-12)
    term_maps = spatial_complementary_family_term_maps(families, cutoff=1.0e-12)

    dmrg = DMRG.__new__(DMRG)
    dmrg.ncas = 2
    dmrg.spatial_family_environment_backend = "block2_table"
    dmrg.spatial_abelian_symbolic_algo = "Hopcroft-Karp"
    dmrg.spatial_native_p_grouping = "first_site_order"
    dmrg.spatial_block2_table_native_p = True

    family_mpos, family_info = DMRG._build_spatial_family_environment_mpos(
        dmrg,
        families,
        term_maps,
        cutoff=1.0e-12,
    )
    DMRG._expose_spatial_family_environment(
        dmrg,
        families,
        term_maps,
        family_mpos,
        expose_direct_terms=True,
    )

    assert "R" in family_mpos
    assert not any(name.split(":", 1)[0] == "P" for name in family_mpos)
    assert family_info["P"]["source"] == "native_direct_generator_table"
    assert family_info["P"]["symbolic_mpo_replaced"] is True
    assert dmrg.complementary_operator_mpos is family_mpos
    assert dmrg.complementary_operator_term_maps is None
    assert set(dmrg.complementary_operator_generator_entries) == {"P"}
    assert dmrg.complementary_operator_generator_entries["P"] == families["P"].entries


def test_generator_table_backend_exposes_generator_entries_not_symbolic_mpos():
    h1 = np.array([[0.1, 0.02], [0.02, -0.1]])
    eri = np.zeros((2, 2, 2, 2, 2, 2))
    eri[:, :, 0, 0, 1, 1] = 0.3
    families = build_spatial_complementary_operator_families(h1, eri, cutoff=1.0e-12)
    term_maps = spatial_complementary_family_term_maps(families, cutoff=1.0e-12)

    dmrg = DMRG.__new__(DMRG)
    dmrg.ncas = 2
    dmrg.spatial_family_environment_backend = "generator_table"
    dmrg.spatial_abelian_symbolic_algo = "Hopcroft-Karp"
    dmrg.spatial_native_p_grouping = "first_site_order"

    family_mpos, family_info = DMRG._build_spatial_family_environment_mpos(
        dmrg,
        families,
        term_maps,
        cutoff=1.0e-12,
    )
    DMRG._expose_spatial_family_environment(
        dmrg,
        families,
        term_maps,
        family_mpos,
        expose_direct_terms=False,
    )

    assert family_mpos == {}
    assert family_info["R"]["source"] == "native_generator_entries"
    assert family_info["P"]["source"] == "native_generator_entries"
    assert dmrg.complementary_operator_mpos is None
    assert dmrg.complementary_operator_term_maps is None
    assert set(dmrg.complementary_operator_generator_entries) == {"R", "P"}
    assert dmrg.complementary_operator_generator_entries["R"]
    assert dmrg.complementary_operator_generator_entries["P"]


def test_generator_table_family_data_skips_symbolic_term_maps():
    class Mol:
        spin = 0

    class MF:
        nelec = 2
        mol = Mol()

    h1 = np.array([[0.1, 0.02], [0.02, -0.1]])
    eri_spatial = np.zeros((2, 2, 2, 2))
    eri_spatial[0, 0, 1, 1] = 0.3
    h2 = np.stack(
        (
            np.stack((eri_spatial, eri_spatial.copy())),
            np.stack((eri_spatial.copy(), eri_spatial.copy())),
        )
    )
    dmrg = DMRG(
        MF(),
        ncas=2,
        nelecas=2,
        D=4,
        site="spatial",
        symmetry=("charge", "sz"),
        spatial_family_environment_backend="generator_table",
    )

    family_data = dmrg._build_spatial_complementary_family_data(
        [h1, h1],
        h2,
        cutoff=1.0e-12,
    )
    timings = family_data["timings"]

    assert family_data["term_maps"] == {}
    assert timings["family_term_map_build_s"] == 0.0
    assert timings["family_term_map_backend_actual"] == "skipped"
    assert timings["family_generator_entry_counts"]["R"] > 0
    assert timings["family_generator_entry_counts"]["P"] > 0


def test_qchem_spatial_initial_guess_builder_can_emit_native_site_data():
    sym_mgr = SymmetryManager(["charge", "sz"])

    legacy = build_spatial_mps_from_configs(
        [((1, 0), 1.0)],
        sym_mgr,
        2,
        noise_scale=0.0,
    )
    native = build_spatial_mps_from_configs(
        [((1, 0), 1.0)],
        sym_mgr,
        2,
        noise_scale=0.0,
        native_site_storage=True,
    )

    assert all(isinstance(site, BlockTensor) for site in legacy)
    assert all(isinstance(site, AbelianSiteTensorData) for site in native)
    assert not any(isinstance(site, BlockTensor) for site in native)
    assert [site.qns for site in native] == [
        tuple(tuple(axis) for axis in site.qns)
        for site in legacy
    ]
    for native_site, legacy_site in zip(native, legacy):
        assert set(native_site.data) == set(legacy_site.data)
        for key, block in legacy_site.data.items():
            np.testing.assert_allclose(native_site.data[key], block, atol=1.0e-12)


def test_qchem_spin_orbital_initial_guess_builder_can_emit_native_site_data():
    sym_mgr = SymmetryManager(["charge", "sz"])
    configs = [((1, 0, 0, 1), 1.0)]

    legacy = build_mps_from_configs(configs, sym_mgr, 4, noise_scale=0.0)
    native = build_mps_from_configs(
        configs,
        sym_mgr,
        4,
        noise_scale=0.0,
        native_site_storage=True,
    )

    assert all(isinstance(site, BlockTensor) for site in legacy)
    assert all(isinstance(site, AbelianSiteTensorData) for site in native)
    assert not any(isinstance(site, BlockTensor) for site in native)
    for native_site, legacy_site in zip(native, legacy):
        assert set(native_site.data) == set(legacy_site.data)
        for key, block in legacy_site.data.items():
            np.testing.assert_allclose(native_site.data[key], block, atol=1.0e-12)


def test_qchem_default_symmetric_initial_guess_uses_native_storage():
    dmrg = DMRG.__new__(DMRG)
    dmrg.site = "spatial"
    dmrg.ncas = 2
    dmrg.nelecas = 2
    dmrg.spin = 0
    dmrg.init_guess = "hf"
    dmrg.verbose = 0
    dmrg.sym_mgr = SymmetryManager(["charge", "sz"])

    guess = dmrg._resolve_initial_guess(
        use_symmetry=True,
        native_site_storage=True,
    )

    assert all(isinstance(site, AbelianSiteTensorData) for site in guess)
    assert not any(isinstance(site, BlockTensor) for site in guess)


def test_qchem_dmrg_main_does_not_expose_legacy_blocktensor_surface():
    import importlib

    qchem_dmrg_module = importlib.import_module("pyqed.qchem.dmrg.dmrg")
    storage_module = importlib.import_module("pyqed.mps.abelian_storage")

    assert not hasattr(qchem_dmrg_module, "BlockTensor")
    assert not hasattr(qchem_dmrg_module, "SYMMETRY_AVAILABLE")
    assert (
        qchem_dmrg_module.make_abelian_site_tensor
        is storage_module.make_abelian_site_tensor
    )


def test_dense_to_symmetric_mpo_can_emit_native_site_data():
    sym_mgr = SymmetryManager(["charge"])
    site_qn_maps = [
        {
            0: sym_mgr.get_phys_qn(0, "emp"),
            1: sym_mgr.get_phys_qn(0, "occ"),
        }
    ]
    W = np.eye(2, dtype=complex).reshape(1, 1, 2, 2)

    legacy = dense_to_symmetric_mpo([W], site_qn_maps)
    native = dense_to_symmetric_mpo(
        [W],
        site_qn_maps,
        native_site_storage=True,
    )

    assert isinstance(legacy[0], BlockTensor)
    assert isinstance(native[0], AbelianSiteTensorData)
    assert not isinstance(native[0], BlockTensor)
    assert native[0].qns == tuple(tuple(axis) for axis in legacy[0].qns)
    assert native[0].dirs == tuple(legacy[0].dirs)
    assert set(native[0].data) == set(legacy[0].data)
    for key, block in legacy[0].data.items():
        np.testing.assert_allclose(native[0].data[key], block, atol=1.0e-12)


def test_qchem_default_run_passes_native_mpo_and_guess_to_tensor_dmrg(monkeypatch):
    import importlib

    qchem_dmrg_module = importlib.import_module("pyqed.qchem.dmrg.dmrg")
    captured = {}

    class DummyState:
        def copy(self):
            return self

    class FakeTensorDMRG:
        def __init__(self, H, **kwargs):
            captured["H"] = H
            captured["kwargs"] = kwargs
            self.abelian_matvec_options = {"native_site_storage": True}
            self.ground_state = DummyState()
            self.e_tot = 0.0
            self.converged = True
            self.sweep_history = []

        def run(self):
            captured["run_called"] = True
            return self

    monkeypatch.setattr(qchem_dmrg_module, "TensorDMRG", FakeTensorDMRG)

    dmrg = DMRG.__new__(DMRG)
    dmrg.verbose = 0
    dmrg.D = 4
    dmrg.tol = 1.0e-6
    dmrg.dmrg_performance = "block2-like"
    dmrg.abelian_matvec_options = None
    dmrg.saved_symmetry_list = ["charge"]
    dmrg.symmetry = ["charge"]
    dmrg.site = "spatial"
    dmrg.site_basis = dmrg.orbital_layout = "spatial"
    dmrg.spatial_site_basis = "canonical"
    dmrg.ncas = 2
    dmrg.nelecas = 2
    dmrg.spin = 0
    dmrg.init_guess = "hf"
    dmrg.H_raw = object()
    ident = np.eye(4, dtype=complex)
    dmrg.H = [ident.reshape(1, 1, 4, 4), ident.reshape(1, 1, 4, 4)]
    dmrg._symmetric_mpo_cache = {}
    dmrg._active_integral_build_info = {"build_timings": {}}
    dmrg.complementary_operator_mpos = None
    dmrg.complementary_operator_term_maps = None
    dmrg.complementary_operator_generator_entries = None
    dmrg.complementary_operators = None
    dmrg.e_core = 0.0
    dmrg.spin_purification = False
    dmrg.mf = SimpleNamespace(e_tot=0.0, mol=SimpleNamespace(spin=0))

    dmrg.run(nsweeps=1, symmetry="charge")

    assert captured["run_called"] is True
    assert captured["kwargs"]["nsweeps"] == 2
    assert captured["kwargs"]["converge_on_full_sweeps"] is True
    assert all(isinstance(site, AbelianSiteTensorData) for site in captured["H"])
    assert not any(isinstance(site, BlockTensor) for site in captured["H"])
    init_guess = captured["kwargs"]["init_guess"]
    assert all(isinstance(site, AbelianSiteTensorData) for site in init_guess)
    assert not any(isinstance(site, BlockTensor) for site in init_guess)
    assert dmrg._active_integral_build_info["native_symmetric_mpo_storage"] is True
    assert dmrg._active_integral_build_info["native_initial_guess_storage"] is True


def test_qchem_global_symmetric_mpo_cache_reuses_native_family_mpos(monkeypatch):
    import importlib

    qchem_dmrg_module = importlib.import_module("pyqed.qchem.dmrg.dmrg")
    qchem_dmrg_module._GLOBAL_SYMMETRIC_MPO_CACHE.clear()
    captured = []

    class DummyState:
        def copy(self):
            return self

    class FakeTensorDMRG:
        def __init__(self, H, **kwargs):
            captured.append({"H": H, "kwargs": kwargs})
            self.abelian_matvec_options = {"native_site_storage": True}
            self.ground_state = DummyState()
            self.e_tot = 0.0
            self.converged = True
            self.sweep_history = []

        def run(self):
            return self

    monkeypatch.setattr(qchem_dmrg_module, "TensorDMRG", FakeTensorDMRG)

    def make_dmrg():
        dmrg = DMRG.__new__(DMRG)
        dmrg.verbose = 0
        dmrg.D = 4
        dmrg.tol = 1.0e-6
        dmrg.dmrg_performance = "block2-like"
        dmrg.abelian_matvec_options = None
        dmrg.saved_symmetry_list = ["charge"]
        dmrg.symmetry = ["charge"]
        dmrg.site = "spatial"
        dmrg.site_basis = dmrg.orbital_layout = "spatial"
        dmrg.spatial_site_basis = "canonical"
        dmrg.ncas = 2
        dmrg.nelecas = 2
        dmrg.spin = 0
        dmrg.init_guess = "hf"
        dmrg.H_raw = object()
        ident = np.eye(4, dtype=complex)
        dense_mpo = [ident.reshape(1, 1, 4, 4), ident.reshape(1, 1, 4, 4)]
        dmrg.H = dense_mpo
        dmrg._hamiltonian_mpo_cache_key = ("unit-test-native-cache",)
        dmrg._symmetric_mpo_cache = {}
        dmrg._active_integral_build_info = {"build_timings": {}}
        dmrg.complementary_operator_mpos = {"R": dense_mpo}
        dmrg.complementary_operator_term_maps = None
        dmrg.complementary_operator_generator_entries = None
        dmrg.complementary_operators = None
        dmrg.e_core = 0.0
        dmrg.spin_purification = False
        dmrg.mf = SimpleNamespace(e_tot=0.0, mol=SimpleNamespace(spin=0))
        return dmrg

    first = make_dmrg()
    second = make_dmrg()

    first.run(nsweeps=1, symmetry="charge")
    second.run(nsweeps=1, symmetry="charge")

    first_timings = first._active_integral_build_info["build_timings"]
    second_timings = second._active_integral_build_info["build_timings"]
    assert first_timings["symmetric_mpo_global_cache_stores"] == 1
    assert second_timings["symmetric_mpo_global_cache_hits"] == 1
    assert "symmetric_family_convert_s" not in second_timings
    assert len(captured) == 2
    first_family = captured[0]["kwargs"]["complementary_operator_mpos"]["R"]
    second_family = captured[1]["kwargs"]["complementary_operator_mpos"]["R"]
    assert first_family is second_family


def test_native_exact_pattern_table_is_exposed_in_family_stats():
    table = AbelianNativeExactPatternOperatorTable(side="left", bond=1)
    block_like = SimpleNamespace(data={("q0",): np.ones((2, 3))})
    table.put((("I",), "C"), (block_like,), family_name="P")
    component_table = AbelianNativeExactPatternComponentTable(bond=1)
    component_table.put_family_records(
        "P",
        (((("I",), "C", "D", ("I",), 0.5 + 0.0j)),),
    )
    component_table.put_family("P", ((block_like,),))
    pair_table = AbelianNativePairBoundaryOperatorTable(side="center", bond=1)
    pair_table.add((0, 0, 1, 1), ((block_like,),))

    family_table = ComplementaryFamilyRenormalizedOperatorTable(
        side="left",
        bond=1,
        family_blocks={},
    )
    family_table.put_native_operator_table(("exact", 1), table)
    family_table.put_native_operator_table(("components", 1), component_table)
    family_table.put_native_operator_table(("pair", 1), pair_table)
    stats = family_table.stats

    assert stats["native_operator_tables"] == 3
    assert stats["native_operator_table_stored_elements"] == 18
    nested = {
        value["kind"]: value
        for value in stats["native_operator_table_stats"].values()
    }
    assert (
        nested["abelian_native_exact_pattern_operator_table"]["family_counts"]
        == {"P": 1}
    )
    assert (
        nested["abelian_native_exact_pattern_component_table"]["family_counts"]
        == {"P": 1}
    )
    assert (
        nested["abelian_native_exact_pattern_component_table"]["record_counts"]
        == {"P": 1}
    )
    assert nested["abelian_native_pair_boundary_operator_table"]["n_terms"] == 1


def test_abelian_native_tables_report_packed_storage_stats():
    op = AbelianPackedBoundaryTensor(
        ((0, 0, 0), (0, 1, 1)),
        (
            np.ones((1, 2, 3), dtype=np.complex128),
            2.0 * np.ones((1, 2, 2), dtype=np.complex128),
        ),
        dirs=(-1, 1, -1),
        qns=((0,), (0, 1), (0, 1)),
    )
    gen_table = AbelianNativeGeneratorOperatorTable(
        side="left",
        bond=2,
        operators={(0, 1): op},
        build_seconds=0.25,
    )

    assert gen_table.n_operators == 1
    assert gen_table.stored_blocks == 2
    assert gen_table.stored_elements == 10
    assert gen_table.stats["kind"] == "abelian_native_generator_operator_table"

    pair_table = AbelianNativePairBoundaryOperatorTable(side="center", bond=2)
    pair_table.add((0, 1, 2, 3), (AbelianPackedIdentityLocalEntry(1.0, op, op),))
    pair_table.add_operator((0, 1, 2, 3), op)

    assert pair_table.n_terms == 1
    assert pair_table.n_entries == 1
    assert pair_table.n_operators == 1
    assert pair_table.stored_blocks == 6
    assert pair_table.stored_elements == 30
    assert pair_table.stats["kind"] == "abelian_native_pair_boundary_operator_table"

    family_table = ComplementaryFamilyRenormalizedOperatorTable(
        side="left",
        bond=2,
        family_blocks={},
    )
    family_table.put_native_operator_table(("packed-gen", 2), gen_table)
    family_table.put_native_operator_table(("packed-pair", 2), pair_table)
    stats = family_table.stats

    assert stats["native_operator_tables"] == 2
    assert stats["native_operator_table_stored_elements"] == 40


def test_component_table_skips_grouping_when_auto_group_cap_is_one():
    table = AbelianNativeExactPatternComponentTable(bond=1)
    entries = (object(), object())
    records = (
        (("I",), "C", "D", ("I",), 0.5 + 0.0j),
        (("I",), "C", "D", ("I",), -0.25 + 0.0j),
    )

    stored = table.put_family(
        "P",
        entries,
        records=records,
        compression_policy="auto",
        max_group_size=1,
    )

    assert tuple(stored) == entries
    assert stored.entry_groups == ()
    assert stored.group_keys == ()
    assert table.get_family("P") is stored


def test_contextual_component_store_coalesces_fast_packed_entries():
    component_table = AbelianNativeExactPatternComponentTable(bond=1)
    stats = {"native_boundary_p": {"validation_policy": "off"}}
    phases = []

    store = AbelianContextualComponentStore(
        component_table=component_table,
        family_options=SimpleNamespace(
            exact_component_compression_policy="auto",
            exact_component_compression_validate=False,
            exact_component_compression_min_reduction=1,
            exact_component_compression_max_group_size=64,
        ),
        matvec_options={
            "generator_table_exact_component_compression_fast_max_group_size": 1,
            "generator_table_coalesce_contextual_entries": True,
        },
        stats=stats,
        record_phase=lambda name, elapsed, **fields: phases.append((name, fields)),
        validate_entries=lambda *_args, **_kwargs: True,
        bond=1,
    )
    E = object()
    W_left = object()
    W_right = object()
    F = object()
    records = (
        (("L",), "A", "B", ("R",), 1.0),
        (("L",), "A", "B", ("R",), 2.0),
    )
    entries = (
        AbelianPackedLocalGeneratorEntry(1.0, E, W_left, W_right, F, source="same"),
        AbelianPackedLocalGeneratorEntry(2.0, E, W_left, W_right, F, source="same"),
    )

    stored = store.store("P", entries, records)

    assert len(stored) == 1
    assert stored[0].coeff == 3.0 + 0.0j
    assert component_table.get_family("P") is stored
    assert stats["contextual_entry_coalesce"]["last_reduction"] == 1
    assert phases[-1][1]["original_entries"] == 2
    assert phases[-1][1]["stored_entries"] == 1


def test_abelian_contextual_builder_precomputes_and_packs_entries():
    stats = {}
    phases = []
    left_calls = []
    right_calls = []

    def record_phase(name, seconds, **fields):
        phases.append((name, seconds, fields))

    def left_builder(pattern, piece, family_name=None):
        left_calls.append((tuple(pattern), str(piece), str(family_name)))
        return (("E", tuple(pattern), str(piece)), ("WL", str(piece)))

    def right_builder(pattern, piece, family_name=None):
        right_calls.append((tuple(pattern), str(piece), str(family_name)))
        return (("WR", str(piece)), ("F", tuple(pattern), str(piece)))

    def fallback_builder(_left_pattern, _left_piece, _right_piece, _right_pattern):
        raise AssertionError("fallback should not be needed")

    builder = AbelianContextualDirectFamilyBuilder(
        stats=stats,
        record_phase=record_phase,
        left_builder=left_builder,
        right_builder=right_builder,
        fallback_builder=fallback_builder,
    )
    records = (
        (("I",), "C", "D", ("I",), 2.0 + 0.0j),
        (("I",), "C", "D", ("I",), -0.5 + 0.0j),
    )
    options = AbelianContextualFamilyBuildOptions(
        precompute_boundaries=True,
        pack_entries=True,
    )

    batch = builder.precompute_boundaries("P", records)
    result = builder.build_entries(
        "P",
        records,
        options=options,
        boundary_batch=batch,
    )

    assert len(left_calls) == 1
    assert len(right_calls) == 1
    assert phases[0][0] == "contextual_boundary_precompute"
    assert phases[0][2]["records"] == 2
    assert phases[0][2]["left_unique"] == 1
    assert phases[0][2]["right_unique"] == 1
    assert "left_build_seconds" in phases[0][2]
    assert "right_build_seconds" in phases[0][2]
    assert stats["contextual_recursive_terms"] == 2
    assert result.entries is not None
    assert [entry.coeff for entry in result.entries] == [2.0 + 0.0j, -0.5 + 0.0j]
    assert all(
        isinstance(entry, AbelianPackedLocalGeneratorEntry)
        for entry in result.entries
    )
    batch = builder.precompute_boundaries("P", records)
    result = builder.build_entries(
        "P",
        records,
        options=options,
        boundary_batch=batch,
    )

    assert len(left_calls) == 1
    assert len(right_calls) == 1
    assert result.entries is not None
    assert len(result.entries) == 2
    precompute_stats = stats["contextual_boundary_precompute"]
    assert precompute_stats["left_hits"] == 1
    assert precompute_stats["left_misses"] == 1
    assert precompute_stats["right_hits"] == 1
    assert precompute_stats["right_misses"] == 1
    assert precompute_stats["left_build_seconds"] >= 0.0
    assert precompute_stats["right_build_seconds"] >= 0.0


def test_make_contextual_family_records_splits_two_site_center():
    records = make_contextual_family_records(
        (
            (("A", "B", "C", "D"), 0.25),
            (("I", "X", "Y", "Z"), -1.0j),
        ),
        bond=1,
    )

    assert records == (
        (("A",), "B", "C", ("D",), 0.25 + 0.0j),
        (("I",), "X", "Y", ("Z",), -1.0j),
    )


def test_abelian_contextual_builder_caches_lazy_boundary_lookups():
    stats = {}
    left_calls = []
    right_calls = []

    def record_phase(*_args, **_kwargs):
        pass

    def left_builder(pattern, piece, family_name=None):
        left_calls.append((tuple(pattern), str(piece), str(family_name)))
        return (("E", tuple(pattern), str(piece)), ("WL", str(piece)))

    def right_builder(pattern, piece, family_name=None):
        right_calls.append((tuple(pattern), str(piece), str(family_name)))
        return (("WR", str(piece)), ("F", tuple(pattern), str(piece)))

    def fallback_builder(*_args, **_kwargs):
        raise AssertionError("fallback should not be needed")

    builder = AbelianContextualDirectFamilyBuilder(
        stats=stats,
        record_phase=record_phase,
        left_builder=left_builder,
        right_builder=right_builder,
        fallback_builder=fallback_builder,
    )
    records = (
        (("L",), "A", "B", ("R",), 1.0),
        (("L",), "A", "B", ("R",), 2.0),
        (("L2",), "A", "B", ("R",), 3.0),
    )

    result = builder.build_entries(
        "P",
        records,
        options=AbelianContextualFamilyBuildOptions(
            precompute_boundaries=False,
            pack_entries=True,
        ),
    )

    assert result.entries is not None
    assert len(result.entries) == 3
    assert len(left_calls) == 2
    assert len(right_calls) == 1
    assert stats["contextual_recursive_terms"] == 3
    cache_stats = stats["contextual_lazy_boundary_cache"]
    assert cache_stats["left_hits"] == 1
    assert cache_stats["left_misses"] == 2
    assert cache_stats["right_hits"] == 2
    assert cache_stats["right_misses"] == 1


def test_abelian_direct_route_plan_drives_contextual_builder():
    stats = {}
    left_calls = []
    right_calls = []

    def record_phase(*_args, **_kwargs):
        pass

    def left_builder(pattern, piece, family_name=None):
        left_calls.append((tuple(pattern), str(piece), str(family_name)))
        return (("E", tuple(pattern), str(piece)), ("WL", str(piece)))

    def right_builder(pattern, piece, family_name=None):
        right_calls.append((tuple(pattern), str(piece), str(family_name)))
        return (("WR", str(piece)), ("F", tuple(pattern), str(piece)))

    def fallback_builder(*_args, **_kwargs):
        raise AssertionError("fallback should not be needed")

    records = (
        (("L",), "A", "B", ("R",), 1.0),
        (("L",), "A", "B", ("R",), 2.0),
        (("L2",), "A", "B", ("R",), 3.0),
    )
    plan = AbelianDirectRoutePlan.from_records("P", records, bond=1)

    assert plan.record_count == 3
    assert plan.pair_count == 2
    assert plan.left_count == 2
    assert plan.right_count == 1
    assert tuple(plan.iter_records()) == records
    np.testing.assert_array_equal(plan.left_ids, np.asarray([0, 0, 1]))
    np.testing.assert_array_equal(plan.right_ids, np.asarray([0, 0, 0]))
    np.testing.assert_array_equal(plan.pair_left_ids, np.asarray([0, 1]))
    np.testing.assert_array_equal(plan.pair_right_ids, np.asarray([0, 0]))
    np.testing.assert_allclose(plan.pair_coeffs, np.asarray([3.0, 3.0]))

    builder = AbelianContextualDirectFamilyBuilder(
        stats=stats,
        record_phase=record_phase,
        left_builder=left_builder,
        right_builder=right_builder,
        fallback_builder=fallback_builder,
    )
    result = builder.build_entries(
        "ignored",
        plan,
        options=AbelianContextualFamilyBuildOptions(
            precompute_boundaries=False,
            pack_entries=True,
        ),
    )

    assert result.entries is not None
    assert len(result.entries) == 3
    assert all(
        isinstance(entry, AbelianPackedLocalGeneratorEntry)
        for entry in result.entries
    )
    assert [entry.coeff for entry in result.entries] == [1.0, 2.0, 3.0]
    assert len(left_calls) == 2
    assert len(right_calls) == 1

    batch = builder.precompute_boundaries("ignored", plan)
    result = builder.build_entries(
        "ignored",
        plan,
        options=AbelianContextualFamilyBuildOptions(
            precompute_boundaries=True,
            pack_entries=True,
        ),
        boundary_batch=batch,
    )

    assert not batch.left
    assert not batch.right
    assert len(batch.left_values) == plan.left_count
    assert len(batch.right_values) == plan.right_count
    assert result.entries is not None
    assert [entry.coeff for entry in result.entries] == [3.0, 3.0]
    assert len(left_calls) == 2
    assert len(right_calls) == 1
    assert stats["contextual_route_fast_pack"]["calls"] == 1
    assert stats["contextual_route_fast_pack"]["entries"] == 2
    assert stats["contextual_route_fast_pack"]["coalesced_records"] == 1


def test_abelian_block2_like_plan_ir_covers_all_layers():
    def packed3(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label),),
            (np.ones((1, 1, 1), dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[label], [label], [label]],
        )

    def packed4(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label, label),),
            (np.ones((1, 1, 1, 1), dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[label], [label], [label], [label]],
        )

    records = (
        ((0,), "A", "B", (2,), 1.0),
        ((0,), "A", "B", (2,), 2.0),
        ((1,), "A", "B", (2,), -0.5),
    )
    route_plan = AbelianDirectRoutePlan.from_records("P", records, bond=1)
    family_plan = AbelianOperatorFamilyPlan.from_route_plan(route_plan)

    assert family_plan.stats["kind"] == "abelian_operator_family_plan"
    assert family_plan.stats["pairs"] == 2

    builder = AbelianContextualDirectFamilyBuilder(
        stats={},
        record_phase=lambda *_args, **_kwargs: None,
        left_builder=lambda *_args, **_kwargs: None,
        right_builder=lambda *_args, **_kwargs: None,
        left_batch_builder=lambda keys, **_kwargs: tuple(
            (packed3(f"L{idx}"), packed4(f"LW{idx}"))
            for idx, _key in enumerate(keys)
        ),
        right_batch_builder=lambda keys, **_kwargs: tuple(
            (packed4(f"RW{idx}"), packed3(f"R{idx}"))
            for idx, _key in enumerate(keys)
        ),
        fallback_builder=lambda *_args, **_kwargs: None,
    )
    batch = builder.precompute_boundaries("P", route_plan)
    moving_tables = AbelianMovingEnvironmentTables.from_contextual_builder(
        builder,
        bond=1,
        revision=4,
    )
    layout = (((0, 0), (2, 3)), ((1, 0), (1, 3)))
    symmetry = AbelianSymmetryAdapter.from_layout(layout, dirs=(1, -1))
    action_plan = AbelianLocalActionPlan.from_boundary_batch(
        family_plan=family_plan,
        moving_tables=moving_tables,
        boundary_batch=batch,
        layout=layout,
        dirs=(1, -1),
        backend="contextual_direct",
    )

    assert symmetry.compatible_layout(layout)
    assert symmetry.stats["sectors"] == 2
    assert moving_tables.stats["left_entries"] == 2
    assert moving_tables.stats["right_entries"] == 1
    assert action_plan.stats["left_table_ids"] == 2
    assert action_plan.stats["right_table_ids"] == 1
    assert action_plan.stats["layout_sectors"] == 2
    assert action_plan.cache_key() == action_plan.signature
    assert not action_plan.stale_for(layout=layout, moving_tables=moving_tables)
    assert action_plan.stale_for(
        layout=(((0, 0), (2, 3)), ((1, 0), (2, 3))),
        moving_tables=moving_tables,
    )


def test_abelian_local_action_plan_cache_rebuilds_on_signature_change():
    route_plan = AbelianDirectRoutePlan.from_records(
        "P",
        (((0,), "A", "B", (2,), 1.0),),
        bond=1,
    )
    family_plan = AbelianOperatorFamilyPlan.from_route_plan(route_plan)
    empty_batch = SimpleNamespace(left_table_ids=(), right_table_ids=())
    moving_tables = AbelianMovingEnvironmentTables(bond=1, revision=0)
    layout = (((0, 0), (1, 1)),)
    cache = AbelianLocalActionPlanCache()

    def build(layout_arg):
        return AbelianLocalActionPlan.from_boundary_batch(
            family_plan=family_plan,
            moving_tables=moving_tables,
            boundary_batch=empty_batch,
            layout=layout_arg,
            dirs=(1, -1),
            backend="contextual_direct",
        )

    key = (
        family_plan.cache_key(layout_signature=layout, revision=0),
        moving_tables.signature,
    )
    first, hit = cache.get_or_build(key, lambda: build(layout))
    second, hit2 = cache.get_or_build(key, lambda: build(layout))
    changed_layout = (((0, 0), (2, 1)),)
    third, hit3 = cache.get_or_build(
        (
            family_plan.cache_key(layout_signature=changed_layout, revision=0),
            moving_tables.signature,
        ),
        lambda: build(changed_layout),
    )

    assert hit is False
    assert hit2 is True
    assert hit3 is False
    assert first is second
    assert third is not first
    assert cache.stats["builds"] == 2
    assert cache.stats["hits"] == 1
    assert cache.invalidate(lambda _key, plan: plan is first) == 1
    assert cache.stats["invalidations"] == 1


def test_packed_boundary_operator_composition_adds_flux_sector():
    first = AbelianPackedBoundaryTensor(
        ((1, 2, 3),),
        (np.arange(6, dtype=complex).reshape(1, 2, 3),),
        dirs=[1, -1, 1],
        qns=[[1], [2], [3]],
    )
    second = AbelianPackedBoundaryTensor(
        ((4, 3, 5),),
        ((np.arange(12, dtype=complex) + 1).reshape(1, 3, 4),),
        dirs=[1, -1, 1],
        qns=[[4], [3], [5]],
    )

    composed = compose_abelian_packed_boundary_operators(first, second)

    assert composed is not None
    assert composed.keys == ((5, 2, 5),)
    np.testing.assert_allclose(
        composed.blocks[0][0],
        first.blocks[0][0] @ second.blocks[0][0],
    )


def test_abelian_direct_route_plan_uses_batch_boundary_builders():
    stats = {}
    phases = []
    left_calls = []
    right_calls = []
    left_batch_calls = []
    right_batch_calls = []

    def record_phase(name, _seconds, **fields):
        phases.append((name, fields))

    def left_builder(pattern, piece, family_name=None):
        left_calls.append((tuple(pattern), str(piece), str(family_name)))
        raise AssertionError("scalar left builder should not be used")

    def right_builder(pattern, piece, family_name=None):
        right_calls.append((tuple(pattern), str(piece), str(family_name)))
        raise AssertionError("scalar right builder should not be used")

    def left_batch_builder(keys, family_name=None):
        left_batch_calls.append((tuple(keys), str(family_name)))
        return tuple(
            (("BE", tuple(pattern), str(piece)), ("BWL", str(piece)))
            for pattern, piece in keys
        )

    def right_batch_builder(keys, family_name=None):
        right_batch_calls.append((tuple(keys), str(family_name)))
        return tuple(
            (("BWR", str(piece)), ("BF", tuple(pattern), str(piece)))
            for pattern, piece in keys
        )

    def fallback_builder(*_args, **_kwargs):
        raise AssertionError("fallback should not be needed")

    records = (
        (("L",), "A", "B", ("R",), 1.0),
        (("L",), "A", "B", ("R",), 2.0),
        (("L2",), "A", "B", ("R",), 3.0),
    )
    plan = AbelianDirectRoutePlan.from_records("P", records, bond=1)
    builder = AbelianContextualDirectFamilyBuilder(
        stats=stats,
        record_phase=record_phase,
        left_builder=left_builder,
        right_builder=right_builder,
        left_batch_builder=left_batch_builder,
        right_batch_builder=right_batch_builder,
        fallback_builder=fallback_builder,
    )

    batch = builder.precompute_boundaries("ignored", plan)
    result = builder.build_entries(
        "ignored",
        plan,
        options=AbelianContextualFamilyBuildOptions(
            precompute_boundaries=True,
            pack_entries=True,
        ),
        boundary_batch=batch,
    )

    assert len(left_calls) == 0
    assert len(right_calls) == 0
    assert len(left_batch_calls) == 1
    assert len(right_batch_calls) == 1
    assert len(batch.left_values) == 2
    assert len(batch.right_values) == 1
    assert result.entries is not None
    assert [entry.coeff for entry in result.entries] == [3.0, 3.0]
    precompute_stats = stats["contextual_boundary_precompute"]
    assert precompute_stats["left_batch_calls"] == 1
    assert precompute_stats["right_batch_calls"] == 1
    assert phases[0][0] == "contextual_boundary_precompute"
    assert phases[0][1]["left_batch"] == 1
    assert phases[0][1]["right_batch"] == 1
    assert phases[0][1]["left_table_ids"] == 0
    assert phases[0][1]["right_table_ids"] == 0


def test_abelian_contextual_boundary_batch_tracks_packed_payloads():
    stats = {}
    phases = []

    def record_phase(name, _seconds, **fields):
        phases.append((name, fields))

    def packed3(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label),),
            (np.ones((1, 1, 1), dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[label], [label], [label]],
            source=f"packed_{label}",
        )

    def packed4(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label, label),),
            (np.ones((1, 1, 1, 1), dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[label], [label], [label], [label]],
            source=f"packed_{label}",
        )

    def left_batch_builder(keys, family_name=None):
        return tuple((packed3(f"L{idx}"), packed4(f"LW{idx}")) for idx, _ in enumerate(keys))

    def right_batch_builder(keys, family_name=None):
        return tuple((packed4(f"RW{idx}"), packed3(f"R{idx}")) for idx, _ in enumerate(keys))

    builder = AbelianContextualDirectFamilyBuilder(
        stats=stats,
        record_phase=record_phase,
        left_builder=lambda *_args, **_kwargs: None,
        right_builder=lambda *_args, **_kwargs: None,
        left_batch_builder=left_batch_builder,
        right_batch_builder=right_batch_builder,
        fallback_builder=lambda *_args, **_kwargs: None,
    )
    plan = AbelianDirectRoutePlan.from_records(
        "P",
        (
            (("L",), "A", "B", ("R",), 1.0),
            (("L2",), "A", "B", ("R",), 2.0),
        ),
        bond=1,
    )

    batch = builder.precompute_boundaries("P", plan)
    result = builder.build_entries(
        "P",
        plan,
        options=AbelianContextualFamilyBuildOptions(
            precompute_boundaries=True,
            pack_entries=True,
            packed_buffer=True,
        ),
        boundary_batch=batch,
    )

    assert batch.packed_boundary_pairs
    assert batch.left_packed_count == 2
    assert batch.right_packed_count == 1
    assert batch.left_table_ids == (0, 1)
    assert batch.right_table_ids == (0,)
    assert stats["contextual_boundary_precompute"]["left_payload_packed"] == 2
    assert stats["contextual_boundary_precompute"]["right_payload_packed"] == 1
    assert stats["contextual_boundary_precompute"]["left_table_ids"] == 2
    assert stats["contextual_boundary_precompute"]["right_table_ids"] == 1
    assert phases[0][1]["left_packed"] == 2
    assert phases[0][1]["right_packed"] == 1
    assert builder.left_packed_boundary_table.n_entries == 2
    assert builder.right_packed_boundary_table.n_entries == 1
    assert stats["packed_contextual_boundary_tables"]["left"]["entries"] == 2
    assert stats["packed_contextual_boundary_tables"]["right"]["entries"] == 1
    assert stats["packed_contextual_boundary_tables"]["left"]["ids"] == 2
    assert stats["packed_contextual_boundary_tables"]["right"]["ids"] == 1
    assert builder.left_packed_boundary_table.batch_resolves == 1
    assert builder.left_packed_boundary_table.last_batch_misses == 2
    assert builder.right_packed_boundary_table.last_batch_misses == 1
    assert getattr(result.entries, "_pyqed_planned_direct_family_entries", False)
    assert tuple(result.entries.local_left_ids.tolist()) == (0, 1)
    assert tuple(result.entries.local_right_ids.tolist()) == (0, 0)
    assert result.entries.local_generator_count == 2
    fast_pack = stats["contextual_route_fast_pack"]
    assert fast_pack["packed_boundary_calls"] == 1
    assert fast_pack["packed_boundary_entries"] == 2
    assert fast_pack["planned_calls"] == 1
    assert fast_pack["planned_entries"] == 2
    assert fast_pack["last_boundary_payload"] == "packed"

    second_batch = builder.precompute_boundaries("P", plan)

    assert second_batch.packed_boundary_pairs
    assert second_batch.left_table_ids == (0, 1)
    assert second_batch.right_table_ids == (0,)
    boundary_cache = stats["contextual_boundary_batch_cache"]
    assert boundary_cache["hits"] == 2
    assert builder.left_packed_boundary_table.batch_resolves == 1
    assert builder.right_packed_boundary_table.batch_resolves == 1


def test_abelian_planned_direct_entries_materialize_only_when_needed():
    def packed3(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label),),
            (np.ones((1, 1, 1), dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[label], [label], [label]],
        )

    def packed4(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label, label),),
            (np.ones((1, 1, 1, 1), dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[label], [label], [label], [label]],
        )

    left_values = (
        (packed3("L0"), packed4("LW0")),
        (packed3("L1"), packed4("LW1")),
    )
    right_values = ((packed4("RW0"), packed3("R0")),)
    batch = SimpleNamespace(left_values=left_values, right_values=right_values)
    plan = AbelianDirectRoutePlan.from_records(
        "P",
        (
            (("L0",), "A", "B", ("R0",), 1.0),
            (("L1",), "A", "B", ("R0",), 2.0),
        ),
        bond=1,
    )

    entries = AbelianPlannedPackedDirectFamilyEntries.from_route_plan(
        plan,
        batch,
        source="planned_test",
    )

    assert entries._local_columns is None
    assert entries.local_generator_count == 2
    assert tuple(entries.local_left_ids.tolist()) == (0, 1)
    assert tuple(entries.local_right_ids.tolist()) == (0, 0)
    assert entries.stats["kind"] == "abelian_planned_packed_direct_family_entries"
    assert entries.local_E[0] is left_values[0][0]
    assert entries.local_W_left[1] is left_values[1][1]
    assert entries.local_W_right[0] is right_values[0][0]
    assert entries.local_F[1] is right_values[0][1]
    materialized = list(entries)
    assert [entry.coeff for entry in materialized] == [1.0, 2.0]
    assert [entry.source for entry in materialized] == ["planned_test", "planned_test"]


def test_abelian_contextual_lazy_planned_entries_can_use_table_ids_only():
    stats = {}

    def packed3(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label),),
            (np.ones((1, 1, 1), dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[label], [label], [label]],
            source=f"packed_{label}",
        )

    def packed4(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label, label),),
            (np.ones((1, 1, 1, 1), dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[label], [label], [label], [label]],
            source=f"packed_{label}",
        )

    def left_batch_builder(keys, family_name=None):
        return tuple((packed3(f"L{idx}"), packed4(f"LW{idx}")) for idx, _ in enumerate(keys))

    def right_batch_builder(keys, family_name=None):
        return tuple((packed4(f"RW{idx}"), packed3(f"R{idx}")) for idx, _ in enumerate(keys))

    builder = AbelianContextualDirectFamilyBuilder(
        stats=stats,
        record_phase=lambda *_args, **_kwargs: None,
        left_builder=lambda *_args, **_kwargs: None,
        right_builder=lambda *_args, **_kwargs: None,
        left_batch_builder=left_batch_builder,
        right_batch_builder=right_batch_builder,
        fallback_builder=lambda *_args, **_kwargs: None,
    )
    plan = AbelianDirectRoutePlan.from_records(
        "P",
        (
            (("L0",), "A", "B", ("R0",), 1.0),
            (("L1",), "A", "B", ("R0",), 2.0),
        ),
        bond=1,
    )

    result = builder.build_entries(
        "P",
        plan,
        options=AbelianContextualFamilyBuildOptions(
            precompute_boundaries=False,
            pack_entries=True,
            packed_buffer=True,
            planned_without_precompute=True,
            planned_without_precompute_batch=True,
            planned_without_precompute_table_lookup=True,
            planned_without_precompute_table_ids_only=True,
            snapshot_table_backed_planned_entries=False,
        ),
    )

    assert getattr(result.entries, "_pyqed_planned_direct_family_entries", False)
    assert result.entries._pyqed_planned_direct_family_table_ids
    assert result.entries.left_values == ()
    assert result.entries.right_values == ()
    assert tuple(result.entries.local_left_ids.tolist()) == (0, 1)
    assert tuple(result.entries.local_right_ids.tolist()) == (0, 0)
    assert tuple(result.entries.left_table_ids.tolist()) == (0, 1)
    assert tuple(result.entries.right_table_ids.tolist()) == (0,)
    assert builder.left_packed_boundary_table.n_entries == 2
    assert builder.right_packed_boundary_table.n_entries == 1
    lazy_stats = stats["contextual_route_lazy_pack"]
    assert lazy_stats["table_ids_only"] is True
    assert lazy_stats["planned_calls"] == 1
    assert lazy_stats["planned_entries"] == 2


def test_abelian_contextual_table_id_resolution_requires_current_entries():
    table = AbelianPackedContextualBoundaryTable(side="left", bond=1, revision=0)
    payload = (
        AbelianPackedBoundaryTensor(
            ((0, 0, 0),),
            (np.ones((1, 1, 1), dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[0], [0], [0]],
        ),
        AbelianPackedBoundaryTensor(
            ((0, 0, 0, 0),),
            (np.ones((1, 1, 1, 1), dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[0], [0], [0], [0]],
        ),
    )
    key = ((0,), "A")
    assert table.put(key, payload, family_name="P")
    ids, missing, positions, hits, misses = table.resolve_current_ids_many((key,))
    assert tuple(ids) == (0,)
    assert missing == ()
    assert positions == ()
    assert hits == 1
    assert misses == 0

    assert table.reset_for_revision(1)
    ids, missing, positions, hits, misses = table.resolve_current_ids_many((key,))
    assert tuple(ids) == (-1,)
    assert missing == (table.normalize_key(key),)
    assert positions == (0,)
    assert hits == 0
    assert misses == 1
    assert len(table.payloads) == 1


def test_same_side_p_route_plan_merges_packed_boundaries_into_table():
    def packed(label, value):
        return AbelianPackedBoundaryTensor(
            ((label, 0, 0),),
            (np.asarray([[[value, value + 1.0]]], dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[label], [0], [0]],
            source=f"packed_{label}",
        )

    left = packed(0, 2.0)
    right = packed(0, 5.0)
    plan = AbelianSameSidePRoutePlan.from_planned_terms(
        side="left",
        bond=2,
        planned_terms=(
            (
                (0, 1, 2, 3),
                (
                    ((("A",), "I"), 1.5),
                    ((("B",), "I"), -0.5),
                    ((("A",), "I"), 0.25),
                ),
            ),
        ),
    )
    assert plan.terms == 2
    boundary_by_key = {
        ((("A",), "I")): left,
        ((("B",), "I")): right,
    }
    boundary_results = tuple(boundary_by_key[key] for key in plan.boundary_keys)
    table = AbelianNativePairBoundaryOperatorTable(side="left", bond=2)

    result = merge_abelian_same_side_p_route_plan(
        plan,
        boundary_results,
        operator_table=table,
    )

    assert result["complete"] is True
    assert result["built"] == 1
    assert result["items"] == ()
    assert table.n_operators == 1
    expected = sum_abelian_packed_boundary_terms(((left, 1.75), (right, -0.5)))
    same, diff, ref = compare_abelian_packed_boundary_tensors(
        table.get_operator((0, 1, 2, 3)),
        expected,
    )
    assert same
    assert diff <= 1.0e-13 * max(ref, 1.0)


def test_same_side_p_boundary_value_table_keeps_stable_ids_across_refresh():
    table = AbelianSameSidePBoundaryValueTable(side="left", bond=2, revision=0)
    key_a = (("A",), "I")
    key_b = (("B",), "I")
    first = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
    )
    second = AbelianPackedBoundaryTensor(
        ((1, 1, 1),),
        (2.0 * np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[1], [1], [1]],
    )

    assert table.put_many((key_a, key_b), (first, second)) == 2
    values, ids, missing, positions, hits, misses = table.resolve_many(
        (key_a, key_b),
        return_ids=True,
    )

    assert values == [first, second]
    assert ids == (0, 1)
    assert missing == ()
    assert positions == ()
    assert hits == 2
    assert misses == 0
    assert table.values_for_ids(ids) == [first, second]

    assert table.reset_for_revision(1)
    refreshed = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (3.0 * np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
    )
    assert table.put(key_a, refreshed)
    values, ids, missing, positions, hits, misses = table.resolve_many(
        (key_a, key_b),
        return_ids=True,
    )

    assert values == [refreshed, None]
    assert ids == (0, -1)
    assert missing == (table.normalize_key(key_b),)
    assert positions == (1,)
    assert hits == 1
    assert misses == 1
    assert table.values_for_ids((0, 1)) == [refreshed, None]


def test_cpp_same_side_p_route_identity_entries_match_explicit_identity():
    from pyqed.mps import cpp_davidson

    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "build_direct_family_payload_fastkeys", None) is None
        or getattr(cpp_davidson, "GroupedRenormalizedTable", None) is None
    ):
        pytest.skip("C++ direct payload backend is unavailable")

    left_a = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.arange(1, 5, dtype=float).reshape(1, 2, 2),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
    )
    left_b = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        ((np.arange(5, 9, dtype=float).reshape(1, 2, 2) + 0.25j),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
    )
    identity_right = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        ((np.arange(1, 10, dtype=float).reshape(1, 3, 3) - 0.5j),),
        dirs=[-1, 1, -1],
        qns=[[0], [0], [0]],
    )
    route_plan = AbelianSameSidePRoutePlan.from_planned_terms(
        side="left",
        bond=2,
        planned_terms=(
            (
                (0, 1, 2, 3),
                (
                    ((("A",), "I"), 1.5),
                    ((("B",), "I"), -0.5),
                ),
            ),
        ),
    )
    value_table = AbelianSameSidePBoundaryValueTable(
        side="left",
        bond=2,
        revision=0,
    )
    value_table.put_many(
        route_plan.boundary_keys,
        (left_a, left_b),
        normalized=True,
    )
    _values, table_ids, *_rest = value_table.resolve_many(
        route_plan.boundary_keys,
        normalized=True,
        return_ids=True,
    )
    compact = AbelianSameSidePRouteIdentityEntries(
        side="left",
        row_ids=(0,),
        row_coeffs=(1.25,),
        route_plan=route_plan,
        boundary_table_ids=table_ids,
        boundary_table=value_table,
        identity_tensor=identity_right,
    )
    explicit = AbelianPackedDirectFamilyEntries()
    explicit.append_identity(1.25 * 1.5, left_a, identity_right)
    explicit.append_identity(1.25 * -0.5, left_b, identity_right)

    assert [entry.coeff for entry in compact] == [1.25 * 1.5, 1.25 * -0.5]

    layout = (((0, 0, 0, 0), (2, 3, 2, 2)),)
    compact_payload = cpp_davidson.build_direct_family_payload_fastkeys(
        {"P": compact},
        {},
        layout,
        True,
    )
    explicit_payload = cpp_davidson.build_direct_family_payload_fastkeys(
        {"P": explicit},
        {},
        layout,
        True,
    )
    compact_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        compact_payload,
        24,
        0.0,
    )
    explicit_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        explicit_payload,
        24,
        0.0,
    )
    rng = np.random.default_rng(12345)
    vec = (
        rng.normal(size=24) + 1j * rng.normal(size=24)
    ).astype(np.complex128)
    np.testing.assert_allclose(
        compact_table.matvec(vec),
        explicit_table.matvec(vec),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_cpp_planned_direct_payload_plan_matches_stateless_builder():
    from pyqed.mps import cpp_davidson

    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "PlannedDirectPayloadPlan", None) is None
        or getattr(cpp_davidson, "GroupedRenormalizedTable", None) is None
    ):
        pytest.skip("C++ planned direct payload backend is unavailable")

    def packed3(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0),),
            (np.asarray([[[value]]], dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[0], [0], [0]],
        )

    def packed4(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0, 0),),
            (np.asarray([[[[value]]]], dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[0], [0], [0], [0]],
        )

    route_plan = AbelianDirectRoutePlan.from_records(
        "P",
        (
            (("a",), "I", "I", ("z",), 2.0),
            (("b",), "I", "I", ("z",), -0.5j),
        ),
        bond=1,
    )
    left_table = AbelianPackedContextualBoundaryTable(side="left", bond=1)
    right_table = AbelianPackedContextualBoundaryTable(side="right", bond=2)
    left_payloads = ((packed3(2.0), packed4(3.0)), (packed3(5.0), packed4(7.0)))
    right_payloads = ((packed4(11.0), packed3(13.0)),)
    for key, payload in zip(route_plan.left_keys, left_payloads):
        assert left_table.put(key, payload)
    for key, payload in zip(route_plan.right_keys, right_payloads):
        assert right_table.put(key, payload)
    _left_values, left_ids, *_ = left_table.resolve_many(
        route_plan.left_keys,
        return_ids=True,
    )
    _right_values, right_ids, *_ = right_table.resolve_many(
        route_plan.right_keys,
        return_ids=True,
    )
    batch = SimpleNamespace(
        left_values=(),
        right_values=(),
        left_table_ids=left_ids,
        right_table_ids=right_ids,
    )
    entries = AbelianPlannedPackedDirectFamilyEntries.from_route_plan(
        route_plan,
        batch,
        left_table=left_table,
        right_table=right_table,
        source="planned_test_identity_local_csr",
    )
    direct_envs = {"P": entries}
    layout = (((0, 0, 0, 0), (1, 1, 1, 1)),)
    stateless = cpp_davidson.build_direct_family_payload_fastkeys(
        direct_envs,
        {},
        layout,
        True,
    )
    planned = cpp_davidson.PlannedDirectPayloadPlan()
    cached = planned.build_payload(direct_envs, {}, layout, True)
    cached_again = planned.build_payload(direct_envs, {}, layout, True)

    assert int(stateless.size()) == int(cached.size())
    assert int(stateless.size()) == int(cached_again.size())
    assert int(planned.stats()["payload_builds"]) == 2
    assert int(planned.stats()["planned_schedule_pointer_cache_entries"]) == 1
    assert int(planned.stats()["planned_schedule_cache_misses"]) == 1
    assert int(planned.stats()["planned_schedule_cache_hits"]) == 1
    assert int(planned.stats()["planned_payload_pair_cache_misses"]) == 2
    assert int(planned.stats()["planned_payload_pair_cache_hits"]) == 2
    stateless_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        stateless,
        1,
        0.0,
    )
    cached_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        cached,
        1,
        0.0,
    )
    vec = np.asarray([1.25 - 0.75j], dtype=np.complex128)
    np.testing.assert_allclose(cached_table.matvec(vec), stateless_table.matvec(vec))
    cached_again_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        cached_again,
        1,
        0.0,
    )
    np.testing.assert_allclose(
        cached_again_table.matvec(vec),
        stateless_table.matvec(vec),
    )


def test_cpp_planned_direct_payload_chunking_preserves_action():
    from pyqed.mps import cpp_davidson

    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "PlannedDirectPayloadPlan", None) is None
        or getattr(cpp_davidson, "GroupedRenormalizedTable", None) is None
    ):
        pytest.skip("C++ planned direct payload backend is unavailable")

    def packed3(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0),),
            (np.asarray([[[value]]], dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[0], [0], [0]],
        )

    def packed4(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0, 0),),
            (np.asarray([[[[value]]]], dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[0], [0], [0], [0]],
        )

    left_values = (
        (packed3(2.0), packed4(3.0)),
        (packed3(-5.0), packed4(7.0)),
        (packed3(11.0), packed4(-13.0)),
    )
    right_values = (
        (packed4(17.0), packed3(19.0)),
        (packed4(-23.0), packed3(29.0)),
    )
    source = "planned_test_identity_local_csr"
    combined = AbelianPlannedPackedDirectFamilyEntries(
        [1.0, -0.25j, 0.75, -1.5],
        [0, 1, 2, 0],
        [0, 0, 1, 1],
        left_values,
        right_values,
        source=source,
    )
    chunked = AbelianCompositePackedDirectFamilyEntries(
        (
            AbelianPlannedPackedDirectFamilyEntries(
                [1.0, -0.25j],
                [0, 1],
                [0, 0],
                left_values,
                right_values,
                source=source,
            ),
            AbelianPlannedPackedDirectFamilyEntries(
                [0.75, -1.5],
                [2, 0],
                [1, 1],
                left_values,
                right_values,
                source=source,
            ),
        )
    )
    layout = (((0, 0, 0, 0), (1, 1, 1, 1)),)
    plan = cpp_davidson.PlannedDirectPayloadPlan()
    combined_payload = plan.build_payload({"P": combined}, {}, layout, True)
    chunked_payload = plan.build_payload({"P": chunked}, {}, layout, True)
    combined_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        combined_payload,
        1,
        0.0,
    )
    chunked_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        chunked_payload,
        1,
        0.0,
    )
    vec = np.asarray([0.5 + 0.125j], dtype=np.complex128)

    np.testing.assert_allclose(
        chunked_table.matvec(vec),
        combined_table.matvec(vec),
        rtol=1e-13,
        atol=1e-13,
    )

    left_table = AbelianPackedContextualBoundaryTable(side="left", bond=1)
    right_table = AbelianPackedContextualBoundaryTable(side="right", bond=2)
    left_keys = tuple(((f"L{idx}",), "I") for idx in range(len(left_values)))
    right_keys = tuple(((f"R{idx}",), "I") for idx in range(len(right_values)))
    for key, value in zip(left_keys, left_values):
        assert left_table.put(key, value, family_name="P_identity")
    for key, value in zip(right_keys, right_values):
        assert right_table.put(key, value, family_name="P_identity")
    left_table_ids = tuple(left_table.ids[left_table.normalize_key(key)] for key in left_keys)
    right_table_ids = tuple(
        right_table.ids[right_table.normalize_key(key)] for key in right_keys
    )
    table_combined = AbelianPlannedPackedDirectFamilyEntries(
        [1.0, -0.25j, 0.75, -1.5],
        [0, 1, 2, 0],
        [0, 0, 1, 1],
        (),
        (),
        left_table_ids=left_table_ids,
        right_table_ids=right_table_ids,
        left_table=left_table,
        right_table=right_table,
        source=source,
    )
    table_chunked = AbelianCompositePackedDirectFamilyEntries(
        (
            AbelianPlannedPackedDirectFamilyEntries(
                [1.0, -0.25j],
                [0, 1],
                [0, 0],
                (),
                (),
                left_table_ids=(left_table_ids[0], left_table_ids[1]),
                right_table_ids=(right_table_ids[0],),
                left_table=left_table,
                right_table=right_table,
                source=source,
            ),
            AbelianPlannedPackedDirectFamilyEntries(
                [0.75, -1.5],
                [0, 1],
                [0, 0],
                (),
                (),
                left_table_ids=(left_table_ids[2], left_table_ids[0]),
                right_table_ids=(right_table_ids[1],),
                left_table=left_table,
                right_table=right_table,
                source=source,
            ),
        )
    )
    table_combined_payload = plan.build_payload({"P": table_combined}, {}, layout, True)
    table_chunked_payload = plan.build_payload({"P": table_chunked}, {}, layout, True)
    table_combined_action = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        table_combined_payload,
        1,
        0.0,
    )
    table_chunked_action = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        table_chunked_payload,
        1,
        0.0,
    )
    np.testing.assert_allclose(
        table_chunked_action.matvec(vec),
        table_combined_action.matvec(vec),
        rtol=1e-13,
        atol=1e-13,
    )


def test_abelian_contextual_builder_can_disable_packed_boundary_tables():
    stats = {}

    def packed3(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label),),
            (np.ones((1, 1, 1), dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[label], [label], [label]],
        )

    def packed4(label):
        return AbelianPackedBoundaryTensor(
            ((label, label, label, label),),
            (np.ones((1, 1, 1, 1), dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[label], [label], [label], [label]],
        )

    builder = AbelianContextualDirectFamilyBuilder(
        stats=stats,
        record_phase=lambda *_args, **_kwargs: None,
        left_builder=lambda pattern, piece, family_name=None: (
            packed3("L"),
            packed4("LW"),
        ),
        right_builder=lambda pattern, piece, family_name=None: (
            packed4("RW"),
            packed3("R"),
        ),
        fallback_builder=lambda *_args, **_kwargs: None,
        enable_packed_boundary_tables=False,
    )
    plan = AbelianDirectRoutePlan.from_records(
        "P",
        ((("L",), "A", "B", ("R",), 1.0),),
        bond=1,
    )

    batch = builder.precompute_boundaries("P", plan)

    assert batch.packed_boundary_pairs
    assert builder.left_packed_boundary_table is None
    assert builder.right_packed_boundary_table is None
    assert "packed_contextual_boundary_tables" not in stats


def test_abelian_packed_direct_entries_coalesce_shared_handles():
    E = object()
    F = object()
    W_left = object()
    W_right = object()
    other_F = object()
    entries = AbelianPackedDirectFamilyEntries()
    entries.append_identity(1.0, E, F, source="id")
    entries.append_identity(2.5, E, F, source="id")
    entries.append_identity(3.0, E, other_F, source="id")
    entries.append_local_generator(1.0, E, W_left, W_right, F, source="local")
    entries.append_local_generator(-1.0, E, W_left, W_right, F, source="local")
    entries.extend_identity([4.0], [other_F], [F], source="bulk_id")
    entries.extend_local_generators(
        [5.0],
        [E],
        [W_left],
        [W_right],
        [other_F],
        source="bulk_local",
    )

    stats = entries.coalesce_in_place()

    assert stats["before"] == 7
    assert stats["after"] == 4
    assert stats["reduction"] == 3
    assert stats["cancelled_local"] == 1
    identity_entries = [
        entry
        for entry in entries
        if isinstance(entry, AbelianPackedIdentityLocalEntry)
    ]
    local_entries = [
        entry
        for entry in entries
        if isinstance(entry, AbelianPackedLocalGeneratorEntry)
    ]
    assert [entry.coeff for entry in identity_entries] == [
        3.5 + 0.0j,
        3.0 + 0.0j,
        4.0 + 0.0j,
    ]
    assert [entry.coeff for entry in local_entries] == [5.0 + 0.0j]


def test_abelian_packed_direct_entries_coalesce_structural_boundary_tensors():
    def packed3(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0),),
            (np.asarray([[[value]]], dtype=np.complex128),),
            dirs=[-1, 1, -1],
            qns=[[0], [0], [0]],
        )

    def packed4(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0, 0),),
            (np.asarray([[[[value]]]], dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[0], [0], [0], [0]],
        )

    entries = AbelianPackedDirectFamilyEntries()
    entries.append_identity(1.0, packed3(1.0), packed3(2.0), source="id")
    entries.append_identity(2.0, packed3(1.0), packed3(2.0), source="id")
    entries.append_local_generator(
        1.0,
        packed3(1.0),
        packed4(3.0),
        packed4(4.0),
        packed3(2.0),
        source="local",
    )
    entries.append_local_generator(
        -1.0,
        packed3(1.0),
        packed4(3.0),
        packed4(4.0),
        packed3(2.0),
        source="local",
    )

    stats = entries.coalesce_in_place()

    assert stats["before"] == 4
    assert stats["after"] == 1
    assert stats["reduction"] == 3
    assert stats["cancelled_local"] == 1
    assert entries.identity_count == 1
    assert entries.local_generator_count == 0
    assert entries[0].coeff == 3.0 + 0.0j


def test_contextual_component_store_coalesces_packed_buffer_entries():
    def packed3(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0),),
            (np.asarray([[[value]]], dtype=np.complex128),),
            dirs=[-1, 1, -1],
            qns=[[0], [0], [0]],
        )

    def packed4(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0, 0),),
            (np.asarray([[[[value]]]], dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[0], [0], [0], [0]],
        )

    component_table = AbelianNativeExactPatternComponentTable(bond=1)
    stats = {"native_boundary_p": {"validation_policy": "off"}}
    store = AbelianContextualComponentStore(
        component_table=component_table,
        family_options=SimpleNamespace(
            exact_component_compression_policy="auto",
            exact_component_compression_validate=False,
            exact_component_compression_min_reduction=1,
            exact_component_compression_max_group_size=64,
        ),
        matvec_options={
            "generator_table_exact_component_compression_fast_max_group_size": 1,
            "generator_table_coalesce_contextual_entries": True,
        },
        stats=stats,
        record_phase=lambda *_args, **_kwargs: None,
        validate_entries=lambda *_args, **_kwargs: True,
        bond=1,
    )
    entries = AbelianPackedDirectFamilyEntries()
    entries.append_local_generator(
        1.0,
        packed3(1.0),
        packed4(3.0),
        packed4(4.0),
        packed3(2.0),
        source="same",
    )
    entries.append_local_generator(
        2.0,
        packed3(1.0),
        packed4(3.0),
        packed4(4.0),
        packed3(2.0),
        source="same",
    )

    stored = store.store("P", entries, ((("L",), "A", "B", ("R",), 1.0),))

    assert len(stored) == 1
    assert stored.entries.local_generator_count == 1
    assert stored.entries[0].coeff == 3.0 + 0.0j
    assert stats["contextual_entry_coalesce"]["last_reduction"] == 1


def test_abelian_packed_boundary_tensor_exposes_columnar_blocks_lazily():
    keys = (("q0", "q1", "q2"), ("q1", "q2", "q3"))
    blocks = (
        np.ones((1, 2, 3), dtype=complex),
        np.arange(8, dtype=float).reshape(2, 2, 2),
    )

    tensor = AbelianPackedBoundaryTensor(
        keys,
        blocks,
        dirs=(-1, 1, -1),
        qns=(("q0", "q1"), ("q1", "q2"), ("q2", "q3")),
        source="unit",
    )

    assert is_abelian_packed_boundary_tensor(tensor)
    assert tensor.keys == keys
    assert tensor.blocks[0] is blocks[0]
    assert tensor.dirs == [-1, 1, -1]
    assert tensor.qns[0] == ("q0", "q1")
    assert tensor._data is None
    assert list(tensor.data) == list(keys)
    assert tensor._data is tensor.data
    assert pack_abelian_boundary_tensor(tensor) is tensor
    assert tensor.block_shape_signature() == (
        ("('q0', 'q1', 'q2')", (1, 2, 3)),
        ("('q1', 'q2', 'q3')", (2, 2, 2)),
    )


def test_abelian_packed_boundary_tensor_coalesces_duplicate_keys():
    first = np.ones((1, 1, 1), dtype=complex)
    second = 2.0 * np.ones((1, 1, 1), dtype=complex)

    tensor = AbelianPackedBoundaryTensor(
        (("q0", "q1", "q2"), ("q0", "q1", "q2")),
        (first, second),
    )

    assert tensor.keys == (("q0", "q1", "q2"),)
    assert len(tensor.blocks) == 1
    np.testing.assert_allclose(tensor.blocks[0], first + second)
    assert len(tensor.data) == 1


def test_abelian_packed_boundary_tensor_axis_filter_prunes_blocks_and_qns():
    tensor = AbelianPackedBoundaryTensor(
        (
            ("l0", "m0", "p0", "p0"),
            ("l0", "m1", "p1", "p0"),
            ("l1", "m2", "p0", "p1"),
        ),
        (
            np.full((1, 1, 1, 1), 1.0, dtype=complex),
            np.full((1, 2, 1, 1), 2.0, dtype=complex),
            np.full((2, 1, 1, 1), 3.0, dtype=complex),
        ),
        dirs=(-1, 1, 1, -1),
        qns=(
            ("l0", "l1"),
            ("m0", "m1", "m2"),
            ("p0", "p1"),
            ("p0", "p1"),
        ),
        source="unit",
    )

    filtered = filter_abelian_packed_boundary_tensor_axis(
        tensor,
        1,
        {"m1", "m2"},
        source="filtered",
    )

    assert filtered.source == "filtered"
    assert filtered.keys == (
        ("l0", "m1", "p1", "p0"),
        ("l1", "m2", "p0", "p1"),
    )
    assert filtered.dirs == [-1, 1, 1, -1]
    assert filtered.qns == [["l0", "l1"], ["m1", "m2"], ["p0", "p1"], ["p0", "p1"]]
    assert filtered.blocks[0] is tensor.blocks[1]
    assert filtered.blocks[1] is tensor.blocks[2]


def test_abelian_packed_tensor_ops_transpose_conjugate_and_tensordot():
    left = AbelianPackedBoundaryTensor(
        (("a", "x"), ("b", "y")),
        (
            np.ones((2, 3), dtype=np.complex128),
            2.0 * np.ones((2, 3), dtype=np.complex128),
        ),
        dirs=(-1, 1),
        qns=(("a", "b"), ("x", "y")),
        source="left",
    )
    right = AbelianPackedBoundaryTensor(
        (("x", "r"), ("z", "drop"), ("y", "s")),
        (
            3.0 * np.ones((3, 4), dtype=np.complex128),
            5.0 * np.ones((3, 4), dtype=np.complex128),
            (4.0 + 1.0j) * np.ones((3, 4), dtype=np.complex128),
        ),
        dirs=(-1, 1),
        qns=(("x", "y", "z"), ("drop", "r", "s")),
        source="right",
    )

    keys, blocks, dirs, qns = abelian_packed_tensor_items(left)
    assert keys == left.keys
    assert blocks == left.blocks
    assert dirs == [-1, 1]
    assert qns == left.qns

    conj = conjugate_abelian_packed_boundary_tensor(right, source="conj")
    assert conj.source == "conj"
    assert conj.dirs == [1, -1]
    np.testing.assert_allclose(conj.blocks[2], np.conj(right.blocks[2]))

    class LegacyTensor:
        pass

    legacy = LegacyTensor()
    legacy.data = {("a", "x"): np.ones((2, 3), dtype=np.complex128)}
    legacy.dirs = [-1, 1]
    legacy.qns = [["a"], ["x"]]
    view_cache = AbelianPackedTensorViewCache(source_prefix="unit")
    packed = view_cache.view(legacy, "legacy")
    assert packed is view_cache.view(legacy, "legacy_again")
    assert packed.source == "unit_legacy"
    packed_conj = view_cache.conj(legacy, "legacy_conj")
    assert packed_conj is view_cache.conj(legacy, "legacy_conj")
    assert view_cache.stats["created"] == 1
    assert view_cache.stats["blocks"] == 1
    assert view_cache.stats["conj_cache"] == 1

    transposed = transpose_abelian_packed_boundary_tensor(left, (1, 0))
    assert transposed.keys == (("x", "a"), ("y", "b"))
    assert transposed.dirs == [1, -1]
    assert transposed.qns == [("x", "y"), ("a", "b")]
    np.testing.assert_allclose(transposed.blocks[1], left.blocks[1].T)

    scaled = scale_abelian_boundary_tensor(left, 2.5j, source="scaled")
    assert is_abelian_packed_boundary_tensor(scaled)
    assert scaled.source == "scaled"
    assert scaled.keys == left.keys
    assert scaled.dirs == left.dirs
    assert scaled.qns == left.qns
    np.testing.assert_allclose(scaled.blocks[0], 2.5j * left.blocks[0])

    no_qns = AbelianPackedBoundaryTensor(
        (("u", "v"),),
        (np.ones((1, 1), dtype=np.complex128),),
        dirs=(-1, 1),
    )
    view = unpack_abelian_packed_boundary_tensor(no_qns)
    assert view.rank == 2
    assert view.dirs == [-1, 1]
    assert view.qns == [["u"], ["v"]]
    np.testing.assert_allclose(view.data[("u", "v")], no_qns.blocks[0])

    contracted = tensordot_abelian_packed_boundary_tensors(
        left,
        right,
        axes=([1], [0]),
        source="contracted",
    )
    contracted_with_map = tensordot_abelian_packed_boundary_tensors(
        left,
        right,
        axes=([1], [0]),
        right_axis_map=abelian_packed_tensor_axis_map(right, [0]),
        source="contracted_with_map",
    )
    assert contracted.source == "contracted"
    assert contracted.keys == (("a", "r"), ("b", "s"))
    assert contracted.dirs == [-1, 1]
    assert contracted.qns == [("a", "b"), ("drop", "r", "s")]
    assert contracted_with_map.keys == contracted.keys
    assert contracted_with_map.dirs == contracted.dirs
    assert contracted_with_map.qns == contracted.qns
    np.testing.assert_allclose(
        contracted.blocks[0],
        np.tensordot(left.blocks[0], right.blocks[0], axes=([1], [0])),
    )
    np.testing.assert_allclose(
        contracted.blocks[1],
        np.tensordot(left.blocks[1], right.blocks[2], axes=([1], [0])),
    )
    for lhs, rhs in zip(contracted_with_map.blocks, contracted.blocks):
        np.testing.assert_allclose(lhs, rhs)


def test_abelian_local_vector_layout_flattens_and_builds_basis_data():
    class LocalTensor:
        pass

    tensor = LocalTensor()
    tensor.data = {
        ("b", "x"): np.asarray([[5.0, 6.0]], dtype=np.complex128),
        ("a", "x"): np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128),
    }
    tensor.qns = [["a", "b"], ["x"]]
    tensor.dirs = [-1, 1]

    layout = AbelianLocalVectorLayout.from_tensor(tensor)
    assert layout.layout == (
        (("a", "x"), (2, 2)),
        (("b", "x"), (1, 2)),
    )
    assert layout.qns == (("a", "b"), ("x",))
    assert layout.dirs == (-1, 1)
    assert layout.size == 6
    assert layout.offsets == ({("a", "x"): (0, 4), ("b", "x"): (4, 2)}, 6)

    flat = layout.flatten_tensor(tensor)
    np.testing.assert_allclose(flat, np.asarray([1, 2, 3, 4, 5, 6], dtype=complex))
    expanded = AbelianLocalVectorLayout.from_layout(
        (
            (("a", "x"), (2, 2)),
            (("b", "x"), (1, 2)),
            (("c", "x"), (1,)),
        ),
        qns=[["a", "b", "c"], ["x"]],
        dirs=[-1, 1],
    )
    np.testing.assert_allclose(
        expanded.flatten_data(tensor.data, dtype=np.complex128),
        np.asarray([1, 2, 3, 4, 5, 6, 0], dtype=np.complex128),
    )
    rebuilt = layout.unflatten_data(flat)
    np.testing.assert_allclose(rebuilt[("a", "x")], tensor.data[("a", "x")])
    np.testing.assert_allclose(rebuilt[("b", "x")], tensor.data[("b", "x")])

    basis = layout.basis_data(5)
    assert tuple(basis) == (("b", "x"),)
    np.testing.assert_allclose(basis[("b", "x")], np.asarray([[0.0, 1.0]]))
    zero = layout.zero_data(dtype=np.float64)
    assert zero[("a", "x")].dtype == np.float64
    assert zero[("a", "x")].shape == (2, 2)


def test_abelian_flat_layout_helpers_are_block_data_only():
    class LocalTensor:
        pass

    tensor = LocalTensor()
    tensor.data = {
        ("a", "x"): np.asarray([1.0, 2.0], dtype=np.float32),
        ("b", "x"): np.asarray([[3.0]], dtype=np.float64),
    }
    tensor.qns = [["a", "b"], ["x"]]
    tensor.dirs = [-1, 1]
    layout = ((("a", "x"), (2,)), (("b", "x"), (1, 1)))

    assert abelian_block_data_dtype(tensor) == np.dtype(np.float64)
    np.testing.assert_allclose(
        abelian_flatten_to_layout(tensor, layout),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
    )
    data, qns, dirs = abelian_unflatten_data_from_layout(
        np.asarray([4.0, 5.0, 6.0], dtype=np.float64),
        layout,
        proto=tensor,
    )
    assert qns == [["a", "b"], ["x"]]
    assert dirs == [-1, 1]
    np.testing.assert_allclose(data[("a", "x")], np.asarray([4.0, 5.0]))
    np.testing.assert_allclose(data[("b", "x")], np.asarray([[6.0]]))

    zero_data, zero_qns, zero_dirs = abelian_zero_data_from_layout(
        layout,
        proto=tensor,
        dtype=np.complex128,
    )
    assert zero_qns == qns
    assert zero_dirs == dirs
    assert zero_data[("a", "x")].dtype == np.complex128
    np.testing.assert_allclose(zero_data[("b", "x")], np.zeros((1, 1)))
    assert abelian_layout_offsets(layout) == (
        {("a", "x"): (0, 2), ("b", "x"): (2, 1)},
        3,
    )


def test_abelian_safe_two_site_layout_helpers_are_block_data_only():
    class TensorData:
        rank = 4

    proto = TensorData()
    proto.data = {
        (0, 0, 0, 0): np.zeros((1, 1, 1, 1)),
        (0, 2, 0, 0): np.zeros((1, 4, 1, 1)),
    }

    w_left = TensorData()
    w_left.data = {(0, 0, 1, 1): np.zeros((1, 1, 2, 2))}
    w_right = TensorData()
    w_right.data = {(0, 0, 1, 1): np.zeros((1, 1, 3, 3))}

    assert abelian_axis_sector_dims(proto, 1) == {0: 1, 2: 4}
    assert abelian_sector_signature((0, 2, 1, 1), [1, -1, 1, 1]) == 0
    assert abelian_two_site_mps_flow_valid((0, 1, 1, 0))
    assert not abelian_two_site_mps_flow_valid((0, 2, 1, 0))

    layout_map = abelian_safe_two_site_layout_map(proto, (w_left, w_right))
    assert layout_map[(0, 0, 0, 0)] == (1, 1, 1, 1)
    assert layout_map[(0, 2, 1, 1)] == (1, 4, 2, 3)
    assert abelian_layout_from_map({("b",): (2,), ("a",): (1,)}) == (
        (("a",), (1,)),
        (("b",), (2,)),
    )


def test_abelian_merge_layout_tensor_enforces_allowed_maps_and_flow():
    class TensorData:
        pass

    tensor = TensorData()
    tensor.data = {
        (0, 1, 1, 0): np.zeros((2, 3, 1, 1)),
        (0, 2, 1, 1): np.zeros((2, 4, 1, 1)),
    }
    layout_map = {(0, 1, 1, 0): (2, 3, 1, 1)}
    allowed = {
        (0, 1, 1, 0): (2, 3, 1, 1),
        (0, 2, 1, 1): (2, 4, 1, 1),
    }

    merged, changed = abelian_merge_layout_tensor(
        layout_map,
        tensor,
        allowed_layout_map=allowed,
        require_two_site_mps_flow=True,
    )
    assert changed is True
    assert merged == abelian_layout_from_map(allowed)
    assert layout_map == allowed

    bad_shape = TensorData()
    bad_shape.data = {(0, 1, 1, 0): np.zeros((2, 2, 1, 1))}
    assert (
        abelian_merge_layout_tensor(
            dict(layout_map),
            bad_shape,
            allowed_layout_map=allowed,
        )
        == (None, False)
    )
    bad_flow = TensorData()
    bad_flow.data = {(0, 3, 1, 0): np.zeros((2, 3, 1, 1))}
    assert (
        abelian_merge_layout_tensor(
            dict(layout_map),
            bad_flow,
            require_two_site_mps_flow=True,
        )
        == (None, False)
    )


def test_abelian_truncate_layout_map_by_norm_keeps_largest_blocks():
    class TensorData:
        pass

    tensor = TensorData()
    tensor.data = {
        ("large",): np.asarray([3.0, 4.0]),
        ("small",): np.asarray([1.0, 0.0]),
        ("wide",): np.ones(4),
    }
    layout_map = {
        ("large",): (2,),
        ("small",): (1,),
        ("wide",): (4,),
    }

    result = abelian_truncate_layout_map_by_norm(
        layout_map,
        tensor,
        max_dim=3,
        current_dim=7,
    )

    assert result.truncated is True
    assert result.layout_map == {("large",): (2,), ("small",): (1,)}
    assert result.retained_blocks == 2
    assert result.retained_norm == pytest.approx((26.0 / 30.0) ** 0.5)

    unchanged = abelian_truncate_layout_map_by_norm(layout_map, tensor, max_dim=8)
    assert unchanged.truncated is False
    assert unchanged.layout_map == layout_map
    too_small = abelian_truncate_layout_map_by_norm(layout_map, tensor, max_dim=0)
    assert too_small.truncated is False
    assert too_small.layout_map == layout_map


def test_abelian_flat_qchem_jacobi_diagonal_uses_block_data_only():
    class TensorData:
        pass

    E = TensorData()
    E.data = {("e0", "L", "L"): 2.0 * np.ones((1, 1, 1))}
    W1 = TensorData()
    W1.data = {
        ("e0", "c0", "p", "p"): np.diag([3.0, 5.0]).reshape(1, 1, 2, 2)
    }
    W2 = TensorData()
    W2.data = {
        ("c0", "f0", "q", "q"): np.diag([7.0, 11.0, 13.0]).reshape(
            1,
            1,
            3,
            3,
        )
    }
    F = TensorData()
    F.data = {("f0", "R", "R"): 17.0 * np.ones((1, 1, 1))}
    layout = ((("L", "R", "p", "q"), (1, 1, 2, 3)),)

    result = abelian_flat_qchem_jacobi_diagonal(layout, E, (W1, W2), F)

    assert result.rejected_reason is None
    assert result.candidate_entries == 1
    assert result.contributions == 1
    expected = 2.0 * 17.0 * np.asarray(
        [3.0 * 7.0, 3.0 * 11.0, 3.0 * 13.0, 5.0 * 7.0, 5.0 * 11.0, 5.0 * 13.0],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(result.flat, expected)
    assert not hasattr(result.block_data, "data")


def test_abelian_block_preconditioner_build_and_apply_are_flat_only():
    layout = ((("a",), (2,)), (("b",), (1,)), (("c",), (2,)))
    matrix = np.asarray(
        [
            [2.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 5.0],
        ],
        dtype=np.complex128,
    )

    result = abelian_build_block_preconditioner_blocks(
        layout,
        lambda basis, _layout: matrix @ basis,
        max_block_dim=2,
        max_total_dim=4,
    )

    assert result.used_dim == 4
    assert result.attempted_blocks == 2
    assert result.failed_blocks == 0
    assert result.skipped_blocks == 1
    assert result.columns == 4
    assert tuple(result.blocks) == (("a",), ("c",))
    np.testing.assert_allclose(result.blocks[("a",)][2], matrix[:2, :2])
    np.testing.assert_allclose(result.blocks[("c",)][2], matrix[3:5, 3:5])

    resid = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.complex128)
    base = np.asarray([0.0, 0.0, 9.0, 0.0, 0.0], dtype=np.complex128)
    out = abelian_apply_block_preconditioner(
        resid,
        10.0,
        base,
        result.blocks,
    )
    expected = base.copy()
    expected[:2] = np.linalg.solve(10.0 * np.eye(2) - matrix[:2, :2], resid[:2])
    expected[3:5] = np.linalg.solve(
        10.0 * np.eye(2) - matrix[3:5, 3:5],
        resid[3:5],
    )
    np.testing.assert_allclose(out, expected)


def test_abelian_jacobi_preconditioner_apply_is_flat_only():
    resid = np.asarray([2.0, 4.0, 6.0], dtype=np.complex128)
    diagonal = np.asarray([1.0, 2.0, 3.0], dtype=np.complex128)
    np.testing.assert_allclose(
        abelian_apply_jacobi_preconditioner(resid, 5.0, diagonal),
        np.asarray([0.5, 4.0 / 3.0, 3.0], dtype=np.complex128),
    )
    floored = abelian_apply_jacobi_preconditioner(
        np.asarray([1.0, 1.0]),
        2.0,
        np.asarray([2.0, 2.0 + 1.0e-12]),
        floor=1.0e-3,
    )
    np.testing.assert_allclose(floored, np.asarray([1.0e3, -1.0e3]))
    sanitized = abelian_apply_jacobi_preconditioner(
        np.asarray([1.0, 2.0, 3.0]),
        2.0,
        np.asarray([np.nan, np.inf, 2.0]),
        floor=1.0e-2,
    )
    assert np.all(np.isfinite(sanitized))
    np.testing.assert_allclose(sanitized, np.asarray([-100.0, -200.0, 300.0]))
    assert (
        abelian_apply_jacobi_preconditioner(
            np.asarray([1.0, 2.0]),
            0.0,
            np.asarray([1.0]),
        )
        is None
    )


def test_abelian_flat_davidson_helpers_build_ritz_and_candidates():
    h = np.asarray([[2.0, 0.5], [0.5, 3.0]], dtype=np.complex128)
    e0 = np.asarray([1.0, 0.0], dtype=np.complex128)
    e1 = np.asarray([0.0, 1.0], dtype=np.complex128)

    T = np.zeros((0, 0), dtype=np.complex128)
    T = abelian_extend_projected_hamiltonian(T, (e0,), h @ e0)
    T = abelian_extend_projected_hamiltonian(T, (e0, e1), h @ e1)
    np.testing.assert_allclose(T, h)

    ritz = abelian_lowest_ritz_state(T, (e0, e1), (h @ e0, h @ e1))
    expected_values, expected_vectors = np.linalg.eigh(h)
    assert ritz.energy == pytest.approx(expected_values[0])
    np.testing.assert_allclose(
        np.abs(np.vdot(expected_vectors[:, 0], ritz.vector)),
        1.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(ritz.residual, np.zeros(2), atol=1.0e-12)

    restarted = abelian_restart_basis_from_vector(2.0 * e0)
    np.testing.assert_allclose(restarted, e0)
    assert abelian_restart_basis_from_vector(np.zeros(2), min_norm=1.0e-12) is None
    normalized = abelian_normalize_flat_vector(3.0 * e1)
    assert normalized.accepted is True
    assert normalized.norm == pytest.approx(3.0)
    np.testing.assert_allclose(normalized.vector, e1)
    rejected = abelian_normalize_flat_vector(np.zeros(2), min_norm=1.0e-12)
    assert rejected.accepted is False
    assert rejected.vector is None

    q, qn = abelian_orthogonalize_candidate(
        np.asarray([1.0, 2.0], dtype=np.complex128),
        (e0,),
    )
    assert qn == pytest.approx(2.0)
    np.testing.assert_allclose(q, e1)
    q, qn = abelian_orthogonalize_candidate(e0, (e0,), min_norm=1.0e-9)
    assert q is None
    assert qn == pytest.approx(0.0)
    q, qn = abelian_orthogonalize_candidate(
        np.asarray([np.nan, 1.0], dtype=np.complex128),
        (),
    )
    assert q is None
    assert np.isnan(qn)


def test_abelian_two_site_svd_from_permuted_data_matches_legacy_wrapper():
    permuted_data = {
        (0, 0, 0, 0): np.asarray([[[[1.0, 0.2]], [[0.3, 2.0]]]]),
        (0, 1, 1, 0): np.asarray([[[[0.7]]]]),
    }
    original_data = {
        (q_left, q_right, q_phys_left, q_phys_right): block.transpose(0, 2, 1, 3)
        for (q_left, q_phys_left, q_right, q_phys_right), block in permuted_data.items()
    }
    aa = BlockTensor(
        original_data,
        [[0], [0, 1], [0, 1], [0]],
        [-1, 1, 1, 1],
    )

    direct = abelian_two_site_svd_from_permuted_data(permuted_data, m_max=2)
    U, V, S, trunc, kept = svd_symmetric(aa, m_max=2)
    native_U, native_V, native_S, native_trunc, native_kept = svd_symmetric(
        AbelianSiteTensorData(aa.data, aa.qns, aa.dirs),
        m_max=2,
    )

    assert direct.kept_states == kept
    assert direct.bond_qns == U.qns[2]
    assert direct.bond_qns == V.qns[0]
    assert direct.truncation_error == pytest.approx(trunc)
    assert set(direct.u_data) == set(U.data)
    assert set(direct.v_data) == set(V.data)
    assert set(direct.s_data) == set(S)
    for key, block in direct.u_data.items():
        np.testing.assert_allclose(block, U.data[key], atol=1.0e-12)
    for key, block in direct.v_data.items():
        np.testing.assert_allclose(block, V.data[key], atol=1.0e-12)
    for key, block in direct.s_data.items():
        np.testing.assert_allclose(block, S[key], atol=1.0e-12)
    assert native_kept == kept
    assert native_trunc == pytest.approx(trunc)
    assert isinstance(native_U, AbelianSiteTensorData)
    assert isinstance(native_V, AbelianSiteTensorData)
    assert not isinstance(native_U, BlockTensor)
    assert not isinstance(native_V, BlockTensor)
    assert native_U.qns == tuple(tuple(axis) for axis in U.qns)
    assert native_V.qns == tuple(tuple(axis) for axis in V.qns)
    assert native_U.dirs == tuple(U.dirs)
    assert native_V.dirs == tuple(V.dirs)
    assert set(native_U.data) == set(U.data)
    assert set(native_V.data) == set(V.data)
    assert set(native_S) == set(S)
    for key, block in U.data.items():
        np.testing.assert_allclose(native_U.data[key], block, atol=1.0e-12)
    for key, block in V.data.items():
        np.testing.assert_allclose(native_V.data[key], block, atol=1.0e-12)
    for key, block in S.items():
        np.testing.assert_allclose(native_S[key], block, atol=1.0e-12)


def test_abelian_two_site_svd_native_split_matches_python_reference():
    permuted_data = {
        (0, 0, 0, 0): np.asarray([[[[1.0, 0.2]], [[0.3, 2.0]]]]),
        (0, 1, 1, 0): np.asarray([[[[0.7]]]]),
    }
    direct = abelian_two_site_svd_from_permuted_data(permuted_data, m_max=2)
    reference = _abelian_two_site_svd_from_permuted_data_python(permuted_data, m_max=2)

    assert direct.bond_qns == reference.bond_qns
    assert direct.kept_states == reference.kept_states
    assert direct.truncation_error == pytest.approx(reference.truncation_error)
    assert set(direct.u_data) == set(reference.u_data)
    assert set(direct.v_data) == set(reference.v_data)
    assert set(direct.s_data) == set(reference.s_data)
    for key, block in reference.u_data.items():
        np.testing.assert_allclose(direct.u_data[key], block, atol=1.0e-12)
    for key, block in reference.v_data.items():
        np.testing.assert_allclose(direct.v_data[key], block, atol=1.0e-12)
    for key, block in reference.s_data.items():
        np.testing.assert_allclose(direct.s_data[key], block, atol=1.0e-12)


def test_abelian_two_site_svd_native_split_supports_qn_sectors():
    z = QN(0)
    o = QN(1)
    permuted_data = {
        (z, z, z, z): np.asarray([[[[1.0, 0.2]], [[0.3, 2.0]]]]),
        (z, o, o, z): np.asarray([[[[0.7]]]]),
    }
    direct = abelian_two_site_svd_from_permuted_data(permuted_data, m_max=2)
    reference = _abelian_two_site_svd_from_permuted_data_python(permuted_data, m_max=2)

    assert direct.bond_qns == reference.bond_qns
    assert direct.kept_states == reference.kept_states
    assert direct.truncation_error == pytest.approx(reference.truncation_error)
    assert set(direct.u_data) == set(reference.u_data)
    assert set(direct.v_data) == set(reference.v_data)
    assert set(direct.s_data) == set(reference.s_data)
    for key, block in reference.u_data.items():
        np.testing.assert_allclose(direct.u_data[key], block, atol=1.0e-12)
    for key, block in reference.v_data.items():
        np.testing.assert_allclose(direct.v_data[key], block, atol=1.0e-12)
    for key, block in reference.s_data.items():
        np.testing.assert_allclose(direct.s_data[key], block, atol=1.0e-12)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_abelian_state_averaged_two_site_svd_matches_legacy_wrapper(direction):
    data0 = {
        (0, 0, 0, 0): np.asarray([[[[1.0, 0.2]], [[0.3, 2.0]]]]),
        (0, 1, 1, 0): np.asarray([[[[0.7]]]]),
    }
    data1 = {
        (0, 0, 0, 0): np.asarray([[[[0.4, 1.1]], [[1.7, 0.6]]]]),
        (0, 1, 1, 0): np.asarray([[[[1.3]]]]),
    }
    original = []
    for data in (data0, data1):
        original.append(
            {
                (q_left, q_right, q_phys_left, q_phys_right): block.transpose(
                    0,
                    2,
                    1,
                    3,
                )
                for (
                    q_left,
                    q_phys_left,
                    q_right,
                    q_phys_right,
                ), block in data.items()
            }
        )
    aa_list = [
        BlockTensor(data, [[0], [0, 1], [0, 1], [0]], [-1, 1, 1, 1])
        for data in original
    ]
    weights = [0.25, 0.75]

    direct = abelian_state_averaged_two_site_svd_from_permuted_data(
        (data0, data1),
        weights,
        direction,
        m_max=2,
    )
    U, V, S, trunc, kept = sa_svd_symmetric(
        aa_list,
        weights,
        dir=direction,
        m_max=2,
    )
    native_aa_list = [
        AbelianSiteTensorData(aa.data, aa.qns, aa.dirs)
        for aa in aa_list
    ]
    native_U, native_V, native_S, native_trunc, native_kept = sa_svd_symmetric(
        native_aa_list,
        weights,
        dir=direction,
        m_max=2,
    )

    assert direct.kept_states == kept
    assert direct.truncation_error == pytest.approx(trunc)
    assert direct.bond_qns == U.qns[2]
    assert direct.bond_qns == V.qns[0]
    assert set(direct.u_data) == set(U.data)
    assert set(direct.v_data) == set(V.data)
    assert set(direct.s_data) == set(S)
    for key, block in direct.u_data.items():
        np.testing.assert_allclose(block, U.data[key], atol=1.0e-12)
    for key, block in direct.v_data.items():
        np.testing.assert_allclose(block, V.data[key], atol=1.0e-12)
    for key, block in direct.s_data.items():
        np.testing.assert_allclose(block, S[key], atol=1.0e-12)
    assert native_kept == kept
    assert native_trunc == pytest.approx(trunc)
    assert isinstance(native_U, AbelianSiteTensorData)
    assert isinstance(native_V, AbelianSiteTensorData)
    assert not isinstance(native_U, BlockTensor)
    assert not isinstance(native_V, BlockTensor)
    assert native_U.qns == tuple(tuple(axis) for axis in U.qns)
    assert native_V.qns == tuple(tuple(axis) for axis in V.qns)
    assert native_U.dirs == tuple(U.dirs)
    assert native_V.dirs == tuple(V.dirs)
    assert set(native_U.data) == set(U.data)
    assert set(native_V.data) == set(V.data)
    assert set(native_S) == set(S)
    for key, block in U.data.items():
        np.testing.assert_allclose(native_U.data[key], block, atol=1.0e-12)
    for key, block in V.data.items():
        np.testing.assert_allclose(native_V.data[key], block, atol=1.0e-12)
    for key, block in S.items():
        np.testing.assert_allclose(native_S[key], block, atol=1.0e-12)


def test_abelian_singular_value_contraction_helpers_match_legacy_wrappers():
    q_mid = "m"
    u_data = {
        ("l", "p", q_mid): np.arange(6, dtype=np.complex128).reshape(1, 2, 3)
    }
    v_data = {
        (q_mid, "r", "q"): (1.0 + np.arange(12, dtype=np.complex128)).reshape(
            3,
            2,
            2,
        )
    }
    s_data = {q_mid: np.diag(np.asarray([2.0, 3.0, 5.0], dtype=np.complex128))}
    U = BlockTensor(u_data, [["l"], ["p"], [q_mid]], [-1, 1, 1])
    V = BlockTensor(v_data, [[q_mid], ["r"], ["q"]], [-1, 1, 1])

    direct_us = abelian_multiply_u_s_data(u_data, s_data)
    direct_sv = abelian_multiply_s_v_data(s_data, v_data)
    legacy_us = multiply_U_S(U, s_data)
    legacy_sv = multiply_S_V(s_data, V)
    native_us = multiply_U_S(
        AbelianSiteTensorData(U.data, U.qns, U.dirs),
        s_data,
    )
    native_sv = multiply_S_V(
        s_data,
        AbelianSiteTensorData(V.data, V.qns, V.dirs),
    )

    assert set(direct_us) == set(legacy_us.data)
    assert set(direct_sv) == set(legacy_sv.data)
    assert isinstance(native_us, AbelianSiteTensorData)
    assert isinstance(native_sv, AbelianSiteTensorData)
    assert not isinstance(native_us, BlockTensor)
    assert not isinstance(native_sv, BlockTensor)
    assert native_us.qns == tuple(tuple(axis) for axis in legacy_us.qns)
    assert native_sv.qns == tuple(tuple(axis) for axis in legacy_sv.qns)
    assert native_us.dirs == tuple(legacy_us.dirs)
    assert native_sv.dirs == tuple(legacy_sv.dirs)
    for key, block in direct_us.items():
        np.testing.assert_allclose(block, legacy_us.data[key], atol=1.0e-12)
        np.testing.assert_allclose(block, native_us.data[key], atol=1.0e-12)
    for key, block in direct_sv.items():
        np.testing.assert_allclose(block, legacy_sv.data[key], atol=1.0e-12)
        np.testing.assert_allclose(block, native_sv.data[key], atol=1.0e-12)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_abelian_split_two_site_svd_data_matches_legacy_split(direction):
    original_data = {
        (0, 0, 0, 0): np.asarray([[[[1.0, 0.2], [0.3, 2.0]]]]),
        (0, 1, 1, 0): np.asarray([[[[0.7]]]]),
    }
    aa = BlockTensor(
        original_data,
        [[0], [0, 1], [0, 1], [0]],
        [-1, 1, 1, 1],
    )

    split = abelian_split_two_site_svd_data(
        original_data,
        qns=aa.qns,
        dirs=aa.dirs,
        direction=direction,
        m_max=2,
    )
    U, V, S, trunc, kept = svd_symmetric(aa, m_max=2)
    if direction == "right":
        A_ref = U.transpose(0, 2, 1)
        B_ref = multiply_S_V(S, V)
    else:
        A_ref = multiply_U_S(U, S).transpose(0, 2, 1)
        B_ref = V

    assert split.kept_states == kept
    assert split.truncation_error == pytest.approx(trunc)
    assert split.a_qns == A_ref.qns
    assert split.b_qns == B_ref.qns
    assert split.a_dirs == A_ref.dirs
    assert split.b_dirs == B_ref.dirs
    assert set(split.a_data) == set(A_ref.data)
    assert set(split.b_data) == set(B_ref.data)
    for key, block in split.a_data.items():
        np.testing.assert_allclose(block, A_ref.data[key], atol=1.0e-12)
    for key, block in split.b_data.items():
        np.testing.assert_allclose(block, B_ref.data[key], atol=1.0e-12)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_abelian_split_two_site_svd_native_update_matches_python_reference(direction):
    z = QN(0)
    o = QN(1)
    original_data = {
        (z, z, z, z): np.asarray([[[[1.0, 0.2], [0.3, 2.0]]]]),
        (z, o, o, z): np.asarray([[[[0.7]]]]),
    }
    qns = [[z], [z, o], [z, o], [z]]
    dirs = [-1, 1, 1, 1]
    direct = abelian_split_two_site_svd_data(
        original_data,
        qns=qns,
        dirs=dirs,
        direction=direction,
        m_max=2,
    )
    reference = _abelian_split_two_site_svd_data_python(
        original_data,
        qns=qns,
        dirs=dirs,
        direction=direction,
        m_max=2,
    )

    assert direct.a_qns == reference.a_qns
    assert direct.b_qns == reference.b_qns
    assert direct.a_dirs == reference.a_dirs
    assert direct.b_dirs == reference.b_dirs
    assert direct.bond_qns == reference.bond_qns
    assert direct.kept_states == reference.kept_states
    assert direct.truncation_error == pytest.approx(reference.truncation_error)
    assert set(direct.a_data) == set(reference.a_data)
    assert set(direct.b_data) == set(reference.b_data)
    assert set(direct.s_data) == set(reference.s_data)
    for key, block in reference.a_data.items():
        np.testing.assert_allclose(direct.a_data[key], block, atol=1.0e-12)
    for key, block in reference.b_data.items():
        np.testing.assert_allclose(direct.b_data[key], block, atol=1.0e-12)
    for key, block in reference.s_data.items():
        np.testing.assert_allclose(direct.s_data[key], block, atol=1.0e-12)


def test_abelian_merge_adjacent_site_tensors_native_matches_python_reference():
    z = QN(0)
    o = QN(1)
    left = AbelianSiteTensorData(
        {
            (z, z, z): np.asarray([[[1.0], [2.0]]]),
            (z, o, z): np.asarray([[[0.5]]]),
        },
        [[z], [z, o], [z]],
        [-1, 1, 1],
    )
    right = AbelianSiteTensorData(
        {
            (z, z, z): np.asarray([[[3.0, 4.0]], [[5.0, 6.0]]]),
            (o, z, z): np.asarray([[[7.0, 8.0]]]),
        },
        [[z, o], [z], [z]],
        [-1, 1, 1],
    )

    direct = abelian_merge_adjacent_site_tensors(left, right)
    reference = _abelian_merge_adjacent_site_tensors_python(left, right)

    assert direct.qns == reference.qns
    assert direct.dirs == reference.dirs
    assert set(direct.data) == set(reference.data)
    for key, block in reference.data.items():
        np.testing.assert_allclose(direct.data[key], block, atol=1.0e-12)


def test_abelian_merge_normalize_adjacent_site_tensors_matches_reference():
    z = QN(0)
    o = QN(1)
    left = AbelianSiteTensorData(
        {
            (z, z, z): np.asarray([[[1.0 + 0.5j], [2.0 - 0.25j]]]),
            (z, o, z): np.asarray([[[0.5 - 0.125j]]]),
        },
        [[z], [z, o], [z]],
        [-1, 1, 1],
    )
    right = AbelianSiteTensorData(
        {
            (z, z, z): np.asarray([[[3.0], [4.0]], [[5.0], [6.0]]]),
            (o, z, z): np.asarray([[[7.0]]]),
        },
        [[z, o], [z], [z]],
        [-1, 1, 1],
    )

    direct, norm = abelian_merge_normalize_adjacent_site_tensors(left, right)
    reference = _abelian_merge_adjacent_site_tensors_python(left, right)
    reference_norm = reference.norm()

    assert norm == pytest.approx(reference_norm)
    assert direct.qns == reference.qns
    assert direct.dirs == reference.dirs
    assert set(direct.data) == set(reference.data)
    for key, block in reference.data.items():
        np.testing.assert_allclose(
            direct.data[key],
            np.asarray(block) / reference_norm,
            atol=1.0e-12,
        )
    assert direct.norm() == pytest.approx(1.0)


def test_abelian_merge_normalize_flatten_adjacent_site_tensors_matches_reference():
    z = QN(0)
    o = QN(1)
    left = AbelianSiteTensorData(
        {
            (z, z, z): np.asarray([[[1.0 + 0.5j], [2.0 - 0.25j]]]),
            (z, o, z): np.asarray([[[0.5 - 0.125j]]]),
        },
        [[z], [z, o], [z]],
        [-1, 1, 1],
    )
    right = AbelianSiteTensorData(
        {
            (z, z, z): np.asarray([[[3.0], [4.0]], [[5.0], [6.0]]]),
            (o, z, z): np.asarray([[[7.0]]]),
        },
        [[z, o], [z], [z]],
        [-1, 1, 1],
    )

    direct, norm, flat, layout = abelian_merge_normalize_flatten_adjacent_site_tensors(
        left,
        right,
    )
    reference = _abelian_merge_adjacent_site_tensors_python(left, right)
    reference_norm = reference.norm()
    reference = reference * (1.0 / reference_norm)
    reference_layout = tuple(
        (key, tuple(block.shape))
        for key, block in sorted(reference.data.items(), key=lambda item: item[0])
    )
    reference_flat = AbelianLocalVectorLayout.from_layout(
        layout,
        proto=reference,
    ).flatten_tensor(reference)

    assert norm == pytest.approx(reference_norm)
    assert direct.qns == reference.qns
    assert direct.dirs == reference.dirs
    assert tuple(layout) == reference_layout
    for key, block in reference.data.items():
        np.testing.assert_allclose(direct.data[key], block, atol=1.0e-12)
    np.testing.assert_allclose(flat, reference_flat, atol=1.0e-12)


def test_abelian_site_tensor_data_norm_and_scale_match_numpy_reference():
    z = QN(0)
    o = QN(1)
    data = {
        (z, z): np.asarray([1.0 + 2.0j, -0.5j]),
        (o, z): np.asarray([[3.0 - 1.0j, 0.25 + 0.75j]]),
    }
    tensor = AbelianSiteTensorData(data, [[z, o], [z]], [-1, 1])

    expected_norm = np.sqrt(
        sum(
            float(np.vdot(block.reshape(-1), block.reshape(-1)).real)
            for block in data.values()
        )
    )
    assert tensor.norm() == pytest.approx(expected_norm)

    scalar = 0.25 - 0.5j
    scaled = tensor.scaled(scalar)
    assert scaled.qns == tensor.qns
    assert scaled.dirs == tensor.dirs
    assert set(scaled.data) == set(data)
    for key, block in data.items():
        np.testing.assert_allclose(scaled.data[key], np.asarray(block) * scalar)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_abelian_split_flat_two_site_svd_data_matches_data_split(direction):
    original_data = {
        (0, 0, 0, 0): np.asarray([[[[1.0, 0.2], [0.3, 2.0]]]]),
        (0, 1, 1, 0): np.asarray([[[[0.7]]]]),
    }
    layout = tuple(
        (key, tuple(block.shape))
        for key, block in sorted(original_data.items(), key=lambda item: repr(item[0]))
    )
    local_layout = AbelianLocalVectorLayout.from_layout(
        layout,
        qns=[[0], [0, 1], [0, 1], [0]],
        dirs=[-1, 1, 1, 1],
    )
    flat = local_layout.flatten_data(original_data)

    direct = abelian_split_two_site_svd_data(
        original_data,
        qns=[[0], [0, 1], [0, 1], [0]],
        dirs=[-1, 1, 1, 1],
        direction=direction,
        m_max=2,
    )
    from_flat = abelian_split_flat_two_site_svd_data(
        flat,
        layout,
        qns=[[0], [0, 1], [0, 1], [0]],
        dirs=[-1, 1, 1, 1],
        direction=direction,
        m_max=2,
    )

    assert from_flat.kept_states == direct.kept_states
    assert from_flat.a_qns == direct.a_qns
    assert from_flat.b_qns == direct.b_qns
    for key, block in direct.a_data.items():
        np.testing.assert_allclose(from_flat.a_data[key], block, atol=1.0e-12)
    for key, block in direct.b_data.items():
        np.testing.assert_allclose(from_flat.b_data[key], block, atol=1.0e-12)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_abelian_split_flat_two_site_svd_native_update_matches_python_reference(direction):
    z = QN(0)
    o = QN(1)
    original_data = {
        (z, z, z, z): np.asarray([[[[1.0, 0.2], [0.3, 2.0]]]]),
        (z, o, o, z): np.asarray([[[[0.7]]]]),
    }
    layout = tuple(
        (key, tuple(block.shape))
        for key, block in sorted(original_data.items(), key=lambda item: repr(item[0]))
    )
    qns = [[z], [z, o], [z, o], [z]]
    dirs = [-1, 1, 1, 1]
    local_layout = AbelianLocalVectorLayout.from_layout(layout, qns=qns, dirs=dirs)
    flat = local_layout.flatten_data(original_data)

    direct = abelian_split_flat_two_site_svd_data(
        flat,
        layout,
        qns=qns,
        dirs=dirs,
        direction=direction,
        m_max=2,
    )
    reference = _abelian_split_two_site_svd_data_python(
        original_data,
        qns=qns,
        dirs=dirs,
        direction=direction,
        m_max=2,
    )

    assert direct.a_qns == reference.a_qns
    assert direct.b_qns == reference.b_qns
    assert direct.bond_qns == reference.bond_qns
    assert direct.kept_states == reference.kept_states
    assert direct.truncation_error == pytest.approx(reference.truncation_error)
    assert set(direct.a_data) == set(reference.a_data)
    assert set(direct.b_data) == set(reference.b_data)
    assert set(direct.s_data) == set(reference.s_data)
    for key, block in reference.a_data.items():
        np.testing.assert_allclose(direct.a_data[key], block, atol=1.0e-12)
    for key, block in reference.b_data.items():
        np.testing.assert_allclose(direct.b_data[key], block, atol=1.0e-12)
    for key, block in reference.s_data.items():
        np.testing.assert_allclose(direct.s_data[key], block, atol=1.0e-12)


def test_abelian_site_tensors_from_split_stays_native():
    split = abelian_split_two_site_svd_data(
        {
            (0, 0, 0, 0): np.asarray([[[[1.0, 0.2], [0.3, 2.0]]]]),
            (0, 1, 1, 0): np.asarray([[[[0.7]]]]),
        },
        qns=[[0], [0, 1], [0, 1], [0]],
        dirs=[-1, 1, 1, 1],
        direction="right",
        m_max=2,
    )

    update = abelian_site_tensors_from_split(split)

    assert not isinstance(update.left, BlockTensor)
    assert not isinstance(update.right, BlockTensor)
    assert update.left.qns == tuple(tuple(axis) for axis in split.a_qns)
    assert update.right.qns == tuple(tuple(axis) for axis in split.b_qns)
    assert update.left.dirs == tuple(split.a_dirs)
    assert update.right.dirs == tuple(split.b_dirs)
    assert update.left.rank == 3
    assert update.right.rank == 3
    assert update.left.block_layout() == tuple(
        (key, tuple(block.shape)) for key, block in update.left.data.items()
    )
    assert update.left.norm() > 0.0
    assert update.right.norm() > 0.0
    for key, block in split.a_data.items():
        np.testing.assert_allclose(update.left.data[key], block)
    for key, block in split.b_data.items():
        np.testing.assert_allclose(update.right.data[key], block)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_abelian_split_state_averaged_two_site_svd_data_matches_legacy_split(direction):
    data0 = {
        (0, 0, 0, 0): np.asarray([[[[1.0, 0.2], [0.3, 2.0]]]]),
        (0, 1, 1, 0): np.asarray([[[[0.7]]]]),
    }
    data1 = {
        (0, 0, 0, 0): np.asarray([[[[0.4, 1.1], [1.7, 0.6]]]]),
        (0, 1, 1, 0): np.asarray([[[[1.3]]]]),
    }
    aa_list = [
        BlockTensor(data, [[0], [0, 1], [0, 1], [0]], [-1, 1, 1, 1])
        for data in (data0, data1)
    ]
    weights = [0.25, 0.75]

    split = abelian_split_state_averaged_two_site_svd_data(
        (data0, data1),
        weights,
        qns=aa_list[0].qns,
        dirs=aa_list[0].dirs,
        direction=direction,
        m_max=2,
    )
    U, V, S, trunc, kept = sa_svd_symmetric(
        aa_list,
        weights,
        dir=direction,
        m_max=2,
    )
    if direction == "right":
        A_ref = U.transpose(0, 2, 1)
        B_ref = multiply_S_V(S, V)
    else:
        A_ref = multiply_U_S(U, S).transpose(0, 2, 1)
        B_ref = V

    assert split.kept_states == kept
    assert split.truncation_error == pytest.approx(trunc)
    assert split.a_qns == A_ref.qns
    assert split.b_qns == B_ref.qns
    assert split.a_dirs == A_ref.dirs
    assert split.b_dirs == B_ref.dirs
    assert set(split.a_data) == set(A_ref.data)
    assert set(split.b_data) == set(B_ref.data)
    for key, block in split.a_data.items():
        np.testing.assert_allclose(block, A_ref.data[key], atol=1.0e-12)
    for key, block in split.b_data.items():
        np.testing.assert_allclose(block, B_ref.data[key], atol=1.0e-12)


def test_abelian_project_block_data_to_layout_policies():
    layout = ((("a", "x"), (2,)), (("b", "x"), (1,)))
    data = {
        ("a", "x"): np.asarray([1.0, 2.0], dtype=np.complex128),
        ("b", "x"): np.asarray([3.0], dtype=np.complex128),
        ("extra", "x"): np.asarray([9.0], dtype=np.complex128),
    }

    projected = abelian_project_block_data_to_layout(
        data,
        layout,
        extra_policy="ignore",
    )
    np.testing.assert_allclose(projected, np.asarray([1.0, 2.0, 3.0]))
    assert (
        abelian_project_block_data_to_layout(
            data,
            layout,
            extra_policy="zero",
            extra_zero_tol=1.0e-14,
        )
        is None
    )
    zero_extra = dict(data)
    zero_extra[("extra", "x")] = np.zeros(1, dtype=np.complex128)
    np.testing.assert_allclose(
        abelian_project_block_data_to_layout(
            zero_extra,
            layout,
            extra_policy="zero",
            extra_zero_tol=1.0e-14,
        ),
        np.asarray([1.0, 2.0, 3.0]),
    )
    bad_shape = dict(data)
    bad_shape[("a", "x")] = np.ones((1, 2), dtype=np.complex128)
    assert (
        abelian_project_block_data_to_layout(
            bad_shape,
            layout,
            extra_policy="ignore",
        )
        is None
    )


def test_abelian_project_tensor_to_layout_with_stats_counts_discarded_blocks():
    class TensorData:
        pass

    tensor = TensorData()
    tensor.data = {
        ("keep",): np.asarray([1.0, 2.0], dtype=np.complex128),
        ("drop",): np.asarray([3.0, 4.0], dtype=np.complex128),
    }
    tensor.qns = [["keep", "drop"]]
    tensor.dirs = [1]
    layout = ((("keep",), (2,)),)

    result = abelian_project_tensor_to_layout_with_stats(
        tensor,
        layout,
        dtype=np.complex128,
    )

    np.testing.assert_allclose(result.flat, np.asarray([1.0, 2.0]))
    assert result.discarded_blocks == 1
    assert result.discarded_norm_sq == pytest.approx(25.0)

    bad = TensorData()
    bad.data = {("keep",): np.zeros((1, 2))}
    failed = abelian_project_tensor_to_layout_with_stats(bad, layout)
    assert failed.flat is None
    assert failed.discarded_blocks == 0


def test_abelian_remap_flat_layout_preserves_compatible_blocks():
    old_layout = ((("a",), (2,)), (("c",), (1,)))
    new_layout = ((("a",), (2,)), (("b",), (3,)), (("c",), (1,)))
    old_vec = np.asarray([1.0, 2.0, 9.0], dtype=np.complex128)

    new_vec = abelian_remap_flat_layout(old_vec, old_layout, new_layout)
    np.testing.assert_allclose(
        new_vec,
        np.asarray([1.0, 2.0, 0.0, 0.0, 0.0, 9.0], dtype=np.complex128),
    )
    with pytest.raises(ValueError):
        abelian_remap_flat_layout(
            old_vec,
            old_layout,
            ((("a",), (3,)), (("c",), (1,))),
        )
    with pytest.raises(ValueError):
        abelian_remap_flat_layout(old_vec, old_layout, ((("a",), (2,)),))


def test_abelian_action_tables_use_local_vector_layout_apply_path():
    layout = ((("a", "x"), (2,)), (("b", "x"), (1,)))
    qns = [["a", "b"], ["x"]]
    dirs = [-1, 1]
    A = BlockTensor(
        {
            ("a", "x"): np.asarray([1.0, 2.0], dtype=np.complex128),
            ("b", "x"): np.asarray([3.0], dtype=np.complex128),
        },
        qns,
        dirs,
    )
    native_A = AbelianSiteTensorData(A.data, A.qns, A.dirs)
    matrix = np.asarray(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=np.complex128,
    )
    dense = AbelianComplementaryBoundaryActionTable(
        matrix,
        layout,
        qns,
        dirs,
        source="unit_dense_table",
        channel_matrices={"full": matrix},
    )
    sparse = AbelianSparseComplementaryBoundaryActionTable(
        rows=np.asarray([0, 0, 1, 2, 2], dtype=np.int64),
        cols=np.asarray([0, 2, 1, 0, 2], dtype=np.int64),
        values=np.asarray([2.0, 1.0, 3.0, 4.0, 5.0], dtype=np.complex128),
        dim=3,
        layout=layout,
        qns=qns,
        dirs=dirs,
        source="unit_sparse_table",
    )
    renorm = AbelianRenormalizedOperatorActionTable(
        {},
        dim=3,
        layout=layout,
        qns=qns,
        dirs=dirs,
        source="unit_empty_renormalized_table",
        kernel_backend=None,
    )

    dense_out = dense.apply(A)
    sparse_out = sparse.apply(A)
    native_dense_out = dense.apply(native_A)
    native_sparse_out = sparse.apply(native_A)
    native_renorm_out = renorm.apply(native_A)
    native_channels = dense.apply_channels(native_A)
    dense_data = dense.apply_data(A.data)
    sparse_data = sparse.apply_data(A.data)
    assert dense.vector_layout is not None
    assert sparse.vector_layout is not None
    assert isinstance(native_dense_out, AbelianSiteTensorData)
    assert isinstance(native_sparse_out, AbelianSiteTensorData)
    assert isinstance(native_renorm_out, AbelianSiteTensorData)
    assert isinstance(native_channels["full"], AbelianSiteTensorData)
    assert not isinstance(native_dense_out, BlockTensor)
    assert not isinstance(native_sparse_out, BlockTensor)
    assert not isinstance(native_renorm_out, BlockTensor)
    np.testing.assert_allclose(dense_out.data[("a", "x")], np.asarray([5.0, 6.0]))
    np.testing.assert_allclose(dense_out.data[("b", "x")], np.asarray([19.0]))
    np.testing.assert_allclose(dense_data[("a", "x")], dense_out.data[("a", "x")])
    np.testing.assert_allclose(dense_data[("b", "x")], dense_out.data[("b", "x")])
    np.testing.assert_allclose(sparse_data[("a", "x")], dense_out.data[("a", "x")])
    np.testing.assert_allclose(sparse_data[("b", "x")], dense_out.data[("b", "x")])
    np.testing.assert_allclose(sparse_out.data[("a", "x")], dense_out.data[("a", "x")])
    np.testing.assert_allclose(sparse_out.data[("b", "x")], dense_out.data[("b", "x")])
    np.testing.assert_allclose(native_dense_out.data[("a", "x")], dense_out.data[("a", "x")])
    np.testing.assert_allclose(native_dense_out.data[("b", "x")], dense_out.data[("b", "x")])
    np.testing.assert_allclose(native_sparse_out.data[("a", "x")], dense_out.data[("a", "x")])
    np.testing.assert_allclose(native_sparse_out.data[("b", "x")], dense_out.data[("b", "x")])
    np.testing.assert_allclose(native_channels["full"].data[("a", "x")], dense_out.data[("a", "x")])
    np.testing.assert_allclose(native_channels["full"].data[("b", "x")], dense_out.data[("b", "x")])
    np.testing.assert_allclose(native_renorm_out.data[("a", "x")], np.zeros(2))
    np.testing.assert_allclose(native_renorm_out.data[("b", "x")], np.zeros(1))


def test_abelian_direct_action_data_tables_do_not_wrap_blocktensor():
    layout = ((("a", "x"), (2,)), (("b", "x"), (1,)))
    qns = [["a", "b"], ["x"]]
    dirs = [-1, 1]
    data = {
        ("a", "x"): np.asarray([1.0, 2.0], dtype=np.complex128),
        ("b", "x"): np.asarray([3.0], dtype=np.complex128),
    }
    matrix = np.asarray(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=np.complex128,
    )
    dense = AbelianDenseBoundaryActionDataTable(
        matrix,
        layout,
        qns,
        dirs,
        source="unit_dense_data_table",
    )
    sparse = AbelianSparseBoundaryActionDataTable(
        rows=np.asarray([0, 0, 1, 2, 2], dtype=np.int64),
        cols=np.asarray([0, 2, 1, 0, 2], dtype=np.int64),
        values=np.asarray([2.0, 1.0, 3.0, 4.0, 5.0], dtype=np.complex128),
        dim=3,
        layout=layout,
        qns=qns,
        dirs=dirs,
        source="unit_sparse_data_table",
    )

    dense_data = dense.apply_data(data)
    sparse_data = sparse.apply_data(data)
    assert isinstance(dense_data, dict)
    assert isinstance(sparse_data, dict)
    assert not isinstance(dense_data, BlockTensor)
    assert not isinstance(sparse_data, BlockTensor)
    assert not hasattr(dense, "apply")
    assert not hasattr(sparse, "apply")
    np.testing.assert_allclose(dense_data[("a", "x")], np.asarray([5.0, 6.0]))
    np.testing.assert_allclose(dense_data[("b", "x")], np.asarray([19.0]))
    np.testing.assert_allclose(sparse_data[("a", "x")], dense_data[("a", "x")])
    np.testing.assert_allclose(sparse_data[("b", "x")], dense_data[("b", "x")])


def test_abelian_grouped_data_table_does_not_wrap_blocktensor():
    class FakeGroupedTable:
        def storage(self):
            return "fake_grouped"

        def last_refresh_kind(self):
            return "build"

        def block_matrix_elements(self):
            return 4

        def block_sparse_nnz(self):
            return 0

        def matvec(self, vector):
            return 2.0 * np.asarray(vector, dtype=np.complex128)

        def diagonal(self):
            return np.asarray([2.0, 2.0], dtype=np.complex128)

        def n_groups(self):
            return 1

        def n_group_channels(self):
            return 1

        def n_blocks(self):
            return 1

    q0 = "q0"
    layout = (((q0, q0, q0, q0), (1, 1, 2, 1)),)
    qns = [[q0], [q0], [q0], [q0]]
    dirs = [-1, 1, 1, 1]
    data = {
        (q0, q0, q0, q0): np.asarray([[[[1.0], [3.0]]]], dtype=np.complex128)
    }
    table = AbelianGroupedRenormalizedDataTable(
        FakeGroupedTable(),
        {"left": (np.ones((1, 1, 1)),), "family_names": ("fake",)},
        dim=2,
        layout=layout,
        qns=qns,
        dirs=dirs,
        source="unit_fake_grouped",
    )

    out = table.apply_data(data)
    assert isinstance(out, dict)
    assert not isinstance(out, BlockTensor)
    assert not hasattr(table, "apply")
    np.testing.assert_allclose(
        out[(q0, q0, q0, q0)],
        np.asarray([[[[2.0], [6.0]]]], dtype=np.complex128),
    )
    np.testing.assert_allclose(table.diagonal_flat(), np.asarray([2.0, 2.0]))
    assert table.stats["kind"] == "moving_environment_cpp_grouped_renormalized_table"
    assert table.stats["groups"] == 1


def test_abelian_compact_data_tables_are_flat_vector_only():
    block_table = AbelianCompactBlockDataTable(
        block_matrices=(np.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=np.complex128),),
        in_starts=np.asarray([0], dtype=np.int64),
        out_starts=np.asarray([0], dtype=np.int64),
        dim=2,
        layout=((("a",), (2,)),),
    )
    vec = np.asarray([5.0, 7.0], dtype=np.complex128)
    np.testing.assert_allclose(block_table.matvec(vec), np.asarray([17.0, 21.0]))
    np.testing.assert_allclose(block_table.diagonal_flat(), np.asarray([2.0, 3.0]))
    assert not hasattr(block_table, "apply")

    class FakeCompactPlan:
        def __init__(self):
            self.refreshed = False

        def matvec(self, vector):
            return 4.0 * np.asarray(vector, dtype=np.complex128)

        def davidson(self, diagonal, v0, tol, max_iter, restart_dim, accept_unconverged):
            return {
                "energy": complex(np.real(diagonal[0])),
                "vector": np.asarray(v0, dtype=np.complex128),
                "iterations": 1,
            }

        def diagonal_from_routes(self, routes):
            return np.asarray([4.0, 4.0], dtype=np.complex128)

        def update_stacks_from_blocks(self, *args):
            self.refreshed = True

    compact = AbelianCompactRenormalizedDataTable(
        FakeCompactPlan(),
        dim=2,
        layout=((("a",), (2,)),),
    )
    compact.install_diagonal_routes(np.asarray([[0, 0]], dtype=np.int64))
    np.testing.assert_allclose(compact.matvec(vec), np.asarray([20.0, 28.0]))
    np.testing.assert_allclose(compact.diagonal_flat(), np.asarray([4.0, 4.0]))
    result = compact.davidson(
        np.asarray([1.0, 2.0], dtype=np.complex128),
        vec,
        1.0e-8,
        4,
        2,
        False,
    )
    assert result["iterations"] == 1
    assert not hasattr(compact, "apply")


def test_abelian_moving_environment_flat_matvec_profiles_flat_vectors_only():
    class FakeBackend:
        def apply_renormalized_operator_table(self, table, vector):
            return table.matvec(vector)

    class FakeEnvironment:
        def __init__(self):
            self.compiled_backend = FakeBackend()
            self.moving_profile_stats = {}

    class FakeOperator:
        bond = 3

        def __init__(self):
            self.profile_stats = {}

    table = AbelianCompactBlockDataTable(
        block_matrices=(np.asarray([[2.0, 0.0], [0.0, 5.0]], dtype=np.complex128),),
        in_starts=np.asarray([0], dtype=np.int64),
        out_starts=np.asarray([0], dtype=np.int64),
        dim=2,
        layout=((("a",), (2,)),),
    )
    env = FakeEnvironment()
    op = FakeOperator()
    helper = AbelianMovingEnvironmentFlatMatvec(
        env,
        op,
        table,
        table.layout,
        (),
    )

    vec = np.asarray([3.0, 7.0], dtype=np.complex128)
    np.testing.assert_allclose(helper.matvec(vec), np.asarray([6.0, 35.0]))
    np.testing.assert_allclose(helper.diagonal(), np.asarray([2.0, 5.0]))
    helper.flush_profile()
    stats = op.profile_stats["packed_flat_complementary_family_action"]
    assert stats["calls"] == 1
    assert stats["compiled_direct_matvec_backend"] == "renormalized_table"
    assert stats["compiled_direct_matvec_entries"] == table.n_entries
    assert stats["last"]["bond"] == 3
    assert env.moving_profile_stats["compiled_flat_matvec_calls"] == 1
    assert not hasattr(helper, "apply")


def test_abelian_packed_local_state_proto_builds_layout_and_basis():
    left_block = np.arange(30, dtype=np.complex128).reshape(2, 3, 5)
    right_block = (1.0 + np.arange(72, dtype=np.complex128)).reshape(3, 4, 6)
    left = AbelianPackedBoundaryTensor(
        (("l0", "m0", "p0"),),
        (left_block,),
        dirs=[-1, 1, 1],
        qns=[["l0"], ["m0"], ["p0"]],
    )
    right = AbelianPackedBoundaryTensor(
        (("m0", "r0", "p1"),),
        (right_block,),
        dirs=[-1, 1, 1],
        qns=[["m0"], ["r0"], ["p1"]],
    )

    proto = AbelianPackedLocalStateProto.from_site_tensors(
        left,
        right,
        source="unit_local_proto",
    )

    assert proto.source == "unit_local_proto"
    assert proto.keys == (("l0", "r0", "p0", "p1"),)
    assert proto.dirs == [-1, 1, 1, 1]
    assert proto.qns == [["l0"], ["r0"], ["p0"], ["p1"]]
    assert proto.layout() == ((("l0", "r0", "p0", "p1"), (2, 4, 5, 6)),)
    expected = np.tensordot(left_block, right_block, axes=([1], [0])).transpose(
        0,
        2,
        1,
        3,
    )
    np.testing.assert_allclose(proto.blocks[0], expected)

    basis = proto.basis(("l0", "r0", "p0", "p1"), (2, 4, 5, 6), offset=17)
    assert basis.keys == proto.keys
    assert basis.dirs == proto.dirs
    assert basis.qns == proto.qns
    assert int(np.count_nonzero(basis.blocks[0])) == 1
    assert basis.blocks[0].reshape(-1)[17] == 1.0


def test_abelian_packed_boundary_sum_and_compose_helpers():
    left = AbelianPackedBoundaryTensor(
        (("a", "b"), ("a", "c")),
        (
            np.asarray([[1.0, 2.0]], dtype=np.complex128),
            np.asarray([[3.0, 4.0]], dtype=np.complex128),
        ),
        dirs=(-1, 1),
        qns=(("a",), ("b", "c")),
    )
    right = AbelianPackedBoundaryTensor(
        (("a", "b"), ("d", "e")),
        (
            np.asarray([[5.0, 6.0]], dtype=np.complex128),
            np.asarray([[7.0, 8.0]], dtype=np.complex128),
        ),
        dirs=(-1, 1),
        qns=(("a", "d"), ("b", "e")),
    )

    total = sum_abelian_packed_boundary_terms(
        ((left, 2.0), (right, -1.0)),
        scale_source="sum_scale",
        sum_source="sum_total",
    )
    assert is_abelian_packed_boundary_tensor(total)
    assert total.keys == (("a", "b"), ("a", "c"), ("d", "e"))
    np.testing.assert_allclose(total.blocks[0], 2.0 * left.blocks[0] - right.blocks[0])
    np.testing.assert_allclose(total.blocks[1], 2.0 * left.blocks[1])
    np.testing.assert_allclose(total.blocks[2], -right.blocks[1])

    first = AbelianPackedBoundaryTensor(
        ((1, 10, 100), (1, 10, 200)),
        (
            np.arange(6, dtype=np.complex128).reshape(1, 2, 3),
            np.ones((1, 2, 2), dtype=np.complex128),
        ),
        dirs=(-1, 1, -1),
        qns=((1,), (10,), (100, 200)),
    )
    second = AbelianPackedBoundaryTensor(
        ((4, 100, 20), (4, 200, 20)),
        (
            np.arange(15, dtype=np.complex128).reshape(1, 3, 5),
            (2.0 * np.ones((1, 2, 5), dtype=np.complex128)),
        ),
        dirs=(-1, 1, -1),
        qns=((4,), (100, 200), (20,)),
    )

    composed = compose_abelian_packed_boundary_operators(
        first,
        second,
        source="compose_test",
    )
    assert is_abelian_packed_boundary_tensor(composed)
    assert composed.keys == ((5, 10, 20),)
    expected = first.blocks[0][0] @ second.blocks[0][0]
    expected += first.blocks[1][0] @ second.blocks[1][0]
    np.testing.assert_allclose(composed.blocks[0], expected.reshape(1, 2, 5))


def test_same_side_p_boundary_value_table_preserves_ids_across_revisions():
    table = AbelianSameSidePBoundaryValueTable(side="left", bond=3, revision=0)
    key = (("A", "I", "B"), "I")
    value0 = AbelianPackedBoundaryTensor(
        (("q0", "q1"),),
        (np.asarray([[1.0, 2.0]], dtype=np.complex128),),
        dirs=(-1, 1),
    )

    values, missing, positions, hits, misses = table.resolve_many((key,))
    assert values == [None]
    assert missing == (key,)
    assert positions == (0,)
    assert hits == 0
    assert misses == 1

    assert table.put_many((key,), (value0,), normalized=True) == 1
    values, missing, positions, hits, misses = table.resolve_many(
        (key,),
        normalized=True,
    )
    assert values == [value0]
    assert missing == ()
    assert positions == ()
    assert hits == 1
    assert misses == 0
    assert table.ids[key] == 0
    assert table.payloads[0] is value0

    assert table.reset_for_revision(1) is True
    assert table.ids[key] == 0
    assert table.payloads[0] is None
    values, missing, positions, hits, misses = table.resolve_many(
        (key,),
        normalized=True,
    )
    assert values == [None]
    assert missing == (key,)
    assert positions == (0,)
    assert hits == 0
    assert misses == 1

    value1 = AbelianPackedBoundaryTensor(
        (("q0", "q1"),),
        (np.asarray([[3.0, 4.0]], dtype=np.complex128),),
        dirs=(-1, 1),
    )
    assert table.put(key, value1) is True
    assert table.ids[key] == 0
    assert table.payloads[0] is value1
    assert table.n_entries == 1
    assert table.stats["resets"] == 1


def test_packed_same_side_p_product_correction_recovers_exact_operator():
    product = AbelianPackedBoundaryTensor(
        ((2, 0, 1), (2, 0, 2)),
        (
            np.asarray([[[1.0, 2.0]]], dtype=np.complex128),
            np.asarray([[[3.0, 4.0]]], dtype=np.complex128),
        ),
        dirs=(-1, 1, -1),
        qns=((2,), (0,), (1, 2, 3)),
    )
    exact = AbelianPackedBoundaryTensor(
        ((2, 0, 1), (2, 0, 3)),
        (
            np.asarray([[[1.5, 1.0]]], dtype=np.complex128),
            np.asarray([[[5.0, 6.0]]], dtype=np.complex128),
        ),
        dirs=(-1, 1, -1),
        qns=((2,), (0,), (1, 2, 3)),
    )

    corrected, correction = packed_same_side_p_product_correction(product, exact)

    assert is_abelian_packed_boundary_tensor(corrected)
    assert is_abelian_packed_boundary_tensor(correction)
    assert corrected.keys == exact.keys
    for lhs, rhs in zip(corrected.blocks, exact.blocks):
        np.testing.assert_allclose(lhs, rhs)
    correction_data = correction.data
    np.testing.assert_allclose(
        correction_data[(2, 0, 1)],
        exact.data[(2, 0, 1)] - product.data[(2, 0, 1)],
    )
    np.testing.assert_allclose(correction_data[(2, 0, 2)], -product.data[(2, 0, 2)])
    np.testing.assert_allclose(correction_data[(2, 0, 3)], exact.data[(2, 0, 3)])


def test_abelian_packed_site_operator_builders_and_boundary_advance():
    entries = (
        (1, 0, 1, 2.0),
        (0, 1, -1, 3.0),
    )

    left_op = make_abelian_packed_site_operator_from_left(
        entries,
        phys_qns=(0, 1),
        left_qns=(0, 1),
        source="left_op",
    )
    right_op = make_abelian_packed_site_operator_from_right(
        entries,
        phys_qns=(0, 1),
        right_qns=(0,),
        source="right_op",
    )

    assert left_op.source == "left_op"
    assert left_op.dirs == [-1, 1, 1, -1]
    assert left_op.qns == [[0, 1], [-1, 0, 1, 2], [0, 1], [0, 1]]
    assert left_op.keys == (
        (0, -1, 1, 0),
        (0, 1, 0, 1),
        (1, 0, 1, 0),
        (1, 2, 0, 1),
    )
    np.testing.assert_allclose(left_op.blocks[0], np.asarray([[[[2.0]]]]))

    assert right_op.source == "right_op"
    assert right_op.qns == [[-1, 1], [0], [0, 1], [0, 1]]
    assert right_op.keys == ((1, 0, 1, 0), (-1, 0, 0, 1))
    np.testing.assert_allclose(right_op.blocks[1], np.asarray([[[[3.0]]]]))

    assert abelian_packed_tensor_axis_qns(left_op, 1) == (-1, 0, 1, 2)
    pair = make_abelian_packed_local_generator_pair(
        left_op,
        right_op,
        left_source="left_common",
        right_source="right_common",
    )
    assert pair is not None
    left_common, right_common, common = pair
    assert common == (-1, 1)
    assert left_common.source == "left_common"
    assert right_common.source == "right_common"
    assert left_common.keys == ((0, -1, 1, 0), (0, 1, 0, 1))
    assert right_common.keys == right_op.keys

    builder = AbelianSpatialLocalOperatorBuilder(
        site_qn_maps=({0: 0, 1: 1},),
        local_ops={"X": np.asarray([[0.0, 3.0], [2.0, 0.0]], dtype=complex)},
        source_prefix="builder",
    )
    built_left = builder.packed_site_operator_from_left("X", 0, (0, 1))
    built_right = builder.packed_site_operator_from_right("X", 0, (0,))
    assert built_left is builder.packed_site_operator_from_left("X", 0, (0, 1))
    assert builder.stats["packed_site_operator_cache"] == 2
    assert built_left.source == "builder_site_operator_left"
    assert set(built_left.keys) == set(left_op.keys)
    assert set(built_right.keys) == set(right_op.keys)

    E = make_abelian_packed_initial_left_environment(0)
    F = make_abelian_packed_initial_right_environment(0, 0)
    A = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.asarray([[[3.0 + 0.0j]]]),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="A",
    )
    W = AbelianPackedBoundaryTensor(
        ((0, 0, 0, 0),),
        (np.asarray([[[[5.0 + 0.0j]]]]),),
        dirs=[-1, 1, 1, -1],
        qns=[[0], [0], [0], [0]],
        source="W",
    )
    B = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.asarray([[[7.0 + 0.0j]]]),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
        source="B",
    )

    left_env = advance_abelian_packed_left_boundary(W, A, E, B)
    right_env = advance_abelian_packed_right_boundary(W, A, F, B)

    assert left_env.keys == ((0, 0, 0),)
    assert right_env.keys == ((0, 0, 0),)
    np.testing.assert_allclose(left_env.blocks[0], np.asarray([[[105.0 + 0.0j]]]))
    np.testing.assert_allclose(right_env.blocks[0], np.asarray([[[105.0 + 0.0j]]]))


def test_abelian_packed_identity_boundary_advance_matches_generic():
    def block(shape, start):
        data = np.arange(start, start + int(np.prod(shape)), dtype=float)
        return (data.reshape(shape) + 0.25j * data.reshape(shape)).astype(complex)

    A = AbelianPackedBoundaryTensor(
        ((0, 0, 0), (0, 1, 1), (1, 1, 0)),
        (block((2, 2, 1), 1), block((2, 3, 1), 11), block((4, 3, 1), 21)),
        dirs=[1, -1, 1],
        qns=[[0, 1], [0, 1], [0, 1]],
        source="A",
    )
    B = AbelianPackedBoundaryTensor(
        ((0, 2, 0), (1, 2, 0), (1, 3, 1)),
        (block((3, 2, 1), 41), block((2, 2, 1), 51), block((2, 4, 1), 61)),
        dirs=[1, -1, 1],
        qns=[[0, 1], [2, 3], [0, 1]],
        source="B",
    )
    E = AbelianPackedBoundaryTensor(
        ((0, 0, 0), (0, 1, 1), (1, 0, 1)),
        (block((1, 2, 3), 101), block((1, 4, 2), 121), block((1, 2, 2), 141)),
        dirs=[1, -1, 1],
        qns=[[0, 1], [0, 1], [0, 1]],
        source="E",
    )
    F = AbelianPackedBoundaryTensor(
        ((0, 0, 2), (0, 1, 3), (1, 1, 2)),
        (block((1, 2, 2), 201), block((1, 3, 4), 221), block((1, 3, 2), 251)),
        dirs=[-1, 1, -1],
        qns=[[0, 1], [0, 1], [2, 3]],
        source="F",
    )
    entries = ((0, 0, 0, 1.0), (1, 1, 0, 1.0))
    W_left = make_abelian_packed_site_operator_from_left(
        entries,
        phys_qns=(0, 1),
        left_qns=abelian_packed_tensor_axis_qns(E, 0),
        source="W_left_identity",
    )
    W_right = make_abelian_packed_site_operator_from_right(
        entries,
        phys_qns=(0, 1),
        right_qns=abelian_packed_tensor_axis_qns(F, 0),
        source="W_right_identity",
    )

    generic_left = advance_abelian_packed_left_boundary(W_left, A, E, B)
    identity_left = advance_abelian_packed_left_identity_boundary(A, E, B)
    same, diff, ref = compare_abelian_packed_boundary_tensors(
        identity_left,
        generic_left,
    )
    assert same
    assert diff <= 1.0e-12 * max(ref, 1.0)
    assert identity_left.dirs == generic_left.dirs

    generic_right = advance_abelian_packed_right_boundary(W_right, A, F, B)
    identity_right = advance_abelian_packed_right_identity_boundary(A, F, B)
    same, diff, ref = compare_abelian_packed_boundary_tensors(
        identity_right,
        generic_right,
    )
    assert same
    assert diff <= 1.0e-12 * max(ref, 1.0)
    assert identity_right.dirs == generic_right.dirs


def test_abelian_packed_local_action_entries_identity_and_generator():
    E = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (2.0 * np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
    )
    F = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (3.0 * np.ones((1, 1, 1), dtype=np.complex128),),
        dirs=[-1, 1, -1],
        qns=[[0], [0], [0]],
    )
    W_left = AbelianPackedBoundaryTensor(
        ((0, 0, 0, 0),),
        (11.0 * np.ones((1, 1, 1, 1), dtype=np.complex128),),
        dirs=[-1, 1, 1, -1],
        qns=[[0], [0], [0], [0]],
    )
    W_right = AbelianPackedBoundaryTensor(
        ((0, 0, 0, 0),),
        (13.0 * np.ones((1, 1, 1, 1), dtype=np.complex128),),
        dirs=[-1, 1, 1, -1],
        qns=[[0], [0], [0], [0]],
    )
    basis = AbelianPackedBoundaryTensor(
        ((0, 0, 0, 0),),
        (5.0 * np.ones((1, 1, 1, 1), dtype=np.complex128),),
        dirs=[-1, 1, 1, 1],
        qns=[[0], [0], [0], [0]],
    )

    identity = AbelianPackedIdentityLocalEntry(7.0, E, F)
    local = AbelianPackedLocalGeneratorEntry(17.0, E, W_left, W_right, F)
    out = apply_abelian_packed_local_action_entries((identity, local), basis)

    expected = 7.0 * 2.0 * 5.0 * 3.0
    expected += 17.0 * 2.0 * 5.0 * 11.0 * 13.0 * 3.0
    assert out.keys == ((0, 0, 0, 0),)
    np.testing.assert_allclose(out.blocks[0], np.asarray([[[[expected + 0.0j]]]]))

    tuple_out = apply_abelian_packed_local_action_entries(
        ((E, (scale_abelian_boundary_tensor(W_left, 17.0), W_right), F),),
        basis,
    )
    local_only = apply_abelian_packed_local_action_entries((local,), basis)
    same_layout, diff, ref_norm = compare_abelian_packed_boundary_tensors(
        tuple_out,
        local_only,
    )
    assert same_layout
    assert diff <= 1.0e-12 * max(ref_norm, 1.0)


def test_cpp_identity_channel_stacks_match_generic_direct_stacks():
    from pyqed.mps import cpp_davidson

    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "identity_channel_left_stack", None) is None
        or getattr(cpp_davidson, "identity_channel_right_stack", None) is None
    ):
        pytest.skip("C++ direct stack helpers are unavailable")

    rng = np.random.default_rng(1234)
    E = (
        rng.normal(size=(2, 3, 4))
        + 1j * rng.normal(size=(2, 3, 4))
    ).astype(np.complex128)
    W_left = np.zeros((2, 2, 3, 3), dtype=np.complex128)
    for a in range(W_left.shape[0]):
        for b in range(W_left.shape[1]):
            for x in range(W_left.shape[2]):
                W_left[a, b, x, x] = (1 + a + 2 * b + x) * (1.0 - 0.25j)

    F = (
        rng.normal(size=(3, 5, 4))
        + 1j * rng.normal(size=(3, 5, 4))
    ).astype(np.complex128)
    W_right = np.zeros((2, 3, 4, 4), dtype=np.complex128)
    for b in range(W_right.shape[0]):
        for c in range(W_right.shape[1]):
            for y in range(W_right.shape[2]):
                W_right[b, c, y, y] = (2 + b + c + y) * (0.5 + 0.125j)

    np.testing.assert_allclose(
        cpp_davidson.identity_channel_left_stack(E, W_left),
        cpp_davidson.direct_left_stack(E, W_left),
    )
    np.testing.assert_allclose(
        cpp_davidson.identity_channel_right_stack(W_right, F),
        cpp_davidson.direct_right_stack(W_right, F),
    )


def test_cpp_packed_identity_payload_matches_local_identity_generator():
    from pyqed.mps import cpp_davidson

    if (
        not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        or getattr(cpp_davidson, "build_direct_family_payload_fastkeys", None) is None
        or getattr(cpp_davidson, "GroupedRenormalizedTable", None) is None
    ):
        pytest.skip("C++ direct payload backend is unavailable")

    E = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        (np.arange(1, 5, dtype=float).reshape(1, 2, 2),),
        dirs=[1, -1, 1],
        qns=[[0], [0], [0]],
    )
    F = AbelianPackedBoundaryTensor(
        ((0, 0, 0),),
        ((np.arange(1, 10, dtype=float).reshape(1, 3, 3) + 0.5j),),
        dirs=[-1, 1, -1],
        qns=[[0], [0], [0]],
    )
    W_left = AbelianPackedBoundaryTensor(
        ((0, 0, 0, 0),),
        (np.eye(2, dtype=np.complex128).reshape(1, 1, 2, 2),),
        dirs=[-1, 1, 1, -1],
        qns=[[0], [0], [0], [0]],
    )
    W_right = AbelianPackedBoundaryTensor(
        ((0, 0, 0, 0),),
        (np.eye(2, dtype=np.complex128).reshape(1, 1, 2, 2),),
        dirs=[-1, 1, 1, -1],
        qns=[[0], [0], [0], [0]],
    )
    identity = AbelianPackedDirectFamilyEntries()
    identity.append_identity(1.25, E, F, source="packed_identity")
    local = AbelianPackedDirectFamilyEntries()
    local.append_local_generator(
        1.25,
        E,
        W_left,
        W_right,
        F,
        source="identity_local_csr",
    )

    layout = (((0, 0, 0, 0), (2, 3, 2, 2)),)
    identity_payload = cpp_davidson.build_direct_family_payload_fastkeys(
        {"P": identity},
        {},
        layout,
        True,
    )
    local_payload = cpp_davidson.build_direct_family_payload_fastkeys(
        {"P": local},
        {},
        layout,
        True,
    )
    identity_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        identity_payload,
        24,
        0.0,
    )
    local_table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        local_payload,
        24,
        0.0,
    )
    rng = np.random.default_rng(123)
    vec = (
        rng.normal(size=24) + 1j * rng.normal(size=24)
    ).astype(np.complex128)
    np.testing.assert_allclose(
        identity_table.matvec(vec),
        local_table.matvec(vec),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_abelian_packed_local_action_validation_helpers():
    def packed3(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0),),
            (np.asarray([[[value]]], dtype=np.complex128),),
            dirs=[1, -1, 1],
            qns=[[0], [0], [0]],
        )

    def packed4(value):
        return AbelianPackedBoundaryTensor(
            ((0, 0, 0, 0),),
            (np.asarray([[[[value]]]], dtype=np.complex128),),
            dirs=[-1, 1, 1, -1],
            qns=[[0], [0], [0], [0]],
        )

    proto = AbelianPackedLocalStateProto(
        AbelianPackedBoundaryTensor(
            ((0, 0, 0, 0),),
            (np.ones((1, 1, 1, 1), dtype=np.complex128),),
            dirs=[-1, 1, 1, 1],
            qns=[[0], [0], [0], [0]],
        )
    )
    E = packed3(2.0)
    F = packed3(3.0)
    W_left = packed4(5.0)
    W_right = packed4(7.0)
    candidate = (AbelianPackedLocalGeneratorEntry(11.0, E, W_left, W_right, F),)
    same = (
        (
            E,
            (scale_abelian_boundary_tensor(W_left, 11.0), W_right),
            F,
        ),
    )
    different = (AbelianPackedLocalGeneratorEntry(13.0, E, W_left, W_right, F),)

    assert abelian_packed_local_action_apply_clean(proto, candidate) is True
    assert abelian_packed_local_action_probe_reference(
        proto,
        candidate,
        same,
        max_vectors=1,
    ) is True
    assert abelian_packed_local_action_matches_reference(proto, candidate, same) is True
    assert (
        abelian_packed_local_action_matches_reference(proto, candidate, different)
        is False
    )


def test_contextual_boundary_keys_deduplicates_and_orders_fragments():
    records = (
        (("B", "C"), "X", "Y", ("R2", "R1"), 1.0),
        (("A",), "X", "Y", ("R1",), 2.0),
        (("B", "C"), "X", "Y", ("R2", "R1"), 3.0),
    )

    left_keys, right_keys = contextual_boundary_keys(records)

    assert left_keys == (
        (("A",), "X"),
        (("B", "C"), "X"),
    )
    assert right_keys == (
        (("R1",), "Y"),
        (("R2", "R1"), "Y"),
    )


def test_native_p_owner_records_classifies_two_generator_pairs():
    support = {
        (0, 1): {0},
        (2, 3): {2, 3},
        (4, 5): {5},
        (1, 4): {1, 4},
    }
    calls = []

    records = native_p_owner_records(
        {
            (0, 1, 2, 3): 1.0,
            (0, 1, 4, 5): 1.0,
            (4, 5, 1, 4): 1.0,
        },
        lambda p, q: calls.append((int(p), int(q))) or support[(int(p), int(q))],
        bond=2,
        nsites=6,
    )

    assert records == (
        ((0, 1, 2, 3), "left", "local"),
        ((0, 1, 4, 5), "left", "right"),
        ((4, 5, 1, 4), "right", None),
    )
    assert calls.count((0, 1)) == 1
    assert calls.count((4, 5)) == 1
    assert abelian_generator_owner_from_support({2, 3}, bond=2, nsites=6) == "local"
    assert (
        abelian_generator_region_from_support({1, 2, 5}, bond=2, nsites=6)
        == "left_local_right"
    )


def test_fresh_casci_like_preserves_spatial_block2_table_settings():
    class DummyDMRG:
        def __init__(self, mf, **kwargs):
            self.mf = mf
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.spin_purification = False
            self.ss = None
            self.shift = None

    source = DummyDMRG(
        object(),
        ncas=4,
        nelecas=4,
        D=10,
        init_guess="hf",
        m_warmup=8,
        tol=1.0e-7,
        low_rank_mpo=True,
        low_rank_mpo_bond=12,
        low_rank_mpo_batch_size=3,
        site="spatial",
        spatial_reduced_mpo=True,
        symmetry=("u1",),
        spatial_site_basis="canonical",
        integral_backend="dense",
        spatial_abelian_mpo="direct",
        spatial_abelian_symbolic_algo="optimal_bipartite",
        spatial_family_environment_backend="block2_table",
        spatial_native_p_grouping="first_two_site_order",
        spatial_block2_table_p_split_metric="span",
        spatial_block2_table_p_split_groups=3,
        spatial_block2_table_native_p=True,
        spatial_complementary_payload_tensor_matvec=False,
        spatial_precontracted_family_environment=True,
        spatial_boundary_table_max_dim=96,
        spatial_exact_component_compression_policy="structural",
        spatial_exact_component_compression_validate=False,
        spatial_exact_component_compression_validation_vectors=5,
        spatial_exact_component_compression_min_reduction=3,
        spatial_exact_component_compression_max_group_size=11,
        spatial_enable_cpp_boundary_p=False,
        spatial_validate_cpp_boundary_p=False,
        spatial_cpp_boundary_p_validation_policy="always",
        spatial_direct_operator_batch_min_entries=5,
        dmrg_performance="packed-compiled-fast",
        abelian_matvec_options={"packed_local_davidson_restart_dim": 12},
        debug_complementary_action_check=True,
        debug_complementary_action_check_tol=1.0e-9,
        debug_complementary_action_check_limit=7,
        debug_spatial_family_hamiltonian_check=True,
        orb_sym=(0, 1, 0, 1),
        verbose=2,
    )

    fresh = _fresh_casci_like(source)

    assert fresh.spatial_abelian_mpo == "direct"
    assert fresh.spatial_family_environment_backend == "block2_table"
    assert fresh.spatial_native_p_grouping == "first_two_site_order"
    assert fresh.spatial_block2_table_p_split_metric == "span"
    assert fresh.spatial_block2_table_p_split_groups == 3
    assert fresh.spatial_block2_table_native_p is True
    assert fresh.spatial_complementary_payload_tensor_matvec is False
    assert fresh.spatial_precontracted_family_environment is True
    assert fresh.spatial_boundary_table_max_dim == 96
    assert fresh.spatial_exact_component_compression_policy == "structural"
    assert fresh.spatial_exact_component_compression_validate is False
    assert fresh.spatial_exact_component_compression_validation_vectors == 5
    assert fresh.spatial_exact_component_compression_min_reduction == 3
    assert fresh.spatial_exact_component_compression_max_group_size == 11
    assert fresh.spatial_enable_cpp_boundary_p is False
    assert fresh.spatial_validate_cpp_boundary_p is False
    assert fresh.spatial_cpp_boundary_p_validation_policy == "always"
    assert fresh.spatial_direct_operator_batch_min_entries == 5
    assert fresh.dmrg_performance == "packed-compiled-fast"
    assert fresh.abelian_matvec_options == {"packed_local_davidson_restart_dim": 12}
    assert fresh.debug_complementary_action_check is True
    assert fresh.debug_complementary_action_check_tol == pytest.approx(1.0e-9)
    assert fresh.debug_complementary_action_check_limit == 7
    assert fresh.debug_spatial_family_hamiltonian_check is True
    assert fresh.integral_backend == "dense"
    assert fresh.orb_sym == (0, 1, 0, 1)


def test_fresh_casci_like_defaults_to_block2_like_dmrg_performance():
    class DummyDMRG:
        def __init__(self, mf, **kwargs):
            self.mf = mf
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.spin_purification = False
            self.ss = None
            self.shift = None

    source = DummyDMRG(
        object(),
        ncas=4,
        nelecas=4,
        D=10,
    )

    fresh = _fresh_casci_like(source)

    assert fresh.dmrg_performance == "block2-like"


def test_spatial_complementary_operator_families_group_integrals():
    h1 = np.zeros((3, 3))
    h1[0, 2] = 0.05
    eri = np.zeros((2, 2, 3, 3, 3, 3))
    eri[:, :, 0, 1, 1, 2] = 0.4
    eri[:, :, 2, 0, 1, 1] = -0.2

    families = build_spatial_complementary_operator_families(h1, eri, cutoff=1.0e-12)

    assert families.names == ("S", "R", "A", "P", "B", "Q")
    assert families["P"].entries[(0, 1, 1, 2)] == pytest.approx(0.2)
    assert families["Q"].entries[(0, 2, 1)] == pytest.approx(-0.2)
    assert families["R"].entries[(0, 2)] == pytest.approx(-0.15)
    assert (0, 1) in families["A"].entries
    assert (1, 2) in families["A"].entries
    assert families.as_metadata()["families"]["P"]["n_terms"] == 2
    assert families.as_metadata()["enable_cpp_boundary_p"] is True
    assert families.as_metadata()["validate_cpp_boundary_p"] is False
    assert (
        families.as_metadata()["cpp_boundary_p_validation_policy"]
        == "off"
    )
    assert families.as_metadata()["direct_operator_batch_min_entries"] == 2


def test_spatial_complementary_local_matrix_matches_two_site_hamiltonian():
    h1 = np.array(
        [
            [0.2, -0.03],
            [-0.03, -0.1],
        ]
    )
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.7
    eri[1, 1, 1, 1] = 0.5
    eri[0, 0, 1, 1] = 0.2
    eri[1, 1, 0, 0] = 0.2
    h2 = np.stack((np.stack((eri, eri.copy())), np.stack((eri.copy(), eri.copy()))))

    families = build_spatial_complementary_operator_families(h1, h2, cutoff=1.0e-12)
    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    dense_local = spatial_complementary_local_matrix(families, 0)
    channel_mats = spatial_complementary_local_matrices(families, 0)

    np.testing.assert_allclose(dense_local, dense_ref, atol=1.0e-12)
    np.testing.assert_allclose(
        sum(channel_mats.values()),
        dense_local,
        atol=1.0e-12,
    )
    assert set(channel_mats) == {"R", "P"}
    assert np.linalg.norm(channel_mats["R"]) > 0.0
    assert np.linalg.norm(channel_mats["P"]) > 0.0


def test_spatial_one_body_term_map_matches_dense_reference():
    h1 = np.array(
        [
            [0.2, -0.03, 0.04],
            [-0.03, -0.1, 0.07],
            [0.04, 0.07, 0.5],
        ]
    )
    h2 = np.zeros((2, 2, 3, 3, 3, 3))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    dense_terms = _dense_from_spatial_term_map(spatial_one_body_term_map(h1), 3)

    np.testing.assert_allclose(dense_terms, dense_ref, atol=1.0e-12)


def test_spatial_two_body_term_map_matches_dense_reference():
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.7
    eri[1, 1, 1, 1] = 0.5
    eri[0, 0, 1, 1] = 0.2
    eri[1, 1, 0, 0] = 0.2
    h1 = np.zeros((2, 2))
    h2 = np.stack((np.stack((eri, eri.copy())), np.stack((eri.copy(), eri.copy()))))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    dense_terms = _dense_from_spatial_term_map(spatial_two_body_term_map(eri), 2)

    np.testing.assert_allclose(dense_terms, dense_ref, atol=1.0e-12)


def test_spatial_two_body_spinfree_term_map_matches_component_reference():
    rng = np.random.default_rng(7)
    eri = rng.normal(size=(3, 3, 3, 3))

    dense_component = _dense_from_spatial_term_map(spatial_two_body_term_map(eri), 3)
    dense_spinfree = _dense_from_spatial_term_map(
        spatial_two_body_spinfree_term_map(eri),
        3,
    )

    np.testing.assert_allclose(dense_spinfree, dense_component, atol=1.0e-12)


def test_spatial_complementary_family_term_maps_reconstruct_hamiltonian_terms():
    rng = np.random.default_rng(19)
    h1 = rng.normal(size=(3, 3))
    h1 = 0.5 * (h1 + h1.T)
    eri = rng.normal(size=(3, 3, 3, 3))
    h2 = np.stack((np.stack((eri, eri.copy())), np.stack((eri.copy(), eri.copy()))))
    families = build_spatial_complementary_operator_families(
        h1,
        h2,
        cutoff=1.0e-12,
        include_half=True,
    )

    family_terms = spatial_complementary_family_hamiltonian_term_map(families)
    reference_terms = merge_term_maps(
        spatial_one_body_term_map(h1),
        spatial_two_body_spinfree_term_map(eri),
    )

    assert len(family_terms) == len(reference_terms)
    np.testing.assert_allclose(
        _dense_from_spatial_term_map(family_terms, 3),
        _dense_from_spatial_term_map(reference_terms, 3),
        atol=1.0e-12,
    )
    family_counts = {
        name: len(term_map)
        for name, term_map in spatial_complementary_family_term_maps(families).items()
    }
    assert set(family_counts) == {"R", "P"}
    assert family_counts["R"] > 0
    assert family_counts["P"] > 0


def test_native_spatial_generator_family_mpos_match_term_maps():
    h1 = np.array(
        [
            [0.2, -0.03],
            [0.04, -0.1],
        ]
    )
    p_entries = {
        (0, 0, 1, 1): 0.35,
        (0, 1, 1, 0): -0.2,
    }
    leg = physical_leg_from_spatial_orbital()

    r_builder = AutoMPO([leg, leg])
    add_spatial_one_body_terms(r_builder, h1, cutoff=1.0e-12, family="R")
    r_dense = _dense_matrix_from_local_mpo(r_builder.build())
    np.testing.assert_allclose(
        r_dense,
        _dense_from_spatial_term_map(spatial_one_body_term_map(h1), 2),
        atol=1.0e-12,
    )

    p_builder = AutoMPO([leg, leg])
    add_spatial_two_generator_product_terms(
        p_builder,
        p_entries,
        cutoff=1.0e-12,
        family="P",
    )
    p_dense = _dense_matrix_from_local_mpo(p_builder.build())
    np.testing.assert_allclose(
        p_dense,
        _dense_from_spatial_term_map(spatial_two_generator_family_term_map(p_entries), 2),
        atol=1.0e-12,
    )


def test_merge_term_maps_cancels_near_zero_terms():
    first = {}
    second = {}
    accumulate_symbolic_term(first, "n", [0], 0.5)
    accumulate_symbolic_term(second, "n", [0], -0.5)

    assert merge_term_maps(first, second) == {}


def test_fully_reduced_spatial_reduced_hamiltonian_builds_one_body_only():
    h1 = np.diag([0.3, -0.2])

    result = build_spatial_reduced_hamiltonian_mpo(
        h1,
        eri=None,
        fully_reduced=True,
        nelec=2,
        spin=0,
        ecore=-1.2,
        orb_sym=(1, 1),
    )

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body"] is False
    assert len(result.factors) == 2
    assert result.mpo is result.factors
    assert result.ncas == 2
    assert result.ecore == pytest.approx(-1.2)
    assert result.initialize_system_kwargs() == {
        "n_sites": 2,
        "n_elec": 2,
        "spin": 0,
        "orb_sym": (1, 1),
    }
    assert result.info["block_hamiltonian"] is True


def test_fully_reduced_spatial_reduced_hamiltonian_builds_four_distinct_eri_strings():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 1, 2, 3] = 0.1

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body"] is True
    assert result.info["two_body_reduced_string_terms"] > 0


def test_fully_reduced_spatial_reduced_hamiltonian_builds_diagonal_density_eri_terms():
    h1 = np.zeros((3, 3))
    eri = np.zeros((2, 2, 3, 3, 3, 3))
    eri[:, :, 0, 0, 1, 1] = 0.2
    eri[:, :, 2, 2, 2, 2] = 0.3

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_density_terms"] == 2
    assert result.info["two_body_representation"] == "fully_reduced_density_eri"


def test_fully_reduced_spatial_reduced_hamiltonian_builds_density_bilinear_eri_terms():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 0, 1, 2] = 0.2
    eri[:, :, 1, 2, 3, 3] = -0.1

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_density_bilinear_terms"] == 2
    assert result.info["two_body_representation"] == "fully_reduced_density_bilinear_eri"


def test_fully_reduced_spatial_reduced_hamiltonian_builds_endpoint_density_bilinear_terms():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 0, 1, 0] = 0.1
    eri[:, :, 1, 0, 1, 1] = -0.2

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_density_bilinear_terms"] == 2
    assert result.info["two_body_representation"] == "fully_reduced_density_bilinear_eri"


def test_fully_reduced_spatial_reduced_hamiltonian_builds_pair_eri_terms():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 1, 0, 2] = 0.1
    eri[:, :, 0, 1, 2, 1] = -0.2
    eri[:, :, 2, 3, 2, 3] = 0.3

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_pair_terms"] == 3
    assert result.info["two_body_representation"] == "fully_reduced_pair_eri"


def test_fully_reduced_pair_eri_keeps_rank_coupled_chain_with_dense_prefix():
    leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())
    eri = np.zeros((3, 3, 3, 3))
    eri[0, 1, 0, 2] = 0.1
    autompo = AutoMPO([leg] * 3)

    SpatialSpinFreeERIBuilder([leg] * 3, eri).add_to(autompo)
    factors = autompo.build()

    assert factors
    assert all(isinstance(factor, RankCoupledMPO) for factor in factors)


def test_fully_reduced_spatial_reduced_hamiltonian_builds_exchange_eri_terms():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 1, 2, 0] = 0.1

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_exchange_terms"] > 0
    assert result.info["two_body_representation"] == "fully_reduced_exchange_eri"


def test_fully_reduced_spatial_reduced_hamiltonian_builds_exchange_with_one_body_correction():
    h1 = np.zeros((4, 4))
    eri = np.zeros((2, 2, 4, 4, 4, 4))
    eri[:, :, 0, 1, 1, 2] = 0.1

    result = build_spatial_reduced_hamiltonian_mpo(h1, eri=eri, fully_reduced=True)

    assert result.info["spatial_site_basis"] == "fully_reduced_su2"
    assert result.info["two_body_fully_reduced_exchange_terms"] > 0
    assert result.info["two_body_one_body_correction_terms"] == 1
