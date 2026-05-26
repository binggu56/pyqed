from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem.dmrg.dmrg import (
    DMRG,
    _build_spatial_active_hamiltonian_matrix,
    _normalize_spatial_family_environment_backend,
    _normalize_spatial_native_p_grouping,
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
from pyqed.mps.nonabelian import (
    AutoMPO,
    FullyReducedSpatialOrbitalSite,
    RankCoupledMPO,
    SpatialSpinFreeERIBuilder,
    physical_leg_from_spatial_orbital,
)
from pyqed.mps.nonabelian.models import (
    _dense_matrix_from_local_mpo,
    add_spatial_one_body_terms,
    add_spatial_two_generator_product_terms,
)
from pyqed.mps.nonabelian.renormalized import (
    ComplementaryFamilyRenormalizedOperatorTable,
    ComplementaryNativeExactPatternComponentTable,
    ComplementaryNativeExactPatternOperatorTable,
    ComplementaryNativePairBoundaryOperatorTable,
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


def test_qchem_spatial_abelian_defaults_to_spatial_block2_table_payload():
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
    assert dmrg.spatial_enable_native_boundary_p is True
    assert dmrg.spatial_validate_native_boundary_p is True
    assert dmrg.spatial_native_boundary_p_validation_policy == "first_pass"
    assert dmrg.spatial_direct_operator_batch_min_entries == 2
    assert dmrg.spatial_reduced_mpo is False
    assert dmrg._can_use_spatial_block2_carrier() is True

    dense = DMRG(MF(), ncas=2, nelecas=2, D=4, site="spatial", symmetry=None)
    assert dense.spatial_abelian_mpo == "spatial"
    assert dense._can_use_spatial_block2_carrier() is False


def test_spatial_block2_carrier_is_d4_scaffold_not_grouped_spin_orbital():
    carrier = build_spatial_block2_carrier_mpo(3)

    assert carrier.info["representation"] == "spatial_block2_table_carrier_mpo"
    assert carrier.info["replaces_grouped_spin_orbital_carrier"] is True
    assert [factor.shape for factor in carrier.factors] == [(1, 1, 4, 4)] * 3
    for factor in carrier.factors:
        np.testing.assert_allclose(factor[0, 0], np.eye(4))


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


def test_native_exact_pattern_table_is_exposed_in_family_stats():
    table = ComplementaryNativeExactPatternOperatorTable(side="left", bond=1)
    block_like = SimpleNamespace(data={("q0",): np.ones((2, 3))})
    table.put((("I",), "C"), (block_like,), family_name="P")
    component_table = ComplementaryNativeExactPatternComponentTable(bond=1)
    component_table.put_family_records(
        "P",
        (((("I",), "C", "D", ("I",), 0.5 + 0.0j)),),
    )
    component_table.put_family("P", ((block_like,),))
    pair_table = ComplementaryNativePairBoundaryOperatorTable(side="center", bond=1)
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
        nested["complementary_native_exact_pattern_operator_table"]["family_counts"]
        == {"P": 1}
    )
    assert (
        nested["complementary_native_exact_pattern_component_table"]["family_counts"]
        == {"P": 1}
    )
    assert (
        nested["complementary_native_exact_pattern_component_table"]["record_counts"]
        == {"P": 1}
    )
    assert nested["complementary_native_pair_boundary_operator_table"]["n_terms"] == 1


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
        spatial_enable_native_boundary_p=False,
        spatial_validate_native_boundary_p=False,
        spatial_native_boundary_p_validation_policy="always",
        spatial_direct_operator_batch_min_entries=5,
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
    assert fresh.spatial_enable_native_boundary_p is False
    assert fresh.spatial_validate_native_boundary_p is False
    assert fresh.spatial_native_boundary_p_validation_policy == "always"
    assert fresh.spatial_direct_operator_batch_min_entries == 5
    assert fresh.debug_complementary_action_check is True
    assert fresh.debug_complementary_action_check_tol == pytest.approx(1.0e-9)
    assert fresh.debug_complementary_action_check_limit == 7
    assert fresh.debug_spatial_family_hamiltonian_check is True
    assert fresh.integral_backend == "dense"
    assert fresh.orb_sym == (0, 1, 0, 1)


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
    assert families.as_metadata()["enable_native_boundary_p"] is True
    assert families.as_metadata()["validate_native_boundary_p"] is True
    assert (
        families.as_metadata()["native_boundary_p_validation_policy"]
        == "first_pass"
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
        n_elec=2,
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
