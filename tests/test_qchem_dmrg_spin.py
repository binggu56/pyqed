import numpy as np
import pytest

import pyqed.mps.abelian_direct as abelian_direct
import pyqed.mps.cpp_davidson as cpp_davidson
from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg.dmrg import (
    DMRG,
    _build_spatial_active_hamiltonian_matrix,
    _build_spatial_fermion_operators,
    _build_spin_orbital_dense_hamiltonian_tensor_mpo,
    _group_spin_orbital_mpo_pairs,
    _build_spatial_hamiltonian_tensor_mpo,
    _build_spatial_s2_matrix,
    _build_s2_term_map,
    _build_spin_purification_term_map,
)
from pyqed.qchem.dmrg.backends import (
    build_spatial_complementary_operator_families,
    build_spatial_reduced_hamiltonian_mpo,
)
from pyqed.qchem.dmrg.backends.nonabelian import (
    ORTHONORMALIZED_OPERATOR_ITERMAX_DEFAULT,
)
from pyqed.qchem.dmrg.spatial_terms import (
    spatial_local_ops,
    spatial_two_body_spinfree_term_map,
)
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.mps.nonabelian import RankCoupledMPO, SpatialSpinFreeERIBuilder
from pyqed.mps.nonabelian import build_random_spatial_mps, build_spatial_hubbard_mpo
from pyqed.mps.nonabelian.environment import BlockSparseEnvironmentChain, contract_chain_expectation
from pyqed.mps.nonabelian.renormalized import ComplementaryFamilyTensorTable
from pyqed.mps.nonabelian.renormalized import FamilyNativeFactorKernel
from pyqed.mps.nonabelian.renormalized import RenormalizedBlockStack
from pyqed.mps.nonabelian.renormalized import configure_su2_kernel_policy
from pyqed.mps.nonabelian.renormalized import symbolic_mpo_core_transitions
from pyqed.mps.nonabelian.su2_kernel import SU2LocalAction
from pyqed.mps.nonabelian.su2_kernel import cython_available as su2_cython_available
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.mps.nonabelian.contraction import merge_mps_sites
from pyqed.mps.nonabelian.solver import (
    _METRIC_BLOCK_TRANSFORM_CACHE,
    _METRIC_BLOCK_TRANSFORM_CACHE_STATS,
    _metric_block_transform,
)
from pyqed.mps.nonabelian.solver import solve_local_two_site
from pyqed.mps.su2 import SU2Irrep, SpinChargeSector
from pyqed.mps.symmetry import AbelianSector


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def test_su2_block2_like_uses_short_local_solver_default():
    assert ORTHONORMALIZED_OPERATOR_ITERMAX_DEFAULT == 30


def _spin_orbital_annihilation(site, nsites):
    ident = np.eye(2)
    z = np.diag([1.0, -1.0])
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    ops = [z] * site + [a] + [ident] * (nsites - site - 1)
    return _kron_all(ops)


def _spin_orbital_creation(site, nsites):
    return _spin_orbital_annihilation(site, nsites).T


def _dense_from_symbolic_term_map(term_map, nsites):
    ident = np.eye(2)
    loc = {
        "a": np.array([[0.0, 1.0], [0.0, 0.0]]),
        r"a^\dagger": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "n": np.diag([0.0, 1.0]),
        "sigma_z": np.diag([1.0, -1.0]),
    }
    out = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for (symbol, dofs), factor in term_map.items():
        pieces = symbol.split()
        ops = [ident.copy() for _ in range(nsites)]
        for site, piece in zip(dofs, pieces):
            ops[site] = loc[piece]
        out += factor * _kron_all(ops)
    return out


def _dense_matrix_from_nonabelian_mpo_list(mpo):
    states = {0: np.array([[1.0]], dtype=complex)}
    for core in mpo:
        dense_core = core.as_dense() if hasattr(core, "as_dense") else np.asarray(core)
        new_states = {}
        for left_index, accum in states.items():
            for right_index in range(dense_core.shape[1]):
                local = dense_core[left_index, right_index]
                if not np.any(local):
                    continue
                contrib = np.kron(accum, local)
                if right_index in new_states:
                    new_states[right_index] += contrib
                else:
                    new_states[right_index] = contrib
        states = new_states
    return states[0]


def test_complementary_family_tensor_table_matches_raw_component_plan():
    class _Entry:
        def __init__(self, shape):
            self.shape = tuple(shape)

    class _Term:
        def __init__(self, kernel, family_names):
            self.kernel = np.asarray(kernel, dtype=complex)
            self.family_names = tuple(family_names)
            self.input_entry = _Entry((self.kernel.shape[1],))

        def apply_block(self, block):
            return self.kernel @ np.asarray(block, dtype=complex).reshape(-1)

    class _ComponentBasis:
        component_indices = (np.array([0, 1]), np.array([2, 3]))
        component_transforms = (np.eye(2, dtype=complex), np.eye(2, dtype=complex))
        orth_offsets = (0, 2)
        orthonormal_dim = 4

    basis = _ComponentBasis()
    terms = (
        (0, 1, slice(0, 2), slice(0, 2), _Term([[1.0, 0.5], [0.0, -0.25]], ("R",))),
        (1, 0, slice(0, 2), slice(0, 2), _Term([[0.2, -0.1], [0.3, 0.7]], ("P",))),
        (0, 0, slice(0, 2), slice(0, 2), _Term([[0.4, 0.0], [0.0, 0.6]], ("Q",))),
    )
    table = ComplementaryFamilyTensorTable.from_component_direct_plan(terms)

    vector = np.array([0.1, -0.3, 0.7, 0.2], dtype=complex)
    parent_inputs = [vector[:2], vector[2:]]
    parent_outputs = [np.zeros(2, dtype=complex), np.zeros(2, dtype=complex)]
    for in_comp, out_comp, in_slice, out_slice, term in terms:
        parent_outputs[out_comp][out_slice] += term.apply_block(
            parent_inputs[in_comp][in_slice]
        )
    expected = np.concatenate(parent_outputs)

    np.testing.assert_allclose(table.matvec(vector, basis), expected, atol=1e-14)
    assert table.stats["source"] == "compiled_factorized_terms"
    assert set(table.stats["family_names"]) == {"P", "Q", "R"}
    assert table.stats["family_term_counts"] == {"P": 1, "Q": 1, "R": 1}


def test_family_native_factor_kernel_matches_compiled_apply_block():
    class _Entry:
        def __init__(self, shape):
            self.shape = tuple(shape)
            self.size = int(np.prod(shape))

    class _Term:
        input_entry = _Entry((2, 2, 2, 2))
        output_size = 16
        left_stack = np.arange(2 * 2 * 2 * 2 * 2 * 2, dtype=float).reshape(
            2, 2, 2, 2, 2, 2
        ) / 17.0
        right_stack = np.arange(2 * 2 * 2 * 2 * 2 * 2, dtype=float).reshape(
            2, 2, 2, 2, 2, 2
        )[::-1] / 19.0
        _use_direct_contraction = False

        def apply_block(self, block_in):
            tmp = np.einsum(
                "tlkwab,kbcr->tlwacr",
                self.left_stack,
                block_in,
                optimize=False,
            )
            out = np.einsum(
                "tlwacr,twqrdc->ladq",
                tmp,
                self.right_stack,
                optimize=False,
            )
            return out.reshape(self.output_size)

    term = _Term()
    kernel = FamilyNativeFactorKernel.from_compiled_term(term)
    block = np.arange(16, dtype=float).reshape(2, 2, 2, 2) / 23.0

    np.testing.assert_allclose(kernel.apply_block(block), term.apply_block(block))
    assert kernel.stored_elements == term.left_stack.size + term.right_stack.size


def _dense_from_spatial_term_map(term_map, nsites):
    ops = spatial_local_ops()
    ident = ops["I"]
    out = np.zeros((4**nsites, 4**nsites), dtype=complex)
    for (symbol, dofs), factor in term_map.items():
        local = [ident.copy() for _ in range(nsites)]
        for piece, site in zip(symbol.split(), dofs):
            local[site] = ops[piece]
        out += factor * _kron_all(local)
    return out


def test_metric_block_transform_fast_paths_are_orthonormal():
    ident = np.eye(5, dtype=complex)
    transform = _metric_block_transform(ident, tol=1.0e-12)
    np.testing.assert_allclose(transform, ident)

    diag = np.diag([4.0, 9.0, 1.0e-16, 16.0]).astype(complex)
    transform = _metric_block_transform(diag, tol=1.0e-12)
    assert transform.shape == (4, 3)
    np.testing.assert_allclose(
        transform.conj().T @ diag @ transform,
        np.eye(3),
        atol=1.0e-12,
    )

    dense = np.array(
        [
            [2.0, 0.2 - 0.1j],
            [0.2 + 0.1j, 1.5],
        ],
        dtype=complex,
    )
    transform = _metric_block_transform(dense, tol=1.0e-12)
    np.testing.assert_allclose(
        transform.conj().T @ dense @ transform,
        np.eye(2),
        atol=1.0e-12,
    )

    _METRIC_BLOCK_TRANSFORM_CACHE.clear()
    for key in _METRIC_BLOCK_TRANSFORM_CACHE_STATS:
        _METRIC_BLOCK_TRANSFORM_CACHE_STATS[key] = 0
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(40, 40))
    large = raw.T @ raw + 40.0 * np.eye(40)
    transform = _metric_block_transform(large, tol=1.0e-12)
    np.testing.assert_allclose(
        transform.conj().T @ large @ transform,
        np.eye(40),
        atol=1.0e-10,
    )
    assert _METRIC_BLOCK_TRANSFORM_CACHE_STATS["cholesky_fast"] >= 1

    _METRIC_BLOCK_TRANSFORM_CACHE.clear()
    for key in _METRIC_BLOCK_TRANSFORM_CACHE_STATS:
        _METRIC_BLOCK_TRANSFORM_CACHE_STATS[key] = 0
    raw = rng.normal(size=(160, 96))
    low_rank = raw @ raw.T
    transform = _metric_block_transform(low_rank, tol=1.0e-12)
    assert transform.shape == (160, 96)
    np.testing.assert_allclose(
        transform.conj().T @ low_rank @ transform,
        np.eye(96),
        atol=1.0e-9,
    )
    assert _METRIC_BLOCK_TRANSFORM_CACHE_STATS["scipy_subset_eigh"] >= 1


def test_su2_local_action_batches_same_shape_parent_blocks():
    class _ComponentBasis:
        def __init__(self):
            self.component_transforms = (
                np.eye(3, dtype=complex),
                np.eye(3, dtype=complex),
                np.eye(3, dtype=complex),
            )
            self.component_indices = (
                np.arange(3),
                np.arange(3, 6),
                np.arange(6, 9),
            )
            self.n_components = 3
            self.orthonormal_dim = 9

        def _orth_slice(self, idx):
            start = 3 * int(idx)
            return slice(start, start + 3)

    rng = np.random.default_rng(321)
    blocks = tuple(
        (
            idx % 3,
            (idx + 1) % 3,
            rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)),
        )
        for idx in range(6)
    )
    action = SU2LocalAction(
        _ComponentBasis(),
        parent_blocks=blocks,
        backend="python",
    )
    assert action.stats["n_parent_block_batch_groups"] == 1
    assert action.stats["n_parent_block_batched_entries"] == 6

    x = rng.normal(size=9) + 1j * rng.normal(size=9)
    expected = np.zeros(9, dtype=complex)
    for in_comp, out_comp, block in blocks:
        in_slice = slice(3 * in_comp, 3 * in_comp + 3)
        out_slice = slice(3 * out_comp, 3 * out_comp + 3)
        expected[out_slice] += block @ x[in_slice]
    np.testing.assert_allclose(action.matvec(x), expected, atol=1.0e-12)


def _assert_grouped_side_tables_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        left_entries = left[key]
        right_entries = right[key]
        assert len(left_entries) == len(right_entries)
        for left_entry, right_entry in zip(left_entries, right_entries):
            assert left_entry[0] == right_entry[0]
            if isinstance(left_entry[1], dict):
                assert set(left_entry[1]) == set(right_entry[1])
                for channel in left_entry[1]:
                    np.testing.assert_allclose(
                        left_entry[1][channel],
                        right_entry[1][channel],
                        atol=1e-12,
                    )
            else:
                np.testing.assert_allclose(left_entry[1], right_entry[1], atol=1e-12)


def _assert_factor_tables_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert len(left[key]) == len(right[key])
        for left_entry, right_entry in zip(left[key], right[key]):
            if len(left_entry) >= 5 or len(right_entry) >= 5:
                assert left_entry[:3] == right_entry[:3]
                np.testing.assert_allclose(left_entry[3], right_entry[3], atol=1e-12)
                assert tuple(left_entry[4]) == tuple(right_entry[4])
            else:
                assert left_entry[:-1] == right_entry[:-1]
                np.testing.assert_allclose(left_entry[-1], right_entry[-1], atol=1e-12)


def _dense_spin_square(ncas):
    nsites = 2 * ncas
    sp = np.zeros((2**nsites, 2**nsites), dtype=complex)
    sm = np.zeros_like(sp)
    sz = np.zeros_like(sp)
    for p in range(ncas):
        cup_dag = _spin_orbital_creation(2 * p, nsites)
        cup = _spin_orbital_annihilation(2 * p, nsites)
        cdn_dag = _spin_orbital_creation(2 * p + 1, nsites)
        cdn = _spin_orbital_annihilation(2 * p + 1, nsites)
        sp += cup_dag @ cdn
        sm += cdn_dag @ cup
        sz += 0.5 * (cup_dag @ cup - cdn_dag @ cdn)
    return sz @ sz + 0.5 * (sp @ sm + sm @ sp)


def test_dmrg_s2_term_map_matches_exact_dense_spin_square():
    ncas = 2
    nsites = 2 * ncas
    exact = _dense_spin_square(ncas)
    symbolic = _dense_from_symbolic_term_map(_build_s2_term_map(ncas), nsites)
    np.testing.assert_allclose(symbolic, exact, atol=1e-12)


def test_spin_purification_term_map_is_scaled_s2():
    ncas = 3
    shift = 0.37
    ref = _build_s2_term_map(ncas, scale=shift)
    got = _build_spin_purification_term_map(ncas, shift)
    assert set(ref) == set(got)
    for key in ref:
        assert np.allclose(got[key], ref[key], atol=1e-12)


def test_spatial_site_spin_square_resolves_singlet_and_triplet():
    ops = _build_spatial_fermion_operators(2)
    s2 = _build_spatial_s2_matrix(ops)
    singlet = np.zeros(16, dtype=complex)
    triplet = np.zeros(16, dtype=complex)
    singlet[1 * 4 + 2] = 1.0 / np.sqrt(2.0)
    singlet[2 * 4 + 1] = -1.0 / np.sqrt(2.0)
    triplet[1 * 4 + 2] = 1.0 / np.sqrt(2.0)
    triplet[2 * 4 + 1] = 1.0 / np.sqrt(2.0)

    assert np.vdot(singlet, s2 @ singlet).real == pytest.approx(0.0, abs=1e-12)
    assert np.vdot(triplet, s2 @ triplet).real == pytest.approx(2.0, abs=1e-12)


def test_spatial_symbolic_mpo_matches_dense_reference():
    h1 = np.array([[0.2, 0.03], [0.03, -0.1]])
    eri_aa = np.zeros((2, 2, 2, 2))
    eri_aa[0, 0, 0, 0] = 0.7
    eri_aa[1, 1, 1, 1] = 0.5
    eri_aa[0, 0, 1, 1] = 0.2
    eri_aa[1, 1, 0, 0] = 0.2
    h2 = np.stack((np.stack((eri_aa, eri_aa.copy())), np.stack((eri_aa.copy(), eri_aa.copy()))))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    mpo, _, _ = _build_spatial_hamiltonian_tensor_mpo([h1, h1], h2)

    np.testing.assert_allclose(_mpo_to_dense_operator(mpo), dense_ref, atol=1e-12)


def test_spatial_reduced_one_body_qchem_mpo_matches_dense_reference():
    h1 = np.array(
        [
            [0.2, -0.03, 0.04],
            [-0.03, -0.1, 0.07],
            [0.04, 0.07, 0.5],
        ]
    )
    h2 = np.zeros((2, 2, 3, 3, 3, 3))
    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    reduced = build_spatial_reduced_hamiltonian_mpo([h1, h1], h2)

    assert reduced.info["representation"] == "spatial_reduced_mixed_mpo"
    assert reduced.info["one_body_reduced"] is True
    assert reduced.info["two_body"] is False
    np.testing.assert_allclose(
        _dense_matrix_from_nonabelian_mpo_list(reduced.factors),
        dense_ref,
        atol=1e-12,
    )


def test_spatial_reduced_qchem_mpo_with_two_body_terms_matches_dense_reference():
    h1 = np.array([[0.2, 0.03], [0.03, -0.1]])
    h2 = np.zeros((2, 2, 2, 2, 2, 2))
    eri_aa = h2[0, 0]
    eri_aa[0, 0, 0, 0] = 0.7
    eri_aa[1, 1, 1, 1] = 0.5
    eri_aa[0, 0, 1, 1] = 0.2
    eri_aa[1, 1, 0, 0] = 0.2
    h2[0, 1] = eri_aa
    h2[1, 0] = eri_aa
    h2[1, 1] = eri_aa

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    reduced = build_spatial_reduced_hamiltonian_mpo([h1, h1], h2)

    assert reduced.info["one_body_reduced"] is True
    assert reduced.info["two_body"] is True
    assert reduced.info["two_body_representation"] == "spinfree_scalar_coupled_eri"
    assert reduced.info["two_body_builder"] == "SpatialSpinFreeERIBuilder"
    assert reduced.info["pipeline"] == (
        "qchem_integrals->spatial_reduced_hamiltonian_builder->spinfree_eri_builder->rank_coupled_mpo"
    )
    assert reduced.info["two_body_reduced_string_terms"] == 0
    assert reduced.info["two_body_scalar_density_terms"] == 0
    assert reduced.info["two_body_scalar_product_terms"] > 0
    assert reduced.info["two_body_symbolic_terms"] == 0
    assert reduced.complementary_operators.names == ("S", "R", "A", "P", "B", "Q")
    assert reduced.info["complementary_operator_family_names"] == ("S", "R", "A", "P", "B", "Q")
    assert reduced.info["complementary_operator_total_terms"] > 0
    assert reduced.info["final_mpo_reduced_metadata"] is True
    assert all(isinstance(core, RankCoupledMPO) for core in reduced.factors)
    np.testing.assert_allclose(
        _dense_matrix_from_nonabelian_mpo_list(reduced.factors),
        dense_ref,
        atol=1e-12,
    )


def test_spatial_spinfree_eri_builder_owns_reduced_two_body_mpo():
    eri = np.zeros((3, 3, 3, 3))
    eri[0, 2, 1, 0] = 0.07
    eri[2, 0, 0, 1] = -0.04
    eri[1, 1, 2, 2] = 0.11
    eri[0, 1, 1, 2] = 0.05

    dense_ref = _dense_from_spatial_term_map(
        spatial_two_body_spinfree_term_map(eri),
        3,
    )
    factors, info = SpatialSpinFreeERIBuilder(3, eri).build(return_info=True)

    assert info["total_terms"] > 0
    assert info["scalar_product_terms"] > 0
    assert all(isinstance(core, RankCoupledMPO) for core in factors)
    np.testing.assert_allclose(
        _dense_matrix_from_nonabelian_mpo_list(factors),
        dense_ref,
        atol=1e-12,
    )


def test_spatial_reduced_qchem_mpo_compresses_non_density_two_body_pairs_exactly():
    h1 = np.array([[0.2, 0.03], [0.03, -0.1]])
    h2 = np.zeros((2, 2, 2, 2, 2, 2))
    eri_aa = h2[0, 0]
    eri_aa[0, 1, 1, 0] = 0.13
    eri_aa[1, 0, 0, 1] = 0.13
    h2[0, 1] = eri_aa
    h2[1, 0] = eri_aa
    h2[1, 1] = eri_aa

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    reduced = build_spatial_reduced_hamiltonian_mpo([h1, h1], h2)

    assert reduced.info["two_body_representation"] == "spinfree_scalar_coupled_eri"
    assert reduced.info["two_body_reduced_string_terms"] == 0
    assert reduced.info["two_body_compressed_pair_terms"] == 0
    assert reduced.info["two_body_scalar_product_terms"] > 0
    assert reduced.info["two_body_symbolic_terms"] == 0
    assert reduced.info["final_mpo_reduced_metadata"] is True
    assert all(isinstance(core, RankCoupledMPO) for core in reduced.factors)
    np.testing.assert_allclose(
        _dense_matrix_from_nonabelian_mpo_list(reduced.factors),
        dense_ref,
        atol=1e-12,
    )


def test_spatial_reduced_qchem_mpo_handles_generic_two_body_terms_without_symbolic_fallback():
    h1 = np.zeros((3, 3))
    h2 = np.zeros((2, 2, 3, 3, 3, 3))
    eri_aa = h2[0, 0]
    eri_aa[0, 2, 1, 0] = 0.07
    eri_aa[2, 0, 0, 1] = -0.04
    eri_aa[1, 1, 2, 2] = 0.11
    eri_aa[0, 1, 1, 2] = 0.05
    h2[0, 1] = eri_aa
    h2[1, 0] = eri_aa
    h2[1, 1] = eri_aa

    dense_ref = _dense_from_spatial_term_map(
        spatial_two_body_spinfree_term_map(eri_aa),
        3,
    )
    reduced = build_spatial_reduced_hamiltonian_mpo([h1, h1], h2)

    assert reduced.info["two_body_representation"] == "spinfree_scalar_coupled_eri"
    assert reduced.info["two_body_reduced_string_terms"] == 0
    assert reduced.info["two_body_symbolic_terms"] == 0
    assert reduced.info["two_body_scalar_product_terms"] > 0
    assert reduced.info["final_mpo_reduced_metadata"] is True
    assert all(isinstance(core, RankCoupledMPO) for core in reduced.factors)
    np.testing.assert_allclose(
        _dense_matrix_from_nonabelian_mpo_list(reduced.factors),
        dense_ref,
        atol=1e-12,
    )


def test_spatial_reduced_qchem_mpo_uses_general_scalar_coupled_builder_for_adjacent_two_body_terms():
    h1 = np.zeros((4, 4))
    h2 = np.zeros((2, 2, 4, 4, 4, 4))
    h2[:, :, 0, 1, 2, 3] = 2.0

    dense_ref = _dense_from_spatial_term_map(
        spatial_two_body_spinfree_term_map(h2[0, 0]),
        4,
    )
    reduced = build_spatial_reduced_hamiltonian_mpo([h1, h1], h2)

    assert reduced.info["representation"] == "spatial_reduced_spinfree_mpo"
    assert reduced.info["two_body_representation"] == "we_general_reduced_strings"
    assert reduced.info["two_body_reduced_string_terms"] > 0
    assert reduced.info["two_body_scalar_product_terms"] == 0
    assert reduced.info["two_body_symbolic_terms"] == 0
    assert reduced.info["final_mpo_reduced_metadata"] is True
    assert all(isinstance(core, RankCoupledMPO) for core in reduced.factors)
    np.testing.assert_allclose(
        _dense_matrix_from_nonabelian_mpo_list(reduced.factors),
        dense_ref,
        atol=1e-12,
    )


def test_spatial_dmrg_build_can_use_reduced_mpo():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    grouped = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    grouped.build()
    reduced = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        site="spatial",
        spatial_reduced_mpo=True,
        verbose=0,
    )
    reduced.build()

    assert reduced._active_integral_build_info["representation"] == "spatial_reduced_spinfree_mpo"
    assert reduced._active_hamiltonian is not None
    assert reduced._active_hamiltonian.initialize_system_kwargs()["n_sites"] == 2
    assert reduced._active_hamiltonian.initialize_system_kwargs()["n_elec"] == 2
    assert reduced._active_hamiltonian.mpo is reduced.H
    np.testing.assert_allclose(
        _dense_matrix_from_nonabelian_mpo_list(reduced.H),
        _dense_matrix_from_nonabelian_mpo_list(grouped.H),
        atol=1e-10,
    )


def test_dmrg_accepts_ri_active_integrals():
    atom = "H 0 0 0; H 0 0 0.74"
    dense_mol = Molecule(atom=atom, unit="angstrom", basis="cc-pvdz")
    dense_mol.build(driver="builtin", eri="dense")
    dense_mf = RHF(dense_mol).run(verbose=0)

    ri_mol = Molecule(atom=atom, unit="angstrom", basis="cc-pvdz")
    ri_mol.build(driver="builtin", eri="ri")
    ri_mf = RHF(ri_mol).run(verbose=0)

    dense = DMRG(dense_mf, ncas=2, nelecas=2, D=8, init_guess="hf", verbose=0)
    dense.run(nsweeps=4, symmetry_list=None)

    ri = DMRG(ri_mf, ncas=2, nelecas=2, D=8, init_guess="hf", verbose=0)
    ri.run(nsweeps=4, symmetry_list=None)

    assert ri._active_integral_build_info["mode"] == "ri"
    assert ri._active_integral_build_info["factorized_integrals"] is True
    assert ri._active_integral_build_info["aux_rank"] == ri_mol.eri_factors.shape[0]
    assert ri.e_tot == pytest.approx(dense.e_tot, abs=5.0e-5)


def test_dmrg_auto_prefers_dense_when_dense_and_factors_exist():
    atom = "H 0 0 0; H 0 0 0.74"
    mol = Molecule(atom=atom, unit="angstrom", basis="cc-pvdz")
    mol.build(driver="builtin", eri="dense+ri")
    mf = RHF(mol).run(verbose=0)

    dense = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", verbose=0)
    dense.run(nsweeps=4, symmetry_list=None)

    ri = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        integral_backend="ri",
        verbose=0,
    )
    ri.run(nsweeps=4, symmetry_list=None)

    assert dense._active_integral_build_info["mode"] == "dense"
    assert dense._active_integral_build_info["factorized_integrals"] is False
    assert ri._active_integral_build_info["mode"] == "ri"
    assert ri._active_integral_build_info["factorized_integrals"] is True


def test_abelian_sz_hf_guess_can_leave_hf_determinant():
    atom = "H 0 0 0; H 0 0 0.74"
    mol = Molecule(atom=atom, unit="angstrom", basis="cc-pvdz")
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    dense = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", verbose=0)
    dense.run(nsweeps=8, symmetry_list=None)

    sz = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        verbose=0,
        symmetry="sz",
    )
    sz.run(nsweeps=8, symmetry_list=None)

    assert sz.e_tot == pytest.approx(dense.e_tot, abs=1.0e-8)
    assert sz.e_tot < mf.e_tot - 1.0e-4


def test_autompo_preserves_recursive_symbolic_renormalized_algebra(monkeypatch):
    sites = build_random_spatial_mps(4, seed=11, bond_multiplicity=2)
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=1.0,
        onsite_u=2.0,
        chemical_potential=0.1,
    )

    assert any(getattr(core, "symbolic_transitions", ()) for core in mpo)
    transitions, used_metadata = symbolic_mpo_core_transitions(mpo[0])
    assert used_metadata is True
    assert transitions

    stack = RenormalizedBlockStack(namespace="hamiltonian")
    sweep = BlockSparseEnvironmentChain.build(
        sites,
        mpo,
        renormalized_blocks=stack,
    ).start_sweep("lr")
    sweep.advance_after_update(0, sites[0], sites[1])

    stats = stack.stats
    assert stats["symbolic_operator_tables"] > 0
    assert stats["symbolic_operator_terms"] > 0
    assert stats["symbolic_operator_numeric_payloads"] > 0
    assert stats["symbolic_operator_max_path_length"] > 0
    assert stats["symbolic_operator_used_mpo_metadata"] is True
    assert any(
        str(source).startswith("symbolic_")
        for source in stats["side_operator_table_sources"]
    )

    from pyqed.mps.nonabelian.environment import (
        _build_rank_coupled_left_factor_table,
        _build_rank_coupled_right_factor_table,
        _group_boundary_blocks_by_ket,
    )

    left_entry = stack.get("left", 1)
    right_entry = stack.get("right", 1)
    assert left_entry is not None
    assert right_entry is not None
    assert left_entry.symbolic_operator_table.stats["owns_numeric_payloads"] is True
    assert right_entry.symbolic_operator_table.stats["owns_numeric_payloads"] is True

    left_symbolic = left_entry.symbolic_operator_table.group_boundary_blocks(
        representation="rank_coupled_by_ket",
    )
    left_numeric = _group_boundary_blocks_by_ket(
        left_entry.block,
        "rank_coupled_by_ket",
    )
    right_symbolic = right_entry.symbolic_operator_table.group_boundary_blocks(
        representation="rank_coupled_by_ket",
    )
    right_numeric = _group_boundary_blocks_by_ket(
        right_entry.block,
        "rank_coupled_by_ket",
    )
    _assert_grouped_side_tables_equal(left_symbolic, left_numeric)
    _assert_grouped_side_tables_equal(right_symbolic, right_numeric)

    _assert_factor_tables_equal(
        _build_rank_coupled_left_factor_table(left_symbolic, mpo[0]),
        _build_rank_coupled_left_factor_table(left_numeric, mpo[0]),
    )
    _assert_factor_tables_equal(
        left_entry.symbolic_operator_table.factor_boundary_blocks(
            "rank_coupled_left_factor_by_ket",
            mpo[0],
        ),
        _build_rank_coupled_left_factor_table(left_numeric, mpo[0]),
    )
    _assert_factor_tables_equal(
        _build_rank_coupled_right_factor_table(right_symbolic, mpo[1]),
        _build_rank_coupled_right_factor_table(right_numeric, mpo[1]),
    )
    _assert_factor_tables_equal(
        right_entry.symbolic_operator_table.factor_boundary_blocks(
            "rank_coupled_right_factor_by_ket",
            mpo[1],
        ),
        _build_rank_coupled_right_factor_table(right_numeric, mpo[1]),
    )

    from pyqed.mps.nonabelian import environment as env_mod
    from pyqed.mps.nonabelian.effective import EffectiveBlockOperator

    def fail_raw_grouping(*args, **kwargs):
        raise AssertionError("raw boundary-map grouping fallback was used")

    original_rank_coupled_precompute = (
        env_mod._precompute_two_site_rank_coupled_factorized_terms
    )

    def require_prepared_factor_tables(*args, **kwargs):
        assert kwargs.get("left_factor_table") is not None
        assert kwargs.get("right_factor_table") is not None
        return original_rank_coupled_precompute(*args, **kwargs)

    monkeypatch.setattr(EffectiveBlockOperator, "_group_side_blocks", fail_raw_grouping)
    monkeypatch.setattr(
        env_mod,
        "_precompute_two_site_rank_coupled_factorized_terms",
        require_prepared_factor_tables,
    )

    merged = merge_mps_sites(sites[1], sites[2])
    operator = sweep.bond_operator(1, merged)

    assert operator.metadata["symbolic_boundary_payloads"][
        "symbolic_boundary_payload_source"
    ] == "symbolic_table"
    assert operator.metadata["symbolic_boundary_payloads"]["symbolic_payloads_owned"] is True
    assert getattr(operator.aux_packed_matvec, "symbolic_boundary_payloads")[
        "symbolic_payloads_owned"
    ] is True


def test_complementary_operator_stack_survives_reapplying_same_families():
    h1 = np.array([[0.2, 0.03], [0.03, -0.1]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.7
    eri[0, 0, 1, 1] = 0.2
    families = build_spatial_complementary_operator_families(h1, eri)

    stack = RenormalizedBlockStack(
        namespace="hamiltonian",
        complementary_operator_families=families,
    )
    stack.put("left", 0, {}, source="initialized")
    complementary_stack = stack.complementary_operator_stack

    stack.set_complementary_operator_families(families)
    stack.put("left", 1, {}, source="advanced_left", parent_key=stack.key("left", 0))

    assert stack.complementary_operator_stack is complementary_stack
    assert stack.stats["complementary_operator_stack"]["n_entries"] == 2
    assert stack.stats["complementary_operator_stack"]["puts"] == 2
    assert stack.stats["complementary_operator_stack"]["advances"] == 1
    assert stack.stats["complementary_operator_stack"]["numeric_payload_terms"] > 0
    assert stack.stats["complementary_operator_stack"]["numeric_payload_cross_terms"] > 0
    left_entry = complementary_stack.get("left", 1)
    assert left_entry is not None
    assert left_entry.family_payloads["P"].cross_terms > 0
    assert left_entry.family_payloads["P"].coefficient_norm > 0.0


def test_block2_like_strict_symbolic_stack_covers_all_bonds(monkeypatch):
    sites = build_random_spatial_mps(4, seed=17, bond_multiplicity=2)
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=0.8,
        onsite_u=1.5,
        chemical_potential=-0.2,
    )
    stack = RenormalizedBlockStack(namespace="hamiltonian")
    chain = BlockSparseEnvironmentChain.build(
        sites,
        mpo,
        renormalized_blocks=stack,
        require_symbolic_payloads=True,
    )

    from pyqed.mps.nonabelian.environment import (
        _build_rank_coupled_left_factor_table,
        _build_rank_coupled_right_factor_table,
        _group_boundary_blocks_by_ket,
    )

    for bond in range(len(sites) - 1):
        left_entry = stack.get("left", bond)
        right_entry = stack.get("right", bond + 1)
        assert left_entry is not None
        assert right_entry is not None

        left_symbolic = left_entry.symbolic_operator_table.group_boundary_blocks(
            representation="rank_coupled_by_ket",
        )
        left_numeric = _group_boundary_blocks_by_ket(
            left_entry.block,
            "rank_coupled_by_ket",
        )
        right_symbolic = right_entry.symbolic_operator_table.group_boundary_blocks(
            representation="rank_coupled_by_ket",
        )
        right_numeric = _group_boundary_blocks_by_ket(
            right_entry.block,
            "rank_coupled_by_ket",
        )
        _assert_grouped_side_tables_equal(left_symbolic, left_numeric)
        _assert_grouped_side_tables_equal(right_symbolic, right_numeric)
        _assert_factor_tables_equal(
            left_entry.symbolic_operator_table.rank_coupled_left_factor_table(mpo[bond]),
            _build_rank_coupled_left_factor_table(left_numeric, mpo[bond]),
        )
        _assert_factor_tables_equal(
            right_entry.symbolic_operator_table.rank_coupled_right_factor_table(mpo[bond + 1]),
            _build_rank_coupled_right_factor_table(right_numeric, mpo[bond + 1]),
        )

    from pyqed.mps.nonabelian import environment as env_mod
    from pyqed.mps.nonabelian.effective import EffectiveBlockOperator

    def fail_raw_grouping(*args, **kwargs):
        raise AssertionError("strict block2-like path used raw boundary-map grouping")

    original_rank_coupled_precompute = (
        env_mod._precompute_two_site_rank_coupled_factorized_terms
    )

    def require_prepared_factor_tables(*args, **kwargs):
        assert kwargs.get("left_factor_table") is not None
        assert kwargs.get("right_factor_table") is not None
        return original_rank_coupled_precompute(*args, **kwargs)

    monkeypatch.setattr(EffectiveBlockOperator, "_group_side_blocks", fail_raw_grouping)
    monkeypatch.setattr(
        env_mod,
        "_precompute_two_site_rank_coupled_factorized_terms",
        require_prepared_factor_tables,
    )

    for bond in range(len(sites) - 1):
        merged = merge_mps_sites(sites[bond], sites[bond + 1])
        local_operator = chain.effective_block_operator(bond, merged).to_local_operator()
        assert local_operator.metadata["symbolic_boundary_payloads"][
            "symbolic_boundary_payload_source"
        ] == "symbolic_table"
        assert local_operator.metadata["symbolic_boundary_payloads"][
            "symbolic_payloads_owned"
        ] is True

    loose_chain = BlockSparseEnvironmentChain.build(
        sites,
        mpo,
        renormalized_blocks=None,
        require_symbolic_payloads=True,
    )
    with pytest.raises(RuntimeError, match="Strict symbolic local build requires"):
        loose_chain.effective_block_operator(0, merge_mps_sites(sites[0], sites[1]))


def test_qchem_su2_block2_like_rejects_raw_boundary_fallback(monkeypatch):
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    from pyqed.mps.nonabelian import environment as env_mod
    from pyqed.mps.nonabelian.effective import EffectiveBlockOperator

    def fail_raw_grouping(*args, **kwargs):
        raise AssertionError("qchem block2_like path used raw boundary-map grouping")

    original_rank_coupled_precompute = (
        env_mod._precompute_two_site_rank_coupled_factorized_terms
    )

    def require_prepared_factor_tables(*args, **kwargs):
        assert kwargs.get("left_factor_table") is not None
        assert kwargs.get("right_factor_table") is not None
        return original_rank_coupled_precompute(*args, **kwargs)

    monkeypatch.setattr(EffectiveBlockOperator, "_group_side_blocks", fail_raw_grouping)
    monkeypatch.setattr(env_mod, "_group_boundary_blocks_by_ket", fail_raw_grouping)
    monkeypatch.setattr(
        env_mod,
        "_precompute_two_site_rank_coupled_factorized_terms",
        require_prepared_factor_tables,
    )

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=2,
        local_basis_policy="block2_like",
        orthonormalized_operator_dim=512,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
        conv_tol=-1.0,
    )

    assert dmrg.e_tot == pytest.approx(-2.177899323464, abs=1e-8)
    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    metadata = [
        objective.get("renormalized_operator_metadata")
        for objective in objectives
        if objective.get("renormalized_operator_metadata") is not None
    ]
    assert metadata
    assert all(
        item["symbolic_boundary_payloads"]["symbolic_boundary_payload_source"]
        == "symbolic_table"
        for item in metadata
    )
    assert all(
        item["symbolic_boundary_payloads"]["symbolic_payloads_owned"] is True
        for item in metadata
    )
    block_stack_stats = dmrg.dmrg.history[-1]["renormalized_block_stack_stats"]
    assert "rank_coupled_left_factor_by_ket" in block_stack_stats[
        "side_operator_table_representations"
    ]
    assert "rank_coupled_right_factor_by_ket" in block_stack_stats[
        "side_operator_table_representations"
    ]
    complementary = block_stack_stats["complementary_operator_families"]
    assert complementary["family_names"] == ("S", "R", "A", "P", "B", "Q")
    assert complementary["families"]["P"]["n_terms"] > 0
    complementary_stack = block_stack_stats["complementary_operator_stack"]
    assert complementary_stack["family_names"] == ("S", "R", "A", "P", "B", "Q")
    assert complementary_stack["n_entries"] > 0
    assert complementary_stack["advances"] > 0
    assert complementary_stack["numeric_payload_terms"] > 0
    assert complementary_stack["numeric_payload_cross_terms"] > 0
    assert complementary_stack["family_operator_tables"] > 0
    assert complementary_stack["family_operator_table_payload_blocks"] > 0
    assert complementary_stack["family_operator_table_stored_elements"] > 0
    assert complementary_stack["family_operator_table_symbolic_terms"] > 0
    assert complementary_stack["numeric_payload_families"]["P"]["cross_terms"] > 0
    assert dmrg.dmrg.history[0]["reused_prebuilt_boundary_side"] is None
    assert dmrg.dmrg.history[1]["reused_prebuilt_boundary_side"] == "left"
    moving_stats = dmrg.dmrg.history[-1]["moving_environment_stats"]
    assert moving_stats["environment_rebuilds"] == 1
    assert moving_stats["boundary_side_reuses"] == 1
    assert moving_stats["valid_boundary_side"] == "right"
    assert moving_stats["complementary_operator_advances"] > 0
    assert dmrg.dmrg.history[-1]["hamiltonian_complementary_operators"][
        "family_names"
    ] == ("S", "R", "A", "P", "B", "Q")


def test_grouped_spin_orbital_mpo_matches_spatial_dense_reference():
    h1 = np.array([[0.2, 0.03], [0.03, -0.1]])
    eri_aa = np.zeros((2, 2, 2, 2))
    eri_aa[0, 0, 0, 0] = 0.7
    eri_aa[1, 1, 1, 1] = 0.5
    eri_aa[0, 0, 1, 1] = 0.2
    eri_aa[1, 1, 0, 0] = 0.2
    h2 = np.stack((np.stack((eri_aa, eri_aa.copy())), np.stack((eri_aa.copy(), eri_aa.copy()))))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    spin_mpo, _, _ = _build_spin_orbital_dense_hamiltonian_tensor_mpo([h1, h1], h2, 2)
    grouped = _group_spin_orbital_mpo_pairs(spin_mpo)

    np.testing.assert_allclose(_mpo_to_dense_operator(grouped), dense_ref, atol=1e-12)


def test_spatial_dmrg_supports_abelian_charge_sz_symmetry():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="angstrom", basis="sto-3g")
    mol.build(driver="pyscf")
    mf = mol.RHF(verbose=0).run()

    dense = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dense.run(nsweeps=4, symmetry_list=None)
    sym = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    sym.run(nsweeps=4, symmetry_list=["charge", "sz"])

    assert hasattr(sym.dmrg.ground_state.Bs[0], "qns")
    assert sym.e_tot == pytest.approx(dense.e_tot, abs=1e-8)


def test_h4_spatial_block2_like_uses_cpp_hot_path_end_to_end():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    svd_stats = getattr(abelian_direct, "_ABELIAN_SVD_KERNEL_STATS", None)
    if hasattr(svd_stats, "clear"):
        svd_stats.clear()
    abelian_direct._ABELIAN_SVD_KERNEL_LAST_ERROR = ""

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        spatial_family_environment_backend="block2_table",
    )

    dmrg.run(nsweeps=4)

    assert dmrg.e_tot == pytest.approx(-2.13944255109448, abs=1e-9)
    assert dmrg._active_integral_build_info["representation"] == (
        "spatial_block2_table_carrier_mpo"
    )
    assert dmrg._active_integral_build_info["carrier_only_family_hamiltonian"] is True
    build_timings = dmrg._active_integral_build_info[
        "complementary_operator_family_build_timings"
    ]
    active_build_timings = dmrg._active_integral_build_info["build_timings"]
    assert dmrg._active_integral_build_info["qchem_compile_backend_actual"] == "cpp"
    assert build_timings["qchem_block2_setup_backend_actual"] == (
        "cpp_qchem_spatial_block2_setup"
    )
    assert build_timings["qchem_family_backend_actual"] == "cpp"
    assert (
        build_timings["family_term_map_backend_actual"]
        == "cpp_spatial_jw_family_term_maps"
    )
    assert build_timings["family_mpo_backend_actual"] == (
        "cpp_spatial_sparse_tt_svd_family_mpos"
    )
    assert build_timings["family_mpo_owner_backend_actual"] == (
        "cpp_moving_environment"
    )
    assert build_timings["family_descriptor_backend_actual"] == (
        "cpp_spatial_qchem_family_descriptor"
    )
    assert build_timings["family_descriptor_families"] == 3
    assert build_timings["family_mpo_owner_builder"] == "cpp_descriptor"
    assert build_timings["family_mpo_route_cache_owner"].startswith(
        "moving-environment:"
    )
    assert active_build_timings["carrier_build_backend_actual"] == (
        "cpp_qchem_spatial_block2_setup"
    )
    assert active_build_timings["cpp_owned_converted_family_mpo_families"] == 3
    assert (
        "moving_environment_cpp_owned_family_mpo_key"
        in dmrg._active_integral_build_info["resolved_abelian_matvec_options"]
    )
    assert (
        "moving_environment_cpp_qchem_family_descriptor_key"
        in dmrg._active_integral_build_info["resolved_abelian_matvec_options"]
    )
    assert (
        dmrg._active_integral_build_info[
            "moving_environment_cpp_owner_reused_from_build"
        ]
        is True
    )
    assert (
        dmrg._active_integral_build_info["resolved_abelian_matvec_options"][
            "moving_environment_cpp_state_owner_instance"
        ]
        == "cpp_moving_environment_build_owner"
    )
    family_mpo_info = dmrg._active_integral_build_info[
        "complementary_operator_family_mpos"
    ]
    assert family_mpo_info["R"]["source"] == "cpp_spatial_sparse_tt_svd_mpo"
    assert family_mpo_info["P"]["source"] == (
        "cpp_spatial_sparse_tt_svd_mpo_split_summary"
    )
    assert family_mpo_info["P:g0"]["source"] == "cpp_spatial_sparse_tt_svd_mpo_split"
    assert family_mpo_info["P:g0"]["tt_svd_backend"] == "dense_tensor_small_system"
    assert family_mpo_info["P:g0"]["tt_svd_sparse_bypass"] == "dense_small_system"
    assert family_mpo_info["P:g0"]["tt_svd_route_cache_backend"] == (
        "dense_small_system_bypass"
    )
    assert family_mpo_info["P:g0"]["tt_svd_route_cache_owner"].startswith(
        "moving-environment:"
    )
    assert family_mpo_info["P:g0"]["tt_svd_dense_elements"] == 16**4
    assert family_mpo_info["P:g0"]["mpo_max_bond"] < (
        family_mpo_info["P:g0"]["uncompressed_terms"]
    )
    moving = dmrg.dmrg.environment_profile["moving_environment"]
    assert moving["environment_update_backend"] == "cpp_native_environment"
    assert moving["cpp_environment_update_backend_actual"] == "cpp_environment_plan"
    assert moving["cpp_environment_update_calls"] >= 12
    assert moving["cpp_environment_update_right_calls"] >= 12
    assert moving["cpp_bond_step_transaction_environment_updates"] >= 24
    assert moving["cpp_environment_update_failures"] == 0
    assert moving["owner_half_sweep_backend_actual"] == (
        "cpp_owner_sweep_schedule_plan"
    )
    assert moving.get("family_environment_cpp_owned_mpo_installs", 0) == 0
    assert moving["family_environment_cpp_owned_descriptor_installs"] >= 1
    assert moving["family_environment_cpp_qchem_descriptor_families"] == 3
    assert moving["cpp_moving_environment_owned_family_mpo_records"] >= 1
    assert (
        moving[
            "cpp_moving_environment_family_mpo_descriptor_from_owned_installs"
        ]
        >= 1
    )
    assert (
        moving[
            "cpp_moving_environment_spatial_qchem_family_descriptor_records"
        ]
        >= 1
    )
    assert (
        moving[
            "cpp_moving_environment_spatial_qchem_family_descriptor_installs"
        ]
        >= 1
    )
    assert (
        moving[
            "cpp_moving_environment_spatial_qchem_family_descriptor_mpo_builds"
        ]
        >= 1
    )
    assert moving["owner_sweep_schedule_backend_actual"] == (
        "cpp_owner_sweep_schedule_plan"
    )
    assert moving["owner_sweep_schedule_builder_actual"] == (
        "cpp_alternating_sweep_schedule_plan"
    )
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_records"] >= 1
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_installs"] >= 1
    assert (
        moving[
            "cpp_moving_environment_owner_sweep_schedule_plan_alternating_installs"
        ]
        >= 1
    )
    assert (
        moving[
            "cpp_moving_environment_owner_sweep_schedule_plan_alternating_expanded_halves"
        ]
        >= len(dmrg.dmrg.sweep_history)
    )
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_noise_sets"] > 0
    assert (
        moving[
            "cpp_moving_environment_owner_sweep_schedule_plan_noise_set_failures"
        ]
        == 0
    )
    assert moving["cpp_moving_environment_owner_local_optimize_record_noise_sets"] > 0
    assert (
        moving["cpp_moving_environment_owner_local_optimize_native_merge_calls"]
        > 0
    )
    assert (
        moving["cpp_moving_environment_owner_local_optimize_native_merge_accepted"]
        == moving["cpp_moving_environment_owner_local_optimize_native_merge_calls"]
    )
    assert (
        moving["cpp_moving_environment_owner_local_optimize_native_merge_failures"]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_local_optimize_native_noise_injections"
        ]
        > 0
    )
    assert (
        moving["cpp_moving_environment_owner_local_optimize_native_noise_blocks"]
        > 0
    )
    assert moving["cpp_moving_environment_owner_local_optimize_bridge_merge_calls"] == 0
    assert moving["cpp_moving_environment_owner_local_optimize_boundary_stack_reads"] > 0
    assert (
        moving[
            "cpp_moving_environment_owner_local_optimize_boundary_bridge_calls"
        ]
        == 0
    )
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_hits"] > 0
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_misses"] == 0
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_runs"] > 0
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_halves"] > 0
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_converged"] > 0
    assert (
        moving["cpp_moving_environment_owner_sweep_schedule_plan_history_rows"]
        == len(dmrg.dmrg.sweep_history)
    )
    assert moving["cpp_moving_environment_owner_typed_half_sweep_plan_records"] >= 2
    assert moving["cpp_moving_environment_owner_typed_half_sweep_plan_installs"] >= 2
    assert moving["cpp_moving_environment_owner_typed_half_sweep_plan_hits"] > 0
    assert moving["cpp_moving_environment_owner_typed_half_sweep_plan_misses"] == 0
    assert moving["cpp_moving_environment_owner_typed_half_sweep_plan_runs"] > 0
    assert moving["cpp_moving_environment_owner_typed_half_sweep_plan_bonds"] > 0
    assert all(row.get("updates") for row in dmrg.dmrg.sweep_history)
    assert moving["owner_typed_half_sweep_key_lookups"] == 0
    assert (
        moving[
            "cpp_moving_environment_owner_typed_half_sweep_template_plan_installs"
        ]
        >= 2
    )
    assert (
        moving["cpp_moving_environment_owner_typed_half_sweep_template_plan_bonds"]
        >= 4
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_half_sweep_template_local_records"
        ]
        >= 4
    )
    assert (
        moving["cpp_moving_environment_owner_typed_half_sweep_template_step_records"]
        >= 4
    )
    assert moving["owner_typed_half_sweep_new_installs"] > 0
    assert moving["owner_typed_half_sweep_python_update_callbacks"] == 0
    assert moving["cpp_moving_environment_owner_typed_bond_step_record_hits"] > 0
    assert moving["cpp_moving_environment_owner_typed_bond_step_record_misses"] == 0
    assert moving["cpp_bond_step_transaction_record_builds"] > 0
    assert moving["cpp_bond_step_transaction_record_prepares"] > 0
    assert moving["cpp_bond_step_transaction_record_consumes"] > 0
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_environment_record_prepares"
        ]
        > 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_environment_record_consumes"
        ]
        > 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_python_prepare_calls"
        ]
        == 0
    )
    assert (
        moving["cpp_moving_environment_owner_typed_bond_step_python_move_calls"]
        == 0
    )
    assert moving["cpp_moving_environment_owner_bond_step_record_installs"] == 0
    assert moving["cpp_moving_environment_owner_bond_step_record_hits"] == 0
    assert moving["cpp_moving_environment_grouped_table_davidson_calls"] > 0
    assert moving["cpp_moving_environment_grouped_table_davidson_workspace_reuses"] > 0
    assert moving["compiled_flat_matvec_builds"] == 0
    assert moving["compiled_flat_matvec_cache_hits"] == 0
    assert moving["local_operator_builds"] == 0
    assert moving["local_operator_reuses"] == 0
    assert moving["operatorless_local_problem_binds"] > 0
    assert moving["owner_local_problem_bind_backend_actual"] == (
        "cpp_owner_operatorless_local_problem"
    )
    assert moving["owner_operatorless_local_problem_binds"] > 0
    assert moving["owner_operatorless_local_problem_rejections"] == 0
    assert moving["cpp_moving_environment_owner_local_problem_bind_owner_calls"] > 0
    assert (
        moving[
            "cpp_moving_environment_owner_local_problem_bind_set_bond_fallbacks"
        ]
        == 0
    )
    assert moving["operatorless_local_problem_solve_accepts"] == 0
    assert moving["operatorless_local_problem_solve_rejections"] == 0
    assert moving["owner_local_optimize_calls"] > 0
    assert moving["owner_local_optimize_accepts"] > 0
    assert moving["owner_local_optimize_rejections"] == 0
    assert moving["owner_local_optimize_failures"] == 0
    assert moving["owner_local_optimize_solve_actual"] == "cpp_grouped_update"
    assert moving["owner_local_optimize_site_commits"] > 0
    assert moving["owner_local_optimize_commit_actual"] == "cpp_owner_site_chain"
    assert moving["owner_site_chain_backend_actual"] == "cpp_owner_site_chain"
    assert moving["owner_site_chain_gets"] > 0
    assert moving["owner_site_chain_sets"] > 0
    assert moving["owner_site_chain_syncs"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_records"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_installs"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_gets"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_sets"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_syncs"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_failures"] == 0
    assert moving["owner_local_optimize_guess_cache_sets"] > 0
    assert moving["owner_local_grouped_solve_update_accepts"] > 0
    assert (
        moving["owner_local_grouped_solve_update_backend_actual"]
        == "cpp_moving_environment_direct_grouped_update"
    )
    assert moving["owner_local_grouped_cpp_table_refresh_attempts"] > 0
    assert (
        moving["owner_local_grouped_cpp_table_refresh_accepts"]
        == moving["owner_local_grouped_cpp_table_refresh_attempts"]
    )
    assert moving.get("owner_local_grouped_cpp_table_refresh_fallbacks", 0) == 0
    assert moving.get("owner_local_grouped_cpp_table_refresh_failures", 0) == 0
    assert (
        moving["owner_local_grouped_cpp_table_refresh_backend_actual"]
        == "cpp_family_mpo_descriptor_table_record"
    )
    assert moving["owner_local_grouped_cpp_prepare_calls"] > 0
    assert moving["owner_local_grouped_cpp_prepare_accepts"] > 0
    assert moving.get("owner_local_grouped_cpp_prepare_failures", 0) == 0
    assert moving["owner_local_grouped_direct_prepare_calls"] == 0
    assert moving["owner_local_grouped_direct_prepare_accepts"] == 0
    assert moving["owner_local_grouped_direct_prepare_failures"] == 0
    assert moving["owner_local_grouped_direct_solve_update_calls"] > 0
    assert moving["owner_local_grouped_direct_solve_update_accepts"] > 0
    assert moving["owner_local_grouped_direct_raw_update_accepts"] > 0
    assert moving["owner_local_grouped_direct_solve_update_failures"] == 0
    assert moving["owner_local_grouped_direct_solve_update_fallbacks"] == 0
    assert moving["owner_local_optimize_update_payload_actual"] == "cpp_raw_site_tensors"
    assert moving["cpp_moving_environment_owner_local_optimize_runner_accepted"] > 0
    assert moving["cpp_moving_environment_owner_local_optimize_runner_rejections"] == 0
    assert moving["family_environment_descriptor_families"] == 3
    assert moving["family_environment_requests"] == 0
    assert moving["family_environment_cpp_descriptor_requests"] > 0
    assert moving["family_environment_cpp_descriptor_installs"] > 0
    assert moving.get("family_environment_cpp_descriptor_payload_builds", 0) == 0
    assert moving["cpp_named_raw_payload_plan_backend_actual"] == (
        "cpp_family_mpo_descriptor_owner_table_refresh"
    )
    assert moving["cpp_named_raw_payload_plan_fused_table_refreshes"] > 0
    assert moving.get("cpp_named_raw_payload_plan_fused_table_failures", 0) == 0
    assert moving["cpp_moving_environment_family_mpo_descriptor_records"] > 0
    assert moving["cpp_moving_environment_family_mpo_descriptor_environment_builds"] == 0
    assert moving["cpp_moving_environment_family_mpo_descriptor_payload_builds"] > 0
    assert moving["cpp_moving_environment_family_mpo_descriptor_failures"] == 0
    assert moving["owner_direct_family_environment_builds"] == 0
    assert moving["cpp_moving_environment_direct_family_payload_builder_records"] == 0
    assert moving["cpp_grouped_renormalized_table_builds"] == 0
    assert moving.get("cpp_grouped_renormalized_table_raw_builder_builds", 0) == 0
    assert moving["cpp_moving_environment_grouped_table_installs"] > 0
    assert moving["cpp_moving_environment_grouped_table_records"] > 0
    assert moving["cpp_grouped_renormalized_table_refreshes"] > 0
    assert moving["cpp_grouped_renormalized_table_fast_refreshes"] > 0
    assert (
        moving.get("cpp_grouped_renormalized_table_rebuild_in_place_refreshes", 0)
        == 0
    )
    assert moving["cpp_grouped_renormalized_table_rebuild_refreshes"] == 0
    assert moving["cpp_grouped_renormalized_table_last_refresh_kind"] in {
        "raw_dense_in_place",
        "raw_dense_schedule_in_place",
    }
    assert moving.get("cpp_grouped_renormalized_table_raw_schedule_hits", 0) > 0
    assert moving["cpp_moving_environment_enabled"] is True
    assert moving["cpp_moving_environment_compact_plan_records"] == 0
    assert moving["compact_plan_builds"] == 0
    assert moving["compact_renormalized_table_cpp_block_constructor_builds"] == 0
    assert moving["compact_renormalized_table_python_stack_constructor_builds"] == 0
    assert moving["compact_renormalized_table_cpp_block_refreshes"] == 0
    assert moving["compact_renormalized_table_python_stack_refreshes"] == 0

    svd_stats = getattr(abelian_direct, "_ABELIAN_SVD_KERNEL_STATS", {})
    assert svd_stats.get("cpp_site_merge_calls", 0) > 0
    assert svd_stats.get("site_update_wrap_calls", 0) > 0
    assert (
        svd_stats.get("cpp_flat_split_update_calls", 0)
        + svd_stats.get("cpp_split_update_calls", 0)
    ) > 0
    assert svd_stats.get("numpy_calls", 0) == 0
    assert getattr(abelian_direct, "_ABELIAN_SVD_KERNEL_LAST_ERROR", "") == ""


def test_h4_spatial_block2_like_default_uses_generator_table_owner_refresh():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
    )

    dmrg.run(nsweeps=6)

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=5e-4)
    active = dmrg._active_integral_build_info
    assert active["representation"] == "spatial_block2_table_carrier_mpo"
    assert active["spatial_family_environment_backend"] == "generator_table"
    assert active["complementary_operator_families"]["enable_native_boundary_r"] is False
    assert "symmetric_family_convert_s" not in active["build_timings"]
    family_mpos = active["complementary_operator_family_mpos"]
    assert family_mpos["R"]["source"] == "native_generator_entries"
    assert family_mpos["P"]["source"] == "native_generator_entries"

    moving = dmrg.dmrg.environment_profile["moving_environment"]
    builder_stats = dmrg.dmrg.complementary_split_stats["direct_family_table_builder"]
    build_options = builder_stats["contextual_build_options"]
    assert build_options["exact_planned_packed_contextual_entries"] is True
    assert build_options["table_backed_planned_contextual_entries"] is False
    assert build_options["table_backed_planned_contextual_entries_policy"] == "False"
    assert builder_stats["native_boundary_r"]["enabled"] is False
    assert moving["owner_local_optimize_commit_actual"] == "cpp_owner_site_chain"
    assert moving["owner_site_chain_backend_actual"] == "cpp_owner_site_chain"
    assert moving["owner_site_chain_gets"] > 0
    assert moving["owner_site_chain_sets"] > 0
    assert moving["owner_site_chain_syncs"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_records"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_gets"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_sets"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_syncs"] > 0
    assert moving["cpp_moving_environment_owner_site_chain_failures"] == 0
    assert moving["owner_half_sweep_backend_actual"] == (
        "cpp_owner_half_sweep_typed_plan"
    )
    assert moving["owner_bond_step_orchestrator_actual"] == (
        "cpp_moving_environment_typed"
    )
    assert moving["cpp_moving_environment_owner_typed_half_sweep_plan_hits"] > 0
    assert moving["cpp_moving_environment_owner_typed_bond_step_record_hits"] > 0
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_python_prepare_calls"
        ]
        == 0
    )
    assert (
        moving["cpp_moving_environment_owner_typed_bond_step_python_move_calls"]
        == 0
    )
    assert moving["owner_direct_family_revision_state_backend_actual"] == (
        "cpp_moving_environment"
    )
    assert moving["owner_direct_family_cache_key_backend_actual"] == (
        "cpp_moving_environment"
    )
    assert moving["owner_direct_family_cpp_key_bundle_backend_actual"] == (
        "cpp_moving_environment"
    )
    assert moving["cpp_moving_environment_direct_family_revision_state_updates"] > 0
    assert moving["cpp_moving_environment_direct_family_revision_state_failures"] == 0
    assert moving["cpp_moving_environment_direct_family_revision_cache_key_builds"] > 0
    assert (
        moving["cpp_moving_environment_direct_family_revision_cache_key_failures"]
        == 0
    )
    assert moving["cpp_moving_environment_direct_family_cpp_key_bundle_builds"] > 0
    assert moving["cpp_moving_environment_direct_family_cpp_key_bundle_failures"] == 0
    assert moving["owner_typed_direct_plan_static_refresh_attempts"] > 0
    assert (
        moving["owner_typed_direct_plan_static_refresh_accepts"]
        == moving["owner_typed_direct_plan_static_refresh_attempts"]
    )
    assert moving["owner_typed_direct_plan_static_refresh_fallbacks"] == 0
    assert moving["owner_typed_direct_plan_static_refresh_failures"] == 0
    assert moving["owner_typed_direct_plan_static_refresh_backend_actual"] == (
        "cpp_owner_typed_bond_step_static_direct_keys"
    )
    assert moving["owner_typed_direct_plan_static_refresh_orchestrator_actual"] == (
        "cpp_owner_typed_bond_step_successor_chain"
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_updates"
        ]
        == moving["owner_typed_direct_plan_static_refresh_attempts"]
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_update_misses"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_update_failures"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_calls"
        ]
        == moving["owner_typed_direct_plan_static_refresh_attempts"]
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_accepts"
        ]
        == moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_calls"
        ]
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_failures"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_calls"
        ]
        > 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_calls"
        ]
        > 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_accepts"
        ]
        == moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_calls"
        ]
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_links"
        ]
        > 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_failures"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_accepts"
        ]
        == moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_calls"
        ]
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_failures"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_calls"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_record_installs"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_accepts"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_failures"
        ]
        == 0
    )
    assert moving["cpp_moving_environment_owner_bond_step_runner_assign_calls"] == 0
    assert moving["cpp_moving_environment_owner_bond_step_runner_assign_skips"] > 0
    assert (
        moving[
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_prepare_calls"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_calls"
        ]
        == 0
    )
    assert moving["owner_direct_family_environment_payload_owner"] == (
        "cpp_moving_environment_static_direct_family_payload"
    )
    assert moving["cpp_moving_environment_direct_family_payload_assembler_calls"] > 0
    assert moving["cpp_moving_environment_direct_family_payload_assembler_builds"] > 0
    assert moving["cpp_moving_environment_direct_family_payload_records"] > 0
    assert moving["owner_local_optimize_direct_payload_key_hits"] > 0
    assert (
        moving[
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_installs"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_uses"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_factory_calls"
        ]
        == 0
    )
    assert (
        moving[
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_failures"
        ]
        == 0
    )
    assert moving["cpp_moving_environment_direct_family_payload_builder_builds"] == 0
    assert moving["owner_direct_family_environment_builds"] == 0
    assert moving["owner_local_grouped_cpp_table_refresh_attempts"] > 0
    assert (
        moving.get("owner_local_grouped_cpp_table_refresh_accepts", 0)
        == moving["owner_local_grouped_cpp_table_refresh_attempts"]
    )
    assert (
        moving.get("owner_local_grouped_cpp_table_refresh_fallbacks", 0)
        == 0
    )
    assert moving["owner_local_grouped_cpp_table_refresh_backend_actual"] == (
        "cpp_direct_family_route_plan_table_record"
    )
    assert moving["owner_local_grouped_cpp_table_refresh_fallback_actual"] == "none"
    assert moving["cpp_moving_environment_direct_family_route_plan_records"] > 0
    assert moving["cpp_moving_environment_direct_family_route_plan_record_hits"] > 0
    assert moving["cpp_moving_environment_direct_family_route_plan_payload_builds"] > 0
    assert moving["cpp_moving_environment_direct_route_plan_build_successes"] > 0
    assert moving["cpp_moving_environment_direct_route_plan_build_failures"] == 0
    assert moving["cpp_moving_environment_direct_route_plan_build_records"] > 0
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_calls"] > 0
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_table_backed"] == 0
    assert moving["cpp_direct_family_route_plan_last_direct_table_execution_plan_builds"] > 0
    assert moving["cpp_direct_family_route_plan_last_direct_table_execution_plan_planned_steps"] > 0
    assert moving["cpp_direct_family_route_plan_last_direct_table_execution_plan_packed_steps"] > 0
    assert (
        moving[
            "cpp_direct_family_route_plan_last_direct_table_execution_plan_unsupported_fallbacks"
        ]
        == 0
    )
    assert moving["cpp_grouped_renormalized_table_fast_refreshes"] > 0
    assert (
        moving["cpp_grouped_renormalized_table_last_refresh_kind"]
        == "direct_route_plan_table_in_place"
    )
    assert (
        moving.get("cpp_grouped_renormalized_table_direct_table_schedule_hits", 0)
        > 0
    )
    assert (
        moving.get("cpp_grouped_renormalized_table_direct_table_schedule_stores", 0)
        > 0
    )
    assert moving.get("cpp_grouped_renormalized_table_raw_schedule_hits", 0) == 0
    assert moving.get("cpp_grouped_renormalized_table_raw_schedule_stores", 0) == 0
    assert moving.get("family_environment_cpp_descriptor_installs", 0) == 0
    assert moving["cpp_moving_environment_family_mpo_descriptor_records"] == 0


def test_h4_spatial_native_boundary_r_is_guarded_with_native_p():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        spatial_enable_native_boundary_r=True,
        init_guess="hf",
    )

    dmrg.run(nsweeps=6)

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=5e-4)
    builder_stats = dmrg.dmrg.complementary_split_stats["direct_family_table_builder"]
    native_r = builder_stats["native_boundary_r"]
    assert native_r["enabled"] is False
    assert native_r["native_p_enabled"] is True
    assert "cross-family route validation" in native_r["reason"]


def test_h4_spatial_packed_contextual_entries_guard_planned_table_path():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_packed_direct_family_entries": True,
            "generator_table_packed_route_table": "raw_cython_validate",
            "generator_table_cpp_direct_family_owner_payload": False,
            "generator_table_allow_planned_packed_contextual_entries": False,
        },
    )

    dmrg.run(nsweeps=2)

    moving = dmrg.dmrg.environment_profile["moving_environment"]
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    build_options = builder_stats["contextual_build_options"]
    assert build_options["packed_direct_family_entries"] is True
    assert build_options["planned_packed_contextual_guard"] is True
    assert build_options["precompute_boundaries"] is False
    assert build_options["planned_without_precompute"] is False
    assert moving["packed_route_table_backend_actual"] in {
        "raw_cython",
        "planned_raw_cython",
    }
    assert moving["packed_route_table_fallback_reason"] == ""
    assert moving["packed_route_validate_max_abs"] < 1e-10
    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)


def test_h4_spatial_planned_contextual_entries_use_exact_expanded_route():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_packed_direct_family_entries": True,
            "generator_table_packed_route_table": "raw_cython_validate",
            "generator_table_cpp_direct_family_owner_payload": False,
            "generator_table_allow_planned_packed_contextual_entries": True,
            "generator_table_allow_table_backed_planned_contextual_entries": False,
        },
    )

    dmrg.run(nsweeps=2)

    moving = dmrg.dmrg.environment_profile["moving_environment"]
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    build_options = builder_stats["contextual_build_options"]
    assert build_options["packed_direct_family_entries"] is True
    assert build_options["planned_packed_contextual_guard"] is False
    assert build_options["exact_planned_packed_contextual_entries"] is True
    assert build_options["table_backed_planned_contextual_entries"] is False
    assert build_options["precompute_boundaries"] is False
    assert build_options["planned_without_precompute"] is True
    assert moving["packed_route_table_backend_actual"] == "planned_raw_cython"
    assert moving["packed_planned_route_cpp_used"] is True
    assert moving["packed_route_table_fallback_reason"] == ""
    assert moving["packed_route_validate_max_abs"] < 1e-10
    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)


def test_h4_spatial_unvalidated_contextual_batch_is_fully_checked():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_reference_contextual_boundary_batch": False,
            "generator_table_allow_unvalidated_contextual_boundary_batch": True,
            "generator_table_validate_contextual_boundary_batch": True,
            "generator_table_packed_route_table": "raw_cython",
        },
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["reference_batch"] is False
    assert plan["allow_unvalidated_batch"] is True
    assert plan["validation_enabled"] is True
    assert plan["validation_limit"] == -1
    assert plan["cold_validation"] is False
    assert plan["generic_identity_advance"] is False
    assert plan["refresh_cached_boundaries"] is True
    validation = builder_stats["contextual_boundary_batch_validation"]
    assert validation["left"]["checked"] > 0
    assert validation["right"]["checked"] > 0
    assert validation["left"].get("mismatches", 0) == 0
    assert validation["right"].get("mismatches", 0) == 0


def test_h4_spatial_unvalidated_contextual_batch_requires_full_validation():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_reference_contextual_boundary_batch": False,
            "generator_table_allow_unvalidated_contextual_boundary_batch": True,
            "generator_table_validate_contextual_boundary_batch": False,
            "generator_table_use_cpp_contextual_boundary_batch_plan": False,
            "generator_table_packed_route_table": "raw_cython",
        },
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["reference_batch"] is True
    assert plan["allow_unvalidated_batch"] is True
    assert plan["validation_enabled"] is False
    assert plan["full_warm_validation"] is False
    assert plan["unsafe_unvalidated_batch"] is False
    assert plan["reference_batch_override_reason"] == (
        "non_reference_contextual_batch_requires_full_warm_validation"
    )


def test_h4_spatial_unsafe_contextual_batch_uses_exact_planned_entries():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_reference_contextual_boundary_batch": False,
            "generator_table_allow_unvalidated_contextual_boundary_batch": True,
            "generator_table_allow_unsafe_unvalidated_contextual_boundary_batch": True,
            "generator_table_validate_contextual_boundary_batch": False,
            "generator_table_use_cpp_contextual_boundary_batch_plan": False,
            "generator_table_packed_route_table": "raw_cython",
        },
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["reference_batch"] is False
    assert plan["unsafe_unvalidated_batch"] is True
    build_options = builder_stats["contextual_build_options"]
    assert build_options["table_backed_planned_contextual_entries"] is False
    assert build_options["exact_planned_packed_contextual_entries"] is True
    assert build_options["table_backed_planned_contextual_guard_reason"] == (
        "table_backed_contextual_entries_require_reference_or_full_warm_validation"
    )


def test_h4_spatial_unsafe_table_backed_contextual_batch_prewarms_exact_entries():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_reference_contextual_boundary_batch": False,
            "generator_table_allow_unvalidated_contextual_boundary_batch": True,
            "generator_table_allow_unsafe_unvalidated_contextual_boundary_batch": True,
            "generator_table_validate_contextual_boundary_batch": False,
            "generator_table_packed_route_table": "raw_cython",
            "generator_table_allow_table_backed_planned_contextual_entries": True,
            "generator_table_allow_unsafe_table_backed_planned_contextual_entries": True,
        },
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["reference_batch"] is False
    assert plan["validation_enabled"] is False
    assert plan["prewarm_batch"] is True
    assert builder_stats["contextual_boundary_batch_prewarm"]["left"]["keys"] > 0
    assert builder_stats["contextual_boundary_batch_prewarm"]["right"]["keys"] > 0
    build_options = builder_stats["contextual_build_options"]
    assert build_options["table_backed_planned_contextual_entries"] is True
    assert build_options["unsafe_table_backed_planned_contextual_entries"] is True
    snapshot = builder_stats["contextual_planned_entry_snapshots"]
    assert snapshot["entries"] > 0
    assert builder_stats["contextual_route_fast_pack"]["last_table_backed"] is False


def test_h4_spatial_cpp_contextual_batch_disable_is_guarded():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_cpp_contextual_batch_construction": False,
            "generator_table_reference_contextual_boundary_batch": False,
            "generator_table_allow_unvalidated_contextual_boundary_batch": True,
            "generator_table_allow_unsafe_unvalidated_contextual_boundary_batch": True,
            "generator_table_validate_contextual_boundary_batch": False,
            "generator_table_packed_route_table": "raw_cython",
        },
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    cpp_batch = builder_stats["contextual_cpp_batch_construction"]
    assert cpp_batch["requested"] is False
    assert cpp_batch["effective"] is True
    assert cpp_batch["unsafe_disable"] is False
    assert cpp_batch["override_reason"] == "python_contextual_batch_is_not_exact"
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["cpp_contextual_batch_construction_requested"] is False
    assert plan["cpp_contextual_batch_construction_effective"] is True
    assert (
        plan["cpp_contextual_batch_construction_override_reason"]
        == "python_contextual_batch_is_not_exact"
    )


def test_h4_spatial_cpp_contextual_batch_plan_is_default_exact():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_reference_contextual_boundary_batch": False,
            "generator_table_allow_unvalidated_contextual_boundary_batch": True,
            "generator_table_allow_unsafe_unvalidated_contextual_boundary_batch": True,
            "generator_table_validate_contextual_boundary_batch": False,
            "generator_table_packed_route_table": "raw_cython",
            "generator_table_allow_table_backed_planned_contextual_entries": True,
            "generator_table_allow_unsafe_table_backed_planned_contextual_entries": True,
        },
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["consume_cpp_plan_requested"] is True
    assert plan["consume_cpp_plan_effective"] is True
    assert plan["unsafe_cpp_plan"] is False
    assert plan["left_attached"] is True
    assert plan["right_attached"] is True
    assert "consume_cpp_plan_override_reason" not in plan


def _assert_cpp_contextual_plan_validation_clean(dmrg):
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["consume_cpp_plan_effective"] is True
    assert plan["left_attached"] is True
    assert plan["right_attached"] is True
    lazy = builder_stats["contextual_route_lazy_pack"]
    assert lazy["validate_native_plan_table_ids"] is True
    assert lazy["native_plan_validation_cold"] is True
    side_cache = builder_stats.get("contextual_boundary_side_precompute_cache")
    if side_cache is not None:
        assert side_cache["native_plan_validation_calls"] > 0
        assert side_cache["native_plan_validation_cold_clears"] > 0
        assert side_cache["native_plan_validation_matches"] > 0
        assert side_cache.get("native_plan_validation_mismatches", 0) == 0
        assert side_cache.get("native_plan_validation_fallbacks", 0) == 0
        assert side_cache["native_plan_validation_max_abs"] == pytest.approx(
            0.0, abs=1e-12
        )
    else:
        assert lazy["calls"] > 0
        assert lazy["table_lookup"] is True
        assert lazy["table_ids_only"] is True
        assert lazy["planned_calls"] > 0


def _cpp_contextual_plan_validation_options():
    return {
        "generator_table_validate_contextual_boundary_batch": True,
        "generator_table_contextual_boundary_batch_validation_limit": -1,
    }


def test_h4_spatial_cpp_contextual_batch_plan_validation_matches():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=4,
        nelecas=4,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options=_cpp_contextual_plan_validation_options(),
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-2.14079716446274, abs=1e-9)
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["consume_cpp_plan_requested"] is True
    assert plan["consume_cpp_plan_effective"] is True
    assert plan["unsafe_cpp_plan"] is False
    assert plan["left_attached"] is True
    assert plan["right_attached"] is True
    _assert_cpp_contextual_plan_validation_clean(dmrg)


def _assert_spatial_generator_table_cpp_contextual_route(dmrg):
    active = dmrg._active_integral_build_info
    assert active["spatial_family_environment_backend"] == "generator_table"
    family_mpos = active["complementary_operator_family_mpos"]
    assert family_mpos["R"]["source"] == "native_generator_entries"
    assert family_mpos["P"]["source"] == "native_generator_entries"

    moving = dmrg.dmrg.environment_profile["moving_environment"]
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    build_options = builder_stats["contextual_build_options"]
    assert build_options["packed_direct_family_entries"] is True
    assert build_options["cpp_contextual_batch_construction_requested"] is True
    assert build_options["cpp_contextual_batch_construction_effective"] is True
    assert build_options["cpp_contextual_batch_construction_unsafe_disable"] is False
    assert build_options["table_backed_planned_contextual_entries"] is True
    assert build_options["table_backed_planned_contextual_entries_policy"] == "auto"
    assert build_options["precompute_boundaries"] is False
    assert build_options["cpp_contextual_precompute_disabled"] is True
    assert (
        build_options["cpp_contextual_precompute_disabled_reason"]
        == "attached_cpp_table_backed_contextual_plan_uses_lazy_table_ids"
    )

    cpp_batch = builder_stats["contextual_cpp_batch_construction"]
    assert cpp_batch["requested"] is True
    assert cpp_batch["effective"] is True
    assert cpp_batch["unsafe_disable"] is False
    plan = builder_stats["contextual_boundary_batch_plan"]
    assert plan["cpp_contextual_batch_construction_requested"] is True
    assert plan["cpp_contextual_batch_construction_effective"] is True
    assert plan["cpp_contextual_batch_construction_unsafe_disable"] is False
    assert plan["consume_cpp_plan_requested"] is True
    assert plan["consume_cpp_plan_effective"] is True
    assert plan["unsafe_cpp_plan"] is False
    assert plan["left_attached"] is True
    assert plan["right_attached"] is True
    assert "consume_cpp_plan_override_reason" not in plan

    planned = builder_stats["contextual_planned_entry_cache"]
    assert planned["backend_actual"] == "cpp_moving_environment"
    assert planned["owner_builds"] > 0
    assert planned["owner_entries"] > 0
    assert planned["last_stable_table_key"] is False
    fast_pack = builder_stats["contextual_route_fast_pack"]
    snapshot = builder_stats["contextual_planned_entry_snapshots"]
    assert snapshot["entries"] > 0
    assert fast_pack["last_table_backed"] is False
    assert fast_pack["left_table_ids"] > 0
    assert fast_pack["right_table_ids"] > 0

    assert moving["owner_local_grouped_cpp_table_refresh_attempts"] > 0
    assert (
        moving["owner_local_grouped_cpp_table_refresh_accepts"]
        == moving["owner_local_grouped_cpp_table_refresh_attempts"]
    )
    assert moving.get("owner_local_grouped_cpp_table_refresh_fallbacks", 0) == 0
    assert moving["owner_local_grouped_cpp_table_refresh_backend_actual"] == (
        "cpp_direct_family_route_plan_table_record"
    )
    assert moving["cpp_grouped_renormalized_table_last_refresh_kind"] in {
        "direct_route_plan_table",
        "direct_route_plan_table_in_place",
    }
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_successes"] > 0
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_failures"] == 0


def _assert_spatial_grouped_table_uses_shape_batches(dmrg):
    moving = dmrg.dmrg.environment_profile["moving_environment"]
    assert moving["cpp_moving_environment_grouped_table_dense_matvec_shape_groups"] > 0
    assert moving["cpp_moving_environment_grouped_table_dense_matvec_shape_group_blocks"] > 0
    assert moving["cpp_moving_environment_grouped_table_dense_matvec_singleton_blocks"] > 0


def test_h6_spatial_block2_like_default_uses_planned_contextual_route():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.4; H 0 0 2.8; "
            "H 0 0 4.2; H 0 0 5.6; H 0 0 7.0"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=6,
        nelecas=6,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-3.13740245377822, abs=1e-9)
    _assert_spatial_generator_table_cpp_contextual_route(dmrg)
    _assert_spatial_grouped_table_uses_shape_batches(dmrg)
    active = dmrg._active_integral_build_info
    assert active["spatial_family_environment_backend"] == "generator_table"
    family_mpos = active["complementary_operator_family_mpos"]
    assert family_mpos["R"]["source"] == "native_generator_entries"
    assert family_mpos["P"]["source"] == "native_generator_entries"

    moving = dmrg.dmrg.environment_profile["moving_environment"]
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    build_options = builder_stats["contextual_build_options"]
    assert build_options["packed_direct_family_entries"] is True
    assert build_options["exact_planned_packed_contextual_entries"] is False
    assert build_options["precompute_boundaries"] is False
    assert build_options["planned_without_precompute"] is True
    assert build_options["table_backed_planned_contextual_entries"] is True
    assert build_options["table_backed_planned_contextual_entries_policy"] == "auto"

    planned = builder_stats["contextual_planned_entry_cache"]
    assert planned["backend_actual"] == "cpp_moving_environment"
    assert planned["owner_builds"] > 0
    assert planned["owner_entries"] > 0
    assert planned["last_stable_table_key"] is False
    fast_pack = builder_stats["contextual_route_fast_pack"]
    snapshot = builder_stats["contextual_planned_entry_snapshots"]
    assert snapshot["entries"] > 0
    assert fast_pack["last_table_backed"] is False
    assert fast_pack["left_table_ids"] > 0
    assert fast_pack["right_table_ids"] > 0
    assert moving["owner_local_grouped_cpp_table_refresh_attempts"] > 0
    assert (
        moving["owner_local_grouped_cpp_table_refresh_accepts"]
        == moving["owner_local_grouped_cpp_table_refresh_attempts"]
    )
    assert moving.get("owner_local_grouped_cpp_table_refresh_fallbacks", 0) == 0
    assert moving["owner_local_grouped_cpp_table_refresh_backend_actual"] == (
        "cpp_direct_family_route_plan_table_record"
    )
    assert moving["cpp_grouped_renormalized_table_last_refresh_kind"] in {
        "direct_route_plan_table",
        "direct_route_plan_table_in_place",
    }
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_successes"] > 0
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_failures"] == 0
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_table_backed"] > 0


def test_h6_spatial_cpp_contextual_batch_plan_validation_matches():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.4; H 0 0 2.8; "
            "H 0 0 4.2; H 0 0 5.6; H 0 0 7.0"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=6,
        nelecas=6,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options=_cpp_contextual_plan_validation_options(),
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-3.13740245377822, abs=1e-9)
    _assert_spatial_generator_table_cpp_contextual_route(dmrg)
    _assert_spatial_grouped_table_uses_shape_batches(dmrg)
    _assert_cpp_contextual_plan_validation_clean(dmrg)


def test_h6_spatial_planned_contextual_no_native_p_keeps_route_table():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.4; H 0 0 2.8; "
            "H 0 0 4.2; H 0 0 5.6; H 0 0 7.0"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=6,
        nelecas=6,
        D=16,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options={
            "generator_table_disable_native_boundary_p": True,
        },
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=120,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-3.13707068520193, abs=1e-9)
    _assert_spatial_generator_table_cpp_contextual_route(dmrg)
    _assert_spatial_grouped_table_uses_shape_batches(dmrg)
    moving = dmrg.dmrg.environment_profile["moving_environment"]
    builder_stats = dmrg.dmrg.complementary_split_stats[
        "direct_family_table_builder"
    ]
    native_p = builder_stats["native_boundary_p"]
    assert native_p["enabled"] is False
    assert "disabled" in native_p["reason"]
    build_options = builder_stats["contextual_build_options"]
    assert build_options["exact_planned_packed_contextual_entries"] is False
    assert build_options["table_backed_planned_contextual_entries"] is True
    planned = builder_stats["contextual_planned_entry_cache"]
    assert planned["last_stable_table_key"] is False
    fast_pack = builder_stats["contextual_route_fast_pack"]
    snapshot = builder_stats["contextual_planned_entry_snapshots"]
    assert snapshot["entries"] > 0
    assert fast_pack["last_table_backed"] is False
    assert fast_pack["left_table_ids"] > 0
    assert fast_pack["right_table_ids"] > 0
    assert moving["owner_local_grouped_cpp_table_refresh_attempts"] > 0
    assert (
        moving["owner_local_grouped_cpp_table_refresh_accepts"]
        == moving["owner_local_grouped_cpp_table_refresh_attempts"]
    )
    assert moving.get("owner_local_grouped_cpp_table_refresh_fallbacks", 0) == 0
    assert moving["owner_local_grouped_cpp_table_refresh_backend_actual"] == (
        "cpp_direct_family_route_plan_table_record"
    )
    assert moving["cpp_grouped_renormalized_table_last_refresh_kind"] in {
        "direct_route_plan_table",
        "direct_route_plan_table_in_place",
    }
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_successes"] > 0
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_failures"] == 0
    assert moving["cpp_moving_environment_planned_direct_family_entry_build_table_backed"] > 0


def test_h8_spatial_block2_like_default_uses_cpp_contextual_route():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2; "
            "H 0 0 5.6; H 0 0 7.0; H 0 0 8.4; H 0 0 9.8"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=8,
        nelecas=8,
        D=8,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-8,
        davidson_max_iter=80,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-4.10718881122356, abs=1e-9)
    _assert_spatial_generator_table_cpp_contextual_route(dmrg)
    _assert_spatial_grouped_table_uses_shape_batches(dmrg)


def test_h8_spatial_cpp_contextual_batch_plan_validation_matches():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2; "
            "H 0 0 5.6; H 0 0 7.0; H 0 0 8.4; H 0 0 9.8"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=8,
        nelecas=8,
        D=8,
        site="spatial",
        symmetry="sz",
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        init_guess="hf",
        abelian_matvec_options=_cpp_contextual_plan_validation_options(),
    )

    dmrg.run(
        nsweeps=2,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-8,
        davidson_max_iter=80,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-4.10718881122356, abs=1e-9)
    _assert_spatial_generator_table_cpp_contextual_route(dmrg)
    _assert_spatial_grouped_table_uses_shape_batches(dmrg)
    _assert_cpp_contextual_plan_validation_clean(dmrg)


def test_h6_spatial_block2_like_folds_recenter_into_cpp_schedule():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.4; H 0 0 2.8; "
            "H 0 0 4.2; H 0 0 5.6; H 0 0 7.0"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = mol.RHF(verbose=0).run()
    dmrg = DMRG(
        mf,
        ncas=6,
        nelecas=6,
        D=16,
        site="spatial",
        symmetry=("charge", "sz"),
        spin=0,
        verbose=0,
        integral_backend="cholesky",
        dmrg_performance="block2-like",
        spatial_family_environment_backend="block2_table",
    )

    dmrg.run(
        nsweeps=4,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=200,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-3.14343934127309, abs=1e-9)
    moving = dmrg.dmrg.environment_profile["moving_environment"]
    assert moving["owner_sweep_schedule_backend_actual"] == (
        "cpp_owner_sweep_schedule_plan"
    )
    assert moving["owner_sweep_schedule_builder_actual"] == (
        "cpp_alternating_sweep_schedule_plan_final_recenter"
    )
    assert moving["owner_half_sweep_backend_actual"] == (
        "cpp_owner_sweep_schedule_plan"
    )
    assert moving["owner_bond_step_orchestrator_actual"] == (
        "cpp_moving_environment_sweep_schedule"
    )
    assert moving["owner_bond_step_backend_actual"] == (
        "cpp_owner_sweep_schedule_plan"
    )
    assert (
        moving[
            "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_configures"
        ]
        == 1
    )
    assert (
        moving[
            "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_runs"
        ]
        == 1
    )
    assert (
        moving[
            "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_skips"
        ]
        == 0
    )
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_halves"] == 5
    assert (
        moving["cpp_moving_environment_owner_sweep_schedule_plan_history_rows"]
        == len(dmrg.dmrg.sweep_history)
        == 5
    )
    assert [row["direction"] for row in dmrg.dmrg.sweep_history] == [
        "lr",
        "rl",
        "lr",
        "rl",
        "recenter-right",
    ]
    assert moving["cpp_moving_environment_owner_local_problem_bind_owner_calls"] > 0
    assert (
        moving[
            "cpp_moving_environment_owner_local_problem_bind_set_bond_fallbacks"
        ]
        == 0
    )
    assert moving["local_operator_builds"] == 0
    assert moving["cpp_named_raw_payload_plan_backend_actual"] == (
        "cpp_family_mpo_descriptor_owner_table_refresh"
    )
    assert moving["cpp_named_raw_payload_plan_fused_table_refreshes"] > 0
    assert moving.get("cpp_named_raw_payload_plan_fused_table_failures", 0) == 0
    assert moving["cpp_named_raw_payload_plan_last_left_stack_same_refresh_hits"] > 0
    assert moving["cpp_named_raw_payload_plan_last_right_stack_same_refresh_hits"] > 0
    assert moving["cpp_grouped_renormalized_table_fast_refreshes"] > 0
    assert (
        moving.get("cpp_grouped_renormalized_table_rebuild_in_place_refreshes", 0)
        == 0
    )
    assert moving["cpp_grouped_renormalized_table_rebuild_refreshes"] == 0
    assert moving["cpp_grouped_renormalized_table_last_refresh_kind"] in {
        "raw_dense_in_place",
        "raw_dense_schedule_in_place",
    }
    assert moving.get("cpp_grouped_renormalized_table_raw_schedule_hits", 0) > 0


def test_spatial_dmrg_routes_su2_to_nonabelian_backend():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dense = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dense.run(nsweeps=4, symmetry_list=None)
    su2 = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        verbose=0,
    )
    su2.run(nsweeps=4)

    assert su2.dmrg.backend == "nonabelian"
    assert su2.symmetry == ["charge", "su2"]
    assert su2.site == "spatial"
    assert su2.spatial_reduced_mpo is True
    assert su2._active_integral_build_info["representation"] == "spatial_reduced_spinfree_mpo"
    assert su2.dmrg.history[-1]["hamiltonian_system"]["n_sites"] == 2
    assert su2.dmrg.history[-1]["hamiltonian_system"]["n_elec"] == 2
    assert su2.dmrg.history[-1]["hamiltonian_symmetry"] == "su2"
    assert su2.dmrg.history[-1]["local_basis_policy"] == "orthonormalized_operator"
    assert su2.dmrg.history[-1]["max_bond_mode"] == "per_sector"
    assert su2.e_tot == pytest.approx(dense.e_tot, abs=1e-7)
    numerator = contract_chain_expectation(su2.dmrg.ground_state.sites, su2.H)
    denominator = contract_chain_expectation(
        su2.dmrg.ground_state.sites,
        _identity_mpo_factors_for_sites_and_mpo(su2.dmrg.ground_state.sites, su2.H),
    )
    assert np.real(numerator / denominator) == pytest.approx(su2.e_tot - su2.e_core, abs=1e-10)

    su2_canonical = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    su2_canonical.run(nsweeps=4, symmetry="su2", canonical_local_norm=True)
    assert su2_canonical.e_tot == pytest.approx(dense.e_tot, abs=1e-7)


def test_spatial_dmrg_symmetry_argument_replaces_backend_selector():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    abelian = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", symmetry="sz", verbose=0)
    abelian.run(nsweeps=4)
    legacy = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    legacy.run(nsweeps=4, symmetry_list=["charge", "sz"])

    assert abelian.symmetry == ["charge", "sz"]
    assert not hasattr(abelian.dmrg, "backend")
    assert abelian.e_tot == pytest.approx(legacy.e_tot, abs=1e-10)


def test_abelian_dmrg_target_uses_explicit_spin_not_molecule_spin():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        site="spatial",
        symmetry="sz",
        spin=2,
        verbose=0,
    )
    dmrg.run(nsweeps=2)

    assert dmrg.mf.mol.spin == 0
    assert dmrg.spin == 2
    assert dmrg.dmrg.target_qn == AbelianSector(("charge", "sz"), (2, 2))


def test_su2_dmrg_target_uses_explicit_charge_and_total_spin():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        spin=2,
        verbose=0,
    )
    dmrg.run(nsweeps=2)

    assert dmrg.dmrg.target_sector == SpinChargeSector(2, SU2Irrep(2))
    assert dmrg.dmrg.ground_state.target_sector == SpinChargeSector(2, SU2Irrep(2))


def test_su2_ground_state_default_trusts_mixed_canonical_norm():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", symmetry="su2", verbose=0)
    dmrg.run(nsweeps=2)

    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    assert objectives
    assert all(objective.get("local_basis_policy") == "mixed_canonical_standard" for objective in objectives)
    assert all(objective.get("canonical_norm_used") is True for objective in objectives)
    assert all(objective.get("effective_local_problem") == "standard" for objective in objectives)
    assert all(objective.get("block_preconditioner") is True for objective in objectives)
    assert all(objective.get("block_preconditioner_blocks", 0) > 0 for objective in objectives)
    assert all(
        entry.get("norm_renormalized_block_stack_stats") is None
        for entry in dmrg.dmrg.history
    )


def test_su2_ground_state_can_still_check_local_norm_debug_path():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    fast = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", symmetry="su2", verbose=0)
    fast.run(nsweeps=2)
    checked = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", symmetry="su2", verbose=0)
    checked.run(nsweeps=2, canonical_local_norm=False)

    objectives = [
        objective
        for entry in checked.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    assert objectives
    assert checked.e_tot == pytest.approx(fast.e_tot, abs=1e-10)
    assert any(
        entry.get("norm_renormalized_block_stack_stats") is not None
        for entry in checked.dmrg.history
    )


def test_su2_ground_state_metric_bonds_use_orthonormal_standard_krylov():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=24, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(nsweeps=2, max_bond_mode="per_sector", mixer_zero_block_noise_scale=0.0)

    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    metric_objectives = [
        objective
        for objective in objectives
        if objective.get("metric_orthonormal_krylov")
    ]
    assert metric_objectives
    assert all(
        objective["effective_local_problem"] == "orthonormalized_standard"
        for objective in metric_objectives
    )
    assert all(
        objective["projected_problem"] == "standard"
        for objective in metric_objectives
    )


def test_su2_block2_policy_uses_orthonormalized_operator_davidson():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=2,
        local_basis_policy="block2_like",
        orthonormalized_operator_dim=512,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
    )
    assert dmrg.e_tot == pytest.approx(-2.177899323464, abs=1e-8)

    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    block2_objectives = [
        objective
        for objective in objectives
        if objective.get("effective_local_problem") == "orthonormalized_operator_standard"
    ]
    assert block2_objectives
    assert all(objective.get("local_basis_policy") == "orthonormalized_operator" for objective in objectives)
    assert all(objective.get("block_davidson") is True for objective in block2_objectives)
    assert all(objective.get("orthonormalized_dim", 0) > 0 for objective in block2_objectives)
    assert max(objective.get("residual", 0.0) for objective in block2_objectives) < 1.0e-7
    storages = [
        objective.get("renormalized_operator_storage")
        for objective in block2_objectives
    ]
    assert "block_sparse_environment_sweep:component_sparse_operator_table" in storages
    assert all(
        storage in {
            "block_sparse_environment_sweep",
            "block_sparse_environment_sweep:block_sparse_operator_table",
            "block_sparse_environment_sweep:component_sparse_operator_table",
        }
        for storage in storages
    )
    table_stats = [
        objective.get("renormalized_operator_table_stats")
        for objective in block2_objectives
        if objective.get("renormalized_operator_table_stats") is not None
    ]
    assert table_stats
    assert all(stats and stats["kind"] in {"block_sparse", "component_sparse"} for stats in table_stats)
    assert all(stats["n_blocks"] > 0 for stats in table_stats)
    assert all(stats["n_nonzero_block_terms"] > 0 for stats in table_stats)
    assert all(stats["parent_dim"] >= stats["orthonormal_dim"] > 0 for stats in table_stats)
    component_stats = [stats for stats in table_stats if stats["kind"] == "component_sparse"]
    if component_stats:
        assert any(
            stats.get("component_parent_block_kernel")
            or stats.get("complementary_payload_tensor_kernel")
            or stats.get("complementary_family_table_kernel")
            for stats in component_stats
        )
        assert all(stats["basis_kind"] == "metric_connected_components" for stats in component_stats)
        assert all(stats["n_components"] > 0 for stats in component_stats)
        assert all(stats["max_component_parent_dim"] > 0 for stats in component_stats)
    metadata = [
        objective.get("renormalized_operator_metadata")
        for objective in block2_objectives
    ]
    assert all(item["renormalized_boundary_source"] == "block_stack" for item in metadata)
    assert all(item["left_boundary"]["stored_elements"] > 0 for item in metadata)
    assert all(item["right_boundary"]["stored_elements"] > 0 for item in metadata)
    assert all(
        item["symbolic_boundary_payloads"]["symbolic_boundary_payload_source"]
        == "symbolic_table"
        for item in metadata
    )
    assert all(
        item["symbolic_boundary_payloads"]["symbolic_payloads_owned"] is True
        for item in metadata
    )
    assert all(
        item["renormalized_local_operator_table"]["representation"]
        == "rank_coupled_complementary"
        for item in metadata
    )
    qchem_plans = [
        item.get("su2_qchem_sweep_plan")
        for item in metadata
        if item.get("su2_qchem_sweep_plan") is not None
    ]
    assert qchem_plans
    assert all(plan["supported"] is True for plan in qchem_plans)
    assert all(plan["left_factor_table"]["n_entries"] > 0 for plan in qchem_plans)
    assert all(plan["right_factor_table"]["n_entries"] > 0 for plan in qchem_plans)
    assert all(plan["factor_match_cache_size"] > 0 for plan in qchem_plans)
    assert all(
        "factor_match_cache_stats" in plan
        for plan in qchem_plans
    )
    assert all(
        item["complementary_operator_families"]["family_names"]
        == ("S", "R", "A", "P", "B", "Q")
        for item in metadata
    )
    assert any(
        (
            item.get("symbolic_boundary_payloads", {})
            .get("complementary_boundary_payloads", {})
            .get("payload_backed")
            is True
        )
        for item in metadata
    )
    assert dmrg.dmrg.history[-1]["renormalized_operator_cache_size"] > 0
    assert set(dmrg.dmrg.history[-1]["renormalized_operator_table_kinds"]) <= {
        "block_sparse",
        "component_sparse",
    }
    block_stack_stats = dmrg.dmrg.history[-1]["renormalized_block_stack_stats"]
    assert block_stack_stats["left_size"] > 0
    assert block_stack_stats["right_size"] > 0
    assert block_stack_stats["puts"] > 0
    assert block_stack_stats["hits"] > 0
    assert block_stack_stats["initialized_entries"] > 0
    assert block_stack_stats["advanced_entries"] > 0
    assert block_stack_stats["side_operator_tables"] > 0
    assert block_stack_stats["side_operator_table_puts"] > 0
    assert block_stack_stats["symbolic_operator_tables"] > 0
    assert block_stack_stats["symbolic_operator_terms"] > 0
    assert block_stack_stats["symbolic_operator_numeric_payloads"] > 0
    assert block_stack_stats["symbolic_operator_max_path_length"] > 0
    assert any(
        str(source).startswith("symbolic_")
        for source in block_stack_stats["side_operator_table_sources"]
    )
    assert any(
        str(source).endswith("prepared")
        for source in block_stack_stats["side_operator_table_sources"]
    )
    assert all(
        "lazy" not in str(source)
        for source in block_stack_stats["side_operator_table_sources"]
    )
    assert any(
        "advanced" in str(source)
        for source in block_stack_stats["side_operator_table_sources"]
    )
    assert "rank_coupled_left_factor_by_ket" in block_stack_stats[
        "side_operator_table_representations"
    ]
    assert "rank_coupled_right_factor_by_ket" in block_stack_stats[
        "side_operator_table_representations"
    ]


def test_su2_block2_complementary_direct_projection_is_opt_in():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.build()
    families = dmrg._active_hamiltonian.complementary_operators
    object.__setattr__(families, "prefer_recursive_operator_matvec", False)
    object.__setattr__(families, "prefer_direct_orthonormal_projection", True)
    try:
        dmrg.run(
            nsweeps=1,
            local_basis_policy="block2_like",
            orthonormalized_operator_dim=512,
            max_bond_mode="per_sector",
            mixer_zero_block_noise_scale=0.0,
            direct_orthonormal_dense_max_elements=0,
            profile=True,
        )
    finally:
        object.__setattr__(families, "prefer_direct_orthonormal_projection", False)
        object.__setattr__(families, "prefer_recursive_operator_matvec", True)

    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    table_stats = [
        objective.get("renormalized_operator_table_stats")
        for objective in objectives
        if objective.get("renormalized_operator_table_stats") is not None
    ]
    assert any(stats.get("component_direct_kernel") is True for stats in table_stats)
    assert any(
        timing.get("component_direct_factorized_kernel", 0.0) > 0.0
        or timing.get("component_recursive_parent_block_kernel", 0.0) > 0.0
        or timing.get("component_complementary_payload_tensor_kernel", 0.0) > 0.0
        or timing.get("component_complementary_family_table_kernel", 0.0) > 0.0
        for timing in (
            objective.get("renormalized_operator_build_timing") or {}
            for objective in objectives
        )
    )
    assert np.isfinite(float(dmrg.e_tot))


def test_su2_kernel_backend_python_fallback_records_reference_path():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=1,
        local_basis_policy="block2_like",
        su2_kernel_backend="python",
    )

    assert dmrg.dmrg.history[-1]["su2_kernel_policy"]["backend"] == "python"
    assert dmrg.dmrg.history[-1]["su2_kernel_backend_actual"] == "python"
    assert np.isfinite(float(dmrg.e_tot))


def test_su2_kernel_backend_cython_requires_extension():
    previous = configure_su2_kernel_policy()
    try:
        if su2_cython_available():
            configure_su2_kernel_policy(backend="cython")
        else:
            with pytest.raises(RuntimeError, match="_su2_kernel"):
                configure_su2_kernel_policy(backend="cython")
    finally:
        configure_su2_kernel_policy(
            backend=previous["backend"],
            debug_check=previous["debug_check"],
            debug_check_tol=previous["debug_check_tol"],
        )


def test_su2_block2_recursive_operator_matvec_avoids_transformed_kernel_build():
    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.6; H 0 0 3.2; "
            "H 0 0 4.8; H 0 0 6.4; H 0 0 8.0"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=6, nelecas=6, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.build()
    families = dmrg._active_hamiltonian.complementary_operators
    object.__setattr__(families, "prefer_recursive_operator_matvec", True)
    object.__setattr__(families, "prefer_complementary_payload_tensor_matvec", True)
    try:
        dmrg.run(
            nsweeps=1,
            local_basis_policy="block2_like",
            orthonormalized_operator_dim=512,
            max_bond_mode="per_sector",
            mixer_zero_block_noise_scale=0.0,
            direct_orthonormal_dense_max_elements=0,
            su2_kernel_backend="auto",
            debug_su2_kernel_check=True,
            profile=True,
        )
    finally:
        object.__setattr__(families, "prefer_recursive_operator_matvec", True)
        object.__setattr__(families, "prefer_complementary_payload_tensor_matvec", True)

    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    timings = [
        objective.get("renormalized_operator_build_timing") or {}
        for objective in objectives
    ]
    assert any(
        timing.get("component_recursive_operator_matvec_preferred", 0.0) > 0.0
        for timing in timings
    )
    assert any(
        timing.get("component_recursive_parent_block_kernel", 0.0) > 0.0
        or timing.get("component_complementary_payload_tensor_kernel", 0.0) > 0.0
        or timing.get("component_complementary_family_table_kernel", 0.0) > 0.0
        for timing in timings
    )
    assert all(
        timing.get("component_factorized_kernel_materialize", 0.0) == 0.0
        for timing in timings
    )
    assert all(
        timing.get("component_factorized_kernel_transform", 0.0) == 0.0
        for timing in timings
    )
    assert any(
        (
            (objective.get("renormalized_operator_table_stats") or {}).get(
                "component_parent_block_kernel"
            )
            is True
            or (objective.get("renormalized_operator_table_stats") or {}).get(
                "complementary_payload_tensor_kernel"
            )
            is True
            or (objective.get("renormalized_operator_table_stats") or {}).get(
                "complementary_family_table_kernel"
            )
            is True
        )
        for objective in objectives
    )
    assert all(
        (
            objective.get("renormalized_operator_table_stats") or {}
        ).get("component_orthonormal_dense_elements", 0)
        == 0
        for objective in objectives
    )
    assert dmrg.dmrg.history[-1]["direct_factorized_orthonormal_kernel_policy"][
        "orthonormal_dense_max_elements"
    ] == 0
    actual_backend = dmrg.dmrg.history[-1]["su2_kernel_backend_actual"]
    assert actual_backend in {"python", "cython"}
    native_actions = [
        (objective.get("renormalized_operator_table_stats") or {}).get(
            "su2_local_action"
        )
        for objective in objectives
    ]
    if actual_backend == "cython":
        assert any(native_actions)
        assert all(
            (
                (objective.get("renormalized_operator_table_stats") or {}).get(
                    "su2_reference_residual"
                )
                or 0.0
            )
            <= 1.0e-10
            for objective in objectives
        )
    else:
        assert not any(native_actions)
    assert any(
        (objective.get("renormalized_operator_table_stats") or {}).get(
            "complementary_direct_matvec"
        )
        is True
        for objective in objectives
    )
    assert any(
        (
            (
                objective.get("renormalized_operator_table_stats") or {}
            ).get("complementary_operator_families")
            or {}
        ).get("family_names")
        == ("S", "R", "A", "P", "B", "Q")
        for objective in objectives
    )
    assert any(
        (objective.get("renormalized_operator_table_stats") or {}).get(
            "complementary_payload_backed"
        )
        is True
        for objective in objectives
    )
    assert any(
        set(
            (objective.get("renormalized_operator_table_stats") or {}).get(
                "family_names",
                (),
            )
        )
        >= {"P", "Q", "R"}
        for objective in objectives
    )
    assert any(
        {
            name
            for name, count in (
                (objective.get("renormalized_operator_table_stats") or {})
                .get("family_term_counts", {})
                .items()
            )
            if count > 0
        }
        >= {"P", "Q", "R"}
        for objective in objectives
    )
    assert any(
        (objective.get("renormalized_operator_table_stats") or {}).get(
            "complementary_payload_terms",
            0,
        )
        > 0
        for objective in objectives
    )
    assert np.isfinite(float(dmrg.e_tot))


def test_su2_block2_operator_table_cache_reuses_same_environment_basis():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=1,
        local_basis_policy="block2_like",
        orthonormalized_operator_dim=512,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
    )

    sites = dmrg.dmrg.ground_state.sites
    env = BlockSparseEnvironmentChain.build(sites, dmrg.H).start_sweep("lr")
    norm_env = BlockSparseEnvironmentChain.build(
        sites,
        _identity_mpo_factors_for_sites_and_mpo(sites, dmrg.H),
    ).start_sweep("lr")
    merged = merge_mps_sites(sites[0], sites[1])
    cache = {}

    first = env.orthonormal_bond_operator(0, merged, norm_env, tol=1.0e-6, max_dim=512, cache=cache)
    second = env.orthonormal_bond_operator(0, merged, norm_env, tol=1.0e-6, max_dim=512, cache=cache)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(cache) == 1
    assert second.source == "block_sparse_environment_sweep:block_sparse_operator_table"


def test_su2_block2_effective_operator_carries_boundary_stack_metadata():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=1,
        local_basis_policy="block2_like",
        orthonormalized_operator_dim=512,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
    )

    sites = dmrg.dmrg.ground_state.sites
    stack = RenormalizedBlockStack(namespace="hamiltonian")
    env = BlockSparseEnvironmentChain.build(
        sites,
        dmrg.H,
        renormalized_blocks=stack,
    ).start_sweep("lr")
    merged = merge_mps_sites(sites[0], sites[1])

    operator = env.bond_operator(0, merged)
    metadata = operator.metadata

    assert metadata["renormalized_boundary_source"] == "block_stack"
    assert metadata["left_boundary"]["side"] == "left"
    assert metadata["left_boundary"]["bond"] == 0
    assert metadata["right_boundary"]["side"] == "right"
    assert metadata["right_boundary"]["bond"] == 1
    assert metadata["left_boundary"]["stored_elements"] > 0
    assert metadata["right_boundary"]["stored_elements"] > 0
    assert metadata["symbolic_boundary_payloads"]["symbolic_boundary_payload_source"] == "symbolic_table"
    assert metadata["symbolic_boundary_payloads"]["symbolic_payloads_owned"] is True
    assert metadata["symbolic_boundary_payloads"]["symbolic_numeric_payloads"] > 0
    assert metadata["renormalized_local_operator_table"]["kind"] in {
        "transition",
        "factorized",
        "rank_coupled_factorized",
        "identity",
    }
    assert metadata["renormalized_local_operator_table"]["owner_side"] == "left"
    assert operator.local_operator_table is not None
    assert stack.stats["local_operator_tables"] > 0
    assert stack.stats["side_operator_tables"] > 0
    assert stack.stats["symbolic_operator_tables"] > 0
    assert stack.stats["symbolic_operator_terms"] > 0
    assert stack.stats["symbolic_operator_numeric_payloads"] > 0
    assert any(
        str(source).startswith("symbolic_")
        for source in stack.stats["side_operator_table_sources"]
    )
    assert any(
        str(source).endswith("prepared")
        for source in stack.stats["side_operator_table_sources"]
    )
    assert all(
        "lazy" not in str(source)
        for source in stack.stats["side_operator_table_sources"]
    )

    env.bond_operator(0, merged)

    assert stack.stats["local_operator_table_hits"] > 0
    assert stack.stats["local_operator_table_reuses"] > 0
    assert stack.stats["side_operator_table_hits"] > 0
    assert stack.stats["side_operator_table_reuses"] > 0
    assert "rank_coupled_left_factor_by_ket" in stack.stats[
        "side_operator_table_representations"
    ]
    assert "rank_coupled_right_factor_by_ket" in stack.stats[
        "side_operator_table_representations"
    ]


def test_su2_block2_operator_table_supports_multi_root_davidson():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=1,
        local_basis_policy="block2_like",
        orthonormalized_operator_dim=512,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
    )

    sites = dmrg.dmrg.ground_state.sites
    merged = merge_mps_sites(sites[0], sites[1])
    env = BlockSparseEnvironmentChain.build(sites, dmrg.H).start_sweep("lr")
    norm_env = BlockSparseEnvironmentChain.build(
        sites,
        _identity_mpo_factors_for_sites_and_mpo(sites, dmrg.H),
    ).start_sweep("lr")
    operator = env.orthonormal_bond_operator(
        0,
        merged,
        norm_env,
        tol=1.0e-6,
        max_dim=512,
        require_block_sparse_table=True,
    )

    _optimized, objective = solve_local_two_site(
        merged,
        operator,
        nstates=2,
        weights=[0.5, 0.5],
        tol=1.0e-6,
        itermax=30,
        max_space=48,
        allow_unconverged_roots=True,
    )

    assert objective["block_davidson"] is True
    assert objective["effective_local_problem"] == "orthonormalized_operator_standard"
    assert len(objective["state_energies"]) == 2
    assert len(objective["optimized_roots"]) == 2
    assert objective["state_energies"][0] <= objective["state_energies"][1]
    assert objective["renormalized_operator_table_stats"]["kind"] == "block_sparse"


def test_su2_block2_state_average_routes_through_operator_table():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dmrg.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=1,
        symmetry_list=["charge", "su2"],
        local_basis_policy="block2_like",
        state_average_validate_spin=False,
        mixer_zero_block_noise_scale=0.0,
    )

    objective = dmrg.dmrg.history[-1]["bond_objectives"][-1]
    assert objective["effective_local_problem"] == "state_averaged_coupled_davidson"
    assert objective["state_averaged_svd"] is True
    assert objective["block_davidson"] is True
    assert objective["target_irrep_filtered"] is True
    assert len(objective["state_energies"]) >= 2


def test_su2_block2_state_average_supports_larger_active_spaces():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=2,
        local_basis_policy="block2_like",
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
    )

    assert len(dmrg.dmrg.states) == 2
    assert all(entry.get("direction") != "dense" for entry in dmrg.dmrg.history)
    np.testing.assert_allclose(dmrg.dmrg.history[-1]["state_s2"], [0.0, 0.0], atol=1e-8)


def test_spatial_su2_dmrg_supports_state_average_roots():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    su2 = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    su2.run(nstates=2, weights=[0.5, 0.5], nsweeps=2, symmetry_list=["charge", "su2"])

    assert su2.dmrg.backend == "nonabelian"
    assert len(su2.dmrg.states) == 2
    assert np.asarray(su2.e_tot).shape == (2,)
    assert su2.e_tot[0] < su2.e_tot[1]
    assert su2.e_tot[1] == pytest.approx(-0.169291740911, abs=1e-7)
    objective = su2.dmrg.history[-1]["bond_objectives"][-1]
    assert "state_energies" in objective
    assert objective["target_irrep_filtered"] is True
    assert objective["effective_local_problem"] == "state_averaged_coupled_davidson"
    assert objective["dense_fallback"] is False
    assert objective["block_davidson"] is True
    assert objective["block_preconditioner"] is False
    assert objective["block_preconditioner_blocks"] >= 0
    assert str(objective["packed_matvec_backend"]).startswith("coupled-")


def test_spatial_su2_state_average_preserves_requested_weights():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dmrg.run(
        nstates=2,
        weights=[0.8, 0.2],
        nsweeps=1,
        symmetry_list=["charge", "su2"],
        state_average_validate_spin=False,
    )

    objective = dmrg.dmrg.history[-1]["bond_objectives"][-1]
    np.testing.assert_allclose(objective["state_average_weights"][:2], [0.8, 0.2])
    assert all(abs(weight) <= 1.0e-15 for weight in objective["state_average_weights"][2:])
    np.testing.assert_allclose(dmrg.dmrg.history[-1]["state_average_weights"], [0.8, 0.2])
    assert dmrg.dmrg.history[-1]["state_average_energy"] == pytest.approx(
        float(np.dot([0.8, 0.2], dmrg.dmrg.history[-1]["state_energies"]))
    )


def test_fully_reduced_spatial_su2_state_average_h2_roots():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    dmrg.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=4,
        mixer_zero_block_noise_scale=0.0,
    )

    assert dmrg._active_integral_build_info["spatial_site_basis"] == "fully_reduced_su2"
    assert dmrg.dmrg.converged is True
    np.testing.assert_allclose(dmrg.e_tot, [-1.137275940288, -0.169291745839], atol=1e-8)
    history = dmrg.dmrg.history[-1]
    np.testing.assert_allclose(history["state_average_weights"], [0.5, 0.5])
    assert history["state_average_energy"] == pytest.approx(
        float(np.dot([0.5, 0.5], history["state_energies"]))
    )
    assert history.get("state_s2") is None


def test_spatial_su2_state_average_supports_multisite_singlet_roots():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=40, init_guess="cid", site="spatial", verbose=0)
    dmrg.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=4,
        symmetry_list=["charge", "su2"],
        local_solver_kwargs={"dense_fallback_dim": 4096},
    )

    assert len(dmrg.dmrg.states) == 2
    assert all(entry.get("direction") != "dense" for entry in dmrg.dmrg.history)
    assert all(
        entry.get("backend") != "dense_target_sector_su2"
        for entry in dmrg.dmrg.history
    )
    np.testing.assert_allclose(
        dmrg.e_tot,
        [-2.177899321294, -1.557192705150],
        atol=1e-8,
    )
    np.testing.assert_allclose(dmrg.dmrg.history[-1]["state_s2"], [0.0, 0.0], atol=1e-8)
    assert dmrg.dmrg.converged is True
    assert dmrg.dmrg.history[-1]["convergence_metric"] == "energy_delta"
    assert dmrg.dmrg.history[-1]["energy_delta"] <= dmrg.tol


def test_spin_adapted_ed_matches_su2_reference_roots():
    from pyqed.qchem.dmrg import ED

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    ed = ED(mf, ncas=4, nelecas=4, symmetry="su2", verbose=0).run(nstates=2)

    assert ed.converged is True
    assert ed.history[-1]["backend"] == "spin_adapted_dense_ed"
    np.testing.assert_allclose(
        ed.e_tot,
        [-2.177899323464, -1.557192712326],
        atol=1e-8,
    )
    np.testing.assert_allclose(ed.state_s2, [0.0, 0.0], atol=1e-8)


def test_two_site_abelian_state_average_returns_root_mps_lists():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dmrg.run(nstates=2, weights=[0.5, 0.5], nsweeps=2, symmetry_list=["charge", "sz"])

    assert len(dmrg.dmrg.states) == 2
    assert np.asarray(dmrg.e_tot).shape == (2,)
    assert dmrg.dmrg.states[0].L == 2
    assert dmrg.dmrg.states[1].L == 2


def test_dmrg_fix_spin_accepts_non_singlet_targets_and_warns_for_linear_penalty():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    dmrg = DMRG(mf, ncas=2, nelecas=2, D=4, init_guess="hf")

    with pytest.warns(RuntimeWarning, match="linear \\+shift\\*S\\^2 penalty"):
        dmrg.fix_spin(ss=2, shift=0.3)

    assert dmrg.spin_purification is True
    assert dmrg.ss == pytest.approx(2.0)
    assert dmrg.shift == pytest.approx(0.3)


def test_rank_coupled_symbolic_payloads_are_packed_until_legacy_access():
    from pyqed.mps.nonabelian.renormalized import (
        SymbolicRenormalizedOperatorTable,
        SymbolicRenormalizedOperatorTerm,
    )
    from pyqed.mps.nonabelian.su2_qchem_plan import (
        PackedSU2BoundaryTable,
        pack_rank_coupled_boundary_table_from_payloads,
    )

    q0 = ("q0",)
    q1 = ("q1",)
    table = SymbolicRenormalizedOperatorTable(
        side="left",
        bond=1,
        terms_by_channel={
            0: (SymbolicRenormalizedOperatorTerm(channel=0),),
            1: (SymbolicRenormalizedOperatorTerm(channel=1),),
        },
    )
    block_map = {
        (q0, q0): (
            np.zeros((1, 1, 1)),
            np.arange(4.0).reshape(1, 2, 2),
        ),
        (q1, q0): (
            np.ones((1, 1, 1)),
            np.zeros((1, 1, 1)),
        ),
    }

    packed_table = table.with_numeric_payload(block_map)
    payloads = packed_table.numeric_payloads

    assert packed_table.stats["owns_numeric_payloads"] is True
    assert packed_table.stats["payload_kind"] == "rank_coupled_packed"
    assert isinstance(payloads.packed_table, PackedSU2BoundaryTable)
    assert payloads.stats["materialized"] is False

    reused = pack_rank_coupled_boundary_table_from_payloads(
        payloads,
        active_channels=table.channels,
        side="left",
        bond=1,
        representation="rank_coupled_by_ket",
    )
    assert reused is payloads.packed_table
    assert payloads.stats["materialized"] is False

    assert len(list(payloads.items())) == payloads.packed_table.n_channel_blocks
    assert payloads.stats["materialized"] is True


def test_su2_qchem_factorized_kernel_matches_reference_einsum():
    from pyqed.mps.nonabelian.su2_qchem_plan import SU2QChemSweepPlan

    class _Entry:
        def __init__(self, shape):
            self.shape = tuple(int(dim) for dim in shape)
            self.size = int(np.prod(self.shape, dtype=int))

    rng = np.random.default_rng(123)
    cases = (
        ((2, 3, 2, 2, 2, 2), (2, 2, 3, 2, 2, 2)),
        ((1, 1, 1, 3, 1, 2), (1, 3, 1, 1, 2, 1)),
    )
    for left_shape, right_shape in cases:
        left = rng.normal(size=left_shape)
        right = rng.normal(size=right_shape)
        out_entry = _Entry(
            (
                left_shape[1],
                left_shape[4],
                right_shape[4],
                right_shape[2],
            )
        )
        in_entry = _Entry(
            (
                left_shape[2],
                left_shape[5],
                right_shape[5],
                right_shape[3],
            )
        )
        got = SU2QChemSweepPlan._factorized_kernel(
            left,
            right,
            in_entry,
            out_entry,
        )
        ref = np.einsum(
            "tlkwab,twqrdc->ladqkbcr",
            left,
            right,
            optimize=False,
        ).reshape(out_entry.size, in_entry.size)
        assert got == pytest.approx(ref, abs=1.0e-12)
