import inspect
from types import SimpleNamespace

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
    SU2DMRG,
    _expectation_from_nonabelian_mps,
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
from pyqed.mps.nonabelian.renormalized import FamilyCppFactorKernel
from pyqed.mps.nonabelian.renormalized import RenormalizedBlockStack
from pyqed.mps.nonabelian.renormalized import configure_su2_kernel_policy
from pyqed.mps.nonabelian.renormalized import symbolic_mpo_core_transitions
from pyqed.mps.nonabelian.su2_kernel import SU2LocalAction
from pyqed.mps.nonabelian.su2_kernel import cpp_available as su2_cpp_available
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


def _build_cpp_integrals(molecule):
    molecule.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    return molecule


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def test_su2_block2_like_uses_short_local_solver_default():
    assert ORTHONORMALIZED_OPERATOR_ITERMAX_DEFAULT == 30


def test_dmrg_run_rejects_ambiguous_su2_controls_before_build():
    dmrg = object.__new__(DMRG)

    assert inspect.signature(DMRG.run).parameters["nsweeps"].default == 50
    with pytest.raises(TypeError, match="positive integer"):
        dmrg.run(nstates=1.5)
    with pytest.raises(TypeError, match="positive integer"):
        dmrg.run(nsweeps=1.5)
    with pytest.raises(ValueError, match="positive"):
        dmrg.run(nsweeps=0)
    with pytest.raises(ValueError, match="auto.*cpp.*python"):
        dmrg.run(su2_kernel_backend="native")
    with pytest.raises(TypeError, match="fully_reduced_state_average was removed"):
        dmrg.run(fully_reduced_state_average=False)


def test_su2_requires_compiled_integral_reference():
    molecule = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(molecule)
    mean_field = RHF(molecule).run()
    molecule._builtin_build_info = {}
    dmrg = DMRG(
        mean_field,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        verbose=0,
    )

    with pytest.raises(
        RuntimeError,
        match=r"compiled C\+\+ integral builder",
    ):
        dmrg.run(nsweeps=1)


def test_su2_cpp_path_uses_compiled_builtin_integrals():
    molecule = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(molecule)
    mean_field = RHF(molecule).run()
    dmrg = DMRG(
        mean_field,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    dmrg.run(
        nsweeps=1,
        conv_tol=-1.0,
        require_convergence=False,
        su2_kernel_backend="cpp",
        mixer_zero_block_noise_scale=0.0,
        mixer_nsweeps=0,
    )

    assert molecule._builtin_build_info["eri_backend"] == "cpp"
    assert str(
        molecule._builtin_build_info["dense_builder"]
    ).startswith("cpp-")
    assert dmrg.dmrg.ncompleted == 1
    assert dmrg.dmrg.ncompleted_half_sweeps == 2
    assert [row["direction"] for row in dmrg.dmrg.history] == ["lr", "rl"]
    assert [row["sweep"] for row in dmrg.dmrg.history] == [1, 1]
    assert [row["sweep_complete"] for row in dmrg.dmrg.history] == [False, True]


def test_su2_run_reports_nonconvergence_after_complete_sweep():
    molecule = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(molecule)
    dmrg = DMRG(
        RHF(molecule).run(),
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )

    with pytest.raises(RuntimeError, match="1 complete sweep"):
        dmrg.run(
            nsweeps=1,
            conv_tol=-1.0,
            su2_kernel_backend="cpp",
            mixer_zero_block_noise_scale=0.0,
            mixer_nsweeps=0,
        )

    assert dmrg.dmrg.converged is False
    assert dmrg.dmrg.success is False
    assert dmrg.states is dmrg.dmrg.states
    assert dmrg.ground_state is dmrg.dmrg.ground_state
    assert dmrg.history is dmrg.dmrg.history
    assert dmrg.dmrg.ncompleted == 1
    assert dmrg.dmrg.ncompleted_half_sweeps == 2


def test_abelian_run_reports_nonconvergence_after_complete_sweep():
    molecule = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(molecule)
    dmrg = DMRG(
        RHF(molecule).run(),
        ncas=4,
        nelecas=4,
        D=8,
        init_guess="hf",
        site="spatial",
        symmetry="sz",
        verbose=0,
    )

    with pytest.raises(RuntimeError, match="1 complete sweep"):
        dmrg.run(nsweeps=1, sweep_tol=-1.0, noise=0.0)

    assert dmrg.dmrg.converged is False
    assert dmrg.dmrg.success is False
    assert dmrg.states is dmrg.dmrg.states
    assert dmrg.ground_state is dmrg.dmrg.ground_state
    assert dmrg.history is dmrg.dmrg.sweep_history
    assert dmrg.dmrg.ncompleted == 1
    assert dmrg.dmrg.ncompleted_half_sweeps == 2
    assert [
        row["direction"]
        for row in dmrg.dmrg.sweep_history
        if row["direction"] in {"lr", "rl"}
    ] == ["lr", "rl"]


def test_fully_reduced_su2_solver_owner_is_not_process_global():
    molecule = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(molecule)
    mean_field = RHF(molecule).run()

    def build_owner():
        dmrg = DMRG(
            mean_field,
            ncas=2,
            nelecas=2,
            D=8,
            init_guess="hf",
            symmetry="su2",
            spatial_site_basis="fully_reduced",
            verbose=0,
        )
        dmrg.build()
        return dmrg._active_hamiltonian.moving_environment

    first = build_owner()
    second = build_owner()

    assert first is not second
    assert first.system_stats["revision"] == second.system_stats["revision"]


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


def test_family_cpp_factor_kernel_matches_compiled_apply_block():
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
    kernel = FamilyCppFactorKernel.from_compiled_term(term)
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
    assert reduced.info["two_body_builder"] == "SU2System[P/Q]"
    assert reduced.info["pipeline"] == (
        "cpp_integrals->su2_system->reduced_complementary_families"
    )
    assert reduced.moving_environment is not None
    assert reduced.info["su2_system"]["backend"] == "cpp"
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
    mol.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
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

    assert reduced.build_info["representation"] == "spatial_reduced_spinfree_mpo"
    assert reduced._active_hamiltonian is not None
    assert reduced._active_hamiltonian.initialize_system_kwargs()["n_sites"] == 2
    assert reduced._active_hamiltonian.initialize_system_kwargs()["n_elec"] == 2
    assert reduced._active_hamiltonian.mpo is reduced.H
    assert reduced.build_info["normal_complementary_production"] is True
    assert reduced.build_info["python_reduced_terms_materialized"] is False
    assert all(not core.dense_blocks for core in reduced.H)


def test_dmrg_accepts_ri_active_integrals():
    atom = "H 0 0 0; H 0 0 0.74"
    dense_mol = Molecule(atom=atom, unit="angstrom", basis="cc-pvdz")
    dense_mol.build(eri="dense")
    dense_mf = RHF(dense_mol).run(verbose=0)

    ri_mol = Molecule(atom=atom, unit="angstrom", basis="cc-pvdz")
    ri_mol.build(eri="ri")
    ri_mf = RHF(ri_mol).run(verbose=0)

    dense = DMRG(
        dense_mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry=None,
        verbose=0,
    )
    dense.run(nsweeps=4, require_convergence=False)

    ri = DMRG(
        ri_mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry=None,
        verbose=0,
    )
    ri.run(nsweeps=4, require_convergence=False)

    assert ri.build_info["mode"] == "ri"
    assert ri.integral_backend_override is None
    assert ri.integral_mode == "ri"
    assert ri.build_info["integral_mode"] == "ri"
    assert ri.build_info["factorized_integrals"] is True
    assert ri.build_info["aux_rank"] == ri_mol.eri_factors.shape[0]
    assert ri.e_tot == pytest.approx(dense.e_tot, abs=5.0e-5)


def test_dmrg_infers_mean_field_factor_backend_when_dense_also_exists():
    atom = "H 0 0 0; H 0 0 0.74"
    mol = Molecule(atom=atom, unit="angstrom", basis="cc-pvdz")
    mol.build(eri="dense+ri")
    mf = RHF(mol).run(verbose=0)

    inferred = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", verbose=0)
    inferred.run(nsweeps=4, symmetry_list=None)

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

    assert inferred.integral_backend_override is None
    assert inferred.integral_mode == "ri"
    assert inferred.build_info["mode"] == "ri"
    assert inferred.build_info["factorized_integrals"] is True
    assert ri.build_info["mode"] == "ri"
    assert ri.build_info["factorized_integrals"] is True


def test_dmrg_rejects_auto_integral_backend_sentinel():
    atom = "H 0 0 0; H 0 0 0.74"
    mol = Molecule(atom=atom, unit="angstrom", basis="sto-3g")
    mol.build(eri="dense")
    mf = RHF(mol).run(verbose=0)

    inferred = DMRG(mf, ncas=2, nelecas=2, D=8)
    assert inferred.integral_backend_override is None
    assert inferred.integral_mode == "dense"

    with pytest.raises(ValueError, match="Omit integral_backend"):
        DMRG(
            mf,
            ncas=2,
            nelecas=2,
            D=8,
            integral_backend="auto",
        )


def test_abelian_sz_hf_guess_can_leave_hf_determinant():
    atom = "H 0 0 0; H 0 0 0.74"
    mol = Molecule(atom=atom, unit="angstrom", basis="cc-pvdz")
    mol.build(eri="dense")
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
    _build_cpp_integrals(mol)
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
        require_convergence=False,
    )

    assert dmrg.e_tot == pytest.approx(-2.177899323464, abs=1e-6)
    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    assert objectives
    assert all(
        objective.get("no_python_bond_callbacks") is True
        for objective in objectives
    )
    assert dmrg.dmrg.diagnostics["kernel_backend"] == "cpp"


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
    mol.build()
    mf = mol.RHF(verbose=0).run()

    dense = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dense.run(nsweeps=4, symmetry_list=None)
    sym = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    sym.run(nsweeps=4, symmetry_list=["charge", "sz"])

    assert hasattr(sym.dmrg.ground_state.factors[0], "qns")
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
    mol.build(eri="dense")
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
        dmrg_performance="symmetric",
        spatial_family_environment_backend="block2_table",
    )

    dmrg.run(nsweeps=4)

    assert dmrg.e_tot == pytest.approx(-2.13944255109448, abs=1e-9)
    assert dmrg.build_info["representation"] == (
        "spatial_block2_table_carrier_mpo"
    )
    assert dmrg.build_info["carrier_only_family_hamiltonian"] is True
    build_timings = dmrg.build_info[
        "complementary_operator_family_build_timings"
    ]
    active_build_timings = dmrg.build_info["build_timings"]
    assert dmrg.build_info["qchem_compile_backend_actual"] == "cpp"
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
        in dmrg.build_info["resolved_abelian_matvec_options"]
    )
    assert (
        "moving_environment_cpp_qchem_family_descriptor_key"
        in dmrg.build_info["resolved_abelian_matvec_options"]
    )
    assert (
        dmrg.build_info[
            "moving_environment_cpp_owner_reused_from_build"
        ]
        is True
    )
    assert (
        dmrg.build_info["resolved_abelian_matvec_options"][
            "moving_environment_cpp_state_owner_instance"
        ]
        == "cpp_moving_environment_build_owner"
    )
    family_mpo_info = dmrg.build_info[
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
        == 0
    )
    assert (
        moving["cpp_moving_environment_owner_local_optimize_native_noise_blocks"]
        == 0
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


def test_h4_spatial_default_uses_compiled_sector_channel_mpo():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense")
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
        dmrg_performance="symmetric",
    )

    dmrg.run(nsweeps=4)

    assert dmrg.e_tot == pytest.approx(-2.13944255109448, abs=1e-9)
    active = dmrg.build_info
    assert active["representation"] == "spatial_direct_symbolic_mpo"
    assert active["spatial_u1_execution"] == "compiled_sector_channel_mpo"
    assert active["sector_complete_environment_movement"] is True
    assert active["complementary_route_expansion"] is False
    assert active["compiled_channel_skipped_redundant_final_expectation"] is True
    assert dmrg.complementary_operators is None
    assert dmrg.complementary_operator_mpos is None
    moving = dmrg.dmrg.environment_profile["moving_environment"]
    assert moving["compact_plan_builds"] > 0
    assert moving["cpp_moving_environment_compact_plan_davidson_calls"] > 0
    assert moving["cpp_davidson_table_source"] == "compact_renormalized_table"


def test_cpp_u1_factorized_table_matvec_matches_direct_contraction():
    if not getattr(cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
        pytest.skip("optional C++ Davidson/backend kernels are not available")

    rng = np.random.default_rng(7319)
    builder = cpp_davidson.RawPayloadBuilder()
    dims = (10, 10, 1, 1, 10, 1, 10, 1)
    dim = 174
    entries = []
    for index in range(75):
        left = np.ascontiguousarray(
            rng.normal(size=(1, 10, 10))
            + 1j * rng.normal(size=(1, 10, 10))
        )
        right = np.ascontiguousarray(
            rng.normal(size=(1, 10, 10))
            + 1j * rng.normal(size=(1, 10, 10))
        )
        scale = complex(rng.normal(), rng.normal())
        in_start = index
        out_start = 74 - index
        builder.add(left, right, dims, in_start, out_start, scale)
        entries.append((left[0], right[0], scale, in_start, out_start))

    analysis = dict(
        cpp_davidson.GroupedRenormalizedTable.raw_builder_hybrid_analysis(
            builder,
            dim,
            0.0,
        )
    )
    assert analysis["would_use_factorized"]
    table = cpp_davidson.GroupedRenormalizedTable.from_raw_builder(
        builder,
        dim,
        0.0,
    )
    assert table.storage() == "cpp_grouped_renormalized_table_factorized"

    vector = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    expected = np.zeros(dim, dtype=complex)
    for left, right, scale, in_start, out_start in entries:
        local = vector[in_start : in_start + 100].reshape(10, 10)
        expected[out_start : out_start + 100] += (
            scale * (left @ local @ right)
        ).reshape(-1)
    assert np.allclose(table.matvec(vector), expected, atol=2.0e-11, rtol=2.0e-12)


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
    mol.build(eri="dense")
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
        dmrg_performance="symmetric",
        spatial_family_environment_backend="block2_table",
    )

    dmrg.run(
        nsweeps=4,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=200,
        noise=0.0,
    )

    assert dmrg.e_tot == pytest.approx(-3.14350797943157, abs=1e-9)
    assert dmrg.dmrg.ncompleted == 4
    assert dmrg.dmrg.ncompleted_half_sweeps == 8
    assert dmrg.dmrg.sweep_history[-2]["direction"] == "rl"
    assert dmrg.dmrg.sweep_history[-1]["sweep"] == 4
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
    assert moving["cpp_moving_environment_owner_sweep_schedule_plan_halves"] == 9
    assert (
        moving["cpp_moving_environment_owner_sweep_schedule_plan_history_rows"]
        == len(dmrg.dmrg.sweep_history)
        == 9
    )
    assert [row["direction"] for row in dmrg.dmrg.sweep_history] == [
        "lr",
        "rl",
        "lr",
        "rl",
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


def test_spatial_dmrg_routes_su2_to_su2_solver():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
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
    solver = su2.run(nsweeps=4)

    assert isinstance(solver, SU2DMRG)
    assert solver is su2.dmrg
    assert su2.states is solver.states
    assert su2.ground_state is solver.ground_state
    assert su2.history is solver.history
    np.testing.assert_allclose(su2.energies, su2.e_tot)
    assert solver.backend == "su2"
    assert solver.ground_state is solver.states[0]
    assert solver.energy == pytest.approx(su2.e_tot)
    np.testing.assert_allclose(solver.energies, [su2.e_tot])
    assert solver.diagnostics["kernel_backend"] == "cpp"
    assert solver.diagnostics["memory_bytes"] > 0
    assert su2.symmetry == ["charge", "su2"]
    assert su2.site == "spatial"
    assert su2.spatial_reduced_mpo is True
    assert su2.build_info["representation"] == "spatial_reduced_spinfree_mpo"
    assert su2.dmrg.history[-1]["hamiltonian_system"]["n_sites"] == 2
    assert su2.dmrg.history[-1]["hamiltonian_system"]["n_elec"] == 2
    assert su2.dmrg.history[-1]["hamiltonian_symmetry"] == "su2"
    assert su2.dmrg.history[-1]["local_basis_policy"] == "orthonormalized_operator"
    assert su2.dmrg.history[-1]["max_bond_mode"] == "reduced"
    assert su2.e_tot == pytest.approx(dense.e_tot, abs=1e-7)
    mps_energy = _expectation_from_nonabelian_mps(
        su2.dmrg.ground_state,
        su2.H,
        moving_environment=su2._active_hamiltonian.moving_environment,
    )
    expected_mpo_energy = (
        su2.e_tot
        if su2.build_info["includes_core_energy"]
        else su2.e_tot - su2.e_core
    )
    assert mps_energy == pytest.approx(expected_mpo_energy, abs=1e-10)

    su2_canonical = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    su2_canonical.run(nsweeps=4, symmetry="su2", canonical_local_norm=True)
    assert su2_canonical.e_tot == pytest.approx(dense.e_tot, abs=1e-7)


def test_spatial_dmrg_symmetry_argument_replaces_backend_selector():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    abelian = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", symmetry="sz", verbose=0)
    abelian.run(nsweeps=4)
    legacy = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    legacy.run(nsweeps=4, symmetry_list=["charge", "sz"])

    assert abelian.symmetry == ["charge", "sz"]
    assert not hasattr(abelian.dmrg, "backend")
    assert abelian.e_tot == pytest.approx(legacy.e_tot, abs=1e-10)


def test_run_time_su2_selection_rebuilds_spin_orbital_hamiltonian_as_spatial_reduced():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", verbose=0)
    dmrg.run(nsweeps=2, symmetry="su2")

    assert dmrg.site == "spatial"
    assert dmrg.spatial_reduced_mpo is True
    assert dmrg.dmrg.backend == "su2"
    assert dmrg.e_tot == pytest.approx(-1.137275943783, abs=1e-9)


def test_run_time_symmetry_switch_invalidates_previous_abelian_mpo():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        site="spatial",
        symmetry="sz",
        verbose=0,
    )
    dmrg.run(nsweeps=1)
    old_h = dmrg.H
    dmrg.run(nsweeps=2, symmetry="su2")

    assert dmrg.H is not old_h
    assert dmrg.spatial_reduced_mpo is True
    assert dmrg.dmrg.backend == "su2"
    assert dmrg.e_tot == pytest.approx(-1.137275943783, abs=1e-9)


def test_abelian_dmrg_target_uses_explicit_spin_not_molecule_spin():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
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
    _build_cpp_integrals(mol)
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

    from pyqed.qchem.dmrg import ED

    exact = ED(mf, ncas=2, nelecas=2, symmetry="su2", spin=2, verbose=0).run()
    exact.qcdmrg.dmrg = SimpleNamespace(
        ground_state=exact.states[0],
        states=exact.states,
    )
    dm1, dm2 = dmrg.make_rdm12(spatial=True)
    exact_dm1, exact_dm2 = exact.qcdmrg.make_rdm12(spatial=True)
    np.testing.assert_allclose(dm1, exact_dm1, atol=1.0e-10)
    np.testing.assert_allclose(dm2, exact_dm2, atol=1.0e-10)


def test_su2_ground_state_default_uses_reduced_cpp_owned_sweeps():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", symmetry="su2", verbose=0)
    dmrg.run(nsweeps=2)

    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    assert objectives
    assert dmrg.spatial_site_basis == "fully_reduced"
    assert dmrg.dmrg.history[-1]["max_bond_mode"] == "reduced"
    assert all(objective.get("local_basis_policy") == "orthonormalized_operator" for objective in objectives)
    assert all(objective.get("cpp_owned_half_sweep") is True for objective in objectives)
    assert all(objective.get("cpp_active_solution_owned") is True for objective in objectives)
    assert all(objective.get("no_python_bond_callbacks") is True for objective in objectives)


def test_su2_ground_state_can_still_check_local_norm_debug_path():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
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
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=24, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=2,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
        require_convergence=False,
    )

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
        objective["projected_problem"] == "canonical_reduced_standard"
        for objective in metric_objectives
    )


def test_su2_block2_policy_uses_cpp_reduced_davidson():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=2,
        local_basis_policy="block2_like",
        orthonormalized_operator_dim=512,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
        require_convergence=False,
    )
    assert dmrg.e_tot == pytest.approx(-2.177899323464, abs=1e-6)

    objectives = [
        objective
        for entry in dmrg.dmrg.history
        for objective in entry.get("bond_objectives", [])
    ]
    assert objectives
    assert dmrg.spatial_site_basis == "fully_reduced"
    assert all(objective.get("local_basis_policy") == "orthonormalized_operator" for objective in objectives)
    assert all(
        objective.get("effective_local_problem") == "orthonormalized_standard"
        for objective in objectives
    )
    assert all(
        objective.get("projected_problem") == "canonical_reduced_standard"
        for objective in objectives
    )
    assert all(
        objective.get("orthonormalized_dim", 0) > 0
        for objective in objectives
    )
    assert all(
        objective.get("no_python_bond_callbacks") is True
        for objective in objectives
    )
    assert dmrg.dmrg.diagnostics["kernel_backend"] == "cpp"


def test_su2_reduced_boundary_pivots_match_component_identity_reference(
    monkeypatch,
):
    if not su2_cpp_available():
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    def solve(*, specialized):
        if specialized:
            monkeypatch.delenv(
                "PYQED_SU2_SPECIALIZE_PIVOT_SCALE_REFRESH",
                raising=False,
            )
        else:
            monkeypatch.setenv(
                "PYQED_SU2_SPECIALIZE_PIVOT_SCALE_REFRESH",
                "0",
            )
        dmrg = DMRG(
            mf,
            ncas=4,
            nelecas=4,
            D=16,
            init_guess="cid",
            symmetry="su2",
            verbose=0,
        )
        dmrg.run(
            nsweeps=2,
            require_convergence=False,
            max_bond_mode="reduced",
            mixer_zero_block_noise_scale=0.0,
            mixer_nsweeps=0,
        )
        return dmrg

    component_reference = solve(specialized=False)
    reduced = solve(specialized=True)

    assert reduced.e_tot == pytest.approx(
        component_reference.e_tot,
        abs=1.0e-11,
    )
    owner = reduced.dmrg.history[-1]["moving_environment_stats"][
        "su2_moving_environment"
    ]
    assert owner["decomposed_action_plan_hits"] > 0
    assert owner["peak_borrowed_reduced_contextual_right_elements"] > 0


def test_su2_block2_complementary_direct_projection_is_opt_in():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.build()
    families = dmrg._active_hamiltonian.complementary_operators
    object.__setattr__(families, "prefer_recursive_operator_matvec", False)
    object.__setattr__(families, "prefer_direct_orthonormal_projection", True)
    try:
        dmrg.run(
            nsweeps=1,
            require_convergence=False,
            local_basis_policy="block2_like",
            orthonormalized_operator_dim=512,
            max_bond_mode="per_sector",
            mixer_zero_block_noise_scale=0.0,
            direct_orthonormal_dense_max_elements=0,
            profile=True,
            su2_kernel_backend="python",
            su2_reference_complementary_families=True,
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
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=1,
        require_convergence=False,
        local_basis_policy="block2_like",
        su2_kernel_backend="python",
    )

    assert dmrg.dmrg.history[-1]["su2_kernel_policy"]["backend"] == "python"
    assert dmrg.dmrg.history[-1]["su2_kernel_backend_actual"] == "python"
    assert np.isfinite(float(dmrg.e_tot))


def test_su2_kernel_backend_cpp_requires_extension():
    previous = configure_su2_kernel_policy()
    try:
        if su2_cpp_available():
            configure_su2_kernel_policy(backend="cpp")
        else:
            with pytest.raises(RuntimeError, match="_su2_kernel"):
                configure_su2_kernel_policy(backend="cpp")
    finally:
        configure_su2_kernel_policy(
            backend=previous["backend"],
            debug_check=previous["debug_check"],
            debug_check_tol=previous["debug_check_tol"],
        )


def test_su2_default_davidson_schedule_tightens_after_timing_sweeps():
    if not su2_cpp_available():
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    molecule = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
    )
    _build_cpp_integrals(molecule)
    dmrg = DMRG(
        RHF(molecule).run(),
        ncas=2,
        nelecas=2,
        D=4,
        init_guess="hf",
        symmetry="su2",
        verbose=0,
    )
    dmrg.run(
        nsweeps=9,
        conv_tol=-1.0,
        require_convergence=False,
        su2_kernel_backend="cpp",
        mixer_zero_block_noise_scale=0.0,
        mixer_nsweeps=0,
    )

    local = [entry["local_solver_kwargs"] for entry in dmrg.dmrg.history]
    assert [entry["tol"] for entry in local] == (
        [1.0e-3] * 8 + [1.0e-5] * 8 + [1.0e-8] * 2
    )
    assert [entry["itermax"] for entry in local] == (
        [30] * 8 + [60] * 8 + [100] * 2
    )


def test_su2_cpp_owner_executes_cold_half_sweeps_without_python_bond_callbacks(
    monkeypatch,
):
    if not su2_cpp_available():
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    monkeypatch.setenv("PYQED_SU2_COMPARE_SHARED_LEFT", "1")
    monkeypatch.setenv("PYQED_SU2_COMPACT_RIGHT_PANEL_ELEMENTS", "4000000")
    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.6; H 0 0 3.2; "
            "H 0 0 4.8; H 0 0 6.4; H 0 0 8.0"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    mf = RHF(mol).run()
    dmrg = DMRG(
        mf,
        ncas=6,
        nelecas=6,
        D=16,
        init_guess="cid",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    dmrg.run(
        nsweeps=4,
        conv_tol=-1.0,
        require_convergence=False,
        local_basis_policy="block2_like",
        su2_kernel_backend="cpp",
        orthonormalized_operator_dim=0,
        max_bond_mode="reduced",
        mixer_zero_block_noise_scale=0.0,
        mixer_nsweeps=0,
        verify_returned_mps_energy=True,
        profile=True,
    )

    history = dmrg.dmrg.history
    assert len(history) == 8
    assert all(entry["cpp_owned_half_sweep"] for entry in history)
    assert all(
        entry["owned_half_sweep_readiness_code"] == 0
        for entry in history
    )
    moving = history[-1]["moving_environment_stats"]
    owner = moving["su2_moving_environment"]
    assert owner["half_sweeps"] == 8
    assert owner["owned_half_sweep_calls"] == 8
    assert owner["owned_half_sweep_bonds"] == 40
    assert owner["half_sweep_executor_calls"] == 0
    assert owner["half_sweep_python_bond_callbacks"] == 0
    assert owner["site_merge_calls"] == 40
    assert owner["active_bond_cpp_splits"] == 40
    assert owner["active_bond_complementary_fallbacks"] == 0
    assert owner["metric_boundary_action_count"] == 0
    assert owner["complementary_execution_graph_builds"] > 0
    assert owner["complementary_execution_graph_hits"] > 0
    assert owner["complementary_execution_graph_bytes"] > 0
    assert owner["direct_complementary_action_calls"] > 0
    assert owner["direct_source_factor_loads"] > 0
    assert owner["compact_right_panel_budget_bytes"] == 32_000_000
    assert (
        owner["compact_right_panel_value_bytes"]
        <= owner["compact_right_panel_budget_bytes"]
    )
    assert owner["compact_right_panel_registry_builds"] > 0
    assert owner["compact_right_panel_numeric_refreshes"] > 0
    assert owner["contextual_compiled_schedule_builds"] > 0
    assert owner["contextual_compiled_schedule_hits"] > 0
    assert owner["contextual_compiled_schedule_bytes"] > 0
    assert owner["peak_borrowed_reduced_contextual_right_elements"] > 0
    assert owner["contextual_route_plan_count"] <= 10
    assert (
        owner["complementary_execution_slab_full_prepares"]
        + owner["complementary_execution_slab_partial_prepares"]
        > 0
    )
    assert (
        owner["complementary_execution_slab_bytes"]
        <= owner["complementary_execution_slab_capacity_bytes"]
        <= owner["complementary_execution_slab_budget_bytes"]
    )
    assert moving["hamiltonian_boundary_advances"] == 0
    assert float(dmrg.e_tot) == pytest.approx(
        history[-1]["returned_mps_energy"],
        abs=1.0e-10,
    )
    left_error, right_error = dmrg.dmrg.ground_state.canonical_errors()
    assert left_error < 1.0e-10
    assert right_error < 1.0e-10


def test_su2_cpp_cas16_fused_actions_match_pointer_with_dense_pairs(
    monkeypatch,
):
    if not su2_cpp_available():
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    monkeypatch.setenv("PYQED_SU2_COMPARE_SHARED_LEFT", "1")
    monkeypatch.delenv(
        "PYQED_SU2_AMORTIZE_OUTPUT_FUSION_RIGHT",
        raising=False,
    )
    monkeypatch.delenv(
        "PYQED_SU2_DIRECT_CHANNEL_OUTPUT_FUSION",
        raising=False,
    )
    molecule = Molecule(
        atom="; ".join(
            f"H 0 0 {1.6 * site}" for site in range(16)
        ),
        unit="bohr",
        basis="sto-3g",
    )
    molecule.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    dmrg = DMRG(
        RHF(molecule).run(),
        ncas=16,
        nelecas=16,
        D=128,
        init_guess="cid",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    dmrg.run(
        nsweeps=1,
        conv_tol=-1.0,
        require_convergence=False,
        local_basis_policy="block2_like",
        su2_kernel_backend="cpp",
        orthonormalized_operator_dim=0,
        max_bond_mode="reduced",
        bond_multiplicity=4,
        davidson_max_iter=2,
        davidson_tol=1.0e-3,
        mixer_zero_block_noise_scale=0.0,
        mixer_nsweeps=0,
        verify_returned_mps_energy=True,
        profile=True,
    )

    owner = dmrg.dmrg.history[-1]["moving_environment_stats"][
        "su2_moving_environment"
    ]
    assert owner["dense_pair_matvec_seconds"] > 0.0
    assert owner["raw_output_fusion_gemm_calls"] > 0
    assert owner["half_sweep_python_bond_callbacks"] == 0
    assert float(dmrg.e_tot) == pytest.approx(
        dmrg.dmrg.history[-1]["returned_mps_energy"],
        abs=1.0e-10,
    )


def test_su2_cpp_direct_complementary_executor_matches_batched(monkeypatch):
    if not su2_cpp_available():
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    monkeypatch.delenv(
        "PYQED_SU2_COMPACT_RIGHT_PANEL_ELEMENTS",
        raising=False,
    )
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    mf = RHF(mol).run()

    def solve(force_direct):
        if force_direct:
            monkeypatch.setenv("PYQED_SU2_POINTER_ACTIONS", "1")
        else:
            monkeypatch.delenv("PYQED_SU2_POINTER_ACTIONS", raising=False)
        dmrg = DMRG(
            mf,
            ncas=2,
            nelecas=2,
            D=4,
            init_guess="cid",
            symmetry="su2",
            spatial_site_basis="fully_reduced",
            verbose=0,
        )
        dmrg.run(
            nsweeps=2,
            conv_tol=-1.0,
            require_convergence=False,
            local_basis_policy="block2_like",
            su2_kernel_backend="cpp",
            orthonormalized_operator_dim=0,
            max_bond_mode="reduced",
            mixer_zero_block_noise_scale=0.0,
            mixer_nsweeps=0,
            verify_returned_mps_energy=True,
            profile=True,
        )
        owner = dmrg.dmrg.history[-1]["moving_environment_stats"][
            "su2_moving_environment"
        ]
        return float(dmrg.e_tot), owner

    batched_energy, batched = solve(False)
    direct_energy, direct = solve(True)
    assert direct_energy == pytest.approx(batched_energy, abs=1.0e-12)
    assert batched["compact_right_panel_budget_bytes"] == 0
    assert batched["compact_right_panel_registry_builds"] == 0
    assert batched["compact_right_panel_value_bytes"] == 0
    assert direct["direct_complementary_action_calls"] > 0
    assert direct["direct_complementary_actions"] > 0
    assert direct["raw_pointer_execution_matvec_calls"] > 0
    assert direct["half_sweep_python_bond_callbacks"] == 0


def test_su2_cpp_shared_right_panels_skip_partial_output_groups(monkeypatch):
    if not su2_cpp_available():
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    molecule = Molecule(
        atom="; ".join(
            f"H 0 0 {1.6 * site}" for site in range(10)
        ),
        unit="bohr",
        basis="sto-3g",
    )
    molecule.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    mean_field = RHF(molecule).run()
    monkeypatch.delenv(
        "PYQED_SU2_DISABLE_SHARED_RIGHT_PANELS",
        raising=False,
    )
    monkeypatch.setenv(
        "PYQED_SU2_COMPARE_SHARED_RIGHT_PANELS",
        "1",
    )
    monkeypatch.setenv(
        "PYQED_SU2_SHARED_RIGHT_COPY_BUDGET",
        "1024",
    )
    dmrg = DMRG(
        mean_field,
        ncas=10,
        nelecas=10,
        D=32,
        init_guess="cid",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    dmrg.run(
        nsweeps=1,
        conv_tol=-1.0,
        require_convergence=False,
        local_basis_policy="block2_like",
        su2_kernel_backend="cpp",
        orthonormalized_operator_dim=0,
        max_bond_mode="reduced",
        mixer_zero_block_noise_scale=0.0,
        mixer_nsweeps=0,
        verify_returned_mps_energy=True,
        profile=True,
    )
    owner = dmrg.dmrg.history[-1]["moving_environment_stats"][
        "su2_moving_environment"
    ]
    assert owner["peak_raw_shared_right_panel_count"] == 0
    assert owner["peak_raw_shared_right_binding_count"] == 0
    assert owner["raw_shared_right_gemm_calls"] == 0
    assert owner["half_sweep_python_bond_callbacks"] == 0
    assert float(dmrg.e_tot) == pytest.approx(
        dmrg.dmrg.history[-1]["returned_mps_energy"],
        abs=1.0e-10,
    )


def test_su2_block2_cpp_owner_avoids_transformed_kernel_build():
    mol = Molecule(
        atom=(
            "H 0 0 0; H 0 0 1.6; H 0 0 3.2; "
            "H 0 0 4.8; H 0 0 6.4; H 0 0 8.0"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    mf = RHF(mol).run()

    dmrg = DMRG(
        mf,
        ncas=6,
        nelecas=6,
        D=16,
        init_guess="cid",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    dmrg.build()
    assert dmrg.build_info[
        "python_reduced_terms_materialized"
    ] is False
    assert all(not factor.reduced_terms for factor in dmrg.H)
    families = dmrg._active_hamiltonian.complementary_operators
    object.__setattr__(families, "prefer_recursive_operator_matvec", True)
    object.__setattr__(families, "prefer_complementary_payload_tensor_matvec", True)
    try:
        dmrg.run(
            nsweeps=1,
            require_convergence=False,
            local_basis_policy="block2_like",
            orthonormalized_operator_dim=512,
            max_bond_mode="per_sector",
            mixer_zero_block_noise_scale=0.0,
            direct_orthonormal_dense_max_elements=0,
            su2_kernel_backend="auto",
            debug_su2_kernel_check=True,
            record_post_update_energy=True,
            davidson_tol=2.5e-9,
            davidson_max_iter=73,
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
    assert dmrg.dmrg.history[0]["local_solver_kwargs"]["tol"] == pytest.approx(
        2.5e-9
    )
    assert dmrg.dmrg.history[0]["local_solver_kwargs"]["itermax"] == 73
    if all(objective.get("cpp_active_solution_owned") for objective in objectives):
        assert all(
            objective["cpp_davidson_kind"]
            == "cpp_su2_active_canonical_solve"
            for objective in objectives
        )
        assert all(
            objective["direct_complementary_action_executor"]
            for objective in objectives
        )
        assert all(
            objective["packed_matvec_backend"] == "su2-contextual-cpp"
            for objective in objectives
        )
        assert all(
            not (objective.get("renormalized_operator_build_timing") or {})
            for objective in objectives
        )
        assert all(
            objective.get("renormalized_operator_table_stats") is None
            for objective in objectives
        )
        owner_stats = dmrg.dmrg.history[-1]["moving_environment_stats"][
            "su2_moving_environment"
        ]
        assert owner_stats["factor_routes_hermitianized"] is False
        assert owner_stats["active_bond_complementary_fallbacks"] == 0
        assert owner_stats["raw_factor_routes"] is False
        assert owner_stats["reduced_contextual_routes"] is True
        assert owner_stats["complementary_local_actions"] is True
        assert owner_stats["half_sweep_python_bond_callbacks"] >= 5
        assert max(
            float(objective["residual"]) for objective in objectives
        ) < 1.0e-8
        assert float(dmrg.e_tot) == pytest.approx(
            -3.2310893994341,
            abs=4.0e-6,
        )
        return
    timings = [
        objective.get("renormalized_operator_build_timing") or {}
        for objective in objectives
    ]
    assert all(timing.get("contextual_cpp_routes", 0.0) > 0.0 for timing in timings)
    assert all(
        timing.get("component_direct_factorized_preferred", 0.0) > 0.0
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
    assert actual_backend in {"python", "cpp"}
    cpp_actions = [
        (objective.get("renormalized_operator_table_stats") or {}).get(
            "su2_local_action"
        )
        for objective in objectives
    ]
    if actual_backend == "cpp":
        assert any(cpp_actions) or any(
            (objective.get("renormalized_operator_table_stats") or {}).get(
                "cpp_block_table"
            )
            is True
            for objective in objectives
        )
        owner_stats = dmrg.dmrg.history[-1]["moving_environment_stats"][
            "su2_moving_environment"
        ]
        assert owner_stats["factor_routes_hermitianized"] is False
        assert owner_stats["half_sweeps"] == 1
        assert owner_stats["half_sweep_executor_calls"] == 1
        assert owner_stats["half_sweep_executor_bonds"] == 5
        assert owner_stats["half_sweep_python_bond_callbacks"] == 5
        assert owner_stats["bond_steps"] == 5
        assert owner_stats["block_svd_calls"] == 5
        assert owner_stats["block_svd_blocks"] >= 5
        assert owner_stats["block_svd_workspace_bytes"] > 0
        assert owner_stats["split_site_installs"] == 16
        assert owner_stats["split_site_boundary_uses"] == 10
        assert owner_stats["split_site_count"] == 6
        assert owner_stats["split_site_bytes"] > 0
        assert owner_stats["site_merge_calls"] == 5
        assert owner_stats["site_merge_blocks"] > 0
        assert owner_stats["site_merge_bytes"] > 0
        assert owner_stats["active_bond_complementary_prepares"] == 5
        assert owner_stats["active_bond_complementary_fallbacks"] == 0
        assert "active_bond_complementary_davidson_calls" in owner_stats
        assert (
            "active_bond_complementary_generalized_davidson_calls"
            in owner_stats
        )
        assert owner_stats["borrowed_local_operator_bytes"] == 0
        assert owner_stats["borrowed_factor_pool_bytes"] == owner_stats[
            "borrowed_raw_factor_source_bytes"
        ]
        assert owner_stats["raw_factor_routes"] is False
        assert owner_stats["reduced_contextual_routes"] is True
        assert owner_stats["complementary_local_actions"] is True
        assert owner_stats["complementary_local_action_count"] > 0
        assert owner_stats["complementary_local_term_count"] > 0
        assert owner_stats["complementary_local_action_bytes"] > 0
        assert owner_stats["raw_route_group_count"] == 0
        assert owner_stats["reduced_contextual_fallbacks"] == 0
        assert (
            owner_stats["complementary_execution_slab_bytes"]
            <= owner_stats["complementary_execution_slab_capacity_bytes"]
            <= owner_stats["complementary_execution_slab_budget_bytes"]
        )
        assert (
            owner_stats["complementary_execution_slab_bytes"]
            <= owner_stats["complementary_execution_slab_required_bytes"]
        )
        assert owner_stats["peak_reduced_contextual_boundary_rank"] >= 1
        assert owner_stats["borrowed_factor_route_transform_bytes"] > 0
        assert owner_stats["factor_route_scratch_bytes"] > 0
        assert owner_stats["factor_route_projection_scratch_bytes"] > 0
        assert owner_stats["real_factor_route_matvec_calls"] > 0
        assert owner_stats["contextual_route_plan_builds"] == 5
        assert owner_stats["contextual_route_plan_count"] == 5
        assert owner_stats["contextual_route_plan_bytes"] > 0
        assert owner_stats["contextual_compiled_schedule_builds"] > 0
        assert owner_stats["contextual_compiled_schedule_bytes"] > 0
        assert (
            owner_stats["raw_input_superchannel_batch_count"] > 0
            or owner_stats["resident_family_route_count"] > 0
        )
        assert (
            owner_stats["resident_family_kernel_bytes"]
            <= owner_stats["resident_family_kernel_budget_bytes"]
        )
        family_route_counts = owner_stats[
            "complementary_family_route_counts"
        ]
        assert set(family_route_counts) == {
            "S", "R", "A", "P", "B", "Q", "unlabeled"
        }
        assert sum(
            count
            for name, count in family_route_counts.items()
            if name != "unlabeled"
        ) > 0
        assert all(
            (
                objective.get("renormalized_operator_table_stats") or {}
            ).get("cpp_factor_route_projection")
            is True
            for objective in objectives
        )
        assert max(float(objective["residual"]) for objective in objectives) < 1.0e-10
        assert max(
            abs(
                float(objective["energy"])
                - float(objective["post_update_energy"])
            )
            for objective in objectives
        ) < 1.0e-10
        for previous, current in zip(objectives, objectives[1:]):
            assert abs(
                float(previous["energy"])
                - float(current["guess_energy"])
            ) < 1.0e-10
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
        assert not any(cpp_actions)
    assert any(
        (
            (objective.get("renormalized_operator_table_stats") or {}).get(
                "complementary_direct_matvec"
            )
            is True
            or (objective.get("renormalized_operator_table_stats") or {}).get(
                "packed_cpp_exclusive_owner"
            )
            is True
        )
        for objective in objectives
    )
    assert all(
        set(
            (objective.get("renormalized_operator_table_stats") or {}).get(
                "family_names",
                (),
            )
        )
        == {"S", "R", "A", "P", "B", "Q"}
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
        == {"S", "R", "A", "P", "B", "Q"}
        for objective in objectives
    )
    assert float(dmrg.e_tot) == pytest.approx(
        -3.2310893994341,
        abs=1.0e-8,
    )


def test_su2_block2_operator_table_cache_reuses_same_environment_basis():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=1,
        require_convergence=False,
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
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=1,
        require_convergence=False,
        local_basis_policy="block2_like",
        orthonormalized_operator_dim=512,
        max_bond_mode="per_sector",
        mixer_zero_block_noise_scale=0.0,
    )

    sites = dmrg.dmrg.ground_state.sites
    from pyqed.qchem.dmrg.backends.reduced import (
        build_su2_normal_complementary_mpo,
    )

    reference_mpo = build_su2_normal_complementary_mpo(
        dmrg.H[0].normal_complementary_owner,
        fully_reduced=True,
        materialize_reduced_terms=True,
    )
    stack = RenormalizedBlockStack(namespace="hamiltonian")
    env = BlockSparseEnvironmentChain.build(
        sites,
        reference_mpo,
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
        "rank_coupled_contextual",
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
    assert all(
        "lazy" not in str(source)
        for source in stack.stats["side_operator_table_sources"]
    )

    env.bond_operator(0, merged)

    assert stack.stats["local_operator_table_hits"] > 0
    assert stack.stats["local_operator_table_reuses"] > 0
    assert stack.stats["side_operator_table_hits"] > 0
    assert stack.stats["side_operator_table_reuses"] > 0
    assert "rank_coupled_by_ket" in stack.stats[
        "side_operator_table_representations"
    ]


def test_su2_block2_operator_table_supports_multi_root_davidson():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", symmetry="su2", verbose=0)
    dmrg.run(
        nsweeps=1,
        require_convergence=False,
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


def test_su2_python_reference_state_average_remains_available():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dmrg.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=1,
        require_convergence=False,
        symmetry_list=["charge", "su2"],
        local_basis_policy="block2_like",
        su2_kernel_backend="python",
        state_average_validate_spin=False,
        mixer_zero_block_noise_scale=0.0,
    )

    objective = dmrg.dmrg.history[-1]["bond_objectives"][-1]
    assert objective["effective_local_problem"] == "state_averaged_dense"
    assert objective["state_averaged_svd"] is True
    assert objective["block_davidson"] is False
    assert objective.get("cpp_state_average") is not True
    assert len(objective["state_energies"]) >= 2


def test_su2_cpp_state_average_selects_fully_reduced_sites():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    dmrg = DMRG(
        RHF(mol).run(),
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        site="spatial",
        verbose=0,
    )

    solver = dmrg.run(
        nstates=2,
        nsweeps=1,
        require_convergence=False,
        symmetry="su2",
        su2_kernel_backend="cpp",
    )

    assert dmrg.spatial_site_basis == "fully_reduced"
    assert solver.diagnostics["kernel_backend"] == "cpp"


def test_su2_block2_state_average_supports_larger_active_spaces():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(mol)
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
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    su2 = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    solver = su2.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=2,
        symmetry="su2",
    )

    assert su2.dmrg.backend == "su2"
    assert su2.spatial_site_basis == "fully_reduced"
    assert solver is su2.dmrg
    assert su2.states is solver.states
    assert su2.ground_state is solver.ground_state
    np.testing.assert_allclose(su2.e_tot, solver.energies)
    assert len(solver.states) == 2
    assert solver.ground_state is solver.states[0]
    assert not hasattr(solver, "multiroot_state")
    np.testing.assert_allclose(solver.weights, [0.5, 0.5])
    np.testing.assert_allclose(solver.energies, su2.e_tot)
    assert solver.state_average_energy == pytest.approx(np.mean(su2.e_tot))
    assert np.asarray(su2.e_tot).shape == (2,)
    assert su2.e_tot[0] < su2.e_tot[1]
    assert su2.e_tot[1] == pytest.approx(-0.169291740911, abs=1e-7)
    objective = su2.dmrg.history[-1]["bond_objectives"][-1]
    assert "state_energies" in objective
    assert objective["effective_local_problem"] == "cpp_state_averaged_canonical_reduced"
    assert objective["cpp_state_average"] is True
    assert objective["cpp_block_davidson"] is True
    assert objective["cpp_owned_half_sweep"] is True
    assert objective["no_python_bond_callbacks"] is True
    assert objective["cpp_post_truncation_expectation"] is True
    assert objective["block_davidson"] is True
    owner = su2.dmrg.history[-1]["moving_environment_stats"][
        "su2_moving_environment"
    ]
    assert owner["state_average_roots"] == 2
    assert owner["half_sweep_python_bond_callbacks"] == 0
    assert owner["owned_half_sweep_bonds"] == 4


def test_spatial_su2_state_average_preserves_requested_weights():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dmrg.run(
        nstates=2,
        weights=[0.8, 0.2],
        nsweeps=1,
        require_convergence=False,
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


@pytest.mark.parametrize(
    "weights, message",
    [
        ([1.0], "match nstates"),
        ([0.5, np.nan], "finite"),
        ([1.0, -0.1], "nonnegative"),
        ([0.0, 0.0], "positive sum"),
    ],
)
def test_state_average_rejects_invalid_weights(weights, message):
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    dmrg = DMRG(
        RHF(mol).run(),
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        verbose=0,
    )
    with pytest.raises(ValueError, match=message):
        dmrg.run(nstates=2, weights=weights, nsweeps=1)


def test_fully_reduced_spatial_su2_state_average_h2_roots():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        verbose=0,
    )
    dmrg.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=4,
        mixer_zero_block_noise_scale=0.0,
    )

    assert dmrg.symmetry == ["charge", "su2"]
    assert dmrg.site == "spatial"
    assert dmrg.spatial_site_basis == "fully_reduced"
    assert dmrg.build_info["spatial_site_basis"] == "fully_reduced_su2"
    assert dmrg.dmrg.converged is True
    np.testing.assert_allclose(dmrg.e_tot, [-1.137275940288, -0.169291745839], atol=1e-8)
    history = dmrg.dmrg.history[-1]
    np.testing.assert_allclose(history["state_average_weights"], [0.5, 0.5])
    assert history["state_average_energy"] == pytest.approx(
        float(np.dot([0.5, 0.5], history["state_energies"]))
    )
    np.testing.assert_allclose(history["state_s2"], [0.0, 0.0], atol=1e-12)


def test_fully_reduced_spatial_su2_keeps_multiplicity_only_basis_across_sweeps():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    dmrg = DMRG(
        RHF(mol).run(),
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    dmrg.run(nsweeps=3)

    assert dmrg.e_tot == pytest.approx(-1.137275943783, abs=1e-9)
    if su2_cpp_available():
        history = dmrg.dmrg.history[-1]
        owner = history["moving_environment_stats"]["su2_moving_environment"]
        assert history["energy_source"] == "cpp_terminal_local"
        assert history["local_solver_kwargs"]["orthonormalize_generalized_dim"] == 0
        assert owner["last_half_sweep_energy"] == pytest.approx(
            history["energy"],
            abs=1e-12,
        )
        assert owner["staged_bond_updates"] == owner["bond_steps"]
        assert owner["committed_bond_updates"] == owner["bond_steps"]
        assert owner["resident_family_kernel_budget_bytes"] == 32_000_000
        assert (
            owner["resident_family_kernel_bytes"]
            <= owner["resident_family_kernel_budget_bytes"]
        )
        assert owner["resident_family_factor_pack_budget_bytes"] == 4_000_000
        assert (
            owner["resident_family_factor_pack_bytes"]
            <= owner["resident_family_factor_pack_budget_bytes"]
        )
    assert all(
        site.metadata.get("physical_basis") == "fully_reduced_su2"
        for site in dmrg.dmrg.ground_state.sites
    )


def test_fully_reduced_spatial_su2_builds_spin_traced_rdms_without_determinants(monkeypatch):
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
    dmrg = DMRG(
        RHF(mol).run(),
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        spatial_site_basis="fully_reduced",
        verbose=0,
    )
    dmrg.run(nsweeps=2)

    from pyqed.mps.nonabelian import models as nonabelian_models

    builds = {"h1": 0, "eri": 0}
    original_h1 = nonabelian_models.build_spatial_one_body_reduced_mpo
    original_eri = nonabelian_models.build_spatial_spinfree_eri_mpo

    def counted_h1(*args, **kwargs):
        builds["h1"] += 1
        return original_h1(*args, **kwargs)

    def counted_eri(*args, **kwargs):
        builds["eri"] += 1
        return original_eri(*args, **kwargs)

    monkeypatch.setattr(nonabelian_models, "build_spatial_one_body_reduced_mpo", counted_h1)
    monkeypatch.setattr(nonabelian_models, "build_spatial_spinfree_eri_mpo", counted_eri)

    dm1, dm2 = dmrg.make_rdm12(spatial=True)
    assert np.trace(dm1) == pytest.approx(2.0, abs=1e-10)
    assert np.einsum("pprr->", dm2) == pytest.approx(2.0, abs=1e-10)
    assert builds == {"h1": 0, "eri": 0}
    assert dmrg.spatial_rdm_diagnostics["algorithm"] == "cpp_su2_wigner_eckart_npdm"
    assert dmrg.spatial_rdm_diagnostics["determinant_expansion"] is False
    assert dmrg.spatial_rdm_diagnostics["magnetic_component_expansion"] is False
    assert dmrg.spatial_rdm_diagnostics["component_max_bond_dimension"] == 0
    assert dmrg.spatial_rdm_diagnostics["reduced_max_bond_dimension"] > 0
    assert dmrg.spatial_rdm_diagnostics["max_operator_channels"] > 0

    repeated_dm1, repeated_dm2 = dmrg.make_rdm12(spatial=True)
    np.testing.assert_allclose(repeated_dm1, dm1, atol=1e-12)
    np.testing.assert_allclose(repeated_dm2, dm2, atol=1e-12)
    assert builds == {"h1": 0, "eri": 0}
    assert dmrg.spatial_rdm_diagnostics["cache_hits"] >= 2

    runtime = dmrg._su2_runtime
    component_reference = runtime.moving_environment.spatial_npdm(
        dmrg.dmrg.ground_state.sites,
        spin_rotation_reduction=True,
        component_reference=True,
    )
    assert component_reference["magnetic_component_expansion"] is True
    np.testing.assert_allclose(component_reference["rdm1"], dm1, atol=1e-12)
    np.testing.assert_allclose(component_reference["rdm2"], dm2, atol=1e-12)

    dmrg._su2_runtime = None
    dmrg._fully_reduced_rdm_state_context = None
    fallback_dm1, fallback_dm2 = dmrg.make_rdm12(spatial=True)
    dmrg._su2_runtime = runtime
    dmrg._fully_reduced_rdm_state_context = None
    assert dmrg.spatial_rdm_diagnostics["algorithm"] == "su2_component_mps_npdm"
    np.testing.assert_allclose(fallback_dm1, dm1, atol=1e-12)
    np.testing.assert_allclose(fallback_dm2, dm2, atol=1e-12)


def test_spatial_su2_state_average_supports_multisite_singlet_roots():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=40, init_guess="cid", site="spatial", verbose=0)
    dmrg.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=4,
        local_solver_kwargs={"dense_fallback_dim": 4096},
    )

    assert dmrg.symmetry == ["charge", "su2"]
    assert dmrg.site == "spatial"
    assert dmrg.spatial_site_basis == "fully_reduced"
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
    assert dmrg.build_info["normal_complementary_production"] is True
    assert dmrg.build_info["python_reduced_terms_materialized"] is False
    np.testing.assert_allclose(dmrg.dmrg.history[-1]["state_s2"], [0.0, 0.0], atol=1e-8)
    assert dmrg.dmrg.converged is True
    assert dmrg.dmrg.history[-1]["convergence_metric"] == "energy_delta"
    assert dmrg.dmrg.history[-1]["energy_delta"] <= dmrg.tol

    from pyqed.qchem.dmrg import ED

    exact = ED(mf, ncas=4, nelecas=4, symmetry="su2", verbose=0).run(nstates=2)
    np.testing.assert_allclose(dmrg.e_tot, exact.e_tot, atol=1.0e-8)
    exact.qcdmrg.dmrg = SimpleNamespace(
        ground_state=exact.states[0],
        states=exact.states,
    )
    for root in range(2):
        dm1, dm2 = dmrg.make_rdm12(root, spatial=True)
        exact_dm1, exact_dm2 = exact.qcdmrg.make_rdm12(root, spatial=True)
        np.testing.assert_allclose(dm1, exact_dm1, atol=2.0e-8)
        np.testing.assert_allclose(dm2, exact_dm2, atol=2.0e-8)
        assert np.trace(dm1) == pytest.approx(4.0, abs=1.0e-9)
        assert np.einsum("pprr->", dm2) == pytest.approx(12.0, abs=1.0e-8)


def test_spin_adapted_ed_matches_su2_reference_roots():
    from pyqed.qchem.dmrg import ED

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    _build_cpp_integrals(mol)
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
    _build_cpp_integrals(mol)
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dmrg.run(nstates=2, weights=[0.5, 0.5], nsweeps=2, symmetry_list=["charge", "sz"])

    assert len(dmrg.dmrg.states) == 2
    assert np.asarray(dmrg.e_tot).shape == (2,)
    assert dmrg.dmrg.states[0].L == 2
    assert dmrg.dmrg.states[1].L == 2


def test_dmrg_fix_spin_accepts_non_singlet_targets_and_warns_for_linear_penalty():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    _build_cpp_integrals(mol)
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
