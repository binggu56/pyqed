import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem import CASSCF, Molecule, MSCASPT2, XMSCASPT2
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf import direct_ci
from pyqed.qchem.mcscf import caspt2 as caspt2_module
from pyqed.qchem.mcscf.caspt2 import CASPT2, DiagonalCASPT2
from pyqed.qchem.mp.mp2 import MP2


def _caspt2_benchmark_module():
    path = Path(__file__).parents[1] / "benchmarks" / "caspt2_openmolcas.py"
    spec = importlib.util.spec_from_file_location("caspt2_openmolcas_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _h2_cas00():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="angstrom", basis="sto-3g")
    mol.build()
    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=0, nelecas=0).run(nstates=1, method="direct_ci")
    return mf, mc


def _lih_cas22(nstates=1):
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build()
    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=nstates, method="direct_ci")
    return mc


def test_caspt2_theory_note_describes_fully_contracted_solver():
    theory = CASPT2.theory()

    assert "CASPT2" in theory
    assert "external" in theory
    assert "imaginary shift" in theory
    assert "fully internally contracted" in theory
    assert "metric" in theory
    assert "contraction=\"strong\"" in theory


def test_openmolcas_benchmark_parses_full_precision_final_result():
    benchmark = _caspt2_benchmark_module()
    output = """
    :: RASSCF root number 1 Total energy: -7.88104527
    FINAL CASPT2 RESULT:
      Total nr of CASPT2 parameters:
       Before reduction: 110
       After reduction: 57
      Reference energy: -7.8810452656
      E2 (Variational): -0.0010903037
      Total energy: -7.8821355694
    :: CASPT2 Root 1 Total energy: -7.88213557
    """

    parsed = benchmark.parse_openmolcas_output(output)

    assert parsed["rasscf_total_energy_hartree"] == -7.8810452656
    assert parsed["caspt2_correction_hartree"] == -0.0010903037
    assert parsed["caspt2_total_energy_hartree"] == -7.8821355694
    assert parsed["contracted_basis_rank"] == 57


def test_openmolcas_benchmark_explicitly_correlates_inactive_orbitals(tmp_path):
    benchmark = _caspt2_benchmark_module()
    mol = SimpleNamespace(
        natom=2,
        nelec=4,
        atom_coords=lambda: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),
        atom_symbols=lambda: ("Li", "H"),
    )

    input_path = benchmark.write_openmolcas_input(
        benchmark.CASES["lih_cas22_sto3g"], mol, tmp_path
    )

    assert "Frozen = 0" in input_path.read_text(encoding="utf-8")


def test_openmolcas_benchmark_uses_matched_prascher_li_ccpvdz_basis():
    benchmark = _caspt2_benchmark_module()

    basis = benchmark._pyqed_basis(benchmark.CASES["lih_cas22_ccpvdz"])
    d_shells = [shell for shell in basis["Li"] if shell[0] == 2]

    assert len(d_shells) == 1
    assert d_shells[0][1][0] == pytest.approx(0.1144)


def test_disabled_openmolcas_benchmark_does_not_require_an_executable(tmp_path):
    benchmark = _caspt2_benchmark_module()
    input_path = tmp_path / "case.input"

    result = benchmark.run_openmolcas_case(
        input_path, command=None, mode="never", timeout=1.0
    )

    assert result["status"] == "not_run"


def test_caspt2_cas00_fock_limit_matches_rmp2():
    mf, mc = _h2_cas00()

    e_caspt2 = CASPT2(mc, zeroth_order="fock").run()
    mp2 = MP2(mf).run()

    np.testing.assert_allclose(e_caspt2, mp2.e_corr, atol=1.0e-10)


def test_fully_contracted_cas00_matches_rmp2_with_multiple_occupied_orbitals():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build()
    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=0, nelecas=0).run(nstates=1, method="direct_ci")

    pt = CASPT2(mc)
    e_caspt2 = pt.run()
    mp2 = MP2(mf).run()

    np.testing.assert_allclose(e_caspt2, mp2.e_corr, atol=2.0e-9)
    assert pt.contracted_basis_rank > 8


def test_caspt2_real_shift_damps_negative_mp2_limit_correction():
    _mf, mc = _h2_cas00()

    unshifted = CASPT2(mc, zeroth_order="fock").run()
    shifted = CASPT2(mc, zeroth_order="fock", real_shift=0.2).run()

    assert unshifted < 0.0
    assert shifted > unshifted


def test_fully_contracted_real_shift_reports_variational_correction():
    _mf, mc = _h2_cas00()

    shift = 0.2
    pt = CASPT2(mc, zeroth_order="fock", real_shift=shift)
    e_corr = pt.run()

    np.testing.assert_allclose(e_corr, pt.e_corr_variational, atol=1.0e-14)
    np.testing.assert_allclose(
        pt.shift_correction,
        -shift * pt.first_order_norm,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        pt.e_corr_variational,
        pt.e_corr_nonvariational + pt.shift_correction,
        atol=1.0e-14,
    )
    assert pt.shift_correction < 0.0


def test_caspt2_lih_cas22_runs_with_components():
    mc = _lih_cas22()

    pt = CASPT2(mc, zeroth_order="fock")
    e_corr = pt.run()

    assert np.isfinite(e_corr)
    assert e_corr < 0.0
    assert tuple(pt.components) == CASPT2.perturber_classes
    np.testing.assert_allclose(
        e_corr,
        sum(component.energy for component in pt.components.values()),
        atol=1.0e-14,
    )
    assert len(pt.external_determinants) > 0
    assert pt.couplings.shape == pt.denominators.shape
    assert pt.external_kernel_backend in {"cpp", "python"}
    assert pt.contraction_backend == "python_fully_internally_contracted"
    assert pt.contracted_basis_size > pt.contracted_basis_rank > len(CASPT2.perturber_classes)
    assert pt.contracted_metric.shape == (pt.contracted_basis_rank,) * 2
    np.testing.assert_allclose(pt.contracted_metric, np.eye(pt.contracted_basis_rank), atol=1.0e-11)
    assert pt.contracted_residual_norm < 1.0e-10
    assert 0.0 < pt.reference_weight <= 1.0


def test_streaming_ic_basis_matches_dense_canonical_reference():
    mc = _lih_cas22()

    dense = CASPT2(mc, ic_basis_backend="dense")
    streaming = CASPT2(mc, ic_basis_backend="streaming")
    dense_energy = dense.run()
    streaming_energy = streaming.run()

    np.testing.assert_allclose(streaming_energy, dense_energy, atol=5.0e-10)
    assert dense.ic_metric_backend == "class_component_canonical"
    assert streaming.ic_basis_backend == "streaming"
    assert streaming.ic_metric_backend == "streaming_mgs"
    assert streaming.contracted_basis_size == dense.contracted_basis_size
    assert streaming.contracted_basis_rank == dense.contracted_basis_rank
    assert (
        dense.work_estimate["dense_class_basis_bytes"]
        < dense.work_estimate["dense_basis_bytes"]
    )
    np.testing.assert_allclose(
        streaming.contracted_metric,
        np.eye(streaming.contracted_basis_rank),
        atol=2.0e-12,
    )


def test_direct_signature_blocks_match_semicanonical_enumerated_reference():
    mc = _lih_cas22()
    direct = CASPT2(
        mc,
        ic_basis_backend="direct",
        max_external_determinants=1,
    )
    direct_energy = direct.run()

    rotation = direct._direct_semicanonical_rotation
    rotated = copy.copy(mc)
    rotated.mo_coeff = np.asarray(mc.mo_coeff) @ rotation
    probe = CASPT2(mc)
    coefficients = probe._mo_coeff()
    h1 = probe._hcore_mo(coefficients)
    eri = probe._eri_mo(coefficients)
    fock, _fock_b, _reference = probe._generalized_fock_mo(h1, eri)
    fock = rotation.T @ fock @ rotation
    ncore = probe._ncore
    nocc = ncore + probe._ncas
    projected_fock = np.zeros_like(fock)
    projected_fock[:ncore, :ncore] = np.diag(np.diag(fock[:ncore, :ncore]))
    projected_fock[ncore:nocc, ncore:nocc] = fock[ncore:nocc, ncore:nocc]
    projected_fock[nocc:, nocc:] = np.diag(np.diag(fock[nocc:, nocc:]))
    explicit = CASPT2(
        rotated,
        ic_basis_backend="dense",
        fock_matrix=projected_fock,
    )

    np.testing.assert_allclose(direct_energy, explicit.run(), atol=2.0e-13)
    assert direct.external_space_backend == "direct_signature_blocks"
    assert direct.external_determinants == []
    assert direct.contracted_basis_rank == explicit.contracted_basis_rank
    assert direct.contracted_residual_norm < 1.0e-11
    for label in CASPT2.perturber_classes:
        np.testing.assert_allclose(
            direct.components[label].energy,
            explicit.components[label].energy,
            atol=2.0e-13,
        )


def test_connected_slater_condon_kernel_matches_general_element():
    mc = _lih_cas22()
    pt = CASPT2(mc)
    coefficients = pt._mo_coeff()
    h1 = pt._hcore_mo(coefficients)
    eri = pt._eri_mo(coefficients)
    references = caspt2_module._embed_active_determinants(
        mc.binary,
        mc.ncore,
        mc.ncas,
        coefficients.shape[1],
    )
    external = caspt2_module._generate_external_determinants(
        references,
        set(references),
        2 * coefficients.shape[1],
    )

    for bra in list(external)[:40]:
        for ket in references:
            rank = caspt2_module._excitation_rank(int(bra), int(ket))
            if rank not in (1, 2):
                continue
            expected = caspt2_module._hamiltonian_element_bits(
                int(bra), int(ket), h1, eri, coefficients.shape[1]
            )
            actual = caspt2_module._connected_hamiltonian_element_bits(
                int(bra), int(ket), h1, eri, coefficients.shape[1]
            )
            np.testing.assert_allclose(actual, expected, atol=2.0e-14)


def test_native_direct_rhs_matches_python_fallback():
    if caspt2_module._cpp_attr("caspt2_direct_couplings_words") is None:
        pytest.skip("native three-word CASPT2 coupling kernel is not built")
    mc = _lih_cas22()
    native = CASPT2(mc, ic_basis_backend="direct")
    native_energy = native.run()

    old_cpp = caspt2_module._casscf_cpp
    caspt2_module._casscf_cpp = None
    try:
        fallback = CASPT2(mc, ic_basis_backend="direct")
        fallback_energy = fallback.run()
    finally:
        caspt2_module._casscf_cpp = old_cpp

    np.testing.assert_allclose(native_energy, fallback_energy, atol=2.0e-13)
    for label in CASPT2.perturber_classes:
        np.testing.assert_allclose(
            native.components[label].energy,
            fallback.components[label].energy,
            atol=2.0e-13,
        )


def test_direct_cholesky_pair_factors_match_full_mo_integrals():
    mc = _lih_cas22()
    full = CASPT2(mc, ic_basis_backend="direct", use_cholesky=False)
    factorized = CASPT2(mc, ic_basis_backend="direct", use_cholesky=True)

    np.testing.assert_allclose(factorized.run(), full.run(), atol=2.0e-12)
    assert full.direct_integral_backend == "full_mo_eri"
    assert factorized.direct_integral_backend == "cholesky_pair_factors"
    assert factorized._direct_two_electron.ndim == 3


def test_direct_parallel_blocks_and_shared_candidate_groups_match_serial():
    mc = _lih_cas22()
    serial = CASPT2(mc, ic_basis_backend="direct", direct_workers=1)
    parallel = CASPT2(mc, ic_basis_backend="direct", direct_workers=2)

    np.testing.assert_allclose(parallel.run(), serial.run(), atol=2.0e-13)
    assert parallel.work_estimate["direct_workers"] == 2
    assert len(parallel.direct_candidate_groups) == len(
        parallel.direct_determinant_words
    )
    assert parallel.direct_candidate_offsets[-1] == len(
        parallel.direct_candidate_indices
    )
    assert parallel.direct_candidate_groups.max(initial=-1) < (
        len(parallel.direct_candidate_offsets) - 1
    )


def test_direct_tensor_builder_matches_online_reference():
    mc = _lih_cas22()
    tensor = CASPT2(
        mc,
        ic_basis_backend="direct",
        direct_build_backend="tensor",
    )
    online = CASPT2(
        mc,
        ic_basis_backend="direct",
        direct_build_backend="online",
    )

    np.testing.assert_allclose(tensor.run(), online.run(), atol=2.0e-13)
    assert tensor.contracted_basis_rank == online.contracted_basis_rank
    assert tensor.ic_metric_backend == "signature_active_tensor_mgs"
    assert online.ic_metric_backend == "signature_block_online_mgs"


def test_direct_symmetry_screening_reduces_rows_without_changing_energy():
    mol = Molecule(
        atom="Li 0 0 0; H 0 0 1.6",
        unit="angstrom",
        basis="sto-3g",
    )
    mol.build(symmetry="c2v")
    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        method="direct_ci",
    )
    symmetry = CASPT2(mc, ic_basis_backend="direct")

    plain_mc = copy.copy(mc)
    plain_mf = copy.copy(mf)
    plain_mf.orb_sym = None
    plain_mc.mf = plain_mf
    plain = CASPT2(plain_mc, ic_basis_backend="direct")

    np.testing.assert_allclose(symmetry.run(), plain.run(), atol=2.0e-13)
    assert symmetry.work_estimate["direct_symmetry_screening"] is True
    assert plain.work_estimate["direct_symmetry_screening"] is False
    assert (
        symmetry.work_estimate["direct_active_rows"]
        < plain.work_estimate["direct_active_rows"]
    )


def test_auto_backend_selects_direct_before_large_fois_generation():
    mc = _lih_cas22()
    pt = CASPT2(
        mc,
        ic_basis_backend="auto",
        max_external_determinants=1,
    )
    plan = pt.estimate_external_space()
    plan["external_determinants"] = 250_000
    pt.estimate_external_space = lambda: plan

    energy = pt.run()

    assert np.isfinite(energy)
    assert pt.ic_basis_backend == "direct"
    assert pt.work_estimate["selected_backend"] == "direct"


def test_ic_resource_planner_fails_before_unsafe_allocation():
    mc = _lih_cas22()
    pt = CASPT2(mc, ic_basis_backend="auto", max_memory_mb=0.01)

    with pytest.raises(MemoryError, match="estimated to require"):
        pt.run()

    assert pt.work_estimate["selected_backend"] == "streaming"
    assert pt.success is False
    assert pt.message.startswith("CASPT2 failed:")


def test_frozen_core_removes_core_hole_perturbers():
    mc = _lih_cas22()
    all_electron = CASPT2(mc)
    frozen = CASPT2(mc, frozen_core=1)

    all_electron.run()
    frozen.run()

    frozen_mask = 1 | (1 << frozen._nmo)
    assert len(frozen.external_determinants) < len(all_electron.external_determinants)
    assert all(det & frozen_mask == frozen_mask for det in frozen.external_determinants)
    assert (
        frozen.work_estimate["estimated_external_determinants"]
        == len(frozen.external_determinants)
    )
    assert (
        all_electron.work_estimate["estimated_external_determinants"]
        == len(all_electron.external_determinants)
    )
    assert frozen.work_estimate["frozen_core_orbitals"] == 1
    assert np.isfinite(frozen.e_corr)


def test_external_space_cap_is_checked_before_determinant_generation(monkeypatch):
    mc = _lih_cas22()

    def unexpected_generation(*_args, **_kwargs):
        raise AssertionError("external determinant generation should not run")

    monkeypatch.setattr(CASPT2, "_native_external_space", unexpected_generation)
    monkeypatch.setattr(
        caspt2_module,
        "_generate_external_determinants",
        unexpected_generation,
    )
    pt = CASPT2(mc, max_external_determinants=100)

    with pytest.raises(MemoryError, match="no integrals or external determinants"):
        pt.run()

    assert pt.work_estimate["estimated_external_determinants"] == 152


def test_external_space_estimate_is_available_without_running_caspt2():
    plan = CASPT2(_lih_cas22(), frozen_core=1).estimate_external_space()

    assert plan["external_determinants"] == 21
    assert sum(plan["external_class_counts"].values()) == 21
    assert plan["external_class_counts"]["Sijrs"] == 0
    assert plan["raw_ic_operators_upper_bound"] > plan["one_body_transitions"]


def test_python_external_builder_never_generates_frozen_core_holes():
    mc = _lih_cas22()
    ref_bits = caspt2_module._embed_active_determinants(
        mc.binary,
        mc.ncore,
        mc.ncas,
        mc.mo_coeff.shape[1],
    )
    external = caspt2_module._generate_external_determinants(
        ref_bits,
        set(ref_bits),
        2 * mc.mo_coeff.shape[1],
        frozen_core=1,
    )
    frozen_mask = 1 | (1 << mc.mo_coeff.shape[1])

    assert len(external) == 21
    assert all(det & frozen_mask == frozen_mask for det in external)


def test_frozen_core_rejects_more_than_available_core_orbitals():
    with pytest.raises(ValueError, match="frozen_core"):
        CASPT2(_lih_cas22(), frozen_core=2).run()


def test_matrix_free_ic_solver_matches_direct_real_shift():
    mc = _lih_cas22()
    options = {"real_shift": 0.1, "ic_basis_backend": "streaming"}
    direct = CASPT2(mc, linear_solver="direct", **options)
    iterative = CASPT2(
        mc,
        linear_solver="iterative",
        solver_tol=1.0e-12,
        max_solver_iterations=300,
        **options,
    )

    np.testing.assert_allclose(iterative.run(), direct.run(), atol=2.0e-11)
    assert iterative.linear_solver == "iterative"
    assert iterative.contracted_solver_backend == "matrix_free_minres_real_shift"
    assert iterative.solver_iterations > 0
    assert iterative.external_operator_nnz > 0
    assert iterative.solver_history == [iterative.contracted_residual_norm]
    assert iterative.contracted_residual_norm < 1.0e-10
    with pytest.raises(ValueError, match="matrix-free"):
        iterative.contracted_linear_system()


def test_matrix_free_ic_solver_matches_direct_imaginary_shift():
    mc = _lih_cas22()
    options = {"imaginary_shift": 0.15, "ic_basis_backend": "streaming"}
    direct = CASPT2(mc, linear_solver="direct", **options)
    iterative = CASPT2(
        mc,
        linear_solver="iterative",
        solver_tol=1.0e-12,
        max_solver_iterations=300,
        **options,
    )

    np.testing.assert_allclose(iterative.run(), direct.run(), atol=2.0e-11)
    assert iterative.contracted_solver_backend == "matrix_free_gmres_imaginary_shift"
    assert iterative.contracted_residual_norm < 1.0e-10


def test_caspt2_uncontracted_diagnostic_remains_explicitly_available():
    mc = _lih_cas22()

    pt = CASPT2(mc, zeroth_order="fock", contraction="uncontracted")
    e_corr = pt.run()

    assert np.isfinite(e_corr)
    assert tuple(pt.components) == ("singles", "doubles")
    assert pt.contraction_backend == "uncontracted"


def test_fully_contracted_caspt2_is_invariant_to_virtual_rotations():
    mc = _lih_cas22()
    reference = CASPT2(mc, zeroth_order="fock").run()

    rotated = copy.copy(mc)
    rotated.mo_coeff = np.asarray(mc.mo_coeff).copy()
    start = int(mc.ncore + mc.ncas)
    nvir = rotated.mo_coeff.shape[1] - start
    trial = np.arange(1, nvir * nvir + 1, dtype=float).reshape(nvir, nvir)
    unitary, _ = np.linalg.qr(trial)
    rotated.mo_coeff[:, start:] = rotated.mo_coeff[:, start:] @ unitary

    got = CASPT2(rotated, zeroth_order="fock").run()

    np.testing.assert_allclose(got, reference, atol=1.0e-10)


def test_fully_contracted_caspt2_accepts_converged_casscf_driver():
    mol = Molecule(
        atom="H 0.35 0 0; H -0.35 0 0",
        unit="angstrom",
        basis="ahlrichs vdz",
    )
    mol.build()
    mf = RHF(mol).run()
    mc = CASSCF(mf, ncas=2, nelecas=2, max_cycle=40, verbose=0).run(nstates=1)

    pt = CASPT2(mc)
    e_corr = pt.run()

    assert mc.converged
    assert np.isfinite(e_corr)
    assert e_corr < 0.0
    assert pt.contracted_basis_rank > 0
    assert pt.contracted_residual_norm < 1.0e-10


def test_caspt2_en_zeroth_order_is_available():
    mc = _lih_cas22()

    e_corr = DiagonalCASPT2(mc, zeroth_order="en").run()

    assert np.isfinite(e_corr)
    assert e_corr < 0.0


def test_caspt2_cpp_external_kernel_matches_python_fallback():
    if caspt2_module._cpp_attr("caspt2_external_kernel") is None:
        import pytest

        pytest.skip("CASPT2 C++ external kernel is not built")

    mc = _lih_cas22()
    native = CASPT2(mc, zeroth_order="fock")
    e_native = native.run()

    old_cpp = caspt2_module._casscf_cpp
    caspt2_module._casscf_cpp = None
    try:
        fallback = CASPT2(mc, zeroth_order="fock")
        e_fallback = fallback.run()
    finally:
        caspt2_module._casscf_cpp = old_cpp

    assert native.external_kernel_backend == "cpp"
    assert fallback.external_kernel_backend == "python"
    np.testing.assert_allclose(e_native, e_fallback, atol=1.0e-12)
    np.testing.assert_allclose(native.couplings, fallback.couplings, atol=1.0e-12)
    np.testing.assert_allclose(native.denominators, fallback.denominators, atol=1.0e-12)


def test_caspt2_cpp_external_space_matches_python_fallback():
    builder = caspt2_module._cpp_attr("caspt2_external_space")
    if builder is None:
        import pytest

        pytest.skip("CASPT2 C++ external-space builder is not built")

    mc = _lih_cas22()
    ncore = int(mc.ncore)
    ncas = int(mc.ncas)
    nmo = int(np.asarray(mc.mo_coeff).shape[1])
    ref_bits = caspt2_module._embed_active_determinants(mc.binary, ncore, ncas, nmo)

    native_dets, native_ranks, native_classes = builder(
        np.asarray(ref_bits, dtype=np.uint64),
        ncore,
        ncas,
        nmo,
    )
    external = caspt2_module._generate_external_determinants(
        ref_bits,
        set(ref_bits),
        2 * nmo,
    )
    py_dets = np.asarray(sorted(external), dtype=np.uint64)
    py_ranks = np.fromiter(
        (external[int(det)] for det in py_dets),
        dtype=np.int8,
        count=py_dets.size,
    )
    py_classes = caspt2_module._classify_external_determinants(
        [int(det) for det in py_dets],
        ncore,
        ncas,
        nmo,
    )

    np.testing.assert_array_equal(native_dets, py_dets)
    np.testing.assert_array_equal(native_ranks, py_ranks)
    np.testing.assert_array_equal(native_classes, py_classes)


def test_sparse_fock_action_matches_dense_determinant_matrix():
    mc = _lih_cas22()
    pt = CASPT2(mc)
    ref_bits = caspt2_module._embed_active_determinants(
        pt._binary,
        pt._ncore,
        pt._ncas,
        pt._nmo,
    )
    external = caspt2_module._generate_external_determinants(
        ref_bits,
        set(ref_bits),
        2 * pt._nmo,
    )
    determinants = sorted(external)
    trial = np.arange(pt._nmo * pt._nmo, dtype=float).reshape(pt._nmo, pt._nmo)
    matrix = 0.5 * (trial + trial.T) / (pt._nmo * pt._nmo)
    vectors = np.arange(2 * len(determinants), dtype=float).reshape(len(determinants), 2)
    vectors /= max(1, vectors.size)

    dense = caspt2_module._one_body_matrix_in_determinant_space(
        determinants,
        matrix,
        matrix,
        pt._nmo,
    )
    sparse_action = caspt2_module._apply_one_body_in_determinant_space(
        determinants,
        vectors,
        matrix,
        matrix,
        pt._nmo,
    )
    sparse_matrix, sparse_backend = (
        caspt2_module._one_body_sparse_matrix_in_determinant_space(
        determinants,
        matrix,
        matrix,
        pt._nmo,
        )
    )

    np.testing.assert_allclose(sparse_action, dense @ vectors, atol=1.0e-13)
    np.testing.assert_allclose(sparse_matrix @ vectors, dense @ vectors, atol=1.0e-13)
    assert sparse_backend in {"cpp_coo_to_scipy_csr", "python_coo_to_scipy_csr"}


def test_caspt2_strong_contraction_uses_standard_perturber_classes():
    mc = _lih_cas22()

    pt = CASPT2(mc, zeroth_order="fock", contraction="strong")
    e_corr = pt.run()

    assert np.isfinite(e_corr)
    assert pt.contraction_backend in {"cpp", "python"}
    assert pt.contracted_solver_backend in {"cpp", "python"}
    assert tuple(pt.components) == CASPT2.perturber_classes
    assert pt.external_classes.shape == pt.external_ranks.shape
    assert np.all(pt.external_classes >= 0)
    np.testing.assert_allclose(
        e_corr,
        sum(component.energy for component in pt.components.values()),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(e_corr, np.dot(pt.couplings, pt.amplitudes), atol=1.0e-14)
    assert any(component.norm > 0.0 for component in pt.components.values())


def test_caspt2_strong_contraction_exposes_linear_system():
    mc = _lih_cas22()

    pt = CASPT2(mc, zeroth_order="fock", contraction="strong")
    e_corr = pt.run()
    labels, metric, denominator_matrix, rhs, amplitudes = pt.contracted_linear_system()

    assert labels == CASPT2.perturber_classes
    assert metric.shape == denominator_matrix.shape == (len(labels), len(labels))
    assert rhs.shape == amplitudes.shape == (len(labels),)
    norms = np.array([pt.components[label].norm for label in labels])
    moments = np.array([pt.components[label].denominator_moment for label in labels])
    class_amplitudes = np.array([pt.components[label].amplitude for label in labels])

    np.testing.assert_allclose(np.diag(metric), norms, atol=1.0e-14)
    np.testing.assert_allclose(np.diag(denominator_matrix), moments, atol=1.0e-14)
    np.testing.assert_allclose(rhs, norms, atol=1.0e-14)
    np.testing.assert_allclose(amplitudes, class_amplitudes, atol=1.0e-14)

    active = np.abs(rhs) > 1.0e-14
    solved = np.zeros_like(rhs)
    solved[active] = np.linalg.solve(
        denominator_matrix[np.ix_(active, active)],
        rhs[active],
    )
    np.testing.assert_allclose(solved, amplitudes, atol=1.0e-12)
    np.testing.assert_allclose(e_corr, rhs @ amplitudes, atol=1.0e-14)


def test_caspt2_en_strong_contraction_uses_coupled_matrix():
    mc = _lih_cas22()

    pt = CASPT2(mc, zeroth_order="en", contraction="strong")
    e_corr = pt.run()
    labels, metric, denominator_matrix, rhs, amplitudes = pt.contracted_linear_system()

    assert labels == CASPT2.perturber_classes
    assert pt.contracted_matrix_kind == "en_coupled"
    assert pt.contracted_matrix_backend in {"cpp", "python"}
    assert metric.shape == denominator_matrix.shape == (len(labels), len(labels))
    np.testing.assert_allclose(metric, np.diag(np.diag(metric)), atol=1.0e-14)
    offdiag = denominator_matrix - np.diag(np.diag(denominator_matrix))
    assert np.count_nonzero(np.abs(offdiag) > 1.0e-14) > 0
    np.testing.assert_allclose(e_corr, rhs @ amplitudes, atol=1.0e-14)


def test_caspt2_cpp_contracted_solve_matches_numpy():
    solver = caspt2_module._cpp_attr("caspt2_solve_contracted")
    if solver is None:
        import pytest

        pytest.skip("CASPT2 C++ contracted linear solver is not built")

    metric = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 1.5, 0.2],
            [0.0, 0.2, 1.0],
        ],
        dtype=float,
    )
    denominator = np.array(
        [
            [-5.0, 0.3, 0.1],
            [0.3, -3.0, 0.2],
            [0.1, 0.2, -2.0],
        ],
        dtype=float,
    )
    rhs = np.array([1.0, 0.5, -0.25], dtype=float)
    real_shift = 0.15

    got = solver(metric, denominator, rhs, real_shift, 1.0e-12)
    ref = np.linalg.solve(denominator - real_shift * metric, rhs)

    np.testing.assert_allclose(got, ref, atol=1.0e-12)


def test_caspt2_cpp_strong_contract_matches_python_fallback():
    reducer = caspt2_module._cpp_attr("caspt2_strong_contract")
    if reducer is None:
        import pytest

        pytest.skip("CASPT2 C++ strong-contraction reducer is not built")

    mc = _lih_cas22()
    native = CASPT2(mc, zeroth_order="fock", contraction="strong")
    e_native = native.run()

    old_cpp = caspt2_module._casscf_cpp
    caspt2_module._casscf_cpp = None
    try:
        fallback = CASPT2(mc, zeroth_order="fock", contraction="strong")
        e_fallback = fallback.run()
    finally:
        caspt2_module._casscf_cpp = old_cpp

    assert native.contraction_backend == "cpp"
    assert native.contracted_solver_backend == "cpp"
    assert fallback.contraction_backend == "python"
    assert fallback.contracted_solver_backend == "python"
    np.testing.assert_allclose(e_native, e_fallback, atol=1.0e-12)
    np.testing.assert_allclose(native.amplitudes, fallback.amplitudes, atol=1.0e-12)
    for label in CASPT2.perturber_classes:
        ncomp = native.components[label]
        pcomp = fallback.components[label]
        assert ncomp.count == pcomp.count
        np.testing.assert_allclose(ncomp.energy, pcomp.energy, atol=1.0e-12)
        np.testing.assert_allclose(ncomp.norm, pcomp.norm, atol=1.0e-12)
        np.testing.assert_allclose(ncomp.denominator, pcomp.denominator, atol=1.0e-12)
        np.testing.assert_allclose(ncomp.denominator_moment, pcomp.denominator_moment, atol=1.0e-12)
        np.testing.assert_allclose(ncomp.amplitude, pcomp.amplitude, atol=1.0e-12)

    native_system = native.contracted_linear_system()
    fallback_system = fallback.contracted_linear_system()
    assert native_system[0] == fallback_system[0]
    for native_item, fallback_item in zip(native_system[1:], fallback_system[1:]):
        np.testing.assert_allclose(native_item, fallback_item, atol=1.0e-12)


def test_caspt2_cpp_en_coupled_contract_matches_python_fallback():
    builder = caspt2_module._cpp_attr("caspt2_en_coupled_contract")
    if builder is None:
        import pytest

        pytest.skip("CASPT2 C++ coupled EN contraction builder is not built")

    mc = _lih_cas22()
    native = CASPT2(mc, zeroth_order="en", contraction="strong")
    e_native = native.run()

    old_cpp = caspt2_module._casscf_cpp
    caspt2_module._casscf_cpp = None
    try:
        fallback = CASPT2(mc, zeroth_order="en", contraction="strong")
        e_fallback = fallback.run()
    finally:
        caspt2_module._casscf_cpp = old_cpp

    assert native.contracted_matrix_kind == "en_coupled"
    assert fallback.contracted_matrix_kind == "en_coupled"
    assert native.contracted_matrix_backend == "cpp"
    assert fallback.contracted_matrix_backend == "python"
    np.testing.assert_allclose(e_native, e_fallback, atol=1.0e-12)

    native_system = native.contracted_linear_system()
    fallback_system = fallback.contracted_linear_system()
    assert native_system[0] == fallback_system[0]
    for native_item, fallback_item in zip(native_system[1:], fallback_system[1:]):
        np.testing.assert_allclose(native_item, fallback_item, atol=1.0e-12)


def test_ms_caspt2_builds_symmetric_effective_hamiltonian():
    mc = _lih_cas22(nstates=2)

    ms = MSCASPT2(mc, roots=(0, 1))
    energies = ms.run()

    assert len(ms.state_specific) == 2
    np.testing.assert_allclose(ms.effective_hamiltonian, ms.effective_hamiltonian.T, atol=1.0e-14)
    np.testing.assert_allclose(
        energies,
        np.linalg.eigvalsh(ms.effective_hamiltonian),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(ms.mixing.T @ ms.mixing, np.eye(2), atol=1.0e-13)
    assert np.all(np.isfinite(energies))


def test_direct_ms_uses_root_specific_semicanonical_transition_blocks():
    mc = _lih_cas22(nstates=2)
    ms = MSCASPT2(mc, roots=(0, 1), ic_basis_backend="direct")

    energies = ms.run()

    assert np.all(np.isfinite(energies))
    np.testing.assert_allclose(
        ms.correction_matrix,
        ms.correction_matrix.T,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.diag(ms.correction_matrix),
        [pt.e_corr for pt in ms.state_specific],
        atol=2.0e-13,
    )


def test_xms_caspt2_diagonalizes_state_average_model_fock():
    mc = _lih_cas22(nstates=2)

    xms = XMSCASPT2(mc, roots=(0, 1))
    energies = xms.run()

    rotated_fock = xms.reference_rotation.T @ xms.reference_fock_matrix @ xms.reference_rotation
    np.testing.assert_allclose(rotated_fock, np.diag(np.diag(rotated_fock)), atol=1.0e-12)
    np.testing.assert_allclose(
        xms.effective_hamiltonian_original,
        xms.reference_rotation @ xms.effective_hamiltonian @ xms.reference_rotation.T,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(energies, np.linalg.eigvalsh(xms.effective_hamiltonian), atol=1.0e-13)


def test_direct_xms_uses_compressed_transition_amplitudes():
    mc = _lih_cas22(nstates=2)
    xms = XMSCASPT2(mc, roots=(0, 1), ic_basis_backend="direct")

    energies = xms.run()

    assert np.all(np.isfinite(energies))
    np.testing.assert_allclose(
        xms.correction_matrix,
        xms.correction_matrix.T,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.diag(xms.correction_matrix),
        [pt.e_corr for pt in xms.state_specific],
        atol=2.0e-13,
    )
    assert all(pt.direct_determinant_words.shape[1] == 3 for pt in xms.state_specific)


def test_xms_caspt2_energies_are_reference_phase_invariant():
    mc = _lih_cas22(nstates=2)
    reference = XMSCASPT2(mc, roots=(0, 1)).run()

    phased = copy.copy(mc)
    phased.ci = [np.asarray(vector).copy() for vector in mc.ci]
    phased.ci[1] *= -1.0
    got = XMSCASPT2(phased, roots=(0, 1)).run()

    np.testing.assert_allclose(got, reference, atol=1.0e-11)


def test_ms_and_xms_match_openmolcas_zero_ipea_lih_casci_reference():
    """OpenMolcas cd52dbe, CIONLY CAS(2,2), singlet roots 1 and 2."""
    mc = _lih_cas22(nstates=4)

    ms = MSCASPT2(mc, roots=(0, 2))
    xms = XMSCASPT2(mc, roots=(0, 2))
    ms_energies = ms.run()
    xms_energies = xms.run()

    np.testing.assert_allclose(
        ms_energies,
        [-7.87521959, -7.74260446],
        atol=5.0e-8,
    )
    np.testing.assert_allclose(
        np.diag(ms.effective_hamiltonian),
        [-7.87495355, -7.74287050],
        atol=5.0e-8,
    )
    np.testing.assert_allclose(
        abs(ms.effective_hamiltonian[0, 1]), 0.00593381, atol=5.0e-8
    )
    np.testing.assert_allclose(
        xms_energies,
        [-7.87689981, -7.74126334],
        atol=5.0e-8,
    )
    np.testing.assert_allclose(
        np.diag(xms.effective_hamiltonian),
        [-7.87351084, -7.74465231],
        atol=6.0e-8,
    )
    np.testing.assert_allclose(
        abs(xms.effective_hamiltonian[0, 1]), 0.02117032, atol=6.0e-8
    )
