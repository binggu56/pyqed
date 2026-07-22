import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf import direct_ci
from pyqed.qchem.mcscf import caspt2 as caspt2_module
from pyqed.qchem.mcscf.caspt2 import CASPT2, DiagonalCASPT2
from pyqed.qchem.mp.mp2 import MP2


def _h2_cas00():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=0, nelecas=0).run(nstates=1, method="direct_ci")
    return mf, mc


def _lih_cas22():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    return mc


def test_caspt2_theory_note_describes_diagonal_starter():
    theory = CASPT2.theory()

    assert "CASPT2" in theory
    assert "external" in theory
    assert "imaginary shift" in theory
    assert "experimental diagonal" in theory
    assert "contraction=\"strong\"" in theory


def test_caspt2_cas00_fock_limit_matches_rmp2():
    mf, mc = _h2_cas00()

    e_caspt2 = CASPT2(mc, zeroth_order="fock").run()
    mp2 = MP2(mf).run()

    np.testing.assert_allclose(e_caspt2, mp2.e_corr, atol=1.0e-10)


def test_caspt2_real_shift_damps_negative_mp2_limit_correction():
    _mf, mc = _h2_cas00()

    unshifted = CASPT2(mc, zeroth_order="fock").run()
    shifted = CASPT2(mc, zeroth_order="fock", real_shift=0.2).run()

    assert unshifted < 0.0
    assert shifted > unshifted


def test_caspt2_lih_cas22_runs_with_components():
    mc = _lih_cas22()

    pt = CASPT2(mc, zeroth_order="fock")
    e_corr = pt.run()

    assert np.isfinite(e_corr)
    assert e_corr < 0.0
    assert tuple(pt.components) == ("singles", "doubles")
    np.testing.assert_allclose(
        e_corr,
        sum(component.energy for component in pt.components.values()),
        atol=1.0e-14,
    )
    assert len(pt.external_determinants) > 0
    assert pt.couplings.shape == pt.denominators.shape
    assert pt.external_kernel_backend in {"cpp", "python"}


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
