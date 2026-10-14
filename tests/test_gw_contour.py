"""Contour integration against full spectral screening and PySCF."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.gw.contour import ContourDeformation

g = importlib.import_module("pyqed.gw.gw")


def model():
    rng = np.random.default_rng(8)
    f = rng.normal(size=(9, 5, 5)) * 0.1
    f = (f + f.transpose(0, 2, 1)) / 2
    e = np.array([-1.0, -0.6, 0.1, 0.4, 0.9])
    return SimpleNamespace(
        nso=10,
        nocc=4,
        e_mf=np.repeat(e, 2),
        _qp_energy_so=None,
        _pair_factors=f,
        screening="TDH",
        eta=1e-8,
        max_memory=0.01,
        mo_coeff=np.eye(5),
        _M=None,
        _sigma_x_matrix=None,
        eri=None,
    )


def test_contour_self_energy_residues_and_orbital_crossings(monkeypatch):
    obj = model()
    poles, vectors = g.rpa(obj)
    g._charge_spatial_couplings(obj, poles, vectors)
    monkeypatch.setattr(
        g.scipy.linalg, "eigh", lambda *a, **k: pytest.fail("dense eigensolve")
    )
    contour = ContourDeformation(obj, obj.e_mf[::2], [2, 0], nw=64)
    for p in [0, 2]:
        for omega in [-1.2, -1.0, -0.600001, -0.6, -0.599999, 0.1, 0.3, 1.1]:
            actual, derivative = contour.evaluate(p, omega)
            expected = g.sigma(obj, 2 * p, 2 * p, omega, poles, vectors)[0]
            assert actual.real == pytest.approx(expected.real, abs=2e-10)
            h = 1e-5
            difference = (
                contour.evaluate(p, omega + h)[0] - contour.evaluate(p, omega - h)[0]
            ) / (2 * h)
            assert derivative.real == pytest.approx(difference.real, abs=2e-7)


def test_quadrature_convergence():
    obj = model()
    poles, vectors = g.rpa(obj)
    g._charge_spatial_couplings(obj, poles, vectors)
    reference = g.sigma(obj, 4, 4, 1.1, poles, vectors)[0].real
    errors = [
        abs(
            ContourDeformation(obj, obj.e_mf[::2], [2], nw=nw).evaluate(2, 1.1)[0].real
            - reference
        )
        for nw in [8, 64]
    ]
    assert errors[1] < errors[0] and errors[1] < 1e-10


def test_matrix_free_screening_action_and_residues(monkeypatch):
    obj = model()
    direct = ContourDeformation(obj, obj.e_mf[::2], [0, 2], nw=64)
    iterative = ContourDeformation(obj, obj.e_mf[::2], [0, 2], nw=64,
                                   residue_solver="gmres", screening_tol=1e-12)
    rng = np.random.default_rng(43)
    vector = rng.normal(size=9) + 1j*rng.normal(size=9)
    frequencies = [0., .3, 1.13, 3.]
    for frequency in frequencies:
        np.testing.assert_allclose(iterative.dielectric_action(frequency, vector),
                                   direct.dielectric(frequency) @ vector, atol=1e-12)
    monkeypatch.setattr(iterative, "dielectric", lambda *a, **k: pytest.fail("matrix build"))
    for frequency in frequencies:
        np.testing.assert_allclose(iterative.residue(frequency, 0, 1),
                                   direct.residue(frequency, 0, 1), atol=2e-11, rtol=2e-11)
    for omega in [-1.2, -.7, .3, 1.13]:
        np.testing.assert_allclose(iterative.evaluate(0, omega), direct.evaluate(0, omega),
                                   atol=2e-11, rtol=2e-11)
    assert iterative.screening_info["max_relative_residual"] <= 1e-12
    assert iterative.screening_info["iterations"] > 0


def test_matrix_free_screening_failure_is_explicit():
    obj = model()
    contour = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8,
                                 residue_solver="gmres", screening_maxiter=1)
    with pytest.raises(np.linalg.LinAlgError, match="Screening GMRES failed"):
        contour.residue(1.1, 0, 1)
    assert contour.screening_info["failed_solves"] == 1


def test_matrix_free_checks_true_residual(monkeypatch):
    obj = model()
    contour = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8,
                                 residue_solver="gmres")
    module = importlib.import_module("pyqed.gw.contour")
    monkeypatch.setattr(module, "gmres", lambda op, rhs, **kw: (np.zeros_like(rhs), 0))
    with pytest.raises(np.linalg.LinAlgError, match="relative residual=1"):
        contour.residue(.3, 0, 1)
    assert contour.screening_info["failed_solves"] == 1


def test_matrix_free_near_screening_pole():
    obj = model()
    obj.eta = 1e-4
    poles, _ = g.rpa(obj)
    frequency = float(poles[0]) + 1e-4
    direct = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8)
    iterative = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8,
                                   residue_solver="gmres", screening_tol=1e-11)
    np.testing.assert_allclose(iterative.residue(frequency, 0, 1),
                               direct.residue(frequency, 0, 1), atol=1e-8, rtol=1e-8)


def test_residue_warm_start_preserves_value_and_derivative(monkeypatch):
    obj = model()
    direct = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8)
    iterative = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8,
                                   residue_solver="gmres", screening_tol=1e-12)
    module = importlib.import_module("pyqed.gw.contour")
    original, guesses = module.gmres, []
    def solve(*args, **kwargs):
        guesses.append(kwargs.get('x0'))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'gmres', solve)
    for frequency in [.3, .30001, .8]:
        np.testing.assert_allclose(iterative.residue(frequency, 0, 1),
                                   direct.residue(frequency, 0, 1), atol=2e-11, rtol=2e-11)
    assert guesses[0] is None and guesses[1] is not None and guesses[2] is None


def test_blocked_dielectric_action_retains_all_auxiliary_rows():
    from benchmarks.benchmark_contour_contractions import reference_and_current
    contour = reference_and_current(259, 30)[1]
    rng = np.random.default_rng(12)
    vector = rng.normal(size=259)+1j*rng.normal(size=259)
    np.testing.assert_allclose(contour.dielectric_action(.37, vector),
                               contour.dielectric(.37)@vector, atol=2e-12, rtol=2e-12)


def test_failed_warm_start_retries_same_equation_from_zero(monkeypatch):
    obj = model()
    direct = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8)
    iterative = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8,
                                   residue_solver='gmres', screening_tol=1e-12)
    iterative.residue(.3, 0, 1)
    module = importlib.import_module('pyqed.gw.contour')
    original = module.gmres
    def solve(operator, rhs, **kwargs):
        if kwargs['x0'] is not None:
            return np.zeros_like(rhs), 1
        return original(operator, rhs, **kwargs)
    monkeypatch.setattr(module, 'gmres', solve)
    np.testing.assert_allclose(iterative.residue(.30001, 0, 1),
                               direct.residue(.30001, 0, 1), atol=2e-11, rtol=2e-11)
    assert iterative.screening_info['warm_retries'] == 1
    assert iterative.screening_info['failed_solves'] == 0


def test_saved_transition_factors_are_reused():
    obj = model()
    pairs = obj._pair_factors[:, :2, 2:].reshape(9, -1).copy()
    cached = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8, transition_factors=pairs)
    baseline = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8)
    assert cached.pairs is pairs
    np.testing.assert_allclose(cached.evaluate(0, -.7), baseline.evaluate(0, -.7), atol=1e-13)
    with pytest.raises(ValueError, match='every occupied-virtual'):
        ContourDeformation(obj, obj.e_mf[::2], [0], nw=8, transition_factors=pairs[:, :-1])


def test_saved_mo_factors_avoid_transformation(water, monkeypatch):
    baseline = g.GW(water, freq_int='contour')
    factors = baseline._pair_factors
    monkeypatch.setattr(g, '_build_spatial_pair_factors_mo',
                        lambda *a: pytest.fail('Repeated MO transformation'))
    cached = g.GW(water, freq_int='contour', mo_pair_factors=factors)
    assert cached._pair_factors is factors
    np.testing.assert_allclose(cached.v_mf, baseline.v_mf, atol=1e-13)
    with pytest.raises(ValueError, match='Saved MO pair factors'):
        g.GW(water, freq_int='contour', mo_pair_factors=factors[:, :-1, :])


@pytest.mark.parametrize('workers', [2, 4])
@pytest.mark.parametrize('solver', ['direct', 'gmres', 'lanczos'])
def test_parallel_orbital_solves_match_serial_and_preserve_shared_state(water, workers, solver):
    import copy
    from benchmarks.tdi_window_convergence import parallel_qp_orbitals
    from pyqed.gw.bse import BSE
    gw = g.GW(water, freq_int='contour', eta=1e-6)
    orbs = list(range(len(water.mo_energy)))
    controls = dict(nw=8, residue_solver=solver, screening_tol=1e-12)
    gw._contour = ContourDeformation(gw, water.mo_energy, orbs, **controls)
    parallel = np.full((3, len(orbs)), np.nan)
    for p, energy, weight, residual, info, duration in parallel_qp_orbitals(gw, orbs, controls, 1e-9, workers):
        parallel[:, p] = energy, weight, residual
        assert info['failed_solves'] == 0 and duration >= 0
    assert gw.e_qp is None and gw._contour.screening_info['solves'] == 0
    assert not hasattr(gw._contour, '_residue_guesses')
    serial = copy.copy(gw)
    serial._contour = copy.copy(gw._contour)
    serial._contour.screening_info = gw._contour.screening_info.copy()
    serial.run(orbs=orbs, reuse_screening=True, tol=1e-9, **controls)
    np.testing.assert_allclose(parallel, [serial.e_qp, serial.qp_weights, serial.qp_residuals],
                               atol=1e-10, rtol=1e-10)
    gw.e_qp = parallel[0]
    np.testing.assert_allclose(BSE(gw).run(nroots=3, low_rank=False).e,
                               BSE(serial).run(nroots=3, low_rank=False).e, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("memory", [0.001, 100.0])
def test_dielectric_and_symmetric_residue_against_general_solve(memory):
    obj = model()
    obj.max_memory = memory
    contour = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8)
    pairs, gaps = contour.pairs, contour.gaps
    for frequency in [0.0, 0.3, 1.2, 2.0]:
        z = frequency + 1j * obj.eta
        for imaginary in [False, True]:
            weights = (-4 * gaps / (frequency**2 + gaps**2) if imaginary
                       else 4 * gaps / (z*z - gaps**2))
            expected = np.eye(len(pairs)) - (pairs * weights) @ pairs.T
            np.testing.assert_allclose(
                contour.dielectric(frequency, imaginary=imaginary), expected,
                atol=2e-13, rtol=2e-13,
            )
        rhs = contour.factors[:, 1, 0]
        solution = g.scipy.linalg.solve(contour.dielectric(frequency), rhs,
                                        assume_a="gen")
        projected = pairs.T @ solution
        value = rhs @ (solution - rhs)
        slope = np.sum(-8*z*gaps/(z*z-gaps**2)**2 * projected**2)
        np.testing.assert_allclose(contour.residue(frequency, 0, 1),
                                   [value, slope], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(nw=2),
        dict(nw=8.5),
        dict(nw=True),
        dict(quadrature_scale=0),
        dict(orbs=[-1]),
        dict(orbs=[5]),
        dict(orbs=[1, 1]),
        dict(orbs=[]),
        dict(orbs=[1.5]),
        dict(residue_solver="unknown"),
        dict(screening_tol=0),
        dict(screening_tol=1),
        dict(screening_maxiter=0),
        dict(screening_restart=True),
    ],
)
def test_contour_invalid_controls(kwargs):
    obj = model()
    controls = dict(orbs=[0], nw=8)
    controls.update(kwargs)
    with pytest.raises(ValueError):
        ContourDeformation(obj, obj.e_mf[::2], **controls)


@pytest.fixture(scope="module")
def water():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df
    from pyqed.qchem.basis import PackedRIFactors

    mol = pyscf.gto.M(
        atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g", verbose=0
    )
    mf = pyscf.scf.RHF(mol)
    mf.chkfile = None
    mf.conv_tol = 1e-12
    mf.kernel()
    fitted = df.DF(mol).build()
    mol.eri_factors = PackedRIFactors(np.concatenate(list(fitted.loop())), mol.nao)
    return mf


def test_molecular_contour_qp_and_pyscf(water, monkeypatch):
    from pyscf.gw import gw_cd

    reference = g.GW(water, eta=1e-8, freq_int='exact').run()
    monkeypatch.setattr(g, "rpa", lambda *a, **k: pytest.fail("spectral screening"))
    contour = g.GW(water, freq_int="cd", eta=1e-8).run(nw=96)
    np.testing.assert_allclose(contour.e_qp, reference.e_qp, atol=2e-8, rtol=0)
    assert (
        contour.converged and contour._M is None and contour._charge_screening is None
    )
    assert np.max(contour.qp_residuals) < 1e-9
    assert np.all(contour.qp_weights > 0)
    assert contour.info["frequency_integration"] == "contour_deformation"
    pyscf_gw = gw_cd.GWCD(water)
    pyscf_gw.eta = 1e-8
    pyscf_gw.Lpq = reference._pair_factors
    # PySCF 2.14 indexes Wmn by absolute MO index: retain its full target axis.
    pyscf_gw.orbs = list(range(len(water.mo_energy)))
    pyscf_gw.nw = 96
    pyscf_gw.vhf_df = True
    pyscf_gw.qpe_tol = 1e-9
    converged, converged_orbitals, energies = gw_cd.kernel(pyscf_gw)
    assert np.all(converged_orbitals[[4, 5]])
    assert converged
    np.testing.assert_allclose(
        contour.e_qp[[4, 5]], energies[[4, 5]], atol=2e-7, rtol=0
    )


def test_selected_orbitals_and_unsupported_methods(water):
    from pyqed.gw.bse import BSE

    obj = g.GW(water, freq_int="contour", eta=1e-8).run(orbs=[5, 4], nw=32)
    assert np.array_equal(obj.orbital_indices, [5, 4])
    assert np.all(np.isnan(obj.e_qp[:4]))
    for p in [4, 5]:
        correlation, exchange = obj.sigma(2 * p, 2 * p, obj.e_qp[p])
        assert (
            abs(
                obj.e_qp[p]
                - water.mo_energy[p]
                - (correlation + exchange - obj.v_mf[2 * p, 2 * p]).real
            )
            < 1e-9
        )
    with pytest.raises(ValueError, match="orbs"):
        obj.sigma(0, 0, -0.5)
    with pytest.raises(ValueError, match="all spatial"):
        BSE(obj)
    with pytest.raises(NotImplementedError, match="G0W0"):
        obj.run(method="evgw")
    with pytest.raises(NotImplementedError, match="diagonal"):
        obj.sigma(8, 10, 0.0)
    with pytest.raises(NotImplementedError, match="TDH"):
        g.GW(water, freq_int="cd", screening="TDHF")
    assert g.GW(water, freq_int="contour").eta == 1e-6
    assert g.GW(water, freq_int='exact').eta == 1e-2


def test_matrix_free_qp_and_bse_against_direct(water):
    from pyqed.gw.bse import BSE, TDA
    references = [g.GW(water, freq_int="contour", eta=1e-6).run(
        nw=64, residue_solver=solver, screening_tol=1e-12)
        for solver in ["direct", "gmres"]]
    direct, iterative = references
    for key in ["e_qp", "qp_weights", "qp_residuals"]:
        np.testing.assert_allclose(getattr(iterative, key), getattr(direct, key),
                                   atol=1e-9, rtol=1e-9)
    assert iterative.contour_info["screening_info"]["failed_solves"] == 0
    for cls in [TDA, BSE]:
        expected, actual = [cls(gw).run(nroots=3, tol=1e-9).e for gw in references]
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def test_incremental_qp_reuses_prepared_full_screening(water, monkeypatch):
    baseline = g.GW(water, freq_int="contour", eta=1e-6).run(nw=64)
    obj = g.GW(water, freq_int="contour", eta=1e-6)
    obj._contour = ContourDeformation(obj, water.mo_energy, [3, 4, 5, 6], nw=64)
    prepared = obj._contour
    monkeypatch.setattr(prepared, "dielectric", lambda *a, **k: pytest.fail("screening rebuild"))
    # For this check use GMRES so real-frequency residues also avoid matrix builds.
    prepared.residue_solver = "gmres"
    for selected in [[4, 5], [3, 6]]:
        obj.run(orbs=selected, nw=64, reuse_screening=True, residue_solver="gmres")
        assert obj._contour is prepared
    np.testing.assert_allclose(obj.e_qp[3:], baseline.e_qp[3:], atol=1e-9)
    assert np.all(np.isnan(obj.e_qp[:3]))
    with pytest.raises(ValueError, match="Prepared contour"):
        obj.run(orbs=[2], nw=64, reuse_screening=True, residue_solver="gmres")
    with pytest.raises(ValueError, match="Prepared contour"):
        obj.run(orbs=[4], nw=32, reuse_screening=True, residue_solver="gmres")


def test_selected_exchange_diagonal_equals_full_exchange(water):
    obj = g.GW(water, freq_int="contour")
    selected = [5, 0, 4]
    expected = np.diag(g._sigma_x_matrix(obj))[2*np.asarray(selected)]
    np.testing.assert_allclose(g._sigma_x_diagonal(obj, selected), expected, atol=1e-12)


def test_independently_collected_screening_matches_serial():
    obj = model()
    serial = ContourDeformation(obj, obj.e_mf[::2], [0, 2], nw=8)
    collected = ContourDeformation(obj, obj.e_mf[::2], [0, 2], nw=8, prepare=False)
    assert not collected.prepared
    for index in reversed(range(9)):
        frequency = np.r_[0., collected.freqs][index]
        chol = g.scipy.linalg.cho_factor(collected.dielectric(frequency, imaginary=True), lower=True)
        elements = collected.screened_elements(chol, collected.orbs)
        collected.set_screening_frequency(index, elements, chol if index == 0 else None)
    assert collected.prepared
    np.testing.assert_allclose(collected.w0, serial.w0, atol=1e-13)
    np.testing.assert_allclose(collected.w_imag, serial.w_imag, atol=1e-13)
    np.testing.assert_allclose(collected.evaluate(0, -.7), serial.evaluate(0, -.7), atol=1e-13)
    with pytest.raises(ValueError, match="duplicate"):
        collected.set_screening_frequency(0, elements, chol)


def test_contiguous_target_factors_retain_full_intermediates():
    obj = model()
    targets = [0, 2]
    selected = obj._pair_factors[:, :, targets].transpose(0, 2, 1).copy()
    baseline = ContourDeformation(obj, obj.e_mf[::2], targets, nw=8)
    cached = ContourDeformation(obj, obj.e_mf[::2], targets, nw=8, target_factors=selected)
    np.testing.assert_allclose(cached.w0, baseline.w0, atol=1e-13)
    np.testing.assert_allclose(cached.w_imag, baseline.w_imag, atol=1e-13)
    for p in targets:
        np.testing.assert_allclose(cached.residue(1.13, p, 1), baseline.residue(1.13, p, 1), atol=1e-13)
        np.testing.assert_allclose(cached.evaluate(p, -.7), baseline.evaluate(p, -.7), atol=1e-13)
    with pytest.raises(ValueError, match="every intermediate"):
        ContourDeformation(obj, obj.e_mf[::2], targets, nw=8, target_factors=selected[:, :, :4])


@pytest.mark.parametrize("tda", [False, True])
@pytest.mark.parametrize("low_rank", [False, True])
def test_bse_reuses_contour_static_screening(water, monkeypatch, tda, low_rank):
    b = importlib.import_module("pyqed.gw.bse")
    spectral = g.GW(water, eta=1e-8, freq_int='exact').run()
    contour = g.GW(water, freq_int="contour", eta=1e-8).run(nw=96)
    cls = b.TDA if tda else b.BSE
    expected = cls(spectral, screening_energies='hf').run(nroots=3, low_rank=False).e

    def forbidden(*args, **kwargs):
        pytest.fail("rebuilding screening instead of reusing the static factor")

    monkeypatch.setattr(g, "rpa", forbidden)
    monkeypatch.setattr(b, "rpa", forbidden)
    monkeypatch.setattr(contour._contour, "dielectric", forbidden)
    actual = cls(contour, screening_energies='hf')
    assert actual._static_screening is contour._contour.static_screening
    actual.run(
        nroots=3,
        low_rank=low_rank,
        tol=1e-9,
        **(dict(batch_columns=2) if low_rank else {}),
    )
    np.testing.assert_allclose(actual.e, expected, atol=2e-8, rtol=0)
    assert actual._M is None and actual.e_rpa is None
    original_energy = contour._contour.energy.copy()
    contour._contour.energy[0] += 0.1
    assert cls(contour, screening_energies='hf')._static_screening is None
    contour._contour.energy[:] = original_energy
    contour._contour.mo_coeff[0, 0] += 0.1
    assert cls(contour, screening_energies='hf')._static_screening is None


def test_static_bse_blocks_against_spectral_kernel():
    b = importlib.import_module("pyqed.gw.bse")
    obj = model()
    rng = np.random.default_rng(137)
    factors = rng.normal(size=(137, 5, 5)) * 0.02
    obj._pair_factors = (factors + factors.transpose(0, 2, 1)) / 2
    poles, vectors = g.rpa(obj)
    couplings = g._charge_spatial_couplings(obj, poles, vectors)
    spectral = SimpleNamespace(
        nso=5,
        nocc=2,
        screening="TDH",
        e_qp=obj.e_mf[::2],
        e_mf=obj.e_mf[::2],
        _pair_factors=obj._pair_factors,
        _M=np.asarray(couplings),
        e_rpa=poles,
        eri=None,
    )
    a, bb = b.bse_AB_matrices(spectral)
    contour = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8)
    static = SimpleNamespace(
        **{
            **spectral.__dict__,
            "_M": None,
            "e_rpa": None,
            "_static_screening": contour.static_screening,
        }
    )
    actual_a, actual_b = b.bse_AB_matrices(static)
    np.testing.assert_allclose(actual_a, a, atol=1e-12)
    np.testing.assert_allclose(actual_b, bb, atol=1e-12)
    trials = rng.normal(size=(12, 5)) + 1j * rng.normal(size=(12, 5))
    expected = np.block([[a, bb], [-bb, -a]]) @ trials
    np.testing.assert_allclose(
        b._bse_full_matmat(static, trials, 2), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        np.column_stack([b._bse_full_matvec(static, x) for x in trials.T]),
        expected,
        atol=1e-12,
    )
    spectral._M = couplings
    np.testing.assert_allclose(
        b._bse_full_matmat(spectral, trials, 2), expected, atol=1e-12
    )


def test_blocked_mo_factor_storage(monkeypatch):
    from pyqed.qchem.basis import PackedRIFactors

    rng = np.random.default_rng(43)
    ao = rng.normal(size=(71, 6, 6))
    ao = (ao + ao.transpose(0, 2, 1)) / 2
    coefficients = np.linalg.qr(rng.normal(size=(6, 6)))[0][:, :4]
    expected = np.einsum("Pmn,mp,nq->Ppq", ao, coefficients, coefficients)
    rows, cols = np.tril_indices(6)
    packed = PackedRIFactors(ao[:, rows, cols], 6)
    monkeypatch.setattr(
        PackedRIFactors, "__array__", lambda *a, **k: pytest.fail("AO unpack")
    )
    for source in (packed, ao):
        obj = SimpleNamespace(
            _scf=SimpleNamespace(eri_factors=source), max_memory=0.001
        )
        actual = g._build_spatial_pair_factors_mo(obj, coefficients)
        assert isinstance(actual, np.memmap)
        np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_direct_screening_checks_true_residual(monkeypatch):
    obj = model()
    contour = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8)
    module = importlib.import_module('pyqed.gw.contour')
    monkeypatch.setattr(module, '_symmetric_solve', lambda matrix, rhs: np.zeros_like(rhs))
    with pytest.raises(np.linalg.LinAlgError, match='Screening direct solve failed'):
        contour.residue(.3, 0, 1)
    assert contour.screening_info['failed_solves'] == 1
    assert contour.screening_info['max_relative_residual'] == pytest.approx(1.)


def test_direct_residue_reuses_only_exact_validated_requests():
    obj = model()
    contour = ContourDeformation(obj, obj.e_mf[::2], [0], nw=8)
    expected = contour.residue(.3, 0, 1)
    solves = contour.screening_info['solves']
    np.testing.assert_array_equal(contour.residue(.3, 0, 1), expected)
    assert contour.screening_info['solves'] == solves
    assert contour.screening_info['residue_cache_hits'] == 1
    contour.residue(np.nextafter(.3, 1.), 0, 1)
    assert contour.screening_info['solves'] == solves+1
    contour.response_cache_bytes = 512
    contour.residue(.4, 0, 1)
    assert len(contour._direct_residues) == 1


def test_symmetric_driver_matches_scipy_and_rejects_singular():
    from pyqed.gw.contour import _symmetric_solve
    import scipy.linalg as la
    rng = np.random.default_rng(78)
    a = rng.normal(size=(17,17))+1j*rng.normal(size=(17,17))
    a = a+a.T
    rhs = rng.normal(size=17)
    expected = la.solve(a,rhs,assume_a='sym')
    np.testing.assert_array_equal(_symmetric_solve(a.copy(),rhs),expected)
    with pytest.raises(np.linalg.LinAlgError,match='Singular'):
        _symmetric_solve(np.zeros((3,3),dtype=complex),np.ones(3))


def test_parallel_direct_preserves_explicit_orbital_reference(water):
    engine = pytest.importorskip('pyqed.gw._contour_contractions')
    if not engine.openmp_available():
        pytest.skip('OpenMP unavailable')
    energy = water.mo_energy + np.linspace(-.01,.01,len(water.mo_energy))
    selected = [len(water.mo_occ[water.mo_occ>0])-1, len(water.mo_occ[water.mo_occ>0])]
    serial = g.GW(water,freq_int='contour',eta=1e-6,threads=1).run(
        mo_energy=energy,orbs=selected,nw=8,residue_solver='direct')
    parallel = g.GW(water,freq_int='contour',eta=1e-6,threads=4).run(
        mo_energy=energy,orbs=selected,nw=8,residue_solver='direct')
    np.testing.assert_allclose(parallel.e_qp,serial.e_qp,atol=1e-10,rtol=0,equal_nan=True)
    assert parallel.threading_info['orbital_workers']==2
