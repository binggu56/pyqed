"""Molecular AC against independent PySCF imaginary-axis integration."""

import importlib
from types import SimpleNamespace
import numpy as np
import pytest
from test_gw_contour import model
from pyqed.gw.ac import kernel, _twopole
from pyqed.gw.contour import ContourDeformation


def test_ac_sigma_against_pyscf_and_threading():
    pyscf_ac = pytest.importorskip("pyscf.gw.gw_ac")
    obj = model()
    obj.v_mf = np.zeros((10, 10))
    obj.max_memory = 10
    obj.threads = 1
    energy = obj.e_mf[::2]
    roots = kernel(obj, energy, obj.mo_coeff, nw=32)
    sigma = obj.ac_sigma.copy()
    contour = ContourDeformation(obj, energy, range(5), nw=32, prepare=False)
    ef = (energy[1] + energy[2]) / 2
    ref = SimpleNamespace(nocc=2, rdm=False, _scf=SimpleNamespace(), verbose=0)
    expected, omega = pyscf_ac.get_sigma(
        ref,
        list(range(5)),
        obj._pair_factors,
        contour.freqs,
        contour.weights,
        ef,
        energy,
        mo_coeff=obj.mo_coeff,
        mo_occ=np.array([2, 2, 0, 0, 0]),
        iw_cutoff=5.0,
        eval_freqs=np.r_[0.0, contour.freqs],
    )
    expected = np.asarray(expected)
    expected[:2] = expected[:2].conj()
    np.testing.assert_allclose(sigma, expected, atol=2e-13)
    signed = np.broadcast_to(omega-ef, obj.ac_omega.shape).copy()
    signed[:2] = signed[:2].conj()
    np.testing.assert_allclose(obj.ac_omega, signed, atol=1e-14)
    obj.threads = 8
    parallel = kernel(obj, energy, obj.mo_coeff, nw=32)
    np.testing.assert_allclose(obj.ac_sigma, sigma, atol=2e-13)
    np.testing.assert_allclose(parallel, roots, atol=2e-8)
    assert np.max(obj.qp_residuals) < 1e-9
    assert np.all(obj.qp_weights > 0)
    subset = kernel(obj, energy, obj.mo_coeff, orbs=[3, 0], nw=32)
    np.testing.assert_allclose(subset[[3, 0]], roots[[3, 0]], atol=2e-8)
    assert np.isnan(subset[[1, 2, 4]]).all()
    assert obj._ac_screening.static_screening is not None


def test_ac_controls():
    obj = model()
    obj.v_mf = np.zeros((10, 10))
    with pytest.raises(ValueError, match="six evaluation"):
        kernel(obj, obj.e_mf[::2], obj.mo_coeff, iw_cutoff=1e-10, ac="twopole")
    with pytest.raises(ValueError, match="distinct"):
        kernel(obj, obj.e_mf[::2], obj.mo_coeff, orbs=[1, 1])
    g = importlib.import_module("pyqed.gw.gw")
    assert g._canonical_freq_int("ac") == "analytic_continuation"


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("selection", ["consecutive", "reordered"])
def test_projection_packing_and_bounded_scratch(monkeypatch, layout, selection):
    # Large enough to require multiple projection blocks at the 1 MB minimum,
    # including a partial final block. Skip fitting to isolate integration.
    module = importlib.import_module("pyqed.gw.ac")
    monkeypatch.setattr(module, "refit", lambda obj, *a, **kw: obj.ac_sigma.copy())
    rng = np.random.default_rng(42)
    nmo, rank, nocc = 48, 32, 12
    factors = rng.normal(scale=0.02, size=(rank, nmo, nmo))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    energy = np.r_[np.linspace(-1.2, -0.2, nocc), np.linspace(0.1, 2, nmo-nocc)]
    obj = model()
    obj.nso, obj.nocc = 2*nmo, 2*nocc
    obj.e_mf, obj.mo_coeff = np.repeat(energy, 2), np.eye(nmo)
    obj.v_mf = np.zeros((2*nmo, 2*nmo))
    obj._pair_factors = factors.copy(order=layout)
    obj.max_memory = 10
    reference = kernel(obj, energy, obj.mo_coeff, nw=16, ac="twopole")
    ids = np.arange(2, 45)
    if selection == "reordered":
        rng.shuffle(ids)
    for memory in [10, 1]:
        obj.max_memory = memory
        actual = kernel(obj, energy, obj.mo_coeff, orbs=ids, nw=16, ac="twopole")
        np.testing.assert_allclose(actual, reference[ids], rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(obj._pair_factors, factors)


from test_gw_contour import water


def test_molecular_ac_driver_and_bse(water):
    from pyqed.gw.gw import GW
    from pyqed.gw.bse import BSE

    with pytest.warns(RuntimeWarning, match="direct contour"):
        obj = GW(water, vhf_df=True, freq_int="ac", threads=8).run(nw=64, on_unstable="contour")
    assert obj.converged
    assert obj.ac_info["method"] == "pade"
    bse = BSE(obj, screening_energies='hf')
    assert bse._static_screening is obj._ac_screening.static_screening
    reference = GW(
        water, freq_int="contour", mo_pair_factors=obj._pair_factors, eta=1e-8
    ).run(nw=64, orbs=[4, 5])
    np.testing.assert_allclose(obj.e_qp[[4, 5]], reference.e_qp[[4, 5]], atol=2e-3)
    for p in (4, 5):
        correlation, exchange = obj.sigma(2 * p, 2 * p, obj.e_qp[p])
        assert (
            abs(
                obj.e_qp[p]
                - water.mo_energy[p]
                - (correlation + exchange - obj.v_mf[2 * p, 2 * p]).real
            )
            < 1e-9
        )
    with pytest.raises(NotImplementedError, match="G0W0"):
        obj.run(method="evgw")


def test_ac_linearized_and_explicit_root_failure(monkeypatch):
    from pyqed.gw.gw import _sigma_x_diagonal

    obj = model()
    obj.v_mf = np.zeros((10, 10))
    energy = obj.e_mf[::2]
    linear = kernel(
        obj, energy, obj.mo_coeff, nw=32, orbs=[1, 2], linearized=True, ac="twopole"
    )
    static = _sigma_x_diagonal(obj, [1, 2])
    for j, p in enumerate([1, 2]):
        sigma = _twopole(
            energy[p] - obj.ac_info["fermi_level"], obj.ac_coeff[:, j]
        ).real
        assert linear[p] == pytest.approx(
            energy[p] + obj.qp_weights[p] * (sigma + static[j])
        )
    module = importlib.import_module("pyqed.gw.ac")

    def fail(*args, **kwargs):
        raise RuntimeError("root did not converge")

    monkeypatch.setattr(module, "newton", fail)
    with pytest.raises(RuntimeError, match="root did not converge"):
        kernel(obj, energy, obj.mo_coeff, nw=8, orbs=[1], ac="twopole")


def test_pade_rational_interpolation_and_derivative():
    from pyqed.gw.ac import _pade, _pade_coefficients, _pade_indices

    reference = pytest.importorskip("pyscf.gw.utils.ac_grid")
    nodes = 1j * np.linspace(0.03, 3.0, 6)
    f = lambda z: 0.2 / (z + 1.3) + 0.13 / (z - 2.1) + 0.08 / (z - 3.2)
    df = lambda z: -0.2 / (z + 1.3) ** 2 - 0.13 / (z - 2.1) ** 2 - 0.08 / (z - 3.2) ** 2
    coeff = _pade_coefficients(f(nodes), nodes)
    np.testing.assert_allclose(
        coeff, reference.thiele_ndarray(f(nodes), nodes), atol=1e-13
    )
    np.testing.assert_allclose(_pade(nodes, coeff, nodes)[0], f(nodes), atol=1e-13)
    z = np.array([-0.8, -0.4, 0.4, 0.8])
    value, derivative = _pade(z, coeff, nodes)
    np.testing.assert_allclose(value, f(z), atol=1e-10)
    np.testing.assert_allclose(derivative, df(z), atol=1e-9)
    np.testing.assert_array_equal(
        _pade_indices(52, 18, 2 / 3), reference._get_ac_idx(52, 18, 2 / 3)
    )
    np.testing.assert_allclose(
        _pade(z, _pade_coefficients(np.ones(6), nodes), nodes)[0], 1.0
    )
    with pytest.raises(ValueError, match="even"):
        _pade_indices(52, 17, 2 / 3)
    with pytest.raises(ValueError, match="frequency count"):
        _pade_indices(18, 18, 2 / 3)
    with pytest.raises(ValueError, match="distinct"):
        _pade_coefficients(np.ones(3), np.zeros(3))


def test_pade_refit_never_screens(monkeypatch):
    from pyqed.gw.ac import refit, evaluate_fit

    obj = model()
    obj.v_mf = np.zeros((10, 10))
    energy = obj.e_mf[::2]
    kernel(obj, energy, obj.mo_coeff, nw=64, orbs=[1, 2])
    sigma = obj.ac_sigma.copy()
    static = obj.ac_static.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("Refitting must not run screening")

    monkeypatch.setattr(ContourDeformation, "dielectric", forbidden)
    root = refit(obj, energy, pade_npts=18)
    assert obj.converged and np.isnan(root[[0, 3, 4]]).all()
    np.testing.assert_array_equal(obj.ac_sigma, sigma)
    np.testing.assert_array_equal(obj.ac_static, static)
    for j, p in enumerate([1, 2]):
        assert (
            abs(root[p] - energy[p] - evaluate_fit(obj, j, root[p])[0].real - static[j])
            < 1e-9
        )
    assert np.max(obj.qp_residuals[[1, 2]]) < 1e-9


def test_pade_public_driver(water):
    from pyqed.gw.gw import GW
    from pyqed.gw.ac import refit

    obj = GW(water, vhf_df=True, freq_int="ac", threads=1).run(nw=32, orbs=[4, 5])
    original = obj.e_qp.copy()
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
    refit(obj, water.mo_energy, ac="twopole")
    refit(obj, water.mo_energy, ac="pade")
    np.testing.assert_allclose(obj.e_qp, original, atol=1e-12)


def test_pade_rejects_singular_data():
    from pyqed.gw.ac import _pade_coefficients

    with pytest.raises(np.linalg.LinAlgError, match="Singular"):
        _pade_coefficients([0.0, 1.0, 2.0], [1j, 2j, 3j])


def test_molecular_ac_defaults_match_pyscf():
    import inspect

    defaults = inspect.signature(kernel).parameters
    expected = dict(
        ac="pade",
        nw=100,
        iw_cutoff=5.0,
        pade_npts=18,
        pade_step_ratio=2 / 3,
        linearized=False,
        tol=1e-6,
        maxiter=100,
    )
    for name, value in expected.items():
        assert defaults[name].default == value


def test_tetracene_saved_fit_stability():
    from pathlib import Path
    from types import SimpleNamespace
    from pyqed.gw.ac import refit
    saved = np.load(Path(__file__).parent/'data/tetracene_ac_stability.npz')
    obj = SimpleNamespace(ac_sigma=saved['sigma'], ac_omega=saved['omega'],
        ac_static=saved['static'], orbital_indices=saved['ids'], nocc=120,
        ac_info={'fermi_level':float(saved['ef'])})
    with pytest.raises(RuntimeError, match='Unstable Padé'):
        refit(obj, saved['energy'], tol=1e-7)
    assert not obj.converged
    assert {13, 239}.issubset(obj.ac_info['stability']['unstable_orbitals'])
    assert np.isnan(obj.e_qp[[13, 239]]).all()
    assert np.isfinite(obj.e_qp[[59, 60]]).all()
    assert obj.ac_info['stability']['passed'] is False
    refit(obj, saved['energy'], tol=1e-7, stability_tol=None)
    assert obj.converged and np.isfinite(obj.e_qp[saved['ids']]).all()
    assert obj.ac_info['stability']['checked'] is False
    for bad in [-1, 0, np.nan, True]:
        with pytest.raises(ValueError, match='stability_tol'):
            refit(obj, saved['energy'], stability_tol=bad)


def test_contour_repair_self_energy_and_failure(water, monkeypatch):
    from pyqed.gw.gw import GW
    from pyqed.gw.ac import refit, repair_continuation, evaluate_fit
    obj = GW(water, vhf_df=True, freq_int='ac', threads=1).run(nw=64, stability_tol=None)
    with pytest.raises(RuntimeError, match='Unstable Padé'):
        refit(obj, water.mo_energy)
    ids = obj.orbital_indices.copy()
    valid = obj.e_qp.copy()
    with pytest.warns(RuntimeWarning, match='direct contour'):
        repair_continuation(obj, water.mo_energy, nw=64, tol=1e-8)
    assert obj.converged and obj.ac_info['contour_orbitals']==[0]
    np.testing.assert_array_equal(obj.orbital_indices, ids)
    np.testing.assert_array_equal(obj.e_qp[1:], valid[1:])
    value, derivative = evaluate_fit(obj, 0, obj.e_qp[0])
    assert abs(obj.e_qp[0]-water.mo_energy[0]-value.real-obj.ac_static[0]) < 1e-8
    assert obj.qp_weights[0] == pytest.approx(1/(1-derivative.real))
    audit = obj.ac_info['contour_repair']
    assert audit['passed'] and audit['eta'] < 1e-6
    assert audit['weak_root_orbitals'] == []
    assert audit['nw'] >= 128
    assert obj.eta == .005  # AC broadening must not set the contour regulator.
    reference = GW(water, vhf_df=True, freq_int='contour', mo_pair_factors=obj._pair_factors,
                   eta=audit['eta']).run(nw=audit['nw'], orbs=[0], tol=1e-8)
    np.testing.assert_allclose(obj.e_qp[0], reference.e_qp[0], atol=1e-10, rtol=0)
    exact = GW(water, vhf_df=True, freq_int='exact', mo_pair_factors=obj._pair_factors, eta=0).run()
    np.testing.assert_allclose(obj.e_qp[0], exact.e_qp[0], atol=1e-8, rtol=0)
    with pytest.raises(ValueError, match='real frequencies'):
        evaluate_fit(obj, 0, 1j)
    with pytest.raises(RuntimeError, match='Unstable Padé'):
        refit(obj, water.mo_energy)
    assert obj._ac_contour is None and 'contour_orbitals' not in obj.ac_info
    assert 'contour_repair' not in obj.ac_info
    module = importlib.import_module('pyqed.gw.contour')
    def fail(*args, **kwargs):
        raise RuntimeError('contour failure')
    monkeypatch.setattr(module, 'kernel', fail)
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match='contour failure'):
        repair_continuation(obj, water.mo_energy)
    assert not obj.converged and np.isnan(obj.e_qp[0])


def test_corrected_reference_evaluator():
    from benchmarks.analyze_matched_gw_bse import corrected_pade
    from pyqed.gw.ac import _pade_coefficients
    nodes = np.array([1j, 2j])
    values = 1/(1+nodes)
    coefficients = _pade_coefficients(values, nodes)
    np.testing.assert_allclose(corrected_pade(nodes, nodes, coefficients), values, atol=1e-15)
    matrix_coefficients = np.stack((coefficients, coefficients.copy()), axis=1)
    matrix_coefficients[0,1] *= 2
    np.testing.assert_allclose(corrected_pade(nodes, nodes, matrix_coefficients),
                               np.column_stack((values, 2*values)), atol=1e-15)


@pytest.mark.parametrize("drift", ["quadrature", "eta"])
def test_contour_repair_rejects_unconverged_refinement(water, monkeypatch, drift):
    from pyqed.gw.gw import GW
    from pyqed.gw.ac import refit, repair_continuation
    obj = GW(water, vhf_df=True, freq_int='ac', threads=1).run(nw=32, stability_tol=None)
    with pytest.raises(RuntimeError, match='Unstable Padé'):
        refit(obj, water.mo_energy)
    module = importlib.import_module('pyqed.gw.contour')
    original = module.kernel
    def drifting(*args, **kwargs):
        energy = original(*args, **kwargs)
        energy[0] += kwargs['nw'] * 1e-3 if drift == 'quadrature' else args[0].eta * 1e5
        return energy
    monkeypatch.setattr(module, 'kernel', drifting)
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match='quadrature/eta'):
        repair_continuation(obj, water.mo_energy, nw=16, max_refinements=1)
    assert not obj.converged and np.isnan(obj.e_qp[0])
    assert np.isnan(obj.qp_weights[0]) and np.isnan(obj.qp_residuals[0])
    assert obj._ac_contour is None
    assert not obj.ac_info['contour_repair']['passed']
    assert obj.ac_info['contour_repair']['history']


@pytest.mark.parametrize('controls', [{'eta': -1}, {'eta': float('nan')},
    {'accuracy_tol': 0}, {'accuracy_tol': True}, {'max_refinements': 0},
    {'max_refinements': 1.5}])
def test_contour_repair_invalid_accuracy_controls(controls):
    from pyqed.gw.ac import repair_continuation
    with pytest.raises(ValueError):
        repair_continuation(None, None, **controls)


def test_contour_repair_reports_weak_roots(water, monkeypatch):
    from pyqed.gw.gw import GW
    from pyqed.gw.ac import refit, repair_continuation
    obj = GW(water, vhf_df=True, freq_int='ac', threads=1).run(nw=32, stability_tol=None)
    with pytest.raises(RuntimeError, match='Unstable Padé'):
        refit(obj, water.mo_energy)
    module = importlib.import_module('pyqed.gw.contour')
    original = module.kernel
    def weak_root(*args, **kwargs):
        energy = original(*args, **kwargs)
        args[0].qp_weights[0] = 1e-4
        return energy
    monkeypatch.setattr(module,'kernel',weak_root)
    with pytest.warns(RuntimeWarning) as caught:
        repair_continuation(obj,water.mo_energy,nw=32)
    assert any('spectral weight below 0.01' in str(w.message) for w in caught)
    audit = obj.ac_info['contour_repair']
    assert audit['passed'] and obj.converged
    assert audit['weak_root_orbitals'] == [0]
    assert audit['weak_root_weight_threshold'] == .01
    assert obj.qp_weights[0] == 1e-4
