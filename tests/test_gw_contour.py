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

    reference = g.GW(water, eta=1e-8).run()
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
    converged, energies, _ = gw_cd.kernel(
        pyscf_gw,
        water.mo_energy,
        water.mo_coeff,
        Lpq=reference._pair_factors,
        orbs=[4, 5],
        nw=96,
        vhf_df=True,
    )
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
    assert g.GW(water).eta == 1e-2


@pytest.mark.parametrize("tda", [False, True])
@pytest.mark.parametrize("low_rank", [False, True])
def test_bse_reuses_contour_static_screening(water, monkeypatch, tda, low_rank):
    b = importlib.import_module("pyqed.gw.bse")
    spectral = g.GW(water, eta=1e-8).run()
    contour = g.GW(water, freq_int="contour", eta=1e-8).run(nw=96)
    cls = b.TDA if tda else b.BSE
    expected = cls(spectral).run(nroots=3, low_rank=False).e

    def forbidden(*args, **kwargs):
        pytest.fail("rebuilding screening instead of reusing the static factor")

    monkeypatch.setattr(g, "rpa", forbidden)
    monkeypatch.setattr(b, "rpa", forbidden)
    monkeypatch.setattr(contour._contour, "dielectric", forbidden)
    actual = cls(contour)
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
    assert cls(contour)._static_screening is None
    contour._contour.energy[:] = original_energy
    contour._contour.mo_coeff[0, 0] += 0.1
    assert cls(contour)._static_screening is None


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
