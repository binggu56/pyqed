"""Exact charge reduction against the unreduced spin-orbital GW equations."""
from copy import deepcopy
from types import SimpleNamespace
import importlib
import numpy as np
import pytest

g = importlib.import_module('pyqed.gw.gw')


@pytest.mark.parametrize('factorized', [False, True])
@pytest.mark.parametrize('updated_energy', [False, True])
def test_charge_screening_preserves_self_energy(factorized, updated_energy):
    rng = np.random.default_rng(84)
    f = rng.normal(size=(9, 5, 5))*.08
    f = (f+f.transpose(0, 2, 1))/2
    spatial = np.einsum('Ppq,Prs->pqrs', f, f)
    model = SimpleNamespace(nso=10, nocc=4, e_mf=np.repeat([-1., -.6, .1, .4, .9], 2),
        _qp_energy_so=None, _M=None, _spin_pair_factors=None, _sigma_x_matrix=None,
        _pair_factors=f if factorized else None,
        eri=None if factorized else g._spin_orbital_eri_from_spatial(spatial),
        mo_coeff=np.eye(5), eta=1e-3)
    if updated_energy:
        model._qp_energy_so = model.e_mf+np.repeat([-.04, -.03, .02, .03, .04], 2)
    a, b = g.rpa_AB_matrices(model)
    old_w, old_t = g._casida_eigh(a, b)
    old = deepcopy(model)
    old._M = g.get_m_rpa(old, old_w, old_t)
    w, t = g.rpa(model)
    assert len(w) == len(old_w)//4 == 6
    model._M = g.get_m_rpa(model, w, t)
    for p in range(10):
        for q in range(10):
            actual = g.sigma(model, p, q, [-1.3, -.2, .35, 1.2], w, t)
            expected = g.sigma(old, p, q, [-1.3, -.2, .35, 1.2], old_w, old_t)
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    # A new RPA call invalidates couplings belonging to the previous modes.
    g.rpa(model)
    assert model._M is None and model._charge_screening is None


@pytest.mark.parametrize('method', ['g0w0', 'evgw'])
def test_molecular_qp_and_screening_reuse(monkeypatch, method):
    pyscf = pytest.importorskip('pyscf')
    from pyqed.gw.bse import BSE, rpa as spatial_rpa, get_m_rpa as spatial_couplings
    mol = pyscf.gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                      basis='sto-3g', verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.chkfile = None
    mf.conv_tol = 1e-12
    mf.kernel()
    original = g.rpa
    def full_spin(obj, **kwargs):
        return g._casida_eigh(*g.rpa_AB_matrices(obj, method=kwargs.get('method', 'TDH')))
    monkeypatch.setattr(g, 'rpa', full_spin)
    controls = dict(max_cycle=2) if method == 'evgw' else {}
    old = g.GW(mf, ao2mofn=pyscf.ao2mo.general, eta=1e-3).run(method=method, **controls)
    monkeypatch.setattr(g, 'rpa', original)
    new = g.GW(mf, ao2mofn=pyscf.ao2mo.general, eta=1e-3).run(method=method, **controls)
    np.testing.assert_allclose(new.e_qp, old.e_qp, atol=1e-9, rtol=0)
    assert np.all(new.qp_weights > 0)
    assert np.max(new.qp_residuals) <= 1e-9
    bse = BSE(new)
    if method == 'evgw':
        # Updated-energy screening must not replace BSE's MF-energy screening.
        assert bse._M is None
        return
    assert bse._M is not None
    for orbital, qp in enumerate(new.e_qp):
        spin = 2*orbital
        correlation, exchange = g.sigma(new, spin, spin, qp,
                                        new._charge_screening['poles'], None)
        residual = qp-new.e_mf[spin]-(correlation+exchange-new.v_mf[spin, spin]).real
        assert abs(residual) <= 1e-9
    poles, vectors = spatial_rpa(bse, method='TDH')
    m = spatial_couplings(bse, poles, vectors)
    np.testing.assert_allclose(bse.e_rpa, poles, atol=1e-12)
    # Mode phases are arbitrary: compare the screened interaction itself.
    expected = np.einsum('pqL,rsL,L->pqrs', m, m, 1/poles)
    actual = np.einsum('pqL,rsL,L->pqrs', bse._M, bse._M, 1/bse.e_rpa)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError('Matching screening should be reused')
    bse.rpa = unexpected_rebuild
    bse.run(nroots=3, low_rank=True, batch_columns=8, tol=1e-9)
    reference = BSE(old).run(nroots=3, low_rank=False)
    np.testing.assert_allclose(bse.e, reference.e, atol=1e-9)
    new._charge_screening['mo_coeff'][0, 0] += .1
    assert BSE(new)._M is None
    new._charge_screening['mo_coeff'] = new.mo_coeff.copy()
    new._charge_screening['energy'][0] += .1
    assert BSE(new)._M is None


def test_charge_screening_rejects_nonpositive_gaps():
    obj = SimpleNamespace(nocc=2, _qp_energy_so=None, e_mf=np.repeat([1., 0.], 2))
    with pytest.raises(np.linalg.LinAlgError, match='positive'):
        g._charge_rpa(obj)


def test_qp_continuation_rejects_negative_weight_crossing():
    correction = lambda w: 1/(w+.1j)
    derivative = lambda w: -1/(w+.1j)**2
    root = g._continue_qp_root(correction, derivative, .2)
    assert root > 1.
    assert abs(root-.2-correction(root).real) < 1e-9
    assert 1-derivative(root).real > 0


def test_qp_continuation_constant_and_failure():
    assert g._continue_qp_root(lambda w: .3, lambda w: 0., -1.) == pytest.approx(-.7)
    with pytest.raises(RuntimeError, match='positive-weight'):
        g._continue_qp_root(lambda w: np.nan, lambda w: np.nan, 0.)


def test_qp_bracket_recovers_when_newton_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('force bracketed solve')
    monkeypatch.setattr(g, 'newton', fail)
    correction = lambda w: 1/(w+.1j)
    derivative = lambda w: -1/(w+.1j)**2
    root = g._continue_qp_root(correction, derivative, .2)
    assert root > 1.
    assert abs(root-.2-correction(root).real) < 1e-9
    assert 1-derivative(root).real > 0
