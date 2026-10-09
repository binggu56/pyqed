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
            spin_couplings, model._M = model._M, None
            reduced = g.sigma(model, p, q, [-1.3, -.2, .35, 1.2], w, t)
            np.testing.assert_allclose(reduced, expected, atol=2e-11, rtol=2e-11)
            assert model._M is None
            model._M = spin_couplings
    # A new RPA call invalidates couplings belonging to the previous modes.
    g.rpa(model)
    assert model._M is None and model._charge_screening is None


@pytest.mark.parametrize('method', ['g0w0', 'evgw'])
@pytest.mark.parametrize('factorized', [False, True])
def test_molecular_qp_and_screening_reuse(monkeypatch, method, factorized):
    pyscf = pytest.importorskip('pyscf')
    from pyqed.gw.bse import BSE, rpa as spatial_rpa, get_m_rpa as spatial_couplings
    mol = pyscf.gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                      basis='sto-3g', verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.chkfile = None
    mf.conv_tol = 1e-12
    mf.kernel()
    if factorized:
        from pyscf import df
        from pyqed.qchem.basis import PackedRIFactors
        fitted = df.DF(mol).build()
        mol.eri_factors = PackedRIFactors(np.concatenate(list(fitted.loop())), mol.nao)
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


def test_charge_casida_in_place_and_blocked_couplings(monkeypatch):
    rng = np.random.default_rng(17)
    f = rng.normal(size=(13, 7, 7))*.05
    f = (f+f.transpose(0, 2, 1))/2
    original = f.copy()
    energy = np.array([-1., -.7, -.4, .2, .5, .8, 1.1])
    obj = SimpleNamespace(nocc=6, nso=14, e_mf=np.repeat(energy, 2),
                          _qp_energy_so=None, _pair_factors=f,
                          max_memory=.001, mo_coeff=np.eye(7))
    gaps = (energy[3:]-energy[:3, None]).ravel()
    pairs = f[:, :3, 3:].reshape(13, -1)
    expected = 4*np.sqrt(gaps[:, None]*gaps[None, :])*(pairs.T @ pairs)
    expected[np.diag_indices(len(gaps))] += gaps**2
    eigh = g.scipy.linalg.eigh
    def consume(matrix, **kwargs):
        assert matrix.flags.f_contiguous and kwargs['overwrite_a']
        np.testing.assert_allclose(matrix, expected, atol=1e-14)
        return eigh(matrix, **kwargs)
    monkeypatch.setattr(g.scipy.linalg, 'eigh', consume)
    poles, vectors = g._charge_rpa(obj)
    np.testing.assert_allclose(expected @ vectors, vectors*poles**2, atol=1e-14)
    reference = np.einsum('Ppq,Pl->pql', f,
        pairs @ (vectors*np.sqrt(gaps)[:, None]/np.sqrt(poles)), optimize=True)
    actual = g._charge_spatial_couplings(obj, poles, vectors)
    np.testing.assert_allclose(actual, reference, atol=1e-14)
    np.testing.assert_array_equal(f, original)


def test_packed_transform_and_disk_couplings(monkeypatch):
    from pyqed.qchem.basis import PackedRIFactors
    from pyqed.gw.bse import _get_mo_pair_factors, get_m_rpa as bse_couplings
    rng = np.random.default_rng(28)
    f = rng.normal(size=(7, 5, 5))*.04
    f = (f+f.transpose(0, 2, 1))/2
    rows, cols = np.tril_indices(5)
    packed = PackedRIFactors(f[:, rows, cols], 5)
    obj = SimpleNamespace(_scf=SimpleNamespace(eri_factors=packed),
        mol=SimpleNamespace(), nso=10, nocc=4, max_memory=.0001,
        e_mf=np.repeat([-1., -.6, .1, .4, .9], 2), _qp_energy_so=None,
        _M=None, _spin_pair_factors=None, _sigma_x_matrix=None,
        mo_coeff=np.eye(5), eta=1e-3, eri=None)
    monkeypatch.setattr(PackedRIFactors, '__array__',
                        lambda *a, **k: pytest.fail('full AO factor expansion'))
    obj._pair_factors = g._build_spatial_pair_factors_mo(obj, obj.mo_coeff)
    np.testing.assert_allclose(obj._pair_factors, f, atol=1e-14)
    obj._scf.mol = obj.mol
    np.testing.assert_allclose(_get_mo_pair_factors(obj._scf, obj.mo_coeff), f, atol=1e-14)
    poles, vectors = g.rpa(obj)
    obj._M = g.get_m_rpa(obj, poles, vectors)
    assert isinstance(obj._M, np.memmap)
    assert isinstance(obj._charge_screening['couplings'], g.FactorizedCouplings)
    assert isinstance(obj._charge_screening['couplings'].projection, np.memmap)
    reference = deepcopy(obj)
    reference.max_memory = 4000
    reference._M = g.get_m_rpa(reference, poles, vectors)
    np.testing.assert_allclose(obj._M, reference._M, atol=1e-14)
    np.testing.assert_allclose(g.sigma(obj, 2, 2, [-.7, .3], poles, vectors),
                               g.sigma(reference, 2, 2, [-.7, .3], poles, vectors))
    assert obj._spin_pair_factors is None
    spatial = SimpleNamespace(nso=5, nocc=2, e_mf=obj.e_mf[::2],
        eri=None, _pair_factors=f, max_memory=.0001)
    actual = bse_couplings(spatial, poles, vectors)
    assert isinstance(actual, np.memmap)
    np.testing.assert_allclose(actual, obj._charge_screening['couplings'], atol=1e-14)


def test_dense_integral_budget_checked_before_transform(monkeypatch):
    obj = SimpleNamespace(_scf=SimpleNamespace(), mol=SimpleNamespace(),
                          nso=40, max_memory=1., v_mf=np.zeros((40, 40)))
    monkeypatch.setattr(g, '_build_spatial_eri_mo',
                        lambda *args: pytest.fail('oversized dense transformation'))
    with pytest.raises(MemoryError, match='CD/RI'):
        g._set_rhf_orbitals(obj, np.arange(20.), np.eye(20))


def test_factorized_screening_tda_without_materialization(monkeypatch):
    from pyqed.gw.bse import _bse_tda_matmat
    rng = np.random.default_rng(51)
    f = rng.normal(size=(137, 7, 7))*.03
    f = (f+f.transpose(0, 2, 1))/2
    projection = rng.normal(size=(137, 11))*.02
    poles = np.linspace(.2, 2., 11)
    couplings = g.FactorizedCouplings(f, projection, poles)
    dense = np.asarray(couplings)
    for key in [(slice(None), 2, slice(None)), (1, slice(2, 6), slice(3, 8)), (2, 4, 6)]:
        np.testing.assert_allclose(couplings[key], dense[key], atol=1e-14)
    obj = SimpleNamespace(nso=7, nocc=3, e_mf=np.arange(7.)*.3,
                          e_qp=None, _M=dense, e_rpa=poles, _pair_factors=f)
    vectors = rng.normal(size=(12, 5))+1j*rng.normal(size=(12, 5))
    expected = _bse_tda_matmat(obj, vectors, 2)
    obj._M = couplings
    monkeypatch.setattr(g.FactorizedCouplings, '__array__',
                        lambda *a, **k: pytest.fail('full spectral tensor materialized'))
    np.testing.assert_allclose(_bse_tda_matmat(obj, vectors, 2), expected,
                               atol=1e-12, rtol=1e-12)
    assert couplings.factors is f and couplings.projection is projection


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
