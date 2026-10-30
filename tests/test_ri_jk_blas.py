"""Blocked packed RI exchange against full-density and scalar references."""
import numpy as np
import pytest
from pyqed.qchem import basis


@pytest.mark.parametrize('nao,naux', [(0, 3), (1, 0), (1, 4), (8, 7)])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
@pytest.mark.parametrize('complex_values', [False, True])
def test_pair_unpack_matches_scatter(monkeypatch, nao, naux, layout, complex_values):
    rng = np.random.default_rng(920)
    packed = rng.normal(size=(naux, nao*(nao+1)//2))
    if complex_values:
        packed = packed+1j*rng.normal(size=packed.shape)
    if layout == 'F':
        packed = np.asfortranarray(packed)
    elif layout == 'strided':
        packed = np.repeat(packed, 2, axis=1)[:, ::2]
    packed.setflags(write=False)
    original = packed.copy()
    actual = basis._pair_factors_to_full(packed, nao)
    monkeypatch.setattr(basis, '_basis_cy', None)
    expected = basis._pair_factors_to_full(packed, nao)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(packed, original)
    assert actual.flags.c_contiguous


def test_pair_unpack_rejects_invalid_shape():
    if not hasattr(basis._basis_cy, "unpack_pair_factors"):
        pytest.skip("Optional pair-unpacking accelerator is unavailable")
    with pytest.raises(ValueError, match='inconsistent'):
        basis._basis_cy.unpack_pair_factors(np.ones((2, 4)), 3)
    with pytest.raises(ValueError, match='inconsistent'):
        basis._basis_cy.unpack_pair_factors(np.ones((2, 0)), -1)


@pytest.mark.parametrize('block', [1, 4, None])
@pytest.mark.parametrize('occupied', [0, 1, 5])
def test_blocked_ri_jk(monkeypatch, block, occupied):
    monkeypatch.delenv('PYQED_QCHEM_CPP_RI_JK', raising=False)
    rng = np.random.default_rng(410)
    nao, naux = 8, 11
    packed = rng.normal(size=(naux, nao*(nao+1)//2))
    coeff, _ = np.linalg.qr(rng.normal(size=(nao, nao)))
    occ = np.zeros(nao)
    occ[:occupied] = np.linspace(.2, 2., occupied)
    density = (coeff*occ)@coeff.T
    expected = basis.contract_jk_ri(basis.PackedRIFactors(packed, nao), density, nao)
    scalar = basis._integrals_cpp.contract_jk_ri_occ_packed(packed, coeff, occ, nao, 1)

    class ForbiddenScalar:
        def contract_jk_ri_occ_packed(self, *args):
            pytest.fail('Default contraction must use blocked BLAS')

    monkeypatch.setattr(basis, '_integrals_cpp', ForbiddenScalar())
    actual = basis._contract_jk_ri_occ_packed(packed, coeff, occ, nao, block_size=block)
    for result, reference, compiled in zip(actual, expected, scalar):
        np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result, compiled, atol=1e-12, rtol=1e-12)


def test_occupied_exchange_accumulates_in_place(monkeypatch):
    rng = np.random.default_rng(413)
    n, p, o = 41, 5, 11
    packed = rng.normal(size=(p, n*(n+1)//2))
    coeff, _ = np.linalg.qr(rng.normal(size=(n, n)))
    occ = np.zeros(n)
    occ[:o] = np.linspace(.2, 2., o)
    expected = basis.contract_jk_ri(basis.PackedRIFactors(packed, n), (coeff*occ)@coeff.T, n)
    original = basis.dsyrk
    panels = []
    def accumulate(*args, **kwargs):
        result = original(*args, **kwargs)
        assert np.shares_memory(result, kwargs['c'])
        panels.append(args[1].shape[0])
        return result
    monkeypatch.delenv('PYQED_QCHEM_CPP_RI_JK', raising=False)
    monkeypatch.setattr(basis, 'dsyrk', accumulate)
    actual = basis._contract_jk_ri_occ_packed(packed, coeff, occ, n, block_size=2)
    assert panels == [2*o, 2*o, o]
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('n,o,p', [(0, 0, 3), (1, 1, 0), (1, 1, 4), (17, 0, 3), (17, 5, 7), (259, 19, 3)])
@pytest.mark.parametrize('workers', [1, 4])
def test_shared_occupied_panels_against_density_reference(n, o, p, workers):
    rng = np.random.default_rng(419)
    pairs = rng.normal(size=(p, n*(n+1)//2))
    c = rng.normal(size=(n, o))
    expected = basis.contract_jk_ri(basis.PackedRIFactors(pairs, n), c@c.T, n)
    actual = basis._integrals_cpp.contract_jk_ri_occupied(
        pairs, c, workers, basis._ri_blas.__pyx_capi__['dgemm'],
        basis._ri_blas.__pyx_capi__['dsyrk'])
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=2e-9, rtol=2e-12)
    np.testing.assert_array_equal(actual[1], actual[1].T)


def test_large_occupied_contraction_uses_shared_panels(monkeypatch):
    rng = np.random.default_rng(420)
    n, p = 259, 3
    pairs = rng.normal(size=(p, n*(n+1)//2))
    c, _ = np.linalg.qr(rng.normal(size=(n, n)))
    occ = np.zeros(n)
    occ[:19] = np.linspace(.1, 2., 19)
    reference = basis._contract_jk_ri_occ_packed(pairs, c, occ, n, block_size=1)
    def forbidden(*args, **kwargs):
        pytest.fail('Expanded one-factor buffers should be shared inside the row kernel')
    monkeypatch.setattr(basis, '_pair_factors_to_full', forbidden)
    actual = basis._contract_jk_ri_occ_packed(pairs, c, occ, n, workers=4)
    for result, expected in zip(actual, reference):
        np.testing.assert_allclose(result, expected, atol=2e-11, rtol=2e-12)


def test_shared_panels_at_tdi_ao_and_occupation_dimensions():
    rng = np.random.default_rng(421)
    n, p, o = 1298, 4, 271
    pairs = rng.normal(scale=.01, size=(p, n*(n+1)//2))
    c, _ = np.linalg.qr(rng.normal(size=(n, o)))
    occ = np.linspace(.5, 2., o)
    reference = basis._contract_jk_ri_occ_packed(pairs, c, occ, n, block_size=1)
    actual = basis._contract_jk_ri_occ_packed(pairs, c, occ, n, workers=4)
    for result, expected in zip(actual, reference):
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=2e-12)
