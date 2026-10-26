import numpy as np
import pytest
from pyqed.qchem import Molecule
from pyqed.qchem.eri_response import CoulombIntegrals


@pytest.mark.parametrize('basis_name', ['sto-3g', '6-31g*'])
@pytest.mark.parametrize('cart', [False, True])
def test_early_rys_and_threads_match_columns(basis_name, cart):
    mol = Molecule(atom='O 0 0 0; H .2 1.4 1.1; H -.1 -1.4 1.2',
                   unit='bohr', basis=basis_name).build(eri='cd', options={
                       'low_rank_max_rank': 3,
                       'coord_type': 'cartesian' if cart else 'spherical'})
    response = CoulombIntegrals(mol)
    rng = np.random.default_rng(117)
    directions = rng.normal(size=(2, 3, 3))
    columns = response._cd_columns(directions, workers=1)
    sensitivity = rng.normal(size=(2, *columns.shape[1:]))
    expected = np.einsum('mPp,oPp->mo', columns, sensitivity)
    for kernel in ('recurrence', 'rys'):
        for workers in (1, 2, 4):
            actual = response._cd_columns(directions, workers=workers,
                                           sensitivities=sensitivity, kernel=kernel)
            np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=2e-10)
            if kernel == 'recurrence':
                assert response.derivative_info['plan_hits'] > 0
                assert response.derivative_info['plan_misses'] > 0
    late = response._cd_columns(directions, workers=1, sensitivities=sensitivity,
                                early_contraction=False)
    np.testing.assert_allclose(late, expected, atol=3e-10, rtol=2e-10)
    rys_columns = response._cd_columns(directions, workers=2, kernel='rys')
    np.testing.assert_allclose(rys_columns, columns, atol=3e-10, rtol=2e-10)


def test_derivative_screening_bound():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4; H 0 0 12; H 0 0 13.4',
                   unit='bohr', basis='sto-3g').build(eri='cd')
    response = CoulombIntegrals(mol)
    rng = np.random.default_rng(14)
    directions = rng.normal(size=(3, 4, 3))
    columns = response._cd_columns(directions, workers=1)
    sensitivities = rng.normal(size=(2, *columns.shape[1:]))
    reference = np.einsum('mPp,oPp->mo', columns, sensitivities)
    counts = []
    for tolerance in (1e-10, 1e-4, 100.):
        value = response._cd_columns(directions, workers=2, sensitivities=sensitivities,
                                     screen_tol=tolerance, kernel='rys')
        info = response.derivative_info
        assert np.max(abs(value-reference)) <= info['error_bound']+1e-12
        assert info['error_bound'] <= tolerance
        counts.append(info['screened'])
    assert counts[-1] > 0
    assert counts == sorted(counts)
    with pytest.raises(ValueError):
        response._cd_columns(directions, screen_tol=1e-6)
    for kwargs in ({'workers': 0}, {'kernel': 'bad'}, {'screen_tol': -1}):
        with pytest.raises(ValueError):
            response._cd_columns(directions, **kwargs)


def test_rys_four_p_centers_and_zero_translation():
    mol = Molecule(atom='N 0 0 0; N .2 .3 2.1', unit='bohr', basis='sto-3g').build(eri='cd')
    data = mol._cd_factor_data
    data['pivots'] = np.arange(data['nao']*(data['nao']+1)//2)
    response = CoulombIntegrals(mol)
    directions = np.array([[[.1, -.2, .3], [.4, .2, -.1]], [[1., 1., 1.], [1., 1., 1.]]])
    reference = response._cd_columns(directions, workers=1)
    actual = response._cd_columns(directions, workers=2, kernel='rys')
    np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-10)
    np.testing.assert_array_equal(actual[1], 0.)




def test_contracted_recurrence_sparse_and_zero_observables():
    mol = Molecule(atom='N 0 0 0; N .2 .3 2.1', unit='bohr',
                   basis='sto-3g').build(eri='cd')
    response = CoulombIntegrals(mol)
    rng = np.random.default_rng(315)
    directions = np.concatenate((rng.normal(size=(2, 2, 3)), np.ones((1, 2, 3))))
    columns = response._cd_columns(directions, workers=1)
    weights = np.zeros((3, *columns.shape[1:]))
    weights[0, 0, ::7] = rng.normal(size=weights[0, 0, ::7].shape)
    weights[1] = -weights[0]
    expected = np.einsum('mPp,oPp->mo', columns, weights)
    for workers in (1, 2):
        actual = response._cd_columns(directions, workers=workers, sensitivities=weights)
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-10)
        np.testing.assert_array_equal(actual[:, 2], 0.)
        np.testing.assert_array_equal(actual[2], 0.)
    zero = response._cd_columns(directions, sensitivities=np.zeros_like(weights))
    np.testing.assert_array_equal(zero, np.zeros((3, 3)))
