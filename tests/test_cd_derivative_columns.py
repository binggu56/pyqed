import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.eri_response import CoulombIntegrals
from benchmarks.cd_derivative_reference import columns, factor_derivative


@pytest.mark.parametrize('layout', ['C', 'F'])
def test_cholesky_factor_response_against_displaced_factorization(layout):
    from scipy.linalg import cholesky, solve_triangular
    from pyqed.qchem.eri_response import _cholesky_factor_response
    rng = np.random.default_rng(319)
    root = np.array(np.tril(rng.normal(size=(5,5)))+5*np.eye(5), order=layout)
    factors = np.array(rng.normal(size=(5,11)), order=layout)
    dv = np.array(rng.normal(size=factors.shape), order=layout)
    dm = rng.normal(size=root.shape)
    dm += dm.T
    inputs = [a.copy() for a in (root, factors, dv, dm)]
    phi, db = _cholesky_factor_response(root, factors, dv, dm)
    metric, source = root@root.T, root@factors
    step = 1e-5
    displaced = [solve_triangular(cholesky(metric+s*step*dm, lower=True),
                                 source+s*step*dv, lower=True) for s in (1,-1)]
    np.testing.assert_allclose(db, (displaced[0]-displaced[1])/(2*step), atol=1e-10, rtol=1e-8)
    dl = root@phi
    np.testing.assert_allclose(dl@root.T+root@dl.T,dm,atol=2e-14)
    for actual, original in zip((root,factors,dv,dm),inputs):
        np.testing.assert_array_equal(actual,original)


@pytest.mark.parametrize('cart', [False, True])
@pytest.mark.parametrize('builder', ['cpp', 'python'])
def test_columns_match_scalar_reference_and_factor_recurrence(cart, builder):
    mol = Molecule(atom='O 0 0 0; H .2 1.4 1.1; H -.1 -1.4 1.2',
                   unit='bohr', basis='6-31g*')
    mol.build(eri='cd', options={'coord_type': 'cartesian' if cart else 'spherical',
                                'eri_backend': builder, 'low_rank_max_rank': 3})
    direction = np.zeros((3, 3))
    direction[1, 2] = 1.
    response = CoulombIntegrals(mol)
    expected = columns(mol, direction)
    for workers in (1, 2, 4):
        actual = response._cd_columns([direction, -direction, np.ones((3, 3))], workers=workers)
        assert response.derivative_info['private_output_bytes'] == 0
        np.testing.assert_allclose(actual[0], expected, atol=2e-10, rtol=2e-9)
        np.testing.assert_allclose(actual[1], -expected, atol=2e-10, rtol=2e-9)
        np.testing.assert_allclose(actual[2], 0., atol=1e-13)
        sensitivity = np.random.default_rng(5).normal(size=(2, *actual.shape[1:]))
        contracted = response._cd_columns([direction, -direction, np.ones((3, 3))],
                                          workers=workers, sensitivities=sensitivity)
        np.testing.assert_allclose(contracted, np.einsum('mPp,oPp->mo', actual, sensitivity),
                                   atol=2e-10, rtol=2e-9)
    from pyqed.qchem.basis import _pair_factors_to_full, mo_pair_factors
    old = _pair_factors_to_full(factor_derivative(mol, expected), mol._cd_factor_data['nao'])
    if mol._cd_factor_data['transform'] is None and mol._ao_cart2sph is not None:
        old = mo_pair_factors(old, mol._ao_cart2sph, mol._ao_cart2sph)
    np.testing.assert_allclose(response._cd_derivative(direction)[1], old, atol=3e-10, rtol=2e-9)


def test_selected_columns_need_neither_dense_derivatives_nor_scalar_calls(monkeypatch):
    mol = Molecule(atom='H 0 0 0; H .1 .2 1.5', unit='bohr', basis='sto-3g')
    mol.build(eri='cd')

    def forbidden(*args, **kwargs):
        raise AssertionError('CD assembly must use selected shell-batched columns')

    monkeypatch.setattr('pyqed.qchem.eri_response.directional_eri_derivatives', forbidden)
    monkeypatch.setattr('pyqed.qchem.basis_derivatives._contracted_eri_from_signatures', forbidden)
    response = CoulombIntegrals(mol)
    derivative = response.derivative(np.array([[0., 0., 0.], [0., 0., 1.]]))
    assert np.all(np.isfinite(derivative.j(np.eye(mol.nao))))


def test_empty_rank_and_column_validation():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.5', unit='bohr', basis='sto-3g')
    mol.build(eri='cd', options={'low_rank_max_rank': 0})
    response = CoulombIntegrals(mol)
    direction = np.zeros((2, 3))
    assert response._cd_columns(direction[None]).shape == (1, 0, 3)
    np.testing.assert_array_equal(response._cd_columns(direction[None],
                                  sensitivities=np.zeros((2, 0, 3))), np.zeros((1, 2)))
    with pytest.raises(ValueError, match='sensitivities'):
        response._cd_columns(direction[None], sensitivities=np.zeros((2, 1, 3)))
    np.testing.assert_array_equal(response.derivative(direction).j(np.eye(2)), np.zeros((2, 2)))
    from pyqed.qchem import basis
    shells, origins, exps, weights, nprim, owners, transform, _ = response._cd_arrays
    with pytest.raises(ValueError, match='Pivot'):
        basis._integrals_cpp.compute_eri_derivative_columns(
            shells, origins, exps, weights, nprim, owners, direction[None],
            transform, np.array([3], dtype=np.int64), 1)
    with pytest.raises(ValueError, match='dimensions'):
        basis._integrals_cpp.compute_eri_derivative_columns(
            shells, origins, exps, weights, nprim, owners, direction[None],
            transform[:1], np.array([0], dtype=np.int64), 1)


def test_concurrent_column_calls_keep_geometry_caches_independent():
    from concurrent.futures import ThreadPoolExecutor
    responses = []
    direction = np.array([[0., 0., 0.], [0., 0., 1.]])
    for z in (1.4, 2.0):
        mol = Molecule(atom=f'Li 0 0 0; H .1 .2 {z}', unit='bohr', basis='sto-3g')
        mol.build(eri='cd')
        responses.append(CoulombIntegrals(mol))
    expected = [obj._cd_columns(direction[None], workers=1) for obj in responses]
    for obj in responses:
        obj._cd_arrays = None
    def evaluate(i):
        return responses[i]._cd_columns(direction[None], workers=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, actual in zip([0, 1]*4, pool.map(evaluate, [0, 1]*4)):
            np.testing.assert_allclose(actual, expected[i], atol=2e-12)


@pytest.mark.parametrize('rank', [3, 12])
@pytest.mark.parametrize('basis_name', ['def2-svp', 'cc-pvtz'])
def test_selected_hrr_recipes_match_uncontracted_columns(rank, basis_name):
    rng = np.random.default_rng(704)
    for atom in ('O 0 0 0; H .2 1.4 1.1; H -.1 -1.4 1.2',
                 'O 0 0 0; H 0 1.4 1.1; H 0 -1.4 1.1'):
        mol = Molecule(atom=atom, unit='bohr', basis=basis_name).build(
            eri='cd', options={'low_rank_max_rank': rank})
        response = CoulombIntegrals(mol)
        directions = rng.normal(size=(2, 3, 3))
        columns = response._cd_columns(directions, workers=1)
        assert response.derivative_info['plan_misses'] > 0
        parallel = response._cd_columns(directions, workers=4)
        assert response.derivative_info['private_output_bytes'] == 0
        np.testing.assert_allclose(parallel, columns, atol=3e-10, rtol=2e-9)
        weights = rng.normal(size=(2, *columns.shape[1:]))
        expected = np.einsum('mPp,oPp->mo', columns, weights)
        for workers in (1, 2):
            actual = response._cd_columns(directions, workers=workers, sensitivities=weights)
            np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=2e-9)
