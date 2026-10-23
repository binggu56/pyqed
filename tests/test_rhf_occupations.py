"""Constrained integer filling against exhaustive subset references."""
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyqed.qchem.hf.occupations import rhf_occupations


def test_aufbau_unsorted():
    assert_allclose(rhf_occupations([2, -3, 0, -1], 4), [0, 2, 0, 2])


@pytest.mark.parametrize('seed', range(5))
def test_multiplet_optimum_against_exhaustive(seed):
    rng = np.random.default_rng(seed)
    sizes = np.array([1, 2, 3, 2, 1])
    ids = np.repeat(np.arange(len(sizes)), sizes)
    energies = np.repeat(rng.normal(size=len(sizes)), sizes)
    order = rng.permutation(len(ids))
    energies, ids = energies[order], ids[order]
    for pairs in range(len(ids) + 1):
        costs = [2 * sum(energies[np.isin(ids, np.flatnonzero(choice))])
                 for choice in product([0, 1], repeat=len(sizes))
                 if np.dot(sizes, choice) == pairs]
        occ = rhf_occupations(energies, 2 * pairs, ids)
        assert_allclose(occ.sum(), 2 * pairs)
        assert_allclose(energies @ occ, min(costs))
        for label in np.unique(ids):
            assert np.unique(occ[ids == label]).size == 1


def test_impossible_and_invalid_filling():
    with pytest.raises(ValueError, match='complete symmetry multiplets'):
        rhf_occupations([-1, -1, 2, 2], 2, [0, 0, 1, 1])
    for nelec in [-2, 1, 6, 2.5]:
        with pytest.raises(ValueError, match='even electron count'):
            rhf_occupations([0, 1], nelec)
    with pytest.raises(ValueError, match='finite vector'):
        rhf_occupations([np.nan], 0)
    with pytest.raises(ValueError, match='match orbital energies'):
        rhf_occupations([0, 1], 2, [0])


def test_degenerate_filling_is_deterministic():
    assert_allclose(rhf_occupations([0, 0, 0, 0], 4, [0, 0, 1, 1]), [2, 2, 0, 0])


def test_stability_uses_occupation_masks(monkeypatch):
    import importlib
    rhf = importlib.import_module('pyqed.qchem.hf.rhf')
    monkeypatch.setattr(rhf, '_rhf_determinant_energy', lambda *a: 0.)
    records, _ = rhf._frontier_stability_directions(
        None, np.array([-3., -2., -2., 3.]), np.eye(4), np.array([0., 2., 2., 0.]),
        np.zeros((4, 4)), 0., 0., None, .1, .001, 1e-5)
    assert {(item['occupied'], item['virtual']) for item in records} == {(1, 0), (2, 0)}
