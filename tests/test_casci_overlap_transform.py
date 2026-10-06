from itertools import combinations
from types import SimpleNamespace
import importlib

import numpy as np
import pytest

c = importlib.import_module('pyqed.qchem.mcscf.casci')


def strings(n, k):
    rows = np.zeros((len(list(combinations(range(n), k))), n), dtype=np.int8)
    for row, occ in zip(rows, combinations(range(n), k)):
        row[list(occ)] = 1
    return rows


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('electrons', [0, 2, 4])
def test_batched_minors_match_scalar(complex_values, electrons):
    rng = np.random.default_rng(51)
    matrix = rng.normal(size=(4, 4))
    if complex_values:
        matrix = matrix+1j*rng.normal(size=(4, 4))
    occupations = c._occupation_lists(strings(4, electrons))
    expected = np.array([[np.linalg.det(matrix[np.ix_(a, b)]) for b in occupations] for a in occupations])
    actual = c._string_overlap_matrix(matrix, occupations, occupations, matrix.dtype)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('same_spin', [False, True])
def test_batched_root_solves_and_factorization_reuse(monkeypatch, same_spin):
    rng = np.random.default_rng(13)
    a = np.eye(4)+.1*(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
    b = a if same_spin else np.eye(3)+.1j*rng.normal(size=(3, 3))
    ci = rng.normal(size=(3, len(a), len(b)))+1j*rng.normal(size=(3, len(a), len(b)))
    expected = np.array([np.linalg.solve(b, np.linalg.solve(a, x).T).T for x in ci])
    calls = []
    factor = c.lu_factor
    def counted(x):
        calls.append(x)
        return factor(x)
    monkeypatch.setattr(c, 'lu_factor', counted)
    actual = c._transform_ci_tensors_to_biorthogonal_basis(ci, a, b)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert len(calls) == (1 if same_spin else 2)


@pytest.mark.parametrize('beta_electrons', [1, 2])
def test_full_complex_overlap_matches_independent_determinant_reference(monkeypatch, beta_electrons):
    rng = np.random.default_rng(27)
    a, b = strings(3, 1), strings(3, beta_electrons)
    binary = np.array([[x, y] for x in a for y in b])
    def frame():
        return SimpleNamespace(binary=binary, ncore=1, ncas=3,
                               ci=rng.normal(size=(2, len(binary)))+1j*rng.normal(size=(2, len(binary))))
    left, right = frame(), frame()
    s = np.eye(4)+.08*(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
    expected = c._overlap_slow_from_mo_overlap(left, right, s=s)
    calls = []
    transform = c._string_transform_matrix
    def counted(*args):
        calls.append(1)
        return transform(*args)
    monkeypatch.setattr(c, '_string_transform_matrix', counted)
    actual = c.overlap(left, right, s=s)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert len(calls) == (2 if beta_electrons == 1 else 4)
