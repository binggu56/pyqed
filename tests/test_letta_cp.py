import numpy as np

from pyqed.letta.cp import cp_als


def test_cp_als_recovers_real_rank_one_tensor():
    left = np.array([1.0, -2.0])
    middle = np.array([0.5, 1.5, -0.25])
    right = np.array([2.0, -1.0, 0.75, 3.0])
    tensor = np.einsum("i,j,k->ijk", left, middle, right)

    decomposition = cp_als(tensor, rank=1, seed=7)

    assert decomposition.weights.shape == (1,)
    assert [factor.shape for factor in decomposition.factors] == [(2, 1), (3, 1), (4, 1)]
    np.testing.assert_allclose(decomposition.reconstruct(), tensor, atol=1.0e-12)
    assert decomposition.relative_error < 1.0e-12


def test_cp_als_recovers_complex_ghz_tensor_at_rank_two():
    tensor = np.zeros((2, 2, 2, 2), dtype=complex)
    tensor[0, 0, 0, 0] = 1.0
    tensor[1, 1, 1, 1] = 0.5j

    decomposition = cp_als(tensor, rank=2, seed=11)

    assert all(factor.shape == (2, 2) for factor in decomposition.factors)
    assert all(np.iscomplexobj(factor) for factor in decomposition.factors)
    np.testing.assert_allclose(decomposition.reconstruct(), tensor, atol=1.0e-12)
    assert decomposition.relative_error < 1.0e-12


def test_cp_als_error_is_nonincreasing_when_rank_resolves_orthogonal_terms():
    tensor = np.zeros((3, 3, 3))
    tensor[0, 0, 0] = 3.0
    tensor[1, 1, 1] = 2.0
    tensor[2, 2, 2] = 1.0

    decompositions = [cp_als(tensor, rank=rank, seed=19) for rank in (1, 2, 3)]
    errors = np.asarray([decomposition.relative_error for decomposition in decompositions])

    assert np.all(np.diff(errors) <= 1.0e-12)
    np.testing.assert_allclose(decompositions[-1].reconstruct(), tensor, atol=1.0e-12)
    assert errors[-1] < 1.0e-12
