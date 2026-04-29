import numpy as np
from scipy import linalg

from pyqed.narg import LETTA


def _random_hermitian(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return 0.5 * (a + a.T)


def test_letta_dense_sweep_matches_exact_with_full_bond_dimension():
    dims = (2, 2, 2)
    h = _random_hermitian(np.prod(dims), seed=1)
    exact = np.linalg.eigvalsh(h)[0]

    letta = LETTA(h, dims, bond_dim=4, seed=2)
    result = letta.run(nsweeps=3, tol=1e-12)

    np.testing.assert_allclose(result.energy, exact, atol=1e-10)
    assert result.ncompleted >= 1


def test_letta_supports_generalized_overlap_metric():
    dims = (2, 2, 2)
    n = int(np.prod(dims))
    h = _random_hermitian(n, seed=3)
    rng = np.random.default_rng(4)
    a = rng.normal(size=(n, n))
    s = np.eye(n) + 0.05 * (a.T @ a)
    exact = linalg.eigh(h, s, eigvals_only=True)[0]

    letta = LETTA(h, dims, bond_dim=4, overlap=s, seed=5)
    result = letta.run(nsweeps=3, tol=1e-12)

    np.testing.assert_allclose(result.energy, exact, atol=1e-10)


def test_letta_respects_requested_bond_dimension():
    dims = (2, 3, 2)
    h = _random_hermitian(np.prod(dims), seed=6)

    letta = LETTA(h, dims, bond_dim=2, seed=7)
    result = letta.run(nsweeps=2)

    assert result.history
    assert max(core.shape[0] for core in result.cores) <= 2
    assert max(core.shape[2] for core in result.cores) <= 2
