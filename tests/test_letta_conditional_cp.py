import numpy as np

from pyqed.letta import conditional_cp_decompose


def test_conditional_cp_keeps_owned_block_dense():
    rng = np.random.default_rng(4)
    tensor = rng.normal(size=(2, 3, 2, 2))

    decomposition = conditional_cp_decompose(tensor, 1, 2)

    assert decomposition.core.shape == (2, 3, 2, 2)
    assert decomposition.parent_factors[0].shape == (2, 2)
    np.testing.assert_allclose(decomposition.reconstruct(), tensor, atol=2.0e-14)


def test_conditional_cp_exact_parent_configuration_expansion():
    rng = np.random.default_rng(7)
    tensor = rng.normal(size=(2, 2, 3, 2, 3))

    decomposition = conditional_cp_decompose(tensor, 2, 6)

    assert decomposition.rank == 6
    np.testing.assert_allclose(decomposition.reconstruct(), tensor, atol=0.0)


def test_conditional_cp_rank_one_matches_separable_label_dependence():
    rng = np.random.default_rng(9)
    core = rng.normal(size=(2, 2, 2))
    first = rng.normal(size=2)
    second = rng.normal(size=3)
    tensor = np.einsum("abc,d,e->abcde", core, first, second)

    decomposition = conditional_cp_decompose(
        tensor,
        2,
        1,
        max_iter=1000,
        tol=1.0e-13,
        seeds=range(3),
    )

    np.testing.assert_allclose(
        decomposition.reconstruct(),
        tensor,
        atol=2.0e-11,
    )
    assert decomposition.relative_error < 1.0e-11
