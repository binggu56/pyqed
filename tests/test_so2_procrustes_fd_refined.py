import numpy as np

from pyqed.units import au2fs

from examples.ldr.so2_procrustes_fd_refined import (
    cell_weights,
    hermitian_krylov_step,
    propagate_sparse,
    refined_grid,
    sparse_ldr,
)


def test_refined_grid_nests_original_points_and_weights_cover_domain():
    coarse = np.asarray([-0.5, 0.0, 0.8])
    fine = refined_grid(coarse, 2)
    np.testing.assert_allclose(fine, [-0.5, -0.25, 0.0, 0.4, 0.8])
    weights = cell_weights(fine, -0.7, 1.0)
    np.testing.assert_allclose(weights.sum(), 1.7)


def test_sparse_ldr_applies_link_blocks_only_on_kinetic_edges():
    kinetic = np.asarray([[1.0, -0.2], [-0.2, 1.5]])
    link = np.asarray([[0.8, 0.1], [-0.2, 0.7]], dtype=complex)
    local = np.zeros((2, 2, 2))
    matrix = sparse_ldr(
        kinetic,
        (2,),
        2,
        {(0, (0,)): link},
        local,
        average_paths=False,
    ).toarray()
    expected = np.block(
        [
            [np.eye(2), -0.2 * link],
            [-0.2 * link.conj().T, 1.5 * np.eye(2)],
        ]
    )
    np.testing.assert_allclose(matrix, expected)


def test_hermitian_krylov_step_matches_dense_exponential():
    from scipy.linalg import expm

    matrix = np.asarray([[0.3, 0.2j], [-0.2j, -0.1]])
    state = np.asarray([0.8, 0.6j])
    expected = expm(-0.4j * matrix) @ state
    result = hermitian_krylov_step(matrix, state, 0.4, 2)
    np.testing.assert_allclose(result, expected, atol=1.0e-13)


def test_sparse_propagation_matches_repeated_dense_steps():
    from scipy.linalg import expm

    matrix = np.asarray([[0.2, -0.1], [-0.1, 0.5]])
    state = np.asarray([1.0, 0.0])
    times_fs = np.asarray([0.0, 0.2, 0.4])
    result = propagate_sparse(matrix, state, times_fs, 2)
    step = expm(-1j * (0.2 / au2fs) * matrix)
    expected = np.asarray([state, step @ state, step @ step @ state])
    np.testing.assert_allclose(result, expected, atol=1.0e-13)
