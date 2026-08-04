import numpy as np
import pytest

from pyqed.letta.matrix_free import lowest_recycled_block_davidson


def _hermitian(size, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(size, size)) + 1.0j * rng.normal(
        size=(size, size)
    )
    return 0.5 * (matrix + matrix.T.conj())


def test_recycled_block_davidson_batches_actions_and_matches_dense():
    size = 20
    hamiltonian = _hermitian(size, 7)
    initial = np.random.default_rng(8).normal(size=(size, 3))
    shapes = []

    def batch_action(vectors):
        shapes.append(vectors.shape)
        return hamiltonian @ vectors

    def forbidden_scalar_action(_vector):
        raise AssertionError("scalar action should not be used with a batch callback")

    energy, vector, recycle, diagnostics = lowest_recycled_block_davidson(
        forbidden_scalar_action,
        initial,
        hamiltonian_batch_action=batch_action,
        diagonal=np.diag(hamiltonian),
        block_size=3,
        recycle_dimension=4,
        max_subspace=14,
        maxiter=160,
        tol=1.0e-10,
    )

    assert diagnostics.converged
    assert energy == pytest.approx(np.linalg.eigvalsh(hamiltonian)[0], abs=1.0e-9)
    assert np.linalg.norm(hamiltonian @ vector - energy * vector) < 1.0e-8
    assert recycle.shape == (size, 4)
    assert np.allclose(recycle.T.conj() @ recycle, np.eye(4), atol=1.0e-12)
    assert diagnostics.batch_action_calls == len(shapes)
    assert diagnostics.scalar_action_calls == 0
    assert diagnostics.hamiltonian_action_calls == len(shapes)
    assert diagnostics.hamiltonian_matvecs == sum(shape[1] for shape in shapes)
    assert diagnostics.hamiltonian_action_calls < diagnostics.hamiltonian_matvecs
    assert diagnostics.recycle_dimension == recycle.shape[1]


def test_returned_ritz_space_can_be_recycled_into_a_changed_problem():
    size = 18
    hamiltonian = _hermitian(size, 11)
    rng = np.random.default_rng(12)
    _, _, recycle, first = lowest_recycled_block_davidson(
        lambda vector: hamiltonian @ vector,
        rng.normal(size=(size, 2)),
        diagonal=np.diag(hamiltonian),
        block_size=2,
        recycle_dimension=5,
        max_subspace=13,
        maxiter=160,
        tol=1.0e-10,
    )
    changed = hamiltonian + np.diag(np.linspace(-0.02, 0.03, size))
    energy, vector, updated_recycle, second = lowest_recycled_block_davidson(
        lambda trial: changed @ trial,
        recycle,
        diagonal=np.diag(changed),
        block_size=3,
        recycle_dimension=5,
        max_subspace=14,
        maxiter=160,
        tol=1.0e-10,
    )

    assert first.converged
    assert second.converged
    assert energy == pytest.approx(np.linalg.eigvalsh(changed)[0], abs=1.0e-9)
    assert np.linalg.norm(changed @ vector - energy * vector) < 1.0e-8
    assert updated_recycle.shape == recycle.shape


def test_exact_excited_invariant_start_is_deterministically_augmented():
    size = 20
    diagonal = np.arange(size, dtype=float)
    diagonal[17] = -3.0
    excited_a = np.eye(size)[:, -2]
    excited_b = np.eye(size)[:, -1]
    excited_subspace = np.column_stack(
        (excited_a + excited_b, excited_a - excited_b)
    )

    energy, vector, _, diagnostics = lowest_recycled_block_davidson(
        lambda trial: diagonal * trial,
        excited_subspace,
        diagonal=diagonal,
        block_size=2,
        recycle_dimension=3,
        max_subspace=8,
        maxiter=100,
        tol=1.0e-12,
    )

    assert diagnostics.converged
    assert diagnostics.deterministic_augmentations >= 1
    assert energy == pytest.approx(-3.0, abs=1.0e-11)
    assert abs(vector[17]) == pytest.approx(1.0, abs=1.0e-10)


def test_disjoint_block_jacobi_data_support_complex_blocks():
    rng = np.random.default_rng(31)
    blocks = []
    matrices = []
    for start, width in ((0, 3), (3, 4), (7, 3)):
        block = _hermitian(width, start + 40)
        blocks.append((slice(start, start + width), block))
        matrices.append(block)
    hamiltonian = np.zeros((10, 10), dtype=complex)
    hamiltonian[:3, :3] = matrices[0]
    hamiltonian[3:7, 3:7] = matrices[1]
    hamiltonian[7:, 7:] = matrices[2]

    energy, _, _, diagnostics = lowest_recycled_block_davidson(
        lambda vector: hamiltonian @ vector,
        rng.normal(size=(10, 2)) + 1.0j * rng.normal(size=(10, 2)),
        preconditioner_blocks=blocks,
        block_size=2,
        recycle_dimension=3,
        max_subspace=8,
        maxiter=100,
        tol=1.0e-11,
    )

    assert diagnostics.converged
    assert energy == pytest.approx(np.linalg.eigvalsh(hamiltonian)[0], abs=1.0e-10)


def test_overlapping_preconditioner_blocks_are_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        lowest_recycled_block_davidson(
            lambda vector: vector,
            np.ones(4),
            preconditioner_blocks=[
                ([0, 1], np.eye(2)),
                ([1, 2], np.eye(2)),
            ],
        )


def test_iteration_limit_returns_a_residual_verified_partial_root():
    diagonal = np.arange(8, dtype=float)
    energy, vector, _, diagnostics = lowest_recycled_block_davidson(
        lambda trial: diagonal * trial,
        np.ones(8),
        block_size=1,
        maxiter=1,
        tol=0.0,
        atol=0.0,
    )

    residual = diagonal * vector - energy * vector
    assert diagnostics.converged is False
    assert diagnostics.message == "maximum iterations reached"
    assert diagnostics.iterations == 1
    assert diagnostics.residual_norm == pytest.approx(np.linalg.norm(residual))
    assert diagnostics.hamiltonian_action_calls == 3
    assert diagnostics.hamiltonian_vector_products == 3
