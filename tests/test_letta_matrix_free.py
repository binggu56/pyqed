import numpy as np
from scipy import linalg

from pyqed.letta.matrix_free import lowest_generalized_davidson


def _random_hermitian(rng, size, *, complex_values):
    matrix = rng.normal(size=(size, size))
    if complex_values:
        matrix = matrix + 1.0j * rng.normal(size=(size, size))
    return 0.5 * (matrix + matrix.T.conj())


def test_generalized_davidson_matches_dense_positive_definite_problem():
    rng = np.random.default_rng(18)
    size = 18
    factor = rng.normal(size=(size, size)) + 1.0j * rng.normal(size=(size, size))
    metric = factor.T.conj() @ factor + 0.5 * np.eye(size)
    hamiltonian = _random_hermitian(rng, size, complex_values=True)
    reference = linalg.eigh(
        hamiltonian,
        metric,
        subset_by_index=[0, 0],
        check_finite=False,
    )[0][0]

    energy, vector, diagnostics = lowest_generalized_davidson(
        hamiltonian.__matmul__,
        metric.__matmul__,
        rng.normal(size=size) + 1.0j * rng.normal(size=size),
        max_subspace=size,
        tol=2.0e-11,
    )

    assert diagnostics.converged
    assert diagnostics.hamiltonian_matvecs <= diagnostics.iterations + 2
    assert diagnostics.metric_matvecs <= diagnostics.iterations + 2
    np.testing.assert_allclose(energy, reference, atol=2.0e-10)
    np.testing.assert_allclose(np.vdot(vector, metric @ vector), 1.0, atol=2.0e-12)
    np.testing.assert_allclose(
        hamiltonian @ vector,
        energy * (metric @ vector),
        atol=2.0e-9,
    )


def test_generalized_davidson_handles_rank_deficient_complex_metric():
    rng = np.random.default_rng(29)
    size = 21
    rank = 7
    unitary, _triangular = np.linalg.qr(
        rng.normal(size=(size, rank)) + 1.0j * rng.normal(size=(size, rank))
    )
    metric_values = np.geomspace(0.4, 3.0, rank)
    metric = (unitary * metric_values[None, :]) @ unitary.T.conj()
    range_hamiltonian = _random_hermitian(rng, rank, complex_values=True)
    hamiltonian = unitary @ range_hamiltonian @ unitary.T.conj()
    metric_basis = unitary / np.sqrt(metric_values)[None, :]
    reduced_hamiltonian = metric_basis.T.conj() @ hamiltonian @ metric_basis
    reference = np.linalg.eigvalsh(reduced_hamiltonian)[0]
    initial = rng.normal(size=size) + 1.0j * rng.normal(size=size)

    energy, vector, diagnostics = lowest_generalized_davidson(
        hamiltonian.__matmul__,
        metric.__matmul__,
        initial,
        max_subspace=size,
        metric_tol=1.0e-11,
        tol=2.0e-11,
    )

    assert diagnostics.converged
    assert diagnostics.projected_rank <= rank
    np.testing.assert_allclose(energy, reference, atol=3.0e-10)
    np.testing.assert_allclose(np.vdot(vector, metric @ vector), 1.0, atol=3.0e-11)
    np.testing.assert_allclose(
        hamiltonian @ vector,
        energy * (metric @ vector),
        atol=3.0e-9,
    )


def test_generalized_davidson_recovers_from_metric_null_initial_vector():
    diagonal_metric = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    diagonal_hamiltonian = np.array([0.0, 0.0, 0.7, -1.0, 4.0])
    metric = np.diag(diagonal_metric)
    hamiltonian = np.diag(diagonal_hamiltonian)
    initial = np.array([1.0, -2.0, 0.0, 0.0, 0.0])

    energy, vector, diagnostics = lowest_generalized_davidson(
        hamiltonian.__matmul__,
        metric.__matmul__,
        initial,
        max_subspace=5,
        tol=1.0e-12,
        random_seed=7,
    )

    assert diagnostics.converged
    np.testing.assert_allclose(energy, -0.5, atol=2.0e-12)
    np.testing.assert_allclose(np.vdot(vector, metric @ vector), 1.0, atol=2.0e-12)


def test_generalized_davidson_uses_vector_actions_and_has_no_dense_fallback():
    size = 12
    diagonal_metric = np.linspace(0.7, 1.8, size)
    diagonal_hamiltonian = np.linspace(-2.0, 3.0, size)
    seen_shapes = []

    def hamiltonian_action(vector):
        seen_shapes.append(vector.shape)
        assert vector.ndim == 1
        return diagonal_hamiltonian * vector

    def metric_action(vector):
        seen_shapes.append(vector.shape)
        assert vector.ndim == 1
        return diagonal_metric * vector

    energy, _vector, diagnostics = lowest_generalized_davidson(
        hamiltonian_action,
        metric_action,
        np.ones(size),
        maxiter=1,
        max_subspace=4,
        tol=0.0,
    )

    assert not diagnostics.converged
    assert diagnostics.iterations == 1
    assert diagnostics.hamiltonian_matvecs == 2
    assert diagnostics.metric_matvecs == 2
    assert seen_shapes and set(seen_shapes) == {(size,)}
    assert np.isfinite(energy)


def test_generalized_davidson_uses_jacobi_and_block_batched_actions():
    size = 18
    diagonal = np.linspace(-3.0, 4.0, size)
    batched = []
    preconditioned = []

    def actions(vectors):
        batched.append(vectors.shape[0])
        return vectors * diagonal[None, :]

    def preconditioner(residual, energy):
        preconditioned.append(float(energy))
        denominator = diagonal - energy
        denominator[np.abs(denominator) < 1.0e-8] = 1.0e-8
        return residual / denominator

    energy, _vector, diagnostics = lowest_generalized_davidson(
        lambda vector: diagonal * vector,
        lambda vector: vector,
        np.ones(size),
        hamiltonian_actions=actions,
        metric_actions=lambda vectors: vectors.copy(),
        preconditioner=preconditioner,
        block_size=3,
        max_subspace=size,
        tol=1.0e-11,
    )

    assert diagnostics.converged
    assert preconditioned
    assert any(batch > 1 for batch in batched)
    np.testing.assert_allclose(energy, diagonal[0], atol=1.0e-10)
