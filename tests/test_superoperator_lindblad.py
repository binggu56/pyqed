import numpy as np
import scipy.sparse as sp

from pyqed import dag
from pyqed.superoperator import (
    Qobj,
    dm2vec,
    lindblad_dissipator,
    liouvillian,
)


def _random_hermitian(rng, dim):
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return 0.5 * (matrix + matrix.conj().T)


def test_lindblad_superoperator_matches_matrix_rhs():
    rng = np.random.default_rng(4)
    dim = 3
    hamiltonian = _random_hermitian(rng, dim)
    collapse = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = _random_hermitian(rng, dim)

    super_rhs = liouvillian(hamiltonian, [collapse]) @ dm2vec(rho)
    matrix_rhs = (
        -1j * (hamiltonian @ rho - rho @ hamiltonian)
        + collapse @ rho @ dag(collapse)
        - 0.5 * ((dag(collapse) @ collapse) @ rho + rho @ (dag(collapse) @ collapse))
    )

    np.testing.assert_allclose(super_rhs, matrix_rhs.reshape(-1), atol=1.0e-12)


def test_lindblad_superoperator_is_trace_preserving():
    rng = np.random.default_rng(8)
    dim = 4
    hamiltonian = _random_hermitian(rng, dim)
    collapse = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

    generator = liouvillian(hamiltonian, [collapse]).toarray()
    trace_vector = np.eye(dim).reshape(-1)

    np.testing.assert_allclose(trace_vector.conj() @ generator, 0.0, atol=1.0e-12)


def test_qobj_lindblad_helper_matches_free_function():
    rng = np.random.default_rng(12)
    collapse = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    gamma = 0.37

    helper = Qobj(collapse).to_lindblad(gamma=gamma).toarray()
    expected = (gamma * lindblad_dissipator(collapse)).toarray()

    np.testing.assert_allclose(helper, expected, atol=1.0e-12)


def test_matrix_free_liouvillian_matches_explicit_superoperator():
    rng = np.random.default_rng(16)
    dim = 3
    hamiltonian = _random_hermitian(rng, dim)
    collapse_ops = [
        rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        for _ in range(2)
    ]
    rho = _random_hermitian(rng, dim)
    vec = dm2vec(rho)

    explicit = liouvillian(hamiltonian, collapse_ops)
    matrix_free = liouvillian(hamiltonian, collapse_ops, matrix_free=True)

    assert matrix_free.shape == explicit.shape
    np.testing.assert_allclose(matrix_free @ vec, explicit @ vec, atol=1.0e-12)


def test_matrix_free_liouvillian_accepts_sparse_operators():
    rng = np.random.default_rng(20)
    dim = 4
    hamiltonian = sp.csr_matrix(_random_hermitian(rng, dim))
    collapse_ops = [
        sp.csr_matrix(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    ]
    rho = _random_hermitian(rng, dim)
    vec = dm2vec(rho)

    explicit = liouvillian(hamiltonian, collapse_ops)
    matrix_free = liouvillian(hamiltonian, collapse_ops, matrix_free=True)

    np.testing.assert_allclose(matrix_free @ vec, explicit @ vec, atol=1.0e-12)
