import numpy as np
import pytest

from pyqed.tn import LocalHamiltonian, LocalTerm
from pyqed.tn import MPO


def _random_hermitian(rng, dimension):
    matrix = rng.normal(size=(dimension, dimension))
    matrix = matrix + 1.0j * rng.normal(size=(dimension, dimension))
    return 0.5 * (matrix + matrix.T.conj())


@pytest.mark.parametrize("seed", range(4))
def test_compress_preserves_random_complex_heterogeneous_hamiltonian(seed):
    rng = np.random.default_rng(seed)
    dims = (2, 3, 2, 2)
    supports = ((0,), (0, 2), (1, 3), (0, 2, 3), (1, 2))
    terms = []
    for sites in supports:
        dimension = int(np.prod([dims[site] for site in sites]))
        terms.append(LocalTerm(sites, _random_hermitian(rng, dimension)))
    hamiltonian = LocalHamiltonian(dims, terms, constant=-0.37)

    mpo = hamiltonian.to_mpo()
    compressed = mpo.compress()
    minimized = hamiltonian.to_mpo(minimize=True)
    vector = rng.normal(size=hamiltonian.shape[0])
    vector = vector + 1.0j * rng.normal(size=hamiltonian.shape[0])

    assert mpo.factors is mpo.tensors
    np.testing.assert_allclose(
        compressed.to_dense(),
        mpo.to_dense(),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        compressed.to_dense() @ vector,
        hamiltonian.matvec(vector),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert max(compressed.bond_dims) < max(mpo.bond_dims)
    assert minimized.bond_dims == compressed.bond_dims
    np.testing.assert_allclose(minimized.to_dense(), compressed.to_dense(), atol=2.0e-13)
    assert compressed.dims == dims
    assert all(not tensor.flags.writeable for tensor in compressed.tensors)


def test_compress_removes_explicit_zero_channels_for_heterogeneous_dims():
    dims = (2, 3, 2)
    first = np.zeros((1, 3, 2, 2), dtype=complex)
    middle = np.zeros((3, 4, 3, 3), dtype=complex)
    last = np.zeros((4, 1, 2, 2), dtype=complex)
    first[0, 0] = (0.4 + 0.3j) * np.eye(2)
    middle[0, 0] = np.eye(3)
    last[0, 0] = np.eye(2)
    mpo = MPO((first, middle, last))

    compressed = mpo.compress(rtol=0.0)

    assert compressed.bond_dims == (1, 1, 1, 1)
    np.testing.assert_allclose(compressed.to_dense(), mpo.to_dense(), atol=2.0e-15)


def test_compress_handles_zero_hamiltonian_and_validates_rtol():
    zero = LocalHamiltonian((2, 3, 2)).to_mpo().compress()

    assert zero.bond_dims == (1, 1, 1, 1)
    np.testing.assert_array_equal(zero.to_dense(), np.zeros((12, 12)))
    with pytest.raises(ValueError, match="rtol"):
        zero.compress(rtol=-1.0)
    with pytest.raises(ValueError, match="rtol"):
        zero.compress(rtol=np.inf)
