import numpy as np
import pytest

from pyqed.letta import (
    LocalHamiltonian,
    LocalTerm,
    frontier_tensors_from_mps,
    frontier_tied_letta_from_mps,
)


def _mps_vector(tensors):
    dtype = np.result_type(*tensors)
    environment = np.ones((1, 1), dtype=dtype)
    for tensor in tensors:
        environment = np.einsum(
            "ca,apb->cpb",
            environment,
            tensor,
            optimize=True,
        ).reshape(-1, tensor.shape[2])
    return environment[:, 0]


def _tied_vector(tensors, dims, parent_sets):
    configs = np.asarray(list(np.ndindex(*dims)), dtype=np.intp)
    environment = np.ones((len(configs), 1), dtype=np.result_type(*tensors))
    for site, (tensor, parents) in enumerate(zip(tensors, parent_sets)):
        physical_sites = (site,) + tuple(parents)
        columns = np.ravel_multi_index(
            tuple(configs[:, physical_site] for physical_site in physical_sites),
            tuple(dims[physical_site] for physical_site in physical_sites),
        )
        left, right = tensor.shape[:2]
        transfer = tensor.reshape(left, right, -1)[:, :, columns].transpose(2, 0, 1)
        environment = np.einsum(
            "ca,cab->cb",
            environment,
            transfer,
            optimize=True,
        )
    return environment[:, 0]


def _random_mps(dims, ranks, seed=3, *, complex_values=False):
    rng = np.random.default_rng(seed)
    tensors = []
    for site, dim in enumerate(dims):
        tensor = rng.normal(size=(ranks[site], dim, ranks[site + 1]))
        if complex_values:
            tensor = tensor + 1.0j * rng.normal(size=tensor.shape)
        tensors.append(tensor)
    return tuple(tensors)


def test_zero_noise_lift_preserves_mps_with_padded_bonds():
    dims = (2, 3, 2, 2)
    ranks = (1, 2, 3, 2, 1)
    parents = ((2, 3), (3,), (3,), ())
    mps = _random_mps(dims, ranks, complex_values=True)

    tied = frontier_tensors_from_mps(mps, parents, bond_dim=4)

    assert tuple(tensor.shape for tensor in tied) == (
        (1, 4, 2, 2, 2),
        (4, 4, 3, 2),
        (4, 4, 2, 2),
        (4, 1, 2),
    )
    np.testing.assert_allclose(
        _tied_vector(tied, dims, parents),
        _mps_vector(mps),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert np.count_nonzero(tied[1][ranks[1] :, ...]) == 0
    assert np.count_nonzero(tied[1][:, ranks[2] :, ...]) == 0


def test_factory_preserves_normalized_mps_state():
    dims = (2, 2, 2, 2)
    ranks = (1, 2, 3, 2, 1)
    parents = ((1, 3), (2,), (3,), ())
    mps = _random_mps(dims, ranks, seed=8, complex_values=True)
    local_z = np.diag([0.4, -0.4])
    hamiltonian = LocalHamiltonian(dims, (LocalTerm((1,), local_z),))

    state = frontier_tied_letta_from_mps(
        hamiltonian,
        parents,
        mps,
        bond_dim=5,
        tie_noise=0.0,
        seed=12,
    )

    expected = _mps_vector(mps)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(
        state.state_vector(normalize=True),
        expected,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert state.bond_dim == 5


def test_tie_noise_is_reproducible_centered_and_tie_only():
    dims = (2, 2, 2)
    ranks = (1, 2, 2, 1)
    parents = ((1, 2), (2,), ())
    mps = _random_mps(dims, ranks, seed=5)
    exact = frontier_tensors_from_mps(mps, parents, bond_dim=3)
    noisy = frontier_tensors_from_mps(
        mps,
        parents,
        bond_dim=3,
        tie_noise=1.0e-3,
        seed=17,
    )
    repeated = frontier_tensors_from_mps(
        mps,
        parents,
        bond_dim=3,
        tie_noise=1.0e-3,
        seed=17,
    )

    for left, right in zip(noisy, repeated):
        np.testing.assert_array_equal(left, right)
    for site in (0, 1):
        parent_axes = tuple(range(3, noisy[site].ndim))
        np.testing.assert_allclose(
            np.mean(noisy[site] - exact[site], axis=parent_axes),
            0.0,
            atol=3.0e-17,
        )
    np.testing.assert_array_equal(noisy[-1], exact[-1])
    assert not np.allclose(noisy[0], exact[0])
    assert not np.allclose(
        _tied_vector(noisy, dims, parents),
        _mps_vector(mps),
    )


def test_mps_lift_rejects_incompatible_bonds_and_target_dimension():
    dims = (2, 2, 2)
    mps = _random_mps(dims, (1, 2, 3, 1))
    parents = ((1,), (2,), ())
    with pytest.raises(ValueError, match="smaller than the largest MPS bond"):
        frontier_tensors_from_mps(mps, parents, bond_dim=2)

    broken = list(mps)
    broken[1] = broken[1][:1]
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        frontier_tensors_from_mps(broken, parents)

    periodic = list(mps)
    periodic[0] = np.repeat(periodic[0], 2, axis=0)
    with pytest.raises(ValueError, match="open boundary"):
        frontier_tensors_from_mps(periodic, parents)

    with pytest.raises(ValueError, match="tie_noise"):
        frontier_tensors_from_mps(mps, parents, tie_noise=-1.0)
