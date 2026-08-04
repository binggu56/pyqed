import numpy as np
import pytest

from pyqed.letta import LocalHamiltonian, LocalTerm
from pyqed.tn import TTN, balanced_ttn


def _path_mps_amplitude(tensors, configuration):
    value = tensors[0][0, configuration[0], :]
    for site in range(1, len(tensors) - 1):
        value = value @ tensors[site][:, configuration[site], :]
    return (value @ tensors[-1][:, configuration[-1], 0]).item()


def _path_ttn_tensors(mps):
    tensors = [mps[0][0].copy()]
    tensors.extend(tensor.transpose(1, 0, 2).copy() for tensor in mps[1:-1])
    tensors.append(mps[-1][:, :, 0].T.copy())
    return tensors


def test_ttn_rejects_invalid_parent_topologies():
    with pytest.raises(ValueError, match="exactly one root"):
        TTN((2, 2), (None, None))
    with pytest.raises(ValueError, match="acyclic"):
        TTN((2, 2, 2), (None, 2, 1))


@pytest.mark.parametrize("dims", [(2.5, 2), (True, 2)])
def test_ttn_rejects_noninteger_dimensions(dims):
    with pytest.raises(ValueError, match="dimensions must be an integer"):
        TTN(dims, (None, 0))


@pytest.mark.parametrize("parents", [(None, 0.9), (None, False)])
def test_ttn_rejects_noninteger_parents(parents):
    with pytest.raises(ValueError, match="parent sites must be an integer"):
        TTN((2, 2), parents)


def test_ttn_accepts_numpy_bond_dimension_sequence():
    state = TTN((2, 2), (None, 0), bond_dim=np.array([1, 3]))
    assert state.edge_dims == {(0, 1): 3}

    with pytest.raises(ValueError, match="no entry for child site 1"):
        TTN((2, 2), (None, 0), bond_dim=[2])


def test_path_ttn_matches_mps_amplitudes_and_state_vector():
    rng = np.random.default_rng(4)
    mps = [
        rng.normal(size=(1, 2, 3)) + 1j * rng.normal(size=(1, 2, 3)),
        rng.normal(size=(3, 3, 2)) + 1j * rng.normal(size=(3, 3, 2)),
        rng.normal(size=(2, 2, 1)) + 1j * rng.normal(size=(2, 2, 1)),
    ]
    state = TTN((2, 3, 2), (None, 0, 1), tensors=_path_ttn_tensors(mps))

    expected = np.array(
        [_path_mps_amplitude(mps, config) for config in np.ndindex(2, 3, 2)]
    )
    np.testing.assert_allclose(state.state_vector(), expected, atol=1.0e-12)
    np.testing.assert_allclose(state.norm(), np.linalg.norm(expected), atol=1.0e-12)


def test_branched_ttn_messages_match_dense_norm():
    state = TTN(
        (2, 3, 2, 2, 3),
        (None, 0, 0, 1, 1),
        bond_dim={(0, 1): 3, (0, 2): 2, (1, 3): 2, (1, 4): 3},
        seed=8,
    )

    assert state.tensors[0].shape == (2, 3, 2)
    assert state.tensors[1].shape == (3, 3, 2, 3)
    np.testing.assert_allclose(
        state.norm_squared(),
        np.vdot(state.state_vector(), state.state_vector()).real,
        atol=1.0e-12,
    )
    assert state.edge_message(1, 0).shape == (3, 3)
    assert state.edge_message(0, 2).shape == (2, 2)


def test_complex64_norm_and_normalization():
    state = TTN(
        (2,) * 9,
        (None, 0, 0, 1, 1, 2, 2, 3, 3),
        bond_dim=5,
        seed=0,
    )
    state.tensors = [
        (tensor + 1j * np.roll(tensor, 1)).astype(np.complex64)
        for tensor in state.tensors
    ]

    dense_norm_squared = np.vdot(state.state_vector(), state.state_vector()).real
    np.testing.assert_allclose(
        state.norm_squared(), dense_norm_squared, rtol=2.0e-6, atol=0.0
    )
    state.normalize()
    np.testing.assert_allclose(state.norm(), 1.0, rtol=2.0e-6, atol=0.0)


def test_product_operator_expectation_matches_dense_state():
    state = TTN((2, 2, 2, 2), (1, None, 1, 2), bond_dim=3, seed=14)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    operators = {0: x, 3: z}

    dense_operator = np.array([1.0])
    for site in range(state.nsites):
        dense_operator = np.kron(
            dense_operator,
            operators.get(site, np.eye(state.dims[site])),
        )
    vector = state.state_vector()
    expected = np.vdot(vector, dense_operator @ vector) / np.vdot(vector, vector)

    np.testing.assert_allclose(
        state.expectation_value(operators),
        expected,
        atol=1.0e-12,
    )


def test_center_effective_product_operator_matches_dense_contraction():
    state = TTN((2, 2, 2, 2, 2), (None, 0, 0, 1, 1), bond_dim=3, seed=5)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    operators = {1: z, 4: x}
    center = 2
    state.canonicalize(center)

    effective = state.effective_product_operator(operators, center=center)
    tensor = state.tensors[center].reshape(-1)
    dense_operator = np.array([1.0])
    for site in range(state.nsites):
        dense_operator = np.kron(
            dense_operator,
            operators.get(site, np.eye(state.dims[site])),
        )
    vector = state.state_vector()

    np.testing.assert_allclose(
        np.vdot(tensor, effective @ tensor),
        np.vdot(vector, dense_operator @ vector),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.effective_product_operator(center=center),
        np.eye(tensor.size),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.effective_operator_sum(
            [(0.7, operators), (-0.2, {0: x})],
            center=center,
        ),
        0.7 * effective
        - 0.2 * state.effective_product_operator({0: x}, center=center),
        atol=1.0e-12,
    )


def test_ttn_normalize_copy_and_configuration_validation():
    state = TTN((2, 2, 2), (1, None, 1), bond_dim=2, seed=3)
    copied = state.copy()
    copied.tensors[0][...] = 0.0
    assert not np.allclose(state.tensors[0], copied.tensors[0])

    state.normalize()
    np.testing.assert_allclose(state.norm(), 1.0, atol=1.0e-12)
    with pytest.raises(ValueError, match="one value per site"):
        state.amplitude((0, 1))
    with pytest.raises(ValueError, match="out-of-range"):
        state.amplitude((0, 2, 0))
    with pytest.raises(ValueError, match="configuration states must be an integer"):
        state.amplitude((0, 0.5, 0))
    with pytest.raises(ValueError, match="center site must be an integer"):
        state.canonicalize(1.5)


def test_ttn_canonicalize_preserves_state_and_makes_branches_isometric():
    state = TTN((2, 2, 3, 2, 2), (None, 0, 0, 1, 1), bond_dim=3, seed=12)
    before = state.state_vector()

    state.canonicalize(4)
    state.normalize()

    np.testing.assert_allclose(
        state.state_vector(), before / np.linalg.norm(before), atol=1.0e-12
    )
    assert state.center == 4
    toward_center, preorder = state._traversal_from(4)
    for site in preorder[1:]:
        axis = state._bond_axis(site, toward_center[site])
        tensor = np.moveaxis(state.tensors[site], axis, -1)
        matrix = tensor.reshape(-1, tensor.shape[-1])
        np.testing.assert_allclose(
            matrix.conj().T @ matrix,
            np.eye(matrix.shape[1]),
            atol=1.0e-12,
        )


def _critical_ising_hamiltonian(state, nspins):
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    terms = [LocalTerm((site,), -z) for site in range(nspins)]
    terms.extend(
        LocalTerm((site, site + 1), -np.kron(x, x))
        for site in range(nspins - 1)
    )
    return LocalHamiltonian(state.dims, terms)


def test_local_hamiltonian_expectation_matches_dense_reference():
    state = balanced_ttn(4, physical_dim=2, bond_dim=3, seed=21)
    hamiltonian = _critical_ising_hamiltonian(state, 4)
    vector = state.state_vector()

    np.testing.assert_allclose(
        state.expectation(hamiltonian),
        hamiltonian.expectation(vector),
        atol=1.0e-12,
    )


def test_full_rank_balanced_ttn_recovers_four_site_critical_ising_ground_state():
    state = balanced_ttn(4, physical_dim=2, bond_dim=4, seed=9)
    hamiltonian = _critical_ising_hamiltonian(state, 4)
    exact = np.linalg.eigvalsh(hamiltonian.to_dense())[0]
    initial = state.expectation(hamiltonian)

    state.run(hamiltonian, nsweeps=4, tol=1.0e-11)

    assert state.energy <= initial + 1.0e-12
    np.testing.assert_allclose(state.energy, exact, atol=1.0e-10)
    assert all(update.accepted for update in state.site_updates)
    assert all(
        later["energy"] <= earlier["energy"] + 1.0e-12
        for earlier, later in zip(state.history, state.history[1:])
    )
