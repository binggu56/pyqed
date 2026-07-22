import numpy as np

from pyqed.ml import (
    ARNN,
    heisenberg_connections,
    transverse_field_ising_connections,
)


def _heisenberg_dense(n_visible):
    configurations = ARNN(
        n_visible, hidden_size=2, seed=0, init_scale=0.0
    ).all_configurations()
    hamiltonian = np.zeros((2**n_visible, 2**n_visible), dtype=complex)
    for row, spins in enumerate(configurations):
        for site in range(n_visible - 1):
            neighbor = site + 1
            hamiltonian[row, row] += 0.25 * spins[site] * spins[neighbor]
            if spins[site] != spins[neighbor]:
                flipped = spins.copy()
                flipped[site] *= -1
                flipped[neighbor] *= -1
                column = np.flatnonzero(np.all(configurations == flipped, axis=1))[0]
                hamiltonian[row, column] += 0.5
    return hamiltonian


def test_autoregressive_state_is_normalized_by_construction():
    state = ARNN(5, hidden_size=7, seed=3)
    psi = state.state_vector()

    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1.0e-12)
    configurations = state.all_configurations()
    np.testing.assert_allclose(
        state.amplitude(configurations),
        np.exp(state.log_amplitude(configurations)),
    )


def test_direct_samples_follow_exact_autoregressive_probabilities():
    state = ARNN(3, hidden_size=5, seed=4, init_scale=0.4)
    samples = state.sample(30_000, seed=8)
    labels = ((samples == -1) * np.array([4, 2, 1])).sum(axis=1)
    observed = np.bincount(labels, minlength=8) / samples.shape[0]
    expected = np.abs(state.state_vector()) ** 2

    np.testing.assert_allclose(observed, expected, atol=1.2e-2)


def test_connected_heisenberg_local_energy_matches_dense_reference():
    state = ARNN(4, hidden_size=6, seed=5, init_scale=0.2)
    configurations = state.all_configurations()
    connected, elements = heisenberg_connections(configurations)
    local = np.asarray(state.local_energies(configurations, connected, elements))

    psi = state.state_vector()
    dense = _heisenberg_dense(4)
    expected = (dense @ psi) / psi
    np.testing.assert_allclose(local, expected, atol=1.0e-11)


def test_connected_tfim_local_energy_matches_dense_reference():
    state = ARNN(3, hidden_size=5, seed=6, init_scale=0.15)
    configurations = state.all_configurations()
    connected, elements = transverse_field_ising_connections(
        configurations, coupling=0.7, field=1.2
    )
    local = np.asarray(state.local_energies(configurations, connected, elements))

    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    eye = np.eye(2)
    dense = np.zeros((8, 8))
    for site in range(2):
        operators = [eye] * 3
        operators[site] = z
        operators[site + 1] = z
        dense -= 0.7 * np.kron(np.kron(*operators[:2]), operators[2])
    for site in range(3):
        operators = [eye] * 3
        operators[site] = x
        dense -= 1.2 * np.kron(np.kron(*operators[:2]), operators[2])
    psi = state.state_vector()
    np.testing.assert_allclose(local, (dense @ psi) / psi, atol=1.0e-11)


def test_large_chain_training_path_does_not_enumerate_state_vector():
    state = ARNN(20, hidden_size=8, seed=7)

    def enumeration_is_forbidden():
        raise AssertionError("scalable VMC path enumerated the state vector")

    state.state_vector = enumeration_is_forbidden
    before = np.asarray(state.parameters["recurrent"]).copy()
    returned = state.train_step(
        heisenberg_connections,
        n_samples=32,
        learning_rate=2.0e-3,
    )

    assert returned is state
    assert np.isfinite(state.energy)
    assert np.isfinite(state.energy_variance)
    assert not np.array_equal(before, np.asarray(state.parameters["recurrent"]))
    assert state.success


def test_autoregressive_save_load_roundtrip(tmp_path):
    state = ARNN(4, hidden_size=6, seed=9)
    filename = tmp_path / "autoregressive.npz"
    state.save(filename)
    loaded = ARNN.load(filename, seed=10)

    np.testing.assert_allclose(loaded.state_vector(), state.state_vector())
    assert loaded.n_visible == state.n_visible
    assert loaded.hidden_size == state.hidden_size
