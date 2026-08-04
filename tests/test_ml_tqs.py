import numpy as np

from pyqed.ml import TQS, heisenberg_connections


def _heisenberg_dense(n_visible):
    configurations = TQS(
        n_visible,
        d_model=4,
        n_heads=1,
        n_layers=1,
        seed=0,
        init_scale=0.0,
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


def test_four_spin_tqs_normalization_and_direct_sampling():
    state = TQS(4, d_model=8, n_heads=2, n_layers=1, seed=3)
    probabilities = np.abs(state.state_vector()) ** 2
    samples = state.sample(30_000, seed=4)
    labels = ((samples == -1) * np.array([8, 4, 2, 1])).sum(axis=1)
    observed = np.bincount(labels, minlength=16) / samples.shape[0]

    np.testing.assert_allclose(np.sum(probabilities), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(observed, probabilities, atol=1.0e-2)


def test_four_spin_tqs_local_energy_matches_dense_heisenberg():
    state = TQS(4, d_model=8, n_heads=2, n_layers=1, seed=5)
    configurations = state.all_configurations()
    connected, elements = heisenberg_connections(configurations)
    local = np.asarray(state.local_energies(configurations, connected, elements))
    psi = state.state_vector()

    np.testing.assert_allclose(
        local,
        (_heisenberg_dense(4) @ psi) / psi,
        atol=1.0e-11,
    )


def test_four_spin_tqs_vmc_step_is_finite_and_updates_parameters():
    state = TQS(4, d_model=8, n_heads=2, n_layers=1, seed=6)
    before = np.asarray(state.parameters["wq"]).copy()
    returned = state.train_step(
        heisenberg_connections,
        n_samples=512,
        learning_rate=2.0e-3,
    )

    assert returned is state
    assert np.isfinite(state.energy)
    assert np.isfinite(state.energy_variance)
    assert not np.array_equal(before, np.asarray(state.parameters["wq"]))
    assert state.success


def test_four_spin_tqs_save_load_roundtrip(tmp_path):
    state = TQS(4, d_model=8, n_heads=2, n_layers=1, seed=8)
    filename = tmp_path / "four_spin_tqs.npz"
    state.save(filename)
    loaded = TQS.load(filename, seed=9)

    np.testing.assert_allclose(loaded.state_vector(), state.state_vector())
    assert loaded.d_model == 8
    assert loaded.n_heads == 2
    assert loaded.n_layers == 1
