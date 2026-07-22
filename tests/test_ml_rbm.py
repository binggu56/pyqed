from itertools import product

import numpy as np

from pyqed.ml import RBM, RestrictedBoltzmannState


def test_rbm_amplitude_matches_explicit_hidden_spin_sum():
    rbm = RBM(2, 3, init_scale=0.0)
    rbm.visible_bias = np.array([0.12 + 0.03j, -0.2j])
    rbm.hidden_bias = np.array([-0.07 + 0.02j, 0.14, -0.09j])
    rbm.weights = np.array(
        [[0.2, -0.1j, 0.07], [0.05j, -0.16, 0.03 + 0.02j]]
    )
    visible = np.array([1.0, -1.0])

    explicit = 0.0j
    for hidden in product((-1.0, 1.0), repeat=rbm.n_hidden):
        hidden = np.asarray(hidden)
        explicit += np.exp(
            rbm.visible_bias @ visible
            + rbm.hidden_bias @ hidden
            + visible @ rbm.weights @ hidden
        )

    np.testing.assert_allclose(rbm.amplitude(visible), explicit, rtol=1e-13)


def test_rbm_log_derivative_matches_complex_finite_difference():
    rbm = RBM(3, 2, seed=4, init_scale=0.1)
    spin = np.array([1, -1, 1])
    analytic = rbm.log_derivative(spin)
    parameters = rbm.parameters
    numerical = np.empty(rbm.n_parameters, dtype=complex)
    step = 1.0e-7

    for index in range(rbm.n_parameters):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        rbm.set_parameters(plus)
        f_plus = rbm.log_amplitude(spin)
        rbm.set_parameters(minus)
        f_minus = rbm.log_amplitude(spin)
        numerical[index] = (f_plus - f_minus) / (2.0 * step)
    rbm.set_parameters(parameters)

    np.testing.assert_allclose(analytic, numerical, atol=2e-9)


def test_rbm_state_vector_expectation_and_flip_ratio():
    rbm = RestrictedBoltzmannState(2, 0, init_scale=0.0)
    rbm.visible_bias = np.array([0.3, -0.2j])
    configurations = rbm.all_configurations()
    psi = rbm.state_vector()

    assert psi.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0)
    np.testing.assert_allclose(
        rbm.flip_ratio(configurations, 0),
        rbm.amplitude(configurations * np.array([-1, 1])) / rbm.amplitude(configurations),
    )
    z0 = np.diag([1.0, 1.0, -1.0, -1.0])
    np.testing.assert_allclose(rbm.expectation(z0), np.vdot(psi, z0 @ psi))


def test_rbm_unnormalized_state_vector_returns_raw_amplitudes():
    rbm = RBM(2, 1, init_scale=0.0)
    rbm.visible_bias = np.array([1.3, -0.4j])

    np.testing.assert_allclose(
        rbm.state_vector(normalize=False),
        rbm.amplitude(rbm.all_configurations()),
    )


def test_rbm_metropolis_sampler_recovers_product_state_distribution():
    rbm = RBM(2, 0, seed=12, init_scale=0.0)
    rbm.visible_bias = np.array([0.35, -0.2])
    samples = rbm.sample(20_000, n_chains=20, burn_in=80)

    exact = np.sum(
        np.abs(rbm.state_vector()) ** 2 * rbm.all_configurations()[:, 0]
    )
    assert abs(np.mean(samples[:, 0]) - exact) < 0.035
    assert 0.0 < rbm.sampler_acceptance < 1.0


def test_rbm_stochastic_reconfiguration_lowers_exact_energy():
    rbm = RBM(1, 0, init_scale=0.0)
    rbm.visible_bias = np.array([0.4])
    hamiltonian = -np.array([[0.0, 1.0], [1.0, 0.0]])
    configurations = rbm.all_configurations()
    psi = rbm.state_vector(normalize=False)
    probabilities = np.abs(psi) ** 2
    probabilities /= np.sum(probabilities)
    local_energies = (hamiltonian @ psi) / psi
    energy_before = np.vdot(rbm.state_vector(), hamiltonian @ rbm.state_vector()).real

    returned = rbm.sr_step(
        configurations,
        local_energies,
        sample_weights=probabilities,
        learning_rate=0.1,
        diagonal_shift=1.0e-3,
    )

    energy_after = np.vdot(rbm.state_vector(), hamiltonian @ rbm.state_vector()).real
    assert returned is rbm
    assert energy_after < energy_before
    np.testing.assert_allclose(rbm.energy, energy_before)
    assert rbm.energy_variance > 0.0
    assert rbm.history[-1]["update_norm"] > 0.0
    assert rbm.success


def test_rbm_save_load_roundtrip(tmp_path):
    rbm = RBM(3, 4, seed=9)
    filename = tmp_path / "state.npz"
    rbm.save(filename)
    loaded = RBM.load(filename)

    assert loaded.n_visible == rbm.n_visible
    assert loaded.n_hidden == rbm.n_hidden
    np.testing.assert_array_equal(loaded.parameters, rbm.parameters)
    np.testing.assert_allclose(loaded.state_vector(), rbm.state_vector())
