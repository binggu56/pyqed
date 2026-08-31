import numpy as np
import pytest

from pyqed.qt import ProjectedTwoParticleJastrow1D


def test_two_particle_quantum_force_matches_potential_gradient():
    solver = ProjectedTwoParticleJastrow1D(ntraj=64, ngrid=81)
    theta = np.log((1.1, 0.6, 0.7))
    point = np.array((0.37, -0.58))
    step = 2.0e-5
    _, analytic = solver.quantum_potential_force(point[0], point[1], theta)
    numerical = np.empty(2)
    for coordinate in range(2):
        plus, minus = point.copy(), point.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        qplus = solver.quantum_potential_force(plus[0], plus[1], theta)[0]
        qminus = solver.quantum_potential_force(minus[0], minus[1], theta)[0]
        numerical[coordinate] = -(qplus - qminus) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-8, atol=2.0e-9)


def test_ad_quantum_force_matches_analytic_backend():
    pytest.importorskip("jax")
    solver = ProjectedTwoParticleJastrow1D(ntraj=64, ngrid=81)
    theta = np.log((1.1, 0.6, 0.7))
    x1 = np.array((0.37, -0.58, 0.11))
    x2 = np.array((-0.58, 0.21, 0.93))
    analytic_q, analytic_force = solver.quantum_potential_force(
        x1, x2, theta, backend="analytic"
    )
    ad_q, ad_force = solver.quantum_potential_force(
        x1, x2, theta, backend="ad"
    )
    np.testing.assert_allclose(ad_q, analytic_q, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(ad_force, analytic_force, rtol=2.0e-11, atol=2.0e-11)


def test_transport_tangent_force_matches_energy_gradient():
    solver = ProjectedTwoParticleJastrow1D(
        ntraj=16384, seed=11, ngrid=121, tangent_backend="transport"
    )
    theta = np.log((1.2, 0.5, 0.5))
    state = solver._state(theta)
    np.testing.assert_allclose(
        state["sampled_force"], -state["gradient"], rtol=1.0e-1, atol=5.0e-3
    )


def test_monte_carlo_estimators_approach_grid_reference():
    solver = ProjectedTwoParticleJastrow1D(ntraj=32768, seed=19, ngrid=121)
    theta = np.log((1.2, 0.5, 0.5))
    state = solver._state(theta, with_tangents=False)
    _, grid_energy, _, grid_gradient = solver._grid_state(theta)
    np.testing.assert_allclose(state["energy"], grid_energy, atol=1.5e-2)
    np.testing.assert_allclose(state["gradient"], grid_gradient, atol=2.0e-2)


def test_real_trajectories_keep_weights_and_are_not_remapped():
    solver = ProjectedTwoParticleJastrow1D(ntraj=1024, seed=31, ngrid=101)
    weights = solver.weights.copy()
    solver.run(
        dt=0.01,
        max_steps=4,
        tolerance=0.0,
        record_every=1,
        parameter_closure="coupled",
    )
    transported = solver._transport(solver._density_grid(solver.theta))

    np.testing.assert_array_equal(solver.weights, weights)
    np.testing.assert_allclose(np.sum(solver.weights), 1.0)
    assert solver.transport_drift > 0.0
    assert np.sqrt(np.mean((solver.x - transported) ** 2)) > 0.0
    np.testing.assert_allclose(solver.history["transport_drift"][0], 0.0, atol=1.0e-14)


def test_stein_identity_reconstructs_jastrow_coefficients():
    solver = ProjectedTwoParticleJastrow1D(ntraj=32768, seed=41, ngrid=121)
    expected = np.array((1.15, 0.72, 0.43))
    trajectories = solver.sample_initial(np.log(expected))
    theta, diagnostics = solver.reconstruct_parameters(trajectories)

    np.testing.assert_allclose(np.exp(theta), expected, rtol=1.5e-2, atol=5.0e-3)
    assert diagnostics["condition"] < 1.0e4


def test_stein_closure_reconstructs_parameters_from_carried_cloud():
    solver = ProjectedTwoParticleJastrow1D(ntraj=2048, seed=43, ngrid=101).run(
        dt=0.01, max_steps=3, tolerance=0.0, record_every=1
    )
    reconstructed, diagnostics = solver.reconstruct_parameters(solver.x)

    np.testing.assert_allclose(solver.theta, reconstructed, atol=1.0e-13)
    np.testing.assert_allclose(solver.stein_residual, diagnostics["residual"])
    assert np.all(np.isfinite(solver.history["stein_condition"]))
    np.testing.assert_allclose(
        solver.history["parameter_velocity"][-1], solver.parameter_velocity
    )
    assert np.max(np.abs(solver.kinematic_error)) < 1.0e-6


def test_implicit_stein_jacobian_matches_coordinate_finite_difference():
    solver = ProjectedTwoParticleJastrow1D(ntraj=256, seed=47, ngrid=101)
    trajectories = solver.sample_initial(np.log((1.1, 0.7, 0.4)))
    jacobian, _ = solver.stein_reconstruction_jacobian(trajectories)
    step = 2.0e-5
    for particle, coordinate in ((3, 0), (17, 1), (83, 0)):
        plus, minus = trajectories.copy(), trajectories.copy()
        plus[particle, coordinate] += step
        minus[particle, coordinate] -= step
        theta_plus = solver.reconstruct_parameters(plus)[0]
        theta_minus = solver.reconstruct_parameters(minus)[0]
        numerical = (theta_plus - theta_minus) / (2.0 * step)
        np.testing.assert_allclose(
            jacobian[particle, coordinate], numerical, rtol=2.0e-5, atol=2.0e-8
        )


def test_minimum_norm_stein_lift_is_kinematically_consistent():
    solver = ProjectedTwoParticleJastrow1D(ntraj=1024, seed=53, ngrid=101)
    trajectories = solver.sample_initial(np.log((1.1, 0.7, 0.4)))
    _, tangents, metric, diagnostics = solver.stein_tangent_lift(trajectories)
    np.testing.assert_allclose(
        diagnostics["lift_identity"], np.eye(3), rtol=2.0e-7, atol=2.0e-7
    )
    np.testing.assert_allclose(
        metric,
        solver.mass
        * np.einsum("n,nka,nkb->ab", solver.weights, tangents, tangents),
        rtol=2.0e-7,
        atol=2.0e-7,
    )


def test_constrained_continuity_lift_matches_vmc_gradient_and_reconstruction():
    solver = ProjectedTwoParticleJastrow1D(ntraj=2048, seed=59, ngrid=101)
    trajectories = solver.sample_initial(np.log((1.1, 0.7, 0.4)))
    theta, tangents, metric, diagnostics = solver.constrained_continuity_lift(
        trajectories
    )
    state = solver._sample_state(
        theta,
        trajectories,
        tangent_data=(tangents, metric, diagnostics),
    )

    np.testing.assert_allclose(
        diagnostics["lift_identity"], np.eye(3), rtol=2.0e-9, atol=2.0e-9
    )
    np.testing.assert_allclose(
        diagnostics["optimality_residual"], 0.0, rtol=2.0e-8, atol=2.0e-8
    )
    np.testing.assert_allclose(
        state["sampled_force"], -state["gradient"], rtol=2.0e-9, atol=2.0e-9
    )


def test_constrained_continuity_flow_is_an_energy_descent_direction():
    solver = ProjectedTwoParticleJastrow1D(ntraj=2048, seed=61, ngrid=101)
    trajectories = solver.sample_initial(np.log((1.2, 0.5, 0.5)))
    _, _, parameter_velocity, state, _ = solver._stein_flow_rhs(trajectories)

    energy_rate = np.dot(state["gradient"], parameter_velocity)
    assert energy_rate < 0.0
    np.testing.assert_allclose(
        state["force"], -state["gradient"], rtol=2.0e-9, atol=2.0e-9
    )


def test_neural_transport_features_work_in_constrained_trajectory_solver():
    pytest.importorskip("jax")
    solver = ProjectedTwoParticleJastrow1D(
        ntraj=256, seed=67, ngrid=81, transport_basis="neural"
    )
    trajectories = solver.sample_initial(np.log((1.2, 0.5, 0.5)))
    theta, _ = solver.reconstruct_parameters(trajectories)
    model = solver.train_neural_transport(
        trajectories, theta, steps=30, learning_rate=2.0e-3
    )
    _, tangents, metric, diagnostics = solver.constrained_continuity_lift(
        trajectories, theta
    )
    state = solver._sample_state(
        theta,
        trajectories,
        tangent_data=(tangents, metric, diagnostics),
    )

    assert model.loss_history[-1] < model.loss_history[0]
    assert diagnostics["retained_basis_rank"] >= 4
    assert diagnostics["constraint_condition"] < 1.0e5
    np.testing.assert_allclose(
        diagnostics["lift_identity"], np.eye(3), atol=2.0e-9
    )
    np.testing.assert_allclose(
        state["force"], -state["gradient"], atol=2.0e-9
    )
