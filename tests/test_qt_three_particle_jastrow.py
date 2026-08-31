import numpy as np

from pyqed.qt import (
    ProjectedThreeParticleJastrow1D,
    exact_three_particle_ground_state,
)


def test_three_particle_quantum_force_matches_potential_gradient():
    solver = ProjectedThreeParticleJastrow1D(ntraj=64, ngrid=25)
    theta = np.log((1.1, 0.8, 0.3))
    coordinates = np.array(((0.31, -0.47, 0.72),))
    _, analytic = solver.quantum_potential_force(coordinates, theta)
    numerical = np.empty(3)
    step = 2.0e-5
    for coordinate in range(3):
        plus, minus = coordinates.copy(), coordinates.copy()
        plus[0, coordinate] += step
        minus[0, coordinate] -= step
        qplus = solver.quantum_potential_force(plus, theta)[0]
        qminus = solver.quantum_potential_force(minus, theta)[0]
        numerical[coordinate] = -(qplus[0] - qminus[0]) / (2.0 * step)
    np.testing.assert_allclose(analytic[0], numerical, rtol=2.0e-8, atol=2.0e-9)


def test_three_particle_constrained_lift_preserves_parameter_and_energy_identities():
    solver = ProjectedThreeParticleJastrow1D(ntraj=512, seed=5, ngrid=25)
    trajectories = solver.sample_initial(
        np.log((1.2, 0.7, 0.35)), warmup=120
    )
    theta, _ = solver.reconstruct_parameters(trajectories)
    _, tangents, metric, diagnostics = solver.constrained_continuity_lift(
        trajectories, theta
    )
    state = solver._sample_state(
        theta, trajectories, (tangents, metric, diagnostics)
    )
    np.testing.assert_allclose(
        diagnostics["lift_identity"], np.eye(3), atol=2.0e-9
    )
    np.testing.assert_allclose(state["force"], -state["gradient"], atol=2.0e-9)


def test_three_particle_jastrow_energy_is_variationally_close_to_exact_grid():
    solver = ProjectedThreeParticleJastrow1D(ntraj=64, ngrid=45)
    variational = solver.optimize_grid_jastrow()
    _, _, exact_energy = exact_three_particle_ground_state(ngrid=61)
    assert variational.fun >= exact_energy
    assert variational.fun - exact_energy < 1.5e-2
