import numpy as np

from pyqed.qt import ProjectedJastrow1D, exact_quartic_ground_state


def test_harmonic_jastrow_quantum_force_cancels_classical_force():
    solver = ProjectedJastrow1D(anharmonicity=0.0)
    x = np.linspace(-2.0, 2.0, 17)
    theta = np.array((0.0, -40.0))
    _, quantum_force = solver.quantum_potential_force(x, theta)

    np.testing.assert_allclose(quantum_force, x, atol=2.0e-15)


def test_projected_force_is_negative_variational_energy_gradient():
    solver = ProjectedJastrow1D(ntraj=1024, ngrid=4001)
    theta = np.log((1.2, 0.2))
    state = solver._density_state(theta)
    gradient = solver.energy_gradient(theta)

    np.testing.assert_allclose(state["force"], -gradient, rtol=3.0e-3, atol=3.0e-5)


def test_projected_quartic_relaxation_lowers_energy():
    solver = ProjectedJastrow1D(ntraj=256, ngrid=3001).run(
        dt=0.08, max_steps=300, tolerance=2.0e-7
    )
    _, _, exact_energy = exact_quartic_ground_state(ngrid=1001)

    assert np.all(np.diff(solver.history["energy"]) <= 2.0e-12)
    assert solver.energy >= exact_energy - 2.0e-5
    assert solver.energy - exact_energy < 3.0e-3
    assert np.linalg.norm(solver.gradient, ord=np.inf) < 2.0e-3
