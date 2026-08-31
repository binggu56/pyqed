import numpy as np
import pytest

from pyqed.qt import JacobianProximalFlow1D


BASELINE = np.array((-0.76831294, -0.30931345, -0.16662284, -0.91031733))


def test_jacobian_proximal_step_lowers_normalized_weak_energy():
    pytest.importorskip("jax")
    solver = JacobianProximalFlow1D(
        ntraj=128,
        seed=7,
        baseline_parameters=BASELINE,
        friction=8.0,
    )
    solver.base_x = solver.sample_initial(warmup=80)
    initial = solver.quadrature_state(ngrid=13)
    _, diagnostics = solver.proximal_step(time_step=0.04, maximum_iterations=20)
    final = solver.quadrature_state(ngrid=13)
    assert diagnostics["accepted"]
    assert final["energy"] < initial["energy"]
    assert final["minimum_determinant"] > 0.0


def test_jacobian_proximal_path_does_not_evaluate_quantum_force(monkeypatch):
    pytest.importorskip("jax")
    solver = JacobianProximalFlow1D(
        ntraj=64,
        seed=9,
        baseline_parameters=BASELINE,
    )
    monkeypatch.setattr(
        solver,
        "quantum_potential_force",
        lambda *_: (_ for _ in ()).throw(AssertionError("quantum force evaluated")),
    )
    solver.run(time_step=0.03, max_steps=1, warmup=50)
    assert solver.success
    assert np.all(solver.history["minimum_determinant"] > 0.0)
