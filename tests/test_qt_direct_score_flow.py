import numpy as np
import pytest

from pyqed.qt import (
    DirectOverdampedScoreFlow1D,
    SharedLinearScoreCorrection1D,
    optimize_global_double_well_jastrow,
)


BASELINE = np.array((-0.76831294, -0.30931345, -0.16662284, -0.91031733))


def test_direct_score_flow_uses_full_cartesian_residual_force():
    solver = DirectOverdampedScoreFlow1D(
        ntraj=128, seed=9, closure="linear", baseline_parameters=BASELINE
    )
    coordinates = solver.sample_initial(warmup=80)
    solver.fit_closure(coordinates)
    state = solver.state(coordinates)
    velocity = state["residual_force"] / solver.friction
    assert velocity.shape == coordinates.shape
    assert not hasattr(solver, "parameter_velocity")
    assert not hasattr(solver, "tangents")


def test_regularized_linear_online_score_flow_stays_finite_and_lowers_energy():
    solver = DirectOverdampedScoreFlow1D(
        ntraj=256,
        seed=11,
        closure="linear",
        baseline_parameters=BASELINE,
        friction=20.0,
    ).run(
        dt=0.002,
        macro_steps=8,
        max_displacement=0.003,
        warmup=120,
    )
    assert np.all(np.isfinite(solver.history["energy"]))
    assert np.all(np.isfinite(solver.history["force_rms"]))
    assert solver.history["energy"][-1] < solver.history["energy"][0]


def test_global_double_well_jastrow_has_room_for_beyond_jastrow_improvement():
    result = optimize_global_double_well_jastrow(ngrid=35)
    assert np.isfinite(result.fun)
    assert result.fun < 5.55


def test_neural_quantum_force_is_finite_after_force_regularized_score_fit():
    pytest.importorskip("jax")
    solver = DirectOverdampedScoreFlow1D(
        ntraj=96, seed=13, closure="neural", baseline_parameters=BASELINE
    )
    coordinates = solver.sample_initial(warmup=60)
    solver.neural_model.fit(
        coordinates,
        BASELINE,
        steps=5,
        learning_rate=1.0e-4,
        correction_regularization=1.0e-2,
        force_smoothness=1.0e-1,
    )
    quantum_potential, quantum_force = solver.quantum_potential_force(coordinates)
    assert np.all(np.isfinite(quantum_potential))
    assert np.all(np.isfinite(quantum_force))
    assert np.max(np.linalg.norm(quantum_force, axis=1)) < 1.0e3

