import importlib.util
import sys
from pathlib import Path

import numpy as np


def _prefer_source_package():
    root = Path(__file__).resolve().parents[1]
    outer_init = (root / "__init__.py").resolve()
    loaded = sys.modules.get("pyqed")
    loaded_file_raw = getattr(loaded, "__file__", "") or ""
    loaded_file = Path(loaded_file_raw).resolve() if loaded_file_raw else None
    if loaded_file == outer_init:
        del sys.modules["pyqed"]
    sys.path.insert(0, str(root))
    return root


def _load_model_module():
    root = _prefer_source_package()
    path = root / "examples" / "namd" / "ldrfg_avoided_crossing.py"
    spec = importlib.util.spec_from_file_location("ldrfg_avoided_crossing", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _second_derivative_kinetic(n, dx, mass=1.0):
    kinetic = np.diag(np.full(n, 1.0 / (mass * dx * dx)))
    kinetic += np.diag(np.full(n - 1, -0.5 / (mass * dx * dx)), k=1)
    kinetic += np.diag(np.full(n - 1, -0.5 / (mass * dx * dx)), k=-1)
    return kinetic


def test_avoided_crossing_overlap_is_orthonormal_on_same_grid_point():
    mod = _load_model_module()
    x = np.linspace(-2.0, 2.0, 5)
    tx = _second_derivative_kinetic(x.size, x[1] - x[0])
    model = mod.AvoidedCrossingLDRFGModel(x, tx)

    overlap = model.overlap(q=[0.3])
    for n in range(x.size):
        np.testing.assert_allclose(overlap[n, :, n, :], np.eye(2), atol=1e-12)


def test_avoided_crossing_analytic_force_matches_energy_finite_difference():
    mod = _load_model_module()
    x = np.linspace(-2.5, 2.5, 7)
    tx = _second_derivative_kinetic(x.size, x[1] - x[0], mass=2.0)
    model = mod.AvoidedCrossingLDRFGModel(
        x,
        tx,
        mass_y=4.0,
        a_x=0.9,
        a_q=0.7,
        delta=0.25,
        k_x=0.01,
        k_q=0.04,
    )
    solver = model.solver(include_berry=False)

    c = np.zeros((x.size, 2), dtype=complex)
    c[:, 0] = np.exp(-0.5 * ((x + 0.8) / 0.8) ** 2)
    c[:, 1] = 0.2j * np.exp(-0.5 * ((x - 0.7) / 0.9) ** 2)
    c /= np.sqrt(np.vdot(c.ravel(), c.ravel()))

    q = np.array([0.15])
    p = np.array([0.4])
    rhs = solver.rhs(c, q, p)
    fd_force = mod.finite_difference_force(solver, c, q, p, eps=2.0e-6)

    np.testing.assert_allclose(rhs.p_dot[0], fd_force, rtol=2e-6, atol=2e-8)


def test_avoided_crossing_demo_runs_and_keeps_state_normalized():
    mod = _load_model_module()
    result = mod.run_demo(nsteps=5, dt=0.01)

    assert result["times"].shape == (6,)
    assert result["q"].shape == (6,)
    assert result["p"].shape == (6,)
    assert np.all(np.isfinite(result["energy"]))


def test_avoided_crossing_compares_to_exact_quantum_reference():
    mod = _load_model_module()
    result = mod.compare_to_exact(nsteps=4, dt=0.005, nq=31)

    assert result["ldrfg"]["q"].shape == result["exact"]["q"].shape
    assert result["ldrfg"]["p"].shape == result["exact"]["p"].shape
    assert result["ldrfg"]["pop_ad"].shape == result["exact"]["pop_ad"].shape
    assert result["diff"]["q_rms"] < 0.05
    assert result["diff"]["p_rms"] < 0.2
    assert np.isfinite(result["diff"]["pop_ad_rms"])
