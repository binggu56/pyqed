import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_model_module():
    root = Path(__file__).resolve().parents[1]
    outer_init = (root / "__init__.py").resolve()
    loaded = sys.modules.get("pyqed")
    loaded_file_raw = getattr(loaded, "__file__", "") or ""
    loaded_file = Path(loaded_file_raw).resolve() if loaded_file_raw else None
    if loaded_file == outer_init:
        del sys.modules["pyqed"]
    sys.path.insert(0, str(root))

    path = root / "examples" / "namd" / "psgldr_avoided_crossing.py"
    spec = importlib.util.spec_from_file_location("psgldr_avoided_crossing", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_psgldr_avoided_crossing_short_time_matches_exact_populations():
    mod = _load_model_module()
    result = mod.compare_to_exact(n_gaussians=9, n_grid=161, nsteps=8, dt=0.1)

    assert result["times"].shape == (9,)
    assert result["exact_pop"].shape == result["psg_pop"].shape
    assert np.all(np.isfinite(result["psg_pop"]))
    assert np.all(result["singular_values"] > 0.0)
    assert result["pop_rms"] < 1.0e-4
    assert result["norm_drift"] < 1.0e-3


def test_moving_psgldr_avoided_crossing_advects_gaussian_centers():
    mod = _load_model_module()
    result = mod.compare_moving_to_exact(n_gaussians=5, n_grid=161, nsteps=8, dt=0.05, p0=2.0)

    assert result["force"] == "diagonal_ehrenfest"
    assert result["times"].shape == (9,)
    assert result["centers"].shape == (9, 5, 1)
    assert result["momenta"].shape == (9, 5, 1)
    assert result["center_displacement"] > 1.0e-2
    assert np.min(result["singular_values"]) > 1.0e-4
    assert result["pop_rms"] < 1.0e-2
    assert result["norm_drift"] < 3.0e-3


def test_diagonal_ehrenfest_force_uses_local_state_populations():
    mod = _load_model_module()
    model = mod.AvoidedCrossingPSGLDRModel()
    centers = np.array([[-1.0], [0.5]])
    c = np.array([[1.0, 0.0], [1.0, 2.0j]], dtype=complex)

    force = model.diagonal_ehrenfest_force(c, centers, np.zeros_like(centers), None)
    grad = model.grad_adiabatic_energies(centers[:, 0])
    expected = np.array(
        [
            -grad[0, 0],
            -(abs(c[1, 0]) ** 2 * grad[1, 0] + abs(c[1, 1]) ** 2 * grad[1, 1])
            / (abs(c[1, 0]) ** 2 + abs(c[1, 1]) ** 2),
        ]
    )

    np.testing.assert_allclose(force[:, 0], expected)
