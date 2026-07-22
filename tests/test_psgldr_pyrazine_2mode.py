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

    path = root / "examples" / "namd" / "psgldr_pyrazine_2mode.py"
    spec = importlib.util.spec_from_file_location("psgldr_pyrazine_2mode", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_moving_psgldr_pyrazine_2mode_short_time_smoke():
    mod = _load_model_module()
    result = mod.run_psgldr_pyrazine(
        n_per_dim=3,
        center_qmax=2.5,
        dvr_npts=13,
        dvr_qmax=5.0,
        dt=1.0,
        nsteps=4,
        moving=True,
    )

    assert result["times"].shape == (5,)
    assert result["exact_pop"].shape == result["psg_pop"].shape
    assert np.all(np.isfinite(result["psg_pop"]))
    assert np.min(result["singular_values"]) > 1.0e-2
    assert result["pop_rms"] < 1.0e-3
    assert result["norm_drift"] < 1.0e-3
