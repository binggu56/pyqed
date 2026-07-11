import importlib.util
import sys
from pathlib import Path


def _load_readiness_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path = [str(repo_root)] + [item for item in sys.path if item != str(repo_root)]
    loaded_pyqed = sys.modules.get("pyqed")
    if loaded_pyqed is not None:
        loaded_path = Path(getattr(loaded_pyqed, "__file__", "")).resolve()
        expected_path = (repo_root / "pyqed" / "__init__.py").resolve()
        if loaded_path != expected_path:
            for name in [name for name in sys.modules if name == "pyqed" or name.startswith("pyqed.")]:
                del sys.modules[name]
    path = repo_root / "examples" / "qchem" / "benchmark_su2_readiness.py"
    spec = importlib.util.spec_from_file_location("benchmark_su2_readiness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_su2_readiness_h2_matches_exact_target_energy():
    readiness = _load_readiness_module()

    result = readiness.run_case(
        "h2",
        bond_dim=8,
        nsweeps=2,
        energy_tol=1.0e-7,
        orthonormalized_operator_dim=128,
    )

    assert result.passed
    assert result.target_charge == 2
    assert result.target_two_s == 0
    assert result.energy_error <= 1.0e-7
    assert result.mps_energy_error <= 1.0e-7


def test_su2_readiness_can_force_factor_family_kernel_with_dense_budget():
    readiness = _load_readiness_module()

    result = readiness.run_case(
        "h4",
        bond_dim=16,
        nsweeps=2,
        energy_tol=1.0e-7,
        family_dense_max_total_elements=0,
    )

    assert result.passed
    assert result.family_backend_counts in ({}, {"family_table_factor_kernel": 3})
    assert result.family_native_kernel_elements == 0
    if result.family_backend_counts:
        assert result.family_factor_kernel_elements > 0
        assert result.family_dense_skipped_total_budget > 0
