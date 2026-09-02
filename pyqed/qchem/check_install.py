"""Verify that an installed PyQED has the production qchem accelerators."""

from __future__ import annotations

import importlib


REQUIRED_ACCELERATORS = (
    (
        "AO integral engine",
        "pyqed.qchem._integrals_cpp",
        (
            "compute_dense_eri_spherical",
            "compute_eri_s8_cartesian",
            "direct_jk_spherical",
            "contract_jk_s8",
            "compute_ri_tensors_packed",
        ),
    ),
    (
        "one-electron and RI kernels",
        "pyqed.qchem._basis_cy",
        ("compute_one_electron", "compute_pair_bounds", "compute_ri_tensors_packed"),
    ),
    (
        "Rys kernels",
        "pyqed.qchem._rys_cy",
        ("compute_dense_eri_blocked_rys", "direct_jk_spherical_sp_rys"),
    ),
    ("CAS kernels", "pyqed.qchem._casscf_cpp", ("ci_hamiltonian",)),
    (
        "periodic density-fitting kernels",
        "pyqed.qchem._gdf_cpp",
        ("gaussian_ft_batch", "periodic_pair_ft_primitive_sum_many"),
    ),
)


def accelerator_status():
    """Return ``(label, module, available, detail)`` entries for qchem kernels."""
    status = []
    for label, module_name, required_symbols in REQUIRED_ACCELERATORS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            status.append((label, module_name, False, f"{type(exc).__name__}: {exc}"))
            continue
        missing = [name for name in required_symbols if not hasattr(module, name)]
        if missing:
            status.append((label, module_name, False, f"missing {', '.join(missing)}"))
            continue
        status.append((label, module_name, True, module.__file__ or "loaded"))
    return status


def main():
    status = accelerator_status()
    for label, module_name, available, detail in status:
        state = "OK" if available else "MISSING"
        print(f"{state:7} {label}: {module_name} ({detail})")
    if all(item[2] for item in status):
        print("PyQED qchem production path is available.")
        return 0
    print(
        "PyQED qchem production path is incomplete. Install a platform wheel or "
        "rebuild from source with a supported C/C++ toolchain."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
