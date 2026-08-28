"""Locally diabatic representation methods.

The coarse-grained LDR path depends on optional tensor backends.  Keep package
import light so submodules such as :mod:`pyqed.ldr.ldr` remain usable without
those optional dependencies.
"""

__all__ = [
    "CGLDR",
    "CGLDRElectronicData",
    "ElectronicPartition",
    "LDR",
    "LDRN",
    "OverlapBasis",
    "SeparableHamiltonian",
    "mps_to_array",
    "nuclear_density_distance",
    "nuclear_observables",
    "project_basis",
    "sync_gauge",
]


def __getattr__(name):
    if name in {
        "CGLDR",
        "CGLDRElectronicData",
        "ElectronicPartition",
        "OverlapBasis",
        "SeparableHamiltonian",
        "project_basis",
        "sync_gauge",
    }:
        from . import cgldr

        value = getattr(cgldr, name)
        globals()[name] = value
        return value
    if name == "LDRN":
        from .ldr import LDRN

        globals()[name] = LDRN
        return LDRN
    if name == "LDR":
        from .core import LDR

        globals()[name] = LDR
        return LDR
    if name in {"mps_to_array", "nuclear_density_distance", "nuclear_observables"}:
        from . import observables

        value = getattr(observables, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
