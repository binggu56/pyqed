"""Locally diabatic representation methods.

The coarse-grained LDR path depends on optional tensor backends. Keep package
imports light so the core solver remains usable without those dependencies.
"""

__all__ = [
    "keo",
    "Coord",
    "AbInitioFit",
    "ElectronicDatabase",
    "SamplingSymmetry",
    "SamplingSymmetryImage",
    "FiniteGroupSamplingSymmetry",
    "PhenolReflectionSymmetry",
    "PhenolSACASSCFProvider",
    "PeriodicSSHHolsteinHalfFilledScan",
    "PeriodicSSHHolsteinGQD",
    "PeriodicSSHHolsteinMomentumGQD",
    "PhenolCASSCFOverlap",
    "phenol_sa6_protocol",
    "ETHYLENE_CI_BOUNDS",
    "ETHYLENE_MECI_ANGSTROM",
    "ETHYLENE_MECI_BOHR",
    "ETHYLENE_CI_PYRAMID_SHIFT",
    "ETHYLENE_SPECIES",
    "EthyleneCIElectronicDriver",
    "default_ethylene_database_path",
    "ethylene_ci_geometry",
    "ethylene_ci_protocol",
    "core",
    "solver",
    "CGLDR",
    "CGLDRElectronicData",
    "DiagonalElectronicContinuum",
    "ElectronicPartition",
    "FeshbachEmbedding",
    "FEMLDR",
    "GraphLDR",
    "GraphMesh",
    "LDR",
    "MatrixElectronicContinuum",
    "OverlapBasis",
    "SeparableHamiltonian",
    "TriangularMesh",
    "mps_to_array",
    "nuclear_density_distance",
    "nuclear_observables",
    "project_basis",
    "sync_gauge",
]


def __getattr__(name):
    import importlib
    if name == "core":
        module = importlib.import_module(".core", __name__)
        globals()[name] = module
        return module
    if name == "solver":
        module = importlib.import_module(".solver", __name__)
        globals()[name] = module
        return module
    if name == "AbInitioFit":
        from .abinitio import AbInitioFit

        globals()[name] = AbInitioFit
        return AbInitioFit
    if name == "ElectronicDatabase":
        from .database import ElectronicDatabase

        globals()[name] = ElectronicDatabase
        return ElectronicDatabase
    if name in {
        "SamplingSymmetry",
        "SamplingSymmetryImage",
        "FiniteGroupSamplingSymmetry",
        "PhenolReflectionSymmetry",
    }:
        from . import sampling_symmetry

        value = getattr(sampling_symmetry, name)
        globals()[name] = value
        return value
    if name in {
        "PhenolSACASSCFProvider",
        "PhenolCASSCFOverlap",
        "phenol_sa6_protocol",
    }:
        from . import phenol

        value = getattr(phenol, name)
        globals()[name] = value
        return value
    if name in {
        "ETHYLENE_CI_BOUNDS",
        "ETHYLENE_MECI_ANGSTROM",
        "ETHYLENE_MECI_BOHR",
        "ETHYLENE_CI_PYRAMID_SHIFT",
        "ETHYLENE_SPECIES",
        "EthyleneCIElectronicDriver",
        "default_ethylene_database_path",
        "ethylene_ci_geometry",
        "ethylene_ci_protocol",
    }:
        from . import ethylene

        value = getattr(ethylene, name)
        globals()[name] = value
        return value

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
    if name == "LDR":
        from .core import LDR

        globals()[name] = LDR
        return LDR
    if name == "Coord":
        from .coord import Coord

        globals()[name] = Coord
        return Coord
    if name in {
        "DiagonalElectronicContinuum",
        "FeshbachEmbedding",
        "MatrixElectronicContinuum",
    }:
        from . import continuum

        value = getattr(continuum, name)
        globals()[name] = value
        return value
    if name in {
        "PeriodicSSHHolsteinGQD",
        "PeriodicSSHHolsteinMomentumGQD",
    }:
        from . import periodic

        value = getattr(periodic, name)
        globals()[name] = value
        return value
    if name == "PeriodicSSHHolsteinHalfFilledScan":
        from .periodic_scan import PeriodicSSHHolsteinHalfFilledScan

        globals()[name] = PeriodicSSHHolsteinHalfFilledScan
        return PeriodicSSHHolsteinHalfFilledScan
    if name in {"GraphLDR", "GraphMesh"}:
        from . import graph

        value = getattr(graph, name)
        globals()[name] = value
        return value
    if name in {"FEMLDR", "TriangularMesh"}:
        from . import fem

        value = getattr(fem, name)
        globals()[name] = value
        return value
    if name in {"mps_to_array", "nuclear_density_distance", "nuclear_observables"}:
        from . import observables

        value = getattr(observables, name)
        globals()[name] = value
        return value
    if name == "keo":
        module = importlib.import_module(".keo", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
