"""Nonadiabatic dynamics with lazy optional-feature imports."""

from importlib import import_module


_EXPORTS = {
    "AbInitioEhrenfest": (".ehrenfest", "AbInitioEhrenfest"),
    "AbInitioLDRFGAdapter": (".ldrfg", "AbInitioLDRFGAdapter"),
    "BornHuang": (".bh", "BornHuang"),
    "BornHuang2": (".bh", "BornHuang2"),
    "CoupledOscillatorModel": (".ehrenfest", "CoupledOscillatorModel"),
    "Ehrenfest": (".ehrenfest", "Ehrenfest"),
    "EhrenfestTrajectory": (".ehrenfest", "EhrenfestTrajectory"),
    "GeometricEhrenfest": (".ehrenfest", "GeometricEhrenfest"),
    "LDRFG": (".ldrfg", "LDRFG"),
    "LDRFGRHS": (".ldrfg", "LDRFGRHS"),
    "TDDFTDriver": (".ehrenfest", "TDDFTDriver"),
    "TDDFTEhrenfest": (".ehrenfest", "TDDFTEhrenfest"),
    "TDDFTTrajectory": (".ehrenfest", "TDDFTTrajectory"),
    "Triatom": (".triatomic", "Triatom"),
    "Triatomic": (".triatomic", "Triatomic"),
    "grad_overlap_from_derivative_couplings": (
        ".ldrfg",
        "grad_overlap_from_derivative_couplings",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
