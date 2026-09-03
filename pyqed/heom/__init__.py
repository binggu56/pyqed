"""Hierarchical equation of motion solvers."""

from __future__ import annotations

from importlib import import_module

_LAZY_ATTRS = {
    "HEOM": ("pyqed.heom.deom", "HEOM"),
    "HighTemperatureHEOM": ("pyqed.heom.heom", "HighTemperatureHEOM"),
    "Bath": ("pyqed.heom.deom", "Bath"),
    "fit_spectrum_prony": ("pyqed.heom.deom", "fit_spectrum_prony"),
    "prony": ("pyqed.heom.deom", "prony"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_LAZY_ATTRS)
