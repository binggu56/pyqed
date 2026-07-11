"""Many-body perturbation-theory entry points."""

from __future__ import annotations

from importlib import import_module


_LAZY_ATTRS = {
    "GW": ("pyqed.gw.gw", "GW"),
    "BSE": ("pyqed.gw.bse", "BSE"),
    "TDA": ("pyqed.gw.bse", "TDA"),
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
