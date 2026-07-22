"""Periodic-system entry points.

This namespace collects periodic workflows without mixing them into the
molecular packages.  The Gaussian periodic GW/BSE drivers live under
``pyqed.pbc.gw``; the native periodic cell and SCF objects are lazily
re-exported from :mod:`pyqed.qchem.pbc` for convenience.
"""

from importlib import import_module

__all__ = [
    "Cell",
    "Chain",
    "EwaldRHF",
    "KRHF",
    "RHF",
]


def __getattr__(name):
    if name in __all__:
        module = import_module("pyqed.qchem.pbc")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
