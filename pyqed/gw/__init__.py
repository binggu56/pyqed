"""Molecular and periodic GW/BSE helpers.

Production drivers expose GW, BSE and TDA. Opt-in prototypes live in
``experimental``; independent comparison implementations live in ``reference``.

The canonical periodic Gaussian GW/BSE API lives in :mod:`pyqed.pbc.gw`.
The ``pyqed.gw.pbc`` namespace is kept as a compatibility alias.
"""

from importlib import import_module

__all__ = [
    "BSE",
    "GW",
    "TDA",
    "pbc",
]


def __getattr__(name):
    if name == "GW":
        from .gw import GW

        return GW
    if name in {"BSE", "TDA"}:
        from .bse import BSE, TDA

        return {"BSE": BSE, "TDA": TDA}[name]
    if name == "pbc":
        return import_module(f"{__name__}.pbc")
    raise AttributeError(name)
