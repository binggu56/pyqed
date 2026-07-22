"""Compatibility wrapper for :mod:`pyqed.pbc.gw`.

New periodic GW/BSE code should import from ``pyqed.pbc.gw``.  This module
keeps the older development path usable while the package layout settles.
"""

from importlib import import_module
import sys

from pyqed.pbc.gw import *  # noqa: F401,F403
from pyqed.pbc.gw import __all__

for _name in (
    "adapter",
    "bse",
    "coulomb",
    "integrals",
    "kbse",
    "kgw",
    "response",
    "self_energy",
):
    sys.modules[f"{__name__}.{_name}"] = import_module(f"pyqed.pbc.gw.{_name}")
