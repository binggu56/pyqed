"""Compatibility namespace for hierarchy solvers.

Use :mod:`pyqed.heom` for the canonical API.
"""

from pyqed.heom import Bath, HEOM, HighTemperatureHEOM
from pyqed.heom.deom import *  # noqa: F401,F403

