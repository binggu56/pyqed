"""Small GDVR RT-LDR HHG helper utilities."""

from __future__ import annotations

from pyqed.dvr import SineDVR


def sine_dvr_grid_and_kinetic(xmin, xmax, npts, mass):
    """Return the interior sine-DVR grid and kinetic matrix."""

    dvr = SineDVR(float(xmin), float(xmax), int(npts), mass=float(mass))
    return dvr.x.copy(), dvr.t().copy()
