#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the active Ehrenfest implementation.

Historically this module contained a second, partially diverged Ehrenfest
implementation.  The maintained code now lives in :mod:`pyqed.namd.mf`.
Importing from ``pyqed.namd.ehrenfest`` remains supported through the re-exports
below so older scripts do not break.
"""

from .mf import (  # noqa: F401
    AbInitioEhrenfest,
    CoupledOscillatorModel,
    Ehrenfest,
    EhrenfestTrajectory,
    GeometricEhrenfest,
    TDDFTDriver,
    TDDFTEhrenfest,
    TDDFTTrajectory,
)

__all__ = [
    "AbInitioEhrenfest",
    "CoupledOscillatorModel",
    "Ehrenfest",
    "EhrenfestTrajectory",
    "GeometricEhrenfest",
    "TDDFTDriver",
    "TDDFTEhrenfest",
    "TDDFTTrajectory",
]
