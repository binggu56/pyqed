#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for linear-response TDDFT.

The canonical module is now :mod:`pyqed.qchem.tddft`.  Importing from
``pyqed.qchem.lrtddft`` remains supported so older code does not break.
"""

from .tddft import *  # noqa: F401,F403
