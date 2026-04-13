#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal libxc wrapper for restricted DFT functionals.

This module intentionally exposes only the small subset of functionality
needed by pyqed's native AO-based DFT layer.
"""

from ctypes import c_char_p, c_double, c_int, c_void_p, cdll
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import numpy as np


_SUPPORTED_XC = {
    'b3lyp': {
        'libxc_name': 'HYB_GGA_XC_B3LYP',
        'xctype': 'GGA',
        'hyb': 0.20,
    },
    'hyb_gga_xc_b3lyp': {
        'libxc_name': 'HYB_GGA_XC_B3LYP',
        'xctype': 'GGA',
        'hyb': 0.20,
    },
}


def _find_libxc_backend():
    """
    Locate a local libxc interface shared library.
    """
    candidates = []

    spec = find_spec('pyscf')
    if spec is not None and spec.origin is not None:
        root = Path(spec.origin).resolve().parent
        candidates.extend([
            root / 'lib' / 'libxc_itrf.dylib',
            root / 'lib' / 'libxc_itrf.so',
        ])

    for path in candidates:
        if path.exists():
            return path

    raise OSError(
        "Could not locate a local libxc interface shared library. "
        "Install libxc or a local package that bundles libxc."
    )


class _LibXC:
    def __init__(self):
        self._lib = cdll.LoadLibrary(str(_find_libxc_backend()))
        self._lib.LIBXC_xc_func_init.restype = c_void_p
        self._lib.LIBXC_xc_func_end.argtypes = (c_int, c_void_p)
        self._lib.LIBXC_eval_xc.argtypes = (
            c_int, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_int,
            c_void_p, c_void_p,
        )
        self._lib.xc_functional_get_number.argtypes = (c_char_p,)
        self._lib.xc_functional_get_number.restype = c_int

    def functional_number(self, name):
        return int(self._lib.xc_functional_get_number(name.encode()))

    def init_functional(self, number):
        numbers = (c_int * 1)(number)
        return self._lib.LIBXC_xc_func_init(1, numbers, 0)

    def end_functional(self, handle):
        self._lib.LIBXC_xc_func_end(1, handle)

    def eval_xc(self, handle, fac, rho, nvar, deriv=1):
        ngrids = int(rho.shape[-1])
        outlen = 2 if nvar == 1 else 3
        rho = np.asarray(rho, dtype=np.double, order='C').reshape(1, nvar, ngrids)
        out = np.zeros((outlen, ngrids), dtype=np.double, order='C')
        factors = (c_double * 1)(fac)
        self._lib.LIBXC_eval_xc(
            1,
            handle,
            factors,
            0,
            deriv,
            nvar,
            ngrids,
            outlen,
            rho.ctypes.data_as(c_void_p),
            out.ctypes.data_as(c_void_p),
        )
        return out


LIBXC = _LibXC()


class RestrictedLibXCFunctional:
    """
    Cached restricted-spin libxc functional handle.
    """

    def __init__(self, xc):
        self.name = xc.lower()
        if self.name not in _SUPPORTED_XC:
            raise ValueError(f"Unsupported libxc functional '{xc}'.")

        info = _SUPPORTED_XC[self.name]
        libxc_name = info['libxc_name']
        number = LIBXC.functional_number(libxc_name)
        if number < 0:
            raise ValueError(f"Unsupported libxc functional '{xc}'.")

        self.handle = LIBXC.init_functional(number)
        if self.handle is None:
            raise ValueError(f"Failed to initialize libxc functional '{xc}'.")

        self.xctype = info['xctype']
        self.nvar = 1 if self.xctype == 'LDA' else 4
        self.hyb = info['hyb']

    def eval(self, rho, grad_rho=None):
        """
        Evaluate ``exc`` and first derivatives for a restricted density.
        """
        rho = np.asarray(rho, dtype=float)
        if self.xctype == 'LDA':
            out = LIBXC.eval_xc(self.handle, 1.0, rho, nvar=1, deriv=1)
            return out[0], out[1]

        if grad_rho is None:
            raise ValueError("GGA functionals require density gradients.")
        grad_rho = np.asarray(grad_rho, dtype=float)
        if grad_rho.shape[0] != 3:
            raise ValueError("grad_rho must have shape (3, ngrids).")

        out = LIBXC.eval_xc(self.handle, 1.0, np.vstack((rho, grad_rho)), nvar=4, deriv=1)
        return out[0], out[1], out[2]

    def __del__(self):
        handle = getattr(self, 'handle', None)
        if handle is not None:
            LIBXC.end_functional(handle)


@lru_cache(maxsize=16)
def get_restricted_functional(xc):
    return RestrictedLibXCFunctional(xc)


def has_libxc_backend():
    try:
        _find_libxc_backend()
        return True
    except OSError:
        return False
