#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local exchange-correlation functionals.
"""

import numpy as np

from pyqed.qchem._libxc import get_restricted_functional, has_libxc_backend


RHO_CUTOFF = 1e-15
VWN_A = 0.0310907
VWN_B = 3.72744
VWN_C = 12.9352
VWN_X0 = -0.10498
_LIBXC_NAMES = {'b3lyp', 'hyb_gga_xc_b3lyp'}


def _clip_density(rho):
    return np.clip(np.asarray(rho, dtype=float), RHO_CUTOFF, None)


def xc_type(xc):
    """
    XC family used by the requested functional.
    """
    name = xc.lower()

    if name in ('lda_x', 'lda', 'lda_c_vwn', 'vwn', 'svwn', 'lda_vwn',
                'lda_xc_vwn', 'lda_x,vwn', 'lda,vwn', 'slater,vwn',
                'slater,vwn5'):
        return 'LDA'

    if name in _LIBXC_NAMES:
        return get_restricted_functional(name).xctype

    raise NotImplementedError(f"XC functional '{xc}' is not implemented.")


def needs_gradients(xc):
    """
    Whether the functional requires density gradients on the numerical grid.
    """
    return xc_type(xc) == 'GGA'


def hybrid_coeff(xc):
    """
    Fraction of exact exchange used by the functional.
    """
    name = xc.lower()
    if name in _LIBXC_NAMES:
        return get_restricted_functional(name).hyb
    return 0.0


def lda_x_energy_density(rho):
    """
    Dirac exchange energy per particle for spin-restricted LDA.
    """
    rho = _clip_density(rho)
    prefactor = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    return prefactor * rho ** (1.0 / 3.0)


def lda_x_potential(rho):
    """
    Dirac exchange potential for spin-restricted LDA.
    """
    rho = _clip_density(rho)
    prefactor = -(3.0 / np.pi) ** (1.0 / 3.0)
    return prefactor * rho ** (1.0 / 3.0)


def lda_x_fxc(rho):
    """
    Derivative of the Dirac exchange potential with respect to density.
    """
    rho = _clip_density(rho)
    prefactor = -(1.0 / 3.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    return prefactor * rho ** (-2.0 / 3.0)


def _vwn_eps_from_x(x):
    """
    Vosko-Wilk-Nusair correlation energy per particle for the unpolarized gas.
    """
    q = np.sqrt(4.0 * VWN_C - VWN_B * VWN_B)
    x = np.asarray(x, dtype=float)

    xx = x * x
    capital_x = xx + VWN_B * x + VWN_C
    x0sq = VWN_X0 * VWN_X0
    capital_x0 = x0sq + VWN_B * VWN_X0 + VWN_C

    atan_term = np.arctan(q / (2.0 * x + VWN_B))
    term1 = np.log(xx / capital_x)
    term2 = 2.0 * VWN_B / q * atan_term
    term3 = (VWN_B * VWN_X0 / capital_x0) * (
        np.log((x - VWN_X0) ** 2 / capital_x)
        + 2.0 * (2.0 * VWN_X0 + VWN_B) / q * atan_term
    )
    return VWN_A * (term1 + term2 - term3)


def lda_c_vwn_energy_density(rho):
    """
    VWN correlation energy per particle for spin-restricted LDA.
    """
    rho = _clip_density(rho)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    x = np.sqrt(rs)
    return _vwn_eps_from_x(x)


def lda_c_vwn_potential(rho):
    """
    VWN correlation potential for spin-restricted LDA.

    The potential is evaluated from v_c = d(rho * eps_c)/d rho using a
    central finite difference with respect to x = sqrt(r_s).
    """
    rho = _clip_density(rho)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    x = np.sqrt(rs)

    eps_c = _vwn_eps_from_x(x)
    dx = 1e-6 * np.maximum(1.0, np.abs(x))
    x_minus = np.maximum(x - dx, 1e-12)
    x_plus = x + dx
    deps_dx = (_vwn_eps_from_x(x_plus) - _vwn_eps_from_x(x_minus)) / (x_plus - x_minus)
    return eps_c - x * deps_dx / 6.0


def lda_c_vwn_fxc(rho):
    """
    Derivative of the VWN correlation potential with respect to density.
    """
    rho = _clip_density(rho)
    drho = 1e-6 * np.maximum(1.0, rho)
    rho_minus = np.maximum(rho - drho, RHO_CUTOFF)
    rho_plus = rho + drho
    return (
        lda_c_vwn_potential(rho_plus) - lda_c_vwn_potential(rho_minus)
    ) / (rho_plus - rho_minus)


def eval_fxc(rho, xc='lda_x'):
    """
    Evaluate the restricted second derivative d(v_xc)/d(rho) on a grid.
    """
    name = xc.lower()

    if name in ('lda_x', 'lda'):
        return lda_x_fxc(rho)

    if name in ('lda_c_vwn', 'vwn'):
        return lda_c_vwn_fxc(rho)

    if name in ('svwn', 'lda_vwn', 'lda_xc_vwn', 'lda_x,vwn', 'lda,vwn',
                'slater,vwn', 'slater,vwn5'):
        return lda_x_fxc(rho) + lda_c_vwn_fxc(rho)

    raise NotImplementedError(
        f"Second-order XC kernel for '{xc}' is not implemented."
    )


def eval_xc(rho, xc='lda_x', grad_rho=None):
    """
    Evaluate exchange-correlation quantities on a numerical grid.
    """
    name = xc.lower()

    if name in ('lda_x', 'lda'):
        eps_xc = lda_x_energy_density(rho)
        v_xc = lda_x_potential(rho)
        return eps_xc, v_xc

    if name in ('lda_c_vwn', 'vwn'):
        eps_xc = lda_c_vwn_energy_density(rho)
        v_xc = lda_c_vwn_potential(rho)
        return eps_xc, v_xc

    if name in ('svwn', 'lda_vwn', 'lda_xc_vwn', 'lda_x,vwn', 'lda,vwn',
                'slater,vwn', 'slater,vwn5'):
        eps_x = lda_x_energy_density(rho)
        v_x = lda_x_potential(rho)
        eps_c = lda_c_vwn_energy_density(rho)
        v_c = lda_c_vwn_potential(rho)
        return eps_x + eps_c, v_x + v_c

    if name in _LIBXC_NAMES:
        exc, vrho, vsigma = get_restricted_functional(name).eval(rho, grad_rho=grad_rho)
        return exc, (vrho, vsigma)

    raise NotImplementedError(
        "Supported functionals are 'lda'/'lda_x', 'lda_c_vwn'/'vwn', "
        "'svwn'/'lda,vwn', and 'b3lyp'."
    )
