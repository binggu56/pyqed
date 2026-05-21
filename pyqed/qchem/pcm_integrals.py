"""
Native PCM Coulomb integrals.

The PCM surface charges are normalized spherical Gaussians, so the only
three-center integral needed by the solvent code is ``(ij|L)`` where ``L`` is
an s-type unit-charge Gaussian centered on a cavity grid point.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import scipy.special

from pyqed.qchem.basis import (
    ContractedGaussian,
    _basis_path,
    _basis_signature,
    _cart2sph_unit_block,
    _cart_shell_blocks,
    _pack_signatures_for_numba,
    make_contractions,
    parse_gbs,
)
from pyqed.qchem.rys import (
    ERI_PREFAC,
    _D_SHELL_TO_AXES,
    _evaluate_promoted_block_dispatch,
    _shell_derivative_terms,
    boys,
)

try:
    from pyqed.qchem import _rys_cy
except Exception:  # pragma: no cover - optional accelerator
    _rys_cy = None


def _native_basis_and_transform(mol):
    basis = getattr(mol, "_bas_cart", None)
    transform = getattr(mol, "_ao_cart2sph", None)
    if basis is not None:
        return list(basis), transform

    basis = getattr(mol, "_bas", None)
    if basis is not None and all(isinstance(fn, ContractedGaussian) for fn in basis):
        return list(basis), None

    basis_dict = parse_gbs(_basis_path(mol.basis))
    basis_cart = make_contractions(
        basis_dict,
        mol.atom_symbols(),
        np.asarray(mol.atom_coords(), dtype=float),
        coord_types="c",
    )

    if bool(getattr(mol, "cart", False)):
        return list(basis_cart), None

    blocks = _cart_shell_blocks(basis_cart)
    nsph = sum(2 * l + 1 for _, _, l in blocks)
    transform = np.zeros((len(basis_cart), nsph), dtype=float)
    col = 0
    for start, stop, l in blocks:
        block = _cart2sph_unit_block(l)
        ncols = block.shape[1]
        transform[start:stop, col:col + ncols] = block
        col += ncols
    return list(basis_cart), transform


def _charge_signature(center, exponent):
    exponent = float(exponent)
    weight = (exponent / math.pi) ** 1.5
    return (
        (0, 0, 0),
        tuple(float(x) for x in center),
        (exponent,),
        (weight,),
    )


def _primitive_sss_params(a, A, b, B, c, C):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    p = float(a) + float(b)
    q = float(c)
    alpha = p * q / (p + q)
    mu_ab = float(a) * float(b) / p
    lam_a = float(a) / p
    lam_b = float(b) / p
    P = (float(a) * A + float(b) * B) / p
    AB = A - B
    PQ = P - C
    pref = ERI_PREFAC * math.exp(-mu_ab * float(np.dot(AB, AB))) / (
        p * q * math.sqrt(p + q)
    )
    return {
        "A": A,
        "B": B,
        "C": C,
        "p": p,
        "q": q,
        "alpha": alpha,
        "mu_ab": mu_ab,
        "mu_cd": 0.0,
        "lam_a": lam_a,
        "lam_b": lam_b,
        "lam_c": 1.0,
        "lam_d": 0.0,
        "AB": AB,
        "CD": np.zeros(3),
        "PQ": PQ,
        "T": alpha * float(np.dot(PQ, PQ)),
        "pref": pref,
    }


def _primitive_three_center_rys(shell_a, exp_a, origin_a, shell_b, exp_b, origin_b, exp_c, origin_c):
    params = _primitive_sss_params(exp_a, origin_a, exp_b, origin_b, exp_c, origin_c)
    terms_a = _shell_derivative_terms(shell_a, float(exp_a))
    terms_b = _shell_derivative_terms(shell_b, float(exp_b))
    force_python = (
        tuple(int(x) for x in shell_a) in _D_SHELL_TO_AXES
        or tuple(int(x) for x in shell_b) in _D_SHELL_TO_AXES
    )

    total = 0.0
    for coeff_a, axes_a in terms_a:
        for coeff_b, axes_b in terms_b:
            coeff = coeff_a * coeff_b
            deriv_axes = axes_a + axes_b
            if not deriv_axes:
                total += coeff * params["pref"] * boys(0, params["T"])
                continue
            centers = ("A",) * len(axes_a) + ("B",) * len(axes_b)
            block = _evaluate_promoted_block_dispatch(
                centers,
                params,
                {"A": float(exp_a), "B": float(exp_b)},
                force_python=force_python,
            )
            total += coeff * float(block[deriv_axes])
    return total


@lru_cache(maxsize=524288)
def _contracted_three_center_surface_rys_cached(sig_a, sig_b, sig_c):
    shell_a, origin_a, exps_a, weights_a = sig_a
    shell_b, origin_b, exps_b, weights_b = sig_b
    _shell_c, origin_c, exps_c, weights_c = sig_c

    value = 0.0
    for ia, wa in enumerate(weights_a):
        for ib, wb in enumerate(weights_b):
            for ic, wc in enumerate(weights_c):
                value += (
                    wa
                    * wb
                    * wc
                    * _primitive_three_center_rys(
                        shell_a,
                        exps_a[ia],
                        origin_a,
                        shell_b,
                        exps_b[ib],
                        origin_b,
                        exps_c[ic],
                        origin_c,
                    )
                )
    return value


def contracted_three_center_surface_rys(sig_a, sig_b, sig_c):
    if sig_b < sig_a:
        sig_a, sig_b = sig_b, sig_a
    return _contracted_three_center_surface_rys_cached(sig_a, sig_b, sig_c)


def surface_charge_ao_coulomb(mol, coords, exponents):
    """Return ``(ij|L)`` for normalized Gaussian PCM surface charges."""
    coords = np.asarray(coords, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    basis_cart, transform = _native_basis_and_transform(mol)
    signatures = tuple(_basis_signature(fn) for fn in basis_cart)
    if _rys_cy is not None and hasattr(_rys_cy, "compute_surface_charge_ao_coulomb_rys"):
        shells, origins, basis_exps, weights, nprim = _pack_signatures_for_numba(signatures)
        tensor = _rys_cy.compute_surface_charge_ao_coulomb_rys(
            np.ascontiguousarray(shells, dtype=np.int64),
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(basis_exps, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(nprim, dtype=np.int64),
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(exponents, dtype=np.float64),
        )
    else:
        ncart = len(signatures)
        ngrids = len(coords)
        tensor = np.zeros((ncart, ncart, ngrids), dtype=float)

        for igrid, (coord, exponent) in enumerate(zip(coords, exponents)):
            sig_c = _charge_signature(coord, exponent)
            for i, sig_i in enumerate(signatures):
                for j, sig_j in enumerate(signatures[: i + 1]):
                    value = contracted_three_center_surface_rys(sig_i, sig_j, sig_c)
                    tensor[i, j, igrid] = value
                    if i != j:
                        tensor[j, i, igrid] = value

    if transform is not None:
        tensor = np.einsum("pi,pqL,qj->ijL", transform, tensor, transform, optimize=True)
    return tensor


def nuclear_potential_at_surface(atom_coords, atom_charges, grid_coords, exponents):
    """Potential of point nuclei on normalized Gaussian PCM surface charges."""
    atom_coords = np.asarray(atom_coords, dtype=float)
    atom_charges = np.asarray(atom_charges, dtype=float)
    grid_coords = np.asarray(grid_coords, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    diff = atom_coords[:, None, :] - grid_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    sqrt_exp = np.sqrt(exponents)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = scipy.special.erf(sqrt_exp * dist) / dist
    values = np.where(dist > 1.0e-14, values, 2.0 * sqrt_exp / math.sqrt(math.pi))
    return np.einsum("A,AL->L", atom_charges, values, optimize=True)
