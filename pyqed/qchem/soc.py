#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spin-orbit coupling helpers for scalar-reference wavefunctions.

The functions in this module are designed for the common "RHF + perturbative
SOC" workflow:

1. Run a scalar RHF calculation to obtain orthonormal MOs.
2. Build one-electron Breit-Pauli SOC integrals in the AO or MO basis.
3. Expand the spatial operator to a spin-orbital matrix for use in
   post-HF/state-interaction models.

Notes
-----
This module currently provides only the one-electron nuclear spin-orbit
operator.  By default it uses the one-center approximation, which is often the
cheapest useful first model.
"""

import numpy as np
from opt_einsum import contract

from pyqed.qchem._libcint import CBasis1e

LIGHT_SPEED = 137.03599967994


def soc_1e_prefactor(light_speed=None):
    """
    Breit-Pauli one-electron SOC prefactor in atomic units.
    """
    if light_speed is None:
        light_speed = LIGHT_SPEED
    return 1.0 / (4.0 * light_speed ** 2)


def _get_cbasis(mol):
    """
    Build a libcint-compatible basis wrapper from a pyqed Molecule.
    """
    if getattr(mol, '_bas', None) is None:
        raise ValueError(
            "mol._bas is not available. Build the molecule with driver='gbasis' "
            "before requesting SOC integrals without PySCF."
        )

    coord_type = getattr(mol._bas[0], 'coord_type', 'spherical')
    return CBasis1e(mol._bas, mol.atom_symbols(), mol.atom_coords(), coord_type=coord_type)


def get_pvxp_ao(mol, one_center=True):
    """
    Raw three-component ``p V x p`` operator in the AO basis.

    Parameters
    ----------
    mol : pyqed.qchem.Molecule-like
        Must provide ``topyscf()``.
    one_center : bool
        If ``True``, build the standard one-center approximation by keeping
        only atom-local shell blocks for each nucleus.  If ``False``, build the
        full one-electron nuclear SOC operator by summing over all nuclei.

    Returns
    -------
    ndarray
        Array of shape ``(3, nao, nao)`` with components ``(x, y, z)``.
    """
    cbasis = _get_cbasis(mol)
    mat = np.zeros((3, cbasis.nbfn, cbasis.nbfn), dtype=float)
    atom_coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_charges = np.asarray(mol.atom_charges(), dtype=float)

    if one_center:
        for ia, coord in enumerate(atom_coords):
            p0, p1 = cbasis.ao_slice_by_atom(ia)
            # ``int1e_prinvxp`` is not shell-Hermitian, so we need the full
            # shell-pair evaluation instead of mirroring the lower triangle.
            w = -atom_charges[ia] * cbasis.int1e(
                'int1e_prinvxp',
                components=(3,),
                inv_origin=coord,
                hermi=False,
            )
            mat[:, p0:p1, p0:p1] = np.moveaxis(w[p0:p1, p0:p1], -1, 0)
    else:
        for ia, coord in enumerate(atom_coords):
            w = -atom_charges[ia] * cbasis.int1e(
                'int1e_prinvxp',
                components=(3,),
                inv_origin=coord,
                hermi=False,
            )
            mat += np.moveaxis(w, -1, 0)

    return mat


def get_soc_1e_ao(mol, one_center=True, with_prefactor=True, light_speed=None):
    """
    One-electron Breit-Pauli SOC vector operator in the AO basis.
    """
    mat = get_pvxp_ao(mol, one_center=one_center)
    if with_prefactor:
        mat = soc_1e_prefactor(light_speed=light_speed) * mat
    return mat


def get_soc_1e_mo(mf, mo_coeff=None, one_center=True, with_prefactor=True,
                  light_speed=None):
    """
    One-electron Breit-Pauli SOC vector operator in the MO basis.

    Parameters
    ----------
    mf : RHF-like object
        Must provide ``mol`` and converged ``mo_coeff``.
    mo_coeff : ndarray, optional
        MO coefficients.  If ``None``, use ``mf.mo_coeff``.
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    if mo_coeff is None:
        raise ValueError("MO coefficients are required. Run RHF first.")

    hso_ao = get_soc_1e_ao(
        mf.mol,
        one_center=one_center,
        with_prefactor=with_prefactor,
        light_speed=light_speed,
    )
    return contract('xpq,pi,qj->xij', hso_ao, mo_coeff.conj(), mo_coeff)


def reorder_spin_orbital_matrix(mat, source='interleaved', target='grouped'):
    """
    Reorder a spin-orbital matrix between interleaved and grouped layouts.

    Parameters
    ----------
    mat : ndarray
        Square matrix with shape ``(2*n, 2*n)``.
    source : {'interleaved', 'grouped'}
        Current spin-orbital ordering of ``mat``.
    target : {'interleaved', 'grouped'}
        Requested spin-orbital ordering.
    """
    source = source.lower()
    target = target.lower()
    if source == target:
        return mat
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] % 2 != 0:
        raise ValueError("mat must be a square (2*n, 2*n) spin-orbital matrix.")
    if source not in {'interleaved', 'grouped'} or target not in {'interleaved', 'grouped'}:
        raise ValueError("source and target must be 'interleaved' or 'grouped'.")

    norb = mat.shape[0] // 2
    interleaved_from_grouped = np.empty(2 * norb, dtype=int)
    interleaved_from_grouped[0::2] = np.arange(norb)
    interleaved_from_grouped[1::2] = norb + np.arange(norb)

    if source == 'grouped' and target == 'interleaved':
        perm = interleaved_from_grouped
    else:
        perm = np.argsort(interleaved_from_grouped)
    return mat[np.ix_(perm, perm)]


def spatial_soc_to_spin_orbital(hso_xyz, order='interleaved'):
    """
    Expand a 3-component spatial SOC operator to a 2-spinor matrix.

    Parameters
    ----------
    hso_xyz : ndarray
        Array of shape ``(3, n, n)``.
    order : {'interleaved', 'grouped'}
        Spin-orbital ordering of the returned matrix.

    Returns
    -------
    ndarray
        Complex Hermitian matrix of shape ``(2*n, 2*n)`` in the spin-orbital
        basis.
    """
    pauli = 1j * np.asarray([
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ], dtype=complex)
    n = hso_xyz.shape[-1]
    mat = np.einsum('sxy,spq->xpyq', pauli, hso_xyz).reshape(2 * n, 2 * n)
    return reorder_spin_orbital_matrix(mat, source='interleaved', target=order)


def get_soc_1e_spin_orbital(mf, representation='mo', mo_coeff=None,
                            one_center=True, with_prefactor=True,
                            light_speed=None, order='interleaved'):
    """
    One-electron SOC Hamiltonian in a spin-orbital basis.

    Parameters
    ----------
    mf : RHF-like object
        Mean-field reference.
    representation : {'ao', 'mo'}
        Spatial basis in which the spin-orbital Hamiltonian is built.
    """
    rep = representation.lower()
    if rep == 'ao':
        hso_xyz = get_soc_1e_ao(
            mf.mol,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    elif rep == 'mo':
        hso_xyz = get_soc_1e_mo(
            mf,
            mo_coeff=mo_coeff,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    else:
        raise ValueError("representation must be 'ao' or 'mo'.")

    return spatial_soc_to_spin_orbital(hso_xyz, order=order)
