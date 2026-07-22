#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native Hessian utilities for AO-based RKS.
"""

from copy import deepcopy

import numpy as np

from pyqed.qchem.mol import Molecule
from pyqed.units import amu_to_au, au2wavenumber

from .grid import AOGrid
from .xc import needs_gradients


def _copy_molecule(mol):
    return Molecule(
        atom=deepcopy(mol.atom),
        charge=mol.charge,
        spin=mol.spin,
        basis=mol.basis,
        unit='bohr',
    )


def _build_grid(mf, mol):
    grid = getattr(mf, 'grid', None)
    if grid is None:
        return AOGrid.atom_centered(mol, with_grad=needs_gradients(mf.xc))

    if getattr(grid, 'kind', None) != 'atom_centered':
        raise NotImplementedError(
            "Native Hessian calculations currently support only atom-centered grids."
        )

    settings = dict(getattr(grid, 'settings', {}))
    settings.setdefault('with_grad', needs_gradients(mf.xc))
    return AOGrid.atom_centered(mol, **settings)


def _n_zero_modes(coords, masses_au, zero_tol=1e-7):
    centered = np.asarray(coords, dtype=float) - np.einsum(
        'i,ij->j',
        masses_au,
        coords,
    ) / masses_au.sum()
    inertia = np.zeros((3, 3), dtype=float)
    for mass, r in zip(masses_au, centered):
        rr = np.dot(r, r)
        inertia += mass * (rr * np.eye(3) - np.outer(r, r))
    eigvals = np.linalg.eigvalsh(inertia)
    if eigvals[-1] < zero_tol:
        return 6
    linear = eigvals[0] < zero_tol * max(1.0, eigvals[-1])
    return 5 if linear else 6


def analyze_cartesian_hessian(
    hess,
    coords,
    masses_amu,
    remove_translation_rotation=True,
    negative_imaginary=True,
    zero_tol=1e-7,
):
    """
    Analyze a Cartesian Hessian into vibrational frequencies and normal modes.
    """
    masses_au = np.asarray(masses_amu, dtype=float) * amu_to_au
    coords = np.asarray(coords, dtype=float)
    natm = coords.shape[0]

    factors = np.repeat(masses_au ** -0.5, 3)
    mass_hess = factors[:, None] * np.asarray(hess, dtype=float) * factors[None, :]

    force_const_au, mode = np.linalg.eigh(mass_hess)
    freq_au = np.lib.scimath.sqrt(force_const_au)
    if negative_imaginary and np.iscomplexobj(freq_au):
        freq_au = freq_au.real - np.abs(freq_au.imag)

    norm_mode = np.einsum(
        'z,zri->izr',
        masses_au ** -0.5,
        mode.reshape(natm, 3, -1),
    )
    reduced_mass = 1.0 / np.einsum('izr,izr->i', norm_mode, norm_mode)

    if remove_translation_rotation:
        n_remove = _n_zero_modes(coords, masses_au, zero_tol=zero_tol)
        freq_au = freq_au[n_remove:]
        norm_mode = norm_mode[n_remove:]
        reduced_mass = reduced_mass[n_remove:]
        force_const_au = force_const_au[n_remove:]

    return {
        'freq_au': np.asarray(freq_au),
        'freq_cm1': np.asarray(freq_au) * au2wavenumber,
        'modes': np.asarray(norm_mode),
        'reduced_mass_au': np.asarray(reduced_mass),
        'reduced_mass_amu': np.asarray(reduced_mass) / amu_to_au,
        'force_constants_au': np.asarray(force_const_au),
    }


class Hessian:
    """
    Cartesian Hessian for native ``pyqed.qchem.dft.RKS`` objects.

    Notes
    -----
    The Hessian is currently evaluated by finite differences of analytic
    nuclear gradients.
    """

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.hess = None
        self.coords = np.asarray(self.mol.atom_coords(), dtype=float)
        self._mf_last = None

    def _evaluate_gradient(self, coords):
        mol = _copy_molecule(self.mol)
        mol.set_geom(coords)
        mol.build(driver='gbasis')

        grid = _build_grid(self.base, mol)
        step_mf = self.base.__class__(mol, grid=grid, xc=self.base.xc, init_guess=self.base.init_guess)
        step_mf.max_cycle = self.base.max_cycle
        step_mf.conv_tol = self.base.conv_tol
        step_mf.damping = self.base.damping
        step_mf.verbose = self.base.verbose
        step_mf.run()

        self._mf_last = step_mf
        grad = step_mf.nuc_grad_method().run()
        return np.asarray(grad, dtype=float)

    def run(self, step=1e-3, symmetrize=True):
        coords0 = np.asarray(self.coords, dtype=float)
        natm = coords0.shape[0]
        ndof = 3 * natm
        hess = np.zeros((ndof, ndof), dtype=float)

        for i in range(ndof):
            disp = np.zeros(ndof, dtype=float)
            disp[i] = step

            g_plus = self._evaluate_gradient((coords0.reshape(-1) + disp).reshape(natm, 3)).reshape(-1)
            g_minus = self._evaluate_gradient((coords0.reshape(-1) - disp).reshape(natm, 3)).reshape(-1)
            hess[:, i] = (g_plus - g_minus) / (2.0 * step)

        if symmetrize:
            hess = 0.5 * (hess + hess.T)

        self.hess = hess
        return self.hess

    def kernel(self, step=1e-3, symmetrize=True):
        """
        Backward-compatible alias for ``run()``.
        """
        return self.run(step=step, symmetrize=symmetrize)

    def vibrational_analysis(
        self,
        remove_translation_rotation=True,
        negative_imaginary=True,
        zero_tol=1e-7,
    ):
        if self.hess is None:
            raise ValueError("Run the Hessian calculation before requesting vibrational analysis.")
        return analyze_cartesian_hessian(
            self.hess,
            self.coords,
            self.mol.atom_mass_list(),
            remove_translation_rotation=remove_translation_rotation,
            negative_imaginary=negative_imaginary,
            zero_tol=zero_tol,
        )

    def frequencies(self, unit='cm^-1', **kwargs):
        data = self.vibrational_analysis(**kwargs)
        unit = unit.lower()
        if unit in ('cm^-1', 'cm-1', 'wavenumber', 'wavenumbers'):
            return data['freq_cm1']
        if unit in ('au', 'a.u.', 'hartree'):
            return data['freq_au']
        raise ValueError("unit must be 'cm^-1' or 'au'.")
