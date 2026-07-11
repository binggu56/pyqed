#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pyqed import au2angstrom
from pyqed.qchem.mol import Molecule, build_atom_from_coords


def _normalize_unit(unit):
    unit_s = str(unit).strip().lower()
    if unit_s in ("b", "bohr", "au"):
        return "bohr"
    if unit_s in ("a", "angstrom", "ang"):
        return "angstrom"
    raise ValueError("unit must be 'bohr'/'b' or 'angstrom'/'a'.")


def _normalize_lattice(a, dimension, vacuum):
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        if int(dimension) != 1:
            raise ValueError("Scalar lattice constant is only supported for dimension=1.")
        vac = float(vacuum)
        return np.asarray([
            [float(arr), 0.0, 0.0],
            [0.0, vac, 0.0],
            [0.0, 0.0, vac],
        ], dtype=float)
    if arr.ndim == 1:
        if arr.size == 1:
            return _normalize_lattice(float(arr[0]), dimension, vacuum)
        if arr.size == 3:
            return np.diag(arr.astype(float))
        raise ValueError("1D lattice vector input must have length 1 or 3.")
    if arr.shape == (3, 3):
        return arr.astype(float)
    raise ValueError("a must be a scalar, length-3 vector, or 3x3 lattice matrix.")


def _normalize_kmesh(nk, dimension):
    if np.isscalar(nk):
        if int(dimension) != 1:
            raise ValueError("Scalar nk is only supported for dimension=1.")
        return [int(nk), 1, 1]

    mesh = [int(x) for x in nk]
    if len(mesh) == 1:
        return [mesh[0], 1, 1]
    if len(mesh) == 2:
        return [mesh[0], mesh[1], 1]
    if len(mesh) == 3:
        return mesh
    raise ValueError("nk must be an int or a length-1/2/3 iterable.")


class Cell:
    """
    Minimal native 1D periodic cell for the first pyqed PBC milestone.

    This implementation is intentionally reference-level: it supports only the
    1D path and uses image-summed molecular integrals from the builtin engine.
    """

    def __init__(
        self,
        atom,
        a,
        basis,
        unit="bohr",
        charge=0,
        spin=0,
        dimension=1,
        vacuum=20.0,
        low_dim_ft_type="inf_vacuum",
        integral_driver="builtin",
        integral_options=None,
    ):
        self.atom = atom
        self.a = a
        self.basis = basis
        self.unit = _normalize_unit(unit)
        self.charge = int(charge)
        self.spin = int(spin)
        self.dimension = int(dimension)
        self.vacuum = float(vacuum)
        self.low_dim_ft_type = low_dim_ft_type
        self.integral_driver = integral_driver
        self.integral_options = {} if integral_options is None else dict(integral_options)

        self._built = False
        self._unit_mol = None
        self._atom_symbols = None
        self._atom_coords = None

        self.nao = None
        self.nelectron = None
        self.lattice_vectors = None

    @property
    def built(self):
        return bool(self._built)

    @property
    def unit_molecule(self):
        if self._unit_mol is None:
            raise RuntimeError("Cell has not been built yet.")
        return self._unit_mol

    def build(self):
        if self.dimension != 1:
            raise NotImplementedError("Native periodic Cell currently supports only dimension=1.")

        lattice = _normalize_lattice(self.a, self.dimension, self.vacuum)
        if self.unit == "angstrom":
            lattice = lattice / au2angstrom
        self.lattice_vectors = np.asarray(lattice, dtype=float)

        mol = Molecule(
            atom=self.atom,
            basis=self.basis,
            unit=self.unit,
            charge=self.charge,
            spin=self.spin,
        )
        build_kwargs = {}
        if self.integral_options:
            build_kwargs["options"] = dict(self.integral_options)
        mol.build(driver=self.integral_driver, **build_kwargs)

        self._unit_mol = mol
        self._atom_symbols = list(mol.atom_symbols())
        self._atom_coords = np.asarray(mol.atom_coords(), dtype=float)
        self.nao = int(mol.nao)
        self.nelectron = int(mol.nelec)
        self._built = True
        return self

    def make_kpts(self, nk):
        mesh = _normalize_kmesh(nk, self.dimension)
        nk1 = int(mesh[0])
        frac = (np.arange(nk1, dtype=float) + 0.5) / nk1 - 0.5
        a1 = np.asarray(self.lattice_vectors[0], dtype=float)
        b1 = 2.0 * np.pi * a1 / np.dot(a1, a1)
        return frac[:, None] * b1[None, :]

    def translation_vector(self, n):
        return float(n) * np.asarray(self.lattice_vectors[0], dtype=float)

    def build_image_molecule(self, nimages):
        if not self._built:
            self.build()

        nimages = int(nimages)
        atom_symbols = []
        atom_coords = []
        for icell in range(-nimages, nimages + 1):
            shift = self.translation_vector(icell)
            for sym, coord in zip(self._atom_symbols, self._atom_coords):
                atom_symbols.append(sym)
                atom_coords.append(coord + shift)
        atom = build_atom_from_coords(atom_symbols, np.asarray(atom_coords, dtype=float))
        mol = Molecule(
            atom=atom,
            basis=self.basis,
            unit="bohr",
            charge=0,
            spin=0,
        )
        build_kwargs = {}
        if self.integral_options:
            build_kwargs["options"] = dict(self.integral_options)
        mol.build(driver=self.integral_driver, **build_kwargs)
        return mol

    def nuclear_repulsion(self, nimages):
        if not self._built:
            self.build()

        charges = np.asarray(self._unit_mol.atom_charges(), dtype=float)
        coords = np.asarray(self._atom_coords, dtype=float)
        e = 0.0
        for ia, (za, ra) in enumerate(zip(charges, coords)):
            for icell in range(-int(nimages), int(nimages) + 1):
                shift = self.translation_vector(icell)
                for ib, (zb, rb) in enumerate(zip(charges, coords)):
                    if icell == 0 and ia == ib:
                        continue
                    diff = ra - (rb + shift)
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        e += za * zb / dist
        return 0.5 * e

    def ewald_nuclear_repulsion(
        self,
        eta=None,
        real_cut=4,
        recip_cut=8,
        neutralizing_background=True,
    ):
        if not self._built:
            self.build()

        from .ewald import ewald_nuclear_repulsion

        charges = np.asarray(self._unit_mol.atom_charges(), dtype=float)
        coords = np.asarray(self._atom_coords, dtype=float)
        return ewald_nuclear_repulsion(
            charges,
            coords,
            self.lattice_vectors,
            eta=eta,
            real_cut=real_cut,
            recip_cut=recip_cut,
            neutralizing_background=neutralizing_background,
        )

    def reciprocal_nuclear_attraction_matrix(self, recip_cut=8, eta=None):
        if not self._built:
            self.build()

        from .ewald import reciprocal_nuclear_attraction_matrix_s

        charges = np.asarray(self._unit_mol.atom_charges(), dtype=float)
        coords = np.asarray(self._atom_coords, dtype=float)
        return reciprocal_nuclear_attraction_matrix_s(
            charges,
            coords,
            self._unit_mol._bas,
            self.lattice_vectors,
            recip_cut=recip_cut,
            eta=eta,
        )

    def short_range_nuclear_attraction_matrix(self, eta, real_cut=4):
        if not self._built:
            self.build()

        from .ewald import short_range_nuclear_attraction_matrix_s

        charges = np.asarray(self._unit_mol.atom_charges(), dtype=float)
        coords = np.asarray(self._atom_coords, dtype=float)
        return short_range_nuclear_attraction_matrix_s(
            charges,
            coords,
            self._unit_mol._bas,
            self.lattice_vectors,
            eta=eta,
            real_cut=real_cut,
        )

    def reciprocal_hartree_matrix(self, dm, recip_cut=8, eta=None):
        if not self._built:
            self.build()

        from .ewald import reciprocal_hartree_matrix_s

        return reciprocal_hartree_matrix_s(
            dm,
            self._unit_mol._bas,
            self.lattice_vectors,
            recip_cut=recip_cut,
            eta=eta,
        )

    def short_range_eri_tensor(self, eta):
        if not self._built:
            self.build()

        from .ewald import short_range_eri_tensor_s

        return short_range_eri_tensor_s(self._unit_mol._bas, eta=eta)

    def reciprocal_eri_tensor(self, recip_cut=8, eta=None):
        if not self._built:
            self.build()

        from .ewald import reciprocal_eri_tensor_s

        return reciprocal_eri_tensor_s(
            self._unit_mol._bas,
            self.lattice_vectors,
            recip_cut=recip_cut,
            eta=eta,
        )

    def RHF(self, kpts=None, nk=None, method="finite_image", **kwargs):
        method = str(method).lower()
        if method in ("finite_image", "finite-image", "image", "native"):
            from .hf import RHF

            return RHF(self, kpts=kpts, nk=nk, **kwargs)
        if method in ("ewald", "aft"):
            if kpts is not None or nk is not None:
                raise NotImplementedError("method='ewald' currently supports gamma point only.")
            from .hf import EwaldRHF

            return EwaldRHF(self, **kwargs)
        raise ValueError("method must be 'finite_image' or 'ewald'.")
