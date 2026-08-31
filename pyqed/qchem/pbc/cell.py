#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pyqed import au2angstrom
from pyqed.qchem.mol import Molecule, build_atom_from_coords
from .pseudo import load_gth_pseudos


def _normalize_unit(unit):
    unit_s = str(unit).strip().lower()
    if unit_s in ("b", "bohr", "au"):
        return "bohr"
    if unit_s in ("a", "angstrom", "ang"):
        return "angstrom"
    raise ValueError("unit must be 'bohr'/'b' or 'angstrom'/'a'.")


def _normalize_lattice(a, dimension, vacuum):
    dimension = int(dimension)
    if dimension not in (1, 3):
        raise NotImplementedError("Native periodic Cell currently supports dimension=1 or dimension=3.")

    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        if dimension != 1:
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
        value = int(nk)
        if int(dimension) == 1:
            return [value, 1, 1]
        return [value, value, value]

    mesh = [int(x) for x in nk]
    if len(mesh) == 1:
        return [mesh[0], 1, 1]
    if len(mesh) == 2:
        return [mesh[0], mesh[1], 1]
    if len(mesh) == 3:
        return mesh
    raise ValueError("nk must be an int or a length-1/2/3 iterable.")


def _dense_integral_options(options):
    dense_options = {} if options is None else dict(options)
    dense_options.setdefault("coord_type", "cartesian")
    dense_options["eri_representation"] = "dense"
    dense_options["aosym"] = "s1"
    return dense_options


def _cell_integral_options(options):
    cell_options = {} if options is None else dict(options)
    cell_options.setdefault("coord_type", "cartesian")
    cell_options.setdefault("eri_representation", "dense")
    if str(cell_options["eri_representation"]).lower() != "direct":
        cell_options.setdefault("aosym", "s1")
    return cell_options


def materialize_dense_eri(mol):
    """Return a dense AO ERI tensor from any builtin dense-like storage."""
    eri = getattr(mol, "eri", None)
    if eri is not None:
        return np.asarray(eri, dtype=float)

    eri_s4 = getattr(mol, "eri_s4", None)
    if eri_s4 is not None:
        from pyqed.qchem.basis import unpack_eri_s4

        eri = unpack_eri_s4(eri_s4, mol.nao)
        mol.eri = eri
        return np.asarray(eri, dtype=float)

    eri_s8 = getattr(mol, "eri_s8", None)
    if eri_s8 is not None:
        from pyqed.qchem.basis import unpack_eri_s8

        eri = unpack_eri_s8(eri_s8, mol.nao)
        mol.eri = eri
        return np.asarray(eri, dtype=float)

    raise ValueError("Dense PBC paths require mol.eri, mol.eri_s4, or mol.eri_s8.")


class Cell:
    """
    Native periodic Gaussian cell for small all-electron or GTH PBC calculations.

    This implementation is intentionally reference-level and correctness-first:
    it supports 1D chains and 3D cells with dense integrals for small systems.
    """

    def __init__(
        self,
        atom,
        a,
        basis,
        unit="bohr",
        charge=0,
        spin=0,
        dimension=3,
        vacuum=20.0,
        low_dim_ft_type="inf_vacuum",
        integral_options=None,
        pseudo=None,
        ecp=None,
    ):
        if pseudo is not None and ecp is not None:
            raise ValueError("Specify either pseudo or ecp, not both.")
        self.atom = atom
        self.a = a
        self.basis = basis
        self.unit = _normalize_unit(unit)
        self.charge = int(charge)
        self.spin = int(spin)
        self.dimension = int(dimension)
        self.vacuum = float(vacuum)
        self.low_dim_ft_type = low_dim_ft_type
        self.integral_options = {} if integral_options is None else dict(integral_options)
        self.pseudo = pseudo if pseudo is not None else ecp

        self._built = False
        self._unit_mol = None
        self._atom_symbols = None
        self._atom_coords = None
        self._pseudos = {}
        self._pseudos_by_atom = None
        self._ionic_charges = None

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

    @property
    def has_pseudo(self):
        return bool(self._pseudos)

    @property
    def ionic_charges(self):
        if self._ionic_charges is None:
            raise RuntimeError("Cell has not been built yet.")
        return np.asarray(self._ionic_charges, dtype=float).copy()

    def build(self):
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
        build_kwargs = {"options": _cell_integral_options(self.integral_options)}
        mol.build(**build_kwargs)
        if any(
            getattr(mol, name, None) is not None
            for name in ("eri", "eri_s4", "eri_s8")
        ):
            materialize_dense_eri(mol)
        elif str(
            getattr(mol, "builtin_resolved_eri_representation", "")
        ).lower() != "direct":
            raise ValueError(
                "Periodic Cell requires a dense-like molecular ERI build or "
                "eri_representation='direct'."
            )

        self._unit_mol = mol
        self._atom_symbols = list(mol.atom_symbols())
        self._atom_coords = np.asarray(mol.atom_coords(), dtype=float)
        self._pseudos = load_gth_pseudos(self.pseudo, self._atom_symbols)
        self._pseudos_by_atom = tuple(
            self._pseudos.get(symbol)
            for symbol in self._atom_symbols
        )
        nuclear_charges = np.asarray(mol.atom_charges(), dtype=float)
        self._ionic_charges = np.asarray([
            nuclear_charge if pseudo is None else pseudo.ionic_charge
            for nuclear_charge, pseudo in zip(nuclear_charges, self._pseudos_by_atom)
        ], dtype=float)
        self.nao = int(mol.nao)
        electron_count = float(np.sum(self._ionic_charges)) - self.charge
        if abs(electron_count - round(electron_count)) > 1.0e-10:
            raise ValueError("The pseudopotential valence electron count must be integral.")
        self.nelectron = int(round(electron_count))
        self._built = True
        return self

    def make_kpts(self, nk, *, gamma_centered=False):
        """Return a uniform reciprocal-space mesh in Cartesian coordinates.

        ``gamma_centered=False`` gives the half-shifted Monkhorst-Pack mesh.
        A Gamma-centered mesh always contains the origin, including for even
        mesh dimensions, and is closed under reciprocal-lattice wrapping.
        """
        if not self._built:
            self.build()
        mesh = _normalize_kmesh(nk, self.dimension)
        recip = 2.0 * np.pi * np.linalg.inv(np.asarray(self.lattice_vectors, dtype=float)).T
        axes = []
        for n in mesh:
            n = int(n)
            if n <= 0:
                raise ValueError("k-point mesh entries must be positive.")
            if gamma_centered:
                axis = np.arange(n, dtype=float) / n
                axis[axis >= 0.5] -= 1.0
            else:
                axis = (np.arange(n, dtype=float) + 0.5) / n - 0.5
            axes.append(axis)

        out = []
        for i in range(mesh[0]):
            for j in range(mesh[1]):
                for k in range(mesh[2]):
                    frac = axes[0][i] * recip[0] + axes[1][j] * recip[1] + axes[2][k] * recip[2]
                    out.append(frac)
        return np.asarray(out, dtype=float)

    def translation_vector(self, n):
        if np.isscalar(n):
            key = (int(n),)
        else:
            key = tuple(int(x) for x in n)
        if self.dimension == 1:
            if len(key) != 1:
                raise ValueError("1D translation keys must have length 1.")
            return float(key[0]) * np.asarray(self.lattice_vectors[0], dtype=float)
        if len(key) != 3:
            raise ValueError("3D translation keys must have length 3.")
        lattice = np.asarray(self.lattice_vectors, dtype=float)
        return key[0] * lattice[0] + key[1] * lattice[1] + key[2] * lattice[2]

    def image_keys(self, nimages):
        nimages = int(nimages)
        if nimages < 0:
            raise ValueError("nimages must be non-negative.")
        rng = range(-nimages, nimages + 1)
        if self.dimension == 1:
            return [(n,) for n in rng]
        return [(i, j, k) for i in rng for j in rng for k in rng]

    def build_image_molecule(self, nimages):
        if not self._built:
            self.build()

        atom_symbols = []
        atom_coords = []
        for key in self.image_keys(nimages):
            shift = self.translation_vector(key)
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
        build_kwargs = {"options": _dense_integral_options(self.integral_options)}
        mol.build(**build_kwargs)
        materialize_dense_eri(mol)
        return mol

    def nuclear_repulsion(self, nimages):
        if not self._built:
            self.build()

        charges = self.ionic_charges
        coords = np.asarray(self._atom_coords, dtype=float)
        e = 0.0
        for ia, (za, ra) in enumerate(zip(charges, coords)):
            for key in self.image_keys(nimages):
                shift = self.translation_vector(key)
                for ib, (zb, rb) in enumerate(zip(charges, coords)):
                    if all(x == 0 for x in key) and ia == ib:
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

        charges = self.ionic_charges
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

        if self.has_pseudo:
            raise NotImplementedError(
                "Use EwaldRHF/KRHF to build k-dependent pseudopotential matrices."
            )
        charges = self.ionic_charges
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

        if self.has_pseudo:
            raise NotImplementedError(
                "Use EwaldRHF/KRHF to build k-dependent pseudopotential matrices."
            )
        charges = self.ionic_charges
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

    def RHF(self, kpts=None, nk=None, method=None, **kwargs):
        if method is None:
            method = "ewald" if self.pseudo is not None else "finite_image"
        method = str(method).lower()
        if method in ("finite_image", "finite-image", "image", "native"):
            if self.pseudo is not None:
                raise NotImplementedError(
                    "Pseudopotentials require method='ewald' with GDF/reciprocal J/K."
                )
            from .hf import RHF

            return RHF(self, kpts=kpts, nk=nk, **kwargs)
        if method in ("ewald", "aft", "krhf"):
            from .hf import EwaldRHF

            return EwaldRHF(self, kpts=kpts, nk=nk, **kwargs)
        raise ValueError("method must be 'finite_image' or 'ewald'.")

    def KRHF(self, kpts=None, nk=None, **kwargs):
        from .hf import KRHF

        return KRHF(self, kpts=kpts, nk=nk, **kwargs)
