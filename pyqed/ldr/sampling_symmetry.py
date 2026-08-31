"""Molecular-symmetry reduction of electronic-structure sampling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class SamplingSymmetryImage:
    """One requested sample expressed through a symmetry representative."""

    representative_coordinates: tuple[float, ...]
    operation: str = "identity"


class SamplingSymmetry:
    """Interface for reducing and transporting molecular-symmetry samples."""

    name = "identity"

    def resolve(self, coordinates):
        coordinates = tuple(float(value) for value in coordinates)
        return SamplingSymmetryImage(coordinates)

    def images(self, coordinates):
        """Return the complete symmetry orbit of one explicit sample."""

        return (tuple(float(value) for value in coordinates),)

    def pair_images(self, left, right):
        """Apply each symmetry operation jointly to an explicit sample pair."""

        return ((
            tuple(float(value) for value in left),
            tuple(float(value) for value in right),
        ),)

    def transform_record(
        self,
        record,
        image,
        *,
        representative_geometry,
        requested_geometry,
        protocol,
    ):
        return record

    def view_key(self, image):
        return {"symmetry": self.name, "operation": str(image.operation)}

    def metadata(self):
        return {"name": self.name}


class PhenolReflectionSymmetry(SamplingSymmetry):
    r"""Identify phenol coordinates related by reflection in the ring plane.

    By default only the CCOH torsion is odd.  Additional out-of-plane reduced
    coordinates, such as Wilson ``16a``, are supplied through ``odd_axes`` and
    are reflected by the same molecular operation.
    """

    name = "phenol-phi-reflection"
    operation = "sigma_xy"
    matrix = np.diag((1.0, 1.0, -1.0))

    def __init__(self, *, torsion_axis=1, odd_axes=None, tolerance=1.0e-12):
        self.torsion_axis = int(torsion_axis)
        self.odd_axes = tuple(
            dict.fromkeys(
                (self.torsion_axis,)
                if odd_axes is None
                else tuple(int(axis) for axis in odd_axes)
            )
        )
        self.tolerance = float(tolerance)
        if self.torsion_axis < 0:
            raise ValueError("torsion_axis must be non-negative")
        if not self.odd_axes or any(axis < 0 for axis in self.odd_axes):
            raise ValueError("odd_axes must contain non-negative coordinate axes")
        if self.torsion_axis not in self.odd_axes:
            raise ValueError("odd_axes must contain torsion_axis")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")

    def resolve(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 1 or max(self.odd_axes) >= coordinates.size:
            raise ValueError(
                "phenol reflection needs a one-dimensional coordinate containing "
                "the configured torsion axis"
            )
        representative = np.array(coordinates, copy=True)
        orientation = 0.0
        for axis in self.odd_axes:
            if abs(representative[axis]) > self.tolerance:
                orientation = float(representative[axis])
                break
        if orientation < 0.0:
            representative[list(self.odd_axes)] *= -1.0
            operation = self.operation
        else:
            operation = "identity"
        for axis in self.odd_axes:
            if abs(representative[axis]) <= self.tolerance:
                representative[axis] = 0.0
        return SamplingSymmetryImage(
            tuple(float(value) for value in representative), operation
        )

    def _reflected_coordinates(self, coordinates):
        reflected = np.asarray(coordinates, dtype=float).copy()
        if reflected.ndim != 1 or max(self.odd_axes) >= reflected.size:
            raise ValueError(
                "phenol reflection needs a one-dimensional coordinate containing "
                "the configured torsion axis"
            )
        reflected[list(self.odd_axes)] *= -1.0
        for axis in self.odd_axes:
            if abs(reflected[axis]) <= self.tolerance:
                reflected[axis] = 0.0
        return tuple(float(value) for value in reflected)

    def images(self, coordinates):
        coordinates = tuple(float(value) for value in coordinates)
        reflected = self._reflected_coordinates(coordinates)
        return tuple(dict.fromkeys((coordinates, reflected)))

    def pair_images(self, left, right):
        left = tuple(float(value) for value in left)
        right = tuple(float(value) for value in right)
        reflected = (
            self._reflected_coordinates(left),
            self._reflected_coordinates(right),
        )
        return tuple(dict.fromkeys(((left, right), reflected)))

    @staticmethod
    def _basis(protocol):
        if not isinstance(protocol, dict) or "basis" not in protocol:
            raise ValueError(
                "PhenolReflectionSymmetry needs protocol['basis'] to transform "
                "molecular orbitals"
            )
        return protocol["basis"]

    @staticmethod
    @lru_cache(maxsize=16)
    def _ao_signs(basis_key):
        from pyqed.models.phenol_coordinates import (
            PHENOL_SPECIES,
            PhenolReactiveChart,
        )
        from pyqed.qchem import Molecule
        from pyqed.qchem.symmetry import _component_parity

        basis = basis_key
        chart = PhenolReactiveChart()
        geometry = chart.geometry(chart.equilibrium)
        molecule = Molecule(
            atom=list(zip(PHENOL_SPECIES, geometry)),
            unit="angstrom",
            basis=basis,
            charge=0,
            spin=0,
        ).topyscf()
        molecule.build(verbose=0)
        signs = []
        for _atom, _symbol, _shell, component in molecule.ao_labels(fmt=False):
            component = str(component).replace("^", "").strip()
            signs.append(_component_parity(component, (1, 1, -1)))
        return np.asarray(signs, dtype=float)

    def _reflection_signs(self, basis):
        if not isinstance(basis, str):
            raise TypeError(
                "PhenolReflectionSymmetry currently requires a named string basis"
            )
        return self._ao_signs(str(basis))

    @staticmethod
    def _reflect_cartesian(value):
        value = np.asarray(value)
        return np.einsum("...a,ba->...b", value, PhenolReflectionSymmetry.matrix)

    def transform_record(
        self,
        record,
        image,
        *,
        representative_geometry,
        requested_geometry,
        protocol,
    ):
        if image.operation == "identity":
            return record
        if image.operation != self.operation:
            raise ValueError(f"unsupported phenol orbit operation {image.operation!r}")
        if not isinstance(record, dict):
            raise TypeError(
                "PhenolReflectionSymmetry expects mapping electronic records"
            )

        representative_geometry = np.asarray(representative_geometry, dtype=float)
        requested_geometry = np.asarray(requested_geometry, dtype=float)
        reflected = representative_geometry @ self.matrix.T
        if not np.allclose(
            reflected,
            requested_geometry,
            atol=max(self.tolerance, 1.0e-10),
            rtol=0.0,
        ):
            raise ValueError(
                "the requested phenol geometry is not the sigma_xy image of its "
                "orbit representative"
            )

        transformed = dict(record)
        transformed["geometry"] = np.array(requested_geometry, copy=True)
        if "mo_coeff" in record:
            signs = self._reflection_signs(self._basis(protocol))
            coefficients = np.asarray(record["mo_coeff"])
            if coefficients.ndim != 2 or coefficients.shape[0] != signs.size:
                raise ValueError(
                    "mo_coeff does not match the AO basis used by the phenol protocol"
                )
            transformed["mo_coeff"] = signs[:, None] * coefficients

        for key in (
            "gradient",
            "gradients",
            "force",
            "forces",
            "dipole",
            "dipoles",
            "transition_dipole",
            "transition_dipoles",
            "nac",
            "nacs",
        ):
            if key in record and np.asarray(record[key]).shape[-1:] == (3,):
                transformed[key] = self._reflect_cartesian(record[key])
        transformed["sampling_symmetry"] = self.view_key(image)
        return transformed

    def metadata(self):
        return {
            "name": self.name,
            "coordinate_axis": self.torsion_axis,
            "odd_coordinate_axes": list(self.odd_axes),
            "canonical_domain": "first nonzero odd coordinate >= 0",
            "operation": self.operation,
            "cartesian_matrix": self.matrix.tolist(),
        }


__all__ = [
    "SamplingSymmetry",
    "SamplingSymmetryImage",
    "PhenolReflectionSymmetry",
]
