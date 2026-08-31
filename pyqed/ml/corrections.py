"""Low-dimensional corrections for fitted matrix fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


class ReflectionScalarMLP:
    """Small NumPy MLP with exact coordinate-reflection symmetrization."""

    def __init__(
        self,
        lower,
        upper,
        parities,
        weights,
        biases,
        *,
        output_shift=0.0,
        output_scale=1.0,
        periodic_axes=(),
        periodic_harmonics=1,
    ):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.parities = np.asarray(parities, dtype=float)
        self.periodic_axes = tuple(sorted({int(axis) for axis in periodic_axes}))
        self.periodic_harmonics = int(periodic_harmonics)
        self.weights = tuple(np.asarray(value, dtype=float) for value in weights)
        self.biases = tuple(np.asarray(value, dtype=float) for value in biases)
        if (
            self.lower.ndim != 1
            or self.upper.shape != self.lower.shape
            or self.parities.shape != self.lower.shape
            or np.any(self.upper <= self.lower)
            or any(axis < 0 or axis >= len(self.lower) for axis in self.periodic_axes)
            or self.periodic_harmonics < 1
        ):
            raise ValueError("scalar MLP coordinate metadata is inconsistent")
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError("scalar MLP weights and biases are inconsistent")
        size = len(self.lower) + len(self.periodic_axes) * (
            2 * self.periodic_harmonics - 1
        )
        for weight, bias in zip(self.weights, self.biases):
            if weight.ndim != 2 or weight.shape[1] != size:
                raise ValueError("scalar MLP layer dimensions are inconsistent")
            if bias.shape != (weight.shape[0],):
                raise ValueError("scalar MLP bias has the wrong shape")
            size = weight.shape[0]
        if size != 1:
            raise ValueError("scalar MLP must have one output")
        self.output_shift = float(output_shift)
        self.output_scale = float(output_scale)
        self.output_shape_ = ()

    @staticmethod
    def _silu(values):
        positive = values >= 0.0
        result = np.empty_like(values)
        result[positive] = values[positive] / (1.0 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        result[~positive] = values[~positive] * exponential / (1.0 + exponential)
        return result

    def _features(self, coordinates):
        blocks = []
        periodic = set(self.periodic_axes)
        for axis in range(len(self.lower)):
            if axis in periodic:
                angle = (
                    2.0
                    * np.pi
                    * (coordinates[:, axis] - 0.5 * (self.lower[axis] + self.upper[axis]))
                    / (self.upper[axis] - self.lower[axis])
                )
                for harmonic in range(1, self.periodic_harmonics + 1):
                    blocks.extend(
                        (np.sin(harmonic * angle), np.cos(harmonic * angle))
                    )
            else:
                blocks.append(
                    2.0
                    * (coordinates[:, axis] - self.lower[axis])
                    / (self.upper[axis] - self.lower[axis])
                    - 1.0
                )
        return np.column_stack(blocks)

    def _raw(self, coordinates):
        values = self._features(coordinates)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values = values @ weight.T + bias
            if index + 1 < len(self.weights):
                values = self._silu(values)
        return values[:, 0]

    def predict(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        scalar = coordinates.ndim == 1
        if scalar:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.lower):
            raise ValueError("coordinates have the wrong shape")
        reflected = coordinates * self.parities
        raw = 0.5 * (self._raw(coordinates) + self._raw(reflected))
        result = self.output_shift + self.output_scale * raw
        return result[0] if scalar else result

    def save(self, filename):
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lower": self.lower,
            "upper": self.upper,
            "parities": self.parities,
            "periodic_axes": np.asarray(self.periodic_axes, dtype=int),
            "periodic_harmonics": np.asarray(self.periodic_harmonics),
            "output_shift": np.asarray(self.output_shift),
            "output_scale": np.asarray(self.output_scale),
            "layers": np.asarray(len(self.weights)),
        }
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"weight_{index}"] = weight
            payload[f"bias_{index}"] = bias
        np.savez_compressed(filename, **payload)
        return filename

    @classmethod
    def load(cls, filename):
        with np.load(filename, allow_pickle=False) as archive:
            layers = int(archive["layers"])
            return cls(
                archive["lower"],
                archive["upper"],
                archive["parities"],
                [archive[f"weight_{index}"] for index in range(layers)],
                [archive[f"bias_{index}"] for index in range(layers)],
                output_shift=float(archive["output_shift"]),
                output_scale=float(archive["output_scale"]),
                periodic_axes=(
                    archive["periodic_axes"]
                    if "periodic_axes" in archive.files
                    else ()
                ),
                periodic_harmonics=(
                    int(archive["periodic_harmonics"])
                    if "periodic_harmonics" in archive.files
                    else 1
                ),
            )


class RadialMatrixCorrection:
    """Shape-preserving one-coordinate correction to a matrix-valued field."""

    def __init__(self, knots, values, *, coordinate=0):
        knots = np.asarray(knots, dtype=float)
        values = np.asarray(values, dtype=complex)
        coordinate = int(coordinate)
        if knots.ndim != 1 or len(knots) < 2 or np.any(np.diff(knots) <= 0.0):
            raise ValueError("knots must be a strictly increasing one-dimensional array")
        if values.ndim != 3 or values.shape[0] != len(knots):
            raise ValueError("values must have shape (knots, rows, columns)")
        if values.shape[1] != values.shape[2]:
            raise ValueError("correction matrices must be square")
        if coordinate < 0:
            raise ValueError("coordinate must be nonnegative")
        self.knots = knots
        self.values = 0.5 * (values + values.conj().swapaxes(-1, -2))
        self.coordinate = coordinate
        self.output_shape_ = values.shape[1:]
        self._real = PchipInterpolator(knots, self.values.real, axis=0)
        self._imag = PchipInterpolator(knots, self.values.imag, axis=0)

    @classmethod
    def fit(
        cls,
        coordinates,
        target,
        baseline,
        *,
        coordinate=0,
        representation=None,
    ):
        """Fit the mean matrix residual independently at every radial knot."""

        coordinates = np.asarray(coordinates, dtype=float)
        target = np.asarray(target, dtype=complex)
        baseline = np.asarray(baseline, dtype=complex)
        coordinate = int(coordinate)
        if coordinates.ndim != 2 or coordinate >= coordinates.shape[1]:
            raise ValueError("coordinates and selected coordinate are incompatible")
        expected = (len(coordinates), *target.shape[1:])
        if target.ndim != 3 or target.shape != expected or baseline.shape != expected:
            raise ValueError("target and baseline must be equally shaped matrix fields")
        residual = target - baseline
        residual = 0.5 * (residual + residual.conj().swapaxes(-1, -2))
        if representation is not None:
            representation = np.asarray(representation, dtype=complex)
            if representation.shape != target.shape[1:]:
                raise ValueError("representation has the wrong matrix shape")
            residual = 0.5 * (
                residual
                + representation.conj().T @ residual @ representation
            )
        radii = coordinates[:, coordinate]
        knots = np.unique(radii)
        values = np.asarray(
            [np.mean(residual[np.isclose(radii, knot)], axis=0) for knot in knots]
        )
        return cls(knots, values, coordinate=coordinate)

    def predict(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2 or self.coordinate >= coordinates.shape[1]:
            raise ValueError("coordinates have the wrong shape")
        radial = np.clip(
            coordinates[:, self.coordinate], self.knots[0], self.knots[-1]
        )
        values = self._real(radial) + 1.0j * self._imag(radial)
        return 0.5 * (values + values.conj().swapaxes(-1, -2))

    def save(self, filename):
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filename,
            knots=self.knots,
            values=self.values,
            coordinate=np.asarray(self.coordinate),
        )
        return filename

    @classmethod
    def load(cls, filename):
        with np.load(filename, allow_pickle=False) as archive:
            return cls(
                archive["knots"], archive["values"],
                coordinate=int(archive["coordinate"]),
            )


class CorrectedMatrixField:
    """Add a saved low-dimensional correction to an existing matrix field."""

    def __init__(self, field, correction):
        if not hasattr(field, "predict"):
            raise TypeError("field must provide predict(coordinates)")
        if tuple(getattr(field, "output_shape_", ())) != correction.output_shape_:
            raise ValueError("field and correction output shapes differ")
        self.field = field
        self.correction = correction
        self.output_shape_ = correction.output_shape_

    def predict(self, coordinates):
        return self.field.predict(coordinates) + self.correction.predict(coordinates)


__all__ = [
    "CorrectedMatrixField",
    "RadialMatrixCorrection",
    "ReflectionScalarMLP",
]
