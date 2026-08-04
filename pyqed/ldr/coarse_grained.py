import os
import pickle
import string
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import psutil
from matplotlib import pyplot as plt
from pyqed.dvr.dvr import DVR
from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPS, MPO, gwp_mps
from pyqed.phys import interval, gwp
from pyqed.units import *

try:
    import torch
    from torch.autograd.functional import jacobian
except ModuleNotFoundError as exc:
    torch = None
    jacobian = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch_backend():
    """Return the optional torch backend or raise a targeted install error."""

    if torch is None:
        raise RuntimeError(
            "This CGLDR path requires the optional torch backend. Install "
            "torch>=2.2, or install this package with the ml extra, before "
            "using Torch autodiff or backend='torch'."
        ) from _TORCH_IMPORT_ERROR
    return torch


def _optional_torch_no_grad():
    if torch is None:
        return nullcontext()
    return torch.no_grad()


def sync_gauge(links, npoints, *, root=0, robust_scale=None, max_cycle=8):
    """Synchronize local electronic frames from polar-overlap graph links.

    Each link is ``(i, j, overlap)`` or ``(i, j, overlap, weight)``. Returned
    matrices rotate the original local states into one smooth global gauge.
    """
    links = list(links)
    npoints = int(npoints)
    root = int(root)
    if npoints < 1 or not 0 <= root < npoints:
        raise ValueError("npoints must be positive and root must be a valid point")
    if not links:
        raise ValueError("links cannot be empty")

    first = np.asarray(links[0][2], dtype=np.complex128)
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("overlap links must be square matrices")
    nstates = first.shape[0]
    parsed = []
    for link in links:
        if len(link) not in (3, 4):
            raise ValueError("each link must contain i, j, overlap, and optional weight")
        i, j = int(link[0]), int(link[1])
        block = np.asarray(link[2], dtype=np.complex128)
        weight = 1.0 if len(link) == 3 else float(link[3])
        if not (0 <= i < npoints and 0 <= j < npoints) or i == j:
            raise ValueError("link indices must be distinct valid points")
        if block.shape != (nstates, nstates):
            raise ValueError("all overlap links must have the same square shape")
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("link weights must be positive and finite")
        left, _singular, right = np.linalg.svd(block, full_matrices=False)
        transport = left @ right
        parsed.append((i, j, transport, weight))

    if robust_scale is not None:
        robust_scale = float(robust_scale)
        if not np.isfinite(robust_scale) or robust_scale <= 0.0:
            raise ValueError("robust_scale must be positive and finite")
    max_cycle = int(max_cycle)
    if max_cycle < 1:
        raise ValueError("max_cycle must be positive")
    effective = np.asarray([link[3] for link in parsed], dtype=float)
    frames = None
    for _cycle in range(max_cycle if robust_scale is not None else 1):
        size = npoints * nstates
        laplacian = np.zeros((size, size), dtype=np.complex128)
        for (i, j, transport, _base_weight), weight in zip(parsed, effective):
            si = slice(i * nstates, (i + 1) * nstates)
            sj = slice(j * nstates, (j + 1) * nstates)
            laplacian[si, si] += weight * np.eye(nstates)
            laplacian[sj, sj] += weight * np.eye(nstates)
            laplacian[si, sj] -= weight * transport
            laplacian[sj, si] -= weight * transport.conj().T
        values, vectors = np.linalg.eigh(
            0.5 * (laplacian + laplacian.conj().T)
        )
        scale = max(float(values[-1]), 1.0)
        if values[nstates] <= 1.0e-12 * scale:
            raise ValueError("overlap-link graph is disconnected")
        modes = vectors[:, :nstates].reshape(npoints, nstates, nstates)
        frames = np.empty_like(modes)
        for point, mode in enumerate(modes):
            left, _singular, right = np.linalg.svd(mode, full_matrices=False)
            frames[point] = left @ right
        if robust_scale is None:
            break
        residuals = np.asarray([
            np.linalg.norm(
                frames[i].conj().T @ transport @ frames[j]
                - np.eye(nstates)
            ) / np.sqrt(nstates)
            for i, j, transport, _base_weight in parsed
        ])
        robust = np.minimum(1.0, robust_scale / np.maximum(residuals, 1.0e-15))
        updated = np.asarray([link[3] for link in parsed]) * robust
        if np.max(np.abs(updated - effective)) <= 1.0e-8 * max(np.max(effective), 1.0):
            break
        effective = updated
    frames = frames @ frames[root].conj().T
    return frames


@dataclass(frozen=True)
class OverlapBasis:
    """Feature basis for a positive-semidefinite electronic overlap Gramian.

    ``vectors[p, i, a]`` embeds electronic state ``i`` at nuclear point ``p``
    in a common feature space, so that
    ``S[p, i, q, j] = sum_a vectors[p, i, a].conj() * vectors[q, j, a]``.
    """

    vectors: np.ndarray
    eigenvalues: np.ndarray
    residual: float = 0.0

    def __post_init__(self):
        vectors = np.asarray(self.vectors, dtype=np.complex128)
        eigenvalues = np.asarray(self.eigenvalues, dtype=float)
        if vectors.ndim != 3:
            raise ValueError("vectors must have shape (npoints, nstates, rank)")
        if eigenvalues.ndim != 1 or eigenvalues.size != vectors.shape[-1]:
            raise ValueError("eigenvalues must contain one value per feature")
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "residual", float(self.residual))

    @property
    def rank(self):
        return self.vectors.shape[-1]

    @classmethod
    def fit(cls, blocks, *, tol=1.0e-12, max_rank=None, check_self=True):
        """Factor overlap blocks by a Hermitian eigendecomposition."""
        blocks = np.asarray(blocks, dtype=np.complex128)
        if (
            blocks.ndim != 4
            or blocks.shape[0] != blocks.shape[2]
            or blocks.shape[1] != blocks.shape[3]
        ):
            raise ValueError(
                "blocks must have shape (npoints, nstates, npoints, nstates)"
            )
        npoints, nstates = blocks.shape[:2]
        if check_self:
            identity = np.eye(nstates)
            error = max(
                np.linalg.norm(blocks[p, :, p, :] - identity)
                for p in range(npoints)
            )
            if error > 1.0e-8:
                raise ValueError(
                    f"overlap self-blocks are not identity (max error={error:.3e})"
                )

        gram = blocks.reshape(npoints * nstates, npoints * nstates)
        hermitian_error = np.linalg.norm(gram - gram.conj().T)
        scale = max(np.linalg.norm(gram), 1.0)
        if hermitian_error > 1.0e-10 * scale:
            raise ValueError(
                f"overlap Gramian is not Hermitian (error={hermitian_error:.3e})"
            )
        gram = 0.5 * (gram + gram.conj().T)
        values, vectors = np.linalg.eigh(gram)
        spectral_scale = max(float(values[-1]), 1.0)
        if values[0] < -float(tol) * spectral_scale:
            raise ValueError(
                "overlap Gramian is not positive semidefinite "
                f"(minimum eigenvalue={values[0]:.3e})"
            )
        keep = np.flatnonzero(values > float(tol) * spectral_scale)
        if max_rank is not None:
            max_rank = int(max_rank)
            if max_rank < 1:
                raise ValueError("max_rank must be positive")
            keep = keep[-max_rank:]
        kept_values = values[keep]
        embedding = (
            vectors[:, keep].conj() * np.sqrt(kept_values)[None, :]
        ).reshape(npoints, nstates, -1)
        reconstructed = embedding.conj().reshape(npoints * nstates, -1) @ (
            embedding.reshape(npoints * nstates, -1).T
        )
        residual = np.linalg.norm(gram - reconstructed) / scale
        return cls(embedding, kept_values, residual)

    def blocks(self):
        """Reconstruct ``S[p, i, q, j]`` from the feature vectors."""
        return np.einsum(
            "pia,qja->piqj",
            self.vectors.conj(),
            self.vectors,
            optimize=True,
        )

    def apply_kinetic(self, kinetic, coefficients):
        """Apply ``T[p,q] S[p,i,q,j]`` without forming that full matrix."""
        kinetic = np.asarray(kinetic)
        coefficients = np.asarray(coefficients)
        if kinetic.shape != (self.vectors.shape[0],) * 2:
            raise ValueError("kinetic must have shape (npoints, npoints)")
        if coefficients.shape != self.vectors.shape[:2]:
            raise ValueError("coefficients must have shape (npoints, nstates)")
        embedded = np.einsum(
            "qja,qj->qa", self.vectors, coefficients, optimize=True
        )
        propagated = kinetic @ embedded
        return np.einsum(
            "pia,pa->pi", self.vectors.conj(), propagated, optimize=True
        )


def project_basis(basis, coordinates, hamiltonians, query, *, neighbors=16, decay=4.0):
    """Interpolate electronic subspaces without choosing a sampled gauge.

    Local projectors and Hamiltonian operators are averaged in the common
    feature space, then reduced to the leading ``nstates``-dimensional
    subspace. The result is covariant under independent unitary rotations of
    every sampled electronic frame.
    """
    if not isinstance(basis, OverlapBasis):
        raise TypeError("basis must be an OverlapBasis")
    coordinates = np.asarray(coordinates, dtype=float)
    query = np.asarray(query, dtype=float)
    hamiltonians = np.asarray(hamiltonians, dtype=np.complex128)
    npoints, nstates, rank = basis.vectors.shape
    if coordinates.ndim != 2 or coordinates.shape[0] != npoints:
        raise ValueError("coordinates must have shape (npoints, ndim)")
    if query.ndim != 2 or query.shape[1] != coordinates.shape[1]:
        raise ValueError("query must have shape (nquery, ndim)")
    if hamiltonians.shape == (npoints, nstates):
        hamiltonians = np.asarray([
            np.diag(values) for values in hamiltonians
        ])
    if hamiltonians.shape != (npoints, nstates, nstates):
        raise ValueError(
            "hamiltonians must have shape (npoints, nstates) or "
            "(npoints, nstates, nstates)"
        )
    neighbors = min(int(neighbors), npoints)
    decay = float(decay)
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if not np.isfinite(decay) or decay <= 0.0:
        raise ValueError("decay must be positive and finite")

    sampled_kets = basis.vectors.transpose(0, 2, 1)
    query_kets = np.empty((len(query), rank, nstates), dtype=np.complex128)
    query_energies = np.empty((len(query), nstates), dtype=float)
    distances = np.linalg.norm(
        query[:, None, :] - coordinates[None, :, :], axis=-1
    )
    for point, row in enumerate(distances):
        nearest = np.argsort(row)[:neighbors]
        if row[nearest[0]] <= 1.0e-14:
            nearest = nearest[:1]
            weights = np.ones(1)
        else:
            width = max(float(row[nearest[-1]]), 1.0e-14)
            weights = np.exp(-decay * (row[nearest] / width) ** 2)
            weights /= np.sum(weights)
        span = np.concatenate([
            np.sqrt(weight) * sampled_kets[index]
            for index, weight in zip(nearest, weights)
        ], axis=1)
        local_basis, _singular, _right = np.linalg.svd(
            span, full_matrices=False
        )
        local_basis = local_basis[:, :nstates]
        metric = np.zeros((nstates, nstates), dtype=np.complex128)
        operator = np.zeros_like(metric)
        for index, weight in zip(nearest, weights):
            bridge = local_basis.conj().T @ sampled_kets[index]
            metric += weight * bridge @ bridge.conj().T
            operator += (
                weight * bridge @ hamiltonians[index] @ bridge.conj().T
            )
        values, vectors = np.linalg.eigh(
            0.5 * (metric + metric.conj().T)
        )
        inverse_sqrt = (
            vectors / np.sqrt(np.maximum(values, 1.0e-12))[None, :]
        ) @ vectors.conj().T
        effective = inverse_sqrt @ operator @ inverse_sqrt
        energies, rotation = np.linalg.eigh(
            0.5 * (effective + effective.conj().T)
        )
        query_energies[point] = energies
        query_kets[point] = local_basis @ rotation
    projected = OverlapBasis(
        query_kets.transpose(0, 2, 1),
        basis.eigenvalues,
        basis.residual,
    )
    return projected, query_energies


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _cubic_hermite_factors(coordinates, anchors, *, extrapolation="linear"):
    """Return nodal value/slope factors for a one-dimensional Hermite spline."""
    coordinates = np.asarray(coordinates, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    if coordinates.ndim != 1:
        raise ValueError("coordinates must be one-dimensional")
    if (
        anchors.ndim != 1
        or anchors.size < 2
        or np.any(np.diff(anchors) <= 0.0)
    ):
        raise ValueError("anchors must contain at least two ordered values")
    if extrapolation not in ("error", "linear"):
        raise ValueError("extrapolation must be 'error' or 'linear'")
    outside = (coordinates < anchors[0]) | (coordinates > anchors[-1])
    if extrapolation == "error" and np.any(outside):
        raise ValueError("coordinates lie outside the anchor interval")

    factors = np.zeros((2 * anchors.size, coordinates.size), dtype=float)
    for coordinate_index, coordinate in enumerate(coordinates):
        if coordinate < anchors[0]:
            factors[0, coordinate_index] = 1.0
            factors[1, coordinate_index] = coordinate - anchors[0]
            continue
        if coordinate > anchors[-1]:
            factors[-2, coordinate_index] = 1.0
            factors[-1, coordinate_index] = coordinate - anchors[-1]
            continue

        interval_index = np.searchsorted(
            anchors,
            coordinate,
            side="right",
        ) - 1
        interval_index = int(np.clip(
            interval_index,
            0,
            anchors.size - 2,
        ))
        left = anchors[interval_index]
        width = anchors[interval_index + 1] - left
        reduced = (coordinate - left) / width
        reduced2 = reduced**2
        reduced3 = reduced2 * reduced
        factors[2 * interval_index, coordinate_index] = (
            2.0 * reduced3 - 3.0 * reduced2 + 1.0
        )
        factors[2 * interval_index + 1, coordinate_index] = width * (
            reduced3 - 2.0 * reduced2 + reduced
        )
        factors[2 * interval_index + 2, coordinate_index] = (
            -2.0 * reduced3 + 3.0 * reduced2
        )
        factors[2 * interval_index + 3, coordinate_index] = width * (
            reduced3 - reduced2
        )
    return factors


@dataclass(frozen=True)
class SeparableHamiltonian:
    """Low-rank coordinate expansion of an electronic Hamiltonian.

    ``operators`` has shape ``(*sampled, nterms, nstates, nstates)``.
    Every entry of ``factors`` has shape ``(nterms, npoints)`` and corresponds
    to one expanded coordinate. Together they represent

    ``sum_r operators[..., r, :, :] * prod_a factors[a][r, :]``.
    """

    operators: np.ndarray
    factors: tuple[np.ndarray, ...]

    def __post_init__(self):
        operators = np.asarray(self.operators)
        factors = tuple(np.asarray(factor) for factor in self.factors)
        if operators.ndim < 3 or operators.shape[-1] != operators.shape[-2]:
            raise ValueError(
                "operators must have shape "
                "(*sampled, nterms, nstates, nstates)"
            )
        nterms = operators.shape[-3]
        if nterms == 0:
            raise ValueError("operators must contain at least one term")
        for factor in factors:
            if factor.ndim != 2 or factor.shape[0] != nterms:
                raise ValueError(
                    "each coordinate factor must have shape (nterms, npoints)"
                )
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "factors", factors)

    @classmethod
    def polynomial(cls, coordinates, coefficients, *, center=0.0):
        """Build a one-coordinate polynomial separable Hamiltonian.

        ``coefficients[..., p, :, :]`` multiplies ``(q - center)**p``.  This
        is useful for locally transported LPA fits when a quadratic secondary
        coordinate is too restrictive but a full electronic grid is unnecessary.
        """
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 1:
            raise ValueError("coordinates must be one-dimensional")
        coefficients = np.asarray(coefficients)
        if (
            coefficients.ndim < 3
            or coefficients.shape[-1] != coefficients.shape[-2]
        ):
            raise ValueError(
                "coefficients must have shape "
                "(*sampled, degree + 1, nstates, nstates)"
            )
        nterms = coefficients.shape[-3]
        if nterms == 0:
            raise ValueError("coefficients must contain at least one term")
        delta = coordinates - float(center)
        factors = np.vstack([delta**power for power in range(nterms)])
        return cls(operators=coefficients, factors=(factors,))

    @classmethod
    def cubic_hermite(
        cls,
        coordinates,
        anchors,
        hamiltonians,
        gradients,
        *,
        extrapolation="linear",
    ):
        """Build a separable one-coordinate piecewise-cubic interpolant."""
        hamiltonians = np.asarray(hamiltonians)
        gradients = np.asarray(gradients)
        anchors = np.asarray(anchors, dtype=float)
        if (
            hamiltonians.shape != gradients.shape
            or hamiltonians.ndim < 3
            or hamiltonians.shape[-3] != anchors.size
            or hamiltonians.shape[-1] != hamiltonians.shape[-2]
        ):
            raise ValueError(
                "anchor values and gradients must have matching shape "
                "(*sampled, nanchors, nstates, nstates)"
            )
        operators = np.stack(
            (hamiltonians, gradients),
            axis=-3,
        ).reshape(
            *hamiltonians.shape[:-3],
            2 * anchors.size,
            *hamiltonians.shape[-2:],
        )
        factors = _cubic_hermite_factors(
            coordinates,
            anchors,
            extrapolation=extrapolation,
        )
        return cls(operators=operators, factors=(factors,))

    @classmethod
    def axial_cubic_hermite(
        cls,
        coordinates,
        anchors,
        hamiltonians,
        gradients,
        *,
        center_hamiltonian,
        centers=None,
        mixed_hessians=None,
        extrapolation="linear",
    ):
        """Build a sparse multi-coordinate mode-combination expansion.

        Each coordinate is interpolated on its own axial anchor line. Optional
        mixed Hessians add pair terms about ``centers`` without introducing a
        Cartesian anchor grid.
        """
        coordinates = tuple(np.asarray(values, dtype=float) for values in coordinates)
        anchors = tuple(np.asarray(values, dtype=float) for values in anchors)
        hamiltonians = tuple(np.asarray(values) for values in hamiltonians)
        gradients = tuple(np.asarray(values) for values in gradients)
        ndim = len(coordinates)
        if (
            ndim == 0
            or len(anchors) != ndim
            or len(hamiltonians) != ndim
            or len(gradients) != ndim
        ):
            raise ValueError(
                "coordinates, anchors, hamiltonians, and gradients must "
                "contain one entry per expanded coordinate"
            )
        if isinstance(extrapolation, str):
            extrapolation = (extrapolation,) * ndim
        else:
            extrapolation = tuple(extrapolation)
        if len(extrapolation) != ndim:
            raise ValueError(
                "extrapolation must contain one policy per expanded coordinate"
            )

        center_hamiltonian = np.asarray(center_hamiltonian)
        if (
            center_hamiltonian.ndim < 2
            or center_hamiltonian.shape[-1]
            != center_hamiltonian.shape[-2]
        ):
            raise ValueError(
                "center_hamiltonian must have shape "
                "(*sampled, nstates, nstates)"
            )
        sampled_shape = center_hamiltonian.shape[:-2]
        matrix_shape = center_hamiltonian.shape[-2:]
        centers = (
            np.zeros(ndim, dtype=float)
            if centers is None
            else np.asarray(centers, dtype=float)
        )
        if centers.shape != (ndim,) or not np.all(np.isfinite(centers)):
            raise ValueError("centers must contain one finite value per coordinate")

        operator_terms = [center_hamiltonian]
        factor_terms = [[
            np.ones(values.size) for values in coordinates
        ]]
        for axis in range(ndim):
            values = hamiltonians[axis]
            slopes = gradients[axis]
            expected = (*sampled_shape, anchors[axis].size, *matrix_shape)
            if values.shape != expected or slopes.shape != expected:
                raise ValueError(
                    f"axis {axis} anchor values and gradients must have "
                    f"shape {expected}"
                )
            center_matches = np.flatnonzero(
                np.isclose(anchors[axis], centers[axis], rtol=0.0, atol=1.0e-12)
            )
            if center_matches.size != 1:
                raise ValueError(
                    f"axis {axis} anchors must contain center {centers[axis]}"
                )
            center_value = values[..., int(center_matches[0]), :, :]
            if not np.allclose(
                center_value,
                center_hamiltonian,
                rtol=1.0e-8,
                atol=1.0e-10,
            ):
                raise ValueError(
                    f"axis {axis} Hamiltonian at the center does not match "
                    "center_hamiltonian"
                )

            delta_values = values - center_hamiltonian[..., None, :, :]
            axis_operators = np.stack(
                (delta_values, slopes),
                axis=-3,
            ).reshape(
                *sampled_shape,
                2 * anchors[axis].size,
                *matrix_shape,
            )
            axis_factors = _cubic_hermite_factors(
                coordinates[axis],
                anchors[axis],
                extrapolation=extrapolation[axis],
            )
            for term in range(axis_operators.shape[-3]):
                operator_terms.append(axis_operators[..., term, :, :])
                factor_terms.append([
                    axis_factors[term]
                    if current == axis
                    else np.ones(coordinates[current].size)
                    for current in range(ndim)
                ])

        if mixed_hessians is not None:
            mixed_hessians = np.asarray(mixed_hessians)
            expected = (*sampled_shape, ndim, ndim, *matrix_shape)
            if mixed_hessians.shape != expected:
                raise ValueError(
                    f"mixed_hessians shape {mixed_hessians.shape} != {expected}"
                )
            if not np.allclose(
                mixed_hessians,
                mixed_hessians.swapaxes(
                    len(sampled_shape),
                    len(sampled_shape) + 1,
                ),
                rtol=1.0e-8,
                atol=1.0e-10,
            ):
                raise ValueError(
                    "mixed_hessians must be symmetric in coordinate indices"
                )
            for first in range(ndim - 1):
                for second in range(first + 1, ndim):
                    operator_terms.append(
                        mixed_hessians[..., first, second, :, :]
                    )
                    factor_terms.append([
                        (
                            coordinates[current] - centers[current]
                            if current in (first, second)
                            else np.ones(coordinates[current].size)
                        )
                        for current in range(ndim)
                    ])

        operators = np.stack(operator_terms, axis=len(sampled_shape))
        factors = tuple(
            np.stack(
                [term[axis] for term in factor_terms],
                axis=0,
            )
            for axis in range(ndim)
        )
        return cls(operators=operators, factors=factors)

    def evaluate(self):
        """Materialize the product-grid field for validation only."""
        sampled_shape = self.operators.shape[:-3]
        nstates = self.operators.shape[-1]
        expanded_shape = tuple(factor.shape[1] for factor in self.factors)
        field = np.zeros(
            (*sampled_shape, *expanded_shape, nstates, nstates),
            dtype=np.result_type(self.operators, *self.factors),
        )
        for term in range(self.operators.shape[-3]):
            value = self.operators[..., term, :, :].reshape(
                *sampled_shape,
                *(1,) * len(self.factors),
                nstates,
                nstates,
            )
            for axis, factor in enumerate(self.factors):
                shape = (
                    *(1,) * len(sampled_shape),
                    *(factor.shape[1] if current == axis else 1
                      for current in range(len(self.factors))),
                    1,
                    1,
                )
                value = value * factor[term].reshape(shape)
            field += value
        return 0.5 * (field + field.swapaxes(-1, -2).conj())


@dataclass(frozen=True)
class ElectronicPartition:
    """Partition DVR axes by their electronic-structure treatment."""

    sampled: tuple[str, ...]
    expanded: tuple[str, ...] = ()
    center: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "sampled", tuple(self.sampled))
        object.__setattr__(self, "expanded", tuple(self.expanded))
        object.__setattr__(
            self,
            "center",
            tuple(float(value) for value in self.center),
        )
        labels = self.sampled + self.expanded
        if any(not isinstance(name, str) or not name for name in labels):
            raise ValueError("partition coordinates must be non-empty names")
        if len(set(labels)) != len(labels):
            raise ValueError("sampled and expanded coordinates must be unique")
        if len(self.center) != len(self.expanded):
            raise ValueError(
                "center must contain one value per expanded coordinate"
            )
        if not np.all(np.isfinite(self.center)):
            raise ValueError("expansion center must be finite")

    def resolve(self, dvr):
        """Return sampled and expanded integer axes for ``dvr``."""
        unknown = [
            name
            for name in self.sampled + self.expanded
            if name not in dvr.axis_by_name
        ]
        if unknown:
            raise ValueError(
                "partition contains coordinates absent from the DVR: "
                + ", ".join(unknown)
            )
        assigned = set(self.sampled + self.expanded)
        missing = [name for name in dvr.names if name not in assigned]
        if missing:
            raise ValueError(
                "partition does not assign every DVR coordinate: "
                + ", ".join(missing)
            )
        return (
            tuple(dvr.axis(name) for name in self.sampled),
            tuple(dvr.axis(name) for name in self.expanded),
        )


@dataclass(frozen=True)
class CGLDRElectronicData:
    """Ab initio inputs required by :class:`CGLDR`.

    Arrays use grid-first ordering. Energies have shape
    ``(*grid, nstates)`` and overlaps have shape
    ``(*grid, nstates, *grid, nstates)``. Hamiltonian gradients and
    Hessians, expressed in the local adiabatic basis, have shapes
    ``(*grid, ncoarse, nstates, nstates)`` and
    ``(*grid, ncoarse, ncoarse, nstates, nstates)``.

    ``separable_hamiltonian`` optionally supplies a low-rank coordinate
    expansion that replaces the single-center derivative expansion during
    propagation.
    """

    energies: np.ndarray
    overlaps: np.ndarray
    hamiltonian_gradients: np.ndarray | None = None
    hamiltonian_hessians: np.ndarray | None = None
    separable_hamiltonian: SeparableHamiltonian | None = None
    reactive_grids: tuple[np.ndarray, ...] | None = None
    expanded_grids: tuple[np.ndarray, ...] | None = None
    basis_transforms: np.ndarray | None = None
    metric_eigenvalues: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_electronic_union(
        cls,
        *,
        overlaps,
        hamiltonian,
        rank=None,
        metric_tolerance=1.0e-8,
        reactive_grids=None,
        expanded_grids=None,
        metadata=None,
    ):
        """Compress a nonorthogonal electronic union into a fixed local basis.

        ``overlaps`` has shape
        ``(*sampled, nraw, *sampled, nraw)`` and includes the nonidentity
        self-overlap (Gram) blocks of the raw anchor union. ``hamiltonian`` is
        expressed in the same raw union and is transformed by canonical
        Löwdin orthogonalization independently at each sampled DVR point.
        """
        if not isinstance(hamiltonian, SeparableHamiltonian):
            raise TypeError("hamiltonian must be a SeparableHamiltonian")
        if not np.isfinite(metric_tolerance) or metric_tolerance <= 0.0:
            raise ValueError("metric_tolerance must be finite and positive")

        raw_operators = np.asarray(hamiltonian.operators)
        sampled_shape = raw_operators.shape[:-3]
        nraw = raw_operators.shape[-1]
        if raw_operators.shape[-2] != nraw:
            raise ValueError("raw union Hamiltonian operators must be square")
        if not np.all(np.isfinite(raw_operators)):
            raise ValueError(
                "raw union Hamiltonian operators contain non-finite values"
            )
        if not np.allclose(
            raw_operators,
            raw_operators.swapaxes(-1, -2).conj(),
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ValueError(
                "raw union Hamiltonian operators must be Hermitian"
            )
        ngrid = int(np.prod(sampled_shape))
        raw_overlaps = np.asarray(overlaps)
        expected_overlaps = (
            *sampled_shape,
            nraw,
            *sampled_shape,
            nraw,
        )
        if raw_overlaps.shape != expected_overlaps:
            raise ValueError(
                f"union overlaps shape {raw_overlaps.shape} "
                f"!= {expected_overlaps}"
            )
        if not np.all(np.isfinite(raw_overlaps)):
            raise ValueError("union overlaps contain non-finite values")

        blocks = raw_overlaps.reshape(ngrid, nraw, ngrid, nraw)
        if not np.allclose(
            blocks,
            blocks.transpose(2, 3, 0, 1).conj(),
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ValueError(
                "union overlaps must satisfy S(I,J) = S(J,I)†"
            )
        point_indices = np.arange(ngrid)
        grams = blocks[point_indices, :, point_indices, :]
        grams = 0.5 * (grams + grams.swapaxes(-1, -2).conj())
        eigenvalues = np.empty((ngrid, nraw), dtype=float)
        eigenvectors = np.empty(
            (ngrid, nraw, nraw),
            dtype=np.result_type(grams, complex),
        )
        available_ranks = np.empty(ngrid, dtype=int)
        for point, gram in enumerate(grams):
            values, vectors = np.linalg.eigh(gram)
            order = np.argsort(values)[::-1]
            values = np.asarray(values[order].real)
            vectors = vectors[:, order]
            scale = max(float(values[0]), 1.0)
            if values[-1] < -metric_tolerance * scale:
                raise ValueError(
                    "electronic-union Gram matrix is not positive "
                    f"semidefinite at sampled point {point}"
                )
            eigenvalues[point] = values
            eigenvectors[point] = vectors
            available_ranks[point] = int(np.count_nonzero(
                values > metric_tolerance * scale
            ))

        common_rank = int(np.min(available_ranks))
        if common_rank == 0:
            raise ValueError(
                "electronic-union Gram matrices have no common retained rank"
            )
        if rank is None:
            rank = common_rank
        elif (
            not isinstance(rank, (int, np.integer))
            or rank <= 0
            or rank > common_rank
        ):
            raise ValueError(
                f"rank must be an integer between 1 and {common_rank}"
            )
        rank = int(rank)
        transforms = (
            eigenvectors[..., :rank]
            / np.sqrt(eigenvalues[..., :rank])[:, None, :]
        )

        transformed_overlaps = np.einsum(
            "par,paqb,qbs->prqs",
            transforms.conj(),
            blocks,
            transforms,
            optimize=True,
        )
        transformed_operators = np.einsum(
            "par,ptab,pbs->ptrs",
            transforms.conj(),
            raw_operators.reshape(
                ngrid,
                raw_operators.shape[-3],
                nraw,
                nraw,
            ),
            transforms,
            optimize=True,
        ).reshape(
            *sampled_shape,
            raw_operators.shape[-3],
            rank,
            rank,
        )
        transformed_operators = 0.5 * (
            transformed_operators
            + transformed_operators.swapaxes(-1, -2).conj()
        )
        transformed_overlaps = transformed_overlaps.reshape(
            *sampled_shape,
            rank,
            *sampled_shape,
            rank,
        )
        output_metadata = {} if metadata is None else dict(metadata)
        output_metadata.update({
            "electronic_basis": "compressed-anchor-union",
            "raw_union_dimension": int(nraw),
            "retained_union_dimension": rank,
            "metric_tolerance": float(metric_tolerance),
            "minimum_retained_metric_eigenvalue": float(
                np.min(eigenvalues[:, rank - 1])
            ),
            "maximum_discarded_metric_eigenvalue": (
                0.0
                if rank == nraw
                else float(np.max(eigenvalues[:, rank]))
            ),
        })
        return cls(
            energies=np.zeros((*sampled_shape, rank), dtype=float),
            overlaps=transformed_overlaps,
            separable_hamiltonian=SeparableHamiltonian(
                operators=transformed_operators,
                factors=hamiltonian.factors,
            ),
            reactive_grids=(
                None
                if reactive_grids is None
                else tuple(np.asarray(grid) for grid in reactive_grids)
            ),
            expanded_grids=(
                None
                if expanded_grids is None
                else tuple(np.asarray(grid) for grid in expanded_grids)
            ),
            basis_transforms=transforms.reshape(
                *sampled_shape,
                nraw,
                rank,
            ),
            metric_eigenvalues=eigenvalues.reshape(
                *sampled_shape,
                nraw,
            ),
            metadata=output_metadata,
        )

    @classmethod
    def from_displaced_adiabatic_data(
        cls,
        *,
        energies,
        overlaps,
        displacements,
        displaced_energies,
        reference_to_displaced_overlaps,
        reactive_grids=None,
        metadata=None,
        rcond=None,
    ):
        """Fit coarse-coordinate Hamiltonian derivatives from ab initio scans.

        ``reference_to_displaced_overlaps[..., k, a, m]`` is
        :math:`\\langle\\phi_a(0)|\\phi_m(\\Delta Q_k)\\rangle`. It transports
        each displaced diagonal adiabatic Hamiltonian into the local reference
        basis before a quadratic least-squares fit.
        """
        energies = np.asarray(energies)
        if energies.ndim < 1:
            raise ValueError("energies must end with an electronic-state axis")
        grid_shape = energies.shape[:-1]
        nstates = energies.shape[-1]

        displacements = np.asarray(displacements, dtype=float)
        if displacements.ndim != 2 or displacements.shape[1] == 0:
            raise ValueError(
                "displacements must have shape (nsamples, ncoarse)"
            )
        nsamples, ncoarse = displacements.shape
        displaced_energies = np.asarray(displaced_energies)
        displaced_overlaps = np.asarray(reference_to_displaced_overlaps)
        expected_energies = (*grid_shape, nsamples, nstates)
        expected_overlaps = (
            *grid_shape,
            nsamples,
            nstates,
            nstates,
        )
        if displaced_energies.shape != expected_energies:
            raise ValueError(
                f"displaced_energies shape {displaced_energies.shape} "
                f"!= {expected_energies}"
            )
        if displaced_overlaps.shape != expected_overlaps:
            raise ValueError(
                "reference_to_displaced_overlaps shape "
                f"{displaced_overlaps.shape} != {expected_overlaps}"
            )
        if not np.all(np.isfinite(displacements)):
            raise ValueError("displacements contain non-finite values")
        if not np.all(np.isfinite(displaced_energies)):
            raise ValueError("displaced_energies contain non-finite values")
        if not np.all(np.isfinite(displaced_overlaps)):
            raise ValueError(
                "reference_to_displaced_overlaps contain non-finite values"
            )

        design_columns = [
            displacements[:, coordinate] for coordinate in range(ncoarse)
        ]
        design_columns.extend(
            0.5 * displacements[:, coordinate] ** 2
            for coordinate in range(ncoarse)
        )
        cross_pairs = [
            (first, second)
            for first in range(ncoarse - 1)
            for second in range(first + 1, ncoarse)
        ]
        design_columns.extend(
            displacements[:, first] * displacements[:, second]
            for first, second in cross_pairs
        )
        design = np.column_stack(design_columns)
        nterms = design.shape[1]
        rank = int(np.linalg.matrix_rank(design))
        if rank < nterms:
            raise ValueError(
                f"Displacement stencil rank {rank} is insufficient for "
                f"{nterms} linear/quadratic coefficients."
            )

        transported = np.einsum(
            "...kam,...km,...kbm->...kab",
            displaced_overlaps,
            displaced_energies,
            displaced_overlaps.conj(),
            optimize=True,
        )
        reference_hamiltonian = np.zeros(
            (*grid_shape, nstates, nstates),
            dtype=np.result_type(transported, energies),
        )
        states = np.arange(nstates)
        reference_hamiltonian[..., states, states] = energies
        differences = transported - reference_hamiltonian[..., None, :, :]
        sample_first = np.moveaxis(differences, len(grid_shape), 0)
        coefficients, residuals, _, _ = np.linalg.lstsq(
            design,
            sample_first.reshape(nsamples, -1),
            rcond=rcond,
        )
        coefficients = coefficients.reshape(
            nterms, *grid_shape, nstates, nstates
        )

        gradients = np.moveaxis(coefficients[:ncoarse], 0, len(grid_shape))
        hessians = np.zeros(
            (*grid_shape, ncoarse, ncoarse, nstates, nstates),
            dtype=coefficients.dtype,
        )
        for coordinate in range(ncoarse):
            hessians[..., coordinate, coordinate, :, :] = coefficients[
                ncoarse + coordinate
            ]
        offset = 2 * ncoarse
        for term, (first, second) in enumerate(cross_pairs):
            matrix = coefficients[offset + term]
            hessians[..., first, second, :, :] = matrix
            hessians[..., second, first, :, :] = matrix

        gradients = 0.5 * (
            gradients + gradients.swapaxes(-1, -2).conj()
        )
        hessians = 0.5 * (
            hessians + hessians.swapaxes(-1, -2).conj()
        )
        fit_residual = float(
            np.linalg.norm(
                design @ coefficients.reshape(nterms, -1)
                - sample_first.reshape(nsamples, -1)
            )
        )
        output_metadata = {} if metadata is None else dict(metadata)
        output_metadata.update({
            "coarse_derivative_source": "displaced_adiabatic_quadratic_fit",
            "displacement_fit_rank": rank,
            "displacement_fit_residual": fit_residual,
        })
        return cls(
            energies=energies,
            overlaps=np.asarray(overlaps),
            hamiltonian_gradients=gradients,
            hamiltonian_hessians=hessians,
            reactive_grids=(
                None
                if reactive_grids is None
                else tuple(np.asarray(grid) for grid in reactive_grids)
            ),
            metadata=output_metadata,
        )

    @classmethod
    def from_npz(cls, filename):
        """Load CGLDR inputs from a portable NumPy archive."""
        with np.load(filename, allow_pickle=False) as archive:
            energy_key = "energies" if "energies" in archive else "apes"
            overlap_key = (
                "overlaps" if "overlaps" in archive else "overlap_matrix"
            )
            if energy_key not in archive or overlap_key not in archive:
                raise ValueError(
                    "Electronic-data archive must contain energies/apes and "
                    "overlaps/overlap_matrix."
                )

            gradients = (
                archive["hamiltonian_gradients"]
                if "hamiltonian_gradients" in archive
                else None
            )
            hessians = (
                archive["hamiltonian_hessians"]
                if "hamiltonian_hessians" in archive
                else None
            )
            separable_hamiltonian = None
            if "separable_operators" in archive:
                factor_keys = sorted(
                    (
                        key
                        for key in archive.files
                        if key.startswith("separable_factor_")
                    ),
                    key=lambda key: int(key.rsplit("_", 1)[1]),
                )
                separable_hamiltonian = SeparableHamiltonian(
                    operators=np.asarray(archive["separable_operators"]),
                    factors=tuple(
                        np.asarray(archive[key]) for key in factor_keys
                    ),
                )
            grid_keys = sorted(
                (key for key in archive.files if key.startswith("grid_")),
                key=lambda key: int(key.split("_", 1)[1]),
            )
            grids = (
                tuple(np.asarray(archive[key]) for key in grid_keys)
                if grid_keys
                else None
            )
            expanded_grid_keys = sorted(
                (
                    key
                    for key in archive.files
                    if key.startswith("expanded_grid_")
                ),
                key=lambda key: int(key.rsplit("_", 1)[1]),
            )
            expanded_grids = (
                tuple(np.asarray(archive[key]) for key in expanded_grid_keys)
                if expanded_grid_keys
                else None
            )
            metadata = {}
            if "metadata_json" in archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
            basis_transforms = (
                np.asarray(archive["basis_transforms"])
                if "basis_transforms" in archive
                else None
            )
            metric_eigenvalues = (
                np.asarray(archive["metric_eigenvalues"])
                if "metric_eigenvalues" in archive
                else None
            )

            return cls(
                energies=np.asarray(archive[energy_key]),
                overlaps=np.asarray(archive[overlap_key]),
                hamiltonian_gradients=gradients,
                hamiltonian_hessians=hessians,
                separable_hamiltonian=separable_hamiltonian,
                reactive_grids=grids,
                expanded_grids=expanded_grids,
                basis_transforms=basis_transforms,
                metric_eigenvalues=metric_eigenvalues,
                metadata=metadata,
            )

    def to_npz(self, filename):
        """Save the electronic data without pickled object arrays."""
        arrays = {
            "energies": np.asarray(self.energies),
            "overlaps": np.asarray(self.overlaps),
            "metadata_json": np.array(json.dumps(
                self.metadata,
                sort_keys=True,
                default=_json_value,
            )),
        }
        if self.hamiltonian_gradients is not None:
            arrays["hamiltonian_gradients"] = np.asarray(
                self.hamiltonian_gradients
            )
        if self.hamiltonian_hessians is not None:
            arrays["hamiltonian_hessians"] = np.asarray(
                self.hamiltonian_hessians
            )
        if self.separable_hamiltonian is not None:
            arrays["separable_operators"] = np.asarray(
                self.separable_hamiltonian.operators
            )
            for axis, factor in enumerate(
                self.separable_hamiltonian.factors
            ):
                arrays[f"separable_factor_{axis}"] = np.asarray(factor)
        if self.reactive_grids is not None:
            for axis, grid in enumerate(self.reactive_grids):
                arrays[f"grid_{axis}"] = np.asarray(grid)
        if self.expanded_grids is not None:
            for axis, grid in enumerate(self.expanded_grids):
                arrays[f"expanded_grid_{axis}"] = np.asarray(grid)
        if self.basis_transforms is not None:
            arrays["basis_transforms"] = np.asarray(self.basis_transforms)
        if self.metric_eigenvalues is not None:
            arrays["metric_eigenvalues"] = np.asarray(
                self.metric_eigenvalues
            )
        np.savez(Path(filename), **arrays)


def clear_memory():
    import gc
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _as_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _decompose_dense(value, max_rank):
    array = _as_numpy(value)
    rank = max(array.shape) if max_rank is None else max_rank
    return decompose(array, rank=rank)


def _dense_to_mpo(value, physical_dims, max_rank):
    factors = _decompose_dense(value, max_rank)
    if len(factors) != len(physical_dims):
        raise ValueError(
            f"Expected {len(physical_dims)} MPO sites; got {len(factors)}."
        )

    cores = []
    for factor, physical_dim in zip(factors, physical_dims):
        left_rank, fused_dim, right_rank = factor.shape
        if fused_dim != physical_dim**2:
            raise ValueError(
                f"Fused MPO dimension {fused_dim} does not equal "
                f"{physical_dim}**2."
            )
        core = factor.reshape(
            left_rank, physical_dim, physical_dim, right_rank
        )
        cores.append(core.transpose(0, 3, 1, 2))
    return MPO(cores)


def _dense_matrix_to_mpo(matrix, physical_dims, max_rank):
    dims = tuple(int(dim) for dim in physical_dims)
    dimension = int(np.prod(dims))
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"dense operator shape {matrix.shape} does not match {dims}."
        )

    nsites = len(dims)
    paired_order = [
        index
        for site in range(nsites)
        for index in (site, nsites + site)
    ]
    fused = matrix.reshape(*dims, *dims).transpose(paired_order).reshape(
        *(dim**2 for dim in dims)
    )
    return _dense_to_mpo(fused, dims, max_rank)


def _identity_mpo(physical_dims, dtype=np.complex128):
    return MPO([
        np.eye(dim, dtype=dtype).reshape(1, 1, dim, dim)
        for dim in physical_dims
    ])


def _diagonal_mpo(values, max_rank):
    cores = []
    for factor in _decompose_dense(values, max_rank):
        physical_dim = factor.shape[1]
        cores.append(np.einsum(
            "lpr,pq->lrpq",
            factor,
            np.eye(physical_dim, dtype=factor.dtype),
        ))
    return MPO(cores)


def _local_matrix_mpo(values, max_rank):
    """Factor a grid-local matrix field directly into an MPO."""
    values = _as_numpy(values)
    if values.ndim < 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError(
            "values must have shape (*grid, physical_dim, physical_dim)"
        )
    grid_ndim = values.ndim - 2
    physical_dim = values.shape[-1]
    coefficient_tensor = values.transpose(
        grid_ndim,
        grid_ndim + 1,
        *range(grid_ndim),
    ).reshape(physical_dim**2, *values.shape[:grid_ndim])
    factors = _decompose_dense(coefficient_tensor, max_rank)

    first = factors[0]
    electronic_core = first.reshape(
        first.shape[0],
        physical_dim,
        physical_dim,
        first.shape[2],
    ).transpose(0, 3, 1, 2)
    cores = [electronic_core]
    for factor in factors[1:]:
        cores.append(np.einsum(
            "lpr,pq->lrpq",
            factor,
            np.eye(factor.shape[1], dtype=factor.dtype),
        ))
    return MPO(cores)


def _append_mpo_site(mpo, core):
    return MPO([*mpo.factors, _as_numpy(core)])


def _prepend_mpo_site(mpo, core):
    return MPO([_as_numpy(core), *mpo.factors])


def _copy_tensor_train(tensor_train):
    return type(tensor_train)([factor.copy() for factor in tensor_train.factors])


def _tensor_train_shapes(tensor_train):
    return [factor.shape for factor in tensor_train.factors]


def _mpo_ranks(mpo):
    return [mpo.factors[0].shape[0], *[
        factor.shape[1] for factor in mpo.factors
    ]]


def _mpo_norm(mpo):
    environment = np.ones((1, 1), dtype=np.complex128)
    for factor in mpo.factors:
        environment = np.einsum(
            "ab,acij,bdij->cd",
            environment,
            factor.conj(),
            factor,
            optimize=True,
        )
    return float(np.sqrt(np.abs(environment[0, 0])))


def _matmul(left, right, max_rank):
    return left.matmul(right, chi_max=max_rank)


def _mpo_exponential(mpo, constant, scale, order, max_rank):
    return mpo.exponential(
        constant=constant,
        D=max_rank,
        scale=scale,
        order=order,
    )


def _cycle_step_counts(steps, output_every):
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if output_every <= 0:
        raise ValueError("output_every must be positive")

    full_cycles, remainder = divmod(steps, output_every)
    counts = [output_every] * full_cycles
    if remainder:
        counts.append(remainder)
    return counts


def gen_einsum_string(D, keyword="kinetic", dr=None, dnr=None):
    """
    General function to generate einsum strings.

    Parameters:
    D: int, number of dimensions
    keyword: str, "kinetic" or "projection"
    dr: int, number of reactive coordinates (for "projection")
    dnr: int, number of non-reactive coordinates (for "projection")

    Returns:
    str: einsum string
    """
    alphabet = list(string.ascii_lowercase)

    if keyword == "kinetic":
        # 'abxiyj,xi,yj->abxyij'
        # abxiyjzk,xi,yj,zk->abxyzijk
        if D > 10:
            raise ValueError('Dimension D = {} cannot be larger than 10.'.format(D))

        first_tensor_indices = []
        input_tensors = []

        first_tensor_indices.append(alphabet[0])  # 'a'
        first_tensor_indices.append(alphabet[1])  # 'b'

        for n in range(D):
            idx1 = alphabet[2 * n + 2]
            idx2 = alphabet[2 * n + 3]
            first_tensor_indices.append(idx1)
            first_tensor_indices.append(idx2)
            input_tensors.append(idx1 + idx2)

        output_indices = [alphabet[0], alphabet[1]]  # 'ab'

        #  (x, y, z, ...)
        for n in range(D):
            output_indices.append(alphabet[2 * n + 2])

        #  (i, j, k, ...)
        for n in range(D):
            output_indices.append(alphabet[2 * n + 3])

        first_tensor = "".join(first_tensor_indices)
        einsum_string = first_tensor

        for tensor in input_tensors:
            einsum_string += ',' + tensor

        finalstring = einsum_string + '->' + "".join(output_indices)

        return finalstring

    elif keyword == "projection":
        # sxyzq,xysb->bxyzq
        if dr is None or dnr is None:
            raise ValueError("For projection mode, both dr and dnr must be provided")

        letters = string.ascii_lowercase
        s, b = letters[0], letters[1]
        rc_letters = ''.join(letters[2:2 + dr])
        nrc_letters = ''.join(letters[2 + dr:2 + dr + dnr])

        psi_sub = f"{s}{rc_letters}{nrc_letters}"
        phi_sub = f"{rc_letters}{s}{b}"
        out_sub = f"{b}{rc_letters}{nrc_letters}"

        einsum_str = f"{psi_sub},{phi_sub}->{out_sub}"
        return einsum_str

    else:
        raise ValueError(f"Unknown keyword: {keyword}. Must be 'kinetic' or 'projection'")


class CGLDR:
    """
    Coarse-grained locally diabatic dynamics using matrix-product states.
    """

    def __init__(
        self,
        dvr,
        partition,
        *,
        state_ids=(0, 1, 2),
        tt_options=None,
        solver=None,
        to_geometry=None,
        expansion_modes=None,
        electronic_cache=None,
        nuclear_kinetic_mpo=None,
        kinetic_exponential_options=None,
        backend="numpy",
    ):
        """
        Initialize a coarse-grained LDR calculation.

        Parameters:
        ----------
        dvr : DVR
            Complete named nuclear product grid.
        partition : ElectronicPartition
            Assignment of DVR coordinates to explicitly sampled and locally
            expanded electronic-structure subspaces.
        state_ids : sequence of int, optional
            Electronic roots retained by CGLDR. For example, ``(1, 2)``
            selects the first two excited states while the electronic solver
            still computes every root through state 2.
        tt_options : dict, optional
            Tensor-train controls. ``max_rank`` sets the maximum bond
            dimension and defaults to 100.
        solver : electronic-structure solver, optional
            Configured solver used to generate the ab initio data. A
            :class:`~pyqed.qchem.mcscf.casci.CASCI` instance is accepted
            directly and converted to its geometry scanner. Other solvers must
            be callable or provide ``point(geometry)``, and must provide
            ``overlap(left, right)`` unless their point results do.
        to_geometry : callable, optional
            Maps a name-to-value coordinate dictionary to the molecular
            geometry accepted by ``solver``.
        expansion_modes : array-like, optional
            Cartesian displacement vectors for the expanded coordinates.
            Required when analytical solver derivatives are Cartesian rather
            than already expressed in the expanded-coordinate basis.
        electronic_cache : path-like, optional
            Portable NPZ cache used by automatic electronic-data preparation.
        nuclear_kinetic_mpo : MPO, optional
            Nuclear kinetic-energy operator over the reordered DVR coordinates.
            When supplied, CGLDR uses the LDR overlap-projected kinetic
            Hamiltonian built from this MPO instead of the default product-DVR
            kinetic splitting.
        kinetic_exponential_options : dict, optional
            Taylor/scaling options used when exponentiating an overlap-projected
            nuclear kinetic MPO. Accepted keys are ``order`` and ``scale``.
        backend : {"numpy", "torch"}, optional
            Array backend for stored electronic data. NumPy is the default and
            does not require Torch; the Torch backend is reserved for explicit
            tensor/autodiff workflows.

        Raises:
        -------
        ValueError
            If the DVR or electronic partition is invalid.
        """
        import time

        if tt_options is None:
            tt_options = {"max_rank": 100}
        else:
            tt_options = dict(tt_options)
            tt_options.setdefault("max_rank", 100)
        unknown_tt_options = set(tt_options) - {"max_rank"}
        if unknown_tt_options:
            names = ", ".join(sorted(unknown_tt_options))
            raise ValueError(f"Unknown tt_options: {names}")
        max_rank = tt_options["max_rank"]
        if max_rank is not None:
            if not isinstance(max_rank, (int, np.integer)) or max_rank <= 0:
                raise ValueError(
                    "tt_options['max_rank'] must be a positive integer or None"
                )
            tt_options["max_rank"] = int(max_rank)
        if not isinstance(dvr, DVR):
            raise TypeError("dvr must be a pyqed.dvr.DVR")
        if not isinstance(partition, ElectronicPartition):
            raise TypeError("partition must be an ElectronicPartition")
        backend = str(backend).lower()
        if backend not in {"numpy", "torch"}:
            raise ValueError("backend must be 'numpy' or 'torch'")
        if backend == "torch":
            require_torch_backend()
        try:
            state_ids = tuple(state_ids)
        except TypeError as exc:
            raise TypeError("state_ids must be a sequence of integers") from exc
        if not state_ids:
            raise ValueError("state_ids cannot be empty")
        if any(
            not isinstance(state_id, (int, np.integer)) or state_id < 0
            for state_id in state_ids
        ):
            raise ValueError("state_ids must contain non-negative integers")
        state_ids = tuple(int(state_id) for state_id in state_ids)
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("state_ids must be unique")

        sampled_axes, expanded_axes = partition.resolve(dvr)
        self.dvr = dvr
        self.partition = partition
        self.original_ndim = dvr.ndim
        self.dr = len(sampled_axes)
        self.dnr = len(expanded_axes)
        self.nsampled = self.dr
        self.nexpanded = self.dnr
        self.ndim = dvr.ndim
        self.sampled_axes = sampled_axes
        self.expanded_axes = expanded_axes
        self.sampled_names = partition.sampled
        self.expanded_names = partition.expanded
        self.reorder_indices = list(sampled_axes + expanded_axes)
        self.inverse_reorder_indices = [0] * self.original_ndim
        for new_idx, original_idx in enumerate(self.reorder_indices):
            self.inverse_reorder_indices[original_idx] = new_idx

        self.axes = tuple(dvr.axes[index] for index in self.reorder_indices)
        self.coordinate_names = tuple(
            dvr.names[index] for index in self.reorder_indices
        )
        self.domains = tuple(
            dvr.domains[index] for index in self.reorder_indices
        )
        self.npts = tuple(dvr.npts[index] for index in self.reorder_indices)
        self.mass = tuple(dvr.mass[index] for index in self.reorder_indices)
        self.state_ids = state_ids
        self.nstates = len(state_ids)
        self.nsite = self.dr + self.dnr + 1
        self.tt_options = tt_options
        self.backend = backend
        self.solver = solver
        self.to_geometry = to_geometry
        self.expansion_modes = (
            None
            if expansion_modes is None
            else np.asarray(expansion_modes, dtype=float)
        )
        if self.expansion_modes is not None:
            if self.expansion_modes.ndim not in (2, 3):
                raise ValueError(
                    "expansion_modes must have shape (nexpanded, 3*natom) "
                    "or (nexpanded, natom, 3)"
                )
            if self.expansion_modes.shape[0] != self.nexpanded:
                raise ValueError(
                    "expansion_modes must contain one mode per expanded coordinate"
                )
            if (
                self.expansion_modes.ndim == 2
                and self.expansion_modes.shape[1] % 3 != 0
            ) or (
                self.expansion_modes.ndim == 3
                and self.expansion_modes.shape[2] != 3
            ):
                raise ValueError(
                    "expansion_modes must contain Cartesian xyz components"
                )
            if not np.all(np.isfinite(self.expansion_modes)):
                raise ValueError("expansion_modes must be finite")
        self.electronic_cache = (
            None if electronic_cache is None else Path(electronic_cache)
        )
        if kinetic_exponential_options is None:
            kinetic_exponential_options = {}
        else:
            kinetic_exponential_options = dict(kinetic_exponential_options)
        unknown_kinetic_options = set(kinetic_exponential_options) - {
            "order",
            "scale",
            "dense_dimension_limit",
        }
        if unknown_kinetic_options:
            names = ", ".join(sorted(unknown_kinetic_options))
            raise ValueError(f"Unknown kinetic_exponential_options: {names}")
        kinetic_exponential_options.setdefault("order", 8)
        kinetic_exponential_options.setdefault("scale", 1)
        kinetic_exponential_options.setdefault("dense_dimension_limit", 2048)
        for name in ("order", "scale", "dense_dimension_limit"):
            value = kinetic_exponential_options[name]
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(
                    f"kinetic_exponential_options['{name}'] must be a "
                    "non-negative integer"
                )
            kinetic_exponential_options[name] = int(value)
        self.kinetic_exponential_options = kinetic_exponential_options

        self.current_time = time.strftime("%Y%m%d_%H%M%S")
        self.output_folder = "."

        self.states = None
        self.populations = None
        self.coordinate_expectations = None
        self.coordinate_variances = None
        self.times = None
        self.steps = None
        self.initial_time = 0.0
        self.time_unit = "au"

        self.x = tuple(np.asarray(axis.x) for axis in self.axes)
        self.nx = tuple(len(x) for x in self.x)
        self.dims = [self.nstates, *self.nx]
        self.dx = tuple(
            dvr.dx[index] for index in self.reorder_indices
        )
        self.dv = np.prod(self.dx)
        self.q_diff = tuple(
            self.x[self.dr + index] - center
            for index, center in enumerate(self.partition.center)
        )

        """Reset all operators and matrices"""
        self.H_matrices = None
        self.adiabatic_states = None
        self.A = None
        self.coarse_hamiltonian_propagator = None
        self.hamiltonian_mpo = None
        self.kinetic_propagator = None
        self.half_kinetic_propagator = None
        self.e0 = None
        self.surface_propagator = None
        self.half_surface_propagator = None
        self.exp2T = None
        self.expT = None
        self.split_step_left = None
        self.split_step_right = None
        self.exp_K = None
        self.propagator = None
        self.apes = None
        self._propagator_time_step = None
        self.nuclear_kinetic_mpo = None
        self.projected_kinetic_mpo = None
        self.projected_kinetic_dense = None
        self.overlap_projection_mpo = None
        self.integrator = None
        self.integrator_history = None
        self.tdvp_truncation_errors = None
        self.bond_dimensions = None
        if nuclear_kinetic_mpo is not None:
            self.set_nuclear_kinetic_mpo(nuclear_kinetic_mpo)

    def _invalidate_propagators(self):
        self.coarse_hamiltonian_propagator = None
        self.hamiltonian_mpo = None
        self.kinetic_propagator = None
        self.half_kinetic_propagator = None
        self.surface_propagator = None
        self.half_surface_propagator = None
        self.expT = None
        self.exp2T = None
        self.split_step_left = None
        self.split_step_right = None
        self._propagator_time_step = None
        self.projected_kinetic_mpo = None
        self.projected_kinetic_dense = None
        self.overlap_projection_mpo = None

    def set_nuclear_kinetic_mpo(self, mpo):
        """Attach a nuclear kinetic-energy MPO over the reordered DVR axes."""
        if not isinstance(mpo, MPO):
            raise TypeError("nuclear_kinetic_mpo must be a pyqed.mps.mps.MPO")
        dims = tuple(int(dim) for dim in mpo.dims)
        if dims != self.nx:
            raise ValueError(
                "nuclear_kinetic_mpo dimensions must match reordered DVR "
                f"grid {self.nx}; got {dims}."
            )
        self.nuclear_kinetic_mpo = MPO([
            np.asarray(core, dtype=np.complex128).copy()
            for core in mpo.factors
        ])
        self._invalidate_propagators()
        return self

    def prepare_electronic_data(self, *, force=False):
        """Generate or load every electronic quantity required by CGLDR."""
        if self.H_matrices is not None and not force:
            return self

        source_path = None
        if self.electronic_cache is not None:
            if self.electronic_cache.exists() and not force:
                source_path = self.electronic_cache
        if source_path is not None:
            return self.load_electronic_data(source_path)

        if self.solver is None:
            raise RuntimeError(
                "Set solver=..., provide an existing electronic_cache, or "
                "call set_electronic_data() before building the propagator."
            )
        if self.to_geometry is None or not callable(self.to_geometry):
            raise ValueError(
                "Automatic electronic calculations require "
                "to_geometry(coordinates)."
            )
        if self.nexpanded and self.expansion_modes is None:
            raise NotImplementedError(
                "Expanded internal coordinates are not supported. Provide "
                "linear Cartesian expansion_modes, or attach precomputed "
                "electronic data."
            )
        solver = self.solver
        if hasattr(solver, "as_scanner"):
            try:
                solver = solver.as_scanner(nstates=max(self.state_ids) + 1)
            except TypeError:
                solver = solver.as_scanner()
        self._active_electronic_solver = solver

        grid_shape = tuple(self.nx[:self.dr])
        reference_objects = np.empty(grid_shape, dtype=object)
        energies = np.empty((*grid_shape, self.nstates), dtype=float)
        if self.dnr:
            gradients = np.empty(
                (*grid_shape, self.dnr, self.nstates, self.nstates),
                dtype=complex,
            )
            hessians = np.empty(
                (
                    *grid_shape,
                    self.dnr,
                    self.dnr,
                    self.nstates,
                    self.nstates,
                ),
                dtype=complex,
            )
        for index in np.ndindex(*grid_shape):
            reordered = [
                self.x[axis][index[axis]] for axis in range(self.dr)
            ] + list(self.partition.center)
            original = self._reorder_coords_to_original(reordered)
            point_energies, point_object = self._electronic_point(original)
            energies[index] = point_energies
            reference_objects[index] = point_object
            if self.dnr:
                gradient, hessian = self._electronic_derivatives(point_object)
                gradients[index] = gradient
                hessians[index] = hessian

        ngrid = int(np.prod(grid_shape))
        flat_objects = reference_objects.reshape(ngrid)
        overlap_blocks = np.empty(
            (ngrid, self.nstates, ngrid, self.nstates),
            dtype=complex,
        )
        for bra in range(ngrid):
            overlap_blocks[bra, :, bra, :] = np.eye(self.nstates)
            for ket in range(bra + 1, ngrid):
                overlap = self._electronic_overlap(
                    flat_objects[bra],
                    flat_objects[ket],
                )
                overlap_blocks[bra, :, ket, :] = overlap
                overlap_blocks[ket, :, bra, :] = overlap.conj().T
        overlaps = overlap_blocks.reshape(
            *grid_shape,
            self.nstates,
            *grid_shape,
            self.nstates,
        )

        if self.dnr:
            data = CGLDRElectronicData(
                energies=energies,
                overlaps=overlaps,
                hamiltonian_gradients=gradients,
                hamiltonian_hessians=hessians,
                reactive_grids=tuple(self.x[:self.dr]),
                metadata=self._electronic_metadata(),
            )
        else:
            data = CGLDRElectronicData(
                energies=energies,
                overlaps=overlaps,
                reactive_grids=tuple(self.x[:self.dr]),
                metadata=self._electronic_metadata(),
            )

        self.set_electronic_data(data)
        if self.electronic_cache is not None:
            self.electronic_cache.parent.mkdir(parents=True, exist_ok=True)
            data.to_npz(self.electronic_cache)
        return self

    def save_electronic_data(self, filename):
        """Save the internally prepared electronic data as a portable NPZ."""
        if not hasattr(self, "electronic_data"):
            raise RuntimeError("No electronic data have been prepared.")
        self.electronic_data.to_npz(filename)

    def _electronic_metadata(self):
        return {
            "solver": type(self.solver).__name__,
            "sampled_coordinates": list(self.partition.sampled),
            "expanded_coordinates": list(self.partition.expanded),
            "expansion_center": list(self.partition.center),
            "state_ids": list(self.state_ids),
            "derivative_source": "analytic_vibronic_couplings",
        }

    def _electronic_derivatives(self, point):
        method = getattr(point, "vibronic_couplings", None)
        if method is None:
            raise NotImplementedError(
                "The electronic solver does not provide analytical "
                "vibronic_couplings(). Finite-difference derivatives are not "
                "supported by automatic CGLDR preparation."
            )

        kwargs = {"state_ids": self.state_ids}
        if self.expansion_modes is not None:
            kwargs["modes"] = self.expansion_modes
        first, second = method(**kwargs)
        first = np.asarray(first, dtype=complex)
        second = np.asarray(second, dtype=complex)
        expected_first = (self.nstates, self.nstates, self.nexpanded)
        expected_second = (
            self.nstates,
            self.nstates,
            self.nexpanded,
            self.nexpanded,
        )
        if first.shape != expected_first or second.shape != expected_second:
            hint = (
                " Provide expansion_modes to project Cartesian derivatives."
                if self.expansion_modes is None
                else ""
            )
            raise ValueError(
                "vibronic_couplings returned shapes "
                f"{first.shape} and {second.shape}; expected "
                f"{expected_first} and {expected_second}.{hint}"
            )
        return (
            np.moveaxis(first, -1, 0),
            np.moveaxis(second, (-2, -1), (0, 1)),
        )

    def _electronic_point(self, coordinates):
        geometry = self.to_geometry(self.dvr.values(coordinates))
        solver = self._active_electronic_solver

        if hasattr(solver, "point"):
            result = solver.point(geometry)
        elif callable(solver):
            result = solver(geometry)
        else:
            raise TypeError(
                "solver must be callable or provide point(geometry)."
            )

        if isinstance(result, dict):
            point_object = result.get("object", result)
            point_energies = result.get("energies")
        elif isinstance(result, tuple) and len(result) == 2:
            point_energies, point_object = result
        else:
            point_object = result
            point_energies = getattr(
                result,
                "e_tot",
                getattr(result, "energies", None),
            )
        if point_energies is None:
            raise ValueError(
                "Electronic point result must provide energies or e_tot."
            )
        point_energies = np.atleast_1d(
            np.asarray(point_energies, dtype=float)
        )
        required_roots = max(self.state_ids) + 1
        if point_energies.size < required_roots:
            raise ValueError(
                f"Electronic solver returned {point_energies.size} states; "
                f"root {max(self.state_ids)} is required."
            )
        return point_energies[np.asarray(self.state_ids)], point_object

    def _electronic_overlap(self, left, right):
        solvers = (
            getattr(self, "_active_electronic_solver", None),
            self.solver,
        )
        value = None
        for solver in solvers:
            if solver is not None:
                overlap_method = getattr(solver, "overlap", None)
                if overlap_method is not None:
                    value = overlap_method(left, right)
                    break
        if value is None and hasattr(left, "wavefunction_overlap"):
            value = left.wavefunction_overlap(right)
        if value is None and hasattr(left, "overlap"):
            value = left.overlap(right)
        if value is None:
            from pyqed.qchem.mcscf.casci import overlap

            value = overlap(left, right)
        value = np.asarray(value, dtype=complex)
        required_roots = max(self.state_ids) + 1
        expected = (self.nstates, self.nstates)
        if value.ndim != 2 or (
            value.shape[0] < required_roots
            or value.shape[1] < required_roots
        ):
            raise ValueError(
                f"Electronic overlap shape {value.shape} cannot supply {expected}"
            )
        return value[np.ix_(self.state_ids, self.state_ids)]

    def set_electronic_data(self, data, *, tolerance=1e-7):
        """Attach validated ab initio energies, overlaps, and derivatives.

        The derivative tensors are derivatives of the electronic Hamiltonian
        with respect to the non-reactive coordinates at
        :attr:`partition.center`. They must already be represented in the same
            locally adiabatic basis used by ``overlaps``.
        """
        if isinstance(data, (str, os.PathLike)):
            data = CGLDRElectronicData.from_npz(data)
        elif isinstance(data, dict):
            data = CGLDRElectronicData(**data)
        if not isinstance(data, CGLDRElectronicData):
            raise TypeError(
                "data must be CGLDRElectronicData, a matching dictionary, "
                "or an NPZ filename"
            )
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        grid_shape = tuple(self.nx[:self.dr])
        energies = np.asarray(data.energies)
        expected_energies = (*grid_shape, self.nstates)
        if energies.shape != expected_energies:
            raise ValueError(
                f"energies shape {energies.shape} != {expected_energies}"
            )
        if np.iscomplexobj(energies) and not np.allclose(
            energies.imag, 0.0, atol=tolerance
        ):
            raise ValueError("adiabatic energies must be real")
        energies = np.asarray(energies.real, dtype=float)
        if not np.all(np.isfinite(energies)):
            raise ValueError("energies contain non-finite values")

        if data.reactive_grids is not None:
            if len(data.reactive_grids) != self.dr:
                raise ValueError(
                    "reactive_grids must contain one array per reactive coordinate"
                )
            for axis, (provided, expected) in enumerate(
                zip(data.reactive_grids, self.x[:self.dr])
            ):
                provided = np.asarray(provided)
                expected = np.asarray(expected)
                if provided.shape != expected.shape or not np.allclose(
                    provided, expected, rtol=0.0, atol=tolerance
                ):
                    raise ValueError(
                        f"reactive grid {axis} does not match the solver DVR grid"
                    )

        overlaps = np.asarray(data.overlaps)
        expected_overlaps = (
            *grid_shape,
            self.nstates,
            *grid_shape,
            self.nstates,
        )
        if overlaps.shape != expected_overlaps:
            raise ValueError(
                f"overlaps shape {overlaps.shape} != {expected_overlaps}"
            )
        if not np.all(np.isfinite(overlaps)):
            raise ValueError("overlaps contain non-finite values")

        ngrid = int(np.prod(grid_shape))
        flat_overlaps = overlaps.reshape(
            ngrid, self.nstates, ngrid, self.nstates
        )
        adjoint_overlaps = flat_overlaps.transpose(2, 3, 0, 1).conj()
        if not np.allclose(
            flat_overlaps, adjoint_overlaps, atol=tolerance, rtol=tolerance
        ):
            raise ValueError(
                "overlaps must satisfy S(a,b) = S(b,a)†"
            )
        identity = np.eye(self.nstates)
        for point in range(ngrid):
            if not np.allclose(
                flat_overlaps[point, :, point, :],
                identity,
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError(
                    f"self-overlap at reactive grid point {point} is not identity"
                )

        if data.basis_transforms is not None:
            transforms = np.asarray(data.basis_transforms)
            if (
                transforms.ndim != self.dr + 2
                or transforms.shape[:self.dr] != grid_shape
                or transforms.shape[-1] != self.nstates
            ):
                raise ValueError(
                    "basis_transforms must have shape "
                    f"(*sampled, nraw, {self.nstates})"
                )
            if not np.all(np.isfinite(transforms)):
                raise ValueError("basis_transforms contain non-finite values")
            if data.metric_eigenvalues is None:
                raise ValueError(
                    "metric_eigenvalues are required with basis_transforms"
                )
            metric_eigenvalues = np.asarray(data.metric_eigenvalues)
            if metric_eigenvalues.shape != transforms.shape[:-1]:
                raise ValueError(
                    "metric_eigenvalues must have shape (*sampled, nraw)"
                )
            if (
                not np.all(np.isfinite(metric_eigenvalues))
                or np.any(metric_eigenvalues < -tolerance)
            ):
                raise ValueError(
                    "metric_eigenvalues must be finite and nonnegative"
                )
        elif data.metric_eigenvalues is not None:
            raise ValueError(
                "basis_transforms are required with metric_eigenvalues"
            )

        separable = data.separable_hamiltonian
        separable_operators = None
        separable_factors = None
        if separable is not None:
            if not self.dnr:
                raise ValueError(
                    "separable_hamiltonian requires expanded coordinates"
                )
            separable_operators = np.asarray(
                separable.operators,
                dtype=complex,
            )
            expected_operator_ndim = self.dr + 3
            if (
                separable_operators.ndim != expected_operator_ndim
                or separable_operators.shape[:self.dr] != grid_shape
                or separable_operators.shape[-2:]
                != (self.nstates, self.nstates)
            ):
                raise ValueError(
                    "separable operators must have shape "
                    f"(*sampled, nterms, {self.nstates}, {self.nstates})"
                )
            nterms = separable_operators.shape[-3]
            if len(separable.factors) != self.dnr:
                raise ValueError(
                    "separable_hamiltonian must contain one factor array per "
                    "expanded coordinate"
                )
            separable_factors = []
            for axis, (factor, npoints) in enumerate(
                zip(separable.factors, self.nx[self.dr:])
            ):
                factor = np.asarray(factor)
                expected_factor_shape = (nterms, npoints)
                if factor.shape != expected_factor_shape:
                    raise ValueError(
                        f"separable factor {axis} shape {factor.shape} "
                        f"!= {expected_factor_shape}"
                    )
                if np.iscomplexobj(factor) and not np.allclose(
                    factor.imag,
                    0.0,
                    atol=tolerance,
                    rtol=tolerance,
                ):
                    raise ValueError(
                        "separable coordinate factors must be real"
                    )
                separable_factors.append(np.asarray(factor.real, dtype=float))
            if not np.all(np.isfinite(separable_operators)):
                raise ValueError(
                    "separable Hamiltonian operators contain non-finite values"
                )
            if not all(np.all(np.isfinite(factor))
                       for factor in separable_factors):
                raise ValueError(
                    "separable Hamiltonian factors contain non-finite values"
                )
            if not np.allclose(
                separable_operators,
                separable_operators.swapaxes(-1, -2).conj(),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError(
                    "separable Hamiltonian operators must be Hermitian"
                )

        if data.expanded_grids is not None:
            if len(data.expanded_grids) != self.dnr:
                raise ValueError(
                    "expanded_grids must contain one array per expanded "
                    "coordinate"
                )
            for axis, (provided, expected) in enumerate(
                zip(data.expanded_grids, self.x[self.dr:])
            ):
                provided = np.asarray(provided)
                expected = np.asarray(expected)
                if provided.shape != expected.shape or not np.allclose(
                    provided,
                    expected,
                    rtol=0.0,
                    atol=tolerance,
                ):
                    raise ValueError(
                        f"expanded grid {axis} does not match the solver DVR grid"
                    )

        derivative_shape = (*grid_shape, self.dnr, self.nstates, self.nstates)
        hessian_shape = (
            *grid_shape,
            self.dnr,
            self.dnr,
            self.nstates,
            self.nstates,
        )
        if self.dnr and separable is None:
            if data.hamiltonian_gradients is None:
                raise ValueError(
                    "hamiltonian_gradients are required for expanded coordinates"
                )
            if data.hamiltonian_hessians is None:
                raise ValueError(
                    "hamiltonian_hessians are required for expanded coordinates"
                )
            gradients = np.asarray(data.hamiltonian_gradients)
            hessians = np.asarray(data.hamiltonian_hessians)
            if gradients.shape != derivative_shape:
                raise ValueError(
                    f"hamiltonian_gradients shape {gradients.shape} "
                    f"!= {derivative_shape}"
                )
            if hessians.shape != hessian_shape:
                raise ValueError(
                    f"hamiltonian_hessians shape {hessians.shape} "
                    f"!= {hessian_shape}"
                )
            if not np.all(np.isfinite(gradients)):
                raise ValueError("hamiltonian_gradients contain non-finite values")
            if not np.all(np.isfinite(hessians)):
                raise ValueError("hamiltonian_hessians contain non-finite values")
            if not np.allclose(
                gradients,
                gradients.swapaxes(-1, -2).conj(),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError("hamiltonian_gradients must be Hermitian")
            if not np.allclose(
                hessians,
                hessians.swapaxes(-1, -2).conj(),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError("hamiltonian_hessians must be Hermitian")
            if not np.allclose(
                hessians,
                hessians.swapaxes(self.dr, self.dr + 1),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError(
                    "hamiltonian_hessians must be symmetric in coarse coordinates"
                )
        else:
            gradients = np.empty((*grid_shape, 0, self.nstates, self.nstates))
            hessians = np.empty(
                (*grid_shape, 0, 0, self.nstates, self.nstates)
            )

        local_h0 = np.zeros(
            (*grid_shape, self.nstates, self.nstates),
            dtype=float,
        )
        state_indices = np.arange(self.nstates)
        local_h0[..., state_indices, state_indices] = energies
        expansion = []
        if separable is None:
            expansion.append(self._local_grid_operator(local_h0))
            for coordinate in range(self.dnr):
                expansion.append(
                    self._local_grid_operator(gradients[..., coordinate, :, :])
                )
            for coordinate in range(self.dnr):
                expansion.append(
                    0.5 * self._local_grid_operator(
                        hessians[..., coordinate, coordinate, :, :]
                    )
                )
            for first in range(self.dnr - 1):
                for second in range(first + 1, self.dnr):
                    expansion.append(
                        self._local_grid_operator(
                            hessians[..., first, second, :, :]
                        )
                    )
        else:
            for term in range(separable_operators.shape[-3]):
                expansion.append(self._local_grid_operator(
                    separable_operators[..., term, :, :]
                ))

        overlap_permutation = [self.dr, 2 * self.dr + 1]
        for axis in range(self.dr):
            overlap_permutation.extend((axis, self.dr + 1 + axis))

        surface_energies = (
            np.zeros_like(energies)
            if separable is not None
            else energies
        )
        if self.backend == "torch":
            torch_backend = require_torch_backend()
            self.e0 = torch_backend.as_tensor(
                np.moveaxis(surface_energies, -1, 0),
                dtype=torch_backend.float64,
            )
            self.A = torch_backend.as_tensor(
                overlaps.transpose(overlap_permutation),
                dtype=torch_backend.complex128,
            )
            self.H_matrices = [
                torch_backend.as_tensor(operator) for operator in expansion
            ]
        else:
            self.e0 = np.asarray(
                np.moveaxis(surface_energies, -1, 0),
                dtype=np.float64,
            )
            self.A = np.asarray(
                overlaps.transpose(overlap_permutation),
                dtype=np.complex128,
            )
            self.H_matrices = [
                np.asarray(operator) for operator in expansion
            ]
        self.separable_factors = separable_factors
        self.adiabatic_states = None
        self.apes = energies.copy()
        self.electronic_data = data
        self._invalidate_propagators()
        return self

    def load_electronic_data(self, filename, *, tolerance=1e-7):
        """Load an NPZ electronic-data archive and attach it to the solver."""
        return self.set_electronic_data(filename, tolerance=tolerance)

    def _local_grid_operator(self, local_matrices):
        """Embed grid-local electronic matrices into a DVR-grid operator."""
        grid_shape = tuple(self.nx[:self.dr])
        expected = (*grid_shape, self.nstates, self.nstates)
        local_matrices = np.asarray(local_matrices)
        if local_matrices.shape != expected:
            raise ValueError(
                f"local matrix shape {local_matrices.shape} != {expected}"
            )

        ngrid = int(np.prod(grid_shape))
        local_flat = local_matrices.reshape(
            ngrid, self.nstates, self.nstates
        )
        operator = np.zeros(
            (self.nstates, self.nstates, ngrid, ngrid),
            dtype=local_matrices.dtype,
        )
        for point, matrix in enumerate(local_flat):
            operator[:, :, point, point] = matrix
        return operator.reshape(
            self.nstates,
            self.nstates,
            *grid_shape,
            *grid_shape,
        )

    def _reorder_coords_to_original(self, reordered_coords):
        """
        将重排后的坐标列表重新映射回原始顺序

        Parameters
        ----------
        reordered_coords : list
            重排后顺序的坐标列表

        Returns
        -------
        original_coords : list
            原始顺序的坐标列表
        """
        if len(reordered_coords) != self.original_ndim:
            raise ValueError(
                f"Expected {self.original_ndim} coordinates, got {len(reordered_coords)}")

        original_coords = [0] * self.original_ndim
        for reordered_idx, coord_val in enumerate(reordered_coords):
            original_idx = self.reorder_indices[reordered_idx]
            original_coords[original_idx] = coord_val

        return original_coords

    def build_overlap(self, U):
        """
        修改后的函数，输出维度顺序为 zy + A1B1 + A2B2 + ...
        """
        # 获取张量U的维度信息
        dims = U.shape
        ndim = len(dims)
        if ndim < 2:
            raise ValueError(f'输入张量维度不足，至少需要电子自旋和电子态两个维度')
        nuclear_dims = ndim - 2
        if nuclear_dims > 10:
            raise ValueError(f'核坐标维度 {nuclear_dims} 不能大于10')
        alphabet = list(string.ascii_lowercase)

        input_indices = "".join(alphabet[:nuclear_dims]) + "x" + "z"
        conj_indices = "".join(alphabet[nuclear_dims:2 * nuclear_dims]) + "x" + "y"

        # 修改输出顺序：zy 在前，然后是 A1B1, A2B2, ...
        output_pairs = []
        for i in range(nuclear_dims):
            output_pairs.append(alphabet[i] + alphabet[nuclear_dims + i])

        output_indices = "z" + "y" + "".join(output_pairs)
        einsum_str = f"{input_indices},{conj_indices}->{output_indices}"
        U_array = _as_numpy(U)
        result = np.einsum(einsum_str, U_array, U_array.conj(), optimize=True)
        return result

    def get_hamiltonian_matrices(self, H_val_func):
        """
        This step can be replaced by electronic structure calculation.

        Parameters
        ----------
        H_val_func : callable
            其中 coord_list 是所有坐标的列表 [x, y, z, q, ...]

        Returns
        -------
        H_matrices : list

        phi : torch.Tensor
            绝热态本征矢量
        """
        require_torch_backend()

        def buildH_general(reordered_coord_values, q0_values):
            """通用哈密顿量构建函数"""

            # 将q0转换为可求导的tensor
            q0_tensors = []
            for q_val in q0_values:
                if isinstance(q_val, torch.Tensor):
                    q0_tensors.append(q_val.clone().detach().requires_grad_(True))
                else:
                    q0_tensors.append(torch.tensor(q_val, dtype=torch.float64, requires_grad=True))

            # 构建重排后的完整坐标列表：反应坐标 + 非反应坐标
            reordered_full_coords = list(reordered_coord_values) + q0_tensors

            # 将坐标重新映射回原始顺序后传递给H_val_func
            def H_wrapper(var_tensor):
                """包装函数，将tensor变量转换为原始顺序的坐标列表"""
                reordered_coords = list(reordered_coord_values) + [var_tensor[i] for i in
                                                                   range(len(var_tensor))]
                original_coords = self._reorder_coords_to_original(reordered_coords)
                return H_val_func(original_coords)

            # 计算哈密顿量矩阵 - 使用原始顺序的坐标
            original_full_coords = self._reorder_coords_to_original(reordered_full_coords)
            h0 = H_val_func(original_full_coords)

            if self.dnr == 0:
                # 没有非反应坐标，只返回基本哈密顿量
                return [h0] + [torch.zeros_like(h0)] * 6

            # 非反应坐标变量
            var = torch.stack(q0_tensors)

            # 一阶导数
            J1 = jacobian(H_wrapper, var, create_graph=True)
            H_derivs_1 = []
            for i in range(self.dnr):
                H_derivs_1.append(J1[..., i].clone())

            # 二阶导数函数
            def get_derivative_fn(coord_idx):
                def deriv_fn(var_tensor):
                    return jacobian(H_wrapper, var_tensor, create_graph=True)[..., coord_idx]

                return deriv_fn

            # 计算二阶导数
            H_derivs_2 = []
            H_cross_derivs = []

            for i in range(self.dnr):
                deriv_fn = get_derivative_fn(i)
                J2 = jacobian(deriv_fn, var)

                # 对角二阶导数
                H_derivs_2.append(J2[..., i].clone())

                # 交叉导数（只计算上三角部分）
                for j in range(i + 1, self.dnr):
                    H_cross_derivs.append(J2[..., j].clone())

            return h0, H_derivs_1, H_derivs_2, H_cross_derivs

        # 获取反应坐标网格
        reactive_coords = self.x[:self.dr]  # 假设self.x包含所有坐标网格
        grid_shapes = [len(coord) for coord in reactive_coords]

        # 初始化结果张量
        total_derivatives = int(
            1 + 2 * self.dnr + self.dnr * (self.dnr - 1) / 2)  # H0, H1, H2, H5, H6, H3
        H_matrices = []
        for _ in range(total_derivatives):  # H0, H1, H2, H3, H4, H5, H6 格式
            shape = tuple(grid_shapes) + (self.nstates, self.nstates)
            H_matrices.append(torch.zeros(shape, dtype=torch.float64))

        e0 = torch.zeros(tuple(grid_shapes) + (self.nstates,), dtype=torch.float64)
        phi = torch.zeros(tuple(grid_shapes) + (self.nstates, self.nstates), dtype=torch.float64)

        print('Building Hamiltonian matrices...')
        from itertools import product
        for indices in product(*[range(len(coord)) for coord in reactive_coords]):
            # 获取当前网格点的坐标值
            current_reordered_coords = [reactive_coords[i][indices[i]] for i in range(self.dr)]

            # 计算哈密顿量及其导数
            result = buildH_general(
                current_reordered_coords,
                self.partition.center,
            )

            if self.dnr == 0:
                # 无非反应坐标情况
                H_matrices[0][indices] = result[0]
            else:
                # 有非反应坐标情况
                h0, h_derivs_1, h_derivs_2, h_cross_derivs = result

                H_matrices[0][indices] = h0  # H0

                # 一阶导数 - H1, H2, ...
                for i in range(self.dnr):
                    H_matrices[i + 1][indices] = h_derivs_1[i]

                # 二阶导数 - H5, H6, ... (从索引1+self.dnr开始)
                for i in range(self.dnr):
                    H_matrices[1 + self.dnr + i][indices] = 0.5 * h_derivs_2[i]

                # 交叉导数 -  (从索引1+2*self.dnr开始)
                cross_idx = 1 + 2 * self.dnr
                for i, h_cross in enumerate(h_cross_derivs):
                    H_matrices[cross_idx + i][indices] = h_cross
            # 计算本征值和本征矢量
            eigenvals, eigenvecs = torch.linalg.eigh(H_matrices[0][indices])
            e0[indices] = eigenvals
            phi[indices] = eigenvecs

        # 转换为矩阵元

        Ixy = torch.eye(torch.prod(torch.tensor(grid_shapes))).reshape(*grid_shapes, *grid_shapes)

        # 转换各个哈密顿量矩阵到绝热表象
        H_adiabatic = []
        for H_mat in H_matrices:
            # 执行绝热变换
            n = len(grid_shapes)
            p = list(string.ascii_lowercase[4:])
            #efab,efac,efcd,efgh->bdefgh
            # 构建清晰的 einsum 字符串
            input1 = "".join(p[:n]) + "ab"
            input2 = "".join(p[:n]) + "ac"
            input3 = "".join(p[:n]) + "cd"
            input4 = "".join(p[:n]) + "".join(p[n:2 * n])
            output = "bd" + "".join(p[:n]) + "".join(p[n:2 * n])

            einsum_str = f"{input1},{input2},{input3},{input4}->{output}"
            H_ad = torch.einsum(einsum_str, phi.conj(), H_mat, phi, Ixy)

            H_adiabatic.append(H_ad)

        self.e0 = e0.permute(-1, *range(len(e0.shape) - 1))

        self.adiabatic_states = phi
        self.H_matrices = H_adiabatic
        self._invalidate_propagators()

        if self.A is None:
            print('Building electronic overlap...')
            self.A = self.build_overlap(phi)

        return H_adiabatic, phi

    def _sampled_kinetic_hamiltonian(self):
        """Return the overlap-projected kinetic matrix on sampled sites."""
        sampled_shape = tuple(self.npts[:self.dr])
        sampled_size = int(np.prod(sampled_shape))
        identities = [np.eye(size) for size in sampled_shape]
        sampled_kinetic = np.zeros(
            (sampled_size, sampled_size),
            dtype=np.complex128,
        )
        for active_axis in range(self.dr):
            factors = list(identities)
            factors[active_axis] = np.asarray(
                self.axes[active_axis].t(),
                dtype=np.complex128,
            )
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            sampled_kinetic += term

        overlap = _as_numpy(self.A)
        grouped_order = [
            0,
            1,
            *(2 + 2 * axis for axis in range(self.dr)),
            *(3 + 2 * axis for axis in range(self.dr)),
        ]
        overlap = overlap.transpose(grouped_order)
        projected = overlap * sampled_kinetic.reshape(
            (*sampled_shape, *sampled_shape)
        )[None, None, ...]

        reactive_dims = (self.nstates, *sampled_shape)
        matrix_order = [
            0,
            *range(2, 2 + self.dr),
            1,
            *range(2 + self.dr, 2 + 2 * self.dr),
        ]
        projected = projected.transpose(matrix_order).reshape(
            self.nstates * sampled_size,
            self.nstates * sampled_size,
        )
        projected = 0.5 * (projected + projected.conj().T)
        return projected, reactive_dims

    def _build_kinetic_propagator(self, exp_T, dt):
        """Exponentiate the Hermitian overlap-projected sampled kinetic energy."""
        from scipy.linalg import expm

        projected, reactive_dims = self._sampled_kinetic_hamiltonian()
        nsites = len(reactive_dims)
        reactive_propagator = expm(-1j * dt * projected)

        operator_tensor = reactive_propagator.reshape(
            *reactive_dims,
            *reactive_dims,
        )
        paired_order = [
            index
            for site in range(nsites)
            for index in (site, nsites + site)
        ]
        fused_operator = operator_tensor.transpose(paired_order).reshape(
            *(dim**2 for dim in reactive_dims)
        )
        mpo = _dense_to_mpo(
            fused_operator,
            reactive_dims,
            self.tt_options["max_rank"],
        )

        for i in range(self.dnr):
            idnr = self.dr + i
            expT_q = _as_numpy(exp_T[idnr]).reshape(
                1, 1, self.npts[idnr], self.npts[idnr]
            )
            mpo = _append_mpo_site(mpo, expT_q)

        return mpo

    def _build_kinetic_hamiltonian_mpo(self):
        """Build the full overlap-projected nuclear kinetic Hamiltonian."""
        max_rank = self.tt_options["max_rank"]
        if self.nuclear_kinetic_mpo is not None:
            return self._build_projected_nuclear_kinetic_mpo()

        projected, reactive_dims = self._sampled_kinetic_hamiltonian()
        kinetic = _dense_matrix_to_mpo(projected, reactive_dims, max_rank)
        for npoints in self.npts[self.dr:]:
            kinetic = _append_mpo_site(
                kinetic,
                np.eye(npoints, dtype=np.complex128).reshape(
                    1, 1, npoints, npoints
                ),
            )

        for axis in range(self.dnr):
            factors = [
                np.eye(dim, dtype=np.complex128).reshape(1, 1, dim, dim)
                for dim in self.dims
            ]
            site = 1 + self.dr + axis
            npoints = self.npts[self.dr + axis]
            factors[site] = np.asarray(
                self.axes[self.dr + axis].t(),
                dtype=np.complex128,
            ).reshape(1, 1, npoints, npoints)
            kinetic = kinetic + MPO(factors)

        if max_rank is not None and max(_mpo_ranks(kinetic)) > max_rank:
            kinetic = kinetic.compress(max_rank)
        return kinetic

    def _build_overlap_projection_mpo(self):
        """Return the full-grid electronic-overlap projection as an MPO."""
        if self.A is None:
            raise RuntimeError(
                "Electronic overlap data are required before building the "
                "overlap-projected nuclear kinetic MPO."
            )
        max_rank = self.tt_options["max_rank"]
        overlap = np.asarray(_as_numpy(self.A), dtype=np.complex128)
        expected = (
            self.nstates,
            self.nstates,
            *(dim for size in self.nx[:self.dr] for dim in (size, size)),
        )
        if overlap.shape != expected:
            raise ValueError(
                f"electronic overlap tensor shape {overlap.shape} != {expected}"
            )
        fused = overlap.reshape(
            self.nstates**2,
            *(size**2 for size in self.nx[:self.dr]),
        )
        for size in self.nx[self.dr:]:
            fused = fused[..., None] * np.ones(size**2, dtype=np.complex128)
        return _dense_to_mpo(fused, self.dims, max_rank)

    def _build_projected_nuclear_kinetic_mpo(self):
        """Build ``T_ab <phi_i(a_s)|phi_j(b_s)>`` as an MPO."""
        if self.nuclear_kinetic_mpo is None:
            raise RuntimeError("No nuclear kinetic MPO has been attached.")
        if self.overlap_projection_mpo is None:
            self.overlap_projection_mpo = self._build_overlap_projection_mpo()

        electronic_ones = np.ones(
            (self.nstates, self.nstates),
            dtype=np.complex128,
        ).reshape(1, 1, self.nstates, self.nstates)
        kinetic_on_full_chain = _prepend_mpo_site(
            self.nuclear_kinetic_mpo,
            electronic_ones,
        )
        projected = kinetic_on_full_chain * self.overlap_projection_mpo
        max_rank = self.tt_options["max_rank"]
        if max_rank is not None:
            projected = projected.compress(max_rank)
        return projected

    def _build_dense_projected_nuclear_kinetic_operator(self):
        """Materialize the overlap-projected nuclear KEO for small grids."""
        if self.nuclear_kinetic_mpo is None:
            raise RuntimeError("No nuclear kinetic MPO has been attached.")
        if self.projected_kinetic_dense is not None:
            return self.projected_kinetic_dense

        from pyqed.mps.mps import _mpo_to_dense_operator

        electronic_ones = np.ones(
            (self.nstates, self.nstates),
            dtype=np.complex128,
        ).reshape(1, 1, self.nstates, self.nstates)
        kinetic_on_full_chain = _prepend_mpo_site(
            self.nuclear_kinetic_mpo,
            electronic_ones,
        )
        kinetic = _mpo_to_dense_operator(kinetic_on_full_chain)
        projection = self._dense_overlap_projection_operator()
        projected = kinetic * projection
        self.projected_kinetic_dense = 0.5 * (projected + projected.conj().T)
        return self.projected_kinetic_dense

    def _dense_overlap_projection_operator(self):
        """Materialize the LDR overlap multiplier over the full chain."""
        if self.A is None:
            raise RuntimeError(
                "Electronic overlap data are required before building the "
                "overlap-projected nuclear kinetic operator."
            )
        sampled_shape = tuple(self.nx[:self.dr])
        overlap = np.asarray(_as_numpy(self.A), dtype=np.complex128)
        expected = (
            self.nstates,
            self.nstates,
            *(dim for size in sampled_shape for dim in (size, size)),
        )
        if overlap.shape != expected:
            raise ValueError(
                f"electronic overlap tensor shape {overlap.shape} != {expected}"
            )

        row_col_order = [
            0,
            *(2 + 2 * axis for axis in range(self.dr)),
            1,
            *(3 + 2 * axis for axis in range(self.dr)),
        ]
        tensor = overlap.transpose(row_col_order).reshape(
            self.nstates,
            *sampled_shape,
            *(1 for _ in range(self.dnr)),
            self.nstates,
            *sampled_shape,
            *(1 for _ in range(self.dnr)),
        )
        expanded_shape = (
            1,
            *(1 for _ in range(self.dr)),
            *self.nx[self.dr:],
            1,
            *(1 for _ in range(self.dr)),
            *self.nx[self.dr:],
        )
        tensor = tensor * np.ones(expanded_shape, dtype=np.complex128)
        dimension = int(np.prod(self.dims))
        return tensor.reshape(dimension, dimension)

    def _build_dense_nuclear_kinetic_mpo_propagator(self, dt):
        """Exponentiate a small dense projected KEO and refactor it as an MPO."""
        from scipy.linalg import expm

        projected = self._build_dense_projected_nuclear_kinetic_operator()
        propagator = expm(-1j * dt * projected)
        return _dense_matrix_to_mpo(
            propagator,
            self.dims,
            self.tt_options["max_rank"],
        )

    def _build_nuclear_kinetic_mpo_propagator(self, dt):
        """Exponentiate an attached overlap-projected nuclear kinetic MPO."""
        dense_limit = self.kinetic_exponential_options["dense_dimension_limit"]
        dimension = int(np.prod(self.dims))
        if dense_limit and dimension <= dense_limit:
            return self._build_dense_nuclear_kinetic_mpo_propagator(dt)

        if self.projected_kinetic_mpo is None:
            self.projected_kinetic_mpo = (
                self._build_projected_nuclear_kinetic_mpo()
            )
        return _mpo_exponential(
            self.projected_kinetic_mpo,
            -1j * dt,
            self.kinetic_exponential_options["scale"],
            self.kinetic_exponential_options["order"],
            self.tt_options["max_rank"],
        )

    def _build_kinetic_matrices(self, dt=1):
        """Build kinetic energy propagators"""
        T = []

        for d in range(self.ndim):
            axis = self.axes[d]
            if hasattr(axis, "expT"):
                T_d = np.asarray(axis.expT(dt))
            else:
                from scipy.linalg import expm

                T_d = expm(-1j * dt * np.asarray(axis.t()))
            if self.backend == "torch":
                T_d = require_torch_backend().as_tensor(T_d)
            T.append(T_d)

        return T

    def _reactive_matrix_mpo(self, matrix):
        """Factor an electronic/sample-grid operator into an MPO."""
        permute_order = [0, 1]
        for axis in range(self.dr):
            permute_order.extend((2 + axis, 2 + self.dr + axis))
        reshaped = _as_numpy(matrix).transpose(permute_order).reshape(
            self.nstates**2,
            *(self.npts[axis] ** 2 for axis in range(self.dr)),
        )
        return _dense_to_mpo(
            reshaped,
            [self.nstates, *self.npts[:self.dr]],
            self.tt_options["max_rank"],
        )

    def _embed_reactive_term(self, reactive, coordinate_factors):
        """Append diagonal expanded-coordinate factors to a reactive MPO."""
        result = _copy_tensor_train(reactive)
        for axis, values in enumerate(coordinate_factors):
            npoints = self.npts[self.dr + axis]
            result = _append_mpo_site(
                result,
                np.diag(np.asarray(values)).reshape(
                    1, 1, npoints, npoints
                ),
            )
        return result

    def _build_coarse_hamiltonian_mpo(self):
        """Build the unexponentiated local coarse Hamiltonian."""
        max_rank = self.tt_options["max_rank"]
        separable = getattr(
            getattr(self, "electronic_data", None),
            "separable_hamiltonian",
            None,
        )
        terms = []
        if separable is not None:
            nterms = separable.operators.shape[-3]
            if len(self.H_matrices) != nterms:
                raise ValueError(
                    f"Expected {nterms} separable operators; "
                    f"got {len(self.H_matrices)}"
                )
            reactive = [
                self._reactive_matrix_mpo(matrix)
                for matrix in self.H_matrices
            ]
            for term, matrix in enumerate(reactive):
                terms.append(self._embed_reactive_term(
                    matrix,
                    [factor[term] for factor in self.separable_factors],
                ))
        elif self.dnr:
            expected = int(
                1
                + 2 * self.dnr
                + self.dnr * (self.dnr - 1) / 2
            )
            if len(self.H_matrices) != expected:
                raise ValueError(
                    f"Number of matrices in H_matrices "
                    f"({len(self.H_matrices)}) does not match expected "
                    f"total ({expected})."
                )
            reactive = [
                self._reactive_matrix_mpo(matrix)
                for matrix in self.H_matrices
            ]
            ones = [
                np.ones(self.npts[self.dr + axis])
                for axis in range(self.dnr)
            ]
            for axis in range(self.dnr):
                linear = list(ones)
                linear[axis] = np.asarray(self.q_diff[axis])
                terms.append(self._embed_reactive_term(
                    reactive[1 + axis], linear
                ))

                quadratic = list(ones)
                quadratic[axis] = np.asarray(self.q_diff[axis]) ** 2
                terms.append(self._embed_reactive_term(
                    reactive[1 + self.dnr + axis], quadratic
                ))

            cross_index = 1 + 2 * self.dnr
            for offset, (first, second) in enumerate(
                (first, second)
                for first in range(self.dnr - 1)
                for second in range(first + 1, self.dnr)
            ):
                factors = list(ones)
                factors[first] = np.asarray(self.q_diff[first])
                factors[second] = np.asarray(self.q_diff[second])
                terms.append(self._embed_reactive_term(
                    reactive[cross_index + offset], factors
                ))

        if not terms:
            return 0.0 * _identity_mpo(self.dims)
        hamiltonian = terms[0]
        for term in terms[1:]:
            hamiltonian = hamiltonian + term
        if max_rank is not None and max(_mpo_ranks(hamiltonian)) > max_rank:
            hamiltonian = hamiltonian.compress(max_rank)
        return hamiltonian

    def build_hamiltonian(self):
        """Build and cache the full Hermitian CGLDR Hamiltonian MPO."""
        if self.H_matrices is None:
            self.prepare_electronic_data()
        if self.hamiltonian_mpo is not None:
            return self.hamiltonian_mpo

        max_rank = self.tt_options["max_rank"]
        surface = _diagonal_mpo(_as_numpy(self.e0), max_rank)
        for npoints in self.npts[self.dr:]:
            surface = _append_mpo_site(
                surface,
                np.eye(npoints, dtype=np.complex128).reshape(
                    1, 1, npoints, npoints
                ),
            )
        hamiltonian = self._build_kinetic_hamiltonian_mpo()
        hamiltonian = hamiltonian + surface
        hamiltonian = hamiltonian + self._build_coarse_hamiltonian_mpo()
        if max_rank is not None and max(_mpo_ranks(hamiltonian)) > max_rank:
            hamiltonian = hamiltonian.compress(max_rank)
        self.hamiltonian_mpo = hamiltonian
        return hamiltonian

    def _build_coarse_hamiltonian_propagator(self, dt):
        """Build a Strang-split coarse Born-Oppenheimer propagator."""
        # 1个H00;dnr个F;dnr个Gii（要乘以1/2）;(dnr)(dnr-1)/2个Gij（不要乘以1/2）
        # 不要乘好0.5后在输入

        max_rank = self.tt_options["max_rank"]
        separable = getattr(
            getattr(self, "electronic_data", None),
            "separable_hamiltonian",
            None,
        )
        if separable is not None:
            nterms = separable.operators.shape[-3]
            if len(self.H_matrices) != nterms:
                raise ValueError(
                    f"Expected {nterms} separable operators; "
                    f"got {len(self.H_matrices)}"
                )
            local_elements = self.nstates**2 * int(np.prod(self.npts))
            if local_elements <= 2_000_000:
                sampled_shape = tuple(self.npts[:self.dr])
                field = np.zeros(
                    (
                        *sampled_shape,
                        *tuple(self.npts[self.dr:]),
                        self.nstates,
                        self.nstates,
                    ),
                    dtype=np.complex128,
                )
                separable_operators = np.asarray(
                    self.electronic_data.separable_hamiltonian.operators,
                    dtype=np.complex128,
                )
                for term in range(nterms):
                    value = separable_operators[..., term, :, :].reshape(
                        *sampled_shape,
                        *(1,) * self.dnr,
                        self.nstates,
                        self.nstates,
                    )
                    for axis, factor in enumerate(self.separable_factors):
                        shape = (
                            *(1,) * self.dr,
                            *(
                                self.npts[self.dr + current]
                                if current == axis
                                else 1
                                for current in range(self.dnr)
                            ),
                            1,
                            1,
                        )
                        value = value * factor[term].reshape(shape)
                    field += value
                field = 0.5 * (field + field.swapaxes(-1, -2).conj())
                eigenvalues, eigenvectors = np.linalg.eigh(field)
                phases = np.exp(-1j * dt * eigenvalues)
                propagator = np.einsum(
                    "...ak,...k,...bk->...ab",
                    eigenvectors,
                    phases,
                    eigenvectors.conj(),
                    optimize=True,
                )
                result = _local_matrix_mpo(propagator, max_rank)
                print(f'Shape of HBO MPO: {_tensor_train_shapes(result)}')
                return result

            matrices = []
            for matrix in self.H_matrices:
                permute_order = [0, 1]
                for axis in range(self.dr):
                    permute_order.extend(
                        (2 + axis, 2 + self.dr + axis)
                    )
                reshaped = _as_numpy(matrix).transpose(permute_order).reshape(
                    self.nstates**2,
                    *[
                        self.npts[axis] ** 2
                        for axis in range(self.dr)
                    ],
                )
                matrices.append(_dense_to_mpo(
                    reshaped,
                    [self.nstates, *self.npts[:self.dr]],
                    max_rank,
                ))

            hamiltonian = None
            for term, matrix in enumerate(matrices):
                term_mpo = _copy_tensor_train(matrix)
                for axis, factor in enumerate(self.separable_factors):
                    npoints = self.npts[self.dr + axis]
                    core = np.diag(factor[term]).reshape(
                        1,
                        1,
                        npoints,
                        npoints,
                    )
                    term_mpo = _append_mpo_site(term_mpo, core)
                hamiltonian = (
                    term_mpo
                    if hamiltonian is None
                    else hamiltonian + term_mpo
                )
            if max_rank is not None:
                hamiltonian = hamiltonian.compress(max_rank)
            norm = _mpo_norm(hamiltonian) * abs(dt)
            scale = max(
                2,
                int(np.ceil(np.log2(norm))) if norm > 1.0 else 0,
            )
            result = _mpo_exponential(
                hamiltonian,
                constant=-1j * dt,
                scale=scale,
                order=10,
                max_rank=max_rank,
            )
            print(f'Shape of HBO MPO: {_tensor_train_shapes(result)}')
            return result

        ntotal = int(1 + 2 * self.dnr + self.dnr * (self.dnr - 1) / 2)

        if ntotal != len(self.H_matrices):
            raise ValueError(
                f'Number of matrices in H_matrices ({len(self.H_matrices)}) does not match '
                f'expected total ({ntotal}).')

        local_elements = (
            self.nstates**2 * int(np.prod(self.npts))
        )
        if (
            self.dnr
            and hasattr(self, "electronic_data")
            and local_elements <= 2_000_000
        ):
            sampled_shape = tuple(self.npts[:self.dr])
            expanded_shape = tuple(self.npts[self.dr:])
            field = np.zeros(
                (*sampled_shape, *expanded_shape, self.nstates, self.nstates),
                dtype=np.complex128,
            )
            singleton_expanded = (1,) * self.dnr
            gradients = np.asarray(
                self.electronic_data.hamiltonian_gradients
            )
            hessians = np.asarray(
                self.electronic_data.hamiltonian_hessians
            )
            q_mesh = np.meshgrid(
                *[np.asarray(q) for q in self.q_diff],
                indexing="ij",
            )
            for coordinate, q in enumerate(q_mesh):
                field += (
                    gradients[..., coordinate, :, :].reshape(
                        *sampled_shape,
                        *singleton_expanded,
                        self.nstates,
                        self.nstates,
                    )
                    * q.reshape(
                        *(1,) * self.dr,
                        *expanded_shape,
                        1,
                        1,
                    )
                )
            for first, q_first in enumerate(q_mesh):
                for second, q_second in enumerate(q_mesh):
                    field += 0.5 * (
                        hessians[..., first, second, :, :].reshape(
                            *sampled_shape,
                            *singleton_expanded,
                            self.nstates,
                            self.nstates,
                        )
                        * (q_first * q_second).reshape(
                            *(1,) * self.dr,
                            *expanded_shape,
                            1,
                            1,
                        )
                    )
            eigenvalues, eigenvectors = np.linalg.eigh(field)
            phases = np.exp(-1j * dt * eigenvalues)
            propagator = np.einsum(
                "...ak,...k,...bk->...ab",
                eigenvectors,
                phases,
                eigenvectors.conj(),
                optimize=True,
            )
            result = _local_matrix_mpo(propagator, max_rank)
            print(f'Shape of HBO MPO: {_tensor_train_shapes(result)}')
            return result

        coarse_terms = []
        with _optional_torch_no_grad():
            if self.H_matrices:
                matrices = []
                for matrix in self.H_matrices:
                    HBO0 = _as_numpy(matrix)

                    permute_order = [0, 1]
                    for d in range(self.dr):
                        permute_order.append(2 + d)
                        permute_order.append(2 + self.dr + d)

                    HBO0 = HBO0.transpose(permute_order)
                    HBO_reshaped = HBO0.reshape(self.nstates * self.nstates,
                                                *[self.npts[i] * self.npts[i] for i in range(
                                                    self.dr)])
                    matrices.append(_dense_to_mpo(
                        HBO_reshaped,
                        [self.nstates, *self.npts[:self.dr]],
                        max_rank,
                    ))

            for i in range(self.dnr):
                nq = self.npts[self.dr + i]
                q_diff = np.asarray(self.q_diff[i])
                F = _append_mpo_site(
                    _copy_tensor_train(matrices[1 + i]),
                    np.diag(q_diff).reshape(1, 1, nq, nq),
                )
                G = _append_mpo_site(
                    _copy_tensor_train(matrices[1 + self.dnr + i]),
                    np.diag(q_diff**2).reshape(1, 1, nq, nq),
                )
                coarse_terms.append(("single", i, F + G))

            cross_idx = 2 * self.dnr + 1
            for idx, (a, b) in enumerate(
                (a, b)
                for a in range(self.dnr - 1)
                for b in range(a + 1, self.dnr)
            ):
                term = _copy_tensor_train(matrices[cross_idx + idx])
                if all(not np.any(core) for core in term.factors):
                    continue
                q1 = np.diag(np.asarray(self.q_diff[a])).reshape(
                    1, 1, self.npts[self.dr + a], self.npts[self.dr + a]
                )
                q2 = np.diag(np.asarray(self.q_diff[b])).reshape(
                    1, 1, self.npts[self.dr + b], self.npts[self.dr + b]
                )
                coarse_terms.append(
                    (
                        "cross",
                        (a, b),
                        _append_mpo_site(_append_mpo_site(term, q1), q2),
                    )
                )

            def exponentiate_and_embed(descriptor, duration):
                kind, coordinates, hamiltonian = descriptor
                norm = _mpo_norm(hamiltonian) * abs(duration)
                scale = int(np.ceil(np.log2(norm))) if norm > 1 else 2
                propagator = _mpo_exponential(
                    hamiltonian,
                    constant=-1j * duration,
                    scale=scale,
                    order=10,
                    max_rank=max_rank,
                )
                reactive_cores = propagator.factors[:self.dr + 1]
                full_chain_cores = list(reactive_cores)

                if kind == "single":
                    coordinate = coordinates
                    q_core = propagator.factors[self.dr + 1]
                    connecting_rank = _mpo_ranks(propagator)[-2]
                    for k in range(self.dnr):
                        if k == coordinate:
                            full_chain_cores.append(q_core)
                        else:
                            phys_dim = self.npts[self.dr + k]
                            bond_dim = connecting_rank if k < coordinate else 1
                            full_chain_cores.append(np.einsum(
                                "ab,ij->abij",
                                np.eye(bond_dim, dtype=np.complex128),
                                np.eye(phys_dim, dtype=np.complex128),
                            ))
                else:
                    a, b = coordinates
                    q_a_core = propagator.factors[self.dr + 1]
                    q_b_core = propagator.factors[self.dr + 2]
                    ranks = _mpo_ranks(propagator)
                    rank_before_a = ranks[self.dr + 1]
                    rank_before_b = ranks[self.dr + 2]
                    for k in range(self.dnr):
                        if k == a:
                            full_chain_cores.append(q_a_core)
                        elif k == b:
                            full_chain_cores.append(q_b_core)
                        else:
                            phys_dim = self.npts[self.dr + k]
                            if k < a:
                                bond_dim = rank_before_a
                            elif k < b:
                                bond_dim = rank_before_b
                            else:
                                bond_dim = 1
                            full_chain_cores.append(np.einsum(
                                "ab,ij->abij",
                                np.eye(bond_dim, dtype=np.complex128),
                                np.eye(phys_dim, dtype=np.complex128),
                            ))
                return MPO(full_chain_cores)

            HBO = _identity_mpo([self.nstates, *self.npts])
            if coarse_terms:
                half_steps = [
                    exponentiate_and_embed(term, 0.5 * dt)
                    for term in coarse_terms[:-1]
                ]
                center_step = exponentiate_and_embed(coarse_terms[-1], dt)
                sequence = [
                    *half_steps,
                    center_step,
                    *reversed(half_steps),
                ]
                for propagator in sequence:
                    HBO = _matmul(propagator, HBO, max_rank)

        print(f'Shape of HBO MPO: {_tensor_train_shapes(HBO)}')

        return HBO

    def build_propagator(self, dt):
        """Build the complete time evolution propagator"""
        if self.H_matrices is None:
            self.prepare_electronic_data()
        if (
            self._propagator_time_step is not None
            and dt != self._propagator_time_step
        ):
            self._invalidate_propagators()

        max_rank = self.tt_options["max_rank"]
        if self.coarse_hamiltonian_propagator is None:
            print('Building coarse Hamiltonian MPO...')
            self.surface_propagator = _diagonal_mpo(
                np.exp(-1j * dt * _as_numpy(self.e0)),
                max_rank,
            )
            self.half_surface_propagator = _diagonal_mpo(
                np.exp(-0.5j * dt * _as_numpy(self.e0)),
                max_rank,
            )
            self.coarse_hamiltonian_propagator = (
                self._build_coarse_hamiltonian_propagator(dt)
            )

        if self.nuclear_kinetic_mpo is None:
            if self.expT is None:
                print('Building kinetic operator...')
                self.exp2T = self._build_kinetic_matrices(dt * 0.5)
                self.expT = self._build_kinetic_matrices(dt)

            if self.kinetic_propagator is None:
                print('Building kinetic propagators...')
                self.kinetic_propagator = self._build_kinetic_propagator(
                    self.expT,
                    dt,
                )
                self.half_kinetic_propagator = self._build_kinetic_propagator(
                    self.exp2T,
                    0.5 * dt,
                )
        elif self.kinetic_propagator is None:
            print('Building overlap-projected nuclear kinetic propagators...')
            self.kinetic_propagator = self._build_nuclear_kinetic_mpo_propagator(
                dt,
            )
            self.half_kinetic_propagator = (
                self._build_nuclear_kinetic_mpo_propagator(0.5 * dt)
            )
            print(
                "Shape of kinetic propagator:"
                f"{_tensor_train_shapes(self.half_kinetic_propagator)}"
            )

        print(
            "Shape of coarse Hamiltonian propagator:"
            f"{_tensor_train_shapes(self.coarse_hamiltonian_propagator)}"
        )
        print(
            "Norm of coarse Hamiltonian propagator: "
            f"{_mpo_norm(self.coarse_hamiltonian_propagator)}"
        )
        self._propagator_time_step = dt

        return

    def thermal_doubling(self, *, time_step=0.5, squarings=10):
        """Build an imaginary-time propagator and repeatedly square it."""
        if squarings < 0:
            raise ValueError("squarings must be non-negative")

        self.build_propagator(1j * time_step)
        max_rank = self.tt_options["max_rank"]
        propagator = _matmul(
            self.half_surface_propagator,
            self.half_kinetic_propagator,
            max_rank,
        )
        propagator = _matmul(
            propagator,
            self.coarse_hamiltonian_propagator,
            max_rank,
        )
        propagator = _matmul(
            propagator,
            self.half_surface_propagator,
            max_rank,
        )
        propagator = _matmul(
            propagator,
            self.half_kinetic_propagator,
            max_rank,
        )

        operators = []
        for _ in range(squarings):
            propagator = _matmul(propagator, propagator, max_rank)
            operators.append(_copy_tensor_train(propagator))

        return {
            "inverse_temperatures": (
                np.power(2.0, np.arange(1, squarings + 1)) * time_step
            ),
            "operators": operators,
        }

    def run(
        self,
        initial_state,
        *,
        time_step=0.5,
        steps=6000,
        output_every=40,
        save_data=True,
        integrator="hybrid",
        tdvp_options=None,
        tdvp_warmup_steps=5,
    ):
        """
        Run time evolution

        Parameters
        ----------
        initial_state : MPS
            Initial wavefunction
        time_step : float
            Time step
        steps : int
            Number of time steps
        output_every : int
            Output frequency
        save_data : bool
            Whether to save intermediate data
        integrator : {"split", "tdvp", "tdvp2", "hybrid"}
            Strang-split MPO propagation, one-/two-site TDVP, or two-site
            bond growth followed by one-site TDVP.
        tdvp_options : dict, optional
            Extra :class:`pyqed.mps.tdvp.TDVPEngine` options. The CGLDR
            ``max_rank`` is always used as the maximum MPS bond dimension.
        tdvp_warmup_steps : int
            Number of initial two-site steps used by the hybrid integrator.

        Returns
        -------
        CGLDR
            This solver with the recorded states attached.
        """
        if not isinstance(initial_state, MPS):
            raise TypeError("initial_state must be a pyqed.mps.mps.MPS")
        integrator_key = str(integrator).lower().replace("_", "-")
        aliases = {
            "split": "split",
            "strang": "split",
            "tdvp": "tdvp",
            "tdvp1": "tdvp",
            "one-site-tdvp": "tdvp",
            "tdvp2": "tdvp2",
            "two-site-tdvp": "tdvp2",
            "hybrid": "hybrid",
            "hybrid-tdvp": "hybrid",
        }
        if integrator_key not in aliases:
            raise ValueError(
                "integrator must be 'split', 'tdvp', 'tdvp2', or 'hybrid'"
            )
        integrator_key = aliases[integrator_key]
        if tdvp_warmup_steps < 0:
            raise ValueError("tdvp_warmup_steps must be non-negative")
        self.integrator = integrator_key
        max_rank = self.tt_options["max_rank"]
        tdvp_engines = {}
        if integrator_key == "split":
            print('Building propagator...')
            self.build_propagator(time_step)
            self.split_step_left = _matmul(
                self.half_surface_propagator,
                self.half_kinetic_propagator,
                max_rank,
            )
            self.split_step_right = _matmul(
                self.half_kinetic_propagator,
                self.half_surface_propagator,
                max_rank,
            )
            output_files = {
                "coarse_hamiltonian_propagator.pkl": (
                    self.coarse_hamiltonian_propagator
                ),
                "kinetic_propagator.pkl": self.kinetic_propagator,
                "half_kinetic_propagator.pkl": self.half_kinetic_propagator,
                "split_step_left.pkl": self.split_step_left,
                "split_step_right.pkl": self.split_step_right,
            }
        else:
            from pyqed.mps.tdvp import TDVPEngine

            print('Building Hamiltonian MPO...')
            hamiltonian = self.build_hamiltonian()
            options = {} if tdvp_options is None else dict(tdvp_options)
            if "integrator" in options or "max_bond" in options:
                raise ValueError(
                    "tdvp_options cannot override integrator or max_bond"
                )
            engine_integrators = (
                ("tdvp2", "tdvp")
                if integrator_key == "hybrid"
                else (integrator_key,)
            )
            tdvp_engines = {
                name: TDVPEngine(
                    hamiltonian,
                    integrator=name,
                    max_bond=max_rank,
                    **options,
                )
                for name in engine_integrators
            }
            output_files = {"hamiltonian_mpo.pkl": hamiltonian}
        if save_data:
            for filename, data in output_files.items():
                with open(os.path.join(self.output_folder, filename), "wb") as f:
                    pickle.dump(data, f)


        print('Starting time evolution...')
        self.time_step = time_step
        self.steps = steps
        clear_memory()

        # Log memory usage
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Memory RSS: {mem_info.rss / 1e9:.2f} GB")

        # Check for checkpoint file
        checkpoint_file = os.path.join(self.output_folder, 'evolution_checkpoint.pkl')
        cycle_step_counts = _cycle_step_counts(steps, output_every)
        total_cycles = len(cycle_step_counts)

        if save_data and os.path.exists(checkpoint_file):
            print("Found checkpoint file, resuming...")
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            if checkpoint.get("integrator", "split") != integrator_key:
                raise ValueError(
                    "Checkpoint integrator does not match this run"
                )
            states = checkpoint["states"]
            state = checkpoint["state"]
            start_cycle = checkpoint["cycle"]
            tdvp_truncation_errors = checkpoint.get(
                "tdvp_truncation_errors", []
            )
            integrator_history = checkpoint.get("integrator_history", [])
            bond_dimensions = checkpoint.get(
                "bond_dimensions", [tuple(state.bond_orders())]
            )
            print(f"Resuming from cycle {start_cycle}")
        else:
            state = initial_state.copy()
            states = [state.copy()]
            start_cycle = 0
            tdvp_truncation_errors = []
            integrator_history = []
            bond_dimensions = [tuple(state.bond_orders())]

        with _optional_torch_no_grad():
            for cycle in range(start_cycle, total_cycles):
                # Determine steps for this cycle
                steps_this_cycle = cycle_step_counts[cycle]

                for local_step in range(steps_this_cycle):
                    if integrator_key == "split":
                        step_integrator = "split"
                        state = _matmul(
                            self.split_step_right, state, max_rank
                        )
                        state = _matmul(
                            self.coarse_hamiltonian_propagator,
                            state,
                            max_rank,
                        )
                        state = _matmul(
                            self.split_step_left, state, max_rank
                        )
                        tdvp_truncation_errors.append(0.0)
                    else:
                        step_integrator = integrator_key
                        if integrator_key == "hybrid":
                            step_integrator = (
                                "tdvp2"
                                if len(integrator_history) < tdvp_warmup_steps
                                else "tdvp"
                            )
                        state, info = tdvp_engines[step_integrator].step(
                            state,
                            time_step,
                            normalize=True,
                            return_info=True,
                        )
                        tdvp_truncation_errors.append(
                            float(info.get("truncation_error", 0.0))
                        )
                    integrator_history.append(step_integrator)

                    # Log progress
                    if local_step == 0 or local_step == steps_this_cycle - 1:
                        step_num = cycle * output_every + local_step
                        print(
                            f"Step {step_num}, tensor shapes: "
                            f"{_tensor_train_shapes(state)}"
                        )

                # Save intermediate results and checkpoint
                if cycle < total_cycles - 1:
                    states.append(state.copy())
                    bond_dimensions.append(tuple(state.bond_orders()))

                    if save_data:
                        checkpoint = {
                            "states": states,
                            "state": state.copy(),
                            "cycle": cycle + 1,
                            "integrator": integrator_key,
                            "tdvp_truncation_errors": tdvp_truncation_errors,
                            "integrator_history": integrator_history,
                            "bond_dimensions": bond_dimensions,
                        }
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(checkpoint, f)
                        print(f"Checkpoint saved at cycle {cycle}")

                clear_memory()

            # Final state
            if total_cycles:
                states.append(state.copy())
                bond_dimensions.append(tuple(state.bond_orders()))
            print(f'Final tensor shapes: {_tensor_train_shapes(state)}')

            # Clean up checkpoint file
            if save_data and os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print("Evolution completed, checkpoint file removed.")

        # Save final results
        if save_data:
            final_time = time_step * steps
            with open(
                os.path.join(self.output_folder, f"state_time_{final_time}.pkl"),
                "wb",
            ) as f:
                pickle.dump(state, f)
            with open(os.path.join(self.output_folder, "states.pkl"), "wb") as f:
                pickle.dump(states, f)

        self.states = states
        self.tdvp_truncation_errors = np.asarray(
            tdvp_truncation_errors, dtype=float
        )
        self.integrator_history = np.asarray(integrator_history, dtype="U5")
        self.bond_dimensions = np.asarray(bond_dimensions, dtype=int)


        print('Time evolution complete')
        return self

    def compute_populations(self, *, plot=False, femtoseconds=True):
        """Compute electronic-state populations from the recorded MPS states."""
        if not self.states:
            raise RuntimeError("Run the dynamics before computing populations.")

        populations = np.zeros((len(self.states), self.nstates))
        for sample, state in enumerate(self.states):
            canonical = state.copy().right_canonicalize()
            electronic_core = canonical.factors[0]
            total_norm = np.linalg.norm(electronic_core) ** 2
            for electronic_state in range(self.nstates):
                state_norm = np.linalg.norm(
                    electronic_core[:, electronic_state, :]
                ) ** 2
                populations[sample, electronic_state] = state_norm / total_norm

        time_scale = au2fs if femtoseconds else 1.0
        self.populations = populations
        self.times = np.linspace(
            self.initial_time,
            self.initial_time + self.time_step * self.steps,
            len(self.states),
        ) * time_scale
        self.time_unit = "fs" if femtoseconds else "au"

        if plot:
            self.plot_populations()
        return populations

    @staticmethod
    def _dense_state_array(state):
        """Contract a dense finite MPS to an array over its physical sites."""
        if not isinstance(state, MPS):
            raise TypeError("state must be a pyqed.mps.mps.MPS")
        first = state._get_std_B(0)
        if hasattr(first, "qns"):
            raise NotImplementedError(
                "Coordinate expectations are currently implemented for dense "
                "MPS tensors."
            )
        dense = np.asarray(first)[0]
        for site in range(1, state.L):
            tensor = state._get_std_B(site)
            if hasattr(tensor, "qns"):
                raise NotImplementedError(
                    "Coordinate expectations are currently implemented for "
                    "dense MPS tensors."
                )
            dense = np.tensordot(
                dense,
                np.asarray(tensor),
                axes=([-1], [0]),
            )
        return np.asarray(dense[..., 0])

    def compute_coordinate_expectations(self, *, femtoseconds=True):
        """Compute nuclear-coordinate means and variances from recorded states.

        The returned arrays have shape ``(nsamples, ndim)`` in the solver's
        reordered coordinate order, i.e. ``self.coordinate_names``.
        """
        if not self.states:
            raise RuntimeError("Run the dynamics before computing expectations.")

        means = np.zeros((len(self.states), self.ndim), dtype=float)
        variances = np.zeros_like(means)
        axes_to_sum = tuple(range(self.ndim + 1))
        for sample, state in enumerate(self.states):
            amplitude = self._dense_state_array(state)
            probability = np.abs(amplitude) ** 2
            norm = float(np.sum(probability))
            if norm <= 0.0:
                raise ValueError("Cannot compute expectations for a zero state.")
            for axis, grid in enumerate(self.x):
                marginal = np.sum(
                    probability,
                    axis=tuple(a for a in axes_to_sum if a != axis + 1),
                )
                mean = float(np.sum(marginal * grid) / norm)
                means[sample, axis] = mean
                variances[sample, axis] = float(
                    np.sum(marginal * (grid - mean) ** 2) / norm
                )

        time_scale = au2fs if femtoseconds else 1.0
        self.times = np.linspace(
            self.initial_time,
            self.initial_time + self.time_step * self.steps,
            len(self.states),
        ) * time_scale
        self.time_unit = "fs" if femtoseconds else "au"
        self.coordinate_expectations = means
        self.coordinate_variances = variances
        return {
            "names": self.coordinate_names,
            "means": means,
            "variances": variances,
            "times": self.times,
            "time_unit": self.time_unit,
        }

    def plot_populations(self, *, save=True, title=None):
        """Plot the electronic-state populations recorded by :meth:`run`."""
        if self.populations is None:
            self.compute_populations()

        time_np = self.times
        p_np = self.populations

        # 固定图像大小
        plt.figure(figsize=(8, 6))

        for state in range(p_np.shape[1]):
            plt.plot(time_np, p_np[:, state], label=f'State {state}', linewidth=1.5, alpha=0.8)

        if title:
            plt.title(title, fontsize=14, pad=10)

        # 固定y轴范围，便于后续对比
        plt.ylim(-0.05, 1.05)
        plt.xlabel(f"Time ({self.time_unit})")
        plt.ylabel("Population")

        plt.figtext(0.99, 0.001, self.current_time, ha='right', va='bottom', fontsize=6)
        plt.legend()
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.output_folder, "population.pdf")
            plt.savefig(save_path)

        plt.show()

        return

    def dump(self, fname):
        """
        save results to disk

        Parameters
        ----------
        fname : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        import pickle
        with open(fname, 'wb') as f:
            pickle.dump(self, f)



def H_val(full_coords):
    """
    简化的哈密顿量函数full_coords: 24个振动模式的坐标，按照modes顺序排列
        v10a,
        v6a,///v1, v9a//v8a, v2
        v4, v5,
        v6b,   v3,v8b,v7b
        v16a,  v17a,
        v12,v18a,  v19a,v13,
        v18b,   v14,  v19b,   v20b,
        v16b, v11
    """
    au2ev = 27.2116

    freq = torch.tensor([
        0.1139, 0.0739, 0.1258, 0.1525, 0.1961, 0.3788,
        0.0937, 0.1219,
        0.0873, 0.1669, 0.1891, 0.3769,
        0.0423, 0.1190,
        0.1266, 0.1408, 0.1840, 0.3734,
        0.1318, 0.1425, 0.1756, 0.3798,
        0.0521, 0.0973,

    ]) / au2ev

    # 线性耦合系数
    ai = torch.tensor([-0.0981, -0.0503, 0.1452, -0.0445, 0.0247]) / au2ev  # S0态
    bi = torch.tensor([0.1355, -0.1710, 0.0375, 0.0168, 0.0162]) / au2ev  # S1态
    ci = torch.tensor([0.2080]) / au2ev  # 非绝热耦合

    aij_matrices = [
        # Ag组 (5x5)
        torch.tensor([
            [0, 0.00108, -0.00204, -0.00135, -0.00285],
            [0.00108, 0, 0.00474, 0.00154, -0.00163],
            [-0.00204, 0.00474, 0, 0.00872, -0.00474],
            [-0.00135, 0.00154, 0.00872, 0, -0.00143],
            [-0.00285, -0.00163, -0.00474, -0.00143, 0],
        ]) / au2ev,
        # B1g组 (1x1)
        torch.tensor([[-0.01159]]) / au2ev,
        # B2g组 (2x2)
        torch.tensor([
            [-0.02252, -0.00049],
            [-0.00049, -0.01825]
        ]) / au2ev,
        # B3g组 (4x4)
        torch.tensor(
            [
                [-0.00741, 0.01321, -0.00717, 0.00515],
                [0.01321, 0.05183, -0.03942, 0.00170],
                [-0.00717, -0.03942, -0.05733, -0.00204],
                [0.00515, 0.00170, -0.00204, -0.00333],
            ]
        ) / au2ev,
        # Au组 (2x2)
        torch.tensor(
            [[0.01145, 0.00100],
             [0.00100, -0.02040]]
        ) / au2ev,
        # B1u组 (4x4)
        torch.tensor(
            [
                [-0.04819, 0.00525, -0.00485, -0.00326],
                [0.00525, -0.00792, 0.00852, 0.00888],
                [-0.00485, 0.00852, -0.02429, -0.00443],
                [-0.00326, 0.00888, -0.00443, -0.00492],
            ]
        ) / au2ev,
        # B2u组 (4x4)
        torch.tensor(
            [
                [-0.00277, 0.00016, -0.00250, 0.00357],
                [0.00016, 0.03924, -0.00197, -0.00355],
                [-0.00250, -0.00197, 0.00992, 0.00623],
                [0.00357, -0.00355, 0.00623, -0.00110],
            ]
        ) / au2ev,
        # B3u组 (2x2)
        torch.tensor([
            [-0.02176, -0.00624],
            [-0.00624, 0.00315]
        ]) / au2ev
    ]

    bij_matrices = [
        # Ag组 (5x5)
        torch.tensor([
            [0, -0.00298, -0.00189, -0.00203, -0.00128],
            [-0.00298, 0, 0.00155, 0.00311, -0.00600],
            [-0.00189, 0.00155, 0, 0.01194, -0.00334],
            [-0.00203, 0.00311, 0.01194, 0, -0.00713],
            [-0.00128, -0.00600, -0.00334, -0.00713, 0],
        ]) / au2ev,
        # B1g组 (1x1)
        torch.tensor([[-0.01159]]) / au2ev,
        # B2g组 (2x2)
        torch.tensor([
            [-0.03445, 0.00911],
            [0.00911, -0.00265],
        ]) / au2ev,
        # B3g组 (4x4)
        torch.tensor([
            [-0.00385, -0.00661, 0.00429, -0.00246],
            [-0.00661, 0.04842, -0.03034, -0.00185],
            [0.00429, -0.03034, -0.06332, -0.00388],
            [-0.00246, -0.00185, -0.00388, -0.00040],
        ]) / au2ev,
        # Au组 (2x2)
        torch.tensor([
            [-0.01459, -0.00091],
            [-0.00091, -0.00618],
        ]) / au2ev,
        # B1u组 (4x4)
        torch.tensor([
            [-0.00840, 0.00536, -0.00097, 0.00034],
            [0.00536, 0.00429, 0.00209, -0.00049],
            [-0.00097, 0.00209, -0.00734, 0.00346],
            [0.00034, -0.00049, 0.00346, 0.00062],
        ]) / au2ev,
        # B2u组 (4x4)
        torch.tensor([
            [-0.01179, -0.00844, 0.07000, -0.01249],
            [-0.00844, 0.04000, -0.05000, 0.00265],
            [0.07000, -0.05000, 0.01246, -0.00422],
            [-0.01249, 0.00265, -0.00422, 0.00069],
        ]) / au2ev,
        # B3u组 (2x2)
        torch.tensor([
            [-0.02214, -0.00261],
            [-0.00261, -0.00496],
        ]) / au2ev
    ]
    # 非绝热二次耦合矩阵
    cij_matrices = [
        torch.tensor([[-0.01000, -0.00551, 0.00127, 0.00799, -0.00512]]) / au2ev,
        torch.tensor([
            [-0.01372, -0.00466, 0.00329, -0.00031],
            [0.00598, -0.00914, 0.00961, 0.00500]
        ]) / au2ev,
        torch.tensor([
            [-0.01056, 0.00559, 0.00401, -0.00226],
            [-0.01200, -0.00213, 0.00328, -0.00396]
        ]) / au2ev,
        torch.tensor([
            [0.00118, -0.00009, -0.00285, -0.00095],
            [0.01281, -0.01780, 0.00134, -0.00481]
        ]) / au2ev
    ]

    # v10a,
    # v6a, /// v1, v9a // v8a, v2
    # v4, v5,
    # v6b, v3, v8b, v7b
    # v16a, v17a,
    # v12, v18a, v19a, v13,
    # v18b, v14, v19b, v20b,
    # v16b, v11
    # 模式分组索引
    groups = [
        [1, 2, 3, 4, 5],  # Ag: v6a, v1, v9a, v8a, v2
        [0],  # B1g: v10a
        [6, 7],  # B2g: v4, v5
        [8, 9, 10, 11],  # B3g: v6b, v3, v8b, v7b
        [12, 13],  # Au: v16a, v17a
        [14, 15, 16, 17],  # B1u: v12, v18a, v19a, v13
        [18, 19, 20, 21],  # B2u: v18b, v14, v19b, v20b
        [22, 23]  # B3u: v16b, v11
    ]

    # 非绝热耦合的分组
    cij_groups = [
        ([0], [1, 2, 3, 4, 5]),  # B1g x Ag
        ([6, 7], [8, 9, 10, 11]),  # B2g x B3g
        ([12, 13], [14, 15, 16, 17]),  # Au x B1u
        ([22, 23], [18, 19, 20, 21])  # B3u x B2u
    ]

    # 能量偏移
    delta = 0.8460 / 2.0 / au2ev

    # 基态振动能量
    vg = 0.0
    for i in range(min(len(full_coords), len(freq))):
        vg += freq[i] * full_coords[i] ** 2 / 2

    # S1, S2态基本能量
    v1 = vg - delta
    v2 = vg + delta

    # 添加线性耦合 (前5个Ag模式)
    for i in range(min(5, len(full_coords))):
        mode_idx = i + 1
        if mode_idx < len(full_coords):
            v1 += ai[i] * full_coords[mode_idx]
            v2 += bi[i] * full_coords[mode_idx]

    # aij and bij
    for group_idx, group in enumerate(groups):
        if group_idx < len(aij_matrices) and group_idx < len(bij_matrices):
            aij = aij_matrices[group_idx]
            bij = bij_matrices[group_idx]

            for i, mode_i in enumerate(group):
                for j, mode_j in enumerate(group):
                    if (mode_i < len(full_coords) and mode_j < len(full_coords) and
                            i < aij.shape[0] and j < aij.shape[1]):
                        coord_i = full_coords[mode_i]
                        coord_j = full_coords[mode_j]
                        v1 += aij[i, j] * coord_i * coord_j
                        v2 += bij[i, j] * coord_i * coord_j

    coup = 0.0
    if len(full_coords) > 0:
        coup += ci[0] * full_coords[0]

    # # 添加所有非绝热二次耦合
    for group_idx, (gi, gj) in enumerate(cij_groups):
        if group_idx < len(cij_matrices):
            cij = cij_matrices[group_idx]

            for i, mode_i in enumerate(gi):
                for j, mode_j in enumerate(gj):
                    if (mode_i < len(full_coords) and mode_j < len(full_coords) and
                            i < cij.shape[0] and j < cij.shape[1]):
                        coupling = cij[i, j] * full_coords[mode_i] * full_coords[mode_j]
                        coup += coupling

    H = torch.zeros((2, 2), dtype=torch.float64)

    H[0, 0] = v1
    H[1, 1] = v2
    H[1, 0] = coup
    H[0, 1] = coup

    return H


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    start_time = time.time()

    #############Change According the system#############

    nstates = 2
    dt = 0.5/au2fs

    nt = 300  # 1au=2.41888432651e-2fs
    nout = 1

    dims = [2, 2]
    ndim = dims[0] + dims[1]  # Reactive + Non-Reactive coordinates
    domains = [[-6, 6], ] * ndim
    npts = [15] * ndim
    freq = torch.tensor([
        0.1139, 0.0739, 0.1258, 0.1525, 0.1961, 0.3788,
        0.0937, 0.1219,
        0.0873, 0.1669, 0.1891, 0.3769,
        0.0423, 0.1190,
        0.1266, 0.1408, 0.1840, 0.3734,
        0.1318, 0.1425, 0.1756, 0.3798,
        0.0521, 0.0973,

    ]) / au2ev
    mass = 1 / freq
    tt_options = {
        "max_rank": 50,
    }

    #q0 = [1.540,0.008,-0.332,0.002]+[0,]* (dims[1] - 4)  # Initial position for reactive
    # coordinates
    #q0 = [1.547,-0.017]
    #q0=[1.540,0.008]
    q0 = (0.0,) * dims[1]
    print("TT options:", tt_options)
    coordinate_names = tuple(f"R{i}" for i in range(dims[0])) + tuple(
        f"q{i}" for i in range(dims[1])
    )
    sol = CGLDR(
        DVR(domains, npts, mass=mass, names=coordinate_names),
        ElectronicPartition(
            sampled=coordinate_names[:dims[0]],
            expanded=coordinate_names[dims[0]:],
            center=q0,
        ),
        tt_options=tt_options,
        state_ids=tuple(range(nstates)),
    )

    current_time_folder = "."
    os.makedirs(current_time_folder, exist_ok=True)
    print(f"Output will be saved to: {current_time_folder}/")
    sol.output_folder = current_time_folder

    ################# Do not need to change below this line #################

    # If you get electonic structures calcuation results
    # Define sol.adiabatic_states and sol.H_matrices with your own results rather than call
    # get_hamiltonian_matrices
    # sol.adiabatic_states is the adiabatic eigenvectors of H0,in MPS format
    # The H_matrices: [H0; F1 to Fdnr; 0.5*G1 to 0.5*Gdnr; G12 G13 ... G{dnr-1,dnr}]
    # H_matrices should be matrix under the adiabatic basis, which is the eigenvectors of H0
    # H_mat shape need to be [ns,ns,nx,ny...,nx,ny...] in tensor format

    H_mat, phi = sol.get_hamiltonian_matrices(H_val)
    sol.adiabatic_states = phi.clone()
    sol.H_matrices = [h.clone() for h in H_mat]

    ################# Do not need to change below this line #################
    coord = sol.x

    ndims = sol.ndim  # All dimensions
    nrdims = sol.dr  # Reactive dimensions

    lengths = [len(coord[i]) for i in range(ndims)]  # Grid points of each dimension
    d_vals = [interval(coord[i]) for i in range(ndims)]  # Interval of each dimension
    print(f'Total Dimension:{ndim},Grid points: {npts}, Intervals: {d_vals}')


    def create_psi(coord, nstates, target_state, nrdims, x0=None, **gwp_kwargs):

        if x0 is None:
            x0 = [0] * nrdims

        meshgrids = np.meshgrid(*coord[:nrdims], indexing='ij')
        grid_shape = tuple(len(coord[i]) for i in range(nrdims))

        psi_shape = (nstates,) + grid_shape
        psi0 = np.zeros(psi_shape, dtype=complex)

        it = np.nditer(meshgrids, flags=['multi_index'])
        for _ in it:
            indices = it.multi_index
            coords = [meshgrids[d][indices] for d in range(nrdims)]
            psi0[(target_state,) + indices] = gwp(np.array(coords), x0=x0, ndim=nrdims,
                                                  **gwp_kwargs)
        psi0 = psi0 * np.sqrt(np.prod(d_vals[:nrdims]))
        return psi0


    psi0 = create_psi(coord, nstates, target_state=1, nrdims=nrdims)

    if nrdims > 0:
        projectionindex = gen_einsum_string(dims[0], keyword='projection', dr=dims[0], dnr=0)
        psi0 = torch.einsum(projectionindex, torch.tensor(psi0, dtype=torch.complex128),
                            sol.adiabatic_states.to(torch.complex128))
    else:
        psi0 = torch.tensor(psi0, dtype=torch.complex128)

    initial_cores = _decompose_dense(psi0, tt_options["max_rank"])

    if len(coord) > nrdims:
        initial_cores.extend(
            gwp_mps(coord[nrdims:], dx=d_vals[nrdims:], nstates=None)
        )

    initial_state = MPS(initial_cores).right_canonicalize()

    print(f"Initial state tensor shapes: {_tensor_train_shapes(initial_state)}")
    print(np.sqrt(initial_state.norm_squared()))
    sol.run(
        initial_state,
        time_step=dt,
        steps=nt,
        output_every=nout,
    )
    sol.dump(os.path.join(current_time_folder, 'Result'))
    state = sol.states[-1]
    print(np.sqrt(state.norm_squared()))
    populations = sol.compute_populations(plot=True)

    np.save(os.path.join(current_time_folder, "populations.npy"), populations)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"----- The total time is：{execution_time} seconds. -----")
