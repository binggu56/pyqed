"""Functional tensor-train regression from scattered tensor-valued data."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from pyqed.mps.dense_canonical import left_qr, right_rq


_BASES = {"chebyshev", "legendre", "fourier"}
_NORMALIZATIONS = {"elementwise", "frobenius"}


def _spec(value, size, name):
    if isinstance(value, str) or np.isscalar(value):
        return (value,) * size
    value = tuple(value)
    if len(value) != size:
        raise ValueError(f"{name} must be scalar or contain one value per coordinate")
    return value


def _rank_limits(sizes, rank):
    size = len(sizes)
    requested = _spec(rank, max(size - 1, 0), "rank")
    ranks = [1]
    for split, value in enumerate(requested, 1):
        value = int(value)
        if value < 1:
            raise ValueError("TT ranks must be positive")
        left = int(np.prod(sizes[:split], dtype=object))
        right = int(np.prod(sizes[split:], dtype=object))
        ranks.append(min(value, left, right))
    return tuple(ranks + [1])


def hermitian_basis(size):
    """Return a flattened orthonormal basis for Hermitian matrices."""
    size = int(size)
    if size < 1:
        raise ValueError("matrix size must be positive")
    basis = np.zeros((size * size, size, size), dtype=complex)
    channel = 0
    for row in range(size):
        basis[channel, row, row] = 1.0
        channel += 1
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    for row in range(size):
        for column in range(row + 1, size):
            basis[channel, row, column] = inverse_sqrt_two
            basis[channel, column, row] = inverse_sqrt_two
            channel += 1
            basis[channel, row, column] = 1j * inverse_sqrt_two
            basis[channel, column, row] = -1j * inverse_sqrt_two
            channel += 1
    return basis.reshape(size * size, size * size)


def pack_hermitian(values, basis=None):
    """Pack Hermitian matrices into real orthonormal-basis coefficients."""
    values = np.asarray(values)
    if values.ndim < 2 or values.shape[-2] != values.shape[-1]:
        raise ValueError("Hermitian values must have square trailing dimensions")
    size = values.shape[-1]
    basis = hermitian_basis(size) if basis is None else np.asarray(basis)
    flat = values.reshape(-1, size * size)
    channels = flat @ basis.conj().T
    imaginary = float(np.max(np.abs(channels.imag)))
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(channels)))
    )
    if imaginary > tolerance:
        raise ValueError("Hermitian-basis coefficients are unexpectedly complex")
    return channels.real.reshape(*values.shape[:-2], size * size)


class FunctionalTT:
    r"""Fixed-rank functional tensor train fitted by alternating least squares.

    The model represents a scalar, vector, or matrix-valued function as

    .. math::

        f(q_1,\ldots,q_d) = G_1(q_1)G_2(q_2)\cdots G_d(q_d)O,

    with each coordinate core expanded in a one-dimensional basis and a
    terminal output core :math:`O`.  All output elements therefore share the
    same coordinate factors.  The fit accepts scattered points; no
    tensor-product training grid is needed.  ``normalization`` selects
    either independent element scaling or one common scale corresponding to
    an unweighted Frobenius-norm objective.  ``hermitian="auto"`` detects
    square Hermitian training data and fits its coefficients in a real,
    orthonormal Hermitian basis.  This guarantees Hermitian predictions while
    retaining the original TT ranks.
    """

    def __init__(
        self,
        *,
        bases="chebyshev",
        degrees=8,
        rank=4,
        bounds=None,
        normalization="frobenius",
        hermitian="auto",
        regularization=1.0e-10,
        sweeps=12,
        rtol=1.0e-8,
        local_rtol=1.0e-10,
        local_maxiter=None,
        patience=3,
        random_state=None,
    ):
        self.bases = bases
        self.degrees = degrees
        self.rank = rank
        self.bounds = bounds
        self.normalization = str(normalization).lower()
        self.hermitian = hermitian
        self.regularization = float(regularization)
        self.sweeps = int(sweeps)
        self.rtol = float(rtol)
        self.local_rtol = float(local_rtol)
        self.local_maxiter = local_maxiter
        self.patience = int(patience)
        self.random_state = None if random_state is None else int(random_state)
        if self.regularization < 0.0:
            raise ValueError("regularization must be nonnegative")
        if self.normalization not in _NORMALIZATIONS:
            raise ValueError(
                "normalization must be 'frobenius' or 'elementwise'"
            )
        if self.hermitian != "auto" and not isinstance(
            self.hermitian, (bool, np.bool_)
        ):
            raise ValueError("hermitian must be 'auto', True, or False")
        if self.hermitian != "auto":
            self.hermitian = bool(self.hermitian)
        if self.sweeps < 1 or self.patience < 1:
            raise ValueError("sweeps and patience must be positive")
        if self.rtol < 0.0 or self.local_rtol <= 0.0:
            raise ValueError("rtol must be nonnegative and local_rtol positive")

    def _configure(self, coordinates, output_shape=()):
        self.ndim = int(coordinates.shape[1])
        self.output_shape_ = tuple(int(size) for size in output_shape)
        if any(size < 1 for size in self.output_shape_):
            raise ValueError("output dimensions must be positive")
        self.output_size_ = (
            int(np.prod(self.output_shape_, dtype=int)) if self.output_shape_ else 1
        )
        self._configure_output_basis()
        self.bases_ = tuple(
            str(value).lower() for value in _spec(self.bases, self.ndim, "bases")
        )
        unknown = set(self.bases_) - _BASES
        if unknown:
            raise ValueError(f"unknown functional TT basis: {sorted(unknown)}")
        self.degrees_ = tuple(
            int(value) for value in _spec(self.degrees, self.ndim, "degrees")
        )
        if any(value < 0 for value in self.degrees_):
            raise ValueError("basis degrees must be nonnegative")

        if self.bounds is None:
            bounds = np.column_stack(
                (np.min(coordinates, axis=0), np.max(coordinates, axis=0))
            )
        else:
            bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (self.ndim, 2):
            raise ValueError("bounds must have shape (n_coordinates, 2)")
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("each coordinate bound must be finite and increasing")
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(bounds))
        if np.any(coordinates < bounds[:, 0] - tolerance[:, 0]) or np.any(
            coordinates > bounds[:, 1] + tolerance[:, 1]
        ):
            raise ValueError("training coordinates lie outside bounds")
        self.bounds_ = bounds
        self.basis_sizes_ = tuple(
            2 * degree + 1 if basis == "fourier" else degree + 1
            for basis, degree in zip(self.bases_, self.degrees_)
        )
        if self.output_size_ == 1:
            self.ranks_ = _rank_limits(self.basis_sizes_, self.rank)
        else:
            full_ranks = _rank_limits(
                self.basis_sizes_ + (self.output_size_,), self.rank
            )
            self.ranks_ = full_ranks[:-1]

    def _basis(self, axis, values, *, extrapolate=False):
        values = np.asarray(values, dtype=float).reshape(-1)
        lower, upper = self.bounds_[axis]
        if not extrapolate:
            tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
            if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
                raise ValueError(f"coordinate {axis} lies outside the fitted bounds")
        scaled = 2.0 * (values - lower) / (upper - lower) - 1.0
        basis = self.bases_[axis]
        degree = self.degrees_[axis]
        if basis == "chebyshev":
            return np.polynomial.chebyshev.chebvander(scaled, degree)
        if basis == "legendre":
            return np.polynomial.legendre.legvander(scaled, degree)
        phase = np.pi * (scaled + 1.0)
        output = np.ones((len(values), 2 * degree + 1), dtype=float)
        for harmonic in range(1, degree + 1):
            output[:, 2 * harmonic - 1] = np.cos(harmonic * phase)
            output[:, 2 * harmonic] = np.sin(harmonic * phase)
        return output

    def _basis_matrices(self, coordinates, *, extrapolate=False):
        return tuple(
            self._basis(axis, coordinates[:, axis], extrapolate=extrapolate)
            for axis in range(self.ndim)
        )

    def _initialize(self):
        rng = np.random.default_rng(self.random_state)
        complex_output = np.issubdtype(self.dtype_, np.complexfloating)

        def random_normal(shape, scale):
            values = rng.normal(scale=scale, size=shape)
            if complex_output:
                values = values + 1j * rng.normal(scale=scale, size=shape)
            return values.astype(self.dtype_, copy=False)

        self.cores = []
        for axis, physical in enumerate(self.basis_sizes_):
            left, right = self.ranks_[axis : axis + 2]
            scale = 1.0 / np.sqrt(max(1, left * physical))
            self.cores.append(random_normal((left, physical, right), scale))
        output_rank = self.ranks_[-1]
        self.output_core = random_normal(
            (output_rank, self.channel_size_),
            1.0 / np.sqrt(max(1, output_rank)),
        )
        self._right_canonicalize_output()
        for axis in range(self.ndim - 1, 0, -1):
            self._right_canonicalize(axis)

    def _core_values(self, basis_matrices):
        return [
            np.einsum("np,apb->nab", basis, core, optimize=True)
            for basis, core in zip(basis_matrices, self.cores)
        ]

    def _contract(self, values):
        count = values[0].shape[0]
        dtype = np.result_type(*(core.dtype for core in values), self.output_core.dtype)
        output = np.ones((count, 1), dtype=dtype)
        for core in values:
            output = np.einsum("na,nab->nb", output, core, optimize=True)
        return output @ self.output_core

    def _predict_normalized(self, coordinates, *, extrapolate=False, batch_size=16384):
        """Contract coordinate samples without materializing rank-squared fields."""
        count = len(coordinates)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        dtype = np.result_type(
            *(core.dtype for core in self.cores),
            self.output_core.dtype,
        )
        result = np.empty((count, self.channel_size_), dtype=dtype)
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            block = coordinates[start:stop]
            state = np.ones((stop - start, 1), dtype=dtype)
            for axis, core in enumerate(self.cores):
                basis = self._basis(
                    axis,
                    block[:, axis],
                    extrapolate=extrapolate,
                )
                left, physical, right = core.shape
                temporary = (state @ core.reshape(left, physical * right)).reshape(
                    stop - start,
                    physical,
                    right,
                )
                state = np.einsum("np,npb->nb", basis, temporary, optimize=True)
            result[start:stop] = state @ self.output_core
        return result

    def _left_canonicalize(self, axis):
        self.cores[axis], center = left_qr(self.cores[axis])
        self.cores[axis + 1] = np.einsum(
            "ab,bpc->apc", center, self.cores[axis + 1], optimize=True
        )

    def _left_canonicalize_output(self):
        self.cores[-1], center = left_qr(self.cores[-1])
        self.output_core = center @ self.output_core

    def _right_canonicalize(self, axis):
        center, self.cores[axis] = right_rq(self.cores[axis])
        self.cores[axis - 1] = np.einsum(
            "apb,bc->apc", self.cores[axis - 1], center, optimize=True
        )

    def _right_canonicalize_output(self):
        q, center = np.linalg.qr(self.output_core.T, mode="reduced")
        self.output_core = q.T
        self.cores[-1] = np.einsum(
            "apb,bc->apc", self.cores[-1], center.T, optimize=True
        )

    def _solve_core(self, left, basis, right, target):
        n_samples = len(target)
        shape = (left.shape[1], basis.shape[1], right.shape[1])
        size = int(np.prod(shape))
        output_size = target.shape[1]
        dtype = np.result_type(left, basis, right, target)

        def matvec(vector):
            core = np.asarray(vector).reshape(shape)
            return np.einsum(
                "na,np,nbo,apb->no", left, basis, right, core, optimize=True
            ).reshape(-1)

        def rmatvec(vector):
            return np.einsum(
                "na,np,nbo,no->apb",
                left.conj(),
                basis.conj(),
                right.conj(),
                np.asarray(vector).reshape(n_samples, output_size),
                optimize=True,
            ).reshape(-1)

        operator = LinearOperator(
            (n_samples * output_size, size),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=dtype,
        )
        if size <= 2_000 and n_samples * output_size * size <= 2_000_000:
            design = np.einsum(
                "na,np,nbo->noapb", left, basis, right, optimize=True
            ).reshape(n_samples * output_size, size)
            rhs = target.reshape(-1)
            if self.regularization:
                design = np.vstack(
                    (
                        design,
                        np.sqrt(self.regularization) * np.eye(size, dtype=dtype),
                    )
                )
                rhs = np.concatenate((rhs, np.zeros(size, dtype=dtype)))
            solution = np.linalg.lstsq(design, rhs, rcond=None)[0]
            return solution.reshape(shape)
        solution = lsmr(
            operator,
            target.reshape(-1),
            damp=np.sqrt(self.regularization),
            atol=self.local_rtol,
            btol=self.local_rtol,
            maxiter=self.local_maxiter,
        )[0]
        return solution.reshape(shape)

    def _solve_output(self, latent, target):
        dtype = np.result_type(latent, target)
        design = latent
        if self.regularization:
            rank = latent.shape[1]
            design = np.vstack(
                (latent, np.sqrt(self.regularization) * np.eye(rank, dtype=dtype))
            )
            target = np.vstack(
                (target, np.zeros((rank, target.shape[1]), dtype=dtype))
            )
        self.output_core = np.linalg.lstsq(design, target, rcond=None)[0]

    def _sweep(self, basis_matrices, target):
        values = self._core_values(basis_matrices)
        n_samples = len(target)
        right = [None] * (self.ndim + 1)
        right[self.ndim] = np.broadcast_to(
            self.output_core, (n_samples,) + self.output_core.shape
        )
        for axis in range(self.ndim - 1, -1, -1):
            right[axis] = np.einsum(
                "nab,nbo->nao", values[axis], right[axis + 1], optimize=True
            )

        left = np.ones((n_samples, 1), dtype=self.dtype_)
        for axis in range(self.ndim):
            self.cores[axis] = self._solve_core(
                left, basis_matrices[axis], right[axis + 1], target
            )
            if axis < self.ndim - 1:
                self._left_canonicalize(axis)
            else:
                self._left_canonicalize_output()
            current = np.einsum(
                "np,apb->nab", basis_matrices[axis], self.cores[axis], optimize=True
            )
            left = np.einsum("na,nab->nb", left, current, optimize=True)
        self._solve_output(left, target)

        values = self._core_values(basis_matrices)
        left = [None] * (self.ndim + 1)
        left[0] = np.ones((n_samples, 1), dtype=self.dtype_)
        for axis in range(self.ndim):
            left[axis + 1] = np.einsum(
                "na,nab->nb", left[axis], values[axis], optimize=True
            )

        right_value = np.broadcast_to(
            self.output_core, (n_samples,) + self.output_core.shape
        )
        for axis in range(self.ndim - 1, -1, -1):
            self.cores[axis] = self._solve_core(
                left[axis], basis_matrices[axis], right_value, target
            )
            if axis > 0:
                self._right_canonicalize(axis)
            current = np.einsum(
                "np,apb->nab", basis_matrices[axis], self.cores[axis], optimize=True
            )
            right_value = np.einsum(
                "nab,nbo->nao", current, right_value, optimize=True
            )

    @staticmethod
    def _relative_error(predicted, target):
        scale = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
        return float(np.linalg.norm(predicted - target) / scale)

    @staticmethod
    def _flatten_values(values, n_samples, *, output_shape=None, name="values"):
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != n_samples:
            raise ValueError(f"{name} must have one value per coordinate sample")
        shape = values.shape[1:]
        if output_shape is not None and shape != tuple(output_shape):
            raise ValueError(f"{name} have the wrong output shape")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
        return values, values.reshape(n_samples, -1), shape

    @staticmethod
    def _hermitian_error(values):
        adjoint = values.conj().swapaxes(-1, -2)
        scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
        return float(np.linalg.norm(values - adjoint) / scale)

    def _prepare_hermitian_values(self, values, *, resolve=False, name="values"):
        values = np.asarray(values)
        square = (
            values.ndim >= 3
            and values.shape[-2] == values.shape[-1]
            and values.shape[1:-2] == ()
        )
        if resolve:
            if self.hermitian == "auto":
                self.hermitian_ = square and self._hermitian_error(values) <= 1.0e-10
            else:
                self.hermitian_ = bool(self.hermitian)
        if not self.hermitian_:
            return values
        if not square:
            raise ValueError("Hermitian output requires square matrix values")
        error = self._hermitian_error(values)
        if error > 1.0e-10:
            raise ValueError(f"{name} are not Hermitian (relative error {error:.3e})")
        return 0.5 * (values + values.conj().swapaxes(-1, -2))

    def _configure_output_basis(self):
        self.channel_size_ = self.output_size_
        self.output_basis_ = None
        if not self.hermitian_:
            return
        self.output_basis_ = hermitian_basis(self.output_shape_[0])

    def _encode_values(self, values):
        values = np.asarray(values)
        if not self.hermitian_:
            return values.reshape(len(values), self.output_size_)
        return pack_hermitian(values, self.output_basis_).reshape(
            len(values), self.channel_size_
        )

    def _decode_channels(self, channels):
        channels = np.asarray(channels)
        if self.hermitian_:
            return channels @ self.output_basis_
        return channels

    def _set_normalization(self, offset, scale):
        self.offset_ = np.asarray(offset).reshape(self.channel_size_)
        self.scale_ = np.asarray(scale, dtype=float).reshape(self.channel_size_)
        physical_offset = self._decode_channels(self.offset_[None, :])[0]
        self.offset = (
            physical_offset.reshape(self.output_shape_)
            if self.output_shape_
            else physical_offset[0].item()
        )
        self.scale = (
            self.scale_.reshape(self.output_shape_)
            if self.output_shape_
            else float(self.scale_[0])
        )

    def _normalization(self, values):
        offset = np.mean(values, axis=0)
        centered = values - offset
        if self.normalization == "elementwise":
            scale = np.std(values, axis=0)
            magnitude = np.max(np.abs(values), axis=0)
            tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, magnitude)
            scale = np.where(scale > tolerance, scale, 0.0)
        else:
            frobenius_scale = float(
                np.sqrt(np.mean(np.sum(np.abs(centered) ** 2, axis=1)))
            )
            magnitude = float(np.max(np.abs(values)))
            tolerance = 64.0 * np.finfo(float).eps * max(1.0, magnitude)
            if frobenius_scale <= tolerance:
                frobenius_scale = 0.0
            scale = np.full(self.channel_size_, frobenius_scale)
        return offset, scale

    def _normalization_vectors(self):
        return self.offset_, self.scale_

    def _denormalize(self, normalized):
        offset, scale = self._normalization_vectors()
        channels = offset + scale * normalized
        return self._decode_channels(channels)

    def fit(self, coordinates, values, *, validation=None):
        """Fit the functional TT to scattered scalar, vector, or matrix values."""
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or len(coordinates) == 0:
            raise ValueError("coordinates must have shape (n_samples, n_coordinates)")
        values, _, output_shape = self._flatten_values(
            values, len(coordinates)
        )
        values = self._prepare_hermitian_values(values, resolve=True)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("training data must be finite")
        self._configure(coordinates, output_shape)
        flat_values = self._encode_values(values)
        self.dtype_ = np.result_type(flat_values.dtype, np.float64)
        flat_values = flat_values.astype(self.dtype_, copy=False)
        basis_matrices = self._basis_matrices(coordinates)
        offset, scale = self._normalization(flat_values)
        self._set_normalization(offset, scale)
        self._initialize()
        self.history = []

        if not np.any(scale):
            self.cores = [
                np.zeros((1, size, 1), dtype=self.dtype_)
                for size in self.basis_sizes_
            ]
            self.output_core = np.zeros((1, self.channel_size_), dtype=self.dtype_)
            self.ranks_ = (1,) * (self.ndim + 1)
            self.error = 0.0
            self.validation_error = 0.0 if validation is not None else None
            self.n_sweeps = 0
            self.success = True
            self.message = "constant function"
            return self

        safe_scale = np.where(scale > 0.0, scale, 1.0)
        target = (flat_values - offset) / safe_scale
        validation_basis = validation_target = None
        if validation is not None:
            validation_coordinates, validation_values = validation
            validation_coordinates = np.asarray(validation_coordinates, dtype=float)
            if (
                validation_coordinates.ndim != 2
                or validation_coordinates.shape[1] != self.ndim
            ):
                raise ValueError("validation coordinates have the wrong shape")
            validation_values, _, _ = self._flatten_values(
                validation_values,
                len(validation_coordinates),
                output_shape=self.output_shape_,
                name="validation values",
            )
            validation_values = self._prepare_hermitian_values(
                validation_values,
                name="validation values",
            )
            validation_flat = self._encode_values(validation_values)
            if not np.all(np.isfinite(validation_coordinates)):
                raise ValueError("validation data must be finite")
            validation_basis = self._basis_matrices(validation_coordinates)
            validation_target = (
                validation_flat.astype(self.dtype_, copy=False) - offset
            ) / safe_scale

        best_error = np.inf
        best_cores = None
        best_output_core = None
        stalled = 0
        previous = np.inf
        for sweep in range(1, self.sweeps + 1):
            self._sweep(basis_matrices, target)
            train_error = self._relative_error(
                self._contract(self._core_values(basis_matrices)), target
            )
            validation_error = None
            score = train_error
            if validation_basis is not None:
                validation_error = self._relative_error(
                    self._contract(self._core_values(validation_basis)),
                    validation_target,
                )
                score = validation_error
            self.history.append(
                {
                    "sweep": sweep,
                    "train_error": train_error,
                    "validation_error": validation_error,
                }
            )
            if score < best_error:
                best_error = score
                best_cores = [core.copy() for core in self.cores]
                best_output_core = self.output_core.copy()
            improvement = (previous - score) / max(abs(previous), 1.0)
            stalled = stalled + 1 if improvement <= self.rtol else 0
            previous = score
            if score <= self.rtol or stalled >= self.patience:
                break

        self.cores = best_cores
        self.output_core = best_output_core
        final_train = self._relative_error(
            self._contract(self._core_values(basis_matrices)), target
        )
        self.error = final_train
        self.validation_error = best_error if validation_basis is not None else None
        self.n_sweeps = len(self.history)
        self.success = np.isfinite(best_error)
        self.message = (
            "converged" if best_error <= self.rtol else "completed ALS sweeps"
        )
        self.ranks_ = tuple([1] + [core.shape[2] for core in self.cores])
        return self

    def _initialize_from_tensor_cores(self, grids, cores):
        if len(cores) != self.ndim + 1:
            raise ValueError("tensor cores must include one terminal output site")
        functional = []
        previous_right = 1
        for axis, (grid, core) in enumerate(zip(grids, cores[:-1])):
            core = np.asarray(core, dtype=self.dtype_)
            if core.ndim != 3 or core.shape[1] != len(grid):
                raise ValueError("each grid core must have shape (left, grid, right)")
            if core.shape[0] != previous_right:
                raise ValueError("grid-core virtual dimensions are incompatible")
            basis = self._basis(axis, grid)
            sampled = core.transpose(1, 0, 2).reshape(len(grid), -1)
            coefficients = np.linalg.lstsq(basis, sampled, rcond=None)[0]
            functional.append(
                coefficients.reshape(basis.shape[1], core.shape[0], core.shape[2])
                .transpose(1, 0, 2)
                .copy()
            )
            previous_right = core.shape[2]
        terminal = np.asarray(cores[-1], dtype=self.dtype_)
        if terminal.shape != (previous_right, self.channel_size_, 1):
            raise ValueError("the terminal core has an incompatible output shape")
        self.cores = functional
        self.output_core = terminal[:, :, 0].copy()
        self._right_canonicalize_output()
        for axis in range(self.ndim - 1, 0, -1):
            self._right_canonicalize(axis)
        self.ranks_ = tuple([1] + [core.shape[2] for core in self.cores])

    def fit_cores(self, grids, cores, output_shape):
        """Build a functional model from discrete TT cores and grid axes."""
        grids = tuple(
            np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
        )
        if not grids or any(grid.ndim != 1 or len(grid) < 2 for grid in grids):
            raise ValueError("grids must contain one-dimensional coordinate arrays")
        if self.hermitian == "auto":
            raise ValueError("fit_cores requires an explicit hermitian setting")
        self.hermitian_ = bool(self.hermitian)
        output_shape = tuple(int(size) for size in output_shape)
        if self.hermitian_ and (
            len(output_shape) != 2 or output_shape[0] != output_shape[1]
        ):
            raise ValueError("Hermitian output requires a square matrix shape")
        mesh = np.meshgrid(*grids, indexing="ij")
        coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
        self._configure(coordinates, output_shape)
        if any(
            basis_size > len(grid)
            for basis_size, grid in zip(self.basis_sizes_, grids)
        ):
            raise ValueError("functional basis size cannot exceed its grid size")
        cores = tuple(np.asarray(core) for core in cores)
        dtype = np.result_type(*(core.dtype for core in cores), np.float64)
        if self.hermitian_ and np.issubdtype(dtype, np.complexfloating):
            imaginary = max(float(np.max(np.abs(core.imag))) for core in cores)
            if imaginary > 1.0e-13:
                raise ValueError("packed Hermitian TT cores must be real")
            cores = tuple(core.real for core in cores)
            dtype = np.dtype(float)
        self.dtype_ = dtype
        self._set_normalization(
            np.zeros(self.channel_size_),
            np.ones(self.channel_size_),
        )
        self._initialize_from_tensor_cores(grids, cores)
        self.history = []
        self.error = np.nan
        self.validation_error = None
        self.n_sweeps = 0
        self.success = True
        self.message = "initialized from discrete TT cores"
        return self

    def fit_grid(self, grids, values):
        """Fit complete product-grid data by TT-SVD and basis interpolation."""
        from pyqed.mps.decompose import decompose

        grids = tuple(
            np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
        )
        if not grids or any(grid.ndim != 1 or len(grid) < 2 for grid in grids):
            raise ValueError("grids must contain one-dimensional coordinate arrays")
        if any(not np.all(np.isfinite(grid)) for grid in grids):
            raise ValueError("grid coordinates must be finite")
        grid_shape = tuple(len(grid) for grid in grids)
        values = np.asarray(values)
        if values.shape[: len(grids)] != grid_shape:
            raise ValueError("value leading dimensions must match the product grid")
        output_shape = values.shape[len(grids) :]
        n_samples = int(np.prod(grid_shape))
        sampled = values.reshape(n_samples, *output_shape)
        sampled, _, _ = self._flatten_values(sampled, n_samples)
        sampled = self._prepare_hermitian_values(sampled, resolve=True)
        mesh = np.meshgrid(*grids, indexing="ij")
        coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
        self._configure(coordinates, output_shape)
        if any(
            basis_size > len(grid)
            for basis_size, grid in zip(self.basis_sizes_, grids)
        ):
            raise ValueError("functional basis size cannot exceed its grid size")
        channels = self._encode_values(sampled)
        self.dtype_ = np.result_type(channels.dtype, np.float64)
        channels = channels.astype(self.dtype_, copy=False)
        offset, scale = self._normalization(channels)
        self._set_normalization(offset, scale)
        safe_scale = np.where(scale > 0.0, scale, 1.0)
        target = (channels - offset) / safe_scale
        self.history = []

        if not np.any(scale):
            self.cores = [
                np.zeros((1, size, 1), dtype=self.dtype_)
                for size in self.basis_sizes_
            ]
            self.output_core = np.zeros((1, self.channel_size_), dtype=self.dtype_)
            self.ranks_ = (1,) * (self.ndim + 1)
        else:
            tensor = target.reshape(*grid_shape, self.channel_size_)
            cores = decompose(tensor, rank=(*self.ranks_, 1))
            self._initialize_from_tensor_cores(grids, cores)

        predicted = self.predict(coordinates).reshape(n_samples, self.output_size_)
        physical = sampled.reshape(n_samples, self.output_size_)
        self.error = self._relative_error(predicted, physical)
        self.rmse = float(np.sqrt(np.mean(np.abs(predicted - physical) ** 2)))
        self.validation_error = None
        self.n_sweeps = 0
        self.success = np.isfinite(self.error)
        self.message = "TT-SVD grid fit"
        return self

    def fit_links(
        self,
        coordinates,
        pairs,
        links,
        *,
        penalty=10.0,
        smoothness=0.0,
        maxiter=500,
        gtol=1.0e-8,
    ):
        r"""Variationally refine a rectangular feature TT against overlap links.

        The fitted output must have shape ``(feature_rank, nstates)``.  ``pairs``
        contains integer row pairs into ``coordinates`` and the objective uses
        ``Y(left).H @ Y(right)`` as its overlap prediction.
        """
        from scipy.optimize import minimize

        if not hasattr(self, "cores") or len(self.output_shape_) != 2:
            raise RuntimeError("fit a rectangular feature model before fitting links")
        if self.hermitian_:
            raise ValueError("overlap features cannot use a Hermitian output basis")
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.ndim:
            raise ValueError("coordinates have the wrong shape")
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
            raise ValueError("pairs must have shape (nlinks, 2)")
        if np.any(pairs < 0) or np.any(pairs >= len(coordinates)):
            raise IndexError("link pair index is outside the coordinate samples")
        feature_rank, nstates = self.output_shape_
        links = np.asarray(links)
        if links.shape != (len(pairs), nstates, nstates):
            raise ValueError("links have the wrong shape")
        parameter_arrays = [self.offset_, *self.cores, self.output_core]
        target_scale = max(float(np.max(np.abs(links))), 1.0)
        real_problem = (
            float(np.max(np.abs(links.imag))) <= 1.0e-12 * target_scale
            and all(not np.iscomplexobj(array) for array in parameter_arrays)
        )
        links = links.real if real_problem else links.astype(complex, copy=False)
        penalty = float(penalty)
        smoothness = float(smoothness)
        if penalty < 0.0 or smoothness < 0.0:
            raise ValueError("penalty and smoothness must be nonnegative")

        basis_matrices = self._basis_matrices(coordinates)
        shapes = [self.offset_.shape, *(core.shape for core in self.cores), self.output_core.shape]
        sizes = [int(np.prod(shape)) for shape in shapes]
        total = sum(sizes)

        def pack(arrays):
            values = np.concatenate([np.asarray(array).reshape(-1) for array in arrays])
            return values if real_problem else np.concatenate((values.real, values.imag))

        def unpack(vector):
            values = vector if real_problem else vector[:total] + 1j * vector[total:]
            arrays = []
            start = 0
            for shape, size in zip(shapes, sizes):
                arrays.append(values[start : start + size].reshape(shape))
                start += size
            return arrays

        initial = pack([self.offset_, *self.cores, self.output_core])
        edge_scale = len(pairs)
        point_scale = max(len(coordinates), 1)
        scale = self.scale_[None, :]

        def objective(vector):
            offset, *parameters = unpack(vector)
            cores, output_core = parameters[:-1], parameters[-1]
            core_values = [
                np.einsum("np,apb->nab", basis, core, optimize=True)
                for basis, core in zip(basis_matrices, cores)
            ]
            states = [
                np.ones(
                    (len(coordinates), 1),
                    dtype=float if real_problem else complex,
                )
            ]
            for value in core_values:
                states.append(
                    np.einsum("na,nab->nb", states[-1], value, optimize=True)
                )
            normalized = states[-1] @ output_core
            channels = offset[None, :] + scale * normalized
            features = channels.reshape(len(coordinates), feature_rank, nstates)
            gradient_features = np.zeros_like(features)
            loss = 0.0
            for (left, right), target in zip(pairs, links):
                error = features[left].conj().T @ features[right] - target
                loss += float(np.vdot(error, error).real) / edge_scale
                gradient_features[left] += (
                    2.0 * features[right] @ error.conj().T / edge_scale
                )
                gradient_features[right] += (
                    2.0 * features[left] @ error / edge_scale
                )
                if smoothness:
                    difference = features[right] - features[left]
                    loss += (
                        smoothness
                        * float(np.vdot(difference, difference).real)
                        / edge_scale
                    )
                    gradient_features[left] -= (
                        2.0 * smoothness * difference / edge_scale
                    )
                    gradient_features[right] += (
                        2.0 * smoothness * difference / edge_scale
                    )
            identity = np.eye(nstates)
            for point, feature in enumerate(features):
                defect = feature.conj().T @ feature - identity
                loss += penalty * float(np.vdot(defect, defect).real) / point_scale
                gradient_features[point] += (
                    4.0 * penalty * feature @ defect / point_scale
                )

            gradient_channels = gradient_features.reshape(len(coordinates), -1)
            gradient_offset = np.sum(gradient_channels, axis=0)
            gradient_normalized = gradient_channels * scale
            gradient_output = states[-1].conj().T @ gradient_normalized
            gradient_state = gradient_normalized @ output_core.conj().T
            gradient_cores = [None] * len(cores)
            for axis in range(len(cores) - 1, -1, -1):
                gradient_values = np.einsum(
                    "na,nb->nab",
                    states[axis].conj(),
                    gradient_state,
                    optimize=True,
                )
                gradient_cores[axis] = np.einsum(
                    "np,nab->apb",
                    basis_matrices[axis],
                    gradient_values,
                    optimize=True,
                )
                gradient_state = np.einsum(
                    "nb,nab->na",
                    gradient_state,
                    core_values[axis].conj(),
                    optimize=True,
                )
            gradient = pack(
                [gradient_offset, *gradient_cores, gradient_output]
            )
            return loss, gradient

        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": 1.0e-14},
        )
        offset, *parameters = unpack(result.x)
        self.offset_ = offset
        self.cores = [np.asarray(core) for core in parameters[:-1]]
        self.output_core = np.asarray(parameters[-1])
        self._set_normalization(self.offset_, self.scale_)
        predicted = np.asarray(self.predict(coordinates))
        predicted_links = np.asarray(
            [predicted[left].conj().T @ predicted[right] for left, right in pairs]
        )
        errors = np.asarray(
            [
                np.linalg.norm(value - target)
                / max(np.linalg.norm(target), np.finfo(float).tiny)
                for value, target in zip(predicted_links, links)
            ]
        )
        orthogonality = np.linalg.norm(
            predicted.conj().swapaxes(-1, -2) @ predicted - np.eye(nstates),
            axis=(-2, -1),
        )
        self.link_info = {
            "backend": "variational-functional-tt-links",
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "maximum_relative_link_error": float(np.max(errors)),
            "rms_relative_link_error": float(np.sqrt(np.mean(errors**2))),
            "maximum_orthogonality_defect": float(np.max(orthogonality)),
            "penalty": penalty,
            "smoothness": smoothness,
            "real_valued": bool(real_problem),
        }
        return self

    def _initialize_from_grid_cores(self, grids, cores):
        grids = tuple(
            np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
        )
        cores = tuple(np.asarray(core, dtype=self.dtype_) for core in cores)
        if len(grids) != self.ndim or len(cores) != self.ndim:
            raise ValueError(
                "cross grids and cores must match the functional dimension"
            )
        functional = []
        previous_right = 1
        for axis, (grid, core) in enumerate(zip(grids, cores)):
            if core.ndim != 3 or core.shape[1] != len(grid):
                raise ValueError("each cross core must have shape (left, grid, right)")
            if core.shape[0] != previous_right:
                raise ValueError("cross-core virtual dimensions are incompatible")
            basis = self._basis(axis, grid)
            sampled = core.transpose(1, 0, 2).reshape(len(grid), -1)
            coefficients = np.linalg.lstsq(basis, sampled, rcond=None)[0]
            functional.append(
                coefficients.reshape(basis.shape[1], core.shape[0], core.shape[2])
                .transpose(1, 0, 2)
                .copy()
            )
            previous_right = core.shape[2]
        if functional[0].shape[0] != 1 or functional[-1].shape[2] != 1:
            raise ValueError("cross cores must have unit boundary ranks")
        self.cores = functional
        self.output_core = np.ones((1, 1), dtype=self.dtype_)
        for axis in range(self.ndim - 1, 0, -1):
            self._right_canonicalize(axis)
        self.ranks_ = tuple([1] + [core.shape[2] for core in self.cores])

    def refine(
        self,
        coordinates,
        values,
        *,
        validation=None,
        sweeps=50,
        rtol=1.0e-8,
        atol=0.0,
        patience=3,
    ):
        """Optimize functional cores with canonical alternating least squares."""
        self._check_fit()
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.ndim:
            raise ValueError("coordinates have the wrong shape")
        if len(coordinates) == 0:
            raise ValueError("refinement coordinates cannot be empty")
        values, _, _ = self._flatten_values(
            values,
            len(coordinates),
            output_shape=self.output_shape_,
        )
        values = self._prepare_hermitian_values(values)
        physical_values = values.reshape(len(coordinates), self.output_size_)
        flat_values = self._encode_values(values)
        flat_values = flat_values.astype(self.dtype_, copy=False)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("refinement data must be finite")
        sweeps = int(sweeps)
        patience = int(patience)
        rtol = float(rtol)
        atol = float(atol)
        if sweeps < 1 or patience < 1 or rtol < 0.0 or atol < 0.0:
            raise ValueError(
                "sweeps and patience must be positive; tolerances nonnegative"
            )
        basis_matrices = self._basis_matrices(coordinates)
        offset, output_scale = self._normalization_vectors()
        safe_scale = np.where(output_scale > 0.0, output_scale, 1.0)
        target = (flat_values - offset) / safe_scale
        scale = max(
            float(
                np.linalg.norm(
                    physical_values - np.mean(physical_values, axis=0)
                )
            ),
            np.finfo(float).tiny,
        )

        validation_basis = validation_physical = validation_scale = None
        if validation is not None:
            validation_coordinates, validation_values = validation
            validation_coordinates = np.asarray(validation_coordinates, dtype=float)
            if (
                validation_coordinates.ndim != 2
                or validation_coordinates.shape[1] != self.ndim
            ):
                raise ValueError("validation data have incompatible shapes")
            validation_values, _, _ = self._flatten_values(
                validation_values,
                len(validation_coordinates),
                output_shape=self.output_shape_,
                name="validation values",
            )
            validation_values = self._prepare_hermitian_values(
                validation_values,
                name="validation values",
            )
            validation_physical = validation_values.reshape(
                len(validation_coordinates), self.output_size_
            )
            validation_basis = self._basis_matrices(validation_coordinates)
            validation_scale = max(
                float(
                    np.linalg.norm(
                        validation_physical
                        - np.mean(validation_physical, axis=0)
                    )
                ),
                np.finfo(float).tiny,
            )

        def errors():
            predicted = self._denormalize(
                self._contract(self._core_values(basis_matrices))
            )
            train = float(np.linalg.norm(predicted - physical_values) / scale)
            valid = None
            if validation_basis is not None:
                validation_prediction = self._denormalize(
                    self._contract(self._core_values(validation_basis))
                )
                valid = float(
                    np.linalg.norm(validation_prediction - validation_physical)
                    / validation_scale
                )
            return train, valid

        initial_error, initial_validation_error = errors()
        best_score = initial_error if validation is None else initial_validation_error
        best_cores = [core.copy() for core in self.cores]
        best_output_core = self.output_core.copy()
        previous = best_score
        stalled = 0
        worsening = 0
        converged = best_score <= atol
        failed = False
        self.history = []
        for sweep in range(1, sweeps + 1):
            if converged:
                break
            self._sweep(basis_matrices, target)
            train_error, validation_error = errors()
            score = train_error if validation is None else validation_error
            relative_improvement = (previous - score) / max(
                abs(previous), np.finfo(float).tiny
            )
            self.history.append(
                {
                    "sweep": sweep,
                    "train_error": train_error,
                    "validation_error": validation_error,
                    "relative_improvement": relative_improvement,
                }
            )
            if score < best_score:
                best_score = score
                best_cores = [core.copy() for core in self.cores]
                best_output_core = self.output_core.copy()
            stalled = stalled + 1 if abs(relative_improvement) <= rtol else 0
            worsening = worsening + 1 if relative_improvement < -rtol else 0
            previous = score
            converged = score <= atol or stalled >= patience
            failed = worsening >= patience
            if failed:
                break

        self.cores = best_cores
        self.output_core = best_output_core
        self.error, self.validation_error = errors()
        residual = (
            np.asarray(self.predict(coordinates)).reshape(physical_values.shape)
            - physical_values
        )
        self.rmse = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
        if converged:
            message = "converged"
        elif failed:
            message = "objective increased in consecutive sweeps"
        else:
            message = "maximum sweeps reached"
        self.refinement = {
            "method": "canonical_als",
            "initial_error": initial_error,
            "final_error": self.error,
            "initial_validation_error": initial_validation_error,
            "final_validation_error": self.validation_error,
            "sweeps": len(self.history),
            "converged": converged,
            "message": message,
        }
        self.n_sweeps = len(self.history)
        self.ranks_ = tuple([1] + [core.shape[2] for core in self.cores])
        self.success = converged
        self.message = message
        return self

    def fit_from_cross(
        self,
        grids,
        cores,
        coordinates,
        values,
        *,
        validation=None,
        sweeps=50,
        **kwargs,
    ):
        """Initialize from discrete TT-cross cores and refine continuously."""
        coordinates = np.asarray(coordinates, dtype=float)
        values = np.asarray(values, dtype=float)
        if coordinates.ndim != 2 or len(coordinates) == 0:
            raise ValueError("coordinates must have shape (n_samples, n_coordinates)")
        if values.shape != (len(coordinates),):
            raise ValueError("values must contain one scalar per coordinate sample")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(values)):
            raise ValueError("training data must be finite")
        self.dtype_ = np.result_type(values.dtype, np.float64)
        self.hermitian_ = False
        self._configure(coordinates, ())
        self._set_normalization(np.zeros(1), np.ones(1))
        self.history = []
        self._initialize_from_grid_cores(grids, cores)
        return self.refine(
            coordinates,
            values,
            validation=validation,
            sweeps=sweeps,
            **kwargs,
        )

    def _check_fit(self):
        if not hasattr(self, "cores"):
            raise RuntimeError("FunctionalTT has not been fitted")

    def predict(self, coordinates, *, extrapolate=False, batch_size=16384):
        """Evaluate the fitted function at arbitrary coordinates."""
        self._check_fit()
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 1:
            if coordinates.shape != (self.ndim,):
                raise ValueError("coordinate point has the wrong dimension")
            leading = ()
            flat = coordinates.reshape(1, self.ndim)
        else:
            if coordinates.shape[-1] != self.ndim:
                raise ValueError("coordinates have the wrong trailing dimension")
            leading = coordinates.shape[:-1]
            flat = coordinates.reshape(-1, self.ndim)
        normalized = self._predict_normalized(
            flat,
            extrapolate=extrapolate,
            batch_size=batch_size,
        )
        output = self._denormalize(normalized)
        shape = leading + self.output_shape_
        output = output.reshape(shape)
        return output.item() if not shape else output

    __call__ = predict

    def realify_features(self):
        r"""Return an exactly equivalent real feature model.

        For a rectangular feature field ``Y = A + i B``, the returned model
        predicts ``[A; B]``.  Its real Gram matrix is therefore
        ``A.T @ A + B.T @ B = Re(Y.H @ Y)`` at every coordinate pair.
        """
        self._check_fit()
        if self.hermitian_ or len(self.output_shape_) != 2:
            raise ValueError("realification requires a rectangular feature output")
        feature_rank, nstates = self.output_shape_
        model = copy.deepcopy(self)
        converted = []
        for axis, core in enumerate(self.cores):
            real = np.asarray(core.real)
            imag = np.asarray(core.imag)
            if axis == 0:
                converted.append(np.concatenate((real, imag), axis=2))
            else:
                top = np.concatenate((real, imag), axis=2)
                bottom = np.concatenate((-imag, real), axis=2)
                converted.append(np.concatenate((top, bottom), axis=0))
        output_real = np.asarray(self.output_core.real)
        output_imag = np.asarray(self.output_core.imag)
        model.output_core = np.concatenate(
            (
                np.concatenate((output_real, output_imag), axis=1),
                np.concatenate((-output_imag, output_real), axis=1),
            ),
            axis=0,
        )
        model.cores = converted
        model.output_shape_ = (2 * feature_rank, nstates)
        model.output_size_ = 2 * self.output_size_
        model.hermitian = False
        model.hermitian_ = False
        model._configure_output_basis()
        model.ranks_ = tuple([1] + [core.shape[2] for core in model.cores])
        model.dtype_ = np.dtype(float)
        model._set_normalization(
            np.concatenate((self.offset_.real, self.offset_.imag)),
            np.concatenate((self.scale_, self.scale_)),
        )
        model.message = "exact realification of complex feature TT"
        return model

    @staticmethod
    def _sum_tensor_trains(left_tt, right_tt):
        if len(left_tt) != len(right_tt):
            raise ValueError("tensor trains must have the same number of sites")
        if len(left_tt) == 1:
            return [left_tt[0] + right_tt[0]]
        result = [np.concatenate((left_tt[0], right_tt[0]), axis=2)]
        for left, right in zip(left_tt[1:-1], right_tt[1:-1]):
            left_rank, physical, right_rank = left.shape
            other_left, other_physical, other_right = right.shape
            if physical != other_physical:
                raise ValueError("tensor-train physical dimensions do not match")
            core = np.zeros(
                (left_rank + other_left, physical, right_rank + other_right),
                dtype=np.result_type(left, right),
            )
            core[:left_rank, :, :right_rank] = left
            core[left_rank:, :, right_rank:] = right
            result.append(core)
        result.append(np.concatenate((left_tt[-1], right_tt[-1]), axis=0))
        return result

    def tensor_cores(self, grids):
        """Evaluate a tensor-valued model as a discrete TT with an output site."""
        self._check_fit()
        grids = tuple(
            np.asarray(getattr(grid, "x", grid), dtype=float) for grid in grids
        )
        if len(grids) != self.ndim:
            raise ValueError("the grid count must match the functional TT dimension")
        coordinate_cores = [
            np.einsum(
                "ip,apb->aib",
                self._basis(axis, grid),
                self.cores[axis],
                optimize=True,
            )
            for axis, grid in enumerate(grids)
        ]
        offset, scale = self._normalization_vectors()
        output_core = self._decode_channels(self.output_core * scale[None, :])
        fitted = coordinate_cores + [output_core[:, :, None]]
        if np.any(offset):
            constant = [
                np.ones((1, len(grid), 1), dtype=self.dtype_) for grid in grids
            ]
            physical_offset = self._decode_channels(offset[None, :])
            constant.append(physical_offset.reshape(1, self.output_size_, 1))
            fitted = self._sum_tensor_trains(fitted, constant)
        return fitted

    def grid_cores(self, grids):
        """Evaluate the model on product-grid axes.

        Tensor-valued models include a final site whose physical index is the
        flattened output.  For scalar models that unit site is absorbed into
        the last coordinate core, preserving the usual scalar-grid TT form.
        """
        cores = self.tensor_cores(grids)
        if self.output_shape_:
            return cores
        last = np.einsum("apb,bqc->apqc", cores[-2], cores[-1], optimize=True)
        cores[-2] = last[:, :, 0, :]
        return cores[:-1]

    def mpo(self, dvrs):
        """Evaluate a scalar or square-matrix field as a diagonal DVR MPO."""
        self._check_fit()
        from pyqed.mps.mps import MPO

        if self.output_shape_ and (
            len(self.output_shape_) != 2
            or self.output_shape_[0] != self.output_shape_[1]
        ):
            raise ValueError("MPO output must be scalar or a square matrix")
        if self.output_shape_:
            cores = self.tensor_cores(dvrs)
            coordinate_cores = cores[:-1]
        else:
            cores = None
            coordinate_cores = self.grid_cores(dvrs)
        factors = []
        for core in coordinate_cores:
            left, physical, right = core.shape
            factor = np.zeros((left, right, physical, physical), dtype=core.dtype)
            diagonal = np.arange(physical)
            factor[:, :, diagonal, diagonal] = core.transpose(0, 2, 1)
            factors.append(factor)
        if self.output_shape_:
            size = self.output_shape_[0]
            output = np.asarray(cores[-1])
            left, physical, right = output.shape
            if physical != size * size or right != 1:
                raise ValueError("functional output core has an incompatible shape")
            factors.append(
                output.reshape(left, size, size, 1).transpose(0, 3, 1, 2)
            )
        return MPO(factors)

    def save(self, filename):
        """Save the fitted model to an ``npz`` archive."""
        self._check_fit()
        rank = (
            int(self.rank)
            if np.isscalar(self.rank)
            else [int(value) for value in self.rank]
        )
        config = {
            "bases": self.bases_,
            "degrees": self.degrees_,
            "rank": rank,
            "normalization": self.normalization,
            "hermitian": self.hermitian,
            "regularization": self.regularization,
            "sweeps": self.sweeps,
            "rtol": self.rtol,
            "local_rtol": self.local_rtol,
            "local_maxiter": self.local_maxiter,
            "patience": self.patience,
            "random_state": self.random_state,
        }
        data = {
            "model_class": np.asarray("FunctionalTT"),
            "config": json.dumps(config),
            "bounds": self.bounds_,
            "offset": self.offset_,
            "scale": self.scale_,
            "output_shape": np.asarray(self.output_shape_, dtype=int),
            "output_core": self.output_core,
            "hermitian_resolved": np.asarray(self.hermitian_),
            "hermitian_packed": np.asarray(self.hermitian_),
        }
        data.update({f"core_{axis}": core for axis, core in enumerate(self.cores)})
        np.savez(Path(filename), **data)

    @classmethod
    def load(cls, filename):
        """Load a fitted model from an ``npz`` archive."""
        with np.load(Path(filename), allow_pickle=False) as data:
            config = json.loads(str(data["config"]))
            model = cls(bounds=np.asarray(data["bounds"]), **config)
            model.ndim = len(config["bases"])
            model.bases_ = tuple(config["bases"])
            model.degrees_ = tuple(int(value) for value in config["degrees"])
            model.bounds_ = np.asarray(data["bounds"], dtype=float)
            model.basis_sizes_ = tuple(
                2 * degree + 1 if basis == "fourier" else degree + 1
                for basis, degree in zip(model.bases_, model.degrees_)
            )
            model.cores = [
                np.asarray(data[f"core_{axis}"]) for axis in range(model.ndim)
            ]
            model.ranks_ = tuple([1] + [core.shape[2] for core in model.cores])
            if "output_shape" in data:
                model.output_shape_ = tuple(
                    int(size) for size in np.asarray(data["output_shape"])
                )
                model.output_core = np.asarray(data["output_core"])
            else:
                model.output_shape_ = ()
                model.output_core = np.ones((1, 1), dtype=model.cores[0].dtype)
            model.output_size_ = (
                int(np.prod(model.output_shape_, dtype=int))
                if model.output_shape_
                else 1
            )
            model.hermitian_ = (
                bool(data["hermitian_resolved"])
                if "hermitian_resolved" in data
                else False
            )
            if model.hermitian_ and (
                "hermitian_packed" not in data or not bool(data["hermitian_packed"])
            ):
                raise ValueError(
                    "the archive uses the obsolete projected Hermitian representation"
                )
            model._configure_output_basis()
            offset = np.asarray(data["offset"])
            scale = np.asarray(data["scale"])
            model._set_normalization(offset, scale)
            model.dtype_ = np.result_type(
                *(core.dtype for core in model.cores), model.output_core.dtype
            )
        model.history = []
        model.error = np.nan
        model.validation_error = None
        model.n_sweeps = 0
        model.success = True
        model.message = "loaded"
        return model


class PiecewisePCHIP:
    """Shape-preserving one-dimensional interpolation of tensor fields.

    The interpolant acts elementwise, while optional Hermitian packing is
    enforced on both the input nodes and the evaluated matrix field.  It is
    intended for strongly nonuniform reaction-coordinate grids where a global
    polynomial can oscillate badly between otherwise smooth samples.
    """

    def __init__(self, *, hermitian="auto"):
        if hermitian != "auto" and not isinstance(hermitian, (bool, np.bool_)):
            raise ValueError("hermitian must be 'auto', True, or False")
        self.hermitian = hermitian if hermitian == "auto" else bool(hermitian)

    def fit(self, coordinates, values):
        """Interpolate values tabulated at ordered one-dimensional nodes."""
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 2 and coordinates.shape[1] == 1:
            coordinates = coordinates[:, 0]
        if coordinates.ndim != 1 or coordinates.size < 2:
            raise ValueError("coordinates must contain at least two 1D nodes")
        if not np.all(np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0.0):
            raise ValueError("coordinates must be finite and strictly increasing")
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != coordinates.size:
            raise ValueError("values must contain one tensor per coordinate")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")

        output_shape = values.shape[1:]
        square = (
            len(output_shape) == 2 and output_shape[0] == output_shape[1]
        )
        if self.hermitian == "auto":
            self.hermitian_ = square and FunctionalTT._hermitian_error(values) <= 1.0e-10
        else:
            self.hermitian_ = bool(self.hermitian)
        if self.hermitian_:
            if not square:
                raise ValueError("Hermitian output requires square matrix values")
            error = FunctionalTT._hermitian_error(values)
            if error > 1.0e-10:
                raise ValueError(f"values are not Hermitian (relative error {error:.3e})")
            values = 0.5 * (values + values.conj().swapaxes(-1, -2))

        self.ndim = 1
        self.coordinates_ = coordinates.copy()
        self.values_ = values.copy()
        self.output_shape_ = output_shape
        self.output_size_ = int(np.prod(output_shape, dtype=int)) if output_shape else 1
        self.bounds_ = np.asarray(((coordinates[0], coordinates[-1]),), dtype=float)
        self.bounds = tuple(map(tuple, self.bounds_))
        self.dtype_ = np.result_type(values.dtype, np.float64)
        self._build()
        self.error = 0.0
        self.rmse = 0.0
        self.validation_error = None
        self.success = True
        self.message = "shape-preserving piecewise-cubic interpolation"
        return self

    def fit_grid(self, grids, values):
        """Fit a single complete one-dimensional grid."""
        grids = tuple(grids)
        if len(grids) != 1:
            raise ValueError("PiecewisePCHIP supports exactly one coordinate")
        return self.fit(grids[0], values)

    def _build(self):
        from scipy.interpolate import PchipInterpolator

        self._real_interpolator = PchipInterpolator(
            self.coordinates_, self.values_.real, axis=0, extrapolate=True
        )
        self._imag_interpolator = (
            None
            if not np.iscomplexobj(self.values_)
            else PchipInterpolator(
                self.coordinates_, self.values_.imag, axis=0, extrapolate=True
            )
        )

    def _check_fit(self):
        if not hasattr(self, "coordinates_"):
            raise RuntimeError("PiecewisePCHIP has not been fitted")

    def predict(self, coordinates, *, extrapolate=False):
        """Evaluate the interpolant at one or more coordinates."""
        self._check_fit()
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim == 1:
            if coordinates.shape != (1,):
                raise ValueError("coordinate point has the wrong dimension")
            leading = ()
            flat = coordinates
        else:
            if coordinates.shape[-1] != 1:
                raise ValueError("coordinates have the wrong trailing dimension")
            leading = coordinates.shape[:-1]
            flat = coordinates.reshape(-1)
        if not np.all(np.isfinite(flat)):
            raise ValueError("coordinates must be finite")
        lower, upper = self.bounds_[0]
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
        if not extrapolate and (
            np.any(flat < lower - tolerance) or np.any(flat > upper + tolerance)
        ):
            raise ValueError("coordinate 0 lies outside the fitted bounds")

        values = self._real_interpolator(flat)
        if self._imag_interpolator is not None:
            values = values + 1j * self._imag_interpolator(flat)
        values = np.asarray(values).reshape(leading + self.output_shape_)
        if self.hermitian_:
            values = 0.5 * (values + values.conj().swapaxes(-1, -2))
        return values.item() if not values.shape else values

    __call__ = predict

    def save(self, filename):
        """Save the interpolation nodes and values to an ``npz`` archive."""
        self._check_fit()
        np.savez(
            Path(filename),
            model_class=np.asarray("PiecewisePCHIP"),
            coordinates=self.coordinates_,
            values=self.values_,
            hermitian_resolved=np.asarray(self.hermitian_),
        )

    @classmethod
    def load(cls, filename):
        """Restore a saved shape-preserving interpolant."""
        with np.load(Path(filename), allow_pickle=False) as data:
            hermitian = bool(data["hermitian_resolved"])
            coordinates = np.asarray(data["coordinates"], dtype=float)
            values = np.asarray(data["values"])
        return cls(hermitian=hermitian).fit(coordinates, values)


def load_field_model(filename):
    """Load any persisted continuous field model used by ``AbInitioFit``."""
    filename = Path(filename)
    with np.load(filename, allow_pickle=False) as data:
        model_class = (
            str(np.asarray(data["model_class"]).item())
            if "model_class" in data
            else "FunctionalTT"
        )
    if model_class == "FunctionalTT":
        return FunctionalTT.load(filename)
    if model_class == "PiecewisePCHIP":
        return PiecewisePCHIP.load(filename)
    raise ValueError(f"unknown continuous field model class {model_class!r}")


__all__ = [
    "FunctionalTT",
    "PiecewisePCHIP",
    "hermitian_basis",
    "load_field_model",
    "pack_hermitian",
]
