"""Overlap-dressed kinetic operators for locally diabatic representations."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import matmul_toeplitz, toeplitz
from scipy.sparse.linalg import LinearOperator

from pyqed.mps.mps import MPO

from . import overlap as overlap_tools


def _entries(kinetic, threshold):
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    if sp.issparse(kinetic):
        kinetic = kinetic.tocoo()
        if kinetic.shape[0] != kinetic.shape[1]:
            raise ValueError("kinetic must be square")
        keep = np.abs(kinetic.data) > threshold
        return (
            kinetic.shape[0],
            kinetic.row[keep],
            kinetic.col[keep],
            kinetic.data[keep],
        )

    kinetic = np.asarray(kinetic, dtype=complex)
    if kinetic.ndim != 2 or kinetic.shape[0] != kinetic.shape[1]:
        raise ValueError("kinetic must be square")
    rows, cols = np.nonzero(np.abs(kinetic) > threshold)
    return kinetic.shape[0], rows, cols, kinetic[rows, cols]


def dress(kinetic, overlap, *, nstates=None, threshold=0.0, symmetrize=False):
    """Return sparse blocks $T_{ij} A_{ij}$ for an overlap callback."""

    size, rows, cols, values = _entries(kinetic, float(threshold))
    if nstates is None:
        data = np.empty(len(values), dtype=complex)
        for edge, (i, j, tij) in enumerate(zip(rows, cols, values)):
            data[edge] = tij * overlap_tools.as_block(
                overlap(int(i), int(j)),
                None,
            )
        result = sp.csr_matrix((data, (rows, cols)), shape=(size, size))
    else:
        nstates = int(nstates)
        if nstates <= 0:
            raise ValueError("nstates must be positive")
        block_rows = []
        block_cols = []
        block_data = []
        for i, j, tij in zip(rows, cols, values):
            value = overlap_tools.as_block(overlap(int(i), int(j)), nstates)
            value = tij * value
            state_rows, state_cols = np.nonzero(np.abs(value) > threshold)
            block_rows.extend((int(i) * nstates + state_rows).tolist())
            block_cols.extend((int(j) * nstates + state_cols).tolist())
            block_data.extend(value[state_rows, state_cols].tolist())
        dimension = size * nstates
        result = sp.csr_matrix(
            (block_data, (block_rows, block_cols)),
            shape=(dimension, dimension),
            dtype=complex,
        )

    if symmetrize:
        result = 0.5 * (result + result.getH())
    return result


def block(i, j, bra, ket, links, *, nstates, average_paths=False):
    """Return one linked overlap block in dense-overlap orientation."""

    if i == j:
        if nstates is None:
            return 1.0 + 0.0j
        return np.eye(int(nstates), dtype=complex)
    if i < j:
        return overlap_tools.between(
            bra,
            ket,
            links,
            nstates=nstates,
            average_paths=average_paths,
        )
    value = overlap_tools.between(
        ket,
        bra,
        links,
        nstates=nstates,
        average_paths=average_paths,
    )
    return value.conjugate() if nstates is None else value.conj().T


def linked(
    kinetic,
    shape,
    links,
    *,
    nstates=None,
    threshold=0.0,
    symmetrize=True,
    average_paths=False,
):
    """Return a sparse kinetic matrix dressed by linked-product overlaps."""

    indices, _, _ = overlap_tools.layout(shape)
    size = len(indices)
    if np.shape(kinetic) != (size, size):
        raise ValueError(f"kinetic shape {np.shape(kinetic)} != {(size, size)}")
    indices = tuple(map(tuple, indices))
    return dress(
        kinetic,
        lambda i, j: block(
            i,
            j,
            indices[i],
            indices[j],
            links,
            nstates=nstates,
            average_paths=average_paths,
        ),
        nstates=nstates,
        threshold=threshold,
        symmetrize=symmetrize,
    )


def _matrix_field_mpo(values):
    values = np.asarray(values, dtype=complex)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("matrix field must have shape (ngrid, nstates, nstates)")
    ngrid, nstates, _ = values.shape
    rank = nstates * nstates
    nuclear = np.zeros((1, rank, ngrid, ngrid), dtype=complex)
    electronic = np.zeros((rank, 1, nstates, nstates), dtype=complex)
    diagonal = np.arange(ngrid)
    for row in range(nstates):
        for column in range(nstates):
            channel = row * nstates + column
            nuclear[0, channel, diagonal, diagonal] = values[:, row, column]
            electronic[channel, 0, row, column] = 1.0
    return MPO((nuclear, electronic))


def _nuclear_mpo(operator, nstates):
    operator = np.asarray(operator, dtype=complex)
    identity = np.eye(int(nstates), dtype=complex)
    return MPO((operator[None, None], identity[None, None]))


def _link_prefixes(ngrid, links, max_condition):
    if ngrid < 2:
        raise ValueError("prefix factorization requires at least two grid points")
    if isinstance(links, dict):
        links = tuple(
            np.asarray(links[(0, (index,))], dtype=complex)
            for index in range(ngrid - 1)
        )
    else:
        links = tuple(np.asarray(link, dtype=complex) for link in links)
    if len(links) != ngrid - 1:
        raise ValueError("provide one forward link between each adjacent grid pair")
    nstates = links[0].shape[0]
    if nstates < 1 or any(link.shape != (nstates, nstates) for link in links):
        raise ValueError("links must be nonempty square matrices with one common size")

    prefixes = np.empty((ngrid, nstates, nstates), dtype=complex)
    prefixes[0] = np.eye(nstates, dtype=complex)
    for index, link in enumerate(links):
        prefixes[index + 1] = prefixes[index] @ link
    link_conditions = np.asarray([np.linalg.cond(link) for link in links])
    prefix_conditions = np.asarray([np.linalg.cond(value) for value in prefixes])
    if not np.all(np.isfinite(prefix_conditions)):
        raise np.linalg.LinAlgError("prefix transport is singular")
    if max_condition is not None and np.max(prefix_conditions) > float(max_condition):
        raise np.linalg.LinAlgError(
            f"prefix condition {np.max(prefix_conditions):.3e} exceeds "
            f"{float(max_condition):.3e}"
        )
    inverses = np.asarray([np.linalg.inv(value) for value in prefixes])
    info = {
        "ngrid": ngrid,
        "nstates": nstates,
        "max_link_condition": float(np.max(link_conditions)),
        "max_prefix_condition": float(np.max(prefix_conditions)),
    }
    return prefixes, inverses, info


def _prefix_data(kinetic, links, max_condition):
    kinetic = np.asarray(kinetic, dtype=complex)
    if kinetic.ndim != 2 or kinetic.shape[0] != kinetic.shape[1]:
        raise ValueError("kinetic must be a square matrix")
    ngrid = kinetic.shape[0]
    if not np.allclose(kinetic, kinetic.conj().T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("prefix factorization requires a Hermitian kinetic matrix")
    prefixes, inverses, info = _link_prefixes(ngrid, links, max_condition)
    scale = max(float(np.linalg.norm(kinetic)), np.finfo(float).tiny)
    first_row = kinetic[0]
    circulant = np.asarray([np.roll(first_row, index) for index in range(ngrid)])
    toeplitz_matrix = toeplitz(kinetic[:, 0], kinetic[0])
    info.update({
        "toeplitz_error": float(np.linalg.norm(kinetic - toeplitz_matrix) / scale),
        "circulant_error": float(np.linalg.norm(kinetic - circulant) / scale),
    })
    return kinetic, prefixes, inverses, info


def _toeplitz_coefficients(kinetic):
    descriptor = isinstance(kinetic, (tuple, list)) and len(kinetic) == 2
    if descriptor:
        column, row = (np.asarray(value, dtype=complex) for value in kinetic)
        if column.ndim != 1 or row.ndim != 1 or column.shape != row.shape:
            raise ValueError("Toeplitz column and row must be equal-length vectors")
        if column.size < 2:
            raise ValueError("prefix factorization requires at least two grid points")
        if not np.allclose(column[0], row[0], rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("Toeplitz column and row must share their first value")
        if not np.allclose(column, row.conj(), rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("prefix factorization requires a Hermitian Toeplitz KEO")
        circulant_column = np.concatenate((row[:1], row[:0:-1]))
        scale = max(float(np.linalg.norm(column)), np.finfo(float).tiny)
        return column, row, {
            "ngrid": int(column.size),
            "toeplitz_error": 0.0,
            "circulant_error": float(
                np.linalg.norm(column - circulant_column) / scale
            ),
            "descriptor": True,
        }

    kinetic = np.asarray(kinetic, dtype=complex)
    if kinetic.ndim != 2 or kinetic.shape[0] != kinetic.shape[1]:
        raise ValueError("kinetic must be a square matrix or Toeplitz descriptor")
    if not np.allclose(kinetic, kinetic.conj().T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("prefix factorization requires a Hermitian kinetic matrix")
    scale = max(float(np.linalg.norm(kinetic)), np.finfo(float).tiny)
    row = kinetic[0].copy()
    column = kinetic[:, 0].copy()
    toeplitz_matrix = toeplitz(column, row)
    circulant = np.asarray([np.roll(row, index) for index in range(len(row))])
    return column, row, {
        "ngrid": int(column.size),
        "toeplitz_error": float(np.linalg.norm(kinetic - toeplitz_matrix) / scale),
        "circulant_error": float(np.linalg.norm(kinetic - circulant) / scale),
        "descriptor": False,
    }


def _sine_coefficients(kinetic):
    if not isinstance(kinetic, dict):
        return None
    if kinetic.get("kind") != "sine-toeplitz-hankel":
        raise ValueError("unknown structured kinetic descriptor")
    column = np.asarray(kinetic.get("column"), dtype=complex)
    row = np.asarray(kinetic.get("row"), dtype=complex)
    hankel = np.asarray(kinetic.get("hankel"), dtype=complex)
    if column.ndim != 1 or row.shape != column.shape:
        raise ValueError("sine descriptor column and row must be equal-length vectors")
    if column.size < 2 or hankel.shape != (2 * column.size - 1,):
        raise ValueError("sine descriptor has incompatible Toeplitz/Hankel sizes")
    if not np.allclose(column, row.conj(), rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("sine descriptor must be Hermitian")
    if not np.allclose(hankel, hankel.conj(), rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("sine descriptor Hankel coefficients must be real")
    return column, row, hankel, {
        "ngrid": int(column.size),
        "toeplitz_error": 0.0,
        "circulant_error": 0.0,
        "descriptor": True,
        "structure": "toeplitz-hankel",
    }


class _TriangularHankel:
    """FFT-recursive strict triangular action for a Hankel matrix."""

    def __init__(self, coefficients, *, upper, workers=None):
        self.coefficients = np.asarray(coefficients, dtype=complex)
        self.ngrid = (self.coefficients.size + 1) // 2
        self.upper = bool(upper)
        self.workers = workers

    def _rectangular(self, row_start, row_stop, col_start, col_stop, values):
        nrows = row_stop - row_start
        ncols = col_stop - col_start
        base = row_start + col_start + ncols - 1
        column = self.coefficients[base + np.arange(nrows)]
        row = self.coefficients[base - np.arange(ncols)]
        return np.asarray(
            matmul_toeplitz(
                (column, row),
                values[::-1],
                check_finite=False,
                workers=self.workers,
            )
        )

    def apply(self, values):
        values = np.asarray(values, dtype=complex)
        if values.shape[0] != self.ngrid:
            raise ValueError("Hankel input has an incompatible leading dimension")
        shape = values.shape
        values = values.reshape(self.ngrid, -1)
        output = np.zeros_like(values)

        def recurse(start, stop):
            if stop - start <= 1:
                return
            middle = (start + stop) // 2
            if self.upper:
                output[start:middle] += self._rectangular(
                    start, middle, middle, stop, values[middle:stop]
                )
            else:
                output[middle:stop] += self._rectangular(
                    middle, stop, start, middle, values[start:middle]
                )
            recurse(start, middle)
            recurse(middle, stop)

        recurse(0, self.ngrid)
        return output.reshape(shape)


def _toeplitz_data(kinetic, links, max_condition):
    column, row, kinetic_info = _toeplitz_coefficients(kinetic)
    prefixes, inverses, link_info = _link_prefixes(
        column.size, links, max_condition
    )
    return column, row, prefixes, inverses, {**link_info, **kinetic_info}


def prefix_mpos(kinetic, links, *, max_condition=None, max_rank="auto"):
    r"""Factor a dense one-dimensional linked KEO into three MPOs.

    With prefix products ``A[i + 1] = A[i] @ L[i]``, the upper-triangular
    linked kinetic operator is ``D[A^-1] T+ D[A]``.  Its adjoint supplies the
    lower triangle, while the diagonal is unchanged.  The factorization is
    exact for invertible links and a Hermitian nuclear KEO.
    """
    kinetic, prefixes, inverses, prefix_info = _prefix_data(
        kinetic, links, max_condition
    )
    ngrid = prefix_info["ngrid"]
    nstates = prefix_info["nstates"]

    diagonal = _nuclear_mpo(np.diag(np.diag(kinetic)), nstates)
    upper = _nuclear_mpo(np.triu(kinetic, 1), nstates)
    forward = _matrix_field_mpo(inverses) @ upper @ _matrix_field_mpo(prefixes)
    rank_cap = nstates * nstates if max_rank == "auto" else max_rank
    if rank_cap is not None:
        rank_cap = int(rank_cap)
        if rank_cap < 1:
            raise ValueError("max_rank must be positive, None, or 'auto'")
        if max(forward.bond_orders(), default=1) > rank_cap:
            forward = forward.compress(rank_cap)
    components = (diagonal, forward, forward.adjoint())

    info = {
        "backend": "prefix-mpo",
        **prefix_info,
        "operator_ranks": [tuple(component.bond_orders()) for component in components],
        "rank_cap": rank_cap,
        "compressed": rank_cap is not None,
    }
    return components, info


class PrefixFFT:
    """FFT application of a structured KEO dressed by invertible links."""

    def __init__(
        self,
        kinetic,
        links,
        *,
        max_condition=None,
        toeplitz_tolerance=1.0e-12,
        workers=None,
    ):
        sine = _sine_coefficients(kinetic)
        if sine is None:
            column, row, prefixes, inverses, info = _toeplitz_data(
                kinetic, links, max_condition
            )
            hankel = None
        else:
            column, row, hankel, kinetic_info = sine
            prefixes, inverses, link_info = _link_prefixes(
                column.size, links, max_condition
            )
            info = {**link_info, **kinetic_info}
        tolerance = float(toeplitz_tolerance)
        if tolerance < 0.0:
            raise ValueError("toeplitz_tolerance must be nonnegative")
        if info["toeplitz_error"] > tolerance:
            raise ValueError(
                f"kinetic Toeplitz error {info['toeplitz_error']:.3e} exceeds "
                f"{tolerance:.3e}"
            )
        self.prefixes = prefixes
        self.inverses = inverses
        self.ngrid = int(info["ngrid"])
        self.nstates = int(info["nstates"])
        self.shape = (self.ngrid * self.nstates,) * 2
        self.dtype = np.dtype(complex)
        self.workers = workers
        zeros = np.zeros(self.ngrid, dtype=complex)
        upper_row = np.array(row, copy=True)
        lower_column = np.array(column, copy=True)
        upper_row[0] = 0.0
        lower_column[0] = 0.0
        self._upper = (zeros, upper_row)
        self._lower = (lower_column, zeros)
        self._upper_hankel = None
        self._lower_hankel = None
        self._diagonal = np.full(self.ngrid, column[0], dtype=complex)
        if hankel is not None:
            self._upper_hankel = _TriangularHankel(
                hankel, upper=True, workers=workers
            )
            self._lower_hankel = _TriangularHankel(
                hankel, upper=False, workers=workers
            )
            self._diagonal += hankel[2 * np.arange(self.ngrid)]
        self.info = {"backend": "prefix-fft", **info}

    def _toeplitz(self, descriptor, values):
        shape = values.shape
        flat = values.reshape(self.ngrid, -1)
        output = matmul_toeplitz(
            descriptor,
            flat,
            check_finite=False,
            workers=self.workers,
        )
        return np.asarray(output).reshape(shape)

    def _structured(self, descriptor, hankel, values):
        output = self._toeplitz(descriptor, values)
        if hankel is not None:
            output += hankel.apply(values)
        return output

    def apply(self, values):
        """Apply to ``(ngrid, nstates, ...)`` data or one flattened vector."""
        values = np.asarray(values, dtype=complex)
        flattened = values.ndim == 1
        if flattened:
            if values.size != self.shape[1]:
                raise ValueError("vector has an incompatible dimension")
            values = values.reshape(self.ngrid, self.nstates)
        elif values.shape[:2] != (self.ngrid, self.nstates):
            raise ValueError("values must start with (ngrid, nstates)")

        forward = np.einsum("iab,ib...->ia...", self.prefixes, values)
        backward = np.einsum(
            "iba,ib...->ia...", self.inverses.conj(), values
        )
        forward = self._structured(self._upper, self._upper_hankel, forward)
        backward = self._structured(self._lower, self._lower_hankel, backward)
        output = self._diagonal.reshape(
            (self.ngrid,) + (1,) * (values.ndim - 1)
        ) * values
        output += np.einsum("iab,ib...->ia...", self.inverses, forward)
        output += np.einsum(
            "iba,ib...->ia...", self.prefixes.conj(), backward
        )
        return output.reshape(-1) if flattened else output

    def matvec(self, vector):
        return self.apply(vector)

    def matmat(self, vectors):
        vectors = np.asarray(vectors, dtype=complex)
        if vectors.ndim != 2 or vectors.shape[0] != self.shape[1]:
            raise ValueError("vectors have an incompatible shape")
        values = vectors.reshape(self.ngrid, self.nstates, vectors.shape[1])
        return self.apply(values).reshape(self.shape[0], vectors.shape[1])

    def aslinearoperator(self):
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.matvec,
            matmat=self.matmat,
            dtype=self.dtype,
        )


class PrefixFFTND:
    """Batched axis-wise prefix FFT for a separable multidimensional KEO."""

    def __init__(
        self,
        kinetics,
        shape,
        links,
        *,
        max_condition=None,
        toeplitz_tolerance=1.0e-12,
        non_toeplitz="direct",
        workers=None,
    ):
        self.shape_grid = tuple(int(value) for value in shape)
        self.ndim = len(self.shape_grid)
        if self.ndim < 2 or any(value < 2 for value in self.shape_grid):
            raise ValueError("PrefixFFTND requires at least two nontrivial axes")
        kinetics = tuple(kinetics)
        if len(kinetics) != self.ndim:
            raise ValueError("provide one Toeplitz KEO per grid axis")
        if not isinstance(links, dict) or not links:
            raise ValueError("links must be a nonempty directional-link dictionary")

        first_link = np.asarray(next(iter(links.values())), dtype=complex)
        if first_link.ndim != 2 or first_link.shape[0] != first_link.shape[1]:
            raise ValueError("links must be square matrices")
        self.nstates = int(first_link.shape[0])
        self.workers = workers
        self.dtype = np.dtype(complex)
        size = int(np.prod(self.shape_grid)) * self.nstates
        self.shape = (size, size)
        tolerance = float(toeplitz_tolerance)
        if tolerance < 0.0:
            raise ValueError("toeplitz_tolerance must be nonnegative")
        non_toeplitz = str(non_toeplitz).lower()
        if non_toeplitz not in {"direct", "error"}:
            raise ValueError("non_toeplitz must be 'direct' or 'error'")

        self._axes = []
        axis_info = []
        for axis, (kinetic, axis_size) in enumerate(
            zip(kinetics, self.shape_grid)
        ):
            sine = _sine_coefficients(kinetic)
            if sine is None:
                column, row, info = _toeplitz_coefficients(kinetic)
                hankel = None
            else:
                column, row, hankel, info = sine
            if column.size != axis_size:
                raise ValueError(
                    f"axis {axis} KEO size {column.size} != grid size {axis_size}"
                )
            use_fft = hankel is not None or info["toeplitz_error"] <= tolerance
            if not use_fft and non_toeplitz == "error":
                raise ValueError(
                    f"axis {axis} Toeplitz error {info['toeplitz_error']:.3e} "
                    f"exceeds {tolerance:.3e}"
                )
            if use_fft:
                zeros = np.zeros(axis_size, dtype=complex)
                upper = (zeros, np.array(row, copy=True))
                lower = (np.array(column, copy=True), zeros)
                upper[1][0] = 0.0
                lower[0][0] = 0.0
                diagonal = np.full(axis_size, column[0], dtype=complex)
                if hankel is None:
                    upper_hankel = lower_hankel = None
                    axis_backend = "fft"
                else:
                    upper_hankel = _TriangularHankel(
                        hankel, upper=True, workers=workers
                    )
                    lower_hankel = _TriangularHankel(
                        hankel, upper=False, workers=workers
                    )
                    diagonal += hankel[2 * np.arange(axis_size)]
                    axis_backend = "sine-fft"
            else:
                matrix = np.asarray(kinetic, dtype=complex)
                upper = np.triu(matrix, 1)
                lower = np.tril(matrix, -1)
                diagonal = np.diag(matrix).copy()
                upper_hankel = lower_hankel = None
                axis_backend = "direct"

            prefixes = np.empty(
                (*self.shape_grid, self.nstates, self.nstates), dtype=complex
            )
            inverses = np.empty_like(prefixes)
            transverse_shape = self.shape_grid[:axis] + self.shape_grid[axis + 1 :]
            link_condition = 0.0
            prefix_condition = 0.0
            for transverse in np.ndindex(*transverse_shape):
                point = list(transverse)
                point.insert(axis, 0)
                fiber_links = []
                for index in range(axis_size - 1):
                    point[axis] = index
                    fiber_links.append(links[(axis, tuple(point))])
                fiber_prefixes, fiber_inverses, fiber_info = _link_prefixes(
                    axis_size, fiber_links, max_condition
                )
                for index in range(axis_size):
                    point[axis] = index
                    prefixes[tuple(point)] = fiber_prefixes[index]
                    inverses[tuple(point)] = fiber_inverses[index]
                link_condition = max(
                    link_condition, fiber_info["max_link_condition"]
                )
                prefix_condition = max(
                    prefix_condition, fiber_info["max_prefix_condition"]
                )

            self._axes.append(
                {
                    "prefixes": prefixes,
                    "inverses": inverses,
                    "upper": upper,
                    "lower": lower,
                    "upper_hankel": upper_hankel,
                    "lower_hankel": lower_hankel,
                    "diagonal": diagonal,
                    "backend": axis_backend,
                }
            )
            axis_info.append(
                {
                    "axis": axis,
                    "backend": axis_backend,
                    **info,
                    "max_link_condition": float(link_condition),
                    "max_prefix_condition": float(prefix_condition),
                }
            )
        self.info = {"backend": "prefix-fft-nd", "axes": axis_info}

    def _local(self, matrices, values):
        tail = values.shape[self.ndim + 1 :]
        flattened = values.reshape(*self.shape_grid, self.nstates, -1)
        output = np.einsum(
            "...ab,...bk->...ak", matrices, flattened, optimize=True
        )
        return output.reshape(*self.shape_grid, self.nstates, *tail)

    def _axis_action(self, axis, axis_data, side, values):
        moved = np.moveaxis(values, axis, 0)
        moved_shape = moved.shape
        flattened = moved.reshape(moved_shape[0], -1)
        operator = axis_data[side]
        if axis_data["backend"] in {"fft", "sine-fft"}:
            output = matmul_toeplitz(
                operator,
                flattened,
                check_finite=False,
                workers=self.workers,
            )
            hankel = axis_data[f"{side}_hankel"]
            if hankel is not None:
                output = np.asarray(output) + hankel.apply(flattened)
        else:
            output = operator @ flattened
        output = np.asarray(output).reshape(moved_shape)
        return np.moveaxis(output, 0, axis)

    def apply(self, values):
        values = np.asarray(values, dtype=complex)
        flattened = values.ndim == 1
        if flattened:
            if values.size != self.shape[1]:
                raise ValueError("vector has an incompatible dimension")
            values = values.reshape(*self.shape_grid, self.nstates)
        elif values.shape[: self.ndim + 1] != (*self.shape_grid, self.nstates):
            raise ValueError("values have incompatible grid or electronic dimensions")

        output = np.zeros_like(values, dtype=complex)
        for axis, axis_data in enumerate(self._axes):
            diagonal_shape = [1] * values.ndim
            diagonal_shape[axis] = self.shape_grid[axis]
            output += axis_data["diagonal"].reshape(diagonal_shape) * values
            prefixes = axis_data["prefixes"]
            inverses = axis_data["inverses"]
            forward = self._local(prefixes, values)
            forward = self._axis_action(axis, axis_data, "upper", forward)
            output += self._local(inverses, forward)

            inverse_adjoint = inverses.conj().swapaxes(-1, -2)
            prefix_adjoint = prefixes.conj().swapaxes(-1, -2)
            backward = self._local(inverse_adjoint, values)
            backward = self._axis_action(axis, axis_data, "lower", backward)
            output += self._local(prefix_adjoint, backward)
        return output.reshape(-1) if flattened else output

    def matvec(self, vector):
        return self.apply(vector)

    def matmat(self, vectors):
        vectors = np.asarray(vectors, dtype=complex)
        if vectors.ndim != 2 or vectors.shape[0] != self.shape[1]:
            raise ValueError("vectors have an incompatible shape")
        values = vectors.reshape(*self.shape_grid, self.nstates, vectors.shape[1])
        return self.apply(values).reshape(self.shape[0], vectors.shape[1])

    def aslinearoperator(self):
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.matvec,
            matmat=self.matmat,
            dtype=self.dtype,
        )


def _setup(kinetic, shape, nstates, overlaps, nrot):
    shape = tuple(int(n) for n in shape)
    indices, _, _ = overlap_tools.layout(shape)
    indices = tuple(map(tuple, indices))
    ngrid = len(indices)
    nstates = int(nstates)
    nrot = int(nrot)
    if nstates <= 0 or nrot <= 0:
        raise ValueError("nstates and nrot must be positive")

    nuclear_size = ngrid * nrot
    if kinetic.shape != (nuclear_size, nuclear_size):
        raise ValueError(
            f"kinetic shape {kinetic.shape} != {(nuclear_size, nuclear_size)}"
        )

    overlap_array = None
    if overlaps is not None:
        overlap_array = np.asarray(overlaps, dtype=complex)
        expected_size = ngrid * nstates * ngrid * nstates
        if overlap_array.size != expected_size:
            raise ValueError("overlaps have incompatible grid or state dimensions")
        overlap_array = overlap_array.reshape(ngrid, nstates, ngrid, nstates)
    return indices, ngrid, nstates, nrot, overlap_array


def matrix(
    kinetic,
    shape,
    nstates,
    *,
    overlaps=None,
    links=None,
    nrot=1,
    threshold=0.0,
    average_paths=False,
    symmetrize=True,
):
    """Materialize an overlap-dressed kinetic matrix."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    indices, ngrid, nstates, nrot, overlap_array = _setup(
        kinetic,
        shape,
        nstates,
        overlaps,
        nrot,
    )
    if overlap_array is None and links is not None and nrot == 1:
        overlap_array = overlap_tools.dense(shape, links, nstates=nstates)

    nuclear = kinetic.toarray() if sp.issparse(kinetic) else np.asarray(kinetic)
    state_eye = np.eye(nstates, dtype=complex)

    if overlap_array is None and links is None:
        result = np.kron(nuclear, state_eye)
    elif overlap_array is not None:
        if nrot == 1:
            result = np.einsum(
                "ij,iajb->iajb",
                nuclear,
                overlap_array,
                optimize=True,
            )
        else:
            nuclear = nuclear.reshape(ngrid, nrot, ngrid, nrot)
            result = np.einsum(
                "irjs,iajb->irajsb",
                nuclear,
                overlap_array,
                optimize=True,
            )
        dimension = ngrid * nrot * nstates
        result = result.reshape(dimension, dimension)
    elif nrot == 1:
        result = np.zeros((ngrid, nstates, ngrid, nstates), dtype=complex)
        for i in range(ngrid):
            for j in range(ngrid):
                tij = nuclear[i, j]
                if abs(tij) <= threshold:
                    continue
                result[i, :, j, :] = tij * block(
                    i,
                    j,
                    indices[i],
                    indices[j],
                    links,
                    nstates=nstates,
                    average_paths=average_paths,
                )
        result = result.reshape(ngrid * nstates, ngrid * nstates)
    else:
        nuclear = nuclear.reshape(ngrid, nrot, ngrid, nrot)
        result = np.zeros(
            (ngrid, nrot, nstates, ngrid, nrot, nstates),
            dtype=complex,
        )
        for i in range(ngrid):
            for j in range(ngrid):
                tij = nuclear[i, :, j, :]
                if np.max(np.abs(tij)) <= threshold:
                    continue
                aij = block(
                    i,
                    j,
                    indices[i],
                    indices[j],
                    links,
                    nstates=nstates,
                    average_paths=average_paths,
                )
                result[i, :, :, j, :, :] = (
                    tij[:, None, :, None] * aij[None, :, None, :]
                )
        dimension = ngrid * nrot * nstates
        result = result.reshape(dimension, dimension)

    if symmetrize:
        result = 0.5 * (result + result.conj().T)
    return result


def operator(
    kinetic,
    shape,
    nstates,
    *,
    overlaps=None,
    links=None,
    nrot=1,
    threshold=0.0,
    average_paths=False,
):
    """Return a matrix-free overlap-dressed kinetic operator."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    indices, ngrid, nstates, nrot, overlap_array = _setup(
        kinetic,
        shape,
        nstates,
        overlaps,
        nrot,
    )
    dimension = ngrid * nrot * nstates
    dtype = np.result_type(kinetic.dtype, complex)
    linked_cache = {}

    def linked_block(i, j):
        key = (i, j)
        value = linked_cache.get(key)
        if value is None:
            value = block(
                i,
                j,
                indices[i],
                indices[j],
                links,
                nstates=nstates,
                average_paths=average_paths,
            )
            linked_cache[key] = value
        return value

    if sp.issparse(kinetic):
        nuclear = kinetic.tocsr()
        if nrot == 1:
            rows = np.repeat(np.arange(ngrid), np.diff(nuclear.indptr))
            cols = nuclear.indices.copy()
            data = nuclear.data.copy()
            if threshold > 0.0:
                keep = np.abs(data) > threshold
                rows, cols, data = rows[keep], cols[keep], data[keep]

            if overlap_array is not None:
                edge_blocks = overlap_array[rows, :, cols, :]
            elif links is not None:
                edge_blocks = np.asarray(
                    [linked_block(int(i), int(j)) for i, j in zip(rows, cols)]
                )
            else:
                edge_blocks = None
                nuclear_action = sp.csr_matrix(
                    (data, (rows, cols)),
                    shape=(ngrid, ngrid),
                )

            def matvec(vector):
                psi = np.asarray(vector).reshape(ngrid, nstates)
                if edge_blocks is None:
                    return (nuclear_action @ psi).reshape(-1)
                transported = np.einsum(
                    "eab,eb->ea",
                    edge_blocks,
                    psi[cols],
                    optimize=True,
                )
                out = np.zeros_like(psi, dtype=dtype)
                np.add.at(out, rows, data[:, None] * transported)
                return out.reshape(-1)

        else:

            def matvec(vector):
                psi = np.asarray(vector).reshape(ngrid, nrot, nstates)
                out = np.zeros_like(psi, dtype=dtype)
                for row in range(nuclear.shape[0]):
                    i, r = divmod(row, nrot)
                    start, stop = nuclear.indptr[row], nuclear.indptr[row + 1]
                    for pointer in range(start, stop):
                        tij = nuclear.data[pointer]
                        if abs(tij) <= threshold:
                            continue
                        j, s = divmod(nuclear.indices[pointer], nrot)
                        if overlap_array is not None:
                            aij = overlap_array[i, :, j, :]
                            out[i, r] += tij * (aij @ psi[j, s])
                        elif links is not None:
                            out[i, r] += tij * (linked_block(i, j) @ psi[j, s])
                        else:
                            out[i, r] += tij * psi[j, s]
                return out.reshape(-1)

    else:
        nuclear = np.asarray(kinetic)
        if nrot == 1 and overlap_array is not None:

            def matvec(vector):
                psi = np.asarray(vector).reshape(ngrid, nstates)
                return np.einsum(
                    "ij,iajb,jb->ia",
                    nuclear,
                    overlap_array,
                    psi,
                    optimize=True,
                ).reshape(-1)

        elif nrot == 1 and links is not None:

            def matvec(vector):
                psi = np.asarray(vector).reshape(ngrid, nstates)
                out = np.zeros_like(psi, dtype=dtype)
                for i in range(ngrid):
                    for j in range(ngrid):
                        tij = nuclear[i, j]
                        if abs(tij) > threshold:
                            out[i] += tij * (linked_block(i, j) @ psi[j])
                return out.reshape(-1)

        elif nrot == 1:

            def matvec(vector):
                psi = np.asarray(vector).reshape(ngrid, nstates)
                return (nuclear @ psi).reshape(-1)

        else:
            nuclear = nuclear.reshape(ngrid, nrot, ngrid, nrot)
            if overlap_array is not None:

                def matvec(vector):
                    psi = np.asarray(vector).reshape(ngrid, nrot, nstates)
                    return np.einsum(
                        "irjs,iajb,jsb->ira",
                        nuclear,
                        overlap_array,
                        psi,
                        optimize=True,
                    ).reshape(-1)

            elif links is not None:

                def matvec(vector):
                    psi = np.asarray(vector).reshape(ngrid, nrot, nstates)
                    out = np.zeros_like(psi, dtype=dtype)
                    for i in range(ngrid):
                        for j in range(ngrid):
                            tij = nuclear[i, :, j, :]
                            if np.max(np.abs(tij)) <= threshold:
                                continue
                            transported = psi[j] @ linked_block(i, j).T
                            out[i] += np.einsum(
                                "rs,sa->ra",
                                tij,
                                transported,
                                optimize=True,
                            )
                    return out.reshape(-1)

            else:

                def matvec(vector):
                    psi = np.asarray(vector).reshape(ngrid, nrot, nstates)
                    return np.einsum(
                        "irjs,jsa->ira",
                        nuclear,
                        psi,
                        optimize=True,
                    ).reshape(-1)

    def matmat(vectors):
        vectors = np.asarray(vectors)
        return np.column_stack(
            [matvec(vectors[:, column]) for column in range(vectors.shape[1])]
        )

    return LinearOperator(
        shape=(dimension, dimension),
        matvec=matvec,
        rmatvec=matvec,
        matmat=matmat,
        dtype=dtype,
    )
