"""Canonical dense matrix-product operators."""

from __future__ import annotations

from numbers import Integral, Number

import numpy as np

from pyqed.lattice import Site


class MPO:
    """Finite open-boundary MPO in ``(left, right, out, in)`` order."""

    STANDARD_LABELS = ("left", "right", "out", "in")

    def __init__(
        self,
        tensors,
        target_qn=None,
        labels=STANDARD_LABELS,
        homogeneous=False,
        *,
        sites=None,
    ):
        tensors = tuple(tensors)
        if not tensors:
            raise ValueError("an MPO must contain at least one site tensor.")
        if any(getattr(tensor, "ndim", getattr(tensor, "rank", None)) != 4 for tensor in tensors):
            raise ValueError("every MPO site tensor must have rank four.")

        labels = tuple(labels)
        if labels != self.STANDARD_LABELS:
            raise ValueError(
                "MPO tensors must use ('left', 'right', 'out', 'in') ordering."
            )

        previous = 1
        dims = []
        copied = []
        for site, tensor in enumerate(tensors):
            shape = tuple(int(value) for value in tensor.shape)
            if shape[0] != previous:
                raise ValueError(f"MPO bond mismatch before site {site}.")
            if shape[2] != shape[3]:
                raise ValueError(f"MPO tensor {site} must have square physical legs.")
            previous = shape[1]
            dims.append(shape[2])
            if isinstance(tensor, np.ndarray):
                if tensor.flags.writeable:
                    value = np.array(tensor, copy=True)
                    value.setflags(write=False)
                else:
                    value = tensor
            else:
                value = tensor
            copied.append(value)
        if previous != 1:
            raise ValueError("the final MPO bond dimension must be one.")

        if sites is None:
            sites = tuple(Site(dim) for dim in dims)
        else:
            sites = tuple(sites)
            if len(sites) != len(dims) or any(not isinstance(site, Site) for site in sites):
                raise TypeError("sites must contain one canonical Site per MPO tensor.")
            site_dims = tuple(site.dim for site in sites)
            if site_dims != tuple(dims):
                raise ValueError(
                    f"site dimensions {site_dims} do not match MPO dimensions {tuple(dims)}."
                )

        self.tensors = tuple(copied)
        self.factors = self.tensors
        self.data = self.tensors
        self.cores = self.tensors
        self.sites = sites
        self.dims = tuple(dims)
        self.target_qn = target_qn
        self.labels = labels
        self.homogeneous = bool(homogeneous)
        self.nsites = self.L = len(self.tensors)
        self.nbonds = self.L - 1
        if self.homogeneous and len(set(self.dims)) != 1:
            raise ValueError(
                "homogeneous=True requires the same physical dimension at every site."
            )

    def __len__(self):
        return self.L

    def __iter__(self):
        return iter(self.tensors)

    def __getitem__(self, site):
        return self.tensors[site]

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return (1,) + tuple(int(tensor.shape[1]) for tensor in self.tensors)

    @property
    def dtype(self):
        return np.dtype(
            np.result_type(*[getattr(tensor, "dtype", float) for tensor in self.tensors])
        )

    def bond_orders(self):
        """Return the right virtual dimension of every site tensor."""
        return list(self.bond_dims[1:])

    def copy(self):
        return type(self)(
            self.tensors,
            target_qn=self.target_qn,
            labels=self.labels,
            homogeneous=self.homogeneous,
            sites=self.sites,
        )

    def to_dense(self) -> np.ndarray:
        """Contract the MPO into a dense matrix for small-system validation."""
        environment = np.ones((1, 1, 1), dtype=self.dtype)
        output_dim = input_dim = 1
        for tensor, dim in zip(self.tensors, self.dims):
            tensor = np.asarray(tensor)
            value = np.tensordot(environment, tensor, axes=(0, 0))
            value = value.transpose(2, 0, 3, 1, 4)
            output_dim *= dim
            input_dim *= dim
            environment = value.reshape(tensor.shape[1], output_dim, input_dim)
        return environment[0]

    def compress(self, max_rank=None, *, rtol=None, atol=None):
        """Return a scale-preserving TT-rounded MPO.

        With no arguments, only numerically redundant channels are removed.
        ``max_rank`` bounds every internal virtual dimension. ``rtol`` and
        ``atol`` discard singular values below the corresponding relative and
        absolute thresholds.
        """
        if max_rank is not None:
            if isinstance(max_rank, bool) or not isinstance(max_rank, Integral):
                raise TypeError("max_rank must be a positive integer or None.")
            max_rank = int(max_rank)
            if max_rank <= 0:
                raise ValueError("max_rank must be a positive integer or None.")
        for name, tolerance in (("rtol", rtol), ("atol", atol)):
            if tolerance is not None:
                tolerance = float(tolerance)
                if not np.isfinite(tolerance) or tolerance < 0.0:
                    raise ValueError(
                        f"{name} must be a finite nonnegative number or None."
                    )
                if name == "rtol":
                    rtol = tolerance
                else:
                    atol = tolerance

        cores = [np.array(tensor, copy=True) for tensor in self.tensors]

        for site in range(len(cores) - 1):
            left_dim, right_dim, out_dim, in_dim = cores[site].shape
            matrix = (
                cores[site]
                .transpose(0, 2, 3, 1)
                .reshape(left_dim * out_dim * in_dim, right_dim)
            )
            left, transfer = np.linalg.qr(matrix, mode="reduced")
            rank = left.shape[1]
            cores[site] = left.reshape(
                left_dim, out_dim, in_dim, rank
            ).transpose(0, 3, 1, 2)
            cores[site + 1] = np.tensordot(transfer, cores[site + 1], axes=(1, 0))

        for site in range(len(cores) - 1, 0, -1):
            left_dim, right_dim, out_dim, in_dim = cores[site].shape
            matrix = (
                cores[site]
                .transpose(0, 2, 3, 1)
                .reshape(left_dim, out_dim * in_dim * right_dim)
            )
            left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
            if singular_values.size and singular_values[0] > 0.0:
                relative_tolerance = (
                    np.finfo(singular_values.dtype).eps * max(matrix.shape)
                    if rtol is None
                    else rtol
                )
                threshold = max(
                    relative_tolerance * singular_values[0],
                    0.0 if atol is None else atol,
                )
                rank = max(1, int(np.count_nonzero(singular_values > threshold)))
            else:
                rank = 1
            if max_rank is not None:
                rank = min(rank, max_rank)

            left = left[:, :rank]
            singular_values = singular_values[:rank]
            right = right[:rank]
            cores[site] = right.reshape(
                rank, out_dim, in_dim, right_dim
            ).transpose(0, 3, 1, 2)
            transfer = left * singular_values[None, :]
            previous = np.tensordot(cores[site - 1], transfer, axes=(1, 0))
            cores[site - 1] = previous.transpose(0, 3, 1, 2)

        return type(self)(
            cores,
            target_qn=self.target_qn,
            labels=self.labels,
            homogeneous=self.homogeneous,
            sites=self.sites,
        )

    def dot(self, state, D=None):
        """Apply the MPO and compress the resulting MPS."""
        from pyqed.mps.mps import MPS, apply_mpo

        if not isinstance(state, MPS):
            raise TypeError("MPO.dot expects an MPS.")
        if D is None:
            D = 2 * max(state.bond_orders())
        return MPS(
            apply_mpo(self, state, D),
            sites=state.sites,
        )

    def matmul(self, other, chi_max=None):
        """Apply or multiply tensor networks, with optional rank truncation."""
        from pyqed.mps.mps import MPS, apply_mpo, product_MPO

        if isinstance(other, MPO):
            product = type(self)(
                product_MPO(self, other),
                sites=self.sites,
            )
            return product if chi_max is None else product.compress(chi_max)
        if isinstance(other, MPS):
            if chi_max is None:
                return self @ other
            return MPS(
                apply_mpo(self, other, chi_max),
                sites=other.sites,
            )
        raise TypeError(f"unsupported operand type: {type(other)}")

    def __matmul__(self, other):
        from pyqed.mps.mps import MPS, _apply_mpo_uncompressed, product_MPO

        if isinstance(other, MPO):
            return type(self)(product_MPO(self, other), sites=self.sites)
        if isinstance(other, MPS):
            return MPS(
                _apply_mpo_uncompressed(self, other),
                sites=other.sites,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            factors = [np.array(tensor, copy=True) for tensor in self.tensors]
            factors[0] *= other
            return type(self)(factors, sites=self.sites)
        if not isinstance(other, MPO):
            return NotImplemented
        self._require_compatible(other)
        factors = []
        for left, right in zip(self.tensors, other.tensors):
            factors.append(
                np.einsum("abij,mnij->ambnij", left, right).reshape(
                    left.shape[0] * right.shape[0],
                    left.shape[1] * right.shape[1],
                    left.shape[2],
                    left.shape[3],
                )
            )
        return type(self)(factors, sites=self.sites)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, MPO):
            return NotImplemented
        self._require_compatible(other)
        if self.L == 1:
            return type(self)(
                [np.asarray(self[0]) + np.asarray(other[0])],
                sites=self.sites,
            )

        factors = []
        for site, (left, right) in enumerate(zip(self.tensors, other.tensors)):
            if site == 0:
                value = np.concatenate((left, right), axis=1)
            elif site == self.L - 1:
                value = np.concatenate((left, right), axis=0)
            else:
                value = np.zeros(
                    (
                        left.shape[0] + right.shape[0],
                        left.shape[1] + right.shape[1],
                        left.shape[2],
                        left.shape[3],
                    ),
                    dtype=np.result_type(left.dtype, right.dtype),
                )
                value[: left.shape[0], : left.shape[1]] = left
                value[left.shape[0] :, left.shape[1] :] = right
            factors.append(value)
        return type(self)(factors, sites=self.sites)

    def exponential(self, constant=1.0, D=None, method="taylor", order=4, scale=0):
        from pyqed.mps.mps import expmpo

        return expmpo(
            self,
            constant=constant,
            D=D,
            method=method,
            order=order,
            scale=scale,
        )

    def _require_compatible(self, other):
        if self.L != other.L:
            raise ValueError(f"MPO lengths differ: {self.L} and {other.L}.")
        if self.dims != other.dims:
            raise ValueError(
                f"MPO physical dimensions differ: {self.dims} and {other.dims}."
            )


__all__ = ["MPO"]
