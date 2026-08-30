"""Variational optimization of finite PEPS."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from pyqed.tn import Hamiltonian


class PEPSOptimizer:
    r"""Exact-environment variational optimizer for finite PEPS.

    With every other tensor fixed, the wavefunction is linear in the active
    tensor, :math:`|\psi\rangle=M|a\rangle`. Each local update therefore solves

    .. math::

        (M^\dagger H M)a = E(M^\dagger M)a.

    The norm matrix is eigendecomposed and exactly null gauge directions are
    removed before the Hermitian eigenproblem is solved. One-site updates keep
    the graph ranks fixed. Two-site updates optimize a joined nearest-neighbor
    block and SVD-split it, allowing the shared bond to grow up to ``max_D``.
    This implementation is intended as a correctness-first optimizer for
    finite and moderate PEPS; it uses exact environments and exposes size
    guards rather than silently switching to an uncontrolled dense calculation.
    """

    def __init__(
        self,
        state,
        hamiltonian,
        *,
        sweeps=4,
        tol=1.0e-9,
        metric_rtol=1.0e-11,
        metric_atol=1.0e-13,
        max_local_size=512,
        update="one-site",
        environment="auto",
        max_D=None,
        split_rtol=1.0e-12,
        max_pair_size=2048,
        max_dense_dimension=16384,
        verbose=False,
    ):
        from .state import PEPS

        if not isinstance(state, PEPS):
            raise TypeError("state must be a PEPS.")
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a pyqed.tn.Hamiltonian.")
        if state.dims != hamiltonian.dims:
            raise ValueError("Hamiltonian physical dimensions do not match the PEPS.")
        if isinstance(sweeps, bool) or not isinstance(sweeps, Integral) or int(sweeps) < 1:
            raise ValueError("sweeps must be a positive integer.")
        if (
            isinstance(max_local_size, bool)
            or not isinstance(max_local_size, Integral)
            or int(max_local_size) < 1
        ):
            raise ValueError("max_local_size must be a positive integer.")
        if (
            isinstance(max_dense_dimension, bool)
            or not isinstance(max_dense_dimension, Integral)
            or int(max_dense_dimension) < 1
        ):
            raise ValueError("max_dense_dimension must be a positive integer.")
        self.state = state
        self.hamiltonian = hamiltonian
        self.sweeps = int(sweeps)
        self.tol = float(tol)
        self.metric_rtol = float(metric_rtol)
        self.metric_atol = float(metric_atol)
        self.max_local_size = int(max_local_size)
        self.update_kind = str(update).lower().replace("_", "-")
        if self.update_kind not in {"one-site", "two-site"}:
            raise ValueError("update must be 'one-site' or 'two-site'.")
        self.environment = str(environment).lower().replace("_", "-")
        if self.environment == "auto":
            self.environment = "network" if self.update_kind == "one-site" else "dense"
        if self.environment not in {"network", "dense"}:
            raise ValueError("environment must be 'auto', 'network', or 'dense'.")
        if self.update_kind == "two-site" and self.environment != "dense":
            raise NotImplementedError(
                "two-site PEPS updates currently require environment='dense'."
            )
        if max_D is None:
            max_D = max(
                (dim for values in state.bond_dims.values() for dim in values),
                default=1,
            )
        if isinstance(max_D, bool) or not isinstance(max_D, Integral) or int(max_D) < 1:
            raise ValueError("max_D must be a positive integer.")
        if (
            isinstance(max_pair_size, bool)
            or not isinstance(max_pair_size, Integral)
            or int(max_pair_size) < 1
        ):
            raise ValueError("max_pair_size must be a positive integer.")
        self.max_D = int(max_D)
        self.split_rtol = float(split_rtol)
        self.max_pair_size = int(max_pair_size)
        self.max_dense_dimension = int(max_dense_dimension)
        self.verbose = bool(verbose)
        for name in ("tol", "metric_rtol", "metric_atol", "split_rtol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")

        dimension = int(np.prod(self.state.dims))
        if self.environment == "dense" and dimension > self.max_dense_dimension:
            raise ValueError(
                f"exact PEPS optimization needs dense state dimension {dimension}, "
                f"exceeding max_dense_dimension={self.max_dense_dimension}."
            )
        largest = max(tensor.size for row in state.tensors for tensor in row)
        if largest > self.max_local_size:
            raise ValueError(
                f"largest local PEPS tensor has {largest} elements, exceeding "
                f"max_local_size={self.max_local_size}."
            )

        self.energy = None
        self.history = []
        self.success = False
        self.message = "not started"
        self.n_sweeps = 0
        self.local_updates = 0
        self.algorithm = (
            f"exact-{self.environment}-environment-{self.update_kind}-peps"
        )

    def _energy(self):
        if self.environment == "network":
            return self.state.expectation(self.hamiltonian, method="exact")
        vector = self.state.to_dense()
        return self.hamiltonian.expectation(vector)

    def _normalize_exact(self):
        if self.environment == "network":
            self.state.normalize(method="exact")
            return
        vector = self.state.to_dense()
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise ValueError("cannot optimize a zero or nonfinite PEPS.")
        self.state.tensors[0][0] = self.state.tensors[0][0] / norm
        self.state._touch((0, 0))

    def _basis_map(self, coordinate):
        row, col = coordinate
        original = self.state.tensors[row][col]
        shape = original.shape
        size = original.size
        dimension = int(np.prod(self.state.dims))
        dtype = np.result_type(
            original.dtype,
            self.hamiltonian.dtype,
            np.complex128,
        )
        mapping = np.empty((dimension, size), dtype=dtype)
        try:
            for parameter in range(size):
                basis = np.zeros(shape, dtype=dtype)
                basis.flat[parameter] = 1.0
                self.state.tensors[row][col] = basis
                mapping[:, parameter] = self.state.to_dense()
        finally:
            self.state.tensors[row][col] = original
        return mapping

    def _solve_generalized(self, effective, metric):
        metric = 0.5 * (metric + metric.conj().T)
        metric_values, metric_vectors = np.linalg.eigh(metric)
        metric_scale = max(float(np.max(metric_values)), 0.0)
        threshold = max(self.metric_atol, self.metric_rtol * metric_scale)
        keep = metric_values > threshold
        rank = int(np.count_nonzero(keep))
        if rank == 0:
            raise np.linalg.LinAlgError("local PEPS norm matrix is numerically null.")
        whitening = metric_vectors[:, keep] / np.sqrt(metric_values[keep])[None, :]
        effective = whitening.conj().T @ effective @ whitening
        effective = 0.5 * (effective + effective.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(effective)
        return whitening @ eigenvectors[:, 0], rank, threshold

    def _solve_effective_problem(self, mapping):
        metric = mapping.conj().T @ mapping
        h_mapping = np.column_stack(
            [
                self.hamiltonian.matvec(mapping[:, column])
                for column in range(mapping.shape[1])
            ]
        )
        effective = mapping.conj().T @ h_mapping
        return self._solve_generalized(effective, metric)

    def update(self, coordinate):
        """Optimize one PEPS tensor and return diagnostics for the update."""

        row, col = coordinate
        current_tensor = self.state.tensors[row][col]
        energy_before = self._energy()
        if self.environment == "network":
            effective, metric, _environment_info = self.state.effective_environment(
                self.hamiltonian,
                (row, col),
            )
            candidate, rank, threshold = self._solve_generalized(effective, metric)
            metric_dimension = metric.shape[0]
        else:
            mapping = self._basis_map((row, col))
            candidate, rank, threshold = self._solve_effective_problem(mapping)
            metric_dimension = mapping.shape[1]
        candidate_tensor = candidate.reshape(current_tensor.shape)
        if np.isrealobj(current_tensor) and np.max(np.abs(np.imag(candidate_tensor))) < 1.0e-12:
            candidate_tensor = np.real(candidate_tensor)

        self.state.tensors[row][col] = candidate_tensor
        energy_after = self._energy()
        accepted = energy_after <= energy_before + 1.0e-10 * max(1.0, abs(energy_before))
        if not accepted:
            self.state.tensors[row][col] = current_tensor
            energy_after = energy_before
        else:
            self.local_updates += 1
            self.state._touch((row, col))
        return {
            "coordinate": (row, col),
            "energy_before": float(energy_before),
            "energy_after": float(energy_after),
            "energy_change": float(energy_after - energy_before),
            "metric_rank": rank,
            "metric_dimension": int(metric_dimension),
            "metric_threshold": float(threshold),
            "environment": self.environment,
            "accepted": bool(accepted),
        }

    def _pair_layout(self, first, second):
        first = tuple(int(value) for value in first)
        second = tuple(int(value) for value in second)
        if first[0] == second[0] and abs(first[1] - second[1]) == 1:
            if second[1] < first[1]:
                first, second = second, first
            left = self.state.tensors[first[0]][first[1]]
            right = self.state.tensors[second[0]][second[1]]
            return {
                "orientation": "horizontal",
                "first": first,
                "second": second,
                "left_dims": (left.shape[0], left.shape[1], left.shape[3], left.shape[4]),
                "right_dims": (
                    right.shape[0],
                    right.shape[1],
                    right.shape[2],
                    right.shape[3],
                ),
            }
        if first[1] == second[1] and abs(first[0] - second[0]) == 1:
            if second[0] < first[0]:
                first, second = second, first
            top = self.state.tensors[first[0]][first[1]]
            bottom = self.state.tensors[second[0]][second[1]]
            return {
                "orientation": "vertical",
                "first": first,
                "second": second,
                "left_dims": (top.shape[0], top.shape[1], top.shape[2], top.shape[4]),
                "right_dims": (
                    bottom.shape[0],
                    bottom.shape[2],
                    bottom.shape[3],
                    bottom.shape[4],
                ),
            }
        raise ValueError("a two-site PEPS update requires nearest neighbors.")

    @staticmethod
    def _pair_tensors(left, right, orientation):
        rank = left.shape[-1]
        if orientation == "horizontal":
            first = left.reshape(left.shape[:-1] + (rank,)).transpose(0, 1, 4, 2, 3)
            second = right.reshape((rank,) + right.shape[1:]).transpose(1, 2, 3, 4, 0)
        else:
            first = left.reshape(left.shape[:-1] + (rank,)).transpose(0, 1, 2, 4, 3)
            second = right.reshape((rank,) + right.shape[1:]).transpose(1, 0, 2, 3, 4)
        return first, second

    def _pair_basis_map(self, layout):
        left_dims = layout["left_dims"]
        right_dims = layout["right_dims"]
        left_size = int(np.prod(left_dims))
        right_size = int(np.prod(right_dims))
        pair_size = left_size * right_size
        if pair_size > self.max_pair_size:
            raise ValueError(
                f"two-site PEPS block has {pair_size} elements, exceeding "
                f"max_pair_size={self.max_pair_size}."
            )
        dimension = int(np.prod(self.state.dims))
        dtype = np.result_type(self.hamiltonian.dtype, np.complex128)
        mapping = np.empty((dimension, pair_size), dtype=dtype)
        first_coordinate = layout["first"]
        second_coordinate = layout["second"]
        original_first = self.state.tensors[first_coordinate[0]][first_coordinate[1]]
        original_second = self.state.tensors[second_coordinate[0]][second_coordinate[1]]
        try:
            for left_index in range(left_size):
                left = np.zeros(left_dims + (1,), dtype=dtype)
                left.reshape(left_size, 1)[left_index, 0] = 1.0
                for right_index in range(right_size):
                    right = np.zeros((1,) + right_dims, dtype=dtype)
                    right.reshape(1, right_size)[0, right_index] = 1.0
                    first, second = self._pair_tensors(
                        left,
                        right,
                        layout["orientation"],
                    )
                    self.state.tensors[first_coordinate[0]][first_coordinate[1]] = first
                    self.state.tensors[second_coordinate[0]][second_coordinate[1]] = second
                    column = left_index * right_size + right_index
                    mapping[:, column] = self.state.to_dense()
        finally:
            self.state.tensors[first_coordinate[0]][first_coordinate[1]] = original_first
            self.state.tensors[second_coordinate[0]][second_coordinate[1]] = original_second
        return mapping, left_size, right_size

    def update_pair(self, first, second):
        """Optimize and SVD-split one horizontal or vertical PEPS pair."""

        layout = self._pair_layout(first, second)
        first_coordinate = layout["first"]
        second_coordinate = layout["second"]
        original_first = self.state.tensors[first_coordinate[0]][first_coordinate[1]]
        original_second = self.state.tensors[second_coordinate[0]][second_coordinate[1]]
        energy_before = self._energy()
        mapping, left_size, right_size = self._pair_basis_map(layout)
        candidate, metric_rank, threshold = self._solve_effective_problem(mapping)
        matrix = candidate.reshape(left_size, right_size)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        singular_scale = float(singular_values[0]) if singular_values.size else 0.0
        numerical_rank = max(
            1,
            int(np.count_nonzero(singular_values > self.split_rtol * singular_scale)),
        )
        bond_rank = min(self.max_D, numerical_rank)
        discarded = float(np.sum(np.abs(singular_values[bond_rank:]) ** 2))
        total = float(np.sum(np.abs(singular_values) ** 2))
        root = np.sqrt(singular_values[:bond_rank])
        left = (u[:, :bond_rank] * root[None, :]).reshape(
            layout["left_dims"] + (bond_rank,)
        )
        right = (root[:, None] * vh[:bond_rank]).reshape(
            (bond_rank,) + layout["right_dims"]
        )
        candidate_first, candidate_second = self._pair_tensors(
            left,
            right,
            layout["orientation"],
        )
        if np.isrealobj(original_first) and np.max(np.abs(np.imag(candidate_first))) < 1.0e-12:
            candidate_first = np.real(candidate_first)
        if np.isrealobj(original_second) and np.max(np.abs(np.imag(candidate_second))) < 1.0e-12:
            candidate_second = np.real(candidate_second)
        self.state.tensors[first_coordinate[0]][first_coordinate[1]] = candidate_first
        self.state.tensors[second_coordinate[0]][second_coordinate[1]] = candidate_second
        energy_after = self._energy()
        accepted = energy_after <= energy_before + 1.0e-10 * max(1.0, abs(energy_before))
        if not accepted:
            self.state.tensors[first_coordinate[0]][first_coordinate[1]] = original_first
            self.state.tensors[second_coordinate[0]][second_coordinate[1]] = original_second
            energy_after = energy_before
        else:
            self.local_updates += 1
            self.state._touch(first_coordinate, second_coordinate)
        return {
            "coordinates": (first_coordinate, second_coordinate),
            "orientation": layout["orientation"],
            "energy_before": float(energy_before),
            "energy_after": float(energy_after),
            "energy_change": float(energy_after - energy_before),
            "metric_rank": metric_rank,
            "metric_dimension": int(mapping.shape[1]),
            "metric_threshold": float(threshold),
            "bond_rank": bond_rank,
            "discarded_weight": discarded,
            "relative_split_error": float(np.sqrt(discarded / total)) if total > 0 else 0.0,
            "accepted": bool(accepted),
        }

    def run(self):
        """Run forward/backward variational sweeps and return this driver."""

        self._normalize_exact()
        self.energy = self._energy()
        coordinates = [
            (row, col)
            for row in range(self.state.nrows)
            for col in range(self.state.ncols)
        ]
        edges = [
            ((row, col), (row, col + 1))
            for row in range(self.state.nrows)
            for col in range(self.state.ncols - 1)
        ] + [
            ((row, col), (row + 1, col))
            for row in range(self.state.nrows - 1)
            for col in range(self.state.ncols)
        ]
        previous = self.energy
        for sweep in range(self.sweeps):
            updates = []
            if self.update_kind == "one-site":
                for coordinate in coordinates:
                    updates.append(self.update(coordinate))
                for coordinate in reversed(coordinates):
                    updates.append(self.update(coordinate))
            else:
                if not edges:
                    raise ValueError("two-site PEPS optimization requires at least one bond.")
                for first, second in edges:
                    updates.append(self.update_pair(first, second))
                for first, second in reversed(edges):
                    updates.append(self.update_pair(first, second))
            self._normalize_exact()
            self.energy = self._energy()
            change = self.energy - previous
            record = {
                "sweep": sweep,
                "energy": float(self.energy),
                "energy_change": float(change),
                "accepted_updates": sum(update["accepted"] for update in updates),
                "updates": tuple(updates),
            }
            self.history.append(record)
            if self.verbose:
                print(
                    f"PEPS sweep={sweep:3d} E={self.energy:.12f} "
                    f"dE={change:+.3e} accepted={record['accepted_updates']}"
                )
            self.n_sweeps = sweep + 1
            if abs(change) <= self.tol * max(1.0, abs(self.energy)):
                self.success = True
                self.message = "energy converged"
                break
            previous = self.energy
        else:
            self.success = True
            self.message = "maximum sweeps reached"

        self.state.energy = float(self.energy)
        self.state.history = list(self.history)
        self.state.success = self.success
        self.state.message = self.message
        return self


__all__ = ["PEPSOptimizer"]
