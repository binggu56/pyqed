"""Real- and imaginary-time evolution for finite PEPS."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.linalg import expm

from pyqed.tn import Hamiltonian


def _retained_rank(singular_values, max_rank, cutoff):
    singular_values = np.asarray(singular_values)
    if singular_values.size == 0:
        return 1
    threshold = float(cutoff) * float(singular_values[0])
    rank = max(1, int(np.count_nonzero(singular_values > threshold)))
    return min(rank, int(max_rank))


def _split_pair_matrix(matrix, left_dims, right_dims, orientation, max_D, cutoff):
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = _retained_rank(singular_values, max_D, cutoff)
    discarded = float(np.sum(np.abs(singular_values[rank:]) ** 2))
    total = float(np.sum(np.abs(singular_values) ** 2))
    root = np.sqrt(singular_values[:rank])
    left = (u[:, :rank] * root[None, :]).reshape(tuple(left_dims) + (rank,))
    right = (root[:, None] * vh[:rank]).reshape((rank,) + tuple(right_dims))
    if orientation == "horizontal":
        first = left.transpose(0, 1, 4, 2, 3)
        second = right.transpose(1, 2, 3, 4, 0)
    else:
        first = left.transpose(0, 1, 2, 4, 3)
        second = right.transpose(1, 0, 2, 3, 4)
    return first, second, {
        "kept_rank": rank,
        "discarded_weight": discarded,
        "relative_error": float(np.sqrt(discarded / total)) if total > 0.0 else 0.0,
    }


def apply_peps_pair_gate(state, first, second, gate, *, max_D, cutoff=1.0e-12):
    """Apply one nearest-neighbor gate and SVD-split the updated PEPS pair."""

    first = tuple(int(value) for value in first)
    second = tuple(int(value) for value in second)
    if first[0] == second[0] and second[1] == first[1] + 1:
        orientation = "horizontal"
        first_axis, second_axis = 2, 4
        left_dims = (
            state.tensors[first[0]][first[1]].shape[0],
            state.tensors[first[0]][first[1]].shape[1],
            state.tensors[first[0]][first[1]].shape[3],
            state.tensors[first[0]][first[1]].shape[4],
        )
        right_dims = (
            state.tensors[second[0]][second[1]].shape[0],
            state.tensors[second[0]][second[1]].shape[1],
            state.tensors[second[0]][second[1]].shape[2],
            state.tensors[second[0]][second[1]].shape[3],
        )
    elif first[1] == second[1] and second[0] == first[0] + 1:
        orientation = "vertical"
        first_axis, second_axis = 3, 1
        left_dims = (
            state.tensors[first[0]][first[1]].shape[0],
            state.tensors[first[0]][first[1]].shape[1],
            state.tensors[first[0]][first[1]].shape[2],
            state.tensors[first[0]][first[1]].shape[4],
        )
        right_dims = (
            state.tensors[second[0]][second[1]].shape[0],
            state.tensors[second[0]][second[1]].shape[2],
            state.tensors[second[0]][second[1]].shape[3],
            state.tensors[second[0]][second[1]].shape[4],
        )
    else:
        raise ValueError("PEPS gates require rightward or downward nearest neighbors.")

    first_tensor = state.tensors[first[0]][first[1]]
    second_tensor = state.tensors[second[0]][second[1]]
    theta = np.tensordot(first_tensor, second_tensor, axes=(first_axis, second_axis))
    first_dim, second_dim = left_dims[0], right_dims[0]
    gate = np.asarray(gate)
    if gate.shape == (first_dim * second_dim, first_dim * second_dim):
        gate = gate.reshape(first_dim, second_dim, first_dim, second_dim)
    if gate.shape != (first_dim, second_dim, first_dim, second_dim):
        raise ValueError("two-site gate has incompatible physical dimensions.")
    theta = np.tensordot(gate, theta, axes=((2, 3), (0, 4))).transpose(
        0, 2, 3, 4, 1, 5, 6, 7
    )
    matrix = theta.reshape(int(np.prod(left_dims)), int(np.prod(right_dims)))
    updated_first, updated_second, info = _split_pair_matrix(
        matrix,
        left_dims,
        right_dims,
        orientation,
        max_D,
        cutoff,
    )
    state.tensors[first[0]][first[1]] = updated_first
    state.tensors[second[0]][second[1]] = updated_second
    state._touch(first, second)
    info.update(
        {
            "coordinates": (first, second),
            "orientation": orientation,
        }
    )
    return info


def apply_peps_local_gate(state, coordinate, gate):
    """Apply a one-site physical gate to a PEPS tensor."""

    row, col = (int(value) for value in coordinate)
    tensor = state.tensors[row][col]
    gate = np.asarray(gate)
    if gate.shape != (tensor.shape[0], tensor.shape[0]):
        raise ValueError("one-site gate has an incompatible physical dimension.")
    state.tensors[row][col] = np.tensordot(gate, tensor, axes=(1, 0))
    state._touch((row, col))


class PEPSEvolution:
    """Nearest-neighbor Trotter evolution using local PEPS SVD updates."""

    def __init__(
        self,
        state,
        hamiltonian,
        *,
        max_D=None,
        cutoff=1.0e-12,
        order=2,
        imaginary=False,
        contraction="boundary",
        chi=64,
        max_frontiers=None,
        contraction_rtol=1.0e-10,
        normalize=True,
        measure_every=1,
        workers=1,
    ):
        from .state import PEPS
        from .symmetry import U1PEPS

        if not isinstance(state, (PEPS, U1PEPS)):
            raise TypeError("state must be a PEPS or U1PEPS.")
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a pyqed.tn.Hamiltonian.")
        if state.dims != hamiltonian.dims:
            raise ValueError("Hamiltonian physical dimensions do not match the PEPS.")
        if max_D is None:
            max_D = max(
                (dim for values in state.bond_dims.values() for dim in values),
                default=1,
            )
        if isinstance(max_D, bool) or not isinstance(max_D, Integral) or int(max_D) < 1:
            raise ValueError("max_D must be a positive integer.")
        if int(order) not in {1, 2}:
            raise ValueError("order must be 1 or 2.")
        self.state = state
        self.block_sparse = isinstance(state, U1PEPS)
        self.hamiltonian = hamiltonian
        self.max_D = int(max_D)
        self.cutoff = float(cutoff)
        if not np.isfinite(self.cutoff) or self.cutoff < 0.0:
            raise ValueError("cutoff must be finite and nonnegative.")
        self.order = int(order)
        self.imaginary = bool(imaginary)
        self.contraction = str(contraction)
        self.chi = chi
        if max_frontiers is not None:
            if (
                isinstance(max_frontiers, bool)
                or not isinstance(max_frontiers, Integral)
                or int(max_frontiers) < 1
            ):
                raise ValueError("max_frontiers must be a positive integer or None.")
            max_frontiers = int(max_frontiers)
        self.max_frontiers = max_frontiers
        self.contraction_rtol = float(contraction_rtol)
        self.normalize_each_step = bool(normalize)
        if (
            isinstance(measure_every, bool)
            or not isinstance(measure_every, Integral)
            or int(measure_every) < 1
        ):
            raise ValueError("measure_every must be a positive integer.")
        self.measure_every = int(measure_every)
        if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
            raise ValueError("workers must be a positive integer.")
        self.workers = int(workers)
        self.time = 0.0
        self.beta = 0.0
        self.history = []
        self.energy = None
        self.success = True
        self.message = "initialized"
        self._known_norm = None
        self._terms = self._validate_terms()

    def _validate_terms(self):
        terms = []
        for term in self.hamiltonian.terms:
            if len(term.sites) == 1:
                terms.append(term)
                continue
            if len(term.sites) != 2:
                raise NotImplementedError(
                    "PEPS Trotter evolution supports one- and two-site terms."
                )
            first = self.state.coordinate(term.sites[0])
            second = self.state.coordinate(term.sites[1])
            distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
            if distance != 1:
                raise NotImplementedError(
                    "PEPS Trotter evolution requires nearest-neighbor terms."
                )
            terms.append(term)
        return tuple(terms)

    def _gate(self, operator, increment):
        coefficient = -float(increment) if self.imaginary else -1j * float(increment)
        return expm(coefficient * np.asarray(operator))

    def _apply_term(self, term, increment):
        gate = self._gate(term.operator, increment)
        if len(term.sites) == 1:
            if self.block_sparse:
                from .symmetry import apply_u1_peps_local_gate

                apply_u1_peps_local_gate(
                    self.state,
                    self.state.coordinate(term.sites[0]),
                    gate,
                )
            else:
                apply_peps_local_gate(
                    self.state,
                    self.state.coordinate(term.sites[0]),
                    gate,
                )
            return {
                "sites": term.sites,
                "kind": "one-site",
                "discarded_weight": 0.0,
                "relative_error": 0.0,
                "backend": "u1-block" if self.block_sparse else "dense",
            }
        first = self.state.coordinate(term.sites[0])
        second = self.state.coordinate(term.sites[1])
        if second < first:
            first, second = second, first
        if self.block_sparse:
            from .symmetry import apply_u1_peps_pair_gate

            info = apply_u1_peps_pair_gate(
                self.state,
                first,
                second,
                gate,
                max_D=self.max_D,
                cutoff=self.cutoff,
            )
        else:
            info = apply_peps_pair_gate(
                self.state,
                first,
                second,
                gate,
                max_D=self.max_D,
                cutoff=self.cutoff,
            )
            info["backend"] = "dense"
        info.update({"sites": term.sites, "kind": "two-site"})
        return info

    def step(self, increment, *, measure=None):
        """Advance by one positive real- or imaginary-time increment."""

        increment = float(increment)
        if not np.isfinite(increment) or increment <= 0.0:
            raise ValueError("increment must be finite and positive.")
        norm_before = self._known_norm
        if norm_before is None:
            norm_before = self._norm_squared()
        updates = []
        if self.order == 1:
            sequence = tuple((term, increment) for term in self._terms)
        else:
            half = tuple((term, 0.5 * increment) for term in self._terms)
            sequence = half + tuple(reversed(half))
        for term, amount in sequence:
            updates.append(self._apply_term(term, amount))

        if self.hamiltonian.constant != 0.0:
            factor = np.exp(
                (-1.0 if self.imaginary else -1j)
                * increment
                * self.hamiltonian.constant
            )
            if self.block_sparse:
                self.state.tensors[0][0] = self.state.tensors[0][0].scaled(factor)
                self.state._touch((0, 0))
            else:
                self.state.tensors[0][0] = factor * self.state.tensors[0][0]
                self.state._touch((0, 0))
        norm_after = self._norm_squared()
        if self.normalize_each_step:
            norm_value = np.real_if_close(norm_after)
            if abs(np.imag(norm_value)) > 1.0e-10 * max(1.0, abs(norm_value)):
                raise FloatingPointError("PEPS evolution produced a complex norm.")
            norm_value = float(np.real(norm_value))
            if not np.isfinite(norm_value) or norm_value <= np.finfo(float).tiny:
                raise FloatingPointError("PEPS evolution produced a nonpositive norm.")
            if self.block_sparse:
                self.state.tensors[0][0] = self.state.tensors[0][0].scaled(
                    1.0 / np.sqrt(norm_value)
                )
                self.state._touch((0, 0))
            else:
                self.state.tensors[0][0] = self.state.tensors[0][0] / np.sqrt(
                    norm_value
                )
                self.state._touch((0, 0))
            self._known_norm = 1.0
        else:
            self._known_norm = norm_after
        if self.imaginary:
            self.beta += increment
        else:
            self.time += increment
        if measure is None:
            measure = (len(self.history) + 1) % self.measure_every == 0
        measure = bool(measure)
        measured_energy = self._energy() if measure else None
        if measured_energy is not None:
            self.energy = measured_energy
        record = {
            "step": len(self.history),
            "time": self.time,
            "beta": self.beta,
            "energy": measured_energy,
            "measured": measure,
            "norm_before": np.real_if_close(norm_before).item(),
            "norm_after": np.real_if_close(norm_after).item(),
            "max_bond": max(
                (dim for values in self.state.bond_dims.values() for dim in values),
                default=1,
            ),
            "discarded_weight": sum(item["discarded_weight"] for item in updates),
            "max_relative_error": max(
                (item["relative_error"] for item in updates),
                default=0.0,
            ),
            "updates": tuple(updates),
        }
        self.history.append(record)
        return record

    def _norm_squared(self):
        if self.block_sparse:
            return self.state.norm_squared(
                method=self.contraction,
                max_frontiers=self.max_frontiers,
                rtol=0.0,
            )
        return self.state.norm_squared(
            method=self.contraction,
            max_bond=self.chi,
            rtol=self.contraction_rtol,
        )

    def _energy(self):
        if self.block_sparse:
            return self.state.expectation(
                self.hamiltonian,
                method=self.contraction,
                max_frontiers=self.max_frontiers,
                rtol=0.0,
                workers=self.workers,
            )
        return self.state.expectation(
            self.hamiltonian,
            method=self.contraction,
            max_bond=self.chi,
            rtol=self.contraction_rtol,
            workers=self.workers,
        )

    def run(self, target, *, step=0.05, verbose=False):
        """Evolve to an absolute target ``time`` or inverse time ``beta``."""

        target = float(target)
        step = float(step)
        current = self.beta if self.imaginary else self.time
        if not np.isfinite(target) or target < current:
            raise ValueError("target must be finite and not smaller than the current value.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive.")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(target))
        while target - current > tolerance:
            increment = min(step, target - current)
            next_step = len(self.history) + 1
            reaches_target = target - (current + increment) <= tolerance
            record = self.step(
                increment,
                measure=(
                    reaches_target
                    or next_step % self.measure_every == 0
                ),
            )
            current = self.beta if self.imaginary else self.time
            if verbose:
                coordinate = "beta" if self.imaginary else "t"
                energy = (
                    f"{record['energy']:.12f}"
                    if record["measured"]
                    else "not measured"
                )
                print(
                    f"PEPS evolution step={record['step']:4d} "
                    f"{coordinate}={current:.8f} E={energy} "
                    f"D={record['max_bond']} trunc={record['discarded_weight']:.3e}"
                )
        if self.imaginary:
            self.beta = target
        else:
            self.time = target
        self.success = True
        self.message = f"reached {'beta' if self.imaginary else 'time'}={target:g}"
        return self


__all__ = [
    "PEPSEvolution",
    "apply_peps_local_gate",
    "apply_peps_pair_gate",
]
