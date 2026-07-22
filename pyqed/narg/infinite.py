"""Infinite-lattice conditional NARG for uniform nearest-neighbor chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "InfiniteNARG",
    "iNARG",
    "iNARGStep",
    "inarg_nearest_neighbor",
]


def _as_two_site_operator(operator, physical_dim=None):
    op = np.asarray(operator)
    if op.ndim == 2:
        if op.shape[0] != op.shape[1]:
            raise ValueError("nearest-neighbor operator must be square.")
        dim = int(round(np.sqrt(op.shape[0])))
        if op.shape != (dim * dim, dim * dim):
            raise ValueError("nearest-neighbor operator must have dimension d**2.")
        if physical_dim is not None and int(physical_dim) != dim:
            raise ValueError("physical_dim is inconsistent with the operator dimension.")
        return op.reshape(dim, dim, dim, dim), dim
    if op.ndim == 4:
        dim = int(op.shape[0])
        if op.shape != (dim, dim, dim, dim):
            raise ValueError("rank-4 nearest-neighbor operator must have shape (d, d, d, d).")
        if physical_dim is not None and int(physical_dim) != dim:
            raise ValueError("physical_dim is inconsistent with the operator dimension.")
        return op, dim
    raise ValueError("nearest-neighbor operator must be rank 2 or rank 4.")


def _as_hermitian(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    return 0.5 * (matrix + matrix.T.conj())


@dataclass(frozen=True)
class iNARGStep:
    """Diagnostics and tensor from one branch-conditioned iNARG growth step."""

    iteration: int
    length: int
    input_dim: int
    kept_dim: int
    energy: float
    energy_density: float
    growth_energy: float | None
    tensor: np.ndarray
    branch_energies: np.ndarray
    effective_hamiltonian: np.ndarray
    boundary_ops: tuple[np.ndarray, ...]


class iNARG:
    """Infinite NARG by iterating a conditional branch basis.

    The state after a growth step is represented in row order
    ``(kept_state, boundary_site)``.  For each possible value of the newly
    appended boundary site, the previous effective Hamiltonian plus the branch
    value of the renormalized boundary coupling is diagonalized.  The lowest
    ``bond_dim`` states in every branch define the NARG tensor
    ``T[(alpha, s), beta, t]``.  Off-diagonal branch couplings and new boundary
    operators are then projected through these conditional bases.
    """

    def __init__(
        self,
        hamiltonian,
        *,
        physical_dim=None,
        bond_dim=16,
        growth_sites=1,
        maxiter=20,
        tol=1.0e-10,
    ):
        h4, dim = _as_two_site_operator(hamiltonian, physical_dim=physical_dim)
        self.hamiltonian = np.asarray(h4, dtype=np.result_type(h4.dtype, np.complex128))
        self.physical_dim = dim
        self.bond_dim = int(bond_dim)
        if self.bond_dim <= 0:
            raise ValueError("bond_dim must be positive.")
        self.growth_sites = int(growth_sites)
        if self.growth_sites not in {1, 2}:
            raise ValueError("growth_sites must be 1 or 2.")
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.effective_hamiltonian: np.ndarray | None = None
        self.boundary_ops: tuple[np.ndarray, ...] = ()
        self.tensor: np.ndarray | None = None
        self.history: tuple[iNARGStep, ...] = ()
        self.energies = np.asarray([], dtype=float)
        self.vectors = np.empty((0, 0), dtype=complex)
        self.energy = float("nan")
        self.energy_density = float("nan")
        self.success = False
        self.message = "not run"
        self.algorithm = "inarg"
        self.metadata: dict[str, object] = {
            "bond_dim": self.bond_dim,
            "physical_dim": self.physical_dim,
            "growth_sites": self.growth_sites,
            "truncation": "conditional-branch-energy",
        }
        self._coeffs, self._left_ops, self._right_ops = self._factorize_interaction()

    def _factorize_interaction(self, tol=1.0e-12):
        d = self.physical_dim
        matrix = self.hamiltonian.transpose(0, 2, 1, 3).reshape(d * d, d * d)
        u, values, vh = np.linalg.svd(matrix, full_matrices=False)
        if values.size == 0:
            keep = np.zeros(0, dtype=bool)
        else:
            keep = values > float(tol) * max(float(values[0]), 1.0)
        coeffs = values[keep].astype(complex, copy=False)
        left_ops = tuple(u[:, idx].reshape(d, d) for idx in np.flatnonzero(keep))
        right_ops = tuple(vh[idx].reshape(d, d) for idx in np.flatnonzero(keep))
        if not left_ops:
            raise ValueError("nearest-neighbor Hamiltonian factorization produced no channels.")
        return coeffs, left_ops, right_ops

    def initial_effective_hamiltonian(self):
        """Return the one-boundary-site starting Hamiltonian."""

        return np.zeros((self.physical_dim, self.physical_dim), dtype=complex)

    def initial_boundary_ops(self):
        """Return operators on the initial boundary site used by the next bond."""

        return tuple(np.asarray(op, dtype=complex) for op in self._left_ops)

    def _branch_coupling(self, boundary_ops, branch_out, branch_in):
        out = np.zeros_like(boundary_ops[0], dtype=complex)
        for coeff, op, right in zip(self._coeffs, boundary_ops, self._right_ops):
            out += coeff * right[int(branch_out), int(branch_in)] * op
        return out

    def _diagonalize_branch(self, matrix, keep):
        values, vectors = np.linalg.eigh(_as_hermitian(matrix))
        keep = min(int(keep), values.size)
        return values[:keep], vectors[:, :keep]

    def _renormalize_boundary_ops(self, tensor):
        d = self.physical_dim
        keep = int(tensor.shape[1])
        next_ops = []
        for local_op in self._left_ops:
            projected = np.zeros((keep * d, keep * d), dtype=complex)
            for out_branch in range(d):
                rows = slice(out_branch, keep * d, d)
                left = tensor[:, :, out_branch]
                for in_branch in range(d):
                    cols = slice(in_branch, keep * d, d)
                    right = tensor[:, :, in_branch]
                    projected[rows, cols] = local_op[out_branch, in_branch] * (left.conj().T @ right)
            next_ops.append(projected)
        return tuple(next_ops)

    def _grow_one_site(self, effective_hamiltonian, boundary_ops, iteration, previous_energy):
        h_eff = _as_hermitian(effective_hamiltonian)
        input_dim = int(h_eff.shape[0])
        d = self.physical_dim
        if h_eff.shape != (input_dim, input_dim) or input_dim % d:
            raise ValueError("effective Hamiltonian dimension must be a multiple of physical_dim.")
        if len(boundary_ops) != len(self._coeffs):
            raise ValueError("boundary_ops must match the Hamiltonian channel count.")
        if any(np.asarray(op).shape != h_eff.shape for op in boundary_ops):
            raise ValueError("all boundary operators must match the effective Hamiltonian shape.")
        keep = min(self.bond_dim, input_dim)

        tensor = np.empty((input_dim, keep, d), dtype=complex)
        branch_energies = np.empty((d, keep), dtype=float)
        diagonal_blocks = []
        for branch in range(d):
            branch_h = h_eff + self._branch_coupling(boundary_ops, branch, branch)
            values, vectors = self._diagonalize_branch(branch_h, keep)
            branch_energies[branch] = values
            tensor[:, :, branch] = vectors
            diagonal_blocks.append(values)

        next_h = np.zeros((keep * d, keep * d), dtype=complex)
        for out_branch in range(d):
            rows = slice(out_branch, keep * d, d)
            next_h[rows, rows] = np.diag(diagonal_blocks[out_branch])
            left = tensor[:, :, out_branch]
            for in_branch in range(d):
                if out_branch == in_branch:
                    continue
                cols = slice(in_branch, keep * d, d)
                coupling = self._branch_coupling(boundary_ops, out_branch, in_branch)
                right = tensor[:, :, in_branch]
                next_h[rows, cols] = left.conj().T @ coupling @ right
        next_h = _as_hermitian(next_h)
        next_boundary_ops = self._renormalize_boundary_ops(tensor)
        energies, vectors = np.linalg.eigh(next_h)
        energy = float(np.real(energies[0]))
        growth = None if previous_energy is None else float(energy - previous_energy)
        density = float(energy) if growth is None else growth

        return iNARGStep(
            iteration=int(iteration),
            length=int(iteration + 1),
            input_dim=input_dim,
            kept_dim=keep,
            energy=energy,
            energy_density=density,
            growth_energy=growth,
            tensor=tensor,
            branch_energies=branch_energies,
            effective_hamiltonian=next_h,
            boundary_ops=next_boundary_ops,
        ), energies, vectors

    def _two_site_branch_coupling(self, boundary_ops, out_branch, in_branch):
        out_first, out_edge = out_branch
        in_first, in_edge = in_branch
        out = self.hamiltonian[out_first, out_edge, in_first, in_edge] * np.eye(
            boundary_ops[0].shape[0],
            dtype=complex,
        )
        if out_edge == in_edge:
            out = out + self._branch_coupling(boundary_ops, out_first, in_first)
        return out

    def _renormalize_two_site_boundary_ops(self, tensor):
        d = self.physical_dim
        keep = int(tensor.shape[1])
        next_ops = []
        for local_op in self._left_ops:
            projected = np.zeros((keep * d * d, keep * d * d), dtype=complex)
            for out_first in range(d):
                for out_edge in range(d):
                    out_pos = out_first * d + out_edge
                    rows = slice(out_pos, keep * d * d, d * d)
                    left = tensor[:, :, out_first, out_edge]
                    for in_first in range(d):
                        if out_first != in_first:
                            continue
                        for in_edge in range(d):
                            in_pos = in_first * d + in_edge
                            cols = slice(in_pos, keep * d * d, d * d)
                            right = tensor[:, :, in_first, in_edge]
                            projected[rows, cols] = (
                                local_op[out_edge, in_edge] * (left.conj().T @ right)
                            )
            next_ops.append(projected)
        return tuple(next_ops)

    def _grow_two_site(self, effective_hamiltonian, boundary_ops, iteration, previous_energy):
        h_eff = _as_hermitian(effective_hamiltonian)
        input_dim = int(h_eff.shape[0])
        d = self.physical_dim
        if h_eff.shape != (input_dim, input_dim) or input_dim % d:
            raise ValueError("effective Hamiltonian dimension must be a multiple of physical_dim.")
        if len(boundary_ops) != len(self._coeffs):
            raise ValueError("boundary_ops must match the Hamiltonian channel count.")
        if any(np.asarray(op).shape != h_eff.shape for op in boundary_ops):
            raise ValueError("all boundary operators must match the effective Hamiltonian shape.")
        keep = min(self.bond_dim, input_dim)

        tensor = np.empty((input_dim, keep, d, d), dtype=complex)
        branch_energies = np.empty((d, d, keep), dtype=float)
        diagonal_blocks = {}
        for first in range(d):
            for edge in range(d):
                branch = (first, edge)
                branch_h = h_eff + self._two_site_branch_coupling(boundary_ops, branch, branch)
                values, vectors = self._diagonalize_branch(branch_h, keep)
                branch_energies[first, edge] = values
                tensor[:, :, first, edge] = vectors
                diagonal_blocks[branch] = values

        branch_dim = d * d
        next_h = np.zeros((keep * branch_dim, keep * branch_dim), dtype=complex)
        for out_first in range(d):
            for out_edge in range(d):
                out_branch = (out_first, out_edge)
                out_pos = out_first * d + out_edge
                rows = slice(out_pos, keep * branch_dim, branch_dim)
                next_h[rows, rows] = np.diag(diagonal_blocks[out_branch])
                left = tensor[:, :, out_first, out_edge]
                for in_first in range(d):
                    for in_edge in range(d):
                        in_branch = (in_first, in_edge)
                        if out_branch == in_branch:
                            continue
                        in_pos = in_first * d + in_edge
                        cols = slice(in_pos, keep * branch_dim, branch_dim)
                        coupling = self._two_site_branch_coupling(
                            boundary_ops,
                            out_branch,
                            in_branch,
                        )
                        right = tensor[:, :, in_first, in_edge]
                        next_h[rows, cols] = left.conj().T @ coupling @ right
        next_h = _as_hermitian(next_h)
        next_boundary_ops = self._renormalize_two_site_boundary_ops(tensor)
        energies, vectors = np.linalg.eigh(next_h)
        energy = float(np.real(energies[0]))
        growth = None if previous_energy is None else float(energy - previous_energy)
        density = float(energy / 2.0) if growth is None else float(growth / 2.0)

        return iNARGStep(
            iteration=int(iteration),
            length=int(1 + 2 * iteration),
            input_dim=input_dim,
            kept_dim=keep,
            energy=energy,
            energy_density=density,
            growth_energy=growth,
            tensor=tensor,
            branch_energies=branch_energies,
            effective_hamiltonian=next_h,
            boundary_ops=next_boundary_ops,
        ), energies, vectors

    def run(self, *, maxiter=None, effective_hamiltonian=None, boundary_ops=None):
        """Run conditional infinite growth, populate this solver, and return it."""

        h_eff = (
            self.initial_effective_hamiltonian()
            if effective_hamiltonian is None
            else _as_hermitian(effective_hamiltonian)
        )
        ops = self.initial_boundary_ops() if boundary_ops is None else tuple(boundary_ops)
        niter = self.maxiter if maxiter is None else int(maxiter)
        history = []
        previous_energy = None
        previous_density = None
        success = False
        message = "maximum iterations reached"
        energies = np.asarray([], dtype=float)
        vectors = np.empty((0, 0), dtype=complex)

        for iteration in range(1, niter + 1):
            if self.growth_sites == 1:
                step, energies, vectors = self._grow_one_site(
                    h_eff,
                    ops,
                    iteration,
                    previous_energy,
                )
            else:
                step, energies, vectors = self._grow_two_site(
                    h_eff,
                    ops,
                    iteration,
                    previous_energy,
                )
            history.append(step)
            h_eff = step.effective_hamiltonian
            ops = step.boundary_ops
            if previous_density is not None and abs(step.energy_density - previous_density) < self.tol:
                success = True
                message = "converged"
                break
            previous_energy = step.energy
            previous_density = step.energy_density

        self.effective_hamiltonian = h_eff
        self.boundary_ops = tuple(ops)
        self.tensor = history[-1].tensor if history else None
        self.history = tuple(history)
        self.energies = np.real_if_close(energies).astype(float, copy=False)
        self.vectors = vectors
        self.energy = float(self.energies[0]) if self.energies.size else float("nan")
        self.energy_density = history[-1].energy_density if history else float("nan")
        self.success = success
        self.message = message
        self.metadata = {
            **self.metadata,
            "nit": len(history),
            "effective_dim": None if h_eff is None else int(h_eff.shape[0]),
        }
        return self

    @property
    def fixed_layer(self):
        """Return the last conditional NARG tensor ``T[(alpha,s), beta, t]``."""

        return self.tensor

    @property
    def transition_tensor(self):
        """Return ``T[alpha, s, beta, t]`` when the input bond is factorizable."""

        if self.tensor is None:
            return None
        d = self.physical_dim
        prev_kept = self.tensor.shape[0] // d
        if self.growth_sites == 1:
            return self.tensor.reshape(prev_kept, d, self.tensor.shape[1], d)
        return self.tensor.reshape(prev_kept, d, self.tensor.shape[1], d, d)

    @classmethod
    def nearest_neighbor(cls, hamiltonian, **kwargs):
        """Construct and run iNARG for a nearest-neighbor Hamiltonian."""

        return cls(hamiltonian, **kwargs).run()


InfiniteNARG = iNARG


def inarg_nearest_neighbor(hamiltonian, **kwargs):
    """Convenience wrapper around :class:`iNARG`."""

    return iNARG.nearest_neighbor(hamiltonian, **kwargs)
