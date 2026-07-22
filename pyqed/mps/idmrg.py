"""Infinite-system DMRG for translationally invariant nearest-neighbor chains.

This module is intentionally separate from the finite sweep DMRG and quantum
chemistry code.  It provides a compact long-term home for infinite-MPS
algorithms: start with dense nearest-neighbor Hamiltonians, keep explicit
renormalized block Hamiltonians and boundary operators, and return a
``UniformMPS`` when the optimized center has a square virtual layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .umps import UniformMPS

__all__ = [
    "InfiniteDMRG",
    "NearestNeighborTerms",
    "factorize_nearest_neighbor_hamiltonian",
    "idmrg_nearest_neighbor",
    "iDMRG",
    "iDMRGBlock",
    "iDMRGStep",
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


def _real_if_close_scalar(value):
    value = np.real_if_close(value)
    if np.ndim(value) == 0:
        return value.item()
    return value


@dataclass(frozen=True)
class NearestNeighborTerms:
    """Local channel factorization of a two-site Hamiltonian.

    The convention is

    ``h[left_out, right_out, left_in, right_in] =
    sum_k coeffs[k] * left_ops[k][left_out, left_in]
    * right_ops[k][right_out, right_in]``.
    """

    coeffs: np.ndarray
    left_ops: tuple[np.ndarray, ...]
    right_ops: tuple[np.ndarray, ...]

    @property
    def physical_dim(self):
        return int(self.left_ops[0].shape[0]) if self.left_ops else 0

    @property
    def nterms(self):
        return int(len(self.coeffs))

    def reconstruct(self):
        d = self.physical_dim
        out = np.zeros((d, d, d, d), dtype=np.result_type(self.coeffs.dtype, np.complex128))
        for coeff, left, right in zip(self.coeffs, self.left_ops, self.right_ops):
            out += coeff * np.einsum("ac,bd->abcd", left, right, optimize=True)
        return np.real_if_close(out)


def factorize_nearest_neighbor_hamiltonian(operator, *, physical_dim=None, tol=1.0e-12):
    """Factor a dense two-site Hamiltonian into local left/right channels."""

    h4, dim = _as_two_site_operator(operator, physical_dim=physical_dim)
    dtype = np.result_type(h4.dtype, np.complex128)
    matrix = np.asarray(h4, dtype=dtype).transpose(0, 2, 1, 3).reshape(dim * dim, dim * dim)
    u, values, vh = np.linalg.svd(matrix, full_matrices=False)
    if values.size == 0:
        keep = np.zeros(0, dtype=bool)
    else:
        cutoff = float(tol) * max(float(values[0]), 1.0)
        keep = values > cutoff
    coeffs = values[keep].astype(dtype, copy=False)
    left_ops = tuple(u[:, i].reshape(dim, dim) for i in np.flatnonzero(keep))
    right_ops = tuple(vh[i, :].reshape(dim, dim) for i in np.flatnonzero(keep))
    return NearestNeighborTerms(coeffs=coeffs, left_ops=left_ops, right_ops=right_ops)


@dataclass(frozen=True)
class iDMRGBlock:
    """Renormalized block data for one side of an infinite-system growth step."""

    length: int
    hamiltonian: np.ndarray
    edge_ops: tuple[np.ndarray, ...]

    @property
    def dim(self):
        return int(self.hamiltonian.shape[0])


@dataclass(frozen=True)
class iDMRGStep:
    """Diagnostics from one iDMRG growth/truncation step."""

    iteration: int
    length: int
    energy: float
    energy_per_site: float
    growth_energy_per_site: float | None
    center_bond_energy: float
    kept_dim: int
    truncation_error: float
    entropy: float


class iDMRG:
    """Two-site infinite-system DMRG for translationally invariant chains."""

    def __init__(
        self,
        hamiltonian,
        *,
        physical_dim=None,
        bond_dim=16,
        maxiter=20,
        tol=1.0e-10,
        factor_tol=1.0e-12,
        svd_tol=0.0,
        solver="auto",
        dense_dim_limit=4096,
        state_energy_tol=1.0e-3,
    ):
        h4, dim = _as_two_site_operator(hamiltonian, physical_dim=physical_dim)
        self.hamiltonian = np.asarray(h4, dtype=np.result_type(h4.dtype, np.complex128))
        self.physical_dim = dim
        self.bond_dim = int(bond_dim)
        if self.bond_dim <= 0:
            raise ValueError("bond_dim must be positive.")
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.factor_tol = float(factor_tol)
        self.svd_tol = float(svd_tol)
        self.solver = str(solver)
        self.dense_dim_limit = int(dense_dim_limit)
        self.state_energy_tol = float(state_energy_tol)
        self.terms = factorize_nearest_neighbor_hamiltonian(
            self.hamiltonian,
            physical_dim=dim,
            tol=factor_tol,
        )
        if self.terms.nterms == 0:
            raise ValueError("nearest-neighbor Hamiltonian factorization produced no channels.")
        self.state: UniformMPS | None = None
        self.left_block: iDMRGBlock | None = None
        self.right_block: iDMRGBlock | None = None
        self.history: tuple[iDMRGStep, ...] = ()
        self.energy = float("nan")
        self.energy_density = float("nan")
        self.center_bond_energy = float("nan")
        self.success = False
        self.message = "not run"
        self.algorithm = "idmrg"
        self.metadata: dict[str, object] = {
            "bond_dim": self.bond_dim,
            "physical_dim": self.physical_dim,
            "nterms": self.terms.nterms,
            "solver": self.solver,
            "state_energy_tol": self.state_energy_tol,
        }

    def initial_blocks(self):
        d = self.physical_dim
        zero = np.zeros((d, d), dtype=np.result_type(self.hamiltonian.dtype, np.complex128))
        left = iDMRGBlock(
            length=1,
            hamiltonian=zero.copy(),
            edge_ops=tuple(op.copy() for op in self.terms.left_ops),
        )
        right = iDMRGBlock(
            length=1,
            hamiltonian=zero.copy(),
            edge_ops=tuple(op.copy() for op in self.terms.right_ops),
        )
        return left, right

    def run(self, *, maxiter=None, left_block=None, right_block=None):
        """Run infinite-system growth, populate this solver, and return ``self``."""

        left, right = (
            self.initial_blocks()
            if left_block is None and right_block is None
            else (left_block, right_block)
        )
        if left is None or right is None:
            raise ValueError("left_block and right_block must be provided together.")

        niter = self.maxiter if maxiter is None else int(maxiter)
        history = []
        success = False
        message = "maximum iterations reached"
        state = None
        last_energy = float("nan")
        last_density = float("nan")
        last_growth_density = float("nan")
        last_center_bond = float("nan")

        for iteration in range(1, niter + 1):
            energy, psi = self._ground_state(left, right)
            theta = psi.reshape(left.dim, self.physical_dim, self.physical_dim, right.dim)
            split = self._split_theta(theta)
            center_bond = self._center_bond_energy(theta)
            length = int(left.length + right.length + 2)
            energy_per_site = float(np.real(energy) / length)
            if history:
                prev = history[-1]
                added_sites = int(length - prev.length)
                growth_density = float((np.real(energy) - prev.energy) / added_sites)
            else:
                growth_density = None
            step = iDMRGStep(
                iteration=iteration,
                length=length,
                energy=float(np.real(energy)),
                energy_per_site=energy_per_site,
                growth_energy_per_site=growth_density,
                center_bond_energy=float(np.real(center_bond)),
                kept_dim=int(split["kept_dim"]),
                truncation_error=float(split["truncation_error"]),
                entropy=float(split["entropy"]),
            )
            history.append(step)

            state = self._uniform_state_from_center(split, left.dim, right.dim)
            left = self._advance_left_block(left, split["u"])
            right = self._advance_right_block(right, split["v"])
            last_energy = step.energy
            last_density = step.energy_per_site
            if step.growth_energy_per_site is not None:
                last_growth_density = step.growth_energy_per_site
            last_center_bond = step.center_bond_energy

            conv_ref = None
            if step.growth_energy_per_site is not None:
                if len(history) >= 3:
                    conv_ref = history[-3].growth_energy_per_site
                elif len(history) >= 2:
                    conv_ref = history[-2].growth_energy_per_site
            if conv_ref is not None and abs(step.growth_energy_per_site - conv_ref) < self.tol:
                success = True
                message = "growth energy converged"
                break

        candidate_state = state
        if state is not None:
            try:
                state = state.left_canonical()
            except (ValueError, NotImplementedError):
                pass
            object.__setattr__(state, "energy", float(np.real(state.energy_density(self.hamiltonian))))
            object.__setattr__(state, "success", success)
            object.__setattr__(state, "message", message)
            object.__setattr__(state, "nit", len(history))
            object.__setattr__(state, "nfev", len(history))
            object.__setattr__(state, "history", tuple(step.center_bond_energy for step in history))
            object.__setattr__(state, "gradient_norm", None)
            object.__setattr__(state, "algorithm", "idmrg")
            candidate_state = state
            candidate_state_energy_density = float(np.real(state.energy))
        else:
            candidate_state_energy_density = float("nan")
        result_density = last_growth_density if np.isfinite(last_growth_density) else last_density
        state_export = "none"
        state_energy_density = None
        state_energy_mismatch = None
        if candidate_state is not None and np.isfinite(candidate_state_energy_density) and np.isfinite(result_density):
            state_energy_mismatch = abs(candidate_state_energy_density - result_density)
            if state_energy_mismatch <= self.state_energy_tol:
                state = candidate_state
                state_energy_density = candidate_state_energy_density
                state_export = "raw"
            else:
                state = None
                state_export = "omitted_energy_mismatch"

        self.state = state
        self.left_block = left
        self.right_block = right
        self.history = tuple(history)
        self.energy = last_energy
        self.energy_density = result_density
        self.center_bond_energy = last_center_bond
        self.success = success
        self.message = message
        self.metadata = {
            "bond_dim": self.bond_dim,
            "finite_superblock_energy_per_site": last_density,
            "candidate_uniform_state_energy_per_site": candidate_state_energy_density,
            "uniform_state_energy_per_site": state_energy_density,
            "state_energy_mismatch": state_energy_mismatch,
            "state_export": state_export,
            "state_energy_tol": self.state_energy_tol,
            "physical_dim": self.physical_dim,
            "nterms": self.terms.nterms,
            "solver": self.solver,
        }
        return self

    def _apply_superblock(self, theta, left, right):
        out = np.einsum("ab,bcdr->acdr", left.hamiltonian, theta, optimize=True)
        out += np.einsum("ab,lijb->lija", right.hamiltonian, theta, optimize=True)
        out += np.einsum("abcd,lcdr->labr", self.hamiltonian, theta, optimize=True)
        for coeff, left_edge, left_site, right_site, right_edge in zip(
            self.terms.coeffs,
            left.edge_ops,
            self.terms.left_ops,
            self.terms.right_ops,
            right.edge_ops,
        ):
            out += coeff * np.einsum("ab,ic,bcjr->aijr", left_edge, right_site, theta, optimize=True)
            out += coeff * np.einsum("jc,ab,licb->lija", left_site, right_edge, theta, optimize=True)
        return out

    def _ground_state(self, left, right):
        dim = int(left.dim * self.physical_dim * self.physical_dim * right.dim)
        if self.solver in {"auto", "dense"} and dim <= self.dense_dim_limit:
            matrix = self._dense_superblock_matrix(left, right, dim)
            values, vectors = np.linalg.eigh(matrix)
            return _real_if_close_scalar(values[0]), vectors[:, 0]
        if self.solver == "dense":
            raise ValueError(
                f"dense iDMRG solve dimension {dim} exceeds dense_dim_limit={self.dense_dim_limit}."
            )
        return self._iterative_ground_state(left, right, dim)

    def _dense_superblock_matrix(self, left, right, dim):
        matrix = np.zeros((dim, dim), dtype=np.result_type(self.hamiltonian.dtype, np.complex128))
        shape = (left.dim, self.physical_dim, self.physical_dim, right.dim)
        for col in range(dim):
            basis = np.zeros(shape, dtype=matrix.dtype)
            basis.reshape(-1)[col] = 1.0
            matrix[:, col] = self._apply_superblock(basis, left, right).reshape(-1)
        return 0.5 * (matrix + matrix.conj().T)

    def _iterative_ground_state(self, left, right, dim):
        try:
            from scipy.sparse.linalg import LinearOperator, eigsh
        except ImportError as exc:  # pragma: no cover - SciPy is an optional runtime dependency.
            raise ImportError("iterative iDMRG solve requires scipy.") from exc

        shape = (left.dim, self.physical_dim, self.physical_dim, right.dim)
        dtype = np.result_type(self.hamiltonian.dtype, np.complex128)

        def matvec(vector):
            theta = np.asarray(vector, dtype=dtype).reshape(shape)
            return self._apply_superblock(theta, left, right).reshape(-1)

        op = LinearOperator((dim, dim), matvec=matvec, dtype=dtype)
        values, vectors = eigsh(op, k=1, which="SA")
        return _real_if_close_scalar(values[0]), vectors[:, 0]

    def _split_theta(self, theta):
        matrix = theta.reshape(theta.shape[0] * self.physical_dim, self.physical_dim * theta.shape[3])
        u, values, vh = np.linalg.svd(matrix, full_matrices=False)
        weights = np.real(values * values)
        norm = float(np.sum(weights))
        if norm <= 0.0:
            raise ValueError("superblock ground state has zero norm.")
        weights = weights / norm
        keep = min(self.bond_dim, values.size)
        if self.svd_tol > 0.0:
            threshold = float(self.svd_tol) * max(float(values[0]), 1.0)
            keep = max(1, min(keep, int(np.count_nonzero(values > threshold))))
        discarded = float(np.sum(weights[keep:]))
        positive = weights[weights > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        return {
            "u": u[:, :keep],
            "v": vh.conj().T[:, :keep],
            "s": values[:keep],
            "kept_dim": keep,
            "truncation_error": discarded,
            "entropy": entropy,
        }

    def _center_bond_energy(self, theta):
        norm = np.vdot(theta, theta)
        if abs(norm) <= 0.0:
            raise ValueError("cannot evaluate center bond energy for a zero tensor.")
        h_theta = np.einsum("abcd,lcdr->labr", self.hamiltonian, theta, optimize=True)
        return _real_if_close_scalar(np.vdot(theta, h_theta) / norm)

    def _advance_left_block(self, block, transform):
        d = self.physical_dim
        eye = np.eye(d, dtype=np.result_type(self.hamiltonian.dtype, np.complex128))
        enlarged = np.kron(block.hamiltonian, eye)
        for coeff, edge, site_op in zip(self.terms.coeffs, block.edge_ops, self.terms.right_ops):
            enlarged += coeff * np.kron(edge, site_op)
        new_h = transform.conj().T @ enlarged @ transform
        new_ops = tuple(
            transform.conj().T @ np.kron(np.eye(block.dim, dtype=new_h.dtype), op) @ transform
            for op in self.terms.left_ops
        )
        return iDMRGBlock(
            length=block.length + 1,
            hamiltonian=0.5 * (new_h + new_h.conj().T),
            edge_ops=new_ops,
        )

    def _advance_right_block(self, block, transform):
        d = self.physical_dim
        eye = np.eye(d, dtype=np.result_type(self.hamiltonian.dtype, np.complex128))
        enlarged = np.kron(eye, block.hamiltonian)
        for coeff, site_op, edge in zip(self.terms.coeffs, self.terms.left_ops, block.edge_ops):
            enlarged += coeff * np.kron(site_op, edge)
        new_h = transform.conj().T @ enlarged @ transform
        new_ops = tuple(
            transform.conj().T @ np.kron(op, np.eye(block.dim, dtype=new_h.dtype)) @ transform
            for op in self.terms.right_ops
        )
        return iDMRGBlock(
            length=block.length + 1,
            hamiltonian=0.5 * (new_h + new_h.conj().T),
            edge_ops=new_ops,
        )

    def _uniform_state_from_center(self, split, left_dim, right_dim):
        kept = split["kept_dim"]
        if int(left_dim) == int(kept) and int(right_dim) == int(kept):
            left_tensor = split["u"].reshape(left_dim, self.physical_dim, kept).transpose(1, 0, 2)
            right_tensor = split["v"].reshape(self.physical_dim, right_dim, kept).transpose(0, 2, 1).conj()
            sroot = np.sqrt(np.asarray(split["s"], dtype=left_tensor.dtype))
            left_tensor = left_tensor * sroot[None, None, :]
            right_tensor = right_tensor * sroot[None, :, None]
            return UniformMPS(np.asarray([left_tensor, right_tensor])).normalize_transfer()
        if int(left_dim) != int(kept):
            return None
        tensor = split["u"].reshape(left_dim, self.physical_dim, kept).transpose(1, 0, 2)
        state = UniformMPS(tensor).normalize_transfer()
        return state


InfiniteDMRG = iDMRG


def idmrg_nearest_neighbor(hamiltonian, **kwargs):
    """Convenience wrapper around :class:`iDMRG`."""

    return iDMRG(hamiltonian, **kwargs).run()
