#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational sweeps for NARG basis optimization.

This module is intentionally small and dense-reference based. It implements
the finite two-site optimization loop needed by NARG while keeping the local
effective Hamiltonian construction explicit. For production-size NARG runs,
the dense projectors here should be replaced by cached left/right
environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.sparse import issparse

try:
    from opt_einsum import contract as _contract
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    _contract = None


def _as_matrix(operator):
    matrix = operator.toarray() if issparse(operator) else np.asarray(operator)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix.")
    return matrix


def _validate_dims(dims):
    dims = tuple(int(d) for d in dims)
    if not dims or any(d < 1 for d in dims):
        raise ValueError("dims must be a non-empty sequence of positive integers.")
    return dims


def _default_bonds(dims, bond_dim):
    bonds = [1]
    left_dim = 1
    total = int(np.prod(dims))
    for d in dims[:-1]:
        left_dim *= d
        right_dim = total // left_dim
        bonds.append(min(int(bond_dim), left_dim, right_dim))
    bonds.append(1)
    return bonds


def _normalize_with_metric(vector, metric):
    norm2 = np.vdot(vector, metric @ vector)
    norm = np.sqrt(float(np.real(norm2)))
    if norm < 1e-14:
        raise ValueError("Cannot normalize a numerically zero state.")
    return vector / norm


def _lowest_generalized_eigenpair(hamiltonian, metric, *, metric_tol=1e-12):
    """
    Solve the lowest generalized eigenpair in the nonsingular metric range.
    """
    metric_vals, metric_vecs = linalg.eigh(metric)
    keep = metric_vals > metric_tol * max(1.0, float(np.max(np.abs(metric_vals))))
    if not np.any(keep):
        raise ValueError("Effective overlap metric is numerically singular.")
    basis = metric_vecs[:, keep] / np.sqrt(metric_vals[keep])[None, :]
    reduced_h = basis.conj().T @ hamiltonian @ basis
    reduced_h = 0.5 * (reduced_h + reduced_h.conj().T)
    evals, evecs = linalg.eigh(reduced_h)
    idx = int(np.argmin(np.real(evals)))
    vector = basis @ evecs[:, idx]
    vector = _normalize_with_metric(vector, metric)
    return float(np.real(evals[idx])), vector


@dataclass
class LETTAResult:
    """
    Result container returned by :meth:`LETTA.run`.
    """

    energy: float
    cores: list
    history: list
    converged: bool
    ncompleted: int


@dataclass
class LegTiedLETTAResult:
    """
    Result container returned by :meth:`LegTiedLETTA.run`.
    """

    energy: float
    tensors: list
    history: list
    converged: bool
    ncompleted: int


class LETTA:
    """
    Local eigensolver tensor-train ansatz for NARG states.

    The state is stored as MPS-like NARG cores with shape
    ``(left, physical, right)``. The physical index may represent a primitive
    coordinate grid, an adiabatic channel, or any retained local NARG basis.

    Parameters
    ----------
    hamiltonian
        Full Hamiltonian in the product basis defined by ``dims``.
    dims
        Local basis dimensions.
    bond_dim
        Maximum number of retained renormalized states on every bond.
    overlap
        Optional full overlap matrix. If provided, local solves use
        ``H_eff c = E S_eff c``. If omitted, the product basis is assumed
        orthonormal, but the local MPS projector metric is still included.
    cores
        Optional initial cores with shape ``(left, physical, right)``.
    seed
        Random seed used when ``cores`` is omitted.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        *,
        bond_dim=32,
        overlap=None,
        cores=None,
        seed=None,
    ):
        self.dims = _validate_dims(dims)
        expected = int(np.prod(self.dims))
        if hamiltonian is None:
            self.hamiltonian = None
        else:
            self.hamiltonian = _as_matrix(hamiltonian)
            if self.hamiltonian.shape != (expected, expected):
                raise ValueError(
                    f"hamiltonian shape {self.hamiltonian.shape} does not match product dimension {expected}."
                )

        self.overlap = None if overlap is None else _as_matrix(overlap)
        if self.overlap is not None and self.overlap.shape != (expected, expected):
            raise ValueError("overlap shape must match product dimension.")

        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")

        self.rng = np.random.default_rng(seed)
        self.cores = self._random_cores() if cores is None else self._validate_cores(cores)
        self.history = []
        self.converged = False
        self.energy = None

    def _random_cores(self):
        bonds = _default_bonds(self.dims, self.bond_dim)
        cores = []
        for n, d in enumerate(self.dims):
            shape = (bonds[n], d, bonds[n + 1])
            core = self.rng.normal(size=shape)
            core = core / np.sqrt(np.prod(shape))
            cores.append(core.astype(float))
        return cores

    def _validate_cores(self, cores):
        cores = [np.asarray(core, dtype=complex if np.iscomplexobj(core) else float) for core in cores]
        if len(cores) != len(self.dims):
            raise ValueError("number of cores must match dims.")
        for n, (core, d) in enumerate(zip(cores, self.dims)):
            if core.ndim != 3 or core.shape[1] != d:
                raise ValueError(f"core {n} must have shape (left, {d}, right).")
            if n == 0 and core.shape[0] != 1:
                raise ValueError("first core must have left bond dimension 1.")
            if n == len(cores) - 1 and core.shape[2] != 1:
                raise ValueError("last core must have right bond dimension 1.")
            if n and cores[n - 1].shape[2] != core.shape[0]:
                raise ValueError(f"bond mismatch between cores {n - 1} and {n}.")
        return cores

    @property
    def nsites(self):
        return len(self.dims)

    def copy(self):
        return LETTA(
            self.hamiltonian.copy(),
            self.dims,
            bond_dim=self.bond_dim,
            overlap=None if self.overlap is None else self.overlap.copy(),
            cores=[core.copy() for core in self.cores],
        )

    def state_vector(self):
        """Return the dense product-basis vector represented by the cores."""
        psi = self.cores[0][0]
        for core in self.cores[1:]:
            psi = np.tensordot(psi, core, axes=([-1], [0]))
        return psi.reshape(-1)

    def norm(self):
        psi = self.state_vector()
        metric = np.eye(psi.size) if self.overlap is None else self.overlap
        return float(np.real(np.vdot(psi, metric @ psi)))

    def expectation(self):
        psi = self.state_vector()
        metric = np.eye(psi.size) if self.overlap is None else self.overlap
        denom = np.vdot(psi, metric @ psi)
        if abs(denom) < 1e-14:
            raise ValueError("State norm is numerically zero.")
        return float(np.real(np.vdot(psi, self.hamiltonian @ psi) / denom))

    def _left_basis(self, stop):
        basis = np.ones((1, 1), dtype=self.cores[0].dtype)
        for core in self.cores[:stop]:
            basis = np.tensordot(basis, core, axes=([1], [0]))
            basis = basis.reshape(-1, core.shape[2])
        return basis

    def _right_basis(self, start):
        dtype = self.cores[-1].dtype
        basis = np.ones((1, 1), dtype=dtype)
        for core in reversed(self.cores[start:]):
            basis = np.tensordot(core, basis, axes=([2], [0]))
            basis = basis.reshape(core.shape[0], -1)
        return basis

    def _bond_projector(self, bond):
        left = self._left_basis(bond)
        right = self._right_basis(bond + 2)
        dl = self.cores[bond].shape[0]
        dr = self.cores[bond + 1].shape[2]
        di = self.dims[bond]
        dj = self.dims[bond + 1]
        eye_i = np.eye(di, dtype=left.dtype)
        eye_j = np.eye(dj, dtype=left.dtype)
        projector = np.einsum("xa,it,ju,by->xijyatub", left, eye_i, eye_j, right)
        return projector.reshape(left.shape[0] * di * dj * right.shape[1], dl * di * dj * dr)

    def _solve_local(self, bond):
        projector = self._bond_projector(bond)
        heff = projector.conj().T @ self.hamiltonian @ projector
        full_metric = np.eye(self.hamiltonian.shape[0]) if self.overlap is None else self.overlap
        seff = projector.conj().T @ full_metric @ projector
        seff = 0.5 * (seff + seff.conj().T)
        heff = 0.5 * (heff + heff.conj().T)

        return _lowest_generalized_eigenpair(heff, seff)

    def _split_local_vector(self, bond, vector, direction):
        left_dim = self.cores[bond].shape[0]
        right_dim = self.cores[bond + 1].shape[2]
        di = self.dims[bond]
        dj = self.dims[bond + 1]
        theta = vector.reshape(left_dim, di, dj, right_dim)
        matrix = theta.reshape(left_dim * di, dj * right_dim)
        u, singular_values, vh = linalg.svd(matrix, full_matrices=False)
        keep = min(self.bond_dim, len(singular_values))
        discarded = singular_values[keep:]
        u = u[:, :keep]
        s = singular_values[:keep]
        vh = vh[:keep]

        if direction == "lr":
            left_core = u.reshape(left_dim, di, keep)
            right_core = (s[:, None] * vh).reshape(keep, dj, right_dim)
        else:
            left_core = (u * s[None, :]).reshape(left_dim, di, keep)
            right_core = vh.reshape(keep, dj, right_dim)

        trunc_err = float(np.sum(discarded**2))
        self.cores[bond] = left_core
        self.cores[bond + 1] = right_core
        return keep, trunc_err

    def sweep(self, direction="lr"):
        """
        Perform one two-site sweep and return per-bond diagnostics.
        """
        if self.nsites < 2:
            raise ValueError("At least two sites are required for two-site sweeps.")
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        bonds = range(self.nsites - 1)
        if direction == "rl":
            bonds = reversed(list(bonds))

        updates = []
        for bond in bonds:
            local_energy, vector = self._solve_local(bond)
            kept, trunc_err = self._split_local_vector(bond, vector, direction)
            updates.append(
                {
                    "bond": int(bond),
                    "local_energy": float(local_energy),
                    "kept": int(kept),
                    "trunc_err": trunc_err,
                }
            )
        return updates

    def run(self, *, nsweeps=4, start_direction="lr", alternate=True, tol=1e-10, verbose=0):
        """
        Run finite two-site sweeps.
        """
        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
        direction = start_direction.lower()
        previous_energy = None
        self.history = []
        self.converged = False

        for sweep_idx in range(int(nsweeps)):
            updates = self.sweep(direction)
            energy = self.expectation()
            delta = None if previous_energy is None else abs(energy - previous_energy)
            entry = {
                "sweep": sweep_idx,
                "direction": direction,
                "energy": energy,
                "delta_energy": delta,
                "updates": updates,
            }
            self.history.append(entry)
            if int(verbose) > 0:
                print(
                    f"sweep {sweep_idx:>2} | dir={direction} | "
                    f"E={energy:.12g} | dE={'-' if delta is None else f'{delta:.3e}'}"
                )
            if delta is not None and delta <= tol:
                self.converged = True
                break
            previous_energy = energy
            if alternate:
                direction = "rl" if direction == "lr" else "lr"

        self.energy = self.history[-1]["energy"]
        return LETTAResult(
            energy=self.energy,
            cores=[core.copy() for core in self.cores],
            history=list(self.history),
            converged=self.converged,
            ncompleted=len(self.history),
        )


class LegTiedLETTA:
    r"""
    Dense-reference nearest-neighbor leg-tied tensor ansatz.

    The represented wavefunction is

    .. math::

        \Psi(\sigma_0,\ldots,\sigma_{L-1}) =
        \sum_{\alpha_0\ldots\alpha_{L-3}}
        \prod_{i=0}^{L-2}
        A^{[i]}_{\alpha_{i-1},\sigma_i,\sigma_{i+1},\alpha_i},

    with boundary bond dimensions ``alpha[-1] = alpha[L-2] = 1``.  The
    physical index ``sigma_i`` is therefore shared by neighboring tensors,
    unlike in an MPS where each physical leg appears in exactly one tensor.

    This class is a small dense prototype for one-site LETTA optimization.  It
    is intended for validating the variational equations and for seeding from a
    NARG/MPS state before replacing dense projectors by cached environments.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        *,
        bond_dim=4,
        overlap=None,
        tensors=None,
        seed=None,
    ):
        self.dims = _validate_dims(dims)
        if len(self.dims) < 2:
            raise ValueError("LegTiedLETTA needs at least two physical sites.")
        expected = int(np.prod(self.dims))
        if hamiltonian is None:
            self.hamiltonian = None
        else:
            self.hamiltonian = _as_matrix(hamiltonian)
            if self.hamiltonian.shape != (expected, expected):
                raise ValueError(
                    f"hamiltonian shape {self.hamiltonian.shape} does not match product dimension {expected}."
                )

        self.overlap = None if overlap is None else _as_matrix(overlap)
        if self.overlap is not None and self.overlap.shape != (expected, expected):
            raise ValueError("overlap shape must match product dimension.")

        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")

        self.rng = np.random.default_rng(seed)
        self.tensors = self._random_tensors() if tensors is None else self._validate_tensors(tensors)
        self.history = []
        self.converged = False
        self.energy = None
        self.normalize()

    @classmethod
    def from_state_vector(
        cls,
        hamiltonian,
        dims,
        state,
        *,
        bond_dim=4,
        overlap=None,
        seed=None,
        fit_sweeps=4,
        ridge=1e-12,
    ):
        """
        Initialize a leg-tied LETTA by least-squares fitting a dense state.

        This is the practical bridge from a NARG/MPS guess: compute the NARG
        state vector, then fit the tied-leg tensors by alternating one-site
        linear least squares.
        """
        obj = cls(hamiltonian, dims, bond_dim=bond_dim, overlap=overlap, seed=seed)
        obj.fit_state(state, nsweeps=fit_sweeps, ridge=ridge)
        return obj

    @classmethod
    def from_narg(
        cls,
        hamiltonian,
        narg_state,
        *,
        dims=None,
        bond_dim=None,
        overlap=None,
        seed=None,
        fit_sweeps=4,
        ridge=1e-12,
    ):
        """
        Initialize from an object exposing ``state_vector()`` and optionally
        ``dims``/``bond_dim`` attributes, such as the dense NARG prototype in
        this module.
        """
        if not hasattr(narg_state, "state_vector"):
            raise TypeError("narg_state must expose a state_vector() method.")
        dims = tuple(dims if dims is not None else getattr(narg_state, "dims"))
        if bond_dim is None:
            bond_dim = getattr(narg_state, "bond_dim", 4)
        return cls.from_state_vector(
            hamiltonian,
            dims,
            narg_state.state_vector(),
            bond_dim=bond_dim,
            overlap=overlap,
            seed=seed,
            fit_sweeps=fit_sweeps,
            ridge=ridge,
        )

    @property
    def nsites(self):
        return len(self.dims)

    @property
    def nbonds(self):
        return len(self.dims) - 1

    def _default_letta_bonds(self):
        return [1] + [self.bond_dim] * max(0, self.nsites - 2) + [1]

    def _random_tensors(self):
        bonds = self._default_letta_bonds()
        tensors = []
        for i in range(self.nbonds):
            shape = (bonds[i], self.dims[i], self.dims[i + 1], bonds[i + 1])
            tensor = self.rng.normal(size=shape)
            tensor = tensor / np.sqrt(np.prod(shape))
            tensors.append(tensor.astype(float))
        return tensors

    def _validate_tensors(self, tensors):
        tensors = [np.asarray(tensor, dtype=complex if np.iscomplexobj(tensor) else float) for tensor in tensors]
        if len(tensors) != self.nbonds:
            raise ValueError("number of LETTA tensors must be len(dims)-1.")
        for i, tensor in enumerate(tensors):
            if tensor.ndim != 4 or tensor.shape[1:3] != self.dims[i:i + 2]:
                raise ValueError(f"tensor {i} must have shape (left, {self.dims[i]}, {self.dims[i + 1]}, right).")
            if i == 0 and tensor.shape[0] != 1:
                raise ValueError("first LETTA tensor must have left bond dimension 1.")
            if i == self.nbonds - 1 and tensor.shape[3] != 1:
                raise ValueError("last LETTA tensor must have right bond dimension 1.")
            if i and tensors[i - 1].shape[3] != tensor.shape[0]:
                raise ValueError(f"bond mismatch between LETTA tensors {i - 1} and {i}.")
        return tensors

    def copy(self):
        return LegTiedLETTA(
            None if self.hamiltonian is None else self.hamiltonian.copy(),
            self.dims,
            bond_dim=self.bond_dim,
            overlap=None if self.overlap is None else self.overlap.copy(),
            tensors=[tensor.copy() for tensor in self.tensors],
        )

    def _amplitude(self, config):
        vec = self.tensors[0][0, config[0], config[1], :]
        for i in range(1, self.nbonds):
            vec = vec @ self.tensors[i][:, config[i], config[i + 1], :]
        return vec[0]

    def state_vector(self):
        """Return the dense product-basis vector represented by tied tensors."""
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        psi = np.empty(int(np.prod(self.dims)), dtype=dtype)
        for flat, config in enumerate(np.ndindex(*self.dims)):
            psi[flat] = self._amplitude(config)
        return psi

    def norm(self):
        psi = self.state_vector()
        metric = np.eye(psi.size) if self.overlap is None else self.overlap
        return float(np.real(np.vdot(psi, metric @ psi)))

    def normalize(self):
        norm = np.sqrt(self.norm())
        if norm < 1e-14:
            raise ValueError("Cannot normalize a numerically zero LETTA state.")
        # Rescale a single tensor; this preserves the tied-leg structure.
        self.tensors[0] = self.tensors[0] / norm
        return self

    def expectation(self):
        if self.hamiltonian is None:
            raise ValueError("dense hamiltonian is not available; use expectation_mpo(mpo).")
        psi = self.state_vector()
        metric = np.eye(psi.size) if self.overlap is None else self.overlap
        denom = np.vdot(psi, metric @ psi)
        if abs(denom) < 1e-14:
            raise ValueError("State norm is numerically zero.")
        return float(np.real(np.vdot(psi, self.hamiltonian @ psi) / denom))

    def _validate_mpo(self, mpo):
        mpo = [np.asarray(site) for site in mpo]
        if len(mpo) != self.nsites:
            raise ValueError("MPO length must match the number of physical sites.")
        for i, site in enumerate(mpo):
            if site.ndim != 4:
                raise ValueError("each MPO tensor must have shape (left, right, bra, ket).")
            if site.shape[2] != self.dims[i] or site.shape[3] != self.dims[i]:
                raise ValueError(f"MPO tensor {i} physical dimensions do not match dims[{i}].")
            if i == 0 and site.shape[0] != 1:
                raise ValueError("first MPO tensor must have left bond dimension 1.")
            if i == self.nsites - 1 and site.shape[1] != 1:
                raise ValueError("last MPO tensor must have right bond dimension 1.")
            if i and mpo[i - 1].shape[1] != site.shape[0]:
                raise ValueError(f"MPO bond mismatch between tensors {i - 1} and {i}.")
        return mpo

    def identity_mpo(self):
        """
        Return the product-basis identity as an MPO.
        """
        return [np.eye(dim, dtype=self.tensors[0].dtype).reshape(1, 1, dim, dim) for dim in self.dims]

    def apply_mpo(self, mpo, vector):
        """
        Apply an MPO to a dense product-basis vector. This is diagnostic; the
        MPO optimizer below does not form dense local projectors.
        """
        mpo = self._validate_mpo(mpo)
        tmp = np.asarray(vector).reshape(self.dims)[None, ...]
        for site, operator in enumerate(mpo):
            nout = site
            rem_after = self.nsites - site - 1
            tmp = np.tensordot(tmp, operator, axes=([0, nout + 1], [0, 3]))
            right_axis = nout + rem_after
            current_output_axis = right_axis + 1
            order = [right_axis] + list(range(nout)) + [current_output_axis] + list(range(nout, nout + rem_after))
            tmp = np.transpose(tmp, order)
        return tmp[0].reshape(-1)

    def expectation_mpo(self, mpo):
        """
        Expectation value with an MPO contracted directly against the LETTA
        double layer.
        """
        value = self._normalized_mpo_expectation(mpo)
        return float(np.real(value))

    def _mpo_matrix_element(self, mpo):
        """
        Contract ``<Psi|MPO|Psi>`` without forming dense state vectors.
        """
        mpo = self._validate_mpo(mpo)
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        env = np.ones((1, 1, mpo[0].shape[0], self.dims[0], self.dims[0]), dtype=dtype)
        for i, tensor in enumerate(self.tensors):
            env = self._advance_left_environment(env, mpo[i], tensor)
        return np.einsum("bkmxy,mnxy->", env, mpo[-1], optimize=True)

    def _mpo_matrix_element_direct(self, mpo):
        """
        Reference full-network contraction for ``_mpo_matrix_element``.
        """
        if _contract is None:
            raise ImportError("opt_einsum is required for direct LETTA contractions.")
        mpo = self._validate_mpo(mpo)
        nbonds = self.nbonds
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        next_label = 0

        def labels(count):
            nonlocal next_label
            out = list(range(next_label, next_label + count))
            next_label += count
            return out

        ket_phys = labels(self.nsites)
        bra_phys = labels(self.nsites)
        ket_bonds = labels(nbonds + 1)
        bra_bonds = labels(nbonds + 1)
        mpo_bonds = labels(self.nsites + 1)

        operands = []
        for site, operator in enumerate(mpo):
            operands.extend(
                [operator, [mpo_bonds[site], mpo_bonds[site + 1], bra_phys[site], ket_phys[site]]]
            )
        operands.extend([np.ones(self.tensors[0].shape[0], dtype=dtype), [ket_bonds[0]]])
        operands.extend([np.ones(self.tensors[0].shape[0], dtype=dtype), [bra_bonds[0]]])
        operands.extend([np.ones(self.tensors[-1].shape[3], dtype=dtype), [ket_bonds[-1]]])
        operands.extend([np.ones(self.tensors[-1].shape[3], dtype=dtype), [bra_bonds[-1]]])
        for i, tensor in enumerate(self.tensors):
            operands.extend([tensor, [ket_bonds[i], ket_phys[i], ket_phys[i + 1], ket_bonds[i + 1]]])
            operands.extend([tensor.conj(), [bra_bonds[i], bra_phys[i], bra_phys[i + 1], bra_bonds[i + 1]]])
        return _contract(*operands, [], optimize="auto")

    def _normalized_mpo_expectation(self, mpo):
        value = self._mpo_matrix_element(mpo)
        denom = self._mpo_matrix_element(self.identity_mpo())
        if abs(denom) < 1e-14:
            raise ValueError("State norm is numerically zero.")
        return value / denom

    def product_operator_mpo(self, operators):
        """
        Build a bond-1 MPO from one local operator per site.

        Each local operator uses the ``(bra, ket)`` convention.
        """
        if len(operators) != self.nsites:
            raise ValueError("number of local operators must match the number of physical sites.")
        mpo = []
        for i, operator in enumerate(operators):
            operator = np.asarray(operator)
            if operator.shape != (self.dims[i], self.dims[i]):
                raise ValueError(f"operator {i} must have shape ({self.dims[i]}, {self.dims[i]}).")
            mpo.append(operator.reshape(1, 1, self.dims[i], self.dims[i]))
        return mpo

    def expectation_product_operator(self, operators):
        """
        Expectation value of a product of local operators.
        """
        return self._normalized_mpo_expectation(self.product_operator_mpo(operators))

    def spatial_correlation(self, op_a, op_b=None, *, connected=False, average=False):
        """
        Compute ``<op_a(i) op_b(j)>`` or its connected correlation matrix.

        Parameters
        ----------
        op_a, op_b
            Local operators. If ``op_b`` is omitted, ``op_a`` is used for both
            sites. On-site entries use the ordered product ``op_a @ op_b``.
        connected
            If true, subtract ``<op_a(i)> <op_b(j)>``.
        average
            If true, return the distance-averaged correlation ``C(r)`` instead
            of the full ``C(i,j)`` matrix.
        """
        if len(set(self.dims)) != 1:
            raise ValueError("spatial_correlation currently requires equal local dimensions.")
        dim = self.dims[0]
        op_a = np.asarray(op_a)
        op_b = op_a if op_b is None else np.asarray(op_b)
        if op_a.shape != (dim, dim) or op_b.shape != (dim, dim):
            raise ValueError(f"local operators must have shape ({dim}, {dim}).")

        eye = np.eye(dim, dtype=np.result_type(op_a.dtype, op_b.dtype, self.tensors[0].dtype))
        one_a = np.empty(self.nsites, dtype=complex)
        one_b = np.empty(self.nsites, dtype=complex)
        corr = np.empty((self.nsites, self.nsites), dtype=complex)

        for i in range(self.nsites):
            ops = [eye] * self.nsites
            ops[i] = op_a
            one_a[i] = self.expectation_product_operator(ops)
            ops = [eye] * self.nsites
            ops[i] = op_b
            one_b[i] = self.expectation_product_operator(ops)

        for i in range(self.nsites):
            for j in range(self.nsites):
                ops = [eye] * self.nsites
                if i == j:
                    ops[i] = op_a @ op_b
                else:
                    ops[i] = op_a
                    ops[j] = op_b
                corr[i, j] = self.expectation_product_operator(ops)

        if connected:
            corr = corr - np.outer(one_a, one_b)
        if average:
            return np.array([np.mean([corr[i, i + r] for i in range(self.nsites - r)]) for r in range(self.nsites)])
        return corr

    def local_effective_matrix(self, mpo, tensor_index):
        """
        Contract ``<dPsi/dA_i|MPO|dPsi/dA_i>`` without forming a dense
        product-basis projector.

        The output matrix is ordered consistently with
        ``self.tensors[tensor_index].reshape(-1)``.
        """
        mpo = self._validate_mpo(mpo)
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nbonds:
            raise IndexError("tensor_index out of range.")
        left_envs = self._left_local_environments(mpo)
        right_envs = self._right_local_environments(mpo)
        return self._local_effective_from_environments(mpo, tensor_index, left_envs, right_envs)

    def local_effective_matrix_direct(self, mpo, tensor_index):
        """
        Reference full-network contraction for ``local_effective_matrix``.
        """
        if _contract is None:
            raise ImportError("opt_einsum is required for direct LETTA contractions.")
        mpo = self._validate_mpo(mpo)
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nbonds:
            raise IndexError("tensor_index out of range.")

        shape = self.tensors[tensor_index].shape
        nbonds = self.nbonds
        next_label = 0

        def labels(count):
            nonlocal next_label
            out = list(range(next_label, next_label + count))
            next_label += count
            return out

        ket_phys = labels(self.nsites)
        bra_phys = labels(self.nsites)
        ket_bonds = labels(nbonds + 1)
        bra_bonds = labels(nbonds + 1)
        mpo_bonds = labels(self.nsites + 1)

        operands = []
        for site, operator in enumerate(mpo):
            operands.extend(
                [operator, [mpo_bonds[site], mpo_bonds[site + 1], bra_phys[site], ket_phys[site]]]
            )

        operands.extend([np.ones(self.tensors[0].shape[0], dtype=shape and self.tensors[0].dtype), [ket_bonds[0]]])
        operands.extend([np.ones(self.tensors[0].shape[0], dtype=shape and self.tensors[0].dtype), [bra_bonds[0]]])
        operands.extend([np.ones(self.tensors[-1].shape[3], dtype=self.tensors[-1].dtype), [ket_bonds[-1]]])
        operands.extend([np.ones(self.tensors[-1].shape[3], dtype=self.tensors[-1].dtype), [bra_bonds[-1]]])

        for i, tensor in enumerate(self.tensors):
            if i == tensor_index:
                continue
            operands.extend([tensor, [ket_bonds[i], ket_phys[i], ket_phys[i + 1], ket_bonds[i + 1]]])
            operands.extend([tensor.conj(), [bra_bonds[i], bra_phys[i], bra_phys[i + 1], bra_bonds[i + 1]]])

        output = [
            bra_bonds[tensor_index],
            bra_phys[tensor_index],
            bra_phys[tensor_index + 1],
            bra_bonds[tensor_index + 1],
            ket_bonds[tensor_index],
            ket_phys[tensor_index],
            ket_phys[tensor_index + 1],
            ket_bonds[tensor_index + 1],
        ]
        heff = _contract(*operands, output, optimize="auto")
        dim = int(np.prod(shape))
        return heff.reshape(dim, dim)

    def _left_local_environments(self, mpo):
        """
        Prefix contractions for LETTA one-site MPO environments.

        ``left[k]`` leaves ``(bra_alpha_k, ket_alpha_k, mpo_w_k,
        bra_sigma_k, ket_sigma_k)`` open for active tensor ``k``.
        """
        mpo = self._validate_mpo(mpo)
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors], *[site.dtype for site in mpo])
        left = []
        env = np.ones((1, 1, mpo[0].shape[0], self.dims[0], self.dims[0]), dtype=dtype)
        left.append(env)
        for i, tensor in enumerate(self.tensors[:-1]):
            env = np.einsum(
                "bkmxy,mnxy,bxuc,kyvd->cdnuv",
                env,
                mpo[i],
                tensor.conj(),
                tensor,
                optimize=True,
            )
            left.append(env)
        return left

    def _advance_left_environment(self, env, mpo_site, tensor):
        return np.einsum(
            "bkmxy,mnxy,bxuc,kyvd->cdnuv",
            env,
            mpo_site,
            tensor.conj(),
            tensor,
            optimize=True,
        )

    def _right_local_environments(self, mpo):
        """
        Suffix contractions for LETTA one-site MPO environments.

        ``right[k]`` leaves ``(bra_alpha_{k+1}, ket_alpha_{k+1},
        mpo_w_{k+2}, bra_sigma_{k+1}, ket_sigma_{k+1})`` open for active
        tensor ``k``.
        """
        mpo = self._validate_mpo(mpo)
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors], *[site.dtype for site in mpo])
        right = [None] * self.nbonds
        env = np.ones((1, 1, mpo[-1].shape[1], self.dims[-1], self.dims[-1]), dtype=dtype)
        right[-1] = env
        for i in range(self.nbonds - 1, 0, -1):
            tensor = self.tensors[i]
            env = np.einsum(
                "cdnuv,mnuv,bxuc,kyvd->bkmxy",
                env,
                mpo[i + 1],
                tensor.conj(),
                tensor,
                optimize=True,
            )
            right[i - 1] = env
        return right

    def _advance_right_environment(self, env, mpo_site, tensor):
        return np.einsum(
            "cdnuv,mnuv,bxuc,kyvd->bkmxy",
            env,
            mpo_site,
            tensor.conj(),
            tensor,
            optimize=True,
        )

    def _local_effective_from_environments(self, mpo, tensor_index, left_envs, right_envs):
        tensor_index = int(tensor_index)
        shape = self.tensors[tensor_index].shape
        heff = np.einsum(
            "bkmxy,mpxy,pnuv,cdnuv->bxuckyvd",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            right_envs[tensor_index],
            optimize=True,
        )
        return heff.reshape(int(np.prod(shape)), int(np.prod(shape)))

    def _solve_one_site_mpo(self, mpo, tensor_index):
        heff = self.local_effective_matrix(mpo, tensor_index)
        seff = self.local_effective_matrix(self.identity_mpo(), tensor_index)
        heff = 0.5 * (heff + heff.conj().T)
        seff = 0.5 * (seff + seff.conj().T)
        return _lowest_generalized_eigenpair(heff, seff)

    def _solve_one_site_mpo_with_environments(self, mpo, tensor_index, left_envs, right_envs, metric_left, metric_right):
        heff = self._local_effective_from_environments(mpo, tensor_index, left_envs, right_envs)
        identity = self.identity_mpo()
        seff = self._local_effective_from_environments(identity, tensor_index, metric_left, metric_right)
        heff = 0.5 * (heff + heff.conj().T)
        seff = 0.5 * (seff + seff.conj().T)
        return _lowest_generalized_eigenpair(heff, seff)

    def _partial_amplitude(self, tensor_index, config, left, right):
        if tensor_index == 0:
            left_coeff = 1.0
        else:
            vec = self.tensors[0][0, config[0], config[1], :]
            for i in range(1, tensor_index):
                vec = vec @ self.tensors[i][:, config[i], config[i + 1], :]
            left_coeff = vec[left]

        if tensor_index == self.nbonds - 1:
            right_coeff = 1.0
        else:
            last = self.nbonds - 1
            rvec = self.tensors[last][:, config[last], config[last + 1], 0]
            for i in range(last - 1, tensor_index, -1):
                rvec = self.tensors[i][:, config[i], config[i + 1], :] @ rvec
            right_coeff = rvec[right]

        return left_coeff * right_coeff

    def _one_site_projector(self, tensor_index):
        tensor = self.tensors[tensor_index]
        left_dim, di, dj, right_dim = tensor.shape
        nrow = int(np.prod(self.dims))
        ncol = left_dim * di * dj * right_dim
        projector = np.zeros((nrow, ncol), dtype=np.result_type(*[t.dtype for t in self.tensors], complex))

        for flat, config in enumerate(np.ndindex(*self.dims)):
            si = config[tensor_index]
            sj = config[tensor_index + 1]
            for left in range(left_dim):
                for right in range(right_dim):
                    col = (((left * di + si) * dj + sj) * right_dim + right)
                    projector[flat, col] = self._partial_amplitude(tensor_index, config, left, right)
        return projector

    def _solve_one_site(self, tensor_index):
        if self.hamiltonian is None:
            raise ValueError("dense hamiltonian is not available; use optimize_tensor_mpo(mpo, tensor_index).")
        projector = self._one_site_projector(tensor_index)
        heff = projector.conj().T @ self.hamiltonian @ projector
        full_metric = np.eye(self.hamiltonian.shape[0]) if self.overlap is None else self.overlap
        seff = projector.conj().T @ full_metric @ projector
        heff = 0.5 * (heff + heff.conj().T)
        seff = 0.5 * (seff + seff.conj().T)
        return _lowest_generalized_eigenpair(heff, seff)

    def optimize_tensor(self, tensor_index):
        """
        Optimize one tied tensor with all other tensors fixed.
        """
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nbonds:
            raise IndexError("tensor_index out of range.")
        local_energy, vector = self._solve_one_site(tensor_index)
        self.tensors[tensor_index] = vector.reshape(self.tensors[tensor_index].shape)
        self.normalize()
        return {"tensor": tensor_index, "local_energy": float(local_energy)}

    def optimize_tensor_mpo(self, mpo, tensor_index):
        """
        Optimize one tied tensor using an MPO-contracted local Hamiltonian.
        """
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nbonds:
            raise IndexError("tensor_index out of range.")
        local_energy, vector = self._solve_one_site_mpo(mpo, tensor_index)
        self.tensors[tensor_index] = vector.reshape(self.tensors[tensor_index].shape)
        return {"tensor": tensor_index, "local_energy": float(local_energy)}

    def _optimize_tensor_mpo_with_environments(
        self,
        mpo,
        tensor_index,
        left_envs,
        right_envs,
        metric_left,
        metric_right,
    ):
        local_energy, vector = self._solve_one_site_mpo_with_environments(
            mpo,
            tensor_index,
            left_envs,
            right_envs,
            metric_left,
            metric_right,
        )
        self.tensors[tensor_index] = vector.reshape(self.tensors[tensor_index].shape)
        return {"tensor": int(tensor_index), "local_energy": float(local_energy)}

    def sweep(self, direction="lr"):
        """
        Perform one one-site variational sweep over tied tensors.
        """
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        indices = range(self.nbonds)
        if direction == "rl":
            indices = reversed(list(indices))
        return [self.optimize_tensor(i) for i in indices]

    def sweep_mpo(self, mpo, direction="lr"):
        """
        Perform one one-site sweep using MPO-contracted local Hamiltonians.
        """
        mpo = self._validate_mpo(mpo)
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        identity = self.identity_mpo()
        updates = []

        if direction == "lr":
            right_envs = self._right_local_environments(mpo)
            metric_right = self._right_local_environments(identity)
            left_envs = [None] * self.nbonds
            metric_left = [None] * self.nbonds
            left_envs[0] = np.ones((1, 1, mpo[0].shape[0], self.dims[0], self.dims[0]), dtype=right_envs[0].dtype)
            metric_left[0] = np.ones(
                (1, 1, identity[0].shape[0], self.dims[0], self.dims[0]),
                dtype=metric_right[0].dtype,
            )
            for i in range(self.nbonds):
                updates.append(
                    self._optimize_tensor_mpo_with_environments(
                        mpo, i, left_envs, right_envs, metric_left, metric_right
                    )
                )
                if i + 1 < self.nbonds:
                    left_envs[i + 1] = self._advance_left_environment(left_envs[i], mpo[i], self.tensors[i])
                    metric_left[i + 1] = self._advance_left_environment(
                        metric_left[i], identity[i], self.tensors[i]
                    )
        else:
            left_envs = self._left_local_environments(mpo)
            metric_left = self._left_local_environments(identity)
            right_envs = [None] * self.nbonds
            metric_right = [None] * self.nbonds
            right_envs[-1] = np.ones(
                (1, 1, mpo[-1].shape[1], self.dims[-1], self.dims[-1]),
                dtype=left_envs[-1].dtype,
            )
            metric_right[-1] = np.ones(
                (1, 1, identity[-1].shape[1], self.dims[-1], self.dims[-1]),
                dtype=metric_left[-1].dtype,
            )
            for i in reversed(range(self.nbonds)):
                updates.append(
                    self._optimize_tensor_mpo_with_environments(
                        mpo, i, left_envs, right_envs, metric_left, metric_right
                    )
                )
                if i:
                    right_envs[i - 1] = self._advance_right_environment(
                        right_envs[i], mpo[i + 1], self.tensors[i]
                    )
                    metric_right[i - 1] = self._advance_right_environment(
                        metric_right[i], identity[i + 1], self.tensors[i]
                    )

        return updates

    def run(self, *, nsweeps=4, start_direction="lr", alternate=True, tol=1e-10, verbose=0):
        """
        Run one-site LETTA variational sweeps.
        """
        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
        direction = start_direction.lower()
        previous_energy = None
        self.history = []
        self.converged = False

        for sweep_idx in range(int(nsweeps)):
            updates = self.sweep(direction)
            energy = self.expectation()
            delta = None if previous_energy is None else abs(energy - previous_energy)
            entry = {
                "sweep": sweep_idx,
                "direction": direction,
                "energy": energy,
                "delta_energy": delta,
                "updates": updates,
            }
            self.history.append(entry)
            if int(verbose) > 0:
                print(
                    f"letta sweep {sweep_idx:>2} | dir={direction} | "
                    f"E={energy:.12g} | dE={'-' if delta is None else f'{delta:.3e}'}"
                )
            if delta is not None and delta <= tol:
                self.converged = True
                break
            previous_energy = energy
            if alternate:
                direction = "rl" if direction == "lr" else "lr"

        self.energy = self.history[-1]["energy"]
        return LegTiedLETTAResult(
            energy=self.energy,
            tensors=[tensor.copy() for tensor in self.tensors],
            history=list(self.history),
            converged=self.converged,
            ncompleted=len(self.history),
        )

    def run_mpo(self, mpo, *, nsweeps=4, start_direction="lr", alternate=True, tol=1e-10, verbose=0):
        """
        Run one-site LETTA sweeps using MPO-contracted local Hamiltonians.
        """
        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
        mpo = self._validate_mpo(mpo)
        direction = start_direction.lower()
        previous_energy = None
        self.history = []
        self.converged = False

        for sweep_idx in range(int(nsweeps)):
            updates = self.sweep_mpo(mpo, direction)
            energy = self.expectation_mpo(mpo)
            delta = None if previous_energy is None else abs(energy - previous_energy)
            entry = {
                "sweep": sweep_idx,
                "direction": direction,
                "energy": energy,
                "delta_energy": delta,
                "updates": updates,
            }
            self.history.append(entry)
            if int(verbose) > 0:
                print(
                    f"letta-mpo sweep {sweep_idx:>2} | dir={direction} | "
                    f"E={energy:.12g} | dE={'-' if delta is None else f'{delta:.3e}'}"
                )
            if delta is not None and delta <= tol:
                self.converged = True
                break
            previous_energy = energy
            if alternate:
                direction = "rl" if direction == "lr" else "lr"

        self.energy = self.history[-1]["energy"]
        return LegTiedLETTAResult(
            energy=self.energy,
            tensors=[tensor.copy() for tensor in self.tensors],
            history=list(self.history),
            converged=self.converged,
            ncompleted=len(self.history),
        )

    def fit_state(self, state, *, nsweeps=4, ridge=1e-12):
        """
        Alternating least-squares fit to a dense target state.
        """
        target = np.asarray(state).reshape(-1)
        if target.size != int(np.prod(self.dims)):
            raise ValueError("target state size does not match product dimension.")
        for _ in range(int(nsweeps)):
            for direction in ("lr", "rl"):
                indices = range(self.nbonds)
                if direction == "rl":
                    indices = reversed(list(indices))
                for i in indices:
                    projector = self._one_site_projector(i)
                    normal = projector.conj().T @ projector
                    rhs = projector.conj().T @ target
                    if ridge:
                        normal = normal + float(ridge) * np.eye(normal.shape[0], dtype=normal.dtype)
                    try:
                        vector = linalg.solve(normal, rhs, assume_a="pos")
                    except Exception:
                        vector = linalg.lstsq(projector, target)[0]
                    self.tensors[i] = vector.reshape(self.tensors[i].shape)
            self.normalize()
        return self
