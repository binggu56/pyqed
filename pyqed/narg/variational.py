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
        self.hamiltonian = _as_matrix(hamiltonian)
        expected = int(np.prod(self.dims))
        if self.hamiltonian.shape != (expected, expected):
            raise ValueError(
                f"hamiltonian shape {self.hamiltonian.shape} does not match product dimension {expected}."
            )

        self.overlap = None if overlap is None else _as_matrix(overlap)
        if self.overlap is not None and self.overlap.shape != self.hamiltonian.shape:
            raise ValueError("overlap shape must match hamiltonian shape.")

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
