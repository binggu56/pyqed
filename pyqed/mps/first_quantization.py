#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic 1D first-quantized finite-dimensional lattice model helpers.

This module provides a reusable class for constructing 1D Hamiltonians in
sum-of-product form, then converting to MPO via AutoMPO.
"""

from __future__ import annotations

import numpy as np

from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSet
from pyqed.mps.autompo.light_automatic_mpo import Mpo as AutoMPO
from pyqed.mps.autompo.model import Model
from pyqed.mps.mps import MPS, MPO, _mpo_to_dense_operator


class FiniteDimLocalBasis(BasisSet):
    """
    Finite-dimensional local basis with built-ins plus custom operator matrices.

    Built-in operator symbols:
    - ``I``: identity
    - ``Pk``: projector |k><k|
    - ``Ei_j``: transition |i><j|
    """

    def __init__(self, dof, dim, operator_mats=None):
        super().__init__(dof, dim, [0] * dim)
        self._operator_mats = {}
        if operator_mats is not None:
            for name, mat in operator_mats.items():
                arr = np.asarray(mat)
                if arr.shape != (dim, dim):
                    raise ValueError(
                        f"Operator '{name}' shape mismatch: {arr.shape} != {(dim, dim)}"
                    )
                self._operator_mats[name] = arr.copy()

    def op_mat(self, op):
        if not isinstance(op, Op):
            op = Op(op, None)
        symbol = op.symbol

        if symbol in self._operator_mats:
            mat = self._operator_mats[symbol]
        elif symbol == "I":
            mat = np.eye(self.nbas, dtype=float)
        elif symbol.startswith("P"):
            idx = int(symbol[1:])
            if not (0 <= idx < self.nbas):
                raise ValueError(f"Projector index out of range: {symbol}")
            mat = np.zeros((self.nbas, self.nbas), dtype=float)
            mat[idx, idx] = 1.0
        elif symbol.startswith("E"):
            body = symbol[1:]
            parts = body.split("_")
            if len(parts) != 2:
                raise ValueError(f"Unsupported transition operator: {symbol}")
            i = int(parts[0])
            j = int(parts[1])
            if not (0 <= i < self.nbas and 0 <= j < self.nbas):
                raise ValueError(f"Transition index out of range: {symbol}")
            mat = np.zeros((self.nbas, self.nbas), dtype=float)
            mat[i, j] = 1.0
        else:
            raise ValueError(f"Unsupported operator symbol: {symbol}")
        factor_dtype = np.asarray(op.factor).dtype
        out_dtype = np.result_type(mat.dtype, factor_dtype)
        return np.asarray(mat, dtype=out_dtype) * op.factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.nbas, operator_mats=self._operator_mats)


class Chain:
    """
    Generic 1D finite-lattice model builder.

    Parameters
    ----------
    nsites : int
        Number of 1D sites.
    local_dim : int
        Local Hilbert-space dimension for each site.
    local_operator_mats : dict[str, np.ndarray], optional
        Optional custom local operators with shape (local_dim, local_dim).
    """

    def __init__(self, nsites, local_dim, local_operator_mats=None):
        if nsites <= 0:
            raise ValueError("nsites must be positive.")
        if local_dim <= 0:
            raise ValueError("local_dim must be positive.")

        self.nsites = int(nsites)
        self.local_dim = int(local_dim)
        self.local_operator_mats = local_operator_mats or {}

        self.basis = [
            FiniteDimLocalBasis(i, self.local_dim, operator_mats=self.local_operator_mats)
            for i in range(self.nsites)
        ]
        self.terms = []

    def clear_terms(self):
        self.terms = []
        return self

    def add_term(self, coeff, op_site_pairs):
        """
        Add a sum-of-product term to the Hamiltonian.

        Examples
        --------
        - Onsite: ``add_term(1.2, [("Q", 3)])``
        - Two-site: ``add_term(-t, [("E2_1", i), ("E0_1", i+1)])``
        """
        pairs = list(op_site_pairs)
        if len(pairs) == 0:
            raise ValueError("op_site_pairs must not be empty.")

        first_symbol, first_site = pairs[0]
        if not (0 <= int(first_site) < self.nsites):
            raise ValueError(f"Site index out of range: {first_site}")

        term = Op(str(first_symbol), int(first_site), coeff)
        for symbol, site in pairs[1:]:
            if not (0 <= int(site) < self.nsites):
                raise ValueError(f"Site index out of range: {site}")
            term = term * Op(str(symbol), int(site))

        self.terms.append(term)
        return self

    def add_uniform_onsite(self, symbol, coeff, start=0, stop=None):
        """Add ``coeff * sum_i O_i`` over sites in [start, stop)."""
        if stop is None:
            stop = self.nsites
        for i in range(int(start), int(stop)):
            self.add_term(coeff, [(symbol, i)])
        return self

    def add_uniform_bond(self, left_symbol, right_symbol, coeff, distance=1, periodic=False):
        """
        Add ``coeff * sum_i O_i P_{i+distance}``.
        """
        dist = int(distance)
        if dist <= 0:
            raise ValueError("distance must be a positive integer.")

        for i in range(self.nsites):
            j = i + dist
            if periodic:
                j = j % self.nsites
            elif j >= self.nsites:
                continue
            self.add_term(coeff, [(left_symbol, i), (right_symbol, j)])
        return self

    def build_model(self):
        """Build and return the underlying AutoMPO symbolic model."""
        return Model(basis=self.basis, ham_terms=self.terms)

    def build_mpo(self, algo="qr"):
        """Build MPO from current terms."""
        model = self.build_model()
        auto_mpo = AutoMPO(model, algo=algo)
        factors = []
        for w in auto_mpo.matrices:
            arr = np.asarray(w)
            if np.max(np.abs(np.imag(arr))) < 1e-12:
                arr = np.real(arr)
            factors.append(arr.transpose(0, 3, 1, 2))
        return MPO(factors)

    def dense_hamiltonian(self, algo="qr"):
        """Build and return dense Hamiltonian matrix."""
        return _mpo_to_dense_operator(self.build_mpo(algo=algo))

    def product_state_mps(self, local_indices, dtype=complex):
        """
        Build product-state MPS with one basis index per site.
        """
        if len(local_indices) != self.nsites:
            raise ValueError(
                f"Expected {self.nsites} local indices, got {len(local_indices)}."
            )
        factors = []
        for idx in local_indices:
            ii = int(idx)
            if not (0 <= ii < self.local_dim):
                raise ValueError(f"Local basis index out of range: {ii}")
            a = np.zeros((1, self.local_dim, 1), dtype=dtype)
            a[0, ii, 0] = 1.0
            factors.append(a)
        return MPS(factors, labels=["lv", "p", "rv"])


# Backward-compatible alias for older imports.
LatticeModel1D = Chain
