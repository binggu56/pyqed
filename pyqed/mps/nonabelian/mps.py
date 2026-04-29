#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin MPS state wrapper for non-Abelian site tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    assert_mixed_canonical_sites,
    left_canonical_error,
    left_canonicalize_sites,
    mixed_canonical_errors,
    mixed_canonicalize_sites,
    right_canonical_error,
    right_canonicalize_sites,
)
from .contraction import merge_mps_sites
from .environment import contract_chain_expectation
from .tensor import NonabelianTensor


def _validate_sites(sites):
    sites = list(sites)
    if any(not isinstance(site, NonabelianTensor) or site.rank != 3 for site in sites):
        raise ValueError("MPS expects a sequence of rank-3 NonabelianTensor site tensors.")
    return sites


@dataclass
class MPS:
    """
    Minimal owner for a non-Abelian matrix product state.

    The optimization code still operates on site lists internally. This wrapper
    makes state-level metadata explicit without moving solver logic into the
    state object.
    """

    sites: list
    center: int | None = None
    target_sector: object | None = None

    def __post_init__(self):
        self.sites = _validate_sites(self.sites)
        if self.center is not None:
            self.center = int(self.center)
            if not 0 <= self.center < len(self.sites):
                raise IndexError(f"Center {self.center} out of range for chain length {len(self.sites)}.")

    @classmethod
    def from_sites(cls, sites, *, center=None, target_sector=None):
        if isinstance(sites, cls):
            copied = sites.copy()
            if center is not None:
                copied.center = int(center)
            if target_sector is not None:
                copied.target_sector = target_sector
            return copied
        return cls(list(sites), center=center, target_sector=target_sector)

    def __len__(self):
        return len(self.sites)

    def __iter__(self):
        return iter(self.sites)

    def __getitem__(self, item):
        return self.sites[item]

    def copy(self):
        return MPS(
            [site.copy() for site in self.sites],
            center=self.center,
            target_sector=self.target_sector,
        )

    def with_sites(self, sites, *, center=None):
        return MPS(
            sites,
            center=self.center if center is None else center,
            target_sector=self.target_sector,
        )

    def canonicalize(
        self,
        center,
        *,
        cutoff=0.0,
        max_bond=None,
        max_bond_mode="states",
        bond_coupling="left",
    ):
        self.sites = mixed_canonicalize_sites(
            self.sites,
            int(center),
            cutoff=cutoff,
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            bond_coupling=bond_coupling,
        )
        self.center = int(center)
        return self

    def left_canonicalize(
        self,
        *,
        cutoff=0.0,
        max_bond=None,
        max_bond_mode="states",
        bond_coupling="left",
    ):
        self.sites = left_canonicalize_sites(
            self.sites,
            cutoff=cutoff,
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            bond_coupling=bond_coupling,
        )
        self.center = len(self.sites) - 1 if self.sites else None
        return self

    def right_canonicalize(
        self,
        *,
        cutoff=0.0,
        max_bond=None,
        max_bond_mode="states",
        bond_coupling="left",
    ):
        self.sites = right_canonicalize_sites(
            self.sites,
            cutoff=cutoff,
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            bond_coupling=bond_coupling,
        )
        self.center = 0 if self.sites else None
        return self

    def canonical_errors(self, center=None):
        center = self.center if center is None else int(center)
        if center is None:
            raise ValueError("canonical_errors requires an explicit center or MPS.center.")
        return mixed_canonical_errors(self.sites, center)

    def assert_canonical(self, center=None, *, tol=1e-10):
        center = self.center if center is None else int(center)
        if center is None:
            raise ValueError("assert_canonical requires an explicit center or MPS.center.")
        assert_mixed_canonical_sites(self.sites, center, tol=tol)
        return self

    def left_error(self, site):
        return left_canonical_error(self.sites[int(site)])

    def right_error(self, site):
        return right_canonical_error(self.sites[int(site)])

    def merge_bond(self, bond):
        bond = int(bond)
        if bond < 0 or bond + 1 >= len(self.sites):
            raise IndexError(f"Bond {bond} out of range for chain length {len(self.sites)}.")
        return merge_mps_sites(self.sites[bond], self.sites[bond + 1])

    def expectation(self, mpo_factors):
        return contract_chain_expectation(self.sites, mpo_factors)
