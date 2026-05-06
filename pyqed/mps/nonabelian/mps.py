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
from .basis import BondBasis, SiteBasis, TwoSiteBasis
from .contraction import merge_mps_sites
from .environment import contract_chain_expectation
from .tensor import NonabelianTensor


def _validate_sites(sites):
    sites = list(sites)
    if any(not isinstance(site, NonabelianTensor) or site.rank != 3 for site in sites):
        raise ValueError("MPS expects a sequence of rank-3 NonabelianTensor site tensors.")
    return sites


def _bond_basis_for_axis(tensor, axis, *, name):
    basis = (tensor.metadata or {}).get("bond_bases", {}).get(axis)
    if isinstance(basis, BondBasis):
        return basis
    return BondBasis.from_tensor_axis(tensor, axis, name=name)


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

    def site_bases(self, site):
        """
        Return explicit ``(left, physical, right)`` bases for one MPS site.
        """
        site = int(site)
        tensor = self.sites[site]
        return (
            _bond_basis_for_axis(tensor, 0, name=f"site-{site}-left"),
            SiteBasis.from_tensor_axis(tensor, 1, name=f"site-{site}-physical"),
            _bond_basis_for_axis(tensor, 2, name=f"site-{site}-right"),
        )

    def bond_basis(self, bond):
        """
        Return the explicit virtual basis on the bond between ``bond`` and ``bond + 1``.
        """
        bond = int(bond)
        if bond < 0 or bond + 1 >= len(self.sites):
            raise IndexError(f"Bond {bond} out of range for chain length {len(self.sites)}.")
        right = self.site_bases(bond)[2]
        left = self.site_bases(bond + 1)[0]
        if not right.dual_compatible_with(left):
            raise ValueError(f"MPS bond {bond} has incompatible right/left basis descriptors.")
        return right

    def local_two_site_basis(self, bond):
        """
        Return the explicit packed two-site basis currently used by the solver.
        """
        from .solver import pack_two_site_state

        merged = self.merge_bond(bond)
        _vec, layout = pack_two_site_state(merged)
        return TwoSiteBasis.from_tensor_and_layout(merged, layout)

    def expectation(self, mpo_factors):
        return contract_chain_expectation(self.sites, mpo_factors)
