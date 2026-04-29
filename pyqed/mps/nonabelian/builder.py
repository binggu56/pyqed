#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small symmetry-aware MPO builders for fixed-layout non-Abelian chains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.mps.su2 import SU2Irrep

from .contraction import normalize_site_tensor_layout
from .mpo import (
    PhysicalLeg,
    SiteOperator,
    MPO,
    IrreducibleChannelTerm,
    IrreducibleMPO,
    RankCoupledChannelTerm,
    RankCoupledMPO,
)
from .tensor import NonabelianTensor


def identity_operator(phys_leg, *, dtype=float):
    """
    Build the identity operator in the sector basis of one site.
    """
    if not isinstance(phys_leg, PhysicalLeg):
        raise TypeError("identity_operator expects a PhysicalLeg.")
    blocks = {}
    for sector in phys_leg.sectors:
        dim = phys_leg.dim(sector)
        blocks[(sector, sector)] = np.eye(dim, dtype=dtype)
    return SiteOperator(blocks=blocks, phys_out_leg=phys_leg, phys_in_leg=phys_leg)


def _compose_site_operators(*operators):
    if not operators:
        raise ValueError("_compose_site_operators requires at least one operator.")
    first = operators[0]
    if not isinstance(first, SiteOperator):
        raise TypeError("_compose_site_operators expects SiteOperator inputs.")
    dense = first.as_dense()
    for operator in operators[1:]:
        if not isinstance(operator, SiteOperator):
            raise TypeError("_compose_site_operators expects SiteOperator inputs.")
        if (
            operator.phys_out_leg != first.phys_out_leg
            or operator.phys_in_leg != first.phys_in_leg
        ):
            raise ValueError("_compose_site_operators requires matching physical legs.")
        dense = dense @ operator.as_dense()
    return SiteOperator.from_dense(
        dense,
        phys_out_leg=first.phys_out_leg,
        phys_in_leg=first.phys_in_leg,
    )


def _parity_operator(phys_leg, *, dtype=float):
    weights = {}
    for sector in phys_leg.sectors:
        charge = getattr(sector, "charge", None)
        if charge is None:
            raise ValueError(
                "Cannot infer fermionic parity for a PhysicalLeg whose sectors have no charge."
            )
        weights[sector] = float((-1) ** int(charge))
    blocks = {}
    for sector in phys_leg.sectors:
        dim = phys_leg.dim(sector)
        blocks[(sector, sector)] = np.asarray(weights[sector], dtype=dtype) * np.eye(
            dim, dtype=dtype
        )
    return SiteOperator(blocks=blocks, phys_out_leg=phys_leg, phys_in_leg=phys_leg)


def _site_physical_leg_from_tensor(site):
    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        raise TypeError("_site_physical_leg_from_tensor expects a rank-3 NonabelianTensor.")
    site = normalize_site_tensor_layout(site)
    dims = {}
    for key, block in site.data.items():
        dims.setdefault(key[1], int(block.shape[1]))
    return PhysicalLeg.from_dims(dims, sectors=tuple(dict.fromkeys(site.qns[1])))


def _validate_site_operator(site_operator, phys_leg, *, label):
    if not isinstance(site_operator, SiteOperator):
        raise TypeError(f"{label} expects a SiteOperator.")
    if site_operator.phys_out_leg != phys_leg or site_operator.phys_in_leg != phys_leg:
        raise ValueError(f"{label} physical legs must match the builder site leg.")


def _validate_reduced_operator(reduced_operator, phys_leg, *, label):
    if getattr(reduced_operator, "phys_out_leg", None) != phys_leg or getattr(
        reduced_operator, "phys_in_leg", None
    ) != phys_leg:
        raise ValueError(f"{label} physical legs must match the builder site leg.")
    if not hasattr(reduced_operator, "components") or not hasattr(
        reduced_operator, "right_multiply_sector_scalar"
    ):
        raise TypeError(
            f"{label} expects a reduced operator exposing components and right_multiply_sector_scalar()."
        )


def _accumulate_site_operator(blocks, operator, row, col, *, coeff=1.0, d_left, d_right, dtype):
    coeff = np.asarray(coeff, dtype=dtype)
    for key, block in operator.blocks.items():
        arr = blocks.get(key)
        if arr is None:
            arr = np.zeros((d_left, d_right) + block.shape, dtype=dtype)
            blocks[key] = arr
        arr[row, col] += coeff * np.asarray(block, dtype=dtype)


@dataclass(frozen=True)
class _ProductTerm:
    sites: tuple[int, ...]
    operators: tuple[SiteOperator, ...]
    coeff: complex


@dataclass(frozen=True)
class _ReducedBilinearTerm:
    left_site: int
    right_site: int
    left_operator: object
    right_operator: object
    middle_operators: tuple[tuple[int, SiteOperator], ...]
    coeff: complex

    @property
    def components(self):
        return tuple(int(component) for component in self.left_operator.components)


class AutoMPO:
    """
    Assemble simple finite-chain MPOs from symmetry-aware site operators.

    Supported terms:
    - arbitrary product strings via ``add_term((i, A_i), (j, B_j), ...)``
    - onsite terms via ``add_onsite(...)``
    - nearest-neighbor products via ``add_nearest_neighbor(...)``
    """

    def __init__(self, site_legs):
        site_legs = tuple(site_legs)
        if len(site_legs) < 2:
            raise ValueError("AutoMPO requires at least two sites.")
        if any(not isinstance(leg, PhysicalLeg) for leg in site_legs):
            raise TypeError("AutoMPO expects PhysicalLeg entries.")
        self.site_legs = site_legs
        self.nsites = len(site_legs)
        self._terms = []

    @classmethod
    def from_sites(cls, sites):
        return cls([_site_physical_leg_from_tensor(site) for site in sites])

    def add_onsite(self, site, operator, *, coeff=1.0):
        return self.add_term((site, operator), coeff=coeff)

    def add_nearest_neighbor(self, site, left_operator, right_operator, *, coeff=1.0):
        return self.add_term(
            (site, left_operator),
            (site + 1, right_operator),
            coeff=coeff,
        )

    def add_fermionic_bilinear(
        self,
        left_site,
        left_operator,
        right_site,
        right_operator,
        *,
        coeff=1.0,
        parity_operator=None,
    ):
        """
        Add a site-ordered fermionic bilinear with Jordan-Wigner string handling.

        Parameters
        ----------
        left_site, right_site
            Endpoints of the bilinear. ``left_site`` must be smaller than
            ``right_site``; the operators are interpreted in this ascending site
            order.
        left_operator, right_operator
            Local odd fermionic operators on the lower/higher site. For example,
            the Hermitian-conjugate hopping term ``c^dagger_j c_i`` with
            ``i < j`` should be supplied as ``(i, c_i)`` and ``(j, c^dagger_j)``.
        parity_operator
            Optional site-local parity operator. If omitted, parity is inferred
            from the site leg charges as ``(-1)^N``.
        """
        left_site = int(left_site)
        right_site = int(right_site)
        if left_site < 0 or right_site >= self.nsites:
            raise IndexError(
                f"Fermionic bilinear sites {(left_site, right_site)!r} out of range."
            )
        if left_site >= right_site:
            raise ValueError(
                "add_fermionic_bilinear requires left_site < right_site in chain order."
            )
        _validate_site_operator(
            left_operator,
            self.site_legs[left_site],
            label="add_fermionic_bilinear",
        )
        _validate_site_operator(
            right_operator,
            self.site_legs[right_site],
            label="add_fermionic_bilinear",
        )

        dtype = np.result_type(
            left_operator.dtype,
            right_operator.dtype,
            np.asarray(coeff).dtype,
            float,
        )

        if parity_operator is None:
            left_parity = _parity_operator(self.site_legs[left_site], dtype=dtype)
            middle_parities = {
                site: _parity_operator(self.site_legs[site], dtype=dtype)
                for site in range(left_site + 1, right_site)
            }
        else:
            if isinstance(parity_operator, SiteOperator):
                parity_ops = {site: parity_operator for site in range(left_site, right_site)}
            else:
                parity_ops = dict(parity_operator)
            left_parity = parity_ops[left_site]
            for site in range(left_site, right_site):
                _validate_site_operator(
                    parity_ops[site],
                    self.site_legs[site],
                    label="add_fermionic_bilinear parity_operator",
                )
            middle_parities = {
                site: parity_ops[site] for site in range(left_site + 1, right_site)
            }

        site_operators = [(left_site, _compose_site_operators(left_operator, left_parity))]
        site_operators.extend(sorted(middle_parities.items(), key=lambda item: item[0]))
        site_operators.append((right_site, right_operator))
        return self.add_term(*site_operators, coeff=coeff)

    def add_fermionic_reduced_bilinear(
        self,
        left_site,
        left_operator,
        right_site,
        right_operator,
        *,
        coeff=1.0,
        parity_operator=None,
    ):
        """
        Add a site-ordered fermionic bilinear using reduced endpoint operators.

        The reduced endpoints are carried through the resulting MPO core without
        expanding all physical-sector blocks up front. Distinct spherical
        components are routed through separate virtual subchannels so the chain
        still contracts componentwise.
        """
        left_site = int(left_site)
        right_site = int(right_site)
        if left_site < 0 or right_site >= self.nsites:
            raise IndexError(
                f"Fermionic reduced bilinear sites {(left_site, right_site)!r} out of range."
            )
        if left_site >= right_site:
            raise ValueError(
                "add_fermionic_reduced_bilinear requires left_site < right_site in chain order."
            )
        _validate_reduced_operator(
            left_operator,
            self.site_legs[left_site],
            label="add_fermionic_reduced_bilinear",
        )
        _validate_reduced_operator(
            right_operator,
            self.site_legs[right_site],
            label="add_fermionic_reduced_bilinear",
        )
        if tuple(int(component) for component in left_operator.components) != tuple(
            int(component) for component in right_operator.components
        ):
            raise ValueError(
                "add_fermionic_reduced_bilinear currently requires matching endpoint component labels."
            )

        dtype = np.result_type(
            getattr(left_operator, "dtype", float),
            getattr(right_operator, "dtype", float),
            np.asarray(coeff).dtype,
            float,
        )

        if parity_operator is None:
            left_parity = _parity_operator(self.site_legs[left_site], dtype=dtype)
            middle_parities = {
                site: _parity_operator(self.site_legs[site], dtype=dtype)
                for site in range(left_site + 1, right_site)
            }
        else:
            if isinstance(parity_operator, SiteOperator):
                parity_ops = {site: parity_operator for site in range(left_site, right_site)}
            else:
                parity_ops = dict(parity_operator)
            left_parity = parity_ops[left_site]
            for site in range(left_site, right_site):
                _validate_site_operator(
                    parity_ops[site],
                    self.site_legs[site],
                    label="add_fermionic_reduced_bilinear parity_operator",
                )
            middle_parities = {
                site: parity_ops[site] for site in range(left_site + 1, right_site)
            }

        self._terms.append(
            _ReducedBilinearTerm(
                left_site=left_site,
                right_site=right_site,
                left_operator=left_operator.right_multiply_sector_scalar(left_parity),
                right_operator=right_operator,
                middle_operators=tuple(sorted(middle_parities.items(), key=lambda item: item[0])),
                coeff=coeff,
            )
        )
        return self

    def add_term(self, *site_operators, coeff=1.0):
        """
        Add a product term ``coeff * O_i O_j ...`` to the MPO.

        Parameters
        ----------
        site_operators
            Sequence of ``(site, operator)`` pairs. Sites may be supplied in any
            order; they are stored in ascending site order.
        coeff
            Scalar coefficient attached to the first operator insertion channel.
        """
        if not site_operators:
            raise ValueError("add_term requires at least one (site, operator) pair.")

        normalized = []
        for item in site_operators:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "add_term expects arguments like (site, operator)."
                )
            site, operator = item
            site = int(site)
            if site < 0 or site >= self.nsites:
                raise IndexError(f"Term site {site} out of range for {self.nsites} sites.")
            _validate_site_operator(
                operator,
                self.site_legs[site],
                label="add_term",
            )
            normalized.append((site, operator))

        normalized.sort(key=lambda item: item[0])
        sites = tuple(site for site, _ in normalized)
        if len(set(sites)) != len(sites):
            raise ValueError("add_term does not allow repeated site indices in one term.")

        self._terms.append(
            _ProductTerm(
                sites=sites,
                operators=tuple(operator for _, operator in normalized),
                coeff=coeff,
            )
        )
        return self

    def build(self):
        channel_offsets = []
        channel_irreps = [SU2Irrep(0)]
        next_channel = 1
        for term in self._terms:
            if isinstance(term, _ProductTerm):
                width = 1
                irrep = SU2Irrep(0)
            elif isinstance(term, _ReducedBilinearTerm):
                width = 1
                irrep = term.left_operator.base_operator.rank_irrep if hasattr(
                    term.left_operator, "base_operator"
                ) else term.left_operator.rank_irrep
            else:
                raise TypeError(f"Unsupported AutoMPO term type {type(term).__name__}.")
            channel_offsets.append((next_channel, width, irrep))
            channel_irreps.extend([irrep] * width)
            next_channel += width
        dim = next_channel + 1
        left_edge = 0
        right_edge = dim - 1
        channel_irreps.append(SU2Irrep(0))

        mpo = []
        dtype = np.result_type(
            *[
                (
                    operator.dtype
                    if isinstance(term, _ProductTerm)
                    else getattr(operator, "dtype", float)
                )
                for term in self._terms
                for operator in (
                    term.operators
                    if isinstance(term, _ProductTerm)
                    else (term.left_operator, term.right_operator)
                )
            ],
            *[
                np.asarray(term.coeff).dtype
                for term in self._terms
            ],
            float,
        )

        term_channels = []
        for (channel_start, width, irrep), term in zip(channel_offsets, self._terms):
            if isinstance(term, _ProductTerm):
                term_map = {
                    site: operator for site, operator in zip(term.sites, term.operators)
                }
                term_channels.append(("dense", channel_start, width, irrep, term, term_map))
            else:
                middle_map = dict(term.middle_operators)
                term_channels.append(("reduced", channel_start, width, irrep, term, middle_map))

        for site, phys_leg in enumerate(self.site_legs):
            visible_blocks = {}
            rank_coupled_terms = []
            ident = identity_operator(phys_leg, dtype=dtype)
            _accumulate_site_operator(
                visible_blocks,
                ident,
                left_edge,
                left_edge,
                d_left=dim,
                d_right=dim,
                dtype=dtype,
            )
            _accumulate_site_operator(
                visible_blocks,
                ident,
                right_edge,
                right_edge,
                d_left=dim,
                d_right=dim,
                dtype=dtype,
            )

            for kind, channel_start, width, irrep, term, term_map in term_channels:
                if kind == "dense":
                    channel = channel_start
                    first_site = term.sites[0]
                    last_site = term.sites[-1]
                    if site < first_site or site > last_site:
                        continue

                    if len(term.sites) == 1 and site == first_site:
                        _accumulate_site_operator(
                            visible_blocks,
                            term_map[site],
                            left_edge,
                            right_edge,
                            coeff=term.coeff,
                            d_left=dim,
                            d_right=dim,
                            dtype=dtype,
                        )
                        continue

                    if site == first_site:
                        _accumulate_site_operator(
                            visible_blocks,
                            term_map[site],
                            left_edge,
                            channel,
                            coeff=term.coeff,
                            d_left=dim,
                            d_right=dim,
                            dtype=dtype,
                        )
                    elif site == last_site:
                        _accumulate_site_operator(
                            visible_blocks,
                            term_map[site],
                            channel,
                            right_edge,
                            d_left=dim,
                            d_right=dim,
                            dtype=dtype,
                        )
                    else:
                        operator = term_map.get(site, ident)
                        _accumulate_site_operator(
                            visible_blocks,
                            operator,
                            channel,
                            channel,
                            d_left=dim,
                            d_right=dim,
                            dtype=dtype,
                        )
                    continue

                first_site = term.left_site
                last_site = term.right_site
                if site < first_site or site > last_site:
                    continue

                channel = channel_start
                if site == first_site:
                    visible_virtual_block = np.zeros((dim, dim), dtype=dtype)
                    visible_virtual_block[left_edge, channel] = np.asarray(term.coeff, dtype=dtype)
                    rank_coupled_terms.append(
                        RankCoupledChannelTerm(
                            reduced_operator=term.left_operator,
                            visible_virtual_block=visible_virtual_block,
                        )
                    )
                elif site == last_site:
                    visible_virtual_block = np.zeros((dim, dim), dtype=dtype)
                    visible_virtual_block[channel, right_edge] = 1.0
                    rank_coupled_terms.append(
                        RankCoupledChannelTerm(
                            reduced_operator=term.right_operator,
                            visible_virtual_block=visible_virtual_block,
                        )
                    )
                else:
                    operator = term_map.get(site, ident)
                    _accumulate_site_operator(
                        visible_blocks,
                        operator,
                        channel,
                        channel,
                        d_left=dim,
                        d_right=dim,
                        dtype=dtype,
                    )

            if site == 0:
                sliced = {
                    key: block[0:1, :, :, :]
                    for key, block in visible_blocks.items()
                }
            elif site == self.nsites - 1:
                sliced = {
                    key: block[:, right_edge:right_edge + 1, :, :]
                    for key, block in visible_blocks.items()
                }
            else:
                sliced = visible_blocks

            left_irreps = tuple(channel_irreps if site > 0 else [channel_irreps[0]])
            right_irreps = tuple(channel_irreps if site < self.nsites - 1 else [channel_irreps[-1]])

            if site == 0 or site == self.nsites - 1:
                sliced_rank_coupled_terms = []
                for term in rank_coupled_terms:
                    trimmed = (
                        term.visible_virtual_block[0:1, :]
                        if site == 0
                        else term.visible_virtual_block[:, right_edge:right_edge + 1]
                    )
                    sliced_rank_coupled_terms.append(
                        RankCoupledChannelTerm(
                            reduced_operator=term.reduced_operator,
                            visible_virtual_block=trimmed,
                        )
                    )
            else:
                sliced_rank_coupled_terms = rank_coupled_terms

            if sliced_rank_coupled_terms:
                mpo.append(
                    RankCoupledMPO(
                        dense_blocks=sliced,
                        left_channel_irreps=left_irreps,
                        right_channel_irreps=right_irreps,
                        reduced_terms=tuple(sliced_rank_coupled_terms),
                        phys_out_leg=phys_leg,
                        phys_in_leg=phys_leg,
                    )
                )
            else:
                mpo.append(
                    MPO(
                        blocks=sliced,
                        phys_out_leg=phys_leg,
                        phys_in_leg=phys_leg,
                    )
                )

        return mpo
