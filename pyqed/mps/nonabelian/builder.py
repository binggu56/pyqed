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


def _dense_signature(array, *, tol=1.0e-14):
    dense = np.asarray(array, dtype=np.complex128)
    dense = np.array(dense, copy=True)
    dense[np.abs(dense) <= tol] = 0.0
    return dense.shape, dense.tobytes()


def _leg_signature(leg):
    return (
        tuple(leg.sectors),
        tuple((sector, int(leg.dim(sector))) for sector in leg.sectors),
    )


def _site_operator_signature(operator):
    return (
        "site",
        _leg_signature(operator.phys_out_leg),
        _leg_signature(operator.phys_in_leg),
        _dense_signature(operator.as_dense()),
    )


def _reduced_operator_signature(operator):
    components = tuple(int(component) for component in operator.components)
    phys_out_leg = operator.phys_out_leg
    phys_in_leg = operator.phys_in_leg
    blocks = []
    for component in components:
        for q_out in phys_out_leg.sectors:
            for q_in in phys_in_leg.sectors:
                block = operator.component_block(component, q_out, q_in)
                if block is None:
                    blocks.append((component, q_out, q_in, None))
                else:
                    blocks.append((component, q_out, q_in, _dense_signature(block)))
    rank_irrep = (
        operator.rank_irrep
        if hasattr(operator, "rank_irrep")
        else operator.base_operator.rank_irrep
        if hasattr(operator, "base_operator")
        else None
    )
    return (
        "reduced",
        rank_irrep,
        components,
        _leg_signature(phys_out_leg),
        _leg_signature(phys_in_leg),
        tuple(blocks),
    )


@dataclass(frozen=True)
class _ProductTerm:
    sites: tuple[int, ...]
    operators: tuple[SiteOperator, ...]
    coeff: complex
    family: object = None


@dataclass(frozen=True)
class _ReducedBilinearTerm:
    left_site: int
    right_site: int
    left_operator: object
    right_operator: object
    middle_operators: tuple[tuple[int, SiteOperator], ...]
    coeff: complex
    family: object = None

    @property
    def components(self):
        return tuple(int(component) for component in self.left_operator.components)


@dataclass(frozen=True)
class _ReducedBilinearProductTerm:
    dense_sites: tuple[int, ...]
    dense_operators: tuple[SiteOperator, ...]
    left_site: int
    right_site: int
    left_operator: object
    right_operator: object
    middle_operators: tuple[tuple[int, SiteOperator], ...]
    coeff: complex
    family: object = None


@dataclass(frozen=True)
class _ReducedStringTerm:
    sites: tuple[int, ...]
    operators: tuple[object, ...]
    intermediate_irreps: tuple[SU2Irrep, ...]
    middle_operators: tuple[tuple[int, SiteOperator], ...]
    coeff: complex
    family: object = None


@dataclass(frozen=True)
class _ReducedStringProductTerm:
    dense_sites: tuple[int, ...]
    dense_operators: tuple[SiteOperator, ...]
    sites: tuple[int, ...]
    operators: tuple[object, ...]
    intermediate_irreps: tuple[SU2Irrep, ...]
    middle_operators: tuple[tuple[int, SiteOperator], ...]
    coeff: complex
    family: object = None


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

    def add_onsite(self, site, operator, *, coeff=1.0, family=None):
        return self.add_term((site, operator), coeff=coeff, family=family)

    def add_nearest_neighbor(
        self,
        site,
        left_operator,
        right_operator,
        *,
        coeff=1.0,
        family=None,
    ):
        return self.add_term(
            (site, left_operator),
            (site + 1, right_operator),
            coeff=coeff,
            family=family,
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
        family=None,
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
        return self.add_term(*site_operators, coeff=coeff, family=family)

    def add_fermionic_reduced_bilinear(
        self,
        left_site,
        left_operator,
        right_site,
        right_operator,
        *,
        coeff=1.0,
        parity_operator=None,
        family=None,
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
                family=family,
            )
        )
        return self

    def add_fermionic_reduced_bilinear_product(
        self,
        *dense_site_operators,
        left_site,
        left_operator,
        right_site,
        right_operator,
        coeff=1.0,
        parity_operator=None,
        family=None,
    ):
        """
        Add dense scalar site operators multiplied by a reduced fermionic bilinear.

        This is used by fully reduced spin-free ERI terms such as
        ``n_p E_rs`` without expanding the reduced bilinear into spin
        components.
        """

        left_site = int(left_site)
        right_site = int(right_site)
        if left_site < 0 or right_site >= self.nsites or left_site >= right_site:
            raise ValueError("add_fermionic_reduced_bilinear_product requires 0 <= left_site < right_site < nsites.")
        _validate_reduced_operator(left_operator, self.site_legs[left_site], label="add_fermionic_reduced_bilinear_product")
        _validate_reduced_operator(right_operator, self.site_legs[right_site], label="add_fermionic_reduced_bilinear_product")

        dense_by_site = {}
        for item in dense_site_operators:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("dense_site_operators must be (site, operator) pairs.")
            site, operator = item
            site = int(site)
            if site in {left_site, right_site}:
                raise ValueError("Dense scalar factors cannot share a reduced bilinear endpoint.")
            _validate_site_operator(operator, self.site_legs[site], label="add_fermionic_reduced_bilinear_product")
            dense_by_site[site] = (
                _compose_site_operators(dense_by_site[site], operator)
                if site in dense_by_site
                else operator
            )

        dtype = np.result_type(
            getattr(left_operator, "dtype", float),
            getattr(right_operator, "dtype", float),
            *[operator.dtype for operator in dense_by_site.values()],
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
            middle_parities = {
                site: parity_ops[site] for site in range(left_site + 1, right_site)
            }
        dense_middle = dict(middle_parities)
        for site, operator in dense_by_site.items():
            dense_middle[site] = (
                _compose_site_operators(dense_middle[site], operator)
                if site in dense_middle
                else operator
            )

        self._terms.append(
            _ReducedBilinearProductTerm(
                dense_sites=tuple(sorted(dense_by_site)),
                dense_operators=tuple(dense_by_site[site] for site in sorted(dense_by_site)),
                left_site=left_site,
                right_site=right_site,
                left_operator=left_operator.right_multiply_sector_scalar(left_parity),
                right_operator=right_operator,
                middle_operators=tuple(sorted(dense_middle.items(), key=lambda item: item[0])),
                coeff=coeff,
                family=family,
            )
        )
        return self

    def add_term(self, *site_operators, coeff=1.0, family=None):
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
                family=family,
            )
        )
        return self

    def add_reduced_string(
        self,
        *site_operators,
        intermediate_irreps,
        coeff=1.0,
        middle_operators=None,
        family=None,
    ):
        """
        Add a Wigner-Eckart coupled ordered string of reduced local tensors.

        ``intermediate_irreps`` gives the virtual SU(2) irrep after each
        insertion except the final one.  The left and right boundary channels
        are scalar, so a four-fermion scalar string typically uses
        ``(1/2, 0, 1/2)`` in doubled-spin notation.
        """
        if len(site_operators) < 2:
            raise ValueError("add_reduced_string requires at least two reduced insertions.")
        intermediate_irreps = tuple(intermediate_irreps)
        if len(intermediate_irreps) != len(site_operators) - 1:
            raise ValueError(
                "intermediate_irreps must contain one irrep after each non-final insertion."
            )
        if any(not isinstance(irrep, SU2Irrep) for irrep in intermediate_irreps):
            raise TypeError("intermediate_irreps must contain SU2Irrep objects.")

        normalized = []
        previous_site = -1
        for item in site_operators:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "add_reduced_string expects arguments like (site, reduced_operator)."
                )
            site, operator = item
            site = int(site)
            if site < 0 or site >= self.nsites:
                raise IndexError(f"Reduced-string site {site} out of range.")
            if site <= previous_site:
                raise ValueError(
                    "add_reduced_string currently requires strictly increasing site order."
                )
            _validate_reduced_operator(
                operator,
                self.site_legs[site],
                label="add_reduced_string",
            )
            normalized.append((site, operator))
            previous_site = site

        middle = []
        for site, operator in dict(middle_operators or {}).items():
            site = int(site)
            if site < normalized[0][0] or site > normalized[-1][0]:
                raise ValueError("middle_operators must lie within the reduced string span.")
            if site in {entry_site for entry_site, _ in normalized}:
                raise ValueError("middle_operators cannot be placed on reduced insertion sites.")
            _validate_site_operator(
                operator,
                self.site_legs[site],
                label="add_reduced_string middle_operators",
            )
            middle.append((site, operator))

        self._terms.append(
            _ReducedStringTerm(
                sites=tuple(site for site, _ in normalized),
                operators=tuple(operator for _, operator in normalized),
                intermediate_irreps=intermediate_irreps,
                middle_operators=tuple(sorted(middle, key=lambda item: item[0])),
                coeff=coeff,
                family=family,
            )
        )
        return self

    def add_reduced_string_product(
        self,
        *site_operators,
        intermediate_irreps,
        dense_site_operators=(),
        coeff=1.0,
        middle_operators=None,
        family=None,
    ):
        """
        Add scalar local factors multiplied by a Wigner-Eckart reduced string.

        :param site_operators: Strictly increasing ``(site, reduced_operator)``
            insertions.
        :param intermediate_irreps: Virtual SU(2) irreps after each non-final
            reduced insertion.
        :param dense_site_operators: Optional ``(site, SiteOperator)`` scalar
            factors.  They may be outside the reduced-string span or share a
            site with middle scalar operators, but not with reduced insertions.
        :param coeff: Scalar coefficient for the complete product.
        :param middle_operators: Optional scalar operators carried between
            reduced insertions, typically Jordan-Wigner parities.
        :returns: ``self``.
        """
        if len(site_operators) < 2:
            raise ValueError("add_reduced_string_product requires at least two reduced insertions.")
        intermediate_irreps = tuple(intermediate_irreps)
        if len(intermediate_irreps) != len(site_operators) - 1:
            raise ValueError(
                "intermediate_irreps must contain one irrep after each non-final insertion."
            )
        if any(not isinstance(irrep, SU2Irrep) for irrep in intermediate_irreps):
            raise TypeError("intermediate_irreps must contain SU2Irrep objects.")

        normalized = []
        previous_site = -1
        for item in site_operators:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "add_reduced_string_product expects reduced insertions like (site, reduced_operator)."
                )
            site, operator = item
            site = int(site)
            if site < 0 or site >= self.nsites:
                raise IndexError(f"Reduced-string site {site} out of range.")
            if site <= previous_site:
                raise ValueError(
                    "add_reduced_string_product requires strictly increasing reduced insertion sites."
                )
            _validate_reduced_operator(
                operator,
                self.site_legs[site],
                label="add_reduced_string_product",
            )
            normalized.append((site, operator))
            previous_site = site
        reduced_sites = {site for site, _ in normalized}

        dense_by_site = {}
        for item in tuple(dense_site_operators):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("dense_site_operators must contain (site, operator) pairs.")
            site, operator = item
            site = int(site)
            if site < 0 or site >= self.nsites:
                raise IndexError(f"Dense scalar site {site} out of range.")
            if site in reduced_sites:
                raise ValueError("Dense scalar factors cannot share reduced insertion sites.")
            _validate_site_operator(operator, self.site_legs[site], label="add_reduced_string_product")
            dense_by_site[site] = (
                _compose_site_operators(dense_by_site[site], operator)
                if site in dense_by_site
                else operator
            )

        middle_by_site = {}
        for site, operator in dict(middle_operators or {}).items():
            site = int(site)
            if site < normalized[0][0] or site > normalized[-1][0]:
                raise ValueError("middle_operators must lie within the reduced string span.")
            if site in reduced_sites:
                raise ValueError("middle_operators cannot be placed on reduced insertion sites.")
            _validate_site_operator(
                operator,
                self.site_legs[site],
                label="add_reduced_string_product middle_operators",
            )
            middle_by_site[site] = (
                _compose_site_operators(middle_by_site[site], operator)
                if site in middle_by_site
                else operator
            )

        for site, operator in middle_by_site.items():
            dense_by_site[site] = (
                _compose_site_operators(dense_by_site[site], operator)
                if site in dense_by_site
                else operator
            )

        self._terms.append(
            _ReducedStringProductTerm(
                dense_sites=tuple(sorted(dense_by_site)),
                dense_operators=tuple(dense_by_site[site] for site in sorted(dense_by_site)),
                sites=tuple(site for site, _ in normalized),
                operators=tuple(operator for _, operator in normalized),
                intermediate_irreps=intermediate_irreps,
                middle_operators=(),
                coeff=coeff,
                family=family,
            )
        )
        return self

    def build(self):
        if not self._terms:
            return []
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
                    else term.operators
                    if isinstance(term, _ReducedStringTerm)
                    else (
                        term.dense_operators
                        + term.operators
                    )
                    if isinstance(term, _ReducedStringProductTerm)
                    else (
                        term.dense_operators
                        + (term.left_operator, term.right_operator)
                    )
                    if isinstance(term, _ReducedBilinearProductTerm)
                    else (term.left_operator, term.right_operator)
                )
            ],
            *[np.asarray(term.coeff).dtype for term in self._terms],
            float,
        )

        scalar = SU2Irrep(0)
        start_state = 0
        final_state = 1
        state_irreps = [scalar, scalar]
        prefix_states = {}

        dense_transitions = [dict() for _ in range(self.nsites)]
        reduced_transitions = [dict() for _ in range(self.nsites)]
        identity_loops = [set() for _ in range(self.nsites)]

        def get_prefix_state(prefix, irrep):
            key = tuple(prefix)
            state = prefix_states.get(key)
            if state is not None:
                if state_irreps[state] != irrep:
                    raise ValueError("Shared AutoMPO prefix resolved to incompatible SU(2) irrep.")
                return state
            state = len(state_irreps)
            prefix_states[key] = state
            state_irreps.append(irrep)
            return state

        def add_identity_loop(site, state):
            if 0 <= site < self.nsites:
                identity_loops[site].add(int(state))

        def _family_tuple(family):
            if family is None:
                return ()
            if isinstance(family, str):
                return (family,)
            return tuple(str(item) for item in family if item is not None)

        def add_dense_transition(
            site,
            left,
            right,
            operator,
            coeff,
            *,
            accumulate=True,
            family=None,
        ):
            if abs(coeff) <= 0.0:
                return
            signature = _site_operator_signature(operator)
            family_key = _family_tuple(family)
            key = (int(left), int(right), signature, family_key)
            if not accumulate and key in dense_transitions[int(site)]:
                return
            old_coeff, stored_operator = dense_transitions[int(site)].get(key, (0.0, operator))
            dense_transitions[int(site)][key] = (old_coeff + coeff, stored_operator)

        def add_reduced_transition(
            site,
            left,
            right,
            operator,
            coeff,
            *,
            use_cg_coupling,
            accumulate=True,
            family=None,
        ):
            if abs(coeff) <= 0.0:
                return
            signature = _reduced_operator_signature(operator)
            family_key = _family_tuple(family)
            key = (int(left), int(right), bool(use_cg_coupling), signature, family_key)
            if not accumulate and key in reduced_transitions[int(site)]:
                return
            old_coeff, stored_operator = reduced_transitions[int(site)].get(key, (0.0, operator))
            reduced_transitions[int(site)][key] = (old_coeff + coeff, stored_operator)

        def insert_path(steps, coeff, *, family=None):
            if not steps:
                return
            prefix = []
            current_state = start_state
            previous_site = None
            for index, step in enumerate(steps):
                site = int(step["site"])
                if previous_site is not None:
                    for gap_site in range(previous_site + 1, site):
                        add_identity_loop(gap_site, current_state)
                is_terminal = index == len(steps) - 1
                next_irrep = step["next_irrep"]
                if is_terminal:
                    next_state = final_state
                    transition_coeff = coeff
                else:
                    if step["kind"] == "dense":
                        signature = _site_operator_signature(step["operator"])
                    else:
                        signature = (
                            _reduced_operator_signature(step["operator"]),
                            bool(step.get("use_cg_coupling", False)),
                        )
                    prefix.append((site, step["kind"], signature, next_irrep))
                    next_state = get_prefix_state(prefix, next_irrep)
                    transition_coeff = 1.0

                if step["kind"] == "dense":
                    add_dense_transition(
                        site,
                        current_state,
                        next_state,
                        step["operator"],
                        transition_coeff,
                        accumulate=is_terminal,
                        family=family,
                    )
                elif step["kind"] == "reduced":
                    add_reduced_transition(
                        site,
                        current_state,
                        next_state,
                        step["operator"],
                        transition_coeff,
                        use_cg_coupling=step.get("use_cg_coupling", False),
                        accumulate=is_terminal,
                        family=family,
                    )
                else:
                    raise TypeError(f"Unsupported AutoMPO path step kind {step['kind']!r}.")
                current_state = next_state
                previous_site = site

        def product_steps(term):
            return [
                {
                    "site": site,
                    "kind": "dense",
                    "operator": operator,
                    "next_irrep": scalar,
                }
                for site, operator in zip(term.sites, term.operators)
            ]

        def reduced_bilinear_steps(term):
            rank_irrep = (
                term.left_operator.base_operator.rank_irrep
                if hasattr(term.left_operator, "base_operator")
                else term.left_operator.rank_irrep
            )
            insertions = {
                int(term.left_site): {
                    "site": int(term.left_site),
                    "kind": "reduced",
                    "operator": term.left_operator,
                    "next_irrep": rank_irrep,
                    "use_cg_coupling": False,
                },
                int(term.right_site): {
                    "site": int(term.right_site),
                    "kind": "reduced",
                    "operator": term.right_operator,
                    "next_irrep": scalar,
                    "use_cg_coupling": False,
                },
            }
            middle = {
                int(site): operator for site, operator in term.middle_operators
            }
            steps = []
            current_irrep = scalar
            for site in sorted(set(insertions) | set(middle)):
                if site in insertions:
                    step = dict(insertions[site])
                    current_irrep = step["next_irrep"]
                else:
                    step = {
                        "site": site,
                        "kind": "dense",
                        "operator": middle[site],
                        "next_irrep": current_irrep,
                    }
                steps.append(step)
            return steps

        def reduced_bilinear_product_steps(term):
            steps = reduced_bilinear_steps(term)
            by_site = {step["site"]: step for step in steps}
            for site, operator in zip(term.dense_sites, term.dense_operators):
                site = int(site)
                if site in by_site:
                    step = by_site[site]
                    if step["kind"] != "dense":
                        raise ValueError("Dense scalar factors cannot overlap reduced insertion sites.")
                    step["operator"] = _compose_site_operators(step["operator"], operator)
                else:
                    by_site[site] = {
                        "site": site,
                        "kind": "dense",
                        "operator": operator,
                        "next_irrep": scalar,
                    }
            ordered = []
            current_irrep = scalar
            for site in sorted(by_site):
                step = dict(by_site[site])
                if step["kind"] == "dense":
                    step["next_irrep"] = current_irrep
                else:
                    current_irrep = step["next_irrep"]
                ordered.append(step)
            return ordered

        def reduced_string_steps(term):
            insertions = {}
            for position, (site, operator) in enumerate(zip(term.sites, term.operators)):
                next_irrep = (
                    term.intermediate_irreps[position]
                    if position < len(term.intermediate_irreps)
                    else scalar
                )
                insertions[int(site)] = {
                    "site": int(site),
                    "kind": "reduced",
                    "operator": operator,
                    "next_irrep": next_irrep,
                    "use_cg_coupling": True,
                }
            middle = {
                int(site): operator for site, operator in term.middle_operators
            }
            steps = []
            current_irrep = scalar
            for site in sorted(set(insertions) | set(middle)):
                if site in insertions:
                    step = dict(insertions[site])
                    current_irrep = step["next_irrep"]
                else:
                    step = {
                        "site": site,
                        "kind": "dense",
                        "operator": middle[site],
                        "next_irrep": current_irrep,
                    }
                steps.append(step)
            return steps

        def reduced_string_product_steps(term):
            steps = reduced_string_steps(term)
            by_site = {step["site"]: step for step in steps}
            for site, operator in zip(term.dense_sites, term.dense_operators):
                site = int(site)
                if site in by_site:
                    step = by_site[site]
                    if step["kind"] != "dense":
                        raise ValueError("Dense scalar factors cannot overlap reduced insertion sites.")
                    step["operator"] = _compose_site_operators(step["operator"], operator)
                else:
                    by_site[site] = {
                        "site": site,
                        "kind": "dense",
                        "operator": operator,
                        "next_irrep": scalar,
                    }
            ordered = []
            current_irrep = scalar
            for site in sorted(by_site):
                step = dict(by_site[site])
                if step["kind"] == "dense":
                    step["next_irrep"] = current_irrep
                else:
                    current_irrep = step["next_irrep"]
                ordered.append(step)
            return ordered

        for term in self._terms:
            if isinstance(term, _ProductTerm):
                steps = product_steps(term)
            elif isinstance(term, _ReducedStringProductTerm):
                steps = reduced_string_product_steps(term)
            elif isinstance(term, _ReducedBilinearProductTerm):
                steps = reduced_bilinear_product_steps(term)
            elif isinstance(term, _ReducedBilinearTerm):
                steps = reduced_bilinear_steps(term)
            elif isinstance(term, _ReducedStringTerm):
                steps = reduced_string_steps(term)
            else:
                raise TypeError(f"Unsupported AutoMPO term type {type(term).__name__}.")
            insert_path(steps, term.coeff, family=getattr(term, "family", None))

        for site in range(self.nsites):
            add_identity_loop(site, start_state)
            add_identity_loop(site, final_state)

        nstates = len(state_irreps)
        transition_records = [[] for _ in range(self.nsites)]
        for site, phys_leg in enumerate(self.site_legs):
            ident = identity_operator(phys_leg, dtype=dtype)
            ident_signature = _site_operator_signature(ident)
            for state in identity_loops[site]:
                transition_records[site].append(
                    {
                        "kind": "dense",
                        "left": int(state),
                        "right": int(state),
                        "operator": ident,
                        "coeff": np.asarray(1.0, dtype=dtype),
                        "label": ("dense", ident_signature, np.asarray(1.0, dtype=dtype).item()),
                    }
                )
            for (left, right, signature, family), (coeff, operator) in dense_transitions[site].items():
                if abs(coeff) <= 0.0:
                    continue
                coeff = np.asarray(coeff, dtype=dtype).item()
                transition_records[site].append(
                    {
                        "kind": "dense",
                        "left": int(left),
                        "right": int(right),
                        "operator": operator,
                        "coeff": coeff,
                        "family": tuple(family),
                        "label": ("dense", signature, coeff, tuple(family)),
                    }
                )
            for (left, right, use_cg_coupling, signature, family), (coeff, operator) in reduced_transitions[site].items():
                if abs(coeff) <= 0.0:
                    continue
                coeff = np.asarray(coeff, dtype=dtype).item()
                transition_records[site].append(
                    {
                        "kind": "reduced",
                        "left": int(left),
                        "right": int(right),
                        "operator": operator,
                        "coeff": coeff,
                        "use_cg_coupling": bool(use_cg_coupling),
                        "family": tuple(family),
                        "label": ("reduced", signature, bool(use_cg_coupling), coeff, tuple(family)),
                    }
                )

        reachable = [set() for _ in range(self.nsites + 1)]
        reachable[0].add(start_state)
        for site in range(self.nsites):
            for record in transition_records[site]:
                if record["left"] in reachable[site]:
                    reachable[site + 1].add(record["right"])

        productive = [set() for _ in range(self.nsites + 1)]
        productive[self.nsites].add(final_state)
        for site in range(self.nsites - 1, -1, -1):
            for record in transition_records[site]:
                if record["right"] in productive[site + 1]:
                    productive[site].add(record["left"])

        active = [
            reachable[site] & productive[site]
            for site in range(self.nsites + 1)
        ]

        suffix_signatures = [None] * (self.nsites + 1)
        suffix_signatures[self.nsites] = {
            state: (state_irreps[state], "accept" if state == final_state else "reject")
            for state in range(nstates)
        }
        for site in range(self.nsites - 1, -1, -1):
            next_signatures = suffix_signatures[site + 1]
            by_left = {}
            for record in transition_records[site]:
                by_left.setdefault(record["left"], []).append(record)
            signatures = {}
            for state in range(nstates):
                outgoing = tuple(
                    sorted(
                        [
                            (
                                record["label"],
                                next_signatures[record["right"]],
                            )
                            for record in by_left.get(state, ())
                        ],
                        key=repr,
                        )
                )
                signatures[state] = (state_irreps[state], outgoing)
            suffix_signatures[site] = signatures

        class_maps = []
        class_irreps = []
        for site in range(self.nsites + 1):
            sig_to_index = {}
            state_to_index = {}
            irreps = []
            for state in sorted(active[site]):
                signature = suffix_signatures[site][state]
                index = sig_to_index.get(signature)
                if index is None:
                    index = len(sig_to_index)
                    sig_to_index[signature] = index
                    irreps.append(state_irreps[state])
                state_to_index[state] = index
            class_maps.append(state_to_index)
            class_irreps.append(tuple(irreps))

        force_rank_coupled_chain = any(
            irrep.dim != 1
            for irreps in class_irreps
            for irrep in irreps
        )

        mpo = []
        for site, phys_leg in enumerate(self.site_legs):
            left_map = class_maps[site]
            right_map = class_maps[site + 1]
            d_left = len(class_irreps[site])
            d_right = len(class_irreps[site + 1])
            visible_blocks = {}
            reduced_blocks = {}
            seen = set()

            for record in transition_records[site]:
                if record["left"] not in left_map or record["right"] not in right_map:
                    continue
                left = left_map[record["left"]]
                right = right_map[record["right"]]
                dedupe_key = (left, right, record["label"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                if record["kind"] == "dense":
                    _accumulate_site_operator(
                        visible_blocks,
                        record["operator"],
                        left,
                        right,
                        coeff=record["coeff"],
                        d_left=d_left,
                        d_right=d_right,
                        dtype=dtype,
                    )
                else:
                    key = (
                        _reduced_operator_signature(record["operator"]),
                        bool(record["use_cg_coupling"]),
                    )
                    item = reduced_blocks.get(key)
                    block = None if item is None else item[0]
                    if block is None:
                        block = np.zeros((d_left, d_right), dtype=dtype)
                        reduced_blocks[key] = (block, record["operator"])
                    block[left, right] += np.asarray(record["coeff"], dtype=dtype)

            symbolic_transitions = []
            for record in transition_records[site]:
                if record["left"] not in left_map or record["right"] not in right_map:
                    continue
                symbolic_transitions.append(
                    (
                        str(record["kind"]),
                        int(left_map[record["left"]]),
                        int(right_map[record["right"]]),
                        record["label"],
                    )
                )
            symbolic_transitions = tuple(symbolic_transitions)

            rank_coupled_terms = [
                RankCoupledChannelTerm(
                    reduced_operator=operator,
                    visible_virtual_block=block,
                    use_cg_coupling=use_cg_coupling,
                )
                for (_signature, use_cg_coupling), (block, operator) in reduced_blocks.items()
                if np.any(block)
            ]

            left_irreps = class_irreps[site]
            right_irreps = class_irreps[site + 1]
            needs_rank_coupled_core = force_rank_coupled_chain or rank_coupled_terms or any(
                irrep.dim != 1 for irrep in left_irreps + right_irreps
            )
            if needs_rank_coupled_core:
                mpo.append(
                    RankCoupledMPO(
                        dense_blocks=visible_blocks,
                        left_channel_irreps=left_irreps,
                        right_channel_irreps=right_irreps,
                        reduced_terms=tuple(rank_coupled_terms),
                        phys_out_leg=phys_leg,
                        phys_in_leg=phys_leg,
                        symbolic_transitions=symbolic_transitions,
                    )
                )
            else:
                mpo.append(
                    MPO(
                        blocks=visible_blocks,
                        phys_out_leg=phys_leg,
                        phys_in_leg=phys_leg,
                        symbolic_transitions=symbolic_transitions,
                    )
                )

        return mpo
