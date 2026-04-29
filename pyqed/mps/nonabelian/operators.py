#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reusable symmetry-aware local operators for non-Abelian tensor models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.mps.su2 import SpinChargeSector, SpatialOrbitalSite, SU2Irrep
from pyqed.mps.symmetry import Sector

from .builder import identity_operator
from .contraction import normalize_site_tensor_layout
from .coupling import clebsch_gordan, ordered_two_m_values
from .mpo import PhysicalLeg, SiteOperator
from .tensor import NonabelianTensor


def _canonical_spatial_leg():
    site = SpatialOrbitalSite()
    return PhysicalLeg.from_dims(
        {
            sector: len(indices)
            for sector, indices in zip(site.qn, site.state_index)
        },
        sectors=site.qn,
    )


def _extract_charge_spin(sector):
    if isinstance(sector, SpinChargeSector):
        return int(sector.charge), sector.irrep
    if isinstance(sector, Sector):
        if "charge" not in sector.labels or "su2" not in sector.labels:
            raise TypeError(f"Sector {sector!r} is not a charge x SU(2) sector.")
        charge = int(sector.components[sector.labels.index("charge")])
        irrep = sector.components[sector.labels.index("su2")]
        if not isinstance(irrep, SU2Irrep):
            raise TypeError(f"Sector {sector!r} has a non-SU(2) spin component {irrep!r}.")
        return charge, irrep
    raise TypeError(f"Unsupported charge-spin sector type {type(sector).__name__}.")


def _sector_scalar_factors(scalar_operator, leg, *, label):
    if not isinstance(scalar_operator, SiteOperator):
        raise TypeError(f"{label} expects a SiteOperator.")
    if scalar_operator.phys_out_leg != leg or scalar_operator.phys_in_leg != leg:
        raise ValueError(f"{label} requires a scalar operator on the requested physical leg.")
    sector_scales = {}
    for q_in in leg.sectors:
        for q_out in leg.sectors:
            block = scalar_operator.block(q_out, q_in)
            if q_out != q_in:
                if block is not None and np.any(block != 0):
                    raise ValueError(f"{label} only supports sector-diagonal operators.")
                continue
            if block is None:
                raise ValueError(f"Missing diagonal block for sector {q_in!r} in {label}.")
            eye = np.eye(block.shape[0], dtype=block.dtype)
            scale = block[0, 0]
            if not np.allclose(block, scale * eye):
                raise ValueError(f"{label} only supports identity-proportional diagonal blocks.")
            sector_scales[q_in] = np.asarray(scale)
    return sector_scales


@dataclass(frozen=True)
class ReducedTensorOperator:
    """
    Reduced local irreducible tensor operator on one spatial-orbital site.

    This lightweight container stores one reduced matrix element per allowed
    sector transition and expands a chosen spherical component back into a
    :class:`SiteOperator` using Clebsch-Gordan coefficients.

    The current implementation targets the canonical spatial-orbital basis and
    assumes every local sector multiplicity is one, i.e. the physical dimension
    of each sector matches the dimension of its SU(2) irrep.
    """

    reduced_blocks: dict[tuple[object, object], complex]
    phys_out_leg: PhysicalLeg
    phys_in_leg: PhysicalLeg
    rank_irrep: SU2Irrep
    component_phases: dict[int, complex] | None = None

    def __post_init__(self):
        if not isinstance(self.phys_out_leg, PhysicalLeg):
            raise TypeError("ReducedTensorOperator phys_out_leg must be a PhysicalLeg.")
        if not isinstance(self.phys_in_leg, PhysicalLeg):
            raise TypeError("ReducedTensorOperator phys_in_leg must be a PhysicalLeg.")
        if not isinstance(self.rank_irrep, SU2Irrep):
            raise TypeError("ReducedTensorOperator rank_irrep must be an SU2Irrep.")
        normalized_blocks = {}
        for (q_out, q_in), value in dict(self.reduced_blocks).items():
            if q_out not in self.phys_out_leg.sectors:
                raise ValueError(f"Undeclared output sector {q_out!r} in reduced operator.")
            if q_in not in self.phys_in_leg.sectors:
                raise ValueError(f"Undeclared input sector {q_in!r} in reduced operator.")
            _, out_irrep = _extract_charge_spin(q_out)
            _, in_irrep = _extract_charge_spin(q_in)
            if self.phys_out_leg.dim(q_out) != out_irrep.dim:
                raise NotImplementedError(
                    "ReducedTensorOperator currently requires multiplicity-one output sectors."
                )
            if self.phys_in_leg.dim(q_in) != in_irrep.dim:
                raise NotImplementedError(
                    "ReducedTensorOperator currently requires multiplicity-one input sectors."
                )
            normalized_blocks[(q_out, q_in)] = np.asarray(value)
        phases = {
            int(two_m): np.asarray(value)
            for two_m, value in dict(self.component_phases or {}).items()
        }
        for two_m in ordered_two_m_values(self.rank_irrep):
            phases.setdefault(int(two_m), np.asarray(1.0))
        object.__setattr__(self, "reduced_blocks", normalized_blocks)
        object.__setattr__(self, "component_phases", phases)

    @property
    def dtype(self):
        return np.result_type(
            *[value.dtype for value in self.reduced_blocks.values()],
            *[value.dtype for value in self.component_phases.values()],
        )

    @property
    def components(self):
        return ordered_two_m_values(self.rank_irrep)

    def component_block(self, two_m_component, q_out, q_in):
        two_m_component = int(two_m_component)
        if two_m_component not in self.component_phases:
            raise ValueError(
                f"Component 2m={two_m_component} is not allowed for rank {self.rank_irrep}."
            )
        reduced_value = self.reduced_blocks.get((q_out, q_in))
        if reduced_value is None:
            return None
        _, out_irrep = _extract_charge_spin(q_out)
        _, in_irrep = _extract_charge_spin(q_in)
        phase = self.component_phases[two_m_component]
        block = np.zeros((out_irrep.dim, in_irrep.dim), dtype=self.dtype)
        for row, two_m_out in enumerate(ordered_two_m_values(out_irrep)):
            for col, two_m_in in enumerate(ordered_two_m_values(in_irrep)):
                coeff = clebsch_gordan(
                    in_irrep,
                    self.rank_irrep,
                    out_irrep,
                    two_m_in,
                    two_m_component,
                    two_m_out,
                )
                if coeff:
                    block[row, col] = phase * reduced_value * coeff
        if np.any(block != 0):
            return block
        return None

    def component(self, two_m_component):
        two_m_component = int(two_m_component)
        if two_m_component not in self.component_phases:
            raise ValueError(
                f"Component 2m={two_m_component} is not allowed for rank {self.rank_irrep}."
            )
        blocks = {}
        for (q_out, q_in), reduced_value in self.reduced_blocks.items():
            out_charge, _ = _extract_charge_spin(q_out)
            in_charge, _ = _extract_charge_spin(q_in)
            charge_delta = out_charge - in_charge
            if abs(charge_delta) > 1:
                raise ValueError(
                    f"Unsupported reduced charge transition {q_in!r} -> {q_out!r}."
                )
            block = self.component_block(two_m_component, q_out, q_in)
            if block is not None:
                blocks[(q_out, q_in)] = block
        return SiteOperator(
            blocks=blocks,
            phys_out_leg=self.phys_out_leg,
            phys_in_leg=self.phys_in_leg,
        )

    def adjoint_component(self, two_m_component):
        component = self.component(two_m_component)
        return SiteOperator.from_dense(
            component.as_dense().T.conj(),
            phys_out_leg=self.phys_in_leg,
            phys_in_leg=self.phys_out_leg,
        )

    def right_multiply_sector_scalar(self, scalar_operator):
        sector_scales = _sector_scalar_factors(
            scalar_operator,
            self.phys_in_leg,
            label="right_multiply_sector_scalar",
        )
        return ReducedTensorOperator(
            reduced_blocks={
                (q_out, q_in): value * sector_scales[q_in]
                for (q_out, q_in), value in self.reduced_blocks.items()
            },
            phys_out_leg=self.phys_out_leg,
            phys_in_leg=self.phys_in_leg,
            rank_irrep=self.rank_irrep,
            component_phases=self.component_phases,
        )

    def adjoint(self):
        return AdjointReducedTensorOperator(self)


@dataclass(frozen=True)
class AdjointReducedTensorOperator:
    base_operator: ReducedTensorOperator
    input_sector_scales: dict[object, complex] | None = None

    @property
    def phys_out_leg(self):
        return self.base_operator.phys_in_leg

    @property
    def phys_in_leg(self):
        return self.base_operator.phys_out_leg

    @property
    def components(self):
        return self.base_operator.components

    @property
    def dtype(self):
        dtypes = [self.base_operator.dtype]
        if self.input_sector_scales:
            dtypes.extend(np.asarray(value).dtype for value in self.input_sector_scales.values())
        return np.result_type(*dtypes)

    def component_block(self, two_m_component, q_out, q_in):
        block = self.base_operator.adjoint_component(two_m_component).block(q_out, q_in)
        if block is None:
            return None
        scale = 1.0 if not self.input_sector_scales else self.input_sector_scales[q_in]
        return np.asarray(scale, dtype=self.dtype) * np.asarray(block, dtype=self.dtype)

    def right_multiply_sector_scalar(self, scalar_operator):
        sector_scales = _sector_scalar_factors(
            scalar_operator,
            self.phys_in_leg,
            label="right_multiply_sector_scalar",
        )
        combined = {}
        for sector in self.phys_in_leg.sectors:
            scale = sector_scales[sector]
            if self.input_sector_scales:
                scale = np.asarray(scale) * np.asarray(self.input_sector_scales.get(sector, 1.0))
            combined[sector] = scale
        return AdjointReducedTensorOperator(
            self.base_operator,
            input_sector_scales=combined,
        )


def physical_leg_from_spatial_orbital(site=None):
    """
    Build the canonical physical leg for one SU(2)-adapted spatial orbital.

    Parameters
    ----------
    site
        Optional source object describing the local spatial-orbital basis.
        Accepts ``None`` (canonical basis), :class:`SpatialOrbitalSite`,
        :class:`PhysicalLeg`, or a rank-3 :class:`NonabelianTensor` MPS site.
    """
    canonical_leg = _canonical_spatial_leg()
    if site is None:
        return canonical_leg
    if isinstance(site, SpatialOrbitalSite):
        return _canonical_spatial_leg()
    if isinstance(site, PhysicalLeg):
        if site != canonical_leg:
            raise ValueError(
                "physical_leg_from_spatial_orbital expects the canonical spatial-orbital PhysicalLeg."
            )
        return site
    if isinstance(site, NonabelianTensor):
        if site.rank != 3:
            raise ValueError(
                "physical_leg_from_spatial_orbital expects a rank-3 NonabelianTensor site tensor."
            )
        site = normalize_site_tensor_layout(site)
        dims = {}
        for key, block in site.data.items():
            sector = key[1]
            dim = int(block.shape[1])
            prev = dims.setdefault(sector, dim)
            if prev != dim:
                raise ValueError(
                    f"Inconsistent physical dimension for sector {sector!r}: {prev} vs {dim}."
                )
        leg = PhysicalLeg.from_dims(dims, sectors=tuple(dict.fromkeys(site.qns[1])))
        if leg != canonical_leg:
            raise ValueError(
                "physical_leg_from_spatial_orbital expects a site tensor in the canonical spatial-orbital basis."
            )
        return leg
    raise TypeError(
        "physical_leg_from_spatial_orbital expects None, SpatialOrbitalSite, PhysicalLeg, "
        "or a rank-3 NonabelianTensor."
    )


def _projector_weights(weights, *, phys_leg, dtype=float):
    blocks = {}
    for sector in phys_leg.sectors:
        blocks[(sector, sector)] = np.asarray(weights[sector], dtype=dtype) * np.eye(
            phys_leg.dim(sector), dtype=dtype
        )
    return SiteOperator(blocks=blocks, phys_out_leg=phys_leg, phys_in_leg=phys_leg)


def compose_site_operators(*operators):
    """
    Compose local operators in dense basis order.
    """
    if not operators:
        raise ValueError("compose_site_operators requires at least one operator.")
    first = operators[0]
    if not isinstance(first, SiteOperator):
        raise TypeError("compose_site_operators expects SiteOperator inputs.")
    phys_out_leg = first.phys_out_leg
    phys_in_leg = first.phys_in_leg
    dense = first.as_dense()
    for operator in operators[1:]:
        if not isinstance(operator, SiteOperator):
            raise TypeError("compose_site_operators expects SiteOperator inputs.")
        if operator.phys_out_leg != phys_out_leg or operator.phys_in_leg != phys_in_leg:
            raise ValueError("compose_site_operators requires matching physical legs.")
        dense = dense @ operator.as_dense()
    return SiteOperator.from_dense(dense, phys_out_leg=phys_out_leg, phys_in_leg=phys_in_leg)


def spatial_identity(site=None, *, dtype=float):
    return identity_operator(physical_leg_from_spatial_orbital(site), dtype=dtype)


def spatial_number(site=None, *, dtype=float):
    phys_leg = physical_leg_from_spatial_orbital(site)
    weights = {sector: float(sector.charge) for sector in phys_leg.sectors}
    return _projector_weights(weights, phys_leg=phys_leg, dtype=dtype)


def spatial_double_occupancy(site=None, *, dtype=float):
    phys_leg = physical_leg_from_spatial_orbital(site)
    weights = {sector: float(sector.charge == 2) for sector in phys_leg.sectors}
    return _projector_weights(weights, phys_leg=phys_leg, dtype=dtype)


def spatial_spin_square(site=None, *, dtype=float):
    phys_leg = physical_leg_from_spatial_orbital(site)
    weights = {
        sector: 0.25 * sector.two_j * (sector.two_j + 2)
        for sector in phys_leg.sectors
    }
    return _projector_weights(weights, phys_leg=phys_leg, dtype=dtype)


def spatial_projector(occupancy, site=None, *, dtype=float):
    """
    Project onto one occupancy sector of a spatial orbital.

    Parameters
    ----------
    occupancy
        One of ``"empty"``, ``"single"``, or ``"double"``.
    """
    if occupancy not in {"empty", "single", "double"}:
        raise ValueError(
            f"Unknown spatial occupancy projector {occupancy!r}; expected empty/single/double."
        )
    target_charge = {"empty": 0, "single": 1, "double": 2}[occupancy]
    phys_leg = physical_leg_from_spatial_orbital(site)
    weights = {sector: float(sector.charge == target_charge) for sector in phys_leg.sectors}
    return _projector_weights(weights, phys_leg=phys_leg, dtype=dtype)


def spatial_parity(site=None, *, dtype=float):
    """
    Fermionic parity operator ``(-1)^N`` in the spatial-orbital basis.
    """
    phys_leg = physical_leg_from_spatial_orbital(site)
    weights = {sector: float((-1) ** sector.charge) for sector in phys_leg.sectors}
    return _projector_weights(weights, phys_leg=phys_leg, dtype=dtype)


def reduced_spatial_fermion_annihilation(site=None, *, dtype=float):
    """
    Reduced rank-1/2 annihilation tensor on one spatial-orbital site.

    The two spherical components expand to the canonical local spin-resolved
    annihilation operators in the basis ``|empty>, |up>, |down>, |double>``.
    """
    phys_leg = physical_leg_from_spatial_orbital(site)
    q_empty, q_single, q_double = phys_leg.sectors
    return ReducedTensorOperator(
        reduced_blocks={
            (q_empty, q_single): np.asarray(np.sqrt(2.0), dtype=dtype),
            (q_single, q_double): np.asarray(1.0, dtype=dtype),
        },
        phys_out_leg=phys_leg,
        phys_in_leg=phys_leg,
        rank_irrep=SU2Irrep(1),
        component_phases={
            -1: np.asarray(1.0, dtype=dtype),
            1: np.asarray(-1.0, dtype=dtype),
        },
    )


def spatial_number_up(site=None, *, dtype=float):
    phys_leg = physical_leg_from_spatial_orbital(site)
    dense = np.diag([0.0, 1.0, 0.0, 1.0]).astype(dtype)
    return SiteOperator.from_dense(dense, phys_out_leg=phys_leg, phys_in_leg=phys_leg)


def spatial_number_down(site=None, *, dtype=float):
    phys_leg = physical_leg_from_spatial_orbital(site)
    dense = np.diag([0.0, 0.0, 1.0, 1.0]).astype(dtype)
    return SiteOperator.from_dense(dense, phys_out_leg=phys_leg, phys_in_leg=phys_leg)


def spatial_annihilate_up(site=None, *, dtype=float):
    """
    Spin-up annihilation operator in basis ``|empty>, |up>, |down>, |double>``.
    """
    return reduced_spatial_fermion_annihilation(site, dtype=dtype).component(-1)


def spatial_create_up(site=None, *, dtype=float):
    return reduced_spatial_fermion_annihilation(site, dtype=dtype).adjoint_component(-1)


def spatial_annihilate_down(site=None, *, dtype=float):
    """
    Spin-down annihilation operator with the correct intra-site fermionic sign.
    """
    return reduced_spatial_fermion_annihilation(site, dtype=dtype).component(1)


def spatial_create_down(site=None, *, dtype=float):
    return reduced_spatial_fermion_annihilation(site, dtype=dtype).adjoint_component(1)
