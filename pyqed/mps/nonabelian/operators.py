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
from .mpo import Leg, SiteOperator
from .tensor import NonabelianTensor


def _canonical_spatial_leg():
    site = SpatialOrbitalSite()
    return Leg.from_dims(
        {
            sector: len(indices)
            for sector, indices in zip(site.qn, site.state_index)
        },
        sectors=site.qn,
    )


def _fully_reduced_spatial_leg():
    site = SpatialOrbitalSite()
    return Leg.from_dims(
        {sector: 1 for sector in site.qn},
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


def _reduced_operator_rank(operator):
    if hasattr(operator, "rank_irrep"):
        return operator.rank_irrep
    if hasattr(operator, "base_operator"):
        return _reduced_operator_rank(operator.base_operator)
    raise TypeError(f"Object {type(operator).__name__} does not expose a reduced tensor rank.")


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
    phys_out_leg: Leg
    phys_in_leg: Leg
    rank_irrep: SU2Irrep
    component_phases: dict[int, complex] | None = None

    def __post_init__(self):
        if not isinstance(self.phys_out_leg, Leg):
            raise TypeError("ReducedTensorOperator phys_out_leg must be a Leg.")
        if not isinstance(self.phys_in_leg, Leg):
            raise TypeError("ReducedTensorOperator phys_in_leg must be a Leg.")
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
            out_dim = self.phys_out_leg.sector_dim(q_out)
            in_dim = self.phys_in_leg.sector_dim(q_in)
            if out_dim not in {1, out_irrep.dim}:
                raise NotImplementedError(
                    "ReducedTensorOperator currently requires canonical irrep dimensions or fully reduced output sectors."
                )
            if in_dim not in {1, in_irrep.dim}:
                raise NotImplementedError(
                    "ReducedTensorOperator currently requires canonical irrep dimensions or fully reduced input sectors."
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
        object.__setattr__(self, "_component_block_cache", {})
        object.__setattr__(self, "_component_cache", {})
        object.__setattr__(self, "_adjoint_component_cache", {})

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
        cache_key = (two_m_component, q_out, q_in)
        cache = self._component_block_cache
        if cache_key in cache:
            return cache[cache_key]
        if two_m_component not in self.component_phases:
            raise ValueError(
                f"Component 2m={two_m_component} is not allowed for rank {self.rank_irrep}."
            )
        reduced_value = self.reduced_blocks.get((q_out, q_in))
        if reduced_value is None:
            cache[cache_key] = None
            return None
        _, out_irrep = _extract_charge_spin(q_out)
        _, in_irrep = _extract_charge_spin(q_in)
        phase = self.component_phases[two_m_component]
        if self.phys_out_leg.sector_dim(q_out) == 1 and self.phys_in_leg.sector_dim(q_in) == 1:
            block = np.asarray([[phase * reduced_value]], dtype=self.dtype)
            cache[cache_key] = block
            return block
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
            cache[cache_key] = block
            return block
        cache[cache_key] = None
        return None

    def component(self, two_m_component):
        two_m_component = int(two_m_component)
        cache = self._component_cache
        if two_m_component in cache:
            return cache[two_m_component]
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
        operator = SiteOperator(
            blocks=blocks,
            phys_out_leg=self.phys_out_leg,
            phys_in_leg=self.phys_in_leg,
        )
        cache[two_m_component] = operator
        return operator

    def adjoint_component(self, two_m_component):
        two_m_component = int(two_m_component)
        cache = self._adjoint_component_cache
        if two_m_component in cache:
            return cache[two_m_component]
        component = self.component(two_m_component)
        operator = SiteOperator.from_dense(
            component.as_dense().T.conj(),
            phys_out_leg=self.phys_in_leg,
            phys_in_leg=self.phys_out_leg,
        )
        cache[two_m_component] = operator
        return operator

    def right_multiply_sector_scalar(self, scalar_operator):
        """
        Return this reduced tensor composed with a sector-diagonal scalar on
        its input leg.

        :param scalar_operator: Scalar :class:`SiteOperator` acting on the
            input physical leg.
        :returns: A reduced tensor with input-sector reduced matrix elements
            scaled by ``scalar_operator``.
        """
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

    def left_multiply_sector_scalar(self, scalar_operator):
        """
        Return a sector-diagonal scalar composed on this tensor's output leg.

        :param scalar_operator: Scalar :class:`SiteOperator` acting on the
            output physical leg.
        :returns: A reduced tensor with output-sector reduced matrix elements
            scaled by ``scalar_operator``.
        """
        sector_scales = _sector_scalar_factors(
            scalar_operator,
            self.phys_out_leg,
            label="left_multiply_sector_scalar",
        )
        return ReducedTensorOperator(
            reduced_blocks={
                (q_out, q_in): sector_scales[q_out] * value
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
    output_sector_scales: dict[object, complex] | None = None

    def __post_init__(self):
        object.__setattr__(self, "_component_block_cache", {})

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
    def rank_irrep(self):
        return self.base_operator.rank_irrep

    @property
    def dtype(self):
        dtypes = [self.base_operator.dtype]
        if self.input_sector_scales:
            dtypes.extend(np.asarray(value).dtype for value in self.input_sector_scales.values())
        if self.output_sector_scales:
            dtypes.extend(np.asarray(value).dtype for value in self.output_sector_scales.values())
        return np.result_type(*dtypes)

    def component_block(self, two_m_component, q_out, q_in):
        cache_key = (int(two_m_component), q_out, q_in)
        cache = self._component_block_cache
        if cache_key in cache:
            return cache[cache_key]
        block = self.base_operator.adjoint_component(two_m_component).block(q_out, q_in)
        if block is None:
            cache[cache_key] = None
            return None
        scale = 1.0
        if self.input_sector_scales:
            scale = np.asarray(scale) * np.asarray(self.input_sector_scales[q_in])
        if self.output_sector_scales:
            scale = np.asarray(scale) * np.asarray(self.output_sector_scales[q_out])
        out = np.asarray(scale, dtype=self.dtype) * np.asarray(block, dtype=self.dtype)
        cache[cache_key] = out
        return out

    def component(self, two_m_component):
        """
        Expand one spherical component into a block-sparse site operator.

        :param two_m_component: Doubled magnetic quantum number of the
            requested component.
        :returns: Component :class:`SiteOperator`.
        """
        blocks = {}
        for q_out in self.phys_out_leg.sectors:
            for q_in in self.phys_in_leg.sectors:
                block = self.component_block(two_m_component, q_out, q_in)
                if block is not None:
                    blocks[(q_out, q_in)] = block
        return SiteOperator(
            blocks=blocks,
            phys_out_leg=self.phys_out_leg,
            phys_in_leg=self.phys_in_leg,
        )

    def right_multiply_sector_scalar(self, scalar_operator):
        """
        Return this adjoint reduced tensor composed with a scalar on its input leg.

        :param scalar_operator: Sector-diagonal scalar operator.
        :returns: Scaled adjoint reduced tensor wrapper.
        """
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
            output_sector_scales=self.output_sector_scales,
        )

    def left_multiply_sector_scalar(self, scalar_operator):
        """
        Return this adjoint reduced tensor composed with a scalar on its output leg.

        :param scalar_operator: Sector-diagonal scalar operator.
        :returns: Scaled adjoint reduced tensor wrapper.
        """
        sector_scales = _sector_scalar_factors(
            scalar_operator,
            self.phys_out_leg,
            label="left_multiply_sector_scalar",
        )
        combined = {}
        for sector in self.phys_out_leg.sectors:
            scale = sector_scales[sector]
            if self.output_sector_scales:
                scale = np.asarray(scale) * np.asarray(self.output_sector_scales.get(sector, 1.0))
            combined[sector] = scale
        return AdjointReducedTensorOperator(
            self.base_operator,
            input_sector_scales=self.input_sector_scales,
            output_sector_scales=combined,
        )


@dataclass(frozen=True)
class TimeReversedReducedTensorOperator:
    """
    Time-reversed spherical tensor wrapper.

    For half-integer spinor fermion tensors this maps components as
    ``tilde T_q = (-1)^(j-q) T_{-q}``, with doubled component labels.  It is
    useful for Wigner-Eckart scalar contractions such as spin-summed
    ``c^dagger c`` bilinears.
    """

    base_operator: object

    def __post_init__(self):
        object.__setattr__(self, "_component_block_cache", {})

    @property
    def phys_out_leg(self):
        return self.base_operator.phys_out_leg

    @property
    def phys_in_leg(self):
        return self.base_operator.phys_in_leg

    @property
    def components(self):
        return self.base_operator.components

    @property
    def rank_irrep(self):
        if hasattr(self.base_operator, "rank_irrep"):
            return self.base_operator.rank_irrep
        return self.base_operator.base_operator.rank_irrep

    @property
    def dtype(self):
        return self.base_operator.dtype

    def component_block(self, two_m_component, q_out, q_in):
        two_m_component = int(two_m_component)
        cache_key = (two_m_component, q_out, q_in)
        cache = self._component_block_cache
        if cache_key in cache:
            return cache[cache_key]
        block = self.base_operator.component_block(-two_m_component, q_out, q_in)
        if block is None:
            cache[cache_key] = None
            return None
        exponent = (self.rank_irrep.two_j - two_m_component) // 2
        out = ((-1.0) ** exponent) * np.asarray(block, dtype=self.dtype)
        cache[cache_key] = out
        return out

    def component(self, two_m_component):
        """
        Expand one time-reversed spherical component into a site operator.

        :param two_m_component: Doubled magnetic quantum number of the
            requested component.
        :returns: Component :class:`SiteOperator`.
        """
        blocks = {}
        for q_out in self.phys_out_leg.sectors:
            for q_in in self.phys_in_leg.sectors:
                block = self.component_block(two_m_component, q_out, q_in)
                if block is not None:
                    blocks[(q_out, q_in)] = block
        return SiteOperator(
            blocks=blocks,
            phys_out_leg=self.phys_out_leg,
            phys_in_leg=self.phys_in_leg,
        )

    def right_multiply_sector_scalar(self, scalar_operator):
        return TimeReversedReducedTensorOperator(
            self.base_operator.right_multiply_sector_scalar(scalar_operator)
        )

    def left_multiply_sector_scalar(self, scalar_operator):
        """
        Return the time-reversed tensor with a scalar composed on its output leg.

        :param scalar_operator: Sector-diagonal scalar operator.
        :returns: Scaled time-reversed reduced tensor wrapper.
        """
        return TimeReversedReducedTensorOperator(
            self.base_operator.left_multiply_sector_scalar(scalar_operator)
        )


def time_reversed_reduced_operator(operator):
    """Return the time-reversed spherical tensor wrapper for ``operator``."""
    return TimeReversedReducedTensorOperator(operator)


@dataclass(frozen=True)
class CoupledReducedTensorProductOperator:
    """
    Component-block representation of a same-site coupled reduced product.

    This keeps the product rank and component labels explicit without forcing
    the result back through one global Wigner-Eckart phase convention.
    """

    component_blocks: dict[int, dict[tuple[object, object], np.ndarray]]
    phys_out_leg: Leg
    phys_in_leg: Leg
    rank_irrep: SU2Irrep

    def __post_init__(self):
        blocks = {}
        for component, by_sector in dict(self.component_blocks).items():
            component = int(component)
            if component not in ordered_two_m_values(self.rank_irrep):
                raise ValueError(
                    f"Component 2m={component} is not allowed for rank {self.rank_irrep}."
                )
            normalized = {}
            for (q_out, q_in), block in dict(by_sector).items():
                block = np.asarray(block)
                if q_out not in self.phys_out_leg.sectors:
                    raise ValueError(f"Undeclared output sector {q_out!r} in coupled product.")
                if q_in not in self.phys_in_leg.sectors:
                    raise ValueError(f"Undeclared input sector {q_in!r} in coupled product.")
                if block.shape != (self.phys_out_leg.sector_dim(q_out), self.phys_in_leg.sector_dim(q_in)):
                    raise ValueError(
                        f"Coupled product block {(q_out, q_in)!r} shape {block.shape!r} "
                        "does not match physical leg dimensions."
                    )
                normalized[(q_out, q_in)] = block
            blocks[component] = normalized
        for component in ordered_two_m_values(self.rank_irrep):
            blocks.setdefault(int(component), {})
        object.__setattr__(self, "component_blocks", blocks)

    @property
    def components(self):
        return ordered_two_m_values(self.rank_irrep)

    @property
    def dtype(self):
        dtypes = [
            block.dtype
            for by_sector in self.component_blocks.values()
            for block in by_sector.values()
        ]
        return np.result_type(*(dtypes or [float]))

    def component_block(self, two_m_component, q_out, q_in):
        return self.component_blocks.get(int(two_m_component), {}).get((q_out, q_in))

    def component(self, two_m_component):
        blocks = dict(self.component_blocks.get(int(two_m_component), {}))
        return SiteOperator(
            blocks=blocks,
            phys_out_leg=self.phys_out_leg,
            phys_in_leg=self.phys_in_leg,
        )

    def right_multiply_sector_scalar(self, scalar_operator):
        """
        Compose a sector-diagonal scalar on the product input leg.

        :param scalar_operator: Sector-diagonal scalar site operator.
        :returns: Scaled coupled product operator.
        """
        sector_scales = _sector_scalar_factors(
            scalar_operator,
            self.phys_in_leg,
            label="right_multiply_sector_scalar",
        )
        return CoupledReducedTensorProductOperator(
            component_blocks={
                component: {
                    (q_out, q_in): np.asarray(block) * sector_scales[q_in]
                    for (q_out, q_in), block in by_sector.items()
                }
                for component, by_sector in self.component_blocks.items()
            },
            phys_out_leg=self.phys_out_leg,
            phys_in_leg=self.phys_in_leg,
            rank_irrep=self.rank_irrep,
        )

    def left_multiply_sector_scalar(self, scalar_operator):
        """
        Compose a sector-diagonal scalar on the product output leg.

        :param scalar_operator: Sector-diagonal scalar site operator.
        :returns: Scaled coupled product operator.
        """
        sector_scales = _sector_scalar_factors(
            scalar_operator,
            self.phys_out_leg,
            label="left_multiply_sector_scalar",
        )
        return CoupledReducedTensorProductOperator(
            component_blocks={
                component: {
                    (q_out, q_in): sector_scales[q_out] * np.asarray(block)
                    for (q_out, q_in), block in by_sector.items()
                }
                for component, by_sector in self.component_blocks.items()
            },
            phys_out_leg=self.phys_out_leg,
            phys_in_leg=self.phys_in_leg,
            rank_irrep=self.rank_irrep,
        )


def coupled_reduced_tensor_product(left_operator, right_operator, rank_irrep, *, tol=1.0e-12):
    """
    Couple two same-site reduced tensors into one irreducible tensor.

    The returned tensor represents

    .. math::

        [A^{j_a} B^{j_b}]^J_M =
        \\sum_{m_a m_b} \\langle j_a m_a, j_b m_b | J M \\rangle
        A_{m_a} B_{m_b}

    with dense operator composition order ``A @ B``.

    :param left_operator: Left local reduced tensor ``A``.
    :param right_operator: Right local reduced tensor ``B``.
    :param rank_irrep: Target coupled SU(2) irrep ``J``.
    :param tol: Numerical tolerance for reduced matrix-element fitting.
    :returns: A :class:`CoupledReducedTensorProductOperator` for the product.
    """
    if not isinstance(rank_irrep, SU2Irrep):
        raise TypeError("rank_irrep must be an SU2Irrep.")
    if getattr(left_operator, "phys_in_leg", None) != getattr(right_operator, "phys_out_leg", None):
        raise ValueError("Coupled reduced products require matching middle physical legs.")
    phys_out_leg = left_operator.phys_out_leg
    phys_mid_leg = left_operator.phys_in_leg
    phys_in_leg = right_operator.phys_in_leg
    left_rank = _reduced_operator_rank(left_operator)
    right_rank = _reduced_operator_rank(right_operator)
    dtype = np.result_type(
        getattr(left_operator, "dtype", float),
        getattr(right_operator, "dtype", float),
        float,
    )
    component_blocks = {}
    for two_m in ordered_two_m_values(rank_irrep):
        blocks = {}
        for q_out in phys_out_leg.sectors:
            for q_in in phys_in_leg.sectors:
                block = np.zeros(
                    (phys_out_leg.sector_dim(q_out), phys_in_leg.sector_dim(q_in)),
                    dtype=dtype,
                )
                for q_mid in phys_mid_leg.sectors:
                    for left_m in left_operator.components:
                        left_block = left_operator.component_block(left_m, q_out, q_mid)
                        if left_block is None:
                            continue
                        for right_m in right_operator.components:
                            if int(left_m) + int(right_m) != int(two_m):
                                continue
                            cg = clebsch_gordan(
                                left_rank,
                                right_rank,
                                rank_irrep,
                                int(left_m),
                                int(right_m),
                                int(two_m),
                            )
                            if not cg:
                                continue
                            right_block = right_operator.component_block(right_m, q_mid, q_in)
                            if right_block is None:
                                continue
                            block += cg * np.asarray(left_block, dtype=dtype) @ np.asarray(
                                right_block,
                                dtype=dtype,
                            )
                if np.linalg.norm(block.reshape(-1)) > tol:
                    blocks[(q_out, q_in)] = block
        component_blocks[int(two_m)] = blocks

    if not any(component_blocks.values()):
        raise ValueError("Coupled reduced tensor product is numerically zero.")
    return CoupledReducedTensorProductOperator(
        component_blocks=component_blocks,
        phys_out_leg=phys_out_leg,
        phys_in_leg=phys_in_leg,
        rank_irrep=rank_irrep,
    )


def physical_leg_from_spatial_orbital(site=None):
    """
    Build the canonical physical leg for one SU(2)-adapted spatial orbital.

    Parameters
    ----------
    site
        Optional source object describing the local spatial-orbital basis.
        Accepts ``None`` (canonical basis), :class:`SpatialOrbitalSite`,
        :class:`Leg`, or a rank-3 :class:`NonabelianTensor` MPS site.
    """
    canonical_leg = _canonical_spatial_leg()
    reduced_leg = _fully_reduced_spatial_leg()
    if site is None:
        return canonical_leg
    if site.__class__.__name__ == "FullyReducedSpatialOrbitalSite":
        return reduced_leg
    if isinstance(site, SpatialOrbitalSite):
        return _canonical_spatial_leg()
    if isinstance(site, Leg):
        if site == canonical_leg or site == reduced_leg:
            return site
        if site != canonical_leg:
            raise ValueError(
                "physical_leg_from_spatial_orbital expects a canonical or fully reduced spatial-orbital Leg."
            )
        return site
    if isinstance(site, NonabelianTensor):
        if site.rank != 3:
            raise ValueError(
                "physical_leg_from_spatial_orbital expects a rank-3 NonabelianTensor site tensor."
            )
        site = normalize_site_tensor_layout(site)
        if site.metadata.get("physical_basis") == "fully_reduced_su2":
            return reduced_leg
        dims = {}
        for key, block in site.data.items():
            sector = key[1]
            dim = int(block.shape[1])
            prev = dims.setdefault(sector, dim)
            if prev != dim:
                raise ValueError(
                    f"Inconsistent physical dimension for sector {sector!r}: {prev} vs {dim}."
                )
        leg = Leg.from_dims(dims, sectors=tuple(dict.fromkeys(site.qns[1])))
        if leg == canonical_leg or leg == reduced_leg:
            return leg
        if leg != canonical_leg:
            raise ValueError(
                "physical_leg_from_spatial_orbital expects a site tensor in the canonical or fully reduced spatial-orbital basis."
            )
        return leg
    raise TypeError(
        "physical_leg_from_spatial_orbital expects None, SpatialOrbitalSite, Leg, "
        "or a rank-3 NonabelianTensor."
    )


def _projector_weights(weights, *, phys_leg, dtype=float):
    blocks = {}
    for sector in phys_leg.sectors:
        blocks[(sector, sector)] = np.asarray(weights[sector], dtype=dtype) * np.eye(
            phys_leg.sector_dim(sector), dtype=dtype
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


def spatial_pair_creation(site=None, *, dtype=float):
    """
    Return the local singlet pair-creation operator.

    :param site: Optional spatial-orbital site descriptor or physical leg.
    :param dtype: Numeric dtype for stored blocks.
    :returns: A scalar :class:`SiteOperator` mapping the empty sector to the
        double-occupancy sector.
    """
    phys_leg = physical_leg_from_spatial_orbital(site)
    q_empty, _q_single, q_double = phys_leg.sectors
    return SiteOperator(
        blocks={(q_double, q_empty): np.asarray([[1.0]], dtype=dtype)},
        phys_out_leg=phys_leg,
        phys_in_leg=phys_leg,
    )


def spatial_pair_annihilation(site=None, *, dtype=float):
    """
    Return the local singlet pair-annihilation operator.

    :param site: Optional spatial-orbital site descriptor or physical leg.
    :param dtype: Numeric dtype for stored blocks.
    :returns: A scalar :class:`SiteOperator` mapping the double-occupancy sector
        to the empty sector.
    """
    phys_leg = physical_leg_from_spatial_orbital(site)
    q_empty, _q_single, q_double = phys_leg.sectors
    return SiteOperator(
        blocks={(q_empty, q_double): np.asarray([[1.0]], dtype=dtype)},
        phys_out_leg=phys_leg,
        phys_in_leg=phys_leg,
    )


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
    empty_single_value = (
        1.0 / np.sqrt(2.0)
        if phys_leg.sector_dim(q_single) == 1
        else np.sqrt(2.0)
    )
    return ReducedTensorOperator(
        reduced_blocks={
            (q_empty, q_single): np.asarray(empty_single_value, dtype=dtype),
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
