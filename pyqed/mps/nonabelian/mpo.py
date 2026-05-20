#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symmetry-aware MPO core containers for the fixed-layout non-Abelian prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyqed.mps.su2 import SU2Irrep
from pyqed.mps.symmetry import Sector

from .coupling import clebsch_gordan, left_or_right_fusion, ordered_two_m_values


@dataclass(frozen=True)
class PhysicalLeg:
    """
    Ordered physical-leg sector metadata for an MPO core.
    """

    sectors: tuple[Sector, ...]
    dims: dict[Sector, int]

    def __post_init__(self):
        sectors = tuple(self.sectors)
        dims = {sector: int(dim) for sector, dim in dict(self.dims).items()}
        if not sectors:
            raise ValueError("PhysicalLeg requires at least one sector.")
        for sector in sectors:
            if sector not in dims:
                raise ValueError(f"Missing physical dimension for sector {sector!r}.")
            if dims[sector] <= 0:
                raise ValueError(f"Physical dimension for sector {sector!r} must be positive.")
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "dims", dims)

    @property
    def total_dim(self):
        return sum(self.dims[sector] for sector in self.sectors)

    def dim(self, sector):
        return self.dims[sector]

    def slices(self):
        offset = 0
        out = {}
        for sector in self.sectors:
            if sector in out:
                continue
            dim = self.dims[sector]
            out[sector] = slice(offset, offset + dim)
            offset += dim
        return out

    @classmethod
    def from_slices(cls, sector_slices):
        return cls(
            sectors=tuple(sector_slices.keys()),
            dims={
                sector: int(slice_.stop - slice_.start)
                for sector, slice_ in sector_slices.items()
            },
        )

    @classmethod
    def from_dims(cls, sector_dims, sectors=None):
        if sectors is None:
            sectors = tuple(sector_dims.keys())
        return cls(
            sectors=tuple(sectors),
            dims={sector: int(sector_dims[sector]) for sector in sectors},
        )


@dataclass(frozen=True)
class SiteOperator:
    """
    Symmetry-aware local physical operator blocks.

    Parameters
    ----------
    blocks
        Mapping ``(phys_out_sector, phys_in_sector) -> dense matrix`` with each
        stored block shaped ``(dim_out, dim_in)``.
    phys_out_leg, phys_in_leg
        Explicit physical bra/ket leg metadata.
    """

    blocks: dict[tuple[Sector, Sector], np.ndarray]
    phys_out_leg: PhysicalLeg
    phys_in_leg: PhysicalLeg

    def __post_init__(self):
        if not isinstance(self.phys_out_leg, PhysicalLeg):
            raise TypeError("SiteOperator phys_out_leg must be a PhysicalLeg.")
        if not isinstance(self.phys_in_leg, PhysicalLeg):
            raise TypeError("SiteOperator phys_in_leg must be a PhysicalLeg.")
        blocks = {
            (q_out, q_in): np.asarray(block)
            for (q_out, q_in), block in self.blocks.items()
        }
        if not blocks:
            raise ValueError("SiteOperator requires at least one stored block.")
        for (q_out, q_in), block in blocks.items():
            if block.ndim != 2:
                raise ValueError(
                    f"Site-operator block {(q_out, q_in)!r} must be rank-2, got {block.shape!r}."
                )
            if q_out not in self.phys_out_leg.sectors:
                raise ValueError(f"Undeclared output sector {q_out!r} in site operator.")
            if q_in not in self.phys_in_leg.sectors:
                raise ValueError(f"Undeclared input sector {q_in!r} in site operator.")
            if int(block.shape[0]) != self.phys_out_leg.dim(q_out):
                raise ValueError(
                    f"Site-operator block {(q_out, q_in)!r} output dimension {block.shape[0]} "
                    f"does not match declared sector dimension {self.phys_out_leg.dim(q_out)}."
                )
            if int(block.shape[1]) != self.phys_in_leg.dim(q_in):
                raise ValueError(
                    f"Site-operator block {(q_out, q_in)!r} input dimension {block.shape[1]} "
                    f"does not match declared sector dimension {self.phys_in_leg.dim(q_in)}."
                )
        object.__setattr__(self, "blocks", blocks)

    @property
    def dtype(self):
        return np.result_type(*(block.dtype for block in self.blocks.values()))

    def block(self, phys_out, phys_in):
        return self.blocks.get((phys_out, phys_in))

    def as_dense(self, phys_out_slices=None, phys_in_slices=None):
        if phys_out_slices is None:
            phys_out_slices = self.phys_out_leg.slices()
        if phys_in_slices is None:
            phys_in_slices = self.phys_in_leg.slices()
        dense = np.zeros(
            (self.phys_out_leg.total_dim, self.phys_in_leg.total_dim),
            dtype=self.dtype,
        )
        for (q_out, q_in), block in self.blocks.items():
            dense[phys_out_slices[q_out], phys_in_slices[q_in]] = block
        return dense

    @classmethod
    def from_dense(
        cls,
        operator,
        *,
        phys_out_slices=None,
        phys_in_slices=None,
        phys_out_dims=None,
        phys_in_dims=None,
        phys_out_leg=None,
        phys_in_leg=None,
        tol=0.0,
    ):
        dense = np.asarray(operator)
        if dense.ndim != 2:
            raise ValueError(
                f"SiteOperator.from_dense expects a rank-2 operator, got {dense.shape!r}."
            )
        if phys_out_leg is None:
            if phys_out_dims is not None:
                phys_out_leg = PhysicalLeg.from_dims(phys_out_dims)
            elif phys_out_slices is not None:
                phys_out_leg = PhysicalLeg.from_slices(phys_out_slices)
            else:
                raise ValueError(
                    "from_dense requires phys_out_leg, phys_out_dims, or phys_out_slices."
                )
        if phys_in_leg is None:
            if phys_in_dims is not None:
                phys_in_leg = PhysicalLeg.from_dims(phys_in_dims)
            elif phys_in_slices is not None:
                phys_in_leg = PhysicalLeg.from_slices(phys_in_slices)
            else:
                phys_in_leg = phys_out_leg
        if phys_out_slices is None:
            phys_out_slices = phys_out_leg.slices()
        if phys_in_slices is None:
            phys_in_slices = phys_in_leg.slices()

        blocks = {}
        for q_out, p_out in phys_out_slices.items():
            for q_in, p_in in phys_in_slices.items():
                block = np.asarray(dense[p_out, p_in])
                if tol > 0.0:
                    keep = np.linalg.norm(block.ravel()) > tol
                else:
                    keep = np.any(block != 0)
                if keep:
                    blocks[(q_out, q_in)] = np.array(block, copy=True)

        if not blocks:
            q_out = next(iter(phys_out_leg.sectors))
            q_in = next(iter(phys_in_leg.sectors))
            blocks[(q_out, q_in)] = np.zeros(
                (phys_out_leg.dim(q_out), phys_in_leg.dim(q_in)),
                dtype=dense.dtype,
            )

        return cls(
            blocks=blocks,
            phys_out_leg=phys_out_leg,
            phys_in_leg=phys_in_leg,
        )


@dataclass(frozen=True)
class MPO:
    """
    MPO core with sector-keyed physical blocks.

    Parameters
    ----------
    blocks
        Mapping ``(phys_out_sector, phys_in_sector) -> dense block`` with
        each stored block shaped ``(w_left, w_right, dim_out, dim_in)``.
    phys_out_leg, phys_in_leg
        Declared physical bra/ket leg metadata in the site-local ordering.
    """

    blocks: dict[tuple[Sector, Sector], np.ndarray]
    phys_out_leg: PhysicalLeg
    phys_in_leg: PhysicalLeg
    symbolic_transitions: tuple = ()

    def __post_init__(self):
        phys_out_leg = self.phys_out_leg
        phys_in_leg = self.phys_in_leg
        if not isinstance(phys_out_leg, PhysicalLeg):
            raise TypeError("MPO phys_out_leg must be a PhysicalLeg.")
        if not isinstance(phys_in_leg, PhysicalLeg):
            raise TypeError("MPO phys_in_leg must be a PhysicalLeg.")
        blocks = {
            (q_out, q_in): np.asarray(block)
            for (q_out, q_in), block in self.blocks.items()
        }
        if not blocks:
            raise ValueError("MPO requires at least one stored block.")

        left_dim = None
        right_dim = None
        for key, block in blocks.items():
            if len(key) != 2:
                raise ValueError(f"Invalid MPO block key {key!r}; expected (phys_out, phys_in).")
            if block.ndim != 4:
                raise ValueError(
                    f"MPO block {key!r} must be rank-4, got shape {block.shape!r}."
                )
            q_out, q_in = key
            if q_out not in phys_out_leg.sectors:
                raise ValueError(
                    f"MPO block uses undeclared output sector {q_out!r}."
                )
            if q_in not in phys_in_leg.sectors:
                raise ValueError(
                    f"MPO block uses undeclared input sector {q_in!r}."
                )
            if int(block.shape[2]) != phys_out_leg.dim(q_out):
                raise ValueError(
                    f"MPO block {key!r} output dimension {block.shape[2]} does not match "
                    f"declared sector dimension {phys_out_leg.dim(q_out)}."
                )
            if int(block.shape[3]) != phys_in_leg.dim(q_in):
                raise ValueError(
                    f"MPO block {key!r} input dimension {block.shape[3]} does not match "
                    f"declared sector dimension {phys_in_leg.dim(q_in)}."
                )
            if left_dim is None:
                left_dim = int(block.shape[0])
                right_dim = int(block.shape[1])
            elif int(block.shape[0]) != left_dim or int(block.shape[1]) != right_dim:
                raise ValueError("All MPO blocks must share the same virtual dimensions.")

        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "phys_out_leg", phys_out_leg)
        object.__setattr__(self, "phys_in_leg", phys_in_leg)
        object.__setattr__(self, "symbolic_transitions", tuple(self.symbolic_transitions))

    @property
    def phys_out_sectors(self):
        return self.phys_out_leg.sectors

    @property
    def phys_in_sectors(self):
        return self.phys_in_leg.sectors

    @property
    def left_dim(self):
        return next(iter(self.blocks.values())).shape[0]

    @property
    def right_dim(self):
        return next(iter(self.blocks.values())).shape[1]

    @property
    def dtype(self):
        return np.result_type(*(block.dtype for block in self.blocks.values()))

    def block(self, phys_out, phys_in):
        return self.blocks.get((phys_out, phys_in))

    def as_dense(self, phys_out_slices=None, phys_in_slices=None):
        """
        Expand the block-sparse core into a dense ``(wL, wR, pOut, pIn)`` array.
        """
        if phys_in_slices is None:
            phys_in_slices = self.phys_in_leg.slices()
        if phys_out_slices is None:
            phys_out_slices = self.phys_out_leg.slices()
        p_out_dim = max(slice_.stop for slice_ in phys_out_slices.values())
        p_in_dim = max(slice_.stop for slice_ in phys_in_slices.values())
        dense = np.zeros((self.left_dim, self.right_dim, p_out_dim, p_in_dim), dtype=self.dtype)
        for (q_out, q_in), block in self.blocks.items():
            dense[:, :, phys_out_slices[q_out], phys_in_slices[q_in]] = block
        return dense

    @classmethod
    def from_dense(
        cls,
        core,
        *,
        phys_out_slices=None,
        phys_in_slices=None,
        phys_out_dims=None,
        phys_in_dims=None,
        phys_out_leg=None,
        phys_in_leg=None,
        tol=0.0,
    ):
        """
        Build a block-sparse MPO core from a dense physical-tensor core.
        """
        dense = np.asarray(core)
        if dense.ndim != 4:
            raise ValueError(
                f"MPO.from_dense expects a rank-4 core, got {dense.shape!r}."
            )
        if phys_out_leg is None:
            if phys_out_dims is not None:
                phys_out_leg = PhysicalLeg.from_dims(phys_out_dims)
            elif phys_out_slices is not None:
                phys_out_leg = PhysicalLeg.from_slices(phys_out_slices)
            else:
                raise ValueError("from_dense requires phys_out_leg, phys_out_dims, or phys_out_slices.")
        if phys_in_leg is None:
            if phys_in_dims is not None:
                phys_in_leg = PhysicalLeg.from_dims(phys_in_dims)
            elif phys_in_slices is not None:
                phys_in_leg = PhysicalLeg.from_slices(phys_in_slices)
            else:
                phys_in_leg = phys_out_leg
        if phys_out_slices is None:
            phys_out_slices = phys_out_leg.slices()
        if phys_in_slices is None:
            phys_in_slices = phys_in_leg.slices()

        blocks = {}
        for q_out, p_out in phys_out_slices.items():
            for q_in, p_in in phys_in_slices.items():
                block = np.asarray(dense[:, :, p_out, p_in])
                if tol > 0.0:
                    keep = np.linalg.norm(block.ravel()) > tol
                else:
                    keep = np.any(block != 0)
                if keep:
                    blocks[(q_out, q_in)] = np.array(block, copy=True)

        if not blocks:
            blocks[(next(iter(phys_out_slices)), next(iter(phys_in_slices)))] = np.zeros(
                (
                    dense.shape[0],
                    dense.shape[1],
                    phys_out_slices[next(iter(phys_out_slices))].stop
                    - phys_out_slices[next(iter(phys_out_slices))].start,
                    phys_in_slices[next(iter(phys_in_slices))].stop
                    - phys_in_slices[next(iter(phys_in_slices))].start,
                ),
                dtype=dense.dtype,
            )

        return cls(
            blocks=blocks,
            phys_out_leg=phys_out_leg,
            phys_in_leg=phys_in_leg,
        )

    @classmethod
    def from_site_operator(
        cls,
        site_operator,
        *,
        virtual_block=None,
        virtual_blocks=None,
    ):
        """
        Build an MPO core directly from a symmetry-aware local site operator.

        Parameters
        ----------
        site_operator
            :class:`SiteOperator` carrying the physical operator
            blocks.
        virtual_block
            Shared dense virtual coefficient block with shape ``(wL, wR)``.
            Defaults to ``[[1.0]]``.
        virtual_blocks
            Optional mapping ``(phys_out_sector, phys_in_sector) -> (wL, wR)``
            for transition-specific coefficients. When given, it overrides
            ``virtual_block`` for matching transitions.
        """
        if not isinstance(site_operator, SiteOperator):
            raise TypeError("from_site_operator expects a SiteOperator.")
        if virtual_block is None:
            virtual_block = np.ones((1, 1), dtype=site_operator.dtype)
        shared_virtual = np.asarray(virtual_block)
        if shared_virtual.ndim != 2:
            raise ValueError(
                f"virtual_block must be rank-2, got shape {shared_virtual.shape!r}."
            )
        virtual_blocks = {
            key: np.asarray(block) for key, block in (virtual_blocks or {}).items()
        }

        blocks = {}
        for key, op_block in site_operator.blocks.items():
            coeff = virtual_blocks.get(key, shared_virtual)
            if coeff.ndim != 2:
                raise ValueError(
                    f"Virtual coefficient block for {key!r} must be rank-2, got {coeff.shape!r}."
                )
            blocks[key] = coeff[:, :, None, None] * np.asarray(op_block)[None, None, :, :]

        return cls(
            blocks=blocks,
            phys_out_leg=site_operator.phys_out_leg,
            phys_in_leg=site_operator.phys_in_leg,
        )


@dataclass(frozen=True)
class IrreducibleChannelTerm:
    """
    One irreducible tensor-operator contribution carried by an MPO core.

    The virtual coefficient matrices are keyed by spherical-component labels in
    doubled-``m`` notation. Distinct components may connect different virtual
    subchannels, which lets the MPO contract componentwise without storing the
    fully expanded physical blocks.
    """

    reduced_operator: object
    component_virtual_blocks: dict[int, np.ndarray]
    _component_cache: dict[tuple[int, Sector, Sector], np.ndarray | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        reduced_operator = self.reduced_operator
        if getattr(reduced_operator, "phys_out_leg", None) is None or getattr(
            reduced_operator, "phys_in_leg", None
        ) is None:
            raise TypeError("IrreducibleChannelTerm expects a reduced operator with physical legs.")
        if not hasattr(reduced_operator, "components") or not hasattr(
            reduced_operator, "component_block"
        ):
            raise TypeError(
                "IrreducibleChannelTerm expects a reduced operator exposing components and component_block()."
            )
        blocks = {
            int(component): np.asarray(block)
            for component, block in dict(self.component_virtual_blocks).items()
        }
        if not blocks:
            raise ValueError("IrreducibleChannelTerm requires at least one component block.")
        left_dim = None
        right_dim = None
        allowed = {int(component) for component in reduced_operator.components}
        for component, block in blocks.items():
            if component not in allowed:
                raise ValueError(
                    f"Component 2m={component} is not allowed for reduced operator {reduced_operator!r}."
                )
            if block.ndim != 2:
                raise ValueError(
                    f"Virtual block for component 2m={component} must be rank-2, got {block.shape!r}."
                )
            if left_dim is None:
                left_dim = int(block.shape[0])
                right_dim = int(block.shape[1])
            elif int(block.shape[0]) != left_dim or int(block.shape[1]) != right_dim:
                raise ValueError("All component virtual blocks must share the same virtual dimensions.")
        object.__setattr__(self, "component_virtual_blocks", blocks)

    @property
    def phys_out_leg(self):
        return self.reduced_operator.phys_out_leg

    @property
    def phys_in_leg(self):
        return self.reduced_operator.phys_in_leg

    @property
    def left_dim(self):
        return next(iter(self.component_virtual_blocks.values())).shape[0]

    @property
    def right_dim(self):
        return next(iter(self.component_virtual_blocks.values())).shape[1]

    @property
    def dtype(self):
        return np.result_type(
            *[block.dtype for block in self.component_virtual_blocks.values()],
            getattr(self.reduced_operator, "dtype", float),
        )

    def block(self, phys_out, phys_in):
        total = None
        for component, virtual_block in self.component_virtual_blocks.items():
            cache_key = (int(component), phys_out, phys_in)
            op_block = self._component_cache.get(cache_key)
            if cache_key not in self._component_cache:
                op_block = self.reduced_operator.component_block(component, phys_out, phys_in)
                self._component_cache[cache_key] = op_block
            if op_block is None:
                continue
            contrib = np.asarray(virtual_block)[:, :, None, None] * np.asarray(op_block)[None, None, :, :]
            if total is None:
                total = contrib
            else:
                total = total + contrib
        return total


@dataclass(frozen=True)
class IrreducibleMPO:
    """
    MPO core that mixes ordinary scalar physical blocks with irreducible channels.

    Physical blocks are expanded lazily per sector pair through the attached
    reduced tensor operators, so the core can carry reduced operator metadata
    directly instead of storing all expanded ``(p_out, p_in)`` arrays up front.
    """

    phys_out_leg: PhysicalLeg
    phys_in_leg: PhysicalLeg
    scalar_blocks: dict[tuple[Sector, Sector], np.ndarray] | None = None
    reduced_terms: tuple[IrreducibleChannelTerm, ...] = ()
    symbolic_transitions: tuple = ()
    _block_cache: dict[tuple[Sector, Sector], np.ndarray | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if not isinstance(self.phys_out_leg, PhysicalLeg):
            raise TypeError("IrreducibleMPO phys_out_leg must be a PhysicalLeg.")
        if not isinstance(self.phys_in_leg, PhysicalLeg):
            raise TypeError("IrreducibleMPO phys_in_leg must be a PhysicalLeg.")
        scalar_blocks = {
            key: np.asarray(block)
            for key, block in dict(self.scalar_blocks or {}).items()
        }
        reduced_terms = tuple(self.reduced_terms)
        if not scalar_blocks and not reduced_terms:
            raise ValueError("IrreducibleMPO requires scalar blocks and/or reduced terms.")
        left_dim = None
        right_dim = None
        for key, block in scalar_blocks.items():
            if len(key) != 2:
                raise ValueError(f"Invalid IrreducibleMPO scalar block key {key!r}.")
            q_out, q_in = key
            if q_out not in self.phys_out_leg.sectors or q_in not in self.phys_in_leg.sectors:
                raise ValueError(f"Undeclared sector pair {key!r} in IrreducibleMPO.")
            if block.ndim != 4:
                raise ValueError(
                    f"IrreducibleMPO scalar block {key!r} must be rank-4, got {block.shape!r}."
                )
            if int(block.shape[2]) != self.phys_out_leg.dim(q_out):
                raise ValueError(
                    f"Scalar block {key!r} output dimension {block.shape[2]} does not match declared dimension."
                )
            if int(block.shape[3]) != self.phys_in_leg.dim(q_in):
                raise ValueError(
                    f"Scalar block {key!r} input dimension {block.shape[3]} does not match declared dimension."
                )
            if left_dim is None:
                left_dim = int(block.shape[0])
                right_dim = int(block.shape[1])
            elif int(block.shape[0]) != left_dim or int(block.shape[1]) != right_dim:
                raise ValueError("All IrreducibleMPO contributions must share the same virtual dimensions.")
        for term in reduced_terms:
            if not isinstance(term, IrreducibleChannelTerm):
                raise TypeError("IrreducibleMPO reduced_terms must contain IrreducibleChannelTerm objects.")
            if term.phys_out_leg != self.phys_out_leg or term.phys_in_leg != self.phys_in_leg:
                raise ValueError("IrreducibleMPO reduced term physical legs must match the core legs.")
            if left_dim is None:
                left_dim = term.left_dim
                right_dim = term.right_dim
            elif term.left_dim != left_dim or term.right_dim != right_dim:
                raise ValueError("All IrreducibleMPO contributions must share the same virtual dimensions.")
        object.__setattr__(self, "scalar_blocks", scalar_blocks)
        object.__setattr__(self, "reduced_terms", reduced_terms)
        object.__setattr__(self, "symbolic_transitions", tuple(self.symbolic_transitions))

    @property
    def left_dim(self):
        if self.scalar_blocks:
            return next(iter(self.scalar_blocks.values())).shape[0]
        return self.reduced_terms[0].left_dim

    @property
    def right_dim(self):
        if self.scalar_blocks:
            return next(iter(self.scalar_blocks.values())).shape[1]
        return self.reduced_terms[0].right_dim

    @property
    def dtype(self):
        dtypes = [block.dtype for block in self.scalar_blocks.values()]
        dtypes.extend(term.dtype for term in self.reduced_terms)
        return np.result_type(*dtypes)

    def block(self, phys_out, phys_in):
        key = (phys_out, phys_in)
        if key in self._block_cache:
            return self._block_cache[key]
        total = None
        scalar_block = self.scalar_blocks.get(key)
        if scalar_block is not None:
            total = np.asarray(scalar_block)
        for term in self.reduced_terms:
            contrib = term.block(phys_out, phys_in)
            if contrib is None:
                continue
            if total is None:
                total = contrib
            else:
                total = total + contrib
        self._block_cache[key] = total
        return total

    def as_dense(self, phys_out_slices=None, phys_in_slices=None):
        if phys_in_slices is None:
            phys_in_slices = self.phys_in_leg.slices()
        if phys_out_slices is None:
            phys_out_slices = self.phys_out_leg.slices()
        p_out_dim = max(slice_.stop for slice_ in phys_out_slices.values())
        p_in_dim = max(slice_.stop for slice_ in phys_in_slices.values())
        dense = np.zeros((self.left_dim, self.right_dim, p_out_dim, p_in_dim), dtype=self.dtype)
        for q_out in self.phys_out_leg.sectors:
            for q_in in self.phys_in_leg.sectors:
                block = self.block(q_out, q_in)
                if block is None:
                    continue
                dense[:, :, phys_out_slices[q_out], phys_in_slices[q_in]] = block
        return dense


@dataclass(frozen=True)
class RankCoupledChannelTerm:
    """
    One reduced operator carried on a visible virtual channel pair.

    The visible MPO channel indices carry SU(2) irreps. The associated hidden
    component spaces are only expanded when the core is queried for a concrete
    sector block or dense tensor.
    """

    reduced_operator: object
    visible_virtual_block: np.ndarray
    use_cg_coupling: bool = False

    def __post_init__(self):
        if not hasattr(self.reduced_operator, "component_block") or not hasattr(
            self.reduced_operator, "components"
        ):
            raise TypeError(
                "RankCoupledChannelTerm expects a reduced operator exposing component_block()."
            )
        block = np.asarray(self.visible_virtual_block)
        if block.ndim != 2:
            raise ValueError(
                f"RankCoupledChannelTerm visible_virtual_block must be rank-2, got {block.shape!r}."
            )
        object.__setattr__(self, "visible_virtual_block", block)
        object.__setattr__(self, "use_cg_coupling", bool(self.use_cg_coupling))

    @property
    def dtype(self):
        return np.result_type(
            self.visible_virtual_block.dtype,
            getattr(self.reduced_operator, "dtype", float),
        )


@dataclass(frozen=True)
class RankCoupledMPO:
    """
    MPO core with visible reduced virtual channels carrying SU(2) irreps.

    The stored ``dense_blocks`` are indexed by visible virtual channels rather
    than fully expanded component-resolved channels. Reduced endpoint operators
    are carried separately in ``reduced_terms`` and expanded only when needed.
    """

    dense_blocks: dict[tuple[Sector, Sector], np.ndarray]
    phys_out_leg: PhysicalLeg
    phys_in_leg: PhysicalLeg
    left_channel_irreps: tuple[SU2Irrep, ...]
    right_channel_irreps: tuple[SU2Irrep, ...]
    left_channel_charges: tuple[int, ...] | None = None
    right_channel_charges: tuple[int, ...] | None = None
    reduced_terms: tuple[RankCoupledChannelTerm, ...] = ()
    symbolic_transitions: tuple = ()
    _reduced_block_cache: dict[tuple[Sector, Sector], dict[tuple[int, int], np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _block_cache: dict[tuple[Sector, Sector], np.ndarray | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _reduced_action_cache: tuple[tuple[object, int, int, int, int, int, object], ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if not isinstance(self.phys_out_leg, PhysicalLeg):
            raise TypeError("RankCoupledMPO phys_out_leg must be a PhysicalLeg.")
        if not isinstance(self.phys_in_leg, PhysicalLeg):
            raise TypeError("RankCoupledMPO phys_in_leg must be a PhysicalLeg.")
        left_channel_irreps = tuple(self.left_channel_irreps)
        right_channel_irreps = tuple(self.right_channel_irreps)
        left_channel_charges = (
            tuple(int(charge) for charge in self.left_channel_charges)
            if self.left_channel_charges is not None
            else tuple(0 for _ in left_channel_irreps)
        )
        right_channel_charges = (
            tuple(int(charge) for charge in self.right_channel_charges)
            if self.right_channel_charges is not None
            else tuple(0 for _ in right_channel_irreps)
        )
        if not left_channel_irreps or not right_channel_irreps:
            raise ValueError("RankCoupledMPO requires at least one left/right virtual channel.")
        if any(not isinstance(irrep, SU2Irrep) for irrep in left_channel_irreps + right_channel_irreps):
            raise TypeError("RankCoupledMPO channel irreps must be SU2Irrep objects.")
        if len(left_channel_charges) != len(left_channel_irreps) or len(right_channel_charges) != len(right_channel_irreps):
            raise ValueError("RankCoupledMPO channel charge counts must match channel irrep counts.")

        dense_blocks = {
            key: np.asarray(block)
            for key, block in dict(self.dense_blocks).items()
        }
        reduced_terms = tuple(self.reduced_terms)
        if not dense_blocks and not reduced_terms:
            raise ValueError("RankCoupledMPO requires dense blocks and/or reduced terms.")

        for key, block in dense_blocks.items():
            if len(key) != 2:
                raise ValueError(f"Invalid RankCoupledMPO block key {key!r}.")
            q_out, q_in = key
            if q_out not in self.phys_out_leg.sectors or q_in not in self.phys_in_leg.sectors:
                raise ValueError(f"Undeclared physical sector pair {key!r} in RankCoupledMPO.")
            if block.ndim != 4:
                raise ValueError(
                    f"RankCoupledMPO block {key!r} must be rank-4 over visible channels and physical legs."
                )
            if block.shape[0] != len(left_channel_irreps) or block.shape[1] != len(right_channel_irreps):
                raise ValueError(
                    f"RankCoupledMPO block {key!r} visible shape {block.shape[:2]!r} does not match "
                    f"declared virtual channel counts {(len(left_channel_irreps), len(right_channel_irreps))!r}."
                )
            if int(block.shape[2]) != self.phys_out_leg.dim(q_out):
                raise ValueError(
                    f"RankCoupledMPO block {key!r} output dimension {block.shape[2]} does not match declared dimension."
                )
            if int(block.shape[3]) != self.phys_in_leg.dim(q_in):
                raise ValueError(
                    f"RankCoupledMPO block {key!r} input dimension {block.shape[3]} does not match declared dimension."
                )

        for term in reduced_terms:
            if not isinstance(term, RankCoupledChannelTerm):
                raise TypeError("RankCoupledMPO reduced_terms must contain RankCoupledChannelTerm objects.")
            if term.visible_virtual_block.shape != (len(left_channel_irreps), len(right_channel_irreps)):
                raise ValueError(
                    "RankCoupledChannelTerm visible_virtual_block shape must match RankCoupledMPO channel counts."
                )
            if getattr(term.reduced_operator, "phys_out_leg", None) != self.phys_out_leg or getattr(
                term.reduced_operator, "phys_in_leg", None
            ) != self.phys_in_leg:
                raise ValueError("RankCoupledChannelTerm physical legs must match the RankCoupledMPO core legs.")

        object.__setattr__(self, "dense_blocks", dense_blocks)
        object.__setattr__(self, "left_channel_irreps", left_channel_irreps)
        object.__setattr__(self, "right_channel_irreps", right_channel_irreps)
        object.__setattr__(self, "left_channel_charges", left_channel_charges)
        object.__setattr__(self, "right_channel_charges", right_channel_charges)
        object.__setattr__(self, "reduced_terms", reduced_terms)
        object.__setattr__(self, "symbolic_transitions", tuple(self.symbolic_transitions))

    @property
    def left_dim(self):
        return sum(irrep.dim for irrep in self.left_channel_irreps)

    @property
    def right_dim(self):
        return sum(irrep.dim for irrep in self.right_channel_irreps)

    @property
    def dtype(self):
        dtypes = [block.dtype for block in self.dense_blocks.values()]
        dtypes.extend(term.dtype for term in self.reduced_terms)
        return np.result_type(*dtypes)

    def _left_slices(self):
        out = []
        offset = 0
        for irrep in self.left_channel_irreps:
            out.append(slice(offset, offset + irrep.dim))
            offset += irrep.dim
        return tuple(out)

    def _right_slices(self):
        out = []
        offset = 0
        for irrep in self.right_channel_irreps:
            out.append(slice(offset, offset + irrep.dim))
            offset += irrep.dim
        return tuple(out)

    def _reduced_actions(self):
        """Return cached channel/component actions for reduced block expansion."""
        cached = self._reduced_action_cache
        if cached is not None:
            return cached

        actions = []
        for term in self.reduced_terms:
            coeffs = term.visible_virtual_block
            rank_irrep = None
            if term.use_cg_coupling:
                rank_irrep = (
                    term.reduced_operator.base_operator.rank_irrep
                    if hasattr(term.reduced_operator, "base_operator")
                    else term.reduced_operator.rank_irrep
                )
            for i, left_irrep in enumerate(self.left_channel_irreps):
                left_ms = ordered_two_m_values(left_irrep)
                for j, right_irrep in enumerate(self.right_channel_irreps):
                    coeff = coeffs[i, j]
                    if coeff == 0:
                        continue
                    if term.use_cg_coupling and right_irrep not in left_or_right_fusion(left_irrep, rank_irrep):
                        continue
                    right_ms = ordered_two_m_values(right_irrep)
                    for row, two_m_left in enumerate(left_ms):
                        for col, two_m_right in enumerate(right_ms):
                            cg_coeff = 1.0
                            if term.use_cg_coupling:
                                component = two_m_right - two_m_left
                                cg_coeff = clebsch_gordan(
                                    left_irrep,
                                    rank_irrep,
                                    right_irrep,
                                    two_m_left,
                                    component,
                                    two_m_right,
                                )
                                if not cg_coeff:
                                    continue
                            elif left_irrep.two_j == 0 and right_irrep.two_j != 0:
                                component = two_m_right
                            elif right_irrep.two_j == 0 and left_irrep.two_j != 0:
                                component = two_m_left
                            else:
                                component = two_m_right - two_m_left
                            actions.append((term.reduced_operator, i, j, row, col, int(component), coeff * cg_coeff))
        cached = tuple(actions)
        object.__setattr__(self, "_reduced_action_cache", cached)
        return cached

    def reduced_block(self, phys_out, phys_in):
        key = (phys_out, phys_in)
        if key in self._reduced_block_cache:
            return self._reduced_block_cache[key]

        reduced = {}
        dense_block = self.dense_blocks.get(key)
        if dense_block is not None:
            dense_block = np.asarray(dense_block)
            nonzero_channels = np.any(dense_block != 0, axis=(2, 3))
            for i, j in zip(*np.nonzero(nonzero_channels)):
                left_irrep = self.left_channel_irreps[int(i)]
                right_irrep = self.right_channel_irreps[int(j)]
                if left_irrep != right_irrep:
                    raise ValueError(
                        f"RankCoupledMPO dense block {key!r} connects incompatible reduced channels {left_irrep!r} and {right_irrep!r}."
                    )
                local = dense_block[int(i), int(j)]
                reduced[(int(i), int(j))] = (
                    np.eye(left_irrep.dim, dtype=self.dtype)[:, :, None, None]
                    * local[None, None, :, :]
                )

        for operator, i, j, row, col, component, coeff in self._reduced_actions():
            op_block = operator.component_block(component, phys_out, phys_in)
            if op_block is None:
                continue
            local = reduced.get((i, j))
            if local is None:
                local = np.zeros(
                    (
                        self.left_channel_irreps[i].dim,
                        self.right_channel_irreps[j].dim,
                        self.phys_out_leg.dim(phys_out),
                        self.phys_in_leg.dim(phys_in),
                    ),
                    dtype=self.dtype,
                )
                reduced[(i, j)] = local
            local[row, col] += np.asarray(coeff, dtype=self.dtype) * np.asarray(
                op_block,
                dtype=self.dtype,
            )

        self._reduced_block_cache[key] = reduced
        return reduced

    def block(self, phys_out, phys_in):
        key = (phys_out, phys_in)
        if key in self._block_cache:
            return self._block_cache[key]

        left_slices = self._left_slices()
        right_slices = self._right_slices()
        total = np.zeros(
            (self.left_dim, self.right_dim, self.phys_out_leg.dim(phys_out), self.phys_in_leg.dim(phys_in)),
            dtype=self.dtype,
        )
        has_data = False
        for (i, j), local in self.reduced_block(phys_out, phys_in).items():
            total[left_slices[i], right_slices[j]] += np.asarray(local, dtype=self.dtype)
            has_data = True

        if not has_data:
            self._block_cache[key] = None
            return None
        self._block_cache[key] = total
        return total

    def as_dense(self, phys_out_slices=None, phys_in_slices=None):
        if phys_in_slices is None:
            phys_in_slices = self.phys_in_leg.slices()
        if phys_out_slices is None:
            phys_out_slices = self.phys_out_leg.slices()
        p_out_dim = max(slice_.stop for slice_ in phys_out_slices.values())
        p_in_dim = max(slice_.stop for slice_ in phys_in_slices.values())
        dense = np.zeros((self.left_dim, self.right_dim, p_out_dim, p_in_dim), dtype=self.dtype)
        for q_out in self.phys_out_leg.sectors:
            for q_in in self.phys_in_leg.sectors:
                block = self.block(q_out, q_in)
                if block is None:
                    continue
                dense[:, :, phys_out_slices[q_out], phys_in_slices[q_in]] = block
        return dense


def as_rank_coupled_mpo(core, *, phys_leg=None, cutoff=0.0):
    """
    Return ``core`` as a :class:`RankCoupledMPO`.

    Ordinary scalar MPO cores are embedded with spin-scalar visible virtual
    channels. Dense rank-4 arrays require ``phys_leg`` so the physical block
    structure is explicit.
    """
    if isinstance(core, RankCoupledMPO):
        return core
    if isinstance(core, MPO):
        scalar_irreps_left = tuple(SU2Irrep(0) for _ in range(core.left_dim))
        scalar_irreps_right = tuple(SU2Irrep(0) for _ in range(core.right_dim))
        return RankCoupledMPO(
            dense_blocks=core.blocks,
            phys_out_leg=core.phys_out_leg,
            phys_in_leg=core.phys_in_leg,
            left_channel_irreps=scalar_irreps_left,
            right_channel_irreps=scalar_irreps_right,
            left_channel_charges=tuple(0 for _ in scalar_irreps_left),
            right_channel_charges=tuple(0 for _ in scalar_irreps_right),
        )
    dense = np.asarray(core)
    if dense.ndim != 4:
        raise TypeError(
            "as_rank_coupled_mpo expects a RankCoupledMPO, MPO, or rank-4 dense core."
        )
    if phys_leg is None:
        raise ValueError("phys_leg is required when embedding a dense MPO core.")
    scalar_mpo = MPO.from_dense(
        dense,
        phys_out_leg=phys_leg,
        phys_in_leg=phys_leg,
        tol=cutoff,
    )
    return as_rank_coupled_mpo(scalar_mpo)


def _pad_visible_virtual_block(block, shape, *, row_offset=0, col_offset=0, dtype):
    out = np.zeros(shape, dtype=dtype)
    arr = np.asarray(block, dtype=dtype)
    out[
        row_offset: row_offset + arr.shape[0],
        col_offset: col_offset + arr.shape[1],
    ] = arr
    return out


def direct_sum_rank_coupled_mpo(left_core, right_core, *, site, nsites, phys_leg=None, cutoff=0.0):
    """
    Direct-sum two MPO cores while preserving reduced virtual-channel metadata.

    This is the reduced-MPO analogue of summing two finite-state MPOs: boundary
    channels are shared at the chain ends and virtual channels are direct-summed
    in the bulk.
    """
    left_core = as_rank_coupled_mpo(left_core, phys_leg=phys_leg, cutoff=cutoff)
    right_core = as_rank_coupled_mpo(right_core, phys_leg=phys_leg, cutoff=cutoff)
    if left_core.phys_out_leg != right_core.phys_out_leg or left_core.phys_in_leg != right_core.phys_in_leg:
        raise ValueError("Cannot sum MPO cores with different physical legs.")

    dtype = np.result_type(left_core.dtype, right_core.dtype)
    if nsites == 1:
        left_irreps = left_core.left_channel_irreps
        right_irreps = left_core.right_channel_irreps
        left_charges = left_core.left_channel_charges
        right_charges = left_core.right_channel_charges
        if left_irreps != right_core.left_channel_irreps or right_irreps != right_core.right_channel_irreps:
            raise ValueError("Single-site MPO sum requires matching virtual channels.")
        if left_charges != right_core.left_channel_charges or right_charges != right_core.right_channel_charges:
            raise ValueError("Single-site MPO sum requires matching virtual channel charges.")
        left_row_offset = right_row_offset = 0
        left_col_offset = right_col_offset = 0
    elif site == 0:
        left_irreps = left_core.left_channel_irreps
        left_charges = left_core.left_channel_charges
        if left_irreps != right_core.left_channel_irreps:
            raise ValueError("Left-edge MPO sum requires matching left boundary channels.")
        if left_charges != right_core.left_channel_charges:
            raise ValueError("Left-edge MPO sum requires matching left boundary channel charges.")
        right_irreps = left_core.right_channel_irreps + right_core.right_channel_irreps
        right_charges = left_core.right_channel_charges + right_core.right_channel_charges
        left_row_offset = right_row_offset = 0
        left_col_offset = 0
        right_col_offset = len(left_core.right_channel_irreps)
    elif site == nsites - 1:
        right_irreps = left_core.right_channel_irreps
        right_charges = left_core.right_channel_charges
        if right_irreps != right_core.right_channel_irreps:
            raise ValueError("Right-edge MPO sum requires matching right boundary channels.")
        if right_charges != right_core.right_channel_charges:
            raise ValueError("Right-edge MPO sum requires matching right boundary channel charges.")
        left_irreps = left_core.left_channel_irreps + right_core.left_channel_irreps
        left_charges = left_core.left_channel_charges + right_core.left_channel_charges
        left_row_offset = 0
        right_row_offset = len(left_core.left_channel_irreps)
        left_col_offset = right_col_offset = 0
    else:
        left_irreps = left_core.left_channel_irreps + right_core.left_channel_irreps
        right_irreps = left_core.right_channel_irreps + right_core.right_channel_irreps
        left_charges = left_core.left_channel_charges + right_core.left_channel_charges
        right_charges = left_core.right_channel_charges + right_core.right_channel_charges
        left_row_offset = 0
        right_row_offset = len(left_core.left_channel_irreps)
        left_col_offset = 0
        right_col_offset = len(left_core.right_channel_irreps)

    dense_blocks = {}
    block_shape = (len(left_irreps), len(right_irreps))
    for key in set(left_core.dense_blocks) | set(right_core.dense_blocks):
        q_out, q_in = key
        shape = block_shape + (
            left_core.phys_out_leg.dim(q_out),
            left_core.phys_in_leg.dim(q_in),
        )
        block = np.zeros(shape, dtype=dtype)
        left_block = left_core.dense_blocks.get(key)
        if left_block is not None:
            block[
                left_row_offset: left_row_offset + left_block.shape[0],
                left_col_offset: left_col_offset + left_block.shape[1],
            ] += np.asarray(left_block, dtype=dtype)
        right_block = right_core.dense_blocks.get(key)
        if right_block is not None:
            block[
                right_row_offset: right_row_offset + right_block.shape[0],
                right_col_offset: right_col_offset + right_block.shape[1],
            ] += np.asarray(right_block, dtype=dtype)
        if np.linalg.norm(block.reshape(-1)) > cutoff:
            dense_blocks[key] = block

    reduced_terms = []
    for term, row_offset, col_offset in (
        *((term, left_row_offset, left_col_offset) for term in left_core.reduced_terms),
        *((term, right_row_offset, right_col_offset) for term in right_core.reduced_terms),
    ):
        reduced_terms.append(
            RankCoupledChannelTerm(
                reduced_operator=term.reduced_operator,
                visible_virtual_block=_pad_visible_virtual_block(
                    term.visible_virtual_block,
                    block_shape,
                    row_offset=row_offset,
                    col_offset=col_offset,
                    dtype=dtype,
                ),
                use_cg_coupling=term.use_cg_coupling,
            )
        )

    symbolic_transitions = []
    for core, row_offset, col_offset in (
        (left_core, left_row_offset, left_col_offset),
        (right_core, right_row_offset, right_col_offset),
    ):
        for record in tuple(getattr(core, "symbolic_transitions", ()) or ()):
            if len(record) < 4:
                continue
            symbolic_transitions.append(
                (
                    record[0],
                    int(record[1]) + int(row_offset),
                    int(record[2]) + int(col_offset),
                    record[3],
                )
            )

    return RankCoupledMPO(
        dense_blocks=dense_blocks,
        phys_out_leg=left_core.phys_out_leg,
        phys_in_leg=left_core.phys_in_leg,
        left_channel_irreps=left_irreps,
        right_channel_irreps=right_irreps,
        left_channel_charges=left_charges,
        right_channel_charges=right_charges,
        reduced_terms=tuple(reduced_terms),
        symbolic_transitions=tuple(symbolic_transitions),
    )


def sum_mpo_chains(*chains, phys_leg=None, cutoff=0.0):
    """
    Sum finite MPO chains while keeping rank-coupled cores rank-coupled.
    """
    nonempty = [list(chain) for chain in chains if chain]
    if not nonempty:
        return []
    nsites = len(nonempty[0])
    if any(len(chain) != nsites for chain in nonempty):
        raise ValueError("Cannot sum MPOs with different chain lengths.")
    out = nonempty[0]
    for chain in nonempty[1:]:
        out = [
            direct_sum_rank_coupled_mpo(
                left_core,
                right_core,
                site=site,
                nsites=nsites,
                phys_leg=phys_leg,
                cutoff=cutoff,
            )
            for site, (left_core, right_core) in enumerate(zip(out, chain))
        ]
    return out
