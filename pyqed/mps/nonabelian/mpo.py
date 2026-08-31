#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symmetry-aware MPOCore core containers for the fixed-layout non-Abelian prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyqed.mps.su2 import SU2Irrep
from pyqed.mps.symmetry import Sector
from pyqed.symmetry import Leg

from .coupling import clebsch_gordan, left_or_right_fusion, ordered_two_m_values


class SparseVirtualBlock:
    """
    Sparse storage for the two virtual axes of an MPOCore block.

    Physical payloads remain small dense arrays.  This avoids padding every
    local operator across the full visible ``(D_left, D_right)`` carrier.
    """

    __slots__ = ("shape", "rows", "cols", "values")

    def __init__(self, shape, rows, cols, values):
        shape = tuple(int(dim) for dim in shape)
        if len(shape) < 2 or any(dim < 0 for dim in shape):
            raise ValueError(f"Invalid sparse virtual block shape {shape!r}.")
        rows = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1)
        cols = np.ascontiguousarray(cols, dtype=np.int64).reshape(-1)
        values = np.ascontiguousarray(values)
        expected = (rows.size,) + shape[2:]
        if cols.size != rows.size or values.shape != expected:
            raise ValueError(
                "Sparse virtual block routes and payloads have incompatible "
                f"shapes: rows={rows.shape!r}, cols={cols.shape!r}, "
                f"values={values.shape!r}, expected={expected!r}."
            )
        if np.any(rows < 0) or np.any(rows >= shape[0]):
            raise ValueError("Sparse virtual block row index is out of bounds.")
        if np.any(cols < 0) or np.any(cols >= shape[1]):
            raise ValueError("Sparse virtual block column index is out of bounds.")
        self.shape = shape
        self.rows = rows
        self.cols = cols
        self.values = values

    @classmethod
    def from_dense(cls, block):
        arr = np.asarray(block)
        if arr.ndim < 2:
            raise ValueError("SparseVirtualBlock requires at least two axes.")
        if arr.ndim == 2:
            active = arr != 0
        else:
            active = np.any(arr != 0, axis=tuple(range(2, arr.ndim)))
        rows, cols = np.nonzero(active)
        values = arr[rows, cols]
        return cls(arr.shape, rows, cols, values)

    @classmethod
    def from_entries(cls, shape, entries, *, dtype=float, retain_zeros=False):
        items = [
            ((int(row), int(col)), np.asarray(value, dtype=dtype))
            for (row, col), value in entries.items()
            if retain_zeros or np.any(np.asarray(value) != 0)
        ]
        if items:
            rows = np.fromiter((key[0] for key, _ in items), dtype=np.int64)
            cols = np.fromiter((key[1] for key, _ in items), dtype=np.int64)
            values = np.stack([value for _, value in items])
        else:
            rows = np.empty(0, dtype=np.int64)
            cols = np.empty(0, dtype=np.int64)
            values = np.empty((0,) + tuple(shape)[2:], dtype=dtype)
        return cls(shape, rows, cols, values)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nnz(self):
        return int(self.rows.size)

    def iter_routes(self):
        for index in range(self.rows.size):
            yield int(self.rows[index]), int(self.cols[index]), self.values[index]

    def with_offsets(self, shape, *, row_offset=0, col_offset=0, dtype=None):
        values = self.values if dtype is None else self.values.astype(dtype, copy=False)
        return type(self)(
            shape,
            self.rows + int(row_offset),
            self.cols + int(col_offset),
            values,
        )

    def to_dense(self, dtype=None):
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        out = np.zeros(self.shape, dtype=dtype)
        for row, col, value in self.iter_routes():
            out[row, col] += np.asarray(value, dtype=dtype)
        return out

    def __array__(self, dtype=None, copy=None):
        out = self.to_dense(dtype=dtype)
        if copy is False:
            return out
        return np.array(out, copy=True)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) >= 2 and isinstance(key[0], (int, np.integer)) and isinstance(
            key[1], (int, np.integer)
        ):
            row = int(key[0])
            col = int(key[1])
            matches = np.nonzero((self.rows == row) & (self.cols == col))[0]
            value = np.zeros(self.shape[2:], dtype=self.dtype)
            for index in matches:
                value = value + self.values[int(index)]
            return value[key[2:]] if len(key) > 2 else value
        return self.to_dense()[key]


def as_sparse_virtual_block(block):
    if isinstance(block, SparseVirtualBlock):
        return block
    return SparseVirtualBlock.from_dense(block)


def iter_virtual_routes(block):
    if isinstance(block, SparseVirtualBlock):
        return block.iter_routes()
    arr = np.asarray(block)
    if arr.ndim == 2:
        active = arr != 0
    else:
        active = np.any(arr != 0, axis=tuple(range(2, arr.ndim)))
    rows, cols = np.nonzero(active)
    return (
        (int(row), int(col), arr[int(row), int(col)])
        for row, col in zip(rows, cols)
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
    phys_out_leg: Leg
    phys_in_leg: Leg

    def __post_init__(self):
        if not isinstance(self.phys_out_leg, Leg):
            raise TypeError("SiteOperator phys_out_leg must be a Leg.")
        if not isinstance(self.phys_in_leg, Leg):
            raise TypeError("SiteOperator phys_in_leg must be a Leg.")
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
            if int(block.shape[0]) != self.phys_out_leg.sector_dim(q_out):
                raise ValueError(
                    f"Site-operator block {(q_out, q_in)!r} output dimension {block.shape[0]} "
                    f"does not match declared sector dimension {self.phys_out_leg.sector_dim(q_out)}."
                )
            if int(block.shape[1]) != self.phys_in_leg.sector_dim(q_in):
                raise ValueError(
                    f"Site-operator block {(q_out, q_in)!r} input dimension {block.shape[1]} "
                    f"does not match declared sector dimension {self.phys_in_leg.sector_dim(q_in)}."
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
                phys_out_leg = Leg.from_dims(phys_out_dims)
            elif phys_out_slices is not None:
                phys_out_leg = Leg.from_slices(phys_out_slices)
            else:
                raise ValueError(
                    "from_dense requires phys_out_leg, phys_out_dims, or phys_out_slices."
                )
        if phys_in_leg is None:
            if phys_in_dims is not None:
                phys_in_leg = Leg.from_dims(phys_in_dims)
            elif phys_in_slices is not None:
                phys_in_leg = Leg.from_slices(phys_in_slices)
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
                (phys_out_leg.sector_dim(q_out), phys_in_leg.sector_dim(q_in)),
                dtype=dense.dtype,
            )

        return cls(
            blocks=blocks,
            phys_out_leg=phys_out_leg,
            phys_in_leg=phys_in_leg,
        )


@dataclass(frozen=True)
class MPOCore:
    """
    MPOCore core with sector-keyed physical blocks.

    Parameters
    ----------
    blocks
        Mapping ``(phys_out_sector, phys_in_sector) -> dense block`` with
        each stored block shaped ``(w_left, w_right, dim_out, dim_in)``.
    phys_out_leg, phys_in_leg
        Declared physical bra/ket leg metadata in the site-local ordering.
    """

    blocks: dict[tuple[Sector, Sector], np.ndarray]
    phys_out_leg: Leg
    phys_in_leg: Leg
    symbolic_transitions: tuple = ()

    def __post_init__(self):
        phys_out_leg = self.phys_out_leg
        phys_in_leg = self.phys_in_leg
        if not isinstance(phys_out_leg, Leg):
            raise TypeError("MPOCore phys_out_leg must be a Leg.")
        if not isinstance(phys_in_leg, Leg):
            raise TypeError("MPOCore phys_in_leg must be a Leg.")
        blocks = {
            (q_out, q_in): np.asarray(block)
            for (q_out, q_in), block in self.blocks.items()
        }
        if not blocks:
            raise ValueError("MPOCore requires at least one stored block.")

        left_dim = None
        right_dim = None
        for key, block in blocks.items():
            if len(key) != 2:
                raise ValueError(f"Invalid MPOCore block key {key!r}; expected (phys_out, phys_in).")
            if block.ndim != 4:
                raise ValueError(
                    f"MPOCore block {key!r} must be rank-4, got shape {block.shape!r}."
                )
            q_out, q_in = key
            if q_out not in phys_out_leg.sectors:
                raise ValueError(
                    f"MPOCore block uses undeclared output sector {q_out!r}."
                )
            if q_in not in phys_in_leg.sectors:
                raise ValueError(
                    f"MPOCore block uses undeclared input sector {q_in!r}."
                )
            if int(block.shape[2]) != phys_out_leg.sector_dim(q_out):
                raise ValueError(
                    f"MPOCore block {key!r} output dimension {block.shape[2]} does not match "
                    f"declared sector dimension {phys_out_leg.sector_dim(q_out)}."
                )
            if int(block.shape[3]) != phys_in_leg.sector_dim(q_in):
                raise ValueError(
                    f"MPOCore block {key!r} input dimension {block.shape[3]} does not match "
                    f"declared sector dimension {phys_in_leg.sector_dim(q_in)}."
                )
            if left_dim is None:
                left_dim = int(block.shape[0])
                right_dim = int(block.shape[1])
            elif int(block.shape[0]) != left_dim or int(block.shape[1]) != right_dim:
                raise ValueError("All MPOCore blocks must share the same virtual dimensions.")

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
        Build a block-sparse MPOCore core from a dense physical-tensor core.
        """
        dense = np.asarray(core)
        if dense.ndim != 4:
            raise ValueError(
                f"MPOCore.from_dense expects a rank-4 core, got {dense.shape!r}."
            )
        if phys_out_leg is None:
            if phys_out_dims is not None:
                phys_out_leg = Leg.from_dims(phys_out_dims)
            elif phys_out_slices is not None:
                phys_out_leg = Leg.from_slices(phys_out_slices)
            else:
                raise ValueError("from_dense requires phys_out_leg, phys_out_dims, or phys_out_slices.")
        if phys_in_leg is None:
            if phys_in_dims is not None:
                phys_in_leg = Leg.from_dims(phys_in_dims)
            elif phys_in_slices is not None:
                phys_in_leg = Leg.from_slices(phys_in_slices)
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
        Build an MPOCore core directly from a symmetry-aware local site operator.

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
    One irreducible tensor-operator contribution carried by an MPOCore core.

    The virtual coefficient matrices are keyed by spherical-component labels in
    doubled-``m`` notation. Distinct components may connect different virtual
    subchannels, which lets the MPOCore contract componentwise without storing the
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
    MPOCore core that mixes ordinary scalar physical blocks with irreducible channels.

    Physical blocks are expanded lazily per sector pair through the attached
    reduced tensor operators, so the core can carry reduced operator metadata
    directly instead of storing all expanded ``(p_out, p_in)`` arrays up front.
    """

    phys_out_leg: Leg
    phys_in_leg: Leg
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
        if not isinstance(self.phys_out_leg, Leg):
            raise TypeError("IrreducibleMPO phys_out_leg must be a Leg.")
        if not isinstance(self.phys_in_leg, Leg):
            raise TypeError("IrreducibleMPO phys_in_leg must be a Leg.")
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
            if int(block.shape[2]) != self.phys_out_leg.sector_dim(q_out):
                raise ValueError(
                    f"Scalar block {key!r} output dimension {block.shape[2]} does not match declared dimension."
                )
            if int(block.shape[3]) != self.phys_in_leg.sector_dim(q_in):
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

    The visible MPOCore channel indices carry SU(2) irreps. The associated hidden
    component spaces are only expanded when the core is queried for a concrete
    sector block or dense tensor.
    """

    reduced_operator: object
    visible_virtual_block: object
    use_cg_coupling: bool = False
    left_component_orientation: int = 1
    right_component_orientation: int = 1
    orient_virtual_coupling: bool = False
    dual_right_coupling: bool = False
    phase_from_charged_scalar_source: bool = False
    phase_to_charged_pair_target: bool = False

    def __post_init__(self):
        if not hasattr(self.reduced_operator, "component_block") or not hasattr(
            self.reduced_operator, "components"
        ):
            raise TypeError(
                "RankCoupledChannelTerm expects a reduced operator exposing component_block()."
            )
        block = as_sparse_virtual_block(self.visible_virtual_block)
        if block.ndim != 2:
            raise ValueError(
                f"RankCoupledChannelTerm visible_virtual_block must be rank-2, got {block.shape!r}."
            )
        object.__setattr__(self, "visible_virtual_block", block)
        object.__setattr__(self, "use_cg_coupling", bool(self.use_cg_coupling))
        object.__setattr__(
            self,
            "orient_virtual_coupling",
            bool(self.orient_virtual_coupling),
        )
        object.__setattr__(
            self,
            "dual_right_coupling",
            bool(self.dual_right_coupling),
        )
        object.__setattr__(
            self,
            "phase_from_charged_scalar_source",
            bool(self.phase_from_charged_scalar_source),
        )
        object.__setattr__(
            self,
            "phase_to_charged_pair_target",
            bool(self.phase_to_charged_pair_target),
        )
        for name in (
            "left_component_orientation",
            "right_component_orientation",
        ):
            orientation = int(getattr(self, name))
            if orientation not in (-1, 1):
                raise ValueError(f"{name} must be -1 or 1.")
            object.__setattr__(self, name, orientation)

    @property
    def dtype(self):
        return np.result_type(
            self.visible_virtual_block.dtype,
            getattr(self.reduced_operator, "dtype", float),
        )


@dataclass(frozen=True)
class RankCoupledMPO:
    """
    MPOCore core with visible reduced virtual channels carrying SU(2) irreps.

    The stored ``dense_blocks`` are indexed by visible virtual channels rather
    than fully expanded component-resolved channels. Reduced endpoint operators
    are carried separately in ``reduced_terms`` and expanded only when needed.
    """

    dense_blocks: dict[tuple[Sector, Sector], np.ndarray]
    phys_out_leg: Leg
    phys_in_leg: Leg
    left_channel_irreps: tuple[SU2Irrep, ...]
    right_channel_irreps: tuple[SU2Irrep, ...]
    left_channel_charges: tuple[int, ...] | None = None
    right_channel_charges: tuple[int, ...] | None = None
    reduced_terms: tuple[RankCoupledChannelTerm, ...] = ()
    symbolic_transitions: tuple = ()
    normal_complementary_site: int | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    normal_complementary_plan: object | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    normal_complementary_owner: object | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    normal_complementary_fully_reduced: bool = field(
        default=False,
        compare=False,
        repr=False,
    )
    normal_complementary_right_dual: bool = field(
        default=False,
        compare=False,
        repr=False,
    )
    _reduced_block_cache: dict[tuple[Sector, Sector], dict[tuple[int, int], np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _environment_reduced_block_cache: dict[tuple, dict] = field(
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
    _dtype_cache: np.dtype | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if not isinstance(self.phys_out_leg, Leg):
            raise TypeError("RankCoupledMPO phys_out_leg must be a Leg.")
        if not isinstance(self.phys_in_leg, Leg):
            raise TypeError("RankCoupledMPO phys_in_leg must be a Leg.")
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
            key: as_sparse_virtual_block(block)
            for key, block in dict(self.dense_blocks).items()
        }
        reduced_terms = tuple(self.reduced_terms)
        native_normal_complementary = bool(
            self.normal_complementary_plan is not None
            and self.normal_complementary_owner is not None
        )
        if not dense_blocks and not reduced_terms and not native_normal_complementary:
            raise ValueError(
                "RankCoupledMPO requires dense blocks, reduced terms, or a "
                "native normal/complementary plan."
            )

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
            if int(block.shape[2]) != self.phys_out_leg.sector_dim(q_out):
                raise ValueError(
                    f"RankCoupledMPO block {key!r} output dimension {block.shape[2]} does not match declared dimension."
                )
            if int(block.shape[3]) != self.phys_in_leg.sector_dim(q_in):
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
        if self.normal_complementary_site is not None:
            object.__setattr__(
                self,
                "normal_complementary_site",
                int(self.normal_complementary_site),
            )
        object.__setattr__(
            self,
            "normal_complementary_fully_reduced",
            bool(self.normal_complementary_fully_reduced),
        )
        object.__setattr__(
            self,
            "normal_complementary_right_dual",
            bool(self.normal_complementary_right_dual),
        )
        dtypes = [block.dtype for block in dense_blocks.values()]
        dtypes.extend(term.dtype for term in reduced_terms)
        object.__setattr__(
            self,
            "_dtype_cache",
            np.dtype(np.result_type(*dtypes) if dtypes else np.float64),
        )
        # The reduced action list is static MPOCore metadata.  Build it once with
        # the MPOCore core so qchem sweeps do not repeatedly pay this bookkeeping
        # cost while advancing rank-coupled environments.
        self._reduced_actions()

    @property
    def left_dim(self):
        return sum(irrep.dim for irrep in self.left_channel_irreps)

    @property
    def right_dim(self):
        return sum(irrep.dim for irrep in self.right_channel_irreps)

    @property
    def dtype(self):
        return self._dtype_cache

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

        try:
            from pyqed.mps import cpp_davidson

            native_builder = getattr(
                cpp_davidson,
                "rank_coupled_reduced_actions",
                None,
            )
            cached = (
                None
                if native_builder is None
                or any(
                    isinstance(term.visible_virtual_block, SparseVirtualBlock)
                    for term in self.reduced_terms
                )
                else native_builder(
                    self.reduced_terms,
                    self.left_channel_irreps,
                    self.right_channel_irreps,
                )
            )
        except Exception:
            cached = None
        if cached is not None:
            cached = tuple(cached)
            object.__setattr__(self, "_reduced_action_cache", cached)
            return cached

        actions = []
        for term in self.reduced_terms:
            rank_irrep = None
            if term.use_cg_coupling:
                rank_irrep = (
                    term.reduced_operator.base_operator.rank_irrep
                    if hasattr(term.reduced_operator, "base_operator")
                    else term.reduced_operator.rank_irrep
                )
            for i, j, coeff in iter_virtual_routes(term.visible_virtual_block):
                left_irrep = self.left_channel_irreps[i]
                right_irrep = self.right_channel_irreps[j]
                left_ms = ordered_two_m_values(left_irrep)
                if term.use_cg_coupling and right_irrep not in left_or_right_fusion(left_irrep, rank_irrep):
                    continue
                right_ms = ordered_two_m_values(right_irrep)
                for row, two_m_left in enumerate(left_ms):
                    for col, two_m_right in enumerate(right_ms):
                        cg_coeff = 1.0
                        if term.use_cg_coupling:
                            oriented_left = (
                                term.left_component_orientation
                                * two_m_left
                            )
                            oriented_right = (
                                term.right_component_orientation
                                * two_m_right
                            )
                            component = oriented_right - oriented_left
                            if term.orient_virtual_coupling:
                                coupling_left = oriented_left
                                coupling_right = oriented_right
                                coupling_component = component
                            else:
                                coupling_left = two_m_left
                                coupling_right = two_m_right
                                coupling_component = (
                                    two_m_right - two_m_left
                                )
                            cg_coeff = clebsch_gordan(
                                left_irrep,
                                rank_irrep,
                                right_irrep,
                                coupling_left,
                                coupling_component,
                                coupling_right,
                            )
                            if not cg_coeff:
                                continue
                            if term.phase_from_charged_scalar_source:
                                cg_coeff *= -two_m_right
                            if term.phase_to_charged_pair_target:
                                cg_coeff *= -two_m_left
                        elif left_irrep.two_j == 0 and right_irrep.two_j != 0:
                            component = two_m_right
                        elif right_irrep.two_j == 0 and left_irrep.two_j != 0:
                            component = two_m_left
                        else:
                            component = two_m_right - two_m_left
                        actions.append(
                            (
                                term.reduced_operator,
                                i,
                                j,
                                row,
                                col,
                                int(component),
                                coeff * cg_coeff,
                            )
                        )
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
            for i, j, local_dense in iter_virtual_routes(dense_block):
                left_irrep = self.left_channel_irreps[i]
                right_irrep = self.right_channel_irreps[j]
                if left_irrep != right_irrep:
                    raise ValueError(
                        f"RankCoupledMPO dense block {key!r} connects incompatible reduced channels {left_irrep!r} and {right_irrep!r}."
                    )
                reduced[(i, j)] = (
                    np.eye(left_irrep.dim, dtype=self.dtype)[:, :, None, None]
                    * np.asarray(local_dense, dtype=self.dtype)[None, None, :, :]
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
                        self.phys_out_leg.sector_dim(phys_out),
                        self.phys_in_leg.sector_dim(phys_in),
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
            (self.left_dim, self.right_dim, self.phys_out_leg.sector_dim(phys_out), self.phys_in_leg.sector_dim(phys_in)),
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


def _rank_coupled_left_row_space(core, *, rtol, atol):
    """Return sector-preserving left-channel row-space isometries."""

    groups = {}
    for index, (irrep, charge) in enumerate(
        zip(core.left_channel_irreps, core.left_channel_charges)
    ):
        groups.setdefault((irrep, int(charge)), []).append(int(index))

    dense_routes = tuple(
        tuple(block.iter_routes())
        for block in core.dense_blocks.values()
    )
    reduced_routes = tuple(
        tuple(term.visible_virtual_block.iter_routes())
        for term in core.reduced_terms
    )
    transforms = []
    for key, indices in groups.items():
        row_for_index = {
            int(index): int(row)
            for row, index in enumerate(indices)
        }
        feature_for_key = {}
        entries = []

        def add_feature(row, feature, value):
            column = feature_for_key.get(feature)
            if column is None:
                column = len(feature_for_key)
                feature_for_key[feature] = column
            entries.append((int(row), int(column), value))

        for block_index, routes in enumerate(dense_routes):
            for left, right, payload in routes:
                row = row_for_index.get(int(left))
                if row is None:
                    continue
                flat = np.asarray(payload).reshape(-1)
                for physical, value in enumerate(flat):
                    if value != 0:
                        add_feature(
                            row,
                            ("dense", int(block_index), int(right), int(physical)),
                            value,
                        )
        for term_index, routes in enumerate(reduced_routes):
            for left, right, payload in routes:
                row = row_for_index.get(int(left))
                if row is None:
                    continue
                value = np.asarray(payload).reshape(()).item()
                if value != 0:
                    add_feature(
                        row,
                        ("reduced", int(term_index), int(right)),
                        value,
                    )

        if not feature_for_key:
            continue
        matrix = np.zeros(
            (len(indices), len(feature_for_key)),
            dtype=core.dtype,
        )
        for row, column, value in entries:
            matrix[row, column] += value
        if len(indices) == 1:
            isometry = np.ones((1, 1), dtype=matrix.dtype)
        else:
            left_vectors, singular_values, _ = np.linalg.svd(
                matrix,
                full_matrices=False,
            )
            largest = (
                0.0
                if singular_values.size == 0
                else float(np.max(np.abs(singular_values)))
            )
            threshold = max(float(atol), float(rtol) * largest)
            rank = int(np.count_nonzero(np.abs(singular_values) > threshold))
            if rank <= 0:
                continue
            if rank == len(indices):
                isometry = np.eye(len(indices), dtype=matrix.dtype)
            else:
                isometry = np.ascontiguousarray(left_vectors[:, :rank])
        transforms.append(
            (
                key,
                np.asarray(indices, dtype=np.int64),
                isometry,
            )
        )
    return tuple(transforms)


def _rank_coupled_left_row_skeleton(core, *, rtol, atol):
    """Return sparse row selectors and exact interpolation gauges by sector."""

    groups = {}
    for index, (irrep, charge) in enumerate(
        zip(core.left_channel_irreps, core.left_channel_charges)
    ):
        groups.setdefault((irrep, int(charge)), []).append(int(index))

    dense_routes = tuple(
        tuple(block.iter_routes())
        for block in core.dense_blocks.values()
    )
    reduced_routes = tuple(
        tuple(term.visible_virtual_block.iter_routes())
        for term in core.reduced_terms
    )
    transforms = []
    for key, indices in groups.items():
        row_for_index = {
            int(index): int(row)
            for row, index in enumerate(indices)
        }
        feature_for_key = {}
        entries = []

        def add_feature(row, feature, value):
            column = feature_for_key.get(feature)
            if column is None:
                column = len(feature_for_key)
                feature_for_key[feature] = column
            entries.append((int(row), int(column), value))

        for block_index, routes in enumerate(dense_routes):
            for left, right, payload in routes:
                row = row_for_index.get(int(left))
                if row is None:
                    continue
                flat = np.asarray(payload).reshape(-1)
                for physical, value in enumerate(flat):
                    if value != 0:
                        add_feature(
                            row,
                            ("dense", int(block_index), int(right), int(physical)),
                            value,
                        )
        for term_index, routes in enumerate(reduced_routes):
            for left, right, payload in routes:
                row = row_for_index.get(int(left))
                if row is None:
                    continue
                value = np.asarray(payload).reshape(()).item()
                if value != 0:
                    add_feature(
                        row,
                        ("reduced", int(term_index), int(right)),
                        value,
                    )

        if not feature_for_key:
            continue
        matrix = np.zeros(
            (len(indices), len(feature_for_key)),
            dtype=core.dtype,
        )
        for row, column, value in entries:
            matrix[row, column] += value
        if len(indices) == 1:
            selector = np.ones((1, 1), dtype=matrix.dtype)
            interpolation = selector
        else:
            from scipy.linalg import qr, solve_triangular

            _, triangular, pivots = qr(
                matrix.T,
                mode="economic",
                pivoting=True,
                check_finite=False,
            )
            diagonal = np.abs(np.diag(triangular))
            largest = 0.0 if diagonal.size == 0 else float(diagonal.max())
            threshold = max(float(atol), float(rtol) * largest)
            rank = int(np.count_nonzero(diagonal > threshold))
            if rank <= 0:
                continue
            selected = np.asarray(pivots[:rank], dtype=np.int64)
            selector = np.zeros((len(indices), rank), dtype=matrix.dtype)
            selector[selected, np.arange(rank)] = 1
            pivot_coordinates = solve_triangular(
                triangular[:rank, :rank],
                triangular[:rank, :],
                lower=False,
                check_finite=False,
            )
            interpolation = np.zeros(
                (len(indices), rank),
                dtype=matrix.dtype,
            )
            interpolation[np.asarray(pivots, dtype=np.int64), :] = (
                pivot_coordinates.T
            )
        transforms.append(
            (
                key,
                np.asarray(indices, dtype=np.int64),
                np.ascontiguousarray(selector),
                np.ascontiguousarray(interpolation),
            )
        )
    return tuple(transforms)


def _transform_sparse_virtual_axis(
    block,
    transforms,
    *,
    axis,
    new_shape,
    cutoff,
):
    """Apply block-diagonal virtual gauges to one sparse MPOCore payload."""

    block = as_sparse_virtual_block(block)
    payload_shape = tuple(block.shape[2:])
    rows = []
    cols = []
    values = []
    new_offset = 0
    for _key, old_indices, transform in transforms:
        old_indices = np.asarray(old_indices, dtype=np.int64)
        transform = np.asarray(transform)
        old_to_local = {
            int(index): int(local)
            for local, index in enumerate(old_indices)
        }
        if axis == 0:
            sub = np.zeros(
                (old_indices.size, block.shape[1]) + payload_shape,
                dtype=np.result_type(block.dtype, transform.dtype),
            )
            for old_left, old_right, payload in block.iter_routes():
                local = old_to_local.get(int(old_left))
                if local is not None:
                    sub[local, int(old_right)] += payload
            result = np.tensordot(
                transform.conj().T,
                sub,
                axes=(1, 0),
            )
            if payload_shape:
                active = np.any(
                    np.abs(result) > float(cutoff),
                    axis=tuple(range(2, result.ndim)),
                )
            else:
                active = np.abs(result) > float(cutoff)
            new_rows, new_cols = np.nonzero(active)
            for new_left, old_right in zip(new_rows, new_cols):
                rows.append(int(new_offset + new_left))
                cols.append(int(old_right))
                values.append(np.asarray(result[new_left, old_right]))
        elif axis == 1:
            sub = np.zeros(
                (block.shape[0], old_indices.size) + payload_shape,
                dtype=np.result_type(block.dtype, transform.dtype),
            )
            for old_left, old_right, payload in block.iter_routes():
                local = old_to_local.get(int(old_right))
                if local is not None:
                    sub[int(old_left), local] += payload
            result = np.tensordot(sub, transform, axes=(1, 0))
            result = np.moveaxis(result, -1, 1)
            if payload_shape:
                active = np.any(
                    np.abs(result) > float(cutoff),
                    axis=tuple(range(2, result.ndim)),
                )
            else:
                active = np.abs(result) > float(cutoff)
            old_rows, new_cols = np.nonzero(active)
            for old_left, new_right in zip(old_rows, new_cols):
                rows.append(int(old_left))
                cols.append(int(new_offset + new_right))
                values.append(np.asarray(result[old_left, new_right]))
        else:
            raise ValueError("Sparse virtual-axis transform expects axis 0 or 1.")
        new_offset += int(transform.shape[1])

    dtype = np.result_type(block.dtype, *(item[2].dtype for item in transforms))
    if values:
        value_array = np.ascontiguousarray(np.stack(values), dtype=dtype)
    else:
        value_array = np.empty((0,) + payload_shape, dtype=dtype)
    return SparseVirtualBlock(
        tuple(int(value) for value in new_shape) + payload_shape,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        value_array,
    )


def _transform_rank_coupled_core_axis(
    core,
    transforms,
    *,
    axis,
    cutoff,
):
    """Return a rank-coupled core after one sector-preserving virtual gauge."""

    transforms = tuple(transforms)
    if not transforms:
        raise ValueError("Rank-coupled virtual compression removed every channel.")
    new_irreps = tuple(
        key[0]
        for key, _indices, transform in transforms
        for _ in range(int(transform.shape[1]))
    )
    new_charges = tuple(
        int(key[1])
        for key, _indices, transform in transforms
        for _ in range(int(transform.shape[1]))
    )
    if axis == 0:
        left_irreps = new_irreps
        left_charges = new_charges
        right_irreps = core.right_channel_irreps
        right_charges = core.right_channel_charges
    elif axis == 1:
        left_irreps = core.left_channel_irreps
        left_charges = core.left_channel_charges
        right_irreps = new_irreps
        right_charges = new_charges
    else:
        raise ValueError("Rank-coupled core transform expects axis 0 or 1.")
    visible_shape = (len(left_irreps), len(right_irreps))
    dense_blocks = {
        key: _transform_sparse_virtual_axis(
            block,
            transforms,
            axis=axis,
            new_shape=visible_shape,
            cutoff=cutoff,
        )
        for key, block in core.dense_blocks.items()
    }
    reduced_terms = tuple(
        RankCoupledChannelTerm(
            reduced_operator=term.reduced_operator,
            visible_virtual_block=_transform_sparse_virtual_axis(
                term.visible_virtual_block,
                transforms,
                axis=axis,
                new_shape=visible_shape,
                cutoff=cutoff,
            ),
            use_cg_coupling=term.use_cg_coupling,
            left_component_orientation=term.left_component_orientation,
            right_component_orientation=term.right_component_orientation,
            orient_virtual_coupling=term.orient_virtual_coupling,
            dual_right_coupling=term.dual_right_coupling,
            phase_from_charged_scalar_source=(
                term.phase_from_charged_scalar_source
            ),
            phase_to_charged_pair_target=(
                term.phase_to_charged_pair_target
            ),
        )
        for term in core.reduced_terms
    )
    return RankCoupledMPO(
        dense_blocks=dense_blocks,
        phys_out_leg=core.phys_out_leg,
        phys_in_leg=core.phys_in_leg,
        left_channel_irreps=left_irreps,
        right_channel_irreps=right_irreps,
        left_channel_charges=left_charges,
        right_channel_charges=right_charges,
        reduced_terms=reduced_terms,
        symbolic_transitions=(),
    )


def compress_rank_coupled_mpo_chain(
    factors,
    *,
    rtol=1.0e-12,
    atol=1.0e-14,
    cutoff=0.0,
    gauge="skeleton",
    return_info=False,
):
    """
    Exactly compress a rank-coupled MPOCore from right to left.

    Virtual channels are reduced only inside equal ``(SU(2) irrep, charge)``
    sectors.  The default skeleton gauge keeps selected symbolic suffix rows
    unchanged and places integral interpolation coefficients on the adjacent
    core.  This preserves substantially more local sparsity than an
    orthonormal SVD gauge.
    """

    factors = list(factors)
    if not factors:
        raise ValueError("Rank-coupled MPOCore compression requires a nonempty chain.")
    if any(not isinstance(core, RankCoupledMPO) for core in factors):
        raise TypeError("Rank-coupled MPOCore compression requires RankCoupledMPO cores.")
    gauge = str(gauge).strip().lower()
    if gauge not in {"skeleton", "orthonormal"}:
        raise ValueError("gauge must be 'skeleton' or 'orthonormal'.")
    before = tuple(int(len(core.right_channel_irreps)) for core in factors)
    sector_ranks = []
    for site in range(len(factors) - 1, 0, -1):
        if gauge == "skeleton":
            skeletons = _rank_coupled_left_row_skeleton(
                factors[site],
                rtol=rtol,
                atol=atol,
            )
            right_transforms = tuple(
                (key, indices, selector)
                for key, indices, selector, _interpolation in skeletons
            )
            left_transforms = tuple(
                (key, indices, interpolation)
                for key, indices, _selector, interpolation in skeletons
            )
        else:
            right_transforms = left_transforms = _rank_coupled_left_row_space(
                factors[site],
                rtol=rtol,
                atol=atol,
            )
        old_dimension = int(len(factors[site].left_channel_irreps))
        new_dimension = int(
            sum(
                transform.shape[1]
                for _, _, transform in right_transforms
            )
        )
        sector_ranks.append(
            {
                "bond": int(site - 1),
                "old_visible_channels": old_dimension,
                "new_visible_channels": new_dimension,
            }
        )
        if new_dimension == old_dimension and all(
            transform.shape[0] == transform.shape[1]
            and np.array_equal(
                transform,
                np.eye(transform.shape[0], dtype=transform.dtype),
            )
            for _key, _indices, transform in right_transforms
        ):
            continue
        factors[site] = _transform_rank_coupled_core_axis(
            factors[site],
            right_transforms,
            axis=0,
            cutoff=cutoff,
        )
        factors[site - 1] = _transform_rank_coupled_core_axis(
            factors[site - 1],
            left_transforms,
            axis=1,
            cutoff=cutoff,
        )
    after = tuple(int(len(core.right_channel_irreps)) for core in factors)
    info = {
        "source": "sector_preserving_operator_skeleton",
        "gauge": gauge,
        "rtol": float(rtol),
        "atol": float(atol),
        "cutoff": float(cutoff),
        "visible_bond_channels_before": before,
        "visible_bond_channels_after": after,
        "max_visible_bond_channels_before": int(max(before)),
        "max_visible_bond_channels_after": int(max(after)),
        "sector_ranks": tuple(reversed(sector_ranks)),
    }
    return (factors, info) if return_info else factors


def expand_rank_coupled_mpo(core):
    """Expand visible SU(2) virtual components into an ordinary block MPOCore.

    Physical state tensors remain fully reduced.  This compatibility view is
    used only by the projected local LETTA fallback; native Wigner--Eckart
    contractions continue to consume the rank-coupled core directly.
    """
    if isinstance(core, MPOCore):
        return core
    if not isinstance(core, RankCoupledMPO):
        raise TypeError("expand_rank_coupled_mpo expects an MPOCore or RankCoupledMPO core.")
    blocks = {}
    for q_out in core.phys_out_leg.sectors:
        for q_in in core.phys_in_leg.sectors:
            block = core.block(q_out, q_in)
            if block is not None:
                blocks[(q_out, q_in)] = block
    return MPOCore(
        blocks=blocks,
        phys_out_leg=core.phys_out_leg,
        phys_in_leg=core.phys_in_leg,
    )


def as_rank_coupled_mpo(core, *, phys_leg=None, cutoff=0.0):
    """
    Return ``core`` as a :class:`RankCoupledMPO`.

    Ordinary scalar MPOCore cores are embedded with spin-scalar visible virtual
    channels. Dense rank-4 arrays require ``phys_leg`` so the physical block
    structure is explicit.
    """
    if isinstance(core, RankCoupledMPO):
        return core
    if isinstance(core, MPOCore):
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
            "as_rank_coupled_mpo expects a RankCoupledMPO, MPOCore, or rank-4 dense core."
        )
    if phys_leg is None:
        raise ValueError("phys_leg is required when embedding a dense MPOCore core.")
    scalar_mpo = MPOCore.from_dense(
        dense,
        phys_out_leg=phys_leg,
        phys_in_leg=phys_leg,
        tol=cutoff,
    )
    return as_rank_coupled_mpo(scalar_mpo)


def _pad_visible_virtual_block(block, shape, *, row_offset=0, col_offset=0, dtype):
    return as_sparse_virtual_block(block).with_offsets(
        shape,
        row_offset=row_offset,
        col_offset=col_offset,
        dtype=dtype,
    )


def direct_sum_rank_coupled_mpo(left_core, right_core, *, site, nsites, phys_leg=None, cutoff=0.0):
    """
    Direct-sum two MPOCore cores while preserving reduced virtual-channel metadata.

    This is the reduced-MPOCore analogue of summing two finite-state MPOs: boundary
    channels are shared at the chain ends and virtual channels are direct-summed
    in the bulk.
    """
    left_core = as_rank_coupled_mpo(left_core, phys_leg=phys_leg, cutoff=cutoff)
    right_core = as_rank_coupled_mpo(right_core, phys_leg=phys_leg, cutoff=cutoff)
    if left_core.phys_out_leg != right_core.phys_out_leg or left_core.phys_in_leg != right_core.phys_in_leg:
        raise ValueError("Cannot sum MPOCore cores with different physical legs.")

    dtype = np.result_type(left_core.dtype, right_core.dtype)
    if nsites == 1:
        left_irreps = left_core.left_channel_irreps
        right_irreps = left_core.right_channel_irreps
        left_charges = left_core.left_channel_charges
        right_charges = left_core.right_channel_charges
        if left_irreps != right_core.left_channel_irreps or right_irreps != right_core.right_channel_irreps:
            raise ValueError("Single-site MPOCore sum requires matching virtual channels.")
        if left_charges != right_core.left_channel_charges or right_charges != right_core.right_channel_charges:
            raise ValueError("Single-site MPOCore sum requires matching virtual channel charges.")
        left_row_offset = right_row_offset = 0
        left_col_offset = right_col_offset = 0
    elif site == 0:
        left_irreps = left_core.left_channel_irreps
        left_charges = left_core.left_channel_charges
        if left_irreps != right_core.left_channel_irreps:
            raise ValueError("Left-edge MPOCore sum requires matching left boundary channels.")
        if left_charges != right_core.left_channel_charges:
            raise ValueError("Left-edge MPOCore sum requires matching left boundary channel charges.")
        right_irreps = left_core.right_channel_irreps + right_core.right_channel_irreps
        right_charges = left_core.right_channel_charges + right_core.right_channel_charges
        left_row_offset = right_row_offset = 0
        left_col_offset = 0
        right_col_offset = len(left_core.right_channel_irreps)
    elif site == nsites - 1:
        right_irreps = left_core.right_channel_irreps
        right_charges = left_core.right_channel_charges
        if right_irreps != right_core.right_channel_irreps:
            raise ValueError("Right-edge MPOCore sum requires matching right boundary channels.")
        if right_charges != right_core.right_channel_charges:
            raise ValueError("Right-edge MPOCore sum requires matching right boundary channel charges.")
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
            left_core.phys_out_leg.sector_dim(q_out),
            left_core.phys_in_leg.sector_dim(q_in),
        )
        pieces = []
        left_block = left_core.dense_blocks.get(key)
        if left_block is not None:
            pieces.append(
                as_sparse_virtual_block(left_block).with_offsets(
                    shape,
                    row_offset=left_row_offset,
                    col_offset=left_col_offset,
                    dtype=dtype,
                )
            )
        right_block = right_core.dense_blocks.get(key)
        if right_block is not None:
            pieces.append(
                as_sparse_virtual_block(right_block).with_offsets(
                    shape,
                    row_offset=right_row_offset,
                    col_offset=right_col_offset,
                    dtype=dtype,
                )
            )
        if not pieces:
            continue
        values = np.concatenate([piece.values for piece in pieces], axis=0)
        if np.linalg.norm(values.reshape(-1)) > cutoff:
            dense_blocks[key] = SparseVirtualBlock(
                shape,
                np.concatenate([piece.rows for piece in pieces]),
                np.concatenate([piece.cols for piece in pieces]),
                values,
            )

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
                left_component_orientation=term.left_component_orientation,
                right_component_orientation=term.right_component_orientation,
                orient_virtual_coupling=term.orient_virtual_coupling,
                dual_right_coupling=term.dual_right_coupling,
                phase_from_charged_scalar_source=(
                    term.phase_from_charged_scalar_source
                ),
                phase_to_charged_pair_target=(
                    term.phase_to_charged_pair_target
                ),
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
    Sum finite MPOCore chains while keeping rank-coupled cores rank-coupled.
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


def scale_mpo_chain(chain, coefficient, *, site=0):
    """Return an MPOCore chain multiplied by one scalar coefficient."""
    chain = list(chain)
    if not chain:
        return []
    site = int(site)
    if site < 0:
        site += len(chain)
    if site < 0 or site >= len(chain):
        raise IndexError("MPOCore scaling site is out of range.")

    core = as_rank_coupled_mpo(chain[site])
    coefficient = np.asarray(coefficient).reshape(()).item()
    dense_blocks = {
        key: SparseVirtualBlock(
            block.shape,
            block.rows,
            block.cols,
            np.asarray(block.values) * coefficient,
        )
        for key, block in core.dense_blocks.items()
    }
    reduced_terms = tuple(
        RankCoupledChannelTerm(
            reduced_operator=term.reduced_operator,
            visible_virtual_block=SparseVirtualBlock(
                term.visible_virtual_block.shape,
                term.visible_virtual_block.rows,
                term.visible_virtual_block.cols,
                np.asarray(term.visible_virtual_block.values) * coefficient,
            ),
            use_cg_coupling=term.use_cg_coupling,
            left_component_orientation=term.left_component_orientation,
            right_component_orientation=term.right_component_orientation,
            orient_virtual_coupling=term.orient_virtual_coupling,
            dual_right_coupling=term.dual_right_coupling,
            phase_from_charged_scalar_source=term.phase_from_charged_scalar_source,
            phase_to_charged_pair_target=term.phase_to_charged_pair_target,
        )
        for term in core.reduced_terms
    )
    chain[site] = RankCoupledMPO(
        dense_blocks=dense_blocks,
        phys_out_leg=core.phys_out_leg,
        phys_in_leg=core.phys_in_leg,
        left_channel_irreps=core.left_channel_irreps,
        right_channel_irreps=core.right_channel_irreps,
        left_channel_charges=core.left_channel_charges,
        right_channel_charges=core.right_channel_charges,
        reduced_terms=reduced_terms,
        symbolic_transitions=core.symbolic_transitions,
    )
    return chain
