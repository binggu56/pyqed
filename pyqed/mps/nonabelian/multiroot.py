#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State-averaged multi-root MPS owner for non-Abelian sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mps import MPS
from pyqed.symmetry import IrrepTensor


_ROOT_AXIS_SECTOR = ("state_average_root", 0)


def _ordered_max_multiplicity_union(legs):
    order = []
    multiplicities = {}
    for leg in legs:
        counts = {}
        for sector in leg:
            counts[sector] = counts.get(sector, 0) + 1
            if sector not in order:
                order.append(sector)
        for sector, count in counts.items():
            multiplicities[sector] = max(
                multiplicities.get(sector, 0),
                count,
            )
    return [
        sector
        for sector in order
        for _ in range(multiplicities[sector])
    ]


def _align_shared_root_chains(root_sites):
    """Embed root chains in the union reduced bond skeleton without projection."""

    nroots = len(root_sites)
    nsites = len(root_sites[0])
    common_legs = []
    for site in range(nsites):
        common_legs.append(
            [
                _ordered_max_multiplicity_union(
                    [root_sites[root][site].qns[axis] for root in range(nroots)]
                )
                for axis in range(3)
            ]
        )
    for bond in range(nsites - 1):
        bond_qns = _ordered_max_multiplicity_union(
            [common_legs[bond][2], common_legs[bond + 1][0]]
        )
        common_legs[bond][2] = bond_qns[:]
        common_legs[bond + 1][0] = bond_qns[:]

    aligned_roots = [[] for _ in range(nroots)]
    for site in range(nsites):
        tensors = [root_sites[root][site] for root in range(nroots)]
        keys = sorted(set().union(*(tensor.data for tensor in tensors)))
        phys_dims = {}
        for tensor in tensors:
            for key, block in tensor.data.items():
                phys_dims[key[1]] = max(
                    phys_dims.get(key[1], 0),
                    int(np.asarray(block).shape[1]),
                )
        left_dims = {
            sector: common_legs[site][0].count(sector)
            for sector in set(common_legs[site][0])
        }
        right_dims = {
            sector: common_legs[site][2].count(sector)
            for sector in set(common_legs[site][2])
        }
        physical_basis = next(
            (
                tensor.metadata.get("physical_basis")
                for tensor in tensors
                if tensor.metadata.get("physical_basis") is not None
            ),
            None,
        )
        for root, tensor in enumerate(tensors):
            dtype = np.result_type(
                *[np.asarray(block).dtype for block in tensor.data.values()],
                float,
            )
            data = {}
            for key in keys:
                shape = (
                    left_dims[key[0]],
                    phys_dims[key[1]],
                    right_dims[key[2]],
                )
                target = np.zeros(shape, dtype=dtype)
                source = tensor.data.get(key)
                if source is not None:
                    source = np.asarray(source)
                    target[tuple(slice(0, size) for size in source.shape)] = source
                data[key] = target
            aligned_roots[root].append(
                IrrepTensor(
                    data,
                    [leg[:] for leg in common_legs[site]],
                    tensor.dirs[:],
                    fusion_legs=[None, tensor.fusion_legs[1], None],
                    metadata=(
                        {"physical_basis": physical_basis}
                        if physical_basis is not None
                        else {}
                    ),
                )
            )
    return aligned_roots


def fuse_root_center_tensors(center_tensors, *, root_sector=_ROOT_AXIS_SECTOR):
    """Stack rank-4 root center tensors behind one leading root axis."""

    tensors = [tensor.copy() for tensor in center_tensors or []]
    if not tensors:
        return None
    first = tensors[0]
    if not isinstance(first, IrrepTensor) or first.rank != 4:
        raise ValueError("Root center tensors must be rank-4 IrrepTensor objects.")
    qns = [leg[:] for leg in first.qns]
    dirs = first.dirs[:]
    fusion_legs = first.fusion_legs[:]
    for tensor in tensors[1:]:
        if not isinstance(tensor, IrrepTensor) or tensor.rank != 4:
            raise ValueError("Root center tensors must be rank-4 IrrepTensor objects.")
        if tensor.qns != qns or tensor.dirs != dirs or tensor.fusion_legs != fusion_legs:
            raise ValueError("Root center tensors must share the same block layout.")

    data = {}
    for key in sorted(
        set().union(*(tensor.data.keys() for tensor in tensors)),
        key=lambda item: tuple(repr(part) for part in item),
    ):
        shape = None
        for tensor in tensors:
            block = tensor.data.get(key)
            if block is not None:
                shape = block.shape
                break
        if shape is None:
            continue
        blocks = []
        for tensor in tensors:
            block = tensor.data.get(key)
            if block is None:
                blocks.append(np.zeros(shape, dtype=complex))
            else:
                if block.shape != shape:
                    raise ValueError("Root center tensor blocks must have matching shapes.")
                blocks.append(np.asarray(block))
        data[(root_sector, *key)] = np.stack(blocks, axis=0)

    metadata = first.metadata.copy()
    metadata.update(
        {
            "state_average_root_axis": True,
            "state_average_nroots": int(len(tensors)),
            "state_average_center_rank": 4,
        }
    )
    return IrrepTensor(
        data=data,
        qns=[[root_sector], *qns],
        dirs=[1, *dirs],
        fusion_legs=[None, *fusion_legs],
        metadata=metadata,
    )


def unfuse_root_center_tensor(center_tensor):
    """Return rank-4 root center tensors from a fused root-axis tensor."""

    if center_tensor is None:
        return None
    if not isinstance(center_tensor, IrrepTensor) or center_tensor.rank != 5:
        raise ValueError("Fused root center tensor must be a rank-5 IrrepTensor.")
    nroots = int(center_tensor.metadata.get("state_average_nroots", 0))
    if nroots <= 0:
        for block in center_tensor.data.values():
            nroots = int(block.shape[0])
            break
    root_data = [dict() for _ in range(nroots)]
    for key, block in center_tensor.data.items():
        block = np.asarray(block)
        if block.shape[0] != nroots:
            raise ValueError("Fused root center blocks disagree on the root-axis dimension.")
        child_key = tuple(key[1:])
        for root_idx in range(nroots):
            root_data[root_idx][child_key] = np.array(block[root_idx], copy=True)
    metadata = center_tensor.metadata.copy()
    metadata.pop("state_average_root_axis", None)
    metadata.pop("state_average_nroots", None)
    metadata.pop("state_average_center_rank", None)
    return [
        IrrepTensor(
            data=data,
            qns=[leg[:] for leg in center_tensor.qns[1:]],
            dirs=center_tensor.dirs[1:],
            fusion_legs=center_tensor.fusion_legs[1:],
            metadata=metadata,
        )
        for data in root_data
    ]


@dataclass
class MultiRootMPS:
    """
    Owner for a state-averaged set of roots sharing one DMRG truncation basis.

    The current sweep engine still stores root-specific site tensors after each
    SA-SVD.  This wrapper makes that ownership explicit and keeps the shared
    representative chain, root list, and weights together at API boundaries.
    """

    roots: list[MPS]
    weights: np.ndarray | None = None
    center: int | None = None
    center_bond: int | None = None
    center_tensor: IrrepTensor | None = None
    center_tensors: list[IrrepTensor] | None = None
    target_sector: object | None = None

    def __post_init__(self):
        self.roots = [MPS.from_tensors(root) for root in self.roots]
        if not self.roots:
            raise ValueError("MultiRootMPS requires at least one root.")
        nsites = len(self.roots[0])
        if any(len(root) != nsites for root in self.roots):
            raise ValueError("All MultiRootMPS roots must have the same chain length.")
        if self.target_sector is None:
            self.target_sector = self.roots[0].target_sector
        for root in self.roots:
            if self.target_sector is not None:
                root.target_sector = self.target_sector
            if self.center is not None:
                root.center = int(self.center)
        if self.weights is None:
            self.weights = np.ones(len(self.roots), dtype=float) / len(self.roots)
        else:
            weights = np.asarray(self.weights, dtype=float).reshape(-1)
            if weights.size != len(self.roots):
                raise ValueError("MultiRootMPS weights must match the number of roots.")
            total = float(np.sum(weights))
            if abs(total) <= 1.0e-15:
                raise ValueError("MultiRootMPS weights must not sum to zero.")
            self.weights = weights / total
        if self.center_bond is not None:
            self.center_bond = int(self.center_bond)
            if not 0 <= self.center_bond < nsites - 1:
                raise IndexError(
                    f"Center bond {self.center_bond} out of range for chain length {nsites}."
                )
        if self.center_tensor is not None and self.center_tensors is not None:
            raise ValueError("Specify either center_tensor or center_tensors, not both.")
        if self.center_tensors is not None:
            if len(self.center_tensors) != len(self.roots):
                raise ValueError("MultiRootMPS center_tensors must match the number of roots.")
            self.center_tensor = fuse_root_center_tensors(self.center_tensors)
            self.center_tensors = None
        if self.center_tensor is not None:
            self.center_tensor = self.center_tensor.copy()
            if not isinstance(self.center_tensor, IrrepTensor) or self.center_tensor.rank != 5:
                raise ValueError("MultiRootMPS center_tensor must be a fused rank-5 IrrepTensor.")
            nroots = int(self.center_tensor.metadata.get("state_average_nroots", len(self.roots)))
            if nroots != len(self.roots):
                raise ValueError("MultiRootMPS center_tensor root axis must match the number of roots.")

    @classmethod
    def from_root_sites(
        cls,
        root_sites,
        *,
        weights=None,
        center=None,
        center_bond=None,
        center_tensor=None,
        center_tensors=None,
        target_sector=None,
    ):
        roots = [
            MPS.from_tensors(root, center=center, target_sector=target_sector)
            for root in root_sites
        ]
        return cls(
            roots,
            weights=weights,
            center=center,
            center_bond=center_bond,
            center_tensor=center_tensor,
            center_tensors=center_tensors,
            target_sector=target_sector,
        )

    @property
    def nroots(self):
        return len(self.roots)

    @property
    def sites(self):
        return self.roots[0].tensors

    def root_site_lists(self):
        return [[site.copy() for site in root.tensors] for root in self.roots]

    def copy(self):
        return MultiRootMPS(
            [root.copy() for root in self.roots],
            weights=np.array(self.weights, copy=True),
            center=self.center,
            center_bond=self.center_bond,
            center_tensor=self.center_tensor.copy() if self.center_tensor is not None else None,
            target_sector=self.target_sector,
        )

    def root_center_tensors(self):
        return unfuse_root_center_tensor(self.center_tensor)

    def with_root_sites(
        self,
        root_sites,
        *,
        center=None,
        center_bond=None,
        center_tensor=None,
        center_tensors=None,
    ):
        if center_tensors is None:
            center_tensor_arg = self.center_tensor if center_tensor is None else center_tensor
        else:
            center_tensor_arg = None
        return MultiRootMPS.from_root_sites(
            root_sites,
            weights=self.weights,
            center=self.center if center is None else center,
            center_bond=self.center_bond if center_bond is None else center_bond,
            center_tensor=center_tensor_arg,
            center_tensors=center_tensors,
            target_sector=self.target_sector,
        )

    def with_center_tensors(self, center_bond, center_tensors):
        return MultiRootMPS(
            [root.copy() for root in self.roots],
            weights=np.array(self.weights, copy=True),
            center=self.center,
            center_bond=center_bond,
            center_tensors=center_tensors,
            target_sector=self.target_sector,
        )

    def with_center_tensor(self, center_bond, center_tensor):
        return MultiRootMPS(
            [root.copy() for root in self.roots],
            weights=np.array(self.weights, copy=True),
            center=self.center,
            center_bond=center_bond,
            center_tensor=center_tensor,
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
        for root in self.roots:
            root.canonicalize(
                center,
                cutoff=cutoff,
                max_bond=max_bond,
                max_bond_mode=max_bond_mode,
                bond_coupling=bond_coupling,
            )
        self.center = int(center)
        return self

    def canonicalize_shared(
        self,
        center,
        *,
        bond_coupling="left",
        max_bond=None,
        max_bond_mode="reduced",
    ):
        """Put every root in one shared isometric basis with one root center."""

        from .canonical import mixed_canonicalize_sites
        from .contraction import merge_mps_sites
        from .decompose import state_averaged_svd_two_site

        center = int(center)
        nsites = len(self.roots[0])
        if center not in {0, nsites - 1}:
            raise ValueError(
                "Shared multi-root initialization currently requires an edge center."
            )
        initial_center = 0 if center == nsites - 1 else nsites - 1
        root_sites = [
            mixed_canonicalize_sites(
                root.tensors,
                initial_center,
                max_bond=None,
                cutoff=0.0,
                max_bond_mode=max_bond_mode,
                bond_coupling=bond_coupling,
                retain_sector_topology=True,
            )
            for root in self.roots
        ]
        root_sites = _align_shared_root_chains(root_sites)
        bonds = (
            range(nsites - 1)
            if center == nsites - 1
            else range(nsites - 2, -1, -1)
        )
        absorb = "right" if center == nsites - 1 else "left"
        for bond in bonds:
            merged = [
                merge_mps_sites(sites[bond], sites[bond + 1])
                for sites in root_sites
            ]
            _left, _right, _singular, _error, _kept, pairs = (
                state_averaged_svd_two_site(
                    merged,
                    self.weights,
                    max_bond=max_bond,
                    cutoff=0.0,
                    absorb=absorb,
                    bond_coupling=bond_coupling,
                    max_bond_mode=max_bond_mode,
                    retain_sector_topology=True,
                )
            )
            for root, (left, right) in enumerate(pairs):
                root_sites[root][bond] = left
                root_sites[root][bond + 1] = right
        self.roots = [
            MPS.from_tensors(
                sites,
                center=center,
                target_sector=self.target_sector,
            )
            for sites in root_sites
        ]
        self.center = center
        self.center_bond = None
        self.center_tensor = None
        return self
