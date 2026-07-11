#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State-averaged multi-root MPS owner for non-Abelian sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mps import MPS
from .tensor import NonabelianTensor


_ROOT_AXIS_SECTOR = ("state_average_root", 0)


def fuse_root_center_tensors(center_tensors, *, root_sector=_ROOT_AXIS_SECTOR):
    """Stack rank-4 root center tensors behind one leading root axis."""

    tensors = [tensor.copy() for tensor in center_tensors or []]
    if not tensors:
        return None
    first = tensors[0]
    if not isinstance(first, NonabelianTensor) or first.rank != 4:
        raise ValueError("Root center tensors must be rank-4 NonabelianTensor objects.")
    qns = [leg[:] for leg in first.qns]
    dirs = first.dirs[:]
    fusion_legs = first.fusion_legs[:]
    for tensor in tensors[1:]:
        if not isinstance(tensor, NonabelianTensor) or tensor.rank != 4:
            raise ValueError("Root center tensors must be rank-4 NonabelianTensor objects.")
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
    return NonabelianTensor(
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
    if not isinstance(center_tensor, NonabelianTensor) or center_tensor.rank != 5:
        raise ValueError("Fused root center tensor must be a rank-5 NonabelianTensor.")
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
        NonabelianTensor(
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
    center_tensor: NonabelianTensor | None = None
    center_tensors: list[NonabelianTensor] | None = None
    target_sector: object | None = None

    def __post_init__(self):
        self.roots = [MPS.from_sites(root) for root in self.roots]
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
            if not isinstance(self.center_tensor, NonabelianTensor) or self.center_tensor.rank != 5:
                raise ValueError("MultiRootMPS center_tensor must be a fused rank-5 NonabelianTensor.")
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
            MPS.from_sites(root, center=center, target_sector=target_sector)
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
        return self.roots[0].sites

    def root_site_lists(self):
        return [[site.copy() for site in root.sites] for root in self.roots]

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
