#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced non-Abelian tensor storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import itertools

import numpy as np

from .coupling import (
    CouplingChannel,
    ReducedBondSpace,
    enumerate_sector_couplings,
    normalize_coupling_scheme,
    recoupling_matrix,
    reduced_bond_space,
)
from pyqed.mps.symmetry import Sector


def _is_nonabelian_sector(sector):
    return isinstance(sector, Sector) and not sector.is_abelian


@dataclass(frozen=True)
class FusionPipeEntry:
    """
    Packed-layout entry for one child-sector tuple inside a fused sector block.
    """

    child_sectors: tuple[Sector, ...]
    fused_sector: Sector
    slot: int
    offset: int
    local_dim: int
    selected_shape: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "child_sectors", tuple(self.child_sectors))
        object.__setattr__(self, "selected_shape", tuple(int(x) for x in self.selected_shape))
        if self.local_dim <= 0:
            raise ValueError(f"FusionPipeEntry local_dim must be positive, got {self.local_dim}.")


@dataclass(frozen=True)
class FusionPipe:
    """
    Fixed-layout packing map for a fused leg.

    This is the non-Abelian analogue of a simple leg-combination map: it records
    how child-sector tuples are packed into the reduced fused axis for each fused
    sector.
    """

    child_legs: tuple[int, ...]
    child_sector_lists: tuple[tuple[Sector, ...], ...]
    child_dirs: tuple[int, ...]
    fused_sectors: tuple[Sector, ...]
    entries: tuple[FusionPipeEntry, ...]
    orientation: int = 1
    coupling: str = "fixed"
    selected_channel: Sector | None = None

    def __post_init__(self):
        child_legs = tuple(self.child_legs)
        child_sector_lists = tuple(tuple(sectors) for sectors in self.child_sector_lists)
        child_dirs = tuple(self.child_dirs)
        fused_sectors = tuple(self.fused_sectors)
        entries = tuple(self.entries)
        if self.orientation not in (-1, 1):
            raise ValueError(f"FusionPipe orientation must be +/-1, got {self.orientation!r}.")
        if len(child_sector_lists) != len(child_legs):
            raise ValueError("FusionPipe child_sector_lists length must match child_legs.")
        if len(child_dirs) != len(child_legs):
            raise ValueError("FusionPipe child_dirs length must match child_legs.")

        seen = {}
        fused_offsets = {}
        for entry in entries:
            key = (entry.child_sectors, entry.fused_sector, entry.slot)
            if key in seen:
                raise ValueError(f"Duplicate FusionPipe entry for {key!r}.")
            seen[key] = entry
            if entry.fused_sector not in fused_sectors:
                raise ValueError(
                    f"FusionPipe entry uses fused sector {entry.fused_sector!r} "
                    f"outside declared fused sectors {fused_sectors!r}."
                )
            fused_offsets.setdefault(entry.fused_sector, []).append((entry.offset, entry.local_dim))
        for fused_sector, spans in fused_offsets.items():
            spans = sorted(spans)
            offset = 0
            for start, local_dim in spans:
                if start != offset:
                    raise ValueError(
                        f"FusionPipe entries for fused sector {fused_sector!r} are not tightly packed: "
                        f"expected offset {offset}, got {start}."
                    )
                offset += local_dim

        object.__setattr__(self, "child_legs", child_legs)
        object.__setattr__(self, "child_sector_lists", child_sector_lists)
        object.__setattr__(self, "child_dirs", child_dirs)
        object.__setattr__(self, "fused_sectors", fused_sectors)
        object.__setattr__(self, "entries", entries)

    @classmethod
    def from_entries(
        cls,
        *,
        child_legs,
        child_sector_lists,
        child_dirs,
        fused_sectors,
        entries,
        orientation=1,
        coupling="fixed",
        selected_channel=None,
    ):
        return cls(
            child_legs=tuple(child_legs),
            child_sector_lists=tuple(tuple(sectors) for sectors in child_sector_lists),
            child_dirs=tuple(child_dirs),
            fused_sectors=tuple(fused_sectors),
            entries=tuple(entries),
            orientation=orientation,
            coupling=coupling,
            selected_channel=selected_channel,
        )

    def entries_for_sector(self, fused_sector):
        return tuple(entry for entry in self.entries if entry.fused_sector == fused_sector)

    def entry_for_child_sectors(self, child_sectors, fused_sector=None):
        child_sectors = tuple(child_sectors)
        matches = [
            entry
            for entry in self.entries
            if entry.child_sectors == child_sectors
            and (fused_sector is None or entry.fused_sector == fused_sector)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise ValueError(f"No FusionPipe entry registered for child sectors {child_sectors!r}.")
        raise ValueError(
            f"Ambiguous FusionPipe entries for child sectors {child_sectors!r}; "
            "specify fused_sector and/or slot."
        )

    def total_dim(self, fused_sector):
        return sum(entry.local_dim for entry in self.entries_for_sector(fused_sector))


@dataclass(frozen=True)
class FusionLeg:
    """
    Metadata for a fused non-Abelian tensor leg.

    Parameters
    ----------
    child_legs
        Indices of the legs fused into this output leg.
    child_sector_lists
        Allowed sectors on each child leg in the selected coupling scheme.
    child_dirs
        Directions of the child legs before fusion.
    sectors
        Allowed fused sectors carried by this leg.
    orientation
        Leg orientation, ``+1`` for outgoing and ``-1`` for incoming.
    coupling
        A lightweight description of the parenthesization / fusion scheme.
        For now this is a string tag such as ``"fixed"``.
    coupling_channels
        Explicit coupling channels for the chosen fusion order.  These preserve
        multiplicity when the same fused sector is reachable through multiple
        intermediate-spin paths.
    fusion_map
        Optional immutable mapping entries describing how child-sector tuples are
        assigned to fused sectors or reduced blocks.  The structure is intentionally
        generic for now and can be tightened later.
    selected_channel
        Optional selected fusion channel used by fixed-layout helpers.
    """

    child_legs: tuple[int, ...]
    child_sector_lists: tuple[tuple[Sector, ...], ...] = ()
    child_dirs: tuple[int, ...] = ()
    sectors: tuple[Sector, ...] = ()
    orientation: int = 1
    coupling: str = "fixed"
    coupling_channels: tuple[CouplingChannel, ...] = ()
    fusion_map: tuple[tuple[tuple[Sector, ...], Sector], ...] = ()
    selected_channel: Sector | None = None
    pipe: FusionPipe | None = None
    _bond_space_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _recoupling_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        child_legs = tuple(self.child_legs)
        child_sector_lists = tuple(tuple(sectors) for sectors in self.child_sector_lists)
        child_dirs = tuple(self.child_dirs)
        sectors = tuple(self.sectors)
        coupling_channels = tuple(self.coupling_channels)
        fusion_map = tuple(self.fusion_map)
        pipe = self.pipe
        if self.orientation not in (-1, 1):
            raise ValueError(f"FusionLeg orientation must be +/-1, got {self.orientation!r}.")
        if child_dirs and len(child_dirs) != len(child_legs):
            raise ValueError("FusionLeg child_dirs length must match child_legs length.")
        if child_sector_lists and len(child_sector_lists) != len(child_legs):
            raise ValueError("FusionLeg child_sector_lists length must match child_legs length.")
        object.__setattr__(self, "child_legs", child_legs)
        object.__setattr__(self, "child_sector_lists", child_sector_lists)
        object.__setattr__(self, "child_dirs", child_dirs)
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "coupling_channels", coupling_channels)
        object.__setattr__(self, "fusion_map", fusion_map)
        if pipe is not None:
            if pipe.child_legs != child_legs:
                raise ValueError("FusionLeg pipe child_legs must match FusionLeg child_legs.")
            if child_sector_lists and pipe.child_sector_lists != child_sector_lists:
                raise ValueError("FusionLeg pipe child_sector_lists must match FusionLeg child_sector_lists.")
            if child_dirs and pipe.child_dirs != child_dirs:
                raise ValueError("FusionLeg pipe child_dirs must match FusionLeg child_dirs.")
            if sectors and pipe.fused_sectors != sectors:
                raise ValueError("FusionLeg pipe fused_sectors must match FusionLeg sectors.")
            if pipe.orientation != self.orientation:
                raise ValueError("FusionLeg pipe orientation must match FusionLeg orientation.")

    @property
    def parents(self):
        """Compatibility alias for older ``FusionEdge`` naming."""
        return self.child_legs

    @property
    def channel(self):
        """Compatibility alias for older ``FusionEdge`` naming."""
        return self.selected_channel

    @classmethod
    def from_edge(cls, parents, channel=None, orientation=1):
        """Compatibility constructor mirroring the old ``FusionEdge`` API."""
        return cls(tuple(parents), orientation=orientation, selected_channel=channel)

    def with_pipe(self, pipe):
        """
        Return a copy of this leg carrying an explicit packing map.
        """
        return replace(self, pipe=pipe)

    @classmethod
    def from_children(
        cls,
        child_legs,
        child_sector_lists,
        *,
        child_dirs=None,
        orientation=1,
        coupling="fixed",
        selected_channel=None,
    ):
        """
        Build a fused leg from child-leg sector lists.

        The current implementation keeps a fixed coupling order and records every
        allowed child-sector tuple together with the compatible fused sectors.
        If `selected_channel` is given, only that fused sector is kept for
        ambiguous multiplets.
        """
        child_legs = tuple(child_legs)
        child_sector_lists = tuple(tuple(sectors) for sectors in child_sector_lists)
        if child_dirs is None:
            child_dirs = (orientation,) * len(child_legs)
        child_dirs = tuple(child_dirs)

        if len(child_legs) == 0:
            raise ValueError("FusionLeg.from_children requires at least one child leg.")
        if len(child_legs) != len(child_sector_lists):
            raise ValueError("child_legs and child_sector_lists must have matching lengths.")
        if len(child_dirs) != len(child_legs):
            raise ValueError("child_dirs must match the number of child legs.")
        coupling_scheme = normalize_coupling_scheme(coupling, default="left")

        coupling_channels = []
        fusion_entries = []
        fused_sector_set = set()

        for child_combo in itertools.product(*child_sector_lists):
            combo_channels = enumerate_sector_couplings(child_combo, scheme=coupling_scheme)
            if selected_channel is not None:
                combo_channels = tuple(
                    channel for channel in combo_channels if channel.fused_sector == selected_channel
                )
            for channel in combo_channels:
                coupling_channels.append(channel)
                fusion_entries.append((tuple(child_combo), channel.fused_sector))
                fused_sector_set.add(channel.fused_sector)

        return cls(
            child_legs=child_legs,
            child_sector_lists=child_sector_lists,
            child_dirs=child_dirs,
            sectors=tuple(sorted(fused_sector_set)),
            orientation=orientation,
            coupling=coupling,
            coupling_channels=tuple(coupling_channels),
            fusion_map=tuple(fusion_entries),
            selected_channel=selected_channel,
        )

    @property
    def coupling_scheme(self):
        return normalize_coupling_scheme(self.coupling, default="left")

    def candidate_sectors(self, child_sectors):
        child_sectors = tuple(child_sectors)
        if self.coupling_channels:
            return tuple(
                channel.fused_sector
                for channel in self.coupling_channels
                if channel.child_sectors == child_sectors
            )
        return tuple(
            fused_sector
            for combo, fused_sector in self.fusion_map
            if combo == child_sectors
        )

    def channels_for(self, child_sectors, fused_sector=None):
        child_sectors = tuple(child_sectors)
        if self.coupling_channels:
            return tuple(
                channel
                for channel in self.coupling_channels
                if channel.child_sectors == child_sectors
                and (fused_sector is None or channel.fused_sector == fused_sector)
            )
        channels = []
        slot_counts = {}
        for combo, fused in self.fusion_map:
            if combo != child_sectors:
                continue
            if fused_sector is not None and fused != fused_sector:
                continue
            slot = slot_counts.get(fused, 0)
            slot_counts[fused] = slot + 1
            channels.append(
                CouplingChannel(
                    child_sectors=tuple(combo),
                    fused_sector=fused,
                    intermediate_sectors=(fused,),
                    slot=slot,
                )
            )
        return tuple(channels)

    def bond_space(self, child_sectors, fused_sector=None, *, scheme=None):
        child_sectors = tuple(child_sectors)
        if fused_sector is None:
            fused_sector = self.resolve_sector(child_sectors)
        if scheme is None:
            scheme = self.coupling_scheme
        key = (child_sectors, fused_sector, normalize_coupling_scheme(scheme, default=self.coupling_scheme))
        cache = self._bond_space_cache
        bond_space = cache.get(key)
        if bond_space is None:
            bond_space = reduced_bond_space(child_sectors, fused_sector, scheme=key[2])
            cache[key] = bond_space
        return bond_space

    def recoupling_matrix(self, child_sectors, fused_sector=None, *, source_scheme=None, target_scheme="right"):
        child_sectors = tuple(child_sectors)
        if fused_sector is None:
            fused_sector = self.resolve_sector(child_sectors)
        if source_scheme is None:
            source_scheme = self.coupling_scheme
        source_scheme = normalize_coupling_scheme(source_scheme, default=self.coupling_scheme)
        target_scheme = normalize_coupling_scheme(target_scheme, default="right")
        key = (child_sectors, fused_sector, source_scheme, target_scheme)
        cache = self._recoupling_cache
        matrix = cache.get(key)
        if matrix is None:
            matrix = recoupling_matrix(
                child_sectors,
                fused_sector,
                source_scheme=source_scheme,
                target_scheme=target_scheme,
            )
            cache[key] = matrix
        return matrix

    def resolve_sector(self, child_sectors):
        child_sectors = tuple(child_sectors)
        candidates = self.candidate_sectors(child_sectors)
        if self.selected_channel is not None and self.selected_channel in candidates:
            return self.selected_channel
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            raise ValueError(f"No fused sector registered for child sectors {child_sectors!r}.")
        raise ValueError(
            f"Ambiguous fused sector for child sectors {child_sectors!r}: {candidates!r}. "
            "Specify selected_channel or a more explicit fusion map."
        )

    def slot_for(self, child_sectors, fused_sector=None, channel_index=None):
        child_sectors = tuple(child_sectors)
        if fused_sector is None:
            fused_sector = self.resolve_sector(child_sectors)
        channels = self.channels_for(child_sectors, fused_sector)
        if len(channels) == 0:
            raise ValueError(
                f"Child sectors {child_sectors!r} not registered in fusion map for fused sector {fused_sector!r}."
            )
        if channel_index is not None:
            if channel_index < 0 or channel_index >= len(channels):
                raise ValueError(
                    f"Invalid channel_index {channel_index} for child sectors {child_sectors!r} "
                    f"and fused sector {fused_sector!r}."
                )
            return channels[channel_index].slot
        if len(channels) != 1:
            raise ValueError(
                f"Ambiguous slot for child sectors {child_sectors!r} and fused sector {fused_sector!r}; "
                f"available channels: {tuple(channel.slot for channel in channels)!r}."
            )
        return channels[0].slot

    def child_combinations(self, fused_sector=None):
        if self.coupling_channels:
            if fused_sector is None:
                return tuple(channel.child_sectors for channel in self.coupling_channels)
            return tuple(
                channel.child_sectors
                for channel in self.coupling_channels
                if channel.fused_sector == fused_sector
            )
        if fused_sector is None:
            return tuple(combo for combo, _ in self.fusion_map)
        return tuple(combo for combo, fused in self.fusion_map if fused == fused_sector)

class FusionEdge(FusionLeg):
    """Backward-compatible wrapper around :class:`FusionLeg`."""

    def __init__(self, parents, channel=None, orientation=1):
        super().__init__(tuple(parents), orientation=orientation, selected_channel=channel)


def _fuse_sector_combo(sectors):
    sectors = tuple(sectors)
    if len(sectors) == 0:
        return tuple()
    current = (sectors[0],)
    for sector in sectors[1:]:
        next_current = []
        for left in current:
            next_current.extend(left.fuse(sector))
        current = tuple(sorted(set(next_current)))
    return current


class NonabelianTensor:
    """
    Block-sparse reduced tensor carrying generic non-Abelian sector metadata.

    Parameters
    ----------
    data
        Dict mapping sector tuples to dense reduced blocks.
    qns
        Per-leg allowed sector lists.
    dirs
        Per-leg directions (+1 outgoing, -1 incoming).
    fusion_legs
        Optional per-leg fusion-leg metadata. Entries may be ``None``.
    metadata
        Optional free-form dict with implementation-specific annotations.
    """

    def __init__(self, data, qns, dirs, fusion_legs=None, metadata=None, fusion_edges=None):
        self.data = {tuple(key): np.asarray(value) for key, value in data.items()}
        self.qns = [list(leg_qns) for leg_qns in qns]
        self.dirs = list(dirs)
        self.rank = len(self.dirs)

        if len(self.qns) != self.rank:
            raise ValueError(f"qns/dirs length mismatch: {len(self.qns)} vs {self.rank}")

        if fusion_legs is not None and fusion_edges is not None:
            raise ValueError("Specify only one of fusion_legs or fusion_edges.")
        if fusion_legs is None:
            fusion_legs = fusion_edges
        if fusion_legs is None:
            fusion_legs = [None] * self.rank
        self.fusion_legs = list(fusion_legs)
        if len(self.fusion_legs) != self.rank:
            raise ValueError(
                f"fusion_legs/dirs length mismatch: {len(self.fusion_legs)} vs {self.rank}"
            )

        self.metadata = dict(metadata or {})
        self._validate()

    def _validate(self):
        sector_sets = [set(leg_qns) for leg_qns in self.qns]
        for key, block in self.data.items():
            if len(key) != self.rank:
                raise ValueError(
                    f"Block key rank mismatch: expected {self.rank}, got {len(key)} for key {key!r}."
                )
            for leg, sector in enumerate(key):
                if sector not in sector_sets[leg]:
                    raise ValueError(
                        f"Sector {sector!r} on leg {leg} is not present in declared leg sectors {self.qns[leg]!r}."
                    )
            if block.ndim != self.rank:
                raise ValueError(
                    f"Reduced block for key {key!r} has ndim {block.ndim}; expected {self.rank}."
                )

    @property
    def shape(self):
        return tuple(len(leg_qns) for leg_qns in self.qns)

    @property
    def blocks(self):
        """
        Compatibility alias for the block dictionary.

        Internally, site/operator tensors store reduced blocks in ``data``.
        ``blocks`` is a clearer public name and matches the surrounding reduced
        linear-algebra helpers.
        """
        return self.data

    @property
    def nblocks(self):
        return len(self.data)

    @property
    def has_nonabelian_symmetry(self):
        return any(_is_nonabelian_sector(sector) for leg_qns in self.qns for sector in leg_qns)

    def copy(self):
        return NonabelianTensor(
            {key: block.copy() for key, block in self.data.items()},
            [leg_qns[:] for leg_qns in self.qns],
            self.dirs[:],
            fusion_legs=self.fusion_legs[:],
            metadata=self.metadata.copy(),
        )

    def _check_compatible(self, other):
        if not isinstance(other, NonabelianTensor):
            raise TypeError(f"Expected NonabelianTensor, got {type(other).__name__}.")
        if self.qns != other.qns or self.dirs != other.dirs or self.fusion_legs != other.fusion_legs:
            raise ValueError("NonabelianTensor metadata mismatch.")

    def __add__(self, other):
        self._check_compatible(other)
        new_data = {key: block.copy() for key, block in self.data.items()}
        for key, block in other.data.items():
            if key in new_data:
                new_data[key] = new_data[key] + block
            else:
                new_data[key] = block.copy()
        return NonabelianTensor(
            new_data,
            [leg_qns[:] for leg_qns in self.qns],
            self.dirs[:],
            fusion_legs=self.fusion_legs[:],
            metadata=self.metadata.copy(),
        )

    def __sub__(self, other):
        self._check_compatible(other)
        new_data = {key: block.copy() for key, block in self.data.items()}
        for key, block in other.data.items():
            if key in new_data:
                new_data[key] = new_data[key] - block
            else:
                new_data[key] = -block
        return NonabelianTensor(
            new_data,
            [leg_qns[:] for leg_qns in self.qns],
            self.dirs[:],
            fusion_legs=self.fusion_legs[:],
            metadata=self.metadata.copy(),
        )

    def __mul__(self, scalar):
        return NonabelianTensor(
            {key: block * scalar for key, block in self.data.items()},
            [leg_qns[:] for leg_qns in self.qns],
            self.dirs[:],
            fusion_legs=self.fusion_legs[:],
            metadata=self.metadata.copy(),
        )

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1.0 / scalar)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = tuple(axes[0])
        if sorted(axes) != list(range(self.rank)):
            raise ValueError(f"Invalid transpose axes {axes!r} for rank-{self.rank} tensor.")

        new_qns = [self.qns[i][:] for i in axes]
        new_dirs = [self.dirs[i] for i in axes]
        new_fusion_legs = [self.fusion_legs[i] for i in axes]
        new_data = {}
        for key, block in self.data.items():
            new_key = tuple(key[i] for i in axes)
            new_data[new_key] = np.transpose(block, axes)
        return NonabelianTensor(
            new_data,
            new_qns,
            new_dirs,
            fusion_legs=new_fusion_legs,
            metadata=self.metadata.copy(),
        )

    def conj(self):
        return NonabelianTensor(
            {key: block.conj() for key, block in self.data.items()},
            [leg_qns[:] for leg_qns in self.qns],
            [-direction for direction in self.dirs],
            fusion_legs=self.fusion_legs[:],
            metadata=self.metadata.copy(),
        )

    def __repr__(self):
        return (
            f"NonabelianTensor(rank={self.rank}, blocks={len(self.data)}, "
            f"nonabelian={self.has_nonabelian_symmetry})"
        )

    @property
    def fusion_edges(self):
        """Compatibility alias for the previous attribute name."""
        return self.fusion_legs
