#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Renormalized local operator storage for non-Abelian DMRG.

The classes in this module describe standard local Hamiltonian problems in an
orthonormal reduced basis.  They intentionally do not solve eigenproblems; the
solver consumes these objects while the environment layer builds and caches
them.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time

import numpy as np

_ORTHONORMAL_BLOCK_DENSE_MATVEC_MAX_ELEMENTS = 1_000_000
_COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS = 65536


def set_complementary_family_native_kernel_max_elements(value):
    """
    Set the dense-kernel threshold for family-table native matvecs.

    :param value: Maximum dense kernel elements. Use ``0`` to disable dense
        materialization and force factor-native contractions.
    :returns: The updated threshold.
    """

    global _COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS
    _COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS = max(0, int(value))
    return int(_COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS)


@dataclass
class MovingEnvironmentContractionCache:
    """
    Bounded cache for block2-like moving-environment contraction plans.

    The cache stores structural artifacts used while constructing local
    effective operators.  It intentionally does not own MPS-dependent numeric
    tensors, so entries may be reused across sweeps when the sector structure
    recurs.

    :param max_size: Maximum number of cached structural plans.
    """

    max_size: int = 512
    entries: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0
    puts: int = 0

    def get(self, key):
        """
        Return a cached structural artifact.

        :param key: Hashable structural key.
        :returns: Cached value or ``None``.
        """

        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        value = self.entries.pop(key)
        self.entries[key] = value
        return value

    def put(self, key, value):
        """
        Store a structural artifact.

        :param key: Hashable structural key.
        :param value: Cached artifact.
        :returns: Stored value.
        """

        if key in self.entries:
            self.entries.pop(key)
        elif len(self.entries) >= int(self.max_size):
            self.entries.popitem(last=False)
        self.entries[key] = value
        self.puts += 1
        return value

    @property
    def stats(self):
        """
        Return cache diagnostics.

        :returns: Dictionary with hit/miss/put counts and occupancy.
        """

        return {
            "kind": "moving_environment_contraction_cache",
            "size": int(len(self.entries)),
            "max_size": int(self.max_size),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "puts": int(self.puts),
        }


def factorize_left_two_site_dense_term(E, W1):
    """
    Precontract a left environment block with an adjacent MPO block.

    :param E: Left renormalized operator block.
    :param W1: Left active-site MPO block.
    :returns: Dense left factor tensor.
    """

    return np.tensordot(np.asarray(E), np.asarray(W1), axes=([0], [0]))


def factorize_right_two_site_dense_term(W2, F):
    """
    Precontract a right environment block with an adjacent MPO block.

    :param W2: Right active-site MPO block.
    :param F: Right renormalized operator block.
    :returns: Dense right factor tensor.
    """

    right = np.tensordot(np.asarray(W2), np.asarray(F), axes=([1], [0]))
    return np.transpose(right, (0, 3, 4, 1, 2))


def group_mpo_blocks_by_input(W, phys_slices):
    """
    Group dense or block-sparse MPO blocks by physical input sector.

    :param W: MPO core.
    :param phys_slices: Physical-sector slices for dense MPO cores.
    :returns: Mapping from physical input sector to output-sector blocks.
    """

    from .mpo import IrreducibleMPO, MPO, RankCoupledMPO

    grouped = {}
    if isinstance(W, (MPO, IrreducibleMPO, RankCoupledMPO)):
        phys_in_sectors = getattr(W, "phys_in_sectors", None)
        phys_out_sectors = getattr(W, "phys_out_sectors", None)
        if phys_in_sectors is None:
            phys_in_sectors = W.phys_in_leg.sectors
        if phys_out_sectors is None:
            phys_out_sectors = W.phys_out_leg.sectors
        for q_in in phys_in_sectors:
            entries = []
            for q_out in phys_out_sectors:
                block = W.block(q_out, q_in)
                if block is not None:
                    entries.append((q_out, np.asarray(block)))
            grouped[q_in] = tuple(entries)
        return grouped
    for q_in, p_in in phys_slices.items():
        entries = []
        for q_out, p_out in phys_slices.items():
            block = np.asarray(W[:, :, p_out, p_in])
            if np.any(block != 0):
                entries.append((q_out, block))
        grouped[q_in] = tuple(entries)
    return grouped


def group_rank_coupled_reduced_blocks_by_input(W):
    """
    Group rank-coupled reduced MPO blocks by physical input sector.

    :param W: Rank-coupled MPO core.
    :returns: Mapping from physical input sector to reduced block entries.
    """

    grouped = {}
    for q_in in W.phys_in_leg.sectors:
        entries = []
        for q_out in W.phys_out_leg.sectors:
            reduced = W.reduced_block(q_out, q_in)
            if reduced:
                entries.append((q_out, reduced))
        grouped[q_in] = tuple(entries)
    return grouped


def build_left_factor_table(left_blocks_by_ket, W, phys_slices):
    """
    Build a dense-MPO left factor table from grouped boundary blocks.

    :param left_blocks_by_ket: Left boundary payloads grouped by ket sector.
    :param W: Adjacent MPO core.
    :param phys_slices: Physical-sector slices for dense MPO cores.
    :returns: Factor table grouped by ``(left_ket, physical_ket)``.
    """

    w_blocks_by_in = group_mpo_blocks_by_input(W, phys_slices)
    out = {}
    cache = {}
    for q_lk, left_entries in left_blocks_by_ket.items():
        for q_p1k, w_entries in w_blocks_by_in.items():
            values = []
            for q_lb, E_block in left_entries:
                for q_p1b, W_slice in w_entries:
                    key = (id(E_block), id(W_slice))
                    factor = cache.get(key)
                    if factor is None:
                        factor = np.asarray(factorize_left_two_site_dense_term(E_block, W_slice))
                        cache[key] = factor
                    values.append((q_lb, q_p1b, factor))
            if values:
                out[(q_lk, q_p1k)] = tuple(values)
    return out


def build_right_factor_table(right_blocks_by_ket, W, phys_slices):
    """
    Build a dense-MPO right factor table from grouped boundary blocks.

    :param right_blocks_by_ket: Right boundary payloads grouped by ket sector.
    :param W: Adjacent MPO core.
    :param phys_slices: Physical-sector slices for dense MPO cores.
    :returns: Factor table grouped by ``(right_ket, physical_ket)``.
    """

    w_blocks_by_in = group_mpo_blocks_by_input(W, phys_slices)
    out = {}
    cache = {}
    for q_rk, right_entries in right_blocks_by_ket.items():
        for q_p2k, w_entries in w_blocks_by_in.items():
            values = []
            for q_rb, F_block in right_entries:
                for q_p2b, W_slice in w_entries:
                    key = (id(W_slice), id(F_block))
                    factor = cache.get(key)
                    if factor is None:
                        factor = np.asarray(factorize_right_two_site_dense_term(W_slice, F_block))
                        cache[key] = factor
                    values.append((q_rb, q_p2b, factor))
            if values:
                out[(q_rk, q_p2k)] = tuple(values)
    return out


def build_rank_coupled_left_factor_table(left_blocks_by_ket, W):
    """
    Build a rank-coupled left factor table from grouped boundary payloads.

    :param left_blocks_by_ket: Left boundary payloads grouped by ket sector.
    :param W: Rank-coupled MPO core.
    :returns: Factor table grouped by ``(left_ket, physical_ket)``.
    """

    w_blocks_by_in = group_rank_coupled_reduced_blocks_by_input(W)
    family_by_middle = _symbolic_transition_families_by_channel(W, side="right")
    out = {}
    cache = {}
    for q_lk, left_entries in left_blocks_by_ket.items():
        for q_p1k, w_entries in w_blocks_by_in.items():
            values = []
            for q_lb, E_entries in left_entries:
                for q_p1b, W_blocks in w_entries:
                    for (left_idx, middle_idx), W_block in W_blocks.items():
                        E_block = E_entries.get(left_idx)
                        if E_block is None:
                            continue
                        key = (id(E_block), id(W_block))
                        factor = cache.get(key)
                        if factor is None:
                            factor = np.asarray(factorize_left_two_site_dense_term(E_block, W_block))
                            cache[key] = factor
                        values.append(
                            (
                                q_lb,
                                q_p1b,
                                middle_idx,
                                factor,
                                family_by_middle.get(int(middle_idx), ()),
                            )
                        )
            if values:
                out[(q_lk, q_p1k)] = tuple(values)
    return out


def build_rank_coupled_right_factor_table(right_blocks_by_ket, W):
    """
    Build a rank-coupled right factor table from grouped boundary payloads.

    :param right_blocks_by_ket: Right boundary payloads grouped by ket sector.
    :param W: Rank-coupled MPO core.
    :returns: Factor table grouped by ``(right_ket, physical_ket)``.
    """

    w_blocks_by_in = group_rank_coupled_reduced_blocks_by_input(W)
    family_by_middle = _symbolic_transition_families_by_channel(W, side="left")
    out = {}
    cache = {}
    for q_rk, right_entries in right_blocks_by_ket.items():
        for q_p2k, w_entries in w_blocks_by_in.items():
            values = []
            for q_rb, F_entries in right_entries:
                for q_p2b, W_blocks in w_entries:
                    for (middle_idx, right_idx), W_block in W_blocks.items():
                        F_block = F_entries.get(right_idx)
                        if F_block is None:
                            continue
                        key = (id(W_block), id(F_block))
                        factor = cache.get(key)
                        if factor is None:
                            factor = np.asarray(factorize_right_two_site_dense_term(W_block, F_block))
                            cache[key] = factor
                        values.append(
                            (
                                q_rb,
                                q_p2b,
                                middle_idx,
                                factor,
                                family_by_middle.get(int(middle_idx), ()),
                            )
                        )
            if values:
                out[(q_rk, q_p2k)] = tuple(values)
    return out


def _family_names_from_symbolic_label(label):
    """Return family labels embedded in an AutoMPO symbolic transition label."""

    if not isinstance(label, tuple) or not label:
        return ()
    if all(isinstance(item, str) for item in label):
        return tuple(sorted({str(item) for item in label if item}))
    candidate = label[-1]
    if isinstance(candidate, str):
        return (candidate,)
    if isinstance(candidate, (tuple, list, set)):
        return tuple(sorted({str(item) for item in candidate if item is not None}))
    return ()


def _symbolic_transition_families_by_channel(W, *, side):
    """
    Group symbolic transition family labels by one visible MPO channel.

    :param W: MPO core carrying ``symbolic_transitions`` metadata.
    :param side: ``"left"`` groups by incoming channel; ``"right"`` groups by
        outgoing channel.
    :returns: Mapping ``channel -> tuple(family labels)``.
    """

    channel_index = 1 if str(side) == "left" else 2
    families = {}
    for record in tuple(getattr(W, "symbolic_transitions", ()) or ()):
        if len(record) < 4:
            continue
        channel = int(record[channel_index])
        names = _family_names_from_symbolic_label(record[3])
        if not names:
            continue
        bucket = families.setdefault(channel, set())
        bucket.update(names)
    return {channel: tuple(sorted(names)) for channel, names in families.items()}


@dataclass(frozen=True)
class SymbolicMPOTransition:
    """
    Symbolic MPO virtual-channel transition used by recursive boundary algebra.

    :param kind: Transition kind, for example ``"dense"`` or ``"reduced"``.
    :param left_channel: Incoming left MPO virtual channel.
    :param right_channel: Outgoing right MPO virtual channel.
    :param label: Stable label describing the local operator carried by the
        transition.
    """

    kind: str
    left_channel: int
    right_channel: int
    label: object

    @property
    def key(self):
        """
        Return a hashable transition key.

        :returns: Tuple suitable for symbolic path storage.
        """

        return (
            str(self.kind),
            int(self.left_channel),
            int(self.right_channel),
            self.label,
        )


@dataclass(frozen=True)
class SymbolicRenormalizedOperatorTerm:
    """
    One symbolic renormalized boundary operator path.

    :param channel: Current MPO virtual channel represented by this boundary
        operator.
    :param path: Ordered tuple of local MPO transition keys absorbed into the
        boundary.
    :param multiplicity: Number of equivalent symbolic paths merged into this
        term.
    """

    channel: int
    path: tuple = ()
    multiplicity: int = 1

    def append(self, transition):
        """
        Return the left-to-right advanced term.

        :param transition: :class:`SymbolicMPOTransition` absorbed on the right.
        :returns: Advanced symbolic term.
        """

        return type(self)(
            channel=int(transition.right_channel),
            path=tuple(self.path) + (transition.key,),
            multiplicity=int(self.multiplicity),
        )

    def prepend(self, transition):
        """
        Return the right-to-left advanced term.

        :param transition: :class:`SymbolicMPOTransition` absorbed on the left.
        :returns: Advanced symbolic term.
        """

        return type(self)(
            channel=int(transition.left_channel),
            path=(transition.key,) + tuple(self.path),
            multiplicity=int(self.multiplicity),
        )


@dataclass(frozen=True)
class SymbolicRenormalizedOperatorTable:
    """
    Recursive symbolic renormalized-operator table for one boundary.

    The table mirrors the block2 view of a boundary: each entry is keyed by a
    visible MPO virtual channel and stores the symbolic MPO path absorbed into
    that renormalized operator.  Numeric environment tensors remain owned by
    :class:`RenormalizedBlockEntry`; this table owns the operator algebra and
    lineage.

    :param side: Boundary side, ``"left"`` or ``"right"``.
    :param bond: Boundary bond index.
    :param terms_by_channel: Mapping from MPO virtual channel to symbolic
        terms.
    :param source: How the table was produced.
    :param parent_key: Optional parent boundary-stack key.
    :param used_mpo_symbolic_metadata: Whether the absorbed MPO core supplied
        preserved AutoMPO symbolic transition records.
    :param numeric_payloads: Numeric renormalized-operator payloads owned by
        this symbolic table.  Entries are keyed by ``(q_out, q_in, channel)``
        for rank-coupled payloads and ``(q_out, q_in, None)`` for ordinary
        block payloads.
    :param payload_kind: Payload layout kind.
    """

    side: str
    bond: int
    terms_by_channel: dict
    source: str = "initialized"
    parent_key: object | None = None
    used_mpo_symbolic_metadata: bool = False
    numeric_payloads: dict = field(default_factory=dict)
    payload_kind: str = "none"

    @classmethod
    def initialize(cls, side, bond, block, *, source="initialized"):
        """
        Build the identity boundary symbolic table.

        :param side: Boundary side.
        :param bond: Boundary bond index.
        :param block: Numeric boundary block used to discover active virtual
            channels.
        :param source: Source label.
        :returns: Initialized symbolic table.
        """

        channels = _active_boundary_channels(block)
        terms = {
            int(channel): (
                SymbolicRenormalizedOperatorTerm(channel=int(channel), path=()),
            )
            for channel in sorted(channels)
        }
        table = cls(
            side=str(side),
            bond=int(bond),
            terms_by_channel=terms,
            source=str(source),
        )
        return table.with_numeric_payload(block)

    def advance_left(self, W, *, bond, block=None, parent_key=None):
        """
        Advance this symbolic table by absorbing an MPO core on the right.

        :param W: MPO core for the absorbed site.
        :param bond: New left-boundary bond.
        :param block: Optional numeric child block for active-channel pruning.
        :param parent_key: Optional parent boundary-stack key.
        :returns: Advanced symbolic table.
        """

        transitions, used_metadata = symbolic_mpo_core_transitions(W)
        active = None if block is None else _active_boundary_channels(block)
        terms_by_channel = _compact_advance_symbolic_terms(
            self.terms_by_channel,
            transitions,
            active=active,
            direction="left",
        )
        table = type(self)(
            side="left",
            bond=int(bond),
            terms_by_channel=terms_by_channel,
            source="advanced_left",
            parent_key=parent_key,
            used_mpo_symbolic_metadata=used_metadata,
        )
        return table if block is None else table.with_numeric_payload(block)

    def advance_right(self, W, *, bond, block=None, parent_key=None):
        """
        Advance this symbolic table by absorbing an MPO core on the left.

        :param W: MPO core for the absorbed site.
        :param bond: New right-boundary bond.
        :param block: Optional numeric child block for active-channel pruning.
        :param parent_key: Optional parent boundary-stack key.
        :returns: Advanced symbolic table.
        """

        transitions, used_metadata = symbolic_mpo_core_transitions(W)
        active = None if block is None else _active_boundary_channels(block)
        terms_by_channel = _compact_advance_symbolic_terms(
            self.terms_by_channel,
            transitions,
            active=active,
            direction="right",
        )
        table = type(self)(
            side="right",
            bond=int(bond),
            terms_by_channel=terms_by_channel,
            source="advanced_right",
            parent_key=parent_key,
            used_mpo_symbolic_metadata=used_metadata,
        )
        return table if block is None else table.with_numeric_payload(block)

    def with_numeric_payload(self, block_map):
        """
        Return a copy of this table owning numeric renormalized payloads.

        :param block_map: Sector-pair keyed numeric boundary block map.
        :returns: Symbolic table carrying numeric boundary payloads.
        """

        payloads, payload_kind = _symbolic_numeric_payloads_from_block_map(
            block_map,
            self.channels,
        )
        return type(self)(
            side=self.side,
            bond=self.bond,
            terms_by_channel=self.terms_by_channel,
            source=self.source,
            parent_key=self.parent_key,
            used_mpo_symbolic_metadata=self.used_mpo_symbolic_metadata,
            numeric_payloads=payloads,
            payload_kind=payload_kind,
        )

    @property
    def channels(self):
        """Return the sorted active MPO virtual channels."""

        return tuple(sorted(int(channel) for channel in self.terms_by_channel))

    @property
    def n_terms(self):
        """Return the multiplicity-counted number of symbolic paths."""

        return int(
            sum(
                int(term.multiplicity)
                for terms in self.terms_by_channel.values()
                for term in terms
            )
        )

    @property
    def max_path_length(self):
        """Return the longest absorbed symbolic MPO path."""

        if not self.terms_by_channel:
            return 0
        return int(
            max(
                len(term.path)
                for terms in self.terms_by_channel.values()
                for term in terms
            )
        )

    @property
    def stats(self):
        """
        Return symbolic table diagnostics.

        :returns: Dictionary with channel, term, and lineage counts.
        """

        return {
            "side": str(self.side),
            "bond": int(self.bond),
            "source": str(self.source),
            "parent_key": self.parent_key,
            "channels": self.channels,
            "n_channels": int(len(self.channels)),
            "n_terms": int(self.n_terms),
            "max_path_length": int(self.max_path_length),
            "used_mpo_symbolic_metadata": bool(self.used_mpo_symbolic_metadata),
            "owns_numeric_payloads": bool(self.numeric_payloads),
            "numeric_payloads": int(len(self.numeric_payloads)),
            "payload_kind": str(self.payload_kind),
        }

    def group_boundary_blocks(self, block_map=None, representation="block_by_ket"):
        """
        Group numeric boundary blocks using this symbolic boundary table.

        The symbolic table owns the active MPO virtual channels.  The numeric
        block map supplies the tensor payloads.  This is the bridge from the
        recursive symbolic operator algebra to the current block-sparse matvec
        kernels.

        :param block_map: Optional sector-pair keyed numeric boundary block map.
            When omitted, the numeric payloads owned by this symbolic table are
            used.
        :param representation: Side-table representation.
        :returns: Boundary blocks grouped by ket sector.
        """

        if block_map is None and self.numeric_payloads:
            return self._group_owned_payloads(representation)
        if block_map is None:
            raise ValueError("group_boundary_blocks requires block_map when no numeric payloads are attached.")
        active_channels = set(self.channels)
        grouped = {}
        if representation == "rank_coupled_by_ket":
            for (q_out, q_in), blocks in block_map.items():
                entries = _nonzero_rank_coupled_blocks_for_channels(
                    blocks,
                    active_channels,
                )
                if entries:
                    grouped.setdefault(q_in, []).append((q_out, dict(entries)))
        elif representation == "array_by_ket":
            for (q_out, q_in), block in block_map.items():
                grouped.setdefault(q_in, []).append((q_out, np.asarray(block)))
        else:
            for (q_out, q_in), block in block_map.items():
                grouped.setdefault(q_in, []).append((q_out, block))
        return {key: tuple(value) for key, value in grouped.items()}

    def _group_owned_payloads(self, representation):
        grouped = {}
        if representation == "rank_coupled_by_ket":
            by_pair = {}
            for (q_out, q_in, channel), block in self.numeric_payloads.items():
                if channel is None:
                    continue
                by_pair.setdefault((q_out, q_in), {})[int(channel)] = block
            for (q_out, q_in), entries in by_pair.items():
                if entries:
                    grouped.setdefault(q_in, []).append((q_out, dict(sorted(entries.items()))))
        elif representation == "array_by_ket":
            for (q_out, q_in, channel), block in self.numeric_payloads.items():
                if channel is None:
                    grouped.setdefault(q_in, []).append((q_out, np.asarray(block)))
        else:
            for (q_out, q_in, channel), block in self.numeric_payloads.items():
                if channel is None:
                    grouped.setdefault(q_in, []).append((q_out, block))
        return {key: tuple(value) for key, value in grouped.items()}

    def factor_boundary_blocks(self, representation, W, *, phys_slices=None):
        """
        Build one-site factor tables from symbolic-owned boundary payloads.

        :param representation: Factor-table representation.
        :param W: MPO core adjacent to this boundary.
        :param phys_slices: Physical sector slices for dense MPO cores.
        :returns: Factor table grouped by ket-sector pair.
        """

        if representation == "left_factor_by_ket":
            return self.left_factor_table(W, phys_slices=phys_slices)
        if representation == "right_factor_by_ket":
            return self.right_factor_table(W, phys_slices=phys_slices)
        if representation == "rank_coupled_left_factor_by_ket":
            return self.rank_coupled_left_factor_table(W)
        if representation == "rank_coupled_right_factor_by_ket":
            return self.rank_coupled_right_factor_table(W)
        raise ValueError(f"Unknown symbolic factor-table representation {representation!r}.")

    def left_factor_table(self, W, *, phys_slices=None):
        """
        Build a left dense-MPO factor table from owned symbolic payloads.

        :param W: MPO core adjacent to this boundary.
        :param phys_slices: Physical sector slices for dense MPO cores.
        :returns: Factor table grouped by ``(left_ket, physical_ket)``.
        """

        return build_left_factor_table(
            self.group_boundary_blocks(representation="array_by_ket"),
            W,
            phys_slices,
        )

    def right_factor_table(self, W, *, phys_slices=None):
        """
        Build a right dense-MPO factor table from owned symbolic payloads.

        :param W: MPO core adjacent to this boundary.
        :param phys_slices: Physical sector slices for dense MPO cores.
        :returns: Factor table grouped by ``(right_ket, physical_ket)``.
        """

        return build_right_factor_table(
            self.group_boundary_blocks(representation="array_by_ket"),
            W,
            phys_slices,
        )

    def rank_coupled_left_factor_table(self, W):
        """
        Build a left rank-coupled factor table from owned symbolic payloads.

        :param W: Rank-coupled MPO core adjacent to this boundary.
        :returns: Factor table grouped by ``(left_ket, physical_ket)``.
        """

        return build_rank_coupled_left_factor_table(
            self.group_boundary_blocks(representation="rank_coupled_by_ket"),
            W,
        )

    def rank_coupled_right_factor_table(self, W):
        """
        Build a right rank-coupled factor table from owned symbolic payloads.

        :param W: Rank-coupled MPO core adjacent to this boundary.
        :returns: Factor table grouped by ``(right_ket, physical_ket)``.
        """

        return build_rank_coupled_right_factor_table(
            self.group_boundary_blocks(representation="rank_coupled_by_ket"),
            W,
        )


@dataclass(frozen=True)
class ComplementaryFamilyRenormalizedOperatorBlock:
    """
    One stored renormalized boundary-operator family block.

    :param family_name: Complementary family label.
    :param channels: MPO virtual channels carrying this family.
    :param symbolic_terms: Multiplicity-counted symbolic path count.
    :param payload_keys: Numeric payload keys owned by these channels.
    :param stored_elements: Number of scalar tensor elements in payloads.
    :param payload_norm: Frobenius norm over owned numeric payload tensors.
    """

    family_name: str
    channels: tuple
    symbolic_terms: int
    payload_keys: tuple
    stored_elements: int
    payload_norm: float
    coefficient_terms: int = 0
    coefficient_cross_terms: int = 0

    @property
    def n_channels(self):
        """Return the number of active virtual channels."""

        return int(len(self.channels))

    @property
    def n_payload_blocks(self):
        """Return the number of numeric payload tensors."""

        return int(len(self.payload_keys))

    @property
    def stats(self):
        """Return compact diagnostics for this family block."""

        return {
            "family_name": str(self.family_name),
            "channels": tuple(int(channel) for channel in self.channels),
            "n_channels": int(self.n_channels),
            "symbolic_terms": int(self.symbolic_terms),
            "n_payload_blocks": int(self.n_payload_blocks),
            "stored_elements": int(self.stored_elements),
            "payload_norm": float(self.payload_norm),
            "coefficient_terms": int(self.coefficient_terms),
            "coefficient_cross_terms": int(self.coefficient_cross_terms),
        }


@dataclass(frozen=True)
class ComplementaryFamilyRenormalizedOperatorTable:
    """
    Stored family-resolved renormalized boundary operator table.

    This table is the persistent block2-style operator-family layer.  It is
    derived from the recursive symbolic boundary table and owns a per-family
    view of the numeric renormalized operator payloads already stored on the
    boundary.

    :param side: Boundary side.
    :param bond: Boundary bond index.
    :param family_blocks: Mapping from family label to stored block metadata.
    :param source: Diagnostic source label.
    """

    side: str
    bond: int
    family_blocks: dict
    source: str = "symbolic_renormalized_operator_table"

    @classmethod
    def from_symbolic_table(cls, symbolic_table, family_names, *, family_payloads=None):
        """
        Build a family table from a symbolic renormalized boundary table.

        :param symbolic_table: Boundary symbolic table with numeric payloads.
        :param family_names: Complementary family labels to expose.
        :param family_payloads: Optional complementary coefficient payloads.
        :returns: Family-resolved renormalized operator table.
        """

        family_names = tuple(str(name) for name in family_names)
        family_payloads = dict(family_payloads or {})
        channel_families = {}
        channel_term_counts = {}
        for channel, terms in symbolic_table.terms_by_channel.items():
            families = set()
            term_count = 0
            for term in terms:
                term_count += int(term.multiplicity)
                for transition_key in tuple(term.path):
                    if len(transition_key) < 4:
                        continue
                    families.update(_family_names_from_symbolic_label(transition_key[3]))
            if families:
                channel_families[int(channel)] = families
                channel_term_counts[int(channel)] = int(term_count)

        payloads_by_channel = {}
        payload_norms_by_channel = {}
        payload_elements_by_channel = {}
        for key, payload in symbolic_table.numeric_payloads.items():
            channel = None if len(key) < 3 else key[2]
            if channel is None:
                continue
            arr = np.asarray(payload)
            payloads_by_channel.setdefault(int(channel), []).append(key)
            payload_norms_by_channel[int(channel)] = (
                payload_norms_by_channel.get(int(channel), 0.0)
                + float(np.linalg.norm(arr)) ** 2
            )
            payload_elements_by_channel[int(channel)] = (
                payload_elements_by_channel.get(int(channel), 0)
                + int(arr.size)
            )

        blocks = {}
        for family in family_names:
            channels = tuple(
                sorted(
                    channel
                    for channel, names in channel_families.items()
                    if family in names
                )
            )
            coefficient_payload = family_payloads.get(family)
            payload_keys = tuple(
                key
                for channel in channels
                for key in payloads_by_channel.get(int(channel), ())
            )
            payload_norm_sq = sum(
                payload_norms_by_channel.get(int(channel), 0.0)
                for channel in channels
            )
            blocks[family] = ComplementaryFamilyRenormalizedOperatorBlock(
                family_name=family,
                channels=channels,
                symbolic_terms=sum(
                    channel_term_counts.get(int(channel), 0)
                    for channel in channels
                ),
                payload_keys=payload_keys,
                stored_elements=sum(
                    payload_elements_by_channel.get(int(channel), 0)
                    for channel in channels
                ),
                payload_norm=float(np.sqrt(payload_norm_sq)),
                coefficient_terms=(
                    0 if coefficient_payload is None else int(coefficient_payload.n_terms)
                ),
                coefficient_cross_terms=(
                    0
                    if coefficient_payload is None
                    else int(coefficient_payload.cross_terms)
                ),
            )
        return cls(
            side=str(symbolic_table.side),
            bond=int(symbolic_table.bond),
            family_blocks=blocks,
        )

    @property
    def family_names(self):
        """Return family labels in this table."""

        return tuple(self.family_blocks)

    @property
    def active_family_names(self):
        """Return family labels with at least one symbolic channel."""

        return tuple(
            name
            for name, block in self.family_blocks.items()
            if block.n_channels > 0
        )

    def active_family_set(self):
        """Return active family labels as a set."""

        return set(self.active_family_names)

    def supports_family_names(self, family_names):
        """
        Return whether this table has an active channel for any label.

        :param family_names: Iterable of family labels from a local term.
        :returns: ``True`` when any label is active in this boundary table.
        """

        active = self.active_family_set()
        return bool(active.intersection(str(name) for name in family_names))

    @property
    def n_channels(self):
        """Return total family-channel assignments."""

        return int(sum(block.n_channels for block in self.family_blocks.values()))

    @property
    def n_payload_blocks(self):
        """Return total numeric payload blocks assigned to families."""

        return int(sum(block.n_payload_blocks for block in self.family_blocks.values()))

    @property
    def stored_elements(self):
        """Return total stored payload tensor elements assigned to families."""

        return int(sum(block.stored_elements for block in self.family_blocks.values()))

    @property
    def symbolic_terms(self):
        """Return total multiplicity-counted symbolic family terms."""

        return int(sum(block.symbolic_terms for block in self.family_blocks.values()))

    @property
    def stats(self):
        """Return compact diagnostics for the family table."""

        return {
            "kind": "complementary_family_renormalized_operator_table",
            "source": str(self.source),
            "side": str(self.side),
            "bond": int(self.bond),
            "family_names": self.family_names,
            "active_family_names": self.active_family_names,
            "n_family_blocks": int(len(self.family_blocks)),
            "n_channels": int(self.n_channels),
            "n_payload_blocks": int(self.n_payload_blocks),
            "stored_elements": int(self.stored_elements),
            "symbolic_terms": int(self.symbolic_terms),
            "families": {
                str(name): block.stats
                for name, block in self.family_blocks.items()
            },
        }


@dataclass(frozen=True)
class FamilyNativeFactorKernel:
    """
    Family-owned factorized local contraction kernel.

    The kernel owns the numerical left/right stacks needed for one block
    contraction.  It is independent of ``CompiledFactorizedBlock.apply_block``
    so family-table matvecs can move toward block2-like native contractions
    while keeping the same tensor algebra.
    """

    left_stack: np.ndarray
    right_stack: np.ndarray
    input_shape: tuple
    output_size: int
    use_direct_contraction: bool

    @classmethod
    def from_compiled_term(cls, term):
        """Build a native factor kernel from a compiled factorized term."""

        return cls(
            left_stack=np.asarray(term.left_stack),
            right_stack=np.asarray(term.right_stack),
            input_shape=tuple(int(dim) for dim in term.input_entry.shape),
            output_size=int(term.output_size),
            use_direct_contraction=bool(
                getattr(term, "_use_direct_contraction", False)
            ),
        )

    @property
    def stored_elements(self):
        """Return the number of scalar elements stored by this kernel."""

        return int(np.asarray(self.left_stack).size + np.asarray(self.right_stack).size)

    def apply_block(self, block_in):
        """
        Apply the factor-native contraction to one input block.

        :param block_in: Input sector block.
        :returns: Flattened output-sector contribution.
        """

        left_stack = np.asarray(self.left_stack)
        right_stack = np.asarray(self.right_stack)
        block_in = np.asarray(block_in)
        if bool(self.use_direct_contraction):
            contrib = np.einsum(
                "tlkwab,kbcr,twqrdc->ladq",
                left_stack,
                block_in,
                right_stack,
                optimize=False,
            )
            return np.asarray(contrib).reshape(int(self.output_size))
        tmp = np.einsum(
            "tlkwab,kbcr->tlwacr",
            left_stack,
            block_in,
            optimize=False,
        )
        contrib = np.einsum(
            "tlwacr,twqrdc->ladq",
            tmp,
            right_stack,
            optimize=False,
        )
        return np.asarray(contrib).reshape(int(self.output_size))


@dataclass(frozen=True)
class ComplementaryFamilyApplyEntry:
    """
    One local application routed by stored family-operator tables.

    The entry already has the component slices needed by the orthonormal
    matvec.  ``compiled_term`` remains the numerical backend until payload
    native contractions replace it.
    """

    in_comp: int
    out_comp: int
    in_slice: slice
    out_slice: slice
    compiled_term: object
    family_names: tuple
    source_tables: tuple = ()
    backend: str = "compiled_factorized_term"
    factor_kernel: FamilyNativeFactorKernel | None = None
    native_kernel: np.ndarray | None = None

    @classmethod
    def from_plan_entry(cls, entry, *, family_names=None, source_tables=()):
        """Build an apply entry from a raw component-direct plan tuple."""

        in_comp, out_comp, in_slice, out_slice, term = entry
        names = tuple(
            str(name)
            for name in (
                family_names
                if family_names is not None
                else (getattr(term, "family_names", ()) or ())
            )
        )
        return cls(
            in_comp=int(in_comp),
            out_comp=int(out_comp),
            in_slice=in_slice,
            out_slice=out_slice,
            compiled_term=term,
            family_names=names,
            source_tables=tuple(source_tables or ()),
        )

    def with_factor_kernel(self):
        """
        Return an entry using a family-native factorized backend.

        :returns: New entry with ``family_table_factor_kernel`` backend.
        """

        return type(self)(
            in_comp=self.in_comp,
            out_comp=self.out_comp,
            in_slice=self.in_slice,
            out_slice=self.out_slice,
            compiled_term=self.compiled_term,
            family_names=self.family_names,
            source_tables=self.source_tables,
            backend="family_table_factor_kernel",
            factor_kernel=FamilyNativeFactorKernel.from_compiled_term(
                self.compiled_term
            ),
            native_kernel=self.native_kernel,
        )

    def with_native_kernel(self, *, max_elements=None):
        """
        Return an entry with a dense family-native kernel when feasible.

        :param max_elements: Maximum dense kernel size to materialize.
        :returns: New entry using ``family_table_dense_kernel`` when available.
        """

        if max_elements is None:
            max_elements = _COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS
        kernel = self.compiled_term.kernel_matrix(
            self.compiled_term.input_entry.shape,
            max_elements=int(max_elements),
        )
        if kernel is None:
            return self
        return type(self)(
            in_comp=self.in_comp,
            out_comp=self.out_comp,
            in_slice=self.in_slice,
            out_slice=self.out_slice,
            compiled_term=self.compiled_term,
            family_names=self.family_names,
            source_tables=self.source_tables,
            backend="family_table_dense_kernel",
            factor_kernel=self.factor_kernel,
            native_kernel=np.ascontiguousarray(kernel),
        )

    @property
    def input_entry(self):
        """Return the compiled term input entry."""

        return self.compiled_term.input_entry

    def apply_block(self, block_in):
        """Apply the current numerical backend to one input block."""

        if self.native_kernel is not None:
            return np.asarray(self.native_kernel @ np.asarray(block_in).reshape(-1))
        if self.factor_kernel is not None:
            return self.factor_kernel.apply_block(block_in)
        return self.compiled_term.apply_block(block_in)

    @property
    def stats(self):
        """Return compact diagnostics for this local application entry."""

        return {
            "family_names": tuple(self.family_names),
            "backend": str(self.backend),
            "source_tables": tuple(str(item) for item in self.source_tables),
            "native_kernel_elements": int(
                0 if self.native_kernel is None else np.asarray(self.native_kernel).size
            ),
            "factor_kernel_elements": int(
                0 if self.factor_kernel is None else self.factor_kernel.stored_elements
            ),
        }


def _active_boundary_channels(block):
    channels = set()
    for value in block.values():
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                arr = np.asarray(item)
                if arr.size and np.any(arr != 0):
                    channels.add(int(index))
        else:
            arr = np.asarray(value)
            if arr.ndim >= 3:
                nonzero = np.any(arr != 0, axis=tuple(range(1, arr.ndim)))
                channels.update(int(index) for index in np.nonzero(nonzero)[0])
            elif arr.size and np.any(arr != 0):
                channels.add(0)
    if not channels:
        channels.add(0)
    return channels


def _nonzero_rank_coupled_blocks_for_channels(blocks, active_channels, *, tol=0.0):
    out = []
    active_channels = set(int(channel) for channel in active_channels)
    for idx, block in enumerate(blocks):
        if idx not in active_channels:
            continue
        arr = np.asarray(block)
        if arr.size and np.any(np.abs(arr) > tol):
            out.append((idx, arr))
    return tuple(out)


def _symbolic_numeric_payloads_from_block_map(block_map, active_channels):
    payloads = {}
    active_channels = set(int(channel) for channel in active_channels)
    rank_coupled = bool(getattr(block_map, "rank_coupled", False))
    for (q_out, q_in), value in block_map.items():
        if isinstance(value, (tuple, list)):
            rank_coupled = True
            for channel, block in _nonzero_rank_coupled_blocks_for_channels(
                value,
                active_channels,
            ):
                payloads[(q_out, q_in, int(channel))] = np.asarray(block)
        else:
            payloads[(q_out, q_in, None)] = np.asarray(value)
    return payloads, "rank_coupled" if rank_coupled else "block"


def _active_virtual_pairs_from_block(block):
    arr = np.asarray(block)
    if arr.ndim < 2:
        return ()
    if arr.ndim == 2:
        nonzero = arr != 0
    else:
        nonzero = np.any(arr != 0, axis=tuple(range(2, arr.ndim)))
    return tuple((int(i), int(j)) for i, j in zip(*np.nonzero(nonzero)))


def _stable_operator_label(operator):
    return (
        type(operator).__name__,
        getattr(operator, "rank_irrep", None),
        tuple(sorted(getattr(operator, "reduced_blocks", {}).keys(), key=repr))
        if hasattr(operator, "reduced_blocks")
        else id(operator),
    )


def symbolic_mpo_core_transitions(core):
    """
    Return symbolic virtual-channel transitions for an MPO core.

    :param core: Dense, block-sparse, irreducible, or rank-coupled MPO core.
    :returns: ``(transitions, used_metadata)``.
    """

    records = tuple(getattr(core, "symbolic_transitions", ()) or ())
    if records:
        return (
            tuple(
                SymbolicMPOTransition(
                    kind=record[0],
                    left_channel=record[1],
                    right_channel=record[2],
                    label=record[3],
                )
                for record in records
            ),
            True,
        )
    transitions = {}
    dense_blocks = getattr(core, "dense_blocks", None)
    if dense_blocks is None:
        dense_blocks = getattr(core, "scalar_blocks", None)
    if dense_blocks is None and hasattr(core, "blocks"):
        dense_blocks = core.blocks
    for key, block in dict(dense_blocks or {}).items():
        for left, right in _active_virtual_pairs_from_block(block):
            transition = SymbolicMPOTransition(
                kind="dense",
                left_channel=left,
                right_channel=right,
                label=("dense", key),
            )
            transitions[transition.key] = transition
    for term_index, term in enumerate(getattr(core, "reduced_terms", ()) or ()):
        visible = getattr(term, "visible_virtual_block", None)
        if visible is None:
            visible_blocks = getattr(term, "component_virtual_blocks", {})
            active_pairs = set()
            for block in visible_blocks.values():
                active_pairs.update(_active_virtual_pairs_from_block(block))
        else:
            active_pairs = _active_virtual_pairs_from_block(visible)
        label = (
            "reduced",
            int(term_index),
            _stable_operator_label(getattr(term, "reduced_operator", None)),
            bool(getattr(term, "use_cg_coupling", False)),
        )
        for left, right in active_pairs:
            transition = SymbolicMPOTransition(
                kind="reduced",
                left_channel=left,
                right_channel=right,
                label=label,
            )
            transitions[transition.key] = transition
    return tuple(transitions.values()), False


def _accumulate_symbolic_term(out, term):
    key = (int(term.channel), tuple(term.path))
    previous = out.get(key)
    if previous is None:
        out[key] = term
    else:
        out[key] = SymbolicRenormalizedOperatorTerm(
            channel=term.channel,
            path=term.path,
            multiplicity=int(previous.multiplicity) + int(term.multiplicity),
        )


def _finalize_symbolic_terms(items):
    out = {}
    for term in items.values():
        out.setdefault(int(term.channel), []).append(term)
    return {
        channel: tuple(sorted(terms, key=lambda term: repr(term.path)))
        for channel, terms in out.items()
    }


def _compact_advance_symbolic_terms(terms_by_channel, transitions, *, active=None, direction):
    """
    Advance symbolic boundary terms in compressed virtual-channel form.

    The numeric renormalized block already owns the exact contracted operator
    payload.  For local table construction the symbolic boundary only needs
    active MPO virtual channels and lineage counts, not every full absorbed
    MPO path.  This mirrors symbolic MPO compression by merging all paths that
    end in the same visible channel while preserving multiplicity diagnostics.

    :param terms_by_channel: Parent channel-to-terms mapping.
    :param transitions: MPO virtual-channel transitions for the absorbed site.
    :param active: Optional active child channels discovered from the numeric
        child block.
    :param direction: ``"left"`` for left-boundary growth or ``"right"`` for
        right-boundary growth.
    :returns: Compact channel-to-term mapping.
    """

    if direction not in {"left", "right"}:
        raise ValueError(f"Unknown symbolic advance direction {direction!r}.")
    active = None if active is None else {int(channel) for channel in active}
    counts = {}
    depth = 0
    for terms in terms_by_channel.values():
        for term in terms:
            depth = max(depth, len(term.path))
    child_depth = int(depth) + 1
    family_sets = {}
    for transition in transitions:
        if direction == "left":
            parent_channel = int(transition.left_channel)
            child_channel = int(transition.right_channel)
        else:
            parent_channel = int(transition.right_channel)
            child_channel = int(transition.left_channel)
        if active is not None and child_channel not in active:
            continue
        parent_terms = tuple(terms_by_channel.get(parent_channel, ()))
        multiplicity = sum(int(term.multiplicity) for term in parent_terms)
        if multiplicity:
            counts[child_channel] = counts.get(child_channel, 0) + multiplicity
            families = family_sets.setdefault(child_channel, set())
            families.update(_family_names_from_symbolic_label(transition.label))
            for term in parent_terms:
                for path_item in tuple(term.path):
                    if len(path_item) >= 4:
                        families.update(_family_names_from_symbolic_label(path_item[3]))
    return {
        channel: (
            SymbolicRenormalizedOperatorTerm(
                channel=channel,
                path=(
                    (
                        "compact",
                        int(child_depth),
                        int(channel),
                        (
                            "families",
                            tuple(sorted(family_sets.get(channel, ()))),
                        ),
                    ),
                ),
                multiplicity=int(multiplicity),
            ),
        )
        for channel, multiplicity in sorted(counts.items())
    }


@dataclass
class RenormalizedOperatorStack:
    """
    Persistent cache of environment-owned renormalized local operators.

    The stack is intentionally small and key-addressed.  It is a stepping stone
    toward block2-style left/right renormalized operator stacks: callers store
    transformed local operators by environment/basis signatures instead of
    anonymous per-bond local metadata.

    :param max_size: Maximum number of cached local operators to retain.
    """

    max_size: int = 256
    entries: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, key):
        """
        Return a cached renormalized operator and mark it recently used.

        :param key: Hashable renormalized-operator stack key.
        :returns: Cached object, or ``None``.
        """

        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        value = self.entries.pop(key)
        self.entries[key] = value
        return value

    def put(self, key, value):
        """
        Insert a renormalized operator into the stack.

        :param key: Hashable renormalized-operator stack key.
        :param value: Renormalized local operator problem.
        :returns: Inserted value.
        """

        if key in self.entries:
            self.entries.pop(key)
        self.entries[key] = value
        self.prune()
        return value

    def prune(self):
        """Prune least-recently-used entries beyond ``max_size``."""

        if int(self.max_size) <= 0:
            self.entries.clear()
            return
        while len(self.entries) > int(self.max_size):
            self.entries.popitem(last=False)

    def __len__(self):
        """Return the number of cached stack entries."""

        return len(self.entries)

    @property
    def stats(self):
        """
        Return cache diagnostics.

        :returns: Dictionary with cache size, hit count, and miss count.
        """

        return {
            "size": int(len(self.entries)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "max_size": int(self.max_size),
        }


@dataclass
class RenormalizedSideOperatorTable:
    """
    Grouped one-sided renormalized operator data for a boundary entry.

    :param key: Hashable table key.
    :param grouped_by_ket: Boundary blocks grouped by ket sector.
    :param owner_side: Boundary side that owns the table.
    :param owner_bond: Boundary bond that owns the table.
    :param representation: Table representation.
    :param parent_key: Optional parent boundary key used to derive this table.
    :param source: How this side table was produced.
    :param derived_from: Optional side-table key used to derive this table.
    """

    key: object
    grouped_by_ket: object
    owner_side: str
    owner_bond: int
    representation: str
    parent_key: object | None = None
    source: str = "lazy"
    derived_from: object | None = None
    hits: int = 0

    def mark_hit(self):
        """
        Mark this side table as reused.

        :returns: ``self`` for call chaining.
        """

        self.hits += 1
        return self

    def derive(
        self,
        *,
        key,
        grouped_by_ket,
        representation=None,
        owner_side=None,
        owner_bond=None,
        parent_key=None,
        source="prepared",
    ):
        """
        Return a side table derived from this table.

        :param key: Hashable child side-table key.
        :param grouped_by_ket: Child grouped payload.
        :param representation: Optional child table representation. Defaults
            to this table's representation.
        :param owner_side: Optional child owner side. Defaults to this table's
            owner side.
        :param owner_bond: Optional child owner bond. Defaults to this table's
            owner bond.
        :param parent_key: Optional parent boundary-stack key.
        :param source: Child table source label.
        :returns: Derived :class:`RenormalizedSideOperatorTable`.
        """

        return RenormalizedSideOperatorTable(
            key=key,
            grouped_by_ket=grouped_by_ket,
            owner_side=str(self.owner_side if owner_side is None else owner_side),
            owner_bond=int(self.owner_bond if owner_bond is None else owner_bond),
            representation=str(
                self.representation if representation is None else representation
            ),
            parent_key=parent_key,
            source=str(source),
            derived_from=self.key,
        )

    def advance_left(self, *, key, grouped_by_ket, owner_bond, parent_key=None, source="advanced_left"):
        """
        Return the table produced by advancing this left-side table one site.

        The grouped payload is supplied by the boundary-block advance layer; this
        method records the structural parent/child relation so callers can treat
        side tables as first-class propagated renormalized operators.

        :param key: Hashable child side-table key.
        :param grouped_by_ket: Child boundary blocks grouped by ket sector.
        :param owner_bond: Child left-boundary bond.
        :param parent_key: Optional parent boundary-stack key.
        :param source: Child table source label.
        :returns: Advanced :class:`RenormalizedSideOperatorTable`.
        """

        return self.derive(
            key=key,
            grouped_by_ket=grouped_by_ket,
            representation=self.representation,
            owner_side="left",
            owner_bond=owner_bond,
            parent_key=parent_key,
            source=source,
        )

    def advance_right(self, *, key, grouped_by_ket, owner_bond, parent_key=None, source="advanced_right"):
        """
        Return the table produced by advancing this right-side table one site.

        :param key: Hashable child side-table key.
        :param grouped_by_ket: Child boundary blocks grouped by ket sector.
        :param owner_bond: Child right-boundary bond.
        :param parent_key: Optional parent boundary-stack key.
        :param source: Child table source label.
        :returns: Advanced :class:`RenormalizedSideOperatorTable`.
        """

        return self.derive(
            key=key,
            grouped_by_ket=grouped_by_ket,
            representation=self.representation,
            owner_side="right",
            owner_bond=owner_bond,
            parent_key=parent_key,
            source=source,
        )

    @property
    def n_ket_sectors(self):
        """Return the number of ket-sector groups in the table."""

        return int(len(self.grouped_by_ket))

    @property
    def n_terms(self):
        """Return the number of grouped boundary terms."""

        return int(sum(len(entries) for entries in self.grouped_by_ket.values()))

    @property
    def stats(self):
        """
        Return side-table diagnostics.

        :returns: Dictionary describing owner, representation, and reuse.
        """

        return {
            "kind": str(self.representation),
            "representation": str(self.representation),
            "owner_side": str(self.owner_side),
            "owner_bond": int(self.owner_bond),
            "parent_key": self.parent_key,
            "source": str(self.source),
            "derived_from": self.derived_from,
            "n_ket_sectors": int(self.n_ket_sectors),
            "n_terms": int(self.n_terms),
            "hits": int(self.hits),
        }


@dataclass
class RenormalizedLocalOperatorTable:
    """
    Compiled local operator table owned by one renormalized boundary entry.

    :param key: Hashable table key.
    :param actions: Compiled solver-facing local actions.
    :param owner_side: Boundary side that owns the table.
    :param owner_bond: Boundary bond that owns the table.
    :param representation: Table representation, for example ``"transition"``
        or ``"rank_coupled_factorized"``.
    :param basis_size: Parent packed local basis size.
    """

    key: object
    actions: object
    owner_side: str
    owner_bond: int
    representation: str = "transition"
    basis_size: int | None = None
    hits: int = 0
    entry_kernel_items_cache: dict = field(default_factory=dict, repr=False)
    transformed_operator_table_cache: dict = field(default_factory=dict, repr=False)

    @property
    def packed_matvec(self):
        """Return the packed-vector matvec owned by this table."""

        return getattr(self.actions, "packed_matvec", None)

    @property
    def compiled_transitions(self):
        """Return compiled transition kernels when this table owns them."""

        return getattr(self.packed_matvec, "compiled_transitions", None)

    @property
    def compiled_factorized_terms(self):
        """Return compiled factorized kernels when this table owns them."""

        return getattr(self.packed_matvec, "compiled_factorized_terms", None)

    def get_entry_kernel_items(self, cache_key):
        """
        Return cached dense entry kernels for this local table.

        :param cache_key: Basis and kernel-size dependent cache key.
        :returns: Cached entry-kernel tuple or ``None``.
        """

        return self.entry_kernel_items_cache.get(cache_key)

    def put_entry_kernel_items(self, cache_key, entry_kernel_items):
        """
        Cache dense entry kernels for this local table.

        :param cache_key: Basis and kernel-size dependent cache key.
        :param entry_kernel_items: Tuple of ``(in_idx, out_idx, kernel)``.
        :returns: Stored entry-kernel tuple.
        """

        self.entry_kernel_items_cache[cache_key] = tuple(entry_kernel_items)
        return self.entry_kernel_items_cache[cache_key]

    def get_transformed_operator_table(self, cache_key):
        """
        Return a cached transformed orthonormal operator table.

        :param cache_key: Basis, metric-component, and transform dependent key.
        :returns: Cached transformed table or ``None``.
        """

        return self.transformed_operator_table_cache.get(cache_key)

    def put_transformed_operator_table(self, cache_key, table):
        """
        Cache a transformed orthonormal operator table.

        :param cache_key: Basis, metric-component, and transform dependent key.
        :param table: Compiled transformed operator table.
        :returns: Stored table.
        """

        self.transformed_operator_table_cache[cache_key] = table
        return table

    def attach_to_compiled_actions(self):
        """
        Attach this table as owner metadata to compiled local kernels.

        :returns: ``self`` for call chaining.
        """

        for compiled in (self.compiled_transitions, self.compiled_factorized_terms):
            if compiled is not None:
                setattr(compiled, "local_operator_table", self)
        return self

    def mark_hit(self):
        """
        Mark this table as reused.

        :returns: ``self`` for call chaining.
        """

        self.hits += 1
        self.attach_to_compiled_actions()
        return self

    @property
    def stats(self):
        """
        Return table diagnostics.

        :returns: Dictionary describing owner and reuse count.
        """

        return {
            "kind": str(self.representation),
            "representation": str(self.representation),
            "owner_side": str(self.owner_side),
            "owner_bond": int(self.owner_bond),
            "basis_size": None if self.basis_size is None else int(self.basis_size),
            "hits": int(self.hits),
            "entry_kernel_item_caches": int(len(self.entry_kernel_items_cache)),
            "transformed_operator_table_caches": int(
                len(self.transformed_operator_table_cache)
            ),
        }


@dataclass(frozen=True)
class RenormalizedBlockEntry:
    """
    One persisted left or right renormalized environment block.

    This object represents the block2-like boundary object: the sweep stores
    renormalized left/right blocks by side and bond index, and local operators
    consume those persisted blocks instead of anonymous metadata.

    :param side: Boundary side, either ``"left"`` or ``"right"``.
    :param bond: Boundary bond index.  ``left`` entries live to the left of
        ``bond``; ``right`` entries live to the right of ``bond``.
    :param block: Environment block object.
    :param signature: Optional boundary signature used for table/cache keys.
    :param namespace: Logical operator namespace, for example ``"hamiltonian"``
        or ``"norm"``.
    :param source: How this boundary entry was produced.
    :param parent_key: Optional stack key of the boundary entry absorbed to
        produce this entry.
    :param symbolic_operator_table: Recursive symbolic boundary operator table.
    :param complementary_operator_entry: Matching complementary-family
        boundary payload entry, when this block belongs to a complementary
        qchem Hamiltonian stack.
    :param local_operator_tables: Mutable cache of
        :class:`RenormalizedLocalOperatorTable` objects derived from this
        boundary entry.
    :param local_operator_table_stats: Mutable cache diagnostics.
    :param side_operator_tables: Mutable cache of grouped one-sided
        renormalized operator tables derived from this boundary entry.
    :param side_operator_table_stats: Mutable side-table diagnostics.
    """

    side: str
    bond: int
    block: object
    signature: object | None = None
    namespace: str = "hamiltonian"
    source: str = "stored"
    parent_key: object | None = None
    symbolic_operator_table: SymbolicRenormalizedOperatorTable | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    complementary_operator_entry: object | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    local_operator_tables: dict = field(default_factory=dict, compare=False, repr=False)
    local_operator_table_stats: dict = field(
        default_factory=lambda: {"hits": 0, "misses": 0, "puts": 0},
        compare=False,
        repr=False,
    )
    side_operator_tables: dict = field(default_factory=dict, compare=False, repr=False)
    side_operator_table_stats: dict = field(
        default_factory=lambda: {"hits": 0, "misses": 0, "puts": 0},
        compare=False,
        repr=False,
    )
    advance_timing: dict = field(default_factory=dict, compare=False, repr=False)

    def put_symbolic_operator_table(self, table):
        """
        Store the recursive symbolic boundary operator table on this entry.

        :param table: :class:`SymbolicRenormalizedOperatorTable` to attach.
        :returns: Stored table.
        """

        object.__setattr__(self, "symbolic_operator_table", table)
        complementary_entry = getattr(self, "complementary_operator_entry", None)
        if complementary_entry is not None:
            object.__setattr__(
                complementary_entry,
                "family_operator_table",
                ComplementaryFamilyRenormalizedOperatorTable.from_symbolic_table(
                    table,
                    complementary_entry.family_names,
                    family_payloads=complementary_entry.family_payloads,
                ),
            )
        return table

    def get_side_operator_table(self, key):
        """
        Return a cached one-sided renormalized operator table.

        :param key: Hashable side-table key.
        :returns: Cached :class:`RenormalizedSideOperatorTable`, or ``None``.
        """

        if key not in self.side_operator_tables:
            self.side_operator_table_stats["misses"] += 1
            return None
        self.side_operator_table_stats["hits"] += 1
        return self.side_operator_tables[key].mark_hit()

    def put_side_operator_table(
        self,
        key,
        grouped_by_ket,
        *,
        representation,
        source="lazy",
        derived_from=None,
        parent_table=None,
        advance_direction=None,
    ):
        """
        Store a grouped one-sided renormalized operator table.

        :param key: Hashable side-table key.
        :param grouped_by_ket: Boundary blocks grouped by ket sector.
        :param representation: Table representation.
        :param source: How this side table was produced.
        :param derived_from: Optional side-table key used to derive this table.
        :param parent_table: Optional parent side table for an advanced or
            locally derived entry.
        :param advance_direction: ``"left"`` or ``"right"`` when advancing
            from ``parent_table``.
        :returns: Stored :class:`RenormalizedSideOperatorTable`.
        """

        if parent_table is not None and advance_direction in {"left", "right"}:
            if str(parent_table.representation) != str(representation):
                raise ValueError(
                    "Cannot derive side table representation "
                    f"{representation!r} from {parent_table.representation!r}."
                )
            if advance_direction == "left":
                table = parent_table.advance_left(
                    key=key,
                    grouped_by_ket=grouped_by_ket,
                    owner_bond=self.bond,
                    parent_key=self.parent_key,
                    source=source,
                )
            elif advance_direction == "right":
                table = parent_table.advance_right(
                    key=key,
                    grouped_by_ket=grouped_by_ket,
                    owner_bond=self.bond,
                    parent_key=self.parent_key,
                    source=source,
                )
        elif parent_table is not None:
            table = parent_table.derive(
                key=key,
                grouped_by_ket=grouped_by_ket,
                representation=representation,
                owner_side=self.side,
                owner_bond=self.bond,
                parent_key=self.parent_key,
                source=source,
            )
        else:
            table = RenormalizedSideOperatorTable(
                key=key,
                grouped_by_ket=grouped_by_ket,
                owner_side=str(self.side),
                owner_bond=int(self.bond),
                representation=str(representation),
                parent_key=self.parent_key,
                source=str(source),
                derived_from=derived_from,
            )
        self.side_operator_tables[key] = table
        self.side_operator_table_stats["puts"] += 1
        return table

    def get_local_operator_table(self, key):
        """
        Return a cached compiled local operator table.

        :param key: Hashable table key.
        :returns: Cached :class:`RenormalizedLocalOperatorTable`, or ``None``.
        """

        if key not in self.local_operator_tables:
            self.local_operator_table_stats["misses"] += 1
            return None
        self.local_operator_table_stats["hits"] += 1
        return self.local_operator_tables[key].mark_hit()

    def put_local_operator_table(self, key, actions, *, representation="transition", basis_size=None):
        """
        Store a compiled local operator table on this boundary entry.

        :param key: Hashable table key.
        :param actions: Compiled solver-facing local actions.
        :param representation: Table representation.
        :param basis_size: Parent packed local basis size.
        :returns: Stored :class:`RenormalizedLocalOperatorTable`.
        """

        table = RenormalizedLocalOperatorTable(
            key=key,
            actions=actions,
            owner_side=str(self.side),
            owner_bond=int(self.bond),
            representation=representation,
            basis_size=basis_size,
        )
        table.attach_to_compiled_actions()
        self.local_operator_tables[key] = table
        self.local_operator_table_stats["puts"] += 1
        return table

    @property
    def rank_coupled(self):
        """Return whether the stored block uses rank-coupled MPO channels."""

        return bool(getattr(self.block, "rank_coupled", False))

    @property
    def n_sector_pairs(self):
        """Return the number of sector-pair environment blocks."""

        return int(len(self.block))

    @property
    def n_arrays(self):
        """Return the number of dense arrays stored in this entry."""

        total = 0
        for value in self.block.values():
            if isinstance(value, (tuple, list)):
                total += len(value)
            else:
                total += 1
        return int(total)

    @property
    def stored_elements(self):
        """Return the number of scalar elements stored in this entry."""

        total = 0
        for value in self.block.values():
            if isinstance(value, (tuple, list)):
                total += sum(int(np.asarray(item).size) for item in value)
            else:
                total += int(np.asarray(value).size)
        return int(total)

    @property
    def stats(self):
        """
        Return compact diagnostics for this boundary entry.

        :returns: Dictionary with side, bond, block count, and storage size.
        """

        return {
            "namespace": str(self.namespace),
            "side": str(self.side),
            "bond": int(self.bond),
            "source": str(self.source),
            "parent_key": self.parent_key,
            "rank_coupled": bool(self.rank_coupled),
            "symbolic_operator_table": (
                None
                if self.symbolic_operator_table is None
                else self.symbolic_operator_table.stats
            ),
            "complementary_operator_entry": (
                None
                if self.complementary_operator_entry is None
                else self.complementary_operator_entry.stats
            ),
            "n_sector_pairs": int(self.n_sector_pairs),
            "n_arrays": int(self.n_arrays),
            "stored_elements": int(self.stored_elements),
            "local_operator_tables": int(len(self.local_operator_tables)),
            "local_operator_table_hits": int(self.local_operator_table_stats["hits"]),
            "local_operator_table_misses": int(self.local_operator_table_stats["misses"]),
            "local_operator_table_puts": int(self.local_operator_table_stats["puts"]),
            "local_operator_table_reuses": int(
                sum(table.hits for table in self.local_operator_tables.values())
            ),
            "side_operator_tables": int(len(self.side_operator_tables)),
            "side_operator_table_hits": int(self.side_operator_table_stats["hits"]),
            "side_operator_table_misses": int(self.side_operator_table_stats["misses"]),
            "side_operator_table_puts": int(self.side_operator_table_stats["puts"]),
            "side_operator_table_reuses": int(
                sum(table.hits for table in self.side_operator_tables.values())
            ),
            "advance_timing": {
                str(key): float(value)
                for key, value in self.advance_timing.items()
            },
            "side_operator_table_sources": sorted(
                {str(table.source) for table in self.side_operator_tables.values()}
            ),
            "side_operator_table_representations": sorted(
                {str(table.representation) for table in self.side_operator_tables.values()}
            ),
        }

@dataclass(frozen=True)
class ComplementaryFamilyBoundaryPayload:
    """
    Numeric sparse complementary-family payload owned by one boundary.

    The payload stores integral-side coefficients after classifying each
    family entry against the sites already absorbed into a left/right block.
    This is the numeric boundary object that later direct ``S/R/A/P/B/Q``
    contractions can consume without rewalking the chemistry integral tensor.

    :param family_name: Complementary family label.
    :param entries: Tuple of ``(index_tuple, coefficient)`` entries.
    :param internal_terms: Number of entries fully inside the boundary block.
    :param cross_terms: Number of entries connecting the block and exterior.
    :param external_terms: Number of entries fully outside the boundary block.
    """

    family_name: str
    entries: tuple
    internal_terms: int
    cross_terms: int
    external_terms: int

    @property
    def n_terms(self):
        """Return the number of stored sparse coefficients."""

        return int(len(self.entries))

    @property
    def coefficient_norm(self):
        """Return the Euclidean norm of stored numeric coefficients."""

        if not self.entries:
            return 0.0
        values = np.asarray(
            [complex(value) for _key, value in self.entries],
            dtype=complex,
        )
        return float(np.linalg.norm(values))

    @property
    def max_abs_coefficient(self):
        """Return the largest absolute coefficient in this payload."""

        return float(
            max((abs(complex(value)) for _key, value in self.entries), default=0.0)
        )

    @property
    def stats(self):
        """Return compact diagnostics for this family payload."""

        return {
            "family_name": str(self.family_name),
            "n_terms": int(self.n_terms),
            "internal_terms": int(self.internal_terms),
            "cross_terms": int(self.cross_terms),
            "external_terms": int(self.external_terms),
            "coefficient_norm": float(self.coefficient_norm),
            "max_abs_coefficient": float(self.max_abs_coefficient),
        }


@dataclass(frozen=True)
class ComplementaryRenormalizedOperatorEntry:
    """
    Recursive complementary-operator boundary record.

    The record tracks block2-style complementary family ownership alongside
    the ordinary left/right environment stack.  Numeric family payloads are
    stored per boundary so direct ``S/R/A/P/B/Q`` contractions can use the
    stack without rewalking qchem integrals.

    :param side: Boundary side, ``"left"`` or ``"right"``.
    :param bond: Boundary bond index.
    :param family_names: Complementary family labels owned at this boundary.
    :param parent_key: Optional previous complementary stack key.
    :param source: How the boundary was produced.
    :param signature: Signature of the matching renormalized environment.
    """

    side: str
    bond: int
    family_names: tuple
    parent_key: object | None = None
    source: str = "stored"
    signature: object | None = None
    family_payloads: dict = field(default_factory=dict)
    family_operator_table: ComplementaryFamilyRenormalizedOperatorTable | None = None

    @property
    def key(self):
        """
        Return the stack key for this complementary boundary.

        :returns: Hashable ``(side, bond)`` key.
        """

        return (str(self.side), int(self.bond))

    @property
    def stats(self):
        """
        Return diagnostics for this complementary boundary record.

        :returns: Dictionary with family labels and recursive provenance.
        """

        return {
            "side": str(self.side),
            "bond": int(self.bond),
            "family_names": tuple(self.family_names),
            "parent_key": self.parent_key,
            "source": str(self.source),
            "signature": self.signature,
            "numeric_payloads": {
                str(name): payload.stats
                for name, payload in self.family_payloads.items()
            },
            "numeric_payload_terms": int(
                sum(payload.n_terms for payload in self.family_payloads.values())
            ),
            "numeric_payload_cross_terms": int(
                sum(payload.cross_terms for payload in self.family_payloads.values())
            ),
            "family_operator_table": (
                None
                if self.family_operator_table is None
                else self.family_operator_table.stats
            ),
        }


@dataclass
class ComplementaryRenormalizedOperatorStack:
    """
    Recursive stack for block2-style complementary operator families.

    :param families: Family container exposing ``names`` and ``as_metadata``.
    :param entries: Stored complementary boundary records.
    """

    families: object
    entries: dict = field(default_factory=dict)
    puts: int = 0
    advances: int = 0

    @property
    def n_sites(self):
        """Return the number of spatial sites described by the families."""

        return int(getattr(self.families, "n_sites", 0) or 0)

    @property
    def family_names(self):
        """
        Return complementary family labels.

        :returns: Tuple such as ``("S", "R", "A", "P", "B", "Q")``.
        """

        return tuple(getattr(self.families, "names", ()))

    def _owned_sites(self, side, bond):
        """
        Return the spatial sites absorbed into a boundary block.

        Left boundary ``b`` owns sites ``0..b-1``.  Right boundary ``b`` owns
        sites ``b+1..n_sites-1``; this matches the environment convention used
        for two-site bond operators.
        """

        side = str(side).lower()
        bond = int(bond)
        n_sites = int(self.n_sites)
        if n_sites <= 0:
            return frozenset()
        if side == "left":
            return frozenset(range(max(0, min(bond, n_sites))))
        if side == "right":
            start = max(0, min(bond + 1, n_sites))
            return frozenset(range(start, n_sites))
        raise ValueError(f"Unknown complementary boundary side {side!r}.")

    def _family_payloads_for_boundary(self, side, bond):
        """Build numeric sparse family payloads for one boundary."""

        owned_sites = self._owned_sites(side, bond)
        payloads = {}
        families = getattr(self.families, "families", {}) or {}
        for name in self.family_names:
            family = families.get(name)
            entries = getattr(family, "entries", {}) if family is not None else {}
            ordered_entries = tuple(
                (
                    tuple(int(index) for index in key),
                    complex(value),
                )
                for key, value in sorted(
                    entries.items(),
                    key=lambda item: tuple(int(index) for index in item[0]),
                )
            )
            internal_terms = 0
            cross_terms = 0
            external_terms = 0
            for key, _value in ordered_entries:
                if not key:
                    external_terms += 1
                    continue
                flags = tuple(int(index) in owned_sites for index in key)
                if all(flags):
                    internal_terms += 1
                elif any(flags):
                    cross_terms += 1
                else:
                    external_terms += 1
            payloads[str(name)] = ComplementaryFamilyBoundaryPayload(
                family_name=str(name),
                entries=ordered_entries,
                internal_terms=int(internal_terms),
                cross_terms=int(cross_terms),
                external_terms=int(external_terms),
            )
        return payloads

    def put(self, side, bond, *, signature=None, source="stored", parent_key=None):
        """
        Store a complementary boundary record.

        :param side: Boundary side.
        :param bond: Boundary bond index.
        :param signature: Matching renormalized environment signature.
        :param source: How this record was produced.
        :param parent_key: Optional previous complementary boundary key.
        :returns: Stored :class:`ComplementaryRenormalizedOperatorEntry`.
        """

        entry = ComplementaryRenormalizedOperatorEntry(
            side=str(side),
            bond=int(bond),
            family_names=self.family_names,
            parent_key=parent_key,
            source=str(source),
            signature=signature,
            family_payloads=self._family_payloads_for_boundary(side, bond),
        )
        self.entries[entry.key] = entry
        self.puts += 1
        if parent_key is not None:
            self.advances += 1
        return entry

    def get(self, side, bond):
        """Return a stored complementary boundary entry, if present."""

        return self.entries.get((str(side), int(bond)))

    @property
    def stats(self):
        """
        Return stack diagnostics.

        :returns: Dictionary describing recursive complementary ownership.
        """

        metadata = (
            self.families.as_metadata()
            if hasattr(self.families, "as_metadata")
            else {"enabled": True, "type": type(self.families).__name__}
        )
        return {
            "enabled": True,
            "family_names": self.family_names,
            "n_entries": int(len(self.entries)),
            "puts": int(self.puts),
            "advances": int(self.advances),
            "numeric_payload_terms": int(
                sum(
                    payload.n_terms
                    for entry in self.entries.values()
                    for payload in entry.family_payloads.values()
                )
            ),
            "numeric_payload_cross_terms": int(
                sum(
                    payload.cross_terms
                    for entry in self.entries.values()
                    for payload in entry.family_payloads.values()
                )
            ),
            "family_operator_tables": int(
                sum(
                    1
                    for entry in self.entries.values()
                    if entry.family_operator_table is not None
                )
            ),
            "family_operator_table_payload_blocks": int(
                sum(
                    entry.family_operator_table.n_payload_blocks
                    for entry in self.entries.values()
                    if entry.family_operator_table is not None
                )
            ),
            "family_operator_table_stored_elements": int(
                sum(
                    entry.family_operator_table.stored_elements
                    for entry in self.entries.values()
                    if entry.family_operator_table is not None
                )
            ),
            "family_operator_table_symbolic_terms": int(
                sum(
                    entry.family_operator_table.symbolic_terms
                    for entry in self.entries.values()
                    if entry.family_operator_table is not None
                )
            ),
            "numeric_payload_families": {
                str(name): {
                    "n_entries": int(
                        sum(
                            1
                            for entry in self.entries.values()
                            if name in entry.family_payloads
                        )
                    ),
                    "n_terms": int(
                        sum(
                            entry.family_payloads[name].n_terms
                            for entry in self.entries.values()
                            if name in entry.family_payloads
                        )
                    ),
                    "cross_terms": int(
                        sum(
                            entry.family_payloads[name].cross_terms
                            for entry in self.entries.values()
                            if name in entry.family_payloads
                        )
                    ),
                }
                for name in self.family_names
            },
            "families": metadata,
        }


@dataclass
class RenormalizedBlockStack:
    """
    Persistent left/right renormalized environment stack.

    Unlike :class:`RenormalizedOperatorStack`, which caches complete local
    two-site problems, this stack stores sweep boundary blocks.  It is the
    structural API needed for block2-like DMRG: environments are addressed by
    side and bond, updated as the sweep moves, and consumed by local effective
    Hamiltonian builders.

    :param namespace: Logical operator namespace for entries in this stack.
    :param complementary_operator_families: Optional block2-style
        complementary operator families owned by this Hamiltonian stack.
    """

    namespace: str = "hamiltonian"
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    puts: int = 0
    complementary_operator_families: object | None = None
    complementary_operator_stack: ComplementaryRenormalizedOperatorStack | None = None
    moving_environment_cache: MovingEnvironmentContractionCache = field(
        default_factory=MovingEnvironmentContractionCache
    )

    def __post_init__(self):
        if (
            self.complementary_operator_families is not None
            and self.complementary_operator_stack is None
        ):
            self.complementary_operator_stack = ComplementaryRenormalizedOperatorStack(
                families=self.complementary_operator_families
            )

    def set_complementary_operator_families(self, families):
        """
        Attach block2-style complementary operator families to this stack.

        :param families: Object exposing ``as_metadata()``, such as the qchem
            spatial ``S/R/A/P/B/Q`` family container.
        :returns: ``self`` for call chaining.
        """

        if families is None:
            self.complementary_operator_families = None
            self.complementary_operator_stack = None
            return self
        if (
            self.complementary_operator_stack is not None
            and self.complementary_operator_families is families
        ):
            return self
        self.complementary_operator_families = families
        self.complementary_operator_stack = ComplementaryRenormalizedOperatorStack(
            families=families
        )
        return self

    @property
    def complementary_operator_family_metadata(self):
        """
        Return compact complementary-family diagnostics.

        :returns: Dictionary metadata, or ``None`` when no families are owned.
        """

        families = self.complementary_operator_families
        if families is None:
            return None
        if hasattr(families, "as_metadata"):
            return families.as_metadata()
        return {"enabled": True, "type": type(families).__name__}

    def key(self, side, bond):
        """
        Return the canonical stack key.

        :param side: ``"left"`` or ``"right"``.
        :param bond: Boundary bond index.
        :returns: Hashable stack key.
        """

        side = str(side).lower()
        if side not in {"left", "right"}:
            raise ValueError(f"Unknown renormalized block side {side!r}.")
        return (str(self.namespace), side, int(bond))

    def put(self, side, bond, block, *, signature=None, source="stored", parent_key=None):
        """
        Store a renormalized boundary block.

        :param side: Boundary side.
        :param bond: Boundary bond index.
        :param block: Environment block to persist.
        :param signature: Optional boundary signature.
        :param source: How this boundary entry was produced.
        :param parent_key: Optional parent stack key.
        :returns: Stored :class:`RenormalizedBlockEntry`.
        """

        _, normalized_side, normalized_bond = self.key(side, bond)
        if signature is None:
            signature = (
                "renormalized_block",
                str(self.namespace),
                normalized_side,
                int(normalized_bond),
                int(self.puts),
                str(source),
                parent_key,
            )
        entry = RenormalizedBlockEntry(
            side=normalized_side,
            bond=normalized_bond,
            block=block,
            signature=signature,
            namespace=str(self.namespace),
            source=str(source),
            parent_key=parent_key,
        )
        self.entries[self.key(side, bond)] = entry
        if self.complementary_operator_stack is not None:
            complementary_entry = self.complementary_operator_stack.put(
                normalized_side,
                normalized_bond,
                signature=signature,
                source=str(source),
                parent_key=parent_key,
            )
            object.__setattr__(
                entry,
                "complementary_operator_entry",
                complementary_entry,
            )
        self.puts += 1
        return entry

    def initialize(self, side, bond, block, *, signature=None, side_table_builders=None):
        """
        Store an initialized boundary block.

        :param side: Boundary side.
        :param bond: Boundary bond index.
        :param block: Initial environment block.
        :param signature: Optional boundary signature.
        :param side_table_builders: Optional representation-to-builder map for
            prepopulating one-sided operator tables.
        :returns: Stored boundary entry.
        """

        entry = self.put(side, bond, block, signature=signature, source="initialized")
        entry.put_symbolic_operator_table(
            SymbolicRenormalizedOperatorTable.initialize(
                entry.side,
                entry.bond,
                entry.block,
                source=entry.source,
            )
        )
        self.prepopulate_side_operator_tables(entry, side_table_builders)
        return entry

    def prepopulate_side_operator_tables(
        self,
        entry,
        side_table_builders=None,
        *,
        parent_entry=None,
        advance_direction=None,
    ):
        """
        Prepopulate grouped one-sided operator tables for an entry.

        :param entry: Boundary entry to populate.
        :param side_table_builders: Mapping from representation name to a
            callable that accepts ``entry.block`` and returns grouped terms.
        :param parent_entry: Optional previous boundary entry for propagated
            side-table provenance.
        :param advance_direction: Direction used when ``parent_entry`` supplies
            the parent side table.
        :returns: ``entry``.
        """

        if entry is None or not side_table_builders:
            return entry
        for representation, builder in side_table_builders.items():
            key = ("side_operator_table", str(representation), entry.signature)
            if key in entry.side_operator_tables:
                continue
            grouped_by_ket, used_symbolic_table = self._build_side_operator_table_payload(
                entry,
                str(representation),
                builder,
            )
            source = str(entry.source)
            if used_symbolic_table:
                source = "symbolic_" + source
            parent_table = None
            derived_from = entry.parent_key
            if parent_entry is not None:
                parent_table_key = (
                    "side_operator_table",
                    str(representation),
                    parent_entry.signature,
                )
                parent_table = parent_entry.side_operator_tables.get(parent_table_key)
                if parent_table is not None:
                    derived_from = parent_table.key
            entry.put_side_operator_table(
                key,
                grouped_by_ket,
                representation=str(representation),
                source=source,
                derived_from=derived_from,
                parent_table=parent_table,
                advance_direction=advance_direction,
            )
        return entry

    def _build_side_operator_table_payload(self, entry, representation, fallback_builder):
        """
        Build grouped side-table payload from symbolic table metadata.

        :param entry: Boundary entry whose numeric block payload is grouped.
        :param representation: Requested side-table representation.
        :param fallback_builder: Numeric grouping fallback.
        :returns: Grouped boundary blocks by ket sector.
        """

        symbolic_table = getattr(entry, "symbolic_operator_table", None)
        if symbolic_table is not None:
            return symbolic_table.group_boundary_blocks(representation=representation), True
        return fallback_builder(entry.block), False

    def advance_left(
        self,
        entry,
        bond,
        W,
        site,
        *,
        phys_slices=None,
        signature=None,
        signature_fn=None,
        side_table_builders=None,
    ):
        """
        Recursively advance a left boundary entry by absorbing one site.

        :param entry: Previous left boundary entry.
        :param bond: New boundary bond index.
        :param W: MPO core for the absorbed site.
        :param site: Mixed-canonical MPS site tensor to absorb.
        :param phys_slices: Optional physical-sector slices.
        :param signature: Optional signature for the advanced block.
        :param signature_fn: Optional callable used to sign the advanced block.
            If omitted, a unique structural stack signature is assigned.
        :param side_table_builders: Optional representation-to-builder map for
            prepopulating one-sided operator tables.
        :returns: Advanced left boundary entry.
        """

        if entry is None:
            raise ValueError("advance_left requires a previous boundary entry.")
        timing = {
            "block_contract": 0.0,
            "signature": 0.0,
            "put": 0.0,
            "symbolic_advance": 0.0,
            "side_table_prepopulate": 0.0,
        }
        t0 = time.perf_counter()
        block = entry.block.advance(W, site, site, phys_slices=phys_slices)
        timing["block_contract"] = time.perf_counter() - t0
        if signature is None and signature_fn is not None:
            t0 = time.perf_counter()
            signature = signature_fn(block)
            timing["signature"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        advanced = self.put(
            "left",
            bond,
            block,
            signature=signature,
            source="advanced_left",
            parent_key=self.key("left", entry.bond),
        )
        timing["put"] = time.perf_counter() - t0
        if entry.symbolic_operator_table is not None:
            t0 = time.perf_counter()
            advanced.put_symbolic_operator_table(
                entry.symbolic_operator_table.advance_left(
                    W,
                    bond=advanced.bond,
                    block=advanced.block,
                    parent_key=advanced.parent_key,
                )
            )
            timing["symbolic_advance"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.prepopulate_side_operator_tables(
            advanced,
            side_table_builders,
            parent_entry=entry,
            advance_direction="left",
        )
        timing["side_table_prepopulate"] = time.perf_counter() - t0
        advanced.advance_timing.update(timing)
        return advanced

    def advance_right(
        self,
        entry,
        bond,
        W,
        site,
        *,
        phys_slices=None,
        signature=None,
        signature_fn=None,
        side_table_builders=None,
    ):
        """
        Recursively advance a right boundary entry by absorbing one site.

        :param entry: Previous right boundary entry.
        :param bond: New boundary bond index.
        :param W: MPO core for the absorbed site.
        :param site: Mixed-canonical MPS site tensor to absorb.
        :param phys_slices: Optional physical-sector slices.
        :param signature: Optional signature for the advanced block.
        :param signature_fn: Optional callable used to sign the advanced block.
            If omitted, a unique structural stack signature is assigned.
        :param side_table_builders: Optional representation-to-builder map for
            prepopulating one-sided operator tables.
        :returns: Advanced right boundary entry.
        """

        if entry is None:
            raise ValueError("advance_right requires a previous boundary entry.")
        timing = {
            "block_contract": 0.0,
            "signature": 0.0,
            "put": 0.0,
            "symbolic_advance": 0.0,
            "side_table_prepopulate": 0.0,
        }
        t0 = time.perf_counter()
        block = entry.block.advance(W, site, site, phys_slices=phys_slices)
        timing["block_contract"] = time.perf_counter() - t0
        if signature is None and signature_fn is not None:
            t0 = time.perf_counter()
            signature = signature_fn(block)
            timing["signature"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        advanced = self.put(
            "right",
            bond,
            block,
            signature=signature,
            source="advanced_right",
            parent_key=self.key("right", entry.bond),
        )
        timing["put"] = time.perf_counter() - t0
        if entry.symbolic_operator_table is not None:
            t0 = time.perf_counter()
            advanced.put_symbolic_operator_table(
                entry.symbolic_operator_table.advance_right(
                    W,
                    bond=advanced.bond,
                    block=advanced.block,
                    parent_key=advanced.parent_key,
                )
            )
            timing["symbolic_advance"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.prepopulate_side_operator_tables(
            advanced,
            side_table_builders,
            parent_entry=entry,
            advance_direction="right",
        )
        timing["side_table_prepopulate"] = time.perf_counter() - t0
        advanced.advance_timing.update(timing)
        return advanced

    def get(self, side, bond):
        """
        Return a persisted renormalized boundary block entry.

        :param side: Boundary side.
        :param bond: Boundary bond index.
        :returns: ``RenormalizedBlockEntry`` or ``None``.
        """

        entry = self.entries.get(self.key(side, bond))
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def block(self, side, bond, default=None):
        """
        Return a stored environment block or ``default`` on miss.

        :param side: Boundary side.
        :param bond: Boundary bond index.
        :param default: Fallback block.
        :returns: Stored environment block or ``default``.
        """

        entry = self.get(side, bond)
        if entry is None:
            return default
        return entry.block

    def __len__(self):
        """Return the number of persisted boundary entries."""

        return len(self.entries)

    @property
    def stats(self):
        """
        Return stack diagnostics.

        :returns: Dictionary describing boundary stack usage and storage.
        """

        left_entries = [
            entry for entry in self.entries.values() if entry.side == "left"
        ]
        right_entries = [
            entry for entry in self.entries.values() if entry.side == "right"
        ]
        advanced_entries = [
            entry for entry in self.entries.values() if str(entry.source).startswith("advanced")
        ]
        advance_timing = {}
        for entry in advanced_entries:
            for key, value in entry.advance_timing.items():
                advance_timing[key] = advance_timing.get(key, 0.0) + float(value)
        return {
            "namespace": str(self.namespace),
            "size": int(len(self.entries)),
            "left_size": int(len(left_entries)),
            "right_size": int(len(right_entries)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "puts": int(self.puts),
            "initialized_entries": int(
                sum(entry.source == "initialized" for entry in self.entries.values())
            ),
            "advanced_entries": int(len(advanced_entries)),
            "advance_timing": {
                str(key): float(value)
                for key, value in advance_timing.items()
            },
            "stored_elements": int(
                sum(entry.stored_elements for entry in self.entries.values())
            ),
            "symbolic_operator_tables": int(
                sum(
                    entry.symbolic_operator_table is not None
                    for entry in self.entries.values()
                )
            ),
            "symbolic_operator_terms": int(
                sum(
                    0
                    if entry.symbolic_operator_table is None
                    else entry.symbolic_operator_table.n_terms
                    for entry in self.entries.values()
                )
            ),
            "symbolic_operator_numeric_payloads": int(
                sum(
                    0
                    if entry.symbolic_operator_table is None
                    else len(entry.symbolic_operator_table.numeric_payloads)
                    for entry in self.entries.values()
                )
            ),
            "symbolic_operator_max_path_length": int(
                max(
                    [
                        0
                        if entry.symbolic_operator_table is None
                        else entry.symbolic_operator_table.max_path_length
                        for entry in self.entries.values()
                    ]
                    or [0]
                )
            ),
            "symbolic_operator_used_mpo_metadata": bool(
                any(
                    entry.symbolic_operator_table is not None
                    and entry.symbolic_operator_table.used_mpo_symbolic_metadata
                    for entry in self.entries.values()
                )
            ),
            "local_operator_tables": int(
                sum(len(entry.local_operator_tables) for entry in self.entries.values())
            ),
            "local_operator_table_hits": int(
                sum(
                    entry.local_operator_table_stats["hits"]
                    for entry in self.entries.values()
                )
            ),
            "local_operator_table_misses": int(
                sum(
                    entry.local_operator_table_stats["misses"]
                    for entry in self.entries.values()
                )
            ),
            "local_operator_table_puts": int(
                sum(
                    entry.local_operator_table_stats["puts"]
                    for entry in self.entries.values()
                )
            ),
            "local_operator_table_reuses": int(
                sum(
                    table.hits
                    for entry in self.entries.values()
                    for table in entry.local_operator_tables.values()
                )
            ),
            "side_operator_tables": int(
                sum(len(entry.side_operator_tables) for entry in self.entries.values())
            ),
            "side_operator_table_hits": int(
                sum(
                    entry.side_operator_table_stats["hits"]
                    for entry in self.entries.values()
                )
            ),
            "side_operator_table_misses": int(
                sum(
                    entry.side_operator_table_stats["misses"]
                    for entry in self.entries.values()
                )
            ),
            "side_operator_table_puts": int(
                sum(
                    entry.side_operator_table_stats["puts"]
                    for entry in self.entries.values()
                )
            ),
            "side_operator_table_reuses": int(
                sum(
                    table.hits
                    for entry in self.entries.values()
                    for table in entry.side_operator_tables.values()
                )
            ),
            "side_operator_table_sources": sorted(
                {
                    str(table.source)
                    for entry in self.entries.values()
                    for table in entry.side_operator_tables.values()
                }
            ),
            "side_operator_table_representations": sorted(
                {
                    str(table.representation)
                    for entry in self.entries.values()
                    for table in entry.side_operator_tables.values()
                }
            ),
            "complementary_operator_families": self.complementary_operator_family_metadata,
            "complementary_operator_stack": (
                None
                if self.complementary_operator_stack is None
                else self.complementary_operator_stack.stats
            ),
            "moving_environment_cache": self.moving_environment_cache.stats,
        }


@dataclass(frozen=True)
class OrthonormalizedLocalProblem:
    """
    Standard local problem stored in an orthonormal reduced two-site basis.

    :param basis: Parent packed two-site basis for the physical local tensor.
    :param transform: Dense map from orthonormal reduced coordinates to parent
        packed coordinates.
    :param metric: Dense local metric in the parent basis.
    :param matvec: Standard Hamiltonian action in orthonormal coordinates.
    :param full_matvec: Hamiltonian action in the parent packed basis.
    :param diag: Optional diagonal estimate in orthonormal coordinates.
    :param name: Optional diagnostic label.
    :param source: Description of where the renormalized operators are stored.
    :param cache_hit: Whether this problem was returned from an operator cache.
    :param metadata: Optional source metadata propagated from the local
        effective operator.
    """

    basis: object
    transform: np.ndarray
    metric: np.ndarray
    matvec: object
    full_matvec: object
    diag: np.ndarray | None = None
    name: str | None = None
    source: str = "renormalized_environment"
    cache_hit: bool = False
    metadata: dict | None = None

    @property
    def parent_dim(self):
        """Return the packed dimension of the parent two-site basis."""

        return int(self.transform.shape[0])

    @property
    def orthonormal_dim(self):
        """Return the reduced orthonormal local dimension."""

        return int(self.transform.shape[1])

    def to_orthonormal(self, vector):
        """
        Project a parent-basis vector into orthonormal local coordinates.

        :param vector: Packed vector in the parent local basis.
        :returns: Orthonormal reduced-coordinate vector.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.parent_dim)
        return self.transform.conj().T @ (self.metric @ vector)

    def from_orthonormal(self, vector):
        """
        Map orthonormal local coordinates back to the parent packed basis.

        :param vector: Orthonormal reduced-coordinate vector.
        :returns: Packed vector in the parent local basis.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.orthonormal_dim)
        return self.transform @ vector


@dataclass(frozen=True)
class CompiledOrthonormalBlockTerm:
    """
    One transformed block-sparse operator term.

    :param input_slice: Slice of the orthonormal input vector.
    :param output_slice: Slice of the orthonormal output vector.
    :param kernel: Dense kernel for this sector-pair contribution.
    """

    input_slice: slice
    output_slice: slice
    kernel: np.ndarray

    @property
    def elements(self):
        """Return the number of dense kernel elements stored by this term."""

        return int(np.asarray(self.kernel).size)

    def apply(self, vector, out):
        """
        Apply this term to an orthonormal-coordinate vector.

        :param vector: Input vector.
        :param out: Output vector updated in-place.
        :returns: Modified output vector.
        """

        out[self.output_slice] += self.kernel @ vector[self.input_slice]
        return out


@dataclass(frozen=True)
class CompiledOrthonormalBlockTable:
    """
    Compiled block-sparse transformed Hamiltonian table.

    :param terms: Flattened transformed block terms.
    :param dim: Total orthonormal-coordinate dimension.
    :param n_blocks: Number of orthonormal sector blocks.
    :param max_block_dim: Largest orthonormal sector-block dimension.
    :param dense_matrix: Optional dense matrix for BLAS-backed matvecs when
        the transformed local problem is small enough to materialize.
    """

    terms: tuple[CompiledOrthonormalBlockTerm, ...]
    dim: int
    n_blocks: int
    max_block_dim: int
    dense_matrix: np.ndarray | None = None

    def matvec(self, vector):
        """
        Apply the compiled transformed operator table.

        :param vector: Input vector in orthonormal coordinates.
        :returns: Output vector in orthonormal coordinates.
        """

        vector = np.asarray(vector, dtype=complex).reshape(int(self.dim))
        if self.dense_matrix is not None:
            return self.dense_matrix @ vector
        out = np.zeros_like(vector, dtype=complex)
        for term in self.terms:
            term.apply(vector, out)
        return out

    @property
    def stats(self):
        """
        Return summary statistics for this compiled operator table.

        :returns: Dictionary describing table dimensions and storage.
        """

        return {
            "kind": "block_sparse",
            "orthonormal_dim": int(self.dim),
            "n_blocks": int(self.n_blocks),
            "n_nonzero_block_terms": int(len(self.terms)),
            "max_block_orthonormal_dim": int(self.max_block_dim),
            "max_kernel_elements": int(max((term.elements for term in self.terms), default=0)),
            "stored_kernel_elements": int(sum(term.elements for term in self.terms)),
            "dense_matvec_elements": (
                0
                if self.dense_matrix is None
                else int(np.asarray(self.dense_matrix).size)
            ),
        }


@dataclass(frozen=True)
class ComplementaryFamilyTensorTable:
    """
    Family-resolved direct tensor table for complementary SU(2) operators.

    The table groups component-direct factorized actions by the symbolic
    complementary family label carried by each local term.  It is intentionally
    a thin numerical layer over compiled tensor blocks: matvecs still use the
    exact factorized kernels, but scheduling and diagnostics are resolved into
    block2-like operator families such as ``R``, ``P``, and ``Q``.

    :param family_blocks: Tuple ``((family, plan_entries), ...)``.
    :param source: Diagnostic source label for the numerical payloads.
    :param operator_table_stats: Stored family-operator table diagnostics used
        to schedule the local tensor plan.
    :param unmatched_family_groups: Local plan family groups not present in the
        stored table active-family union.
    """

    family_blocks: tuple
    source: str = "compiled_factorized_terms"
    operator_table_stats: tuple = ()
    unmatched_family_groups: tuple = ()
    backend: str = "compiled_factorized_term"
    native_kernel_elements: int = 0
    factor_kernel_elements: int = 0

    @classmethod
    def from_component_direct_plan(cls, plan, *, source="compiled_factorized_terms"):
        """
        Build a family table from a component-direct factorized plan.

        :param plan: Plan entries produced by
            :meth:`DirectOrthonormalFactorizedTable._build_component_direct_plan`.
        :param source: Diagnostic label for the plan source.
        :returns: Family table, or ``None`` when no plan is available.
        """

        if plan is None:
            return None
        grouped = OrderedDict()
        for entry in plan:
            term = entry[4]
            names = tuple(getattr(term, "family_names", ()) or ())
            family = "+".join(str(name) for name in names) if names else "unlabeled"
            grouped.setdefault(family, []).append(
                ComplementaryFamilyApplyEntry.from_plan_entry(entry, family_names=names)
            )
        return cls(
            family_blocks=tuple(
                (str(family), tuple(entries))
                for family, entries in sorted(grouped.items(), key=lambda item: item[0])
            ),
            source=str(source),
        )

    @classmethod
    def from_family_operator_tables(cls, plan, family_operator_tables):
        """
        Build a tensor table scheduled by stored family operator tables.

        The actual numeric term kernels remain the compiled factorized tensor
        blocks, but family grouping is now driven by the active family labels
        carried by the stored left/right renormalized family tables.

        :param plan: Component-direct factorized plan.
        :param family_operator_tables: Stored boundary family tables.
        :returns: Family tensor table, or ``None`` when no plan is available.
        """

        if plan is None:
            return None
        tables = tuple(table for table in tuple(family_operator_tables or ()) if table is not None)
        if not tables:
            return cls.from_component_direct_plan(plan)
        active_families = set()
        family_source_tables = {}
        for table in tables:
            for name in table.active_family_names:
                active_families.add(name)
                family_source_tables.setdefault(str(name), []).append(
                    (str(table.side), int(table.bond))
                )
        grouped = OrderedDict()
        unmatched = set()
        for entry in plan:
            term = entry[4]
            names = tuple(str(name) for name in (getattr(term, "family_names", ()) or ()))
            group_names = tuple(name for name in names if name in active_families)
            if not group_names:
                group_names = names
                if names:
                    unmatched.add("+".join(names))
            family = "+".join(group_names) if group_names else "unlabeled"
            source_tables = tuple(
                source
                for name in group_names
                for source in family_source_tables.get(str(name), ())
            )
            grouped.setdefault(family, []).append(
                ComplementaryFamilyApplyEntry.from_plan_entry(
                    entry,
                    family_names=group_names,
                    source_tables=source_tables,
                ).with_factor_kernel().with_native_kernel()
            )
        factor_kernel_elements = int(
            sum(
                0
                if apply_entry.factor_kernel is None
                else apply_entry.factor_kernel.stored_elements
                for entries in grouped.values()
                for apply_entry in entries
            )
        )
        native_kernel_elements = int(
            sum(
                0
                if apply_entry.native_kernel is None
                else np.asarray(apply_entry.native_kernel).size
                for entries in grouped.values()
                for apply_entry in entries
            )
        )
        return cls(
            family_blocks=tuple(
                (str(family), tuple(entries))
                for family, entries in sorted(grouped.items(), key=lambda item: item[0])
            ),
            source="renormalized_family_operator_tables",
            operator_table_stats=tuple(table.stats for table in tables),
            unmatched_family_groups=tuple(sorted(unmatched)),
            backend=(
                "family_table_hybrid_kernel"
                if native_kernel_elements > 0 and factor_kernel_elements > 0
                else (
                    "family_table_dense_kernel"
                    if native_kernel_elements > 0
                    else "family_table_factor_kernel"
                )
            ),
            native_kernel_elements=int(native_kernel_elements),
            factor_kernel_elements=int(factor_kernel_elements),
        )

    @property
    def family_names(self):
        """Return the table's family labels."""

        names = set()
        for family, _entries in self.family_blocks:
            names.update(str(family).split("+"))
        names.discard("")
        return tuple(sorted(names))

    @property
    def family_term_counts(self):
        """Return per-family direct tensor term counts."""

        counts = {}
        for family, entries in self.family_blocks:
            for name in str(family).split("+"):
                if name:
                    counts[name] = counts.get(name, 0) + int(len(entries))
        return dict(sorted(counts.items()))

    @property
    def n_terms(self):
        """Return the total number of direct tensor plan entries."""

        return int(sum(len(entries) for _family, entries in self.family_blocks))

    def matvec(self, vector, component_basis):
        """
        Apply the family-grouped direct tensor table.

        :param vector: Input vector in orthonormal component coordinates.
        :param component_basis: Component orthonormal basis owning transforms.
        :returns: Output vector in orthonormal component coordinates.
        """

        parent_inputs = []
        parent_outputs = []
        for idx, indices in enumerate(component_basis.component_indices):
            transform = component_basis.component_transforms[idx]
            start = int(component_basis.orth_offsets[idx])
            stop = start + int(transform.shape[1])
            parent_inputs.append(transform @ vector[start:stop])
            parent_outputs.append(np.zeros(int(np.asarray(indices).size), dtype=complex))
        for _family, entries in self.family_blocks:
            for entry in entries:
                block_in = parent_inputs[int(entry.in_comp)][entry.in_slice].reshape(
                    entry.input_entry.shape
                )
                parent_outputs[int(entry.out_comp)][entry.out_slice] += entry.apply_block(
                    block_in
                )
        out = np.zeros(int(component_basis.orthonormal_dim), dtype=complex)
        for idx, parent_out in enumerate(parent_outputs):
            transform = component_basis.component_transforms[idx]
            start = int(component_basis.orth_offsets[idx])
            stop = start + int(transform.shape[1])
            out[start:stop] = transform.conj().T @ parent_out
        return out

    @property
    def stats(self):
        """
        Return family-table diagnostics.

        :returns: Dictionary describing family labels and payload counts.
        """

        return {
            "kind": "complementary_family_tensor_table",
            "source": str(self.source),
            "family_names": self.family_names,
            "family_groups": tuple(family for family, _entries in self.family_blocks),
            "family_term_counts": self.family_term_counts,
            "n_family_blocks": int(len(self.family_blocks)),
            "n_terms": int(self.n_terms),
            "operator_table_backed": bool(self.operator_table_stats),
            "operator_tables": self.operator_table_stats,
            "unmatched_family_groups": tuple(self.unmatched_family_groups),
            "unmatched_family_group_count": int(len(self.unmatched_family_groups)),
            "backend": str(self.backend),
            "payload_native_backend": bool(
                self.backend
                in {
                    "family_table_dense_kernel",
                    "family_table_factor_kernel",
                    "family_table_hybrid_kernel",
                }
            ),
            "native_kernel_elements": int(self.native_kernel_elements),
            "factor_kernel_elements": int(self.factor_kernel_elements),
        }


@dataclass(frozen=True)
class DirectOrthonormalFactorizedTable:
    """
    Matrix-free transformed Hamiltonian table for factorized local operators.

    This is the block2-like path for component local problems: the table does
    not own dense transformed sector-pair kernels.  When compiled factorized
    terms and entry components are supplied, matvecs stay in component-local
    buffers and avoid calling the full packed parent matvec.

    :param component_basis: Orthonormal component basis defining ``X``.
    :param packed_matvec: Parent-basis Hamiltonian matvec fallback.
    :param source: Diagnostic source label.
    :param compiled_factorized_terms: Optional compiled factorized block
        kernels for component-direct application.
    :param components: Optional entry-index components aligned with
        ``component_basis``.
    """

    component_basis: object
    packed_matvec: object
    source: str = "direct_factorized"
    compiled_factorized_terms: object | None = None
    components: tuple | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "_component_direct_plan",
            self._build_component_direct_plan(),
        )
        object.__setattr__(
            self,
            "_complementary_family_tensor_table",
            (
                self._build_complementary_family_tensor_table(
                    getattr(self, "_component_direct_plan", None)
                )
                if self.uses_complementary_payload_tensor_kernel
                else None
            ),
        )
        object.__setattr__(
            self,
            "_component_parent_blocks",
            (
                None
                if self.uses_complementary_payload_tensor_kernel
                else self._build_component_parent_blocks()
            ),
        )

    @property
    def dim(self):
        """Return the orthonormal local dimension."""

        return int(self.component_basis.orthonormal_dim)

    def matvec(self, vector):
        """
        Apply ``X^H H X`` without materializing transformed kernels.

        :param vector: Input vector in orthonormal component coordinates.
        :returns: Transformed Hamiltonian action.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.dim)
        family_table = getattr(self, "_complementary_family_tensor_table", None)
        if family_table is not None:
            return family_table.matvec(vector, self.component_basis)
        parent_blocks = getattr(self, "_component_parent_blocks", None)
        if parent_blocks is not None:
            return self._component_parent_block_matvec(vector, parent_blocks)
        plan = getattr(self, "_component_direct_plan", None)
        if plan is not None:
            return self._component_direct_matvec(vector, plan)
        parent = self.component_basis.from_orthonormal(vector)
        parent_out = np.asarray(
            self.packed_matvec(parent),
            dtype=complex,
        ).reshape(self.component_basis.parent_dim)
        out = np.zeros(self.dim, dtype=complex)
        for idx, indices in enumerate(self.component_basis.component_indices):
            start = int(self.component_basis.orth_offsets[idx])
            stop = start + int(self.component_basis.component_transforms[idx].shape[1])
            out[start:stop] = (
                self.component_basis.component_transforms[idx].conj().T
                @ parent_out[indices]
            )
        return out

    @property
    def uses_component_direct_kernel(self):
        """
        Return whether this table owns a component-direct factorized plan.

        :returns: ``True`` when matvecs avoid the full packed parent matvec.
        """

        return getattr(self, "_component_direct_plan", None) is not None

    @property
    def uses_component_parent_block_kernel(self):
        """
        Return whether matvecs use compiled parent component blocks.

        :returns: ``True`` when recursive matvecs are BLAS-backed by parent
            component blocks rather than Python per-term contractions.
        """

        return getattr(self, "_component_parent_blocks", None) is not None

    @property
    def uses_complementary_payload_tensor_kernel(self):
        """
        Return whether complementary payloads force tensor-level contractions.

        This is the direct block2-like experimental path: matvecs consume the
        payload-backed compiled renormalized tensor factors through
        ``CompiledFactorizedBlock.apply_block`` instead of first materializing
        dense parent component kernels.
        """

        compiled = self.compiled_factorized_terms
        return bool(
            compiled is not None
            and getattr(compiled, "complementary_payload_backed", False)
            and getattr(compiled, "prefer_complementary_payload_tensor_matvec", False)
        )

    @property
    def uses_complementary_family_table_kernel(self):
        """
        Return whether matvecs use a family-resolved complementary table.

        :returns: ``True`` when complementary payload tensor terms are grouped
            and applied through :class:`ComplementaryFamilyTensorTable`.
        """

        return getattr(self, "_complementary_family_tensor_table", None) is not None

    def _build_complementary_family_tensor_table(self, plan):
        """
        Build the family-resolved tensor table for complementary matvecs.

        :param plan: Component-direct factorized application plan.
        :returns: :class:`ComplementaryFamilyTensorTable` or ``None``.
        """

        table_objects = getattr(
            self.compiled_factorized_terms,
            "complementary_family_operator_table_objects",
            (),
        )
        if table_objects:
            return ComplementaryFamilyTensorTable.from_family_operator_tables(
                plan,
                table_objects,
            )
        return ComplementaryFamilyTensorTable.from_component_direct_plan(
            plan,
            source=(
                "renormalized_family_operator_tables"
                if self.uses_complementary_family_operator_table_source
                else "compiled_factorized_terms"
            ),
        )

    @property
    def uses_complementary_family_operator_table_source(self):
        """
        Return whether complementary matvecs are backed by stored family tables.

        The numerical tensor kernels are still the compiled factorized blocks,
        but this flag verifies that the boundary stack owns independent
        family-resolved renormalized operator tables for the same local solve.
        """

        metadata = getattr(
            self.compiled_factorized_terms,
            "complementary_family_operator_tables",
            None,
        )
        return bool(
            metadata is not None
            and metadata.get("family_operator_table_backed", False)
        )


    def _build_component_direct_plan(self):
        """
        Build a component-local factorized block application plan.

        :returns: Tuple of plan entries or ``None`` when metadata is missing.
        """

        compiled = self.compiled_factorized_terms
        components = self.components
        basis = getattr(self.component_basis, "parent_basis", None)
        if compiled is None or components is None or basis is None:
            return None
        compiled_basis = getattr(compiled, "basis", None)
        if compiled_basis is not basis and not basis.compatible_with_layout(
            getattr(compiled_basis, "entries", basis.entries)
        ):
            return None
        entry_to_component = {}
        for comp_idx, component in enumerate(components):
            cursor = 0
            for entry_idx in component:
                entry = basis.entries[int(entry_idx)]
                entry_to_component[int(entry_idx)] = (
                    int(comp_idx),
                    slice(cursor, cursor + int(entry.size)),
                )
                cursor += int(entry.size)
        plan = []
        for in_idx, terms in enumerate(getattr(compiled, "items", ())):
            in_info = entry_to_component.get(int(in_idx))
            if in_info is None:
                return None
            in_comp, in_slice = in_info
            for term in terms:
                out_idx = basis.entry_index(term.output_entry.key)
                out_info = entry_to_component.get(int(out_idx))
                if out_info is None:
                    return None
                out_comp, out_slice = out_info
                plan.append((int(in_comp), int(out_comp), in_slice, out_slice, term))
        return tuple(plan)

    def _build_component_parent_blocks(self):
        """
        Assemble dense parent component blocks for recursive matvecs.

        These blocks live in the non-orthonormal component parent basis.  They
        avoid materializing transformed ``X^H H X`` kernels while making each
        Davidson matvec a small set of BLAS matrix-vector products.

        :returns: Tuple ``((in_comp, out_comp, block), ...)`` or ``None``.
        """

        plan = getattr(self, "_component_direct_plan", None)
        if plan is None:
            return None
        component_dims = tuple(
            int(np.asarray(indices).size)
            for indices in self.component_basis.component_indices
        )
        blocks = {}
        for in_comp, out_comp, in_slice, out_slice, term in plan:
            key = (int(in_comp), int(out_comp))
            block = blocks.get(key)
            if block is None:
                block = np.zeros(
                    (component_dims[int(out_comp)], component_dims[int(in_comp)]),
                    dtype=complex,
                )
                blocks[key] = block
            kernel = term.kernel_matrix(
                term.input_entry.shape,
                max_elements=max(int(term.input_entry.size) * int(term.output_entry.size), 1),
            )
            if kernel is None:
                return None
            block[out_slice, in_slice] += np.asarray(kernel, dtype=complex)
        return tuple(
            (in_comp, out_comp, np.ascontiguousarray(block))
            for (in_comp, out_comp), block in sorted(blocks.items())
        )

    def _component_parent_block_matvec(self, vector, parent_blocks):
        """
        Apply recursive parent component blocks in orthonormal coordinates.

        :param vector: Orthonormal-coordinate input vector.
        :param parent_blocks: Compiled parent component blocks.
        :returns: Orthonormal-coordinate output vector.
        """

        parent_inputs = []
        parent_outputs = []
        for idx, indices in enumerate(self.component_basis.component_indices):
            transform = self.component_basis.component_transforms[idx]
            start = int(self.component_basis.orth_offsets[idx])
            stop = start + int(transform.shape[1])
            parent_inputs.append(transform @ vector[start:stop])
            parent_outputs.append(np.zeros(int(np.asarray(indices).size), dtype=complex))
        for in_comp, out_comp, block in parent_blocks:
            parent_outputs[int(out_comp)] += block @ parent_inputs[int(in_comp)]
        out = np.zeros(self.dim, dtype=complex)
        for idx, parent_out in enumerate(parent_outputs):
            transform = self.component_basis.component_transforms[idx]
            start = int(self.component_basis.orth_offsets[idx])
            stop = start + int(transform.shape[1])
            out[start:stop] = transform.conj().T @ parent_out
        return out

    def _component_direct_matvec(self, vector, plan):
        """
        Apply factorized kernels through component-local parent buffers.

        :param vector: Orthonormal-coordinate input vector.
        :param plan: Component-direct factorized block plan.
        :returns: Orthonormal-coordinate output vector.
        """

        parent_inputs = []
        parent_outputs = []
        for idx, indices in enumerate(self.component_basis.component_indices):
            transform = self.component_basis.component_transforms[idx]
            start = int(self.component_basis.orth_offsets[idx])
            stop = start + int(transform.shape[1])
            parent_inputs.append(transform @ vector[start:stop])
            parent_outputs.append(np.zeros(int(np.asarray(indices).size), dtype=complex))
        for in_comp, out_comp, in_slice, out_slice, term in plan:
            block_in = parent_inputs[in_comp][in_slice].reshape(term.input_entry.shape)
            parent_outputs[out_comp][out_slice] += term.apply_block(block_in)
        out = np.zeros(self.dim, dtype=complex)
        for idx, parent_out in enumerate(parent_outputs):
            transform = self.component_basis.component_transforms[idx]
            start = int(self.component_basis.orth_offsets[idx])
            stop = start + int(transform.shape[1])
            out[start:stop] = transform.conj().T @ parent_out
        return out

    def complementary_family_table_equivalence_residual(self, seed=0):
        """
        Compare family-table matvecs against the raw component-direct plan.

        :param seed: Random seed for the probe vector.
        :returns: Relative 2-norm residual, or ``None`` when either path is
            unavailable.
        """

        family_table = getattr(self, "_complementary_family_tensor_table", None)
        plan = getattr(self, "_component_direct_plan", None)
        if family_table is None or plan is None:
            return None
        rng = np.random.default_rng(int(seed))
        probe = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
        direct = self._component_direct_matvec(probe, plan)
        grouped = family_table.matvec(probe, self.component_basis)
        scale = max(float(np.linalg.norm(direct)), 1.0)
        return float(np.linalg.norm(grouped - direct) / scale)

    @property
    def stats(self):
        """
        Return summary statistics for the matrix-free transformed table.

        :returns: Dictionary describing the direct factorized matvec table.
        """

        complementary_families = getattr(
            self.compiled_factorized_terms,
            "complementary_operator_families",
            None,
        )
        complementary_metadata = (
            complementary_families.as_metadata()
            if hasattr(complementary_families, "as_metadata")
            else None
        )
        complementary_payloads = getattr(
            self.compiled_factorized_terms,
            "complementary_boundary_payloads",
            None,
        )
        complementary_family_operator_tables = getattr(
            self.compiled_factorized_terms,
            "complementary_family_operator_tables",
            None,
        )
        complementary_payload_terms = int(
            0
            if complementary_payloads is None
            else complementary_payloads.get("numeric_payload_terms", 0)
        )
        family_names = tuple(
            getattr(self.compiled_factorized_terms, "family_names", ()) or ()
        )
        family_term_counts = dict(
            getattr(self.compiled_factorized_terms, "family_term_counts", {}) or {}
        )
        family_table = getattr(self, "_complementary_family_tensor_table", None)
        return {
            "kind": (
                "recursive_parent_block_factorized"
                if self.uses_component_parent_block_kernel
                else (
                    "complementary_family_table_factorized"
                    if self.uses_complementary_family_table_kernel
                    else (
                        "direct_component_factorized"
                        if self.uses_component_direct_kernel
                        else "direct_factorized"
                    )
                )
            ),
            "source": str(self.source),
            "orthonormal_dim": int(self.dim),
            "n_blocks": int(self.component_basis.n_components),
            "n_nonzero_block_terms": int(
                len(
                    getattr(self, "_component_parent_blocks", None)
                    or getattr(self, "_component_direct_plan", ())
                    or (1,)
                )
            ),
            "max_block_orthonormal_dim": int(
                max(
                    (
                        transform.shape[1]
                        for transform in self.component_basis.component_transforms
                    ),
                    default=0,
                )
            ),
            "stored_kernel_elements": 0,
            "dense_matvec_elements": 0,
            "component_direct_kernel": bool(self.uses_component_direct_kernel),
            "component_parent_block_kernel": bool(
                self.uses_component_parent_block_kernel
            ),
            "complementary_payload_tensor_kernel": bool(
                self.uses_complementary_payload_tensor_kernel
                and self.uses_component_direct_kernel
            ),
            "complementary_family_table_kernel": bool(
                self.uses_complementary_family_table_kernel
            ),
            "complementary_family_table_matvec": bool(
                self.uses_complementary_family_table_kernel
            ),
            "complementary_family_table": (
                None if family_table is None else family_table.stats
            ),
            "complementary_family_table_source": (
                None if family_table is None else str(family_table.source)
            ),
            "complementary_family_operator_table_source": bool(
                self.uses_complementary_family_operator_table_source
            ),
            "complementary_family_operator_tables": (
                complementary_family_operator_tables
            ),
            "complementary_direct_matvec": bool(complementary_metadata is not None),
            "complementary_operator_families": complementary_metadata,
            "complementary_payload_backed": bool(
                complementary_payloads is not None
                and complementary_payloads.get("payload_backed", False)
            ),
            "complementary_boundary_payloads": complementary_payloads,
            "complementary_payload_terms": int(complementary_payload_terms),
            "family_resolved_tensor_kernel": bool(family_names),
            "family_names": family_names,
            "family_term_counts": family_term_counts,
            "component_parent_block_elements": int(
                sum(
                    np.asarray(block).size
                    for _in_comp, _out_comp, block in (
                        getattr(self, "_component_parent_blocks", None) or ()
                    )
                )
            ),
        }


@dataclass(frozen=True)
class BlockSparseOrthonormalizedLocalProblem:
    """
    Block-sparse standard local problem in an orthonormal reduced basis.

    :param basis: Parent packed two-site basis.
    :param block_transforms: Per-entry transforms from orthonormal block
        coordinates to parent packed block coordinates.
    :param metric_blocks: Per-entry parent-basis metric blocks.
    :param block_table: Compiled transformed output block kernels.
    :param orth_offsets: Starting offsets of each orthonormal block.
    :param full_matvec: Hamiltonian action in the parent packed basis.
    :param diag: Optional diagonal estimate in orthonormal coordinates.
    :param name: Optional diagnostic label.
    :param source: Description of the renormalized-operator owner.
    :param cache_hit: Whether this problem was returned from an operator cache.
    :param metadata: Optional source metadata propagated from the local
        effective operator.
    """

    basis: object
    block_transforms: tuple
    metric_blocks: tuple
    block_table: CompiledOrthonormalBlockTable
    orth_offsets: tuple[int, ...]
    full_matvec: object
    diag: np.ndarray | None = None
    name: str | None = None
    source: str = "block_sparse_operator_table"
    cache_hit: bool = False
    metadata: dict | None = None

    @property
    def parent_dim(self):
        """Return the packed dimension of the parent two-site basis."""

        return int(self.basis.size)

    @property
    def orthonormal_dim(self):
        """Return the total dimension of the block-sparse orthonormal basis."""

        if not self.block_transforms:
            return 0
        last = len(self.block_transforms) - 1
        return int(self.orth_offsets[last] + self.block_transforms[last].shape[1])

    @property
    def table_stats(self):
        """
        Return summary statistics for the transformed local operator table.

        :returns: Dictionary describing block-sparse table storage.
        """

        stats = dict(self.block_table.stats)
        stats.update({"parent_dim": int(self.parent_dim)})
        return stats

    def _orth_slice(self, index):
        start = int(self.orth_offsets[int(index)])
        stop = start + int(self.block_transforms[int(index)].shape[1])
        return slice(start, stop)

    def to_orthonormal(self, vector):
        """
        Project a parent-basis vector into block-sparse orthonormal coordinates.

        :param vector: Packed vector in the parent local basis.
        :returns: Packed vector in the orthonormal block basis.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.parent_dim)
        out = np.zeros(self.orthonormal_dim, dtype=complex)
        for idx, entry in enumerate(self.basis):
            parent_piece = vector[entry.slice]
            metric_piece = self.metric_blocks[idx] @ parent_piece
            out[self._orth_slice(idx)] = self.block_transforms[idx].conj().T @ metric_piece
        return out

    def from_orthonormal(self, vector):
        """
        Map block-sparse orthonormal coordinates back to the parent basis.

        :param vector: Packed vector in the orthonormal block basis.
        :returns: Packed vector in the parent local basis.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.orthonormal_dim)
        out = np.zeros(self.parent_dim, dtype=complex)
        for idx, entry in enumerate(self.basis):
            out[entry.slice] = self.block_transforms[idx] @ vector[self._orth_slice(idx)]
        return out

    def matvec(self, vector):
        """
        Apply the transformed block-sparse Hamiltonian table.

        :param vector: Packed vector in the orthonormal block basis.
        :returns: Packed transformed Hamiltonian action.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.orthonormal_dim)
        return self.block_table.matvec(vector)

    def metric_matvec(self, vector):
        """
        Apply the parent-basis block-diagonal local metric.

        :param vector: Packed vector in the parent local basis.
        :returns: Packed metric action in the parent local basis.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.parent_dim)
        out = np.zeros_like(vector, dtype=complex)
        for idx, entry in enumerate(self.basis):
            out[entry.slice] = self.metric_blocks[idx] @ vector[entry.slice]
        return out


@dataclass(frozen=True)
class RenormalizedComponentBasis:
    """
    Orthonormal reduced basis built from metric-connected components.

    :param parent_basis: Parent packed two-site basis.
    :param component_indices: Parent packed indices for each metric-connected
        component.
    :param component_transforms: Per-component maps from orthonormal
        coordinates to parent component coordinates.
    :param metric_blocks: Exact per-component parent-basis metric blocks.
    :param orth_offsets: Starting offsets of each orthonormal component.
    """

    parent_basis: object
    component_indices: tuple[np.ndarray, ...]
    component_transforms: tuple[np.ndarray, ...]
    metric_blocks: tuple[np.ndarray, ...]
    orth_offsets: tuple[int, ...]

    @property
    def parent_dim(self):
        """Return the packed dimension of the parent two-site basis."""

        return int(self.parent_basis.size)

    @property
    def orthonormal_dim(self):
        """Return the total dimension of the component orthonormal basis."""

        if not self.component_transforms:
            return 0
        last = len(self.component_transforms) - 1
        return int(self.orth_offsets[last] + self.component_transforms[last].shape[1])

    @property
    def n_components(self):
        """Return the number of metric-connected components."""

        return int(len(self.component_indices))

    @property
    def max_component_parent_dim(self):
        """Return the largest parent packed dimension of any component."""

        return int(max((indices.size for indices in self.component_indices), default=0))

    @property
    def stats(self):
        """
        Return summary statistics for the component basis.

        :returns: Dictionary describing component counts and dimensions.
        """

        return {
            "basis_kind": "metric_connected_components",
            "parent_dim": int(self.parent_dim),
            "orthonormal_dim": int(self.orthonormal_dim),
            "n_components": int(self.n_components),
            "max_component_parent_dim": int(self.max_component_parent_dim),
        }

    def _orth_slice(self, index):
        start = int(self.orth_offsets[int(index)])
        stop = start + int(self.component_transforms[int(index)].shape[1])
        return slice(start, stop)

    def to_orthonormal(self, vector):
        """
        Project a parent-basis vector into component orthonormal coordinates.

        :param vector: Packed vector in the parent local basis.
        :returns: Packed vector in the orthonormal component basis.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.parent_dim)
        out = np.zeros(self.orthonormal_dim, dtype=complex)
        for idx, indices in enumerate(self.component_indices):
            parent_piece = vector[indices]
            metric_piece = self.metric_blocks[idx] @ parent_piece
            out[self._orth_slice(idx)] = self.component_transforms[idx].conj().T @ metric_piece
        return out

    def from_orthonormal(self, vector):
        """
        Map component orthonormal coordinates back to the parent basis.

        :param vector: Packed vector in the orthonormal component basis.
        :returns: Packed vector in the parent local basis.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.orthonormal_dim)
        out = np.zeros(self.parent_dim, dtype=complex)
        for idx, indices in enumerate(self.component_indices):
            out[indices] = self.component_transforms[idx] @ vector[self._orth_slice(idx)]
        return out

    def matvec(self, vector):
        """
        Apply the transformed component-sparse Hamiltonian table.

        :param vector: Packed vector in the orthonormal component basis.
        :returns: Packed transformed Hamiltonian action.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.orthonormal_dim)
        return self.block_table.matvec(vector)

    def metric_matvec(self, vector):
        """
        Apply the parent-basis component-block local metric.

        :param vector: Packed vector in the parent local basis.
        :returns: Packed metric action in the parent local basis.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.parent_dim)
        out = np.zeros_like(vector, dtype=complex)
        for idx, indices in enumerate(self.component_indices):
            out[indices] = self.metric_blocks[idx] @ vector[indices]
        return out


@dataclass(frozen=True)
class ComponentOrthonormalizedLocalProblem:
    """
    Component-sparse standard local problem in an orthonormal reduced basis.

    The component basis is a first-class object owned by the local
    renormalized problem, so future environment stacks can persist and update
    it independently of the Hamiltonian table.

    :param component_basis: Orthonormal component basis.
    :param block_table: Compiled transformed Hamiltonian kernels.
    :param full_matvec: Hamiltonian action in the parent packed basis.
    :param diag: Optional diagonal estimate in orthonormal coordinates.
    :param name: Optional diagnostic label.
    :param source: Description of the renormalized-operator owner.
    :param cache_hit: Whether this problem was returned from an operator cache.
    :param metadata: Optional source metadata propagated from the local
        effective operator.
    """

    component_basis: RenormalizedComponentBasis
    block_table: CompiledOrthonormalBlockTable
    full_matvec: object
    diag: np.ndarray | None = None
    name: str | None = None
    source: str = "component_sparse_operator_table"
    cache_hit: bool = False
    metadata: dict | None = None

    @property
    def basis(self):
        """Return the parent packed two-site basis."""

        return self.component_basis.parent_basis

    @property
    def component_indices(self):
        """Return parent packed indices for each component."""

        return self.component_basis.component_indices

    @property
    def component_transforms(self):
        """Return per-component orthonormal transforms."""

        return self.component_basis.component_transforms

    @property
    def metric_blocks(self):
        """Return exact per-component metric blocks."""

        return self.component_basis.metric_blocks

    @property
    def orth_offsets(self):
        """Return orthonormal-coordinate offsets for each component."""

        return self.component_basis.orth_offsets

    @property
    def parent_dim(self):
        """Return the packed dimension of the parent two-site basis."""

        return self.component_basis.parent_dim

    @property
    def orthonormal_dim(self):
        """Return the total dimension of the component orthonormal basis."""

        return self.component_basis.orthonormal_dim

    @property
    def table_stats(self):
        """
        Return summary statistics for the transformed local operator table.

        :returns: Dictionary describing component-sparse table storage.
        """

        stats = dict(self.block_table.stats)
        stats.update({"kind": "component_sparse"})
        stats.update(self.component_basis.stats)
        return stats

    def to_orthonormal(self, vector):
        """
        Project a parent-basis vector into component orthonormal coordinates.

        :param vector: Packed vector in the parent local basis.
        :returns: Packed vector in the orthonormal component basis.
        """

        return self.component_basis.to_orthonormal(vector)

    def from_orthonormal(self, vector):
        """
        Map component orthonormal coordinates back to the parent basis.

        :param vector: Packed vector in the orthonormal component basis.
        :returns: Packed vector in the parent local basis.
        """

        return self.component_basis.from_orthonormal(vector)

    def metric_matvec(self, vector):
        """
        Apply the parent-basis component-block local metric.

        :param vector: Packed vector in the parent local basis.
        :returns: Packed metric action in the parent local basis.
        """

        return self.component_basis.metric_matvec(vector)

    def matvec(self, vector):
        """
        Apply the transformed component-sparse Hamiltonian table.

        :param vector: Packed vector in the orthonormal component basis.
        :returns: Packed transformed Hamiltonian action.
        """

        vector = np.asarray(vector, dtype=complex).reshape(self.orthonormal_dim)
        return self.block_table.matvec(vector)


def compile_orthonormal_block_table(block_terms, block_transforms, orth_offsets):
    """
    Compile transformed sector-pair kernels into flat matvec terms.

    :param block_terms: Per-input-entry ``(out_idx, kernel)`` terms.
    :param block_transforms: Per-entry orthonormal transforms.
    :param orth_offsets: Per-entry offsets in orthonormal coordinates.
    :returns: ``CompiledOrthonormalBlockTable``.
    """

    def _slice(idx):
        start = int(orth_offsets[int(idx)])
        stop = start + int(block_transforms[int(idx)].shape[1])
        return slice(start, stop)

    compiled_terms = []
    for in_idx, terms in enumerate(block_terms):
        in_slice = _slice(in_idx)
        for out_idx, kernel in terms:
            compiled_terms.append(
                CompiledOrthonormalBlockTerm(
                    input_slice=in_slice,
                    output_slice=_slice(out_idx),
                    kernel=np.ascontiguousarray(kernel),
                )
            )
    if block_transforms:
        last = len(block_transforms) - 1
        dim = int(orth_offsets[last] + block_transforms[last].shape[1])
        max_block_dim = max(int(transform.shape[1]) for transform in block_transforms)
    else:
        dim = 0
        max_block_dim = 0
    dense_matrix = None
    if dim > 0 and dim * dim <= _ORTHONORMAL_BLOCK_DENSE_MATVEC_MAX_ELEMENTS:
        dense_matrix = np.zeros((dim, dim), dtype=complex)
        for term in compiled_terms:
            dense_matrix[term.output_slice, term.input_slice] += term.kernel
    return CompiledOrthonormalBlockTable(
        terms=tuple(compiled_terms),
        dim=int(dim),
        n_blocks=int(len(block_transforms)),
        max_block_dim=int(max_block_dim),
        dense_matrix=None if dense_matrix is None else np.ascontiguousarray(dense_matrix),
    )
