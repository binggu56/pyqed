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
_DIRECT_FACTORIZED_ORTHONORMAL_BLOCK_MAX_ELEMENTS = 16_000_000
_DIRECT_FACTORIZED_ORTHONORMAL_DENSE_MAX_ELEMENTS = 16_000_000


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
                        values.append((q_lb, q_p1b, middle_idx, factor))
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
                        values.append((q_rb, q_p2b, middle_idx, factor))
            if values:
                out[(q_rk, q_p2k)] = tuple(values)
    return out


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
    for transition in transitions:
        if direction == "left":
            parent_channel = int(transition.left_channel)
            child_channel = int(transition.right_channel)
        else:
            parent_channel = int(transition.right_channel)
            child_channel = int(transition.left_channel)
        if active is not None and child_channel not in active:
            continue
        multiplicity = sum(
            int(term.multiplicity)
            for term in terms_by_channel.get(parent_channel, ())
        )
        if multiplicity:
            counts[child_channel] = counts.get(child_channel, 0) + multiplicity
    return {
        channel: (
            SymbolicRenormalizedOperatorTerm(
                channel=channel,
                path=tuple(("compact", step, int(channel)) for step in range(child_depth)),
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
class ComplementaryRenormalizedOperatorEntry:
    """
    Recursive complementary-operator boundary record.

    The record tracks block2-style complementary family ownership alongside
    the ordinary left/right environment stack.  Numeric complementary tensors
    are still supplied by the local symbolic/factorized tables, but the sweep
    now has a first-class recursive stack boundary for future direct
    ``S/R/A/P/B/Q`` updates.

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
    def family_names(self):
        """
        Return complementary family labels.

        :returns: Tuple such as ``("S", "R", "A", "P", "B", "Q")``.
        """

        return tuple(getattr(self.families, "names", ()))

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
        )
        self.entries[entry.key] = entry
        self.puts += 1
        if parent_key is not None:
            self.advances += 1
        return entry

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

    def set_complementary_operator_families(self, families):
        """
        Attach block2-style complementary operator families to this stack.

        :param families: Object exposing ``as_metadata()``, such as the qchem
            spatial ``S/R/A/P/B/Q`` family container.
        :returns: ``self`` for call chaining.
        """

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
            self.complementary_operator_stack.put(
                normalized_side,
                normalized_bond,
                signature=signature,
                source=str(source),
                parent_key=parent_key,
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
        component_direct_plan = self._build_component_direct_plan()
        component_parent_blocks = self._build_component_parent_blocks(
            component_direct_plan,
        )
        component_orthonormal_blocks = self._build_component_orthonormal_blocks(
            component_parent_blocks,
        )
        component_orthonormal_dense_matrix = self._build_component_orthonormal_dense_matrix(
            component_orthonormal_blocks,
        )
        object.__setattr__(
            self,
            "_component_direct_plan",
            component_direct_plan,
        )
        object.__setattr__(
            self,
            "_component_parent_blocks",
            component_parent_blocks,
        )
        object.__setattr__(
            self,
            "_component_orthonormal_blocks",
            component_orthonormal_blocks,
        )
        object.__setattr__(
            self,
            "_component_orthonormal_dense_matrix",
            component_orthonormal_dense_matrix,
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
        orthonormal_dense = getattr(self, "_component_orthonormal_dense_matrix", None)
        if orthonormal_dense is not None:
            return orthonormal_dense @ vector
        orthonormal_blocks = getattr(self, "_component_orthonormal_blocks", None)
        if orthonormal_blocks is not None:
            return self._component_orthonormal_block_matvec(vector, orthonormal_blocks)
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
    def uses_component_orthonormal_block_kernel(self):
        """
        Return whether matvecs use transformed orthonormal component blocks.

        :returns: ``True`` when the parent component blocks were projected
            once into ``X^H H X`` block form.
        """

        return getattr(self, "_component_orthonormal_blocks", None) is not None

    @property
    def uses_component_orthonormal_dense_kernel(self):
        """
        Return whether matvecs use one dense orthonormal local matrix.

        :returns: ``True`` when transformed component blocks were assembled
            into a single BLAS-backed matrix-vector kernel.
        """

        return getattr(self, "_component_orthonormal_dense_matrix", None) is not None

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

    def _build_component_parent_blocks(self, plan=None):
        """
        Assemble dense parent component blocks for recursive matvecs.

        These blocks live in the non-orthonormal component parent basis.  They
        avoid materializing transformed ``X^H H X`` kernels while making each
        Davidson matvec a small set of BLAS matrix-vector products.

        :returns: Tuple ``((in_comp, out_comp, block), ...)`` or ``None``.
        """

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

    def _build_component_orthonormal_blocks(self, parent_blocks):
        """
        Project moderate parent component blocks into orthonormal coordinates.

        This is the block2-style tradeoff used by the local Davidson loop:
        spend a bounded amount of work once when the effective Hamiltonian is
        built, then keep every Krylov matvec in the smaller orthonormal
        component basis.

        :param parent_blocks: Parent component block tuple.
        :returns: Tuple ``((in_comp, out_comp, block), ...)`` or ``None`` when
            the transformed storage would exceed the adaptive cap.
        """

        if parent_blocks is None:
            return None
        transforms = self.component_basis.component_transforms
        total_elements = 0
        for in_comp, out_comp, _block in parent_blocks:
            total_elements += (
                int(transforms[int(out_comp)].shape[1])
                * int(transforms[int(in_comp)].shape[1])
            )
            if total_elements > _DIRECT_FACTORIZED_ORTHONORMAL_BLOCK_MAX_ELEMENTS:
                return None
        orthonormal_blocks = []
        for in_comp, out_comp, parent_block in parent_blocks:
            X_in = np.asarray(transforms[int(in_comp)], dtype=complex)
            X_out = np.asarray(transforms[int(out_comp)], dtype=complex)
            transformed = X_out.conj().T @ np.asarray(parent_block, dtype=complex) @ X_in
            if np.linalg.norm(transformed.reshape(-1)) > 1.0e-15:
                orthonormal_blocks.append(
                    (int(in_comp), int(out_comp), np.ascontiguousarray(transformed))
                )
        return tuple(orthonormal_blocks)

    def _build_component_orthonormal_dense_matrix(self, orthonormal_blocks):
        """
        Assemble transformed component blocks into one dense local matrix.

        :param orthonormal_blocks: Transformed component block tuple.
        :returns: Dense matrix or ``None`` when the local dimension is too big.
        """

        if orthonormal_blocks is None:
            return None
        dim = int(self.dim)
        if dim <= 0 or dim * dim > _DIRECT_FACTORIZED_ORTHONORMAL_DENSE_MAX_ELEMENTS:
            return None
        matrix = np.zeros((dim, dim), dtype=complex)
        for in_comp, out_comp, block in orthonormal_blocks:
            in_slice = self.component_basis._orth_slice(int(in_comp))
            out_slice = self.component_basis._orth_slice(int(out_comp))
            matrix[out_slice, in_slice] += np.asarray(block, dtype=complex)
        return np.ascontiguousarray(matrix)

    def _component_orthonormal_block_matvec(self, vector, orthonormal_blocks):
        """
        Apply transformed component blocks directly in orthonormal coordinates.

        :param vector: Orthonormal-coordinate input vector.
        :param orthonormal_blocks: Compiled ``X^H H X`` component blocks.
        :returns: Orthonormal-coordinate output vector.
        """

        out = np.zeros(self.dim, dtype=complex)
        for in_comp, out_comp, block in orthonormal_blocks:
            in_slice = self.component_basis._orth_slice(int(in_comp))
            out_slice = self.component_basis._orth_slice(int(out_comp))
            out[out_slice] += block @ vector[in_slice]
        return out

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

    @property
    def stats(self):
        """
        Return summary statistics for the matrix-free transformed table.

        :returns: Dictionary describing the direct factorized matvec table.
        """

        orthonormal_block_elements = int(
            sum(
                np.asarray(block).size
                for _in_comp, _out_comp, block in (
                    getattr(self, "_component_orthonormal_blocks", None) or ()
                )
            )
        )
        orthonormal_dense_elements = int(
            0
            if getattr(self, "_component_orthonormal_dense_matrix", None) is None
            else np.asarray(self._component_orthonormal_dense_matrix).size
        )
        return {
            "kind": (
                "recursive_parent_block_factorized"
                if self.uses_component_parent_block_kernel
                else (
                    "direct_component_factorized"
                    if self.uses_component_direct_kernel
                    else "direct_factorized"
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
            "stored_kernel_elements": int(
                orthonormal_dense_elements or orthonormal_block_elements
            ),
            "dense_matvec_elements": int(orthonormal_dense_elements),
            "component_direct_kernel": bool(self.uses_component_direct_kernel),
            "component_parent_block_kernel": bool(
                self.uses_component_parent_block_kernel
            ),
            "component_orthonormal_block_kernel": bool(
                self.uses_component_orthonormal_block_kernel
            ),
            "component_orthonormal_dense_kernel": bool(
                self.uses_component_orthonormal_dense_kernel
            ),
            "component_parent_block_elements": int(
                sum(
                    np.asarray(block).size
                    for _in_comp, _out_comp, block in (
                        getattr(self, "_component_parent_blocks", None) or ()
                    )
                )
            ),
            "component_orthonormal_block_elements": orthonormal_block_elements,
            "component_orthonormal_dense_elements": orthonormal_dense_elements,
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
