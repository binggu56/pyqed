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
import hashlib
import time
import weakref

import numpy as np

from .su2_kernel import (
    SU2LocalAction,
    build_component_parent_blocks as _su2_build_component_parent_blocks,
    cpp_available as _su2_cpp_available,
    project_component_orthonormal_blocks as _su2_project_component_orthonormal_blocks,
    resolve_backend as _resolve_su2_kernel_backend,
)

_ORTHONORMAL_BLOCK_DENSE_MATVEC_MAX_ELEMENTS = 1_000_000
_COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS = 65536
_COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_TOTAL_ELEMENTS = 8_000_000
_COMPLEMENTARY_FAMILY_KERNEL_BACKEND = "auto"
_COMPLEMENTARY_FAMILY_FACTOR_BATCH_MIN_ENTRIES = 4
_ORTHONORMAL_BLOCK_BATCH_MIN_ENTRIES = 4
_DIRECT_FACTORIZED_ORTHONORMAL_BLOCK_MAX_ELEMENTS = 16_000_000
_DIRECT_FACTORIZED_ORTHONORMAL_DENSE_MAX_ELEMENTS = 16_000_000
_SU2_QCHEM_DIRECT_PARENT_BLOCKS = False
# H8 remains substantially faster with persistent parent blocks.  H10 exceeds
# this cap (about 427 million complex elements) and stays on the factorized
# family route instead of transiently allocating several gigabytes.
_SU2_QCHEM_DIRECT_PARENT_BLOCK_MAX_ELEMENTS = 400_000_000
_SU2_KERNEL_BACKEND = "auto"
_SU2_KERNEL_DEBUG_CHECK = False
_SU2_KERNEL_DEBUG_CHECK_TOL = 1.0e-10
_UNSET = object()
_SYMBOLIC_MPO_TRANSITION_CACHE = {}
_SYMBOLIC_TRANSITION_SUMMARY_CACHE = {}


def _stable_array_revision(*arrays):
    """Return a deterministic 64-bit revision for array topology or values."""

    digest = hashlib.blake2b(digest_size=8)
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8).tobytes())
    return int.from_bytes(digest.digest(), "little", signed=False)


def get_su2_kernel_policy():
    """Return the SU(2) local-action backend policy."""

    requested = str(_SU2_KERNEL_BACKEND)
    try:
        actual = _resolve_su2_kernel_backend(requested)
    except RuntimeError:
        actual = "unavailable"
    return {
        "backend": requested,
        "actual": actual,
        "cpp_available": bool(_su2_cpp_available()),
        "debug_check": bool(_SU2_KERNEL_DEBUG_CHECK),
        "debug_check_tol": float(_SU2_KERNEL_DEBUG_CHECK_TOL),
    }


def configure_su2_kernel_policy(*, backend=None, debug_check=None, debug_check_tol=None):
    """
    Configure the SU(2) local-action backend.

    :param backend: ``"auto"``, ``"cpp"``, or ``"python"``.
    :returns: Previous policy dictionary, suitable for restoring later.
    """

    global _SU2_KERNEL_BACKEND
    global _SU2_KERNEL_DEBUG_CHECK
    global _SU2_KERNEL_DEBUG_CHECK_TOL

    previous = get_su2_kernel_policy()
    if backend is not None:
        normalized = str(backend).lower().replace("-", "_")
        if normalized == "default":
            normalized = "auto"
        if normalized == "cython":
            normalized = "cpp"
        if normalized not in {"auto", "cpp", "python"}:
            raise ValueError("su2_kernel_backend must be 'auto', 'cpp', or 'python'.")
        if normalized == "cpp":
            _resolve_su2_kernel_backend(normalized)
        _SU2_KERNEL_BACKEND = normalized
    if debug_check is not None:
        _SU2_KERNEL_DEBUG_CHECK = bool(debug_check)
    if debug_check_tol is not None:
        _SU2_KERNEL_DEBUG_CHECK_TOL = float(debug_check_tol)
    return previous


def get_direct_factorized_orthonormal_kernel_policy():
    """
    Return the direct-factorized orthonormal local-kernel policy.

    The dense local-matrix cap remains configurable because the fastest choice
    is workload dependent: component-block matvecs reduce setup and memory,
    while dense local matrices can win when Davidson needs many matvecs.
    """

    return {
        "orthonormal_block_max_elements": int(
            _DIRECT_FACTORIZED_ORTHONORMAL_BLOCK_MAX_ELEMENTS
        ),
        "orthonormal_dense_max_elements": int(
            _DIRECT_FACTORIZED_ORTHONORMAL_DENSE_MAX_ELEMENTS
        ),
        "su2_qchem_direct_parent_blocks": bool(_SU2_QCHEM_DIRECT_PARENT_BLOCKS),
        "su2_qchem_direct_parent_block_max_elements": int(
            _SU2_QCHEM_DIRECT_PARENT_BLOCK_MAX_ELEMENTS
        ),
    }


def configure_direct_factorized_orthonormal_kernel_policy(
    *,
    orthonormal_block_max_elements=None,
    orthonormal_dense_max_elements=None,
    su2_qchem_direct_parent_blocks=None,
    su2_qchem_direct_parent_block_max_elements=None,
):
    """
    Configure direct-factorized orthonormal local-kernel materialization.

    :param orthonormal_block_max_elements: Maximum total transformed
        component-block elements. Set to ``0`` to use parent component blocks.
    :param orthonormal_dense_max_elements: Maximum dense local matrix elements.
        Set to ``0`` to avoid a global dense local matrix.
    :returns: Previous policy dictionary, suitable for restoring later.
    """

    global _DIRECT_FACTORIZED_ORTHONORMAL_BLOCK_MAX_ELEMENTS
    global _DIRECT_FACTORIZED_ORTHONORMAL_DENSE_MAX_ELEMENTS
    global _SU2_QCHEM_DIRECT_PARENT_BLOCKS
    global _SU2_QCHEM_DIRECT_PARENT_BLOCK_MAX_ELEMENTS

    previous = get_direct_factorized_orthonormal_kernel_policy()
    if orthonormal_block_max_elements is not None:
        _DIRECT_FACTORIZED_ORTHONORMAL_BLOCK_MAX_ELEMENTS = max(
            0, int(orthonormal_block_max_elements)
        )
    if orthonormal_dense_max_elements is not None:
        _DIRECT_FACTORIZED_ORTHONORMAL_DENSE_MAX_ELEMENTS = max(
            0, int(orthonormal_dense_max_elements)
        )
    if su2_qchem_direct_parent_blocks is not None:
        _SU2_QCHEM_DIRECT_PARENT_BLOCKS = bool(su2_qchem_direct_parent_blocks)
    if su2_qchem_direct_parent_block_max_elements is not None:
        _SU2_QCHEM_DIRECT_PARENT_BLOCK_MAX_ELEMENTS = max(
            0,
            int(su2_qchem_direct_parent_block_max_elements),
        )
    return previous


def get_complementary_family_kernel_policy():
    """
    Return the current complementary-family matvec kernel policy.

    :returns: Dictionary with ``backend``, ``dense_threshold``, and
        ``dense_max_total_elements``.
    """

    return {
        "backend": str(_COMPLEMENTARY_FAMILY_KERNEL_BACKEND),
        "dense_threshold": int(_COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS),
        "dense_max_total_elements": (
            None
            if _COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_TOTAL_ELEMENTS is None
            else int(_COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_TOTAL_ELEMENTS)
        ),
    }


def configure_complementary_family_kernel_policy(
    *,
    backend=None,
    dense_threshold=None,
    dense_max_total_elements=_UNSET,
):
    """
    Configure the complementary-family matvec kernel policy.

    :param backend: ``"auto"``, ``"dense"``, or ``"factor"``.
    :param dense_threshold: Maximum dense elements for one local block.
    :param dense_max_total_elements: Maximum dense elements materialized by one
        family table. Use ``None`` for no table-level cap.
    :returns: Previous policy dictionary, suitable for restoring later.
    """

    global _COMPLEMENTARY_FAMILY_KERNEL_BACKEND
    global _COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS
    global _COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_TOTAL_ELEMENTS

    previous = get_complementary_family_kernel_policy()
    if backend is not None:
        normalized = str(backend).lower().replace("-", "_")
        if normalized in {"factorized", "factor_native"}:
            normalized = "factor"
        if normalized in {"hybrid", "default"}:
            normalized = "auto"
        if normalized not in {"auto", "dense", "factor"}:
            raise ValueError("family kernel backend must be 'auto', 'dense', or 'factor'.")
        _COMPLEMENTARY_FAMILY_KERNEL_BACKEND = normalized
    if dense_threshold is not None:
        _COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_ELEMENTS = max(0, int(dense_threshold))
    if dense_max_total_elements is not _UNSET:
        _COMPLEMENTARY_FAMILY_NATIVE_KERNEL_MAX_TOTAL_ELEMENTS = (
            None
            if dense_max_total_elements is None
            else max(0, int(dense_max_total_elements))
        )
    return previous


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

    @staticmethod
    def _pack_rank_coupled_numeric_payload(block_map, *, side, bond, active_channels=None):
        """
        Pack a rank-coupled boundary block map for symbolic payload ownership.

        :returns: Packed boundary table or ``None`` when the block map is not a
            supported rank-coupled payload.
        """

        try:
            ensure_packed = getattr(block_map, "ensure_packed", None)
            if ensure_packed is not None:
                packed = ensure_packed(side=side, bond=bond)
                if packed is not None:
                    return packed
            from .su2_qchem_plan import pack_rank_coupled_boundary_table_from_block_map

            return pack_rank_coupled_boundary_table_from_block_map(
                block_map,
                active_channels=active_channels,
                side=side,
                bond=bond,
                representation="rank_coupled_by_ket",
            )
        except Exception:
            return None

    @staticmethod
    def _filter_rank_coupled_numeric_payload(packed_table, active_channels):
        """Filter a packed rank-coupled payload to active symbolic channels."""

        try:
            from .su2_qchem_plan import filter_rank_coupled_boundary_table_channels

            return filter_rank_coupled_boundary_table_channels(
                packed_table,
                active_channels,
            )
        except Exception:
            return None

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
        packed_payload = None
        if block is None:
            active = None
        else:
            packed_payload = self._pack_rank_coupled_numeric_payload(
                block,
                side="left",
                bond=bond,
            )
            active = (
                _active_boundary_channels(block)
                if packed_payload is None
                else set(int(channel) for channel in packed_payload.channel_ids)
            )
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
        if block is None:
            return table
        if packed_payload is not None:
            packed_payload = self._filter_rank_coupled_numeric_payload(
                packed_payload,
                table.channels,
            )
        return table.with_numeric_payload(block, packed_boundary_table=packed_payload)

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
        packed_payload = None
        if block is None:
            active = None
        else:
            packed_payload = self._pack_rank_coupled_numeric_payload(
                block,
                side="right",
                bond=bond,
            )
            active = (
                _active_boundary_channels(block)
                if packed_payload is None
                else set(int(channel) for channel in packed_payload.channel_ids)
            )
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
        if block is None:
            return table
        if packed_payload is not None:
            packed_payload = self._filter_rank_coupled_numeric_payload(
                packed_payload,
                table.channels,
            )
        return table.with_numeric_payload(block, packed_boundary_table=packed_payload)

    def with_numeric_payload(self, block_map, *, packed_boundary_table=None):
        """
        Return a copy of this table owning numeric renormalized payloads.

        :param block_map: Sector-pair keyed numeric boundary block map.
        :returns: Symbolic table carrying numeric boundary payloads.
        """

        payloads = None
        payload_kind = "none"
        if packed_boundary_table is None:
            packed_boundary_table = self._pack_rank_coupled_numeric_payload(
                block_map,
                active_channels=self.channels,
                side=self.side,
                bond=self.bond,
            )
        try:
            from .su2_qchem_plan import PackedRankCoupledBoundaryPayloads

            if packed_boundary_table is not None:
                payloads = PackedRankCoupledBoundaryPayloads(packed_boundary_table)
                payload_kind = "rank_coupled_packed"
        except Exception:
            payloads = None
        if payloads is None:
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
    numeric_action_tables: dict = field(default_factory=dict, compare=False, repr=False)
    native_operator_tables: dict = field(default_factory=dict, compare=False, repr=False)

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
        packed_payload = getattr(symbolic_table.numeric_payloads, "packed_table", None)
        if packed_payload is not None:
            sectors = tuple(packed_payload.sector_codec.sectors)
            for row_idx, ket_id in enumerate(packed_payload.ket_sector_ids):
                q_in = sectors[int(ket_id)]
                entry_start = int(packed_payload.entry_offsets[row_idx])
                entry_stop = int(packed_payload.entry_offsets[row_idx + 1])
                for entry_idx in range(entry_start, entry_stop):
                    q_out = sectors[int(packed_payload.out_sector_ids[entry_idx])]
                    channel_start = int(packed_payload.channel_offsets[entry_idx])
                    channel_stop = int(packed_payload.channel_offsets[entry_idx + 1])
                    for channel_idx in range(channel_start, channel_stop):
                        channel = int(packed_payload.channel_ids[channel_idx])
                        key = (q_out, q_in, channel)
                        arr = np.asarray(packed_payload.block_pool.array(channel_idx))
                        payloads_by_channel.setdefault(channel, []).append(key)
                        payload_norms_by_channel[channel] = (
                            payload_norms_by_channel.get(channel, 0.0)
                            + float(np.linalg.norm(arr)) ** 2
                        )
                        payload_elements_by_channel[channel] = (
                            payload_elements_by_channel.get(channel, 0)
                            + int(arr.size)
                        )
        else:
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

    def get_numeric_action_table(self, key):
        """
        Return a cached numeric boundary action table for ``key``.

        The table object is intentionally opaque to this non-Abelian storage
        layer; Abelian sweep code owns the concrete action-table type.  Keeping
        the cache here makes the complementary-family table the persistent owner
        of boundary actions rather than a passive metadata record.
        """

        return self.numeric_action_tables.get(key)

    def put_numeric_action_table(self, key, table):
        """
        Store an opaque numeric boundary action table and return it.
        """

        self.numeric_action_tables[key] = table
        return table

    def get_native_operator_table(self, key):
        """Return a cached native renormalized operator table."""

        return self.native_operator_tables.get(key)

    def put_native_operator_table(self, key, table):
        """Store a native renormalized operator table and return it."""

        self.native_operator_tables[key] = table
        return table

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
            "numeric_action_tables": int(len(self.numeric_action_tables)),
            "numeric_action_table_stored_elements": int(
                sum(
                    int(getattr(table, "matrix", np.asarray(())).size)
                    for table in self.numeric_action_tables.values()
                )
            ),
            "native_operator_tables": int(len(self.native_operator_tables)),
            "native_operator_table_stored_elements": int(
                sum(
                    int(getattr(table, "stored_elements", 0))
                    for table in self.native_operator_tables.values()
                )
            ),
            "native_operator_table_stats": {
                repr(key): getattr(table, "stats", {})
                for key, table in self.native_operator_tables.items()
            },
            "families": {
                str(name): block.stats
                for name, block in self.family_blocks.items()
            },
        }


@dataclass(frozen=True)
class ComplementaryNativeGeneratorOperatorTable:
    """
    Abelian-native renormalized spin-free generator operators for one boundary.

    The table stores already-renormalized boundary tensors for spin-summed
    one-body generators ``E_pq``.  It is intentionally independent from the
    local Davidson action table so the sweep layer can reuse these blocks when
    assembling R/P/Q family contractions.
    """

    side: str
    bond: int
    operators: dict
    source: str = "abelian_native_spinfree_generator_boundary_table"
    build_seconds: float = 0.0

    @property
    def n_operators(self):
        """Return the number of stored generator operators."""

        return int(len(self.operators))

    @property
    def stored_blocks(self):
        """Return the number of block-sparse tensor blocks stored."""

        return int(
            sum(
                len(getattr(operator, "data", {}) or {})
                for operator in self.operators.values()
            )
        )

    @property
    def stored_elements(self):
        """Return the number of scalar tensor elements stored."""

        return int(
            sum(
                int(np.asarray(block).size)
                for operator in self.operators.values()
                for block in (getattr(operator, "data", {}) or {}).values()
            )
        )

    @property
    def stats(self):
        """Return compact diagnostics for this native operator table."""

        return {
            "kind": "complementary_native_generator_operator_table",
            "source": str(self.source),
            "side": str(self.side),
            "bond": int(self.bond),
            "n_operators": int(self.n_operators),
            "stored_blocks": int(self.stored_blocks),
            "stored_elements": int(self.stored_elements),
            "build_seconds": float(self.build_seconds),
            "operator_keys": tuple(
                tuple(int(index) for index in key)
                for key in sorted(self.operators)
            ),
        }


@dataclass
class ComplementaryNativePairBoundaryOperatorTable:
    """
    Abelian-native renormalized pair-complement boundary table.

    The table records block2-like ``P`` candidates that are consumed through
    renormalized generator boundary operators instead of expanded exact JW
    patterns.  Entries are still validated by the sweep code before they are
    allowed to replace exact-pattern components.
    """

    side: str
    bond: int
    entries: dict = field(default_factory=dict)
    operators: dict = field(default_factory=dict)
    source: str = "abelian_native_pair_complement_boundary_table"
    build_seconds: float = 0.0
    validated_terms: int = 0
    rejected_terms: int = 0

    def get_operator(self, key):
        """Return a stored renormalized pair boundary operator."""

        return self.operators.get(tuple(key))

    def get(self, key):
        """Return stored validated component entries for one ``P`` key."""

        return self.entries.get(tuple(key))

    def add_operator(self, key, operator):
        """Store one renormalized pair boundary operator."""

        self.operators[tuple(key)] = operator
        return operator

    def add(self, key, entries):
        """Store validated component entries for one ``P`` integral key."""

        entries = tuple(entries or ())
        self.entries[tuple(key)] = entries
        self.validated_terms += 1
        return entries

    def reject(self):
        """Record one rejected native ``P`` candidate."""

        self.rejected_terms += 1

    @property
    def n_terms(self):
        """Return the number of accepted ``P`` generator keys."""

        return int(len(self.entries))

    @property
    def n_operators(self):
        """Return the number of stored renormalized pair boundary operators."""

        return int(len(self.operators))

    @property
    def n_entries(self):
        """Return the number of executable component entries."""

        return int(sum(len(entries) for entries in self.entries.values()))

    @property
    def stored_blocks(self):
        """Return the number of stored block-sparse tensor blocks."""

        return int(
            sum(
                ComplementaryNativeExactPatternComponentTable._stored_blocks_for(entries)
                for entries in self.entries.values()
            )
            + sum(
                ComplementaryNativeExactPatternComponentTable._stored_blocks_for(operator)
                for operator in self.operators.values()
            )
        )

    @property
    def stored_elements(self):
        """Return the number of stored scalar tensor elements."""

        return int(
            sum(
                ComplementaryNativeExactPatternComponentTable._stored_elements_for(entries)
                for entries in self.entries.values()
            )
            + sum(
                ComplementaryNativeExactPatternComponentTable._stored_elements_for(operator)
                for operator in self.operators.values()
            )
        )

    @property
    def stats(self):
        """Return compact diagnostics for this native pair table."""

        return {
            "kind": "complementary_native_pair_boundary_operator_table",
            "source": str(self.source),
            "side": str(self.side),
            "bond": int(self.bond),
            "n_terms": int(self.n_terms),
            "n_operators": int(self.n_operators),
            "n_entries": int(self.n_entries),
            "stored_blocks": int(self.stored_blocks),
            "stored_elements": int(self.stored_elements),
            "build_seconds": float(self.build_seconds),
            "validated_terms": int(self.validated_terms),
            "rejected_terms": int(self.rejected_terms),
            "operator_keys": tuple(
                tuple(int(index) for index in key)
                for key in sorted(set(self.entries) | set(self.operators))
            ),
        }


@dataclass
class ComplementaryNativeExactPatternOperatorTable:
    """
    Abelian-native exact JW-pattern boundary table for direct family actions.

    The table stores contextual renormalized boundary pieces produced from the
    exact Jordan-Wigner-expanded patterns.  Unlike the spin-free generator
    table, these entries do not assume that generator products can be split
    across a boundary; they preserve the already-validated direct path algebra.
    """

    side: str
    bond: int
    entries: dict = field(default_factory=dict)
    family_counts: dict = field(default_factory=dict)
    source: str = "abelian_native_exact_jw_pattern_boundary_table"
    build_seconds: float = 0.0

    def get(self, key):
        """Return a stored exact-pattern boundary entry."""

        return self.entries.get(key)

    def put(self, key, value, family_name=None):
        """Store an exact-pattern boundary entry and record its owner family."""

        is_new = key not in self.entries
        self.entries[key] = value
        if is_new and family_name is not None:
            name = str(family_name)
            self.family_counts[name] = int(self.family_counts.get(name, 0)) + 1
        return value

    @staticmethod
    def _stored_blocks_for(value):
        if value is None:
            return 0
        data = getattr(value, "data", None)
        if isinstance(data, dict):
            return int(len(data))
        entries = getattr(value, "entries", None)
        if entries is not None:
            return ComplementaryNativeExactPatternComponentTable._stored_blocks_for(
                tuple(entries)
            )
        if isinstance(value, (tuple, list)):
            return int(
                sum(
                    ComplementaryNativeExactPatternOperatorTable._stored_blocks_for(item)
                    for item in value
                )
            )
        return 0

    @staticmethod
    def _stored_elements_for(value):
        if value is None:
            return 0
        data = getattr(value, "data", None)
        if isinstance(data, dict):
            return int(sum(int(np.asarray(block).size) for block in data.values()))
        entries = getattr(value, "entries", None)
        if entries is not None:
            return ComplementaryNativeExactPatternComponentTable._stored_elements_for(
                tuple(entries)
            )
        if isinstance(value, (tuple, list)):
            return int(
                sum(
                    ComplementaryNativeExactPatternOperatorTable._stored_elements_for(item)
                    for item in value
                )
            )
        return 0

    @property
    def n_entries(self):
        """Return the number of stored exact-pattern entries."""

        return int(len(self.entries))

    @property
    def stored_blocks(self):
        """Return the number of stored block-sparse tensor blocks."""

        return int(
            sum(self._stored_blocks_for(value) for value in self.entries.values())
        )

    @property
    def stored_elements(self):
        """Return the number of stored scalar tensor elements."""

        return int(
            sum(self._stored_elements_for(value) for value in self.entries.values())
        )

    @property
    def stats(self):
        """Return compact diagnostics for this exact-pattern table."""

        return {
            "kind": "complementary_native_exact_pattern_operator_table",
            "source": str(self.source),
            "side": str(self.side),
            "bond": int(self.bond),
            "n_entries": int(self.n_entries),
            "stored_blocks": int(self.stored_blocks),
            "stored_elements": int(self.stored_elements),
            "build_seconds": float(self.build_seconds),
            "family_counts": {
                str(name): int(count)
                for name, count in sorted(self.family_counts.items())
            },
        }


@dataclass(frozen=True)
class ComplementaryNativeExactPatternFamilyEntries:
    """
    Iterable family component entries plus exact grouping metadata.
    """

    family_name: str
    entries: tuple
    entry_groups: tuple = ()
    group_keys: tuple = ()
    source: str = "native_exact_pattern_family_entries"

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return int(len(self.entries))

    def __bool__(self):
        return bool(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    @property
    def n_groups(self):
        """Return the number of stored entry groups."""

        return int(len(self.entry_groups))

    @property
    def n_group_entries(self):
        """Return the total number of entries after exact group aggregation."""

        return int(sum(len(group) for group in self.entry_groups))

    @property
    def stats(self):
        """Return compact diagnostics for these grouped entries."""

        return {
            "kind": "complementary_native_exact_pattern_family_entries",
            "source": str(self.source),
            "family_name": str(self.family_name),
            "n_entries": int(len(self.entries)),
            "n_groups": int(self.n_groups),
            "n_group_entries": int(self.n_group_entries),
            "entry_reduction": int(self.n_group_entries - len(self.entries)),
            "group_sizes": tuple(int(len(group)) for group in self.entry_groups),
            "group_keys": tuple(repr(key) for key in self.group_keys),
        }


@dataclass
class ComplementaryNativeExactPatternComponentTable:
    """
    Center-bond exact JW-pattern component table for direct family actions.

    Boundary tables own the contextual left/right operators.  This table owns
    the assembled two-site family components that consume those boundary
    pieces.  It keeps the direct path algebra exact while giving the sweep a
    block2-like table object for P/R component ownership.
    """

    bond: int
    families: dict = field(default_factory=dict)
    family_records: dict = field(default_factory=dict)
    source: str = "abelian_native_exact_jw_pattern_component_table"
    build_seconds: float = 0.0

    def get_family_records(self, family_name):
        """Return exact component records for one family."""

        return self.family_records.get(str(family_name))

    def put_family_records(self, family_name, records):
        """Store exact component records for one family and return them."""

        records = tuple(records or ())
        self.family_records[str(family_name)] = records
        return records

    def get_family(self, family_name):
        """Return stored component entries for one family."""

        return self.families.get(str(family_name))

    def put_family(
        self,
        family_name,
        entries,
        records=None,
        *,
        compression_policy="auto",
        min_reduction=1,
        max_group_size=None,
    ):
        """Store component entries for one family and return them."""

        entries_is_packed = bool(
            getattr(entries, "_pyqed_packed_direct_family_entries", False)
        )
        entries = entries if entries_is_packed else tuple(entries or ())
        if records is None:
            records = self.family_records.get(str(family_name), ())
        records = tuple(records or ())
        policy = str(compression_policy or "auto").lower().replace("-", "_")
        if policy in {"off", "false", "disabled", "uncompressed"}:
            policy = "none"
        if policy not in {"none", "auto", "structural"}:
            policy = "auto"
        if (
            policy == "auto"
            and max_group_size is not None
            and int(max_group_size) <= 1
        ):
            family_entries = ComplementaryNativeExactPatternFamilyEntries(
                family_name=str(family_name),
                entries=entries,
                entry_groups=(),
                group_keys=(),
            )
            self.families[str(family_name)] = family_entries
            return family_entries
        grouped = {}
        for index, entry in enumerate(entries):
            key = (
                self._record_boundary_pair(records[index])
                if index < len(records)
                else ((), ())
            )
            grouped.setdefault(key, []).append(entry)

        def _direct_sum_w_pair(group):
            if len(group) <= 1:
                return tuple(group)
            parsed = []
            mid_dims = {}
            for entry_index, entry in enumerate(group):
                try:
                    E_term, W_pair, F_term = entry
                    W_left, W_right = W_pair
                except Exception:
                    return tuple(group)
                left_data = getattr(W_left, "data", {}) or {}
                right_data = getattr(W_right, "data", {}) or {}
                left_mid_dims = {}
                right_mid_dims = {}
                for key, block in left_data.items():
                    q_mid = key[1]
                    dim = int(np.asarray(block).shape[1])
                    old = left_mid_dims.get(q_mid)
                    if old is not None and int(old) != int(dim):
                        return tuple(group)
                    left_mid_dims[q_mid] = int(dim)
                for key, block in right_data.items():
                    q_mid = key[0]
                    dim = int(np.asarray(block).shape[0])
                    old = right_mid_dims.get(q_mid)
                    if old is not None and int(old) != int(dim):
                        return tuple(group)
                    right_mid_dims[q_mid] = int(dim)
                left_mids = set(left_mid_dims)
                right_mids = set(right_mid_dims)
                mids = left_mids.intersection(right_mids)
                if not mids:
                    return tuple(group)
                for q_mid in mids:
                    left_dim = int(left_mid_dims[q_mid])
                    right_dim = int(right_mid_dims[q_mid])
                    if left_dim != right_dim:
                        return tuple(group)
                    mid_dims[(entry_index, q_mid)] = int(left_dim)
                parsed.append((E_term, W_left, W_right, F_term, tuple(sorted(mids))))

            E_term = parsed[0][0]
            F_term = parsed[0][3]
            mid_sectors = tuple(sorted({q for item in parsed for q in item[4]}))
            offsets = {}
            totals = {q_mid: 0 for q_mid in mid_sectors}
            for entry_index, _item in enumerate(parsed):
                for q_mid in mid_sectors:
                    dim = int(mid_dims.get((entry_index, q_mid), 0))
                    offsets[(entry_index, q_mid)] = totals[q_mid]
                    totals[q_mid] += dim
            try:
                W_left_qns = [
                    sorted({q for _E, W_left, _W_right, _F, _mids in parsed for q in W_left.qns[0]}),
                    list(mid_sectors),
                    sorted({q for _E, W_left, _W_right, _F, _mids in parsed for q in W_left.qns[2]}),
                    sorted({q for _E, W_left, _W_right, _F, _mids in parsed for q in W_left.qns[3]}),
                ]
                W_right_qns = [
                    list(mid_sectors),
                    sorted({q for _E, _W_left, W_right, _F, _mids in parsed for q in W_right.qns[1]}),
                    sorted({q for _E, _W_left, W_right, _F, _mids in parsed for q in W_right.qns[2]}),
                    sorted({q for _E, _W_left, W_right, _F, _mids in parsed for q in W_right.qns[3]}),
                ]
            except Exception:
                return tuple(group)

            left_data = {}
            right_data = {}

            def _accumulate_block(data, key, block, axis, start, total_dim):
                shape = list(block.shape)
                shape[axis] = int(total_dim)
                if key not in data:
                    data[key] = np.zeros(shape, dtype=np.asarray(block).dtype)
                elif tuple(data[key].shape) != tuple(shape):
                    return False
                slices = [slice(None)] * len(shape)
                slices[axis] = slice(int(start), int(start) + int(block.shape[axis]))
                data[key][tuple(slices)] = block
                return True

            for entry_index, (_E, W_left, W_right, _F, _mids) in enumerate(parsed):
                for key, block in W_left.data.items():
                    q_mid = key[1]
                    if q_mid not in totals:
                        continue
                    if not _accumulate_block(
                        left_data,
                        key,
                        np.asarray(block),
                        1,
                        offsets[(entry_index, q_mid)],
                        totals[q_mid],
                    ):
                        return tuple(group)
                for key, block in W_right.data.items():
                    q_mid = key[0]
                    if q_mid not in totals:
                        continue
                    if not _accumulate_block(
                        right_data,
                        key,
                        np.asarray(block),
                        0,
                        offsets[(entry_index, q_mid)],
                        totals[q_mid],
                    ):
                        return tuple(group)
            if not left_data or not right_data:
                return tuple(group)
            try:
                W_class = parsed[0][1].__class__
                W_left = W_class(left_data, W_left_qns, parsed[0][1].dirs[:])
                W_right = W_class(right_data, W_right_qns, parsed[0][2].dirs[:])
            except Exception:
                return tuple(group)
            return ((E_term, [W_left, W_right], F_term),)

        def _compressed_group(group):
            group = tuple(group)
            if len(group) <= 1:
                return group
            if (
                policy == "auto"
                and max_group_size is not None
                and int(len(group)) > int(max_group_size)
            ):
                return group
            return _direct_sum_w_pair(group)

        if policy == "none":
            group_items = tuple(
                (key, tuple(grouped[key]))
                for key in sorted(grouped, key=repr)
            )
        else:
            group_items = tuple(
                (
                    key,
                    _compressed_group(grouped[key]),
                )
                for key in sorted(grouped, key=repr)
            )
        compressed_entries = tuple(
            entry
            for _key, group in group_items
            for entry in group
        )
        reduction = int(len(entries) - len(compressed_entries))
        largest_group = max((len(group) for _key, group in group_items), default=0)
        if (
            policy == "auto"
            and (
                reduction < int(min_reduction)
                or (
                    max_group_size is not None
                    and int(largest_group) > int(max_group_size)
                )
            )
        ):
            group_items = tuple(
                (key, tuple(grouped[key]))
                for key in sorted(grouped, key=repr)
            )
            compressed_entries = entries
        family_entries = ComplementaryNativeExactPatternFamilyEntries(
            family_name=str(family_name),
            entries=compressed_entries,
            entry_groups=tuple(group for _key, group in group_items),
            group_keys=tuple(key for key, _group in group_items),
        )
        self.families[str(family_name)] = family_entries
        return family_entries

    @staticmethod
    def _stored_blocks_for(value):
        if value is None:
            return 0
        data = getattr(value, "data", None)
        if isinstance(data, dict):
            return int(len(data))
        entries = getattr(value, "entries", None)
        if entries is not None:
            return ComplementaryNativeExactPatternComponentTable._stored_blocks_for(
                tuple(entries)
            )
        if isinstance(value, (tuple, list)):
            return int(
                sum(
                    ComplementaryNativeExactPatternComponentTable._stored_blocks_for(item)
                    for item in value
                )
            )
        return 0

    @staticmethod
    def _stored_elements_for(value):
        if value is None:
            return 0
        data = getattr(value, "data", None)
        if isinstance(data, dict):
            return int(sum(int(np.asarray(block).size) for block in data.values()))
        entries = getattr(value, "entries", None)
        if entries is not None:
            return ComplementaryNativeExactPatternComponentTable._stored_elements_for(
                tuple(entries)
            )
        if isinstance(value, (tuple, list)):
            return int(
                sum(
                    ComplementaryNativeExactPatternComponentTable._stored_elements_for(item)
                    for item in value
                )
            )
        return 0

    @property
    def n_families(self):
        """Return the number of stored families."""

        return int(len(self.families))

    @property
    def n_entries(self):
        """Return the total number of stored component entries."""

        return int(sum(len(entries) for entries in self.families.values()))

    @property
    def n_records(self):
        """Return the total number of exact symbolic component records."""

        return int(sum(len(records) for records in self.family_records.values()))

    @staticmethod
    def _record_local_pair(record):
        try:
            return (str(record[1]), str(record[2]))
        except Exception:
            return ("?", "?")

    @staticmethod
    def _record_boundary_pair(record):
        try:
            return (tuple(record[0]), tuple(record[3]))
        except Exception:
            return ((), ())

    def _record_group_counts(self, indexer):
        out = {}
        for name, records in self.family_records.items():
            groups = {}
            for record in records:
                key = indexer(record)
                groups[key] = int(groups.get(key, 0)) + 1
            out[str(name)] = {
                repr(key): int(count)
                for key, count in sorted(groups.items(), key=lambda item: repr(item[0]))
            }
        return out

    @property
    def stored_blocks(self):
        """Return the number of stored block-sparse tensor blocks."""

        return int(
            sum(self._stored_blocks_for(entries) for entries in self.families.values())
        )

    @property
    def stored_elements(self):
        """Return the number of stored scalar tensor elements."""

        return int(
            sum(self._stored_elements_for(entries) for entries in self.families.values())
        )

    @property
    def stats(self):
        """Return compact diagnostics for this component table."""

        return {
            "kind": "complementary_native_exact_pattern_component_table",
            "source": str(self.source),
            "bond": int(self.bond),
            "n_families": int(self.n_families),
            "n_records": int(self.n_records),
            "n_entries": int(self.n_entries),
            "stored_blocks": int(self.stored_blocks),
            "stored_elements": int(self.stored_elements),
            "build_seconds": float(self.build_seconds),
            "record_counts": {
                str(name): int(len(records))
                for name, records in sorted(self.family_records.items())
            },
            "family_counts": {
                str(name): int(len(entries))
                for name, entries in sorted(self.families.items())
            },
            "family_group_counts": {
                str(name): int(getattr(entries, "n_groups", 0))
                for name, entries in sorted(self.families.items())
            },
            "family_group_entry_counts": {
                str(name): int(getattr(entries, "n_group_entries", len(entries)))
                for name, entries in sorted(self.families.items())
            },
            "family_entry_reductions": {
                str(name): int(
                    len(self.family_records.get(str(name), ())) - len(entries)
                )
                for name, entries in sorted(self.families.items())
            },
            "family_group_sizes": {
                str(name): tuple(
                    int(len(group))
                    for group in getattr(entries, "entry_groups", ())
                )
                for name, entries in sorted(self.families.items())
            },
            "local_pair_group_counts": self._record_group_counts(
                self._record_local_pair
            ),
            "boundary_pair_group_counts": self._record_group_counts(
                self._record_boundary_pair
            ),
        }


@dataclass(frozen=True)
class FamilyCppFactorKernel:
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
    matmul_two_step: bool = True
    left_matrix: np.ndarray | None = None
    right_matrix: np.ndarray | None = None
    tmp_shape: tuple = ()
    output_shape: tuple = ()

    @classmethod
    def from_compiled_term(cls, term):
        """Build a native factor kernel from a compiled factorized term."""

        left_stack = np.asarray(term.left_stack)
        right_stack = np.asarray(term.right_stack)
        tdim, ldim, kdim, wdim, adim, bdim = (
            int(dim) for dim in left_stack.shape
        )
        _tdim2, _wdim2, qdim, rdim, ddim, cdim = (
            int(dim) for dim in right_stack.shape
        )
        left_matrix = np.ascontiguousarray(
            left_stack.transpose(0, 1, 3, 4, 2, 5).reshape(
                tdim * ldim * wdim * adim,
                kdim * bdim,
            )
        )
        right_matrix = np.ascontiguousarray(
            right_stack.transpose(0, 1, 3, 5, 4, 2).reshape(
                tdim * wdim * rdim * cdim,
                ddim * qdim,
            )
        )
        return cls(
            left_stack=left_stack,
            right_stack=right_stack,
            input_shape=tuple(int(dim) for dim in term.input_entry.shape),
            output_size=int(term.output_size),
            use_direct_contraction=bool(
                getattr(term, "_use_direct_contraction", False)
            ),
            matmul_two_step=True,
            left_matrix=left_matrix,
            right_matrix=right_matrix,
            tmp_shape=(tdim, ldim, wdim, adim, cdim, rdim),
            output_shape=(ldim, adim, ddim, qdim),
        )

    @property
    def stored_elements(self):
        """Return the number of scalar elements stored by this kernel."""

        return int(np.asarray(self.left_stack).size + np.asarray(self.right_stack).size)

    @property
    def batch_signature(self):
        """Return a shape signature for batched factor-native contractions."""

        if self.left_matrix is None or self.right_matrix is None:
            return None
        return (
            tuple(int(dim) for dim in np.asarray(self.left_matrix).shape),
            tuple(int(dim) for dim in np.asarray(self.right_matrix).shape),
            tuple(int(dim) for dim in self.tmp_shape),
            tuple(int(dim) for dim in self.output_shape),
            tuple(int(dim) for dim in self.input_shape),
            int(self.output_size),
        )

    def apply_block(self, block_in):
        """
        Apply the factor-native contraction to one input block.

        :param block_in: Input sector block.
        :returns: Flattened output-sector contribution.
        """

        left_stack = np.asarray(self.left_stack)
        right_stack = np.asarray(self.right_stack)
        block_in = np.asarray(block_in)
        if bool(self.use_direct_contraction) and not bool(self.matmul_two_step):
            contrib = np.einsum(
                "tlkwab,kbcr,twqrdc->ladq",
                left_stack,
                block_in,
                right_stack,
                optimize=False,
            )
            return np.asarray(contrib).reshape(int(self.output_size))
        if bool(self.matmul_two_step):
            kin, bin_, cin, rin = (int(dim) for dim in block_in.shape)
            if self.left_matrix is None or self.right_matrix is None:
                raise RuntimeError("FamilyCppFactorKernel is missing BLAS matrices.")
            input_matrix = block_in.reshape(kin * bin_, cin * rin)
            tmp = (self.left_matrix @ input_matrix).reshape(tuple(self.tmp_shape))
            ldim, adim, ddim, qdim = (int(dim) for dim in self.output_shape)
            tmp_matrix = np.ascontiguousarray(
                tmp.transpose(1, 3, 0, 2, 5, 4).reshape(
                    ldim * adim,
                    -1,
                )
            )
            contrib = tmp_matrix @ self.right_matrix
            return np.asarray(contrib).reshape(ldim, adim, ddim, qdim).reshape(
                int(self.output_size)
            )
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

    def apply_blocks(self, block_inputs):
        """Apply this factor kernel to a trailing block-vector dimension."""

        block_inputs = np.asarray(block_inputs)
        if block_inputs.ndim != 5:
            raise ValueError("Factor-kernel block input must have rank five.")
        nvec = int(block_inputs.shape[-1])
        if nvec == 1:
            return self.apply_block(block_inputs[..., 0]).reshape(-1, 1)
        if not bool(self.matmul_two_step):
            return np.column_stack(
                [self.apply_block(block_inputs[..., idx]) for idx in range(nvec)]
            )
        kin, bin_, cin, rin, _ = (int(dim) for dim in block_inputs.shape)
        if self.left_matrix is None or self.right_matrix is None:
            raise RuntimeError("FamilyCppFactorKernel is missing BLAS matrices.")
        input_matrix = block_inputs.reshape(kin * bin_, cin * rin * nvec)
        tmp = (self.left_matrix @ input_matrix).reshape(
            tuple(self.tmp_shape) + (nvec,)
        )
        ldim, adim, ddim, qdim = (int(dim) for dim in self.output_shape)
        tmp_matrices = np.ascontiguousarray(
            tmp.transpose(6, 1, 3, 0, 2, 5, 4).reshape(
                nvec,
                ldim * adim,
                -1,
            )
        )
        contrib = np.matmul(tmp_matrices, self.right_matrix)
        return np.ascontiguousarray(
            contrib.transpose(1, 2, 0).reshape(int(self.output_size), nvec)
        )


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
    factor_kernel: FamilyCppFactorKernel | None = None
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
            factor_kernel=FamilyCppFactorKernel.from_compiled_term(
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
    def dense_kernel_elements(self):
        """Return the dense kernel elements required for this apply entry."""

        input_size = int(getattr(self.compiled_term.input_entry, "size", 0))
        output_size = int(getattr(self.compiled_term, "output_size", 0))
        return int(input_size * output_size)

    @property
    def input_entry(self):
        """Return the compiled term input entry."""

        return self.compiled_term.input_entry

    @property
    def factor_batch_signature(self):
        """Return a batching signature for this entry, if batchable."""

        if self.native_kernel is not None or self.factor_kernel is None:
            return None
        if not self.factor_kernel.matmul_two_step:
            return None
        return self.factor_kernel.batch_signature

    def apply_block(self, block_in):
        """Apply the current numerical backend to one input block."""

        if self.native_kernel is not None:
            return np.asarray(self.native_kernel @ np.asarray(block_in).reshape(-1))
        if self.factor_kernel is not None:
            return self.factor_kernel.apply_block(block_in)
        return self.compiled_term.apply_block(block_in)

    def apply_blocks(self, block_inputs):
        """Apply the current backend to a trailing block-vector dimension."""

        block_inputs = np.asarray(block_inputs)
        nvec = int(block_inputs.shape[-1])
        if self.native_kernel is not None:
            return self.native_kernel @ block_inputs.reshape(-1, nvec)
        if self.factor_kernel is not None:
            return self.factor_kernel.apply_blocks(block_inputs)
        return np.column_stack(
            [self.compiled_term.apply_block(block_inputs[..., idx]) for idx in range(nvec)]
        )

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
        if hasattr(value, "items"):
            for index, item in value.items():
                arr = np.asarray(item)
                if arr.size and np.any(arr != 0):
                    channels.add(int(index))
        elif isinstance(value, (tuple, list)):
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
    block_items = (
        blocks.items()
        if hasattr(blocks, "items")
        else enumerate(tuple(blocks or ()))
    )
    for idx, block in block_items:
        if idx not in active_channels:
            continue
        arr = np.asarray(block)
        if arr.size and (
            np.any(arr) if float(tol) <= 0.0 else np.any(np.abs(arr) > float(tol))
        ):
            out.append((idx, arr))
    return tuple(out)


def _symbolic_numeric_payloads_from_block_map(block_map, active_channels):
    payloads = {}
    active_channels = set(int(channel) for channel in active_channels)
    rank_coupled = bool(getattr(block_map, "rank_coupled", False))
    for (q_out, q_in), value in block_map.items():
        if hasattr(value, "items") or isinstance(value, (tuple, list)):
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
    if hasattr(block, "iter_routes"):
        return tuple(
            (int(left), int(right))
            for left, right, _payload in block.iter_routes()
        )
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

    cache_key = id(core)
    cached = _SYMBOLIC_MPO_TRANSITION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    records = tuple(getattr(core, "symbolic_transitions", ()) or ())
    if records:
        result = (
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
        if len(_SYMBOLIC_MPO_TRANSITION_CACHE) > 512:
            _SYMBOLIC_MPO_TRANSITION_CACHE.clear()
        _SYMBOLIC_MPO_TRANSITION_CACHE[cache_key] = result
        return result
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
    result = (tuple(transitions.values()), False)
    if len(_SYMBOLIC_MPO_TRANSITION_CACHE) > 512:
        _SYMBOLIC_MPO_TRANSITION_CACHE.clear()
    _SYMBOLIC_MPO_TRANSITION_CACHE[cache_key] = result
    return result


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


def _symbolic_transition_summary(transitions, direction):
    """
    Return cached ``(parent, child, families)`` transition records.
    """

    key = (id(transitions), str(direction))
    cached = _SYMBOLIC_TRANSITION_SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached
    summary = []
    for transition in transitions:
        if direction == "left":
            parent_channel = int(transition.left_channel)
            child_channel = int(transition.right_channel)
        else:
            parent_channel = int(transition.right_channel)
            child_channel = int(transition.left_channel)
        summary.append(
            (
                parent_channel,
                child_channel,
                tuple(sorted(_family_names_from_symbolic_label(transition.label))),
            )
        )
    out = tuple(summary)
    if len(_SYMBOLIC_TRANSITION_SUMMARY_CACHE) > 512:
        _SYMBOLIC_TRANSITION_SUMMARY_CACHE.clear()
    _SYMBOLIC_TRANSITION_SUMMARY_CACHE[key] = out
    return out


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
    parent_summary = {}
    depth = 0
    for channel, terms in terms_by_channel.items():
        multiplicity = 0
        families = set()
        for term in terms:
            multiplicity += int(term.multiplicity)
            depth = max(depth, len(term.path))
            for path_item in tuple(term.path):
                if len(path_item) >= 4:
                    families.update(_family_names_from_symbolic_label(path_item[3]))
        parent_summary[int(channel)] = (int(multiplicity), tuple(sorted(families)))
    child_depth = int(depth) + 1
    counts = {}
    family_sets = {}
    for parent_channel, child_channel, transition_families in _symbolic_transition_summary(
        transitions,
        direction,
    ):
        if active is not None and child_channel not in active:
            continue
        multiplicity, parent_families = parent_summary.get(parent_channel, (0, ()))
        if multiplicity:
            counts[child_channel] = counts.get(child_channel, 0) + multiplicity
            families = family_sets.setdefault(child_channel, set())
            families.update(transition_families)
            families.update(parent_families)
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

    def prepare_miss(self, key):
        """Release the sole stale numeric problem before building its replacement."""

        if int(self.max_size) == 1 and key not in self.entries:
            self.entries.clear()

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
    packed_table: object | None = field(default=None, compare=False, repr=False)
    qchem_sweep_plan_cache: dict = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    qchem_sweep_plan_cache_stats: dict = field(
        default_factory=lambda: {"hits": 0, "misses": 0, "puts": 0},
        compare=False,
        repr=False,
    )

    def __post_init__(self):
        if self.packed_table is not None:
            return
        try:
            from .su2_qchem_plan import pack_side_operator_table

            self.packed_table = pack_side_operator_table(
                self.grouped_by_ket,
                side=self.owner_side,
                bond=self.owner_bond,
                representation=self.representation,
            )
        except Exception:
            self.packed_table = None

    def mark_hit(self):
        """
        Mark this side table as reused.

        :returns: ``self`` for call chaining.
        """

        self.hits += 1
        return self

    def get_qchem_sweep_plan(self, key):
        """
        Return a cached packed SU(2) qchem plan for this side table.

        :param key: Structural key describing the opposite table/boundaries.
        :returns: Cached plan or ``None``.
        """

        if key not in self.qchem_sweep_plan_cache:
            self.qchem_sweep_plan_cache_stats["misses"] += 1
            return None
        self.qchem_sweep_plan_cache_stats["hits"] += 1
        return self.qchem_sweep_plan_cache[key]

    def put_qchem_sweep_plan(self, key, plan):
        """
        Store a packed SU(2) qchem plan owned by this side table.

        :param key: Structural key describing the opposite table/boundaries.
        :param plan: :class:`SU2QChemSweepPlan` instance.
        :returns: Stored plan.
        """

        self.qchem_sweep_plan_cache[key] = plan
        self.qchem_sweep_plan_cache_stats["puts"] += 1
        return plan

    def grouped_payload(self):
        """
        Return the legacy grouped payload, materializing it from packed storage.

        The packed SU(2) qchem path does not call this in the hot local-action
        route.  It exists so the Python reference path can still be requested
        explicitly without forcing every boundary advance to rebuild dicts.
        """

        if self.grouped_by_ket is not None:
            return self.grouped_by_ket
        packed = getattr(self, "packed_table", None)
        if packed is None:
            return None
        if getattr(packed, "representation", None) == "rank_coupled_by_ket":
            sectors = tuple(packed.sector_codec.sectors)
            grouped = {}
            for row_idx, ket_id in enumerate(packed.ket_sector_ids):
                q_ket = sectors[int(ket_id)]
                entries = []
                start = int(packed.entry_offsets[row_idx])
                stop = int(packed.entry_offsets[row_idx + 1])
                for entry_idx in range(start, stop):
                    q_out = sectors[int(packed.out_sector_ids[entry_idx])]
                    channel_blocks = {}
                    c_start = int(packed.channel_offsets[entry_idx])
                    c_stop = int(packed.channel_offsets[entry_idx + 1])
                    for channel_idx in range(c_start, c_stop):
                        channel = int(packed.channel_ids[channel_idx])
                        channel_blocks[channel] = packed.block_pool.array(channel_idx)
                    if channel_blocks:
                        entries.append((q_out, channel_blocks))
                if entries:
                    grouped[q_ket] = tuple(entries)
            self.grouped_by_ket = grouped
            return self.grouped_by_ket
        if str(getattr(packed, "representation", "")).startswith("rank_coupled_"):
            boundary_sectors = tuple(packed.boundary_codec.sectors)
            physical_sectors = tuple(packed.physical_codec.sectors)
            grouped = {}
            for row_idx, (boundary_id, phys_id) in enumerate(
                zip(packed.key_boundary_ids, packed.key_physical_ids)
            ):
                key = (
                    boundary_sectors[int(boundary_id)],
                    physical_sectors[int(phys_id)],
                )
                entries = []
                start = int(packed.entry_offsets[row_idx])
                stop = int(packed.entry_offsets[row_idx + 1])
                for entry_idx in range(start, stop):
                    families = packed.families(entry_idx)
                    entry = (
                        boundary_sectors[int(packed.out_boundary_ids[entry_idx])],
                        physical_sectors[int(packed.out_physical_ids[entry_idx])],
                        int(packed.middle_ids[entry_idx]),
                        packed.factor(entry_idx),
                        families,
                    )
                    entries.append(entry)
                if entries:
                    grouped[key] = tuple(entries)
            self.grouped_by_ket = grouped
            return self.grouped_by_ket
        return None

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
        packed_table=None,
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
            packed_table=packed_table,
        )

    def advance_left(
        self,
        *,
        key,
        grouped_by_ket,
        owner_bond,
        parent_key=None,
        source="advanced_left",
        packed_table=None,
    ):
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
            packed_table=packed_table,
        )

    def advance_right(
        self,
        *,
        key,
        grouped_by_ket,
        owner_bond,
        parent_key=None,
        source="advanced_right",
        packed_table=None,
    ):
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
            packed_table=packed_table,
        )

    @property
    def n_ket_sectors(self):
        """Return the number of ket-sector groups in the table."""

        if self.grouped_by_ket is None:
            packed = getattr(self, "packed_table", None)
            if packed is None:
                return 0
            return int(getattr(packed, "n_keys", getattr(packed, "n_ket_sectors", 0)))
        return int(len(self.grouped_by_ket))

    @property
    def n_terms(self):
        """Return the number of grouped boundary terms."""

        if self.grouped_by_ket is None:
            packed = getattr(self, "packed_table", None)
            return int(0 if packed is None else getattr(packed, "n_entries", 0))
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
            "packed_only": bool(
                self.grouped_by_ket is None and self.packed_table is not None
            ),
            "packed_table": (
                None
                if self.packed_table is None
                else getattr(self.packed_table, "stats", None)
            ),
            "qchem_sweep_plan_cache_size": int(len(self.qchem_sweep_plan_cache)),
            "qchem_sweep_plan_cache_hits": int(
                self.qchem_sweep_plan_cache_stats["hits"]
            ),
            "qchem_sweep_plan_cache_misses": int(
                self.qchem_sweep_plan_cache_stats["misses"]
            ),
            "qchem_sweep_plan_cache_puts": int(
                self.qchem_sweep_plan_cache_stats["puts"]
            ),
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
                setattr(compiled, "local_operator_table", weakref.proxy(self))
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
        if (
            complementary_entry is not None
            and bool(
                getattr(
                    complementary_entry,
                    "materialize_family_operator_table",
                    True,
                )
            )
        ):
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
        packed_table=None,
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
                    packed_table=packed_table,
                )
            elif advance_direction == "right":
                table = parent_table.advance_right(
                    key=key,
                    grouped_by_ket=grouped_by_ket,
                    owner_bond=self.bond,
                    parent_key=self.parent_key,
                    source=source,
                    packed_table=packed_table,
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
                packed_table=packed_table,
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
                packed_table=packed_table,
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
    :param entries: Shared read-only mapping of index tuples to coefficients.
    :param internal_terms: Number of entries fully inside the boundary block.
    :param cross_terms: Number of entries connecting the block and exterior.
    :param external_terms: Number of entries fully outside the boundary block.
    """

    family_name: str
    entries: object
    internal_terms: int
    cross_terms: int
    external_terms: int
    coefficient_norm_value: float | None = None
    max_abs_coefficient_value: float | None = None

    @property
    def n_terms(self):
        """Return the number of stored sparse coefficients."""

        return int(len(self.entries))

    @property
    def coefficient_norm(self):
        """Return the Euclidean norm of stored numeric coefficients."""

        if self.coefficient_norm_value is not None:
            return float(self.coefficient_norm_value)
        if not self.entries:
            return 0.0
        values = np.asarray(
            [complex(value) for _key, value in self.entries.items()],
            dtype=complex,
        )
        return float(np.linalg.norm(values))

    @property
    def max_abs_coefficient(self):
        """Return the largest absolute coefficient in this payload."""

        if self.max_abs_coefficient_value is not None:
            return float(self.max_abs_coefficient_value)
        return float(
            max(
                (
                    abs(complex(value))
                    for _key, value in self.entries.items()
                ),
                default=0.0,
            )
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
    materialize_family_operator_table: bool = True

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
    materialize_family_operator_tables: bool = True
    entries: dict = field(default_factory=dict)
    puts: int = 0
    advances: int = 0
    _family_payload_sources: dict = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    _family_payload_norms: dict = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    _family_payload_max_abs: dict = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

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

        side = str(side).lower()
        bond = int(bond)
        payloads = {}
        families = getattr(self.families, "families", {}) or {}
        for name in self.family_names:
            family = families.get(name)
            entries = getattr(family, "entries", {}) if family is not None else {}
            if name not in self._family_payload_sources:
                self._family_payload_sources[name] = entries
                values = np.asarray(
                    [complex(value) for _key, value in entries.items()],
                    dtype=complex,
                )
                self._family_payload_norms[name] = float(np.linalg.norm(values))
                self._family_payload_max_abs[name] = float(
                    np.max(np.abs(values)) if values.size else 0.0
                )
            entries = self._family_payload_sources[name]
            partition_counts = getattr(entries, "partition_counts", None)
            native_counts = (
                None
                if partition_counts is None
                else partition_counts(side, bond)
            )
            native_indices = getattr(entries, "indices", None)
            if native_counts is not None:
                internal_terms, cross_terms, external_terms = (
                    int(value) for value in native_counts
                )
            elif native_indices is not None:
                native_indices = np.asarray(native_indices)
                if native_indices.size:
                    minimum = np.min(native_indices, axis=1)
                    maximum = np.max(native_indices, axis=1)
                    if side == "left":
                        internal_terms = int(np.count_nonzero(maximum < bond))
                        external_terms = int(np.count_nonzero(minimum >= bond))
                    elif side == "right":
                        internal_terms = int(np.count_nonzero(minimum > bond))
                        external_terms = int(np.count_nonzero(maximum <= bond))
                    else:
                        raise ValueError(
                            f"Unknown complementary boundary side {side!r}."
                        )
                    cross_terms = int(
                        len(entries) - internal_terms - external_terms
                    )
                else:
                    internal_terms = 0
                    cross_terms = 0
                    external_terms = int(len(entries))
            else:
                owned_sites = self._owned_sites(side, bond)
                internal_terms = 0
                cross_terms = 0
                external_terms = 0
                for key in entries:
                    key = tuple(int(index) for index in key)
                    if not key:
                        external_terms += 1
                        continue
                    flags = tuple(index in owned_sites for index in key)
                    if all(flags):
                        internal_terms += 1
                    elif any(flags):
                        cross_terms += 1
                    else:
                        external_terms += 1
            payloads[str(name)] = ComplementaryFamilyBoundaryPayload(
                family_name=str(name),
                entries=entries,
                internal_terms=int(internal_terms),
                cross_terms=int(cross_terms),
                external_terms=int(external_terms),
                coefficient_norm_value=self._family_payload_norms[name],
                max_abs_coefficient_value=self._family_payload_max_abs[name],
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
            materialize_family_operator_table=bool(
                self.materialize_family_operator_tables
            ),
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
        native_storage_bytes = int(
            sum(
                int(getattr(getattr(entries, "indices", None), "nbytes", 0))
                + int(getattr(getattr(entries, "values", None), "nbytes", 0))
                for entries in self._family_payload_sources.values()
            )
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
            "unique_numeric_payload_terms": int(
                sum(
                    len(entries)
                    for entries in self._family_payload_sources.values()
                )
            ),
            "shared_numeric_payload_sources": int(
                len(self._family_payload_sources)
            ),
            "native_numeric_payload_storage_bytes": native_storage_bytes,
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
    materialize_complementary_family_operator_tables: bool = True
    complementary_operator_stack: ComplementaryRenormalizedOperatorStack | None = None
    moving_environment_cache: MovingEnvironmentContractionCache = field(
        default_factory=MovingEnvironmentContractionCache
    )
    su2_operator_engine: object | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    su2_moving_environment: object | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    su2_boundary_environment: object | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    cpp_boundary_syncs: int = 0
    cpp_boundary_sync_failures: int = 0
    released_consumed_numeric_tables: int = 0
    released_consumed_boundaries: int = 0

    def __post_init__(self):
        if (
            self.complementary_operator_families is not None
            and self.complementary_operator_stack is None
        ):
            self.complementary_operator_stack = ComplementaryRenormalizedOperatorStack(
                families=self.complementary_operator_families,
                materialize_family_operator_tables=bool(
                    self.materialize_complementary_family_operator_tables
                ),
            )
        if (
            self.complementary_operator_families is not None
            and self.su2_operator_engine is None
        ):
            from .su2_qchem_plan import SU2OperatorEngine

            self.su2_operator_engine = SU2OperatorEngine(
                max_factor_tables=2,
                max_plans=1,
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
            self.su2_operator_engine = None
            return self
        if (
            self.complementary_operator_stack is not None
            and self.complementary_operator_families is families
        ):
            return self
        self.complementary_operator_families = families
        self.complementary_operator_stack = ComplementaryRenormalizedOperatorStack(
            families=families,
            materialize_family_operator_tables=bool(
                self.materialize_complementary_family_operator_tables
            ),
        )
        if self.su2_operator_engine is None:
            from .su2_qchem_plan import SU2OperatorEngine

            self.su2_operator_engine = SU2OperatorEngine(
                max_factor_tables=2,
                max_plans=1,
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
        if str(self.namespace) != "norm":
            entry.put_symbolic_operator_table(
                SymbolicRenormalizedOperatorTable.initialize(
                    entry.side,
                    entry.bond,
                    entry.block,
                    source=entry.source,
                )
            )
        self.prepopulate_side_operator_tables(entry, side_table_builders)
        self._sync_cpp_boundary(entry)
        return entry

    def _sync_cpp_boundary(self, entry):
        """Copy one reduced boundary arena into the persistent C++ owner."""

        owner = self.su2_boundary_environment or self.su2_moving_environment
        namespace = str(self.namespace)
        if (
            owner is None
            or namespace not in {"hamiltonian", "norm"}
            or entry is None
            or not bool(getattr(entry.block, "rank_coupled", False))
        ):
            return False
        try:
            ensure_packed = getattr(entry.block, "ensure_packed", None)
            packed = (
                ensure_packed(side=entry.side, bond=entry.bond)
                if ensure_packed is not None
                else None
            )
            if packed is None:
                from .su2_qchem_plan import (
                    pack_rank_coupled_boundary_table_from_block_map,
                )

                packed = pack_rank_coupled_boundary_table_from_block_map(
                    entry.block,
                    side=entry.side,
                    bond=entry.bond,
                    representation="rank_coupled_by_ket",
                )
            if packed is None or np.iscomplexobj(packed.block_pool.data):
                return False
            topology_arrays = (
                np.asarray(packed.ket_sector_ids, dtype=np.int64),
                np.asarray(packed.entry_offsets, dtype=np.int64),
                np.asarray(packed.out_sector_ids, dtype=np.int64),
                np.asarray(packed.channel_offsets, dtype=np.int64),
                np.asarray(packed.channel_ids, dtype=np.int64),
                np.asarray(packed.block_pool.shape_offsets, dtype=np.int64),
                np.asarray(packed.block_pool.shapes, dtype=np.int64),
            )
            header = np.asarray(
                [array.size for array in topology_arrays],
                dtype=np.int64,
            )
            labels = np.concatenate((header, *topology_arrays))
            digest = hashlib.blake2b(labels.tobytes(), digest_size=8).digest()
            topology_revision = int.from_bytes(digest, "little") or 1
            installed = (
                owner.metric_boundary_installed
                if namespace == "norm"
                else owner.boundary_installed
            )
            install = (
                owner.install_metric_boundary
                if namespace == "norm"
                else owner.install_boundary
            )
            if (
                bool(getattr(entry.block, "cpp_owned_boundary", False))
                and int(
                    getattr(entry.block, "cpp_topology_revision", 0)
                ) == int(topology_revision)
                and installed(
                    entry.side,
                    entry.bond,
                    int(topology_revision),
                    int(getattr(entry.block, "cpp_numeric_revision", 0)),
                )
            ):
                self.cpp_boundary_syncs += 1
                return True
            install(
                entry.side,
                entry.bond,
                np.asarray(packed.block_pool.data, dtype=float),
                np.asarray(packed.block_pool.offsets, dtype=np.int64),
                labels,
                topology_revision,
                int(self.puts),
            )
            self.cpp_boundary_syncs += 1
            return True
        except Exception:
            self.cpp_boundary_sync_failures += 1
            return False

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
            (
                grouped_by_ket,
                used_symbolic_table,
                packed_table,
            ) = self._build_side_operator_table_payload(
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
                packed_table=packed_table,
            )
        return entry

    def _build_side_operator_table_payload(self, entry, representation, fallback_builder):
        """
        Build grouped side-table payload from symbolic table metadata.

        :param entry: Boundary entry whose numeric block payload is grouped.
        :param representation: Requested side-table representation.
        :param fallback_builder: Numeric grouping fallback.
        :returns: ``(grouped_by_ket, used_symbolic_table, packed_table)``.
        """

        symbolic_table = getattr(entry, "symbolic_operator_table", None)
        if representation == "rank_coupled_by_ket":
            packed_table = None
            try:
                from .su2_qchem_plan import (
                    pack_rank_coupled_boundary_table_from_block_map,
                    pack_rank_coupled_boundary_table_from_payloads,
                )

                # A C++ boundary advance already produced the canonical packed
                # arena.  Repacking its symbolic payloads below promotes the
                # otherwise-real arena to complex128 and breaks raw compiled
                # factor routes on every interior bond.  Keep the boundary
                # owner's arena as the single numerical source of truth.
                packed_table = getattr(entry.block, "packed_table", None)
                if (
                    packed_table is not None
                    and str(getattr(packed_table, "side", "")) == str(entry.side)
                    and int(getattr(packed_table, "bond", -1)) == int(entry.bond)
                    and str(getattr(packed_table, "representation", ""))
                    == str(representation)
                ):
                    return None, symbolic_table is not None, packed_table
                if symbolic_table is not None and getattr(
                    symbolic_table,
                    "numeric_payloads",
                    None,
                ):
                    packed_table = pack_rank_coupled_boundary_table_from_payloads(
                        symbolic_table.numeric_payloads,
                        active_channels=getattr(symbolic_table, "channels", None),
                        side=entry.side,
                        bond=entry.bond,
                        representation=representation,
                    )
                    if packed_table is not None:
                        return None, True, packed_table
                active_channels = (
                    None
                    if symbolic_table is None
                    else getattr(symbolic_table, "channels", None)
                )
                packed_table = pack_rank_coupled_boundary_table_from_block_map(
                    entry.block,
                    active_channels=active_channels,
                    side=entry.side,
                    bond=entry.bond,
                    representation=representation,
                )
                if packed_table is not None:
                    return None, symbolic_table is not None, packed_table
            except Exception:
                packed_table = None
        if symbolic_table is not None:
            return (
                symbolic_table.group_boundary_blocks(representation=representation),
                True,
                None,
            )
        return fallback_builder(entry.block), False, None

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
        block = entry.block.advance(
            W,
            site,
            site,
            phys_slices=phys_slices,
            moving_environment=(
                self.su2_boundary_environment or self.su2_moving_environment
                if str(self.namespace) in {"hamiltonian", "norm"}
                else None
            ),
            parent_bond=entry.bond,
            child_bond=bond,
            numeric_revision=int(self.puts + 1),
        )
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
        self._sync_cpp_boundary(advanced)
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
        block = entry.block.advance(
            W,
            site,
            site,
            phys_slices=phys_slices,
            moving_environment=(
                self.su2_boundary_environment or self.su2_moving_environment
                if str(self.namespace) in {"hamiltonian", "norm"}
                else None
            ),
            parent_bond=entry.bond,
            child_bond=bond,
            numeric_revision=int(self.puts + 1),
        )
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
        self._sync_cpp_boundary(advanced)
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

    def release_consumed_numeric_tables(self, side, bond):
        """Release bond-local plans after a boundary is consumed."""

        entry = self.entries.get(self.key(side, bond))
        if entry is None:
            return 0
        released = int(len(entry.local_operator_tables))
        entry.local_operator_tables.clear()
        for table in tuple(entry.side_operator_tables.values()):
            released += int(len(table.qchem_sweep_plan_cache))
            table.qchem_sweep_plan_cache.clear()
        self.released_consumed_numeric_tables += int(released)
        return int(released)

    def release_consumed_boundary(self, side, bond):
        """Drop an obsolete prebuilt boundary after its bond was advanced."""

        owner = self.su2_boundary_environment or self.su2_moving_environment
        if owner is None:
            return False
        normalized_side = str(side).lower()
        normalized_bond = int(bond)
        system_stats = getattr(
            owner,
            "system_stats",
            {},
        )
        n_sites = int(system_stats.get("n_sites", 0))
        if (
            (normalized_side == "left" and normalized_bond == 0)
            or (
                normalized_side == "right"
                and normalized_bond == n_sites - 1
            )
        ):
            # These two scalar edge boundaries seed the opposite half sweep.
            # Their NumPy arenas are borrowed by the persistent C++ owner, so
            # retain the stack entries that keep those buffers alive.
            return False
        key = self.key(normalized_side, normalized_bond)
        entry = self.entries.pop(key, None)
        if entry is None:
            return False
        if self.complementary_operator_stack is not None:
            self.complementary_operator_stack.entries.pop(
                (normalized_side, int(bond)),
                None,
            )
        if str(self.namespace) != "norm":
            release = getattr(
                owner,
                "release_boundary",
                None,
            )
            if release is not None:
                release(normalized_side, normalized_bond)
        self.released_consumed_boundaries += 1
        return True

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
            "released_consumed_numeric_tables": int(
                self.released_consumed_numeric_tables
            ),
            "released_consumed_boundaries": int(
                self.released_consumed_boundaries
            ),
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
            "su2_operator_engine": (
                None
                if self.su2_operator_engine is None
                else self.su2_operator_engine.stats
            ),
            "su2_moving_environment": (
                None
                if self.su2_moving_environment is None
                else self.su2_moving_environment.stats
            ),
            "cpp_boundary_syncs": int(self.cpp_boundary_syncs),
            "cpp_boundary_sync_failures": int(
                self.cpp_boundary_sync_failures
            ),
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

    def matmat(self, vectors):
        """Apply the compiled table to a block of Davidson vectors."""

        vectors = np.asarray(vectors, dtype=complex)
        if vectors.ndim != 2 or int(vectors.shape[0]) != self.dim:
            raise ValueError(
                f"Expected a ({self.dim}, nvec) block, got {vectors.shape}."
            )
        if int(vectors.shape[1]) == 1:
            return self.matvec(vectors[:, 0]).reshape(self.dim, 1)
        if self.dense_matrix is not None:
            return self.dense_matrix @ vectors
        out = np.zeros_like(vectors, dtype=complex)
        for term in self.terms:
            out[term.output_slice, :] += (
                term.kernel @ vectors[term.input_slice, :]
            )
        return out

    def dense_operator_matrix(self):
        """Return the materialized orthonormal matrix when already owned."""

        return self.dense_matrix

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
    dense_kernel_skipped_total_budget: int = 0
    dense_kernel_skipped_threshold: int = 0
    kernel_policy: dict | None = None
    factor_batch_groups: int = 0
    factor_batched_entries: int = 0

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
        policy = get_complementary_family_kernel_policy()
        requested_backend = str(policy["backend"])
        dense_threshold = int(policy["dense_threshold"])
        dense_total_cap = policy["dense_max_total_elements"]
        dense_total_cap = None if dense_total_cap is None else int(dense_total_cap)
        dense_total_used = 0
        skipped_total_budget = 0
        skipped_threshold = 0

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
            apply_entry = ComplementaryFamilyApplyEntry.from_plan_entry(
                entry,
                family_names=group_names,
                source_tables=source_tables,
            )
            if requested_backend in {"auto", "factor"}:
                apply_entry = apply_entry.with_factor_kernel()
            if requested_backend in {"auto", "dense"} and dense_threshold > 0:
                dense_elements = apply_entry.dense_kernel_elements
                if dense_total_cap is not None and dense_total_used + dense_elements > dense_total_cap:
                    skipped_total_budget += 1
                else:
                    with_dense = apply_entry.with_native_kernel(
                        max_elements=dense_threshold
                    )
                    if with_dense.native_kernel is None:
                        skipped_threshold += 1
                    else:
                        apply_entry = with_dense
                        dense_total_used += int(np.asarray(with_dense.native_kernel).size)
            elif requested_backend in {"auto", "dense"}:
                skipped_threshold += 1
            if requested_backend == "dense" and apply_entry.native_kernel is None:
                apply_entry = apply_entry.with_factor_kernel()
            grouped.setdefault(family, []).append(apply_entry)
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
            dense_kernel_skipped_total_budget=int(skipped_total_budget),
            dense_kernel_skipped_threshold=int(skipped_threshold),
            kernel_policy={
                "backend": str(requested_backend),
                "dense_threshold": int(dense_threshold),
                "dense_max_total_elements": dense_total_cap,
                "dense_used_elements": int(native_kernel_elements),
            },
            factor_batch_groups=int(
                sum(
                    1
                    for entries in grouped.values()
                    for _signature, batch_entries in _group_factor_batch_entries(
                        entries
                    ).items()
                    if len(batch_entries)
                    >= _COMPLEMENTARY_FAMILY_FACTOR_BATCH_MIN_ENTRIES
                )
            ),
            factor_batched_entries=int(
                sum(
                    len(batch_entries)
                    for entries in grouped.values()
                    for _signature, batch_entries in _group_factor_batch_entries(
                        entries
                    ).items()
                    if len(batch_entries)
                    >= _COMPLEMENTARY_FAMILY_FACTOR_BATCH_MIN_ENTRIES
                )
            ),
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
            batched = set()
            for batch_entries in _group_factor_batch_entries(entries).values():
                if (
                    len(batch_entries)
                    < _COMPLEMENTARY_FAMILY_FACTOR_BATCH_MIN_ENTRIES
                ):
                    continue
                self._apply_factor_batch(batch_entries, parent_inputs, parent_outputs)
                batched.update(id(entry) for entry in batch_entries)
            for entry in entries:
                if id(entry) in batched:
                    continue
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

    def _apply_factor_batch(self, entries, parent_inputs, parent_outputs):
        """
        Apply a group of same-shape factor-native kernels with batched matmul.

        :param entries: Batchable :class:`ComplementaryFamilyApplyEntry` objects.
        :param parent_inputs: Component parent input buffers.
        :param parent_outputs: Component parent output buffers updated in place.
        """

        kernels = [entry.factor_kernel for entry in entries]
        first = kernels[0]
        left_mats = np.stack([kernel.left_matrix for kernel in kernels], axis=0)
        right_mats = np.stack([kernel.right_matrix for kernel in kernels], axis=0)
        input_mats = np.stack(
            [
                parent_inputs[int(entry.in_comp)][entry.in_slice]
                .reshape(entry.input_entry.shape)
                .reshape(
                    int(np.prod(entry.input_entry.shape[:2], dtype=int)),
                    int(np.prod(entry.input_entry.shape[2:], dtype=int)),
                )
                for entry in entries
            ],
            axis=0,
        )
        tmp = np.matmul(left_mats, input_mats).reshape(
            (len(entries),) + tuple(first.tmp_shape)
        )
        ldim, adim, ddim, qdim = (int(dim) for dim in first.output_shape)
        tmp_mats = np.ascontiguousarray(
            tmp.transpose(0, 2, 4, 1, 3, 6, 5).reshape(
                len(entries),
                ldim * adim,
                -1,
            )
        )
        contribs = np.matmul(tmp_mats, right_mats).reshape(
            len(entries),
            ldim * adim * ddim * qdim,
        )
        for entry, contrib in zip(entries, contribs):
            parent_outputs[int(entry.out_comp)][entry.out_slice] += contrib

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
            "dense_kernel_skipped_total_budget": int(
                self.dense_kernel_skipped_total_budget
            ),
            "dense_kernel_skipped_threshold": int(
                self.dense_kernel_skipped_threshold
            ),
            "kernel_policy": dict(self.kernel_policy or {}),
            "factor_batch_groups": int(self.factor_batch_groups),
            "factor_batched_entries": int(self.factor_batched_entries),
        }


def _group_factor_batch_entries(entries):
    """Group batchable family apply entries by factor-kernel shape."""

    groups = OrderedDict()
    for entry in tuple(entries or ()):
        signature = entry.factor_batch_signature
        if signature is None:
            continue
        groups.setdefault(signature, []).append(entry)
    return groups


def _component_block_residual(blocks, reference_blocks):
    """Return relative residual between two component-block tables."""

    if blocks is None or reference_blocks is None:
        return float("inf")
    table = {
        (int(in_comp), int(out_comp)): np.asarray(block, dtype=complex)
        for in_comp, out_comp, block in blocks
    }
    reference = {
        (int(in_comp), int(out_comp)): np.asarray(block, dtype=complex)
        for in_comp, out_comp, block in reference_blocks
    }
    keys = set(table) | set(reference)
    diff_norm = 0.0
    ref_norm = 0.0
    for key in keys:
        a = table.get(key)
        b = reference.get(key)
        if a is None:
            diff_norm += float(np.linalg.norm(b.reshape(-1)) ** 2)
            ref_norm += float(np.linalg.norm(b.reshape(-1)) ** 2)
            continue
        if b is None:
            diff_norm += float(np.linalg.norm(a.reshape(-1)) ** 2)
            continue
        diff_norm += float(np.linalg.norm((a - b).reshape(-1)) ** 2)
        ref_norm += float(np.linalg.norm(b.reshape(-1)) ** 2)
    return float(np.sqrt(diff_norm) / max(np.sqrt(ref_norm), 1.0))


@dataclass(frozen=True)
class MovingEnvironmentFactorRouteTable:
    """Thin Python handle for a projected route owned entirely by C++."""

    owner: object
    projection_key: str
    dim: int
    active_complementary: bool = False

    def matvec(self, vector):
        return self.owner.factor_route_projected_matvec(
            self.projection_key,
            np.asarray(vector),
        )

    def matmat(self, vectors):
        vectors = np.asarray(vectors, dtype=complex)
        return np.column_stack(
            [self.matvec(vectors[:, index]) for index in range(vectors.shape[1])]
        )

    def diagonal(self):
        return None

    def davidson(
        self,
        diagonal,
        guess,
        tolerance,
        max_iterations,
        restart_dimension,
        accept_unconverged=False,
    ):
        if self.active_complementary and hasattr(
            self.owner,
            "active_bond_complementary_davidson",
        ) and hasattr(
            self.owner,
            "active_bond_complementary_action_ready",
        ) and self.owner.active_bond_complementary_action_ready(
            self.projection_key,
            int(self.dim),
        ):
            return self.owner.active_bond_complementary_davidson(
                self.projection_key,
                guess,
                float(tolerance),
                int(max_iterations),
                int(restart_dimension),
                bool(accept_unconverged),
            )
        return self.owner.factor_route_projected_davidson(
            self.projection_key,
            diagonal,
            guess,
            float(tolerance),
            int(max_iterations),
            int(restart_dimension),
            bool(accept_unconverged),
        )

    @property
    def stats(self):
        owner_stats = self.owner.stats
        projected_matvec_calls = int(
            owner_stats.get("factor_route_projected_matvec_calls", 0)
        )
        projected_davidson_calls = int(
            owner_stats.get("factor_route_projected_davidson_calls", 0)
        )
        return {
            "kind": "cpp_su2_moving_environment_factor_routes",
            "projection_key": str(self.projection_key),
            "dimension": int(self.dim),
            "raw_factor_routes": bool(owner_stats.get("raw_factor_routes", False)),
            "factor_route_count": int(owner_stats.get("factor_route_count", 0)),
            "raw_factor_cache_bytes": int(
                owner_stats.get("raw_factor_cache_bytes", 0)
            ),
            "raw_factor_gemm_calls": int(
                owner_stats.get("raw_factor_gemm_calls", 0)
            ),
            "raw_route_group_count": int(
                owner_stats.get("raw_route_group_count", 0)
            ),
            "fused_raw_route_group_count": int(
                owner_stats.get("fused_raw_route_group_count", 0)
            ),
            "fused_raw_route_count": int(
                owner_stats.get("fused_raw_route_count", 0)
            ),
            "dense_pair_kernel_count": int(
                owner_stats.get("dense_pair_kernel_count", 0)
            ),
            "dense_pair_execution_count": int(
                owner_stats.get("dense_pair_execution_count", 0)
            ),
            "dense_pair_kernel_elements": int(
                owner_stats.get("dense_pair_kernel_elements", 0)
            ),
            "dense_pair_route_count": int(
                owner_stats.get("dense_pair_route_count", 0)
            ),
            "raw_execution_group_count": int(
                owner_stats.get("raw_execution_group_count", 0)
            ),
            "raw_execution_action_count": int(
                owner_stats.get("raw_execution_action_count", 0)
            ),
            "raw_input_superchannel_count": int(
                owner_stats.get("raw_input_superchannel_count", 0)
            ),
            "raw_input_superchannel_tile_count": int(
                owner_stats.get("raw_input_superchannel_tile_count", 0)
            ),
            "matvec_calls": projected_matvec_calls,
            "davidson_calls": projected_davidson_calls,
            "factor_route_projected_matvec_calls": projected_matvec_calls,
            "factor_route_projected_davidson_calls": projected_davidson_calls,
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
    su2_moving_environment: object | None = None
    local_operator_key: object | None = None

    def __post_init__(self):
        compiled_owner = getattr(
            self.compiled_factorized_terms,
            "su2_moving_environment",
            None,
        )
        if self.su2_moving_environment is None and compiled_owner is not None:
            object.__setattr__(
                self,
                "su2_moving_environment",
                compiled_owner,
            )
        timing = {}
        t0 = time.perf_counter()
        estimated_parent_block_elements = (
            self._estimate_qchem_component_parent_block_elements()
            if _SU2_QCHEM_DIRECT_PARENT_BLOCKS
            else None
        )
        oversized_parent_blocks = bool(
            estimated_parent_block_elements is not None
            and int(estimated_parent_block_elements)
            > int(_SU2_QCHEM_DIRECT_PARENT_BLOCK_MAX_ELEMENTS)
        )
        force_family_tensor = bool(
            self.uses_complementary_payload_tensor_kernel
        )
        force_component_direct = bool(
            getattr(
                self.compiled_factorized_terms,
                "explicit_direct_orthonormal_projection",
                False,
            )
        )
        packed_oversized_family_route = bool(
            not force_family_tensor
            and
            not force_component_direct
            and
            _resolve_su2_kernel_backend(_SU2_KERNEL_BACKEND) == "cpp"
            and getattr(
                self.compiled_factorized_terms,
                "qchem_packed_entry_kernel_provider",
                False,
            )
        )
        qchem_parent_blocks = (
            self._build_qchem_component_parent_blocks()
            if (
                _SU2_QCHEM_DIRECT_PARENT_BLOCKS
                and not oversized_parent_blocks
                and not force_family_tensor
                and not force_component_direct
                and not packed_oversized_family_route
            )
            else None
        )
        timing["direct_table_qchem_parent_blocks"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        factorized_fallback_terms = None
        if (
            oversized_parent_blocks or force_component_direct
        ) and not packed_oversized_family_route:
            factorized_provider = getattr(
                self.compiled_factorized_terms,
                "factorized_terms",
                None,
            )
            if factorized_provider is not None:
                factorized_fallback_terms = factorized_provider()
        component_direct_plan = (
            None
            if (
                qchem_parent_blocks is not None
                or packed_oversized_family_route
            )
            else self._build_component_direct_plan(
                compiled=factorized_fallback_terms,
            )
        )
        timing["direct_table_component_plan"] = time.perf_counter() - t0
        object.__setattr__(
            self,
            "_component_direct_plan",
            component_direct_plan,
        )
        compiled = self.compiled_factorized_terms
        cpp_owner_factor_route = bool(
            packed_oversized_family_route
            and self.su2_moving_environment is not None
            and getattr(compiled, "_cpp_factor_routes_installed", False)
            and getattr(compiled, "su2_moving_environment", None)
            is self.su2_moving_environment
        )
        object.__setattr__(
            self,
            "_cpp_owner_factor_route_status",
            {
                "eligible": bool(cpp_owner_factor_route),
                "packed_route_requested": bool(packed_oversized_family_route),
                "moving_environment": bool(
                    self.su2_moving_environment is not None
                ),
                "compiled_route_installed": bool(
                    getattr(compiled, "_cpp_factor_routes_installed", False)
                ),
                "shared_owner": bool(
                    compiled is not None
                    and getattr(compiled, "su2_moving_environment", None)
                    is self.su2_moving_environment
                ),
            },
        )
        oversized_family_fallback = bool(
            oversized_parent_blocks
            and component_direct_plan is not None
        )
        use_family_tensor = bool(
            force_family_tensor
            or oversized_family_fallback
        )
        t0 = time.perf_counter()
        family_table = (
            self._build_complementary_family_tensor_table(
                component_direct_plan
            )
            if use_family_tensor
            else None
        )
        timing["direct_table_family_table"] = time.perf_counter() - t0
        object.__setattr__(
            self,
            "_complementary_family_tensor_table",
            family_table,
        )
        t0 = time.perf_counter()
        component_parent_blocks = (
            None if use_family_tensor else qchem_parent_blocks
        )
        if (
            component_parent_blocks is None
            and not use_family_tensor
            and not oversized_parent_blocks
            and not packed_oversized_family_route
        ):
            component_parent_blocks = self._build_component_parent_blocks(
                component_direct_plan
            )
        timing["direct_table_parent_blocks"] = (
            timing.get("direct_table_parent_blocks", 0.0)
            + (time.perf_counter() - t0)
        )
        t0 = time.perf_counter()
        component_orthonormal_blocks = self._build_component_orthonormal_blocks(
            component_parent_blocks,
        )
        timing["direct_table_orthonormal_blocks"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cpp_davidson_table = self._build_cpp_davidson_table(
            component_orthonormal_blocks,
        )
        timing["direct_table_cpp_davidson"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        component_orthonormal_dense_matrix = self._build_component_orthonormal_dense_matrix(
            component_orthonormal_blocks,
        )
        timing["direct_table_dense_matrix"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        component_orthonormal_block_batches = (
            None
            if component_orthonormal_dense_matrix is not None
            else self._build_component_orthonormal_block_batches(
                component_orthonormal_blocks,
            )
        )
        timing["direct_table_orthonormal_block_batches"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if cpp_owner_factor_route:
            cpp_davidson_table = self._install_cpp_factor_route_projection(
                compiled
            )
            su2_action = None
        elif family_table is not None:
            su2_action = SU2LocalAction.from_family_table(
                self.component_basis,
                family_table,
                backend=_SU2_KERNEL_BACKEND,
            )
        elif component_orthonormal_blocks is None:
            su2_action = SU2LocalAction.from_parent_blocks(
                self.component_basis,
                component_parent_blocks,
                backend=_SU2_KERNEL_BACKEND,
            )
        else:
            su2_action = None
        if (
            su2_action is None
            and packed_oversized_family_route
            and not cpp_owner_factor_route
        ):
            su2_action = SU2LocalAction.from_packed_family_terms(
                self.component_basis,
                self.compiled_factorized_terms,
                backend=_SU2_KERNEL_BACKEND,
            )
        timing["direct_table_su2_action"] = time.perf_counter() - t0
        if (
            cpp_davidson_table is None
            and su2_action is not None
            and getattr(su2_action, "_cpp_family_table", None) is not None
        ):
            cpp_davidson_table = su2_action._cpp_family_table
        su2_residual = None
        if su2_action is not None and _SU2_KERNEL_DEBUG_CHECK:
            t0 = time.perf_counter()
            rng = np.random.default_rng(1234)
            probe = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
            if family_table is not None:
                reference = family_table.matvec(probe, self.component_basis)
            elif component_parent_blocks is not None:
                reference = self._component_parent_block_matvec(
                    probe,
                    component_parent_blocks,
                )
            else:
                parent = self.component_basis.from_orthonormal(probe)
                parent_out = np.asarray(
                    self.packed_matvec(parent),
                    dtype=complex,
                ).reshape(self.component_basis.parent_dim)
                reference = np.zeros(self.dim, dtype=complex)
                for idx, indices in enumerate(
                    self.component_basis.component_indices
                ):
                    transform = self.component_basis.component_transforms[idx]
                    start = int(self.component_basis.orth_offsets[idx])
                    stop = start + int(transform.shape[1])
                    reference[start:stop] = (
                        transform.conj().T @ parent_out[indices]
                    )
            candidate = su2_action.matvec(probe)
            scale = max(float(np.linalg.norm(reference)), 1.0)
            su2_residual = float(np.linalg.norm(candidate - reference) / scale)
            if su2_residual > float(_SU2_KERNEL_DEBUG_CHECK_TOL):
                raise RuntimeError(
                    "SU2 local action disagrees with Python reference: "
                    f"residual={su2_residual:.3e}"
                )
            timing["direct_table_su2_debug_check"] = time.perf_counter() - t0
        cpp_family_table = (
            None
            if su2_action is None
            else getattr(su2_action, "_cpp_family_table", None)
        )
        cpp_family_source_stats = (
            None if family_table is None else dict(family_table.stats)
        ) if cpp_family_table is not None else None
        if (
            (cpp_family_table is not None or cpp_owner_factor_route)
            and cpp_family_source_stats is None
        ):
            source_families = getattr(
                self.compiled_factorized_terms,
                "complementary_operator_families",
                None,
            )
            source_family_metadata = (
                source_families.as_metadata()
                if hasattr(source_families, "as_metadata")
                else None
            )
            source_payloads = getattr(
                self.compiled_factorized_terms,
                "complementary_boundary_payloads",
                None,
            )
            cpp_family_source_stats = {
                "kind": "packed_qchem_factor_routes",
                "source": "packed_su2_qchem_factor_pools",
                "family_names": tuple(
                    getattr(self.compiled_factorized_terms, "family_names", ())
                ),
                "family_term_counts": dict(
                    getattr(
                        self.compiled_factorized_terms,
                        "family_term_counts",
                        {},
                    )
                    or {}
                ),
                "complementary_operator_families": source_family_metadata,
                "complementary_payload_backed": bool(
                    source_payloads is not None
                    and source_payloads.get("payload_backed", False)
                ),
                "complementary_payload_terms": int(
                    0
                    if source_payloads is None
                    else source_payloads.get("numeric_payload_terms", 0)
                ),
                "su2_qchem_sweep_plan": getattr(
                    self.compiled_factorized_terms,
                    "su2_qchem_sweep_plan",
                    None,
                ),
                "su2_qchem_factor_match_backend": getattr(
                    self.compiled_factorized_terms,
                    "su2_qchem_factor_match_backend",
                    None,
                ),
                "su2_qchem_factor_match_count": getattr(
                    self.compiled_factorized_terms,
                    "su2_qchem_factor_match_count",
                    None,
                ),
                "cpp_table": dict(cpp_davidson_table.stats),
            }
        packed_cpp_owner = bool(
            cpp_owner_factor_route
            or (
                cpp_family_table is not None
                and str(cpp_family_table.stats.get("kind", ""))
                == "cpp_su2_packed_factorized_family_table"
            )
        )
        if cpp_family_table is not None or cpp_owner_factor_route:
            family_table = None
            component_direct_plan = None
            factorized_fallback_terms = None
        object.__setattr__(self, "_build_timing", timing)
        object.__setattr__(self, "_su2_action", su2_action)
        object.__setattr__(
            self,
            "_su2_action_reference_residual",
            su2_residual,
        )
        object.__setattr__(
            self,
            "_cpp_family_source_stats",
            cpp_family_source_stats,
        )
        object.__setattr__(
            self,
            "_packed_cpp_exclusive_owner",
            packed_cpp_owner,
        )
        if packed_cpp_owner:
            object.__setattr__(self, "compiled_factorized_terms", None)
            object.__setattr__(self, "packed_matvec", None)
            object.__setattr__(self, "components", None)
        object.__setattr__(
            self,
            "_component_parent_blocks",
            component_parent_blocks,
        )
        object.__setattr__(
            self,
            "_estimated_parent_block_elements",
            estimated_parent_block_elements,
        )
        object.__setattr__(
            self,
            "_oversized_parent_block_fallback",
            oversized_parent_blocks,
        )
        object.__setattr__(
            self,
            "_oversized_parent_block_family_fallback",
            oversized_family_fallback,
        )
        object.__setattr__(
            self,
            "_factorized_fallback_terms",
            factorized_fallback_terms,
        )
        object.__setattr__(
            self,
            "_component_orthonormal_blocks",
            component_orthonormal_blocks,
        )
        object.__setattr__(
            self,
            "_cpp_davidson_table",
            cpp_davidson_table,
        )
        object.__setattr__(
            self,
            "_cpp_factor_route_projection",
            cpp_owner_factor_route,
        )
        object.__setattr__(
            self,
            "_component_orthonormal_block_batches",
            component_orthonormal_block_batches,
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
        family_table = getattr(self, "_complementary_family_tensor_table", None)
        su2_action = getattr(self, "_su2_action", None)
        cpp_davidson_table = getattr(self, "_cpp_davidson_table", None)
        if family_table is not None:
            if su2_action is not None:
                return su2_action.matvec(vector)
            return family_table.matvec(vector, self.component_basis)
        cpp_table = getattr(self, "_cpp_davidson_table", None)
        if cpp_table is not None:
            return cpp_table.matvec(vector)
        orthonormal_dense = getattr(self, "_component_orthonormal_dense_matrix", None)
        if orthonormal_dense is not None:
            return orthonormal_dense @ vector
        orthonormal_blocks = getattr(self, "_component_orthonormal_blocks", None)
        if orthonormal_blocks is not None:
            batches = getattr(self, "_component_orthonormal_block_batches", None)
            if batches is not None:
                return self._component_orthonormal_batched_block_matvec(
                    vector,
                    batches,
                )
            return self._component_orthonormal_block_matvec(vector, orthonormal_blocks)
        parent_blocks = getattr(self, "_component_parent_blocks", None)
        if parent_blocks is not None:
            su2_action = getattr(self, "_su2_action", None)
            if su2_action is not None:
                return su2_action.matvec(vector)
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

    def matmat(self, vectors):
        """Apply ``X^H H X`` to a block of Davidson vectors."""

        vectors = np.asarray(vectors, dtype=complex)
        if vectors.ndim != 2 or int(vectors.shape[0]) != self.dim:
            raise ValueError(
                f"Expected a ({self.dim}, nvec) block, got {vectors.shape}."
            )
        if int(vectors.shape[1]) == 1:
            return self.matvec(vectors[:, 0]).reshape(self.dim, 1)
        family_table = getattr(self, "_complementary_family_tensor_table", None)
        su2_action = getattr(self, "_su2_action", None)
        if family_table is not None and su2_action is not None:
            return su2_action.matmat(vectors)
        cpp_table = getattr(self, "_cpp_davidson_table", None)
        cpp_matmat = getattr(cpp_table, "matmat", None)
        if callable(cpp_matmat):
            return np.asarray(cpp_matmat(vectors))
        orthonormal_dense = getattr(self, "_component_orthonormal_dense_matrix", None)
        if orthonormal_dense is not None:
            return orthonormal_dense @ vectors
        return np.column_stack(
            [self.matvec(vectors[:, idx]) for idx in range(vectors.shape[1])]
        )

    def dense_operator_matrix(self):
        """Return the materialized orthonormal matrix when already owned."""

        return getattr(self, "_component_orthonormal_dense_matrix", None)

    def cpp_davidson(
        self,
        diag,
        guess,
        *,
        tol,
        max_iter,
        restart_dim,
        accept_unconverged=False,
        block_size=1,
    ):
        """Solve this transformed block table without Python matvec callbacks."""

        table = getattr(self, "_cpp_davidson_table", None)
        if table is None:
            return None
        cpp_diag = np.ascontiguousarray(diag, dtype=complex)
        table_diagonal = getattr(table, "diagonal", None)
        table_diagonal_used = False
        if callable(table_diagonal):
            candidate = table_diagonal()
            if candidate is not None:
                candidate = np.asarray(candidate, dtype=float).reshape(-1)
                if candidate.size == cpp_diag.size:
                    cpp_diag = np.ascontiguousarray(
                        candidate,
                        dtype=complex,
                    )
                    table_diagonal_used = True
        block_solver = getattr(table, "davidson_block", None)
        if int(block_size) > 1 and callable(block_solver):
            result = block_solver(
                cpp_diag,
                np.ascontiguousarray(guess, dtype=complex),
                float(tol),
                int(max_iter),
                int(restart_dim),
                bool(accept_unconverged),
                int(block_size),
            )
        else:
            result = table.davidson(
                cpp_diag,
                np.ascontiguousarray(guess, dtype=complex),
                float(tol),
                int(max_iter),
                int(restart_dim),
                bool(accept_unconverged),
            )
        result["table_diagonal_used"] = bool(table_diagonal_used)
        return result

    def cpp_lanczos_expm_apply(self, vector, dt, *, krylov_dim, tol):
        """Propagate with the complex Lanczos loop owned by the C++ local table."""
        table = getattr(self, "_cpp_davidson_table", None)
        apply = getattr(table, "lanczos_expm_apply", None)
        if not callable(apply):
            dense = getattr(self, "_component_orthonormal_dense_matrix", None)
            blocks = getattr(self, "_component_orthonormal_blocks", None)
            table = getattr(self, "_cpp_tdvp_lanczos_table", None)
            if (dense is not None or blocks is not None) and table is None:
                try:
                    from pyqed.mps import cpp_davidson

                    table_cls = getattr(
                        cpp_davidson,
                        "SU2FactorizedFamilyTable",
                        None,
                    )
                    if table_cls is not None and hasattr(
                        table_cls,
                        "lanczos_expm_apply",
                    ):
                        dim = int(self.dim)
                        transform = (
                            "diagonal",
                            0,
                            dim,
                            np.arange(dim, dtype=np.int64),
                            np.ones(dim, dtype=complex),
                        )
                        if dense is not None:
                            kernels = ((0, dim, 0, dim, dense),)
                        else:
                            kernels = tuple(
                                (
                                    int(self.component_basis._orth_slice(in_comp).start),
                                    int(block.shape[1]),
                                    int(self.component_basis._orth_slice(out_comp).start),
                                    int(block.shape[0]),
                                    block,
                                )
                                for in_comp, out_comp, block in blocks
                            )
                        entries = tuple(
                            (
                                0,
                                0,
                                in_start,
                                in_size,
                                out_start,
                                out_size,
                                np.ascontiguousarray(kernel, dtype=complex),
                                np.ones((1, 1), dtype=complex),
                                (1, out_size, 1, 1, 1, 1),
                                (out_size, 1, 1, 1),
                                (in_size, 1, 1, 1),
                                out_size,
                            )
                            for in_start, in_size, out_start, out_size, kernel in kernels
                        )
                        table = table_cls((transform,), entries, dim)
                        object.__setattr__(self, "_cpp_tdvp_lanczos_table", table)
                except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
                    table = None
            apply = getattr(table, "lanczos_expm_apply", None)
        if not callable(apply):
            return None
        try:
            return apply(
                np.ascontiguousarray(vector, dtype=complex),
                float(dt),
                int(krylov_dim),
                float(tol),
            )
        except RuntimeError:
            return None

    def _install_cpp_factor_route_projection(self, compiled):
        """Install the orthonormal transform beside the owner's raw routes."""

        owner = self.su2_moving_environment
        factor_route_key = getattr(compiled, "_cpp_factor_route_key", None)
        if owner is None or factor_route_key is None:
            raise RuntimeError(
                "The C++ SU(2) factor route is unavailable for projection."
            )
        indices = tuple(
            np.ascontiguousarray(value, dtype=np.int64)
            for value in self.component_basis.component_indices
        )
        transforms = tuple(
            np.ascontiguousarray(value)
            for value in self.component_basis.component_transforms
        )
        offsets = tuple(int(value) for value in self.component_basis.orth_offsets)
        topology_arrays = [
            np.asarray(
                [
                    int(self.component_basis.parent_dim),
                    int(self.component_basis.orthonormal_dim),
                    len(indices),
                ],
                dtype=np.int64,
            ),
            np.asarray(offsets, dtype=np.int64),
        ]
        for index, transform in zip(indices, transforms):
            topology_arrays.extend(
                (
                    index,
                    np.asarray(transform.shape, dtype=np.int64),
                )
            )
        topology_revision = _stable_array_revision(*topology_arrays)
        numeric_revision = _stable_array_revision(*transforms)
        projection_key = (
            f"projection:{factor_route_key}:{int(topology_revision)}"
        )
        owner.install_factor_route_projection(
            projection_key,
            factor_route_key,
            indices,
            transforms,
            offsets,
            int(self.component_basis.parent_dim),
            int(self.component_basis.orthonormal_dim),
            int(topology_revision),
            int(numeric_revision),
        )
        return MovingEnvironmentFactorRouteTable(
            owner=owner,
            projection_key=projection_key,
            dim=int(self.component_basis.orthonormal_dim),
            active_complementary=bool(
                getattr(compiled, "cpp_owned_basis_topology", False)
            ),
        )

    def _build_cpp_davidson_table(self, orthonormal_blocks):
        """Build one persistent C++ block table for matvec and Davidson."""

        if (
            orthonormal_blocks is None
            or str(_SU2_KERNEL_BACKEND).lower().replace("-", "_") == "python"
        ):
            return None
        try:
            from pyqed.mps import cpp_davidson

            table_cls = getattr(cpp_davidson, "BlockTable", None)
            if table_cls is None:
                return None
            blocks = []
            in_starts = []
            out_starts = []
            for in_comp, out_comp, block in orthonormal_blocks:
                in_slice = self.component_basis._orth_slice(int(in_comp))
                out_slice = self.component_basis._orth_slice(int(out_comp))
                blocks.append(np.ascontiguousarray(block, dtype=complex))
                in_starts.append(int(in_slice.start))
                out_starts.append(int(out_slice.start))
            if not blocks:
                return None
            return table_cls(
                tuple(blocks),
                np.asarray(in_starts, dtype=np.int64),
                np.asarray(out_starts, dtype=np.int64),
                int(self.dim),
            )
        except Exception:
            return None

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
        """

        return getattr(self, "_component_orthonormal_blocks", None) is not None

    @property
    def uses_component_orthonormal_dense_kernel(self):
        """
        Return whether matvecs use one dense orthonormal local matrix.
        """

        return getattr(self, "_component_orthonormal_dense_matrix", None) is not None

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

        return bool(
            getattr(self, "_complementary_family_tensor_table", None) is not None
            or getattr(self, "_cpp_family_source_stats", None) is not None
        )

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


    def _build_component_direct_plan(self, *, compiled=None):
        """
        Build a component-local factorized block application plan.

        :returns: Tuple of plan entries or ``None`` when metadata is missing.
        """

        if compiled is None:
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
        if not plan:
            return None
        return tuple(plan)

    def _estimate_qchem_component_parent_block_elements(self):
        """
        Estimate dense parent-block storage without materializing any blocks.

        The packed qchem schedule already owns input/output entry indices.
        Mapping those indices to metric components gives the unique dense
        component-pair shapes that a parent-block action would allocate.
        """

        compiled = self.compiled_factorized_terms
        components = self.components
        if compiled is None or components is None:
            return None
        in_indices = getattr(compiled, "in_indices", None)
        out_indices = getattr(compiled, "out_indices", None)
        if in_indices is None or out_indices is None:
            return None
        in_indices = np.asarray(in_indices, dtype=np.int64).reshape(-1)
        out_indices = np.asarray(out_indices, dtype=np.int64).reshape(-1)
        if in_indices.size != out_indices.size:
            return None
        entry_to_component = {}
        for comp_idx, component in enumerate(components):
            for entry_idx in component:
                entry_to_component[int(entry_idx)] = int(comp_idx)
        pairs = set()
        for in_idx, out_idx in zip(in_indices, out_indices):
            in_comp = entry_to_component.get(int(in_idx))
            out_comp = entry_to_component.get(int(out_idx))
            if in_comp is None or out_comp is None:
                return None
            pairs.add((in_comp, out_comp))
        component_dims = tuple(
            int(np.asarray(indices).size)
            for indices in self.component_basis.component_indices
        )
        return int(
            sum(
                component_dims[out_comp] * component_dims[in_comp]
                for in_comp, out_comp in pairs
            )
        )

    def _build_qchem_component_parent_blocks(self):
        """
        Build component parent blocks directly from a packed SU(2) qchem plan.

        :returns: Parent component blocks, or ``None`` when unavailable.
        """

        compiled = self.compiled_factorized_terms
        components = self.components
        basis = getattr(self.component_basis, "parent_basis", None)
        qchem_plan = getattr(compiled, "su2_qchem_sweep_plan_object", None)
        if compiled is None or components is None or basis is None:
            return None
        compiled_basis = getattr(compiled, "basis", None)
        if compiled_basis is not basis and not basis.compatible_with_layout(
            getattr(compiled_basis, "entries", basis.entries)
        ):
            return None
        component_dims = tuple(
            int(np.asarray(indices).size)
            for indices in self.component_basis.component_indices
        )
        builder = getattr(compiled, "build_component_parent_blocks", None)
        if builder is not None:
            object.__setattr__(
                self,
                "_component_parent_block_builder_backend",
                "packed_qchem_matches",
            )
            blocks = builder(components, component_dims)
            try:
                from . import su2_qchem_plan as qchem_plan_mod
            except Exception:
                qchem_plan_mod = None
            if (
                qchem_plan_mod is not None
                and getattr(qchem_plan_mod, "_DEBUG_PACKED_COMPILED_TERMS", False)
                and hasattr(compiled, "in_indices")
            ):
                matches = (
                    compiled.in_indices,
                    compiled.out_indices,
                    compiled.left_indices,
                    compiled.right_indices,
                )
                legacy = qchem_plan._compile_factorized_terms_from_matches(
                    basis,
                    matches,
                )
                legacy_blocks = self._build_component_parent_blocks_from_compiled(
                    legacy,
                    components,
                    basis,
                    component_dims,
                )
                residual = _component_block_residual(blocks, legacy_blocks)
                if residual > 1.0e-10:
                    raise RuntimeError(
                        "Packed SU2 qchem parent blocks disagree with legacy "
                        f"parent blocks: residual={residual:.3e}"
                    )
            return blocks
        if qchem_plan is None:
            return None
        object.__setattr__(
            self,
            "_component_parent_block_builder_backend",
            "packed_qchem_python",
        )
        return qchem_plan.build_component_parent_blocks(
            basis,
            components,
            component_dims,
            use_matches=False,
        )

    @staticmethod
    def _build_component_parent_blocks_from_compiled(
        compiled,
        components,
        basis,
        component_dims,
    ):
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
        blocks = {}
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
                key = (int(in_comp), int(out_comp))
                block = blocks.get(key)
                if block is None:
                    block = np.zeros(
                        (
                            int(component_dims[int(out_comp)]),
                            int(component_dims[int(in_comp)]),
                        ),
                        dtype=complex,
                    )
                    blocks[key] = block
                kernel = term.kernel_matrix(
                    term.input_entry.shape,
                    max_elements=max(
                        int(term.input_entry.size) * int(term.output_entry.size),
                        1,
                    ),
                )
                if kernel is None:
                    return None
                block[out_slice, in_slice] += np.asarray(kernel, dtype=complex)
        return tuple(
            (in_comp, out_comp, np.ascontiguousarray(block))
            for (in_comp, out_comp), block in sorted(blocks.items())
        )

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
        object.__setattr__(self, "_component_parent_block_builder_backend", "python")
        if str(_SU2_KERNEL_BACKEND).lower().replace("-", "_") != "python":
            blocks, actual = _su2_build_component_parent_blocks(
                plan,
                component_dims,
                backend=_SU2_KERNEL_BACKEND,
            )
            object.__setattr__(
                self,
                "_component_parent_block_builder_backend",
                str(actual),
            )
            if actual != "python" or blocks is not None:
                return blocks
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
        """

        if parent_blocks is None:
            return None
        transforms = self.component_basis.component_transforms
        object.__setattr__(
            self,
            "_component_orthonormal_block_builder_backend",
            "python",
        )
        if str(_SU2_KERNEL_BACKEND).lower().replace("-", "_") != "python":
            blocks, actual = _su2_project_component_orthonormal_blocks(
                parent_blocks,
                transforms,
                _DIRECT_FACTORIZED_ORTHONORMAL_BLOCK_MAX_ELEMENTS,
                backend=_SU2_KERNEL_BACKEND,
            )
            object.__setattr__(
                self,
                "_component_orthonormal_block_builder_backend",
                str(actual),
            )
            if actual != "python" or blocks is not None:
                return blocks
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

    def _build_component_orthonormal_block_batches(self, orthonormal_blocks):
        """
        Group same-shape orthonormal component blocks for batched matvecs.
        """

        if orthonormal_blocks is None:
            return None
        groups = OrderedDict()
        singles = []
        for in_comp, out_comp, block in tuple(orthonormal_blocks):
            block = np.asarray(block, dtype=complex)
            key = tuple(int(dim) for dim in block.shape)
            groups.setdefault(key, []).append((int(in_comp), int(out_comp), block))
        batches = []
        for _shape, entries in groups.items():
            if len(entries) < _ORTHONORMAL_BLOCK_BATCH_MIN_ENTRIES:
                singles.extend(entries)
                continue
            blocks = np.ascontiguousarray(
                np.stack([entry[2] for entry in entries], axis=0)
            )
            in_slices = tuple(
                self.component_basis._orth_slice(int(entry[0])) for entry in entries
            )
            out_slices = tuple(
                self.component_basis._orth_slice(int(entry[1])) for entry in entries
            )
            batches.append((blocks, in_slices, out_slices))
        if not batches:
            return None
        return {
            "batches": tuple(batches),
            "singles": tuple(singles),
            "n_batches": int(len(batches)),
            "n_batched_blocks": int(sum(batch[0].shape[0] for batch in batches)),
            "n_single_blocks": int(len(singles)),
        }

    def _component_orthonormal_block_matvec(self, vector, orthonormal_blocks):
        """
        Apply transformed component blocks directly in orthonormal coordinates.
        """

        out = np.zeros(self.dim, dtype=complex)
        for in_comp, out_comp, block in orthonormal_blocks:
            in_slice = self.component_basis._orth_slice(int(in_comp))
            out_slice = self.component_basis._orth_slice(int(out_comp))
            out[out_slice] += block @ vector[in_slice]
        return out

    def _component_orthonormal_batched_block_matvec(self, vector, batch_table):
        """
        Apply transformed orthonormal blocks with grouped batched GEMVs.
        """

        out = np.zeros(self.dim, dtype=complex)
        for blocks, in_slices, out_slices in batch_table.get("batches", ()):
            inputs = np.ascontiguousarray(
                np.stack([vector[in_slice] for in_slice in in_slices], axis=0)
            )
            contribs = np.matmul(blocks, inputs[:, :, None])[:, :, 0]
            for out_slice, contrib in zip(out_slices, contribs):
                out[out_slice] += contrib
        for in_comp, out_comp, block in batch_table.get("singles", ()):
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

    def complementary_family_table_equivalence_residual(self, seed=0):
        """
        Compare family-table matvecs against the raw component-direct plan.

        :param seed: Random seed for the probe vector.
        :returns: Relative 2-norm residual, or ``None`` when either path is
            unavailable.
        """

        family_table = getattr(self, "_complementary_family_tensor_table", None)
        su2_action = getattr(self, "_su2_action", None)
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

        cpp_source_stats = dict(
            getattr(self, "_cpp_family_source_stats", None) or {}
        )
        complementary_families = getattr(
            self.compiled_factorized_terms,
            "complementary_operator_families",
            None,
        )
        complementary_metadata = (
            complementary_families.as_metadata()
            if hasattr(complementary_families, "as_metadata")
            else cpp_source_stats.get("complementary_operator_families")
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
            cpp_source_stats.get("complementary_payload_terms", 0)
            if complementary_payloads is None
            else complementary_payloads.get("numeric_payload_terms", 0)
        )
        family_names = tuple(
            getattr(self.compiled_factorized_terms, "family_names", ())
            or cpp_source_stats.get("family_names", ())
            or ()
        )
        family_term_counts = dict(
            getattr(self.compiled_factorized_terms, "family_term_counts", {})
            or cpp_source_stats.get("family_term_counts", {})
            or {}
        )
        su2_qchem_sweep_plan = getattr(
            self.compiled_factorized_terms,
            "su2_qchem_sweep_plan",
            cpp_source_stats.get("su2_qchem_sweep_plan"),
        )
        su2_qchem_factor_match_backend = getattr(
            self.compiled_factorized_terms,
            "su2_qchem_factor_match_backend",
            cpp_source_stats.get("su2_qchem_factor_match_backend"),
        )
        su2_qchem_factor_match_count = getattr(
            self.compiled_factorized_terms,
            "su2_qchem_factor_match_count",
            cpp_source_stats.get("su2_qchem_factor_match_count"),
        )
        family_table = getattr(self, "_complementary_family_tensor_table", None)
        su2_action = getattr(self, "_su2_action", None)
        cpp_davidson_table = getattr(self, "_cpp_davidson_table", None)
        orthonormal_block_batches = getattr(
            self,
            "_component_orthonormal_block_batches",
            None,
        ) or {}
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
            "cpp_block_table": bool(cpp_davidson_table is not None),
            "cpp_block_table_stats": (
                None
                if cpp_davidson_table is None
                else cpp_davidson_table.stats
            ),
            "component_orthonormal_block_batch_kernel": bool(
                orthonormal_block_batches
            ),
            "component_orthonormal_block_batch_groups": int(
                orthonormal_block_batches.get("n_batches", 0)
            ),
            "component_orthonormal_block_batched_terms": int(
                orthonormal_block_batches.get("n_batched_blocks", 0)
            ),
            "component_orthonormal_block_single_terms": int(
                orthonormal_block_batches.get("n_single_blocks", 0)
            ),
            "component_parent_block_builder_backend": str(
                getattr(self, "_component_parent_block_builder_backend", "python")
            ),
            "component_orthonormal_block_builder_backend": str(
                getattr(self, "_component_orthonormal_block_builder_backend", "python")
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
                (
                    getattr(self, "_cpp_family_source_stats", None)
                    if family_table is None
                    else family_table.stats
                )
            ),
            "su2_local_action": (
                None if su2_action is None else su2_action.stats
            ),
            "su2_reference_residual": getattr(
                self,
                "_su2_action_reference_residual",
                None,
            ),
            "su2_kernel_backend_actual": (
                "cpp"
                if cpp_davidson_table is not None
                else ("python" if su2_action is None else str(su2_action.backend))
            ),
            "packed_cpp_exclusive_owner": bool(
                getattr(self, "_packed_cpp_exclusive_owner", False)
            ),
            "cpp_owner_factor_route_status": dict(
                getattr(self, "_cpp_owner_factor_route_status", {}) or {}
            ),
            "complementary_family_table_source": (
                (
                    (getattr(self, "_cpp_family_source_stats", None) or {}).get(
                        "source"
                    )
                    if family_table is None
                    else str(family_table.source)
                )
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
                (
                    complementary_payloads is not None
                    and complementary_payloads.get("payload_backed", False)
                )
                or cpp_source_stats.get("complementary_payload_backed", False)
            ),
            "complementary_boundary_payloads": complementary_payloads,
            "complementary_payload_terms": int(complementary_payload_terms),
            "family_resolved_tensor_kernel": bool(family_names),
            "family_names": family_names,
            "family_term_counts": family_term_counts,
            "su2_qchem_sweep_plan": su2_qchem_sweep_plan,
            "su2_qchem_factor_match_backend": su2_qchem_factor_match_backend,
            "su2_qchem_factor_match_count": (
                None
                if su2_qchem_factor_match_count is None
                else int(su2_qchem_factor_match_count)
            ),
            "su2_qchem_cpp_factor_route_calls": int(
                getattr(
                    self.compiled_factorized_terms,
                    "cpp_factor_route_calls",
                    0,
                )
            ),
            "su2_qchem_cpp_factor_diagonal_calls": int(
                getattr(
                    self.compiled_factorized_terms,
                    "cpp_factor_diagonal_calls",
                    0,
                )
            ),
            "cpp_factor_route_projection": bool(
                getattr(self, "_cpp_factor_route_projection", False)
            ),
            "component_parent_block_elements": int(
                sum(
                    np.asarray(block).size
                    for _in_comp, _out_comp, block in (
                        getattr(self, "_component_parent_blocks", None) or ()
                    )
                )
            ),
            "estimated_component_parent_block_elements": (
                None
                if getattr(self, "_estimated_parent_block_elements", None) is None
                else int(self._estimated_parent_block_elements)
            ),
            "oversized_parent_block_fallback": bool(
                getattr(self, "_oversized_parent_block_fallback", False)
            ),
            "oversized_parent_block_family_fallback": bool(
                getattr(self, "_oversized_parent_block_family_fallback", False)
            ),
            "component_orthonormal_block_elements": orthonormal_block_elements,
            "component_orthonormal_dense_elements": orthonormal_dense_elements,
            "build_timing": {
                str(key): float(value)
                for key, value in (getattr(self, "_build_timing", None) or {}).items()
            },
        }


@dataclass(frozen=True)
class DiagonalMetricBlock:
    """Compact diagonal local metric block."""

    diagonal: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "diagonal",
            np.ascontiguousarray(self.diagonal, dtype=complex).reshape(-1),
        )

    @property
    def shape(self):
        dim = int(self.diagonal.size)
        return (dim, dim)

    @property
    def dtype(self):
        return self.diagonal.dtype

    @property
    def stored_elements(self):
        return int(self.diagonal.size)

    def __matmul__(self, value):
        array = np.asarray(value)
        scale = self.diagonal if array.ndim == 1 else self.diagonal[:, None]
        return scale * array

    def __array__(self, dtype=None, copy=None):
        out = np.diag(self.diagonal)
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return np.array(out, copy=True) if copy is not False else out


class _DiagonalMetricTransformTranspose:
    __slots__ = ("transform",)

    def __init__(self, transform):
        self.transform = transform

    @property
    def shape(self):
        rows, cols = self.transform.shape
        return (cols, rows)

    def __matmul__(self, value):
        array = np.asarray(value)
        selected = array[self.transform.rows]
        scale = (
            self.transform.values
            if selected.ndim == 1
            else self.transform.values[:, None]
        )
        return scale * selected

    def __array__(self, dtype=None, copy=None):
        out = np.asarray(self.transform).T
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return np.array(out, copy=True) if copy is not False else out


@dataclass(frozen=True)
class DiagonalMetricTransform:
    """Compact selected diagonal map from orthonormal to parent coordinates."""

    parent_dim: int
    rows: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        rows = np.ascontiguousarray(self.rows, dtype=np.int64).reshape(-1)
        values = np.ascontiguousarray(self.values, dtype=complex).reshape(-1)
        if rows.size != values.size:
            raise ValueError("Diagonal metric transform rows and values must align.")
        if np.any(rows < 0) or np.any(rows >= int(self.parent_dim)):
            raise ValueError("Diagonal metric transform row is out of bounds.")
        object.__setattr__(self, "parent_dim", int(self.parent_dim))
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_metric_diagonal(cls, diagonal, *, tol):
        diagonal = np.ascontiguousarray(diagonal, dtype=complex).reshape(-1)
        scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 0.0)
        if np.max(np.abs(np.imag(diagonal)), initial=0.0) > float(tol) * scale:
            return None
        real = np.real(diagonal)
        keep = real > max(float(tol), 1.0e-14)
        if not np.any(keep):
            return None
        rows = np.flatnonzero(keep)
        return cls(
            parent_dim=int(diagonal.size),
            rows=rows,
            values=1.0 / np.sqrt(real[keep]),
        )

    @property
    def shape(self):
        return (int(self.parent_dim), int(self.rows.size))

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def stored_elements(self):
        return int(self.rows.size + self.values.size)

    @property
    def T(self):
        return _DiagonalMetricTransformTranspose(self)

    def conj(self):
        return type(self)(
            parent_dim=self.parent_dim,
            rows=self.rows,
            values=self.values.conj(),
        )

    def __matmul__(self, value):
        array = np.asarray(value)
        shape = (self.parent_dim,) + tuple(array.shape[1:])
        out = np.zeros(shape, dtype=np.result_type(self.dtype, array.dtype))
        scale = self.values if array.ndim == 1 else self.values[:, None]
        out[self.rows] = scale * array
        return out

    def project_diagonal(self, parent_diagonal):
        parent_diagonal = np.asarray(parent_diagonal)
        return np.abs(self.values) ** 2 * parent_diagonal[self.rows]

    def __array__(self, dtype=None, copy=None):
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        out = np.zeros(self.shape, dtype=dtype)
        out[self.rows, np.arange(self.rows.size)] = self.values
        return np.array(out, copy=True) if copy is not False else out


@dataclass(frozen=True)
class KroneckerMetricBlock:
    """Compact ``left ⊗ I ⊗ I ⊗ right`` metric block."""

    left: np.ndarray
    right: np.ndarray
    phys_dims: tuple[int, int]

    def __post_init__(self):
        left = np.ascontiguousarray(self.left, dtype=complex)
        right = np.ascontiguousarray(self.right, dtype=complex)
        phys_dims = tuple(int(dim) for dim in self.phys_dims)
        if left.ndim != 2 or left.shape[0] != left.shape[1]:
            raise ValueError("Kronecker metric left factor must be square.")
        if right.ndim != 2 or right.shape[0] != right.shape[1]:
            raise ValueError("Kronecker metric right factor must be square.")
        if len(phys_dims) != 2 or any(dim <= 0 for dim in phys_dims):
            raise ValueError("Kronecker metric physical dimensions must be positive.")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "phys_dims", phys_dims)

    @property
    def tensor_shape(self):
        return (
            int(self.left.shape[0]),
            int(self.phys_dims[0]),
            int(self.phys_dims[1]),
            int(self.right.shape[0]),
        )

    @property
    def shape(self):
        dim = int(np.prod(self.tensor_shape, dtype=np.int64))
        return (dim, dim)

    @property
    def dtype(self):
        return np.result_type(self.left.dtype, self.right.dtype)

    @property
    def stored_elements(self):
        return int(self.left.size + self.right.size)

    def __matmul__(self, value):
        array = np.asarray(value)
        trailing = tuple(array.shape[1:])
        tensor = array.reshape(self.tensor_shape + trailing)
        if trailing:
            result = np.einsum(
                "lk,kbcr...,qr->lbcq...",
                self.left,
                tensor,
                self.right,
                optimize=True,
            )
        else:
            result = np.einsum(
                "lk,kbcr,qr->lbcq",
                self.left,
                tensor,
                self.right,
                optimize=True,
            )
        return result.reshape((self.shape[0],) + trailing)

    def __array__(self, dtype=None, copy=None):
        out = np.kron(
            np.kron(
                np.kron(self.left, np.eye(self.phys_dims[0])),
                np.eye(self.phys_dims[1]),
            ),
            self.right,
        )
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return np.array(out, copy=True) if copy is not False else out


class _KroneckerMetricTransformTranspose:
    __slots__ = ("transform",)

    def __init__(self, transform):
        self.transform = transform

    @property
    def shape(self):
        rows, cols = self.transform.shape
        return (cols, rows)

    def __matmul__(self, value):
        return self.transform.transpose_apply(value)

    def __array__(self, dtype=None, copy=None):
        out = np.asarray(self.transform).T
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return np.array(out, copy=True) if copy is not False else out


@dataclass(frozen=True)
class KroneckerMetricTransform:
    """Compact tensor-product orthonormalization map for one basis entry."""

    left: np.ndarray
    right: np.ndarray
    phys_dims: tuple[int, int]

    def __post_init__(self):
        left = np.ascontiguousarray(self.left, dtype=complex)
        right = np.ascontiguousarray(self.right, dtype=complex)
        phys_dims = tuple(int(dim) for dim in self.phys_dims)
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("Kronecker metric transform factors must be matrices.")
        if len(phys_dims) != 2 or any(dim <= 0 for dim in phys_dims):
            raise ValueError("Kronecker metric physical dimensions must be positive.")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "phys_dims", phys_dims)

    @property
    def parent_shape(self):
        return (
            int(self.left.shape[0]),
            int(self.phys_dims[0]),
            int(self.phys_dims[1]),
            int(self.right.shape[0]),
        )

    @property
    def orth_shape(self):
        return (
            int(self.left.shape[1]),
            int(self.phys_dims[0]),
            int(self.phys_dims[1]),
            int(self.right.shape[1]),
        )

    @property
    def shape(self):
        return (
            int(np.prod(self.parent_shape, dtype=np.int64)),
            int(np.prod(self.orth_shape, dtype=np.int64)),
        )

    @property
    def dtype(self):
        return np.result_type(self.left.dtype, self.right.dtype)

    @property
    def stored_elements(self):
        return int(self.left.size + self.right.size)

    @property
    def T(self):
        return _KroneckerMetricTransformTranspose(self)

    def conj(self):
        return type(self)(
            left=self.left.conj(),
            right=self.right.conj(),
            phys_dims=self.phys_dims,
        )

    def __matmul__(self, value):
        array = np.asarray(value)
        trailing = tuple(array.shape[1:])
        tensor = array.reshape(self.orth_shape + trailing)
        if trailing:
            result = np.einsum(
                "lk,kbcr...,qr->lbcq...",
                self.left,
                tensor,
                self.right,
                optimize=True,
            )
        else:
            result = np.einsum(
                "lk,kbcr,qr->lbcq",
                self.left,
                tensor,
                self.right,
                optimize=True,
            )
        return result.reshape((self.shape[0],) + trailing)

    def transpose_apply(self, value):
        array = np.asarray(value)
        trailing = tuple(array.shape[1:])
        tensor = array.reshape(self.parent_shape + trailing)
        if trailing:
            result = np.einsum(
                "kl,lbcq...,rq->kbcr...",
                self.left.T,
                tensor,
                self.right.T,
                optimize=True,
            )
        else:
            result = np.einsum(
                "kl,lbcq,rq->kbcr",
                self.left.T,
                tensor,
                self.right.T,
                optimize=True,
            )
        return result.reshape((self.shape[1],) + trailing)

    def project_diagonal(self, parent_diagonal):
        parent = np.asarray(parent_diagonal).reshape(self.parent_shape)
        projected = np.einsum(
            "lk,lbcq,qr->kbcr",
            np.abs(self.left) ** 2,
            parent,
            np.abs(self.right) ** 2,
            optimize=True,
        )
        return projected.reshape(self.shape[1])

    def __array__(self, dtype=None, copy=None):
        out = np.kron(
            np.kron(
                np.kron(self.left, np.eye(self.phys_dims[0])),
                np.eye(self.phys_dims[1]),
            ),
            self.right,
        )
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return np.array(out, copy=True) if copy is not False else out


@dataclass(frozen=True)
class FactorizedRouteMetricBlock:
    """Sparse sum of boundary-factor Kronecker routes for one component."""

    dim: int
    routes: tuple

    @property
    def shape(self):
        return (int(self.dim), int(self.dim))

    @property
    def dtype(self):
        return np.result_type(
            *(
                np.result_type(
                    np.asarray(route[4]).dtype,
                    np.asarray(route[5]).dtype,
                )
                for route in self.routes
            ),
            float,
        )

    @property
    def stored_elements(self):
        return int(
            sum(
                np.asarray(route[4]).size + np.asarray(route[5]).size
                for route in self.routes
            )
        )

    def __matmul__(self, value):
        array = np.asarray(value)
        trailing = tuple(array.shape[1:])
        out = np.zeros(
            (int(self.dim),) + trailing,
            dtype=np.result_type(self.dtype, array.dtype),
        )
        for in_slice, out_slice, in_shape, out_shape, left, right in self.routes:
            tensor = array[in_slice].reshape(tuple(in_shape) + trailing)
            if trailing:
                contribution = np.einsum(
                    "lk,kbcr...,qr->lbcq...",
                    left,
                    tensor,
                    right,
                    optimize=True,
                )
            else:
                contribution = np.einsum(
                    "lk,kbcr,qr->lbcq",
                    left,
                    tensor,
                    right,
                    optimize=True,
                )
            out[out_slice] += contribution.reshape(
                (int(np.prod(out_shape, dtype=np.int64)),) + trailing
            )
        return out

    def to_dense(self, *, dtype=None, order="C"):
        """Materialize the route metric in the requested memory order."""

        dtype = self.dtype if dtype is None else np.dtype(dtype)
        out = np.zeros(self.shape, dtype=dtype, order=str(order))
        for in_slice, out_slice, in_shape, out_shape, left, right in self.routes:
            kernel = np.kron(
                np.kron(
                    np.kron(left, np.eye(int(in_shape[1]), dtype=dtype)),
                    np.eye(int(in_shape[2]), dtype=dtype),
                ),
                right,
            )
            out[out_slice, in_slice] += kernel.reshape(
                int(np.prod(out_shape, dtype=np.int64)),
                int(np.prod(in_shape, dtype=np.int64)),
            )
        return out

    def __array__(self, dtype=None, copy=None):
        out = self.to_dense(dtype=dtype)
        return np.array(out, copy=True) if copy is not False else out


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

    def matmat(self, vectors):
        """Apply the transformed Hamiltonian to several column vectors."""

        vectors = np.asarray(vectors, dtype=complex)
        if vectors.ndim != 2 or int(vectors.shape[0]) != self.orthonormal_dim:
            raise ValueError(
                f"Expected a ({self.orthonormal_dim}, nvec) block, got {vectors.shape}."
            )
        matmat = getattr(self.block_table, "matmat", None)
        if callable(matmat):
            return np.asarray(matmat(vectors))
        return np.column_stack(
            [self.block_table.matvec(vectors[:, idx]) for idx in range(vectors.shape[1])]
        )

    def cpp_lanczos_expm_apply(self, vector, dt, *, krylov_dim, tol):
        """Apply the compiled complex Lanczos propagator to this block table."""
        table = getattr(self, "_cpp_tdvp_lanczos_table", None)
        if table is None:
            try:
                from pyqed.mps import cpp_davidson

                table_cls = getattr(cpp_davidson, "SU2FactorizedFamilyTable", None)
                if table_cls is None or not hasattr(table_cls, "lanczos_expm_apply"):
                    return None
                dim = int(self.orthonormal_dim)
                transform = (
                    "diagonal",
                    0,
                    dim,
                    np.arange(dim, dtype=np.int64),
                    np.ones(dim, dtype=complex),
                )
                if self.block_table.dense_matrix is not None:
                    kernels = (
                        (slice(0, dim), slice(0, dim), self.block_table.dense_matrix),
                    )
                else:
                    kernels = tuple(
                        (term.input_slice, term.output_slice, term.kernel)
                        for term in self.block_table.terms
                    )
                entries = tuple(
                    (
                        0,
                        0,
                        int(input_slice.start),
                        int(input_slice.stop - input_slice.start),
                        int(output_slice.start),
                        int(output_slice.stop - output_slice.start),
                        np.ascontiguousarray(kernel, dtype=complex),
                        np.ones((1, 1), dtype=complex),
                        (
                            1,
                            int(output_slice.stop - output_slice.start),
                            1,
                            1,
                            1,
                            1,
                        ),
                        (int(output_slice.stop - output_slice.start), 1, 1, 1),
                        (int(input_slice.stop - input_slice.start), 1, 1, 1),
                        int(output_slice.stop - output_slice.start),
                    )
                    for input_slice, output_slice, kernel in kernels
                )
                table = table_cls((transform,), entries, dim)
                object.__setattr__(self, "_cpp_tdvp_lanczos_table", table)
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
                return None
        try:
            return table.lanczos_expm_apply(
                np.ascontiguousarray(vector, dtype=complex),
                float(dt),
                int(krylov_dim),
                float(tol),
            )
        except RuntimeError:
            return None

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

        metric_storage = int(
            sum(
                (
                    int(block.stored_elements)
                    if hasattr(block, "stored_elements")
                    else int(np.asarray(block).size)
                )
                for block in self.metric_blocks
            )
        )
        transform_storage = int(
            sum(
                (
                    int(transform.stored_elements)
                    if hasattr(transform, "stored_elements")
                    else int(np.asarray(transform).size)
                )
                for transform in self.component_transforms
            )
        )
        return {
            "basis_kind": "metric_connected_components",
            "parent_dim": int(self.parent_dim),
            "orthonormal_dim": int(self.orthonormal_dim),
            "n_components": int(self.n_components),
            "max_component_parent_dim": int(self.max_component_parent_dim),
            "metric_storage_elements": metric_storage,
            "transform_storage_elements": transform_storage,
            "compact_diagonal_metric": bool(
                any(
                    isinstance(block, DiagonalMetricBlock)
                    for block in self.metric_blocks
                )
            ),
            "compact_factorized_metric": bool(
                any(
                    isinstance(block, KroneckerMetricBlock)
                    for block in self.metric_blocks
                )
            ),
            "compact_route_metric": bool(
                any(
                    isinstance(block, FactorizedRouteMetricBlock)
                    for block in self.metric_blocks
                )
            ),
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

    def dense_operator_matrix(self):
        """Return the materialized orthonormal matrix when already owned."""

        getter = getattr(self.block_table, "dense_operator_matrix", None)
        if getter is not None:
            return getter()
        return None

    def dense_operator_matrix(self):
        """Return the materialized orthonormal matrix when already owned."""

        getter = getattr(self.block_table, "dense_operator_matrix", None)
        if getter is not None:
            return getter()
        return None

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
    metric_factor_blocks: dict | None = None
    metric_factor_routes: tuple | None = None

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

    def matmat(self, vectors):
        """Apply the transformed Hamiltonian to several column vectors."""

        vectors = np.asarray(vectors, dtype=complex)
        if vectors.ndim != 2 or int(vectors.shape[0]) != self.orthonormal_dim:
            raise ValueError(
                f"Expected a ({self.orthonormal_dim}, nvec) block, got {vectors.shape}."
            )
        matmat = getattr(self.block_table, "matmat", None)
        if callable(matmat):
            return np.asarray(matmat(vectors))
        return np.column_stack(
            [self.block_table.matvec(vectors[:, idx]) for idx in range(vectors.shape[1])]
        )

    def dense_operator_matrix(self):
        """Return the materialized orthonormal matrix when already owned."""

        getter = getattr(self.block_table, "dense_operator_matrix", None)
        if getter is not None:
            return getter()
        return None


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
