#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-site effective block operators for non-Abelian DMRG.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import weakref

import numpy as np

from .local_operator import (
    CompiledLocalActions,
    apply_transition_reduced,
    apply_transition_tensor,
    compile_packed_transitions,
    diagonal_from_factorized_terms,
    build_identity_mpo_local_actions,
    identity_mpo_transitions,
    transitions_are_identity_operator,
)
from .mpo import MPOCore, IrreducibleMPO, RankCoupledMPO
from .solver import pack_two_site_state, unpack_two_site_state


def _su2_qchem_direct_parent_blocks_enabled():
    """Return whether the direct packed qchem parent-block route is active."""

    try:
        from .renormalized import get_direct_factorized_orthonormal_kernel_policy

        return bool(
            get_direct_factorized_orthonormal_kernel_policy().get(
                "su2_qchem_direct_parent_blocks",
                False,
            )
        )
    except Exception:
        return False


@dataclass
class RenormalizedLocalOperatorTableBuilder:
    """
    Build and cache typed local effective-H tables on boundary entries.

    :param operator: Effective two-site operator whose boundary entries own
        the compiled table.
    """

    operator: object

    @property
    def owner(self):
        """Return the boundary entry that owns the local table."""

        if self.operator.left_entry is not None:
            return self.operator.left_entry
        return self.operator.right_entry

    def basis_signature(self):
        """Return a hashable signature for the local two-site basis."""

        return tuple((entry.key, entry.shape, entry.size) for entry in self.operator.basis)

    def representation(self):
        """
        Return the local table representation selected for this operator.

        :returns: One of ``"identity"``, ``"transition"``,
            ``"factorized"``, ``"rank_coupled_factorized"``,
            ``"rank_coupled_contextual"``, or
            ``"rank_coupled_complementary"``.
        """

        from .environment import _FACTORIZED_PACKED_LOCAL_DIM, _is_identity_mpo_core

        if self.operator.rank_coupled:
            if (
                getattr(self.operator.basis, "channel_resolved", False)
                and all(
                    int(entry.shape[1]) == 1 and int(entry.shape[2]) == 1
                    for entry in self.operator.basis
                )
            ):
                return "rank_coupled_contextual"
            if self.operator.complementary_operator_families is not None:
                return "rank_coupled_complementary"
            return "rank_coupled_factorized"
        if _is_identity_mpo_core(self.operator.mpo_left) and _is_identity_mpo_core(
            self.operator.mpo_right
        ):
            return "identity"
        if self.operator.basis.size > _FACTORIZED_PACKED_LOCAL_DIM:
            return "factorized"
        return "transition"

    def key(self):
        """Return the cache key for this local operator table."""

        left_signature = getattr(self.operator.left_entry, "signature", None)
        right_signature = getattr(self.operator.right_entry, "signature", None)
        return (
            "renormalized_local_operator_table",
            self.representation(),
            bool(self.operator.rank_coupled),
            id(self.operator.mpo_left),
            id(self.operator.mpo_right),
            left_signature,
            right_signature,
            self.basis_signature(),
            str(np.dtype(self.operator.output_dtype())),
        )

    def get(self):
        """
        Return a cached typed table from the owning boundary entry.

        :returns: Cached table or ``None``.
        """

        owner = self.owner
        if owner is None:
            return None
        return owner.get_local_operator_table(self.key())

    def build(self):
        """
        Build or reuse the typed local table.

        :returns: ``RenormalizedLocalOperatorTable`` or ``None`` when no
            boundary owner is available.
        """

        timing = {}
        owner = self.owner
        if owner is None:
            return None
        t0 = time.perf_counter()
        representation = self.representation()
        timing["local_table_representation"] = time.perf_counter() - t0
        from .environment import _is_identity_mpo_core

        identity_mpo = (
            (
                _is_identity_mpo_core(self.operator.mpo_left)
                or bool(
                    getattr(
                        self.operator.mpo_left,
                        "fully_reduced_identity",
                        False,
                    )
                )
            )
            and (
                _is_identity_mpo_core(self.operator.mpo_right)
                or bool(
                    getattr(
                        self.operator.mpo_right,
                        "fully_reduced_identity",
                        False,
                    )
                )
            )
        )
        require_symbolic_payloads = representation in {
            "transition",
            "factorized",
            "rank_coupled_factorized",
            "rank_coupled_contextual",
            "rank_coupled_complementary",
        } and not identity_mpo
        t0 = time.perf_counter()
        require_eager_factor_tables = representation == "factorized"
        self.operator.ensure_side_operator_tables(
            representation,
            require_factor_tables=require_eager_factor_tables,
            require_symbolic_payloads=require_symbolic_payloads,
        )
        timing["local_table_side_tables"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cached = owner.get_local_operator_table(self.key())
        timing["local_table_cache_lookup"] = time.perf_counter() - t0
        if cached is not None:
            self.operator._attach_table_metadata(cached)
            actions = cached.actions
            if actions.metadata is not None:
                build_timing = actions.metadata.setdefault(
                    "renormalized_operator_build_timing",
                    {},
                )
                for key, value in timing.items():
                    build_timing[key] = build_timing.get(key, 0.0) + float(value)
            return cached
        t0 = time.perf_counter()
        actions = self.operator._compile_actions_uncached(
            representation,
            require_symbolic_payloads=require_symbolic_payloads,
        )
        timing["local_table_compile_actions"] = time.perf_counter() - t0
        if actions.metadata is not None:
            build_timing = actions.metadata.setdefault(
                "renormalized_operator_build_timing",
                {},
            )
            for key, value in timing.items():
                build_timing[key] = build_timing.get(key, 0.0) + float(value)
        t0 = time.perf_counter()
        table = owner.put_local_operator_table(
            self.key(),
            actions,
            representation=representation,
            basis_size=self.operator.basis.size,
        )
        if actions.metadata is not None:
            actions.metadata["renormalized_operator_build_timing"][
                "local_table_put"
            ] = time.perf_counter() - t0
        actions.local_operator_table = weakref.proxy(table)
        self.operator._attach_table_metadata(table)
        return table


@dataclass
class EffectiveBlockOperator:
    """
    Two-site effective operator assembled from explicit renormalized blocks.

    :param left_block: Left renormalized environment block.
    :param mpo_left: MPOCore core on the left active site.
    :param mpo_right: MPOCore core on the right active site.
    :param right_block: Right renormalized environment block.
    :param two_site_template: Rank-4 two-site tensor defining local sectors.
    :param basis: Explicit local two-site basis.
    :param phys1_slices: Physical sector slices for the left active site.
    :param phys2_slices: Physical sector slices for the right active site.
    :param rank_coupled: Whether the MPOCore/environment path is rank-coupled.
    :param left_entry: Optional persisted left boundary-stack entry.
    :param right_entry: Optional persisted right boundary-stack entry.
    :param su2_operator_engine: Persistent owner for packed SU(2) factors,
        structural schedules, and local-action plans.
    :param su2_moving_environment: Persistent C++ sweep owner for decoded
        factor routes and reusable numerical workspaces.
    :param name: Optional local operator name.
    """

    left_block: object
    mpo_left: object
    mpo_right: object
    right_block: object
    two_site_template: object
    basis: object
    phys1_slices: object | None = None
    phys2_slices: object | None = None
    rank_coupled: bool = False
    left_entry: object | None = None
    right_entry: object | None = None
    local_operator_table: object | None = None
    complementary_operator_families: object | None = None
    su2_operator_engine: object | None = None
    su2_moving_environment: object | None = None
    name: str | None = None

    def boundary_metadata(self):
        """
        Return block2-like renormalized-boundary source metadata.

        :returns: Dictionary describing the persisted left/right entries used
            to assemble this effective local operator.
        """

        def _entry_stats(entry):
            if entry is None:
                return None
            stats = getattr(entry, "stats", None)
            return dict(stats) if stats is not None else None

        return {
            "renormalized_boundary_source": "block_stack",
            "left_boundary": _entry_stats(self.left_entry),
            "right_boundary": _entry_stats(self.right_entry),
        }

    def symbolic_boundary_metadata(self):
        """
        Return symbolic boundary-payload metadata for the local operator.

        :returns: Dictionary describing whether local actions were compiled
            from symbolic-owned boundary payloads.
        """

        def _symbolic_stats(entry):
            table = None if entry is None else getattr(entry, "symbolic_operator_table", None)
            return None if table is None else dict(table.stats)

        left_stats = _symbolic_stats(self.left_entry)
        right_stats = _symbolic_stats(self.right_entry)
        return {
            "symbolic_boundary_payload_source": (
                "symbolic_table"
                if left_stats is not None or right_stats is not None
                else "raw_boundary_map"
            ),
            "left_symbolic_boundary": left_stats,
            "right_symbolic_boundary": right_stats,
            "symbolic_numeric_payloads": int(
                (0 if left_stats is None else left_stats.get("numeric_payloads", 0))
                + (0 if right_stats is None else right_stats.get("numeric_payloads", 0))
            ),
            "symbolic_payloads_owned": bool(
                (left_stats is not None and left_stats.get("owns_numeric_payloads", False))
                or (right_stats is not None and right_stats.get("owns_numeric_payloads", False))
            ),
            "complementary_boundary_payloads": (
                self.complementary_boundary_payload_metadata()
                if self.complementary_operator_families is not None
                else None
            ),
        }

    def _attach_table_metadata(self, table):
        actions = table.actions
        if actions.metadata is not None:
            actions.metadata["renormalized_local_operator_table"] = table.stats
            actions.metadata["symbolic_boundary_payloads"] = self.symbolic_boundary_metadata()
            if self.complementary_operator_families is not None:
                actions.metadata["complementary_operator_families"] = (
                    self.complementary_operator_family_metadata()
                )
                actions.metadata["complementary_boundary_payloads"] = (
                    self.complementary_boundary_payload_metadata()
                )
        table_proxy = (
            table
            if type(table) in weakref.ProxyTypes
            else weakref.proxy(table)
        )
        actions.local_operator_table = table_proxy
        self.local_operator_table = table_proxy
        return table

    def complementary_operator_family_metadata(self):
        """
        Return complementary-family metadata attached to this local problem.

        :returns: Dictionary metadata or ``None``.
        """

        families = self.complementary_operator_families
        if families is None:
            return None
        if hasattr(families, "as_metadata"):
            return families.as_metadata()
        return {"enabled": True, "type": type(families).__name__}

    def complementary_boundary_payload_metadata(self):
        """
        Return numeric complementary payload metadata for local boundaries.

        :returns: Dictionary describing left/right ``S/R/A/P/B/Q`` payloads.
        """

        def _payload_entry(entry):
            comp_entry = (
                None
                if entry is None
                else getattr(entry, "complementary_operator_entry", None)
            )
            if comp_entry is None:
                return None
            return dict(comp_entry.stats)

        left_payload = _payload_entry(self.left_entry)
        right_payload = _payload_entry(self.right_entry)
        family_tables = tuple(
            table
            for table in (
                None
                if left_payload is None
                else left_payload.get("family_operator_table"),
                None
                if right_payload is None
                else right_payload.get("family_operator_table"),
            )
            if table is not None
        )
        total_terms = int(
            (0 if left_payload is None else left_payload.get("numeric_payload_terms", 0))
            + (0 if right_payload is None else right_payload.get("numeric_payload_terms", 0))
        )
        cross_terms = int(
            (
                0
                if left_payload is None
                else left_payload.get("numeric_payload_cross_terms", 0)
            )
            + (
                0
                if right_payload is None
                else right_payload.get("numeric_payload_cross_terms", 0)
            )
        )
        return {
            "payload_backed": bool(total_terms > 0),
            "family_operator_table_backed": bool(family_tables),
            "family_operator_tables": int(len(family_tables)),
            "family_operator_table_payload_blocks": int(
                sum(table.get("n_payload_blocks", 0) for table in family_tables)
            ),
            "family_operator_table_stored_elements": int(
                sum(table.get("stored_elements", 0) for table in family_tables)
            ),
            "family_operator_table_symbolic_terms": int(
                sum(table.get("symbolic_terms", 0) for table in family_tables)
            ),
            "numeric_payload_terms": int(total_terms),
            "numeric_payload_cross_terms": int(cross_terms),
            "left_boundary": left_payload,
            "right_boundary": right_payload,
        }

    def complementary_boundary_payload_signature(self):
        """
        Return a hashable signature for complementary payload-backed caches.

        :returns: Tuple summarizing boundary payload identity and term counts.
        """

        def _entry_signature(entry):
            comp_entry = (
                None
                if entry is None
                else getattr(entry, "complementary_operator_entry", None)
            )
            if comp_entry is None:
                return None
            return (
                comp_entry.key,
                tuple(comp_entry.family_names),
                (
                    None
                    if comp_entry.family_operator_table is None
                    else (
                        comp_entry.family_operator_table.family_names,
                        comp_entry.family_operator_table.active_family_names,
                        int(comp_entry.family_operator_table.n_payload_blocks),
                        int(comp_entry.family_operator_table.stored_elements),
                        int(comp_entry.family_operator_table.symbolic_terms),
                    )
                ),
                int(
                    sum(
                        payload.n_terms
                        for payload in comp_entry.family_payloads.values()
                    )
                ),
                int(
                    sum(
                        payload.cross_terms
                        for payload in comp_entry.family_payloads.values()
                    )
                ),
                tuple(
                    (
                        str(name),
                        int(payload.n_terms),
                        int(payload.cross_terms),
                        float(payload.coefficient_norm),
                    )
                    for name, payload in sorted(comp_entry.family_payloads.items())
                ),
            )

        return (_entry_signature(self.left_entry), _entry_signature(self.right_entry))

    def complementary_family_operator_table_objects(self):
        """
        Return live family-operator tables attached to the local boundaries.

        :returns: Tuple of stored family-resolved renormalized operator tables.
        """

        tables = []
        for entry in (self.left_entry, self.right_entry):
            comp_entry = (
                None
                if entry is None
                else getattr(entry, "complementary_operator_entry", None)
            )
            table = (
                None
                if comp_entry is None
                else getattr(comp_entry, "family_operator_table", None)
            )
            if table is not None:
                tables.append(table)
        return tuple(tables)

    def _annotate_symbolic_payload_source(self, packed_apply, compiled_terms=None):
        """
        Attach symbolic boundary-payload metadata to a packed matvec.

        :param packed_apply: Packed matvec callable.
        :param compiled_terms: Optional compiled factorized term provider.
        :returns: ``packed_apply``.
        """

        metadata = self.symbolic_boundary_metadata()
        packed_apply.symbolic_boundary_payloads = metadata
        if compiled_terms is not None:
            compiled_terms.symbolic_boundary_payloads = metadata
        return packed_apply

    def _annotate_complementary_payload_source(self, packed_apply, compiled_terms=None):
        """
        Attach numeric complementary-boundary payload metadata to a matvec.

        :param packed_apply: Packed matvec callable.
        :param compiled_terms: Optional compiled factorized term provider.
        :returns: ``packed_apply``.
        """

        metadata = self.complementary_boundary_payload_metadata()
        signature = self.complementary_boundary_payload_signature()
        table_objects = self.complementary_family_operator_table_objects()
        packed_apply.complementary_boundary_payloads = metadata
        packed_apply.complementary_payload_signature = signature
        packed_apply.complementary_payload_backed = bool(
            metadata.get("payload_backed", False)
        )
        packed_apply.complementary_family_operator_tables = metadata
        packed_apply.complementary_family_operator_table_objects = table_objects
        if compiled_terms is not None:
            compiled_terms.complementary_boundary_payloads = metadata
            compiled_terms.complementary_payload_signature = signature
            compiled_terms.complementary_payload_backed = bool(
                metadata.get("payload_backed", False)
            )
            compiled_terms.complementary_family_operator_tables = metadata
            compiled_terms.complementary_family_operator_table_objects = table_objects
        return packed_apply

    def _side_table_key(self, side, representation):
        """Return the cache key for one boundary side-table representation."""

        entry = self.left_entry if side == "left" else self.right_entry
        return (
            "side_operator_table",
            representation,
            None if entry is None else getattr(entry, "signature", None),
        )

    def _side_table_record(
        self,
        side,
        representation,
        *,
        require_existing=False,
        require_symbolic_payloads=False,
        source="lazy",
    ):
        """
        Return a cached side-table record for one side.

        :param side: ``"left"`` or ``"right"``.
        :param representation: Side-table representation.
        :param require_existing: Require a previously prepared table.
        :param require_symbolic_payloads: Require the table to come from a
            symbolic boundary table that owns numeric payloads.
        :param source: Source label for a lazily created table.
        :returns: :class:`RenormalizedSideOperatorTable` or ``None``.
        """

        entry = self.left_entry if side == "left" else self.right_entry
        block_map = self.left_block if side == "left" else self.right_block
        if entry is None:
            return None
        key = self._side_table_key(side, representation)
        table = entry.get_side_operator_table(key)
        if table is None:
            if require_existing:
                raise RuntimeError(
                    f"Missing prepared {representation!r} side table for "
                    f"{side} boundary {entry.bond}."
                )
            symbolic_table = self._symbolic_table(
                entry,
                side,
                require_payloads=require_symbolic_payloads,
            )
            packed_table = None
            if symbolic_table is not None:
                if representation == "rank_coupled_by_ket":
                    try:
                        from .su2_qchem_plan import (
                            pack_rank_coupled_boundary_table_from_block_map,
                            pack_rank_coupled_boundary_table_from_payloads,
                        )

                        if getattr(symbolic_table, "numeric_payloads", None):
                            packed_table = pack_rank_coupled_boundary_table_from_payloads(
                                symbolic_table.numeric_payloads,
                                active_channels=getattr(symbolic_table, "channels", None),
                                side=side,
                                bond=getattr(entry, "bond", 0),
                                representation=representation,
                            )
                        if packed_table is None:
                            packed_table = pack_rank_coupled_boundary_table_from_block_map(
                                block_map,
                                active_channels=getattr(symbolic_table, "channels", None),
                                side=side,
                                bond=getattr(entry, "bond", 0),
                                representation=representation,
                            )
                    except Exception:
                        packed_table = None
                grouped = (
                    None
                    if packed_table is not None
                    else symbolic_table.group_boundary_blocks(representation=representation)
                )
                source = "symbolic_" + str(source)
            elif require_symbolic_payloads:
                raise RuntimeError(
                    f"Missing symbolic-owned payload table for {side} boundary "
                    f"{entry.bond}."
                )
            else:
                if representation == "rank_coupled_by_ket":
                    try:
                        from .su2_qchem_plan import pack_rank_coupled_boundary_table_from_block_map

                        packed_table = pack_rank_coupled_boundary_table_from_block_map(
                            block_map,
                            side=side,
                            bond=getattr(entry, "bond", 0),
                            representation=representation,
                        )
                    except Exception:
                        packed_table = None
                grouped = (
                    None
                    if packed_table is not None
                    else self._group_side_blocks(block_map, representation)
                )
            table = entry.put_side_operator_table(
                key,
                grouped,
                representation=representation,
                source=source,
                packed_table=packed_table,
            )
        self._require_symbolic_side_record(table, side, representation, require_symbolic_payloads)
        return table

    def _side_table(
        self,
        side,
        representation,
        *,
        require_existing=False,
        require_symbolic_payloads=False,
        source="lazy",
    ):
        """
        Return grouped boundary blocks for one side, using the boundary cache.

        :param side: ``"left"`` or ``"right"``.
        :param representation: Side-table representation.
        :param require_existing: Require a previously prepared table.
        :param require_symbolic_payloads: Require a symbolic-owned payload
            source and reject raw boundary-map fallback.
        :param source: Source label for a lazily created table.
        :returns: Grouped blocks by ket sector.
        """

        entry = self.left_entry if side == "left" else self.right_entry
        block_map = self.left_block if side == "left" else self.right_block
        if entry is None:
            return self._group_side_blocks(block_map, representation)
        table = self._side_table_record(
            side,
            representation,
            require_existing=require_existing,
            require_symbolic_payloads=require_symbolic_payloads,
            source=source,
        )
        if hasattr(table, "grouped_payload"):
            return table.grouped_payload()
        return table.grouped_by_ket

    def _factor_side_table(
        self,
        side,
        representation,
        *,
        require_existing=False,
        require_symbolic_payloads=False,
        source="lazy",
    ):
        entry = self.left_entry if side == "left" else self.right_entry
        if entry is None:
            return None
        record = self._factor_side_table_record(
            side,
            representation,
            require_existing=require_existing,
            require_symbolic_payloads=require_symbolic_payloads,
            source=source,
        )
        if hasattr(record, "grouped_payload"):
            return record.grouped_payload()
        return record.grouped_by_ket

    def _factor_side_table_record(
        self,
        side,
        representation,
        *,
        require_existing=False,
        require_symbolic_payloads=False,
        source="lazy",
    ):
        """
        Return a cached factor side-table record for one boundary.

        :param side: ``"left"`` or ``"right"``.
        :param representation: Factor-table representation.
        :param require_existing: Require a previously prepared table.
        :param require_symbolic_payloads: Require a symbolic-owned payload
            source and reject raw boundary-map fallback.
        :param source: Source label for a lazily created table.
        :returns: :class:`RenormalizedSideOperatorTable`.
        """

        entry = self.left_entry if side == "left" else self.right_entry
        if entry is None:
            raise RuntimeError(f"Cannot build {side} factor side table without a boundary entry.")
        W = self.mpo_left if side == "left" else self.mpo_right
        phys_slices = self.phys1_slices if side == "left" else self.phys2_slices
        base_record = None
        key = (
            "side_operator_table",
            representation,
            getattr(entry, "signature", None),
            id(W),
        )
        table = entry.get_side_operator_table(key)
        if table is not None:
            self._require_symbolic_side_record(
                table,
                side,
                representation,
                require_symbolic_payloads,
            )
            return table
        if require_existing:
            raise RuntimeError(
                f"Missing prepared {representation!r} factor side table for "
                f"{side} boundary {entry.bond}."
            )
        symbolic_table = self._symbolic_table(
            entry,
            side,
            require_payloads=require_symbolic_payloads,
        )

        packed_table = None
        if symbolic_table is not None:
            if representation == "left_factor_by_ket":
                base_record = self._side_table_record(
                    "left",
                    "array_by_ket",
                    require_symbolic_payloads=require_symbolic_payloads,
                    source=source,
                )
            elif representation == "right_factor_by_ket":
                base_record = self._side_table_record(
                    "right",
                    "array_by_ket",
                    require_symbolic_payloads=require_symbolic_payloads,
                    source=source,
                )
            elif representation == "rank_coupled_left_factor_by_ket":
                base_record = self._side_table_record(
                    "left",
                    "rank_coupled_by_ket",
                    require_symbolic_payloads=require_symbolic_payloads,
                    source=source,
                )
            elif representation == "rank_coupled_right_factor_by_ket":
                base_record = self._side_table_record(
                    "right",
                    "rank_coupled_by_ket",
                    require_symbolic_payloads=require_symbolic_payloads,
                    source=source,
                )
            else:
                raise ValueError(f"Unknown factor side-table representation {representation!r}.")
            grouped = None
            if str(representation).startswith("rank_coupled_") and base_record is not None:
                try:
                    engine = self.su2_operator_engine
                    if engine is not None:
                        packed_table = engine.factor_table(
                            getattr(base_record, "packed_table", None),
                            W,
                            side=side,
                            bond=getattr(entry, "bond", 0),
                            representation=representation,
                        )
                    else:
                        from .su2_qchem_plan import (
                            pack_rank_coupled_factor_table_from_boundary,
                        )

                        packed_table = pack_rank_coupled_factor_table_from_boundary(
                            getattr(base_record, "packed_table", None),
                            W,
                            side=side,
                            bond=getattr(entry, "bond", 0),
                            representation=representation,
                        )
                except Exception:
                    packed_table = None
            if packed_table is None:
                grouped = self._symbolic_factor_table(
                    symbolic_table,
                    representation,
                    W,
                    phys_slices=phys_slices,
                )
        elif require_symbolic_payloads:
            raise RuntimeError(
                f"Missing symbolic-owned factor payloads for {side} boundary "
                f"{entry.bond}."
            )
        elif representation == "left_factor_by_ket":
            from .renormalized import build_left_factor_table

            base_record = self._side_table_record("left", "array_by_ket", source=source)
            grouped = build_left_factor_table(
                base_record.grouped_by_ket,
                W,
                phys_slices,
            )
        elif representation == "right_factor_by_ket":
            from .renormalized import build_right_factor_table

            base_record = self._side_table_record("right", "array_by_ket", source=source)
            grouped = build_right_factor_table(
                base_record.grouped_by_ket,
                W,
                phys_slices,
            )
        elif representation == "rank_coupled_left_factor_by_ket":
            from .renormalized import build_rank_coupled_left_factor_table

            base_record = self._side_table_record("left", "rank_coupled_by_ket", source=source)
            grouped = build_rank_coupled_left_factor_table(
                base_record.grouped_by_ket,
                W,
            )
        elif representation == "rank_coupled_right_factor_by_ket":
            from .renormalized import build_rank_coupled_right_factor_table

            base_record = self._side_table_record("right", "rank_coupled_by_ket", source=source)
            grouped = build_rank_coupled_right_factor_table(
                base_record.grouped_by_ket,
                W,
            )
        else:
            raise ValueError(f"Unknown factor side-table representation {representation!r}.")
        table_source = source
        if base_record is not None and str(base_record.source).startswith("symbolic_"):
            table_source = "symbolic_" + str(source)
        table = entry.put_side_operator_table(
            key,
            grouped,
            representation=representation,
            source=table_source,
            parent_table=base_record,
            packed_table=packed_table,
        )
        self._require_symbolic_side_record(table, side, representation, require_symbolic_payloads)
        return table

    def _symbolic_table(self, entry, side, *, require_payloads=False):
        """
        Return a symbolic boundary table and optionally require owned payloads.

        :param entry: Renormalized boundary entry.
        :param side: Boundary side used in diagnostics.
        :param require_payloads: Require non-empty numeric payload ownership.
        :returns: Symbolic boundary table or ``None``.
        """

        symbolic_table = getattr(entry, "symbolic_operator_table", None)
        if symbolic_table is None:
            return None
        if require_payloads and not symbolic_table.stats.get("owns_numeric_payloads", False):
            raise RuntimeError(
                f"Symbolic table for {side} boundary {entry.bond} does not own "
                "numeric payloads."
            )
        return symbolic_table

    def _symbolic_factor_table(self, symbolic_table, representation, W, *, phys_slices=None):
        """
        Build a factor table through the symbolic boundary-table API.

        :param symbolic_table: Symbolic boundary table owning numeric payloads.
        :param representation: Factor-table representation.
        :param W: Adjacent MPOCore core.
        :param phys_slices: Optional physical-sector slices.
        :returns: Factor table grouped by ket-sector pair.
        """

        if representation == "left_factor_by_ket":
            return symbolic_table.left_factor_table(W, phys_slices=phys_slices)
        if representation == "right_factor_by_ket":
            return symbolic_table.right_factor_table(W, phys_slices=phys_slices)
        if representation == "rank_coupled_left_factor_by_ket":
            return symbolic_table.rank_coupled_left_factor_table(W)
        if representation == "rank_coupled_right_factor_by_ket":
            return symbolic_table.rank_coupled_right_factor_table(W)
        return symbolic_table.factor_boundary_blocks(
            representation,
            W,
            phys_slices=phys_slices,
        )

    def _require_symbolic_side_record(self, table, side, representation, required):
        """
        Validate that a side table was sourced from symbolic-owned payloads.

        :param table: Side table record to validate.
        :param side: Boundary side used in diagnostics.
        :param representation: Required representation.
        :param required: Whether validation is active.
        :returns: ``None``.
        """

        if not required:
            return
        if not str(table.source).startswith("symbolic_"):
            raise RuntimeError(
                f"{side} {representation!r} side table came from {table.source!r}, "
                "not symbolic-owned payloads."
            )

    def _group_side_blocks(self, block_map, representation):
        from .environment import _group_boundary_blocks_by_ket

        return _group_boundary_blocks_by_ket(block_map, representation)

    def ensure_side_operator_tables(
        self,
        representation,
        *,
        require_factor_tables=False,
        require_symbolic_payloads=False,
        source="lazy",
    ):
        """
        Ensure side-table providers exist for a local table representation.

        :param representation: Local table representation.
        :param require_factor_tables: Require pre-existing factor tables.
        :param require_symbolic_payloads: Require side/factor tables to come
            from symbolic-owned payloads.
        :param source: Source label for created tables.
        :returns: ``None``.
        """

        if representation in {
            "rank_coupled_factorized",
            "rank_coupled_contextual",
            "rank_coupled_complementary",
        }:
            side_representation = "rank_coupled_by_ket"
        elif representation == "factorized":
            side_representation = "array_by_ket"
        elif representation == "transition":
            side_representation = "block_by_ket"
        else:
            return
        self._side_table(
            "left",
            side_representation,
            require_existing=require_factor_tables,
            require_symbolic_payloads=require_symbolic_payloads,
            source=source,
        )
        self._side_table(
            "right",
            side_representation,
            require_existing=require_factor_tables,
            require_symbolic_payloads=require_symbolic_payloads,
            source=source,
        )
        if representation in {"rank_coupled_factorized", "rank_coupled_complementary"}:
            self._factor_side_table_record(
                "left",
                "rank_coupled_left_factor_by_ket",
                require_existing=require_factor_tables,
                require_symbolic_payloads=require_symbolic_payloads,
                source=source,
            )
            self._factor_side_table_record(
                "right",
                "rank_coupled_right_factor_by_ket",
                require_existing=require_factor_tables,
                require_symbolic_payloads=require_symbolic_payloads,
                source=source,
            )
        elif representation == "factorized":
            self._factor_side_table_record(
                "left",
                "left_factor_by_ket",
                require_existing=require_factor_tables,
                require_symbolic_payloads=require_symbolic_payloads,
                source=source,
            )
            self._factor_side_table_record(
                "right",
                "right_factor_by_ket",
                require_existing=require_factor_tables,
                require_symbolic_payloads=require_symbolic_payloads,
                source=source,
            )

    def prepare_local_side_operator_tables(self, representation, *, require_symbolic_payloads=False):
        """
        Prepare side-factor tables needed by a local table representation.

        :param representation: Local table representation.
        :param require_symbolic_payloads: Require side/factor tables to come
            from symbolic-owned payloads.
        :returns: ``None``.
        """

        self.ensure_side_operator_tables(
            representation,
            require_factor_tables=False,
            require_symbolic_payloads=require_symbolic_payloads,
            source="prepared",
        )

    def output_dtype(self):
        """
        Infer the scalar dtype for this local effective operator.

        :returns: NumPy result dtype from MPOCore cores and the two-site template.
        """

        from .environment import _mpo_dtype

        return np.result_type(
            _mpo_dtype(self.mpo_left),
            _mpo_dtype(self.mpo_right),
            *(np.asarray(block).dtype for block in self.two_site_template.data.values()),
        )

    def _physical_diagonal_slice(self, mpo_core, sector, phys_slices):
        if isinstance(mpo_core, (MPOCore, IrreducibleMPO, RankCoupledMPO)):
            return mpo_core.block(sector, sector)
        if phys_slices is None:
            return None
        phys_slice = phys_slices.get(sector)
        return None if phys_slice is None else np.asarray(mpo_core[:, :, phys_slice, phys_slice])

    def diagonal(self, *, factorized_terms=None, require_symbolic_payloads=False):
        """
        Build the packed diagonal for this two-site effective operator.

        :param factorized_terms: Optional uncompiled rank-coupled factorized
            terms used by rank-coupled local operators.
        :param require_symbolic_payloads: For rank-coupled operators, require
            the caller to provide factorized terms built from prepared symbolic
            side/factor tables.
        :returns: Packed real diagonal aligned with ``basis``.
        """

        from .environment import (
            _is_identity_mpo_core,
            _precompute_two_site_rank_coupled_factorized_terms,
        )

        out_dtype = self.output_dtype()
        if self.rank_coupled:
            if factorized_terms is None:
                if require_symbolic_payloads:
                    raise RuntimeError(
                        "Strict symbolic rank-coupled diagonals require "
                        "factorized_terms from the local factor-table build."
                    )
                _out_entries, factorized_terms = _precompute_two_site_rank_coupled_factorized_terms(
                    self.left_block,
                    self.mpo_left,
                    self.mpo_right,
                    self.right_block,
                    self.basis,
                )
            return diagonal_from_factorized_terms(
                factorized_terms,
                self.basis,
                dtype=out_dtype,
            )
        if _is_identity_mpo_core(self.mpo_left) and _is_identity_mpo_core(self.mpo_right):
            return self._identity_diagonal(out_dtype)
        return self._block_sparse_diagonal()

    def _identity_diagonal(self, dtype):
        diag = np.zeros(self.basis.size, dtype=float)
        for entry in self.basis:
            q_l, _q_p1, _q_p2, q_r = entry.key
            E_block = self.left_block.get((q_l, q_l))
            F_block = self.right_block.get((q_r, q_r))
            if E_block is None or F_block is None:
                continue
            diag_left = np.real(np.diag(self._identity_env_to_matrix(E_block, dtype=dtype)))
            diag_right = np.real(np.diag(self._identity_env_to_matrix(F_block, dtype=dtype)))
            diag_block = np.einsum(
                "l,p,q,r->lpqr",
                diag_left,
                np.ones(int(entry.shape[1]), dtype=float),
                np.ones(int(entry.shape[2]), dtype=float),
                diag_right,
                optimize=True,
            )
            self.basis.write_packed_block(diag, entry, diag_block)
        return diag

    def _identity_env_to_matrix(self, block, *, dtype):
        arr = np.asarray(block, dtype=dtype)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        raise ValueError(
            "Identity-MPOCore local diagonal expects rank-2 environment blocks or "
            f"rank-3 blocks with leading dimension 1, got {arr.shape!r}."
        )

    def _block_sparse_diagonal(self):
        from .environment import _two_site_diagonal_block

        diag = np.zeros(self.basis.size, dtype=float)
        for entry in self.basis:
            q_l, q_p1, q_p2, q_r = entry.key
            E_block = self.left_block.get((q_l, q_l))
            F_block = self.right_block.get((q_r, q_r))
            if E_block is None or F_block is None:
                continue
            W1_slice = self._physical_diagonal_slice(self.mpo_left, q_p1, self.phys1_slices)
            W2_slice = self._physical_diagonal_slice(self.mpo_right, q_p2, self.phys2_slices)
            if W1_slice is None or W2_slice is None:
                continue
            diag_block = _two_site_diagonal_block(E_block, W1_slice, W2_slice, F_block)
            self.basis.write_packed_block(diag, entry, np.real(diag_block))
        return diag

    def local_actions(self):
        """
        Build tensor, reduced, and packed local actions for this effective operator.

        :returns: ``(tensor_matvec, reduced_matvec, packed_matvec, diag,
            identity_like)``.
        """

        return self.compile_actions().as_tuple()

    def compile_actions(self):
        """
        Compile this effective block operator into solver-facing local actions.

        :returns: ``CompiledLocalActions`` object owning local matvecs,
            diagonal metadata, and identity metadata.
        """

        from .environment import _is_identity_mpo_core

        if self.local_operator_table is not None:
            self._attach_table_metadata(self.local_operator_table)
            return self.local_operator_table.actions
        table = RenormalizedLocalOperatorTableBuilder(self).build()
        if table is not None:
            return table.actions

        return self._compile_actions_uncached(None)

    def _compile_actions_uncached(self, representation, *, require_symbolic_payloads=False):
        from .environment import _is_identity_mpo_core

        require_prepared_tables = representation is not None
        if self.rank_coupled:
            if (
                representation
                in {"rank_coupled_complementary", "rank_coupled_contextual"}
                and self.complementary_operator_families is not None
            ):
                tensor_apply, reduced_apply, packed_apply, diag, identity_like = self._rank_coupled_complementary_local_actions(
                    require_prepared_tables=require_prepared_tables,
                    require_symbolic_payloads=require_symbolic_payloads,
                )
            else:
                tensor_apply, reduced_apply, packed_apply, diag, identity_like = self._rank_coupled_local_actions(
                    require_prepared_tables=require_prepared_tables,
                    require_symbolic_payloads=require_symbolic_payloads,
                )
        elif _is_identity_mpo_core(self.mpo_left) and _is_identity_mpo_core(self.mpo_right):
            tensor_apply, reduced_apply, packed_apply, diag, identity_like = self._identity_local_actions(
                out_dtype=self.output_dtype(),
            )
        else:
            tensor_apply, reduced_apply, packed_apply, identity_like = self._standard_local_actions(
                out_dtype=self.output_dtype(),
                require_prepared_tables=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            )
            diag = self.diagonal()
        actions = CompiledLocalActions(
            basis=self.basis,
            tensor_matvec=tensor_apply,
            reduced_matvec=reduced_apply,
            packed_matvec=packed_apply,
            diag=diag,
            identity_like=identity_like,
            name=self.name,
            metadata=self.boundary_metadata(),
        )
        build_timing = getattr(packed_apply, "renormalized_operator_build_timing", None)
        if build_timing:
            actions.metadata["renormalized_operator_build_timing"] = {
                str(key): float(value)
                for key, value in build_timing.items()
            }
        qchem_plan = getattr(packed_apply, "su2_qchem_sweep_plan", None)
        if qchem_plan is not None:
            actions.metadata["su2_qchem_sweep_plan"] = qchem_plan
        actions.metadata["symbolic_boundary_payloads"] = self.symbolic_boundary_metadata()
        if self.complementary_operator_families is not None:
            actions.metadata["complementary_operator_families"] = (
                self.complementary_operator_family_metadata()
            )
        return actions

    def _standard_local_actions(
        self,
        *,
        out_dtype,
        require_prepared_tables=False,
        require_symbolic_payloads=False,
    ):
        """
        Build local actions for a standard block-sparse effective operator.

        :param out_dtype: Scalar dtype for the local operator.
        :param require_prepared_tables: Require cached side/factor tables instead
            of building lazy tables during local-operator compilation.
        :param require_symbolic_payloads: Require cached side/factor tables to
            be sourced from symbolic-owned boundary payloads.
        :returns: ``(tensor_matvec, reduced_matvec, packed_matvec,
            identity_like)``.
        """

        from .environment import (
            _FACTORIZED_PACKED_LOCAL_DIM,
            _compile_factorized_terms,
            _compile_packed_transitions,
            _is_identity_mpo_core,
            _precompute_two_site_block_env_factorized_terms,
            _precompute_two_site_block_env_transitions,
        )

        if _is_identity_mpo_core(self.mpo_left) and _is_identity_mpo_core(self.mpo_right):
            return self._identity_local_actions(out_dtype=out_dtype)

        if self.basis.size > _FACTORIZED_PACKED_LOCAL_DIM:
            transition_cache = {}

            def _lazy_transitions():
                cached = transition_cache.get("value")
                if cached is None:
                    cached = _precompute_two_site_block_env_transitions(
                        self.left_block,
                        self.mpo_left,
                        self.mpo_right,
                        self.right_block,
                        self.basis,
                        self.phys1_slices,
                        self.phys2_slices,
                        left_blocks_by_ket=self._side_table(
                            "left",
                        "block_by_ket",
                        require_existing=require_prepared_tables,
                        require_symbolic_payloads=require_symbolic_payloads,
                    ),
                    right_blocks_by_ket=self._side_table(
                        "right",
                        "block_by_ket",
                        require_existing=require_prepared_tables,
                        require_symbolic_payloads=require_symbolic_payloads,
                    ),
                    )
                    transition_cache["value"] = cached
                return cached

            def tensor_apply(two_site):
                out_entries, transitions = _lazy_transitions()
                return apply_transition_tensor(
                    transitions,
                    two_site,
                    out_entries,
                    base_dtype=out_dtype,
                )

            def reduced_apply(state):
                out_entries, transitions = _lazy_transitions()
                return apply_transition_reduced(
                    transitions,
                    state,
                    out_entries,
                    base_dtype=out_dtype,
                )

            _packed_out_entries, factorized_terms = _precompute_two_site_block_env_factorized_terms(
                self.left_block,
                self.mpo_left,
                self.mpo_right,
                self.right_block,
                self.basis,
                self.phys1_slices,
                self.phys2_slices,
                left_blocks_by_ket=self._side_table(
                    "left",
                    "array_by_ket",
                    require_existing=require_prepared_tables,
                    require_symbolic_payloads=require_symbolic_payloads,
                ),
                right_blocks_by_ket=self._side_table(
                    "right",
                    "array_by_ket",
                    require_existing=require_prepared_tables,
                    require_symbolic_payloads=require_symbolic_payloads,
                ),
                left_factor_table=self._factor_side_table(
                    "left",
                    "left_factor_by_ket",
                    require_existing=require_prepared_tables,
                    require_symbolic_payloads=require_symbolic_payloads,
                ),
                right_factor_table=self._factor_side_table(
                    "right",
                    "right_factor_by_ket",
                    require_existing=require_prepared_tables,
                    require_symbolic_payloads=require_symbolic_payloads,
                ),
            )
            compiled_factorized_terms = _compile_factorized_terms(factorized_terms, self.basis)
            packed_apply = compiled_factorized_terms.packed_matvec(
                base_dtype=out_dtype,
                backend="factorized-batched",
                out_entries=_packed_out_entries,
                block_matrices=compiled_factorized_terms,
            )
            self._annotate_symbolic_payload_source(packed_apply, compiled_factorized_terms)
            return tensor_apply, reduced_apply, packed_apply, False

        out_entries, transitions = _precompute_two_site_block_env_transitions(
            self.left_block,
            self.mpo_left,
            self.mpo_right,
            self.right_block,
            self.basis,
            self.phys1_slices,
            self.phys2_slices,
            left_blocks_by_ket=self._side_table(
                "left",
                "block_by_ket",
                require_existing=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            ),
            right_blocks_by_ket=self._side_table(
                "right",
                "block_by_ket",
                require_existing=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            ),
        )

        compiled_transitions = _compile_packed_transitions(transitions, self.basis)

        def tensor_apply(two_site):
            return compiled_transitions.apply_tensor(two_site, base_dtype=out_dtype)

        def reduced_apply(state):
            return compiled_transitions.apply_reduced(state, base_dtype=out_dtype)

        packed_apply = compiled_transitions.packed_matvec(base_dtype=out_dtype)
        self._annotate_symbolic_payload_source(packed_apply)

        identity_like = transitions_are_identity_operator(self.basis, transitions)
        return tensor_apply, reduced_apply, packed_apply, identity_like

    def _identity_local_actions(self, *, out_dtype):
        """
        Build local actions for an identity-MPOCore effective block operator.

        :param out_dtype: Scalar dtype for the local operator.
        :returns: ``(tensor_matvec, reduced_matvec, packed_matvec,
            identity_like)``.
        """

        tensor_apply, reduced_apply, packed_apply, diag, identity_like = (
            build_identity_mpo_local_actions(
            self.left_block,
            self.right_block,
            self.basis,
            base_dtype=out_dtype,
        )
        )
        self._annotate_symbolic_payload_source(packed_apply)
        return tensor_apply, reduced_apply, packed_apply, diag, identity_like

    def _rank_coupled_local_actions(
        self,
        *,
        require_prepared_tables=False,
        require_symbolic_payloads=False,
    ):
        """
        Build local actions for a rank-coupled effective block operator.

        :param require_prepared_tables: Require cached rank-coupled side/factor
            tables instead of constructing lazy tables during compilation.
        :param require_symbolic_payloads: Require cached side/factor tables to
            be sourced from symbolic-owned boundary payloads.
        :returns: ``(tensor_matvec, reduced_matvec, packed_matvec, diag,
            identity_like)``.
        """

        from .environment import (
            _compile_factorized_terms,
            _precompute_two_site_rank_coupled_factorized_terms,
        )

        out_dtype = self.output_dtype()
        timing = {}
        t0 = time.perf_counter()
        left_boundary_record = self._side_table_record(
            "left",
            "rank_coupled_by_ket",
            require_existing=require_prepared_tables,
            require_symbolic_payloads=require_symbolic_payloads,
        )
        right_boundary_record = self._side_table_record(
            "right",
            "rank_coupled_by_ket",
            require_existing=require_prepared_tables,
            require_symbolic_payloads=require_symbolic_payloads,
        )
        left_blocks = (
            self._group_side_blocks(self.left_block, "rank_coupled_by_ket")
            if left_boundary_record is None
            else left_boundary_record.grouped_by_ket
        )
        right_blocks = (
            self._group_side_blocks(self.right_block, "rank_coupled_by_ket")
            if right_boundary_record is None
            else right_boundary_record.grouped_by_ket
        )
        if (
            getattr(self.basis, "channel_resolved", False)
            and all(
                int(entry.shape[1]) == 1 and int(entry.shape[2]) == 1
                for entry in self.basis
            )
        ):
            from .su2_qchem_plan import (
                build_contextual_channel_compiled_terms,
                pack_rank_coupled_boundary_table_from_block_map,
            )

            left_packed = getattr(left_boundary_record, "packed_table", None)
            right_packed = getattr(right_boundary_record, "packed_table", None)
            if left_packed is None:
                left_packed = pack_rank_coupled_boundary_table_from_block_map(
                    self.left_block,
                    side="left",
                    bond=int(getattr(self.left_entry, "bond", 0)),
                )
            if right_packed is None:
                right_packed = pack_rank_coupled_boundary_table_from_block_map(
                    self.right_block,
                    side="right",
                    bond=int(getattr(self.right_entry, "bond", 0)),
                )
            compiled = build_contextual_channel_compiled_terms(
                self.basis,
                self.mpo_left,
                self.mpo_right,
                left_packed,
                right_packed,
                bond=int(getattr(self.left_entry, "bond", 0)),
                moving_environment=self.su2_moving_environment,
            )
            if compiled is None:
                raise RuntimeError(
                    "Could not compile contextual SU(2) channel routes."
                )
            packed_apply = compiled.packed_matvec(
                base_dtype=out_dtype,
                backend="su2-contextual-cpp",
                out_entries=self.basis.out_entries,
                block_matrices=compiled,
            )
            qchem_stats = dict(compiled.plan.stats)
            qchem_stats["contextual_channel_resolved"] = True
            packed_apply.su2_qchem_sweep_plan = qchem_stats
            compiled.su2_qchem_sweep_plan = qchem_stats
            timing["contextual_cpp_routes"] = time.perf_counter() - t0
            packed_apply.renormalized_operator_build_timing = timing
            self._annotate_symbolic_payload_source(packed_apply, compiled)
            if self.complementary_operator_families is not None:
                compiled.complementary_operator_families = (
                    self.complementary_operator_families
                )
                self._annotate_complementary_payload_source(
                    packed_apply,
                    compiled,
                )

            def tensor_apply(two_site):
                packed, _ = pack_two_site_state(
                    two_site,
                    layout=self.basis,
                )
                return unpack_two_site_state(
                    packed_apply(packed),
                    two_site,
                    layout=self.basis,
                )

            def reduced_apply(state):
                return state.layout.from_packed(
                    packed_apply(state.to_packed(dtype=out_dtype))
                )

            return (
                tensor_apply,
                reduced_apply,
                packed_apply,
                (
                    None
                    if bool(
                        getattr(
                            compiled,
                            "cpp_deferred_diagonal",
                            False,
                        )
                    )
                    else np.asarray(compiled.diagonal(), dtype=float)
                ),
                False,
            )
        left_factor_record = (
            self._factor_side_table_record(
                "left",
                "rank_coupled_left_factor_by_ket",
                require_existing=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            )
            if self.left_entry is not None
            else None
        )
        right_factor_record = (
            self._factor_side_table_record(
                "right",
                "rank_coupled_right_factor_by_ket",
                require_existing=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            )
            if self.right_entry is not None
            else None
        )
        left_factors = (
            None
            if left_factor_record is None
            else getattr(left_factor_record, "grouped_by_ket", None)
        )
        right_factors = (
            None
            if right_factor_record is None
            else getattr(right_factor_record, "grouped_by_ket", None)
        )
        timing["rank_coupled_side_table_fetch"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        qchem_plan_stats = None
        qchem_schedule = None
        qchem_compiled_terms = None
        left_packed = getattr(left_factor_record, "packed_table", None)
        right_packed = getattr(right_factor_record, "packed_table", None)
        if left_packed is not None and right_packed is not None:
            try:
                from .su2_qchem_plan import SU2QChemSweepPlan

                left_boundary_packed = getattr(
                    left_boundary_record,
                    "packed_table",
                    None,
                )
                right_boundary_packed = getattr(
                    right_boundary_record,
                    "packed_table",
                    None,
                )
                qchem_plan_key = (
                    "su2_qchem_sweep_plan",
                    int(getattr(left_factor_record, "owner_bond", 0)),
                    int(getattr(left_packed, "revision", 0)),
                    int(getattr(right_packed, "revision", 0)),
                    int(getattr(left_boundary_packed, "revision", 0)),
                    int(getattr(right_boundary_packed, "revision", 0)),
                )
                qchem_plan = None
                qchem_plan_cache_hit = False
                engine = self.su2_operator_engine
                if engine is not None:
                    before_hits = int(engine.stats.get("plan_hits", 0))
                    qchem_plan = engine.sweep_plan(
                        bond=getattr(left_factor_record, "owner_bond", 0),
                        left_factor_table=left_packed,
                        right_factor_table=right_packed,
                        left_boundary_table=left_boundary_packed,
                        right_boundary_table=right_boundary_packed,
                        su2_moving_environment=self.su2_moving_environment,
                    )
                    qchem_plan_cache_hit = (
                        int(engine.stats.get("plan_hits", 0)) > before_hits
                    )
                else:
                    plan_getter = getattr(
                        left_factor_record,
                        "get_qchem_sweep_plan",
                        None,
                    )
                    if plan_getter is not None:
                        qchem_plan = plan_getter(qchem_plan_key)
                        qchem_plan_cache_hit = qchem_plan is not None
                if qchem_plan is None:
                    qchem_plan = SU2QChemSweepPlan(
                        bond=getattr(left_factor_record, "owner_bond", 0),
                        left_factor_table=left_packed,
                        right_factor_table=right_packed,
                        left_boundary_table=left_boundary_packed,
                        right_boundary_table=right_boundary_packed,
                        su2_moving_environment=self.su2_moving_environment,
                    )
                    if engine is None:
                        plan_putter = getattr(
                            left_factor_record,
                            "put_qchem_sweep_plan",
                            None,
                        )
                        if plan_putter is not None:
                            plan_putter(qchem_plan_key, qchem_plan)
                qchem_direct_parent_blocks = _su2_qchem_direct_parent_blocks_enabled()
                qchem_compiled_terms = qchem_plan.compile_factorized_terms(
                    self.basis,
                    prefer_packed=qchem_direct_parent_blocks,
                )
                qchem_schedule = None if qchem_compiled_terms is None else self.basis.out_entries
                qchem_plan_stats = qchem_plan.stats
                qchem_plan_stats["plan_cache_hit"] = bool(qchem_plan_cache_hit)
                qchem_plan_stats["packed_compiled_terms"] = bool(
                    getattr(
                        qchem_compiled_terms,
                        "qchem_packed_entry_kernel_provider",
                        False,
                    )
                )
                qchem_plan_stats["plan_cache_owner"] = {
                    "kind": (
                        "su2_operator_engine"
                        if engine is not None
                        else "boundary_entry"
                    ),
                    **(
                        dict(engine.stats)
                        if engine is not None
                        else {
                            "side": str(
                                getattr(left_factor_record, "owner_side", "")
                            ),
                            "bond": int(
                                getattr(left_factor_record, "owner_bond", 0)
                            ),
                            "cache_size": int(
                                len(
                                    getattr(
                                        left_factor_record,
                                        "qchem_sweep_plan_cache",
                                        {},
                                    )
                                )
                            ),
                            "cache_hits": int(
                                getattr(
                                    left_factor_record,
                                    "qchem_sweep_plan_cache_stats",
                                    {},
                                ).get("hits", 0)
                            ),
                            "cache_misses": int(
                                getattr(
                                    left_factor_record,
                                    "qchem_sweep_plan_cache_stats",
                                    {},
                                ).get("misses", 0)
                            ),
                            "cache_puts": int(
                                getattr(
                                    left_factor_record,
                                    "qchem_sweep_plan_cache_stats",
                                    {},
                                ).get("puts", 0)
                            ),
                        }
                    ),
                }
                if qchem_compiled_terms is not None:
                    qchem_compiled_terms.su2_qchem_sweep_plan_object = qchem_plan
            except Exception as exc:
                qchem_plan_stats = {
                    "kind": "su2_qchem_sweep_plan",
                    "supported": False,
                    "error": str(exc),
                }
                qchem_schedule = None
                qchem_compiled_terms = None
        timing["cpp_factor_schedule"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if qchem_compiled_terms is None:
            if left_blocks is None and left_boundary_record is not None:
                left_blocks = left_boundary_record.grouped_payload()
            if right_blocks is None and right_boundary_record is not None:
                right_blocks = right_boundary_record.grouped_payload()
            if left_factors is None and left_factor_record is not None:
                left_factors = left_factor_record.grouped_payload()
            if right_factors is None and right_factor_record is not None:
                right_factors = right_factor_record.grouped_payload()
            packed_out_entries, factorized_terms = _precompute_two_site_rank_coupled_factorized_terms(
                self.left_block,
                self.mpo_left,
                self.mpo_right,
                self.right_block,
                self.basis,
                left_blocks_by_ket=left_blocks,
                right_blocks_by_ket=right_blocks,
                left_factor_table=left_factors,
                right_factor_table=right_factors,
            )
            timing["rank_coupled_factorized_terms"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            compiled_factorized_terms = _compile_factorized_terms(factorized_terms, self.basis)
            timing["rank_coupled_compile_factorized_terms"] = time.perf_counter() - t0
        else:
            packed_out_entries = qchem_schedule
            factorized_terms = None
            compiled_factorized_terms = qchem_compiled_terms
            timing["rank_coupled_factorized_terms"] = 0.0
            timing["rank_coupled_compile_factorized_terms"] = 0.0
        t0 = time.perf_counter()
        packed_apply = compiled_factorized_terms.packed_matvec(
            base_dtype=out_dtype,
            backend="rank-coupled-factorized-batched",
            out_entries=packed_out_entries,
            block_matrices=compiled_factorized_terms,
        )
        timing["rank_coupled_packed_matvec_factory"] = time.perf_counter() - t0
        self._annotate_symbolic_payload_source(packed_apply, compiled_factorized_terms)

        def tensor_apply(two_site):
            packed, _ = pack_two_site_state(two_site, layout=self.basis)
            return unpack_two_site_state(packed_apply(packed), two_site, layout=self.basis)

        def reduced_apply(state):
            return state.layout.from_packed(packed_apply(state.to_packed(dtype=out_dtype)))

        t0 = time.perf_counter()
        if factorized_terms is None:
            diagonal_provider = getattr(compiled_factorized_terms, "diagonal", None)
            if diagonal_provider is not None:
                diag = np.asarray(diagonal_provider(), dtype=float)
            else:
                diag = np.zeros(self.basis.size, dtype=float)
                for entry in self.basis:
                    block = compiled_factorized_terms.block_matrix_for(entry)
                    if block is not None:
                        diag[entry.slice] = np.real(np.diag(block))
        else:
            diag = self.diagonal(
                factorized_terms=factorized_terms,
                require_symbolic_payloads=require_symbolic_payloads,
            )
        timing["rank_coupled_diagonal"] = time.perf_counter() - t0
        if qchem_plan_stats is not None:
            packed_apply.su2_qchem_sweep_plan = qchem_plan_stats
            compiled_factorized_terms.su2_qchem_sweep_plan = qchem_plan_stats
        packed_apply.renormalized_operator_build_timing = timing
        return tensor_apply, reduced_apply, packed_apply, diag, False

    def _rank_coupled_complementary_local_actions(
        self,
        *,
        require_prepared_tables=False,
        require_symbolic_payloads=False,
    ):
        """
        Build rank-coupled local actions through the complementary-family seam.

        This is the first replacement point for the old scalar-coupled local
        table path.  The current numeric kernel still delegates to the
        rank-coupled factorized implementation, but the compiled action is
        explicitly marked as complementary-family driven so the next step can
        swap in direct ``S/R/A/P/B/Q`` contractions without changing the solver
        or cache ownership API.

        :param require_prepared_tables: Require prebuilt symbolic factor
            tables on the boundary entries.
        :param require_symbolic_payloads: Require symbolic-owned numeric
            boundary payloads.
        :returns: ``(tensor_matvec, reduced_matvec, packed_matvec, diag,
            identity_like)``.
        """

        out = self._rank_coupled_local_actions(
            require_prepared_tables=require_prepared_tables,
            require_symbolic_payloads=require_symbolic_payloads,
        )
        packed_apply = out[2]
        metadata = dict(getattr(packed_apply, "symbolic_boundary_payloads", {}) or {})
        metadata["complementary_operator_source"] = "spatial_S/R/A/P/B/Q"
        metadata["complementary_operator_families"] = self.complementary_operator_family_metadata()
        metadata["complementary_boundary_payloads"] = (
            self.complementary_boundary_payload_metadata()
        )
        packed_apply.symbolic_boundary_payloads = metadata
        compiled_terms = getattr(packed_apply, "compiled_factorized_terms", None)
        if compiled_terms is not None:
            compiled_terms.symbolic_boundary_payloads = metadata
            compiled_terms.complementary_operator_families = (
                self.complementary_operator_families
            )
            self._annotate_complementary_payload_source(
                packed_apply,
                compiled_terms,
            )
            compiled_terms.complementary_direct_orthonormal_projection_available = True
            compiled_terms.explicit_direct_orthonormal_projection = bool(
                getattr(
                    self.complementary_operator_families,
                    "prefer_direct_orthonormal_projection",
                    False,
                )
            )
            compiled_terms.prefer_direct_orthonormal_projection = bool(
                compiled_terms.explicit_direct_orthonormal_projection
            )
            compiled_terms.prefer_direct_component_transform = bool(
                getattr(
                    self.complementary_operator_families,
                    "prefer_direct_component_transform",
                    False,
                )
            )
            compiled_terms.prefer_recursive_operator_matvec = bool(
                getattr(
                    self.complementary_operator_families,
                    "prefer_recursive_operator_matvec",
                    False,
                )
            )
            compiled_terms.prefer_complementary_payload_tensor_matvec = bool(
                getattr(compiled_terms, "complementary_payload_backed", False)
                and getattr(
                    self.complementary_operator_families,
                    "prefer_complementary_payload_tensor_matvec",
                    False,
                )
            )
            compiled_terms.direct_orthonormal_projection_source = (
                "rank_coupled_complementary"
            )
            qchem_direct_parent_blocks = bool(
                getattr(compiled_terms, "su2_qchem_sweep_plan_object", None)
                is not None
                and _su2_qchem_direct_parent_blocks_enabled()
            )
            if qchem_direct_parent_blocks:
                compiled_terms.prefer_direct_orthonormal_projection = True
                compiled_terms.direct_orthonormal_projection_source = (
                    "rank_coupled_qchem_parent_blocks"
                )
            if bool(
                getattr(
                    compiled_terms,
                    "qchem_packed_entry_kernel_provider",
                    False,
                )
            ) and not qchem_direct_parent_blocks and not bool(
                getattr(
                    self.complementary_operator_families,
                    "prefer_direct_orthonormal_projection",
                    False,
                )
            ):
                compiled_terms.complementary_direct_orthonormal_projection_available = False
                compiled_terms.prefer_direct_orthonormal_projection = False
                compiled_terms.prefer_direct_component_transform = False
                compiled_terms.prefer_recursive_operator_matvec = False
                compiled_terms.prefer_complementary_payload_tensor_matvec = False
                compiled_terms.direct_orthonormal_projection_source = (
                    "rank_coupled_packed_entry_kernels"
                )
        return out

    def to_local_operator(self):
        """
        Convert this effective block operator to the solver-facing object.

        :returns: ``LocalOperator`` carrying tensor/reduced/packed matvecs.
        """

        return self.compile_actions().to_local_operator()
