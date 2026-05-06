#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-site effective block operators for non-Abelian DMRG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .local_operator import (
    CompiledLocalActions,
    apply_transition_reduced,
    apply_transition_tensor,
    compile_packed_transitions,
    diagonal_from_factorized_terms,
    identity_mpo_transitions,
    transitions_are_identity_operator,
)
from .mpo import MPO, IrreducibleMPO, RankCoupledMPO
from .solver import pack_two_site_state, unpack_two_site_state


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
            ``"factorized"``, ``"rank_coupled_factorized"``, or
            ``"rank_coupled_complementary"``.
        """

        from .environment import _FACTORIZED_PACKED_LOCAL_DIM, _is_identity_mpo_core

        if self.operator.rank_coupled:
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

        owner = self.owner
        if owner is None:
            return None
        representation = self.representation()
        require_symbolic_payloads = representation in {
            "transition",
            "factorized",
            "rank_coupled_factorized",
            "rank_coupled_complementary",
        }
        self.operator.ensure_side_operator_tables(
            representation,
            require_factor_tables=representation
            in {"factorized", "rank_coupled_factorized", "rank_coupled_complementary"},
            require_symbolic_payloads=require_symbolic_payloads,
        )
        cached = owner.get_local_operator_table(self.key())
        if cached is not None:
            self.operator._attach_table_metadata(cached)
            return cached
        actions = self.operator._compile_actions_uncached(
            representation,
            require_symbolic_payloads=require_symbolic_payloads,
        )
        table = owner.put_local_operator_table(
            self.key(),
            actions,
            representation=representation,
            basis_size=self.operator.basis.size,
        )
        actions.local_operator_table = table
        self.operator._attach_table_metadata(table)
        return table


@dataclass
class EffectiveBlockOperator:
    """
    Two-site effective operator assembled from explicit renormalized blocks.

    :param left_block: Left renormalized environment block.
    :param mpo_left: MPO core on the left active site.
    :param mpo_right: MPO core on the right active site.
    :param right_block: Right renormalized environment block.
    :param two_site_template: Rank-4 two-site tensor defining local sectors.
    :param basis: Explicit local two-site basis.
    :param phys1_slices: Physical sector slices for the left active site.
    :param phys2_slices: Physical sector slices for the right active site.
    :param rank_coupled: Whether the MPO/environment path is rank-coupled.
    :param left_entry: Optional persisted left boundary-stack entry.
    :param right_entry: Optional persisted right boundary-stack entry.
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
        actions.local_operator_table = table
        self.local_operator_table = table
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
            if symbolic_table is not None:
                grouped = symbolic_table.group_boundary_blocks(representation=representation)
                source = "symbolic_" + str(source)
            elif require_symbolic_payloads:
                raise RuntimeError(
                    f"Missing symbolic-owned payload table for {side} boundary "
                    f"{entry.bond}."
                )
            else:
                grouped = self._group_side_blocks(block_map, representation)
            table = entry.put_side_operator_table(
                key,
                grouped,
                representation=representation,
                source=source,
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
        :param W: Adjacent MPO core.
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

        if representation in {"rank_coupled_factorized", "rank_coupled_complementary"}:
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
            self._factor_side_table(
                "left",
                "rank_coupled_left_factor_by_ket",
                require_existing=require_factor_tables,
                require_symbolic_payloads=require_symbolic_payloads,
                source=source,
            )
            self._factor_side_table(
                "right",
                "rank_coupled_right_factor_by_ket",
                require_existing=require_factor_tables,
                require_symbolic_payloads=require_symbolic_payloads,
                source=source,
            )
        elif representation == "factorized":
            self._factor_side_table(
                "left",
                "left_factor_by_ket",
                require_existing=require_factor_tables,
                require_symbolic_payloads=require_symbolic_payloads,
                source=source,
            )
            self._factor_side_table(
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

        :returns: NumPy result dtype from MPO cores and the two-site template.
        """

        from .environment import _mpo_dtype

        return np.result_type(
            _mpo_dtype(self.mpo_left),
            _mpo_dtype(self.mpo_right),
            *(np.asarray(block).dtype for block in self.two_site_template.data.values()),
        )

    def _physical_diagonal_slice(self, mpo_core, sector, phys_slices):
        if isinstance(mpo_core, (MPO, IrreducibleMPO, RankCoupledMPO)):
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
            "Identity-MPO local diagonal expects rank-2 environment blocks or "
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
            if representation == "rank_coupled_complementary":
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
            tensor_apply, reduced_apply, packed_apply, identity_like = self._identity_local_actions(
                out_dtype=self.output_dtype(),
            )
            diag = self.diagonal()
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
        Build local actions for an identity-MPO effective block operator.

        :param out_dtype: Scalar dtype for the local operator.
        :returns: ``(tensor_matvec, reduced_matvec, packed_matvec,
            identity_like)``.
        """

        out_entries, transitions = identity_mpo_transitions(
            self.left_block,
            self.right_block,
            self.basis,
            base_dtype=out_dtype,
        )
        compiled_transitions = compile_packed_transitions(transitions, self.basis)

        def tensor_apply(two_site):
            return compiled_transitions.apply_tensor(two_site, base_dtype=out_dtype)

        def reduced_apply(state):
            return compiled_transitions.apply_reduced(state, base_dtype=out_dtype)

        packed_apply = compiled_transitions.packed_matvec(base_dtype=out_dtype)
        self._annotate_symbolic_payload_source(packed_apply)

        identity_like = transitions_are_identity_operator(self.basis, transitions)
        return tensor_apply, reduced_apply, packed_apply, identity_like

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
        packed_out_entries, factorized_terms = _precompute_two_site_rank_coupled_factorized_terms(
            self.left_block,
            self.mpo_left,
            self.mpo_right,
            self.right_block,
            self.basis,
            left_blocks_by_ket=self._side_table(
                "left",
                "rank_coupled_by_ket",
                require_existing=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            ),
            right_blocks_by_ket=self._side_table(
                "right",
                "rank_coupled_by_ket",
                require_existing=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            ),
            left_factor_table=self._factor_side_table(
                "left",
                "rank_coupled_left_factor_by_ket",
                require_existing=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            ),
            right_factor_table=self._factor_side_table(
                "right",
                "rank_coupled_right_factor_by_ket",
                require_existing=require_prepared_tables,
                require_symbolic_payloads=require_symbolic_payloads,
            ),
        )
        compiled_factorized_terms = _compile_factorized_terms(factorized_terms, self.basis)
        packed_apply = compiled_factorized_terms.packed_matvec(
            base_dtype=out_dtype,
            backend="rank-coupled-factorized-batched",
            out_entries=packed_out_entries,
            block_matrices=compiled_factorized_terms,
        )
        self._annotate_symbolic_payload_source(packed_apply, compiled_factorized_terms)

        def tensor_apply(two_site):
            packed, _ = pack_two_site_state(two_site, layout=self.basis)
            return unpack_two_site_state(packed_apply(packed), two_site, layout=self.basis)

        def reduced_apply(state):
            return state.layout.from_packed(packed_apply(state.to_packed(dtype=out_dtype)))

        diag = self.diagonal(
            factorized_terms=factorized_terms,
            require_symbolic_payloads=require_symbolic_payloads,
        )
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
        packed_apply.symbolic_boundary_payloads = metadata
        compiled_terms = getattr(packed_apply, "compiled_factorized_terms", None)
        if compiled_terms is not None:
            compiled_terms.symbolic_boundary_payloads = metadata
            compiled_terms.complementary_operator_families = (
                self.complementary_operator_families
            )
            compiled_terms.complementary_direct_orthonormal_projection_available = True
            compiled_terms.prefer_direct_orthonormal_projection = bool(
                getattr(
                    self.complementary_operator_families,
                    "prefer_direct_orthonormal_projection",
                    False,
                )
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
            compiled_terms.direct_orthonormal_projection_source = (
                "rank_coupled_complementary"
            )
        return out

    def to_local_operator(self):
        """
        Convert this effective block operator to the solver-facing object.

        :returns: ``LocalOperator`` carrying tensor/reduced/packed matvecs.
        """

        return self.compile_actions().to_local_operator()
