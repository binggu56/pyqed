#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal sweep drivers for fixed-layout non-Abelian tensor chains.
"""

from __future__ import annotations

import inspect
import resource
import sys
import time
import numpy as np

from pyqed.mps.su2 import SpinChargeSector, fuse_charge_spin_sectors

from .canonical import (
    assert_mixed_canonical_sites,
    left_canonicalize_sites,
    mixed_canonicalize_sites,
    right_canonicalize_sites,
)
from .environment import (
    BlockSparseEnvironmentChain,
    contract_chain_expectation,
    rank_coupled_real_term_coalesce_stats,
)
from .contraction import (
    merge_mps_sites,
    merge_mps_sites_from_packed,
    mps_site_from_packed,
    normalize_site_tensor_layout,
    split_mps_sites_from_packed,
)
from .decompose import state_averaged_svd_two_site
from .linalg import sector_state_weight
from .mps import MPS
from .multiroot import MultiRootMPS, fuse_root_center_tensors
from .renormalized import RenormalizedBlockStack, RenormalizedOperatorStack
from .solver import TwoSiteEffectiveH, solve_local_two_site
from .solver import (
    _materialize_local_matrix,
    _layout_entries,
    _normalize_local_operator,
    _operator_basis_for_layout,
    _resolve_davidson_operator,
    _target_projector_basis_by_blocks,
    pack_two_site_state,
    unpack_two_site_state,
    two_site_state_basis,
)
from .tensor import NonabelianTensor
from .update import _expand_two_site_support, two_site_update


def _peak_rss_bytes():
    """Return this process's peak resident set size in bytes."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _current_rss_bytes():
    """Return this process's current resident set size when available."""

    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return _peak_rss_bytes()


def _ordered_union_qns(primary, secondary):
    ordered = []
    for sector in list(primary) + list(secondary):
        if sector not in ordered:
            ordered.append(sector)
    return ordered


def _sector_multiplicity(qns, sector):
    return sum(1 for item in qns if item == sector)


def _fuse_sectors(left, right):
    if hasattr(left, "fuse"):
        return tuple(left.fuse(right))
    if isinstance(left, SpinChargeSector) and isinstance(right, SpinChargeSector):
        return tuple(fuse_charge_spin_sectors(left, right))
    return ()


def _restore_site_bond_skeleton(site, reference):
    """
    Re-expose a site's left/right bond-sector skeleton after exact gauge moves.

    Exact canonicalization preserves the state but can collapse zero-valued
    bond sectors back to the occupied product path. For MPO sweeps we want to
    preserve any sector skeleton already present on the input MPS so the first
    local solve can immediately explore those symmetry-allowed channels.
    """
    if not isinstance(site, NonabelianTensor) or site.rank != 3:
        return site
    if not isinstance(reference, NonabelianTensor) or reference.rank != 3:
        return site

    left_order = _ordered_union_qns(reference.qns[0], site.qns[0])
    right_order = _ordered_union_qns(reference.qns[2], site.qns[2])
    phys_order = _ordered_union_qns(site.qns[1], reference.qns[1])

    left_qns = [
        sector
        for sector in left_order
        for _ in range(
            max(_sector_multiplicity(reference.qns[0], sector), _sector_multiplicity(site.qns[0], sector))
        )
    ]
    right_qns = [
        sector
        for sector in right_order
        for _ in range(
            max(_sector_multiplicity(reference.qns[2], sector), _sector_multiplicity(site.qns[2], sector))
        )
    ]

    if left_qns == site.qns[0] and right_qns == site.qns[2] and phys_order == site.qns[1]:
        return site

    left_dims = {sector: _sector_multiplicity(left_qns, sector) for sector in set(left_qns)}
    right_dims = {sector: _sector_multiplicity(right_qns, sector) for sector in set(right_qns)}
    phys_dims = {}
    for sector in phys_order:
        if sector in phys_dims:
            continue
        for key, block in site.data.items():
            if key[1] == sector:
                phys_dims[sector] = int(np.asarray(block).shape[1])
                break
        else:
            dim = getattr(sector, "dim", None)
            phys_dims[sector] = int(dim) if dim is not None else 1

    dtype = np.result_type(*[np.asarray(block).dtype for block in site.data.values()], float)
    data = {}
    for q_left in left_order:
        for q_phys in phys_order:
            fused = set(_fuse_sectors(q_left, q_phys))
            if not fused:
                continue
            for q_right in right_order:
                if q_right not in fused:
                    continue
                shape = (left_dims[q_left], phys_dims[q_phys], right_dims[q_right])
                block = np.zeros(shape, dtype=dtype)
                existing = site.data.get((q_left, q_phys, q_right))
                if existing is not None:
                    existing = np.asarray(existing)
                    block[: existing.shape[0], : existing.shape[1], : existing.shape[2]] = existing
                data[(q_left, q_phys, q_right)] = block

    return NonabelianTensor(
        data,
        [left_qns, phys_order[:], right_qns],
        site.dirs[:],
        fusion_legs=site.fusion_legs[:],
        metadata={},
    )


def _restore_chain_bond_skeleton(sites, reference_sites):
    return [
        _restore_site_bond_skeleton(site, reference)
        for site, reference in zip(sites, reference_sites)
    ]


def _call_solver(solver, bond, merged):
    if solver is None:
        return None
    try:
        signature = inspect.signature(solver)
    except (TypeError, ValueError):
        return solver(merged)

    positional = [
        param for param in signature.parameters.values()
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    has_varargs = any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in signature.parameters.values()
    )
    if has_varargs or len(positional) >= 2:
        return solver(bond, merged)
    return solver(merged)


def _normalize_direction(direction):
    direction = direction.lower()
    if direction in {"lr", "left-to-right", "left_to_right", "right"}:
        return "lr"
    if direction in {"rl", "right-to-left", "right_to_left", "left"}:
        return "rl"
    raise ValueError(f"Unknown sweep direction {direction!r}.")


def _emit_verbose(message, *, verbose):
    if int(verbose) > 0:
        print(message)


def _format_verbose_number(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.12g}"
    except Exception:
        return str(value)


def _format_metric_number(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.3e}"
    except Exception:
        return str(value)


def _format_verbose_sequence(values, *, max_items=4):
    if values is None:
        return "-"
    try:
        seq = list(values)
    except TypeError:
        return _format_verbose_number(values)
    shown = ", ".join(_format_verbose_number(value) for value in seq[:max_items])
    if len(seq) > max_items:
        shown += ", ..."
    return "[" + shown + "]"


def _format_bond_update_line(bond, update):
    objective = dict(update.get("local_objective") or {})
    return (
        f"  bond {bond:>2} | "
        f"problem={objective.get('effective_local_problem', '-'):>11} | "
        f"E={_format_verbose_number(objective.get('energy')):>14} | "
        f"Eroots={_format_verbose_sequence(objective.get('state_energies')):>28} | "
        f"E_post={_format_verbose_number(objective.get('post_update_energy')):>14} | "
        f"root_post={_format_verbose_sequence(objective.get('post_update_root_energies')):>28} | "
        f"kept={str(update.get('kept', '-')):>4} | "
        f"trunc={_format_verbose_number(update.get('trunc_err')):>10}"
    )


def _compact_sweep_update(bond, update):
    """Keep sweep diagnostics without retaining active two-site tensors."""
    compact = {
        "bond": int(bond),
        "trunc_err": float(update.get("trunc_err", 0.0)),
        "kept": update.get("kept"),
        "kept_states": int(update.get("kept_states", 0)),
        "local_guess_used": bool(update.get("local_guess_used", False)),
        "local_objective": dict(update.get("local_objective") or {}),
        "payload": "compact",
    }
    return compact


def _format_sweep_line(sweep_idx, direction, history_entry):
    delta = history_entry.get("energy_delta")
    delta_text = "" if delta is None else f" | dE={_format_metric_number(delta):>10}"
    return (
        f"sweep {sweep_idx:>2} | "
        f"dir={direction} | "
        f"E={_format_verbose_number(history_entry.get('energy')):>14} | "
        f"E_obj={_format_verbose_number(history_entry.get('objective_energy')):>14} | "
        f"metric={_format_metric_number(history_entry.get('metric')):>10}"
        f"{delta_text}"
    )


def _identity_mpo_factors_for_sites_and_mpo(sites, mpo_factors):
    from .builder import identity_operator
    from .environment import _tensor_dense_layout
    from .mpo import (
        MPO,
        IrreducibleMPO,
        RankCoupledMPO,
        Leg,
        as_rank_coupled_mpo,
    )

    identity_factors = []
    for site, factor in zip(sites, mpo_factors):
        if isinstance(factor, (MPO, IrreducibleMPO, RankCoupledMPO)):
            phys_leg = factor.phys_out_leg
        else:
            physical_slices = _tensor_dense_layout(site)["sector_slices"][1]
            phys_leg = Leg.from_slices(physical_slices)
        identity = MPO.from_site_operator(identity_operator(phys_leg))
        if (site.metadata or {}).get("physical_basis") == "fully_reduced_su2":
            identity = as_rank_coupled_mpo(identity)
            object.__setattr__(identity, "fully_reduced_identity", True)
        identity_factors.append(identity)
    return identity_factors


class MovingEnvironment:
    """
    Persistent moving-environment owner for MPO sweeps.

    A completed left-to-right sweep leaves valid left boundary entries for the
    next right-to-left sweep, and vice versa.  This object tracks that validity
    so :func:`sweep_once` can skip rebuilding the prebuilt side and consume the
    boundary stack produced by the previous sweep.
    """

    def __init__(
        self,
        sites,
        *,
        mpo_factors,
        root_target_mpo_factors=None,
        complementary_operator_families=None,
        materialize_complementary_family_operator_tables=True,
        su2_moving_environment=None,
        su2_boundary_environment=None,
        renormalized_operator_cache_max_size=256,
    ):
        self.mpo_factors = tuple(mpo_factors)
        self.hamiltonian_stack = RenormalizedBlockStack(
            namespace="hamiltonian",
            complementary_operator_families=complementary_operator_families,
            materialize_complementary_family_operator_tables=bool(
                materialize_complementary_family_operator_tables
            ),
            su2_moving_environment=su2_moving_environment,
            su2_boundary_environment=su2_boundary_environment,
        )
        self.su2_moving_environment = su2_moving_environment
        self.norm_stack = RenormalizedBlockStack(
            namespace="norm",
            su2_moving_environment=su2_moving_environment,
            su2_boundary_environment=su2_boundary_environment,
        )
        self.target_stack = (
            RenormalizedBlockStack(namespace="target")
            if root_target_mpo_factors is not None
            else None
        )
        self.identity_mpo_factors = _identity_mpo_factors_for_sites_and_mpo(
            sites,
            mpo_factors,
        )
        self.renormalized_operator_cache = RenormalizedOperatorStack(
            max_size=renormalized_operator_cache_max_size,
        )
        self.hamiltonian_valid_boundary_side = None
        self.norm_valid_boundary_side = None
        self.environment_rebuilds = 0
        self.boundary_side_reuses = 0
        self.boundary_side_rebuilds = 0
        self.norm_boundary_side_reuses = 0
        self.hamiltonian_numeric_refreshes = 0
        self.completed_sweeps = 0
        self.last_reused_prebuilt_side = None
        self.cursor_calls = 0
        self.cursor_steps = 0
        self.cursor_failures = 0
        self.cpp_state_owned = False
        self.cursor_owner = self.su2_moving_environment
        if self.cursor_owner is None:
            try:
                from pyqed.mps import cpp_davidson

                owner_cls = getattr(cpp_davidson, "MovingEnvironment", None)
                self.cursor_owner = None if owner_cls is None else owner_cls()
            except Exception:
                self.cursor_owner = None

    @staticmethod
    def needed_prebuilt_side(direction):
        direction = _normalize_direction(direction)
        return "right" if direction == "lr" else "left"

    @staticmethod
    def produced_boundary_side(direction):
        direction = _normalize_direction(direction)
        return "left" if direction == "lr" else "right"

    def reuse_side_for(self, direction):
        """
        Return the valid prebuilt side to reuse for the requested sweep.
        """

        hamiltonian_side, norm_side = self.reuse_sides_for(direction)
        return hamiltonian_side if hamiltonian_side == norm_side else None

    def reuse_sides_for(self, direction):
        """Return independently reusable Hamiltonian and norm boundary sides."""

        needed = self.needed_prebuilt_side(direction)
        hamiltonian_side = None
        norm_side = None
        if self.hamiltonian_valid_boundary_side == needed:
            self.boundary_side_reuses += 1
            self.last_reused_prebuilt_side = needed
            hamiltonian_side = needed
        else:
            self.environment_rebuilds += 1
            self.boundary_side_rebuilds += 1
            self.last_reused_prebuilt_side = None
        if self.norm_valid_boundary_side == needed:
            self.norm_boundary_side_reuses += 1
            norm_side = needed
        return hamiltonian_side, norm_side

    def finish_sweep(self, direction):
        """
        Mark the side advanced by a completed sweep as reusable.
        """

        produced = self.produced_boundary_side(direction)
        self.hamiltonian_valid_boundary_side = produced
        self.norm_valid_boundary_side = produced
        self.completed_sweeps += 1

    def refresh_hamiltonian(self):
        """Invalidate field-dependent numerics while retaining norm boundaries."""

        self.hamiltonian_stack.entries.clear()
        complementary = self.hamiltonian_stack.complementary_operator_stack
        if complementary is not None:
            complementary.entries.clear()
        self.renormalized_operator_cache.entries.clear()
        operator_engine = self.hamiltonian_stack.su2_operator_engine
        if operator_engine is not None:
            operator_engine.release_numeric()
        self.hamiltonian_valid_boundary_side = None
        self.hamiltonian_numeric_refreshes += 1
        return self

    def begin_half_sweep(self, direction, n_sites, sites=None):
        """Start one C++ moving-environment accounting interval."""

        if self.su2_moving_environment is not None:
            if sites is not None:
                self.su2_moving_environment.install_mps(sites)
            self.su2_moving_environment.begin_half_sweep(direction, int(n_sites))

    def begin_bond(self, bond):
        owner = self.su2_moving_environment
        if owner is not None:
            owner.begin_bond(int(bond))

    def claim_next_bond(self):
        """Claim the next bond from the active C++ half-sweep transaction."""

        owner = self.su2_moving_environment
        if owner is None:
            return -1
        return int(owner.claim_next_bond())

    def execute_half_sweep(self, callback):
        """Run all active bond transactions through one C++ entry point."""

        owner = self.su2_moving_environment
        if owner is None:
            raise RuntimeError("The SU(2) C++ sweep owner is unavailable.")
        return int(owner.execute_half_sweep(callback))

    def export_owned_sites(self, templates, direction):
        """Materialize the persistent C++ MPS once after all half sweeps."""

        owner = self.su2_moving_environment
        export = getattr(owner, "export_owned_split_sites", None)
        if owner is None or not callable(export):
            return [site.copy() for site in templates]
        records = export()
        if len(records) != len(templates):
            raise RuntimeError(
                f"C++ exported {len(records)} MPS sites; "
                f"expected {len(templates)}."
            )
        sites = [None] * len(templates)
        for record_index, record in enumerate(records):
            site = int(record["site"])
            if site < 0 or site >= len(templates) or sites[site] is not None:
                raise RuntimeError(
                    f"C++ exported an invalid final MPS site index {site}."
                )
            offsets = np.asarray(
                record["leg_sector_offsets"],
                dtype=np.int64,
            ).reshape(-1)
            labels = np.asarray(
                record["leg_sector_labels"],
                dtype=np.int64,
            ).reshape(-1, 2)
            dims = np.asarray(
                record["leg_sector_dims"],
                dtype=np.int64,
            ).reshape(-1)
            if offsets.size != 4 or labels.shape[0] != dims.size:
                raise RuntimeError(
                    "C++ exported malformed final-site leg topology."
                )

            def leg_topology(axis):
                start = int(offsets[axis])
                stop = int(offsets[axis + 1])
                return labels[start:stop], dims[start:stop]

            role = (
                "left"
                if (
                    direction == "lr" and site + 1 < len(templates)
                    or direction == "rl" and site == 0
                )
                else "right"
            )
            sites[site] = mps_site_from_packed(
                templates[site],
                record["tensor"],
                left_bond=leg_topology(0) if site > 0 else None,
                right_bond=(
                    leg_topology(2)
                    if site + 1 < len(templates)
                    else None
                ),
                svd_role=role,
            )
            owner.record_cpp_split_site(
                site,
                sites[site],
                record["revision"],
            )
        owner.release_workspaces()
        return sites

    def export_owned_state_average_sites(self, templates, direction):
        """Materialize all roots from the shared chain and C++ center bundle."""

        owner = self.su2_moving_environment
        center_export = (
            None
            if owner is None
            else owner.export_state_average_center()
        )
        if center_export is None:
            return None
        common = self.export_owned_sites(templates, direction)
        center_site = int(center_export["site"])
        root_sites = []
        for packed_values in center_export["values"]:
            sites = [site.copy() for site in common]
            center = sites[center_site].copy()
            cursor = 0
            data = {}
            for key, block in center.data.items():
                size = int(np.asarray(block).size)
                data[key] = np.asarray(
                    packed_values[cursor : cursor + size],
                    dtype=float,
                ).reshape(np.asarray(block).shape).copy()
                cursor += size
            if cursor != int(np.asarray(packed_values).size):
                raise RuntimeError(
                    "C++ state-average center does not cover its site topology."
                )
            center.data = data
            sites[center_site] = center
            root_sites.append(sites)
        return root_sites

    def clear_factor_routes(self):
        """Release bond-local C++ routes and their Python packed-table owners."""

        owner = self.su2_moving_environment
        if owner is not None:
            owner.clear_factor_routes()
        operator_engine = self.hamiltonian_stack.su2_operator_engine
        if operator_engine is not None:
            operator_engine.release_numeric()

    def release_operator_numeric(self):
        """Invalidate Python bond tables while retaining the C++ route owner."""

        operator_engine = self.hamiltonian_stack.su2_operator_engine
        if operator_engine is not None:
            operator_engine.release_numeric()

    def clear_local_operator(self):
        owner = self.su2_moving_environment
        if owner is not None:
            owner.clear_local_operator()

    def mark_bond_solved(self):
        owner = self.su2_moving_environment
        if owner is not None:
            owner.mark_bond_solved()

    def mark_bond_split(self, kept_states=0, truncation_seconds=0.0):
        owner = self.su2_moving_environment
        if owner is not None:
            owner.mark_bond_split(
                kept_states=int(kept_states),
                truncation_seconds=float(truncation_seconds),
            )

    def commit_bond(self, update):
        owner = self.su2_moving_environment
        if owner is None:
            return
        objective = update.get("local_objective") or {}
        solver_timing = objective.get("solver_timing") or {}
        commit = (
            owner.commit_bond_update
            if hasattr(owner, "commit_bond_update")
            else None
        )
        if commit is None:
            owner.mark_bond_advanced()
            commit = owner.commit_bond
        commit(
            matvec_calls=int(
                objective.get(
                    "matvec_count",
                    objective.get("matvec_calls", 0),
                )
                or 0
            ),
            davidson_iterations=int(
                objective.get("davidson_iterations", 0) or 0
            ),
            matvec_seconds=float(solver_timing.get("matvec", 0.0)),
            davidson_seconds=float(solver_timing.get("davidson", 0.0)),
            energy=objective.get("energy"),
        )

    def finish_half_sweep(self):
        """Commit a fully completed C++-owned half sweep."""

        owner = self.su2_moving_environment
        if owner is not None:
            owner.finish_half_sweep()

    def abort_half_sweep(self):
        owner = self.su2_moving_environment
        if owner is not None:
            owner.abort_half_sweep()

    def sweep_bonds(self, direction, n_sites):
        """Return the half-sweep bond schedule from the C++ owner."""

        direction = _normalize_direction(direction)
        owner = self.cursor_owner
        if owner is not None and hasattr(owner, "sweep_bonds"):
            try:
                if owner is self.su2_moving_environment:
                    scheduled = owner.sweep_bonds(direction, int(n_sites))
                else:
                    scheduled = owner.sweep_bonds(int(n_sites), direction)
                bonds = tuple(int(bond) for bond in scheduled)
                expected = tuple(
                    range(max(0, int(n_sites) - 1))
                    if direction == "lr"
                    else range(int(n_sites) - 2, -1, -1)
                )
                # The Abelian owner leaves the final edge step to its C++
                # boundary finalizer.  SU(2) still performs that update in the
                # regular two-site loop, so append the one missing edge bond.
                if bonds == expected[:-1]:
                    bonds = bonds + expected[-1:]
                if bonds != expected:
                    raise ValueError("C++ cursor returned an incompatible bond schedule")
                self.cursor_calls += 1
                self.cursor_steps += len(bonds)
                return bonds
            except Exception:
                self.cursor_failures += 1
        bonds = tuple(range(max(0, int(n_sites) - 1)))
        return bonds if direction == "lr" else tuple(reversed(bonds))

    @property
    def stats(self):
        h_stats = self.hamiltonian_stack.stats
        comp_stats = h_stats.get("complementary_operator_stack") or {}
        return {
            "completed_sweeps": int(self.completed_sweeps),
            "valid_boundary_side": (
                self.hamiltonian_valid_boundary_side
                if self.hamiltonian_valid_boundary_side
                == self.norm_valid_boundary_side
                else None
            ),
            "hamiltonian_valid_boundary_side": self.hamiltonian_valid_boundary_side,
            "norm_valid_boundary_side": self.norm_valid_boundary_side,
            "last_reused_prebuilt_side": self.last_reused_prebuilt_side,
            "environment_rebuilds": int(self.environment_rebuilds),
            "boundary_side_reuses": int(self.boundary_side_reuses),
            "boundary_side_rebuilds": int(self.boundary_side_rebuilds),
            "norm_boundary_side_reuses": int(self.norm_boundary_side_reuses),
            "hamiltonian_numeric_refreshes": int(
                self.hamiltonian_numeric_refreshes
            ),
            "cursor_backend": (
                "su2_moving_environment"
                if self.su2_moving_environment is not None
                else "cpp_moving_environment"
                if self.cursor_owner is not None
                else "python"
            ),
            "cursor_calls": int(self.cursor_calls),
            "cursor_steps": int(self.cursor_steps),
            "cursor_failures": int(self.cursor_failures),
            "hamiltonian_boundary_entries": int(h_stats.get("size", 0)),
            "hamiltonian_boundary_advances": int(h_stats.get("advanced_entries", 0)),
            "complementary_operator_entries": int(comp_stats.get("n_entries", 0)),
            "complementary_operator_advances": int(comp_stats.get("advances", 0)),
            "moving_environment_cache": h_stats.get("moving_environment_cache"),
            "su2_moving_environment": (
                None
                if self.su2_moving_environment is None
                else self.su2_moving_environment.stats
            ),
            "su2_boundary_environment": (
                None
                if self.hamiltonian_stack.su2_boundary_environment is None
                else self.hamiltonian_stack.su2_boundary_environment.stats
            ),
        }


def sweep_once(
    sites,
    *,
    direction="lr",
    solver=None,
    local_operator=None,
    local_solver=None,
    post_split=None,
    mpo_factors=None,
    root_target_mpo_factors=None,
    local_solver_kwargs=None,
    local_guess_cache=None,
    initial_root_sites=None,
    bond_coupling="left",
    max_bond=None,
    max_bond_mode=None,
    cutoff=1e-10,
    retain_sector_topology=False,
    prefer_reduced_local_operator=False,
    canonical_local_norm=False,
    warm_start_bonds=False,
    compact_updates=False,
    mixer_zero_block_noise_scale=0.0,
    mixer_rng=None,
    record_post_update_energy=False,
    compute_final_expectation=True,
    state_average_root_environments=False,
    state_average_local_norm=False,
    store_orthonormal_renormalized_operators=False,
    renormalized_operator_cache=None,
    renormalized_operator_cache_max_size=256,
    renormalized_block_stack=None,
    norm_renormalized_block_stack=None,
    target_renormalized_block_stack=None,
    complementary_operator_families=None,
    identity_mpo_factors=None,
    reuse_prebuilt_boundary_side=None,
    reuse_prebuilt_norm_boundary_side=None,
    input_is_canonical=False,
    require_block_sparse_renormalized_operator_table=False,
    require_symbolic_renormalized_operators=False,
    bond_cursor=None,
    lifecycle_owner=None,
    profile=False,
    verbose=0,
):
    """
    Perform one minimal sweep over a chain of non-Abelian site tensors.

    Parameters
    ----------
    sites
        Sequence of neighboring rank-3 :class:`NonabelianTensor` site tensors.
    direction
        ``"lr"`` for left-to-right or ``"rl"`` for right-to-left.
    solver
        Optional callback applied at each bond. It may accept either
        ``solver(merged)`` or ``solver(bond, merged)`` and should return an
        optimized rank-4 two-site tensor.
    local_operator
        Optional local-operator callback/specification used to drive the
        built-in Davidson local solver. It may accept either
        ``local_operator(merged)`` or ``local_operator(bond, merged)`` and
        should return a local operator specification understood by
        :func:`solve_local_two_site`.
    mpo_factors
        Optional dense MPO factor list. When provided, the sweep builds a dense
        effective local operator from the current chain state at each bond.
    root_target_mpo_factors
        Optional MPO used only to rank/select multi-root local Davidson
        candidates, e.g. a local effective S^2 operator for spin-targeted
        state averaging.
    local_solver_kwargs
        Optional keyword arguments forwarded to the Davidson local solver.
    local_guess_cache
        Optional mapping from bond index to a previously optimized rank-4
        two-site tensor used as the initial guess for the local Davidson solve.
    initial_root_sites
        Optional list of per-root MPS site lists from the previous sweep. When
        state-averaged local solves return root-specific site pairs, these root
        chains are updated at every bond and returned as full root MPSs.
    bond_coupling, max_bond, max_bond_mode, cutoff, prefer_reduced_local_operator
        Passed through to :func:`two_site_update`.
    canonical_local_norm
        If True, use the standard local problem only when the explicit norm
        environment verifies that the active two-site norm is identity. The
        diagnostic value ``"force"`` skips that check and assumes identity.
    warm_start_bonds
        If True, reuse cached same-bond two-site tensors from earlier sweeps as
        Davidson initial guesses when no explicit ``guess`` is supplied.
    compact_updates
        If True, retain only scalar diagnostics for completed bond updates.
        The updated MPS remains available through ``result["sites"]``.
    mixer_zero_block_noise_scale, mixer_rng
        Optional tiny Gaussian noise used only to seed the *active two-site
        initial guess* on zero-valued local blocks. Unlike a global site-tensor
        mixer, this leaves the canonical chain/environment untouched.
    record_post_update_energy
        If True and ``mpo_factors`` are provided, record the full-chain MPO
        expectation value immediately after each bond update under
        ``update["local_objective"]["post_update_energy"]``.
    state_average_root_environments
        If True for multi-root MPO sweeps, rebuild the local effective
        Hamiltonian from each root MPS before the state-averaged SVD. This is
        slower than the default shared-environment path, but preserves the
        root-specific mixed-canonical centers required by sweep-based SA-DMRG.
    state_average_local_norm
        If True, build an explicit local norm operator for state-averaged
        local solves. The default keeps the legacy canonical-norm assumption
        because the generalized SA path is currently diagnostic.
    store_orthonormal_renormalized_operators
        If True, build the local effective Hamiltonian as a standard operator
        in an orthonormal reduced basis owned by the environment sweep. This is
        the block-DMRG style path: the norm operator is consumed while building
        the local renormalized operator, rather than passed to the local solver.
    renormalized_operator_cache
        Optional persistent cache for environment-owned orthonormalized
        renormalized-operator tables. Cache keys include the active
        environment identity and local-basis signature, so stale tables are not
        reused after an environment update.
    renormalized_operator_cache_max_size
        Maximum number of transformed local operator tables retained in the
        persistent cache. Oldest entries are pruned first.
    renormalized_block_stack, norm_renormalized_block_stack, target_renormalized_block_stack
        Optional persistent left/right boundary-stack owners.  Passing these
        from the sweep driver makes the renormalized block table a first-class
        sweep object instead of a per-sweep temporary.
    complementary_operator_families
        Optional block2-style complementary Hamiltonian families attached to
        the Hamiltonian renormalized block stack.
    identity_mpo_factors
        Optional prebuilt identity MPO cores used for the norm environment.
    reuse_prebuilt_boundary_side
        Optional side, ``"left"`` or ``"right"``, already valid in the
        persistent boundary stacks.  The sweep skips rebuilding that side and
        consumes it as a block2-like moving environment.
    require_block_sparse_renormalized_operator_table
        If True, transformed local operators must use the block-sparse
        renormalized-operator table.
    require_symbolic_renormalized_operators
        If True, Hamiltonian local operators must be assembled from symbolic
        renormalized boundary payloads rather than raw environment maps.
    require_cpp_owned_sweep
        If True, require the native owner to execute the complete half-sweep.
        Raise instead of falling back to Python bond callbacks.
    verbose
        Logging level. ``0`` is silent, ``1`` is sweep-level only, ``2`` also
        prints per-bond updates.

    Returns
    -------
    dict
        Dictionary with updated ``sites`` and the ordered per-bond ``updates``.
        Each bond update may include a ``local_objective`` payload reported by
        the solver callback.
    """
    input_mps = sites if isinstance(sites, MPS) else None
    sites = input_mps.sites if input_mps is not None else sites
    if len(sites) < 2:
        raise ValueError("sweep_once requires at least two site tensors.")
    if any(not isinstance(site, NonabelianTensor) or site.rank != 3 for site in sites):
        raise ValueError("sweep_once expects a sequence of rank-3 NonabelianTensor site tensors.")
    if solver is not None and local_operator is not None:
        raise ValueError("Specify only one of solver or local_operator for sweep_once.")
    if local_solver is not None and mpo_factors is None and local_operator is None:
        raise ValueError("local_solver requires mpo_factors or local_operator.")
    if post_split is not None and local_solver is None:
        raise ValueError("post_split requires local_solver.")
    if solver is not None and mpo_factors is not None:
        raise ValueError("Specify mpo_factors only when using the built-in local-operator path.")
    if local_operator is not None and mpo_factors is not None:
        raise ValueError("Specify only one of local_operator or mpo_factors for sweep_once.")
    if mpo_factors is not None and len(mpo_factors) != len(sites):
        raise ValueError("mpo_factors must match the number of site tensors.")
    if root_target_mpo_factors is not None and len(root_target_mpo_factors) != len(sites):
        raise ValueError("root_target_mpo_factors must match the number of site tensors.")

    direction = _normalize_direction(direction)
    if reuse_prebuilt_norm_boundary_side is None:
        reuse_prebuilt_norm_boundary_side = reuse_prebuilt_boundary_side
    absorb = "right" if direction == "lr" else "left"
    expected_bonds = tuple(
        range(len(sites) - 1)
        if direction == "lr"
        else range(len(sites) - 2, -1, -1)
    )
    cpp_claimed_bonds = (
        lifecycle_owner is not None
        and hasattr(lifecycle_owner, "claim_next_bond")
    )
    if cpp_claimed_bonds:
        def claimed_bonds():
            for expected in expected_bonds:
                bond = int(lifecycle_owner.claim_next_bond())
                if bond != expected:
                    raise ValueError(
                        f"C++ half-sweep cursor expected bond {expected}, got {bond}."
                    )
                yield bond
            trailing = int(lifecycle_owner.claim_next_bond())
            if trailing != -1:
                raise ValueError(
                    f"C++ half-sweep cursor returned trailing bond {trailing}."
                )

        bonds = claimed_bonds()
    else:
        bonds = (
            tuple(int(bond) for bond in bond_cursor(direction, len(sites)))
            if bond_cursor is not None
            else expected_bonds
        )
        if bonds != expected_bonds:
            raise ValueError(
                f"Invalid {direction!r} half-sweep cursor: expected {expected_bonds}, got {bonds}."
            )

    cpp_state_average_owned = bool(
        lifecycle_owner is not None
        and lifecycle_owner.su2_moving_environment is not None
        and int(
            getattr(
                lifecycle_owner.su2_moving_environment,
                "state_average_roots",
                0,
            )
        ) > 1
    )
    cpp_state_already_owned = bool(
        lifecycle_owner is not None
        and lifecycle_owner.su2_moving_environment is not None
        and (
            getattr(lifecycle_owner, "cpp_state_owned", False)
            or cpp_state_average_owned
        )
    )
    updated_sites = (
        list(sites)
        if cpp_state_already_owned
        else [site.copy() for site in sites]
    )
    if mpo_factors is not None and not cpp_state_already_owned:
        if reuse_prebuilt_boundary_side is None and not input_is_canonical:
            canonical_center = min(1, len(updated_sites) - 1) if direction == "lr" else max(0, len(updated_sites) - 2)
            updated_sites = mixed_canonicalize_sites(
                updated_sites,
                canonical_center,
                max_bond=None,
                cutoff=0.0,
                max_bond_mode=max_bond_mode or "states",
                bond_coupling=bond_coupling,
                retain_sector_topology=retain_sector_topology,
            )
            assert_mixed_canonical_sites(updated_sites, canonical_center)
    local_solver_kwargs = dict(local_solver_kwargs or {})
    nlocal_states = int(local_solver_kwargs.get("nstates", 1))
    use_root_environment_path = bool(
        state_average_root_environments and mpo_factors is not None and nlocal_states > 1
    )
    if cpp_state_average_owned:
        root_sites = None
    elif initial_root_sites is not None:
        root_sites = [
            [site.copy() for site in root]
            for root in initial_root_sites
        ]
    elif nlocal_states > 1:
        root_sites = [
            [site.copy() for site in updated_sites]
            for _ in range(nlocal_states)
        ]
    else:
        root_sites = None
    root_center_tensor = None
    root_center_bond = None
    if root_sites is not None and mpo_factors is not None:
        initial_center = min(1, len(updated_sites) - 1) if direction == "lr" else max(0, len(updated_sites) - 2)
        root_sites = [
            mixed_canonicalize_sites(
                sites_for_root,
                initial_center,
                max_bond=None,
                cutoff=0.0,
                max_bond_mode=max_bond_mode or "states",
                bond_coupling=bond_coupling,
                retain_sector_topology=retain_sector_topology,
            )
            for sites_for_root in root_sites
        ]
    if lifecycle_owner is not None and not use_root_environment_path:
        split_owner = lifecycle_owner.su2_moving_environment
        if (
            not cpp_state_already_owned
            and split_owner is not None
            and hasattr(split_owner, "install_mps")
        ):
            split_owner.install_mps(updated_sites)
    local_guess_cache = dict(local_guess_cache or {})
    next_local_guess_cache = {}
    if max_bond_mode is None:
        # In the reduced non-Abelian representation, ``max_bond`` is most useful
        # as a multiplet count. Counting full state degeneracies makes moderate
        # SU(2) bond dimensions overly aggressive and noticeably degrades DMRG.
        max_bond_mode = "reduced"
    if (local_operator is not None or mpo_factors is not None) and "couple_physical" not in local_solver_kwargs:
        # The uncoupled physical-leg path is currently faster than the coupled
        # basis path for the non-Abelian MPO sweeps in this codebase.
        local_solver_kwargs["couple_physical"] = False
    updates = []
    if renormalized_operator_cache is None:
        renormalized_operator_cache = RenormalizedOperatorStack(
            max_size=renormalized_operator_cache_max_size,
        )
    elif isinstance(renormalized_operator_cache, RenormalizedOperatorStack):
        renormalized_operator_cache.max_size = int(renormalized_operator_cache_max_size)
    renormalized_operator_cache_max_size = int(renormalized_operator_cache_max_size)
    force_canonical_local_norm = canonical_local_norm is True or str(canonical_local_norm).lower() in {
        "force", "forced", "unsafe"
    }
    if mpo_factors is not None and not use_root_environment_path:
        if renormalized_block_stack is None:
            renormalized_block_stack = RenormalizedBlockStack(
                namespace="hamiltonian",
                complementary_operator_families=complementary_operator_families,
            )
        elif complementary_operator_families is not None:
            renormalized_block_stack.set_complementary_operator_families(
                complementary_operator_families
            )
        if force_canonical_local_norm:
            norm_renormalized_block_stack = None
        elif norm_renormalized_block_stack is None:
            norm_renormalized_block_stack = RenormalizedBlockStack(namespace="norm")
    else:
        renormalized_block_stack = None
        norm_renormalized_block_stack = None
    if root_target_mpo_factors is not None and not use_root_environment_path:
        if target_renormalized_block_stack is None:
            target_renormalized_block_stack = RenormalizedBlockStack(namespace="target")
    else:
        target_renormalized_block_stack = None
    env_sweep = None
    norm_env_sweep = None
    target_env_sweep = None
    timing = {
        "environment_build": 0.0,
        "environment_build_hamiltonian": 0.0,
        "environment_build_norm": 0.0,
        "environment_build_target": 0.0,
        "bond_operator": 0.0,
        "two_site_update": 0.0,
        "environment_advance": 0.0,
        "post_update_energy": 0.0,
        "update_merge_expand": 0.0,
        "update_operator_factory": 0.0,
        "update_local_solve": 0.0,
        "update_optimized_expand": 0.0,
        "update_svd": 0.0,
        "local_davidson": 0.0,
        "local_matvec": 0.0,
        "local_residual": 0.0,
    } if profile else None
    sweep_t0 = time.perf_counter() if profile else None
    if profile:
        rank_coupled_real_term_coalesce_stats(reset=True)
    owned_sweep_candidate = (
        lifecycle_owner is not None
        and solver is None
        and local_operator is None
        and local_solver is None
        and post_split is None
        and mpo_factors is not None
        and root_sites is None
        and root_target_mpo_factors is None
        and not force_canonical_local_norm
        and str(max_bond_mode).lower() == "reduced"
        and (
            int(nlocal_states) == 1
            or int(
                getattr(split_owner, "state_average_roots", 0)
            ) == int(nlocal_states)
        )
        and float(mixer_zero_block_noise_scale) == 0.0
        and not record_post_update_energy
        and split_owner is not None
        and callable(
            getattr(split_owner, "owned_half_sweep_ready", None)
        )
        and callable(
            getattr(split_owner, "execute_owned_half_sweep", None)
        )
    )
    if (
        owned_sweep_candidate
        and callable(
            getattr(split_owner, "prepare_owned_half_sweep", None)
        )
    ):
        owned_prepare_started = time.perf_counter() if profile else None
        split_owner.prepare_owned_half_sweep()
        if profile:
            timing["environment_build"] += (
                time.perf_counter() - owned_prepare_started
            )
    owned_half_sweep_readiness_code = (
        int(split_owner.owned_half_sweep_readiness_code())
        if owned_sweep_candidate
        and callable(
            getattr(
                split_owner,
                "owned_half_sweep_readiness_code",
                None,
            )
        )
        else None
    )
    owned_sweep = bool(
        owned_sweep_candidate
        and (
            owned_half_sweep_readiness_code == 0
            if owned_half_sweep_readiness_code is not None
            else split_owner.owned_half_sweep_ready()
        )
    )
    if owned_sweep:
        records = split_owner.execute_owned_half_sweep(
            cutoff=float(cutoff),
            max_bond=max_bond,
            max_bond_mode=max_bond_mode,
            retain_sector_topology=bool(retain_sector_topology),
            projection_tolerance=max(
                float(local_solver_kwargs.get("lindep", 1.0e-12)),
                1.0e-12,
            ),
            max_component_elements=4 * 1024 * 1024,
            max_transform_elements=4 * 1024 * 1024,
            davidson_tolerance=float(
                local_solver_kwargs.get("tol", 1.0e-8)
            ),
            max_iterations=int(
                local_solver_kwargs.get("itermax", 100)
            ),
            max_space=local_solver_kwargs.get("max_space"),
            workspace_budget_bytes=32 * 1024 * 1024,
            workspace_basis_arrays=3,
            accept_unconverged=True,
        )
        lifecycle_owner.cpp_state_owned = True
        for record_index, record in enumerate(records):
            bond = int(record["bond"])
            packed = record["split"]
            solve_record = record["solve"]
            objective = {
                "energy": float(solve_record["energy"]),
                "metric": float(solve_record["residual_norm"]),
                "residual": float(solve_record["residual_norm"]),
                "davidson_iterations": int(solve_record["iterations"]),
                "davidson_converged": bool(solve_record["converged"]),
                "subspace_dim": int(solve_record["basis_size"]),
                "restarts": int(solve_record["restarts"]),
                "packed_dimension": int(
                    solve_record["parent_dimension"]
                ),
                "orthonormalized_dim": int(
                    solve_record["orthonormal_dimension"]
                ),
                "matvec_count": int(solve_record["matvec_calls"]),
                "requested_max_space": int(
                    solve_record["requested_max_space"]
                ),
                "workspace_max_space": int(
                    solve_record["workspace_max_space"]
                ),
                "estimated_basis_workspace_bytes": int(
                    solve_record["estimated_basis_workspace_bytes"]
                ),
                "workspace_limited": bool(
                    solve_record["workspace_max_space"]
                    < solve_record["requested_max_space"]
                ),
                "cpp_davidson": True,
                "cpp_davidson_kind": str(solve_record["kind"]),
                "cpp_workspace_reused": bool(
                    solve_record["workspace_reused"]
                ),
                "direct_complementary_action_executor": True,
                "direct_cpp_metric": True,
                "canonical_reduced_basis": True,
                "cpp_active_solution_owned": True,
                "cpp_active_bond_split": True,
                "cpp_owned_half_sweep": True,
                "no_python_bond_callbacks": True,
                "solver_timing": {
                    "davidson": float(solve_record["solve_seconds"]),
                    "matvec": float(solve_record["solve_seconds"]),
                    "canonical_projection": float(
                        solve_record["projection_build_seconds"]
                    ),
                },
            }
            if bool(solve_record.get("state_average", False)):
                state_energies = [
                    float(value)
                    for value in solve_record.get("state_energies", ())
                ]
                state_residuals = [
                    float(value)
                    for value in solve_record.get(
                        "state_residual_norms",
                        (),
                    )
                ]
                state_weights = _state_average_weights(
                    local_solver_kwargs,
                    len(state_energies),
                )
                objective.update(
                    {
                        "state_energies": state_energies,
                        "state_residual_norms": state_residuals,
                        "state_average_weights": [
                            float(value) for value in state_weights
                        ],
                        "state_average_energy": float(
                            np.dot(state_weights, state_energies)
                        ),
                        "state_averaged_svd": True,
                        "block_davidson": True,
                        "cpp_block_davidson": True,
                        "cpp_state_average": True,
                        "cpp_post_truncation_expectation": bool(
                            record_index == len(records) - 1
                        ),
                        "effective_local_problem": (
                            "cpp_state_averaged_canonical_reduced"
                        ),
                    }
                )
            update = {
                "bond": bond,
                "trunc_err": float(packed["truncation_error"]),
                "kept_states": int(packed["kept_states"]),
                "local_objective": objective,
            }
            updates.append(_compact_sweep_update(bond, update))
        if len(records) != len(expected_bonds):
            raise RuntimeError(
                f"C++ owned half sweep completed {len(records)} bonds; "
                f"expected {len(expected_bonds)}."
            )
        last_bond = int(records[-1]["bond"]) if records else None
        if profile:
            timing["total"] = time.perf_counter() - sweep_t0
        return {
            "direction": direction,
            "sites": updated_sites,
            "mps": MPS(
                updated_sites,
                center=(
                    last_bond + 1
                    if direction == "lr"
                    else last_bond
                ),
            ),
            "root_sites": None,
            "root_center_tensor": None,
            "root_center_bond": None,
            "updates": updates,
            "local_guess_cache": {},
            "renormalized_operator_cache": renormalized_operator_cache,
            "renormalized_block_stack": renormalized_block_stack,
            "norm_renormalized_block_stack": norm_renormalized_block_stack,
            "target_renormalized_block_stack": target_renormalized_block_stack,
            "renormalized_operator_cache_size": len(
                renormalized_operator_cache
            ),
            "renormalized_operator_cache_stats": (
                renormalized_operator_cache.stats
                if isinstance(
                    renormalized_operator_cache,
                    RenormalizedOperatorStack,
                )
                else None
            ),
            "reused_prebuilt_boundary_side": reuse_prebuilt_boundary_side,
            "renormalized_block_stack_stats": (
                renormalized_block_stack.stats
                if renormalized_block_stack is not None
                else None
            ),
            "rank_coupled_real_term_coalesce_stats": (
                rank_coupled_real_term_coalesce_stats(reset=False)
                if profile
                else None
            ),
            "norm_renormalized_block_stack_stats": (
                norm_renormalized_block_stack.stats
                if norm_renormalized_block_stack is not None
                else None
            ),
            "target_renormalized_block_stack_stats": (
                target_renormalized_block_stack.stats
                if target_renormalized_block_stack is not None
                else None
            ),
            "final_mpo_numerator": None,
            "final_mpo_denominator": None,
            "terminal_local_energy": (
                None
                if not updates
                else (updates[-1].get("local_objective") or {}).get(
                    "energy"
                )
            ),
            "timing": timing,
            "cpp_owned_half_sweep": True,
            "owned_half_sweep_readiness_code": 0,
        }
    if mpo_factors is not None and not use_root_environment_path:
        t0 = time.perf_counter() if profile else None
        sub_t0 = time.perf_counter() if profile else None
        env_sweep = BlockSparseEnvironmentChain.build(
            updated_sites,
            mpo_factors,
            renormalized_blocks=renormalized_block_stack,
            require_symbolic_payloads=require_symbolic_renormalized_operators,
            sweep_direction=direction,
            reuse_prebuilt_boundary_side=reuse_prebuilt_boundary_side,
        ).start_sweep(direction)
        if profile:
            timing["environment_build_hamiltonian"] += time.perf_counter() - sub_t0
        if not force_canonical_local_norm and (
            nlocal_states <= 1
            or state_average_local_norm
            or store_orthonormal_renormalized_operators
        ):
            if identity_mpo_factors is None:
                identity_mpo_factors = _identity_mpo_factors_for_sites_and_mpo(
                    updated_sites,
                    mpo_factors,
                )
            sub_t0 = time.perf_counter() if profile else None
            norm_env_sweep = BlockSparseEnvironmentChain.build(
                updated_sites,
                identity_mpo_factors,
                renormalized_blocks=norm_renormalized_block_stack,
                sweep_direction=direction,
                reuse_prebuilt_boundary_side=reuse_prebuilt_norm_boundary_side,
            ).start_sweep(direction)
            if profile:
                timing["environment_build_norm"] += time.perf_counter() - sub_t0
        if root_target_mpo_factors is not None:
            sub_t0 = time.perf_counter() if profile else None
            target_env_sweep = BlockSparseEnvironmentChain.build(
                updated_sites,
                root_target_mpo_factors,
                renormalized_blocks=target_renormalized_block_stack,
                sweep_direction=direction,
                reuse_prebuilt_boundary_side=reuse_prebuilt_boundary_side,
            ).start_sweep(direction)
            if profile:
                timing["environment_build_target"] += time.perf_counter() - sub_t0
        if profile:
            timing["environment_build"] += time.perf_counter() - t0
    last_bond = None
    def execute_bond(bond):
        nonlocal last_bond, root_sites, root_center_tensor, root_center_bond
        last_bond = int(bond)
        if lifecycle_owner is not None and not cpp_claimed_bonds:
            lifecycle_owner.begin_bond(bond)
        bond_peak_rss_before = _peak_rss_bytes() if profile else None
        bond_current_rss_before = _current_rss_bytes() if profile else None
        bond_local_solver_kwargs = dict(local_solver_kwargs)
        merged_two_site = None
        if lifecycle_owner is not None and not use_root_environment_path:
            split_owner = lifecycle_owner.su2_moving_environment
            if split_owner is not None and hasattr(split_owner, "merge_active_bond"):
                merged_two_site = merge_mps_sites_from_packed(
                    updated_sites[bond],
                    updated_sites[bond + 1],
                    split_owner.merge_active_bond(),
                )
        guess_source = None
        if (
            warm_start_bonds
            and (local_operator is not None or mpo_factors is not None)
            and "guess" not in bond_local_solver_kwargs
        ):
            cached_guess = local_guess_cache.get(bond)
            if cached_guess is not None:
                bond_local_solver_kwargs["guess"] = cached_guess
                guess_source = "bond_cache"
        if (
            root_sites is not None
            and len(updated_sites) > 2
            and int(bond_local_solver_kwargs.get("nstates", 1)) > 1
        ):
            root_guesses = []
            for sites_for_root in root_sites:
                if bond + 1 >= len(sites_for_root):
                    continue
                try:
                    root_guesses.append(
                        merge_mps_sites(sites_for_root[bond], sites_for_root[bond + 1])
                    )
                except ValueError:
                    continue
            if root_guesses:
                bond_local_solver_kwargs["root_guesses"] = root_guesses
                if "guess" not in bond_local_solver_kwargs:
                    bond_local_solver_kwargs["guess"] = root_guesses[0]
        cpp_owned_solve = None
        cpp_owned_objective = None
        cpp_owner = (
            None
            if lifecycle_owner is None
            else lifecycle_owner.su2_moving_environment
        )
        cpp_owned_bond_loop = bool(
            getattr(lifecycle_owner, "cpp_owned_bond_loop", False)
        )
        if (
            cpp_owned_bond_loop
            and solver is None
            and local_operator is None
            and mpo_factors is not None
            and root_sites is None
            and root_target_mpo_factors is None
            and norm_env_sweep is not None
            and not force_canonical_local_norm
            and int(bond_local_solver_kwargs.get("nstates", 1)) == 1
            and float(mixer_zero_block_noise_scale) == 0.0
            and int(bond) >= 2
            and int(bond) + 3 < len(updated_sites)
            and cpp_owner is not None
            and callable(
                getattr(cpp_owner, "solve_active_bond_canonical", None)
            )
        ):
            cpp_started = time.perf_counter() if profile else None
            try:
                cpp_owned_solve = cpp_owner.solve_active_bond_canonical(
                    prepare_owned=True,
                    left_boundary_bond=int(bond),
                    right_boundary_bond=int(bond + 1),
                    dual_right_basis=bool(
                        getattr(
                            mpo_factors[bond + 1],
                            "normal_complementary_right_dual",
                            False,
                        )
                    ),
                    projection_tolerance=max(
                        float(
                            bond_local_solver_kwargs.get(
                                "lindep",
                                1.0e-12,
                            )
                        ),
                        1.0e-12,
                    ),
                    davidson_tolerance=float(
                        bond_local_solver_kwargs.get("tol", 1.0e-8)
                    ),
                    max_iterations=int(
                        bond_local_solver_kwargs.get("itermax", 100)
                    ),
                    max_space=bond_local_solver_kwargs.get("max_space"),
                    accept_unconverged=True,
                )
            except (RuntimeError, ValueError):
                cpp_owned_solve = None
            if (
                cpp_owned_solve is not None
                and bool(cpp_owned_solve.get("compatible", False))
            ):
                if not bool(cpp_owned_solve.get("accepted", False)):
                    raise RuntimeError(
                        "The C++ owned active-bond solve was not accepted."
                    )
                cpp_elapsed = (
                    time.perf_counter() - cpp_started
                    if profile
                    else None
                )
                projection_seconds = float(
                    cpp_owned_solve.get(
                        "projection_build_seconds",
                        0.0,
                    )
                )
                solve_seconds = float(
                    cpp_owned_solve.get(
                        "solve_seconds",
                        cpp_elapsed or 0.0,
                    )
                )
                cpp_owned_objective = {
                    "energy": float(cpp_owned_solve["energy"]),
                    "metric": float(
                        cpp_owned_solve["residual_norm"]
                    ),
                    "residual": float(
                        cpp_owned_solve["residual_norm"]
                    ),
                    "davidson_iterations": int(
                        cpp_owned_solve["iterations"]
                    ),
                    "davidson_converged": bool(
                        cpp_owned_solve["converged"]
                    ),
                    "subspace_dim": int(
                        cpp_owned_solve["basis_size"]
                    ),
                    "restarts": int(cpp_owned_solve["restarts"]),
                    "packed_dimension": int(
                        cpp_owned_solve["parent_dimension"]
                    ),
                    "orthonormalized_dim": int(
                        cpp_owned_solve["orthonormal_dimension"]
                    ),
                    "requested_max_space": int(
                        cpp_owned_solve["requested_max_space"]
                    ),
                    "workspace_max_space": int(
                        cpp_owned_solve["workspace_max_space"]
                    ),
                    "estimated_basis_workspace_bytes": int(
                        cpp_owned_solve[
                            "estimated_basis_workspace_bytes"
                        ]
                    ),
                    "cpp_davidson": True,
                    "cpp_davidson_kind": str(
                        cpp_owned_solve["kind"]
                    ),
                    "cpp_workspace_reused": bool(
                        cpp_owned_solve.get(
                            "workspace_reused",
                            False,
                        )
                    ),
                    "matvec_count": int(
                        cpp_owned_solve.get("matvec_calls", 0)
                    ),
                    "norm_matvec_count": 1,
                    "generalized_norm": False,
                    "tensor_davidson": True,
                    "packed_krylov": True,
                    "metric_orthonormal_krylov": True,
                    "canonical_reduced_basis": True,
                    "projected_problem": (
                        "canonical_reduced_standard"
                    ),
                    "preconditioner_mode": (
                        "projected_packed_diagonal"
                    ),
                    "direct_cpp_metric": True,
                    "direct_complementary_action_executor": True,
                    "cpp_active_solution_owned": True,
                    "cpp_owned_merged_guess": True,
                    "cpp_owned_local_problem": True,
                    "canonical_projection_reused": bool(
                        cpp_owned_solve.get(
                            "projection_reused",
                            False,
                        )
                    ),
                    "canonical_projection_components": int(
                        cpp_owned_solve.get(
                            "projection_components",
                            0,
                        )
                    ),
                    "canonical_projection_max_component_dimension": int(
                        cpp_owned_solve.get(
                            "projection_max_component_dimension",
                            0,
                        )
                    ),
                    "canonical_projection_transform_elements": int(
                        cpp_owned_solve.get(
                            "projection_transform_elements",
                            0,
                        )
                    ),
                    "canonical_projection_whitening_residual": float(
                        cpp_owned_solve.get(
                            "projection_whitening_residual",
                            0.0,
                        )
                    ),
                    "effective_local_problem": (
                        "cpp_owned_canonical_reduced"
                    ),
                }
                if profile:
                    cpp_owned_objective["solver_timing"] = {
                        "davidson": max(
                            0.0,
                            solve_seconds - projection_seconds,
                        ),
                        "matvec": max(
                            0.0,
                            solve_seconds - projection_seconds,
                        ),
                        "metric_setup": 0.0,
                        "canonical_projection": projection_seconds,
                        "projected": 0.0,
                        "precondition": 0.0,
                        "orthogonalize": 0.0,
                        "restart": 0.0,
                        "basis_update": 0.0,
                        "final_reference": 0.0,
                    }
        merged_solver = None
        if cpp_owned_objective is not None:
            def merged_solver(
                merged,
                objective=cpp_owned_objective,
            ):
                return merged, dict(objective)
        elif solver is not None:
            def merged_solver(merged, bond=bond, solver=solver):
                return _call_solver(solver, bond, merged)
        merged_local_operator = None
        if cpp_owned_objective is not None:
            pass
        elif local_operator is not None:
            def merged_local_operator(merged, bond=bond, local_operator=local_operator):
                return _call_solver(local_operator, bond, merged)
            merged_local_operator._is_local_operator_factory = True
        elif mpo_factors is not None:
            def merged_local_operator(
                merged,
                bond=bond,
                env_sweep=env_sweep,
                norm_env_sweep=norm_env_sweep,
                force_canonical_local_norm=force_canonical_local_norm,
                state_averaged_local=int(bond_local_solver_kwargs.get("nstates", 1)) > 1,
            ):
                t0 = time.perf_counter() if profile else None
                operator = None
                norm_operator = None
                if (
                    store_orthonormal_renormalized_operators
                    and not force_canonical_local_norm
                ):
                    operator = env_sweep.orthonormal_bond_operator(
                        bond,
                        merged,
                        norm_env_sweep,
                        tol=float(bond_local_solver_kwargs.get("tol", 1.0e-8)),
                        max_dim=bond_local_solver_kwargs.get("orthonormalize_generalized_dim"),
                        cache=renormalized_operator_cache,
                        require_block_sparse_table=require_block_sparse_renormalized_operator_table,
                        profile=profile,
                    )
                if operator is None:
                    operator = env_sweep.bond_operator(bond, merged)
                    norm_operator = None
                    if not force_canonical_local_norm and not (
                        state_averaged_local and not state_average_local_norm
                    ):
                        norm_blocks = norm_env_sweep.chain.renormalized_blocks
                        previous_metric_owner = (
                            None
                            if norm_blocks is None
                            else norm_blocks.su2_moving_environment
                        )
                        metric_owner = (
                            None
                            if lifecycle_owner is None
                            else lifecycle_owner.su2_moving_environment
                        )
                        if norm_blocks is not None:
                            norm_blocks.su2_moving_environment = metric_owner
                        try:
                            norm_operator = norm_env_sweep.bond_operator(
                                bond,
                                merged,
                            )
                        finally:
                            if norm_blocks is not None:
                                norm_blocks.su2_moving_environment = (
                                    previous_metric_owner
                                )
                norm_is_identity = (
                    True
                    if force_canonical_local_norm or operator is not None and norm_operator is None
                    else getattr(norm_operator, "identity_like", False)
                )
                result = TwoSiteEffectiveH(
                    operator=operator,
                    norm_operator=norm_operator,
                    canonical_norm=(
                        True
                        if norm_is_identity or (
                            state_averaged_local and not state_average_local_norm
                        )
                        else False
                    ),
                    name=f"bond-{bond}-effective-H",
                )
                if profile:
                    timing["bond_operator"] += time.perf_counter() - t0
                return result
            merged_local_operator._is_local_operator_factory = True
            if target_env_sweep is not None:
                def merged_root_target_operator(
                    merged,
                    bond=bond,
                    target_env_sweep=target_env_sweep,
                ):
                    return target_env_sweep.bond_operator(bond, merged)
                merged_root_target_operator._is_local_operator_factory = True
                bond_local_solver_kwargs.setdefault(
                    "root_target_operator",
                    merged_root_target_operator,
                )

        bond_local_solver = None
        if local_solver is not None:
            def bond_local_solver(
                merged,
                operator_spec,
                *,
                norm_operator=None,
                canonical_norm=False,
                profile=False,
                bond=bond,
                direction=direction,
                local_solver=local_solver,
                **kwargs,
            ):
                return local_solver(
                    bond,
                    direction,
                    merged,
                    operator_spec,
                    norm_operator=norm_operator,
                    canonical_norm=canonical_norm,
                    profile=profile,
                    **kwargs,
                )

        bond_post_split = None
        if post_split is not None:
            def bond_post_split(
                left,
                right,
                operator_spec,
                *,
                norm_operator=None,
                canonical_norm=False,
                bond=bond,
                direction=direction,
                post_split=post_split,
            ):
                return post_split(
                    bond,
                    direction,
                    left,
                    right,
                    operator_spec,
                    norm_operator=norm_operator,
                    canonical_norm=canonical_norm,
                )

        t0 = time.perf_counter() if profile else None
        if (
            use_root_environment_path
            and root_sites is not None
            and int(bond_local_solver_kwargs.get("nstates", 1)) > 1
        ):
            update = _state_average_root_environment_update(
                root_sites,
                bond,
                direction=direction,
                mpo_factors=mpo_factors,
                root_target_mpo_factors=root_target_mpo_factors,
                local_solver_kwargs=bond_local_solver_kwargs,
                bond_coupling=bond_coupling,
                max_bond=max_bond,
                max_bond_mode=max_bond_mode,
                cutoff=cutoff,
                retain_sector_topology=retain_sector_topology,
                absorb=absorb,
                profile=profile,
            )
        else:
            update = two_site_update(
                updated_sites[bond],
                updated_sites[bond + 1],
                merged_two_site=merged_two_site,
                solver=merged_solver,
                local_operator=merged_local_operator,
                local_solver=bond_local_solver,
                post_split=bond_post_split,
                local_solver_kwargs=bond_local_solver_kwargs,
                bond_coupling=bond_coupling,
                max_bond=max_bond,
                max_bond_mode=max_bond_mode,
                cutoff=cutoff,
                retain_sector_topology=retain_sector_topology,
                absorb=absorb,
                prefer_reduced_local_operator=prefer_reduced_local_operator,
                mixer_zero_block_noise_scale=mixer_zero_block_noise_scale,
                mixer_rng=mixer_rng,
                profile=profile,
                lifecycle_owner=lifecycle_owner,
            )
        if (
            lifecycle_owner is not None
            and use_root_environment_path
            and root_sites is not None
            and int(bond_local_solver_kwargs.get("nstates", 1)) > 1
        ):
            owner = lifecycle_owner.su2_moving_environment
            owner.mark_bond_solved()
            owner.mark_bond_split(
                int(update.get("kept_states", 0) or 0),
                float(
                    ((update.get("local_objective") or {}).get("update_timing") or {}).get(
                        "svd",
                        0.0,
                    )
                ),
            )
        if profile:
            timing["two_site_update"] += time.perf_counter() - t0
            update.setdefault("local_objective", {})
            update["local_objective"]["memory_profile"] = {
                "peak_rss_before_update_bytes": int(bond_peak_rss_before),
                "peak_rss_after_update_bytes": int(_peak_rss_bytes()),
                "current_rss_before_update_bytes": int(bond_current_rss_before),
                "current_rss_after_update_bytes": int(_current_rss_bytes()),
            }
            objective_timing = dict(update.get("local_objective") or {})
            update_timing = objective_timing.get("update_timing") or {}
            solver_timing = objective_timing.get("solver_timing") or {}
            timing["update_merge_expand"] += float(update_timing.get("merge_expand", 0.0))
            timing["update_operator_factory"] += float(update_timing.get("operator_factory", 0.0))
            timing["update_local_solve"] += float(update_timing.get("local_solve", 0.0))
            timing["update_optimized_expand"] += float(update_timing.get("optimized_expand", 0.0))
            timing["update_svd"] += float(update_timing.get("svd", 0.0))
            timing["local_davidson"] += float(solver_timing.get("davidson", 0.0))
            timing["local_matvec"] += float(solver_timing.get("matvec", 0.0))
            timing["local_residual"] += float(solver_timing.get("residual", 0.0))
        if (
            warm_start_bonds
            and (local_operator is not None or mpo_factors is not None)
            and isinstance(update.get("optimized"), NonabelianTensor)
            and not bool(
                (update.get("local_objective") or {}).get(
                    "cpp_active_solution_owned",
                    False,
                )
            )
        ):
            next_local_guess_cache[bond] = update["optimized"].copy()
        if guess_source is not None and update.get("local_guess_used"):
            update.setdefault("local_objective", {})
            update["local_objective"]["warm_start"] = guess_source
        update["left"] = normalize_site_tensor_layout(update["left"])
        update["right"] = normalize_site_tensor_layout(update["right"])
        updated_sites[bond] = update["left"]
        updated_sites[bond + 1] = update["right"]
        if root_sites is not None:
            root_pairs = update.get("root_site_pairs") or []
            optimized_roots = update.get("optimized_roots")
            if optimized_roots is not None:
                root_center_tensor = fuse_root_center_tensors([
                    root.copy() for root in optimized_roots
                    if isinstance(root, NonabelianTensor)
                ])
                root_center_bond = int(bond)
            if (
                int(bond_local_solver_kwargs.get("nstates", 1)) > 1
                and len(root_pairs) < len(root_sites)
            ):
                if optimized_roots is not None and root_pairs:
                    update.setdefault("local_objective", {})
                    update["local_objective"]["root_site_count_truncated"] = {
                        "from": int(len(root_sites)),
                        "to": int(len(root_pairs)),
                    }
                    root_sites = root_sites[: len(root_pairs)]
                else:
                    raise RuntimeError(
                        "State-averaged local update did not return one site pair per root; "
                        "refusing to duplicate the first root into missing roots."
                    )
            for root_idx, sites_for_root in enumerate(root_sites):
                if root_idx < len(root_pairs):
                    root_left, root_right = root_pairs[root_idx]
                else:
                    root_left, root_right = update["left"], update["right"]
                sites_for_root[bond] = normalize_site_tensor_layout(root_left).copy()
                sites_for_root[bond + 1] = normalize_site_tensor_layout(root_right).copy()
        if mpo_factors is not None:
            next_center = bond + 1 if direction == "lr" else bond
            if not any(
                (site.metadata or {}).get("canonical_metric")
                == "factorized_boundary"
                for site in updated_sites
            ):
                assert_mixed_canonical_sites(updated_sites, next_center)
        if env_sweep is not None:
            t0 = time.perf_counter() if profile else None
            env_sweep.advance_after_update(
                bond,
                update["left"],
                update["right"],
            )
            if norm_env_sweep is not None:
                norm_env_sweep.advance_after_update(
                    bond,
                    update["left"],
                    update["right"],
                )
            if target_env_sweep is not None:
                target_env_sweep.advance_after_update(
                    bond,
                    update["left"],
                    update["right"],
                )
            if profile:
                timing["environment_advance"] += time.perf_counter() - t0
                update.setdefault("local_objective", {})
                update["local_objective"].setdefault("memory_profile", {})[
                    "peak_rss_after_environment_bytes"
                ] = int(_peak_rss_bytes())
                update["local_objective"]["memory_profile"][
                    "current_rss_after_environment_bytes"
                ] = int(_current_rss_bytes())
        if isinstance(renormalized_operator_cache, RenormalizedOperatorStack):
            renormalized_operator_cache.prune()
        elif renormalized_operator_cache_max_size > 0:
            while len(renormalized_operator_cache) > renormalized_operator_cache_max_size:
                renormalized_operator_cache.pop(next(iter(renormalized_operator_cache)))
        if lifecycle_owner is not None:
            lifecycle_owner.commit_bond(update)
        if record_post_update_energy and mpo_factors is not None:
            t0 = time.perf_counter() if profile else None
            update.setdefault("local_objective", {})
            post_energy, post_error = _try_compute_state_energy_from_mpo(
                updated_sites,
                mpo_factors,
                identity_mpo_factors=identity_mpo_factors,
            )
            update["local_objective"]["post_update_energy"] = post_energy
            if post_error is not None:
                update["local_objective"]["post_update_energy_error"] = post_error
            if root_sites is not None:
                root_energies = []
                root_errors = []
                for sites_for_root in root_sites:
                    energy, error = _try_compute_state_energy_from_mpo(
                        sites_for_root,
                        mpo_factors,
                    )
                    root_energies.append(energy)
                    root_errors.append(error)
                update["local_objective"]["post_update_root_energies"] = root_energies
                if any(error is not None for error in root_errors):
                    update["local_objective"]["post_update_root_energy_errors"] = root_errors
            if profile:
                timing["post_update_energy"] += time.perf_counter() - t0
        if int(verbose) >= 2:
            _emit_verbose(_format_bond_update_line(bond, update), verbose=verbose)
        if compact_updates:
            updates.append(_compact_sweep_update(bond, update))
            del update
        else:
            updates.append({"bond": bond, **update})


    cpp_half_sweep_executor = (
        cpp_claimed_bonds
        and hasattr(lifecycle_owner, "execute_half_sweep")
    )
    if cpp_half_sweep_executor:
        completed_bonds = lifecycle_owner.execute_half_sweep(execute_bond)
        if completed_bonds != len(expected_bonds):
            raise ValueError(
                f"C++ half-sweep executor completed {completed_bonds} bonds; "
                f"expected {len(expected_bonds)}."
            )
    else:
        for bond in bonds:
            execute_bond(bond)

    if profile:
        timing["total"] = time.perf_counter() - sweep_t0

    return {
        "direction": direction,
        "sites": updated_sites,
        "mps": MPS(
            updated_sites,
            center=(last_bond + 1 if direction == "lr" else last_bond),
        ),
        "root_sites": root_sites,
        "root_center_tensor": root_center_tensor,
        "root_center_bond": root_center_bond,
        "updates": updates,
        "local_guess_cache": next_local_guess_cache,
        "renormalized_operator_cache": renormalized_operator_cache,
        "renormalized_block_stack": renormalized_block_stack,
        "norm_renormalized_block_stack": norm_renormalized_block_stack,
        "target_renormalized_block_stack": target_renormalized_block_stack,
        "renormalized_operator_cache_size": len(renormalized_operator_cache),
        "renormalized_operator_cache_stats": (
            renormalized_operator_cache.stats
            if isinstance(renormalized_operator_cache, RenormalizedOperatorStack)
            else None
        ),
        "reused_prebuilt_boundary_side": reuse_prebuilt_boundary_side,
        "renormalized_block_stack_stats": (
            renormalized_block_stack.stats if renormalized_block_stack is not None else None
        ),
        "rank_coupled_real_term_coalesce_stats": (
            rank_coupled_real_term_coalesce_stats(reset=False) if profile else None
        ),
        "norm_renormalized_block_stack_stats": (
            norm_renormalized_block_stack.stats
            if norm_renormalized_block_stack is not None
            else None
        ),
        "target_renormalized_block_stack_stats": (
            target_renormalized_block_stack.stats
            if target_renormalized_block_stack is not None
            else None
        ),
        "final_mpo_numerator": (
            env_sweep.final_expectation(updated_sites)
            if compute_final_expectation and env_sweep is not None
            else None
        ),
        "final_mpo_denominator": (
            norm_env_sweep.final_expectation(updated_sites)
            if compute_final_expectation and norm_env_sweep is not None
            else None
        ),
        "terminal_local_energy": (
            None
            if not updates
            else (updates[-1].get("local_objective") or {}).get("energy")
        ),
        "timing": timing,
        "cpp_owned_half_sweep": False,
        "owned_half_sweep_readiness_code": (
            owned_half_sweep_readiness_code
        ),
    }


def _single_root_solver_kwargs(local_solver_kwargs, *, guess):
    """
    Return local solver options for one root-specific SA subproblem.

    Parameters
    ----------
    local_solver_kwargs
        Multi-root solver keyword payload from the sweep driver.
    guess
        Active two-site tensor used as the initial local vector.

    Returns
    -------
    dict
        Keyword arguments suitable for :func:`solve_local_two_site` with
        ``nstates=1``.
    """

    kwargs = dict(local_solver_kwargs or {})
    for key in (
        "nstates",
        "weights",
        "root_guesses",
        "root_target_operator",
        "root_target_value",
        "root_target_tol",
        "root_selection_buffer",
        "root_projector_dim",
        "root_projector_dense_dim",
        "root_projector_block_dim",
        "root_projector_block_max_columns",
        "root_projector_block_offdiag_tol",
        "filter_coupled_boundary",
    ):
        kwargs.pop(key, None)
    kwargs["nstates"] = 1
    kwargs["guess"] = guess
    return kwargs


def _state_average_weights(local_solver_kwargs, nroots):
    """
    Normalize state-average weights for a local SA update.

    Parameters
    ----------
    local_solver_kwargs
        Local solver keyword payload.
    nroots
        Number of propagated root MPSs.

    Returns
    -------
    numpy.ndarray
        Normalized weights with length ``nroots``.
    """

    weights = local_solver_kwargs.get("weights")
    if weights is None:
        weights = np.ones(int(nroots), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size < int(nroots):
            weights = np.pad(weights, (0, int(nroots) - weights.size))
        elif weights.size > int(nroots):
            weights = weights[: int(nroots)]
    total = float(np.sum(weights))
    if abs(total) <= 1.0e-15:
        weights = np.ones(int(nroots), dtype=float)
        total = float(np.sum(weights))
    return weights / total


def _orthonormal_columns(columns, *, dim, tol=1.0e-10):
    """
    Build an orthonormal dense column basis from candidate vectors.

    Parameters
    ----------
    columns
        Iterable of one-dimensional candidate vectors.
    dim
        Full vector-space dimension.
    tol
        Linear-dependence threshold.

    Returns
    -------
    numpy.ndarray
        Matrix with orthonormal columns.
    """

    basis = []
    for column in columns:
        vec = np.asarray(column, dtype=complex).reshape(dim)
        for prev in basis:
            vec = vec - prev * np.vdot(prev, vec)
        norm = float(np.linalg.norm(vec))
        if norm > float(tol):
            basis.append(vec / norm)
    if not basis:
        return np.zeros((dim, 0), dtype=complex)
    return np.column_stack(basis)


def _dense_deflated_local_root(
    merged,
    operator,
    previous_roots,
    *,
    target_operator=None,
    target_value=None,
    target_tol=1.0e-6,
    tol=1.0e-10,
):
    """
    Solve one dense local root with orthogonality to previous local roots.

    Parameters
    ----------
    merged
        Active two-site tensor template.
    operator
        Local effective Hamiltonian.
    previous_roots
        Earlier optimized root tensors to project out in the current local
        packed basis.
    tol
        Numerical threshold for rank decisions.

    Returns
    -------
    tuple
        ``(optimized_tensor, objective)``.
    """

    op = _normalize_local_operator(operator)
    vec0, raw_layout = pack_two_site_state(merged)
    layout = (
        _operator_basis_for_layout(op, raw_layout)
        or two_site_state_basis(merged, layout=raw_layout)
    )
    vec0, _ = pack_two_site_state(merged, layout=layout)
    dim = int(np.asarray(vec0).reshape(-1).size)
    op_resolved, _diag = _resolve_davidson_operator(op, merged, layout)
    H = (
        _materialize_local_matrix(op_resolved, dim)
        if callable(op_resolved)
        else np.asarray(op_resolved, dtype=complex)
    )
    H = 0.5 * (H + H.conj().T)

    packed_previous = []
    for root in previous_roots:
        try:
            root_vec, _ = pack_two_site_state(root, layout=layout)
        except ValueError:
            continue
        packed_previous.append(root_vec)
    Qprev = _orthonormal_columns(packed_previous, dim=dim, tol=tol)
    Qtarget = None
    target_values = None
    if target_operator is not None and target_value is not None:
        Qtarget, target_values = _target_projector_basis_by_blocks(
            target_operator,
            merged,
            layout,
            target_value=target_value,
            target_tol=target_tol,
            min_dim=1,
            max_block_size=max(entry.size for entry in _layout_entries(layout)),
            max_columns=dim,
        )
        if Qtarget is None:
            raise ValueError(
                "Root-specific target operator does not expose a block-local target-spin subspace."
            )
    candidates = np.eye(dim, dtype=complex) if Qtarget is None else Qtarget
    if Qprev.shape[1] > 0:
        candidates = candidates - Qprev @ (Qprev.conj().T @ candidates)
    Q = _orthonormal_columns(
        [candidates[:, index] for index in range(candidates.shape[1])],
        dim=dim,
        tol=tol,
    )
    if Q.shape[1] <= 0:
        raise ValueError("Deflated local root subspace is empty.")
    Hproj = Q.conj().T @ H @ Q
    evals, evecs = np.linalg.eigh(0.5 * (Hproj + Hproj.conj().T))
    idx = int(np.argmin(np.real(evals)))
    vec = Q @ evecs[:, idx]
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-15:
        raise ValueError("Dense deflated local root has zero norm.")
    vec = vec / norm
    energy = float(np.real(evals[idx]))
    optimized = unpack_two_site_state(vec, merged, layout=layout)
    residual = float(np.linalg.norm(H @ vec - energy * vec))
    return optimized, {
        "energy": energy,
        "residual": residual,
        "dense_deflated": True,
        "deflated_roots": int(Qprev.shape[1]),
        "target_projector_dim": None if Qtarget is None else int(Qtarget.shape[1]),
        "target_projector_values": (
            None if target_values is None else [float(x) for x in target_values]
        ),
        "subspace_dim": int(Q.shape[1]),
        "layout_size": int(dim),
    }


def _state_average_root_environment_update(
    root_sites,
    bond,
    *,
    direction,
    mpo_factors,
    root_target_mpo_factors,
    local_solver_kwargs,
    bond_coupling,
    max_bond,
    max_bond_mode,
    cutoff,
    retain_sector_topology,
    absorb,
    profile=False,
):
    """
    Update one SA bond using root-specific effective Hamiltonians.

    This follows the root-propagating SA-DMRG structure: each root MPS is kept
    in mixed-canonical gauge at the active bond, each local root is optimized in
    its own environment, and the averaged density matrix/SVD is built from the
    optimized root center tensors.

    Parameters
    ----------
    root_sites
        Mutable list of per-root site tensor lists.
    bond
        Active left-site index.
    direction
        Sweep direction, ``"lr"`` or ``"rl"``.
    mpo_factors
        Hamiltonian MPO factors.
    root_target_mpo_factors
        Optional target-operator MPO factors, normally ``S^2``, used to keep
        every root-specific local solve inside the requested spin sector.
    local_solver_kwargs
        Multi-root local solver options.
    bond_coupling, max_bond, max_bond_mode, cutoff, absorb
        State-averaged SVD options.
    profile
        If True, record coarse per-root timing.

    Returns
    -------
    dict
        Update payload compatible with :func:`two_site_update`.
    """

    nroots = int(local_solver_kwargs.get("nstates", len(root_sites)))
    nroots = min(nroots, len(root_sites))
    weights = _state_average_weights(local_solver_kwargs, nroots)
    center = bond + 1 if direction == "lr" else bond
    optimized_roots = []
    root_objectives = []
    timing = {"canonicalize": 0.0, "environment": 0.0, "solve": 0.0} if profile else None

    for root_idx in range(nroots):
        canonical_sites = mixed_canonicalize_sites(
            root_sites[root_idx],
            center,
            max_bond=None,
            cutoff=0.0,
            max_bond_mode=max_bond_mode or "states",
            bond_coupling=bond_coupling,
            retain_sector_topology=retain_sector_topology,
        )
        root_sites[root_idx] = canonical_sites
        merged = merge_mps_sites(canonical_sites[bond], canonical_sites[bond + 1])
        t0 = time.perf_counter() if profile else None
        operator = BlockSparseEnvironmentChain.build(canonical_sites, mpo_factors).bond_operator(
            bond,
            merged,
        )
        target_operator = None
        if root_target_mpo_factors is not None:
            target_operator = BlockSparseEnvironmentChain.build(
                canonical_sites,
                root_target_mpo_factors,
            ).bond_operator(bond, merged)
        if profile:
            timing["environment"] += time.perf_counter() - t0

        t0 = time.perf_counter() if profile else None
        optimized, objective = _dense_deflated_local_root(
            merged,
            operator,
            optimized_roots,
            target_operator=target_operator,
            target_value=local_solver_kwargs.get("root_target_value"),
            target_tol=float(local_solver_kwargs.get("root_target_tol", 1.0e-6)),
            tol=float(local_solver_kwargs.get("tol", 1.0e-10)),
        )
        optimized = _expand_two_site_support(
            canonical_sites[bond],
            canonical_sites[bond + 1],
            optimized,
        )
        if profile:
            timing["solve"] += time.perf_counter() - t0
        optimized_roots.append(optimized)
        root_objectives.append(dict(objective))

    left, right, singular_values, trunc_err, kept, root_site_pairs = state_averaged_svd_two_site(
        optimized_roots,
        weights,
        bond_coupling=bond_coupling,
        max_bond=max_bond,
        max_bond_mode=max_bond_mode,
        cutoff=cutoff,
        retain_sector_topology=retain_sector_topology,
        absorb=absorb,
    )
    kept_states = sum(
        sector_state_weight(q_mid) * int(block.shape[0])
        for q_mid, block in singular_values.items()
    )
    state_energies = [
        float(obj["energy"])
        for obj in root_objectives
        if "energy" in obj
    ]
    local_objective = {
        "effective_local_problem": "state_averaged_root_environment_davidson",
        "state_averaged_svd": True,
        "state_average_weights": [float(x) for x in weights],
        "state_energies": state_energies,
        "state_average_energy": float(np.dot(weights[: len(state_energies)], state_energies))
        if len(state_energies) == len(weights)
        else None,
        "root_objectives": root_objectives,
        "trunc_err": float(trunc_err),
        "kept": {key: list(value) for key, value in kept.items()} if isinstance(kept, dict) else int(kept),
        "kept_states": int(kept_states),
    }
    if timing is not None:
        local_objective["root_environment_timing"] = timing

    return {
        "merged": merge_mps_sites(root_sites[0][bond], root_sites[0][bond + 1]),
        "optimized": optimized_roots[0],
        "optimized_roots": optimized_roots,
        "root_site_pairs": root_site_pairs,
        "left": left,
        "right": right,
        "singular_values": singular_values,
        "trunc_err": trunc_err,
        "kept": kept,
        "kept_states": kept_states,
        "local_objective": local_objective,
        "local_guess_used": False,
    }


def _default_sweep_measure(sweep_result):
    updates = sweep_result["updates"]
    if not updates:
        return 0.0
    return max(update["trunc_err"] for update in updates)


def _state_average_energy(history_entry):
    weights = np.asarray(
        history_entry.get("state_average_weights", []),
        dtype=float,
    ).reshape(-1)
    energies = history_entry.get("state_energies")
    if energies is None:
        energies = history_entry.get("target_state_energies")
    if energies is not None:
        energies = np.asarray(energies, dtype=float).reshape(-1)
        if weights.size == energies.size and weights.size:
            return float(np.dot(weights, energies))
        return float(np.mean(energies))
    if "state_average_energy" in history_entry:
        return float(history_entry["state_average_energy"])
    if "energy" in history_entry:
        return float(history_entry["energy"])
    return None


def _state_average_energy_delta(prev_entry, curr_entry):
    prev = _state_average_energy(prev_entry)
    curr = _state_average_energy(curr_entry)
    if prev is None or curr is None:
        return None
    return abs(curr - prev)


def _summarize_objectives(updates):
    bond_objectives = []
    energies = []
    metrics = []
    values = []
    cache_hits = 0
    cache_lookups = 0
    renormalized_operator_storages = set()
    renormalized_operator_table_kinds = set()
    terminal_state_average = None
    for update in updates:
        objective = dict(update.get("local_objective") or {})
        if not objective:
            continue
        bond_objectives.append({"bond": update["bond"], **objective})
        if "energy" in objective:
            energies.append(float(objective["energy"]))
        if "metric" in objective:
            metrics.append(float(objective["metric"]))
        if "value" in objective:
            values.append(float(objective["value"]))
        if "renormalized_operator_cache_hit" in objective:
            cache_lookups += 1
            if objective.get("renormalized_operator_cache_hit"):
                cache_hits += 1
        storage = objective.get("renormalized_operator_storage")
        if storage is not None:
            renormalized_operator_storages.add(str(storage))
        table_stats = objective.get("renormalized_operator_table_stats") or {}
        table_kind = table_stats.get("kind")
        if table_kind is not None:
            renormalized_operator_table_kinds.add(str(table_kind))
        if objective.get("state_energies") is not None:
            terminal_state_average = objective

    summary = {"bond_objectives": bond_objectives}
    if energies:
        summary["objective_energy"] = sum(energies) / len(energies)
    if metrics:
        summary["objective_metric"] = sum(metrics) / len(metrics)
    if values:
        summary["objective_value"] = sum(values) / len(values)
    if cache_lookups:
        summary["renormalized_operator_cache_hits"] = int(cache_hits)
        summary["renormalized_operator_cache_lookups"] = int(cache_lookups)
    if renormalized_operator_storages:
        summary["renormalized_operator_storages"] = sorted(renormalized_operator_storages)
    if renormalized_operator_table_kinds:
        summary["renormalized_operator_table_kinds"] = sorted(renormalized_operator_table_kinds)
    if terminal_state_average is not None:
        state_energies = [
            float(value)
            for value in terminal_state_average["state_energies"]
        ]
        state_weights = np.asarray(
            terminal_state_average.get(
                "state_average_weights",
                np.ones(len(state_energies)) / len(state_energies),
            ),
            dtype=float,
        ).reshape(-1)
        if state_weights.size < len(state_energies):
            state_weights = np.pad(
                state_weights,
                (0, len(state_energies) - state_weights.size),
            )
        elif state_weights.size > len(state_energies):
            state_weights = state_weights[: len(state_energies)]
        weight_sum = float(np.sum(state_weights))
        if weight_sum <= 0.0:
            state_weights = np.ones(len(state_energies)) / len(state_energies)
        else:
            state_weights = state_weights / weight_sum
        summary["state_energies"] = state_energies
        summary["state_average_weights"] = [float(value) for value in state_weights]
        summary["state_average_energy"] = float(
            np.dot(state_weights, state_energies)
        )
        summary["energy"] = float(state_energies[0])
    return summary


def _compute_state_energy_from_mpo(sites, mpo_factors, *, identity_mpo_factors=None):
    """
    Return the normalized MPO expectation value for one MPS.

    :param sites: MPS site tensors.
    :param mpo_factors: Hamiltonian MPO cores.
    :param identity_mpo_factors: Optional prebuilt identity MPO cores.
    :returns: Real normalized expectation value.
    """

    numerator = contract_chain_expectation(sites, mpo_factors)
    if identity_mpo_factors is None:
        identity_mpo_factors = _identity_mpo_factors_for_sites_and_mpo(sites, mpo_factors)
    denominator = contract_chain_expectation(
        sites,
        identity_mpo_factors,
    )
    denom = float(np.real(denominator))
    if abs(denom) < 1e-15:
        raise ValueError("State norm is numerically zero while computing sweep energy.")
    return float(np.real(numerator / denominator))


def _try_compute_state_energy_from_mpo(sites, mpo_factors, *, identity_mpo_factors=None):
    try:
        return _compute_state_energy_from_mpo(
            sites,
            mpo_factors,
            identity_mpo_factors=identity_mpo_factors,
        ), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _infer_converged_from_objectives(
    history,
    *,
    energy_tol=1e-10,
    metric_tol=1e-10,
):
    """
    Infer convergence from the recent sweep objective history.

    This is used only when ``conv_tol`` is not supplied explicitly. It avoids
    marking bare truncation-only sweeps as converged, but lets fully
    objective-driven runs report convergence once the last two sweeps have
    stabilized in energy and the final objective metric is tiny.
    """
    if len(history) < 2:
        return False
    prev = history[-2]
    curr = history[-1]
    if "energy" not in prev or "energy" not in curr:
        return False
    if "objective_metric" not in curr:
        return False
    if abs(float(curr["objective_metric"])) > metric_tol:
        return False
    return abs(float(curr["energy"]) - float(prev["energy"])) <= energy_tol


def _resolve_local_solver_schedule(schedule, *, sweep_idx, direction, history):
    if schedule is None:
        return {}
    if callable(schedule):
        try:
            resolved = schedule(
                sweep_idx=sweep_idx,
                direction=direction,
                history=tuple(history),
            )
        except TypeError:
            resolved = schedule(sweep_idx, direction)
        if resolved is None:
            return {}
        return dict(resolved)
    if isinstance(schedule, dict):
        return dict(schedule)
    schedule = list(schedule)
    if not schedule:
        return {}
    resolved = schedule[min(int(sweep_idx), len(schedule) - 1)]
    if resolved is None:
        return {}
    return dict(resolved)


def _default_mpo_local_solver_schedule(*, sweep_idx, direction, history):
    _ = direction, history
    if int(sweep_idx) <= 0:
        return {
            "tol": 1e-10,
            "itermax": 80,
            "max_space": 128,
        }
    return {
        "tol": 1e-10,
        "itermax": 80,
        "max_space": 128,
    }


def run_sweeps(
    sites,
    *,
    nsweeps=1,
    start_direction="lr",
    alternate=True,
    solver=None,
    local_operator=None,
    mpo_factors=None,
    root_target_mpo_factors=None,
    local_solver_kwargs=None,
    local_solver_schedule=None,
    initial_root_sites=None,
    bond_coupling="left",
    max_bond=None,
    max_bond_mode=None,
    cutoff=1e-10,
    retain_sector_topology=False,
    conv_tol=None,
    converge_on_full_sweeps=False,
    measure=None,
    prefer_reduced_local_operator=False,
    canonical_local_norm=False,
    warm_start_bonds=False,
    compact_history_updates=False,
    mixer_zero_block_noise_scale=0.0,
    mixer_zero_block_noise_seed=None,
    mixer_nsweeps=1,
    record_post_update_energy=False,
    compute_final_expectation=True,
    evaluate_root_energies_each_sweep=True,
    state_average_root_environments=False,
    state_average_local_norm=False,
    store_orthonormal_renormalized_operators=False,
    renormalized_operator_cache=None,
    renormalized_operator_cache_max_size=256,
    require_block_sparse_renormalized_operator_table=False,
    require_symbolic_renormalized_operators=False,
    require_cpp_owned_sweeps=False,
    complementary_operator_families=None,
    materialize_complementary_family_operator_tables=True,
    su2_moving_environment=None,
    profile=False,
    verbose=0,
):
    """
    Run repeated non-Abelian sweeps with simple history/convergence tracking.

    Parameters
    ----------
    sites
        Sequence of rank-3 :class:`NonabelianTensor` site tensors.
    nsweeps
        Maximum number of sweeps to perform.
    start_direction
        Initial sweep direction, ``"lr"`` or ``"rl"``.
    alternate
        If True, alternate the sweep direction after each pass.
    solver, local_operator, mpo_factors, local_solver_kwargs, local_solver_schedule,
    initial_root_sites, bond_coupling, max_bond, max_bond_mode, cutoff, root_target_mpo_factors,
    prefer_reduced_local_operator, canonical_local_norm, warm_start_bonds,
    compact_history_updates
        Passed through to :func:`sweep_once`.
    require_cpp_owned_sweeps
        Require every half-sweep to be executed by the native C++ owner.
        This disables silent fallback to Python bond callbacks.
    conv_tol
        Optional convergence tolerance applied to ``measure(sweep_result)``.
    measure
        Optional callable returning a scalar diagnostic for one sweep result.
        Defaults to the maximum per-bond truncation error of that sweep.
    mixer_zero_block_noise_scale
        Optional tiny Gaussian noise used to seed the active two-site initial
        guess on symmetry-allowed zero-valued local blocks during the first few
        sweeps. This acts like a lightweight mixer for exact product starts
        while preserving the canonical chain/environment outside the active
        center.
    mixer_zero_block_noise_seed
        Optional seed for the mixer noise.
    mixer_nsweeps
        Number of initial sweeps on which the zero-block mixer is applied.
    record_post_update_energy
        If True and ``mpo_factors`` are provided, store the post-update chain
        energy after every bond update in the per-bond objective payload.
    evaluate_root_energies_each_sweep
        If False, skip full root-MPS MPO expectation evaluations during the
        sweep history. The final caller can still evaluate selected roots once.
    state_average_root_environments
        If True for multi-root MPO sweeps, use root-specific local Hamiltonian
        environments before the state-averaged SVD.
    state_average_local_norm
        If True, use explicit norm environments for state-averaged local solves.
    store_orthonormal_renormalized_operators
        If True, keep persistent renormalized boundary stacks and local
        orthonormal operator tables across sweeps.
    complementary_operator_families
        Optional block2-style complementary Hamiltonian families attached to
        the persistent Hamiltonian renormalized block stack.
    profile
        If True, attach a coarse timing breakdown to each sweep history entry.
    verbose
        Logging level. ``0`` is silent, ``1`` prints one summary line per
        sweep, and ``2`` additionally prints one line per bond update.

    Returns
    -------
    dict
        Dictionary with ``sites``, ``history``, ``converged``, ``last_direction``,
        and ``ncompleted``. History entries also include any per-bond objective
        payloads summarized into ``bond_objectives``. When ``mpo_factors`` are
        provided, ``energy`` is the true MPO expectation value of the current
        state while ``objective_energy`` keeps the sweep-averaged local solve
        trace. Without ``mpo_factors``, ``energy`` falls back to the objective
        trace if available.
    """
    input_multiroot = sites if isinstance(sites, MultiRootMPS) else None
    input_mps = sites if isinstance(sites, MPS) else None
    if input_multiroot is not None:
        target_sector = input_multiroot.target_sector
        if initial_root_sites is None:
            initial_root_sites = input_multiroot.root_site_lists()
        sites = input_multiroot.sites
    else:
        target_sector = input_mps.target_sector if input_mps is not None else None
        sites = input_mps.sites if input_mps is not None else sites
    if nsweeps < 1:
        raise ValueError("run_sweeps requires nsweeps >= 1.")

    direction = _normalize_direction(start_direction)
    full_sweep_end_direction = "rl" if direction == "lr" else "lr"
    measure_fn = _default_sweep_measure if measure is None else measure
    current_sites = [site.copy() for site in sites]
    current_center = (
        None
        if input_mps is None
        else input_mps.center
    )
    history = []
    converged = False
    best_sites = None
    best_center = None
    best_root_sites = None
    best_root_center_tensor = None
    best_root_center_bond = None
    best_state_energies = None
    best_energy = None
    last_root_sites = (
        [[site.copy() for site in root] for root in initial_root_sites]
        if initial_root_sites is not None
        else None
    )
    last_root_center_tensor = None
    last_root_center_bond = None
    last_state_energies = None
    local_guess_cache = {}
    mixer_zero_block_noise_scale = float(mixer_zero_block_noise_scale)
    mixer_nsweeps = int(mixer_nsweeps)
    mixer_rng = np.random.default_rng(mixer_zero_block_noise_seed)
    run_uses_root_environment_path = bool(
        state_average_root_environments
        and mpo_factors is not None
        and int((local_solver_kwargs or {}).get("nstates", 1)) > 1
    )
    force_canonical_local_norm = canonical_local_norm is True or str(canonical_local_norm).lower() in {
        "force", "forced", "unsafe"
    }
    moving_environment = None
    if mpo_factors is not None and not run_uses_root_environment_path:
        moving_environment = MovingEnvironment(
            current_sites,
            mpo_factors=mpo_factors,
            root_target_mpo_factors=root_target_mpo_factors,
            complementary_operator_families=complementary_operator_families,
            materialize_complementary_family_operator_tables=bool(
                materialize_complementary_family_operator_tables
            ),
            su2_moving_environment=su2_moving_environment,
            renormalized_operator_cache_max_size=renormalized_operator_cache_max_size,
        )
        if renormalized_operator_cache is not None:
            moving_environment.renormalized_operator_cache = renormalized_operator_cache
        if isinstance(moving_environment.renormalized_operator_cache, RenormalizedOperatorStack):
            moving_environment.renormalized_operator_cache.max_size = int(
                renormalized_operator_cache_max_size
            )
        identity_mpo_factors = moving_environment.identity_mpo_factors
        persistent_renormalized_block_stack = moving_environment.hamiltonian_stack
        persistent_norm_renormalized_block_stack = (
            None if force_canonical_local_norm else moving_environment.norm_stack
        )
        persistent_target_renormalized_block_stack = moving_environment.target_stack
        renormalized_operator_cache = moving_environment.renormalized_operator_cache
    else:
        renormalized_operator_cache = (
            renormalized_operator_cache
            if renormalized_operator_cache is not None
            else RenormalizedOperatorStack(max_size=renormalized_operator_cache_max_size)
        )
        if isinstance(renormalized_operator_cache, RenormalizedOperatorStack):
            renormalized_operator_cache.max_size = int(renormalized_operator_cache_max_size)
        persistent_renormalized_block_stack = None
        persistent_norm_renormalized_block_stack = None
        persistent_target_renormalized_block_stack = None
        identity_mpo_factors = (
            _identity_mpo_factors_for_sites_and_mpo(current_sites, mpo_factors)
            if mpo_factors is not None
            else None
        )

    for sweep_idx in range(int(nsweeps)):
        resolved_schedule = _resolve_local_solver_schedule(
            local_solver_schedule,
            sweep_idx=sweep_idx,
            direction=direction,
            history=history,
        )
        if local_solver_schedule is None and mpo_factors is not None:
            sweep_local_solver_kwargs = _default_mpo_local_solver_schedule(
                sweep_idx=sweep_idx,
                direction=direction,
                history=history,
            )
            sweep_local_solver_kwargs.update(local_solver_kwargs or {})
        else:
            sweep_local_solver_kwargs = dict(local_solver_kwargs or {})
            sweep_local_solver_kwargs.update(resolved_schedule)
        sweep_nlocal_states = int(sweep_local_solver_kwargs.get("nstates", 1))
        reuse_prebuilt_boundary_side = (
            moving_environment.reuse_side_for(direction)
            if moving_environment is not None
            else None
        )
        if moving_environment is not None:
            moving_environment.begin_half_sweep(
                direction,
                len(current_sites),
            )
        try:
            sweep_result = sweep_once(
                current_sites,
                direction=direction,
                solver=solver,
                local_operator=local_operator,
                mpo_factors=mpo_factors,
                root_target_mpo_factors=root_target_mpo_factors,
                local_solver_kwargs=sweep_local_solver_kwargs,
                local_guess_cache=local_guess_cache,
                initial_root_sites=last_root_sites,
                bond_coupling=bond_coupling,
                max_bond=max_bond,
                max_bond_mode=max_bond_mode,
                cutoff=cutoff,
                retain_sector_topology=retain_sector_topology,
                prefer_reduced_local_operator=prefer_reduced_local_operator,
                canonical_local_norm=canonical_local_norm,
                warm_start_bonds=warm_start_bonds,
                compact_updates=compact_history_updates,
                mixer_zero_block_noise_scale=(
                    mixer_zero_block_noise_scale if sweep_idx < mixer_nsweeps else 0.0
                ),
                mixer_rng=mixer_rng,
                record_post_update_energy=record_post_update_energy,
                compute_final_expectation=compute_final_expectation,
                state_average_root_environments=state_average_root_environments,
                state_average_local_norm=state_average_local_norm,
                store_orthonormal_renormalized_operators=store_orthonormal_renormalized_operators,
                renormalized_operator_cache=renormalized_operator_cache,
                renormalized_operator_cache_max_size=renormalized_operator_cache_max_size,
                renormalized_block_stack=persistent_renormalized_block_stack,
                norm_renormalized_block_stack=persistent_norm_renormalized_block_stack,
                target_renormalized_block_stack=persistent_target_renormalized_block_stack,
                complementary_operator_families=complementary_operator_families,
                identity_mpo_factors=identity_mpo_factors,
                reuse_prebuilt_boundary_side=reuse_prebuilt_boundary_side,
                require_block_sparse_renormalized_operator_table=require_block_sparse_renormalized_operator_table,
                require_symbolic_renormalized_operators=require_symbolic_renormalized_operators,
                bond_cursor=(
                    None
                    if moving_environment is None
                    else moving_environment.sweep_bonds
                ),
                lifecycle_owner=(
                    moving_environment
                    if moving_environment is not None
                    and moving_environment.su2_moving_environment is not None
                    else None
                ),
                profile=profile,
                verbose=verbose,
            )
        except Exception:
            if moving_environment is not None:
                moving_environment.abort_half_sweep()
            raise
        if moving_environment is not None:
            moving_environment.finish_half_sweep()
            moving_environment.finish_sweep(direction)
        last_root_sites = sweep_result.get("root_sites")
        last_root_center_tensor = sweep_result.get("root_center_tensor")
        last_root_center_bond = sweep_result.get("root_center_bond")
        if run_uses_root_environment_path and last_root_sites:
            current_sites = [site.copy() for site in last_root_sites[0]]
        else:
            current_sites = sweep_result["sites"]
        current_center = getattr(
            sweep_result.get("mps"),
            "center",
            None,
        )
        local_guess_cache = dict(sweep_result.get("local_guess_cache") or {})
        metric = float(measure_fn(sweep_result))
        objective_summary = _summarize_objectives(sweep_result["updates"])
        if mpo_factors is not None and last_root_sites and evaluate_root_energies_each_sweep:
            root_energies = []
            root_errors = []
            for root in last_root_sites:
                energy, error = _try_compute_state_energy_from_mpo(root, mpo_factors)
                root_energies.append(energy)
                root_errors.append(error)
            if any(error is not None for error in root_errors):
                objective_summary["state_energy_errors"] = root_errors
                last_state_energies = None
            else:
                last_state_energies = root_energies
                state_average_weights = _state_average_weights(
                    sweep_local_solver_kwargs,
                    len(last_state_energies),
                )
                objective_summary["state_energies"] = list(last_state_energies)
                objective_summary["state_average_weights"] = [
                    float(x) for x in state_average_weights
                ]
                objective_summary["state_average_energy"] = float(
                    np.dot(
                        state_average_weights,
                        np.asarray(last_state_energies, dtype=float),
                    )
                )
                objective_summary["energy"] = float(last_state_energies[0])
        if mpo_factors is not None and not run_uses_root_environment_path:
            numerator = sweep_result.get("final_mpo_numerator")
            denominator = sweep_result.get("final_mpo_denominator")
            if "energy" in objective_summary:
                pass
            elif numerator is not None and denominator is not None:
                denom = float(np.real(denominator))
                if abs(denom) < 1e-15:
                    if best_sites is not None:
                        converged = False
                        break
                    raise ValueError("State norm is numerically zero while computing sweep energy.")
                objective_summary["energy"] = float(np.real(numerator / denominator))
                objective_summary["energy_source"] = "final_mpo_expectation"
            elif sweep_result.get("terminal_local_energy") is not None:
                objective_summary["energy"] = float(
                    sweep_result["terminal_local_energy"]
                )
                objective_summary["energy_source"] = (
                    "cpp_terminal_local"
                    if moving_environment is not None
                    and moving_environment.su2_moving_environment is not None
                    else "terminal_local"
                )
            else:
                objective_summary["energy"] = _compute_state_energy_from_mpo(
                    current_sites,
                    mpo_factors,
                    identity_mpo_factors=identity_mpo_factors,
                )
        elif (
            mpo_factors is not None
            and run_uses_root_environment_path
            and last_root_sites
        ):
            root_energies = []
            root_errors = []
            for root in last_root_sites:
                energy, error = _try_compute_state_energy_from_mpo(root, mpo_factors)
                root_energies.append(energy)
                root_errors.append(error)
            if any(error is not None for error in root_errors):
                objective_summary["state_energy_errors"] = root_errors
                last_state_energies = None
            else:
                last_state_energies = root_energies
                state_average_weights = _state_average_weights(
                    sweep_local_solver_kwargs,
                    len(last_state_energies),
                )
                objective_summary["state_energies"] = list(last_state_energies)
                objective_summary["state_average_weights"] = [
                    float(x) for x in state_average_weights
                ]
                objective_summary["state_average_energy"] = float(
                    np.dot(
                        state_average_weights,
                        np.asarray(last_state_energies, dtype=float),
                    )
                )
                objective_summary["energy"] = float(last_state_energies[0])
        elif "objective_energy" in objective_summary:
            objective_summary["energy"] = objective_summary["objective_energy"]
        history.append(
            {
                "sweep": sweep_idx,
                "direction": direction,
                "metric": metric,
                "updates": sweep_result["updates"],
                "local_solver_kwargs": sweep_local_solver_kwargs,
                "warm_start_bonds": bool(warm_start_bonds),
                "mixer_applied": bool(mixer_zero_block_noise_scale > 0.0 and sweep_idx < mixer_nsweeps),
                "timing": sweep_result.get("timing"),
                "renormalized_operator_cache_size": sweep_result.get("renormalized_operator_cache_size"),
                "renormalized_operator_cache_stats": sweep_result.get("renormalized_operator_cache_stats"),
                "reused_prebuilt_boundary_side": sweep_result.get("reused_prebuilt_boundary_side"),
                "cpp_owned_half_sweep": bool(
                    sweep_result.get("cpp_owned_half_sweep", False)
                ),
                "owned_half_sweep_readiness_code": sweep_result.get(
                    "owned_half_sweep_readiness_code"
                ),
                "moving_environment_stats": (
                    moving_environment.stats if moving_environment is not None else None
                ),
                "renormalized_block_stack_stats": sweep_result.get("renormalized_block_stack_stats"),
                "rank_coupled_real_term_coalesce_stats": sweep_result.get(
                    "rank_coupled_real_term_coalesce_stats"
                ),
                "norm_renormalized_block_stack_stats": sweep_result.get(
                    "norm_renormalized_block_stack_stats"
                ),
                "target_renormalized_block_stack_stats": sweep_result.get(
                    "target_renormalized_block_stack_stats"
                ),
                **objective_summary,
            }
        )
        if run_uses_root_environment_path or converge_on_full_sweeps:
            previous_entry = next(
                (
                    entry
                    for entry in reversed(history[:-1])
                    if entry.get("direction") == history[-1].get("direction")
                ),
                None,
            )
        else:
            previous_entry = history[-2] if len(history) >= 2 else None
        energy_delta = (
            None
            if previous_entry is None
            else _state_average_energy_delta(previous_entry, history[-1])
        )
        history[-1]["energy_delta"] = energy_delta
        if int(verbose) >= 1:
            _emit_verbose(_format_sweep_line(sweep_idx, direction, history[-1]), verbose=verbose)
        if (
            mpo_factors is not None
            and "energy" in history[-1]
            and not (sweep_nlocal_states > 1 and not evaluate_root_energies_each_sweep)
        ):
            energy = float(history[-1]["energy"])
            if sweep_nlocal_states > 1:
                best_energy = energy
                best_sites = [site.copy() for site in current_sites]
                best_center = current_center
                best_root_sites = (
                    [[site.copy() for site in root] for root in last_root_sites]
                    if last_root_sites
                    else None
                )
                best_root_center_tensor = (
                    last_root_center_tensor.copy()
                    if last_root_center_tensor is not None
                    else None
                )
                best_root_center_bond = last_root_center_bond
                best_state_energies = list(last_state_energies) if last_state_energies else None
            elif best_energy is None or energy < best_energy:
                best_energy = energy
                if not history[-1].get("cpp_owned_half_sweep", False):
                    best_sites = [site.copy() for site in current_sites]
                    best_center = current_center
                    best_root_sites = (
                        [[site.copy() for site in root] for root in last_root_sites]
                        if last_root_sites
                        else None
                    )
                    best_root_center_tensor = (
                        last_root_center_tensor.copy()
                        if last_root_center_tensor is not None
                        else None
                    )
                    best_root_center_bond = last_root_center_bond
                    best_state_energies = list(last_state_energies) if last_state_energies else None
        convergence_boundary = (
            not converge_on_full_sweeps
            or direction == full_sweep_end_direction
        )
        if conv_tol is not None and convergence_boundary:
            if (
                sweep_nlocal_states > 1
                and energy_delta is not None
                and energy_delta <= float(conv_tol)
                and history[-1].get("state_average_energy") is not None
            ):
                converged = True
                history[-1]["converged"] = True
                history[-1]["convergence_metric"] = "energy_delta"
                break
            if (
                sweep_nlocal_states > 1
                and energy_delta is not None
                and energy_delta <= float(conv_tol)
                and metric <= float(conv_tol)
            ):
                converged = True
                history[-1]["converged"] = True
                history[-1]["convergence_metric"] = "energy_delta+metric"
                break
            if sweep_nlocal_states <= 1:
                if mpo_factors is not None and "energy" in history[-1]:
                    if (
                        energy_delta is not None
                        and energy_delta <= float(conv_tol)
                        and metric <= float(conv_tol)
                    ):
                        converged = True
                        history[-1]["converged"] = True
                        history[-1]["convergence_metric"] = "energy_delta+metric"
                        break
                elif metric <= conv_tol:
                    converged = True
                    history[-1]["converged"] = True
                    history[-1]["convergence_metric"] = "metric"
                    break
        if alternate:
            direction = "rl" if direction == "lr" else "lr"

    if conv_tol is None and not converged:
        converged = _infer_converged_from_objectives(history)

    if (
        moving_environment is not None
        and moving_environment.cpp_state_owned
        and history
        and history[-1].get("cpp_owned_half_sweep", False)
    ):
        final_direction = history[-1]["direction"]
        native_root_count = int(
            getattr(
                moving_environment.su2_moving_environment,
                "state_average_roots",
                0,
            )
        )
        if native_root_count > 1:
            last_root_sites = (
                moving_environment.export_owned_state_average_sites(
                    current_sites,
                    final_direction,
                )
            )
            current_sites = [
                site.copy() for site in last_root_sites[0]
            ]
            last_state_energies = history[-1].get("state_energies")
            best_root_sites = None
            best_state_energies = None
        else:
            current_sites = moving_environment.export_owned_sites(
                current_sites,
                final_direction,
            )
        current_center = (
            len(current_sites) - 1
            if final_direction == "lr"
            else 0
        )
        best_sites = None
        best_center = None
        if history[-1].get("energy") is not None:
            best_energy = float(history[-1]["energy"])
        history[-1]["moving_environment_stats"] = (
            moving_environment.stats
        )

    final_sites = best_sites if best_sites is not None else current_sites
    final_center = (
        best_center
        if best_sites is not None
        else current_center
    )
    final_root_sites = best_root_sites if best_root_sites is not None else last_root_sites
    final_root_center_tensor = (
        best_root_center_tensor
        if best_root_center_tensor is not None
        else last_root_center_tensor
    )
    final_root_center_bond = (
        best_root_center_bond
        if best_root_center_bond is not None
        else last_root_center_bond
    )
    final_root_weights = None
    if final_root_sites:
        n_final_roots = len(final_root_sites)
        if (
            input_multiroot is not None
            and input_multiroot.weights is not None
            and len(input_multiroot.weights) == n_final_roots
        ):
            final_root_weights = input_multiroot.weights
        else:
            final_root_weights = _state_average_weights(
                local_solver_kwargs or {},
                n_final_roots,
            )
    multiroot_mps = (
        MultiRootMPS.from_root_sites(
            final_root_sites,
            weights=final_root_weights,
            center_bond=final_root_center_bond,
            center_tensor=final_root_center_tensor,
            target_sector=target_sector,
        )
        if final_root_sites
        else None
    )
    return {
        "sites": final_sites,
        "mps": MPS(
            final_sites,
            center=final_center,
            target_sector=target_sector,
        ),
        "root_sites": final_root_sites,
        "root_mps": (
            multiroot_mps.roots
            if final_root_sites
            else None
        ),
        "multiroot_mps": multiroot_mps,
        "root_center_tensor": final_root_center_tensor,
        "state_energies": best_state_energies if best_state_energies is not None else last_state_energies,
        "history": history,
        "converged": converged,
        "last_direction": history[-1]["direction"] if history else direction,
        "ncompleted": len(history),
        "best_energy": best_energy,
        "moving_environment_stats": (
            moving_environment.stats if moving_environment is not None else None
        ),
    }
