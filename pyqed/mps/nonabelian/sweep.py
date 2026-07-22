#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal sweep drivers for fixed-layout non-Abelian tensor chains.
"""

from __future__ import annotations

import inspect
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
from .contraction import merge_mps_sites, normalize_site_tensor_layout
from .decompose import state_averaged_svd_two_site
from .linalg import sector_state_weight
from .mps import MPS
from .multiroot import MultiRootMPS, fuse_root_center_tensors
from .renormalized import RenormalizedBlockStack, RenormalizedOperatorStack
from .solver import TwoSiteEffectiveH, solve_local_two_site
from .solver import (
    _materialize_local_matrix,
    _normalize_local_operator,
    _operator_basis_for_layout,
    _resolve_davidson_operator,
    pack_two_site_state,
    unpack_two_site_state,
    two_site_state_basis,
)
from .tensor import NonabelianTensor
from .update import _expand_two_site_support, two_site_update


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
    from .mpo import MPO, IrreducibleMPO, RankCoupledMPO, PhysicalLeg

    identity_factors = []
    for site, factor in zip(sites, mpo_factors):
        if isinstance(factor, (MPO, IrreducibleMPO, RankCoupledMPO)):
            phys_leg = factor.phys_out_leg
        else:
            physical_slices = _tensor_dense_layout(site)["sector_slices"][1]
            phys_leg = PhysicalLeg.from_slices(physical_slices)
        identity_factors.append(MPO.from_site_operator(identity_operator(phys_leg)))
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
        renormalized_operator_cache_max_size=256,
    ):
        self.hamiltonian_stack = RenormalizedBlockStack(
            namespace="hamiltonian",
            complementary_operator_families=complementary_operator_families,
        )
        self.norm_stack = RenormalizedBlockStack(namespace="norm")
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
        self.valid_boundary_side = None
        self.environment_rebuilds = 0
        self.boundary_side_reuses = 0
        self.boundary_side_rebuilds = 0
        self.completed_sweeps = 0
        self.last_reused_prebuilt_side = None

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

        needed = self.needed_prebuilt_side(direction)
        if self.valid_boundary_side == needed:
            self.boundary_side_reuses += 1
            self.last_reused_prebuilt_side = needed
            return needed
        self.environment_rebuilds += 1
        self.boundary_side_rebuilds += 1
        self.last_reused_prebuilt_side = None
        return None

    def finish_sweep(self, direction):
        """
        Mark the side advanced by a completed sweep as reusable.
        """

        self.valid_boundary_side = self.produced_boundary_side(direction)
        self.completed_sweeps += 1

    @property
    def stats(self):
        h_stats = self.hamiltonian_stack.stats
        comp_stats = h_stats.get("complementary_operator_stack") or {}
        return {
            "completed_sweeps": int(self.completed_sweeps),
            "valid_boundary_side": self.valid_boundary_side,
            "last_reused_prebuilt_side": self.last_reused_prebuilt_side,
            "environment_rebuilds": int(self.environment_rebuilds),
            "boundary_side_reuses": int(self.boundary_side_reuses),
            "boundary_side_rebuilds": int(self.boundary_side_rebuilds),
            "hamiltonian_boundary_entries": int(h_stats.get("size", 0)),
            "hamiltonian_boundary_advances": int(h_stats.get("advanced_entries", 0)),
            "complementary_operator_entries": int(comp_stats.get("n_entries", 0)),
            "complementary_operator_advances": int(comp_stats.get("advances", 0)),
            "moving_environment_cache": h_stats.get("moving_environment_cache"),
        }


def sweep_once(
    sites,
    *,
    direction="lr",
    solver=None,
    local_operator=None,
    mpo_factors=None,
    root_target_mpo_factors=None,
    local_solver_kwargs=None,
    local_guess_cache=None,
    initial_root_sites=None,
    bond_coupling="left",
    max_bond=None,
    max_bond_mode=None,
    cutoff=1e-10,
    prefer_reduced_local_operator=False,
    canonical_local_norm=False,
    warm_start_bonds=False,
    mixer_zero_block_noise_scale=0.0,
    mixer_rng=None,
    record_post_update_energy=False,
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
    require_block_sparse_renormalized_operator_table=False,
    require_symbolic_renormalized_operators=False,
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
    if solver is not None and mpo_factors is not None:
        raise ValueError("Specify mpo_factors only when using the built-in local-operator path.")
    if local_operator is not None and mpo_factors is not None:
        raise ValueError("Specify only one of local_operator or mpo_factors for sweep_once.")
    if mpo_factors is not None and len(mpo_factors) != len(sites):
        raise ValueError("mpo_factors must match the number of site tensors.")
    if root_target_mpo_factors is not None and len(root_target_mpo_factors) != len(sites):
        raise ValueError("root_target_mpo_factors must match the number of site tensors.")

    direction = _normalize_direction(direction)
    absorb = "right" if direction == "lr" else "left"
    bonds = list(range(len(sites) - 1))
    if direction == "rl":
        bonds.reverse()

    updated_sites = [site.copy() for site in sites]
    if mpo_factors is not None:
        if reuse_prebuilt_boundary_side is None:
            canonical_center = min(1, len(updated_sites) - 1) if direction == "lr" else max(0, len(updated_sites) - 2)
            updated_sites = mixed_canonicalize_sites(
                updated_sites,
                canonical_center,
                max_bond=None,
                cutoff=0.0,
                max_bond_mode=max_bond_mode or "states",
                bond_coupling=bond_coupling,
            )
            assert_mixed_canonical_sites(updated_sites, canonical_center)
    local_solver_kwargs = dict(local_solver_kwargs or {})
    nlocal_states = int(local_solver_kwargs.get("nstates", 1))
    use_root_environment_path = bool(
        state_average_root_environments and mpo_factors is not None and nlocal_states > 1
    )
    if initial_root_sites is not None:
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
            )
            for sites_for_root in root_sites
        ]
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
    force_canonical_local_norm = str(canonical_local_norm).lower() in {"force", "forced", "unsafe"}
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
                reuse_prebuilt_boundary_side=reuse_prebuilt_boundary_side,
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
    for bond in bonds:
        bond_local_solver_kwargs = dict(local_solver_kwargs)
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
        merged_solver = None
        if solver is not None:
            def merged_solver(merged, bond=bond, solver=solver):
                return _call_solver(solver, bond, merged)
        merged_local_operator = None
        if local_operator is not None:
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
                    norm_operator = (
                        None
                        if force_canonical_local_norm or (
                            state_averaged_local and not state_average_local_norm
                        )
                        else norm_env_sweep.bond_operator(bond, merged)
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
                local_solver_kwargs=bond_local_solver_kwargs,
                bond_coupling=bond_coupling,
                max_bond=max_bond,
                max_bond_mode=max_bond_mode,
                cutoff=cutoff,
                absorb=absorb,
                profile=profile,
            )
        else:
            update = two_site_update(
                updated_sites[bond],
                updated_sites[bond + 1],
                solver=merged_solver,
                local_operator=merged_local_operator,
                local_solver_kwargs=bond_local_solver_kwargs,
                bond_coupling=bond_coupling,
                max_bond=max_bond,
                max_bond_mode=max_bond_mode,
                cutoff=cutoff,
                absorb=absorb,
                prefer_reduced_local_operator=prefer_reduced_local_operator,
                mixer_zero_block_noise_scale=mixer_zero_block_noise_scale,
                mixer_rng=mixer_rng,
                profile=profile,
            )
        if profile:
            timing["two_site_update"] += time.perf_counter() - t0
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
        if isinstance(renormalized_operator_cache, RenormalizedOperatorStack):
            renormalized_operator_cache.prune()
        elif renormalized_operator_cache_max_size > 0:
            while len(renormalized_operator_cache) > renormalized_operator_cache_max_size:
                renormalized_operator_cache.pop(next(iter(renormalized_operator_cache)))
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
        updates.append({"bond": bond, **update})

    if profile:
        timing["total"] = time.perf_counter() - sweep_t0

    return {
        "direction": direction,
        "sites": updated_sites,
        "mps": MPS(updated_sites, center=(bonds[-1] + 1 if direction == "lr" else bonds[-1])),
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
            env_sweep.final_expectation(updated_sites) if env_sweep is not None else None
        ),
        "final_mpo_denominator": (
            norm_env_sweep.final_expectation(updated_sites) if norm_env_sweep is not None else None
        ),
        "timing": timing,
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
    if Qprev.shape[1] > 0:
        full = np.eye(dim, dtype=complex)
        candidates = full - Qprev @ Qprev.conj().T
        Q = _orthonormal_columns(candidates.T, dim=dim, tol=tol)
    else:
        Q = np.eye(dim, dtype=complex)
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
        "subspace_dim": int(Q.shape[1]),
        "layout_size": int(dim),
    }


def _state_average_root_environment_update(
    root_sites,
    bond,
    *,
    direction,
    mpo_factors,
    local_solver_kwargs,
    bond_coupling,
    max_bond,
    max_bond_mode,
    cutoff,
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
        )
        root_sites[root_idx] = canonical_sites
        merged = merge_mps_sites(canonical_sites[bond], canonical_sites[bond + 1])
        t0 = time.perf_counter() if profile else None
        operator = BlockSparseEnvironmentChain.build(canonical_sites, mpo_factors).bond_operator(
            bond,
            merged,
        )
        if profile:
            timing["environment"] += time.perf_counter() - t0

        t0 = time.perf_counter() if profile else None
        optimized, objective = _dense_deflated_local_root(
            merged,
            operator,
            optimized_roots,
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
    conv_tol=None,
    measure=None,
    prefer_reduced_local_operator=False,
    canonical_local_norm=False,
    warm_start_bonds=False,
    mixer_zero_block_noise_scale=0.0,
    mixer_zero_block_noise_seed=None,
    mixer_nsweeps=1,
    record_post_update_energy=False,
    evaluate_root_energies_each_sweep=True,
    state_average_root_environments=False,
    state_average_local_norm=False,
    store_orthonormal_renormalized_operators=False,
    renormalized_operator_cache=None,
    renormalized_operator_cache_max_size=256,
    require_block_sparse_renormalized_operator_table=False,
    require_symbolic_renormalized_operators=False,
    complementary_operator_families=None,
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
    prefer_reduced_local_operator, canonical_local_norm, warm_start_bonds
        Passed through to :func:`sweep_once`.
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
    measure_fn = _default_sweep_measure if measure is None else measure
    current_sites = [site.copy() for site in sites]
    history = []
    converged = False
    best_sites = None
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
    force_canonical_local_norm = str(canonical_local_norm).lower() in {"force", "forced", "unsafe"}
    moving_environment = None
    if mpo_factors is not None and not run_uses_root_environment_path:
        moving_environment = MovingEnvironment(
            current_sites,
            mpo_factors=mpo_factors,
            root_target_mpo_factors=root_target_mpo_factors,
            complementary_operator_families=complementary_operator_families,
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
            prefer_reduced_local_operator=prefer_reduced_local_operator,
            canonical_local_norm=canonical_local_norm,
            warm_start_bonds=warm_start_bonds,
            mixer_zero_block_noise_scale=(
                mixer_zero_block_noise_scale if sweep_idx < mixer_nsweeps else 0.0
            ),
            mixer_rng=mixer_rng,
            record_post_update_energy=record_post_update_energy,
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
            profile=profile,
            verbose=verbose,
        )
        if moving_environment is not None:
            moving_environment.finish_sweep(direction)
        last_root_sites = sweep_result.get("root_sites")
        last_root_center_tensor = sweep_result.get("root_center_tensor")
        last_root_center_bond = sweep_result.get("root_center_bond")
        if run_uses_root_environment_path and last_root_sites:
            current_sites = [site.copy() for site in last_root_sites[0]]
        else:
            current_sites = sweep_result["sites"]
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
        energy_delta = (
            None
            if len(history) < 2
            else _state_average_energy_delta(history[-2], history[-1])
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
                best_sites = [site.copy() for site in current_sites]
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
        if conv_tol is not None:
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

    final_sites = best_sites if best_sites is not None else current_sites
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
        "mps": MPS(final_sites, target_sector=target_sector),
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
