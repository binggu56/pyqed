#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal sweep drivers for fixed-layout non-Abelian tensor chains.
"""

from __future__ import annotations

import inspect
import numpy as np

from pyqed.mps.su2 import SpinChargeSector, fuse_charge_spin_sectors

from .canonical import (
    assert_mixed_canonical_sites,
    left_canonicalize_sites,
    mixed_canonicalize_sites,
    right_canonicalize_sites,
)
from .environment import BlockSparseEnvironmentChain, contract_chain_expectation
from .contraction import merge_mps_sites
from .mps import MPS
from .solver import TwoSiteEffectiveH
from .tensor import NonabelianTensor
from .update import two_site_update


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


def _format_bond_update_line(bond, update):
    objective = dict(update.get("local_objective") or {})
    return (
        f"  bond {bond:>2} | "
        f"problem={objective.get('effective_local_problem', '-'):>11} | "
        f"E={_format_verbose_number(objective.get('energy')):>14} | "
        f"E_post={_format_verbose_number(objective.get('post_update_energy')):>14} | "
        f"kept={str(update.get('kept', '-')):>4} | "
        f"trunc={_format_verbose_number(update.get('trunc_err')):>10}"
    )


def _format_sweep_line(sweep_idx, direction, history_entry):
    return (
        f"sweep {sweep_idx:>2} | "
        f"dir={direction} | "
        f"E={_format_verbose_number(history_entry.get('energy')):>14} | "
        f"E_obj={_format_verbose_number(history_entry.get('objective_energy')):>14} | "
        f"metric={_format_verbose_number(history_entry.get('metric')):>10}"
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
    local_guess_cache = dict(local_guess_cache or {})
    next_local_guess_cache = {}
    if max_bond_mode is None:
        max_bond_mode = "states" if mpo_factors is not None else "reduced"
    if (local_operator is not None or mpo_factors is not None) and "couple_physical" not in local_solver_kwargs:
        # The uncoupled physical-leg path is currently faster than the coupled
        # basis path for the non-Abelian MPO sweeps in this codebase.
        local_solver_kwargs["couple_physical"] = False
    updates = []
    env_sweep = None
    norm_env_sweep = None
    target_env_sweep = None
    force_canonical_local_norm = str(canonical_local_norm).lower() in {"force", "forced", "unsafe"}
    if mpo_factors is not None:
        env_sweep = BlockSparseEnvironmentChain.build(updated_sites, mpo_factors).start_sweep(direction)
        if not force_canonical_local_norm:
            norm_env_sweep = BlockSparseEnvironmentChain.build(
                updated_sites,
                _identity_mpo_factors_for_sites_and_mpo(updated_sites, mpo_factors),
            ).start_sweep(direction)
        if root_target_mpo_factors is not None:
            target_env_sweep = BlockSparseEnvironmentChain.build(
                updated_sites,
                root_target_mpo_factors,
            ).start_sweep(direction)
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
                operator = env_sweep.bond_operator(bond, merged)
                norm_operator = (
                    None
                    if force_canonical_local_norm
                    else norm_env_sweep.bond_operator(bond, merged)
                )
                norm_is_identity = (
                    True
                    if force_canonical_local_norm
                    else getattr(norm_operator, "identity_like", False)
                )
                return TwoSiteEffectiveH(
                    operator=operator,
                    norm_operator=norm_operator,
                    canonical_norm=(
                        True
                        if norm_is_identity or state_averaged_local
                        else False
                    ),
                    name=f"bond-{bond}-effective-H",
                )
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
        )
        if (
            warm_start_bonds
            and (local_operator is not None or mpo_factors is not None)
            and isinstance(update.get("optimized"), NonabelianTensor)
        ):
            next_local_guess_cache[bond] = update["optimized"].copy()
        if guess_source is not None and update.get("local_guess_used"):
            update.setdefault("local_objective", {})
            update["local_objective"]["warm_start"] = guess_source
        updated_sites[bond] = update["left"]
        updated_sites[bond + 1] = update["right"]
        if root_sites is not None:
            root_pairs = update.get("root_site_pairs") or []
            for root_idx, sites_for_root in enumerate(root_sites):
                if root_idx < len(root_pairs):
                    root_left, root_right = root_pairs[root_idx]
                else:
                    root_left, root_right = update["left"], update["right"]
                sites_for_root[bond] = root_left.copy()
                sites_for_root[bond + 1] = root_right.copy()
        if mpo_factors is not None:
            next_center = bond + 1 if direction == "lr" else bond
            assert_mixed_canonical_sites(updated_sites, next_center)
        if env_sweep is not None:
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
        if record_post_update_energy and mpo_factors is not None:
            update.setdefault("local_objective", {})
            update["local_objective"]["post_update_energy"] = _compute_state_energy_from_mpo(
                updated_sites,
                mpo_factors,
            )
        if int(verbose) >= 2:
            _emit_verbose(_format_bond_update_line(bond, update), verbose=verbose)
        updates.append({"bond": bond, **update})

    return {
        "direction": direction,
        "sites": updated_sites,
        "mps": MPS(updated_sites, center=(bonds[-1] + 1 if direction == "lr" else bonds[-1])),
        "root_sites": root_sites,
        "updates": updates,
        "local_guess_cache": next_local_guess_cache,
        "final_mpo_numerator": (
            env_sweep.final_expectation(updated_sites) if env_sweep is not None else None
        ),
        "final_mpo_denominator": (
            norm_env_sweep.final_expectation(updated_sites) if norm_env_sweep is not None else None
        ),
    }


def _default_sweep_measure(sweep_result):
    updates = sweep_result["updates"]
    if not updates:
        return 0.0
    return max(update["trunc_err"] for update in updates)


def _summarize_objectives(updates):
    bond_objectives = []
    energies = []
    metrics = []
    values = []
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

    summary = {"bond_objectives": bond_objectives}
    if energies:
        summary["objective_energy"] = sum(energies) / len(energies)
    if metrics:
        summary["objective_metric"] = sum(metrics) / len(metrics)
    if values:
        summary["objective_value"] = sum(values) / len(values)
    return summary


def _compute_state_energy_from_mpo(sites, mpo_factors):
    numerator = contract_chain_expectation(sites, mpo_factors)
    denominator = contract_chain_expectation(
        sites,
        _identity_mpo_factors_for_sites_and_mpo(sites, mpo_factors),
    )
    denom = float(np.real(denominator))
    if abs(denom) < 1e-15:
        raise ValueError("State norm is numerically zero while computing sweep energy.")
    return float(np.real(numerator / denominator))


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
    bond_coupling, max_bond, max_bond_mode, cutoff, root_target_mpo_factors,
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
    input_mps = sites if isinstance(sites, MPS) else None
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
    best_state_energies = None
    best_energy = None
    last_root_sites = None
    last_state_energies = None
    local_guess_cache = {}
    mixer_zero_block_noise_scale = float(mixer_zero_block_noise_scale)
    mixer_nsweeps = int(mixer_nsweeps)
    mixer_rng = np.random.default_rng(mixer_zero_block_noise_seed)

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
            verbose=verbose,
        )
        current_sites = sweep_result["sites"]
        last_root_sites = sweep_result.get("root_sites")
        local_guess_cache = dict(sweep_result.get("local_guess_cache") or {})
        metric = float(measure_fn(sweep_result))
        objective_summary = _summarize_objectives(sweep_result["updates"])
        if mpo_factors is not None and last_root_sites and evaluate_root_energies_each_sweep:
            last_state_energies = [
                _compute_state_energy_from_mpo(root, mpo_factors)
                for root in last_root_sites
            ]
            objective_summary["state_energies"] = list(last_state_energies)
            objective_summary["energy"] = float(last_state_energies[0])
        if mpo_factors is not None:
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
                )
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
                **objective_summary,
            }
        )
        if int(verbose) >= 1:
            _emit_verbose(_format_sweep_line(sweep_idx, direction, history[-1]), verbose=verbose)
        if mpo_factors is not None and "energy" in history[-1]:
            energy = float(history[-1]["energy"])
            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_sites = [site.copy() for site in current_sites]
                best_root_sites = (
                    [[site.copy() for site in root] for root in last_root_sites]
                    if last_root_sites
                    else None
                )
                best_state_energies = list(last_state_energies) if last_state_energies else None
        if conv_tol is not None and metric <= conv_tol:
            converged = True
            break
        if alternate:
            direction = "rl" if direction == "lr" else "lr"

    if conv_tol is None and not converged:
        converged = _infer_converged_from_objectives(history)

    final_sites = best_sites if best_sites is not None else current_sites
    final_root_sites = best_root_sites if best_root_sites is not None else last_root_sites
    return {
        "sites": final_sites,
        "mps": MPS(final_sites, target_sector=target_sector),
        "root_sites": final_root_sites,
        "root_mps": (
            [MPS(root, target_sector=target_sector) for root in final_root_sites]
            if final_root_sites
            else None
        ),
        "state_energies": best_state_energies if best_state_energies is not None else last_state_energies,
        "history": history,
        "converged": converged,
        "last_direction": history[-1]["direction"] if history else direction,
        "ncompleted": len(history),
        "best_energy": best_energy,
    }
