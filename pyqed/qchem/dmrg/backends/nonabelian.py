"""Non-Abelian backend adapter for qchem spatial-orbital DMRG."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pyqed.mps.nonabelian import (
    MPS,
    MultiRootMPS,
    build_product_spatial_mps,
    build_random_spatial_mps,
    build_random_reduced_spatial_mps,
    contract_chain_expectation,
    run_sweeps,
    spatial_target_sector,
)
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.mps.nonabelian.renormalized import (
    configure_complementary_family_kernel_policy,
    get_complementary_family_kernel_policy,
)


@dataclass
class NonAbelianDMRGResult:
    """Small result object matching the attributes used by the qchem wrapper."""

    e_tot: float | list[float]
    ground_state: MPS
    states: list[MPS]
    history: list[dict]
    converged: bool
    ncompleted: int
    target_sector: object
    multiroot_state: MultiRootMPS | None = None
    backend: str = "nonabelian"


def _qchem_sweep_measure(sweep_result):
    """
    Converge qchem sweeps on the local eigensolver objective, not truncation.

    A large enough bond dimension can make the truncation error exactly zero
    after one pass even when the two-site local residuals are still large.
    """

    metrics = []
    for update in sweep_result.get("updates", []):
        objective = update.get("local_objective") or {}
        if "metric" in objective:
            metrics.append(abs(float(objective["metric"])))
        elif "residual" in objective:
            metrics.append(abs(float(objective["residual"])))
    if metrics:
        return max(metrics)
    updates = sweep_result.get("updates", [])
    if not updates:
        return 0.0
    return max(float(update.get("trunc_err", 0.0)) for update in updates)


def _hf_spatial_labels(nelecas, ncas, spin):
    """Return a compact high-spin product occupation in spatial-site labels."""
    nelecas = int(nelecas)
    ncas = int(ncas)
    two_s = int(spin)
    if nelecas < 0 or nelecas > 2 * ncas:
        raise ValueError(f"Invalid active electron count {nelecas} for ncas={ncas}.")
    if abs(two_s) > nelecas or (nelecas + two_s) % 2:
        raise ValueError(f"Invalid spin={spin!r} for nelecas={nelecas}.")

    nalpha = (nelecas + two_s) // 2
    nbeta = (nelecas - two_s) // 2
    if nalpha > ncas or nbeta > ncas:
        raise ValueError(f"Spin/electron count does not fit in {ncas} spatial orbitals.")

    labels = ["empty"] * ncas
    for i in range(nbeta):
        labels[i] = "double"
    for i in range(nbeta, nalpha):
        labels[i] = "up"
    return labels


def _make_initial_mps(qcdmrg, *, target_sector, initial_guess=None, bond_multiplicity=2, seed=7):
    guess = qcdmrg.init_guess if initial_guess is None else initial_guess
    if isinstance(guess, MPS):
        mps = guess.copy()
        mps.target_sector = target_sector
        return mps

    if not isinstance(guess, str):
        raise TypeError(f"Unsupported non-Abelian initial guess type: {type(guess)}")

    method = guess.lower()
    fully_reduced = getattr(qcdmrg, "spatial_site_basis", "canonical") in {
        "fully_reduced",
        "fully_reduced_su2",
    }
    if method in {"hf", "product"}:
        if fully_reduced:
            # A fully reduced SU(2) site has no spin-projection product state;
            # seed the target-sector manifold directly.  This is the actual
            # initial state, not just product-state noise, so keep it normalized
            # at a regular random scale before the first canonicalization.
            sites = build_random_reduced_spatial_mps(
                qcdmrg.ncas,
                target_sector=target_sector,
                bond_multiplicity=bond_multiplicity,
                seed=seed,
            )
        else:
            sites = build_product_spatial_mps(
                _hf_spatial_labels(qcdmrg.nelecas, qcdmrg.ncas, qcdmrg.spin),
                enrich_bond_sectors=True,
                bond_multiplicity=bond_multiplicity,
                zero_block_noise_scale=1.0e-5,
                zero_block_noise_seed=seed,
            )
    elif method in {"cid", "cisd", "random", "previous"}:
        if fully_reduced:
            sites = build_random_reduced_spatial_mps(
                qcdmrg.ncas,
                target_sector=target_sector,
                bond_multiplicity=bond_multiplicity,
                seed=seed,
            )
        else:
            sites = build_random_spatial_mps(
                qcdmrg.ncas,
                target_sector=target_sector,
                bond_multiplicity=bond_multiplicity,
                seed=seed,
            )
    else:
        raise ValueError(f"Unsupported non-Abelian initial guess {guess!r}.")
    return MPS.from_sites(sites, target_sector=target_sector)


def _make_state_average_root_sites(
    qcdmrg,
    *,
    target_sector,
    nroots,
    initial_guess=None,
    bond_multiplicity=2,
    seed=7,
):
    """
    Build distinct initial root chains for state-averaged SU(2) sweeps.

    Block2 seeds a multi-root MPS object directly.  PyQED currently carries the
    roots as separate site lists, so the closest analogue is to give each root a
    different target-sector MPS from the first sweep onward.
    """

    nroots = int(nroots)
    if nroots <= 1:
        return None
    roots = []
    roots.append(
        _make_initial_mps(
            qcdmrg,
            target_sector=target_sector,
            initial_guess=initial_guess,
            bond_multiplicity=bond_multiplicity,
            seed=seed,
        ).sites
    )
    for root_idx in range(1, nroots):
        root = _make_initial_mps(
            qcdmrg,
            target_sector=target_sector,
            initial_guess="random",
            bond_multiplicity=bond_multiplicity,
            seed=seed + 104729 * root_idx,
        )
        roots.append(root.sites)
    return [[site.copy() for site in root] for root in roots]


def _make_state_average_multiroot_mps(
    qcdmrg,
    *,
    target_sector,
    nroots,
    weights=None,
    initial_guess=None,
    bond_multiplicity=2,
    seed=7,
):
    root_sites = _make_state_average_root_sites(
        qcdmrg,
        target_sector=target_sector,
        nroots=nroots,
        initial_guess=initial_guess,
        bond_multiplicity=bond_multiplicity,
        seed=seed,
    )
    if root_sites is None:
        return None
    return MultiRootMPS.from_root_sites(
        root_sites,
        weights=weights,
        target_sector=target_sector,
    )


def _expectation_from_nonabelian_mps(state, mpo_factors):
    numerator = contract_chain_expectation(state.sites, mpo_factors)
    denominator = contract_chain_expectation(
        state.sites,
        _identity_mpo_factors_for_sites_and_mpo(state.sites, mpo_factors),
    )
    denom = float(np.real(denominator))
    if abs(denom) < 1.0e-15:
        raise ValueError("State norm is numerically zero while evaluating an expectation value.")
    return float(np.real(numerator / denominator))


def _spin_square_mpo_factors(qcdmrg):
    from pyqed.qchem.dmrg.dmrg import _build_grouped_spatial_s2_tensor_mpo

    return _build_grouped_spatial_s2_tensor_mpo(qcdmrg.ncas).factors


def _finalize_spin_targeted_roots(
    qcdmrg,
    root_mps,
    nstates,
    *,
    compute_s2=True,
    select_by_spin=False,
    spin_tol=1.0e-6,
):
    """
    Select and report final state-averaged SU(2) roots.

    Parameters
    ----------
    qcdmrg
        Chemistry DMRG driver carrying the active Hamiltonian and spin target.
    root_mps
        Candidate root MPSs propagated by the sweep engine.
    nstates
        Number of physical roots requested by the caller.
    compute_s2
        If True, report ``<S^2>`` for the selected roots.
    select_by_spin
        If True, evaluate candidate ``<S^2>`` once at the end and select the
        lowest-energy roots compatible with the target spin.  This avoids the
        much more expensive dense local ``<S^2>`` projector on large active
        spaces while still filtering the candidate root buffer.
    spin_tol
        Absolute tolerance for accepting candidate roots by ``<S^2>``.

    Returns
    -------
    tuple
        ``(roots, energies, s2_values, info)`` for the selected roots and
        diagnostics for all candidate roots.
    """
    if not root_mps:
        return root_mps, None, None, {}
    candidate_roots = []
    candidate_energies = []
    candidate_source_indices = []
    discarded_zero_norm_roots = []
    for root_idx, state in enumerate(root_mps):
        try:
            energy = _expectation_from_nonabelian_mps(state, qcdmrg.H)
        except ValueError as exc:
            if "State norm is numerically zero" not in str(exc):
                raise
            discarded_zero_norm_roots.append(int(root_idx))
            continue
        candidate_source_indices.append(int(root_idx))
        candidate_roots.append(state)
        candidate_energies.append(energy)
    if not candidate_roots:
        return [], None, None, {
            "discarded_zero_norm_roots": discarded_zero_norm_roots,
        }
    need_s2 = bool(compute_s2 or select_by_spin)
    candidate_s2 = None
    if need_s2:
        s2_factors = _spin_square_mpo_factors(qcdmrg)
        candidate_s2 = [
            _expectation_from_nonabelian_mps(state, s2_factors)
            for state in candidate_roots
        ]
    selected_indices = list(range(min(int(nstates), len(candidate_roots))))
    root_selection_used = False
    target_spin_valid = None
    if select_by_spin and candidate_s2 is not None:
        target_s = 0.5 * abs(float(qcdmrg.spin))
        target_s2 = target_s * (target_s + 1.0)
        matching = [
            idx
            for idx, s2 in enumerate(candidate_s2)
            if abs(float(s2) - target_s2) <= float(spin_tol)
        ]
        if len(matching) >= int(nstates):
            selected_indices = sorted(
                matching,
                key=lambda idx: (float(candidate_energies[idx]), idx),
            )[: int(nstates)]
            root_selection_used = True
        target_spin_valid = all(
            abs(float(candidate_s2[idx]) - target_s2) <= float(spin_tol)
            for idx in selected_indices
        )
    selected_roots = [candidate_roots[idx] for idx in selected_indices]
    state_energies = [candidate_energies[idx] for idx in selected_indices]
    s2_values = (
        None
        if candidate_s2 is None
        else [candidate_s2[idx] for idx in selected_indices]
    )
    info = {
        "candidate_state_energies": [float(x) for x in candidate_energies],
        "candidate_state_s2": (
            None if candidate_s2 is None else [float(x) for x in candidate_s2]
        ),
        "selected_root_indices": [int(candidate_source_indices[x]) for x in selected_indices],
        "final_root_selection_used": bool(root_selection_used),
        "target_spin_valid": target_spin_valid,
        "discarded_zero_norm_roots": discarded_zero_norm_roots,
    }
    return (
        selected_roots,
        [float(x) for x in state_energies],
        None if s2_values is None else [float(x) for x in s2_values],
        info,
    )


def run_spatial_qchem_dmrg(
    qcdmrg,
    *,
    nsweeps=50,
    max_bond=None,
    initial_guess=None,
    bond_multiplicity=2,
    seed=7,
    conv_tol=None,
    nstates=1,
    weights=None,
    local_solver_kwargs=None,
    verbose=0,
    **sweep_kwargs,
):
    """
    Run the lower-level non-Abelian sweep engine on a qchem spatial MPO.

    The qchem wrapper owns chemistry concerns: active-space integrals, core
    energy, physical site ordering, and reporting. This adapter only converts
    those objects to the non-Abelian MPS sweep API.
    """
    if qcdmrg.site != "spatial":
        raise NotImplementedError("The non-Abelian qchem DMRG backend currently requires site='spatial'.")
    if qcdmrg.H is None:
        raise ValueError("DMRG Hamiltonian MPO is not built. Call build() before the backend.")
    if qcdmrg.spin_purification:
        raise NotImplementedError("Spin-purification penalties are not supported by the SU(2) backend.")
    active_hamiltonian = getattr(qcdmrg, "_active_hamiltonian", None)
    mpo_factors = active_hamiltonian.mpo if active_hamiltonian is not None else qcdmrg.H
    complementary_operator_families = (
        getattr(active_hamiltonian, "complementary_operators", None)
        if active_hamiltonian is not None
        else None
    )
    nstates = int(nstates)
    if nstates < 1:
        raise ValueError("nstates must be positive.")
    if weights is None:
        weights = np.ones(nstates, dtype=float) / nstates
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != nstates:
            raise ValueError("weights must match nstates.")
        weights = weights / np.sum(weights)

    target_sector = spatial_target_sector(qcdmrg.nelecas, int(qcdmrg.spin))
    if max_bond is None:
        max_bond = qcdmrg.D

    fully_reduced_sites = getattr(qcdmrg, "spatial_site_basis", "canonical") in {
        "fully_reduced",
        "fully_reduced_su2",
    }
    state_average_spin_tol = float(sweep_kwargs.pop("state_average_spin_tol", 1.0e-6))
    state_average_validate_spin = bool(
        sweep_kwargs.pop("state_average_validate_spin", not fully_reduced_sites)
    )
    state_average_spin_projector = bool(
        sweep_kwargs.pop("state_average_spin_projector", not fully_reduced_sites)
    )
    debug_state_average = bool(sweep_kwargs.pop("debug_state_average", False))
    if debug_state_average and int(verbose) < 2:
        verbose = 2
    allow_experimental_su2_state_average = bool(
        sweep_kwargs.pop("allow_experimental_su2_state_average", False)
    )
    default_local_basis_policy = (
        "block2_like"
        if nstates == 1 or fully_reduced_sites
        else "mixed_canonical_standard"
    )
    requested_policy_name = str(
        sweep_kwargs.pop("local_basis_policy", default_local_basis_policy)
    ).lower().replace("-", "_")
    block2_like_state_average = False
    if requested_policy_name in {"block2", "block2_like"}:
        if nstates > 1:
            # The production block2-like SA path is the metric-aware two-site
            # sweep with state-averaged density-matrix truncation.  The older
            # orthonormalized-operator transform remains available by asking
            # for local_basis_policy='orthonormalized_operator' explicitly.
            local_basis_policy = "mixed_canonical_standard"
            block2_like_state_average = True
        else:
            local_basis_policy = "orthonormalized_operator"
    elif requested_policy_name in {"orthonormalized", "metric_orthonormalized"}:
        local_basis_policy = "mixed_canonical_standard"
    else:
        local_basis_policy = requested_policy_name
    if local_basis_policy not in {
        "mixed_canonical_standard",
        "orthonormalized_operator",
        "legacy_generalized",
    }:
        raise ValueError(
            "local_basis_policy must be 'mixed_canonical_standard', "
            "'block2_like', 'orthonormalized_operator', or 'legacy_generalized'."
        )
    allow_experimental_block2_state_average = bool(
        sweep_kwargs.pop("allow_experimental_block2_state_average", False)
        or allow_experimental_su2_state_average
    )
    if (
        nstates > 1
        and local_basis_policy == "orthonormalized_operator"
        and int(qcdmrg.ncas) > 2
        and not allow_experimental_block2_state_average
    ):
        raise NotImplementedError(
            "orthonormalized_operator state-averaged SU(2) DMRG "
            "is currently validated only for two-site smoke tests. Use "
            "local_basis_policy='block2_like' for the metric-aware block2-like "
            "SA path, or pass allow_experimental_block2_state_average=True for "
            "orthonormalized-operator debugging."
        )
    requested_orthonormalize_dim = sweep_kwargs.pop(
        "orthonormalize_generalized_dim",
        None,
    )
    orthonormalized_operator_dim = int(
        sweep_kwargs.pop("orthonormalized_operator_dim", 512)
    )
    mps0 = _make_initial_mps(
        qcdmrg,
        target_sector=target_sector,
        initial_guess=initial_guess,
        bond_multiplicity=bond_multiplicity,
        seed=seed,
    )
    if nstates > 1:
        orthonormalize_generalized_dim = (
            orthonormalized_operator_dim
            if local_basis_policy == "orthonormalized_operator"
            else None
        )
        solver_kwargs = {
            "tol": 1.0e-7,
            "itermax": 30,
            "max_space": 96,
            "dense_fallback_dim": int(
                sweep_kwargs.pop("state_average_dense_fallback_dim", 8192)
            ),
            "allow_unconverged_roots": True,
            "use_block_preconditioner": False,
            "orthonormalize_generalized_dim": orthonormalize_generalized_dim,
            "orthonormalize_generalized_operator": (
                local_basis_policy == "orthonormalized_operator"
            ),
        }
    else:
        if requested_orthonormalize_dim is not None:
            orthonormalize_generalized_dim = int(requested_orthonormalize_dim)
        elif local_basis_policy == "orthonormalized_operator":
            # Match the block-DMRG local problem structure for small active
            # bonds by building an explicit metric-orthonormal local basis and
            # solving a standard Davidson problem in that basis. Larger bonds
            # stay on the packed metric-Krylov path unless callers raise this
            # cap explicitly.
            orthonormalize_generalized_dim = orthonormalized_operator_dim
        elif local_basis_policy == "mixed_canonical_standard":
            # The packed Davidson path metric-orthonormalizes its local Krylov
            # basis when a norm operator is present. That gives the block-DMRG
            # algorithmic structure without forcing dense metric transforms.
            orthonormalize_generalized_dim = None
        else:
            orthonormalize_generalized_dim = 256 if int(qcdmrg.ncas) >= 6 else None
        solver_kwargs = {
            "tol": 1.0e-8 if local_basis_policy == "orthonormalized_operator" else 1.0e-6,
            "tol_residual": (
                1.0e-8
                if local_basis_policy == "orthonormalized_operator"
                else None
            ),
            "itermax": 80 if local_basis_policy == "orthonormalized_operator" else 40,
            "max_space": 96 if local_basis_policy == "orthonormalized_operator" else 48,
            "dense_fallback_dim": 512,
            "orthonormalized_dense_dim": None,
            "orthonormalize_generalized_dim": orthonormalize_generalized_dim,
            "orthonormalize_generalized_operator": (
                local_basis_policy == "orthonormalized_operator"
            ),
            "use_block_preconditioner": True,
        }
    solver_kwargs.update(local_solver_kwargs or {})
    candidate_nstates = nstates
    if nstates > 1:
        root_selection_buffer = int(sweep_kwargs.pop("state_average_root_buffer", 2))
        candidate_nstates = int(
            sweep_kwargs.pop(
                "state_average_candidate_roots",
                nstates + max(0, root_selection_buffer),
            )
        )
        candidate_nstates = max(nstates, candidate_nstates)
        if (
            state_average_spin_projector
            and getattr(qcdmrg, "spatial_site_basis", "canonical") == "canonical"
        ):
            # The canonical spatial SU(2) basis still carries explicit spin
            # components on the singly occupied site.  Until the fully reduced
            # Wigner-Eckart Hamiltonian is the production path, keep enough
            # internal target-selection roots to skip the low triplet even when
            # callers request exactly ``nstates`` candidates.
            candidate_nstates = max(
                candidate_nstates,
                nstates + max(0, root_selection_buffer),
            )
        solver_kwargs["nstates"] = candidate_nstates
        local_weights = np.zeros(candidate_nstates, dtype=float)
        local_weights[: min(nstates, candidate_nstates)] = weights[: min(nstates, candidate_nstates)]
        local_weight_sum = float(np.sum(local_weights))
        if abs(local_weight_sum) <= 1.0e-15:
            local_weights[: min(nstates, candidate_nstates)] = 1.0
            local_weight_sum = float(np.sum(local_weights))
        solver_kwargs["weights"] = local_weights / local_weight_sum
        solver_kwargs.setdefault("couple_physical", not fully_reduced_sites)
        solver_kwargs.setdefault("filter_coupled_boundary", True)
        target_s = 0.5 * abs(float(qcdmrg.spin))
        solver_kwargs.setdefault("root_target_value", target_s * (target_s + 1.0))
        solver_kwargs.setdefault("root_target_tol", state_average_spin_tol)
        solver_kwargs.setdefault("root_selection_buffer", root_selection_buffer)
        state_average_projector_dim = sweep_kwargs.pop(
            "state_average_projector_dim",
            nstates + max(0, root_selection_buffer),
        )
        if state_average_projector_dim is not None:
            solver_kwargs.setdefault("root_projector_dim", int(state_average_projector_dim))
        solver_kwargs.setdefault(
            "root_projector_dense_dim",
            int(sweep_kwargs.pop("state_average_projector_dense_dim", 512)),
        )
        solver_kwargs.setdefault(
            "root_projector_block_dim",
            int(sweep_kwargs.pop("state_average_projector_block_dim", 512)),
        )
        solver_kwargs.setdefault(
            "root_projector_block_max_columns",
            int(sweep_kwargs.pop("state_average_projector_block_max_columns", 512)),
        )
    initial_multiroot_mps = (
        _make_state_average_multiroot_mps(
            qcdmrg,
            target_sector=target_sector,
            nroots=candidate_nstates,
            weights=solver_kwargs.get("weights"),
            initial_guess=initial_guess,
            bond_multiplicity=bond_multiplicity,
            seed=seed,
        )
        if nstates > 1
        else None
    )
    root_target_mpo_factors = (
        _spin_square_mpo_factors(qcdmrg)
        if nstates > 1 and state_average_spin_projector
        else None
    )
    canonical_local_norm = sweep_kwargs.pop(
        "canonical_local_norm",
        (
            "force"
            if local_basis_policy == "mixed_canonical_standard" and nstates == 1
            else False
        ),
    )
    family_kernel_backend = sweep_kwargs.pop("family_kernel_backend", None)
    family_dense_threshold = sweep_kwargs.pop("family_dense_threshold", None)
    family_dense_total_provided = "family_dense_max_total_elements" in sweep_kwargs
    family_dense_max_total_elements = sweep_kwargs.pop("family_dense_max_total_elements", None)
    family_policy_kwargs = {
        "backend": family_kernel_backend,
        "dense_threshold": family_dense_threshold,
    }
    if family_dense_total_provided:
        family_policy_kwargs["dense_max_total_elements"] = family_dense_max_total_elements
    family_policy_previous = configure_complementary_family_kernel_policy(
        **family_policy_kwargs
    )
    family_policy_active = get_complementary_family_kernel_policy()
    if local_basis_policy == "orthonormalized_operator":
        default_max_bond_mode = "per_sector"
    elif block2_like_state_average:
        default_max_bond_mode = "states"
    else:
        default_max_bond_mode = "reduced"
    max_bond_mode = sweep_kwargs.pop("max_bond_mode", default_max_bond_mode)

    try:
        sweep_initial_state = initial_multiroot_mps if initial_multiroot_mps is not None else mps0
        result = run_sweeps(
            sweep_initial_state,
            nsweeps=int(nsweeps),
            mpo_factors=mpo_factors,
            root_target_mpo_factors=root_target_mpo_factors,
            max_bond=int(max_bond),
            max_bond_mode=max_bond_mode,
            canonical_local_norm=canonical_local_norm,
            prefer_reduced_local_operator=sweep_kwargs.pop("prefer_reduced_local_operator", True),
            store_orthonormal_renormalized_operators=(
                local_basis_policy == "orthonormalized_operator"
            ),
            require_block_sparse_renormalized_operator_table=(
                local_basis_policy == "orthonormalized_operator" and nstates > 1
            ),
            require_symbolic_renormalized_operators=(
                local_basis_policy == "orthonormalized_operator"
            ),
            complementary_operator_families=complementary_operator_families,
            warm_start_bonds=sweep_kwargs.pop("warm_start_bonds", True),
            mixer_zero_block_noise_scale=sweep_kwargs.pop("mixer_zero_block_noise_scale", 1.0e-5),
            mixer_zero_block_noise_seed=sweep_kwargs.pop("mixer_zero_block_noise_seed", seed + 4),
            mixer_nsweeps=sweep_kwargs.pop("mixer_nsweeps", 2),
            record_post_update_energy=sweep_kwargs.pop(
                "record_post_update_energy",
                debug_state_average,
            ),
            state_average_local_norm=sweep_kwargs.pop(
                "state_average_local_norm",
                nstates > 1,
            ),
            conv_tol=conv_tol,
            measure=sweep_kwargs.pop("measure", _qchem_sweep_measure),
            local_solver_kwargs=solver_kwargs,
            evaluate_root_energies_each_sweep=sweep_kwargs.pop(
                "evaluate_root_energies_each_sweep",
                True,
            ),
            verbose=verbose,
            **sweep_kwargs,
        )
    finally:
        configure_complementary_family_kernel_policy(**family_policy_previous)
    for entry in result.get("history", []):
        entry["local_basis_policy"] = (
            "block2_like" if block2_like_state_average else local_basis_policy
        )
        entry["max_bond_mode"] = max_bond_mode
        entry["family_kernel_policy"] = dict(family_policy_active)
        for objective in entry.get("bond_objectives", []) or []:
            objective.setdefault(
                "local_basis_policy",
                "block2_like" if block2_like_state_average else local_basis_policy,
            )
    energy = result["best_energy"]
    if energy is None:
        for entry in reversed(result["history"]):
            if "energy" in entry:
                energy = entry["energy"]
                break
    if energy is None:
        energy = np.nan
    energy = float(np.real(energy))
    state_energies = result.get("state_energies")
    if state_energies is None:
        for entry in reversed(result["history"]):
            for objective in reversed(entry.get("bond_objectives") or []):
                if "state_energies" in objective:
                    state_energies = objective["state_energies"]
                    break
            if state_energies is not None:
                break
    if state_energies is not None:
        state_energies = [float(np.real(x)) for x in state_energies]
    if active_hamiltonian is not None and result["history"]:
        result["history"][-1]["hamiltonian_system"] = active_hamiltonian.initialize_system_kwargs()
        result["history"][-1]["hamiltonian_symmetry"] = active_hamiltonian.symmetry
        if complementary_operator_families is not None and hasattr(
            complementary_operator_families,
            "as_metadata",
        ):
            result["history"][-1]["hamiltonian_complementary_operators"] = (
                complementary_operator_families.as_metadata()
            )
    ground_state = result["mps"]
    root_mps = result.get("root_mps")
    state_s2 = None
    if nstates > 1 and root_mps is not None:
        root_mps, state_energies, state_s2, root_selection_info = _finalize_spin_targeted_roots(
            qcdmrg,
            root_mps,
            nstates,
            compute_s2=state_average_validate_spin,
            select_by_spin=(
                False
                if fully_reduced_sites
                else (
                    not state_average_spin_projector
                    or int(candidate_nstates) > int(nstates)
                )
            ),
            spin_tol=state_average_spin_tol,
        )
        if result["history"]:
            n_selected = len(state_energies)
            selected_weights = np.asarray(weights[:n_selected], dtype=float).reshape(-1)
            selected_weight_sum = float(np.sum(selected_weights))
            if abs(selected_weight_sum) <= 1.0e-15:
                selected_weights = np.ones(int(n_selected), dtype=float) / int(n_selected)
            else:
                selected_weights = selected_weights / selected_weight_sum
            result["history"][-1]["target_state_energies"] = list(state_energies)
            result["history"][-1]["state_energies"] = list(state_energies)
            result["history"][-1]["state_average_weights"] = [
                float(x) for x in selected_weights
            ]
            result["history"][-1]["state_average_energy"] = float(
                np.dot(selected_weights, np.asarray(state_energies, dtype=float))
            )
            result["history"][-1]["state_average_candidate_roots"] = int(candidate_nstates)
            result["history"][-1].update(root_selection_info)
            if state_s2 is not None:
                result["history"][-1]["state_s2"] = list(state_s2)
                if root_selection_info.get("target_spin_valid") is None:
                    target_s = 0.5 * abs(float(qcdmrg.spin))
                    target_s2 = target_s * (target_s + 1.0)
                    result["history"][-1]["target_spin_valid"] = all(
                        abs(float(x) - target_s2) <= state_average_spin_tol
                        for x in state_s2
                    )
        if (
            state_average_validate_spin
            and result["history"]
            and result["history"][-1].get("target_spin_valid") is False
        ):
            raise RuntimeError(
                "SU(2) state-averaged sweep did not produce target-spin roots. "
                "Enable state_average_spin_projector=True for a local spin projector, "
                "disable state_average_validate_spin if contaminated roots are acceptable, "
                "or compare against pyqed.qchem.mcscf.direct_ci.CASCI as a separate reference."
            )
    active_energy = state_energies if state_energies is not None else energy
    if int(verbose) >= 1 and result["history"]:
        final_metric = result["history"][-1].get("metric")
        metric_text = "-" if final_metric is None else f"{float(final_metric):.3e}"
        tol_text = "-" if conv_tol is None else f"{float(conv_tol):.3e}"
        backend_text = result["history"][-1].get("backend", "sweep")
        print(
            "  DMRG convergence: "
            f"backend={backend_text} | "
            f"converged={bool(result['converged'])} | "
            f"sweeps={int(result['ncompleted'])} | "
            f"metric={metric_text} | "
            f"conv_tol={tol_text}"
        )
    return NonAbelianDMRGResult(
        e_tot=active_energy,
        ground_state=ground_state,
        states=root_mps if root_mps is not None else [ground_state.copy() for _ in range(nstates)],
        history=result["history"],
        converged=bool(result["converged"]),
        ncompleted=int(result["ncompleted"]),
        target_sector=target_sector,
        multiroot_state=result.get("multiroot_mps"),
    )
