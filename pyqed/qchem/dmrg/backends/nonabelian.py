"""Non-Abelian backend adapter for qchem spatial-orbital DMRG."""

from __future__ import annotations

import time
import numpy as np

from pyqed.mps.nonabelian import (
    MPS,
    MultiRootMPS,
    build_random_reduced_spatial_mps,
    contract_chain_expectation,
    run_sweeps,
    spatial_target_sector,
)
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.mps.nonabelian.su2_kernel import cpp_available as su2_cpp_available
from pyqed.mps.nonabelian.renormalized import (
    configure_su2_kernel_policy,
    get_su2_kernel_policy,
)


ORTHONORMALIZED_OPERATOR_ITERMAX_DEFAULT = 30


class SU2DMRG:
    """Stateful owner for one spatial-orbital SU(2) DMRG calculation."""

    backend = "su2"

    def __init__(self):
        self.engine = None
        self.energy = None
        self.energies = None
        self.nstates = 0
        self.weights = None
        self.state_average_energy = None
        self.e_active = None
        self.e_core = None
        self.e_tot = None
        self.includes_core_energy = False
        self.ground_state = None
        self.states = []
        self.history = []
        self.diagnostics = {}
        self.converged = False
        self.ncompleted = 0
        self.ncompleted_half_sweeps = 0
        self.max_sweeps = 0
        self.target_sector = None
        self.success = False
        self.message = "not run"

    def run(self, qcdmrg, **kwargs):
        _run_spatial_qchem_dmrg(self, qcdmrg, **kwargs)
        return self


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


def _make_initial_mps(
    qcdmrg,
    *,
    target_sector,
    initial_guess=None,
    bond_multiplicity=2,
    seed=7,
):
    guess = qcdmrg.init_guess if initial_guess is None else initial_guess
    if isinstance(guess, MPS):
        mps = guess.copy()
        mps.target_sector = target_sector
        return mps

    if not isinstance(guess, str):
        raise TypeError(f"Unsupported non-Abelian initial guess type: {type(guess)}")

    method = guess.lower()
    if method not in {"hf", "product", "cid", "cisd", "random", "previous"}:
        raise ValueError(f"Unsupported non-Abelian initial guess {guess!r}.")
    # A fully reduced SU(2) site has no spin-projection product state. Seed the
    # requested target-sector manifold directly for every public guess label.
    sites = build_random_reduced_spatial_mps(
        qcdmrg.ncas,
        target_sector=target_sector,
        bond_multiplicity=bond_multiplicity,
        seed=seed,
    )
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


def _expectation_from_nonabelian_mps(
    state,
    mpo_factors,
    *,
    moving_environment=None,
):
    numerator = contract_chain_expectation(
        state.sites,
        mpo_factors,
        moving_environment=moving_environment,
    )
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
    precomputed_energies=None,
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
    precomputed_energies
        Optional normalized root expectations already evaluated by the C++
        moving environment after terminal truncation.

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
    provided_energies = None
    if precomputed_energies is not None:
        provided_energies = np.asarray(precomputed_energies, dtype=float).reshape(-1)
        if provided_energies.size != len(root_mps):
            raise ValueError(
                "C++ state-average energies must match the exported roots."
            )
    for root_idx, state in enumerate(root_mps):
        if provided_energies is not None:
            energy = float(provided_energies[root_idx])
        else:
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


def run_spatial_qchem_dmrg(qcdmrg, **kwargs):
    """Create, run, and return an :class:`SU2DMRG` solver."""
    return SU2DMRG().run(qcdmrg, **kwargs)


def _native_su2_hamiltonian(qcdmrg):
    """Validate and return the production C++ normal/complementary owner."""

    if not su2_cpp_available():
        raise RuntimeError(
            "SU(2)-QCDMRG requires the compiled native C++ extension. "
            "Rebuild PyQED with PYQED_BUILD_EXTENSIONS=1 and "
            "PYQED_EXTENSION_GROUPS=mps."
        )
    active_hamiltonian = getattr(qcdmrg, "_active_hamiltonian", None)
    info = (
        {}
        if active_hamiltonian is None
        else (getattr(active_hamiltonian, "info", None) or {})
    )
    if active_hamiltonian is None or not info.get(
        "normal_complementary_production",
        False,
    ):
        raise RuntimeError(
            "SU(2)-QCDMRG requires a native C++ normal/complementary "
            "Hamiltonian. Rebuild the active Hamiltonian through DMRG.build()."
        )
    if info.get("spatial_site_basis") != "fully_reduced_su2":
        raise RuntimeError(
            "The native SU(2)-QCDMRG backend requires fully reduced SU(2) sites."
        )
    if info.get("python_reduced_terms_materialized", True):
        raise RuntimeError(
            "The production SU(2) Hamiltonian unexpectedly materialized "
            "Python reduced terms."
        )
    owner = getattr(active_hamiltonian, "moving_environment", None)
    if owner is None:
        raise RuntimeError(
            "The native SU(2) Hamiltonian does not own an SU2MovingEnvironment."
        )
    factors = tuple(active_hamiltonian.mpo)
    if not factors or any(
        getattr(factor, "normal_complementary_owner", None) is not owner
        or getattr(factor, "dense_blocks", None)
        or getattr(factor, "reduced_terms", None)
        for factor in factors
    ):
        raise RuntimeError(
            "The active SU(2) Hamiltonian is not a lightweight native "
            "normal/complementary route view."
        )
    return active_hamiltonian, owner, list(factors)


def _run_spatial_qchem_dmrg(
    solver,
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
    n_threads=None,
    verbose=0,
    **sweep_kwargs,
):
    """
    Run the lower-level non-Abelian sweep engine on a qchem spatial MPO.

    The qchem wrapper owns chemistry concerns: active-space integrals, core
    energy, physical site ordering, and reporting. This adapter only converts
    those objects to the non-Abelian MPS sweep API. ``nsweeps`` counts complete
    left-to-right plus right-to-left sweeps.
    """
    nsweeps = int(nsweeps)
    if nsweeps < 1:
        raise ValueError("nsweeps must be positive.")
    if n_threads is not None:
        if isinstance(n_threads, (bool, np.bool_)) or not isinstance(
            n_threads, (int, np.integer)
        ):
            raise TypeError("n_threads must be a positive integer or None.")
        if int(n_threads) < 1:
            raise ValueError("n_threads must be positive or None.")
        n_threads = int(n_threads)
    if qcdmrg.site != "spatial":
        raise NotImplementedError("The non-Abelian qchem DMRG backend currently requires site='spatial'.")
    if qcdmrg.H is None:
        raise ValueError("DMRG Hamiltonian MPO is not built. Call build() before the backend.")
    if qcdmrg.spin_purification:
        raise NotImplementedError("Spin-purification penalties are not supported by the SU(2) backend.")
    for removed_option in (
        "su2_kernel_backend",
        "su2_reference_complementary_families",
        "su2_force_family_table",
        "family_kernel_backend",
        "family_dense_threshold",
        "family_dense_max_total_elements",
        "direct_orthonormal_block_max_elements",
        "direct_orthonormal_dense_max_elements",
        "su2_qchem_direct_parent_blocks",
        "su2_qchem_direct_parent_block_max_elements",
        "canonical_local_norm",
        "prefer_reduced_local_operator",
        "local_basis_policy",
        "orthonormalized_operator_dim",
        "orthonormalize_generalized_dim",
        "max_bond_mode",
        "state_average_dense_fallback_dim",
        "state_average_projector_dim",
        "state_average_projector_dense_dim",
        "state_average_projector_block_dim",
        "state_average_projector_block_max_columns",
    ):
        if removed_option in sweep_kwargs:
            raise TypeError(
                f"{removed_option} was removed. SU(2)-QCDMRG always uses "
                "the native C++ normal/complementary backend."
            )
    for incompatible_option in (
        "record_post_update_energy",
        "state_average_local_norm",
        "state_average_root_environments",
        "state_average_spin_projector",
    ):
        if bool(sweep_kwargs.get(incompatible_option, False)):
            raise TypeError(
                f"{incompatible_option}=True is unavailable in the C++-only "
                "SU(2)-QCDMRG backend."
            )
    if float(sweep_kwargs.get("mixer_zero_block_noise_scale", 0.0)) != 0.0:
        raise TypeError(
            "mixer_zero_block_noise_scale must be zero in the C++-only "
            "SU(2)-QCDMRG backend."
        )
    active_hamiltonian, su2_moving_environment, mpo_factors = (
        _native_su2_hamiltonian(qcdmrg)
    )
    complementary_operator_families = None
    normal_complementary_production = True
    fully_reduced_sites = getattr(qcdmrg, "spatial_site_basis", "canonical") in {
        "fully_reduced",
        "fully_reduced_su2",
    }
    if not fully_reduced_sites:
        raise RuntimeError(
            "SU(2)-QCDMRG requires the fully reduced native C++ site basis."
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
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("weights must be nonnegative.")
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("weights must have a positive sum.")
        weights = weights / weight_sum

    target_sector = spatial_target_sector(qcdmrg.nelecas, int(qcdmrg.spin))
    if max_bond is None:
        max_bond = qcdmrg.D

    state_average_spin_tol = float(sweep_kwargs.pop("state_average_spin_tol", 1.0e-6))
    state_average_validate_spin = bool(
        sweep_kwargs.pop("state_average_validate_spin", False)
    )
    sweep_kwargs.pop("state_average_spin_projector", False)
    debug_state_average = bool(sweep_kwargs.pop("debug_state_average", False))
    if debug_state_average and int(verbose) < 2:
        verbose = 2
    local_basis_policy = "native_cpp"
    dual_right_environment = bool(
        sweep_kwargs.pop(
            "su2_dual_right_environment",
            fully_reduced_sites,
        )
    )
    for factor in mpo_factors:
        if getattr(factor, "normal_complementary_plan", None) is not None:
            object.__setattr__(
                factor,
                "normal_complementary_right_dual",
                dual_right_environment,
            )
    state_initialization_started = time.perf_counter()
    mps0 = _make_initial_mps(
        qcdmrg,
        target_sector=target_sector,
        initial_guess=initial_guess,
        bond_multiplicity=bond_multiplicity,
        seed=seed,
    )
    state_initialization_seconds = (
        time.perf_counter() - state_initialization_started
    )
    local_solver_kwargs = dict(local_solver_kwargs or {})
    for removed_local_option in (
        "dense_fallback_dim",
        "orthonormalized_dense_dim",
        "orthonormalize_generalized_dim",
        "orthonormalize_generalized_operator",
        "use_block_preconditioner",
        "couple_physical",
        "filter_coupled_boundary",
        "root_target_value",
        "root_target_tol",
        "root_selection_buffer",
        "root_projector_dim",
        "root_projector_dense_dim",
        "root_projector_block_dim",
        "root_projector_block_max_columns",
    ):
        if removed_local_option in local_solver_kwargs:
            raise TypeError(
                f"local_solver_kwargs[{removed_local_option!r}] was removed. "
                "The C++-owned Davidson solver controls its reduced basis "
                "internally."
            )
    if nstates > 1:
        solver_kwargs = {
            "tol": 1.0e-7,
            "itermax": 30,
            "max_space": 96,
        }
    else:
        solver_kwargs = {
            "tol": 1.0e-8,
            "itermax": ORTHONORMALIZED_OPERATOR_ITERMAX_DEFAULT,
            "max_space": 48,
        }
    solver_kwargs.update(local_solver_kwargs)
    candidate_nstates = nstates
    if nstates > 1:
        root_selection_buffer = int(
            sweep_kwargs.pop(
                "state_average_root_buffer",
                0,
            )
        )
        candidate_nstates = int(
            sweep_kwargs.pop(
                "state_average_candidate_roots",
                nstates + max(0, root_selection_buffer),
            )
        )
        candidate_nstates = max(nstates, candidate_nstates)
        solver_kwargs["nstates"] = candidate_nstates
        local_weights = np.zeros(candidate_nstates, dtype=float)
        local_weights[: min(nstates, candidate_nstates)] = weights[: min(nstates, candidate_nstates)]
        local_weight_sum = float(np.sum(local_weights))
        if abs(local_weight_sum) <= 1.0e-15:
            local_weights[: min(nstates, candidate_nstates)] = 1.0
            local_weight_sum = float(np.sum(local_weights))
        solver_kwargs["weights"] = local_weights / local_weight_sum
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
    root_target_mpo_factors = None
    debug_su2_kernel_check = bool(
        sweep_kwargs.pop("debug_su2_kernel_check", False)
    )
    debug_su2_kernel_check_tol = sweep_kwargs.pop(
        "debug_su2_kernel_check_tol",
        None,
    )
    verify_returned_energy = bool(
        sweep_kwargs.pop("verify_returned_mps_energy", False)
    )
    su2_kernel_previous = configure_su2_kernel_policy(
        backend="cpp",
        debug_check=debug_su2_kernel_check,
        debug_check_tol=debug_su2_kernel_check_tol,
    )
    su2_kernel_active = get_su2_kernel_policy()
    if su2_kernel_active.get("actual") != "cpp":
        raise RuntimeError(
            "SU(2)-QCDMRG failed to activate the native C++ backend."
        )
    su2_moving_environment.set_num_threads(
        1 if n_threads is None else n_threads
    )
    cpp_state_average = bool(int(nstates) > 1)
    if cpp_state_average:
        initial_multiroot_mps.canonicalize_shared(
            0,
            max_bond=int(max_bond),
            max_bond_mode="reduced",
        )
        su2_moving_environment.install_state_average_mps(
            initial_multiroot_mps.roots,
            initial_multiroot_mps.weights,
            0,
        )
    max_bond_mode = "reduced"
    try:
        sweep_initial_state = initial_multiroot_mps if initial_multiroot_mps is not None else mps0
        result = run_sweeps(
            sweep_initial_state,
            nsweeps=2 * nsweeps,
            converge_on_full_sweeps=True,
            mpo_factors=mpo_factors,
            root_target_mpo_factors=root_target_mpo_factors,
            max_bond=int(max_bond),
            max_bond_mode=max_bond_mode,
            retain_sector_topology=sweep_kwargs.pop(
                "retain_sector_topology",
                False,
            ),
            canonical_local_norm=False,
            prefer_reduced_local_operator=True,
            store_orthonormal_renormalized_operators=False,
            require_block_sparse_renormalized_operator_table=False,
            require_symbolic_renormalized_operators=False,
            require_cpp_owned_sweeps=True,
            complementary_operator_families=None,
            materialize_complementary_family_operator_tables=False,
            su2_moving_environment=su2_moving_environment,
            renormalized_operator_cache_max_size=int(
                sweep_kwargs.pop(
                    "renormalized_operator_cache_max_size",
                    1,
                )
            ),
            warm_start_bonds=sweep_kwargs.pop(
                "warm_start_bonds",
                False,
            ),
            compact_history_updates=sweep_kwargs.pop(
                "compact_history_updates",
                True,
            ),
            mixer_zero_block_noise_scale=sweep_kwargs.pop(
                "mixer_zero_block_noise_scale",
                0.0,
            ),
            mixer_zero_block_noise_seed=sweep_kwargs.pop("mixer_zero_block_noise_seed", seed + 4),
            mixer_nsweeps=2 * int(sweep_kwargs.pop("mixer_nsweeps", 2)),
            record_post_update_energy=sweep_kwargs.pop(
                "record_post_update_energy",
                False,
            ),
            compute_final_expectation=sweep_kwargs.pop(
                "compute_final_expectation",
                not (normal_complementary_production and int(nstates) == 1),
            ),
            state_average_local_norm=sweep_kwargs.pop(
                "state_average_local_norm",
                False,
            ),
            state_average_root_environments=sweep_kwargs.pop(
                "state_average_root_environments",
                False,
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
        configure_su2_kernel_policy(
            backend=su2_kernel_previous["backend"],
            debug_check=su2_kernel_previous["debug_check"],
            debug_check_tol=su2_kernel_previous["debug_check_tol"],
        )
    native_bond_updates = 0
    for entry in result.get("history", []):
        entry["local_basis_policy"] = local_basis_policy
        entry["max_bond_mode"] = max_bond_mode
        entry["su2_kernel_policy"] = dict(su2_kernel_active)
        entry["threading"] = dict(su2_moving_environment.threading_info)
        for objective in entry.get("bond_objectives", []) or []:
            native_bond_updates += 1
            if (
                objective.get("cpp_owned_half_sweep") is not True
                or objective.get("no_python_bond_callbacks") is not True
            ):
                raise RuntimeError(
                    "SU(2)-QCDMRG left the required C++-owned half-sweep path."
                )
            objective.setdefault(
                "local_basis_policy",
                local_basis_policy,
            )
            table_stats = objective.get("renormalized_operator_table_stats") or {}
            reported_backend = table_stats.get("su2_kernel_backend_actual")
            if reported_backend not in {None, "cpp"}:
                raise RuntimeError(
                    "The native SU(2)-QCDMRG sweep entered a non-C++ local "
                    f"backend ({reported_backend!r})."
                )
        entry["su2_kernel_backend_actual"] = "cpp"
        entry["backend_actual"] = "cpp_su2_normal_complementary"
    if native_bond_updates == 0:
        raise RuntimeError("SU(2)-QCDMRG produced no C++-owned bond updates.")
    energy = result["best_energy"]
    if energy is None:
        for entry in reversed(result["history"]):
            if "energy" in entry:
                energy = entry["energy"]
                break
    if energy is None:
        energy = np.nan
    energy = float(np.real(energy))
    state_energies = result.get("state_energies") if nstates > 1 else None
    if nstates > 1 and state_energies is None:
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
        result["history"][-1]["state_initialization_seconds"] = float(
            state_initialization_seconds
        )
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
    if nstates == 1:
        if verify_returned_energy:
            returned_energy = _expectation_from_nonabelian_mps(
                ground_state,
                mpo_factors,
                moving_environment=(
                    None
                    if active_hamiltonian is None
                    else active_hamiltonian.moving_environment
                ),
            )
            if result["history"]:
                result["history"][-1]["returned_mps_energy"] = float(returned_energy)
            energy = float(returned_energy)
            result["best_energy"] = float(returned_energy)
        elif result["history"]:
            result["history"][-1]["returned_mps_energy"] = float(energy)
    root_mps = result.get("root_mps")
    state_s2 = None
    if nstates > 1 and root_mps is not None:
        root_mps, state_energies, state_s2, root_selection_info = _finalize_spin_targeted_roots(
            qcdmrg,
            root_mps,
            nstates,
            compute_s2=state_average_validate_spin,
            select_by_spin=False,
            spin_tol=state_average_spin_tol,
            precomputed_energies=(state_energies if cpp_state_average else None),
        )
        if fully_reduced_sites and state_s2 is None:
            target_s = 0.5 * abs(float(qcdmrg.spin))
            target_s2 = target_s * (target_s + 1.0)
            state_s2 = [float(target_s2)] * len(state_energies)
            root_selection_info["candidate_state_s2"] = list(state_s2)
            root_selection_info["target_spin_valid"] = True
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
                "Disable state_average_validate_spin if contaminated roots are acceptable, "
                "or compare against pyqed.qchem.mcscf.direct_ci.CASCI as a separate reference."
            )
    ncompleted_half_sweeps = int(result["ncompleted"])
    ncompleted_sweeps = ncompleted_half_sweeps // 2
    for half_index, row in enumerate(result["history"]):
        row["half_sweep"] = half_index + 1
        row["sweep"] = half_index // 2 + 1
        row["sweep_complete"] = bool(half_index % 2 == 1)
    if int(verbose) >= 1 and result["history"]:
        final_metric = result["history"][-1].get("metric")
        metric_text = "-" if final_metric is None else f"{float(final_metric):.3e}"
        tol_text = "-" if conv_tol is None else f"{float(conv_tol):.3e}"
        backend_text = result["history"][-1].get("backend", "sweep")
        print(
            "  DMRG convergence: "
            f"backend={backend_text} | "
            f"converged={bool(result['converged'])} | "
            f"sweeps={ncompleted_sweeps} | "
            f"metric={metric_text} | "
            f"conv_tol={tol_text}"
        )
    states = list(root_mps) if root_mps is not None else [ground_state]
    energies = np.asarray(
        state_energies if state_energies is not None else [energy],
        dtype=float,
    ).reshape(-1)
    selected_weights = np.asarray(weights[: len(energies)], dtype=float)
    selected_weight_sum = float(np.sum(selected_weights))
    if selected_weight_sum <= 0.0:
        selected_weights = np.ones(len(energies), dtype=float) / len(energies)
    else:
        selected_weights /= selected_weight_sum

    solver.engine = su2_moving_environment
    solver.energy = float(energies[0])
    solver.energies = energies
    solver.nstates = len(states)
    solver.weights = selected_weights
    solver.state_average_energy = float(np.dot(selected_weights, energies))
    solver.e_active = float(energies[0]) if nstates == 1 else energies.copy()
    solver.e_tot = solver.e_active
    solver.includes_core_energy = bool(normal_complementary_production)
    solver.ground_state = states[0]
    solver.states = states
    solver.history = result["history"]
    solver.converged = bool(result["converged"])
    solver.ncompleted = ncompleted_sweeps
    solver.ncompleted_half_sweeps = ncompleted_half_sweeps
    solver.max_sweeps = nsweeps
    solver.target_sector = target_sector
    solver.success = bool(result["converged"])
    solver.message = (
        "converged"
        if result["converged"]
        else "completed requested SU(2) complete sweeps without convergence"
    )
    final_history = solver.history[-1] if solver.history else {}
    moving_stats = final_history.get("moving_environment_stats") or {}
    engine_stats = moving_stats.get("su2_moving_environment") or {}
    python_bond_callbacks = int(
        engine_stats.get("half_sweep_python_bond_callbacks", -1)
    )
    if python_bond_callbacks != 0:
        raise RuntimeError(
            "SU(2)-QCDMRG requires zero Python bond callbacks, got "
            f"{python_bond_callbacks}."
        )
    solver.diagnostics = {
        "kernel_backend": final_history.get("su2_kernel_backend_actual"),
        "kernel_policy": final_history.get("su2_kernel_policy"),
        "timing": final_history.get("timing"),
        "state_initialization_seconds": float(state_initialization_seconds),
        "memory_bytes": engine_stats.get("memory_bytes"),
        "route_count": engine_stats.get("factor_route_count"),
        "matvec_calls": engine_stats.get("matvec_calls"),
        "davidson_iterations": engine_stats.get("davidson_iterations"),
        "truncation_seconds": engine_stats.get("truncation_seconds"),
        "environment_seconds": engine_stats.get("boundary_update_seconds"),
        "owned_half_sweep_bonds": engine_stats.get("owned_half_sweep_bonds"),
        "python_bond_callbacks": python_bond_callbacks,
        "threading": final_history.get("threading"),
        "operator_schedule": {
            "kind": engine_stats.get("dense_pair_scheduler"),
            "executions": engine_stats.get("dense_pair_execution_count"),
            "waves": engine_stats.get("dense_pair_wave_count"),
            "max_wave_width": engine_stats.get(
                "dense_pair_max_wave_width"
            ),
        },
    }
    return solver
