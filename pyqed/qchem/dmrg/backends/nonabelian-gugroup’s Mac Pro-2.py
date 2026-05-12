"""Non-Abelian backend adapter for qchem spatial-orbital DMRG."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPS as DenseMPS
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.mps.nonabelian import (
    MPS,
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
    backend: str = "nonabelian"


class _DenseMPOView:
    """Small adapter for dense MPO contraction helpers."""

    def __init__(self, factors):
        self.factors = [
            core.as_dense() if hasattr(core, "as_dense") else np.asarray(core)
            for core in factors
        ]
        self.dims = [int(core.shape[2]) for core in self.factors]


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
            # seed the target-sector manifold directly with small random blocks.
            sites = build_random_reduced_spatial_mps(
                qcdmrg.ncas,
                target_sector=target_sector,
                bond_multiplicity=bond_multiplicity,
                seed=seed,
                scale=1.0e-5,
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


def _charge_basis_indices(nsites, nelec):
    """Return full spatial-basis indices with total particle number ``nelec``."""
    charges = np.array([0, 1, 1, 2], dtype=int)
    indices = []
    labels = []

    def visit(site, remaining, index, occ):
        if site == nsites:
            if remaining == 0:
                indices.append(index)
                labels.append(tuple(occ))
            return
        power = 4 ** (nsites - site - 1)
        for state, charge in enumerate(charges):
            if charge > remaining:
                continue
            occ.append(state)
            visit(site + 1, remaining - int(charge), index + state * power, occ)
            occ.pop()

    visit(0, int(nelec), 0, [])
    return np.asarray(indices, dtype=int), labels


def _spin_square_in_charge_basis(labels):
    """Build S^2 in the fixed-charge spatial occupation basis."""
    dim = len(labels)
    pos = {label: i for i, label in enumerate(labels)}
    sz = np.zeros(dim, dtype=float)
    sp = np.zeros((dim, dim), dtype=float)
    sm = np.zeros((dim, dim), dtype=float)

    for col, label in enumerate(labels):
        nup = sum(1 for state in label if state in (1, 3))
        ndn = sum(1 for state in label if state in (2, 3))
        sz[col] = 0.5 * (nup - ndn)
        for site, state in enumerate(label):
            if state == 2:
                flipped = list(label)
                flipped[site] = 1
                sp[pos[tuple(flipped)], col] += 1.0
            elif state == 1:
                flipped = list(label)
                flipped[site] = 2
                sm[pos[tuple(flipped)], col] += 1.0

    return np.diag(sz * sz) + 0.5 * (sp @ sm + sm @ sp)


def _dense_mps_from_spatial_vector(vector, nsites):
    tensor = np.asarray(vector, dtype=complex).reshape((4,) * int(nsites))
    factors = decompose(tensor, rank=tensor.size)
    return DenseMPS(factors, labels=["lv", "p", "rv"]).normalize()


def _spin_adapted_dense_roots(qcdmrg, nstates, *, max_dense_dim=4096, spin_tol=1.0e-7):
    """
    Exact target-spin root solver for small spatial active spaces.

    This is used for multi-state SU(2) calculations until the sweep engine
    carries root-specific center tensors across bonds.
    """
    nsites = int(qcdmrg.ncas)
    full_dim = 4 ** nsites
    if full_dim > int(max_dense_dim):
        raise NotImplementedError(
            "Multi-state SU(2) root solving currently uses a dense target-sector "
            f"fallback limited to full spatial dimension <= {max_dense_dim}; got {full_dim}."
        )

    charge_indices, labels = _charge_basis_indices(nsites, qcdmrg.nelecas)
    if charge_indices.size == 0:
        raise ValueError(
            f"No spatial determinants with nelec={qcdmrg.nelecas} for ncas={qcdmrg.ncas}."
        )

    h_dense = _mpo_to_dense_operator(_DenseMPOView(qcdmrg.H))
    h_charge = h_dense[np.ix_(charge_indices, charge_indices)]
    h_charge = 0.5 * (h_charge + h_charge.conj().T)
    s2_charge = _spin_square_in_charge_basis(labels)

    evals, evecs = np.linalg.eigh(h_charge)
    target_s = 0.5 * abs(float(qcdmrg.spin))
    target_s2 = target_s * (target_s + 1.0)
    candidates = []
    start = 0
    while start < evals.size:
        stop = start + 1
        while stop < evals.size and abs(float(evals[stop] - evals[start])) <= 1.0e-8:
            stop += 1
        sub = evecs[:, start:stop]
        s2_sub = sub.conj().T @ s2_charge @ sub
        s2_vals, s2_vecs = np.linalg.eigh(0.5 * (s2_sub + s2_sub.conj().T))
        for col, s2_val in enumerate(np.real(s2_vals)):
            vec_charge = sub @ s2_vecs[:, col]
            energy = float(np.real(np.vdot(vec_charge, h_charge @ vec_charge)))
            if abs(s2_val - target_s2) <= spin_tol:
                full_vec = np.zeros(full_dim, dtype=complex)
                full_vec[charge_indices] = vec_charge
                norm = np.linalg.norm(full_vec)
                if norm > 0.0:
                    full_vec /= norm
                candidates.append((energy, float(s2_val), full_vec))
        start = stop

    candidates.sort(key=lambda item: item[0])
    if len(candidates) < nstates:
        raise RuntimeError(
            f"Found only {len(candidates)} roots with target <S^2>={target_s2:.8g}; "
            f"requested {nstates}."
        )

    energies = [item[0] for item in candidates[:nstates]]
    s2_values = [item[1] for item in candidates[:nstates]]
    states = [_dense_mps_from_spatial_vector(item[2], nsites) for item in candidates[:nstates]]
    return energies, s2_values, states


def _run_dense_state_average_qchem(
    qcdmrg,
    *,
    nstates,
    target_sector,
    weights,
    max_dense_dim=4096,
    spin_tol=1.0e-7,
):
    energies, s2_values, states = _spin_adapted_dense_roots(
        qcdmrg,
        nstates,
        max_dense_dim=max_dense_dim,
        spin_tol=spin_tol,
    )
    history = [
        {
            "sweep": 0,
            "direction": "dense",
            "energy": float(energies[0]),
            "state_energies": [float(x) for x in energies],
            "state_average_energy": float(np.dot(weights, energies)),
            "state_average_weights": [float(x) for x in weights],
            "state_s2": [float(x) for x in s2_values],
            "backend": "dense_target_sector_su2",
            "target_spin_valid": True,
            "converged": True,
        }
    ]
    return NonAbelianDMRGResult(
        e_tot=[float(x) for x in energies],
        ground_state=states[0],
        states=states,
        history=history,
        converged=True,
        ncompleted=1,
        target_sector=target_sector,
        backend="nonabelian",
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


def _fixed_charge_spatial_dim(nsites, nelec):
    """Return the number of spatial occupation strings with ``nelec`` electrons."""
    nspin = 2 * int(nsites)
    nelec = int(nelec)
    if nelec < 0 or nelec > nspin:
        return 0
    return int(comb(nspin, nelec))


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
    candidate_roots = list(root_mps)
    candidate_energies = [
        _expectation_from_nonabelian_mps(state, qcdmrg.H)
        for state in candidate_roots
    ]
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
        "selected_root_indices": [int(x) for x in selected_indices],
        "final_root_selection_used": bool(root_selection_used),
        "target_spin_valid": target_spin_valid,
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
    state_average_backend = str(sweep_kwargs.pop("state_average_backend", "sweep")).lower()
    state_average_dense_dim = int(sweep_kwargs.pop("state_average_dense_dim", 4096))
    state_average_fallback_dense = bool(sweep_kwargs.pop("state_average_fallback_dense", False))
    state_average_reference_dense = bool(
        sweep_kwargs.pop("state_average_reference_dense", state_average_fallback_dense)
    )
    state_average_spin_tol = float(sweep_kwargs.pop("state_average_spin_tol", 1.0e-6))
    state_average_validate_spin = bool(sweep_kwargs.pop("state_average_validate_spin", True))
    state_average_spin_projector = bool(
        sweep_kwargs.pop("state_average_spin_projector", True)
    )
    local_basis_policy = str(
        sweep_kwargs.pop("local_basis_policy", "mixed_canonical_standard")
    ).lower().replace("-", "_")
    if local_basis_policy in {"block2", "block2_like"}:
        local_basis_policy = "orthonormalized_operator"
    elif local_basis_policy in {"orthonormalized", "metric_orthonormalized"}:
        local_basis_policy = "mixed_canonical_standard"
    if local_basis_policy not in {
        "mixed_canonical_standard",
        "orthonormalized_operator",
        "legacy_generalized",
    }:
        raise ValueError(
            "local_basis_policy must be 'mixed_canonical_standard', "
            "'orthonormalized_operator', or 'legacy_generalized'."
        )
    allow_experimental_block2_state_average = bool(
        sweep_kwargs.pop("allow_experimental_block2_state_average", False)
    )
    if (
        nstates > 1
        and local_basis_policy == "orthonormalized_operator"
        and int(qcdmrg.ncas) > 2
        and not allow_experimental_block2_state_average
    ):
        raise NotImplementedError(
            "block2_like/orthonormalized_operator state-averaged SU(2) DMRG "
            "is currently validated only for two-site smoke tests. Use the "
            "default mixed_canonical_standard SA path for larger active spaces, "
            "or pass allow_experimental_block2_state_average=True for debugging."
        )
    requested_orthonormalize_dim = sweep_kwargs.pop(
        "orthonormalize_generalized_dim",
        None,
    )
    orthonormalized_operator_dim = int(
        sweep_kwargs.pop("orthonormalized_operator_dim", 512)
    )
    if state_average_backend == "auto":
        state_average_backend = "sweep"
    if state_average_backend not in {"sweep", "dense"}:
        raise ValueError("state_average_backend must be 'sweep' or 'dense'.")
    dense_sa_dim = 4 ** int(qcdmrg.ncas)
    fixed_charge_sa_dim = _fixed_charge_spatial_dim(qcdmrg.ncas, qcdmrg.nelecas)
    if nstates > 1 and state_average_backend == "dense":
        if int(verbose) >= 1:
            print(
                "  [SU2-SA reference] Using dense target-spin solver (not DMRG) "
                f"(full_dim={dense_sa_dim}, fixed_charge_dim={fixed_charge_sa_dim}, "
                f"max_dense_dim={state_average_dense_dim})"
            )
        return _run_dense_state_average_qchem(
            qcdmrg,
            nstates=nstates,
            target_sector=target_sector,
            weights=weights,
            max_dense_dim=state_average_dense_dim,
            spin_tol=state_average_spin_tol,
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
            "tol": 1.0e-10,
            "itermax": 30,
            "max_space": 96,
            "dense_fallback_dim": 512,
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
        if not state_average_spin_projector:
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
        solver_kwargs.setdefault("couple_physical", True)
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
    root_target_mpo_factors = (
        _spin_square_mpo_factors(qcdmrg)
        if nstates > 1 and state_average_spin_projector
        else None
    )
    evaluate_root_energies_each_sweep = bool(
        sweep_kwargs.pop("evaluate_root_energies_each_sweep", nstates > 1)
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

    try:
        if nstates > 1:
            sweep_kwargs.setdefault("cutoff", 0.0)
        result = run_sweeps(
            mps0,
            nsweeps=int(nsweeps),
            mpo_factors=mpo_factors,
            root_target_mpo_factors=root_target_mpo_factors,
            max_bond=int(max_bond),
            max_bond_mode=sweep_kwargs.pop("max_bond_mode", "reduced"),
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
            conv_tol=None if nstates > 1 else conv_tol,
            local_solver_kwargs=solver_kwargs,
            evaluate_root_energies_each_sweep=evaluate_root_energies_each_sweep,
            verbose=verbose,
            **sweep_kwargs,
        )
    finally:
        configure_complementary_family_kernel_policy(**family_policy_previous)
    for entry in result.get("history", []):
        entry["local_basis_policy"] = local_basis_policy
        entry["family_kernel_policy"] = dict(family_policy_active)
        for objective in entry.get("bond_objectives", []) or []:
            objective.setdefault("local_basis_policy", local_basis_policy)
    energy = result["best_energy"]
    if energy is None:
        energy = result["history"][-1]["energy"]
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
            select_by_spin=not state_average_spin_projector,
            spin_tol=state_average_spin_tol,
        )
        if result["history"]:
            result["history"][-1]["target_state_energies"] = list(state_energies)
            result["history"][-1]["state_energies"] = list(state_energies)
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
            if state_average_reference_dense:
                try:
                    dense_reference = _run_dense_state_average_qchem(
                        qcdmrg,
                        nstates=nstates,
                        target_sector=target_sector,
                        weights=weights,
                        max_dense_dim=state_average_dense_dim,
                        spin_tol=state_average_spin_tol,
                    )
                except (RuntimeError, NotImplementedError, ValueError) as exc:
                    result["history"][-1]["dense_reference_error"] = str(exc)
                else:
                    dense_energies = np.asarray(dense_reference.e_tot, dtype=float)
                    sweep_energies = np.asarray(state_energies, dtype=float)
                    result["history"][-1]["dense_reference_backend"] = "dense_target_sector_su2"
                    result["history"][-1]["dense_reference_state_energies"] = [
                        float(x) for x in dense_energies
                    ]
                    result["history"][-1]["dense_reference_state_s2"] = list(
                        dense_reference.history[0].get("state_s2", [])
                    )
                    if dense_energies.shape == sweep_energies.shape:
                        result["history"][-1]["dense_reference_energy_errors"] = [
                            float(x)
                            for x in (sweep_energies - dense_energies)
                        ]
        if (
            state_average_validate_spin
            and result["history"]
            and result["history"][-1].get("target_spin_valid") is False
        ):
            raise RuntimeError(
                "SU(2) state-averaged sweep did not produce target-spin roots. "
                "Enable state_average_spin_projector=True for a local spin projector, "
                "set state_average_reference_dense=True to record dense reference "
                "energies for debugging, or disable state_average_validate_spin if "
                "contaminated roots are acceptable."
            )
    active_energy = state_energies if state_energies is not None else energy
    return NonAbelianDMRGResult(
        e_tot=active_energy,
        ground_state=ground_state,
        states=root_mps if root_mps is not None else [ground_state.copy() for _ in range(nstates)],
        history=result["history"],
        converged=bool(result["converged"]),
        ncompleted=int(result["ncompleted"]),
        target_sector=target_sector,
    )
