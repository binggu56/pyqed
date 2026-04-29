"""Non-Abelian backend adapter for qchem spatial-orbital DMRG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPS as DenseMPS
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.mps.nonabelian import (
    MPS,
    build_product_spatial_mps,
    build_random_spatial_mps,
    contract_chain_expectation,
    run_sweeps,
    spatial_target_sector,
)
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


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
        self.factors = [np.asarray(core) for core in factors]
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
    if method in {"hf", "product"}:
        sites = build_product_spatial_mps(
            _hf_spatial_labels(qcdmrg.nelecas, qcdmrg.ncas, qcdmrg.spin),
            enrich_bond_sectors=True,
            bond_multiplicity=bond_multiplicity,
            zero_block_noise_scale=1.0e-5,
            zero_block_noise_seed=seed,
        )
    elif method in {"cid", "cisd", "random", "previous"}:
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


def _finalize_spin_targeted_roots(qcdmrg, root_mps, nstates, *, compute_s2=True):
    if not root_mps:
        return root_mps, None, None
    selected_roots = list(root_mps[: int(nstates)])
    state_energies = [
        _expectation_from_nonabelian_mps(state, qcdmrg.H)
        for state in selected_roots
    ]
    s2_values = None
    if compute_s2:
        s2_factors = _spin_square_mpo_factors(qcdmrg)
        s2_values = [
            _expectation_from_nonabelian_mps(state, s2_factors)
            for state in selected_roots
        ]
    return (
        selected_roots,
        [float(x) for x in state_energies],
        None if s2_values is None else [float(x) for x in s2_values],
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
    state_average_spin_tol = float(sweep_kwargs.pop("state_average_spin_tol", 1.0e-6))
    state_average_validate_spin = bool(sweep_kwargs.pop("state_average_validate_spin", True))
    if nstates > 1 and qcdmrg.ncas > 2 and state_average_backend == "dense":
        return _run_dense_state_average_qchem(
            qcdmrg,
            nstates=nstates,
            target_sector=target_sector,
            weights=weights,
            max_dense_dim=sweep_kwargs.pop("state_average_dense_dim", 4096),
            spin_tol=state_average_spin_tol,
        )

    mps0 = _make_initial_mps(
        qcdmrg,
        target_sector=target_sector,
        initial_guess=initial_guess,
        bond_multiplicity=bond_multiplicity,
        seed=seed,
    )
    solver_kwargs = {
        "tol": 1.0e-10,
        "itermax": 100,
        "max_space": 160,
        "dense_fallback_dim": 512,
    }
    solver_kwargs.update(local_solver_kwargs or {})
    if nstates > 1:
        root_selection_buffer = int(sweep_kwargs.pop("state_average_root_buffer", 2))
        solver_kwargs["nstates"] = nstates
        solver_kwargs["weights"] = np.ones(nstates, dtype=float) / nstates
        solver_kwargs.setdefault("couple_physical", True)
        target_s = 0.5 * abs(float(qcdmrg.spin))
        solver_kwargs.setdefault("root_target_value", target_s * (target_s + 1.0))
        solver_kwargs.setdefault("root_target_tol", state_average_spin_tol)
        solver_kwargs.setdefault("root_selection_buffer", root_selection_buffer)
    root_target_mpo_factors = _spin_square_mpo_factors(qcdmrg) if nstates > 1 else None

    result = run_sweeps(
        mps0,
        nsweeps=int(nsweeps),
        mpo_factors=qcdmrg.H,
        root_target_mpo_factors=root_target_mpo_factors,
        max_bond=int(max_bond),
        max_bond_mode=sweep_kwargs.pop("max_bond_mode", "states"),
        canonical_local_norm=sweep_kwargs.pop("canonical_local_norm", False),
        warm_start_bonds=sweep_kwargs.pop("warm_start_bonds", True),
        mixer_zero_block_noise_scale=sweep_kwargs.pop("mixer_zero_block_noise_scale", 1.0e-5),
        mixer_zero_block_noise_seed=sweep_kwargs.pop("mixer_zero_block_noise_seed", seed + 4),
        mixer_nsweeps=sweep_kwargs.pop("mixer_nsweeps", 2),
        conv_tol=None if nstates > 1 else conv_tol,
        local_solver_kwargs=solver_kwargs,
        evaluate_root_energies_each_sweep=False if nstates > 1 else True,
        verbose=verbose,
        **sweep_kwargs,
    )
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
    ground_state = result["mps"]
    root_mps = result.get("root_mps")
    state_s2 = None
    if nstates > 1 and root_mps is not None:
        root_mps, state_energies, state_s2 = _finalize_spin_targeted_roots(
            qcdmrg,
            root_mps,
            nstates,
            compute_s2=state_average_validate_spin,
        )
        if result["history"]:
            result["history"][-1]["target_state_energies"] = list(state_energies)
            result["history"][-1]["state_energies"] = list(state_energies)
            if state_s2 is not None:
                result["history"][-1]["state_s2"] = list(state_s2)
                target_s = 0.5 * abs(float(qcdmrg.spin))
                target_s2 = target_s * (target_s + 1.0)
                result["history"][-1]["target_spin_valid"] = all(
                    abs(float(x) - target_s2) <= state_average_spin_tol
                    for x in state_s2
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
