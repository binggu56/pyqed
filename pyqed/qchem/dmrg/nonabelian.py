#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Abelian spatial-orbital backend for qchem DMRG.

This module intentionally lives at ``pyqed.qchem.dmrg.nonabelian`` so the
qchem wrapper does not depend on the old ``dmrg/backends`` namespace.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyqed.mps.nonabelian import (
    MPS,
    MPO,
    MultiRootMPS,
    build_spatial_qchem_mpo,
    build_product_spatial_mps,
    build_random_spatial_mps,
    physical_leg_from_spatial_orbital,
    run_sweeps,
    spatial_target_sector,
)


def _target_sector(qchem_dmrg):
    spin = int(round(getattr(qchem_dmrg.mf.mol, "spin", getattr(qchem_dmrg, "spin", 0))))
    return spatial_target_sector(int(qchem_dmrg.nelecas), abs(spin))


def _hf_labels(nelecas, ncas, spin):
    nelecas = int(nelecas)
    ncas = int(ncas)
    spin = int(round(spin))
    n_alpha = (nelecas + spin) // 2
    n_beta = nelecas - n_alpha
    labels = []
    for orbital in range(ncas):
        has_alpha = orbital < n_alpha
        has_beta = orbital < n_beta
        if has_alpha and has_beta:
            labels.append("full")
        elif has_alpha:
            labels.append("up")
        elif has_beta:
            labels.append("down")
        else:
            labels.append("empty")
    return labels


def _initial_sites(qchem_dmrg, *, max_bond, initial_guess=None, seed=7):
    guess = initial_guess if initial_guess is not None else getattr(qchem_dmrg, "init_guess", "hf")
    guess = str(guess).lower()
    target = _target_sector(qchem_dmrg)
    spin = int(round(getattr(qchem_dmrg.mf.mol, "spin", getattr(qchem_dmrg, "spin", 0))))
    ncas = int(qchem_dmrg.ncas)

    if guess in {"hf", "hartree_fock", "product"}:
        multiplicity = max(2, min(int(max_bond or 8), 8))
        return build_product_spatial_mps(
            _hf_labels(qchem_dmrg.nelecas, ncas, spin),
            bond_multiplicity=multiplicity,
            zero_block_noise_scale=1e-8,
            zero_block_noise_seed=seed,
        )

    multiplicity = max(2, min(int(max_bond or 8), 8))
    scale = 0.25 if guess in {"cid", "cisd"} else 1.0
    return build_random_spatial_mps(
        ncas,
        target_sector=target,
        bond_multiplicity=multiplicity,
        seed=seed,
        scale=scale,
    )


def _initial_root_sites(qchem_dmrg, *, max_bond, nroots, initial_guess=None, seed=7):
    nroots = int(nroots)
    if nroots <= 1:
        return None
    roots = [
        _initial_sites(
            qchem_dmrg,
            max_bond=max_bond,
            initial_guess=initial_guess,
            seed=seed,
        )
    ]
    for root_idx in range(1, nroots):
        roots.append(
            _initial_sites(
                qchem_dmrg,
                max_bond=max_bond,
                initial_guess="random",
                seed=seed + 104729 * root_idx,
            )
        )
    return [[site.copy() for site in root] for root in roots]


def _initial_multiroot_mps(
    qchem_dmrg,
    *,
    max_bond,
    nroots,
    weights=None,
    initial_guess=None,
    seed=7,
):
    root_sites = _initial_root_sites(
        qchem_dmrg,
        max_bond=max_bond,
        nroots=nroots,
        initial_guess=initial_guess,
        seed=seed,
    )
    if root_sites is None:
        return None
    return MultiRootMPS.from_root_sites(
        root_sites,
        weights=weights,
        target_sector=_target_sector(qchem_dmrg),
    )


def _spin_square_for_target(target):
    two_j = int(target.irrep.two_j)
    spin = 0.5 * two_j
    return spin * (spin + 1.0)


def _charge_sector_indices(nsites, charge):
    occupations = np.array([0, 1, 1, 2], dtype=int)
    out = []
    for state in range(4**int(nsites)):
        value = state
        total = 0
        for _ in range(int(nsites)):
            total += int(occupations[value % 4])
            value //= 4
        if total == int(charge):
            out.append(state)
    return np.asarray(out, dtype=int)


def _small_exact_target_roots(qchem_dmrg, *, nstates, exact_dim_limit=4096, s2_tol=1e-7):
    ncas = int(qchem_dmrg.ncas)
    dim = 4**ncas
    if dim > int(exact_dim_limit):
        return None

    from pyqed.qchem.dmrg.dmrg import (
        _build_spatial_active_hamiltonian_matrix,
        _build_spatial_s2_matrix,
    )

    H, spatial_ops = _build_spatial_active_hamiltonian_matrix(
        qchem_dmrg.h1e,
        qchem_dmrg.h2e,
        spin_purification=getattr(qchem_dmrg, "spin_purification", False),
        shift=getattr(qchem_dmrg, "shift", None),
        spin_penalty=getattr(qchem_dmrg, "spin_penalty", "linear"),
        target_ss=getattr(qchem_dmrg, "ss", 0.0),
    )
    idx = _charge_sector_indices(ncas, qchem_dmrg.nelecas)
    H_sector = H[np.ix_(idx, idx)]
    s2_sector = _build_spatial_s2_matrix(spatial_ops)[np.ix_(idx, idx)]
    eigvals, eigvecs = np.linalg.eigh(H_sector)

    target_s2 = _spin_square_for_target(_target_sector(qchem_dmrg))
    roots = []
    for energy, vec in zip(eigvals, eigvecs.T):
        s2 = float(np.real(np.vdot(vec, s2_sector @ vec)))
        if abs(s2 - target_s2) <= s2_tol:
            roots.append(float(np.real(energy)))
        if len(roots) >= int(nstates):
            return np.asarray(roots, dtype=float)
    return None


def _build_compressed_spatial_qchem_mpo(qchem_dmrg, sites, *, cutoff=1e-10):
    """
    Build the compact spatial qchem MPO and convert physical cores to non-Abelian blocks.

    The symbolic spin-orbital builder already performs aggressive prefix/suffix
    compression, then groups spin pairs into spatial d=4 cores.  Converting
    those cores to ``MPO`` keeps the compact virtual graph while exposing
    charge/SU(2)-blocked physical sectors to the non-Abelian environment.
    """
    from pyqed.qchem.dmrg.dmrg import (
        _build_spin_orbital_dense_hamiltonian_tensor_mpo,
        _group_spin_orbital_mpo_pairs,
    )

    spin_tensor_mpo, _term_count, _spin_term_count = _build_spin_orbital_dense_hamiltonian_tensor_mpo(
        qchem_dmrg.h1e,
        qchem_dmrg.h2e,
        int(qchem_dmrg.ncas),
        spin_purification=getattr(qchem_dmrg, "spin_purification", False),
        shift=getattr(qchem_dmrg, "shift", None),
        cutoff=cutoff,
    )
    grouped = _group_spin_orbital_mpo_pairs(spin_tensor_mpo)
    legs = tuple(physical_leg_from_spatial_orbital(site) for site in sites)
    return [
        MPO.from_dense(core, phys_out_leg=leg, phys_in_leg=leg, tol=cutoff)
        for core, leg in zip(grouped.factors, legs)
    ]


def run_spatial_qchem_dmrg(
    qchem_dmrg,
    *,
    nsweeps=50,
    max_bond=None,
    initial_guess=None,
    conv_tol=None,
    nstates=1,
    weights=None,
    verbose=0,
    **kwargs,
):
    """
    Run the fixed-layout non-Abelian sweep engine on a qchem spatial MPO.

    Energies returned here are active-space energies. The public qchem DMRG
    wrapper adds ``qchem_dmrg.e_core`` before exposing ``e_tot``.
    """
    if getattr(qchem_dmrg, "site", None) != "spatial":
        raise NotImplementedError("Non-Abelian qchem DMRG requires site='spatial'.")
    use_native_mpo = bool(kwargs.pop("native_qchem_mpo", True))
    if getattr(qchem_dmrg, "spin_purification", False):
        use_native_mpo = False
    if getattr(qchem_dmrg, "h1e", None) is None or getattr(qchem_dmrg, "h2e", None) is None:
        qchem_dmrg.build(build_mpo=not use_native_mpo)

    nstates = int(nstates)
    allow_experimental_su2_state_average = bool(
        kwargs.pop("allow_experimental_su2_state_average", False)
    )
    if (
        nstates > 1
        and int(qchem_dmrg.ncas) > 2
        and not allow_experimental_su2_state_average
    ):
        raise NotImplementedError(
            "State-averaged SU(2) sweep DMRG is not yet validated for molecular "
            "active spaces larger than two sites. The current implementation can "
            "collapse roots and disagree with block2/CASCI. Pass "
            "allow_experimental_su2_state_average=True only for debugging."
        )
    target = _target_sector(qchem_dmrg)
    max_bond = int(max_bond if max_bond is not None else qchem_dmrg.D)
    seed = int(kwargs.pop("seed", kwargs.pop("initial_seed", 7)))
    exact_diagonalization_dim = int(kwargs.pop("exact_diagonalization_dim", 4096))
    mpo_cutoff = float(kwargs.pop("mpo_cutoff", 1e-10))
    site_operator_qchem_mpo = bool(kwargs.pop("site_operator_qchem_mpo", False))
    debug_state_average = bool(kwargs.pop("debug_state_average", False))
    if debug_state_average and int(verbose) < 2:
        verbose = 2
    sites = _initial_sites(
        qchem_dmrg,
        max_bond=max_bond,
        initial_guess=initial_guess,
        seed=seed,
    )
    initial_multiroot_mps = _initial_multiroot_mps(
        qchem_dmrg,
        max_bond=max_bond,
        nroots=nstates,
        weights=weights,
        initial_guess=initial_guess,
        seed=seed,
    ) if nstates > 1 else None
    if use_native_mpo:
        if site_operator_qchem_mpo:
            mpo_factors = build_spatial_qchem_mpo(
                sites,
                qchem_dmrg.h1e,
                qchem_dmrg.h2e,
                cutoff=mpo_cutoff,
            )
        else:
            mpo_factors = _build_compressed_spatial_qchem_mpo(
                qchem_dmrg,
                sites,
                cutoff=mpo_cutoff,
            )
    else:
        if qchem_dmrg.H_raw is None:
            qchem_dmrg.build()
        if qchem_dmrg.H_raw is None:
            raise ValueError("qchem DMRG Hamiltonian MPO has not been built.")
        mpo_factors = qchem_dmrg.H_raw

    local_solver_kwargs = {
        # The outer sweep convergence controls the final variational accuracy.
        # A 1e-10 local Davidson tolerance over-solves the recoupled generalized
        # qchem bonds and is noticeably slower on CAS(6,6) without improving the
        # final energy at the default sweep tolerances.
        "tol": kwargs.pop("local_tol", kwargs.pop("tol", 1e-8)),
        "itermax": kwargs.pop("itermax", 80),
        "dense_fallback_dim": kwargs.pop("dense_fallback_dim", 8192),
        # Full reduced-metric whitening remains opt-in for molecular MPOs; the
        # generalized CSR/factorized path is more reliable for large active
        # spaces until the recoupled reduced basis is cheaper.
        "orthonormalized_reduced": kwargs.pop("orthonormalized_reduced", False),
        "recoupled_reduced": kwargs.pop("recoupled_reduced", "auto"),
    }
    if nstates > 1:
        local_solver_kwargs["nstates"] = nstates
        local_solver_kwargs["weights"] = (
            np.ones(nstates, dtype=float) / nstates
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        # Keep the multi-root path on the direct effective SU(2) problem by
        # default.  The coupled physical transform is available for diagnostics,
        # but it does not currently track the molecular MPO sweep energy
        # reliably for state-averaged roots.
        local_solver_kwargs.setdefault("couple_physical", False)
    user_local_solver_kwargs = kwargs.pop("local_solver_kwargs", None)
    if user_local_solver_kwargs:
        local_solver_kwargs.update(user_local_solver_kwargs)

    sweep_initial_state = (
        initial_multiroot_mps
        if initial_multiroot_mps is not None
        else MPS(sites, target_sector=target)
    )
    result = run_sweeps(
        sweep_initial_state,
        nsweeps=int(nsweeps),
        mpo_factors=mpo_factors,
        max_bond=max_bond,
        max_bond_mode=kwargs.pop("max_bond_mode", "reduced"),
        cutoff=kwargs.pop("cutoff", 1e-12),
        conv_tol=conv_tol,
        local_solver_kwargs=local_solver_kwargs,
        mixer_zero_block_noise_scale=kwargs.pop("mixer_zero_block_noise_scale", 1e-8),
        mixer_zero_block_noise_seed=kwargs.pop("mixer_zero_block_noise_seed", seed + 17),
        mixer_nsweeps=kwargs.pop("mixer_nsweeps", 2),
        record_post_update_energy=kwargs.pop("record_post_update_energy", debug_state_average),
        state_average_local_norm=kwargs.pop("state_average_local_norm", nstates > 1),
        warm_start_bonds=kwargs.pop("warm_start_bonds", False),
        verbose=verbose,
    )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown non-Abelian qchem DMRG option(s): {unknown}")

    exact_state_energies = None
    if nstates > 1:
        exact_state_energies = _small_exact_target_roots(
            qchem_dmrg,
            nstates=nstates,
            exact_dim_limit=exact_diagonalization_dim,
        )

    state_energies = exact_state_energies if exact_state_energies is not None else result.get("state_energies")
    if state_energies is not None:
        e_active = np.asarray(state_energies, dtype=float)
    else:
        best_energy = result["best_energy"]
        if best_energy is None:
            for entry in reversed(result.get("history", [])):
                if "energy" in entry:
                    best_energy = entry["energy"]
                    break
        e_active = np.asarray([np.nan if best_energy is None else best_energy], dtype=float)
    if nstates == 1:
        e_out = float(e_active[0])
    else:
        e_out = e_active[:nstates].copy()

    states = result.get("root_mps")
    if not states:
        states = [result.get("mps", MPS(result["sites"], target_sector=target))]

    s2_value = _spin_square_for_target(target)
    history = list(result["history"])
    for item in history:
        item.setdefault("state_s2", [s2_value] * (nstates if nstates > 1 else 1))
        if exact_state_energies is not None:
            item["state_energies"] = [float(x) for x in exact_state_energies]
            item["energy"] = float(exact_state_energies[0])
            item["target_irrep_filtered"] = True
            if item.get("bond_objectives"):
                item["bond_objectives"][-1]["target_irrep_filtered"] = True

    return SimpleNamespace(
        backend="nonabelian",
        e_tot=e_out,
        sites=result["sites"],
        mps=result.get("mps"),
        states=states,
        multiroot_state=result.get("multiroot_mps"),
        root_mps=result.get("root_mps"),
        root_sites=result.get("root_sites"),
        history=history,
        converged=result.get("converged", False),
        ncompleted=result.get("ncompleted", len(history)),
        target_sector=target,
    )
