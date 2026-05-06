#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small symmetry-aware model builders on top of AutoMPO and spatial-orbital operators.
"""

from __future__ import annotations

import numpy as np

from .builder import AutoMPO
from .mpo import PhysicalLeg
from .operators import (
    physical_leg_from_spatial_orbital,
    reduced_physical_leg_from_spatial_orbital,
    reduced_spatial_fermion_annihilation,
    reduced_spatial_double_occupancy,
    reduced_spatial_number,
    reduced_spatial_parity,
    spatial_annihilate_down,
    spatial_annihilate_up,
    spatial_create_down,
    spatial_create_up,
    spatial_double_occupancy,
    spatial_number,
    spatial_parity,
)
from .tensor import NonabelianTensor


def _normalize_site_legs(sites_or_legs, *, reduced_physical=False):
    if isinstance(sites_or_legs, int):
        if sites_or_legs < 2:
            raise ValueError("Model builders require at least two sites.")
        leg_factory = (
            reduced_physical_leg_from_spatial_orbital
            if reduced_physical
            else physical_leg_from_spatial_orbital
        )
        return tuple(leg_factory() for _ in range(int(sites_or_legs)))

    site_legs = []
    for item in sites_or_legs:
        if isinstance(item, PhysicalLeg):
            site_legs.append(item)
        elif isinstance(item, NonabelianTensor):
            if item.metadata.get("physical_basis") == "reduced_spatial":
                site_legs.append(reduced_physical_leg_from_spatial_orbital(item))
            else:
                site_legs.append(physical_leg_from_spatial_orbital(item))
        else:
            raise TypeError(
                "Model builders expect a site count, PhysicalLegs, or rank-3 NonabelianTensor sites."
            )
    if len(site_legs) < 2:
        raise ValueError("Model builders require at least two sites.")
    return tuple(site_legs)


def _spatial_scalar_operator_factories(site_legs, *, label):
    canonical_leg = physical_leg_from_spatial_orbital()
    reduced_leg = reduced_physical_leg_from_spatial_orbital()
    if all(phys_leg == canonical_leg for phys_leg in site_legs):
        return spatial_number, spatial_double_occupancy
    if all(phys_leg == reduced_leg for phys_leg in site_legs):
        return reduced_spatial_number, reduced_spatial_double_occupancy
    raise ValueError(
        f"{label} expects all sites to use either explicit spatial PhysicalLegs "
        "or degeneracy-only reduced spatial PhysicalLegs."
    )


def add_spatial_density_terms(
    autompo,
    *,
    chemical_potential=0.0,
    onsite_u=0.0,
    nearest_neighbor_v=0.0,
):
    """
    Add density-based spatial-orbital Hamiltonian terms to an AutoMPO.

    The resulting model is

        H = sum_i (-mu * n_i + U * n_{i,up} n_{i,down})
            + sum_i V * n_i n_{i+1}

    represented in the SU(2)-adapted spatial-orbital basis.
    """
    if not isinstance(autompo, AutoMPO):
        raise TypeError("add_spatial_density_terms expects an AutoMPO.")
    number_op, double_op = _spatial_scalar_operator_factories(
        autompo.site_legs,
        label="add_spatial_density_terms",
    )

    for site, phys_leg in enumerate(autompo.site_legs):
        if chemical_potential:
            autompo.add_onsite(site, number_op(dtype=float), coeff=-chemical_potential)
        if onsite_u:
            autompo.add_onsite(site, double_op(dtype=float), coeff=onsite_u)

    if nearest_neighbor_v:
        density_ops = [number_op(dtype=float) for _ in autompo.site_legs]
        for site in range(autompo.nsites - 1):
            autompo.add_nearest_neighbor(
                site,
                density_ops[site],
                density_ops[site + 1],
                coeff=nearest_neighbor_v,
            )
    return autompo


def build_spatial_density_mpo(
    sites_or_legs,
    *,
    chemical_potential=0.0,
    onsite_u=0.0,
    nearest_neighbor_v=0.0,
    reduced_physical=False,
):
    """
    Build a density-based spatial-orbital MPO directly from local model parameters.
    """
    site_legs = _normalize_site_legs(sites_or_legs, reduced_physical=reduced_physical)
    autompo = AutoMPO(site_legs)
    add_spatial_density_terms(
        autompo,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
        nearest_neighbor_v=nearest_neighbor_v,
    )
    return autompo.build()


def add_spatial_hubbard_terms(
    autompo,
    *,
    hopping_t=1.0,
    chemical_potential=0.0,
    onsite_u=0.0,
):
    """
    Add an open-chain real Hubbard Hamiltonian in the spatial-orbital basis.

    The model is

        H = -t sum_{<ij>,sigma} (c^dagger_{i,sigma} c_{j,sigma} + h.c.)
            - mu sum_i n_i
            + U sum_i n_{i,up} n_{i,down}

    with Jordan-Wigner strings handled internally by ``AutoMPO``.
    """
    if not isinstance(autompo, AutoMPO):
        raise TypeError("add_spatial_hubbard_terms expects an AutoMPO.")
    canonical_leg = physical_leg_from_spatial_orbital()
    reduced_leg = reduced_physical_leg_from_spatial_orbital()
    reduced_sites = all(phys_leg == reduced_leg for phys_leg in autompo.site_legs)
    if not reduced_sites and any(phys_leg != canonical_leg for phys_leg in autompo.site_legs):
        raise ValueError(
            "add_spatial_hubbard_terms expects explicit or reduced spatial-orbital PhysicalLegs."
        )

    add_spatial_density_terms(
        autompo,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
        nearest_neighbor_v=0.0,
    )

    if not hopping_t:
        return autompo

    representative_leg = autompo.site_legs[0]
    reduced_fermion = reduced_spatial_fermion_annihilation(representative_leg, dtype=float)
    reduced_creation = reduced_fermion.adjoint()
    parity = (
        reduced_spatial_parity(representative_leg, dtype=float)
        if reduced_sites
        else spatial_parity(dtype=float)
    )

    for site in range(autompo.nsites - 1):
        autompo.add_fermionic_reduced_bilinear(
            site,
            reduced_creation,
            site + 1,
            reduced_fermion,
            coeff=-hopping_t,
            parity_operator=parity,
        )
        # Ordered-site form: c^\dagger_{j} c_{i} = - c_{i} c^\dagger_{j} for i < j.
        autompo.add_fermionic_reduced_bilinear(
            site,
            reduced_fermion,
            site + 1,
            reduced_creation,
            coeff=+hopping_t,
            parity_operator=parity,
        )

    return autompo


def build_spatial_hubbard_mpo(
    sites_or_legs,
    *,
    hopping_t=1.0,
    chemical_potential=0.0,
    onsite_u=0.0,
    reduced_physical=False,
):
    """
    Build an open-chain spatial-orbital Hubbard MPO with Jordan-Wigner strings.
    """
    site_legs = _normalize_site_legs(sites_or_legs, reduced_physical=reduced_physical)
    autompo = AutoMPO(site_legs)
    add_spatial_hubbard_terms(
        autompo,
        hopping_t=hopping_t,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
    )
    return autompo.build()


def build_hubbard_mpo(
    sites_or_legs,
    *,
    hopping_t=1.0,
    chemical_potential=0.0,
    onsite_u=0.0,
    reduced_physical=False,
):
    """
    Short alias for :func:`build_spatial_hubbard_mpo`.
    """
    return build_spatial_hubbard_mpo(
        sites_or_legs,
        hopping_t=hopping_t,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
        reduced_physical=reduced_physical,
    )


def _spatial_one_body_integrals(h1e):
    h1e = np.asarray(h1e)
    if h1e.ndim == 2:
        return h1e
    if h1e.ndim >= 3:
        return np.asarray(h1e[0])
    raise ValueError("Spatial qchem MPO expects h1e with shape (n,n) or spin blocks.")


def _spatial_two_body_integrals(eri):
    eri = np.asarray(eri)
    if eri.ndim == 4:
        return 0.5 * eri
    if eri.ndim >= 6:
        return 0.5 * np.asarray(eri[0, 0])
    raise ValueError("Spatial qchem MPO expects eri with shape (n,n,n,n) or spin blocks.")


def add_spatial_qchem_terms(
    autompo,
    h1e,
    eri,
    *,
    cutoff=1e-10,
):
    """
    Add a spatial-orbital quantum-chemistry Hamiltonian to an AutoMPO.

    The added active-space operator is

        sum_pq h_pq (c^dagger_pu c_qu + c^dagger_pd c_qd)
        + 1/2 sum_pqrs (pq|rs) c^dagger_p c^dagger_r c_s c_q

    with explicit Jordan-Wigner strings generated from site-local
    spatial-orbital ``SiteOperator`` pieces.
    """
    if not isinstance(autompo, AutoMPO):
        raise TypeError("add_spatial_qchem_terms expects an AutoMPO.")
    canonical_leg = physical_leg_from_spatial_orbital()
    if any(phys_leg != canonical_leg for phys_leg in autompo.site_legs):
        raise ValueError(
            "add_spatial_qchem_terms currently expects spatial-orbital PhysicalLegs."
        )

    h_spatial = _spatial_one_body_integrals(h1e)
    eri_spatial = _spatial_two_body_integrals(eri)
    ncas = h_spatial.shape[0]
    if h_spatial.shape != (ncas, ncas):
        raise ValueError(f"h1e must be square, got shape {h_spatial.shape!r}.")
    if eri_spatial.shape != (ncas, ncas, ncas, ncas):
        raise ValueError(
            f"eri must have spatial shape {(ncas, ncas, ncas, ncas)!r}, "
            f"got {eri_spatial.shape!r}."
        )
    if autompo.nsites != ncas:
        raise ValueError(
            f"AutoMPO has {autompo.nsites} sites but integrals describe {ncas} orbitals."
        )

    dtype = np.result_type(h_spatial.dtype, eri_spatial.dtype, float)
    parity = spatial_parity(dtype=dtype)
    c_up = spatial_annihilate_up(dtype=dtype)
    cd_up = spatial_create_up(dtype=dtype)
    c_down = spatial_annihilate_down(dtype=dtype)
    cd_down = spatial_create_down(dtype=dtype)

    for p, q in np.argwhere(np.abs(h_spatial) > cutoff):
        val = h_spatial[p, q]
        autompo.add_fermionic_string(
            (p, cd_up),
            (q, c_up),
            coeff=val,
            parity_operator=parity,
            tol=cutoff,
        )
        autompo.add_fermionic_string(
            (p, cd_down),
            (q, c_down),
            coeff=val,
            parity_operator=parity,
            tol=cutoff,
        )

    for p, q, r, s in np.argwhere(np.abs(eri_spatial) > cutoff):
        val = eri_spatial[p, q, r, s]
        if p != r and s != q:
            autompo.add_fermionic_string(
                (p, cd_up),
                (r, cd_up),
                (s, c_up),
                (q, c_up),
                coeff=val,
                parity_operator=parity,
                tol=cutoff,
            )
            autompo.add_fermionic_string(
                (p, cd_down),
                (r, cd_down),
                (s, c_down),
                (q, c_down),
                coeff=val,
                parity_operator=parity,
                tol=cutoff,
            )
        autompo.add_fermionic_string(
            (p, cd_up),
            (r, cd_down),
            (s, c_down),
            (q, c_up),
            coeff=val,
            parity_operator=parity,
            tol=cutoff,
        )
        autompo.add_fermionic_string(
            (p, cd_down),
            (r, cd_up),
            (s, c_up),
            (q, c_down),
            coeff=val,
            parity_operator=parity,
            tol=cutoff,
        )

    return autompo


def build_spatial_qchem_mpo(
    sites_or_legs,
    h1e,
    eri,
    *,
    cutoff=1e-10,
):
    """
    Build a non-Abelian spatial-orbital qchem MPO from one- and two-electron integrals.
    """
    site_legs = _normalize_site_legs(sites_or_legs)
    autompo = AutoMPO(site_legs)
    add_spatial_qchem_terms(
        autompo,
        h1e,
        eri,
        cutoff=cutoff,
    )
    return autompo.build()
