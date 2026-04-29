#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small symmetry-aware model builders on top of AutoMPO and spatial-orbital operators.
"""

from __future__ import annotations

from .builder import AutoMPO
from .mpo import PhysicalLeg
from .operators import (
    physical_leg_from_spatial_orbital,
    reduced_spatial_fermion_annihilation,
    spatial_double_occupancy,
    spatial_number,
)
from .tensor import NonabelianTensor


def _normalize_site_legs(sites_or_legs):
    if isinstance(sites_or_legs, int):
        if sites_or_legs < 2:
            raise ValueError("Model builders require at least two sites.")
        return tuple(physical_leg_from_spatial_orbital() for _ in range(int(sites_or_legs)))

    site_legs = []
    for item in sites_or_legs:
        if isinstance(item, PhysicalLeg):
            site_legs.append(item)
        elif isinstance(item, NonabelianTensor):
            site_legs.append(physical_leg_from_spatial_orbital())
        else:
            raise TypeError(
                "Model builders expect a site count, PhysicalLegs, or rank-3 NonabelianTensor sites."
            )
    if len(site_legs) < 2:
        raise ValueError("Model builders require at least two sites.")
    return tuple(site_legs)


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
    canonical_leg = physical_leg_from_spatial_orbital()
    if any(phys_leg != canonical_leg for phys_leg in autompo.site_legs):
        raise ValueError(
            "add_spatial_density_terms currently expects spatial-orbital PhysicalLegs."
        )

    for site, phys_leg in enumerate(autompo.site_legs):
        if chemical_potential:
            autompo.add_onsite(site, spatial_number(dtype=float), coeff=-chemical_potential)
        if onsite_u:
            autompo.add_onsite(site, spatial_double_occupancy(dtype=float), coeff=onsite_u)

    if nearest_neighbor_v:
        density_ops = [spatial_number(dtype=float) for _ in autompo.site_legs]
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
):
    """
    Build a density-based spatial-orbital MPO directly from local model parameters.
    """
    site_legs = _normalize_site_legs(sites_or_legs)
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
    if any(phys_leg != canonical_leg for phys_leg in autompo.site_legs):
        raise ValueError(
            "add_spatial_hubbard_terms currently expects spatial-orbital PhysicalLegs."
        )

    add_spatial_density_terms(
        autompo,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
        nearest_neighbor_v=0.0,
    )

    if not hopping_t:
        return autompo

    reduced_fermion = reduced_spatial_fermion_annihilation(dtype=float)
    reduced_creation = reduced_fermion.adjoint()

    for site in range(autompo.nsites - 1):
        autompo.add_fermionic_reduced_bilinear(
            site,
            reduced_creation,
            site + 1,
            reduced_fermion,
            coeff=-hopping_t,
        )
        # Ordered-site form: c^\dagger_{j} c_{i} = - c_{i} c^\dagger_{j} for i < j.
        autompo.add_fermionic_reduced_bilinear(
            site,
            reduced_fermion,
            site + 1,
            reduced_creation,
            coeff=+hopping_t,
        )

    return autompo


def build_spatial_hubbard_mpo(
    sites_or_legs,
    *,
    hopping_t=1.0,
    chemical_potential=0.0,
    onsite_u=0.0,
):
    """
    Build an open-chain spatial-orbital Hubbard MPO with Jordan-Wigner strings.
    """
    site_legs = _normalize_site_legs(sites_or_legs)
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
):
    """
    Short alias for :func:`build_spatial_hubbard_mpo`.
    """
    return build_spatial_hubbard_mpo(
        sites_or_legs,
        hopping_t=hopping_t,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
    )
