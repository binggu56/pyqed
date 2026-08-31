#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small symmetry-aware model builders on top of AutoMPO and spatial-orbital operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .builder import AutoMPO
from .mpo import Leg, as_rank_coupled_mpo
from .operators import (
    ReducedTensorOperator,
    SiteOperator,
    compose_site_operators,
    coupled_reduced_tensor_product,
    physical_leg_from_spatial_orbital,
    reduced_spatial_fermion_annihilation,
    spatial_double_occupancy,
    spatial_pair_annihilation,
    spatial_pair_creation,
    spatial_create_down,
    spatial_create_up,
    spatial_annihilate_down,
    spatial_annihilate_up,
    spatial_number,
    spatial_parity,
    spatial_projector,
    time_reversed_reduced_operator,
)
from pyqed.mps.su2 import SU2Irrep
from .coupling import left_or_right_fusion
from pyqed.symmetry import IrrepTensor


_SQRT3 = float(np.sqrt(3.0))
_FULLY_REDUCED_PAIR_RECOUPLING = 2.0
_FULLY_REDUCED_EXCHANGE_RECOUPLING = 1.0
_FULLY_REDUCED_ONE_BODY_SPLIT_FAMILY = "__fully_reduced_one_body_split__"


_ONE_BODY_MIDDLE_CHANNEL_COEFFS = {
    ("forward", 1, 0): {
        ("c01/a01", ("D_single",)): 0.7071067811865479,
        ("c01/a01", ("D_double",)): -0.7071067811865479,
        ("c01/a01", ("R_single",)): -0.707106781186548,
        ("c01/a01", ("R_double",)): -0.7071067811865478,
        ("c01/a12", ("D_empty",)): -0.7071067811865479,
        ("c01/a12", ("D_single",)): 0.25881904510252074,
        ("c01/a12", ("D_double",)): -0.7071067811865472,
        ("c01/a12", ("R_empty",)): -0.7071067811865479,
        ("c01/a12", ("R_single",)): 0.9659258262890679,
        ("c01/a12", ("R_double",)): -0.7071067811865474,
        ("c12/a01", ("D_empty",)): -0.7071067811865479,
        ("c12/a01", ("D_single",)): 1.4142135623731056,
        ("c12/a01", ("D_double",)): -0.707106781186548,
        ("c12/a01", ("R_empty",)): -0.7071067811865477,
        ("c12/a01", ("R_double",)): -0.7071067811865477,
        ("c12/a12", ("D_empty",)): 0.3535533905932738,
        ("c12/a12", ("D_single",)): 0.2828427124746191,
        ("c12/a12", ("R_empty",)): 0.3535533905932738,
        ("c12/a12", ("R_single",)): 0.5656854249492382,
    },
    ("forward", 1, 1): {
        ("c01/a01", ("D_single",)): 0.707106781186548,
        ("c01/a01", ("D_double",)): -0.7071067811865478,
        ("c01/a01", ("R_single",)): -0.7071067811865479,
        ("c01/a01", ("R_double",)): -0.7071067811865479,
        ("c01/a12", ("D_empty",)): -0.7071067811865476,
        ("c01/a12", ("D_single",)): 1.414213562373114,
        ("c01/a12", ("D_double",)): -0.707106781186548,
        ("c01/a12", ("R_empty",)): -0.7071067811865477,
        ("c01/a12", ("R_double",)): -0.7071067811865474,
        ("c12/a01", ("D_empty",)): -0.7071067811865476,
        ("c12/a01", ("D_single",)): 0.25881904510252074,
        ("c12/a01", ("D_double",)): -0.7071067811865475,
        ("c12/a01", ("R_empty",)): -0.7071067811865476,
        ("c12/a01", ("R_single",)): 0.9659258262890686,
        ("c12/a01", ("R_double",)): -0.7071067811865477,
        ("c12/a12", ("D_empty",)): 0.35355339059327384,
        ("c12/a12", ("D_single",)): 0.2828427124746191,
        ("c12/a12", ("R_empty",)): 0.35355339059327384,
        ("c12/a12", ("R_single",)): 0.5656854249492382,
    },
    ("backward", 1, None): {
        ("a01/c01", ("D_single",)): 0.5656854249492385,
        ("a01/c01", ("D_double",)): 0.7071067811865478,
        ("a01/c01", ("R_single",)): 1.131370849898477,
        ("a01/c01", ("R_double",)): 0.7071067811865478,
        ("a01/c12", ("D_empty",)): 0.7071067811865478,
        ("a01/c12", ("D_single",)): -1.4142135623731065,
        ("a01/c12", ("D_double",)): 0.707106781186548,
        ("a01/c12", ("R_empty",)): 0.7071067811865478,
        ("a01/c12", ("R_double",)): 0.7071067811865477,
        ("a12/c01", ("D_empty",)): 0.7071067811865478,
        ("a12/c01", ("D_single",)): -0.25881904510252074,
        ("a12/c01", ("D_double",)): 0.7071067811865472,
        ("a12/c01", ("R_empty",)): 0.7071067811865478,
        ("a12/c01", ("R_single",)): -0.9659258262890679,
        ("a12/c01", ("R_double",)): 0.7071067811865474,
        ("a12/c12", ("D_empty",)): -0.3535533905932738,
        ("a12/c12", ("D_single",)): 0.3535533905932737,
        ("a12/c12", ("R_empty",)): -0.35355339059327373,
        ("a12/c12", ("R_single",)): -0.35355339059327373,
    },
    ("backward", 2, None): {
        ("a01/c01", ("D_single", "D_double")): 0.2828427124746191,
        ("a01/c01", ("D_single", "R_double")): 0.2828427124746191,
        ("a01/c01", ("R_single", "D_double")): 0.5656854249492382,
        ("a01/c01", ("R_single", "R_double")): 0.5656854249492382,
        ("a01/c01", ("D_double", "D_single")): 0.2828427124746191,
        ("a01/c01", ("D_double", "R_single")): 0.5656854249492382,
        ("a01/c01", ("R_double", "D_single")): 0.28284271247461923,
        ("a01/c01", ("R_double", "R_single")): 0.5656854249492382,
        ("a01/c12", ("D_empty", "D_double")): 0.35355339059327384,
        ("a01/c12", ("D_empty", "R_double")): 0.35355339059327384,
        ("a01/c12", ("R_empty", "D_double")): 0.35355339059327384,
        ("a01/c12", ("R_empty", "R_double")): 0.35355339059327384,
        ("a01/c12", ("D_single", "D_single")): 0.44446711960297297,
        ("a01/c12", ("D_single", "R_single")): 0.1616244071283538,
        ("a01/c12", ("R_single", "D_single")): 0.6327698185865018,
        ("a01/c12", ("R_single", "R_single")): 0.06708439363726355,
        ("a01/c12", ("D_double", "D_empty")): 0.3535533905932739,
        ("a01/c12", ("D_double", "R_empty")): 0.35355339059327384,
        ("a01/c12", ("R_double", "D_empty")): 0.35355339059327384,
        ("a01/c12", ("R_double", "R_empty")): 0.35355339059327384,
        ("a12/c01", ("D_empty", "D_double")): 0.3535533905932739,
        ("a12/c01", ("D_empty", "R_double")): 0.35355339059327384,
        ("a12/c01", ("R_empty", "D_double")): 0.35355339059327384,
        ("a12/c01", ("R_empty", "R_double")): 0.35355339059327384,
        ("a12/c01", ("D_single", "D_single")): 0.4444671196029728,
        ("a12/c01", ("D_single", "R_single")): 0.6327698185865015,
        ("a12/c01", ("R_single", "D_single")): 0.16162440712835363,
        ("a12/c01", ("R_single", "R_single")): 0.06708439363726344,
        ("a12/c01", ("D_double", "D_empty")): 0.3535533905932739,
        ("a12/c01", ("D_double", "R_empty")): 0.35355339059327384,
        ("a12/c01", ("R_double", "D_empty")): 0.35355339059327384,
        ("a12/c01", ("R_double", "R_empty")): 0.35355339059327384,
        ("a12/c12", ("D_empty", "D_single")): 0.17677669529663695,
        ("a12/c12", ("D_empty", "R_single")): -0.17677669529663698,
        ("a12/c12", ("R_empty", "D_single")): 0.17677669529663698,
        ("a12/c12", ("R_empty", "R_single")): -0.17677669529663698,
        ("a12/c12", ("D_single", "D_empty")): 0.17677669529663695,
        ("a12/c12", ("D_single", "R_empty")): 0.17677669529663698,
        ("a12/c12", ("R_single", "D_empty")): -0.17677669529663698,
        ("a12/c12", ("R_single", "R_empty")): -0.17677669529663698,
    },
}


def _su2_ranks(*two_j):
    """
    Return an immutable tuple of SU(2) irreps from doubled-spin labels.

    :param two_j: Doubled-spin labels for the reduced tensor ranks.
    :returns: Tuple of :class:`SU2Irrep` objects.
    """

    return tuple(SU2Irrep(int(rank)) for rank in two_j)


_SPINFREE_WE_RECOUPLING_CACHE = {
    # Recoupling coefficients for all permutations of
    # (c^dag_p, c_q, c^dag_r, c_s). They are molecule-independent and avoid
    # rebuilding dense four-site AutoMPO probes during every first SU(2) MPO build.
    (0, 1, 2, 3): (2.0, 0.0),
    (0, 1, 3, 2): (2.0, 0.0),
    (0, 2, 1, 3): (-1.0, -_SQRT3),
    (0, 2, 3, 1): (-1.0, _SQRT3),
    (0, 3, 1, 2): (-1.0, -_SQRT3),
    (0, 3, 2, 1): (-1.0, _SQRT3),
    (1, 0, 2, 3): (2.0, 0.0),
    (1, 0, 3, 2): (2.0, 0.0),
    (1, 2, 0, 3): (-1.0, -_SQRT3),
    (1, 2, 3, 0): (-1.0, _SQRT3),
    (1, 3, 0, 2): (-1.0, -_SQRT3),
    (1, 3, 2, 0): (-1.0, _SQRT3),
    (2, 0, 1, 3): (-1.0, _SQRT3),
    (2, 0, 3, 1): (-1.0, -_SQRT3),
    (2, 1, 0, 3): (-1.0, _SQRT3),
    (2, 1, 3, 0): (-1.0, -_SQRT3),
    (2, 3, 0, 1): (2.0, 0.0),
    (2, 3, 1, 0): (2.0, 0.0),
    (3, 0, 1, 2): (-1.0, _SQRT3),
    (3, 0, 2, 1): (-1.0, -_SQRT3),
    (3, 1, 0, 2): (-1.0, _SQRT3),
    (3, 1, 2, 0): (-1.0, -_SQRT3),
    (3, 2, 0, 1): (2.0, 0.0),
    (3, 2, 1, 0): (2.0, 0.0),
}
_SPINFREE_EXCHANGE_RECOUPLING_CACHE = {
    # Repeated-site exchange recoupling patterns for
    # (c^dag_p, c_q, c^dag_r, c_s), keyed by compressed site positions.
    # These are symmetry constants; the dense verifier remains as a fallback
    # for unsupported patterns but normal fully reduced MPO builds should use
    # this table directly.
    (0, 1, 1, 0): ((-1.0, _su2_ranks(0, 0)), (-_SQRT3, _su2_ranks(2, 2))),
    (0, 1, 1, 2): ((-1.0, _su2_ranks(1, 0, 1)), (_SQRT3, _su2_ranks(1, 2, 1))),
    (0, 1, 2, 0): ((-1.0, _su2_ranks(0, 1, 1)), (-_SQRT3, _su2_ranks(2, 1, 1))),
    (0, 2, 1, 0): ((-1.0, _su2_ranks(0, 1, 1)), (_SQRT3, _su2_ranks(2, 1, 1))),
    (0, 2, 2, 1): ((-1.0, _su2_ranks(1, 1, 0)), (-_SQRT3, _su2_ranks(1, 1, 2))),
    (1, 0, 0, 1): ((-1.0, _su2_ranks(0, 0)), (-_SQRT3, _su2_ranks(2, 2))),
    (1, 0, 0, 2): ((-1.0, _su2_ranks(0, 1, 1)), (-_SQRT3, _su2_ranks(2, 1, 1))),
    (1, 0, 2, 1): ((-1.0, _su2_ranks(1, 0, 1)), (_SQRT3, _su2_ranks(1, 2, 1))),
    (1, 2, 0, 1): ((-1.0, _su2_ranks(1, 0, 1)), (-_SQRT3, _su2_ranks(1, 2, 1))),
    (1, 2, 2, 0): ((-1.0, _su2_ranks(1, 1, 0)), (_SQRT3, _su2_ranks(1, 1, 2))),
    (2, 0, 0, 1): ((-1.0, _su2_ranks(0, 1, 1)), (_SQRT3, _su2_ranks(2, 1, 1))),
    (2, 0, 1, 2): ((-1.0, _su2_ranks(1, 1, 0)), (-_SQRT3, _su2_ranks(1, 1, 2))),
    (2, 1, 0, 2): ((-1.0, _su2_ranks(1, 1, 0)), (_SQRT3, _su2_ranks(1, 1, 2))),
    (2, 1, 1, 0): ((-1.0, _su2_ranks(1, 0, 1)), (-_SQRT3, _su2_ranks(1, 2, 1))),
}


def _normalize_site_legs(sites_or_legs, *, min_sites=2):
    if isinstance(sites_or_legs, int):
        if sites_or_legs < min_sites:
            raise ValueError(f"Model builders require at least {min_sites} site(s).")
        return tuple(physical_leg_from_spatial_orbital() for _ in range(int(sites_or_legs)))

    site_legs = []
    for item in sites_or_legs:
        if isinstance(item, Leg):
            site_legs.append(item)
        elif isinstance(item, IrrepTensor):
            site_legs.append(physical_leg_from_spatial_orbital(item))
        elif item.__class__.__name__ == "FullyReducedSpatialOrbitalSite":
            site_legs.append(physical_leg_from_spatial_orbital(item))
        else:
            raise TypeError(
                "Model builders expect a site count, PhysicalLegs, or rank-3 IrrepTensor sites."
            )
    if len(site_legs) < min_sites:
        raise ValueError(f"Model builders require at least {min_sites} site(s).")
    return tuple(site_legs)


def _is_fully_reduced_spatial_leg(phys_leg):
    return physical_leg_from_spatial_orbital(phys_leg) != physical_leg_from_spatial_orbital()


def _split_spatial_fermion_annihilation_channels(phys_leg, *, dtype):
    annihilation = reduced_spatial_fermion_annihilation(phys_leg, dtype=dtype)
    q_empty, q_single, q_double = phys_leg.sectors
    return (
        ReducedTensorOperator(
            reduced_blocks={
                (q_empty, q_single): annihilation.reduced_blocks[(q_empty, q_single)],
            },
            phys_out_leg=phys_leg,
            phys_in_leg=phys_leg,
            rank_irrep=SU2Irrep(1),
            component_phases=annihilation.component_phases,
        ),
        ReducedTensorOperator(
            reduced_blocks={
                (q_single, q_double): annihilation.reduced_blocks[(q_single, q_double)],
            },
            phys_out_leg=phys_leg,
            phys_in_leg=phys_leg,
            rank_irrep=SU2Irrep(1),
            component_phases=annihilation.component_phases,
        ),
    )


def _fully_reduced_one_body_middle_channel_ops(phys_leg, *, dtype):
    q_empty, q_single, q_double = phys_leg.sectors
    return {
        "D_empty": spatial_projector("empty", phys_leg, dtype=dtype),
        "D_single": spatial_projector("single", phys_leg, dtype=dtype),
        "D_double": spatial_projector("double", phys_leg, dtype=dtype),
        "R_empty": ReducedTensorOperator(
            {(q_empty, q_empty): np.asarray(1.0, dtype=dtype)},
            phys_leg,
            phys_leg,
            SU2Irrep(0),
        ),
        "R_single": ReducedTensorOperator(
            {(q_single, q_single): np.asarray(1.0, dtype=dtype)},
            phys_leg,
            phys_leg,
            SU2Irrep(0),
        ),
        "R_double": ReducedTensorOperator(
            {(q_double, q_double): np.asarray(1.0, dtype=dtype)},
            phys_leg,
            phys_leg,
            SU2Irrep(0),
        ),
    }


def _add_fully_reduced_one_body_middle_channel(
    autompo,
    endpoint_ops,
    middle_sites,
    middle_labels,
    term_coeff,
    *,
    phys_leg,
    dtype,
    family,
):
    middle_ops = _fully_reduced_one_body_middle_channel_ops(phys_leg, dtype=dtype)
    dense_middle = {}
    reduced_insertions = [endpoint_ops[0]]
    intermediate_irreps = [SU2Irrep(1)]
    for site, label in zip(middle_sites, middle_labels):
        operator = middle_ops[label]
        if label.startswith("D_"):
            dense_middle[int(site)] = operator
        else:
            reduced_insertions.append((int(site), operator))
            intermediate_irreps.append(SU2Irrep(1))
    reduced_insertions.append(endpoint_ops[1])
    if len(reduced_insertions) == 2:
        autompo.add_reduced_string_product(
            *reduced_insertions,
            intermediate_irreps=(SU2Irrep(1),),
            middle_operators=dense_middle,
            coeff=term_coeff,
            family=family,
        )
    else:
        autompo.add_reduced_string(
            *reduced_insertions,
            intermediate_irreps=tuple(intermediate_irreps),
            middle_operators=dense_middle,
            coeff=term_coeff,
            family=family,
        )


def _with_prefix_family(family, prefix):
    visible = prefix.replace("__prefix_", "__channel_", 1)
    if family is None:
        return (visible, prefix)
    if isinstance(family, str):
        return (family, visible, prefix)
    return tuple(family) + (visible, prefix)


def _add_fully_reduced_spinfree_bilinear(
    autompo,
    create_site,
    annihilate_site,
    coeff,
    *,
    phys_leg,
    dtype,
    density_site=None,
    density_operator=None,
    family=None,
    split_channels=True,
):
    """
    Add ``coeff * E_pq`` with ``E_pq = sum_sigma c^dagger[p,sigma] c[q,sigma]``.

    Fully reduced sites must form the spin scalar through an explicit
    Wigner-Eckart coupling, rather than the componentwise shortcut used by the
    canonical local basis.
    """

    create_site = int(create_site)
    annihilate_site = int(annihilate_site)
    if create_site == annihilate_site:
        operator = spatial_number(phys_leg, dtype=dtype)
        if density_site is not None:
            density = (
                density_operator
                if density_operator is not None
                else spatial_number(phys_leg, dtype=dtype)
            )
            operator = compose_site_operators(density, operator)
        autompo.add_onsite(create_site, operator, coeff=coeff, family=family)
        return autompo

    annihilation = reduced_spatial_fermion_annihilation(phys_leg, dtype=dtype)
    creation = annihilation.adjoint()
    parity = spatial_parity(phys_leg, dtype=dtype)
    double_phase = _fully_reduced_double_transition_phase(phys_leg, dtype=dtype)
    density = (
        density_operator
        if density_operator is not None
        else spatial_number(phys_leg, dtype=dtype) if density_site is not None else None
    )

    def with_density(operator, site):
        if density is not None and int(density_site) == int(site):
            return operator.left_multiply_sector_scalar(density)
        return operator

    dense_scalars = ()
    if density is not None and int(density_site) not in {create_site, annihilate_site}:
        dense_scalars = ((int(density_site), density),)
    if not split_channels:
        if create_site < annihilate_site:
            autompo.add_reduced_string_product(
                (
                    create_site,
                    with_density(
                        creation.left_multiply_sector_scalar(double_phase),
                        create_site,
                    ).right_multiply_sector_scalar(parity),
                ),
                (
                    annihilate_site,
                    time_reversed_reduced_operator(
                        with_density(annihilation, annihilate_site)
                    ),
                ),
                intermediate_irreps=(SU2Irrep(1),),
                dense_site_operators=dense_scalars,
                coeff=-np.sqrt(2.0) * coeff,
                family=family,
            )
        else:
            autompo.add_reduced_string_product(
                (
                    annihilate_site,
                    with_density(
                        annihilation.right_multiply_sector_scalar(double_phase),
                        annihilate_site,
                    ).right_multiply_sector_scalar(parity),
                ),
                (
                    create_site,
                    time_reversed_reduced_operator(with_density(creation, create_site)),
                ),
                intermediate_irreps=(SU2Irrep(1),),
                dense_site_operators=dense_scalars,
                coeff=np.sqrt(2.0) * coeff,
                family=family,
            )
        return autompo

    direct_one_body = density is None
    split_family = (
        (family, _FULLY_REDUCED_ONE_BODY_SPLIT_FAMILY)
        if direct_one_body
        else family
    )
    left_site = min(create_site, annihilate_site)
    right_site = max(create_site, annihilate_site)
    middle = (
        {site: parity for site in range(left_site + 1, right_site)}
        if direct_one_body
        else {}
    )

    annihilate_empty_single, annihilate_single_double = (
        _split_spatial_fermion_annihilation_channels(phys_leg, dtype=dtype)
    )
    create_empty_single = annihilate_empty_single.adjoint()
    create_single_double = annihilate_single_double.adjoint()

    middle_sites = tuple(range(left_site + 1, right_site))
    middle_key = None
    direction = "forward" if create_site < annihilate_site else "backward"
    if direct_one_body and middle_sites:
        if direction == "forward" and len(middle_sites) == 1:
            middle_key = (direction, len(middle_sites), create_site)
        elif direction == "backward" and len(middle_sites) in {1, 2}:
            middle_key = (direction, len(middle_sites), None)
    middle_channel_coeffs = _ONE_BODY_MIDDLE_CHANNEL_COEFFS.get(middle_key)
    if middle_channel_coeffs is not None:
        if create_site < annihilate_site:
            labelled_terms = {
                "c01/a01": (
                    create_empty_single.left_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                    time_reversed_reduced_operator(annihilate_empty_single),
                ),
                "c01/a12": (
                    create_empty_single.left_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                    time_reversed_reduced_operator(annihilate_single_double),
                ),
                "c12/a01": (
                    create_single_double.left_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                    time_reversed_reduced_operator(annihilate_empty_single),
                ),
                "c12/a12": (
                    create_single_double.left_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                    time_reversed_reduced_operator(annihilate_single_double),
                ),
            }
            for (label, middle_labels), channel_coeff in middle_channel_coeffs.items():
                left_operator, right_operator = labelled_terms[label]
                channel_family = _with_prefix_family(
                    split_family,
                    "__prefix_fr_ob_" + label.replace("/", "_") + "_" + "_".join(middle_labels),
                )
                _add_fully_reduced_one_body_middle_channel(
                    autompo,
                    ((create_site, left_operator), (annihilate_site, right_operator)),
                    middle_sites,
                    middle_labels,
                    channel_coeff * coeff,
                    phys_leg=phys_leg,
                    dtype=dtype,
                    family=channel_family,
                )
        else:
            labelled_terms = {
                "a01/c01": (
                    annihilate_empty_single.right_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                    time_reversed_reduced_operator(create_empty_single),
                ),
                "a01/c12": (
                    annihilate_empty_single.right_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                    time_reversed_reduced_operator(create_single_double),
                ),
                "a12/c01": (
                    annihilate_single_double.right_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                    time_reversed_reduced_operator(create_empty_single),
                ),
                "a12/c12": (
                    annihilate_single_double.right_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                    time_reversed_reduced_operator(create_single_double),
                ),
            }
            for (label, middle_labels), channel_coeff in middle_channel_coeffs.items():
                left_operator, right_operator = labelled_terms[label]
                channel_family = _with_prefix_family(
                    split_family,
                    "__prefix_fr_ob_" + label.replace("/", "_") + "_" + "_".join(middle_labels),
                )
                _add_fully_reduced_one_body_middle_channel(
                    autompo,
                    ((annihilate_site, left_operator), (create_site, right_operator)),
                    middle_sites,
                    middle_labels,
                    channel_coeff * coeff,
                    phys_leg=phys_leg,
                    dtype=dtype,
                    family=channel_family,
                )
        return autompo

    if create_site < annihilate_site:
        channel_terms = (
            (create_empty_single, annihilate_empty_single, -np.sqrt(2.0)),
            (create_empty_single, annihilate_single_double, -np.sqrt(2.0)),
            (create_single_double, annihilate_empty_single, -np.sqrt(2.0)),
            (create_single_double, annihilate_single_double, 1.0 / np.sqrt(2.0)),
        )
        for creation, annihilation, channel_coeff in channel_terms:
            autompo.add_reduced_string_product(
                (
                    create_site,
                    with_density(
                        creation.left_multiply_sector_scalar(double_phase),
                        create_site,
                    ).right_multiply_sector_scalar(parity),
                ),
                (
                    annihilate_site,
                    time_reversed_reduced_operator(
                        with_density(annihilation, annihilate_site)
                    ),
                ),
                intermediate_irreps=(SU2Irrep(1),),
                dense_site_operators=dense_scalars,
                middle_operators=middle,
                coeff=channel_coeff * coeff,
                family=split_family,
            )
    else:
        channel_terms = (
            (annihilate_empty_single, create_empty_single, np.sqrt(2.0)),
            (annihilate_empty_single, create_single_double, np.sqrt(2.0)),
            (annihilate_single_double, create_empty_single, np.sqrt(2.0)),
            (annihilate_single_double, create_single_double, -1.0 / np.sqrt(2.0)),
        )
        for annihilation, creation, channel_coeff in channel_terms:
            autompo.add_reduced_string_product(
                (
                    annihilate_site,
                    with_density(
                        annihilation.right_multiply_sector_scalar(double_phase),
                        annihilate_site,
                    ).right_multiply_sector_scalar(parity),
                ),
                (
                    create_site,
                    time_reversed_reduced_operator(with_density(creation, create_site)),
                ),
                intermediate_irreps=(SU2Irrep(1),),
                dense_site_operators=dense_scalars,
                middle_operators=middle,
                coeff=channel_coeff * coeff,
                family=split_family,
            )
    return autompo


def _fully_reduced_double_transition_phase(phys_leg, *, dtype):
    blocks = {}
    for sector in phys_leg.sectors:
        scale = -1.0 if int(getattr(sector, "charge", -1)) == 2 else 1.0
        blocks[(sector, sector)] = np.asarray(scale, dtype=dtype) * np.eye(
            phys_leg.sector_dim(sector),
            dtype=dtype,
        )
    return SiteOperator(blocks=blocks, phys_out_leg=phys_leg, phys_in_leg=phys_leg)


def add_spatial_one_body_terms(
    autompo,
    h1e,
    *,
    cutoff=1.0e-12,
    family="R",
    split_fully_reduced=True,
):
    """
    Add a spin-summed spatial-orbital one-body Hamiltonian.

    The operator is

        H_1 = sum_{pq,sigma} h[p,q] c^dagger_{p,sigma} c_{q,sigma}

    with off-diagonal terms represented through reduced SU(2) rank-1/2
    endpoint tensors and Jordan-Wigner parity strings.
    """
    if not isinstance(autompo, AutoMPO):
        raise TypeError("add_spatial_one_body_terms expects an AutoMPO.")
    h1e = np.asarray(h1e)
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError("h1e must be a square matrix.")
    if h1e.shape[0] != autompo.nsites:
        raise ValueError(
            f"h1e dimension {h1e.shape[0]} does not match AutoMPO site count {autompo.nsites}."
        )
    if any(phys_leg != autompo.site_legs[0] for phys_leg in autompo.site_legs):
        raise ValueError(
            "add_spatial_one_body_terms expects a uniform spatial-orbital Leg across all sites."
        )
    phys_leg = autompo.site_legs[0]
    # Validate the leg is one of the supported spatial-orbital conventions.
    physical_leg_from_spatial_orbital(phys_leg)

    dtype = np.result_type(h1e.dtype, float)
    fully_reduced = _is_fully_reduced_spatial_leg(phys_leg)
    reduced_fermion = reduced_spatial_fermion_annihilation(phys_leg, dtype=dtype)
    reduced_creation = reduced_fermion.adjoint()

    for site, coeff in enumerate(np.diag(h1e)):
        if abs(coeff) > cutoff:
            autompo.add_onsite(
                site,
                spatial_number(phys_leg, dtype=dtype),
                coeff=coeff,
                family=family,
            )

    for left_site in range(autompo.nsites):
        for right_site in range(left_site + 1, autompo.nsites):
            forward = h1e[left_site, right_site]
            if abs(forward) > cutoff:
                if fully_reduced:
                    _add_fully_reduced_spinfree_bilinear(
                        autompo,
                        left_site,
                        right_site,
                        forward,
                        phys_leg=phys_leg,
                        dtype=dtype,
                        family=family,
                        split_channels=split_fully_reduced,
                    )
                else:
                    autompo.add_fermionic_reduced_bilinear(
                        left_site,
                        reduced_creation,
                        right_site,
                        reduced_fermion,
                        coeff=forward,
                        family=family,
                    )
            backward = h1e[right_site, left_site]
            if abs(backward) > cutoff:
                # Ordered-site form: c^dagger_j c_i = - c_i c^dagger_j for i < j.
                if fully_reduced:
                    _add_fully_reduced_spinfree_bilinear(
                        autompo,
                        right_site,
                        left_site,
                        backward,
                        phys_leg=phys_leg,
                        dtype=dtype,
                        family=family,
                        split_channels=split_fully_reduced,
                    )
                else:
                    autompo.add_fermionic_reduced_bilinear(
                        left_site,
                        reduced_fermion,
                        right_site,
                        reduced_creation,
                        coeff=-backward,
                        family=family,
                    )

    return autompo


def build_spatial_one_body_reduced_mpo(sites_or_legs, h1e, *, cutoff=1.0e-12):
    """
    Build the reduced-channel MPO for a spin-summed spatial one-body Hamiltonian.
    """
    site_legs = _normalize_site_legs(sites_or_legs)
    autompo = AutoMPO(site_legs)
    add_spatial_one_body_terms(autompo, h1e, cutoff=cutoff)
    factors = autompo.build()
    if factors and _is_fully_reduced_spatial_leg(site_legs[0]):
        factors = [as_rank_coupled_mpo(factor) for factor in factors]
    return factors


def _canonical_spatial_jw_site_operators(operators, sites, *, phys_leg, dtype, cutoff):
    """
    Convert a global spatial-fermion string into ordered local site operators.
    """
    parity = spatial_parity(dtype=dtype)
    identity = np.eye(phys_leg.total_dim, dtype=dtype)
    grouped = {}
    for operator, site in zip(operators, sites):
        site = int(site)
        for parity_site in range(site):
            grouped.setdefault(parity_site, []).append(parity)
        grouped.setdefault(site, []).append(operator)

    site_operators = []
    for site in sorted(grouped):
        local_operator = compose_site_operators(*grouped[site])
        dense = local_operator.as_dense()
        if np.linalg.norm(dense.reshape(-1)) <= cutoff:
            return ()
        if np.allclose(dense, identity, atol=cutoff, rtol=0.0):
            continue
        site_operators.append((site, local_operator))
    return tuple(site_operators)


def _dense_matrix_from_local_mpo(mpo):
    states = {0: np.array([[1.0]], dtype=complex)}
    for core in mpo:
        dense_core = core.as_dense() if hasattr(core, "as_dense") else np.asarray(core)
        new_states = {}
        for left_index, accum in states.items():
            for right_index in range(dense_core.shape[1]):
                local = dense_core[left_index, right_index]
                if not np.any(local):
                    continue
                contrib = np.kron(accum, local)
                if right_index in new_states:
                    new_states[right_index] += contrib
                else:
                    new_states[right_index] = contrib
        states = new_states
    return states[0]


def _kron_all_dense(operators):
    out = np.asarray(operators[0], dtype=complex)
    for operator in operators[1:]:
        out = np.kron(out, np.asarray(operator, dtype=complex))
    return out


def _spinfree_component_target_dense(site_order):
    """
    Dense four-site target for E_pq E_rs in the sorted site order.

    ``site_order`` maps sorted-chain positions to original operator positions
    in ``(c^dag_p, c_q, c^dag_r, c_s)``.
    """
    position_of_original = {original: position for position, original in enumerate(site_order)}
    parity = spatial_parity().as_dense().astype(complex)
    identity = np.eye(4, dtype=complex)
    spin_ops = {
        ("create", 0): spatial_create_up().as_dense().astype(complex),
        ("create", 1): spatial_create_down().as_dense().astype(complex),
        ("annihilate", 0): spatial_annihilate_up().as_dense().astype(complex),
        ("annihilate", 1): spatial_annihilate_down().as_dense().astype(complex),
    }
    original_kinds = ("create", "annihilate", "create", "annihilate")
    target = np.zeros((4**4, 4**4), dtype=complex)
    for sigma in (0, 1):
        for tau in (0, 1):
            spin_labels = (sigma, sigma, tau, tau)
            grouped = {}
            for original, (kind, spin) in enumerate(zip(original_kinds, spin_labels)):
                site = position_of_original[original]
                for parity_site in range(site):
                    grouped.setdefault(parity_site, []).append(parity)
                grouped.setdefault(site, []).append(spin_ops[(kind, spin)])
            local = [identity.copy() for _ in range(4)]
            for site, pieces in grouped.items():
                operator = identity.copy()
                for piece in pieces:
                    operator = operator @ piece
                local[site] = operator
            target += _kron_all_dense(local)
    return target


def _we_basis_dense_for_spinfree_order(site_order, middle_irrep):
    phys_leg = physical_leg_from_spatial_orbital()
    annihilation = reduced_spatial_fermion_annihilation()
    creation = annihilation.adjoint()
    dual_annihilation = time_reversed_reduced_operator(annihilation)
    parity = spatial_parity()
    original_kinds = ("create", "annihilate", "create", "annihilate")
    operators = []
    for position, original in enumerate(site_order):
        operator = creation if original_kinds[original] == "create" else dual_annihilation
        if position in (0, 2):
            operator = operator.right_multiply_sector_scalar(parity)
        operators.append((position, operator))
    builder = AutoMPO([phys_leg] * 4)
    builder.add_reduced_string(
        *operators,
        intermediate_irreps=(SU2Irrep(1), SU2Irrep(middle_irrep), SU2Irrep(1)),
    )
    return _dense_matrix_from_local_mpo(builder.build())


def _reference_spinfree_we_recoupling_coefficients(site_order, *, cutoff=1.0e-12):
    """
    Dense reference coefficients for the K=0 and K=1 WE channels.

    The coefficients are the SU(2) recoupling from pair-coupled
    ``E_pq E_rs`` to the left-associated chain order used by ``AutoMPO``.
    This determinant-space projection is retained only for validation tests.
    """
    site_order = tuple(int(index) for index in site_order)
    cached = _SPINFREE_WE_RECOUPLING_CACHE.get(site_order)
    if cached is not None:
        return cached
    target = _spinfree_component_target_dense(site_order).reshape(-1)
    basis = np.column_stack(
        [
            _we_basis_dense_for_spinfree_order(site_order, middle_irrep).reshape(-1)
            for middle_irrep in (0, 2)
        ]
    )
    coeffs, *_ = np.linalg.lstsq(basis, target, rcond=None)
    residual = np.linalg.norm(basis @ coeffs - target)
    if residual > cutoff * max(1.0, np.linalg.norm(target)):
        raise RuntimeError(
            f"Failed to recouple spin-free WE channels for site order {site_order}: residual={residual:.3e}."
        )
    coeffs = tuple(
        0.0 if abs(coeff) <= cutoff else np.real_if_close(coeff).item()
        for coeff in coeffs
    )
    _SPINFREE_WE_RECOUPLING_CACHE[site_order] = coeffs
    return coeffs


_SPINFREE_WE_CHANNELS = {
    (0, 1, 2, 3): (2.0, 0.0),
    (0, 1, 3, 2): (2.0, 0.0),
    (1, 0, 2, 3): (2.0, 0.0),
    (1, 0, 3, 2): (2.0, 0.0),
    (2, 3, 0, 1): (2.0, 0.0),
    (2, 3, 1, 0): (2.0, 0.0),
    (3, 2, 0, 1): (2.0, 0.0),
    (3, 2, 1, 0): (2.0, 0.0),
    (0, 2, 1, 3): (-1.0, -_SQRT3),
    (0, 3, 1, 2): (-1.0, -_SQRT3),
    (1, 2, 0, 3): (-1.0, -_SQRT3),
    (1, 3, 0, 2): (-1.0, -_SQRT3),
    (2, 0, 3, 1): (-1.0, -_SQRT3),
    (2, 1, 3, 0): (-1.0, -_SQRT3),
    (3, 0, 2, 1): (-1.0, -_SQRT3),
    (3, 1, 2, 0): (-1.0, -_SQRT3),
    (0, 2, 3, 1): (-1.0, _SQRT3),
    (0, 3, 2, 1): (-1.0, _SQRT3),
    (1, 2, 3, 0): (-1.0, _SQRT3),
    (1, 3, 2, 0): (-1.0, _SQRT3),
    (2, 0, 1, 3): (-1.0, _SQRT3),
    (2, 1, 0, 3): (-1.0, _SQRT3),
    (3, 0, 1, 2): (-1.0, _SQRT3),
    (3, 1, 0, 2): (-1.0, _SQRT3),
}


def _spinfree_we_recoupling_coefficients(site_order, *, cutoff=1.0e-12):
    """Analytic SU(2) recoupling coefficients for four distinct sites."""
    del cutoff
    key = tuple(int(index) for index in site_order)
    try:
        return _SPINFREE_WE_CHANNELS[key]
    except KeyError as exc:
        raise ValueError(f"Expected a permutation of four operator positions, got {key!r}.") from exc


def _middle_parity_operators_for_four_sites(sorted_sites, *, phys_leg, dtype):
    parity = spatial_parity(phys_leg, dtype=dtype)
    middle = {}
    for position in (0, 2):
        left = int(sorted_sites[position])
        right = int(sorted_sites[position + 1])
        for site in range(left + 1, right):
            middle[site] = parity
    return middle


def _accumulate_four_distinct_spinfree_we_product(
    pending_we,
    p,
    q,
    r,
    s,
    coeff,
    *,
    cutoff,
    middle_irreps=(0, 2),
):
    """
    Accumulate shared reduced WE strings for one four-distinct ERI product.
    """
    original = (
        (int(p), 0),
        (int(q), 1),
        (int(r), 2),
        (int(s), 3),
    )
    sorted_entries = tuple(sorted(original, key=lambda item: item[0]))
    sorted_sites = tuple(site for site, _ in sorted_entries)
    site_order = tuple(original_index for _, original_index in sorted_entries)
    recoupling = _spinfree_we_recoupling_coefficients(site_order, cutoff=cutoff)
    allowed_middle_irreps = {int(irrep) for irrep in middle_irreps}

    count = 0
    for middle_irrep, recoupling_coeff in zip((0, 2), recoupling):
        if int(middle_irrep) not in allowed_middle_irreps:
            continue
        term_coeff = coeff * recoupling_coeff
        if abs(term_coeff) <= cutoff:
            continue
        key = (sorted_sites, site_order, int(middle_irrep))
        pending_we[key] = pending_we.get(key, 0.0) + term_coeff
        if abs(pending_we[key]) <= cutoff:
            pending_we.pop(key, None)
        count += 1
    return count


def _emit_shared_spinfree_we_terms(
    autompo,
    pending_we,
    *,
    phys_leg,
    dtype,
    cutoff,
    family="P",
):
    annihilation = reduced_spatial_fermion_annihilation(phys_leg, dtype=dtype)
    creation = annihilation.adjoint()
    dual_annihilation = time_reversed_reduced_operator(annihilation)
    parity = spatial_parity(phys_leg, dtype=dtype)
    original_kinds = ("create", "annihilate", "create", "annihilate")
    count = 0
    for (sorted_sites, site_order, middle_irrep), coeff in pending_we.items():
        if abs(coeff) <= cutoff:
            continue
        middle = _middle_parity_operators_for_four_sites(
            sorted_sites,
            phys_leg=phys_leg,
            dtype=dtype,
        )
        operators = []
        for position, (site, original_index) in enumerate(zip(sorted_sites, site_order)):
            kind = original_kinds[original_index]
            operator = creation if kind == "create" else dual_annihilation
            if position in (0, 2):
                operator = operator.right_multiply_sector_scalar(parity)
            operators.append((site, operator))
        autompo.add_reduced_string(
            *operators,
            intermediate_irreps=(SU2Irrep(1), SU2Irrep(middle_irrep), SU2Irrep(1)),
            coeff=coeff,
            middle_operators=middle,
            family=family,
        )
        count += 1
    return count


def _spinfree_component_target_dense_for_sites(original_sites):
    """
    Dense target for ``E_pq E_rs`` on the unique sorted spatial sites.
    """
    original_sites = tuple(int(site) for site in original_sites)
    unique_sites = tuple(sorted(set(original_sites)))
    site_positions = {site: position for position, site in enumerate(unique_sites)}
    positions = tuple(site_positions[site] for site in original_sites)
    nsites = len(unique_sites)
    phys_leg = physical_leg_from_spatial_orbital()
    identity = np.eye(phys_leg.total_dim, dtype=complex)
    spin_ops = {
        ("create", 0): spatial_create_up(dtype=complex),
        ("create", 1): spatial_create_down(dtype=complex),
        ("annihilate", 0): spatial_annihilate_up(dtype=complex),
        ("annihilate", 1): spatial_annihilate_down(dtype=complex),
    }
    target = np.zeros((4**nsites, 4**nsites), dtype=complex)
    original_kinds = ("create", "annihilate", "create", "annihilate")
    for sigma in (0, 1):
        for tau in (0, 1):
            spin_labels = (sigma, sigma, tau, tau)
            operators = tuple(
                spin_ops[(kind, spin)]
                for kind, spin in zip(original_kinds, spin_labels)
            )
            site_ops = _canonical_spatial_jw_site_operators(
                operators,
                positions,
                phys_leg=phys_leg,
                dtype=complex,
                cutoff=1.0e-14,
            )
            if not site_ops:
                continue
            local = [identity.copy() for _ in range(nsites)]
            for site, operator in site_ops:
                local[site] = operator.as_dense().astype(complex)
            target += _kron_all_dense(local)
    return target


def _scalar_chain_intermediates(ranks):
    """
    Return left-associated intermediate irreps for a scalar reduced string.
    """
    ranks = tuple(ranks)
    scalar = SU2Irrep(0)
    if len(ranks) < 2:
        return None
    if len(ranks) == 2:
        if ranks[0] == ranks[1]:
            return (ranks[0],)
        return None
    if len(ranks) == 3:
        first = ranks[0]
        second = ranks[2]
        if second in left_or_right_fusion(first, ranks[1]) and scalar in left_or_right_fusion(second, ranks[2]):
            return (first, second)
        return None
    return None


def _exchange_reduced_string_specs(original_sites, *, phys_leg, dtype):
    """
    Build reduced-string candidates for exchange-style repeated ERI products.
    """
    original_sites = tuple(int(site) for site in original_sites)
    unique_sites = tuple(sorted(set(original_sites)))
    site_to_originals = {
        site: tuple(index for index, original_site in enumerate(original_sites) if original_site == site)
        for site in unique_sites
    }
    annihilation = reduced_spatial_fermion_annihilation(phys_leg, dtype=dtype)
    creation = annihilation.adjoint()
    dual_annihilation = time_reversed_reduced_operator(annihilation)
    fully_reduced = _is_fully_reduced_spatial_leg(phys_leg)
    double_phase = (
        _fully_reduced_double_transition_phase(phys_leg, dtype=dtype)
        if fully_reduced
        else None
    )
    original_ops = (creation, dual_annihilation, creation, dual_annihilation)
    original_ranks = (SU2Irrep(1), SU2Irrep(1), SU2Irrep(1), SU2Irrep(1))
    parity = spatial_parity(phys_leg, dtype=dtype)

    sorted_entries = tuple(sorted(
        ((site, original_index) for original_index, site in enumerate(original_sites)),
        key=lambda item: (item[0], item[1]),
    ))
    parity_sites = {}
    for start in (0, 2):
        if start + 1 >= len(sorted_entries):
            continue
        left_site = sorted_entries[start][0]
        right_site = sorted_entries[start + 1][0]
        for site in range(left_site, right_site):
            parity_sites[site] = parity_sites.get(site, 0) + 1

    choices = []
    for site in unique_sites:
        originals = site_to_originals[site]
        if len(originals) == 1:
            original = originals[0]
            operator = original_ops[original]
            if fully_reduced:
                forward_repeated_exchange = not (
                    original == 0
                    and original_sites[1] == original_sites[2]
                    and original_sites[0] > original_sites[3]
                )
                if original in (0, 2) and forward_repeated_exchange:
                    operator = operator.left_multiply_sector_scalar(double_phase)
                elif original == 3 and site < original_sites[0]:
                    operator = operator.right_multiply_sector_scalar(double_phase)
            choices.append(((operator, original_ranks[original]),))
            continue
        if len(originals) != 2:
            return ()
        first, second = originals
        other_sites = tuple(
            original_sites[index]
            for index in range(len(original_sites))
            if index not in originals
        )
        pair_between_other_sites = min(other_sites) < site < max(other_sites)
        left_operator = (
            original_ops[first].right_multiply_sector_scalar(double_phase)
            if fully_reduced and pair_between_other_sites
            else original_ops[first]
        )
        right_operator = original_ops[second]
        choices.append(tuple(
            (
                coupled_reduced_tensor_product(
                    left_operator,
                    right_operator,
                    rank,
                ),
                rank,
            )
            for rank in (SU2Irrep(0), SU2Irrep(2))
        ))

    specs = []
    for selected in np.array(np.meshgrid(*[np.arange(len(choice)) for choice in choices])).T.reshape(-1, len(choices)):
        operators = []
        ranks = []
        for site, choice_index, site_choices in zip(unique_sites, selected, choices):
            operator, rank = site_choices[int(choice_index)]
            if parity_sites.get(site, 0) % 2:
                operator = operator.right_multiply_sector_scalar(parity)
            operators.append(operator)
            ranks.append(rank)
        intermediate_irreps = _scalar_chain_intermediates(ranks)
        if intermediate_irreps is None:
            continue
        middle = {
            site: parity
            for start, stop in zip(unique_sites, unique_sites[1:])
            for site in range(start + 1, stop)
            if parity_sites.get(site, 0) % 2
        }
        specs.append(
            (
                tuple(zip(unique_sites, operators)),
                tuple(intermediate_irreps),
                middle,
                tuple(ranks),
            )
        )
    return tuple(specs)


def _exchange_recoupling_key(original_sites):
    original_sites = tuple(int(site) for site in original_sites)
    unique_sites = tuple(sorted(set(original_sites)))
    site_positions = {site: position for position, site in enumerate(unique_sites)}
    return tuple(site_positions[site] for site in original_sites)


def _fully_reduced_exchange_order_phase(original_sites):
    """
    Return the residual exchange-site phase for reduced repeated-index strings.
    """
    p, q, r, s = (int(site) for site in original_sites)
    if p == s and not (min(q, r) < p < max(q, r)):
        return -1.0
    return 1.0


def _reference_spinfree_exchange_recoupling_coefficients(original_sites, *, cutoff=1.0e-12):
    """
    Dense reference recoupling for exchange repeated-site products.

    This determinant-space projection is retained only for validation tests.
    """
    key = _exchange_recoupling_key(original_sites)
    cached = _SPINFREE_EXCHANGE_RECOUPLING_CACHE.get(key)
    if cached is not None:
        return cached
    canonical_leg = physical_leg_from_spatial_orbital()
    specs = _exchange_reduced_string_specs(key, phys_leg=canonical_leg, dtype=complex)
    if not specs:
        raise NotImplementedError("No scalar reduced exchange recoupling channels are available.")
    target = _spinfree_component_target_dense_for_sites(key).reshape(-1)
    basis_columns = []
    for site_ops, intermediate_irreps, middle, _ranks in specs:
        builder = AutoMPO([canonical_leg] * len(set(key)))
        builder.add_reduced_string(
            *site_ops,
            intermediate_irreps=intermediate_irreps,
            middle_operators=middle,
        )
        basis_columns.append(_dense_matrix_from_local_mpo(builder.build()).reshape(-1))
    basis = np.column_stack(basis_columns)
    coeffs, *_ = np.linalg.lstsq(basis, target, rcond=None)
    residual = np.linalg.norm(basis @ coeffs - target)
    if residual > cutoff * max(1.0, np.linalg.norm(target)):
        raise RuntimeError(
            f"Failed to recouple exchange repeated-site ERI pattern {key}: residual={residual:.3e}."
        )
    coeffs = tuple(
        0.0 if abs(coeff) <= cutoff else np.real_if_close(coeff).item()
        for coeff in coeffs
    )
    cached = tuple((coeff, specs[index][3]) for index, coeff in enumerate(coeffs) if abs(coeff) > cutoff)
    _SPINFREE_EXCHANGE_RECOUPLING_CACHE[key] = cached
    return cached


_SPINFREE_EXCHANGE_CHANNELS = {
    (0, 1, 1, 0): ((-1.0, (0, 0)), (-_SQRT3, (2, 2))),
    (0, 1, 1, 2): ((-1.0, (1, 0, 1)), (_SQRT3, (1, 2, 1))),
    (0, 1, 2, 0): ((-1.0, (0, 1, 1)), (-_SQRT3, (2, 1, 1))),
    (0, 2, 1, 0): ((-1.0, (0, 1, 1)), (_SQRT3, (2, 1, 1))),
    (0, 2, 2, 1): ((-1.0, (1, 1, 0)), (-_SQRT3, (1, 1, 2))),
    (1, 0, 0, 1): ((-1.0, (0, 0)), (-_SQRT3, (2, 2))),
    (1, 0, 0, 2): ((-1.0, (0, 1, 1)), (-_SQRT3, (2, 1, 1))),
    (1, 0, 2, 1): ((-1.0, (1, 0, 1)), (_SQRT3, (1, 2, 1))),
    (1, 2, 0, 1): ((-1.0, (1, 0, 1)), (-_SQRT3, (1, 2, 1))),
    (1, 2, 2, 0): ((-1.0, (1, 1, 0)), (_SQRT3, (1, 1, 2))),
    (2, 0, 0, 1): ((-1.0, (0, 1, 1)), (_SQRT3, (2, 1, 1))),
    (2, 0, 1, 2): ((-1.0, (1, 1, 0)), (-_SQRT3, (1, 1, 2))),
    (2, 1, 0, 2): ((-1.0, (1, 1, 0)), (_SQRT3, (1, 1, 2))),
    (2, 1, 1, 0): ((-1.0, (1, 0, 1)), (-_SQRT3, (1, 2, 1))),
}


def _spinfree_exchange_recoupling_coefficients(original_sites, *, cutoff=1.0e-12):
    """Analytic SU(2) recoupling for repeated-site exchange strings."""
    key = _exchange_recoupling_key(original_sites)
    try:
        channels = _SPINFREE_EXCHANGE_CHANNELS[key]
    except KeyError as exc:
        raise NotImplementedError(
            f"No scalar reduced exchange recoupling channels for pattern {key!r}."
        ) from exc
    return tuple(
        (coeff, tuple(SU2Irrep(rank) for rank in ranks))
        for coeff, ranks in channels
        if abs(coeff) > cutoff
    )


def _site_operator_signature(site_operator, *, cutoff):
    dense = np.asarray(site_operator.as_dense(), dtype=np.complex128)
    dense = np.array(dense, copy=True)
    dense[np.abs(dense) <= cutoff] = 0.0
    return dense.shape, dense.tobytes()


def _scalar_product_key(site_ops, *, cutoff):
    return tuple(
        (int(site), _site_operator_signature(operator, cutoff=cutoff))
        for site, operator in site_ops
    )


def _accumulate_spinfree_component_product_terms(pending_scalar, p, q, r, s, coeff, *, dtype, cutoff):
    create_destroy = (
        (spatial_create_up(dtype=dtype), spatial_annihilate_up(dtype=dtype)),
        (spatial_create_down(dtype=dtype), spatial_annihilate_down(dtype=dtype)),
    )
    canonical_leg = physical_leg_from_spatial_orbital()
    count = 0
    for left_create, left_destroy in create_destroy:
        for right_create, right_destroy in create_destroy:
            site_ops = _canonical_spatial_jw_site_operators(
                (left_create, left_destroy, right_create, right_destroy),
                (p, q, r, s),
                phys_leg=canonical_leg,
                dtype=dtype,
                cutoff=cutoff,
            )
            if not site_ops:
                continue
            key = _scalar_product_key(site_ops, cutoff=cutoff)
            total, stored_site_ops = pending_scalar.get(key, (0.0, site_ops))
            total = total + coeff
            if abs(total) <= cutoff:
                pending_scalar.pop(key, None)
            else:
                pending_scalar[key] = (total, stored_site_ops)
            count += 1
    return count


def _emit_shared_scalar_product_terms(autompo, pending_scalar, *, cutoff, family="P"):
    count = 0
    for coeff, site_ops in pending_scalar.values():
        if abs(coeff) <= cutoff:
            continue
        autompo.add_term(*site_ops, coeff=coeff, family=family)
        count += 1
    return count


def _add_fully_reduced_density_eri_term(autompo, p, r, coeff, *, phys_leg, dtype, cutoff):
    """
    Add the fully reduced density part of one diagonal ERI contribution.

    This covers ``p=q`` and ``r=s`` in
    ``1/2 (pq|rs) (E_pq E_rs - delta_qr E_ps)``.  For ``p != r`` the term is
    ``coeff * n_p n_r``.  For ``p == r`` it reduces to
    ``2 * coeff * D_p`` where ``D_p`` projects onto double occupation because
    ``n_p^2 - n_p = 2 D_p``.
    """

    p = int(p)
    r = int(r)
    if abs(coeff) <= cutoff:
        return 0
    if p == r:
        autompo.add_onsite(
            p,
            spatial_double_occupancy(phys_leg, dtype=dtype),
            coeff=2.0 * coeff,
        )
    else:
        density = spatial_number(phys_leg, dtype=dtype)
        autompo.add_term((p, density), (r, density), coeff=coeff)
    return 1


def _add_fully_reduced_density_bilinear_eri_term(
    autompo,
    density_site,
    create_site,
    annihilate_site,
    coeff,
    *,
    phys_leg,
    dtype,
    cutoff,
):
    """
    Add ``coeff * n_i E_pq`` in the fully reduced spatial basis.

    If the density site coincides with a bilinear endpoint, the same-site
    product is represented by composing the density scalar on the output sector
    of the reduced endpoint tensor, avoiding any spin-component fallback.
    """

    density_site = int(density_site)
    create_site = int(create_site)
    annihilate_site = int(annihilate_site)
    if abs(coeff) <= cutoff:
        return 0
    if create_site == annihilate_site:
        return _add_fully_reduced_density_eri_term(
            autompo,
            density_site,
            create_site,
            coeff,
            phys_leg=phys_leg,
            dtype=dtype,
            cutoff=cutoff,
        )
    _add_fully_reduced_spinfree_bilinear(
        autompo,
        create_site,
        annihilate_site,
        coeff,
        phys_leg=phys_leg,
        dtype=dtype,
        density_site=density_site,
    )
    return 1


def _add_fully_reduced_pair_eri_term(
    autompo,
    p,
    q,
    r,
    s,
    coeff,
    *,
    phys_leg,
    dtype,
    cutoff,
):
    """
    Add fully reduced ERI terms with same-site pair creation or annihilation.

    This covers repeated patterns ``p == r`` and/or ``q == s`` after density
    terms have already been handled.  Same-site two-creation/two-annihilation
    products are local SU(2) scalars, while the remaining two odd endpoints are
    emitted as a Wigner-Eckart coupled reduced string.
    """

    p = int(p)
    q = int(q)
    r = int(r)
    s = int(s)
    if abs(coeff) <= cutoff:
        return 0
    coeff = coeff * _FULLY_REDUCED_PAIR_RECOUPLING

    pair_creation = spatial_pair_creation(phys_leg, dtype=dtype)
    pair_annihilation = spatial_pair_annihilation(phys_leg, dtype=dtype)
    double_phase = _fully_reduced_double_transition_phase(phys_leg, dtype=dtype)
    annihilation = reduced_spatial_fermion_annihilation(phys_leg, dtype=dtype)
    creation = annihilation.adjoint()

    if p == r and q == s:
        if p == q:
            return 0
        autompo.add_term((p, pair_creation), (q, pair_annihilation), coeff=coeff)
        return 1

    parity = spatial_parity(phys_leg, dtype=dtype)

    def add_pair_times_reduced_string(
        pair_site,
        pair_operator,
        first_site,
        first_operator,
        second_site,
        second_operator,
        term_coeff,
    ):
        if first_site == second_site:
            return 0
        if pair_site < min(first_site, second_site):
            if pair_operator.block(
                phys_leg.sectors[2],
                phys_leg.sectors[0],
            ) is not None:
                pair_operator = compose_site_operators(double_phase, pair_operator)
            elif pair_operator.block(
                phys_leg.sectors[0],
                phys_leg.sectors[2],
            ) is not None:
                pair_operator = compose_site_operators(pair_operator, double_phase)
        if first_site < second_site:
            left_site, left_operator = first_site, first_operator
            right_site, right_operator = second_site, second_operator
            sign = 1.0
        else:
            left_site, left_operator = second_site, second_operator
            right_site, right_operator = first_site, first_operator
            sign = -1.0
        middle = {
            site: parity
            for site in range(left_site + 1, right_site)
        }
        autompo.add_reduced_string_product(
            (left_site, left_operator.right_multiply_sector_scalar(parity)),
            (right_site, right_operator),
            intermediate_irreps=(SU2Irrep(1),),
            dense_site_operators=((pair_site, pair_operator),),
            middle_operators=middle,
            coeff=sign * term_coeff,
        )
        return 1

    if p == r:
        return add_pair_times_reduced_string(
            p,
            pair_creation,
            q,
            annihilation,
            s,
            annihilation,
            coeff,
        )
    if q == s:
        return add_pair_times_reduced_string(
            q,
            pair_annihilation,
            p,
            creation,
            r,
            creation,
            coeff,
        )
    return 0


def _add_fully_reduced_exchange_eri_term(
    autompo,
    p,
    q,
    r,
    s,
    coeff,
    *,
    phys_leg,
    dtype,
    cutoff,
):
    """
    Add exchange-style repeated ERI products in fully reduced form.

    This covers the remaining two-index repeats ``p == s`` and/or ``q == r``.
    Same-site ``c^dagger c`` or ``c c^dagger`` products are represented as
    rank-coupled local reduced tensors, with recoupling coefficients obtained
    from the canonical dense spin-free reference and cached by pattern.
    """

    original_sites = (int(p), int(q), int(r), int(s))
    if abs(coeff) <= cutoff:
        return 0
    order_phase = _fully_reduced_exchange_order_phase(original_sites)
    recoupling = _spinfree_exchange_recoupling_coefficients(
        original_sites,
        cutoff=cutoff,
    )
    coeff_by_ranks = {ranks: factor for factor, ranks in recoupling}
    count = 0
    for site_ops, intermediate_irreps, middle, ranks in _exchange_reduced_string_specs(
        original_sites,
        phys_leg=phys_leg,
        dtype=dtype,
    ):
        factor = coeff_by_ranks.get(ranks, 0.0)
        term_coeff = _FULLY_REDUCED_EXCHANGE_RECOUPLING * order_phase * coeff * factor
        if abs(term_coeff) <= cutoff:
            continue
        autompo.add_reduced_string(
            *site_ops,
            intermediate_irreps=intermediate_irreps,
            middle_operators=middle,
            coeff=term_coeff,
        )
        count += 1
    return count


@dataclass(frozen=True, init=False)
class SpatialSpinFreeERIBuilder:
    """
    Build scalar-coupled reduced MPO terms for restricted spatial ERIs.

    The builder owns the conversion

    ``(pq|rs) -> 1/2 * (E_pq E_rs - delta_qr E_ps)``

    for canonical spatial-orbital SU(2) sites.  It accumulates shared
    Wigner-Eckart strings for four-distinct index patterns and uses local
    scalar products for repeated-site patterns, so callers do not need to
    inspect or emit raw ``AutoMPO`` terms directly.

    :param sites_or_legs: Site count, spatial physical legs, or spatial MPS
        site tensors describing the target MPO chain.
    :param eri_spatial: Restricted spatial ERI tensor in chemist notation
        ``(pq|rs)``.
    :param cutoff: Absolute screening threshold for ERI values and cancelled
        symbolic terms.
    :param include_half: Whether to apply the conventional two-electron
        prefactor ``1/2`` inside the builder.
    :param reduced_we: Whether to emit four-distinct spin-free products through
        reduced Wigner-Eckart strings.
    """

    site_legs: tuple
    eri_spatial: np.ndarray
    cutoff: float
    include_half: bool
    reduced_we: bool

    def __init__(
        self,
        sites_or_legs,
        eri_spatial,
        *,
        cutoff=1.0e-12,
        include_half=True,
        reduced_we=True,
    ):
        site_legs = _normalize_site_legs(sites_or_legs)
        if any(phys_leg != site_legs[0] for phys_leg in site_legs):
            raise ValueError("SpatialSpinFreeERIBuilder expects a uniform spatial physical leg.")
        physical_leg_from_spatial_orbital(site_legs[0])

        eri_spatial = np.asarray(eri_spatial)
        nsites = len(site_legs)
        if eri_spatial.ndim != 4:
            raise ValueError("eri_spatial must have shape (n, n, n, n).")
        if eri_spatial.shape != (nsites, nsites, nsites, nsites):
            raise ValueError(
                f"eri_spatial shape {eri_spatial.shape!r} does not match site count {nsites}."
            )

        object.__setattr__(self, "site_legs", tuple(site_legs))
        object.__setattr__(self, "eri_spatial", eri_spatial)
        object.__setattr__(self, "cutoff", float(cutoff))
        object.__setattr__(self, "include_half", bool(include_half))
        object.__setattr__(self, "reduced_we", bool(reduced_we))

    @property
    def nsites(self):
        """
        Return the number of spatial orbitals represented by this builder.

        :returns: Number of spatial sites in the target MPO chain.
        """
        return len(self.site_legs)

    def add_to(self, autompo, *, return_info=False):
        """
        Add the scalar-coupled ERI Hamiltonian to an ``AutoMPO``.

        :param autompo: Target :class:`AutoMPO` with the same canonical spatial
            physical legs as this builder.
        :param return_info: When ``True``, return a term-count metadata
            dictionary.  Otherwise return only the total emitted term count.
        :returns: Integer term count or metadata dictionary.
        """
        if not isinstance(autompo, AutoMPO):
            raise TypeError("SpatialSpinFreeERIBuilder.add_to expects an AutoMPO.")
        if tuple(autompo.site_legs) != self.site_legs:
            raise ValueError("AutoMPO site legs do not match the ERI builder site legs.")

        dtype = np.result_type(self.eri_spatial.dtype, float)
        values = 0.5 * self.eri_spatial if self.include_half else self.eri_spatial
        cutoff = self.cutoff
        phys_leg = self.site_legs[0]
        canonical_leg = physical_leg_from_spatial_orbital()
        fully_reduced = phys_leg != canonical_leg
        if fully_reduced:
            four_distinct = [
                tuple(int(index) for index in entry)
                for entry in np.argwhere(np.abs(values) > cutoff)
                if len({int(index) for index in entry}) == 4
            ]
            if four_distinct:
                raise NotImplementedError(
                    "Fully reduced spin-free ERI growth does not yet carry the four-site "
                    "recoupling data required for four-distinct index strings. Use the "
                    "sitewise SU(2) component NPDM route for 2-RDMs; determinant-space "
                    "projection is reserved for validation and is not substituted into "
                    "the active solver."
                )
        we_product_terms = 0
        scalar_product_terms = 0
        fully_reduced_density_terms = 0
        fully_reduced_density_bilinear_terms = 0
        fully_reduced_pair_terms = 0
        fully_reduced_exchange_terms = 0
        one_body_correction = np.zeros((self.nsites, self.nsites), dtype=dtype)
        pending_we = {}
        pending_scalar = {}
        for p, q, r, s in np.argwhere(np.abs(values) > cutoff):
            val = values[p, q, r, s]
            fully_reduced_diagonal_density = (
                fully_reduced and int(p) == int(q) and int(r) == int(s)
            )
            fully_reduced_two_site_exchange = (
                fully_reduced
                and int(p) == int(s)
                and int(q) == int(r)
                and int(p) != int(q)
            )
            if q == r and not fully_reduced_diagonal_density and not fully_reduced_two_site_exchange:
                one_body_correction[int(p), int(s)] -= val
            if self.reduced_we and len({int(p), int(q), int(r), int(s)}) == 4:
                added = _accumulate_four_distinct_spinfree_we_product(
                    pending_we,
                    p,
                    q,
                    r,
                    s,
                    val,
                    cutoff=cutoff,
                )
                we_product_terms += added
            else:
                if fully_reduced:
                    if fully_reduced_diagonal_density:
                        fully_reduced_density_terms += _add_fully_reduced_density_eri_term(
                            autompo,
                            p,
                            r,
                            val,
                            phys_leg=phys_leg,
                            dtype=dtype,
                            cutoff=cutoff,
                        )
                        continue
                    if fully_reduced_two_site_exchange:
                        first_site, second_site = sorted((int(p), int(q)))
                        single_projector = spatial_projector("single", phys_leg, dtype=dtype)
                        autompo.add_term(
                            (first_site, single_projector),
                            (second_site, single_projector),
                            coeff=val,
                            family="B",
                        )
                        fully_reduced_exchange_terms += 1
                        continue
                    if int(p) == int(q):
                        fully_reduced_density_bilinear_terms += _add_fully_reduced_density_bilinear_eri_term(
                            autompo,
                            p,
                            r,
                            s,
                            val,
                            phys_leg=phys_leg,
                            dtype=dtype,
                            cutoff=cutoff,
                        )
                        continue
                    if int(r) == int(s):
                        fully_reduced_density_bilinear_terms += _add_fully_reduced_density_bilinear_eri_term(
                            autompo,
                            r,
                            p,
                            q,
                            val,
                            phys_leg=phys_leg,
                            dtype=dtype,
                            cutoff=cutoff,
                        )
                        continue
                    if int(p) == int(r) or int(q) == int(s):
                        fully_reduced_pair_terms += _add_fully_reduced_pair_eri_term(
                            autompo,
                            p,
                            q,
                            r,
                            s,
                            val,
                            phys_leg=phys_leg,
                            dtype=dtype,
                            cutoff=cutoff,
                        )
                        continue
                    if int(p) == int(s) or int(q) == int(r):
                        fully_reduced_exchange_terms += _add_fully_reduced_exchange_eri_term(
                            autompo,
                            p,
                            q,
                            r,
                            s,
                            val,
                            phys_leg=phys_leg,
                            dtype=dtype,
                            cutoff=cutoff,
                        )
                        if int(q) == int(r) and int(p) < min(int(q), int(s)):
                            _add_fully_reduced_spinfree_bilinear(
                                autompo,
                                p,
                                s,
                                -val,
                                phys_leg=phys_leg,
                                dtype=dtype,
                                density_site=q,
                                density_operator=spatial_projector("single", phys_leg, dtype=dtype),
                                family=("Q", "__prefix_projected_exchange_correction"),
                            )
                            fully_reduced_exchange_terms += 1
                        continue
                    raise NotImplementedError(
                        "Fully reduced spin-free ERI support currently covers four-distinct "
                        "Wigner-Eckart strings, diagonal density ERI terms, and density-bilinear "
                        "ERI terms, same-site pair creation/annihilation products, and "
                        "exchange repeated-site reduced products. Other repeated-site scalar "
                        "local products are not implemented yet."
                    )
                added = _accumulate_spinfree_component_product_terms(
                    pending_scalar,
                    p,
                    q,
                    r,
                    s,
                    val,
                    dtype=dtype,
                    cutoff=cutoff,
                )
                scalar_product_terms += added

        we_product_terms = _emit_shared_spinfree_we_terms(
            autompo,
            pending_we,
            phys_leg=phys_leg,
            dtype=dtype,
            cutoff=cutoff,
            family="P",
        )
        scalar_product_terms = _emit_shared_scalar_product_terms(
            autompo,
            pending_scalar,
            cutoff=cutoff,
            family="P",
        )
        correction_terms = int(np.count_nonzero(np.abs(one_body_correction) > cutoff))
        if correction_terms:
            add_spatial_one_body_terms(
                autompo,
                one_body_correction,
                cutoff=cutoff,
                family="Q",
                split_fully_reduced=False,
            )
        term_count = we_product_terms + scalar_product_terms + correction_terms
        if fully_reduced:
            term_count += fully_reduced_density_terms
            term_count += fully_reduced_density_bilinear_terms
            term_count += fully_reduced_pair_terms
            term_count += fully_reduced_exchange_terms

        info = {
            "total_terms": int(term_count),
            "we_product_terms": int(we_product_terms),
            "scalar_product_terms": int(scalar_product_terms),
            "fully_reduced_density_terms": int(fully_reduced_density_terms),
            "fully_reduced_density_bilinear_terms": int(fully_reduced_density_bilinear_terms),
            "fully_reduced_pair_terms": int(fully_reduced_pair_terms),
            "fully_reduced_exchange_terms": int(fully_reduced_exchange_terms),
            "one_body_correction_terms": int(correction_terms),
        }
        return info if return_info else info["total_terms"]

    def build(self, *, return_info=False):
        """
        Build the reduced MPO chain for the ERI contribution only.

        :param return_info: When ``True``, return ``(factors, info)``.
        :returns: MPO factors, or a pair of factors and metadata.
        """
        autompo = AutoMPO(self.site_legs)
        info = self.add_to(autompo, return_info=True)
        factors = autompo.build() if info["total_terms"] else []
        if factors and _is_fully_reduced_spatial_leg(self.site_legs[0]):
            factors = [as_rank_coupled_mpo(factor) for factor in factors]
        if return_info:
            return factors, info
        return factors


def add_cpp_spatial_spinfree_family_terms(
    autompo,
    families,
    *,
    cutoff=1.0e-12,
    return_info=False,
):
    """
    Add the canonical spin-free ERI carrier from C++ ``P/Q`` families.

    The compiled SU(2) system has already screened the active-space ERIs,
    applied the conventional one-half prefactor, and accumulated the
    ``-delta_qr E_ps`` correction into ``Q``.  This function only converts
    those C++ family records into the reference rank-coupled carrier used
    by the current environment implementation; it never revisits the raw
    four-index integral tensor.
    """

    if not isinstance(autompo, AutoMPO):
        raise TypeError(
            "add_cpp_spatial_spinfree_family_terms expects an AutoMPO."
        )
    phys_leg = autompo.site_legs[0]
    if any(leg != phys_leg for leg in autompo.site_legs):
        raise ValueError("C++ spin-free family terms require uniform site legs.")
    if phys_leg != physical_leg_from_spatial_orbital():
        raise NotImplementedError(
            "C++ P/Q carrier construction currently requires canonical "
            "spatial SU(2) sites."
        )

    p_family = families["P"]
    q_family = families["Q"]
    p_entries = getattr(p_family, "entries", p_family)
    q_entries = getattr(q_family, "entries", q_family)
    cutoff = float(cutoff)
    pending_we = {}
    pending_scalar = {}
    dtype = np.dtype(float)

    for key, coefficient in p_entries.items():
        p, q, r, s = (int(index) for index in key)
        coefficient = float(np.real(coefficient))
        if abs(coefficient) <= cutoff:
            continue
        if len({p, q, r, s}) == 4:
            _accumulate_four_distinct_spinfree_we_product(
                pending_we,
                p,
                q,
                r,
                s,
                coefficient,
                cutoff=cutoff,
            )
        else:
            _accumulate_spinfree_component_product_terms(
                pending_scalar,
                p,
                q,
                r,
                s,
                coefficient,
                dtype=dtype,
                cutoff=cutoff,
            )

    we_product_terms = _emit_shared_spinfree_we_terms(
        autompo,
        pending_we,
        phys_leg=phys_leg,
        dtype=dtype,
        cutoff=cutoff,
        family="P",
    )
    scalar_product_terms = _emit_shared_scalar_product_terms(
        autompo,
        pending_scalar,
        cutoff=cutoff,
        family="P",
    )

    correction = np.zeros(
        (len(autompo.site_legs), len(autompo.site_legs)),
        dtype=float,
    )
    for key, coefficient in q_entries.items():
        p, s, _q = (int(index) for index in key)
        correction[p, s] += float(np.real(coefficient))
    correction_terms = int(np.count_nonzero(np.abs(correction) > cutoff))
    if correction_terms:
        add_spatial_one_body_terms(
            autompo,
            correction,
            cutoff=cutoff,
            family="Q",
            split_fully_reduced=False,
        )

    info = {
        "total_terms": int(
            we_product_terms + scalar_product_terms + correction_terms
        ),
        "we_product_terms": int(we_product_terms),
        "scalar_product_terms": int(scalar_product_terms),
        "fully_reduced_density_terms": 0,
        "fully_reduced_density_bilinear_terms": 0,
        "fully_reduced_pair_terms": 0,
        "fully_reduced_exchange_terms": 0,
        "one_body_correction_terms": int(correction_terms),
    }
    return info if return_info else info["total_terms"]


def add_spatial_spinfree_eri_terms(
    autompo,
    eri_spatial,
    *,
    cutoff=1.0e-12,
    include_half=True,
    reduced_we=True,
    return_info=False,
):
    """
    Add a general restricted two-electron ERI Hamiltonian in spin-free form.

    ``eri_spatial`` is interpreted as ``(pq|rs)`` and is represented through
    the scalar generators

        E_pq = sum_sigma c^dagger[p,sigma] c[q,sigma]

    using ``E_pq E_rs - delta_qr E_ps``.  Jordan-Wigner signs and repeated-site
    collapses are handled at the local-operator level, so this covers arbitrary
    ERI index patterns rather than only adjacent/disjoint strings.
    """
    builder = SpatialSpinFreeERIBuilder(
        autompo.site_legs,
        eri_spatial,
        cutoff=cutoff,
        include_half=include_half,
        reduced_we=reduced_we,
    )
    return builder.add_to(autompo, return_info=return_info)


def add_spatial_two_generator_product_terms(
    autompo,
    entries,
    *,
    cutoff=1.0e-12,
    family="P",
    reduced_we=True,
    we_middle_irreps=(0, 2),
    return_info=False,
):
    """
    Add ``sum_pqrs entries[p,q,r,s] E_pq E_rs`` in the canonical spatial basis.

    This intentionally does not emit the ``-delta_qr E_ps`` one-body
    correction used by the full ERI Hamiltonian builder; callers that use the
    complementary R/P decomposition should carry that correction in the R
    family.
    """
    if not isinstance(autompo, AutoMPO):
        raise TypeError("add_spatial_two_generator_product_terms expects an AutoMPO.")
    if any(phys_leg != autompo.site_legs[0] for phys_leg in autompo.site_legs):
        raise ValueError(
            "add_spatial_two_generator_product_terms expects a uniform spatial-orbital Leg."
        )
    phys_leg = physical_leg_from_spatial_orbital(autompo.site_legs[0])
    if phys_leg != physical_leg_from_spatial_orbital():
        raise NotImplementedError(
            "Native two-generator product terms are currently implemented for the canonical spatial basis."
        )

    pending_we = {}
    pending_scalar = {}
    dtype = np.result_type(
        *[np.asarray(value).dtype for value in dict(entries or {}).values()],
        float,
    )
    raw_terms = 0
    input_terms = 0
    four_distinct_entries = 0
    repeated_entries = 0
    repeat_histogram = {}
    for key, coeff in dict(entries or {}).items():
        p, q, r, s = (int(index) for index in key)
        coeff = np.asarray(coeff, dtype=dtype).item()
        if abs(coeff) <= cutoff:
            continue
        input_terms += 1
        unique = len({p, q, r, s})
        repeat_histogram[unique] = int(repeat_histogram.get(unique, 0)) + 1
        if unique == 4:
            four_distinct_entries += 1
        else:
            repeated_entries += 1
        if reduced_we and unique == 4:
            raw_terms += _accumulate_four_distinct_spinfree_we_product(
                pending_we,
                p,
                q,
                r,
                s,
                coeff,
                cutoff=cutoff,
                middle_irreps=we_middle_irreps,
            )
        else:
            raw_terms += _accumulate_spinfree_component_product_terms(
                pending_scalar,
                p,
                q,
                r,
                s,
                coeff,
                dtype=dtype,
                cutoff=cutoff,
            )
    we_terms = _emit_shared_spinfree_we_terms(
        autompo,
        pending_we,
        phys_leg=phys_leg,
        dtype=dtype,
        cutoff=cutoff,
        family=family,
    )
    scalar_terms = _emit_shared_scalar_product_terms(
        autompo,
        pending_scalar,
        cutoff=cutoff,
        family=family,
    )
    info = {
        "raw_spin_component_terms": int(raw_terms),
        "we_product_terms": int(we_terms),
        "symbolic_product_terms": int(scalar_terms),
        "total_product_terms": int(we_terms + scalar_terms),
        "input_generator_terms": int(input_terms),
        "four_distinct_generator_terms": int(four_distinct_entries),
        "repeated_generator_terms": int(repeated_entries),
        "unique_index_histogram": {
            str(key): int(value)
            for key, value in sorted(repeat_histogram.items())
        },
    }
    return info if return_info else info["total_product_terms"]


def build_spatial_spinfree_eri_mpo(sites_or_legs, eri_spatial, *, cutoff=1.0e-12, include_half=True):
    """
    Build a general spin-free scalar-coupled ERI MPO for spatial orbitals.
    """
    builder = SpatialSpinFreeERIBuilder(
        sites_or_legs,
        eri_spatial,
        cutoff=cutoff,
        include_half=include_half,
    )
    return builder.build()


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
    if any(phys_leg != autompo.site_legs[0] for phys_leg in autompo.site_legs):
        raise ValueError(
            "add_spatial_density_terms expects a uniform spatial-orbital Leg across all sites."
        )
    phys_leg = autompo.site_legs[0]
    physical_leg_from_spatial_orbital(phys_leg)

    for site, _site_leg in enumerate(autompo.site_legs):
        if chemical_potential:
            autompo.add_onsite(site, spatial_number(phys_leg, dtype=float), coeff=-chemical_potential)
        if onsite_u:
            autompo.add_onsite(site, spatial_double_occupancy(phys_leg, dtype=float), coeff=onsite_u)

    if nearest_neighbor_v:
        density_ops = [spatial_number(phys_leg, dtype=float) for _ in autompo.site_legs]
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
