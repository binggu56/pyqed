#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State/MPS builders for the fixed-layout non-Abelian prototype.
"""

from __future__ import annotations

import numpy as np

from pyqed.mps.su2 import (
    SpinChargeSector,
    SpatialOrbitalSite,
    SU2Irrep,
    fuse_charge_spin_sectors,
)

from .coupling import clebsch_gordan, ordered_two_m_values
from .tensor import NonabelianTensor


class FullyReducedSpatialOrbitalSite(SpatialOrbitalSite):
    """
    Multiplicity-only SU(2) spatial-orbital site descriptor.

    ``SpatialOrbitalSite`` stores the singly occupied local multiplet with its
    explicit two-dimensional ``m`` basis. That is convenient for dense
    spin-resolved MPOs, but it is not the fully reduced Wigner-Eckart
    convention used by block2-style SU(2) DMRG. This descriptor keeps one
    reduced basis vector per local irrep:

    - N=0, S=0
    - N=1, S=1/2
    - N=2, S=0

    Site operators used with this layout must be reduced tensor operators; a
    dense spin-resolved physical MPO is intentionally incompatible with it.
    """

    labels = ("empty", "single", "double")
    state_index = ((0,), (0,), (0,))

    @property
    def d(self) -> int:
        return 3

    @property
    def degeneracy(self) -> tuple[int, ...]:
        return (1, 1, 1)


def build_reduced_product_spatial_mps(labels):
    """
    Build a product MPS in the fully reduced spatial-orbital convention.

    Only ``"empty"``, ``"single"``, ``"double"``, and ``"full"`` are accepted:
    spin-projection labels such as ``"up"``/``"down"`` are not states in a
    Wigner-Eckart reduced local basis.
    """
    if not labels:
        raise ValueError("build_reduced_product_spatial_mps requires at least one label.")
    site = FullyReducedSpatialOrbitalSite()
    phys_by_label = {
        "empty": site.qn[0],
        "single": site.qn[1],
        "double": site.qn[2],
        "full": site.qn[2],
    }
    vacuum = spatial_target_sector(0, 0)
    left = vacuum
    parsed = []
    for raw_label in labels:
        label = str(raw_label).lower()
        if label not in phys_by_label:
            raise ValueError(
                "Fully reduced product MPS labels must be empty/single/double/full; "
                f"got {raw_label!r}."
            )
        q_phys = phys_by_label[label]
        fused = _fuse_spatial_sectors(left, q_phys)
        if len(fused) != 1:
            raise ValueError(
                f"Label sequence {labels!r} is not a unique reduced product path at {raw_label!r}."
            )
        right = fused[0]
        parsed.append((q_phys, left, right))
        left = right

    tensors = []
    for q_phys, q_left, q_right in parsed:
        block = np.ones((1, 1, 1), dtype=float)
        tensors.append(
            NonabelianTensor(
                data={(q_left, q_phys, q_right): block},
                qns=[[q_left], list(site.qn), [q_right]],
                dirs=[-1, 1, 1],
                metadata={"physical_basis": "fully_reduced_su2"},
            )
        )
    return tensors


def spatial_target_sector(charge, two_j=0):
    """
    Build a total ``charge x SU(2)`` target sector for a spatial-orbital chain.
    """
    return SpinChargeSector(int(charge), SU2Irrep(int(two_j)))


def half_filled_singlet_sector(nsites):
    """
    Canonical half-filled singlet target sector for an even number of sites.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("half_filled_singlet_sector requires at least one site.")
    if nsites % 2 != 0:
        raise ValueError("half_filled_singlet_sector requires an even number of sites.")
    return spatial_target_sector(nsites, 0)


def build_product_spatial_mps(
    labels,
    *,
    enrich_bond_sectors=True,
    bond_multiplicity=2,
    zero_block_noise_scale=0.0,
    zero_block_noise_seed=None,
):
    """
    Build an exact product-state spatial-orbital MPS from human-readable labels.

    Supported labels are ``"empty"``, ``"up"``, ``"down"``, ``"double"``,
    and ``"full"`` (an alias of ``"double"``). The resulting chain is an
    exact bond-dimension-1 MPS in the cumulative ``charge x SU(2)`` sectors
    implied by the local occupations.

    When ``enrich_bond_sectors=True`` (the default), the tensors keep the same
    exact product-state amplitudes but expose the full target-compatible bond
    sector skeleton, similar to :func:`build_random_spatial_mps`. This gives
    subsequent DMRG sweeps room to grow the virtual sector support away from
    the initial product path.

    ``zero_block_noise_scale`` can be used to seed the symmetry-allowed
    zero-valued blocks with tiny deterministic noise. This leaves the intended
    product-state amplitude dominant, while giving DMRG a mixer-like way to
    escape product-state local minima on larger systems.
    """
    if not labels:
        raise ValueError("build_product_spatial_mps requires at least one site label.")
    bond_multiplicity = int(bond_multiplicity)
    if bond_multiplicity < 1:
        raise ValueError("bond_multiplicity must be a positive integer.")
    zero_block_noise_scale = float(zero_block_noise_scale)
    if zero_block_noise_scale < 0.0:
        raise ValueError("zero_block_noise_scale must be non-negative.")
    rng = np.random.default_rng(zero_block_noise_seed)

    site = SpatialOrbitalSite()
    phys_by_label = {
        "empty": site.qn[0],
        "up": site.qn[1],
        "down": site.qn[1],
        "double": site.qn[2],
        "full": site.qn[2],
    }
    state_index_by_label = {
        "empty": 0,
        "up": 0,
        "down": 1,
        "double": 0,
        "full": 0,
    }

    left = spatial_target_sector(0, 0)
    parsed = []
    path_sectors = [left]
    for raw_label in labels:
        label = str(raw_label).lower()
        if label not in phys_by_label:
            supported = ", ".join(sorted(phys_by_label))
            raise ValueError(
                f"Unsupported spatial product label {raw_label!r}. Supported labels: {supported}."
            )
        q_phys = phys_by_label[label]
        right = fuse_charge_spin_sectors(left, q_phys)[0]
        parsed.append((label, q_phys, left, right))
        path_sectors.append(right)
        left = right

    if enrich_bond_sectors:
        phys_sectors = tuple(site.qn)
        vacuum = phys_sectors[0]
        forward = _forward_reachable(phys_sectors, len(labels), vacuum)
        backward = _backward_reachable(phys_sectors, forward, path_sectors[-1])
        bond_sector_sets = [
            tuple(sorted(forward[i].intersection(backward[i])))
            for i in range(len(labels) + 1)
        ]
    else:
        bond_sector_sets = [tuple([sector]) for sector in path_sectors]

    tensors = []
    for site_index, (label, q_phys, q_left_path, q_right_path) in enumerate(parsed):
        if site_index == 0:
            left_qns = [q_left_path]
        else:
            left_qns = [
                sector
                for sector in bond_sector_sets[site_index]
                for _ in range(bond_multiplicity if enrich_bond_sectors else 1)
            ]
        if site_index == len(parsed) - 1:
            right_qns = [q_right_path]
        else:
            right_qns = [
                sector
                for sector in bond_sector_sets[site_index + 1]
                for _ in range(bond_multiplicity if enrich_bond_sectors else 1)
            ]

        data = {}
        for q_left in set(left_qns):
            d_left = _sector_multiplicity(left_qns, q_left)
            for q_site in site.qn:
                d_phys = len(site.state_index[site.qn.index(q_site)])
                fused = _fuse_spatial_sectors(q_left, q_site)
                for q_right in set(right_qns):
                    if q_right not in fused:
                        continue
                    d_right = _sector_multiplicity(right_qns, q_right)
                    block = np.zeros((d_left, d_phys, d_right), dtype=float)
                    if zero_block_noise_scale > 0.0:
                        block += rng.normal(scale=zero_block_noise_scale, size=block.shape)
                    data[(q_left, q_site, q_right)] = block

        block = data[(q_left_path, q_phys, q_right_path)]
        block[0, state_index_by_label[label], 0] = 1.0
        tensors.append(
            NonabelianTensor(
                data=data,
                qns=[left_qns, list(site.qn), right_qns],
                dirs=[-1, 1, 1],
            )
        )

    return tensors


def build_product_state(labels):
    """
    Build an exact spin-projected spatial-orbital product state.

    Unlike :func:`build_product_spatial_mps`, this builder treats ``"up"`` and
    ``"down"`` as uncoupled spin-projection labels, not as a single fixed
    cumulative SU(2) coupling path. The resulting MPS therefore keeps all
    coupled-basis bond states needed to represent the specified spin product
    exactly in the current reduced tensor model.

    Notes
    -----
    The internal bond legs use repeated copies of the same charge-spin sector
    to represent distinct coupled-basis states ``(channel, M)``. This keeps the
    initializer exact for spin-projected inputs without changing the existing
    contraction engine.
    """
    if not labels:
        raise ValueError("build_product_state requires at least one site label.")

    site = SpatialOrbitalSite()
    label_specs = {
        "empty": (site.qn[0], 0, 0),
        "up": (site.qn[1], 0, +1),
        "down": (site.qn[1], 1, -1),
        "double": (site.qn[2], 0, 0),
        "full": (site.qn[2], 0, 0),
    }

    def _basis_slots(basis_states):
        counts = {}
        slots = []
        for sector, _history, _two_m in basis_states:
            slot = counts.get(sector, 0)
            counts[sector] = slot + 1
            slots.append(slot)
        return slots

    def _backward_reachable_any_target(phys_sectors, forward, targets):
        nsites = len(forward) - 1
        reachable = [set() for _ in range(nsites + 1)]
        reachable[-1].update(targets)
        for site_index in range(nsites - 1, -1, -1):
            for left in forward[site_index]:
                for phys in phys_sectors:
                    fused = _fuse_spatial_sectors(left, phys)
                    if any(right in reachable[site_index + 1] for right in fused):
                        reachable[site_index].add(left)
                        break
        return reachable

    vacuum = spatial_target_sector(0, 0)
    boundary_basis = [[(vacuum, (), 0)]]

    for raw_label in labels:
        label = str(raw_label).lower()
        if label not in label_specs:
            supported = ", ".join(sorted(label_specs))
            raise ValueError(
                f"Unsupported spatial spin-product label {raw_label!r}. Supported labels: {supported}."
            )
        q_phys, phys_idx, two_m_phys = label_specs[label]

        right_basis_map = {}
        transitions = []
        for left_idx, (q_left, history, two_m_left) in enumerate(boundary_basis[-1]):
            for q_right in _fuse_spatial_sectors(q_left, q_phys):
                for two_m_right in ordered_two_m_values(q_right.irrep):
                    coeff = clebsch_gordan(
                        q_left.irrep,
                        q_phys.irrep,
                        q_right.irrep,
                        two_m_left,
                        two_m_phys,
                        two_m_right,
                    )
                    if abs(coeff) <= 1.0e-14:
                        continue
                    key = (q_right, history + (q_right,), two_m_right)
                    right_idx = right_basis_map.setdefault(key, len(right_basis_map))
                    transitions.append((left_idx, right_idx, float(coeff)))

        if not right_basis_map:
            raise ValueError(
                f"Spin label sequence {labels!r} is incompatible with local sector {q_phys!r}."
            )

        right_basis = [None] * len(right_basis_map)
        for key, idx in right_basis_map.items():
            right_basis[idx] = key
        boundary_basis.append(right_basis)

    final_targets = {sector for sector, _history, _two_m in boundary_basis[-1]}
    phys_sectors = tuple(site.qn)
    forward = _forward_reachable(phys_sectors, len(labels), vacuum)
    backward = _backward_reachable_any_target(phys_sectors, forward, final_targets)
    bond_sector_sets = [
        tuple(sorted(forward[i].intersection(backward[i])))
        for i in range(len(labels) + 1)
    ]

    tensors = []
    for site_index, raw_label in enumerate(labels):
        label = str(raw_label).lower()
        q_phys, phys_idx, _two_m_phys = label_specs[label]
        basis_states = boundary_basis[site_index]
        right_basis = boundary_basis[site_index + 1]

        left_qns = [sector for sector, _history, _two_m in basis_states]
        for sector in bond_sector_sets[site_index]:
            if sector not in left_qns:
                left_qns.append(sector)
        right_qns = [sector for sector, _history, _two_m in right_basis]
        for sector in bond_sector_sets[site_index + 1]:
            if sector not in right_qns:
                right_qns.append(sector)
        left_slots = _basis_slots(basis_states)
        right_slots = _basis_slots(right_basis)

        data = {}
        dtype = float
        for q_left in set(left_qns):
            d_left = _sector_multiplicity(left_qns, q_left)
            for q_site in site.qn:
                d_phys = len(site.state_index[site.qn.index(q_site)])
                fused = _fuse_spatial_sectors(q_left, q_site)
                for q_right in set(right_qns):
                    if q_right not in fused:
                        continue
                    d_right = _sector_multiplicity(right_qns, q_right)
                    data[(q_left, q_site, q_right)] = np.zeros(
                        (d_left, d_phys, d_right),
                        dtype=dtype,
                    )

        transitions = []
        right_basis_index = {state: idx for idx, state in enumerate(right_basis)}
        for left_idx, (q_left, history, two_m_left) in enumerate(basis_states):
            for q_right in _fuse_spatial_sectors(q_left, q_phys):
                for two_m_right in ordered_two_m_values(q_right.irrep):
                    coeff = clebsch_gordan(
                        q_left.irrep,
                        q_phys.irrep,
                        q_right.irrep,
                        two_m_left,
                        label_specs[label][2],
                        two_m_right,
                    )
                    if abs(coeff) <= 1.0e-14:
                        continue
                    right_idx = right_basis_index[(q_right, history + (q_right,), two_m_right)]
                    transitions.append((left_idx, right_idx, float(coeff)))

        for left_idx, right_idx, coeff in transitions:
            q_left = left_qns[left_idx]
            q_right = right_qns[right_idx]
            data[(q_left, q_phys, q_right)][
                left_slots[left_idx],
                phys_idx,
                right_slots[right_idx],
            ] = coeff

        tensors.append(
            NonabelianTensor(
                data=data,
                qns=[left_qns, list(site.qn), right_qns],
                dirs=[-1, 1, 1],
            )
        )

    return tensors


def build_spin_spatial_mps(labels):
    """
    Compatibility alias for :func:`build_product_state`.
    """
    return build_product_state(labels)


def _sector_multiplicity(qns, sector):
    return sum(1 for item in qns if item == sector)


def _fuse_spatial_sectors(left, right):
    if isinstance(left, SpinChargeSector) and isinstance(right, SpinChargeSector):
        return tuple(sorted(set(fuse_charge_spin_sectors(left, right))))
    raise TypeError(
        f"_fuse_spatial_sectors expects SpinChargeSector inputs, got {type(left).__name__} and {type(right).__name__}."
    )


def _forward_reachable(phys_sectors, nsites, vacuum):
    reachable = [set() for _ in range(nsites + 1)]
    reachable[0].add(vacuum)
    for site in range(nsites):
        for left in reachable[site]:
            for phys in phys_sectors:
                reachable[site + 1].update(_fuse_spatial_sectors(left, phys))
    return reachable


def _backward_reachable(phys_sectors, forward, target):
    nsites = len(forward) - 1
    reachable = [set() for _ in range(nsites + 1)]
    reachable[-1].add(target)
    for site in range(nsites - 1, -1, -1):
        for left in forward[site]:
            for phys in phys_sectors:
                fused = _fuse_spatial_sectors(left, phys)
                if any(right in reachable[site + 1] for right in fused):
                    reachable[site].add(left)
                    break
    return reachable


def build_random_spatial_mps(
    nsites,
    *,
    target_sector=None,
    bond_multiplicity=2,
    seed=None,
    scale=1.0,
    dtype=float,
):
    """
    Build a random spatial-orbital MPS with cumulative symmetry-resolved bond sectors.

    Parameters
    ----------
    nsites
        Number of spatial-orbital sites.
    target_sector
        Total charge-spin sector on the right boundary. Defaults to the
        half-filled singlet for even ``nsites`` and the nearest odd-particle
        doublet for odd ``nsites``.
    bond_multiplicity
        Initial multiplicity copies for each intermediate bond sector.
    seed
        Optional NumPy RNG seed.
    """
    nsites = int(nsites)
    if nsites < 2:
        raise ValueError("build_random_spatial_mps requires at least two sites.")
    bond_multiplicity = int(bond_multiplicity)
    if bond_multiplicity < 1:
        raise ValueError("bond_multiplicity must be a positive integer.")

    site = SpatialOrbitalSite()
    phys_sectors = tuple(site.qn)
    phys_dims = {sector: len(indices) for sector, indices in zip(site.qn, site.state_index)}
    vacuum = phys_sectors[0]

    if target_sector is None:
        if nsites % 2 == 0:
            target_sector = half_filled_singlet_sector(nsites)
        else:
            target_sector = spatial_target_sector(nsites, 1)

    forward = _forward_reachable(phys_sectors, nsites, vacuum)
    backward = _backward_reachable(phys_sectors, forward, target_sector)
    bond_sector_sets = [
        tuple(sorted(forward[i].intersection(backward[i])))
        for i in range(nsites + 1)
    ]
    if bond_sector_sets[0] != (vacuum,):
        raise ValueError("Failed to anchor the left boundary to the vacuum sector.")
    if bond_sector_sets[-1] != (target_sector,):
        raise ValueError(
            f"Target sector {target_sector!r} is not reachable with {nsites} spatial sites."
        )

    rng = np.random.default_rng(seed)
    tensors = []
    for site_index in range(nsites):
        left_sectors = bond_sector_sets[site_index]
        right_sectors = bond_sector_sets[site_index + 1]

        if site_index == 0:
            left_qns = [left_sectors[0]]
        else:
            left_qns = [sector for sector in left_sectors for _ in range(bond_multiplicity)]

        if site_index == nsites - 1:
            right_qns = [right_sectors[0]]
        else:
            right_qns = [sector for sector in right_sectors for _ in range(bond_multiplicity)]

        data = {}
        for q_left in left_sectors:
            d_left = _sector_multiplicity(left_qns, q_left)
            for q_phys in phys_sectors:
                fused = _fuse_spatial_sectors(q_left, q_phys)
                for q_right in right_sectors:
                    if q_right not in fused:
                        continue
                    d_right = _sector_multiplicity(right_qns, q_right)
                    d_phys = phys_dims[q_phys]
                    block = rng.normal(scale=scale, size=(d_left, d_phys, d_right)).astype(dtype)
                    data[(q_left, q_phys, q_right)] = block

        tensors.append(
            NonabelianTensor(
                data=data,
                qns=[left_qns, list(phys_sectors), right_qns],
                dirs=[-1, 1, 1],
            )
        )

    return tensors


def build_random_reduced_spatial_mps(
    nsites,
    *,
    target_sector=None,
    bond_multiplicity=2,
    seed=None,
    scale=1.0,
    dtype=float,
):
    """
    Build a random MPS in the fully reduced spatial-orbital convention.

    This is the multiplicity-only analogue of :func:`build_random_spatial_mps`:
    the physical ``N=1, S=1/2`` site irrep has local dimension one rather than
    explicit ``m=+/-1/2`` components.  It is the local basis needed for a
    block2-style Wigner-Eckart SU(2) sweep path.

    :param nsites: Number of spatial-orbital sites.
    :param target_sector: Total charge-spin sector on the right boundary.
    :param bond_multiplicity: Initial multiplicity copies per reachable
        intermediate sector.
    :param seed: Optional random seed.
    :param scale: Gaussian scale for initialized reduced blocks.
    :param dtype: Block dtype.
    :returns: List of rank-3 fully reduced MPS site tensors.
    """

    nsites = int(nsites)
    if nsites < 2:
        raise ValueError("build_random_reduced_spatial_mps requires at least two sites.")
    bond_multiplicity = int(bond_multiplicity)
    if bond_multiplicity < 1:
        raise ValueError("bond_multiplicity must be a positive integer.")

    site = FullyReducedSpatialOrbitalSite()
    phys_sectors = tuple(site.qn)
    phys_dims = {sector: 1 for sector in phys_sectors}
    vacuum = phys_sectors[0]

    if target_sector is None:
        if nsites % 2 == 0:
            target_sector = half_filled_singlet_sector(nsites)
        else:
            target_sector = spatial_target_sector(nsites, 1)

    forward = _forward_reachable(phys_sectors, nsites, vacuum)
    backward = _backward_reachable(phys_sectors, forward, target_sector)
    bond_sector_sets = [
        tuple(sorted(forward[i].intersection(backward[i])))
        for i in range(nsites + 1)
    ]
    if bond_sector_sets[0] != (vacuum,):
        raise ValueError("Failed to anchor the left boundary to the vacuum sector.")
    if bond_sector_sets[-1] != (target_sector,):
        raise ValueError(
            f"Target sector {target_sector!r} is not reachable with {nsites} reduced spatial sites."
        )

    rng = np.random.default_rng(seed)
    tensors = []
    for site_index in range(nsites):
        left_sectors = bond_sector_sets[site_index]
        right_sectors = bond_sector_sets[site_index + 1]
        left_qns = (
            [left_sectors[0]]
            if site_index == 0
            else [sector for sector in left_sectors for _ in range(bond_multiplicity)]
        )
        right_qns = (
            [right_sectors[0]]
            if site_index == nsites - 1
            else [sector for sector in right_sectors for _ in range(bond_multiplicity)]
        )

        data = {}
        for q_left in left_sectors:
            d_left = _sector_multiplicity(left_qns, q_left)
            for q_phys in phys_sectors:
                fused = _fuse_spatial_sectors(q_left, q_phys)
                for q_right in right_sectors:
                    if q_right not in fused:
                        continue
                    d_right = _sector_multiplicity(right_qns, q_right)
                    d_phys = phys_dims[q_phys]
                    data[(q_left, q_phys, q_right)] = rng.normal(
                        scale=scale,
                        size=(d_left, d_phys, d_right),
                    ).astype(dtype)

        tensors.append(
            NonabelianTensor(
                data=data,
                qns=[left_qns, list(phys_sectors), right_qns],
                dirs=[-1, 1, 1],
                metadata={"physical_basis": "fully_reduced_su2"},
            )
        )

    return tensors
