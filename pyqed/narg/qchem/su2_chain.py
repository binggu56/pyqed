#!/usr/bin/env python3
"""Reusable direct-reduced SU(2)-NARG chain growth utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
import time

import numpy as np

from pyqed.narg.irrep_tensor import Irrep, IrrepSite, IrrepTensor, OpIrrep
from .su2_three_site import (
    CDD,
    CDU,
    CD,
    CU,
    ThreeSiteSU2NARG,
    JW,
    block_density_tensor,
    block_spin_density_tensor,
    block_retained_scalar_tensor,
    branch_irrep_site,
    coupled_product_states,
    direct_reduced_full_hamiltonian_tensor,
    expanded_operator_from_reduced,
    grow_su2_block_by_one_site,
    local_reduced_operator,
    profile_function,
    profile_section,
    reduced_product_tensor_irrep,
    reset_su2_profile,
    rotate_reduced_tensors_to_truncated,
    su2_profile_snapshot,
)
from .su2_two_site import (
    AdaptiveD,
    RenormalizedSU2Block,
    SectorRoot,
    TruncatedSU2NARG,
    build_renormalized_two_site_block,
    retained_multiplets,
    truncate_to_D,
)
from .su2_reduced_tensor import (
    ReducedSU2Tensor,
    add_reduced_tensors,
    coupled_reduced_product,
    coupled_reduced_products,
    reduced_tensor_from_components,
    scale_reduced_tensor,
)
from .su2_core import asarray, full_jw_model
from .su2_core import Multiplet, cg, local_site_multiplets
from .su2_backend import resolve_su2_narg_backend


@dataclass
class SU2ChainResult:
    """Result of a direct-reduced SU(2)-NARG chain growth."""

    final: ThreeSiteSU2NARG
    blocks: dict[int, RenormalizedSU2Block]
    timings: dict[str, object] = field(default_factory=dict)
    backend: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _ComponentState:
    """One abstract magnetic component in a retained multiplet basis."""

    irrep: Irrep
    root_index: int
    local_index: int
    m2: int
    energy: float
    vector: np.ndarray


@dataclass(frozen=True)
class LowRankERI:
    """Pair-factorized ERI tensor ``eri[p,q,r,s] = sum_L w_L M_L[p,q] M_L[r,s]``."""

    weights: np.ndarray
    modes: np.ndarray

    @classmethod
    def from_dense(cls, eri, *, tol: float = 1.0e-10, max_rank: int | None = None):
        eri = np.asarray(eri)
        n = int(eri.shape[0])
        pair_matrix = np.asarray(eri.reshape(n * n, n * n), dtype=float)
        pair_matrix = 0.5 * (pair_matrix + pair_matrix.T)
        values, vectors = np.linalg.eigh(pair_matrix)
        order = np.argsort(np.abs(values))[::-1]
        values = values[order]
        vectors = vectors[:, order]
        keep = np.abs(values) > float(tol)
        if max_rank is not None:
            selected = np.flatnonzero(keep)[: int(max_rank)]
        else:
            selected = np.flatnonzero(keep)
        if selected.size == 0:
            return cls(
                weights=np.zeros(0, dtype=float),
                modes=np.zeros((0, n, n), dtype=float),
            )
        return cls(
            weights=np.ascontiguousarray(values[selected], dtype=float),
            modes=np.ascontiguousarray(vectors[:, selected].T.reshape(selected.size, n, n), dtype=float),
        )

    @property
    def rank(self) -> int:
        return int(self.weights.size)


@lru_cache(maxsize=None)
def _local_v1_reduced_operator(name: str) -> ReducedSU2Tensor:
    """One-site reduced tensors needed by the exact V1 growth recurrence."""
    local_density = CDU @ CU + CDD @ CD
    multiplets = local_site_multiplets()
    if name == "CdagJWCtildeDensity":
        return reduced_tensor_from_components(
            multiplets,
            {0: CDU @ JW @ CU + CDD @ JW @ CD},
            OpIrrep((0, 0)),
        )
    if name == "CdagJWCtildeSpinDensity":
        return reduced_tensor_from_components(
            multiplets,
            {
                -2: CDD @ JW @ CU,
                0: (CDU @ JW @ CU - CDD @ JW @ CD) / np.sqrt(2.0),
                2: -(CDU @ JW @ CD),
            },
            OpIrrep((0, 2)),
        )
    if name == "JWDensity":
        return reduced_tensor_from_components(
            multiplets,
            {0: JW @ local_density},
            OpIrrep((0, 0)),
        )
    if name == "CdagDensity":
        return reduced_tensor_from_components(
            multiplets,
            {1: CDU @ local_density, -1: CDD @ local_density},
            OpIrrep((1, 1)),
        )
    raise KeyError(f"unknown local V1 reduced operator {name!r}")


def block_dim(narg: ThreeSiteSU2NARG) -> int:
    """Total multiplicity-space dimension of an IrrepTensor Hamiltonian."""
    return int(sum(narg.site.dims.values()))


def block_identity_reduced_tensor(block: RenormalizedSU2Block):
    """Identity as a scalar reduced tensor on a retained block."""
    return block_retained_scalar_tensor(
        block,
        {
            irrep: np.eye(block.truncated.site.sector_dim(irrep), dtype=complex)
            for irrep in block.truncated.site.irreps
        },
    )


def retained_component_multiplets(
    block: RenormalizedSU2Block,
) -> tuple[list[_ComponentState], list[Multiplet]]:
    """Represent retained multiplets in an abstract magnetic-component basis."""
    states: list[_ComponentState] = []
    multiplets: list[Multiplet] = []
    local_indices: dict[Irrep, int] = {}

    for root_index, root in enumerate(block.truncated.kept_roots):
        irrep = root.irrep
        nelec, j2 = irrep.charge
        local_index = local_indices.get(irrep, 0)
        local_indices[irrep] = local_index + 1
        components = {}
        for m2 in range(-j2, j2 + 1, 2):
            pos = len(states)
            vec = np.zeros(pos + 1, dtype=complex)
            vec[pos] = 1.0
            states.append(
                _ComponentState(
                    irrep=irrep,
                    root_index=root_index,
                    local_index=local_index,
                    m2=m2,
                    energy=root.energy,
                    vector=vec,
                )
            )
            components[m2] = vec
        multiplets.append(Multiplet(nelec=nelec, j2=j2, states=components))

    dim = len(states)
    if dim:
        for mp in multiplets:
            mp.states.update({m2: _pad_vector(vec, dim) for m2, vec in mp.states.items()})
    return states, multiplets


def _pad_vector(vec: np.ndarray, dim: int) -> np.ndarray:
    """Pad an abstract basis vector to the final component dimension."""
    if vec.size == dim:
        return vec
    out = np.zeros(int(dim), dtype=complex)
    out[: vec.size] = vec
    return out


def block_component_fermion_ops(
    block: RenormalizedSU2Block,
    site_count: int,
) -> tuple[list[Multiplet], dict[str, list[np.ndarray]]]:
    """Reconstruct current-site fermion components in retained component space."""
    states, multiplets = retained_component_multiplets(block)
    dim = len(states)
    ops = {name: [] for name in ("Cdu", "Cdd", "Cu", "Cd")}
    for site_index in range(int(site_count)):
        cdag = block.reduced_operators[("Cdag", site_index)]
        ctilde = block.reduced_operators[("Ctilde", site_index)]
        ops["Cdu"].append(expanded_operator_from_reduced(states, cdag, q2=1))
        ops["Cdd"].append(expanded_operator_from_reduced(states, cdag, q2=-1))
        ops["Cu"].append(expanded_operator_from_reduced(states, ctilde, q2=-1))
        ops["Cd"].append(-expanded_operator_from_reduced(states, ctilde, q2=1))
    if not ops["Cdu"] and dim:
        zero = np.zeros((dim, dim), dtype=complex)
        for name in ops:
            ops[name].append(zero)
    return multiplets, ops


def grown_component_multiplets_and_ops(
    source_block: RenormalizedSU2Block,
) -> tuple[list[Multiplet], dict[str, list[np.ndarray]]]:
    """Build grown one-site fermion components without determinant-space vectors."""
    block_states, _ = retained_component_multiplets(source_block)
    block_dim = len(block_states)
    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    if not old_site_indices:
        raise ValueError("source block does not carry reduced Cdag/Ctilde operators")
    old_site_count = old_site_indices[-1] + 1
    site_count = old_site_count + 1
    _, block_ops = block_component_fermion_ops(source_block, old_site_count)

    ib = np.eye(block_dim, dtype=complex)
    ops = {name: [] for name in ("Cdu", "Cdd", "Cu", "Cd")}
    for site_index in range(old_site_count):
        ops["Cdu"].append(np.kron(block_ops["Cdu"][site_index], JW))
        ops["Cdd"].append(np.kron(block_ops["Cdd"][site_index], JW))
        ops["Cu"].append(np.kron(block_ops["Cu"][site_index], JW))
        ops["Cd"].append(np.kron(block_ops["Cd"][site_index], JW))
    ops["Cdu"].append(np.kron(ib, CDU))
    ops["Cdd"].append(np.kron(ib, CDD))
    ops["Cu"].append(np.kron(ib, CU))
    ops["Cd"].append(np.kron(ib, CD))

    block_pos = {
        (state.irrep, state.local_index, state.m2): pos
        for pos, state in enumerate(block_states)
    }
    local_vectors = {}
    local_indices: dict[Irrep, int] = {}
    for mp in local_site_multiplets():
        irrep = Irrep((mp.nelec, mp.j2))
        local_index = local_indices.get(irrep, 0)
        local_indices[irrep] = local_index + 1
        for m2, vec in mp.states.items():
            local_vectors[(irrep, local_index, m2)] = vec

    dim = block_dim * 4
    multiplets = []
    for state in coupled_product_states(source_block.truncated):
        allowed = getattr(source_block, "_su2_allowed_final_nelec", None)
        if allowed is not None and state.total_irrep.charge[0] not in allowed:
            continue
        total_nelec, total_j2 = state.total_irrep.charge
        block_j2 = state.block_irrep.charge[1]
        local_j2 = state.local_irrep.charge[1]
        components = {}
        for total_m2 in range(-total_j2, total_j2 + 1, 2):
            vec = np.zeros(dim, dtype=complex)
            for block_m2 in range(-block_j2, block_j2 + 1, 2):
                local_m2 = total_m2 - block_m2
                if local_m2 < -local_j2 or local_m2 > local_j2:
                    continue
                coeff = cg(block_j2, block_m2, local_j2, local_m2, total_j2, total_m2)
                if abs(coeff) <= 1.0e-14:
                    continue
                bpos = block_pos.get((state.block_irrep, state.block_local_index, block_m2))
                lvec = local_vectors.get((state.local_irrep, state.local_index, local_m2))
                if bpos is None or lvec is None:
                    continue
                vec[4 * bpos : 4 * bpos + 4] += coeff * lvec
            norm = np.linalg.norm(vec)
            if norm > 1.0e-12:
                components[total_m2] = vec / norm
        if components:
            multiplets.append(Multiplet(nelec=total_nelec, j2=total_j2, states=components))

    if site_count != len(ops["Cdu"]):
        raise RuntimeError("grown component operator site count mismatch")
    return multiplets, ops


def v1_spinor_packages_from_components(
    multiplets: list[Multiplet],
    ops: dict[str, list[np.ndarray]],
    h1e,
    eri,
    site_count: int,
    future_sites,
    *,
    only_site: int | None = None,
) -> dict[tuple, ReducedSU2Tensor]:
    """Build weighted V1 spinors in retained component space."""
    out = {}
    site_count = int(site_count)
    dim = ops["Cdu"][0].shape[0] if ops["Cdu"] else 0
    for q in future_sites:
        q = int(q)
        v1u = np.zeros((dim, dim), dtype=complex)
        v1d = np.zeros((dim, dim), dtype=complex)
        for i in range(site_count):
            if only_site is not None and i != int(only_site):
                continue
            coeff = h1e[i, q]
            if abs(coeff) > 0.0:
                v1u += coeff * ops["Cdu"][i]
                v1d += coeff * ops["Cdd"][i]

        for i in range(site_count):
            for j in range(site_count):
                residual = None
                for k in range(site_count):
                    if only_site is not None and int(only_site) not in (i, j, k):
                        continue
                    coeff = eri[k, q, j, i]
                    if abs(coeff) <= 0.0:
                        continue
                    if residual is None:
                        residual = ops["Cdu"][j] @ ops["Cu"][i] + ops["Cdd"][j] @ ops["Cd"][i]
                    v1u += coeff * (ops["Cdu"][k] @ residual)
                    v1d += coeff * (ops["Cdd"][k] @ residual)

        if np.any(np.abs(v1u) > 1.0e-14) or np.any(np.abs(v1d) > 1.0e-14):
            tensor = reduced_tensor_from_components(
                multiplets,
                {1: v1u, -1: v1d},
                OpIrrep((1, 1)),
            )
            if tensor.blocks:
                out[("NextV1Spinor", q)] = tensor
    return out


@profile_function("seed_component_v1_packages")
def seed_component_v1_packages(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_count: int,
    future_sites,
) -> dict[tuple, ReducedSU2Tensor]:
    """Build initial V1 packages from retained components, not primitive determinants."""
    multiplets, ops = block_component_fermion_ops(block, int(site_count))
    return v1_spinor_packages_from_components(
        multiplets,
        ops,
        h1e,
        eri,
        int(site_count),
        future_sites,
    )


@profile_function("grown_component_v1_packages")
def grown_component_v1_packages(
    source_block: RenormalizedSU2Block,
    h1e,
    eri,
    future_sites,
) -> dict[tuple, ReducedSU2Tensor]:
    """Build new-site V1 contributions in the grown retained-component basis."""
    multiplets, _ = grown_component_multiplets_and_ops(source_block)
    block_states, _ = retained_component_multiplets(source_block)
    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    if not old_site_indices:
        return {}
    old_site_count = old_site_indices[-1] + 1
    new_site = old_site_count
    block_dim = len(block_states)
    ib = np.eye(block_dim, dtype=complex)
    local_density = CDU @ CU + CDD @ CD
    local_cdu_cdd_jw = CDU @ CDD @ JW
    local_cdd_cdu_jw = CDD @ CDU @ JW
    local_cdu_jw_cu = CDU @ JW @ CU
    local_cdu_jw_cd = CDU @ JW @ CD
    local_cdd_jw_cu = CDD @ JW @ CU
    local_cdd_jw_cd = CDD @ JW @ CD
    local_jw_density = JW @ local_density
    _, singles = block_component_fermion_ops(source_block, old_site_count)

    hop_cache = {}
    pair_cache = {}

    def add_kron_scaled(target: np.ndarray, block: np.ndarray, local: np.ndarray, coeff) -> None:
        """Accumulate ``coeff * kron(block, local)`` into ``target`` in place."""
        if block.size == 0 or abs(coeff) <= 0.0:
            return
        local = np.asarray(local)
        rows, cols = np.nonzero(np.abs(local) > 0.0)
        for row, col in zip(rows, cols):
            value = coeff * local[row, col]
            if abs(value) > 0.0:
                target[row::4, col::4] += value * block

    def component(tensor: ReducedSU2Tensor, q2: int) -> np.ndarray:
        return expanded_operator_from_reduced(block_states, tensor, q2=q2)

    def optional_component(key: tuple, q2: int) -> np.ndarray:
        tensor = source_block.reduced_operators.get(key)
        if tensor is None:
            return np.zeros((block_dim, block_dim), dtype=complex)
        return component(tensor, q2)

    def density_matrix(j: int, i: int) -> np.ndarray:
        return optional_component(("Density", int(j), int(i)), 0)

    def hop_products(k: int, i: int) -> dict[str, np.ndarray]:
        key = (int(k), int(i))
        cached = hop_cache.get(key)
        if cached is not None:
            return cached
        density = optional_component(("Density", key[0], key[1]), 0)
        spin_m = optional_component(("SpinDensity", key[0], key[1]), -2)
        spin_0 = optional_component(("SpinDensity", key[0], key[1]), 0)
        spin_p = optional_component(("SpinDensity", key[0], key[1]), 2)
        cached = {
            "uu": 0.5 * (density + np.sqrt(2.0) * spin_0),
            "ud": -spin_p,
            "du": spin_m,
            "dd": 0.5 * (density - np.sqrt(2.0) * spin_0),
        }
        hop_cache[key] = cached
        return cached

    def pair_products(k: int, j: int) -> dict[str, np.ndarray]:
        key = (int(k), int(j))
        cached = pair_cache.get(key)
        if cached is not None:
            return cached
        t0 = optional_component(("PairCreate0", key[0], key[1]), 0)
        t20 = optional_component(("PairCreate2", key[0], key[1]), 0)
        t2p = optional_component(("PairCreate2", key[0], key[1]), 2)
        t2m = optional_component(("PairCreate2", key[0], key[1]), -2)
        a = cg(1, 1, 1, -1, 0, 0)
        b = cg(1, -1, 1, 1, 0, 0)
        c = cg(1, 1, 1, -1, 2, 0)
        d = cg(1, -1, 1, 1, 2, 0)
        det = a * d - b * c
        cached = {
            "uu": t2p / cg(1, 1, 1, 1, 2, 2),
            "ud": (d * t0 - b * t20) / det,
            "du": (-c * t0 + a * t20) / det,
            "dd": t2m / cg(1, -1, 1, -1, 2, -2),
        }
        pair_cache[key] = cached
        return cached

    out = {}
    dim = block_dim * 4
    for q in future_sites:
        q = int(q)
        v1u = np.zeros((dim, dim), dtype=complex)
        v1d = np.zeros((dim, dim), dtype=complex)

        coeff = h1e[new_site, q]
        if abs(coeff) > 0.0:
            add_kron_scaled(v1u, ib, CDU, coeff)
            add_kron_scaled(v1d, ib, CDD, coeff)

        for i in range(new_site + 1):
            for j in range(new_site + 1):
                for k in range(new_site + 1):
                    if new_site not in (i, j, k):
                        continue
                    coeff = eri[k, q, j, i]
                    if abs(coeff) <= 0.0:
                        continue

                    if k == new_site and j < new_site and i < new_site:
                        density = density_matrix(j, i)
                        add_kron_scaled(v1u, density, CDU, coeff)
                        add_kron_scaled(v1d, density, CDD, coeff)
                    elif k < new_site and j == new_site and i < new_site:
                        hop = hop_products(k, i)
                        add_kron_scaled(v1u, hop["uu"], CDU, -coeff)
                        add_kron_scaled(v1u, hop["ud"], CDD, -coeff)
                        add_kron_scaled(v1d, hop["du"], CDU, -coeff)
                        add_kron_scaled(v1d, hop["dd"], CDD, -coeff)
                    elif k < new_site and j < new_site and i == new_site:
                        pair = pair_products(k, j)
                        add_kron_scaled(v1u, pair["uu"], CU, coeff)
                        add_kron_scaled(v1u, pair["ud"], CD, coeff)
                        add_kron_scaled(v1d, pair["du"], CU, coeff)
                        add_kron_scaled(v1d, pair["dd"], CD, coeff)
                    elif k == new_site and j == new_site and i < new_site:
                        add_kron_scaled(v1u, singles["Cd"][i], local_cdu_cdd_jw, coeff)
                        add_kron_scaled(v1d, singles["Cu"][i], local_cdd_cdu_jw, coeff)
                    elif k == new_site and j < new_site and i == new_site:
                        add_kron_scaled(v1u, singles["Cdu"][j], local_cdu_jw_cu, coeff)
                        add_kron_scaled(v1u, singles["Cdd"][j], local_cdu_jw_cd, coeff)
                        add_kron_scaled(v1d, singles["Cdu"][j], local_cdd_jw_cu, coeff)
                        add_kron_scaled(v1d, singles["Cdd"][j], local_cdd_jw_cd, coeff)
                    elif k < new_site and j == new_site and i == new_site:
                        add_kron_scaled(v1u, singles["Cdu"][k], local_jw_density, coeff)
                        add_kron_scaled(v1d, singles["Cdd"][k], local_jw_density, coeff)
                    else:
                        add_kron_scaled(v1u, ib, CDU @ local_density, coeff)
                        add_kron_scaled(v1d, ib, CDD @ local_density, coeff)

        if np.any(np.abs(v1u) > 1.0e-14) or np.any(np.abs(v1d) > 1.0e-14):
            tensor = reduced_tensor_from_components(
                multiplets,
                {1: v1u, -1: v1d},
                OpIrrep((1, 1)),
            )
            if tensor.blocks:
                out[("NextV1Spinor", q)] = tensor
    return out


@profile_function("grown_reduced_v1_packages")
def grown_reduced_v1_packages(
    source_block: RenormalizedSU2Block,
    grown: dict[tuple[str, int], ReducedSU2Tensor],
    h1e,
    eri,
    future_sites,
) -> dict[tuple, ReducedSU2Tensor]:
    """Build new-site V1 contributions by exact reduced recurrences."""
    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    if not old_site_indices:
        return {}

    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    new_site = site_count - 1
    if old_site_indices[-1] + 1 != new_site:
        raise ValueError("grown V1 recurrence source/grown site counts disagree")

    block_identity = block_identity_reduced_tensor(source_block)
    local_cdag = local_reduced_operator("Cdag")
    local_ctilde = local_reduced_operator("Ctilde")
    local_pair_create = local_reduced_operator("PairCreate")
    local_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeDensity")
    local_spin_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeSpinDensity")
    local_jw_density = _local_v1_reduced_operator("JWDensity")
    local_cdag_density = _local_v1_reduced_operator("CdagDensity")

    identity_cdag = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag,
        total_rank2=1,
    )
    identity_cdag_density = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag_density,
        total_rank2=1,
    )

    product_cache: dict[tuple, ReducedSU2Tensor | None] = {}

    def product(key: tuple, tensor: ReducedSU2Tensor | None, local: ReducedSU2Tensor, rank2: int):
        if key in product_cache:
            return product_cache[key]
        if tensor is None:
            product_cache[key] = None
            return None
        out = reduced_product_tensor_irrep(
            source_block,
            tensor,
            local,
            total_rank2=rank2,
        )
        product_cache[key] = out
        return out

    def source_tensor(key: tuple) -> ReducedSU2Tensor | None:
        return source_block.reduced_operators.get(key)

    out = {}
    for q in future_sites:
        q = int(q)
        terms = []

        coeff = h1e[new_site, q]
        if abs(coeff) > 0.0:
            terms.append((identity_cdag, coeff))

        for i in range(site_count):
            for j in range(site_count):
                for k in range(site_count):
                    if new_site not in (i, j, k):
                        continue
                    coeff = eri[k, q, j, i]
                    if abs(coeff) <= 0.0:
                        continue

                    if k == new_site and j < new_site and i < new_site:
                        tensor = product(
                            ("DensityCdag", j, i),
                            source_tensor(("Density", j, i)),
                            local_cdag,
                            1,
                        )
                        if tensor is not None:
                            terms.append((tensor, coeff))
                    elif k < new_site and j == new_site and i < new_site:
                        density = product(
                            ("HopDensityCdag", k, i),
                            source_tensor(("Density", k, i)),
                            local_cdag,
                            1,
                        )
                        if density is not None:
                            terms.append((density, -0.5 * coeff))
                        spin_density = product(
                            ("HopSpinDensityCdag", k, i),
                            source_tensor(("SpinDensity", k, i)),
                            local_cdag,
                            1,
                        )
                        if spin_density is not None:
                            terms.append((spin_density, np.sqrt(1.5) * coeff))
                    elif k < new_site and j < new_site and i == new_site:
                        pair0 = product(
                            ("Pair0Ctilde", k, j),
                            source_tensor(("PairCreate0", k, j)),
                            local_ctilde,
                            1,
                        )
                        if pair0 is not None:
                            terms.append((pair0, -coeff / np.sqrt(2.0)))
                        pair2 = product(
                            ("Pair2Ctilde", k, j),
                            source_tensor(("PairCreate2", k, j)),
                            local_ctilde,
                            1,
                        )
                        if pair2 is not None:
                            terms.append((pair2, np.sqrt(1.5) * coeff))
                    elif k == new_site and j == new_site and i < new_site:
                        tensor = product(
                            ("CtildePairCreate", i),
                            source_tensor(("Ctilde", i)),
                            local_pair_create,
                            1,
                        )
                        if tensor is not None:
                            terms.append((tensor, -coeff))
                    elif k == new_site and j < new_site and i == new_site:
                        density = product(
                            ("CdagDensityMidJW", j),
                            source_tensor(("Cdag", j)),
                            local_density_mid_jw,
                            1,
                        )
                        if density is not None:
                            terms.append((density, 0.5 * coeff))
                        spin_density = product(
                            ("CdagSpinDensityMidJW", j),
                            source_tensor(("Cdag", j)),
                            local_spin_density_mid_jw,
                            1,
                        )
                        if spin_density is not None:
                            terms.append((spin_density, np.sqrt(1.5) * coeff))
                    elif k < new_site and j == new_site and i == new_site:
                        tensor = product(
                            ("CdagJWDensity", k),
                            source_tensor(("Cdag", k)),
                            local_jw_density,
                            1,
                        )
                        if tensor is not None:
                            terms.append((tensor, coeff))
                    else:
                        terms.append((identity_cdag_density, coeff))

        total = add_optional_reduced_terms(terms)
        if total is not None:
            out[("NextV1Spinor", q)] = total
    return out


@profile_function("grown_coupling_operators")
def grown_coupling_operators(
    narg: ThreeSiteSU2NARG,
    *,
    include_even_composites: bool = True,
    even_composites: set[str] | None = None,
) -> dict[tuple[str, int], ReducedSU2Tensor]:
    """Grow site spinors from the source block to the untruncated branch basis."""
    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    if source_block is None:
        raise ValueError("grown NARG object does not carry its source block")

    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    if not old_site_indices:
        raise ValueError("source block does not carry reduced Cdag/Ctilde operators")
    nsites = old_site_indices[-1] + 2
    new_site_index = nsites - 1

    grown = {}
    local_jw = local_reduced_operator("JW")
    for site_index in old_site_indices:
        grown[("Cdag", site_index)] = reduced_product_tensor_irrep(
            source_block,
            source_block.reduced_operators[("Cdag", site_index)],
            local_jw,
            total_rank2=1,
        )
        grown[("Ctilde", site_index)] = reduced_product_tensor_irrep(
            source_block,
            source_block.reduced_operators[("Ctilde", site_index)],
            local_jw,
            total_rank2=1,
        )

    if include_even_composites:
        if even_composites is None:
            even_composites = {
                "Density",
                "SpinDensity",
                "PairCreate0",
                "PairCreate2",
            }
        else:
            even_composites = set(even_composites)
        local_identity = local_reduced_operator("I")
        for key, tensor in source_block.reduced_operators.items():
            if not isinstance(key, tuple) or key[0] not in even_composites:
                continue
            grown[key] = reduced_product_tensor_irrep(
                source_block,
                tensor,
                local_identity,
                total_rank2=tensor.op.charge[1],
            )

    block_identity = block_identity_reduced_tensor(source_block)
    grown[("Cdag", new_site_index)] = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_reduced_operator("Cdag"),
        total_rank2=1,
    )
    grown[("Ctilde", new_site_index)] = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_reduced_operator("Ctilde"),
        total_rank2=1,
    )
    return grown


def reduced_density_tensor(operators: dict[tuple[str, int], ReducedSU2Tensor], i: int, j: int):
    """Reduced scalar ``sum_sigma c^dag_i,sigma c_j,sigma``."""
    return coupled_reduced_product(
        operators[("Cdag", i)],
        operators[("Ctilde", j)],
        rank2=0,
        scale=np.sqrt(2.0),
    )


def reduced_spin_density_tensor(operators: dict[tuple[str, int], ReducedSU2Tensor], i: int, j: int):
    """Reduced rank-1 spin density from ``Cdag_i x Ctilde_j``."""
    return coupled_reduced_product(operators[("Cdag", i)], operators[("Ctilde", j)], rank2=2)


def reduced_pair_annihilate_tensor(operators: dict[tuple[str, int], ReducedSU2Tensor], i: int, j: int):
    """Reduced singlet pair annihilation."""
    return coupled_reduced_product(
        operators[("Ctilde", i)],
        operators[("Ctilde", j)],
        rank2=0,
        scale=-1.0 / np.sqrt(2.0),
    )


def reduced_pair_create_tensor(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    i: int,
    j: int,
    rank2: int,
):
    """Reduced pair-creation tensor ``[c^dag_i x c^dag_j]^rank``."""
    return coupled_reduced_product(
        operators[("Cdag", i)],
        operators[("Cdag", j)],
        rank2=int(rank2),
    )


def reduced_cdag_density_tensor(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    density_cache: dict[tuple[int, int], ReducedSU2Tensor],
    k: int,
    j: int,
    i: int,
):
    """Reduced spinor ``c^dag_k sum_sigma c^dag_j,sigma c_i,sigma``."""
    direct = operators.get(("CdagDensity", int(k), int(j), int(i)))
    if direct is not None:
        return direct
    density = cached_density_tensor(operators, density_cache, j, i)
    return coupled_reduced_product(operators[("Cdag", k)], density, rank2=1)


def cached_density_tensor(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    cache: dict[tuple[int, int], ReducedSU2Tensor],
    i: int,
    j: int,
) -> ReducedSU2Tensor:
    """Return cached scalar density without evaluating the fallback eagerly."""
    key = (int(i), int(j))
    tensor = cache.get(key)
    if tensor is None:
        tensor = operators.get(("Density", int(i), int(j)))
        if tensor is None:
            tensor = reduced_density_tensor(operators, int(i), int(j))
        cache[key] = tensor
    return tensor


def complete_pair_composites(
    operators: dict[tuple, ReducedSU2Tensor],
    site_count: int,
    source_block: RenormalizedSU2Block | None = None,
) -> dict[tuple, ReducedSU2Tensor]:
    """Ensure exact pair composites are available for all current-site pairs."""
    out = dict(operators)
    new_site = int(site_count) - 1
    local_jw_ctilde = local_reduced_operator("JWCtilde")
    local_jw_cdag = local_reduced_operator("JWCdag")
    product_requests = []

    def direct_old_new_composite(name: str, i: int, j: int) -> ReducedSU2Tensor | None:
        if source_block is None:
            return None
        if name == "Density" and int(i) < new_site and int(j) < new_site:
            return grow_source_tensor(
                source_block,
                block_density_tensor(source_block, int(i), int(j)),
                local_name="I",
            )
        if name == "Density" and int(i) == new_site and int(j) == new_site:
            return reduced_product_tensor_irrep(
                source_block,
                block_identity_reduced_tensor(source_block),
                local_reduced_operator("Ntot"),
                total_rank2=0,
            )
        if name == "Density" and int(i) == new_site and int(j) < new_site:
            forward = direct_old_new_composite("Density", int(j), int(i))
            return ReducedSU2Tensor(forward.tensor.adjoint()) if forward is not None else None
        if int(j) != new_site or int(i) >= new_site:
            return None
        if name == "Density":
            return scale_reduced_tensor(
                reduced_product_tensor_irrep(
                    source_block,
                    source_block.reduced_operators[("Cdag", int(i))],
                    local_jw_ctilde,
                    total_rank2=0,
                ),
                np.sqrt(2.0),
            )
        if name == "SpinDensity":
            return reduced_product_tensor_irrep(
                source_block,
                source_block.reduced_operators[("Cdag", int(i))],
                local_jw_ctilde,
                total_rank2=2,
            )
        if name == "PairCreate0":
            return reduced_product_tensor_irrep(
                source_block,
                source_block.reduced_operators[("Cdag", int(i))],
                local_jw_cdag,
                total_rank2=0,
            )
        if name == "PairCreate2":
            return reduced_product_tensor_irrep(
                source_block,
                source_block.reduced_operators[("Cdag", int(i))],
                local_jw_cdag,
                total_rank2=2,
            )
        return None

    for i in range(int(site_count)):
        for j in range(int(site_count)):
            key = ("Density", i, j)
            if key not in out:
                tensor = direct_old_new_composite("Density", i, j)
                if tensor is not None:
                    out[key] = tensor
                else:
                    product_requests.append(
                        (
                            key,
                            out[("Cdag", i)],
                            out[("Ctilde", j)],
                            0,
                            np.sqrt(2.0),
                        )
                    )
            key = ("SpinDensity", i, j)
            if key not in out:
                tensor = direct_old_new_composite("SpinDensity", i, j)
                if tensor is not None:
                    out[key] = tensor
                else:
                    product_requests.append(
                        (key, out[("Cdag", i)], out[("Ctilde", j)], 2)
                    )
            key = ("PairCreate0", i, j)
            if key not in out:
                tensor = direct_old_new_composite("PairCreate0", i, j)
                if tensor is not None:
                    out[key] = tensor
                else:
                    product_requests.append(
                        (key, out[("Cdag", i)], out[("Cdag", j)], 0)
                    )
            key = ("PairCreate2", i, j)
            if key not in out:
                tensor = direct_old_new_composite("PairCreate2", i, j)
                if tensor is not None:
                    out[key] = tensor
                else:
                    product_requests.append(
                        (key, out[("Cdag", i)], out[("Cdag", j)], 2)
                    )
    if product_requests:
        out.update(coupled_reduced_products(product_requests))
    return out


def complete_density_composites(
    operators: dict[tuple, ReducedSU2Tensor],
    site_count: int,
    source_block: RenormalizedSU2Block | None = None,
    *,
    include_spin: bool = False,
) -> dict[tuple, ReducedSU2Tensor]:
    """Ensure one-body density composites are available for all site pairs."""
    out = dict(operators)
    new_site = int(site_count) - 1
    local_jw_ctilde = local_reduced_operator("JWCtilde")
    product_requests = []

    def direct_density(i: int, j: int) -> ReducedSU2Tensor | None:
        if source_block is None:
            return None
        i = int(i)
        j = int(j)
        if i < new_site and j < new_site:
            return grow_source_tensor(
                source_block,
                block_density_tensor(source_block, i, j),
                local_name="I",
            )
        if i < new_site and j == new_site:
            return scale_reduced_tensor(
                reduced_product_tensor_irrep(
                    source_block,
                    source_block.reduced_operators[("Cdag", i)],
                    local_jw_ctilde,
                    total_rank2=0,
                ),
                np.sqrt(2.0),
            )
        if i == new_site and j < new_site:
            forward = direct_density(j, i)
            return ReducedSU2Tensor(forward.tensor.adjoint()) if forward is not None else None
        if i == new_site and j == new_site:
            return reduced_product_tensor_irrep(
                source_block,
                block_identity_reduced_tensor(source_block),
                local_reduced_operator("Ntot"),
                total_rank2=0,
            )
        return None

    def direct_spin_density(i: int, j: int) -> ReducedSU2Tensor | None:
        if source_block is None:
            return None
        i = int(i)
        j = int(j)
        if i < new_site and j < new_site:
            return grow_source_tensor(
                source_block,
                block_spin_density_tensor(source_block, i, j),
                local_name="I",
            )
        if i < new_site and j == new_site:
            return reduced_product_tensor_irrep(
                source_block,
                source_block.reduced_operators[("Cdag", i)],
                local_jw_ctilde,
                total_rank2=2,
            )
        if i == new_site and j < new_site:
            forward = direct_spin_density(j, i)
            return ReducedSU2Tensor(forward.tensor.adjoint()) if forward is not None else None
        return None

    for i in range(int(site_count)):
        for j in range(int(site_count)):
            key = ("Density", i, j)
            if key not in out:
                tensor = direct_density(i, j)
                if tensor is not None:
                    out[key] = tensor
                else:
                    product_requests.append(
                        (key, out[("Cdag", i)], out[("Ctilde", j)], 0, np.sqrt(2.0))
                    )
            if not include_spin:
                continue
            key = ("SpinDensity", i, j)
            if key in out:
                continue
            tensor = direct_spin_density(i, j)
            if tensor is not None:
                out[key] = tensor
            else:
                product_requests.append(
                    (key, out[("Cdag", i)], out[("Ctilde", j)], 2)
                )
    if product_requests:
        out.update(coupled_reduced_products(product_requests))
    return out


def cached_spin_density_tensor(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    cache: dict[tuple[int, int], ReducedSU2Tensor],
    i: int,
    j: int,
) -> ReducedSU2Tensor:
    """Return cached rank-1 spin density without eager fallback work."""
    key = (int(i), int(j))
    tensor = cache.get(key)
    if tensor is None:
        tensor = operators.get(("SpinDensity", int(i), int(j)))
        if tensor is None:
            tensor = reduced_spin_density_tensor(operators, int(i), int(j))
        cache[key] = tensor
    return tensor


def cached_pair_annihilate_tensor(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    cache: dict[tuple[int, int], ReducedSU2Tensor],
    i: int,
    j: int,
) -> ReducedSU2Tensor:
    """Return cached singlet pair annihilator without eager fallback work."""
    key = (int(i), int(j))
    tensor = cache.get(key)
    if tensor is None:
        tensor = reduced_pair_annihilate_tensor(operators, int(i), int(j))
        cache[key] = tensor
    return tensor


def add_optional_reduced_terms(terms: list[ReducedSU2Tensor]) -> ReducedSU2Tensor | None:
    """Add reduced tensors when at least one nonzero term is present."""
    weighted_terms = []
    plain_terms = []
    for term in terms:
        if isinstance(term, tuple) and len(term) == 2 and hasattr(term[0], "blocks"):
            tensor, coeff = term
            if tensor.blocks and abs(coeff) > 0.0:
                weighted_terms.append((tensor, coeff))
        elif term.blocks:
            plain_terms.append(term)
    if weighted_terms:
        weighted = add_weighted_reduced_terms(weighted_terms)
        if weighted is not None:
            plain_terms.append(weighted)
    terms = plain_terms
    if not terms:
        return None
    return add_reduced_tensors(*terms)


def add_weighted_reduced_terms(
    terms: list[tuple[ReducedSU2Tensor, complex]],
    *,
    atol: float = 1e-14,
) -> ReducedSU2Tensor | None:
    """Accumulate ``sum_i coeff_i * tensor_i`` without per-term tensor copies."""
    terms = [(tensor, coeff) for tensor, coeff in terms if tensor.blocks and abs(coeff) > 0.0]
    if not terms:
        return None

    site = terms[0][0].site
    op = terms[0][0].op
    for tensor, _ in terms:
        if tensor.site != site or tensor.op != op:
            raise ValueError("weighted reduced tensor site/op mismatch")

    blocks = {}
    keys = set().union(*(tensor.blocks.keys() for tensor, _ in terms))
    for key in keys:
        block = None
        for tensor, coeff in terms:
            term = tensor.blocks.get(key)
            if term is None:
                continue
            if block is None:
                block = coeff * np.array(term, copy=True)
            else:
                block += coeff * term
        if block is not None and np.any(np.abs(block) > atol):
            blocks[key] = block
    if not blocks:
        return None
    return ReducedSU2Tensor(IrrepTensor(site, site, op, blocks))


def _pair_create_component_ops(ops, i: int, j: int, rank2: int) -> dict[int, np.ndarray]:
    """Primitive components for ``[Cdag_i x Cdag_j]^rank``."""
    cdag_components = {
        1: ops["Cdu"][int(i)],
        -1: ops["Cdd"][int(i)],
    }
    right_components = {
        1: ops["Cdu"][int(j)],
        -1: ops["Cdd"][int(j)],
    }
    components = {}
    for q2 in range(-int(rank2), int(rank2) + 1, 2):
        mat = None
        for left_q2, left in cdag_components.items():
            right_q2 = q2 - left_q2
            right = right_components.get(right_q2)
            if right is None:
                continue
            coeff = cg(1, left_q2, 1, right_q2, int(rank2), q2)
            if abs(coeff) <= 1.0e-14:
                continue
            term = coeff * (left @ right)
            mat = term if mat is None else mat + term
        if mat is not None and np.any(np.abs(mat) > 1.0e-14):
            components[q2] = mat
    return components


@profile_function("seed_exact_pair_composites")
def seed_exact_pair_composites(block: RenormalizedSU2Block, site_count: int = 2) -> None:
    """Attach exact two-operator composites in the constant two-site seed space."""
    site_count = int(site_count)
    if site_count != 2:
        raise ValueError("seed_exact_pair_composites currently expects the two-site seed")
    multiplets = retained_multiplets(block.truncated)
    model = full_jw_model(np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), nelec=2)
    ops = {
        name: [asarray(op) for op in getattr(model, name)]
        for name in ("Cdu", "Cdd", "Cu", "Cd")
    }
    for i in range(site_count):
        for j in range(site_count):
            density = ops["Cdu"][i] @ ops["Cu"][j] + ops["Cdd"][i] @ ops["Cd"][j]
            tensor = reduced_tensor_from_components(
                multiplets,
                {0: density},
                OpIrrep((0, 0)),
            )
            if tensor.blocks:
                block.reduced_operators[("Density", i, j)] = tensor

            spin_components = {
                -2: ops["Cdd"][i] @ ops["Cu"][j],
                0: (ops["Cdu"][i] @ ops["Cu"][j] - ops["Cdd"][i] @ ops["Cd"][j])
                / np.sqrt(2.0),
                2: -(ops["Cdu"][i] @ ops["Cd"][j]),
            }
            tensor = reduced_tensor_from_components(
                multiplets,
                spin_components,
                OpIrrep((0, 2)),
            )
            if tensor.blocks:
                block.reduced_operators[("SpinDensity", i, j)] = tensor

            for rank2, name in ((0, "PairCreate0"), (2, "PairCreate2")):
                components = _pair_create_component_ops(ops, i, j, rank2)
                if components:
                    tensor = reduced_tensor_from_components(
                        multiplets,
                        components,
                        OpIrrep((2, rank2)),
                    )
                    if tensor.blocks:
                        block.reduced_operators[(name, i, j)] = tensor


@profile_function("grow_source_tensor")
def grow_source_tensor(
    source_block: RenormalizedSU2Block,
    tensor: ReducedSU2Tensor,
    *,
    local_name: str,
) -> ReducedSU2Tensor:
    """Extend a retained-block tensor to the untruncated grown basis."""
    return reduced_product_tensor_irrep(
        source_block,
        tensor,
        local_reduced_operator(local_name),
        total_rank2=tensor.op.charge[1],
    )


@profile_function("weighted_packages_from_operators")
def weighted_packages_from_operators(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    h1e,
    eri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = True,
) -> dict[tuple, ReducedSU2Tensor]:
    """Build weighted next-site packages from reduced operators on one site."""
    out = {}
    density_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}
    spin_density_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}
    pair_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}
    cdag_density_cache: dict[tuple[int, int, int], ReducedSU2Tensor] = {}

    for q in future_sites:
        density_terms = []
        exchange_density_terms = []
        exchange_spin_terms = []
        pair_terms = []
        v1_terms = []
        v3_terms = []

        if build_v1:
            for i in range(site_count):
                coeff = h1e[i, q]
                if abs(coeff) > 0.0:
                    v1_terms.append((operators[("Cdag", i)], coeff))

        for i in range(site_count):
            coeff = eri[i, q, q, q]
            if abs(coeff) > 0.0:
                v3_terms.append((operators[("Cdag", i)], coeff))

            for j in range(site_count):
                coeff = eri[i, j, q, q]
                if abs(coeff) > 0.0:
                    density = cached_density_tensor(operators, density_cache, i, j)
                    density_terms.append((density, coeff))

                coeff = eri[i, q, q, j]
                if abs(coeff) > 0.0:
                    density = cached_density_tensor(operators, density_cache, i, j)
                    spin_density = cached_spin_density_tensor(operators, spin_density_cache, i, j)
                    exchange_density_terms.append((density, coeff))
                    exchange_spin_terms.append((spin_density, coeff))

                coeff = eri[q, i, q, j]
                if abs(coeff) > 0.0:
                    pair = cached_pair_annihilate_tensor(operators, pair_cache, i, j)
                    pair_terms.append((pair, coeff))

        if build_v1:
            for i in range(site_count):
                for j in range(site_count):
                    for k in range(site_count):
                        coeff = eri[k, q, j, i]
                        if abs(coeff) > 0.0:
                            key = (k, j, i)
                            cdag_density = cdag_density_cache.get(key)
                            if cdag_density is None:
                                cdag_density = reduced_cdag_density_tensor(operators, density_cache, k, j, i)
                                cdag_density_cache[key] = cdag_density
                            v1_terms.append((cdag_density, coeff))

        packages = {
            ("NextDensity", q): add_optional_reduced_terms(density_terms),
            ("NextExchangeDensity", q): add_optional_reduced_terms(exchange_density_terms),
            ("NextExchangeSpinDensity", q): add_optional_reduced_terms(exchange_spin_terms),
            ("NextPairAnnihilate", q): add_optional_reduced_terms(pair_terms),
            ("NextV1Spinor", q): add_optional_reduced_terms(v1_terms),
            ("NextV3Cdag", q): add_optional_reduced_terms(v3_terms),
        }
        out.update({key: tensor for key, tensor in packages.items() if tensor is not None})
    return out


@profile_function("factorized_new_site_v1_terms")
def factorized_new_site_v1_terms(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    density_cache: dict[tuple[int, int], ReducedSU2Tensor],
    low_rank_eri: LowRankERI,
    site_count: int,
    future_sites,
) -> dict[int, list[ReducedSU2Tensor]]:
    """Build the ``c^dag_k D_ji`` V1 packages through pair-factorized ERIs."""
    new_site = int(site_count) - 1
    all_density_basis = []
    new_density_basis = []
    for factor_index in range(low_rank_eri.rank):
        mode = low_rank_eri.modes[factor_index]
        weight = low_rank_eri.weights[factor_index]
        all_terms = []
        new_terms = []
        for i in range(site_count):
            for j in range(site_count):
                coeff = weight * mode[j, i]
                if abs(coeff) <= 0.0:
                    continue
                density = cached_density_tensor(operators, density_cache, j, i)
                all_terms.append((density, coeff))
                if i == new_site or j == new_site:
                    new_terms.append((density, coeff))
        all_tensor = add_optional_reduced_terms(all_terms)
        new_tensor = add_optional_reduced_terms(new_terms)
        all_density_basis.append(all_tensor)
        new_density_basis.append(new_tensor)

    out: dict[int, list[ReducedSU2Tensor]] = {int(q): [] for q in future_sites}
    for q in future_sites:
        q = int(q)
        for k in range(site_count):
            basis = all_density_basis if k == new_site else new_density_basis
            weighted_density_terms = []
            for factor_index, density_basis in enumerate(basis):
                if density_basis is None:
                    continue
                coeff = low_rank_eri.modes[factor_index, k, q]
                if abs(coeff) > 0.0:
                    weighted_density_terms.append((density_basis, coeff))
            weighted_density = add_optional_reduced_terms(weighted_density_terms)
            if weighted_density is not None:
                out[q].append(
                    coupled_reduced_product(
                        operators[("Cdag", k)],
                        weighted_density,
                        rank2=1,
                    )
                )
    return out


@profile_function("new_site_weighted_packages_from_operators")
def new_site_weighted_packages_from_operators(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    h1e,
    eri,
    site_count: int,
    future_sites,
    low_rank_eri: LowRankERI | None = None,
    *,
    build_v1: bool = True,
) -> dict[tuple, ReducedSU2Tensor]:
    """Build only weighted terms involving the newest site in ``site_count``."""
    out = {}
    new_site = int(site_count) - 1
    density_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}
    spin_density_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}
    pair_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}
    factorized_v1 = (
        factorized_new_site_v1_terms(
            operators,
            density_cache,
            low_rank_eri,
            site_count,
            future_sites,
        )
        if build_v1 and low_rank_eri is not None and low_rank_eri.rank > 0
        else None
    )

    for q in future_sites:
        density_terms = []
        exchange_density_terms = []
        exchange_spin_terms = []
        pair_terms = []
        v1_terms = []
        v3_terms = []

        if build_v1:
            coeff = h1e[new_site, q]
            if abs(coeff) > 0.0:
                v1_terms.append((operators[("Cdag", new_site)], coeff))

        coeff = eri[new_site, q, q, q]
        if abs(coeff) > 0.0:
            v3_terms.append((operators[("Cdag", new_site)], coeff))

        for i in range(site_count):
            for j in range(site_count):
                if i != new_site and j != new_site:
                    continue

                coeff = eri[i, j, q, q]
                if abs(coeff) > 0.0:
                    density = cached_density_tensor(operators, density_cache, i, j)
                    density_terms.append((density, coeff))

                coeff = eri[i, q, q, j]
                if abs(coeff) > 0.0:
                    density = cached_density_tensor(operators, density_cache, i, j)
                    spin_density = cached_spin_density_tensor(operators, spin_density_cache, i, j)
                    exchange_density_terms.append((density, coeff))
                    exchange_spin_terms.append((spin_density, coeff))

                coeff = eri[q, i, q, j]
                if abs(coeff) > 0.0:
                    pair = cached_pair_annihilate_tensor(operators, pair_cache, i, j)
                    pair_terms.append((pair, coeff))

        if not build_v1:
            pass
        elif any(key[0] == "CdagDensity" for key in operators):
            cdag_density_cache: dict[tuple[int, int, int], ReducedSU2Tensor] = {}
            for k in range(site_count):
                for i in range(site_count):
                    for j in range(site_count):
                        if i != new_site and j != new_site and k != new_site:
                            continue
                        coeff = eri[k, q, j, i]
                        if abs(coeff) <= 0.0:
                            continue
                        key = (k, j, i)
                        cdag_density = cdag_density_cache.get(key)
                        if cdag_density is None:
                            cdag_density = reduced_cdag_density_tensor(
                                operators,
                                density_cache,
                                k,
                                j,
                                i,
                            )
                            cdag_density_cache[key] = cdag_density
                        v1_terms.append((cdag_density, coeff))
        elif factorized_v1 is None:
            for k in range(site_count):
                weighted_density_terms = []
                for i in range(site_count):
                    for j in range(site_count):
                        if i != new_site and j != new_site and k != new_site:
                            continue
                        coeff = eri[k, q, j, i]
                        if abs(coeff) > 0.0:
                            density = cached_density_tensor(operators, density_cache, j, i)
                            weighted_density_terms.append((density, coeff))
                weighted_density = add_optional_reduced_terms(weighted_density_terms)
                if weighted_density is not None:
                    v1_terms.append(
                        coupled_reduced_product(
                            operators[("Cdag", k)],
                            weighted_density,
                            rank2=1,
                        )
                    )
        else:
            v1_terms.extend(factorized_v1.get(int(q), ()))

        packages = {
            ("NextDensity", q): add_optional_reduced_terms(density_terms),
            ("NextExchangeDensity", q): add_optional_reduced_terms(exchange_density_terms),
            ("NextExchangeSpinDensity", q): add_optional_reduced_terms(exchange_spin_terms),
            ("NextPairAnnihilate", q): add_optional_reduced_terms(pair_terms),
            ("NextV1Spinor", q): add_optional_reduced_terms(v1_terms),
            ("NextV3Cdag", q): add_optional_reduced_terms(v3_terms),
        }
        out.update({key: tensor for key, tensor in packages.items() if tensor is not None})
    return out


@profile_function("seed_future_weighted_packages")
def seed_future_weighted_packages(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_count: int,
    future_sites,
    *,
    project_v1_packages: bool = False,
) -> None:
    """Attach initial weighted packages without storing individual composites."""
    packages = weighted_packages_from_operators(
        block.reduced_operators,
        h1e,
        eri,
        site_count,
        future_sites,
        build_v1=not project_v1_packages,
    )
    if project_v1_packages:
        packages.update(seed_component_v1_packages(block, h1e, eri, site_count, future_sites))
    block.reduced_operators.update(packages)


@profile_function("update_future_weighted_packages")
def update_future_weighted_packages(
    narg: ThreeSiteSU2NARG,
    grown: dict[tuple[str, int], ReducedSU2Tensor],
    h1e,
    eri,
    future_sites,
    low_rank_eri: LowRankERI | None = None,
    project_v1_packages: bool = False,
) -> dict[tuple, ReducedSU2Tensor]:
    """Recursively update weighted packages for future growth sites."""
    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    if source_block is None:
        raise ValueError("grown NARG object does not carry its source block")
    if not future_sites:
        return {}

    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    new_involving = new_site_weighted_packages_from_operators(
        grown,
        h1e,
        eri,
        site_count,
        future_sites,
        low_rank_eri=low_rank_eri,
        build_v1=not project_v1_packages,
    )
    if project_v1_packages:
        new_involving.update(
            grown_reduced_v1_packages(
                source_block,
                grown,
                h1e,
                eri,
                future_sites,
            )
        )

    out: dict[tuple, ReducedSU2Tensor] = {}
    even_prefixes = {
        "NextDensity",
        "NextExchangeDensity",
        "NextExchangeSpinDensity",
        "NextPairAnnihilate",
    }
    odd_prefixes = {"NextV1Spinor", "NextV3Cdag"}
    for q in future_sites:
        for prefix in even_prefixes:
            key = (prefix, q)
            terms = []
            carried = source_block.reduced_operators.get(key)
            if carried is not None:
                terms.append(grow_source_tensor(source_block, carried, local_name="I"))
            if key in new_involving:
                terms.append(new_involving[key])
            total = add_optional_reduced_terms(terms)
            if total is not None:
                out[key] = total
        for prefix in odd_prefixes:
            key = (prefix, q)
            terms = []
            carried = source_block.reduced_operators.get(key)
            if carried is not None:
                terms.append(grow_source_tensor(source_block, carried, local_name="JW"))
            if key in new_involving:
                terms.append(new_involving[key])
            total = add_optional_reduced_terms(terms)
            if total is not None:
                out[key] = total
    return out


@profile_function("reduced_coupling_operators_from_growth")
def reduced_coupling_operators_from_growth(
    narg: ThreeSiteSU2NARG,
    truncated,
    h1e=None,
    eri=None,
    future_sites=(),
    backend=None,
    low_rank_eri: LowRankERI | None = None,
    project_v1_packages: bool = False,
    carry_rdm_operators: bool = False,
    carry_spin_rdm_operators: bool = False,
    rotate: bool = True,
) -> dict[tuple, ReducedSU2Tensor]:
    """Grow, optionally compose, and rotate reduced operators after truncation."""
    backend = resolve_su2_narg_backend(backend)
    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    future_sites = tuple(future_sites)
    carry_pair_composites = bool(project_v1_packages and len(future_sites) > 1)
    grown = grown_coupling_operators(
        narg,
        include_even_composites=(
            (not project_v1_packages)
            or carry_pair_composites
            or carry_rdm_operators
            or carry_spin_rdm_operators
        ),
        even_composites=(
            {"Density"}
            if (
                carry_rdm_operators
                and not carry_spin_rdm_operators
                and project_v1_packages
                and not carry_pair_composites
            )
            else None
        ),
    )
    if carry_rdm_operators or carry_spin_rdm_operators:
        if source_block is None:
            raise ValueError("grown NARG object does not carry its source block")
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        grown = complete_density_composites(
            grown,
            site_count,
            source_block=source_block,
            include_spin=carry_spin_rdm_operators,
        )
    if carry_pair_composites:
        if source_block is None:
            raise ValueError("grown NARG object does not carry its source block")
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        grown = complete_pair_composites(grown, site_count, source_block=source_block)
    weighted = (
        update_future_weighted_packages(
            narg,
            grown,
            h1e,
            eri,
            future_sites,
            low_rank_eri=low_rank_eri,
            project_v1_packages=project_v1_packages,
        )
        if h1e is not None
        else {}
    )
    tensors = {**grown, **weighted}
    if rotate:
        return rotate_reduced_tensors_to_truncated(
            truncated,
            tensors,
            backend=backend,
        )

    selected = set(truncated.site.irreps)
    return {
        key: ReducedSU2Tensor(
            IrrepTensor(
                truncated.site,
                truncated.site,
                tensor.op,
                {
                    (bra, ket): value
                    for (bra, ket), value in tensor.blocks.items()
                    if bra in selected and ket in selected
                },
            )
        )
        for key, tensor in tensors.items()
    }


def feasible_nelec_for_target(block_sites: int, final_sites: int, target_nelec: int) -> set[int]:
    """Particle-number sectors that can still reach ``target_nelec``."""
    remaining = int(final_sites) - int(block_sites)
    lo = max(0, int(target_nelec) - 2 * remaining)
    hi = min(2 * int(block_sites), int(target_nelec))
    return set(range(lo, hi + 1))


def env_spin_can_couple(block_j2: int, env_nelec: int, env_sites: int, target_j2: int) -> bool:
    """Return whether a remaining-space spin can couple a block spin to the target."""
    env_nelec = int(env_nelec)
    env_sites = int(env_sites)
    if env_nelec < 0 or env_nelec > 2 * env_sites:
        return False
    env_min_j2 = env_nelec % 2
    env_max_j2 = min(env_nelec, 2 * env_sites - env_nelec)
    for env_j2 in range(env_min_j2, env_max_j2 + 1, 2):
        if (int(block_j2) + env_j2 + int(target_j2)) % 2:
            continue
        if abs(int(block_j2) - env_j2) <= int(target_j2) <= int(block_j2) + env_j2:
            return True
    return False


def feasible_irreps_for_target_spin(
    irreps,
    *,
    block_sites: int,
    final_sites: int,
    target_nelec: int,
    target_j2: int | None,
) -> set[Irrep] | None:
    """Filter block irreps that can still reach the final particle and spin sector."""
    if target_j2 is None:
        return None
    return set(irreps) & feasible_target_irreps(
        block_sites=block_sites,
        final_sites=final_sites,
        target_nelec=target_nelec,
        target_j2=target_j2,
    )


def feasible_target_irreps(
    *,
    block_sites: int,
    final_sites: int,
    target_nelec: int,
    target_j2: int | None,
) -> set[Irrep] | None:
    """All block irreps that can still reach the requested final sector."""
    if target_j2 is None:
        return None
    allowed_nelec = feasible_nelec_for_target(block_sites, final_sites, target_nelec)
    remaining_sites = int(final_sites) - int(block_sites)
    allowed = set()
    for block_nelec in allowed_nelec:
        min_j2 = int(block_nelec) % 2
        max_j2 = min(int(block_nelec), 2 * int(block_sites) - int(block_nelec))
        for block_j2 in range(min_j2, max_j2 + 1, 2):
            env_nelec = int(target_nelec) - int(block_nelec)
            if env_spin_can_couple(block_j2, env_nelec, remaining_sites, int(target_j2)):
                allowed.add(Irrep((int(block_nelec), int(block_j2))))
    return allowed


def future_one_body_fill_energies(h1e_full, future_sites) -> np.ndarray:
    """Noninteracting filling estimate for the remaining spatial orbitals."""
    future_sites = tuple(int(site) for site in future_sites)
    nfuture = len(future_sites)
    fill = np.full(2 * nfuture + 1, np.inf, dtype=float)
    fill[0] = 0.0
    if nfuture == 0:
        return fill
    h_future = np.asarray(h1e_full)[np.ix_(future_sites, future_sites)]
    h_future = 0.5 * (h_future + h_future.conj().T)
    orbital_energies = np.linalg.eigvalsh(h_future)
    spin_orbital_energies = np.sort(np.repeat(np.real(orbital_energies), 2))
    fill[1:] = np.cumsum(spin_orbital_energies)
    return fill


def adaptive_root_scorer(
    D: int | AdaptiveD,
    *,
    h1e_full=None,
    future_sites=(),
    target_nelec: int | None = None,
):
    """Build the root scorer requested by an adaptive truncation rule."""
    if not isinstance(D, AdaptiveD) or D.criterion == "energy":
        return None
    if h1e_full is None or target_nelec is None:
        return None
    env_fill = future_one_body_fill_energies(h1e_full, future_sites)

    def score(root):
        block_nelec, _ = root.irrep.charge
        env_nelec = int(target_nelec) - int(block_nelec)
        if env_nelec < 0 or env_nelec >= env_fill.size:
            return np.inf
        return float(root.energy + env_fill[env_nelec])

    return score


def seed_D_from_spec(D: int | AdaptiveD) -> int:
    """Use a small fixed seed size for the initial two-site block."""
    if isinstance(D, AdaptiveD):
        return min(10, int(D.D_min))
    return int(D)


def describe_D_spec(D: int | AdaptiveD) -> int | dict[str, object]:
    """Serialize a fixed or adaptive truncation rule for timings metadata."""
    if isinstance(D, AdaptiveD):
        return {
            "adaptive": True,
            "D_min": int(D.D_min),
            "D_max": int(D.D_max),
            "energy_window": float(D.energy_window),
            "criterion": str(D.criterion),
        }
    return int(D)


@profile_function("renormalized_block_from_narg")
def renormalized_block_from_narg(
    narg: ThreeSiteSU2NARG,
    h1e_block,
    eri_block,
    D: int | AdaptiveD,
    *,
    allowed_nelec: set[int] | None = None,
    h1e_full=None,
    eri_full=None,
    future_sites=(),
    target_nelec: int | None = None,
    target_j2: int | None = None,
    backend=None,
    low_rank_eri: LowRankERI | None = None,
    project_v1_packages: bool = False,
    carry_rdm_operators: bool = False,
    carry_spin_rdm_operators: bool = False,
    retain_all: bool = False,
) -> RenormalizedSU2Block:
    """Truncate a grown SU2-NARG object and attach reduced coupling operators."""
    backend = resolve_su2_narg_backend(backend)
    block_sites = int(h1e_block.shape[0])
    final_sites = (
        int(h1e_full.shape[0])
        if h1e_full is not None
        else block_sites + len(tuple(future_sites))
    )
    allowed_irreps = feasible_irreps_for_target_spin(
        narg.site.irreps,
        block_sites=block_sites,
        final_sites=final_sites,
        target_nelec=target_nelec if target_nelec is not None else final_sites,
        target_j2=target_j2,
    )
    if retain_all:
        selected_irreps = [
            irrep
            for irrep in narg.site.irreps
            if (allowed_nelec is None or irrep.charge[0] in allowed_nelec)
            and (allowed_irreps is None or irrep in allowed_irreps)
        ]
        if not selected_irreps:
            raise ValueError(
                "no feasible SU2 multiplets remain in the exact cluster interior"
            )
        dims = {irrep: narg.site.sector_dim(irrep) for irrep in selected_irreps}
        site = IrrepSite(narg.site.symmetry, dims)
        roots = []
        transform_blocks = {}
        hamiltonian_blocks = {}
        for irrep in selected_irreps:
            dim = dims[irrep]
            identity = np.eye(dim, dtype=complex)
            transform_blocks[(irrep, irrep)] = identity
            h_block = narg.hamiltonian.block(irrep, irrep)
            hamiltonian_blocks[(irrep, irrep)] = h_block
            diagonal = np.real(np.diag(h_block))
            for local_index in range(dim):
                roots.append(
                    SectorRoot(
                        energy=float(diagonal[local_index]),
                        irrep=irrep,
                        local_index=local_index,
                        vector=identity[:, local_index].copy(),
                    )
                )
        transform = IrrepTensor(
            narg.site,
            site,
            OpIrrep((0, 0)),
            transform_blocks,
        )
        hamiltonian = IrrepTensor(site, site, OpIrrep((0, 0)), hamiltonian_blocks)
        bases = {
            irrep: narg.bases[irrep]
            for irrep in selected_irreps
            if irrep in narg.bases
        }
        truncated = TruncatedSU2NARG(
            narg,
            roots,
            site,
            bases,
            transform,
            hamiltonian,
        )
    else:
        scorer = adaptive_root_scorer(
            D,
            h1e_full=h1e_full,
            future_sites=future_sites,
            target_nelec=target_nelec,
        )
        with profile_section("truncate_to_D"):
            truncated = truncate_to_D(
                narg,
                D=D,
                allowed_nelec=allowed_nelec,
                allowed_irreps=allowed_irreps,
                root_scorer=scorer,
                backend=backend,
            )
    reduced_operators = reduced_coupling_operators_from_growth(
        narg,
        truncated,
        h1e=h1e_full,
        eri=eri_full,
        future_sites=future_sites,
        backend=backend,
        low_rank_eri=low_rank_eri,
        project_v1_packages=project_v1_packages,
        carry_rdm_operators=carry_rdm_operators,
        carry_spin_rdm_operators=carry_spin_rdm_operators,
        rotate=not retain_all,
    )
    block = RenormalizedSU2Block(
        truncated=truncated,
        hamiltonian=truncated.hamiltonian,
        transform=truncated.transform,
        operators={},
        reduced_operators=reduced_operators,
        parity=None,
    )
    block._su2_multiplets = []
    block._su2_requested_D = D
    block._su2_chosen_D = len(truncated.kept_roots)
    block._su2_exact_basis = bool(retain_all)
    return block


@profile_function("grow_one_site_direct_reduced")
def grow_one_site_direct_reduced(
    h1e_block,
    eri_block,
    source_block: RenormalizedSU2Block,
    *,
    target_nelec: int | None = None,
    build_branch_basis: bool = True,
) -> ThreeSiteSU2NARG:
    """Grow a retained block by one site using direct reduced SU(2) tensors."""
    site_index = h1e_block.shape[0] - 1
    if target_nelec is None:
        if hasattr(source_block, "_su2_allowed_final_nelec"):
            delattr(source_block, "_su2_allowed_final_nelec")
    else:
        source_block._su2_allowed_final_nelec = {int(target_nelec)}
    hamiltonian = direct_reduced_full_hamiltonian_tensor(
        source_block, h1e_block, eri_block, site_index=site_index
    )
    if build_branch_basis:
        branch_states = grow_su2_block_by_one_site(retained_multiplets(source_block.truncated))
        site, bases, provenance = branch_irrep_site(branch_states)
        site = hamiltonian.bra
    else:
        branch_states = []
        site = hamiltonian.bra
        bases = {}
        provenance = {}
    site = hamiltonian.bra
    narg = ThreeSiteSU2NARG(source_block.truncated, branch_states, site, bases, provenance, hamiltonian)
    narg._su2_source_renormalized_block = source_block
    return narg


def run_su2_narg_chain(
    h1e,
    eri,
    D_by_size: dict[int, int | AdaptiveD],
    *,
    final_size: int | None = None,
    target_nelec: int | None = None,
    target_j2: int | None = None,
    backend=None,
    low_rank_eri: LowRankERI | str | bool | None = None,
    build_branch_basis: bool = False,
    project_growth_hamiltonian: bool = False,
    project_v1_packages: bool = True,
    carry_rdm_operators: bool = False,
    carry_spin_rdm_operators: bool = False,
    cluster_boundaries: tuple[int, ...] | None = None,
) -> SU2ChainResult:
    """Grow a direct-reduced SU(2)-NARG chain.

    ``D_by_size[n]`` is used when an ``n``-site block must be retained for the
    next growth step.  The final size is not truncated unless it also serves as
    an intermediate block in a longer run.  When ``cluster_boundaries`` is
    provided, intermediate sizes inside a cluster retain their complete
    feasible reduced basis; NARG truncation is applied only at those boundaries.
    """
    final_size = h1e.shape[0] if final_size is None else int(final_size)
    target_nelec = final_size if target_nelec is None else int(target_nelec)
    if final_size < 2:
        raise ValueError("final_size must be at least 2")
    if cluster_boundaries is None:
        cluster_boundaries = tuple(range(2, final_size + 1))
    else:
        cluster_boundaries = tuple(sorted({int(size) for size in cluster_boundaries}))
        if not cluster_boundaries or cluster_boundaries[-1] != final_size:
            raise ValueError("cluster_boundaries must end at final_size")
        if cluster_boundaries[0] < 2 or any(
            size > final_size for size in cluster_boundaries
        ):
            raise ValueError("cluster_boundaries must lie between 2 and final_size")
    cluster_boundary_set = set(cluster_boundaries)
    backend = resolve_su2_narg_backend(backend)
    if project_growth_hamiltonian:
        build_branch_basis = True
    if low_rank_eri is None:
        low_rank_setting = os.environ.get("SU2_NARG_LOW_RANK_ERI", "0").strip().lower()
        if low_rank_setting in {"1", "true", "yes", "on", "exact"}:
            tol = float(os.environ.get("SU2_NARG_LOW_RANK_ERI_TOL", "1e-10"))
            low_rank_eri = LowRankERI.from_dense(eri, tol=tol)
        else:
            low_rank_eri = None
    elif isinstance(low_rank_eri, LowRankERI):
        pass
    elif low_rank_eri:
        tol = float(os.environ.get("SU2_NARG_LOW_RANK_ERI_TOL", "1e-10"))
        low_rank_eri = LowRankERI.from_dense(eri, tol=tol)
    else:
        low_rank_eri = None

    timings = {
        "D_by_size": {int(k): describe_D_spec(v) for k, v in dict(D_by_size).items()},
        "kept_by_size": {},
        "cluster_boundaries": cluster_boundaries,
        "exact_internal_sizes": tuple(
            size for size in range(2, final_size) if size not in cluster_boundary_set
        ),
    }
    blocks: dict[int, RenormalizedSU2Block] = {}
    reset_su2_profile()

    start = time.perf_counter()
    # The two-site seed lies inside a larger first supersite when size 2 is
    # not a cluster boundary, so retain its complete ten-multiplet basis.
    d2 = (
        seed_D_from_spec(D_by_size.get(2, 10))
        if 2 in cluster_boundary_set
        else 10
    )
    allowed_nelec_2 = feasible_nelec_for_target(2, final_size, target_nelec)
    allowed_irreps_2 = feasible_target_irreps(
        block_sites=2,
        final_sites=final_size,
        target_nelec=target_nelec,
        target_j2=target_j2,
    )
    blocks[2] = build_renormalized_two_site_block(
        h1e[:2, :2],
        eri[:2, :2, :2, :2],
        D=d2,
        allowed_nelec=allowed_nelec_2,
        allowed_irreps=allowed_irreps_2,
        backend=backend,
    )
    seed_future = tuple(range(2, final_size))
    if project_v1_packages or carry_rdm_operators or carry_spin_rdm_operators:
        seed_exact_pair_composites(blocks[2], site_count=2)
    seed_future_weighted_packages(
        blocks[2],
        h1e,
        eri,
        2,
        seed_future,
        project_v1_packages=project_v1_packages,
    )
    timings["build_block_2"] = time.perf_counter() - start
    timings["kept_by_size"][2] = len(blocks[2].truncated.kept_roots)

    source = blocks[2]
    # For a two-site active space the seed block is already the final NARG
    # object; no growth step will run.
    final = source if final_size == 2 else None
    for nsites in range(3, final_size + 1):
        h1e_n = h1e[:nsites, :nsites]
        eri_n = eri[:nsites, :nsites, :nsites, :nsites]

        start = time.perf_counter()
        final = grow_one_site_direct_reduced(
            h1e_n,
            eri_n,
            source,
            target_nelec=target_nelec if nsites == final_size else None,
            build_branch_basis=build_branch_basis,
        )
        if project_growth_hamiltonian:
            final.hamiltonian = project_hamiltonian_irrep_tensor(
                final,
                h1e_n,
                eri_n,
                backend=backend,
            )
        timings[f"grow_{nsites}"] = time.perf_counter() - start

        if nsites < final_size:
            start = time.perf_counter()
            if nsites in cluster_boundary_set:
                retain = D_by_size[nsites]
            else:
                # Keeping every feasible multiplet makes the internal growth an
                # exact reduced-basis realization of one composite supersite.
                retain = sum(int(dim) for dim in final.site.dims.values())
            blocks[nsites] = renormalized_block_from_narg(
                final,
                h1e_n,
                eri_n,
                D=retain,
                allowed_nelec=feasible_nelec_for_target(nsites, final_size, target_nelec),
                h1e_full=h1e,
                eri_full=eri,
                future_sites=tuple(range(nsites, final_size)),
                target_nelec=target_nelec,
                target_j2=target_j2,
                backend=backend,
                low_rank_eri=low_rank_eri,
                project_v1_packages=project_v1_packages,
                carry_rdm_operators=carry_rdm_operators,
                carry_spin_rdm_operators=carry_spin_rdm_operators,
                retain_all=nsites not in cluster_boundary_set,
            )
            timings[f"renormalize_{nsites}"] = time.perf_counter() - start
            timings["kept_by_size"][nsites] = len(blocks[nsites].truncated.kept_roots)
            source = blocks[nsites]

    if final is None:
        raise RuntimeError("chain growth did not produce a final NARG object")
    profile = su2_profile_snapshot()
    if profile:
        timings["profile"] = profile
    timings["project_growth_hamiltonian"] = bool(project_growth_hamiltonian)
    timings["project_v1_packages"] = bool(project_v1_packages)
    timings["carry_rdm_operators"] = bool(carry_rdm_operators)
    timings["carry_spin_rdm_operators"] = bool(carry_spin_rdm_operators)
    return SU2ChainResult(final=final, blocks=blocks, timings=timings, backend=backend.summary())


def diagonalize_block(
    narg: ThreeSiteSU2NARG,
    nelec: int,
    j2: int,
    nroots: int = 6,
    *,
    backend=None,
    return_vectors: bool = False,
):
    """Diagonalize one final SU2 sector."""
    backend = resolve_su2_narg_backend(backend)
    irrep = Irrep((nelec, j2))
    block = narg.hamiltonian.block(irrep, irrep)
    if block.size == 0:
        if return_vectors:
            return np.array([]), np.zeros((0, 0), dtype=complex), block
        return np.array([]), block
    result = backend.diagonalize_sector(block, nroots=nroots)
    if return_vectors:
        return result.values, result.vectors, block
    return result.values, block


def _direct_ci_binary(norb: int, nelec: int, j2: int) -> np.ndarray:
    """Build the determinant basis matching a highest-weight SU(2) sector."""
    if (int(nelec) + int(j2)) % 2:
        raise ValueError(f"Incompatible nelec={nelec} and doubled spin j2={j2}.")
    nalpha = (int(nelec) + int(j2)) // 2
    nbeta = int(nelec) - nalpha
    if nalpha < 0 or nbeta < 0 or nalpha > int(norb) or nbeta > int(norb):
        raise ValueError(
            f"Invalid electron sector for norb={norb}: nalpha={nalpha}, nbeta={nbeta}."
        )
    from pyqed.qchem.ci.fci import get_fci_combos

    mo_occ = np.zeros((2, int(norb)), dtype=int)
    mo_occ[0, :nalpha] = 1
    mo_occ[1, :nbeta] = 1
    return get_fci_combos(mo_occ=mo_occ)


def _direct_ci_rows_and_phases(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map direct-CI determinants to local-site Fock rows and phases.

    SU2-NARG primitive vectors use interleaved local states
    ``|0>, |up>, |down>, |up down>`` per site.  The direct-CI backend orders
    determinants as all alpha creators followed by all beta creators.  The
    phase below is the fermionic permutation between those conventions, with an
    additional local-double sign for the chosen one-site basis convention.
    """
    norb = int(binary.shape[2])
    powers = 4 ** np.arange(norb - 1, -1, -1, dtype=np.int64)
    rows = np.empty(binary.shape[0], dtype=np.int64)
    phases = np.empty(binary.shape[0], dtype=float)
    for det_index, det in enumerate(binary):
        alpha = np.asarray(det[0], dtype=np.int64)
        beta = np.asarray(det[1], dtype=np.int64)
        rows[det_index] = int(np.dot(alpha + 2 * beta, powers))
        inversions = 0
        for site in range(norb):
            if beta[site]:
                inversions += int(np.sum(alpha[site + 1 :]))
        local_doubles = int(np.sum(alpha & beta))
        phases[det_index] = -1.0 if (inversions + local_doubles) % 2 else 1.0
    return rows, phases


def _sigma_compact_columns(h1e, eri, binary: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply the compact direct-CI Hamiltonian to every coefficient column."""
    from pyqed.qchem.mcscf.direct_ci import (
        _sigma_compact_spin_string,
        _compute_diag_compact,
        build_spin_string_connectivity,
    )

    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    same_spin_eri = eri - eri.swapaxes(1, 3)
    cross_spin_eri = eri
    h_diag = _compute_diag_compact(h1e, same_spin_eri, cross_spin_eri, binary)
    conn = build_spin_string_connectivity(binary)

    coeffs = np.asarray(coeffs)
    out = np.empty_like(coeffs, dtype=np.result_type(coeffs, float))

    def apply_one(vec):
        vec = np.asarray(vec)
        if np.iscomplexobj(vec):
            real = _sigma_compact_spin_string(
                h1e,
                same_spin_eri,
                cross_spin_eri,
                h_diag,
                np.ascontiguousarray(vec.real),
                conn.alpha_occ,
                conn.beta_occ,
                conn.I_A,
                conn.J_A,
                conn.p_A,
                conn.q_A,
                conn.phase_A,
                conn.I_B,
                conn.J_B,
                conn.p_B,
                conn.q_B,
                conn.phase_B,
                conn.I_AA,
                conn.J_AA,
                conn.p_AA,
                conn.q_AA,
                conn.r_AA,
                conn.s_AA,
                conn.phase_AA,
                conn.I_BB,
                conn.J_BB,
                conn.p_BB,
                conn.q_BB,
                conn.r_BB,
                conn.s_BB,
                conn.phase_BB,
            )
            imag = _sigma_compact_spin_string(
                h1e,
                same_spin_eri,
                cross_spin_eri,
                h_diag,
                np.ascontiguousarray(vec.imag),
                conn.alpha_occ,
                conn.beta_occ,
                conn.I_A,
                conn.J_A,
                conn.p_A,
                conn.q_A,
                conn.phase_A,
                conn.I_B,
                conn.J_B,
                conn.p_B,
                conn.q_B,
                conn.phase_B,
                conn.I_AA,
                conn.J_AA,
                conn.p_AA,
                conn.q_AA,
                conn.r_AA,
                conn.s_AA,
                conn.phase_AA,
                conn.I_BB,
                conn.J_BB,
                conn.p_BB,
                conn.q_BB,
                conn.r_BB,
                conn.s_BB,
                conn.phase_BB,
            )
            return real + 1j * imag
        return _sigma_compact_spin_string(
            h1e,
            same_spin_eri,
            cross_spin_eri,
            h_diag,
            np.ascontiguousarray(vec),
            conn.alpha_occ,
            conn.beta_occ,
            conn.I_A,
            conn.J_A,
            conn.p_A,
            conn.q_A,
            conn.phase_A,
            conn.I_B,
            conn.J_B,
            conn.p_B,
            conn.q_B,
            conn.phase_B,
            conn.I_AA,
            conn.J_AA,
            conn.p_AA,
            conn.q_AA,
            conn.r_AA,
            conn.s_AA,
            conn.phase_AA,
            conn.I_BB,
            conn.J_BB,
            conn.p_BB,
            conn.q_BB,
            conn.r_BB,
            conn.s_BB,
            conn.phase_BB,
        )

    for col in range(coeffs.shape[1]):
        out[:, col] = apply_one(coeffs[:, col])
    return out


def project_hamiltonian_irrep_tensor(
    narg,
    h1e,
    eri,
    *,
    backend=None,
) -> IrrepTensor:
    """Project the exact active-space Hamiltonian into every retained SU(2) sector."""
    backend = resolve_su2_narg_backend(backend)
    blocks = {}
    norb = np.asarray(h1e).shape[0]
    for irrep, basis in narg.bases.items():
        if irrep not in narg.site.dims:
            continue
        if basis.size == 0:
            continue
        nelec, j2 = irrep.charge
        binary = _direct_ci_binary(norb, int(nelec), int(j2))
        rows, phases = _direct_ci_rows_and_phases(binary)
        ci_basis = phases[:, None] * basis[rows, :]
        ci_gram = ci_basis.conj().T @ ci_basis
        ci_gram_error = float(np.max(np.abs(ci_gram - np.eye(ci_gram.shape[0]))))
        if ci_gram_error > 1.0e-8:
            raise ValueError(
                "SU2-NARG growth basis is not orthonormal after direct-CI mapping; "
                f"max Gram error {ci_gram_error:g} in sector {irrep}"
            )
        sigma = _sigma_compact_columns(h1e, eri, binary, ci_basis)
        block = ci_basis.conj().T @ sigma
        block = 0.5 * (block + block.conj().T)
        if block.size and np.any(np.abs(block) > 1.0e-14):
            blocks[(irrep, irrep)] = block
    return IrrepTensor(narg.site, narg.site, OpIrrep((0, 0)), blocks)
