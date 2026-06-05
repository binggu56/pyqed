#!/usr/bin/env python3
"""Reusable direct-reduced SU(2)-NARG chain growth utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from pyqed.narg.irrep_tensor import Irrep
from .su2_three_site import (
    ThreeSiteSU2NARG,
    block_retained_scalar_tensor,
    branch_irrep_site,
    direct_reduced_full_hamiltonian_tensor,
    grow_su2_block_by_one_site,
    local_reduced_operator,
    profile_function,
    profile_section,
    reduced_product_tensor_irrep,
    reset_su2_profile,
    rotate_reduced_tensor_to_truncated,
    su2_profile_snapshot,
)
from .su2_two_site import (
    RenormalizedSU2Block,
    build_renormalized_two_site_block,
    retained_multiplets,
    truncate_to_D,
)
from .su2_reduced_tensor import (
    ReducedSU2Tensor,
    add_reduced_tensors,
    coupled_reduced_product,
    scale_reduced_tensor,
)


@dataclass
class SU2ChainResult:
    """Result of a direct-reduced SU(2)-NARG chain growth."""

    final: ThreeSiteSU2NARG
    blocks: dict[int, RenormalizedSU2Block]
    timings: dict[str, float] = field(default_factory=dict)


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


@profile_function("grown_coupling_operators")
def grown_coupling_operators(narg: ThreeSiteSU2NARG) -> dict[tuple[str, int], ReducedSU2Tensor]:
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
    return scale_reduced_tensor(
        coupled_reduced_product(operators[("Cdag", i)], operators[("Ctilde", j)], rank2=0),
        np.sqrt(2.0),
    )


def reduced_spin_density_tensor(operators: dict[tuple[str, int], ReducedSU2Tensor], i: int, j: int):
    """Reduced rank-1 spin density from ``Cdag_i x Ctilde_j``."""
    return coupled_reduced_product(operators[("Cdag", i)], operators[("Ctilde", j)], rank2=2)


def reduced_pair_annihilate_tensor(operators: dict[tuple[str, int], ReducedSU2Tensor], i: int, j: int):
    """Reduced singlet pair annihilation."""
    return scale_reduced_tensor(
        coupled_reduced_product(operators[("Ctilde", i)], operators[("Ctilde", j)], rank2=0),
        -1.0 / np.sqrt(2.0),
    )


def reduced_cdag_density_tensor(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    density_cache: dict[tuple[int, int], ReducedSU2Tensor],
    k: int,
    j: int,
    i: int,
):
    """Reduced spinor ``c^dag_k sum_sigma c^dag_j,sigma c_i,sigma``."""
    density = density_cache.setdefault((j, i), reduced_density_tensor(operators, j, i))
    return coupled_reduced_product(operators[("Cdag", k)], density, rank2=1)


def add_optional_reduced_terms(terms: list[ReducedSU2Tensor]) -> ReducedSU2Tensor | None:
    """Add reduced tensors when at least one nonzero term is present."""
    terms = [term for term in terms if term.blocks]
    if not terms:
        return None
    return add_reduced_tensors(*terms)


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

        for i in range(site_count):
            coeff = h1e[i, q]
            if abs(coeff) > 0.0:
                v1_terms.append(scale_reduced_tensor(operators[("Cdag", i)], coeff))

            coeff = eri[i, q, q, q]
            if abs(coeff) > 0.0:
                v3_terms.append(scale_reduced_tensor(operators[("Cdag", i)], coeff))

            for j in range(site_count):
                coeff = eri[i, j, q, q]
                if abs(coeff) > 0.0:
                    density = density_cache.setdefault((i, j), reduced_density_tensor(operators, i, j))
                    density_terms.append(scale_reduced_tensor(density, coeff))

                coeff = eri[i, q, q, j]
                if abs(coeff) > 0.0:
                    density = density_cache.setdefault((i, j), reduced_density_tensor(operators, i, j))
                    spin_density = spin_density_cache.setdefault((i, j), reduced_spin_density_tensor(operators, i, j))
                    exchange_density_terms.append(scale_reduced_tensor(density, coeff))
                    exchange_spin_terms.append(scale_reduced_tensor(spin_density, coeff))

                coeff = eri[q, i, q, j]
                if abs(coeff) > 0.0:
                    pair = pair_cache.setdefault((i, j), reduced_pair_annihilate_tensor(operators, i, j))
                    pair_terms.append(scale_reduced_tensor(pair, coeff))

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
                        v1_terms.append(
                            scale_reduced_tensor(
                                cdag_density,
                                coeff,
                            )
                        )

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


@profile_function("new_site_weighted_packages_from_operators")
def new_site_weighted_packages_from_operators(
    operators: dict[tuple[str, int], ReducedSU2Tensor],
    h1e,
    eri,
    site_count: int,
    future_sites,
) -> dict[tuple, ReducedSU2Tensor]:
    """Build only weighted terms involving the newest site in ``site_count``."""
    out = {}
    new_site = int(site_count) - 1
    density_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}
    spin_density_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}
    pair_cache: dict[tuple[int, int], ReducedSU2Tensor] = {}

    for q in future_sites:
        density_terms = []
        exchange_density_terms = []
        exchange_spin_terms = []
        pair_terms = []
        v1_terms = []
        v3_terms = []

        coeff = h1e[new_site, q]
        if abs(coeff) > 0.0:
            v1_terms.append(scale_reduced_tensor(operators[("Cdag", new_site)], coeff))

        coeff = eri[new_site, q, q, q]
        if abs(coeff) > 0.0:
            v3_terms.append(scale_reduced_tensor(operators[("Cdag", new_site)], coeff))

        for i in range(site_count):
            for j in range(site_count):
                if i != new_site and j != new_site:
                    continue

                coeff = eri[i, j, q, q]
                if abs(coeff) > 0.0:
                    density = density_cache.setdefault((i, j), reduced_density_tensor(operators, i, j))
                    density_terms.append(scale_reduced_tensor(density, coeff))

                coeff = eri[i, q, q, j]
                if abs(coeff) > 0.0:
                    density = density_cache.setdefault((i, j), reduced_density_tensor(operators, i, j))
                    spin_density = spin_density_cache.setdefault((i, j), reduced_spin_density_tensor(operators, i, j))
                    exchange_density_terms.append(scale_reduced_tensor(density, coeff))
                    exchange_spin_terms.append(scale_reduced_tensor(spin_density, coeff))

                coeff = eri[q, i, q, j]
                if abs(coeff) > 0.0:
                    pair = pair_cache.setdefault((i, j), reduced_pair_annihilate_tensor(operators, i, j))
                    pair_terms.append(scale_reduced_tensor(pair, coeff))

        for k in range(site_count):
            weighted_density_terms = []
            for i in range(site_count):
                for j in range(site_count):
                    if i != new_site and j != new_site and k != new_site:
                        continue
                    coeff = eri[k, q, j, i]
                    if abs(coeff) > 0.0:
                        density = density_cache.setdefault((j, i), reduced_density_tensor(operators, j, i))
                        weighted_density_terms.append(scale_reduced_tensor(density, coeff))
            weighted_density = add_optional_reduced_terms(weighted_density_terms)
            if weighted_density is not None:
                v1_terms.append(
                    coupled_reduced_product(
                        operators[("Cdag", k)],
                        weighted_density,
                        rank2=1,
                    )
                )

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
) -> None:
    """Attach initial weighted packages without storing individual composites."""
    block.reduced_operators.update(
        weighted_packages_from_operators(block.reduced_operators, h1e, eri, site_count, future_sites)
    )


@profile_function("update_future_weighted_packages")
def update_future_weighted_packages(
    narg: ThreeSiteSU2NARG,
    grown: dict[tuple[str, int], ReducedSU2Tensor],
    h1e,
    eri,
    future_sites,
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
) -> dict[tuple, ReducedSU2Tensor]:
    """Grow, optionally compose, and rotate reduced operators after truncation."""
    grown = grown_coupling_operators(narg)
    reduced = {
        key: rotate_reduced_tensor_to_truncated(truncated, tensor)
        for key, tensor in grown.items()
    }
    weighted = update_future_weighted_packages(narg, grown, h1e, eri, tuple(future_sites)) if h1e is not None else {}
    for key, tensor in weighted.items():
        reduced[key] = rotate_reduced_tensor_to_truncated(truncated, tensor)
    return reduced


def feasible_nelec_for_target(block_sites: int, final_sites: int, target_nelec: int) -> set[int]:
    """Particle-number sectors that can still reach ``target_nelec``."""
    remaining = int(final_sites) - int(block_sites)
    lo = max(0, int(target_nelec) - 2 * remaining)
    hi = min(2 * int(block_sites), int(target_nelec))
    return set(range(lo, hi + 1))


@profile_function("renormalized_block_from_narg")
def renormalized_block_from_narg(
    narg: ThreeSiteSU2NARG,
    h1e_block,
    eri_block,
    D: int,
    *,
    allowed_nelec: set[int] | None = None,
    h1e_full=None,
    eri_full=None,
    future_sites=(),
) -> RenormalizedSU2Block:
    """Truncate a grown SU2-NARG object and attach reduced coupling operators."""
    with profile_section("truncate_to_D"):
        truncated = truncate_to_D(narg, D=D, allowed_nelec=allowed_nelec)
    with profile_section("retained_multiplets_after_truncate"):
        multiplets = []
    reduced_operators = reduced_coupling_operators_from_growth(
        narg,
        truncated,
        h1e=h1e_full,
        eri=eri_full,
        future_sites=future_sites,
    )
    block = RenormalizedSU2Block(
        truncated=truncated,
        hamiltonian=truncated.hamiltonian,
        transform=truncated.transform,
        operators={},
        reduced_operators=reduced_operators,
        parity=None,
    )
    block._su2_multiplets = multiplets
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
    D_by_size: dict[int, int],
    *,
    final_size: int | None = None,
    target_nelec: int | None = None,
) -> SU2ChainResult:
    """Grow a direct-reduced SU(2)-NARG chain.

    ``D_by_size[n]`` is used when an ``n``-site block must be retained for the
    next growth step.  The final size is not truncated unless it also serves as
    an intermediate block in a longer run.
    """
    final_size = h1e.shape[0] if final_size is None else int(final_size)
    target_nelec = final_size if target_nelec is None else int(target_nelec)
    if final_size < 2:
        raise ValueError("final_size must be at least 2")

    timings = {}
    blocks: dict[int, RenormalizedSU2Block] = {}
    reset_su2_profile()

    start = time.perf_counter()
    d2 = int(D_by_size.get(2, 10))
    blocks[2] = build_renormalized_two_site_block(h1e[:2, :2], eri[:2, :2, :2, :2], D=d2)
    seed_future_weighted_packages(blocks[2], h1e, eri, 2, tuple(range(2, final_size)))
    timings["build_block_2"] = time.perf_counter() - start

    source = blocks[2]
    final = None
    for nsites in range(3, final_size + 1):
        h1e_n = h1e[:nsites, :nsites]
        eri_n = eri[:nsites, :nsites, :nsites, :nsites]

        start = time.perf_counter()
        final = grow_one_site_direct_reduced(
            h1e_n,
            eri_n,
            source,
            target_nelec=target_nelec if nsites == final_size else None,
            build_branch_basis=False,
        )
        timings[f"grow_{nsites}"] = time.perf_counter() - start

        if nsites < final_size:
            start = time.perf_counter()
            blocks[nsites] = renormalized_block_from_narg(
                final,
                h1e_n,
                eri_n,
                D=int(D_by_size[nsites]),
                allowed_nelec=feasible_nelec_for_target(nsites, final_size, target_nelec),
                h1e_full=h1e,
                eri_full=eri,
                future_sites=tuple(range(nsites, final_size)),
            )
            timings[f"renormalize_{nsites}"] = time.perf_counter() - start
            source = blocks[nsites]

    if final is None:
        raise RuntimeError("chain growth did not produce a final NARG object")
    profile = su2_profile_snapshot()
    if profile:
        timings["profile"] = profile
    return SU2ChainResult(final=final, blocks=blocks, timings=timings)


def diagonalize_block(narg: ThreeSiteSU2NARG, nelec: int, j2: int, nroots: int = 6):
    """Diagonalize one final SU2 sector."""
    irrep = Irrep((nelec, j2))
    block = narg.hamiltonian.block(irrep, irrep)
    if block.size == 0:
        return np.array([]), block
    evals = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
    return evals[:nroots], block
