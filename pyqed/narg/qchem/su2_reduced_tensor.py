#!/usr/bin/env python3
"""Reduced SU(2) tensor operators for the NARG prototypes.

An ordinary component operator stores matrices for a chosen magnetic component,
for example ``Cdu`` or ``Cdd``.  A reduced SU(2) tensor stores one block per
``(Ne, j2)`` sector pair and recovers all magnetic components from
Clebsch-Gordan coefficients:

    <Jb Mb alpha_b | T^K_Q | Jk Mk alpha_k>
      = CG(Jk Mk, K Q -> Jb Mb) / sqrt(2*Jb + 1)
        * <Jb alpha_b || T^K || Jk alpha_k>

The integer labels use doubled quantum numbers:
``j2 = 2*J``, ``m2 = 2*M``, ``rank2 = 2*K``, ``q2 = 2*Q``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.narg.irrep_tensor import Irrep, IrrepSite, IrrepTensor, OpIrrep
from .su2_core import Multiplet, asarray, cg, su2_product_symmetry


@dataclass
class ReducedSU2Tensor:
    """Reduced matrix elements for a U(1)xSU(2) tensor operator."""

    tensor: IrrepTensor

    @property
    def op(self) -> OpIrrep:
        return self.tensor.op

    @property
    def site(self) -> IrrepSite:
        return self.tensor.bra

    @property
    def blocks(self):
        return self.tensor.blocks

    def block(self, bra: Irrep, ket: Irrep) -> np.ndarray:
        return self.tensor.block(bra, ket)


def group_multiplets(multiplets: list[Multiplet]) -> dict[Irrep, list[Multiplet]]:
    groups: dict[Irrep, list[Multiplet]] = {}
    for mp in multiplets:
        groups.setdefault(Irrep((mp.nelec, mp.j2)), []).append(mp)
    return dict(sorted(groups.items(), key=lambda item: item[0].charge))


def site_from_multiplets(multiplets: list[Multiplet]) -> IrrepSite:
    groups = group_multiplets(multiplets)
    return IrrepSite(su2_product_symmetry(), {irrep: len(mps) for irrep, mps in groups.items()})


def component_basis(groups: dict[Irrep, list[Multiplet]], irrep: Irrep, m2: int) -> np.ndarray:
    cols = []
    for mp in groups.get(irrep, ()):
        vec = mp.states.get(m2)
        if vec is None:
            raise ValueError(f"missing m2={m2} component for {irrep}")
        cols.append(vec)
    if not cols:
        return np.zeros((0, 0), dtype=complex)
    return np.column_stack(cols)


def reduced_tensor_from_components(
    multiplets: list[Multiplet],
    component_ops: dict[int, np.ndarray],
    op_irrep: OpIrrep,
    *,
    atol: float = 1e-12,
) -> ReducedSU2Tensor:
    """Extract reduced matrix elements from explicit component operators.

    ``component_ops`` maps component ``q2`` to a primitive-basis matrix.
    """
    groups = group_multiplets(multiplets)
    site = site_from_multiplets(multiplets)
    dnelec, rank2 = op_irrep.charge
    dense_component_ops = {q2: asarray(op) for q2, op in component_ops.items()}
    basis_cache = {}
    action_cache = {}

    def cached_component_basis(irrep: Irrep, m2: int) -> np.ndarray:
        key = (irrep, m2)
        basis = basis_cache.get(key)
        if basis is None:
            basis = component_basis(groups, irrep, m2)
            basis_cache[key] = basis
        return basis

    def cached_action(q2: int, ket_irrep: Irrep, ket_m2: int) -> np.ndarray:
        key = (q2, ket_irrep, ket_m2)
        action = action_cache.get(key)
        if action is None:
            action = dense_component_ops[q2] @ cached_component_basis(ket_irrep, ket_m2)
            action_cache[key] = action
        return action

    blocks = {}

    for bra_irrep in site.irreps:
        bra_nelec, bra_j2 = bra_irrep.charge
        for ket_irrep in site.irreps:
            ket_nelec, ket_j2 = ket_irrep.charge
            if bra_nelec != ket_nelec + dnelec:
                continue
            if not site.symmetry.allows(bra_irrep.charge, op_irrep.charge, ket_irrep.charge):
                continue

            estimates = []
            for q2 in dense_component_ops:
                for ket_m2 in range(-ket_j2, ket_j2 + 1, 2):
                    bra_m2 = ket_m2 + q2
                    if bra_m2 < -bra_j2 or bra_m2 > bra_j2:
                        continue
                    coeff = cg(ket_j2, ket_m2, rank2, q2, bra_j2, bra_m2)
                    if abs(coeff) <= atol:
                        continue
                    bra_basis = cached_component_basis(bra_irrep, bra_m2)
                    component_block = bra_basis.conj().T @ cached_action(q2, ket_irrep, ket_m2)
                    estimates.append(component_block * np.sqrt(bra_j2 + 1.0) / coeff)

            if not estimates:
                continue
            block = sum(estimates) / len(estimates)
            if np.any(np.abs(block) > atol):
                blocks[(bra_irrep, ket_irrep)] = block

    return ReducedSU2Tensor(IrrepTensor(site, site, op_irrep, blocks))


def reconstruct_component_block(
    reduced: ReducedSU2Tensor,
    bra_irrep: Irrep,
    ket_irrep: Irrep,
    ket_m2: int,
    q2: int,
) -> np.ndarray:
    """Reconstruct one component matrix block from reduced matrix elements."""
    _, bra_j2 = bra_irrep.charge
    _, ket_j2 = ket_irrep.charge
    _, rank2 = reduced.op.charge
    bra_m2 = ket_m2 + q2
    coeff = cg(ket_j2, ket_m2, rank2, q2, bra_j2, bra_m2)
    return coeff * reduced.block(bra_irrep, ket_irrep) / np.sqrt(bra_j2 + 1.0)


def scale_reduced_tensor(tensor: ReducedSU2Tensor, factor: complex) -> ReducedSU2Tensor:
    """Scale all reduced matrix-element blocks."""
    blocks = {
        key: factor * block
        for key, block in tensor.blocks.items()
        if np.any(np.abs(factor * block) > 0.0)
    }
    return ReducedSU2Tensor(IrrepTensor(tensor.site, tensor.site, tensor.op, blocks))


def add_reduced_tensors(*tensors: ReducedSU2Tensor, atol: float = 1e-14) -> ReducedSU2Tensor:
    """Add reduced tensors with identical sites and operator irreps."""
    if not tensors:
        raise ValueError("at least one reduced tensor is required")
    site = tensors[0].site
    op = tensors[0].op
    for tensor in tensors:
        if tensor.site != site or tensor.op != op:
            raise ValueError("reduced tensor site/op mismatch")

    blocks = {}
    keys = set().union(*(tensor.blocks.keys() for tensor in tensors))
    for key in keys:
        block = sum(tensor.block(*key) for tensor in tensors)
        if np.any(np.abs(block) > atol):
            blocks[key] = block
    return ReducedSU2Tensor(IrrepTensor(site, site, op, blocks))


def coupled_reduced_product(
    left: ReducedSU2Tensor,
    right: ReducedSU2Tensor,
    rank2: int,
    *,
    atol: float = 1e-12,
) -> ReducedSU2Tensor:
    """Compose two reduced tensors into ``[left x right]^rank``.

    The product means ordinary operator composition with ``right`` acting first:
    ``left @ right``.  The angular part is contracted explicitly with
    Clebsch-Gordan coefficients, while multiplicity blocks are multiplied in
    reduced space.
    """
    if left.site != right.site:
        raise ValueError("left and right reduced tensors must use the same site")

    site = left.site
    left_dnelec, left_rank2 = left.op.charge
    right_dnelec, right_rank2 = right.op.charge
    op_irrep = OpIrrep((left_dnelec + right_dnelec, int(rank2)))
    blocks = {}

    for bra_irrep in site.irreps:
        bra_nelec, bra_j2 = bra_irrep.charge
        for ket_irrep in site.irreps:
            ket_nelec, ket_j2 = ket_irrep.charge
            if bra_nelec != ket_nelec + op_irrep.charge[0]:
                continue
            if not site.symmetry.allows(bra_irrep.charge, op_irrep.charge, ket_irrep.charge):
                continue

            estimates = []
            for total_q2 in range(-rank2, rank2 + 1, 2):
                for ket_m2 in range(-ket_j2, ket_j2 + 1, 2):
                    bra_m2 = ket_m2 + total_q2
                    if bra_m2 < -bra_j2 or bra_m2 > bra_j2:
                        continue
                    out_coeff = cg(ket_j2, ket_m2, rank2, total_q2, bra_j2, bra_m2)
                    if abs(out_coeff) <= atol:
                        continue

                    component_block = np.zeros(
                        (site.sector_dim(bra_irrep), site.sector_dim(ket_irrep)),
                        dtype=complex,
                    )
                    for mid_irrep in site.irreps:
                        mid_nelec, mid_j2 = mid_irrep.charge
                        if mid_nelec != ket_nelec + right_dnelec:
                            continue
                        if not site.symmetry.allows(mid_irrep.charge, right.op.charge, ket_irrep.charge):
                            continue
                        if not site.symmetry.allows(bra_irrep.charge, left.op.charge, mid_irrep.charge):
                            continue

                        left_block = left.block(bra_irrep, mid_irrep)
                        right_block = right.block(mid_irrep, ket_irrep)
                        if left_block.size == 0 or right_block.size == 0:
                            continue
                        block_product = (
                            left_block
                            @ right_block
                            / np.sqrt((bra_j2 + 1.0) * (mid_j2 + 1.0))
                        )
                        for right_q2 in range(-right_rank2, right_rank2 + 1, 2):
                            left_q2 = total_q2 - right_q2
                            if left_q2 < -left_rank2 or left_q2 > left_rank2:
                                continue
                            tensor_coeff = cg(
                                left_rank2,
                                left_q2,
                                right_rank2,
                                right_q2,
                                rank2,
                                total_q2,
                            )
                            if abs(tensor_coeff) <= atol:
                                continue
                            mid_m2 = ket_m2 + right_q2
                            if mid_m2 < -mid_j2 or mid_m2 > mid_j2:
                                continue
                            left_coeff = cg(
                                mid_j2,
                                mid_m2,
                                left_rank2,
                                left_q2,
                                bra_j2,
                                bra_m2,
                            )
                            right_coeff = cg(
                                ket_j2,
                                ket_m2,
                                right_rank2,
                                right_q2,
                                mid_j2,
                                mid_m2,
                            )
                            if abs(left_coeff) <= atol or abs(right_coeff) <= atol:
                                continue
                            component_block += (
                                tensor_coeff
                                * left_coeff
                                * right_coeff
                                * block_product
                            )

                    if np.any(np.abs(component_block) > atol):
                        estimates.append(component_block * np.sqrt(bra_j2 + 1.0) / out_coeff)

            if estimates:
                block = sum(estimates) / len(estimates)
                if np.any(np.abs(block) > atol):
                    blocks[(bra_irrep, ket_irrep)] = block

    return ReducedSU2Tensor(IrrepTensor(site, site, op_irrep, blocks))


def validate_reduced_tensor_components(
    multiplets: list[Multiplet],
    reduced: ReducedSU2Tensor,
    component_ops: dict[int, np.ndarray],
    *,
    atol: float = 1e-12,
) -> dict[str, float]:
    """Compare all reconstructed components with explicit component matrices."""
    groups = group_multiplets(multiplets)
    errors = {}
    dnelec, rank2 = reduced.op.charge

    for bra_irrep in reduced.site.irreps:
        bra_nelec, bra_j2 = bra_irrep.charge
        for ket_irrep in reduced.site.irreps:
            ket_nelec, ket_j2 = ket_irrep.charge
            if bra_nelec != ket_nelec + dnelec:
                continue
            if not reduced.site.symmetry.allows(bra_irrep.charge, reduced.op.charge, ket_irrep.charge):
                continue
            if reduced.block(bra_irrep, ket_irrep).size == 0:
                continue

            for q2, op in component_ops.items():
                op = asarray(op)
                for ket_m2 in range(-ket_j2, ket_j2 + 1, 2):
                    bra_m2 = ket_m2 + q2
                    if bra_m2 < -bra_j2 or bra_m2 > bra_j2:
                        continue
                    coeff = cg(ket_j2, ket_m2, rank2, q2, bra_j2, bra_m2)
                    if abs(coeff) <= atol:
                        continue
                    bra_basis = component_basis(groups, bra_irrep, bra_m2)
                    ket_basis = component_basis(groups, ket_irrep, ket_m2)
                    actual = bra_basis.conj().T @ op @ ket_basis
                    recon = reconstruct_component_block(reduced, bra_irrep, ket_irrep, ket_m2, q2)
                    key = f"{ket_irrep.charge},m2={ket_m2};q2={q2}->{bra_irrep.charge}"
                    errors[key] = float(np.linalg.norm(actual - recon))
    return errors


def max_error(errors: dict[str, float]) -> float:
    return max(errors.values()) if errors else 0.0
