#!/usr/bin/env python3
"""Two-site SU(2)-NARG prototype.

This is the smallest useful SU(2)-NARG step:

1. Start from a one-orbital block represented by SU(2) multiplets.
2. Add one spatial orbital using the three local SU(2) branches:
   empty, single spinor, double.
3. Build the two-site Hamiltonian blocks in the coupled ``(Ne, j2)`` basis.
4. Diagonalize scalar blocks, truncate to the lowest ``D`` SU(2) multiplets,
   and compare with the exact SU(2) projection.

The Hamiltonian is still projected from the exact two-site determinant
Hamiltonian.  The point is to make the future NARG branch and truncation data
structures explicit before porting the renormalized operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from pyqed.narg.irrep_tensor import Irrep, IrrepSite, IrrepTensor, OpIrrep, spin_label
from .su2_core import (
    Multiplet,
    asarray,
    build_site_csf_multiplets,
    full_jw_model,
    local_su2_branches,
    qchem_integrals,
    scalar_hamiltonian_irrep_tensor,
    su2_branch_update,
    su2_irrep_tensor_roots,
    su2_product_symmetry,
)
from .su2_reduced_tensor import (
    ReducedSU2Tensor,
    max_error,
    reduced_tensor_from_components,
    validate_reduced_tensor_components,
)


@dataclass(frozen=True)
class BranchMultiplet:
    """A coupled multiplet plus the local SU(2) branch that produced it."""

    branch: str
    multiplet: Multiplet


@dataclass
class TwoSiteSU2NARG:
    """Untruncated two-site SU(2)-NARG data."""

    branch_states: list[BranchMultiplet]
    site: IrrepSite
    bases: dict[Irrep, np.ndarray]
    provenance: dict[Irrep, list[BranchMultiplet]]
    hamiltonian: IrrepTensor


@dataclass(frozen=True)
class SectorRoot:
    """One eigenmultiplet in a scalar SU(2) sector block."""

    energy: float
    irrep: Irrep
    local_index: int
    vector: np.ndarray


@dataclass
class TruncatedSU2NARG:
    """A truncated SU(2)-NARG block in the retained eigenmultiplet basis."""

    source: TwoSiteSU2NARG
    kept_roots: list[SectorRoot]
    site: IrrepSite
    bases: dict[Irrep, np.ndarray]
    transform: IrrepTensor
    hamiltonian: IrrepTensor


@dataclass
class RenormalizedSU2Block:
    """Truncated block plus renormalized operators needed by the next site."""

    truncated: TruncatedSU2NARG
    hamiltonian: IrrepTensor
    transform: IrrepTensor
    operators: dict[tuple[str, int], IrrepTensor]
    reduced_operators: dict[tuple[str, int], ReducedSU2Tensor]
    parity: IrrepTensor


def two_site_branch_states(block_multiplets: list[Multiplet]) -> list[BranchMultiplet]:
    """Apply one SU(2) local-site update and keep branch provenance."""
    grouped = su2_branch_update(block_multiplets, local_su2_branches())
    states: list[BranchMultiplet] = []
    for branch in local_su2_branches():
        states.extend(BranchMultiplet(branch.name, mp) for mp in grouped[branch.name])
    return states


def branch_sector_counts(states: list[BranchMultiplet]) -> dict[str, dict[tuple[int, int], int]]:
    """Count ``(Ne, j2)`` sectors branch by branch."""
    out: dict[str, dict[tuple[int, int], int]] = {}
    for state in states:
        key = (state.multiplet.nelec, state.multiplet.j2)
        branch_counts = out.setdefault(state.branch, {})
        branch_counts[key] = branch_counts.get(key, 0) + 1
    return {branch: dict(sorted(counts.items())) for branch, counts in out.items()}


def basis_from_branch_states(
    states: list[BranchMultiplet], nelec: int, j2: int, m2: int = 0
) -> tuple[np.ndarray, list[BranchMultiplet]]:
    """Primitive-basis columns for a target ``(Ne, j2, m2)`` sector."""
    cols = []
    kept = []
    for state in states:
        mp = state.multiplet
        vec = mp.states.get(m2)
        if mp.nelec == nelec and mp.j2 == j2 and vec is not None:
            cols.append(vec)
            kept.append(state)
    if not cols:
        return np.zeros((16, 0)), kept

    basis = np.column_stack(cols)
    gram = basis.conj().T @ basis
    err = np.max(np.abs(gram - np.eye(gram.shape[0])))
    if err > 1e-10:
        raise ValueError(f"SU2-NARG branch basis is not orthonormal; max Gram error {err:g}")
    return basis, kept


def component_vector(mp: Multiplet, m2: int | None = None) -> np.ndarray | None:
    """Choose a concrete magnetic component for a multiplet basis vector.

    ``m2=None`` means use the highest-weight component.  That gives one
    representative vector for every integer and half-integer SU(2) irrep.
    """
    selected_m2 = mp.j2 if m2 is None else m2
    return mp.states.get(selected_m2)


def branch_irrep_site(
    states: list[BranchMultiplet], m2: int | None = None
) -> tuple[IrrepSite, dict[Irrep, np.ndarray], dict[Irrep, list[BranchMultiplet]]]:
    """Build an IrrepSite from branch-generated two-site multiplets."""
    sectors: dict[Irrep, list[np.ndarray]] = {}
    provenance: dict[Irrep, list[BranchMultiplet]] = {}
    for state in states:
        vec = component_vector(state.multiplet, m2=m2)
        if vec is None:
            continue
        irrep = Irrep((state.multiplet.nelec, state.multiplet.j2))
        sectors.setdefault(irrep, []).append(vec)
        provenance.setdefault(irrep, []).append(state)

    dims = {irrep: len(cols) for irrep, cols in sectors.items()}
    bases = {irrep: np.column_stack(cols) for irrep, cols in sectors.items()}
    return IrrepSite(su2_product_symmetry(), dims), bases, provenance


def build_two_site_su2_narg(h1e, eri, m2: int | None = None) -> TwoSiteSU2NARG:
    """Construct the untruncated two-site SU(2)-NARG Hamiltonian."""
    block_multiplets = build_site_csf_multiplets(1)
    branch_states = two_site_branch_states(block_multiplets)
    site, bases, provenance = branch_irrep_site(branch_states, m2=m2)
    model = full_jw_model(h1e, eri, nelec=2)
    hamiltonian = scalar_hamiltonian_irrep_tensor(model.H, site, bases)
    return TwoSiteSU2NARG(branch_states, site, bases, provenance, hamiltonian)


def diagonalize_sector(narg: TwoSiteSU2NARG, nelec: int, j2: int, nroots: int = 8):
    """Diagonalize one scalar ``(Ne, j2)`` Hamiltonian block."""
    irrep = Irrep((nelec, j2))
    block = narg.hamiltonian.block(irrep, irrep)
    evals, evecs = eigh(block)
    return evals[:nroots], evecs[:, :nroots], block


def diagonalize_all_sectors(
    narg: TwoSiteSU2NARG,
    *,
    allowed_nelec: set[int] | None = None,
    allowed_irreps: set[Irrep] | None = None,
) -> list[SectorRoot]:
    """Diagonalize every allowed scalar sector and sort roots by energy."""
    roots: list[SectorRoot] = []
    for irrep in narg.site.irreps:
        nelec, _ = irrep.charge
        if allowed_nelec is not None and nelec not in allowed_nelec:
            continue
        if allowed_irreps is not None and irrep not in allowed_irreps:
            continue
        block = narg.hamiltonian.block(irrep, irrep)
        if block.size == 0:
            continue
        evals, evecs = eigh(block)
        for local_index, energy in enumerate(evals):
            roots.append(
                SectorRoot(
                    energy=float(np.real(energy)),
                    irrep=irrep,
                    local_index=local_index,
                    vector=evecs[:, local_index].copy(),
                )
            )

    roots.sort(key=lambda root: (root.energy, root.irrep.charge, root.local_index))
    return roots


def truncate_to_D(
    narg: TwoSiteSU2NARG,
    D: int,
    *,
    allowed_nelec: set[int] | None = None,
    allowed_irreps: set[Irrep] | None = None,
) -> TruncatedSU2NARG:
    """Keep the lowest ``D`` SU(2) eigenmultiplets and rebuild block data."""
    kept = diagonalize_all_sectors(
        narg, allowed_nelec=allowed_nelec, allowed_irreps=allowed_irreps
    )[: int(D)]
    grouped: dict[Irrep, list[SectorRoot]] = {}
    for root in kept:
        grouped.setdefault(root.irrep, []).append(root)

    dims = {irrep: len(roots) for irrep, roots in grouped.items()}
    bases = {}
    blocks = {}
    transform_blocks = {}
    for irrep, roots in grouped.items():
        transform_block = np.column_stack([root.vector for root in roots])
        transform_blocks[(irrep, irrep)] = transform_block
        source_basis = narg.bases.get(irrep)
        if source_basis is not None:
            primitive_cols = [source_basis @ root.vector for root in roots]
            bases[irrep] = np.column_stack(primitive_cols)
        blocks[(irrep, irrep)] = np.diag([root.energy for root in roots])

    site = IrrepSite(su2_product_symmetry(), dims)
    transform = IrrepTensor(narg.site, site, OpIrrep((0, 0)), transform_blocks)
    hamiltonian = IrrepTensor(site, site, OpIrrep((0, 0)), blocks)
    return TruncatedSU2NARG(narg, kept, site, bases, transform, hamiltonian)


def root_branch_weights(narg: TwoSiteSU2NARG, root: SectorRoot) -> dict[str, float]:
    """Resolve a retained eigenmultiplet back onto local SU(2) branches."""
    states = narg.provenance[root.irrep]
    weights: dict[str, float] = {}
    for coeff, state in zip(root.vector, states):
        weights[state.branch] = weights.get(state.branch, 0.0) + float(abs(coeff) ** 2)
    return {branch: weight for branch, weight in weights.items() if weight > 1e-12}


def retained_multiplets(truncated: TruncatedSU2NARG) -> list[Multiplet]:
    """Reconstruct full SU(2) multiplets from retained sector eigenvectors.

    The scalar Hamiltonian diagonalization is performed in one representative
    component, but each eigenvector is a multiplicity-space vector.  Reusing
    those coefficients for every ``m2`` component gives the full retained
    multiplet needed for adding the next site.
    """
    multiplets = []
    for root in truncated.kept_roots:
        if root.irrep not in truncated.source.provenance:
            nelec, j2 = root.irrep.charge
            multiplets.append(Multiplet(nelec=nelec, j2=j2, states={}))
            continue
        source_states = truncated.source.provenance[root.irrep]
        nelec, j2 = root.irrep.charge
        states = {}
        for m2 in range(-j2, j2 + 1, 2):
            vec = None
            for coeff, state in zip(root.vector, source_states):
                component = state.multiplet.states.get(m2)
                if component is None:
                    continue
                term = coeff * component
                vec = term if vec is None else vec + term
            if vec is not None:
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    states[m2] = vec / norm
        multiplets.append(Multiplet(nelec=nelec, j2=j2, states=states))
    return multiplets


def project_primitive_operator(
    narg: TwoSiteSU2NARG,
    op,
    op_irrep: OpIrrep,
    *,
    atol: float = 1e-12,
) -> IrrepTensor:
    """Project a primitive determinant operator into the source SU(2) basis.

    This is a component-level bridge for preparing the next-site coupling
    machinery.  Full SU(2) reduced operators will replace this projection later.
    """
    dense_op = asarray(op)
    blocks = {}
    for bra_irrep, bra_basis in narg.bases.items():
        for ket_irrep, ket_basis in narg.bases.items():
            if not narg.site.symmetry.allows(bra_irrep.charge, op_irrep.charge, ket_irrep.charge):
                continue
            block = bra_basis.conj().T @ dense_op @ ket_basis
            if np.any(np.abs(block) > atol):
                blocks[(bra_irrep, ket_irrep)] = block
    return IrrepTensor(narg.site, narg.site, op_irrep, blocks)


def project_primitive_operator_to_truncated(
    truncated: TruncatedSU2NARG,
    op,
    op_irrep: OpIrrep,
    *,
    atol: float = 1e-12,
) -> IrrepTensor:
    """Directly project a primitive operator into retained primitive bases."""
    dense_op = asarray(op)
    blocks = {}
    for bra_irrep, bra_basis in truncated.bases.items():
        for ket_irrep, ket_basis in truncated.bases.items():
            if not truncated.site.symmetry.allows(bra_irrep.charge, op_irrep.charge, ket_irrep.charge):
                continue
            block = bra_basis.conj().T @ dense_op @ ket_basis
            if np.any(np.abs(block) > atol):
                blocks[(bra_irrep, ket_irrep)] = block
    return IrrepTensor(truncated.site, truncated.site, op_irrep, blocks)


def rotate_operator_to_truncated(truncated: TruncatedSU2NARG, operator: IrrepTensor) -> IrrepTensor:
    """Rotate a source-basis operator with the truncation transform ``U``.

    If ``U`` maps retained states into the source branch basis, the rotated
    operator is ``U_bra.conj().T @ O @ U_ket`` block by block.
    """
    blocks = {}
    for bra_irrep in truncated.site.irreps:
        source_bra_dim = truncated.source.site.sector_dim(bra_irrep)
        if source_bra_dim == 0:
            continue
        u_bra = truncated.transform.block(bra_irrep, bra_irrep)
        for ket_irrep in truncated.site.irreps:
            source_ket_dim = truncated.source.site.sector_dim(ket_irrep)
            if source_ket_dim == 0:
                continue
            if not truncated.site.symmetry.allows(
                bra_irrep.charge, operator.op.charge, ket_irrep.charge
            ):
                continue
            old_block = operator.block(bra_irrep, ket_irrep)
            if old_block.size == 0:
                continue
            u_ket = truncated.transform.block(ket_irrep, ket_irrep)
            new_block = u_bra.conj().T @ old_block @ u_ket
            if np.any(np.abs(new_block) > 1e-12):
                blocks[(bra_irrep, ket_irrep)] = new_block
    return IrrepTensor(truncated.site, truncated.site, operator.op, blocks)


def primitive_parity_operator(nsites: int) -> np.ndarray:
    """Fermion parity/Jordan-Wigner string ``(-1)^N`` in primitive site order."""
    diag = np.empty(4**nsites)
    local_nelec = np.array([0, 1, 1, 2], dtype=int)
    for flat in range(4**nsites):
        digits = np.unravel_index(flat, (4,) * nsites)
        nelec = int(np.sum(local_nelec[list(digits)]))
        diag[flat] = -1.0 if nelec % 2 else 1.0
    return np.diag(diag)


def component_operator_specs() -> dict[str, OpIrrep]:
    """Component-level creation/annihilation operator irreps."""
    return {
        "Cdu": OpIrrep((1, 1)),
        "Cdd": OpIrrep((1, 1)),
        "Cu": OpIrrep((-1, 1)),
        "Cd": OpIrrep((-1, 1)),
    }


def build_renormalized_two_site_block(
    h1e,
    eri,
    D: int = 8,
    *,
    allowed_nelec: set[int] | None = None,
) -> RenormalizedSU2Block:
    """Build a truncated two-site block with rotated coupling operators."""
    if allowed_nelec is None:
        allowed_nelec = {0, 1, 2, 3, 4}

    narg = build_two_site_su2_narg(h1e, eri)
    truncated = truncate_to_D(narg, D=D, allowed_nelec=allowed_nelec)
    model = full_jw_model(h1e, eri, nelec=2)
    primitive_ops = {
        "Cdu": model.Cdu,
        "Cdd": model.Cdd,
        "Cu": model.Cu,
        "Cd": model.Cd,
    }

    operators = {}
    for name, op_irrep in component_operator_specs().items():
        for site_index, primitive_op in enumerate(primitive_ops[name]):
            source = project_primitive_operator(narg, primitive_op, op_irrep)
            operators[(name, site_index)] = rotate_operator_to_truncated(truncated, source)

    multiplets = retained_multiplets(truncated)
    reduced_operators = {}
    for site_index in range(2):
        reduced_operators[("Cdag", site_index)] = reduced_tensor_from_components(
            multiplets,
            {1: primitive_ops["Cdu"][site_index], -1: primitive_ops["Cdd"][site_index]},
            OpIrrep((1, 1)),
        )
        # The annihilation spinor is the SU(2)-covariant conjugate tensor:
        # T_{-1/2}=c_up, T_{+1/2}=-c_down.
        reduced_operators[("Ctilde", site_index)] = reduced_tensor_from_components(
            multiplets,
            {-1: primitive_ops["Cu"][site_index], 1: -primitive_ops["Cd"][site_index]},
            OpIrrep((-1, 1)),
        )

    parity_source = project_primitive_operator(narg, primitive_parity_operator(2), OpIrrep((0, 0)))
    parity = rotate_operator_to_truncated(truncated, parity_source)
    block = RenormalizedSU2Block(
        truncated=truncated,
        hamiltonian=truncated.hamiltonian,
        transform=truncated.transform,
        operators=operators,
        reduced_operators=reduced_operators,
        parity=parity,
    )
    block._su2_multiplets = multiplets
    block._su2_primitive_ops = primitive_ops
    return block


def irrep_tensor_difference_norm(left: IrrepTensor, right: IrrepTensor) -> float:
    """Max blockwise norm difference for tensors on matching sites."""
    keys = set(left.blocks) | set(right.blocks)
    err = 0.0
    for bra, ket in keys:
        diff = left.block(bra, ket) - right.block(bra, ket)
        if diff.size:
            err = max(err, float(np.linalg.norm(diff)))
    return err


def validate_rotated_operators_against_direct(block: RenormalizedSU2Block, h1e, eri) -> dict[str, float]:
    """Check ``U.conj().T @ O @ U`` against direct retained-basis projection."""
    model = full_jw_model(h1e, eri, nelec=2)
    primitive_ops = {
        "Cdu": model.Cdu,
        "Cdd": model.Cdd,
        "Cu": model.Cu,
        "Cd": model.Cd,
    }
    errors = {}
    for name, op_irrep in component_operator_specs().items():
        for site_index, primitive_op in enumerate(primitive_ops[name]):
            direct = project_primitive_operator_to_truncated(
                block.truncated, primitive_op, op_irrep
            )
            rotated = block.operators[(name, site_index)]
            errors[f"{name}[{site_index}]"] = irrep_tensor_difference_norm(rotated, direct)

    direct_parity = project_primitive_operator_to_truncated(
        block.truncated, primitive_parity_operator(2), OpIrrep((0, 0))
    )
    errors["parity"] = irrep_tensor_difference_norm(block.parity, direct_parity)
    return errors


def validate_adjoint_pairs(block: RenormalizedSU2Block) -> dict[str, float]:
    """Check creation/annihilation adjoint consistency after truncation."""
    return {
        f"Cdu[{i}] vs Cu[{i}]": irrep_tensor_difference_norm(
            block.operators[("Cdu", i)].adjoint(), block.operators[("Cu", i)]
        )
        for i in range(2)
    } | {
        f"Cdd[{i}] vs Cd[{i}]": irrep_tensor_difference_norm(
            block.operators[("Cdd", i)].adjoint(), block.operators[("Cd", i)]
        )
        for i in range(2)
    }


def validate_parity(block: RenormalizedSU2Block) -> dict[str, float]:
    """Check parity is Hermitian and squares to identity in retained sectors."""
    parity = block.parity
    identity = IrrepTensor.identity(block.truncated.site)
    return {
        "P-Pdag": irrep_tensor_difference_norm(parity, parity.adjoint()),
        "P2-I": irrep_tensor_difference_norm(parity.scalar_matmul(parity), identity),
    }


def validate_reduced_operators(block: RenormalizedSU2Block, h1e, eri) -> dict[str, float]:
    """Validate reduced tensors reconstruct their component operators."""
    model = full_jw_model(h1e, eri, nelec=2)
    multiplets = retained_multiplets(block.truncated)
    errors = {}
    for site_index in range(2):
        cdag = block.reduced_operators[("Cdag", site_index)]
        cdag_errors = validate_reduced_tensor_components(
            multiplets,
            cdag,
            {1: model.Cdu[site_index], -1: model.Cdd[site_index]},
        )
        errors[f"Cdag[{site_index}]"] = max_error(cdag_errors)

        ctilde = block.reduced_operators[("Ctilde", site_index)]
        ctilde_errors = validate_reduced_tensor_components(
            multiplets,
            ctilde,
            {-1: model.Cu[site_index], 1: -model.Cd[site_index]},
        )
        errors[f"Ctilde[{site_index}]"] = max_error(ctilde_errors)
    return errors


def print_branch_table(narg: TwoSiteSU2NARG) -> None:
    print("SU2-NARG local branch update:")
    for branch, counts in branch_sector_counts(narg.branch_states).items():
        pieces = [
            f"Ne={nelec} S={spin_label(j2)} dim={dim}"
            for (nelec, j2), dim in counts.items()
        ]
        print(f"  {branch}: " + "; ".join(pieces))


def print_sector_basis(narg: TwoSiteSU2NARG, nelec: int, j2: int) -> None:
    _, kept = basis_from_branch_states(narg.branch_states, nelec, j2)
    print(f"Sector Ne={nelec} S={spin_label(j2)} basis:")
    for col, state in enumerate(kept):
        print(f"  alpha={col} branch={state.branch}")


def print_truncated_block(truncated: TruncatedSU2NARG, enuc: float) -> None:
    print(f"Truncated SU2-NARG block: D={len(truncated.kept_roots)}")
    for idx, root in enumerate(truncated.kept_roots):
        nelec, j2 = root.irrep.charge
        weights = root_branch_weights(truncated.source, root)
        weight_text = ", ".join(f"{name}:{weight:.3f}" for name, weight in weights.items())
        print(
            f"  {idx:2d} E={root.energy + enuc: .10f} "
            f"Ne={nelec} S={spin_label(j2)}  branches[{weight_text}]"
        )
    print("  kept sectors:")
    for irrep, dim in truncated.site.dims.items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)} dim={dim}")


def print_transform_blocks(truncated: TruncatedSU2NARG) -> None:
    print("Truncation transform U blocks: source branch basis -> retained basis")
    for (bra, ket), block in truncated.transform.blocks.items():
        nelec, j2 = bra.charge
        print(f"  Ne={nelec} S={spin_label(j2)} U shape={block.shape}")


def print_operator_blocks(name: str, operator: IrrepTensor) -> None:
    print(f"{name} blocks:")
    if not operator.blocks:
        print("  <none>")
        return
    for (bra, ket), block in sorted(operator.blocks.items(), key=lambda item: (item[0][1].charge, item[0][0].charge)):
        bra_ne, bra_j2 = bra.charge
        ket_ne, ket_j2 = ket.charge
        print(
            f"  (Ne={ket_ne}, S={spin_label(ket_j2)}) -> "
            f"(Ne={bra_ne}, S={spin_label(bra_j2)}) "
            f"shape={block.shape} norm={np.linalg.norm(block):.6f}"
        )


def print_renormalized_operator_summary(block: RenormalizedSU2Block) -> None:
    print("Renormalized component operators:")
    for name in ("Cdu", "Cdd", "Cu", "Cd"):
        op_irrep = component_operator_specs()[name].charge
        print(f"  {name} op_irrep={op_irrep}")
        for site_index in range(2):
            operator = block.operators[(name, site_index)]
            nblocks = len(operator.blocks)
            norm = np.sqrt(sum(np.linalg.norm(b) ** 2 for b in operator.blocks.values()))
            print(f"    site {site_index}: blocks={nblocks} frob_norm={norm:.6f}")
    print(f"  parity blocks={len(block.parity.blocks)}")
    print("Reduced SU2 tensor operators:")
    for name in ("Cdag", "Ctilde"):
        op_irrep = block.reduced_operators[(name, 0)].op.charge
        print(f"  {name} op_irrep={op_irrep}")
        for site_index in range(2):
            operator = block.reduced_operators[(name, site_index)]
            nblocks = len(operator.blocks)
            norm = np.sqrt(sum(np.linalg.norm(b) ** 2 for b in operator.blocks.values()))
            print(f"    site {site_index}: reduced_blocks={nblocks} reduced_frob_norm={norm:.6f}")


def print_validation_errors(title: str, errors: dict[str, float]) -> None:
    print(title)
    for name, err in errors.items():
        print(f"  {name}: {err:.3e}")


def validate_two_site_su2_narg() -> None:
    mol, mf, h1e, eri = qchem_integrals(2, span=1.0, basis="sto6g")
    enuc = mol.energy_nuc()
    narg = build_two_site_su2_narg(h1e, eri)

    print("Two-site SU2-NARG prototype")
    print("spacing = 2.0 Bohr")
    print(f"RHF total energy = {mf.e_tot:.12f}")
    print(f"nuclear repulsion = {enuc:.12f}")
    print()
    print_branch_table(narg)
    print()

    for nelec, j2, name in [(2, 0, "singlet"), (2, 2, "triplet")]:
        print_sector_basis(narg, nelec, j2)
        roots, _, block = diagonalize_sector(narg, nelec, j2)
        ref_roots, _, _ = su2_irrep_tensor_roots(h1e, eri, nelec, j2, nroots=len(roots))
        diff = np.max(np.abs(roots - ref_roots)) if len(roots) else 0.0
        print(f"{name} block shape = {block.shape}")
        print(f"{name} total roots = {np.array2string(roots + enuc, precision=10, separator=', ')}")
        print(f"max electronic-energy diff vs exact SU2 block = {diff:.3e}")
        print()

    truncated = truncate_to_D(narg, D=2, allowed_nelec={2})
    print_truncated_block(truncated, enuc)
    print()

    operator_ready = truncate_to_D(narg, D=8, allowed_nelec={1, 2, 3})
    print("Operator-ready truncation for coupling to a third site:")
    print_truncated_block(operator_ready, enuc)
    print_transform_blocks(operator_ready)
    model = full_jw_model(h1e, eri, nelec=2)
    cdag_up_source = project_primitive_operator(narg, model.Cdu[0], OpIrrep((1, 1)))
    cdag_up_truncated = rotate_operator_to_truncated(operator_ready, cdag_up_source)
    print_operator_blocks("rotated Cdu[0] component", cdag_up_truncated)
    print()

    renormalized = build_renormalized_two_site_block(h1e, eri, D=10)
    print("Renormalized two-site block for site-3 coupling:")
    print_truncated_block(renormalized.truncated, enuc)
    print_renormalized_operator_summary(renormalized)
    print_validation_errors(
        "Rotation vs direct retained-basis projection errors:",
        validate_rotated_operators_against_direct(renormalized, h1e, eri),
    )
    print_validation_errors("Adjoint-pair errors:", validate_adjoint_pairs(renormalized))
    print_validation_errors("Parity errors:", validate_parity(renormalized))
    print_validation_errors(
        "Reduced SU2 tensor reconstruction errors:",
        validate_reduced_operators(renormalized, h1e, eri),
    )
    print()

    dense_dim = narg.hamiltonian.to_dense().shape[0]
    truncated_dim = truncated.hamiltonian.to_dense().shape[0]
    operator_ready_dim = operator_ready.hamiltonian.to_dense().shape[0]
    renormalized_dim = renormalized.hamiltonian.to_dense().shape[0]
    full_dim = asarray(full_jw_model(h1e, eri, nelec=2).H).shape[0]
    print(f"SU2-NARG IrrepTensor dense dim = {dense_dim}")
    print(f"Truncated SU2-NARG dense dim = {truncated_dim}")
    print(f"Operator-ready SU2-NARG dense dim = {operator_ready_dim}")
    print(f"Renormalized SU2 block dense dim = {renormalized_dim}")
    print(f"Primitive determinant dim = {full_dim}")


def main() -> None:
    validate_two_site_su2_narg()


if __name__ == "__main__":
    main()
