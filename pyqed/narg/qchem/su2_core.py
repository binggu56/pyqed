#!/usr/bin/env python3
"""Exact SU(2)-adapted prototype for small spinful quantum-chemistry chains.

This is a validation scaffold for a future SU(2)-NARG implementation.  It builds
spin-adapted configuration-state functions (CSFs) by recursively coupling block
multiplets with the local spinful site irreps, then projects the scalar
Hamiltonian into fixed (Ne, S, M=0) sectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isclose, sqrt
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from scipy.linalg import eigh

from pyqed.mps.fermion import SpinHalfFermionChain
from pyqed.qchem import Molecule, build_atom_from_coords

from pyqed.narg.irrep_tensor import (
    Irrep,
    Leg,
    IrrepTensor,
    OpIrrep,
    ProductSymmetry,
    SU2Symmetry,
    U1Symmetry,
    spin_label,
    u1_su2_site_from_spin,
)


@dataclass
class Multiplet:
    nelec: int
    j2: int
    states: dict[int, np.ndarray]


@dataclass(frozen=True)
class LocalIrrepBranch:
    name: str
    nelec: int
    j2: int
    multiplet: Multiplet


def _phase_from_doubled(exponent2: int) -> float:
    """Return ``(-1)**(exponent2/2)`` for integer exponents."""
    exponent = int(exponent2) // 2
    return -1.0 if exponent % 2 else 1.0


def _sqrt_ratio(numer: float, denom: float) -> float:
    if numer <= 0.0:
        return 0.0
    return sqrt(numer / denom)


def _cg_right_spin_half(left_j2: int, left_m2: int, right_m2: int, coupled_j2: int) -> float:
    """Analytic ``<j m; 1/2 q | J M>`` in doubled labels."""
    j2 = int(left_j2)
    m2 = int(left_m2)
    if int(right_m2) == 1:
        if coupled_j2 == j2 + 1:
            return _sqrt_ratio(j2 + m2 + 2.0, 2.0 * (j2 + 1.0))
        if coupled_j2 == j2 - 1:
            return -_sqrt_ratio(j2 - m2, 2.0 * (j2 + 1.0))
    elif int(right_m2) == -1:
        if coupled_j2 == j2 + 1:
            return _sqrt_ratio(j2 - m2 + 2.0, 2.0 * (j2 + 1.0))
        if coupled_j2 == j2 - 1:
            return _sqrt_ratio(j2 + m2, 2.0 * (j2 + 1.0))
    return 0.0


def _cg_right_spin_one(left_j2: int, left_m2: int, right_m2: int, coupled_j2: int) -> float:
    """Analytic ``<j m; 1 q | J M>`` in doubled labels."""
    j2 = int(left_j2)
    m2 = int(left_m2)
    q2 = int(right_m2)

    if coupled_j2 == j2 + 2:
        denom = 4.0 * (j2 + 1.0) * (j2 + 2.0)
        if q2 == 2:
            return _sqrt_ratio((j2 + m2 + 2.0) * (j2 + m2 + 4.0), denom)
        if q2 == 0:
            return _sqrt_ratio((j2 - m2 + 2.0) * (j2 + m2 + 2.0), 0.5 * denom)
        if q2 == -2:
            return _sqrt_ratio((j2 - m2 + 2.0) * (j2 - m2 + 4.0), denom)

    if coupled_j2 == j2 and j2 > 0:
        denom = 2.0 * j2 * (j2 + 2.0)
        if q2 == 2:
            return -_sqrt_ratio((j2 - m2) * (j2 + m2 + 2.0), denom)
        if q2 == 0:
            return m2 / sqrt(j2 * (j2 + 2.0))
        if q2 == -2:
            return _sqrt_ratio((j2 + m2) * (j2 - m2 + 2.0), denom)

    if coupled_j2 == j2 - 2 and j2 >= 2:
        denom = 4.0 * j2 * (j2 + 1.0)
        if q2 == 2:
            return _sqrt_ratio((j2 - m2) * (j2 - m2 - 2.0), denom)
        if q2 == 0:
            return -_sqrt_ratio((j2 - m2) * (j2 + m2), 0.5 * denom)
        if q2 == -2:
            return _sqrt_ratio((j2 + m2) * (j2 + m2 - 2.0), denom)

    return 0.0


@lru_cache(maxsize=None)
def cg(
    left_j2: int,
    left_m2: int,
    right_j2: int,
    right_m2: int,
    coupled_j2: int,
    coupled_m2: int,
) -> float:
    """Clebsch-Gordan coefficient using doubled integer quantum numbers.

    The SU(2)-NARG code only couples with local/operator ranks 0, 1/2, and 1.
    These analytic Condon-Shortley formulas avoid SymPy in the hot path.
    """
    left_j2 = int(left_j2)
    left_m2 = int(left_m2)
    right_j2 = int(right_j2)
    right_m2 = int(right_m2)
    coupled_j2 = int(coupled_j2)
    coupled_m2 = int(coupled_m2)

    if left_m2 + right_m2 != coupled_m2:
        return 0.0
    if abs(left_m2) > left_j2 or abs(right_m2) > right_j2 or abs(coupled_m2) > coupled_j2:
        return 0.0
    if (left_j2 - left_m2) % 2 or (right_j2 - right_m2) % 2 or (coupled_j2 - coupled_m2) % 2:
        return 0.0
    if coupled_j2 < abs(left_j2 - right_j2) or coupled_j2 > left_j2 + right_j2:
        return 0.0
    if (left_j2 + right_j2 + coupled_j2) % 2:
        return 0.0

    if right_j2 == 0:
        return 1.0 if coupled_j2 == left_j2 and coupled_m2 == left_m2 and right_m2 == 0 else 0.0
    if right_j2 == 1:
        return _cg_right_spin_half(left_j2, left_m2, right_m2, coupled_j2)
    if right_j2 == 2:
        return _cg_right_spin_one(left_j2, left_m2, right_m2, coupled_j2)

    if left_j2 in (0, 1, 2):
        phase = _phase_from_doubled(left_j2 + right_j2 - coupled_j2)
        return phase * cg(right_j2, right_m2, left_j2, left_m2, coupled_j2, coupled_m2)

    raise NotImplementedError(
        "fast cg only supports coupling against SU(2) rank/spin 0, 1/2, or 1"
    )


def local_site_multiplets() -> list[Multiplet]:
    """Local spinful site irreps in basis [empty, up, down, full]."""
    eye = np.eye(4)
    return [
        Multiplet(nelec=0, j2=0, states={0: eye[:, 0]}),
        Multiplet(nelec=1, j2=1, states={1: eye[:, 1], -1: eye[:, 2]}),
        Multiplet(nelec=2, j2=0, states={0: eye[:, 3]}),
    ]


def local_su2_branches() -> tuple[LocalIrrepBranch, ...]:
    """The three SU(2) local branches replacing ``0/up/down/ud``."""
    empty, single, double = local_site_multiplets()
    return (
        LocalIrrepBranch("empty", empty.nelec, empty.j2, empty),
        LocalIrrepBranch("single", single.nelec, single.j2, single),
        LocalIrrepBranch("double", double.nelec, double.j2, double),
    )


def local_su2_site() -> Leg:
    """One spatial orbital as U(1)xSU(2) irreps.

    Dims are multiplicity dimensions.  The spin multiplet dimension ``2S+1`` is
    carried by SU(2) representation theory, not by this degeneracy index.
    """
    return u1_su2_site_from_spin([
        (0, 0, 1),
        (1, "1/2", 1),
        (2, 0, 1),
    ])


def couple_multiplets(left: Multiplet, right: Multiplet) -> list[Multiplet]:
    """Couple two multiplets and return all allowed total-spin multiplets."""
    out = []
    for j2 in range(abs(left.j2 - right.j2), left.j2 + right.j2 + 1, 2):
        states = {}
        for m2 in range(-j2, j2 + 1, 2):
            vec = None
            for left_m2, left_vec in left.states.items():
                right_m2 = m2 - left_m2
                right_vec = right.states.get(right_m2)
                if right_vec is None:
                    continue
                coeff = cg(left.j2, left_m2, right.j2, right_m2, j2, m2)
                if isclose(coeff, 0.0, abs_tol=1e-14):
                    continue
                term = coeff * np.kron(left_vec, right_vec)
                vec = term if vec is None else vec + term
            if vec is not None:
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    states[m2] = vec / norm
        if states:
            out.append(Multiplet(left.nelec + right.nelec, j2, states))
    return out


def su2_branch_update(
    block_multiplets: list[Multiplet],
    branches: tuple[LocalIrrepBranch, ...] | None = None,
) -> dict[str, list[Multiplet]]:
    """Couple a block basis to one local site, grouped by SU(2) local branch."""
    if branches is None:
        branches = local_su2_branches()
    grouped = {}
    for branch in branches:
        coupled = []
        for block in block_multiplets:
            coupled.extend(couple_multiplets(block, branch.multiplet))
        grouped[branch.name] = coupled
    return grouped


def flatten_branch_update(branch_groups: dict[str, list[Multiplet]]) -> list[Multiplet]:
    """Return branch states in deterministic local-branch order."""
    out = []
    for branch in local_su2_branches():
        out.extend(branch_groups.get(branch.name, ()))
    return out


def build_site_csf_multiplets(nsites: int) -> list[Multiplet]:
    """Recursively build a complete SU(2)-adapted CSF basis."""
    multiplets = [Multiplet(nelec=0, j2=0, states={0: np.array([1.0])})]
    for _ in range(nsites):
        multiplets = flatten_branch_update(su2_branch_update(multiplets))
    return multiplets


def multiplet_counts(multiplets: list[Multiplet]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for mp in multiplets:
        key = (mp.nelec, mp.j2)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def su2_sector_counts(nsites: int) -> dict[tuple[int, int], int]:
    """Multiplicity count of SU(2) irreps after coupling ``nsites`` orbitals."""
    return multiplet_counts(build_site_csf_multiplets(nsites))


def validate_two_site_sector_counts() -> dict[tuple[int, int], int]:
    """Check the first nontrivial SU(2) product basis: one site times one site."""
    expected = {
        (0, 0): 1,
        (1, 1): 2,
        (2, 0): 3,
        (2, 2): 1,
        (3, 1): 2,
        (4, 0): 1,
    }
    actual = su2_sector_counts(2)
    if actual != expected:
        raise AssertionError(f"two-site SU2 sector counts differ: {actual} != {expected}")
    return actual


def validate_two_site_branch_update() -> dict[str, dict[tuple[int, int], int]]:
    """Check one-site block plus local SU(2) branches before flattening."""
    branch_counts = {
        name: multiplet_counts(multiplets)
        for name, multiplets in su2_branch_update(build_site_csf_multiplets(1)).items()
    }
    expected = {
        "empty": {(0, 0): 1, (1, 1): 1, (2, 0): 1},
        "single": {(1, 1): 1, (2, 0): 1, (2, 2): 1, (3, 1): 1},
        "double": {(2, 0): 1, (3, 1): 1, (4, 0): 1},
    }
    if branch_counts != expected:
        raise AssertionError(f"two-site SU2 branch counts differ: {branch_counts} != {expected}")
    return branch_counts


def spin_adapted_basis(nsites: int, nelec: int, j2: int, m2: int = 0):
    """Return primitive-basis columns for fixed (Ne, S, M)."""
    cols = []
    kept = []
    for mp in build_site_csf_multiplets(nsites):
        vec = mp.states.get(m2)
        if mp.nelec == nelec and mp.j2 == j2 and vec is not None:
            cols.append(vec)
            kept.append(mp)
    if not cols:
        return np.zeros((4**nsites, 0)), kept
    basis = np.column_stack(cols)
    gram = basis.conj().T @ basis
    err = np.max(np.abs(gram - np.eye(gram.shape[0])))
    if err > 1e-10:
        raise ValueError(f"CSF basis is not orthonormal; max Gram error {err:g}")
    return basis, kept


def su2_product_symmetry() -> ProductSymmetry:
    return ProductSymmetry((U1Symmetry("Ne"), SU2Symmetry("SU2")), name="U1xSU2")


def csf_irrep_site(nsites: int, m2: int | None = 0) -> tuple[Leg, dict[Irrep, np.ndarray]]:
    """Build an Leg and CSF basis columns for every sector with this M."""
    sectors: dict[Irrep, list[np.ndarray]] = {}
    for mp in build_site_csf_multiplets(nsites):
        selected_m2 = mp.j2 if m2 is None else m2
        vec = mp.states.get(selected_m2)
        if vec is None:
            continue
        irrep = Irrep((mp.nelec, mp.j2))
        sectors.setdefault(irrep, []).append(vec)

    dims = {irrep: len(cols) for irrep, cols in sectors.items()}
    bases = {irrep: np.column_stack(cols) for irrep, cols in sectors.items()}
    return Leg(dims, symmetry=su2_product_symmetry()), bases


def scalar_hamiltonian_irrep_tensor(H, site: Leg, bases: dict[Irrep, np.ndarray]) -> IrrepTensor:
    """Represent a scalar Hamiltonian as an IrrepTensor over CSF sectors."""
    H = asarray(H)
    blocks = {}
    for irrep, basis in bases.items():
        block = basis.conj().T @ H @ basis
        blocks[(irrep, irrep)] = 0.5 * (block + block.conj().T)
    return IrrepTensor(site, site, OpIrrep((0, 0)), blocks)


def spin_adapted_operator_matrix(op, nsites: int, bra, ket):
    """Project an operator between two spin-adapted CSF sectors.

    bra/ket are tuples ``(nelec, j2, m2)``.  This is the bridge used by
    the future SU(2)-NARG residual machinery: scalar, spinor, and coupled
    tensor operators can first be checked as ordinary CSF-sector maps before
    being converted to reduced matrix elements.
    """
    bra_basis, _ = spin_adapted_basis(nsites, *bra)
    ket_basis, _ = spin_adapted_basis(nsites, *ket)
    if bra_basis.shape[1] == 0 or ket_basis.shape[1] == 0:
        return np.zeros((bra_basis.shape[1], ket_basis.shape[1]), dtype=complex)
    return bra_basis.conj().T @ asarray(op) @ ket_basis


def spin_adapted_operator_tensor(op, nsites: int, op_irrep: OpIrrep, m2: int | None = 0) -> IrrepTensor:
    """Wrap a projected CSF operator as an IrrepTensor."""
    site, bases = csf_irrep_site(nsites, m2=m2)
    dense_op = asarray(op)
    blocks = {}
    for bra_irrep, bra_basis in bases.items():
        for ket_irrep, ket_basis in bases.items():
            if not site.symmetry.allows(bra_irrep.charge, op_irrep.charge, ket_irrep.charge):
                continue
            block = bra_basis.conj().T @ dense_op @ ket_basis
            if np.any(np.abs(block) > 1e-12):
                blocks[(bra_irrep, ket_irrep)] = block
    return IrrepTensor(site, site, op_irrep, blocks)


def validate_operator_bridge(h1e, eri, nelec: int):
    """Smoke-test spin-adapted operator projections for residual porting."""
    nsites = h1e.shape[0]
    model = full_jw_model(h1e, eri, nelec)
    # S+ maps triplet M=0 -> triplet M=1 with norm sqrt(2), and kills singlet.
    sp_triplet = spin_adapted_operator_matrix(model.Sp, nsites, (nelec, 2, 2), (nelec, 2, 0))
    sp_singlet = spin_adapted_operator_matrix(model.Sp, nsites, (nelec, 0, 0), (nelec, 0, 0))
    triplet_norm = np.linalg.norm(sp_triplet)
    singlet_norm = np.linalg.norm(sp_singlet)
    print(f"operator bridge ||S+ triplet M0->M1||={triplet_norm:.10f}")
    print(f"operator bridge ||S+ singlet M0->M0||={singlet_norm:.3e}")

    sz_tensor = spin_adapted_operator_tensor(model.Sz, nsites, OpIrrep((0, 0)), m2=0)
    irrep = Irrep((nelec, 0))
    print(f"IrrepTensor scalar Sz singlet block norm={np.linalg.norm(sz_tensor.block(irrep, irrep)):.3e}")


def atomic_chain(natom: int, z, element: str = "H", basis: str = "sto6g", spin: int = 0):
    elements = [element] * natom
    coords = np.zeros((natom, 3))
    coords[:, 2] = z
    return Molecule(atom=build_atom_from_coords(elements, coords), basis=basis, unit="b", spin=spin)


def qchem_integrals(natom: int, span: float = 4.0, basis: str = "sto6g"):
    z = np.linspace(-span, span, natom)
    mol = atomic_chain(natom, z, basis=basis)
    mol.build()
    mf = mol.RHF()
    mf.run()
    return mol, mf, mf.get_hcore_mo(), mf.get_eri_mo()


def full_jw_model(h1e, eri, nelec):
    model = SpinHalfFermionChain(h1e, eri, nelec=nelec)
    model.jordan_wigner(forward=False)
    return model


def asarray(op):
    return op.toarray() if issparse(op) else np.asarray(op)


def su2_projected_roots(h1e, eri, nelec: int, j2: int, nroots: int = 8):
    """Diagonalize the scalar Hamiltonian in an exact SU(2)-adapted CSF sector."""
    nsites = h1e.shape[0]
    model = full_jw_model(h1e, eri, nelec)
    H = asarray(model.H)
    basis, multiplets = spin_adapted_basis(nsites, nelec, j2, m2=0)
    Hs = basis.conj().T @ H @ basis
    Hs = 0.5 * (Hs + Hs.conj().T)
    evals, evecs = eigh(Hs)
    return evals[:nroots], evecs[:, :nroots], basis, multiplets


def su2_irrep_tensor_roots(h1e, eri, nelec: int, j2: int, nroots: int = 8, m2: int | None = None):
    """Same roots, but through an IrrepTensor scalar Hamiltonian block."""
    nsites = h1e.shape[0]
    model = full_jw_model(h1e, eri, nelec)
    site, bases = csf_irrep_site(nsites, m2=m2)
    Ht = scalar_hamiltonian_irrep_tensor(model.H, site, bases)
    irrep = Irrep((nelec, j2))
    block = Ht.block(irrep, irrep)
    evals, evecs = eigh(block)
    return evals[:nroots], evecs[:, :nroots], Ht


def primitive_qn_labels(nsites: int):
    local = np.array([[0, 0], [1, 1], [1, -1], [2, 0]], dtype=int)
    labels = np.zeros((4**nsites, 2), dtype=int)
    dims = (4,) * nsites
    for flat in range(labels.shape[0]):
        labels[flat] = np.sum(local[list(np.unravel_index(flat, dims))], axis=0)
    return labels


def exact_sz0_spin_labeled_roots(h1e, eri, nelec: int, nroots: int = 24):
    """Reference diagonalization in (Ne, Sz=0), labeled by <S^2>."""
    nsites = h1e.shape[0]
    model = full_jw_model(h1e, eri, nelec)
    H = asarray(model.H)
    S2 = asarray(model.S2)
    labels = primitive_qn_labels(nsites)
    idx = np.flatnonzero((labels[:, 0] == nelec) & (labels[:, 1] == 0))
    Hblk = H[np.ix_(idx, idx)]
    S2blk = S2[np.ix_(idx, idx)]
    evals, evecs = eigh(0.5 * (Hblk + Hblk.conj().T))
    evals = evals[:nroots]
    evecs = evecs[:, :nroots]
    s2 = np.real(np.einsum("ia,ij,ja->a", evecs.conj(), S2blk, evecs, optimize=True))
    spin = 0.5 * (np.sqrt(np.maximum(0.0, 1.0 + 4.0 * s2)) - 1.0)
    return evals, s2, spin


def validate(natom: int, span: float = 4.0, nroots: int = 6):
    mol, mf, h1e, eri = qchem_integrals(natom, span=span)
    nelec = int(np.asarray(mol.nelec).sum())
    enuc = mol.energy_nuc()
    print(f"H{natom} span={span:g} spacing={2*span/(natom-1):.10f} Bohr")
    print(f"RHF {mf.e_tot:.12f}  Enuc {enuc:.12f}  determinants Sz=0 reference")

    ref_e, ref_s2, ref_spin = exact_sz0_spin_labeled_roots(h1e, eri, nelec, nroots=4 * nroots)
    for j2, label in [(0, "singlet"), (2, "triplet")]:
        roots, _, basis, _ = su2_projected_roots(h1e, eri, nelec, j2=j2, nroots=nroots)
        tensor_roots, _, _ = su2_irrep_tensor_roots(h1e, eri, nelec, j2=j2, nroots=nroots)
        print(f"\nSU2 {label} sector dim={basis.shape[1]}")
        for i, e in enumerate(roots):
            print(f"  {i:2d} {e + enuc: .10f}")
        print(f"  max diff direct vs IrrepTensor block: {np.max(np.abs(roots - tensor_roots)):.3e}")

        target_s2 = (j2 / 2) * (j2 / 2 + 1)
        matched = ref_e[np.abs(ref_s2 - target_s2) < 1e-7][:nroots]
        if len(matched):
            diff = roots[: len(matched)] - matched
            print(f"  max electronic-energy diff vs Sz=0 labeled ref: {np.max(np.abs(diff)):.3e}")
        else:
            print("  no matching reference roots found in requested window")

    validate_operator_bridge(h1e, eri, nelec)


def main():
    site = local_su2_site()
    print("Local SU2 site:")
    for irrep, dim in site.dims.items():
        nelec, j2 = irrep.charge
        print(f"  Ne={nelec} S={spin_label(j2)} dim={dim}")

    print("\nOne-site block plus local SU2 branch update:")
    for branch, counts in validate_two_site_branch_update().items():
        pieces = [
            f"Ne={nelec} S={spin_label(j2)} dim={dim}"
            for (nelec, j2), dim in counts.items()
        ]
        print(f"  {branch}: " + "; ".join(pieces))

    print("\nTwo-site SU2 sectors:")
    for (nelec, j2), dim in validate_two_site_sector_counts().items():
        print(f"  Ne={nelec} S={spin_label(j2)} dim={dim}")

    for natom in (4, 6):
        print()
        validate(natom, span=4.0, nroots=6)


if __name__ == "__main__":
    main()
